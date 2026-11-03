"""ch20 · redis —— 用 Redis 语义承接热路径：缓存、队列、幂等锁与频控。

ch19 的 SQLite 是**真相源**：作业终态、trace 索引、版本与 checkpoint 都落库。
但每次提交/查询都打数据库既慢又浪费；多实例部署时进程内的 `asyncio.Queue`
和 `dict` 也无法共享。本章把「热路径」搬到 Redis：

```
eh:job:{run_id}          Hash：作业视图（task/status/result/error/时间戳），TTL 1 天
eh:queue                 List：待跑 run_id；LPUSH 入队 / RPOP 出队
eh:lock:{run_id}         String：提交幂等锁（SET NX EX），跨实例防重复建单
eh:rate:{scope}:{window} String：固定窗口计数，INCR + EXPIRE
```

分工：**Redis 热、SQL 冷**。Redis 丢了可以回数据库重建；数据库是权威记录。
本章默认用 `MemoryKV`（同一套命令语义、零依赖、L0 离线），设置 `EH_REDIS_URL`
后自动换成 `redis.asyncio` 客户端。

关键设计：
- **队列换实现，不换语义**：worker 从 `RPOP` 取 run_id，本实例没有 Job 时从 Hash
  水合；多实例共享同一个队列。
- **提交幂等**：`SET NX` 抢锁，Hash 已存在则直接返回旧作业；避免跨实例重复建单。
- **缓存回填**：`GET /v1/runs/{id}` 先内存、再 Redis Hash、最后 SQL，并把命中写回 Redis。
- **频控**：按客户端固定窗口计数，超限返回 429 + `Retry-After`；频控键与作业无关。
- **原子性边界**：单条 Redis 命令原子；多命令流程（建 Hash → 入队）不是事务，
  失败时靠幂等重试与 TTL 收敛，而不是分布式事务。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evolving_harness.agent import (
    Agent,
    AgentConfigError,
    MaxStepsExceeded,
    RunResult,
    State,
    build_tools,
    create_model,
)
from evolving_harness.models.base import ChatModel, ModelError
from evolving_harness.tools.base import Tool, ToolRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT = REPO_ROOT / "agents/default/agent.yaml"
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"
DEFAULT_TRACE_DIR = REPO_ROOT / "traces"

ModelFactory = Callable[[State], ChatModel]
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
SSE_POLL_SECONDS = 0.05
SSE_HEARTBEAT_SECONDS = 15.0
WS_POLL_SECONDS = 0.05
WS_HEARTBEAT_SECONDS = 15.0

REDIS_PREFIX = "eh"
JOB_TTL_SECONDS = 24 * 3600
LOCK_TTL_SECONDS = 30
QUEUE_POLL_SECONDS = 0.05
STOP = "__stop__"

# 有副作用、需要人工确认的工具。读工具不设门。
DEFAULT_APPROVAL_TOOLS = ("write_file", "bash")


# --------------------------------------------------------------------------- #
# 0. HTTP 契约
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=20_000)
    run_id: str | None = Field(default=None, max_length=64)

    @field_validator("task")
    @classmethod
    def _task_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task must not be blank")
        return value

    @field_validator("run_id")
    @classmethod
    def _run_id_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or not all(ch.isalnum() or ch in "-_" for ch in value):
            raise ValueError("run_id must contain only letters, digits, '-' or '_'")
        return value


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RunResponse(BaseModel):
    run_id: str
    final: str
    status: str
    steps: int
    usage: UsageOut
    state_version: int
    state_hash: str
    trace_path: str
    workspace: str

    @classmethod
    def from_result(cls, result: RunResult, workspace: str | Path) -> RunResponse:
        return cls(
            run_id=result.run_id,
            final=result.final,
            status=result.status,
            steps=result.steps,
            usage=UsageOut(**result.usage.to_dict()),
            state_version=result.state_version,
            state_hash=result.state_hash,
            trace_path=str(result.trace_path),
            workspace=str(workspace),
        )


class JobView(BaseModel):
    run_id: str
    task: str
    status: str  # queued | running | succeeded | failed | cancelled
    workspace: str
    result: RunResponse | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class HealthResponse(BaseModel):
    status: str
    agent: str
    concurrency: int
    active: int
    queued: int


class AgentVersionView(BaseModel):
    version: int
    agent_hash: str
    name: str
    path: str
    created_at: float


class CheckpointView(BaseModel):
    run_id: str
    state: dict[str, Any]
    saved_at: float


# --------------------------------------------------------------------------- #
# 1. 关系存储：runs / run_events / agent_versions / checkpoints
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_versions (
    version     INTEGER NOT NULL,
    agent_hash  TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    static_json TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    PRIMARY KEY (version, agent_hash)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    status        TEXT NOT NULL,
    agent_version INTEGER NOT NULL DEFAULT 0,
    agent_hash    TEXT    NOT NULL DEFAULT '',
    workspace     TEXT    NOT NULL,
    trace_path    TEXT    NOT NULL,
    result_json   TEXT,
    error         TEXT,
    created_at    REAL    NOT NULL,
    started_at    REAL,
    finished_at   REAL
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id     TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    ts         REAL    NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    data_json  TEXT    NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    run_id     TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    saved_at   REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind  ON run_events(run_id, kind);
"""


class RunStore:
    """SQLite 实现；每次操作新开连接 + to_thread，避免阻塞事件循环。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    async def register_agent(self, state: State) -> None:
        await asyncio.to_thread(self._register_agent_sync, state)

    def _register_agent_sync(self, state: State) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_versions
                    (version, agent_hash, name, path, static_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    state.version,
                    state.hash,
                    state.name,
                    str(state.path),
                    json.dumps(state.raw, ensure_ascii=False, sort_keys=True, default=str),
                    time.time(),
                ),
            )

    async def create_run(
        self, job: Job, *, version: int, agent_hash: str, trace_path: Path
    ) -> None:
        await asyncio.to_thread(
            self._create_run_sync, job, version, agent_hash, trace_path
        )

    def _create_run_sync(
        self, job: Job, version: int, agent_hash: str, trace_path: Path
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (run_id, task, status, agent_version, agent_hash,
                     workspace, trace_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (
                    job.run_id,
                    job.task,
                    job.status,
                    version,
                    agent_hash,
                    str(job.workspace),
                    str(trace_path),
                    job.created_at,
                ),
            )

    async def mark_running(self, run_id: str, started_at: float) -> None:
        await asyncio.to_thread(self._mark_running_sync, run_id, started_at)

    def _mark_running_sync(self, run_id: str, started_at: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = 'running', started_at = ? WHERE run_id = ?",
                (started_at, run_id),
            )

    async def finish_run(self, job: Job) -> None:
        await asyncio.to_thread(self._finish_run_sync, job)

    def _finish_run_sync(self, job: Job) -> None:
        result = job.result
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    status = ?,
                    result_json = ?,
                    error = ?,
                    finished_at = ?,
                    agent_version = COALESCE(?, agent_version),
                    agent_hash = COALESCE(?, agent_hash),
                    trace_path = COALESCE(?, trace_path)
                WHERE run_id = ?
                """,
                (
                    job.status,
                    result.model_dump_json() if result is not None else None,
                    job.error,
                    job.finished_at or time.time(),
                    result.state_version if result is not None else None,
                    result.state_hash if result is not None else None,
                    str(result.trace_path) if result is not None else None,
                    job.run_id,
                ),
            )

    async def index_trace(self, run_id: str, trace_path: Path) -> int:
        return await asyncio.to_thread(self._index_trace_sync, run_id, trace_path)

    def _index_trace_sync(self, run_id: str, trace_path: Path) -> int:
        if not trace_path.is_file():
            return 0
        rows: list[tuple[Any, ...]] = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                (
                    run_id,
                    int(record.get("seq", 0)),
                    float(record.get("ts", 0.0)),
                    int(record.get("elapsed_ms", 0)),
                    str(record.get("kind", "trace")),
                    json.dumps(record.get("data", {}), ensure_ascii=False, default=str),
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO run_events
                    (run_id, seq, ts, elapsed_ms, kind, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    async def save_checkpoint(self, run_id: str, checkpoint_path: Path) -> bool:
        return await asyncio.to_thread(
            self._save_checkpoint_sync, run_id, checkpoint_path
        )

    def _save_checkpoint_sync(self, run_id: str, checkpoint_path: Path) -> bool:
        if not checkpoint_path.is_file():
            return False
        state_json = checkpoint_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (run_id, state_json, saved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    saved_at = excluded.saved_at
                """,
                (run_id, state_json, time.time()),
            )
        return True

    async def get_run(self, run_id: str) -> JobView | None:
        return await asyncio.to_thread(self._get_run_sync, run_id)

    def _get_run_sync(self, run_id: str) -> JobView | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_view(row) if row is not None else None

    async def list_runs(self, *, limit: int = 20, offset: int = 0) -> list[JobView]:
        return await asyncio.to_thread(self._list_runs_sync, limit, offset)

    def _list_runs_sync(self, limit: int, offset: int) -> list[JobView]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_view(row) for row in rows]

    @staticmethod
    def _row_to_view(row: sqlite3.Row) -> JobView:
        result = (
            RunResponse.model_validate(json.loads(row["result_json"]))
            if row["result_json"]
            else None
        )
        return JobView(
            run_id=row["run_id"],
            task=row["task"],
            status=row["status"],
            workspace=row["workspace"],
            result=result,
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    async def events(self, run_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._events_sync, run_id, after)

    def _events_sync(self, run_id: str, after: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, ts, elapsed_ms, kind, data_json
                FROM run_events WHERE run_id = ? AND seq > ?
                ORDER BY seq ASC
                """,
                (run_id, after),
            ).fetchall()
        return [
            {
                "v": 1,
                "run_id": run_id,
                "seq": row["seq"],
                "ts": row["ts"],
                "elapsed_ms": row["elapsed_ms"],
                "kind": row["kind"],
                "data": json.loads(row["data_json"]),
            }
            for row in rows
        ]

    async def get_checkpoint(self, run_id: str) -> CheckpointView | None:
        return await asyncio.to_thread(self._get_checkpoint_sync, run_id)

    def _get_checkpoint_sync(self, run_id: str) -> CheckpointView | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json, saved_at FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return CheckpointView(
            run_id=run_id,
            state=json.loads(row["state_json"]),
            saved_at=row["saved_at"],
        )

    async def agent_versions(self, *, limit: int = 50) -> list[AgentVersionView]:
        return await asyncio.to_thread(self._agent_versions_sync, limit)

    def _agent_versions_sync(self, limit: int) -> list[AgentVersionView]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT version, agent_hash, name, path, created_at
                FROM agent_versions ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            AgentVersionView(
                version=row["version"],
                agent_hash=row["agent_hash"],
                name=row["name"],
                path=row["path"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


# --------------------------------------------------------------------------- #
# 2. 热路径：KV 抽象、内存实现、Redis 适配与高层 RunState
# --------------------------------------------------------------------------- #
class KV(Protocol):
    """本章用到的 Redis 命令子集；高低层语义只依赖这组原语。"""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, *, ex: int | None = None) -> None: ...
    async def set_nx(self, key: str, value: str, *, ex: int | None = None) -> bool: ...
    async def delete(self, key: str) -> None: ...
    async def incr(self, key: str) -> int: ...
    async def expire(self, key: str, seconds: int) -> None: ...
    async def hset(self, key: str, mapping: dict[str, str]) -> None: ...
    async def hgetall(self, key: str) -> dict[str, str]: ...
    async def lpush(self, key: str, value: str) -> None: ...
    async def rpop(self, key: str) -> str | None: ...
    async def llen(self, key: str) -> int: ...


class MemoryKV:
    """离线 L0 用的 Redis 语义子集：String / Hash / List + 惰性 TTL。

    只在单进程内有效，不是 Redis 的替代品；它的价值是让高层逻辑与测试
    完全共享，真 Redis 只替换这一层。
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        # key -> [kind, payload, expires_at]
        self._data: dict[str, list[Any]] = {}

    def _live(self, key: str) -> list[Any] | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at = item[2]
        if expires_at is not None and expires_at <= self.clock():
            del self._data[key]
            return None
        return item

    def _typed(self, key: str, kind: str) -> list[Any] | None:
        item = self._live(key)
        if item is None:
            return None
        if item[0] != kind:
            raise TypeError(f"key {key!r} holds {item[0]}, not {kind}")
        return item

    async def get(self, key: str) -> str | None:
        item = self._typed(key, "str")
        return None if item is None else str(item[1])

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        expires_at = self.clock() + ex if ex else None
        self._data[key] = ["str", str(value), expires_at]

    async def set_nx(self, key: str, value: str, *, ex: int | None = None) -> bool:
        if self._live(key) is not None:
            return False
        await self.set(key, value, ex=ex)
        return True

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def incr(self, key: str) -> int:
        item = self._typed(key, "str")
        current = int(item[1]) if item is not None else 0
        current += 1
        expires_at = item[2] if item is not None else None
        self._data[key] = ["str", str(current), expires_at]
        return current

    async def expire(self, key: str, seconds: int) -> None:
        item = self._live(key)
        if item is not None:
            item[2] = self.clock() + seconds

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        item = self._typed(key, "hash")
        if item is None:
            item = ["hash", {}, None]
            self._data[key] = item
        item[1].update({str(k): str(v) for k, v in mapping.items()})

    async def hgetall(self, key: str) -> dict[str, str]:
        item = self._typed(key, "hash")
        return {} if item is None else dict(item[1])

    async def lpush(self, key: str, value: str) -> None:
        item = self._typed(key, "list")
        if item is None:
            item = ["list", [], None]
            self._data[key] = item
        item[1].insert(0, str(value))

    async def rpop(self, key: str) -> str | None:
        item = self._typed(key, "list")
        if not item or not item[1]:
            return None
        return item[1].pop()

    async def llen(self, key: str) -> int:
        item = self._typed(key, "list")
        return 0 if item is None else len(item[1])

    def flush(self) -> None:
        self._data.clear()


class RedisKV:
    """`redis.asyncio` 适配层；只在配置了 EH_REDIS_URL 时实例化。"""

    def __init__(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_url(cls, url: str) -> RedisKV:
        try:
            import redis.asyncio as redis
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确指引
            raise RuntimeError(
                "EH_REDIS_URL is set but the 'redis' package is missing; "
                "install it with `uv pip install redis`"
            ) from exc
        return cls(redis.from_url(url, decode_responses=True))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        await self.client.set(key, value, ex=ex)

    async def set_nx(self, key: str, value: str, *, ex: int | None = None) -> bool:
        return bool(await self.client.set(key, value, ex=ex, nx=True))

    async def delete(self, key: str) -> None:
        await self.client.delete(key)

    async def incr(self, key: str) -> int:
        return int(await self.client.incr(key))

    async def expire(self, key: str, seconds: int) -> None:
        await self.client.expire(key, seconds)

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        await self.client.hset(key, mapping=mapping)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(await self.client.hgetall(key))

    async def lpush(self, key: str, value: str) -> None:
        await self.client.lpush(key, value)

    async def rpop(self, key: str) -> str | None:
        return await self.client.rpop(key)

    async def llen(self, key: str) -> int:
        return int(await self.client.llen(key))


def _job_mapping(view: JobView) -> dict[str, str]:
    return {
        "run_id": view.run_id,
        "task": view.task,
        "status": view.status,
        "workspace": view.workspace,
        "result_json": view.result.model_dump_json() if view.result is not None else "",
        "error": view.error or "",
        "created_at": repr(view.created_at),
        "started_at": "" if view.started_at is None else repr(view.started_at),
        "finished_at": "" if view.finished_at is None else repr(view.finished_at),
    }


class RunState:
    """Redis 热路径的高层封装：作业 Hash、队列 List、幂等锁、固定窗口频控。"""

    def __init__(
        self,
        kv: KV,
        *,
        prefix: str = REDIS_PREFIX,
        job_ttl: int = JOB_TTL_SECONDS,
        lock_ttl: int = LOCK_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.kv = kv
        self.prefix = prefix
        self.job_ttl = job_ttl
        self.lock_ttl = lock_ttl
        self.clock = clock

    def _job_key(self, run_id: str) -> str:
        return f"{self.prefix}:job:{run_id}"

    def _lock_key(self, run_id: str) -> str:
        return f"{self.prefix}:lock:{run_id}"

    @property
    def queue_key(self) -> str:
        return f"{self.prefix}:queue"

    async def put_job(self, job: Job) -> None:
        await self.put_view(job.view())

    async def put_view(self, view: JobView) -> None:
        await self.kv.hset(self._job_key(view.run_id), _job_mapping(view))
        await self.kv.expire(self._job_key(view.run_id), self.job_ttl)

    async def get_view(self, run_id: str) -> JobView | None:
        data = await self.kv.hgetall(self._job_key(run_id))
        if not data:
            return None
        result = (
            RunResponse.model_validate_json(data["result_json"])
            if data.get("result_json")
            else None
        )
        return JobView(
            run_id=data["run_id"],
            task=data["task"],
            status=data["status"],
            workspace=data["workspace"],
            result=result,
            error=data.get("error") or None,
            created_at=float(data["created_at"]),
            started_at=float(data["started_at"]) if data.get("started_at") else None,
            finished_at=float(data["finished_at"]) if data.get("finished_at") else None,
        )

    async def get_job(self, run_id: str) -> Job | None:
        view = await self.get_view(run_id)
        if view is None:
            return None
        return Job(
            run_id=view.run_id,
            task=view.task,
            workspace=Path(view.workspace),
            status=view.status,
            result=view.result,
            error=view.error,
            created_at=view.created_at,
            started_at=view.started_at,
            finished_at=view.finished_at,
        )

    async def claim(self, run_id: str) -> bool:
        return await self.kv.set_nx(self._lock_key(run_id), "1", ex=self.lock_ttl)

    async def release(self, run_id: str) -> None:
        await self.kv.delete(self._lock_key(run_id))

    async def enqueue(self, run_id: str) -> None:
        await self.kv.lpush(self.queue_key, run_id)

    async def dequeue(self) -> str | None:
        return await self.kv.rpop(self.queue_key)

    async def queue_len(self) -> int:
        return await self.kv.llen(self.queue_key)

    async def rate_limit(
        self, scope: str, *, limit: int, window: float
    ) -> tuple[bool, int, int]:
        """固定窗口：返回 (allowed, remaining, retry_after_seconds)。"""
        now = self.clock()
        bucket = int(now // window)
        key = f"{self.prefix}:rate:{scope}:{bucket}"
        count = await self.kv.incr(key)
        if count == 1:
            await self.kv.expire(key, int(window) + 1)
        retry_after = max(1, int(window - (now % window)))
        return count <= limit, max(0, limit - count), retry_after


def build_state(
    *, redis_url: str | None = None, kv: KV | None = None
) -> RunState:
    if kv is not None:
        return RunState(kv)
    if redis_url:
        return RunState(RedisKV.from_url(redis_url))
    return RunState(MemoryKV())


# --------------------------------------------------------------------------- #
# 3. 控制面：审批 Future + 下行消息队列（一个 job 一份）
# --------------------------------------------------------------------------- #
class RunControl:
    """run 与 WebSocket 之间的控制面；不依赖 Agent 内部结构。

    - `queue`：所有要发给客户端的控制类消息（approval.request / pong / ack），
      由 WS 的单一 sender 消费。
    - `_approvals`：approval_id → Future[bool]；工具 handler 在边界 await 它。
    """

    def __init__(self, *, approval_timeout: float = 60.0) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.approval_timeout = approval_timeout
        self.closed = False
        self._approvals: dict[str, asyncio.Future[bool]] = {}

    def request_approval(
        self, tool: str, arguments: dict[str, Any]
    ) -> tuple[str, asyncio.Future[bool]]:
        approval_id = uuid.uuid4().hex[:8]
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._approvals[approval_id] = future
        self.queue.put_nowait(
            {
                "type": "approval.request",
                "approval_id": approval_id,
                "tool": tool,
                "arguments": arguments,
            }
        )
        return approval_id, future

    def resolve(self, approval_id: str, approved: bool) -> bool:
        future = self._approvals.pop(approval_id, None)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def close(self) -> None:
        """客户端断开：所有未决审批按拒绝处理，run 不会永远挂住。"""
        self.closed = True
        for future in self._approvals.values():
            if not future.done():
                future.set_result(False)
        self._approvals.clear()


def _install_approval_gates(
    tools: ToolRegistry, control: RunControl, names: tuple[str, ...]
) -> None:
    """在工具边界包一层人工确认；核心循环只看到普通 Tool。"""

    async def _invoke(handler: Callable[..., Any], arguments: dict[str, Any]) -> str:
        result = handler(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else str(result)

    for name in names:
        item: Tool | None = tools.get(name)
        if item is None:
            continue
        original = item.handler

        async def gated(
            _name: str = name,
            _original: Callable[..., Any] = original,
            **arguments: Any,
        ) -> str:
            if control.closed:
                return "ERROR: approval unavailable; control channel closed"
            approval_id, future = control.request_approval(_name, arguments)
            try:
                approved = await asyncio.wait_for(future, timeout=control.approval_timeout)
            except TimeoutError:
                control._approvals.pop(approval_id, None)
                return "ERROR: approval timeout"
            if not approved:
                return "ERROR: approval denied by user"
            return await _invoke(_original, arguments)

        item.handler = gated


# --------------------------------------------------------------------------- #
# 4. 服务层与作业管理
# --------------------------------------------------------------------------- #
@dataclass
class RunService:
    agent_path: str | Path = DEFAULT_AGENT
    workspace: str | Path = DEFAULT_WORKSPACE
    trace_dir: str | Path = DEFAULT_TRACE_DIR
    mock: bool = False
    model_factory: ModelFactory | None = None
    approval_tools: tuple[str, ...] = DEFAULT_APPROVAL_TOOLS
    approval_timeout: float = 60.0

    async def run(
        self,
        task: str,
        run_id: str | None = None,
        workspace: str | Path | None = None,
        control: RunControl | None = None,
    ) -> RunResponse:
        state = State.load(self.agent_path)
        model = (
            self.model_factory(state)
            if self.model_factory is not None
            else create_model(state, mock=self.mock)
        )
        root = Path(workspace or self.workspace)
        trace_dir = Path(self.trace_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)
            tools = build_tools(state, root)
            if control is not None and self.approval_tools:
                _install_approval_gates(tools, control, self.approval_tools)
            agent = Agent(
                model=model,
                tools=tools,
                state=state,
                trace_dir=trace_dir,
                workspace=root,
            )
            result = await agent.run(task, run_id=run_id)
        finally:
            await model.aclose()
        return RunResponse.from_result(result, root)


class QueueFull(RuntimeError):
    pass


@dataclass
class Job:
    run_id: str
    task: str
    workspace: Path
    status: str = "queued"
    result: RunResponse | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    control: RunControl = field(default_factory=RunControl)
    handle: asyncio.Task | None = None
    cancel_requested: bool = False

    def view(self) -> JobView:
        return JobView(
            run_id=self.run_id,
            task=self.task,
            status=self.status,
            workspace=str(self.workspace),
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class DuplicateRun(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        service: RunService,
        *,
        max_concurrency: int = 2,
        queue_size: int = 32,
        workspace_root: str | Path | None = None,
        store: RunStore | None = None,
        state: RunState | None = None,
    ) -> None:
        self.service = service
        self.store = store
        self.state = state or build_state()
        self.max_concurrency = max(1, int(max_concurrency))
        self.queue_size = max(1, int(queue_size))
        self.workspace_root = Path(workspace_root or (Path(service.workspace) / "runs"))
        self.jobs: dict[str, Job] = {}
        self._workers: list[asyncio.Task] = []
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return self._active

    async def queued_count(self) -> int:
        return await self.state.queue_len()

    async def start(self) -> None:
        if self._workers:
            return
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"eh-worker-{i}")
            for i in range(self.max_concurrency)
        ]

    async def stop(self) -> None:
        for _ in self._workers:
            await self.state.enqueue(STOP)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, task: str, run_id: str | None = None) -> Job:
        await self.start()
        rid = run_id or uuid.uuid4().hex[:12]
        existing = self.jobs.get(rid)
        if existing is not None:
            return existing
        cached = await self.state.get_job(rid)
        if cached is not None:
            return cached
        if not await self.state.claim(rid):
            # 另一个实例正在建单：等它的 Hash 出现，再返回同一作业。
            for _ in range(20):
                cached = await self.state.get_job(rid)
                if cached is not None:
                    return cached
                await asyncio.sleep(0.01)
            raise DuplicateRun(f"run is being created elsewhere: {rid}")
        if await self.state.queue_len() >= self.queue_size:
            await self.state.release(rid)
            raise QueueFull(f"queue is full ({self.queue_size} waiting)")

        workspace = self.workspace_root / rid
        job = Job(
            run_id=rid,
            task=task,
            workspace=workspace,
            control=RunControl(approval_timeout=self.service.approval_timeout),
        )
        try:
            # 先落库、再写热状态、最后入队：任何一步失败都释放幂等锁，
            # 客户端重试会走同一条路径（SQL 的 ON CONFLICT 保证幂等）。
            if self.store is not None:
                version, agent_hash = 0, ""
                try:
                    agent_state = State.load(self.service.agent_path)
                except AgentConfigError:  # 坏 YAML：交给 worker 失败，但作业仍登记
                    agent_state = None
                if agent_state is not None:
                    version, agent_hash = agent_state.version, agent_state.hash
                    await self.store.register_agent(agent_state)
                await self.store.create_run(
                    job,
                    version=version,
                    agent_hash=agent_hash,
                    trace_path=trace_path_for(self.service, rid),
                )
            await self.state.put_job(job)
            self.jobs[rid] = job
            self._idle.clear()
            await self.state.enqueue(rid)
        except Exception:
            await self.state.release(rid)
            raise
        return job

    def get(self, run_id: str) -> Job | None:
        return self.jobs.get(run_id)

    async def cancel(self, run_id: str) -> bool:
        """取消排队中或本实例运行中的 job；已完成/不存在返回 False。"""
        job = self.jobs.get(run_id)
        if job is None:
            cached = await self.state.get_job(run_id)
            if cached is None or cached.status in TERMINAL:
                return False
            if cached.status != "queued":
                # 运行在别的实例：本进程没有 Task 句柄，停止需要广播（未实现）。
                return False
            cached.status = "cancelled"
            cached.error = "cancelled before start"
            cached.finished_at = time.time()
            await self._persist_finish(cached)
            return True
        if job.status in TERMINAL:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.error = "cancelled before start"
            job.finished_at = time.time()
            await self._persist_finish(job)
            return True
        job.cancel_requested = True
        if job.handle is not None and not job.handle.done():
            job.handle.cancel()
            return True
        return False

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def _persist_finish(self, job: Job) -> None:
        """终态：SQL 权威记录 + Redis 热视图；任一失败都不改业务结果。"""
        if self.store is not None:
            try:
                await self.store.finish_run(job)
                await self.store.index_trace(
                    job.run_id, trace_path_for(self.service, job.run_id)
                )
                await self.store.save_checkpoint(
                    job.run_id, checkpoint_path_for(self.service, job.run_id)
                )
            except Exception as exc:  # noqa: BLE001
                note = f"persist: {type(exc).__name__}: {exc}"
                job.error = f"{job.error}; {note}" if job.error else note
        try:
            await self.state.put_job(job)
        except Exception as exc:  # noqa: BLE001
            note = f"cache: {type(exc).__name__}: {exc}"
            job.error = f"{job.error}; {note}" if job.error else note

    async def _next_run(self) -> str:
        while True:
            run_id = await self.state.dequeue()
            if run_id is not None:
                return run_id
            await asyncio.sleep(QUEUE_POLL_SECONDS)

    async def _worker(self, index: int) -> None:
        while True:
            rid = await self._next_run()
            if rid == STOP:
                return
            job = self.jobs.get(rid)
            if job is None:
                job = await self.state.get_job(rid)
                if job is None:  # Hash 已过期，作业无处可跑
                    continue
                job.control = RunControl(approval_timeout=self.service.approval_timeout)
                self.jobs[rid] = job
            if job.status in TERMINAL or job.status == "cancelled":
                if self._active == 0 and await self.state.queue_len() == 0:
                    self._idle.set()
                continue
            job.status = "running"
            job.started_at = time.time()
            self._active += 1
            await self.state.put_job(job)
            if self.store is not None:
                await self.store.mark_running(job.run_id, job.started_at)
            job.handle = asyncio.create_task(
                self.service.run(
                    job.task,
                    job.run_id,
                    workspace=job.workspace,
                    control=job.control,
                ),
                name=f"eh-job-{job.run_id}",
            )
            try:
                job.result = await job.handle
                job.status = "succeeded"
            except asyncio.CancelledError:
                if not job.cancel_requested:
                    raise  # 外层关停：继续向上传播
                job.status = "cancelled"
                job.error = "CancelledError: cancelled by user"
            except (AgentConfigError, MaxStepsExceeded, ModelError) as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - 一个作业失败不能杀死 worker
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.handle = None
                job.control.close()
                job.finished_at = time.time()
                await self._persist_finish(job)
                self._active -= 1
                if self._active == 0 and await self.state.queue_len() == 0:
                    self._idle.set()


# --------------------------------------------------------------------------- #
# 5. SSE：trace tailer + 事件编码
# --------------------------------------------------------------------------- #
def trace_path_for(service: RunService, run_id: str) -> Path:
    return Path(service.trace_dir) / f"{run_id}.jsonl"


def checkpoint_path_for(service: RunService, run_id: str) -> Path:
    return Path(service.trace_dir) / f"{run_id}.checkpoint.json"


class TraceTail:
    """增量读取 Append-only JSONL；文件尚未创建时返回空。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: Any = None

    def read_new(self) -> list[dict[str, Any]]:
        if self._fh is None:
            if not self.path.is_file():
                return []
            self._fh = self.path.open(encoding="utf-8")
        events: list[dict[str, Any]] = []
        while True:
            line = self._fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def sse_frame(
    event: str,
    data: Any,
    *,
    event_id: int | None = None,
    retry: int | None = None,
) -> str:
    lines: list[str] = []
    if retry is not None:
        lines.append(f"retry: {retry}")
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    lines.extend(f"data: {part}" for part in payload.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


async def event_stream(
    job: Job,
    trace_path: Path,
    request: Request,
    *,
    after: int = 0,
) -> AsyncIterator[str]:
    """从 trace 第 after 条之后开始推送，直到 job 进入终态并发出 job.end。"""
    tail = TraceTail(trace_path)
    sent_status = ""
    last_beat = asyncio.get_running_loop().time()
    try:
        yield sse_frame("open", {"run_id": job.run_id, "after": after}, retry=3000)
        while True:
            if await request.is_disconnected():
                return

            for record in tail.read_new():
                seq = int(record.get("seq", 0))
                if seq <= after:
                    continue
                after = seq
                yield sse_frame(
                    str(record.get("kind", "trace")),
                    record,
                    event_id=seq,
                )

            if job.status != sent_status and job.status not in TERMINAL:
                # 终态由 job.end 表达，避免同一状态发两遍。
                sent_status = job.status
                yield sse_frame("job.status", {"run_id": job.run_id, "status": job.status})

            if job.status in TERMINAL:
                # worker 在 run() 返回后才写终态；此时 trace 已 flush 完，补读一次。
                for record in tail.read_new():
                    seq = int(record.get("seq", 0))
                    if seq <= after:
                        continue
                    after = seq
                    yield sse_frame(
                        str(record.get("kind", "trace")),
                        record,
                        event_id=seq,
                    )
                yield sse_frame("job.end", job.view().model_dump())
                return

            now = asyncio.get_running_loop().time()
            if now - last_beat >= SSE_HEARTBEAT_SECONDS:
                last_beat = now
                yield ": keep-alive\n\n"
            await asyncio.sleep(SSE_POLL_SECONDS)
    finally:
        tail.close()


async def stored_event_stream(
    job: JobView,
    store: RunStore,
    request: Request,
    *,
    after: int = 0,
) -> AsyncIterator[str]:
    """重启后从 run_events 回放；数据源是数据库而不是进程内 Job。"""
    yield sse_frame(
        "open",
        {"run_id": job.run_id, "after": after, "source": "db"},
        retry=3000,
    )
    for record in await store.events(job.run_id, after=after):
        if await request.is_disconnected():
            return
        seq = int(record.get("seq", 0))
        yield sse_frame(str(record.get("kind", "trace")), record, event_id=seq)
    yield sse_frame("job.end", job.model_dump())


# --------------------------------------------------------------------------- #
# 6. WebSocket：单 writer 推事件 + reader 接控制
# --------------------------------------------------------------------------- #
async def _drain_control(websocket: WebSocket, control: RunControl) -> None:
    while not control.queue.empty():
        await websocket.send_json(control.queue.get_nowait())


async def ws_event_stream(
    websocket: WebSocket,
    job: Job,
    trace_path: Path,
    control: RunControl,
    *,
    after: int = 0,
) -> None:
    """唯一的下行 writer：trace 事件 + 控制消息 + 心跳 + job.end。"""
    tail = TraceTail(trace_path)
    sent_status = ""
    last_beat = asyncio.get_running_loop().time()
    try:
        await websocket.send_json({"type": "hello", "run_id": job.run_id, "after": after})
        while True:
            await _drain_control(websocket, control)

            for record in tail.read_new():
                seq = int(record.get("seq", 0))
                if seq <= after:
                    continue
                after = seq
                await websocket.send_json(
                    {
                        "type": "event",
                        "event": record.get("kind", "trace"),
                        "seq": seq,
                        "record": record,
                    }
                )

            if job.status != sent_status and job.status not in TERMINAL:
                sent_status = job.status
                await websocket.send_json({"type": "job.status", "status": job.status})

            if job.status in TERMINAL:
                for record in tail.read_new():
                    seq = int(record.get("seq", 0))
                    if seq <= after:
                        continue
                    after = seq
                    await websocket.send_json(
                        {
                            "type": "event",
                            "event": record.get("kind", "trace"),
                            "seq": seq,
                            "record": record,
                        }
                    )
                await _drain_control(websocket, control)
                await websocket.send_json({"type": "job.end", "job": job.view().model_dump()})
                await websocket.close()
                return

            now = asyncio.get_running_loop().time()
            if now - last_beat >= WS_HEARTBEAT_SECONDS:
                last_beat = now
                await websocket.send_json({"type": "heartbeat", "ts": time.time()})
            await asyncio.sleep(WS_POLL_SECONDS)
    finally:
        tail.close()


async def handle_ws_command(manager: JobManager, job: Job, raw: Any) -> None:
    """把客户端命令翻译成控制面动作；不直接写 WebSocket（单 writer 约束）。"""
    control = job.control
    if not isinstance(raw, dict):
        control.queue.put_nowait({"type": "error", "message": "command must be an object"})
        return

    kind = raw.get("type")
    if kind == "ping":
        control.queue.put_nowait({"type": "pong", "ts": time.time()})
        return

    if kind == "cancel":
        accepted = await manager.cancel(job.run_id)
        control.queue.put_nowait({"type": "cancel.ack", "accepted": accepted})
        return

    if kind == "approve":
        approval_id = raw.get("approval_id")
        approved = raw.get("approved")
        if not isinstance(approval_id, str) or not approval_id or not isinstance(approved, bool):
            control.queue.put_nowait(
                {"type": "error", "message": "approve needs approval_id and bool approved"}
            )
            return
        accepted = control.resolve(approval_id, approved)
        control.queue.put_nowait(
            {
                "type": "approval.ack",
                "approval_id": approval_id,
                "approved": approved,
                "accepted": accepted,
            }
        )
        return

    control.queue.put_nowait({"type": "error", "message": f"unknown command: {kind!r}"})


# --------------------------------------------------------------------------- #
# 7. 应用工厂
# --------------------------------------------------------------------------- #
def create_app(
    service: RunService | None = None,
    *,
    agent_path: str | Path = DEFAULT_AGENT,
    workspace: str | Path = DEFAULT_WORKSPACE,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    mock: bool = False,
    max_concurrency: int = 2,
    queue_size: int = 32,
    store: RunStore | None = None,
    db_path: str | Path | None = None,
    hot_state: RunState | None = None,
    redis_url: str | None = None,
    rate_limit: int = 60,
    rate_window: float = 60.0,
) -> FastAPI:
    service = service or RunService(
        agent_path=agent_path, workspace=workspace, trace_dir=trace_dir, mock=mock
    )
    store = store or RunStore(db_path or (Path(service.workspace) / "harness.db"))
    hot_state = hot_state or build_state(redis_url=redis_url or os.getenv("EH_REDIS_URL"))
    manager = JobManager(
        service,
        max_concurrency=max_concurrency,
        queue_size=queue_size,
        workspace_root=Path(service.workspace) / "runs",
        store=store,
        state=hot_state,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await store.init()
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()
            close = getattr(hot_state.kv, "aclose", None)
            if close is not None:
                await close()

    app = FastAPI(title="evolving-harness", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.manager = manager
    app.state.store = store
    app.state.hot = hot_state

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            agent=str(service.agent_path),
            concurrency=manager.max_concurrency,
            active=manager.active,
            queued=await manager.queued_count(),
        )

    @app.post("/v1/runs", response_model=JobView, status_code=202)
    async def create_run(
        payload: RunRequest, response: Response, request: Request
    ) -> JobView:
        if rate_limit > 0:
            scope = request.client.host if request.client else "anonymous"
            try:
                allowed, remaining, retry_after = await hot_state.rate_limit(
                    scope, limit=rate_limit, window=rate_window
                )
            except Exception:  # noqa: BLE001 - 缓存故障不应阻断提交（fail-open）
                allowed, remaining, retry_after = True, rate_limit, 0
            response.headers["X-RateLimit-Limit"] = str(rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="rate limit exceeded",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(rate_limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )
        try:
            job = await manager.submit(payload.task, payload.run_id)
        except DuplicateRun as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(
                status_code=429, detail=str(exc), headers={"Retry-After": "1"}
            ) from exc
        return job.view()

    @app.get("/v1/runs", response_model=list[JobView])
    async def list_runs(limit: int = 20, offset: int = 0) -> list[JobView]:
        limit = min(max(1, limit), 100)
        return await store.list_runs(limit=limit, offset=max(0, offset))

    @app.get("/v1/runs/{run_id}", response_model=JobView)
    async def get_run(run_id: str) -> JobView:
        job = manager.get(run_id)
        if job is not None:
            return job.view()
        try:
            cached = await hot_state.get_view(run_id)
        except Exception:  # noqa: BLE001 - 缓存故障回退数据库
            cached = None
        if cached is not None:
            return cached
        stored = await store.get_run(run_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        try:  # 缓存回填；失败不影响响应
            await hot_state.put_view(stored)
        except Exception:  # noqa: BLE001, S110 - 缓存回填失败不影响响应
            pass
        return stored

    @app.get("/v1/runs/{run_id}/events")
    async def stream_run_events(
        run_id: str, request: Request, after: int = 0
    ) -> StreamingResponse:
        last_event_id = request.headers.get("last-event-id", "")
        if last_event_id.isdigit():
            after = max(after, int(last_event_id))
        after = max(0, after)

        job = manager.get(run_id)
        if job is not None:
            body = event_stream(
                job, trace_path_for(service, run_id), request, after=after
            )
        else:
            stored = await store.get_run(run_id)
            if stored is None:
                raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
            if stored.status not in TERMINAL:
                # 运行在别的实例：本实例没有 TraceTail，实时流需要广播（未实现）。
                raise HTTPException(
                    status_code=409,
                    detail=f"run is hosted on another instance: {run_id}; poll GET",
                )
            body = stored_event_stream(stored, store, request, after=after)

        return StreamingResponse(
            body,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/v1/runs/{run_id}/checkpoint", response_model=CheckpointView)
    async def get_checkpoint(run_id: str) -> CheckpointView:
        checkpoint = await store.get_checkpoint(run_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404, detail=f"checkpoint not found: {run_id}"
            )
        return checkpoint

    @app.get("/v1/agent/versions", response_model=list[AgentVersionView])
    async def agent_versions(limit: int = 50) -> list[AgentVersionView]:
        return await store.agent_versions(limit=min(max(1, limit), 200))

    @app.websocket("/v1/runs/{run_id}/ws")
    async def run_socket(websocket: WebSocket, run_id: str, after: int = 0) -> None:
        job = manager.get(run_id)
        await websocket.accept()
        if job is None:
            try:
                remote = await hot_state.get_view(run_id) is not None
            except Exception:  # noqa: BLE001
                remote = False
            message = (
                f"run is hosted on another instance: {run_id}"
                if remote
                else f"run not found: {run_id}"
            )
            await websocket.send_json({"type": "error", "message": message})
            await websocket.close(code=4404)
            return

        sender = asyncio.create_task(
            ws_event_stream(
                websocket,
                job,
                trace_path_for(service, run_id),
                job.control,
                after=max(0, after),
            ),
            name=f"eh-ws-{run_id}",
        )
        try:
            while not sender.done():
                try:
                    raw = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                except TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    job.control.queue.put_nowait(
                        {"type": "error", "message": "invalid JSON command"}
                    )
                    continue
                await handle_ws_command(manager, job, raw)
        finally:
            # 客户端断开或 job 已结束：拒绝所有未决审批，再收掉 sender。
            job.control.close()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    return app


app = create_app(mock=os.getenv("EH_HTTP_MOCK") == "1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("EH_HOST", "127.0.0.1"),
        port=int(os.getenv("EH_PORT", "8000")),
    )
