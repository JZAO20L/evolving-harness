"""ch16 · sse —— 把「轮询 JobView」升级成服务器推送的事件流。

ch15 的客户端只能反复 `GET /v1/runs/{id}`，既慢又浪费。本章加一条
`GET /v1/runs/{id}/events`，用 **SSE（Server-Sent Events）** 把 run 的进展
单向推给客户端：

```
POST /v1/runs ──> 202 + job_id
                      │
                      ▼
GET /v1/runs/{id}/events ──> text/event-stream
                      │
                      ├─ event: job.status    queued / running
                      ├─ event: run.start
                      ├─ event: model.request
                      ├─ event: model.response
                      ├─ event: tool.call / tool.result
                      ├─ event: run.end
                      └─ event: job.end       succeeded | failed（随后关流）
```

事件源就是 Agent 已经落盘的 **trace JSONL**：`TraceRecorder` 每写一行就 flush，
本章的 tailer 增量读取并转成 SSE。这样不必改核心循环，也天然支持**断线重连**：
客户端带上 `Last-Event-ID`，服务端从该序号之后补发。

关键约束：
- SSE 是**单向**的（服务端→客户端）；取消/审批要等 ch17 的 WebSocket。
- 每条事件带 `id: <seq>`，按 run 内严格递增；重连靠它去重/续传。
- 事件流必须有终止：`job.end` 之后关流，否则浏览器 EventSource 会自动重连，
  已完成的任务被反复重放。
- 生产代理常缓冲响应；需要 `Cache-Control: no-cache` + `X-Accel-Buffering: no`。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT = REPO_ROOT / "agents/default/agent.yaml"
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"
DEFAULT_TRACE_DIR = REPO_ROOT / "traces"

ModelFactory = Callable[[State], ChatModel]
TERMINAL = frozenset({"succeeded", "failed"})
SSE_POLL_SECONDS = 0.05
SSE_HEARTBEAT_SECONDS = 15.0


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
    status: str  # queued | running | succeeded | failed
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


# --------------------------------------------------------------------------- #
# 1. 服务层与作业管理（与 ch15 相同，只是 trace 路径要能被 SSE 定位）
# --------------------------------------------------------------------------- #
@dataclass
class RunService:
    agent_path: str | Path = DEFAULT_AGENT
    workspace: str | Path = DEFAULT_WORKSPACE
    trace_dir: str | Path = DEFAULT_TRACE_DIR
    mock: bool = False
    model_factory: ModelFactory | None = None

    async def run(
        self, task: str, run_id: str | None = None, workspace: str | Path | None = None
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
            agent = Agent(
                model=model,
                tools=build_tools(state, root),
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


class JobManager:
    def __init__(
        self,
        service: RunService,
        *,
        max_concurrency: int = 2,
        queue_size: int = 32,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.service = service
        self.max_concurrency = max(1, int(max_concurrency))
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=max(1, int(queue_size)))
        self.workspace_root = Path(workspace_root or (Path(service.workspace) / "runs"))
        self.jobs: dict[str, Job] = {}
        self._workers: list[asyncio.Task] = []
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return self.queue.qsize()

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
            await self.queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, task: str, run_id: str | None = None) -> Job:
        await self.start()
        rid = run_id or uuid.uuid4().hex[:12]
        existing = self.jobs.get(rid)
        if existing is not None:
            return existing
        workspace = self.workspace_root / rid
        job = Job(run_id=rid, task=task, workspace=workspace)
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFull(f"queue is full ({self.queue.maxsize} waiting)") from exc
        self.jobs[rid] = job
        self._idle.clear()
        return job

    def get(self, run_id: str) -> Job | None:
        return self.jobs.get(run_id)

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def _worker(self, index: int) -> None:
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            job.status = "running"
            job.started_at = time.time()
            self._active += 1
            try:
                job.result = await self.service.run(
                    job.task, job.run_id, workspace=job.workspace
                )
                job.status = "succeeded"
            except (AgentConfigError, MaxStepsExceeded, ModelError) as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - 一个作业失败不能杀死 worker
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = time.time()
                self._active -= 1
                self.queue.task_done()
                if self._active == 0 and self.queue.empty():
                    self._idle.set()


# --------------------------------------------------------------------------- #
# 2. SSE：trace tailer + 事件编码
# --------------------------------------------------------------------------- #
def trace_path_for(service: RunService, run_id: str) -> Path:
    return Path(service.trace_dir) / f"{run_id}.jsonl"


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


# --------------------------------------------------------------------------- #
# 3. 应用工厂
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
) -> FastAPI:
    service = service or RunService(
        agent_path=agent_path, workspace=workspace, trace_dir=trace_dir, mock=mock
    )
    manager = JobManager(
        service,
        max_concurrency=max_concurrency,
        queue_size=queue_size,
        workspace_root=Path(service.workspace) / "runs",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(title="evolving-harness", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.manager = manager

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            agent=str(service.agent_path),
            concurrency=manager.max_concurrency,
            active=manager.active,
            queued=manager.queued,
        )

    @app.post("/v1/runs", response_model=JobView, status_code=202)
    async def create_run(request: RunRequest, response: Response) -> JobView:
        try:
            job = await manager.submit(request.task, request.run_id)
        except QueueFull as exc:
            response.headers["Retry-After"] = "1"
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return job.view()

    @app.get("/v1/runs/{run_id}", response_model=JobView)
    async def get_run(run_id: str) -> JobView:
        job = manager.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return job.view()

    @app.get("/v1/runs/{run_id}/events")
    async def stream_run_events(
        run_id: str, request: Request, after: int = 0
    ) -> StreamingResponse:
        job = manager.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        last_event_id = request.headers.get("last-event-id", "")
        if last_event_id.isdigit():
            after = max(after, int(last_event_id))
        return StreamingResponse(
            event_stream(
                job,
                trace_path_for(service, run_id),
                request,
                after=max(0, after),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app(mock=os.getenv("EH_HTTP_MOCK") == "1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("EH_HOST", "127.0.0.1"),
        port=int(os.getenv("EH_PORT", "8000")),
    )
