"""ch15 · concurrency —— 作业队列 + worker + 并发上限 + per-run workspace。

ch14 的 `POST /v1/runs` 同步阻塞：一个长 run 占住一条连接，并发数 = 连接数。
本章把它改成**异步作业模型**：

```
POST /v1/runs ──> 202 + job_id ──> GET /v1/runs/{id} 轮询
                     │
                     ▼
              asyncio.Queue ──> N 个 worker（并发上限）
                                  └─ 每个 run 独立 workspace/<run_id>
```

关键约束：
- 并发只在**跨 run** 打开；单个 run 内部仍是顺序 ReAct（可恢复、可回放）。
- 队列有上限（背压）：满了返回 429，而不是无限吃内存。
- worker 是同一 event loop 上的 task；阻塞型工具/同步 SDK 必须 `asyncio.to_thread`
  或独立进程，不能直接塞进事件循环。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
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
# 1. 服务层：一次 run 的装配（与 ch14 相同，只是 workspace 可由调度器注入）
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


# --------------------------------------------------------------------------- #
# 2. 作业模型
# --------------------------------------------------------------------------- #
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
    """固定数量 worker + 有界队列。

    并发上限 = worker 数，而不是信号量：worker 自己就是消费者，天然限制同时执行的 run。
    队列只统计**等待中**的作业；正在跑的作业不占队列容量。
    """

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
            return existing  # 幂等：同 run_id 重试返回同一作业，不重复执行
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

    return app


app = create_app(mock=os.getenv("EH_HTTP_MOCK") == "1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("EH_HOST", "127.0.0.1"),
        port=int(os.getenv("EH_PORT", "8000")),
    )
