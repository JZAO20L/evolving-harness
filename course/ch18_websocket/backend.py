"""ch18 · websocket —— 双向流：run 事件推送 + cancel / approve 实时控制。

ch17 的 SSE 只能服务端→客户端：用户想中途取消，或对一次危险工具调用点「批准/拒绝」，
都没有回程通道。本章加一条 WebSocket：

```
POST /v1/runs ──> 202 + job_id
                      │
                      ▼
WS /v1/runs/{id}/ws <──> 双向 JSON
   服务端 → 客户端：hello / job.status / trace event / approval.request
                    / pong / heartbeat / job.end
   客户端 → 服务端：{"type":"cancel"}
                    {"type":"approve","approval_id":"..","approved":true}
                    {"type":"ping"}
```

关键设计：
- **控制不进入循环**：取消 = 取消该 job 的 asyncio.Task；审批 = 在工具 handler 外包一层
  `await Future` 的门。Agent 核心仍是顺序 ReAct，不知道 WebSocket 的存在。
- **审批门放在工具边界**：只有 `write_file` / `bash` 这类有副作用的工具需要人确认；
  读工具直接执行。拒绝返回的是 `ERROR: ...`（错误当数据），循环照常继续。
- **单 writer**：所有下行帧走同一个 sender 协程 / control queue，避免 reader 与 sender
  并发写同一个 WebSocket。
- **心跳与终止**：15s 发应用层 heartbeat；`job.end` 后主动 close，客户端不会误以为掉线。
- **断连即拒绝**：客户端消失时，未决审批全部置为 denied，避免 run 永久挂起。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# --------------------------------------------------------------------------- #
# 1. 控制面：审批 Future + 下行消息队列（一个 job 一份）
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
# 2. 服务层与作业管理
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
        job = Job(
            run_id=rid,
            task=task,
            workspace=workspace,
            control=RunControl(approval_timeout=self.service.approval_timeout),
        )
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFull(f"queue is full ({self.queue.maxsize} waiting)") from exc
        self.jobs[rid] = job
        self._idle.clear()
        return job

    def get(self, run_id: str) -> Job | None:
        return self.jobs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        """取消排队中或运行中的 job；已完成/不存在返回 False。"""
        job = self.jobs.get(run_id)
        if job is None or job.status in TERMINAL:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.error = "cancelled before start"
            job.finished_at = time.time()
            return True
        job.cancel_requested = True
        if job.handle is not None and not job.handle.done():
            job.handle.cancel()
            return True
        return False

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def _worker(self, index: int) -> None:
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            if job.status == "cancelled":  # 排队期间被取消，直接跳过
                self.queue.task_done()
                if self._active == 0 and self.queue.empty():
                    self._idle.set()
                continue
            job.status = "running"
            job.started_at = time.time()
            self._active += 1
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
# 3. WebSocket：单 writer 推事件 + reader 接控制
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


def handle_ws_command(manager: JobManager, job: Job, raw: Any) -> None:
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
        accepted = manager.cancel(job.run_id)
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
# 4. 应用工厂
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

    @app.websocket("/v1/runs/{run_id}/ws")
    async def run_socket(websocket: WebSocket, run_id: str, after: int = 0) -> None:
        job = manager.get(run_id)
        await websocket.accept()
        if job is None:
            await websocket.send_json({"type": "error", "message": f"run not found: {run_id}"})
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
                handle_ws_command(manager, job, raw)
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
