"""ch14 · fastapi —— 把一个 Agent run 包成 HTTP API。

本文件是本章 hero。它不重新实现 Agent：`agent.py`（生产核）仍是冻结的循环，
`backend.py` 只负责 **路由 → 校验 → 调 Agent → 错误码 → 响应模型**。

设计边界：
- 请求体只带 task / run_id；模型 key 走服务端环境变量，永不出现在 HTTP 里。
- 服务层 `RunService` 可注入模型工厂，L0 测试用脚本化 Mock，不联网。
- 错误码语义：配置/参数错误 400，步数耗尽 409，模型上游失败 502。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
# 0. HTTP 契约（pydantic）
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    """一次 run 的公开输入。私有 rubric / 模型 key 都不在这里。"""

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

    @classmethod
    def from_result(cls, result: RunResult) -> RunResponse:
        return cls(
            run_id=result.run_id,
            final=result.final,
            status=result.status,
            steps=result.steps,
            usage=UsageOut(**result.usage.to_dict()),
            state_version=result.state_version,
            state_hash=result.state_hash,
            trace_path=str(result.trace_path),
        )


class HealthResponse(BaseModel):
    status: str
    agent: str


# --------------------------------------------------------------------------- #
# 1. 服务层：HTTP 与 Agent 之间的薄编排
# --------------------------------------------------------------------------- #
@dataclass
class RunService:
    """一次 HTTP 请求 = 加载 YAML → 造 model/tools/Agent → await run。

    生产里 model 应由进程级连接池持有；本章先每次请求创建、请求结束关闭，
    把生命周期问题留给 ch15（并发）与 ch18（持久化）。
    """

    agent_path: str | Path = DEFAULT_AGENT
    workspace: str | Path = DEFAULT_WORKSPACE
    trace_dir: str | Path = DEFAULT_TRACE_DIR
    mock: bool = False
    model_factory: ModelFactory | None = None
    runs: dict[str, RunResponse] = field(default_factory=dict)

    async def run(self, task: str, run_id: str | None = None) -> RunResponse:
        state = State.load(self.agent_path)
        model = (
            self.model_factory(state)
            if self.model_factory is not None
            else create_model(state, mock=self.mock)
        )
        try:
            workspace = Path(self.workspace)
            trace_dir = Path(self.trace_dir)
            workspace.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)
            agent = Agent(
                model=model,
                tools=build_tools(state, workspace),
                state=state,
                trace_dir=trace_dir,
                workspace=workspace,
            )
            result = await agent.run(task, run_id=run_id)
        finally:
            await model.aclose()

        response = RunResponse.from_result(result)
        self.runs[response.run_id] = response
        return response


# --------------------------------------------------------------------------- #
# 2. 应用工厂：测试注入 service，生产用默认装配
# --------------------------------------------------------------------------- #
def create_app(
    service: RunService | None = None,
    *,
    agent_path: str | Path = DEFAULT_AGENT,
    workspace: str | Path = DEFAULT_WORKSPACE,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    mock: bool = False,
) -> FastAPI:
    service = service or RunService(
        agent_path=agent_path, workspace=workspace, trace_dir=trace_dir, mock=mock
    )
    app = FastAPI(title="evolving-harness", version="0.1.0")
    app.state.service = service

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", agent=str(service.agent_path))

    @app.post("/v1/runs", response_model=RunResponse, status_code=201)
    async def create_run(request: RunRequest) -> RunResponse:
        try:
            return await service.run(request.task, request.run_id)
        except AgentConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MaxStepsExceeded as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ModelError as exc:
            raise HTTPException(status_code=502, detail=f"model error: {exc}") from exc

    @app.get("/v1/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str) -> RunResponse:
        result = service.runs.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return result

    return app


app = create_app(mock=os.getenv("EH_HTTP_MOCK") == "1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("EH_HOST", "127.0.0.1"),
        port=int(os.getenv("EH_PORT", "8000")),
    )
