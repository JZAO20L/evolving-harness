"""L0 离线：ch15 的作业队列、并发上限、per-run workspace、背压与失败隔离。"""

import asyncio
from pathlib import Path

import httpx
import pytest
from backend import (
    JobManager,
    QueueFull,
    RunResponse,
    RunService,
    UsageOut,
    create_app,
)

from evolving_harness.core.message import Message
from evolving_harness.models.base import ChatModel, ModelError, ModelResponse
from evolving_harness.models.mock import MockChatModel


def _make_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


def _mock_factory(*responses: ModelResponse):
    def factory(_state):
        return MockChatModel(list(responses))

    return factory


class BoomModel(ChatModel):
    async def complete(self, messages: list[Message], tools=None) -> ModelResponse:
        raise ModelError("upstream exploded")


class GateService:
    """可控服务：所有 run 在 gate 上排队，用来观察并发上限。"""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.workspaces: list[Path] = []
        self.gate = asyncio.Event()

    async def run(self, task, run_id=None, workspace=None) -> RunResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.workspaces.append(Path(workspace))
        await self.gate.wait()
        self.active -= 1
        return RunResponse(
            run_id=run_id or "x",
            final=task,
            status="ok",
            steps=1,
            usage=UsageOut(),
            state_version=1,
            state_hash="0" * 64,
            trace_path="trace.jsonl",
            workspace=str(workspace),
        )


async def _wait_job(client, run_id: str, *, timeout: float = 3.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/v1/runs/{run_id}")
        body = response.json()
        if body.get("status") in ("succeeded", "failed"):
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {run_id} did not finish: {body}")


async def _wait_active(manager: JobManager, wanted: int, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while manager.active != wanted:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"active={manager.active}, wanted={wanted}")
        await asyncio.sleep(0.005)


async def test_submit_is_async_and_poll_reaches_success(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="队列答案")),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        created = await client.post("/v1/runs", json={"task": "排队", "run_id": "job-1"})
        assert created.status_code == 202
        assert created.json()["status"] in ("queued", "running")

        body = await _wait_job(client, "job-1")
        assert body["status"] == "succeeded"
        assert body["result"]["final"] == "队列答案"
        workspace = Path(body["workspace"])
        assert workspace.name == "job-1"
        assert workspace.is_dir()
        assert (tmp_path / "traces" / "job-1.jsonl").is_file()

    await app.state.manager.stop()


async def test_duplicate_run_id_is_idempotent(make_agent, tmp_path):
    calls = 0

    def factory(_state):
        nonlocal calls
        calls += 1
        return MockChatModel([ModelResponse(content="once")])

    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=factory,
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        first = await client.post("/v1/runs", json={"task": "x", "run_id": "same"})
        second = await client.post("/v1/runs", json={"task": "x", "run_id": "same"})
        assert first.status_code == 202 and second.status_code == 202
        assert first.json()["run_id"] == second.json()["run_id"] == "same"
        await _wait_job(client, "same")

    assert calls == 1, "同一 run_id 只能触发一次模型调用"
    await app.state.manager.stop()


async def test_two_runs_get_distinct_workspaces(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="ok")),
        ),
        max_concurrency=2,
    )
    async with _make_client(app) as client:
        await client.post("/v1/runs", json={"task": "a", "run_id": "a"})
        await client.post("/v1/runs", json={"task": "b", "run_id": "b"})
        a = await _wait_job(client, "a")
        b = await _wait_job(client, "b")

    assert a["workspace"] != b["workspace"]
    assert Path(a["workspace"]).is_dir() and Path(b["workspace"]).is_dir()
    await app.state.manager.stop()


async def test_worker_count_caps_concurrency(tmp_path):
    service = GateService()
    manager = JobManager(
        service,
        max_concurrency=2,
        queue_size=8,
        workspace_root=tmp_path / "runs",
    )
    for i in range(5):
        await manager.submit(f"task-{i}", f"j{i}")

    await _wait_active(manager, 2)
    assert manager.max_concurrency == 2
    assert manager.queued == 3

    service.gate.set()
    await manager.wait_idle()
    assert service.max_active == 2
    assert all(job.status == "succeeded" for job in manager.jobs.values())
    assert len({str(p) for p in service.workspaces}) == 5
    await manager.stop()


async def test_queue_full_rejects_with_backpressure(tmp_path):
    service = GateService()
    manager = JobManager(
        service,
        max_concurrency=1,
        queue_size=1,
        workspace_root=tmp_path / "runs",
    )
    await manager.submit("running", "j1")
    await _wait_active(manager, 1)
    await manager.submit("waiting", "j2")
    assert manager.queued == 1

    with pytest.raises(QueueFull):
        await manager.submit("rejected", "j3")

    service.gate.set()
    await manager.wait_idle()
    await manager.stop()


async def test_model_failure_is_isolated_in_job(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            model_factory=lambda _state: BoomModel(),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        await client.post("/v1/runs", json={"task": "fail", "run_id": "bad"})
        body = await _wait_job(client, "bad")

    assert body["status"] == "failed"
    assert "ModelError" in body["error"]
    assert body["result"] is None
    await app.state.manager.stop()


async def test_validation_and_unknown_job(make_agent, tmp_path):
    app = create_app(service=RunService(agent_path=make_agent(), workspace=tmp_path / "ws"))
    async with _make_client(app) as client:
        assert (await client.post("/v1/runs", json={"task": " "})).status_code == 422
        assert (
            await client.post("/v1/runs", json={"task": "x", "api_key": "leak"})
        ).status_code == 422
        assert (
            await client.post("/v1/runs", json={"task": "x", "run_id": "../evil"})
        ).status_code == 422
        assert (await client.get("/v1/runs/nope")).status_code == 404

        health = await client.get("/health")
        assert health.json()["concurrency"] == 2

    await app.state.manager.stop()
