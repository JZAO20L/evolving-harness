"""L0 离线：ch15 的 HTTP 契约、服务层装配、错误码。全部用 Mock 模型，不联网。"""

import json

import httpx
from backend import RunService, create_app

from evolving_harness.core.message import Message, ToolCall
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


async def test_health_reports_agent_path(make_agent, tmp_path):
    app = create_app(service=RunService(agent_path=make_agent(), workspace=tmp_path / "ws"))
    async with _make_client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["agent"].endswith("agent.yaml")


async def test_post_run_returns_contract_and_persists_lookup(make_agent, tmp_path):
    path = make_agent()
    app = create_app(
        service=RunService(
            agent_path=path,
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="HTTP 答案")),
        )
    )
    async with _make_client(app) as client:
        response = await client.post("/v1/runs", json={"task": "说一句话", "run_id": "r-1"})
        assert response.status_code == 201
        body = response.json()

        assert body["run_id"] == "r-1"
        assert body["final"] == "HTTP 答案"
        assert body["status"] == "ok"
        assert body["steps"] == 1
        assert body["usage"]["total_tokens"] == 0
        assert body["state_version"] == 1
        assert len(body["state_hash"]) == 64
        assert (tmp_path / "traces" / "r-1.jsonl").is_file()

        found = await client.get("/v1/runs/r-1")
        assert found.status_code == 200
        assert found.json() == body

        missing = await client.get("/v1/runs/nope")
        assert missing.status_code == 404


async def test_request_validation_rejects_blank_and_unknown_fields(make_agent, tmp_path):
    app = create_app(service=RunService(agent_path=make_agent(), workspace=tmp_path / "ws"))
    async with _make_client(app) as client:
        blank = await client.post("/v1/runs", json={"task": "   "})
        assert blank.status_code == 422

        empty = await client.post("/v1/runs", json={})
        assert empty.status_code == 422

        extra = await client.post("/v1/runs", json={"task": "x", "api_key": "leak"})
        assert extra.status_code == 422

        bad_id = await client.post("/v1/runs", json={"task": "x", "run_id": "../etc"})
        assert bad_id.status_code == 422


async def test_bad_agent_config_maps_to_400(tmp_path):
    app = create_app(
        service=RunService(agent_path=tmp_path / "missing.yaml", workspace=tmp_path / "ws")
    )
    async with _make_client(app) as client:
        response = await client.post("/v1/runs", json={"task": "x"})

    assert response.status_code == 400
    assert "agent config not found" in response.json()["detail"]


async def test_max_steps_maps_to_409(make_agent, tmp_path):
    call = ToolCall(id="c1", name="read_file", arguments={"path": "missing.txt"})
    app = create_app(
        service=RunService(
            agent_path=make_agent(limits={"max_steps": 1}),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(tool_calls=[call])),
        )
    )
    async with _make_client(app) as client:
        response = await client.post("/v1/runs", json={"task": "loop"})

    assert response.status_code == 409
    assert "exceeded 1 steps" in response.json()["detail"]


async def test_model_error_maps_to_502(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            model_factory=lambda _state: BoomModel(),
        )
    )
    async with _make_client(app) as client:
        response = await client.post("/v1/runs", json={"task": "x"})

    assert response.status_code == 502
    assert "model error" in response.json()["detail"]


async def test_trace_contains_http_run_events(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="done")),
        )
    )
    async with _make_client(app) as client:
        response = await client.post("/v1/runs", json={"task": "hello", "run_id": "trace-1"})

    trace = tmp_path / "traces" / "trace-1.jsonl"
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert events[0]["kind"] == "run.start"
    assert events[-1]["kind"] == "run.end"
    assert events[-1]["data"]["status"] == "ok"
    assert response.json()["trace_path"].endswith("trace-1.jsonl")
