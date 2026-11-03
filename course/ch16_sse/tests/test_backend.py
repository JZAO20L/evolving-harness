"""L0 离线：SSE 事件序、终止、断线续传、工具事件与失败收尾。"""

import json
from typing import Any

import httpx
from backend import RunService, create_app

from evolving_harness.core.message import Message, ToolCall
from evolving_harness.models.base import ChatModel, ModelError, ModelResponse
from evolving_harness.models.mock import MockChatModel


def _make_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=15.0,
    )


def _mock_factory(*responses: ModelResponse):
    def factory(_state):
        return MockChatModel(list(responses))

    return factory


class BoomModel(ChatModel):
    async def complete(self, messages: list[Message], tools=None) -> ModelResponse:
        raise ModelError("upstream exploded")


async def _collect(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """读完整条 SSE，返回响应头与 [{event, id, data}]。"""
    events: list[dict[str, Any]] = []
    async with client.stream("GET", url, headers=headers) as response:
        assert response.status_code == 200, await response.aread()
        headers_out = {
            "content-type": response.headers.get("content-type", ""),
            "cache-control": response.headers.get("cache-control", ""),
        }
        event: str | None = None
        event_id: int | None = None
        data: list[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if event is not None:
                    events.append(
                        {
                            "event": event,
                            "id": event_id,
                            "data": json.loads("\n".join(data)) if data else None,
                        }
                    )
                event, event_id, data = None, None, []
            elif line.startswith("event: "):
                event = line[7:]
            elif line.startswith("id: "):
                event_id = int(line[4:])
            elif line.startswith("data: "):
                data.append(line[6:])
            elif line.startswith(":"):
                continue
    return headers_out, events


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [item["event"] for item in events]


async def test_stream_pushes_trace_then_job_end(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="流式答案")),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        created = await client.post("/v1/runs", json={"task": "推送", "run_id": "sse-1"})
        assert created.status_code == 202
        headers, events = await _collect(client, "/v1/runs/sse-1/events")

    kinds = _kinds(events)
    assert kinds[0] == "open"
    assert "run.start" in kinds
    assert "model.request" in kinds
    assert "model.response" in kinds
    assert kinds[-2:] == ["run.end", "job.end"]
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"

    trace_ids = [e["id"] for e in events if e["id"] is not None]
    assert trace_ids == sorted(trace_ids)
    assert len(trace_ids) == len(set(trace_ids))

    run_end = next(e for e in events if e["event"] == "run.end")
    assert run_end["data"]["data"]["status"] == "ok"
    job_end = events[-1]["data"]
    assert job_end["status"] == "succeeded"
    assert job_end["result"]["final"] == "流式答案"
    await app.state.manager.stop()


async def test_stream_includes_tool_call_and_result(make_agent, tmp_path):
    workspace = tmp_path / "ws" / "runs" / "tool-job"
    workspace.mkdir(parents=True)
    (workspace / "hello.txt").write_text("hi from workspace\n", encoding="utf-8")

    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(
                ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "hello.txt"})
                    ],
                ),
                ModelResponse(content="读到了"),
            ),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        await client.post("/v1/runs", json={"task": "读文件", "run_id": "tool-job"})
        _, events = await _collect(client, "/v1/runs/tool-job/events")

    kinds = _kinds(events)
    assert kinds.count("model.request") == 2
    assert "tool.call" in kinds and "tool.result" in kinds
    assert kinds.index("tool.call") < kinds.index("tool.result") < kinds.index("run.end")
    tool_result = next(e for e in events if e["event"] == "tool.result")
    assert "hi from workspace" in tool_result["data"]["data"]["content"]
    assert events[-1]["data"]["result"]["final"] == "读到了"
    await app.state.manager.stop()


async def test_last_event_id_resumes_without_replay(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="done")),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        await client.post("/v1/runs", json={"task": "续传", "run_id": "resume-1"})
        _, full = await _collect(client, "/v1/runs/resume-1/events")
        request_event = next(e for e in full if e["event"] == "model.request")
        resume_id = request_event["id"]
        _, resumed = await _collect(
            client,
            "/v1/runs/resume-1/events",
            headers={"Last-Event-ID": str(resume_id)},
        )

    replay_ids = [e["id"] for e in resumed if e["id"] is not None]
    assert replay_ids, "completed job should still emit later trace events"
    assert min(replay_ids) > resume_id
    assert "run.start" not in _kinds(resumed)
    assert any(e["event"] == "run.end" for e in resumed)
    assert resumed[-1]["event"] == "job.end"
    await app.state.manager.stop()


async def test_completed_job_replays_from_zero(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=_mock_factory(ModelResponse(content="once")),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        await client.post("/v1/runs", json={"task": "重放", "run_id": "replay-1"})
        await _collect(client, "/v1/runs/replay-1/events")
        _, again = await _collect(client, "/v1/runs/replay-1/events")

    assert "run.start" in _kinds(again)
    assert again[-1]["event"] == "job.end"
    assert again[-1]["data"]["status"] == "succeeded"
    await app.state.manager.stop()


async def test_stream_finishes_when_model_fails(make_agent, tmp_path):
    app = create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=lambda _state: BoomModel(),
        ),
        max_concurrency=1,
    )
    async with _make_client(app) as client:
        await client.post("/v1/runs", json={"task": "炸", "run_id": "bad-1"})
        _, events = await _collect(client, "/v1/runs/bad-1/events")

    kinds = _kinds(events)
    assert "run.end" in kinds
    run_end = next(e for e in events if e["event"] == "run.end")
    assert run_end["data"]["data"]["status"] == "error"
    job_end = events[-1]
    assert job_end["event"] == "job.end"
    assert job_end["data"]["status"] == "failed"
    assert "ModelError" in job_end["data"]["error"]
    await app.state.manager.stop()


async def test_unknown_run_returns_404(make_agent, tmp_path):
    app = create_app(
        service=RunService(agent_path=make_agent(), workspace=tmp_path / "ws")
    )
    async with _make_client(app) as client:
        response = await client.get("/v1/runs/nope/events")
        assert response.status_code == 404
    await app.state.manager.stop()
