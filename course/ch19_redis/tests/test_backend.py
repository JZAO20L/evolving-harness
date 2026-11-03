"""L0 离线：WebSocket 事件推送、ping/pong、cancel、approve/deny 与单 writer 约束。"""

import asyncio
from typing import Any

import pytest
from backend import RunService, create_app
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from evolving_harness.core.message import Message, ToolCall
from evolving_harness.models.base import ChatModel, ModelResponse
from evolving_harness.models.mock import MockChatModel


def _mock_factory(*responses: ModelResponse):
    def factory(_state):
        return MockChatModel(list(responses))

    return factory


class SleepyModel(ChatModel):
    """跑得足够慢，让测试有机会在 run 期间发控制命令。"""

    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay

    async def complete(self, messages: list[Message], tools=None) -> ModelResponse:
        await asyncio.sleep(self.delay)
        return ModelResponse(content="slow done")


class HangingModel(ChatModel):
    async def complete(self, messages: list[Message], tools=None) -> ModelResponse:
        await asyncio.sleep(3600)
        return ModelResponse(content="never")


def _app(
    make_agent,
    tmp_path,
    factory,
    *,
    approval_timeout: float = 30.0,
    tools: dict | None = None,
):
    agent_path = make_agent(tools=tools) if tools else make_agent()
    return create_app(
        service=RunService(
            agent_path=agent_path,
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=factory,
            approval_timeout=approval_timeout,
        ),
        max_concurrency=1,
    )


def _recv_until(
    ws, predicate, *, limit: int = 300, allow_close: bool = False
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        try:
            frame = ws.receive_json()
        except Exception:
            if allow_close:
                return frames
            raise
        frames.append(frame)
        if predicate(frame):
            return frames
    raise AssertionError(f"condition not met; tail={frames[-8:]}")


def _of_type(frames, kind: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == kind]


def _event_kinds(frames) -> list[str]:
    return [frame["event"] for frame in _of_type(frames, "event")]


def test_ws_pushes_events_and_job_end(make_agent, tmp_path):
    app = _app(make_agent, tmp_path, _mock_factory(ModelResponse(content="ws 答案")))
    with TestClient(app) as client:
        assert client.post("/v1/runs", json={"task": "推送", "run_id": "ws-1"}).status_code == 202
        with client.websocket_connect("/v1/runs/ws-1/ws") as ws:
            frames = _recv_until(ws, lambda f: f["type"] == "job.end")

    assert frames[0]["type"] == "hello"
    assert frames[0]["run_id"] == "ws-1"
    kinds = _event_kinds(frames)
    assert "run.start" in kinds
    assert "model.request" in kinds
    assert "model.response" in kinds
    assert kinds[-1] == "run.end"
    job_end = frames[-1]["job"]
    assert job_end["status"] == "succeeded"
    assert job_end["result"]["final"] == "ws 答案"


def test_ws_ping_pong_while_running(make_agent, tmp_path):
    app = _app(make_agent, tmp_path, lambda _state: SleepyModel(0.6))
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "慢", "run_id": "ping-1"})
        with client.websocket_connect("/v1/runs/ping-1/ws") as ws:
            ws.send_json({"type": "ping"})
            frames = _recv_until(ws, lambda f: f["type"] == "pong")
            assert frames[-1]["ts"] > 0
            frames += _recv_until(ws, lambda f: f["type"] == "job.end")

    assert frames[-1]["job"]["status"] == "succeeded"


def test_ws_cancel_running_job(make_agent, tmp_path):
    app = _app(make_agent, tmp_path, lambda _state: HangingModel())
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "取消我", "run_id": "cancel-1"})
        with client.websocket_connect("/v1/runs/cancel-1/ws") as ws:
            # 等 model.request 落 trace，确认 job 已真正进入运行态
            _recv_until(
                ws,
                lambda f: f["type"] == "event" and f["event"] == "model.request",
            )
            ws.send_json({"type": "cancel"})
            frames = _recv_until(ws, lambda f: f["type"] == "job.end")

    acks = _of_type(frames, "cancel.ack")
    assert acks and acks[0]["accepted"] is True
    assert frames[-1]["job"]["status"] == "cancelled"
    assert "CancelledError" in frames[-1]["job"]["error"]


def test_ws_cancel_queued_job(make_agent, tmp_path):
    app = _app(make_agent, tmp_path, lambda _state: HangingModel())
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "占住 worker", "run_id": "busy-1"})
        client.post("/v1/runs", json={"task": "排队", "run_id": "queued-1"})

        body = client.get("/v1/runs/queued-1").json()
        assert body["status"] == "queued"

        with client.websocket_connect("/v1/runs/queued-1/ws") as ws:
            ws.send_json({"type": "cancel"})
            frames = _recv_until(ws, lambda f: f["type"] == "job.end")

        assert frames[-1]["job"]["status"] == "cancelled"
        assert "cancelled before start" in frames[-1]["job"]["error"]

        with client.websocket_connect("/v1/runs/busy-1/ws") as ws:
            _recv_until(
                ws,
                lambda f: f["type"] == "event" and f["event"] == "model.request",
            )
            ws.send_json({"type": "cancel"})
            frames = _recv_until(ws, lambda f: f["type"] == "job.end")
        assert frames[-1]["job"]["status"] == "cancelled"


def test_ws_approval_approve_writes_file(make_agent, tmp_path):
    factory = _mock_factory(
        ModelResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="w1",
                    name="write_file",
                    arguments={"path": "out.txt", "content": "approved!"},
                )
            ],
        ),
        ModelResponse(content="写完了"),
    )
    app = _app(
        make_agent,
        tmp_path,
        factory,
        tools={"enabled": ["read_file", "write_file"]},
    )
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "写文件", "run_id": "approve-1"})
        with client.websocket_connect("/v1/runs/approve-1/ws") as ws:
            frames = _recv_until(ws, lambda f: f["type"] == "approval.request")
            request = frames[-1]
            assert request["tool"] == "write_file"
            assert request["arguments"]["path"] == "out.txt"

            ws.send_json(
                {
                    "type": "approve",
                    "approval_id": request["approval_id"],
                    "approved": True,
                }
            )
            frames += _recv_until(ws, lambda f: f["type"] == "job.end")

    acks = _of_type(frames, "approval.ack")
    assert acks and acks[0]["accepted"] is True
    assert frames[-1]["job"]["status"] == "succeeded"
    assert (tmp_path / "ws" / "runs" / "approve-1" / "out.txt").read_text() == "approved!"


def test_ws_approval_deny_becomes_tool_error(make_agent, tmp_path):
    factory = _mock_factory(
        ModelResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="w2",
                    name="write_file",
                    arguments={"path": "nope.txt", "content": "blocked"},
                )
            ],
        ),
        ModelResponse(content="被拒绝了"),
    )
    app = _app(
        make_agent,
        tmp_path,
        factory,
        tools={"enabled": ["read_file", "write_file"]},
    )
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "写文件", "run_id": "deny-1"})
        with client.websocket_connect("/v1/runs/deny-1/ws") as ws:
            frames = _recv_until(ws, lambda f: f["type"] == "approval.request")
            request = frames[-1]
            ws.send_json(
                {
                    "type": "approve",
                    "approval_id": request["approval_id"],
                    "approved": False,
                }
            )
            frames += _recv_until(ws, lambda f: f["type"] == "job.end")

    assert frames[-1]["job"]["status"] == "succeeded"
    tool_results = [
        f for f in _of_type(frames, "event") if f["event"] == "tool.result"
    ]
    assert tool_results
    assert "approval denied" in tool_results[0]["record"]["data"]["content"]
    assert not (tmp_path / "ws" / "runs" / "deny-1" / "nope.txt").exists()


def test_ws_unknown_command_reports_error(make_agent, tmp_path):
    app = _app(make_agent, tmp_path, lambda _state: SleepyModel(0.6))
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "命令", "run_id": "cmd-1"})
        with client.websocket_connect("/v1/runs/cmd-1/ws") as ws:
            ws.send_json({"type": "nope"})
            frames = _recv_until(ws, lambda f: f["type"] == "error")
            assert "unknown command" in frames[-1]["message"]
            ws.send_json({"type": "cancel"})
            frames += _recv_until(ws, lambda f: f["type"] == "job.end")

    assert frames[-1]["job"]["status"] == "cancelled"


def test_ws_unknown_run_is_rejected(make_agent, tmp_path):
    app = _app(make_agent, tmp_path, _mock_factory(ModelResponse(content="x")))
    with TestClient(app) as client, client.websocket_connect("/v1/runs/nope/ws") as ws:
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "not found" in frame["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
