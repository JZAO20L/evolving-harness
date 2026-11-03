"""L0：URL/协议解析、状态归约、渲染、错误提示、HTTP 客户端与 WS 重连（全离线）。"""

import asyncio
import json

import frontend
import httpx
import pytest
from frontend import (
    BackendClient,
    RunSession,
    UiState,
    apply_message,
    approval_preview,
    describe_error,
    parse_ws_message,
    render_approval,
    render_chat,
    render_status,
    render_trace,
    stream_state,
    with_after,
    ws_url,
)


class FakeWS:
    def __init__(self, incoming, *, error=None):
        self.incoming = [item if isinstance(item, str) else json.dumps(item) for item in incoming]
        self.sent: list[dict] = []
        self.closed = False
        self.error = error

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        self._iterator = iter(self.incoming)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            if self.error is not None:
                raise self.error
            raise StopAsyncIteration from None


def event(seq: int, kind: str, **data) -> dict:
    return {
        "type": "event",
        "event": kind,
        "seq": seq,
        "record": {"seq": seq, "kind": kind, "elapsed_ms": seq * 10, "data": data},
    }


def test_ws_url_from_http_and_https():
    assert ws_url("http://h:8000", "r1") == "ws://h:8000/v1/runs/r1/ws"
    assert ws_url("http://h:8000/", "r1") == "ws://h:8000/v1/runs/r1/ws"
    assert ws_url("https://h/api", "r 1") == "wss://h/api/v1/runs/r%201/ws"
    with pytest.raises(ValueError):
        ws_url("h:8000", "r1")


def test_with_after_appends_and_replaces_seq():
    assert with_after("ws://h/x", 0) == "ws://h/x"
    assert with_after("ws://h/x", 2) == "ws://h/x?after=2"
    assert with_after("ws://h/x?foo=1", 2) == "ws://h/x?foo=1&after=2"
    assert with_after("ws://h/x?after=1&foo=1", 3) == "ws://h/x?foo=1&after=3"


def _status_error(status: int, **kwargs) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend/v1/runs")
    response = httpx.Response(status, request=request, **kwargs)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_describe_error_maps_http_and_network_errors():
    rate_limited = describe_error(
        _status_error(
            429, json={"detail": "rate limit exceeded"}, headers={"Retry-After": "7"}
        )
    )
    assert "请求过于频繁" in rate_limited
    assert "rate limit exceeded" in rate_limited
    assert "7 秒后可重试" in rate_limited

    assert "run_id 冲突" in describe_error(_status_error(409, json={"detail": "dup"}))
    assert "任务不存在" in describe_error(_status_error(404))
    assert "无法连接后端" in describe_error(httpx.ConnectError("refused"))
    assert "超时" in describe_error(httpx.ReadTimeout("slow"))
    assert "ValueError" in describe_error(ValueError("bad"))


def test_status_label_is_human_readable():
    assert render_status(UiState(status="succeeded")).startswith("**状态**：已完成")
    assert "等待审批" in render_status(UiState(status="waiting_approval"))


def test_parse_ws_message_accepts_str_bytes_dict():
    assert parse_ws_message('{"type": "hello"}') == {"type": "hello"}
    assert parse_ws_message(b'{"type": "hello"}') == {"type": "hello"}
    assert parse_ws_message({"type": "hello"}) == {"type": "hello"}
    with pytest.raises(TypeError):
        parse_ws_message("[]")
    with pytest.raises(TypeError):
        parse_ws_message(42)


def test_model_response_appends_assistant_text():
    state = UiState()
    apply_message(state, event(1, "model.response", step=1, content="hello"))
    apply_message(state, event(2, "model.response", step=2, content="   "))
    assert state.messages == [{"role": "assistant", "content": "hello"}]
    assert [item["kind"] for item in state.trace] == ["model.response", "model.response"]


def test_tool_events_become_chat_and_mark_errors():
    state = UiState()
    apply_message(state, event(1, "tool.call", name="bash", arguments={"command": "false"}))
    apply_message(
        state,
        event(2, "tool.result", name="bash", content="ERROR: exit=1", is_error=True),
    )
    assert "`bash`" in state.messages[0]["content"]
    assert state.messages[1]["content"].startswith("[ERROR]")
    assert state.status == "running_tool"


def test_approval_request_and_ack_update_pending():
    state = UiState()
    apply_message(
        state,
        {
            "type": "approval.request",
            "approval_id": "a1",
            "tool": "write_file",
            "arguments": {"path": "notes.md", "content": "hi"},
        },
    )
    assert state.status == "waiting_approval"
    assert state.pending["a1"]["tool"] == "write_file"
    markdown, approval_id = render_approval(state)
    assert approval_id == "a1" and "notes.md" in markdown

    apply_message(state, {"type": "approval.ack", "approval_id": "a1", "approved": True})
    assert state.pending == {}
    assert state.status == "running"


def test_job_end_sets_terminal_final_and_clears_pending():
    state = UiState(pending={"a1": {"tool": "bash", "arguments": {}}})
    apply_message(
        state,
        {
            "type": "job.end",
            "job": {
                "run_id": "r1",
                "status": "succeeded",
                "result": {"final": "done"},
                "error": None,
            },
        },
    )
    assert state.status == "succeeded"
    assert state.final == "done"
    assert state.pending == {}
    assert state.terminal


def test_error_and_heartbeat_messages():
    state = UiState()
    apply_message(state, {"type": "heartbeat", "ts": 1})
    assert state.status == "idle"
    apply_message(state, {"type": "error", "message": "boom"})
    assert state.status == "failed"
    assert state.error == "boom"


def test_trace_is_bounded():
    state = UiState()
    for seq in range(frontend.MAX_TRACE_EVENTS + 5):
        apply_message(state, event(seq, "model.request", step=seq))
    assert len(state.trace) == frontend.MAX_TRACE_EVENTS
    assert state.trace[0]["seq"] == 5


def test_render_chat_status_and_trace():
    state = UiState(run_id="r1", status="running")
    state.messages.append({"role": "user", "content": "hi"})
    apply_message(state, event(1, "model.response", content="hello"))
    assert render_chat(state)[-1]["content"] == "hello"
    assert "r1" in render_status(state)
    markdown = render_trace(state)
    assert "#1 model.response" in markdown and "hello" in markdown
    assert render_trace(UiState()) == "_暂无事件_"


def test_approval_preview_is_a_diff_for_write_file():
    preview = approval_preview(
        "write_file", {"path": "skills/a.md", "content": "line1\nline2"}
    )
    assert "skills/a.md" in preview
    assert "+ line1" in preview and "+ line2" in preview
    assert "echo hi" in approval_preview("bash", {"command": "echo hi"})


def _client_calls(calls: list[tuple[str, str, dict]]):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode()) if request.content else {}
        calls.append((request.method, request.url.path, payload))
        if request.url.path == "/v1/runs" and request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "run_id": payload["run_id"],
                    "task": payload["task"],
                    "status": "queued",
                    "workspace": "/tmp/ws",
                    "created_at": 1.0,
                },
            )
        if request.url.path == "/v1/runs":
            return httpx.Response(200, json=[])
        if request.url.path.startswith("/v1/runs/"):
            return httpx.Response(200, json={"run_id": "r1", "status": "succeeded"})
        return httpx.Response(200, json={"status": "ok"})

    return handler


def test_backend_client_http_round_trips():
    calls: list[tuple[str, str, dict]] = []
    transport = httpx.MockTransport(_client_calls(calls))
    http = httpx.AsyncClient(transport=transport)

    async def run():
        client = BackendClient("http://backend/", http=http)
        created = await client.create_run("hello", "r1")
        assert created["run_id"] == "r1"
        assert (await client.get_run("r1"))["status"] == "succeeded"
        assert await client.list_runs() == []
        assert (await client.health())["status"] == "ok"
        await client.aclose()


    asyncio.run(run())
    assert calls[0] == ("POST", "/v1/runs", {"task": "hello", "run_id": "r1"})
    assert calls[1] == ("GET", "/v1/runs/r1", {})
    assert calls[2] == ("GET", "/v1/runs", {})
    assert calls[3] == ("GET", "/health", {})


def test_backend_client_generates_run_id_and_raises_on_errors():
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "rate limit exceeded"})

    async def run():
        client = BackendClient(
            "http://backend", http=httpx.AsyncClient(transport=httpx.MockTransport(error_handler))
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.create_run("hello")
        await client.aclose()


    asyncio.run(run())


def test_run_session_sends_control_commands_on_same_socket():
    ws = FakeWS([{"type": "hello", "run_id": "r1"}, {"type": "job.end", "job": {"status": "succeeded"}}])
    seen_urls = []

    async def connect(url):
        seen_urls.append(url)
        return ws

    async def run():
        session = RunSession("r1", "ws://h/v1/runs/r1/ws", connect=connect)
        await session.open()
        await session.approve("a1", True)
        await session.cancel()
        await session.ping()
        messages = [message async for message in session.messages()]
        await session.close()
        return messages


    messages = asyncio.run(run())
    assert seen_urls == ["ws://h/v1/runs/r1/ws"]
    assert messages[0]["type"] == "hello"
    assert ws.sent == [
        {"type": "approve", "approval_id": "a1", "approved": True},
        {"type": "cancel"},
        {"type": "ping"},
    ]
    assert ws.closed


def test_run_session_reconnects_with_after_and_continues():
    ws1 = FakeWS(
        [
            {"type": "hello", "run_id": "r1"},
            event(1, "model.response", content="part1"),
            event(2, "model.response", content="part2"),
        ],
        error=ConnectionError("dropped"),
    )
    ws2 = FakeWS(
        [
            {"type": "hello", "run_id": "r1", "after": 2},
            event(3, "model.response", content="part3"),
            {"type": "job.end", "job": {"status": "succeeded", "result": {"final": "part3"}}},
        ]
    )
    urls: list[str] = []

    async def connect(url):
        urls.append(url)
        return ws1 if len(urls) == 1 else ws2

    async def run():
        state = UiState()
        session = RunSession(
            "r1",
            "ws://h/v1/runs/r1/ws",
            connect=connect,
            reconnect_attempts=2,
            reconnect_delay=0,
        )
        async for _ in stream_state(session, state):
            pass
        await session.close()
        return state, session

    state, session = asyncio.run(run())
    assert urls == ["ws://h/v1/runs/r1/ws", "ws://h/v1/runs/r1/ws?after=2"]
    assert session.reconnects == 1
    assert state.reconnects == 1
    assert [record["seq"] for record in state.trace] == [1, 2, 3]
    assert state.status == "succeeded"
    assert state.final == "part3"


def test_run_session_gives_up_after_reconnect_budget():
    urls: list[str] = []

    async def connect(url):
        urls.append(url)
        return FakeWS([{"type": "hello", "run_id": "r1"}], error=ConnectionError("drop"))

    async def run():
        session = RunSession(
            "r1", "ws://h/v1/runs/r1/ws", connect=connect, reconnect_attempts=1, reconnect_delay=0
        )
        async for _ in session.messages():
            pass

    with pytest.raises(ConnectionError):
        asyncio.run(run())
    assert len(urls) == 2


def test_run_session_does_not_reconnect_after_job_end():
    urls: list[str] = []

    async def connect(url):
        urls.append(url)
        return FakeWS(
            [{"type": "job.end", "job": {"status": "succeeded", "result": {"final": "ok"}}}]
        )

    async def run():
        session = RunSession("r1", "ws://h", connect=connect, reconnect_delay=0)
        async for _ in session.messages():
            pass
        await session.close()

    asyncio.run(run())
    assert urls == ["ws://h"]


def test_stream_state_reduces_until_terminal():
    ws = FakeWS(
        [
            {"type": "hello", "run_id": "r1"},
            event(1, "model.response", content="hello"),
            {"type": "job.status", "status": "running"},
            {
                "type": "job.end",
                "job": {"status": "succeeded", "result": {"final": "hello"}},
            },
        ]
    )

    async def connect(_url):
        return ws

    async def run():
        state = UiState()
        session = RunSession("r1", "ws://h", connect=connect)
        async for _ in stream_state(session, state):
            pass
        await session.close()
        return state


    state = asyncio.run(run())
    assert state.run_id == "r1"
    assert state.status == "succeeded"
    assert state.final == "hello"
    assert state.messages[-1]["content"] == "hello"


def test_build_ui_reports_missing_gradio(monkeypatch):
    def missing():
        raise RuntimeError("gradio 未安装：uv sync --extra frontend")

    monkeypatch.setattr(frontend, "_import_gradio", missing)
    with pytest.raises(RuntimeError, match="gradio"):
        frontend.build_ui("http://backend")


def test_build_ui_wires_when_gradio_is_installed():
    pytest.importorskip("gradio")
    client = BackendClient("http://backend")
    demo = frontend.build_ui("http://backend", client=client)
    assert demo is not None
