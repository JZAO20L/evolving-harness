"""ch22 · gradio —— 最简 WebUI：运行、流式、trace 与工具审批。

前端不 import 任何后端的 Python 对象，只消费 ch15–ch18 的 HTTP / WebSocket 协议：

```
浏览器 ── Gradio ──► BackendClient ──POST /v1/runs──► 后端
                          │
                          └──WS /v1/runs/{id}/ws──► 事件流（trace + 控制消息）
                                     ▲
                     approve / cancel│（同一个 WS 连接）
```

设计要点：
- **一个 run 一个 WS 会话**：后端把审批 Future 绑在 WS 连接上，连接一断就拒绝
  所有未决审批。因此 UI 必须复用同一条连接来发 `approve`/`cancel`，不能再开
  一条短连接；
- **状态归约**：把 WS 消息折叠成 `UiState`（对话 / trace / 待审批 / 终态），
  UI 只负责渲染，便于离线测试；
- **gradio 是可选依赖**：`frontend.py` 顶层不 import gradio，L0 测试可以只测
  客户端协议；真正启动 UI 时才提示安装 `--extra frontend`；
- **审批卡片是 diff 预览**：`write_file` 展示“将写入的路径 + 新增内容”，
  `bash` 展示完整命令；生产版可换成 side-by-side diff。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Self
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import httpx

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
MAX_TRACE_EVENTS = 500
TRACE_RENDER_LIMIT = 25
MAX_TEXT = 4000
MAX_TOOL_TEXT = 1500

STATUS_LABELS = {
    "idle": "空闲",
    "queued": "排队中",
    "running": "运行中",
    "running_tool": "执行工具",
    "waiting_approval": "等待审批",
    "streaming": "已连接",
    "cancelling": "取消中",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}

WsConnect = Callable[[str], Awaitable[Any]]


# --------------------------------------------------------------------------- #
# 1. 协议：URL、消息解析、状态归约
# --------------------------------------------------------------------------- #
def ws_url(base_url: str, run_id: str) -> str:
    """http(s)://host[/prefix] → ws(s)://host[/prefix]/v1/runs/{id}/ws。"""
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid backend URL: {base_url!r}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    prefix = parsed.path.rstrip("/")
    path = f"{prefix}/v1/runs/{quote(run_id, safe='')}/ws"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def parse_ws_message(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        message = raw
    else:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise TypeError(f"WebSocket message must be str/bytes/dict, got {type(raw)!r}")
        message = json.loads(raw)
    if not isinstance(message, dict):
        raise TypeError("WebSocket message must be a JSON object")
    return message


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (truncated, {len(text)} chars)"


@dataclass
class UiState:
    """前端唯一的可变状态；所有渲染都从它派生。"""

    run_id: str = ""
    status: str = "idle"
    messages: list[dict[str, str]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    final: str = ""
    error: str = ""
    reconnects: int = 0

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


def apply_message(state: UiState, message: dict[str, Any]) -> UiState:
    """把一条 WS 消息折叠进 UiState；未知类型忽略，保证前后端可独立演进。"""
    kind = message.get("type")

    if kind == "hello":
        state.run_id = str(message.get("run_id") or state.run_id)
        if not state.terminal:
            state.status = "streaming"
        return state

    if kind == "event":
        record = message.get("record")
        if isinstance(record, dict):
            state.trace.append(record)
            if len(state.trace) > MAX_TRACE_EVENTS:
                del state.trace[: len(state.trace) - MAX_TRACE_EVENTS]
            _apply_trace_record(state, record)
        return state

    if kind == "job.status":
        state.status = str(message.get("status") or state.status)
        return state

    if kind == "approval.request":
        approval_id = str(message.get("approval_id") or "")
        if approval_id:
            state.pending[approval_id] = {
                "tool": message.get("tool"),
                "arguments": message.get("arguments") or {},
            }
            state.status = "waiting_approval"
            state.messages.append(
                {
                    "role": "assistant",
                    "content": f"[审批] 工具 `{message.get('tool')}` 等待人工批准（`{approval_id}`）",
                }
            )
        return state

    if kind == "approval.ack":
        approval_id = str(message.get("approval_id") or "")
        state.pending.pop(approval_id, None)
        if not state.pending and state.status == "waiting_approval":
            state.status = "running"
        return state

    if kind == "cancel.ack":
        if message.get("accepted") is False:
            state.error = "取消失败：任务已结束或不属于本实例"
        return state

    if kind == "job.end":
        job = message.get("job") or {}
        state.status = str(job.get("status") or "succeeded")
        result = job.get("result") or {}
        if isinstance(result, dict) and result.get("final"):
            state.final = str(result["final"])
        if job.get("error"):
            state.error = str(job["error"])
        state.pending.clear()
        return state

    if kind == "error":
        state.error = str(message.get("message") or "unknown frontend error")
        state.status = "failed"
        return state

    # heartbeat / pong / 未知控制消息：不影响 UI。
    return state


def _apply_trace_record(state: UiState, record: dict[str, Any]) -> None:
    kind = record.get("kind")
    data = record.get("data") or {}
    if not isinstance(data, dict):
        return

    if kind == "model.response":
        content = str(data.get("content") or "").strip()
        if content:
            state.messages.append({"role": "assistant", "content": content})
        return

    if kind == "tool.call":
        name = data.get("name") or "tool"
        arguments = json.dumps(data.get("arguments") or {}, ensure_ascii=False)
        state.messages.append(
            {"role": "assistant", "content": f"[调用] `{name}` {_clip(arguments, MAX_TOOL_TEXT)}"}
        )
        state.status = "running_tool"
        return

    if kind == "tool.result":
        body = _clip(str(data.get("content") or ""), MAX_TOOL_TEXT)
        mark = "ERROR" if data.get("is_error") else "result"
        state.messages.append({"role": "assistant", "content": f"[{mark}] {body}"})
        return

    if kind == "run.end":
        final = data.get("final")
        if final:
            state.final = str(final)
        if data.get("error"):
            state.error = str(data["error"])
        return

    if kind in ("memory.write", "skill.load", "subagent.start", "subagent.end"):
        state.messages.append(
            {
                "role": "assistant",
                "content": f"· {kind} {json.dumps(data, ensure_ascii=False, default=str)[:300]}",
            }
        )


# --------------------------------------------------------------------------- #
# 2. 渲染：状态 → Gradio 组件值
# --------------------------------------------------------------------------- #
def render_chat(state: UiState) -> list[dict[str, str]]:
    return [dict(item) for item in state.messages]


def render_status(state: UiState) -> str:
    label = STATUS_LABELS.get(state.status, state.status)
    lines = [
        f"**状态**：{label}（`{state.status}`）",
        f"**run_id**：`{state.run_id or '-'}`  ·  **事件**：{len(state.trace)}",
    ]
    if state.reconnects:
        lines.append(f"**重连**：{state.reconnects} 次（按 seq 续传）")
    if state.pending:
        lines.append(f"**待批准**：{', '.join(f'`{key}`' for key in state.pending)}")
    if state.final:
        lines.append(f"\n**最终结果**\n\n{_clip(state.final)}")
    if state.error:
        lines.append(f"\n**错误**\n\n```\n{_clip(state.error)}\n```")
    return "\n\n".join(lines)


def render_trace(state: UiState) -> str:
    if not state.trace:
        return "_暂无事件_"
    records = state.trace[-TRACE_RENDER_LIMIT:]
    chunks = []
    for record in records:
        seq = record.get("seq", "?")
        kind = record.get("kind", "trace")
        elapsed = record.get("elapsed_ms", 0)
        payload = json.dumps(record.get("data") or {}, ensure_ascii=False, default=str)
        chunks.append(f"**#{seq} {kind}** · {elapsed}ms\n```json\n{_clip(payload, 1200)}\n```")
    if len(state.trace) > TRACE_RENDER_LIMIT:
        chunks.insert(0, f"_仅显示最近 {TRACE_RENDER_LIMIT} / {len(state.trace)} 条事件_")
    return "\n\n".join(chunks)


def approval_preview(tool: str | None, arguments: dict[str, Any]) -> str:
    """审批卡片：把“将要发生的副作用”摊开给人看。"""
    if tool == "bash":
        return f"**命令**\n```bash\n{_clip(str(arguments.get('command') or ''), MAX_TOOL_TEXT)}\n```"
    if tool == "write_file":
        path = arguments.get("path") or "?"
        content = str(arguments.get("content") or "")
        added = "\n".join(f"+ {line}" for line in _clip(content, MAX_TEXT).splitlines())
        return f"**将写入** `{path}`\n```diff\n{added or '+'}\n```"
    payload = json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
    return f"**参数**\n```json\n{_clip(payload, MAX_TEXT)}\n```"


def render_approval(state: UiState) -> tuple[str, str]:
    """返回 (markdown, 第一条待批准 id)。"""
    if not state.pending:
        return "_没有待批准的工具调用_", ""
    approval_id, request = next(iter(state.pending.items()))
    tool = request.get("tool")
    arguments = request.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    header = f"### 待批准：`{tool}`\n\n`approval_id`: `{approval_id}`"
    if len(state.pending) > 1:
        header += f"\n\n_还有 {len(state.pending) - 1} 条待批准；批准后自动显示下一条_"
    return header + "\n\n" + approval_preview(tool, arguments), approval_id


# --------------------------------------------------------------------------- #
# 3. HTTP + WebSocket 客户端
# --------------------------------------------------------------------------- #
def describe_error(exc: BaseException) -> str:
    """把 HTTP/网络异常翻译成用户能采取行动的中文提示。"""
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        try:
            payload = response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else payload
        except Exception:  # noqa: BLE001 - 非 JSON 错误体
            detail = response.text
        hints = {
            404: "任务不存在",
            409: "run_id 冲突，或任务运行在其他实例",
            429: "请求过于频繁",
            503: "后端暂不可用",
        }
        hint = hints.get(response.status_code, "请求失败")
        parts = [f"{hint}（HTTP {response.status_code}）"]
        if detail:
            parts.append(str(detail).strip())
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            parts.append(f"{retry_after} 秒后可重试")
        return "：".join(parts)
    if isinstance(exc, (httpx.ConnectError, ConnectionError)):
        return "无法连接后端，请确认服务已启动且地址正确"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "请求后端超时，请稍后重试"
    return f"{type(exc).__name__}: {exc}"


def with_after(url: str, after: int) -> str:
    """给 WS URL 追加 after=seq，用于断线续传（服务端跳过已收到的 trace）。"""
    if after <= 0:
        return url
    parts = urlparse(url)
    query = [(key, value) for key, value in parse_qsl(parts.query) if key != "after"]
    query.append(("after", str(int(after))))
    return urlunparse(parts._replace(query=urlencode(query)))


def _message_seq(message: dict[str, Any]) -> int:
    value = message.get("seq")
    if value is None and isinstance(message.get("record"), dict):
        value = message["record"].get("seq")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return True
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:  # pragma: no cover - 未安装时只用标准异常判断
        return False
    return isinstance(exc, ConnectionClosed)


async def _default_ws_connect(url: str) -> Any:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise RuntimeError("websockets 未安装：uv sync --extra dev") from exc
    return await websockets.connect(url, ping_interval=20, ping_timeout=20)


class RunSession:
    """一个 run 的 WS 会话；发送控制命令与接收事件复用同一条连接。

    断线后按 `after=seq` 自动续传，最多重试 `reconnect_attempts` 次。审批/取消
    仍走当前连接；重连后的新连接只负责继续下行事件。
    """

    def __init__(
        self,
        run_id: str,
        url: str,
        *,
        connect: WsConnect | None = None,
        after: int = 0,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 0.5,
    ) -> None:
        self.run_id = run_id
        self.url = url
        self._connect = connect or _default_ws_connect
        self.after = max(0, int(after))
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_delay = max(0.0, float(reconnect_delay))
        self.reconnects = 0
        self.done = False
        self._closed = False
        self._ws: Any = None

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _connect_url(self) -> str:
        return with_after(self.url, self.after)

    async def open(self) -> None:
        if self._closed:
            raise RuntimeError("WebSocket session is closed")
        if self._ws is None:
            self._ws = await self._connect(self._connect_url())

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        attempt = 0
        last_error: BaseException | None = None
        while not self._closed:
            try:
                await self.open()
            except Exception as exc:
                last_error = exc
                if not _is_disconnect(exc) or attempt >= self.reconnect_attempts:
                    raise
                attempt += 1
                self.reconnects += 1
                await asyncio.sleep(self.reconnect_delay * attempt)
                continue
            try:
                async for raw in self._ws:
                    message = parse_ws_message(raw)
                    seq = _message_seq(message)
                    self.after = max(self.after, seq)
                    if message.get("type") == "job.end":
                        self.done = True
                    if message.get("type") == "event" and seq > 0:
                        attempt = 0  # 有真实事件推进才算连接健康，重置退避
                    yield message
                    if self.done:
                        return
            except Exception as exc:
                await self._drop()
                last_error = exc
                if not _is_disconnect(exc) or self._closed:
                    raise
            else:
                await self._drop()

            if self.done or self._closed:
                return
            if attempt >= self.reconnect_attempts:
                if last_error is not None:
                    raise last_error
                raise ConnectionError("WebSocket closed before job.end")
            attempt += 1
            self.reconnects += 1
            await asyncio.sleep(self.reconnect_delay * attempt)

    async def send(self, payload: dict[str, Any]) -> None:
        await self.open()
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def approve(self, approval_id: str, approved: bool) -> None:
        await self.send(
            {"type": "approve", "approval_id": approval_id, "approved": bool(approved)}
        )

    async def cancel(self) -> None:
        await self.send({"type": "cancel"})

    async def ping(self) -> None:
        await self.send({"type": "ping"})

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001, S110 - 清理失败不影响重连
                pass

    async def close(self) -> None:
        self._closed = True
        self.done = True
        await self._drop()


class BackendClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        http: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        ws_connect: WsConnect | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http
        self._owns_http = http is None
        self.timeout = timeout
        self._ws_connect = ws_connect

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def create_run(self, task: str, run_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"task": task}
        if run_id:
            payload["run_id"] = run_id
        else:
            payload["run_id"] = f"ui-{uuid.uuid4().hex[:10]}"
        response = await self.http.post(f"{self.base_url}/v1/runs", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        response = await self.http.get(f"{self.base_url}/v1/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        response = await self.http.get(
            f"{self.base_url}/v1/runs", params={"limit": max(1, min(limit, 100))}
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict[str, Any]:
        response = await self.http.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def session(self, run_id: str) -> RunSession:
        return RunSession(run_id, ws_url(self.base_url, run_id), connect=self._ws_connect)


async def stream_state(
    session: RunSession, state: UiState
) -> AsyncIterator[UiState]:
    """消费一条 WS 会话，逐条归约状态；UI 与测试共用这条路径。"""
    async for message in session.messages():
        apply_message(state, message)
        state.reconnects = session.reconnects
        yield state


# --------------------------------------------------------------------------- #
# 4. Gradio UI（仅在此处引入 gradio）
# --------------------------------------------------------------------------- #
def _import_gradio() -> Any:
    try:
        import gradio
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise RuntimeError(
            "gradio 未安装：uv sync --extra dev --extra frontend"
        ) from exc
    return gradio


def _make_chatbot(gr: Any, **kwargs: Any) -> Any:
    """gradio 4/5 的 Chatbot 需要 type="messages"；gradio 6 已默认 messages 且移除该参数。"""
    import inspect

    if "type" in inspect.signature(gr.Chatbot).parameters:
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


EXAMPLE_TASKS = [
    ["只回复：hello"],
    ["读取 README 并总结这个项目"],
    ["列出当前工作区文件，然后告诉我最大的文件"],
]


def _view(gr: Any, state: UiState, session: RunSession | None) -> tuple[Any, ...]:
    approval_md, approval_id = render_approval(state)
    pending = bool(approval_id)
    return (
        render_chat(state),
        render_status(state),
        render_trace(state),
        approval_md,
        approval_id,
        gr.update(interactive=pending),
        gr.update(interactive=pending),
        state,
        session,
    )


def build_ui(base_url: str = "http://127.0.0.1:8000", *, client: BackendClient | None = None):
    """构造 Gradio Blocks；不启动服务器，便于测试 import 与手工 launch。"""
    gr = _import_gradio()
    client = client or BackendClient(base_url)

    async def start(task_text: str, run_id: str, state: UiState):
        state = UiState()
        task = (task_text or "").strip()
        if not task:
            state.status = "failed"
            state.error = "task 不能为空"
            yield _view(gr, state, None)
            return

        state.messages.append({"role": "user", "content": task})
        yield _view(gr, state, None)

        try:
            job = await client.create_run(task, (run_id or "").strip() or None)
        except Exception as exc:  # noqa: BLE001 - 展示给用户
            state.status = "failed"
            state.error = describe_error(exc)
            yield _view(gr, state, None)
            return

        state.run_id = str(job.get("run_id") or "")
        state.status = str(job.get("status") or "queued")
        session = client.session(state.run_id)
        try:
            await session.open()
        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            state.error = f"WebSocket 连接失败：{describe_error(exc)}"
            await session.close()
            yield _view(gr, state, None)
            return

        yield _view(gr, state, session)
        try:
            async for current in stream_state(session, state):
                yield _view(gr, current, session)
        except Exception as exc:  # noqa: BLE001
            state.error = describe_error(exc)
        finally:
            await session.close()
        yield _view(gr, state, None)

    async def decide(approval_id: str, approved: bool, state: UiState, session: RunSession | None):
        if session is None or not approval_id:
            yield _view(gr, state, session)
            return
        try:
            await session.approve(approval_id, approved)
        except Exception as exc:  # noqa: BLE001
            state.error = f"审批发送失败：{describe_error(exc)}"
            yield _view(gr, state, session)
            return
        state.pending.pop(approval_id, None)
        if not state.pending and state.status == "waiting_approval":
            state.status = "running"
        yield _view(gr, state, session)

    async def approve(approval_id: str, state: UiState, session: RunSession | None):
        async for view in decide(approval_id, True, state, session):
            yield view

    async def deny(approval_id: str, state: UiState, session: RunSession | None):
        async for view in decide(approval_id, False, state, session):
            yield view

    async def cancel(state: UiState, session: RunSession | None):
        if session is None:
            yield _view(gr, state, session)
            return
        try:
            await session.cancel()
            state.status = "cancelling"
        except Exception as exc:  # noqa: BLE001
            state.error = f"取消失败：{describe_error(exc)}"
        yield _view(gr, state, session)

    async def clear(state: UiState, session: RunSession | None):
        if session is not None:
            await session.close()
        yield _view(gr, UiState(), None)

    with gr.Blocks(title="Evolving Harness") as demo:
        gr.Markdown(
            "# Evolving Harness\n"
            f"后端：`{client.base_url}` · 运行任务、查看 trace、审批有副作用的工具调用。\n\n"
            "断线后会自动按事件序号续传；审批/取消复用同一条 WebSocket。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                task = gr.Textbox(
                    label="任务",
                    lines=3,
                    placeholder="例如：读取 README 并总结这个项目",
                )
                gr.Examples(examples=EXAMPLE_TASKS, inputs=task, label="示例任务")
                run_id_box = gr.Textbox(label="run_id（留空自动生成）", max_lines=1)
                with gr.Row():
                    run_btn = gr.Button("运行", variant="primary")
                    cancel_btn = gr.Button("取消")
                    clear_btn = gr.Button("清空")
                chat = _make_chatbot(gr, label="对话与工具轨迹", height=460)
            with gr.Column(scale=2):
                status = gr.Markdown("_空闲_")
                approval = gr.Markdown("_没有待批准的工具调用_")
                approval_id = gr.Textbox(label="待批准 id", interactive=False, max_lines=1)
                with gr.Row():
                    approve_btn = gr.Button(
                        "批准", variant="primary", interactive=False
                    )
                    deny_btn = gr.Button("拒绝", variant="stop", interactive=False)
                with gr.Accordion("事件 trace（最近 25 条）", open=False):
                    trace = gr.Markdown("_暂无事件_")

        ui_state = gr.State(UiState())
        session_state = gr.State(None)
        outputs = [
            chat,
            status,
            trace,
            approval,
            approval_id,
            approve_btn,
            deny_btn,
            ui_state,
            session_state,
        ]

        run_btn.click(start, inputs=[task, run_id_box, ui_state], outputs=outputs)
        approve_btn.click(
            approve, inputs=[approval_id, ui_state, session_state], outputs=outputs
        )
        deny_btn.click(
            deny, inputs=[approval_id, ui_state, session_state], outputs=outputs
        )
        cancel_btn.click(cancel, inputs=[ui_state, session_state], outputs=outputs)
        clear_btn.click(clear, inputs=[ui_state, session_state], outputs=outputs)

    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evolving Harness Gradio 前端")
    parser.add_argument(
        "--backend",
        default=os.getenv("EH_BACKEND_URL", "http://127.0.0.1:8000"),
        help="后端地址（默认 EH_BACKEND_URL 或 http://127.0.0.1:8000）",
    )
    parser.add_argument("--host", default=os.getenv("EH_FRONTEND_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("EH_FRONTEND_PORT", "7860"))
    )
    parser.add_argument("--share", action="store_true", help="使用 Gradio 公网分享链接")
    args = parser.parse_args(argv)

    demo = build_ui(args.backend)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
