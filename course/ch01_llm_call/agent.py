"""course/ch01 · llm-call —— 一次 LLM 调用的最小实现。

本课只做一件事：把"一段消息发出去，拿回一段文本 + 用量"。不引入 loop / act /
function calling（那是 ch02 起的事）。用最薄的 httpx 直接打 ``/chat/completions``，
零额外依赖，方便读、测、改。

涉及网络的适配器叫 ``OpenAIChatModel``；离线自测用 ``MockChatModel``。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import httpx


# --------------------------------------------------------------------------- #
# 消息协议：循环里唯一传播的结构（ch01 先只存 role + content）
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str  # system | user | assistant | （tool 留到 ch03）
    content: str


# --------------------------------------------------------------------------- #
# 返回结构与用量记账
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelResult:
    content: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


class ModelError(Exception):
    """模型端点不可达、异常返回、或解析失败（重试耗尽后抛出）。"""


# 这些状态码意味着"端点暂时不可用/被限流"，值得退避重试；业务性 4xx 不重试。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class _Retryable(Exception):
    """内部信号：本次请求失败但值得重试。"""


# --------------------------------------------------------------------------- #
# 抽象接口
# --------------------------------------------------------------------------- #
class ChatModel:
    """一切模型适配器实现``：``建实例后，循环只调 ``complete`` 与 ``aclose``。"""

    async def complete(self, messages: list[Message]) -> ModelResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 真实适配器：chat/completions（httpx，零依赖）
# --------------------------------------------------------------------------- #
class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.2,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url 必填")
        if not model:
            raise ValueError("model 必填")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        if http_client is not None:
            self._http = http_client
        else:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            self._http = httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout, headers=headers
            )

    async def complete(self, messages: list[Message]) -> ModelResult:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        # 重试：可重试状态(408/429/5xx)或传输错误 -> 指数退避 + 抖动；重试耗尽抛 ModelError。
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    raise _Retryable(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                if response.status_code != 200:
                    raise ModelError(f"端点返回 {response.status_code}: {response.text[:200]}")
                return self._complete_ok(response.json())
            except (httpx.TransportError, _Retryable) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    backoff = min(8.0, 0.5 * (2**attempt)) + random.uniform(0, 0.2)
                    await asyncio.sleep(backoff)
        raise ModelError(f"模型请求重试后仍失败: {last_error}")

    def _complete_ok(self, data: dict) -> ModelResult:
        try:
            choice = data["choices"][0]
            usage = data.get("usage", {})
        except (KeyError, IndexError) as exc:
            raise ModelError(f"响应结构异常: {data}") from exc
        content = (choice.get("message") or {}).get("content") or ""
        return ModelResult(
            content=content,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._http.aclose()


# --------------------------------------------------------------------------- #
# 离线假实现：测试与"污染端点前先跑通"用
# --------------------------------------------------------------------------- #
class MockChatModel(ChatModel):
    """脚本化响应；不联网。

    - 给了 ``script``：逐个弹出预设的 ``ModelResult``（出空后抛 StopIteration 语义错误）。
    - 没给：把最后一条 user 内容回显为 ``mock:<content>``，用量固定。
    """

    def __init__(self, script: list[ModelResult] | None = None) -> None:
        self.script = list(script or [])

    async def complete(self, messages: list[Message]) -> ModelResult:
        if self.script:
            return self.script.pop(0)
        last = messages[-1].content if messages else ""
        return ModelResult(content=f"mock:{last}", usage=Usage(10, 5), finish_reason="stop")

    async def aclose(self) -> None:
        pass