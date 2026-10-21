from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from ..core.message import Message, ToolCall
from .base import ChatModel, ModelError, ModelResponse, Usage

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class OpenAIChatModel(ChatModel):
    """Any OpenAI-compatible /chat/completions endpoint: vLLM, DeepSeek, Qwen, ..."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.2,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        if not model:
            raise ValueError("model id is required")
        if not base_url:
            raise ValueError("base_url is required")
        self.name = model
        self.temperature = temperature
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [message.to_openai() for message in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    raise _Retryable(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                return self._parse(response.json())
            except (httpx.TransportError, _Retryable) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    backoff = min(8.0, 0.5 * (2**attempt)) + random.uniform(0, 0.2)
                    await asyncio.sleep(backoff)

        raise ModelError(f"model request failed after retries: {last_error}")

    def _parse(self, data: dict[str, Any]) -> ModelResponse:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise ModelError(f"malformed response: {data}") from exc

        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        raw_usage = data.get("usage") or {}
        return ModelResponse(
            content=content,
            tool_calls=[ToolCall.from_openai(tc) for tc in message.get("tool_calls") or []],
            usage=Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            ),
            model=data.get("model") or self.name,
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class _Retryable(Exception):
    pass
