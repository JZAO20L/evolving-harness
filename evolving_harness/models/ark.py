"""ArkChatModel: ARK (Volcengine Volcano Engine) via the OpenAI SDK Responses API.

Talks to ARK's ``/responses`` endpoint (``client.responses.create``), the same
surface documented in docs/models/ark_text_skill.md. Text-only this round; the
``input`` content-blocks can grow ``input_image`` items later for multimodal.

Uses the official ``openai`` SDK. The import is lazy so the rest of the package
imports and tests run even when ``openai`` is not installed; only constructing
or using ArkChatModel needs it.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.message import Message, ToolCall
from .base import ChatModel, ModelError, ModelResponse, Usage

VALID_KEYS = ("name", "description", "parameters")


def _build_input(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split a chat-completions-style history into Responses ``instructions`` + ``input`` items."""
    instructions = "\n\n".join(m.content for m in messages if m.role == "system")
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.role
        if role == "system":
            continue
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": message.content,
                }
            )
        elif role == "assistant":
            if message.content:
                items.append({"role": "assistant", "content": message.content})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    }
                )
        else:  # user
            items.append({"role": "user", "content": message.content})
    return instructions, items


def specs_to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Flatten our chat-style function specs into Responses ``tools`` items."""
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for spec in tools:
        fn = spec.get("function", spec) or {}
        out.append(
            {"type": "function", **{key: fn[key] for key in VALID_KEYS if key in fn}}
        )
    return out


def _parse_output(response: Any) -> tuple[str, list[ToolCall], Usage, str | None]:
    """Turn a Responses ``output`` list into content + tool calls + usage."""
    content: list[str] = []
    calls: list[ToolCall] = []
    for item in getattr(response, "output", None) or []:
        kind = getattr(item, "type", None)
        if kind == "message":
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) in ("output_text", "text"):
                    content.append(getattr(block, "text", "") or "")
        elif kind == "function_call":
            calls.append(
                ToolCall(
                    id=getattr(item, "call_id", "") or "",
                    name=getattr(item, "name", "") or "",
                    arguments=_json_args(getattr(item, "arguments", "{}")),
                )
            )
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "input_tokens", 0) if usage else 0
    completion = getattr(usage, "output_tokens", 0) if usage else 0
    return "".join(content), calls, Usage(prompt, completion), getattr(response, "status", None)


def _json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "{}")
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        return {"__raw__": text}
    return parsed if isinstance(parsed, dict) else {"__value__": parsed}


class ArkChatModel(ChatModel):
    """ARK Responses API adapter, text-only. OpenAI-SDK-backed; same base as openai_compat."""

    name: str = "ark"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model id is required")
        if not base_url:
            raise ValueError("base_url is required")
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client = http_client
        self._cli: Any = None

    def _client(self) -> Any:
        if self._cli is None:
            try:
                import openai
            except ImportError as exc:
                raise ModelError(
                    "openai SDK required for ArkChatModel; run `uv sync` after adding it"
                ) from exc
            self._cli = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=self.timeout,
                http_client=self._http_client,
            )
        return self._cli

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ModelResponse:
        instructions, items = _build_input(messages)
        kwargs: dict[str, Any] = {
            "model": self.name,
            "input": items,
            "temperature": self.temperature,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if self.max_tokens:
            kwargs["max_output_tokens"] = self.max_tokens
        tool_defs = specs_to_tools(tools)
        if tool_defs:
            kwargs["tools"] = tool_defs
        try:
            response = await self._client().responses.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"ARK response failed: {exc}") from exc
        content, calls, usage, reason = _parse_output(response)
        return ModelResponse(
            content=content,
            tool_calls=calls,
            usage=usage,
            model=self.name,
            finish_reason=reason,
        )

    async def aclose(self) -> None:
        if self._cli is not None:
            await self._cli.close()