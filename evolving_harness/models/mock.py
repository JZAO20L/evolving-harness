from __future__ import annotations

from typing import Any

from ..core.message import Message
from .base import ChatModel, ModelResponse, Usage


class MockChatModel(ChatModel):
    """Deterministic model for tests and offline demos.

    Pass `responses` to script a conversation; otherwise echoes the last message.
    """

    name = "mock"

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        last = messages[-1].content if messages else ""
        return ModelResponse(
            content=f"MOCK: {last}",
            usage=Usage(prompt_tokens=1, completion_tokens=1),
            model=self.name,
        )
