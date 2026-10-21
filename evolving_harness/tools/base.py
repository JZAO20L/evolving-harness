from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..core.message import Message, ToolCall

ToolHandler = Callable[..., Awaitable[str] | str]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, **kwargs: Any) -> str:
        result = self.handler(**kwargs)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[assignment]
        return result if isinstance(result, str) else str(result)


def tool(name: str, description: str, parameters: dict[str, Any]):
    def decorator(handler: ToolHandler) -> Tool:
        return Tool(name=name, description=description, parameters=parameters, handler=handler)

    return decorator


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools or []:
            self.register(item)

    def register(self, item: Tool) -> None:
        if item.name in self._tools:
            raise ValueError(f"duplicate tool: {item.name}")
        self._tools[item.name] = item

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [item.spec() for item in self._tools.values()]

    async def dispatch(self, call: ToolCall) -> Message:
        item = self._tools.get(call.name)
        if item is None:
            return Message(
                role="tool",
                content=f"ERROR: unknown tool {call.name!r}; available: {self.names()}",
                tool_call_id=call.id,
            )
        try:
            output = await item.run(**call.arguments)
        except Exception as exc:  # noqa: BLE001 - tool failures are data, not crashes
            output = f"ERROR: {type(exc).__name__}: {exc}"
        return Message(role="tool", content=output, tool_call_id=call.id)
