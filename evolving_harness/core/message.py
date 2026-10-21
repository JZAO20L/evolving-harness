from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> ToolCall:
        fn = raw.get("function") or {}
        args: Any = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"__raw__": args}
        if not isinstance(args, dict):
            args = {"__value__": args}
        return cls(id=raw.get("id") or "", name=fn.get("name") or "", arguments=args)


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> Message:
        content = raw.get("content") or ""
        if not isinstance(content, str):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return cls(
            role=raw.get("role", "assistant"),
            content=content,
            tool_calls=[ToolCall.from_openai(tc) for tc in raw.get("tool_calls") or []],
            tool_call_id=raw.get("tool_call_id"),
        )
