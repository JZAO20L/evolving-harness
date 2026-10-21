from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..models.base import ChatModel, Usage
from ..tools.base import ToolRegistry
from ..tracing.recorder import TraceRecorder
from .message import Message


class MaxStepsExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class RunResult:
    run_id: str
    final: str
    messages: list[Message]
    steps: int
    usage: Usage
    trace_path: Path


class AgentLoop:
    """One loop: model -> tool calls -> tool results -> model."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry | None = None,
        system_prompt: str = "",
        max_steps: int = 20,
        trace_dir: str | Path = "traces",
    ) -> None:
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.trace_dir = trace_dir

    async def run(self, task: str, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        tracer = TraceRecorder(run_id, self.trace_dir)
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=task))

        usage = Usage()
        step = 0
        tracer.event(
            "run.start",
            task=task,
            model=getattr(self.model, "name", "unknown"),
            tools=self.tools.names(),
            max_steps=self.max_steps,
        )

        try:
            for step in range(1, self.max_steps + 1):
                specs = self.tools.specs() or None
                tracer.event("model.request", step=step, message_count=len(messages))
                started = time.perf_counter()
                response = await self.model.complete(messages, tools=specs)
                usage.add(response.usage)
                tracer.event(
                    "model.response",
                    step=step,
                    duration_ms=_ms(started),
                    content=response.content,
                    tool_calls=[call.to_openai() for call in response.tool_calls],
                    usage=response.usage.to_dict(),
                    finish_reason=response.finish_reason,
                )

                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                if not response.tool_calls:
                    tracer.event(
                        "run.end", status="ok", steps=step, final=response.content,
                        usage=usage.to_dict(),
                    )
                    return RunResult(
                        run_id=run_id,
                        final=response.content,
                        messages=messages,
                        steps=step,
                        usage=usage,
                        trace_path=tracer.path,
                    )

                for call in response.tool_calls:
                    tracer.event(
                        "tool.call", step=step, tool_call_id=call.id,
                        name=call.name, arguments=call.arguments,
                    )
                    started = time.perf_counter()
                    result = await self.tools.dispatch(call)
                    tracer.event(
                        "tool.result",
                        step=step,
                        tool_call_id=call.id,
                        name=call.name,
                        duration_ms=_ms(started),
                        content=result.content,
                        is_error=result.content.startswith("ERROR:"),
                    )
                    messages.append(result)

            tracer.event(
                "run.end", status="max_steps", steps=self.max_steps, usage=usage.to_dict()
            )
            raise MaxStepsExceeded(f"agent exceeded {self.max_steps} steps")
        except MaxStepsExceeded:
            raise
        except Exception as exc:
            tracer.event(
                "run.end", status="error", step=step, error=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            tracer.close()


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
