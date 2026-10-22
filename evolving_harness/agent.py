"""One agent: an agent.yaml state plus the loop that runs it.

The YAML is the mutable state. This file is the frozen runtime.
RSI (rsi.py) edits YAML, never this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .core.message import Message
from .models.base import ChatModel, Usage
from .models.mock import MockChatModel
from .models.openai_compat import OpenAIChatModel
from .tools.base import Tool, ToolRegistry
from .tools.fs import fs_tools
from .tracing.recorder import TraceRecorder

DEFAULT_TOOLS = ("read_file", "write_file", "list_dir")
SKILL_TOOL = "load_skill"
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class AgentConfigError(ValueError):
    pass


class MaxStepsExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str


@dataclass(frozen=True, slots=True)
class AgentState:
    path: Path
    version: int
    name: str
    system_prompt: str
    guidance: str
    skills: tuple[Skill, ...]
    max_steps: int
    tool_names: tuple[str, ...]
    model_config: dict[str, Any]
    raw: dict[str, Any]
    digest: str

    @classmethod
    def load(cls, path: str | Path) -> AgentState:
        config_path = Path(path).resolve()
        if not config_path.is_file():
            raise AgentConfigError(f"agent config not found: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise AgentConfigError(f"invalid YAML in {config_path}: {exc}") from None
        if not isinstance(raw, dict):
            raise AgentConfigError(f"agent config must be a mapping: {config_path}")

        version = raw.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise AgentConfigError("version must be a positive integer")

        name = raw.get("name") or config_path.parent.name
        if not isinstance(name, str) or not name.strip():
            raise AgentConfigError("name must be a non-empty string")

        prompt = _mapping(raw, "prompt")
        system_prompt = prompt.get("system")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise AgentConfigError("prompt.system must be a non-empty string")
        guidance = prompt.get("guidance", "")
        if not isinstance(guidance, str):
            raise AgentConfigError("prompt.guidance must be a string")

        skills = _parse_skills(raw.get("skills") or [])

        tools = _mapping(raw, "tools")
        enabled = tools.get("enabled") or list(DEFAULT_TOOLS)
        if not isinstance(enabled, list) or not enabled:
            raise AgentConfigError("tools.enabled must be a non-empty list")
        if not all(isinstance(item, str) for item in enabled):
            raise AgentConfigError("tools.enabled must contain strings")
        if SKILL_TOOL in enabled:
            raise AgentConfigError(f"{SKILL_TOOL} is added automatically; do not list it")
        if len(set(enabled)) != len(enabled):
            raise AgentConfigError("tools.enabled contains duplicates")

        limits = _mapping(raw, "limits")
        max_steps = limits.get("max_steps", 20)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise AgentConfigError("limits.max_steps must be an integer")
        if not 1 <= max_steps <= 100:
            raise AgentConfigError("limits.max_steps must be between 1 and 100")

        model_config = _mapping(raw, "model")

        snapshot = json.loads(json.dumps(raw, default=str))
        canonical = json.dumps(
            snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        return cls(
            path=config_path,
            version=version,
            name=name,
            system_prompt=system_prompt,
            guidance=guidance,
            skills=tuple(skills),
            max_steps=max_steps,
            tool_names=tuple(enabled),
            model_config=model_config,
            raw=snapshot,
            digest=digest,
        )

    @property
    def hash(self) -> str:
        return self.digest

    def skill(self, name: str) -> Skill | None:
        return next((item for item in self.skills if item.name == name), None)

    def render_system_prompt(self) -> str:
        parts = [self.system_prompt.strip()]
        if self.guidance.strip():
            parts.append("## Working guidance\n" + self.guidance.strip())
        if self.skills:
            lines = [
                "## Skills",
                f"Load full instructions with {SKILL_TOOL}(name) before using a skill.",
            ]
            lines.extend(f"- {item.name}: {item.description}" for item in self.skills)
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def runtime(self) -> dict[str, Any]:
        config = self.model_config
        raw_temperature = config.get("temperature", 0.2)
        if isinstance(raw_temperature, bool):
            raise AgentConfigError("model.temperature must be a number")
        try:
            temperature = float(raw_temperature)
        except (TypeError, ValueError):
            raise AgentConfigError("model.temperature must be a number") from None

        max_tokens = config.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise AgentConfigError("model.max_tokens must be a positive integer")

        return {
            "provider": str(config.get("provider", "openai-compatible")),
            "id": _env(config.get("id")),
            "base_url": _env(config.get("base_url")),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


@dataclass(slots=True)
class RunResult:
    run_id: str
    final: str
    messages: list[Message]
    steps: int
    usage: Usage
    trace_path: Path
    status: str
    state_version: int
    state_hash: str


class Agent:
    """model -> tool calls -> tool results -> model, until a final answer."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        state: AgentState,
        trace_dir: str | Path = "traces",
    ) -> None:
        self.model = model
        self.tools = tools
        self.state = state
        self.trace_dir = trace_dir

    async def run(self, task: str, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        runtime = self.state.runtime()
        messages = [
            Message(role="system", content=self.state.render_system_prompt()),
            Message(role="user", content=task),
        ]
        usage = Usage()
        step = 0
        tracer = TraceRecorder(run_id, self.trace_dir)
        tracer.event(
            "run.start",
            task=task,
            model=getattr(self.model, "name", "unknown"),
            tools=self.tools.names(),
            max_steps=self.state.max_steps,
            state={
                "name": self.state.name,
                "version": self.state.version,
                "hash": self.state.hash,
                "path": str(self.state.path),
            },
            runtime=runtime,
        )

        try:
            for step in range(1, self.state.max_steps + 1):
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
                    after = self._state_hash()
                    status = "ok" if after == self.state.hash else "drift"
                    tracer.event(
                        "run.end",
                        status=status,
                        steps=step,
                        final=response.content,
                        usage=usage.to_dict(),
                        state_hash=after,
                    )
                    return RunResult(
                        run_id=run_id,
                        final=response.content,
                        messages=messages,
                        steps=step,
                        usage=usage,
                        trace_path=tracer.path,
                        status=status,
                        state_version=self.state.version,
                        state_hash=self.state.hash,
                    )

                for call in response.tool_calls:
                    tracer.event(
                        "tool.call",
                        step=step,
                        tool_call_id=call.id,
                        name=call.name,
                        arguments=call.arguments,
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
                "run.end",
                status="max_steps",
                steps=self.state.max_steps,
                usage=usage.to_dict(),
                state_hash=self._state_hash(),
            )
            raise MaxStepsExceeded(f"agent exceeded {self.state.max_steps} steps")
        except MaxStepsExceeded:
            raise
        except Exception as exc:
            tracer.event(
                "run.end",
                status="error",
                step=step,
                error=f"{type(exc).__name__}: {exc}",
                state_hash=self._state_hash(),
            )
            raise
        finally:
            tracer.close()

    def _state_hash(self) -> str:
        try:
            return AgentState.load(self.state.path).hash
        except AgentConfigError:
            return ""


def build_tools(state: AgentState, workspace: str | Path) -> ToolRegistry:
    available = {item.name: item for item in fs_tools(workspace)}
    selected: list[Tool] = []
    for name in state.tool_names:
        item = available.get(name)
        if item is None:
            raise AgentConfigError(
                f"unknown tool in tools.enabled: {name!r}; available: {sorted(available)}"
            )
        selected.append(item)

    if state.skills:
        selected.append(_skill_tool(state))
    return ToolRegistry(selected)


def create_model(
    state: AgentState,
    *,
    mock: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatModel:
    if mock:
        return MockChatModel()

    runtime = state.runtime()
    model_id = model or runtime["id"]
    url = base_url or runtime["base_url"]
    key = api_key if api_key is not None else os.getenv("EH_API_KEY", "")
    if not model_id:
        raise AgentConfigError("model id is required; set EH_MODEL or pass --model")
    if not url:
        raise AgentConfigError("base_url is required; set EH_BASE_URL or pass --base-url")
    try:
        return OpenAIChatModel(
            model=model_id,
            base_url=url,
            api_key=key,
            temperature=runtime["temperature"],
            max_tokens=runtime["max_tokens"],
        )
    except ValueError as exc:
        raise AgentConfigError(str(exc)) from None


async def run_agent(
    agent_path: str | Path,
    task: str,
    *,
    model: ChatModel | None = None,
    workspace: str | Path = "workspace",
    trace_dir: str | Path = "traces",
    run_id: str | None = None,
    mock: bool = False,
) -> RunResult:
    state = AgentState.load(agent_path)
    owns_model = model is None
    if model is None:
        model = create_model(state, mock=mock)
    agent = Agent(
        model=model,
        tools=build_tools(state, workspace),
        state=state,
        trace_dir=trace_dir,
    )
    try:
        return await agent.run(task, run_id=run_id)
    finally:
        if owns_model:
            await model.aclose()


def _parse_skills(items: Any) -> list[Skill]:
    if not isinstance(items, list):
        raise AgentConfigError("skills must be a list")
    skills: list[Skill] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise AgentConfigError(f"skills[{index}] must be a mapping")
        name = item.get("name")
        if not isinstance(name, str) or not SKILL_NAME.fullmatch(name):
            raise AgentConfigError(f"skills[{index}].name must match {SKILL_NAME.pattern}")
        if name in seen:
            raise AgentConfigError(f"duplicate skill name: {name}")
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise AgentConfigError(f"skills[{index}].description must be a non-empty string")
        instructions = item.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise AgentConfigError(f"skills[{index}].instructions must be a non-empty string")
        seen.add(name)
        skills.append(Skill(name=name, description=description, instructions=instructions))
    return skills


def _skill_tool(state: AgentState) -> Tool:
    def load_skill(name: str) -> str:
        item = state.skill(name)
        if item is None:
            available = ", ".join(skill.name for skill in state.skills)
            return f"ERROR: unknown skill {name!r}; available: {available}"
        return item.instructions

    return Tool(
        name=SKILL_TOOL,
        description="Load the full instructions for a skill listed in the system prompt.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name from the catalog."},
            },
            "required": ["name"],
        },
        handler=load_skill,
    )


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AgentConfigError(f"{key} must be a mapping")
    return value


def _env(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    expanded = os.path.expandvars(text)
    if expanded == text and text.startswith("$"):
        return ""
    return expanded


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
