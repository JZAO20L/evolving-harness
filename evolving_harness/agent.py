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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .core.message import Message, ToolCall
from .models.base import ChatModel, Usage
from .models.mock import MockChatModel
from .models.openai_compat import OpenAIChatModel
from .tools.base import Tool, ToolRegistry
from .tools.fs import fs_tools
from .tracing.recorder import TraceRecorder

DEFAULT_TOOLS = ("read_file", "write_file", "list_dir")
SKILL_TOOL = "load_skill"
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
DEFAULT_MAX_CONTEXT_TOKENS = 32000
DEFAULT_KEEP_RECENT_ROUNDS = 6
TOOL_HEAD_TAIL_CHARS = 500


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
class Goal:
    id: str
    statement: str
    priority: int


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    enabled: bool
    path: str
    max_tokens: int = 1024


@dataclass(slots=True)
class State:
    """Single state class. Two roles, one class.

    Config role (from ``load``): static fields populated, dynamic fields default.
    Run role (from ``begin_run`` / ``from_dict``): a run's dynamic fields layered on
    the static identity (name / version / hash). A run is ``config.begin_run(task)``.

    ``snapshot()`` serializes only the dynamic section plus the agent identity as
    provenance; checkpoints never carry the full static config. On resume the
    checkpoint's identity is validated against a freshly loaded config State, so a
    mismatched agent is rejected instead of producing irreproducible results.
    """

    # --- static / identity (the RSI-mutable YAML; hash = canonical JSON of it) ---
    path: Path = Path("")
    version: int = 0
    name: str = ""
    system_prompt: str = ""
    guidance: str = ""
    skills: tuple[Skill, ...] = ()
    goals: tuple[Goal, ...] = ()
    memory: MemoryPolicy = field(
        default_factory=lambda: MemoryPolicy(False, ".memory/notes.md", 1024)
    )
    max_steps: int = 0
    max_context_tokens: int = 0
    keep_recent_rounds: int = 0
    tool_names: tuple[str, ...] = ()
    model_config: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    # --- dynamic (one run; excluded from the identity hash; checkpointed) ---
    run_id: str | None = None
    task: str = ""
    status: str | None = None
    step: int = 0
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    pending: list[ToolCall] = field(default_factory=list)
    parent_run_id: str | None = None
    created_at: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> State:
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
        goals = _parse_goals(raw.get("goals") or [])
        memory = _parse_memory(raw.get("memory"))

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

        max_context_tokens = limits.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens < 1
        ):
            raise AgentConfigError("limits.max_context_tokens must be a positive integer")

        keep_recent_rounds = limits.get("keep_recent_rounds", DEFAULT_KEEP_RECENT_ROUNDS)
        if (
            isinstance(keep_recent_rounds, bool)
            or not isinstance(keep_recent_rounds, int)
            or keep_recent_rounds < 1
        ):
            raise AgentConfigError("limits.keep_recent_rounds must be a positive integer")

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
            goals=tuple(goals),
            memory=memory,
            max_steps=max_steps,
            max_context_tokens=max_context_tokens,
            keep_recent_rounds=keep_recent_rounds,
            tool_names=tuple(enabled),
            model_config=model_config,
            raw=snapshot,
            digest=digest,
        )

    def begin_run(
        self,
        task: str,
        *,
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> State:
        """Seed a run State from this config State: copy the static identity, set
        the run's dynamic fields and the initial system/user messages."""
        return replace(
            self,
            run_id=run_id or uuid.uuid4().hex[:12],
            task=task,
            status="running",
            step=0,
            messages=[
                Message(role="system", content=self.render_system_prompt()),
                Message(role="user", content=task),
            ],
            usage=Usage(),
            pending=[],
            parent_run_id=parent_run_id,
            created_at=time.time(),
        )

    def snapshot(self) -> dict[str, Any]:
        """Dynamic section only + agent identity as provenance for resume."""
        return {
            "run_id": self.run_id,
            "agent_name": self.name,
            "agent_version": self.version,
            "agent_hash": self.hash,
            "task": self.task,
            "status": self.status,
            "step": self.step,
            "messages": [msg.to_openai() for msg in self.messages],
            "usage": self.usage.to_dict(),
            "pending": [call.to_openai() for call in self.pending],
            "parent_run_id": self.parent_run_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> State:
        """Reconstruct a run State from a checkpoint (identity + dynamic fields).
        The full static config is not present here; a freshly loaded config State
        validates this one's identity before a resume."""
        usage = data.get("usage") or {}
        return cls(
            name=data.get("agent_name", ""),
            version=data.get("agent_version", 0),
            digest=data.get("agent_hash", ""),
            run_id=data["run_id"],
            task=data.get("task", ""),
            status=data.get("status", "running"),
            step=data.get("step", 0),
            messages=[Message.from_openai(msg) for msg in data.get("messages", [])],
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            pending=[ToolCall.from_openai(call) for call in data.get("pending", [])],
            parent_run_id=data.get("parent_run_id"),
            created_at=float(data.get("created_at", 0.0)),
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
        if self.goals:
            lines = ["## Goals"]
            lines.extend(
                f"- {goal.statement}"
                for goal in sorted(self.goals, key=lambda item: item.priority)
            )
            parts.append("\n".join(lines))
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


class Checkpointer:
    """Writes state snapshots so runs can resume after a restart."""

    def __init__(self, trace_dir: str | Path = "traces") -> None:
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.checkpoint.json"

    def save(self, state: State) -> None:
        if state.run_id is None:
            raise ValueError("cannot checkpoint a config-only State without a run_id")
        path = self.path_for(state.run_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state.snapshot(), ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)

    def load(self, run_id: str) -> State | None:
        path = self.path_for(run_id)
        if not path.is_file():
            return None
        try:
            return State.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError):
            return None


class MemoryStore:
    """Layered file memory under the workspace. A no-op when disabled.

    L0 ``raw/`` append-only JSONL event log, one file per run.
    L1 ``notes.md`` keyed markdown lines (upsert), the durable facts.
    L2 ``summary.md`` low-frequency model rewrite of L1, preferred at injection.
    """

    MAX_NOTES_LINES = 200

    def __init__(self, workspace: str | Path, policy: MemoryPolicy) -> None:
        self.policy = policy
        self.enabled = policy.enabled
        self.root = Path(workspace).resolve()
        notes_path = (self.root / policy.path).resolve()
        if not notes_path.is_relative_to(self.root):
            raise AgentConfigError("memory.path escapes the workspace")
        self.notes_path = notes_path
        self.summary_path = self.notes_path.parent / "summary.md"
        self.raw_dir = self.notes_path.parent / "raw"

    def record(self, kind: str, *, run_id: str, **data: Any) -> None:
        if not self.enabled:
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        entry = {"ts": round(time.time(), 6), "kind": kind, **data}
        with (self.raw_dir / f"{run_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def remember(self, key: str, note: str) -> None:
        if not self.enabled:
            return
        key = " ".join(str(key).split())
        note = " ".join(str(note).split())
        if not key or len(key) > 120:
            raise ValueError("memory key must be 1-120 characters")
        if not note:
            raise ValueError("memory note must not be empty")
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"- [{key}] "
        lines = []
        if self.notes_path.is_file():
            lines = [
                line
                for line in self.notes_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith(prefix)
            ]
        lines.append(prefix + note)
        self.notes_path.write_text(
            "\n".join(lines[-self.MAX_NOTES_LINES:]) + "\n", encoding="utf-8"
        )

    def notes_text(self) -> str:
        if not self.enabled or not self.notes_path.is_file():
            return ""
        return self.notes_path.read_text(encoding="utf-8")

    def write_summary(self, text: str) -> None:
        if not self.enabled:
            return
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(text.strip() + "\n", encoding="utf-8")

    def context(self) -> tuple[str, int]:
        """L2 summary first, then the most recent L1 notes, capped by max_tokens."""
        if not self.enabled:
            return "", 0
        budget = self.policy.max_tokens
        used = 0
        sections: list[str] = []

        summary = ""
        if self.summary_path.is_file():
            summary = self.summary_path.read_text(encoding="utf-8").strip()
        if summary:
            summary = summary[: max(0, budget * 4 - 20)]
            text = "### Summary\n" + summary
            used = _estimate_tokens(text)
            sections.append(text)

        selected: list[str] = []
        for line in reversed(self.notes_text().splitlines()):
            if not line.strip():
                continue
            tokens = _estimate_tokens(line)
            if used + tokens > budget:
                break
            selected.append(line)
            used += tokens
        if selected:
            sections.append("### Notes\n" + "\n".join(reversed(selected)))

        if not sections:
            return "", 0
        block = "## Memory\n" + "\n\n".join(sections)
        return block, _estimate_tokens(block)


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
        state: State,
        trace_dir: str | Path = "traces",
        workspace: str | Path = "workspace",
    ) -> None:
        self.model = model
        self.tools = tools
        self.state = state
        self.trace_dir = trace_dir
        self.memory = MemoryStore(workspace, state.memory)

    async def run(
        self,
        task: str,
        run_id: str | None = None,
        *,
        resume: bool = False,
    ) -> RunResult:
        checkpointer = Checkpointer(self.trace_dir)
        tracer: TraceRecorder
        state: State
        resumed = False

        if resume:
            if run_id is None:
                raise ValueError("resume requires run_id")
            saved = checkpointer.load(run_id)
            if saved is None:
                raise RuntimeError(f"no checkpoint for run {run_id}")
            if saved.name != self.state.name or saved.hash != self.state.hash:
                raise AgentConfigError("checkpoint agent does not match current YAML")
            state = saved
            resumed = True
            tracer = TraceRecorder(run_id, self.trace_dir)
            tracer.event("run.recover", from_step=state.step, agent_hash=state.hash)
            self.memory.record("run.recover", run_id=run_id, from_step=state.step)
            # Tool calls dispatched but never settled before a restart: keep the
            # intent, mark the outcome unknown, and never re-dispatch them.
            for call in state.pending:
                tracer.event(
                    "tool.call",
                    step=state.step,
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
                tracer.event(
                    "tool.result",
                    step=state.step,
                    tool_call_id=call.id,
                    name=call.name,
                    duration_ms=0,
                    content="ERROR: resumed before result; outcome unknown, not re-run",
                    is_error=True,
                )
                state.messages.append(
                    Message(
                        role="tool",
                        content="ERROR: resumed before result; outcome unknown, not re-run",
                        tool_call_id=call.id,
                    )
                )
            state.pending = []
            state.status = "running"
            checkpointer.save(state)
        else:
            state = self.state.begin_run(task=task, run_id=run_id)
            run_id = state.run_id
            tracer = TraceRecorder(run_id, self.trace_dir)
            runtime = self.state.runtime()
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
            if self.memory.enabled:
                block, memory_tokens = self.memory.context()
                if block:
                    state.messages[0].content += "\n\n" + block
                    tracer.event(
                        "memory.inject",
                        tokens=memory_tokens,
                        max_tokens=self.state.memory.max_tokens,
                    )
                self.memory.record("run.start", run_id=run_id, task=task)

        try:
            if resumed:
                tracer.event(
                    "model.request",
                    step=state.step,
                    message_count=len(state.messages),
                    est_tokens=_estimate_messages(state.messages),
                    message_tokens=[_estimate_message(msg) for msg in state.messages],
                    resume=True,
                )
            for step in range(state.step + 1, self.state.max_steps + 1):
                state.status = "running"
                state.step = step
                est_tokens, compacted = self._fit_context(state, step, tracer)
                if compacted:
                    checkpointer.save(state)
                specs = self.tools.specs() or None
                tracer.event(
                    "model.request",
                    step=step,
                    message_count=len(state.messages),
                    est_tokens=est_tokens,
                    message_tokens=[_estimate_message(msg) for msg in state.messages],
                    max_context_tokens=self.state.max_context_tokens,
                )
                started = time.perf_counter()
                response = await self.model.complete(state.messages, tools=specs)
                state.usage.add(response.usage)
                tracer.event(
                    "model.response",
                    step=step,
                    duration_ms=_ms(started),
                    content=response.content,
                    tool_calls=[call.to_openai() for call in response.tool_calls],
                    usage=response.usage.to_dict(),
                    finish_reason=response.finish_reason,
                )
                state.messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                if not response.tool_calls:
                    after = self._state_hash()
                    status = "ok" if after == self.state.hash else "drift"
                    state.status = status
                    if status == "ok":
                        await self._summarize_memory(state, step, tracer)
                    self.memory.record("run.end", run_id=run_id, status=status, steps=step)
                    checkpointer.save(state)
                    tracer.event(
                        "run.end",
                        status=status,
                        steps=step,
                        final=response.content,
                        usage=state.usage.to_dict(),
                        state_hash=after,
                    )
                    return RunResult(
                        run_id=run_id,
                        final=response.content,
                        messages=state.messages,
                        steps=step,
                        usage=state.usage,
                        trace_path=tracer.path,
                        status=status,
                        state_version=self.state.version,
                        state_hash=self.state.hash,
                    )

                for call in response.tool_calls:
                    state.status = "waiting_tool"
                    state.pending.append(call)
                    tracer.event(
                        "tool.call",
                        step=step,
                        tool_call_id=call.id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                    checkpointer.save(state)
                    started = time.perf_counter()
                    result = await self.tools.dispatch(call)
                    state.pending.remove(call)
                    tracer.event(
                        "tool.result",
                        step=step,
                        tool_call_id=call.id,
                        name=call.name,
                        duration_ms=_ms(started),
                        content=result.content,
                        is_error=result.content.startswith("ERROR:"),
                    )
                    state.messages.append(result)
                    self._note_tool_outcome(state, call, result.content, step, tracer)
                    checkpointer.save(state)

            state.status = "max_steps"
            self.memory.record(
                "run.end", run_id=run_id, status="max_steps", steps=self.state.max_steps
            )
            checkpointer.save(state)
            tracer.event(
                "run.end",
                status="max_steps",
                steps=self.state.max_steps,
                usage=state.usage.to_dict(),
                state_hash=self._state_hash(),
            )
            raise MaxStepsExceeded(f"agent exceeded {self.state.max_steps} steps")
        except MaxStepsExceeded:
            raise
        except Exception as exc:
            state.status = "error"
            self.memory.record(
                "run.end",
                run_id=run_id,
                status="error",
                step=state.step,
                error=f"{type(exc).__name__}: {exc}",
            )
            tracer.event(
                "run.end",
                status="error",
                step=state.step,
                error=f"{type(exc).__name__}: {exc}",
                state_hash=self._state_hash(),
            )
            raise
        finally:
            tracer.close()

    async def _summarize_memory(
        self, state: State, step: int, tracer: TraceRecorder
    ) -> None:
        """L2: rewrite oversized L1 notes into summary.md at low frequency."""
        if not self.memory.enabled:
            return
        notes = self.memory.notes_text()
        before = _estimate_tokens(notes)
        if before <= self.state.memory.max_tokens:
            return
        if (
            self.memory.summary_path.is_file()
            and self.memory.notes_path.is_file()
            and self.memory.summary_path.stat().st_mtime
            >= self.memory.notes_path.stat().st_mtime
        ):
            return
        response = await self.model.complete(
            [
                Message(
                    role="system",
                    content=(
                        "Rewrite the project memory notes as durable, deduplicated facts. "
                        "Drop ephemeral details and prefer short bullets."
                    ),
                ),
                Message(role="user", content=notes),
            ]
        )
        state.usage.add(response.usage)
        if response.content.strip():
            self.memory.write_summary(response.content)
        tracer.event(
            "memory.summarize",
            step=step,
            before_tokens=before,
            after_tokens=_estimate_tokens(response.content),
            usage=response.usage.to_dict(),
        )

    def _note_tool_outcome(
        self,
        state: State,
        call: ToolCall,
        content: str,
        step: int,
        tracer: TraceRecorder,
    ) -> None:
        """L0 event log + L1 rule-based notes for writes and tool errors."""
        if not self.memory.enabled:
            return
        is_error = content.startswith("ERROR:")
        self.memory.record(
            "tool.call",
            run_id=state.run_id,
            step=step,
            tool_call_id=call.id,
            name=call.name,
            arguments=call.arguments,
        )
        self.memory.record(
            "tool.result",
            run_id=state.run_id,
            step=step,
            tool_call_id=call.id,
            name=call.name,
            is_error=is_error,
            content=content[:500],
        )
        if call.name == "remember":
            if not is_error:
                tracer.event("memory.write", step=step, key=call.arguments.get("key"))
            return
        path = call.arguments.get("path") or call.arguments.get("name")
        if is_error:
            key = f"error:{call.name}:{path or call.id}"
            note = f"step {step}: {content[:200]}"
        elif call.name == "write_file" and path:
            key = f"write:{path}"
            note = f"step {step}: wrote {path}"
        else:
            return
        self.memory.remember(key, note)
        tracer.event("memory.write", step=step, key=key)

    def _fit_context(
        self, state: State, step: int, tracer: TraceRecorder
    ) -> tuple[int, bool]:
        """Estimate tokens and compact until the run fits ``max_context_tokens``.

        Recent rounds stay verbatim. Older tool results are first rewritten to a
        head/tail window; if that is not enough, the oldest whole rounds are dropped.
        """
        limit = self.state.max_context_tokens
        before = _estimate_messages(state.messages)
        if before <= limit:
            return before, False

        prefix_end, blocks = _turns(state.messages)
        if not blocks:
            return before, False

        names = {
            call.id: call.name
            for message in state.messages
            for call in message.tool_calls
        }
        keep = self.state.keep_recent_rounds
        middle = blocks[:-keep]
        recent = blocks[-keep:]

        current = before
        truncated = 0
        for start, end in middle + recent:
            if current <= limit:
                break
            for index in range(start, end):
                if current <= limit:
                    break
                message = state.messages[index]
                if message.role != "tool":
                    continue
                replacement = _summarize_tool_result(
                    message.content, names.get(message.tool_call_id or "", "tool")
                )
                if replacement is None:
                    continue
                current += _estimate_tokens(replacement) - _estimate_tokens(message.content)
                message.content = replacement
                truncated += 1

        dropped = 0
        if current > limit and middle:
            while dropped < len(middle):
                start, end = middle[dropped]
                current -= sum(_estimate_message(item) for item in state.messages[start:end])
                dropped += 1
                if current <= limit:
                    break
            if dropped:
                next_start = blocks[dropped][0]
                notice = Message(
                    role="assistant",
                    content=f"[context compacted: {dropped} earlier rounds omitted]",
                )
                state.messages = (
                    state.messages[:prefix_end] + [notice] + state.messages[next_start:]
                )
                current = _estimate_messages(state.messages)

        if not truncated and not dropped:
            return before, False

        tracer.event(
            "context.compact",
            step=step,
            before_tokens=before,
            after_tokens=current,
            truncated_tool_results=truncated,
            dropped_rounds=dropped,
            max_context_tokens=limit,
        )
        return current, True

    def _state_hash(self) -> str:
        try:
            return State.load(self.state.path).hash
        except AgentConfigError:
            return ""


def build_tools(state: State, workspace: str | Path) -> ToolRegistry:
    root = Path(workspace)
    protected: tuple[str, ...] = ()
    if state.memory.enabled:
        memory_dir = Path(state.memory.path).parent
        if str(memory_dir) not in ("", "."):
            protected = (str(memory_dir),)

    available = {item.name: item for item in fs_tools(root, protected=protected)}
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
    if state.memory.enabled:
        selected.append(_memory_tool(MemoryStore(root, state.memory)))
    return ToolRegistry(selected)


def create_model(
    state: State,
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
    state = State.load(agent_path)
    owns_model = model is None
    if model is None:
        model = create_model(state, mock=mock)
    agent = Agent(
        model=model,
        tools=build_tools(state, workspace),
        state=state,
        trace_dir=trace_dir,
        workspace=workspace,
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


def _skill_tool(state: State) -> Tool:
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


def _memory_tool(memory: MemoryStore) -> Tool:
    def remember(key: str, note: str) -> str:
        try:
            memory.remember(key, note)
        except ValueError as exc:
            return f"ERROR: {exc}"
        return f"remembered {key}"

    return Tool(
        name="remember",
        description="Store a durable fact for future runs under a stable key.",
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Stable key; overwrites the previous note for this key.",
                },
                "note": {"type": "string", "description": "One-line fact to remember."},
            },
            "required": ["key", "note"],
        },
        handler=remember,
    )


def _parse_goals(items: Any) -> list[Goal]:
    if not isinstance(items, list):
        raise AgentConfigError("goals must be a list")
    goals: list[Goal] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise AgentConfigError(f"goals[{index}] must be a mapping")
        goal_id = item.get("id")
        if not isinstance(goal_id, str) or not SKILL_NAME.fullmatch(goal_id):
            raise AgentConfigError(f"goals[{index}].id must match {SKILL_NAME.pattern}")
        if goal_id in seen:
            raise AgentConfigError(f"duplicate goal id: {goal_id}")
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise AgentConfigError(f"goals[{index}].statement must be a non-empty string")
        priority = item.get("priority", 1)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise AgentConfigError(f"goals[{index}].priority must be an integer")
        seen.add(goal_id)
        goals.append(Goal(id=goal_id, statement=statement, priority=priority))
    return goals


def _parse_memory(item: Any) -> MemoryPolicy:
    if item is None:
        return MemoryPolicy(enabled=False, path=".memory/notes.md", max_tokens=1024)
    if not isinstance(item, dict):
        raise AgentConfigError("memory must be a mapping")
    enabled = item.get("enabled", False)
    if isinstance(enabled, bool) is False:
        raise AgentConfigError("memory.enabled must be a boolean")
    path = item.get("path", ".memory/notes.md")
    if not isinstance(path, str) or not path.strip():
        raise AgentConfigError("memory.path must be a non-empty string")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise AgentConfigError("memory.path must stay inside the workspace")
    max_tokens = item.get("max_tokens", 1024)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise AgentConfigError("memory.max_tokens must be a positive integer")
    return MemoryPolicy(enabled=enabled, path=path, max_tokens=max_tokens)


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


def _estimate_tokens(text: str) -> int:
    """Rough budget estimate: ~4 characters per token."""
    return (len(text) + 3) // 4


def _estimate_message(message: Message) -> int:
    tokens = _estimate_tokens(message.content) + 4
    for call in message.tool_calls:
        tokens += _estimate_tokens(call.name)
        tokens += _estimate_tokens(json.dumps(call.arguments, ensure_ascii=False))
    return tokens


def _estimate_messages(messages: list[Message]) -> int:
    return sum(_estimate_message(message) for message in messages)


def _turns(messages: list[Message]) -> tuple[int, list[tuple[int, int]]]:
    """Split messages into (prefix_end, assistant-led blocks).

    The prefix holds the system prompt and the initial task; each block is one
    assistant message plus the tool results that answer it.
    """
    starts = [
        index for index, message in enumerate(messages) if message.role == "assistant"
    ]
    if not starts:
        return len(messages), []
    blocks = [
        (start, starts[position + 1] if position + 1 < len(starts) else len(messages))
        for position, start in enumerate(starts)
    ]
    return starts[0], blocks


def _summarize_tool_result(content: str, name: str) -> str | None:
    if len(content) <= TOOL_HEAD_TAIL_CHARS * 2 + 200:
        return None
    size = len(content.encode("utf-8"))
    exit_code = 1 if content.startswith("ERROR:") else 0
    return (
        f"[tool result truncated: {name}, {size}B, exit={exit_code}]\n"
        f"{content[:TOOL_HEAD_TAIL_CHARS]}\n...\n{content[-TOOL_HEAD_TAIL_CHARS:]}"
    )
