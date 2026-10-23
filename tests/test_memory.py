import json

import pytest

from evolving_harness.agent import Agent, AgentConfigError, State, build_tools
from evolving_harness.core.message import ToolCall
from evolving_harness.models.base import ModelResponse
from evolving_harness.models.mock import MockChatModel


def build_memory_agent(
    make_agent,
    tmp_path,
    responses,
    *,
    enabled=True,
    max_tokens=256,
    max_steps=5,
):
    state = State.load(
        make_agent(
            limits={"max_steps": max_steps},
            memory={
                "enabled": enabled,
                "path": ".memory/notes.md",
                "max_tokens": max_tokens,
            },
        )
    )
    workspace = tmp_path / "ws"
    agent = Agent(
        model=MockChatModel(responses),
        tools=build_tools(state, workspace),
        state=state,
        trace_dir=tmp_path / "traces",
        workspace=workspace,
    )
    return agent, state, workspace


async def test_remember_persists_and_is_injected_next_run(make_agent, tmp_path):
    call = ToolCall(
        id="c1",
        name="remember",
        arguments={"key": "fact", "note": "The answer is 42."},
    )
    agent, _, workspace = build_memory_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    result = await agent.run("remember the fact")

    assert result.final == "done"
    notes = workspace / ".memory" / "notes.md"
    assert "[fact] The answer is 42." in notes.read_text()
    assert list((workspace / ".memory" / "raw").glob("*.jsonl"))
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert any(event["kind"] == "memory.write" for event in events)

    second, _, _ = build_memory_agent(make_agent, tmp_path, [])
    await second.run("recall the fact")

    system = second.model.calls[0][0]
    assert system.role == "system"
    assert "## Memory" in system.content
    assert "[fact] The answer is 42." in system.content


async def test_memory_disabled_reads_and_writes_nothing(make_agent, tmp_path):
    call = ToolCall(
        id="c1", name="write_file", arguments={"path": "note.txt", "content": "hi"}
    )
    agent, _, workspace = build_memory_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
        enabled=False,
    )

    await agent.run("write a note")

    assert (workspace / "note.txt").read_text() == "hi"
    assert not (workspace / ".memory").exists()
    assert "remember" not in build_tools(agent.state, workspace).names()


async def test_write_file_cannot_overwrite_memory(make_agent, tmp_path):
    workspace = tmp_path / "ws"
    notes = workspace / ".memory" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("- [fact] keep me\n", encoding="utf-8")

    call = ToolCall(
        id="c1",
        name="write_file",
        arguments={"path": ".memory/notes.md", "content": "overwrite"},
    )
    agent, _, _ = build_memory_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    result = await agent.run("overwrite the memory file")

    content = notes.read_text()
    assert "- [fact] keep me" in content
    assert "overwrite" not in content
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is True
    assert "reserved for harness state" in tool_result["data"]["content"]


async def test_memory_injection_is_capped_and_keeps_recent_notes(make_agent, tmp_path):
    workspace = tmp_path / "ws"
    notes = workspace / ".memory" / "notes.md"
    notes.parent.mkdir(parents=True)
    lines = [f"- [k{index}] fact {index} " + "x" * 80 for index in range(50)]
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    agent, state, _ = build_memory_agent(
        make_agent,
        tmp_path,
        [ModelResponse(content="done"), ModelResponse(content="- compressed facts")],
        max_tokens=64,
    )

    result = await agent.run("recall")

    system = agent.model.calls[0][0].content
    assert "## Memory" in system
    assert "[k49]" in system
    assert "[k0]" not in system
    assert len(system) < len(state.render_system_prompt()) + 64 * 4 + 200

    assert (workspace / ".memory" / "summary.md").read_text().strip() == "- compressed facts"
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert any(event["kind"] == "memory.summarize" for event in events)


def test_memory_config_is_validated(make_agent):
    with pytest.raises(AgentConfigError):
        State.load(make_agent(memory={"enabled": True, "path": "../notes.md"}))

    with pytest.raises(AgentConfigError):
        State.load(make_agent(memory={"enabled": True, "max_tokens": 0}))
