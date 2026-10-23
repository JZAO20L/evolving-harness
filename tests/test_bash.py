import json
import shutil
import time

import pytest

from evolving_harness.agent import Agent, State, build_tools
from evolving_harness.core.message import ToolCall
from evolving_harness.models.base import ModelResponse
from evolving_harness.models.mock import MockChatModel

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not available")


def build_agent(make_agent, tmp_path, responses, max_steps=5, **overrides):
    state = State.load(
        make_agent(
            limits={"max_steps": max_steps},
            tools={"enabled": ["read_file", "write_file", "list_dir", "bash"]},
            **overrides,
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
    return agent, workspace


async def test_bash_runs_in_workspace_with_spawn_isolation(make_agent, tmp_path):
    call = ToolCall(id="c1", name="bash", arguments={"command": "pwd; echo hi"})
    agent, workspace = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    result = await agent.run("show the workspace")

    assert result.final == "done"
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is False
    assert str(workspace.resolve()) in tool_result["data"]["content"]
    assert "hi" in tool_result["data"]["content"]

    tool_call = next(event for event in events if event["kind"] == "tool.call")
    isolation = tool_call["data"]["isolation"]
    assert isolation["level"] == "spawn"
    assert isolation["network"] == "none"
    assert isolation["fs_scope"] == "workspace_writable"


async def test_bash_nonzero_exit_is_error_but_run_continues(make_agent, tmp_path):
    call = ToolCall(
        id="c1", name="bash", arguments={"command": "echo boom; exit 3"}
    )
    agent, _ = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    result = await agent.run("fail on purpose")

    assert result.final == "done"
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is True
    assert "exit=3" in tool_result["data"]["content"]


async def test_bash_timeout_kills_the_process(make_agent, tmp_path):
    call = ToolCall(id="c1", name="bash", arguments={"command": "sleep 5", "timeout_ms": 100})
    agent, _ = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    started = time.monotonic()
    result = await agent.run("wait too long")
    elapsed = time.monotonic() - started

    assert result.final == "done"
    assert elapsed < 3
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is True
    assert "timeout>100ms" in tool_result["data"]["content"]


async def test_bash_output_is_capped(make_agent, tmp_path):
    call = ToolCall(id="c1", name="bash", arguments={"command": "yes x", "timeout_ms": 3000})
    agent, _ = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    result = await agent.run("flood the output")

    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is True
    assert "output truncated" in tool_result["data"]["content"]
    assert len(tool_result["data"]["content"]) < 64 * 1024 + 200


async def test_bash_parent_path_does_not_escape_the_host(make_agent, tmp_path):
    call = ToolCall(
        id="c1",
        name="bash",
        arguments={"command": "echo x > ../escaped.txt; echo done"},
    )
    agent, _ = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="done")],
    )

    await agent.run("write outside the workspace")

    assert not (tmp_path / "escaped.txt").exists()


async def test_bash_cannot_write_harness_memory(make_agent, tmp_path):
    workspace = tmp_path / "ws"
    notes = workspace / ".memory" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("- [fact] keep me\n", encoding="utf-8")

    agent, _ = build_agent(
        make_agent,
        tmp_path,
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"command": "echo overwrite > .memory/notes.md"},
                    )
                ]
            ),
            ModelResponse(content="done"),
        ],
        memory={"enabled": True, "path": ".memory/notes.md", "max_tokens": 256},
    )

    result = await agent.run("overwrite memory from bash")

    assert result.final == "done"
    assert "- [fact] keep me" in notes.read_text()
    assert "overwrite" not in notes.read_text()


async def test_bash_fails_closed_without_bwrap(monkeypatch, make_agent, tmp_path):
    monkeypatch.setattr("evolving_harness.tools.exec.shutil.which", lambda name: None)
    state = State.load(make_agent(tools={"enabled": ["bash"]}))
    registry = build_tools(state, tmp_path / "ws")

    result = await registry.dispatch(
        ToolCall(id="c1", name="bash", arguments={"command": "touch x"})
    )

    assert result.content.startswith("ERROR: bash requires bwrap")
    assert not (tmp_path / "ws" / "x").exists()
