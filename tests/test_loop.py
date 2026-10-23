import json

import pytest

from evolving_harness.agent import Agent, MaxStepsExceeded, State, build_tools
from evolving_harness.core.message import ToolCall
from evolving_harness.models.base import ModelResponse, Usage
from evolving_harness.models.mock import MockChatModel


def build_agent(make_agent, tmp_path, responses, max_steps=5):
    state = State.load(make_agent(limits={"max_steps": max_steps}))
    return Agent(
        model=MockChatModel(responses),
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
    )


async def test_loop_executes_tool_then_finishes(make_agent, tmp_path):
    call = ToolCall(
        id="c1", name="write_file", arguments={"path": "notes.txt", "content": "hello"}
    )
    agent = build_agent(
        make_agent,
        tmp_path,
        [
            ModelResponse(tool_calls=[call], usage=Usage(10, 2)),
            ModelResponse(content="done", usage=Usage(20, 3)),
        ],
    )

    result = await agent.run("write notes.txt")

    assert result.final == "done"
    assert result.steps == 2
    assert result.usage.total_tokens == 35
    assert (tmp_path / "ws" / "notes.txt").read_text() == "hello"

    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "run.start"
    assert "tool.call" in kinds and "tool.result" in kinds
    assert events[-1]["kind"] == "run.end"
    assert events[-1]["data"]["status"] == "ok"

    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is False
    assert tool_result["data"]["content"].startswith("wrote 5 bytes")


async def test_loop_reports_tool_errors_as_data(make_agent, tmp_path):
    call = ToolCall(id="c1", name="read_file", arguments={"path": "missing.txt"})
    agent = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="gave up")],
    )

    result = await agent.run("read a missing file")

    assert result.final == "gave up"
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is True


async def test_loop_stops_at_max_steps(make_agent, tmp_path):
    call = ToolCall(id="c1", name="list_dir", arguments={})
    agent = build_agent(
        make_agent,
        tmp_path,
        [ModelResponse(tool_calls=[call]) for _ in range(3)],
        max_steps=3,
    )

    with pytest.raises(MaxStepsExceeded):
        await agent.run("loop forever")

    trace_path = next((tmp_path / "traces").glob("*.jsonl"))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert events[-1]["data"]["status"] == "max_steps"
