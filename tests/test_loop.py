import json

import pytest

from evolving_harness.core.loop import AgentLoop, MaxStepsExceeded
from evolving_harness.core.message import ToolCall
from evolving_harness.models.base import ModelResponse, Usage
from evolving_harness.models.mock import MockChatModel
from evolving_harness.tools.base import ToolRegistry
from evolving_harness.tools.fs import fs_tools


def build_loop(tmp_path, responses, max_steps=5):
    workspace = tmp_path / "ws"
    return AgentLoop(
        model=MockChatModel(responses),
        tools=ToolRegistry(fs_tools(workspace)),
        system_prompt="be brief",
        max_steps=max_steps,
        trace_dir=tmp_path / "traces",
    )


async def test_loop_executes_tool_then_finishes(tmp_path):
    call = ToolCall(
        id="c1", name="write_file", arguments={"path": "notes.txt", "content": "hello"}
    )
    loop = build_loop(
        tmp_path,
        [
            ModelResponse(tool_calls=[call], usage=Usage(10, 2)),
            ModelResponse(content="done", usage=Usage(20, 3)),
        ],
    )

    result = await loop.run("write notes.txt")

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


async def test_loop_reports_tool_errors_as_data(tmp_path):
    call = ToolCall(id="c1", name="read_file", arguments={"path": "missing.txt"})
    loop = build_loop(
        tmp_path,
        [ModelResponse(tool_calls=[call]), ModelResponse(content="gave up")],
    )

    result = await loop.run("read a missing file")

    assert result.final == "gave up"
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["is_error"] is True


async def test_loop_stops_at_max_steps(tmp_path):
    call = ToolCall(id="c1", name="list_dir", arguments={})
    loop = build_loop(
        tmp_path, [ModelResponse(tool_calls=[call]) for _ in range(3)], max_steps=3
    )

    with pytest.raises(MaxStepsExceeded):
        await loop.run("loop forever")

    trace_path = next((tmp_path / "traces").glob("*.jsonl"))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert events[-1]["data"]["status"] == "max_steps"
