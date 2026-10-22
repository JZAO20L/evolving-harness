import json

import pytest
import yaml

from evolving_harness.agent import (
    Agent,
    AgentConfigError,
    AgentState,
    build_tools,
)
from evolving_harness.core.message import ToolCall
from evolving_harness.models.base import ModelResponse
from evolving_harness.models.mock import MockChatModel


def test_state_loads_and_renders_prompt(make_agent):
    state = AgentState.load(make_agent())

    assert state.version == 1
    assert len(state.hash) == 64

    prompt = state.render_system_prompt()
    assert "You are a test agent." in prompt
    assert "Check the workspace." in prompt
    assert "verify: Verify the result." in prompt
    assert "load_skill(name)" in prompt


def test_state_hash_tracks_semantic_content(make_agent):
    first = AgentState.load(make_agent())
    second = AgentState.load(make_agent(prompt={"guidance": "Check twice."}))

    assert first.hash != second.hash


async def test_load_skill_tool(make_agent, tmp_path):
    state = AgentState.load(make_agent())
    registry = build_tools(state, tmp_path / "ws")

    loaded = await registry.dispatch(
        ToolCall(id="c1", name="load_skill", arguments={"name": "verify"})
    )
    assert loaded.content == "1. Re-read the file."

    missing = await registry.dispatch(
        ToolCall(id="c2", name="load_skill", arguments={"name": "nope"})
    )
    assert missing.content.startswith("ERROR: unknown skill")


@pytest.mark.parametrize(
    "skills",
    [
        [{"name": "Bad Name", "description": "x", "instructions": "y"}],
        [
            {"name": "verify", "description": "x", "instructions": "y"},
            {"name": "verify", "description": "x", "instructions": "y"},
        ],
    ],
)
def test_invalid_skills_are_rejected(make_agent, skills):
    with pytest.raises(AgentConfigError):
        AgentState.load(make_agent(skills=skills))


def test_unknown_tool_is_rejected(make_agent, tmp_path):
    state = AgentState.load(make_agent(tools={"enabled": ["nope"]}))

    with pytest.raises(AgentConfigError):
        build_tools(state, tmp_path / "ws")


async def test_trace_records_state_and_runtime(make_agent, tmp_path):
    state = AgentState.load(make_agent())
    agent = Agent(
        model=MockChatModel(),
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
    )

    result = await agent.run("say hello")

    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    start = events[0]["data"]
    assert start["state"]["name"] == "test"
    assert start["state"]["version"] == 1
    assert start["state"]["hash"] == state.hash
    assert start["runtime"]["id"] == "mock"
    assert start["runtime"]["temperature"] == 0.2
    assert start["runtime"]["max_tokens"] == 64
    assert events[-1]["data"]["status"] == "ok"
    assert result.status == "ok"


class DriftingModel(MockChatModel):
    def __init__(self, path):
        super().__init__([ModelResponse(content="done")])
        self.path = path

    async def complete(self, messages, tools=None):
        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        data["prompt"]["guidance"] = "changed during the run"
        self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return await super().complete(messages, tools)


async def test_state_drift_invalidates_the_run(make_agent, tmp_path):
    path = make_agent()
    state = AgentState.load(path)
    agent = Agent(
        model=DriftingModel(path),
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
    )

    result = await agent.run("say hello")

    assert result.status == "drift"
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert events[-1]["data"]["status"] == "drift"
    assert events[-1]["data"]["state_hash"] != state.hash
