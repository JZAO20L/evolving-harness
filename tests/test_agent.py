import json

import pytest
import yaml

from evolving_harness.agent import (
    Agent,
    AgentConfigError,
    Checkpointer,
    State,
    build_tools,
)
from evolving_harness.core.message import Message, ToolCall
from evolving_harness.models.base import ModelResponse, Usage
from evolving_harness.models.mock import MockChatModel
from evolving_harness.tracing.recorder import TraceRecorder


def test_state_loads_and_renders_prompt(make_agent):
    state = State.load(make_agent())

    assert state.version == 1
    assert len(state.hash) == 64

    prompt = state.render_system_prompt()
    assert "You are a test agent." in prompt
    assert "Check the workspace." in prompt
    assert "verify: Verify the result." in prompt
    assert "load_skill(name)" in prompt


def test_state_hash_tracks_semantic_content(make_agent):
    first = State.load(make_agent())
    second = State.load(make_agent(prompt={"guidance": "Check twice."}))

    assert first.hash != second.hash


def test_goals_rendered_into_prompt_and_tracked_by_identity(make_agent):
    state = State.load(
        make_agent(
            goals=[
                {"id": "g1", "statement": "Stay grounded in tool evidence.", "priority": 2},
                {"id": "g2", "statement": "Finish in the workspace.", "priority": 1},
            ]
        )
    )

    prompt = state.render_system_prompt()
    assert "## Goals" in prompt
    assert "Finish in the workspace." in prompt
    assert "Stay grounded in tool evidence." in prompt

    without_goals = State.load(make_agent())
    assert without_goals.hash != state.hash  # goals are part of the identity


async def test_load_skill_tool(make_agent, tmp_path):
    state = State.load(make_agent())
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
        State.load(make_agent(skills=skills))


def test_unknown_tool_is_rejected(make_agent, tmp_path):
    state = State.load(make_agent(tools={"enabled": ["nope"]}))

    with pytest.raises(AgentConfigError):
        build_tools(state, tmp_path / "ws")


async def test_trace_records_state_and_runtime(make_agent, tmp_path):
    state = State.load(make_agent())
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
    state = State.load(path)
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


def test_state_snapshot_roundtrips(make_agent):
    state = State.load(make_agent())
    run = state.begin_run(task="hi", run_id="abc")

    back = State.from_dict(run.snapshot())

    assert back.run_id == "abc"
    assert back.hash == state.hash
    assert back.name == "test"
    assert back.task == "hi"
    assert back.step == 0
    assert [m.role for m in back.messages] == ["system", "user"]
    assert back.messages[1].content == "hi"
    assert back.usage.total_tokens == 0


def test_checkpointer_saves_and_loads(make_agent, tmp_path):
    state = State.load(make_agent())
    run = state.begin_run(task="t", run_id="ck")

    cp = Checkpointer(tmp_path / "traces")
    cp.save(run)
    loaded = cp.load("ck")

    assert loaded is not None
    assert loaded.run_id == "ck"
    assert loaded.task == "t"
    assert loaded.name == "test"
    assert loaded.hash == state.hash


async def test_resume_continues_from_checkpoint(make_agent, tmp_path):
    path = make_agent()
    state = State.load(path)
    call = ToolCall(id="c1", name="list_dir", arguments={})
    rs = state.begin_run(task="continue", run_id="r1")
    rs.step = 1
    rs.messages = [
        Message(role="system", content=state.render_system_prompt()),
        Message(role="user", content="continue"),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="ok", tool_call_id="c1"),
    ]
    Checkpointer(tmp_path / "traces").save(rs)

    agent = Agent(
        model=MockChatModel([ModelResponse(content="final", usage=Usage(5, 1))]),
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
    )
    result = await agent.run("continue", run_id="r1", resume=True)

    assert result.status == "ok"
    assert result.steps == 2
    assert len(agent.model.calls) == 1  # resumed loop does not redo step 1
    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    kinds = [event["kind"] for event in events]
    assert "run.recover" in kinds
    recover = events[kinds.index("run.recover")]["data"]
    assert recover["from_step"] == 1
    assert events[-1]["kind"] == "run.end"
    assert events[-1]["data"]["status"] == "ok"


async def test_resume_appends_to_existing_trace(make_agent, tmp_path):
    state = State.load(make_agent())
    rs = state.begin_run(task="append", run_id="r3")
    rs.step = 1
    Checkpointer(tmp_path / "traces").save(rs)
    recorder = TraceRecorder("r3", tmp_path / "traces")
    recorder.event("run.start", task="append")
    recorder.event("model.request", step=1)
    recorder.close()

    agent = Agent(
        model=MockChatModel([ModelResponse(content="final")]),
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
    )
    result = await agent.run("append", run_id="r3", resume=True)

    events = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    kinds = [event["kind"] for event in events]
    assert kinds[:2] == ["run.start", "model.request"]  # original evidence kept
    assert "run.recover" in kinds
    seqs = [event["seq"] for event in events]
    assert seqs == list(range(1, len(seqs) + 1))


async def test_resume_marks_pending_call_unknown_not_rerun(make_agent, tmp_path):
    path = make_agent()
    state = State.load(path)
    call = ToolCall(
        id="c1", name="write_file", arguments={"path": "x.txt", "content": "boom"}
    )
    rs = state.begin_run(task="pend", run_id="r2")
    rs.step = 1
    rs.messages = [
        Message(role="system", content=state.render_system_prompt()),
        Message(role="user", content="pend"),
        Message(role="assistant", content="", tool_calls=[call]),
    ]
    rs.pending = [call]
    Checkpointer(tmp_path / "traces").save(rs)

    agent = Agent(
        model=MockChatModel([ModelResponse(content="final")]),
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
    )
    result = await agent.run("pend", run_id="r2", resume=True)

    assert result.status == "ok"
    assert not (tmp_path / "ws" / "x.txt").exists()  # tool was NOT re-dispatched
    assert any(
        "outcome unknown" in msg.content for msg in result.messages
    )


def test_skill_catalog_is_capped_sorted_and_summarized(make_agent):
    skills = [
        {
            "name": f"skill-{index}",
            "description": f"desc {index}",
            "instructions": "body",
            "summary": f"hint {index}",
            "priority": index % 5,
        }
        for index in range(20)
    ]
    state = State.load(make_agent(skills=skills, limits={"skill_catalog_limit": 5}))

    prompt = state.render_system_prompt()
    lines = [line for line in prompt.splitlines() if line.startswith("- skill-")]

    assert len(lines) == 5
    assert "+15 more" in prompt
    assert "hint 19" in prompt
    priorities = [int(line.rsplit("[p", 1)[1].split("]", 1)[0]) for line in lines]
    assert priorities == sorted(priorities, reverse=True)


async def test_list_skills_paginates(make_agent, tmp_path):
    skills = [
        {
            "name": f"skill-{index}",
            "description": f"desc {index}",
            "instructions": "body",
            "priority": index % 5,
        }
        for index in range(20)
    ]
    state = State.load(make_agent(skills=skills))
    registry = build_tools(state, tmp_path / "ws")

    result = await registry.dispatch(
        ToolCall(
            id="c1",
            name="list_skills",
            arguments={"page": 1, "page_size": 5},
        )
    )

    assert "skills 6-10 of 20" in result.content
    assert result.content.count("- skill-") == 5
    assert "[p" in result.content


def test_skill_priority_is_validated(make_agent):
    with pytest.raises(AgentConfigError):
        State.load(
            make_agent(
                skills=[
                    {
                        "name": "broken",
                        "description": "d",
                        "instructions": "i",
                        "priority": "high",
                    }
                ]
            )
        )


def test_auto_skill_tools_cannot_be_declared(make_agent):
    with pytest.raises(AgentConfigError):
        State.load(make_agent(tools={"enabled": ["list_skills"]}))
