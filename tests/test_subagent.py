import json

from evolving_harness.agent import Agent, State, build_tools
from evolving_harness.core.message import Message, ToolCall
from evolving_harness.models.base import ModelResponse, Usage
from evolving_harness.models.mock import MockChatModel


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_agent(make_agent, tmp_path, model, max_steps=5):
    state = State.load(make_agent(limits={"max_steps": max_steps}))
    return Agent(
        model=model,
        tools=build_tools(state, tmp_path / "ws"),
        state=state,
        trace_dir=tmp_path / "traces",
        workspace=tmp_path / "ws",
    )


class RoutingModel(MockChatModel):
    """Answers by inspecting the last message, so parent and child share a model."""

    def __init__(self, routes, default, child_routes=None):
        super().__init__()
        self.routes = routes
        self.child_routes = child_routes or {}
        self.default = default
        self.contexts: list[list[Message]] = []

    async def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        self.contexts.append(list(messages))
        last = messages[-1]
        if last.role == "user":
            handler = self.routes.get(last.content) or self.child_routes.get(last.content)
            if handler is not None:
                return handler()
        return self.default(messages)


async def test_subagent_runs_in_fresh_context_and_links_traces(make_agent, tmp_path):
    def parent_task():
        return ModelResponse(
            tool_calls=[
                ToolCall(id="s1", name="subagent", arguments={"task": "child task"}),
            ],
            usage=Usage(3, 1),
        )

    model = RoutingModel(
        routes={"parent task": parent_task},
        child_routes={"child task": lambda: ModelResponse(content="child done")},
        default=lambda messages: ModelResponse(content="parent done"),
    )
    agent = build_agent(make_agent, tmp_path, model)

    result = await agent.run("parent task")

    assert result.status == "ok"
    assert result.final == "parent done"
    tool_result = next(
        event for event in read_events(result.trace_path) if event["kind"] == "tool.result"
    )
    assert tool_result["data"]["content"] == "child done"

    events = read_events(result.trace_path)
    start = next(event for event in events if event["kind"] == "subagent.start")
    end = next(event for event in events if event["kind"] == "subagent.end")
    child_id = start["data"]["child_run_id"]
    assert start["data"]["task"] == "child task"
    assert start["data"]["depth"] == 1
    assert end["data"]["child_run_id"] == child_id
    assert end["data"]["status"] == "ok"

    child_events = read_events(tmp_path / "traces" / f"{child_id}.jsonl")
    child_start = child_events[0]["data"]
    assert child_start["parent_run_id"] == result.run_id
    assert child_events[-1]["kind"] == "run.end"
    assert child_events[-1]["data"]["status"] == "ok"

    # The child starts from system + its own task, not the parent's transcript.
    child_context = next(
        context for context in model.contexts if context[-1].content == "child task"
    )
    assert [message.role for message in child_context] == ["system", "user"]
    assert "parent task" not in child_context[0].content
    assert "parent task" not in child_context[1].content


async def test_subagent_failure_is_data_and_parent_finishes(make_agent, tmp_path):
    def parent_task():
        return ModelResponse(
            tool_calls=[ToolCall(id="s1", name="subagent", arguments={"task": "child task"})]
        )

    def child_spins():
        return ModelResponse(
            tool_calls=[ToolCall(id="c1", name="list_dir", arguments={})]
        )

    model = RoutingModel(
        routes={"parent task": parent_task},
        child_routes={"child task": child_spins},
        default=lambda messages: ModelResponse(content="parent done"),
    )
    agent = build_agent(make_agent, tmp_path, model, max_steps=4)

    result = await agent.run("parent task")

    assert result.status == "ok"
    assert result.final == "parent done"
    events = read_events(result.trace_path)
    tool_result = next(event for event in events if event["kind"] == "tool.result")
    assert tool_result["data"]["content"].startswith("ERROR: subagent failed:")
    assert "MaxStepsExceeded" in tool_result["data"]["content"]

    end = next(event for event in events if event["kind"] == "subagent.end")
    assert end["data"]["status"] == "error"
    child_id = end["data"]["child_run_id"]
    child_events = read_events(tmp_path / "traces" / f"{child_id}.jsonl")
    assert child_events[-1]["kind"] == "run.end"
    assert child_events[-1]["data"]["status"] == "max_steps"


async def test_subagent_depth_is_capped_at_two(make_agent, tmp_path):
    def delegate(task):
        return lambda: ModelResponse(
            tool_calls=[ToolCall(id=f"s-{task}", name="subagent", arguments={"task": task})]
        )

    model = RoutingModel(
        routes={"root task": delegate("child task")},
        child_routes={
            "child task": delegate("grand task"),
            "grand task": delegate("too deep"),
        },
        default=lambda messages: ModelResponse(content="done"),
    )
    agent = build_agent(make_agent, tmp_path, model, max_steps=20)

    result = await agent.run("root task")

    assert result.status == "ok"
    parent_starts = [
        event["data"]
        for event in read_events(result.trace_path)
        if event["kind"] == "subagent.start"
    ]
    assert [start["depth"] for start in parent_starts] == [1]

    child_id = parent_starts[0]["child_run_id"]
    child_starts = [
        event["data"]
        for event in read_events(tmp_path / "traces" / f"{child_id}.jsonl")
        if event["kind"] == "subagent.start"
    ]
    assert [start["depth"] for start in child_starts] == [2]

    grand_id = child_starts[0]["child_run_id"]
    grand_events = read_events(tmp_path / "traces" / f"{grand_id}.jsonl")
    grand_tools = [
        event["data"]["content"]
        for event in grand_events
        if event["kind"] == "tool.result"
    ]
    assert any("depth limit" in content for content in grand_tools)


async def test_subagent_rejects_bad_step_budget(make_agent, tmp_path):
    def parent_task():
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id="s1",
                    name="subagent",
                    arguments={"task": "child task", "max_steps": 0},
                )
            ]
        )

    model = RoutingModel(
        routes={"parent task": parent_task},
        child_routes={"child task": lambda: ModelResponse(content="child done")},
        default=lambda messages: ModelResponse(content="parent done"),
    )
    agent = build_agent(make_agent, tmp_path, model)

    result = await agent.run("parent task")

    assert result.status == "ok"
    tool_result = next(
        event for event in read_events(result.trace_path) if event["kind"] == "tool.result"
    )
    assert tool_result["data"]["is_error"] is True
    assert "max_steps must be a positive integer" in tool_result["data"]["content"]
    # No child run was started for the rejected call.
    assert not any(
        event["kind"] == "subagent.start" for event in read_events(result.trace_path)
    )
