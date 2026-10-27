import json

import pytest
from agent import (
    AgentLoop,
    Message,
    MockChatModel,
    ModelError,
    ModelResult,
    Usage,
    parse_act,
)


def _json_act(action_type="finish", **params):
    return json.dumps({"thought": "想了一下", "act": {"action_type": action_type, "params": params}})


class _Recorder(MockChatModel):
    """脚本化响应 + 记录每次收到的完整 messages，便于断言 context 逐轮推进。"""

    def __init__(self, script):
        super().__init__(script=script)
        self.calls: list[list[Message]] = []

    async def complete(self, messages):
        self.calls.append(list(messages))
        return self.script.pop(0)


def _result(content, *, prompt=1, completion=1):
    return ModelResult(content, usage=Usage(prompt, completion))


# --- parse_act -------------------------------------------------------------- #
def test_parse_act_clean_finish():
    act = parse_act(_json_act(answer="最终答案", note="x"))
    assert act.action_type == "finish"
    assert act.params == {"answer": "最终答案", "note": "x"}


def test_parse_act_fenced_code_block():
    act = parse_act("```json\n" + _json_act(answer="a") + "\n```")
    assert act.params == {"answer": "a"}


def test_parse_act_narrative_wrapped():
    text = "好的，让我想想。\n" + _json_act(answer="b") + "\n（以上是结果）"
    assert parse_act(text).params["answer"] == "b"


def test_parse_act_without_params_defaults_empty():
    act = parse_act('{"act": {"action_type": "finish"}}')
    assert act.params == {}


def test_parse_act_missing_act_raises():
    with pytest.raises(ModelError):
        parse_act('{"thought": "oops"}')


def test_parse_act_garbage_raises():
    with pytest.raises(ModelError):
        parse_act("完全没有 json 结构的一句话")


# --- AgentLoop: 终止 / 步数 / 用量 ------------------------------------------- #
async def test_run_terminates_on_finish_single_step():
    rec, loop = _Recorder([_result(_json_act(answer="你好"), prompt=2, completion=1)]), None
    loop = AgentLoop(rec, max_steps=5)
    res = await loop.run("hi")
    assert res.finished is True
    assert res.answer == "你好"
    assert res.steps == 1
    assert res.usage.total_tokens == 3
    assert [a.action_type for a in res.acts] == ["finish"]


async def test_run_accumulates_usage_across_steps():
    rec = _Recorder([_result(_json_act("check", task="x"), prompt=10, completion=5),
                     _result(_json_act("finish", answer="done"), prompt=20, completion=6)])
    loop = AgentLoop(rec, max_steps=5, dispatch=lambda act: "observe")
    res = await loop.run("task")
    assert res.finished is True
    assert res.steps == 2  # step1=check(观察) -> step2=finish
    assert res.usage.prompt_tokens == 30 and res.usage.completion_tokens == 11


async def test_run_exhausts_max_steps_when_never_finish():
    rec = _Recorder([_result(_json_act("think")) for _ in range(3)])
    loop = AgentLoop(rec, max_steps=3)
    res = await loop.run("go")
    assert res.finished is False
    assert res.steps == 3


async def test_run_context_starts_with_system_then_user():
    rec = _Recorder([_result(_json_act("finish"))])
    loop = AgentLoop(rec, max_steps=5)
    await loop.run("hello")
    first = rec.calls[0]
    assert first[0].role == "system" and first[0].content
    assert first[1].role == "user" and first[1].content == "hello"


async def test_run_remounts_act_and_tool_observation_into_context():
    rec = _Recorder([_result(_json_act("search", q="x")),
                     _result(_json_act("finish", answer="found"))])
    loop = AgentLoop(rec, max_steps=5, dispatch=lambda act: f"observe:{act.params['q']}")
    res = await loop.run("find")
    # 第二次调用（finish）时 context 已带上 assistant act + tool observation
    second = rec.calls[1]
    assert second[-2].role == "assistant" and "search" in second[-2].content
    assert second[-1].role == "tool" and second[-1].content == "observe:x"
    assert res.answer == "found" and res.steps == 2


async def test_run_default_dispatch_stubs_unregistered_act():
    rec = _Recorder([_result(_json_act("whatever")),
                     _result(_json_act("finish", answer="ok"))])
    loop = AgentLoop(rec, max_steps=5)
    res = await loop.run("x")
    assert "未注册" in rec.calls[1][-1].content
    assert res.finished is True


async def test_run_ignores_model_error_via_adapter():
    # 循环只看到"结果 or ModelError"；这里用脚本抛错验证循环能把错误当数据（ch03 正式打磨）
    class _Boom(MockChatModel):
        def __init__(self):
            super().__init__(script=[])
            self.calls = 0

        async def complete(self, messages):
            self.calls += 1
            raise ModelError("boom")
    loop = AgentLoop(_Boom(), max_steps=3)
    with pytest.raises(ModelError):
        await loop.run("x")