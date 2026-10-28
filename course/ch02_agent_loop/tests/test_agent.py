import asyncio
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


# --- 关键设计 · 运行期用户输入注入（not 跑完才理人） ---------------------------- #
def test_drain_input_empty_queue_returns_none():
    q: asyncio.Queue = asyncio.Queue()
    loop = AgentLoop(_Recorder([_result(_json_act("finish"))]), input_queue=q)
    assert loop._drain_input() is None                 # 空队列非阻塞返回 None，不卡循环
    assert AgentLoop(_Recorder([_result(_json_act("finish"))]))._drain_input() is None


async def test_input_injection_surfaces_as_priority_user_turn():
    # 用户已在运行中注入"改方向"指令（队列里已有一条）；循环第一步边界就把它以优先级塞入。
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("改成先调研 B")
    rec = _Recorder([_result(_json_act("check", task="x")),
                     _result(_json_act("finish", answer="已按新方向处理"))])
    await AgentLoop(rec, max_steps=5, dispatch=lambda act: "obs", input_queue=q).run("先做 A")

    first = rec.calls[0]
    injected = [m for m in first if m.role == "user" and "优先处理" in m.content]
    assert len(injected) == 1 and "改成先调研 B" in injected[0].content
    # 优先级 = 最新 user 位：注入出现在原始用户指令**之后**
    assert first[1].role == "user" and first[1].content == "先做 A"
    assert first[2].role == "user" and "优先处理" in first[2].content


class _FeedMock(_Recorder):
    """首个 complete 时把一条用户注入推进队列（模拟**运行中途**用户插话），再正常回脚本。"""

    def __init__(self, script, queue, inject="中途改成先做 C"):
        super().__init__(script=script)
        self.q = queue
        self.inject = inject
        self.fed = False

    async def complete(self, messages):
        self.calls.append(list(messages))
        if not self.fed:
            self.fed = True
            self.q.put_nowait(self.inject)
            await asyncio.sleep(0)                 # 让出一次事件循环，确认"运行期间"注入已入队
        return self.script.pop(0)


async def test_input_injection_mid_run_picked_up_next_boundary():
    q: asyncio.Queue = asyncio.Queue()
    rec = _FeedMock([_result(_json_act("check", task="x")),
                     _result(_json_act("finish", answer="引用新指令：已改做 C"))],
                    q)
    res = await AgentLoop(rec, max_steps=5, dispatch=lambda act: "obs", input_queue=q).run("先做 A")

    assert res.finished and res.answer == "引用新指令：已改做 C"
    # step1 的 complete 还看不到注入（它在运行中途才入队）
    assert not any("中途改成先做 C" in m.content for m in rec.calls[0])
    # step2 边界已把它按优先级（最新 user 位）塞入：位于 step1 的 assistant/tool 之后
    second = rec.calls[1]
    marker = [m for m in second if m.role == "user" and "优先处理" in m.content]
    assert len(marker) == 1 and "中途改成先做 C" in marker[0].content
    assert second[-1] is marker[0]                       # 它是上下文中最新的 user（优先级最高）