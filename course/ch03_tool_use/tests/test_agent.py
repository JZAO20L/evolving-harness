import json

import httpx
from agent import (
    AgentLoop,
    Message,
    MockChatModel,
    ModelResult,
    OpenAIChatModel,
    Tool,
    ToolCall,
    ToolRegistry,
    Usage,
)


def _result(content="", *, tool_calls=None, prompt=1, completion=1):
    return ModelResult(content, usage=Usage(prompt, completion),
                       tool_calls=tool_calls)


def _add_tool():
    return Tool(
        name="add",
        description="两数相加",
        parameters={"type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"]},
        handler=lambda args, _s=[]: (_s.append(args), str(args["a"] + args["b"]))[1],
    )


def _tc(name="add", **args):
    return ToolCall(call_id="call_1", name=name, arguments=args)


class _Recorder(MockChatModel):
    """脚本化响应 + 记录每轮 (messages, tools)，便于断言 context / schema 推进。"""

    def __init__(self, script):
        super().__init__(script=script)
        self.calls: list[tuple[list[Message], list | None]] = []

    async def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.script.pop(0)


# --- ToolRegistry: 注册 / 注入 / 分发 / 错误当数据 -------------------------- #
def test_registry_register_get_schema():
    reg = ToolRegistry()
    reg.register(_add_tool())
    assert reg.get("add").name == "add"
    schema = reg.schema()
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "add"


def test_registry_available_lists_tool_with_params():
    reg = ToolRegistry()
    reg.register(_add_tool())
    text = reg.available()
    assert "add(a*, b*)" in text and "两数相加" in text


async def test_registry_call_dispatches_arguments():
    reg = ToolRegistry()
    reg.register(_add_tool())
    out = await reg.call(_tc("add", a=1, b=2))
    assert out == "3"


async def test_registry_call_error_as_data():
    def boom(args):
        raise ValueError("bad input")
    reg = ToolRegistry()
    reg.register(Tool(name="boom", description="", parameters={}, handler=boom))
    out = await reg.call(_tc("boom", x=1))
    assert out.startswith("错误（ValueError）") and "bad input" in out


async def test_registry_call_unknown_tool():
    reg = ToolRegistry()
    out = await reg.call(_tc("nope", x=1))
    assert "nope" in out and "错误" in out


# --- OpenAIChatModel: 传 tools + 解析 tool_calls + 序列化往返 ------------------ #
def _chat_payload(content="", tool_calls=None, *, prompt=5, completion=1):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}}


async def test_openai_parses_tool_calls_from_response():
    def handler(_):
        return httpx.Response(200, json=_chat_payload(
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}]))

    model = OpenAIChatModel(
        base_url="http://localhost:8000/v1", model="m", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await model.complete([Message("user", "hi")], tools=[{"type": "function"}])
    assert result.tool_calls == [ToolCall(call_id="c1", name="add", arguments={"a": 1, "b": 2})]
    assert result.content == ""
    await model.aclose()


async def test_openai_serializes_assistant_tool_calls_and_tool_role():
    body = {}

    def handler(request):
        body["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json=_chat_payload("ok"))

    model = OpenAIChatModel(
        base_url="http://localhost:8000/v1", model="m", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    messages = [
        Message("assistant", "", tool_calls=[ToolCall("c1", "add", {"a": 1, "b": 2})]),
        Message("tool", "3", tool_call_id="c1"),
        Message("user", "then?"),
    ]
    await model.complete(messages)
    ser = body["body"]["messages"]
    assert ser[0]["role"] == "assistant"
    assert ser[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1, "b": 2}'
    assert ser[1] == {"role": "tool", "tool_call_id": "c1", "content": "3"}
    assert ser[2] == {"role": "user", "content": "then?"}
    await model.aclose()


# --- AgentLoop: 工具闭环 / 错误当数据 / 停止 / 步数 / 用量 -------------------- #
async def test_loop_invokes_tool_then_answers():
    reg = ToolRegistry()
    reg.register(_add_tool())
    rec = _Recorder([_result(tool_calls=[_tc("add", a=1, b=2)]),
                     _result("答案是 3", prompt=9, completion=3)])
    loop = AgentLoop(rec, registry=reg, max_steps=5)
    res = await loop.run("1+2=?")

    # 第一步把 tool_calls 的执行结果回填进 context，第二步再答
    second = rec.calls[1][0]
    assert second[-2].role == "assistant" and second[-2].tool_calls[0].name == "add"
    assert second[-1].role == "tool" and second[-1].content == "3"
    assert res.answer == "答案是 3"
    assert res.finished is True and res.steps == 2
    assert res.tool_calls == [_tc("add", a=1, b=2)]
    assert res.usage.total_tokens == 14


async def test_loop_stops_on_plain_text_no_tool():
    rec = _Recorder([_result("直接答完", prompt=2, completion=4)])
    loop = AgentLoop(rec, max_steps=5)
    res = await loop.run("hi")
    assert res.finished is True and res.steps == 1
    assert res.answer == "直接答完"
    assert res.tool_calls == []


async def test_loop_error_as_data_keeps_going():
    def boom(args):
        raise RuntimeError("bad arg")
    reg = ToolRegistry()
    reg.register(Tool(name="boom", description="", parameters={}, handler=boom))
    rec = _Recorder([_result(tool_calls=[_tc("boom", x=1)]),
                     _result("我注意到工具出错了，直接答")])
    loop = AgentLoop(rec, registry=reg, max_steps=5)
    res = await loop.run("go")
    assert "RuntimeError" in rec.calls[1][0][-1].content  # 错误被当观察回填
    assert res.finished is True and res.answer == "我注意到工具出错了，直接答"


async def test_loop_exhausts_steps_when_tool_fatigue():
    reg = ToolRegistry()
    reg.register(_add_tool())
    rec = _Recorder([_result(tool_calls=[_tc("add", a=1, b=1)]) for _ in range(4)])
    loop = AgentLoop(rec, registry=reg, max_steps=4)
    res = await loop.run("loop")
    assert res.finished is False and res.steps == 4
    assert len(res.tool_calls) == 4


async def test_loop_injects_tools_schema_every_step():
    reg = ToolRegistry()
    reg.register(_add_tool())
    rec = _Recorder([_result(tool_calls=[_tc("add", a=1, b=2)]),
                     _result("done")])
    loop = AgentLoop(rec, registry=reg, max_steps=5)
    await loop.run("x")
    # 每次 complete 都把 registry.schema() 注入 tools
    assert all(tools and tools[0]["function"]["name"] == "add" for _, tools in rec.calls)


async def test_loop_default_system_lists_tools():
    reg = ToolRegistry()
    reg.register(_add_tool())
    rec = _Recorder([_result("ok")])
    loop = AgentLoop(rec, registry=reg, max_steps=5)
    await loop.run("x")
    sys_msg = rec.calls[0][0][0]
    assert sys_msg.role == "system" and "add" in sys_msg.content

async def test_loop_unknown_tool_error_as_data():
    # 未知工具也"当数据"回填且不抛
    rec = _Recorder([_result(tool_calls=[_tc("ghost", x=1)]),
                     _result("最终答案")])
    loop = AgentLoop(rec, max_steps=5)
    res = await loop.run("x")
    assert "ghost" in rec.calls[1][0][-1].content
    assert res.answer == "最终答案"