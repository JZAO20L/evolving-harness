import json

import httpx
from agent import (
    AgentLoop,
    Message,
    MockChatModel,
    ModelResult,
    OpenAIChatModel,
    PathPolicy,
    Tool,
    ToolCall,
    ToolRegistry,
    Usage,
    builtin_read_tools,
    list_dir,
    read_file,
    web_fetch,
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

# --- PathPolicy：静态 allow/deny + realpath 规约，拒绝当数据 ------------------- #
def test_policy_allows_within_root(tmp_path):
    whitelisted = tmp_path / "agent.py"
    whitelisted.write_text("x")
    policy = PathPolicy(allow_root=str(tmp_path))
    ok, why = policy.judge(str(whitelisted))
    assert ok and why == "ok"


def test_policy_denies_sensitive_name(tmp_path):
    (tmp_path / ".env").write_text("ARK_API_KEY=secret")
    policy = PathPolicy(allow_root=str(tmp_path))
    ok, why = policy.judge(str(tmp_path / ".env"))
    assert not ok and ".env" in why


def test_policy_denies_escape_via_parent_dotdot(tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("s")  # allow_root 之外
    policy = PathPolicy(allow_root=str(tmp_path))
    # ../ 逃逸：canonical 后仍落在根外 -> 拒
    ok, why = policy.judge(str(tmp_path / ".." / outside.name))
    assert not ok and "越出" in why


def test_policy_denies_system_prefix(tmp_path):
    policy = PathPolicy(allow_root=str(tmp_path))
    ok, why = policy.judge("/etc/hosts")
    assert not ok and ("系统前缀" in why or "越出" in why)


def test_read_file_with_policy_denies_env_when_data(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ARK_API_KEY=secret")
    policy = PathPolicy(allow_root=str(tmp_path))
    out = read_file(str(env), policy=policy)
    assert out.startswith("拒绝") and ".env" in out


def test_read_file_with_policy_allows_inside(tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("hello world")
    policy = PathPolicy(allow_root=str(tmp_path))
    assert read_file(str(f), policy=policy) == "hello world"


async def test_builtin_read_tools_carry_default_policy():
    reg = ToolRegistry()
    for t in builtin_read_tools():
        reg.register(t)
    # 默认 policy allow_root = agent.py 所在目录；对外部系统文件应拒绝（当数据）
    out = await reg.call(ToolCall("c1", "read_file", {"path": "/etc/hosts"}))
    assert out.startswith("拒绝")
    # allow_root 内文件仍可读
    here = PathPolicy()._base()
    ok = await reg.call(ToolCall("c1", "list_dir", {"path": here}))
    assert ok.startswith(here) and not ok.startswith("拒绝")


def test_read_file_reads_existing():
    out = read_file(__file__, max_chars=6000)
    assert out.startswith("import json")
    assert "ToolRegistry" in out  # 覆盖到中部内容，未被截断到


def test_read_file_truncates_over_limit():
    out = read_file(__file__, max_chars=10)
    assert len(out.split("\n", 1)[0]) == 10 and "截断" in out


def test_read_file_missing_returns_readable_error():
    out = read_file("/no/such/path.txt")
    assert out.startswith("错误") and "no/such/path" in out


def test_list_dir_lists_this_test_dir():
    out = list_dir(".")
    assert ".py" in out or "tests" in out
    assert "错误" not in out


def test_list_dir_missing_returns_readable_error():
    assert list_dir("/no/such/dir").startswith("错误")


# --- web_fetch：只读网络 GET（scheme 白名单 / 超时 / 截断 / 错误当数据）------ #
def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(handler):
    return _mock_client(lambda req: httpx.Response(200, text=handler(req)))


async def test_web_fetch_gets_http_text_offline():
    client = _ok(lambda req: f"<title>{req.url} hello</title>")
    out = await web_fetch("https://example.com/page", max_chars=200, http_client=client)
    assert "hello" in out and not out.startswith("错误")


async def test_web_fetch_rejects_non_http_scheme():
    out = await web_fetch("file:///etc/passwd")
    assert out.startswith("错误") and "只支持 http/https" in out
    assert "只支持 http/https" in await web_fetch("gopher://x/1")


async def test_web_fetch_error_as_data_on_http_error():
    client = _mock_client(lambda req: httpx.Response(404, text="missing"))
    out = await web_fetch("https://example.com/nope", http_client=client)
    assert out.startswith("错误") and "404" in out


async def test_web_fetch_truncates_over_limit():
    client = _ok(lambda req: "x" * 100)
    out = await web_fetch("https://example.com/big", max_chars=10, http_client=client)
    assert len(out.split("\n", 1)[0]) == 10 and "截断" in out


async def test_builtin_read_tools_registers_web_fetch():
    reg = ToolRegistry()
    for t in builtin_read_tools():
        reg.register(t)
    names = {t["function"]["name"] for t in reg.schema()}
    assert {"list_dir", "read_file", "web_fetch"} <= names


async def test_builtin_read_tools_register_and_call():
    reg = ToolRegistry()
    for t in builtin_read_tools():
        reg.register(t)
    out = await reg.call(ToolCall("c1", "read_file", {"path": __file__, "max_chars": 20000}))
    assert out.startswith("import json") and "ToolRegistry" in out


async def test_loop_can_use_read_file_to_answer():
    reg = ToolRegistry()
    for t in builtin_read_tools():
        reg.register(t)
    rec = _Recorder([_result(tool_calls=[ToolCall("c1", "read_file", {"path": __file__})]),
                     _result("我读了文件，里面有测试。")])
    loop = AgentLoop(rec, registry=reg, max_steps=5)
    res = await loop.run("读一下")
    assert "import json" in rec.calls[1][0][-1].content  # 观察回填了真实文件内容
    assert res.answer == "我读了文件，里面有测试。"


async def test_loop_unknown_tool_error_as_data():
    # 未知工具也"当数据"回填且不抛
    rec = _Recorder([_result(tool_calls=[_tc("ghost", x=1)]),
                     _result("最终答案")])
    loop = AgentLoop(rec, max_steps=5)
    res = await loop.run("x")
    assert "ghost" in rec.calls[1][0][-1].content
    assert res.answer == "最终答案"