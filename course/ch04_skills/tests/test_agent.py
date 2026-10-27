from agent import (
    AgentLoop,
    Message,
    MockChatModel,
    ModelResult,
    Skill,
    SkillLibrary,
    ToolCall,
    ToolRegistry,
    Usage,
    builtin_read_tools,
)


def _result(content="", *, tool_calls=None, prompt=1, completion=1):
    return ModelResult(content, usage=Usage(prompt, completion), tool_calls=tool_calls)


def _skills():
    return SkillLibrary(
        Skill(name="concise_cn", description="用一句简洁中文给出结论",
              instructions="回答必须：用中文、一句话、结尾直接给结论，不铺陈。", priority=10),
        Skill(name="summarize_file", description="读取文件后给出结构化总结",
              instructions="先 read_file 读指定文件，再用 类/函数/要点 三节总结。", priority=5),
    )


class _Recorder(MockChatModel):
    def __init__(self, script):
        super().__init__(script=script)
        self.calls: list[tuple[list[Message], list | None]] = []

    async def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.script.pop(0)


# --- SkillLibrary: 渐进披露目录 --------------------------------------------- #
def test_list_summary_exposes_desc_but_not_instructions():
    text = _skills().list_summary()
    assert "concise_cn" in text and "summarize_file" in text
    assert "用一句简洁中文给出结论" in text      # desc 可进目录
    assert "不铺陈" not in text                 # instructions 绝不泄露（渐进披露的关键）


def test_list_summary_sorts_by_priority_desc():
    text = _skills().list_summary()
    assert text.index("concise_cn") < text.index("summarize_file")  # priority 10 在前


def test_list_summary_truncates_over_max_items():
    lib = SkillLibrary()
    for i in range(5):
        lib.add(Skill(name=f"s{i}", description=f"desc{i}", instructions="instr", priority=i))
    text = lib.list_summary(max_items=2)
    assert text.count("- s") == 2 and "只显示前 2" in text


def test_skill_get_returns_instructions_and_missing_raises():
    assert _skills().get("concise_cn").instructions.startswith("回答必须")
    try:
        _skills().get("nope")
        raise SystemExit("应 KeyError")
    except KeyError:
        pass


# --- AgentLoop + skills ----------------------------------------------------- #
def test_loop_registers_load_skill_tool():
    loop = AgentLoop(_Recorder([_result("ok")]), skills=_skills(), max_steps=5)
    names = {t["function"]["name"] for t in loop.registry.schema()}
    assert "load_skill" in names
    # 无 skills 时不注册
    loop2 = AgentLoop(_Recorder([_result("ok")]), max_steps=5)
    assert "load_skill" not in {t["function"]["name"] for t in loop2.registry.schema()}


async def test_loop_injects_skills_directory_into_system():
    rec = _Recorder([_result("ok")])
    await AgentLoop(rec, skills=_skills(), max_steps=5).run("x")
    sys_msg = rec.calls[0][0][0]
    assert sys_msg.role == "system"
    assert "可用技能" in sys_msg.content and "summarize_file" in sys_msg.content
    assert "不铺陈" not in sys_msg.content  # 目录里只有 desc，没有 instructions


async def test_loop_load_skill_injects_instructions_into_context():
    rec = _Recorder([
        _result("", tool_calls=[ToolCall("c1", "load_skill", {"name": "concise_cn"})]),
        _result("照做了：简洁回答。"),
    ])
    loop = AgentLoop(rec, skills=_skills(), max_steps=5)
    res = await loop.run("来个结论")
    second = rec.calls[1][0]
    assert any(m.role == "system" and "不铺陈" in m.content for m in second)  # 指令已注入
    assert res.answer == "照做了：简洁回答。" and res.steps == 2
    assert [t.name for t in res.tool_calls] == ["load_skill"]


async def test_loop_load_skill_unknown_is_error_as_data():
    rec = _Recorder([
        _result("", tool_calls=[ToolCall("c1", "load_skill", {"name": "ghost"})]),
        _result("没这技能，我直接答。"),
    ])
    loop = AgentLoop(rec, skills=_skills(), max_steps=5)
    res = await loop.run("x")
    assert "没有技能" in rec.calls[1][0][-1].content  # 错误当数据回填
    assert res.answer == "没这技能，我直接答。"


async def test_loop_load_then_use_read_file_skill_compose():
    rec = _Recorder([
        _result("", tool_calls=[ToolCall("c1", "load_skill", {"name": "summarize_file"})]),
        _result("", tool_calls=[ToolCall("c2", "read_file", {"path": __file__})]),
        _result("总结如下。"),
    ])
    reg = ToolRegistry()
    for t in builtin_read_tools():
        reg.register(t)
    loop = AgentLoop(rec, registry=reg, skills=_skills(), max_steps=6)
    res = await loop.run("总结这个文件")
    third = rec.calls[2][0]
    assert any("结构化总结" in m.content for m in third)      # 技能指令仍可见
    assert res.steps == 3 and res.answer == "总结如下。"      # load -> read -> ans
    assert [t.name for t in res.tool_calls] == ["load_skill", "read_file"]