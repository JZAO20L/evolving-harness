import asyncio
from pathlib import Path

import httpx
from agent import (
    AgentLoop,
    Message,
    MockChatModel,
    ModelResult,
    Skill,
    SkillLibrary,
    Tool,
    ToolCall,
    ToolRegistry,
    Usage,
    builtin_read_tools,
    read_skill_file,
    web_fetch,
)

SKILLS_DIR = str(Path(__file__).resolve().parent.parent / "skills")
ARXIV_XML = (
    "<?xml version='1.0'?><feed><entry><id>http://arxiv.org/abs/2607.00001v1</id>"
    "<title>Red-Teaming RL Agents with Language Models</title>"
    "<published>2026-07-01T00:00:00Z</published><summary>We study jailbreaks in RL.</summary>"
    "<category term='cs.LG'/><category term='cs.CL'/></entry></feed>"
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


# --- 技能文件加载器：frontmatter{name/desc/priority} + 正文 instructions ---------- #
# --- 上手 oxo：把 arxiv_search.skill.md（磁盘真实技能）读进 Skill / SkillLibrary。---- #
def test_skill_from_markdown_parses_frontmatter_and_body():
    skill = Skill.from_markdown(
        "---\nname: demo\n"
        "description: 一个演示技能\n"
        "priority: 7\n"
        "---\n先 read_file 再总结。")
    assert skill.name == "demo"
    assert skill.description == "一个演示技能"
    assert skill.instructions == "先 read_file 再总结。"
    assert skill.priority == 7


def test_skill_from_markdown_no_frontmatter_missing_required_raises():
    # 没有 frontmatter -> 没有 name/description -> Skill 必需字段缺失，报错（校验在加载端）
    try:
        Skill.from_markdown("就一段正文", priority=3)
        raise SystemExit("应报缺 name/description")
    except ValueError:
        pass


def test_read_skill_file_loads_arxiv_skill_from_disk():
    skill = read_skill_file(f"{SKILLS_DIR}/arxiv_search.skill.md")
    assert skill.name == "arxiv_search"
    assert "arxiv" in skill.description
    assert "web_fetch" in skill.instructions and "export.arxiv.org/api/query" in skill.instructions


def test_skill_library_load_dir_batches_skill_files():
    lib = SkillLibrary.load_dir(SKILLS_DIR)
    assert lib.get("arxiv_search").name == "arxiv_search"
    text = lib.list_summary()
    assert "arxiv_search" in text                                   # 目录可见名+desc
    assert "export.arxiv.org" not in text                          # 指令不被目录泄露


# --- 组合：加载磁盘 arxiv 技能 -> 用 web_fetch（MockTransport 离线）查询 ------------ #
def _arxiv_client():
    return httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, text=ARXIV_XML)))


async def test_loop_load_arxiv_skill_then_web_fetch_offline():
    rec = _Recorder([
        _result("", tool_calls=[ToolCall("c1", "load_skill", {"name": "arxiv_search"})]),
        _result("", tool_calls=[ToolCall("c2", "web_fetch", {
            "url": "https://export.arxiv.org/api/query?search_query=all:RL", "max_chars": 8000})]),
        _result("查到一篇：Red-Teaming RL Agents with Language Models。"),
    ])
    reg = ToolRegistry()
    for t in builtin_read_tools():
        reg.register(t)
    client = _arxiv_client()
    reg.register(Tool(name="web_fetch", description="", parameters={},
                      handler=lambda a: web_fetch(a["url"], http_client=client)))  # 覆盖为 mock
    lib = SkillLibrary.load_dir(SKILLS_DIR)
    loop = AgentLoop(rec, registry=reg, skills=lib, max_steps=6)
    res = await loop.run("帮我搜一下 RL 相关的论文")
    third = rec.calls[2][0]
    obs = [m for m in third if m.role == "tool"][-1].content
    assert "entry" in obs and "Red-Teaming" in obs      # web_fetch 观察回填了真 parse 的 XML
    assert res.steps == 3 and "Red-Teaming" in res.answer
    assert [t.name for t in res.tool_calls] == ["load_skill", "web_fetch"]


# --- 关键设计 · 运行期用户输入注入（ch02 引入：agent 不是跑完才理人） ---------------------- #
def test_drain_input_empty_queue_returns_none():
    q: asyncio.Queue = asyncio.Queue()
    assert AgentLoop(_Recorder([]), input_queue=q)._drain_input() is None   # 空队列不阻塞
    assert AgentLoop(_Recorder([]))._drain_input() is None                 # 未接队列


async def test_input_injection_priority_user_turn():
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("改成先调研 B")                                   # 运行前已注入
    rec = _Recorder([_result("已按新方向处理")])
    await AgentLoop(rec, max_steps=5, input_queue=q).run("先做 A")

    first = rec.calls[0][0]                                        # 第一步的 messages
    injected = [m for m in first if m.role == "user" and "优先处理" in m.content]
    assert len(injected) == 1 and "改成先调研 B" in injected[0].content
    assert first[1].role == "user" and first[1].content == "先做 A"   # 原始指令在前
    assert first[2] is injected[0]                                    # 注入 = 最新 user（优先级最高）


class _FeedMock(_Recorder):
    """首个 complete 时把一条注入推进队列（模拟**运行中途**插话），再正常回脚本。"""

    def __init__(self, script, queue, inject="中途改成先做 C"):
        super().__init__(script=script)
        self.q, self.inject, self.fed = queue, inject, False

    async def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        if not self.fed:
            self.fed = True
            self.q.put_nowait(self.inject)
            await asyncio.sleep(0)                             # 让出一次事件循环，确认"运行期间"已入队
        return self.script.pop(0)


async def test_input_injection_mid_run_picked_up_next_boundary():
    q: asyncio.Queue = asyncio.Queue()
    rec = _FeedMock([_result("", tool_calls=[ToolCall("c1", "ghost", {})]),  # step1 先跑一个工具步
                     _result("引用新指令：已改做 C")],                        # step2 引用注入收尾
                    q)
    res = await AgentLoop(rec, max_steps=5, input_queue=q).run("先做 A")
    assert res.finished and res.answer == "引用新指令：已改做 C"
    assert not any("中途改成先做 C" in m.content for m in rec.calls[0][0])   # step1 还没看到
    second = rec.calls[1][0]
    marker = [m for m in second if m.role == "user" and "优先处理" in m.content]
    assert len(marker) == 1 and "中途改成先做 C" in marker[0].content
    assert second[-1] is marker[0]                             # 上下文中最新的 user（优先级最高）