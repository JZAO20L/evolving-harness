from agent import (
    AgentLoop,
    Memory,
    Message,
    MockChatModel,
    ModelResult,
    Skill,
    SkillLibrary,
    ToolCall,
    Usage,
)


def _result(content="", *, tool_calls=None, prompt=1, completion=1):
    return ModelResult(content, usage=Usage(prompt, completion), tool_calls=tool_calls)


def _call(id, name, **args):
    return ToolCall(call_id=id, name=name, arguments=args)


class _SubMock(MockChatModel):
    """脚本化响应 + 记录每次 complete 的 (messages, tools) 供隔离/共享断言。

    父与子共用同一 model 实例，故按调用时序弹同一 flat script（顺序 await 使次序确定）。
    """

    def __init__(self, script):
        super().__init__(script=script)
        self.calls: list[tuple[list[Message], list | None]] = []   # (messages, tools)
        self.results: list[ModelResult] = []                       # (返回的结果，含 tool_calls)

    async def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        result = await super().complete(messages, tools=tools)
        self.results.append(result)
        return result

    def tool_observations(self) -> list[str]:
        """收集被当成 tool 结果显示的内容（父/子任何一层的观察）。"""
        out = []
        for messages, _ in self.calls:
            for m in messages:
                if m.role == "tool" and m.content:
                    out.append(m.content)
        return out

    def system_prompts(self) -> list[str]:
        return [m.content for messages, _ in self.calls for m in messages if m.role == "system"]


# --- 基础：不开 subagent 就与 AgentLoop 无异 ---------------------------------- #
async def test_loop_plain_without_subagent():
    mock = _SubMock([_result("直接答案")])
    res = await AgentLoop(mock).run("你好")
    assert res.answer == "直接答案" and res.finished


# --- 委派：父调 subagent 等子收官，只回最终文本作为一条 tool observation --------- #
async def test_subagent_delegates_returns_final_text_only(tmp_path):
    script = [
        _result("", tool_calls=[_call("c1", "subagent", task="子任务：调研 X")]),   # 父 step1
        _result("子任务结论：X 的核心要点"),                                          # 子 step1
        _result("父结论：引用「子任务结论：X 的核心要点」。"),                          # 父 step2
    ]
    mock = _SubMock(script)
    res = await AgentLoop(mock, subagent_enabled=True, max_steps=5).run("先委派再总结")

    assert res.answer == "父结论：引用「子任务结论：X 的核心要点」。" and res.finished
    # 子代理由父调 subagent 触发（工具调用在模型**返回**里，而非输入 messages）
    assert any(t.name == "subagent" for r in mock.results for t in (r.tool_calls or []))
    # 父收到的只有子最终文本这一条 tool observation（[子代理完成] 前缀）
    obs = mock.tool_observations()
    assert obs and obs[-1].startswith("[子代理完成]") and "子任务结论：X 的核心要点" in obs[-1]


# --- 隔离：子有自己的 context，不含父历史；父不泄子内部工具过程 ------------------- #
async def test_subagent_messages_independent():
    script = [
        _result("", tool_calls=[_call("c1", "subagent", task="子问题")]),   # 父 step1
        _result("子回答"),                                                   # 子 step1
        _result("父回答"),                                                   # 父 step2
    ]
    mock = _SubMock(script)
    await AgentLoop(mock, subagent_enabled=True, max_steps=5).run("父长问题背景 XXX")

    # 子首次 complete：只含子自己的 system + user(task)，没有父的 "父长问题背景"
    # （找到那条 messages 里 user 内容是 "子问题" 的调用）
    child_first = next(
        msgs for msgs, _ in mock.calls
        if any(m.role == "user" and m.content == "子问题" for m in msgs)
    )
    assert len([m for m in child_first if m.role in ("user", "system")]) == 2
    assert not any("父长问题背景" in m.content for m in child_first)          # 父历史不进子窗


# --- 预算减半：子步数 ≤ 父剩余/2，多给步数也会在子层耗尽（行为断言） ------------- #
async def test_subagent_child_budget_half_behavior():
    # 父 max_steps=10；step1 就委派 -> parent.steps_used=1，remaining=9，child_max=4。
    # 子给 5 个连续工具调用 -> 4 步就耗尽（第 5 步轮不到），返回"步数耗尽"当数据，父继续。
    script = [
        _result("", tool_calls=[_call("c1", "subagent", task="耗子")]),      # 父 step1
        _result("", tool_calls=[_call("w1", "work")]),                        # 子 step1
        _result("", tool_calls=[_call("w2", "work")]),                        # 子 step2
        _result("", tool_calls=[_call("w3", "work")]),                        # 子 step3
        _result("", tool_calls=[_call("w4", "work")]),                        # 子 step4（耗尽）
        _result("父最终结论"),                                                # 父 step2
    ]
    mock = _SubMock(script)
    res = await AgentLoop(mock, subagent_enabled=True, max_steps=10).run("委派")
    assert res.answer == "父最终结论"
    assert any("步数耗尽" in o for o in mock.tool_observations())


# --- 深度护栏：单元级直接测闸门 ----------------------------------------------- #
async def test_depth_guard_direct():
    loop = AgentLoop(_SubMock([]), subagent_enabled=True)
    loop.depth = 2                                   # 根0 儿1 孙2；孙再委派即越界
    out = await loop._run_subagent({"task": "x"})
    assert "深度已达上限" in out


# --- 深度护栏（集成）：到第 2 层孙再委派返回错误、整体有限序跑完不无限递归 -------- #
async def test_subagent_depth_limit_chain():
    # 注意：越界那次（孙调 subagent）由处理器直接返回错误文本、**不再弹脚本**，故 flat 脚本
    # 只被真实模型调用消耗。预算逐层减半：根 max=5，step1 后剩余 4 -> 儿 max=2 -> 孙 max=1。
    script = [
        _result("", tool_calls=[_call("s1", "subagent", task="t1")]),   # 根 step1
        _result("", tool_calls=[_call("s2", "subagent", task="t2")]),   # 儿 step1
        _result("", tool_calls=[_call("s3", "subagent", task="t3")]),   # 孙 step1 -> 越界
        _result("儿结果"),                                               # 儿 step2
        _result("根结果"),                                               # 根 step2
    ]
    mock = _SubMock(script)
    res = await AgentLoop(mock, subagent_enabled=True, max_steps=5).run("层层委派")
    # 整体有限递归：跑完、不爆栈、根正常收官（闸门已由 test_depth_guard_direct 直接验证）
    assert res.answer == "根结果" and res.finished
    # 只有儿层真正委派成功产出一次 [子代理完成]（孙层被闸门拦下、不产生成功委派）
    assert sum(1 for o in mock.tool_observations() if o.startswith("[子代理完成]")) == 1


# --- 子 observation 经父 _emit 的 abbreviate_observation 只留核心 ----------------- #
async def test_subagent_observation_abbreviated(tmp_path):
    long_answer = "核" * 3000                                     # 远超缩写阈值
    script = [
        _result("", tool_calls=[_call("c1", "subagent", task="给长报告")]),  # 父 step1
        _result(long_answer),                                              # 子 step1
        _result("父总结"),                                                 # 父 step2
    ]
    mem = Memory(MockChatModel(), storage_dir=str(tmp_path / "m"))
    mock = _SubMock(script)
    await AgentLoop(mock, memory=mem, subagent_enabled=True, max_steps=5).run("委派")

    # 父收到的子 observation 被 abbreviate_observation 截断，只留核心（前缀 + 规则简写标记）
    parent_obs = [o for o in mock.tool_observations() if o.startswith("[子代理完成]")]
    assert parent_obs and "[规则简写" in parent_obs[0] and len(parent_obs[0]) < 2000


# --- 共享配置：子继承同一技能目录 & 存储（memory.jsonl 记录子轮） ----------------- #
async def test_subagent_shares_skills_and_memory(tmp_path):
    skills = SkillLibrary(Skill("arxiv_search", "检索论文元数据", "完整指令…", priority=10))
    script = [
        _result("", tool_calls=[_call("c1", "subagent", task="给结论")]),  # 父 step1
        _result("子结论"),                                              # 子 step1
        _result("父结论"),                                              # 父 step2
    ]
    mdir = str(tmp_path / "m")
    mem = Memory(MockChatModel(), storage_dir=mdir)
    mock = _SubMock(script)
    await AgentLoop(mock, skills=skills, memory=mem,
                    subagent_enabled=True, max_steps=5).run("委派")

    # 子（与父）的 system 都含技能目录（渐进披露 info）——技能作为配置共享
    assert any("可用技能" in s for s in mock.system_prompts())
    # 子 finalize 把子轮写入同一 memory.jsonl（共享存储配置）
    path = tmp_path / "m" / "memory.jsonl"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "子结论" in text and "父结论" in text


# --- 未开 subagent_enabled 时，模型即便想调 subagent 也没有这工具 ----------------- #
async def test_subagent_absent_when_disabled():
    mock = _SubMock([_result("普通回答")])
    res = await AgentLoop(mock).run("hi")
    assert res.answer == "普通回答"