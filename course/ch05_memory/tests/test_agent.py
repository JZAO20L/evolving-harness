
from agent import (
    AgentLoop,
    ContextBudget,
    Memory,
    MemoryStore,
    Message,
    MockChatModel,
    ModelResult,
    Tool,
    ToolCall,
    ToolRegistry,
    Usage,
    UserProfile,
    WorkingMemory,
    abbreviate_observation,
    compact_messages,
    estimate_tokens,
    has_profile_cue,
)


def _result(content="", *, tool_calls=None, prompt=1, completion=1):
    return ModelResult(content, usage=Usage(prompt, completion), tool_calls=tool_calls)


def _add_tool():
    return Tool(name="add", description="两数相加",
                parameters={"type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"]},
                handler=lambda a: str(a["a"] + a["b"]))


class _Recorder(MockChatModel):
    def __init__(self, script):
        super().__init__(script=script)
        self.calls: list[tuple[list[Message], list | None]] = []

    async def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.script.pop(0)


# --- token 估算 / 预算阈值 ----------------------------------------------------- #
def test_estimate_tokens_heuristic():
    assert estimate_tokens("12345") == 2
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 300) == 100


def test_context_budget_is_over_boundary():
    b = ContextBudget(max_len=100, threshold=0.5)
    assert b.is_over(51) and not b.is_over(50) and not b.is_over(1)


# --- 规则简写 / 工作记忆 -------------------------------------------------------- #
def test_abbreviate_collapses_whitespace_and_truncates():
    out = abbreviate_observation("  a   b\n\n  c  ", max_chars=100)
    assert out == "a b c"
    long = abbreviate_observation("x" * 1000, max_chars=50)
    assert len(long.split("[", 1)[0]) == 50 and "简写" in long


def test_working_memory_ingress_abbreviates_tool_resync_and_budget():
    wm = WorkingMemory(ContextBudget(max_len=200, threshold=2.0))
    m = wm.ingress(Message(role="tool", tool_call_id="c", content="x" * 2000))
    assert m.role == "tool" and "简写" in m.content and len(m.content) < 600
    plain = wm.ingress(Message(role="user", content="hi"))
    assert plain.content == "hi"                    # 只有 tool 被简写
    wm.resync([plain, m])
    assert wm.size() > 0
    assert not wm.is_over()                          # 2000->简写 ~<600 字符，tokens < 400 of threshold
    big = WorkingMemory(ContextBudget(max_len=60, threshold=0.5))
    big.resync([Message(role="user", content="y" * 400)])
    assert big.is_over()


# --- 模型改写压缩 --------------------------------------------------------------- #
async def test_compact_messages_folds_old_prefix_keeps_recent_tail():
    rec = MockChatModel(script=[_result("摘要：A 做了 X，得到 Y。")])
    msgs = [Message("system", "s"), Message("user", "u"),
            Message("assistant", "", tool_calls=[ToolCall("c", "add", {})]),
            Message("tool", "3", tool_call_id="c"),
            Message("user", "再来"), Message("assistant", "答"), Message("user", "继续")]
    budget = ContextBudget(keep_recent=3)
    new, digested = await compact_messages(rec, msgs, budget)
    assert digested == "摘要：A 做了 X，得到 Y。"
    assert new[0].role == "system" and "压缩摘要" in new[0].content
    assert new[1:] == msgs[-3:]                      # recent tail 原样保留
    assert len(new) == 4


async def test_compact_messages_short_context_is_noop():
    rec = MockChatModel(script=[_result("x")])
    msgs = [Message("system", "s"), Message("user", "u")]
    new, digested = await compact_messages(rec, msgs, ContextBudget(keep_recent=2))
    assert new == msgs and digested is None          # 没有旧前缀，不压缩也不调模型


# --- 用户画像：规则门控 + 模型总结 ------------------------------------------------ #
def test_profile_cue_gate_hit_and_miss():
    assert has_profile_cue([Message("user", "我喜欢 GRPO，偏好中文回答。")])
    assert not has_profile_cue([Message("user", "1 加 2 等于几？")])


async def test_profile_maybe_update_only_on_cue_and_persists(tmp_path):
    rec = _Recorder([_result('{"identity":"研究者","domain":"RL 红队","preferences":["GRPO"]}')])
    prof = UserProfile(rec, path=str(tmp_path / "mem" / "profile.json"))
    hit = await prof.maybe_update([Message("user", "我是做 RL 的，喜欢 GRPO。"),
                                   Message("assistant", "明白。")])
    assert hit and len(rec.calls) == 1
    assert prof.data["domain"] == "RL 红队" and prof.data["preferences"] == ["GRPO"]
    assert '"GRPO"' in (tmp_path / "mem" / "profile.json").read_text()


async def test_profile_maybe_update_without_cue_skips_model():
    rec = _Recorder([_result("1")])
    prof = UserProfile(rec, schema={"type": "object", "properties": {}})
    hit = await prof.maybe_update([Message("user", "1 加 2 等于几？"), Message("assistant", "3。")])
    assert hit is False and rec.calls == []          # 按需：无信号零模型调用


# --- 完整记忆落盘 + 自动回填 ----------------------------------------------------- #
def _big_store(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    store.append_turn([Message("user", "我喜欢朱雀"), Message("assistant", "玄武是前辈口述里的法宝")])
    return store


def test_store_append_turn_writes_transcript(tmp_path):
    _big_store(tmp_path)
    blob = (tmp_path / "mem" / "transcript.md").read_text()
    assert "玄武是前辈口述里的法宝" in blob and "**user**" in blob and "**assistant**" in blob


def test_store_recall_hits_missing_term_from_file():
    store = MemoryStore()
    store.append_turn([Message("assistant", "偏好 GRPO 优化器")])
    ctx = [Message("system", "你是教学 Agent。"), Message("user", "我之前说的 GRPO 是什么？")]
    snippet = store.recall(ctx)
    assert snippet and snippet.startswith("[从完整记忆检索到]") and "GRPO" in snippet


def test_store_recall_none_when_already_in_knowledge_or_absent():
    store = MemoryStore()
    store.append_turn([Message("assistant", "玄武是法宝")])
    # 知识(除本次提问外)已含 GRPO -> 不回填
    assert store.recall([Message("system", "GRPO 聊过。"), Message("user", "GRPO 是啥？")]) is None
    # 文件里没有的词 -> 不回填
    assert store.recall([Message("system", "s"), Message("user", "珊瑚礁是什么？")]) is None


# --- 组合：带 memory 的循环（压缩 / 回填 / 落盘） ---------------------------------- #
class _AutoMemoryModel(MockChatModel):
    """主循环脚本化；压缩调用(tools=None + 压缩提示)自动回摘要，不占主脚本。"""

    def __init__(self, script):
        super().__init__(script=script)
        self.calls: list[bool] = []                  # True=压缩调用

    async def complete(self, messages, tools=None):
        if tools is None and any("压缩" in m.content for m in messages):
            self.calls.append(True)
            return _result("（压缩）旧对话已被重写浓缩。")
        self.calls.append(False)
        return self.script.pop(0)


async def test_loop_with_memory_compresses_and_persists(tmp_path):
    model = _AutoMemoryModel([
        _result("", tool_calls=[ToolCall("c1", "echo", {})]),
        _result("收到，已压缩。"),
    ])
    # Memory 与 AgentLoop **共享同一个 model**——压缩是内部一次模型改写，也用同一实例。
    # keep_recent=2 < 当前 context 条数(4)，旧前缀才够压缩；echo 长观察撑满预算必然超限触发。
    mem = Memory(model, budget=ContextBudget(max_len=80, threshold=0.5, keep_recent=2),
                 storage_dir=str(tmp_path / "mem"))
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="返回一大段话，撑满工作记忆",
                      parameters={"type": "object", "properties": {}},
                      handler=lambda a: "a" * 600))     # 长观察 -> 预算必然超限，触发模型压缩
    loop = AgentLoop(model, registry=reg, memory=mem, max_steps=3)
    res = await loop.run("echo 一次")

    assert res.answer == "收到，已压缩。" and res.finished and res.steps == 2
    assert any(model.calls)                            # 压缩真实触发过（tools=None 调用）
    assert mem.compacted                                # Memory 记录了压缩事件
    blob = (tmp_path / "mem" / "transcript.md").read_text()
    assert "压缩摘要" in blob                            # 模型改写摘要已落盘（旧前缀折叠）
    assert "收到，已压缩。" in blob                     # 最终结论作为记忆一份子落盘


async def test_loop_with_memory_recalls_missing_prior_from_file(tmp_path):
    mem = Memory(MockChatModel([]), budget=ContextBudget(max_len=100000, threshold=0.9),
                 storage_dir=str(tmp_path / "mem"))
    mem.store.append_turn([Message("assistant", "玄武是前辈口述里的法宝")])  # 上一轮落盘
    rec = _Recorder([_result("没错，玄武是法宝。")])
    loop = AgentLoop(rec, memory=mem, max_steps=2)
    res = await loop.run("你提过玄武吗？")

    first = rec.calls[0][0]
    assert any(m.role == "system" and "[从完整记忆检索到]" in m.content for m in first)
    assert res.answer == "没错，玄武是法宝。"


async def test_loop_without_memory_behaves_as_ch04():
    rec = _Recorder([_result("", tool_calls=[ToolCall("c", "add", {"a": 1, "b": 2})]),
                     _result("和是 3。")])
    reg = ToolRegistry()
    reg.register(_add_tool())
    loop = AgentLoop(rec, registry=reg, max_steps=3)
    res = await loop.run("1+2")
    assert res.answer == "和是 3。" and res.steps == 2     # 无 memory 原样