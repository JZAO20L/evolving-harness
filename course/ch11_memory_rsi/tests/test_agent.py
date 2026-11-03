"""L0 离线：ch11 memory-rsi 的策略参数化 / trace / 记忆粒度定位与编辑 / 端到端演化。

全部用脚本化 Mock，不联网；真模型只在 ``test_smoke.py``（EH_SMOKE=1）里跑。
"""

import json
from pathlib import Path

import pytest
from agent import ChatModel, ContextBudget, Message, ModelResult, Tool, ToolCall
from rsi import (
    AgentConfig,
    Diagnosis,
    MemoryLLMImprover,
    MemoryPolicy,
    PolicyMemory,
    PolicyMemoryStore,
    PolicyWorkingMemory,
    PromptPack,
    RolloutResult,
    Rsi,
    RuleMemoryLocator,
    RuleMemoryProposer,
    Task,
    TwoStageImprover,
    VersionRegistry,
    load_synth,
    rollout,
    update_memory_policy,
)


def _config(**memory) -> AgentConfig:
    policy = MemoryPolicy(**memory) if memory else MemoryPolicy()
    return AgentConfig(name="base", prompt=PromptPack(guidance="先查事实再答。"), memory=policy)


def _task(checks=("42",), task_id: str = "mth-1") -> Task:
    return Task(id=task_id, statement="题目", checks=tuple(checks))


def _result(answer: str, *, finished: bool = True, abbreviated: bool = False,
            compacted: bool = False, recalled: bool = False,
            profile_updated: bool = False) -> RolloutResult:
    return RolloutResult(task_id="mth-1", answer=answer, finished=finished,
                         abbreviated=abbreviated, compacted=compacted,
                         recalled=recalled, profile_updated=profile_updated)


class ScriptModel(ChatModel):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        self.calls += 1
        return ModelResult(content=self.replies.pop(0))


# --------------------------------------------------------------------------- #
# 1. 策略进入版本指纹，更新不可变且带安全夹取
# --------------------------------------------------------------------------- #
def test_memory_policy_is_versioned_and_immutable():
    c1 = _config(abbreviate_chars=100)
    c2 = _config(abbreviate_chars=200)
    assert c1.canonical()["memory"]["abbreviate_chars"] == 100
    assert c1.hash() != c2.hash(), "记忆策略变化必须改变版本指纹"

    cand = update_memory_policy(c1, max_len=0, threshold=9, keep_recent=999,
                                recall_top_k=0, unknown=1)
    assert c1.memory.max_len == 6000, "原 config 不能被污染"
    assert cand.memory.max_len == 200
    assert cand.memory.threshold == 1.0
    assert cand.memory.keep_recent == 50
    assert cand.memory.recall_top_k == 1
    assert update_memory_policy(c1) is c1
    assert update_memory_policy(c1, max_len=6000) is c1, "无变化 → 原对象（哈希 noop）"


def test_memory_policy_from_canonical_tolerates_missing_and_bad_values():
    assert MemoryPolicy.from_canonical(None) == MemoryPolicy()
    p = MemoryPolicy.from_canonical({"max_len": 1, "threshold": -1, "recall_top_k": 999})
    assert p.max_len == 200 and p.threshold == 0.1 and p.recall_top_k == 50


# --------------------------------------------------------------------------- #
# 2. PolicyWorkingMemory / PolicyMemoryStore：截断阈值与召回 top-k 可调
# --------------------------------------------------------------------------- #
def test_policy_working_memory_tracks_truncation():
    wm = PolicyWorkingMemory(ContextBudget(), max_chars=5)
    msg = wm.ingress(Message(role="tool", content="abcdefgh", tool_call_id="t1"))
    assert wm.truncated and "截断" in msg.content

    wm2 = PolicyWorkingMemory(ContextBudget(), max_chars=50)
    wm2.ingress(Message(role="tool", content="abcdefgh", tool_call_id="t1"))
    assert not wm2.truncated


def test_policy_memory_store_recall_respects_top_k(tmp_path):
    store = PolicyMemoryStore(tmp_path, top_k=2)
    store.append_turn([
        Message(role="user", content="alpha one"),
        Message(role="user", content="alpha two"),
        Message(role="user", content="alpha three"),
    ])
    snippet = store.recall([Message(role="system", content="x"), Message(role="user", content="alpha")])
    assert snippet is not None
    assert snippet.count(" | ") == 1, "top_k=2 只应拼接两条命中"


async def test_policy_memory_switches_and_trace(tmp_path):
    m = PolicyMemory(ScriptModel([]), MemoryPolicy(recall_enabled=False,
                                                   store_full_turns=False,
                                                   profile_enabled=False),
                     storage_dir=tmp_path)
    context = [Message(role="system", content="sys"), Message(role="user", content="alpha")]
    await m.hook_pre_step(context)
    assert not m.recalled
    assert m.overhead() == ""
    await m.finalize(context)
    assert not (tmp_path / "memory.jsonl").exists(), "store_full_turns=False 不落盘"


async def test_policy_memory_recall_injects_and_marks_trace(tmp_path):
    m = PolicyMemory(ScriptModel([]), MemoryPolicy(recall_top_k=1), storage_dir=tmp_path)
    m.store.append_turn([Message(role="user", content="alpha old fact")])
    context = [Message(role="system", content="sys"), Message(role="user", content="alpha")]
    await m.hook_pre_step(context)
    assert m.recalled
    assert any("[从完整记忆检索到]" in msg.content for msg in context)


# --------------------------------------------------------------------------- #
# 3. rollout 把记忆行为写进 trace（截断 → 模型看不到事实）
# --------------------------------------------------------------------------- #
class LookupModel(ChatModel):
    """先调 lookup，再依据 tool 观察里有没有 MAGIC 作答。"""

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        if not any(m.role == "tool" for m in messages):
            return ModelResult(content="", tool_calls=[ToolCall("c1", "lookup", {})])
        seen = any("MAGIC" in m.content for m in messages if m.role == "tool")
        return ModelResult(content="MAGIC" if seen else "0")


def _lookup_tool() -> Tool:
    return Tool(name="lookup", description="查资料",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args: "x" * 100 + " MAGIC")


async def test_rollout_trace_reports_abbreviation(tmp_path):
    task = _task(("MAGIC",))

    def cfg(chars: int, name: str) -> AgentConfig:
        return AgentConfig(name=name, prompt=PromptPack(guidance="先查事实再答。"),
                           base_tools=(_lookup_tool(),),
                           memory=MemoryPolicy(abbreviate_chars=chars, recall_enabled=False,
                                               store_full_turns=False))

    tight = await rollout(cfg(20, "tight"), task, LookupModel(), memory_dir=tmp_path / "tight")
    assert tight.abbreviated and tight.answer == "0"

    loose = await rollout(cfg(2000, "loose"), task, LookupModel(), memory_dir=tmp_path / "loose")
    assert not loose.abbreviated and loose.answer == "MAGIC"


# --------------------------------------------------------------------------- #
# 4. 记忆粒度定位
# --------------------------------------------------------------------------- #
async def test_rule_memory_locator_components():
    locator = RuleMemoryLocator()
    task = _task(("42", "~计算过程"))

    d = await locator.locate(task, _result("0", finished=False, compacted=True), 0.0)
    assert d.component == "memory_overflow"
    assert (await locator.locate(task, _result("0", finished=False), 0.0)).component == "skip"

    d = await locator.locate(task, _result("0", abbreviated=True), 0.0)
    assert d.component == "memory_miss" and "截断" in d.finding
    d = await locator.locate(task, _result("0", compacted=True), 0.0)
    assert d.component == "memory_miss" and "压缩" in d.finding

    d = await locator.locate(task, _result("42 计算过程", recalled=True), 0.5)
    assert d.component == "memory_pollution"
    d = await locator.locate(task, _result("42 计算过程"), 0.5)
    assert d.component == "format"

    d = await locator.locate(task, _result("0", recalled=True), 0.0)
    assert d.component == "memory_pollution"
    assert (await locator.locate(task, _result(""), 0.0)).component == "skip"
    assert (await locator.locate(task, _result("42"), 1.0)).component == "ok"


# --------------------------------------------------------------------------- #
# 5. 规则策略调整：目标值幂等，不无限膨胀
# --------------------------------------------------------------------------- #
async def test_rule_memory_proposer_relaxes_on_miss():
    cfg = _config()
    diag = Diagnosis("memory_miss", "截断丢了事实")
    cand = await RuleMemoryProposer().propose(cfg, _task(), _result("0", abbreviated=True), diag)
    assert cfg.memory.abbreviate_chars == 500, "原 config 不变"
    assert cand.memory.abbreviate_chars == 2000
    assert cand.memory.keep_recent == 6
    again = await RuleMemoryProposer().propose(cand, _task(), _result("0", abbreviated=True), diag)
    assert again is cand, "同一诊断重复应用必须幂等"


async def test_rule_memory_proposer_tightens_on_overflow_and_pollution():
    proposer = RuleMemoryProposer()
    cfg = _config()

    overflow = await proposer.propose(cfg, _task(), _result("", finished=False, compacted=True),
                                      Diagnosis("memory_overflow", "budget"))
    assert overflow.memory.max_len == 4000
    assert overflow.memory.threshold == 0.6
    assert overflow.memory.keep_recent == 2
    assert overflow.memory.abbreviate_chars == 300

    pollution = await proposer.propose(cfg, _task(), _result("0", recalled=True),
                                      Diagnosis("memory_pollution", "recall"))
    assert pollution.memory.recall_top_k == 3 and pollution.memory.recall_enabled
    pollution2 = await proposer.propose(pollution, _task(), _result("0", recalled=True),
                                       Diagnosis("memory_pollution", "recall"))
    assert not pollution2.memory.recall_enabled
    assert await proposer.propose(pollution2, _task(), _result("0", recalled=True),
                                  Diagnosis("memory_pollution", "recall")) is pollution2

    profile = await proposer.propose(cfg, _task(), _result("0"),
                                     Diagnosis("memory_pollution", "profile"))
    assert not profile.memory.profile_enabled

    assert await proposer.propose(cfg, _task(), _result("0"), Diagnosis("ok", "")) is cfg


# --------------------------------------------------------------------------- #
# 6. 模型驱动策略改写
# --------------------------------------------------------------------------- #
async def test_memory_llm_improver_merges_and_clamps():
    reply = json.dumps({"component": "memory_miss", "finding": "截断",
                        "policy": {"abbreviate_chars": 3000, "max_len": 999999}},
                       ensure_ascii=False)
    diag, cand = await MemoryLLMImprover(ScriptModel([reply])).improve(
        _config(), _task(), _result("0", abbreviated=True))
    assert diag.component == "memory_miss"
    assert cand.memory.abbreviate_chars == 3000
    assert cand.memory.max_len == 100_000, "模型给的值要夹到安全上限"

    diag, cand = await MemoryLLMImprover(ScriptModel(["不是 JSON"])).improve(
        _config(), _task(), _result("0"))
    assert diag.component == "ok" and cand.memory == MemoryPolicy()


# --------------------------------------------------------------------------- #
# 7. 版本回滚 / synth 加载
# --------------------------------------------------------------------------- #
def test_version_registry_rolls_back_memory_policy(tmp_path):
    reg = VersionRegistry(tmp_path)
    cfg = _config(max_len=1234, keep_recent=7, recall_enabled=False)
    version = reg.register(cfg, score=0.5, verdict="accepted")
    restored = reg.rollback(version)
    assert restored.hash() == cfg.hash()
    assert restored.memory == cfg.memory


def test_load_synth_still_available():
    path = Path(__file__).resolve().parents[3] / "dataset" / "synth" / "math" / "train.jsonl"
    if not path.exists():
        pytest.skip(f"缺少合成数据 {path}")
    assert len(load_synth("math", "train")) == 30


# --------------------------------------------------------------------------- #
# 8. 端到端：截断阈值太狠 → memory_miss → 放宽 → promote + commit
# --------------------------------------------------------------------------- #
async def test_e2e_memory_policy_relaxation_promotes_and_commits(tmp_path):
    tool = _lookup_tool()
    task = Task(id="mth-1", statement="请查工具", checks=("MAGIC",))
    val = Task(id="mth-val", statement="请查工具", checks=("MAGIC",))
    test = Task(id="mth-test", statement="请查工具", checks=("MAGIC",))
    cfg = AgentConfig(name="base", prompt=PromptPack(guidance="先查事实再答。"),
                      base_tools=(tool,),
                      memory=MemoryPolicy(abbreviate_chars=20, recall_enabled=False,
                                          store_full_turns=False))
    rsi = Rsi(cfg, LookupModel(), train=[task], val=[val], test=[test],
              improver=TwoStageImprover(RuleMemoryLocator(), RuleMemoryProposer()),
              storage_dir=str(tmp_path / "versions"), max_rounds=1, val_every=1)

    report = await rsi.run()

    assert report.baseline_train_score == 0.0
    assert report.rounds[0].verdict == "promote"
    assert report.final_config.memory.abbreviate_chars == 2000
    assert report.final_config.memory.keep_recent == 6
    assert report.segments[0].verdict == "commit"
    assert report.test_score == 1.0
