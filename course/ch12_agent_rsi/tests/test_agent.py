"""L0 离线：ch12 agent-rsi 的结构参数化 / 护栏 / trace / 结构定位与编辑 / 搜索预算。

全部用脚本化 Mock，不联网；真模型只在 ``test_smoke.py``（EH_SMOKE=1）里跑。
"""

import json

from agent import ChatModel, Message, ModelResult, Tool, ToolCall
from rsi import (
    AgentConfig,
    Diagnosis,
    MemoryPolicy,
    PromptPack,
    RolloutResult,
    Rsi,
    RuleStructureLocator,
    RuleStructureProposer,
    StructureGuard,
    StructureLLMImprover,
    StructureSearchRsi,
    Task,
    VersionRegistry,
    rollout,
    update_structure,
)


def _tool(name: str, reply: str = "ok") -> Tool:
    return Tool(name=name, description=f"{name} 工具",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args: reply)


def _config(**structure) -> AgentConfig:
    base = {"name": "base", "prompt": PromptPack(guidance="先查事实再答。"),
            "memory": MemoryPolicy(recall_enabled=False, store_full_turns=False,
                                   profile_enabled=False)}
    base.update(structure)
    return AgentConfig(**base)


def _task(checks=("MAGIC",), task_id: str = "mth-1",
          statement: str = "请查工具") -> Task:
    return Task(id=task_id, statement=statement, checks=tuple(checks))


def _result(answer: str, *, finished: bool = True, steps: int = 1,
            step_limit: int = 8, tool_calls=(), tools_available=("lookup",),
            tools_enabled=("lookup",), subagent_enabled: bool = False,
            subagent_calls: int = 0) -> RolloutResult:
    return RolloutResult(task_id="mth-1", answer=answer, finished=finished,
                         steps=steps, step_limit=step_limit,
                         tool_calls=tuple(tool_calls),
                         tools_available=tuple(tools_available),
                         tools_enabled=tuple(tools_enabled),
                         subagent_enabled=subagent_enabled,
                         subagent_calls=subagent_calls)


class ScriptModel(ChatModel):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        self.calls += 1
        return ModelResult(content=self.replies.pop(0))


# --------------------------------------------------------------------------- #
# 1. 结构进入版本指纹；护栏夹取与不可变更新
# --------------------------------------------------------------------------- #
def test_structure_is_versioned_and_guarded():
    c1 = _config(base_tools=(_tool("a"),), max_steps=4)
    c2 = _config(base_tools=(_tool("a"),), max_steps=6)
    assert c1.canonical()["max_steps"] == 4
    assert c1.hash() != c2.hash(), "结构变化必须改变版本指纹"

    pool = (_tool("a"), _tool("b"))
    wild = _config(base_tools=pool, max_steps=99, tool_names=("a", "zzz"),
                   subagent_enabled=True, max_subagent_depth=9)
    clamped = StructureGuard().clamp(wild)
    assert clamped.max_steps == 12
    assert clamped.tool_names == ("a",), "未知工具名要被丢弃"
    assert clamped.subagent_enabled
    assert clamped.max_subagent_depth == 2

    no_sub = StructureGuard(allow_subagent=False).clamp(wild)
    assert not no_sub.subagent_enabled

    # 空 tool_names = 启用全部；但"全部"也必须受 max_tools 约束。
    pool3 = (_tool("a"), _tool("b"), _tool("c"))
    capped = StructureGuard(max_tools=2).clamp(_config(base_tools=pool3))
    assert capped.tool_names == ("a", "b")
    invalid = StructureGuard(max_tools=2).clamp(_config(base_tools=pool3, tool_names=("zzz",)))
    assert invalid.tool_names == ("a", "b"), "全部非法 → 退回全部时也要受上限约束"


def test_update_structure_is_immutable_and_noop_aware():
    cfg = _config(base_tools=(_tool("a"), _tool("b")), max_steps=4)
    cand = update_structure(cfg, max_steps=10, tool_names=["a"])
    assert cfg.max_steps == 4, "原 config 不能被污染"
    assert cand.max_steps == 10 and cand.tool_names == ("a",)

    assert update_structure(cfg) is cfg
    assert update_structure(cfg, unknown=1) is cfg
    assert update_structure(cfg, max_steps=4) is cfg, "无变化 → 原对象（哈希 noop）"
    assert update_structure(cfg, tool_names=["nope"]) is cfg, "未知工具 → 无变化"


# --------------------------------------------------------------------------- #
# 2. rollout 把结构写进 trace（步数耗尽 / 工具过滤 / subagent）
# --------------------------------------------------------------------------- #
class DelayedModel(ChatModel):
    """先连续查 ``n`` 次工具，再看观察作答（用来制造"步数不够"）。"""

    def __init__(self, n: int = 2) -> None:
        self.n = n

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        seen = sum(1 for m in messages if m.role == "tool")
        if seen < self.n:
            return ModelResult(content="", tool_calls=[ToolCall(f"c{seen + 1}", "lookup", {})])
        return ModelResult(content="MAGIC")


class LookupModel(ChatModel):
    """无论工具是否可用都尝试 look；工具观察里有 MAGIC 才答 MAGIC。"""

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        if not any(m.role == "tool" for m in messages):
            return ModelResult(content="", tool_calls=[ToolCall("c1", "lookup", {})])
        seen = any("MAGIC" in m.content for m in messages if m.role == "tool")
        return ModelResult(content="MAGIC" if seen else "0")


async def test_rollout_step_limit_trace(tmp_path):
    task = _task()
    tight = await rollout(_config(base_tools=(_tool("lookup", "MAGIC"),), max_steps=2),
                          task, DelayedModel(2), memory_dir=tmp_path / "tight")
    assert not tight.finished
    assert tight.steps == 2 and tight.step_limit == 2

    loose = await rollout(_config(base_tools=(_tool("lookup", "MAGIC"),), max_steps=4),
                          task, DelayedModel(2), memory_dir=tmp_path / "loose")
    assert loose.finished and loose.answer == "MAGIC"
    assert loose.steps == 3


async def test_rollout_tool_filtering_trace(tmp_path):
    pool = (_tool("lookup", "MAGIC"), _tool("other"))
    cfg = _config(base_tools=pool, tool_names=("other",), max_steps=4)
    r = await rollout(cfg, _task(), LookupModel(), memory_dir=tmp_path)
    assert r.tools_available == ("lookup", "other")
    assert r.tools_enabled == ("other",)
    assert r.answer == "0", "lookup 被禁用，模型只能看到错误观察"
    assert r.tool_calls == ("lookup",)

    all_on = await rollout(_config(base_tools=pool, max_steps=4), _task(), LookupModel(),
                           memory_dir=tmp_path / "all")
    assert all_on.tools_enabled == ("lookup", "other") and all_on.answer == "MAGIC"


async def test_rollout_subagent_trace(tmp_path):
    r = await rollout(_config(base_tools=(_tool("lookup"),), subagent_enabled=True,
                              max_subagent_depth=1),
                      _task(), ScriptModel(["done"]), memory_dir=tmp_path)
    assert r.subagent_enabled and r.subagent_calls == 0


# --------------------------------------------------------------------------- #
# 3. 结构粒度定位
# --------------------------------------------------------------------------- #
async def test_rule_structure_locator_components():
    locator = RuleStructureLocator()

    d = await locator.locate(_task(), _result("", finished=False, steps=2, step_limit=2), 0.0)
    assert d.component == "step_limit" and "2/2" in d.finding

    d = await locator.locate(_task(), _result("", finished=True), 0.0)
    assert d.component == "skip"

    d = await locator.locate(_task(), _result("0", finished=False, steps=2, step_limit=2), 0.0)
    assert d.component == "step_limit" and "2/2" in d.finding

    d = await locator.locate(_task(), _result("MAGIC"), 1.0)
    assert d.component == "ok"

    d = await locator.locate(_task(("42", "~步骤")), _result("42 步骤"), 0.5)
    assert d.component == "format"

    # 工具池里有 lookup，但只启用了 other；模型调 lookup 失败 → tool_missing
    d = await locator.locate(
        _task(), _result("0", tool_calls=("lookup",),
                         tools_available=("lookup", "other"), tools_enabled=("other",)), 0.0)
    assert d.component == "tool_missing" and "lookup" in d.finding

    # 多要点 + 并列词 + 全工具已开 + 没开 subagent → subagent_missing
    multi = _task(("A", "B"), statement="请分别给出 A 和 B")
    d = await locator.locate(multi, _result("0"), 0.0)
    assert d.component == "subagent_missing"

    # 工具已调过，结构没问题 → frame
    d = await locator.locate(_task(), _result("0", tool_calls=("lookup",),
                                              tools_available=("lookup",),
                                              tools_enabled=("lookup",)), 0.0)
    assert d.component == "frame"


# --------------------------------------------------------------------------- #
# 4. 规则结构编辑：目标档位幂等
# --------------------------------------------------------------------------- #
async def test_rule_structure_proposer_edits_and_is_idempotent():
    proposer = RuleStructureProposer()
    cfg = _config(base_tools=(_tool("a"), _tool("b")), max_steps=4,
                  tool_names=("a",), subagent_enabled=False)

    diag = Diagnosis("step_limit", "步数不够")
    cand = await proposer.propose(cfg, _task(), _result("0", finished=False), diag)
    assert cfg.max_steps == 4, "原 config 不变"
    assert cand.max_steps == 12
    assert await proposer.propose(cand, _task(), _result("0", finished=False), diag) is cand

    tool = await proposer.propose(cfg, _task(), _result("0"), Diagnosis("tool_missing", ""))
    assert {t.name for t in tool.enabled_tools()} == {"a", "b"}
    assert await proposer.propose(tool, _task(), _result("0"),
                                  Diagnosis("tool_missing", "")) is tool

    sub = await proposer.propose(cfg, _task(), _result("0"),
                                 Diagnosis("subagent_missing", ""))
    assert sub.subagent_enabled
    assert await proposer.propose(sub, _task(), _result("0"),
                                  Diagnosis("subagent_missing", "")) is sub

    blocked = RuleStructureProposer(StructureGuard(allow_subagent=False))
    assert await blocked.propose(cfg, _task(), _result("0"),
                                 Diagnosis("subagent_missing", "")) is cfg
    assert await proposer.propose(cfg, _task(), _result("0"), Diagnosis("ok", "")) is cfg


# --------------------------------------------------------------------------- #
# 5. 模型驱动结构改写
# --------------------------------------------------------------------------- #
async def test_structure_llm_improver_merges_and_clamps():
    reply = json.dumps({"component": "step_limit", "finding": "步数不够",
                        "structure": {"max_steps": 999, "tool_names": ["a"],
                                      "subagent_enabled": True,
                                      "max_subagent_depth": 9,
                                      "unknown": 1}}, ensure_ascii=False)
    cfg = _config(base_tools=(_tool("a"), _tool("b")), max_steps=4)
    diag, cand = await StructureLLMImprover(ScriptModel([reply])).improve(
        cfg, _task(), _result("0", finished=False))
    assert diag.component == "step_limit"
    assert cand.max_steps == 12 and cand.tool_names == ("a",)
    assert cand.subagent_enabled and cand.max_subagent_depth == 2

    diag, cand = await StructureLLMImprover(ScriptModel(["不是 JSON"])).improve(
        cfg, _task(), _result("0"))
    assert diag.component == "ok" and cand is cfg


# --------------------------------------------------------------------------- #
# 6. 版本回滚恢复结构
# --------------------------------------------------------------------------- #
def test_version_registry_rolls_back_structure(tmp_path):
    reg = VersionRegistry(tmp_path)
    pool = (_tool("a"), _tool("b"))
    cfg = _config(base_tools=pool, max_steps=10, tool_names=("a",),
                  subagent_enabled=True, max_subagent_depth=1)
    version = reg.register(cfg, score=0.5, verdict="accepted")
    restored = reg.rollback(version, base_tools=pool)
    assert restored.hash() == cfg.hash()
    assert restored.max_steps == 10 and restored.tool_names == ("a",)
    assert restored.subagent_enabled and restored.max_subagent_depth == 1


# --------------------------------------------------------------------------- #
# 7. 结构搜索：邻居生成 / 预算计数
# --------------------------------------------------------------------------- #
def test_structure_search_neighbors_and_dedup(tmp_path):
    pool = (_tool("a"), _tool("b"))
    rsi = StructureSearchRsi(_config(base_tools=pool, max_steps=4, tool_names=("a",)),
                             ScriptModel([]), train=[_task()], val=[_task(task_id="v")],
                             test=[_task(task_id="t")], storage_dir=tmp_path)
    comps = [c for c, _ in rsi._neighbors(rsi.config)]
    assert comps == ["step_limit", "tool_missing", "subagent_missing"]

    at_limit = _config(base_tools=pool, max_steps=12, subagent_enabled=True)
    rsi2 = StructureSearchRsi(at_limit, ScriptModel([]), train=[_task()],
                              val=[_task(task_id="v")], test=[_task(task_id="t")],
                              storage_dir=tmp_path / "2")
    assert rsi2._neighbors(at_limit) == [], "已到护栏边界 → 无邻居"

    tight = StructureSearchRsi(_config(base_tools=pool, max_steps=4), ScriptModel([]),
                               train=[_task()], val=[_task(task_id="v")],
                               test=[_task(task_id="t")], storage_dir=tmp_path / "3",
                               search_width=2)
    assert len(tight._neighbors(tight.config)) == 2, "search_width 限制邻居数"


# --------------------------------------------------------------------------- #
# 8. 端到端：步数不够 → 加步数 → promote + commit
# --------------------------------------------------------------------------- #
async def test_e2e_step_limit_promotes_and_commits(tmp_path):
    tool = _tool("lookup", "MAGIC")
    task = _task()
    cfg = _config(base_tools=(tool,), max_steps=2)
    rsi = StructureSearchRsi(
        cfg, DelayedModel(2), train=[task], val=[_task(task_id="v")],
        test=[_task(task_id="t")], storage_dir=str(tmp_path / "versions"),
        max_rounds=1, val_every=1, probe_tasks=1, search_width=3)

    report = await rsi.run()

    assert report.baseline_train_score == 0.0
    assert report.rounds[0].verdict == "promote"
    assert report.final_config.max_steps == 6
    assert report.segments[0].verdict == "commit"
    assert report.test_score == 1.0
    assert rsi.search_evals == 2, "两个邻居 × 1 个 probe 任务"


# --------------------------------------------------------------------------- #
# 9. 端到端：工具没放开 → 放开工具 → promote + commit
# --------------------------------------------------------------------------- #
async def test_e2e_tool_missing_promotes_and_commits(tmp_path):
    pool = (_tool("lookup", "MAGIC"), _tool("other"))
    task = _task()
    cfg = _config(base_tools=pool, tool_names=("other",), max_steps=4)
    rsi = StructureSearchRsi(
        cfg, LookupModel(), train=[task], val=[_task(task_id="v")],
        test=[_task(task_id="t")], storage_dir=str(tmp_path / "versions"),
        max_rounds=1, val_every=1, probe_tasks=1, search_width=3)

    report = await rsi.run()

    assert report.baseline_train_score == 0.0
    assert report.rounds[0].verdict == "promote"
    assert {t.name for t in report.final_config.enabled_tools()} == {"lookup", "other"}
    assert report.segments[0].verdict == "commit"
    assert report.test_score == 1.0
    assert rsi.search_evals == 3, "step + tool + subagent 三个邻居"


# --------------------------------------------------------------------------- #
# 10. 默认 Rsi 也走结构规则后端
# --------------------------------------------------------------------------- #
async def test_default_rsi_uses_structure_improver(tmp_path):
    tool = _tool("lookup", "MAGIC")
    cfg = _config(base_tools=(tool,), max_steps=2)
    report = await Rsi(cfg, DelayedModel(2), train=[_task()],
                       val=[_task(task_id="v")], test=[_task(task_id="t")],
                       storage_dir=str(tmp_path), max_rounds=1, val_every=1).run()
    assert report.rounds[0].verdict == "promote"
    assert report.final_config.max_steps == 12
    assert report.test_score == 1.0
