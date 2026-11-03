"""L0 离线：ch13 meta-rsi 的过程配置 / 过程定位与编辑 / 元门禁 / 三层数据纪律。

全部用脚本化 Mock，不联网；真模型只在 ``test_smoke.py``（EH_SMOKE=1）里跑。
"""

import json
from types import SimpleNamespace

from agent import ChatModel, Message, ModelResult, Tool
from rsi import (
    AgentConfig,
    InnerOutcome,
    LLMJudge,
    MemoryPolicy,
    MetaDiagnosis,
    MetaGate,
    MetaLLMImprover,
    MetaPolicy,
    MetaRsi,
    ProcessFactory,
    PromptPack,
    RuleJudge,
    RuleMetaLocator,
    RuleMetaProposer,
    Task,
    TwoStageImprover,
    update_meta_policy,
)


def _tool(name: str) -> Tool:
    return Tool(name=name, description=name,
                parameters={"type": "object", "properties": {}},
                handler=lambda _args: "ok")


def _cfg(**changes) -> AgentConfig:
    base = {"name": "base", "prompt": PromptPack(guidance="先算再答。"), "max_steps": 3,
            "memory": MemoryPolicy(recall_enabled=False, store_full_turns=False,
                                   profile_enabled=False)}
    base.update(changes)
    return AgentConfig(**base)


def _task(task_id: str = "t1", statement: str = "算一下 40+2",
          checks=("42", "~步骤")) -> Task:
    return Task(id=task_id, statement=statement, checks=tuple(checks))


def _report(config: AgentConfig, *, promotes: int = 0, commits: int = 0,
            rollbacks: int = 0):
    rounds = ([SimpleNamespace(verdict="promote")] * promotes
              + [SimpleNamespace(verdict="noop", candidate_scores=[])])
    segments = ([SimpleNamespace(verdict="commit")] * commits
                + [SimpleNamespace(verdict="rollback")] * rollbacks)
    return SimpleNamespace(rounds=rounds, segments=segments, final_config=config)


def _outcome(meta_score: float, baseline: float = 0.0, *,
             policy: MetaPolicy | None = None, config: AgentConfig | None = None,
             promotes: int = 0, commits: int = 0, rollbacks: int = 0) -> InnerOutcome:
    policy = policy or MetaPolicy()
    config = config or _cfg()
    return InnerOutcome(policy, _report(config, promotes=promotes, commits=commits,
                                        rollbacks=rollbacks),
                        baseline, meta_score, policy.cost())


class ScriptModel(ChatModel):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        self.calls += 1
        return ModelResult(content=self.replies.pop(0))


# --------------------------------------------------------------------------- #
# 1. MetaPolicy：过程配置的指纹 / 夹取 / 不可变更新
# --------------------------------------------------------------------------- #
def test_meta_policy_is_versioned_clamped_and_immutable():
    p1 = MetaPolicy()
    p2 = update_meta_policy(p1, improver="llm")
    assert p1.hash() != p2.hash(), "过程配置变化必须改变指纹"
    assert p1.cost() == 0 and p2.cost() == 1

    wild = MetaPolicy(judge="wat", improver="wat", gate_min_delta=-1,
                      val_min_delta=9, n_rollouts=0)
    safe = update_meta_policy(wild, judge="wat")
    assert safe.judge == "rule" and safe.improver == "rule"
    assert safe.gate_min_delta == 0.0 and safe.val_min_delta == 0.5
    assert safe.n_rollouts == 1

    assert update_meta_policy(p1, n_rollouts=9).n_rollouts == 3
    assert update_meta_policy(p1) is p1
    assert update_meta_policy(p1, unknown=1) is p1
    assert update_meta_policy(p1, n_rollouts=1) is p1, "无变化 → 原对象"

    restored = MetaPolicy.from_canonical({"judge": "llm", "n_rollouts": 99})
    assert restored.judge == "llm" and restored.n_rollouts == 3
    assert MetaPolicy.from_canonical(None) == MetaPolicy()


def test_process_factory_wires_rule_and_llm_backends():
    model = ScriptModel([])
    factory = ProcessFactory(model)
    assert isinstance(factory.judge(MetaPolicy()), RuleJudge)
    assert isinstance(factory.judge(MetaPolicy(judge="llm")), LLMJudge)
    assert isinstance(factory.improver(MetaPolicy()), TwoStageImprover)
    assert isinstance(factory.improver(MetaPolicy(improver="llm")), MetaLLMImprover)


# --------------------------------------------------------------------------- #
# 2. 复合改进器：一次调用同时改 prompt + structure
# --------------------------------------------------------------------------- #
async def test_meta_llm_improver_merges_prompt_and_structure():
    reply = json.dumps({
        "component": "format", "finding": "输出含禁止词",
        "prompt": {"final_format": "只输出数字，不要写步骤",
                   "task_frame": {"suffix": "谢谢"}},
        "structure": {"max_steps": 999, "tool_names": ["a"]},
    }, ensure_ascii=False)
    cfg = _cfg(base_tools=(_tool("a"), _tool("b")), max_steps=4)
    diag, cand = await MetaLLMImprover(ScriptModel([reply])).improve(
        cfg, _task(), SimpleNamespace(answer="步骤：42", tool_calls=(), steps=1,
                                      step_limit=4, tools_available=("a", "b"),
                                      tools_enabled=("a", "b"), subagent_enabled=False,
                                      score=0.5))

    assert diag.component == "format"
    assert cand.prompt.final_format == "只输出数字，不要写步骤"
    assert cand.prompt.task_frame.suffix == "谢谢"
    assert cand.max_steps == 12 and cand.tool_names == ("a",)
    assert cfg.max_steps == 4, "原 config 不变"

    diag, same = await MetaLLMImprover(ScriptModel(["不是 JSON"])).improve(
        cfg, _task(), SimpleNamespace(answer="0", tool_calls=(), steps=1, step_limit=4,
                                      tools_available=(), tools_enabled=(),
                                      subagent_enabled=False, score=0.0))
    assert diag.component == "ok" and same is cfg

    diag, noop = await MetaLLMImprover(ScriptModel([
        json.dumps({"component": "ok", "prompt": {}, "structure": {}})])).improve(
        cfg, _task(), SimpleNamespace(answer="0", tool_calls=(), steps=1, step_limit=4,
                                      tools_available=(), tools_enabled=(),
                                      subagent_enabled=False, score=0.0))
    assert noop is cfg, "空 patch → 原对象"


# --------------------------------------------------------------------------- #
# 3. 过程定位
# --------------------------------------------------------------------------- #
async def test_rule_meta_locator_components():
    locator = RuleMetaLocator()

    d = await locator.locate(_outcome(0.8, baseline=0.5, promotes=1, commits=1))
    assert d.component == "ok"

    d = await locator.locate(_outcome(0.5, baseline=0.5, promotes=1, rollbacks=1))
    assert d.component == "gate_too_permissive"

    d = await locator.locate(_outcome(0.5, baseline=0.5, promotes=0))
    assert d.component == "process_stuck"

    d = await locator.locate(_outcome(0.5, baseline=0.5, promotes=1, commits=0))
    assert d.component == "segment_blocked"

    d = await locator.locate(_outcome(0.4, baseline=0.5, promotes=1, commits=0))
    assert d.component in ("segment_blocked", "no_gain")


# --------------------------------------------------------------------------- #
# 4. 过程配置编辑：目标档位幂等
# --------------------------------------------------------------------------- #
async def test_rule_meta_proposer_target_levels_and_idempotence():
    proposer = RuleMetaProposer()
    stuck = MetaDiagnosis("process_stuck", "")
    p = MetaPolicy()

    p1 = await proposer.propose(p, stuck)
    assert p1.improver == "llm"
    p2 = await proposer.propose(p1, stuck)
    assert p2.judge == "llm"
    p3 = await proposer.propose(p2, stuck)
    assert p3.n_rollouts == 2
    assert await proposer.propose(p3, stuck) is p3

    gate = MetaDiagnosis("gate_too_permissive", "")
    g1 = await proposer.propose(p, gate)
    assert g1.val_min_delta == 0.05
    g2 = await proposer.propose(g1, gate)
    assert g2.gate_min_delta == 0.05
    assert await proposer.propose(g2, gate) is g2

    blocked = MetaDiagnosis("segment_blocked", "")
    b1 = await proposer.propose(g1, blocked)
    assert b1.val_min_delta == 0.0
    b2 = await proposer.propose(MetaPolicy(), blocked)
    assert b2.n_rollouts == 2

    assert await proposer.propose(p, MetaDiagnosis("ok", "")) is p


def test_meta_gate_requires_meta_val_gain():
    gate = MetaGate()
    base = _outcome(0.5, baseline=0.5)
    assert gate.decide(base, _outcome(0.6, baseline=0.5))[0] == "promote"
    assert gate.decide(base, _outcome(0.5, baseline=0.5))[0] == "reject"
    assert gate.decide(base, _outcome(0.4, baseline=0.5))[0] == "reject"


# --------------------------------------------------------------------------- #
# 5. 元环（fake runner）：stuck → 换 LLM improver → 元门禁 promote
# --------------------------------------------------------------------------- #
async def test_meta_rsi_promotes_llm_improver(tmp_path):
    base_policy = MetaPolicy()
    llm_policy = update_meta_policy(base_policy, improver="llm")
    cfg = _cfg()
    outcomes = {
        base_policy.hash(): _outcome(0.5, baseline=0.5, policy=base_policy, config=cfg),
        llm_policy.hash(): _outcome(0.8, baseline=0.5, policy=llm_policy, config=cfg),
    }

    async def runner(policy):
        return outcomes[policy.hash()]

    meta = MetaRsi(cfg, base_policy, ScriptModel(["42"]),
                   train=[_task()], val=[_task("v")], test=[_task("t", "test 题")],
                   runner=runner, max_rounds=1)
    report = await meta.run()

    assert report.rounds[0].diagnosis.component == "process_stuck"
    assert report.rounds[0].verdict == "promote"
    assert report.final_policy.improver == "llm"
    assert report.final.meta_score == 0.8
    assert abs(report.gain - 0.3) < 1e-9
    assert report.test_score == 1.0


async def test_meta_rsi_noop_when_candidate_does_not_improve():
    base_policy = MetaPolicy()
    llm_policy = update_meta_policy(base_policy, improver="llm")
    cfg = _cfg()
    outcomes = {
        base_policy.hash(): _outcome(0.5, baseline=0.5, policy=base_policy, config=cfg),
        llm_policy.hash(): _outcome(0.5, baseline=0.5, policy=llm_policy, config=cfg),
    }

    async def runner(policy):
        return outcomes[policy.hash()]

    meta = MetaRsi(cfg, base_policy, ScriptModel(["42"]), train=[_task()],
                   val=[_task("v")], test=[_task("t", "test 题")],
                   runner=runner, max_rounds=1)
    report = await meta.run()

    assert report.rounds[0].verdict == "reject"
    assert report.final_policy == base_policy
    assert report.gain == 0.0
    assert report.test_score == 1.0


# --------------------------------------------------------------------------- #
# 6. 端到端：规则 proposer 卡在 format；LLM 复合改进器改 prompt 解锁
# --------------------------------------------------------------------------- #
class PromptAwareModel(ChatModel):
    """看到"只输出数字"就答 42；否则带步骤。收到复合改进器 prompt 时返回 JSON patch。"""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.improver_calls = 0

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        system = "\n".join(m.content for m in messages if m.role == "system")
        user = next((m.content for m in messages if m.role == "user"), "")
        self.statements.append(user)
        if "复合改进器" in system:
            self.improver_calls += 1
            return ModelResult(content=json.dumps({
                "component": "format",
                "finding": "输出含禁止词「步骤」，需要格式指令",
                "prompt": {"final_format": "只输出数字，不要写步骤"},
            }, ensure_ascii=False))
        if "只输出数字" in system:
            return ModelResult(content="42")
        return ModelResult(content="步骤：42")


async def test_e2e_meta_switches_proposer_and_fixes_prompt(tmp_path):
    model = PromptAwareModel()
    train = [_task("t1", "请计算 40+2")]
    val = [_task("v1", "请计算 20+22")]
    meta_val = [_task("m1", "请计算 21+21")]
    test = [_task("test1", "请计算 19+23")]
    cfg = _cfg()

    meta = MetaRsi(cfg, MetaPolicy(), model, train=train, val=val, test=test,
                   meta_val=meta_val, storage_dir=str(tmp_path / "meta"),
                   memory_dir=str(tmp_path / "memory"),
                   inner_rounds=1, inner_val_every=1, max_rounds=1)
    report = await meta.run()

    assert report.rounds[0].diagnosis.component == "process_stuck"
    assert report.rounds[0].verdict == "promote"
    assert report.final_policy.improver == "llm"
    assert report.gain > 0
    assert report.test_score == 1.0
    assert model.improver_calls == 1
    assert report.final.report.final_config.prompt.final_format == "只输出数字，不要写步骤"

    # test 只在末尾碰到一次；内层 RSI 的所有 run 都用 test=[]。
    assert sum(1 for s in model.statements if "19+23" in s) == 1
    # 两次内层 run（base + llm）各做 baseline/final 两次 meta-val，末尾 test 一次。
    assert meta.meta_evals == 2 * (2 * len(meta_val)) + len(test)
