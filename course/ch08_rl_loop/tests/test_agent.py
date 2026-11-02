"""ch08 rl-loop 的 L0 离线自测：`rsi.py` 各对象全 Mock（不联网）。

分层：prompt 面 / config 哈希 / reward(Rule+Judge+Pair) / credit-assignment / proposer /
门禁 / 版本登记与回滚 / Rsi 外环端到端。真模型冒烟在 test_smoke.py。
"""

import asyncio

import pytest
from agent import MockChatModel, ModelResult, Usage
from rsi import (
    AgentConfig,
    Diagnosis,
    Gate,
    JudgeScorer,
    Locator,
    PairScorer,
    PromptPack,
    Proposer,
    RolloutResult,
    Rsi,
    RuleScorer,
    Task,
    TaskFrame,
    TaskScore,
    VersionRegistry,
    _parse_score,
    aggregate,
    evaluate,
    rollout,
)


def _result(content="", *, prompt=1, completion=1, tool_calls=None):
    return ModelResult(content, usage=Usage(prompt, completion), tool_calls=tool_calls)


def _pack(guidance="g", fmt="", prefix="", suffix="", exemplars=()):
    return PromptPack(guidance=guidance, final_format=fmt,
                      task_frame=TaskFrame(prefix=prefix, suffix=suffix, exemplars=exemplars))


def _ts(task_id="T1", mean=1.0):
    return TaskScore(task_id=task_id, mean=mean, best=mean, min_=mean)


# --- prompt 三件套：渲染 + 哈希随编辑改变 -------------------------------------------- #
def test_task_frame_render_prefix_suffix_exemplars():
    tf = TaskFrame(prefix="请认真回答：", suffix="（只给结论）",
                   exemplars=(("例任务", "例答案"),))
    out = tf.render("今天星期几")
    assert "请认真回答：" in out and out.endswith("（只给结论）")
    assert "例任务" in out and "任务：今天星期几" in out       # few-shot 在任务前


def test_prompt_render_system_includes_final_format():
    p = _pack(guidance="你是助手", fmt="只给一句话结论")
    s = p.render_system()
    assert s.startswith("你是助手") and "输出要求" in s and "只给一句话结论" in s
    # 无 final_format → 不夹"输出要求"
    assert "输出要求" not in _pack(guidance="你是助手").render_system()


def test_prompt_render_user_only_uses_statement_not_checks():
    p = _pack()
    task = Task("t1", "1+1", checks=("=2", "~=3"))             # checks 私有，不得漏给模型
    u = p.render_user(task.statement)
    assert u == "1+1" and "checks" not in u and "~=3" not in u  # 评测污染隔离


def test_agent_config_hash_stable_and_changes_on_prompt_edit():
    a = AgentConfig(prompt=_pack(guidance="g"), max_steps=4)
    b = AgentConfig(prompt=_pack(guidance="g"), max_steps=4)
    assert a.hash() == b.hash()                                 # 同配置 → 同版本
    b.prompt.guidance = "g 改"
    assert a.hash() != b.hash()                                 # 改 prompt → 新版本


# --- reward：RuleScorer（pointwise 确定性） ------------------------------------------- #
def test_rule_scorer_positive_negation_and_empty():
    sc = RuleScorer()
    assert sc.score(Task("t1", "x", checks=("2", "~3")), RolloutResult("t1", "结果是 2")) == 1.0  # 含 2、不含 3 → 满分
    assert sc.score(Task("t1", "x", checks=("2", "7")), RolloutResult("t1", "结果是 2")) == 0.5   # 缺 7 → 半
    assert sc.score(Task("t1", "x", checks=("~3",)), RolloutResult("t1", "结果是 3")) == 0.0      # 含禁止词 → 0
    assert sc.score(Task("t1", "x", checks=()), RolloutResult("t1", "给答案")) == 1.0             # 无 checks → 非空即 1
    assert sc.score(Task("t1", "x", checks=()), RolloutResult("t1", answer="")) == 0.0


async def test_judge_scorer_parses_and_uses_model(tmp_path):
    # 打分模型脚本化返回 "0.8" → JudgeScorer 剥出 0.8
    model = MockChatModel(script=[_result("得分 0.8")])
    r = RolloutResult("t1", answer="答案")
    score = await JudgeScorer().score(Task("t1", "x", ("要点",)), r, model)
    assert score == 0.8
    await model.aclose()


def test_parse_score_tolerates_noise():
    assert _parse_score("0.5 分") == 0.5
    assert _parse_score("满分 1") == 1.0
    assert _parse_score("无法判断") == 0.0
    assert _parse_score("得分 0") == 0.0


# --- pairwise：PairScorer 对组做差分 -------------------------------------------------- #
def test_pair_scorer_diff_preference():
    ps = PairScorer()
    base = [_ts("t1", 0.5), _ts("t2", 0.8)]
    cand = [_ts("t1", 0.9), _ts("t2", 0.7)]
    d = ps.diff(base, cand)
    assert d.preference == 1 and abs(d.delta - 0.15) < 1e-9       # (0.4-0.1)/2
    assert d.per_task[0] == ("t1", 0.4)
    assert abs(d.per_task[1][1] - (-0.1)) < 1e-9                  # 0.7-0.8 = -0.1000..009


# --- rollout：由 config 组装 AgentLoop 并跑 task --------------------------------------- #
async def test_rollout_runs_config_builds_loop():
    model = MockChatModel(script=[_result("最终答案")])
    cfg = AgentConfig(prompt=_pack(guidance="g", prefix="P:"), max_steps=3)
    r = await rollout(cfg, Task("t1", "任务A"), model)
    assert r.task_id == "t1" and r.answer == "最终答案" and r.finished
    assert r.usage.total_tokens >= 2
    await model.aclose()


async def test_evaluate_group_rollouts_mean_best_min():
    class _Rotate(MockChatModel):                       # 组内 3 次 rollout 给 1/0/1
        def __init__(self):
            super().__init__(script=[])
            self.i = 0
        async def complete(self, messages, tools=None):
            self.i += 1
            return _result("命中要点" if self.i % 2 else "我不清楚")   # 后者不含"要点" → miss

    model = _Rotate()
    tasks = [Task("t1", "x", checks=("要点",))]
    scores = await evaluate(tasks, AgentConfig(prompt=_pack()), model, RuleScorer(), n_rollouts=3)
    t = scores[0]
    assert t.mean == pytest.approx(2 / 3) and t.best == 1.0 and t.min_ == 0.0
    await model.aclose()


# --- credit assignment：Locator 归因到组件 ---------------------------------------------- #
def _R(ans="", finished=True, tools=()):
    return RolloutResult(task_id="T", answer=ans, finished=finished,
                         tool_calls=tuple(tools), score=float("nan"))


def test_locator_classes_failures_to_component():
    loc = Locator()
    task = Task("t", "x", checks=("要点", "~空谈"))
    assert loc.locate(task, _R(ans=""), 0.0).component == "skip"                       # 空答
    assert loc.locate(task, _R(ans="a", finished=False), 0.0).component == "skip"      # 未收尾
    assert loc.locate(task, _R(ans="结论：绝不空谈", tools=("read",)), 0.5).component == "format"  # 含禁词
    assert loc.locate(task, _R(ans="给个结论", tools=()), 0.5).component == "halluc"    # 缺要点且没查工具
    assert loc.locate(task, _R(ans="结论在此", tools=("read",)), 0.5).component == "frame"  # 有工具但缺要点
    assert loc.locate(task, _R(ans="有要点且不空谈"), 1.0).component == "ok"


# --- proposer：依 diagnosis 改对组件（不可变） ------------------------------------------- #
def test_proposer_format_edits_final_format_and_is_immutable():
    cfg = AgentConfig(prompt=_pack(guidance="g"))
    p = Proposer()
    cand = p.propose(cfg, Task("t", "x"), _R(ans="含禁词"), Diagnosis("format", "f", "t", 0.0))
    assert cand.prompt.final_format and "输出要求" in cand.prompt.render_system()
    assert cand.hash() != cfg.hash()                          # 提议产生新版本
    assert cfg.prompt.final_format == ""                       # 原 config 未被改（不可变）
    assert cand is not cfg


def test_proposer_others_edit_guidance_or_noop():
    p = Proposer()
    cfg = AgentConfig(prompt=_pack(guidance="g"))
    cand_h = p.propose(cfg, Task("t", "x"), _R(ans="a"), Diagnosis("halluc", "h", "t", 0.0))
    assert "调用工具" in cand_h.prompt.guidance                # halluc → guidance 规则
    noop = p.propose(cfg, Task("t", "x"), _R(ans="a"), Diagnosis("ok", "ok", "t", 1.0))
    assert noop is cfg                                          # ok/guidance/未知 → 不改


# --- gate：提升则 promote，回归则 reject，含不回归约束 -------------------------------------- #
def test_gate_promote_reject_and_regress():
    g = Gate()
    base = [_ts("t1", 0.5), _ts("t2", 0.5)]
    assert g.decide(base, [_ts("t1", 0.9), _ts("t2", 0.6)])[0] == "promote"
    assert g.decide(base, [_ts("t1", 0.4), _ts("t2", 0.6)])[0] == "reject"     # t1 回归
    # 不回归约束：整体升但最差 task 大幅回退 → 仍 reject
    g2 = Gate(regress_tol=0.05)
    assert g2.decide(base, [_ts("t1", 1.0), _ts("t2", 0.1)])[0] == "reject"


# --- 版本登记 / 回滚 ---------------------------------------------------------------------- #
def test_version_registry_register_list_latest(tmp_path):
    reg = VersionRegistry(str(tmp_path / "v"))
    c1 = AgentConfig(name="base", prompt=_pack(guidance="g"))
    v1 = reg.register(c1, score=0.5, verdict="accepted")
    c2 = AgentConfig(name="cand", prompt=_pack(guidance="g2"))
    v2 = reg.register(c2, score=0.8, verdict="promote", parent=v1)
    entries = reg.list()
    assert [e["verdict"] for e in entries] == ["accepted", "promote"]
    assert reg.get(v2)["parent"] == v1
    assert reg.latest()["version"] == v2


def test_version_registry_rollback_restores_config(tmp_path):
    reg = VersionRegistry(str(tmp_path / "v"))
    base = AgentConfig(name="base", prompt=_pack(guidance="g", fmt="旧格式"), max_steps=4)
    v1 = reg.register(base, score=0.5, verdict="accepted")
    cand = AgentConfig(name="cand", prompt=_pack(guidance="g2"), max_steps=8)
    reg.register(cand, score=0.3, verdict="reject", parent=v1)

    restored = reg.rollback(v1)
    assert restored.name == "base" and restored.prompt.guidance == "g"
    assert restored.prompt.final_format == "旧格式" and restored.max_steps == 4
    with pytest.raises(KeyError):
        reg.rollback("nope")


# --- Rsi 外环端到端：reward → locate → propose → gate 贯通 --------------------------------- #
class _FormatMock(MockChatModel):
    """关键：mock 的答案依赖 system 是否带 `输出要求`(final_format)。基线不满足 → 含禁词第；加规则后 → 合规。
    用这个确定性模拟"改 prompt 真能提分"，而不赌真模型。"""

    def __init__(self):
        super().__init__(script=[])

    async def complete(self, messages, tools=None):
        sys_text = next((m.content for m in messages if m.role == "system"), "")
        content = "结论" if ("输出要求" in sys_text) else "我们就空谈吧"
        return _result(content, prompt=1, completion=1)

    async def aclose(self):
        pass


async def test_rsi_improve_promotes_and_registers_version(tmp_path):
    tasks = [Task("t1", "给个结论", checks=("~空谈",))]
    cfg = AgentConfig(name="base", prompt=_pack(guidance="你是助手"), max_steps=3)
    rsi = Rsi(cfg, _FormatMock(), tasks, storage_dir=str(tmp_path / "v"))
    report = await rsi.improve(max_rounds=2)

    assert report.baseline_score == 0.0                    # 基线含禁词 → 0
    assert report.rounds and report.rounds[0].verdict == "promote"
    assert report.rounds[0].score == 1.0
    assert report.final_version is not None
    assert report.final_config.prompt.final_format         # 最终版本带上了 format 规则
    reg = VersionRegistry(str(tmp_path / "v"))
    verdicts = [e["verdict"] for e in reg.list()]
    assert verdicts[0] == "accepted" and verdicts[-1] == "promote"   # 基线 + 被采纳候选都在册


async def test_rsi_improve_rejects_when_no_gain_and_rollback_keeps_baseline(tmp_path):
    class _Stubborn(MockChatModel):                       # 改不改 prompt 都给同样错答案 → 门禁 reject
        def __init__(self):
            super().__init__(script=[])
        async def complete(self, messages, tools=None):
            return _result("我们就空谈吧")
        async def aclose(self):
            pass

    tasks = [Task("t1", "给个结论", checks=("~空谈",))]
    cfg = AgentConfig(name="base", prompt=_pack(guidance="你是助手"), max_steps=3)
    rsi = Rsi(cfg, _Stubborn(), tasks, storage_dir=str(tmp_path / "v"))
    report = await rsi.improve(max_rounds=2)

    assert report.baseline_score == 0.0
    assert report.final_config.hash() == cfg.hash()        # 没采纳 → 最终仍是基线
    assert all(r.verdict == "reject" for r in report.rounds)
    # 回滚到基线版本 → 还原基线 config
    reg = VersionRegistry(str(tmp_path / "v"))
    base_v = next(e["version"] for e in reg.list() if e["verdict"] == "accepted")
    restored = reg.rollback(base_v)
    assert restored.prompt.guidance == cfg.prompt.guidance


async def test_proposer_edits_then_verifier_scores_higher():
    """直接验证 reward→propose 的收益：同一 mock 下，候选 config 的 verify 分数应变高。"""
    tasks = [Task("t1", "给个结论", checks=("~空谈",))]
    cfg = AgentConfig(prompt=_pack(guidance="你是助手"), max_steps=3)
    diag = Diagnosis("format", "含禁词", "t1", 0.0)
    cand = Proposer().propose(cfg, tasks[0], RolloutResult("t1", "我们就空谈吧"), diag)
    base_score = aggregate(await evaluate(tasks, cfg, _FormatMock(), RuleScorer(), 1))
    cand_score = aggregate(await evaluate(tasks, cand, _FormatMock(), RuleScorer(), 1))
    assert base_score == 0.0 and cand_score == 1.0           # propose 后 reward 提升


async def test_input_injection_still_works_on_plain_agent_snapshot():
    """顺延证明：本课 agent.py 仍带 ch02 的 input_queue（agent 不是跑完才理人）。"""
    from agent import AgentLoop
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("改成先做 C")
    rec = MockChatModel(script=[_result("已按新指令")])
    loop = AgentLoop(rec, max_steps=5, input_queue=q)
    res = await loop.run("先做 A")
    assert res.finished and res.answer == "已按新指令"