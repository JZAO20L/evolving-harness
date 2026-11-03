"""L0 离线：ch10 skills-rsi 的技能粒度定位 / 编辑 / 版本哈希 / 端到端演化。

全部用脚本化 Mock，不联网；真模型只在 ``test_smoke.py``（EH_SMOKE=1）里跑。
"""

import json
from pathlib import Path

import pytest
from agent import ChatModel, Message, ModelResult, Skill, SkillLibrary, ToolCall
from rsi import (
    AgentConfig,
    Diagnosis,
    PromptPack,
    RolloutResult,
    Rsi,
    RuleScorer,
    RuleSkillLocator,
    RuleSkillProposer,
    SkillEdit,
    SkillLLMImprover,
    Task,
    TwoStageImprover,
    VersionRegistry,
    apply_skill_edit,
    load_synth,
    rollout,
)

SKILL_A = Skill(name="skill-a", description="覆盖要点", instructions="回答必须逐字包含 42。",
                priority=3)
SKILL_B = Skill(name="skill-b", description="输出纪律", instructions="不要写计算过程。", priority=1)


def _config(*skills: Skill, name: str = "base") -> AgentConfig:
    return AgentConfig(name=name, prompt=PromptPack(guidance="先算准再答。"), skills=tuple(skills))


def _task(checks=("42",), task_id: str = "mth-1") -> Task:
    return Task(id=task_id, statement="题目", checks=tuple(checks))


def _result(answer: str, *, loaded: tuple[str, ...] = (), finished: bool = True,
            tool_calls: tuple[str, ...] = ()) -> RolloutResult:
    return RolloutResult(task_id="mth-1", answer=answer, finished=finished,
                         tool_calls=tool_calls, loaded_skills=loaded)


# --------------------------------------------------------------------------- #
# 1. 版本哈希 & 不可变编辑
# --------------------------------------------------------------------------- #
def test_canonical_serializes_full_skills_and_hash_tracks_instructions():
    s1 = Skill("a", "目录描述", "旧指令", 2)
    s2 = Skill("a", "目录描述", "新指令", 2)
    c1, c2 = _config(s1), _config(s2)
    assert c1.canonical()["skills"] == [
        {"name": "a", "description": "目录描述", "instructions": "旧指令", "priority": 2}]
    assert c1.hash() != c2.hash(), "只改指令也必须改变版本指纹"


def test_apply_skill_edit_add_rewrite_delete_is_immutable():
    base = _config(SKILL_A)
    added = apply_skill_edit(base, SkillEdit("add", "skill-c", "新技能", "必须给出 7。", 4))
    assert base.skills == (SKILL_A,), "原 config 不能被污染"
    assert [s.name for s in added.skills] == ["skill-a", "skill-c"]
    assert added.skills[1].instructions == "必须给出 7。"

    rewritten = apply_skill_edit(added, SkillEdit("rewrite", "skill-a", "", "回答必须逐字包含 42，且不写步骤。"))
    assert rewritten.skills[0].instructions.endswith("且不写步骤。")
    assert rewritten.skills[0].description == SKILL_A.description, "空字段沿用旧值"

    deleted = apply_skill_edit(rewritten, SkillEdit("delete", "skill-c"))
    assert [s.name for s in deleted.skills] == ["skill-a"]

    assert apply_skill_edit(base, SkillEdit("delete", "不存在")) is base
    assert apply_skill_edit(base, SkillEdit("rewrite", "不存在", "", "x")) is base
    assert apply_skill_edit(base, SkillEdit("add", "空", "", "")) is base


def test_apply_skill_edit_add_same_name_replaces():
    base = _config(SKILL_A)
    replaced = apply_skill_edit(base, SkillEdit("add", "skill-a", "新描述", "新指令", 9))
    assert replaced.skills[0].instructions == "新指令"
    assert replaced.skills[0].priority == 9


# --------------------------------------------------------------------------- #
# 2. skill 粒度定位
# --------------------------------------------------------------------------- #
async def test_rule_skill_locator_missing_and_misleading():
    locator = RuleSkillLocator()
    task = _task(("42",))

    d = await locator.locate(task, _result("0"), 0.0)
    assert d.component == "skill_missing" and d.skill == ""
    assert "42" in d.finding

    d = await locator.locate(task, _result("0", loaded=("skill-a",)), 0.0)
    assert d.component == "skill_misleading" and d.skill == "skill-a"

    d = await locator.locate(task, _result("0", loaded=("skill-a",)), 0.5)
    assert d.component == "skill_misleading"


async def test_rule_skill_locator_forbidden_and_terminal_states():
    locator = RuleSkillLocator()
    task = _task(("42", "~计算过程"))

    d = await locator.locate(task, _result("42 计算过程：41+1", loaded=("skill-b",)), 0.5)
    assert d.component == "skill_misleading" and d.skill == "skill-b"
    assert "计算过程" in d.finding

    d = await locator.locate(task, _result("42 计算过程：41+1"), 0.5)
    assert d.component == "format"

    assert (await locator.locate(task, _result(""), 0.0)).component == "skip"
    assert (await locator.locate(task, _result("x", finished=False), 0.5)).component == "skip"
    assert (await locator.locate(task, _result("42"), 1.0)).component == "ok"


# --------------------------------------------------------------------------- #
# 3. 规则 proposer：缺失 → 新增；误导 → 改写（幂等）
# --------------------------------------------------------------------------- #
async def test_rule_skill_proposer_adds_then_rewrites_idempotently():
    task = _task(("42",))
    r = _result("0")
    diag = Diagnosis("skill_missing", "缺 42", task.id, 0.0)
    proposer = RuleSkillProposer()
    base = _config()

    cand = await proposer.propose(base, task, r, diag)
    assert base.skills == ()
    assert len(cand.skills) == 1
    assert cand.skills[0].name == "skill-mth-1"
    assert "必须逐字包含以下要点：42" in cand.skills[0].instructions

    diag2 = Diagnosis("skill_misleading", "技能没帮上", task.id, 0.0, "skill-a")
    r2 = _result("0", loaded=("skill-a",))
    rewritten = await proposer.propose(_config(SKILL_A), task, r2, diag2)
    assert "必须逐字包含以下要点：42" in rewritten.skills[0].instructions
    again = await proposer.propose(rewritten, task, r2, diag2)
    assert again is rewritten, "同一补充重复应用必须是 noop"

    unchanged = await proposer.propose(_config(SKILL_A), task, r, Diagnosis("ok", ""))
    assert unchanged.skills == (SKILL_A,)


async def test_rule_skill_proposer_forbidden_rewrites_constraint():
    task = _task(("42", "~计算过程"))
    r = _result("42 计算过程", loaded=("skill-b",))
    diag = Diagnosis("skill_misleading", "输出禁词", task.id, 0.5, "skill-b")
    cand = await RuleSkillProposer().propose(_config(SKILL_B), task, r, diag)
    assert "不得出现以下内容：计算过程" in cand.skills[0].instructions


# --------------------------------------------------------------------------- #
# 4. 模型驱动 improver（一次调用完成定位 + 编辑）
# --------------------------------------------------------------------------- #
class ScriptModel(ChatModel):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        self.calls += 1
        return ModelResult(content=self.replies.pop(0))


async def test_skill_llm_improver_add_and_delete():
    task = _task(("42",))
    r = _result("0")
    add_json = json.dumps({"component": "skill_missing", "finding": "缺 42", "skill": "skill-new",
                           "action": "add", "description": "覆盖 42", "instructions": "必须写 42。"},
                          ensure_ascii=False)
    diag, cand = await SkillLLMImprover(ScriptModel([add_json])).improve(_config(), task, r)
    assert diag.component == "skill_missing" and diag.skill == "skill-new"
    assert [s.name for s in cand.skills] == ["skill-new"]

    delete_json = json.dumps({"component": "skill_misleading", "finding": "带偏", "skill": "skill-a",
                              "action": "delete"})
    diag, cand = await SkillLLMImprover(ScriptModel([delete_json])).improve(_config(SKILL_A), task, r)
    assert cand.skills == ()

    bad = await SkillLLMImprover(ScriptModel(["不是 JSON"])).improve(_config(SKILL_A), task, r)
    assert bad[0].component == "ok" and bad[1].skills == (SKILL_A,)

    none_json = json.dumps({"component": "ok", "action": "none"})
    diag, cand = await SkillLLMImprover(ScriptModel([none_json])).improve(_config(SKILL_A), task, r)
    assert cand.skills == (SKILL_A,)


# --------------------------------------------------------------------------- #
# 5. rollout trace：loaded_skills 只认当前 config 里真实加载成功的技能
# --------------------------------------------------------------------------- #
async def test_rollout_records_loaded_skills():
    skill = Skill("skill-a", "覆盖要点", "回答必须逐字包含 42。", 3)

    class LoadThenAnswer(ChatModel):
        def __init__(self) -> None:
            self.step = 0

        async def complete(self, messages, tools=None):
            self.step += 1
            if self.step == 1:
                return ModelResult(content="", tool_calls=[
                    ToolCall("c1", "load_skill", {"name": "skill-a"}),
                    ToolCall("c2", "load_skill", {"name": "不存在"}),
                ])
            return ModelResult(content="42")

    r = await rollout(_config(skill), _task(("42",)), LoadThenAnswer())
    assert r.answer == "42"
    assert r.loaded_skills == ("skill-a",), "未知技能加载失败，不能记入"
    assert "load_skill" in r.tool_calls

    # 直接用 AgentLoop 也验证同一条 trace 缝：目录确实进了 system（显式 system 下由 rollout 组合）。
    lib = SkillLibrary(skill)
    assert "skill-a" in lib.list_summary()


# --------------------------------------------------------------------------- #
# 6. 版本回滚恢复完整技能
# --------------------------------------------------------------------------- #
def test_version_registry_roundtrips_full_skills(tmp_path):
    reg = VersionRegistry(tmp_path)
    cfg = _config(SKILL_A, SKILL_B)
    version = reg.register(cfg, score=0.5, verdict="accepted")
    restored = reg.rollback(version)
    assert restored.hash() == cfg.hash()
    assert [s.instructions for s in restored.skills] == [s.instructions for s in cfg.skills]

    # ch09 旧登记只有技能名 → 无法还原指令，跳过但不炸。
    legacy = {"version": "legacy", "config": {"name": "x", "max_steps": 8,
              "prompt": {"guidance": "g", "task_frame": {"prefix": "", "suffix": "",
                                                         "exemplars": []}, "final_format": ""},
              "skills": ["skill-a"], "base_tools": []}, "score": 0.0, "verdict": "accepted"}
    with (tmp_path / "versions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(legacy) + "\n")
    old = reg.rollback("legacy")
    assert old.skills == () and old.prompt.guidance == "g"


# --------------------------------------------------------------------------- #
# 7. synth 数据加载
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain,count", [("math", 30), ("tooluse", 30)])
def test_load_synth(domain, count):
    path = Path(__file__).resolve().parents[3] / "dataset" / "synth" / domain / "train.jsonl"
    if not path.exists():
        pytest.skip(f"缺少合成数据 {path}")
    tasks = load_synth(domain, "train")
    assert len(tasks) == count
    assert all(isinstance(t, Task) and t.checks for t in tasks[:3])
    with pytest.raises(ValueError):
        load_synth("unknown", "train")
    with pytest.raises(ValueError):
        load_synth("math", "dev")


# --------------------------------------------------------------------------- #
# 8. 端到端：基线无技能 → 定位缺失 → 新增技能 → promote + commit
# --------------------------------------------------------------------------- #
class SkillAwareModel(ChatModel):
    """回答由「当前 config 的技能目录里有没有 skill-mth-1」决定：有 → 42，没有 → 0。"""

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        system = messages[0].content
        return ModelResult(content="42" if "skill-mth-1" in system else "0")


async def test_e2e_skill_add_promotes_and_commits(tmp_path):
    task = _task(("42",))
    val = Task(id="mth-val", statement="题目", checks=("42",))
    test = Task(id="mth-test", statement="题目", checks=("42",))
    cfg = _config()
    model = SkillAwareModel()
    rsi = Rsi(cfg, model, train=[task], val=[val], test=[test],
              improver=TwoStageImprover(RuleSkillLocator(), RuleSkillProposer()),
              storage_dir=str(tmp_path / "versions"), max_rounds=1, val_every=1)

    report = await rsi.run()

    assert report.baseline_train_score == 0.0
    assert report.rounds[0].verdict == "promote"
    assert [s.name for s in report.final_config.skills] == ["skill-mth-1"]
    assert "必须逐字包含以下要点：42" in report.final_config.skills[0].instructions
    assert report.segments[0].verdict == "commit"
    assert report.test_score == 1.0


class ValBlindModel(ChatModel):
    """train 任务看技能（有则答 42），val 任务永远答 0 → 段级 rollback。"""

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        user = messages[-1].content
        if "VAL" in user:
            return ModelResult(content="0")
        return ModelResult(content="42" if "skill-mth-1" in messages[0].content else "0")


async def test_e2e_skill_add_rolls_back_when_val_does_not_improve(tmp_path):
    train = Task(id="mth-1", statement="TRAIN", checks=("42",))
    val = Task(id="mth-val", statement="VAL", checks=("42",))
    test = Task(id="mth-test", statement="TEST", checks=("42",))
    rsi = Rsi(_config(), ValBlindModel(), train=[train], val=[val], test=[test],
              improver=TwoStageImprover(RuleSkillLocator(), RuleSkillProposer()),
              storage_dir=str(tmp_path / "versions"), max_rounds=1, val_every=1)
    report = await rsi.run()
    assert report.rounds[0].verdict == "promote"
    assert report.segments[0].verdict == "rollback"
    assert report.final_config.skills == (), "回滚必须连技能一起回退"
    assert report.test_score == 0.0


async def test_scorer_still_works_for_skill_checks():
    task = _task(("42", "~计算过程"))
    scorer = RuleScorer()
    assert scorer.score(task, _result("42")) == 1.0
    assert scorer.score(task, _result("42 计算过程")) == 0.5
