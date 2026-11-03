"""L1 冒烟：真模型（deepseek-V4 @ ARK）把 ch10 的 skill 粒度演化链路真跑一遍。

断言**不是技能一定被新增/分数一定涨**（真模型在这些算式题上可能 baseline 已满），而是闭环活着：
synth math 基线 → 规则 skill 定位 → 规则技能编辑 → 轮级/段级门禁 → test 只评一次 → 版本登记。
另一条用真模型走 ``SkillLLMImprover`` 的 JSON 编辑路径（解析失败也允许保守不改）。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import OpenAIChatModel
from rsi import (
    COMPONENTS,
    AgentConfig,
    PromptPack,
    Rsi,
    RuleSkillLocator,
    RuleSkillProposer,
    SkillLLMImprover,
    Task,
    TwoStageImprover,
    VersionRegistry,
    load_synth,
    rollout,
)

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    for key in ("ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


def _need_key() -> str:
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    key = os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")
    if not key:
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")
    return key


def _make_model(api_key: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        base_url=os.getenv("EH_BASE_URL", ARK_BASE),
        model=os.getenv("EH_MODEL", DEEPSEEK),
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )


@pytest.mark.smoke
async def test_smoke_skill_loop_on_synth_math(tmp_path):
    model = _make_model(_need_key())
    train = load_synth("math", "train")[:3]
    val = load_synth("math", "val")[:2]
    test = load_synth("math", "test")[:2]
    cfg = AgentConfig(name="base", prompt=PromptPack(
        guidance="你是算式题助手：先算准，再严格按题目要求输出。"), max_steps=3)
    storage = str(tmp_path / "versions")
    try:
        report = await Rsi(
            cfg, model, train=train, val=val, test=test,
            improver=TwoStageImprover(RuleSkillLocator(), RuleSkillProposer()),
            storage_dir=storage, n_rollouts=1, max_rounds=1, val_every=1,
        ).run()
    finally:
        await model.aclose()

    assert 0.0 <= report.baseline_train_score <= 1.0
    assert 0.0 <= report.baseline_val_score <= 1.0
    assert report.rounds and report.rounds[0].verdict in ("promote", "reject", "noop")
    assert report.segments and report.segments[0].verdict in ("commit", "rollback", "hold")
    assert report.final_version == report.final_config.hash()
    assert 0.0 <= report.test_score <= 1.0
    for skill in report.final_config.skills:
        assert skill.name and skill.description and skill.instructions

    entries = VersionRegistry(storage).list()
    assert entries and entries[0]["verdict"] == "accepted"

    print(f"\n[smoke] baseline train={report.baseline_train_score:.2f} "
          f"val={report.baseline_val_score:.2f} test={report.test_score:.2f}")
    print(f"   round: {report.rounds[0].verdict} (delta={report.rounds[0].delta:+.2f})")
    print(f"   segment: {report.segments[0].verdict} "
          f"(train_delta={report.segments[0].train_delta:+.2f}, "
          f"val_delta={report.segments[0].val_delta:+.2f})")
    print(f"   final skills: {[s.name for s in report.final_config.skills] or '（无）'}")


@pytest.mark.smoke
async def test_smoke_skill_llm_improver_on_failure_trace(tmp_path):
    model = _make_model(_need_key())
    task = Task(id="mth-smoke", statement="3 加 4 等于多少？",
                checks=("7", "~计算过程"))
    cfg = AgentConfig(name="base", prompt=PromptPack())
    try:
        r = await rollout(cfg, task, model)
        r.score = 0.0
        diag, cand = await SkillLLMImprover(model).improve(cfg, task, r)
    finally:
        await model.aclose()

    assert diag.component in COMPONENTS, diag
    assert isinstance(cand, AgentConfig)
    assert all(s.name and s.instructions for s in cand.skills)
    print(f"\n[smoke] llm-improver component={diag.component} skill={diag.skill!r} "
          f"finding={diag.finding!r}")
