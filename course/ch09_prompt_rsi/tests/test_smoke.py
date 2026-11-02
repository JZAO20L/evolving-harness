"""L1 冒烟：真模型（deepseek-V4 @ ARK）把 ch09 的**模型驱动 eval/locate + 评估纪律**真跑一遍。

断言的**不是分数必涨**（真模型有抖动），而是闭环本身活着：train/val 基线 → 模型判分 → 模型定位 →
规则改写 → 轮级/段级门禁 → test 只评一次 → 版本登记。另有一条 System One 决策 API 试连
（beta/模型可用性未验证，失败即 skip）。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import ModelError, OpenAIChatModel
from rsi import (
    LLMJudge,
    LLMLocator,
    PromptPack,
    Rsi,
    RuleProposer,
    SystemOneClient,
    SystemOneJudge,
    Task,
    TwoStageImprover,
    VersionRegistry,
    load_split,
)

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
SYS_ONE_MODEL = "doubao-seed-2-1-lite-260915"


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
async def test_smoke_model_driven_eval_and_two_level_gate(tmp_path):
    api_key = _need_key()
    train = load_split("train")[:6]
    val = load_split("val")[:3]
    test = load_split("test")[:3]
    cfg_prompt = PromptPack(guidance="你是数学/常识助手：先在心里算准，再按题目要求作答。")
    from rsi import AgentConfig
    cfg = AgentConfig(name="base", prompt=cfg_prompt, max_steps=4)
    model = _make_model(api_key)
    storage = str(tmp_path / "versions")
    try:
        rsi = Rsi(
            cfg, model, train=train, val=val, test=test,
            judge=LLMJudge(model),
            improver=TwoStageImprover(LLMLocator(model), RuleProposer()),
            storage_dir=storage, n_rollouts=1, max_rounds=2, val_every=1,
        )
        report = await rsi.run()
    finally:
        await model.aclose()

    assert 0.0 <= report.baseline_train_score <= 1.0
    assert 0.0 <= report.baseline_val_score <= 1.0
    assert report.rounds, "应至少产生一轮改进"
    assert all(r.verdict in ("promote", "reject", "noop") for r in report.rounds), report.rounds
    # 冒烟用 val_every=1：每轮边界都结算一段（哪怕第一轮就 noop，也有一段 hold）。
    assert report.segments, "val_every=1 时每轮边界都应结算一段"
    assert all(s.verdict in ("commit", "rollback", "hold") for s in report.segments), report.segments
    assert report.final_version and report.final_config.hash() == report.final_version
    assert 0.0 <= report.test_score <= 1.0
    reg = VersionRegistry(storage)
    entries = reg.list()
    assert entries and entries[0]["verdict"] == "accepted", "首次登记应为基线 accepted"

    print(f"\n[smoke] baseline train={report.baseline_train_score:.2f} "
          f"val={report.baseline_val_score:.2f} test={report.test_score:.2f}")
    print("   train:", ", ".join(f"{s.task_id}={s.mean:.2f}" for s in report.baseline_train))
    print("   val:  ", ", ".join(f"{s.task_id}={s.mean:.2f}" for s in report.baseline_val))
    for r_ in report.rounds:
        print(f"   round {r_.round}: -> {r_.verdict} score={r_.score:.2f} delta={r_.delta:+.2f}")
    for s_ in report.segments:
        print(f"   segment {s_.segment} (round {s_.round}): {s_.verdict} "
              f"trainΔ={s_.train_delta:+.2f} valΔ={s_.val_delta:+.2f}")


@pytest.mark.smoke
async def test_smoke_systemone_decision_api():
    api_key = _need_key()
    client = SystemOneClient(
        model=os.getenv("EH_SYSTEMONE_MODEL", SYS_ONE_MODEL),
        api_key=api_key,
        max_retries=1,
    )
    try:
        try:
            answers = await client.ask(
                state={"task": "9×9 等于多少？只给数字。", "answer": "81"},
                questions={"only_number_v1": {
                    "type": "score",
                    "instructions": "回答与 rubric 的符合程度？",
                    "criteria": ["不符合", "部分符合", "完全符合"],
                }},
            )
        except ModelError as exc:
            pytest.skip(f"System One beta 不可用/模型未开通：{exc}")
    finally:
        await client.aclose()

    assert isinstance(answers, dict) and "only_number_v1" in answers
    judge = SystemOneJudge(FakeResponder(answers))
    score = await judge.score(Task("t", "x", checks=("81",)), _row("81"))
    assert 0.0 <= score <= 1.0


class FakeResponder:
    def __init__(self, answers):
        self.answers = answers

    async def ask(self, state, questions):
        return self.answers


def _row(answer):
    from rsi import RolloutResult
    return RolloutResult(task_id="t", answer=answer, finished=True)
