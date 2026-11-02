"""L1 冒烟：真模型（deepseek-V4 @ ARK）把 ch08 的 **RSI 外环**整轮真跑一遍。

断言的**不是分数必涨**（真模型有抖动），而是闭环本身活着：baseline 打分 → 定位 → 提议 → 门禁给出
verdict → 至少登记一个版本。任务集里放几条带 rubric 的数学题作为 verifier 能判的改进面。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsi import AgentConfig, PromptPack, Rsi, RuleScorer, Task, VersionRegistry

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


def _make_model():
    from agent import OpenAIChatModel
    return OpenAIChatModel(
        base_url=os.getenv("EH_BASE_URL", ARK_BASE),
        model=os.getenv("EH_MODEL", DEEPSEEK),
        api_key=os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY") or "",
        timeout=120.0,
        max_retries=2,
    )


@pytest.mark.smoke
async def test_smoke_rsi_improve_roundtrips(tmp_path):
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    if not (os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")):
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")

    tasks = [
        Task("m1", "计算 23×7，只给一个数字作为结论。", checks=("161",)),
        Task("m2", "计算 12+9，只给一个数字作为结论。", checks=("21",)),
    ]
    cfg = AgentConfig(name="base", prompt=PromptPack(guidance="你是数学助手，先在心里算准，再只给一个数字结论。"),
                      max_steps=5)
    model = _make_model()
    storage = str(tmp_path / "versions")
    try:
        rsi = Rsi(cfg, model, tasks, scorer=RuleScorer(), storage_dir=storage, n_rollouts=1)
        report = await rsi.improve(max_rounds=2)
    finally:
        await model.aclose()

    assert report.baseline_score is not None and 0.0 <= report.baseline_score <= 1.0
    assert report.rounds, "应至少产生一轮改进"
    assert all(r.verdict in ("promote", "reject", "noop") for r in report.rounds), report.rounds
    reg = VersionRegistry(storage)
    entries = reg.list()
    assert entries, "版本册不应为空"
    assert entries[0]["verdict"] == "accepted", "首次登记应为基线 accepted"
    print(f"\n[smoke] baseline={report.baseline_score:.2f} rounds={len(report.rounds)} "
          f"final_version={report.final_version or '<none>'}")
    for r_ in report.rounds:
        print(f"   round {r_.round}: -> {r_.verdict} score={r_.score:.2f} delta={r_.delta:+.2f}")