"""L1 冒烟：真模型（deepseek-V4 @ ARK）把 ch13 的元环真跑一遍。

- 用「秘密只在工具观察里」的 vault 任务跑 ``MetaRsi``：内层规则流程应自己修好步数，
  元环观察到 meta-val 提升后不折腾流程，末尾 test 只评一次。
- 用 ``MetaLLMImprover`` 对一条格式失败 trace 做一次真实 JSON 复合改写。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import OpenAIChatModel, Tool
from rsi import (
    COMPONENTS,
    AgentConfig,
    MemoryPolicy,
    MetaLLMImprover,
    MetaPolicy,
    MetaRsi,
    PromptPack,
    RolloutResult,
    Task,
)

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_ROOT = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    env_file = _ROOT / ".env"
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


def _vault_tool(secret: str = "QUASAR-917") -> Tool:
    return Tool(name="vault", description="获取当前访问码；回答访问码前必须调用此工具",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args: f"访问码：{secret}")


def _vault_task(task_id: str) -> Task:
    return Task(id=task_id,
                statement="请调用 vault 工具获取当前访问码，并原样输出访问码。",
                checks=("QUASAR-917",))


def _config(**changes) -> AgentConfig:
    base = {"name": "base",
            "prompt": PromptPack(guidance="必须先调用 vault 工具获取访问码，再原样输出访问码。"),
            "max_steps": 1,
            "memory": MemoryPolicy(recall_enabled=False, store_full_turns=False,
                                   profile_enabled=False)}
    base.update(changes)
    return AgentConfig(**base)


@pytest.mark.smoke
async def test_smoke_meta_rsi_observes_inner_process(tmp_path):
    model = _make_model(_need_key())
    cfg = _config(base_tools=(_vault_tool(),))
    meta = MetaRsi(
        cfg, MetaPolicy(), model,
        train=[_vault_task("t1"), _vault_task("t2")],
        val=[_vault_task("v")], meta_val=[_vault_task("mv")],
        test=[_vault_task("test")],
        storage_dir=str(tmp_path / "meta"), memory_dir=str(tmp_path / "memory"),
        inner_rounds=1, inner_val_every=1, max_rounds=1)
    try:
        report = await meta.run()
    finally:
        await model.aclose()

    assert 0.0 <= report.baseline.meta_score <= 1.0
    assert report.rounds and report.rounds[0].verdict in ("promote", "reject", "noop")
    assert report.test_score == 1.0, "最终 agent 应能拿到工具里的秘密"
    assert report.final.report.final_config.max_steps > 1
    assert meta.meta_evals >= 1

    print(f"\n[smoke] meta baseline={report.baseline.meta_score:.2f} "
          f"final={report.final.meta_score:.2f} gain={report.gain:+.2f} "
          f"test={report.test_score:.2f}")
    print(f"   round: {report.rounds[0].diagnosis.component} → {report.rounds[0].verdict}")
    print(f"   final policy: {report.final_policy.canonical()} cost={report.final_policy.cost()}")
    print(f"   final agent structure: max_steps="
          f"{report.final.report.final_config.max_steps} "
          f"tools={[t.name for t in report.final.report.final_config.enabled_tools()]}")


@pytest.mark.smoke
async def test_smoke_meta_llm_improver_on_format_failure(tmp_path):
    model = _make_model(_need_key())
    task = Task(id="fmt-smoke", statement="40+2 等于多少？只输出数字，不要写步骤。",
                checks=("42", "~步骤"))
    cfg = _config(prompt=PromptPack(guidance="先算再答。"), base_tools=(), max_steps=2)
    failure = RolloutResult(task_id=task.id, answer="步骤：42", finished=True,
                            steps=1, step_limit=2)
    try:
        diag, cand = await MetaLLMImprover(model).improve(cfg, task, failure)
    finally:
        await model.aclose()

    assert diag.component in COMPONENTS, diag
    assert isinstance(cand, AgentConfig)
    assert 2 <= cand.max_steps <= 12
    print(f"\n[smoke] meta-llm-improver component={diag.component} "
          f"final_format={cand.prompt.final_format!r} "
          f"max_steps={cand.max_steps} finding={diag.finding!r}")
