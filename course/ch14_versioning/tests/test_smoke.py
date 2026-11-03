"""L1 冒烟：真模型（deepseek-V4 @ ARK）+ 真 rollout 跑 VersionedRsi。

一个 ``@pytest.mark.smoke``：先评估基线，再让 proposer 改一次 prompt，经合并 / 门控后由
``AgentRepo`` 决定提交或留在基线。真实模型可以合法 ``hold/rollback``，不强制 commit。

离线默认跳过：需 EH_SMOKE=1 + ARK key（key 只从环境变量读，可用 ``source .env`` 载入）。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import OpenAIChatModel, Tool
from rsi import AgentConfig, MemoryPolicy, PromptPack, RuleJudge, Task, VersionedRsi
from versions import AgentRepo, ChangeSet

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


def _task(task_id: str) -> Task:
    return Task(id=task_id, statement="请调用 vault 工具获取访问码，并只回答访问码。",
                checks=("QUASAR-917",))


def _config(base_tools: tuple, *, final_format: str = "") -> AgentConfig:
    return AgentConfig(
        name="versioned-smoke",
        prompt=PromptPack(guidance="回答访问码前必须先调用 vault 工具。",
                          final_format=final_format),
        max_steps=4,
        base_tools=base_tools,
        memory=MemoryPolicy(recall_enabled=False, store_full_turns=False,
                            profile_enabled=False),
    )


@pytest.mark.smoke
async def test_smoke_versioned_rsi_records_decisions(tmp_path):
    model = _make_model(_need_key())
    vault = _vault_tool()
    base = _config((vault,))
    improved = _config((vault,), final_format="只回答访问码本身。")
    repo = AgentRepo(tmp_path / "repo")

    async def propose(current: AgentConfig, _scores):
        return ChangeSet.from_config(current, improved)

    versioned = VersionedRsi(base, model, repo=repo, train=[_task("tr1"), _task("tr2")],
                             val=[_task("va1"), _task("va2")], proposers=[propose],
                             judge=RuleJudge(), n_rollouts=1, max_rounds=1,
                             memory_dir=str(tmp_path / "memory"))
    report = await versioned.run()

    assert report.head, "空仓库基线提交后必须有 HEAD"
    assert repo.get(report.head) is not None
    assert report.outcomes[0].verdict in {"commit", "rollback", "hold", "conflict", "noop"}
    assert report.head_config.hash() == repo.checkout(
        report.head, base_tools=(vault,)).hash()
    if report.outcomes[0].verdict == "commit":
        assert repo.get(report.head).verdict == "commit"
        assert repo.get(report.head).parents
    print(f"\n[smoke] verdict={report.outcomes[0].verdict} head={report.head} "
          f"evals={versioned.evals}")
