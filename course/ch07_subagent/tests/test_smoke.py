"""L1 冒烟：真 deepseek-V4 @ ARK 跑 ch07 的「sub-agent 委派」闭环。

真冒烟：给一个多面任务，真模型调 `subagent` 把其中一个子问题**委派给独立的子循环**去解决，
子返回 `[子代理完成] <最终文本>` 作为一条 tool observation，父循环引用它继续给结论。

离线默认跳过：需 EH_SMOKE=1 + ARK key。子代理独立 run 会叠加真模型请求，更费额度。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import AgentLoop, Memory

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


def _skip_unless_smoke() -> None:
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    if not (os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")):
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")


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
async def test_smoke_subagent_delegates_and_concludes(tmp_path):
    _skip_unless_smoke()
    model = _make_model()
    try:
        mem = Memory(model, storage_dir=str(tmp_path / "m"))
        loop = AgentLoop(model, memory=mem, subagent_enabled=True, max_steps=10)
        res = await loop.run(
            "请先**用一个子代理**调研『RL 红队』并给它一个核心要点（调用 subagent 委派，"
            "子代理在独立上下文里完成），然后在你的最终回答里引用子代理的要点，再给一个一句话结论。"
        )
    finally:
        await model.aclose()

    assert res.answer.strip(), "父应产出最终结论"
    assert any(t.name == "subagent" for t in res.tool_calls), "父应真调用 subagent 委派"
    assert res.finished, "应在其预算内收官"
    print(f"\n[smoke] steps={res.steps} tools={[t.name for t in res.tool_calls]} "
          f"usage={res.usage.total_tokens} -> {res.answer[:80]!r}\n")
    assert (tmp_path / "m" / "memory.jsonl").exists(), "共用存储：memory.jsonl 应落盘（子 finalize 写入）"