"""L1 冒烟：用真模型（deepseek-v4-flash-ga-260731 @ ARK /api/v3/chat/completions）跑一遍 ch02 的 ReAct 循环。

离线默认跳过：需要 `EH_SMOKE=1` 且拿到 ARK key（.env 的 ARK_API_KEY / EH_API_KEY）。
松断言：能连通、循环能终结（finish 或步数耗尽）、模型非空。真模型可能夹叙述/代码块，
正好顺带验证 parse_act 的容错。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import AgentLoop, OpenAIChatModel

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


@pytest.mark.smoke
async def test_smoke_react_loop_real_deepseek():
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    key = os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY") or ""
    if not key:
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")

    model = OpenAIChatModel(
        base_url=os.getenv("EH_BASE_URL", ARK_BASE),
        model=os.getenv("EH_MODEL", DEEPSEEK),
        api_key=key,
        timeout=60.0,
        max_retries=2,
    )
    loop = AgentLoop(model, max_steps=4)
    try:
        res = await loop.run("用中文回答：1+1 等于多少？只给结论。")
    finally:
        await model.aclose()

    assert res.acts, "循环应至少选中一个 act"
    assert res.answer.strip(), "模型应产出非空 answer"
    assert res.usage.total_tokens > 0
    print(f"\n[smoke] finished={res.finished} steps={res.steps} "
          f"usage={res.usage.total_tokens}t acts={[a.action_type for a in res.acts]} "
          f"-> {res.answer!r}")