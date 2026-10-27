"""L1 冒烟：真 deepseek-V4 @ ARK 跑 ch03 的"工具驱动 ReAct 循环"，并**真读仓库文件**。

注册 ``read_file``（只读、超长截断），让模型用原生 function calling 读 ``agent.py``，
再回答其中定义了哪些对象——验证「function calling 真正连到世界」。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import AgentLoop, OpenAIChatModel, ToolRegistry, builtin_read_tools

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"

AGENT_PY = str(Path(__file__).resolve().parent.parent / "agent.py")


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
async def test_smoke_tool_loop_reads_local_file():
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
        timeout=90.0,
        max_retries=2,
    )

    registry = ToolRegistry()
    for t in builtin_read_tools():
        registry.register(t)

    loop = AgentLoop(model, registry=registry, max_steps=5)
    try:
        res = await loop.run(
            f"先用 read_file 工具读取 {AGENT_PY} 的前若干字符，"
            "再告诉我：这个文件里定义了哪些类（class）和哪些工具函数。"
        )
    finally:
        await model.aclose()

    assert res.answer.strip(), "模型应产出最终文本"
    assert res.usage.total_tokens > 0
    used = [t.name for t in res.tool_calls]
    print(f"\n[smoke] finished={res.finished} steps={res.steps} usage={res.usage.total_tokens}t "
          f"tools={used} -> {res.answer!r}")