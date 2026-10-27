"""L1 冒烟：真 deepseek-V4 @ ARK 跑 ch04 的"渐进披露技能加载"闭环。

注册一个 ``summarize_file`` 技能（指令 = 先 read_file 再按三节总结），配内置只读工具：
模型从目录（只见 name+desc，不见指令）里 **load_skill**，指令才注入 context；
随后真正用 read_file 读文件、按技能指令作答——验证渐进披露 + 加载后闭环。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import AgentLoop, OpenAIChatModel, Skill, SkillLibrary, ToolRegistry, builtin_read_tools

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


def _summarize_skill() -> Skill:
    return Skill(
        name="summarize_file",
        description="读取文件后按 类/函数/要点 三节给出结构化总结",
        instructions=(
            "流程：1) 用 read_file 读指定文件；2) 把内容整理成三节 —— 类 / 函数 / 要点；"
            "3) 每节用中文一句话概括。绝不编造文件中不存在的对象。"
        ),
        priority=10,
    )


@pytest.mark.smoke
async def test_smoke_skill_progressive_disclosure_loop():
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

    loop = AgentLoop(model, registry=registry, skills=SkillLibrary(_summarize_skill()),
                     max_steps=6)
    try:
        res = await loop.run(f"用技能总结 {AGENT_PY} 这个文件。")
    finally:
        await model.aclose()

    assert res.answer.strip(), "模型应产出最终文本"
    used = [t.name for t in res.tool_calls]
    assert "load_skill" in used, "模型应通过 load_skill 加载技能"
    print(f"\n[smoke] finished={res.finished} steps={res.steps} usage={res.usage.total_tokens}t "
          f"tools={used} -> {res.answer!r}")