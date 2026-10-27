"""L1 冒烟：真 deepseek-V4 @ ARK + 真 arxiv 网络，跑 ch04 的"磁盘技能 + 渐进披露"闭环。

从 ``skills/arxiv_search.skill.md`` 用 ``SkillLibrary.load_dir`` 加载技能（frontmatter 给
name/desc/priority，正文给 instructions），配上 ``web_fetch`` 只读网络工具：
模型看目录决定 ``load_skill`` 加载 arxiv 技能 -> 按技能指令用 ``web_fetch`` 真打 arxiv API
搜一轮论文元数据 -> 整理成条目作答。验证技能 = 真实可复用指令包 + 只读网络闭环。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import AgentLoop, OpenAIChatModel, SkillLibrary, ToolRegistry, builtin_read_tools

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"

SKILLS_DIR = str(Path(__file__).resolve().parent.parent / "skills")


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
async def test_smoke_load_arxiv_skill_then_arxiv_search():
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
    for t in builtin_read_tools():          # 含 web_fetch：只读网络 GET
        registry.register(t)

    skills = SkillLibrary.load_dir(SKILLS_DIR)   # 从磁盘加载 arxiv 技能
    loop = AgentLoop(model, registry=registry, skills=skills, max_steps=8)
    try:
        res = await loop.run(
            "用 arxiv 技能帮我搜一下 reinforcement learning 与 red-teaming/language model "
            "相关的论文，列出前 3 条（标题 + 年份），用中文。"
        )
    finally:
        await model.aclose()

    assert res.answer.strip(), "模型应产出最终文本"
    assert "load_skill" in [t.name for t in res.tool_calls], "模型应 load_skill 加载 arxiv 技能"
    assert any(t.name == "web_fetch" for t in res.tool_calls), "模型应按技能用 web_fetch 真查 arxiv"
    print(f"\n[smoke] finished={res.finished} steps={res.steps} usage={res.usage.total_tokens}t "
          f"tools={[t.name for t in res.tool_calls]} -> {res.answer!r}\n")