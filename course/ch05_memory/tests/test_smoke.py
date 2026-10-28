"""L1 冒烟：真 deepseek-V4 @ ARK + 真网络/真模型改写，跑 ch05 的记忆闭环。

两条真冒烟：
1. **跨 run 自动回填**：run1 让真模型聊出一个词（「玄武」）并落盘成 transcript；run2 另起
   ``Memory`` + ``AgentLoop``（同一 ``storage_dir``）问同样这个词——验证**循环内置自动回填**
   从完整记忆文件 grep 出片段，注入 run2 的系统首条。
2. **模型改写压缩**：给一个**极小的 ``ContextBudget``**（max_len×threshold 很小），配上返回
   长文本的 ``big`` 工具，让 run1 中途真实撑爆预算、真 deepseek **模型改写**旧前缀成摘要——
   验证 ``mem.compacted``（工作记忆压缩发生过）。

离线默认跳过：需 EH_SMOKE=1 + ARK key。每个模型请求真花额度。
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import (
    AgentLoop,
    ContextBudget,
    Memory,
    Message,
    OpenAIChatModel,
    Tool,
    ToolRegistry,
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


def _make_rec() -> "_Rec":
    """真模型 + 记录每次 complete 的 messages（观察自动回填注入 / 压缩触发）。"""
    key = os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY") or ""
    return _Rec(
        base_url=os.getenv("EH_BASE_URL", ARK_BASE),
        model=os.getenv("EH_MODEL", DEEPSEEK),
        api_key=key,
        timeout=90.0,
        max_retries=2,
    )


class _Rec(OpenAIChatModel):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        return await super().complete(messages, tools)


def _skip_unless_smoke() -> None:
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    if not (os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")):
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")


@pytest.mark.smoke
async def test_smoke_cross_run_recall(tmp_path):
    _skip_unless_smoke()
    store_dir = str(tmp_path / "mem")

    # run1：真模型聊出「玄武」并落盘（MemoryStore 本轮完整记忆 -> transcript.md）
    r1 = _make_rec()
    m1 = Memory(r1, storage_dir=store_dir)
    try:
        res1 = await AgentLoop(r1, memory=m1, max_steps=4).run(
            "请记住：玄武是前辈口述里的一件法宝，保存到你的长期记忆里。")
    finally:
        await r1.aclose()
    assert res1.answer.strip()
    assert "玄武" in (tmp_path / "mem" / "memory.jsonl").read_text(), "run1 记忆应已落盘"

    # run2：另起 Memory + 循环，同一 storage_dir，问「玄武」——应从文件自动回填
    r2 = _make_rec()
    m2 = Memory(r2, storage_dir=store_dir)
    try:
        res2 = await AgentLoop(r2, memory=m2, max_steps=4).run("我之前提到的玄武是什么？")
    finally:
        await r2.aclose()
    assert res2.answer.strip()
    first = r2.calls[0]                                        # run2 首次模型调用的完整 messages
    assert any(m.role == "system" and "[从完整记忆检索到]" in m.content
               and "玄武" in m.content for m in first), "run2 应自动回填玄武片段"


@pytest.mark.smoke
async def test_smoke_compression_triggers_with_tiny_budget(tmp_path):
    _skip_unless_smoke()
    rec = _make_rec()
    registry = ToolRegistry()
    registry.register(Tool(name="big", description="返回一大段长文本，撑满工作记忆",
                           parameters={"type": "object", "properties": {}},
                           handler=lambda a: "x" * 2000))       # ~666 token 观察
    # 极小预算：一出现长观察/较长对话就超限 -> 真模型改写压缩旧前缀成摘要
    mem = Memory(rec, budget=ContextBudget(max_len=400, threshold=0.5, keep_recent=2),
                 storage_dir=str(tmp_path / "mem2"))
    try:
        res = await AgentLoop(rec, registry=registry, memory=mem, max_steps=6).run(
            "先调用 big 工具取一段长文本，然后用中文把它的要点概括给我，并补一两句话说明你记性很好。")
    finally:
        await rec.aclose()
    assert res.answer.strip()
    assert mem.compacted, "极小的 budget 应在中途真实触发模型改写压缩"
    events = [json.loads(line) for line in (tmp_path / "mem2" / "memory.jsonl").read_text().splitlines()]
    assert any("压缩摘要" in e["content"] for e in events), "模型改写摘要应落盘"
    assert any(e["role"] == "assistant" and res.answer[-12:] in e["content"] for e in events)
    print(f"\n[smoke] recall/compressed finished={res.finished} steps={res.steps} "
          f"compacted={mem.compacted} -> {res.answer!r}\n")