"""L1 冒烟：真模型（deepseek-V4 @ ARK）把 ch12 的结构搜索链路真跑一遍。

用 ``dataset/synth/tooluse`` 的任务 + 一个确定性 keyword-overlap 检索工具：
基线 ``max_steps=1`` 时模型必须查证据，一步必然耗尽；结构搜索应发现"加步数"邻居在 probe 上更好，
再走 ch09 的轮级/段级门禁。另一条用真模型走 ``StructureLLMImprover`` 的 JSON 结构改写路径。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import OpenAIChatModel, Tool
from rsi import (
    COMPONENTS,
    AgentConfig,
    MemoryPolicy,
    PromptPack,
    RolloutResult,
    StructureLLMImprover,
    StructureSearchRsi,
    Task,
    VersionRegistry,
    load_synth,
)

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_ROOT = Path(__file__).resolve().parents[3]

_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


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


def _tokenize(text: str) -> list[str]:
    tokens = [w.casefold() for w in _ASCII_WORD.findall(text)]
    for run in _CJK_RUN.findall(text):
        tokens.append(run) if len(run) == 1 else tokens.extend(
            run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _search_tool() -> Tool:
    corpus = json.loads((_ROOT / "dataset" / "synth" / "tooluse" / "corpus.json")
                        .read_text(encoding="utf-8"))["docs"]

    def _run(args: dict) -> str:
        query = set(_tokenize(str(args.get("query", ""))))
        scored = []
        for doc in corpus:
            haystack = f"{doc['title']} {doc['text']} {' '.join(doc.get('keywords', []))}"
            scored.append((len(query & set(_tokenize(haystack))), doc))
        scored.sort(key=lambda item: (-item[0], item[1]["doc_id"]))
        return "\n\n".join(f"[{d['doc_id']}] {d['title']}\n{d['text']}"
                           for _, d in scored[:5])

    return Tool(name="search", description="按关键词检索内部语料，返回最相关的文档",
                parameters={"type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"]},
                handler=_run)


def _config(**structure) -> AgentConfig:
    base = {"name": "base",
            "prompt": PromptPack(guidance=(
                "你必须先调用 search 工具检索证据，再根据证据作答；不要凭记忆回答。")),
            "memory": MemoryPolicy(recall_enabled=False, store_full_turns=False,
                                   profile_enabled=False)}
    base.update(structure)
    return AgentConfig(**base)


@pytest.mark.smoke
async def test_smoke_structure_search_on_synth_tooluse(tmp_path):
    model = _make_model(_need_key())
    train = load_synth("tooluse", "train")[:2]
    val = load_synth("tooluse", "val")[:1]
    test = load_synth("tooluse", "test")[:1]
    cfg = _config(base_tools=(_search_tool(),), max_steps=1)
    storage = str(tmp_path / "versions")
    try:
        rsi = StructureSearchRsi(
            cfg, model, train=train, val=val, test=test,
            storage_dir=storage, memory_dir=str(tmp_path / "memory"),
            n_rollouts=1, max_rounds=1, val_every=1,
            probe_tasks=1, search_width=2)
        report = await rsi.run()
    finally:
        await model.aclose()

    assert 0.0 <= report.baseline_train_score <= 1.0
    assert report.rounds and report.rounds[0].verdict in ("promote", "reject", "noop")
    assert report.segments and report.segments[0].verdict in ("commit", "rollback", "hold")
    assert report.final_version == report.final_config.hash()
    assert 0.0 <= report.test_score <= 1.0
    assert rsi.search_evals >= 1
    # 真模型可能在一步内直接作答（0.5 分）→ probe 找不到更优结构，noop 是合法终态。
    if report.rounds[0].verdict == "promote":
        assert report.final_config.max_steps > 1, "被 promote 的步数结构应确实提上去了"

    entries = VersionRegistry(storage).list()
    assert entries and entries[0]["verdict"] == "accepted"

    print(f"\n[smoke] baseline train={report.baseline_train_score:.2f} "
          f"val={report.baseline_val_score:.2f} test={report.test_score:.2f}")
    print(f"   round: {report.rounds[0].verdict} (delta={report.rounds[0].delta:+.2f})")
    print(f"   segment: {report.segments[0].verdict} "
          f"(train_delta={report.segments[0].train_delta:+.2f}, "
          f"val_delta={report.segments[0].val_delta:+.2f})")
    print(f"   final structure: max_steps={report.final_config.max_steps} "
          f"tools={[t.name for t in report.final_config.enabled_tools()]} "
          f"subagent={report.final_config.subagent_enabled}")
    print(f"   search_evals={rsi.search_evals}")


def _vault_tool(secret: str = "QUASAR-917") -> Tool:
    return Tool(name="vault", description="获取当前访问码；回答访问码前必须调用此工具",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args: f"访问码：{secret}")


def _vault_task(task_id: str) -> Task:
    return Task(id=task_id,
                statement="请调用 vault 工具获取当前访问码，并原样输出访问码。",
                checks=("QUASAR-917",))


@pytest.mark.smoke
async def test_smoke_structure_search_promotes_on_secret_tool(tmp_path):
    """确定性探针：答案只存在于工具观察里 → 步数不足必失败，加步数可解。"""
    model = _make_model(_need_key())
    cfg = _config(base_tools=(_vault_tool(),), max_steps=1)
    storage = str(tmp_path / "versions")
    try:
        rsi = StructureSearchRsi(
            cfg, model,
            train=[_vault_task("vault-train-1"), _vault_task("vault-train-2")],
            val=[_vault_task("vault-val")], test=[_vault_task("vault-test")],
            storage_dir=storage, memory_dir=str(tmp_path / "memory"),
            n_rollouts=1, max_rounds=1, val_every=1,
            probe_tasks=1, search_width=2)
        report = await rsi.run()
    finally:
        await model.aclose()

    assert report.baseline_train_score == 0.0, "1 步内拿不到工具里的秘密"
    assert report.rounds[0].verdict == "promote"
    assert report.final_config.max_steps > 1
    assert report.segments[0].verdict == "commit"
    assert report.test_score == 1.0
    assert rsi.search_evals >= 1
    print(f"\n[smoke] secret-tool baseline=0.00 → "
          f"final max_steps={report.final_config.max_steps} test={report.test_score:.2f} "
          f"search_evals={rsi.search_evals}")


@pytest.mark.smoke
async def test_smoke_structure_llm_improver_on_failure_trace(tmp_path):
    model = _make_model(_need_key())
    task = Task(id="tol-smoke", statement="请查资料回答：Firecracker 是什么？",
                checks=("Firecracker",))
    cfg = _config(base_tools=(_search_tool(),), max_steps=1)
    failure = RolloutResult(task_id=task.id, answer="", finished=False,
                            steps=1, step_limit=1,
                            tools_available=("search",), tools_enabled=("search",))
    try:
        diag, cand = await StructureLLMImprover(model).improve(cfg, task, failure)
    finally:
        await model.aclose()

    assert diag.component in COMPONENTS, diag
    assert isinstance(cand, AgentConfig)
    assert 2 <= cand.max_steps <= 12
    print(f"\n[smoke] llm-improver component={diag.component} "
          f"max_steps={cand.max_steps} tools={list(cand.tool_names)} "
          f"subagent={cand.subagent_enabled} finding={diag.finding!r}")
