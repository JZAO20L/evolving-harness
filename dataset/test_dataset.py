"""RSI 探针集的 schema 校验：纯 JSON、离线、零额度。

守住契约：每行是 `{id, statement, checks}`、id 全局唯一、split 不相交、checks 合法（`word`/`~word`）。
数据本身对齐 `course/ch08_rl_loop/rsi.py` 的 `Task`；此处不 import 任何章代码（避免跨目录耦合）。
"""

import json
from pathlib import Path

import pytest

SPLITS = ("train", "val", "test")
DATA_DIR = Path(__file__).resolve().parent / "rsi"


def _load(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    assert path.exists(), f"缺少 {path}"
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows.append((i, json.loads(line)))
    return rows


def _all() -> list[tuple[str, int, dict]]:
    return [(s, i, row) for s in SPLITS for (i, row) in _load(s)]


@pytest.mark.parametrize("split", SPLITS)
def test_split_schema_and_checks(split):
    for lineno, row in _load(split):
        where = f"{split}.jsonl:{lineno}"
        assert set(row) == {"id", "statement", "checks"}, f"{where} 字段必须恰为 id/statement/checks"
        assert isinstance(row["id"], str) and row["id"].strip(), f"{where} id 非空"
        assert isinstance(row["statement"], str) and row["statement"].strip(), f"{where} statement 非空"
        checks = row["checks"]
        assert isinstance(checks, list) and checks, f"{where} checks 非空数组"
        for c in checks:
            key = c.removeprefix("~")
            assert isinstance(c, str) and key.strip(), f"{where} check 非空（去掉 ~ 后仍有内容）：{c!r}"
        assert len(checks) == len(set(checks)), f"{where} checks 有重复项"


def test_ids_globally_unique():
    ids = [row["id"] for _, _, row in _all()]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"id 全局重复：{sorted(dup)}"


def test_splits_disjoint_by_id():
    seen: dict[str, str] = {}
    for split, _, row in _all():
        if row["id"] in seen:
            pytest.fail(f"id {row['id']!r} 同时出现在 {seen[row['id']]} 与 {split}")
        seen[row["id"]] = split


def test_sizes_and_failure_mode_prefixes():
    for split in SPLITS:
        rows = [row for _, row in _load(split)]
        assert rows, f"{split} 不应为空"
        for row in rows:
            assert row["id"].split("-", 1)[0] in {"fmt", "multi", "num", "hard"}, \
                f"{row['id']} 前缀应为 fmt/multi/num/hard（编码失败模式）"


def test_checks_deterministic_substring_contract():
    """checks 必须能被子串判定：正 check 不得是 `~`，负 check 需 `~` 前缀（RuleScorer 契约）。"""
    for split, lineno, row in _all():
        for c in row["checks"]:
            assert len(c) >= 2 or c.isdigit() or c.startswith("~"), \
                f"{split}.jsonl:{lineno} check 过短难判定：{c!r}"


# --------------------------------------------------------------------------- #
# dataset/synth：合成链路产物（不存在则跳过，方便未生成时跑 dataset/）
# --------------------------------------------------------------------------- #
SYNTH_DIR = Path(__file__).resolve().parent / "synth"
SYNTH_DOMAINS = ("math", "tooluse")
SYNTH_PREFIXES = {
    "math": ("mth-num-", "mth-fmt-"),
    "tooluse": ("tol-single-", "tol-multi-"),
}


def _synth_base(domain: str) -> Path:
    base = SYNTH_DIR / domain
    if not base.exists():
        pytest.skip(f"未生成 dataset/synth/{domain}，跳过 synth 产物校验")
    return base


def _synth_rows(domain: str) -> list[tuple[str, int, dict]]:
    base = _synth_base(domain)
    rows = []
    for split in SPLITS:
        path = base / f"{split}.jsonl"
        assert path.exists(), f"缺少 {path}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows.append((split, i, json.loads(line)))
    return rows


@pytest.mark.parametrize("domain", SYNTH_DOMAINS)
def test_synth_split_schema_and_sizes(domain):
    base = _synth_base(domain)
    seen = {}
    for split in SPLITS:
        path = base / f"{split}.jsonl"
        assert path.exists(), f"缺少 {path}"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == {"train": 30, "val": 10, "test": 10}[split], \
            f"{domain}/{split}.jsonl 应为 30/10/10 配比中的对应规模"
        for row in rows:
            assert set(row) == {"id", "statement", "checks"}, f"{domain}/{split} 字段必须恰为 3 个"
            assert row["id"].startswith(SYNTH_PREFIXES[domain]), \
                f"{row['id']} 前缀应为 {SYNTH_PREFIXES[domain]}（编码领域+失败模式）"
            assert row["statement"].strip()
            assert row["checks"] and len(row["checks"]) == len(set(row["checks"]))
            for c in row["checks"]:
                key = c.removeprefix("~")
                assert key.strip(), f"{domain}/{split} check 去掉 ~ 后不能为空"
            assert row["id"] not in seen, f"{row['id']} 同时出现在 {seen.get(row['id'])} 与 {split}"
            seen[row["id"]] = split


def test_synth_ids_globally_unique_across_domains():
    ids = []
    for domain in SYNTH_DOMAINS:
        ids += [row["id"] for _, _, row in _synth_rows(domain)]
    assert ids, "synth 产物为空"
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"synth id 全局重复：{sorted(dup)}"


def test_synth_corpus_schema():
    base = _synth_base("tooluse")
    corpus = json.loads((base / "corpus.json").read_text(encoding="utf-8"))
    docs = corpus.get("docs")
    assert isinstance(docs, list) and docs, "corpus.docs 非空"
    ids = [doc["doc_id"] for doc in docs]
    assert len(ids) == len(set(ids)), "corpus doc_id 重复"
    for doc in docs:
        assert set(doc) == {"doc_id", "title", "text", "keywords"}
        assert doc["title"].strip() and doc["text"].strip()
        assert doc["keywords"]
        for kw in doc["keywords"]:
            assert kw in doc["text"], f"{doc['doc_id']} 的 keyword 未逐字出现在 text：{kw}"


def test_synth_generation_log_matches_tasks():
    log_path = SYNTH_DIR / "generation_log.jsonl"
    if not log_path.exists():
        pytest.skip("未生成 generation_log.jsonl")
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = {
        row["id"] for domain in SYNTH_DOMAINS for _, _, row in _synth_rows(domain)
    }
    logged_ids = [row["id"] for row in rows]
    assert len(logged_ids) == len(set(logged_ids)), "generation_log id 重复"
    assert set(logged_ids) == task_ids, "generation_log 与落盘 task 的 id 集合必须一致"
    assert {row["domain"] for row in rows} == set(SYNTH_DOMAINS)
    for row in rows:
        assert row["model"] and row["seed"] == 42, "provenance 必须记录 model/seed"
