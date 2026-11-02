"""L0 离线：合成链路的机械校验、去重、切分、落盘与端到端（全 Mock，零额度）。"""

import json
from fractions import Fraction

import pytest
import synth
from synth import (
    Budget,
    MockChatModel,
    ModelError,
    ModelResult,
    Task,
    UnsafeExpr,
    ValidationError,
    _safe_eval,
    _validate_doc,
    _validate_docs,
    _validate_math,
    _validate_tooluse,
    assign_ids,
    dedupe_math,
    dedupe_tooluse,
    split,
    write_split_files,
)

DOCS = [
    {
        "doc_id": "doc-1",
        "title": "Redis 缓存",
        "text": "Redis 是内存数据库，RDB 持久化会把数据写成快照。",
        "keywords": ["Redis", "RDB", "快照"],
    },
    {
        "doc_id": "doc-2",
        "title": "Docker 镜像",
        "text": "Docker 镜像由镜像分层组成，Dockerfile 每条指令生成一层。",
        "keywords": ["Docker", "Dockerfile", "镜像分层"],
    },
    {
        "doc_id": "doc-3",
        "title": "HTTP 状态码",
        "text": "HTTP 404 表示资源不存在，HTTP 500 表示服务器错误。",
        "keywords": ["HTTP", "404", "500"],
    },
    {
        "doc_id": "doc-4",
        "title": "SSE 流",
        "text": "SSE 使用 text/event-stream 单向推送事件。",
        "keywords": ["SSE", "event-stream", "单向推送"],
    },
]

QA_ITEMS = [
    {
        "statement": "Redis 的哪种持久化会把数据写成快照？",
        "answers": ["RDB"],
        "evidence": ["doc-1"],
    },
    {
        "statement": "Docker 镜像由哪个文件定义构建步骤？",
        "answers": ["Dockerfile"],
        "evidence": ["doc-2"],
    },
    {
        "statement": "HTTP 中哪个状态码表示资源不存在？",
        "answers": ["404"],
        "evidence": ["doc-3"],
    },
    {
        "statement": "HTTP 中哪个状态码表示服务器错误？",
        "answers": ["500"],
        "evidence": ["doc-3"],
    },
    {
        "statement": "Redis 的持久化快照格式，以及 SSE 的事件流媒体类型分别是什么？",
        "answers": ["RDB", "event-stream"],
        "evidence": ["doc-1", "doc-4"],
    },
    {
        "statement": "HTTP 的服务器错误码，以及 Docker 镜像的构建文件分别是什么？",
        "answers": ["500", "Dockerfile"],
        "evidence": ["doc-3", "doc-2"],
    },
]

MATH_ITEMS = [
    {
        "statement": "每支笔 7 元，买 3 支再加 2 元文具袋，共多少元？只给数字，不要写步骤。",
        "expr": "3*7+2",
        "answer": "23",
        "forbid": ["步骤"],
    },
    {"statement": "每盒 12 个，共 4 盒，送出 18 个后还剩多少个？",
     "expr": "12*4-18", "answer": "30", "forbid": []},
    {"statement": "把 25 与 15 的和乘以 2 再除以 2，结果是多少？",
     "expr": "(25+15)*2/2", "answer": "40", "forbid": []},
    {"statement": "144 除以 12 再加 7 是多少？",
     "expr": "144/12+7", "answer": "19", "forbid": []},
    {"statement": "9 乘 9 再减去 16，结果是多少？",
     "expr": "9*9-16", "answer": "65", "forbid": []},
    {"statement": "1000 除以 25 再减 6 是多少？",
     "expr": "1000/25-6", "answer": "34", "forbid": []},
    {"statement": "8 乘 8，再加上 8 除以 2，是多少？",
     "expr": "8*8+8/2", "answer": "68", "forbid": []},
    {"statement": "7 乘 6 再除以 3 等于多少？",
     "expr": "7*6/3", "answer": "14", "forbid": []},
]


def _valid_math(**overrides):
    item = {"statement": "每支 7 元，买 3 支再加 2 元，共多少元？", "expr": "3*7+2",
            "answer": "23", "forbid": []}
    item.update(overrides)
    return item


def _reply(payload) -> ModelResult:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return ModelResult(content=text)


# --------------------------------------------------------------------------- #
# 1. safe eval
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("expr", "expected"),
    [("1+2*3", "7"), ("10/4", "5/2"), ("-3+1", "-2"), ("2**3", "8"), ("7//2+1", "4")],
)
def test_safe_eval_accepts(expr, expected):
    assert _safe_eval(expr) == Fraction(expected)


@pytest.mark.parametrize(
    "expr", ["1+x", "import os", "[].__class__", "2**999999", "1+" * 150, "open('a')"]
)
def test_safe_eval_rejects(expr):
    with pytest.raises(UnsafeExpr):
        _safe_eval(expr)


# --------------------------------------------------------------------------- #
# 2. math 校验
# --------------------------------------------------------------------------- #
def test_math_valid_items_pass_and_mode_derives():
    num = _validate_math(_valid_math())
    assert num["mode"] == "mth-num"
    assert num["checks"] == ["23"]
    fmt = _validate_math(
        _valid_math(statement="3 乘 7 再加 2，只给数字，不要写步骤。", forbid=["步骤"])
    )
    assert fmt["mode"] == "mth-fmt"
    assert fmt["checks"] == ["23", "~步骤"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"answer": "22"}, "求值不符"),
        ({"expr": "3+7", "answer": "10"}, "运算符少于"),
        ({"expr": "3*7+2", "answer": "23", "statement": "答案是 23 元。"}, "答案出现在题干"),
        ({"forbid": ["23"]}, "forbid 与答案"),
        ({"expr": "2**999999", "answer": "1"}, "指数超出范围"),
    ],
)
def test_math_invalid_items_rejected(overrides, reason):
    with pytest.raises(ValidationError, match=reason):
        _validate_math(_valid_math(**overrides))


# --------------------------------------------------------------------------- #
# 3. docs 校验
# --------------------------------------------------------------------------- #
def test_docs_validation_accepts_and_rejects():
    assert len(_validate_docs(DOCS)) == len(DOCS)
    with pytest.raises(ValidationError, match="doc_id 重复"):
        _validate_docs([DOCS[0], dict(DOCS[1], doc_id=DOCS[0]["doc_id"])])
    with pytest.raises(ValidationError, match="keyword 未逐字出现"):
        _validate_doc(dict(DOCS[0], keywords=["Redis", "不存在词"]))


# --------------------------------------------------------------------------- #
# 4. tooluse 校验
# --------------------------------------------------------------------------- #
def test_tooluse_valid_items_pass():
    single = _validate_tooluse(QA_ITEMS[0], DOCS)
    assert single["mode"] == "tol-single" and single["checks"] == ["RDB"]
    multi = _validate_tooluse(QA_ITEMS[4], DOCS)
    assert multi["mode"] == "tol-multi"
    assert multi["checks"] == ["RDB", "event-stream"]


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        ({"statement": "Redis 的哪种持久化会保存快照？", "answers": ["RDB"],
          "evidence": ["doc-99"]}, "不存在的 doc_id"),
        ({"statement": "Redis 文档讲什么？", "answers": ["SSE"],
          "evidence": ["doc-1"]}, "答案不在证据文档"),
        ({"statement": "Redis 的哪种持久化会保存 RDB 快照？", "answers": ["RDB"],
          "evidence": ["doc-1"]}, "答案出现在题干"),
        ({"statement": "哪种持久化方式最常用？", "answers": ["RDB"],
          "evidence": ["doc-1"]}, "distinctive"),
        ({"statement": "短答？", "answers": ["A"], "evidence": ["doc-1"]}, "过短"),
    ],
)
def test_tooluse_invalid_items_rejected(item, reason):
    with pytest.raises(ValidationError, match=reason):
        _validate_tooluse(item, DOCS)


def test_tooluse_answer_must_be_unique_to_evidence():
    docs = _validate_docs(
        [
            {"doc_id": "doc-a", "title": "A", "text": "共享词：阿尔法。独特：贝塔。",
             "keywords": ["阿尔法", "贝塔"]},
            {"doc_id": "doc-b", "title": "B", "text": "共享词：阿尔法。",
             "keywords": ["共享词"]},
        ]
    )
    item = {"statement": "贝塔 是什么？", "answers": ["阿尔法"], "evidence": ["doc-a"]}
    with pytest.raises(ValidationError, match="非证据文档"):
        _validate_tooluse(item, docs)


def test_tooluse_evidence_must_be_retrievable_in_top5():
    docs = []
    for i in range(1, 6):
        docs.append(
            {"doc_id": f"f-0{i}", "title": f"filler {i}",
             "text": f"关键词 k{i}a k{i}b 的说明。", "keywords": [f"k{i}a", f"k{i}b"]}
        )
    docs.append(
        {"doc_id": "doc-z", "title": "evidence",
         "text": "伽马 德尔塔 的说明。", "keywords": ["伽马", "德尔塔"]}
    )
    docs = _validate_docs(docs)
    statement = " ".join(f"k{i}{s}" for i in range(1, 6) for s in ("a", "b")) + " 伽马"
    item = {"statement": statement, "answers": ["德尔塔"], "evidence": ["doc-z"]}
    with pytest.raises(ValidationError, match="检索 top-5 未命中"):
        _validate_tooluse(item, docs)


# --------------------------------------------------------------------------- #
# 5. 去重
# --------------------------------------------------------------------------- #
def test_dedupe_math_by_statement_or_calculation():
    items = [
        {"statement": "Compute 3*7+2", "expr": "3*7+2", "answer": "23"},
        {"statement": "compute   3*7+2", "expr": "3*7+2", "answer": "23"},
        {"statement": "另一道题", "expr": "3*7+2", "answer": "23"},
    ]
    assert len(dedupe_math(items)) == 1


def test_dedupe_tooluse_by_statement_or_answers():
    items = [
        {"statement": "Q A", "answers": ["Redis", "RDB"]},
        {"statement": "q   a", "answers": ["x"]},
        {"statement": "Other", "answers": ["redis", "rdb"]},
    ]
    assert len(dedupe_tooluse(items)) == 1


# --------------------------------------------------------------------------- #
# 6/7. split 与 id
# --------------------------------------------------------------------------- #
def test_split_is_reproducible_and_disjoint():
    items = [{"id": f"x-{i:02d}"} for i in range(10)]
    train, val, test = split(items, seed=42)
    assert (len(train), len(val), len(test)) == (6, 2, 2)
    again = split(items, seed=42)
    assert [r["id"] for r in again[0]] == [r["id"] for r in train]
    other = split(items, seed=7)
    assert {r["id"] for part in other for r in part} == {r["id"] for r in items}
    assert [r["id"] for r in other[0]] != [r["id"] for r in train]
    split_ids = [[r["id"] for r in part] for part in (train, val, test)]
    assert not (set(split_ids[0]) & set(split_ids[1]))
    assert not (set(split_ids[0]) & set(split_ids[2]))
    assert not (set(split_ids[1]) & set(split_ids[2]))


def test_assign_ids_encodes_mode_and_is_globally_unique():
    items = [{"mode": "mth-fmt", "statement": f"s{i}"} for i in range(3)]
    items += [{"mode": "mth-num", "statement": f"n{i}"} for i in range(2)]
    rows = assign_ids(items)
    ids = [r["id"] for r in rows]
    assert ids[:3] == ["mth-fmt-01", "mth-fmt-02", "mth-fmt-03"]
    assert ids[3:] == ["mth-num-01", "mth-num-02"]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------- #
# 8. 落盘
# --------------------------------------------------------------------------- #
def test_write_split_files_rows_match_task_contract(tmp_path):
    items = [
        {"id": f"mth-num-{i:02d}", "statement": f"题 {i}", "checks": [str(i)],
         "mode": "mth-num"}
        for i in range(1, 11)
    ]
    counts = write_split_files(tmp_path, "math", items, seed=42)
    assert counts == {"train": 6, "val": 2, "test": 2}
    for name in ("train", "val", "test"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "math" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        for row in rows:
            assert set(row) == {"id", "statement", "checks"}
            assert row["checks"]
            Task(**row)


# --------------------------------------------------------------------------- #
# 10. 端到端 Mock
# --------------------------------------------------------------------------- #
async def test_end_to_end_mock_writes_all_artifacts(tmp_path):
    model = MockChatModel(
        script=[_reply(MATH_ITEMS), _reply(DOCS), _reply(QA_ITEMS[:4]), _reply(QA_ITEMS[4:])]
    )
    report = await synth.synthesize(
        model,
        math_count=6,
        tooluse_count=6,
        seed=42,
        max_calls=20,
        out_dir=tmp_path,
        model_name="mock",
        docs_wanted=4,
    )
    assert len(report.math) == 6
    assert len(report.tooluse) == 6
    assert len(report.docs) == 4
    assert report.calls == 4
    assert report.counts == {
        "math": {"train": 3, "val": 1, "test": 2},
        "tooluse": {"train": 3, "val": 1, "test": 2},
    }
    for domain in ("math", "tooluse"):
        for name in ("train", "val", "test"):
            path = tmp_path / domain / f"{name}.jsonl"
            assert path.exists()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            for row in rows:
                assert set(row) == {"id", "statement", "checks"}
                Task(**row)
    corpus = json.loads((tmp_path / "tooluse" / "corpus.json").read_text(encoding="utf-8"))
    assert len(corpus["docs"]) == 4
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "generation_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(log_rows) == 12
    assert {row["domain"] for row in log_rows} == {"math", "tooluse"}
    assert {row["id"].split("-", 1)[0] for row in log_rows} >= {"mth", "tol"}
    assert any(row["id"].startswith("mth-fmt-") for row in log_rows)
    assert any(row["id"].startswith("tol-multi-") for row in log_rows)


# --------------------------------------------------------------------------- #
# 11. 失败兜底
# --------------------------------------------------------------------------- #
async def test_bad_json_retries_half_batch(tmp_path):
    model = MockChatModel(script=[_reply("这不是 JSON"), _reply(MATH_ITEMS[:2])])
    report = await synth.synthesize(
        model, math_count=2, tooluse_count=0, max_calls=10, out_dir=tmp_path, seed=42
    )
    assert len(report.math) == 2
    assert report.calls == 2
    for name in ("train", "val", "test"):
        path = tmp_path / "math" / f"{name}.jsonl"
        assert path.exists()
        for line in path.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"id", "statement", "checks"}


class BoomModel(MockChatModel):
    async def complete(self, messages, tools=None):
        raise ModelError("boom")


async def test_model_error_writes_no_dirty_rows(tmp_path):
    report = await synth.synthesize(
        BoomModel(), math_count=2, tooluse_count=0, max_calls=2, out_dir=tmp_path, seed=42
    )
    assert report.math == []
    assert report.calls == 2
    assert (tmp_path / "generation_log.jsonl").read_text(encoding="utf-8") == ""
    for name in ("train", "val", "test"):
        assert (tmp_path / "math" / f"{name}.jsonl").read_text(encoding="utf-8") == ""


def test_main_returns_nonzero_on_shortfall(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setattr(synth, "OpenAIChatModel", lambda **kwargs: BoomModel())
    rc = synth.main(
        ["--math", "2", "--tooluse", "0", "--max-calls", "1", "--out", str(tmp_path / "out")]
    )
    assert rc == 1
    assert (tmp_path / "out" / "generation_log.jsonl").exists()


def test_budget_limits_calls():
    budget = Budget(1)
    budget.spend()
    with pytest.raises(synth.BudgetExhausted):
        budget.spend()
