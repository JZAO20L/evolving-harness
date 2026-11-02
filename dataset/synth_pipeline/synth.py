"""dataset/synth_pipeline · 100 条混合数据合成链路（非课程章节）。

verifier-first：先让 chat LLM 批量出题，再用**机械校验**逐条裁决，只落盘通过的条目；
随后按领域去重、6:2:2 划分 train/val/test。

两个领域：
- math：纯算术题，answer 必须与安全求值结果完全相等；
- tooluse：本地假语料（docs）+ 查证型 QA，答案必须在证据文档中、全语料唯一、题面可检索到证据。

产物与 `course/chNN_*/rsi.py` 的 `Task{id, statement, checks}` 严格兼容：
checks 是确定性子串 rubric（`word` 必含、`~word` 必不含），可离线判分。

CLI 只读 `os.environ`（shell 里先 `set -a; source .env; set +a`），代码不主动读 `.env`。
"""

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "deepseek-v4-flash-ga-260731"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# 模型适配器（自包含最小版，复制自 course/ch09_prompt_rsi/agent.py）
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str
    content: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelResult:
    content: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


class ModelError(Exception):
    """模型端点不可达、异常返回、或解析失败（重试耗尽后抛出）。"""


RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class _Retryable(Exception):
    """内部信号：本次请求失败但值得重试。"""


class ChatModel:
    async def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.4,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url 必填")
        if not model:
            raise ValueError("model 必填")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        if http_client is not None:
            self._http = http_client
        else:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            self._http = httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout, headers=headers
            )

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> ModelResult:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            payload["tools"] = tools
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    raise _Retryable(f"HTTP {response.status_code}: {response.text[:200]}")
                if response.status_code != 200:
                    raise ModelError(f"端点返回 {response.status_code}: {response.text[:200]}")
                return self._complete_ok(response.json())
            except (httpx.TransportError, _Retryable) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    backoff = min(8.0, 0.5 * (2**attempt)) + random.uniform(0, 0.2)
                    await asyncio.sleep(backoff)
        raise ModelError(f"模型请求重试后仍失败: {last_error}")

    def _complete_ok(self, data: dict) -> ModelResult:
        try:
            choice = data["choices"][0]
            usage = data.get("usage", {})
        except (KeyError, IndexError) as exc:
            raise ModelError(f"响应结构异常: {data}") from exc
        message = choice.get("message") or {}
        return ModelResult(
            content=message.get("content") or "",
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._http.aclose()


class MockChatModel(ChatModel):
    """脚本化响应；不联网。script 出空后回显最后一条 user 内容。"""

    def __init__(self, script: list[ModelResult] | None = None) -> None:
        self.script = list(script or [])
        self.messages: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> ModelResult:
        self.messages.append(list(messages))
        if self.script:
            return self.script.pop(0)
        last = messages[-1].content if messages else ""
        return ModelResult(content=f"mock:{last}", usage=Usage(10, 5), finish_reason="stop")

    async def aclose(self) -> None:
        pass


def _extract_json(text: str):
    """从模型回复容错剥出第一个 JSON 对象/数组（容忍 ```json 包裹、前后夹文）。"""
    s = text.strip()
    for fence in ("```json", "```"):
        if fence in s:
            parts = s.split(fence)
            if len(parts) >= 3:
                s = parts[1].strip()
            break
    candidates = [k for k in (s.find("{"), s.find("[")) if k != -1]
    if not candidates:
        return None
    start = min(candidates)
    for end in range(start + 1, len(s) + 1):
        try:
            return json.loads(s[start:end])
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Task 与校验错误
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    id: str
    statement: str
    checks: list[str]


class ValidationError(ValueError):
    """机械校验不通过：该条数据必须丢弃，绝不落盘。"""


class UnsafeExpr(ValidationError):
    """表达式超出安全 AST 白名单。"""


# --------------------------------------------------------------------------- #
# math：安全求值 + 校验
# --------------------------------------------------------------------------- #
_MAX_EXPR_LEN = 200
_MAX_OPERATORS = 8
_MAX_LITERAL = 10**12
_MAX_EXPONENT = 6
_MAX_ANSWER_LEN = 8
_MAX_COMPONENT_DIGITS = 12


def _parse_expr(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpr("表达式为空")
    if len(expr) > _MAX_EXPR_LEN:
        raise UnsafeExpr("表达式过长")
    try:
        return ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise UnsafeExpr(f"表达式语法错误: {exc}") from exc


def _eval_node(node: ast.AST, depth: int = 0) -> Fraction:
    if depth > 20:
        raise UnsafeExpr("表达式嵌套过深")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise UnsafeExpr("只允许整数字面量")
        if abs(node.value) > _MAX_LITERAL:
            raise UnsafeExpr("字面量过大")
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand, depth + 1)
        if isinstance(node.op, ast.UAdd):
            return _eval_node(node.operand, depth + 1)
        raise UnsafeExpr("不支持的单元运算符")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeExpr("除数为 0")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise UnsafeExpr("除数为 0")
            return Fraction(left // right)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise UnsafeExpr("除数为 0")
            return Fraction(left % right)
        if isinstance(node.op, ast.Pow):
            if not (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and not isinstance(node.right.value, bool)
            ):
                raise UnsafeExpr("指数必须是整数字面量")
            exponent = node.right.value
            if exponent < 0 or exponent > _MAX_EXPONENT:
                raise UnsafeExpr("指数超出范围")
            return left**exponent
        raise UnsafeExpr("不支持的二元运算符")
    raise UnsafeExpr("不允许的语法结构")


def _safe_eval(expr: str) -> Fraction:
    """白名单 AST 求值，返回精确 Fraction；任何越界都抛 UnsafeExpr。"""
    tree = _parse_expr(expr)
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise UnsafeExpr("除数为 0") from exc


def _operator_count(expr: str) -> int:
    return sum(isinstance(n, ast.BinOp) for n in ast.walk(_parse_expr(expr)))


def _normalize_answer(text: str) -> Fraction | None:
    s = text.strip().replace(" ", "")
    if not s:
        return None
    try:
        if "/" in s:
            num, _, den = s.partition("/")
            return Fraction(int(num), int(den))
        return Fraction(int(s))
    except (ValueError, ZeroDivisionError):
        return None


def _canonical_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _nonempty_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} 必须是非空字符串")
    return value.strip()


def _normalize_forbid(value, answer: str | None = None) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValidationError("forbid 必须是字符串数组")
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValidationError("forbid 含空词")
        word = raw.strip()
        if answer is not None and word == answer:
            raise ValidationError("forbid 与答案相同")
        if word not in out:
            out.append(word)
    return out


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _validate_math(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValidationError("math 项必须是对象")
    statement = _nonempty_str(raw.get("statement"), "statement")
    expr = _nonempty_str(raw.get("expr"), "expr")
    answer_text = _nonempty_str(raw.get("answer"), "answer")
    value = _safe_eval(expr)
    ops = _operator_count(expr)
    if ops < 2:
        raise ValidationError("表达式运算符少于 2 个")
    if ops > _MAX_OPERATORS:
        raise ValidationError("表达式运算符过多")
    normalized = _normalize_answer(answer_text)
    if normalized is None:
        raise ValidationError("answer 不是整数/分数")
    if normalized != value:
        raise ValidationError(f"answer 与 expr 求值不符：{normalized} != {value}")
    canonical = _canonical_fraction(normalized)
    if len(canonical) > _MAX_ANSWER_LEN:
        raise ValidationError("答案过长")
    if len(str(abs(normalized.numerator))) > _MAX_COMPONENT_DIGITS:
        raise ValidationError("答案分子过长")
    if len(str(normalized.denominator)) > _MAX_COMPONENT_DIGITS:
        raise ValidationError("答案分母过长")
    if canonical in statement:
        raise ValidationError("答案出现在题干")
    forbid = _normalize_forbid(raw.get("forbid"), canonical)
    mode = "mth-fmt" if forbid else "mth-num"
    checks = _unique([canonical] + [f"~{w}" for w in forbid])
    return {
        "statement": statement,
        "expr": expr,
        "answer": canonical,
        "forbid": forbid,
        "mode": mode,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# 检索：确定性 keyword-overlap（ASCII 词 + CJK bigram，tie-break by doc_id）
# --------------------------------------------------------------------------- #
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


def _tokenize(text: str) -> list[str]:
    tokens = [w.casefold() for w in _ASCII_WORD.findall(text)]
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def search(statement: str, corpus: list[dict], top_k: int = 5) -> list[dict]:
    """按词元重叠打分，返回 top-k 文档（分数降序、doc_id 升序）。"""
    query = set(_tokenize(statement))
    scored = []
    for doc in corpus:
        haystack = f"{doc['title']} {doc['text']} {' '.join(doc.get('keywords', []))}"
        score = len(query & set(_tokenize(haystack)))
        scored.append({"doc_id": doc["doc_id"], "score": score})
    scored.sort(key=lambda item: (-item["score"], item["doc_id"]))
    return scored[:top_k]


def _distinctive_keywords(corpus: list[dict]) -> dict[str, set[str]]:
    """每篇文档中，在整个语料 text 里只出现于该篇的 keywords。"""
    all_keywords = {
        kw for doc in corpus for kw in doc["keywords"] if kw in doc["text"]
    }
    counts = {
        kw: sum(1 for doc in corpus if kw in doc["text"]) for kw in all_keywords
    }
    return {
        doc["doc_id"]: {
            kw
            for kw in doc["keywords"]
            if kw in doc["text"] and counts.get(kw, 0) == 1
        }
        for doc in corpus
    }


# --------------------------------------------------------------------------- #
# tooluse：docs 校验 + QA 校验
# --------------------------------------------------------------------------- #
def _validate_doc(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValidationError("doc 必须是对象")
    doc_id = _nonempty_str(raw.get("doc_id"), "doc_id")
    title = _nonempty_str(raw.get("title"), "title")
    text = _nonempty_str(raw.get("text"), "text")
    keywords = raw.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValidationError("keywords 必须是非空数组")
    words: list[str] = []
    for kw in keywords:
        if not isinstance(kw, str) or not kw.strip():
            raise ValidationError("keyword 为空")
        kw = kw.strip()
        if kw not in text:
            raise ValidationError(f"keyword 未逐字出现在 text：{kw}")
        if kw not in words:
            words.append(kw)
    return {"doc_id": doc_id, "title": title, "text": text, "keywords": words}


def _validate_docs(raw) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValidationError("docs 必须是非空数组")
    docs: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        doc = _validate_doc(item)
        if doc["doc_id"] in seen:
            raise ValidationError(f"doc_id 重复：{doc['doc_id']}")
        seen.add(doc["doc_id"])
        docs.append(doc)
    return docs


def _is_substantive(text: str) -> bool:
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    ascii_chars = sum(1 for ch in text if ch.isascii() and ch.isalnum())
    return cjk >= 2 or ascii_chars >= 3


def _validate_tooluse(raw, corpus: list[dict]) -> dict:
    if not isinstance(raw, dict):
        raise ValidationError("tooluse 项必须是对象")
    statement = _nonempty_str(raw.get("statement"), "statement")
    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, list) or not answers_raw:
        raise ValidationError("answers 必须是非空数组")
    answers: list[str] = []
    for a in answers_raw:
        if not isinstance(a, str) or not a.strip():
            raise ValidationError("answer 为空")
        a = a.strip()
        if not _is_substantive(a):
            raise ValidationError(f"answer 过短难判定：{a}")
        if a not in answers:
            answers.append(a)
    evidence_raw = raw.get("evidence")
    if not isinstance(evidence_raw, list) or not 1 <= len(evidence_raw) <= 2:
        raise ValidationError("evidence 必须是 1-2 个 doc_id")
    evidence: list[str] = []
    for e in evidence_raw:
        if not isinstance(e, str) or not e.strip():
            raise ValidationError("evidence 含空 doc_id")
        if e not in evidence:
            evidence.append(e)
    docs_by_id = {doc["doc_id"]: doc for doc in corpus}
    missing = [e for e in evidence if e not in docs_by_id]
    if missing:
        raise ValidationError(f"evidence 引用了不存在的 doc_id：{missing}")
    evidence_docs = [docs_by_id[e] for e in evidence]
    evidence_set = set(evidence)
    for answer in answers:
        if answer in statement:
            raise ValidationError(f"答案出现在题干：{answer}")
        if not any(answer in doc["text"] for doc in evidence_docs):
            raise ValidationError(f"答案不在证据文档中：{answer}")
        leaked = [
            doc["doc_id"]
            for doc in corpus
            if doc["doc_id"] not in evidence_set and answer in doc["text"]
        ]
        if leaked:
            raise ValidationError(f"答案出现在非证据文档 {leaked}：{answer}")
    distinctive = _distinctive_keywords(corpus)
    for doc in evidence_docs:
        own = distinctive.get(doc["doc_id"], set())
        if not any(kw in statement for kw in own):
            raise ValidationError(f"题面未点出 {doc['doc_id']} 的 distinctive 关键词")
    hits = {row["doc_id"] for row in search(statement, corpus, top_k=5)}
    unreachable = [e for e in evidence if e not in hits]
    if unreachable:
        raise ValidationError(f"检索 top-5 未命中证据：{unreachable}")
    forbid = _normalize_forbid(raw.get("forbid"))
    conflict = [w for w in forbid if w in answers]
    if conflict:
        raise ValidationError(f"forbid 与答案冲突：{conflict}")
    mode = "tol-single" if len(evidence) == 1 else "tol-multi"
    checks = _unique(answers + [f"~{w}" for w in forbid])
    return {
        "statement": statement,
        "answers": answers,
        "evidence": evidence,
        "forbid": forbid,
        "mode": mode,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# 去重 / id / split / 落盘
# --------------------------------------------------------------------------- #
def _norm_statement(statement: str) -> str:
    return "".join(statement.split()).casefold()


def dedupe_math(items: list[dict]) -> list[dict]:
    seen_statement: set[str] = set()
    seen_calc: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in items:
        statement = _norm_statement(item["statement"])
        calc = (item["expr"], item["answer"])
        if statement in seen_statement or calc in seen_calc:
            continue
        seen_statement.add(statement)
        seen_calc.add(calc)
        out.append(item)
    return out


def dedupe_tooluse(items: list[dict]) -> list[dict]:
    seen_statement: set[str] = set()
    seen_answers: set[tuple[str, ...]] = set()
    out: list[dict] = []
    for item in items:
        statement = _norm_statement(item["statement"])
        answers = tuple(sorted(a.casefold() for a in item["answers"]))
        if statement in seen_statement or answers in seen_answers:
            continue
        seen_statement.add(statement)
        seen_answers.add(answers)
        out.append(item)
    return out


def assign_ids(items: list[dict]) -> list[dict]:
    counters: dict[str, int] = defaultdict(int)
    out: list[dict] = []
    for item in items:
        mode = item["mode"]
        counters[mode] += 1
        row = dict(item)
        row["id"] = f"{mode}-{counters[mode]:02d}"
        out.append(row)
    return out


def split(items: list[dict], seed: int, ratios: tuple[int, int, int] = (6, 2, 2)):
    """按 ratios/10 切分；同 seed 结果可复现，余数归入 test。"""
    if len(ratios) != 3 or sum(ratios) != 10 or any(r < 0 for r in ratios):
        raise ValueError("ratios 必须是三项非负数且和为 10")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = n * ratios[0] // 10
    n_val = n * ratios[1] // 10
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def _row(item: dict) -> dict:
    return {"id": item["id"], "statement": item["statement"], "checks": item["checks"]}


def write_split_files(out_dir: Path, domain: str, items: list[dict], seed: int) -> dict:
    train, val, test = split(items, seed)
    target = out_dir / domain
    target.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in zip(SPLITS, (train, val, test), strict=True):
        path = target / f"{name}.jsonl"
        lines = [
            json.dumps(_row(item), ensure_ascii=False, sort_keys=True) for item in part
        ]
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        counts[name] = len(part)
    return counts


def write_corpus(out_dir: Path, docs: list[dict]) -> None:
    target = out_dir / "tooluse"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "docs": [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": doc["text"],
                "keywords": doc["keywords"],
            }
            for doc in docs
        ]
    }
    (target / "corpus.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _log_rows(math_items, tooluse_items, model_name: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    for item in math_items:
        rows.append(
            {
                "id": item["id"],
                "domain": "math",
                "mode": item["mode"],
                "statement": item["statement"],
                "expr": item["expr"],
                "answer": item["answer"],
                "forbid": item["forbid"],
                "model": model_name,
                "seed": seed,
            }
        )
    for item in tooluse_items:
        rows.append(
            {
                "id": item["id"],
                "domain": "tooluse",
                "mode": item["mode"],
                "statement": item["statement"],
                "answers": item["answers"],
                "evidence": item["evidence"],
                "forbid": item["forbid"],
                "model": model_name,
                "seed": seed,
            }
        )
    return rows


def write_log(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    (out_dir / "generation_log.jsonl").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# 生成 prompt
# --------------------------------------------------------------------------- #
MATH_SYSTEM = "你是算术题出题器。只输出一个 JSON 数组，不要输出数组之外的任何文字。"
DOCS_SYSTEM = "你是技术语料作者。只输出一个 JSON 数组，不要输出数组之外的任何文字。"
QA_SYSTEM = "你是查证型出题器：只依据给定文档出题，只输出一个 JSON 数组。"

_MATH_TOPICS = (
    "购物找零", "速度与时间", "面积与周长", "比例分配", "分段计价",
    "库存增减", "平均数", "倍数关系", "打折与税费", "行程与油耗",
)

_TOOL_TOPICS = (
    "Python 异步编程", "HTTP 状态码", "Redis 缓存策略", "Docker 镜像分层",
    "SSE 与 WebSocket", "SQL 索引", "消息队列", "Agent 循环",
    "function calling", "token 预算", "上下文压缩", "版本回滚",
    "限流与退避", "评测污染", "子代理隔离", "向量检索",
    "事务与隔离级别", "进程与线程", "沙箱隔离", "可观测性",
)


def _math_prompt(n: int, topic: str) -> str:
    return (
        f"请围绕「{topic}」生成 {n} 道算术题。\n"
        "每道题：先设计一个纯算术表达式 expr，再写一句中文 statement 描述它，answer 为 expr 的精确值。\n"
        "硬性要求：\n"
        "1. expr 只含整数、括号与 + - * / // % **；** 的指数必须是 0..6 的整数字面量；"
        "表达式不超过 200 字符。\n"
        "2. 至少有 2 个运算符。\n"
        "3. answer 必须与 expr 的计算结果完全相等，写成整数或最简分数 n/d（如 5/2）。\n"
        "4. statement 中不得出现 answer 的文本；答案不能被题干直接抄走。\n"
        "5. forbid 可为空数组，或给 1-2 个应被禁止出现的废话词（如「步骤」「推导」）；"
        "若使用 forbid，statement 要明确要求只给数字/禁止这些词。\n"
        '输出形如 [{"statement":"...","expr":"...","answer":"...","forbid":["..."]}] 的 JSON 数组。'
    )


def _docs_prompt(n: int, topics: list[str]) -> str:
    topic_line = "、".join(topics)
    return (
        f"请围绕这些工程主题写 {n} 篇互补的技术短文档：{topic_line}。\n"
        "每篇 2-4 句、信息具体、彼此不重复；doc_id 从 doc-01 起唯一编号。\n"
        "每篇给出 2-4 个 keywords，每个 keyword 必须逐字出现在该篇 text 中（用于检索）。\n"
        '输出 JSON 数组：[{"doc_id":"doc-01","title":"...","text":"...",'
        '"keywords":["..."]}]'
    )


def _render_corpus(corpus: list[dict], distinctive: dict[str, set[str]]) -> str:
    lines: list[str] = []
    for doc in corpus:
        lines.append(f"[{doc['doc_id']}] {doc['title']}")
        lines.append(f"text: {doc['text']}")
        lines.append(f"keywords: {'、'.join(doc['keywords'])}")
        own = distinctive.get(doc["doc_id"], set())
        if own:
            lines.append(f"distinctive: {'、'.join(sorted(own))}")
    return "\n".join(lines)


def _qa_prompt(n: int, corpus: list[dict], distinctive: dict[str, set[str]]) -> str:
    return (
        f"以下是 {len(corpus)} 篇技术文档：\n{_render_corpus(corpus, distinctive)}\n\n"
        f"请针对这些文档生成 {n} 道必须查证才能答对的问答题。\n"
        "硬性要求：\n"
        "1. evidence 为 1 篇或 2 篇 doc_id；answers 是 1-3 个短语，"
        "每个都必须逐字出现在某篇 evidence 的 text 中。\n"
        "2. answers 绝不能出现在 statement 中，也不能出现在任何非 evidence 文档的 text 中"
        "（全语料唯一，逼模型去查证）。\n"
        "3. statement 必须点出每篇 evidence 至少一个 distinctive 关键词。\n"
        "4. 单篇证据 → tol-single；两篇证据 → tol-multi。\n"
        "5. forbid 可为空数组，或给 1-2 个应被禁止出现的错误/废话词。\n"
        '输出 JSON 数组：[{"statement":"...","answers":["..."],"evidence":["doc-01"],'
        '"forbid":["..."]}]'
    )


# --------------------------------------------------------------------------- #
# 调用预算 + 生成器
# --------------------------------------------------------------------------- #
class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise BudgetExhausted
        self.used += 1


async def _request_json(
    model: ChatModel, system: str, user: str, budget: Budget
) -> object | None:
    budget.spend()
    try:
        result = await model.complete([Message("system", system), Message("user", user)])
    except ModelError:
        return None
    return _extract_json(result.content)


async def _request_with_retry(
    model: ChatModel, system: str, make_prompt, n: int, budget: Budget
) -> object | None:
    data = await _request_json(model, system, make_prompt(n), budget)
    if isinstance(data, list) or n <= 1:
        return data
    return await _request_json(model, system, make_prompt(max(1, n // 2)), budget)


async def generate_math(
    model: ChatModel, need: int, *, budget: Budget, seed: int = 42, batch_size: int = 10
) -> list[dict]:
    accepted: list[dict] = []
    seen_statement: set[str] = set()
    seen_calc: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    topics = list(_MATH_TOPICS)
    rng.shuffle(topics)
    max_attempts = max(4, ((need + batch_size - 1) // batch_size) * 3)
    attempts = 0
    while len(accepted) < need and attempts < max_attempts:
        attempts += 1
        want = min(batch_size, max(2, need - len(accepted) + 1))
        topic = topics[attempts % len(topics)]
        try:
            data = await _request_with_retry(
                model, MATH_SYSTEM, lambda n, t=topic: _math_prompt(n, t), want, budget
            )
        except BudgetExhausted:
            break
        if not isinstance(data, list):
            continue
        for raw in data:
            if len(accepted) >= need:
                break
            try:
                item = _validate_math(raw)
            except ValidationError:
                continue
            statement = _norm_statement(item["statement"])
            calc = (item["expr"], item["answer"])
            if statement in seen_statement or calc in seen_calc:
                continue
            seen_statement.add(statement)
            seen_calc.add(calc)
            accepted.append(item)
    return accepted


async def generate_corpus(
    model: ChatModel,
    need_docs: int,
    *,
    budget: Budget,
    seed: int = 42,
    batch_size: int = 30,
) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(seed + 1)
    topics = list(_TOOL_TOPICS)
    rng.shuffle(topics)
    request = min(batch_size, need_docs + max(2, need_docs // 4))
    attempts = 0
    while len(docs) < need_docs and attempts < 3:
        attempts += 1
        chunk = (topics * 2)[attempts * 8 : attempts * 8 + 8]
        try:
            data = await _request_with_retry(
                model, DOCS_SYSTEM, lambda n, c=chunk: _docs_prompt(n, c), request, budget
            )
        except BudgetExhausted:
            break
        if not isinstance(data, list):
            continue
        for raw in data:
            if len(docs) >= need_docs:
                break
            try:
                doc = _validate_doc(raw)
            except ValidationError:
                continue
            if doc["doc_id"] in seen:
                continue
            seen.add(doc["doc_id"])
            docs.append(doc)
    return docs


async def generate_tooluse(
    model: ChatModel,
    need: int,
    docs: list[dict],
    *,
    budget: Budget,
    batch_size: int = 5,
) -> list[dict]:
    accepted: list[dict] = []
    seen_statement: set[str] = set()
    seen_answers: set[tuple[str, ...]] = set()
    distinctive = _distinctive_keywords(docs)
    max_attempts = max(4, ((need + batch_size - 1) // batch_size) * 3)
    attempts = 0
    while len(accepted) < need and attempts < max_attempts:
        attempts += 1
        want = min(batch_size, max(2, need - len(accepted) + 1))
        try:
            data = await _request_with_retry(
                model,
                QA_SYSTEM,
                lambda n: _qa_prompt(n, docs, distinctive),
                want,
                budget,
            )
        except BudgetExhausted:
            break
        if not isinstance(data, list):
            continue
        for raw in data:
            if len(accepted) >= need:
                break
            try:
                item = _validate_tooluse(raw, docs)
            except ValidationError:
                continue
            statement = _norm_statement(item["statement"])
            answers = tuple(sorted(a.casefold() for a in item["answers"]))
            if statement in seen_statement or answers in seen_answers:
                continue
            seen_statement.add(statement)
            seen_answers.add(answers)
            accepted.append(item)
    return accepted


# --------------------------------------------------------------------------- #
# 编排 + CLI
# --------------------------------------------------------------------------- #
@dataclass
class SynthesisReport:
    math: list[dict] = field(default_factory=list)
    tooluse: list[dict] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)
    calls: int = 0
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


async def synthesize(
    model: ChatModel,
    *,
    math_count: int = 50,
    tooluse_count: int = 50,
    seed: int = 42,
    max_calls: int | None = 80,
    out_dir: Path | None = None,
    model_name: str = "",
    docs_wanted: int | None = None,
) -> SynthesisReport:
    budget = Budget(max_calls)
    math_raw = await generate_math(model, math_count, budget=budget, seed=seed)
    math_items = assign_ids(dedupe_math(math_raw))

    tooluse_items: list[dict] = []
    docs: list[dict] = []
    if tooluse_count > 0:
        wanted = docs_wanted or max(24, min(40, tooluse_count // 2 + 4))
        docs = await generate_corpus(model, wanted, budget=budget, seed=seed)
        if docs:
            raw = await generate_tooluse(model, tooluse_count, docs, budget=budget)
            tooluse_items = assign_ids(dedupe_tooluse(raw))

    report = SynthesisReport(
        math=math_items, tooluse=tooluse_items, docs=docs, calls=budget.used
    )
    if out_dir is not None:
        report.counts["math"] = write_split_files(out_dir, "math", math_items, seed)
        report.counts["tooluse"] = write_split_files(out_dir, "tooluse", tooluse_items, seed)
        write_corpus(out_dir, docs)
        write_log(out_dir, _log_rows(math_items, tooluse_items, model_name, seed))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合成 math/tooluse 混合数据集（~100 条）")
    parser.add_argument("--math", type=int, default=50, help="math 目标条数")
    parser.add_argument("--tooluse", type=int, default=50, help="tooluse 目标条数")
    parser.add_argument("--out", default="dataset/synth", help="产物根目录")
    parser.add_argument("--seed", type=int, default=42, help="去重后切分种子")
    parser.add_argument(
        "--max-calls", type=int, default=80, help="模型调用封顶（含失败重试）"
    )
    parser.add_argument(
        "--model", default=os.getenv("EH_MODEL", DEFAULT_MODEL), help="chat 模型名"
    )
    parser.add_argument(
        "--base-url", default=os.getenv("EH_BASE_URL", DEFAULT_BASE_URL), help="端点"
    )
    parser.add_argument(
        "--docs", type=int, default=None, help="tooluse 语料目标篇数（默认按题量推导）"
    )
    return parser


def _print_report(report: SynthesisReport) -> None:
    print(f"calls: {report.calls}")
    print(f"docs:  {len(report.docs)}")
    for domain, counts in report.counts.items():
        sizes = "/".join(str(counts[name]) for name in SPLITS)
        print(f"{domain:8s} accepted={sum(counts.values()):3d} split={sizes}")


async def _run_cli(args, model: OpenAIChatModel) -> SynthesisReport:
    try:
        return await synthesize(
            model,
            math_count=args.math,
            tooluse_count=args.tooluse,
            seed=args.seed,
            max_calls=args.max_calls,
            out_dir=Path(args.out),
            model_name=args.model,
            docs_wanted=args.docs,
        )
    finally:
        await model.aclose()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY") or ""
    if not api_key:
        print(
            "缺少 ARK_API_KEY / EH_API_KEY；请先 `set -a; source .env; set +a`",
            file=sys.stderr,
        )
        return 2
    model = OpenAIChatModel(base_url=args.base_url, model=args.model, api_key=api_key)
    report = asyncio.run(_run_cli(args, model))
    _print_report(report)
    shortfall = len(report.math) < args.math or len(report.tooluse) < args.tooluse
    if shortfall:
        print(
            f"配额不足：math {len(report.math)}/{args.math}，"
            f"tooluse {len(report.tooluse)}/{args.tooluse}；已落盘通过校验的数据。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
