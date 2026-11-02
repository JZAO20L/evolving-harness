"""course/ch09 · prompt-rsi —— 模型驱动的 eval / locate + train/val/test 评估纪律。

``rsi.py`` = ch08 快照 + 本课增量；``agent.py`` = ch08 原样（冻结，被改进对象）。

ch08 已跑通「reward → locate → propose → gate/版本」的无训练外环，但两个短板：
1. **评分只能同步**：``evaluate()`` 直接调 ``scorer.score``（未 await、不传 model），规则打分之外接不上模型；
2. **没有评估纪律**：全部任务混在一起，RSI 可以对着同一个集子反复调 → 背题（评测污染）。

本课把 RSI 明确成 **二阶 Agent / meta-agent**：

```
act   = rollout(current config, train)   → 分数 + trace（观测）
react = locate(定位) → propose(改组件)    # 本课把 locate/propose 模型化，且可合并为一次生成
```

并补齐**评估纪律**：train 上按轮改进（轮级门禁），每 ``val_every`` 轮在 train+val 上对**血线**
（watermark）做段级门禁（严格双方提升才 commit，否则 rollback），test 只在最后碰一次报泛化。

设计取向（与 ch08 一脉相承）：
- **双后端统一接口**：``Judge`` / ``Locator`` / ``Proposer`` / ``Improver`` 都各自持有依赖
  （rules / chat LLM / System One 决策模型），循环只面对 async 接口，不把 model 到处穿透。
- **System One 只判断、不生成**：评分/定位可选决策模型后端；改写策略仍由 chat LLM 完成。
- **规则后端默认**：离线 L0 全规则；L1 冒烟显式接真模型后端，"模型驱动"体现在用例与实现。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx
from agent import (
    RETRYABLE_STATUS,
    AgentLoop,
    ChatModel,
    Message,
    ModelError,
    SkillLibrary,
    Usage,
    _extract_json,
)


def _sha256(text: str) -> str:
    """可变段 canonical JSON 的 SHA-256 —— 每个 proposal 的版本指纹（版本 = 可变段哈希）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _append_line(existing: str, line: str) -> str:
    return f"{existing}\n{line}".strip("\n")


# --------------------------------------------------------------------------- #
# 0. 任务集 + 数据集拆分加载
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """一条评估任务。agent 只看到 ``statement``；``checks`` 是**私有 rubric**（防评测污染/self-drift）。"""

    id: str
    statement: str
    checks: tuple[str, ...] = ()   # 每条是一个"必须含"的子串；`~word` 前缀 = "必须不含 word"


def load_split(name: str) -> list[Task]:
    """读仓库根 ``dataset/rsi/{name}.jsonl``（与 ``course/`` 平级、跨章共享的探针集）。"""
    if name not in {"train", "val", "test"}:
        raise ValueError(f"未知 split: {name!r}（只支持 train/val/test）")
    path = Path(__file__).resolve().parents[2] / "dataset" / "rsi" / f"{name}.jsonl"
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks.append(Task(id=row["id"], statement=row["statement"], checks=tuple(row["checks"])))
    return tasks


# --------------------------------------------------------------------------- #
# 1. prompt 定义：prompt ≠ system prompt 一个口
# --------------------------------------------------------------------------- #
@dataclass
class TaskFrame:
    """把原始任务句包成 user 消息的模板：前缀/后缀 + few-shot exemplars（提问方式与示范）。"""

    prefix: str = ""
    suffix: str = ""
    exemplars: tuple[tuple[str, str], ...] = ()   # (示例任务, 示例答案)

    def render(self, statement: str) -> str:
        if not self.exemplars:
            return f"{self.prefix}{statement}{self.suffix}"
        shots = "\n\n".join(f"例：{q}\n答：{a}" for q, a in self.exemplars)
        return f"{self.prefix}{shots}\n\n任务：{statement}{self.suffix}"


@dataclass
class PromptPack:
    """RSI 能改的「提示面」三件套（guidance 面 + 包装/格式缝）。"""

    guidance: str = "你是教学 Agent。先用工具拿事实，再给出最终结论。"
    task_frame: TaskFrame = field(default_factory=TaskFrame)
    final_format: str = ""        # 输出形态指令（verifier rubric 最常抓的格式违约点）

    def render_system(self) -> str:
        s = self.guidance
        if self.final_format:
            s += "\n\n输出要求：" + self.final_format
        return s

    def render_user(self, statement: str) -> str:
        """RSI 组合 agent 输入的**唯一入口**：agent 只看到这里产出的 user 消息（不泄 checks）。"""
        return self.task_frame.render(statement)

    def canonical(self) -> dict:
        return {"guidance": self.guidance,
                "task_frame": asdict(self.task_frame),
                "final_format": self.final_format}


# --------------------------------------------------------------------------- #
# 2. AgentConfig —— 被版本登记/可回滚的「可改进配置」
# --------------------------------------------------------------------------- #
@dataclass
class AgentConfig:
    """一个可跑 agent 的完整配置。本课可变面 = ``prompt``；其它字段只是承载（ch10~12 逐章放开）。"""

    name: str = "agent:ch09"
    prompt: PromptPack = field(default_factory=PromptPack)
    max_steps: int = 8
    skills: tuple = ()            # tuple[Skill]
    base_tools: tuple = ()        # tuple[Tool]

    def canonical(self) -> dict:
        return {"name": self.name,
                "prompt": self.prompt.canonical(),
                "max_steps": self.max_steps,
                "skills": [s.name for s in self.skills],
                "base_tools": [t.name for t in self.base_tools]}

    def hash(self) -> str:
        return _sha256(json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# 3. rollout：把 config × task 跑一次 → 由 agent 产出结果（只组合输入，不动 agent.py）
# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    task_id: str
    answer: str = ""
    finished: bool = False
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[str, ...] = ()   # 实际执行过的工具名（locate 判断"查没查"用）
    score: float = 0.0                 # verifier 回填的 pointwise 分


async def rollout(config: AgentConfig, task: Task, model: ChatModel) -> RolloutResult:
    """用 ``config`` 构造一个 AgentLoop，跑 ``task.statement``（经 prompt 包装）。

    镜像产品「同一循环、不同 YAML」：改 prompt → 输入变 → 行为变；**不动 agent.py**。
    """
    skills = SkillLibrary(*config.skills) if config.skills else None
    loop = AgentLoop(model, system=config.prompt.render_system(),
                     skills=skills, base_tools=list(config.base_tools),
                     max_steps=config.max_steps)
    res = await loop.run(config.prompt.render_user(task.statement))
    return RolloutResult(task_id=task.id, answer=res.answer, finished=res.finished,
                         steps=res.steps, usage=res.usage,
                         tool_calls=tuple(tc.name for tc in res.tool_calls))


# --------------------------------------------------------------------------- #
# 4. Judge：统一 async 评分接口（规则 / chat LLM / System One 决策模型）
# --------------------------------------------------------------------------- #
class RuleScorer:
    """pointwise **确定性**打分核心：正 check 要含词，`~词` 要不含。规则优先、离线可测。"""

    def __init__(self, casefold: bool = True) -> None:
        self.casefold = casefold

    def _norm(self, text: str) -> str:
        return text.lower() if self.casefold else text

    def check_hit(self, check: str, answer: str) -> bool:
        neg = check.startswith("~")
        key = self._norm(check[1:] if neg else check)
        present = key in self._norm(answer)
        return (not present) if neg else present

    def score(self, task: Task, r: RolloutResult) -> float:
        a = r.answer
        if not task.checks:
            return 1.0 if a.strip() else 0.0
        hits = sum(1 for c in task.checks if self.check_hit(c, a))
        return hits / len(task.checks)


def _parse_score(text: str) -> float:
    """从打分模型的回答里剥出 0..1 的数（容错夹带叙述）。"""
    m = re.search(r"0\.\d+|\b1(?:\.0)?\b|\b0\b", text)
    if not m:
        return 0.0
    v = float(m.group(0))
    return max(0.0, min(1.0, v))


class Judge:
    """统一评分接口：``async score(task, r) -> [0,1]``。后端各自持有自己的依赖。"""

    async def score(self, task: Task, r: RolloutResult) -> float:
        raise NotImplementedError


class RuleJudge(Judge):
    """规则后端的 async façade（包 ``RuleScorer`` 的确定性逻辑）。"""

    def __init__(self, casefold: bool = True) -> None:
        self.scorer = RuleScorer(casefold)

    async def score(self, task: Task, r: RolloutResult) -> float:
        return self.scorer.score(task, r)


def _rubric(task: Task) -> str:
    return "\n".join(f"- {c}" for c in task.checks) or "（无 rubric）"


class LLMJudge(Judge):
    """chat LLM pointwise：打分模型把 answer 对照 rubric 打 0..1 分。"""

    system = ("你是严格、一成不变的评分员。把候选回答对照 rubric 逐项判，只回一个 0 到 1 的小数"
              "（1=完全符合）。不要解释。")

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def score(self, task: Task, r: RolloutResult) -> float:
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"任务：{task.statement}\nrubric：\n{_rubric(task)}\n答案：\n{r.answer}\n\n分数：")),
        ])
        return _parse_score(res.content)


class SystemOneJudge(Judge):
    """System One **决策模型**判分（``score`` 型）：只判断，不生成自然语言。"""

    def __init__(self, client: SystemOneClient, question_id: str = "rubric_score_v1") -> None:
        self.client = client
        self.question_id = question_id
        self.criteria = ["完全不符合 rubric", "部分符合 rubric", "完全符合 rubric"]

    async def score(self, task: Task, r: RolloutResult) -> float:
        answers = await self.client.ask(
            state={"task": task.statement, "rubric": list(task.checks), "answer": r.answer},
            questions={self.question_id: {
                "type": "score",
                "instructions": "候选回答与 rubric 的符合程度如何？",
                "criteria": self.criteria,
            }},
        )
        answer = answers.get(self.question_id) or {}
        raw = float(answer.get("score", 0.0))
        top = len(self.criteria) - 1
        return max(0.0, min(1.0, raw / top)) if top else 0.0


# --------------------------------------------------------------------------- #
# 5. 评估：rollout × n → Judge 判分（全 split 同一 n，修掉 ch08 的不对称）
# --------------------------------------------------------------------------- #
@dataclass
class TaskScore:
    """单个 task 在一组 rollout 上的分数（组内多 rollout 取 mean/best/min 形成差异表现）。"""

    task_id: str
    mean: float
    best: float
    min_: float


async def evaluate(tasks, config: AgentConfig, model: ChatModel, judge: Judge | None = None,
                   n_rollouts: int = 1) -> list[TaskScore]:
    """rollout 每个 task 各 ``n_rollouts`` 次，**await** ``judge.score`` 后返回 (mean, best, min)。

    ``model`` 只用于 rollout；打分依赖由 ``judge`` 自己持有（规则 / LLM / System One 可换）。
    """
    judge = judge or RuleJudge()
    out: list[TaskScore] = []
    for t in tasks:
        scores: list[float] = []
        for _ in range(n_rollouts):
            r = await rollout(config, t, model)
            scores.append(await judge.score(t, r))
        out.append(TaskScore(t.id, _mean(scores), max(scores), min(scores)))
    return out


def aggregate(scores: list[TaskScore]) -> float:
    """整组聚合分（默认各 task 均值再平均）。"""
    return _mean([s.mean for s in scores])


@dataclass
class Differential:
    """组内差分（pairwise reward 的结果）：候选相对基线的差异表现。"""

    delta: float                          # 整组：候选 mean 分 - 基线 mean 分
    preference: int                       # +1 候选优 · 0 平 · -1 基线优
    per_task: list[tuple[str, float]]     # (task_id, 单 task 差分)


class PairScorer:
    """对同 task 的 baseline vs candidate **组**做差分 → 差异表现（reward=pairwise）。"""

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def diff(self, base: list[TaskScore], cand: list[TaskScore]) -> Differential:
        per = [(b.task_id, c.mean - b.mean) for b, c in zip(base, cand)]
        delta = _mean([d for _, d in per])
        pref = 1 if delta > self.eps else (-1 if delta < -self.eps else 0)
        return Differential(delta, pref, per)


# --------------------------------------------------------------------------- #
# 6. Locator：async credit assignment —— 规则 / LLM / System One
# --------------------------------------------------------------------------- #
@dataclass
class Diagnosis:
    component: str        # guidance | frame | format | skip | halluc | ok
    finding: str
    task_id: str = ""
    score: float = 0.0


COMPONENTS = ("format", "frame", "halluc", "skip", "guidance", "ok")

SYS_ONE_COMPONENT_CRITERIA: dict[str, str] = {
    "format": "输出了 rubric 禁止的内容，或没按要求的输出形态作答",
    "frame": "任务包装/示范不足，漏掉了 rubric 要求的要点",
    "halluc": "没查工具/事实就下结论，导致缺要点",
    "skip": "空答，或步数耗尽没给出最终结论",
    "guidance": "分数低但规则看不出具体组件问题",
    "信息不足": "证据不足，无法判断问题出在哪个组件",
}


class Locator:
    """统一定位接口：``async locate(task, r, score) -> Diagnosis``。"""

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        raise NotImplementedError


class RuleLocator(Locator):
    """规则式 credit assignment：从总分低 + trace 反推问题出在 prompt 哪个组件（ch08 逻辑）。"""

    def __init__(self, scorer: RuleScorer | None = None) -> None:
        self.scorer = scorer or RuleScorer()

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        if not r.answer.strip():
            return Diagnosis("skip", "空答：模型没有给出任何回答。", task.id, score)
        if not r.finished:
            return Diagnosis("skip", "步数耗尽：没在预算内给出最终结论。", task.id, score)
        if score >= 1.0:
            return Diagnosis("ok", "无错。", task.id, score)
        # 禁词违约优先级更高：输出了"不该出现的内容"是硬伤，先抓格式问题。
        for c in task.checks:
            if c.startswith("~") and not self.scorer.check_hit(c, r.answer):
                return Diagnosis("format", f"输出了禁止内容：{c[1:]!r}。归因到 final_format 格式违约。",
                                 task.id, score)
        # 再查漏掉的正面要点：没查工具 → 疑似幻觉；查了但仍缺 → 任务包装/示范不足。
        for c in task.checks:
            if c.startswith("~") or self.scorer.check_hit(c, r.answer):
                continue
            if not r.tool_calls:
                return Diagnosis("halluc", f"缺要点 {c!r} 且全程没调工具：疑似没查事实就下结论。归因到 guidance。",
                                 task.id, score)
            return Diagnosis("frame", f"缺要点 {c!r}：任务包装/示范不足。归因到 task_frame。",
                             task.id, score)
        return Diagnosis("guidance", "分数低但规则命中，留给模型驱动分析。", task.id, score)


class LLMLocator(Locator):
    """chat LLM 归因：喂 task + rubric + trace + 分数，让模型选组件并给一句 finding。"""

    system = ("你是 Agent 失败归因分析员。读任务、私有 rubric、候选回答与工具调用记录，判断问题主要"
              "出在哪个 prompt 组件。只回 JSON：{\"component\": 组件名, \"finding\": 一句话依据}。"
              "组件名只能是 format/frame/halluc/skip/guidance/ok。证据不足时回 ok。")

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        if score >= 1.0:
            return Diagnosis("ok", "满分，无需定位。", task.id, score)
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"任务：{task.statement}\nrubric：\n{_rubric(task)}\n"
                f"回答：\n{r.answer}\n工具调用：{', '.join(r.tool_calls) or '无'}\n"
                f"分数：{score:.2f}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict) or data.get("component") not in COMPONENTS:
            return Diagnosis("ok", "定位解析失败，保守不改。", task.id, score)
        return Diagnosis(str(data["component"]), str(data.get("finding") or ""), task.id, score)


class SystemOneLocator(Locator):
    """System One **决策模型**归因（``choice`` 型）：选组件，带「信息不足」兜底。"""

    def __init__(self, client: SystemOneClient, question_id: str = "failure_component_v1") -> None:
        self.client = client
        self.question_id = question_id

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        answers = await self.client.ask(
            state={"task": task.statement, "rubric": list(task.checks), "answer": r.answer,
                   "tool_calls": list(r.tool_calls), "score": score},
            questions={self.question_id: {
                "type": "choice",
                "instructions": "该回答未通过评测，判断问题主要出在哪个 prompt 组件。",
                "criteria": SYS_ONE_COMPONENT_CRITERIA,
            }},
        )
        answer = answers.get(self.question_id) or {}
        choice = str(answer.get("choice", "信息不足"))
        component = choice if choice in COMPONENTS else "ok"
        finding = f"决策模型选择：{choice}（confidence={answer.get('confidence', 'n/a')}）"
        return Diagnosis(component, finding, task.id, score)


# --------------------------------------------------------------------------- #
# 7. System One 决策通道（beta）：只判断、不生成
# --------------------------------------------------------------------------- #
class SystemOneClient:
    """ARK System One（beta）客户端：``ask(state, questions) -> answers``。

    规格（``/home/tiger/docs/models/ark_decision.md``）：POST ``/api/v1/systemone``，头
    ``ark-beta-systemone-api: true``，体 ``{model, state, questions}``；响应 ``answers`` 按问题 ID
    返回结构化判断。**仅判断、不生成**（生成只能走 chat LLM）。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model:
            raise ValueError("model 必填")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._headers = {"ark-beta-systemone-api": "true"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._http = http_client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def ask(self, state, questions: dict) -> dict:
        """返回 ``{question_id: {...}}``；可重试状态(408/429/5xx)与传输错误指数退避。"""
        payload = {"model": self.model, "state": state, "questions": questions}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.post(
                    f"{self.base_url}/systemone", json=payload, headers=self._headers)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    data = response.json()
                    answers = data.get("answers")
                    if not isinstance(answers, dict):
                        raise ModelError(f"System One 响应缺 answers: {str(data)[:200]}")
                    return answers
                if response.status_code not in RETRYABLE_STATUS:
                    raise ModelError(
                        f"System One 返回 {response.status_code}: {response.text[:200]}")
                last_error = ModelError(f"System One HTTP {response.status_code}")
            if attempt < self.max_retries:
                backoff = min(8.0, 0.5 * (2**attempt)) + random.uniform(0, 0.2)
                await asyncio.sleep(backoff)
        raise ModelError(f"System One 请求重试后仍失败: {last_error}")

    async def aclose(self) -> None:
        await self._http.aclose()


# --------------------------------------------------------------------------- #
# 8. Proposer：依 Diagnosis 改 prompt（规则 / chat LLM）
# --------------------------------------------------------------------------- #
class Proposer:
    """统一改进接口：``async propose(config, task, r, diag) -> 新 AgentConfig``（不可变）。"""

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        raise NotImplementedError


class RuleProposer(Proposer):
    """确定性 prompt 编辑：依 ``Diagnosis.component`` 编辑 ``PromptPack`` 对应字段。"""

    _RULE: ClassVar[dict[str, str]] = {
        "format":  "严格按输出要求作答；有一个要点没覆盖就算不通过，不要夹带题干里禁止的内容。",
        "frame":   "作答要逐点覆盖任务的所有要求，先给结论再分点，确保每个要点都出现。",
        "halluc":  "不要凭印象下结论：必须先调用工具拿到事实证据，再给出最终答案。",
        "skip":    "若任务未完成或不确定，必须明确说明卡在哪里，并给出可继续的查证路径。",
    }

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        if diag.component in ("ok", "guidance") or diag.component not in self._RULE:
            return config                      # 无有效归因 → 不改（候选 = 基线）
        cand = copy.deepcopy(config)           # 不可变：propose 返回新 config，不污染当期
        if diag.component == "format":
            cand.prompt.final_format = _append_line(cand.prompt.final_format, self._RULE["format"])
        else:
            cand.prompt.guidance = _append_line(cand.prompt.guidance, self._RULE[diag.component])
        return cand


class LLMProposer(Proposer):
    """chat LLM 按 Diagnosis 生成一条可追加的 prompt 规则（结构化 JSON）。"""

    system = ("你是 Agent 的提示改进器。根据一次失败归因，写出能修复该问题的一条简短、可执行的"
              "system 指令。只回 JSON：{\"text\": 要追加的规则}（format 类问题也可给 "
              "{\"final_format\": 完整输出要求}）。不要解释。")

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        if diag.component not in COMPONENTS or diag.component == "ok":
            return config
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"归因组件：{diag.component}\n依据：{diag.finding}\n当前 guidance：\n"
                f"{config.prompt.guidance}\n当前 final_format：{config.prompt.final_format or '（空）'}\n"
                f"任务：{task.statement}\n回答：\n{r.answer}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict):
            return config
        cand = copy.deepcopy(config)
        text = str(data.get("text") or "").strip()
        fmt = str(data.get("final_format") or "").strip()
        if fmt:
            cand.prompt.final_format = fmt
        if text:
            cand.prompt.guidance = _append_line(cand.prompt.guidance, text)
        return cand


# --------------------------------------------------------------------------- #
# 9. Improver：两阶段（locate → propose）vs 合并（thought+act 一次生成）
# --------------------------------------------------------------------------- #
class Improver:
    """统一改进器接口：``async improve(config, task, r) -> (Diagnosis, AgentConfig)``。"""

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        raise NotImplementedError


class TwoStageImprover(Improver):
    """先 ``locate`` 再 ``propose``（两个后端可各自独立替换）。"""

    def __init__(self, locator: Locator, proposer: Proposer) -> None:
        self.locator = locator
        self.proposer = proposer

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        diag = await self.locator.locate(task, r, r.score)
        return diag, await self.proposer.propose(config, task, r, diag)


class MergedLLMImprover(Improver):
    """**一次** chat 调用同时完成定位 + 改写（meta-agent 的 thought+act 合并生成）。

    返回 ``{component, finding, guidance, final_format}``；``guidance`` 为改写后的**完整**文本
    （空串 = 不改），``final_format`` 同理。解析失败 → ``ok`` + 原 config（保守不动）。
    """

    system = ("你是 Agent 策略改进器。读完任务、私有 rubric、候选回答和分数后，一次性完成两件事："
              "1）定位问题出在哪个 prompt 组件；2）改写 prompt 修复它。只回 JSON："
              "{\"component\": format/frame/halluc/skip/guidance/ok, \"finding\": 一句话依据, "
              "\"guidance\": 改写后的完整 system guidance（不改则空串）, "
              "\"final_format\": 改写后的完整输出要求（不改则空串）}。")

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"任务：{task.statement}\nrubric：\n{_rubric(task)}\n回答：\n{r.answer}\n"
                f"工具调用：{', '.join(r.tool_calls) or '无'}\n分数：{r.score:.2f}\n"
                f"当前 guidance：\n{config.prompt.guidance}\n"
                f"当前 final_format：{config.prompt.final_format or '（空）'}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict):
            return Diagnosis("ok", "合并改进解析失败，保守不改。", task.id, r.score), config
        component = str(data.get("component") or "ok")
        diag = Diagnosis(component if component in COMPONENTS else "ok",
                         str(data.get("finding") or ""), task.id, r.score)
        cand = copy.deepcopy(config)
        guidance = str(data.get("guidance") or "").strip()
        final_format = str(data.get("final_format") or "").strip()
        if guidance:
            cand.prompt.guidance = guidance
        if final_format:
            cand.prompt.final_format = final_format
        return diag, cand


# --------------------------------------------------------------------------- #
# 10. 门禁：轮级（train-only）＋ 段级（train+val 对血线）
# --------------------------------------------------------------------------- #
class Gate:
    """轮级门禁：候选相对当前 train 分数提升且不回归才 ``promote``；否则 ``reject``（只看 train）。"""

    def __init__(self, min_delta: float = 0.0, regress_tol: float = 0.0) -> None:
        self.min_delta = min_delta
        self.regress_tol = regress_tol

    def decide(self, base: list[TaskScore], cand: list[TaskScore]) -> tuple[str, Differential]:
        diff = PairScorer().diff(base, cand)
        worst = min((d for _, d in diff.per_task), default=0.0)   # 最差单 task 差分（不回归约束）
        if diff.delta > self.min_delta and worst >= -self.regress_tol:
            return "promote", diff
        return "reject", diff


@dataclass
class Checkpoint:
    """段级**血线**：一份 config + 它在 train/val 上的完整分数（回滚时直接复用，免重算）。"""

    config: AgentConfig
    train_scores: list[TaskScore]
    val_scores: list[TaskScore]
    version: str | None = None


class SegmentGate:
    """段级门禁：**严格双方提升**（train 与 val 都超过血线）才 ``commit``；否则 ``rollback``/``hold``。"""

    def __init__(self, train_min_delta: float = 0.0, val_min_delta: float = 0.0,
                 train_regress_tol: float = 0.0) -> None:
        self.train_min_delta = train_min_delta
        self.val_min_delta = val_min_delta
        self.train_regress_tol = train_regress_tol

    def decide(self, watermark: Checkpoint, cur_train: list[TaskScore],
               cur_val: list[TaskScore]) -> tuple[str, float, float]:
        train_delta = aggregate(cur_train) - aggregate(watermark.train_scores)
        val_delta = aggregate(cur_val) - aggregate(watermark.val_scores)
        worst = min((c.mean - b.mean for b, c in zip(watermark.train_scores, cur_train)), default=0.0)
        if (train_delta > self.train_min_delta and val_delta > self.val_min_delta
                and worst >= -self.train_regress_tol):
            return "commit", train_delta, val_delta
        unchanged = (all(abs(c.mean - b.mean) < 1e-9 for b, c in zip(watermark.train_scores, cur_train))
                     and all(abs(c.mean - b.mean) < 1e-9 for b, c in zip(watermark.val_scores, cur_val)))
        return ("hold" if unchanged else "rollback"), train_delta, val_delta


class VersionRegistry:
    """版本登记：把「config × 分数 × 裁决」按 config 哈希登记为一条版本；可 list / rollback。"""

    def __init__(self, storage_dir: str | Path) -> None:
        self.path = Path(storage_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._file = self.path / "versions.jsonl"

    def register(self, config: AgentConfig, *, score: float, verdict: str,
                 parent: str | None = None, meta: dict | None = None) -> str:
        version = config.hash()
        entry = {"version": version, "config": config.canonical(), "score": score,
                 "verdict": verdict, "parent": parent, "meta": meta or {}}
        with self._file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return version

    def list(self) -> list[dict]:
        if not self._file.exists():
            return []
        return [json.loads(line) for line in self._file.read_text(encoding="utf-8").splitlines() if line]

    def get(self, version: str) -> dict | None:
        return next((e for e in self.list() if e["version"] == version), None)

    def latest(self) -> dict | None:
        entries = self.list()
        return entries[-1] if entries else None

    def rollback(self, version: str) -> AgentConfig:
        """返回该版本 config 的还原（prompt / max_steps / name；skills & base_tools 名还原留给 ch10）。"""
        entry = self.get(version)
        if entry is None:
            raise KeyError(f"没有版本 {version}")
        cfg = entry["config"]
        return AgentConfig(name=cfg["name"],
                           prompt=PromptPack(guidance=cfg["prompt"]["guidance"],
                                             task_frame=TaskFrame(**cfg["prompt"]["task_frame"]),
                                             final_format=cfg["prompt"]["final_format"]),
                           max_steps=cfg["max_steps"])


# --------------------------------------------------------------------------- #
# 11. Rsi —— meta-agent 外环：act(rollout+eval) / react(locate+propose) / 两级门禁
# --------------------------------------------------------------------------- #
@dataclass
class RoundOutcome:
    round: int
    verdict: str                      # promote | reject | noop
    version: str | None
    proposed: AgentConfig
    delta: float
    candidate_scores: list[TaskScore]

    @property
    def score(self) -> float:
        return aggregate(self.candidate_scores)


@dataclass
class SegmentOutcome:
    segment: int
    round: int                        # 触发该段的轮次
    verdict: str                      # commit | rollback | hold
    train_delta: float                # 相对血线
    val_delta: float
    version: str | None


@dataclass
class RsiReport:
    baseline_train: list[TaskScore]
    baseline_val: list[TaskScore]
    rounds: list[RoundOutcome]
    segments: list[SegmentOutcome]
    final_version: str
    final_config: AgentConfig
    test_score: float

    @property
    def baseline_train_score(self) -> float:
        return aggregate(self.baseline_train)

    @property
    def baseline_val_score(self) -> float:
        return aggregate(self.baseline_val)


class Rsi:
    """RSI 二阶 Agent 外环：train 上按轮改进；每 ``val_every`` 轮对血线做段级门禁；test 只碰一次。

    - ``judge``：async 评分后端（``RuleJudge`` / ``LLMJudge`` / ``SystemOneJudge``）。
    - ``improver``：``TwoStageImprover``（locate+propose）或 ``MergedLLMImprover``（一次生成）。
    - ``n_rollouts``：**所有 split 同一值**（修正 ch08 基线=1、候选=N 的不对称）。
    """

    def __init__(self, config: AgentConfig, model: ChatModel,
                 *, train, val, test,
                 judge: Judge | None = None, improver: Improver | None = None,
                 gate: Gate | None = None, segment_gate: SegmentGate | None = None,
                 storage_dir: str | Path | None = None, n_rollouts: int = 1,
                 max_rounds: int = 2, val_every: int = 2) -> None:
        self.config = config
        self.model = model
        self.train = list(train)
        self.val = list(val)
        self.test = list(test)
        self.judge = judge or RuleJudge()
        self.improver = improver or TwoStageImprover(RuleLocator(), RuleProposer())
        self.gate = gate or Gate()
        self.segment_gate = segment_gate or SegmentGate()
        self.registry = VersionRegistry(storage_dir) if storage_dir else None
        self.n_rollouts = n_rollouts
        self.max_rounds = max_rounds
        self.val_every = max(1, val_every)

    def _register(self, config: AgentConfig, scores, verdict: str,
                  parent: str | None = None, meta: dict | None = None) -> str | None:
        if self.registry is None:
            return None
        return self.registry.register(config, score=aggregate(scores), verdict=verdict,
                                      parent=parent, meta=meta)

    async def evaluate(self, config: AgentConfig, tasks) -> list[TaskScore]:
        return await evaluate(tasks, config, self.model, self.judge, self.n_rollouts)

    async def propose_improvement(self, current: AgentConfig,
                                  cur_scores: list[TaskScore]) -> tuple[Diagnosis, AgentConfig]:
        """定位**当前 train** 上最差的 task → 回填分数 → improver 产出 (诊断, 候选 config)。"""
        worst = min(cur_scores, key=lambda s: s.mean)
        task = next(t for t in self.train if t.id == worst.task_id)
        r = await rollout(current, task, self.model)
        r.score = worst.mean
        return await self.improver.improve(current, task, r)

    def _commit_segment(self, index: int, round_index: int, current: AgentConfig,
                        cur_train, cur_val,
                        watermark: Checkpoint) -> tuple[str, Checkpoint, SegmentOutcome]:
        verdict, train_delta, val_delta = self.segment_gate.decide(watermark, cur_train, cur_val)
        if verdict == "commit":
            new_watermark = Checkpoint(current, cur_train, cur_val, current.hash())
            meta = {"segment": index, "train_delta": train_delta, "val_delta": val_delta}
            version = self._register(current, cur_train, "commit", parent=watermark.version, meta=meta)
        else:
            new_watermark = watermark
            version = watermark.version
        outcome = SegmentOutcome(index, round_index, verdict, train_delta, val_delta, version)
        return verdict, new_watermark, outcome

    async def run(self) -> RsiReport:
        base_train = await self.evaluate(self.config, self.train)
        base_val = await self.evaluate(self.config, self.val)
        base_version = self._register(self.config, base_train, "accepted",
                                      meta={"split": "train", "val_score": aggregate(base_val)})

        watermark = Checkpoint(self.config, base_train, base_val, base_version)
        current, cur_train, cur_val = self.config, base_train, base_val
        rounds: list[RoundOutcome] = []
        segments: list[SegmentOutcome] = []
        last_round = 0

        for i in range(self.max_rounds):
            last_round = i + 1
            _diag, cand = await self.propose_improvement(current, cur_train)
            if cand.hash() == current.hash():
                rounds.append(RoundOutcome(i + 1, "noop", current.hash(), cand, 0.0, cur_train))
                # 无可改进点 → 停；但此前 promote 的改动仍要在段边界结算，不能悬空。
                if (i + 1) % self.val_every == 0:
                    if current.hash() != watermark.config.hash():
                        cur_val = await self.evaluate(current, self.val)
                    seg_verdict, watermark, outcome = self._commit_segment(
                        len(segments) + 1, i + 1, current, cur_train, cur_val, watermark)
                    segments.append(outcome)
                    if seg_verdict == "rollback":
                        current, cur_train = watermark.config, watermark.train_scores
                        cur_val = watermark.val_scores
                break
            cand_train = await self.evaluate(cand, self.train)
            verdict, diff = self.gate.decide(cur_train, cand_train)   # 轮级：只看 train
            version = self._register(cand, cand_train, verdict, parent=current.hash())
            rounds.append(RoundOutcome(i + 1, verdict, version, cand, diff.delta, cand_train))
            if verdict == "promote":
                current, cur_train = cand, cand_train

            if (i + 1) % self.val_every == 0:
                if current.hash() != watermark.config.hash():
                    cur_val = await self.evaluate(current, self.val)   # 段级才碰 val
                seg_verdict, watermark, outcome = self._commit_segment(
                    len(segments) + 1, i + 1, current, cur_train, cur_val, watermark)
                segments.append(outcome)
                if seg_verdict == "rollback":       # 回到血线 config 与分数（免重算）
                    current, cur_train, cur_val = watermark.config, watermark.train_scores, watermark.val_scores

        # 末轮若还没做段级门禁，且当前已偏离血线 → 补一次（否则已偏离的改动永远悬空）
        if (rounds and last_round % self.val_every != 0
                and current.hash() != watermark.config.hash()):
            cur_val = await self.evaluate(current, self.val)
            seg_verdict, watermark, outcome = self._commit_segment(
                len(segments) + 1, last_round, current, cur_train, cur_val, watermark)
            segments.append(outcome)
            if seg_verdict == "rollback":
                current, cur_train, cur_val = watermark.config, watermark.train_scores, watermark.val_scores

        test_score = aggregate(await self.evaluate(current, self.test))   # test 只碰一次
        return RsiReport(base_train, base_val, rounds, segments, current.hash(), current, test_score)


__all__ = [
    "AgentConfig",
    "Checkpoint",
    "Diagnosis",
    "Differential",
    "Gate",
    "Improver",
    "Judge",
    "LLMJudge",
    "LLMLocator",
    "LLMProposer",
    "Locator",
    "MergedLLMImprover",
    "PairScorer",
    "PromptPack",
    "Proposer",
    "RoundOutcome",
    "Rsi",
    "RsiReport",
    "RuleJudge",
    "RuleLocator",
    "RuleProposer",
    "RuleScorer",
    "SegmentGate",
    "SegmentOutcome",
    "SystemOneClient",
    "SystemOneJudge",
    "SystemOneLocator",
    "Task",
    "TaskFrame",
    "TaskScore",
    "TwoStageImprover",
    "VersionRegistry",
    "_parse_score",
    "aggregate",
    "evaluate",
    "load_split",
    "rollout",
]
