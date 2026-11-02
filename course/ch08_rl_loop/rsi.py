"""course/ch08 · rl-loop —— 无训练 RL 式自改进闭环（RSI 轨第 1 章）。

ch01~ch07 让 agent 有「调模型 / ReAct 循环 / 工具 / 技能 / 记忆 / 规划 / 子代理」，但**改进它本身**不需要
人类重写 prompt：本课补上 **RSI 外环**——把 agent 当作可洞察、可打分、只改 prompt 的**被改进对象**，
用类似 RL 的「reward → 定位问题(credit assignment) → propose → gate/版本」闭环做训练无关(self)改进。

对齐产线设计：product 的 `evolving_harness/rsi.py` 只有 seam（`Rsi` 持一个 `Agent`，标注 `propose() /
verify() / gate()`），无实现——本章是**洁净室实现**镜像其概念。并贯彻 DESIGN 约束：

- **只改 prompt，永不改代码**：`agent.py`（ch07 快照）一行不动；`rollout()` 只组合模型**输入**
  （`AgentLoop(system=pack.guidance)` ＋ `run(pack.render_user(statement))`），镜像「RSI 改 YAML/prompt 不改 code」。
- **prompt 三件套**（回答"除 system 还能改什么"）：`guidance`(system) + `task_frame`(user 包装模板/few-shot)
  + `final_format`(输出形态) —— 都是"模型看到什么文本"的配置，任意改 → hash 变 → 新版本。
- **reward = verifier**：pointwise（`RuleScorer` 确定性 / `JudgeScorer` 打分模型）或 **pairwise**（`PairScorer`
  对同 task 的 baseline vs candidate 组做差分 → 差异表现）。
- **credit assignment = `Locator`**：不只给总分，归因到**哪一环/哪个 prompt 组件**出错，propose 才有的放矢。
- **评测污染隔离**：`Task.checks` 私有（verifier 用），`rollout` 只喂 `statement` 给 agent。
- **门禁 + 版本登记/回滚**：分数提升且不回归才 `promote`；每次提议登记版本，可 `rollback` 到任一版 config。
- **贯穿原则**：顺序 async、循环内不并行；规则优先、模型低频（本课 `Proposer` 确定性、`RuleScorer` 主导；
  `JudgeScorer` 供真模型/后续章）。可变面只开 prompt，ch09 起才逐步开 skills/memory/agent 结构。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

from agent import AgentLoop, ChatModel, Message, SkillLibrary, Usage


def _sha256(text: str) -> str:
    """可变段 canonical JSON 的 SHA-256 —— 每个 proposal 的版本指纹（DESIGN：版本 = 可变段哈希）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# 0. 任务集（task set）：``statement`` 公开给 agent，``checks`` 私有给 verifier
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """一条评估任务。agent 只看到 ``statement``；``checks`` 是**私有 rubric**（防评测污染/self-drift）。"""

    id: str
    statement: str
    checks: tuple[str, ...] = ()   # 每条是一个"必须含"的子串；`~word` 前缀 = "必须不含 word"


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
    """RSI 能改的「提示面」三件套（DESIGN §7.0 的 guidance 面 + 包装/格式缝）。"""

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
# 2. AgentConfig —— 被版本登记/可回滚的「可改进配置」（YAML 的等价物，只开 prompt 面）
# --------------------------------------------------------------------------- #
@dataclass
class AgentConfig:
    """一个可跑 agent 的完整配置。本课可变面 = ``prompt``；其它字段只是承载（ch09~12 逐章放开）。"""

    name: str = "agent:ch08"
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
# 4. reward：Scorer（pointwise）＋ PairScorer（组内差分 = pairwise）
# --------------------------------------------------------------------------- #
class RuleScorer:
    """pointwise **确定性**打分：对 answer 逐 check 判命中（正 check 要含词，`~词` 要不含）。

    规则优先、离线可测；真模型判分见 ``JudgeScorer``。这是 RSI 里 reward 的最小实现。
    """

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


class JudgeScorer:
    """LLM pointwise：打分模型把 answer 对 rubric 打 0..1 分（分层 verifier 的一层）。

    L0 离线用脚本化 ``MockChatModel``；L1 用真模型。与 ``RuleScorer`` 可互换（同一个 ``score`` 接口）。
    """

    system = ("你是严格、一成不变的评分员。把候选回答对照 rubric 逐项判，只回一个 0 到 1 的小数"
              "（1=完全符合）。不要解释。")

    def _judge_user(self, task: Task, r: RolloutResult) -> str:
        rubric = "\n".join(f"- {c}" for c in task.checks)
        return (f"任务：{task.statement}\nrubric：\n{rubric}\n答案：\n{r.answer}\n\n分数：")

    async def score(self, task: Task, r: RolloutResult, model: ChatModel) -> float:
        res = await model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=self._judge_user(task, r)),
        ])
        return _parse_score(res.content)


@dataclass
class TaskScore:
    """单个 task 在一组 rollout 上的分数（组内多 rollout 取 mean/best/min 形成差异表现）。"""

    task_id: str
    mean: float
    best: float
    min_: float


async def evaluate(tasks, config: AgentConfig, model: ChatModel, scorer,
                   n_rollouts: int = 1) -> list[TaskScore]:
    """rollout 每个 task 各 ``n_rollouts`` 次，返回每 task 的 (mean, best, min) 分数组。"""
    out: list[TaskScore] = []
    for t in tasks:
        scores: list[float] = []
        for _ in range(n_rollouts):
            r = await rollout(config, t, model)
            scores.append(scorer.score(t, r))
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
    """对同 task 的 baseline vs candidate **组**做差分 → 差异表现（reward=pairwise）。
    ``n_rollouts>1`` 时候选组取均值，差分更稳——对应"组内多个 rollout 形成差异表现"。"""

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def diff(self, base: list[TaskScore], cand: list[TaskScore]) -> Differential:
        per = [(b.task_id, c.mean - b.mean) for b, c in zip(base, cand)]
        delta = _mean([d for _, d in per])
        pref = 1 if delta > self.eps else (-1 if delta < -self.eps else 0)
        return Differential(delta, pref, per)


# --------------------------------------------------------------------------- #
# 5. credit assignment：Locator — 归因到"哪一环 / 哪个 prompt 组件"错
# --------------------------------------------------------------------------- #
@dataclass
class Diagnosis:
    component: str        # guidance | frame | format | skip | halluc | ok
    finding: str
    task_id: str = ""
    score: float = 0.0


class Locator:
    """规则式 credit assignment：从总分低 + trace 反推问题出在 prompt 哪个组件。
    只归因到**组件**（而非模糊的"表现不佳"），propose 才有的放矢。ch09 可换模型驱动的分析器。"""

    def __init__(self, scorer: RuleScorer | None = None) -> None:
        self.scorer = scorer or RuleScorer()

    def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        if not r.answer.strip():
            return Diagnosis("skip", "空答：模型没有给出任何回答。", task.id, score)
        if not r.finished:
            return Diagnosis("skip", "步数耗尽：没在预算内给出最终结论。", task.id, score)
        if score >= 1.0:
            return Diagnosis("ok", "无错。", task.id, score)
        # 禁词（~）违约优先级更高：输出了"不该出现的内容"是硬伤，先抓格式问题。
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
            return Diagnosis("frame", f"缺要点 {c!r}：任务包装/示范不足，没按要的点数作答。归因到 task_frame。",
                             task.id, score)
        return Diagnosis("guidance", "分数低但规则命中，留给模型驱动分析（ch09）。", task.id, score)


# --------------------------------------------------------------------------- #
# 6. propose：确定性 prompt 编辑（ch08 规则优先；ch09 换模型驱动 + 吃更多上下文）
# --------------------------------------------------------------------------- #
class Proposer:
    """依 ``Diagnosis.component`` 编辑 ``PromptPack`` 对应字段，产出**候选 AgentConfig**。
    确定性（不依赖真模型）、离线可测；接口预留 ``context``（ch09 可把 locate+propose 合并/给更多上下文）。"""

    _RULE: ClassVar[dict[str, str]] = {
        "format":  "严格按输出要求作答；有一个要点没覆盖就算不通过，不要夹带题干里禁止的内容。",
        "frame":   "作答要逐点覆盖任务的所有要求，先给结论再分点，确保每个要点都出现。",
        "halluc":  "不要凭印象下结论：必须先调用工具拿到事实证据，再给出最终答案。",
        "skip":    "若任务未完成或不确定，必须明确说明卡在哪里，并给出可继续的查证路径。",
    }

    def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                diag: Diagnosis, context: str = "") -> AgentConfig:
        if diag.component in ("ok", "guidance") or diag.component not in self._RULE:
            return config                      # 无有效归因 → 不改（候选 = 基线）
        cand = copy.deepcopy(config)           # 不可变：propose 返回新 config，不污染当期
        if diag.component == "format":
            cand.prompt.final_format = _append_line(cand.prompt.final_format, self._RULE["format"])
        else:
            # guidance / frame / halluc / skip 都收敛到补一条 guidance 规则（未开放 frame 编辑前）
            cand.prompt.guidance = _append_line(cand.prompt.guidance, self._RULE[diag.component])
        return cand


def _append_line(existing: str, line: str) -> str:
    return f"{existing}\n{line}".strip("\n")


# --------------------------------------------------------------------------- #
# 7. gate + 版本登记
# --------------------------------------------------------------------------- #
class Gate:
    """门禁：候选相对基线分数提升且不回归才 ``promote``；否则 ``reject``（防劣化）。"""

    def __init__(self, min_delta: float = 0.0, regress_tol: float = 0.0) -> None:
        self.min_delta = min_delta
        self.regress_tol = regress_tol

    def decide(self, base: list[TaskScore], cand: list[TaskScore]) -> tuple[str, Differential]:
        diff = PairScorer().diff(base, cand)
        worst = min((d for _, d in diff.per_task), default=0.0)   # 最差单 task 差分（不回归约束）
        if diff.delta > self.min_delta and worst >= -self.regress_tol:
            return "promote", diff
        return "reject", diff


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
# 8. Rsi —— 三阶段外环（这个类镜像 product 的 Rsi 面：propose / verify / gate）
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
class RsiReport:
    baseline_score: float
    baseline_version: str
    rounds: list[RoundOutcome]
    final_version: str
    final_config: AgentConfig


class Rsi:
    """无训练 RL 式自改进外环：reward → locate(credit assignment) → propose → gate/版本。

    - ``scorer`` 负责 pointwise reward（Rule 规则 / Judge 打分模型）。
    - ``n_rollouts`` 候选组内 rollout 数：=1 → pointwise（单 rollout 打分）；>1 → 组内多 rollout 取均值
      （差分更稳 = pairwise/差异表现）。gate 用 ``PairScorer`` 对 baseline vs candidate 组做差分判 promote。
    """

    def __init__(self, config: AgentConfig, model: ChatModel, tasks,
                 *, scorer=None, locator: Locator | None = None,
                 proposer: Proposer | None = None, gate: Gate | None = None,
                 storage_dir: str | Path | None = None, n_rollouts: int = 1) -> None:
        self.config = config
        self.model = model
        self.tasks = list(tasks)
        self.scorer = scorer or RuleScorer()
        self.locator = locator or Locator()
        self.proposer = proposer or Proposer()
        self.gate = gate or Gate()
        self.registry = VersionRegistry(storage_dir) if storage_dir else None
        self.n_rollouts = n_rollouts

    def _register(self, config: AgentConfig, scores, verdict: str, parent=None) -> str | None:
        if self.registry is None:
            return None
        return self.registry.register(config, score=aggregate(scores), verdict=verdict, parent=parent)

    async def evaluate(self, config: AgentConfig) -> list[TaskScore]:
        return await evaluate(self.tasks, config, self.model, self.scorer)

    async def propose_improvement(self, current: AgentConfig, cur_scores) -> AgentConfig:
        """定位**当前**全体最差的 task → credit-assign 到组件 → 提出对该组件的 prompt 编辑。
        基于当前 config（而非历史基线）提议，避免把已提升的改动又改回去。"""
        worst = min(cur_scores, key=lambda s: s.mean)
        task = next(t for t in self.tasks if t.id == worst.task_id)
        r = await rollout(current, task, self.model)
        diag = self.locator.locate(task, r, worst.mean)
        return self.proposer.propose(current, task, r, diag)

    async def improve(self, max_rounds: int = 2) -> RsiReport:
        base = await self.evaluate(self.config)
        base_version = self._register(self.config, base, "accepted")   # 基线先登记
        baseline_score = aggregate(base)

        current = self.config
        cur_scores = base
        rounds: list[RoundOutcome] = []
        for i in range(max_rounds):
            cand = await self.propose_improvement(current, cur_scores)
            if cand.hash() == current.hash():
                rounds.append(RoundOutcome(i + 1, "noop", current.hash(), cand, 0.0, cur_scores))
                break                            # 无可改进点 → 停（避免空转/回退）
            cand_scores = await evaluate(self.tasks, cand, self.model, self.scorer, self.n_rollouts)
            verdict, diff = self.gate.decide(cur_scores, cand_scores)
            version = self._register(cand, cand_scores, verdict, parent=current.hash())
            rounds.append(RoundOutcome(i + 1, verdict, version, cand, diff.delta, cand_scores))
            if verdict == "promote":
                current = cand
                cur_scores = cand_scores          # 采纳后基线移动，便于多轮持续改进
        return RsiReport(baseline_score, base_version, rounds, current.hash(), current)


__all__ = [
    "AgentConfig",
    "Diagnosis",
    "Differential",
    "Gate",
    "JudgeScorer",
    "Locator",
    "PairScorer",
    "PromptPack",
    "Proposer",
    "RoundOutcome",
    "Rsi",
    "RsiReport",
    "RuleScorer",
    "Task",
    "TaskFrame",
    "TaskScore",
    "VersionRegistry",
    "_parse_score",
    "aggregate",
    "evaluate",
    "rollout",
]