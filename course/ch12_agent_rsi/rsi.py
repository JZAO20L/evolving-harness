"""course/ch12 · agent-rsi —— 把 agent 结构当成搜索空间（loop 参数 / 工具集 / subagent）。

``rsi.py`` = ch11 快照 + 本课增量；``agent.py`` = ch11 原样（冻结，被改进对象）。

ch09–ch11 依次打开 prompt、skills、memory 三个可变面，但**结构**仍然写死：``max_steps=8``、
所有工具永远注册、subagent 永远关闭。真实 harness 的结构搜索要回答两类问题：

1. **搜索什么**：``max_steps`` / 启用哪些工具 / 开不开 subagent / 子代理深度；
2. **怎么护栏**：结构不能无限膨胀（步数、工具数、深度、每次搜索的评测预算）。

本课把结构抽成 ``AgentConfig`` 上的离散字段，并把 act/react 换成：

```
act   = rollout(current config, train) → 分数 + trace（step_limit / tools_enabled / subagent）
react = 定位「缺步数 / 缺工具 / 缺委派」 → 在 StructureGuard 内做一次离散结构编辑
```

``StructureSearchRsi`` 额外把「一次提议」升级成「小规模 beam」：对若干邻居在 train 子集上做低成本
probe，再挑最优候选交给原门禁。**搜索宽度与 probe 任务数就是显式的评测开销预算**。

关键增量：
- ``StructureGuard``：步数 2–12、工具数上限、subagent 深度 ≤2；越界一律夹回；
- ``RolloutResult`` 记录 ``step_limit / tools_available / tools_enabled / subagent_*``，结构失败可归因；
- ``RuleStructureLocator`` / ``RuleStructureProposer`` / ``StructureLLMImprover``；
- ``StructureSearchRsi``：邻居生成 + probe + 成本上限 + 去重；最终仍走 ch09 的两级门禁。

设计取向不变：规则后端离线默认，模型后端显式接入；改写永远返回新 config；test 只碰一次。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import ClassVar

import httpx
from agent import (
    RETRYABLE_STATUS,
    AgentLoop,
    ChatModel,
    ContextBudget,
    Memory,
    MemoryStore,
    Message,
    ModelError,
    Skill,
    SkillLibrary,
    Usage,
    UserProfile,
    WorkingMemory,
    _context_has,
    _extract_json,
    _meaningful_terms,
    abbreviate_observation,
    compact_messages,
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


def _load_tasks(path: Path) -> list[Task]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks.append(Task(id=row["id"], statement=row["statement"], checks=tuple(row["checks"])))
    return tasks


def load_split(name: str) -> list[Task]:
    """读仓库根 ``dataset/rsi/{name}.jsonl``（与 ``course/`` 平级、跨章共享的探针集）。"""
    if name not in {"train", "val", "test"}:
        raise ValueError(f"未知 split: {name!r}（只支持 train/val/test）")
    return _load_tasks(Path(__file__).resolve().parents[2] / "dataset" / "rsi" / f"{name}.jsonl")


def load_synth(domain: str, split: str) -> list[Task]:
    """读仓库根 ``dataset/synth/{domain}/{split}.jsonl``（ch10 起用合成数据，避开探针集饱和）。

    ``domain`` ∈ ``math`` / ``tooluse``；``split`` ∈ ``train`` / ``val`` / ``test``。行 schema 与
    ``dataset/rsi`` 相同，直接映射成 ``Task``。
    """
    if domain not in {"math", "tooluse"}:
        raise ValueError(f"未知 domain: {domain!r}（只支持 math/tooluse）")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"未知 split: {split!r}（只支持 train/val/test）")
    path = Path(__file__).resolve().parents[2] / "dataset" / "synth" / domain / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少合成数据 {path}；先跑 dataset/synth_pipeline/synth.py 生成 dataset/synth/")
    return _load_tasks(path)


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
def _skill_canonical(skill: Skill) -> dict:
    """技能的**完整**序列化：名字不够，指令才是被演化的内容。"""
    return {"name": skill.name, "description": skill.description,
            "instructions": skill.instructions, "priority": skill.priority}


def _skill_from_canonical(data: dict) -> Skill:
    return Skill(name=str(data["name"]), description=str(data.get("description", "")),
                 instructions=str(data.get("instructions", "")),
                 priority=int(data.get("priority", 0) or 0))


@dataclass(frozen=True)
class MemoryPolicy:
    """记忆策略：压多少（budget/截断）、留多久（keep_recent/落盘）、读什么（召回/画像）。

    全部是**可调参数**而不是代码分支：改它 → AgentConfig 哈希变 → 外环把它当成一个新候选。
    """

    max_len: int = 6000          # 工作记忆 token 预算
    threshold: float = 0.8       # 超过 max_len*threshold 触发压缩
    keep_recent: int = 4         # 压缩时原样保留的最近消息数
    abbreviate_chars: int = 500  # 工具观察规则简写的最大字符数
    store_full_turns: bool = True    # 是否把完整轮次落盘（MemoryStore）
    recall_enabled: bool = True      # 是否从完整记忆自动回填
    recall_top_k: int = 8            # 回填时最多取多少条命中
    profile_enabled: bool = True     # 是否维护/注入用户画像

    def budget(self) -> ContextBudget:
        return ContextBudget(max_len=self.max_len, threshold=self.threshold,
                             keep_recent=self.keep_recent)

    def canonical(self) -> dict:
        return asdict(self)

    @classmethod
    def from_canonical(cls, data: dict | None) -> MemoryPolicy:
        data = data or {}
        default = cls()
        return _clamp_memory(cls(
            max_len=int(data.get("max_len", default.max_len) or default.max_len),
            threshold=float(data.get("threshold", default.threshold) or default.threshold),
            keep_recent=int(data.get("keep_recent", default.keep_recent) or default.keep_recent),
            abbreviate_chars=int(data.get("abbreviate_chars", default.abbreviate_chars)
                                 or default.abbreviate_chars),
            store_full_turns=bool(data.get("store_full_turns", default.store_full_turns)),
            recall_enabled=bool(data.get("recall_enabled", default.recall_enabled)),
            recall_top_k=int(data.get("recall_top_k", default.recall_top_k)
                             or default.recall_top_k),
            profile_enabled=bool(data.get("profile_enabled", default.profile_enabled)),
        ))


def _clamp_memory(policy: MemoryPolicy) -> MemoryPolicy:
    """把策略夹到安全区间：模型/规则提议再离谱也不能把预算调成 0 或召回成灾。"""
    return MemoryPolicy(
        max_len=max(200, min(100_000, int(policy.max_len))),
        threshold=max(0.1, min(1.0, float(policy.threshold))),
        keep_recent=max(1, min(50, int(policy.keep_recent))),
        abbreviate_chars=max(20, min(20_000, int(policy.abbreviate_chars))),
        store_full_turns=bool(policy.store_full_turns),
        recall_enabled=bool(policy.recall_enabled),
        recall_top_k=max(1, min(50, int(policy.recall_top_k))),
        profile_enabled=bool(policy.profile_enabled),
    )


@dataclass
class AgentConfig:
    """一个可跑 agent 的完整配置。本课可变面 = prompt + skills + memory + structure。"""

    name: str = "agent:ch12"
    prompt: PromptPack = field(default_factory=PromptPack)
    max_steps: int = 8
    skills: tuple = ()            # tuple[Skill]
    base_tools: tuple = ()        # tuple[Tool] —— 可用工具池
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    # ch12 · 结构：空 tool_names = 启用全部；子代理深度由护栏限制。
    tool_names: tuple[str, ...] = ()
    subagent_enabled: bool = False
    max_subagent_depth: int = 2

    def enabled_tools(self) -> tuple:
        if not self.tool_names:
            return tuple(self.base_tools)
        wanted = set(self.tool_names)
        return tuple(t for t in self.base_tools if t.name in wanted)

    def canonical(self) -> dict:
        return {"name": self.name,
                "prompt": self.prompt.canonical(),
                "max_steps": self.max_steps,
                "skills": [_skill_canonical(s) for s in self.skills],
                "base_tools": [t.name for t in self.base_tools],
                "memory": self.memory.canonical(),
                "tool_names": list(self.tool_names),
                "subagent_enabled": self.subagent_enabled,
                "max_subagent_depth": self.max_subagent_depth}

    def hash(self) -> str:
        return _sha256(json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False))


def update_memory_policy(config: AgentConfig, **changes) -> AgentConfig:
    """不可变地换一份 MemoryPolicy；非法字段/无变化 → 原 config（哈希不变 = noop）。"""
    allowed = set(MemoryPolicy.__dataclass_fields__)
    clean = {k: v for k, v in changes.items() if k in allowed}
    if not clean:
        return config
    policy = _clamp_memory(replace(config.memory, **clean))
    if policy == config.memory:
        return config
    cand = copy.deepcopy(config)
    cand.memory = policy
    return cand


# --------------------------------------------------------------------------- #
# 2.2 ch12 · StructureGuard：结构搜索的硬护栏
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StructureGuard:
    """结构搜索的安全边界：步数、工具数、深度、是否允许 subagent。"""

    min_steps: int = 1
    max_steps: int = 12
    max_tools: int = 8
    allow_subagent: bool = True
    max_depth: int = 2

    def clamp(self, config: AgentConfig) -> AgentConfig:
        """把结构字段夹回边界；未知工具名丢弃；无变化返回原对象。"""
        pool = tuple(t.name for t in config.base_tools)
        known = set(pool)
        names = tuple(n for n in config.tool_names if n in known)
        if config.tool_names and not names:
            names = ()                         # 全被过滤掉 → 退回"启用全部"
        if not names and len(pool) > self.max_tools:
            names = pool[:self.max_tools]      # "全部"也要受工具数上限约束
        if len(names) > self.max_tools:
            names = names[:self.max_tools]
        steps = max(self.min_steps, min(self.max_steps, int(config.max_steps)))
        depth = max(0, min(self.max_depth, int(config.max_subagent_depth)))
        subagent = bool(config.subagent_enabled and self.allow_subagent)
        if (steps, names, subagent, depth) == (
                config.max_steps, config.tool_names, config.subagent_enabled,
                config.max_subagent_depth):
            return config
        cand = copy.deepcopy(config)
        cand.max_steps, cand.tool_names = steps, names
        cand.subagent_enabled, cand.max_subagent_depth = subagent, depth
        return cand


def update_structure(config: AgentConfig, *, guard: StructureGuard | None = None,
                     **changes) -> AgentConfig:
    """不可变地改结构字段并过护栏；非法字段/无变化 → 原 config（哈希不变 = noop）。"""
    allowed = {"max_steps", "tool_names", "subagent_enabled", "max_subagent_depth"}
    clean = {k: v for k, v in changes.items() if k in allowed}
    if not clean:
        return config
    if "tool_names" in clean:
        clean["tool_names"] = tuple(clean["tool_names"] or ())
    cand = copy.deepcopy(config)
    for key, value in clean.items():
        setattr(cand, key, value)
    cand = (guard or StructureGuard()).clamp(cand)
    return config if cand.hash() == config.hash() else cand


# --------------------------------------------------------------------------- #
# 2.5 ch11 · PolicyMemory：用策略参数替换 agent.py 里写死的记忆行为
# --------------------------------------------------------------------------- #
class PolicyMemoryStore(MemoryStore):
    """在冻结的 ``MemoryStore`` 上加一个 ``top_k``：召回最多取几条命中。"""

    def __init__(self, storage_dir: str | Path | None = None, top_k: int = 8) -> None:
        super().__init__(storage_dir)
        self.top_k = top_k

    def recall(self, context: list[Message]) -> str | None:
        idx = next((i for i, m in enumerate(context) if m.role == "user"), None)
        if idx is None or not self.path.exists():
            return None
        knowledge = context[:idx]
        query = context[idx].content
        candidates = [t for t in _meaningful_terms(query) if not _context_has(knowledge, t)]
        if not candidates:
            return None
        hits: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            content = str(record.get("content", ""))
            low = content.lower()
            if any(t.lower() in low for t in candidates):
                hits.append(content or line)
        if not hits:
            return None
        snippet = " | ".join(hits[:self.top_k])
        return f"[从完整记忆检索到] {snippet}"


class PolicyWorkingMemory(WorkingMemory):
    """工具观察的简写阈值可调，并记录本次 run 是否真的发生过截断。"""

    def __init__(self, budget: ContextBudget, max_chars: int = 500) -> None:
        super().__init__(budget)
        self.max_chars = max_chars
        self.truncated = False

    def ingress(self, msg: Message) -> Message:
        if msg.role == "tool" and msg.content:
            compact = " ".join(msg.content.split())
            if len(compact) > self.max_chars:
                self.truncated = True
            return Message(role="tool", tool_call_id=msg.tool_call_id,
                           content=abbreviate_observation(msg.content, self.max_chars))
        return msg


class PolicyMemory(Memory):
    """记忆聚合的可调版：开关画像/落盘/召回，压缩预算与截断阈值来自 ``MemoryPolicy``。"""

    def __init__(self, model: ChatModel, policy: MemoryPolicy,
                 storage_dir: str | Path | None = None) -> None:
        super().__init__(model, budget=policy.budget(), storage_dir=storage_dir)
        self.policy = policy
        self.store = PolicyMemoryStore(storage_dir, top_k=policy.recall_top_k)
        self.profile = UserProfile(model, path=self.store.dir / "profile.json")
        self.wm = PolicyWorkingMemory(self.budget, max_chars=policy.abbreviate_chars)
        self.recalled = False
        self.recall_attempted = False
        self.compacted = False
        self.profile_updated = False

    def overhead(self) -> str:
        return super().overhead() if self.policy.profile_enabled else ""

    async def hook_pre_step(self, context: list[Message]) -> bool:
        self.wm.resync(context)
        if self.policy.recall_enabled and not self.recall_attempted:
            recalled = self.store.recall(context)
            if recalled and not any("[从完整记忆检索到]" in m.content for m in context):
                context.insert(0, Message(role="system", content=recalled))
                self.wm.resync(context)
                self.recalled = True
            self.recall_attempted = True
        if self.wm.is_over():
            new_context, _ = await compact_messages(self.model, context, self.budget)
            context[:] = new_context
            self.wm.resync(context)
            self.compacted = True
        return self.compacted

    async def finalize(self, context: list[Message]) -> tuple[bool, bool]:
        if self.policy.store_full_turns:
            self.store.append_turn(context)
        updated = (await self.profile.maybe_update(context)
                   if self.policy.profile_enabled else False)
        self.profile_updated = updated
        return updated, self.compacted


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
    loaded_skills: tuple[str, ...] = ()  # 实际 load 成功、且属于当前 config 的技能名（ch10）
    score: float = 0.0                 # verifier 回填的 pointwise 分
    abbreviated: bool = False          # 工具观察被规则截断（记忆"压多少"伤到事实）
    compacted: bool = False            # 工作记忆触发过模型压缩
    recalled: bool = False             # 从完整记忆回填过内容
    profile_updated: bool = False      # 本轮用户画像被更新
    # ch12 · 结构 trace：失败时区分"步数不够 / 工具没开 / 没开委派"
    step_limit: int = 0                # 本次运行的 max_steps
    tools_available: tuple[str, ...] = ()   # 工具池（base_tools）全量
    tools_enabled: tuple[str, ...] = ()     # 本次实际注册给模型的工具
    subagent_enabled: bool = False
    subagent_calls: int = 0             # 模型实际发起的 subagent 调用次数


async def rollout(config: AgentConfig, task: Task, model: ChatModel,
                  *, memory_dir: str | Path | None = None) -> RolloutResult:
    """用 ``config`` 构造一个 AgentLoop，跑 ``task.statement``（经 prompt 包装）。

    镜像产品「同一循环、不同 YAML」：改 prompt / skills / memory → 输入变 → 行为变；**不动 agent.py**。
    """
    skills = SkillLibrary(*config.skills) if config.skills else None
    system = config.prompt.render_system()
    if skills is not None:
        # AgentLoop 在显式 system 下不自动拼目录；这里由 RSI 侧组合输入（不动 agent.py）。
        system += (f"\n\n可用技能（渐进披露：先看目录，需要时用 load_skill 加载完整指令）：\n"
                   f"{skills.list_summary()}")
    memory = PolicyMemory(model, config.memory, storage_dir=memory_dir)
    tools = config.enabled_tools()
    loop = AgentLoop(model, system=system, memory=memory,
                     skills=skills, base_tools=list(tools),
                     max_steps=config.max_steps,
                     subagent_enabled=config.subagent_enabled)
    loop.max_subagent_depth = config.max_subagent_depth
    res = await loop.run(config.prompt.render_user(task.statement))
    known = {s.name for s in config.skills}
    loaded = tuple(dict.fromkeys(
        tc.arguments.get("name", "")
        for tc in res.tool_calls
        if tc.name == "load_skill" and tc.arguments.get("name", "") in known
    ))
    return RolloutResult(task_id=task.id, answer=res.answer, finished=res.finished,
                         steps=res.steps, usage=res.usage,
                         tool_calls=tuple(tc.name for tc in res.tool_calls),
                         loaded_skills=loaded,
                         abbreviated=memory.wm.truncated,
                         compacted=memory.compacted,
                         recalled=memory.recalled,
                         profile_updated=memory.profile_updated,
                         step_limit=config.max_steps,
                         tools_available=tuple(t.name for t in config.base_tools),
                         tools_enabled=tuple(t.name for t in tools),
                         subagent_enabled=config.subagent_enabled,
                         subagent_calls=sum(1 for tc in res.tool_calls
                                            if tc.name == "subagent"))


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
                   n_rollouts: int = 1, *, memory_dir: str | Path | None = None) -> list[TaskScore]:
    """rollout 每个 task 各 ``n_rollouts`` 次，**await** ``judge.score`` 后返回 (mean, best, min)。

    ``model`` 只用于 rollout；打分依赖由 ``judge`` 自己持有（规则 / LLM / System One 可换）。
    ``memory_dir`` 让同一次 RSI run 的 rollout 共享完整记忆（否则每次 run 各用临时目录）。
    """
    judge = judge or RuleJudge()
    out: list[TaskScore] = []
    for t in tasks:
        scores: list[float] = []
        for _ in range(n_rollouts):
            r = await rollout(config, t, model, memory_dir=memory_dir)
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
    component: str        # guidance | frame | format | skip | halluc | skill_missing | skill_misleading | ok
    finding: str
    task_id: str = ""
    score: float = 0.0
    skill: str = ""       # 归因到的具体技能名（ch10；空 = 没有具体目标）


COMPONENTS = ("format", "frame", "halluc", "skip",
              "skill_missing", "skill_misleading",
              "memory_miss", "memory_overflow", "memory_pollution",
              "step_limit", "tool_missing", "subagent_missing",
              "guidance", "ok")

SYS_ONE_COMPONENT_CRITERIA: dict[str, str] = {
    "format": "输出了 rubric 禁止的内容，或没按要求的输出形态作答",
    "frame": "任务包装/示范不足，漏掉了 rubric 要求的要点",
    "halluc": "没查工具/事实就下结论，导致缺要点",
    "skip": "空答，或步数耗尽没给出最终结论",
    "skill_missing": "没有可用/没有加载合适的技能，导致漏掉要点；需要新增技能",
    "skill_misleading": "加载了技能但指令误导了模型，导致漏点或输出禁词；需要改写/删除该技能",
    "memory_miss": "记忆截断/压缩丢了关键事实，导致漏点；需要放宽保留或截断阈值",
    "memory_overflow": "上下文预算过紧/压缩时机不当，导致没能在步数内收尾；需要更早压缩或压缩更狠",
    "memory_pollution": "召回/画像带回了无关或过时内容，干扰了回答；需要收紧召回或关闭画像",
    "step_limit": "max_steps 太小，模型没能在预算内完成多步任务；需要提高步数上限",
    "tool_missing": "完成该任务需要的工具没有启用，模型查不到事实；需要放开工具集",
    "subagent_missing": "任务需要隔离/委派，但 subagent 未启用；需要打开子代理",
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
    """chat LLM 归因：喂 task + rubric + trace + 分数，让模型选组件（可落到具体 skill）。"""

    system = ("你是 Agent 失败归因分析员。读任务、私有 rubric、候选回答、工具调用、已加载技能、记忆"
              "trace 与结构 trace，判断问题主要出在哪个组件。只回 JSON：{\"component\": 组件名, "
              "\"finding\": 一句话依据, \"skill\": 目标技能名（没有则空串）}。组件名只能是 "
              "format/frame/halluc/skip/skill_missing/skill_misleading/"
              "memory_miss/memory_overflow/memory_pollution/"
              "step_limit/tool_missing/subagent_missing/guidance/ok。证据不足时回 ok。")

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
                f"已加载技能：{', '.join(r.loaded_skills) or '无'}\n"
                f"记忆 trace：截断={r.abbreviated} 压缩={r.compacted} "
                f"召回={r.recalled} 画像更新={r.profile_updated}\n"
                f"结构 trace：步数={r.steps}/{r.step_limit} "
                f"工具池={', '.join(r.tools_available) or '无'} "
                f"已启用={', '.join(r.tools_enabled) or '无'} "
                f"subagent={r.subagent_enabled}（调用 {r.subagent_calls} 次）\n"
                f"分数：{score:.2f}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict) or data.get("component") not in COMPONENTS:
            return Diagnosis("ok", "定位解析失败，保守不改。", task.id, score)
        return Diagnosis(str(data["component"]), str(data.get("finding") or ""), task.id, score,
                         str(data.get("skill") or ""))


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
# 6.4 改进接口（提前定义：skill 粒度的实现紧随其后使用）
# --------------------------------------------------------------------------- #
class Proposer:
    """统一改进接口：``async propose(config, task, r, diag) -> 新 AgentConfig``（不可变）。"""

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        raise NotImplementedError


class Improver:
    """统一改进器接口：``async improve(config, task, r) -> (Diagnosis, AgentConfig)``。"""

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 6.5 ch10 · skill 粒度的定位与编辑
# --------------------------------------------------------------------------- #
def _missing_checks(task: Task, r: RolloutResult, scorer: RuleScorer) -> list[str]:
    """漏掉的**正向** rubric 要点（`~word` 禁词不算）。"""
    return [c for c in task.checks
            if not c.startswith("~") and not scorer.check_hit(c, r.answer)]


def _violated_forbidden(task: Task, r: RolloutResult, scorer: RuleScorer) -> list[str]:
    """命中（即违规）的禁词 check。"""
    return [c for c in task.checks
            if c.startswith("~") and not scorer.check_hit(c, r.answer)]


class RuleSkillLocator(Locator):
    """规则式 skill 归因：把「漏点/禁词」落到「技能缺失」或「技能误导」。

    - 一个技能都没加载且漏正向要点 → ``skill_missing``（该有技能却没有）；
    - 加载了技能仍漏点 / 输出禁词 → ``skill_misleading``（技能指令没帮上忙甚至带偏）；
    - 空答/未收尾 → ``skip``；满分 → ``ok``。目标技能从 trace 的 ``loaded_skills`` 里取。
    """

    def __init__(self, scorer: RuleScorer | None = None) -> None:
        self.scorer = scorer or RuleScorer()

    def _target(self, r: RolloutResult) -> str:
        return r.loaded_skills[0] if r.loaded_skills else ""

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        if not r.answer.strip():
            return Diagnosis("skip", "空答：模型没有给出任何回答。", task.id, score)
        if not r.finished:
            return Diagnosis("skip", "步数耗尽：没在预算内给出最终结论。", task.id, score)
        if score >= 1.0:
            return Diagnosis("ok", "无错。", task.id, score)

        violated = _violated_forbidden(task, r, self.scorer)
        missing = _missing_checks(task, r, self.scorer)
        if violated:
            word = violated[0][1:]
            if r.loaded_skills:
                target = self._target(r)
                return Diagnosis(
                    "skill_misleading",
                    f"输出禁词 {word!r}，且加载了技能 {target!r}：技能指令未约束输出形态。",
                    task.id, score, target)
            return Diagnosis("format", f"输出了禁止内容 {word!r}，且没有加载技能。", task.id, score)

        if missing:
            detail = "、".join(repr(c) for c in missing)
            if not r.loaded_skills:
                return Diagnosis(
                    "skill_missing",
                    f"漏要点 {detail}，且全程没有加载任何技能：缺少覆盖该要点的技能。",
                    task.id, score)
            target = self._target(r)
            return Diagnosis(
                "skill_misleading",
                f"加载了技能 {target!r} 仍漏要点 {detail}：技能指令没有覆盖这些要求。",
                task.id, score, target)

        return Diagnosis("guidance", "分数低但规则命中，留给模型驱动分析。", task.id, score)


@dataclass(frozen=True)
class SkillEdit:
    """对 skills 集的一次不可变编辑：add（同名则替换）/ delete / rewrite。"""

    action: str                   # add | delete | rewrite
    name: str
    description: str = ""
    instructions: str = ""
    priority: int = 0


def apply_skill_edit(config: AgentConfig, edit: SkillEdit) -> AgentConfig:
    """把 ``SkillEdit`` 应用到 **副本** 上；非法编辑/无实际变化 → 原 config（哈希不变 = noop）。"""
    action, name = edit.action, edit.name.strip()
    if action not in {"add", "delete", "rewrite"} or not name:
        return config
    skills = list(config.skills)
    index = next((i for i, s in enumerate(skills) if s.name == name), None)

    if action == "delete":
        if index is None:
            return config
        del skills[index]
    elif action == "add":
        description = edit.description.strip()
        instructions = edit.instructions.strip()
        if not description or not instructions:
            return config
        skill = Skill(name=name, description=description,
                      instructions=instructions, priority=edit.priority)
        if index is None:
            skills.append(skill)
        else:
            skills[index] = skill
    else:  # rewrite
        if index is None:
            return config
        old = skills[index]
        instructions = edit.instructions.strip() or old.instructions
        description = edit.description.strip() or old.description
        priority = edit.priority if edit.priority else old.priority
        if (instructions, description, priority) == (old.instructions, old.description, old.priority):
            return config
        skills[index] = Skill(name=name, description=description,
                              instructions=instructions, priority=priority)

    cand = copy.deepcopy(config)
    cand.skills = tuple(skills)
    return cand


def _coverage_instructions(missing: list[str], violated: list[str]) -> str:
    lines = ["执行本技能时，严格检查最终回答："]
    if missing:
        lines.append("必须逐字包含以下要点：" + "；".join(missing))
    if violated:
        lines.append("不得出现以下内容：" + "；".join(c[1:] for c in violated))
    return "\n".join(lines)


class RuleSkillProposer(Proposer):
    """确定性 skill 编辑：``skill_missing`` → 新增覆盖技能；``skill_misleading`` → 改写目标技能。"""

    def __init__(self, scorer: RuleScorer | None = None) -> None:
        self.scorer = scorer or RuleScorer()

    def _pick_target(self, config: AgentConfig, diag: Diagnosis,
                     r: RolloutResult) -> str:
        names = {s.name for s in config.skills}
        for candidate in (diag.skill, *(r.loaded_skills or ())):
            if candidate in names:
                return candidate
        return config.skills[0].name if config.skills else ""

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        if diag.component not in {"skill_missing", "skill_misleading"}:
            return config
        missing = _missing_checks(task, r, self.scorer)
        violated = _violated_forbidden(task, r, self.scorer)
        if not missing and not violated:
            return config
        text = _coverage_instructions(missing, violated)

        if diag.component == "skill_missing":
            name = f"skill-{re.sub(r'[^a-zA-Z0-9_-]+', '-', task.id).strip('-')}"
            return apply_skill_edit(config, SkillEdit(
                "add", name, f"确保任务 {task.id} 的必需要点被逐字覆盖", text, priority=5))

        target = self._pick_target(config, diag, r)
        if not target:
            return config
        old = next((s for s in config.skills if s.name == target), None)
        if old is None or text in old.instructions:
            return config
        instructions = _append_line(old.instructions, text)
        return apply_skill_edit(config, SkillEdit(
            "rewrite", target, old.description, instructions, old.priority))


class SkillLLMImprover(Improver):
    """**一次** chat 调用完成 skill 定位 + 编辑（meta-agent 的 thought+act 合并，作用于 skills）。

    返回 ``{component, finding, skill, action, description, instructions}``；解析失败或动作非法 →
    ``ok`` + 原 config。规则兜底顺序：``skill_missing`` 无目标时要求模型给新 ``name``。
    """

    system = ("你是 Agent 的技能库改进器。读任务、私有 rubric、候选回答、加载过的技能与当前技能库，"
              "一次性完成：1）定位问题；2）对技能库做一次编辑。只回 JSON："
              "{\"component\": skill_missing/skill_misleading/format/frame/halluc/skip/guidance/ok, "
              "\"finding\": 一句话依据, \"skill\": 目标技能名（add 时为新技能名）, "
              "\"action\": add/delete/rewrite/none, "
              "\"description\": 技能目录描述, \"instructions\": 完整执行指令}。"
              "没有把握时 action=none、component=ok。")

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        catalog = "\n".join(
            f"- {s.name} (优先 {s.priority}): {s.description}\n  {s.instructions}"
            for s in config.skills) or "（空）"
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"任务：{task.statement}\nrubric：\n{_rubric(task)}\n回答：\n{r.answer}\n"
                f"工具调用：{', '.join(r.tool_calls) or '无'}\n"
                f"已加载技能：{', '.join(r.loaded_skills) or '无'}\n分数：{r.score:.2f}\n"
                f"当前技能库：\n{catalog}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict):
            return Diagnosis("ok", "技能改进解析失败，保守不改。", task.id, r.score), config
        component = str(data.get("component") or "ok")
        component = component if component in COMPONENTS else "ok"
        action = str(data.get("action") or "none")
        skill = str(data.get("skill") or "").strip()
        diag = Diagnosis(component, str(data.get("finding") or ""), task.id, r.score, skill)
        if action not in {"add", "delete", "rewrite"} or not skill:
            return diag, config
        edit = SkillEdit(action, skill, str(data.get("description") or ""),
                         str(data.get("instructions") or ""))
        return diag, apply_skill_edit(config, edit)


# --------------------------------------------------------------------------- #
# 6.6 ch11 · memory 粒度的定位与策略编辑
# --------------------------------------------------------------------------- #
class RuleMemoryLocator(Locator):
    """规则式记忆归因：用 trace 的截断/压缩/召回信号区分三种记忆失败。

    - 未收尾且压缩过 → ``memory_overflow``（预算/压缩时机太紧）；
    - 漏要点且观察被截断或压缩过 → ``memory_miss``（关键事实被压没了）；
    - 输出禁词/漏点且召回或画像动过 → ``memory_pollution``（记忆带回无关内容）。
    """

    def __init__(self, scorer: RuleScorer | None = None) -> None:
        self.scorer = scorer or RuleScorer()

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        if not r.answer.strip():
            return Diagnosis("skip", "空答：模型没有给出任何回答。", task.id, score)
        if not r.finished:
            if r.compacted:
                return Diagnosis(
                    "memory_overflow",
                    "步数耗尽且期间触发过压缩：预算/压缩时机让工作记忆越跑越紧。",
                    task.id, score)
            return Diagnosis("skip", "步数耗尽：没在预算内给出最终结论。", task.id, score)
        if score >= 1.0:
            return Diagnosis("ok", "无错。", task.id, score)

        violated = _violated_forbidden(task, r, self.scorer)
        missing = _missing_checks(task, r, self.scorer)
        if violated:
            word = violated[0][1:]
            if r.recalled or r.profile_updated:
                return Diagnosis(
                    "memory_pollution",
                    f"输出禁词 {word!r}，本轮发生过召回/画像更新：记忆带回了无关内容。",
                    task.id, score)
            return Diagnosis("format", f"输出了禁止内容 {word!r}，与记忆无关。", task.id, score)

        if missing:
            detail = "、".join(repr(c) for c in missing)
            if r.abbreviated or r.compacted:
                source = "工具观察被截断" if r.abbreviated else "工作记忆被压缩"
                return Diagnosis(
                    "memory_miss",
                    f"漏要点 {detail}，且{source}：关键事实可能在压缩/截断中丢失。",
                    task.id, score)
            if r.recalled:
                return Diagnosis(
                    "memory_pollution",
                    f"漏要点 {detail}，但发生过召回：召回内容没能补上缺口。",
                    task.id, score)
            return Diagnosis("guidance", "分数低但没有记忆信号，留给模型驱动分析。", task.id, score)

        return Diagnosis("guidance", "分数低但规则命中，留给模型驱动分析。", task.id, score)


class RuleMemoryProposer(Proposer):
    """确定性策略调整：漏 → 放宽保留/截断；溢出 → 压缩更早更狠；污染 → 收紧召回/画像。

    目标值取「阈值」而非「每轮加一点」，保证同一诊断重复出现时是幂等的。
    """

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        p = config.memory
        if diag.component == "memory_miss":
            return update_memory_policy(
                config,
                abbreviate_chars=max(2000, p.abbreviate_chars),
                keep_recent=max(6, p.keep_recent),
                threshold=min(0.9, p.threshold),
            )
        if diag.component == "memory_overflow":
            return update_memory_policy(
                config,
                max_len=min(4000, p.max_len),
                threshold=min(0.6, p.threshold),
                keep_recent=min(2, p.keep_recent),
                abbreviate_chars=min(300, p.abbreviate_chars),
            )
        if diag.component == "memory_pollution":
            if r.recalled:
                changes = {"recall_top_k": min(3, p.recall_top_k)}
                if p.recall_top_k <= 3:
                    changes["recall_enabled"] = False
                return update_memory_policy(config, **changes)
            return update_memory_policy(config, profile_enabled=False)
        return config


class MemoryLLMImprover(Improver):
    """一次 chat 调用完成记忆定位 + 策略改写；返回 JSON 的 ``policy`` 做增量合并。

    解析失败/非 dict/无可调字段 → ``ok`` + 原 config。所有值经 ``_clamp_memory`` 夹到安全区间。
    """

    system = ("你是 Agent 的记忆策略改进器。读任务、私有 rubric、回答、记忆 trace 与当前策略，"
              "一次性完成：1）定位记忆问题；2）给出策略调整。只回 JSON："
              "{\"component\": memory_miss/memory_overflow/memory_pollution/"
              "format/frame/halluc/skip/skill_missing/skill_misleading/guidance/ok, "
              "\"finding\": 一句话依据, \"policy\": {要修改的 MemoryPolicy 字段}}。"
              "policy 可含 max_len/threshold/keep_recent/abbreviate_chars/"
              "store_full_turns/recall_enabled/recall_top_k/profile_enabled。没有把握时 policy 留空。")

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"任务：{task.statement}\nrubric：\n{_rubric(task)}\n回答：\n{r.answer}\n"
                f"记忆 trace：截断={r.abbreviated} 压缩={r.compacted} "
                f"召回={r.recalled} 画像更新={r.profile_updated}\n"
                f"分数：{r.score:.2f}\n当前策略：\n"
                f"{json.dumps(config.memory.canonical(), ensure_ascii=False, sort_keys=True)}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict):
            return Diagnosis("ok", "记忆改进解析失败，保守不改。", task.id, r.score), config
        component = str(data.get("component") or "ok")
        component = component if component in COMPONENTS else "ok"
        diag = Diagnosis(component, str(data.get("finding") or ""), task.id, r.score)
        patch = data.get("policy")
        if not isinstance(patch, dict):
            return diag, config
        merged = {**config.memory.canonical(), **patch}
        policy = MemoryPolicy.from_canonical(merged)
        return diag, update_memory_policy(config, **policy.canonical())


# --------------------------------------------------------------------------- #
# 6.7 ch12 · structure 粒度的定位与结构编辑
# --------------------------------------------------------------------------- #
def _looks_delegable(task: Task) -> bool:
    """多要点 + 题面含并列词 → 值得隔离/委派（结构定位的便宜启发式）。"""
    positives = [c for c in task.checks if not c.startswith("~")]
    return len(positives) >= 2 and any(
        w in task.statement for w in ("分别", "各自", "多个", "两种", "并且", "以及"))


class RuleStructureLocator(Locator):
    """规则式结构归因：步数耗尽 / 工具没放开 / 该委派却没开 subagent。

    优先级：空答 → skip；未收尾 → ``step_limit``；禁词 → format；漏点 →
    ``tool_missing``（有禁用工具且没调过任何已启用工具）→ ``subagent_missing``
    （多要点任务、全工具已开、未开委派）→ frame/halluc。
    """

    def __init__(self, scorer: RuleScorer | None = None) -> None:
        self.scorer = scorer or RuleScorer()

    async def locate(self, task: Task, r: RolloutResult, score: float) -> Diagnosis:
        if not r.finished:
            if r.step_limit and r.steps >= r.step_limit:
                return Diagnosis(
                    "step_limit",
                    f"步数耗尽（{r.steps}/{r.step_limit}）：max_steps 不够完成该任务。",
                    task.id, score)
            return Diagnosis("skip", "未收尾且未到步数上限：循环异常。", task.id, score)
        if not r.answer.strip():
            return Diagnosis("skip", "空答：模型没有给出任何回答。", task.id, score)
        if score >= 1.0:
            return Diagnosis("ok", "无错。", task.id, score)

        violated = _violated_forbidden(task, r, self.scorer)
        if violated:
            return Diagnosis("format", f"输出了禁止内容 {violated[0][1:]!r}，与结构无关。",
                             task.id, score)

        missing = _missing_checks(task, r, self.scorer)
        if not missing:
            return Diagnosis("guidance", "分数低但规则命中，留给模型驱动分析。", task.id, score)

        disabled = tuple(n for n in r.tools_available if n not in set(r.tools_enabled))
        used_enabled = any(tc in set(r.tools_enabled) for tc in r.tool_calls)
        if disabled and not used_enabled:
            return Diagnosis(
                "tool_missing",
                f"漏要点 {missing!r}；工具池里有 {disabled!r} 未启用，模型查不到事实。",
                task.id, score)
        if not r.subagent_enabled and _looks_delegable(task) and not used_enabled:
            return Diagnosis(
                "subagent_missing",
                f"漏要点 {missing!r}；任务含多个并列要点，但 subagent 未启用。",
                task.id, score)
        if not r.tool_calls:
            return Diagnosis("halluc", f"漏要点 {missing!r} 且没调任何工具：疑似没查事实。",
                             task.id, score)
        return Diagnosis("frame", f"漏要点 {missing!r}：结构已够，问题在任务包装/示范。",
                         task.id, score)


class RuleStructureProposer(Proposer):
    """确定性结构编辑：缺步数 → 提到护栏上限；缺工具 → 全开；缺委派 → 开 subagent。

    与 ch11 一样用**目标档位**而非"每轮 +1"，重复诊断返回原 config（哈希 noop）。
    """

    def __init__(self, guard: StructureGuard | None = None) -> None:
        self.guard = guard or StructureGuard()

    async def propose(self, config: AgentConfig, task: Task, r: RolloutResult,
                      diag: Diagnosis) -> AgentConfig:
        if diag.component == "step_limit":
            target = min(self.guard.max_steps, max(config.max_steps, self.guard.max_steps))
            return update_structure(config, guard=self.guard, max_steps=target)
        if diag.component == "tool_missing":
            enabled = {t.name for t in config.enabled_tools()}
            missing = tuple(t.name for t in config.base_tools if t.name not in enabled)
            if not missing:
                return config
            return update_structure(config, guard=self.guard,
                                    tool_names=(*config.tool_names, *missing))
        if diag.component == "subagent_missing":
            return update_structure(config, guard=self.guard, subagent_enabled=True)
        return config


class StructureLLMImprover(Improver):
    """一次 chat 调用完成结构定位 + 结构 patch；解析失败/空 patch → ``ok`` + 原 config。"""

    system = ("你是 Agent 结构搜索器。读任务、私有 rubric、回答、结构 trace 与当前结构，"
              "一次性完成：1）定位结构问题；2）给出结构调整。只回 JSON："
              "{\"component\": step_limit/tool_missing/subagent_missing/"
              "format/frame/halluc/skip/skill_missing/skill_misleading/"
              "memory_miss/memory_overflow/memory_pollution/guidance/ok, "
              "\"finding\": 一句话依据, \"structure\": {要修改的结构字段}}。"
              "structure 可含 max_steps/tool_names/subagent_enabled/max_subagent_depth。"
              "tool_names 为空数组表示启用全部。没有把握时 structure 留空。")

    def __init__(self, model: ChatModel, guard: StructureGuard | None = None) -> None:
        self.model = model
        self.guard = guard or StructureGuard()

    async def improve(self, config: AgentConfig, task: Task,
                      r: RolloutResult) -> tuple[Diagnosis, AgentConfig]:
        res = await self.model.complete([
            Message(role="system", content=self.system),
            Message(role="user", content=(
                f"任务：{task.statement}\nrubric：\n{_rubric(task)}\n回答：\n{r.answer}\n"
                f"结构 trace：步数={r.steps}/{r.step_limit} "
                f"工具池={', '.join(r.tools_available) or '无'} "
                f"已启用={', '.join(r.tools_enabled) or '无'} "
                f"subagent={r.subagent_enabled}（调用 {r.subagent_calls} 次）\n"
                f"分数：{r.score:.2f}\n当前结构：\n"
                f"{json.dumps({'max_steps': config.max_steps, 'tool_names': list(config.tool_names), 'subagent_enabled': config.subagent_enabled, 'max_subagent_depth': config.max_subagent_depth}, ensure_ascii=False, sort_keys=True)}\n\nJSON：")),
        ])
        data = _extract_json(res.content)
        if not isinstance(data, dict):
            return Diagnosis("ok", "结构改进解析失败，保守不改。", task.id, r.score), config
        component = str(data.get("component") or "ok")
        component = component if component in COMPONENTS else "ok"
        diag = Diagnosis(component, str(data.get("finding") or ""), task.id, r.score)
        patch = data.get("structure")
        if not isinstance(patch, dict):
            return diag, config
        return diag, update_structure(config, guard=self.guard, **patch)


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

    def rollback(self, version: str, *, base_tools: tuple = ()) -> AgentConfig:
        """返回该版本 config 的还原：prompt / max_steps / **完整 skills**。

        ``base_tools`` 只登记了名字、无法从磁盘还原 handler，所以由调用方在回滚时重新注入；
        本章的演化面是 skills，默认空即可。旧 ch09 登记里 skills 只有名字 → 无法还原指令，跳过。
        """
        entry = self.get(version)
        if entry is None:
            raise KeyError(f"没有版本 {version}")
        cfg = entry["config"]
        skills = tuple(_skill_from_canonical(s) for s in cfg.get("skills", [])
                       if isinstance(s, dict) and s.get("name"))
        return AgentConfig(name=cfg["name"],
                           prompt=PromptPack(guidance=cfg["prompt"]["guidance"],
                                             task_frame=TaskFrame(**cfg["prompt"]["task_frame"]),
                                             final_format=cfg["prompt"]["final_format"]),
                           max_steps=cfg["max_steps"], skills=skills,
                           base_tools=base_tools,
                           memory=MemoryPolicy.from_canonical(cfg.get("memory")),
                           tool_names=tuple(cfg.get("tool_names") or ()),
                           subagent_enabled=bool(cfg.get("subagent_enabled", False)),
                           max_subagent_depth=int(cfg.get("max_subagent_depth", 2)))


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
                 max_rounds: int = 2, val_every: int = 2,
                 memory_dir: str | Path | None = None) -> None:
        self.config = config
        self.model = model
        self.train = list(train)
        self.val = list(val)
        self.test = list(test)
        self.judge = judge or RuleJudge()
        self.improver = improver or TwoStageImprover(RuleStructureLocator(),
                                                     RuleStructureProposer())
        self.gate = gate or Gate()
        self.segment_gate = segment_gate or SegmentGate()
        self.registry = VersionRegistry(storage_dir) if storage_dir else None
        if memory_dir is not None:
            self.memory_dir: Path = Path(memory_dir)
        elif storage_dir is not None:
            self.memory_dir = Path(storage_dir) / "memory"
        else:
            self.memory_dir = Path(tempfile.mkdtemp(prefix="eh_rsi_mem_"))
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
        return await evaluate(tasks, config, self.model, self.judge, self.n_rollouts,
                              memory_dir=self.memory_dir)

    async def propose_improvement(self, current: AgentConfig,
                                  cur_scores: list[TaskScore]) -> tuple[Diagnosis, AgentConfig]:
        """定位**当前 train** 上最差的 task → 回填分数 → improver 产出 (诊断, 候选 config)。"""
        worst = min(cur_scores, key=lambda s: s.mean)
        task = next(t for t in self.train if t.id == worst.task_id)
        r = await rollout(current, task, self.model, memory_dir=self.memory_dir)
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


class StructureSearchRsi(Rsi):
    """带**显式评测预算**的结构搜索：邻居生成 → train 子集 probe → 最优候选交给原门禁。

    - ``search_width``：每轮最多 probe 几个邻居（结构编辑的宽度）；
    - ``probe_tasks``：每次 probe 用 train 里最差的几个任务（低成本代理）；
    - ``max_search_evals``：可选的全局 probe 预算（rollout 次数上限）；到顶即停；
    - ``search_evals``：实际消耗的 probe rollout 数（可观测的成本）。
    """

    def __init__(self, *args, search_width: int = 3, probe_tasks: int = 3,
                 guard: StructureGuard | None = None,
                 max_search_evals: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.search_width = max(1, int(search_width))
        self.probe_tasks = max(1, int(probe_tasks))
        self.guard = guard or StructureGuard()
        self.max_search_evals = max_search_evals
        self.search_evals = 0

    def _neighbors(self, config: AgentConfig) -> list[tuple[str, AgentConfig]]:
        """离散结构邻居：加步数 / 放开工具 / 开委派；去重且排除当前 config。"""
        edits: list[tuple[str, AgentConfig]] = []
        if config.max_steps < self.guard.max_steps:
            edits.append(("step_limit",
                          update_structure(config, guard=self.guard,
                                          max_steps=min(self.guard.max_steps,
                                                        config.max_steps + 4))))
        enabled = {t.name for t in config.enabled_tools()}
        missing = tuple(t.name for t in config.base_tools if t.name not in enabled)
        if missing:
            edits.append(("tool_missing",
                          update_structure(config, guard=self.guard,
                                          tool_names=(*config.tool_names, *missing))))
        if not config.subagent_enabled and self.guard.allow_subagent:
            edits.append(("subagent_missing",
                          update_structure(config, guard=self.guard, subagent_enabled=True)))
        seen = {config.hash()}
        out: list[tuple[str, AgentConfig]] = []
        for component, cand in edits:
            h = cand.hash()
            if h in seen:
                continue
            seen.add(h)
            out.append((component, cand))
        return out[:self.search_width]

    async def propose_improvement(self, current: AgentConfig,
                                  cur_scores: list[TaskScore]) -> tuple[Diagnosis, AgentConfig]:
        """在 train 最差任务上 probe 邻居，挑 probe 分最高且成本最低的结构。"""
        neighbors = self._neighbors(current)
        worst_ids = [s.task_id for s in sorted(cur_scores, key=lambda s: s.mean)[:self.probe_tasks]]
        by_id = {t.id: t for t in self.train}
        probe = [by_id[i] for i in worst_ids if i in by_id]
        if not neighbors or not probe:
            return Diagnosis("ok", "结构已到护栏边界，无可用邻居。", "", 0.0), current

        baseline = _mean([s.mean for s in cur_scores if s.task_id in set(worst_ids)])
        best_key = None
        best: tuple[str, AgentConfig, float] | None = None
        for component, cand in neighbors:
            if (self.max_search_evals is not None
                    and self.search_evals >= self.max_search_evals):
                break
            probe_scores = await self.evaluate(cand, probe)
            self.search_evals += len(probe) * self.n_rollouts
            score = aggregate(probe_scores)
            # 同分时偏好更省的结构：步数少、工具少、不开委派。
            key = (score, -cand.max_steps, -len(cand.enabled_tools()),
                   -int(cand.subagent_enabled), -cand.max_subagent_depth)
            if best is None or key > best_key:
                best_key, best = key, (component, cand, score)

        if best is None:
            return Diagnosis("ok", "probe 预算耗尽，保留当前结构。", "", 0.0), current
        component, cand, score = best
        if score <= baseline + 1e-9:
            return Diagnosis("ok", "所有结构邻居都没能在 probe 上超过基线。", "", baseline), current
        diag = Diagnosis(component, f"结构搜索：probe 上 {baseline:.2f} → {score:.2f} "
                                    f"（{len(probe)} 个任务）选中最优邻居。", "", score)
        return diag, cand


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
    "MemoryLLMImprover",
    "MemoryPolicy",
    "MergedLLMImprover",
    "PairScorer",
    "PolicyMemory",
    "PolicyMemoryStore",
    "PolicyWorkingMemory",
    "PromptPack",
    "Proposer",
    "RoundOutcome",
    "Rsi",
    "RsiReport",
    "RuleJudge",
    "RuleLocator",
    "RuleMemoryLocator",
    "RuleMemoryProposer",
    "RuleProposer",
    "RuleScorer",
    "RuleSkillLocator",
    "RuleSkillProposer",
    "RuleStructureLocator",
    "RuleStructureProposer",
    "SegmentGate",
    "SegmentOutcome",
    "SkillEdit",
    "SkillLLMImprover",
    "StructureGuard",
    "StructureLLMImprover",
    "StructureSearchRsi",
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
    "apply_skill_edit",
    "evaluate",
    "load_split",
    "load_synth",
    "rollout",
    "update_memory_policy",
    "update_structure",
]
