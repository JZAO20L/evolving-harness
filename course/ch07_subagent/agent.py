"""course/ch07 · sub-agent —— 子代理：act:subagent，独立 context 的一次委派 run。

ch01~ch06 让 agent 能"调模型 / ReAct 循环 / 工具 / 技能渐进披露 / 分层记忆 / 规划"，但一个循环
只能说完一件事：没有"把一个子问题**委派给一个独立循环**去解决、父循环等它回来再继续"的能力。
本课补上 **sub-agent**（对齐 course/README ch07 的"``act:subagent`` 委派子 run（独立 context）、
等待/深度上限"与产线 DESIGN §7.4"子任务用独立 messages 列表跑同一个 Agent 类的新实例、只把
最终文本作为一条 tool 结果返回父循环、深度 ≤ 2、子任务步数 ≤ 父任务剩余预算的一半"）。

设计取向（已与用户确认）：
- **子代理对主 Agent 的视角 = 一个有 Agent 能力的工具**：模型在循环里调 ``subagent(task)``；
- **子代理的返回按工具结果处理**：父循环只把子**最终文本**作为一条 tool observation 追加（经既有
  ``_emit``/``wm.ingress`` -> ``abbreviate_observation`` 截断，只保留核心内容）；
- **一切在一个 ``agent.py``**：不拆 subagent.py，最小侵入地挂到 ch06 的 ``AgentLoop`` 之上。

实现要点（``AgentLoop`` 追加，向后兼容）：
- 新增带默认参数：``base_tools``（工作工具，父/子共享同一工具能力）、``subagent_enabled``；
  及 ``depth``（根=0，每委派一层 +1）、``max_subagent_depth=2``、``steps_used``（预算测算）。
- ``_register_subagent`` 注册 ``subagent(task)`` 工具；``_run_subagent`` 依次：深度护栏（错误当数据）
  -> 子预算 = 父剩余 // 2 -> 新建 ``AgentLoop`` 实例（独立 context / 注册表 / 工作记忆；同一 model/
  skills/system/base_tools/存储目录）-> 顺序 await 直到子收官 -> 只回最终文本。子预算耗尽同样当数据。

贯彻原则不变：规则优先、模型低频；循环内顺序 async（intra-turn 不并行，并发 run 在边界注入）；
重试/异常分档归适配器；工具/技能/记忆/路径安全全部从 ch06 继承。
涉及网络的适配器是 ``OpenAIChatModel``（httpx，支持 tools）；离线自测用 ``MockChatModel``。
"""
from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx


# --------------------------------------------------------------------------- #
# 消息协议：循环里唯一传播的结构（ch03 起 support tool role + tool_calls）
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """模型一次原生 function calling：名字 + 已解析的 JSON arguments。"""

    call_id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Message:
    role: str                                # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] | None = None  # 仅 assistant 回传模型发起的调用
    tool_call_id: str | None = None           # 仅 tool：这条观察回答的是哪个 tool_calls


# --------------------------------------------------------------------------- #
# 返回结构与用量记账
# --------------------------------------------------------------------------- #
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
    tool_calls: list[ToolCall] | None = None  # 原生 function calling（ch03 起）


class ModelError(Exception):
    """模型端点不可达、异常返回、或解析失败（重试耗尽后抛出）。"""


# 这些状态码意味着"端点暂时不可用/被限流"，值得退避重试；业务性 4xx 不重试。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class _Retryable(Exception):
    """内部信号：本次请求失败但值得重试。"""


# --------------------------------------------------------------------------- #
# 抽象接口
# --------------------------------------------------------------------------- #
class ChatModel:
    """一切模型适配器实现``：``建实例后，循环只调 ``complete`` 与 ``aclose``。"""

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> ModelResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


def _serialize_message(m: Message) -> dict:
    """把我们的 Message 翻译成 OpenAI messages 协议：
    assistant 带 tool_calls 时回传结构化调用；tool 消息带 tool_call_id。"""
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content,
            "tool_calls": [
                {"id": t.call_id, "type": "function",
                 "function": {"name": t.name, "arguments": json.dumps(t.arguments)}}
                for t in m.tool_calls
            ],
        }
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    return {"role": m.role, "content": m.content}


# --------------------------------------------------------------------------- #
# 真实适配器：chat/completions（httpx，零依赖，ch03 起支持 tools）
# --------------------------------------------------------------------------- #
class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.2,
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
            "messages": [_serialize_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
        # 重试：可重试状态(408/429/5xx)或传输错误 -> 指数退避 + 抖动；重试耗尽抛 ModelError。
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    raise _Retryable(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
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
        content = message.get("content") or ""
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            try:
                fn = tc.get("function") or {}
                arguments = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError) as exc:
                raise ModelError(f"tool_calls.arguments 非 JSON: {tc}") from exc
            tool_calls.append(ToolCall(call_id=tc.get("id", ""),
                                       name=fn.get("name", ""),
                                       arguments=arguments))
        return ModelResult(
            content=content,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls or None,
        )

    async def aclose(self) -> None:
        await self._http.aclose()


# --------------------------------------------------------------------------- #
# 离线假实现：测试与"污染端点前先跑通"用
# --------------------------------------------------------------------------- #
class MockChatModel(ChatModel):
    """脚本化响应；不联网。

    - 给了 ``script``：逐个弹出预设的 ``ModelResult``（出空后抛 StopIteration 语义错误）。
    - 没给：把最后一条 user 内容回显为 ``mock:<content>``，用量固定。
    """

    def __init__(self, script: list[ModelResult] | None = None) -> None:
        self.script = list(script or [])

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> ModelResult:
        if self.script:
            return self.script.pop(0)
        last = messages[-1].content if messages else ""
        return ModelResult(content=f"mock:{last}", usage=Usage(10, 5), finish_reason="stop")

    async def aclose(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# ch03 · 工具：Tool + ToolRegistry（可注册、可注入、可演化的工具注册表）
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema（OpenAI function calling 用）
    handler: object           # async (dict) -> str；异常当数据（循环不炸，observation 吞错）


class ToolRegistry:
    """登记工具并对外三种能力：
    - ``register(tool)`` 记住工具；
    - ``schema()``     吐给模型的 OpenAI function tools（调用时注入）；
    - ``call(tc)``     执行一次作用调用，**错误当数据**（handler 抛异常 -> 返回错误文本，不往上抛）。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schema(self) -> list[dict]:
        return [
            {"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.parameters}}
            for t in self._tools.values()
        ]

    def available(self) -> str:
        """可读清单：注入 system prompt，让模型知道能用哪些工具、长什么样。"""
        if not self._tools:
            return "（无可用工具）"
        return "\n".join(
            f"- {t.name}({_params_summary(t.parameters)}): {t.description}"
            for t in self._tools.values()
        )

    async def call(self, tc: ToolCall) -> str:
        tool = self._tools.get(tc.name)
        if tool is None:
            return f"错误：没有叫 {tc.name!r} 的工具"
        try:
            # handler 可为 async 或 sync（dict)->str。工具常是 IO 重活，async 推荐；纯函数兼容 sync。
            result = tool.handler(tc.arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as exc:  # noqa: BLE001 —— 错误当数据：任何工具异常都要转观察文本，不能捂死
            return f"错误（{type(exc).__name__}）: {exc}"


def _params_summary(parameters: dict) -> str:
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    return ", ".join(f"{k}{'*' if k in required else ''}" for k in props)


# --------------------------------------------------------------------------- #
# ch04 · 技能：Skill + SkillLibrary（渐进披露，按需加载）
# --------------------------------------------------------------------------- #
@dataclass
class Skill:
    name: str
    description: str       # 一句话说干嘛——只进目录（渐进披露）
    instructions: str      # 完整执行指令——load 时才回填（占预算大户，不能全塞）
    priority: int = 0      # 目录排序用，优先级高的在前

    @classmethod
    def from_markdown(cls, text: str, priority: int = 0) -> Skill:
        """从「frontmatter + 正文」的 markdown 技能文件建 Skill。

        格式（对照 arxiv_search.skill.md 参考例）：

            ---
            name: arxiv_search
            description: 用 arxiv Atom API 检索论文元数据（标题/摘要/作者/分类）#
            priority: 10            # 可选的 frontmatter 键，覆盖参数
            ---
            body……即为 instructions（完整执行指令）

        frontmatter 夹在首行 ``---`` 与下一个 ``---`` 之间，用最简单的 ``key: value`` 解析
        （不做完整 YAML —— 零依赖、只取我们要的 name/description/priority 就够）。
        缺 name/description 视为格式错误。
        """
        meta, body = _split_frontmatter(text)
        name = meta.get("name", "")
        description = meta.get("description", "")
        if not name or not description:
            raise ValueError(f"skill markdown 缺 frontmatter 的 name/description: {text[:60]!r}…")
        if "priority" in meta:
            priority = int(meta["priority"])
        return cls(name=name, description=description,
                   instructions=body.strip(), priority=priority)


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """拆「--- 前 matter ---  正文」；没有 frontmatter 则整段当正文。零依赖 mini-YAML。"""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[end + 4:].lstrip("\n")


def read_skill_file(path: str, priority: int = 0) -> Skill:
    """读一个 .skill.markdown 技能文件成 Skill（磁盘上存放技能的方式）。"""
    with open(path, encoding="utf-8") as fh:
        return Skill.from_markdown(fh.read(), priority=priority)


class SkillLibrary:
    """技能的库：add / get / 渐进披露目录 ``list_summary``。

    关键设计 = **渐进披露**：目录只给"名 + desc + 优先级"，不泄 instructions；
    模型看目录决定要不要 `load_skill(name)` 取完整指令。这和"把一切全塞 system
    prompt"相比：省预算、无关技能不占窗、目录可维护。
    """

    def __init__(self, *skills: Skill) -> None:
        self._skills: dict[str, Skill] = {}
        for s in skills:
            self.add(s)

    @classmethod
    def load_dir(cls, path: str = "skills", *, glob: str = "*.skill.md",
                 priority: int = 0) -> SkillLibrary:
        """从目录批量加载技能文件（每个文件一个技能）。"""
        lib = cls()
        for fp in sorted(Path(path).glob(glob)):
            lib.add(read_skill_file(str(fp), priority=priority))
        return lib

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def list_summary(self, max_items: int = 20) -> str:
        """渐进披露目录：按优先级降序，只给 name + desc，超长截断（目录截断）。"""
        items = sorted(self._skills.values(), key=lambda s: s.priority, reverse=True)
        if not items:
            return "（无可用技能）"
        lines = [f"- {s.name} (优先 {s.priority}): {s.description}" for s in items[:max_items]]
        if len(items) > max_items:
            lines.append(f"…（共 {len(items)} 个技能，只显示前 {max_items}）")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ch05 · 记忆：token 估算 + 预算 + 规则简写 + 模型压缩 + 画像 + 落盘/回填
# --------------------------------------------------------------------------- #
# 用户 4 点设计（对齐生产 DESIGN §7.2/§7.3）：
#   ① 用户画像=长期记忆（模型总结+schema+规则门控轮末更新）；
#   ② 工作记忆=循环的 list[Message]（四类 role），tool 结果进内存先规则简写，超预算模型改写压缩；
#   ③ 每轮完整记忆落盘成文件，context 缺用户提的词就从文件 grep 回填；
#   ④ 工作记忆以 messages 存于循环。
# 贯穿原则：**规则优先、模型只低频重写**——压缩/画像更新才调模型，简写/门控/落盘回填全纯规则，
# 成本和保真度平衡（DESIGN §7.3 记忆总原则）。


def estimate_tokens(text: str) -> int:
    """本地启发式 token 估算：中英混排折中 ≈ len/3。零依赖、压缩前就能算（模型 usage 只有调用后才有）。

    粗糙度接受，校准留给 trace 里的真实 usage（DESIGN："token 估算粗糙……M1 校准"）。
    """
    return max(1, math.ceil(len(text) / 3))


@dataclass
class ContextBudget:
    """工作记忆的上下文预算：超过 ``max_len * threshold`` 触发模型压缩。"""

    max_len: int = 6000          # 工作记忆允许的 token 预算
    threshold: float = 0.8      # 超过 max_len*threshold 触发压缩（比例阈值）
    keep_recent: int = 4        # 压缩时保留最近 N 条原样，只压缩更旧的

    def is_over(self, tokens: int) -> bool:
        return tokens > self.max_len * self.threshold


def abbreviate_observation(content: str, max_chars: int = 500) -> str:
    """基于规则的 tool 结果"提取 + 简写"：压空白、折叠行、超长截断。不调模型，廉价高保真。

    进工作记忆时对 tool 观察用（spec ②），让短期记忆保持精简；关键信息在前段因此不丢。
    """
    content = " ".join(content.split())                 # 压掉一切连续空白，一行内折叠
    if len(content) > max_chars:
        original_len = len(content)
        content = content[:max_chars] + f"[规则简写: 原{original_len}字符，截断]"
    return content


@dataclass
class WorkingMemory:
    """短期记忆 = 循环的 context（同一份 list[Message]，四类 role）。

    - ``ingress`` 规整一条消息：tool 结果按规则简写（返回规整后的消息，循环据以挂载）；
    - ``resync`` 让 ``messages`` 始终等于当前 context（recall 注入 / 压缩后同步）；
    - ``size``/``is_over`` 供预算判断（触发模型压缩）。
    """

    budget: ContextBudget
    messages: list[Message] = field(default_factory=list)

    def ingress(self, msg: Message) -> Message:
        if msg.role == "tool" and msg.content:
            return Message(role="tool", tool_call_id=msg.tool_call_id,
                           content=abbreviate_observation(msg.content))
        return msg

    def resync(self, context: list[Message]) -> None:
        self.messages[:] = context

    def size(self) -> int:
        return sum(estimate_tokens(m.content) for m in self.messages)

    def is_over(self) -> bool:
        return self.budget.is_over(self.size())


def _serialize_for_compress(messages: list[Message]) -> str:
    """把一批 messages 文本化，喂给压缩模型（不保留 tool_call 协议细节，只要内容）。"""
    lines = []
    for m in messages:
        if m.role == "tool":
            lines.append(f"[tool] 结果: {m.content}")
        elif m.role == "assistant" and m.tool_calls:
            names = ", ".join(t.name for t in m.tool_calls)
            lines.append(f"[assistant 调用了 {names}] {m.content}")
        else:
            lines.append(f"[{m.role}] {m.content}")
    return "\n".join(lines)


async def compact_messages(model: ChatModel, messages: list[Message],
                           budget: ContextBudget) -> tuple[list[Message], str | None]:
    """模型改写压缩：旧前缀（除最近 ``keep_recent`` 条）折叠成一条高密度摘要(system)，
    新 messages = [摘要] + recent 尾。返回 (messages, 摘要文本 或 None[没触发])。

    压缩调用用 Mock 脚本化；tools=None（这是摘要，不是动作）。
    """
    older = messages[:-budget.keep_recent] if len(messages) > budget.keep_recent else []
    recent = messages[-budget.keep_recent:]
    if not older:
        return messages, None
    summary = await model.complete(
        [Message(role="system", content="把所有历史工作记忆压缩成一段高密度中文摘要，"
                                       "保留关键事实、结论、已用工具与结果，去掉过程噪声。"),
         Message(role="user", content=_serialize_for_compress(older))],
        tools=None,
    )
    digested = summary.content.strip() or "（压缩摘要为空）"
    condensed = [Message(role="system", content=f"【工作记忆压缩摘要】\n{digested}")]
    return condensed + recent, digested


# --- 用户画像（长期记忆）：规则信号门控 + 轮末模型总结 -------------------------------- #
PROFILE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "identity": {"type": "string", "description": "用户身份/背景"},
        "domain": {"type": "string", "description": "关注领域"},
        "preferences": {"type": "array", "items": {"type": "string"},
                        "description": "明确的偏好"},
        "style": {"type": "string", "description": "偏好回答风格"},
    },
}

# 规则信号：本轮用户文本命中任一 cue 才折叠画像（按需触发，命中才付模型成本）
PROFILE_CUES = ("我是", "我是做", "我喜欢", "我希望", "我倾向于", "偏好", "倾向于",
                "领域", "身份", "风格", "我喜欢", "注意我", "请记住")


def has_profile_cue(messages: list[Message]) -> bool:
    """便宜规则信号：本轮有没有可能写进画像的内容。有才走 LLM（按需）。"""
    text = "".join(m.content for m in messages if m.role == "user")
    return any(cue in text for cue in PROFILE_CUES)


class UserProfile:
    """长期记忆：维护固定 schema 的用户画像，规则门控 + 轮末模型总结更新，常驻 system。"""

    def __init__(self, model: ChatModel, *, schema: dict | None = None,
                 path: str | Path | None = None) -> None:
        self.model = model
        self.schema = dict(PROFILE_SCHEMA if schema is None else schema)
        self.path = Path(path) if path else None
        self.data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.path and Path(self.path).exists():
            try:
                loaded = json.loads(Path(self.path).read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except (ValueError, OSError):
                self.data = {}

    def into_system(self) -> str:
        return f"你是教学 Agent。用户画像：\n{json.dumps(self.data, ensure_ascii=False, indent=1) or '（暂无）'}"

    async def maybe_update(self, messages: list[Message]) -> bool:
        """有肖像信号才折叠；把本轮 messages + old data + schema 交给模型，解析 JSON 存回。"""
        if not has_profile_cue(messages):
            return False
        prompt = (
            f"根据这次对话更新用户画像。当前画像是：{json.dumps(self.data, ensure_ascii=False)}。\n"
            f"只输出一个符合 schema 的 JSON：\n{json.dumps(self.schema, ensure_ascii=False)}"
        )
        last = messages[-1].content if messages else ""
        result = await self.model.complete(
            [Message(role="system", content=prompt),
             Message(role="user", content=f"对话本轮：\n{_serialize_for_compress(messages)}\n最近问我：{last}")],
            tools=None,
        )
        try:
            parsed = json.loads(result.content)
            if not isinstance(parsed, dict):
                return False
        except ValueError:
            return False
        merged = dict(parsed)                      # 模型给出新值；空值沿用
        for k, v in self.data.items():
            merged.setdefault(k, v)
        self.data = merged
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        return True


# --- 完整记忆落盘 + 自动回填（grep 式） --------------------------------------------- #
_STOPWORDS = frozenset({"怎么", "什么", "一个", "这个", "那个", "可以", "大概", "我的",
                        "我会", "请你", "需要", "对于", "如何", "一下", "的是", "以及"})


def _meaningful_terms(text: str) -> list[str]:
    """从查询里抽可检索词，顺带做 CJK 无分词近似：
    - 按标点/空白切出整词；整词若是中文长词再滑窗补出 2 字 bigram（找回可能命中文件的片段）；
    - ASCII/技术词独立抽出（如 GRPO、RL）。
    返回去重后列表（召回用；粗糙度接受，真分词留给 ch07/产品）。
    """
    terms: set[str] = set()
    for t in re.split(r"[\s,，。；;、：:!！?？()（）\[\]\"'#]+", text):
        t = t.strip()
        if len(t) < 2 or t in _STOPWORDS or t.isdigit():
            continue
        terms.add(t)
        if all("一" <= ch <= "鿿" for ch in t):
            for i in range(len(t) - 1):            # 中文整词 -> 补 2 字 bigram
                terms.add(t[i:i + 2])
        else:
            terms.update(re.findall(r"[A-Za-z][A-Za-z0-9_+\-]{1,}", t))  # ASCII/技术词
    return sorted(terms)


def _context_has(context: list[Message], term: str) -> bool:
    blob = " ".join(m.content for m in context)
    return term in blob


class MemoryStore:
    """完整记忆=每轮 messages 落盘成 **JSONL** 文件；按对象解析回填缺失片段。

    做记录存成 ``memory.jsonl``：**一行一个 JSON 对象**（一条 message），同轮共享一个
    ``turn`` 时间戳——比 markdown 稳：content 里任何换行/缩进/`**` 都只是 JSON 字符串里的
    普通字符，不会被错当成结构；角色、tool_call_id、tool_calls 都是结构化字段可恢复。
    ``recall`` 逐行 ``json.loads`` 后按 term 扫 content，命中即取该片段回填。
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.dir = Path(storage_dir) if storage_dir else Path(tempfile.mkdtemp(prefix="eh_mem_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "memory.jsonl"

    def append_turn(self, messages: list[Message]) -> None:
        """把一整轮 messages 追加进 JSONL：同轮共用一个 ``turn`` 时间戳，每条 message 一行。"""
        turn = datetime.datetime.now().isoformat(timespec="seconds")
        lines = []
        for m in messages:
            record = {"turn": turn, "role": m.role,
                      "content": m.content, "tool_call_id": m.tool_call_id}
            if m.tool_calls:
                record["tool_calls"] = [
                    {"call_id": t.call_id, "name": t.name, "arguments": t.arguments}
                    for t in m.tool_calls
                ]
            lines.append(json.dumps(record, ensure_ascii=False))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def recall(self, context: list[Message]) -> str | None:
        """自动回填：最新 user 提到的词若**不在已有 context(除本次提问外)**、却曾在 JSONL 出现，
        取命中的历史片段回填。知识=当前上下文去掉本次提问，避免"用户自己刚提过"被当成已具备。"""
        idx = next((i for i, m in enumerate(context) if m.role == "user"), None)
        if idx is None or not self.path.exists():
            return None
        knowledge = context[:idx]                       # 本次 user 之前的所有消息
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
                continue                             # 坏行跳过（文件脏了不能吞掉召回）
            content = str(record.get("content", ""))
            low = content.lower()
            if any(t.lower() in low for t in candidates):
                hits.append(content or line)
        if not hits:
            return None
        snippet = " | ".join(hits[:8])
        return f"[从完整记忆检索到] {snippet}"


class Memory:
    """记忆聚合：工作记忆(压缩) + 画像(常驻) + 完整记忆(落盘/回填)。给 ``AgentLoop`` 挂钩用。"""

    def __init__(self, model: ChatModel, *, budget: ContextBudget | None = None,
                 profile_schema: dict | None = None, storage_dir: str | Path | None = None,
                 profile_path: str | Path | None = None) -> None:
        self.model = model
        self.budget = budget or ContextBudget()
        self.store = MemoryStore(storage_dir)
        self.profile = UserProfile(model, schema=profile_schema,
                                   path=profile_path or (self.store.dir / "profile.json"))
        self.wm = WorkingMemory(self.budget)
        self.recalled: bool = False
        self.compacted: bool = False

    def overhead(self) -> str:
        """常驻 system 段：用户画像（长期记忆）。"""
        return self.profile.into_system()

    async def hook_pre_step(self, context: list[Message]) -> bool:
        """每步前：①首次 且 用户缺词->自动回填补缺；②预算超限->模型压缩。
        返回是否发生过压缩（供测试/计 trace 用）。"""
        self.wm.resync(context)                       # 工作记忆 = 循环 context（同一份）
        if not self.recalled:
            recalled = self.store.recall(context)
            if recalled and not any("[从完整记忆检索到]" in m.content for m in context):
                context.insert(0, Message(role="system", content=recalled))
                self.wm.resync(context)
            self.recalled = True
        if self.wm.is_over():
            new_context, _ = await compact_messages(self.model, context, self.budget)
            context[:] = new_context          # 就地替换：调用方的 context 列表也一起更新
            self.wm.resync(context)
            self.compacted = True
        return self.compacted

    async def finalize(self, context: list[Message]) -> tuple[bool, bool]:
        """轮末：落盘完整记忆 + 画像按需更新。返回 (profile_updated?, compacted_in_this_run)。"""
        self.store.append_turn(context)
        updated = await self.profile.maybe_update(context)
        return updated, self.compacted


# --------------------------------------------------------------------------- #
# ch03 · 循环（AgentLoop）：ReAct，act = 原生 function call
# --------------------------------------------------------------------------- #
# 并发与健壮性都放边界，不放循环（贯穿原则）：
#   - Agent **内部是"顺序的 async"**：每步依赖上一步，await 让出事件循环即可，intra-turn 不并行。
#   - 重试/指数退避/异常分类归模型适配器（ch01 的 ChatModel），循环只看到"结果 or ModelError"。
#   - 并发 run 在"调用 Agent 的边界"注入（后端轨 ch15 示范）。


@dataclass
class LoopResult:
    answer: str = ""              # 模型不再调工具时的最终文本
    finished: bool = False        # 是否在预算内出最终答案；False = 步数耗尽
    steps: int = 0                # 实际走的步数（每次调模型 = 一步）
    usage: Usage = field(default_factory=Usage)     # 本 run 累计用量
    tool_calls: list[ToolCall] = field(default_factory=list)  # 本 run 实际执行过的工具调用


class AgentLoop:
    """工具 + 技能 + 记忆驱动的 ReAct 循环。构造接"模型 + 工具注册表 + 技能库 + 记忆 + 系统提示 + 步数"。

    - 每步把 ``registry.schema()`` 注入 complete，模型可发起原生 function calling。
    - 给了 ``skills``：自动注册 ``load_skill`` 工具，目录进 system prompt（渐进披露），
      模型 load 后完整指令注入 context（后续步可见），observation 只回一句确认。
    - 给了 ``memory``：走记忆挂钩——画像进 system、tool 结果进工作记忆按规则简写、
      预算超限模型压缩、context 缺词从完整记忆自动回填、轮末落盘+画像更新。
    - 有 tool_calls：回填 assistant 的调用 + 每个 tool 的观察（错误当数据），续。
    - 无 tool_calls：``result.content`` 即最终答案，收尾。
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        registry: ToolRegistry | None = None,
        skills: SkillLibrary | None = None,
        memory: Memory | None = None,
        system: str | None = None,
        max_steps: int = 8,
        # ch07 · sub-agent：可选；给了 `subagent_enabled` 才注册 `subagent` 工具
        base_tools: list[Tool] | None = None,
        subagent_enabled: bool = False,
    ) -> None:
        self.model = model
        self.registry = registry or ToolRegistry()
        self.skills = skills
        self.memory = memory
        self.max_steps = max_steps
        self.system = system
        self.base_tools = list(base_tools or [])
        self.subagent_enabled = subagent_enabled
        self.depth = 0                     # 根 = 0；每委派一层 +1（子代理由 _run_subagent 设置）
        self.max_subagent_depth = 2        # DESIGN §7.4：深度 ≤ 2
        self.steps_used = 0                # 预算测算：子步数 ≤ 父剩余步数的一半
        if base_tools:
            for t in base_tools:
                self.registry.register(t)
        if skills is not None:
            self._register_load_skill()
        if subagent_enabled:
            self._register_subagent()

    def _register_load_skill(self) -> None:
        def _load(args: dict) -> str:
            name = args.get("name", "")
            if name not in self.skills._skills:  # 未知技能：错误当数据
                return f"错误：没有技能 {name!r}；可用：{self._skill_names()}"
            return f"技能 {name} 已加载，完整指令已注入上下文，请严格执行。"
        self.registry.register(Tool(
            name="load_skill",
            description="按名字加载一个技能，把它的完整指令注入上下文以便执行",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            handler=_load,
        ))

    def _skill_names(self) -> str:
        return ", ".join(sorted(self.skills._skills))

    def _system_prompt(self) -> str:
        if self.system is not None:
            return self.system
        prompt = f"你是教学 Agent。先用工具拿事实，再给出最终结论。可用工具：\n{self.registry.available()}"
        if self.skills is not None:
            prompt += (f"\n\n可用技能（渐进披露：先看目录，需要时用 load_skill 加载完整指令）：\n"
                       f"{self.skills.list_summary()}")
        return prompt

    def _inject_skill(self, context: list[Message], name: str) -> None:
        skill = self.skills.get(name)
        context.append(Message(role="system",
                               content=f"[已加载技能 {skill.name}]\n{skill.instructions}"))

    # ---- ch07 · sub-agent：子代理 = 有 Agent 能力的工具 ------------------------- #
    def _register_subagent(self) -> None:
        """把 `subagent(task)` 注册成函数工具：模型在循环里委派子 run。"""
        self.registry.register(Tool(
            name="subagent",
            description=("把一个子任务委派给一个独立的子代理去执行（独立上下文），"
                         "等它完成后只回传最终文本。适合值得单独隔离、深度较小的子问题。"),
            parameters={"type": "object",
                        "properties": {"task": {"type": "string",
                                                "description": "交给子代理的一句话子任务"}},
                        "required": ["task"]},
            handler=self._run_subagent,
        ))

    async def _run_subagent(self, args: dict) -> str:
        """执行一次委派：深度护栏 -> 子预算减半 -> 新 AgentLoop 实例（独立 context）。

        - 只回**最终文本**作为一条 tool observation（父循环按工具结果处理）；
        - 递归护栏 / 子预算耗尽都当数据返回可读串，不炸父循环。
        """
        if self.depth >= self.max_subagent_depth:
            return (f"错误：子代理深度已达上限 {self.max_subagent_depth}，"
                    f"无法在第 {self.depth} 层继续委派子代理")
        task = str(args.get("task", "")).strip()
        if not task:
            return "错误：subagent 需要非空的 task"
        remaining = self.max_steps - self.steps_used
        child_max = max(1, remaining // 2)          # 子步数 ≤ 父剩余预算的一半（DESIGN §7.4）
        # 子代理 = 新 AgentLoop 实例：独立 context / 注册表 / 工作记忆；仍沿用同一模型、技能、
        # 系统提示、工作工具、存储目录（继承同一 State 配置与 runtime）。
        child_memory = None
        if self.memory is not None:
            child_memory = Memory(self.model, storage_dir=self.memory.store.dir,
                                  profile_path=self.memory.profile.path)
        child = AgentLoop(self.model,
                          base_tools=self.base_tools, skills=self.skills,
                          memory=child_memory, system=self.system, max_steps=child_max,
                          subagent_enabled=True)
        child.depth = self.depth + 1
        child.steps_used = 0
        res = await child.run(task)                 # 顺序 await：同一 event loop 交错，intra-turn 不并行
        if not res.finished:
            return f"（子代理步数耗尽，未完成）{res.answer or '无结论'}"
        return f"[子代理完成] {res.answer}"          # 只回最终文本；父 _emit 会将它简写保留核心

    async def run(self, user: str) -> LoopResult:
        sys_content = self._system_prompt()
        if self.memory is not None:
            sys_content += "\n\n" + self.memory.overhead()
        context: list[Message] = [Message(role="system", content=sys_content),
                                  Message(role="user", content=user)]

        # 记忆的"规整挂载"：进工作记忆的同时挂到 context（tool 结果按规则简写）
        def _emit(msg: Message) -> None:
            if self.memory is not None:
                msg = self.memory.wm.ingress(msg)
            context.append(msg)

        async def _finish(result: LoopResult) -> LoopResult:
            if self.memory is not None:
                # 最终答案也进记忆：结论是记忆的一部分，跨轮可被 Recall（工作+完整记忆共伴生成）
                if result.answer:
                    _emit(Message(role="assistant", content=result.answer))
                await self.memory.finalize(context)
            return result

        total = Usage(0, 0)
        done: list[ToolCall] = []

        for step in range(1, self.max_steps + 1):
            self.steps_used = step                     # ch07：子代理预算测算用（父剩余步数）
            if self.memory is not None:
                await self.memory.hook_pre_step(context)  # 自动回填 + 预算超限压缩
            result = await self.model.complete(context, tools=self.registry.schema())
            total = Usage(total.prompt_tokens + result.usage.prompt_tokens,
                          total.completion_tokens + result.usage.completion_tokens)

            if not result.tool_calls:
                # 模型不再要工具 -> 这段文本就是答案
                return await _finish(LoopResult(answer=result.content, finished=True,
                                                steps=step, usage=total, tool_calls=done))

            # 原生 function calling：回填 assistant 的调用，逐个执行观察（顺序 async）
            _emit(Message(role="assistant", content=result.content,
                          tool_calls=result.tool_calls))
            for tc in result.tool_calls:
                done.append(tc)
                observation = await self.registry.call(tc)
                # 技能加载的特殊缝：load_skill 成功后，把完整指令注入 context（不只一次性观察）
                if (self.skills is not None and tc.name == "load_skill"
                        and not observation.startswith("错误")):
                    self._inject_skill(context, tc.arguments["name"])
                _emit(Message(role="tool", tool_call_id=tc.call_id, content=observation))

        return await _finish(LoopResult(finished=False, steps=self.max_steps,
                                        usage=total, tool_calls=done))

        return LoopResult(finished=False, steps=self.max_steps, usage=total, tool_calls=done)


# --------------------------------------------------------------------------- #
# ch03 · 路径安全策略（PathPolicy）：静态 allow/deny + realpath 规约
# --------------------------------------------------------------------------- #
# 静态为主的第一道安全：只读工具也要限——否则 agent 能读仓库里任何文件（.env/.git/…）。
# 机制（薄版）：
#   - canonical 规约：realpath 消掉 ../ 与符号链接（堵路径逃逸）；
#   - allow_root：只许"允许根"以内的路径（默认 = 本文件所在目录）；
#   - deny_names / deny_prefixes：命中的名字（.env/.git/__pycache__…）或系统前缀（/etc 等）直接拒。
#   - 拒绝**当数据**：拒绝返回可读原因串，agent 据此换方向，循环不炸。


@dataclass
class PathPolicy:
    allow_root: str = ""   # 空 -> 默认取 agent.py 所在目录；只在它内部放行
    deny_names: frozenset = frozenset({".env", ".git", ".ssh", "__pycache__",
                                       "node_modules", "dist", "build"})
    deny_prefixes: tuple = ("/etc", "/proc", "/sys", "/dev", "/var/run")

    def _base(self) -> str:
        if self.allow_root:
            return os.path.realpath(self.allow_root)
        return os.path.dirname(os.path.abspath(__file__))

    def judge(self, path: str) -> tuple[bool, str]:
        """canonical 后判定是否放行。返回 (允许?, 原因)。拒绝也是给 agent 的信息，不抛。"""
        p = os.path.abspath(path)
        rp = os.path.realpath(p)                    # 消 ../ 与符号链接
        for comp in rp.split("/"):
            if comp in self.deny_names:
                return False, f"命中被禁名: {comp!r}"
        for prefix in self.deny_prefixes:
            if rp.startswith(prefix):
                return False, f"命中禁读系统前缀: {prefix}"
        base = self._base()
        if rp != base and not rp.startswith(base + os.sep):
            return False, f"越出允许根 {base!r}"
        return True, "ok"


# --------------------------------------------------------------------------- #
# ch03 · 演示性只读工具（仿 nano-harness：连到世界 + 信息丰富错误串 + 静态路径安全）
# --------------------------------------------------------------------------- #
# 这两只是"把 agent 连到世界"的最小只读工具。核心三点（nano_harness 提炼）：
#   - 每个工具要有大小上限 / 超时 / 信息丰富的错误串，agent 才能据以适应；
#   - 错误尽量**当数据**返回可读字符串，而不是让循环炸掉；
#   - 只读也要过 PathPolicy 静态安全（默认 deny .env/.git/系统前缀，且不许逃出允许根）。
# 只读工具先到此为止：本地（list_dir/read_file）+ 网络 GET（web_fetch）。
# 写文件 / 跑任意命令 / 带可写语义网络这类更高权限工具留到后续章（ch20 沙箱、写保护与命令白名单）。


def list_dir(path: str = ".", policy: PathPolicy | None = None) -> str:
    """列出目录内容。路径不存在/不是目录时返回信息丰富的错误串（不抛）。"""
    if policy is not None:
        allowed, why = policy.judge(path)
        if not allowed:
            return f"拒绝：{why}"
    p = os.path.abspath(path)
    if not os.path.isdir(p):
        return f"错误：不是目录: {path!r}"
    try:
        entries = sorted(os.listdir(p))
    except OSError as exc:
        return f"错误（{type(exc).__name__}）: {exc}"
    if not entries:
        return f"{p}: （空目录）"
    return f"{p}:\n" + "\n".join(entries)


def read_file(path: str, max_chars: int = 4000, policy: PathPolicy | None = None) -> str:
    """读取文本文件，超 ``max_chars`` 截断。缺失/不可读/被拒时返回可读错误串（当数据）。"""
    if policy is not None:
        allowed, why = policy.judge(path)
        if not allowed:
            return f"拒绝：{why}"
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        return f"错误：不是文件: {path!r}"
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        return f"错误（{type(exc).__name__}）: {exc}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n…[已截断：全文 {len(text)} 字符，只显示前 {max_chars}]"
    return text


async def web_fetch(url: str, max_chars: int = 8000, timeout: float = 20.0,
                    http_client: httpx.AsyncClient | None = None) -> str:
    """只读网络 GET：抓取一个 ``http``/``https`` URL 的文本开头（可截断、有超时）。

    网络当「连到世界的只读工具」，套同一套原则：大小上限（``max_chars``）+ 超时（``timeout``）+
    信息丰富错误串（错误当数据）。静态护栏（薄版）：只放行 http/https scheme；SSRF / 域名白名单
    等更重网络治理留 ch20。``http_client`` 供测试注入 ``httpx.MockTransport``。
    """
    scheme = url.split("://", 1)[0] if "://" in url else ""
    if scheme not in ("http", "https"):
        return f"错误：只支持 http/https，得到 {scheme!r}"
    owns = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    try:
        response = await client.get(url, headers={"User-Agent": "eh-teaching/0.1"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"错误（{type(exc).__name__}）: {exc}"
    finally:
        if owns:
            await client.aclose()
    text = response.text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n…[已截断：全文 {len(text)} 字符，只显示前 {max_chars}]"
    return text


def builtin_read_tools(policy: PathPolicy | None = None) -> list[Tool]:
    """打包只读工具成可直接 register 的列表（demo/冒烟用）。

    - 本地文件（list_dir / read_file）：默认挂一个 PathPolicy（只许读本目录、拒 .env/.git/系统前缀）。
    - 网络（web_fetch）：只读 http/https GET，自带超时 + 大小截断 + 错误当数据。
    写文件 / 任意命令 / 带 cookie 网络等更高权限工具留到 ch20 沙箱。
    """
    if policy is None:
        policy = PathPolicy()
    return [
        Tool(name="list_dir",
             description=f"列出一个目录的内容（只读；只允许 {policy._base()} 内）",
             parameters={"type": "object",
                         "properties": {"path": {"type": "string"}}},
             handler=lambda a: list_dir(a.get("path", "."), policy=policy)),
        Tool(name="read_file",
             description=f"读取文本文件开头若干字符（只读、可截断；只允许 {policy._base()} 内）",
             parameters={"type": "object",
                         "properties": {"path": {"type": "string"},
                                        "max_chars": {"type": "integer"}},
                         "required": ["path"]},
             handler=lambda a: read_file(a["path"], int(a.get("max_chars", 4000)),
                                         policy=policy)),
        Tool(name="web_fetch",
             description="只读网络 GET：抓取一个 http/https URL 的文本开头（有超时、可截断）",
             parameters={"type": "object",
                         "properties": {"url": {"type": "string"},
                                        "max_chars": {"type": "integer"}},
                         "required": ["url"]},
             handler=lambda a: web_fetch(a["url"], int(a.get("max_chars", 8000)))),
    ]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ch06 · 规划：Todo / Plan（长期） + 意图识别 + PlanAgent 单循环
# --------------------------------------------------------------------------- #
# 用户 2 点要求（对齐 course/README ch06）：
#   ① 长期 Plan + TodoList：大目标先拆成稳定阶段(Plan)，再落成可执行子任务(Todo，ref_step 归属)，
#      逐个执行并监控进度（进度表随步注入 system，模型用 todo_update 标记）。
#   ② 识别"改 Plan"意图：纯规则 cue 门控(is_plan_update)，命中且已有 Plan 才付一次模型低频重规划
#      (revise_plan) —— 同时调整长期 Plan 与后续 Todo（保已完成、只重排剩余）。
# 设计叉：单循环 + todo_update 工具驱动完成/监控；plan+todo 落盘 plan.jsonl cross-run 恢复。
# 贯穿：规则优先、模型低频；工具/技能/记忆/路径安全全部从 ch05 继承。

_TODO_STATUS = ("pending", "in_progress", "done", "skipped")


@dataclass
class Todo:
    """一个可执行子任务；ref_step 指明它归属长期 Plan 的哪个稳定阶段。"""

    id: str
    summary: str
    status: str = "pending"        # pending | in_progress | done | skipped
    ref_step: str = ""


class TodoList:
    """子任务清单：分解出来的可执行列表 + 进度监控（progress / inject_table 供 system 注入）。"""

    def __init__(self, items: list[Todo] | None = None) -> None:
        self.items: list[Todo] = items or []

    def add(self, t: Todo) -> None:
        if self.get(t.id) is not None:
            raise ValueError(f"todo id 冲突: {t.id!r}")
        self.items.append(t)

    def get(self, tid: str) -> Todo | None:
        return next((t for t in self.items if t.id == tid), None)

    def mark(self, tid: str, status: str) -> bool:
        t = self.get(tid)
        if t is None:
            return False
        t.status = status
        return True

    def by_status(self, status: str) -> list[Todo]:
        return [t for t in self.items if t.status == status]

    def done(self) -> list[Todo]:
        return self.by_status("done")

    def pending(self) -> list[Todo]:
        return [t for t in self.items if t.status not in ("done", "skipped")]

    def progress(self) -> str:
        n = len(self.items)
        if n == 0:
            return "0/0"
        pct = round(len(self.done()) / n * 100)
        return f"{len(self.done())}/{n}（{pct}%）"

    def inject_table(self) -> str:
        """进度表：给模型看的当前子任务状态（含 id，模型据此调用 todo_update）。"""
        icon = {"pending": "⬜", "in_progress": "▶", "done": "✔", "skipped": "–"}
        lines = [f"### 子任务进度（{self.progress()}）", "- [状态] id 说明"]
        for t in self.items:
            lines.append(f"- [{icon.get(t.status, '?')}] {t.id} {t.summary}（→{t.ref_step}）")
        return "\n".join(lines)


@dataclass
class PlanStep:
    id: str
    statement: str
    status: str = "pending"


@dataclass
class Plan:
    """长期计划：把大目标拆成的稳定阶段。Todo 的 ref_step 归属到这里。"""

    goal: str
    steps: list[PlanStep]

    def by_id(self, pid: str) -> PlanStep | None:
        return next((s for s in self.steps if s.id == pid), None)

    def step_done(self, pid: str) -> bool:
        s = self.by_id(pid)
        if s is None:
            return False
        s.status = "done"
        return True


def _extract_json(text: str):
    """从模型回复容错剥出第一个 JSON 对象/数组（容忍 ```json 包裹、前后夹文）。解析出就返回，否则 None。"""
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


def _plan_from(data) -> Plan:
    return Plan(
        goal=str(data.get("goal", "")),
        steps=[PlanStep(id=str(s.get("id", f"P{i + 1}")),
                        statement=str(s.get("statement", "")))
               for i, s in enumerate(data.get("steps") or [])],
    )


def _todos_from(data) -> TodoList:
    raw = data.get("todos") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return TodoList()
    return TodoList([Todo(id=str(t.get("id", f"T{i + 1}")),
                          summary=str(t.get("summary", "")),
                          ref_step=str(t.get("ref_step", "")))
                     for i, t in enumerate(raw)])


async def plan_task(model: ChatModel, task: str) -> Plan:
    """task -> 长期 Plan。模型产出 {goal, steps:[{id,statement}]}，规则侧容错解析。"""
    sys = ("你是规划器：把用户的长目标拆成一份长期计划。只输出一个 JSON，不要任何多余文字。"
           "格式：{\"goal\":\"一句话目标\",\"steps\":[{\"id\":\"P1\",\"statement\":\"阶段描述\"}]}")
    result = await model.complete([Message("system", sys), Message("user", task)], tools=None)
    data = _extract_json(result.content)
    if not isinstance(data, dict):
        raise ValueError("plan_task 未能解析出 Plan JSON")  # noqa: TRY004
    plan = _plan_from(data)
    if not plan.steps:
        raise ValueError("plan_task 产出的 Plan 没有 step")
    return plan


async def decompose(model: ChatModel, plan: Plan) -> TodoList:
    """Plan -> TodoList：把每个稳定阶段再拆成可执行子任务（ref_step 归属阶段）。"""
    steps_blob = "\n".join(f"- {s.id}: {s.statement}" for s in plan.steps)
    sys = ("你是任务分解器：把长期计划的每个阶段拆成可执行的子任务 todo。只输出一个 JSON："
           "{\"todos\":[{\"id\":\"T1\",\"ref_step\":\"P1\",\"summary\":\"可执行的一句话\"}]}，"
           "ref_step 填它归属的阶段 id。")
    result = await model.complete(
        [Message("system", sys), Message("user", f"目标：{plan.goal}\n阶段：\n{steps_blob}")],
        tools=None)
    todo = _todos_from(_extract_json(result.content))
    if not todo.items:
        raise ValueError("decompose 未能产 todo")
    return todo


# 规则信号：用户文本命中任一 cue -> 视为"要改 Plan"意图（门控，命中才付模型重规划）
PLAN_UPDATE_CUES = ("不要", "改为", "改成", "先做", "先别", "跳过", "去掉", "别做", "补充",
                    "加一个", "顺序", "方向", "调整", "重新", "新加", "换成", "别再")


def is_plan_update(user_text: str) -> bool:
    """便宜规则信号：这段话是不是在改 Plan（改方向/增删/改序）。"""
    return any(cue in user_text for cue in PLAN_UPDATE_CUES)


async def revise_plan(model: ChatModel, old_plan: Plan, done_todos: list[Todo],
                      user_text: str) -> tuple[Plan, TodoList]:
    """用户改动意图 -> 新长期 Plan + 重生成的（仅剩余）TodoList。保住已完成，只重排没做的部分。"""
    steps_blob = "\n".join(f"- {s.id}: {s.statement} [{s.status}]" for s in old_plan.steps)
    done_blob = "\n".join(f"- {t.ref_step} {t.summary}" for t in done_todos) or "（暂无已完成）"
    sys = ("你是规划修订器：用户改动了方向，请你同步修订长期计划。依据旧的 plan 阶段、已完成 todo、"
           "以及改动要求，输出新 JSON："
           "{\"goal\":\"修订后目标\",\"steps\":[{\"id\":\"P1\",\"statement\":\"..\"}],"
           "\"todos\":[{\"id\":\"T1\",\"ref_step\":\"P1\",\"summary\":\"..\"}]}；"
           "todos 只需列**尚未完成**的部分（已完成的不重复列）。")
    result = await model.complete(
        [Message("system", sys),
         Message("user", f"改动要求：{user_text}\n\n旧长期计划：\n{steps_blob}"
                         f"\n\n已完成 todo：\n{done_blob}")],
        tools=None)
    data = _extract_json(result.content)
    if not isinstance(data, dict):
        raise ValueError("revise_plan 未能解析出 JSON")  # noqa: TRY004
    plan = _plan_from(data)
    todo = _todos_from(data)
    if not plan.steps:
        raise ValueError("revise_plan 新 Plan 没有 step")
    return plan, todo


class PlanTracker:
    """持 Plan(长期) + Todo(子任务)：progress() 监控进度；save/load 为 plan.jsonl（kind 分行 JSONL）。"""

    def __init__(self, plan: Plan | None = None, todo: TodoList | None = None,
                 storage_dir: str | Path | None = None) -> None:
        self.plan = plan
        self.todo = todo or TodoList()
        self.dir = Path(storage_dir) if storage_dir else Path(tempfile.mkdtemp(prefix="eh_plan_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "plan.jsonl"

    def has_plan(self) -> bool:
        return self.plan is not None

    def progress(self) -> str:
        if self.plan is None:
            return "（暂无长期计划）"
        return self.todo.progress()

    def inject_table(self) -> str:
        """长期计划 + 子任务进度表：随步注入 system，监控进度的核心。"""
        if self.plan is None:
            return "# 长期计划\n（暂无长期计划，等待首次任务分解）"
        lines = [f"# 长期计划：{self.plan.goal}"]
        lines += [f"- {s.id}: {s.statement} [{s.status}]" for s in self.plan.steps]
        lines.append(self.todo.inject_table())
        return "\n".join(lines)

    def save(self) -> None:
        recs = [{"kind": "meta", "goal": self.plan.goal}] if self.plan else [{"kind": "none"}]
        if self.plan:
            recs += [{"kind": "step", "id": s.id, "statement": s.statement, "status": s.status}
                     for s in self.plan.steps]
        recs += [{"kind": "todo", "id": t.id, "summary": t.summary,
                  "status": t.status, "ref_step": t.ref_step} for t in self.todo.items]
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in recs))

    @classmethod
    def load(cls, storage_dir: str | Path) -> PlanTracker:
        """从 plan.jsonl 恢复 plan+todo（跨 run/进程）。文件缺失 -> 空 tracker。"""
        tr = cls(storage_dir=storage_dir)
        if not tr.path.exists():
            return tr
        for line in tr.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            kind = r.get("kind")
            if kind == "meta":
                tr.plan = Plan(goal=str(r.get("goal", "")), steps=[])
            elif kind == "step" and tr.plan is not None:
                tr.plan.steps.append(PlanStep(id=str(r.get("id", "")),
                                              statement=str(r.get("statement", "")),
                                              status=str(r.get("status", "pending"))))
            elif kind == "todo":
                tr.todo.add(Todo(id=str(r.get("id", "")), summary=str(r.get("summary", "")),
                                 status=str(r.get("status", "pending")),
                                 ref_step=str(r.get("ref_step", ""))))
        return tr


_PROGRESS_MARK = "# 长期计划"


class PlanAgent:
    """规划驱动的单 ReAct 循环。三分支：改方向(重规划) / 首次(分解) / 继续(沿用) => 执行。

    - 构造：懒加载 tracker（给 storage_dir 则 `PlanTracker.load` 恢复）；总是注册 `todo_update` 工具
      让模型驱动完成与监控；给了 skills 再注册 `load_skill`（渐进披露）。
    - run(user)：①有 Plan 且 `is_plan_update(user)` -> `revise_plan`（调整长期 Plan + 后续 Todo）；
      ②无 Plan -> `plan_task` + `decompose` 首建；③否则沿用现有 Plan+Todo 直接执行。
      执行循环每步把 `tracker.inject_table()` 进度表刷新进 system，模型用 `todo_update` 标记进度，
      并用工具/技能/记忆完成子任务；无 tool_calls -> 最终答案收尾（落盘 plan.jsonl + memory）。
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        registry: ToolRegistry | None = None,
        skills: SkillLibrary | None = None,
        memory: Memory | None = None,
        tracker: PlanTracker | None = None,
        storage_dir: str | Path | None = None,
        system: str | None = None,
        max_steps: int = 20,
    ) -> None:
        self.model = model
        self.registry = registry or ToolRegistry()
        self.skills = skills
        self.memory = memory
        self.system = system
        self.max_steps = max_steps
        self.tracker = tracker or (PlanTracker.load(storage_dir) if storage_dir else PlanTracker())
        self._register_todo_update()
        if skills is not None:
            self._register_load_skill()

    # ---- 工具注册 ----------------------------------------------------------- #
    def _register_todo_update(self) -> None:
        def _upd(args: dict) -> str:
            tid = args.get("id", "")
            status = args.get("status", "")
            if status not in _TODO_STATUS:
                return f"错误：status 必须是 {'|'.join(_TODO_STATUS)}，得到 {status!r}"
            if self.tracker.todo.get(tid) is None:
                return f"错误：没有叫 {tid!r} 的 todo；当前进度 {self.tracker.todo.progress()}"
            self.tracker.todo.mark(tid, status)
            return f"已把 {tid} 标记为 {status}（当前进度 {self.tracker.todo.progress()}）"
        self.registry.register(Tool(
            name="todo_update",
            description=("更新一个子任务的完成状态（监控进度用）。id 取进度表里的 todo id，"
                         "status∈pending|in_progress|done|skipped"),
            parameters={"type": "object",
                        "properties": {"id": {"type": "string"},
                                       "status": {"type": "string",
                                                  "enum": list(_TODO_STATUS)}},
                        "required": ["id", "status"]},
            handler=_upd,
        ))

    def _register_load_skill(self) -> None:
        def _load(args: dict) -> str:
            name = args.get("name", "")
            if name not in self.skills._skills:
                return f"错误：没有技能 {name!r}；可用：{self._skill_names()}"
            return f"技能 {name} 已加载，完整指令已注入上下文，请严格执行。"
        self.registry.register(Tool(
            name="load_skill",
            description="按名字加载一个技能，把它的完整指令注入上下文以便执行",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            handler=_load,
        ))

    def _skill_names(self) -> str:
        return ", ".join(sorted(self.skills._skills))

    def _inject_skill(self, context: list[Message], name: str) -> None:
        skill = self.skills.get(name)
        context.append(Message(role="system",
                               content=f"[已加载技能 {skill.name}]\n{skill.instructions}"))

    # ---- 进度刷新（监控） ---------------------------------------------------- #
    def _refresh_progress(self, context: list[Message]) -> None:
        """把最新进度表写进 context：命中 marker 就地更新，否则追加一条 system。"""
        table = self.tracker.inject_table()
        for i, m in enumerate(context):
            if m.role == "system" and m.content.startswith(_PROGRESS_MARK):
                context[i] = Message(role="system", content=table)
                return
        context.append(Message(role="system", content=table))

    # ---- 主入口 -------------------------------------------------------------- #
    def progress(self) -> str:
        return self.tracker.progress()

    async def run(self, user: str) -> LoopResult:
        # ① 规划阶段：三分支（改方向 / 首次分解 / 继续沿用）
        try:
            if self.tracker.has_plan() and is_plan_update(user):
                self.tracker.plan, self.tracker.todo = await revise_plan(
                    self.model, self.tracker.plan, self.tracker.todo.done(), user)
                self.tracker.save()
            elif not self.tracker.has_plan():
                self.tracker.plan = await plan_task(self.model, user)
                self.tracker.todo = await decompose(self.model, self.tracker.plan)
                self.tracker.save()
        except ValueError as exc:
            return LoopResult(answer=f"规划失败：{exc}", finished=False, steps=0)

        # ② 单执行循环（镜像 AgentLoop 骨架；工具/技能/记忆全部复用）
        sys_content = self.system if self.system is not None else None
        if sys_content is None:
            sys_content = f"你是教学 Agent。先用工具拿事实，再给出最终结论。可用工具：\n{self.registry.available()}"
            if self.skills is not None:
                sys_content += (f"\n\n可用技能（渐进披露：需要时用 load_skill 加载完整指令）：\n"
                                f"{self.skills.list_summary()}")
        if self.memory is not None:
            sys_content += "\n\n" + self.memory.overhead()

        context: list[Message] = [Message(role="system", content=sys_content),
                                  Message(role="user", content=user)]
        self._refresh_progress(context)

        def _emit(msg: Message) -> None:
            if self.memory is not None:
                msg = self.memory.wm.ingress(msg)
            context.append(msg)

        async def _finish(result: LoopResult) -> LoopResult:
            if self.memory is not None:
                if result.answer:
                    _emit(Message(role="assistant", content=result.answer))
                await self.memory.finalize(context)
            self.tracker.save()                       # 持久化 plan+todo（含进度，跨 run 恢复）
            return result

        total = Usage(0, 0)
        done: list[ToolCall] = []
        for step in range(1, self.max_steps + 1):
            if self.memory is not None:
                await self.memory.hook_pre_step(context)
            self._refresh_progress(context)
            result = await self.model.complete(context, tools=self.registry.schema())
            total = Usage(total.prompt_tokens + result.usage.prompt_tokens,
                          total.completion_tokens + result.usage.completion_tokens)
            if not result.tool_calls:
                return await _finish(LoopResult(answer=result.content, finished=True,
                                                steps=step, usage=total, tool_calls=done))
            _emit(Message(role="assistant", content=result.content, tool_calls=result.tool_calls))
            for tc in result.tool_calls:
                done.append(tc)
                observation = await self.registry.call(tc)
                if (self.skills is not None and tc.name == "load_skill"
                        and not observation.startswith("错误")):
                    self._inject_skill(context, tc.arguments["name"])
                _emit(Message(role="tool", tool_call_id=tc.call_id, content=observation))
        return await _finish(LoopResult(finished=False, steps=self.max_steps,
                                        usage=total, tool_calls=done))
