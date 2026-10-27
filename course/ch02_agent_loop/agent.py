"""course/ch02 · agent-loop —— 把"单次调用"变成"循环"（ReAct 最小实现）。

ch01 建立了一次 LLM 调用（Message / Usage / ModelResult / ChatModel 三适配器）。
本课在它之上加一个**循环**：组装 context → 调模型 → **解析 act（action_type + JSON params）**
→ 分发/观察回填 → 判定停止/继续，并累计步数 + 用量。

ReAct 范式：循环检测的是 **act**（一次"动作选择 + JSON 参数"），而非仅 tool_call。
ch02 先做**纯 JSON act**（模型输出里嵌 JSON，循环解析，讲清 ReAct 概念）；
ch03 才升级为原生 function calling。tool 角色在 ch02 的循环里作为观察回填出现，
正式的工具注册表/function-calling 协议留给 ch03。

涉及网络的适配器依次叫 ``OpenAIChatModel``（httpx，ch01 同款）；离线自测用 ``MockChatModel``。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import httpx


# --------------------------------------------------------------------------- #
# 消息协议：循环里唯一传播的结构（ch01 先只存 role + content）
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str  # system | user | assistant | （tool 留到 ch03）
    content: str


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

    async def complete(self, messages: list[Message]) -> ModelResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 真实适配器：chat/completions（httpx，零依赖）
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

    async def complete(self, messages: list[Message]) -> ModelResult:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
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
        content = (choice.get("message") or {}).get("content") or ""
        return ModelResult(
            content=content,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason"),
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

    async def complete(self, messages: list[Message]) -> ModelResult:
        if self.script:
            return self.script.pop(0)
        last = messages[-1].content if messages else ""
        return ModelResult(content=f"mock:{last}", usage=Usage(10, 5), finish_reason="stop")

    async def aclose(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# ch02 · act：循环里携带的"一次动作选择 + JSON 参数"
# --------------------------------------------------------------------------- #
@dataclass
class Act:
    action_type: str  # 动作种类：ch02 只有 finish；工具类 act 留给 ch03 注册
    params: dict = field(default_factory=dict)

    @classmethod
    def finish(cls, answer: str = "", **extra) -> Act:
        return cls(action_type="finish", params={"answer": answer, **extra})


def parse_act(text: str) -> Act:
    """从模型输出里抠出 ``{act: {action_type, params}}``，容忍被代码块/叙述包裹。

    约定（写进 system prompt，让模型端到端只吐一个 JSON 对象）：::

        {"thought": "...", "act": {"action_type": "finish", "params": {...}}}

    若整段都是合法 JSON 直接用；否则去掉 fenced 代码块后，取第一个含 ``"act"`` 键的 JSON 对象。
    解析失败抛 ``ModelError``（模型没按协议输出 -> 当错误处理，循环上层可据此重试/终止）。
    """

    import json
    import re

    def _parse(obj) -> Act:
        if not isinstance(obj, dict) or "act" not in obj:
            raise ModelError(f"响应缺少 act 字段: {obj!r}")
        a = obj["act"]
        if not isinstance(a, dict) or "action_type" not in a:
            raise ModelError(f"act 结构异常: {a}")
        return Act(action_type=str(a["action_type"]), params=dict(a.get("params") or {}))

    text = text.strip()
    # 去掉 fenced 代码块（```json ... ``` 或 ``` ... ```），剩下若整段是干净 JSON 直接解析
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates = fenced or [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return _parse(json.loads(cand))
        except (json.JSONDecodeError, TypeError):
            continue
    # 整段不是干净 JSON（模型夹叙述/语气词）：用花括号配平扫描器逐个试 { 起的对象，
    # 第一个合法且含 "act" 的 JSON 对象胜出。配平而非正则，能跨 params 里的嵌套 { }。
    for m in _scan_json_objects(text):
        try:
            return _parse(json.loads(m))
        except (json.JSONDecodeError, TypeError):
            continue
    raise ModelError(f"无法从模型输出解析 act: {text[:200]!r}")


def _scan_json_objects(text: str):
    """从任意文本里逐个产出以 { 开头、花括号配平的 JSON 对象（用 dict 判配平，允许嵌套）。"""
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue  # 多余的右括号，忽略
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


# --------------------------------------------------------------------------- #
# ch02 · 循环（AgentLoop）：ReAct —— Thought/Act/Observation
# --------------------------------------------------------------------------- #
# 并发与健壮性都放边界，不放循环（贯穿原则）：
#   - Agent **内部是"顺序的 async"**：每步依赖上一步，await 让出事件循环即可，intra-turn 不并行。
#   - 重试/指数退避/异常分类归模型适配器（ch01 的 ChatModel），循环只看到"结果 or ModelError"。
#   - 并发 run 在"调用 Agent 的边界"注入（后端轨 ch15 示范）。


@dataclass
class LoopResult:
    answer: str = ""          # finish 时 answer
    finished: bool = False    # 是否在预算内走到 finish；False = 步数耗尽
    steps: int = 0            # 实际走的步数（每次调模型 = 一步）
    usage: Usage = field(default_factory=Usage)   # 本 run 累计用量
    acts: list[Act] = field(default_factory=list) # 本 run 出现的所有 act


_STUB_DISPATCH_SYSTEM = """\
你是教学 Agent。每次回复只输出一个 JSON 对象，不要额外文字：
{"thought": "你的中间推理", "act": {"action_type": "finish", "params": {"answer": "最终回答"}}}

请按 ReAct 格式思考，最后以 action_type=finish 收尾给出 answer。
"""


class AgentLoop:
    """把单次调用包成循环。构造时只接"模型 + 系统提示 + 步数预算 + 可选分发器"。"""

    def __init__(
        self,
        model: ChatModel,
        *,
        system: str = _STUB_DISPATCH_SYSTEM,
        max_steps: int = 8,
        dispatch: callable[[Act], str] | None = None,
    ) -> None:
        self.model = model
        self.system = system
        self.max_steps = max_steps
        # 分发器：act -> observation 文本。缺省返回"未注册工具"提示；ch03 给工具注册表闭环。
        self._dispatch = dispatch or (lambda act: f"[未注册] act={act.action_type}，工具注册表在 ch03 实现")

    async def run(self, user: str) -> LoopResult:
        context: list[Message] = []
        if self.system:
            context.append(Message(role="system", content=self.system))
        context.append(Message(role="user", content=user))

        total = Usage(0, 0)
        acts: list[Act] = []

        for step in range(1, self.max_steps + 1):
            result = await self.model.complete(context)
            total = Usage(total.prompt_tokens + result.usage.prompt_tokens,
                          total.completion_tokens + result.usage.completion_tokens)

            act = parse_act(result.content)
            acts.append(act)

            if act.action_type == "finish":
                return LoopResult(
                    answer=act.params.get("answer", ""), finished=True,
                    steps=step, usage=total, acts=acts,
                )

            # 其余 act 视为"待观察"：回填 assistant act + observation，再进入下一步
            observation = self._dispatch(act)
            context.append(Message(role="assistant", content=result.content))
            context.append(Message(role="tool", content=observation))
            # 注：本步步数已 +1；若还想留观察/再思考的预算，可把 max_steps 调大

        return LoopResult(finished=False, steps=self.max_steps, usage=total, acts=acts)