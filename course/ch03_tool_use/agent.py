"""course/ch03 · tool-use —— 模型的"原生 function calling"闭环 + 工具注册表。

ch01 一次调用、ch02 一个循环（用纯 JSON act 讲清 ReAct）。本课把 ch02 的"act 是 JSON 文本"
升级为**原生 function calling**：模型直接吐结构化的 ``tool_calls``（name + JSON arguments），
循环**不再靠解析文本**下 act，而是：

    complete(messages, tools=registry.schema())
      -> result.tool_calls?  -> 逐个执行（registry.call，错误当数据）-> 观察回填 -> 续
      -> 无 tool_calls?      -> result.content 即最终答案 -> 停

工具从"循环随手写的字符串"变成**可复用、可注入、可演化的注册表**（ch04 技能 / ch10 演化它的土壤）。
贯彻原则不变：循环内顺序 async、重试/异常分档归适配器、并发在边界注入。

涉及网络的适配器是 ``OpenAIChatModel``（httpx，支持 tools 参数）；离线自测用 ``MockChatModel``。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
from dataclasses import dataclass, field

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
    """工具驱动的 ReAct 循环。构造时接"模型 + 工具注册表 + 系统提示 + 步数预算"。

    - 每步把 ``registry.schema()`` 注入 complete，模型可发起原生 function calling。
    - 有 tool_calls：回填 assistant 的调用 + 每个 tool 的观察（registry.call，错误当数据），续。
    - 无 tool_calls：``result.content`` 即最终答案，收尾。
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        registry: ToolRegistry | None = None,
        system: str | None = None,
        max_steps: int = 8,
        # 关键设计 · 运行期用户输入注入（ch02 引入、顺延本课）：
        input_queue: asyncio.Queue | None = None,
    ) -> None:
        self.model = model
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps
        self.system = system  # 不传就自动按可用工具生成默认提示
        self.input_queue = input_queue

    # ---- 关键设计 · 运行期用户输入注入（ch02 引入；每步边界优先处理） ------------ #
    def _drain_input(self) -> str | None:
        if self.input_queue is None:
            return None
        try:
            return self.input_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _ingest_user_input(self, context: list[Message]) -> int:
        n = 0
        while True:
            text = self._drain_input()
            if text is None:
                break
            context.append(Message(role="user",
                                   content=f"【用户运行中插入，请优先处理】{text}"))
            n += 1
        return n

    def _system_prompt(self) -> str:
        return (
            self.system
            if self.system is not None
            else f"你是教学 Agent。先用工具拿事实，再给出最终结论。可用工具：\n{self.registry.available()}"
        )

    async def run(self, user: str) -> LoopResult:
        context: list[Message] = [Message(role="system", content=self._system_prompt()),
                                  Message(role="user", content=user)]

        total = Usage(0, 0)
        done: list[ToolCall] = []

        for step in range(1, self.max_steps + 1):
            self._ingest_user_input(context)          # 每步边界先收用户注入（优先处理）
            result = await self.model.complete(context, tools=self.registry.schema())
            total = Usage(total.prompt_tokens + result.usage.prompt_tokens,
                          total.completion_tokens + result.usage.completion_tokens)

            if not result.tool_calls:
                # 模型不再要工具 -> 这段文本就是答案
                return LoopResult(answer=result.content, finished=True,
                                  steps=step, usage=total, tool_calls=done)

            # 原生 function calling：回填 assistant 的调用，逐个执行观察（顺序 async）
            context.append(Message(role="assistant", content=result.content,
                                   tool_calls=result.tool_calls))
            for tc in result.tool_calls:
                done.append(tc)
                observation = await self.registry.call(tc)
                context.append(Message(role="tool", tool_call_id=tc.call_id,
                                       content=observation))

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

    网络也当"连到世界的只读工具"，套同一套原则：**大小上限**（``max_chars`` 截断）+
    **超时**（``timeout``）+ **信息丰富错误串**（错误当数据，agent 可据以调整）。
    静态护栏（薄版）：只放行 http/https scheme，堵 ``file://``、``gopher://`` 等非容器协议。
    对内网扫描(SSRF)、域名白名单等更重的网络治理留到 ch20 沙箱。

    ``http_client`` 可选：测试注入 ``httpx.MockTransport`` 用；不传则临时建一个（用完即关）。
    """
    scheme = url.split("://", 1)[0] if "://" in url else ""
    if scheme not in ("http", "https"):
        return f"错误：只支持 http/https，得到 {scheme!r}"
    owns = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    try:
        response = await client.get(url, headers={"User-Agent": "eh-teaching/0.1"})
        # 非 2xx（404/5xx…）也当数据回填，不往上抛
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
    写文件 / 任意命令 / 带 cookie 的网络等更高权限工具留到 ch20 沙箱。
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