# ch03 · 工具使用（tool-use）

ch02 用「纯 JSON act」把 ReAct 讲清楚了；但模型把动作塞进**文本**里由循环解析，既不稳又费劲。
本课升级为**原生 function calling**：模型直接吐结构化的 `tool_calls`，循环不再解析文本、直接分发。
同时把工具收进一个**可注册、可注入、可演化**的注册表 `ToolRegistry`（ch04 技能 / ch10 演化它的土壤）。

文件：`agent.py`（ch01/02 的适配层原样保留，本课**新增** `ToolCall` / `Tool` / `ToolRegistry`，
并用原生 function calling 重构了循环；ch02 的 `Act` / `parse_act` 被取代，不再需要）。

---

## 0. 流程图

```
        ┌─────────────────────────────────────────────┐
        │  run(user)                                  │
        │  context = [system(含工具清单), user]         │
        └──────────┬──────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────────────────────┐
   │  for step = 1 .. max_steps              │
   └──────────┬─────────────────────────────┘
              ▼
   ┌────────────────────────────────────────┐
   │  complete(context, tools=schema)        │◄──────────┐
   │   （原生 function calling）              │           │
   └──────────┬─────────────────────────────┘           │
              ▼                                        │
   ┌────────────────────────────────────────┐           │
   │  result.tool_calls ?                    │           │
   └───────┬─────────────────────┬──────────┘           │
       有 tool_calls           无 tool_calls            │
           │                     │                      │
           ▼                     ▼                      │
   ┌────────────────┐   ┌───────────────────┐           │
   │ 回填 assistant   │   │ result.content 即  │          │
   │ 的调用（结构化）  │   │ 最终答案 -> 返回     │          │
   └───────┬────────┘   └───────────────────┘           │
           ▼                                           │
   ┌──────────────────────────────────────┐            │
   │ for tc in tool_calls:                 │            │
   │   obs = await registry.call(tc)       │            │
   │   // 错误当数据：hander 抛异常 → obs=   │            │
   │   //   错误文本，不往上抛              │            │
   │   context += [tool tc, obs]          │            │
   └───────────────────┬──────────────────┘            │
                       ▼                              │
              继续下一轮 ───────────────────────────────┘
```

## 1. 用的工具

| 工具 | 作用 |
|---|---|
| `OpenAI function calling`（`tools` 参数） | 模型原生吐 `tool_calls`（name + JSON arguments），结构有保障 |
| `inspect.isawaitable` | 让工具 handler 兼容 async / sync（纯函数免 async 包装） |
| `json` | 序列化 assistant 的 tool_calls、解析模型返回的 arguments |

## 2. 实现的对象 + 用法

### `ToolCall(call_id, name, arguments)`
模型一次原生调用：id + 工具名 + **已解析的 JSON 参数**（dict）。`ModelResult.tool_calls` 携带。
`Message` 相应扩展：`assistant` 可带 `tool_calls`，`tool` 消息带 `tool_call_id` 指向回答哪次调用。

### `Tool(name, description, parameters, handler)` + `ToolRegistry`
把工具说清楚 + 收进来：`register(tool)` / `get(name)` / `schema()`→OpenAI function tools /
`available()`→可读清单（注入 system prompt）。核心分发在 `call(tc)`：

```python
async def call(tc) -> str:
    # 工具名未知：错误文本；handler 抛任何异常：错误文本（错误当数据）
    # handler 可为 async 或 sync
```

### `AgentLoop(model, registry=..., system=..., max_steps=...)`
ch02 循环的「act = 原生 function call」版。`run(user)` 每步 `complete(context, tools=schema)`，
**有 tool_calls → 逐个执行、观察回填、继续；没有 → content 就是答案，收尾**。

```python
import asyncio
from agent import AgentLoop, Message, MockChatModel, ModelResult, Tool, ToolCall, ToolRegistry, Usage

async def demo():
    reg = ToolRegistry()
    reg.register(Tool(name="add", description="两数相加",
                      parameters={"type": "object",
                                  "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                                  "required": ["a", "b"]},
                      handler=lambda a: str(a["a"] + a["b"])))
    loop = AgentLoop(MockChatModel(script=[
        ModelResult("", tool_calls=[ToolCall("c1", "add", {"a": 1, "b": 2})], usage=Usage(1, 1)),
        ModelResult("和是 3", usage=Usage(1, 1)),
    ]), registry=reg, max_steps=5)
    res = await loop.run("1+2=?")
    print(res.answer, res.steps, [t.name for t in res.tool_calls])  # 和是 3 2 ['add']

asyncio.run(demo())
```

### 只读内置工具 `list_dir` / `read_file` / `builtin_read_tools()`（仿 nano-harness）
演示「工具连到世界」的最小只读集：`list_dir(path)` 列目录、`read_file(path, max_chars=4000)`
读文件（超长截断）。三个关键设计来自 nano-harness：**大小上限**（`max_chars`）+ **信息丰富错误串**
（路径不存在返回「错误：不是文件: …」而非抛异常，agent 才能据以修正）+ **静态路径安全**。
`builtin_read_tools()` 把它们打包成可直接 `register` 的 `Tool` 列表。写文件 / 跑命令 / 网络等
高权限工具刻意**不在此章**，留到 ch20 沙箱 / 写保护 / 命令白名单。

```python
from agent import ToolRegistry, builtin_read_tools
reg = ToolRegistry()
for t in builtin_read_tools():
    reg.register(t)          # 默认自带 PathPolicy：只在 agent.py 所在目录内放行
```

### `PathPolicy(allow_root, deny_names, deny_prefixes)` —— 静态路径安全
只读也要限，否则 agent 能读仓库里任何文件（`.env` / `.git` / `/etc` …）。`PathPolicy` 是最薄的一道：

1. **canonical 规约**：`realpath` 消掉 `../` 与符号链接，堵死路径逃逸；
2. **allow_root**：只允许「允许根」以内（默认 = `agent.py` 所在目录）；
3. **deny_names / deny_prefixes**：命中被禁名（`.env`/`.git`/`__pycache__`…）或系统前缀（`/etc`/`/proc`…）直接拒。

`judge(path) -> (allowed, reason)` 不抛；`read_file`/`list_dir` 收到 policy 时，拒绝返回「拒绝：原因」
**当数据**回填——agent 会换方向，循环不炸（和「错误当数据」同一句）。`builtin_read_tools()` 默认就挂上
`PathPolicy()`。动态授权（临时放行某路径、approve/deny 按钮）是它的**第二道**，因需把决定送到 UI，
留到 ch17 WebSocket / ch21 gradio 那章。

## 3. 本课做到 / 没做到

**做到**：原生 function calling 闭环（注册表 → 注入 schema → 模型吐 tool_calls → 执行 → 观察回填）；
工具 handler 兼容 async/sync；**错误当数据**（未知工具 / handler 异常都转观察文本，循环不炸）；
一个**真实只读工具集**（`list_dir` / `read_file`，带 `max_chars` 截断 + 信息丰富错误串）；
**静态路径安全**（`PathPolicy`：realpath 规约 + allow_root + deny 名单，拒绝当数据）；
步数 + 用量 + 工具调用记录。
**没做到（留给后续）**：技能（ch04）；记忆（ch05）；子代理（ch07）；**写文件 / 跑命令 / 网络**等高权限
工具及其沙箱 / 写保护 / 命令白名单（ch20）。工具的执行仍是循环里顺序 async，并发在边界注入。

## 4. 关联八股（~12 题）

1. function calling 与「模型输出 JSON 文本」的区别？为什么原生更稳？
2. function-calling 协议往返：assistant 的 `tool_calls`、`tool` 消息的 `tool_call_id` 各起什么作用？
3. `arguments` 为什么是**字符串 JSON**？解析失败的职责在哪（适配器/循环）？
4. tools schema 用什么格式方言（JSON Schema 子集）？`required` / properties 的角色？
5. 为什么「错误当数据」？工具一旦抛异常，往上抛 vs 转观察文本，对循环稳定性分别意味着什么？
6. handler 该 async 还是 sync？纯函数怎么办？（`inspect.isawaitable`）
7. 工具注册表 vs「循环里写死 if/else」？对**可复用**和**可演化**（ch10）各有什么意义？
8. `schema()` 与 `available()` 有什么区别？分别喂给协议的哪里？
9. 一次 complete 会返回多个 tool_calls（并行意图）——本课是顺序执行还是并发？为什么这么选？
10. 模型「一直调用工具不停」怎么办？`max_steps` 的角色？`finished=False` 对上层（verifier）意味着什么？
11. prompt 注入面：工具名/描述会进 system prompt，恶意工具的 description 可能诱导模型——怎么防？
12. 无工具时循环退化为什么？`complete(context, tools=[])` 与「直接答」等价吗？
13. 为什么每个工具都要有**大小上限 / 超时 / 信息丰富的错误串**？（nano-harness：tool 连到世界，agent 靠错误串适应）
14. 工具的安全边界：读 vs 写、命令白名单、网络——为什么高权限工具要靠后/单独治理（对应 ch20 沙箱）？
15. 路径逃逸有哪几招？（`../`、符号链接、绝对路径）为什么先 `realpath` 规约再做 allow/deny？
16. deny 名单 vs allow 白名单：哪个更安全？`allow_root` 默认取 agent.py 目录够用吗，真实产品该取哪？
17. 「拒绝当数据」vs「拒绝就抛」：agent 拿到拒绝串能自救吗？为什么拒绝信息要**可读、可据以修正**？
18. 静态策略挡不住什么（用户临时想读某处）？动态授权为何必须把决定前送到 UI（ch17/21）？

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/core/message.py` 已把 `tool_calls` 做进 `Message`；`tools/base.py` 有工具抽象、
`tools/` 下自带一批内建工具（read/write/bash 等）、`tracing/` 记录每次 tool 调用与结果。
对比：教学版为讲清「注册表 + function calling + 错误当数据」而最小化；产品版把工具做成了
**可插拔 + 可追溯 + 可授权**的完整子系统（ch04~ch07 会逐步补齐这些接缝）。

另见 `docs/reference/nano_harness.md`（HF 的 nano-harness）作对照：它是**代码执行型** harness
（agent 输出代码块、循环 `exec`，工具以函数名注入 prompt），与我们的 function calling 是两条路线。
但它的工具设计原则完全可移植——「每个工具要 size limits、timeout、informative error strings，
agent 才能适应」，加上 `ALLOW_WRITE` 写保护 / `ALLOW_COMMANDS` 命令白名单。这些正是 ch20 沙箱的雏形。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：21 个用例，scripted tool_calls + MockTransport 抓协议 + 只读工具
uv run pytest course/ch03_tool_use/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key）：真 deepseek 用 read_file 读仓库 agent.py 再作答
EH_SMOKE=1 uv run pytest course/ch03_tool_use/ -m smoke -s
uv run ruff check course/
```