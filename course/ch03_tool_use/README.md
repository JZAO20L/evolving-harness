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

## 3. 本课做到 / 没做到

**做到**：原生 function calling 闭环（注册表 → 注入 schema → 模型吐 tool_calls → 执行 → 观察回填）；
工具 handler 兼容 async/sync；**错误当数据**（未知工具 / handler 异常都转观察文本，循环不炸）；
步数 + 用量 + 工具调用记录。
**没做到（留给后续）**：技能（ch04，把「名+desc+指令」结构化按需加载）；记忆（ch05）；子代理（ch07）；
工具沙箱/权限（后端 ch20）。工具的执行仍是循环里顺序 async，并发在边界注入。

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

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/core/message.py` 已把 `tool_calls` 做进 `Message`；`tools/base.py` 有工具抽象、
`tools/` 下自带一批内建工具（read/write/bash 等）、`tracing/` 记录每次 tool 调用与结果。
对比：教学版为讲清「注册表 + function calling + 错误当数据」而最小化；产品版把工具做成了
**可插拔 + 可追溯 + 可授权**的完整子系统（ch04~ch07 会逐步补齐这些接缝）。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：14 个用例，scripted tool_calls + MockTransport 抓协议
uv run pytest course/ch03_tool_use/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key）：真 deepseek 用 add 工具算两数之和
EH_SMOKE=1 uv run pytest course/ch03_tool_use/ -m smoke -s
uv run ruff check course/
```