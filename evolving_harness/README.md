# evolving_harness · Python 运行时

一个 agent 的动态逻辑集中在这里；静态状态在 `agents/<name>/agent.yaml`。

核心产品 = **两个 hero 文件**（`agent.py` + `rsi.py`）+ 支撑子包。

## 文件

| 文件 | 职责 |
|---|---|
| `agent.py` | `State`（加载 / 校验 / hash / 渲染）、`Agent` 循环、模型与工具装配、subagent |
| `rsi.py` | RSI 骨架（v0.2 占位）：只依赖 `agent.py` 公开入口 |
| `cli/` | `eh run` 入口点，薄壳；pyproject 指向 `evolving_harness.cli:main` |
| `core/` | 消息与 tool_call 协议（最底层） |
| `models/` | `ChatModel` 抽象 + Mock / OpenAI-compatible / ARK(Responses) 适配器 |
| `tools/` | 工具注册表、文件 / bash 工具（隔离元数据） |
| `tracing/` | JSONL trace 记录器 |

## 依赖方向（单向下行）

```
cli/ ──> agent.py ──> core.message
                   ├──> models
                   ├──> tools
                   └──> tracing
rsi.py ──> agent.py   （只依赖公开入口）
```

`agent.py` 不 import `cli/`；下层模块不 import `agent.py`。外部（CLI、评测器）只通过
`State` / `Agent` / `run_agent` 调用，不改循环内部。模型选择由 `create_model` 按 base_url
宿主 / `provider` 路由：ARK（volces.com）→ `models.ark.ArkChatModel`（openai SDK Responses
API），其余 → `OpenAIChatModel`，`--mock` → `MockChatModel`。

## 约定

- 一次 run 只持有一份 `messages`；其余状态全部来自 `State` 与 trace。
- 工具错误作为数据回填，模型错误才中断。
- Python 是实验中冻结的运行时；可变面只在 YAML。