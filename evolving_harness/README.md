# evolving_harness · Python 运行时

一个 agent 的动态逻辑集中在这里；静态状态在 `agents/<name>/agent.yaml`。

## 文件

| 文件 | 职责 |
|---|---|
| `agent.py` | `AgentState`（加载 / 校验 / hash / 渲染）、`Agent` 循环、模型与工具装配 |
| `cli.py` | `eh run`，薄壳；不硬编码 prompt |
| `core/` | 消息与 tool_call 协议 |
| `models/` | OpenAI 兼容适配器、Mock 模型 |
| `tools/` | 工具注册表、文件工具 |
| `tracing/` | JSONL trace |

## 依赖方向

```
cli.py ──> agent.py ──> core.message
                    ├──> models
                    ├──> tools
                    └──> tracing
```

`agent.py` 不 import `cli.py`；下层模块不 import `agent.py`。外部（CLI、评测器）只通过
`AgentState` / `Agent` / `run_agent` 调用，不改循环内部。

## 约定

- 一次 run 只持有一份 `messages`；其余状态全部来自 `AgentState` 与 trace。
- 工具错误作为数据回填，模型错误才中断。
- Python 是实验中冻结的运行时；可变面只在 YAML。
