# core · 消息协议

单 agent 与模型、工具之间共用的消息类型。循环本身在 `evolving_harness/agent.py`，这里只放协议。

## 文件

| 文件 | 职责 | 入口 |
|---|---|---|
| `message.py` | `Message` / `ToolCall`，与 OpenAI 消息格式互转 | `Message.to_openai()` / `.from_openai()` |

## 依赖方向

```
message.py        （无内部依赖）
    ^
models/ tools/ tracing/ agent.py
```

所有上层模块只依赖这里的类型，不反向依赖具体实现。

## 约定

- 协议保持 OpenAI 兼容，新增字段视为协议变更。
- `ToolCall.from_openai` 对坏 JSON 容错，原始参数放进 `__raw__`。
- 不在本目录放循环、工具执行或持久化逻辑。
