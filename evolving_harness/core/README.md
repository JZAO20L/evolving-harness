# core · Agent 内核

单 agent 循环与消息类型。这里是"当下怎么运转"的唯一实现，不包含协议适配、具体工具或持久化。

## 文件

| 文件 | 职责 | 入口 |
|---|---|---|
| `message.py` | `Message` / `ToolCall`，与 OpenAI 消息格式互转 | `Message.to_openai()` / `.from_openai()` |
| `loop.py` | 模型 → 工具调用 → 结果回填 的循环，步数硬上限，逐事件写 trace | `AgentLoop.run()` |

## 依赖方向

```
message.py        （无内部依赖）
    ^
loop.py  ──> models/base.py, tools/base.py, tracing/recorder.py
```

`loop` 只依赖三者的抽象接口，不依赖 OpenAI 适配器或文件工具的具体实现。

## 约定

- 一次 run 的状态只有 `messages`；循环不持有隐式状态。
- 工具错误不抛异常，作为 `tool` 消息回填。
- 任何新增终止条件都必须写 trace。
