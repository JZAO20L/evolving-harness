# models · 模型适配层

把不同推理端点统一成 `ChatModel.complete()`。循环只认这个接口。

## 文件

| 文件 | 职责 |
|---|---|
| `base.py` | `ChatModel` 抽象、`ModelResponse`、`Usage`、`ModelError` |
| `openai_compat.py` | OpenAI `/chat/completions` 兼容端点（vLLM / DeepSeek / Qwen 等），仅对 429/5xx 重试 |
| `mock.py` | 离线确定性模型，测试与 `--mock` 使用；可脚本化多轮响应 |

## 入口

```python
model = OpenAIChatModel(model=..., base_url=..., api_key=...)
response = await model.complete(messages, tools=registry.specs())
await model.aclose()
```

## 约定

- 新增端点优先复用 `openai_compat`（多数推理框架都兼容），只有协议确实不同才新增适配器。
- 适配器负责把厂商响应归一化为 `ModelResponse`，循环里不出现厂商字段。
- 重试只覆盖传输错误、429 与 5xx；4xx 直接失败。
