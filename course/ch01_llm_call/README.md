# ch01 · LLM 调用（llm-call）

一个 agent 的起点：**把一段「消息」发给模型，拿回「一段文本 + 用量」**；
并把消息、结果、用量都定义成可复用的最小结构。本课**不**引入 loop / act /
function calling（那是 ch02 起的增量），先把"一次调用"这层地基立稳。

文件：`agent.py`（本章唯一 hero）。

---

## 1. 用的工具

| 工具 | 作用 |
|---|---|
| `httpx.AsyncClient` | 打 OpenAI 兼容的 `/chat/completions`；异步、自带 MockTransport 可离线测试 |
| `dataclasses` | 定义 `Message` / `Usage` / `ModelResult` 三个值对象 |

> 刻意零第三方 LLM 依赖：为的是把"协议长什么样"看清楚，而不是包一层 SDK 掩盖细节。
> （产品版 `evolving_harness` 用 openai SDK / Responses API 是 ch01 之后的"生产版延伸阅读"。）

## 2. 实现的对象 + 用法

本课定义了四个对象：

### `Message(role, content)`
循环里唯一传播的消息结构。`role` 取 `system` / `user` / `assistant`（`tool` 留到 ch03）。

### `Usage(prompt_tokens, completion_tokens)`
用量记账，`total_tokens` 是派生属性。agent 靠它统计一次 run 花了多少 token。

### `ModelResult(content, usage, finish_reason)`
一次调用的完整返回：文本 + 用量 + 为什么结束（`stop` / `length` 等）。

### `ChatModel`（接口）+ `OpenAIChatModel` + `MockChatModel`
一切适配器都实现「建实例 → `complete(messages)` → 用完 `aclose()`」。循环只依赖这个接口，
换真模型、换 Mock 都不改上层。

`OpenAIChatModel` 内置**异常分类 + 指数退避重试**：`408/429/5xx` 与传输错误走重试
（`max_retries=2`，`backoff = min(8s, 0.5·2^n)` + 随机抖动；业务性 4xx 不重试、直接失败）。
循环看到的是"结果 or `ModelError`"，重试细节被挡在适配器里——这是"健壮性放边界"的一个实例。

```python
import asyncio
from agent import Message, OpenAIChatModel

async def main():
    model = OpenAIChatModel(
        base_url="https://ark.cn-beijing.volces.com/api/v3",  # 或任何 /v1 兼容端点
        model="deepseek-v4-flash-ga-260731",
        api_key="YOUR_KEY",
    )
    result = await model.complete([Message(role="user", content="用一句话介绍什么是强化学习")])
    print(result.content)
    print(f"prompt={result.usage.prompt_tokens}, completion={result.usage.completion_tokens}")
    await model.aclose()

asyncio.run(main())
```

离线试跑（不消耗额度）：

```python
from agent import Message, MockChatModel

async def demo():
    model = MockChatModel()                       # 回显 mock:<content>
    print((await model.complete([Message("user", "hi")])).content)  # -> mock:hi

asyncio.run(demo())
```

## 3. 本课做到 / 没做到

**做到**：消息、结果、用量三件套；一个可换的模型接口；一个真实适配器 + 一个离线假实现。
**没做到（留给后续）**：没有"循环"——`complete` 只调一次就返回；没有 act/工具；
没有把上一轮的回答拼回下一轮。这些是 ch02 agent-loop 的增量。

## 4. 关联八股（~10 题）

1. chat/completions 与 Responses API 是什么关系？completion 的 `messages` 长什么样？
2. `system` / `user` / `assistant` 三种 role 各负责什么？顺序、角色混淆会怎样？
3. token 是什么？`prompt_tokens` / `completion_tokens` 和`total` 怎么换算？为什么 agent 要记账？
4. `temperature` 语义？为什么 agent 类应用常用低温？`top_p` 与之的关系？
5. `max_tokens` / `n`/ `stop` 各自控制什么？设得太小会发生什么？
6. `finish_reason` 有哪些取值（stop / length / …）？agent 该根据它决定是否续轮？
7. 一次调用是"无状态"的：模型记不住上一轮。上下文为什么必须由调用方拼好？
8. HTTP 层要点：base_url 的 `/v1` 后缀、鉴权头（Bearer）、超时与重试。
9. 为什么 LLM 调用适合异步（async）？对后端并发 / SSE 有什么意义？
10. 发送非 2xx 时该抛什么、怎么把"模型错误"与"业务逻辑错误"区分开？
11. 哪些状态码值得重试？为什么退避要**指数且带随机抖动**（避免惊群）？LLM 调用是非幂等的——重试时要意识到会重复扣费/消费。
12. Agent 内部为什么要"顺序的 async"而不是到处并发？并发 run 应该在哪里注入？

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/models/`（产品）已把同一层做进包：`ChatModel` 抽象 + `MockChatModel`
+ `openai_compat.OpenAIChatModel`（httpx）与 `ark.ArkChatModel`（openai SDK / Responses）；
`evolving_harness/core/message.py` 定义了含 `tool_calls` 的 `Message`。读它们对比：
教学版为讲清协议而零依赖，产品版为多样性/健壮性而分包+多适配器。

## 6. 运行与验证

```bash
# 从仓库根跑本章测试（离线，不耗额度）
uv run pytest course/ch01_llm_call/ -q
uv run ruff check course/
```