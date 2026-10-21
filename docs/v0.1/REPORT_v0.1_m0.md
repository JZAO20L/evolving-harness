# REPORT v0.1 · M0

> **状态**：`active`
> **日期**：2026-10-21
> **范围**：M0（单 loop、模型适配、文件工具、trace、CLI）
> **代码版本**：工作区未提交（首个提交后补 `sha`）

## 1. 结论

M0 达成验收：单 agent 循环闭环可用，工具失败可恢复，trace 可复现，路径边界有效。
当前实现适合作为 v0.1 的地基，但不具备真实模型端到端验证与并发/服务能力。

## 2. 证据

| 项 | 结果 | 来源 |
|---|---|---|
| 单元测试 | 10 passed | `uv run pytest -q` |
| 静态检查 | All checks passed | `uv run ruff check .` |
| 核心代码量 | 697 行（`evolving_harness/**/*.py`） | `wc -l` |
| mock 端到端 | `MOCK: write a note` + 4 事件 trace | `traces/176b5dc6df3a.jsonl` |

mock trace 事件序：

```
run.start → model.request → model.response → run.end(status=ok)
```

带工具调用的路径由测试覆盖（`tests/test_loop.py::test_loop_executes_tool_then_finishes`），
事件序为 `run.start → model.request → model.response → tool.call → tool.result → model.request → model.response → run.end`。

## 3. 交付内容

| 模块 | 文件 | 说明 |
|---|---|---|
| 消息类型 | `core/message.py` | `Message` / `ToolCall` 与 OpenAI 格式互转 |
| 循环 | `core/loop.py` | `AgentLoop`、`RunResult`、`MaxStepsExceeded` |
| 模型 | `models/base.py`、`openai_compat.py`、`mock.py` | 抽象、兼容适配器、离线模型 |
| 工具 | `tools/base.py`、`tools/fs.py` | 注册表 + 三个文件工具 |
| trace | `tracing/recorder.py` | JSONL 追加写 |
| CLI | `cli.py` | `eh run` |

## 4. 与设计的偏差

| 设计项 | 实际 | 原因 |
|---|---|---|
| 期望 M0 即验证真实端点 | 未验证 | 本机无可用 GPU 端点；已列为 T1.5 |
| 期望 M0 有 `context.compact` | 未实现 | 按 DESIGN 推迟到 M1 |
| 期望目录结构与文档同步 | 已同步 | 本次补齐 docs 与目录 README |

## 5. 问题与风险

| 风险 | 等级 | 说明 | 下一步 |
|---|---|---|---|
| 适配器未接真实端点 | 高 | tool_calls / 空 content / tool_choice 的协议差异未验证 | T1.5 |
| 无代码执行能力 | 中 | 当前不能做 shell 类任务 | v0.2 容器池 |
| 上下文无上限 | 中 | 长任务会撑爆上下文 | M1 T1.1/T1.2 |
| trace 无索引 | 低 | 运行多了只能 grep | v0.2 入 SQLite |
| `MockChatModel` 的状态在实例内 | 低 | 并发复用同一实例会串响应 | 仅测试使用，服务层每 run 新建 |

## 6. 下一步

1. T1.5 接真实端点做 smoke，锁定适配器行为。
2. T1.1/T1.2 上下文预算与压缩。
3. T1.3 文件记忆、T1.4 subagent。
4. 全部完成后关闭 v0.1：版本 README 置 `frozen`，打 tag `v0.1-closed`。
