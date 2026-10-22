# REPORT v0.1 · M0 + M0.5

> **状态**：`active`
> **日期**：2026-10-21
> **范围**：M0（单 loop、模型适配、文件工具、trace、CLI）+ M0.5（Agent state：一个 YAML + 一个 Python）
> **代码版本**：M0 = `b17a8ba`；M0.5 = 工作区未提交
> **对应设计**：[DESIGN_v0.1_agent_core.md](DESIGN_v0.1_agent_core.md) §4、§5、§7.0

## 1. 结论

**M0** 达成验收：单 agent 循环闭环可用，工具失败可恢复，trace 可复现，路径边界有效。

**M0.5** 把 agent 从"CLI 硬编码 prompt + 裸循环"改为**一个 YAML + 一个 Python**：
`agents/default/agent.yaml` 是唯一静态状态，`evolving_harness/agent.py` 是唯一动态逻辑。
`run.start` 记录 `state.version` / `state.hash` / `runtime`；run 结束重算 hash，变化即
`status=drift`。这为 RSI 提供了唯一可变面：优化器只改 YAML 的 `prompt.guidance` 与
`skills`，不改 Python。

当前验收通过，但真实模型端到端与上下文管理仍未完成。

## 2. 证据

### 2.1 M0（历史快照，`b17a8ba`）

| 项 | 结果 | 来源 |
|---|---|---|
| 单元测试 | 10 passed | `uv run pytest -q` |
| 静态检查 | All checks passed | `uv run ruff check .` |
| 核心代码量 | 697 行（`evolving_harness/**/*.py`） | `wc -l` |
| mock 端到端 | `MOCK: write a note` + 4 事件 trace | `traces/176b5dc6df3a.jsonl`（历史文件，后续清理） |

M0 的 trace 事件序为
`run.start → model.request → model.response → run.end(status=ok)`；带工具调用的路径由
`tests/test_loop.py::test_loop_executes_tool_then_finishes` 覆盖。

### 2.2 M0.5（当前工作区）

| 项 | 结果 | 来源 |
|---|---|---|
| 单元测试 | 18 passed（M0 的 10 个 + 8 个 state 测试） | `uv run pytest -q` |
| 静态检查 | All checks passed | `uv run ruff check .` |
| 包代码量 | 1030 行（`evolving_harness/**/*.py`） | `wc -l` |
| mock run | `[agent] default v1 76ef3bde` | `traces/db42674f0386.jsonl` |
| state hash | `76ef3bde…`（64 位 SHA-256） | trace `run.start.data.state.hash` |
| runtime 记录 | provider / id / base_url / temperature / max_tokens | trace `run.start.data.runtime` |
| drift | 测试内修改 YAML → `run.end.status=drift` | `tests/test_agent.py::test_state_drift_invalidates_the_run` |

## 3. 交付

### M0

| 模块 | 文件 | 说明 |
|---|---|---|
| 消息类型 | `core/message.py` | `Message` / `ToolCall` 与 OpenAI 格式互转 |
| 循环 | `core/loop.py` | `AgentLoop`、`RunResult`、`MaxStepsExceeded`（M0.5 已并入 `agent.py`） |
| 模型 | `models/base.py`、`openai_compat.py`、`mock.py` | 抽象、兼容适配器、离线模型 |
| 工具 | `tools/base.py`、`tools/fs.py` | 注册表 + 三个文件工具 |
| trace | `tracing/recorder.py` | JSONL 追加写 |
| CLI | `cli.py` | `eh run` |

### M0.5

| 文件 | 变化 |
|---|---|
| `agents/default/agent.yaml` | 新增：默认 agent 状态 |
| `agents/README.md` | 新增：状态字段与可变边界 |
| `evolving_harness/agent.py` | 新增：`AgentState`、`Agent`、`build_tools`、`create_model`、`run_agent` |
| `evolving_harness/cli.py` | 改为加载 YAML；删除硬编码 prompt 与 `--max-steps` |
| `evolving_harness/core/loop.py` | 删除，循环并入 `agent.py` |
| `tests/test_agent.py`、`tests/conftest.py` | 新增：state / hash / skill / drift 测试 |
| `pyproject.toml`、`uv.lock` | 新增 `pyyaml` |
| `evolving_harness/{rsi,sandbox,serving,store,ui,skills}/` | 删除空脚手架目录；未来模块改为单文件 |

## 4. 与设计的偏差

| 阶段 | 设计项 | 实际 | 原因 |
|---|---|---|---|
| M0 | 期望 M0 即验证真实端点 | 未验证 | 本机无可用 GPU 端点；列为 T1.5 |
| M0 | 期望 M0 有 `context.compact` | 未实现 | 按 DESIGN 推迟到 M1 |
| M0.5 | `max_steps` 由 CLI 覆盖 | 只认 YAML | CLI 覆盖会让 trace 的 hash 与实际行为不一致 |
| M0.5 | state hash 用文件字节 | 用解析后 YAML 的规范 JSON | 注释与格式不属于状态；语义变化才改变身份 |
| M0.5 | skills 独立目录 | 内联 YAML + `load_skill` | 保持"一个 YAML"；正文按需加载，避免上下文线性膨胀 |

## 5. 问题与风险

| 风险 | 等级 | 说明 | 下一步 |
|---|---|---|---|
| 适配器未接真实端点 | 高 | tool_calls / 空 content / tool_choice 的协议差异未验证 | T1.5 |
| 无代码执行能力 | 中 | 当前不能做 shell 类任务 | v0.2 容器池 |
| 上下文无上限 | 中 | 长任务会撑爆上下文 | M1 T1.1/T1.2 |
| skills 目录常驻 system prompt | 中 | 技能多时上下文仍会增长 | 按 token 预算裁剪目录；正文始终按需加载 |
| drift 检查是单进程语义 | 中 | 并发写同一 YAML 可能漏检 | RSI 控制器串行执行；服务化后按 run 复制 state |
| `model` / `limits` 仍在 YAML | 中 | 优化器理论上可以改 | RSI 控制器硬编码 allowlist，加载后校验 |
| trace 无索引 | 低 | 运行多了只能 grep | v0.2 入 SQLite |
| `MockChatModel` 的状态在实例内 | 低 | 并发复用同一实例会串响应 | 仅测试使用，服务层每 run 新建 |

## 6. 下一步

1. T1.5 接真实端点做 smoke，锁定适配器行为。
2. T1.1/T1.2 上下文预算与压缩。
3. T1.3 文件记忆、T1.4 subagent。
4. v0.3：benchmark + 分层 verifier + `rsi.py`（外环）。
5. 全部完成后关闭 v0.1：版本 README 置 `frozen`，打 tag `v0.1-closed`。
