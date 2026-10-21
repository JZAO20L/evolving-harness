# v0.1 · Agent 内核

> **状态**：`active`
> **范围**：M0（循环 / 模型适配 / trace）、M1（记忆 / 上下文压缩 / subagent）
> **开始**：2026-10-21

## 目标

把 agent 的"当下"做扎实：一条可读、可测、可扩展的循环，加一层能承载长任务的上下文与记忆。
这一版不碰服务化、沙箱与 RSI，但所有接口都为它们留好位置。

## 里程碑

- [x] **M0** 单 loop + 模型适配 + 文件工具 + trace + CLI（2026-10-21）
- [ ] **M1** 上下文预算与压缩 + 文件记忆 + subagent

## 交付物

| 内容 | 位置 |
|---|---|
| `AgentLoop`、消息与工具调用类型 | `evolving_harness/core/` |
| OpenAI 兼容适配器、Mock 模型 | `evolving_harness/models/` |
| Tool / ToolRegistry / 文件工具 | `evolving_harness/tools/` |
| JSONL trace | `evolving_harness/tracing/` |
| CLI `eh run` | `evolving_harness/cli.py` |

## 文档索引

| 文档 | 内容 |
|---|---|
| [DESIGN_v0.1_agent_core.md](DESIGN_v0.1_agent_core.md) | 内核设计、接口、取舍、M1 方案 |
| [TODO_v0.1.md](TODO_v0.1.md) | M0 完成情况与 M1 任务清单 |
| [COOKBOOK_v0.1.md](COOKBOOK_v0.1.md) | 安装、测试、mock / 真实端点运行、trace 查看 |
| [REPORT_v0.1_m0.md](REPORT_v0.1_m0.md) | M0 结果、证据、偏差与风险 |

## 验收

- `uv run pytest` 全绿
- `uv run eh run "..." --mock` 产出 trace，事件序完整
- 路径越界被拒绝，且作为工具错误回填而非使进程崩溃
