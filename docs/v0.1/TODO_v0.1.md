# TODO v0.1

> **状态**：`active`
> **对应设计**：[DESIGN_v0.1_agent_core.md](DESIGN_v0.1_agent_core.md)
> **规则**：每轮以本文件为驱动清单；完成项保留并勾选，未完成项写清阻塞原因。

## 0. M0（已完成，2026-10-21）

- [x] **T0.1** 消息与工具调用类型，OpenAI 格式双向转换
  - 验收：roundtrip 测试；坏 JSON 容错
- [x] **T0.2** `ChatModel` 抽象 + OpenAI 兼容适配器（429/5xx 重试）+ Mock 模型
  - 验收：Mock 可脚本化驱动多轮；适配器未接真实端点（见 REPORT 风险）
- [x] **T0.3** Tool / ToolRegistry，工具错误作为数据回填
  - 验收：未知工具、参数错误、文件不存在均返回 `ERROR: ...` 且 run 不中断
- [x] **T0.4** 文件工具 `read_file` / `write_file` / `list_dir`，路径限定 workspace
  - 验收：`../` 与绝对路径均被拒绝
- [x] **T0.5** JSONL trace + `AgentLoop.run` + CLI `eh run`
  - 验收：mock run 产出 4 个事件；`max_steps` 落 `status=max_steps`

## 1. M1（进行中）

- [x] **T1.0** Agent state：一个 YAML + 一个 Python（2026-10-21）
  - 内容：`agents/<name>/agent.yaml` 存 prompt / guidance / skills / tools / limits；`agent.py` 负责校验、渲染、循环；`load_skill` 按需加载 skill 正文
  - 验收：YAML 字段校验；`run.start` 记录 `state.version/hash/runtime`；run 期间改 YAML → `status=drift`；无技能时自动省略 `load_skill`
  - 证据：`tests/test_agent.py`（18 passed）；mock run trace `state.hash=76ef3bde…`

- [ ] **T1.1** 上下文预算与 token 估算
  - 内容：每步估算 messages token；`max_context_tokens` 可配置；超阈值抛/触发压缩
  - 验收：构造超长工具结果时不触发模型 4xx；trace 记录估算值
  - 依据：DESIGN §7.1

- [ ] **T1.2** 压缩策略
  - 内容：保留 system + 最近 K 轮；中段工具结果替换为摘要 + 首尾窗口
  - 验收：压缩后 token 估算下降 ≥ 50%；最终答案不因压缩丢失关键工具输出
  - 依据：DESIGN §7.2

- [ ] **T1.3** 文件记忆
  - 内容：`<workspace>/.memory/notes.md`，`remember` 工具追加条目；每轮注入摘要
  - 验收：跨轮任务能引用前几轮写入的记忆；记忆文件不可被普通写文件工具覆盖
  - 依据：DESIGN §7.3

- [ ] **T1.4** subagent
  - 内容：独立 messages 的子循环，结果作为一条 tool 结果回填；父子 trace 关联
  - 验收：父 trace 有 `subagent.start/end` 与子 run_id；子任务失败不中断父任务
  - 依据：DESIGN §7.4

- [ ] **T1.5** 真实端点 smoke
  - 内容：接一次 vLLM 或云 OpenAI 兼容端点，跑工具调用任务
  - 验收：COOKBOOK 增加实测命令与输出；REPORT 记录端点、模型名、run_id
  - 依据：DESIGN §10 风险 1

## 2. 优先级

1. T1.5（先坐实端到端，避免适配器问题拖到后面）
2. T1.1 → T1.2（压缩依赖估算）
3. T1.3
4. T1.4

T1.0 已完成，作为 M1 的前置：后续压缩、记忆、subagent 都继承同一 `AgentState` 与 runtime。

## 3. 阻塞项

| 项 | 阻塞原因 | 解除条件 |
|---|---|---|
| 真实端点验证 | 本机暂无可用 GPU 端点 | 提供 `EH_BASE_URL` / `EH_API_KEY` 或可访问的 vLLM |
| 沙箱相关（v0.2） | 本机未安装 docker | 部署机可用 docker |
