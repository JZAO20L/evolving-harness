# DESIGN v0.1 · Agent 内核

> **状态**：`active`
> **范围**：M0–M1
> **对应代码**：`evolving_harness/agent.py`、`agents/*/agent.yaml`、`core/`、`models/`、`tools/`、`tracing/`

## 1. 目标

1. 一条行为可预测、可测试的单 agent 循环。
2. **一个 agent = 一个 YAML + 一个 Python**：YAML 放静态状态（prompt、guidance、skills、工具选择、限制），Python 放动态逻辑（加载、渲染、循环）。YAML 是 RSI 唯一可变面，Python 在实验中冻结。
3. 稳定的模型 / 工具 / trace 接口，后续 skills 演化与 serving 只依赖接口，不依赖实现。
4. 核心代码保持 nano：读者一个下午能读完并改得动。

## 2. 非目标

- 多 agent 编排、Agent 团队
- 模型训练（只保留数据导出的开口）
- 容器沙箱（v0.2）
- 公网服务（v0.2）

## 3. 约束

| 约束 | 含义 |
|---|---|
| 无框架依赖 | 不引入 LangChain / AutoGen / smolagents 作为运行时底座；只用 `httpx` |
| OpenAI 兼容协议 | 消息与 tool_calls 采用 OpenAI 格式，vLLM / DeepSeek / Qwen 可直接接 |
| 单向依赖 | `core.message` 是最底层；`models`、`tools`、`tracing` 依赖它；`agent.py` 依赖三者；`rsi.py` 只依赖 `agent.py` 的公开入口 |
| 状态与运行时分离 | YAML 可被 RSI 改写；Python、工具实现、benchmark 与 evaluator 不可被改写 |
| 一切都是有限资源 | 步数、输出、文件读取都有硬上限；后续版本再加时间和内存 |

## 4. 架构

```
CLI ──> AgentState.load(agent.yaml)
              │  system prompt + guidance + skill catalog
              ▼
         Agent ──> ChatModel (OpenAI-compatible / Mock)
              │
              ├──> ToolRegistry ──> Tool (fs: read/write/list, load_skill)
              │
              └──> TraceRecorder ──> traces/<run_id>.jsonl

rsi.py ──> Agent.run()  (same loop, different YAML)
```

一次 `run` 的状态只有 `messages: list[Message]`。**循环不持有隐式状态**，恢复、重放、导出都从
messages + trace 出发。`AgentState` 在 run 开始时加载一次，run 结束时重新计算 hash；不一致即
`status=drift`，这次测量作废。

## 5. 关键接口

```python
class ChatModel:
    async def complete(messages, tools=None) -> ModelResponse   # content + tool_calls + usage

class Tool:
    name, description, parameters   # JSON Schema
    async def run(**kwargs) -> str

class ToolRegistry:
    async def dispatch(ToolCall) -> Message                     # 永不抛异常，错误也是内容

class AgentState:
    @classmethod
    def load(path) -> AgentState                                # 校验 YAML，计算 version / hash
    def render_system_prompt() -> str                           # system + guidance + skill catalog
    def runtime() -> dict                                       # provider / id / base_url / temperature

class Agent:
    async def run(task, run_id=None) -> RunResult               # max_steps 硬上限；结束时检查 drift

class TraceRecorder:
    def event(kind, **data) -> None                             # 追加写 + 立即 flush
```

## 6. 决策与取舍

| 决策点 | 选择 | 放弃的方案 | 原因 |
|---|---|---|---|
| 消息表示 | dataclass + `to_openai()` | 全程裸 dict | 可类型检查、可测试，协议转换集中在一处 |
| 工具失败 | 作为 `tool` 消息回填 | 抛异常终止 run | 模型可据此自我修正；服务不因一次工具错误崩溃 |
| 模型失败 | 仅 429/5xx 重试，4xx 直接失败 | 全部重试 | 4xx 重试无意义，只会放大故障 |
| trace 存储 | JSONL 追加写 | M0 直接上 DB | 零依赖、可 grep、可 diff；v0.2 再入库 |
| 路径安全 | `resolve()` + `is_relative_to()` | 字符串前缀判断 | 前缀判断挡不住 `../` 与同前缀目录 |
| 循环终止 | `max_steps` 硬上限 | 仅依赖模型不再调用工具 | 防死循环与成本失控 |
| 多轮上下文 | 全量 messages（M0） | M0 就做摘要压缩 | 先保证可观测，压缩策略在 M1 用真实数据驱动 |
| subagent | 预留，M1 实现 | v0.1 做多 agent 编排 | 单 loop 是 nano 的前提 |
| agent 配置 | 一个 YAML + 一个 Python | 配置散落在 CLI / 代码常量 | 静态状态可见、可快照、可被 RSI 整文件替换；YAML 是唯一可变面 |
| state 身份 | 解析后 YAML 的规范 JSON SHA-256 | 文件字节 hash | 注释与格式不属于状态；语义变化才改变 version/hash 身份 |
| skills | YAML 内联 + 目录注入 + `load_skill` | 每个 skill 一个目录 / 全量注入 | 保持"一个 YAML"；目录按需加载，避免上下文随技能数线性膨胀 |
| 运行参数 | `max_steps` 等只在 YAML | CLI 覆盖关键参数 | 避免测量时 state 与实际行为不一致；临时实验可复制一份 YAML |

## 7. M1 设计

**7.0 Agent state（已实现）**
- `agents/<name>/agent.yaml` 是唯一状态：`version`、`model`、`limits`、`prompt.system`（冻结）、`prompt.guidance`（可变）、`tools.enabled`（冻结）、`skills`（可变）。
- `AgentState.load` 校验字段；解析后 YAML 的规范 JSON SHA-256 作为 state hash，写入 `run.start`。
- system prompt = `prompt.system` + `prompt.guidance` + skill catalog；skill 正文通过 `load_skill` 按需加载。
- run 结束时重读 YAML 计算 hash；不一致 → `status=drift`，这次测量不可用于对比。
- RSI 可变边界固定为 `prompt.guidance` 与 `skills`；其余字段在一次实验内不可改。

**7.1 上下文预算**
- 用 `len(text)/4` 粗估 token，记录每条消息的估算值。
- 预算 `max_context_tokens` 可配置；超过阈值触发压缩，而非等模型报错。

**7.2 压缩策略**
- 永远保留 `system` 与最近 K 轮（默认 6）。
- 中段工具结果替换为摘要：`[tool result truncated: <name>, <bytes>B, exit=<code>]` + 首尾各 500 字符。
- 压缩事件写 trace（`context.compact`），记录压缩前后 token 估算，便于后续调参。

**7.3 文件记忆**
- 位置：`<workspace>/.memory/notes.md`，与普通文件工具隔离，避免模型误删。
- 写入由模型通过 `remember` 工具触发（追加条目，带日期）；读取在每轮开始时注入摘要，不注入全文。
- 只在明确"这条信息跨轮有用"时写入，避免变成第二个上下文。
- 记忆写在 workspace，不写 `agent.yaml`；评测 run 默认关闭记忆，否则 target 自己会触发 state drift。

**7.4 subagent**
- 子任务用独立 `messages` 列表跑同一个 `Agent` 类的新实例（新的 run_id，继承同一 `AgentState` 与 runtime）。
- 只把最终文本作为一条 `tool` 结果返回父循环；子 trace 单独落盘，父 trace 记录 `subagent.start/end` 与子 run_id。
- 限制：深度 ≤ 2，子任务步数 ≤ 父任务剩余预算的一半。

## 8. Trace schema

版本号固定为 `v: 1`。每条事件：`{v, run_id, seq, ts, elapsed_ms, kind, data}`。

| kind | data |
|---|---|
| `run.start` | `task, model, tools, max_steps, state {name, version, hash, path}, runtime {provider, id, base_url, temperature, max_tokens}` |
| `model.request` | `step, message_count` |
| `model.response` | `step, duration_ms, content, tool_calls, usage, finish_reason` |
| `tool.call` | `step, tool_call_id, name, arguments` |
| `tool.result` | `step, tool_call_id, name, duration_ms, content, is_error` |
| `run.end` | `status: ok \| drift \| max_steps \| error`, `steps`, `usage` / `error`, `state_hash` |

M1 追加：`context.compact`、`memory.write`、`subagent.start`、`subagent.end`。

## 9. 验收标准

- 工具调用链路：模型发起调用 → 工具执行 → 结果回填 → 模型给出终答，trace 事件完整。
- 工具错误不中断 run，且 `is_error=true`。
- 超过 `max_steps` 抛 `MaxStepsExceeded`，trace 落 `status=max_steps`。
- 越界路径（`../`、绝对路径）被拒绝。
- 同一份 YAML 的 `version` / `hash` 在 `run.start` 中可查；run 期间 YAML 被改则 `status=drift`。
- `uv run pytest` 全绿。

## 10. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 适配器未对真实端点验证 | 协议细节（空 content、tool_choice）可能不兼容 | v0.1 收尾前接一次 vLLM 或云 API 做 smoke |
| token 估算粗糙 | 压缩过早或过晚 | M1 用 trace 中真实 usage 校准比例 |
| 无沙箱 | 不能安全执行 shell / 代码 | v0.2 接容器池；在此之前不提供代码执行工具 |
| JSONL trace 无索引 | 版本多后检索慢 | v0.2 入 SQLite，JSONL 保留为原始证据 |
