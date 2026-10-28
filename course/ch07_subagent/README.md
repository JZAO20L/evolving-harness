# ch07 · 子代理（sub-agent）

ch01~ch06 让 agent 能"调模型 / ReAct 循环 / 工具 / 技能渐进披露 / 分层记忆 / 规划"，但**一个循环只能
说完一件事**——碰到"子问题值得**单开一个隔离 run** 去解决"的场景还是不受支持：没有"把一个子问题
**委派给一个独立循环**去解决、父循环等它回来再继续"的能力。本课补上 **sub-agent**：子代理对主 Agent
的视角 = **一个有 Agent 能力的工具（`act:subagent`）**，父循环等它收官、只收回最终文本。

文件：`agent.py` = ch06 原样 + sub-agent 增量（本课 hero）：`subagent` 函数工具、`_run_subagent`
（深度护栏 → 预算减半 → 新 `AgentLoop` 实例独立 context → 顺序 await → 只回最终文本）。`skills/`
顺延自 ch06（自包含，非重点）。

---

## 0. 流程图

```
                父循环 run(user) 的某一步
                       │ （模型决定调用 subagent 工具）
                       ▼
        ┌───────────────────────────────────────────────┐
        │ _run_subagent(args)                              │
        │  ① 深度护栏：self.depth >= 2 → 返回错误当数据，不递归 │
        │  ② 子预算 child_max = max(1, 父剩余步数 // 2)      │
        │  ③ 新建 AgentLoop 实例（独立 messages / 独立注册表  │
        │      / 独立工作记忆；沿用同一 model/技能/system/     │
        │      工作工具/存储目录），depth+1                  │
        │  ④ await child.run(task) —— 顺序 await，等子收官     │
        │  ⑤ 子预算耗尽 → "（子代理步数耗尽…）"当数据           │
        │     否则 → 只回 "[子代理完成] <子最终文本>"          │
        └───────────────────────────────────────────────┘
                       │ 回填为一条 tool observation
                       ▼ （经父 _emit → abbreviate_observation 只留核心）
                父循环继续，引用其要点给结论
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `subagent(task)`（本课注册的函数工具） | 模型在循环里把一句话子任务**委派**给独立子循环 |
| `_run_subagent` | 一次委派：**深度护栏 → 预算减半 → 新实例独立 context → await → 只回最终文本** |
| `depth` / `max_subagent_depth=2` | **递归护栏**（DESIGN §7.4：深度 ≤ 2）；越界当数据返回错误 |
| `steps_used` / `child_max = 剩余 // 2` | **预算减半**（子步数 ≤ 父剩余预算的一半）；子耗尽当数据 |
| `base_tools` | 工作工具在父子间**共享同一工具能力**（`builtin_read_tools()` 等） |
| 复用 `AgentLoop` 全基础设施 | 工具 / 技能 / 记忆 / 路径安全原样继承，不重写 |

## 2. 实现的对象 + 用法

### `AgentLoop` 追加（向后兼容：新增参数都带默认值）
构造多两个可选参数：`base_tools`（工作工具列表）、`subagent_enabled`。还加了 `depth`（根=0）、
`max_subagent_depth=2`、`steps_used`（预算测算）。run 循环顶部同步 `self.steps_used = step`。
`subagent_enabled=True` 才注册 `subagent` 工具（不干扰既有用法/既有测试）。

### 委派一次子 run（`_run_subagent`）
`depth >= 2` → 返回 `错误：子代理深度已达上限 …`（**当数据**，不递归、不炸父循环）。
否则算 `child_max = max(1, (max_steps - steps_used) // 2)`，构造**新 `AgentLoop` 实例**：
- **独立 context**：`child.run(task)` 内部新建自己的 `[system, user]`，父历史不进子窗（隔离）；
- **独立注册表 / 独立工作记忆**：各建新实例（同 `base_tools` / 同 skills / 若父有 memory 则
  子用**同一存储目录**新建 `Memory`，记忆落盘仍共一份，解决与写不打架）；
- **共享配置**：同一 `model`、`skills`、`system`、`base_tools`（继承同一 State 配置与 runtime）。
顺序 `await child.run(task)` 等子收官；子预算耗尽 → `（子代理步数耗尽，未完成）…` 当数据。
成功则只回 `[子代理完成] <子最终文本>`——父 `_emit` 把它当 tool 观察挂进上下文，经
`abbreviate_observation` 截断，**只保留核心内容**。

## 3. 本课做到 / 没做到

**做到**：`act:subagent` 作为"有 Agent 能力的工具"被模型调用；**独立 context**（子窗不泄父历史、父
只收回一条最终文本，防内部工具过程泄露/防上下文膨胀）；**深度护栏**（≤2，越界当数据，防递归失控）；
**预算减半**（子步数 ≤ 父剩余/2，子耗尽当数据父继续）；**新实例隔离**（注册表 / 工作记忆各自独立，存储
目录共享）；**只回最终文本**（经 `abbreviate_observation` 只留核心）；**同一 model/技能/工作工具共享
**（新实例继承同一配置）；顺序 await（intra-turn 不并行）。L1 冒烟真 deepseek：父委派子代理调研并引用其要点给结论。
**没做到（留给后续）**：子代理**并行**跑多个（`asyncio.gather` 并发委派 = 并发在边界注入，后端轨
ch15 示范，本课单循环顺序 await）；子 trace 单独落盘 / 父 trace 记录 `subagent.start/end`+子 run_id
（产线运行态 §7.5，教学版最小化到"只回文本"）；子代理失败后的**自动重试/换法**；与 ch06 规划联动
（某个 todo 委派给子代理执行的接缝已留，未当期打通）；eval 隔离关子代理（防状态漂移）。

## 4. 关联八股（~12 题）

1. 为什么要子代理？"一个循环只能说完一件事"，哪些任务值得单开一个隔离 run（子问题深度、上下文隔离、可并行物）？
2. **子代理 = 有 Agent 能力的工具**：把它建模成函数工具 vs 独立方法/独立类，对编排、错误处理、可测试性各有什么差别？
3. **独立 context vs 共享配置**：为什么子窗不与父共享 messages，但又不重写 model/技能/工具（继承 vs 隔离的边界）？
4. **并行 vs 顺序**：为什么本课子代理用顺序 await（intra-turn 不并行）？真并行该放哪（边界注入 / `asyncio.gather`）？
5. **深度上限为何这么定**：递归失控的风险点？`depth ≤ 2` 是从哪来的（DESIGN §7.4），越界该怎么处理？
6. **预算减半为何**：为什么子步数要 `≤ 父剩余/2`？不设限会发生什么（无限套娃 / 父自己饿死）？
7. **只回最终文本**：为什么父只收回一句最终文本、不收回子内部工具过程？好处（防泄过程 / 防上下文膨胀）与代价（父做不了精细判断）？
8. **观察怎么"只留核心"**：子返回值经 `abbreviate_observation` 截断的意义？和 tool 结果进工作记忆的简写是同一套吗？
9. **错误当数据**：子"深度越界"和"预算耗尽"为什么都返回可读串而不是抛异常？父循环据此能做什么？
10. **新实例隔离**：为什么要新建注册表 / 工作记忆而非父子共用？共用一个 Memory 的 `messages` 引用会出什么问题（resync 互相踩）？
11. 与 ch06 规划衔接：某个 todo 要委派子代理、await 回来再标 done——接缝在哪、本课为何不当期打通？
12. 产线/评测：子 trace 单独落盘、eval 隔离关子代理的意义？token 成本翻倍怎么兜（预算减半是其一）？

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/agent.py` 的对应愿景见 `docs/v0.1/DESIGN_v0.1_agent_core.md` §7.4：子任务用独立
`messages` 列表跑同一个 `Agent` 类的新实例（新 run_id、继承同一 State 配置与 runtime）；只把最终文本
作为一条 `tool` 结果返回父循环；子 trace 单独落盘，父 trace 记录 `subagent.start/end` 与子 run_id；
限制深度 ≤ 2、子步数 ≤ 父剩余预算一半。教学版最小化为 **`act:subagent` 工具 + `_run_subagent`
（深度护栏/预算减半/新实例独立 context）+ 只回最终文本**几条缝；并行子代理（ch15）、子 trace 单独
落盘（运行态 §7.5）、eval 隔离关子代理留后端轨 / RSI 轨逐步补齐。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：委派/只回最终文本/隔离/预算减半/深度护栏/耗尽当数据/简写/共享配置 全走 Mock
uv run pytest course/ch07_subagent/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）：真 deepseek 委派子代理调研并引用要点给结论
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch07_subagent/ -m smoke -s
uv run ruff check course/
```