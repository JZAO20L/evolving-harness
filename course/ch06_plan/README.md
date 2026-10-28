# ch06 · 规划（plan）

ch01~ch05 让 agent 能"调模型 / ReAct 循环 / 工具 / 技能渐进披露 / 分层记忆"，但处理**长目标**仍是
"一次 run 一条指令"——没有"把大目标拆成能执行完的子任务、跟踪进度、被中途改方向"的能力。本课补上
**规划**这一层：面对多阶段目标，agent 先拆**长期计划（Plan）**，再落到**子任务清单（TodoList）**
逐个执行并**监控进度**；当用户在过程中抛"改方向"的意图时，agent 要**识别出来**，并相应**调整长期
Plan 与后续 Todo**（用户两点要求，对齐 course/README ch06 的"拆任务→plan-then-execute→重规划"）。

文件：`agent.py` = ch05 原样 + 规划子系统（本课 hero）：`Todo / TodoList`、`PlanStep / Plan`、
`plan_task / decompose / revise_plan`、`is_plan_update` 规则门控、`PlanTracker`（JSONL 落盘）、
`PlanAgent`（单执行循环，注册 `todo_update` 驱动完成与监控）。顺延 ch02 关键设计：**agent 不是跑完才理人**
（`input_queue` 运行期用户输入注入，`AgentLoop` 与 `PlanAgent` 双循环每步边界均优先处理；见 ch02 §0.5）。
`skills/` 顺延自 ch05（自包含，非重点）。

---

## 0. 流程图

```
                    run(user)
                       │
        ┌──────────────┴─────────────────────────────────┐
        ▼                                               (有 Plan 且 is_plan_update)
 ① 无 Plan ──plan_task→Plan─decompose→TodoList            │
   （首次分解）    ↓ save plan.jsonl                       ▼
        │                                         ② 改方向 → revise_plan（模型低频重写）
        │                                            调整【长期 Plan 步骤 + 剩余 Todo】
        │                                            保已完成、只重排没做的；save
        ▼                                               │
  ③ 继续（有 Plan 且非更新意图 → 沿用，不重建）◄──────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 单执行循环 for step=1..max_steps:                            │
  │   memory.hook_pre_step;  refresh_progress(进度表注入 system) │
  │   complete(context, tools=schema)                          │
  │     ├ tool_calls ──► 执行（工作工具 / load_skill /           │
  │     │                    todo_update 更新状态 → 刷新进度）   │
  │     └ 无 ──► result.content 即最终答案                        │
  └────────────────────────────────────────────────────────────┘
        │
        ▼
   finialize: memory 落盘 + plan.jsonl(save)   ← 跨 run / 进程恢复
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `plan_task`（模型低频） | 长目标 task -> **长期 Plan**（goal + 稳定 step 阶段） |
| `decompose`（模型低频） | 长期 Plan -> **TodoList**（每 step 拆 1..n 个 todo，`ref_step` 归属阶段） |
| `TodoList.progress()/inject_table()` | **监控进度**：进度表随 step 注入 system，模型据此决策 |
| `todo_update`（本课注册的函数工具） | 模型调用标记 pending/in_progress/done/skipped，**驱动完成 + 监控** |
| `is_plan_update`（纯规则 cue 门控） | **识别"改 Plan"意图**：命中且已有 Plan 才付一次重规划 |
| `revise_plan`（模型低频重写） | 改方向 -> **新长期 Plan + 重生成的剩余 Todo**（保已完成） |
| `PlanTracker` | 持 Plan+Todo；`save/load` 至 **`plan.jsonl`**（JSONL，跨 run/进程恢复） |

## 2. 实现的对象 + 用法

### `Todo` / `TodoList`（子任务，可更替）
一个 todo = `id + summary + status(pending|in_progress|done|skipped) + ref_step`（归属长期 Plan 哪个
阶段）。`TodoList`：`add/get/mark/by_status/done/pending`；`progress() -> "3/5（60%）"` 是监控进度的
可读糖；`inject_table()` 渲染成进度表（含 todo id）——**进 system**，模型据此知道剩下什么。

### `Plan` / `PlanStep`（长期计划，稳定）
一个 step = `id + statement`，`Plan.goal` 概括目标。**长期 Plan 粒度粗、稳定**；Todo 粒度细、可被
`todo_update` 一条条改状态。`step_done` 标记某阶段收官。

### 模型分解（低频）＋ 规则容错解析
`_extract_json(text)` 从模型回复容错剥出第一个 JSON（容忍 ```json 包裹 / 前后夹文）——**解析校验放
规则侧**，模型只负责产出。`plan_task(task) -> Plan`、`decompose(plan) -> TodoList` 都是"模型给 JSON，
规则侧解析+校验"（坏输出 `ValueError`，`PlanAgent.run` 当数据返回，不炸）。

### 识别"改 Plan"意图（规则 cue 门控）＋ 重规划
`PLAN_UPDATE_CUES`（"不要/改为/先做/跳过/去掉/补充/顺序/调整…"）纯规则判 `is_plan_update(user_text)`。
**门控纪律（防误判）**：只有 `tracker.has_plan()` 且命中 cue 才分派 `revise_plan`；全新任务（无 Plan）
永远不会误走重规划。`revise_plan(model, old_plan, done_todos, user_text)` 喂"旧阶段状态 + 已完成 todo
+ 改动要求"，模型**重写长期 Plan 与剩余 TodoList**（保住已完成、只列还没做的）→ 返回 `(Plan, TodoList)`。

### `PlanTracker`：持 Plan+Todo、load/save JSONL、监控
`save()` 把 plan+todo 写 `plan.jsonl`（一行一个 JSON 对象，`kind` 标 meta/step/todo）；`load(dir)`
从文件恢复（**跨 run / 进程**），文件缺失给空 tracker。`progress()` 供外部/冒烟监控。

### `PlanAgent`（本课 hero 循环）
构造：懒加载 tracker（给 `storage_dir` 则 `PlanTracker.load` 恢复）；**总是注册 `todo_update`**；
给了 skills 再注册 `load_skill`。`run(user)` **三分支**（上文流程图）+ 单执行循环（镜像 AgentLoop
骨架，复用 `ToolRegistry.call`(错误当数据)、`load_skill` 注入缝、`Memory.hook_pre_step/finalize`、
`builtin_read_tools`、`SkillLibrary`）。每步 `_refresh_progress` 把最新进度表写进 context 的
`# 长期计划` 段（命中 marker 就地更新，不污染历史）；模型调用 `todo_update` 即更新状态并反映到进度。

## 3. 本课做到 / 没做到

**做到**：长期 Plan + TodoList 双层分解（`ref_step` 耦合）；`todo_update` 工具**由模型驱动完成与监控**
（单循环、系统每步注入进度表）；`is_plan_update` 纯规则门控 + **防误判纪律**（无 Plan 绝不重规划）；
`revise_plan` 重写长期 Plan 与剩余 Todo、**保已完成**；`PlanTracker` save/load 到 `plan.jsonl`
（跨 run/进程恢复）；`_extract_json` 容错把解析校验放规则侧；L1 冒烟真 deepseek：把 3 步拆成 10 个
todo、全用 `todo_update` 推到 10/10，改方向后长期 Plan 步骤被重排、剩余 todo 重生成。
**没做到（留给后续）**：Todo 间的**依赖/拓扑排序**（当前按列表顺序执行）；失败重规划的**自动触发**
（现在只有用户显式改方向才重规划，ch07 子代理 / 后续章可做"某 todo 卡住自动 replan"）；真意图理解
（目前 cue 门控是词表近似，VLLM/语义分类留给产品）；评测隔离关 plan（防状态漂移，对齐 DESIGN）；
`plan.jsonl` 与 `memory.jsonl`（ch05）分目录共同持久化的统一治理。

## 4. 关联八股（~14 题）

1. 为什么要"先 Plan 再 Todo"？直接让模型一口气做完，差在哪（长任务、窗口、可中断点）？
2. Plan（长期、粗、稳定）vs Todo（短期、细、可更替）粒度怎么取舍？拆得太细/太粗各有什么代价？
3. single-loop + `todo_update` 工具 vs 外层挂内层 loop：各自的可测试性、上下文清晰度、步数记账差异？
4. 为什么完成判定由**模型显式调用 `todo_update`** 而不是"run 结束就算 done"？谁来保证完成与输出一致？
5. 意图识别为什么用**纯规则 cue 门控**而非每次上模型？真假阳的代价：漏识别 vs 白付一次重规划。
6. **防误判门控**为什么"无 Plan 绝不重规划"？一条"先做 X"可能同时是首任务与改方向，怎么消歧？
7. `revise_plan` 怎么"保已完成、只重排剩余"？把已完成 todo 喂给它，模型据此不重复列——可靠性边界？
8. 为什么"解析校验放规则侧、模型只给 JSON"？`_extract_json` 容忍 ```json/前后夹文的意义是什么？
9. 进度表注入 system（`_refresh_progress` 命中 marker 就地更新）为什么好过"每步 append 新进度消息"？
10. `plan.jsonl` 与 ch05 `memory.jsonl` 分工：一个是"本 task 的执行蓝图"，一个是"历史事实"，为什么分开？
11. 改方向的语义模型：用户一条消息既可能是"继续执行"也可能是"重规划"——三分支靠什么判别/兜底？
12. 重规划的**丢保真**：模型重写 Plan 时会不会丢掉已完成步骤的细节？已完成成果怎么不丢（记在 todo/memory）？
13. token/步数兜底：`max_steps` 耗尽时 PlanAgent 怎么收尾？部分完成的 todo 留在 plan.jsonl 供下轮续跑？
14. 与 ch05 记忆、ch07 子代理衔接：Plan 需要记忆续跑、Todo 可能需要子代理执行——接缝在哪？

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/agent.py` 的规划愿景对应生产 `docs/v0.1/DESIGN_v0.1_agent_core.md` §7.4
（子任务 / 独立的 run）与记忆 §7.3 的"评测关 memory 防 drift"。产品版把规划做成 **State 动态段 +
trace 里的 plan/todo 事件** + 真意图理解；教学版最小化为**三分支决策 + todo_update 工具 + JSONL 落盘**
几条缝。依赖拓扑、失败自动 replan、评测隔离关 plan（ch07 子代理 / RSI 轨 / 后续章会逐步补齐）。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：Todo/Plan/Tracker/JSON 容错/意图门控/重规划/单循环组合 全走 Mock
uv run pytest course/ch06_plan/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）：真 deepseek 拆多目标任务 + todo_update 监控 + 改方向重规划
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch06_plan/ -m smoke -s
uv run ruff check course/
```