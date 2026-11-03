# ch08 · rl-loop（RSI 轨第 1 章）——无训练 RL 式自改进闭环

ch01~ch07 让 agent 会"调模型 / ReAct 循环 / 工具 / 技能 / 记忆 / 规划 / 子代理"，但**改进它本身**仍要靠
**人**重写 prompt。本课补上 **RSI 外环**：把 agent 当作一个**可洞察、可打分、只暴露给"改 prompt"的
被改进对象**，用一条类似 RL 的「reward → 定位问题(credit assignment) → propose → gate/版本」闭环做
**训练无关(self) 的自改进**——就像 RL 里的奖赏驱动策略更新，这里用 verifier 奖赏驱动 prompt 更新。

文件：`rsi.py` = ch08 新增（本课 hero：外环全部对象）；`agent.py` = ch07 快照**原样不动**（被改进对象，
`rsi.py` 只组合它的**输入**喂给它，镜像 RSI 轨"改 YAML/prompt、永不改 Python"）。

---

## 0. 流程图

```
             frozen task set（statement 公开给 agent / checks 私有给 verifier）
                                  │
                     ┌────────────▼────────────────────────┐
   repeat           │  Rsi.improve(max_rounds)             │
                     │  ① baseline：对 config 测每 task 分    │
                     │  ② locate：挑最差 task → Locator       │
                     │     把失败 credit-assign 到 prompt     │
                     │     组件(guidance/frame/format)        │
                     │  ③ propose：依 diagnosis 编辑对应       │
                     │     组件 → 候选 AgentConfig(新 hash)     │
                     │  ④ evaluate 候选（n_rollouts≥1，组内    │
                     │     多 rollout → 差分= PairScorer）      │
                     │  ⑤ gate：聚合升 + 不回归？→ promote      │
                     │     否则 reject；每步 VersionRegistry    │
                     │     登记 {hash, config, score, verdict}  │
                     │     可 rollback 到任一版                 │
                     └────────────▲────────────────────────┘
                                  │ max_rounds 到 / 不再提升
                        final_version + final_config 落报告
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `reward = verifier` | 不靠 learn reward signal，用 **Rule(确定性) / Judge(LLM) pointwise / 组内差分 = Pairwise** 给分 |
| `credit assignment = Locator` | 不只给总分，归因到**哪一环 / 哪个 prompt 组件**错，propose 才有的放矢 |
| `prompt 三件套 = PromptPack` | 回答"除 system 还能改什么"：`guidance`(system) + `task_frame`(user 包装模板) + `final_format`(输出形态) |
| `版本 = 可变段 SHA-256` | 任一 prompt 字段变 → hash 变 → **新版本**；`VersionRegistry` JSONL 登记 / 可回滚 |
| `Gate` | 聚合分提升**且不回归**其它 task 才 promote，否则 reject |
| 复用 `agent.py`(ch07) | `rollout()` 只组装 `AgentLoop(system=…)` + `run(render_user(statement))`，**不动 agent.py** |

## 2. 实现的对象 + 用法

### prompt 定义（`PromptPack`）——不只 system 一个口
`PromptPack{guidance, task_frame: TaskFrame{prefix, suffix, exemplars}, final_format}`。模型实际看到的是
`[render_system(), render_user(statement)]` 两段：`guidance` 进 system 前置指令；`task_frame` 把原始任务句
包装成 user 消息（前缀/后缀 + few-shot 示范）；`final_format` 追加到 system 尾的"输出要求"。改任一字段
→ `canonical()` 变 → `config.hash()` 变 → 新版本。**首次定义"RSI 能改的面"，ch09~12 往这加字段**。

### AgentConfig / rollout
`AgentConfig{name, prompt, max_steps, skills, base_tools}` 是被登记/回滚的可跑配置；`.hash()` = 可变段
canonical JSON 的 SHA-256。`rollout(config, task, model)` 构造 `AgentLoop` 跑一次 → `RolloutResult`（trace）。
关键：**只喂 `task.statement`**，`checks` 永不给模型 —— 这就是「评测污染隔离 / 防 self-drift」的反模式规避。

### reward（`Scorer`）
`RuleScorer`（pointwise 确定性：正 check 要含词、`~词`要不含，离线骨干）· `JudgeScorer`（LLM pointwise，
剥 `_parse_score`）· `PairScorer.diff(base, cand)`（组内差分 → `Differential{delta, preference, per_task}`，
`n_rollouts>1` 候选组取均值，差分更稳）。`evaluate()` 返每 task 的 `(mean, best, min)`，`aggregate()` 算整组。

### credit assignment（`Locator`）
`locate(task, rollout, score)` 规则式反推失败成因：空答/未收尾 → `skip`；禁词违约（`~`) 优先 → `format`
（硬伤：输出了不该出现的）；漏正面要点：全程没调工具 → `halluc`（疑似瞎编）；查了仍缺 → `frame`（包装/
示范不足）。归因到**组件**而非"表现不佳"，propose 才有的放矢。

### propose（`Proposer`）
`propose(config, task, rollout, diag)` 依 `diag.component` 给 `PromptPack` 对应字段加一条修正规则（`_RULE`），
`copy.deepcopy` 返回**新 config（不可变）**。确定性、离线可测；接口预留 `context`（ch09 可模型驱动 / 定位与
propose 合并 / 吃更多上下文）。

### gate / 版本（`Gate` / `VersionRegistry`）
`Gate.decide(baseline, cand)`：聚合升**且无 task 回退超 `regress_tol`** → `promote`，否则 `reject`。
`VersionRegistry(storage_dir)` 往 `versions.jsonl` 追加 `{version, config, score, verdict, parent}`；
`list/get/latest/rollback(version)→config` 支持回到任一历史版。

### 外环（`Rsi`）
`Rsi(config, model, tasks, *, scorer, storage_dir, n_rollouts).improve(max_rounds)`：baseline → 每轮
`propose_improvement(current)`（定位**当前**最差 task → 基于**当前**配置提议，不把已提升的又改回去）→
evaluate 候选 → gate → 登记 → 循环 / 落 `RsiReport{baseline, rounds[], final_version, final_config}`。

## 3. 本课做到 / 没做到

**做到**：把 agent 当"只改 prompt"的被改进对象（`agent.py` 一行不动）；prompt 定义成三件套并纳入版本哈希；
reward 支持 pointwise(规则/打分模型) 与组内差分 pairwise；`Locator` 把失败 credit-assign 到具体组件；
`Proposer` 确定性改对组件（不可变）；`Gate` 提升且不回归才 promote；`VersionRegistry` 登记 / 可回滚；评测
污染隔离（agent 只见 statement，verifier 见私有 checks）；Rsi 外环端到端。L1 冒烟：真 deepseek 把整轮闭环跑通、
至少登记一个版本、门禁给显式 verdict。
**没做到（留给后续）**：可变面只开了 prompt（ch09 prompt-RSI、ch10 skills、ch11 memory、ch12 agent 结构逐章
放开字段，环不重写）；`Proposer` 仍是确定性规则（ch09 换模型驱动 / 合并 locate+propose / 吃更多上下文）；
`JudgeScorer` 仅接口就位未作主导 verifier；只测单个 task 集，**对当前 task 集过拟合 / 泛化**未做（后续
benchmark 拆 train/val）；并发评估（边界注入，后端轨 ch16）。

## 4. 关联八股（~14 题）

1. **RL 式自改进 vs 训练 RL**：这里为什么叫"无训练 RL"？reward 从哪来（verifier 而非 learn signal）？
2. **reward = verifier**：为什么"打分模型/规则判分"能当奖赏？pointwise（单 rollout 打分）vs pairwise（组内
   多 rollout 差分比优劣）各自适合什么场景？
3. **credit assignment**：只看总分为什么不够？把低分**归因到某个组件**对下一步 propose 有什么意义？
4. **只改 prompt 不改代码**：为什么 RSI 轨把可变面限死在 prompt/YAML？改 Python 会破坏什么（不可回滚 /
   无法版本哈希 / 语义跳变）？
5. **prompt 不是 system 一个口**：prompt 至少能拆成哪几块（guidance / 任务包装 / 输出形态）？改输出形态 vs
   改方向各自应验哪种失败？
6. **版本 = 可变段哈希**：为什么版本用"可变段 canonical JSON 的 SHA-256"而不是 git commit？好处（可比较、
   幂等、快照化）与局限（不含代码/规模）？
7. **评测污染隔离 / self-drift**：评测 run 写 memory 为什么是反模式？把 checks 私有、只喂 statement 解决什么？
8. **门禁与回滚**：为什么"升分就收"不够，还要**不回归**约束？reject 后为什么要有版本回滚（保底）？
9. **不可变性**：propose 为什么返回新对象而非改原 config？（防同一候选被反复当作基线、保证版本追溯）
10. **规则式 propose 的边界**：确定性 edit 的好处（可测、可控、离线）与天花板（不会产出全新写法）？
11. **对 task 集过拟合**：RSI 在单个 eval 集上调 prompt，怎么防过度适配这个集？（拆 train/val、更多 task、
    门禁不回归是其一）
12. **组内多个 rollout / 差分**：为什么同一配置跑多次取均值/做差分更能反映真实表现？（压掉单次抽样的噪声）
13. **分层 verifier**：确定性规则 vs LLM judge vs 未来多 agent，逐层加码的取舍？
14. **复用 Agent 接缝**：为什么 RSI 电源源复用已完成的 agent 循环（可观察/可挂 hook）而非另写？
    "agent 是工具、RSI 是搬 prompt 的优化器"——这条分界如何保证外环不改内环？

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/rsi.py` 目前只有 **seam**：`Rsi` 类持一个 `Agent`，标注 `propose()` / `verify()` / `gate()`
三个设计面，无实现——本章是**洁净室实现**镜像其概念。设计约束见 `docs/v0.1/DESIGN_v0.1_agent_core.md`
§4/§6/§7：RSI 只能改 YAML/prompt（guidance/skills/memory），**永不改 Python**；版本 = 可变段 SHA-256 哈希；
门禁决定 promote/revert + 版本登记；评测 run 写 memory（self-drift）是反模式；agent 只见 statement、
verifier 只见私有 rubric（分层 verifier）；RSI 不能调低隔离档。产品把"改进点"横向做在 State 的可变段上，
教学版以 `AgentConfig`（只开 prompt 面）+ `VersionRegistry` 落盘逼近。ch09 起逐章放开 variable face 并对照本
章外环，环（reward→locate→propose→gate）不重写。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：prompt 渲染/hash/rule 判分/组内差分/locator 归因/proposer 不可变/gate/版本登记回滚/
#   Rsi improve 提升即受理、不提升即回滚 全走 Mock
uv run pytest course/ch08_rl_loop/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）：真 deepseek 把 RSI 外环整轮跑通、登记版本、门禁给 verdict
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch08_rl_loop/ -m smoke -s
uv run ruff check course/
```