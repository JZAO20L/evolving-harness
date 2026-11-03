# ch10 · skills-rsi（RSI 轨第 3 章）——把 skills 集当成演化单元

`rsi.py` = **ch09 拷贝 + 本课增量**（hero）；`agent.py` = **ch09 原样（冻结，被改进对象）**。

ch09 把「可改进配置」缩小到 prompt：guidance / task_frame / final_format 三个字段，规则或 LLM 往里面
追加一句话。但真实 harness 的策略不止一段 system prompt：**技能（Skill）** 是可复用、可维护、按需加载的
指令包，`AgentLoop` 用渐进披露（目录进 system、`load_skill` 才注入完整指令）控制上下文预算。

本课仍跑同一个 meta-agent 外环，只把 **react 的作用对象从 prompt 换成 skills 集**：

```
act   = rollout(current config, train) → 分数 + trace（新增 loaded_skills）
react = 定位「哪条 skill 缺失 / 误导」 → add / delete / rewrite 一次技能编辑
```

难点依旧是 **eval + credit assignment**：分数怎么打（沿用 ch09 的 Judge 后端）之后，本课要回答
**「失败该记到哪个技能的账上？」**——没加载任何技能却漏点 = `skill_missing`；加载了技能仍漏点或
输出禁词 = `skill_misleading`。归因落到具体技能名，后续编辑才是可执行的。

---

## 0. 流程图

```
        dataset/synth/{math,tooluse}/{train,val,test}.jsonl（~100 条合成数据，行 schema 同 dataset/rsi）
                                    │
                                    ▼
   ┌──────────────────── act：rollout(current, train) ────────────────────┐
   │  AgentConfig{skills} → SkillLibrary → AgentLoop                      │
   │      目录进 system ──load_skill──▶ 完整指令注入 context               │
   │  观测 = 分数 + trace（tool_calls + loaded_skills）                    │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      ▼
   ┌──────────────── react：skill 粒度 credit assignment ─────────────────┐
   │  RuleSkillLocator：                                                  │
   │    没加载技能 + 漏正向要点        → skill_missing                     │
   │    加载了技能 + 漏点/输出禁词      → skill_misleading(skill=某技能)    │
   │    空答/步数耗尽 → skip；满分 → ok；禁词且无技能 → format              │
   │                                                                       │
   │  RuleSkillProposer： add 覆盖技能 / rewrite 目标技能（幂等追加）        │
   │  SkillLLMImprover：  一次 chat 调用 = 定位 + 编辑（JSON）              │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      ▼
   ┌──────────────────────── 候选 config（新 hash） ──────────────────────┐
   │  轮级 Gate（train-only）→ promote / reject                            │
   │  每 val_every 轮 SegmentGate 对血线 → commit / rollback / hold        │
   │  rollback 回到 watermark.config（**连 skills 一起回退**）              │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      ▼
                       末尾 evaluate(test) 恰好一次 → RsiReport
```

ch09 的评估纪律原样保留：train 按轮改，val 低频守门，test 只碰一次。本课新增的是**可变面**与**归因粒度**。

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `Skill` / `SkillLibrary` | ch04 冻结件：`{name, description, instructions, priority}` + 渐进披露目录 |
| `load_skill` | AgentLoop 内建工具：按名加载完整指令；未知技能返回错误（错误当数据） |
| `RolloutResult.loaded_skills` | 从 trace 还原「本次 rollout 实际加载成功、且属于当前 config 的技能」 |
| `SkillEdit` / `apply_skill_edit` | 不可变技能编辑：`add`（同名替换）/ `delete` / `rewrite` |
| `RuleSkillLocator` | 规则归因：漏点/禁词 → `skill_missing` / `skill_misleading`（带目标技能名） |
| `RuleSkillProposer` | 规则编辑：缺失 → 新增覆盖技能；误导 → 改写目标技能（重复追加幂等） |
| `SkillLLMImprover` | 一次 chat 调用完成定位 + 编辑；解析失败/动作非法 → 保守不改 |
| `load_synth(domain, split)` | 读 `dataset/synth/{math,tooluse}/{split}.jsonl`，避开旧探针集饱和 |
| `Gate` / `SegmentGate` / `Checkpoint` | ch09 两级门禁原样复用（本课没有放松纪律） |

## 2. 实现的对象 + 用法

### 版本指纹必须覆盖「指令」，不能只覆盖名字
```python
def _skill_canonical(skill: Skill) -> dict:
    return {"name": skill.name, "description": skill.description,
            "instructions": skill.instructions, "priority": skill.priority}

class AgentConfig:
    def canonical(self):
        return {..., "skills": [_skill_canonical(s) for s in self.skills], ...}
    def hash(self):
        return _sha256(json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False))
```
ch09 的 canonical 只存 `[s.name for s in skills]`——改写指令后哈希不变，整个版本/门禁体系会把
「不同的策略」误认成「同一版本」。本课先修这个缝：**技能指令进入版本指纹**。

`VersionRegistry.rollback` 同步还原完整 skills；`base_tools` 只登记名字、无 handler，回滚时由调用方注入。

### trace 里要有「到底加载了什么」
```python
loaded = tuple(dict.fromkeys(
    tc.arguments.get("name", "")
    for tc in res.tool_calls
    if tc.name == "load_skill" and tc.arguments.get("name", "") in known
))
```
只记**加载成功且属于当前 config** 的技能。没有这条 trace，credit assignment 只能猜「模型可能用了技能」；
有了它，才能区分 `skill_missing`（没得用/没用）与 `skill_misleading`（用了但被带偏）。

### 归因：skill_missing vs skill_misleading
```python
missing   = [c for c in task.checks if not c.startswith("~") and not scorer.check_hit(c, r.answer)]
violated  = [c for c in task.checks if c.startswith("~") and not scorer.check_hit(c, r.answer)]

if violated and r.loaded_skills:   → skill_misleading(target=loaded_skills[0])
if violated and not loaded:        → format（没技能的格式锅，交给 prompt 面）
if missing and not loaded:         → skill_missing
if missing and loaded:             → skill_misleading(target=loaded_skills[0])
```
组件集合在 ch09 六个之外新增 `skill_missing` / `skill_misleading`；`Diagnosis.skill` 携带具体技能名，
使下一步编辑从「改个方向」变成「改哪一条」。

### 编辑：不可变 + 幂等
```python
edits = [
    SkillEdit("add", "skill-mth-1", "覆盖任务 mth-1 的要点", "必须逐字包含：42", priority=5),
    SkillEdit("rewrite", "skill-a", "", "回答必须逐字包含 42，且不写步骤。"),
    SkillEdit("delete", "skill-b"),
]
cand = apply_skill_edit(config, edits[0])    # 永远返回副本；原 config 不动
```
- `add` 同名 = 替换（upsert）；`delete`/`rewrite` 目标不存在 = no-op；
- 规则 proposer 追加覆盖指令前先检查是否已包含，重复应用同一归因返回原对象 → `hash` 不变 → 外环判 noop；
- 编辑非法（空名、add 缺 description/instructions）→ 原 config，绝不写脏技能。

### 模型驱动：一次生成定位 + 编辑
```python
improver = SkillLLMImprover(model)
diag, cand = await improver.improve(config, task, rollout_result)
# 期望 JSON：{component, finding, skill, action: add|delete|rewrite|none,
#             description, instructions}
```
与 ch09 `MergedLLMImprover` 同构（thought+act 合并），只是 act 从「改 prompt」变成「改技能库」。
`action=none`、JSON 解析失败、技能名不存在都走保守路径：`ok` + 原 config。

### 端到端
```python
from rsi import Rsi, TwoStageImprover, RuleSkillLocator, RuleSkillProposer, load_synth

train, val, test = (load_synth("math", s)[:6] for s in ("train", "val", "test"))
report = await Rsi(
    AgentConfig(name="base", prompt=PromptPack(guidance="先算准再答。")),
    model, train=train, val=val, test=test,
    improver=TwoStageImprover(RuleSkillLocator(), RuleSkillProposer()),
    storage_dir="runs/ch10", max_rounds=2, val_every=1,
).run()
```
行为：基线无技能 → 最差 task rollout → 定位 `skill_missing` → 新增覆盖技能 → 候选 train 提升 →
轮级 promote → 段级对血线 commit/rollback（回滚连 skills 一起退）→ test 一次。

## 3. 本课做到 / 没做到

**做到**：skills 成为一等演化单元（增/删/改写）；完整技能进入版本指纹与回滚；trace 记录 loaded_skills；
规则与 LLM 两套 skill 粒度定位/编辑；合成数据接入（math / tooluse）；两级门禁与 test-once 纪律原样保留；
L0 全 Mock 覆盖缺失/误导/禁词/add/delete/rewrite/幂等/回滚/端到端 promote 与 rollback。

**没做到（留给后续）**：记忆策略仍固定（ch11）；loop 参数/工具集/limits 仍不可变（ch12）；技能膨胀没有
显式惩罚（新增技能可能只对单个 task 有用，跨 task 泛化靠 val 段级门禁兜底，更系统的「技能复用率」留给
ch12/ch13）；规则 proposer 不主动 delete（delete 由 LLM 后端或显式 `SkillEdit` 触发）；评估器/定位器
本身还不可被改进（ch13 meta）。

## 4. 关联八股（~12 题）

1. **skill vs system prompt**：为什么把可复用指令做成技能而不是全塞 system？渐进披露省了什么、代价是什么？
2. **技能作为演化单元**：为什么「改技能」比「改 prompt」更接近真实 harness 的自改进？哪些东西没变？
3. **版本指纹**：canonical 只存技能名会出什么问题？为什么指令、description、priority 都要进哈希？
4. **loaded_skills 从哪来**：为什么必须记录「实际加载成功」而不是「config 里有」？未知技能加载失败怎么处理？
5. **credit assignment**：`skill_missing` 与 `skill_misleading` 的判据是什么？为什么先看 loaded_skills？
6. **禁词归因**：为什么加载了技能时的禁词归 `skill_misleading`，没加载时归 `format`？这个规则何时会错？
7. **不可变编辑**：`apply_skill_edit` 为什么总是返回副本？no-op 返回原对象对外环有什么意义？
8. **幂等追加**：同一失败重复出现时，为什么不能无限往技能指令里追加同一句话？
9. **add 的命名**：规则 proposer 用 `skill-<task-id>` 有什么好处与风险（一 task 一技能 → 技能膨胀）？
10. **技能膨胀 vs 泛化**：为什么 val 段级门禁能兜底一部分？还缺什么指标（复用率/长度/加载率）？
11. **LLM 定位+编辑合并**：一次生成相比两阶段少什么成本？可控性/可观测性损失怎么补（JSON 校验+保守 no-op）？
12. **为什么沿用两级门禁**：可变面从 prompt 换成 skills 后，轮级/段级的分工为什么不需要改？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py` 的 `Skill` / `SkillLibrary`：产品版的技能加载与渐进披露（本课 `agent.py` 冻结自课程版）。
- `evolving_harness/rsi.py`：仍是 seam；本课把「技能集」接入 propose 面，为产品版把不可变配置扩展成
  `{prompt, skills, memory, loop}` 提供最小样例。
- 合成数据：`dataset/synth_pipeline/README.md`（verifier-first、机械校验、30/10/10 切分、test-only-once）。
- ch09 的 Judge/Locator/System One 后端：本课直接复用；技能归因的模型后端就是 `SkillLLMImprover`。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch10_skills_rsi/ -q
#   → 完整技能哈希 / add-rewrite-delete 不可变 / missing-misleading-format-skip-ok 定位 /
#     规则新增与幂等改写 / LLM 改进 JSON 与保守 no-op / loaded_skills trace /
#     版本回滚还原技能 / synth 加载 / 端到端 promote+commit 与 rollback

# 数据集仍要可解析（synth 已生成）
uv run pytest dataset/ -q

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch10_skills_rsi/ -m smoke -s
#   → 真 deepseek 在 synth math 上跑「skill 定位 → 规则编辑 → 两级门禁 → test 一次」
#   → 真 deepseek 走 SkillLLMImprover 的 JSON 编辑路径（解析失败也允许保守不改）

uv run ruff check course/ch10_skills_rsi/
```

> 注：合成 math 题对 deepseek 可能 baseline 已高 → 第一轮 noop、段级 hold。这是**合法终态**：没有改进
> 空间就不改，不为「看起来动了」接受噪声。想稳定看到 add/promote/rollback，看 L0 的两个端到端用例。
