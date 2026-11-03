# ch13 · meta-rsi（RSI 轨第 6 章）——把 RSI 过程本身当成被改进对象

`rsi.py` = **ch12 拷贝 + 本课增量**（hero）；`agent.py` = **ch12 原样（冻结，被改进对象）**。

ch09–ch12 改进的都是 **agent**：prompt → skills → memory → 结构。但「改进 agent 的那套流程」一直写死：

```python
judge      = RuleJudge()                 # 评估器固定
improver   = TwoStageImprover(...)       # 定位器 + proposer 固定
gate       = Gate(min_delta=0.0)         # 门禁阈值固定
n_rollouts = 1                           # 每任务只跑一次
```

当规则 proposer 遇到它**改不动**的失败（例如纯 prompt/格式问题），内层 RSI 会一直 `noop`。
本课把过程抽成 `MetaPolicy`，让外层观察内层报告并**改过程配置**：

```
act   = run_inner(meta_policy) → 内层 RSI 报告 + meta-val 分数 + 过程成本
react = 观察 promote / commit / rollback / gain → 改一处过程配置（改进器 / 评估器 / 门禁 / rollouts）
meta-gate = 候选过程配置在 meta-val 上严格更好才 promote
```

关键是把「**评估与定位本身也是可改进组件**」落到代码：`judge`、`improver`、门禁阈值、rollout 次数
都进 `MetaPolicy`，与 `AgentConfig` 一样需要哈希、不可变更新与夹取。

---

## 0. 流程图

```
        train / val / meta_val / test（四层数据，职责分离）
                              │
                              ▼
   ┌──────────── act：run_inner(MetaPolicy) ───────────────────────────────┐
   │  内层 Rsi(config, train, val, test=[])                                 │
   │    judge      = RuleJudge | LLMJudge                                   │
   │    improver   = 规则结构改进器 | MetaLLMImprover（复合改进器）           │
   │    gate       = Gate(min_delta=policy.gate_min_delta)                  │
   │    segment    = SegmentGate(val_min_delta=policy.val_min_delta)        │
   │    rollout    = policy.n_rollouts 次/任务                              │
   │  test 永远是 []：内层只拿 train + val，元层也只拿 meta_val 选过程        │
   └──────────────────────────────┬────────────────────────────────────────┘
                                  ▼
   ┌──────── react：过程级 credit assignment ───────────────────────────────┐
   │  InnerOutcome：promotes / commits / rollbacks / gain / cost           │
   │  RuleMetaLocator：                                                    │
   │    gain > 0 且 commit            → ok                                 │
   │    rollback > 0 且 gain ≤ 0      → gate_too_permissive                │
   │    一次都没 promote              → process_stuck                      │
   │    promote > 0 但没 commit       → segment_blocked                    │
   │    其它                          → no_gain                            │
   │                                                                       │
   │  RuleMetaProposer（目标档位、重复诊断幂等）：                          │
   │    process_stuck / no_gain → improver=llm → judge=llm → n_rollouts=2  │
   │    gate_too_permissive     → val_min_delta=0.05 → gate_min_delta=0.05 │
   │    segment_blocked         → val_min_delta=0.0 → n_rollouts=2         │
   └──────────────────────────────┬────────────────────────────────────────┘
                                  ▼
   _clamp_meta（judge/improver 非法退回 rule；阈值 0–0.5；rollouts 1–3）
                                  ▼
   MetaGate（meta-val 严格提升才 promote）→ 更新 current_policy
                                  ▼
   循环 max_rounds 次；最后用「最终 agent 配置 + 最终过程评估器」评 test 恰好一次
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `MetaPolicy` | 过程配置：`judge` / `improver` / `gate_min_delta` / `val_min_delta` / `n_rollouts`，可哈希、可夹取 |
| `update_meta_policy` | 不可变更新；非法字段丢弃、无变化返回原对象（哈希 noop） |
| `ProcessFactory` | 按 policy 装配一次内层 RSI 的 `judge` 与 `improver` |
| `InnerOutcome` | 一次内层运行的完整观测：报告、meta-val 基线/最终分、成本、promote/commit/rollback |
| `RuleMetaLocator` | 过程级 credit assignment：`ok / gate_too_permissive / process_stuck / segment_blocked / no_gain` |
| `RuleMetaProposer` | 目标档位式过程编辑；同一诊断重复应用幂等 |
| `MetaGate` | 元门禁：只用 `meta_val` 决定过程配置是否更好，不碰 test |
| `MetaLLMImprover` | 复合改进器：一次 chat 调用同时给 prompt patch + structure patch |
| `MetaRsi` | 二阶循环：四层数据纪律、内层 test=[]、末尾 test 一次、`meta_evals` 计数 |

## 2. 实现的对象 + 用法

### 过程配置也是版本化配置

```python
@dataclass(frozen=True)
class MetaPolicy:
    judge: str = "rule"            # rule = RuleJudge；llm = LLMJudge
    improver: str = "rule"         # rule = 规则结构改进器；llm = MetaLLMImprover
    gate_min_delta: float = 0.0    # 轮级门禁：train 提升至少多少才 promote
    val_min_delta: float = 0.0     # 段级门禁：val 提升至少多少才 commit
    n_rollouts: int = 1            # 每个任务跑几次

    def hash(self): ...            # 过程变化必须改变指纹
    def cost(self):                # LLM 组件越多、rollout 越多，成本越高
        return int(self.judge != "rule") + int(self.improver != "rule") + (self.n_rollouts - 1)
```

`AgentConfig` 和 `MetaPolicy` 是两层不同的「被改进对象」：前者决定 agent 怎么工作，后者决定
**RSI 怎么工作**。两层都需要哈希，因为元层也要做 promote/回滚与 noop 判断。

### 过程编辑永远返回新对象

```python
def update_meta_policy(config, **changes):
    allowed = set(MetaPolicy.__dataclass_fields__)
    clean = {k: v for k, v in changes.items() if k in allowed}
    if not clean:
        return config
    cand = _clamp_meta(replace(config, **clean))
    return config if cand == config else cand
```

- 未知后端（`judge="wat"`）退回 `"rule"`；
- 阈值夹到 `0.0–0.5`，`n_rollouts` 夹到 `1–3`；
- 同一改动应用两次 → 返回同一个对象 → 元环判 `noop`。

### 过程归因：先看「内层到底卡在哪」

```python
if gain > 0 and commits > 0:          → ok
if rollbacks > 0 and gain <= 0:       → gate_too_permissive
if promotes == 0:                     → process_stuck
if promotes > 0 and commits == 0:     → segment_blocked
else:                                 → no_gain
```

`process_stuck` 是最重要的诊断：内层一次都没 `promote`，说明当前规则改进器找不到可改点，
**不是 agent 不行，而是改进器不够强**。这正是元层要换 `improver` 的信号。

### 过程编辑：目标档位而不是每轮 +1

```python
# process_stuck / no_gain
improver != "llm"        → improver="llm"
judge != "llm"           → judge="llm"
n_rollouts < 2           → n_rollouts=2

# gate_too_permissive（坏候选被放过）
val_min_delta < 0.05     → val_min_delta=0.05
gate_min_delta < 0.05    → gate_min_delta=0.05

# segment_blocked（好候选被 val 卡住）
val_min_delta > 0        → val_min_delta=0.0
n_rollouts < 2           → n_rollouts=2
```

同一诊断第二次到来时条件已满足 → 返回原 policy → 元环 `noop` 并停止，不会无限加码。

### 复合改进器：一次调用跨面改写

```python
diag, cand = await MetaLLMImprover(model, guard).improve(config, task, rollout_result)
# JSON: {component, finding,
#        prompt:    {guidance?, final_format?, task_frame?},
#        structure: {max_steps?, tool_names?, subagent_enabled?, max_subagent_depth?}}
```

ch09–ch12 的模型驱动一次只改一个面；元层需要能**跨面**改动，否则过程搜索会被单面 proposer 卡住。
`_apply_meta_patch` 先把 prompt 字段合并进 `PromptPack`，再把 structure patch 交给
`update_structure` 统一夹取；解析失败或空 patch → `ok` + 原 config。

### 四层数据纪律

| 数据 | 谁用 | 用途 |
|---|---|---|
| `train` | 内层 RSI | rollout / 轮级 Gate / 结构搜索 probe |
| `val` | 内层 RSI | 每 `val_every` 轮的 SegmentGate（commit / rollback / hold） |
| `meta_val` | 元层 | 比较不同 `MetaPolicy` 产出的 agent 优劣，决定过程配置是否 promote |
| `test` | 只在最末尾 | 用最终 agent 配置 + 最终过程评估器评一次 |

实现上 `MetaRsi._run_inner` 永远构造 `Rsi(..., test=[])`：

```python
rsi = Rsi(self.config, self.model, train=self.train, val=self.val, test=[],
          judge=judge, improver=self.factory.improver(policy),
          gate=Gate(min_delta=policy.gate_min_delta),
          segment_gate=SegmentGate(val_min_delta=policy.val_min_delta),
          storage_dir=storage, n_rollouts=policy.n_rollouts, ...)
report = await rsi.run()
base_meta  = aggregate(await evaluate(self.meta_val, self.config, ...))
final_meta = aggregate(await evaluate(self.meta_val, report.final_config, ...))
```

`meta_val` 默认等于 `val` 方便小数据；生产里应单独切分，否则「选过程」和「内层门禁」共享同一批题，
选择偏差会被低估。

### 元门禁只看 meta-val

```python
class MetaGate:
    def decide(self, base, cand):
        delta = cand.meta_score - base.meta_score
        if delta > self.min_gain and cand.meta_score >= base.meta_score - self.regress_tol:
            return "promote", delta
        return "reject", delta
```

与内层 `Gate` 同构，但比较对象从「agent 配置」换成了「过程配置」。test 不参与任何选择。

### 评测次数是可解释的

```python
self.meta_evals += 2 * len(self.meta_val) * policy.n_rollouts   # 基线 + 最终
self.meta_evals += len(self.test) * current_policy.n_rollouts   # 末尾一次
```

L0 集成用例断言：两次内层运行（base + llm）各做 baseline/final 两次 meta-val，末尾 test 一次，
即 `meta_evals == 2 * (2 * len(meta_val)) + len(test)`；同时 test 题面在整轮中只出现一次。

### 端到端（L0 确定性驱动）

`PromptAwareModel` 看到「只输出数字」才答 `42`，否则输出 `步骤：42`：

1. 基线 `MetaPolicy()` 的规则改进器只会改结构，格式问题改不动 → 内层 0 promote；
2. `RuleMetaLocator` 判 `process_stuck`；
3. `RuleMetaProposer` 把 `improver` 换成 `"llm"`；
4. `MetaLLMImprover` 返回 `prompt.final_format = "只输出数字，不要写步骤"`；
5. 内层这次 promote + commit，`meta_val` 提升 → `MetaGate` promote；
6. 末尾 test 评一次 = 1.0。

这条链路证明了元层的价值：**只改 agent 改不动时，改「改进 agent 的方式」**。

## 3. 本课做到 / 没做到

**做到**：`MetaPolicy` 过程配置（哈希 / 夹取 / 不可变更新）；`ProcessFactory` 按策略装配；
`InnerOutcome` 过程观测；规则过程定位与目标档位编辑；`MetaGate` 只用 meta-val；
`MetaLLMImprover` 跨面复合改写；`MetaRsi` 四层数据纪律、内层 test=[]、末尾 test 一次、
`meta_evals` 可计数；L0 全 Mock 的确定性「换 proposer 解锁」端到端。

**没做到（留给后续）**：过程搜索仍是单点邻居，没有组合过程配置或历史学习；`MetaLLMImprover`
只改 prompt + structure，不生成新的 locator/gate 逻辑；过程成本只用粗粒度 `cost()` 代理，
没有真实 token/时延预算；`meta_val` 选择偏差与多重比较未校正；没有把过程配置的版本历史
持久化成可回滚的 registry；跨 run 的过程复用与缓存未实现。

## 4. 关联八股（~12 题）

1. **为什么改进 RSI 过程本身**？只改进 agent 会遇到什么天花板？
2. **`MetaPolicy` 与 `AgentConfig` 的分层**：为什么两层都要哈希、都不可变？
3. **四层数据纪律**：train / val / meta_val / test 各自防住了什么泄漏？
4. **test 为什么必须只碰一次**？内层 RSI 的 `test=[]` 在代码里怎样保证？
5. **`process_stuck` 的判据**：为什么「一次都没 promote」比「分数低」更能说明改进器有问题？
6. **`gate_too_permissive` vs `segment_blocked`**：两种门禁失败为什么需要不同的修复方向？
7. **目标档位式编辑**：为什么不是每轮把阈值 +0.01？重复诊断怎样保证幂等？
8. **元门禁与内层门禁**：两者结构相同，比较对象和防泄漏职责有什么不同？
9. **复合改进器**：一次调用同时改 prompt + structure 相比单面改进器解决了什么？风险是什么？
10. **过程成本**：`cost()` 为什么把 judge/improver/n_rollouts 粗粒度相加？真实系统该怎样计量？
11. **`meta_val` 默认等于 `val`**：方便了什么、又引入了什么选择偏差？生产该如何切分？
12. **学到的东西如何持久化**：最终产出是 agent 配置还是过程配置？两者怎样一起版本化与回滚？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py` 的 `AgentLoop`：本课 `agent.py` 冻结自课程版，仍是被改进对象。
- `evolving_harness/rsi.py`：生产 seam；本课展示了「评估器 / 定位器 / 门禁本身也是可搜索组件」。
- ch09：prompt 演化；ch10：skills；ch11：memory；ch12：结构；ch13 把前三者的**流程**也纳入演化。
- ARK System One 决策通道：`SystemOneJudge / SystemOneLocator`（只判断、不生成），可作为过程评估器的
  决策式实现，替换 `judge` / 定位后端。
- 合成数据：`dataset/synth_pipeline/README.md`（math/tooluse，30/10/10，test-only-once）。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch13_meta_rsi/ -q
#   → MetaPolicy 哈希/夹取/不可变/from_canonical
#   → ProcessFactory 装配 rule/llm 后端
#   → MetaLLMImprover 跨面 patch 合并 + 非 JSON/空 patch noop
#   → 过程定位五类组件 / 目标档位幂等 / 元门禁严格提升
#   → fake runner：process_stuck → 换 LLM improver → meta promote
#   → 确定性端到端：规则 proposer 卡住 → 元层换 llm → prompt 被修 → test 1.0

uv run ruff check course/ch13_meta_rsi/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch13_meta_rsi/ -m smoke -s
#   → 真 deepseek 跑 MetaRsi：内层 RSI 在 vault 任务上把 max_steps 从 1 修到 12，
#     meta-val 基线已 1.0 → 元环判 ok/noop 是合法终态，末尾 test = 1.0
#   → 真 deepseek 走 MetaLLMImprover 的复合 JSON 改写路径（返回合法 component + 夹取后的结构）
```

> 注：L1 元环允许 `ok → noop`。当基线 agent 在 meta-val 上已经满分时，**没有过程配置能进一步
> 提升分数**，此时不折腾流程才是正确行为；想稳定看到过程 promote 路径，看 L0 的确定性用例。
