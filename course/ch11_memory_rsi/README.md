# ch11 · memory-rsi（RSI 轨第 4 章）——把记忆策略当成可调参数

`rsi.py` = **ch10 拷贝 + 本课增量**（hero）；`agent.py` = **ch10 原样（冻结，被改进对象）**。

ch09 演化 prompt，ch10 演化 skills。两者都改「模型看到什么指令」；本课改**模型如何记住/忘掉东西**。
冻结的 `agent.py` 里记忆行为是写死的：

```python
abbreviate_observation(content)          # 工具结果一律截断到 500 字符
ContextBudget(max_len=6000, threshold=0.8, keep_recent=4)
MemoryStore.recall(...)                  # 永远最多拼 8 条命中
profile.into_system()                    # 画像永远常驻
store.append_turn(context)               # 完整记忆永远落盘
```

真实 harness 里这些全是**策略**：截多少、留几条、何时压缩、读不读历史、画像开不开。本课把它们抽成
`MemoryPolicy`，放进 `AgentConfig`（因此进入版本指纹），再让 meta-agent 外环去调：

```
act   = rollout(config, train) → 分数 + trace（abbreviated / compacted / recalled / profile_updated）
react = 定位「记忆写/读/压哪里出错」 → 调一次 MemoryPolicy（放宽截断 / 提前压缩 / 收紧召回）
```

难点仍是 eval + credit assignment：分数由 ch09 的 Judge 后端给；本课要回答
**「这次失败该记到记忆的哪一环？」**——是工具观察被截断了（`memory_miss`）、压缩时机把上下文压爆了
（`memory_overflow`）、还是召回把无关旧内容带回来了（`memory_pollution`）。

---

## 0. 流程图

```
        dataset/synth/{math,tooluse}/*.jsonl（沿用 ch10 的合成数据）
                              │
                              ▼
   ┌──────────────── act：rollout(current, train) ────────────────────────┐
   │  AgentConfig.memory: MemoryPolicy                                    │
   │      ├─ PolicyWorkingMemory   截断阈值 = abbreviate_chars             │
   │      ├─ PolicyMemoryStore     召回 top_k / recall_enabled             │
   │      ├─ ContextBudget         压缩 max_len/threshold/keep_recent      │
   │      └─ 开关                  store_full_turns / profile_enabled      │
   │  trace = 分数 + {abbreviated, compacted, recalled, profile_updated}   │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      ▼
   ┌──────────────── react：memory 粒度 credit assignment ────────────────┐
   │  RuleMemoryLocator：                                                 │
   │    未收尾 + 压缩过             → memory_overflow                      │
   │    漏点 + 截断/压缩过           → memory_miss                          │
   │    禁词/漏点 + 召回或画像动过   → memory_pollution                     │
   │    其它 → format / skip / ok / guidance                              │
   │                                                                       │
   │  RuleMemoryProposer：放宽保留/截断 / 提前压缩 / 收紧召回或关画像       │
   │  MemoryLLMImprover： 一次 chat 调用 = 定位 + 返回 policy patch（JSON） │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      ▼
   轮级 Gate（train-only）→ promote/reject
   每 val_every 轮 SegmentGate → commit/rollback/hold（rollback 连 memory policy 一起退）
   末尾 evaluate(test) 恰好一次 → RsiReport
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `MemoryPolicy` | 八个可调参数：`max_len / threshold / keep_recent / abbreviate_chars / store_full_turns / recall_enabled / recall_top_k / profile_enabled` |
| `PolicyMemory` | 冻结 `Memory` 的可调包装：替换 `WorkingMemory`、`MemoryStore`、画像/落盘开关 |
| `PolicyWorkingMemory` | 工具观察按 `abbreviate_chars` 截断，并记录 `truncated` |
| `PolicyMemoryStore` | `MemoryStore.recall` + `top_k` 上限 |
| `RolloutResult` 新 trace | `abbreviated / compacted / recalled / profile_updated` |
| `RuleMemoryLocator` | 规则归因：`memory_miss / memory_overflow / memory_pollution` |
| `RuleMemoryProposer` | 规则调参：目标值幂等（重复诊断不会无限改） |
| `MemoryLLMImprover` | 一次 chat 调用完成定位 + `policy` patch；值统一夹到安全区间 |
| `update_memory_policy` | 不可变替换 + `_clamp_memory`：预算/阈值/条数都有上下界 |

## 2. 实现的对象 + 用法

### 策略进入版本指纹
```python
@dataclass(frozen=True)
class MemoryPolicy:
    max_len: int = 6000
    threshold: float = 0.8
    keep_recent: int = 4
    abbreviate_chars: int = 500
    store_full_turns: bool = True
    recall_enabled: bool = True
    recall_top_k: int = 8
    profile_enabled: bool = True

class AgentConfig:
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    def canonical(self):
        return {..., "memory": self.memory.canonical()}
```
ch10 的版本指纹只覆盖 prompt + skills；不把 memory 放进去，就会出现「策略换了、版本 hash 没变」，
门禁与回滚全部失真。`VersionRegistry.rollback` 同步恢复 memory policy。

### 不改 agent.py，替换记忆行为
```python
class PolicyWorkingMemory(WorkingMemory):
    def ingress(self, msg):
        if msg.role == "tool":
            self.truncated |= len(" ".join(msg.content.split())) > self.max_chars
            return Message(role="tool", tool_call_id=msg.tool_call_id,
                           content=abbreviate_observation(msg.content, self.max_chars))
        return msg

class PolicyMemory(Memory):
    def __init__(self, model, policy, storage_dir=None):
        super().__init__(model, budget=policy.budget(), storage_dir=storage_dir)
        self.store = PolicyMemoryStore(storage_dir, top_k=policy.recall_top_k)
        self.wm = PolicyWorkingMemory(self.budget, policy.abbreviate_chars)
        ...
```
`PolicyMemory` 通过**组合/覆写**拿到策略，不需要改冻结的 `agent.py`；这是「边界适配」原则在记忆上的应用。

### trace：没有信号就没有 credit assignment
```python
RolloutResult(
    ...,
    abbreviated=memory.wm.truncated,   # 工具观察被规则截断
    compacted=memory.compacted,        # 触发过模型压缩
    recalled=memory.recalled,          # 从完整记忆回填过
    profile_updated=memory.profile_updated,
)
```
只用分数无法区分「模型不会」与「事实被截断」。trace 把这些内部动作暴露出来，规则定位器才能把账记到
`memory_miss / memory_overflow / memory_pollution` 上。

### 归因规则
```python
if not finished and compacted:                 → memory_overflow
if missing and (abbreviated or compacted):     → memory_miss
if (violated or missing) and (recalled or profile_updated): → memory_pollution
```
这些是**便宜的一阶信号**，会有误判（例如截断后模型本来就答不出）。误判由两级门禁兜底：
候选只有在 train 提升且 val 不回归时才会 commit；否则 rollback。

### 调参而不是无限追加
```python
if diag.component == "memory_miss":
    return update_memory_policy(config,
        abbreviate_chars=max(2000, p.abbreviate_chars),
        keep_recent=max(6, p.keep_recent),
        threshold=min(0.9, p.threshold))
```
规则 proposer 把参数推向一个**目标档位**，而不是每轮 +1：同一诊断重复出现时返回原对象 → hash 不变 →
外环判 noop；避免「越改越离谱」。`update_memory_policy` 再用 `_clamp_memory` 把模型给的值夹到
`[200, 100000]` 等安全区间。

### 模型驱动
```python
diag, cand = await MemoryLLMImprover(model).improve(config, task, rollout_result)
# JSON: {component, finding, policy: {max_len, threshold, abbreviate_chars, ...}}
```
`policy` 是增量 patch，与当前策略合并后再夹取；解析失败/空 patch → `ok` + 原 config。

### 端到端（L0 确定性驱动）
`lookup` 工具返回 `"x"*100 + " MAGIC"`：
- 基线 `abbreviate_chars=20` → 观察截断在 MAGIC 之前 → 答 `"0"` → `memory_miss`；
- 规则 proposer 把阈值推到 2000 → 候选看到 MAGIC → train 提升 → promote；
- val 也提升 → segment commit；test 评一次 = 1.0。

## 3. 本课做到 / 没做到

**做到**：MemoryPolicy 八个维度参数化并进入版本指纹/回滚；不改 agent.py 替换截断/召回/画像/落盘；
四类记忆 trace；规则与 LLM 两套记忆归因/调参；参数夹取；目标档位幂等；端到端「截断太狠 → 放宽 →
promote+commit」；L0 全 Mock。

**没做到（留给后续）**：loop 结构（步数、工具集、subagent 开关）仍不可变（ch12）；「记忆收益/成本」
没有显式预算（只靠 val 门禁）；画像 schema 仍固定；没有跨 run 的记忆库版本管理（ch13 meta 可做）；
规则定位只用了 trace 的一阶信号，复杂记忆失败仍需 LLM/决策模型。

## 4. 关联八股（~12 题）

1. **记忆策略为什么是可调参数而不是代码分支**？参数化对版本/门禁/回滚有什么直接好处？
2. **「存什么 / 留多久 / 压多少」**分别对应 MemoryPolicy 的哪些字段？它们之间如何相互影响？
3. **截断 vs 压缩**：规则简写（便宜、确定性）与模型压缩（昂贵、有损）的边界应该怎么划？
4. **keep_recent 的取舍**：保留太少会丢什么？保留太多会挤掉什么？
5. **压缩时机**：threshold 调低（更早压缩）与调高（更晚压缩）分别有什么风险？
6. **召回 top-k**：召回太少与太多分别怎样伤到回答？为什么召回也可能造成 `memory_pollution`？
7. **画像常驻**：用户画像进 system 的成本是什么？什么时候应该关掉？
8. **trace 的必要性**：只看分数为什么无法区分「模型不会」与「记忆丢了」？四类 trace 各自对应哪种失败？
9. **credit assignment 的误判**：截断后答错不一定怪记忆——门禁如何兜底？你会怎样验证归因？
10. **参数修改的幂等性**：为什么「每轮 +1」会失控？目标档位法有什么局限？
11. **安全夹取**：模型/规则给出的 max_len=0、threshold=9 会造成什么？`_clamp_memory` 应放在哪一层？
12. **回滚语义**：回滚记忆策略意味着什么？如果记忆文件已经写入，回滚 config 是否足够？（提示：状态 vs 策略）

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py` 的 `ContextBudget / WorkingMemory / MemoryStore / UserProfile / Memory`：
  产品版记忆实现（本课 `agent.py` 冻结自课程版）。
- `evolving_harness/rsi.py`：仍是 seam；本课展示了如何把「记忆策略」并入不可变配置与 propose 面。
- ch10：skills 演化；ch12 会把 loop 参数/工具集也纳入同一个配置。
- 合成数据：`dataset/synth_pipeline/README.md`（math/tooluse，30/10/10，test-only-once）。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch11_memory_rsi/ -q
#   → 策略哈希 / 夹取与不可变更新 / 截断阈值与 trace / 召回 top-k 与开关 /
#     memory_miss-overflow-pollution 定位 / 规则调参幂等 / LLM policy 合并 /
#     版本回滚 / synth 加载 / 端到端「放宽截断 → promote+commit」

uv run pytest dataset/ -q

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch11_memory_rsi/ -m smoke -s
#   → 真 deepseek 在 synth math 上跑「记忆定位 → 策略调参 → 两级门禁 → test 一次」
#   → 真 deepseek 走 MemoryLLMImprover 的 JSON policy 改写路径

uv run ruff check course/ch11_memory_rsi/
```

> 注：合成 math 题对 deepseek 可能 baseline 已高 → 第一轮 noop、段级 hold。这是**合法终态**。
> 想稳定看到「截断丢失事实 → 放宽策略 → promote + commit」，看 L0 的端到端用例（确定性 lookup 工具）。
