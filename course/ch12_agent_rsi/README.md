# ch12 · agent-rsi（RSI 轨第 5 章）——把 agent 结构当成搜索空间

`rsi.py` = **ch11 拷贝 + 本课增量**（hero）；`agent.py` = **ch11 原样（冻结，被改进对象）**。

ch09–ch11 依次打开了 prompt、skills、memory 三个可变面，但**结构**仍然写死在冻结的 `agent.py` 里：

```python
AgentLoop(..., max_steps=8)          # 步数固定
AgentLoop(..., base_tools=all)       # 工具永远全开
AgentLoop(..., subagent_enabled=False)  # 子代理永远关闭
```

真实 harness 里这些全是**可搜索的离散结构**：给几步、开放哪些工具、要不要委派、最深几层。
本课把结构抽成 `AgentConfig` 上的字段，并回答两个新问题：

1. **搜索什么**：`max_steps` / `tool_names` / `subagent_enabled` / `max_subagent_depth`；
2. **怎么护栏**：结构不能无限膨胀——步数上限、工具数上限、深度上限，以及**每次搜索的评测预算**。

```
act   = rollout(current, train) → 分数 + trace（step_limit / tools_available / tools_enabled / subagent）
react = 定位「缺步数 / 缺工具 / 缺委派」→ 在 StructureGuard 内做一次离散结构编辑
```

`StructureSearchRsi` 再把「一次提议」升级成**小规模 beam**：对若干邻居在 train 的最差子集上 probe，
挑 probe 分最高、成本最低的候选，再交给 ch09 的两级门禁。**搜索宽度 × probe 任务数 = 显式评测开销**。

---

## 0. 流程图

```
        dataset/synth|dataset/rsi（train/val/test 不相交）
                              │
                              ▼
   ┌──────────────── act：rollout(current, train) ─────────────────────────┐
   │  AgentConfig 结构字段：max_steps / tool_names / subagent_enabled /    │
   │                       max_subagent_depth                             │
   │  rollout trace：steps/step_limit、tools_available、tools_enabled、   │
   │                 subagent_enabled、subagent_calls                     │
   └──────────────────────────────────┬────────────────────────────────────┘
                                      ▼
   ┌──────── react：结构粒度 credit assignment ─────────────────────────────┐
   │  RuleStructureLocator：                                               │
   │    未收尾 + 步数到顶            → step_limit                          │
   │    漏点 + 有禁用工具且没调成     → tool_missing                        │
   │    漏点 + 多要点 + 未开委派      → subagent_missing                    │
   │    其它 → format / halluc / frame / skip / guidance / ok              │
   │                                                                       │
   │  RuleStructureProposer：目标档位（加步数 / 全开工具 / 开 subagent）    │
   │  StructureLLMImprover： 一次 chat 调用 = 定位 + structure patch（JSON）│
   │  StructureSearchRsi：  邻居生成 → worst-k probe → 分数优先/成本次之    │
   └──────────────────────────────────┬────────────────────────────────────┘
                                      ▼
   StructureGuard.clamp（步数 1–12 / 工具 ≤8 / 深度 ≤2）
                                      ▼
   轮级 Gate（train-only）→ promote/reject
   每 val_every 轮 SegmentGate → commit/rollback/hold
   末尾 evaluate(test) 恰好一次 → RsiReport
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `AgentConfig.max_steps / tool_names / subagent_enabled / max_subagent_depth` | 四个可搜索结构维度；进版本指纹 |
| `AgentConfig.enabled_tools()` | `tool_names` 为空 = 全部；否则按名字过滤工具池 |
| `StructureGuard` | 硬护栏：步数 1–12、工具数 ≤8、深度 ≤2、可禁 subagent；越界一律夹回 |
| `update_structure` | 不可变结构编辑；非法字段丢弃、无变化返回原对象（哈希 noop） |
| `RolloutResult` 新 trace | `step_limit / tools_available / tools_enabled / subagent_enabled / subagent_calls` |
| `RuleStructureLocator` | 规则归因：`step_limit / tool_missing / subagent_missing` |
| `RuleStructureProposer` | 目标档位式编辑：重复诊断幂等 |
| `StructureLLMImprover` | 一次 chat 调用完成定位 + `structure` patch；值统一过护栏 |
| `StructureSearchRsi` | 邻居生成 + worst-k probe + 成本次优 + 显式 `search_evals` 预算 |
| `VersionRegistry.rollback` | 同步恢复 `tool_names / subagent_enabled / max_subagent_depth` |

## 2. 实现的对象 + 用法

### 结构进入版本指纹
```python
@dataclass
class AgentConfig:
    max_steps: int = 8
    tool_names: tuple[str, ...] = ()     # 空 = 启用全部
    subagent_enabled: bool = False
    max_subagent_depth: int = 2

    def enabled_tools(self):
        if not self.tool_names:
            return tuple(self.base_tools)
        wanted = set(self.tool_names)
        return tuple(t for t in self.base_tools if t.name in wanted)

    def canonical(self):
        return {..., "max_steps": self.max_steps, "tool_names": list(self.tool_names),
                "subagent_enabled": self.subagent_enabled,
                "max_subagent_depth": self.max_subagent_depth}
```
结构换了而哈希不变，门禁/回滚会全部失真；所以结构字段必须进 `canonical()`。

### 不改 agent.py，只改构造参数
```python
tools = config.enabled_tools()
loop = AgentLoop(model, system=system, memory=memory, skills=skills,
                 base_tools=list(tools), max_steps=config.max_steps,
                 subagent_enabled=config.subagent_enabled)
loop.max_subagent_depth = config.max_subagent_depth
```
冻结的 `AgentLoop` 早就支持这四个构造参数；本课只是把它们从常量变成**被搜索的配置**。
`agent.py` 一行不改——仍是「同一循环、不同装配」。

### 护栏必须放在边界
```python
@dataclass(frozen=True)
class StructureGuard:
    min_steps: int = 1
    max_steps: int = 12
    max_tools: int = 8
    allow_subagent: bool = True
    max_depth: int = 2

    def clamp(self, config): ...
```
- 步数/深度直接夹到区间；
- 未知工具名丢弃；
- `tool_names` 为空表示"全部"，但**全部也要受 `max_tools` 约束**（池大于上限时截到前 N 个）；
- 禁用 subagent 时强制关闭。

护栏在 `update_structure` 的出口统一执行，而不是散落在 locator/proposer/search 里：
规则和模型提议都可能给 `max_steps=999`，边界只认一个地方。

### 结构编辑是不可变的
```python
def update_structure(config, *, guard=None, **changes):
    allowed = {"max_steps", "tool_names", "subagent_enabled", "max_subagent_depth"}
    clean = {k: v for k, v in changes.items() if k in allowed}
    if not clean:
        return config
    cand = copy.deepcopy(config)
    for k, v in clean.items():
        setattr(cand, k, v)
    cand = (guard or StructureGuard()).clamp(cand)
    return config if cand.hash() == config.hash() else cand
```
`max_steps=4` 应用两次返回同一个对象 → 哈希不变 → 外环判 `noop`；避免重复诊断把配置越推越远。

### trace：结构失败要能被看见
```python
RolloutResult(
    ...,
    step_limit=config.max_steps,
    tools_available=tuple(t.name for t in config.base_tools),
    tools_enabled=tuple(t.name for t in tools),
    subagent_enabled=config.subagent_enabled,
    subagent_calls=sum(1 for tc in res.tool_calls if tc.name == "subagent"),
)
```
只看分数无法区分「模型不会」与「步数不够 / 工具没开 / 没委派」。trace 把装配信息暴露出来，
规则定位器才能把账记到具体结构维度上。

### 归因规则
```python
if not finished and steps >= step_limit:            → step_limit
if missing and disabled_tools and no enabled call:  → tool_missing
if missing and delegable and not subagent_enabled:  → subagent_missing
```
优先级刻意排在 `format` 之前：一个 1 步就耗尽的任务，先怪结构，而不是先怪模型输出格式。
误判仍由两级门禁兜底：候选只有在 train 提升且 val 不回归时才 commit。

### 目标档位而不是每轮 +1
```python
if diag.component == "step_limit":
    target = min(guard.max_steps, max(config.max_steps, guard.max_steps))
    return update_structure(config, guard=guard, max_steps=target)
```
同一诊断重复出现时返回原对象；避免「每轮 +1」在真实模型噪声下无限膨胀。

### 模型驱动结构改写
```python
diag, cand = await StructureLLMImprover(model, guard).improve(config, task, rollout_result)
# JSON: {component, finding, structure: {max_steps, tool_names, subagent_enabled, max_subagent_depth}}
```
`structure` 是增量 patch，经 `update_structure` 合并并夹取；解析失败/空 patch → `ok` + 原 config。

### 搜索：probe 是显式的评测预算
```python
class StructureSearchRsi(Rsi):
    search_width: int = 3        # 每轮最多 probe 几个邻居
    probe_tasks: int = 3         # 用 train 里最差的 k 个任务做代理
    max_search_evals: int | None = None
    search_evals: int = 0        # 实际消耗的 probe rollout 数
```
邻居：加步数（+4，夹到上限）/ 放开禁用工具 / 打开 subagent；按哈希去重、排除当前配置。
选择：`(probe 分, -步数, -工具数, -subagent, -深度)` 字典序最大 → **同分偏好更省的结构**。
probe 没有超过基线 → 返回原 config（noop），不浪费一次全量 train 评估。

### 正交性：一个容易踩的坑
护栏的 `min_steps` 如果设成 2，那么"只开 subagent"的邻居会被顺手抬到 2 步，导致
「加步数」和「开委派」两个编辑不再正交，probe 归因失真。本课把 `min_steps` 设为 1，
让每个邻居只动一个维度；这也是 L1 冒烟里能稳定看到 `max_steps=1 → 5` 的原因。

### 端到端（L0 确定性驱动）
- **步数不足**：工具在第 3 次调用后才给答案，基线 `max_steps=2` 必然收不了尾 → `step_limit`；
  搜索把步数提到 6 → train 提升 → promote；val 提升 → commit；test 评一次 = 1.0。
- **工具没开**：`lookup` 在池里但被 `tool_names` 禁用；模型只能拿到「没有该工具」的错误观察 →
  `tool_missing`；搜索放开工具 → promote + commit；test = 1.0。

## 3. 本课做到 / 没做到

**做到**：四个结构维度参数化并进入版本指纹/回滚；`StructureGuard` 统一夹取；不改 `agent.py`
只换装配；结构 trace；规则与 LLM 两套定位/编辑；目标档位幂等；带 `search_width / probe_tasks /
search_evals` 的 beam 搜索与成本次优；端到端「步数不足 / 工具没开」两条 promote+commit；L0 全 Mock。

**没做到（留给后续）**：工具集只做「开/关」，没有工具 schema/描述的演化；subagent 的**任务拆分策略**
仍写死；搜索只看单点邻居，没有组合编辑或历史学习；probe 用最差任务有选择偏差；
评测器/定位器本身仍不可变（ch13 meta 的入口）；没有跨 run 的结构搜索缓存。

## 4. 关联八股（~12 题）

1. **结构为什么是离散搜索**？它和 prompt/skills/memory 的参数化有什么本质区别？
2. **四个结构维度**分别解决什么失败？它们之间会怎样相互掩盖？
3. **护栏为什么放在边界**而不是循环里？如果把 clamp 散落到 locator/proposer 会出什么问题？
4. **`tool_names=()` 的语义**：为什么"空 = 全部"仍要受 `max_tools` 约束？
5. **正交性**：`min_steps` 为什么会污染 subagent 邻居？你会在测试里怎样抓这种污染？
6. **步数 vs 工具**：什么时候加步数有用、什么时候必须放开工具？举一个 trace 上的判别信号。
7. **subagent 的成本**：开委派会增加哪些开销？为什么不能"默认开"？
8. **probe 子集**：为什么用 train 最差 k 个任务？这种选择有什么偏差？如何校准？
9. **同分成本次优**：为什么同分时偏好更少步数/更少工具/不开委派？这是不是一种归纳偏置？
10. **评测预算**：`search_width × probe_tasks × n_rollouts` 怎样随轮数增长？`max_search_evals` 该放在哪一层？
11. **归因误判**：1 步耗尽也可能只是模型不会；两级门禁怎样兜底？你会怎样验证 `step_limit` 归因？
12. **回滚语义**：回滚结构 config 是否足够？已经在跑的 run、已经写入的记忆/版本怎么办？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py` 的 `AgentLoop`：`base_tools / subagent_enabled / max_subagent_depth`
  的生产实现（本课 `agent.py` 冻结自课程版）。
- `evolving_harness/rsi.py`：仍是 seam；本课展示了如何把结构并入不可变配置、propose 面与搜索预算。
- ch11：memory 策略演化；ch13 会把 proposer/门禁/评估器本身也变成被改进对象。
- 合成数据：`dataset/synth_pipeline/README.md`（math/tooluse，30/10/10，test-only-once）。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch12_agent_rsi/ -q
#   → 结构哈希 / 护栏夹取与不可变更新 / 步数与工具过滤 trace /
#     step_limit-tool_missing-subagent_missing 定位 / 规则编辑幂等 /
#     LLM structure patch 合并 / 版本回滚 / 邻居生成与 search_width /
#     端到端「步数不足」「工具没开」两条 promote+commit

uv run pytest dataset/ -q

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch12_agent_rsi/ -m smoke -s
#   → 真 deepseek 在 synth tooluse 上跑「结构搜索 → probe → 两级门禁 → test 一次」
#   → 真 deepseek 用「秘密只在工具观察里」的 vault 探针稳定走通「1 步 → 5 步」promote+commit
#   → 真 deepseek 走 StructureLLMImprover 的 JSON structure 改写路径

uv run ruff check course/ch12_agent_rsi/
```

> 注：synth tooluse 对 deepseek 可能一步内直接作答（0.5 分）→ probe 找不到更优结构、noop/hold，
> 这是**合法终态**。想稳定看到结构搜索的 promote 路径，看 L0 的确定性用例或 L1 的 vault 探针。
