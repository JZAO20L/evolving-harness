# ch14 · versioning（RSI 轨尾）—— agent 版本库：合并 / 门控提交 / 历史回滚

ch08–ch13 的 `VersionRegistry` 只是 **append-only 登记**：版本号 = config 哈希，
`latest()` 取最后一行，`rollback()` 返回 `AgentConfig` 但生产代码从不调用，也没有
“当前生效版本”指针。本章把“登记”升级成**版本库 + 编排器**：

```
proposers ──► ChangeSet(s) ──► merge（单基线、多组件）
                                    │
                                    ▼
                   evaluate(train) + evaluate(val)（可复用缓存分数）
                                    │
                                    ▼
                   SegmentGate（commit / rollback / hold）
                                    │ commit
                                    ▼
      AgentRepo.commit：versions/<hash>.json（原子写）+ HEAD（原子前进）
                                    │ rollback / checkout
                                    ▼
           log() / ancestors() / diff() 里选历史版本当新基线
```

本章**不改写 ch13 的 `Rsi` 主循环**：`Rsi` 继续负责“定位 + 提议”，本章新增
`versions.py`（仓库 / diff / 合并）和 `VersionedRsi`（合并 → 评估 → 门控 → 提交）。

## 0. 流程图

```
┌──────────────────────── VersionedRsi.run() ────────────────────────┐
│                                                                    │
│  base = base_version or repo.head()                                │
│    │                                                               │
│    ├─ base is None（空仓库）                                        │
│    │     evaluate(train) + evaluate(val)                           │
│    │     repo.commit(verdict="base", parent=None)                  │
│    │                                                               │
│    └─ base is 历史版本                                              │
│          repo.checkout(base, move_head=True)   ← 从历史版本选基线   │
│          基线分数优先读版本记录缓存，缺失才重算                      │
│                                                                    │
│  ┌────────────────────── 每轮（max_rounds）─────────────────────┐  │
│  │  proposers → ChangeSet × N                                  │  │
│  │        │                                                   │  │
│  │        ▼                                                   │  │
│  │  merge_changes(base, changesets)                            │  │
│  │    ├─ MergeConflict / BaseMismatch → verdict=conflict      │  │
│  │    │                                 （不评估，HEAD 不动）    │  │
│  │    └─ 合并后 hash 不变 → verdict=noop（HEAD 不动）           │  │
│  │        │                                                   │  │
│  │        ▼                                                   │  │
│  │  evaluate(candidate, train) + evaluate(candidate, val)      │  │
│  │        │                                                   │  │
│  │        ▼                                                   │  │
│  │  SegmentGate.decide(watermark, train, val)                  │  │
│  │    ├─ commit   → repo.commit（对象 → 校验 parent → HEAD）    │  │
│  │    │              watermark 前进，继续下一轮                 │  │
│  │    ├─ hold     → HEAD 不动，结束                             │  │
│  │    └─ rollback → 候选不落盘，HEAD 不动，结束                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  return VersionedReport(outcomes, repo.head(), checkout(HEAD))     │
└────────────────────────────────────────────────────────────────────┘
```

裁决语义：

| verdict | 含义 | HEAD |
|---|---|---|
| `commit` | 合并候选在 train + val 都过血线，落盘并前进 | 前进到新版本 |
| `rollback` | 候选让分数回退 | 不动 |
| `hold` | 候选分数与血线完全一致（无提升） | 不动 |
| `conflict` | 两个 proposer 对同一组件给出不同新值 / 基线不匹配 | 不动 |
| `noop` | 没有改动，或合并结果哈希与基线相同 | 不动 |

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `AgentConfig.canonical()` / `hash()`（ch13） | 内容寻址的完整配置快照；id = SHA-256 |
| `AgentConfig.from_canonical()`（本章新增） | 从磁盘记录重建 config，并重新注入 live tool handler |
| `Version` | 版本记录：id / parents / message / author / 时间 / canonical / train+val 分数 / 裁决 / meta |
| `AgentRepo` | 文件版本库：`HEAD` + `versions/<id>.json` + `.lock` |
| `fcntl.flock` | 保护“校验 parent + 推进 HEAD”的临界区 |
| `os.replace` | 对象与 HEAD 的原子替换（同目录） |
| `ConfigDiff` | 组件级差异，`before/after` 都是绝对值 |
| `ChangeSet` | 一组“组件 → 绝对新值”，绑定在某一基线上 |
| `merge_changes` | 单基线合并：异组件组合、同组件同值幂等、异值冲突 |
| `SegmentGate` / `Checkpoint`（ch13） | train+val 血线门禁；`Checkpoint` 复用基线缓存分数 |
| `VersionedRsi` | 编排器：proposer → 合并 → 评估 → 门控 → 提交 / 回滚 |
| `repo.diff` / `repo.log` / `repo.checkout` | 历史查看与“从历史版本选择基线” |

## 2. 实现的对象 + 用法

### 存储布局

```
<storage_dir>/
  versions/<id>.json      # 内容寻址对象，id == AgentConfig.hash()
  HEAD                    # 当前生效版本 id；空仓库时不存在
  .lock                   # fcntl.flock 锁文件
```

- 对象一旦落盘不再改写；同一 canonical → 同一 id；
- `HEAD` 是唯一可变指针，回滚 / checkout 本质上都是移动它；
- 对象与 HEAD 都用 `*.tmp` + `os.replace`，避免读到半截文件。

### 提交 / 查询 / 历史

```python
repo = AgentRepo("data/versions")
vid = repo.commit(config, parent=repo.head(), message="round 1: prompt",
                  train_scores=train, val_scores=val, verdict="commit",
                  meta={"diff": diff.changed})
repo.head()                       # 当前生效版本
repo.get(vid)                     # -> Version | None
repo.log(limit=10)                # 全部版本，新→旧（含已回滚的旧分支）
repo.ancestors()                  # 只沿 parents 走当前谱系
repo.diff(a, b)                   # a -> b 的组件级 diff
```

- **`log()` vs `ancestors()`**：`log()` 是“仓库里所有对象”，回滚后不再位于 HEAD
  祖先链上的版本仍可查；`ancestors()` 才是当前生效谱系。要“从历史版本选择”，用
  `log()` 找到 id，再 `checkout`。
- **原子提交两步**：先写 `versions/<id>.json`，再加锁校验 `parent == HEAD` 并原子
  替换 `HEAD`。对象已落盘、HEAD 未更新时崩溃 → 新对象成为孤儿，可被 `log()` 看到，
  但 HEAD 不变。这是本章明确接受的“两阶段”语义。
- **父版本校验**：锁内再读 HEAD；`parent` 不一致抛 `ConcurrentCommit`，不覆盖别人
  先提交的结果。
- **幂等**：同一 canonical 重复 `commit` 返回旧 id，不重写对象，HEAD 不动。
  代价是同一配置从不同基线到达会折叠 lineage（parent 以首次登记为准）。

### 回滚 / 历史选择

```python
repo.rollback()                    # HEAD -> HEAD.parent（根版本则 ValueError）
restored = repo.rollback(v3)      # HEAD -> 指定历史版本，返回等价 config
```

- 回滚**只移动 HEAD**，不删除任何对象；
- `checkout(v, move_head=True)` 是“把历史版本设为新基线”，后续提交的 parent 就是 v；
- **handler 不可序列化**：canonical 只存 `base_tools` 名字。`checkout`/`rollback`
  必须由调用方注入 live handler：

```python
config = repo.checkout(vid, base_tools=(bash_tool, read_file_tool))
config.enabled_tools()[0].handler({"command": "echo hi"})   # 可以真正执行
```

### 组件 diff 与 ChangeSet

```python
diff = config_diff(base, cand)
diff.components                   # ("prompt", "structure")
diff.changed["structure"]         # {"before": {...}, "after": {...}}

changeset = ChangeSet.from_config(base, cand)       # 只含变化组件
changeset = ChangeSet.from_changes(base, {"structure": {"max_steps": 5}})  # 允许部分字段
changeset.to_config(base_tools=(tool,))             # 重建候选 config
```

组件键固定为：

| 组件 | 内容 |
|---|---|
| `prompt` | guidance / task_frame / final_format |
| `skills` | 完整 skills 列表（含 instructions） |
| `memory` | MemoryPolicy 全字段 |
| `structure` | `max_steps` / `tool_names` / `subagent_enabled` / `max_subagent_depth` |

- `name` 与 `base_tools` 视为**不可变**；改动它们抛 `ImmutableChange`（base_tools
  的 handler 无法序列化，也意味着“换工具池”不能靠文件版本回放）；
- `ChangeSet.hash()` 对排序后的 changed JSON 求哈希，与 dict 插入顺序无关；
- `structure` 允许只写变化的键，`apply()` 会合并回基线；其余组件要求完整 payload。

### 单基线合并

```python
merged = merge_changes(base, [prompt_changeset, structure_changeset])
candidate = merged.to_config(base_tools=current.base_tools)
```

| 情形 | 结果 |
|---|---|
| 不同组件 | 直接组合 |
| 同一组件、绝对新值相同 | 幂等合并 |
| 同一组件、绝对新值不同 | `MergeConflict(component, variants)` |
| `ChangeSet.base_version != base.hash()` | `BaseMismatch` |

本章只做**单基线、多组件改动**：不做多亲三方合并，也不自动“择优”解决冲突。
冲突直接终止本轮，HEAD 不动，交给上层决定换 proposer 还是换基线。

### VersionedRsi

```python
repo = AgentRepo("data/versions")

async def propose(current, scores):
    # 这里可以接 ch08–ch13 的 RuleStructureProposer / LLM improver / 自定义策略
    candidate = ...
    return ChangeSet.from_config(current, candidate)

versioned = VersionedRsi(
    config, model, repo=repo,
    train=load_split("train"), val=load_split("val"),
    proposers=[propose],
    judge=RuleJudge(), segment_gate=SegmentGate(),
    n_rollouts=1, max_rounds=2,
    base_version=None,          # None = HEAD；“从历史版本选择”传具体 id
)
report = await versioned.run()
report.outcomes                     # 每轮 verdict / 分数 / diff / version
report.head                         # 最终 HEAD
report.head_config                  # 重建后的最终 config（live tools 由 config 注入）
```

内部流程：

1. **基线**：`base_version or repo.head()`；空仓库先评估 train + val，提交
   `verdict="base"` 的根版本；指定历史版本则 `checkout(move_head=True)`。
2. **缓存分数**：优先用版本记录里的 train/val 分数构造 `Checkpoint`；缺失才重算。
   `self.evals` 暴露实际评估次数（每评估一个 split 记 1）。
3. **提议**：`proposers` 列表顺序调用，返回 `None` 视为本轮无改动；默认适配器
   包装 ch13 `Rsi.propose_improvement`（建单轮 `Rsi`，再转成 `ChangeSet`）。
4. **合并**：`merge_changes`；冲突 → `verdict="conflict"`，**不评估、HEAD 不动**。
5. **评估**：合并候选跑 train + val；`SegmentGate.decide(watermark, ...)` 给出
   `commit / rollback / hold`。
6. **提交**：只有 `commit` 调 `repo.commit`；`meta.diff` 记录组件差异，
   `meta.sources` 记录本轮合并了哪些组件；`watermark` 与基线前进，继续下一轮。
7. **结束**：返回 `VersionedReport(outcomes, repo.head(), checkout(HEAD))`。

注意：

- 门禁只看 **train + val**；`test` 不在 `VersionedRsi` 的入参里，避免每轮偷看测试集；
- `rollback` 是“候选不落盘、HEAD 不动”，不是“把 HEAD 挪到上一个版本”；
- 同一配置若已在仓库里，`commit` 幂等返回旧 id 但**不移动 HEAD**——所以编排器在
  提交前先比较 `candidate.hash() == current.hash()`，把这种情形归为 `noop`。
- `VersionedRsi` 仍通过 `rollout()`（ch13 唯一 config→agent 接缝）执行；`agent.py`
  本章原样冻结，不改一行。

### 错误与失败语义

| 情况 | 行为 |
|---|---|
| 提交时 HEAD 被其他写者推进 | `ConcurrentCommit`，本次不落对象 |
| 版本对象 JSON 损坏 | `VersionStoreError`（错误信息带文件路径） |
| 指定版本不存在 | `checkout/get` 的 `KeyError`（`get` 返回 `None`） |
| 空仓库 checkout / 根版本回滚 | `ValueError` |
| 合并冲突 | `MergeConflict`，携带 `component` 与 `variants` |
| ChangeSet 基线不匹配 | `BaseMismatch`，携带 `expected` / `got` |
| 改 `name` / `base_tools` | `ImmutableChange` |

## 3. 本课做到 / 没做到

**做到**：`AgentConfig.from_canonical`（含 live handler 重注入）；`Version` 记录；
`AgentRepo` 内容寻址对象 + HEAD + `fcntl` 锁 + 两阶段原子提交 + 幂等去重；
`head/log/ancestors/get/checkout/rollback/diff`；`ConfigDiff` 组件级绝对差异；
`ChangeSet` 与单基线 `merge_changes`（异组件组合、同值幂等、异值冲突）；历史版本
选择（`checkout(move_head=True)`）；`VersionedRsi` 合并 → 评估 → 门控 → 提交 /
回滚；基线分数缓存复用；L0 全离线 27 例；L1 真模型冒烟（真实模型合法 `hold`）。

**没做到（留给后续）**：没有多亲三方合并 / 自动冲突解决；没有 tags、branches、
reflog 与孤儿对象 GC；没有跨进程写者之外的分布式事务（文件锁只在单机有效）；
没有把最终 HEAD 自动部署回 `agents/*.yaml`；没有 SQL 查询/分页（ch19 只在
`agent_versions` 表里存版本号 + 静态 YAML）；没有回放页面；没有把 skills/memory
diff 做成结构化审批对象；没有对 `canonical` 做 schema 迁移与向后兼容。

## 4. 关联八股（~12 题）

1. **内容寻址 vs 顺序号**：为什么版本 id = config 哈希？同一配置从不同基线到达会
   发生什么？何时需要 `(config, parent)` 复合 id？
2. **HEAD vs latest**：`head()` 与 `log()[0]` 的区别？回滚后 `log()` 里为什么还能
   看到旧版本？`ancestors()` 与 `log()` 各适合什么界面？
3. **两阶段原子性**：先写对象再写 HEAD，崩溃点各会造成什么？为什么不用一个文件
   同时存内容与指针？`os.replace` 保证了什么？
4. **锁与 parent 校验**：为什么锁内还要再读一次 HEAD？`ConcurrentCommit` 与
   “后写覆盖”相比，哪个语义更适合版本库？
5. **幂等提交**：重复提交同一 canonical 为什么返回旧 id？它如何影响 lineage？
   如果希望同一配置在多条基线出现，需要改什么？
6. **handler 不可序列化**：为什么 canonical 只存工具名？`checkout` 为什么必须由
   调用方注入 live handler？这限制了哪些“版本可回放”的承诺？
7. **单基线合并**：不同组件为什么可以直接组合？同组件同值为什么幂等？同组件异值
   为什么不能自动选一个？多亲三方合并需要什么额外信息？
8. **base 必须匹配**：`ChangeSet.base_version` 的作用是什么？如果允许跨基线应用，
   会引入哪些“文本合并 vs 语义合并”的问题？
9. **缓存分数何时失效**：版本记录的 train/val 分数能直接复用吗？更换 judge、
   n_rollouts、数据集或模型后，缓存分数还可信吗？
10. **train+val 门禁与 test 纪律**：为什么本章门禁不碰 test？`SegmentGate` 的
    `hold` 与 `rollback` 如何区分“没提升”和“变差”？
11. **文件仓库 vs SQL 版本表**：`AgentRepo` 与 ch19 `agent_versions` 各承担什么？
    事务、并发写、按组件查询、GC 分别该放在哪一层？
12. **rollout vs checkout**：`rollout(config)` 把配置变成一次 agent 执行；
    `checkout(version)` 把磁盘记录变成配置。两者为什么必须分开？缺少 live handler
    时哪一步会失败？

## 5. 生产版延伸阅读（复用 + 回填）

- ch08–ch13：`VersionRegistry` / `Checkpoint` / `SegmentGate` / `rollout` / `evaluate`
  是本章的输入；本章把“登记”升级为“仓库”，主循环本身不改。
- ch19 SQL：`agent_versions(version, agent_hash, static YAML)` + `GET /v1/agent/versions`
  是版本库的服务端持久化形态；本章的文件仓库是它的离线 / 单机版本。
- `evolving_harness/agent.py`：`State` 的 hash / checkpoint 与本章“内容寻址 +
  可回放快照”是同一思路，只是范围从“一次 run 的状态”扩大到“agent 配置”。
- `rsi.py` v0.2 占位：产品侧 RSI 的版本改进可以复用本章的 `AgentRepo` + `ChangeSet`，
  把“提议 → 门控 → 提交”接进现有 `Rsi` 外环。
- ch22 Gradio：可把 `repo.log()` / `repo.diff()` 做成历史版本列表与 diff 预览，
  复用 `approval.request` 的审批闭环。

## 6. 运行与验证

```bash
# L0 离线（不需要模型 / key）
uv run pytest course/ch14_versioning/ -q
#   → 27 passed, 1 skipped
uv run ruff check course/ch14_versioning/

# L1 冒烟：真模型 + 真 rollout（key 只经 shell 环境载入）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch14_versioning/ -m smoke -s

# 章节改名后的逐章回归（注意：pytest 必须按章跑）
for d in ch15_fastapi ch16_concurrency ch17_sse ch18_websocket \
         ch19_sql ch20_redis ch21_docker ch22_gradio; do
  uv run pytest course/$d/ -q
done
```

> 本章由 RSI 轨尾部插入，原后端 / 前端 8 章整体后移一位（fastapi…gradio 现在是
> ch15–ch22）。本章额外带 `versions.py` 作为新 hero；`agent.py` 与 `rsi.py` 是 ch13
> 快照 + 增量。
