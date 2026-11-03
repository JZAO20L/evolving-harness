# ch09 · prompt-rsi（RSI 轨第 2 章）——模型驱动 eval/locate + train/val/test 评估纪律

`rsi.py` = **ch08 拷贝 + 本课增量**（hero）；`agent.py` = **ch08 原样（冻结，被改进对象）**。

ch08 证明了无训练外环能跑：reward → locate → propose → gate/版本。但它有两个短板，本课补上：

1. **评分路径只能同步**：ch08 `evaluate()` 直接调 `scorer.score`（未 `await`、不传 model），所以只有
   规则打分能跑，`JudgeScorer` 这类 async 打分一接就崩。
2. **没有评估纪律**：所有任务混在一个集子上反复调 → RSI 可以把评测集本身背下来（**评测污染/背题**）。

本课把 RSI 明确成 **二阶 Agent / meta-agent**：

```
一阶 Agent：act = 调工具/写答案        react = 看观测决定下一步
RSI（二阶）：act = rollout 当前策略      react = 定位失败 + 改组件
               （观测 = 分数 + trace）          （本课把 locate/propose 模型化，且可合并成一次生成）
```

**RSI 的难点不是"怎么改"（对着组件改就行），而是"怎么评估 + 怎么定位"（eval + credit assignment）。**
本课是这条主轴的**第一步**：判分与定位从纯规则升级到「**chat LLM** / **决策式模型（System One）**」，
并用 train/val/test + 两级门禁把评估纪律立起来。

---

## 0. 流程图

```
        dataset/rsi/{train,val,test}.jsonl（verifier-first 探针集，30 条，互不相交）

   train ───────────────────────────┐          val ─────────────┐        test（只碰一次）
                                    │                            │              │
                    ┌───────────────▼──────────────────────────┐ │              │
   repeat           │ act：rollout(current, train) → Judge 判分 │ │              │
   (每一轮)          │      → 分数 + trace                       │ │              │
                    │ react：worst task → Locator/LLM/SystemOne │ │              │
                    │      定位组件 → Proposer/LLM 改写 prompt   │ │              │
                    │      候选 config（新 hash，不可变）        │ │              │
                    │                                          │ │              │
                    │ 轮级 Gate（只看 train）：聚合升且不回归？  │ │              │
                    │   promote → current 前移                  │ │              │
                    │   reject  → current 不动                  │ │              │
                    │                                          │ │              │
                    │ 每 val_every 轮：SegmentGate             │ ▼              │
                    │   cur_train 与 cur_val **都**超过血线？   │ evaluate(val)  │
                    │   commit → watermark = 当前 (config+分)   │                │
                    │   rollback → 回到 watermark 的 config/分  │                │
                    │   hold  → 无变化                          │                │
                    └───────────────▲──────────────────────────┘                │
                                    │ max_rounds 到 / noop                      │
                          ┌─────────▼─────────────────┐                         │
                          │ 末尾 evaluate(test) 恰好一次 │◄───────────────────────┘
                          │ → RsiReport(test_score)    │
                          └───────────────────────────┘
```

两条门禁的分工：**轮级管"这一步值不值"（train 内部）**；**段级管"这段改动能不能泛化"（train+val 对血线）**。

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `Judge`（async） | 统一判分接口：`RuleJudge`（确定性）· `LLMJudge`（chat 打分）· `SystemOneJudge`（决策模型 score 型） |
| `Locator`（async） | 统一归因接口：`RuleLocator`（规则）· `LLMLocator`（chat 选组件）· `SystemOneLocator`（决策模型 choice 型） |
| `Proposer`（async） | 统一改写接口：`RuleProposer`（确定性规则）· `LLMProposer`（按归因生成规则） |
| `Improver` | 把 locate+propose 打包：`TwoStageImprover`（两段）· `MergedLLMImprover`（**一次**生成 thought+act） |
| `SystemOneClient` | ARK 决策评估（beta）：`POST /api/v1/systemone`，头 `ark-beta-systemone-api: true`，**仅判断不生成** |
| `Gate` / `SegmentGate` | 轮级（train-only）/ 段级（train+val 对 `Checkpoint` 血线，严格双方提升） |
| `Checkpoint` | 血线快照：`{config, train_scores, val_scores, version}`；回滚直接复用分数，免重算 |
| `load_split` | 读仓库根 `dataset/rsi/{train,val,test}.jsonl` → `Task`（与 `course/` 平级、跨章共享） |

**System One 决策 API 要点**（详见 `/home/tiger/docs/models/ark_decision.md`）：`state`（待判断内容）+
`questions`（`type` ∈ `noul|choice|score`，`instructions` + `criteria`）→ `answers`；按**输入 token** 计费；
`choice` 建议配「信息不足」兜底；**只做结构化判断，生成必须走 chat LLM**。

## 2. 实现的对象 + 用法

### 修 async 评分路径
```python
async def evaluate(tasks, config, model, judge=None, n_rollouts=1) -> list[TaskScore]:
    ...
    scores.append(await judge.score(t, r))     # model 只用于 rollout；judge 自己持有依赖
```
`train/val/test` **共用同一个 `n_rollouts`**，修掉 ch08「基线 1、候选 N」的不对称。

### Judge —— 三种后端一个接口
```python
class Judge:
    async def score(self, task: Task, r: RolloutResult) -> float: ...

RuleJudge()                     # 包 RuleScorer，确定性、离线骨干
LLMJudge(model)                 # chat：rubric+answer → _parse_score
SystemOneJudge(client)          # 决策：score 型，2 档加权 → 归一化到 [0,1]
```

### Locator —— 三种后端一个接口
```python
class Locator:
    async def locate(self, task, r, score) -> Diagnosis: ...

RuleLocator()                   # 空答/未收尾→skip；禁词→format；缺要点且没查工具→halluc；查了仍缺→frame
LLMLocator(model)               # 喂 trace+rubric+分数，只回 JSON {component, finding}；解析失败→ok
SystemOneLocator(client)        # choice 型选组件，带「信息不足」兜底（映射到 ok）
```
组件集合固定为 `format/frame/halluc/skip/guidance/ok`，保证「定位」结果对下一步「改写」是可执行的。

### Proposer / Improver
```python
RuleProposer()                      # 依 Diagnosis 往 final_format / guidance 追加规则（不可变）
LLMProposer(model)                  # 依归因生成一条可追加的规则（JSON）
TwoStageImprover(locator, proposer) # locate → propose（两个后端可自由组合）
MergedLLMImprover(model)            # 一次生成 {component, finding, guidance, final_format}
```
`MergedLLMImprover` 是 meta-agent 的 **thought+act 合并**：定位与改写共享同一份上下文，少一次往返。
不可变约定不变：改写永远 `deepcopy` 出新 config，原配置不动。

### SystemOneClient
```python
client = SystemOneClient(model="doubao-seed-2-1-lite-260915", api_key=...)
answers = await client.ask(
    state={"task": ..., "answer": ...},
    questions={"q": {"type": "choice", "instructions": ..., "criteria": {...}}},
)
await client.aclose()
```
httpx + `RETRYABLE_STATUS` 指数退避（镜像 `OpenAIChatModel`）；非 200 直接 `ModelError`。

### 四级评估纪律
```python
Rsi(config, model, *, train, val, test,
    judge=None, improver=None, gate=None, segment_gate=None,
    storage_dir=None, n_rollouts=1, max_rounds=2, val_every=2).run()
```
- **train**：轮级 Gate 判 promote/reject（只看 train，聚合升且最差 task 不回归）。
- **val**：每 `val_every` 轮评一次，`SegmentGate` 对 `watermark` 做**严格双方提升**：
  `Δtrain > train_min_delta` **且** `Δval > val_min_delta` → commit（血线前移）；否则 rollback / hold。
- **test**：`run()` 末尾**恰好评一次**，报泛化；从不进 train/val。
- `RsiReport{baseline_train, baseline_val, rounds[], segments[], final_version, final_config, test_score}`。

## 3. 本课做到 / 没做到

**做到**：async 评分路径（`evaluate` await judge、model 与判分解耦）；Judge/Locator/Proposer 三套双后端
统一接口（规则 / chat LLM / System One 决策模型，各持依赖）；`MergedLLMImprover` 把定位+改写合并为一次生成；
train/val/test 拆分 + 轮级/段级两级门禁 + 血线（commit/rollback/hold）；test 只评一次；`n_rollouts` 全 split
对称；`load_split` 接自建探针集；规则后端完整保留（离线 L0 全绿）。L1 冒烟：真 deepseek 跑通
「模型判分 → 模型定位 → 规则改写 → 两级门禁 → test 一次」，System One 决策 API 试连成功。
**没做到（留给后续）**：可变面仍只有 prompt（ch10 skills、ch11 memory、ch12 agent 结构）；`LLMProposer`
产出的是追加规则，不做整段重写/结构搜索（ch12）；评估器/定位器本身还不被改进（ch13 meta）；没有自动化
benchmark 跑分（L2）；并发评估仍留在后端轨（边界注入）。

## 4. 关联八股（~14 题）

1. **RSI 为什么是二阶 Agent / meta-agent**？一阶的 act/react 与二阶的 act(=rollout)/react(=改组件)如何同构？
2. **为什么说难点是评估+定位，不是改进算子**？把「改 prompt」换成「改 skills」时，不变的东西是什么？
3. **LLM judge vs 决策式模型**：各自适合什么？为什么 System One 只做判断、生成仍要 chat LLM？
4. **LLM judge 的偏差**：同一条回答规则判分 0.33、LLM 判 1.0——judge 宽松/位置偏差/自洽性怎么防？
5. **train/val/test 为什么必须拆**？不拆会怎样背题？"test 只碰一次"保护的是什么？
6. **train/val/test 与 iid**：三条集同分布、按构造不相交，为什么比随机切分更可靠（探针集按失败模式设计）？
7. **血线（watermark）**：为什么段级门禁要对比血线而不是上一轮？rollback 时复用血线分数省了什么？
8. **严格双方提升**：为什么要求 train 与 val **同时**超过血线？只升 train 会过拟合什么？
9. **轮级 vs 段级门禁的分工**：为什么轮级只看 train（便宜、高频），段级才付 val 的成本？
10. **n_rollouts 对称性**：基线评 1 次、候选评 N 次为什么会高估候选？统一 n 修掉了什么偏差？
11. **合并 locate+propose（thought+act 一次生成）**：相比两阶段少什么、代价是什么（可控性/可观测性）？
12. **不可变 config**：为什么改写永远返回新对象？（版本追溯 / 防同一候选被反复当基线 / 可回滚）
13. **规则优先、模型低频**：为什么默认后端是规则，只在 L1 显式接模型？（成本、确定性、可测性）
14. **评测污染隔离**：agent 只见 `statement`、verifier 才见私有 `checks`；如果 rollout 把 checks 喂给模型会怎样？
15. **async 统一**：循环内顺序 `await`、不并行；并发 run 为什么放"调用 Agent 的边界"（后端轨 ch16）？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/rsi.py` 仍是 **seam**（`propose/verify/gate` 三个设计面，无实现）——本章继续做洁净室镜像：
  环（reward→locate→propose→gate）不重写，把 **judge/locator/proposer 换成可插后端**，把 **评估纪律**做成
  train/val/test + 两级门禁。
- System One 决策评估：`/home/tiger/docs/models/ark_decision.md`（beta；头 `ark-beta-systemone-api: true`；
  `noul/choice/score` 三类问题；按输入 token 计费；只判断不生成）。
- 探针集：仓库根 `dataset/README.md`（verifier-first、按失败模式前缀 `fmt/multi/num/hard`、train/val/test 不相交；
  `uv run pytest dataset/ -q` 校验 schema）。

## 6. 运行与验证

```bash
# L0 离线（零额度）：async 判分接线 / LLM 定位抠 JSON / SystemOne 定位映射 / 合并改进 /
#   轮级+段级门禁 / test 只评一次 / 无 rubric 泄漏 / 规则后端不回退 / SystemOne 客户端重试 / 数据集加载 /
#   两级门禁端到端（commit 与 rollback）
uv run pytest course/ch09_prompt_rsi/ -q
uv run pytest dataset/ -q

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch09_prompt_rsi/ -m smoke -s
#   → 真 deepseek：LLMJudge 判分 + LLMLocator 定位 + RuleProposer 改写，两级门禁给 verdict，test 评一次
#   → System One 决策 API 试连（beta 不可用/模型未开通则自动 skip）

uv run ruff check course/
```

> 注：轻松探针集上真模型可能 baseline 全 1.0 → 第一轮 noop、段级 hold——这是**合法终态**（无改进空间就不改，
> 不为了"看起来动了"而接受噪声）。想看到 promote/commit/rollback，L0 的两个端到端用例是确定性驱动的。
