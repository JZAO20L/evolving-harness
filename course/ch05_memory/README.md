# ch05 · 记忆（memory）

ch01~ch04 让 agent 能「调模型 / ReAct 循环 / function calling+工具 / 技能渐进披露」，但每次
`run` 的 messages 都是**一次性**的：上一轮的结论、用户偏好、tool 结果全丢，多轮不连贯。本课再往
上扣一层：**分层记忆**。依据用户 4 点设计（对齐生产 `docs/v0.1/DESIGN_v0.1_agent_core.md`
§7.2/§7.3 的记忆总原则）——

1. **用户画像（长期记忆）**：模型总结 + 维护固定 schema + **规则信号门控**，轮末折叠进 profile，常驻 system。
2. **工作记忆（短期记忆）**：就是循环的 `list[Message]`（四类 role）；tool 结果进内存前**规则简写**；
   维护 `max_len × threshold` 预算，超了**模型改写压缩**（旧前缀折叠成摘要，留最近几条原样）。
3. **完整记忆落盘成文件**：每轮完整 messages 追加成 **JSONL**（`memory.jsonl`，一行一条消息）；context 里没有用户提的词、却在文件里
   出现过 → 循环**自动回填**（grep 式取片段注入）。
4. 工作记忆**以 messages（system|user|assistant|tool）形式**存于循环。

文件：`agent.py` = ch04 原样 + 新增 `Memory` 子系统（token 估算/预算/规则简写/模型压缩/画像/落盘回填），
`AgentLoop` 加一个可选 `memory=` 挂钩，不给行为保持 ch04 原样。顺延 ch02 关键设计：**agent 不是跑完才理人**
（`input_queue` 运行期用户输入注入，每步边界优先处理，先于记忆挂钩；见 ch02 §0.5）。
`skills/` 顺延自 ch04（自包含，非本课重点）。

---

## 0. 流程图

```
            用户画像(长期记忆) :: system 常驻
            schema{identity/domain/preferences/style} + 规则门控(PROFILE_CUES)
                    ▲ 轮末 maybe_update()（命中 cue 才调模型总结）
                    │
       完整记忆(文件) ◄──────────────每轮 finalize() 追加 memory.jsonl
       MemoryStore.path                     │
                ▲                          │
    召回: context 缺词 + 文件命中          │
    (grep 式按行) 自动回填注入 system       │
                │                          │
   context = [system(工具清单+画像), user]
                    │
                    ▼
   ┌─────────────────────────────────────────────┐
   │  for step = 1..max_steps:                    │
   │   memory.hook_pre_step(context)  ◄──────────┐│
   │     ① 首次缺词→store.recall 注入回填         ││
   │     ② wm.is_over()→ 模型改写压缩            ││
   │        旧前缀(除最近) → 1 条 system 摘要     ││
   │   complete(context, tools=schema)           ││
   │   └─ tool_calls? ──yes──► 执行+观察回填      ││
   │   └─ no ─► 答案即最终，收尾                 ││
   └─────────────────────────────────────────────┘
              │  工作记忆(临时)：tool 进内存先规则简写(abbreviate_observation)
              ▼
   finalize(): 落盘完整记忆 + 画像按需更新
```

## 1. 用的工具

| 层 | 机制 | 角色 |
|---|---|---|
| 长期记忆 | `UserProfile` + `PROFILE_CUES` 规则门控 + 轮末 `maybe_update` | 常驻 system 的用户画像；模型低频总结、按需触发 |
| 短期记忆 | `WorkingMemory` + `abbreviate_observation` 规则简写 | tool 结果进内存先低成本"提取+简写"，保短时精简 |
| 预算/压缩 | `estimate_tokens`（启发式）+ `ContextBudget` + `compact_messages` | 超 `max_len×threshold` 用**模型改写**折叠旧前缀成摘要 |
| 完整记忆 | `MemoryStore`（`append_turn` / `recall`） | 每轮 messages 落盘 `memory.jsonl`；缺词自动回填片段 |
| 聚合 | `Memory`（`overhead` / `hook_pre_step` / `finalize`） | 给 `AgentLoop` 一根记忆缝，循环只多几行 |

## 2. 实现的对象 + 用法

### token 估算与预算
`estimate_tokens(text) = max(1, ceil(len/3))`——中英混排折中启发式，零依赖，**压缩前就能算**（真实
usage 只有调用后才有）。粗糙度接受，校准留给 trace（DESIGN：M1 校准）。`ContextBudget(max_len,
threshold, keep_recent)`：`max_len×threshold` 为压缩触发线，`keep_recent` 是压缩时保留的最近条数。

### 工作记忆 = messages + 规则简写
`WorkingMemory(budget)`：

- `ingress(msg)` 规整一条消息：`role=="tool"` 且非空 → `abbreviate_observation`（压空白、折叠行、
  超 `max_chars` 截断 + 原文长度标注）。返回规整后的消息，循环据以挂载。
- `resync(context)` 让内部 `messages` 始终等于当前 context（回填注入 / 压缩后同步）；
- `size()` / `is_over()` 供预算判断（触发压缩）。

工作记忆 = 循环的 context，**同一份** list[Message]，四类 role——这正是 spec 第 4 点。

### 模型改写压缩
`compact_messages(model, messages, budget)`：把**除最近 keep_recent 条外**的旧前缀做一次模型
`complete`，改写成一条高密度中文摘要（`Message(system, "【工作记忆压缩摘要】\n…")`）；`tools=None`
（这是摘要，不是动作）。新 messages = [摘要] + recent 尾原样保留。规则优先（简写免费），模型只在
**超预算**这一低频时机出场——成本和保真度的取舍（DESIGN §7.3）。

### 用户画像（长期记忆，规则门控 + 轮末模型总结）
`PROFILE_CUES` = 一组规则信号（"我是"、"我喜欢"、"偏好"、"领域"…）。`has_profile_cue(messages)`
便宜地看本轮 user 文本有没有可能写进画像。`UserProfile(model, schema=..., path=...)`：

- `into_system()` → 常驻 system 的 `你是…用户画像:<json>`；
- `maybe_update(messages)`：**门控命中才问模型**，喂「old data + schema + 本轮 messages」，解析返回
  JSON 存回并落盘 `profile.json`；未命中**零模型调用**。

### 完整记忆落盘 + 自动回填
`MemoryStore(dir)`（传 None → tempfile，不入库）：

- `append_turn(messages)` 把一整轮 messages 追加成 **JSONL**：一行一个 JSON 对象（一条 message），
  同轮共享一个 `turn` 时间戳，内容角色/tool_call_id/tool_calls 结构化字段可恢复；
- `recall(context)`：取最新 user 文本的可检索词（对中文做 2 字 bigram 近似，ASCII/技术词独立抽出；
  停用词过滤），若该词**不在已有 knowledge（除去本次提问）**里、却**曾在 memory.jsonl 出现**，
  逐行 `json.loads` 命中 content 取片段返回 `[从完整记忆检索到] …`；否则 None（不误召回）。

`Memory(model, *, budget, schema, storage_dir)` 聚合：

- `overhead()` 画像常驻 system 段；
- `hook_pre_step(context)`：① 首次 + 用户缺词 → `store.recall` 注入 system 首条（自动回填）；
  ② `is_over()` → 压缩。
- `finalize(context)`：落盘完整记忆 + 画像门控更新。

> 压缩时 `hook_pre_step` 对传入的 context **就地替换**（`context[:] = 摘要+recent`），调用方的
> 循环 context 才会一起变——传 list 进去重新绑定局部名不改调用方（易踩的坑）。

### `AgentLoop(model, …, memory=None, …)`
ch04 循环加了**一根记忆缝**：给了 `memory=` 就——

1. system prompt 末尾追加 `memory.overhead()`（画像）；
2. 每步 complete 前 `await memory.hook_pre_step(context)`（回填/压缩）；
3. 工具观察经 `working_memory.ingress` 挂载（tool 按规则简写）；
4. 收尾 `_finish` 把最终答案也作为 assistant 消息交给 `finalize`（结论是记忆的一部分），落盘 + 画像更新。

不给 memory = ch04 原样（向后兼容）。

## 3. 本课做到 / 没做到

**做到**：三层记忆（画像/工作记忆/完整记忆）成体系落地；token 启发式 + 预算 + **规则优先、模型低频
重写**的压缩；画像**规则门控按需触发**（无信号零模型调用）；完整记忆落盘 + **循环内置自动回填**
（跨 run 第二次提问自动取回）；工作记忆以 messages 形态存于循环；`AgentLoop` 只用一根 `memory=`
缝接入、向后兼容；L1 冒烟用真 deepseek 验证**跨 run 回填 + 极小预算真实触发压缩**。
**没做到（留给后续）**：真分词（CJK bigram 是近似，ch07 子代理/产品用真 tokenizer）；文档/概念
特征化召回（目前按词 grep，无稠密检索）；压缩的保真度评测与遗忘策略（ch11 memory-rsi 演化"存什么/
留多久/压多少"）；记忆与技能联合调度（ch10 演化），以及评测关 memory 防状态漂移（DESIGN 明确要求）。

## 4. 关联八股（~14 题）

1. 长短期记忆怎么分工？画像(跨轮恒常) vs 工作记忆(本轮上下文) vs 完整记忆(文件)各保什么？
2. 为什么"规则优先、模型只低频重写"？简写/门控/落盘全免费，压缩/画像才付一次模型——成本怎么换保真度？
3. `estimate_tokens` 启发式为什么"压缩前"才用得上？真实 usage 为什么只有调用后才有？
4. 压缩的**触发线**为何是 `max_len×threshold` 而不是固定值？threshold 偏大/偏小各有什么症状？
5. 压缩**丢什么保真度**？旧前缀折叠成一条摘要，关键结论/工具结果可能被吞——多久压一次、留几条
   recent 是保真度权衡？
6. 画像为什么用**规则信号门控**而不是每次都总结？命中才付一次成本；漏掉 signal 的代价 vs 白付成本的代价？
7. 自动回填为什么"按词 + bigram 近似"？中文没有空格分词，2 字滑窗怎么找回跨轮片段？误召回怎么兜？（不在 knowledge 且文件命中才回）
8. 完整记忆**落盘的意义**：为什么多轮只靠 messages 不够，要落盘成文件？跨 run / 重启后靠它续上。
9. 压缩时"就地替换 context"为什么是必须的？只 rebind 局部名会发生什么（调用方还是旧列表）？
10. 评测为什么要求**关 memory 防 drift**？带上记忆的 agent 每次状态不同，想稳定复现评测怎么处理？
11. 工作记忆以 messages 存 vs 以结构化 dict 存，各有利弊？四类 role 怎么自然承载"读写过什么"？
12. `finalize` 把**最终答案也进记忆**，为什么？"它上一轮说过什么"对下一轮是很有价值的上下文。
13. 记忆与技能（ch04）在 memory 里的**角色差异**？技能=方法、记忆=事实状态，怎么互补？
14. 如果允许 Rch10/11 演化**记忆策略**（存什么/留多久/压多少），本课哪些参数是调参面？（budget 三件套、cues、bigram 窗）

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/agent.py` 的记忆愿景对应生产 `docs/v0.1/DESIGN_v0.1_agent_core.md` §7.2/§7.3：
规则优先、模型低频重写、`写入 .memory/`、评测关 memory 防 drift。产品版把记忆做成 **tracing 里的
session 上下文** + 可持久化 storage；教学版最小化为**画像门控 + 预算压缩 + 落盘回填**几条缝。真分词、
特征化召回、遗忘策略、可演化的记忆策略（ch10 演化 / ch11 memory-rsi / ch13 会逐步补齐）。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：token/预算/简写/压缩/画像门控/落盘回填/循环组合 全走 Mock
uv run pytest course/ch05_memory/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）：真 deepseek 跨 run 自动回填 + 极小预算真实触发压缩
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch05_memory/ -m smoke -s
uv run ruff check course/
```