# course · 逐章递进的教学实现（项目从 0 到 1）

把 `evolving-harness` 重定位为**教学项目**：每一章 = 一个教学单元，在上章基础上增量改动。
`course/` 内的代码与文档是**交付物**（进 git）；`docs/` 下的 dev-aid 文档仍保持本地不发布。

## 0. 仓库视角：course 与 harness 两个一级目录

仓库根下并列两个一级目录：

- **`course/`**（本目录）—— 教学的"项目从 **0 到 1**"：逐章自包含的教学内容与最小实现。
- **`harness/`** —— 产品完整工程。**等 course 开发完后**，把散在根的 `evolving_harness/` + `agents/` + `tests/` 等归拢整理进这里，形成可交付的产品。

课程期间**只动 `course/`**；产品代码保持现状，不折腾。

## 1. 定位与原则

- **章节递进**：每章一个自包含目录 `chNN_slug/`，只在前章基础上增量；目录内容**完整可运行**，可 `git tag chapter-NN-*` 指到"该章已落成"的提交。
- **每章只带用到的 hero 文件**：为避免代码量爆炸，每个 chx 只放**本章真正用到**的 hero 文件，不带用不上的。例如 **Agent 轨每章只有一个 `agent.py`**（LLMCall → loop → tool → … 逐章长进）；后端轨才带 `backend.py`；RSI 轨带 `agent.py` + `rsi.py`；前端轨 `frontend.py`。
- **md 是交付物**：每章 md =「用的工具 | 实现的对象 + 用法 + md 内示例代码 | ~10 道关联八股」。
- **复用 + 回填**：产品（`evolving_harness`，course 完成后成为 `harness/`）已实现的后半段（loop/tools/subagent/memory/RSI 骨架），教学章不再重写，由 md 串引现有产品实现当"生产版延伸阅读"；真正缺的章节才在 hero 文件里放干净实现。
- **并发与健壮性都放边界，不放循环**（贯穿原则）：
  - Agent **内部是"顺序的 async"**：ReAct 循环每步依赖上一步，intra-turn 不显式并行；`await` 让出事件循环即可。
  - **并发 run 在"调用 Agent 的边界"注入**（多个对话/run 在同一个 event loop 上交错 = 后端轨 ch15）。
  - **重试 / 指数退避 / 异常分类归模型适配器**（ch01 的 `ChatModel` 实现）；循环只看到"结果 or `ModelError`"。
  - 不在循环里每个函数调用点打散并发 / 重试 / MQ 逻辑（同 v0.3 划掉"处处 Redis/MQ"的反模式）。

## 2. 结构

```
course/
  README.md                  # 轨序、约定、进度、章节索引
  ch01_llm_call/             # 每章 = 自包含目录
    agent.py                 # hero：Agent 轨只有这一个文件，逐章长进
    README.md                # 教学单元 md
    tests/                   # 本章测试（或 test_agent.py）
  ch02_agent_loop/
    agent.py                 # = ch01 agent.py + loop 增量，完整可运行
    README.md
    tests/
  ch03_tool_use/ ...
  ...（agent 轨 ch01–ch07）
  ch08_rl_loop/              # RSI 轨：带 agent.py（被改进的对象）+ rsi.py
    agent.py
    rsi.py
    README.md
    tests/
  ...（rsi 轨 ch09–ch13）
  ch14_fastapi/ ...          # 后端轨：带 backend.py（+ 需要的 agent）
  ch21_gradio/               # 前端轨：frontend.py
```

- `[复用+回填]` 章节：md 为主，hero 文件放**最小可运行版**，另在"生产版延伸阅读"串引 `evolving_harness` 实现。
- 章节编号**顺着轨序统一递增**（agent ch01–07 → rsi ch08–13 → backend ch14–20 → frontend ch21），便于"第 N 章 = 第 N 堂课"。

## 3. 轨序（锁定）

**Agent(7) → RSI(6) → Backend(7) → Frontend(1)**

- 能力先于交付：RSI 是本项目主线论点（harness 自进化），先证能力、再包壳。
- RSI 复用 Agent 循环的"可观察 / 可中断 / 可挂钩子"接缝；紧邻放置 = 一个抽象服务两轨。
- Backend 的 SSE/WebSocket 接 Agent 轨的 LLMCall 流式（跨轨接缝，详见各章目标）。
- 轨序只决定**教案编写与 tag 推进次序**；每章目录可先建空壳，再按轨序逐章填充内容。

## 4. 每章目标

> 每节：`本方英雄机对象`（本章 hero 文件的增量）· `关键能力` · `八股面试面`。`[复用+回填]` 标记章节以 md 为主。

### Agent 轨 —— 各章 hero = `agent.py`

> **贯穿范式（ReAct）**：循环检测的是 **act（一次动作选择 + JSON 参数）**，而非仅 tool_call。
> `toolcall` / `skilluse` / `subagent`（及 plan、memory 写）都是 act 的种类；循环解析 act → 分发 → 观察回填。
>
> **act 表示（分步）**：ch02 先用**纯 JSON act**（模型输出里带 JSON，循环解析，讲清 ReAct 概念）→ ch03 升级为**原生 function calling**（工具各注册 function，act=function call，保证结构）。

| ch | 章 | 实现对象（agent.py 增量） | 关键能力 | 八股面 |
|---|---|---|---|---|
| 01 | llm-call | 消息类型 + 模型适配器（chat/completions）+ usage 记账 | 一次 LLM 调用、角色/function 协议、token 统计 | completion vs response；token；temperature；system/user/assistant；function calling；stop；context window |
| 02 | agent-loop | 循环：组装 context → 调模型 → **检测 act（action_type + JSON params）** → 判定停止/继续 → 步数/usage 记账 | 把单次调用变成"循环"，ReAct 的 act 抽取 | 循环 vs 单次调用；ReAct（Thought/Act/Observation）；JSON act 解析；停止条件；max_steps |
| 03 | tool-use | `act:tool` 注册表 + act 分发 → 执行 → 结果回填；**错误当数据** | 模型驱动工具调用闭环 | function-calling 协议；为何错误当数据；tool schema/tool_choice；prompt 注入 |
| 04 | skills | `act:skill`+ skill（名+desc+指令）+ list/load + 渐进披露（priority + 目录截断） | 把可复用指令结构化、按需加载 | vs system prompt；目录 vs 全量注入；可维护性 |
| 05 | memory | L1/L2 笔记 + 上下文预算/压缩 + 注入 cap | 让长任务不撑爆上下文 | 短/长期记忆；token 预算；压缩陷阱；写保护 |
| 06 | plan | 多步规划：拆任务 → plan-then-execute；失败重规划 | 结构化拆解长任务 | plan vs 直接做；失败重规划；依赖/拓扑；token 成本 |
| 07 | sub-agent | **`act:subagent`** 委派子 run（独立 context）、等待/深度上限 | 树状执行、上下文隔离 | 为何子代理；隔离 vs 共享；并行 vs 顺序；编排 |

### RSI 轨 —— 各章 hero = `agent.py` + `rsi.py`

| ch | 章 | 实现对象（rsi.py 增量，改 agent.py） | 关键能力 | 八股面 |
|---|---|---|---|---|
| 08 | rl-loop | 外环：task set → verifier 裁决 → 提议改进 → 门禁 + 版本登记 | 无训练 RL 式自改进闭环 | RL vs training-free；reward/verifier；自改进；评测污染；版本/回滚 |
| 09 | prompt-rsi | 按 verifier 反馈优化 system prompt，测 delta | 把 prompt 当可优化对象 | prompt 工程 = 优化；过拟合；baseline |
| 10 | skills-rsi | 演化 skills 集（增/删/改写）对评测 | skill 作为演化单元 | skill vs prompt；技能膨胀 |
| 11 | memory-rsi | 演化记忆策略（存什么/留多久/压多少） | 记忆策略可调参 | 记忆策略优化；遗忘/保留 |
| 12 | agent-rsi | 演化 agent 结构（loop 参数/工具集/limits） | 结构搜索 + 护栏 | 结构化搜索；任务过拟合；护栏 |
| 13 | meta-rsi | 改进 RSI 过程本身（proposer/门禁） | 元层、复合自改进 | 元学习；能力复合；涌现 |

### Backend 轨 —— 各章 hero = `backend.py`（需要时并附 agent）

| ch | 章 | 实现对象（backend.py 增量） | 关键能力 | 八股面 |
|---|---|---|---|---|
| 14 | fastapi | 路由 + pydantic 模型，HTTP 跑一个 agent | 把循环包成 API | REST 语义；请求/响应模型；错误码 |
| 15 | concurrency | async/threadpool + 作业队列 + worker + 并发上限 + per-run workspace | 并发服务 | async vs 多线程；任务队列；全局限流；进程/线程模型 |
| 16 | sse | 服务器→客户端流式事件（接 ch01 llm-call 流式） | 单向流、事件序 | SSE vs 轮询；EventSource；断线重连 |
| 17 | websocket | 双向流：cancel/approve 控制 | 双向实时控制 | WS vs SSE；心跳；反压 |
| 18 | sql | Postgres/MySQL 存 State 动态段 + trace 索引 + 版本登记（替文件 Checkpointer） | 关系存储、事务 | 索引设计；事务/ACID；ORM vs raw SQL |
| 19 | redis | 缓存 / 队列 / 频控（替 asyncio 队列、会话） | 外部中间件 | 缓存穿透/击穿/雪崩；Redis Streams；原子性 |
| 20 | docker | 服务打包 + 容器沙箱池 | 部署 + 工具隔离 | 镜像分层；网络模式；沙箱边界 |

### Frontend 轨 —— hero = `frontend.py`

| ch | 章 | 实现对象（frontend.py 增量） | 关键能力 | 八股面 |
|---|---|---|---|---|
| 21 | gradio | Gradio WebUI：运行 / 流式 / trace / 技能 diff 审批（对标典型 GPT/Qwen 界面） | 最简前端闭环 | 前端-后端交互；流式渲染；状态回填 |

## 5. 进度（逐章勾选）

```
Agent
  [ ] ch01 llm-call    [ ] ch02 agent-loop  [ ] ch03 tool-use
  [ ] ch04 skills      [ ] ch05 memory      [ ] ch06 plan
  [ ] ch07 sub-agent
RSI
  [ ] ch08 rl-loop     [ ] ch09 prompt      [ ] ch10 skills
  [ ] ch11 memory      [ ] ch12 agent       [ ] ch13 meta
Backend
  [ ] ch14 fastapi     [ ] ch15 concurrency [ ] ch16 sse
  [ ] ch17 websocket   [ ] ch18 sql         [ ] ch19 redis
  [ ] ch20 docker
Frontend
  [ ] ch21 gradio
```

## 6. 约定

- 每章一个自包含目录，hero 文件**只放本章用到**的（Agent 轨每章只有 `agent.py`）。
- **每章的 hero 文件 = 当章开发内容 + 前序章已开发内容**（自包含快照，完整可运行）。如 RSI 章带 `agent.py`+`rsi.py` 两个 hero；后端章"需要时并附 agent"。不引用前章目录（避免跨目录耦合），改动随快照复制。
- `[复用+回填]` 章节：md 为主，hero 放最小可运行版，并在 md 里串引产品实现当延伸阅读。
- 每章带配套测试，保证「新增一行能力必有一行测试」；涉及模型的测试用假实现（Mock/脚本化响应），离线可跑。