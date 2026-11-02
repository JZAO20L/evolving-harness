# ch00 · Agent 软件工程设计（架构总览）

> 定位：**动手写码之前先立的架构梁**。它不是某一轨的 hero 文件章，而是贯穿 Agent / Backend 两轨的工程设计第 0 章。
> 一句话主线：**Agent 是一层抽象（port），不同服务/厂商是最外层可插拔的 adapter；接口切换只动外沿，核心逻辑与依赖方向始终保持"向内指"。**

---

## 0.1 核心心智模型：Agent 抽象 + 多 adapter（端口 & 适配器 / hexagonal）

```
       CLI           REST/SSE       WebSocket      WebUI         MCP/别的 Agent
        │                │             │             │                │
   ┌────┴────────────────┴─────────────┴─────────────┴────────────────┴────┐
   │       传输/服务适配层（可插拔、薄）：HTTP·SSE·WS·gRPC·stdio·MQ·MCP       │
   │              unify 成对下面的"内部契约 / port" 调用                        │
   ├───────────────────────────────────────────────────────────────────────┤
   │        Agent core（抽象，与接口无关）                                     │
   │        run() / stream() / cancel() / status() / inbox()                │
   │        ├─ think-act-observe 循环（ReAct）                              │
   │        ├─ tool 注册表 + 执行器                                          │
   │        ├─ memory / context 管理                                         │
   │        └─ 模型客户端（vLLM / API）—— 经 port 注入                         │
   └───────────────────────────────────────────────────────────────────────┘
```

- **接口不是 Agent 的一部分，接口是 Agent 的"插座"**：core 只暴露一组固定的方法签名（内部契约）`run/stream/cancel/status`，CLI/FastAPI/WebUI/别的服务只是把这组方法映射到不同传输协议上的薄 adapter。换接口 = 换 adapter，**core 一行不改**。
- 拿 Python 说：core 里只 `from typing import Protocol` 定义抽象，**不 import 任何 SDK/框架**；具体模型客户端、存储、工具、Web 框架都在 adapter 层实现。

---

## 0.2 完整服务架构：五层 + 横向三条

> 上节是"逻辑与依赖的切法"，这节是"一个真实 Agent 服务跑起来要哪些层"。

```
 ① 客户端层 client             WebUI · CLI/TUI · 别的服务 SDK · 别的 Agent/MCP
                                  │   "谁在使用，只在自己那端渲染输入输出"
 ② 传输层 transport           HTTP REST · SSE · WebSocket · gRPC · MQ · stdio
                                  │   "字节怎么在客户端与服务端之间流动"
 ③ 后端服务层 backend        入口路由 · 鉴权 · task/会话状态机 · 幂等 · 限流 · 观测 · 编排调度
                                  │   "谁来承接请求、管理生命周期"
 ④ Agent core（领域/抽象）     think-act-observe 循环 · tool 执行器 · 记忆 · 模型客户端 · 契约 port
                                  │
 ⑤ 底座（基础设施/模型服务）  vLLM 推理 · 向量库 · KV 缓存 · 工具服务
                                  │
 ─── 横向基础设施（被 ③④ 持有引用，补"时间轴 + 状态"）─────────────────────
   Cache(Redis)：会话态 / 幂等键 / 限流计数 / RAG&模型结果缓存（快、可丢）
   DB         ：task/会话/用户/审计 持久化、Checkpoint 断点（慢、可靠）
   MQ         ：长任务下发解耦、削峰、可恢复、回调下游（异步、不丢）
```

三层关键结论：

1. **接口 adapter 横跨 ②+③**：换接口 = 同时看传输(②)与服务(③)两层，**core(④) 不碰**。
   - 如 Gradio 是 ①客户端库，底下走 ②WebSocket，去调 ③里的 FastAPI 路由，路由再碰 ④core。
   - 别让"WebUI"和"API/CLI"平列成三种入口——WebUI 是**接在传输之上的客户端**，它走 WS/SSE 就是了，不算第三种通道。
2. **"流式/异步还是阻塞"是接口层的属性，不进 core**：core 产出事件/增量（`yield`），由 adapter 决定透传成 CLI 打印 / SSE `data:` / 轮询聚合。
3. **后端服务层(③)与传输层(②)别混**——⑧把"请求接进来"、"管生命周期"（task_id 幂等 + 状态机 + 检查点）归 ③，"决定用哪种协议把结果流出去"归 ②。这是"demo vs 上线"的分水岭。

横向三条的一句话分工：
> 缓存负责"会话里快取可丢"，DB 负责"跨请求可靠的持久化 + Checkpoint"，MQ 负责"长任务解耦 + 可恢复"。三者分别补上空口流模型缺的"状态与时间轴"。

---

## 0.3 传输通道清单（一个 core、多出口）

| 通道 | 形态 | 适用 | 选型一句话 |
|---|---|---|---|
| **CLI/TUI** | stdio 收输入、stdout 吐事件 | 本地单用户、本地 agent（Claude Code 即此类） | 朴素不依赖服务，本地实验默认形态 |
| **HTTP REST** | request↔response | 程序化调用 | 配`task_id`轮询拿长任务结果 |
| **SSE** | 服务器→客户端单向流（同走 HTTP 一条连接） | token/进度条流式 | 打字机效果，如`EventSource` |
| **WebSocket** | 全双工 | 前端双向低延迟：边发指令边收流 | Gradio 内部；打断/补充靠它 |
| **gRPC/Thrift** | 内部服务间、强类型、双向流 | 大厂内部 agent 服务被别的服务调 | 高性能，内部链路标配 |
| **MQ** | 异步队列 | 批量/长任务/削峰/可恢复 | 生产级 10w+ 评测的扛法 |
| **MCP** | JSON-RPC（stdio 或 HTTP/SSE） | Agent↔工具 / Agent↔Agent 标准化 | "agent 的 USB-C"，Claude Code/Cursor 都当 client |

> 面试落点：把上面这表讲成"一个 core、多出口"，每个通道标"干什么、何时选"，比单讲 REST/CLI 显功底。

---

## 0.4 参考结构对比：domain/app/adapters 三层

常见教学/生产结构把 core 再拆两层 + 一层实现：

| 层 | 是什么 | 是否依赖外部 |
|---|---|---|
| **domain**（实体层） | `Session/UserMessage/ToolCall/Role` 等**纯数据结构** | 不依赖任何东西 |
| **app**（用例/编排层） | `AgentService` 业务编排 + `ports`（`ModelClient/SessionStore` 等**抽象接口**） | 只依赖抽象（DIP） |
| **adapters**（实现层） | `OpenAIClient/AnthropicClient`、`JsonlSessionStore`、`WebApi`、`BashTool` | 实现接口，依赖 domain/app |
| **main.py**（组装根） | 唯一依赖注入处：new 具体适配器注入 AgentService | — |

价值：① 依赖倒置贯彻干净（依赖方向向内）；② 想换模型/换存储只改 `main.py` 一行；③ 给 `ModelClient` 写 mock 就能不烧 token 测编排；④ 有 `ToolCallReviewer`（角色审查/纠错）这类"工具调用审查"的加分设计。

但有边界：这结构**立起了壳、没立起会跑的心跳**——缺真正的 Agent 主循环运行时（stream/inbox/可打断/可恢复）、缺后端服务层与传输层的拆解、缺横向 DB/MQ/Cache。作教学骨架好，作生产要补 0.2 的层。

---

## 0.5 分层与过程式表达：正交，别混淆

> 分层管"逻辑如何不依赖外部"，过程式管"逻辑如何在内部表达"——两件**正交**的事。

- 分层（hexagonal）是为了**可替换性**：换模型/换存储/可 mock 测试。它不取消过程式——`AgentService.run()` 完全可以是几百行命令式 while/if/状态机，只要用的是抽象 port 就行。
- 复杂逻辑的难，来自**状态到处 + 副作用到处 + 顺序敏感**，不来自要不要对象。分层帮你隔离了**外部副作用**（模型/DB/工具都从 port 进来），但 core 内部状态要你自己管：

| 复杂度 | 组织方式 | 适用 |
|---|---|---|
| 低 | 过程式函数 + 少量数据结构 | 简单 agent / review 流程 |
| 中 | **显式状态机**（`phase` 字段 + `next(phase,event)` 转移表） | think-act-observe、rubric 迭代 |
| 高 | **流水线/Worker 编排 + 副作用隔离**（决策与 LLM/工具副作用拆开，副作用统一走 port） | SESS 技能进化、多 Agent、JudgeForge 三路径 |

- 一个现实事实：**复杂 Agent（Claude Code 类）的核心编排，现实里就是命令式主循环 + 事件状态机**（`while: ev = await inbox.get(); match ev: ...`），对象封装主要在端口/工具注册表/状态容器这类边界上。
- 两个反模式要避开：① 把每个逻辑点硬包成对象（过度设计，读代码跳 5 层）；② 因分层就放弃过程式（没必要，core 内部尽情过程式）。
- **错误的是"由外在内硬套分层"，而不参与"该过程式仍过程式"**。记住这句：**对象管边界，过程管逻辑**。

---

## 0.6 分层决策判据：何时两层 / 何时三层 / 何时单文件（收尾）

> 前面讨论里的"结论段落"。"两层就够"的实质：**依赖方向向内 + 可测试 + 换厂商，这些两层已经完整做到**。

- **为什么两层就够换厂商**：`core 定义 Protocol + 写业务`，`adapters 实现`。注入 `OpenAIClient` 还是 `AnthropicClient` 都只改组装处，core 零改动。这条核心诉求两层已闭环。
- **第三层（app/ports）为什么存在**：它不服务"换厂商"，而是把 core 再拆成"领域实体(domain) + 业务用例(app)"。价值只在：**① 多个 Agent/用例复用同一批端口；② 编排逻辑厚到值得和实体分开测**。若只是把 `ModelClient.stream()` 抄一遍签名，它就是空壳、是过度设计。

**判据表单**：

| 情境 | 推荐 |
|---|---|
| 单个 Agent、少数端口、就为换厂商 / 可测 | **两层**（core + adapters）即可 |
| 多个 Agent / 用例共享同一组 `ModelClient/SessionStore/Judge` | 三层或把 ports 独立出 app/app |
| 编排逻辑很厚、要和实体分开测 | 三层（domain/app/adapters） |
| `ports.py` 只会把接口签名再抄一遍 | 砍掉那层，是空壳 |

**研究型 vs 正式项目的拆别**：
> 研究型 Agent（如 evolving-harness 前段）：**单文件 + 命令式状态机**写复杂逻辑即可；**只把 ModelClient 与 Tool 这两条高变动缝薄抽成 adapter**（换模型/可 mock 两件事，成本几乎为零）。正式项目则在出现"**多端消费 / 多用例共享端口 / 需可替换验证 / 复杂逻辑与对外暴露纠缠**"任一情形时，再长成「抽象逻辑层 - 具体实现层」两层（架构不够才补第三层）。

**一句话收尾**：
> 分层是为**变化**付费，不为**仪式**付费。核心先守住两件事——**依赖方向向内**（domain/app 不碰 SDK/框架）、**模型与工具两个高变动接口薄抽成 port**；复杂逻辑在 core 内部尽情用状态机 + 过程式表达。等代码被多人/多服务/多用例消费，再决定要不要长 layer。**对象管边界，过程管逻辑；需要时再分，不预先透支。**