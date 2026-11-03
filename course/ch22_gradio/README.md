# ch22 · gradio（前端轨）——最简 WebUI：运行 / 流式 / trace / 审批

后端（ch15–ch21）已经能通过 HTTP 提交任务、通过 WebSocket 推事件和接收审批。
本章写一个**只依赖协议、不 import 后端代码**的 Gradio 前端：

```
浏览器
  │  输入任务 / 点批准 / 点取消
  ▼
Gradio Blocks ──► BackendClient ──POST /v1/runs──► 后端
                      │
                      └──WS /v1/runs/{id}/ws──► trace 事件 + job.status + job.end
                                 ▲
                  approve/cancel │（必须复用同一条 WS）
```

三个前端核心问题：

1. **流式渲染**：WS 每来一条事件就更新 UI，而不是等整轮结束；
2. **状态回填**：Gradio 回调之间没有共享内存，必须把 `UiState` 放进 `gr.State`；
3. **审批闭环**：审批 Future 绑在 WS 连接上，**关掉连接就自动拒绝**，所以批准/
   拒绝必须走同一条连接，不能另开短连接。

## 0. 流程图

```
┌────────────── Gradio 会话（浏览器标签页）──────────────┐
│  gr.State: UiState（对话/ trace/ pending/ status）      │
│  gr.State: RunSession（同一个 run 的 WS 句柄）          │
│                                                        │
│  [运行] ──► start() ──► POST /v1/runs ──► run_id        │
│                    │                                   │
│                    └──► WS connect ──► async for msg    │
│                                           │            │
│                         apply_message(UiState, msg)    │
│                                           │            │
│              ┌────────────────────────────┴───────┐    │
│              ▼                                    ▼    │
│        Chatbot / Trace                  approval.request│
│                                                │       │
│                                   [批准]/[拒绝] ─┤       │
│                                                ▼       │
│                                   session.approve(id)  │
└────────────────────────────────────────────────────────┘
                    │ 同一 WS 发送 approve
                    ▼
              后端 RunControl.resolve → 工具继续执行
```

消息归约表：

| WS 消息 | UiState 变化 |
|---|---|
| `hello` | 记录 run_id，状态 `streaming` |
| `event` + `model.response` | 追加 assistant 文本 |
| `event` + `tool.call` / `tool.result` | 追加工具轨迹；错误标 `ERROR` |
| `event` + `run.end` | 记录 final / error |
| `job.status` | 更新状态徽标 |
| `approval.request` | 进入 `pending`，状态 `waiting_approval` |
| `approval.ack` | 移除 pending，回到 `running` |
| `cancel.ack` | `accepted=false` 时展示取消失败 |
| `job.end` | 终态 + final/error，清空 pending |
| `error` | 状态 `failed`，展示错误 |
| `heartbeat` / `pong` | 忽略 |

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `Gradio Blocks` | 页面布局与事件回调；可选依赖，懒加载 |
| `gr.Chatbot(type="messages")` | 对话与工具轨迹 |
| `gr.State` | 跨回调保存 `UiState` 与 `RunSession` |
| `httpx.AsyncClient` | 提交/查询 run；测试可注入 `MockTransport` |
| `websockets` | 前端 WS 客户端；`RunSession` 封装一条连接 |
| WS 协议（ch18） | 事件、审批、取消、心跳全部走同一连接 |
| `apply_message` | 纯函数状态归约，L0 可离线测 |
| `approval_preview` | 把工具参数渲染成 diff/命令预览 |

## 2. 实现的对象 + 用法

### 启动顺序

```bash
# 1) 装依赖：后端 + 前端
uv sync --extra dev --extra frontend

# 2) 起后端（真实 key 只进服务端环境）
set -a; source .env; set +a
uv run python course/ch21_docker/backend.py        # 默认 127.0.0.1:8000

# 3) 起前端
uv run python course/ch22_gradio/frontend.py \
  --backend http://127.0.0.1:8000 --port 7860
# 浏览器打开 http://127.0.0.1:7860
```

前端**不读取、不转发任何模型 key**；所有模型调用都在后端完成。

### 协议客户端（可脱离 Gradio 使用）

```python
client = BackendClient("http://127.0.0.1:8000")
job = await client.create_run("读取 README 并总结", run_id="demo-1")

state = UiState()
async with client.session(job["run_id"]) as session:
    async for current in stream_state(session, state):
        render(current)          # 每来一条事件渲染一次
        if current.terminal:
            break
```

- `ws_url` 把 `http(s)` 转成 `ws(s)`，保留反向代理的路径前缀；
- `parse_ws_message` 接受 `str/bytes/dict`，非对象 JSON 直接报错（协议违约）；
- `stream_state` 是 UI 与测试共用的消费循环，避免“测试一套、UI 另一套”。

### 状态归约：纯函数，便于测试

```python
def apply_message(state: UiState, message: dict) -> UiState:
    if message["type"] == "event":
        record = message["record"]
        state.trace.append(record)
        _apply_trace_record(state, record)
    elif message["type"] == "approval.request":
        state.pending[message["approval_id"]] = {
            "tool": message["tool"],
            "arguments": message["arguments"],
        }
        state.status = "waiting_approval"
    elif message["type"] == "job.end":
        state.status = message["job"]["status"]
        state.final = (message["job"].get("result") or {}).get("final", "")
    ...
```

- UI 只读 `UiState`，不直接解释原始 JSON；协议演进时改一处归约；
- `trace` 最多保留 500 条，渲染只显示最近 25 条，防止长任务撑爆浏览器；
- 工具结果与模型输出都做长度截断，避免一个 `yes` 把页面卡死。

### 审批：diff 预览 + 同连接回复

```python
async def decide(approval_id, approved, state, session):
    await session.approve(approval_id, approved)   # 同一条 WS
    state.pending.pop(approval_id, None)
```

`approval_preview` 的展示规则：

| 工具 | 预览 |
|---|---|
| `bash` | 完整命令（bash 代码块） |
| `write_file` | 路径 + `+ 新增行` 的 diff |
| 其他 | JSON 参数 |

- **为什么必须复用 WS**：后端在连接断开时调用 `RunControl.close()`，拒绝所有
  未决审批；新开一条 WS 无法解析旧连接的 Future；
- 多笔待审批时，批准一笔后自动显示下一笔（`render_approval` 取 `pending` 首项）；
- 审批超时由后端控制（默认 60s），前端只负责把结果送回去。

### Gradio 回调模型

```python
ui_state = gr.State(UiState())
session_state = gr.State(None)

run_btn.click(start, inputs=[task, run_id_box, ui_state], outputs=outputs)
approve_btn.click(approve,
                  inputs=[approval_id, ui_state, session_state], outputs=outputs)
deny_btn.click(deny,
               inputs=[approval_id, ui_state, session_state], outputs=outputs)
cancel_btn.click(cancel, inputs=[ui_state, session_state], outputs=outputs)
clear_btn.click(clear, inputs=[ui_state, session_state], outputs=outputs)
```

- `start` 是 **async generator**：每消费一条 WS 消息 `yield` 一次，Gradio 增量刷新；
- `_view` 统一产出 9 元组（对话 / 状态 / trace / 审批卡片 / approval_id /
  批准按钮 / 拒绝按钮 / `UiState` / `RunSession`）；没有待审批时两个审批按钮
  `interactive=False`，避免误点；
- `RunSession` 存进 `gr.State`，批准按钮才能拿到同一条连接；
- Gradio 默认按会话排队回调；流式生成器长时间运行时，审批回调仍可进入（同一
  session 的并发事件），因此两个回调都会改写同一个 `UiState`——本章接受这种
  “最后写入为准”，生产可换成显式事件队列 + 单 writer；
- 顶部有示例任务、`run_id` 输入框、运行/取消/清空按钮；trace 收进默认折叠的
  `gr.Accordion`，对话区固定 460px，长任务不会把页面撑爆；
- `build_ui` 不启动服务器，便于测试和嵌入。

### 错误提示：把异常翻译成可行动的下一步

`describe_error` 统一处理 HTTP 与网络异常：

| 异常 | 提示 |
|---|---|
| `404` | 任务不存在 |
| `409` | run_id 冲突，或任务运行在其他实例 |
| `429` | 请求过于频繁 + `Retry-After` 秒后可重试 |
| `503` | 后端暂不可用 |
| `ConnectError` | 无法连接后端，请确认服务已启动且地址正确 |
| `TimeoutException` | 请求后端超时，请稍后重试 |

`start` 在提交任务、建立 WS、消费事件三个阶段分别捕获异常并写入
`state.error`，状态面板用代码块展示；WS 断线则由 `RunSession` 先尝试续传，
预算耗尽后才冒泡到 UI。

### 断线、重连与多实例（本章的边界）

- WS 断开后 `RunSession` 自动重连，最多 `reconnect_attempts=3` 次（指数退避），
  每次携带 `after=<已收到的最大 seq>`，服务端只补发后续事件；
- 只有收到**真实事件**（`type=="event"` 且 `seq>0`）才重置退避和重连计数，
  避免服务端“能握手但不推进”时无限重试；
- 收到 `job.end` 后不再重连；重连预算耗尽则抛出最后一次连接异常，UI 展示友好提示；
- 审批/取消仍发往**当前**连接；重连只恢复下行事件流，未决审批由后端在旧连接
  断开时拒绝，这是后端协议语义，前端不做“跨连接补批”；
- `UiState.reconnects` 记录重连次数，状态面板显示“重连 N 次（按 seq 续传）”。
- 多实例部署时，SSE/WS 只连得到本实例的 run（ch20 的边界）；前端应把 409/404
  提示为“任务在其他实例，请轮询 `GET /v1/runs/{id}`”。

## 3. 本课做到 / 没做到

**做到**：`BackendClient`（HTTP 提交/查询/健康检查）；`RunSession`（同连接
事件 + 审批/取消/心跳，断线自动重连 + `after=seq` 续传 + 退避预算）；`apply_message`
状态归约；对话、trace、审批 diff 三种渲染；`describe_error` 把 HTTP/网络异常
翻译成中文可行动提示；示例任务、运行/取消/清空按钮、无审批时禁用批准按钮、
折叠 trace 面板；Gradio Blocks 布局与 async generator 流式刷新；`gr.State` 跨
回调保存状态（已用 `gradio_client` 驱动 `/start` 与 `/approve`，验证真实模型的
任务流与 bash 审批闭环）；兼容 Gradio 4/5/6 的 Chatbot 参数差异；懒加载 gradio
（未安装时给出安装提示）；L0 全离线 22 例；L1 连真实后端的真模型冒烟。

**没做到（留给后续）**：没有浏览器自动化测试（L1 覆盖客户端协议，Gradio handler
用 API 客户端验证，视觉交互仍需手测）；重连只恢复下行事件，不跨连接恢复未决
审批；没有多标签页/多用户隔离之外的服务端会话；没有 side-by-side 文件 diff；
没有历史 run 列表与回放页面；没有鉴权、CORS、CSRF、速率限制（前端直连内网
后端）；没有 Markdown 消毒/富文本安全策略；没有移动端适配；没有 Gradio queue
的背压与单 writer 重构；没有把技能/记忆 diff 做成结构化审批对象（当前复用工具
参数）。

## 4. 关联八股（~12 题）

1. **前端-后端交互**：REST + WebSocket + SSE 各自适合什么？为什么“提交任务”
   用 POST、“看事件”用 WS，而不是全部走 WS？
2. **流式渲染**：async generator 与一次性返回的区别？Gradio 如何把 `yield`
   映射成增量更新？为什么要给事件加 `seq`？
3. **状态回填**：Gradio 回调之间为什么不能靠闭包变量？`gr.State` 的生命周期与
   并发语义？多用户时状态如何隔离？
4. **WS vs SSE**：双向控制（审批/取消）为什么选 WS？SSE 的 `Last-Event-ID`
   如何做断点续传？两者在代理/缓冲上的差异？
5. **连接生命周期**：为什么审批 Future 绑在连接上？连接断开后未决审批会怎样？
   如何设计“刷新页面仍能继续审批”？
6. **事件顺序**：`seq`、`after` 与消息乱序/重放；客户端去重与幂等渲染怎么做？
7. **并发回调**：流式生成器与审批按钮同时改状态会出什么问题？单 writer、
   事件队列、乐观更新分别如何解决？
8. **限流与背压**：浏览器渲染跟不上事件速度怎么办？截断、节流（throttle）、
   批量合并（coalesce）与虚拟列表的取舍？
9. **错误呈现**：HTTP 4xx 与 WS `error` 消息如何区分？如何把 `429 Retry-After`
   反馈到 UI？失败任务的重试入口放哪？
10. **安全**：前端为什么不持有模型 key？Markdown 渲染工具输出有什么 XSS 风险？
    内网后端直连时 CORS/CSRF/鉴权应放在哪一层？
11. **多实例与粘性会话**：run 在实例 B、前端连到 A 时，WS/SSE 为什么失败？
    粘性会话、Redis Pub/Sub、客户端轮询各自的代价？
12. **可测试性**：为什么把协议客户端与 UI 分开？`MockTransport` + fake WS 能覆盖
    什么、不能覆盖什么？浏览器端 E2E 该用什么工具？

## 5. 生产版延伸阅读（复用 + 回填）

- ch15–ch18：HTTP 提交、并发队列、SSE、WS 控制协议；本章是它们的消费端。
- ch19–ch21：SQL 历史、Redis 热状态、容器化部署；前端只依赖公开 API。
- `evolving_harness/agent.py`：`TraceRecorder` 产生的 `model.request/response`、
  `tool.call/result`、`run.end` 是 trace 面板的数据源。
- 技能/记忆审批：可把 RSI 的 diff 提案包装成新的 approval 对象，复用本章的
  `approval.request` → `approve` 闭环。

## 6. 运行与验证

```bash
# L0 离线（不需要 gradio / 后端 / key）
uv run pytest course/ch22_gradio/ -q
#   → ws_url / 消息解析 / 状态归约 / 渲染 / 错误翻译 / httpx 客户端 /
#      fake WS 会话与断线续传
uv run ruff check course/ch22_gradio/

# 启动后端（另一个终端；真实 key 只进服务端环境）
set -a; source .env; set +a
uv run python course/ch21_docker/backend.py

# L1 冒烟：真实后端 + 真模型，走 WS 流式到终态
EH_SMOKE=1 EH_BACKEND_URL=http://127.0.0.1:8000 \
  uv run pytest course/ch22_gradio/ -m smoke -s

# 浏览器手测 Gradio
uv sync --extra dev --extra frontend
uv run python course/ch22_gradio/frontend.py \
  --backend http://127.0.0.1:8000 --port 7860
```

> L1 依赖 `websockets`（已在 `dev` extra 中）。gradio 只在启动 UI 时需要；
> 未安装时 `build_ui` 会提示 `uv sync --extra frontend`。
> Gradio 4/5 的 `Chatbot` 需要 `type="messages"`，6.x 已移除该参数；`_make_chatbot`
> 用签名探测兼容两者。
>
> 代理环境提示：若 shell 的 `no_proxy` 含裸 IPv6 CIDR（如 `fe80::/10`），httpx
> 可能在 import gradio 时报 `Invalid port: ':'`；启动前清掉 `NO_PROXY` 并只保留
> `no_proxy=localhost,127.0.0.1`。
