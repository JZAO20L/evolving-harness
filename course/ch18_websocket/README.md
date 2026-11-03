# ch18 · websocket（后端轨第 4 章）——双向流：cancel 与 approve

ch17 的 SSE 是单行道：服务端能推事件，用户却没法在 run 跑到一半时取消，也没法对一次危险工具调用
（写文件、执行 shell）说「批准 / 拒绝」。本章加一条 WebSocket，把**事件下行**和**控制上行**放进
同一条连接：

```
POST /v1/runs ──> 202 + job_id
                      │
                      ▼
WS /v1/runs/{id}/ws  <──> 双向 JSON
   下行：hello / job.status / trace event / approval.request / pong / heartbeat / job.end
   上行：{"type":"cancel"} / {"type":"approve", ...} / {"type":"ping"}
```

`backend.py` = 本章 hero。关键设计：**控制不进入 Agent 循环**。

- 取消 = 取消该 job 的 `asyncio.Task`，core 仍是被动执行者；
- 审批 = 在工具 handler 外包一层 `await Future` 的门，Agent 只看到「工具调用变慢/返回错误」；
- 单 writer = 所有下行帧由一个 sender 协程发出，reader 只改状态、不直接写 socket。

---

## 0. 流程图

```
                    ┌────────────────────────────────┐
                    │ client: 浏览器 / websockets    │
                    └───────────────┬────────────────┘
                                    │ ① POST /v1/runs → 202
                                    │ ② WS /v1/runs/{id}/ws
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ FastAPI WebSocket 路由                                               │
   │  reader loop: receive_json → handle_ws_command                       │
   │    cancel  → JobManager.cancel(run_id) → job.handle.cancel()         │
   │    approve → RunControl.resolve(approval_id, bool)                   │
   │    ping    → control.queue.put({"type":"pong"})                      │
   │  sender task: ws_event_stream（唯一 writer）                          │
   │    control.queue → approval.request / pong / ack                     │
   │    TraceTail     → event(run.start / model.* / tool.* / run.end)     │
   │    job.status    → queued / running                                  │
   │    heartbeat     → 每 15s 一帧                                        │
   │    job.end       → 发送后 close                                       │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ JobManager worker                                                    │
   │  job.handle = asyncio.create_task(service.run(..., control=...))     │
   │  await handle → succeeded / failed / CancelledError → cancelled      │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ RunService / Agent                                                   │
   │  build_tools → write_file / bash 的 handler 包上 approval gate       │
   │  Agent.run → 工具边界 await Future[bool] → 批准才执行                 │
   └──────────────────────────────────────────────────────────────────────┘
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `WebSocket` / `WebSocketDisconnect` | FastAPI 双向连接与断开异常 |
| `RunControl` | 控制面：审批 Future 表 + 下行消息队列 |
| `asyncio.Task.cancel()` | 取消正在跑的 run |
| `asyncio.Future[bool]` | 工具边界的审批挂起点 |
| `Tool.handler` 包装 | 不改 `ToolRegistry` / Agent，只替换单个工具实现 |
| 单 sender 协程 | 保证同一 socket 不被 reader 与 sender 并发写 |
| `TraceTail`（复用 ch17） | 把 trace JSONL 增量转成下行 `event` |
| `TestClient.websocket_connect` | L0 同步测试 WS 全流程，无需起端口 |

## 2. 实现的对象 + 用法

### 协议

**服务端 → 客户端**

| `type` | 何时 | 关键字段 |
|---|---|---|
| `hello` | 连接建立 | `run_id`, `after` |
| `job.status` | queued → running（终态不发） | `status` |
| `event` | trace 新行 | `event`(kind), `seq`, `record` |
| `approval.request` | 危险工具被调用 | `approval_id`, `tool`, `arguments` |
| `approval.ack` | 服务端收到审批 | `approval_id`, `approved`, `accepted` |
| `pong` | 收到 `ping` | `ts` |
| `heartbeat` | 每 15s 空闲 | `ts` |
| `job.end` | 终态（随后 close） | `job`（完整 JobView） |
| `error` | 非法命令 | `message` |

**客户端 → 服务端**

```json
{"type": "cancel"}
{"type": "approve", "approval_id": "6f1c2a9b", "approved": true}
{"type": "ping"}
```

手机客户端示例（浏览器 `WebSocket` 同理）：

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/v1/runs/demo-1/ws") as ws:
        await ws.send(json.dumps({"type": "ping"}))
        async for raw in ws:
            frame = json.loads(raw)
            if frame["type"] == "approval.request":
                await ws.send(json.dumps({
                    "type": "approve",
                    "approval_id": frame["approval_id"],
                    "approved": True,
                }))
            if frame["type"] == "job.end":
                print(frame["job"]["status"])
                break

asyncio.run(main())
```

### 取消：排队中 vs 运行中

```python
def cancel(self, run_id: str) -> bool:
    job = self.jobs.get(run_id)
    if job is None or job.status in TERMINAL:
        return False
    if job.status == "queued":
        job.status = "cancelled"          # worker 出队时看到终态直接跳过
        job.error = "cancelled before start"
        return True
    job.cancel_requested = True
    job.handle.cancel()                    # 运行中：取消 asyncio.Task
    return True
```

worker 里必须区分「用户取消」和「服务关停」：

```python
except asyncio.CancelledError:
    if not job.cancel_requested:
        raise                              # 外层 cancel：继续向上传播
    job.status = "cancelled"
```

- 排队取消不会真的从 `asyncio.Queue` 里摘除，而是标记终态，出队时跳过；
- 运行中取消会把 `CancelledError` 注入模型/工具 await 点；`Agent.run` 的 `finally`
  会关 trace 文件，但不一定写 `run.end`（核心没把取消当普通异常）——所以 `job.end` 是权威终态；
- 终态集合是 `succeeded | failed | cancelled`。

### 审批：门放在工具边界

```python
async def gated(**arguments):
    approval_id, future = control.request_approval(name, arguments)
    approved = await asyncio.wait_for(future, timeout=control.approval_timeout)
    if not approved:
        return "ERROR: approval denied by user"   # 错误当数据，循环继续
    return await _invoke(original, arguments)
```

- 默认只拦 `write_file` / `bash`（`DEFAULT_APPROVAL_TOOLS`）；`read_file` 等只读工具直接执行。
- 工具必须先在 YAML `tools.enabled` 里启用，gate 才会包到它。
- 拒绝返回 `ERROR: ...`，`tool.result` 的 `is_error=True`，模型据此改计划。
- 超时（默认 60s）按拒绝处理，避免 run 永久挂起。
- 客户端断开时 `RunControl.close()` 把**所有未决审批**置为 denied；新审批直接失败。

### 单 writer 与心跳

```python
sender = asyncio.create_task(ws_event_stream(...))
while not sender.done():
    raw = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
    handle_ws_command(manager, job, raw)     # 只入 control.queue，不 send
```

reader 从不直接 `send_json`，所有下行都排进 `control.queue` 由 sender 独占写出。否则两个协程同时
写同一个 WebSocket 会产生难以复现的帧交错/协议错误。

`heartbeat` 是**应用层**保活：WebSocket 协议本身有 ping/pong 控制帧（uvicorn 会处理），但中间的
反向代理、负载均衡通常只认应用层流量。空闲 15s 发一帧 `{"type":"heartbeat"}`，同时保留客户端
主动 `ping` → `pong` 的探测能力。

### 运行

```bash
EH_HTTP_MOCK=1 uv run python course/ch18_websocket/backend.py
curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"只回复：hello","run_id":"demo-1"}'
# 用浏览器控制台或上面的 websockets 脚本连 ws://localhost:8000/v1/runs/demo-1/ws

# 真模型
set -a; source .env; set +a
uv run python course/ch18_websocket/backend.py
```

## 3. 本课做到 / 没做到

**做到**：WS 双向 JSON 协议；trace 事件下行；`job.status` / `job.end` 终止；`ping/pong` 与 15s
heartbeat；运行中 cancel（`asyncio.Task`）与排队 cancel；`write_file`/`bash` 的 approve/deny 门；
拒绝变成工具错误、run 继续；审批超时与断连自动拒绝；单 writer；未知 run/命令的错误帧；SSE 端点保留；
L0 全离线（TestClient）；L1 真模型 WS 冒烟。

**没做到（留给后续）**：取消不一定写 `run.end`（核心不把取消当异常），权威状态看 `job.end`；
审批门是本章在边界包的，不是 core 的正式能力；没有鉴权/Origin 校验（生产必须做）；没有命令幂等键
（重发 `approve` 第二次会 `accepted=false`）；没有跨实例路由（run 在 B 实例、WS 连到 A 实例时需要
粘性会话或消息总线，见 ch20）；没有对慢客户端的丢帧策略（sender 会被 `send_json` 背压）。

## 4. 关联八股（~12 题）

1. **WS vs SSE**：为什么取消/审批必须用 WebSocket？SSE + 另一个 POST 端点不行吗？
2. **握手与升级**：WebSocket 如何从 HTTP/1.1 升级？`Sec-WebSocket-Key` 的作用是什么？
3. **帧 vs 消息**：WebSocket 的 frame/message 区别？分片消息如何重组？文本/二进制帧怎么选？
4. **心跳**：协议级 ping/pong 和应用层 heartbeat 各解决什么问题？为什么代理场景常常都需要？
5. **取消语义**：`Task.cancel()` 注入到哪里？为什么 `CancelledError` 是 `BaseException`？
   不区分「用户取消」和「服务关停」会出什么故障？
6. **审批挂起**：为什么把 gate 放在工具边界而不是循环里？超时/断连应批准还是拒绝？为什么？
7. **单 writer**：两个协程并发 `send_json` 会怎样？如何保证帧顺序？
8. **背压**：客户端读得慢时 `send` 会阻塞谁？内存队列无限增长有什么风险？如何丢帧/断开？
9. **断连清理**：客户端消失后要清理哪些资源（sender、审批 Future、trace 文件、worker）？
10. **终态**：为什么 `job.end` 后必须主动 close？客户端自动重连会造成什么？
11. **鉴权**：WS 握手阶段如何带 token？为什么必须校验 `Origin`？查询参数带 token 有什么风险？
12. **水平扩展**：run 跑在实例 B、WS 连到实例 A 时怎么把事件送过去？粘性会话 vs 发布订阅。

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py`：`Agent.run` 的异常/`finally` 路径；取消时 trace 的落盘边界。
- `evolving_harness/tools/base.py`：`Tool.handler` 与 `ToolRegistry.dispatch`；gate 只替换 handler。
- ch17：SSE 的 TraceTail 与事件编码；本章复用并扩展成双向协议。
- ch19：作业与事件落库，支持跨进程恢复；ch20：跨实例广播与连接/限流。
- ch21：把 `bash` 等危险工具放进容器沙箱，审批门与隔离层配合。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch18_websocket/ -q
#   → 事件下行 + job.end；ping/pong
#   → 运行中取消（cancelled + CancelledError）；排队中取消（cancelled before start）
#   → approval.request → approve 后文件落盘；deny 后 tool.result 是 ERROR 且文件不存在
#   → 未知命令 error；未知 run 4404 + error 帧

uv run ruff check course/ch18_websocket/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch18_websocket/ -m smoke -s
#   → 真 deepseek 作业 + WS 收完整事件流并 ping/pong

# 手工验证
EH_HTTP_MOCK=1 uv run python course/ch18_websocket/backend.py
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
> L0 使用 `starlette.testclient`，会打印一条 `httpx2` 的弃用警告，不影响测试结果。
