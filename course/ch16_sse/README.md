# ch16 · sse（后端轨第 3 章）——把轮询升级成服务器推送的事件流

ch15 的客户端只能反复 `GET /v1/runs/{id}`：每次都要建连接、解析完整 JobView，事件之间的顺序和
间隔全丢了。本章给每个 run 加一条 **SSE（Server-Sent Events）** 通道，服务端把循环里的
`run.start / model.request / model.response / tool.call / tool.result / run.end` 按发生顺序推给客户端。

`backend.py` = 本章 hero。数据源直接复用 Agent 已落盘的 **trace JSONL**：`TraceRecorder` 每写一行
立即 flush，服务端增量 tail 并转成 SSE——**不改核心循环**，也天然获得可回放、可续传的事件历史。

```
POST /v1/runs ──> 202 + job_id
                      │
                      ▼
GET /v1/runs/{id}/events ──> text/event-stream
                      │
                      ├─ event: open           握手（retry 建议）
                      ├─ event: job.status     queued → running
                      ├─ event: run.start      ┐
                      ├─ event: model.request  │ 直接来自 trace，
                      ├─ event: model.response │ 每行一条 SSE
                      ├─ event: tool.call      │
                      ├─ event: tool.result    │
                      ├─ event: run.end        ┘
                      └─ event: job.end        succeeded | failed，然后关流
```

---

## 0. 流程图

```
                         ┌──────────────────────────────┐
                         │ client: EventSource / curl   │
                         └──────────────┬───────────────┘
                                        │ ① POST /v1/runs
                                        ▼
                         ┌──────────────────────────────┐
                         │ 202 + JobView(status=queued) │
                         └──────────────┬───────────────┘
                                        │ ② GET /v1/runs/{id}/events
                                        ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ FastAPI 路由层                                                       │
   │  Last-Event-ID / ?after= → 从第 N 条之后续传                          │
   │  StreamingResponse(event_stream(...), media_type=text/event-stream)  │
   │  no-cache / X-Accel-Buffering: no（防代理缓冲）                       │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ event_stream（每 50ms 轮询）                                          │
   │  TraceTail.read_new()   ← 增量读 traces/<run_id>.jsonl               │
   │  seq > after ? → sse_frame(kind, record, id=seq)                     │
   │  job.status 变化 → event: job.status（终态除外）                       │
   │  job 终态 → 补读一次 → event: job.end → return（关流）                 │
   │  每 15s 无事件 → ": keep-alive" 注释帧                                │
   │  await request.is_disconnected() → 提前退出                          │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
              traces/<run_id>.jsonl（Append-only，逐行 flush）
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `StreamingResponse` | 持续 `yield` 文本块，响应不结束即连接不关 |
| `text/event-stream` | SSE 媒体类型；浏览器 `EventSource` 原生解析 |
| `TraceTail` | 增量 tail JSONL，文件尚未创建时返回空 |
| `sse_frame()` | 编码 `retry:` / `id:` / `event:` / `data:` 帧 |
| `Last-Event-ID` | 浏览器重连自动携带；服务端据此跳过已收事件 |
| `?after=<seq>` | 非浏览器客户端的手动续传参数 |
| `request.is_disconnected()` | 客户端断开后停止 tail，避免泄漏协程 |
| `: keep-alive` | SSE 注释帧，防止空闲连接被代理/网关掐断 |
| `httpx.ASGITransport` | L0 测试直接消费 ASGI 流，不启端口 |

## 2. 实现的对象 + 用法

### SSE 线格式

一条事件 = 若干字段行 + 一个空行：

```
retry: 3000
event: model.response
id: 3
data: {"v":1,"run_id":"demo","seq":3,"ts":...,"kind":"model.response","data":{...}}

```

- `id` 是 trace 的 `seq`，**同一 run 内严格递增**；重连时浏览器放进 `Last-Event-ID` 请求头。
- `event` 直接复用 trace 的 `kind`，客户端可以按类型订阅。
- `data` 是**整条 trace record**（含 `v/run_id/seq/ts/elapsed_ms/kind/data`），方便排查与回放。
- 首帧 `event: open` 带 `retry: 3000`，告诉浏览器断线后 3 秒重连。

### 从 trace 增量读，不重新实现事件总线

```python
class TraceTail:
    def read_new(self) -> list[dict]:
        if self._fh is None:
            if not self.path.is_file():
                return []              # job 还在排队
            self._fh = self.path.open(encoding="utf-8")
        ...
```

`TraceRecorder.event()` 每行 `flush()`，所以 tail 能近实时拿到。好处：

- **零侵入**：不需要在 Agent 里加 hook，也不会漏掉任何已记录事件；
- **可回放**：晚到的客户端从 `after=0` 开始，能把历史完整补一遍；
- **可续传**：`Last-Event-ID` 只是「跳过 seq ≤ N」。

代价是 **50ms 轮询延迟**（教学可接受）；生产可换成 `inotify` / 队列广播 / 直接把 tracer 包成发布者。

### 终止条件必须明确

```python
if job.status in TERMINAL:
    # worker 在 run() 返回后才写终态；此时 trace 已 flush 完，补读一次
    ...
    yield sse_frame("job.end", job.view().model_dump())
    return
```

`job.end` 之后**主动关流**。否则 EventSource 会把「连接正常结束」当成断线并自动重连，已完成的任务
被反复重放，形成隐式忙循环。终态本身不再发 `job.status`，避免与 `job.end` 重复表达。

### 断线重连

```python
@app.get("/v1/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request, after: int = 0):
    ...
    last_event_id = request.headers.get("last-event-id", "")
    if last_event_id.isdigit():
        after = max(after, int(last_event_id))
```

浏览器 EventSource 重连时会自动带 `Last-Event-ID`；非浏览器客户端用 `?after=`。两条路径最终都归一到
「从 seq > after 开始发」。**幂等消费**由客户端负责：同一条事件可能因为重连边界被重发一次。

### 慢客户端与背压

本章的消费者是 `yield`，Starlette 会在发送缓冲区满时施加背压（协程挂起）。但事件源是文件而不是
内存队列，所以慢客户端**不会拖住 Agent**，只会拖慢自己的读取。若换成实时广播，就需要 per-client
有界队列 + 丢帧/断开策略，这是 ch17 反压讨论的起点。

### 客户端示例

```bash
# -N 关闭 curl 缓冲，实时打印
curl -N localhost:8000/v1/runs/demo-1/events
# 断线续传：从 seq 3 之后继续
curl -N 'localhost:8000/v1/runs/demo-1/events?after=3'
```

浏览器：

```js
const es = new EventSource(`/v1/runs/${runId}/events`);
es.addEventListener("model.response", (e) => console.log(JSON.parse(e.data)));
es.addEventListener("job.end", (e) => { console.log(JSON.parse(e.data)); es.close(); });
```

### 运行

```bash
EH_HTTP_MOCK=1 uv run python course/ch16_sse/backend.py
curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"只回复：hello","run_id":"demo-1"}'
curl -N localhost:8000/v1/runs/demo-1/events

# 真模型
set -a; source .env; set +a
uv run python course/ch16_sse/backend.py
```

## 3. 本课做到 / 没做到

**做到**：`GET /v1/runs/{id}/events` 的 SSE 事件流；事件来自 trace，按 `seq` 严格有序；`id` +
`Last-Event-ID` / `?after=` 续传；心跳注释帧；客户端断开检测；`job.end` 终止关流；晚到客户端完整
回放；工具调用事件；失败 run 的 `run.end(status=error)` + `job.end(failed)`；L0 全离线；L1 真模型流式冒烟。

**没做到（留给后续）**：**token 级增量**——当前生产核的 `ChatModel.complete()` 一次性返回完整
`ModelResponse`，所以 SSE 推的是 run 级事件；要做逐字输出，需在模型适配器加 `stream()` 并把 delta
转发成 `event: token`（见延伸阅读）。也没有：取消/审批（SSE 单向，ch17）；多实例广播（ch19）；
事件持久化分页（ch18）；鉴权与单用户连接数限制（生产加固）。

## 4. 关联八股（~12 题）

1. **SSE vs 轮询**：相同事件量下，连接数、延迟、服务端开销、客户端实现复杂度各差在哪？
2. **SSE vs WebSocket**：什么时候单向流足够？双向控制为什么不能用 SSE 做？
3. **SSE 线格式**：`event` / `id` / `data` / `retry` / 注释帧分别是什么语义？多行 data 怎么处理？
4. **EventSource 自动重连**：什么情况下触发？为什么服务端必须显式关流？`retry` 控制什么？
5. **Last-Event-ID**：它解决什么问题？为什么 id 必须单调递增？重连边界会不会重发？
6. **事件顺序**：为什么用 trace 的 `seq` 而不是时间戳排序？时间戳相同怎么办？
7. **代理缓冲**：Nginx/网关为什么会攒住 SSE？`X-Accel-Buffering: no` 和 `Cache-Control` 各起什么作用？
8. **心跳**：为什么需要 `: keep-alive`？不发心跳会出现什么故障？间隔怎么定？
9. **慢客户端**：SSE 的背压如何体现？从文件 tail 和从内存广播，慢客户端的影响有何不同？
10. **终止与资源**：客户端断开后如何停止 tail？不处理 `is_disconnected` 会泄漏什么？
11. **token 级流式**：为什么当前只能在 run 事件级推送？模型适配器要加什么接口？工具调用增量怎么拼？
12. **多实例部署**：客户端连到 A 实例、run 跑在 B 实例时怎么办？需要哪些外部设施？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/tracing/recorder.py`：`TraceRecorder` 逐行 flush 是本章能 tail 的前提。
- `evolving_harness/agent.py`：`model.request / model.response / tool.call / tool.result / run.end`
  的事件字段；本章不改循环。
- ch15：作业队列与 per-run workspace；本章在其上加事件通道。
- ch17：用 WebSocket 补上 cancel/approve 的双向控制与反压。
- ch18：trace 索引与事件分页；ch19：跨实例广播与连接管理。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch16_sse/ -q
#   → 事件序 open → run.* → job.end；trace id 严格递增且不重复
#   → tool.call / tool.result 顺序与内容
#   → Last-Event-ID 续传不重放旧事件；完成后仍可从头回放
#   → 模型失败：run.end(error) + job.end(failed) 后关流
#   → 未知 run → 404；响应头 text/event-stream + no-cache

uv run ruff check course/ch16_sse/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch16_sse/ -m smoke -s
#   → 真 deepseek 异步作业 + SSE 收完整事件流，断言 job.end 成功

# 手工验证
EH_HTTP_MOCK=1 uv run python course/ch16_sse/backend.py
curl -N localhost:8000/v1/runs/<run_id>/events
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
