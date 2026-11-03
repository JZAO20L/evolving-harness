# ch16 · concurrency（后端轨第 2 章）——把同步 HTTP 升级成异步作业队列

ch15 的 `POST /v1/runs` 要等整个 Agent 跑完才返回：一个长 run 占住一条连接，服务能同时处理的 run 数
约等于连接数，客户端还会撞上网关超时。本章把「一次 run」从请求生命周期里拆出来，变成**可轮询的作业**：

- 提交即返回 `202 + job_id`，客户端用 `GET /v1/runs/{id}` 轮询；
- 固定数量 worker（= 并发上限）+ 有界队列（= 背压）；
- 每个 run 独立 `workspace/runs/<run_id>`，互不踩文件；
- 单个 run 失败只污染自己那个 job，不杀死 worker。

`backend.py` = 本章 hero。Agent 核仍复用生产实现 `evolving_harness.agent`，本章只改**边界**：
并发开在跨 run 的调度层，单个 run 内部依旧是顺序 ReAct。

```
POST /v1/runs ──> 202 + JobView(status=queued)
                     │
                     ▼
              asyncio.Queue（有界）
                     │  满了 → 429 + Retry-After
                     ▼
          N 个 worker（并发上限 = N）
                     │
                     ▼
          RunService.run(task, run_id, workspace=workspace/runs/<run_id>)
                     │
                     ▼
          JobView(status=succeeded|failed) ←── GET /v1/runs/{id} 轮询
```

---

## 0. 流程图

```
                         ┌──────────────────────────────┐
                         │ client: curl / SDK / WebUI   │
                         └──────────────┬───────────────┘
                                        │
              POST /v1/runs             │              GET /v1/runs/{id}
                     │                  │                     │
                     ▼                  │                     ▼
   ┌────────────────────────────────────┴────────────────────────────────┐
   │ FastAPI 路由层                                                       │
   │  POST /v1/runs   → pydantic 校验 → JobManager.submit → 202 JobView   │
   │  GET  /v1/runs/{id} → JobManager.get → JobView | 404                 │
   │  GET  /health    → concurrency / active / queued                     │
   │  队列满 → 429 + Retry-After: 1                                       │
   └──────────────────────────────┬──────────────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ JobManager（调度边界）                                                │
   │  jobs: dict[run_id, Job]        ← 幂等：同 id 返回同一作业            │
   │  queue: asyncio.Queue(maxsize)  ← 只装「等待中」；满了背压 429        │
   │  workers: N 个 asyncio.Task     ← N = 并发上限                        │
   │  submit / get / wait_idle / start / stop                             │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼  worker 取到 job，status: queued → running
   ┌──────────────────────────────────────────────────────────────────────┐
   │ RunService.run(task, run_id, workspace)                              │
   │  State.load → create_model → build_tools(state, run_workspace)       │
   │  Agent(...).run(task, run_id)                                        │
   │  → RunResponse（含 workspace 路径）                                   │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
     workspace/runs/<run_id>/        traces/<run_id>.jsonl
     job.status=succeeded ──┐        （run.start … run.end）
     job.error=...          ├──> GET /v1/runs/{id} 返回 JobView
     job.status=failed  ────┘
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `asyncio.Queue(maxsize=N)` | 有界作业队列；`put_nowait` 满则抛 `QueueFull` → HTTP 429 |
| `asyncio.create_task` | worker；N 个 worker 同时消费 = 并发上限 |
| `JobManager` | 作业登记、幂等提交、worker 生命周期、失败隔离 |
| `Job` / `JobView` | 内部状态（含时间戳）与对外响应模型分离 |
| `lifespan` | 应用启动/关闭时拉起/收拢 worker |
| `RunService(workspace=...)` | 调度器注入 per-run workspace，其余装配与 ch15 相同 |
| `httpx.ASGITransport` | L0 测试不启端口；轮询同一 ASGI app |

## 2. 实现的对象 + 用法

### 作业模型

```python
@dataclass
class Job:
    run_id: str
    task: str
    workspace: Path
    status: str = "queued"        # queued | running | succeeded | failed
    result: RunResponse | None = None
    error: str | None = None
    created_at: float = ...
    started_at: float | None = None
    finished_at: float | None = None
```

`JobView` 是它的 HTTP 投影：客户端只看得到 `run_id/task/status/workspace/result/error/时间戳`。
内部字段（队列引用等）不泄漏到契约里。

### 并发上限用 worker 数，不用信号量

```python
async def start(self):
    self._workers = [
        asyncio.create_task(self._worker(i), name=f"eh-worker-{i}")
        for i in range(self.max_concurrency)
    ]

async def _worker(self, index):
    while True:
        job = await self.queue.get()
        if job is None:            # stop() 的哨兵
            self.queue.task_done()
            return
        ...
```

worker 自己就是消费者，天然限制「同时执行的 run 数」；再加信号量是重复限流。队列只统计**等待中**
的作业，正在跑的作业不占队列容量——所以 5 个作业、2 个 worker、队列 8 时，`queued == 3`。

### 提交幂等：同 run_id 不重复跑

```python
async def submit(self, task, run_id=None):
    await self.start()
    rid = run_id or uuid.uuid4().hex[:12]
    existing = self.jobs.get(rid)
    if existing is not None:
        return existing          # 返回同一作业，不重复入队
    ...
```

客户端重试 POST（超时但服务端已收到）不会触发第二次模型调用。注意这一实现依赖**单个 event loop**：
`get` 与 `put_nowait` 之间没有 `await`，检查与入队是原子的；多进程/多 loop 部署要换成数据库唯一键
（ch19）或 Redis 原子操作（ch20）。

### 背压：队列满返回 429

```python
try:
    self.queue.put_nowait(job)
except asyncio.QueueFull as exc:
    raise QueueFull(...) from exc
```

路由层把它映射成 `429 Too Many Requests` + `Retry-After: 1`。没有背压时，突发流量会把内存吃穿：
作业对象、任务文本、后续模型请求全在排队。**拒绝比无限排队更安全**。

### 失败隔离

```python
try:
    job.result = await self.service.run(...)
    job.status = "succeeded"
except (AgentConfigError, MaxStepsExceeded, ModelError) as exc:
    job.status = "failed"
    job.error = f"{type(exc).__name__}: {exc}"
except Exception as exc:          # 一个作业失败不能杀死 worker
    job.status = "failed"
    job.error = f"{type(exc).__name__}: {exc}"
finally:
    job.finished_at = time.time()
    self._active -= 1
    self.queue.task_done()
```

失败写进 `job.error`，worker 回到循环继续取下一个。注意：把异常分类存进 job 只是**记账**，
HTTP 层的同步错误码（ch15 的 400/409/502）在异步模型里变成了 `GET` 结果里的 `status/error`——
客户端要根据 `failed` + `error` 自行区分。

### per-run workspace

```python
workspace = self.workspace_root / rid          # workspace/runs/<run_id>
job = Job(run_id=rid, task=task, workspace=workspace)
...
await self.service.run(job.task, job.run_id, workspace=job.workspace)
```

两个 run 同时写 `notes.md`/`output.txt` 也不再互相覆盖；trace 仍按 `run_id` 落到共享
`traces/<run_id>.jsonl`。workspace 路径随 `JobView` 返回，客户端可定位产物。

### 应用工厂与生命周期

```python
def create_app(service=None, *, max_concurrency=2, queue_size=32, ...):
    manager = JobManager(service, max_concurrency=..., queue_size=...)

    @asynccontextmanager
    async def lifespan(_app):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(..., lifespan=lifespan)
    app.state.manager = manager
```

`stop()` 给每个 worker 塞一个 `None` 哨兵再 `gather`，保证关停时没有任务被半途丢弃。
测试里用 `httpx.ASGITransport` 不会自动触发 lifespan，所以 `submit` 自己也会 `start()`，
并在结尾显式 `await app.state.manager.stop()`。

### 运行

```bash
# 离线：不联网，用 Mock 模型
EH_HTTP_MOCK=1 uv run python course/ch16_concurrency/backend.py
curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"只回复：hello","run_id":"demo-1"}'
# → 202 {"run_id":"demo-1","status":"queued",...}
curl -s localhost:8000/v1/runs/demo-1
curl -s localhost:8000/health
# → {"concurrency":2,"active":0,"queued":0,...}

# 真模型：先载入环境变量，再启动
set -a; source .env; set +a
uv run python course/ch16_concurrency/backend.py
```

## 3. 本课做到 / 没做到

**做到**：`POST /v1/runs` 立即 202；`GET` 轮询 queued/running/succeeded/failed；固定 worker 并发上限；
有界队列 + 429 背压；`run_id` 幂等；per-run workspace；失败隔离不杀 worker；`/health` 暴露
active/queued；lifespan 优雅关停；trace 仍按 run 落盘；L0 全离线；L1 真模型异步作业冒烟。

**没做到（留给后续）**：作业状态只在进程内存，重启即丢（ch19）；没有跨进程/跨实例并发控制（ch19/19）；
没有 SSE/WS，客户端只能轮询（ch17/17）；没有取消/审批挂起点（ch18）；没有鉴权、限流、优先级队列
（ch20/生产加固）；worker 里若调用**同步阻塞**工具（如当前文件工具）会卡住整个 event loop，生产要
`asyncio.to_thread` 或独立进程（ch21 沙箱）；模型客户端每 run 新建，没有连接池复用。

## 4. 关联八股（~12 题）

1. **async vs 多线程**：asyncio 的并发单位是什么？为什么 I/O 密集的 Agent 服务优先用它？
2. **并发上限为什么用 worker 数而不是信号量**？两者分别适合什么场景？重复限流会怎样？
3. **有界队列 vs 无界队列**：背压如何保护内存与上游？为什么拒绝（429）比无限排队好？
4. **202 + 轮询 vs 200 同步**：各自的连接占用、超时、用户体验、重试语义有何不同？
5. **作业幂等**：同一 `run_id` 重复提交会发生什么？当前实现为何依赖单 event loop？跨进程怎么做？
6. **事件循环阻塞**：worker 里执行同步文件 I/O 或同步 SDK 会怎样？`to_thread` / 进程隔离的取舍？
7. **per-run workspace**：它解决哪些串扰问题？和 trace 共享目录为什么不冲突？
8. **失败隔离**：为什么 worker 要捕获 `Exception`？不捕获会怎样？为何还要单独列领域异常？
9. **任务取消与关停**：`stop()` 为什么放哨兵而不是直接 `cancel()`？正在跑的 run 应该等它跑完吗？
10. **状态存放**：`dict` 作业表重启即丢，哪些接口语义会变？ch19 如何替换为数据库？
11. **公平性**：FIFO 队列对长短任务混跑有什么问题？优先级/配额队列怎么设计？
12. **可观测性**：`/health` 的 active/queued 够不够？生产还需要哪些指标（等待时长、失败率、队列深度）？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py`：本章不改循环；单 run 内部仍是顺序 ReAct。
- `evolving_harness/tools/exec.py`：`bash` 已是 `asyncio.create_subprocess_exec`；文件工具是同步
  函数——在 worker 里直接调用会阻塞 event loop，这是 ch21 隔离要解决的问题。
- ch15：同步 HTTP 与错误码映射；本章把错误记账搬进 Job。
- ch17/17：把轮询升级为 SSE / WebSocket 推送。
- ch19：作业表与 checkpoint 落到数据库；ch20：外部队列与限流。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch16_concurrency/ -q
#   → 异步提交 → 轮询成功；per-run workspace + trace
#   → 同 run_id 只触发一次模型调用
#   → 两个 run 的 workspace 不同
#   → worker 数封顶并发；queued 只计等待中
#   → 队列满 → QueueFull / HTTP 429
#   → 模型失败只标记该 job failed，worker 存活
#   → 空白 task / 未知字段 / 非法 run_id → 422；未知 job → 404

uv run ruff check course/ch16_concurrency/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch16_concurrency/ -m smoke -s
#   → 真 deepseek 走 POST(202) → 轮询 succeeded → workspace + trace 断言

# 手工起服务
EH_HTTP_MOCK=1 uv run python course/ch16_concurrency/backend.py
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
