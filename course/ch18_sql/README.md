# ch18 · sql（后端轨第 5 章）——作业、trace、版本与 checkpoint 落库

ch15–ch17 的服务状态都在进程内存：重启后 `GET /v1/runs/{id}` 404，SSE 历史消失，也不记得
某次 run 跑的是哪份 agent。本章把四类数据放进关系库：

```
runs            作业行：task / status / result / error / 时间戳 / workspace
run_events      trace 索引：(run_id, seq) → kind + JSON data
agent_versions  版本登记：version + hash + 静态 YAML
checkpoints     Agent 动态状态快照：run_id → state JSON
```

`backend.py` = 本章 hero。实现用 **SQLite + 标准库 `sqlite3`**：零外部依赖、L0 完全离线；
SQL 保持通用，生产换 Postgres/MySQL 只需替换驱动与少量方言。所有阻塞 DB 调用走
`asyncio.to_thread`，不卡事件循环。

---

## 0. 流程图

```
POST /v1/runs
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ JobManager.submit                                                    │
│  1. State.load → (version, hash)                                     │
│  2. RunStore.register_agent   INSERT OR IGNORE agent_versions        │
│  3. RunStore.create_run       INSERT runs (status=queued)            │
│  4. queue.put_nowait(job)     ← 必须“先落库、再入队”                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ worker                                                               │
│  mark_running   UPDATE runs SET status='running', started_at=?       │
│  Agent.run(...)                                                      │
│  finally: _persist_finish                                            │
│     finish_run      UPDATE runs SET status/result_json/error/...     │
│     index_trace     读 trace JSONL → INSERT OR REPLACE run_events    │
│     save_checkpoint 读 core 的 checkpoint 文件 → UPSERT checkpoints  │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 读路径                                                               │
│  GET /v1/runs/{id}        内存 Job 优先，否则 store.get_run          │
│  GET /v1/runs             直接 store.list_runs（newest first 分页）  │
│  GET /v1/runs/{id}/events 内存中无 Job → stored_event_stream(source=db)│
│  GET /v1/runs/{id}/checkpoint      store.get_checkpoint               │
│  GET /v1/agent/versions            store.agent_versions              │
└──────────────────────────────────────────────────────────────────────┘
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `sqlite3`（标准库） | 零依赖关系库；WAL 允许读写并发 |
| `asyncio.to_thread` | 把同步 DB I/O 移出事件循环 |
| `SCHEMA` / `executescript` | 启动时幂等建表、建索引 |
| `INSERT ... ON CONFLICT DO NOTHING` | 重复 `run_id`、重复版本登记不报错 |
| `INSERT OR REPLACE` | trace 重放/重试幂等（`(run_id, seq)` 主键） |
| `UPSERT`（`ON CONFLICT ... DO UPDATE`） | checkpoint 以最新快照为准 |
| 外键 + `PRAGMA foreign_keys=ON` | `run_events` / `checkpoints` 必须挂在已有 run 上 |
| `PRAGMA journal_mode=WAL` | 读不阻塞写、写不阻塞读；崩溃恢复更好 |
| `TraceTail`（复用 ch16） | 实时 SSE 仍读 JSONL；DB 只在回放时用 |
| 内存优先 + DB 回退 | 不改变在线路径性能，只让重启后可读 |

## 2. 实现的对象 + 用法

### Schema

```sql
CREATE TABLE runs (
    run_id        TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    status        TEXT NOT NULL,          -- queued|running|succeeded|failed|cancelled
    agent_version INTEGER NOT NULL,
    agent_hash    TEXT NOT NULL,
    workspace     TEXT NOT NULL,
    trace_path    TEXT,
    result_json   TEXT,
    error         TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);

CREATE TABLE run_events (
    run_id    TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    ts        REAL NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    kind      TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE agent_versions (
    version     INTEGER NOT NULL,
    agent_hash  TEXT NOT NULL,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    static_json TEXT NOT NULL,            -- 静态 YAML 快照，用于复现/回滚
    created_at  REAL NOT NULL,
    PRIMARY KEY (version, agent_hash)
);

CREATE TABLE checkpoints (
    run_id     TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    saved_at   REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_runs_created ON runs(created_at DESC);
CREATE INDEX idx_events_kind  ON run_events(run_id, kind);
```

- `(version, agent_hash)` 复合主键：内容相同但版本号被改过的 YAML 也不会互相覆盖；
- `run_events` 的 `(run_id, seq)` 主键让索引天然幂等：同一 trace 重放不会产生重复事件；
- `idx_events_kind` 服务「某 run 的全部 tool.result」这类按 kind 的过滤；
- 时间戳用 `REAL`（epoch 秒），与内存 `Job` 字段一致，方便直接序列化。

### 先落库、再入队（本章修掉的一个真实竞态）

```python
# submit
if store is not None:
    state = State.load(agent_path)          # 坏 YAML → version/hash = 0/""
    await store.register_agent(state)       # 有 await，worker 可能开始跑
    await store.create_run(job, ...)        # 必须在入队之前
try:
    queue.put_nowait(job)
except QueueFull:
    job.status, job.error = "failed", "queue is full"
    await self._persist_finish(job)         # 让 429 也留下审计行
    raise
```

如果先 `put_nowait` 再 `await create_run`，worker 可能在 await 间隙把 job 跑到终态；
随后 `create_run` 才插入，行里就是「终态但 result/finished_at 为空」。这类问题在
生产里等价于「事务边界跨越了并发副作用」，修复方式不是加锁，而是**把持久化放到触发点之前**。

### 终态落库

```python
async def _persist_finish(self, job):
    if self.store is None:
        return
    try:
        await self.store.finish_run(job)          # 状态、结果、错误、end 时间
        await self.store.index_trace(             # JSONL → run_events
            job.run_id, trace_path_for(self.service, job.run_id)
        )
        await self.store.save_checkpoint(         # core 写的文件 → checkpoints
            job.run_id, checkpoint_path_for(self.service, job.run_id)
        )
    except Exception as exc:
        note = f"persist: {type(exc).__name__}: {exc}"
        job.error = f"{job.error}; {note}" if job.error else note
```

- 三步是**逻辑顺序**，但不是单个 SQL 事务（三者跨连接、跨文件）；任何一步失败都
  不改变业务终态，只在 `job.error` 追加 `persist: ...`，便于告警；
- trace 索引失败时 `run_events` 可能不全，重试只需再跑一次 `index_trace`（主键幂等）；
- checkpoint 只抄 core 已经写好的文件，本章不重新实现状态序列化。

### 重启后读取

```python
@app.get("/v1/runs/{run_id}")
async def get_run(run_id: str):
    job = manager.get(run_id)
    if job is not None:
        return job.view()               # 热路径：内存
    stored = await store.get_run(run_id)
    if stored is None:
        raise HTTPException(404)
    return stored                       # 冷路径：DB
```

```python
async def stored_event_stream(job, store, request, *, after=0):
    yield sse_frame("open", {"run_id": job.run_id, "after": after, "source": "db"})
    for record in await store.events(job.run_id, after=after):
        yield sse_frame(record["kind"], record, event_id=record["seq"])
    yield sse_frame("job.end", job.model_dump())
```

- 在线 run 仍走 `TraceTail` 读 JSONL（低延迟、增量）；
- 重启后没有内存 Job，就改读 `run_events`，客户端看到 `source: db`；
- `after` / `Last-Event-ID` 语义不变，跨重启也能续传。

### 版本登记

```python
await store.register_agent(state)   # INSERT OR IGNORE (version, agent_hash)
```

- `runs.agent_version / agent_hash` 记录该次 run 的静态身份；
- `GET /v1/agent/versions` 返回登记表（newest first），回答「历史上有哪些 agent」；
- YAML 内容哈希见 core 的 `State.hash`；版本号由 YAML 的 `version` 提供。

### 运行

```bash
EH_HTTP_MOCK=1 uv run python course/ch18_sql/backend.py
curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"只回复：hello","run_id":"demo-1"}'
# 重启进程后仍可：
curl -s localhost:8000/v1/runs/demo-1
curl -s localhost:8000/v1/runs
curl -s localhost:8000/v1/runs/demo-1/checkpoint
curl -s localhost:8000/v1/agent/versions
curl -N localhost:8000/v1/runs/demo-1/events   # source: db 回放

# 真模型
set -a; source .env; set +a
uv run python course/ch18_sql/backend.py
```

> 默认 DB 在 `workspace/harness.db`；测试通过 `db_path=` 注入临时库。
> HTTP / WS 请求体**不带任何模型 key**，key 只从服务端环境变量读取。

## 3. 本课做到 / 没做到

**做到**：四张表 + WAL + 外键 + 两个索引；submit 登记版本与 run 行；worker 标记 running、
终态写 result/error/时间戳；trace JSONL 索引化；checkpoint 落库；排队取消也落库；
`GET /v1/runs` 分页历史；重启后 `GET /v1/runs/{id}`、SSE 回放、checkpoint、版本列表均可读；
持久化故障不改变业务终态；L0 全离线（14 例）；L1 真模型跑通「落库 → 重启 → 读取」。

**没做到（留给后续）**：不是单事务（终态 / 事件 / checkpoint 跨连接）；没有 schema migration
（生产用 Alembic）；SQLite 单机文件不适合多实例；没有在途 run 的恢复 worker（重启时 `running`
行需要启动清扫标记为 `failed` 或按 checkpoint 继续）；没有事件保留/归档策略；没有把 trace
实时写库（仍是结束后批量索引）；没有为列表加 cursor 分页。

## 4. 关联八股（~12 题）

1. **索引设计**：为什么 `run_events` 用 `(run_id, seq)` 复合主键，而不是自增 id？
   `idx_runs_created` 为什么带 `DESC`？
2. **ACID**：本章的「终态 + trace + checkpoint」是原子事务吗？不是的话，哪种失败会导致
   哪种不一致，如何补偿？
3. **WAL**：`journal_mode=WAL` 解决什么问题？为什么它允许「读写并发」却不能替代
   「多写者」的锁？
4. **ORM vs raw SQL**：什么时候该用 ORM？本章为什么选标准库 + 手写 SQL？
5. **幂等**：`INSERT OR IGNORE` / `INSERT OR REPLACE` / UPSERT 各适合哪张表？
   重放 trace 会不会产生重复事件？
6. **竞态**：为什么必须「先落库再入队」？把 `await` 放在 `put_nowait` 与 `create_run`
   之间会发生什么？修复为什么不是加锁？
7. **分页**：`LIMIT/OFFSET` 深分页的问题是什么？游标分页（`created_at, run_id`）怎么改？
8. **N+1**：历史列表如果每条都要查 checkpoint，会出现什么？如何一次查询或延迟加载？
9. **Checkpoint vs Event Sourcing**：保存状态快照和只保存事件流各有什么取舍？
   为什么本章两者都保留？
10. **恢复**：进程重启时数据库里还有 `status='running'` 的行，应该怎么处理？
    「标记失败」和「从 checkpoint 继续」各需要什么前提？
11. **水平扩展**：SQLite 文件为什么不适合多实例？换 Postgres 后哪些代码要改，
    哪些 SQL 方言要调整？
12. **保留与审计**：`run_events` 会无限增长吗？如何做 TTL / 分区 / 归档，
    同时保证审计与回放需求？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py`：`State.snapshot()` 的字段选择、`TraceRecorder` 的
  JSONL 落盘与 `flush`——本章的 `index_trace` 与 `save_checkpoint` 直接消费这些产物。
- ch16：`TraceTail`、SSE 帧编码、`Last-Event-ID`；本章只新增「数据源换成 DB」的回放分支。
- ch17：取消/审批语义；本章让排队取消也写一条终态记录，运行中取消走同一 `_persist_finish`。
- ch19：Redis 承接缓存、队列与频控；DB 仍是权威存储，二者是「热路径 / 真相源」分工。
- ch20：容器化后 SQLite 文件需要挂载卷；多副本必须换共享数据库。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch18_sql/ -q
#   → runs/run_events/agent_versions/checkpoints 落库与重启读取
#   → 历史 newest-first + 分页 + limit 上限
#   → 失败 run 仍索引 trace；排队/运行中取消落库；坏 YAML 记 failed 且 version=0
#   → 表/索引存在，journal_mode=wal

uv run ruff check course/ch18_sql/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch18_sql/ -m smoke -s
#   → 真 deepseek 作业落库；重启新 app 读 run / history / versions / checkpoint

# 手工验证
EH_HTTP_MOCK=1 uv run python course/ch18_sql/backend.py
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
> L0 使用 `starlette.testclient`，会打印一条 `httpx` 弃用警告，不影响测试结果。
