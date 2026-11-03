# ch19 · redis（后端轨第 6 章）——缓存、队列、幂等锁与频控

ch18 的 SQLite 是**真相源**：作业终态、trace 索引、版本、checkpoint 都在库里。但每次
提交/查询都打数据库既慢又浪费；多实例部署时，进程内的 `asyncio.Queue` 和 `dict` 也无法
共享。本章把「热路径」搬到 Redis：

```
eh:job:{run_id}          Hash：作业视图（task/status/result/error/时间戳），TTL 1 天
eh:queue                 List：待跑 run_id；LPUSH 入队 / RPOP 出队
eh:lock:{run_id}         String：提交幂等锁（SET NX EX），跨实例防重复建单
eh:rate:{scope}:{window} String：固定窗口计数，INCR + EXPIRE
```

分工是 **Redis 热、SQL 冷**：Redis 丢了能回数据库重建，数据库才是权威记录。
本章默认用 `MemoryKV`（同一套 Redis 命令语义、零依赖、L0 离线）；设置
`EH_REDIS_URL` 后自动换成 `redis.asyncio`。

---

## 0. 流程图

```
POST /v1/runs
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 频控  INCR eh:rate:{client}:{window}  →  >limit ? 429 + Retry-After   │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ JobManager.submit                                                    │
│  1. 本进程 jobs[rid] 命中        → 直接返回                          │
│  2. Redis Hash eh:job:{rid} 命中 → 返回旧作业（跨实例幂等）          │
│  3. SET NX eh:lock:{rid}         → 抢建单权（失败则等对方 Hash）      │
│  4. LLEN eh:queue >= queue_size  → 429，释放锁                       │
│  5. SQL create_run（queued）                                          │
│  6. HSET eh:job:{rid} + EXPIRE                                       │
│  7. LPUSH eh:queue rid                                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ worker（任意实例）                                                    │
│  RPOP eh:queue → rid                                                  │
│  本实例无 Job？HGETALL eh:job:{rid} 水合（task/workspace/status）     │
│  已经是 cancelled/终态？跳过                                          │
│  status=running → HSET + SQL mark_running                             │
│  Agent.run(...)                                                       │
│  finally：SQL finish/index/checkpoint + HSET 终态视图                │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GET /v1/runs/{id}                                                    │
│  内存 Job → Redis Hash（热）→ SQL（冷，回填 Hash）                     │
└──────────────────────────────────────────────────────────────────────┘
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `KV` 协议 | 本章用到的 Redis 命令子集（String/Hash/List/TTL） |
| `MemoryKV` | 离线实现：惰性 TTL、INCR、SET NX、LPUSH/RPOP |
| `RedisKV` | `redis.asyncio` 适配层，`EH_REDIS_URL` 时启用 |
| `RunState` | 高层封装：作业 Hash、队列、幂等锁、固定窗口频控 |
| `SET NX EX` | 跨实例建单锁；锁过期兜底进程崩溃 |
| `LPUSH` + `RPOP` | FIFO 作业队列；消费者与生产者解耦 |
| `INCR` + `EXPIRE` | 固定窗口频控；窗口键自然过期 |
| `asyncio.to_thread`（ch18） | SQL 仍走线程池；Redis 客户端本身是 async |
| SQL（ch18） | 真相源：终态、trace、版本、checkpoint |

## 2. 实现的对象 + 用法

### KV 抽象与两套实现

```python
class KV(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, *, ex: int | None = None) -> None: ...
    async def set_nx(self, key: str, value: str, *, ex: int | None = None) -> bool: ...
    async def incr(self, key: str) -> int: ...
    async def hset(self, key: str, mapping: dict[str, str]) -> None: ...
    async def lpush(self, key: str, value: str) -> None: ...
    async def rpop(self, key: str) -> str | None: ...
```

```python
hot = build_state(redis_url=os.getenv("EH_REDIS_URL"))  # 无 URL → MemoryKV
app = create_app(service=service, db_path=db, hot_state=hot)
```

- `MemoryKV` 只用于单进程/测试，**不是 Redis 替代品**；它让高层逻辑与 L0 测试
  完全共享，真 Redis 只替换这一层；
- `RedisKV` 用 `decode_responses=True`，Hash/List 都返回 `str`；
- `RunState` 不关心底层是内存还是 Redis，只依赖 `KV` 协议。

### 作业 Hash：热视图

```python
async def put_view(self, view: JobView) -> None:
    await self.kv.hset(self._job_key(view.run_id), _job_mapping(view))
    await self.kv.expire(self._job_key(view.run_id), self.job_ttl)
```

- 字段与 `JobView` 一致；`result` 存 `result_json`，读时用 Pydantic 还原；
- TTL 默认 1 天：热状态不是权威记录，过期后由 SQL 回退；
- 每次状态迁移（queued → running → 终态）都覆盖 Hash，最后一次写入为准。

### 队列：`LPUSH` 入、`RPOP` 出

```python
async def enqueue(self, run_id: str) -> None:
    await self.kv.lpush(self.queue_key, run_id)

async def dequeue(self) -> str | None:
    return await self.kv.rpop(self.queue_key)
```

- `LPUSH` 把新任务放头部，`RPOP` 从尾部取 → **FIFO**；
- 队列只存 `run_id`，任务详情在 Hash；worker 在任意实例都能水合；
- `BRPOP` 可以省掉轮询，但 `MemoryKV` 没有阻塞语义，本章统一 50ms 轮询；
- `RPOP` 是 **at-most-once**：取出后进程崩溃，任务会丢。生产应换
  `BRPOPLPUSH`（处理中列表 + 超时回收）或 Redis Streams 消费者组（at-least-once）。

### 提交幂等：`SET NX`

```python
cached = await self.state.get_job(rid)
if cached is not None:
    return cached
if not await self.state.claim(rid):          # SET NX EX 30
    ...                                       # 等对方实例写入 Hash
...
await self.state.put_job(job)
await self.state.enqueue(rid)
```

- 两道防线：Hash 已存在 → 直接返回旧作业；Hash 还没写 → 用锁挡住并发建单；
- 锁 TTL 30s 只是「建单窗口」，不是任务租约；进程崩溃后锁自然释放；
- 建单失败（队列满等）会 `DELETE` 锁，允许客户端重试；
- SQL 的 `ON CONFLICT DO NOTHING` 仍是最后一道幂等保障。

### 缓存旁路：内存 → Redis → SQL

```python
job = manager.get(run_id)
if job is not None:
    return job.view()
cached = await hot_state.get_view(run_id)
if cached is not None:
    return cached
stored = await store.get_run(run_id)
if stored is None:
    raise HTTPException(404)
await hot_state.put_view(stored)      # 回填；失败不影响响应
return stored
```

- 热路径不碰 SQL；冷路径读 SQL 后回填 Redis；
- 回填失败只丢性能，不影响正确性；
- 数据库更新先于 Hash 更新；Hash 短暂旧值由 TTL 与下一次写收敛。

### 频控：固定窗口

```python
count = await kv.incr(f"eh:rate:{scope}:{window_id}")
if count == 1:
    await kv.expire(key, int(window) + 1)
allowed = count <= limit
retry_after = window - (now % window)
```

- 超限返回 `429` + `Retry-After`，并带 `X-RateLimit-Limit/Remaining`；
- 客户端标识默认用 `request.client.host`；生产在反向代理后要配置可信代理，
  或改用 API key / 用户 ID，**不要直接信任任意 `X-Forwarded-For`**；
- `INCR` 与 `EXPIRE` 是两条命令，不是原子操作：进程在两者之间崩溃会留下
  永不过期的计数键。生产用 Lua 脚本、`SET ... NX EX` 初始化窗口，或 Redis 7 的
  `EXPIRE NX` 组合；
- 固定窗口有边界突刺（窗口交界处可放行 2×limit）；需要平滑时换滑动窗口
  （ZSET）或令牌桶。

### 多实例边界

- **共享**：队列、作业 Hash、幂等锁、频控计数；
- **不共享**：`asyncio.Task` 句柄、`RunControl` 审批 Future、本机 trace 文件；
- 因此本章做到「跨实例排队与状态查询」，但 **WS/SSE 实时流仍只支持本实例的
  run**；运行在别的实例时：
  - `GET /v1/runs/{id}` 从 Redis/SQL 正常返回；
  - `GET .../events` 对未终态 run 返回 `409`（本机没有 TraceTail）；
  - `WS .../ws` 返回 `4404` 与「hosted on another instance」；
  - 取消运行中的远端任务需要广播，本章未实现（排队中的远端任务可以取消）。

### 运行

```bash
# 默认 MemoryKV（零依赖）
EH_HTTP_MOCK=1 uv run python course/ch19_redis/backend.py

# 真 Redis
uv pip install redis
EH_REDIS_URL=redis://localhost:6379/0 EH_HTTP_MOCK=1 \
  uv run python course/ch19_redis/backend.py

curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"只回复：hello","run_id":"demo-1"}' -i
# 连续请求可观察 429 / Retry-After 与 X-RateLimit-* 头
```

> API key 只从服务端环境变量读取；HTTP 请求体不携带任何模型 key。

## 3. 本课做到 / 没做到

**做到**：`KV` 协议 + `MemoryKV` + `RedisKV`；作业 Hash（TTL 1 天）；Redis List
FIFO 队列替代 `asyncio.Queue`，支持跨实例水合；`SET NX` 提交幂等；缓存旁路
（内存 → Redis → SQL → 回填）；固定窗口频控与 429/Retry-After；排队中的远端
任务可取消；`/health` 的 queued 改读 Redis；L0 全离线（23 例，含 ch18 的落库
回归）；L1 真模型验证「热状态命中 + 重启后 SQL 回退」。

**没做到（留给后续）**：队列不是可靠队列（RPOP 至多一次，无处理中列表/ACK）；
频控的 INCR+EXPIRE 非原子；没有跨实例事件广播（Redis Pub/Sub 或 Streams），
实时 SSE/WS 仍限本实例；没有运行中远端任务的取消；没有缓存击穿保护（互斥重建/
逻辑过期）；没有热点 key 拆分；没有 Redis Cluster/哨兵；没有把 trace 事件写入
Redis Stream；没有真正的会话存储（本章只缓存作业视图）。

## 4. 关联八股（~13 题）

1. **缓存穿透 / 击穿 / 雪崩**：三者的区别？空值缓存、布隆过滤器、互斥重建、
   逻辑过期、随机 TTL 各解决哪一个？
2. **缓存一致性**：先更新数据库再删缓存，还是先删缓存再更新数据库？延迟双删
   解决什么问题？本章为什么能接受短暂旧值？
3. **队列可靠性**：`RPOP` 为什么是 at-most-once？`BRPOPLPUSH` 的处理中列表和
   超时回收怎么做？Redis Streams 消费者组如何做 ACK 与 pending 重投？
4. **原子性**：`INCR` + `EXPIRE` 为什么不是原子的？Lua 脚本、`MULTI/EXEC`、
   `SET NX EX` 分别适合什么场景？
5. **分布式锁**：`SET NX EX` 的基本用法？为什么释放锁要校验持有者（value +
   Lua）？Redlock 的争议是什么？
6. **幂等**：提交幂等和任务执行幂等有什么区别？本章的 Hash + 锁 + SQL
   `ON CONFLICT` 各自防住了什么？
7. **限流算法**：固定窗口、滑动窗口、漏桶、令牌桶的取舍？固定窗口的边界突刺
   如何产生？
8. **TTL**：为什么作业 Hash 只保留 1 天？TTL 过期后如何回源？大量 key 同时
   过期会有什么问题？
9. **热 key / 大 key**：一个 run 的 Hash 过大或某个 key 被高频访问会怎样？
   如何拆分、限流或本地缓存？
10. **多实例实时流**：run 在实例 B、SSE 连到实例 A 时怎么办？Pub/Sub 与
    Streams 的可靠性差异？粘性会话的代价？
11. **Redis 持久化**：RDB 与 AOF 的区别？本章把 Redis 当缓存时，为什么可以
    接受丢数据？
12. **连接管理**：Redis 客户端要不要连接池？`decode_responses` 有什么影响？
    异步客户端与事件循环的关系？
13. **观测**：应该监控哪些 Redis 指标（命中率、内存、慢查询、大 key、连接数）？
    热路径故障如何降级？

## 5. 生产版延伸阅读（复用 + 回填）

- ch18：SQL 权威存储与 trace 索引；本章只在其前面加一层热路径。
- ch15–ch17：作业模型、取消/审批、SSE/WS；本章替换其中的进程内队列与状态，
  控制面仍留在本实例。
- `evolving_harness/agent.py`：`TraceRecorder` 的 JSONL 仍是实时事件来源；
  真正的跨实例事件流可把每行 `XADD` 到 Redis Stream。
- ch20：容器化后 Redis 成为独立服务；多副本共享队列与状态，SQL 也需换成
  共享数据库。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch19_redis/ -q
#   → MemoryKV：String/Hash/List、TTL、INCR、SET NX
#   → RunState：作业往返、FIFO 队列、跨实例取消排队任务、固定窗口频控
#   → HTTP：缓存命中；重启后 SQL 回退并回填；重复 run_id 只执行一次
#   → worker 从共享 Hash 水合任务；429 + Retry-After；ch18 落库回归

uv run ruff check course/ch19_redis/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch19_redis/ -m smoke -s
#   → 真 deepseek 作业；热状态命中；重启后由 SQL 回退读取

# 真 Redis（可选）
uv pip install redis
EH_REDIS_URL=redis://localhost:6379/0 EH_HTTP_MOCK=1 \
  uv run python course/ch19_redis/backend.py
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
> L0 使用 `starlette.testclient`，会打印一条 `httpx` 弃用警告，不影响测试结果。
