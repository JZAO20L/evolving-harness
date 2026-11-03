# ch14 · fastapi（后端轨第 1 章）——把一次 Agent run 包成 HTTP API

前 13 章都在进程内：`Agent.run(task)` 拿字符串、返回 `RunResult`。本章第一次把 Agent 放到**服务边界**后面：
客户端发 HTTP 请求，服务端加载 YAML、跑循环、返回结构化响应。

`backend.py` = 本章 hero；`agent.py` 不复制、不改写——直接复用生产核
`evolving_harness.agent`。这正是 ch00 的结论：**接口是插座，core 不因换传输而改**。

```
POST /v1/runs ──> pydantic 校验 ──> RunService ──> State.load(agent.yaml)
                                                     │
                                                     ├─ create_model / build_tools
                                                     ├─ Agent.run(task)
                                                     └─ trace -> traces/<run_id>.jsonl
                                     │
                                     ▼
                          RunResponse（run_id/final/status/steps/usage/hash）
```

---

## 0. 流程图

```
                         ┌──────────────────────────────┐
                         │ client: curl / SDK / WebUI   │
                         └──────────────┬───────────────┘
                                        │ HTTP JSON
                                        ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ FastAPI 路由层                                                        │
   │  GET  /health              → 服务活着 + 当前 agent 路径                 │
   │  POST /v1/runs             → 同步跑一次 run，201                     │
   │  GET  /v1/runs/{run_id}    → 查本次进程内已完成的结果，404 兜底        │
   │                                                                      │
   │  校验：task 非空、run_id 只含 [A-Za-z0-9_-]、extra=forbid              │
   │  错误码：400 配置/参数 / 409 步数耗尽 / 422 校验 / 502 模型上游 / 500 │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ RunService（薄编排，不实现循环）                                       │
   │  State.load(agent.yaml) → 冻结静态段 + hash                           │
   │  create_model(state)                                                  │
   │  build_tools(state, workspace)                                        │
   │  Agent(...).run(task, run_id)   ← 同一个 v0.1 循环                     │
   │  RunResponse.from_result(...)                                         │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  ▼
                    traces/<run_id>.jsonl（run.start … run.end）
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `FastAPI` | 路由、依赖注入容器、OpenAPI 文档（`/docs`） |
| `pydantic` `RunRequest` / `RunResponse` | 请求/响应契约；`extra="forbid"` 防未知字段夹带 |
| `RunService` | HTTP 与 Agent 之间的薄编排；可注入 `model_factory` 供离线测试 |
| `evolving_harness.agent` | 冻结的 `State` / `Agent` / `build_tools` / `create_model` |
| `RunResult` | 循环的产出：final / status / steps / usage / trace_path |
| `HTTPException` + 分类捕获 | 把领域异常映射为稳定 HTTP 语义 |
| `httpx.ASGITransport` | L0 测试不启端口、直接打 ASGI app |

## 2. 实现的对象 + 用法

### 请求/响应契约

```python
class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=20_000)
    run_id: str | None = Field(default=None, max_length=64)

class RunResponse(BaseModel):
    run_id: str
    final: str
    status: str          # ok / drift / max_steps / error
    steps: int
    usage: UsageOut      # prompt / completion / total
    state_version: int
    state_hash: str      # YAML 静态段指纹，用于复现与 RSI 版本对齐
    trace_path: str
```

`task` 是**唯一公开输入**；私有 rubric、模型 key、其它请求方不该看的东西都不进 HTTP。

### 服务层是可注入的

```python
@dataclass
class RunService:
    agent_path: str | Path = DEFAULT_AGENT
    workspace: str | Path = DEFAULT_WORKSPACE
    trace_dir: str | Path = DEFAULT_TRACE_DIR
    mock: bool = False
    model_factory: Callable[[State], ChatModel] | None = None
    runs: dict[str, RunResponse] = field(default_factory=dict)

    async def run(self, task: str, run_id: str | None = None) -> RunResponse: ...
```

- 生产：`model_factory=None`，走 `create_model(state)`（读环境变量）。
- 测试：注入 `lambda state: MockChatModel([...])`，不联网、不烧额度。
- 每次请求 `State.load` 一遍，保证 run 期间 YAML 是同一份静态身份；run 结束的 drift 检查仍在 Agent 内部。

### 错误码是契约的一部分

| 场景 | 状态码 | 来源 |
|---|---|---|
| task 空白 / 缺字段 / 夹带未知字段 | 422 | pydantic |
| YAML 不存在、字段非法 | 400 | `AgentConfigError` |
| `max_steps` 耗尽 | 409 | `MaxStepsExceeded` |
| 模型上游 4xx/5xx/超时 | 502 | `ModelError` |
| 未捕获异常 | 500 | FastAPI 默认 |

不要让所有失败都变成 500：客户端要能区分「我的请求错了」和「你服务坏了」。

### 应用工厂与生产装配

```python
def create_app(service: RunService | None = None, *, agent_path=..., ...) -> FastAPI:
    service = service or RunService(...)
    app = FastAPI(...)
    app.state.service = service
    ...
```

工厂模式让测试注入 service，也让 ch15–ch20 可以在这里替换队列、SSE、数据库版本，而不改路由语义。

### 运行

```bash
# 离线：不联网，用 Mock 模型
EH_HTTP_MOCK=1 uv run python course/ch14_fastapi/backend.py
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"只回复：hello"}'

# 真模型：先载入环境变量，再启动
set -a; source .env; set +a
uv run python course/ch14_fastapi/backend.py
```

模型 key 只从服务端环境变量读取，**永不进入请求体/响应体**。

## 3. 本课做到 / 没做到

**做到**：`POST /v1/runs` 同步跑完整 Agent；`GET /v1/runs/{id}` 查已完成结果；`/health`；pydantic 请求/响应契约；配置/步数/模型错误映射到 400/409/502；run_id 安全校验；trace 落盘；服务层可注入 Mock；L0 全离线；L1 真模型走 HTTP 契约。

**没做到（留给后续）**：请求是同步阻塞的（长任务会占住连接）；没有作业队列/并发上限（ch15）；没有 SSE/WS 流式（ch16/17）；结果只在进程内存，重启即丢（ch18）；没有鉴权、限流、幂等键（ch19/生产加固）；没有 per-run workspace 隔离（ch15）；模型客户端每请求新建、未做连接池复用。

## 4. 关联八股（~12 题）

1. **为什么用 FastAPI**？它相比 Flask/Django 在 async、pydantic 校验、OpenAPI 上分别给了什么？
2. **pydantic 在边界的作用**：`extra="forbid"`、长度限制、自定义 validator 各防住了什么？
3. **REST 语义**：为什么创建 run 用 `POST /v1/runs` 而不是 `GET /v1/run?task=...`？
4. **错误码分层**：400 / 409 / 422 / 502 各自代表谁的错？把所有错误都返 500 有什么问题？
5. **同步 HTTP 的局限**：一个 run 跑 2 分钟时，连接、超时、重试、用户等待分别发生什么？
6. **应用工厂**：为什么用 `create_app(service)` 而不是模块级全局 app？测试与生产各得到什么？
7. **可注入模型工厂**：它怎样让 L0 测试完全离线？会不会掩盖真实适配器的问题？
8. **`state_hash` 进响应**：它解决什么复现问题？为什么请求里不该让客户端指定 hash？
9. **密钥边界**：为什么模型 key 只能从环境变量读，不能进请求体？日志/trace 还要注意什么？
10. **结果存储**：本章用进程内 `dict`，重启丢失；哪些接口语义会因此改变？ch18 怎么补？
11. **幂等**：同一个 `run_id` 重试 POST 会发生什么？应该返回第一次结果还是重新跑？
12. **OpenAPI / `/docs`**：为什么把 schema 暴露出来是能力而不是风险？生产要不要关？

## 5. 生产版延伸阅读（复用 + 回填）

- `evolving_harness/agent.py`：`State` / `Agent` / `RunResult` 的生产实现；本章不改它。
- `evolving_harness/cli/__init__.py`：同一个 `Agent.run` 的 CLI adapter，对比 HTTP adapter。
- `docs/v0.1/DESIGN_v0.1_agent_core.md`：v0.1 的接口与约束；v0.3 后端服务的设计入口。
- ch15：把同步 HTTP 升级成作业队列 + worker + 并发上限。
- ch16/17：把一次性响应升级成 SSE/WS 事件流。
- ch18：把进程内 `runs` 与文件 checkpoint 换成数据库。

## 6. 运行与验证

```bash
# L0 离线（零额度）
uv run pytest course/ch14_fastapi/ -q
#   → health / 契约字段 / run 后可按 id 查询 / 404
#   → 空白 task、未知字段、非法 run_id → 422
#   → 坏 YAML → 400；步数耗尽 → 409；模型错误 → 502
#   → trace 事件序 run.start … run.end

uv run ruff check course/ch14_fastapi/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch14_fastapi/ -m smoke -s
#   → 真 deepseek 经 httpx ASGITransport 走 POST /v1/runs，断言 201 + status=ok + trace 落盘
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
