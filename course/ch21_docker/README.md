# ch21 · docker（后端轨第 7 章）——服务打包与容器沙箱池

前 6 章把 agent 包成了可并发、可持久化、可横向排队的服务，但 `bash` 仍然跑在
**宿主进程**里（core 的 bwrap 只是弱隔离）。本章补上交付前最后两块：

1. **打包**：`Dockerfile` + `compose.yaml`，把 API、Redis、数据卷变成一个可启动的栈；
2. **隔离**：`SandboxPool` 把每次 `bash` 调用放进**一次性容器**，用 Docker 的
   namespace/cgroup 边界替代宿主直跑。

```
HTTP 请求 → JobManager → Agent loop → ToolRegistry.dispatch("bash")
                                              │
                                              ▼
                        SandboxPool.bash_handler(workspace)
                                              │  semaphore(max_size)
                                              ▼
   docker run -d --name eh-sbx-xxx  --network none --memory 512m --cpus 1
              --pids-limit 128 --read-only --tmpfs /tmp
              --cap-drop ALL --security-opt no-new-privileges
              -v <workspace>:/workspace:rw -w /workspace eh-sandbox:latest
                                              │
                              docker exec eh-sbx-xxx sh -c "<command>"
                                              │
                         timeout? ── yes ──► docker kill
                                              │
                                        docker rm -f
                                              │
                          输出截断 64 KiB → "exit=N ..." / "ERROR: ..."
```

关键取舍：**失败关闭**。Docker CLI 不可用、镜像起不来、容器启动失败，一律返回
`ERROR: ...` 给模型，**绝不退回宿主执行**。

---

## 0. 流程图（部署与生命周期）

```
┌──────────────────────── docker compose ────────────────────────┐
│                                                                │
│  ┌──────────────┐   redis://redis:6379   ┌──────────────────┐  │
│  │  api 容器     │ ─────────────────────► │  redis 容器       │  │
│  │ backend.py   │  队列 / 作业 Hash /     │ 7-alpine          │  │
│  │ SandboxPool  │  幂等锁 / 频控          └──────────────────┘  │
│  └──────┬───────┘                                              │
│         │ 挂载 /var/run/docker.sock                             │
└─────────┼──────────────────────────────────────────────────────┘
          ▼
   宿主 docker daemon
          │ docker run / exec / rm
          ▼
   ┌──────────────────┐   -v <同一绝对路径>/workspace:/workspace
   │ eh-sandbox 容器   │◄──────────────────────────────────────────
   │ alpine, uid 1000 │   --network none / --read-only / caps=∅
   └──────────────────┘
```

单次工具调用的状态机：

```
run(container) → exec(command) → 正常退出 ─┐
                     │                     ├─► rm -f → 返回 exit=N\n<output>
                     ├─ 超时 → kill ───────┤
                     └─ 启动失败 → rm -f ──┴─► 返回 ERROR: ...
```

## 1. 用的工具 / 机制

| 件 | 角色 |
|---|---|
| `SandboxSpec` | 镜像、网络、内存、CPU、pids、挂载点、user 的声明 |
| `SandboxPool` | 信号量限并发；每次调用独立容器；超时 kill；输出截断 |
| `DockerCLI` / `SubprocessDockerCLI` | 调 `docker` 的薄适配层，便于测试注入 fake |
| `sandbox_run_args` | 把硬化参数拼成 `docker run` 参数列表 |
| `ToolIsolation` | 把 `bash` 工具的隔离级别标成 `container`，供观测/审计 |
| `Dockerfile` | 应用镜像：Python + uv + docker CLI + 项目代码 |
| `sandbox.Dockerfile` | 沙箱镜像：alpine、非 root、`sleep infinity` |
| `compose.yaml` | api + redis + docker.sock + 同路径数据卷 |
| Docker namespace/cgroup | `--network none`、`--memory`、`--cpus`、`--pids-limit` |
| `--read-only` + `--tmpfs /tmp` | 根文件系统只读，仅 workspace 与 /tmp 可写 |
| `--cap-drop ALL` + `no-new-privileges` | 去 capability，禁止 setuid 提权 |

## 2. 实现的对象 + 用法

### 硬化参数：一条命令说清边界

```python
args = sandbox_run_args(spec, "eh-sbx-1", tmp_path)
# ['run', '-d', '--name', 'eh-sbx-1', '--label', 'eh.sandbox=1',
#  '--network', 'none', '--memory', '256m', '--cpus', '0.5',
#  '--pids-limit', '128', '--read-only', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
#  '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
#  '-v', '/abs/ws:/workspace:rw', '-w', '/workspace', 'sbx:test']
```

| 参数 | 挡住什么 |
|---|---|
| `--network none` | 外连、内网横向移动、把数据带出 |
| `--memory` / `--cpus` | 内存炸弹、CPU 打满宿主 |
| `--pids-limit` | fork 炸弹 |
| `--read-only` | 改写镜像层、落持久化后门 |
| `--tmpfs /tmp` | 给临时文件一个可写但易失的位置 |
| `--cap-drop ALL` | 原始套接字、挂载、ptrace 等能力 |
| `no-new-privileges` | setuid 二进制提权 |
| `-v ws:/workspace:rw` | 只把当前 run 的 workspace 暴露给容器 |

### 一次调用的完整生命周期

```python
async with self._slots:                       # 并发上限 = 同时存在的沙箱数
    name = f"eh-sbx-{uuid.uuid4().hex[:12]}"
    started = await self.cli.run(sandbox_run_args(self.spec, name, workspace),
                                 timeout=self.startup_timeout)
    if started.exit_code != 0:
        await self.cli.run(["rm", "-f", name])
        return f"ERROR: sandbox start failed: {started.output.strip()}"
    try:
        result = await self.cli.run(sandbox_exec_args(name, command),
                                    timeout=timeout_ms / 1000)
        if result.timed_out:
            await self.cli.run(["kill", name])
    finally:
        await self.cli.run(["rm", "-f", name])   # 无论成功、失败、取消都清理
```

- **池不是预热池**：信号量只限制“同时在跑的容器数”，每个 run 仍拿全新容器，
  避免上一个任务的残留文件/进程污染下一个任务；
- **双重超时**：CLI 层 `wait_for` 先杀 `docker exec` 子进程，外层再用
  `timeout_ms/1000 + 1` 兜底进程卡死；两条路径最终都会 `docker kill`；
- **失败关闭**：启动失败直接返回 `ERROR`，不重试、不降级到宿主；
- **输出截断**：stdout+stderr 合并后截到 64 KiB，避免 `yes` 之类把上下文打爆。

### 替换 core 的 bash handler（Agent 无感知）

```python
tools = build_tools(state, root)
if self.sandbox is not None:
    bash = tools.get("bash")
    if bash is not None:
        bash.handler = self.sandbox.bash_handler(root)
        bash.isolation = CONTAINER_ISOLATION
```

- 只换 `ToolRegistry` 里 `bash` 的 handler 与隔离标记，Agent 循环、schema、
  审批逻辑都不变；
- `bash_handler` 的签名仍是 `(command, timeout_ms)`，`ToolRegistry.dispatch`
  继续把异常/失败转成 `ERROR: ...` 数据；
- 没有 `bash` 工具（例如只开 `read_file`）时自然跳过。

### 启用开关

```python
pool = build_sandbox_pool()   # EH_SANDBOX=docker 才有；否则 None → core bwrap
```

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `EH_SANDBOX` | 空 | 设为 `docker` 才启用容器沙箱 |
| `EH_SANDBOX_IMAGE` | `eh-sandbox:latest` | 沙箱镜像 |
| `EH_SANDBOX_POOL` | `4` | 同时存在的沙箱上限 |
| `EH_SANDBOX_USER` | 镜像 USER | 覆盖容器内 uid:gid |
| `EH_DOCKER` | `docker` | Docker CLI 路径 |
| `EH_WORKSPACE` / `EH_TRACE_DIR` | 仓库内路径 | 容器部署时的数据目录 |

### 打包：应用镜像与沙箱镜像分离

```bash
# 沙箱镜像（小、少工具、非 root）
docker build -f course/ch21_docker/sandbox.Dockerfile \
  -t eh-sandbox:latest course/ch21_docker

# 应用镜像（构建上下文 = 仓库根）
docker build -f course/ch21_docker/Dockerfile \
  -t evolving-harness-api:latest .
```

- 两个镜像分开：应用镜像里有 uv、项目代码和 docker CLI；沙箱镜像只留 alpine
  与一个非 root 用户；
- 应用容器通过挂载 `/var/run/docker.sock` 使用宿主 daemon（docker-out-of-docker）；
- **sibling 容器路径陷阱**：应用把 workspace 的**容器内路径**交给宿主 daemon，
  所以 bind 挂载必须让宿主机与容器内路径**字符串一致**。`compose.yaml` 用同一个
  `EH_DATA_DIR` 作 `source` 和 `target` 来保证这一点：

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
  - type: bind
    source: "${EH_DATA_DIR:-/tmp/eh-data}"
    target: "${EH_DATA_DIR:-/tmp/eh-data}"
```

- `EH_HTTP_MOCK=1` 可无 key 启动；真实模型 key 只从**服务端环境**读取，
  HTTP 请求体不携带任何 key。

## 3. 本课做到 / 没做到

**做到**：`SandboxSpec` + `sandbox_run_args` 全套硬化参数；`DockerCLI` 协议与
子进程实现；`SandboxPool` 的并发上限、启动失败关闭、超时 kill、`rm -f` 兜底、
64 KiB 输出截断；`bash` handler 无侵入替换与 `ToolIsolation(level="container")`；
`EH_SANDBOX` 开关；应用/沙箱两个 Dockerfile、compose 栈、`.dockerignore`；
L0 全离线（33 例，含 ch15–ch20 回归；沙箱用 fake Docker CLI），L1 真模型冒烟。

**没做到（留给后续）**：容器不是虚拟机，仍共享宿主内核，内核/镜像漏洞可逃逸；
没有 rootless Docker、seccomp/AppArmor 自定义策略、用户命名空间；没有镜像签名
与漏洞扫描；没有预热的容器池（每 run 起停有百毫秒级开销）；没有限制磁盘写入
（workspace 无 quota）；没有 GPU/特权模式的例外通道；没有跨宿主调度（多机
`docker context` / Kubernetes）；没有把 trace 与 workspace 做成对象存储；
compose 的单机 Redis 仍非高可用。

## 4. 关联八股（~12 题）

1. **镜像分层**：`Dockerfile` 每条指令如何形成 layer？为什么先 `COPY pyproject`
   再 `COPY` 源码？多阶段构建解决什么？
2. **容器 vs 虚拟机**：namespace 与 cgroup 分别隔离什么？为什么“共享内核”意味着
   沙箱不是强安全边界？
3. **网络模式**：`none` / `host` / `bridge` / 自定义 bridge 的差异？沙箱为什么
   选 `none`？需要访问包管理器时怎么办（离线镜像 / 代理白名单）？
4. **资源限制**：`--memory` 与 `--memory-swap` 的关系？`--cpus` 是配额还是绑核？
   `--pids-limit` 如何挡住 fork 炸弹？
5. **文件系统**：`--read-only` + `--tmpfs` + bind mount 的组合为什么常用？
   `:ro` / `:rw` / `nosuid,noexec` 各防什么？
6. **capability**：`--cap-drop ALL` 之后容器还能做什么？`no-new-privileges`
   如何阻止 setuid 提权？
7. **docker-out-of-docker**：挂 `docker.sock` 与 DinD 的区别？前者为什么等于把
   宿主 root 权限交给容器？sibling 容器的 bind 路径为什么会错位？
8. **沙箱池**：为什么不做预热池？信号量限并发与连接池限流的异同？容器起停
   ~100ms 的成本如何摊销（池化 / 长驻 / 微虚拟机）？
9. **超时与清理**：只杀 `docker exec` 为什么不等于杀掉容器内进程？为什么必须
   `docker kill`？`rm -f` 放 `finally` 的意义？
10. **失败关闭**：沙箱不可用时为什么不能退回宿主执行？可用性与安全性的取舍？
11. **输出治理**：为什么合并 stdout/stderr 并截断？截断会不会破坏 JSON 工具协议？
    如何给模型保留“被截断”信号？
12. **部署观测**：容器化后应监控哪些指标（重启次数、OOMKilled、pids、磁盘、
    exec 延迟）？`HEALTHCHECK` 与 liveness/readiness 的关系？

## 5. 生产版延伸阅读（复用 + 回填）

- ch15–ch18：HTTP/并发/SSE/WS；本章只替换 `bash` 的执行位置，控制面不变。
- ch19–ch20：SQL 权威存储与 Redis 热路径；compose 里 Redis 成为独立服务。
- `evolving_harness/tools/bwrap.py`（core）：产品默认的弱隔离；本章的
  `SandboxPool` 是它的容器化升级路径。
- `evolving_harness/agent.py`：`TraceRecorder` 仍在宿主写 JSONL；容器内命令的
  输出经工具结果回填，不直接写 trace。

## 6. 运行与验证

```bash
# L0 离线（零额度、零 Docker）：fake CLI 验证参数硬化/并发/超时/失败关闭
uv run pytest course/ch21_docker/ -q
#   → 33 例（含 ch15–ch20 回归）；1 例 L1 冒烟默认跳过
uv run ruff check course/ch21_docker/

# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch21_docker/ -m smoke -s

# 本地跑（不开容器沙箱；bash 仍用 core bwrap）
EH_HTTP_MOCK=1 uv run python course/ch21_docker/backend.py

# 真实容器沙箱（需本机 Docker）
docker build -f course/ch21_docker/sandbox.Dockerfile \
  -t eh-sandbox:latest course/ch21_docker
EH_SANDBOX=docker EH_HTTP_MOCK=1 uv run python course/ch21_docker/backend.py

# 整套栈
export EH_DATA_DIR=$PWD/course/ch21_docker/.data
mkdir -p "$EH_DATA_DIR"/{workspace,traces}
docker build -f course/ch21_docker/sandbox.Dockerfile \
  -t eh-sandbox:latest course/ch21_docker
set -a; source .env; set +a          # key 只进服务端环境
docker compose -f course/ch21_docker/compose.yaml up --build

curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"task":"用 bash 输出 hello","run_id":"docker-1"}' -i
```

> 首次运行需要 FastAPI 依赖：`uv sync --extra dev`。
> L0 使用 `starlette.testclient`，会打印一条 `httpx` 弃用警告，不影响测试结果。
