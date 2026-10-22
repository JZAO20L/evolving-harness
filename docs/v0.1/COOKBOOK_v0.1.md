# COOKBOOK v0.1

> **状态**：`active`
> **原则**：本文件只放可复制执行的步骤；为什么这样设计见 [DESIGN_v0.1_agent_core.md](DESIGN_v0.1_agent_core.md)。

## 1. 安装

```bash
cd evolving-harness
uv sync --extra dev
```

## 2. 跑测试

```bash
uv run pytest -q
uv run ruff check .
```

预期：`18 passed`，`All checks passed!`。

## 3. 离线运行（不消耗模型额度）

```bash
uv run eh run "write a note" --mock --workspace ./workspace --traces ./traces
```

预期输出：

```
MOCK: write a note
[agent] default v1 <hash8>
[trace] traces/<run_id>.jsonl
```

默认加载 `agents/default/agent.yaml`；换一个 agent 用 `--agent <path>`。

## 4. Agent 配置

一个 agent = 一个 YAML + 一个 Python：

- `agents/<name>/agent.yaml`：静态状态。prompt、guidance、skills、工具选择、步数上限都在这里。
- `evolving_harness/agent.py`：动态逻辑。加载、渲染、循环、trace，不随 RSI 改变。

常用改动：

```yaml
limits:
  max_steps: 30          # 模型一直调工具时提高上限

prompt:
  guidance: |            # RSI 的可变面之一
    - Inspect before writing.

skills:                  # RSI 的可变面之二
  - name: verify-before-finish
    description: Re-read changed files before answering.
    instructions: |
      1. List the files you changed.
      2. Re-read each one.
      3. Compare with the task.
```

约束：

- `tools.enabled` 只列真实存在的工具；`load_skill` 在存在 skills 时自动加入，不要手写。
- `skills[].name` 只用小写字母、数字和连字符。
- `model.id` / `model.base_url` 建议写成 `${EH_MODEL}` / `${EH_BASE_URL}`，密钥只放环境变量。
- 改完 YAML 后，trace 里的 `state.version` / `state.hash` 会变化；一次 run 期间改文件会得到 `status=drift`。

## 5. 接真实端点

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
EH_MODEL=<模型名>
EH_BASE_URL=http://<host>:<port>/v1
EH_API_KEY=<key，可留空>
```

运行：

```bash
uv run eh run "在 workspace 下创建 notes.txt，写入三条待办" --workspace ./workspace
```

vLLM 示例：`EH_BASE_URL=http://localhost:8000/v1`，`EH_API_KEY` 可为空。

也可以用命令行临时覆盖：`--model` / `--base-url` / `--api-key`；它们不写回 YAML。

## 6. 查看 trace

```bash
ls traces/
cat traces/<run_id>.jsonl
```

看 state 与 runtime：

```bash
grep '"kind": "run.start"' traces/<run_id>.jsonl
```

按事件类型过滤：

```bash
grep '"kind": "tool.result"' traces/<run_id>.jsonl
```

事件字段定义见 DESIGN §8。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `error: agent config not found` | `--agent` 路径不对 | 检查 `agents/<name>/agent.yaml` |
| `error: model id is required` | YAML / 环境变量都没给模型名 | 设置 `EH_MODEL`，或传 `--model` |
| `error: base_url is required` | 未设置 `EH_BASE_URL` | 设置 `.env` 或传 `--base-url` |
| `error: unknown tool in tools.enabled` | 写了不存在的工具名 | 用 `read_file` / `write_file` / `list_dir` |
| `error: duplicate skill name` | skills 里有重名 | 改掉重复的 `name` |
| HTTP 401 | key 缺失或不匹配 | 检查 `EH_API_KEY` |
| HTTP 404 | base_url 少了或多了 `/v1` | 兼容端点通常以 `/v1` 结尾 |
| HTTP 400 + tool_calls 相关 | 端点不支持 tools | 换支持 function calling 的模型 |
| `agent exceeded N steps` | 模型没有停止调用工具 | 提高 YAML 里的 `limits.max_steps` |
| `[agent]` 后的 hash 变了 | YAML 内容变了 | 确认这是预期修改；RSI 靠它区分版本 |
| 工具返回 `ERROR: path escapes workspace` | 路径越界 | 使用 workspace 内的相对路径 |

## 8. 清理

```bash
rm -rf traces/ workspace/ .pytest_cache/
```
