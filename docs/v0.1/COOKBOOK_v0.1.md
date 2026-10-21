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

预期：`10 passed`，`All checks passed!`。

## 3. 离线运行（不消耗模型额度）

```bash
uv run eh run "write a note" --mock --workspace ./workspace --traces ./traces
```

预期输出：

```
MOCK: write a note
[trace] traces/<run_id>.jsonl
```

## 4. 接真实端点

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

## 5. 查看 trace

```bash
ls traces/
cat traces/<run_id>.jsonl
```

按事件类型过滤：

```bash
grep '"kind": "tool.result"' traces/<run_id>.jsonl
```

事件字段定义见 DESIGN §8。

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `error: model id is required` | 未设置 `EH_MODEL` | 设置 `.env` 或传 `--model` |
| `error: base_url is required` | 未设置 `EH_BASE_URL` | 设置 `.env` 或传 `--base-url` |
| HTTP 401 | key 缺失或不匹配 | 检查 `EH_API_KEY` |
| HTTP 404 | base_url 少了或多了 `/v1` | 兼容端点通常以 `/v1` 结尾 |
| HTTP 400 + tool_calls 相关 | 端点不支持 tools | 换支持 function calling 的模型 |
| `agent exceeded N steps` | 模型没有停止调用工具 | 提高 `--max-steps` 或收紧系统提示 |
| 工具返回 `ERROR: path escapes workspace` | 路径越界 | 使用 workspace 内的相对路径 |

## 7. 清理

```bash
rm -rf traces/ workspace/ .pytest_cache/
```
