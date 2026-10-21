# tracing · 运行轨迹

每次 run 一条 JSONL，事件追加写、立即 flush。它是评测、复现和训练数据导出的底座。

## 文件

| 文件 | 职责 |
|---|---|
| `recorder.py` | `TraceRecorder`：`traces/<run_id>.jsonl`，事件 `{v, run_id, seq, ts, elapsed_ms, kind, data}` |

## 入口

```python
tracer = TraceRecorder(run_id, "traces")
tracer.event("tool.call", name="read_file", arguments={...})
tracer.close()
```

## 事件类型

`run.start`、`model.request`、`model.response`、`tool.call`、`tool.result`、`run.end`。
M1 增加：`context.compact`、`memory.write`、`subagent.start`、`subagent.end`。

## 约定

- 字段定义以 DESIGN 为准（`docs/v0.1/design/DESIGN_v0.1_agent_core.md` §8）。
- 只追加，不改写；需要修正就新开 run。
- trace 中的内容不做脱敏前不得直接公开（可能包含文件内容）。
