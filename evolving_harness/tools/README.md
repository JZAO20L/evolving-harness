# tools · 工具层

工具是模型的行动接口。每个工具自带 JSON Schema，注册表负责暴露与派发。

## 文件

| 文件 | 职责 |
|---|---|
| `base.py` | `Tool`、`ToolIsolation`、`@tool` 装饰器、`ToolRegistry`（`specs()` / `dispatch()`） |
| `fs.py` | 文件工具：`read_file` / `write_file` / `list_dir`，全部限制在 workspace 内 |
| `exec.py` | `bash`：bwrap 子进程隔离（无网络、workspace 可写、资源硬限），无 bwrap 时 fail-closed |

## 入口

```python
registry = ToolRegistry(fs_tools(workspace))
specs = registry.specs()                    # 传给模型
message = await registry.dispatch(tool_call) # 永远返回 Message
```

## 约定

- `dispatch` 不抛异常：未知工具、参数错误、业务错误都变成 `ERROR: ...` 内容。
- 路径一律 `resolve()` 后校验 `is_relative_to(workspace)`。
- 工具输出有字节上限（当前 64KB），超出截断并标注。
- 每个工具声明 `ToolIsolation{level, network, fs_scope, resources}`，声明写入 `tool.call` trace；档位只升不降。
- 执行类工具在 v0.1 以 spawn 级接入：`bash` 必须经 bwrap（`--unshare-net`、仅 workspace 可写、`.memory` 只读挂载、CPU/内存/输出硬限）；bwrap 缺失时拒绝执行而不是裸 `subprocess`。正式容器沙箱在 v0.3 落地。
