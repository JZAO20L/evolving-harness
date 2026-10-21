# tools · 工具层

工具是模型的行动接口。每个工具自带 JSON Schema，注册表负责暴露与派发。

## 文件

| 文件 | 职责 |
|---|---|
| `base.py` | `Tool`、`@tool` 装饰器、`ToolRegistry`（`specs()` / `dispatch()`） |
| `fs.py` | 文件工具：`read_file` / `write_file` / `list_dir`，全部限制在 workspace 内 |

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
- 执行类工具（shell / 代码）在 v0.2 接入，必须声明隔离档位与资源预算，不能直接 `subprocess` 起在 API 进程里。
