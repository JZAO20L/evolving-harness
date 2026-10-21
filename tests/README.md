# tests · 测试

离线测试，不需要模型额度。所有测试用 `tmp_path` 隔离文件操作。

## 文件

| 文件 | 覆盖 |
|---|---|
| `test_message.py` | OpenAI 格式互转、坏 JSON 容错 |
| `test_loop.py` | 工具调用闭环、工具错误回填、超步数终止、trace 事件序 |
| `test_fs.py` | 读写、`../` 越界、绝对路径、未知工具 |

## 运行

```bash
uv run pytest -q
```

## 约定

- 新增行为必须带测试；外部依赖（模型、网络、docker）用假实现替代。
- 测试不写仓库内固定路径，一律 `tmp_path`。
- 涉及模型的测试用 `MockChatModel` 脚本化响应，不做真实调用。
