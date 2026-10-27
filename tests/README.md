# tests · 测试

离线测试，不需要模型额度。所有测试用 `tmp_path` 隔离文件操作。

## 文件

| 文件 | 覆盖 |
|---|---|
| `conftest.py` | `make_agent` fixture：生成合法 agent.yaml |
| `test_message.py` | OpenAI 格式互转、坏 JSON 容错 |
| `test_agent.py` | YAML 校验、state hash、prompt 渲染、`load_skill`、skill 目录 cap/分页、checkpoint/resume、trace 元数据、drift |
| `test_loop.py` | 工具调用闭环、工具错误回填、超步数终止、上下文预算与压缩、trace 事件序 |
| `test_fs.py` | 读写、`../` 越界、绝对路径、未知工具 |
| `test_memory.py` | L1/L2 记忆、注入 cap、写保护、`enabled=false` 零副作用 |
| `test_bash.py` | bwrap 隔离、超时/输出上限、越界写、`.memory` 只读、无 bwrap 拒绝执行 |
| `test_subagent.py` | 子 run 独立上下文、父子 trace 关联、子失败回填、深度上限、步数预算校验 |
| `test_ark.py` | Responses `_build_input`/`specs_to_tools`/`_parse_output`、MockTransport 往返、`create_model` ARK 路由（离线） |

## 运行

```bash
uv run pytest -q
uv run ruff check .
```

## 约定

- 新增行为必须带测试；外部依赖（模型、网络、docker）用假实现替代。
- 测试不写仓库内固定路径，一律 `tmp_path`。
- 涉及模型的测试用 `MockChatModel` 脚本化响应，不做真实调用。
- 涉及 state 的测试从 `agents/*/agent.yaml` 同构的最小 YAML 生成，不依赖仓库默认配置。
