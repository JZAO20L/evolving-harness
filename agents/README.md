# agents · Agent state

一个 agent = 一个 YAML + 一个 Python：本目录只放 YAML 静态状态，动态逻辑在
`evolving_harness/agent.py`。

## 文件

| 路径 | 职责 |
|---|---|
| `default/agent.yaml` | 默认 agent：prompt、guidance、skills、工具选择、限制 |

## 字段

| 字段 | 作用 | RSI 可变 |
|---|---|---|
| `name` | agent 名称，写入 trace | 否 |
| `version` | 状态版本，改动后递增；被拒版本号不复用 | 否（由控制器写） |
| `model` | provider / id / base_url / temperature / max_tokens | 否 |
| `limits.max_steps` | 单次 run 的步数硬上限 | 否 |
| `prompt.system` | harness 契约，所有 agent 共享 | 否 |
| `prompt.guidance` | 行为引导 | **是** |
| `tools.enabled` | 工具面；`load_skill` 自动加入 | 否 |
| `skills` | 技能目录与正文，按需 `load_skill` | **是** |

## 约定

- 密钥不进 YAML：用 `${EH_MODEL}` / `${EH_BASE_URL}`，密钥只放环境变量。
- 同一实验内 `model` 冻结；RSI 只改 `prompt.guidance` 与 `skills`。
- `skills[].name` 用小写字母、数字、连字符；名称唯一。
- 手工改内容时同步递增 `version`；`state.hash` 由解析后的 YAML 规范 JSON 计算。
