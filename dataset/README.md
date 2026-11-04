# dataset · RSI 探针集（verifier-first）

RSI 轨（course/ch08–ch14）用的**自建评估数据集**。与 `course/` 平级、跨章共享（各章 `rsi.py` 快照按需读取）。
定位是**最小探针集**（~30 条）：不是大 benchmark，而是**专门为了让"评估 + 定位"有东西可咬**——每条 task
带**确定性 rubric**，能离线（L0）判分；真模型只在 L1 冒烟里判。

另有 `synth/`：由 `synth_pipeline/` 用 chat LLM 合成、机械校验的 ~100 条混合数据（math 50 + tooluse 50），
用于 `rsi/` 对真模型饱和后的后续评估（详见 `synth_pipeline/README.md`）。

## 目录

```
dataset/
  README.md              # 本文件（schema / 拆分 / 覆盖 / 用法）
  test_dataset.py        # schema 校验（离线可跑）
  rsi/                   # 最小探针集（30 条，手写混合失败模式）
    train.jsonl          # 18 条
    val.jsonl            #  6 条
    test.jsonl           #  6 条
  synth_pipeline/        # 合成链路（非课程章节）
    synth.py             # 适配器 + 校验器 + 生成器 + 去重 + split + CLI
    README.md
    tests/               # L0（Mock）+ L1（smoke）
  synth/                 # 合成产物（运行 synth.py 后生成）
    math/{train,val,test}.jsonl        # 30 / 10 / 10
    tooluse/{train,val,test}.jsonl     # 30 / 10 / 10
    tooluse/corpus.json                # 本地假语料（QA 的查证来源）
    generation_log.jsonl               # 溯源：id/domain/mode/.../model/seed
```

## schema

每行一个 JSON 对象，字段与 `course/ch08_rl_loop/rsi.py` 的 `Task` 一一对应：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 全局唯一；**前缀编码失败模式**（`fmt-` / `multi-` / `num-` / `hard-`） |
| `statement` | str | **公开给 agent** 的任务句（rollout 只喂这段） |
| `checks` | str[] | **私有 rubric**（verifier 用，永不给模型）：`word` = 答案必须含；`~word` = 答案必须不含 |

`checks` 之所以是"子串"制，是为了 `RuleScorer` **确定性、离线**判分；真模型 judge 见各章 L1。

## 拆分（train / val / test）

- **iid、按构造不相交**：三条集同分布、不同条目（无重叠 id，`test_dataset.py` 断言）。
- 用法约定（见 `course/README.md` RSI 轨）：**train 上跑 RSI 取反馈**；**val 低频守卫 + 血线（watermark）**；
  **test 只碰一次报泛化**。目的：防 RSI 把 eval 也背下来（评测污染）。
- 当前规模刻意偏小：够每段有若干条、又能离线秒跑。后续要扩再往各 split 增行（id 勿与既有重叠）。

## 失败模式覆盖（对齐 `Locator` 的归因组件）

| 前缀 | 探针意图 | 想触发的归因 |
|---|---|---|
| `fmt-` | 要求"只给结论/禁词/不写步骤"，`checks` 带 `~` 禁词 | `format`（输出形态违约 / final_format） |
| `multi-` | 要求覆盖多个要点，`checks` 拆成几条正向子串 | 漏点：没查工具→`halluc`；查了仍缺→`frame` |
| `num-` | 干净算术/推理，单一精确数字 | 多数应命中（做对照 / `ok` 基线） |
| `hard-` | 需读仓库文件/查证才能答准 | 无工具→`halluc`（疑似没查事实就下结论） |

> 说明：探针**定义 rubric**，不保证真模型一定在该模式上失败——真模型行为在 L1 才见分晓；L0 用 Mock 精确驱动。

## 各章怎么用（示例）

```python
import json, pathlib
from rsi import Task                       # 各章自己的 rsi.py 快照

def load_split(name: str) -> list[Task]:
    p = pathlib.Path("dataset/rsi") / f"{name}.jsonl"
    return [Task(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line]

train, val, test = load_split("train"), load_split("val"), load_split("test")
```

## synth · 合成数据（~100 条）

- 由 `synth_pipeline/synth.py` 生成：chat LLM 批量出题 → 机械校验 → 去重 → 每领域 6:2:2。
- 两个领域：`math/`（算式题，`mth-num` / `mth-fmt`）、`tooluse/`（假语料 + 查证型 QA，
  `tol-single` / `tol-multi`）。
- 行 schema 与 `rsi/` 完全相同（`{id, statement, checks}`）；消费时把目录从 `rsi` 换成
  `synth/<domain>` 即可，`Task` 不需要改。
- 用法纪律：train 上跑 RSI 取反馈；val 低频守卫/血线；**test 只碰一次**。
- 生成命令、校验规则与风险见 `synth_pipeline/README.md`。

## 校验

```bash
uv run pytest dataset/ -q     # rsi schema + synth 产物校验（synth 尚未生成则自动 skip），离线零额度
uv run pytest dataset/synth_pipeline/tests -q   # 合成链路 L0（全 Mock）
```
