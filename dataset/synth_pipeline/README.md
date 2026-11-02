# dataset/synth_pipeline · 100 条混合数据合成链路（非正式章节）

RSI 轨（ch10–ch13）的数据基建。`dataset/rsi` 的 30 条探针对 deepseek 已饱和（基线全 1.0 → RSI 只能
noop/hold），所以这里用 **chat LLM 批量出题 + 机械校验** 造一批可持续、判分确定、按领域可分的评估数据。

**不是课程章节**：代码在 `dataset/synth_pipeline/`，产物在 `dataset/synth/`，不进 chNN 轨序。
行 schema 与 `course/chNN_*/rsi.py` 的 `Task{id, statement, checks}` 严格兼容，可直接被 RSI 章节消费。

## 0. 流程图

```
chat LLM 批量生成
   ├─ math：一次生成多道算式题 {statement, expr, answer, forbid}
   └─ tooluse：先写本地假语料 docs，再据 docs 出查证型 QA
                    {statement, answers, evidence, forbid}
                │
                ▼
机械校验（verifier-first：任一不过即丢，绝不写脏行）
   math：安全 AST 求值 == answer；≥2 运算符；答案不在题干；禁词合法
   tooluse：证据存在；答案逐字在证据中且全语料唯一；题面含证据的 distinctive 关键词；
            确定性检索 top-5 必须命中所有证据
                │
                ▼
去重（题干归一化 / 算式 / 答案集合） → 每领域 6:2:2
                │
                ▼
dataset/synth/{math,tooluse}/{train,val,test}.jsonl
        + tooluse/corpus.json + generation_log.jsonl
```

原则：**先生成再校验，只落盘通过校验的行**；领域与失败模式只编码在 `id` 前缀，生成 prompt 里没有
split 信息；**test 只用于最终评估，永不参与生成/改写提示的迭代**。

## 1. 产物与 schema

```
dataset/synth/
  math/{train,val,test}.jsonl        # 30 / 10 / 10
  tooluse/{train,val,test}.jsonl     # 30 / 10 / 10
  tooluse/corpus.json                # {"docs":[{doc_id,title,text,keywords}]}
  generation_log.jsonl               # 溯源：id/domain/mode/.../model/seed
```

每行恰为三个字段（与 `Task` 兼容）：

| 字段 | 说明 |
|---|---|
| `id` | 全局唯一；前缀编码领域+失败模式：`mth-num-` / `mth-fmt-` / `tol-single-` / `tol-multi-` |
| `statement` | **公开给 agent** 的任务句 |
| `checks` | **私有 rubric**：`word` 必含、`~word` 必不含（`RuleScorer` 确定性子串判分） |

`generation_log.jsonl` 保留 `expr/answer` 或 `answers/evidence`、`model`、`seed`，用于溯源；
消费侧只读三字段行，不被 provenance 污染。

## 2. 两个领域

### math（算式题）

生成元素：`{statement, expr, answer, forbid[]}`。校验规则：

| 规则 | 说明 |
|---|---|
| 安全 AST | 只允许整数、括号、`+ - * / // % **`；`**` 指数须为 `0..6` 的整数字面量 |
| 长度 | 表达式 ≤200 字符、运算符 ≤8 |
| 精确相等 | `answer` 归一化为 `Fraction` 后必须等于表达式求值（支持 `n/d`） |
| 难度下限 | 运算符 ≥2（防 `1+1` 水文） |
| 防抄题 | 规范化答案不得作为子串出现在 `statement` |
| 答案尺度 | 规范化答案 ≤8 字符、分子/分母 ≤12 位 |
| 禁词 | 非空、唯一、不得等于答案；`checks = [answer] + ["~"+w]` |

`forbid` 为空 → `mth-num`；有禁词（"只给数字/禁止写步骤"）→ `mth-fmt`。

### tooluse（本地假语料 + 查证型 QA）

**A. 语料**：每篇 `{doc_id, title, text, keywords[]}`；`doc_id` 唯一、`text` 非空、
每个 keyword 逐字出现在该篇 `text` 中。

**B. QA**：`{statement, answers[], evidence[], forbid[]}`。校验规则：

| 规则 | 说明 |
|---|---|
| 证据存在 | `evidence` 为 1–2 个已存在的 `doc_id` |
| 答案有据 | 每个 answer（≥2 CJK 或 ≥3 ASCII 字符）逐字出现在 ≥1 篇证据文本 |
| 全语料唯一 | answer 不得出现在任何非证据文档；也不得出现在题干 |
| 可检索 | 每篇证据至少有一个 **distinctive keyword**（全语料只出现在该篇）出现在题干 |
| top-5 命中 | `search(statement, corpus, top_k=5)` 必须包含全部证据 |

检索是**确定性 keyword-overlap**：ASCII 词 + CJK bigram，按重叠数降序、`doc_id` 升序平局。
`tol-single` = 1 篇证据；`tol-multi` = 2 篇（为 RSI 的 `frame` 归因提供素材）。

## 3. 去重与切分

- 去重键：题干 `去空白 + casefold`；math 另加 `(expr, answer)`，tooluse 另加答案集合。
- `split(items, seed, ratios=(6,2,2))`：`random.Random(seed)` 洗牌后按 `n*6//10` / `n*2//10`
  切分，余数归 test；同 seed 可复现，split 两两不相交、id 全局唯一。
- 50 条 → 30/10/10；小配额（如 6 条）→ 3/1/2，测试按公式断言。

## 4. 命令

```bash
# 正式生成 100 条（真模型；产物落 dataset/synth/）
set -a; source .env; set +a
uv run python dataset/synth_pipeline/synth.py --math 50 --tooluse 50 \
  --out dataset/synth --seed 42
# 可选：--max-calls 80（默认） / --docs 24 / --model / --base-url

# L0 离线（全 Mock，零额度）
uv run pytest dataset/synth_pipeline/tests -q

# L1 冒烟（真 deepseek，2 math + 2 tooluse）
EH_SMOKE=1 uv run pytest dataset/synth_pipeline/tests/test_smoke.py -q -s

# 生成后复验（含 rsi 探针集 schema）
uv run pytest dataset/ -q
```

环境变量：模型 `EH_MODEL`（默认 `deepseek-v4-flash-ga-260731`）、端点 `EH_BASE_URL`（默认 ARK v3）、
key `ARK_API_KEY` 或 `EH_API_KEY`。**CLI 只读 `os.environ`，代码不主动读 `.env`**；
测试里的 `_load_env()` 仅供测试使用。

退出码：完整配额 → `0`；配额不足（预算/校验淘汰）→ 已落盘通过的数据并返回 `1`；
缺 key → `2`。

> **代理坑**：若 shell 的 `NO_PROXY` 含裸 IPv6 CIDR，httpx 建 client 时会报
> `InvalidURL: Invalid port ':'`。跑 CLI 时在前面加
> `NO_PROXY= all_proxy= ALL_PROXY= no_proxy=localhost,127.0.0.1`（保留 `HTTP_PROXY`）；
> L1 测试的 autouse fixture 已自动做同样的清理。

## 5. 扩展

- **新领域**：加一个 `generate_<domain>` + `_validate_<domain>` + `dedupe`，在 `synthesize` 里接上
  并落 `dataset/synth/<domain>/`。校验必须纯机械、离线可跑。
- **新 mode**：让 `_validate_*` 推导 mode 并在 `assign_ids` 时编码进 id 前缀；不要给模型看 split。
- **外部硬 test**（如 HMMT）：暂不接入；需要时写独立转换器，只读外部数据、不改生成链路。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 批量 JSON 坏格式（头号失败） | batch ≤10；解析失败减半重试一次；再丢；每批独立计数 |
| 检索校验误杀 | 题干必须点出证据的 distinctive keyword；top-5 命中；不过即丢，靠重试补配额 |
| 短答案歧义 | answer 最小长度 + 全语料扫描；有歧义直接丢 |
| 成本/时延 | docs 1 批 + math 数批 + QA 数批；`--max-calls` 封顶；数据落盘后不重跑 |
| 评测污染 | 校验 prompt 不含 split；test 只评一次；`generation_log` 记录 model/seed |
| 密钥纪律 | 代码不读 `.env`；key 只从环境变量进入 |
