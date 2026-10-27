# ch04 · 技能（skills）

ch03 用**原生 function calling + 工具注册表**让 agent 连到了世界。本课再往上一层：把
**可复用的指令包**从「写死在 system prompt」中解放出来，做成**渐进披露的技能库** `SkillLibrary`——
目录只给「名 + desc + 优先级」，模型看目录决定要不要 `load_skill(name)` 把完整指令加载进 context。

对「agent 演化 / RSI」（ch10）尤其关键：技能就是 agent 能力的最小演化单元——下次可以把
`SkillLibrary` 交给演化器去增删改。

文件：`agent.py`（ch03 原样保留，本课**新增** `Skill` / `SkillLibrary`，加上**磁盘技能文件加载**
`Skill.from_markdown` / `read_skill_file` / `SkillLibrary.load_dir`，并在 `AgentLoop` 上加
`skills` 注入、`load_skill` 工具、`_inject_skill` 加载缝）。示例技能放在 `skills/` 目录：
`arxiv_search.skill.md` 是一个**真实可用的网络技能**（用只读 `web_fetch` 打 arxiv API 查论文），
也是本课的主示例。

---

## 0. 流程图

```
  磁盘技能文件（skills/*.skill.md，frontmatter: name/description/priority）
        │  SkillLibrary.load_dir()  /  Skill.from_markdown()
        ▼
        ┌────────────────────────────────────────────────────────┐
        │  run(user)                                             │
        │  context = [system(工具清单 + 技能目录), user]           │
        └──────────┬─────────────────────────────────────────────┘
                   │  目录 = 渐进披露：只有 name+desc+priority，
                   │  绝无 instructions
                   ▼
   ┌──────────────────────────────────────────────┐
   │  for step = 1 .. max_steps                    │
   └──────────┬───────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────┐
   │  complete(context, tools=schema + load_skill) │◄─────────┐
   │  （load_skill 已自动注册进 registry）           │          │
   └──────────┬───────────────────────────────────┘          │
              ▼                                               │
   ┌──────────────────────────────┐                            │
   │  result.tool_calls ?          │                            │
   └──────┬──────────────┬────────┘                            │
      有 tool_calls     无 tool_calls                          │
          │                 │  result.content 即答案            │
          ▼                 ▼        -> 返回                   │
   ┌─────────────────────────────────────┐                    │
   │ for tc in tool_calls:               │                    │
   │   obs = await registry.call(tc)     │                    │
   │                                     │                    │
   │   ├ 是 load_skill 且成功 ──────────►│                    │
   │   │   _inject_skill(context, name)  │                    │
   │   │   → 把完整 instructions 以       │                    │
   │   │     system 消息注入 context      │                    │
   │   │   （后续步可见，不只一次性观察）  │                    │
   │   │                                │                    │
   │   ├ 其余：按注册的工具执行          │                    │
   │   │   例：arxiv 技能里的 web_fetch  │                    │
   │   │   （只读网络 GET）真查 API       │                    │
   │   │                                │                    │
   │   context += [tool tc, obs]        │                    │
   └──────────────┬─────────────────────┘                    │
                  ▼                                          │
         继续下一轮 ──────────────────────────────────────────┘
```

## 1. 用的工具

| 工具 | 作用 |
|---|---|
| `load_skill`（本课自动注册的 function tool） | 模型按名字加载技能；成功 → 完整指令注入 context |
| 原生 function calling | 复用 ch03：模型吐 `tool_calls` 决定加载哪个技能 |
| `SkillLibrary.list_summary()` | 渐进披露目录：按优先级降序，只给 name+desc，可截断 |
| `Skill.from_markdown` / `read_skill_file` / `load_dir` | **磁盘技能文件加载**：frontmatter{name/desc/priority}+正文 instructions |
| `web_fetch`（ch03 补齐的只读网络 GET） | 供 arxiv 技能执行：`load_skill` 后模型按指令真打网络 API |

## 2. 实现的对象 + 用法

### `Skill(name, description, instructions, priority=0)`
一个技能 = 名字 + **给模型看的一句话描述**（directory 用）+ **完整指令包**（instructions，model
加载后才看）+ 优先级（目录排序）。描述和指令一薄一厚，正是渐进披露的原料。

### `SkillLibrary(*skills)` —— 渐进披露目录
`add(skill)` / `get(name)`（缺失抛 KeyError）/ `list_summary(max_items=20)`：

- 按 `priority` 降序列出，只给 `- 名 (优先 N): desc`，**绝不泄 instructions**（关键防线）；
- 超 `max_items` 时尾部加 `…（共 N 个技能，只显示前 M）` 截断——目录本身也可控预算。

### 磁盘技能文件：`Skill.from_markdown` / `read_skill_file` / `SkillLibrary.load_dir`
技能不只在内存里硬编码，还能**存在磁盘上**（每个 `.skill.md` 一个技能）。格式：

```markdown
---
name: arxiv_search
description: 用 arxiv Atom API 检索论文元数据（标题/摘要/作者/分类/时间）
priority: 10
---
……正文（markdown）就是 instructions，模型 load 后才注入 context……
```

`Skill.from_markdown(text)` 用极简 **frontmatter 解析**（`---` 夹 name/description/priority，
零依赖 mini-YAML，不引 PyYAML）把 `{name, description}` 当目录字段、正文当 instructions；
缺 name/description 报 `ValueError`（加载端就校验，别让坏技能进库）。`read_skill_file(path)` 读单文件，
`SkillLibrary.load_dir(dir, glob="*.skill.md")` 批量加载目录——真实技能就以这种文件组织、可版本、可演化。

`course/ch04_skills/skills/arxiv_search.skill.md` 是主示例：一个**真实可用的网络技能**，指令让它
用只读 `web_fetch` 打 arxiv Atom API（`export.arxiv.org/api/query`）查论文元数据。真冒烟里模型
`load_skill` 后连打两次 `web_fetch`，筛掉无关结果、按「标题+年份+要点」列了 3 篇论文——技能从
「教学占位」变成了**真能推动工作的指令包**。

### `AgentLoop(model, registry=..., skills=..., system=..., max_steps=...)`
ch03 循环加了一个「加载缝」：

1. 构造时给了 `skills` → 自动 `_register_load_skill()` 把 `load_skill` 工具注册进 registry；
2. `_system_prompt()` 末尾追加 `可用技能（渐进披露：先看目录…）：\n{list_summary()}`；
3. 每步 `complete(context, tools=registry.schema())` 照常；
4. 执行 `tool_calls` 时**特判**：`tc.name == "load_skill"` 且成功 → `_inject_skill(context, name)`，
   把完整 `instructions` 以 **system 消息**注入 context，**后续所有步都可见**（不是只回一次性观察）。

```python
from agent import AgentLoop, MockChatModel, ModelResult, Skill, SkillLibrary, ToolCall, Usage

skills = SkillLibrary(Skill(
    name="summarize_file", description="读取文件后给出结构化总结",
    instructions="先 read_file 读文件，再用 类/函数/要点 三节总结。", priority=10))

loop = AgentLoop(MockChatModel(script=[
    ModelResult("", tool_calls=[ToolCall("c1", "load_skill", {"name": "summarize_file"})],
                usage=Usage(1, 1)),
    ModelResult("总结如下。", usage=Usage(1, 1)),
]), skills=skills, max_steps=5)
```

### 渐进披露的取舍
坏做法是构造时把**所有技能的全部 instructions** 拼进 system prompt——技能一多，预算爆、窗口被无关
内容填满。渐进披露让 agent **按需取用**：目录开销恒定且小，`load_skill` 才是大头、只在模型真要时发生。
代价是多一轮 function call 往返——但对「目录可能很大」的场景，这性价比很高（ch10 演化用它作土壤）。

## 3. 本课做到 / 没做到

**做到**：`Skill` / `SkillLibrary` 渐进披露目录（只给 name+desc、优先级排序、可截断、绝不泄
instructions）；**磁盘技能文件加载**（frontmatter 解析 + `read_skill_file` + `load_dir`，坏技能加载端
报错）；`AgentLoop` 自动注册 `load_skill`、把目录注入 system、加载后完整指令注入 context（后续步
可见）；未知技能/加载失败**错误当数据**；与 ch03 只读工具**组合**——尤其 `web_fetch` 让 arxiv 技能
**真网络执行**（L1 冒烟：模型 load_skill → 两次 web_fetch 打 arxiv → 筛无关、列 3 篇论文）。
**没做到（留给后续）**：技能的选择融入到记忆/缓存（ch05）；子代理各自带技能（ch07）；技能作为**演化
单元**被 RSI 自动增删改（ch10 演化、ch13）——本课磁盘 `.skill.md` 文件正是它们要操作的"技能库"。

## 4. 关联八股（~12 题）

1. 「技能」和「工具」有什么区别？指令包 vs 可执行代码，粒度与触发方式差在哪？
2. 渐进披露 vs 全量塞 system prompt：预算、窗口、无关内容污染上各怎样？目录多大开始该披露？
3. 目录为何**绝不粗泄 instructions**？描述与指令一薄一厚，泄露会带来什么风险（prompt 注入/诱导）？
4. 模型怎么知道"该加载哪个技能"？只凭 desc 决定,什么叫"目录质量决定选型质量"？
5. `load_skill` 为什么做成 function tool 而不是让模型直接输出技能名？返回结构/错误处理差异？
6. 技能加载后为何注入 **system** 消息而非普通 user？role 的语义差异影响模型服从度吗？
7. 为什么只靠"一次性 observation"不够，还需要 `_inject_skill` 持久进 context？后续步看不到指令会怎样？
8. 未知技能怎么走**错误当数据**？返回的可读错误串里带上"可用列表"，模型才能自救——为什么？
9. 优先级 `priority` 的角色：目录排序 vs 能力等级？演化(Rch10)增删技能时怎样利用它？
10. 技能与 system prompt 如何共存？`self.system` 显式给了还拼不拼目录？
11. 渐进披露在**多技能组合**场景退化到什么情况（模型反复 load_skill 却不用）？max_steps 兜底？
12. 技能是否是 RSI 的"最小演化单元"？把 SkillLibrary 交给演化器后会涌现什么问题（版本、回滚）？

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/agent.py` 的 `Agent` 已把「system prompt 分层拼接 + 工具/技能注入」做成
`_system_prompt` 组合式逻辑；技能在产品版对应 `tracing` 里的 session 上下文注入。教学版最小化为
**渐进披露 + load_skill 一条缝**；产品版把技能做成可插件、可版本、可授权（ch05 记忆 / ch07 子代理 /
ch10 演化 / ch13 RSI 会逐步补齐这些接缝）。设计关键词可在 `docs/v0.1/` 的 DESIGN 中找到出处。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：渐进披露 / 注入 / 加载 / 组合 全走 Mock（含 web_fetch 的 MockTransport）
uv run pytest course/ch04_skills/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key + 可联网）：真 deepseek load_skill -> 真 web_fetch 打 arxiv
set -a; source .env; set +a
EH_SMOKE=1 uv run pytest course/ch04_skills/ -m smoke -s
uv run ruff check course/
```