# ch02 · Agent 循环（agent-loop）

上一课（ch01）把「一次调用」立稳了。本课把它变成**循环**：不再调一次就返回，而是
**组 context → 调模型 → 解析 act → 观察回填 → 判停**，一直走到「回答 or 步数耗尽」。
这是 ReAct（Thought / Act / Observation）的最小落地。

文件：`agent.py`（ch01 的四个对象原样保留，本课**新增** `Act` / `parse_act` / `AgentLoop` / `LoopResult`）。

---

## 0. 流程图

```
   ┌─────────────────────────────────────────────┐
   │  run(user)                                  │
   │  context = [system 提示, user 输入]           │
   └──────────┬──────────────────────────────────┘
              │
              ▼
   ┌───────────────────────────┐            ┌────────────────────────┐
   │  for step = 1 .. max_steps │◄───────────│ 观察回填：把本轮结果    │
   └──────────┬────────────────┘            │ 拼回 context，带上一步  │
              ▼                             └──────────┬─────────────┘
   ┌───────────────────────────┐                       ▲
   │ ① complete(context)       │                       │
   │   （调模型拿回文本+用量）    │                       │
   └──────────┬────────────────┘                       │
              ▼                                        │
   ┌───────────────────────────┐                       │
   │ ② parse_act(文本)          │   （context 越滚越大，  │
   │   → Act{action_type,       │    引出 ch05 预算/压缩) │
   │      params: JSON}         │                       │
   └──────────┬────────────────┘                       │
              ▼                                        │
   ┌───────────────────────────┐                       │
   │ ③ 判停：action_type==finish ?│                      │
   └──────┬──────────────┬─────┘                       │
          │是            │否                           │
          ▼              ▼                            │
   ┌───────────────┐   ┌─────────────────────────┐     │
   │ return         │   │ context += [assistant     │     │
   │ LoopResult(    │   │   act, tool observation]│     │
   │  answer,       │   └──────────┬──────────────┘     │
   │  finished,     │              │                    │
   │  steps, usage) │              └────────────────────┘
   └───────────────┘
     （步数耗尽 -> finished=False）
```

## 0.5 关键设计 · agent 不是跑完才理人（运行期用户输入注入）

一个常被忽略但很重要的点：agent **不是跑完一整轮才回话**，而是运行中**持续可被用户注入口头指令**。
用户随时可能在中途插一句（改方向 / 插一步 / 换对象），agent 应该**优先处理**它，而不是装作没听见、
等整轮跑完。本课在循环上加一个**注入通道 + 每步边界优先处理**的最小落地：

```
外部协程（用户/事件源）                  AgentLoop.run(user)
        │ 运行期间 push 指令                 │
        ▼                                   ▼
  ┌────────────────┐   for step=1..max:    ┌─────────────────────────────┐
  │ input_queue     │ ──每步边界非阻塞取出──▶│ _ingest_user_input(context) │
  │  (asyncio.Queue)│                       │  把注入塞成“最新 user 位”：   │
  └────────────────┘                       │  Message(user, token=        │
                                           │   【用户运行中插入，请优先处理】)│
                                           └──────────────┬──────────────┘
                                                          ▼
                                            模型下“一步”优先据此重新定向
                                            （不是等整轮跑完才理人）
```

要点（服务"放边界、不放循环"的贯穿原则）：
- **通道是外部注入的**：`AgentLoop(..., input_queue=asyncio.Queue())`。模型/tool `await` 期间事件循环
  空闲，注入协程趁隙把指令推进来——所以 agent 是"运行中持续可被接近"，不是阻塞式"跑完再回"。
- **边界非阻塞检查**：每步 `get_nowait()`，没新指令立刻继续（不卡循环）；有指令就把它置为**最新一条 user**
  并加显式"请优先处理"标注——ReAct 里越新的 user 越具权威，模型据此重新定向，下一"步"就优先响应。
- **优先 ≠ 丢弃**：旧 context 不删（仍有连续性），只是新指令压到顶部优先处理；丢弃/替换规划留给 ch06。
- 更重的**阻塞等用户确认**（cancel / approve、反压、心跳）是挂起点，属于后端轨 ch17 / ch20 的活。

## 1. 用的工具

| 工具 | 作用 |
|---|---|
| `json` | 解析模型输出的 JSON act（含 fenced 代码块 / 叙述包裹的容错） |
| `dataclasses`（沿用） | `Act` / `LoopResult` 两个新的值对象 |

> 本课**刻意不引入工具注册表 / function calling**（那是 ch03）。先让你看清「循环」这层，
> act 只是 `action_type + JSON params`，分发是留白——ch03 才把它接实。

## 2. 实现的对象 + 用法

### `Act(action_type, params)`
循环里携带的「一次动作选择」。ch02 只有一个能落地的 action_type：**`finish`**（带 `answer`）。
工具类 act（search / compute …）结构上已经支持，分发器默认返回「未注册」提示——留给 ch03。

```python
from agent import Act
Act(action_type="finish", params={"answer": "你好"})   # 或快捷 Act.finish(answer="你好")
```

### `parse_act(text) → Act`
从模型输出里**抠出** act 对象。约定模型端到端只吐一个 JSON 对象：
`{"thought": "...", "act": {"action_type": "...", "params": {...}}}`。容忍三种情况：
整段干净 JSON、被 ` ```json ``` ` 代码块包裹、夹在叙述文字里（用花括号配平扫描器，跨嵌套找对象）。
解析不出来抛 `ModelError` ——「模型没按协议输出 = 当错误处理」。

### `AgentLoop(model, system=..., max_steps=8, dispatch=...)` + `.run(user) → LoopResult`
本课 hero。构造时只接「模型 + 系统提示 + 步数预算 + 可选分发器」。`run` 内部：

```
context = [system, user]
repeat (最多 max_steps 次)：
    result = await model.complete(context)          # 一次 act 选择
    act = parse_act(result.content)
    if act is finish: return 答案
    observation = dispatch(act)                     # ch02 留白；ch03 接真工具
    context += [assistant act, tool observation]    # 观察回填，进入下一步
```

**参数转身就可演示**（离线 mock）：

```python
import asyncio
from agent import AgentLoop, MockChatModel, ModelResult, Usage, parse_act

async def demo():
    import json
    loop = AgentLoop(MockChatModel(script=[          # 脚本化：先 think 一步，再 finish
        ModelResult(json.dumps({"thought": "...", "act": {"action_type": "think", "params": {}}}),
                    usage=Usage(10, 5)),
        ModelResult(json.dumps({"thought": "...", "act": {"action_type": "finish",
                                                          "params": {"answer": "你好"}}}),
                    usage=Usage(20, 6)),
    ]), max_steps=5, dispatch=lambda act: "（ch03 才接真工具）")
    res = await loop.run("打个招呼")
    print(res.answer, res.steps, res.usage.total_tokens)   # 你好 2 41

asyncio.run(demo())
```

## 3. 本课做到 / 没做到

**做到**：循环（组 context / 调模型 / 解析 act / 观察回填 / 判停）；步数与用量累计；
`max_steps` 预算；纯 JSON act 的 ReAct 概念；`parse_act` 对 fenced / 叙述包裹的容错；
**运行期用户输入注入**（`input_queue` 注入通道 + 每步边界优先处理：最新 user 位 + "请优先处理"标注，
真模型中途插指令会据此重新定向）。
**没做到（留给后续）**：没有工具注册表 / function calling（ch03）；没有 skill / sub-agent
（ch04 / ch07）；act 目前只有 `finish` 真正落地；注入的**阻塞等待确认 / 取消**挂起点（ch17）；
注入与规划联动的**真正重规划**（ch06）。

## 4. 关联八股（~12 题）

1. 为什么「循环」是 agent 的骨架？单次调用 vs 循环的本质区别？
2. ReAct 的 Thought / Act / Observation 三件套各指什么？为什么比「直接答」更稳？
3. 为什么循环检测的是 **act**（action_type + JSON params）而不是仅 tool_call？两者什么关系？
4. 循环的停止条件有哪些？`max_steps` 预算的意义？设太小会发生什么？
5. 观察（observation）为什么要回填进 context？不回填会怎样（模型失忆）？
6. 「模型给非 JSON 输出」算正常失败吗？为什么 parse 失败要当错误处理而不是默默往前？
7. 为什么本课 agent 内部是**顺序的 async**，不显式并行？并发该在哪注入？
8. 用量/步数记账放循环里还是适配器里？为什么循环要累计 usage？
9. 分发器（dispatch）返回 observation 是同步还是异步合适？为什么？（ch03 工具可能是跑命令/查库的 IO 重活）
10. `parse_act` 对 fenced / 叙述包裹的容错，反映工程上对「模型不守格式」的什么态度？
11. 步数耗尽返回「未完成」与「完成」如何区分？上层（ch08 RSI 的 verifier）为什么需要这个 flag？
12. 上下文越滚越大（每次回填 observation）会不会把窗撑爆？——引出 ch05 memory 的上下文预算。
13. agent **不是跑完才理人**：运行期用户注入的实现思路？为什么用每步边界的非阻塞检查而非全程阻塞？
14. 注入为什么"置为最新 user 位 + 显式标注"就算"优先"？ReAct 里消息的权威顺序是怎么规定的？
15. "优先"和"丢弃"的区别？优先处理后旧 context 该不该删？（→ ch06 重规划、ch17 cancel/approve）

## 5. 生产版延伸阅读（复用 + 回填）

`evolving_harness/agent.py`（产品）的循环已是状态机：`State` + adapter（`core/state.py`、
`adapters/state_*.py`）+ checkpoint/resume（M1 T1.6）。对比读：教学版用 `for _ in range(max_steps)`
讲清 ReAct 概念；产品版分「状态建模 + 可中断可恢复 + 钩子」，是把循环"工程化"后的样子。
`models/` 的 `ChatModel` 与本课同名——循环只见「结果 or ModelError」，重试/退避都在适配器里（ch01 已讲）。

## 6. 运行与验证

```bash
# L0 离线（不耗额度）：13 个用例，mock 脚本化
uv run pytest course/ch02_agent_loop/ -q
# L1 冒烟（需 EH_SMOKE=1 + ARK key，真 deepseek 跑一遍循环）
EH_SMOKE=1 uv run pytest course/ch02_agent_loop/ -m smoke -s
uv run ruff check course/
```