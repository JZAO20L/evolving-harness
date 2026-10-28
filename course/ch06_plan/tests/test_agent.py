from agent import (
    MockChatModel,
    ModelResult,
    Plan,
    PlanAgent,
    PlanStep,
    PlanTracker,
    Todo,
    TodoList,
    ToolCall,
    ToolRegistry,
    Usage,
    _extract_json,
    decompose,
    is_plan_update,
    plan_task,
    revise_plan,
)


def _result(content="", *, tool_calls=None, prompt=1, completion=1):
    return ModelResult(content, usage=Usage(prompt, completion), tool_calls=tool_calls)


class _PlanModel(MockChatModel):
    """规划/执行双脚本：tools=None -> 规划调用(plan_task/decompose/revise_plan)；否则主循环。"""

    def __init__(self, plan_script, loop_script):
        super().__init__(script=list(loop_script))
        self.plan_script = list(plan_script)
        self.calls: list[tuple[list, bool]] = []     # (messages, tools_is_none)

    async def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools is None))
        if tools is None:
            return self.plan_script.pop(0)
        return self.script.pop(0)


# --- Todo / Plan 基本结构与进度监控 ---------------------------------------------- #
def test_todolist_mark_progress_inject():
    tl = TodoList([Todo("T1", "给结论", "done", "P1"),
                   Todo("T2", "列论文", "pending", "P2"),
                   Todo("T3", "写摘要", "in_progress", "P3")])
    assert tl.done() == [tl.get("T1")]
    assert len(tl.pending()) == 2                        # T2 + T3（in_progress 也算未完成）
    assert tl.progress() == "1/3（33%）"
    assert tl.mark("T2", "done") and tl.get("T2").status == "done"
    assert tl.progress() == "2/3（67%）"
    table = tl.inject_table()
    assert "T1" in table and "T2" in table and "T3" in table and "67%" in table   # 进度表供 system 注入
    assert not tl.mark("T99", "done")                    # 未知 id 返回 False


def test_plan_step_done():
    p = Plan("红队目标", [PlanStep("P1", "结论"), PlanStep("P2", "论文")])
    assert p.step_done("P2") and p.by_id("P2").status == "done"
    assert not p.step_done("P9")


# --- 规则容错解析（模型 JSON 夹前后文 / 代码块） ----------------------------------- #
def test_extract_json_tolerant():
    assert _extract_json('{"a":1}') == {"a": 1}
    assert _extract_json('前缀 {"x":[1,2]} 后缀')["x"] == [1, 2]     # 取第一个合法 JSON
    assert _extract_json('```json\n{"goal":"g","steps":[]}\n```')["goal"] == "g"
    assert _extract_json("这里没有 json") is None


# --- 意图识别（规则 cue 门控） ----------------------------------------------------- #
def test_is_plan_update_cues():
    assert is_plan_update("B 不要做了，先做 C")
    assert is_plan_update("调整一下顺序，先做第②步")
    assert not is_plan_update("1 加 2 等于几？")
    assert not is_plan_update("帮我总结一段话")


# --- 模型分解：plan_task / decompose / revise_plan -------------------------------- #
async def test_plan_task_builds_long_plan():
    model = _PlanModel([_result('{"goal":"搞定红队","steps":['
                                '{"id":"P1","statement":"调研"},{"id":"P2","statement":"整理"}]}')], [])
    plan = await plan_task(model, "帮我做红队调研")
    assert plan.goal == "搞定红队" and len(plan.steps) == 2 and plan.steps[1].id == "P2"


async def test_decompose_assigns_ref_step():
    plan = Plan("g", [PlanStep("P1", "a"), PlanStep("P2", "b")])
    model = _PlanModel([_result('{"todos":[{"id":"T1","ref_step":"P1","summary":"读文件"},'
                                '{"id":"T2","ref_step":"P2","summary":"写总结"}]}')], [])
    tl = await decompose(model, plan)
    assert len(tl.items) == 2 and tl.get("T1").ref_step == "P1"


async def test_revise_plan_regenerates_remaining_only():
    old = Plan("g", [PlanStep("P1", "a"), PlanStep("P2", "b"), PlanStep("P3", "c")])
    done = [Todo("T1", "read", "done", "P1")]
    model = _PlanModel([_result('{"goal":"新目标","steps":[{"id":"P1","statement":"a"},'
                                '{"id":"P3","statement":"c改"}],"todos":'
                                '[{"id":"T3","ref_step":"P3","summary":"做c"}]}')], [])
    np_, nt = await revise_plan(model, old, done, "去掉 P2，先做 P3")
    assert np_.goal == "新目标"
    assert [s.id for s in np_.steps] == ["P1", "P3"]          # P2 被删、步骤重排（调整长期 Plan）
    assert len(nt.items) == 1 and nt.get("T3").ref_step == "P3"  # 只重排剩余，done 不重复


# --- PlanTracker：JSONL 落盘 + 跨 run 恢复 ------------------------------------------ #
def test_tracker_save_load_roundtrip(tmp_path):
    p = Plan("g", [PlanStep("P1", "a", "done"), PlanStep("P2", "b", "pending")])
    tl = TodoList([Todo("T1", "a", "done", "P1"), Todo("T2", "b", "pending", "P2")])
    PlanTracker(p, tl, storage_dir=str(tmp_path / "p")).save()
    tr = PlanTracker.load(str(tmp_path / "p"))
    assert tr.plan.goal == "g" and len(tr.plan.steps) == 2
    assert tr.plan.by_id("P1").status == "done"              # step 状态也恢复
    assert tr.todo.get("T1").status == "done" and tr.todo.get("T2").status == "pending"
    assert tr.progress() == "1/2（50%）"


def test_tracker_load_missing_is_empty(tmp_path):
    tr = PlanTracker.load(str(tmp_path / "nope"))
    assert not tr.has_plan() and tr.progress() == "（暂无长期计划）"


# --- 组合 A：首次分解 + 单循环执行 + 持久化 ------------------------------------------- #
async def test_planagent_first_run_builds_executes_persists(tmp_path):
    model = _PlanModel(
        plan_script=[_result('{"goal":"红队","steps":[{"id":"P1","statement":"调研"}]}'),
                     _result('{"todos":[{"id":"T1","ref_step":"P1","summary":"读 arxiv"}]}')],
        loop_script=[_result("", tool_calls=[ToolCall("c1", "todo_update", {"id": "T1", "status": "done"})]),
                     _result("全部完成。")],
    )
    ag = PlanAgent(model, storage_dir=str(tmp_path / "p"), max_steps=5)
    res = await ag.run("帮我调研红队")
    assert res.answer == "全部完成。" and res.finished
    assert ag.progress() == "1/1（100%）"                     # 进度被 todo_update 驱动到满
    assert [t.name for t in res.tool_calls] == ["todo_update"]
    # 持久化：另起实例从同一目录恢复，plan+todo（含 done 状态）都在
    ag2 = PlanAgent(_PlanModel([], []), storage_dir=str(tmp_path / "p"))
    assert ag2.tracker.has_plan() and ag2.tracker.todo.get("T1").status == "done"
    assert (tmp_path / "p" / "plan.jsonl").exists()


# --- 组合 B：两次 run 跨实例恢复 + 改方向触发重规划 ------------------------------------- #
async def test_planagent_revision_across_runs(tmp_path):
    d = str(tmp_path / "p")
    ag1 = PlanAgent(_PlanModel(
        plan_script=[_result('{"goal":"红队","steps":[{"id":"P1","statement":"结论"},'
                             '{"id":"P2","statement":"论文"}]}'),
                     _result('{"todos":[{"id":"T1","ref_step":"P1","summary":"给结论"},'
                             '{"id":"T2","ref_step":"P2","summary":"给论文"}]}')],
        loop_script=[_result("", tool_calls=[ToolCall("c1", "todo_update", {"id": "T1", "status": "done"})]),
                     _result("第一步完成。")],
    ), storage_dir=d, max_steps=5)
    await ag1.run("帮我做红队调研")
    assert ag1.tracker.todo.get("T1").status == "done" and ag1.tracker.todo.get("T2").status == "pending"

    # run2：新实例从 plan.jsonl 恢复，用户改方向 -> 走 revise_plan，丢被删步骤的 todo
    ag2 = PlanAgent(_PlanModel(
        plan_script=[_result('{"goal":"红队(改)","steps":[{"id":"P1","statement":"结论"},'
                             '{"id":"P3","statement":"摘要"}],"todos":'
                             '[{"id":"T3","ref_step":"P3","summary":"写摘要"}]}')],
        loop_script=[_result("", tool_calls=[ToolCall("c3", "todo_update", {"id": "T3", "status": "done"})]),
                     _result("改完并完成摘要。")],
    ), storage_dir=d, max_steps=5)
    res2 = await ag2.run("去掉论文 P2，先做摘要")
    assert [s.id for s in ag2.tracker.plan.steps] == ["P1", "P3"]     # 长期 Plan 已调整（P2 删除、改序）
    assert [t.id for t in ag2.tracker.todo.items] == ["T3"]           # 后续 Todo 只剩剩余
    assert res2.answer == "改完并完成摘要。"
    assert ag2.progress() == "1/1（100%）"


# --- 门控护栏：无 Plan 不误判重规划；未知 todo 错误当数据 ---------------------------------- #
async def test_planagent_no_plan_never_revises(tmp_path):
    model = _PlanModel(
        plan_script=[_result('{"goal":"g","steps":[{"id":"P1","statement":"a"}]}'),
                     _result('{"todos":[{"id":"T1","ref_step":"P1","summary":"a"}]}')],
        loop_script=[_result(""), ],
    )
    ag = PlanAgent(model, storage_dir=str(tmp_path / "p"), max_steps=5)
    # 文本虽含更新 cue（"先做"），但无 Plan -> 只走首次分解，不误走重规划
    await ag.run("先做红队调研吧")
    assert ag.tracker.has_plan() and [s.id for s in ag.tracker.plan.steps] == ["P1"]


def test_todo_update_unknown_and_bad_status_is_error_as_data():
    reg = ToolRegistry()
    PlanAgent(_PlanModel([], []), registry=reg)            # 注册 todo_update 到 reg
    upd = reg.get("todo_update")
    assert "没有叫 'NOPE'" in upd.handler({"id": "NOPE", "status": "done"})
    assert "必须是" in upd.handler({"id": "NOPE", "status": "bogus"})