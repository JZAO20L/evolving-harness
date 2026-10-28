"""L1 冒烟：真 deepseek-V4 @ ARK 跑 ch06 的「长期 Plan + TodoList + 改方向重规划」闭环。

两条真冒烟：
1. **首次分解 + 单循环执行 + 监控**：给一个多阶段中文目标，真模型先 `plan_task`+`decompose` 拆出
   长期 Plan 与子任务 Todo，再在单循环里用 `todo_update` 推进进度并落地，最终给结论；`plan.jsonl`
   落盘（跨 run 恢复）。
2. **改方向重规划**：对同一 tracker（新实例从 `plan.jsonl` 恢复）抛「去掉某步、先做另一步」——
   命中 `is_plan_update` 门控走 `revise_plan`：长期 Plan 步骤被调整、剩余 Todo 被重排，再继续执行。

离线默认跳过：需 EH_SMOKE=1 + ARK key。模型分解/执行请求真花额度。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import PlanAgent, is_plan_update

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    for key in ("ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


def _skip_unless_smoke() -> None:
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    if not (os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")):
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")


def _make_model():
    from agent import OpenAIChatModel
    return OpenAIChatModel(
        base_url=os.getenv("EH_BASE_URL", ARK_BASE),
        model=os.getenv("EH_MODEL", DEEPSEEK),
        api_key=os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY") or "",
        timeout=120.0,
        max_retries=2,
    )


@pytest.mark.smoke
async def test_smoke_plan_then_execute_with_todo_update(tmp_path):
    _skip_unless_smoke()
    d = str(tmp_path / "p")
    model = _make_model()
    try:
        ag = PlanAgent(model, storage_dir=d, max_steps=12)
        res = await ag.run(
            "请帮我完成一个多阶段任务：调研『RL 红队』并整理，分三步："
            "①给出核心结论；②罗列 3 篇相关论文；③写一段中文摘要。"
            "请先用 todo_update 把每一步子任务标记为 in_progress 再逐个完成、再标记 done，最后给出结论。"
        )
    finally:
        await model.aclose()

    assert res.answer.strip(), "模型应产出最终结论"
    assert ag.tracker.has_plan(), "应已拆出长期 Plan"
    assert len(ag.tracker.todo.items) >= 2, "应拆出多个子任务"
    assert any(t.name == "todo_update" for t in res.tool_calls), "模型应用 todo_update 推进进度"
    assert (tmp_path / "p" / "plan.jsonl").exists(), "plan.jsonl 应落盘（跨 run 恢复）"
    print(f"\n[smoke-1] goal={ag.tracker.plan.goal!r} steps={len(ag.tracker.plan.steps)} "
          f"todo={len(ag.tracker.todo.items)} progress={ag.progress()} "
          f"tools={[t.name for t in res.tool_calls]} -> {res.answer[:60]!r}\n")


@pytest.mark.smoke
async def test_smoke_revision_changes_plan(tmp_path):
    _skip_unless_smoke()
    d = str(tmp_path / "p")
    model = _make_model()
    try:
        ag = PlanAgent(model, storage_dir=d, max_steps=10)
        res1 = await ag.run("帮我做『RL 红队』调研，分两步：①核心结论；②整理论文。"
                            "请用 todo_update 逐步推进，最后给出结论。")
        before = " | ".join(f"{s.id}:{s.statement}" for s in ag.tracker.plan.steps)
        assert ag.tracker.has_plan() and any(t.name == "todo_update" for t in res1.tool_calls)

        # run2：新实例从 plan.jsonl 恢复，抛改方向意图 -> 走 revise_plan
        model2 = _make_model()
        try:
            ag2 = PlanAgent(model2, storage_dir=d, max_steps=10)
            assert is_plan_update("去掉整理论文那步，先做中文摘要"), "改方向文本应命中闸门"
            res2 = await ag2.run("改一下方向：去掉整理论文那一步，先做『补一段中文摘要』，再给结论。")
        finally:
            await model2.aclose()

        assert res2.answer.strip()
        after = " | ".join(f"{s.id}:{s.statement}" for s in ag2.tracker.plan.steps)
        assert after != before, "长期 Plan 应被重规划调整"
        print(f"\n[smoke-2] before={before}\n          after ={after}\n"
              f"          progress={ag2.progress()} -> {res2.answer[:60]!r}\n")
    finally:
        await model.aclose()