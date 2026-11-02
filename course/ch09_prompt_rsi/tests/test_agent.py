"""ch09 prompt-rsi 的 L0 离线自测：全 Mock / Fake，不联网。

分层：async 评分路径 / 模型定位（LLM+SystemOne）/ 合并改进 / 轮级与段级门禁 /
train-val-test 纪律（test 只碰一次、不泄漏 rubric）/ 规则后端不回退 / SystemOne 客户端 / 数据集加载 /
两级门禁端到端。真模型冒烟在 test_smoke.py。
"""

import json

import httpx
import pytest
from agent import MockChatModel, ModelError, ModelResult, Usage
from rsi import (
    AgentConfig,
    Checkpoint,
    Diagnosis,
    Gate,
    Judge,
    LLMJudge,
    LLMLocator,
    MergedLLMImprover,
    PromptPack,
    RolloutResult,
    Rsi,
    RuleJudge,
    RuleLocator,
    RuleProposer,
    RuleScorer,
    SegmentGate,
    SystemOneClient,
    SystemOneJudge,
    SystemOneLocator,
    Task,
    TaskFrame,
    TaskScore,
    TwoStageImprover,
    evaluate,
    load_split,
)


def _result(content="", *, prompt=1, completion=1, tool_calls=None):
    return ModelResult(content, usage=Usage(prompt, completion), tool_calls=tool_calls)


def _pack(guidance="g", fmt="", prefix="", suffix="", exemplars=()):
    return PromptPack(guidance=guidance, final_format=fmt,
                      task_frame=TaskFrame(prefix=prefix, suffix=suffix, exemplars=exemplars))


def _ts(task_id="T1", mean=1.0):
    return TaskScore(task_id=task_id, mean=mean, best=mean, min_=mean)


def _R(ans="", finished=True, tools=(), score=0.0):
    return RolloutResult(task_id="T", answer=ans, finished=finished,
                         tool_calls=tuple(tools), score=score)


# --- 1. async 评分路径：evaluate 必须 await judge；RuleJudge 与 ch08 规则一致 ----------------- #
async def test_evaluate_awaits_rule_judge_and_matches_rule_scorer():
    tasks = [Task("t1", "x", checks=("2", "~3"))]
    cfg = AgentConfig(prompt=_pack())
    model = MockChatModel(script=[_result("结果是 2")])
    scores = await evaluate(tasks, cfg, model, RuleJudge())
    assert scores[0].mean == 1.0
    assert RuleScorer().score(tasks[0], RolloutResult("t1", "结果是 2")) == 1.0


async def test_evaluate_awaits_llm_judge():
    model = MockChatModel(script=[_result("最终答案"), _result("得分 0.8")])
    scores = await evaluate([Task("t1", "x", checks=("要点",))], AgentConfig(prompt=_pack()),
                            model, LLMJudge(model))
    assert scores[0].mean == 0.8                     # 第二次 complete = judge，被 await 并解析出 0.8
    await model.aclose()


# --- 2. LLMLocator：抠 JSON → 组件；解析失败保守 ok ---------------------------------------- #
async def test_llm_locator_parses_json_and_falls_back_to_ok():
    task = Task("t", "x", checks=("要点", "~空谈"))
    model = MockChatModel(script=[_result('{"component": "format", "finding": "含禁词"}')])
    diag = await LLMLocator(model).locate(task, _R(ans="绝不空谈"), 0.5)
    assert diag.component == "format" and "含禁词" in diag.finding

    bad = MockChatModel(script=[_result("我觉得是格式问题（没给 JSON）")])
    diag2 = await LLMLocator(bad).locate(task, _R(ans="随便"), 0.5)
    assert diag2.component == "ok"                    # 解析失败 → 保守不改

    full = MockChatModel(script=[])                   # 满分直接短路，不调用模型
    diag3 = await LLMLocator(full).locate(task, _R(ans="有要点且不空谈"), 1.0)
    assert diag3.component == "ok" and full.script == []


# --- 3. SystemOneLocator：choice → 组件映射 ------------------------------------------------- #
class FakeSystemOneClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return {"failure_component_v1": self.answer}


async def test_systemone_locator_maps_choice_to_component():
    client = FakeSystemOneClient({"type": "choice", "choice": "format", "confidence": 0.9})
    diag = await SystemOneLocator(client).locate(
        Task("t", "x", checks=("要点", "~空谈")), _R(ans="绝不空谈"), 0.5)
    assert diag.component == "format" and "confidence=0.9" in diag.finding
    state = client.calls[0]["state"]
    assert state["task"] == "x" and state["answer"] == "绝不空谈" and state["score"] == 0.5
    q = client.calls[0]["questions"]["failure_component_v1"]
    assert q["type"] == "choice" and "信息不足" in q["criteria"]   # 兜底选项必须在


async def test_systemone_locator_unknown_choice_falls_back_to_ok():
    client = FakeSystemOneClient({"type": "choice", "choice": "信息不足"})
    diag = await SystemOneLocator(client).locate(Task("t", "x", checks=("要点",)), _R(ans="无"), 0.5)
    assert diag.component == "ok"


# --- 4. MergedLLMImprover：一次生成得 (诊断, 新 config)，原 config 不动 ----------------------- #
async def test_merged_improver_single_call_returns_diag_and_new_config():
    payload = {"component": "format", "finding": "输出了禁词",
               "guidance": "只输出结论，不要写推导。", "final_format": "只给数字"}
    model = MockChatModel(script=[_result(json.dumps(payload))])
    cfg = AgentConfig(prompt=_pack(guidance="旧 guidance"))
    diag, cand = await MergedLLMImprover(model).improve(cfg, Task("t", "x"), _R(ans="推导 1"))
    assert diag.component == "format" and "禁词" in diag.finding
    assert cand.prompt.guidance == "只输出结论，不要写推导。"
    assert cand.prompt.final_format == "只给数字"
    assert cand.hash() != cfg.hash() and cfg.prompt.guidance == "旧 guidance"   # 不可变
    assert model.script == []                                  # 只调用一次（locate+propose 合并）


async def test_merged_improver_parse_failure_keeps_config():
    model = MockChatModel(script=[_result("不给你 JSON")])
    cfg = AgentConfig(prompt=_pack())
    diag, cand = await MergedLLMImprover(model).improve(cfg, Task("t", "x"), _R(ans="a"))
    assert diag.component == "ok" and cand is cfg


# --- 5. 轮级门禁：只看 train 的 pairwise 差分 ----------------------------------------------- #
def test_round_gate_promote_reject_and_regress():
    g = Gate()
    base = [_ts("t1", 0.5), _ts("t2", 0.5)]
    assert g.decide(base, [_ts("t1", 0.9), _ts("t2", 0.6)])[0] == "promote"
    assert g.decide(base, [_ts("t1", 0.4), _ts("t2", 0.6)])[0] == "reject"       # t1 回归
    g2 = Gate(regress_tol=0.05)
    assert g2.decide(base, [_ts("t1", 1.0), _ts("t2", 0.1)])[0] == "reject"       # 最差 task 大幅回退


# --- 6. 段级门禁：双方提升才 commit；val 回归则 rollback；无变化 hold ------------------------- #
def _ckpt(train=(0.5, 0.5), val=(0.5, 0.5)):
    return Checkpoint(AgentConfig(prompt=_pack(guidance="wm")),
                      [_ts(f"t{i}", v) for i, v in enumerate(train, 1)],
                      [_ts(f"v{i}", v) for i, v in enumerate(val, 1)], "wm-version")


def test_segment_gate_requires_both_sides_to_improve():
    g = SegmentGate()
    wm = _ckpt()
    ok = g.decide(wm, [_ts("t1", 0.9), _ts("t2", 0.6)], [_ts("v1", 0.9), _ts("v2", 0.6)])
    assert ok[0] == "commit" and ok[1] == pytest.approx(0.25) and ok[2] == pytest.approx(0.25)
    # train 升、val 没升 → 不许 commit
    bad = g.decide(wm, [_ts("t1", 0.9), _ts("t2", 0.6)], [_ts("v1", 0.5), _ts("v2", 0.5)])
    assert bad[0] == "rollback" and bad[2] == pytest.approx(0.0)
    # 什么都没变 → hold
    hold = g.decide(wm, [_ts("t1", 0.5), _ts("t2", 0.5)], [_ts("v1", 0.5), _ts("v2", 0.5)])
    assert hold[0] == "hold"


def test_segment_gate_train_only_gain_but_worst_task_regresses():
    g = SegmentGate(train_regress_tol=0.05)
    wm = _ckpt()
    verdict = g.decide(wm, [_ts("t1", 1.0), _ts("t2", 0.1)], [_ts("v1", 0.8), _ts("v2", 0.8)])
    assert verdict[0] == "rollback"                   # 均值升但最差 task 回退 → 不 commit


# --- 7. 评估纪律：test 只碰一次；test 从不进 train/val ------------------------------------ #
class CountingJudge(Judge):
    def __init__(self):
        self.ids: list[str] = []

    async def score(self, task, r):
        self.ids.append(task.id)
        return RuleScorer().score(task, r)


class StubbornModel(MockChatModel):
    async def complete(self, messages, tools=None):
        return _result("我就这样答")


async def test_test_split_evaluated_exactly_once_and_never_in_train(tmp_path):
    train = [Task("tr1", "题一", checks=("要点",)), Task("tr2", "题二", checks=("要点",))]
    val = [Task("va1", "题三", checks=("要点",))]
    test = [Task("te1", "题四", checks=("要点",)), Task("te2", "题五", checks=("要点",))]
    judge = CountingJudge()
    rsi = Rsi(AgentConfig(prompt=_pack()), StubbornModel(), train=train, val=val, test=test,
              judge=judge, storage_dir=str(tmp_path / "v"), max_rounds=2, val_every=2)
    report = await rsi.run()

    assert 0.0 <= report.test_score <= 1.0
    assert {t.id for t in test} <= set(judge.ids)                       # test 被评过
    test_ids = [i for i in judge.ids if i.startswith("te")]
    assert sorted(test_ids) == ["te1", "te2"]                          # 恰好一次（n_rollouts=1）
    # train 被多次评测（基线 + 每轮候选），val 不在 train 里
    assert [i for i in judge.ids if i.startswith("tr")].count("tr1") >= 2
    assert "va1" not in [i for i in judge.ids if i.startswith("tr")]


# --- 8. 无评测泄漏：rollout 的 user 消息只用 statement，不含私有 checks ----------------------- #
async def test_rollout_prompts_never_leak_checks():
    class Recorder(MockChatModel):
        def __init__(self):
            super().__init__(script=[])
            self.messages: list[list] = []

        async def complete(self, messages, tools=None):
            self.messages.append(list(messages))
            return _result("答案")

    secret = "SECRET-RUBRIC"
    task = Task("t", "1+1", checks=(secret,))
    cfg = AgentConfig(prompt=_pack(guidance="你是助手"))
    assert secret not in cfg.prompt.render_user(task.statement)
    from rsi import rollout
    model = Recorder()
    await rollout(cfg, task, model)
    joined = "\n".join(m.content for m in model.messages[0])
    assert secret not in joined and "1+1" in joined


# --- 9. 规则后端不回退：RuleJudge + RuleLocator + RuleProposer 复现 ch08 关键用例 ------------- #
async def test_rule_backends_reproduce_ch08_cases():
    task = Task("t", "x", checks=("要点", "~空谈"))
    assert await RuleJudge().score(task, _R(ans="有要点，没有废话")) == 1.0
    assert await RuleJudge().score(task, _R(ans="有要点而且空谈")) == pytest.approx(0.5)

    loc = RuleLocator()
    assert (await loc.locate(task, _R(ans=""), 0.0)).component == "skip"
    assert (await loc.locate(task, _R(ans="结论：绝不空谈", tools=("read",)), 0.5)).component == "format"
    assert (await loc.locate(task, _R(ans="给个结论", tools=()), 0.5)).component == "halluc"
    assert (await loc.locate(task, _R(ans="结论在此", tools=("read",)), 0.5)).component == "frame"
    assert (await loc.locate(task, _R(ans="有要点且不空谈"), 1.0)).component == "ok"

    cfg = AgentConfig(prompt=_pack(guidance="g"))
    fmt = await RuleProposer().propose(cfg, task, _R(ans="含禁词"), Diagnosis("format", "f"))
    assert fmt.prompt.final_format and fmt.hash() != cfg.hash() and cfg.prompt.final_format == ""
    hal = await RuleProposer().propose(cfg, task, _R(ans="a"), Diagnosis("halluc", "h"))
    assert "调用工具" in hal.prompt.guidance
    ok = await RuleProposer().propose(cfg, task, _R(ans="a"), Diagnosis("ok", "ok"))
    assert ok is cfg                                 # 无有效归因 → 不改


async def test_two_stage_improver_locates_then_proposes():
    cfg = AgentConfig(prompt=_pack(guidance="g"))
    diag, cand = await TwoStageImprover(RuleLocator(), RuleProposer()).improve(
        cfg, Task("t", "x", checks=("要点",)), _R(ans="给个结论", tools=(), score=0.0))
    assert diag.component == "halluc" and "调用工具" in cand.prompt.guidance


# --- 10. SystemOneClient：头/体/解析/重试 --------------------------------------------------- #
async def test_systemone_client_sets_beta_header_body_and_retries(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={
            "model": "m",
            "answers": {"q1": {"type": "choice", "choice": "format", "confidence": 0.9}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("rsi.asyncio.sleep", _no_sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SystemOneClient(model="m", api_key="k", http_client=http, max_retries=1)
    answers = await client.ask({"task": "t", "answer": "a"},
                               {"q1": {"type": "choice", "instructions": "选组件", "criteria": {"format": "..."}}})
    assert answers["q1"]["choice"] == "format"

    assert len(calls) == 2                                          # 503 后重试一次
    req = calls[-1]
    assert str(req.url).endswith("/api/v1/systemone")
    assert req.headers["ark-beta-systemone-api"] == "true"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["model"] == "m" and body["state"]["task"] == "t" and "q1" in body["questions"]
    await http.aclose()


async def test_systemone_client_non_retryable_raises_model_error():
    def handler(_request):
        return httpx.Response(400, text="bad request")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SystemOneClient(model="m", http_client=http, max_retries=2)
    with pytest.raises(ModelError):
        await client.ask({"x": 1}, {"q": {"type": "noul", "instructions": "?"}})
    await http.aclose()


async def test_systemone_judge_normalizes_score_response():
    class Fake:
        async def ask(self, state, questions):
            return {"rubric_score_v1": {"type": "score", "score": 2.0, "confidence": 0.9}}

    assert await SystemOneJudge(Fake()).score(Task("t", "x", checks=("a",)), _R(ans="a")) == 1.0


# --- 11. load_split：读仓库根 dataset/rsi/*.jsonl ------------------------------------------- #
def test_load_split_reads_dataset_splits():
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    assert train and val and test
    assert all(isinstance(t, Task) and t.statement and isinstance(t.checks, tuple)
               for t in train + val + test)
    ids = {t.id for t in train} | {t.id for t in val} | {t.id for t in test}
    assert len(ids) == len(train) + len(val) + len(test)          # 拆分不相交


# --- 12. 两级门禁端到端：train+val 都升 → commit；val 不动 → rollback ------------------------ #
class FormatMock(MockChatModel):
    """答案依赖 system 是否带「输出要求」；可选让部分任务永远 stubborn（模拟 val 没泛化）。"""

    def __init__(self, stubborn_marker: str = ""):
        super().__init__(script=[])
        self.stubborn_marker = stubborn_marker

    async def complete(self, messages, tools=None):
        sys_text = next((m.content for m in messages if m.role == "system"), "")
        user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if self.stubborn_marker and self.stubborn_marker in user_text:
            return _result("我们就空谈吧")
        return _result("结论" if "输出要求" in sys_text else "我们就空谈吧")


async def test_run_commits_when_train_and_val_both_improve(tmp_path):
    tasks = [Task("t1", "给个结论", checks=("~空谈",))]
    val = [Task("v1", "给个结论", checks=("~空谈",))]
    test = [Task("s1", "给个结论", checks=("~空谈",))]
    cfg = AgentConfig(name="base", prompt=_pack(guidance="你是助手"), max_steps=3)
    rsi = Rsi(cfg, FormatMock(), train=tasks, val=val, test=test,
              storage_dir=str(tmp_path / "v"), max_rounds=2, val_every=2)
    report = await rsi.run()

    assert report.baseline_train_score == 0.0 and report.baseline_val_score == 0.0
    assert report.rounds[0].verdict == "promote" and report.rounds[0].score == 1.0
    assert report.segments and report.segments[-1].verdict == "commit"
    assert report.segments[-1].train_delta == pytest.approx(1.0)
    assert report.segments[-1].val_delta == pytest.approx(1.0)
    assert report.final_config.prompt.final_format
    assert report.test_score == 1.0                   # test 泛化也升
    entries = (tmp_path / "v" / "versions.jsonl").read_text(encoding="utf-8").splitlines()
    verdicts = [json.loads(e)["verdict"] for e in entries]
    assert verdicts[0] == "accepted" and "commit" in verdicts


async def test_run_rolls_back_to_watermark_when_val_does_not_transfer(tmp_path):
    train = [Task("tr", "TRAIN 给个结论", checks=("~空谈",))]
    val = [Task("va", "VAL 给个结论", checks=("~空谈",))]
    test = [Task("te", "TEST 给个结论", checks=("~空谈",))]
    cfg = AgentConfig(name="base", prompt=_pack(guidance="你是助手"), max_steps=3)
    rsi = Rsi(cfg, FormatMock(stubborn_marker="VAL"),
              train=train, val=val, test=test, storage_dir=str(tmp_path / "v"),
              max_rounds=2, val_every=2)
    report = await rsi.run()

    assert report.rounds[0].verdict == "promote"       # train 升了 → 轮级先 promote
    assert report.rounds[0].score == 1.0
    assert report.segments[-1].verdict == "rollback"   # val 没升 → 段级回滚
    assert report.final_config.hash() == cfg.hash()    # 最终回到血线 config
    assert report.test_score == 0.0                    # test 用回滚后的基线评
