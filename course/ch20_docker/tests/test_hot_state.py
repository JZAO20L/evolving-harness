"""L0：MemoryKV 语义、RunState 队列/锁/频控，以及接入 HTTP 后的缓存与幂等。"""

import asyncio
import time
from pathlib import Path

from backend import (
    Job,
    JobManager,
    MemoryKV,
    RunResponse,
    RunService,
    RunState,
    UsageOut,
    create_app,
)
from fastapi.testclient import TestClient

from evolving_harness.models.base import ModelResponse
from evolving_harness.models.mock import MockChatModel


def _factory(*responses: ModelResponse):
    def build(_state):
        return MockChatModel(list(responses))

    return build


def _app(make_agent, tmp_path, db, hot, *, factory=None, rate_limit=0):
    return create_app(
        service=RunService(
            agent_path=make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=factory,
        ),
        max_concurrency=1,
        db_path=db,
        hot_state=hot,
        rate_limit=rate_limit,
    )


def _wait(client, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish: {body}")


async def test_memory_kv_string_hash_list_and_ttl():
    now = [100.0]
    kv = MemoryKV(clock=lambda: now[0])

    await kv.set("s", "1", ex=5)
    assert await kv.get("s") == "1"
    now[0] = 106
    assert await kv.get("s") is None

    assert await kv.set_nx("lock", "1", ex=10) is True
    assert await kv.set_nx("lock", "1", ex=10) is False

    assert await kv.incr("counter") == 1
    assert await kv.incr("counter") == 2
    await kv.expire("counter", 5)
    now[0] = 112
    assert await kv.incr("counter") == 1

    await kv.hset("h", {"a": "1"})
    await kv.hset("h", {"b": "2"})
    assert await kv.hgetall("h") == {"a": "1", "b": "2"}

    await kv.lpush("q", "first")
    await kv.lpush("q", "second")
    assert await kv.llen("q") == 2
    assert await kv.rpop("q") == "first"
    assert await kv.rpop("q") == "second"
    assert await kv.rpop("q") is None


async def test_run_state_job_roundtrip_and_ttl():
    now = [0.0]
    state = RunState(MemoryKV(clock=lambda: now[0]), job_ttl=10)
    job = Job(run_id="r-1", task="查", workspace=Path("/tmp/ws/r-1"))
    await state.put_job(job)

    got = await state.get_job("r-1")
    assert got is not None
    assert got.task == "查"
    assert got.status == "queued"

    job.status = "succeeded"
    job.result = RunResponse(
        run_id="r-1",
        final="done",
        status="ok",
        steps=1,
        usage=UsageOut(),
        state_version=1,
        state_hash="h" * 64,
        trace_path="/tmp/traces/r-1.jsonl",
        workspace="/tmp/ws/r-1",
    )
    await state.put_job(job)
    got = await state.get_job("r-1")
    assert got is not None and got.result is not None
    assert got.result.final == "done"

    now[0] = 11
    assert await state.get_job("r-1") is None


async def test_rate_limit_is_fixed_window_and_resets():
    now = [5.0]
    state = RunState(MemoryKV(clock=lambda: now[0]), clock=lambda: now[0])

    allowed, remaining, retry_after = await state.rate_limit("client", limit=2, window=10)
    assert (allowed, remaining) == (True, 1)
    allowed, remaining, _ = await state.rate_limit("client", limit=2, window=10)
    assert (allowed, remaining) == (True, 0)
    allowed, remaining, retry_after = await state.rate_limit("client", limit=2, window=10)
    assert allowed is False and remaining == 0 and retry_after == 5

    now[0] = 15.0
    allowed, remaining, _ = await state.rate_limit("client", limit=2, window=10)
    assert (allowed, remaining) == (True, 1)


async def test_queue_is_fifo_and_shared_between_state_handles():
    kv = MemoryKV()
    producer = RunState(kv)
    consumer = RunState(kv)

    await producer.enqueue("a")
    await producer.enqueue("b")
    assert await consumer.queue_len() == 2
    assert await consumer.dequeue() == "a"
    assert await consumer.dequeue() == "b"
    assert await consumer.dequeue() is None


async def test_cancel_queued_job_from_another_instance():
    kv = MemoryKV()
    state = RunState(kv)
    await state.put_job(Job(run_id="remote", task="x", workspace=Path("/tmp/x")))
    await state.enqueue("remote")

    manager = JobManager(RunService(), state=state)
    assert await manager.cancel("remote") is True
    view = await state.get_view("remote")
    assert view is not None
    assert view.status == "cancelled"
    assert view.error == "cancelled before start"
    assert await manager.cancel("remote") is False


def test_http_cache_falls_back_to_db_after_restart(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    hot1 = RunState(MemoryKV())
    factory = _factory(ModelResponse(content="缓存答案"))
    with TestClient(_app(make_agent, tmp_path, db, hot1, factory=factory)) as client:
        assert (
            client.post("/v1/runs", json={"task": "缓存", "run_id": "cache-1"}).status_code
            == 202
        )
        assert _wait(client, "cache-1")["status"] == "succeeded"
        cached = client.get("/v1/runs/cache-1").json()
        assert cached["result"]["final"] == "缓存答案"

    hot2 = RunState(MemoryKV())
    with TestClient(_app(make_agent, tmp_path, db, hot2, factory=factory)) as client:
        body = client.get("/v1/runs/cache-1").json()
        assert body["status"] == "succeeded"
        assert body["result"]["final"] == "缓存答案"

    backfilled = asyncio.run(hot2.get_view("cache-1"))
    assert backfilled is not None and backfilled.status == "succeeded"


def test_duplicate_run_id_is_served_from_shared_state(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    hot = RunState(MemoryKV())
    calls: list[int] = []

    def factory(_state):
        calls.append(1)
        return MockChatModel([ModelResponse(content="只跑一次")])

    with TestClient(_app(make_agent, tmp_path, db, hot, factory=factory)) as client:
        client.post("/v1/runs", json={"task": "dup", "run_id": "dup-1"})
        assert _wait(client, "dup-1")["status"] == "succeeded"

    with TestClient(_app(make_agent, tmp_path, db, hot, factory=factory)) as client:
        again = client.post("/v1/runs", json={"task": "dup", "run_id": "dup-1"})
        assert again.status_code == 202, again.text
        assert again.json()["status"] == "succeeded"
        assert again.json()["result"]["final"] == "只跑一次"

    assert len(calls) == 1


def test_worker_hydrates_job_from_shared_state(make_agent, tmp_path):
    hot = RunState(MemoryKV())

    async def seed() -> None:
        await hot.put_job(
            Job(run_id="hyd-1", task="从状态水合", workspace=tmp_path / "ws/hyd-1")
        )
        await hot.enqueue("hyd-1")

    asyncio.run(seed())
    factory = _factory(ModelResponse(content="水合成功"))
    with TestClient(_app(make_agent, tmp_path, tmp_path / "h.db", hot, factory=factory)) as client:
        body = _wait(client, "hyd-1")
        assert body["status"] == "succeeded"
        assert body["result"]["final"] == "水合成功"


def test_http_rate_limit_rejects_with_retry_after(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    hot = RunState(MemoryKV())
    factory = _factory(ModelResponse(content="ok"))
    app = _app(make_agent, tmp_path, db, hot, factory=factory, rate_limit=2)

    with TestClient(app) as client:
        assert client.post("/v1/runs", json={"task": "1", "run_id": "rl-1"}).status_code == 202
        assert client.post("/v1/runs", json={"task": "2", "run_id": "rl-2"}).status_code == 202
        blocked = client.post("/v1/runs", json={"task": "3", "run_id": "rl-3"})
        assert blocked.status_code == 429
        assert blocked.headers["retry-after"].isdigit()
        assert blocked.headers["x-ratelimit-remaining"] == "0"

        _wait(client, "rl-1")
        _wait(client, "rl-2")
        history = client.get("/v1/runs", params={"limit": 10}).json()
        assert not any(item["run_id"] == "rl-3" for item in history)
