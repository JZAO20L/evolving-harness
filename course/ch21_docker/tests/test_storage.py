"""L0 离线：runs / run_events / agent_versions / checkpoints 的落库与重启后读取。"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import pytest
from backend import RunService, create_app
from fastapi.testclient import TestClient

from evolving_harness.core.message import Message
from evolving_harness.models.base import ChatModel, ModelError, ModelResponse
from evolving_harness.models.mock import MockChatModel


class BoomModel(ChatModel):
    async def complete(self, messages: list[Message], tools=None) -> ModelResponse:
        raise ModelError("db-test exploded")


class HangingModel(ChatModel):
    async def complete(self, messages: list[Message], tools=None) -> ModelResponse:
        await asyncio.sleep(3600)
        return ModelResponse(content="never")


def _factory(*responses: ModelResponse):
    def build(_state):
        return MockChatModel(list(responses))

    return build


def _app(make_agent, tmp_path, db_path, *, factory=None, agent_path=None):
    return create_app(
        service=RunService(
            agent_path=agent_path or make_agent(),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
            model_factory=factory,
        ),
        max_concurrency=1,
        db_path=db_path,
    )


def _wait(client, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        response = client.get(f"/v1/runs/{run_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {run_id} did not finish: {body}")


def _wait_db(db_path: Path, run_id: str, timeout: float = 5.0) -> dict:
    """等异步落库到终态；内存里的 job 可能先于 DB 写入变终态。"""
    deadline = time.time() + timeout
    row: dict = {}
    while time.time() < deadline:
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                found = conn.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if found is not None:
                    row = dict(found)
                    if row["status"] in ("succeeded", "failed", "cancelled"):
                        return row
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} was not persisted: {row}")


def _wait_checkpoint(client, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/runs/{run_id}/checkpoint")
        if response.status_code == 200:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"checkpoint {run_id} was not persisted")


def _parse_sse(text: str) -> list[dict]:
    events: list[dict] = []
    event = None
    event_id = None
    data: list[str] = []
    for line in text.splitlines() + [""]:
        if line == "":
            if event is not None:
                events.append(
                    {
                        "event": event,
                        "id": event_id,
                        "data": json.loads("\n".join(data)) if data else None,
                    }
                )
            event, event_id, data = None, None, []
        elif line.startswith("event: "):
            event = line[7:]
        elif line.startswith("id: "):
            event_id = int(line[4:])
        elif line.startswith("data: "):
            data.append(line[6:])
    return events


def test_run_persists_and_survives_restart(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    app1 = _app(make_agent, tmp_path, db, factory=_factory(ModelResponse(content="落库答案")))

    with TestClient(app1) as client:
        assert client.post("/v1/runs", json={"task": "持久化", "run_id": "db-1"}).status_code == 202
        body = _wait(client, "db-1")
        assert body["status"] == "succeeded"
        checkpoint = _wait_checkpoint(client, "db-1")

    assert checkpoint["state"]["run_id"] == "db-1"
    assert checkpoint["state"]["task"] == "持久化"

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM runs WHERE run_id = 'db-1'").fetchone()
        assert run["status"] == "succeeded"
        assert run["agent_version"] == 1
        assert len(run["agent_hash"]) == 64
        assert json.loads(run["result_json"])["final"] == "落库答案"
        events = conn.execute(
            "SELECT kind, data_json FROM run_events WHERE run_id = 'db-1' ORDER BY seq"
        ).fetchall()
        assert events[0]["kind"] == "run.start"
        assert events[-1]["kind"] == "run.end"
        assert any(row["kind"] == "model.response" for row in events)
        version = conn.execute(
            "SELECT * FROM agent_versions WHERE agent_hash = ?", (run["agent_hash"],)
        ).fetchone()
        assert version is not None and version["version"] == 1

    # 模拟重启：新 app、空 JobManager，只共享同一个 DB。
    app2 = _app(make_agent, tmp_path, db)
    with TestClient(app2) as client:
        body = client.get("/v1/runs/db-1").json()
        assert body["status"] == "succeeded"
        assert body["result"]["final"] == "落库答案"

        history = client.get("/v1/runs", params={"limit": 10}).json()
        assert any(item["run_id"] == "db-1" for item in history)

        versions = client.get("/v1/agent/versions").json()
        assert any(item["agent_hash"] == run["agent_hash"] for item in versions)

        sse = client.get("/v1/runs/db-1/events")
        assert sse.headers["content-type"].startswith("text/event-stream")
        replay = _parse_sse(sse.text)
        assert replay[0]["event"] == "open"
        assert replay[0]["data"]["source"] == "db"
        kinds = [item["event"] for item in replay]
        assert "run.start" in kinds
        assert "run.end" in kinds
        assert kinds[-1] == "job.end"
        assert replay[-1]["data"]["status"] == "succeeded"

        assert client.get("/v1/runs/missing/events").status_code == 404
        assert client.get("/v1/runs/missing/checkpoint").status_code == 404


def test_history_is_newest_first_and_paginates(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    app = _app(make_agent, tmp_path, db, factory=_factory(ModelResponse(content="ok")))

    with TestClient(app) as client:
        for index in (1, 2, 3):
            run_id = f"h-{index}"
            client.post("/v1/runs", json={"task": run_id, "run_id": run_id})
            assert _wait(client, run_id)["status"] == "succeeded"
            _wait_checkpoint(client, run_id)

        first = client.get("/v1/runs", params={"limit": 2}).json()
        assert [item["run_id"] for item in first] == ["h-3", "h-2"]

        second = client.get("/v1/runs", params={"limit": 2, "offset": 2}).json()
        assert [item["run_id"] for item in second] == ["h-1"]

        capped = client.get("/v1/runs", params={"limit": 10_000}).json()
        assert len(capped) == 3


def test_failed_run_still_indexes_trace(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    app = _app(make_agent, tmp_path, db, factory=lambda _state: BoomModel())

    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "炸", "run_id": "bad-db"})
        body = _wait(client, "bad-db")
        assert body["status"] == "failed"
        assert "ModelError" in body["error"]

    run = _wait_db(db, "bad-db")
    assert run["status"] == "failed"
    assert "ModelError" in run["error"]
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        events = conn.execute(
            "SELECT kind, data_json FROM run_events WHERE run_id = 'bad-db' ORDER BY seq"
        ).fetchall()
        assert events
        end = next(row for row in events if row["kind"] == "run.end")
        assert json.loads(end["data_json"])["status"] == "error"


def test_cancelled_queued_job_is_persisted(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    app = _app(make_agent, tmp_path, db, factory=lambda _state: HangingModel())

    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "占住", "run_id": "busy-db"})
        client.post("/v1/runs", json={"task": "排队", "run_id": "queued-db"})

        with client.websocket_connect("/v1/runs/queued-db/ws") as ws:
            ws.send_json({"type": "cancel"})
            frames = _recv_until(ws, lambda f: f["type"] == "job.end")
        assert frames[-1]["job"]["status"] == "cancelled"

        with client.websocket_connect("/v1/runs/busy-db/ws") as ws:
            _recv_until(
                ws,
                lambda f: f["type"] == "event" and f["event"] == "model.request",
            )
            ws.send_json({"type": "cancel"})
            _recv_until(ws, lambda f: f["type"] == "job.end")

    queued = _wait_db(db, "queued-db")
    assert queued["status"] == "cancelled"
    assert "cancelled before start" in queued["error"]
    busy = _wait_db(db, "busy-db")
    assert busy["status"] == "cancelled"


def test_bad_yaml_is_recorded_as_failed(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: [not, an, int]\n", encoding="utf-8")
    app = _app(make_agent, tmp_path, db, agent_path=bad)

    with TestClient(app) as client:
        assert client.post("/v1/runs", json={"task": "坏配置", "run_id": "bad-yaml"}).status_code == 202
        body = _wait(client, "bad-yaml")
        assert body["status"] == "failed"
        assert "AgentConfigError" in body["error"]

    row = _wait_db(db, "bad-yaml")
    assert row["status"] == "failed"
    assert "AgentConfigError" in row["error"]
    assert row["agent_version"] == 0


def test_schema_has_indexes_and_wal(make_agent, tmp_path):
    db = tmp_path / "harness.db"
    app = _app(make_agent, tmp_path, db)
    with TestClient(app) as client:
        client.post("/v1/runs", json={"task": "schema", "run_id": "schema-1"})
        _wait(client, "schema-1")

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"runs", "run_events", "agent_versions", "checkpoints"} <= tables
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert {"idx_runs_created", "idx_events_kind"} <= indexes
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def _recv_until(ws, predicate, *, limit: int = 300) -> list[dict]:
    frames: list[dict] = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if predicate(frame):
            return frames
    raise AssertionError(f"condition not met; tail={frames[-8:]}")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
