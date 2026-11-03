"""L1 冒烟：真模型跑一次作业；热状态命中 Redis 语义层，重启后仍能从 SQL 回退读取。

离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest
import yaml
from backend import MemoryKV, RunService, RunState, create_app
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    env_file = ROOT / ".env"
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
    if not os.getenv("EH_API_KEY") and os.getenv("ARK_API_KEY"):
        os.environ["EH_API_KEY"] = os.environ["ARK_API_KEY"]


@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    for key in ("ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


def _write_agent(tmp_path: Path) -> Path:
    data = {
        "version": 1,
        "name": "sql-smoke",
        "model": {
            "id": os.getenv("EH_MODEL", "deepseek-v4-flash-ga-260731"),
            "base_url": os.getenv("EH_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "limits": {"max_steps": 2},
        "prompt": {
            "system": "You are a terse assistant. Reply with the exact text requested.",
            "guidance": "Do not call tools unless the task requires reading files.",
        },
        "tools": {"enabled": ["read_file"]},
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _app(agent_path: Path, tmp_path: Path, db_path: Path, hot: RunState):
    return create_app(
        service=RunService(
            agent_path=agent_path,
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
        ),
        max_concurrency=1,
        db_path=db_path,
        hot_state=hot,
    )


def _wait(client: TestClient, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish: {body}")


def _wait_checkpoint(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/runs/{run_id}/checkpoint")
        if response.status_code == 200:
            return response.json()
        time.sleep(0.05)
    raise AssertionError(f"checkpoint {run_id} was not persisted")


@pytest.mark.smoke
def test_smoke_redis_run_and_restart(tmp_path):
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    if not (os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")):
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")

    db = tmp_path / "smoke.db"
    agent_path = _write_agent(tmp_path)
    hot1 = RunState(MemoryKV())

    with TestClient(_app(agent_path, tmp_path, db, hot1)) as client:
        created = client.post(
            "/v1/runs", json={"task": "只回复：hello", "run_id": "redis-smoke"}
        )
        assert created.status_code == 202, created.text
        body = _wait(client, "redis-smoke")
        assert body["status"] == "succeeded", body
        result = body["result"]
        assert result["status"] == "ok"
        assert result["final"].strip()
        checkpoint = _wait_checkpoint(client, "redis-smoke")
        assert checkpoint["state"]["run_id"] == "redis-smoke"

    cached = asyncio.run(hot1.get_view("redis-smoke"))
    assert cached is not None and cached.status == "succeeded"

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE run_id = 'redis-smoke'").fetchone()
        assert row["status"] == "succeeded"
        assert row["agent_version"] == 1
        assert len(row["agent_hash"]) == 64
        assert json.loads(row["result_json"])["final"] == result["final"]
        kinds = [
            item["kind"]
            for item in conn.execute(
                "SELECT kind FROM run_events WHERE run_id = 'redis-smoke' ORDER BY seq"
            )
        ]
        assert kinds[0] == "run.start"
        assert kinds[-1] == "run.end"
        assert "model.response" in kinds

    # 重启：新建空的热状态，只共享 SQL 与 trace 文件 → 走数据库回退。
    hot2 = RunState(MemoryKV())
    with TestClient(_app(agent_path, tmp_path, db, hot2)) as client:
        replay = client.get("/v1/runs/redis-smoke").json()
        assert replay["status"] == "succeeded"
        assert replay["result"]["final"] == result["final"]
        history = client.get("/v1/runs", params={"limit": 10}).json()
        assert any(item["run_id"] == "redis-smoke" for item in history)
        versions = client.get("/v1/agent/versions").json()
        assert any(item["agent_hash"] == row["agent_hash"] for item in versions)

    print(f"\n[smoke] final={result['final']!r} steps={result['steps']} events={len(kinds)}")
