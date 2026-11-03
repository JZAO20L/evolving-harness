"""L1 冒烟：真模型（deepseek @ ARK）跑一次作业，WebSocket 收事件并 ping/pong。

离线默认跳过：需 EH_SMOKE=1 + ARK key。断言事件序与终止，不赌模型措辞。
"""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from backend import RunService, create_app
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
        "name": "ws-smoke",
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


def _recv_until(ws, predicate, *, limit: int = 500) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if predicate(frame):
            return frames
    raise AssertionError(f"condition not met; tail={frames[-8:]}")


@pytest.mark.smoke
def test_smoke_ws_run_with_real_model(tmp_path):
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    if not (os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")):
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")

    app = create_app(
        service=RunService(
            agent_path=_write_agent(tmp_path),
            workspace=tmp_path / "ws",
            trace_dir=tmp_path / "traces",
        ),
        max_concurrency=1,
    )
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"task": "只回复：hello", "run_id": "ws-smoke"})
        assert created.status_code == 202, created.text
        with client.websocket_connect("/v1/runs/ws-smoke/ws") as ws:
            ws.send_json({"type": "ping"})
            frames = _recv_until(ws, lambda f: f["type"] == "job.end")

    types = [frame["type"] for frame in frames]
    kinds = [f["event"] for f in frames if f["type"] == "event"]
    assert frames[0]["type"] == "hello"
    assert "run.start" in kinds
    assert "model.response" in kinds
    assert kinds[-1] == "run.end"

    job_end = frames[-1]["job"]
    assert job_end["status"] == "succeeded", job_end
    result = job_end["result"]
    assert result["status"] == "ok"
    assert result["final"].strip()
    assert (tmp_path / "traces" / "ws-smoke.jsonl").is_file()

    print(
        f"\n[smoke] frames={len(frames)} pong={'pong' in types} "
        f"steps={result['steps']} final={result['final']!r}"
    )
