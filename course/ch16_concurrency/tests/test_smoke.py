"""L1 冒烟：真模型（deepseek @ ARK）走异步作业队列完整跑一次。

离线默认跳过：需 EH_SMOKE=1 + ARK key。断言 202 → 轮询 succeeded → trace 落盘，
不赌模型措辞。
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
import yaml
from backend import RunService, create_app

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
        "name": "concurrency-smoke",
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


async def _poll(client: httpx.AsyncClient, run_id: str, *, timeout: float = 180.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    body: dict = {}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/v1/runs/{run_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        await asyncio.sleep(0.25)
    raise AssertionError(f"job {run_id} did not finish: {body}")


@pytest.mark.smoke
async def test_smoke_async_job_with_real_model(tmp_path):
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
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            timeout=180.0,
        ) as client:
            created = await client.post(
                "/v1/runs", json={"task": "只回复：hello", "run_id": "job-smoke"}
            )
            assert created.status_code == 202, created.text
            assert created.json()["status"] in ("queued", "running")

            body = await _poll(client, "job-smoke")
    finally:
        await app.state.manager.stop()

    assert body["status"] == "succeeded", body
    result = body["result"]
    assert result["status"] == "ok"
    assert result["steps"] >= 1
    assert result["final"].strip()
    assert Path(body["workspace"]).name == "job-smoke"
    assert (tmp_path / "ws" / "runs" / "job-smoke").is_dir()

    trace = tmp_path / "traces" / "job-smoke.jsonl"
    assert trace.is_file()
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert events[0]["kind"] == "run.start"
    assert events[-1]["kind"] == "run.end"

    print(f"\n[smoke] {body['status']} steps={result['steps']} final={result['final']!r}")
