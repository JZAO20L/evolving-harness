"""L1 冒烟：真模型（deepseek @ ARK）跑一次作业，并用 SSE 收完整事件流。

离线默认跳过：需 EH_SMOKE=1 + ARK key。断言事件序与终止，不赌模型措辞。
"""

import json
import os
from pathlib import Path
from typing import Any

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
        "name": "sse-smoke",
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


async def _collect(client: httpx.AsyncClient, url: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with client.stream("GET", url) as response:
        assert response.status_code == 200, await response.aread()
        event: str | None = None
        event_id: int | None = None
        data: list[str] = []
        async for line in response.aiter_lines():
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


@pytest.mark.smoke
async def test_smoke_sse_run_with_real_model(tmp_path):
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
                "/v1/runs", json={"task": "只回复：hello", "run_id": "sse-smoke"}
            )
            assert created.status_code == 202, created.text
            events = await _collect(client, "/v1/runs/sse-smoke/events")
    finally:
        await app.state.manager.stop()

    kinds = [item["event"] for item in events]
    assert kinds[0] == "open"
    assert "run.start" in kinds
    assert "model.response" in kinds
    assert kinds[-2:] == ["run.end", "job.end"]

    job_end = events[-1]["data"]
    assert job_end["status"] == "succeeded", job_end
    result = job_end["result"]
    assert result["status"] == "ok"
    assert result["final"].strip()
    assert (tmp_path / "traces" / "sse-smoke.jsonl").is_file()

    print(
        f"\n[smoke] events={len(events)} steps={result['steps']} "
        f"final={result['final']!r}"
    )
