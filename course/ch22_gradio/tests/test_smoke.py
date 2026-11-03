"""L1 冒烟：前端客户端连接一个**真实运行中的后端**，跑真模型并流式到终态。

需要：
  EH_SMOKE=1
  EH_BACKEND_URL=http://127.0.0.1:8000   （后端需已用真实 ARK key 启动）

离线 / 后端未启动时自动跳过。Gradio 的浏览器 UI 不在此测试范围内。
"""

import asyncio
import os
import time

import pytest
from frontend import BackendClient, UiState, stream_state


@pytest.mark.smoke
def test_smoke_frontend_streams_real_run():
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    pytest.importorskip("websockets")

    backend = (
        os.getenv("EH_BACKEND_URL")
        or os.getenv("EH_FRONTEND_BACKEND")
        or "http://127.0.0.1:8000"
    )
    run_id = f"ui-smoke-{int(time.time())}"

    async def run() -> UiState:
        client = BackendClient(backend)
        try:
            try:
                await client.health()
            except Exception as exc:  # noqa: BLE001 - 后端未启动 → skip
                pytest.skip(f"后端不可达 {backend}: {exc}")
            await client.create_run("只回复：hello", run_id)
            state = UiState()
            session = client.session(run_id)
            async with session:
                async for current in stream_state(session, state):
                    if current.terminal:
                        break
            return state
        finally:
            await client.aclose()

    state = asyncio.run(run())
    assert state.status == "succeeded", state
    assert state.final.strip(), state
    assert state.run_id == run_id
    assert any(record.get("kind") == "model.response" for record in state.trace)
    print(f"\n[smoke] final={state.final!r} events={len(state.trace)}")
