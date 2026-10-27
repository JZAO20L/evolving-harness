import asyncio

import httpx
import pytest
from agent import (
    Message,
    MockChatModel,
    ModelError,
    ModelResult,
    OpenAIChatModel,
    Usage,
)


async def _no_sleep(_):
    return None


# 本机 shell 的 no_proxy 带裸 IPv6 CIDR（fe80::/10 等），个别 httpx 版本在建
# client 时会解析失败（InvalidURL）。测试一律隔离，避免被环境代理坑到。
@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


def test_usage_total():
    assert Usage(prompt_tokens=3, completion_tokens=2).total_tokens == 5


async def test_mock_echoes_last_user_and_uses_fixed_usage():
    model = MockChatModel()
    result = await model.complete([Message(role="user", content="hi")])
    assert result.content == "mock:hi"
    assert result.usage == Usage(prompt_tokens=10, completion_tokens=5)
    assert result.finish_reason == "stop"


async def test_mock_script_pops_results_in_order():
    model = MockChatModel(script=[
        ModelResult("one", usage=Usage(prompt_tokens=1, completion_tokens=1)),
        ModelResult("two", usage=Usage(prompt_tokens=2, completion_tokens=2)),
    ])
    assert (await model.complete([])).content == "one"
    assert (await model.complete([])).content == "two"


def _chat_payload(text, *, prompt=5, completion=1, finish="stop"):
    return {
        "id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    }


async def test_openai_roundtrip_via_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = request.read().decode()
        assert '"model": "m"' in body or '"model":"m"' in body
        return httpx.Response(200, json=_chat_payload("你好", prompt=7, completion=3))

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAIChatModel(
        base_url="http://localhost:8000/v1", model="m",
        http_client=transport,
    )
    result = await model.complete([Message(role="user", content="hi")])
    assert result.content == "你好"
    assert result.usage == Usage(prompt_tokens=7, completion_tokens=3)
    assert result.finish_reason == "stop"
    await model.aclose()


async def test_openai_non_200_raises_model_error():
    def handler(_):
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAIChatModel(base_url="http://localhost:8000/v1", model="m",
                            max_retries=0, http_client=transport)
    with pytest.raises(ModelError):
        await model.complete([Message(role="user", content="hi")])
    await model.aclose()


async def test_openai_requires_model_and_base_url():
    with pytest.raises(ValueError):
        OpenAIChatModel(base_url="", model="m")
    with pytest.raises(ValueError):
        OpenAIChatModel(base_url="http://localhost:8000/v1", model="")


async def test_openai_retries_on_500_then_succeeds(monkeypatch):
    calls = []

    def handler(_):
        calls.append(1)
        if len(calls) < 2:
            return httpx.Response(500, json={})
        return httpx.Response(200, json=_chat_payload("ok"))

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    model = OpenAIChatModel(
        base_url="http://localhost:8000/v1", model="m", max_retries=2,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await model.complete([Message(role="user", content="hi")])
    assert result.content == "ok"
    assert len(calls) == 2
    await model.aclose()


async def test_openai_only_retries_retryable_status(monkeypatch):
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(400, json={})  # 业务性 4xx：不重试

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    model = OpenAIChatModel(
        base_url="http://localhost:8000/v1", model="m", max_retries=3,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ModelError):
        await model.complete([Message(role="user", content="hi")])
    assert len(calls) == 1  # 一次就失败退出
    await model.aclose()


async def test_openai_exhausts_retries_then_raises(monkeypatch):
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(503, json={})

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    model = OpenAIChatModel(
        base_url="http://localhost:8000/v1", model="m", max_retries=2,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ModelError):
        await model.complete([Message(role="user", content="hi")])
    assert len(calls) == 3  # 初始 1 次 + 2 次重试
    await model.aclose()