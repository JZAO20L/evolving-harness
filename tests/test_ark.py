import json
from types import SimpleNamespace

import httpx
import pytest

from evolving_harness.agent import State, create_model
from evolving_harness.core.message import Message, ToolCall
from evolving_harness.models.ark import (
    ArkChatModel,
    _build_input,
    _parse_output,
    specs_to_tools,
)
from evolving_harness.models.base import Usage
from evolving_harness.models.openai_compat import OpenAIChatModel

ARK_URL = "https://ark.cn-beijing.volces.com/api/v3"

# The build shell exports a no_proxy with raw IPv6 CIDRs (fe80::/10, fd00::/8)
# that some httpx versions misparse as bracket hosts -> httpx.InvalidURL on
# client construction. Tests that build a real model client must be hermetic.
@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


def test_build_input_maps_system_to_instructions():
    messages = [
        Message(role="system", content="You are strict."),
        Message(role="user", content="hi"),
    ]
    instructions, items = _build_input(messages)
    assert instructions == "You are strict."
    assert items == [{"role": "user", "content": "hi"}]


def test_build_input_pairs_function_call_and_output():
    call = ToolCall(id="c1", name="read_file", arguments={"path": "x.txt"})
    messages = [
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="hello", tool_call_id="c1"),
        Message(role="user", content="thanks"),
    ]
    instructions, items = _build_input(messages)
    assert instructions == ""
    assert items == [
        {"type": "function_call", "call_id": "c1", "name": "read_file",
         "arguments": '{"path": "x.txt"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "hello"},
        {"role": "user", "content": "thanks"},
    ]


def test_specs_to_tools_flattens_function_schema():
    specs = [
        {"type": "function", "function": {
            "name": "read_file", "description": "d", "parameters": {"type": "object"}}}
    ]
    assert specs_to_tools(specs) == [
        {"type": "function", "name": "read_file",
         "description": "d", "parameters": {"type": "object"}}
    ]
    assert specs_to_tools(None) is None
    assert specs_to_tools([]) is None


def text_item(text):
    return SimpleNamespace(type="output_text", text=text)


def message_item(text):
    return SimpleNamespace(type="message", content=[text_item(text)])


def test_parse_output_extracts_text_and_function_calls():
    fn = SimpleNamespace(
        type="function_call", call_id="c9", name="write_file",
        arguments='{"path": "n.txt"}',
    )
    response = SimpleNamespace(
        status="completed",
        output=[message_item("hello"), message_item(" world"), fn],
        usage=SimpleNamespace(input_tokens=7, output_tokens=2),
    )
    content, calls, usage, reason = _parse_output(response)
    assert content == "hello world"
    assert calls == [ToolCall(id="c9", name="write_file", arguments={"path": "n.txt"})]
    assert usage == Usage(7, 2)
    assert reason == "completed"


def _responses_payload(text, call=None, status="completed"):
    output = [
        {"type": "message", "id": "m1", "status": "completed", "role": "assistant",
         "content": [{"type": "output_text", "annotations": [], "logprobs": [],
                      "text": text}]},
    ]
    if call is not None:
        output.append({
            "type": "function_call", "id": "fc1", "call_id": call["id"],
            "name": call["name"], "arguments": call["arguments"], "status": "completed",
        })
    return {
        "id": "resp_1", "object": "response", "status": status, "model": "mock",
        "created_at": 0, "incomplete_details": None, "instructions": None,
        "max_output_tokens": None, "metadata": {}, "output": output,
        "parallel_tool_calls": True, "previous_response_id": None, "reasoning": None,
        "store": False, "temperature": 0.2, "text": {"format": {"type": "text"}},
        "tool_choice": "auto", "tools": None, "top_p": 1, "truncation": "disabled",
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        "user": None, "n": 1, "output_audio": None, "include": [],
    }


async def test_complete_roundtrip_via_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/responses"
        body = json.loads(request.content.decode())
        assert body["input"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(200, json=_responses_payload("你好", call={
            "id": "call_x", "name": "list_dir",
            "arguments": json.dumps({"page": 1}),
        }))

    client = ArkChatModel(
        model="doubao-seed-2-1-pro-260915",
        base_url=ARK_URL,
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await client.complete([Message(role="user", content="hi")])

    assert result.content == "你好"
    assert result.tool_calls == [
        ToolCall(id="call_x", name="list_dir", arguments={"page": 1})
    ]
    assert result.usage == Usage(5, 1)
    assert result.model == "doubao-seed-2-1-pro-260915"
    await client.aclose()


def test_create_model_selects_ark_by_host(make_agent):
    path = make_agent(model={
        "id": "doubao-seed-2-1-pro-260915", "base_url": ARK_URL, "api_key": "x",
    })
    state = State.load(path)
    model = create_model(state)
    assert isinstance(model, ArkChatModel)
    assert model.name == "doubao-seed-2-1-pro-260915"


def test_create_model_defaults_to_openai_compat(make_agent):
    model = create_model(State.load(make_agent()))
    assert isinstance(model, OpenAIChatModel)


def test_create_model_provider_override_force_ark(make_agent):
    path = make_agent(
        model={"id": "m", "base_url": "http://localhost:8000/v1", "provider": "ark"}
    )
    model = create_model(State.load(path))
    assert isinstance(model, ArkChatModel)