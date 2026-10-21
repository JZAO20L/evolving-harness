from evolving_harness.core.message import Message, ToolCall


def test_tool_call_roundtrip():
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt", "limit": 10})
    raw = call.to_openai()
    assert raw["function"]["name"] == "read_file"
    assert ToolCall.from_openai(raw) == call


def test_tool_call_tolerates_bad_json():
    call = ToolCall.from_openai(
        {"id": "x", "function": {"name": "f", "arguments": "{not json"}}
    )
    assert call.name == "f"
    assert call.arguments == {"__raw__": "{not json"}


def test_message_serialization_with_tool_calls():
    message = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="list_dir", arguments={})],
    )
    raw = message.to_openai()
    assert raw["role"] == "assistant"
    assert raw["tool_calls"][0]["function"]["name"] == "list_dir"

    tool_message = Message(role="tool", content="ok", tool_call_id="c1")
    assert tool_message.to_openai()["tool_call_id"] == "c1"
