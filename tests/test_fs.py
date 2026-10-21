from evolving_harness.core.message import ToolCall
from evolving_harness.tools.base import ToolRegistry
from evolving_harness.tools.fs import fs_tools


async def dispatch(tmp_path, name, **arguments):
    registry = ToolRegistry(fs_tools(tmp_path / "ws"))
    return await registry.dispatch(ToolCall(id="c1", name=name, arguments=arguments))


async def test_write_then_read(tmp_path):
    written = await dispatch(tmp_path, "write_file", path="a/b.txt", content="one\ntwo")
    assert written.content.startswith("wrote 7 bytes")

    read = await dispatch(tmp_path, "read_file", path="a/b.txt")
    assert "1\tone" in read.content and "2\ttwo" in read.content


async def test_path_escape_is_blocked(tmp_path):
    result = await dispatch(tmp_path, "read_file", path="../../etc/passwd")
    assert result.content.startswith("ERROR: ValueError")
    assert "escapes workspace" in result.content


async def test_absolute_path_is_blocked(tmp_path):
    result = await dispatch(tmp_path, "write_file", path="/tmp/evil.txt", content="x")
    assert result.content.startswith("ERROR: ValueError")


async def test_unknown_tool(tmp_path):
    result = await dispatch(tmp_path, "nope")
    assert result.content.startswith("ERROR: unknown tool")
