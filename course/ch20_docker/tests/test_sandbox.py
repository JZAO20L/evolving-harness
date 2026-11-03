"""L0：Docker 参数硬化、池的并发/超时/失败关闭，以及 bash 工具替换。"""

import asyncio
import json

import pytest
import yaml
from backend import (
    CONTAINER_ISOLATION,
    CommandResult,
    RunService,
    SandboxPool,
    SandboxSpec,
    build_sandbox_pool,
    sandbox_run_args,
)

from evolving_harness.core.message import ToolCall
from evolving_harness.models.base import ModelResponse
from evolving_harness.models.mock import MockChatModel
from evolving_harness.tools.base import Tool, ToolRegistry


class FakeDockerCLI:
    def __init__(
        self,
        *,
        exec_output: str = "hi\n",
        exec_code: int = 0,
        exec_delay: float = 0.0,
        start_code: int = 0,
        exec_times_out: bool = False,
    ) -> None:
        self.exec_output = exec_output
        self.exec_code = exec_code
        self.exec_delay = exec_delay
        self.start_code = start_code
        self.exec_times_out = exec_times_out
        self.calls: list[tuple[list[str], float]] = []
        self.running: set[str] = set()
        self.max_running = 0

    async def run(self, args: list[str], *, timeout: float = 30.0) -> CommandResult:
        self.calls.append((list(args), timeout))
        if args[0] == "run":
            name = args[args.index("--name") + 1]
            if self.start_code:
                return CommandResult(self.start_code, "image not found")
            self.running.add(name)
            self.max_running = max(self.max_running, len(self.running))
            return CommandResult(0, "container-id")
        if args[0] == "exec":
            if self.exec_times_out:
                return CommandResult(124, "", timed_out=True)
            if self.exec_delay:
                try:
                    await asyncio.wait_for(asyncio.sleep(self.exec_delay), timeout)
                except TimeoutError:
                    return CommandResult(124, "", timed_out=True)
            return CommandResult(self.exec_code, self.exec_output)
        if args[0] == "rm":
            self.running.discard(args[-1])
            return CommandResult(0, "")
        if args[0] == "kill":
            return CommandResult(0, "")
        return CommandResult(0, "")

    def of(self, verb: str) -> list[list[str]]:
        return [args for args, _ in self.calls if args[0] == verb]


def test_run_args_harden_container_boundary(tmp_path):
    spec = SandboxSpec(image="sbx:test", memory_mb=256, cpus=0.5, user="1000:1000")
    args = sandbox_run_args(spec, "eh-sbx-1", tmp_path)

    assert args[:3] == ["run", "-d", "--name"]
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert args[args.index("--memory") + 1] == "256m"
    assert args[args.index("--cpus") + 1] == "0.5"
    assert args[args.index("--pids-limit") + 1] == "128"
    assert "--read-only" in args
    assert "--cap-drop" in args and "ALL" in args
    assert "no-new-privileges" in args
    assert args[args.index("--user") + 1] == "1000:1000"
    assert f"{tmp_path.resolve()}:/workspace:rw" in args
    assert args[-1] == "sbx:test"
    assert CONTAINER_ISOLATION.level == "container"


async def test_pool_runs_exec_and_removes_container(tmp_path):
    cli = FakeDockerCLI(exec_output="hello\n")
    pool = SandboxPool(cli, max_size=2)

    out = await pool.run("echo hello", tmp_path / "ws")

    assert out == "exit=0\nhello\n"
    assert [args[0] for args, _ in cli.calls] == ["run", "exec", "rm"]
    assert cli.running == set()
    assert pool.active == 0


async def test_pool_caps_concurrent_containers(tmp_path):
    cli = FakeDockerCLI(exec_delay=0.03)
    pool = SandboxPool(cli, max_size=2)

    await asyncio.gather(
        *(pool.run(f"echo {index}", tmp_path / f"ws-{index}") for index in range(5))
    )

    assert cli.max_running == 2
    assert cli.running == set()
    assert len(cli.of("run")) == 5
    assert len(cli.of("rm")) == 5


async def test_pool_timeout_kills_and_cleans_container(tmp_path):
    cli = FakeDockerCLI(exec_delay=0.5)
    pool = SandboxPool(cli)

    out = await pool.run("sleep 1", tmp_path, timeout_ms=20)

    assert out.startswith("ERROR: exit=124")
    assert "timeout>20ms" in out
    assert len(cli.of("kill")) == 1
    assert cli.running == set()


async def test_pool_fails_closed_when_start_fails(tmp_path):
    cli = FakeDockerCLI(start_code=125)
    pool = SandboxPool(cli)

    out = await pool.run("echo hi", tmp_path)

    assert out.startswith("ERROR: sandbox start failed:")
    assert cli.of("exec") == []
    assert len(cli.of("rm")) == 1


async def test_pool_truncates_large_output(tmp_path):
    cli = FakeDockerCLI(exec_output="x" * (70 * 1024))
    pool = SandboxPool(cli)

    out = await pool.run("yes x", tmp_path)

    assert "output truncated" in out
    assert len(out.encode()) < 66 * 1024


async def test_pool_rejects_bad_commands(tmp_path):
    pool = SandboxPool(FakeDockerCLI())
    with pytest.raises(ValueError):
        await pool.run("   ", tmp_path)
    with pytest.raises(ValueError):
        await pool.run("x" * 9000, tmp_path)
    with pytest.raises(ValueError):
        await pool.run("echo", tmp_path, timeout_ms=0)


async def test_registry_turns_sandbox_failure_into_tool_data(tmp_path):
    pool = SandboxPool(FakeDockerCLI(start_code=125))
    tool = Tool(
        name="bash",
        description="container bash",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        handler=pool.bash_handler(tmp_path),
    )
    registry = ToolRegistry([tool])

    message = await registry.dispatch(
        ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})
    )

    assert message.content.startswith("ERROR: sandbox start failed:")


def test_service_uses_container_bash_for_agent(tmp_path):
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "docker-test",
                "model": {"id": "mock", "base_url": "http://x", "max_tokens": 32},
                "limits": {"max_steps": 3},
                "prompt": {"system": "s", "guidance": "g"},
                "tools": {"enabled": ["bash"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    script = [
        ModelResponse(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})
            ],
        ),
        ModelResponse(content="done"),
    ]
    cli = FakeDockerCLI(exec_output="hi\n")
    pool = SandboxPool(cli)
    service = RunService(
        agent_path=agent_path,
        workspace=tmp_path / "ws",
        trace_dir=tmp_path / "traces",
        model_factory=lambda _state: MockChatModel(list(script)),
        sandbox=pool,
    )

    result = asyncio.run(service.run("run bash", run_id="sbx-1"))

    assert result.status == "ok"
    assert result.final == "done"
    exec_calls = cli.of("exec")
    assert len(exec_calls) == 1
    assert exec_calls[0][-1] == "echo hi"
    trace = (tmp_path / "traces" / "sbx-1.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in trace.splitlines()]
    tool_results = [item for item in records if item["kind"] == "tool.result"]
    assert tool_results and "hi" in json.dumps(tool_results[-1]["data"], ensure_ascii=False)


def test_build_pool_only_when_env_requests_docker(monkeypatch):
    monkeypatch.delenv("EH_SANDBOX", raising=False)
    assert build_sandbox_pool() is None

    monkeypatch.setenv("EH_SANDBOX", "docker")
    monkeypatch.setenv("EH_SANDBOX_POOL", "2")
    monkeypatch.setenv("EH_SANDBOX_IMAGE", "sbx:env")
    pool = build_sandbox_pool()
    assert pool is not None
    assert pool.max_size == 2
    assert pool.spec.image == "sbx:env"
