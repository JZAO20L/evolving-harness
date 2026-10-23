from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
from functools import partial
from pathlib import Path

from .base import Tool, ToolIsolation

MAX_OUTPUT_BYTES = 64 * 1024
MAX_TIMEOUT_MS = 30_000
MEMORY_MB = 512
MAX_COMMAND_CHARS = 8192

BASH_ISOLATION = ToolIsolation(
    level="spawn",
    network="none",
    fs_scope="workspace_writable",
    resources={
        "time_ms": MAX_TIMEOUT_MS,
        "mem_mb": MEMORY_MB,
        "output_bytes": MAX_OUTPUT_BYTES,
    },
)


def bash_tool(root: str | Path, protected: tuple[str, ...] = ()) -> Tool:
    """Run a non-interactive command in a bubblewrap sandbox under `root`.

    The workspace is the only writable host path (protected subpaths are mounted
    read-only), the network is unshared, and CPU/memory/output are hard-capped.
    Without bwrap the tool fails closed rather than running a bare subprocess.
    """
    workspace = Path(root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    async def bash(command: str, timeout_ms: int = MAX_TIMEOUT_MS) -> str:
        command = str(command)
        if not command.strip():
            raise ValueError("command must not be empty")
        if len(command) > MAX_COMMAND_CHARS:
            raise ValueError(f"command must be at most {MAX_COMMAND_CHARS} characters")
        if isinstance(timeout_ms, bool):
            raise TypeError("timeout_ms must be an integer")
        try:
            timeout_ms = int(timeout_ms)
        except (TypeError, ValueError):
            raise ValueError("timeout_ms must be an integer") from None
        if not 1 <= timeout_ms <= MAX_TIMEOUT_MS:
            raise ValueError(f"timeout_ms must be between 1 and {MAX_TIMEOUT_MS}")

        bwrap = shutil.which("bwrap")
        if bwrap is None:
            return "ERROR: bash requires bwrap; refusing to run without a sandbox"

        argv = [bwrap]
        for path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(path).exists():
                argv += ["--ro-bind", path, path]
        argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        argv += ["--bind", str(workspace), str(workspace)]
        for item in protected:
            target = (workspace / item).resolve()
            if target.is_dir() and target.is_relative_to(workspace):
                argv += ["--ro-bind", str(target), str(target)]
        argv += [
            "--chdir",
            str(workspace),
            "--unshare-net",
            "--die-with-parent",
            "--setenv",
            "HOME",
            str(workspace),
            "--setenv",
            "PATH",
            "/usr/local/bin:/usr/bin:/bin",
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
        ]

        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=partial(_limit_resources, timeout_ms),
            start_new_session=True,
        )
        buffer = bytearray()
        truncated = False
        timed_out = False
        try:
            async with asyncio.timeout(timeout_ms / 1000):
                assert process.stdout is not None
                while True:
                    chunk = await process.stdout.read(8192)
                    if not chunk:
                        break
                    room = MAX_OUTPUT_BYTES - len(buffer)
                    if room > 0:
                        buffer.extend(chunk[:room])
                    if len(chunk) > room:
                        truncated = True
                    if len(buffer) >= MAX_OUTPUT_BYTES:
                        truncated = True
                        _kill_group(process)
                        break
                if truncated:
                    await process.communicate()
                else:
                    await process.wait()
        except TimeoutError:
            timed_out = True
            _kill_group(process)
            await process.communicate()

        text = buffer.decode("utf-8", errors="replace")
        notes = []
        if timed_out:
            notes.append(f"timeout>{timeout_ms}ms")
        if truncated:
            notes.append("output truncated")
        header = f"exit={process.returncode}"
        if notes:
            header += " (" + ", ".join(notes) + ")"
        body = f"\n{text}" if text else ""
        if process.returncode == 0 and not timed_out:
            return header + body
        return "ERROR: " + header + body

    return Tool(
        name="bash",
        description=(
            "Run a non-interactive shell command inside the workspace. "
            "stdout and stderr are merged; the command has no network access."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_ms": {
                    "type": "integer",
                    "description": f"Timeout in ms, 1..{MAX_TIMEOUT_MS} (default {MAX_TIMEOUT_MS}).",
                },
            },
            "required": ["command"],
        },
        handler=bash,
        isolation=BASH_ISOLATION,
    )


def _limit_resources(timeout_ms: int) -> None:
    cpu_seconds = max(1, timeout_ms // 1000 + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    memory = MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))


def _kill_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
