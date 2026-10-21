from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core.loop import AgentLoop, MaxStepsExceeded
from .models.mock import MockChatModel
from .models.openai_compat import OpenAIChatModel
from .tools.base import ToolRegistry
from .tools.fs import fs_tools

SYSTEM_PROMPT = """You are an agent running inside a harness.
- Act only through the provided tools; all paths are relative to the workspace.
- Take one step at a time and treat tool results as evidence.
- When the task is done, reply with the final answer and no tool calls."""


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="eh", description="An agent harness that evolves its own skills."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run one task")
    run.add_argument("task")
    run.add_argument("--mock", action="store_true", help="use the offline deterministic model")
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--workspace", default=None)
    run.add_argument("--traces", default="traces")
    run.add_argument("--max-steps", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    return 1


async def _run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace or os.getenv("EH_WORKSPACE", "workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if args.mock:
        model: MockChatModel | OpenAIChatModel = MockChatModel()
    else:
        try:
            model = OpenAIChatModel(
                model=args.model or os.getenv("EH_MODEL", ""),
                base_url=args.base_url or os.getenv("EH_BASE_URL", ""),
                api_key=args.api_key or os.getenv("EH_API_KEY", ""),
            )
        except ValueError as exc:
            print(
                f"error: {exc}. Set EH_MODEL/EH_BASE_URL or pass --model/--base-url, "
                "or use --mock for an offline run.",
                file=sys.stderr,
            )
            return 2

    loop = AgentLoop(
        model=model,
        tools=ToolRegistry(fs_tools(workspace)),
        system_prompt=SYSTEM_PROMPT,
        max_steps=args.max_steps,
        trace_dir=args.traces,
    )
    try:
        result = await loop.run(args.task)
    except MaxStepsExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        await model.aclose()

    print(result.final, flush=True)
    print(f"[trace] {result.trace_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
