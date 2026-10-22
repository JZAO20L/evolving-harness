from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import (
    Agent,
    AgentConfigError,
    AgentState,
    MaxStepsExceeded,
    build_tools,
    create_model,
)

DEFAULT_AGENT = "agents/default/agent.yaml"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="eh", description="An agent harness that evolves its own skills."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run one task")
    run.add_argument("task")
    run.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        help=f"agent YAML path (default: {DEFAULT_AGENT})",
    )
    run.add_argument("--mock", action="store_true", help="use the offline deterministic model")
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--workspace", default=None)
    run.add_argument("--traces", default="traces")

    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    return 1


async def _run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace or os.getenv("EH_WORKSPACE", "workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        state = AgentState.load(args.agent)
        model = create_model(
            state,
            mock=args.mock,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        tools = build_tools(state, workspace)
    except AgentConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    agent = Agent(model=model, tools=tools, state=state, trace_dir=args.traces)
    try:
        result = await agent.run(args.task)
    except MaxStepsExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        await model.aclose()

    print(result.final, flush=True)
    print(f"[agent] {state.name} v{state.version} {state.hash[:8]}", file=sys.stderr)
    print(f"[trace] {result.trace_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
