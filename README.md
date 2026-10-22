# Evolving Harness

An agent harness that evolves its own skills.

Two loops:

- **Agent loop** — the model does the work: messages, tool calls, observations, repeat.
- **Skill loop** — the harness makes it better: rollout, verify, evolve a skill, gate the change, keep or roll back.

The project is intentionally small: a readable core, a hardened execution path, and a
self-improvement loop that can be reproduced and rolled back. It is meant to be deployed
on a public endpoint, not just run on a laptop.

## Status

M0/M0.5 complete — YAML-defined agent state, loop, adapters, filesystem tools,
JSONL traces, CLI. Working on M1: run state, context budget, compaction, memory,
subagent.

## Docs

Working docs live under `docs/` (Chinese) and are maintained locally as development
aids; they are not published in this repo. The root README is the stable public entry
point.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env        # set EH_MODEL / EH_BASE_URL for a real model

# offline smoke test
uv run eh run "say hello" --mock

# against any OpenAI-compatible endpoint (vLLM, DeepSeek, Qwen, ...)
uv run eh run "create notes.txt with a todo list" --workspace ./workspace

# a different agent definition
uv run eh run "say hello" --mock --agent agents/default/agent.yaml
```

Every run loads `agents/default/agent.yaml`, sends `state.version` and `state.hash` to
the trace, and writes `traces/<run_id>.jsonl`.

## Design rules

- **nano**: agent = one YAML + one Python. The YAML is versioned static state (prompt,
  guidance, skills, enabled tools); `agent.py` is the dynamic loop. Plumbing stays in
  small modules; the agent itself stays readable in one sitting. New concerns start as
  one file; a directory appears only when it holds multiple real files.
- **production-grade**: auth, rate limits, timeouts, retries, idempotency, restart-safe
  runs, sandboxed execution, metrics. Not multi-tenant RBAC or billing.
- **RSI**: `rsi.py` is the outer loop. It measures a frozen agent, edits only the YAML,
  and keeps a change only if the validation gate accepts it. Python is never rewritten
  by the optimizer.

Tool boundaries are declared, not negotiated at runtime:

```python
class ExecTool(Tool):
    isolation = "container"   # none | process | container | microvm
    network   = "none"        # none | proxy-allowlist
    fs_scope  = ["/workspace"]
    budget    = {"wall_s": 30, "mem_mb": 512, "pids": 64}
```

## Layout

```
agents/
  default/agent.yaml   agent state: prompt, guidance, skills, tools, limits
benchmarks/            frozen cases: statement (public) + checks/rubric (private)
evolving_harness/
  agent.py      state loader, prompt rendering, agent loop
  cli.py        eh run; thin wrapper over agent.py
  core/         message and tool-call protocol
  models/       OpenAI-compatible adapter, mock model
  tools/        registry, filesystem tools
  tracing/      unified run trace (JSONL)

  # later: one file per concern, split only when it stops being readable
  rsi.py        outer loop: rollout, tiered verifier, gate, version registry (M2-M3)
  serving.py    single-file FastAPI + SQLite + SSE + run state (M4)
  sandbox.py    resident container pool (M5)
  ui.py         Gradio (v0.4, M6)
```

## Roadmap

M0 loop and traces · M1 complete agent (run state, context, memory, subagent) ·
M2 RSI benchmark and tiered verifiers · M3 evolution gate and version registry ·
M4 single-file backend serving with persistent state · M5 sandbox pool ·
M6 web frontend · M7 public deploy and evaluation report.

## License

MIT
