# Evolving Harness

An agent harness that evolves its own skills.

Two loops:

- **Agent loop** — the model does the work: messages, tool calls, observations, repeat.
- **Skill loop** — the harness makes it better: rollout, verify, evolve a skill, gate the change, keep or roll back.

The project is intentionally small: a readable core, a hardened execution path, and a
self-improvement loop that can be reproduced and rolled back. It is meant to be deployed
on a public endpoint, not just run on a laptop.

## Status

M0 — YAML-defined agent state, agent loop, model adapters, filesystem tools, JSONL
traces, CLI.

## Docs

- [docs/README.md](docs/README.md) — documentation system: layers, doc types, update rules (Chinese)
- [docs/v0.1/README.md](docs/v0.1/README.md) — current version: scope, milestones, index

The root README is the stable entry point; working docs live under `docs/` in Chinese.

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
  rsi.py        outer loop: evaluate, propose, gate, snapshot, rollback (M5-M6)
  sandbox.py    resident container pool (M3)
  serving.py    FastAPI control plane, SSE, auth (M2)
  store.py      runs, traces, evals (M2)
  ui.py         Gradio (M7)
```

## Roadmap

M0 loop and traces · M1 memory and context compaction · M2 serving and persistence ·
M3 sandbox pool · M4 YAML skills and on-demand loading · M5 tiered verifiers ·
M6 evolution gate · M7 public deployment · M8 evaluation report.

## License

MIT
