# Evolving Harness

An agent harness that evolves its own skills.

Two loops:

- **Agent loop** — the model does the work: messages, tool calls, observations, repeat.
- **Skill loop** — the harness makes it better: rollout, verify, evolve a skill, gate the change, keep or roll back.

The project is intentionally small: a readable core, a hardened execution path, and a
self-improvement loop that can be reproduced and rolled back. It is meant to be deployed
on a public endpoint, not just run on a laptop.

## Status

M0 — agent loop, model adapters, filesystem tools, JSONL traces, CLI.

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
```

A trace is written to `traces/<run_id>.jsonl` for every run.

## Design rules

- **nano**: core stays under ~4k lines; one dependency direction; no hidden magic.
- **production-grade**: auth, rate limits, timeouts, retries, idempotency, restart-safe
  runs, sandboxed execution, metrics. Not multi-tenant RBAC or billing.
- **RSI**: evolve the skills layer first. Promotions require a validation gate and a
  rollback path; no unreviewed self-modification.

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
evolving_harness/
  core/       loop, messages, context, memory
  models/     OpenAI-compatible adapter, mock model
  tools/      registry, filesystem tools
  tracing/    unified run trace (JSONL)
  sandbox/    resident container pool (M3)
  skills/     skill cards, registry, retrieval, evolution (M4)
  rsi/        rollout, verifiers, promotion gate (M5-M6)
  serving/    FastAPI control plane, SSE, auth (M2)
  store/      runs, traces, skills, evals (M2)
  ui/         Gradio (M7)
```

## Roadmap

M0 loop and traces · M1 memory and context compaction · M2 serving and persistence ·
M3 sandbox pool · M4 skills registry · M5 tiered verifiers · M6 evolution gate ·
M7 public deployment · M8 evaluation report.

## License

MIT
