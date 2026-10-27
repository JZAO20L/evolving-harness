"""RSI loop (v0.2). Depends only on ``agent.py``'s public entry.

v0.1 ships the agent; RSI — the Recursive Self-Improvement loop where the
harness proposes, verifies, gates and versions its own state edits — is v0.2
per DESIGN (docs/v0.1/DESIGN_v0.1_agent_core.md §8) and does not run here.

This skeleton fixes the architectural seam now: ``rsi.py`` may import only from
``agent`` (the public entry), never from internals, so a real proposer/verifier
can be added to the harness without disturbing the run loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent


@dataclass(slots=True)
class Rsi:
    """Placeholder for the v0.2 self-improvement loop.

    Intended surface (designed, not implemented until v0.2): ``propose()``
    drafts a state (YAML) edit from a completed run, ``verify()`` runs it
    against a benchmark + layered verifier, ``gate()`` decides promote / revert
    and registers a version. Before that lands, constructing Rsi is a no-op.
    """

    agent: Agent