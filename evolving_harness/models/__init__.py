"""Model adapters behind a common ``ChatModel`` interface.

- ``mock`` — scripted ; only for ``--mock`` / tests.
- ``openai_compat`` — zero-dep httpx ``/chat/completions``.
- ``ark`` — ARK (Volcengine) via the openai SDK Responses API ``/responses``.

Selection by ``create_model`` (agent.py): ARK when ``provider: ark`` or base_url
host is ``volces.com``; otherwise OpenAI-compatible.
"""
