# FluxMem

FluxMem is a Python/PostgreSQL prototype for lifecycle-aware conversational
memory. It combines PostgreSQL full-text search and pgvector retrieval, streams
answers, and performs memory extraction, reconciliation, and storage after each
completed turn.

## Local setup

```bash
python -m venv .venv
.venv/bin/pip install -e '.[local]'
cp .env.example .env
make db-setup
```

Add `OPENROUTER_API_KEY` to the untracked `.env` file. Do not place real
credentials in `.env.example` or source code.

## Developer API

The package root exposes the supported entry points and result types. The local
entry point reads `.env`, constructs the PostgreSQL/OpenRouter stack, and closes
its background executor and database engine through the context manager.

```python
from uuid import uuid4

from fluxmem import bootstrap_from_env


user_id = uuid4()
with bootstrap_from_env() as fluxmem:
    session = fluxmem.start_session(user_id=user_id)
    stream = fluxmem.stream_turn(
        user_id=user_id,
        session_id=session.session_id,
        content="What do I prefer to drink?",
        diagnostics=True,
    )
    for text_delta in stream:
        print(text_delta, end="", flush=True)

    result = stream.wait_for_post_answer()
    print(result.memory_outcomes)
```

Sequential evaluation harnesses can use `run_turn(...)` instead. It collects the
answer and waits for post-answer memory work before returning, ensuring the next
turn observes the preceding turn's completed writes.

Advanced integrations can import `bootstrap` from `fluxmem`, implement the
provider ports under `fluxmem.application.ports`, and supply their own model or
embedding adapters. Canonical domain types remain available from
`fluxmem.domain` as well as the curated types exported at the package root.
