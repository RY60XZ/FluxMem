# FluxMem

FluxMem is an independent conversational memory layer. It stores messages,
extracts and reconciles memories, retrieves relevant context, and records memory
usage. It does not generate answers or cache answering-model responses.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .
make db-setup
```

## Public API

Applications provide memory extraction, reconciliation, lifecycle, and embedding
implementations through provider-neutral ports. The `FluxMem` facade is the
supported integration boundary:

```python
from fluxmem import bootstrap


memory = bootstrap(
    database_url="postgresql+psycopg://localhost/fluxmem",
    memory_extractor=extractor,
    memory_reconciler=reconciler,
    embedding_provider=embedder,
)
session = memory.start_session(user_id=user_id)
memory.ingest_messages(user_id=user_id, messages=(message,))

result = memory.retrieve(
    user_id=user_id,
    session_id=session.session_id,
    query="What does the user prefer to drink?",
)
memory.record_usage(
    user_id=user_id,
    session_id=session.session_id,
    query_id=result.query_id,
    used_memory_ids=tuple(
        item.memory.memory_id for item in result.context.memories
    ),
)
memory.close()
```

The facade also exposes `store_messages(...)` for persistence without memory
learning, `get_session_history(...)`, and `reindex_pending_memories(...)`.

## Optional infrastructure

The separate [`infrastructure`](infrastructure) project demonstrates how an
answering agent, concrete OpenRouter providers, environment configuration, and
the LoCoMo harness can consume FluxMem without entering the core package.

```bash
.venv/bin/pip install -e './infrastructure[locomo]'
cp .env.example .env
fluxmem-locomo run --all --output benchmark-runs/full
fluxmem-locomo run --conversation conv-26 --output benchmark-runs/conv-26
```

The infrastructure runtime defaults its answering, extraction, reconciliation,
and lifecycle tasks to `deepseek/deepseek-v4-flash-0731`.
