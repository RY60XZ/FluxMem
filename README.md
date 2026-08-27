# FluxMem

FluxMem is a provider-neutral conversational memory layer backed by PostgreSQL.
It compresses messages into attributable memories, tracks temporal and lifecycle
metadata, retrieves context with hybrid dense and lexical search, and records
which memories were used.

## How it works

1. Ingest messages and extract directly supported, compressed facts.
2. Store source attribution, temporal bounds, embeddings, and lifecycle scores.
3. Retrieve with dense and lexical search, reciprocal-rank fusion, and lifecycle
   weighting.
4. Optionally rerank candidates and generate answers through the separate
   `infrastructure` package.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e .
make db-setup
```

```python
from fluxmem import bootstrap


memory = bootstrap(
    database_url="postgresql+psycopg://localhost/fluxmem",
    memory_extractor=extractor,
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

## LoCoMo evaluation

The optional [`infrastructure`](infrastructure) package provides OpenRouter
adapters and a resumable LoCoMo harness. In memory mode it learns the source
conversation, searches using only each benchmark question, retrieves 50
candidates, and reranks all 50 with `voyageai/rerank-2.5`. 

LoCoMo result from `conv-30` on the pinned official dataset
(categories 1-4), recorded on 2026-08-27 with
`google/gemma-4-26b-a4b-it` for extraction, lifecycle, answering, and judging;
`text-embedding-3-small`; and `voyageai/rerank-2.5`:

| Category | Correct | Accuracy |
|---|---:|---:|
| Overall | 73/81 | 90.12% |
| Single-hop | 38/44 | 86.36% |
| Temporal | 25/26 | 96.15% |
| Multi-hop | 10/11 | 90.91% |
