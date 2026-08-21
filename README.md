# FluxMem

FluxMem is a synchronous Python/PostgreSQL memory service with lifecycle-aware
hybrid retrieval. Canonical memories and all retrieval projections remain in
one PostgreSQL transaction.

## Retrieval behavior

Each query uses the available retrieval legs independently:

- PostgreSQL full-text search over a stored `TSVECTOR` projection.
- Cosine similarity over a 1536-dimensional pgvector projection.

The candidate ranks are combined with weighted reciprocal-rank fusion (RRF),
then adjusted by the memory's lazily calculated lifecycle retention. Deleted,
cross-user, cross-session, and currently invalid memories are hard-filtered
before they can be returned.

Embedding failures do not make a memory unavailable. Its lexical projection is
stored with `index_status = pending`, and it remains full-text searchable until
the vector is repaired.

## Setup

Create the virtual environment and install the package, then start and migrate
the pgvector-enabled PostgreSQL service:

```bash
python -m venv .venv
.venv/bin/pip install -e .
make db-setup
```

Migration `0002` enables the `vector` extension, backfills lexical projections,
and marks pre-existing memories as pending vector indexing.

## Embedding provider

The core does not depend on a vendor SDK. Supply an implementation of
`EmbeddingProvider` when bootstrapping:

```python
from fluxmem.application.ports import EmbeddingProviderError
from fluxmem.bootstrap import bootstrap
from fluxmem.domain import EMBEDDING_DIMENSIONS, Embedding


class MyEmbeddingProvider:
    dimensions = EMBEDDING_DIMENSIONS

    def embed(self, *, text: str) -> Embedding:
        try:
            values = call_embedding_service(text)
        except TemporaryProviderFailure as error:
            raise EmbeddingProviderError(str(error)) from error
        return Embedding(
            values=tuple(values),
            model="my-embedding-model-v1",
        )


services = bootstrap(
    database_url="postgresql+psycopg://fluxmem:fluxmem_dev@localhost:5433/fluxmem",
    embedding_provider=MyEmbeddingProvider(),
)
```

Only stored vectors with the same `embedding.model` identifier as the query
vector participate in dense retrieval. Changing dimensions requires a schema
and projection-version migration.

Pending projections can be repaired in bounded batches:

```python
summary = services.reindex_pending_memories.execute(limit=100)
```

`reindex_pending_memories` is `None` when FluxMem is bootstrapped without an
embedding provider.

## Retrieval feedback

Every `MemoryPack` contains a persisted `query_id`. A feedback request must use
that identifier, and every reported memory must have appeared in that query's
persisted candidate set. Repeating feedback for the same `(query_id,
memory_id)` is idempotent; a later query may reinforce the same memory again.

## Tuning

`HybridRetrievalSettings` exposes bounded query length, candidate-pool size,
RRF weights, the RRF constant, and the lifecycle relevance floor. The defaults
are intentionally conservative and should be tuned against a representative
retrieval corpus rather than raw vector and text-search scores.

## Tests

Fast application, SQL-compilation, indexing-fallback, provenance, and lifecycle
tests run without an external database:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The executable hybrid-search test requires PostgreSQL with pgvector. It creates
an isolated transactional schema and rolls it back after the test:

```bash
FLUXMEM_TEST_DATABASE_URL="postgresql+psycopg://.../fluxmem_test" \
  .venv/bin/python -m unittest tests.test_hybrid_retrieval_postgres -v
```

Entity expansion, conflict adjacency, and historical-query modes are outside
the first hybrid-retrieval slice.
