# FluxMem

FluxMem is a Python/PostgreSQL memory service with lifecycle-aware hybrid
retrieval. Canonical memories and all retrieval projections remain in one
PostgreSQL transaction.

When configured with a structured model provider, FluxMem can also coordinate
the complete conversational lifecycle: turn retrieval, answer generation,
usage feedback, atomic memory extraction, batched reconciliation,
potential-conflict detection, and memory storage.

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

## Potential-conflict adjacency

Potential conflicts are stored as undirected memory pairs in canonical UUID
order. An edge means only that both memories may make incompatible claims; it
does not choose a winner, suppress either memory, or change lifecycle state.

Conflict classification remains upstream of `StoreMemory`. The caller may
submit proposals selected from the persisted turn retrieval:

```python
from fluxmem.domain import ConflictProposal

services.store_memory.execute(
    user_id=user_id,
    memory=new_memory,
    write_context_query_id=turn_memory_packs.expanded.query_id,
    conflict_proposals=(
        ConflictProposal(
            neighbor_memory_id=existing_memory_id,
            confidence=0.9,
        ),
    ),
)
```

FluxMem accepts an edge only when the query belongs to the same user and origin
session, has type `answering`, and actually returned the proposed neighbor. Self,
duplicate, fabricated, and out-of-context proposals are discarded without a
second similarity search.

Each turn performs one hybrid search for normally ranked seed memories and then
exactly one adjacency lookup from those fixed seeds. The resulting
`TurnMemoryPacks` contains two views over the same `query_id`: `seeds` and
`expanded`. Eligible neighbors are appended fairly within per-seed and global
budgets, and expanded neighbors never initiate another hop. The expanded view
is the persisted candidate set used for answering and reconciliation; the seed
view is supplied to extraction. The defaults are three conflicts per seed and
ten expanded memories globally.

## LLM integration

The application layer owns prompts, structured write-task schemas, validation,
and repair. Provider adapters stream plain answer text and execute structured
memory requests. This keeps model output outside authorization and persistence
boundaries: a model may refer to a returned prompt-local memory reference, but
it cannot choose a user, session, query, or arbitrary conflict target.

Prompts are plain-text resources under
`src/fluxmem/application/llm/prompts/`:

- `answer.txt`
- `memory_extraction.txt`
- `memory_reconciliation.txt`
- `lifecycle_evaluation.txt`
- `repair.txt`

They are loaded for each task invocation, so an editable installation picks up
prompt changes without rebuilding the package. Keep the
`{validation_error}` placeholder exactly once in `repair.txt`; the loader
validates it before issuing a repair request. Prompt files are included in built
wheels through the setuptools package-data configuration.

Canonical UUIDs are never supplied as model-facing metadata. Each prompt builds
its own deterministic 1-based reference map and renders compact fields such as
`message_ref: 1` and `memory_ref: 2`. Structured outputs return those integer
references through fields such as `source_message_ref`,
`equivalent_memory_ref`, and `neighbor_memory_ref`.
FluxMem validates each reference against that exact prompt and converts it back
to the canonical UUID before reconciliation or persistence. Answer streaming
does not request model-generated attribution; memories supplied to the answer
are deterministically recorded as context-included feedback.

Answer- and extraction-facing memory records contain only content,
`valid_from`, `valid_to`, and `created_at`; reconciliation adds a prompt-local
`memory_ref` because that task must select records. Ranked seeds and their
one-hop conflict expansions are otherwise rendered identically. Pack headers,
conflict edges, scores, ranks, retention, retrieval reasons, and scope metadata
remain inside the application. Explicit validity boundaries take precedence;
`created_at` is observation-time evidence, not proof that a newer claim is
correct.

Extraction and reconciliation receive bounded recent conversation. Write-path
history defaults to six messages and 4,000 characters, with 1,000 characters
per message; write-task memory input defaults to 8,000 characters. Extraction
also receives the seed view as interpretation context, but only explicitly
listed target messages may be returned as a memory's source. All extracted
candidates are reconciled together in one structured request against the
expanded view.

Prompts request semantic judgment only. In particular, lifecycle evaluation
returns `importance`, `reason_codes`, and `confidence`; deterministic policy
derives tier, initial retention, and decay class. Extraction is instructed not
to repeat the same candidate, and deterministic normalized-text deduplication
provides a second guard. Stored-memory equivalence and potential conflicts stay
in batched reconciliation; lifecycle scoring and persistence policy stay out of
extraction. Each prompt contains an exact representative input. Write-task
examples match their strict JSON schemas, while the answer example shows plain
streamed text.

Install the optional OpenAI adapter and configure task models explicitly:

```bash
.venv/bin/pip install -e '.[llm-openai]'
```

```python
import os

from fluxmem.adapters.llm import OpenAIResponsesProvider
from fluxmem.application.llm import LLMIntegrationSettings, LLMTaskSettings
from fluxmem.bootstrap import bootstrap


task = LLMTaskSettings(
    model=os.environ["FLUXMEM_LLM_MODEL"],
    timeout_seconds=30,
    maximum_output_tokens=2048,
)
services = bootstrap(
    database_url=os.environ["DATABASE_URL"],
    structured_model_provider=OpenAIResponsesProvider(),
    llm_settings=LLMIntegrationSettings(
        answer=task,
        extraction=task,
        reconciliation=task,
        lifecycle=task,
        # Set False while observing extraction and reconciliation without writes.
        enable_memory_writes=False,
    ),
)

assert services.process_turn is not None
answer_stream = services.process_turn.execute(
    user_id=user_id,
    message=user_message,
    agent_id="answering-agent",
)
for delta in answer_stream:
    send_delta_to_user(delta)

# Optional: observe feedback, extraction, reconciliation, and write outcomes.
result = answer_stream.wait_for_post_answer()
```

One turn executes in this order:

1. Persist the user message and perform one hybrid retrieval plus one-hop
   conflict expansion, producing seed and expanded views with one `query_id`.
2. Render the expanded view as uniform, bounded semantic records containing
   content and temporal metadata.
3. Stream answer text to the caller as the default behavior.
4. After the answer stream completes, persist the assistant message and submit
   feedback, extraction, reconciliation, and memory writes to a background
   executor.
5. Extract bounded atomic candidates from the target message using recent
   conversation and the unexpanded seed view as interpretation context.
6. Reconcile all candidates in one model request against the expanded view,
   returning `NONE` for equivalents or `ADD` with conflict references drawn
   only from the persisted turn retrieval.
7. Evaluate lifecycle semantics and atomically store each accepted memory.

Only answering retrieval and answer generation are on the response path.
`execute()` returns a `ConversationTurnStream`; consuming it forwards provider
deltas without waiting for the full response. Once exhausted,
`post_answer_future` exposes the background result, and
`wait_for_post_answer()` waits for it when needed. The explicit
`execute_and_wait()` helper preserves synchronous full-turn behavior for batch
jobs and tests. `FluxMemServices.close()` waits for submitted work before
disposing the database engine.

Feedback, extraction, reconciliation, and individual memory writes report
failures in `ConversationTurnResult` without discarding the persisted answer.
Structured write-task outputs get at most one validation-repair attempt.
Lifecycle provider failure falls back to the deterministic rule evaluator, and
external lifecycle calls run outside the authoritative PostgreSQL write
transaction.

Rollout controls in `LLMIntegrationSettings` can disable extraction, canonical
memory writes, conflict-edge writes, or LLM lifecycle evaluation independently.
Model and prompt metadata belong in provider logging or tracing; they are not
part of the canonical conflict-edge representation.

## Tuning

`HybridRetrievalSettings` exposes bounded query length, candidate-pool size,
RRF weights, the RRF constant, the lifecycle relevance floor, and conflict
expansion limits. The defaults are intentionally conservative and should be
tuned against a representative retrieval corpus rather than raw vector and
text-search scores.

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

The OpenAI adapter smoke test is also opt-in and performs one billed API call:

```bash
OPENAI_API_KEY="..." FLUXMEM_TEST_OPENAI_MODEL="..." \
  .venv/bin/python -m unittest tests.test_openai_llm_live -v
```

Entity expansion, automatic conflict resolution, and historical query modes
remain outside this slice.
