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
.venv/bin/pip install -e '.[local]'
cp .env.example .env
make db-setup
```

Put the OpenRouter key in the untracked `.env` file:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
```

`.env` and `.env.local` are ignored by Git. Do not put real keys in
`.env.example` or source code.

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

The local runtime sends every model request through OpenRouter:

- OpenRouter Chat Completions with `google/gemma-4-31b-it:free` for answering,
  extraction, reconciliation, and lifecycle evaluation.
- OpenRouter Embeddings with `openai/text-embedding-3-small` and 1536
  dimensions, matching the current pgvector schema.

The local extra uses the OpenAI-compatible Python SDK as an HTTP client, but
both adapters set its base URL to `https://openrouter.ai/api/v1`; no direct
OpenAI credential or endpoint is used by `bootstrap_from_env()`.

Copy `.env.example` to `.env`, add the OpenRouter key, and build the service
graph with one call:

```bash
.venv/bin/pip install -e '.[local]'
```

```python
from fluxmem.local import bootstrap_from_env


# Loads .env first; real process environment variables take precedence.
services = bootstrap_from_env()

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

# Provider-reported usage across answer, extraction, reconciliation, repairs,
# and per-memory lifecycle evaluation.
totals = result.llm_usage.token_totals
if totals is not None:
    print(totals.input_tokens, totals.output_tokens, totals.total_tokens)
    print(totals.cached_input_tokens, totals.cache_write_input_tokens)
```

`FLUXMEM_LLM_MODEL` changes the shared OpenRouter model. Independent
`FLUXMEM_ANSWER_MODEL`, `FLUXMEM_EXTRACTION_MODEL`,
`FLUXMEM_RECONCILIATION_MODEL`, and `FLUXMEM_LIFECYCLE_MODEL` values override it
per task. `FLUXMEM_EMBEDDING_MODEL` configures the embedding model routed by
OpenRouter; changing its vector space or dimensions requires a
database/projection migration.

Answering loads and renders up to 128 recent messages. Its default 64,000-token
input budget covers instructions, recent conversation, and retrieved memories;
configure these independently with `FLUXMEM_ANSWER_HISTORY_MESSAGES` and
`FLUXMEM_ANSWER_INPUT_TOKEN_BUDGET`. OpenRouter reports exact native input-token
usage only after a response, so preflight selection uses a portable UTF-8
estimate and the reported `input_tokens` remains authoritative. Extraction and
reconciliation retain their separate six-message/character budgets.

## Live workflow inspection

`ProcessConversationTurn.execute_text()` creates user-message IDs and timestamps
for callers. Pass `diagnostics=True` to retain an opt-in trace containing the
retrieval seed and conflict-expanded packs, exact model instructions and input,
canonical IDs actually included in each request, raw and validated model
outputs, repair attempts, token usage, feedback, and persistence outcomes.
Diagnostics contain private conversation data and are disabled by default.

After starting and migrating PostgreSQL, run one six-round LoCoMo-derived case
through real OpenRouter:

```bash
.venv/bin/python scripts/live_workflow_test.py
```

The script creates a fresh user and session, replays selected evidence and
distractor turns from one published case, asks its reference question, and
writes the complete per-round trace under the ignored `workflow-traces/`
directory. It is an integration smoke test rather than a canonical score over
the full LoCoMo conversation.

The chosen Gemma endpoint supports JSON output but does not advertise strict
JSON-Schema enforcement. The OpenRouter adapter therefore supplies the schema
to the model in compact form, requests JSON-object mode, validates the result
inside FluxMem, and uses the existing single repair attempt when needed. Keep
`FLUXMEM_OPENROUTER_STRICT_JSON_SCHEMA=false` for this model. A future model
that advertises strict structured outputs can opt in explicitly.

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

`ConversationTurnStream.llm_usage` exposes usage observed so far (normally the
answer call once streaming finishes). The completed `ConversationTurnResult`
contains the full `LLMUsageReport`. Every observed attempt records its task,
model, response ID, attempt number, and provider-reported token details. This
includes both the primary and repair calls when structured validation requires
a retry. Failed or cancelled attempts remain visible with unknown token usage
when the provider supplies no terminal usage metadata.
`usage_complete` is false when a provider omits usage for any observed call;
`token_totals` sums only reported calls and is `None` when none were reported.
Usage is operational telemetry and is not stored in canonical memory records.

Answer generation is shaped for prompt-prefix caching independently of the
background memory tasks. The provider receives one stable text block per
bounded conversation record, with cache breakpoints after each record, followed
by the changing retrieved-memory evidence as an unmarked suffix. A hashed key
scoped to the user, session, and answer prompt improves cache routing without
exposing canonical IDs to the model. The OpenRouter adapter renders the marked
blocks as `cache_control` breakpoints and maps the hashed key to OpenRouter's
sticky `session_id`; this preserves the best available routing and prefix shape
without enabling whole-response caching. Provider-side cache support is still
model dependent, so Gemma may legitimately report zero cached tokens. Set
`FLUXMEM_OPENROUTER_PROMPT_CACHING=false` if its routed endpoint rejects or
ignores explicit breakpoints. The separate OpenAI Responses adapter retains its
GPT-5.6+ explicit-cache behavior for callers that use it directly.

The reusable prefix becomes eligible only after it reaches the provider's
minimum cacheable length. Bounded-history rollover or edits to prior rendered
messages naturally start a new prefix. The default answer history now aligns
its 128-message bound with turn retrieval and shares a 64,000-token estimated
input budget with instructions and retrieved memories. No rolling summary or
provider-side context compression is enabled. Inspect
`cached_input_tokens` and `cache_write_input_tokens` by `LLMTaskKind.ANSWER` to
measure real hit rates.

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

The OpenRouter smoke test is opt-in. It loads `.env`, calls the configured free
answer model once, and makes one small billed embedding request:

```bash
FLUXMEM_TEST_OPENROUTER=1 \
  .venv/bin/python -m unittest tests.test_openrouter_live -v
```

Entity expansion, automatic conflict resolution, and historical query modes
remain outside this slice.
