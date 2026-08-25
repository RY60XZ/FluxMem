# FluxMem infrastructure

This optional package composes the independent `fluxmem` memory layer with:

- a streaming answering agent;
- OpenRouter and OpenAI model adapters;
- `.env`-based local runtime configuration; and
- the LoCoMo full-run and single-conversation harness.

Install it from the repository root:

```bash
.venv/bin/pip install -e ./infrastructure
```

The answering agent depends only on FluxMem's public operations: retrieve
context, read history, record usage, ingest user messages, and store generated
assistant messages. Other agent systems can implement the same composition
without depending on this package.

The LoCoMo harness judges categories 1-4 with a binary LLM judge and excludes
adversarial category 5. The judge defaults to the answer model and can be set
independently with `FLUXMEM_JUDGE_MODEL`.

Run one conversation into a new or empty output directory:

```bash
.venv/bin/python -B -m fluxmem_infrastructure.locomo.cli run \
  --conversation conv-30 \
  --output runs/locomo-conv-30
```

Run the same conversation as a full-context baseline, storing the raw
transcript without memory learning and supplying every source turn to each
question:

```bash
.venv/bin/python -B -m fluxmem_infrastructure.locomo.cli run \
  --conversation conv-30 \
  --context-mode full \
  --output runs/locomo-conv-30-full-context
```

The default `--context-mode memory` answers from retrieved FluxMem memories
without raw transcript history. Full-context mode answers from transcript
history without retrieval, extraction, reconciliation, or lifecycle calls.

The harness atomically updates `checkpoint.json` after every completed source
turn and scored question. If the process is interrupted, repeat the same
selection, dataset, model configuration, and output directory with `--resume`:

```bash
.venv/bin/python -B -m fluxmem_infrastructure.locomo.cli run \
  --conversation conv-30 \
  --output runs/locomo-conv-30 \
  --resume
```

Resume validates the run ID, dataset, selected conversations, categories, and
models before continuing. It also prevents resuming a memory run as a
full-context run or vice versa. The operation that was in flight at interruption
may be repeated. Session and message writes are idempotent; memory
reconciliation handles any memory writes that completed before the checkpoint
was updated.

Progress can be inspected while the run is active:

```bash
jq '.conversations["conv-30"] | {
  completed_turn_count,
  completed_question_count: (.question_results | length),
  complete
}' runs/locomo-conv-30/checkpoint.json
```
