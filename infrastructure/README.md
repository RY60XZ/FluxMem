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

The LoCoMo harness follows Mem0's default evaluation path: it judges categories
1-4 with a binary LLM judge and excludes adversarial category 5. The judge uses
the configured answer model, so answer generation and evaluation stay on the
same model unless that shared setting changes.
