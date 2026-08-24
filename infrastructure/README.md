# FluxMem infrastructure

This optional package composes the independent `fluxmem` memory layer with:

- a streaming answering agent;
- OpenRouter and OpenAI model adapters;
- `.env`-based local runtime configuration; and
- the LoCoMo full-run and single-conversation harness.

Install it from the repository root:

```bash
.venv/bin/pip install -e './infrastructure[locomo]'
```

The answering agent depends only on FluxMem's public operations: retrieve
context, read history, record usage, ingest user messages, and store generated
assistant messages. Other agent systems can implement the same composition
without depending on this package.
