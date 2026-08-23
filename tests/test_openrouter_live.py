from __future__ import annotations

import json
import os
import unittest

from fluxmem.adapters.embeddings import OpenRouterEmbeddingProvider
from fluxmem.adapters.llm import OpenRouterChatCompletionsProvider
from fluxmem.domain import EMBEDDING_DIMENSIONS
from fluxmem.local import (
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_MODEL,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@unittest.skipUnless(
    os.environ.get("FLUXMEM_TEST_OPENROUTER") == "1"
    and os.environ.get("OPENROUTER_API_KEY"),
    "set FLUXMEM_TEST_OPENROUTER=1 and OPENROUTER_API_KEY to run live tests",
)
class OpenRouterLiveIntegrationTests(unittest.TestCase):
    def test_structured_generation_and_embedding_round_trip(self) -> None:
        model_provider = OpenRouterChatCompletionsProvider()
        response = model_provider.generate(
            model=DEFAULT_OPENROUTER_MODEL,
            instructions="Return JSON.",
            input_text="Set ok to true.",
            schema_name="fluxmem_live_smoke_test",
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
            timeout_seconds=60,
            maximum_output_tokens=64,
        )
        self.assertEqual(json.loads(response.output_text), {"ok": True})
        self.assertIsNotNone(response.usage)

        embedding_provider = OpenRouterEmbeddingProvider(
            model=DEFAULT_OPENROUTER_EMBEDDING_MODEL,
        )
        embedding = embedding_provider.embed(text="The user prefers tea.")
        self.assertEqual(len(embedding.values), EMBEDDING_DIMENSIONS)
        self.assertEqual(embedding.model, DEFAULT_OPENROUTER_EMBEDDING_MODEL)


if __name__ == "__main__":
    unittest.main()
