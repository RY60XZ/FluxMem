from __future__ import annotations

import json
import os
import unittest

from fluxmem.adapters.llm import OpenAIResponsesProvider


@unittest.skipUnless(
    os.environ.get("FLUXMEM_TEST_OPENAI_MODEL"),
    "set FLUXMEM_TEST_OPENAI_MODEL to run the live OpenAI adapter test",
)
class OpenAILiveIntegrationTests(unittest.TestCase):
    def test_strict_structured_response_round_trip(self) -> None:
        provider = OpenAIResponsesProvider()

        response = provider.generate(
            model=os.environ["FLUXMEM_TEST_OPENAI_MODEL"],
            instructions="Return the requested test result.",
            input_text="Set ok to true.",
            schema_name="fluxmem_live_smoke_test",
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
            timeout_seconds=30,
            maximum_output_tokens=64,
        )

        self.assertEqual(json.loads(response.output_text), {"ok": True})


if __name__ == "__main__":
    unittest.main()
