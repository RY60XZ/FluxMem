from __future__ import annotations

import unittest
from types import SimpleNamespace

from fluxmem.adapters.embeddings import OpenRouterEmbeddingProvider
from fluxmem.adapters.llm import OpenRouterChatCompletionsProvider
from fluxmem.application.ports import EmbeddingProviderError, ModelInputTextBlock
from fluxmem.domain import EMBEDDING_DIMENSIONS


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=12,
        total_tokens=132,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=64,
            cache_write_tokens=32,
        ),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )


class FakeChatStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self) -> None:
        self.closed = True


class FakeChatCompletions:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}
        self.stream: FakeChatStream | None = None

    def create(self, **values):
        self.request = values
        if values.get("stream"):
            self.stream = FakeChatStream(
                [
                    SimpleNamespace(
                        id="generation-2",
                        model=values["model"],
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="Hello, ")
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        id="generation-2",
                        model=values["model"],
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="world.")
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        id="generation-2",
                        model=values["model"],
                        choices=[],
                        usage=_usage(),
                    ),
                ]
            )
            return self.stream
        return SimpleNamespace(
            id="generation-1",
            model=values["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok":true}')
                )
            ],
            usage=_usage(),
        )


class FakeOpenRouterClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())
        self.timeout: float | None = None

    def with_options(self, *, timeout: float):
        self.timeout = timeout
        return self


class OpenRouterProviderTests(unittest.TestCase):
    def test_uses_json_object_mode_and_supplies_schema_to_gemma(self) -> None:
        client = FakeOpenRouterClient()
        provider = OpenRouterChatCompletionsProvider(client=client)

        response = provider.generate(
            model="google/gemma-4-31b-it:free",
            instructions="Return JSON.",
            input_text="Set ok to true.",
            schema_name="test_schema",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
            timeout_seconds=12,
            maximum_output_tokens=128,
        )

        request = client.chat.completions.request
        self.assertEqual(client.timeout, 12)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(
            request["extra_body"],
            {"provider": {"require_parameters": True}},
        )
        self.assertIn('"required":["ok"]', request["messages"][0]["content"])
        self.assertEqual(response.output_text, '{"ok":true}')
        self.assertEqual(response.response_id, "generation-1")
        self.assertEqual(response.usage.cached_input_tokens, 64)
        self.assertEqual(response.usage.cache_write_input_tokens, 32)
        self.assertEqual(response.usage.reasoning_output_tokens, 4)

    def test_streams_text_with_cache_breakpoints_and_sticky_session(self) -> None:
        client = FakeOpenRouterClient()
        provider = OpenRouterChatCompletionsProvider(client=client)
        blocks = (
            ModelInputTextBlock("header\n"),
            ModelInputTextBlock("message\n", cache_breakpoint=True),
            ModelInputTextBlock("memory"),
        )

        stream = provider.stream_text(
            model="google/gemma-4-31b-it:free",
            instructions="Answer.",
            input_text="".join(block.text for block in blocks),
            input_text_blocks=blocks,
            prompt_cache_key="fluxmem-answer:test",
            timeout_seconds=12,
            maximum_output_tokens=128,
        )

        self.assertEqual(tuple(stream), ("Hello, ", "world."))
        request = client.chat.completions.request
        content = request["messages"][1]["content"]
        self.assertNotIn("cache_control", content[0])
        self.assertEqual(
            content[1]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertNotIn("cache_control", content[2])
        self.assertEqual(
            request["extra_body"],
            {"session_id": "fluxmem-answer:test"},
        )
        self.assertEqual(stream.response_id, "generation-2")
        self.assertEqual(stream.usage.total_tokens, 132)
        self.assertTrue(client.chat.completions.stream.closed)


class FakeEmbeddings:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.request: dict[str, object] = {}

    def create(self, **values):
        self.request = values
        if self.fails:
            raise RuntimeError("unavailable")
        vector = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
        return SimpleNamespace(
            model=values["model"],
            data=[SimpleNamespace(embedding=vector)],
        )


class FakeEmbeddingClient:
    def __init__(self, *, fails: bool = False) -> None:
        self.embeddings = FakeEmbeddings(fails=fails)
        self.timeout: float | None = None

    def with_options(self, *, timeout: float):
        self.timeout = timeout
        return self


class OpenRouterEmbeddingProviderTests(unittest.TestCase):
    def test_requests_fixed_fluxmem_dimensions(self) -> None:
        client = FakeEmbeddingClient()
        provider = OpenRouterEmbeddingProvider(
            client=client,
            model="openai/text-embedding-3-small",
            timeout_seconds=15,
        )

        embedding = provider.embed(text="The user prefers tea.")

        self.assertEqual(client.timeout, 15)
        self.assertEqual(
            client.embeddings.request,
            {
                "model": "openai/text-embedding-3-small",
                "input": "The user prefers tea.",
                "dimensions": EMBEDDING_DIMENSIONS,
                "encoding_format": "float",
            },
        )
        self.assertEqual(len(embedding.values), EMBEDDING_DIMENSIONS)
        self.assertEqual(embedding.model, "openai/text-embedding-3-small")

    def test_maps_failures_to_lexical_fallback_error(self) -> None:
        provider = OpenRouterEmbeddingProvider(
            client=FakeEmbeddingClient(fails=True)
        )

        with self.assertRaises(EmbeddingProviderError):
            provider.embed(text="The user prefers tea.")


if __name__ == "__main__":
    unittest.main()
