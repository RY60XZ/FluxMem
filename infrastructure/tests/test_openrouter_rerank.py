from __future__ import annotations

import unittest

from fluxmem_infrastructure.providers.openrouter_rerank import (
    OpenRouterRerankProvider,
)


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "id": "gen-rerank-1",
            "model": "voyageai/rerank-2.5",
            "provider": "Voyage AI",
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ],
            "usage": {"total_tokens": 24, "search_units": 1},
        }


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, path, **values):
        self.calls.append({"path": path, **values})
        return _Response()


class OpenRouterRerankProviderTests(unittest.TestCase):
    def test_calls_native_rerank_endpoint_and_parses_scores(self) -> None:
        client = _Client()
        provider = OpenRouterRerankProvider(client=client)

        response = provider.rerank(
            model="voyageai/rerank-2.5",
            query="Where does Alice live?",
            documents=("Alice likes tea.", "Alice lives in Toronto."),
            top_n=2,
            timeout_seconds=10,
        )

        self.assertEqual(client.calls[0]["path"], "rerank")
        self.assertEqual(
            client.calls[0]["json"]["model"],
            "voyageai/rerank-2.5",
        )
        self.assertEqual(response.results[0].index, 1)
        self.assertEqual(response.results[0].relevance_score, 0.9)
        self.assertEqual(response.total_tokens, 24)
        self.assertEqual(response.search_units, 1)


if __name__ == "__main__":
    unittest.main()
