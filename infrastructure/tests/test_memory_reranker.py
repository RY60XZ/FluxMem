from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem import (
    Memory,
    MemoryPack,
    Message,
    RetrievalCandidateDiagnostics,
    RetrievedMemory,
)
from fluxmem_infrastructure.answering import (
    MemoryRerankerSettings,
    ModelMemoryReranker,
    RerankProviderResponse,
    RerankProviderResult,
)


class _Provider:
    def __init__(self, results: tuple[RerankProviderResult, ...]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def rerank(self, **values) -> RerankProviderResponse:
        self.calls.append(values)
        return RerankProviderResponse(
            results=self.results,
            model="voyageai/rerank-2.5",
            response_id="reranker-response",
            provider="Voyage AI",
            total_tokens=10,
        )


class MemoryRerankerTests(unittest.TestCase):
    def test_reorders_every_candidate_and_preserves_first_stage_scores(
        self,
    ) -> None:
        query, pack = _query_and_pack()
        provider = _Provider(
            (
                RerankProviderResult(index=1, relevance_score=0.95),
                RerankProviderResult(index=0, relevance_score=0.10),
            )
        )
        reranker = ModelMemoryReranker(
            provider=provider,
            settings=MemoryRerankerSettings(
                model="voyageai/rerank-2.5",
                maximum_candidates=50,
            ),
        )

        outcome = reranker.rerank(query=query, memory_pack=pack)

        self.assertEqual(
            tuple(item.memory.content for item in outcome.memory_pack.memories),
            ("Alice lives in Toronto.", "Alice likes tea."),
        )
        self.assertEqual(
            tuple(item.rank for item in outcome.memory_pack.memories),
            (1, 2),
        )
        self.assertEqual(
            outcome.memory_pack.memories[0].diagnostics.initial_rank,
            2,
        )
        self.assertIn(
            "reranker",
            outcome.memory_pack.memories[0].retrieval_reasons,
        )
        self.assertEqual(outcome.diagnostics.status, "completed")
        self.assertTrue(outcome.diagnostics.attempted)
        self.assertEqual(outcome.diagnostics.provider, "Voyage AI")
        self.assertEqual(
            outcome.diagnostics.candidates[0].relevance_score,
            0.95,
        )
        request = provider.calls[0]
        self.assertEqual(request["query"], "Where does Alice live?")
        self.assertEqual(request["top_n"], 2)
        documents = request["documents"]
        self.assertEqual(len(documents), 2)
        self.assertEqual(
            json.loads(documents[1])["content"],
            "Alice lives in Toronto.",
        )

    def test_invalid_ranking_falls_back_and_records_provider_diagnostics(
        self,
    ) -> None:
        query, pack = _query_and_pack()
        provider = _Provider(
            (
                RerankProviderResult(index=0, relevance_score=0.8),
                RerankProviderResult(index=0, relevance_score=0.7),
            )
        )
        reranker = ModelMemoryReranker(
            provider=provider,
            settings=MemoryRerankerSettings(
                model="voyageai/rerank-2.5"
            ),
        )

        outcome = reranker.rerank(query=query, memory_pack=pack)

        self.assertEqual(outcome.memory_pack, pack)
        self.assertEqual(outcome.diagnostics.status, "failed")
        self.assertTrue(outcome.diagnostics.attempted)
        self.assertEqual(
            outcome.diagnostics.response_id,
            "reranker-response",
        )
        self.assertIsNotNone(outcome.diagnostics.usage)
        self.assertIn("every candidate exactly once", outcome.diagnostics.error)


def _query_and_pack() -> tuple[Message, MemoryPack]:
    user_id = uuid4()
    session_id = uuid4()
    created_at = datetime(2023, 5, 8, tzinfo=timezone.utc)
    query = Message(
        message_id=uuid4(),
        session_id=session_id,
        role="user",
        agent_id=None,
        content="Where does Alice live?",
        created_at=created_at,
    )
    contents = ("Alice likes tea.", "Alice lives in Toronto.")
    memories = tuple(
        RetrievedMemory(
            memory=Memory(
                memory_id=uuid4(),
                message_id=uuid4(),
                content=content,
                created_at=created_at,
            ),
            rank=index,
            score=1.0 / index,
            retention=0.9,
            retrieval_reasons=("dense",),
            diagnostics=RetrievalCandidateDiagnostics(
                initial_rank=index,
                fusion_score=1.0 / index,
                lifecycle_multiplier=0.9,
                dense_rank=index,
                dense_similarity=1.0 / index,
            ),
        )
        for index, content in enumerate(contents, start=1)
    )
    return query, MemoryPack(
        query_id=uuid4(),
        user_id=user_id,
        session_id=session_id,
        memories=memories,
    )


if __name__ == "__main__":
    unittest.main()
