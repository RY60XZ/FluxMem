from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID

from fluxmem.application.read.retrieval import (
    HybridRetrievalSettings,
    _expand_conflict_neighbors,
)
from fluxmem.domain.conflict import (
    ConflictNeighbor,
    ConflictProposal,
    MemoryConflict,
)
from fluxmem.domain.info_pack import RetrievedMemory
from fluxmem.domain.memory import Memory


class ConflictDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 21, tzinfo=timezone.utc)

    def test_conflict_is_canonicalized_and_confidence_is_validated(self) -> None:
        conflict = MemoryConflict.between(
            memory_id=UUID(int=20),
            neighbor_memory_id=UUID(int=10),
            confidence=0.8,
            created_at=self.now,
        )

        self.assertEqual(conflict.memory_a_id, UUID(int=10))
        self.assertEqual(conflict.memory_b_id, UUID(int=20))
        with self.assertRaises(ValueError):
            ConflictProposal(neighbor_memory_id=UUID(int=1), confidence=1.1)
        with self.assertRaises(ValueError):
            MemoryConflict.between(
                memory_id=UUID(int=1),
                neighbor_memory_id=UUID(int=1),
                confidence=None,
                created_at=self.now,
            )

    def test_expansion_uses_fair_budget_and_reports_truncation(self) -> None:
        seed_a = self._retrieved(memory_id=UUID(int=1), rank=1, score=1.0)
        seed_b = self._retrieved(memory_id=UUID(int=2), rank=2, score=0.9)
        neighbors = (
            self._neighbor(seed_id=1, neighbor_id=10, confidence=0.9),
            self._neighbor(seed_id=1, neighbor_id=11, confidence=0.8),
            self._neighbor(seed_id=2, neighbor_id=12, confidence=0.7),
        )

        expansion = _expand_conflict_neighbors(
            seeds=(seed_a, seed_b),
            neighbors=neighbors,
            settings=HybridRetrievalSettings(
                maximum_conflicts_per_seed=2,
                maximum_conflict_expansions=2,
            ),
        )

        self.assertEqual(
            tuple(item.memory.memory_id for item in expansion.memories),
            (UUID(int=1), UUID(int=2), UUID(int=10), UUID(int=12)),
        )
        self.assertTrue(expansion.truncated)
        self.assertEqual(len(expansion.conflicts), 2)
        self.assertEqual(
            expansion.memories[2].retrieval_reasons,
            ("conflict", "lifecycle"),
        )

    def test_existing_seed_is_not_duplicated(self) -> None:
        seed_a = self._retrieved(memory_id=UUID(int=1), rank=1, score=1.0)
        seed_b = self._retrieved(memory_id=UUID(int=2), rank=2, score=0.9)
        neighbor = self._neighbor(
            seed_id=1,
            neighbor_id=2,
            confidence=0.9,
        )

        expansion = _expand_conflict_neighbors(
            seeds=(seed_a, seed_b),
            neighbors=(neighbor,),
            settings=HybridRetrievalSettings(),
        )

        self.assertEqual(expansion.memories, (seed_a, seed_b))
        self.assertEqual(len(expansion.conflicts), 1)
        self.assertFalse(expansion.truncated)

    def _retrieved(
        self,
        *,
        memory_id: UUID,
        rank: int,
        score: float,
    ) -> RetrievedMemory:
        return RetrievedMemory(
            memory=self._memory(memory_id),
            rank=rank,
            score=score,
            retention=0.8,
            retrieval_reasons=("dense", "lifecycle"),
        )

    def _neighbor(
        self,
        *,
        seed_id: int,
        neighbor_id: int,
        confidence: float,
    ) -> ConflictNeighbor:
        conflict = MemoryConflict.between(
            memory_id=UUID(int=seed_id),
            neighbor_memory_id=UUID(int=neighbor_id),
            confidence=confidence,
            created_at=self.now,
        )
        return ConflictNeighbor(
            seed_memory_id=UUID(int=seed_id),
            memory=self._memory(UUID(int=neighbor_id)),
            conflict=conflict,
            retention=0.7,
        )

    def _memory(self, memory_id: UUID) -> Memory:
        return Memory(
            memory_id=memory_id,
            message_id=UUID(int=memory_id.int + 100),
            content=f"memory {memory_id.int}",
            created_at=self.now,
        )


if __name__ == "__main__":
    unittest.main()
