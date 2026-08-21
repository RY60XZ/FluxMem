from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from fluxmem.adapters.postgres.repositories.conflicts import (
    SqlAlchemyConflictRepository,
)
from fluxmem.adapters.postgres.repositories.memories import (
    SqlAlchemyMemoryRepository,
)
from fluxmem.domain.conflict import MemoryConflict
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    Embedding,
    MemorySearchQuery,
)


class _EmptyResult:
    def all(self):
        return []


class _RecordingSession:
    def __init__(self) -> None:
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return _EmptyResult()


class HybridQuerySqlTests(unittest.TestCase):
    def test_postgres_query_contains_indexable_legs_and_rrf_fusion(self) -> None:
        session = _RecordingSession()
        repository = SqlAlchemyMemoryRepository(session)

        result = repository.search(
            query=MemorySearchQuery(
                user_id=uuid4(),
                session_id=uuid4(),
                text="volcano preference",
                embedding=Embedding(
                    values=(1.0,)
                    + (0.0,) * (EMBEDDING_DIMENSIONS - 1),
                    model="test-embedding-v1",
                ),
                as_of=datetime(2026, 8, 21, tzinfo=timezone.utc),
                limit=10,
                candidate_limit=50,
                rrf_k=60,
                dense_weight=1.0,
                lexical_weight=1.0,
                relevance_floor=0.6,
            )
        )

        self.assertEqual(result, ())
        sql = str(
            session.statement.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": False},
            )
        )
        self.assertIn("memory_indexes.search_text @@ to_tsquery", sql)
        self.assertIn("memory_indexes.embedding <=>", sql)
        self.assertIn("dense_nearest", sql)
        self.assertIn("UNION ALL", sql)
        self.assertIn("fused_candidates", sql)
        self.assertIn("memory_lifecycle.status", sql)
        self.assertIn("memories.session_applicability", sql)

    def test_conflict_expansion_is_one_scoped_adjacency_query(self) -> None:
        session = _RecordingSession()
        repository = SqlAlchemyConflictRepository(session)

        result = repository.expand(
            seed_memory_ids=(uuid4(), uuid4()),
            user_id=uuid4(),
            session_id=uuid4(),
            as_of=datetime(2026, 8, 21, tzinfo=timezone.utc),
        )

        self.assertEqual(result, ())
        sql = str(
            session.statement.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": False},
            )
        )
        self.assertIn("FROM memory_conflicts", sql)
        self.assertIn("memory_conflicts.memory_a_id IN", sql)
        self.assertIn("memory_conflicts.memory_b_id IN", sql)
        self.assertIn("memory_lifecycle.status", sql)
        self.assertIn("memories.session_applicability", sql)

    def test_conflict_insert_is_canonical_and_idempotent(self) -> None:
        session = _RecordingSession()
        repository = SqlAlchemyConflictRepository(session)
        now = datetime(2026, 8, 21, tzinfo=timezone.utc)

        repository.add(
            conflict=MemoryConflict.between(
                memory_id=uuid4(),
                neighbor_memory_id=uuid4(),
                confidence=0.8,
                created_at=now,
            )
        )

        sql = str(
            session.statement.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": False},
            )
        )
        self.assertIn("INSERT INTO memory_conflicts", sql)
        self.assertIn("ON CONFLICT", sql)
        self.assertIn("DO NOTHING", sql)


if __name__ == "__main__":
    unittest.main()
