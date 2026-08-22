from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema

from fluxmem.adapters.postgres.models import (
    Base,
    MemoryLifecycleRow,
    MemoryRow,
    MessageRow,
    RetrievalCandidateRow,
    RetrievalQueryRow,
    SessionRow,
    UserRow,
)
from fluxmem.adapters.postgres.repositories.memory_indexes import (
    SqlAlchemyMemoryIndexRepository,
)
from fluxmem.adapters.postgres.repositories.conflicts import (
    SqlAlchemyConflictRepository,
)
from fluxmem.adapters.postgres.unit_of_work import SqlAlchemyUnitOfWork
from fluxmem.application.read.retrieval import (
    HybridRetrievalSettings,
    RetrievalForAnswering,
)
from fluxmem.domain.conflict import MemoryConflict
from fluxmem.domain.info_pack import MessagePack
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    Embedding,
    IndexStatus,
    MemoryIndex,
)


TEST_DATABASE_URL = os.getenv("FLUXMEM_TEST_DATABASE_URL")


class FixedEmbeddingProvider:
    dimensions = EMBEDDING_DIMENSIONS

    def embed(self, *, text: str) -> Embedding:
        return Embedding(
            values=(1.0, 0.0) + (0.0,) * (EMBEDDING_DIMENSIONS - 2),
            model="test-embedding-v1",
        )


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "set FLUXMEM_TEST_DATABASE_URL to run PostgreSQL retrieval tests",
)
class HybridRetrievalPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = create_engine(TEST_DATABASE_URL)
        cls.connection = cls.engine.connect()
        cls.transaction = cls.connection.begin()
        cls.connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        cls.schema = f"fluxmem_test_{uuid4().hex}"
        cls.connection.execute(CreateSchema(cls.schema))
        cls.connection.execute(
            text(f'SET LOCAL search_path TO "{cls.schema}", public')
        )
        Base.metadata.create_all(cls.connection, checkfirst=False)
        cls.session_factory = sessionmaker(
            bind=cls.connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.transaction.rollback()
        cls.connection.close()
        cls.engine.dispose()

    def setUp(self) -> None:
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.now = datetime(2026, 8, 21, 12, tzinfo=timezone.utc)
        with self.session_factory() as session:
            session.add(UserRow(user_id=self.user_id))
            session.add(
                SessionRow(
                    session_id=self.session_id,
                    user_id=self.user_id,
                )
            )
            session.commit()

        self.dense_only_id = self._add_memory(
            content="geology field notes",
            embedding=(1.0, 0.0),
        )
        self.lexical_only_id = self._add_memory(
            content="volcano recipe",
            embedding=None,
        )
        self.hybrid_id = self._add_memory(
            content="volcano geology preference",
            embedding=(0.8, 0.2),
        )

    def test_dense_and_lexical_candidates_are_fused_and_persisted(self) -> None:
        def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
            return SqlAlchemyUnitOfWork(self.session_factory)

        service = RetrievalForAnswering(
            unit_of_work_factory=unit_of_work_factory,
            embedding_provider=FixedEmbeddingProvider(),
        )
        message = Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id=None,
            content="volcano",
            created_at=self.now,
        )

        result = service.execute(
            message=message,
            session_history=MessagePack(
                user_id=self.user_id,
                session_id=self.session_id,
                messages=(),
            ),
            limit=3,
        )

        self.assertEqual(
            result.seeds.memories[0].memory.memory_id,
            self.hybrid_id,
        )
        by_id = {
            item.memory.memory_id: item for item in result.seeds.memories
        }
        self.assertEqual(
            set(by_id),
            {self.dense_only_id, self.lexical_only_id, self.hybrid_id},
        )
        self.assertIn("dense", by_id[self.dense_only_id].retrieval_reasons)
        self.assertNotIn(
            "lexical",
            by_id[self.dense_only_id].retrieval_reasons,
        )
        self.assertIn(
            "lexical",
            by_id[self.lexical_only_id].retrieval_reasons,
        )
        self.assertNotIn(
            "dense",
            by_id[self.lexical_only_id].retrieval_reasons,
        )
        self.assertIn("dense", by_id[self.hybrid_id].retrieval_reasons)
        self.assertIn("lexical", by_id[self.hybrid_id].retrieval_reasons)

        with self.session_factory() as session:
            query = session.get(RetrievalQueryRow, result.expanded.query_id)
            candidates = session.scalars(
                select(RetrievalCandidateRow).where(
                    RetrievalCandidateRow.query_id
                    == result.expanded.query_id
                )
            ).all()
        self.assertIsNotNone(query)
        self.assertEqual(len(candidates), 3)

    def test_conflict_expansion_stops_after_one_hop(self) -> None:
        neighbor_id = self._add_memory(
            content="unrelated prior claim",
            embedding=None,
        )
        second_hop_id = self._add_memory(
            content="another unrelated claim",
            embedding=None,
        )
        self._add_conflict(self.hybrid_id, neighbor_id)
        self._add_conflict(neighbor_id, second_hop_id)

        def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
            return SqlAlchemyUnitOfWork(self.session_factory)

        service = RetrievalForAnswering(
            unit_of_work_factory=unit_of_work_factory,
            embedding_provider=FixedEmbeddingProvider(),
            settings=HybridRetrievalSettings(
                maximum_conflicts_per_seed=3,
                maximum_conflict_expansions=3,
            ),
        )
        message = Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id=None,
            content="volcano",
            created_at=self.now,
        )

        result = service.execute(
            message=message,
            session_history=MessagePack(
                user_id=self.user_id,
                session_id=self.session_id,
                messages=(),
            ),
            limit=1,
        )

        seed_ids = tuple(
            item.memory.memory_id for item in result.seeds.memories
        )
        returned_ids = tuple(
            item.memory.memory_id for item in result.expanded.memories
        )
        self.assertEqual(seed_ids, (self.hybrid_id,))
        self.assertEqual(returned_ids, (self.hybrid_id, neighbor_id))
        self.assertNotIn(second_hop_id, returned_ids)
        self.assertEqual(len(result.expanded.conflicts), 1)
        self.assertFalse(result.expanded.conflict_expansion_truncated)
        with self.session_factory() as session:
            candidate_ids = set(
                session.scalars(
                    select(RetrievalCandidateRow.memory_id).where(
                        RetrievalCandidateRow.query_id
                        == result.expanded.query_id
                    )
                )
            )
        self.assertEqual(candidate_ids, {self.hybrid_id, neighbor_id})

    def _add_conflict(self, first_id, second_id) -> None:
        with self.session_factory() as session:
            SqlAlchemyConflictRepository(session).add(
                conflict=MemoryConflict.between(
                    memory_id=first_id,
                    neighbor_memory_id=second_id,
                    confidence=0.9,
                    created_at=self.now,
                )
            )
            session.commit()

    def _add_memory(
        self,
        *,
        content: str,
        embedding: tuple[float, float] | None,
    ):
        message_id = uuid4()
        memory_id = uuid4()
        with self.session_factory() as session:
            session.add(
                MessageRow(
                    message_id=message_id,
                    session_id=self.session_id,
                    role="user",
                    agent_id=None,
                    content=content,
                    created_at=self.now,
                )
            )
            session.add(
                MemoryRow(
                    memory_id=memory_id,
                    message_id=message_id,
                    content=content,
                    created_at=self.now,
                    valid_from=None,
                    valid_to=None,
                    session_applicability=None,
                )
            )
            session.add(
                MemoryLifecycleRow(
                    memory_id=memory_id,
                    tier="long_term",
                    status="active",
                    importance=0.8,
                    decay_class="slow",
                    retention_snapshot=0.8,
                    retention_anchor=self.now,
                    reinforcement_count=0,
                    use_count=0,
                    last_used_at=None,
                    decision_source="rules_fallback",
                    updated_at=self.now,
                )
            )
            session.flush()
            vector = None
            if embedding is not None:
                vector = Embedding(
                    values=embedding
                    + (0.0,) * (EMBEDDING_DIMENSIONS - len(embedding)),
                    model="test-embedding-v1",
                )
            SqlAlchemyMemoryIndexRepository(session).add(
                index=MemoryIndex(
                    memory_id=memory_id,
                    text=content,
                    embedding=vector,
                    status=(
                        IndexStatus.READY
                        if vector is not None
                        else IndexStatus.PENDING
                    ),
                    indexed_at=self.now,
                )
            )
            session.commit()
        return memory_id


if __name__ == "__main__":
    unittest.main()
