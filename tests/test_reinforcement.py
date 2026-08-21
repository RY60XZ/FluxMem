from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from fluxmem.adapters.postgres.models import (
    Base,
    MemoryLifecycleRow,
    MemoryRow,
    MemoryUsageRow,
    MessageRow,
    RetrievalCandidateRow,
    RetrievalQueryRow,
    SessionRow,
    UserRow,
)
from fluxmem.adapters.postgres.unit_of_work import SqlAlchemyUnitOfWork
from fluxmem.application.errors import (
    InvalidMemoryFeedbackError,
    RetrievalNotFoundError,
)
from fluxmem.application.write.reinforcement import ReinforceMemory
from fluxmem.domain.info_pack import FeedbackPack, MemoryUsage, UsageType
from fluxmem.domain.lifecycle import DecayClass, Tier


@dataclass
class MutableClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


class ReinforcementIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.memory_id = uuid4()
        self.started_at = datetime(2026, 1, 1, 12)
        self.clock = MutableClock(self.started_at)

        with self.session_factory() as session:
            message_id = uuid4()
            session.add_all(
                [
                    UserRow(user_id=self.user_id),
                    SessionRow(
                        session_id=self.session_id,
                        user_id=self.user_id,
                    ),
                    MessageRow(
                        message_id=message_id,
                        session_id=self.session_id,
                        role="user",
                        agent_id=None,
                        content="Remember this",
                        created_at=self.started_at,
                    ),
                    MemoryRow(
                        memory_id=self.memory_id,
                        message_id=message_id,
                        content="Remember this",
                        created_at=self.started_at,
                        valid_from=None,
                        valid_to=None,
                        session_applicability=None,
                    ),
                    MemoryLifecycleRow(
                        memory_id=self.memory_id,
                        tier=Tier.WORKING.value,
                        status="active",
                        importance=0.5,
                        decay_class=DecayClass.FAST.value,
                        retention_snapshot=0.5,
                        retention_anchor=self.started_at,
                        reinforcement_count=0,
                        use_count=0,
                        last_used_at=None,
                        decision_source="rules_fallback",
                        updated_at=self.started_at,
                    ),
                ]
            )
            session.commit()

        def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
            return SqlAlchemyUnitOfWork(self.session_factory)

        self.reinforce = ReinforceMemory(
            unit_of_work_factory=unit_of_work_factory,
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_every_new_query_reinforces_but_promotion_needs_usage_days(self) -> None:
        first = self._attributed_use()
        second = self._attributed_use()

        self.assertAlmostEqual(first.importance, 0.6)
        self.assertAlmostEqual(second.importance, 0.68)
        self.assertEqual(second.reinforcement_count, 2)
        self.assertEqual(second.tier, Tier.WORKING)

        self.clock.current += timedelta(days=1)
        third = self._attributed_use()
        fourth = self._attributed_use()

        self.assertEqual(third.tier, Tier.SHORT_TERM)
        self.assertEqual(third.decay_class, DecayClass.STANDARD)
        self.assertAlmostEqual(fourth.importance, 0.7952)
        self.assertEqual(fourth.tier, Tier.SHORT_TERM)

        self.clock.current += timedelta(days=1)
        fifth = self._attributed_use()

        self.assertAlmostEqual(fifth.importance, 0.83616)
        self.assertEqual(fifth.reinforcement_count, 5)
        self.assertEqual(fifth.use_count, 5)
        self.assertEqual(fifth.tier, Tier.LONG_TERM)
        self.assertEqual(fifth.decay_class, DecayClass.SLOW)

    def test_query_retry_updates_signal_without_reinforcing_twice(self) -> None:
        query_id = uuid4()
        first = self._execute(
            query_id,
            MemoryUsage(
                memory_id=self.memory_id,
                usage_type=UsageType.CONTEXT_INCLUDED,
                rank=2,
            ),
        )[0]
        retry = self._execute(
            query_id,
            MemoryUsage(
                memory_id=self.memory_id,
                usage_type=UsageType.MODEL_ATTRIBUTED,
                rank=1,
                contribution=0.8,
            ),
        )[0]

        self.assertEqual(first.reinforcement_count, 1)
        self.assertEqual(retry.reinforcement_count, 1)
        self.assertAlmostEqual(retry.importance, first.importance)
        with self.session_factory() as session:
            usage = session.scalar(select(MemoryUsageRow))
            self.assertIsNotNone(usage)
            self.assertEqual(usage.query_id, query_id)
            self.assertEqual(
                usage.usage_type,
                UsageType.MODEL_ATTRIBUTED.value,
            )
            self.assertEqual(usage.rank, 1)

    def test_feedback_rejects_memory_not_returned_by_query(self) -> None:
        query_id = uuid4()
        self._record_query(query_id)

        with self.assertRaises(InvalidMemoryFeedbackError):
            self.reinforce.execute(
                feedback_pack=FeedbackPack(
                    query_id=query_id,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    used_memories=(
                        MemoryUsage(
                            memory_id=uuid4(),
                            usage_type=UsageType.CONTEXT_INCLUDED,
                        ),
                    ),
                )
            )

    def test_feedback_rejects_unknown_query(self) -> None:
        with self.assertRaises(RetrievalNotFoundError):
            self.reinforce.execute(
                feedback_pack=FeedbackPack(
                    query_id=uuid4(),
                    user_id=self.user_id,
                    session_id=self.session_id,
                    used_memories=(),
                )
            )

    def _attributed_use(self):
        query_id = uuid4()
        return self._execute(
            query_id,
            MemoryUsage(
                memory_id=self.memory_id,
                usage_type=UsageType.MODEL_ATTRIBUTED,
            ),
        )[0]

    def _execute(
        self,
        query_id: UUID,
        *usages: MemoryUsage,
    ):
        self._record_query(query_id)
        return self.reinforce.execute(
            feedback_pack=FeedbackPack(
                query_id=query_id,
                user_id=self.user_id,
                session_id=self.session_id,
                used_memories=tuple(usages),
            )
        )

    def _record_query(self, query_id: UUID) -> None:
        with self.session_factory() as session:
            if session.get(RetrievalQueryRow, query_id) is None:
                session.add(
                    RetrievalQueryRow(
                        query_id=query_id,
                        session_id=self.session_id,
                        query_type="answering",
                        created_at=self.clock.current,
                    )
                )
                session.add(
                    RetrievalCandidateRow(
                        query_id=query_id,
                        memory_id=self.memory_id,
                        rank=1,
                        score=1.0,
                    )
                )
                session.commit()


if __name__ == "__main__":
    unittest.main()
