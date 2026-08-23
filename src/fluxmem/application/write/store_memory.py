from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from fluxmem.application.errors import (
    InvalidSessionApplicabilityError,
    MessageNotFoundError,
)
from fluxmem.application.lifecycle import (
    LifecyclePolicyExecutor,
    RuleLifecycleEvaluator,
)
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from fluxmem.application.ports.lifecycle import LifecycleEvaluator
from fluxmem.application.ports.llm import ModelUsageRecorder
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.conflict import ConflictProposal, MemoryConflict
from fluxmem.domain.lifecycle import MemoryLifecycle
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    IndexStatus,
    MemoryIndex,
    QueryType,
)


class StoreMemory:
    """Persist one already-reconciled memory within its ownership boundaries."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        lifecycle_evaluator: LifecycleEvaluator | None = None,
        lifecycle_policy: LifecyclePolicyExecutor | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._lifecycle_evaluator = lifecycle_evaluator or RuleLifecycleEvaluator()
        self._fallback_evaluator = RuleLifecycleEvaluator()
        self._lifecycle_policy = lifecycle_policy or LifecyclePolicyExecutor()
        self._embedding_provider = embedding_provider
        self._clock = clock or SystemClock()
        if (
            embedding_provider is not None
            and embedding_provider.dimensions != EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                "embedding provider dimensions do not match the database schema"
            )

    def execute(
        self,
        *,
        user_id: UUID,
        memory: Memory,
        write_context_query_id: UUID | None = None,
        conflict_proposals: tuple[ConflictProposal, ...] = (),
        usage_recorder: ModelUsageRecorder | None = None,
    ) -> UUID:
        # Resolve immutable evidence in a short read transaction. Provider calls
        # happen only after this context has closed.
        with self._unit_of_work_factory() as unit_of_work:
            source_message = self._validated_source_message(
                unit_of_work=unit_of_work,
                user_id=user_id,
                memory=memory,
            )

        embedding = None
        if self._embedding_provider is not None:
            try:
                embedding = self._embedding_provider.embed(text=memory.content)
            except EmbeddingProviderError:
                embedding = None
            if embedding is not None and len(embedding.values) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    "embedding provider returned a vector with invalid dimensions"
                )

        initialized_at = self._clock.now()
        lifecycle = self._initial_lifecycle(
            memory=memory,
            source_role=source_message.role,
            initialized_at=initialized_at,
            usage_recorder=usage_recorder,
        )
        memory_index = MemoryIndex(
            memory_id=memory.memory_id,
            text=memory.content,
            embedding=embedding,
            status=(
                IndexStatus.READY
                if embedding is not None
                else IndexStatus.PENDING
            ),
            indexed_at=initialized_at,
        )

        with self._unit_of_work_factory() as unit_of_work:
            message = self._validated_source_message(
                unit_of_work=unit_of_work,
                user_id=user_id,
                memory=memory,
            )
            conflicts = self._validated_conflicts(
                unit_of_work=unit_of_work,
                user_id=user_id,
                session_id=message.session_id,
                memory=memory,
                query_id=write_context_query_id,
                proposals=conflict_proposals,
                created_at=initialized_at,
            )

            unit_of_work.memories.add(memory=memory)
            unit_of_work.flush()
            unit_of_work.memory_indexes.add(index=memory_index)
            unit_of_work.lifecycles.add(lifecycle=lifecycle)
            for conflict in conflicts:
                unit_of_work.conflicts.add(conflict=conflict)
            unit_of_work.commit()

        return memory.memory_id

    @staticmethod
    def _validated_source_message(
        *,
        unit_of_work: UnitOfWork,
        user_id: UUID,
        memory: Memory,
    ) -> Message:
        message = unit_of_work.messages.get(
            message_id=memory.message_id,
            user_id=user_id,
        )
        if message is None:
            raise MessageNotFoundError(
                "origin message does not belong to the requested user"
            )
        if memory.session_applicability not in (None, message.session_id):
            raise InvalidSessionApplicabilityError(
                "session-limited memory must use its origin message's session"
            )
        return message

    @staticmethod
    def _validated_conflicts(
        *,
        unit_of_work: UnitOfWork,
        user_id: UUID,
        session_id: UUID,
        memory: Memory,
        query_id: UUID | None,
        proposals: tuple[ConflictProposal, ...],
        created_at: datetime,
    ) -> tuple[MemoryConflict, ...]:
        if not proposals or query_id is None:
            return ()

        candidate_ids = unit_of_work.retrievals.candidate_ids_for_query(
            query_id=query_id,
            user_id=user_id,
            session_id=session_id,
            query_type=QueryType.ANSWERING,
        )
        if candidate_ids is None:
            return ()

        eligible_ids = set(candidate_ids)
        seen_ids: set[UUID] = set()
        conflicts: list[MemoryConflict] = []
        for proposal in proposals:
            neighbor_id = proposal.neighbor_memory_id
            if (
                neighbor_id == memory.memory_id
                or neighbor_id in seen_ids
                or neighbor_id not in eligible_ids
            ):
                continue
            seen_ids.add(neighbor_id)
            conflicts.append(
                MemoryConflict.between(
                    memory_id=memory.memory_id,
                    neighbor_memory_id=neighbor_id,
                    confidence=proposal.confidence,
                    created_at=created_at,
                )
            )
        return tuple(conflicts)

    def _initial_lifecycle(
        self,
        *,
        memory: Memory,
        source_role: str,
        initialized_at: datetime,
        usage_recorder: ModelUsageRecorder | None,
    ) -> MemoryLifecycle:
        """Use the rule evaluator when a configured semantic evaluator fails."""

        try:
            decision = self._lifecycle_evaluator.evaluate(
                memory=memory,
                source_role=source_role,
                evaluated_at=initialized_at,
                usage_recorder=usage_recorder,
            )
            return self._lifecycle_policy.initialize(
                memory=memory,
                decision=decision,
                initialized_at=initialized_at,
            )
        except Exception:
            if isinstance(self._lifecycle_evaluator, RuleLifecycleEvaluator):
                raise
            decision = self._fallback_evaluator.evaluate(
                memory=memory,
                source_role=source_role,
                evaluated_at=initialized_at,
                usage_recorder=usage_recorder,
            )
            return self._lifecycle_policy.initialize(
                memory=memory,
                decision=decision,
                initialized_at=initialized_at,
            )
