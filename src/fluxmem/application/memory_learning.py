from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from fluxmem.application.diagnostics import MemoryDiagnosticsCollector
from fluxmem.application.lifecycle import LifecycleAssigner
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.ports.lifecycle import LifecycleEvaluationInput
from fluxmem.application.ports.llm import MemoryExtractor
from fluxmem.application.read.retrieval import HybridMemoryRetriever
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.write.store_memory import StoreMemory
from fluxmem.application.write.store_message import StoreMessage
from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.llm import (
    MessageIngestionResult,
    MemoryWriteOutcome,
    MemoryWriteStatus,
    ProposedMemory,
    proposed_memory_to_memory,
)
from fluxmem.domain.lifecycle import MemoryLifecycle
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


@dataclass(frozen=True, slots=True)
class MemoryLearningResult:
    memory_outcomes: tuple[MemoryWriteOutcome, ...]
    errors: tuple[str, ...]


class MemoryLearning:
    """Extract and persist memories from messages."""

    def __init__(
        self,
        *,
        store_memory: StoreMemory,
        memory_extractor: MemoryExtractor,
        lifecycle_assigner: LifecycleAssigner,
        id_factory: Callable[[], UUID] = uuid4,
        enable_memory_extraction: bool = True,
        enable_memory_writes: bool = True,
    ) -> None:
        self._store_memory = store_memory
        self._memory_extractor = memory_extractor
        self._lifecycle_assigner = lifecycle_assigner
        self._id_factory = id_factory
        self._enable_memory_extraction = enable_memory_extraction
        self._enable_memory_writes = enable_memory_writes

    def execute(
        self,
        *,
        user_id: UUID,
        session_history: MessagePack,
        target_messages: tuple[Message, ...],
        extraction_pack: MemoryPack,
        usage_collector: ModelUsageCollector,
        diagnostics_collector: MemoryDiagnosticsCollector | None,
    ) -> MemoryLearningResult:
        if not target_messages:
            return MemoryLearningResult(memory_outcomes=(), errors=())
        session_id = session_history.session_id
        if any(message.session_id != session_id for message in target_messages):
            raise ValueError("learning targets must share the history session")
        if extraction_pack.session_id != session_id:
            raise ValueError("extraction memory must share the history session")

        errors: list[str] = []
        if self._enable_memory_extraction:
            try:
                candidates = self._memory_extractor.extract(
                    target_messages=target_messages,
                    session_history=session_history,
                    memory_pack=extraction_pack,
                    usage_recorder=usage_collector,
                    diagnostics_recorder=diagnostics_collector,
                )
            except Exception as error:
                errors.append(error_text("extraction", error))
                candidates = ()
        else:
            candidates = ()

        outcomes = self._store_candidates(
            user_id=user_id,
            session_id=session_id,
            candidate_sources={
                message.message_id: message
                for message in target_messages
            },
            candidates=candidates,
            query_id=extraction_pack.query_id,
            usage_collector=usage_collector,
            diagnostics_collector=diagnostics_collector,
        )
        return MemoryLearningResult(
            memory_outcomes=outcomes,
            errors=tuple(errors),
        )

    def _store_candidates(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        candidate_sources: dict[UUID, Message],
        candidates: tuple[ProposedMemory, ...],
        query_id: UUID,
        usage_collector: ModelUsageCollector,
        diagnostics_collector: MemoryDiagnosticsCollector | None,
    ) -> tuple[MemoryWriteOutcome, ...]:
        if not candidates:
            return ()

        valid_candidates: list[ProposedMemory] = []
        valid_indices: list[int] = []
        outcomes_by_index: dict[int, MemoryWriteOutcome] = {}
        for index, candidate in enumerate(candidates):
            error = self._candidate_validation_error(
                candidate=candidate,
                session_id=session_id,
                candidate_source_ids=set(candidate_sources),
            )
            if error is None:
                valid_candidates.append(candidate)
                valid_indices.append(index)
            else:
                outcomes_by_index[index] = MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.FAILED,
                    write_context_query_id=query_id,
                    error=error,
                )

        additions: list[tuple[int, ProposedMemory, Memory]] = []
        for index, candidate in zip(
            valid_indices,
            valid_candidates,
        ):
            if not self._enable_memory_writes:
                outcomes_by_index[index] = MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.DRY_RUN,
                    write_context_query_id=query_id,
                )
                continue
            source = candidate_sources[candidate.source_message_id]
            additions.append(
                (
                    index,
                    candidate,
                    proposed_memory_to_memory(
                        proposal=candidate,
                        memory_id=self._id_factory(),
                        created_at=source.created_at,
                    ),
                )
            )

        if not additions:
            return tuple(
                outcomes_by_index[index] for index in range(len(candidates))
            )
        initialized_at = max(memory.created_at for _, _, memory in additions)
        lifecycles = self._lifecycle_assigner.assign(
            items=tuple(
                LifecycleEvaluationInput(
                    memory=memory,
                    source_role=(
                        candidate_sources[candidate.source_message_id].role
                    ),
                )
                for _, candidate, memory in additions
            ),
            evaluated_at=initialized_at,
            usage_recorder=usage_collector,
            diagnostics_recorder=diagnostics_collector,
        )
        for (index, candidate, memory), lifecycle in zip(
            additions,
            lifecycles,
        ):
            outcomes_by_index[index] = self._store_candidate(
                user_id=user_id,
                candidate=candidate,
                memory=memory,
                lifecycle=lifecycle,
                initialized_at=initialized_at,
                query_id=query_id,
            )
        return tuple(
            outcomes_by_index[index] for index in range(len(candidates))
        )

    @staticmethod
    def _candidate_validation_error(
        *,
        candidate: ProposedMemory,
        session_id: UUID,
        candidate_source_ids: set[UUID],
    ) -> str | None:
        if candidate.source_message_id not in candidate_source_ids:
            return "candidate source is absent from extraction targets"
        if candidate.session_applicability not in (None, session_id):
            return "candidate applicability references a different session"
        return None

    def _store_candidate(
        self,
        *,
        user_id: UUID,
        candidate: ProposedMemory,
        memory: Memory,
        lifecycle: MemoryLifecycle,
        initialized_at: datetime,
        query_id: UUID,
    ) -> MemoryWriteOutcome:
        try:
            memory_id = self._store_memory.execute(
                user_id=user_id,
                memory=memory,
                lifecycle=lifecycle,
                indexed_at=initialized_at,
            )
            return MemoryWriteOutcome(
                candidate=candidate,
                status=MemoryWriteStatus.STORED,
                write_context_query_id=query_id,
                memory_id=memory_id,
            )
        except Exception as error:
            return MemoryWriteOutcome(
                candidate=candidate,
                status=MemoryWriteStatus.FAILED,
                write_context_query_id=query_id,
                error=error_text("candidate", error),
            )


class LearnFromMessages:
    """Persist transcript messages and learn memories from them."""

    def __init__(
        self,
        *,
        get_session_history: GetSessionHistory,
        retriever: HybridMemoryRetriever,
        store_message: StoreMessage,
        memory_learning: MemoryLearning,
        maximum_batch_messages: int,
        history_limit: int = 128,
        retrieval_limit: int = 10,
    ) -> None:
        if maximum_batch_messages < 1:
            raise ValueError("learning batch limit must be positive")
        if history_limit < 1 or retrieval_limit < 1:
            raise ValueError("learning context limits must be positive")
        self._get_session_history = get_session_history
        self._retriever = retriever
        self._store_message = store_message
        self._memory_learning = memory_learning
        self._maximum_batch_messages = maximum_batch_messages
        self._history_limit = history_limit
        self._retrieval_limit = retrieval_limit

    def execute(
        self,
        *,
        user_id: UUID,
        messages: Sequence[Message],
        diagnostics: bool = False,
    ) -> MessageIngestionResult:
        imported = tuple(messages)
        if not imported:
            return MessageIngestionResult(messages=())
        session_id = imported[0].session_id
        if any(message.session_id != session_id for message in imported):
            raise ValueError("imported messages must share one session")
        if len({message.message_id for message in imported}) != len(imported):
            raise ValueError("imported message IDs must be unique")
        if any(not message.content.strip() for message in imported):
            raise ValueError("imported message content cannot be blank")

        usage_collector = ModelUsageCollector()
        diagnostics_collector = (
            MemoryDiagnosticsCollector() if diagnostics else None
        )
        outcomes: list[MemoryWriteOutcome] = []
        errors: list[str] = []
        for start in range(0, len(imported), self._maximum_batch_messages):
            batch = imported[start : start + self._maximum_batch_messages]
            for message in batch:
                self._store_message.execute(user_id=user_id, message=message)
            history = self._get_session_history.execute(
                user_id=user_id,
                session_id=session_id,
                limit=self._history_limit,
            )
            retrieval = self._retriever.execute(
                message=batch[-1],
                session_history=history,
                limit=self._retrieval_limit,
            )
            if diagnostics_collector is not None:
                diagnostics_collector.record_retrieval(retrieval)
            learned = self._memory_learning.execute(
                user_id=user_id,
                session_history=history,
                target_messages=batch,
                extraction_pack=retrieval,
                usage_collector=usage_collector,
                diagnostics_collector=diagnostics_collector,
            )
            outcomes.extend(learned.memory_outcomes)
            errors.extend(learned.errors)

        diagnostic_snapshot = None
        if diagnostics_collector is not None:
            diagnostics_collector.complete(
                memory_outcomes=tuple(outcomes),
                errors=tuple(errors),
            )
            diagnostic_snapshot = diagnostics_collector.snapshot()
        return MessageIngestionResult(
            messages=imported,
            memory_outcomes=tuple(outcomes),
            errors=tuple(errors),
            llm_usage=usage_collector.snapshot(),
            diagnostics=diagnostic_snapshot,
        )


def error_text(stage: str, error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{stage}: {detail}"
