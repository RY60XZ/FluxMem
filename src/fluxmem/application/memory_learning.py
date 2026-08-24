from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from fluxmem.application.diagnostics import MemoryDiagnosticsCollector
from fluxmem.application.lifecycle import LifecycleAssigner
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.lifecycle import LifecycleEvaluationInput
from fluxmem.application.ports.llm import MemoryExtractor, MemoryReconciler
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
    ReconciliationAction,
    ReconciliationDecision,
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
    """Extract, reconcile, and persist memories from existing messages."""

    def __init__(
        self,
        *,
        store_memory: StoreMemory,
        memory_extractor: MemoryExtractor,
        memory_reconciler: MemoryReconciler,
        lifecycle_assigner: LifecycleAssigner,
        id_factory: Callable[[], UUID] = uuid4,
        clock: Clock | None = None,
        enable_memory_extraction: bool = True,
        enable_memory_writes: bool = True,
        enable_conflict_detection: bool = True,
    ) -> None:
        self._store_memory = store_memory
        self._memory_extractor = memory_extractor
        self._memory_reconciler = memory_reconciler
        self._lifecycle_assigner = lifecycle_assigner
        self._id_factory = id_factory
        self._clock = clock or SystemClock()
        self._enable_memory_extraction = enable_memory_extraction
        self._enable_memory_writes = enable_memory_writes
        self._enable_conflict_detection = enable_conflict_detection

    def execute(
        self,
        *,
        user_id: UUID,
        session_history: MessagePack,
        target_messages: tuple[Message, ...],
        evidence_messages: tuple[Message, ...],
        extraction_pack: MemoryPack,
        reconciliation_pack: MemoryPack,
        usage_collector: ModelUsageCollector,
        diagnostics_collector: MemoryDiagnosticsCollector | None,
    ) -> MemoryLearningResult:
        if not target_messages:
            return MemoryLearningResult(memory_outcomes=(), errors=())
        session_id = session_history.session_id
        if any(message.session_id != session_id for message in target_messages):
            raise ValueError("learning targets must share the history session")
        if any(message.session_id != session_id for message in evidence_messages):
            raise ValueError("learning evidence must share the history session")
        if extraction_pack.session_id != session_id:
            raise ValueError("extraction memory must share the history session")
        if reconciliation_pack.session_id != session_id:
            raise ValueError("reconciliation memory must share the history session")

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

        outcomes = self._reconcile_and_store_candidates(
            user_id=user_id,
            session_id=session_id,
            session_history=session_history,
            evidence_messages=evidence_messages,
            candidate_sources={
                message.message_id: message
                for message in target_messages
            },
            candidates=candidates,
            memory_pack=reconciliation_pack,
            errors=errors,
            usage_collector=usage_collector,
            diagnostics_collector=diagnostics_collector,
        )
        return MemoryLearningResult(
            memory_outcomes=outcomes,
            errors=tuple(errors),
        )

    def _reconcile_and_store_candidates(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        session_history: MessagePack,
        evidence_messages: tuple[Message, ...],
        candidate_sources: dict[UUID, Message],
        candidates: tuple[ProposedMemory, ...],
        memory_pack: MemoryPack,
        errors: list[str],
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
                    write_context_query_id=memory_pack.query_id,
                    error=error,
                )

        decisions: tuple[ReconciliationDecision, ...]
        if valid_candidates:
            try:
                decisions = self._memory_reconciler.reconcile(
                    candidates=tuple(valid_candidates),
                    memory_pack=memory_pack,
                    evidence_messages=evidence_messages,
                    session_history=session_history,
                    usage_recorder=usage_collector,
                    diagnostics_recorder=diagnostics_collector,
                )
                if len(decisions) != len(valid_candidates):
                    raise ValueError(
                        "reconciliation must decide every valid candidate"
                    )
            except Exception as error:
                detail = error_text("reconciliation", error)
                errors.append(detail)
                decisions = ()
                for index, candidate in zip(valid_indices, valid_candidates):
                    outcomes_by_index[index] = MemoryWriteOutcome(
                        candidate=candidate,
                        status=MemoryWriteStatus.FAILED,
                        write_context_query_id=memory_pack.query_id,
                        error=detail,
                    )
        else:
            decisions = ()

        additions: list[
            tuple[int, ProposedMemory, ReconciliationDecision, Memory]
        ] = []
        allowed_ids = {
            retrieved.memory.memory_id for retrieved in memory_pack.memories
        }
        for index, candidate, decision in zip(
            valid_indices,
            valid_candidates,
            decisions,
        ):
            try:
                self._validate_decision(
                    decision=decision,
                    allowed_ids=allowed_ids,
                )
            except (TypeError, ValueError) as error:
                outcomes_by_index[index] = MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.FAILED,
                    write_context_query_id=memory_pack.query_id,
                    error=error_text("candidate", error),
                )
                continue
            if decision.action is ReconciliationAction.NONE:
                outcomes_by_index[index] = MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.EQUIVALENT,
                    write_context_query_id=memory_pack.query_id,
                    equivalent_memory_id=decision.equivalent_memory_id,
                )
                continue
            if not self._enable_memory_writes:
                outcomes_by_index[index] = MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.DRY_RUN,
                    write_context_query_id=memory_pack.query_id,
                )
                continue
            source = candidate_sources[candidate.source_message_id]
            additions.append(
                (
                    index,
                    candidate,
                    decision,
                    proposed_memory_to_memory(
                        proposal=candidate,
                        memory_id=self._id_factory(),
                        created_at=source.created_at,
                    ),
                )
            )

        initialized_at = self._clock.now()
        lifecycles = self._lifecycle_assigner.assign(
            items=tuple(
                LifecycleEvaluationInput(
                    memory=memory,
                    source_role=(
                        candidate_sources[candidate.source_message_id].role
                    ),
                )
                for _, candidate, _, memory in additions
            ),
            evaluated_at=initialized_at,
            usage_recorder=usage_collector,
            diagnostics_recorder=diagnostics_collector,
        )
        for (index, candidate, decision, memory), lifecycle in zip(
            additions,
            lifecycles,
        ):
            outcomes_by_index[index] = self._store_candidate(
                user_id=user_id,
                candidate=candidate,
                decision=decision,
                memory=memory,
                lifecycle=lifecycle,
                initialized_at=initialized_at,
                query_id=memory_pack.query_id,
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

    @staticmethod
    def _validate_decision(
        *,
        decision: ReconciliationDecision,
        allowed_ids: set[UUID],
    ) -> None:
        if decision.action is ReconciliationAction.NONE:
            if decision.equivalent_memory_id not in allowed_ids:
                raise ValueError(
                    "equivalent memory is absent from persisted context"
                )
            return
        proposed_neighbor_ids = [
            proposal.neighbor_memory_id
            for proposal in decision.conflict_proposals
        ]
        if len(set(proposed_neighbor_ids)) != len(proposed_neighbor_ids):
            raise ValueError("conflict proposals contain duplicate IDs")
        if not set(proposed_neighbor_ids).issubset(allowed_ids):
            raise ValueError(
                "conflict proposal is absent from persisted context"
            )

    def _store_candidate(
        self,
        *,
        user_id: UUID,
        candidate: ProposedMemory,
        decision: ReconciliationDecision,
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
                write_context_query_id=query_id,
                conflict_proposals=(
                    decision.conflict_proposals
                    if self._enable_conflict_detection
                    else ()
                ),
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
                evidence_messages=batch,
                extraction_pack=retrieval.seeds,
                reconciliation_pack=retrieval.expanded,
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
