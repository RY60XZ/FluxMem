from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import Lock
from uuid import UUID, uuid4

from fluxmem.application.diagnostics import TurnDiagnosticsCollector
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.llm import (
    AnswerGenerator,
    MemoryExtractor,
    MemoryReconciler,
)
from fluxmem.application.read.retrieval import RetrievalForAnswering
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.write.reinforcement import ReinforceMemory
from fluxmem.application.write.store_memory import StoreMemory
from fluxmem.application.write.store_message import StoreMessage
from fluxmem.domain.info_pack import (
    FeedbackPack,
    MemoryPack,
    MemoryUsage,
    MessagePack,
    UsageType,
)
from fluxmem.domain.llm import (
    ConversationTurnDiagnostics,
    ConversationTurnResult,
    LLMUsageReport,
    MemoryWriteOutcome,
    MemoryWriteStatus,
    ProposedMemory,
    ReconciliationAction,
    ReconciliationDecision,
    proposed_memory_to_memory,
)
from fluxmem.domain.message import Message


class ConversationTurnStream(Iterator[str]):
    """Answer deltas whose completed stream starts post-answer work."""

    def __init__(
        self,
        *,
        chunks: Iterable[str],
        finalize: Callable[
            [str], tuple[Message, Future[ConversationTurnResult]]
        ],
        usage_collector: ModelUsageCollector,
        diagnostics_collector: TurnDiagnosticsCollector | None,
    ) -> None:
        self._chunks = iter(chunks)
        self._finalize = finalize
        self._usage_collector = usage_collector
        self._diagnostics_collector = diagnostics_collector
        self._parts: list[str] = []
        self._finished = False
        self._answer: Message | None = None
        self._post_answer_future: Future[ConversationTurnResult] | None = None

    def __iter__(self) -> ConversationTurnStream:
        return self

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        while True:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                self._finished = True
                content = "".join(self._parts)
                if not content.strip():
                    raise ValueError("generated answer cannot be blank")
                self._answer, self._post_answer_future = self._finalize(content)
                raise
            except BaseException:
                self.close()
                raise
            if not isinstance(chunk, str):
                self.close()
                raise TypeError("answer stream chunks must be strings")
            if chunk:
                self._parts.append(chunk)
                return chunk

    def close(self) -> None:
        """Cancel an incomplete stream without persisting a partial answer."""

        if self._finished:
            return
        self._finished = True
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()

    @property
    def answer(self) -> Message:
        if self._answer is None:
            raise RuntimeError("answer is available after the stream completes")
        return self._answer

    @property
    def post_answer_future(self) -> Future[ConversationTurnResult]:
        if self._post_answer_future is None:
            raise RuntimeError(
                "post-answer work starts after the stream completes"
            )
        return self._post_answer_future

    def wait_for_post_answer(
        self, timeout: float | None = None
    ) -> ConversationTurnResult:
        return self.post_answer_future.result(timeout=timeout)

    @property
    def llm_usage(self) -> LLMUsageReport:
        """Usage observed so far; answer usage appears after exhaustion."""

        return self._usage_collector.snapshot()

    @property
    def diagnostics(self) -> ConversationTurnDiagnostics | None:
        if self._diagnostics_collector is None:
            return None
        return self._diagnostics_collector.snapshot()


class ProcessConversationTurn:
    """Coordinate answering first, then bounded post-answer memory writes."""

    def __init__(
        self,
        *,
        get_session_history: GetSessionHistory,
        retrieval_for_answering: RetrievalForAnswering,
        store_message: StoreMessage,
        store_memory: StoreMemory,
        reinforce_memory: ReinforceMemory,
        answer_generator: AnswerGenerator,
        memory_extractor: MemoryExtractor,
        memory_reconciler: MemoryReconciler,
        clock: Clock | None = None,
        id_factory: Callable[[], UUID] = uuid4,
        history_limit: int = 128,
        answering_limit: int = 10,
        enable_memory_extraction: bool = True,
        enable_memory_writes: bool = True,
        enable_conflict_detection: bool = True,
        post_answer_executor: Executor | None = None,
        post_answer_workers: int = 4,
    ) -> None:
        if history_limit < 1 or answering_limit < 1:
            raise ValueError("turn-processing limits must be positive")
        if post_answer_workers < 1:
            raise ValueError("post-answer worker count must be positive")
        self._get_session_history = get_session_history
        self._retrieval_for_answering = retrieval_for_answering
        self._store_message = store_message
        self._store_memory = store_memory
        self._reinforce_memory = reinforce_memory
        self._answer_generator = answer_generator
        self._memory_extractor = memory_extractor
        self._memory_reconciler = memory_reconciler
        self._clock = clock or SystemClock()
        self._id_factory = id_factory
        self._history_limit = history_limit
        self._answering_limit = answering_limit
        self._enable_memory_extraction = enable_memory_extraction
        self._enable_memory_writes = enable_memory_writes
        self._enable_conflict_detection = enable_conflict_detection
        self._owns_post_answer_executor = post_answer_executor is None
        self._post_answer_executor = post_answer_executor or ThreadPoolExecutor(
            max_workers=post_answer_workers,
            thread_name_prefix="fluxmem-post-answer",
        )
        self._close_lock = Lock()
        self._closed = False

    def execute(
        self,
        *,
        user_id: UUID,
        message: Message,
        agent_id: str | None = None,
        diagnostics: bool = False,
    ) -> ConversationTurnStream:
        """Return the answer as a stream; defer memory work until it ends."""

        if self._closed:
            raise RuntimeError("conversation turn processor is closed")
        if message.role != "user":
            raise ValueError("conversation turn input must have role 'user'")

        self._store_message.execute(user_id=user_id, message=message)
        history = self._get_session_history.execute(
            user_id=user_id,
            session_id=message.session_id,
            limit=self._history_limit,
        )
        turn_memory_packs = self._retrieval_for_answering.execute(
            message=message,
            session_history=history,
            limit=self._answering_limit,
        )
        diagnostics_collector = TurnDiagnosticsCollector() if diagnostics else None
        if diagnostics_collector is not None:
            diagnostics_collector.record_retrieval(turn_memory_packs)
        answering_pack = turn_memory_packs.expanded
        usage_collector = ModelUsageCollector()
        generated = self._answer_generator.stream(
            message=message,
            session_history=history,
            memory_pack=answering_pack,
            usage_recorder=usage_collector,
            diagnostics_recorder=diagnostics_collector,
        )
        if diagnostics_collector is not None:
            diagnostics_collector.record_answer_context(
                generated.context_memory_ids
            )

        return ConversationTurnStream(
            chunks=generated,
            usage_collector=usage_collector,
            diagnostics_collector=diagnostics_collector,
            finalize=lambda content: self._finish_answer(
                user_id=user_id,
                message=message,
                agent_id=agent_id,
                history=history,
                extraction_pack=turn_memory_packs.seeds,
                answering_pack=answering_pack,
                content=content,
                context_memory_ids=generated.context_memory_ids,
                usage_collector=usage_collector,
                diagnostics_collector=diagnostics_collector,
            ),
        )

    def execute_text(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        agent_id: str | None = None,
        diagnostics: bool = False,
    ) -> ConversationTurnStream:
        """Create user-message metadata and begin one streamed turn."""

        if not content.strip():
            raise ValueError("conversation turn content cannot be blank")
        return self.execute(
            user_id=user_id,
            message=Message(
                message_id=self._id_factory(),
                session_id=session_id,
                role="user",
                agent_id=None,
                content=content,
                created_at=self._clock.now(),
            ),
            agent_id=agent_id,
            diagnostics=diagnostics,
        )

    def execute_and_wait(
        self,
        *,
        user_id: UUID,
        message: Message,
        agent_id: str | None = None,
        timeout: float | None = None,
        diagnostics: bool = False,
    ) -> ConversationTurnResult:
        """Explicit compatibility path that collects and completes one turn."""

        stream = self.execute(
            user_id=user_id,
            message=message,
            agent_id=agent_id,
            diagnostics=diagnostics,
        )
        for _ in stream:
            pass
        return stream.wait_for_post_answer(timeout=timeout)

    def close(self) -> None:
        """Wait for submitted post-answer tasks when this service owns them."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_post_answer_executor:
                self._post_answer_executor.shutdown(wait=True)

    def _finish_answer(
        self,
        *,
        user_id: UUID,
        message: Message,
        agent_id: str | None,
        history: MessagePack,
        extraction_pack: MemoryPack,
        answering_pack: MemoryPack,
        content: str,
        context_memory_ids: tuple[UUID, ...],
        usage_collector: ModelUsageCollector,
        diagnostics_collector: TurnDiagnosticsCollector | None,
    ) -> tuple[Message, Future[ConversationTurnResult]]:
        answer = Message(
            message_id=self._id_factory(),
            session_id=message.session_id,
            role="assistant",
            agent_id=agent_id,
            content=content,
            created_at=self._clock.now(),
        )
        self._store_message.execute(user_id=user_id, message=answer)

        future = self._post_answer_executor.submit(
            self._complete_post_answer,
            user_id=user_id,
            message=message,
            answer=answer,
            history=history,
            extraction_pack=extraction_pack,
            answering_pack=answering_pack,
            context_memory_ids=context_memory_ids,
            usage_collector=usage_collector,
            diagnostics_collector=diagnostics_collector,
        )
        return answer, future

    def _complete_post_answer(
        self,
        *,
        user_id: UUID,
        message: Message,
        answer: Message,
        history: MessagePack,
        extraction_pack: MemoryPack,
        answering_pack: MemoryPack,
        context_memory_ids: tuple[UUID, ...],
        usage_collector: ModelUsageCollector,
        diagnostics_collector: TurnDiagnosticsCollector | None,
    ) -> ConversationTurnResult:
        errors: list[str] = []
        feedback_applied = self._apply_feedback(
            user_id=user_id,
            answer=answer,
            answering_pack=answering_pack,
            context_memory_ids=context_memory_ids,
            errors=errors,
        )

        if self._enable_memory_extraction:
            try:
                candidates = self._memory_extractor.extract(
                    target_messages=(message,),
                    session_history=history,
                    memory_pack=extraction_pack,
                    usage_recorder=usage_collector,
                    diagnostics_recorder=diagnostics_collector,
                )
            except Exception as error:
                errors.append(_error_text("extraction", error))
                candidates = ()
        else:
            candidates = ()

        outcomes = self._reconcile_and_store_candidates(
            user_id=user_id,
            session_id=message.session_id,
            session_history=history,
            evidence_messages=(message, answer),
            candidate_source_ids={message.message_id},
            candidates=candidates,
            memory_pack=answering_pack,
            errors=errors,
            usage_collector=usage_collector,
            diagnostics_collector=diagnostics_collector,
        )
        diagnostic_snapshot = None
        if diagnostics_collector is not None:
            diagnostics_collector.complete(
                feedback_applied=feedback_applied,
                memory_outcomes=outcomes,
                post_answer_errors=tuple(errors),
            )
            diagnostic_snapshot = diagnostics_collector.snapshot()
        return ConversationTurnResult(
            answer=answer,
            answering_memory_pack=answering_pack,
            feedback_applied=feedback_applied,
            memory_outcomes=outcomes,
            post_answer_errors=tuple(errors),
            llm_usage=usage_collector.snapshot(),
            diagnostics=diagnostic_snapshot,
        )

    def _apply_feedback(
        self,
        *,
        user_id: UUID,
        answer: Message,
        answering_pack: MemoryPack,
        context_memory_ids: tuple[UUID, ...],
        errors: list[str],
    ) -> bool:
        ranked = {
            retrieved.memory.memory_id: retrieved.rank
            for retrieved in answering_pack.memories
        }
        candidate_ids = set(ranked)
        if not set(context_memory_ids).issubset(candidate_ids):
            errors.append("feedback: context contained a non-returned memory")
            return False
        usages = tuple(
            MemoryUsage(
                memory_id=memory_id,
                usage_type=UsageType.CONTEXT_INCLUDED,
                rank=ranked[memory_id],
            )
            for memory_id in context_memory_ids
        )
        if not usages:
            return True
        try:
            self._reinforce_memory.execute(
                feedback_pack=FeedbackPack(
                    query_id=answering_pack.query_id,
                    user_id=user_id,
                    session_id=answer.session_id,
                    used_memories=usages,
                )
            )
        except Exception as error:
            errors.append(_error_text("feedback", error))
            return False
        return True

    def _reconcile_and_store_candidates(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        session_history: MessagePack,
        evidence_messages: tuple[Message, ...],
        candidate_source_ids: set[UUID],
        candidates: tuple[ProposedMemory, ...],
        memory_pack: MemoryPack,
        errors: list[str],
        usage_collector: ModelUsageCollector,
        diagnostics_collector: TurnDiagnosticsCollector | None,
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
                candidate_source_ids=candidate_source_ids,
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
                detail = _error_text("reconciliation", error)
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

        for index, candidate, decision in zip(
            valid_indices,
            valid_candidates,
            decisions,
        ):
            outcomes_by_index[index] = self._process_candidate(
                user_id=user_id,
                candidate=candidate,
                decision=decision,
                memory_pack=memory_pack,
                usage_collector=usage_collector,
                diagnostics_collector=diagnostics_collector,
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

    def _process_candidate(
        self,
        *,
        user_id: UUID,
        candidate: ProposedMemory,
        decision: ReconciliationDecision,
        memory_pack: MemoryPack,
        usage_collector: ModelUsageCollector,
        diagnostics_collector: TurnDiagnosticsCollector | None,
    ) -> MemoryWriteOutcome:
        try:
            allowed_ids = {
                retrieved.memory.memory_id for retrieved in memory_pack.memories
            }
            if decision.action is ReconciliationAction.NONE:
                if decision.equivalent_memory_id not in allowed_ids:
                    raise ValueError(
                        "equivalent memory is absent from persisted turn context"
                    )
                return MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.EQUIVALENT,
                    write_context_query_id=memory_pack.query_id,
                    equivalent_memory_id=decision.equivalent_memory_id,
                )

            proposed_neighbor_ids = [
                proposal.neighbor_memory_id
                for proposal in decision.conflict_proposals
            ]
            if len(set(proposed_neighbor_ids)) != len(proposed_neighbor_ids):
                raise ValueError("conflict proposals contain duplicate IDs")
            if not set(proposed_neighbor_ids).issubset(allowed_ids):
                raise ValueError(
                    "conflict proposal is absent from persisted turn context"
                )

            if not self._enable_memory_writes:
                return MemoryWriteOutcome(
                    candidate=candidate,
                    status=MemoryWriteStatus.DRY_RUN,
                    write_context_query_id=memory_pack.query_id,
                )

            memory = proposed_memory_to_memory(
                proposal=candidate,
                memory_id=self._id_factory(),
                created_at=self._clock.now(),
            )
            memory_id = self._store_memory.execute(
                user_id=user_id,
                memory=memory,
                write_context_query_id=memory_pack.query_id,
                conflict_proposals=(
                    decision.conflict_proposals
                    if self._enable_conflict_detection
                    else ()
                ),
                usage_recorder=usage_collector,
                diagnostics_recorder=diagnostics_collector,
            )
            return MemoryWriteOutcome(
                candidate=candidate,
                status=MemoryWriteStatus.STORED,
                write_context_query_id=memory_pack.query_id,
                memory_id=memory_id,
            )
        except Exception as error:
            return MemoryWriteOutcome(
                candidate=candidate,
                status=MemoryWriteStatus.FAILED,
                write_context_query_id=memory_pack.query_id,
                error=_error_text("candidate", error),
            )


def _error_text(stage: str, error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{stage}: {detail}"
