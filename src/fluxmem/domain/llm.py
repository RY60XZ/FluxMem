from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping
from uuid import UUID

from fluxmem.domain.conflict import ConflictProposal
from fluxmem.domain.info_pack import MemoryPack, TurnMemoryPacks
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


class LLMTaskKind(StrEnum):
    ANSWER = "answer"
    EXTRACTION = "extraction"
    RECONCILIATION = "reconciliation"
    LIFECYCLE = "lifecycle"


@dataclass(frozen=True, slots=True)
class ModelTokenUsage:
    """Provider-neutral token counts reported for one model response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_input_tokens,
            self.cache_write_input_tokens,
            self.reasoning_output_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("model token counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("model token counts cannot be negative")
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError(
                "total tokens cannot be less than input plus output tokens"
            )
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed input tokens")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning output tokens cannot exceed output tokens")

    def __add__(self, other: ModelTokenUsage) -> ModelTokenUsage:
        if not isinstance(other, ModelTokenUsage):
            return NotImplemented
        return ModelTokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=(
                self.cached_input_tokens + other.cached_input_tokens
            ),
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
            reasoning_output_tokens=(
                self.reasoning_output_tokens + other.reasoning_output_tokens
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelCallUsage:
    """Identity and usage metadata for one attempted model response."""

    task: LLMTaskKind
    model: str
    attempt: int
    token_usage: ModelTokenUsage | None
    response_id: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model name cannot be blank")
        if self.attempt < 1:
            raise ValueError("model-call attempt must be positive")


@dataclass(frozen=True, slots=True)
class LLMUsageReport:
    """All observed model calls, including repair attempts, for an operation."""

    calls: tuple[ModelCallUsage, ...] = ()

    @property
    def token_totals(self) -> ModelTokenUsage | None:
        reported = tuple(
            call.token_usage
            for call in self.calls
            if call.token_usage is not None
        )
        if not reported:
            return None
        total = reported[0]
        for usage in reported[1:]:
            total += usage
        return total

    @property
    def usage_complete(self) -> bool:
        return all(call.token_usage is not None for call in self.calls)

    @property
    def reported_call_count(self) -> int:
        return sum(call.token_usage is not None for call in self.calls)

    def for_task(self, task: LLMTaskKind) -> LLMUsageReport:
        return LLMUsageReport(
            calls=tuple(call for call in self.calls if call.task is task)
        )


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """An answer plus memory-use metadata validated by the application."""

    content: str
    context_memory_ids: tuple[UUID, ...]
    attributed_memory_ids: tuple[UUID, ...] = ()
    llm_usage: LLMUsageReport = LLMUsageReport()

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("generated answer cannot be blank")
        if len(set(self.context_memory_ids)) != len(self.context_memory_ids):
            raise ValueError("context memory IDs must be unique")
        if len(set(self.attributed_memory_ids)) != len(
            self.attributed_memory_ids
        ):
            raise ValueError("attributed memory IDs must be unique")
        if not set(self.attributed_memory_ids).issubset(self.context_memory_ids):
            raise ValueError("attributed memories must have been placed in context")


@dataclass(frozen=True, slots=True)
class GeneratedAnswerStream:
    """Provider-neutral answer text deltas plus the supplied memory context."""

    chunks: Iterable[str]
    context_memory_ids: tuple[UUID, ...]
    _usage_supplier: Callable[[], LLMUsageReport] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if len(set(self.context_memory_ids)) != len(self.context_memory_ids):
            raise ValueError("context memory IDs must be unique")

    def __iter__(self) -> Iterator[str]:
        return iter(self.chunks)

    @property
    def llm_usage(self) -> LLMUsageReport:
        if self._usage_supplier is None:
            return LLMUsageReport()
        return self._usage_supplier()


@dataclass(frozen=True, slots=True)
class ProposedMemory:
    """One atomic fact extracted from persisted conversational evidence."""

    content: str
    source_message_id: UUID
    session_applicability: UUID | None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("proposed memory cannot be blank")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_from > self.valid_to
        ):
            raise ValueError("memory validity cannot end before it begins")


class ReconciliationAction(StrEnum):
    ADD = "ADD"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class ReconciliationDecision:
    """ADD or reuse an equivalent memory from persisted turn context."""

    action: ReconciliationAction
    equivalent_memory_id: UUID | None = None
    conflict_proposals: tuple[ConflictProposal, ...] = ()

    def __post_init__(self) -> None:
        if self.action is ReconciliationAction.ADD:
            if self.equivalent_memory_id is not None:
                raise ValueError("ADD cannot identify an equivalent memory")
            return
        if self.equivalent_memory_id is None:
            raise ValueError("NONE must identify an equivalent memory")
        if self.conflict_proposals:
            raise ValueError("NONE cannot create conflict proposals")


class MemoryWriteStatus(StrEnum):
    STORED = "stored"
    EQUIVALENT = "equivalent"
    DRY_RUN = "dry_run"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryWriteOutcome:
    candidate: ProposedMemory
    status: MemoryWriteStatus
    write_context_query_id: UUID | None = None
    memory_id: UUID | None = None
    equivalent_memory_id: UUID | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is MemoryWriteStatus.STORED and self.memory_id is None:
            raise ValueError("stored outcome requires a memory ID")
        if (
            self.status is MemoryWriteStatus.EQUIVALENT
            and self.equivalent_memory_id is None
        ):
            raise ValueError("equivalent outcome requires an existing memory ID")
        if self.status is MemoryWriteStatus.FAILED and not self.error:
            raise ValueError("failed outcome requires an error")


@dataclass(frozen=True, slots=True)
class ModelCallDiagnostics:
    """Exact opt-in model request, raw response, and validated result."""

    task: LLMTaskKind
    model: str
    attempt: int
    instructions: str
    input_text: str
    schema_name: str | None
    schema: Mapping[str, Any] | None
    supplied_memory_ids: tuple[UUID, ...]
    supplied_message_ids: tuple[UUID, ...]
    output_text: str | None
    validated_output: object | None
    response_id: str | None
    token_usage: ModelTokenUsage | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationTurnDiagnostics:
    """Immutable snapshot of one opt-in workflow trace."""

    retrieval: TurnMemoryPacks | None = None
    answer_context_memory_ids: tuple[UUID, ...] = ()
    model_calls: tuple[ModelCallDiagnostics, ...] = ()
    feedback_applied: bool | None = None
    memory_outcomes: tuple[MemoryWriteOutcome, ...] = ()
    post_answer_errors: tuple[str, ...] = ()
    complete: bool = False

    def calls_for_task(
        self, task: LLMTaskKind
    ) -> tuple[ModelCallDiagnostics, ...]:
        return tuple(call for call in self.model_calls if call.task is task)


@dataclass(frozen=True, slots=True)
class ConversationTurnResult:
    """Observable result of the critical answer path and post-answer work."""

    answer: Message
    answering_memory_pack: MemoryPack
    feedback_applied: bool
    memory_outcomes: tuple[MemoryWriteOutcome, ...]
    post_answer_errors: tuple[str, ...] = ()
    llm_usage: LLMUsageReport = LLMUsageReport()
    diagnostics: ConversationTurnDiagnostics | None = None


def proposed_memory_to_memory(
    *,
    proposal: ProposedMemory,
    memory_id: UUID,
    created_at: datetime,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        message_id=proposal.source_message_id,
        content=proposal.content,
        created_at=created_at,
        valid_from=proposal.valid_from,
        valid_to=proposal.valid_to,
        session_applicability=proposal.session_applicability,
    )
