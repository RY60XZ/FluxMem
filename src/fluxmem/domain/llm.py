from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping
from uuid import UUID

from fluxmem.domain.conflict import ConflictProposal
from fluxmem.domain.info_pack import TurnMemoryPacks
from fluxmem.domain.memory import Memory, validate_memory_fields
from fluxmem.domain.message import Message


class LLMTaskKind(StrEnum):
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
class ProposedMemory:
    """One atomic fact extracted from persisted conversational evidence."""

    content: str
    source_message_id: UUID
    session_applicability: UUID | None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        validate_memory_fields(
            content=self.content,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
        )


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
class MemoryDiagnostics:
    """Immutable snapshot of one opt-in memory-learning trace."""

    retrieval: TurnMemoryPacks | None = None
    model_calls: tuple[ModelCallDiagnostics, ...] = ()
    memory_outcomes: tuple[MemoryWriteOutcome, ...] = ()
    errors: tuple[str, ...] = ()
    complete: bool = False

    def calls_for_task(
        self, task: LLMTaskKind
    ) -> tuple[ModelCallDiagnostics, ...]:
        return tuple(call for call in self.model_calls if call.task is task)


@dataclass(frozen=True, slots=True)
class MessageIngestionResult:
    """Result of memory learning from imported messages."""

    messages: tuple[Message, ...]
    memory_outcomes: tuple[MemoryWriteOutcome, ...] = ()
    errors: tuple[str, ...] = ()
    llm_usage: LLMUsageReport = LLMUsageReport()
    diagnostics: MemoryDiagnostics | None = None


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
