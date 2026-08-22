from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from fluxmem.domain.conflict import ConflictProposal
from fluxmem.domain.info_pack import MemoryPack
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """An answer plus memory-use metadata validated by the application."""

    content: str
    context_memory_ids: tuple[UUID, ...]
    attributed_memory_ids: tuple[UUID, ...] = ()

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

    def __post_init__(self) -> None:
        if len(set(self.context_memory_ids)) != len(self.context_memory_ids):
            raise ValueError("context memory IDs must be unique")

    def __iter__(self) -> Iterator[str]:
        return iter(self.chunks)


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
class ConversationTurnResult:
    """Observable result of the critical answer path and post-answer work."""

    answer: Message
    answering_memory_pack: MemoryPack
    feedback_applied: bool
    memory_outcomes: tuple[MemoryWriteOutcome, ...]
    post_answer_errors: tuple[str, ...] = ()


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
