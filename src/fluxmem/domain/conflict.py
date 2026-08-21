from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fluxmem.domain.memory import Memory


@dataclass(frozen=True, slots=True)
class ConflictProposal:
    """A potential conflict selected from one write-time context pack."""

    neighbor_memory_id: UUID
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("conflict confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """One undirected potential-conflict edge in canonical UUID order."""

    memory_a_id: UUID
    memory_b_id: UUID
    confidence: float | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.memory_a_id >= self.memory_b_id:
            raise ValueError("conflict IDs must be distinct and canonically ordered")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("conflict confidence must be between 0 and 1")

    @classmethod
    def between(
        cls,
        *,
        memory_id: UUID,
        neighbor_memory_id: UUID,
        confidence: float | None,
        created_at: datetime,
    ) -> MemoryConflict:
        memory_a_id, memory_b_id = sorted((memory_id, neighbor_memory_id))
        return cls(
            memory_a_id=memory_a_id,
            memory_b_id=memory_b_id,
            confidence=confidence,
            created_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class ConflictNeighbor:
    """An eligible one-hop neighbor linked to one selected seed memory."""

    seed_memory_id: UUID
    memory: Memory
    conflict: MemoryConflict
    retention: float

    def __post_init__(self) -> None:
        if self.seed_memory_id not in (
            self.conflict.memory_a_id,
            self.conflict.memory_b_id,
        ):
            raise ValueError("conflict neighbor seed is not an edge endpoint")
        if self.memory.memory_id not in (
            self.conflict.memory_a_id,
            self.conflict.memory_b_id,
        ):
            raise ValueError("conflict neighbor memory is not an edge endpoint")
        if self.memory.memory_id == self.seed_memory_id:
            raise ValueError("conflict neighbor cannot be its own seed")
        if not 0.0 <= self.retention <= 1.0:
            raise ValueError("conflict neighbor retention must be between 0 and 1")
