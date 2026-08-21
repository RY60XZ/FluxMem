from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from fluxmem.domain.info_pack import MemoryUsage, RetrievedMemory
from fluxmem.domain.lifecycle import MemoryLifecycle
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import (
    Embedding,
    MemoryIndex,
    MemorySearchQuery,
    QueryType,
)


class MessageRepository(Protocol):
    def add(self, *, message: Message) -> None: ...

    def get(self, *, message_id: UUID, user_id: UUID) -> Message | None: ...

    def list_for_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int | None = None,
    ) -> tuple[Message, ...]: ...


class SessionRepository(Protocol):
    def is_owned_by(self, *, session_id: UUID, user_id: UUID) -> bool: ...


class MemoryRepository(Protocol):
    def add(self, *, memory: Memory) -> None: ...

    def get(self, *, memory_id: UUID, user_id: UUID) -> Memory | None: ...

    def search(
        self,
        *,
        query: MemorySearchQuery,
    ) -> tuple[RetrievedMemory, ...]: ...


class MemoryIndexRepository(Protocol):
    def add(self, *, index: MemoryIndex) -> None: ...

    def list_pending(self, *, limit: int) -> tuple[Memory, ...]: ...

    def mark_ready(
        self,
        *,
        memory_id: UUID,
        embedding: Embedding,
        indexed_at: datetime,
    ) -> bool: ...


class RetrievalRepository(Protocol):
    def add(
        self,
        *,
        query_id: UUID,
        session_id: UUID,
        query_type: QueryType,
        candidates: tuple[RetrievedMemory, ...],
        created_at: datetime,
    ) -> None: ...

    def candidate_ids_for_query(
        self,
        *,
        query_id: UUID,
        user_id: UUID,
        session_id: UUID,
    ) -> tuple[UUID, ...] | None: ...


class LifecycleRepository(Protocol):
    def add(self, *, lifecycle: MemoryLifecycle) -> None: ...

    def get(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
    ) -> MemoryLifecycle | None: ...

    def reinforce(
        self,
        *,
        memory_id: UUID,
        query_id: UUID,
        user_id: UUID,
        session_id: UUID,
        usage: MemoryUsage,
        used_at: datetime,
    ) -> MemoryLifecycle | None: ...
