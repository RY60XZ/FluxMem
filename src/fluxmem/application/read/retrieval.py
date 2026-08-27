from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from fluxmem.application.errors import (
    InvalidRetrievalContextError,
    SessionNotFoundError,
)
from fluxmem.application.ports.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.info_pack import (
    MemoryPack,
    MessagePack,
    RetrievalQueryDiagnostics,
)
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    MemorySearchQuery,
)


@dataclass(frozen=True, slots=True)
class HybridRetrievalSettings:
    """Bound query size and expose provider-neutral fusion settings."""

    max_query_messages: int = 16
    max_query_characters: int = 8_000
    candidate_multiplier: int = 5
    minimum_candidate_limit: int = 50
    maximum_candidate_limit: int = 200
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    relevance_floor: float = 0.6

    def __post_init__(self) -> None:
        if self.max_query_messages < 1 or self.max_query_characters < 1:
            raise ValueError("query bounds must be positive")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate multiplier must be positive")
        if not 1 <= self.minimum_candidate_limit <= self.maximum_candidate_limit:
            raise ValueError("candidate limit bounds are invalid")
        if self.rrf_k < 1:
            raise ValueError("RRF constant must be positive")
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.dense_weight == 0 and self.lexical_weight == 0:
            raise ValueError("at least one retrieval source must be weighted")
        if not 0.0 <= self.relevance_floor <= 1.0:
            raise ValueError("relevance floor must be between 0 and 1")

    def candidate_limit(self, result_limit: int) -> int:
        requested = max(
            self.minimum_candidate_limit,
            result_limit * self.candidate_multiplier,
        )
        return min(requested, self.maximum_candidate_limit)


@dataclass(frozen=True, slots=True)
class _BoundedQuery:
    text: str
    message_count: int


def _validate_context_message(*, message: Message, history: MessagePack) -> None:
    if message.session_id != history.session_id:
        raise InvalidRetrievalContextError(
            "retrieval message and session history belong to different sessions"
        )


def _extend_message_pack(
    *,
    history: MessagePack,
    messages: Iterable[Message],
) -> MessagePack:
    """Merge messages while preserving metadata and unique identities."""

    combined_messages: list[Message] = []
    seen_message_ids: set[UUID] = set()
    for message in (*history.messages, *messages):
        if message.message_id in seen_message_ids:
            continue
        seen_message_ids.add(message.message_id)
        combined_messages.append(message)

    return MessagePack(
        user_id=history.user_id,
        session_id=history.session_id,
        messages=tuple(combined_messages),
    )


def _bounded_query(
    *,
    message_pack: MessagePack,
    settings: HybridRetrievalSettings,
) -> _BoundedQuery:
    """Prefer recent content while keeping the final query chronological."""

    remaining = settings.max_query_characters
    selected: list[str] = []
    recent_messages = message_pack.messages[-settings.max_query_messages :]
    for message in reversed(recent_messages):
        content = message.content.strip()
        if not content:
            continue
        separator_size = 1 if selected else 0
        if remaining <= separator_size:
            break
        remaining -= separator_size
        piece = content[:remaining]
        selected.append(piece)
        remaining -= len(piece)
        if remaining == 0:
            break
    return _BoundedQuery(
        text="\n".join(reversed(selected)),
        message_count=len(selected),
    )


class HybridMemoryRetriever:
    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        embedding_provider: EmbeddingProvider | None = None,
        settings: HybridRetrievalSettings | None = None,
        query_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._embedding_provider = embedding_provider
        self._settings = settings or HybridRetrievalSettings()
        self._query_id_factory = query_id_factory
        if (
            embedding_provider is not None
            and embedding_provider.dimensions != EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                "embedding provider dimensions do not match the database schema"
            )

    def _retrieve(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        query_text: str,
        query_message_count: int,
        as_of: datetime,
        limit: int,
    ) -> MemoryPack:
        if limit < 1:
            raise ValueError("retrieval limit must be positive")
        if limit > self._settings.maximum_candidate_limit:
            raise ValueError("retrieval limit exceeds the configured maximum")

        query_embedding = None
        embedding_error = None
        if self._embedding_provider is not None and query_text:
            try:
                query_embedding = self._embedding_provider.embed(text=query_text)
            except EmbeddingProviderError as error:
                embedding_error = f"{type(error).__name__}: {error}"
                query_embedding = None
            if (
                query_embedding is not None
                and len(query_embedding.values) != EMBEDDING_DIMENSIONS
            ):
                raise ValueError(
                    "embedding provider returned a vector with invalid dimensions"
                )

        query_id = self._query_id_factory()
        search_query = MemorySearchQuery(
            user_id=user_id,
            session_id=session_id,
            text=query_text,
            embedding=query_embedding,
            as_of=as_of,
            limit=limit,
            candidate_limit=self._settings.candidate_limit(limit),
            rrf_k=self._settings.rrf_k,
            dense_weight=self._settings.dense_weight,
            lexical_weight=self._settings.lexical_weight,
            relevance_floor=self._settings.relevance_floor,
        )

        with self._unit_of_work_factory() as unit_of_work:
            if not unit_of_work.sessions.is_owned_by(
                session_id=session_id,
                user_id=user_id,
            ):
                raise SessionNotFoundError(
                    "retrieval session does not belong to the requested user"
                )

            memories = unit_of_work.memories.search(query=search_query)
            unit_of_work.retrievals.add(
                query_id=query_id,
                session_id=session_id,
                candidates=memories,
                created_at=as_of,
            )
            unit_of_work.commit()

        return MemoryPack(
            query_id=query_id,
            user_id=user_id,
            session_id=session_id,
            memories=memories,
            diagnostics=RetrievalQueryDiagnostics(
                search_text=query_text,
                message_count=query_message_count,
                result_limit=limit,
                source_candidate_limit=search_query.candidate_limit,
                returned_count=len(memories),
                dense_enabled=(
                    query_embedding is not None
                    and self._settings.dense_weight > 0
                ),
                lexical_enabled=(
                    bool(query_text) and self._settings.lexical_weight > 0
                ),
                embedding_model=(
                    query_embedding.model
                    if query_embedding is not None
                    else None
                ),
                embedding_error=embedding_error,
            ),
        )

    def execute(
        self,
        *,
        message: Message,
        session_history: MessagePack,
        limit: int = 10,
    ) -> MemoryPack:
        _validate_context_message(message=message, history=session_history)
        retrieval_messages = _extend_message_pack(
            history=session_history,
            messages=(message,),
        )
        bounded_query = _bounded_query(
            message_pack=retrieval_messages,
            settings=self._settings,
        )
        return self._retrieve(
            user_id=retrieval_messages.user_id,
            session_id=retrieval_messages.session_id,
            query_text=bounded_query.text,
            query_message_count=bounded_query.message_count,
            as_of=message.created_at,
            limit=limit,
        )
