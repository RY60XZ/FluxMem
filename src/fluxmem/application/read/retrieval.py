from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from uuid import UUID, uuid4

from fluxmem.application.errors import (
    InvalidRetrievalContextError,
    SessionNotFoundError,
)
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.conflict import ConflictNeighbor, MemoryConflict
from fluxmem.domain.info_pack import (
    MemoryPack,
    MessagePack,
    RetrievedMemory,
    TurnMemoryPacks,
)
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    MemorySearchQuery,
)


@dataclass(frozen=True, slots=True)
class HybridRetrievalSettings:
    """Bound query size and expose fusion knobs without adapter coupling."""

    max_query_messages: int = 16
    max_query_characters: int = 8_000
    candidate_multiplier: int = 5
    minimum_candidate_limit: int = 50
    maximum_candidate_limit: int = 200
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    relevance_floor: float = 0.6
    maximum_conflicts_per_seed: int = 3
    maximum_conflict_expansions: int = 10

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
        if self.maximum_conflicts_per_seed < 0:
            raise ValueError("per-seed conflict limit cannot be negative")
        if self.maximum_conflict_expansions < 0:
            raise ValueError("global conflict limit cannot be negative")

    def candidate_limit(self, result_limit: int) -> int:
        requested = max(
            self.minimum_candidate_limit,
            result_limit * self.candidate_multiplier,
        )
        return min(requested, self.maximum_candidate_limit)


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
    """Add messages without losing metadata or duplicating message identities."""

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


def _bounded_query_text(
    *,
    message_pack: MessagePack,
    settings: HybridRetrievalSettings,
) -> str:
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
    return "\n".join(reversed(selected))


@dataclass(frozen=True, slots=True)
class _ConflictExpansion:
    memories: tuple[RetrievedMemory, ...]
    conflicts: tuple[MemoryConflict, ...]
    truncated: bool


def _expand_conflict_neighbors(
    *,
    seeds: tuple[RetrievedMemory, ...],
    neighbors: tuple[ConflictNeighbor, ...],
    settings: HybridRetrievalSettings,
) -> _ConflictExpansion:
    """Apply fair per-seed and global budgets without adding another hop."""

    if not seeds or not neighbors:
        return _ConflictExpansion(
            memories=seeds,
            conflicts=(),
            truncated=False,
        )

    ordered_seeds = tuple(sorted(seeds, key=lambda seed: seed.rank))
    seed_by_id = {seed.memory.memory_id: seed for seed in ordered_seeds}
    seed_ids = set(seed_by_id)
    grouped: dict[UUID, list[ConflictNeighbor]] = {
        memory_id: [] for memory_id in seed_by_id
    }
    for neighbor in neighbors:
        if neighbor.seed_memory_id in grouped:
            grouped[neighbor.seed_memory_id].append(neighbor)

    def priority(neighbor: ConflictNeighbor) -> tuple[bool, float, float, int]:
        confidence = neighbor.conflict.confidence
        return (
            confidence is None,
            -(confidence if confidence is not None else 0.0),
            -neighbor.retention,
            neighbor.memory.memory_id.int,
        )

    queues: dict[UUID, list[ConflictNeighbor]] = {}
    eligible_neighbor_ids: set[UUID] = set()
    for seed in ordered_seeds:
        seed_id = seed.memory.memory_id
        expansion_candidates = [
            neighbor
            for neighbor in grouped[seed_id]
            if neighbor.memory.memory_id not in seed_ids
        ]
        eligible_neighbor_ids.update(
            neighbor.memory.memory_id for neighbor in expansion_candidates
        )
        queues[seed_id] = sorted(
            expansion_candidates,
            key=priority,
        )[: settings.maximum_conflicts_per_seed]

    selected: dict[UUID, ConflictNeighbor] = {}
    while len(selected) < settings.maximum_conflict_expansions:
        made_progress = False
        for seed in ordered_seeds:
            queue = queues[seed.memory.memory_id]
            while queue:
                neighbor = queue.pop(0)
                neighbor_id = neighbor.memory.memory_id
                if neighbor_id in selected:
                    continue
                selected[neighbor_id] = neighbor
                made_progress = True
                break
            if len(selected) >= settings.maximum_conflict_expansions:
                break
        if not made_progress:
            break

    expanded_memories = list(ordered_seeds)
    for neighbor in selected.values():
        seed = seed_by_id[neighbor.seed_memory_id]
        confidence = neighbor.conflict.confidence
        conflict_weight = confidence if confidence is not None else 1.0
        expanded_memories.append(
            RetrievedMemory(
                memory=neighbor.memory,
                rank=len(expanded_memories) + 1,
                score=seed.score * conflict_weight,
                retention=neighbor.retention,
                retrieval_reasons=("conflict", "lifecycle"),
            )
        )

    included_ids = seed_ids.union(selected)
    conflicts_by_pair = {
        (neighbor.conflict.memory_a_id, neighbor.conflict.memory_b_id): (
            neighbor.conflict
        )
        for neighbor in neighbors
        if neighbor.conflict.memory_a_id in included_ids
        and neighbor.conflict.memory_b_id in included_ids
    }
    conflicts = tuple(
        conflicts_by_pair[key] for key in sorted(conflicts_by_pair)
    )
    return _ConflictExpansion(
        memories=tuple(expanded_memories),
        conflicts=conflicts,
        truncated=set(selected) != eligible_neighbor_ids,
    )


class HybridMemoryRetriever:
    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        embedding_provider: EmbeddingProvider | None = None,
        settings: HybridRetrievalSettings | None = None,
        clock: Clock | None = None,
        query_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._embedding_provider = embedding_provider
        self._settings = settings or HybridRetrievalSettings()
        self._clock = clock or SystemClock()
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
        limit: int,
    ) -> TurnMemoryPacks:
        if limit < 1:
            raise ValueError("retrieval limit must be positive")
        if limit > self._settings.maximum_candidate_limit:
            raise ValueError("retrieval limit exceeds the configured maximum")

        query_embedding = None
        if self._embedding_provider is not None and query_text:
            try:
                query_embedding = self._embedding_provider.embed(text=query_text)
            except EmbeddingProviderError:
                query_embedding = None
            if (
                query_embedding is not None
                and len(query_embedding.values) != EMBEDDING_DIMENSIONS
            ):
                raise ValueError(
                    "embedding provider returned a vector with invalid dimensions"
                )

        retrieved_at = self._clock.now()
        query_id = self._query_id_factory()
        search_query = MemorySearchQuery(
            user_id=user_id,
            session_id=session_id,
            text=query_text,
            embedding=query_embedding,
            as_of=retrieved_at,
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

            seed_memories = unit_of_work.memories.search(query=search_query)
            conflict_neighbors = ()
            if (
                seed_memories
                and self._settings.maximum_conflicts_per_seed > 0
                and self._settings.maximum_conflict_expansions > 0
            ):
                conflict_neighbors = unit_of_work.conflicts.expand(
                    seed_memory_ids=tuple(
                        memory.memory.memory_id for memory in seed_memories
                    ),
                    user_id=user_id,
                    session_id=session_id,
                    as_of=retrieved_at,
                )
            expansion = _expand_conflict_neighbors(
                seeds=seed_memories,
                neighbors=conflict_neighbors,
                settings=self._settings,
            )
            unit_of_work.retrievals.add(
                query_id=query_id,
                session_id=session_id,
                candidates=expansion.memories,
                created_at=retrieved_at,
            )
            unit_of_work.commit()

        return TurnMemoryPacks(
            seeds=MemoryPack(
                query_id=query_id,
                user_id=user_id,
                session_id=session_id,
                memories=seed_memories,
            ),
            expanded=MemoryPack(
                query_id=query_id,
                user_id=user_id,
                session_id=session_id,
                memories=expansion.memories,
                conflicts=expansion.conflicts,
                conflict_expansion_truncated=expansion.truncated,
            ),
        )

    def execute(
        self,
        *,
        message: Message,
        session_history: MessagePack,
        limit: int = 10,
    ) -> TurnMemoryPacks:
        _validate_context_message(message=message, history=session_history)
        retrieval_messages = _extend_message_pack(
            history=session_history,
            messages=(message,),
        )
        return self._retrieve(
            user_id=retrieval_messages.user_id,
            session_id=retrieval_messages.session_id,
            query_text=_bounded_query_text(
                message_pack=retrieval_messages,
                settings=self._settings,
            ),
            limit=limit,
        )
