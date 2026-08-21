from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import case, func, literal, or_, select, union_all
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.lifecycle import MemoryLifecycleRow
from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.memory import MemoryIndexRow, MemoryRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.info_pack import RetrievedMemory
from fluxmem.domain.lifecycle import (
    RETENTION_FLOOR,
    STABILITY_DAYS,
    DecayClass,
    Status,
)
from fluxmem.domain.memory import Memory
from fluxmem.domain.retrieval import IndexStatus, MemorySearchQuery


def _to_domain(row: MemoryRow) -> Memory:
    return Memory(
        memory_id=row.memory_id,
        message_id=row.message_id,
        content=row.content,
        created_at=row.created_at,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        session_applicability=row.session_applicability,
    )


class SqlAlchemyMemoryRepository:
    """Map between domain memories and their PostgreSQL rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, *, memory: Memory) -> None:
        self._session.add(
            MemoryRow(
                memory_id=memory.memory_id,
                message_id=memory.message_id,
                content=memory.content,
                created_at=memory.created_at,
                valid_from=memory.valid_from,
                valid_to=memory.valid_to,
                session_applicability=memory.session_applicability,
            )
        )

    def get(self, *, memory_id: UUID, user_id: UUID) -> Memory | None:
        statement = (
            select(MemoryRow)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .where(
                MemoryRow.memory_id == memory_id,
                SessionRow.user_id == user_id,
            )
        )
        row = self._session.scalar(statement)
        if row is None:
            return None
        return _to_domain(row)

    def search(
        self,
        *,
        query: MemorySearchQuery,
    ) -> tuple[RetrievedMemory, ...]:
        """Fuse indexed lexical and dense candidates, then apply lifecycle."""

        candidate_legs = []
        eligibility = (
            SessionRow.user_id == query.user_id,
            MemoryLifecycleRow.status == Status.ACTIVE.value,
            or_(
                MemoryRow.session_applicability.is_(None),
                MemoryRow.session_applicability == query.session_id,
            ),
            or_(MemoryRow.valid_from.is_(None), MemoryRow.valid_from <= query.as_of),
            or_(MemoryRow.valid_to.is_(None), MemoryRow.valid_to >= query.as_of),
        )

        tokens = re.findall(r"\w+", query.text.casefold())
        recent_unique_terms = tuple(dict.fromkeys(reversed(tokens)))[:256]
        lexical_terms = tuple(reversed(recent_unique_terms))
        if lexical_terms and query.lexical_weight > 0:
            search_query = func.to_tsquery(
                "english",
                " | ".join(lexical_terms),
            )
            lexical_score = func.ts_rank_cd(
                MemoryIndexRow.search_text,
                search_query,
            )
            lexical_rank = func.row_number().over(
                order_by=(
                    lexical_score.desc(),
                    MemoryRow.created_at.desc(),
                    MemoryRow.memory_id,
                )
            )
            candidate_legs.append(
                select(
                    MemoryRow.memory_id.label("memory_id"),
                    literal("lexical").label("source"),
                    lexical_rank.label("source_rank"),
                )
                .select_from(MemoryIndexRow)
                .join(
                    MemoryRow,
                    MemoryRow.memory_id == MemoryIndexRow.memory_id,
                )
                .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
                .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
                .join(
                    MemoryLifecycleRow,
                    MemoryLifecycleRow.memory_id == MemoryRow.memory_id,
                )
                .where(
                    *eligibility,
                    MemoryIndexRow.search_text.op("@@")(search_query),
                )
                .order_by(
                    lexical_score.desc(),
                    MemoryRow.created_at.desc(),
                    MemoryRow.memory_id,
                )
                .limit(query.candidate_limit)
            )

        if query.embedding is not None and query.dense_weight > 0:
            distance = MemoryIndexRow.embedding.cosine_distance(
                list(query.embedding.values)
            )
            dense_nearest = (
                select(
                    MemoryRow.memory_id.label("memory_id"),
                    distance.label("distance"),
                )
                .select_from(MemoryIndexRow)
                .join(
                    MemoryRow,
                    MemoryRow.memory_id == MemoryIndexRow.memory_id,
                )
                .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
                .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
                .join(
                    MemoryLifecycleRow,
                    MemoryLifecycleRow.memory_id == MemoryRow.memory_id,
                )
                .where(
                    *eligibility,
                    MemoryIndexRow.index_status == IndexStatus.READY.value,
                    MemoryIndexRow.embedding.is_not(None),
                    MemoryIndexRow.embedding_model == query.embedding.model,
                )
                .order_by(distance)
                .limit(query.candidate_limit)
                .cte("dense_nearest")
            )
            dense_rank = func.row_number().over(
                order_by=(
                    dense_nearest.c.distance,
                    dense_nearest.c.memory_id,
                )
            )
            candidate_legs.append(
                select(
                    dense_nearest.c.memory_id,
                    literal("dense").label("source"),
                    dense_rank.label("source_rank"),
                )
                .select_from(dense_nearest)
            )

        if not candidate_legs:
            return ()

        if len(candidate_legs) == 1:
            candidates = candidate_legs[0].cte("retrieval_candidates")
        else:
            candidates = union_all(*candidate_legs).cte("retrieval_candidates")

        source_weight = case(
            (
                candidates.c.source == "dense",
                literal(query.dense_weight),
            ),
            else_=literal(query.lexical_weight),
        )
        rrf_contribution = source_weight / (
            literal(float(query.rrf_k)) + candidates.c.source_rank
        )
        fused = (
            select(
                candidates.c.memory_id,
                func.sum(rrf_contribution).label("base_score"),
                func.max(
                    case((candidates.c.source == "dense", 1), else_=0)
                ).label("dense_match"),
                func.max(
                    case((candidates.c.source == "lexical", 1), else_=0)
                ).label("lexical_match"),
            )
            .group_by(candidates.c.memory_id)
            .cte("fused_candidates")
        )

        elapsed_days = func.greatest(
            0.0,
            func.extract(
                "epoch",
                literal(query.as_of) - MemoryLifecycleRow.retention_anchor,
            )
            / 86_400.0,
        )
        stability_days = case(
            (
                MemoryLifecycleRow.decay_class == DecayClass.FAST.value,
                STABILITY_DAYS[DecayClass.FAST],
            ),
            (
                MemoryLifecycleRow.decay_class == DecayClass.STANDARD.value,
                STABILITY_DAYS[DecayClass.STANDARD],
            ),
            else_=STABILITY_DAYS[DecayClass.SLOW],
        )
        retention = (
            RETENTION_FLOOR
            + (MemoryLifecycleRow.retention_snapshot - RETENTION_FLOOR)
            * func.exp(-elapsed_days / stability_days)
        ).label("retention")
        score = (
            fused.c.base_score
            * (
                query.relevance_floor
                + (1.0 - query.relevance_floor) * retention
            )
        ).label("score")
        statement = (
            select(
                MemoryRow,
                score,
                retention,
                fused.c.base_score,
                fused.c.dense_match,
                fused.c.lexical_match,
            )
            .join(fused, fused.c.memory_id == MemoryRow.memory_id)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .join(
                MemoryLifecycleRow,
                MemoryLifecycleRow.memory_id == MemoryRow.memory_id,
            )
            .where(
                SessionRow.user_id == query.user_id,
                MemoryLifecycleRow.status == Status.ACTIVE.value,
                or_(
                    MemoryRow.session_applicability.is_(None),
                    MemoryRow.session_applicability == query.session_id,
                ),
                or_(
                    MemoryRow.valid_from.is_(None),
                    MemoryRow.valid_from <= query.as_of,
                ),
                or_(
                    MemoryRow.valid_to.is_(None),
                    MemoryRow.valid_to >= query.as_of,
                ),
            )
            .order_by(
                score.desc(),
                fused.c.base_score.desc(),
                MemoryRow.created_at.desc(),
                MemoryRow.memory_id,
            )
            .limit(query.limit)
        )
        rows = self._session.execute(statement).all()
        retrieved: list[RetrievedMemory] = []
        for rank, result in enumerate(rows, start=1):
            reasons: list[str] = []
            if result.dense_match:
                reasons.append("dense")
            if result.lexical_match:
                reasons.append("lexical")
            reasons.append("lifecycle")
            retrieved.append(
                RetrievedMemory(
                    memory=_to_domain(result[0]),
                    rank=rank,
                    score=float(result.score),
                    retention=float(result.retention),
                    retrieval_reasons=tuple(reasons),
                )
            )
        return tuple(retrieved)
