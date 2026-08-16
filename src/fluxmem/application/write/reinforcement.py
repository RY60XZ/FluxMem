from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from fluxmem.application.errors import SessionNotFoundError
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.info_pack import FeedbackPack, MemoryUsage, UsageType
from fluxmem.domain.lifecycle import MemoryLifecycle


class ReinforceMemory:
    """Apply actual-use feedback to active memories in one transaction."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        clock: Clock | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._clock = clock or SystemClock()

    def execute(
        self,
        *,
        feedback_pack: FeedbackPack,
    ) -> tuple[MemoryLifecycle, ...]:
        used_at = self._clock.now()
        usages = _strongest_usage_per_memory(feedback_pack.used_memories)

        with self._unit_of_work_factory() as unit_of_work:
            if not unit_of_work.sessions.is_owned_by(
                session_id=feedback_pack.session_id,
                user_id=feedback_pack.user_id,
            ):
                raise SessionNotFoundError(
                    "feedback session does not belong to the requested user"
                )

            reinforced: list[MemoryLifecycle] = []
            for usage in usages:
                lifecycle = unit_of_work.lifecycles.reinforce(
                    memory_id=usage.memory_id,
                    query_id=feedback_pack.query_id,
                    user_id=feedback_pack.user_id,
                    session_id=feedback_pack.session_id,
                    usage=usage,
                    used_at=used_at,
                )
                if lifecycle is None:
                    raise LookupError(
                        "feedback memory is not active in the requested scope"
                    )
                reinforced.append(lifecycle)

            unit_of_work.commit()

        return tuple(reinforced)


def _strongest_usage_per_memory(
    usages: tuple[MemoryUsage, ...],
) -> tuple[MemoryUsage, ...]:
    """Avoid reinforcing twice when attribution implies context inclusion."""

    priority = {
        UsageType.CONTEXT_INCLUDED: 0,
        UsageType.MODEL_ATTRIBUTED: 1,
    }
    strongest: dict[UUID, MemoryUsage] = {}
    for usage in usages:
        if usage.rank is not None and usage.rank < 1:
            raise ValueError("usage rank must be positive")
        if usage.contribution is not None and not 0.0 <= usage.contribution <= 1.0:
            raise ValueError("usage contribution must be between 0 and 1")

        previous = strongest.get(usage.memory_id)
        if (
            previous is None
            or priority[usage.usage_type] > priority[previous.usage_type]
        ):
            strongest[usage.memory_id] = usage

    return tuple(strongest.values())
