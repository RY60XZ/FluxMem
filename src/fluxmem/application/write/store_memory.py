from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from fluxmem.application.errors import (
    InvalidSessionApplicabilityError,
    MessageNotFoundError,
)
from fluxmem.application.lifecycle import (
    LifecyclePolicyExecutor,
    RuleLifecycleEvaluator,
)
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.lifecycle import LifecycleEvaluator
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.lifecycle import MemoryLifecycle
from fluxmem.domain.memory import Memory


class StoreMemory:
    """Persist one already-reconciled memory within its ownership boundaries."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        lifecycle_evaluator: LifecycleEvaluator | None = None,
        lifecycle_policy: LifecyclePolicyExecutor | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._lifecycle_evaluator = lifecycle_evaluator or RuleLifecycleEvaluator()
        self._fallback_evaluator = RuleLifecycleEvaluator()
        self._lifecycle_policy = lifecycle_policy or LifecyclePolicyExecutor()
        self._clock = clock or SystemClock()

    def execute(self, *, user_id: UUID, memory: Memory) -> UUID:
        with self._unit_of_work_factory() as unit_of_work:
            message = unit_of_work.messages.get(
                message_id=memory.message_id,
                user_id=user_id,
            )
            if message is None:
                raise MessageNotFoundError(
                    "origin message does not belong to the requested user"
                )

            if memory.session_applicability not in (None, message.session_id):
                raise InvalidSessionApplicabilityError(
                    "session-limited memory must use its origin message's session"
                )

            initialized_at = self._clock.now()
            lifecycle = self._initial_lifecycle(
                memory=memory,
                source_role=message.role,
                initialized_at=initialized_at,
            )

            unit_of_work.memories.add(memory=memory)
            unit_of_work.flush()
            unit_of_work.lifecycles.add(lifecycle=lifecycle)
            unit_of_work.commit()

        return memory.memory_id

    def _initial_lifecycle(
        self,
        *,
        memory: Memory,
        source_role: str,
        initialized_at: datetime,
    ) -> MemoryLifecycle:
        """Use the rule evaluator when a configured semantic evaluator fails."""

        try:
            decision = self._lifecycle_evaluator.evaluate(
                memory=memory,
                source_role=source_role,
                evaluated_at=initialized_at,
            )
            return self._lifecycle_policy.initialize(
                memory=memory,
                decision=decision,
                initialized_at=initialized_at,
            )
        except Exception:
            if isinstance(self._lifecycle_evaluator, RuleLifecycleEvaluator):
                raise
            decision = self._fallback_evaluator.evaluate(
                memory=memory,
                source_role=source_role,
                evaluated_at=initialized_at,
            )
            return self._lifecycle_policy.initialize(
                memory=memory,
                decision=decision,
                initialized_at=initialized_at,
            )
