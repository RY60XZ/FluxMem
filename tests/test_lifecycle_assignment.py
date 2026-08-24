from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem import LLMTaskSettings, LLMLifecycleEvaluator
from fluxmem.application.lifecycle import LifecycleAssigner
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.ports.lifecycle import LifecycleEvaluationInput
from fluxmem.application.ports.llm import StructuredModelResponse
from fluxmem.domain.lifecycle import DecisionSource, Tier
from fluxmem.domain.memory import Memory


class _Provider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **values) -> StructuredModelResponse:
        self.calls += 1
        payload = json.loads(values["input_text"])
        if len(payload["memories"]) != 2:
            raise AssertionError("lifecycle request must contain the full batch")
        return StructuredModelResponse(
            output_text=json.dumps(
                {
                    "decisions": [
                        {
                            "memory_ref": 1,
                            "importance": 0.9,
                            "reason_codes": ["durable"],
                            "confidence": 0.9,
                        },
                        {
                            "memory_ref": 2,
                            "importance": 2.0,
                            "reason_codes": ["invalid"],
                            "confidence": 0.9,
                        },
                    ]
                }
            ),
            model="test-model",
        )


class LifecycleAssignmentTests(unittest.TestCase):
    def test_one_model_call_with_per_memory_deterministic_fallback(self) -> None:
        provider = _Provider()
        evaluator = LLMLifecycleEvaluator(
            provider=provider,
            settings=LLMTaskSettings(
                model="test-model",
                repair_invalid_output=False,
            ),
        )
        evaluated_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
        items = tuple(
            LifecycleEvaluationInput(
                memory=Memory(
                    memory_id=uuid4(),
                    message_id=uuid4(),
                    content=content,
                    created_at=evaluated_at,
                ),
                source_role="user",
            )
            for content in ("A durable preference", "An invalid decision")
        )
        usage = ModelUsageCollector()

        lifecycles = LifecycleAssigner(evaluator=evaluator).assign(
            items=items,
            evaluated_at=evaluated_at,
            usage_recorder=usage,
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(usage.snapshot().calls), 1)
        self.assertEqual(lifecycles[0].importance, 0.9)
        self.assertEqual(lifecycles[0].tier, Tier.LONG_TERM)
        self.assertEqual(
            lifecycles[0].decision_source,
            DecisionSource.LLM_PRIMARY,
        )
        self.assertEqual(lifecycles[1].importance, 0.65)
        self.assertEqual(lifecycles[1].tier, Tier.SHORT_TERM)
        self.assertEqual(
            lifecycles[1].decision_source,
            DecisionSource.RULES_FALLBACK,
        )


if __name__ == "__main__":
    unittest.main()
