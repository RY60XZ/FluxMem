from __future__ import annotations

import unittest

from fluxmem_infrastructure.config import (
    DEFAULT_OPENROUTER_MODEL,
    InfrastructureSettings,
)


class InfrastructureSettingsTests(unittest.TestCase):
    def test_paid_gemma_4_is_default_for_answering_and_llm_tasks(self) -> None:
        settings = InfrastructureSettings.from_environment(
            {
                "FLUXMEM_DATABASE_URL": "postgresql+psycopg://localhost/fluxmem",
                "OPENROUTER_API_KEY": "test-key",
            }
        )

        self.assertEqual(
            DEFAULT_OPENROUTER_MODEL,
            "google/gemma-4-26b-a4b-it",
        )
        self.assertEqual(settings.answer_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.judge_model, settings.answer_model)
        self.assertEqual(settings.extraction_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.lifecycle_model, DEFAULT_OPENROUTER_MODEL)

    def test_judge_model_defaults_to_answer_model_and_can_be_overridden(
        self,
    ) -> None:
        base = {
            "FLUXMEM_DATABASE_URL": "postgresql+psycopg://localhost/fluxmem",
            "OPENROUTER_API_KEY": "test-key",
            "FLUXMEM_ANSWER_MODEL": "answer-model",
        }

        shared = InfrastructureSettings.from_environment(base)
        separate = InfrastructureSettings.from_environment(
            {**base, "FLUXMEM_JUDGE_MODEL": "judge-model"}
        )

        self.assertEqual(shared.judge_model, "answer-model")
        self.assertEqual(separate.judge_model, "judge-model")
        self.assertEqual(
            separate.judge_model_settings().model,
            "judge-model",
        )

if __name__ == "__main__":
    unittest.main()
