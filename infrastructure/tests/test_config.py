from __future__ import annotations

import unittest

from fluxmem_infrastructure.config import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OPENROUTER_RERANKER_MODEL,
    InfrastructureSettings,
)
from fluxmem_infrastructure.locomo.cli import _LOCOMO_RETRIEVAL_SETTINGS


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
        self.assertEqual(
            DEFAULT_OPENROUTER_RERANKER_MODEL,
            "voyageai/rerank-2.5",
        )
        self.assertEqual(
            settings.reranker_model,
            DEFAULT_OPENROUTER_RERANKER_MODEL,
        )
        self.assertEqual(settings.extraction_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.lifecycle_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.memory_write_history_messages, 20)
        self.assertEqual(
            settings.memory_context_settings().maximum_write_history_messages,
            20,
        )

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

    def test_reranker_model_can_be_overridden(self) -> None:
        settings = InfrastructureSettings.from_environment(
            {
                "FLUXMEM_DATABASE_URL": (
                    "postgresql+psycopg://localhost/fluxmem"
                ),
                "OPENROUTER_API_KEY": "test-key",
                "FLUXMEM_RERANKER_MODEL": "reranker-model",
            }
        )

        self.assertEqual(settings.reranker_model, "reranker-model")
        self.assertEqual(
            settings.reranker_settings().maximum_candidates,
            50,
        )

    def test_locomo_retrieval_uses_only_the_benchmark_question(self) -> None:
        self.assertEqual(_LOCOMO_RETRIEVAL_SETTINGS.max_query_messages, 1)

    def test_extractor_history_window_can_be_overridden(self) -> None:
        settings = InfrastructureSettings.from_environment(
            {
                "FLUXMEM_DATABASE_URL": "postgresql+psycopg://localhost/fluxmem",
                "OPENROUTER_API_KEY": "test-key",
                "FLUXMEM_MEMORY_WRITE_HISTORY_MESSAGES": "24",
            }
        )

        self.assertEqual(settings.memory_write_history_messages, 24)
        self.assertEqual(
            settings.memory_context_settings().maximum_write_history_messages,
            24,
        )

if __name__ == "__main__":
    unittest.main()
