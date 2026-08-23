from __future__ import annotations

import unittest

from fluxmem.local import (
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    LocalRuntimeSettings,
    load_local_settings,
)


class LocalRuntimeSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "FLUXMEM_DATABASE_URL": "postgresql+psycopg://localhost/fluxmem",
            "OPENROUTER_API_KEY": "openrouter-secret",
        }

    def test_defaults_to_requested_models_for_all_tasks(self) -> None:
        settings = LocalRuntimeSettings.from_environment(self.environment)

        self.assertEqual(settings.answer_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.extraction_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.reconciliation_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.lifecycle_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(
            settings.embedding_model,
            DEFAULT_OPENROUTER_EMBEDDING_MODEL,
        )
        llm_settings = settings.llm_settings()
        self.assertEqual(llm_settings.answer.model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(llm_settings.answer.timeout_seconds, 60)
        self.assertEqual(
            llm_settings.context.maximum_history_messages,
            128,
        )
        self.assertEqual(
            llm_settings.context.maximum_answer_input_tokens,
            64_000,
        )

    def test_task_specific_model_overrides_shared_model(self) -> None:
        environment = {
            **self.environment,
            "FLUXMEM_LLM_MODEL": "shared-model",
            "FLUXMEM_ANSWER_MODEL": "answer-model",
            "FLUXMEM_ANSWER_HISTORY_MESSAGES": "96",
            "FLUXMEM_ANSWER_INPUT_TOKEN_BUDGET": "48000",
            "FLUXMEM_ENABLE_MEMORY_WRITES": "false",
        }

        settings = load_local_settings(
            dotenv_path=None,
            environ=environment,
        )

        self.assertEqual(settings.answer_model, "answer-model")
        self.assertEqual(settings.extraction_model, "shared-model")
        self.assertEqual(
            settings.llm_settings().context.maximum_history_messages,
            96,
        )
        self.assertEqual(
            settings.llm_settings().context.maximum_answer_input_tokens,
            48_000,
        )
        self.assertFalse(settings.llm_settings().enable_memory_writes)

    def test_secret_values_are_excluded_from_repr(self) -> None:
        settings = LocalRuntimeSettings.from_environment(self.environment)

        rendered = repr(settings)

        self.assertNotIn("openrouter-secret", rendered)

    def test_missing_api_key_fails_before_clients_are_built(self) -> None:
        environment = dict(self.environment)
        del environment["OPENROUTER_API_KEY"]

        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            LocalRuntimeSettings.from_environment(environment)

    def test_invalid_boolean_is_rejected(self) -> None:
        environment = {
            **self.environment,
            "FLUXMEM_OPENROUTER_PROMPT_CACHING": "sometimes",
        }

        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            LocalRuntimeSettings.from_environment(environment)


if __name__ == "__main__":
    unittest.main()
