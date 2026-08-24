from __future__ import annotations

import unittest

from fluxmem_infrastructure.config import (
    DEFAULT_OPENROUTER_MODEL,
    InfrastructureSettings,
)


class InfrastructureSettingsTests(unittest.TestCase):
    def test_deepseek_v4_flash_is_default_for_answering_and_llm_tasks(self) -> None:
        settings = InfrastructureSettings.from_environment(
            {
                "FLUXMEM_DATABASE_URL": "postgresql+psycopg://localhost/fluxmem",
                "OPENROUTER_API_KEY": "test-key",
            }
        )

        self.assertEqual(
            DEFAULT_OPENROUTER_MODEL,
            "deepseek/deepseek-v4-flash-0731",
        )
        self.assertEqual(settings.answer_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.extraction_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.reconciliation_model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(settings.lifecycle_model, DEFAULT_OPENROUTER_MODEL)


if __name__ == "__main__":
    unittest.main()
