from __future__ import annotations

import unittest

import fluxmem


class PublicBoundaryTests(unittest.TestCase):
    def test_core_exposes_memory_operations_without_answering_runtime(self) -> None:
        self.assertTrue(hasattr(fluxmem.FluxMem, "retrieve"))
        self.assertTrue(hasattr(fluxmem.FluxMem, "store_messages"))
        for name in (
            "AnswerGenerator",
            "bootstrap_from_env",
            "query",
            "run_turn",
            "stream_turn",
        ):
            self.assertFalse(hasattr(fluxmem, name), name)

    def test_default_retrieval_limit_is_fifty(self) -> None:
        self.assertEqual(fluxmem.MemoryLayerSettings().retrieval_limit, 50)


if __name__ == "__main__":
    unittest.main()
