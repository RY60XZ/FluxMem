from __future__ import annotations

import unittest

from fluxmem_infrastructure.locomo.scoring import (
    evidence_recall,
    score_answer,
    token_f1,
)


class LocomoScoringTests(unittest.TestCase):
    def test_normalized_stemmed_token_f1(self) -> None:
        self.assertEqual(token_f1("The dogs running.", "dog run"), 1.0)

    def test_multi_answer_uses_partial_matching(self) -> None:
        self.assertEqual(
            score_answer(
                prediction="Toronto, tea",
                ground_truth="tea, Toronto",
                category=1,
            ),
            1.0,
        )

    def test_temporal_ground_truth_uses_first_semicolon_variant(self) -> None:
        self.assertEqual(
            score_answer(
                prediction="7 May 2023",
                ground_truth="7 May 2023; May 7, 2023",
                category=3,
            ),
            1.0,
        )

    def test_adversarial_requires_official_unanswerable_phrase(self) -> None:
        self.assertEqual(
            score_answer(
                prediction="That is not mentioned in the conversation.",
                ground_truth=None,
                category=5,
            ),
            1.0,
        )
        self.assertEqual(
            score_answer(
                prediction="I do not know.",
                ground_truth=None,
                category=5,
            ),
            0.0,
        )

    def test_evidence_recall_matches_dialogue_ids(self) -> None:
        self.assertEqual(
            evidence_recall(
                expected=("D1:1", "D2:3"),
                retrieved=("D2:3",),
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
