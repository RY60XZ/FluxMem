from __future__ import annotations

import json
import unittest

from fluxmem import StructuredModelResponse
from fluxmem_infrastructure.answering import AnswerModelSettings
from fluxmem_infrastructure.locomo.judge import (
    LLMLocomoJudge,
    LocomoJudgeLabel,
)


class _Provider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **values):
        self.calls.append(values)
        return StructuredModelResponse(
            output_text=json.dumps(
                {
                    "reasoning": "The answers identify the same place.",
                    "label": "CORRECT",
                }
            ),
            model=str(values["model"]),
            response_id="judge-1",
        )


class LocomoJudgeTests(unittest.TestCase):
    def test_uses_answer_model_and_mem0_open_domain_preprocessing(self) -> None:
        provider = _Provider()
        judge = LLMLocomoJudge(
            provider=provider,
            settings=AnswerModelSettings(model="same-answer-model"),
        )

        judgment = judge.judge(
            category=3,
            question="Where was the trip?",
            ground_truth="Toronto; inferred from the conversation",
            prediction="Toronto",
        )

        self.assertIs(judgment.label, LocomoJudgeLabel.CORRECT)
        self.assertEqual(judgment.score, 1.0)
        self.assertEqual(provider.calls[0]["model"], "same-answer-model")
        prompt = str(provider.calls[0]["input_text"])
        self.assertIn("Gold answer: Toronto\n", prompt)
        self.assertNotIn("inferred from the conversation", prompt)


if __name__ == "__main__":
    unittest.main()
