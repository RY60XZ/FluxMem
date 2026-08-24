from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fluxmem_infrastructure.locomo.dataset import (
    load_dataset,
    select_conversations,
)


class LocomoDatasetTests(unittest.TestCase):
    def test_loads_named_speakers_timestamp_caption_and_numeric_answer(self) -> None:
        payload = [
            {
                "sample_id": "conv-test",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {
                            "speaker": "Alice",
                            "text": "Look at this.",
                            "blip_caption": "a red bicycle",
                            "query": "red bicycle",
                        },
                        {
                            "speaker": "Bob",
                            "text": "Nice bicycle.",
                        },
                    ],
                },
                "qa": [
                    {
                        "question": "How many bicycles?",
                        "answer": 1,
                        "category": 2,
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            conversations = load_dataset(path)

        conversation = conversations[0]
        self.assertEqual(conversation.questions[0].answer, "1")
        self.assertEqual(
            conversation.sessions[0].turns[0].content,
            (
                "Look at this. [Sharing image - query: red bicycle. "
                "The image shows: a red bicycle]"
            ),
        )
        self.assertLess(
            conversation.sessions[0].turns[0].occurred_at,
            conversation.sessions[0].turns[1].occurred_at,
        )

    def test_single_selection_uses_sample_id(self) -> None:
        conversation = _minimal_conversation_payload("conv-26")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps([conversation]), encoding="utf-8")
            loaded = load_dataset(path)
        selected = select_conversations(
            loaded,
            run_all=False,
            sample_id="conv-26",
        )
        self.assertEqual([item.sample_id for item in selected], ["conv-26"])

    def test_full_selection_requires_exactly_ten_conversations(self) -> None:
        conversation = _minimal_conversation_payload("conv-26")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps([conversation]), encoding="utf-8")
            loaded = load_dataset(path)
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            select_conversations(loaded, run_all=True, sample_id=None)


def _minimal_conversation_payload(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "text": "Hello"}
            ],
        },
        "qa": [
            {
                "question": "Who spoke?",
                "answer": "Alice",
                "category": 2,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
