from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem import LLMMemoryExtractor, LLMTaskSettings
from fluxmem.application.ports.llm import StructuredModelResponse
from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.message import Message


class _Provider:
    def __init__(self, output: dict[str, object]) -> None:
        self._output = output
        self.calls: list[dict[str, object]] = []

    def generate(self, **values) -> StructuredModelResponse:
        self.calls.append(values)
        return StructuredModelResponse(
            output_text=json.dumps(self._output),
            model="test-model",
        )


class MemoryLLMTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.created_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
        self.message = Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id="Alice",
            content="I prefer tea.",
            created_at=self.created_at,
        )
        self.history = MessagePack(
            user_id=self.user_id,
            session_id=self.session_id,
            messages=(self.message,),
        )

    def test_extractor_preserves_duplicate_candidates(self) -> None:
        raw_candidate = {
            "content": "Alice prefers tea.",
            "source_message_ref": 1,
            "session_limited": False,
            "valid_from": None,
            "valid_to": None,
        }
        provider = _Provider(
            {"memories": [raw_candidate, raw_candidate.copy()]}
        )
        extractor = LLMMemoryExtractor(
            provider=provider,
            settings=LLMTaskSettings(
                model="test-model",
                repair_invalid_output=False,
            ),
        )
        pack = MemoryPack(
            query_id=uuid4(),
            user_id=self.user_id,
            session_id=self.session_id,
            memories=(),
        )

        candidates = extractor.extract(
            target_messages=(self.message,),
            session_history=self.history,
            memory_pack=pack,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0], candidates[1])



if __name__ == "__main__":
    unittest.main()
