from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem import LLMMemoryExtractor, LLMTaskSettings
from fluxmem.application.ports.llm import StructuredModelResponse
from fluxmem.domain.info_pack import MemoryPack, MessagePack, RetrievedMemory
from fluxmem.domain.memory import Memory
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
            content="Alice: I prefer tea.",
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
            memories=(
                RetrievedMemory(
                    memory=Memory(
                        memory_id=uuid4(),
                        message_id=uuid4(),
                        content="Alice previously preferred coffee.",
                        created_at=self.created_at,
                    ),
                    rank=1,
                    score=0.9,
                    retention=1.0,
                    retrieval_reasons=("lexical",),
                ),
            ),
        )

        candidates = extractor.extract(
            target_messages=(self.message,),
            session_history=self.history,
            memory_pack=pack,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0], candidates[1])
        input_text = str(provider.calls[0]["input_text"])
        self.assertNotIn('"speaker"', input_text)
        self.assertIn('"role":"user"', input_text)
        self.assertIn("Alice: I prefer tea.", input_text)
        self.assertIn("Alice previously preferred coffee.", input_text)

    def test_default_extractor_context_includes_twenty_messages(self) -> None:
        messages = tuple(
            Message(
                message_id=uuid4(),
                session_id=self.session_id,
                role="user",
                agent_id="Alice",
                content=f"Alice: message-{index}",
                created_at=self.created_at,
            )
            for index in range(21)
        )
        provider = _Provider({"memories": []})
        extractor = LLMMemoryExtractor(
            provider=provider,
            settings=LLMTaskSettings(
                model="test-model",
                repair_invalid_output=False,
            ),
        )

        extractor.extract(
            target_messages=(messages[-1],),
            session_history=MessagePack(
                user_id=self.user_id,
                session_id=self.session_id,
                messages=messages,
            ),
            memory_pack=MemoryPack(
                query_id=uuid4(),
                user_id=self.user_id,
                session_id=self.session_id,
                memories=(),
            ),
        )

        input_text = str(provider.calls[0]["input_text"])
        conversation = input_text.split("\n\nMEMORIES:", maxsplit=1)[0]
        self.assertNotIn("Alice: message-0", conversation)
        self.assertIn("Alice: message-1", conversation)
        self.assertIn("Alice: message-20", conversation)
        self.assertEqual(conversation.count('"message_ref"'), 20)


if __name__ == "__main__":
    unittest.main()
