from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fluxmem import (
    MemoryPack,
    MemoryRetrievalResult,
    Message,
    MessageIngestionResult,
    RetrievedMemory,
    Session,
)
from fluxmem.domain.info_pack import TurnMemoryPacks
from fluxmem.domain.memory import Memory
from fluxmem_infrastructure.answering import AnswerResult
from fluxmem_infrastructure.locomo.dataset import (
    LocomoConversation,
    LocomoQuestion,
    LocomoSession,
    LocomoTurn,
)
from fluxmem_infrastructure.locomo.runner import LocomoRunner


class _Memory:
    def __init__(self) -> None:
        self.messages_by_user: dict[UUID, list[Message]] = {}

    def start_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID | None = None,
    ) -> Session:
        assert session_id is not None
        self.messages_by_user[user_id] = []
        return Session(session_id=session_id, user_id=user_id)

    def ingest_messages(self, *, user_id, messages, diagnostics=False):
        del diagnostics
        self.messages_by_user[user_id].extend(messages)
        return MessageIngestionResult(messages=tuple(messages))


class _Answering:
    def __init__(self, memory: _Memory) -> None:
        self._memory = memory

    def answer(
        self,
        *,
        user_id,
        session_id,
        content,
        created_at=None,
        retrieval_limit=None,
    ):
        del retrieval_limit
        source = self._memory.messages_by_user[user_id][0]
        memory_id = uuid4()
        memory = Memory(
            memory_id=memory_id,
            message_id=source.message_id,
            content="Alice lives in Toronto.",
            created_at=source.created_at,
        )
        pack = MemoryPack(
            query_id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            memories=(
                RetrievedMemory(
                    memory=memory,
                    rank=1,
                    score=1.0,
                    retention=1.0,
                    retrieval_reasons=("lexical",),
                ),
            ),
        )
        query = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="user",
            agent_id=None,
            content=content,
            created_at=created_at,
        )
        answer = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="assistant",
            agent_id="test-answerer",
            content="Toronto",
            created_at=created_at,
        )
        retrieval = MemoryRetrievalResult(
            query=query,
            retrieval=TurnMemoryPacks(seeds=pack, expanded=pack),
        )
        return AnswerResult(
            answer=answer,
            retrieval=retrieval,
            context_memory_ids=(memory_id,),
            model="test-answerer",
            response_id=None,
            usage=None,
        )


class LocomoRunnerTests(unittest.TestCase):
    def test_full_mode_writes_exactly_ten_conversation_artifacts(self) -> None:
        memory = _Memory()
        conversations = tuple(
            _conversation(f"conv-{index}") for index in range(10)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            summary = LocomoRunner(
                memory=memory,
                answering=_Answering(memory),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="all",
                model_settings={"answer": "test"},
                run_id=UUID("00000000-0000-0000-0000-000000000001"),
            ).run(conversations)
            artifacts = tuple((output / "conversations").glob("*.json"))

        self.assertEqual(summary["conversation_count"], 10)
        self.assertEqual(summary["question_count"], 10)
        self.assertEqual(summary["answer_f1"], 1.0)
        self.assertEqual(summary["evidence_recall"], 1.0)
        self.assertEqual(len(artifacts), 10)
        self.assertEqual(
            {messages[0].agent_id for messages in memory.messages_by_user.values()},
            {"Alice"},
        )

    def test_single_mode_uses_the_same_conversation_pipeline(self) -> None:
        memory = _Memory()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = LocomoRunner(
                memory=memory,
                answering=_Answering(memory),
                output_dir=root / "results",
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                model_settings={"answer": "test"},
            ).run((_conversation("conv-26"),))

        self.assertEqual(summary["conversation_count"], 1)
        self.assertEqual(summary["question_count"], 1)
        self.assertEqual(summary["failed_conversations"], [])


def _conversation(sample_id: str) -> LocomoConversation:
    occurred_at = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    return LocomoConversation(
        sample_id=sample_id,
        speaker_a="Alice",
        speaker_b="Bob",
        sessions=(
            LocomoSession(
                number=1,
                occurred_at=occurred_at,
                turns=(
                    LocomoTurn(
                        dia_id="D1:1",
                        speaker="Alice",
                        text="I live in Toronto.",
                        occurred_at=occurred_at,
                    ),
                ),
            ),
        ),
        questions=(
            LocomoQuestion(
                question="Where does Alice live?",
                answer="Toronto",
                category=2,
                evidence=("D1:1",),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
