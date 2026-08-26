from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fluxmem import (
    LLMUsageReport,
    MemoryPack,
    MemoryRetrievalResult,
    Message,
    MessageIngestionResult,
    ModelTokenUsage,
    RetrievedMemory,
    Session,
)
from fluxmem.domain.llm import LLMTaskKind, ModelCallUsage
from fluxmem.domain.memory import Memory
from fluxmem_infrastructure.answering import AnswerResult
from fluxmem_infrastructure.locomo.dataset import (
    LocomoConversation,
    LocomoQuestion,
    LocomoSession,
    LocomoTurn,
)
from fluxmem_infrastructure.locomo.judge import (
    LocomoJudgment,
    LocomoJudgeLabel,
)
from fluxmem_infrastructure.locomo.runner import LocomoRunner


class _Memory:
    def __init__(self, *, cancel_on_ingest_call: int | None = None) -> None:
        self.messages_by_user: dict[UUID, list[Message]] = {}
        self.cancel_on_ingest_call = cancel_on_ingest_call
        self.ingest_call_count = 0
        self.store_call_count = 0
        self.ingested_message_ids: list[UUID] = []

    def start_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID | None = None,
    ) -> Session:
        assert session_id is not None
        self.messages_by_user.setdefault(user_id, [])
        return Session(session_id=session_id, user_id=user_id)

    def ingest_messages(self, *, user_id, messages, diagnostics=False):
        del diagnostics
        self.ingest_call_count += 1
        for message in messages:
            if message.message_id not in self.ingested_message_ids:
                self.messages_by_user[user_id].append(message)
                self.ingested_message_ids.append(message.message_id)
        if self.ingest_call_count == self.cancel_on_ingest_call:
            raise KeyboardInterrupt
        return MessageIngestionResult(
            messages=tuple(messages),
            llm_usage=LLMUsageReport(
                calls=(
                    ModelCallUsage(
                        task=LLMTaskKind.EXTRACTION,
                        model="test-extractor",
                        attempt=1,
                        token_usage=ModelTokenUsage(
                            input_tokens=2,
                            output_tokens=1,
                            total_tokens=3,
                        ),
                    ),
                )
            ),
        )

    def store_messages(self, *, user_id, messages):
        self.store_call_count += 1
        stored: list[UUID] = []
        for message in messages:
            if message.message_id not in self.ingested_message_ids:
                self.messages_by_user[user_id].append(message)
                self.ingested_message_ids.append(message.message_id)
            stored.append(message.message_id)
        return tuple(stored)


class _Answering:
    def __init__(
        self,
        memory: _Memory,
        *,
        cancel_on_call: int | None = None,
    ) -> None:
        self._memory = memory
        self.cancel_on_call = cancel_on_call
        self.calls: list[str] = []
        self.created_ats: list[datetime | None] = []

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
        self.calls.append(content)
        self.created_ats.append(created_at)
        if len(self.calls) == self.cancel_on_call:
            raise KeyboardInterrupt
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
            content="Reasoning\nANSWER: Toronto",
            created_at=created_at,
        )
        retrieval = MemoryRetrievalResult(
            query=query,
            context=pack,
        )
        return AnswerResult(
            answer=answer,
            retrieval=retrieval,
            context_memory_ids=(memory_id,),
            model="test-answerer",
            response_id=None,
            usage=ModelTokenUsage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
        )


class _Judge:
    def judge(self, **values):
        del values
        return LocomoJudgment(
            label=LocomoJudgeLabel.CORRECT,
            reasoning="The prediction matches the reference answer.",
            model="test-answerer",
            response_id="judge-response",
            usage=ModelTokenUsage(
                input_tokens=4,
                output_tokens=2,
                total_tokens=6,
            ),
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
                judge=_Judge(),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="all",
                model_settings={"answer": "test", "judge": "test"},
                run_id=UUID("00000000-0000-0000-0000-000000000001"),
            ).run(conversations)
            artifacts = tuple((output / "conversations").glob("*.json"))
            artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))

        self.assertEqual(summary["conversation_count"], 10)
        self.assertEqual(summary["question_count"], 10)
        self.assertEqual(summary["metrics"]["overall"]["accuracy"], 100.0)
        self.assertEqual(
            {
                stage: values["average_tokens_per_operation"]["total"]
                for stage, values in summary["token_usage"].items()
            },
            {"ingestion": 3.0, "answer": 12.0, "judge": 6.0},
        )
        self.assertEqual(len(artifacts), 10)
        self.assertEqual(
            {
                tuple(
                    (message.role, message.agent_id, message.content)
                    for message in messages
                )
                for messages in memory.messages_by_user.values()
            },
            {
                (
                    ("user", "Alice", "Alice: I live in Toronto."),
                    ("assistant", "Bob", "Bob: That sounds great."),
                )
            },
        )
        question = artifact["question_results"][0]
        self.assertEqual(question["prediction"], "Toronto")
        self.assertEqual(question["answer"], "Toronto")
        self.assertEqual(question["judgment"]["label"], "CORRECT")
        self.assertEqual(question["judgment"]["model"], "test-answerer")
        self.assertEqual(
            question["retrieval"]["memories"][0]["content"],
            "Alice lives in Toronto.",
        )

    def test_single_mode_uses_the_same_conversation_pipeline(self) -> None:
        memory = _Memory()
        conversation = _conversation_with_second_session("conv-26")
        answering = _Answering(memory)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = LocomoRunner(
                memory=memory,
                answering=answering,
                judge=_Judge(),
                output_dir=root / "results",
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                model_settings={"answer": "test", "judge": "test"},
            ).run((conversation,))

        self.assertEqual(summary["conversation_count"], 1)
        self.assertEqual(summary["question_count"], 1)
        self.assertEqual(summary["failed_conversations"], [])
        self.assertEqual(
            answering.created_ats,
            [conversation.turns[-1].occurred_at],
        )

    def test_full_context_mode_stores_transcript_without_memory_learning(
        self,
    ) -> None:
        memory = _Memory()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            summary = LocomoRunner(
                memory=memory,
                answering=_Answering(memory),
                judge=_Judge(),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                context_mode="full",
                model_settings={"answer": "test", "judge": "test"},
            ).run((_conversation("conv-30"),))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            artifact = json.loads(
                (output / "conversations" / "conv-30.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(memory.ingest_call_count, 0)
        self.assertEqual(memory.store_call_count, 2)
        self.assertEqual(len(memory.ingested_message_ids), 2)
        self.assertEqual(manifest["context_mode"], "full")
        self.assertEqual(summary["context_mode"], "full")
        self.assertEqual(artifact["context_mode"], "full")
        self.assertEqual(artifact["ingestion"]["usage"]["call_count"], 0)
        self.assertEqual(artifact["ingestion"]["memory_outcome_counts"], {})

    def test_resume_skips_checkpointed_ingestion_turns(self) -> None:
        memory = _Memory(cancel_on_ingest_call=2)
        conversation = _conversation("conv-30")
        run_id = UUID("00000000-0000-0000-0000-000000000030")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            runner = LocomoRunner(
                memory=memory,
                answering=_Answering(memory),
                judge=_Judge(),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                model_settings={"answer": "test", "judge": "test"},
                run_id=run_id,
            )
            with self.assertRaises(KeyboardInterrupt):
                runner.run((conversation,))

            checkpoint = json.loads(
                (output / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint["conversations"]["conv-30"][
                    "completed_turn_count"
                ],
                1,
            )

            memory.cancel_on_ingest_call = None
            summary = LocomoRunner(
                memory=memory,
                answering=_Answering(memory),
                judge=_Judge(),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                model_settings={"answer": "test", "judge": "test"},
                resume=True,
            ).run((conversation,))

            checkpoint = json.loads(
                (output / "checkpoint.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["failed_conversations"], [])
        self.assertTrue(checkpoint["conversations"]["conv-30"]["complete"])
        self.assertEqual(len(memory.ingested_message_ids), 2)
        self.assertEqual(len(set(memory.ingested_message_ids)), 2)

    def test_resume_skips_checkpointed_questions(self) -> None:
        memory = _Memory()
        conversation = _conversation_with_two_questions("conv-30")
        answering = _Answering(memory, cancel_on_call=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            runner = LocomoRunner(
                memory=memory,
                answering=answering,
                judge=_Judge(),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                model_settings={"answer": "test", "judge": "test"},
            )
            with self.assertRaises(KeyboardInterrupt):
                runner.run((conversation,))

            answering.cancel_on_call = None
            summary = LocomoRunner(
                memory=memory,
                answering=answering,
                judge=_Judge(),
                output_dir=output,
                dataset_path=root / "locomo.json",
                dataset_sha256="fixture",
                mode="single",
                model_settings={"answer": "test", "judge": "test"},
                resume=True,
            ).run((conversation,))
            artifact = json.loads(
                (output / "conversations" / "conv-30.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(summary["question_count"], 2)
        self.assertEqual(len(artifact["question_results"]), 2)
        self.assertEqual(
            answering.calls.count("Where does Alice live?"),
            1,
        )


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
                        speaker="Alice",
                        text="I live in Toronto.",
                        occurred_at=occurred_at,
                    ),
                    LocomoTurn(
                        speaker="Bob",
                        text="That sounds great.",
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
            ),
            LocomoQuestion(
                question="What did Alice not say?",
                category=5,
                adversarial_answer="She did not discuss Montreal.",
            ),
        ),
    )


def _conversation_with_two_questions(sample_id: str) -> LocomoConversation:
    base = _conversation(sample_id)
    return LocomoConversation(
        sample_id=base.sample_id,
        speaker_a=base.speaker_a,
        speaker_b=base.speaker_b,
        sessions=base.sessions,
        questions=(
            base.questions[0],
            LocomoQuestion(
                question="Which city is Alice based in?",
                answer="Toronto",
                category=4,
            ),
        ),
    )


def _conversation_with_second_session(sample_id: str) -> LocomoConversation:
    base = _conversation(sample_id)
    occurred_at = datetime(2023, 5, 10, 9, 30, tzinfo=timezone.utc)
    return LocomoConversation(
        sample_id=base.sample_id,
        speaker_a=base.speaker_a,
        speaker_b=base.speaker_b,
        sessions=(
            *base.sessions,
            LocomoSession(
                number=2,
                occurred_at=occurred_at,
                turns=(
                    LocomoTurn(
                        speaker="Alice",
                        text="I still live in Toronto.",
                        occurred_at=occurred_at,
                    ),
                ),
            ),
        ),
        questions=base.questions,
    )


if __name__ == "__main__":
    unittest.main()
