from __future__ import annotations

import unittest
from concurrent.futures import Future
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fluxmem.application.process_turn import ProcessConversationTurn
from fluxmem.domain.conflict import ConflictProposal
from fluxmem.domain.info_pack import (
    MemoryPack,
    MessagePack,
    RetrievedMemory,
    TurnMemoryPacks,
    UsageType,
)
from fluxmem.domain.llm import (
    GeneratedAnswer,
    GeneratedAnswerStream,
    LLMTaskKind,
    MemoryWriteStatus,
    ModelCallUsage,
    ModelTokenUsage,
    ProposedMemory,
    ReconciliationAction,
    ReconciliationDecision,
)
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current


class FakeStoreMessage:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.messages: list[Message] = []

    def execute(self, *, user_id: UUID, message: Message) -> UUID:
        del user_id
        self.events.append(f"store:{message.role}")
        self.messages.append(message)
        return message.message_id


class FakeHistory:
    def __init__(
        self, events: list[str], user_id: UUID, message: Message
    ) -> None:
        self.events = events
        self.user_id = user_id
        self.message = message

    def execute(self, **values) -> MessagePack:
        self.events.append("history")
        return MessagePack(
            user_id=self.user_id,
            session_id=values["session_id"],
            messages=(self.message,),
        )


class FakeAnsweringRetrieval:
    def __init__(self, events: list[str], packs: TurnMemoryPacks) -> None:
        self.events = events
        self.packs = packs

    def execute(self, **values) -> TurnMemoryPacks:
        del values
        self.events.append("answering_retrieval")
        return self.packs


class FakeAnswerGenerator:
    def __init__(
        self,
        events: list[str],
        generated: GeneratedAnswer,
        token_usage: ModelTokenUsage | None = None,
    ) -> None:
        self.events = events
        self.generated = generated
        self.token_usage = token_usage

    def stream(self, **values) -> GeneratedAnswerStream:
        self.events.append("answer_generation")
        midpoint = max(1, len(self.generated.content) // 2)

        def chunks():
            yield self.generated.content[:midpoint]
            yield self.generated.content[midpoint:]
            recorder = values.get("usage_recorder")
            if recorder is not None and self.token_usage is not None:
                recorder.record(
                    ModelCallUsage(
                        task=LLMTaskKind.ANSWER,
                        model="answer-model",
                        attempt=1,
                        response_id="answer-response",
                        token_usage=self.token_usage,
                    )
                )

        return GeneratedAnswerStream(
            chunks=chunks(),
            context_memory_ids=self.generated.context_memory_ids,
        )

    def generate(self, **values) -> GeneratedAnswer:
        del values
        return self.generated


class ManualExecutor:
    def __init__(self) -> None:
        self.pending: list[tuple[Future, object, tuple, dict]] = []

    def submit(self, function, /, *args, **kwargs) -> Future:
        future = Future()
        self.pending.append((future, function, args, kwargs))
        return future

    def run_next(self) -> None:
        future, function, args, kwargs = self.pending.pop(0)
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)


class FakeReinforceMemory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.feedback = None

    def execute(self, *, feedback_pack):
        self.events.append("feedback")
        self.feedback = feedback_pack
        return ()


class FakeExtractor:
    def __init__(
        self,
        events: list[str],
        candidates: tuple[ProposedMemory, ...],
        token_usage: ModelTokenUsage | None = None,
    ) -> None:
        self.events = events
        self.candidates = candidates
        self.token_usage = token_usage
        self.calls = []

    def extract(self, **values) -> tuple[ProposedMemory, ...]:
        self.events.append("extraction")
        self.calls.append(values)
        recorder = values.get("usage_recorder")
        if recorder is not None and self.token_usage is not None:
            recorder.record(
                ModelCallUsage(
                    task=LLMTaskKind.EXTRACTION,
                    model="extraction-model",
                    attempt=1,
                    response_id="extraction-response",
                    token_usage=self.token_usage,
                )
            )
        return self.candidates


class FakeReconciler:
    def __init__(
        self,
        events: list[str],
        decisions: dict[str, ReconciliationDecision],
        token_usage: ModelTokenUsage | None = None,
    ) -> None:
        self.events = events
        self.decisions = decisions
        self.token_usage = token_usage
        self.calls = []

    def reconcile(self, **values) -> tuple[ReconciliationDecision, ...]:
        candidates = values["candidates"]
        self.events.append("reconcile_batch")
        self.calls.append(values)
        recorder = values.get("usage_recorder")
        if recorder is not None and self.token_usage is not None:
            recorder.record(
                ModelCallUsage(
                    task=LLMTaskKind.RECONCILIATION,
                    model="reconciliation-model",
                    attempt=1,
                    response_id="reconciliation-response",
                    token_usage=self.token_usage,
                )
            )
        return tuple(self.decisions[candidate.content] for candidate in candidates)


class FakeStoreMemory:
    def __init__(
        self,
        events: list[str],
        token_usage: ModelTokenUsage | None = None,
    ) -> None:
        self.events = events
        self.token_usage = token_usage
        self.calls = []

    def execute(self, **values) -> UUID:
        self.events.append(f"store_memory:{values['memory'].content}")
        self.calls.append(values)
        recorder = values.get("usage_recorder")
        if recorder is not None and self.token_usage is not None:
            recorder.record(
                ModelCallUsage(
                    task=LLMTaskKind.LIFECYCLE,
                    model="lifecycle-model",
                    attempt=1,
                    response_id="lifecycle-response",
                    token_usage=self.token_usage,
                )
            )
        return values["memory"].memory_id


class ProcessConversationTurnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[str] = []
        self.now = datetime(2026, 8, 21, 12, tzinfo=timezone.utc)
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.user_message = Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id=None,
            content="I prefer tea and live in Toronto",
            created_at=self.now,
        )
        self.answer_seed = self._memory("The user prefers coffee")
        self.location_context = self._memory("The user lives in Toronto")
        self.answer_neighbor = self._memory("The user prefers tea")
        query_id = uuid4()
        self.seed_pack = self._pack(
            query_id,
            (self.answer_seed, self.location_context),
        )
        self.answering_pack = self._pack(
            query_id,
            (
                self.answer_seed,
                self.location_context,
                self.answer_neighbor,
            ),
        )
        self.turn_memory_packs = TurnMemoryPacks(
            seeds=self.seed_pack,
            expanded=self.answering_pack,
        )
        self.preference_candidate = ProposedMemory(
            content="The user prefers tea",
            source_message_id=self.user_message.message_id,
            session_applicability=None,
        )
        self.location_candidate = ProposedMemory(
            content="The user lives in Toronto",
            source_message_id=self.user_message.message_id,
            session_applicability=None,
        )
        self.answer_id = uuid4()
        self.new_memory_id = uuid4()
        generated_ids = iter((self.answer_id, self.new_memory_id))
        self.id_factory = lambda: next(generated_ids)

    def test_one_retrieval_feeds_extraction_and_one_batch_reconciliation(self) -> None:
        store_message = FakeStoreMessage(self.events)
        reinforce = FakeReinforceMemory(self.events)
        store_memory = FakeStoreMemory(
            self.events,
            ModelTokenUsage(20, 5, 25, reasoning_output_tokens=2),
        )
        extractor = FakeExtractor(
            self.events,
            (self.preference_candidate, self.location_candidate),
            ModelTokenUsage(80, 10, 90),
        )
        reconciler = FakeReconciler(
            self.events,
            {
                self.preference_candidate.content: ReconciliationDecision(
                    action=ReconciliationAction.ADD,
                    conflict_proposals=(
                        ConflictProposal(
                            self.answer_seed.memory_id,
                            confidence=0.9,
                        ),
                    ),
                ),
                self.location_candidate.content: ReconciliationDecision(
                    action=ReconciliationAction.NONE,
                    equivalent_memory_id=self.location_context.memory_id,
                ),
            },
            ModelTokenUsage(90, 10, 100),
        )
        processor = ProcessConversationTurn(
            get_session_history=FakeHistory(
                self.events, self.user_id, self.user_message
            ),
            retrieval_for_answering=FakeAnsweringRetrieval(
                self.events, self.turn_memory_packs
            ),
            store_message=store_message,
            store_memory=store_memory,
            reinforce_memory=reinforce,
            answer_generator=FakeAnswerGenerator(
                self.events,
                GeneratedAnswer(
                    content="Your latest preference is tea.",
                    context_memory_ids=(
                        self.answer_seed.memory_id,
                        self.location_context.memory_id,
                        self.answer_neighbor.memory_id,
                    ),
                    attributed_memory_ids=(self.answer_neighbor.memory_id,),
                ),
                ModelTokenUsage(
                    100,
                    20,
                    120,
                    cached_input_tokens=32,
                ),
            ),
            memory_extractor=extractor,
            memory_reconciler=reconciler,
            clock=FixedClock(self.now),
            id_factory=self.id_factory,
        )

        result = processor.execute_and_wait(
            user_id=self.user_id,
            message=self.user_message,
            agent_id="answering-agent",
        )

        self.assertEqual(result.answer.message_id, self.answer_id)
        self.assertEqual(result.answer.agent_id, "answering-agent")
        self.assertTrue(result.feedback_applied)
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(
            extractor.calls[0]["target_messages"],
            (self.user_message,),
        )
        self.assertNotIn("evidence_messages", extractor.calls[0])
        self.assertIs(extractor.calls[0]["memory_pack"], self.seed_pack)
        self.assertEqual(len(reconciler.calls), 1)
        self.assertEqual(
            reconciler.calls[0]["candidates"],
            (self.preference_candidate, self.location_candidate),
        )
        self.assertIs(
            reconciler.calls[0]["memory_pack"],
            self.answering_pack,
        )
        self.assertIs(
            reconciler.calls[0]["session_history"].messages[0],
            self.user_message,
        )
        self.assertEqual(
            reconciler.calls[0]["evidence_messages"],
            (self.user_message, result.answer),
        )
        self.assertEqual(
            result.memory_outcomes[0].write_context_query_id,
            result.memory_outcomes[1].write_context_query_id,
        )
        self.assertIs(
            result.memory_outcomes[0].status, MemoryWriteStatus.STORED
        )
        self.assertEqual(result.memory_outcomes[0].memory_id, self.new_memory_id)
        self.assertIs(
            result.memory_outcomes[1].status, MemoryWriteStatus.EQUIVALENT
        )
        self.assertEqual(len(store_memory.calls), 1)
        self.assertEqual(
            store_memory.calls[0]["write_context_query_id"],
            self.answering_pack.query_id,
        )

        usage_by_id = {
            usage.memory_id: usage for usage in reinforce.feedback.used_memories
        }
        self.assertIs(
            usage_by_id[self.answer_seed.memory_id].usage_type,
            UsageType.CONTEXT_INCLUDED,
        )
        self.assertIs(
            usage_by_id[self.answer_neighbor.memory_id].usage_type,
            UsageType.CONTEXT_INCLUDED,
        )
        self.assertLess(
            self.events.index("answer_generation"),
            self.events.index("extraction"),
        )
        self.assertLess(
            self.events.index("extraction"),
            self.events.index("reconcile_batch"),
        )
        totals = result.llm_usage.token_totals
        self.assertIsNotNone(totals)
        self.assertEqual(totals.input_tokens, 290)
        self.assertEqual(totals.output_tokens, 45)
        self.assertEqual(totals.total_tokens, 335)
        self.assertEqual(totals.cached_input_tokens, 32)
        self.assertEqual(totals.reasoning_output_tokens, 2)
        self.assertTrue(result.llm_usage.usage_complete)
        self.assertEqual(
            [call.task for call in result.llm_usage.calls],
            [
                LLMTaskKind.ANSWER,
                LLMTaskKind.EXTRACTION,
                LLMTaskKind.RECONCILIATION,
                LLMTaskKind.LIFECYCLE,
            ],
        )

    def test_fabricated_equivalent_id_fails_candidate_without_storing(self) -> None:
        store_memory = FakeStoreMemory(self.events)
        processor = ProcessConversationTurn(
            get_session_history=FakeHistory(
                self.events, self.user_id, self.user_message
            ),
            retrieval_for_answering=FakeAnsweringRetrieval(
                self.events, self.turn_memory_packs
            ),
            store_message=FakeStoreMessage(self.events),
            store_memory=store_memory,
            reinforce_memory=FakeReinforceMemory(self.events),
            answer_generator=FakeAnswerGenerator(
                self.events,
                GeneratedAnswer(
                    content="Okay.",
                    context_memory_ids=(),
                ),
            ),
            memory_extractor=FakeExtractor(
                self.events, (self.location_candidate,)
            ),
            memory_reconciler=FakeReconciler(
                self.events,
                {
                    self.location_candidate.content: ReconciliationDecision(
                        action=ReconciliationAction.NONE,
                        equivalent_memory_id=uuid4(),
                    )
                },
            ),
            clock=FixedClock(self.now),
            id_factory=self.id_factory,
        )

        result = processor.execute_and_wait(
            user_id=self.user_id,
            message=self.user_message,
        )

        self.assertIs(result.memory_outcomes[0].status, MemoryWriteStatus.FAILED)
        self.assertIn(
            "absent from persisted turn context",
            result.memory_outcomes[0].error,
        )
        self.assertEqual(store_memory.calls, [])

    def test_dry_run_reconciles_without_storing_memory(self) -> None:
        store_memory = FakeStoreMemory(self.events)
        processor = ProcessConversationTurn(
            get_session_history=FakeHistory(
                self.events, self.user_id, self.user_message
            ),
            retrieval_for_answering=FakeAnsweringRetrieval(
                self.events, self.turn_memory_packs
            ),
            store_message=FakeStoreMessage(self.events),
            store_memory=store_memory,
            reinforce_memory=FakeReinforceMemory(self.events),
            answer_generator=FakeAnswerGenerator(
                self.events,
                GeneratedAnswer(content="Okay.", context_memory_ids=()),
            ),
            memory_extractor=FakeExtractor(
                self.events, (self.preference_candidate,)
            ),
            memory_reconciler=FakeReconciler(
                self.events,
                {
                    self.preference_candidate.content: ReconciliationDecision(
                        action=ReconciliationAction.ADD,
                        conflict_proposals=(
                            ConflictProposal(self.answer_seed.memory_id),
                        ),
                    )
                },
            ),
            clock=FixedClock(self.now),
            id_factory=self.id_factory,
            enable_memory_writes=False,
        )

        result = processor.execute_and_wait(
            user_id=self.user_id,
            message=self.user_message,
        )

        self.assertIs(result.memory_outcomes[0].status, MemoryWriteStatus.DRY_RUN)
        self.assertEqual(
            result.memory_outcomes[0].write_context_query_id,
            self.answering_pack.query_id,
        )
        self.assertEqual(store_memory.calls, [])

    def test_stream_returns_answer_before_post_answer_work_starts(self) -> None:
        manual_executor = ManualExecutor()
        extractor = FakeExtractor(self.events, ())
        store_message = FakeStoreMessage(self.events)
        processor = ProcessConversationTurn(
            get_session_history=FakeHistory(
                self.events, self.user_id, self.user_message
            ),
            retrieval_for_answering=FakeAnsweringRetrieval(
                self.events, self.turn_memory_packs
            ),
            store_message=store_message,
            store_memory=FakeStoreMemory(self.events),
            reinforce_memory=FakeReinforceMemory(self.events),
            answer_generator=FakeAnswerGenerator(
                self.events,
                GeneratedAnswer(
                    content="Your latest preference is tea.",
                    context_memory_ids=(self.answer_neighbor.memory_id,),
                ),
            ),
            memory_extractor=extractor,
            memory_reconciler=FakeReconciler(self.events, {}),
            clock=FixedClock(self.now),
            id_factory=self.id_factory,
            post_answer_executor=manual_executor,
        )

        stream = processor.execute(
            user_id=self.user_id,
            message=self.user_message,
        )

        first_delta = next(stream)
        self.assertTrue(first_delta)
        self.assertNotIn("extraction", self.events)
        remaining_deltas = tuple(stream)
        self.assertEqual(
            first_delta + "".join(remaining_deltas),
            "Your latest preference is tea.",
        )
        self.assertEqual(
            [message.role for message in store_message.messages],
            ["user", "assistant"],
        )
        self.assertNotIn("extraction", self.events)
        self.assertEqual(len(manual_executor.pending), 1)

        manual_executor.run_next()
        result = stream.wait_for_post_answer()

        self.assertEqual(result.answer.content, "Your latest preference is tea.")
        self.assertIn("extraction", self.events)

    def _memory(self, content: str) -> Memory:
        return Memory(
            memory_id=uuid4(),
            message_id=uuid4(),
            content=content,
            created_at=self.now,
        )

    def _pack(
        self, query_id: UUID, memories: tuple[Memory, ...]
    ) -> MemoryPack:
        return MemoryPack(
            query_id=query_id,
            user_id=self.user_id,
            session_id=self.session_id,
            memories=tuple(
                RetrievedMemory(
                    memory=memory,
                    rank=index,
                    score=1.0 / index,
                    retention=0.9,
                    retrieval_reasons=("lexical",),
                )
                for index, memory in enumerate(memories, start=1)
            ),
        )


if __name__ == "__main__":
    unittest.main()
