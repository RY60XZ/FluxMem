from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Protocol
from uuid import UUID, uuid4, uuid5

from fluxmem import (
    LLMUsageReport,
    Message,
    MessageIngestionResult,
    ModelTokenUsage,
    Session,
)
from fluxmem_infrastructure.answering import AnswerResult
from fluxmem_infrastructure.locomo.artifacts import ArtifactWriter
from fluxmem_infrastructure.locomo.dataset import (
    EXPECTED_FULL_CONVERSATIONS,
    LOCOMO_DATASET_SHA256,
    LOCOMO_DATASET_URL,
    LOCOMO_LICENSE,
    LOCOMO_REVISION,
    LocomoConversation,
)
from fluxmem_infrastructure.locomo.scoring import (
    CATEGORY_NAMES,
    evidence_recall,
    score_answer,
)


class MemoryService(Protocol):
    def start_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID | None = None,
    ) -> Session: ...

    def ingest_messages(
        self,
        *,
        user_id: UUID,
        messages: Sequence[Message],
        diagnostics: bool = False,
    ) -> MessageIngestionResult: ...


class AnsweringService(Protocol):
    def answer(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        created_at: datetime | None = None,
        retrieval_limit: int | None = None,
    ) -> AnswerResult: ...


class LocomoRunner:
    """Run one or more selected LoCoMo conversations through one pipeline."""

    def __init__(
        self,
        *,
        memory: MemoryService,
        answering: AnsweringService,
        output_dir: Path,
        dataset_path: Path,
        dataset_sha256: str,
        mode: str,
        model_settings: Mapping[str, str],
        git_revision: str | None = None,
        run_id: UUID | None = None,
    ) -> None:
        if mode not in {"all", "single"}:
            raise ValueError("LoCoMo mode must be 'all' or 'single'")
        self._memory = memory
        self._answering = answering
        self._artifacts = ArtifactWriter(output_dir)
        self._dataset_path = dataset_path
        self._dataset_sha256 = dataset_sha256
        self._mode = mode
        self._model_settings = dict(model_settings)
        self._git_revision = git_revision
        self._run_id = run_id or uuid4()

    def run(
        self,
        conversations: tuple[LocomoConversation, ...],
    ) -> dict[str, Any]:
        expected = EXPECTED_FULL_CONVERSATIONS if self._mode == "all" else 1
        if len(conversations) != expected:
            raise ValueError(
                f"{self._mode} mode requires {expected} conversation(s); "
                f"received {len(conversations)}"
            )
        started_at = datetime.now(timezone.utc)
        is_official_dataset = self._dataset_sha256 == LOCOMO_DATASET_SHA256
        self._artifacts.write_manifest(
            {
                "run_id": str(self._run_id),
                "mode": self._mode,
                "started_at": started_at.isoformat(),
                "dataset": {
                    "path": str(self._dataset_path),
                    "sha256": self._dataset_sha256,
                    "source_url": (
                        LOCOMO_DATASET_URL if is_official_dataset else None
                    ),
                    "source_revision": (
                        LOCOMO_REVISION if is_official_dataset else None
                    ),
                    "license": LOCOMO_LICENSE if is_official_dataset else None,
                    "timestamp_convention": (
                        "Source-local timestamps represented with UTC offsets "
                        "for deterministic ordering"
                    ),
                    "selected_conversation_count": len(conversations),
                    "selected_turn_count": sum(
                        len(conversation.turns)
                        for conversation in conversations
                    ),
                    "selected_question_count": sum(
                        len(conversation.questions)
                        for conversation in conversations
                    ),
                },
                "models": self._model_settings,
                "git_revision": self._git_revision,
                "selected_conversations": [
                    conversation.sample_id for conversation in conversations
                ],
            }
        )

        results: list[dict[str, Any]] = []
        for conversation in conversations:
            try:
                result = self._run_conversation(conversation)
            except Exception as error:
                result = {
                    "sample_id": conversation.sample_id,
                    "status": "failed",
                    "error": _error_text(error),
                    "question_results": [],
                }
            self._artifacts.write_conversation(
                sample_id=conversation.sample_id,
                value=result,
            )
            results.append(result)

        summary = _summarize_run(
            run_id=self._run_id,
            mode=self._mode,
            started_at=started_at,
            results=results,
        )
        self._artifacts.write_summary(summary)
        return summary

    def _run_conversation(
        self,
        conversation: LocomoConversation,
    ) -> dict[str, Any]:
        conversation_started = perf_counter()
        user_id = uuid5(
            self._run_id,
            f"locomo:{conversation.sample_id}:user",
        )
        session_id = uuid5(
            self._run_id,
            f"locomo:{conversation.sample_id}:session",
        )
        self._memory.start_session(user_id=user_id, session_id=session_id)

        dia_id_by_message_id: dict[UUID, str] = {}
        ingestion_reports: list[LLMUsageReport] = []
        ingestion_errors: list[str] = []
        outcome_counts: Counter[str] = Counter()
        for source_session in conversation.sessions:
            messages: list[Message] = []
            for turn in source_session.turns:
                message_id = uuid5(
                    self._run_id,
                    f"locomo:{conversation.sample_id}:message:{turn.dia_id}",
                )
                dia_id_by_message_id[message_id] = turn.dia_id
                messages.append(
                    Message(
                        message_id=message_id,
                        session_id=session_id,
                        role=(
                            "user"
                            if turn.speaker == conversation.speaker_a
                            else "assistant"
                        ),
                        agent_id=turn.speaker,
                        content=turn.content,
                        created_at=turn.occurred_at,
                    )
                )
            imported = self._memory.ingest_messages(
                user_id=user_id,
                messages=messages,
            )
            ingestion_reports.append(imported.llm_usage)
            ingestion_errors.extend(imported.errors)
            outcome_counts.update(
                outcome.status.value for outcome in imported.memory_outcomes
            )

        last_turn_at = conversation.sessions[-1].turns[-1].occurred_at
        question_results: list[dict[str, Any]] = []
        for question_index, question in enumerate(conversation.questions):
            question_started = perf_counter()
            try:
                result = self._answering.answer(
                    user_id=user_id,
                    session_id=session_id,
                    content=question.question,
                    created_at=last_turn_at + timedelta(seconds=1),
                )
                prediction = result.answer.content
                retrieved_dia_ids = _retrieved_dia_ids(
                    result=result,
                    dia_id_by_message_id=dia_id_by_message_id,
                )
                answer_score = score_answer(
                    prediction=prediction,
                    ground_truth=question.answer,
                    category=question.category,
                )
                recall = evidence_recall(
                    expected=question.evidence,
                    retrieved=retrieved_dia_ids,
                )
                error = None
                usage = _model_usage_dict(result.usage)
            except Exception as query_error:
                prediction = ""
                retrieved_dia_ids = ()
                answer_score = 0.0
                recall = 0.0 if question.evidence else 1.0
                error = _error_text(query_error)
                usage = _model_usage_dict(None)
            question_results.append(
                {
                    "index": question_index,
                    "question": question.question,
                    "answer": question.answer,
                    "adversarial_answer": question.adversarial_answer,
                    "prediction": prediction,
                    "category": question.category,
                    "category_name": CATEGORY_NAMES[question.category],
                    "evidence": list(question.evidence),
                    "retrieved_dia_ids": list(retrieved_dia_ids),
                    "answer_f1": answer_score,
                    "evidence_recall": recall,
                    "latency_seconds": perf_counter() - question_started,
                    "usage": usage,
                    "error": error,
                }
            )

        scores = [row["answer_f1"] for row in question_results]
        recalls = [row["evidence_recall"] for row in question_results]
        has_question_errors = any(
            row["error"] is not None for row in question_results
        )
        status = (
            "completed"
            if not ingestion_errors and not has_question_errors
            else "completed_with_errors"
        )
        return {
            "sample_id": conversation.sample_id,
            "status": status,
            "user_id": str(user_id),
            "session_id": str(session_id),
            "speakers": [conversation.speaker_a, conversation.speaker_b],
            "source_session_count": len(conversation.sessions),
            "turn_count": len(conversation.turns),
            "question_count": len(question_results),
            "ingestion": {
                "memory_outcome_counts": dict(sorted(outcome_counts.items())),
                "errors": ingestion_errors,
                "usage": _usage_dict(_combine_usage(ingestion_reports)),
            },
            "answer_f1": fmean(scores) if scores else 0.0,
            "evidence_recall": fmean(recalls) if recalls else 0.0,
            "latency_seconds": perf_counter() - conversation_started,
            "question_results": question_results,
        }


def _retrieved_dia_ids(
    *,
    result: AnswerResult,
    dia_id_by_message_id: Mapping[UUID, str],
) -> tuple[str, ...]:
    context_ids = set(result.context_memory_ids)
    source_ids: list[str] = []
    seen: set[str] = set()
    for retrieved in result.retrieval.context.memories:
        if retrieved.memory.memory_id not in context_ids:
            continue
        dia_id = dia_id_by_message_id.get(retrieved.memory.message_id)
        if dia_id is not None and dia_id not in seen:
            seen.add(dia_id)
            source_ids.append(dia_id)
    return tuple(source_ids)


def _combine_usage(reports: Sequence[LLMUsageReport]) -> LLMUsageReport:
    return LLMUsageReport(
        calls=tuple(call for report in reports for call in report.calls)
    )


def _usage_dict(report: LLMUsageReport) -> dict[str, Any]:
    totals = report.token_totals
    return {
        "call_count": len(report.calls),
        "reported_call_count": report.reported_call_count,
        "usage_complete": report.usage_complete,
        "tokens": (
            {
                "input": totals.input_tokens,
                "output": totals.output_tokens,
                "total": totals.total_tokens,
                "cached_input": totals.cached_input_tokens,
                "cache_write_input": totals.cache_write_input_tokens,
                "reasoning_output": totals.reasoning_output_tokens,
            }
            if totals is not None
            else None
        ),
    }


def _model_usage_dict(usage: ModelTokenUsage | None) -> dict[str, Any]:
    return {
        "call_count": 1,
        "reported_call_count": int(usage is not None),
        "usage_complete": usage is not None,
        "tokens": (
            {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "total": usage.total_tokens,
                "cached_input": usage.cached_input_tokens,
                "cache_write_input": usage.cache_write_input_tokens,
                "reasoning_output": usage.reasoning_output_tokens,
            }
            if usage is not None
            else None
        ),
    }


def _summarize_run(
    *,
    run_id: UUID,
    mode: str,
    started_at: datetime,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for result in results
        for row in result.get("question_results", [])
        if isinstance(row, dict)
    ]
    scores_by_category: defaultdict[int, list[float]] = defaultdict(list)
    for row in rows:
        scores_by_category[int(row["category"])].append(
            float(row["answer_f1"])
        )
    failed = [
        str(result["sample_id"])
        for result in results
        if result.get("status") == "failed"
    ]
    with_errors = [
        str(result["sample_id"])
        for result in results
        if result.get("status") != "completed"
    ]
    completed_at = datetime.now(timezone.utc)
    return {
        "run_id": str(run_id),
        "mode": mode,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": (completed_at - started_at).total_seconds(),
        "conversation_count": len(results),
        "completed_conversation_count": len(results) - len(failed),
        "failed_conversations": failed,
        "conversations_with_errors": with_errors,
        "question_count": len(rows),
        "question_error_count": sum(
            row.get("error") is not None for row in rows
        ),
        "answer_f1": (
            fmean(float(row["answer_f1"]) for row in rows) if rows else 0.0
        ),
        "evidence_recall": (
            fmean(float(row["evidence_recall"]) for row in rows)
            if rows
            else 0.0
        ),
        "categories": {
            str(category): {
                "name": CATEGORY_NAMES[category],
                "question_count": len(scores_by_category.get(category, [])),
                "answer_f1": (
                    fmean(scores_by_category[category])
                    if scores_by_category.get(category)
                    else 0.0
                ),
            }
            for category in sorted(CATEGORY_NAMES)
        },
    }


def _error_text(error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {detail}"
