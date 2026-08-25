from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    CATEGORY_NAMES,
    EXPECTED_FULL_CONVERSATIONS,
    LOCOMO_DATASET_SHA256,
    LOCOMO_DATASET_URL,
    LOCOMO_LICENSE,
    LOCOMO_REVISION,
    SCORABLE_CATEGORIES,
    LocomoConversation,
    LocomoTurn,
)
from fluxmem_infrastructure.locomo.judge import LocomoJudgment


_TOKEN_FIELDS = (
    "input",
    "output",
    "total",
    "cached_input",
    "cache_write_input",
    "reasoning_output",
)
_CHECKPOINT_VERSION = 1


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

    def store_messages(
        self,
        *,
        user_id: UUID,
        messages: Sequence[Message],
    ) -> tuple[UUID, ...]: ...


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


class JudgeService(Protocol):
    def judge(
        self,
        *,
        category: int,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> LocomoJudgment: ...


class LocomoRunner:
    """Run one or more selected LoCoMo conversations through one pipeline."""

    def __init__(
        self,
        *,
        memory: MemoryService,
        answering: AnsweringService,
        judge: JudgeService,
        output_dir: Path,
        dataset_path: Path,
        dataset_sha256: str,
        mode: str,
        model_settings: Mapping[str, str],
        context_mode: str = "memory",
        git_revision: str | None = None,
        run_id: UUID | None = None,
        resume: bool = False,
    ) -> None:
        if mode not in {"all", "single"}:
            raise ValueError("LoCoMo mode must be 'all' or 'single'")
        if context_mode not in {"memory", "full"}:
            raise ValueError("LoCoMo context mode must be 'memory' or 'full'")
        self._memory = memory
        self._answering = answering
        self._judge = judge
        self._artifacts = ArtifactWriter(output_dir, resume=resume)
        self._dataset_path = dataset_path
        self._dataset_sha256 = dataset_sha256
        self._mode = mode
        self._context_mode = context_mode
        self._model_settings = dict(model_settings)
        self._git_revision = git_revision
        self._resume = resume
        if resume:
            manifest = self._artifacts.read_manifest()
            try:
                resumed_run_id = UUID(str(manifest.get("run_id", "")))
            except ValueError as error:
                raise ValueError("benchmark manifest has an invalid run_id") from error
            if run_id is not None and run_id != resumed_run_id:
                raise ValueError("resume run_id does not match benchmark manifest")
            self._run_id = resumed_run_id
        else:
            self._run_id = run_id or uuid4()
        self._checkpoint: dict[str, Any] = {}

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
        manifest = self._manifest(conversations=conversations)
        if self._resume:
            existing_manifest = self._artifacts.read_manifest()
            _validate_resume_manifest(
                existing=existing_manifest,
                expected=manifest,
            )
            started_at = _manifest_started_at(existing_manifest)
            self._checkpoint = self._artifacts.read_checkpoint()
            _validate_checkpoint(
                checkpoint=self._checkpoint,
                run_id=self._run_id,
            )
        else:
            started_at = datetime.now(timezone.utc)
            manifest["started_at"] = started_at.isoformat()
            self._artifacts.write_manifest(manifest)
            self._checkpoint = {
                "version": _CHECKPOINT_VERSION,
                "run_id": str(self._run_id),
                "updated_at": started_at.isoformat(),
                "conversations": {},
            }
            self._write_checkpoint()

        results: list[dict[str, Any]] = []
        for conversation in conversations:
            progress = self._conversation_progress(conversation)
            if progress.get("complete") is True:
                result = self._artifacts.read_conversation(
                    sample_id=conversation.sample_id
                )
                results.append(result)
                continue
            try:
                result = self._run_conversation(conversation, progress=progress)
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
            if result.get("status") != "failed":
                progress["complete"] = True
                progress["status"] = result.get("status")
                self._write_checkpoint()
            results.append(result)

        summary = _summarize_run(
            run_id=self._run_id,
            mode=self._mode,
            context_mode=self._context_mode,
            started_at=started_at,
            results=results,
        )
        self._artifacts.write_summary(summary)
        return summary

    def _manifest(
        self,
        *,
        conversations: tuple[LocomoConversation, ...],
    ) -> dict[str, Any]:
        is_official_dataset = self._dataset_sha256 == LOCOMO_DATASET_SHA256
        return {
            "run_id": str(self._run_id),
            "mode": self._mode,
            "context_mode": self._context_mode,
            "started_at": None,
            "checkpoint_version": _CHECKPOINT_VERSION,
            "dataset": {
                "path": str(self._dataset_path),
                "sha256": self._dataset_sha256,
                "source_url": LOCOMO_DATASET_URL if is_official_dataset else None,
                "source_revision": LOCOMO_REVISION if is_official_dataset else None,
                "license": LOCOMO_LICENSE if is_official_dataset else None,
                "timestamp_convention": (
                    "Source-local timestamps represented with UTC offsets "
                    "for deterministic ordering"
                ),
                "ingestion_granularity": "source_turn",
                "role_convention": "speaker_a=user, speaker_b=assistant",
                "evaluated_categories": sorted(SCORABLE_CATEGORIES),
                "selected_conversation_count": len(conversations),
                "selected_turn_count": sum(
                    len(conversation.turns) for conversation in conversations
                ),
                "selected_question_count": sum(
                    sum(
                        question.category in SCORABLE_CATEGORIES
                        for question in conversation.questions
                    )
                    for conversation in conversations
                ),
            },
            "models": self._model_settings,
            "git_revision": self._git_revision,
            "selected_conversations": [
                conversation.sample_id for conversation in conversations
            ],
        }

    def _conversation_progress(
        self,
        conversation: LocomoConversation,
    ) -> dict[str, Any]:
        conversations = self._checkpoint.get("conversations")
        if not isinstance(conversations, dict):
            raise ValueError("benchmark checkpoint conversations must be an object")
        user_id = uuid5(
            self._run_id,
            f"locomo:{conversation.sample_id}:user",
        )
        session_id = uuid5(
            self._run_id,
            f"locomo:{conversation.sample_id}:session",
        )
        value = conversations.get(conversation.sample_id)
        if value is None:
            value = {
                "user_id": str(user_id),
                "session_id": str(session_id),
                "completed_turn_count": 0,
                "turn_usages": [],
                "ingestion_errors": [],
                "memory_outcome_counts": {},
                "question_results": [],
                "elapsed_seconds": 0.0,
                "complete": False,
            }
            conversations[conversation.sample_id] = value
            self._write_checkpoint()
        if not isinstance(value, dict):
            raise ValueError("conversation checkpoint must be an object")
        if value.get("user_id") != str(user_id):
            raise ValueError("checkpoint user_id does not match run identity")
        if value.get("session_id") != str(session_id):
            raise ValueError("checkpoint session_id does not match run identity")
        return value

    def _write_checkpoint(self) -> None:
        self._checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._artifacts.write_checkpoint(self._checkpoint)

    def _run_conversation(
        self,
        conversation: LocomoConversation,
        *,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        segment_started = perf_counter()
        prior_elapsed = _nonnegative_number(
            progress.get("elapsed_seconds", 0.0),
            field="checkpoint elapsed_seconds",
        )
        user_id = uuid5(
            self._run_id,
            f"locomo:{conversation.sample_id}:user",
        )
        session_id = uuid5(
            self._run_id,
            f"locomo:{conversation.sample_id}:session",
        )
        self._memory.start_session(user_id=user_id, session_id=session_id)

        completed_turn_count = _checkpoint_count(
            progress.get("completed_turn_count"),
            field="completed_turn_count",
            maximum=len(conversation.turns),
        )
        turn_usages = _checkpoint_list(progress, "turn_usages")
        ingestion_errors = _checkpoint_string_list(
            progress,
            "ingestion_errors",
        )
        outcome_counts = Counter(
            _checkpoint_counts(progress, "memory_outcome_counts")
        )
        if len(turn_usages) != completed_turn_count:
            raise ValueError(
                "checkpoint turn usage count does not match completed turns"
            )
        positioned_turns = tuple(
            (source_session.number, turn_index, turn)
            for source_session in conversation.sessions
            for turn_index, turn in enumerate(source_session.turns)
        )
        for position, (source_session_number, turn_index, turn) in enumerate(
            positioned_turns
        ):
            if position < completed_turn_count:
                continue
            message = _turn_message(
                run_id=self._run_id,
                conversation=conversation,
                session_id=session_id,
                source_session_number=source_session_number,
                turn_index=turn_index,
                turn=turn,
            )
            if self._context_mode == "memory":
                imported = self._memory.ingest_messages(
                    user_id=user_id,
                    messages=(message,),
                )
                turn_usages.append(_usage_dict(imported.llm_usage))
                ingestion_errors.extend(imported.errors)
                outcome_counts.update(
                    outcome.status.value for outcome in imported.memory_outcomes
                )
            else:
                self._memory.store_messages(
                    user_id=user_id,
                    messages=(message,),
                )
                turn_usages.append(_usage_dict(LLMUsageReport()))
            completed_turn_count = position + 1
            progress["completed_turn_count"] = completed_turn_count
            progress["turn_usages"] = turn_usages
            progress["ingestion_errors"] = ingestion_errors
            progress["memory_outcome_counts"] = dict(
                sorted(outcome_counts.items())
            )
            progress["elapsed_seconds"] = (
                prior_elapsed + perf_counter() - segment_started
            )
            self._write_checkpoint()

        last_turn_at = conversation.sessions[-1].turns[-1].occurred_at
        question_results = _checkpoint_question_results(progress)
        completed_question_indices = {
            int(row["index"]) for row in question_results
        }
        for question_index, question in enumerate(conversation.questions):
            if question.category not in SCORABLE_CATEGORIES:
                continue
            if question_index in completed_question_indices:
                continue
            question_started = perf_counter()
            prediction = ""
            retrieval = {
                "query": question.question,
                "query_id": None,
                "memories": [],
            }
            judgment = None
            usage = _model_usage_dict(None)
            error = None
            try:
                result = self._answering.answer(
                    user_id=user_id,
                    session_id=session_id,
                    content=question.question,
                    created_at=last_turn_at + timedelta(seconds=1),
                )
                prediction = _final_answer(result.answer.content)
                retrieval = _retrieval_dict(result)
                usage = _model_usage_dict(result.usage)
            except Exception as answer_error:
                error = f"answer: {_error_text(answer_error)}"

            if error is None:
                try:
                    if question.answer is None:
                        raise ValueError(
                            "scorable LoCoMo question has no answer"
                        )
                    judged = self._judge.judge(
                        category=question.category,
                        question=question.question,
                        ground_truth=question.answer,
                        prediction=prediction,
                    )
                    judgment = _judgment_dict(judged)
                except Exception as judge_error:
                    error = f"judge: {_error_text(judge_error)}"
            question_results.append(
                {
                    "index": question_index,
                    "question": question.question,
                    "answer": question.answer,
                    "prediction": prediction,
                    "category": question.category,
                    "category_name": CATEGORY_NAMES[question.category],
                    "retrieval": retrieval,
                    "judgment": judgment,
                    "latency_seconds": perf_counter() - question_started,
                    "usage": usage,
                    "error": error,
                }
            )
            question_results.sort(key=lambda row: int(row["index"]))
            progress["question_results"] = question_results
            progress["elapsed_seconds"] = (
                prior_elapsed + perf_counter() - segment_started
            )
            self._write_checkpoint()

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
            "context_mode": self._context_mode,
            "status": status,
            "user_id": str(user_id),
            "session_id": str(session_id),
            "speakers": [conversation.speaker_a, conversation.speaker_b],
            "source_session_count": len(conversation.sessions),
            "turn_count": len(conversation.turns),
            "question_count": len(question_results),
            "metrics": _judge_metrics(question_results),
            "ingestion": {
                "memory_outcome_counts": dict(sorted(outcome_counts.items())),
                "errors": ingestion_errors,
                "usage": _merge_usage_dicts(turn_usages),
            },
            "latency_seconds": prior_elapsed + perf_counter() - segment_started,
            "question_results": question_results,
        }


def _validate_resume_manifest(
    *,
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if existing.get("run_id") != expected.get("run_id"):
        raise ValueError("resume manifest run_id does not match")
    for field in ("mode", "selected_conversations", "models"):
        if existing.get(field) != expected.get(field):
            raise ValueError(f"resume manifest {field} does not match")
    if existing.get("context_mode", "memory") != expected.get("context_mode"):
        raise ValueError("resume manifest context_mode does not match")
    existing_dataset = existing.get("dataset")
    expected_dataset = expected.get("dataset")
    if not isinstance(existing_dataset, Mapping) or not isinstance(
        expected_dataset,
        Mapping,
    ):
        raise ValueError("resume manifest dataset must be an object")
    for field in (
        "sha256",
        "evaluated_categories",
        "selected_conversation_count",
        "selected_turn_count",
        "selected_question_count",
    ):
        if existing_dataset.get(field) != expected_dataset.get(field):
            raise ValueError(f"resume manifest dataset {field} does not match")


def _manifest_started_at(manifest: Mapping[str, Any]) -> datetime:
    value = manifest.get("started_at")
    if not isinstance(value, str):
        raise ValueError("benchmark manifest started_at must be a string")
    try:
        started_at = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("benchmark manifest has invalid started_at") from error
    if started_at.utcoffset() is None:
        raise ValueError("benchmark manifest started_at must include a timezone")
    return started_at


def _validate_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    run_id: UUID,
) -> None:
    if checkpoint.get("version") != _CHECKPOINT_VERSION:
        raise ValueError("benchmark checkpoint version is unsupported")
    if checkpoint.get("run_id") != str(run_id):
        raise ValueError("benchmark checkpoint run_id does not match manifest")
    if not isinstance(checkpoint.get("conversations"), dict):
        raise ValueError("benchmark checkpoint conversations must be an object")


def _checkpoint_count(
    value: object,
    *,
    field: str,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint {field} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"checkpoint {field} is out of range")
    return value


def _checkpoint_list(
    progress: Mapping[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    value = progress.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise ValueError(f"checkpoint {field} must be an array of objects")
    return [dict(item) for item in value]


def _checkpoint_string_list(
    progress: Mapping[str, Any],
    field: str,
) -> list[str]:
    value = progress.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"checkpoint {field} must be an array of strings")
    return list(value)


def _checkpoint_counts(
    progress: Mapping[str, Any],
    field: str,
) -> dict[str, int]:
    value = progress.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {field} must be an object")
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint {field} keys must be strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"checkpoint {field} values must be nonnegative integers"
            )
        counts[key] = count
    return counts


def _checkpoint_question_results(
    progress: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _checkpoint_list(progress, "question_results")
    indices: set[int] = set()
    for row in rows:
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("checkpoint question index is invalid")
        if index in indices:
            raise ValueError("checkpoint contains a duplicate question index")
        indices.add(index)
    return sorted(rows, key=lambda row: int(row["index"]))


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if number < 0:
        raise ValueError(f"{field} cannot be negative")
    return number


def _turn_message(
    *,
    run_id: UUID,
    conversation: LocomoConversation,
    session_id: UUID,
    source_session_number: int,
    turn_index: int,
    turn: LocomoTurn,
) -> Message:
    return Message(
        message_id=uuid5(
            run_id,
            f"locomo:{conversation.sample_id}:session:"
            f"{source_session_number}:turn:{turn_index}",
        ),
        session_id=session_id,
        role=(
            "user"
            if turn.speaker == conversation.speaker_a
            else "assistant"
        ),
        agent_id=turn.speaker,
        content=f"{turn.speaker}: {turn.content}",
        created_at=turn.occurred_at,
    )


def _retrieval_dict(result: AnswerResult) -> dict[str, Any]:
    context_ids = set(result.context_memory_ids)
    return {
        "query": result.query.content,
        "query_id": str(result.retrieval.query_id),
        "memories": [
            {
                "memory_id": str(retrieved.memory.memory_id),
                "source_message_id": str(retrieved.memory.message_id),
                "content": retrieved.memory.content,
                "rank": retrieved.rank,
                "score": retrieved.score,
                "retention": retrieved.retention,
                "reasons": list(retrieved.retrieval_reasons),
                "included_in_answer_context": (
                    retrieved.memory.memory_id in context_ids
                ),
            }
            for retrieved in result.retrieval.context.memories
        ],
    }


def _judgment_dict(judgment: LocomoJudgment) -> dict[str, Any]:
    return {
        "label": judgment.label.value,
        "score": judgment.score,
        "reason": judgment.reasoning,
        "model": judgment.model,
        "response_id": judgment.response_id,
        "usage": _model_usage_dict(judgment.usage),
    }


def _final_answer(content: str) -> str:
    if "ANSWER:" in content:
        return content.rsplit("ANSWER:", 1)[-1].strip()
    return content.strip()


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


def _merge_usage_dicts(
    usages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = _aggregate_usage(
        usages=usages,
        operation="turn",
        operation_count=len(usages),
    )
    return {
        "call_count": aggregate["call_count"],
        "reported_call_count": aggregate["reported_call_count"],
        "usage_complete": aggregate["usage_complete"],
        "tokens": aggregate["tokens"],
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
    context_mode: str,
    started_at: datetime,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for result in results
        for row in result.get("question_results", [])
        if isinstance(row, dict)
    ]
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
        "context_mode": context_mode,
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
        "metrics": _judge_metrics(rows),
        "token_usage": _run_token_usage(results=results, rows=rows),
    }


def _judge_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def score(row: Mapping[str, Any]) -> float:
        judgment = row.get("judgment")
        if not isinstance(judgment, Mapping):
            return 0.0
        value = judgment.get("score")
        return float(value) if isinstance(value, (int, float)) else 0.0

    def metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scores = [score(row) for row in selected]
        correct = sum(value >= 0.5 for value in scores)
        total = len(selected)
        return {
            "total": total,
            "judged": sum(
                isinstance(row.get("judgment"), Mapping) for row in selected
            ),
            "correct": correct,
            "accuracy": correct / total * 100 if total else 0.0,
        }

    return {
        "overall": metrics(rows),
        "by_category": {
            CATEGORY_NAMES[category]: metrics(
                [row for row in rows if row.get("category") == category]
            )
            for category in sorted(SCORABLE_CATEGORIES)
        },
    }


def _run_token_usage(
    *,
    results: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ingestion_usage: list[Mapping[str, Any] | None] = []
    for result in results:
        ingestion = result.get("ingestion")
        usage = ingestion.get("usage") if isinstance(ingestion, Mapping) else None
        ingestion_usage.append(usage if isinstance(usage, Mapping) else None)

    answer_usage: list[Mapping[str, Any] | None] = []
    judge_usage: list[Mapping[str, Any] | None] = []
    for row in rows:
        usage = row.get("usage")
        answer_usage.append(usage if isinstance(usage, Mapping) else None)
        judgment = row.get("judgment")
        usage = judgment.get("usage") if isinstance(judgment, Mapping) else None
        judge_usage.append(usage if isinstance(usage, Mapping) else None)

    return {
        "ingestion": _aggregate_usage(
            usages=ingestion_usage,
            operation="turn",
            operation_count=sum(
                int(result.get("turn_count", 0)) for result in results
            ),
        ),
        "answer": _aggregate_usage(
            usages=answer_usage,
            operation="question",
            operation_count=len(rows),
        ),
        "judge": _aggregate_usage(
            usages=judge_usage,
            operation="question",
            operation_count=len(rows),
        ),
    }


def _aggregate_usage(
    *,
    usages: Sequence[Mapping[str, Any] | None],
    operation: str,
    operation_count: int,
) -> dict[str, Any]:
    available = [usage for usage in usages if usage is not None]
    token_records = [
        tokens
        for usage in available
        for tokens in (usage.get("tokens"),)
        if isinstance(tokens, Mapping)
    ]
    totals = (
        {
            field: sum(int(tokens.get(field, 0)) for tokens in token_records)
            for field in _TOKEN_FIELDS
        }
        if token_records
        else None
    )
    usage_complete = (
        len(available) == len(usages)
        and all(bool(usage.get("usage_complete")) for usage in available)
    )
    averages = (
        {
            field: totals[field] / operation_count
            for field in _TOKEN_FIELDS
        }
        if totals is not None and operation_count and usage_complete
        else None
    )
    return {
        "operation": operation,
        "operation_count": operation_count,
        "call_count": sum(
            int(usage.get("call_count", 0)) for usage in available
        ),
        "reported_call_count": sum(
            int(usage.get("reported_call_count", 0)) for usage in available
        ),
        "usage_complete": usage_complete,
        "tokens": totals,
        "average_tokens_per_operation": averages,
    }


def _error_text(error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {detail}"
