from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    f"{LOCOMO_REVISION}/data/locomo10.json"
)
LOCOMO_DATASET_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
LOCOMO_LICENSE = "CC BY-NC 4.0"
EXPECTED_FULL_CONVERSATIONS = 10
DEFAULT_DATA_PATH = Path(".cache/locomo/locomo10.json")
_SESSION_KEY = re.compile(r"session_(\d+)")
_DATE_FORMAT = "%I:%M %p on %d %B, %Y"


@dataclass(frozen=True, slots=True)
class LocomoTurn:
    dia_id: str
    speaker: str
    text: str
    occurred_at: datetime
    caption: str | None = None

    @property
    def content(self) -> str:
        if self.caption is None:
            return self.text
        if self.text:
            return f"{self.text}\n[Image caption: {self.caption}]"
        return f"[Image caption: {self.caption}]"


@dataclass(frozen=True, slots=True)
class LocomoSession:
    number: int
    occurred_at: datetime
    turns: tuple[LocomoTurn, ...]


@dataclass(frozen=True, slots=True)
class LocomoQuestion:
    question: str
    category: int
    evidence: tuple[str, ...]
    answer: str | None = None
    adversarial_answer: str | None = None


@dataclass(frozen=True, slots=True)
class LocomoConversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[LocomoSession, ...]
    questions: tuple[LocomoQuestion, ...]

    @property
    def turns(self) -> tuple[LocomoTurn, ...]:
        return tuple(turn for session in self.sessions for turn in session.turns)


def ensure_official_dataset(path: Path = DEFAULT_DATA_PATH) -> Path:
    """Download the pinned official data when absent and verify its digest."""

    if path.is_file():
        _verify_digest(path, LOCOMO_DATASET_SHA256)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial")
    try:
        with urllib.request.urlopen(LOCOMO_DATASET_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        _verify_digest(temporary, LOCOMO_DATASET_SHA256)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[LocomoConversation, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid LoCoMo JSON in {path}: {error}") from error
    if not isinstance(raw, list):
        raise ValueError("LoCoMo dataset root must be an array")
    conversations = tuple(
        _parse_conversation(item, index=index)
        for index, item in enumerate(raw)
    )
    sample_ids = [conversation.sample_id for conversation in conversations]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("LoCoMo sample_id values must be unique")
    return conversations


def select_conversations(
    conversations: tuple[LocomoConversation, ...],
    *,
    run_all: bool,
    sample_id: str | None,
) -> tuple[LocomoConversation, ...]:
    if run_all == (sample_id is not None):
        raise ValueError("select exactly one of --all or --conversation")
    if run_all:
        if len(conversations) != EXPECTED_FULL_CONVERSATIONS:
            raise ValueError(
                "a full LoCoMo run requires exactly "
                f"{EXPECTED_FULL_CONVERSATIONS} conversations; found "
                f"{len(conversations)}"
            )
        return conversations
    selected = tuple(
        conversation
        for conversation in conversations
        if conversation.sample_id == sample_id
    )
    if not selected:
        available = ", ".join(
            conversation.sample_id for conversation in conversations
        )
        raise ValueError(
            f"unknown LoCoMo conversation {sample_id!r}; available: {available}"
        )
    return selected


def _parse_conversation(value: object, *, index: int) -> LocomoConversation:
    item = _object(value, context=f"sample {index}")
    sample_id = _nonblank_string(item.get("sample_id"), field="sample_id")
    conversation = _object(
        item.get("conversation"),
        context=f"{sample_id}.conversation",
    )
    speaker_a = _nonblank_string(
        conversation.get("speaker_a"), field=f"{sample_id}.speaker_a"
    )
    speaker_b = _nonblank_string(
        conversation.get("speaker_b"), field=f"{sample_id}.speaker_b"
    )
    if speaker_a == speaker_b:
        raise ValueError(f"{sample_id} must have two distinct speakers")

    session_numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := _SESSION_KEY.fullmatch(key)) is not None
    )
    if not session_numbers:
        raise ValueError(f"{sample_id} contains no dialogue sessions")
    if session_numbers != list(range(1, session_numbers[-1] + 1)):
        raise ValueError(f"{sample_id} session numbers must be contiguous")

    seen_dia_ids: set[str] = set()
    sessions: list[LocomoSession] = []
    for number in session_numbers:
        date_field = f"session_{number}_date_time"
        occurred_at = _parse_timestamp(
            conversation.get(date_field),
            field=f"{sample_id}.{date_field}",
        )
        raw_turns = conversation[f"session_{number}"]
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(
                f"{sample_id}.session_{number} must be a non-empty array"
            )
        turns: list[LocomoTurn] = []
        for turn_index, raw_turn in enumerate(raw_turns):
            turn = _object(
                raw_turn,
                context=f"{sample_id}.session_{number}[{turn_index}]",
            )
            dia_id = _nonblank_string(
                turn.get("dia_id"), field=f"{sample_id}.dia_id"
            )
            if dia_id in seen_dia_ids:
                raise ValueError(f"{sample_id} contains duplicate {dia_id}")
            seen_dia_ids.add(dia_id)
            speaker = _nonblank_string(
                turn.get("speaker"), field=f"{sample_id}.{dia_id}.speaker"
            )
            if speaker not in (speaker_a, speaker_b):
                raise ValueError(
                    f"{sample_id}.{dia_id} uses unknown speaker {speaker!r}"
                )
            text = _string(turn.get("text"), field=f"{sample_id}.{dia_id}.text")
            caption = _optional_nonblank_string(turn.get("blip_caption"))
            if not text.strip() and caption is None:
                raise ValueError(f"{sample_id}.{dia_id} has no usable content")
            turns.append(
                LocomoTurn(
                    dia_id=dia_id,
                    speaker=speaker,
                    text=text.strip(),
                    caption=caption,
                    occurred_at=occurred_at + timedelta(seconds=turn_index),
                )
            )
        sessions.append(
            LocomoSession(
                number=number,
                occurred_at=occurred_at,
                turns=tuple(turns),
            )
        )

    raw_questions = item.get("qa")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(f"{sample_id}.qa must be a non-empty array")
    questions = tuple(
        _parse_question(question, sample_id=sample_id, index=question_index)
        for question_index, question in enumerate(raw_questions)
    )
    return LocomoConversation(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=tuple(sessions),
        questions=questions,
    )


def _parse_question(
    value: object,
    *,
    sample_id: str,
    index: int,
) -> LocomoQuestion:
    item = _object(value, context=f"{sample_id}.qa[{index}]")
    category = item.get("category")
    if isinstance(category, bool) or not isinstance(category, int):
        raise TypeError(f"{sample_id}.qa[{index}].category must be an integer")
    if category not in {1, 2, 3, 4, 5}:
        raise ValueError(f"{sample_id}.qa[{index}] has invalid category")
    evidence = item.get("evidence")
    if not isinstance(evidence, list) or not all(
        isinstance(entry, str) and entry.strip() for entry in evidence
    ):
        raise TypeError(f"{sample_id}.qa[{index}].evidence must be strings")
    answer = _optional_answer(item.get("answer"))
    adversarial_answer = _optional_string(item.get("adversarial_answer"))
    if category != 5 and answer is None:
        raise ValueError(f"{sample_id}.qa[{index}] requires an answer")
    return LocomoQuestion(
        question=_nonblank_string(
            item.get("question"), field=f"{sample_id}.qa[{index}].question"
        ),
        category=category,
        evidence=tuple(entry.strip() for entry in evidence),
        answer=answer,
        adversarial_answer=adversarial_answer,
    )


def _parse_timestamp(value: object, *, field: str) -> datetime:
    raw = _nonblank_string(value, field=field)
    try:
        parsed = datetime.strptime(raw, _DATE_FORMAT)
    except ValueError as error:
        raise ValueError(f"{field} has an unsupported timestamp: {raw!r}") from error
    return parsed.replace(tzinfo=timezone.utc)


def _verify_digest(path: Path, expected: str) -> None:
    actual = dataset_sha256(path)
    if actual != expected:
        raise ValueError(
            f"LoCoMo dataset checksum mismatch for {path}: "
            f"expected {expected}, found {actual}"
        )


def _object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _nonblank_string(value: object, *, field: str) -> str:
    text = _string(value, field=field).strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    return text


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional LoCoMo text must be a string")
    return value


def _optional_answer(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("LoCoMo answer must be a string or number")
    return str(value)


def _optional_nonblank_string(value: object) -> str | None:
    text = _optional_string(value)
    if text is None or not text.strip():
        return None
    return text.strip()
