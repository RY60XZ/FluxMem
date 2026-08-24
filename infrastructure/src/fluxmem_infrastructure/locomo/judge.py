from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files

from fluxmem import ModelTokenUsage, StructuredModelProvider
from fluxmem_infrastructure.answering import AnswerModelSettings
from fluxmem_infrastructure.locomo.dataset import SCORABLE_CATEGORIES


_JUDGE_SYSTEM_PROMPT = (
    "You are evaluating conversational AI memory recall. Return JSON only "
    "with the format requested."
)
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "label": {
            "type": "string",
            "enum": ["CORRECT", "WRONG"],
        },
    },
    "required": ["reasoning", "label"],
    "additionalProperties": False,
}


class LocomoJudgeLabel(StrEnum):
    CORRECT = "CORRECT"
    WRONG = "WRONG"


@dataclass(frozen=True, slots=True)
class LocomoJudgment:
    label: LocomoJudgeLabel
    reasoning: str
    model: str
    response_id: str | None
    usage: ModelTokenUsage | None

    @property
    def score(self) -> float:
        return float(self.label is LocomoJudgeLabel.CORRECT)


class LLMLocomoJudge:
    """Mem0-compatible binary LoCoMo judge without dialogue evidence."""

    def __init__(
        self,
        *,
        provider: StructuredModelProvider,
        settings: AnswerModelSettings,
    ) -> None:
        self._provider = provider
        self._settings = settings

    def judge(
        self,
        *,
        category: int,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> LocomoJudgment:
        if category not in SCORABLE_CATEGORIES:
            raise ValueError("only LoCoMo categories 1-4 are judged")
        values = (question, ground_truth, prediction)
        if any(not value.strip() for value in values):
            raise ValueError("judge inputs cannot be blank")
        reference_answer = _preprocess_answer(
            category=category,
            answer=ground_truth,
        )
        response = self._provider.generate(
            model=self._settings.model,
            instructions=_JUDGE_SYSTEM_PROMPT,
            input_text=_judge_prompt().format(
                question=question,
                answer=reference_answer,
                response=prediction,
            ),
            schema_name="locomo_judgment",
            schema=_JUDGE_SCHEMA,
            timeout_seconds=self._settings.timeout_seconds,
            maximum_output_tokens=self._settings.maximum_output_tokens,
        )
        try:
            value = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise ValueError("judge returned invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("judge output must be a JSON object")
        label_value = value.get("label")
        reasoning = value.get("reasoning")
        if not isinstance(label_value, str):
            raise ValueError("judge label must be a string")
        try:
            label = LocomoJudgeLabel(label_value.upper())
        except ValueError as error:
            raise ValueError("judge label must be CORRECT or WRONG") from error
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("judge reasoning cannot be blank")
        return LocomoJudgment(
            label=label,
            reasoning=reasoning.strip(),
            model=response.model,
            response_id=response.response_id,
            usage=response.usage,
        )


def _preprocess_answer(*, category: int, answer: str) -> str:
    if category == 3 and ";" in answer:
        return answer.split(";", 1)[0].strip()
    return answer


def _judge_prompt() -> str:
    return (
        files("fluxmem_infrastructure.locomo.prompts")
        .joinpath("judge.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
