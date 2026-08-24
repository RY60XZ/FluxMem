from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache
from statistics import fmean
from typing import Final


CATEGORY_NAMES: Final[dict[int, str]] = {
    1: "multi_hop",
    2: "single_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}
_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_UNANSWERABLE_PHRASES = ("no information available", "not mentioned")


def normalize_answer(value: str) -> str:
    lowered = value.lower().replace(",", "")
    without_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    stemmer = _porter_stemmer()
    prediction_tokens = [
        stemmer.stem(word) for word in normalize_answer(prediction).split()
    ]
    truth_tokens = [
        stemmer.stem(word) for word in normalize_answer(ground_truth).split()
    ]
    common = Counter(prediction_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [part.strip() for part in prediction.split(",")]
    truths = [part.strip() for part in ground_truth.split(",")]
    return fmean(
        max(token_f1(candidate, truth) for candidate in predictions)
        for truth in truths
    )


def score_answer(
    *,
    prediction: str,
    ground_truth: str | None,
    category: int,
) -> float:
    if category == 5:
        lowered = prediction.lower()
        return float(
            any(phrase in lowered for phrase in _UNANSWERABLE_PHRASES)
        )
    if ground_truth is None:
        raise ValueError("non-adversarial questions require a ground truth")
    if category == 1:
        return multi_answer_f1(prediction, ground_truth)
    if category == 3:
        ground_truth = ground_truth.split(";", maxsplit=1)[0].strip()
    if category in {2, 3, 4}:
        return token_f1(prediction, ground_truth)
    raise ValueError(f"unsupported LoCoMo category: {category}")


def evidence_recall(
    *,
    expected: tuple[str, ...],
    retrieved: tuple[str, ...],
) -> float:
    if not expected:
        return 1.0
    returned = set(retrieved)
    return sum(evidence in returned for evidence in expected) / len(expected)


@lru_cache(maxsize=1)
def _porter_stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as error:
        raise RuntimeError(
            "LoCoMo scoring requires the optional 'locomo' dependencies"
        ) from error
    return PorterStemmer()
