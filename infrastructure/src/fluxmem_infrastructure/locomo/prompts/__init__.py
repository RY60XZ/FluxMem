"""Prompts used by the LoCoMo benchmark harness."""

from importlib.resources import files


def answer_prompt() -> str:
    return _load("answer.txt")


def full_context_answer_prompt() -> str:
    return _load("full_context_answer.txt")


def judge_prompt() -> str:
    return _load("judge.txt")


def _load(name: str) -> str:
    return files(__name__).joinpath(name).read_text(encoding="utf-8").strip()


__all__ = ("answer_prompt", "full_context_answer_prompt", "judge_prompt")
