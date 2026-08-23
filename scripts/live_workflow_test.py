#!/usr/bin/env python3
"""Run one six-round LoCoMo-derived test with real PostgreSQL and OpenRouter."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from fluxmem.diagnostics import diagnostics_to_dict
from fluxmem.domain.llm import LLMTaskKind
from fluxmem.local import bootstrap_from_env


LOCOMO_CASE = {
    "dataset": "LoCoMo",
    "sample_id": "conv-26",
    "category": 1,
    "original_question": "What career path has Caroline decided to persue?",
    "adapted_question": "What career path have I decided to pursue?",
    "reference_answer": "counseling or mental health for Transgender people",
    "evidence_dialogue_ids": ["D1:11", "D4:13"],
    "note": (
        "Selected evidence and distractor turns are replayed as one FluxMem user; "
        "this is a smoke test, not a canonical full-conversation benchmark score."
    ),
}

TURNS = (
    (
        "D1:3",
        "I went to a LGBTQ support group yesterday and it was so powerful.",
    ),
    (
        "D1:11",
        (
            "I'm keen on counseling or working in mental health - I'd love to "
            "support those with similar issues."
        ),
    ),
    (
        "D2:8",
        (
            "Researching adoption agencies — it's been a dream to have a family "
            "and give a loving home to kids who need it."
        ),
    ),
    (
        "D3:11",
        (
            "Thanks, Mel! My friends, family and mentors are my rocks – they "
            "motivate me and give me the strength to push on."
        ),
    ),
    (
        "D4:13",
        (
            "I'm still figuring out the details, but I'm thinking of working "
            "with trans people, helping them accept themselves and supporting "
            "their mental health. Last Friday, I went to an LGBTQ+ counseling "
            "workshop and it was really enlightening. They talked about "
            "different therapeutic methods and how to best work with trans "
            "people. Seeing how passionate these pros were about making a safe "
            "space for people like me was amazing."
        ),
    ),
    ("QA", LOCOMO_CASE["adapted_question"]),
)


def main() -> None:
    arguments = _arguments()
    user_id = uuid4()
    services = bootstrap_from_env(dotenv_path=arguments.dotenv)
    records: list[dict[str, object]] = []
    try:
        if services.process_turn is None:
            raise RuntimeError("LLM turn processing is not configured")
        session = services.create_session.execute(user_id=user_id)
        output = Path(arguments.output) if arguments.output else Path(
            "workflow-traces"
        ) / f"live-{session.session_id}.json"
        print(f"user_id={user_id}")
        print(f"session_id={session.session_id}")
        print(
            "locomo_case="
            f"{LOCOMO_CASE['sample_id']} evidence="
            f"{LOCOMO_CASE['evidence_dialogue_ids']}"
        )

        for round_number, (source_id, question) in enumerate(TURNS, start=1):
            print(f"\nRound {round_number}\nUser: {question}\nAgent: ", end="")
            stream = None
            answer_parts: list[str] = []
            try:
                stream = services.process_turn.execute_text(
                    user_id=user_id,
                    session_id=session.session_id,
                    content=question,
                    agent_id="live-workflow-test",
                    diagnostics=True,
                )
                for delta in stream:
                    answer_parts.append(delta)
                    print(delta, end="", flush=True)
                print()
                result = stream.wait_for_post_answer(timeout=arguments.timeout)
            except BaseException as error:
                trace = stream.diagnostics if stream is not None else None
                records.append(
                    {
                        "round": round_number,
                        "source_id": source_id,
                        "question": question,
                        "answer": "".join(answer_parts),
                        "diagnostics": (
                            diagnostics_to_dict(trace) if trace is not None else None
                        ),
                        "total_usage": (
                            asdict(stream.llm_usage) if stream is not None else None
                        ),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                _write_trace(
                    output=output,
                    user_id=user_id,
                    session_id=session.session_id,
                    records=records,
                )
                print(f"\nFailed trace: {output.resolve()}")
                raise
            if result.diagnostics is None:
                raise RuntimeError("turn diagnostics were not returned")

            trace = result.diagnostics
            records.append(
                {
                    "round": round_number,
                    "source_id": source_id,
                    "question": question,
                    "answer": "".join(answer_parts),
                    "diagnostics": diagnostics_to_dict(trace),
                    "total_usage": asdict(result.llm_usage),
                }
            )
            _print_round_summary(trace)
            _write_trace(
                output=output,
                user_id=user_id,
                session_id=session.session_id,
                records=records,
            )

        print(f"Reference answer: {LOCOMO_CASE['reference_answer']}")
        print(f"\nFull trace: {output.resolve()}")
    finally:
        services.close()


def _print_round_summary(trace) -> None:
    retrieval = trace.retrieval
    seeds = retrieval.seeds.memories if retrieval is not None else ()
    expanded = retrieval.expanded.memories if retrieval is not None else ()
    print("Retrieved seeds:", [item.memory.content for item in seeds])
    print("Answer materials:", [item.memory.content for item in expanded])
    extraction = trace.calls_for_task(LLMTaskKind.EXTRACTION)
    reconciliation = trace.calls_for_task(LLMTaskKind.RECONCILIATION)
    print(
        "Extracted:",
        extraction[-1].validated_output if extraction else (),
    )
    print(
        "Reconciliation:",
        reconciliation[-1].validated_output if reconciliation else (),
    )
    print(
        "Stored:",
        [(item.status.value, item.candidate.content) for item in trace.memory_outcomes],
    )
    if trace.post_answer_errors:
        print("Errors:", trace.post_answer_errors)


def _write_trace(
    *,
    output: Path,
    user_id: UUID,
    session_id: UUID,
    records: list[dict[str, object]],
) -> None:
    """Persist progress after every round, including provider failures."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc),
                "user_id": user_id,
                "session_id": session_id,
                "locomo_case": LOCOMO_CASE,
                "rounds": records,
            },
            default=_json_default,
            indent=2,
        ),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
