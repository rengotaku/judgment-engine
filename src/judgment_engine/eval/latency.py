"""P95 を「入力長 × 候補数」で実測する。

同じ入力で繰り返し測り、中央値・P95・P99 を出す。ウォームアップは捨てる。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import DEFAULT_MODEL_ID, Settings
from ..inference.engine import JudgmentEngine
from ..inference.factory import build_scorer
from ..schemas import parse_questions

WORDS = (
    "the service returned an error while the operator checked the deployment and the queue "
    "the customer asked about the invoice and the refund was requested after the charge "
    "ログには認証の失敗と再試行の記録が残り、担当者は設定を確認した "
).split()


def make_state(target_chars: int) -> str:
    pieces: list[str] = []
    length = 0
    index = 0
    while length < target_chars:
        word = WORDS[index % len(WORDS)]
        pieces.append(word)
        length += len(word) + 1
        index += 1
    return " ".join(pieces)[:target_chars]


def make_choice_question(option_count: int) -> dict[str, object]:
    return {
        "type": "choice",
        "instructions": "Which option matches the state best?",
        "criteria": {
            f"option_{index}": f"Candidate option number {index} for this decision" for index in range(option_count)
        },
    }


def make_score_question(level_count: int) -> dict[str, object]:
    return {
        "type": "score",
        "instructions": "How severe is the reported issue?",
        "criteria": [f"Level {index} of the severity scale" for index in range(level_count)],
    }


NOUL = {"type": "noul", "instructions": "Does the state require an immediate action?"}


@dataclass
class Result:
    label: str
    state_chars: int
    questions: int
    pairs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


def measure(
    engine: JudgmentEngine,
    label: str,
    state: str,
    questions_raw: dict[str, dict[str, object]],
    repeats: int,
) -> Result:
    questions = parse_questions(questions_raw, engine.settings.limits)
    pairs = engine.count_pairs(questions.values(), state)
    for _ in range(2):  # ウォームアップ
        engine.decide(state, questions)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        engine.decide(state, questions)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return Result(
        label=label,
        state_chars=len(state),
        questions=len(questions),
        pairs=pairs,
        p50_ms=_percentile(samples, 0.50),
        p95_ms=_percentile(samples, 0.95),
        p99_ms=_percentile(samples, 0.99),
        mean_ms=statistics.fmean(samples),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cross_encoder", choices=["lexical", "cross_encoder"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="evals/runs/latency.json")
    args = parser.parse_args()

    settings = Settings(backend=args.backend, model_id=args.model_id, temperature=args.temperature)
    scorer = build_scorer(settings)
    results: list[Result] = []
    try:
        engine = JudgmentEngine(scorer, settings)
        for state_chars in (200, 1000, 4000):
            state = make_state(state_chars)
            for option_count in (2, 4, 8):
                results.append(
                    measure(
                        engine,
                        f"choice/{state_chars}chars/{option_count}options",
                        state,
                        {"q": make_choice_question(option_count)},
                        args.repeat,
                    )
                )
            for level_count in (3, 10):
                results.append(
                    measure(
                        engine,
                        f"score/{state_chars}chars/{level_count}levels",
                        state,
                        {"q": make_score_question(level_count)},
                        args.repeat,
                    )
                )
            results.append(measure(engine, f"noul/{state_chars}chars", state, {"q": dict(NOUL)}, args.repeat))
            # 1 リクエストに複数質問（API の並列評価）
            results.append(
                measure(
                    engine,
                    f"mixed/{state_chars}chars/8questions",
                    state,
                    {
                        "c": make_choice_question(4),
                        "s": make_score_question(3),
                        "n": dict(NOUL),
                        "c2": make_choice_question(8),
                        "n2": dict(NOUL),
                        "s2": make_score_question(5),
                        "n3": dict(NOUL),
                        "c3": make_choice_question(2),
                    },
                    args.repeat,
                )
            )
    finally:
        scorer.close()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": scorer.name,
        "model_id": args.model_id,
        "temperature": args.temperature,
        "repeat": args.repeat,
        "platform": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "results": [result.__dict__ for result in results],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result.label:34s} pairs={result.pairs:3d} p50={result.p50_ms:7.1f}ms p95={result.p95_ms:7.1f}ms")
    print(f"out={path}")


if __name__ == "__main__":
    main()
