"""評価ハーネス。同じ行を API と同じ `JudgmentEngine` に通す。

HTTP を経由しないのは、レイテンシに HTTP の分を混ぜないため。API の形は
`tests/test_api.py` が別に担保している。
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..config import Settings
from ..inference.engine import JudgmentEngine
from ..schemas import parse_questions
from .datasets import Row
from .metrics import Prediction, summarize


def predict_row(engine: JudgmentEngine, row: Row) -> Prediction:
    questions = parse_questions({row.id: row.to_question()}, engine.settings.limits)
    started = time.perf_counter()
    answers, routing, _pairs, _tokens = engine.decide(row.state, questions)
    latency_ms = (time.perf_counter() - started) * 1000.0
    answer = answers[row.id]
    routing_entry = routing[row.id]

    if row.type == "choice":
        probabilities = dict(answer.probabilities)  # type: ignore[union-attr]
        predicted = answer.choice  # type: ignore[union-attr]
        score_value = None
        confidence = answer.confidence  # type: ignore[union-attr]
    elif row.type == "score":
        probabilities = dict(answer.probabilities)  # type: ignore[union-attr]
        score_value = float(answer.score)  # type: ignore[union-attr]
        predicted = str(int(round(score_value)))
        confidence = answer.confidence  # type: ignore[union-attr]
    else:
        p_yes = float(answer.noul)  # type: ignore[union-attr]
        probabilities = {"yes": p_yes, "no": 1.0 - p_yes}
        predicted = "yes" if p_yes >= 0.5 else "no"
        score_value = None
        confidence = max(p_yes, 1.0 - p_yes)

    return Prediction(
        row_id=row.id,
        type=row.type,
        domain=row.domain,
        subset=row.subset,
        label=row.label,
        predicted=predicted,
        score_value=score_value,
        probabilities=probabilities,
        confidence=confidence,
        route=routing_entry.route,
        latency_ms=latency_ms,
    )


def run_rows(engine: JudgmentEngine, rows: Sequence[Row]) -> list[Prediction]:
    return [predict_row(engine, row) for row in rows]


def summarize_by_subset(predictions: Sequence[Prediction]) -> dict[str, object]:
    groups: dict[str, list[Prediction]] = {}
    for item in predictions:
        groups.setdefault(item.subset, []).append(item)
    return {name: summarize(items).as_dict() for name, items in sorted(groups.items())}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_report(
    *,
    predictions: Sequence[Prediction],
    rows: Sequence[Row],
    dataset_paths: Iterable[Path],
    digests: dict[str, str],
    settings: Settings,
    scorer_name: str,
    label: str,
) -> dict[str, object]:
    level_counts = {row.id: len(row.levels or []) for row in rows if row.type == "score"}
    overall = summarize(predictions)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "label": label,
        "git_sha": git_sha(),
        "backend": scorer_name,
        "model_id": settings.model_id if settings.backend == "cross_encoder" else None,
        "temperature": settings.temperature,
        "thresholds": {"abstain_below": settings.abstain_below, "minimum_margin": settings.minimum_margin},
        "datasets": {str(path): digests.get(str(path), "") for path in dataset_paths},
        "counts": {
            "rows": len(rows),
            "by_type": {name: sum(1 for row in rows if row.type == name) for name in ("choice", "noul", "score")},
            "by_reliability": {
                name: sum(1 for row in rows if row.reliability == name)
                for name in sorted({row.reliability for row in rows})
            },
        },
        "overall": overall.as_dict(),
        "by_subset": summarize_by_subset(predictions),
        "score_level_counts": level_counts,
        "rows": [asdict(item) for item in predictions],
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
