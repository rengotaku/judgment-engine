from __future__ import annotations

import pytest

from judgment_engine.eval.metrics import (
    Prediction,
    accuracy,
    brier,
    ece,
    mean_absolute_error,
    percentile,
    quadratic_weighted_kappa,
    summarize,
)


def row(
    label: str,
    predicted: str,
    probabilities: dict[str, float],
    confidence: float,
    type_name: str = "choice",
    score_value: float | None = None,
    route: str = "local",
) -> Prediction:
    return Prediction(
        row_id=f"{label}-{predicted}",
        type=type_name,
        domain="d",
        subset="s",
        label=label,
        predicted=predicted,
        score_value=score_value,
        probabilities=probabilities,
        confidence=confidence,
        route=route,
        latency_ms=10.0,
    )


def test_accuracy_and_brier():
    rows = [
        row("a", "a", {"a": 1.0, "b": 0.0}, 0.9),
        row("b", "a", {"a": 1.0, "b": 0.0}, 0.9),
    ]
    assert accuracy(rows) == pytest.approx(0.5)
    assert brier(rows) == pytest.approx(0.5)


def test_ece_is_zero_when_confidence_matches_accuracy():
    rows = [row("a", "a", {"a": 1.0, "b": 0.0}, 1.0) for _ in range(10)]
    assert ece(rows) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    rows = [row("a", "b", {"a": 0.8, "b": 0.2}, 0.8) for _ in range(10)]
    assert ece(rows) == pytest.approx(0.8)


def test_qwk_perfect_and_inverted():
    perfect = [
        row(str(i), str(i), {"0": 0.0}, 1.0, "score", score_value=float(i)) for i in range(4)
    ]
    assert quadratic_weighted_kappa(perfect, levels=4) == pytest.approx(1.0)

    inverted = [
        row(str(i), str(3 - i), {"0": 0.0}, 1.0, "score", score_value=float(3 - i)) for i in range(4)
    ]
    assert quadratic_weighted_kappa(inverted, levels=4) < 0


def test_qwk_chance_level_is_near_zero():
    # 予測が常に同じレベルなら、偶然並み（kappa=0）
    constant = [row(str(i % 4), "1", {"0": 0.0}, 1.0, "score", score_value=1.0) for i in range(40)]
    assert abs(quadratic_weighted_kappa(constant, levels=4)) < 0.2


def test_mae():
    rows = [
        row("1", "1", {"0": 0.0}, 1.0, "score", score_value=1.4),
        row("2", "2", {"0": 0.0}, 1.0, "score", score_value=2.6),
    ]
    assert mean_absolute_error(rows) == pytest.approx(0.5)


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile(values, 1.0) == pytest.approx(4.0)
    assert percentile([], 0.95) == 0.0


def test_coverage_counts_only_local():
    rows = [
        row("a", "a", {"a": 1.0}, 0.9, route="local"),
        row("b", "a", {"a": 1.0}, 0.9, route="teacher"),
    ]
    summary = summarize(rows)
    assert summary.coverage == pytest.approx(0.5)
    assert summary.coverage_accuracy == pytest.approx(1.0)
