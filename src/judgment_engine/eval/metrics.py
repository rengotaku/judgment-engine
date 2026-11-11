"""評価指標。定義をここに固定する（ハーネスの外で計算しない）。

- accuracy: choice / noul の一致率
- Brier: 候補ごとの二乗誤差の平均（2 値では通常の Brier と一致する。0〜1）
- ECE: 信頼度 10 ビンの重み付き |正答率 - 平均信頼度|。低いほど校正が良い
- QWK: 順序付きラベル（score）の二次重み付き kappa。0 は偶然並み、負は逆順
- MAE: score の平均絶対誤差（レベル単位）
- coverage: local で答えた割合（teacher / human へ回さなかった割合）
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class Prediction:
    """1 行分の予測。`probabilities` は候補（choice のキー / score のレベル番号）ごとの確率。"""

    row_id: str
    type: str
    domain: str
    subset: str
    label: str
    predicted: str
    score_value: float | None
    probabilities: dict[str, float]
    confidence: float
    route: str
    latency_ms: float


def accuracy(predictions: Iterable[Prediction]) -> float:
    items = list(predictions)
    if not items:
        return 0.0
    return sum(1 for item in items if item.predicted == item.label) / len(items)


def brier(predictions: Iterable[Prediction]) -> float:
    items = [item for item in predictions if item.probabilities]
    if not items:
        return 0.0
    total = 0.0
    for item in items:
        keys = set(item.probabilities) | {item.label}
        squared = sum(
            (item.probabilities.get(key, 0.0) - (1.0 if key == item.label else 0.0)) ** 2 for key in keys
        )
        # 候補数で割る（2 値のとき通常の Brier と同じ値になる）
        total += squared / len(keys)
    return total / len(items)


def ece(predictions: Sequence[Prediction], bins: int = 10) -> float:
    if not predictions:
        return 0.0
    buckets: list[list[Prediction]] = [[] for _ in range(bins)]
    for item in predictions:
        index = min(bins - 1, max(0, int(item.confidence * bins)))
        buckets[index].append(item)
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        correct = sum(1 for item in bucket if item.predicted == item.label) / len(bucket)
        mean_confidence = sum(item.confidence for item in bucket) / len(bucket)
        total += (len(bucket) / len(predictions)) * abs(correct - mean_confidence)
    return total


def mean_absolute_error(predictions: Sequence[Prediction]) -> float:
    items = [item for item in predictions if item.score_value is not None]
    if not items:
        return 0.0
    return sum(abs(float(item.score_value) - float(item.label)) for item in items) / len(items)


def quadratic_weighted_kappa(predictions: Sequence[Prediction], levels: int | None = None) -> float:
    items = [item for item in predictions if item.score_value is not None]
    if not items:
        return 0.0
    top = levels - 1 if levels is not None else max(
        max(int(item.label) for item in items), max(int(round(item.score_value or 0)) for item in items)
    )
    if top <= 0:
        return 1.0
    size = top + 1
    observed = [[0.0] * size for _ in range(size)]
    for item in items:
        true_index = _clamp(int(float(item.label)), size)
        predicted_index = _clamp(int(round(float(item.score_value or 0))), size)
        observed[true_index][predicted_index] += 1.0

    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[row][col] for row in range(size)) for col in range(size)]
    count = sum(row_sums)
    if count == 0:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for row in range(size):
        for col in range(size):
            weight = ((row - col) ** 2) / (top**2)
            numerator += weight * observed[row][col]
            denominator += weight * row_sums[row] * col_sums[col] / count
    if denominator == 0:
        return 1.0
    return 1.0 - numerator / denominator


def _clamp(value: int, size: int) -> int:
    return max(0, min(size - 1, value))


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass
class Summary:
    n: int = 0
    accuracy: float = 0.0
    brier: float = 0.0
    ece: float = 0.0
    mae: float = 0.0
    qwk: float = 0.0
    coverage: float = 0.0
    coverage_accuracy: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    by_type: dict[str, Summary] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "mae": round(self.mae, 4),
            "qwk": round(self.qwk, 4),
            "coverage": round(self.coverage, 4),
            "coverage_accuracy": round(self.coverage_accuracy, 4),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "by_type": {name: summary.as_dict() for name, summary in self.by_type.items()},
        }


def summarize(
    predictions: Sequence[Prediction],
    score_levels: dict[str, int] | None = None,
    with_types: bool = True,
) -> Summary:
    """1 群の指標をまとめる。`with_types=False` は型ごとの内訳を作らない（無限再帰を防ぐ）。"""
    if not predictions:
        return Summary()
    score_levels = score_levels or {}
    score_items = [item for item in predictions if item.type == "score"]
    local_items = [item for item in predictions if item.route == "local"]
    latencies = [item.latency_ms for item in predictions]

    summary = Summary(
        n=len(predictions),
        accuracy=accuracy(predictions),
        brier=brier(predictions),
        ece=ece(predictions),
        mae=mean_absolute_error(score_items),
        qwk=quadratic_weighted_kappa(score_items) if score_items else 0.0,
        coverage=len(local_items) / len(predictions),
        coverage_accuracy=accuracy(local_items) if local_items else 0.0,
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        p99_ms=percentile(latencies, 0.99),
    )
    if with_types:
        for type_name in ("choice", "noul", "score"):
            group = [item for item in predictions if item.type == type_name]
            if group:
                summary.by_type[type_name] = summarize(group, score_levels, with_types=False)
    return summary
