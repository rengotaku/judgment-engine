from __future__ import annotations

import math

import pytest

from judgment_engine.inference.engine import confidence_of, margin_of, softmax


def test_confidence_matches_published_examples():
    # docs.typesafe.ai の例（Choice / Score）と同じ値になること
    assert confidence_of([0.61, 0.35, 0.04]) == pytest.approx(0.415, abs=0.005)  # 公表 0.42
    assert confidence_of([0.84, 0.16, 0.0]) == pytest.approx(0.76, abs=0.005)
    assert confidence_of([0.4, 0.34, 0.24, 0.02]) == pytest.approx(0.20, abs=0.005)
    assert confidence_of([0.76, 0.24, 0.0]) == pytest.approx(0.64, abs=0.005)
    assert confidence_of([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_edges():
    assert confidence_of([]) == 0.0
    assert confidence_of([1.0]) == 1.0
    # 一様分布は 0
    assert confidence_of([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_softmax_is_stable_for_large_logits():
    probabilities = softmax([1000.0, 1000.0], 1.0)
    assert probabilities == pytest.approx([0.5, 0.5])
    assert all(math.isfinite(value) for value in probabilities)


def test_softmax_temperature_sharpens():
    cold = softmax([2.0, 0.0], 0.5)
    warm = softmax([2.0, 0.0], 4.0)
    assert cold[0] > warm[0]


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], 0.0)


def test_margin_of():
    assert margin_of([0.6, 0.3, 0.1]) == pytest.approx(0.3)
    assert margin_of([1.0]) == 1.0
