from __future__ import annotations

from pathlib import Path

from judgment_engine.config import Settings
from judgment_engine.eval.datasets import load_rows
from judgment_engine.eval.harness import run_rows, summarize_by_subset
from judgment_engine.inference.engine import JudgmentEngine
from judgment_engine.inference.lexical import LexicalScorer

AUTHORED = Path(__file__).resolve().parents[1] / "datasets" / "authored"


def test_harness_is_reproducible_on_a_slice():
    rows = load_rows([AUTHORED])[:12]
    engine = JudgmentEngine(LexicalScorer(), Settings(backend="lexical"))
    first = run_rows(engine, rows)
    second = run_rows(engine, rows)

    assert [item.predicted for item in first] == [item.predicted for item in second]
    assert [item.probabilities for item in first] == [item.probabilities for item in second]
    def metrics_only(predictions):
        # レイテンシは実行ごとに変わるので比較から除く
        groups = summarize_by_subset(predictions)
        for group in groups.values():
            for key in ("p50_ms", "p95_ms", "p99_ms"):
                group.pop(key, None)
            for nested in group.get("by_type", {}).values():
                for key in ("p50_ms", "p95_ms", "p99_ms"):
                    nested.pop(key, None)
        return groups

    assert metrics_only(first) == metrics_only(second)
