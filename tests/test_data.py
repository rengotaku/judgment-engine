from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgment_engine.data.builder import build_dataset
from judgment_engine.data.catalog import FAMILIES
from judgment_engine.data.checks import (
    CheckFailure,
    check_gold_isolation,
    check_group_splits,
    check_required_fields,
    check_unique,
    sample_key,
)
from judgment_engine.data.splits import assign_group_splits, assign_split
from judgment_engine.teacher.base import SampleRequest, TeacherError
from judgment_engine.teacher.openai_compatible import (
    allowed_labels,
    build_payload,
    parse_response,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED = REPO_ROOT / "datasets" / "generated" / "v1"


def test_build_is_deterministic_for_small_family_set():
    families = FAMILIES[:3]
    first = build_dataset(seed=7, families=families, reserve=3)
    second = build_dataset(seed=7, families=families, reserve=3)
    other = build_dataset(seed=8, families=families, reserve=3)
    assert first == second
    assert first != other


def test_small_build_passes_checks():
    samples = build_dataset(seed=11, families=FAMILIES[:4], reserve=2)
    check_required_fields(samples)
    check_unique(samples)
    check_group_splits(samples)


def test_gold_isolation_detects_a_copied_row():
    samples = build_dataset(seed=11, families=FAMILIES[:1], reserve=0)
    gold = [{"id": "gold-1", "state": samples[0]["state"], "question": samples[0]["question"]}]
    with pytest.raises(CheckFailure):
        check_gold_isolation(samples, gold)


def test_group_splits_detects_leakage():
    samples = build_dataset(seed=11, families=FAMILIES[:2], reserve=0)
    broken = [dict(samples[0], split="train"), dict(samples[0], id="copy", split="test_iid")]
    with pytest.raises(CheckFailure):
        check_group_splits(broken)


def test_unique_detects_duplicates():
    samples = build_dataset(seed=11, families=FAMILIES[:1], reserve=0)[:2]
    duplicated = [samples[0], dict(samples[1], state=samples[0]["state"], question=samples[0]["question"])]
    with pytest.raises(CheckFailure):
        check_unique(duplicated)


def test_split_assignment_is_balanced_and_stable():
    sizes = {f"g{index}": 100 for index in range(20)}
    first = assign_group_splits(sizes, {})
    second = assign_group_splits(sizes, {})
    assert first == second
    counts = {"train": 0, "dev": 0, "test_iid": 0}
    for group, split in first.items():
        counts[split] += sizes[group]
    assert counts["train"] == pytest.approx(1400, abs=100)
    assert counts["dev"] == pytest.approx(300, abs=100)
    assert counts["test_iid"] == pytest.approx(300, abs=100)


def test_reserved_groups_are_pinned():
    assert assign_split("anything", "unseen_domain") == "test_unseen_domain"
    assert assign_split("anything", "adversarial") == "test_adversarial"


def test_teacher_labels_are_validated():
    request = SampleRequest(
        id="q1",
        state="x",
        question={"type": "choice", "criteria": {"a": "A", "b": "B"}},
    )
    payload = build_payload([request], model="m")
    assert payload["messages"][1]["role"] == "user"
    results = parse_response('{"labels":[{"id":"q1","label":"a"}]}', [request])
    assert results[0].label == "a"
    with pytest.raises(TeacherError):
        parse_response('{"labels":[{"id":"q1","label":"z"}]}', [request])


def test_allowed_labels_per_type():
    assert allowed_labels({"type": "noul"}) == ["yes", "no"]
    assert allowed_labels({"type": "score", "criteria": ["a", "b", "c"]}) == ["0", "1", "2"]
    assert allowed_labels({"type": "choice", "criteria": {"x": "X", "y": "Y"}}) == ["x", "y"]


@pytest.mark.skipif(not (GENERATED / "samples.jsonl").exists(), reason="生成データがない")
def test_committed_dataset_has_5000_rows_and_passes_checks():
    rows = [json.loads(line) for line in (GENERATED / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 5000
    check_required_fields(rows)
    check_unique(rows)
    check_group_splits(rows)
    counts = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    assert counts["train"] > 0
    assert counts["reserve"] == 50
    assert len({sample_key(row) for row in rows}) == len(rows)
