from __future__ import annotations

from pathlib import Path

import pytest

from judgment_engine.eval.datasets import load_rows

AUTHORED = Path(__file__).resolve().parents[1] / "datasets" / "authored"


@pytest.fixture(scope="module")
def rows():
    return load_rows([AUTHORED])


def test_authored_set_has_100_unique_rows(rows):
    assert len(rows) == 100
    assert len({row.id for row in rows}) == 100


def test_labels_are_valid_for_each_type(rows):
    for row in rows:
        if row.type == "choice":
            assert row.label in row.options, f"{row.id}: {row.label} が options にない"
        elif row.type == "score":
            assert row.levels is not None
            assert 0 <= int(row.label) < len(row.levels), f"{row.id}: レベル外の label"
        else:
            assert row.label in {"yes", "no"}, f"{row.id}: noul の label は yes/no"


def test_all_subsets_present(rows):
    subsets = {row.subset for row in rows}
    assert subsets == {
        "iid",
        "unseen_question",
        "unseen_criteria",
        "unseen_domain",
        "adversarial",
        "japanese",
    }


def test_all_types_present(rows):
    assert {row.type for row in rows} == {"choice", "noul", "score"}
