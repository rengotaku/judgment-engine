"""group_id 単位の分割。

同じ group（言い換え・派生）は必ず同じ側に入れる。予約した家族は、その評価セットへ固定する。
"""

from __future__ import annotations

import hashlib

SPLITS = (
    "train",
    "dev",
    "test_iid",
    "test_unseen_question",
    "test_unseen_criteria",
    "test_unseen_domain",
    "test_adversarial",
    "reserve",
)

RESERVED_SPLIT = {
    "unseen_question": "test_unseen_question",
    "unseen_criteria": "test_unseen_criteria",
    "unseen_domain": "test_unseen_domain",
    "adversarial": "test_adversarial",
}

# train / dev / test の配分（%）。合計 100。
TRAIN_PERCENT = 70
DEV_PERCENT = 85


def bucket(group_id: str) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def assign_split(group_id: str, reserved: str | None = None) -> str:
    if reserved is not None:
        return RESERVED_SPLIT[reserved]
    value = bucket(group_id)
    if value < TRAIN_PERCENT:
        return "train"
    if value < DEV_PERCENT:
        return "dev"
    return "test_iid"


def is_eval_split(split: str) -> bool:
    return split != "train"


def train_splits() -> set[str]:
    return {"train"}


def assign_group_splits(
    group_sizes: dict[str, int],
    reserved_groups: dict[str, str],
) -> dict[str, str]:
    """group 単位で train / dev / test_iid を配分する。

    予約した group は固定。残りは大きい group から順に、目標件数に対する不足が
    最も大きい側へ入れる（件数の偏りを抑える）。同じ入力なら同じ結果になる。
    """
    assignment: dict[str, str] = dict(reserved_groups)
    free = {group: size for group, size in group_sizes.items() if group not in reserved_groups}
    if not free:
        return assignment

    total = sum(free.values())
    targets = {
        "train": total * TRAIN_PERCENT / 100,
        "dev": total * (DEV_PERCENT - TRAIN_PERCENT) / 100,
        "test_iid": total * (100 - DEV_PERCENT) / 100,
    }
    assigned = {split: 0 for split in targets}

    order = sorted(free, key=lambda group: (-free[group], hashlib.sha256(group.encode()).hexdigest()))
    for group in order:
        size = free[group]
        split = max(targets, key=lambda name: (targets[name] - assigned[name], name == "train"))
        assignment[group] = split
        assigned[split] += size
    return assignment
