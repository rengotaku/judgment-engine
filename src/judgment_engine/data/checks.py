"""生成データと評価データの検査。

「作った」ではなく「壊れていない」を機械的に示すための関数群。CLI はここが例外を出さなければ
成功にする。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from .splits import SPLITS, is_eval_split

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def sample_key(sample: dict[str, Any]) -> str:
    """本文と質問が同じなら同じサンプルとみなすためのキー。"""
    payload = json.dumps(
        {
            "state": normalize(sample["state"]),
            "question": sample["question"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CheckFailure(AssertionError):
    pass


def check_required_fields(samples: Sequence[dict[str, Any]]) -> None:
    required = ("id", "group_id", "type", "domain", "state", "question", "label", "split", "provenance")
    for sample in samples:
        missing = [key for key in required if key not in sample]
        if missing:
            raise CheckFailure(f"{sample.get('id')}: 欠けている項目 {missing}")
        provenance = sample["provenance"]
        for key in ("source", "model", "prompt_version", "verified"):
            if key not in provenance:
                raise CheckFailure(f"{sample['id']}: provenance に {key} がない")
        if sample["split"] not in SPLITS:
            raise CheckFailure(f"{sample['id']}: 未知の split {sample['split']}")
        if sample["type"] == "choice" and sample["label"] not in sample["question"]["criteria"]:
            raise CheckFailure(f"{sample['id']}: choice の label が候補にない")
        if sample["type"] == "score":
            if not str(sample["label"]).isdigit():
                raise CheckFailure(f"{sample['id']}: score の label が数値でない")
            if int(sample["label"]) >= len(sample["question"]["criteria"]):
                raise CheckFailure(f"{sample['id']}: score の label がレベル外")
        if sample["type"] == "noul" and sample["label"] not in {"yes", "no"}:
            raise CheckFailure(f"{sample['id']}: noul の label は yes/no")


def check_unique(samples: Sequence[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for sample in samples:
        key = sample_key(sample)
        if key in seen:
            raise CheckFailure(f"重複したサンプル: {seen[key]} と {sample['id']}")
        seen[key] = sample["id"]
    ids = Counter(sample["id"] for sample in samples)
    duplicates = [sample_id for sample_id, count in ids.items() if count > 1]
    if duplicates:
        raise CheckFailure(f"id が重複: {duplicates[:5]}")


def check_group_splits(samples: Sequence[dict[str, Any]]) -> None:
    by_group: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        by_group[sample["group_id"]].add(sample["split"])
    leaked = {group: splits for group, splits in by_group.items() if len(splits) > 1}
    if leaked:
        examples = list(leaked.items())[:3]
        raise CheckFailure(f"group_id が複数の split にまたがっています: {examples}")


def check_gold_isolation(train_samples: Sequence[dict[str, Any]], gold_rows: Iterable[dict[str, Any]]) -> None:
    """Gold の本文が学習側に 1 件でも現れたら失敗させる。"""
    train_keys = {sample_key(sample): sample["id"] for sample in train_samples if sample["split"] == "train"}
    collisions = []
    for row in gold_rows:
        question = row.get("question") or {}
        key = sample_key({"state": row["state"], "question": question})
        if key in train_keys:
            collisions.append((row.get("id"), train_keys[key]))
    if collisions:
        raise CheckFailure(f"Gold が学習データに混ざっています: {collisions[:5]}")


def near_duplicate_pairs(
    samples: Sequence[dict[str, Any]], threshold: float = 0.95, limit: int = 20
) -> list[tuple[str, str, float]]:
    """文字 3-gram の Jaccard がしきい値を超える組を返す（監査用。生成は止めない）。

    同じ group の中の言い換え（state が同じで質問だけ違う）は意図した派生なので除く。
    """

    group_of_id = {sample["id"]: sample["group_id"] for sample in samples}

    def shingles(text: str) -> set[str]:
        compact = normalize(text)
        return {compact[index : index + 3] for index in range(max(0, len(compact) - 2))}

    shingle_sets = [(sample["id"], shingles(sample["state"])) for sample in samples]
    pairs: list[tuple[str, str, float]] = []
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, (_sample_id, values) in enumerate(shingle_sets):
        for shingle in list(values)[:40]:
            buckets[shingle].append(index)
    checked: set[tuple[int, int]] = set()
    for indexes in buckets.values():
        if len(indexes) < 2:
            continue
        for left in range(len(indexes)):
            for right in range(left + 1, len(indexes)):
                pair = (indexes[left], indexes[right])
                if pair in checked:
                    continue
                checked.add(pair)
                first, second = shingle_sets[pair[0]], shingle_sets[pair[1]]
                if group_of_id.get(first[0]) == group_of_id.get(second[0]):
                    continue
                if not first[1] or not second[1]:
                    continue
                score = len(first[1] & second[1]) / len(first[1] | second[1])
                if score >= threshold:
                    pairs.append((first[0], second[0], round(score, 4)))
    pairs.sort(key=lambda item: -item[2])
    return pairs[:limit]


def summarize(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def count(field: str) -> dict[str, int]:
        return dict(sorted(Counter(sample[field] for sample in samples).items()))

    languages = dict(sorted(Counter(sample.get("language", "unknown") for sample in samples).items()))
    main_rows = [sample for sample in samples if sample["split"] != "reserve"]
    reserve_rows = [sample for sample in samples if sample["split"] == "reserve"]
    sources = dict(sorted(Counter(sample["provenance"]["source"] for sample in samples).items()))
    groups = len({sample["group_id"] for sample in samples})
    return {
        "rows": len(samples),
        "groups": groups,
        "by_type": count("type"),
        "by_type_main": dict(sorted(Counter(sample["type"] for sample in main_rows).items())),
        "by_type_reserve": dict(sorted(Counter(sample["type"] for sample in reserve_rows).items())),
        "by_domain": count("domain"),
        "by_language": languages,
        "by_split": count("split"),
        "by_subset": count("subset"),
        "by_source": sources,
        "train_rows": sum(1 for sample in samples if sample["split"] == "train"),
        "eval_rows": sum(1 for sample in samples if is_eval_split(sample["split"])),
    }
