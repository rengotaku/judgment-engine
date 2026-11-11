"""テンプレートから蒸留用データを組み立てる。

何を作っているか:
- 家族（catalog.py）ごとに semantic スロットの組合せを作り、decor で見た目を散らす
- 1 つの本文から「英語の言い換え」「日本語」を派生させ、同じ group_id を持たせる
- 答えはテンプレートの構成から決まる。LLM は呼ばない（課金なし）

実教師でラベルを付け直したい場合は、CLI の `--relabel openai` を使う（別承認が必要）。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .catalog import FAMILIES, Family
from .checks import sample_key
from .splits import assign_group_splits, assign_split

BUILDER_VERSION = "template-v1"
PROVENANCE_BASE = {
    "source": "synthetic",
    "model": BUILDER_VERSION,
    "prompt_version": "n/a",
    "verified": False,
}

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_]+)\}")


def render(template: str, slots: dict[str, str], phrases: dict[str, dict[str, str]]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in slots:
            return slots[name]
        table = phrases.get(name)
        if table is None:
            raise KeyError(f"テンプレートの {{{name}}} を埋められません")
        base = name
        for suffix in ("_text_ja", "_text"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        key = slots.get(base)
        if key is None:
            raise KeyError(f"{name} の参照先スロット {base} がありません")
        if key not in table:
            raise KeyError(f"{name}[{key}] が未定義です")
        return table[key].format(**slots)

    return _PLACEHOLDER.sub(replace, template)


def group_of(family: Family, semantic: Sequence[str]) -> str:
    """同じ家族・同じ semantic の値は同じ group。decor（見た目）の違いは group を分けない。

    分けると、日付やサービス名だけが違うほぼ同じ文が train と test に散らばる。
    """
    key = "|".join([family.name, *semantic])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Variant:
    language: str
    state: str
    instructions: str
    criteria: Any
    label: str


def variants_for(family: Family, slots: dict[str, str], rng: random.Random) -> list[Variant]:
    label = family.label(slots)
    state_en = render(family.state, slots, family.phrases)
    output: list[Variant] = [
        Variant("en", state_en, family.instructions[0], family.criteria, label),
    ]
    if len(family.instructions) > 1 and rng.random() < 0.6:
        output.append(Variant("en", state_en, family.instructions[1], family.criteria, label))
    if family.state_ja and rng.random() < family.japanese_ratio:
        state_ja = render(family.state_ja, slots, family.phrases)
        instructions_ja = family.instructions_ja or family.instructions
        criteria_ja = family.criteria_ja if family.criteria_ja is not None else family.criteria
        output.append(Variant("ja", state_ja, instructions_ja[0], criteria_ja, label))
    return output


def build_family(family: Family, rng: random.Random, start_index: int) -> list[dict[str, Any]]:
    semantic_keys = list(family.semantic)
    combos = list(itertools.product(*(family.semantic[key] for key in semantic_keys)))
    rng.shuffle(combos)
    combos_cycle = itertools.cycle(combos)

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = family.target * 40
    while len(rows) < family.target and attempts < max_attempts:
        attempts += 1
        combo = next(combos_cycle)
        slots = dict(zip(semantic_keys, combo, strict=True))
        for key, values in family.decor.items():
            slots[key] = rng.choice(values)

        try:
            variants = variants_for(family, slots, rng)
        except KeyError as exc:
            raise ValueError(f"{family.name}: テンプレートを埋められません: {exc}") from exc

        group_id = group_of(family, [slots[key] for key in semantic_keys])
        for variant in variants:
            if len(rows) >= family.target:
                break
            question = _question(family, variant)
            key_payload = json.dumps({"state": variant.state, "question": question}, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            subset = family.reserved or ("japanese" if variant.language == "ja" else family.domain)
            rows.append(
                {
                    "id": f"syn-{start_index + len(rows):06d}",
                    "group_id": group_id,
                    "type": family.qtype,
                    "domain": family.domain,
                    "language": variant.language,
                    "state": variant.state,
                    "question": question,
                    "label": variant.label,
                    "reliability": "silver",
                    "eval_metric": "ordinal" if family.qtype == "score" else "exact",
                    "subset": subset,
                    "split": "train",  # あとで group 単位に振り直す
                    "family": family.name,
                    "provenance": dict(PROVENANCE_BASE),
                }
            )
    return rows


def _question(family: Family, variant: Variant) -> dict[str, Any]:
    if family.qtype == "choice":
        return {"type": "choice", "instructions": variant.instructions, "criteria": variant.criteria}
    if family.qtype == "score":
        return {"type": "score", "instructions": variant.instructions, "criteria": list(variant.criteria)}
    return {"type": "noul", "instructions": variant.instructions}


def build_dataset(
    seed: int = 20261111, families: Sequence[Family] = FAMILIES, reserve: int = 50
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for family in families:
        samples.extend(build_family(family, rng, start_index=len(samples)))

    samples = _assign_splits(samples, families)

    if reserve > 0:
        existing_keys = {sample_key(sample) for sample in samples}
        existing_groups = {sample["group_id"] for sample in samples}
        samples.extend(
            _build_reserve(
                families,
                rng,
                start_index=len(samples),
                count=reserve,
                existing_keys=existing_keys,
                existing_groups=existing_groups,
            )
        )
    return samples


def _assign_splits(samples: Sequence[dict[str, Any]], families: Sequence[Family]) -> list[dict[str, Any]]:
    reserved_by_family = {family.name: family.reserved for family in families if family.reserved}
    group_sizes: dict[str, int] = {}
    group_reserved: dict[str, str] = {}
    group_family: dict[str, str] = {}
    for sample in samples:
        group_sizes[sample["group_id"]] = group_sizes.get(sample["group_id"], 0) + 1
        group_family[sample["group_id"]] = sample["family"]

    for group, family_name in group_family.items():
        reserved = reserved_by_family.get(family_name)
        if reserved:
            group_reserved[group] = assign_split(group, reserved)

    assignment = assign_group_splits(group_sizes, group_reserved)
    return [{**sample, "split": assignment[sample["group_id"]]} for sample in samples]


def _build_reserve(
    families: Sequence[Family],
    rng: random.Random,
    start_index: int,
    count: int,
    existing_keys: set[str],
    existing_groups: set[str],
) -> list[dict[str, Any]]:
    """予備の行。学習にも評価にも使わない buffer で、split を reserve に固定する。

    既にある行と同じ本文・同じ group は使わない（split のまたぎを作らない）。
    """
    rows: list[dict[str, Any]] = []
    pool = [family for family in families if family.reserved is None]
    index = 0
    while len(rows) < count and index < count * 40:
        family = pool[index % len(pool)]
        index += 1
        semantic_keys = list(family.semantic)
        slots = {key: rng.choice(family.semantic[key]) for key in semantic_keys}
        for key, values in family.decor.items():
            slots[key] = rng.choice(values)
        variants = variants_for(family, slots, rng)
        group_id = "reserve-" + group_of(family, [slots[key] for key in semantic_keys])
        if group_id in existing_groups:
            continue
        for variant in variants:
            if len(rows) >= count:
                break
            question = _question(family, variant)
            key = sample_key({"state": variant.state, "question": question})
            if key in existing_keys:
                continue
            existing_keys.add(key)
            existing_groups.add(group_id)
            rows.append(
                {
                    "id": f"syn-{start_index + len(rows):06d}",
                    "group_id": group_id,
                    "type": family.qtype,
                    "domain": family.domain,
                    "language": variant.language,
                    "state": variant.state,
                    "question": _question(family, variant),
                    "label": variant.label,
                    "reliability": "silver",
                    "eval_metric": "ordinal" if family.qtype == "score" else "exact",
                    "subset": "reserve",
                    "split": "reserve",
                    "family": family.name,
                    "provenance": dict(PROVENANCE_BASE),
                }
            )
    return rows


def relabel(samples: Sequence[dict[str, Any]], adapter: Any) -> list[dict[str, Any]]:
    """教師でラベルを付け直す。`provenance` を教師の情報で置き換える。"""
    from ..teacher.base import SampleRequest, validate_labels

    requests = [
        SampleRequest(id=sample["id"], state=sample["state"], question=sample["question"]) for sample in samples
    ]
    results = adapter.label(requests)
    validate_labels(requests, results)
    relabeled: list[dict[str, Any]] = []
    for sample, result in zip(samples, results, strict=True):
        updated = dict(sample)
        updated["label"] = result.label
        updated["provenance"] = {
            "source": "teacher",
            "model": adapter.model,
            "prompt_version": adapter.prompt_version,
            "verified": False,
        }
        relabeled.append(updated)
    return relabeled


def write_jsonl(path: Any, samples: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
