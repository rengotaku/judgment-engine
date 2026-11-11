"""評価データの読み込み。

行の形は JevBench（`model-collapse/jev-bench`）に合わせる。同じローダーで

- JevBench の `data/<topic>/<type>.jsonl`
- このリポジトリの `datasets/authored/*.jsonl`

の両方を読めるようにして、「同じ形式で回す」を実際に同じコードで満たす。

    1 行 = {"id", "type", "domain", "state", "question", "label",
            "reliability", "eval_metric", "subset"(任意)}
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TYPES = ("choice", "noul", "score")


@dataclass(frozen=True)
class Row:
    id: str
    type: str
    domain: str
    state: str
    instructions: Any
    options: dict[str, Any] | None
    levels: list[Any] | None
    label: str
    reliability: str
    eval_metric: str
    subset: str
    source: str

    def to_question(self) -> dict[str, Any]:
        if self.type == "choice":
            return {"type": "choice", "instructions": self.instructions, "criteria": self.options}
        if self.type == "score":
            return {"type": "score", "instructions": self.instructions, "criteria": self.levels}
        return {"type": "noul", "instructions": self.instructions}


def load_rows(paths: Iterable[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.jsonl")):
                rows.extend(_load_file(child))
        else:
            rows.extend(_load_file(path))
    seen: set[str] = set()
    unique: list[Row] = []
    for row in rows:
        key = f"{row.source}:{row.id}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _load_file(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} が JSON として読めません: {exc}") from exc
            rows.append(_parse_row(raw, path, line_no))
    return rows


def _parse_row(raw: dict[str, Any], path: Path, line_no: int) -> Row:
    where = f"{path}:{line_no}"
    row_type = raw.get("type")
    if row_type not in TYPES:
        raise ValueError(f"{where}: type は {TYPES} のいずれか: {row_type!r}")
    question = raw.get("question") or {}
    if not isinstance(question, dict):
        raise ValueError(f"{where}: question はオブジェクト")
    label = raw.get("label")
    if label is None:
        raise ValueError(f"{where}: label がありません")
    options = question.get("options")
    levels = question.get("levels")
    if row_type == "choice" and not isinstance(options, dict):
        raise ValueError(f"{where}: choice には question.options が必要")
    if row_type == "score" and not isinstance(levels, list):
        raise ValueError(f"{where}: score には question.levels が必要")
    return Row(
        id=str(raw["id"]),
        type=row_type,
        domain=str(raw.get("domain", "unknown")),
        state=raw["state"] if isinstance(raw["state"], str) else json.dumps(raw["state"], ensure_ascii=False),
        instructions=question.get("instructions", ""),
        options=options,
        levels=levels,
        label=str(label),
        reliability=str(raw.get("reliability", "unknown")),
        eval_metric=str(raw.get("eval_metric", "exact")),
        subset=str(raw.get("subset") or raw.get("domain") or "unknown"),
        source=path.parent.name if path.parent.name not in {"", "."} else path.stem,
    )


def file_digests(paths: Iterable[Path]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            digests[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()[:16]
    return digests
