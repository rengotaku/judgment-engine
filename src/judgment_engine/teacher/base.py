"""教師モデルの差し替え口。

`TeacherAdapter` は「本文（state と question）だけを見てラベルを返す」役に限定する。
データの生成（テンプレート・収集）とラベル付けを混ぜないため、builder は本文だけを作り、
教師は本文からラベルを付ける。テンプレート由来のラベルは builder が構成する（教師ではない）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SampleRequest:
    id: str
    state: str
    question: dict[str, Any]


@dataclass(frozen=True)
class LabelResult:
    label: str
    # Soft label を使うときだけ入る。既定は Hard Label のみ。
    probabilities: dict[str, float] | None = None
    # 教師の生出力（監査用。データには保存しない）
    raw: str | None = None


class TeacherAdapter(Protocol):
    name: str
    model: str
    prompt_version: str

    def label(self, requests: Sequence[SampleRequest]) -> list[LabelResult]:
        """入力と同じ順序でラベルを返す。件数が合わなければ例外。"""
        ...


class TeacherError(RuntimeError):
    """教師の呼び出し失敗。呼び出し側はデータを書かずに止める。"""


def validate_labels(requests: Sequence[SampleRequest], results: Sequence[LabelResult]) -> None:
    if len(requests) != len(results):
        raise TeacherError(f"ラベルの件数が合いません: {len(requests)} 件の入力に {len(results)} 件")
    for request, result in zip(requests, results, strict=True):
        if result.probabilities is not None:
            total = sum(result.probabilities.values())
            if abs(total - 1.0) > 1e-6:
                raise TeacherError(f"{request.id}: probabilities の合計が 1 ではありません（{total}）")
