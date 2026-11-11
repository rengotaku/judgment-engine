"""スコアラの共通インタフェース。

スコアラは `(context, candidate)` の組を受け取り、実数のスコア（ロジット）を返す。
確率への変換（softmax・温度）は `engine` 側で行い、スコアラは確率を名乗らない。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Scorer(Protocol):
    name: str
    # True なら usage のトークン数が実測値、False なら文字数からの推定値。
    exact_token_counts: bool

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """組ごとのスコア。順序は入力と同じ長さで返す。"""
        ...

    def input_tokens(self, pairs: Sequence[tuple[str, str]]) -> int:
        """おおよその入力トークン数（usage 表示用）。"""
        ...

    def close(self) -> None:
        ...
