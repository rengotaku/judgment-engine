"""依存なしの語彙スコアラ。

本物のベースラインではない。CI・起動確認・API の形を確認するための**決定的な**実装で、
数字は「モデルが入っていない状態」の下限としてだけ読む。日本語でも動くように
単語分割は文字種で切り、CJK は文字 bigram で重なりを見る。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _bigrams(text: str) -> set[str]:
    compact = _SPACE_RE.sub("", text.lower())
    compact = "".join(ch for ch in compact if _CJK_RE.match(ch) or ch.isalnum())
    if len(compact) < 2:
        return set(compact)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def similarity(context: str, candidate: str) -> float:
    """0〜1 の類似度。単語の重なりと文字 bigram の重なりを等分する。"""
    return 0.5 * _overlap(_words(context), _words(candidate)) + 0.5 * _overlap(
        _bigrams(context), _bigrams(candidate)
    )


class LexicalScorer:
    name = "lexical"
    exact_token_counts = False

    def __init__(self, scale: float = 4.0) -> None:
        self.scale = scale

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        # 0〜1 の類似度を 0 中心のロジットへ広げる（softmax が極端にならない範囲）
        return [(similarity(context, candidate) - 0.5) * self.scale for context, candidate in pairs]

    def input_tokens(self, pairs: Sequence[tuple[str, str]]) -> int:
        # 推定: 4 文字 ≒ 1 トークン
        return max(1, sum(len(c) + len(o) for c, o in pairs) // 4)

    def close(self) -> None:  # noqa: D401 - 何もしない
        return None
