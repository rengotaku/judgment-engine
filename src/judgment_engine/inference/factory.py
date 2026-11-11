"""設定からスコアラを作る。API と評価で同じ経路を使う。"""

from __future__ import annotations

from ..config import Settings
from .base import Scorer
from .lexical import LexicalScorer


def build_scorer(settings: Settings) -> Scorer:
    if settings.backend == "lexical":
        return LexicalScorer()
    if settings.backend == "cross_encoder":
        from .cross_encoder import CrossEncoderScorer

        return CrossEncoderScorer(settings.model_id)
    raise RuntimeError(f"未知の JE_BACKEND: {settings.backend!r}（lexical / cross_encoder）")
