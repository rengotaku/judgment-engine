"""多言語 Cross-Encoder によるゼロショットのスコアラ。

学習はしない。`(context, candidate)` をそのまま 1 ペアとしてエンコーダに通し、
relevance のロジットを候補間で softmax する。Jev の数値と比べるための**未学習の**
ベースラインであり、確率の校正は Phase 3 の範囲。
"""

from __future__ import annotations

import os
from collections.abc import Sequence


class CrossEncoderScorer:
    name = "cross_encoder"

    def __init__(self, model_id: str, max_length: int = 512, batch_size: int = 16) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 環境依存
            raise RuntimeError(
                "Cross-Encoder を使うには `make install-ce`（torch / sentence-transformers）が必要です"
            ) from exc

        threads = int(os.environ.get("JE_TORCH_THREADS", "8"))
        torch.set_num_threads(max(1, threads))
        self._torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        # 生のロジットがほしいので活性化は恒等関数
        self._model = CrossEncoder(model_id, max_length=max_length, activation_fn=torch.nn.Identity())
        self._model.model.eval()
        self.exact_token_counts = True
        self.name = f"cross_encoder:{model_id}"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        with self._torch.inference_mode():
            scores = self._model.predict(
                [(context, candidate) for context, candidate in pairs],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(value) for value in scores]

    def input_tokens(self, pairs: Sequence[tuple[str, str]]) -> int:
        tokenizer = self._model.tokenizer
        total = 0
        for context, candidate in pairs:
            # truncation 後の実トークン数。max_length を超える分は数えない。
            total += len(tokenizer(context, candidate, truncation=True, max_length=self.max_length)["input_ids"])
        return total

    def close(self) -> None:  # pragma: no cover - 環境依存
        del self._model
