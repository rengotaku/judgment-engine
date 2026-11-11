"""環境変数から設定を読む。秘密は値を持たず、参照だけを保持する。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # 起動時に落とす（黙って既定へ落とさない）
        raise RuntimeError(f"{name} は整数で指定してください: {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} は数値で指定してください: {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Limits:
    """入力の上限。バイト数は UTF-8 にした本文全体で数える。"""

    max_body_bytes: int = 256 * 1024
    max_state_chars: int = 8_000
    max_instructions_chars: int = 4_000
    max_questions: int = 32
    max_options: int = 255
    min_options: int = 2
    max_option_key_chars: int = 64
    max_option_text_chars: int = 400
    max_score_levels: int = 10
    min_score_levels: int = 2
    max_question_id_chars: int = 64


@dataclass(frozen=True)
class Settings:
    backend: str = "lexical"
    model_id: str = DEFAULT_MODEL_ID
    temperature: float = 1.0
    max_batch_pairs: int = 64
    rate_limit_per_min: int = 600
    # 確率がこの値未満なら local ではなく teacher へ回す（routing の形だけ固定する）
    abstain_below: float = 0.60
    # 上位 2 候補の差がこれ未満なら teacher へ回す
    minimum_margin: float = 0.10
    data_dir: Path = field(default_factory=lambda: Path("var"))
    allow_insecure: bool = False
    log_prompts: bool = False
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            backend=os.environ.get("JE_BACKEND", "lexical").strip() or "lexical",
            model_id=os.environ.get("JE_MODEL_ID", DEFAULT_MODEL_ID),
            temperature=_float("JE_TEMPERATURE", 1.0),
            max_batch_pairs=_int("JE_MAX_BATCH_PAIRS", 64),
            rate_limit_per_min=_int("JE_RATE_LIMIT_PER_MIN", 600),
            abstain_below=_float("JE_ABSTAIN_BELOW", 0.60),
            minimum_margin=_float("JE_MINIMUM_MARGIN", 0.10),
            data_dir=Path(os.environ.get("JE_DATA_DIR", "var")),
            allow_insecure=_bool("JE_ALLOW_INSECURE", False),
            log_prompts=_bool("JE_LOG_PROMPTS", False),
            limits=Limits(
                max_body_bytes=_int("JE_MAX_BODY_BYTES", 256 * 1024),
                max_state_chars=_int("JE_MAX_STATE_CHARS", 8_000),
                max_instructions_chars=_int("JE_MAX_INSTRUCTIONS_CHARS", 4_000),
                max_questions=_int("JE_MAX_QUESTIONS", 32),
                max_options=_int("JE_MAX_OPTIONS", 255),
            ),
        )


def read_api_key() -> str | None:
    """Bearer トークンを返す。`JE_API_KEY` か `JE_API_KEY_FILE` のどちらか。

    どちらも無ければ None（認証なしでは起動しない。`JE_ALLOW_INSECURE=1` のときだけ許可）。
    """
    key = os.environ.get("JE_API_KEY")
    if key:
        return key
    path = os.environ.get("JE_API_KEY_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip() or None
    return None
