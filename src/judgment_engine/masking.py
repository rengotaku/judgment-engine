"""ログ・保存用の伏字。

入力はそのまま残さない。パス・URL・メール・鍵の形・長い英数字の並びを置き換える。
過剰に伏せる側へ倒す（残るより消す方が安い）。
"""

from __future__ import annotations

import re

_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "key",
        re.compile(
            r"(?:sk-[A-Za-z0-9_\-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|xox[abps]-[A-Za-z0-9\-]{8,}"
            r"|AKIA[0-9A-Z]{16})"
        ),
        "<key>",
    ),
    ("url", re.compile(r"https?://[^\s\"'<>]+"), "<url>"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    ("path", re.compile(r"(?:~|/(?:home|Users|root|var|tmp|etc|opt|mnt|srv|data))(?:/[\w.@%+-]+)+"), "<path>"),
    ("ipv4", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    # 12 文字以上の英数字の並び（鍵・ID・ハッシュ）。日本語の長い単語は巻き込まない。
    ("token", re.compile(r"\b[A-Za-z0-9_\-]{12,}\b"), "<token>"),
)


def mask_text(text: str, limit: int | None = None) -> str:
    if not text:
        return ""
    masked = text
    for _kind, pattern, replacement in _PATTERNS:
        masked = pattern.sub(replacement, masked)
    if limit is not None and len(masked) > limit:
        masked = masked[:limit] + "…"
    return masked


def mask_value(value: object, limit: int | None = None) -> str:
    return mask_text(str(value), limit=limit)
