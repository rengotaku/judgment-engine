from __future__ import annotations

from judgment_engine.masking import mask_text


def test_masks_paths_urls_emails_and_keys():
    text = (
        "see /home/takuya/Workspace/x.txt and https://example.com/a?b=1 "
        "mail a.b+c@example.org key sk-abcdefgh12345678 ip 192.168.1.10"
    )
    masked = mask_text(text)
    assert "/home/takuya" not in masked
    assert "example.com" not in masked
    assert "a.b+c@example.org" not in masked
    assert "sk-abcdefgh12345678" not in masked
    assert "192.168.1.10" not in masked


def test_masks_long_tokens_but_keeps_japanese():
    masked = mask_text("トークン abcdef0123456789xyz と 日本語の説明")
    assert "abcdef0123456789xyz" not in masked
    assert "日本語の説明" in masked


def test_limit_truncates():
    assert mask_text("あ" * 50, limit=10).endswith("…")
