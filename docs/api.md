# API 契約（Phase 0）

- 最終更新: 2026-11-11
- 対象: `judgment-engine` 0.1.0（`POST /v1/decisions`）
- 互換の基準: TypeSafe System One API（`POST /v1/systemone`）のサブセット

Jev（TypeSafe System One）の形に合わせ、`answers` の読み方をそのまま使えるようにしている。
独自拡張は `routing`（local / teacher / human）と、省略可能な `policy` だけ。

## エンドポイント

| メソッド | パス | 認証 | 内容 |
|---|---|---|---|
| POST | `/v1/decisions` | 要 | 判断（クエリを 1 リクエストで複数） |
| POST | `/v1/systemone` | 要 | 上と同じ（Jev クライアントの差し替え用） |
| POST | `/v1/feedback` | 要 | 正解・訂正・実行結果の記録 |
| GET | `/v1/models` | 要 | モデル・上限の一覧 |
| GET | `/health` | 不要 | 稼働確認（バックエンド名のみ） |
| GET | `/metrics` | 不要 | Prometheus 形式の件数・レイテンシ |

## リクエスト

```json
{
  "model": "judgment-engine-0:cross_encoder",
  "state": {"ticket": "The export button crashes the settings page in Safari."},
  "questions": {
    "needs_human": {"type": "noul", "instructions": "Does the customer want a person?"},
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"returns": "Exchanges and wrong items", "billing": "Charges and invoices"}
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is it?",
      "criteria": ["Cosmetic", "Workaround exists", "Blocking"]
    }
  },
  "policy": {"abstain_below": 0.6, "minimum_margin": 0.1, "risk": "low", "allow_teacher": true}
}
```

| フィールド | 必須 | 内容 |
|---|---|---|
| `state` | はい | 文字列・オブジェクト・配列。8,000 文字まで |
| `questions` | はい | 質問 ID → 質問。32 件まで。ID は `[A-Za-z0-9_.:-]{1,64}` |
| `model` | いいえ | 応答の `model` にそのまま返す（サーバの挙動は変えない） |
| `policy` | いいえ | しきい値とリスク。省略時はサーバ既定 |
| `request_id` | いいえ | 省略時はサーバが採番 |

### 質問の型

| 型 | フィールド | 制約 |
|---|---|---|
| `noul` | `instructions`、任意の `criteria.true/false` | 2 値 |
| `choice` | `instructions`、`criteria`（キー → 説明。説明は `null` 可・構造化可） | 候補 2〜255（Phase 0 の動作確認は 2〜8） |
| `score` | `instructions`、`criteria`（配列。並び順がレベル番号） | レベル 2〜10 |

## レスポンス

```json
{
  "model": "judgment-engine-0:cross_encoder",
  "answers": {
    "needs_human": {"type": "noul", "noul": 0.62},
    "department": {"type": "choice", "choice": "billing", "confidence": 0.31,
                   "probabilities": {"returns": 0.34, "billing": 0.66}},
    "severity": {"type": "score", "score": 1.21, "confidence": 0.42,
                 "legend": {"0": "Cosmetic", "1": "Workaround exists", "2": "Blocking"},
                 "probabilities": {"0": 0.02, "1": 0.75, "2": 0.23}}
  },
  "routing": {
    "needs_human": {"route": "teacher", "reason": "confidence_below_threshold", "confidence": 0.62, "margin": 0.24},
    "department": {"route": "local", "reason": "confident", "confidence": 0.31, "margin": 0.32},
    "severity": {"route": "teacher", "reason": "margin_below_threshold", "confidence": 0.42, "margin": 0.52}
  },
  "usage": {"input_tokens": 412, "output_tokens": 3, "estimated": false, "latency_ms": 128.4, "pairs": 7},
  "request_id": "3f0c…",
  "backend": "cross_encoder"
}
```

`confidence` は Jev と同じ式 `(p_max - 1/n) / (1 - 1/n)`。公式ドキュメントの 4 例
（0.42 / 0.76 / 0.20 / 0.64）と一致することを `tests/test_engine.py` で固定している。

`routing.route` は Phase 0 では**行き先の表示だけ**で、teacher / human への転送は未実装。
`teacher` でも `answers` にはローカルの答えが入る（呼び出し側が `routing` を見て捨てる）。

### routing の決め方

| 条件 | route | reason |
|---|---|---|
| `policy.risk == "high"` | `human` | `high_risk` |
| `confidence < abstain_below` | `teacher`（`allow_teacher=false` なら `human`） | `confidence_below_threshold` |
| `margin < minimum_margin` | `teacher`（同上） | `margin_below_threshold` |
| それ以外 | `local` | `confident` |

`noul` の `confidence` は `max(p, 1-p)`、`margin` は `|p - 0.5| * 2`。

## エラー

| ステータス | 意味 | 本文 |
|---|---|---|
| 401 | Bearer が無い・違う | `{"detail": "invalid bearer token"}` |
| 413 | 本文が上限超え | `{"error": {"type": "payload_too_large"}}` |
| 422 | 形・上限の違反 | `{"error": {"type": "validation_error", "message": "...", "details": [...]}}` |
| 429 | レート制限 | `{"detail": {"type": "rate_limited"}}` |
| 503 | 鍵が設定されていない | `{"detail": "api key is not configured"}` |

422 の `details` は `loc` / `msg` / `type` だけを返し、入力値そのものは返さない。

## ログと機密

- 既定では**入力の中身をログに残さない**。1 リクエストにつき `request_id` / 質問数 / 組数 / レイテンシだけを出す
- `JE_LOG_PROMPTS=1` のときだけ `state` を出す。ただし `masking.py` でパス・URL・メール・鍵の形・12 文字以上の英数字を伏字にしてから出す
- `/v1/feedback` は `note` を伏字にして `var/feedback.jsonl` へ追記する。入力そのものは保存しない
- `/metrics` は件数とレイテンシだけで、入力も質問 ID も含まない
- 422 の `details` に入力値を含めない（値そのものが秘密でも返らない）

