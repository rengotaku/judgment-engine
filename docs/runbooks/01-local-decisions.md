# 01 ローカルで起動して判断を返す

- 最終確認: 2026-11-11
- 対象: `judgment-engine` 0.1.0 / `feat/phase0-api`
- 所要: 初回 約 4 分（依存の取得と Cross-Encoder のダウンロードを含む）。2 回目以降 約 15 秒
- 実費: 0 円（すべてローカル CPU）

## 目的

`POST /v1/decisions` に Choice / Score / Noul を 1 リクエストで送り、型付きの答えと `routing` を受け取る。

## 入力

| 名前 | 場所 | 備考 |
|---|---|---|
| ソース | リポジトリ直下 | `pyproject.toml` / `src/judgment_engine` |
| 認証鍵 | 環境変数 `JE_API_KEY` | 値は書かない。ローカル確認なら `JE_ALLOW_INSECURE=1` でも可 |
| モデル | `BAAI/bge-reranker-v2-m3` | 初回起動時に Hugging Face から取得（約 2.2 GB） |

## 前提

- Python 3.12、`uv`、`make`
- ポート 8088 が空いていること
- GPU は使わない（CPU 推論）

## コマンド

```bash
cd /home/takuya/Workspace/judgment-engine/main
make install          # 依存（Cross-Encoder なし）
make install-ce       # Cross-Encoder を使う場合だけ
make run              # JE_ALLOW_INSECURE=1 で 127.0.0.1:8088 に起動
```

別ターミナルで:

```bash
curl -sS http://127.0.0.1:8088/health
```

```bash
curl -sS -X POST http://127.0.0.1:8088/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"state":"HTTP 429 returned. Retry-After: 30","questions":{"next":{"type":"choice","instructions":"What next?","criteria":{"retry":"Wait and retry","abort":"Stop"}},"urgent":{"type":"noul","instructions":"Is this urgent?"},"severity":{"type":"score","instructions":"How bad?","criteria":["low","medium","high"]}}}'
```

止める:

```bash
make stop PORT=8088
```

## 期待結果

- `/health` が `{"status":"ok","version":"0.1.0","backend":"lexical"}` を返す
- `/v1/decisions` が 200 で、`answers` に `noul` / `choice` / `score` が 1 つずつ入る
- `routing` の各値が `local` / `teacher` / `human` のいずれか
- `usage.pairs` が「noul 2 + choice の候補数 + score のレベル数」になる

## 確認方法

```bash
curl -sS http://127.0.0.1:8088/metrics | grep -E 'judgment_(decisions|answers)_total'
```

`judgment_decisions_total{backend="cross_encoder"}` が 1 以上になり、
`judgment_answers_total` に型と route ごとの件数が出れば通っている。

## 失敗時の戻し方

| 症状 | 見るところ | 対処 |
|---|---|---|
| `JE_API_KEY ... がありません` | 起動ログ | `JE_ALLOW_INSECURE=1` を付けるか `JE_API_KEY` を渡す |
| 401 が返る | `Authorization` ヘッダ | `Bearer <JE_API_KEY>` を付ける |
| 413 が返る | リクエストの大きさ | 256 KiB 以下にする（`JE_MAX_BODY_BYTES` で変更可） |
| 起動が遅い | 初回のモデル取得 | 2 回目以降はローカルキャッシュから読む |
| 429 が返る | `JE_RATE_LIMIT_PER_MIN` | 既定は 600/分。テストでは下げてある |

## 未確認

- Bearer 鍵のローテーション手順（鍵は現在 `.env` ではなく起動時の環境変数のみ）
- 複数ワーカー（`--workers 2` 以上）でのレート制限は共有されない

## 変更履歴

- 2026-11-11: 初版（Phase 0）
