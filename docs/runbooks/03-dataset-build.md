# 03 蒸留データを作って検査する

- 最終確認: 2026-11-11
- 対象: `judgment-engine` 0.2.0 / `datasets/generated/v1`
- 所要: 約 10 秒（5,000 件・CPU・課金なし）
- 実費: 0 円（テンプレート生成。LLM は呼ばない）

## 目的

5,000 件の学習用データを、出所・確度・分割が追跡できる形で作り、検査を通す。

## 入力

| 名前 | 場所 | 備考 |
|---|---|---|
| 家族（テンプレート） | `src/judgment_engine/data/catalog.py` | 26 家族。型・ドメイン・semantic スロット・答えの決め方 |
| 検査対象の Gold | `datasets/authored/*.jsonl` | 手作りの評価セット 100 件 |
| JevBench gold（任意） | `data/external/jevbench/` | 取得済みなら Gold 隔離の検査に含める |
| 乱数の種 | `--seed`（既定 20261111） | 同じ種なら同じデータになる |

## 前提

- `make install` 済み
- 実教師でラベルを付け直す場合だけ、API キー（`JE_TEACHER_API_KEY`）と課金の承認

## コマンド

```bash
cd /home/takuya/Workspace/judgment-engine/main
make data
```

中身は次と同じ。

```bash
.venv/bin/python -m judgment_engine.data.cli --out datasets/generated/v1 --seed 20261111
```

実教師でラベルを付け直す場合（**課金あり**。承認を得てから）:

```bash
JE_TEACHER_API_KEY=... .venv/bin/python -m judgment_engine.data.cli \
  --out datasets/generated/v1-teacher \
  --relabel openai --allow-paid --teacher-model deepseek-chat
```

## 期待結果

- `datasets/generated/v1/samples.jsonl` が 5,000 行
- `manifest.json` に `counts`（型・ドメイン・言語・split）と `shortfalls`（空）が入る
- `checks.json` の全項目が `ok: true`
- 終了コード 0

## 確認方法

```bash
.venv/bin/python - <<'PY'
import json
rows = [json.loads(line) for line in open("datasets/generated/v1/samples.jsonl")]
print(len(rows), len({row["group_id"] for row in rows}))
print(json.load(open("datasets/generated/v1/checks.json")))
PY
```

`5,000` と group 数（111）、`checks` の全項目 `ok: true` が出れば通っている。
同じ `--seed` で 2 回実行し、`samples.jsonl` の SHA-256 が一致することも確かめる。

## 失敗時の戻し方

| 症状 | 見るところ | 対処 |
|---|---|---|
| `shortfalls` が出る | `manifest.json` の `shortfalls` | その家族の semantic / decor の組合せが足りない。テンプレートを増やす |
| `gold_isolation` が落ちる | `checks.json` の `detail` | 生成文が Gold と同じ。テンプレートの文面を変える |
| `near_duplicates` が落ちる | `cross_split_examples` | 同じ事例が split をまたいでいる。group の決め方を直す |
| `group_splits` が落ちる | `detail` | 同じ `group_id` に複数の split。分割の配分を直す |
| 生成が遅い | — | 5,000 件で 10 秒程度。異常に遅いときは decor の組合せ不足で再試行が増えている |

## 未確認

- 実教師（`--relabel openai`）での通し実行。課金が発生するため、Phase 1 では実測していない
- 生成データで学習したときの効果（Phase 2）
- 評価セットの難易度。テンプレート由来なので、同じ語彙が train と test に出る

## 変更履歴

- 2026-11-11: 初版（Phase 1）
