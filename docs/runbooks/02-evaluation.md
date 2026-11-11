# 02 評価を回して指標 JSON を残す

- 最終確認: 2026-11-11
- 対象: `judgment-engine` 0.1.0 / `datasets/authored`、`data/external/jevbench`
- 所要: 同梱 100 件（Cross-Encoder・CPU）で 約 20 秒。JevBench gold 1,682 件で 約 7 分
- 実費: 0 円

## 目的

同じ入力から同じ指標（Accuracy / Brier / ECE / MAE / QWK / Coverage / P50 / P95）を出し、
Phase 2 で学習したモデルと比べられる基準値を JSON で残す。

## 入力

| 名前 | 場所 | 備考 |
|---|---|---|
| 同梱データ | `datasets/authored/*.jsonl` | 100 件。iid / unseen_question / unseen_criteria / unseen_domain / adversarial / japanese |
| 外部データ | `data/external/jevbench/<commit>/` | `make fetch-jevbench` で取得（約 1.1 MB、リポジトリには入れない） |
| 設定 | CLI 引数 | `--backend` / `--temperature` / `--abstain-below` |

## 前提

- `make install` 済み（Cross-Encoder を使うなら `make install-ce` も）
- 行の形は JevBench と同じ（`id` / `type` / `domain` / `state` / `question` / `label` / `reliability` / `eval_metric`）

## コマンド

```bash
cd /home/takuya/Workspace/judgment-engine/main
make fetch-jevbench
```

```bash
.venv/bin/python -m judgment_engine.eval.cli --authored --backend cross_encoder --tag authored --out evals/runs/authored.json
```

```bash
.venv/bin/python -m judgment_engine.eval.cli --jevbench --gold-only --backend cross_encoder --tag jevbench-gold --out evals/runs/jevbench-gold.json
```

レイテンシだけを測る場合:

```bash
make eval-latency
```

## 期待結果

- `evals/runs/*.json` ができ、`overall` に `n` / `accuracy` / `brier` / `ece` / `coverage` / `p95_ms` が入る
- 同じコマンドを 2 回実行すると `overall` が一致する（`generated_at` と `latency_ms` は変わる）
- `counts.by_type` が Choice / Noul / Score の件数と一致する

## 確認方法

```bash
.venv/bin/python - <<'PY'
import json
a = json.load(open("evals/runs/authored.json"))
print(a["overall"])
print(a["counts"])
print({k: round(v["accuracy"], 3) for k, v in a["by_subset"].items()})
PY
```

`overall.n` が 100、`by_subset` に 6 群（iid / unseen_question / unseen_criteria / unseen_domain / adversarial / japanese）が出れば通っている。

## 失敗時の戻し方

| 症状 | 見るところ | 対処 |
|---|---|---|
| `data/external/jevbench` が無い | `make fetch-jevbench` | 固定 commit を取得する。manifest にファイルハッシュが残る |
| 422 が出て止まる | データ行 | `question.options` / `question.levels` の欠落。行番号がエラーに出る |
| 指標が前回と違う | `temperature` / `abstain_below` / モデル | 出力 JSON の `temperature` と `model_id` を前回と突き合わせる |
| 極端に遅い | CPU の混雑 | `JE_TORCH_THREADS` を下げる。他セッションの負荷を確認する |

## 基準値

Phase 0 の実測値は `evals/baseline/README.md` にある。Phase 2 の比較はこの値に対して行う。

## 未確認

- 他マシン（GPU あり）での同じ数字。CPU 前提の値しか取っていない
- JevBench の全行（gold 以外を含む 2,934 件）での実行。Phase 0 では gold のみ

## 変更履歴

- 2026-11-11: 初版（Phase 0）
