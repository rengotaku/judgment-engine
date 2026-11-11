# judgment-engine

Jev 互換の判断エンジン。任意の `state` / `instructions` / `criteria` を受け取り、**型付きの判断（Choice / Score / Noul）と確率**を返す。

- 推論: CPU・常駐・FastAPI（学習時だけ GPU）
- 学習: `gpu-broker` + Modal（このリポジトリは学習コードと評価だけを持ち、GPU の管理は gpu-broker に任せる）
- 目的: 「何でも正しく判定する小型モデル」ではなく、**得意な判断は自律処理し、不得意な判断を検出して教師へ委ねる**エンジン

## 状態（Phase 0〜1）

| 項目 | 状態 |
|---|---|
| API | `POST /v1/decisions`（Jev 互換。`/v1/systemone` も同じ処理） |
| 型 | Choice / Score / Noul を 1 リクエストで複数 |
| 推論 | 学習なしのゼロショット。既定は多言語 Cross-Encoder（`BAAI/bge-reranker-v2-m3`）。`lexical` は CI 用の決定的な代替 |
| 評価 | 同梱 100 件 + JevBench gold 1,682 件。指標は JSON で保存 |
| 転送 | `routing` の形だけ固定。teacher / human への実転送は未実装 |
| データ | テンプレート由来の 5,000 件（choice 2,200 / noul 1,650 / score 1,100 / 予備 50）。provenance と group 分割つき |
| 教師 | OpenAI 互換の `TeacherAdapter`。**実教師での通し実行は未実施**（課金の承認待ち） |
| 未達 | P95 100ms。最短構成（200 文字・2 候補）でも 128ms |

```bash
make install && make install-ce
make run
```

```bash
curl -sS -X POST http://127.0.0.1:8088/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"state":"HTTP 429 returned. Retry-After: 30","questions":{"next":{"type":"choice","instructions":"What next?","criteria":{"retry":"Wait and retry","abort":"Stop"}}}}'
```

## ドキュメント

- `docs/design-v1.md` — 設計案 v1（アーキテクチャ・モデル・データ・API・ロードマップ）
- `docs/api.md` — API 契約（リクエスト/レスポンス、routing、エラー）
- `docs/data-format.md` — データ形式（provenance・group・split・検査）
- `docs/runbooks/01-local-decisions.md` — ローカル起動
- `docs/runbooks/02-evaluation.md` — 評価の実行と指標の読み方
- `docs/runbooks/03-dataset-build.md` — 蒸留データの生成と検査
- `evals/baseline/README.md` — Phase 0 の未学習ベースライン（指標とレイテンシの実測）
- `docs/runbook-establishment.md` — 手順を確立する方法（runbook の作り方と検証の型）
- `docs/runbook-template.md` — 1 手順を書くときのテンプレート

## 進め方（Phase）

| Phase | 内容 | 完了条件 |
|---|---|---|
| 0 | 学習なしで Jev 互換 API と評価ハーネスを動かす | 3 種の API と評価が動き、基準値が残る |
| 1 | 教師蒸留データ（5,000 件目安）と評価セットの隔離 | 出所・確度・分割が追跡できる |
| 2 | gpu-broker で Cross-Encoder を蒸留学習 | ベースラインを上回り、再現可能な成果物が残る |
| 3 | 校正・棄権・教師フォールバック | 未知ドメインの失敗と転送率を定量化 |
| 4 | 継続学習と昇格・ロールバック | 劣化を検出し、旧版へ戻せる |

各 Phase の issue は GitHub を参照。

## 評価データ

- `datasets/authored/` — このリポジトリの独自セット 100 件（iid / unseen_question / unseen_criteria / unseen_domain / adversarial / japanese）
- JevBench（`model-collapse/jev-bench`）— `make fetch-jevbench` で固定 commit から取得。由来が複数ありライセンスが混在するため、リポジトリには含めない

行の形は JevBench に合わせてある（`id` / `type` / `domain` / `state` / `question` / `label` / `reliability` / `eval_metric`）。

## 依存

- `gpu-broker`（学習の実行基盤、Modal の管理、Volume）
- 教師 LLM（Claude / DeepSeek など。Adapter で交換可能）

## ライセンス

private（個人用）
