# judgment-engine

Jev 互換の判断エンジン。任意の `state` / `instructions` / `criteria` を受け取り、**型付きの判断（Choice / Score / Noul）と確率**を返す。

- 推論: CPU・常駐・FastAPI（学習時だけ GPU）
- 学習: `gpu-broker` + Modal（このリポジトリは学習コードと評価だけを持ち、GPU の管理は gpu-broker に任せる）
- 目的: 「何でも正しく判定する小型モデル」ではなく、**得意な判断は自律処理し、不得意な判断を検出して教師へ委ねる**エンジン

## ドキュメント

- `docs/design-v1.md` — 設計案 v1（アーキテクチャ・モデル・データ・API・ロードマップ）
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

## 依存

- `gpu-broker`（学習の実行基盤、Modal の管理、Volume）
- 教師 LLM（Claude / DeepSeek など。Adapter で交換可能）

## ライセンス

private（個人用）
