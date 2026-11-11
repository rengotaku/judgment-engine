# 設計案 v1（2026-11-11）

## 結論

Embedding + 小型 Cross-Encoder + 教師 LLM 蒸留のハイブリッド。小型 LLM に文章を生成させるのではなく、**状態・質問・判断基準を入力して型付きの判断と確率を返す専用モデル**を育てる。学習は `gpu-broker` + Modal、推論は CPU で Modal から切り離す。

## 目標（実測ではない）

| 指標 | 目標 |
|---|---|
| 対応形式 | Choice / Score / Noul |
| Choice の選択肢数 | 2〜8 から開始 |
| 推論 | CPU・常駐・バッチ対応 |
| P95 レイテンシ | 100ms 以下（未検証） |
| 自動判断率 | 60% 以上（初期目標） |
| 自動判断の正答率 | 99%（評価で検証） |

100ms と 99% は同時に達成できるとは限らない。モデル容量・入力長・CPU 性能・判断対象で測る。

## アーキテクチャ

```
クライアント (OpenCode / Claude Code / n8n / 任意のアプリ)
  → Judgment API (FastAPI)
     入力検証・正規化・キャッシュ・ポリシー制御
     → Inference Engine
         Stage 1: Embedding + 小型 Head（高速スクリーニング）
         Stage 2: Cross-Encoder（難しい判断の再評価）
         校正・OOD 検出・棄権判定
     → 判断結果と確率
  → Teacher LLM（難問・未知問題）
Learning Pipeline（教師データ蓄積 → gpu-broker / Modal で蒸留 → 評価 → 登録 → CPU 推論へ反映）
```

推論サービスと学習サービスを分離する。学習中でも公開中のモデルは同じバージョンで安定させる。

## モデルの設計

判断カテゴリを固定しない。入力を 3 要素に分解する。

- `state`: 判断対象
- `instructions`: 何を判断するか
- `criteria`: 選択肢・評価基準の意味

選択肢ごとにスコアを計算する（固定クラス分類にしない）。

```
[STATE] API returned 429 Too Many Requests
[QUESTION] What should the agent do next?
[OPTION] retry
[CRITERION] Retry after a delay
```

| 型 | 内部処理 | 出力 |
|---|---|---|
| Noul | 命題の成立を判定 | 0〜1 |
| Choice | 各候補をスコア化して Softmax | 選択肢と確率分布 |
| Score | 順序付き基準をスコア化 | 期待値・段階ごとの確率 |

Choice は `P(i) = exp(z_i/T) / Σ exp(z_j/T)`。`T` は検証データで調整する。Score は段階確率の期待値に加えて Ordinal Loss の比較も行う。Noul は Yes/No を独立に校正し、`confidence` と混同しない API にする。

## 推論エンジンの構成（3 方式を同じ評価基盤で比較）

| モデル | 実装 | 役割 |
|---|---|---|
| M0: Embedding | EmbeddingGemma + MLP | 最速の一次判定 |
| M1: Cross-Encoder | 多言語 Encoder + Binary Head | 本命 |
| M2: 小型 LLM | Qwen 系 0.5〜1B + LoRA | 難問の比較対象 |

M0 は `state + instructions` と `option + criteria` を別々に埋め込み、積・差・連結を MLP に入れる。M1 は 4 要素を 1 シーケンスで入力する。M0 が自信を持てない場合に M1 へ進めるが、M1 単独の経路も残す。

参考実装: hunch（Embedding による教師蒸留）、LLM2Jev（LLM の logit を利用）。

**汎化が中心**: 同じ質問を大量に学習しても未知の質問には効かない。未知の質問・未知の判断基準・未知のドメインへの汎化を学習の中心に置く。

## 蒸留パイプライン

問題生成（ログ・公開データ・合成・手作業）→ 教師 LLM のラベル付け（正解候補・確率分布・判断不能フラグ）→ 品質管理・Dataset Registry（重複排除・矛盾検出・Gold の隔離）→ gpu-broker で蒸留学習（LoRA・チェックポイント）→ 評価・校正・モデル登録 → 品質ゲート通過時のみデプロイ。

### データ形式

```json
{
  "id": "sample_000001",
  "group_id": "scenario_042",
  "domain": "software_operations",
  "language": "en",
  "state": "HTTP 429 returned. Retry-After: 30",
  "question": {
    "type": "choice",
    "instructions": "Choose the next safe action.",
    "criteria": {
      "retry": "Wait and retry",
      "continue": "Continue normally",
      "abort": "Stop the task"
    }
  },
  "label": {
    "choice": "retry",
    "probabilities": {"retry": 0.98, "continue": 0.01, "abort": 0.01}
  },
  "provenance": {
    "source": "teacher",
    "model": "teacher-model-id",
    "prompt_version": "v1",
    "verified": false
  }
}
```

`probabilities` は例示値。**教師の数値を真の確率として扱わない。** Hard Label Distillation から始め、教師の分布が安定してから Soft Label を足す。

### 初期 5,000 件の配分（設計案）

| 型 | 件数 |
|---|---|
| Choice | 2,200 |
| Noul | 1,650 |
| Score | 1,100 |
| （予備） | 50 |

ドメイン・言語・否定文・曖昧表現・選択肢の欠落などを分散させる。言い換え元と派生は同じ分割に入れる。

### 教師の品質区分

| 区分 | 内容 | 使い方 |
|---|---|---|
| Gold | 人間または実行結果で確認 | 評価と高品質学習 |
| Silver | 教師 LLM で生成・検証 | 通常の蒸留学習 |
| Unverified | 未確認・教師間で不一致 | 再判定・人手確認 |

## 評価

- 外部: JevBench（Choice / Score / Noul を同じ枠組みで評価）
- 独自: IID Holdout / Unseen Question / Unseen Criteria / Unseen Domain / Adversarial / Japanese / Calibration
- 指標: Accuracy、Brier、ECE、Latency、Coverage。Score は MAE・QWK も
- 教師への一致率と実際の正解への一致率は**別々**に集計する

### 自動判断の採用基準

最大確率だけで決めない。校正済み確率・上位 2 候補の差（margin）・OOD・リスクを組み合わせ、条件を満たさなければ教師へ、高リスクは人間へ回す。

## API（独自拡張は routing と policy）

```
POST /v1/decisions   # 同期判断
POST /v1/feedback    # 正解・訂正・実行結果
GET  /v1/models
GET  /health
GET  /metrics
```

`routing` でローカル/教師/人間のどこが答えたかを返す。`policy` は省略可能にして Jev 互換クライアントを壊さない。ローカルに閉じ、Bearer 認証・レート制限・入力サイズ制限を入れる。教師へ送る前に機密の除去と送信許可を確認する。

## ロードマップ

| Phase | 内容 | 完了条件 |
|---|---|---|
| 0 | 学習なしで API と評価ハーネス | 3 種の API と評価が動き、基準値が残る |
| 1 | 教師蒸留データ | 出所・確度・分割が追跡できる |
| 2 | gpu-broker で蒸留学習 | ベースラインを上回り、再現可能な成果物 |
| 3 | 校正・棄権・フォールバック | 未知ドメインの失敗と転送率を定量化 |
| 4 | 継続学習 | 劣化を検出し、旧版へ戻せる |

## 重要な設計判断

| 項目 | 決定 |
|---|---|
| アーキテクチャ | 小型モデルによる候補スコアリング |
| 主力 | 多言語 Cross-Encoder |
| 高速化 | Embedding + MLP |
| 教師 | 交換可能な Adapter |
| 学習 | gpu-broker + Modal L4 |
| 推論 | Ubuntu 上の FastAPI / CPU |
| データ | Jev 型 JSONL |
| 評価 | JevBench + 独自の未知タスク |
| 更新 | バージョン管理・品質ゲート・ロールバック |

## 未確定・注意

- P95 100ms と正答率 99% の同時達成は未検証
- Jev の公開仕様は情報源・バージョンで差がある。MVP は互換サブセットとし、完全互換を名乗る前に契約テストを置く
- Volume の commit と、Sandbox 経由の学習での永続化は実装上の検証項目
- 学習ジョブの期限はインスタンスの期限より短くし、チェックポイント確定の余裕を残す
