# データ形式（Phase 1）

生成データと評価データは同じ 1 行 JSON で、JevBench の行形式を拡張したもの。評価ハーネス
（`judgment_engine.eval.datasets`）はこの形式をそのまま読む。

## 行

```json
{
  "id": "syn-000123",
  "group_id": "9c1f0a2b7d44",
  "type": "choice",
  "domain": "software_operations",
  "language": "en",
  "state": "checkout in us-east-1 answered HTTP 429 during the morning peak. Retry-After: 30.",
  "question": {
    "type": "choice",
    "instructions": "Choose the next safe action.",
    "criteria": {"retry_30s": "Wait 30 seconds", "retry_2m": "Wait 2 minutes", "abort": "Stop the task"}
  },
  "label": "retry_30s",
  "reliability": "silver",
  "eval_metric": "exact",
  "subset": "software_operations",
  "split": "train",
  "family": "ops_rate_limit",
  "provenance": {
    "source": "synthetic",
    "model": "template-v1",
    "prompt_version": "n/a",
    "verified": false
  }
}
```

| フィールド | 内容 |
|---|---|
| `id` | 一意。`syn-` は生成データ、`auth-` は手作りの評価セット |
| `group_id` | 同じ判断事例の言い換え・派生をまとめる単位。**split をまたがない** |
| `type` | `choice` / `noul` / `score` |
| `state` | 判断対象の本文 |
| `question` | `instructions` と、choice なら `criteria`（キー→説明）、score なら `criteria`（レベル配列） |
| `label` | Hard Label。choice はキー、noul は `yes`/`no`、score はレベル番号の文字列 |
| `reliability` | `gold` / `silver` / `unverified`。生成データは `silver` |
| `eval_metric` | `exact`（choice/noul）か `ordinal`（score） |
| `split` | `train` / `dev` / `test_iid` / `test_unseen_*` / `test_adversarial` / `reserve` |
| `provenance` | `source` / `model` / `prompt_version` / `verified` の 4 つを必ず持つ |

JevBench は choice の候補を `options`、score のレベルを `levels` に置く。読み込み側は
`options` / `levels` が無ければ `criteria` を使うので、どちらの書き方でも同じ行として扱える。

## provenance の意味

| source | 意味 | verified |
|---|---|---|
| `synthetic` | テンプレートの構成からラベルを決めた。LLM は呼んでいない | `false` |
| `teacher` | 教師モデルが本文だけを見て付けた | `false`（人間か実行結果で確認したら `true`） |
| `human` | 人が付けた | `true` を想定 |

`probabilities`（Soft Label）は既定では持たない。教師の分布を真の確率として扱わないため、
Hard Label だけを学習に使う。Soft を使うときは `reliability` と `verified` を別に管理する。

## split の決め方

1. 予約した家族（未知質問・未知基準・未知ドメイン・敵対）は、その評価セットへ固定する
2. それ以外は **group 単位**で `train` 70% / `dev` 15% / `test_iid` 15% に配る
3. 配分は group の大きさを見て、件数の不足が大きい側へ入れる（偏りを抑える）
4. `reserve` は学習にも評価にも使わない予備。件数だけ確保する

group は「家族 + semantic スロットの値」で決める。日付・サービス名などの decor が違うだけの
行は同じ group なので、ほぼ同じ文が train と test に分かれることはない。

## 検査

`python -m judgment_engine.data.cli` が次を検査し、1 つでも失敗したら終了コード 1 を返す。

| 検査 | 内容 |
|---|---|
| `required_fields` | 必須項目と `provenance` の 4 項目、label の範囲 |
| `unique` | 本文 + 質問の正規化キーが全件で一意 |
| `group_splits` | 同じ `group_id` が複数の split に出ていない |
| `gold_isolation` | Gold（手作り評価セット + JevBench gold）の本文が `train` に無い |
| `near_duplicates` | 文字 3-gram の Jaccard 0.95 以上の組を報告。**reserve を除く split をまたぐ組があれば失敗** |

近似重複は「同じテンプレートで数値だけ違う」組が多く出る。ラベルが違う組は難しい事例なので
残す。異なる split にまたがる組だけを漏洩として止める。
