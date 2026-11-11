# Phase 0 のベースライン（未学習）

- 測定日: 2026-11-11
- モデル: `BAAI/bge-reranker-v2-m3`（多言語 Cross-Encoder、学習なし）
- 推論: CPU（`JE_TORCH_THREADS=8`）。GPU は使っていない
- 温度: `temperature=1.0`（校正はしていない）
- routing: `abstain_below=0.6`、`minimum_margin=0.1`

Phase 2 で蒸留学習したモデルは、このファイルの数値を上回ることを合格条件にする。
数値の定義は `src/judgment_engine/eval/metrics.py` にあり、指標の計算をここで独自に書かない。

## 何を測ったか

| データ | 件数 | 内容 |
|---|---|---|
| 同梱セット | 100 | iid / unseen_question / unseen_criteria / unseen_domain / adversarial / japanese |
| JevBench gold | 1,682 | `model-collapse/jev-bench` の `reliability == "gold"`。commit `fcec62f13ce9` |

JevBench のデータはリポジトリに含めず、`make fetch-jevbench` で固定 commit から取得する
（`data/manifests/jevbench-fcec62f13ce9.json` にファイルハッシュを残してある）。

## 結果: 同梱 100 件

**表1: 同梱セットの指標**

| 群 | 件数 | Accuracy | Coverage | Coverage の Accuracy |
|---|---:|---:|---:|---:|
| 全体 | 100 | 0.570 | 0.500 | 0.600 |
| iid | 30 | 0.600 | 0.500 | — |
| unseen_question | 15 | 0.467 | 0.533 | — |
| unseen_criteria | 15 | 0.533 | 0.533 | — |
| unseen_domain | 15 | 0.600 | 0.467 | — |
| adversarial | 10 | 0.400 | 0.500 | — |
| japanese | 15 | 0.733 | 0.467 | — |

全体の Brier は 0.213、ECE は 0.163。
Coverage は `routing.route == "local"` になった割合（teacher / human へ回さなかった割合）、
Coverage の Accuracy はその部分集合だけの正答率。

## 結果: JevBench gold 1,682 件

**表2: JevBench gold の指標**

| 型 | 件数 | Accuracy | Brier | ECE | Coverage | Coverage の Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 全体 | 1,682 | 0.477 | 0.206 | 0.122 | 0.255 | 0.671 |
| choice | 750 | 0.532 | 0.138 | 0.132 | 0.292 | 0.858 |
| noul | 625 | 0.571 | 0.242 | 0.024 | 0.162 | 0.663 |
| score | 307 | 0.150 | 0.299 | 0.420 | 0.355 | 0.303 |

score は MAE 1.204、QWK −0.062。

score の QWK が負なのは、順序を当てるのではなく「それらしいレベル」に寄せているため。
JevBench の公開リーダーボードでも cross-encoder（bge-reranker）は score-QWK −0.16 で、
同じ傾向が出ている。**正解率 15% は偶然（3 レベルなら 33%）より低い**ので、
score は「未学習の Cross-Encoder では使い物にならない」という前提で Phase 2 に入る。

## 結果: レイテンシ（P50 / P95）

`repeat=5`、ウォームアップ 2 回を捨てた値。1 リクエストの処理時間で、HTTP は含まない。

**表3: 入力長と候補数ごとの処理時間（ミリ秒）**

| 入力 | 質問 | 組数 | P50 | P95 |
|---|---|---:|---:|---:|
| 200 文字 | choice 2 候補 | 2 | 115 | 128 |
| 200 文字 | choice 4 候補 | 4 | 220 | 329 |
| 200 文字 | choice 8 候補 | 8 | 462 | 679 |
| 200 文字 | score 3 レベル | 3 | 155 | 170 |
| 200 文字 | score 10 レベル | 10 | 485 | 715 |
| 200 文字 | noul | 2 | 105 | 134 |
| 200 文字 | 8 質問（計 28 組） | 28 | 1,452 | 2,212 |
| 1,000 文字 | choice 2 候補 | 2 | 364 | 434 |
| 1,000 文字 | choice 8 候補 | 8 | 1,630 | 1,923 |
| 1,000 文字 | 8 質問（計 28 組） | 28 | 6,082 | 6,284 |
| 4,000 文字 | choice 2 候補 | 2 | 657 | 790 |
| 4,000 文字 | choice 8 候補 | 8 | 3,475 | 4,052 |
| 4,000 文字 | noul | 2 | 635 | 723 |
| 4,000 文字 | 8 質問（計 28 組） | 28 | 11,396 | 12,948 |

設計案の目標は P95 100ms だが、**最短の構成（200 文字・2 候補）でも P95 128ms** で届いていない。
100ms を狙うには、このモデル（568M パラメータ）より小さいエンコーダか、
M0（Embedding + 小さな Head）の経路が要る。

候補数にほぼ比例して増えるので、コストの大半は候補ごとの forward 回数で決まる。
`JE_TORCH_THREADS` を 16 / 28 に上げると遅くなった（実測: 200 文字 4 候補で 235ms → 366ms → 2,801ms）。
8 が既定値なのはこのため。ただし測定は他セッションと CPU を共有した状態で、
スレッド数の比較は 3 回だけなので、この順序以上は言えない。

## 読み方の注意

- **Brier は候補数で割った平均**（0〜1）。2 値では通常の Brier と同じ値になる
- **Coverage の Accuracy が全体より高いとは限らない**。同梱セットでは 0.600 対 0.570 で、
  校正なしの確からしさでは選別がほとんどできていない。これは Phase 3 の校正で扱う
- **JevBench の公式リーダーボード（232 件の gold サンプル）とは件数が違う**。
  bge-reranker は 40.9、こちらは 47.7 だが、同じ土俵の比較ではない。帯が同じ程度、とだけ読む
- score の Accuracy は「丸めた予測レベルが一致した割合」。丸めの分だけ score の連続値より厳しい

## 再現

```bash
make install-ce
make fetch-jevbench
.venv/bin/python -m judgment_engine.eval.cli --authored --backend cross_encoder --tag authored --out evals/runs/authored.json
.venv/bin/python -m judgment_engine.eval.cli --jevbench --gold-only --backend cross_encoder --tag jevbench-gold --out evals/runs/jevbench-gold.json
make eval-latency
```

同じ入力なら同じ指標になる（レイテンシだけは変わる）。`generated_at` と `p50/p95/p99` を除いて
2 回の出力を突き合わせれば一致する。

## 未確認

- 温度（`temperature`）の調整。1.0 のままで、校正はしていない
- JevBench の gold 以外 1,252 件。Phase 0 では gold のみ
- `JE_TORCH_THREADS` の最適値。3 回の試行だけで、負荷の影響を切り分けていない
- GPU で動かした場合の数字
