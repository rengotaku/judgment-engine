"""評価の CLI。

    python -m judgment_engine.eval.cli --authored --out evals/runs/authored.json
    python -m judgment_engine.eval.cli --jevbench --gold-only --limit 300 --out evals/runs/jevbench.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DEFAULT_MODEL_ID, Settings
from ..inference.engine import JudgmentEngine
from ..inference.factory import build_scorer
from .datasets import file_digests, load_rows
from .harness import build_report, run_rows, write_report

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORED_DIR = REPO_ROOT / "datasets" / "authored"
JEVBENCH_DIR = REPO_ROOT / "data" / "external" / "jevbench"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="judgment-engine の評価")
    parser.add_argument("--data", nargs="*", default=[], help="JSONL かディレクトリ（JevBench 形式）")
    parser.add_argument("--authored", action="store_true", help="datasets/authored を使う")
    parser.add_argument("--jevbench", action="store_true", help="data/external/jevbench を使う")
    parser.add_argument("--backend", default="cross_encoder", choices=["lexical", "cross_encoder"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--abstain-below", type=float, default=0.60)
    parser.add_argument("--minimum-margin", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="evals/runs/eval.json")
    parser.add_argument("--no-rows", action="store_true", help="行ごとの予測を出力に含めない（既定は含める）")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.data]
    if args.authored:
        paths.append(AUTHORED_DIR)
    if args.jevbench:
        paths.append(JEVBENCH_DIR)
    if not paths:
        raise SystemExit("データがありません（--data / --authored / --jevbench のどれかを指定）")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"見つかりません: {', '.join(missing)}")
    return paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)

    settings = Settings(
        backend=args.backend,
        model_id=args.model_id,
        temperature=args.temperature,
        abstain_below=args.abstain_below,
        minimum_margin=args.minimum_margin,
    )
    rows = load_rows(paths)
    if args.gold_only:
        rows = [row for row in rows if row.reliability == "gold"]
    rows.sort(key=lambda row: (row.source, row.id))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("条件に合う行がありません")

    scorer = build_scorer(settings)
    try:
        engine = JudgmentEngine(scorer, settings)
        predictions = run_rows(engine, rows)
    finally:
        scorer.close()

    report = build_report(
        predictions=predictions,
        rows=rows,
        dataset_paths=paths,
        digests=file_digests(paths),
        settings=settings,
        scorer_name=scorer.name,
        label=args.tag or ("authored" if args.authored and not args.jevbench else "custom"),
    )
    if args.no_rows:
        report.pop("rows", None)
    write_report(Path(args.out), report)

    overall = report["overall"]
    print(f"rows={overall['n']} accuracy={overall['accuracy']} brier={overall['brier']} ece={overall['ece']}")
    print(f"coverage={overall['coverage']} coverage_accuracy={overall['coverage_accuracy']}")
    print(f"p50={overall['p50_ms']}ms p95={overall['p95_ms']}ms p99={overall['p99_ms']}ms")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
