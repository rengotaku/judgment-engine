"""データ生成の CLI。

    python -m judgment_engine.data.cli --out datasets/generated/v1

課金なしで 5,000 件を作る。実教師でラベルを付け直すときだけ `--relabel openai --allow-paid`。

出力:
- samples.jsonl  生成データ（1 行 1 件）
- manifest.json  件数・内訳・家族ごとの達成数・ファイルハッシュ
- checks.json    検査の結果（合否と根拠）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .builder import BUILDER_VERSION, build_dataset, relabel, write_jsonl
from .catalog import FAMILIES
from .checks import (
    CheckFailure,
    check_gold_isolation,
    check_group_splits,
    check_required_fields,
    check_unique,
    near_duplicate_pairs,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORED_DIR = REPO_ROOT / "datasets" / "authored"
JEVBENCH_DIR = REPO_ROOT / "data" / "external" / "jevbench"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_gold_rows() -> list[dict[str, Any]]:
    paths: list[Path] = sorted(AUTHORED_DIR.glob("*.jsonl"))
    if JEVBENCH_DIR.exists():
        paths.extend(sorted(JEVBENCH_DIR.rglob("*.jsonl")))
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("reliability") == "gold":
                rows.append(row)
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_adapter(args: argparse.Namespace) -> Any:
    if args.relabel == "none":
        return None
    if args.relabel == "openai":
        if not args.allow_paid:
            raise SystemExit("実教師は課金されます。使う場合は --allow-paid を明示してください")
        from ..teacher.openai_compatible import OpenAICompatibleTeacher

        return OpenAICompatibleTeacher.from_env(
            model=args.teacher_model,
            base_url=args.teacher_base_url,
        )
    raise SystemExit(f"未知の --relabel: {args.relabel}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="蒸留データの生成と検査")
    parser.add_argument("--out", type=Path, default=Path("datasets/generated/v1"))
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--reserve", type=int, default=50)
    parser.add_argument("--relabel", choices=["none", "openai"], default="none")
    parser.add_argument("--allow-paid", action="store_true", help="実教師（課金あり）を許可する")
    parser.add_argument("--teacher-model", default="deepseek-chat")
    parser.add_argument("--teacher-base-url", default="https://api.deepseek.com/v1")
    args = parser.parse_args(argv)

    adapter = build_adapter(args)
    samples = build_dataset(seed=args.seed, reserve=args.reserve)
    if adapter is not None:
        samples = relabel(samples, adapter)

    checks: dict[str, Any] = {}

    def run_check(name: str, func, *call_args) -> None:
        try:
            func(*call_args)
            checks[name] = {"ok": True}
        except CheckFailure as exc:
            checks[name] = {"ok": False, "detail": str(exc)}

    run_check("required_fields", check_required_fields, samples)
    run_check("unique", check_unique, samples)
    run_check("group_splits", check_group_splits, samples)
    run_check("gold_isolation", check_gold_isolation, samples, load_gold_rows())

    duplicates = near_duplicate_pairs(samples, limit=200)
    split_of = {sample["id"]: sample["split"] for sample in samples}
    # reserve は学習にも評価にも使わない buffer。split のまたぎ判定から外す。
    cross_split = [
        pair
        for pair in duplicates
        if split_of[pair[0]] != split_of[pair[1]]
        and "reserve" not in {split_of[pair[0]], split_of[pair[1]]}
    ]
    checks["near_duplicates"] = {
        "ok": not cross_split,
        "pairs_ge_0_95": len(duplicates),
        "cross_split_pairs": len(cross_split),
        "cross_split_examples": cross_split[:5],
        "examples": duplicates[:5],
    }

    counts = summarize(samples)
    shortfalls = {}
    for family in FAMILIES:
        produced = sum(1 for sample in samples if sample["family"] == family.name)
        if produced < family.target:
            shortfalls[family.name] = {"target": family.target, "produced": produced}

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    write_jsonl(samples_path, samples)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "builder_version": BUILDER_VERSION,
        "seed": args.seed,
        "teacher": None if adapter is None else {"model": adapter.model, "prompt_version": adapter.prompt_version},
        "counts": counts,
        "targets": {
            "total": sum(family.target for family in FAMILIES) + args.reserve,
            "by_type": {
                name: sum(family.target for family in FAMILIES if family.qtype == name)
                for name in ("choice", "noul", "score")
            },
            "reserve": args.reserve,
        },
        "shortfalls": shortfalls,
        "families": [
            {
                "name": family.name,
                "type": family.qtype,
                "domain": family.domain,
                "target": family.target,
                "reserved": family.reserved,
            }
            for family in FAMILIES
        ],
        "files": {"samples.jsonl": digest(samples_path)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = [name for name, result in checks.items() if not result["ok"]]
    print(f"rows={counts['rows']} groups={counts['groups']} shortfalls={len(shortfalls)}")
    print(f"by_type={counts['by_type']} (main={counts['by_type_main']}, reserve={counts['by_type_reserve']})")
    print(f"by_split={counts['by_split']}")
    print(f"by_language={counts['by_language']}")
    print(f"out={out_dir}")
    if shortfalls:
        print(f"shortfalls={shortfalls}")
    if failed:
        print(f"FAILED checks: {failed}")
        return 1
    print("checks: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
