"""JevBench（model-collapse/jev-bench）を固定 commit で取得する。

データはこのリポジトリに含めない（由来が複数あり、ライセンスが混在するため）。
取得したファイルのハッシュを manifest に残し、評価結果からは manifest を参照する。

    python scripts/fetch_jevbench.py --ref fcec62f13ce961115af0d16317f0ff15986a9a25
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DEFAULT_REF = "fcec62f13ce961115af0d16317f0ff15986a9a25"
REPO = "model-collapse/jev-bench"
TOPICS = (
    "agent_trace_observability",
    "banking",
    "commerce",
    "customer_service",
    "invoice_processing",
    "invoices",
    "logic",
    "moderation",
    "observability",
    "org",
    "quant",
    "reviews",
    "security_incidents",
    "support",
)
TYPES = ("choice", "noul", "score")
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/data/{topic}/{type}.jsonl"
OUTPUT = Path("data/external/jevbench")
MANIFEST_DIR = Path("data/manifests")


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:  # noqa: S310 - 固定の https URL
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    target = args.out / args.ref
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"repo": REPO, "ref": args.ref, "files": {}}
    files = manifest["files"]
    assert isinstance(files, dict)

    for topic in TOPICS:
        for type_name in TYPES:
            url = RAW.format(repo=REPO, ref=args.ref, topic=topic, type=type_name)
            try:
                content = fetch(url)
            except Exception as exc:  # noqa: BLE001 - 404 は「その組み合わせが無い」だけ
                print(f"skip {topic}/{type_name}: {exc}")
                continue
            path = target / topic / f"{type_name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            files[str(path.relative_to(target))] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "rows": content.count(b"\n"),
            }
            print(f"ok   {topic}/{type_name} ({len(content)} bytes)")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"jevbench-{args.ref[:12]}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
