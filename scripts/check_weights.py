#!/usr/bin/env python3
"""Check checkpoint files against weights/weights_manifest.tsv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RELEASE = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--existence-only", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    with (RELEASE / "weights" / "weights_manifest.tsv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    errors = []
    for row in rows:
        path = args.checkpoint_root / row["path"]
        if not path.is_file():
            errors.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append(f"size mismatch: {row['path']}")
        if not args.existence_only and digest(path) != row["sha256"]:
            errors.append(f"SHA256 mismatch: {row['path']}")
    report = {
        "status": "failed" if errors else "passed",
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "files": len(rows),
        "hashes_checked": not args.existence_only,
        "errors": errors,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
