#!/usr/bin/env python3
"""Render all code-derived manuscript figures from the release manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "figure_manifest.csv"


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_rows(rows: list[dict[str, str]], requested: list[str]) -> list[dict[str, str]]:
    if not requested:
        return [row for row in rows if row["entrypoint"]]
    wanted = {name.casefold() for name in requested}
    selected = [row for row in rows if row["figure"].casefold() in wanted]
    missing = wanted - {row["figure"].casefold() for row in selected}
    if missing:
        raise SystemExit("Unknown figure name(s): " + ", ".join(sorted(missing)))
    return [row for row in selected if row["entrypoint"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", action="append", default=[], help="Exact manuscript figure name; repeat as needed")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = load_rows()
    if args.list:
        for row in rows:
            print(f"{row['manuscript_order']:>2}  {row['figure']:<24} {row['generation']}")
        return

    selected = choose_rows(rows, args.figure)
    missing = [row["entrypoint"] for row in selected if not (ROOT / row["entrypoint"]).is_file()]
    if missing:
        raise SystemExit("Missing entrypoints:\n" + "\n".join(missing))
    if args.check_only:
        print(f"OK: {len(rows)} manuscript figures; {len(selected)} code-derived entrypoints")
        return

    report = {"python": sys.executable, "figures": [], "status": "passed"}
    for row in selected:
        entrypoint = ROOT / row["entrypoint"]
        started = time.time()
        print(f"[render] {row['figure']} -> {row['entrypoint']}", flush=True)
        result = subprocess.run([sys.executable, str(entrypoint)], cwd=entrypoint.parent)
        report["figures"].append({
            "figure": row["figure"],
            "entrypoint": row["entrypoint"],
            "returncode": result.returncode,
            "seconds": round(time.time() - started, 3),
        })
        if result.returncode:
            report["status"] = "failed"
            break

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
