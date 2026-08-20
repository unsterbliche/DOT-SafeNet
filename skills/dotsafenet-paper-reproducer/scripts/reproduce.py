#!/usr/bin/env python3
"""Stable command-line entrypoint for the DOT-SafeNet manuscript skill."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MARKERS = (
    "data/model/target_order.csv",
    "scripts/validate_release.py",
    "configs/pipeline.yaml",
)


def is_release(path: Path) -> bool:
    return path.is_dir() and all((path / marker).is_file() for marker in MARKERS)


def find_release(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("DOTSAFENET_RELEASE_ROOT"):
        candidates.append(Path(os.environ["DOTSAFENET_RELEASE_ROOT"]))
    if os.environ.get("MULTISAFE_RELEASE_ROOT"):
        candidates.append(Path(os.environ["MULTISAFE_RELEASE_ROOT"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    skill_file = Path(__file__).resolve()
    candidates.extend(skill_file.parents)
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if is_release(candidate):
            return candidate
    raise SystemExit(
        "DOT-SafeNet publication release root not found. Pass --release-root or set DOTSAFENET_RELEASE_ROOT."
    )


def execute(root: Path, command: list[str]) -> None:
    print("release_root:", root)
    print("python:", sys.executable)
    print("command:", " ".join(command))
    subprocess.run([sys.executable, *command], cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    subparsers.add_parser("audit")
    subparsers.add_parser("metrics")
    subparsers.add_parser("tests")

    figures = subparsers.add_parser("figures")
    figures.add_argument("--figure", action="append", default=[])
    figures.add_argument("--check-only", action="store_true")
    figures.add_argument("--report")

    weights = subparsers.add_parser("weights")
    weights.add_argument("--checkpoint-root", required=True)
    weights.add_argument("--existence-only", action="store_true")
    weights.add_argument("--report")

    args = parser.parse_args()
    root = find_release(args.release_root)
    if args.operation == "audit":
        execute(root, ["scripts/validate_release.py"])
    elif args.operation == "metrics":
        execute(root, ["scripts/reproduce_metrics.py"])
    elif args.operation == "tests":
        execute(root, ["scripts/run_tests.py"])
    elif args.operation == "figures":
        command = ["scripts/render_all_figures.py"]
        for name in args.figure:
            command.extend(["--figure", name])
        if args.check_only:
            command.append("--check-only")
        if args.report:
            command.extend(["--report", args.report])
        execute(root, command)
    elif args.operation == "weights":
        command = ["scripts/check_weights.py", "--checkpoint-root", args.checkpoint_root]
        if args.existence_only:
            command.append("--existence-only")
        if args.report:
            command.extend(["--json-out", args.report])
        execute(root, command)


if __name__ == "__main__":
    main()
