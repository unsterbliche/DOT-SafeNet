#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def is_release(candidate: Path) -> bool:
    return (
        (candidate / "scripts/profile_molecule.py").is_file()
        and (candidate / "data/manifests/target_order.csv").is_file()
    )


def find_release_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("DOTSAFENET_RELEASE_ROOT"):
        candidates.append(Path(os.environ["DOTSAFENET_RELEASE_ROOT"]))
    if os.environ.get("MULTISAFE_RELEASE_ROOT"):
        candidates.append(Path(os.environ["MULTISAFE_RELEASE_ROOT"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    start = Path(__file__).resolve()
    candidates.extend([start, *start.parents])
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if is_release(candidate):
            return candidate
    raise SystemExit(
        "DOT-SafeNet publication release root not found. Pass --release-root or set DOTSAFENET_RELEASE_ROOT."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--release-root")
    args, remaining = parser.parse_known_args()
    root = find_release_root(args.release_root)
    print("release_root:", root)
    raise SystemExit(subprocess.call([sys.executable, str(root / "scripts/profile_molecule.py"), *remaining], cwd=root))
