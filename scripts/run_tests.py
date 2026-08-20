#!/usr/bin/env python3
"""Run publication-package tests with the active Python environment."""

from __future__ import annotations

import compileall
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    if not compileall.compile_dir(ROOT / "scripts", quiet=1):
        raise SystemExit("Python compilation failed under scripts/")
    if not compileall.compile_dir(ROOT / "src", quiet=1):
        raise SystemExit("Python compilation failed under src/")
    if not compileall.compile_dir(ROOT / "reproduce", quiet=1):
        raise SystemExit("Python compilation failed under reproduce/")
    run("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v")
    run("scripts/validate_release.py")
    run("scripts/render_all_figures.py", "--check-only")
    with tempfile.TemporaryDirectory(prefix="multisafe_metrics_") as directory:
        run("scripts/reproduce_metrics.py", "--output-dir", directory)
    print("All publication-package tests passed.")


if __name__ == "__main__":
    main()
