#!/usr/bin/env python3
"""Render code-derived DOT-SafeNet manuscript figures in manuscript order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Supplementary Figures 1 and 5 are model schematics and are supplied as final TIFFs.
FIGURES = (
    ("Figure 1", "reproduce/figure_1/run.py"),
    ("Figure 2", "reproduce/figure_2/run.py"),
    ("Figure 3", "reproduce/figure_3/run.py"),
    ("Figure 4", "reproduce/figure_4/run.py"),
    ("Figure 5", "reproduce/figure_5/run.py"),
    ("Figure 6", "reproduce/figure_6/run.py"),
    ("Supplementary Figure 1", ""),
    ("Supplementary Figure 2", "reproduce/supplementary_figure_2/run.py"),
    ("Supplementary Figure 3", "reproduce/supplementary_figure_3/run.py"),
    ("Supplementary Figure 4", "reproduce/supplementary_figure_4/run.py"),
    ("Supplementary Figure 5", ""),
    ("Supplementary Figure 6", "reproduce/supplementary_figure_6/run.py"),
    ("Supplementary Figure 7", "reproduce/supplementary_figure_7/run.py"),
    ("Supplementary Figure 8", "reproduce/supplementary_figure_8/run.py"),
)


def choose_figures(requested: list[str]) -> list[tuple[str, str]]:
    if not requested:
        return [(name, path) for name, path in FIGURES if path]
    wanted = {name.casefold() for name in requested}
    selected = [(name, path) for name, path in FIGURES if name.casefold() in wanted]
    missing = wanted - {name.casefold() for name, _ in selected}
    if missing:
        raise SystemExit("Unknown figure name(s): " + ", ".join(sorted(missing)))
    unavailable = [name for name, path in selected if not path]
    if unavailable:
        raise SystemExit("No code-derived entrypoint for: " + ", ".join(unavailable))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", action="append", default=[], help="Exact manuscript figure name")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.list:
        for index, (name, entrypoint) in enumerate(FIGURES, 1):
            print(f"{index:>2}  {name:<24} {entrypoint or 'final TIFF only'}")
        return

    selected = choose_figures(args.figure)
    missing = [path for _, path in selected if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing entrypoints:\n" + "\n".join(missing))
    if args.check_only:
        print(f"OK: {len(FIGURES)} manuscript figures; {len(selected)} code-derived entrypoints")
        return

    report = {"python": sys.executable, "figures": [], "status": "passed"}
    render_env = os.environ.copy()
    render_env.setdefault("MPLBACKEND", "Agg")
    for name, relative_path in selected:
        entrypoint = ROOT / relative_path
        started = time.time()
        print(f"[render] {name} -> {relative_path}", flush=True)
        result = subprocess.run(
            [sys.executable, str(entrypoint)],
            cwd=entrypoint.parent,
            env=render_env,
        )
        report["figures"].append({
            "figure": name,
            "entrypoint": relative_path,
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
