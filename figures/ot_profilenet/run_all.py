"""Generate every independently reproducible panel and complete figure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    path = ROOT / script
    print("[figure] {}".format(path.relative_to(ROOT)))
    subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing-sfig2b",
        action="store_true",
        help="Generate all available outputs while Supplementary Figure 2b source data are absent.",
    )
    args = parser.parse_args()

    run("prepare_source_data.py")

    scripts = [
        "figure_2/scripts/plot_panel_a.py",
        "figure_2/scripts/plot_panel_b.py",
        "figure_2/scripts/plot_panel_c.py",
        "figure_2/scripts/plot_panel_d.py",
        "figure_2/scripts/plot_panel_e.py",
        "figure_2/scripts/plot_panel_f.py",
        "figure_2/scripts/compose_figure_2.py",
        "supplementary_figure_2/scripts/plot_panel_a.py",
        "supplementary_figure_2/scripts/plot_panel_c.py",
        "supplementary_figure_2/scripts/plot_panel_d.py",
        "supplementary_figure_3/scripts/plot_panel_a.py",
        "supplementary_figure_3/scripts/plot_panel_b.py",
        "supplementary_figure_3/scripts/plot_panel_c.py",
        "supplementary_figure_3/scripts/plot_panel_d.py",
        "supplementary_figure_3/scripts/compose_supplementary_figure_3.py",
        "supplementary_figure_4/scripts/plot_panel_a.py",
        "supplementary_figure_4/scripts/plot_panel_b.py",
        "supplementary_figure_4/scripts/plot_panel_c.py",
        "supplementary_figure_4/scripts/plot_panel_d.py",
        "supplementary_figure_4/scripts/plot_panel_e.py",
        "supplementary_figure_4/scripts/plot_panel_f.py",
        "supplementary_figure_4/scripts/compose_supplementary_figure_4.py",
    ]
    for script in scripts:
        run(script)

    panel_b = ROOT / "supplementary_figure_2" / "data" / "panel_b_compounds_per_target.csv"
    if panel_b.exists():
        run("supplementary_figure_2/scripts/plot_panel_b.py")
        run("supplementary_figure_2/scripts/compose_supplementary_figure_2.py")
    elif not args.allow_missing_sfig2b:
        raise FileNotFoundError(
            "Supplementary Figure 2b source data are absent. Set SFIG2B_SOURCE or use "
            "--allow-missing-sfig2b for an explicitly incomplete internal build."
        )
    else:
        print("[missing] Supplementary Figure 2b and its complete composition were not generated")

    validator = [sys.executable, str(ROOT / "validate_package.py")]
    if args.allow_missing_sfig2b:
        validator.append("--allow-missing-sfig2b")
    subprocess.run(validator, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()

