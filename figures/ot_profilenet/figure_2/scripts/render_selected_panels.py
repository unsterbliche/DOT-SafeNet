"""Render only the Figure 2 panels selected for manual manuscript assembly."""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "plot_panel_a.py",
    "plot_panel_c.py",
    "plot_panel_d.py",
    "plot_panel_d_legend.py",
    "plot_panel_e.py",
]


def main():
    for name in SCRIPTS:
        subprocess.run([sys.executable, str(SCRIPT_DIR / name)], check=True)


if __name__ == "__main__":
    main()