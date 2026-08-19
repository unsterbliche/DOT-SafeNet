# Academic Figure Skill Asset Confirmation (verified against local Figure 2c)
# (a-f) marginal regression -> figure_2/scripts/plot_panel_c.py -> param inherit
# RULE: The six archived pK datasets are rendered without row selection or numerical changes.
"""Compose Supplementary Figure 4 using the Figure 2c visual system."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import plot_panel_a
import plot_panel_b
import plot_panel_c
import plot_panel_d
import plot_panel_e
import plot_panel_f
from common.figure_style import MM_TO_INCH, save_publication_figure

OUTPUT = PACKAGE_ROOT / "supplementary_figure_4" / "outputs" / "supplementary_figure_4"


def main():
    fig = plt.figure(figsize=(88 * MM_TO_INCH, 132 * MM_TO_INCH), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        left=0.12,
        right=0.98,
        bottom=0.07,
        top=0.98,
        wspace=0.12,
        hspace=0.18,
    )
    modules = [
        plot_panel_a,
        plot_panel_b,
        plot_panel_c,
        plot_panel_d,
        plot_panel_e,
        plot_panel_f,
    ]
    for index, module in enumerate(modules):
        ax = module.draw(fig, grid[index // 2, index % 2], label="")
        row, column = divmod(index, 2)
        if row < 2:
            ax.set_xlabel("")
        if column > 0:
            ax.set_ylabel("")
    save_publication_figure(fig, OUTPUT, write_tiff=True)
    plt.close(fig)


if __name__ == "__main__":
    main()