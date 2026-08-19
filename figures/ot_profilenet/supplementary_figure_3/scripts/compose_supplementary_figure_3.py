"""Compose Supplementary Figure 3 from four independently executable panels."""

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
from common.figure_style import MM_TO_INCH, save_publication_figure

OUTPUT = PACKAGE_ROOT / "supplementary_figure_3" / "outputs" / "supplementary_figure_3"


def main():
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM_TO_INCH, 132 * MM_TO_INCH))
    plot_panel_a.draw(axes[0, 0], label="", show_legend=False)
    plot_panel_b.draw(axes[0, 1], label="", show_legend=False)
    plot_panel_c.draw(axes[1, 0], label="", show_legend=False)
    plot_panel_d.draw(axes[1, 1], label="", show_legend=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.98, wspace=0.28, hspace=0.34)
    save_publication_figure(fig, OUTPUT, write_tiff=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

