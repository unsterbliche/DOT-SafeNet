"""Compose Figure 2 from the same drawing functions used for each panel."""

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

OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "figure_2"


def main():
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 222 * MM_TO_INCH), facecolor="white")
    outer = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.62, 1.55, 0.90],
        left=0.055,
        right=0.992,
        bottom=0.045,
        top=0.985,
        hspace=0.27,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(outer[0, 0:2])
    plot_panel_a.draw(ax_a, label="")
    plot_panel_b.draw(fig, outer[0, 2:6], label="")
    plot_panel_c.draw(fig, outer[1, :], label="")
    plot_panel_d.draw(fig, outer[2, 0:3], label="")
    ax_e = fig.add_subplot(outer[2, 3:4])
    plot_panel_e.draw(ax_e, label="")
    ax_f = fig.add_subplot(outer[2, 4:6])
    plot_panel_f.draw(ax_f, label="")
    save_publication_figure(fig, OUTPUT, write_tiff=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

