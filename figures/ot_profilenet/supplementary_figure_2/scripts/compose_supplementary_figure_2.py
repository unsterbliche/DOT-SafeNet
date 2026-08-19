"""Compose Supplementary Figure 2 after all four panel data files are available."""

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

OUTPUT = PACKAGE_ROOT / "supplementary_figure_2" / "outputs" / "supplementary_figure_2"


def main():
    plot_panel_b.load_data()
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM_TO_INCH, 132 * MM_TO_INCH))
    plot_panel_a.draw(axes[0, 0], label="a")
    plot_panel_b.draw(axes[0, 1], label="b")
    plot_panel_c.draw(axes[1, 0], label="")
    plot_panel_d.draw(axes[1, 1], label="d")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.97, wspace=0.30, hspace=0.38)
    save_publication_figure(fig, OUTPUT, write_tiff=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

