"""Standalone legend for Figure 2d."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common.figure_style import METHOD_COLORS, MM_TO_INCH, save_publication_figure
from figure_2.scripts.plot_panel_d import MODELS

OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2d_legend"


def main():
    fig, ax = plt.subplots(figsize=(78 * MM_TO_INCH, 24 * MM_TO_INCH))
    ax.axis("off")
    handles = [Patch(facecolor=METHOD_COLORS[model], edgecolor="none", label=model) for model in MODELS]
    top = ax.legend(
        handles=handles[:3],
        labels=MODELS[:3],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=3,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.1,
        frameon=False,
    )
    ax.add_artist(top)
    ax.legend(
        handles=handles[3:],
        labels=["OT-ProfileNet\n(from-scratch)", "OT-ProfileNet\n(fine-tuned)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.18),
        ncol=2,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()