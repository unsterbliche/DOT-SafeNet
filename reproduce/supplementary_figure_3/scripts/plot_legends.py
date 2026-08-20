"""Export the Supplementary Figure 3 legends as independent artwork."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common.figure_style import COLORS, METHOD_COLORS, MM_TO_INCH, save_publication_figure
from metric_panel import MODELS

OUTPUT_DIR = PACKAGE_ROOT / "supplementary_figure_3" / "outputs" / "legends"


def save_legend(handles, labels, output, width_mm, height_mm, ncol):
    fig = plt.figure(figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=8,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0,
    )
    save_publication_figure(fig, output)
    plt.close(fig)


def main():
    loss_handles = [
        Line2D([0], [0], color=COLORS["baseline_dark"], lw=1.2),
        Line2D([0], [0], color=COLORS["orange"], lw=1.2),
    ]
    save_legend(
        loss_handles,
        ["Training", "Validation"],
        OUTPUT_DIR / "supplementary_figure_3a_legend",
        width_mm=58,
        height_mm=9,
        ncol=2,
    )

    model_handles = [Patch(facecolor=METHOD_COLORS[name], edgecolor="none") for name in MODELS]
    save_legend(
        model_handles,
        MODELS,
        OUTPUT_DIR / "supplementary_figure_3b-d_legend",
        width_mm=150,
        height_mm=9,
        ncol=3,
    )


if __name__ == "__main__":
    main()