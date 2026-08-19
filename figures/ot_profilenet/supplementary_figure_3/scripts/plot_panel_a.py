"""Supplementary Figure 3a: pretraining and validation loss."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, save_publication_figure

DATA = PACKAGE_ROOT / "supplementary_figure_3" / "data" / "panel_a_pretraining_loss.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_3" / "outputs" / "panels" / "supplementary_figure_3a"


def draw(ax, data_path=DATA, label="", show_legend=True):
    data = pd.read_csv(data_path)
    if data["epoch"].min() != 1 or data["epoch"].max() != 100:
        raise ValueError("Expected epochs 1–100 for Supplementary Figure 3a")
    ax.plot(data["epoch"], data["train_loss"], color=COLORS["baseline_dark"], lw=1.2,
            label="Training")
    ax.plot(data["epoch"], data["val_loss"], color=COLORS["orange"], lw=1.2,
            label="Validation")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Binary cross-entropy loss", fontsize=9)
    clean_axis(ax, grid_axis="y")
    ax.tick_params(labelsize=8, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(COLORS["black"])
    if show_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8)
    if label:
        add_panel_label(ax, label)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(82 * MM_TO_INCH, 62 * MM_TO_INCH))
    draw(ax, show_legend=False)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.96)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

