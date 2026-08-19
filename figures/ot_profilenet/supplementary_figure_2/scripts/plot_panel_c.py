"""Supplementary Figure 2c: positive and negative pretraining pairs."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, save_publication_figure

DATA = PACKAGE_ROOT / "supplementary_figure_2" / "data" / "panel_c_class_balance.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_2" / "outputs" / "panels" / "supplementary_figure_2c"


def draw(ax, data_path=DATA, label=""):
    data = pd.read_csv(data_path).set_index("measure")
    order = ["positive_pairs", "negative_pairs"]
    labels = ["Positive", "Negative"]
    values = data.loc[order, "value"].to_numpy(float)
    bars = ax.bar(labels, values, color=[COLORS["baseline_dark"], COLORS["baseline_soft"]],
                  edgecolor="white", linewidth=0.3)
    ax.set_ylim(0, values.max() * 1.18)
    ax.set_ylabel("Drug–target pairs", fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.025, "{:,}".format(int(value)),
                ha="center", va="bottom", fontsize=8)
    clean_axis(ax, grid_axis="y")
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.yaxis.get_offset_text().set_fontsize(9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(COLORS["black"])
    if label:
        add_panel_label(ax, label)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(82 * MM_TO_INCH, 50 * MM_TO_INCH))
    draw(ax, label="")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.97)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

