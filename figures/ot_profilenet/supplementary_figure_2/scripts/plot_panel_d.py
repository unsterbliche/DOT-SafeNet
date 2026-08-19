"""Supplementary Figure 2d: overall pretraining dataset summary."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, save_publication_figure

DATA = PACKAGE_ROOT / "supplementary_figure_2" / "data" / "panel_d_dataset_summary.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_2" / "outputs" / "panels" / "supplementary_figure_2d"


def draw(ax, data_path=DATA, label="d"):
    data = pd.read_csv(data_path).set_index("measure")
    order = ["protein_targets", "unique_compounds", "total_pairs"]
    labels = ["Protein\ntargets", "Unique\ncompounds", "Drug–target\npairs"]
    values = data.loc[order, "value"].to_numpy(float)
    bars = ax.bar(labels, values, color=[COLORS["baseline_soft"], COLORS["baseline_mid"],
                                        COLORS["baseline_dark"]], edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_ylim(10 ** 3, values.max() * 2.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, "{:,}".format(int(value)),
                ha="center", va="bottom", fontsize=6)
    clean_axis(ax, grid_axis="y")
    if label:
        add_panel_label(ax, label)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(89 * MM_TO_INCH, 67 * MM_TO_INCH))
    draw(ax)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.22, top=0.92)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

