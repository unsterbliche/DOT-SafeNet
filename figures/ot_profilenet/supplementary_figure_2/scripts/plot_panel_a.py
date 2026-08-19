"""Supplementary Figure 2a: protein sequence-length distribution."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, save_publication_figure

DATA = PACKAGE_ROOT / "supplementary_figure_2" / "data" / "panel_a_protein_sequence_lengths.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_2" / "outputs" / "panels" / "supplementary_figure_2a"


def draw(ax, data_path=DATA, label=""):
    data = pd.read_csv(data_path)
    values = data["sequence_length"].to_numpy(float)
    if values.size != 7258 or (values <= 0).any():
        raise ValueError("Unexpected protein sequence-length data")
    bins = np.geomspace(values.min(), values.max(), 42)
    ax.hist(values, bins=bins, color=COLORS["baseline_mid"], edgecolor="white", linewidth=0.2)
    ax.axvline(2000, color=COLORS["red"], lw=1.0, ls=(0, (3, 2)))
    ax.text(2000, ax.get_ylim()[1] * 0.95, "2,000 aa", color=COLORS["red"],
            ha="right", va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Protein sequence length (amino acids; log scale)", fontsize=9)
    ax.set_ylabel("Number of proteins", fontsize=9)
    clean_axis(ax, grid_axis="y")
    ax.tick_params(labelsize=8, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(COLORS["black"])
    if label:
        add_panel_label(ax, label)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(82 * MM_TO_INCH, 62 * MM_TO_INCH))
    draw(ax)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.95)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

