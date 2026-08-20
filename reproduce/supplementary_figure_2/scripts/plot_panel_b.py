"""Supplementary Figure 2b: distribution of compound counts per target."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, save_publication_figure

DATA = PACKAGE_ROOT / "supplementary_figure_2" / "data" / "panel_b_compounds_per_target.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_2" / "outputs" / "panels" / "supplementary_figure_2b"


def load_data(data_path=DATA):
    if not Path(data_path).exists():
        raise FileNotFoundError(
            "Supplementary Figure 2b requires panel_b_compounds_per_target.csv "
            "with target_id and compound_count columns"
        )
    data = pd.read_csv(data_path)
    if not {"target_id", "compound_count"}.issubset(data.columns) or data.empty:
        raise ValueError("Supplementary Figure 2b source data are incomplete")
    if (data["compound_count"] < 0).any():
        raise ValueError("compound_count cannot be negative")
    return data


def draw(ax, data_path=DATA, label=""):
    data = load_data(data_path)
    values = data["compound_count"].to_numpy(float)
    upper = max(10.0, float(values.max()))
    bins = np.unique(np.concatenate(([0.0, 1.0], np.geomspace(2.0, upper + 1.0, 16))))
    counts, edges = np.histogram(values, bins=bins)
    centers = np.sqrt(np.maximum(edges[:-1], 1.0) * np.maximum(edges[1:], 1.0))
    widths = np.diff(edges) * 0.88
    ax.bar(centers, counts, width=widths, color=COLORS["baseline_mid"],
           edgecolor="white", linewidth=0.25, align="center")
    ax.set_xscale("log")
    ax.set_xlabel("Compounds per target (log scale)", fontsize=9)
    ax.set_ylabel("Number of targets", fontsize=9)
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

