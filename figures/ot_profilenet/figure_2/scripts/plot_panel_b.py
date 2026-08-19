"""Figure 2b: target, compound and interaction counts by target family."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, save_publication_figure

DATA = PACKAGE_ROOT / "figure_2" / "data" / "panel_b_dataset_statistics.csv"
OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2b"

METRICS = [
    ("target_count", "Targets"),
    ("compound_count", "Compounds"),
    ("compound_target_pair_count", "Drug–target pairs"),
]


def draw(fig, subplot_spec, data_path=DATA, label=""):
    data = pd.read_csv(data_path)
    required = {"target_family"} | {item[0] for item in METRICS}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Figure 2b source data are incomplete")
    order = ["GPCR", "Ion channel", "Enzyme", "Kinases", "NR", "Transporter", "Other"]
    data = data.set_index("target_family").loc[order].reset_index()
    grid = subplot_spec.subgridspec(1, 3, wspace=0.30)
    axes = []
    y = np.arange(len(data))
    for index, (column, title) in enumerate(METRICS):
        ax = fig.add_subplot(grid[index])
        axes.append(ax)
        values = data[column].to_numpy(float)
        colors = [COLORS["baseline_dark"] if name == "GPCR" else COLORS["baseline_soft"]
                  for name in data["target_family"]]
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_title(title, loc="left", fontsize=7, fontweight="bold")
        ax.set_ylim(-0.6, len(data) - 0.4)
        if index == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(data["target_family"])
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        clean_axis(ax, grid_axis="x")
    if label:
        add_panel_label(axes[0], label, x=-0.42, y=1.08)
    return axes


def main():
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 58 * MM_TO_INCH))
    spec = fig.add_gridspec(1, 1, left=0.10, right=0.98, bottom=0.20, top=0.90)[0]
    draw(fig, spec)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

