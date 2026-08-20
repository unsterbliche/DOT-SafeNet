"""Figure 2e: seven-compound external hERG affinity comparison."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, clean_axis, regression_metrics, save_publication_figure

DATA = PACKAGE_ROOT / "figure_2" / "data" / "panel_e_f_herg_external_validation.csv"
OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2e"


def draw(ax, data_path=DATA, label=""):
    data = pd.read_csv(data_path)
    observed = data["pIC50"].to_numpy(float)
    predicted = data["predict_result_pAC50"].to_numpy(float)
    current = regression_metrics(observed, predicted)
    lo = min(observed.min(), predicted.min()) - 0.25
    hi = max(observed.max(), predicted.max()) + 0.25
    limits = (float(lo), float(hi))
    ax.scatter(observed, predicted, s=26, color=COLORS["orange"], edgecolor="white",
               linewidth=0.5, zorder=3)
    ax.plot(limits, limits, color=COLORS["neutral_mid"], lw=0.7, ls=(0, (3, 2)))
    slope, intercept = np.polyfit(observed, predicted, 1)
    xx = np.linspace(limits[0], limits[1], 100)
    ax.plot(xx, slope * xx + intercept, color=COLORS["baseline_dark"], lw=1.0)
    ax.text(0.04, 0.96, "r = {:.3f}\nRMSE = {:.3f}\nn = 7".format(current["PCC"], current["RMSE"]),
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed pIC$_{50}$", fontsize=9)
    ax.set_ylabel("Predicted pAC$_{50}$", fontsize=9)
    ax.text(
        0.96, 0.06, "hERG", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.5,
        fontweight="bold", color=COLORS["black"],
    )
    clean_axis(ax)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=8, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(COLORS["black"])
    if label:
        add_panel_label(ax, label)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(82 * MM_TO_INCH, 65.6 * MM_TO_INCH))
    draw(ax)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.17, top=0.95)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

