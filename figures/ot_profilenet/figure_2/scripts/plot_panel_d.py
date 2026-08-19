"""Figure 2d: benchmark comparison across safety-target families."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.figure_style import MM_TO_INCH, add_panel_label, draw_grouped_metric_bars, save_publication_figure

DATA = PACKAGE_ROOT / "figure_2" / "data" / "panel_d_model_comparison.csv"
OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2d"

FAMILIES = ["GPCR", "Ion channel", "Enzyme", "Kinase", "NR", "Transporter", "Other"]
MODELS = ["GraphDTA", "BACPI", "PMF-CPI", "OT-ProfileNet (from scratch)", "OT-ProfileNet (fine-tuned)"]


def load_data(data_path=DATA):
    data = pd.read_csv(data_path).copy()
    data["model_display"] = data["model"].replace({
        "PreMOTA (from-scratch)": "OT-ProfileNet (from scratch)",
        "PreMOTA (fine-tuned)": "OT-ProfileNet (fine-tuned)",
    })
    data["family"] = data["target_family"].replace({"Kinases": "Kinase"})
    return data


def draw(fig, subplot_spec, data_path=DATA, label=""):
    data = load_data(data_path)
    grid = subplot_spec.subgridspec(2, 1, hspace=0.12)
    ax_rmse = fig.add_subplot(grid[0])
    ax_pcc = fig.add_subplot(grid[1], sharex=ax_rmse)
    draw_grouped_metric_bars(
        ax_rmse, data, "family", "model_display", "RMSE",
        FAMILIES, MODELS, "RMSE", (0.5, 0.95)
    )
    draw_grouped_metric_bars(
        ax_pcc, data, "family", "model_display", "PCC",
        FAMILIES, MODELS, "Pearson r", (0.6, 1.02)
    )
    for ax in (ax_rmse, ax_pcc):
        ax.tick_params(labelsize=7.5, pad=2)
        ax.yaxis.label.set_size(9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_color("#272727")
    ax_rmse.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if label:
        add_panel_label(ax_rmse, label, x=-0.18, y=1.08)
    return [ax_rmse, ax_pcc]


def main():
    fig = plt.figure(figsize=(68 * MM_TO_INCH, 105.6 * MM_TO_INCH))
    spec = fig.add_gridspec(1, 1, left=0.18, right=0.98, bottom=0.11, top=0.98)[0]
    draw(fig, spec)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()


