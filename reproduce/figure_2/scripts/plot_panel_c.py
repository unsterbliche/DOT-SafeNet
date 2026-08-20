"""Figure 2c: observed versus predicted pAC50 across six target families."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.figure_style import MM_TO_INCH, add_panel_label, draw_joint_regression, save_publication_figure

DATA = PACKAGE_ROOT / "figure_2" / "data" / "panel_c_pac50_predictions.csv"
OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2c"
FAMILIES = [("GPCR", "GPCR"), ("Ion channel", "Ion channel"), ("Enzyme", "Enzyme"), ("Kinases", "Kinase"), ("NR", "NR"), ("Transporter", "Transporter")]


def draw(fig, subplot_spec, data_path=DATA, label=""):
    data = pd.read_csv(data_path)
    required = {"target_family", "observed", "predicted"}
    if not required.issubset(data.columns):
        raise ValueError("Figure 2c source data are incomplete")
    grid = subplot_spec.subgridspec(3, 2, wspace=0.12, hspace=0.18)
    axes = []
    for index, (family, display_name) in enumerate(FAMILIES):
        subset = data.loc[data["target_family"].eq(family)]
        if subset.empty:
            raise ValueError("No pAC50 observations for {}".format(family))
        main_ax, top_ax, _ = draw_joint_regression(
            fig,
            grid[index // 2, index % 2],
            subset,
            display_name,
            "pAC$_{50}$",
            marginal_gap=0.0,
        )
        top_ax.set_title("", loc="left")
        main_ax.set_box_aspect(1)
        main_ax.tick_params(labelsize=7.5, pad=2)
        main_ax.xaxis.label.set_size(8.5)
        main_ax.yaxis.label.set_size(8.5)
        row, column = divmod(index, 2)
        if row < 2:
            main_ax.set_xlabel("")
        if column > 0:
            main_ax.set_ylabel("")
        main_ax.texts[0].set_fontsize(7.5)
        for spine in main_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_color("#272727")
        main_ax.text(
            0.96, 0.06, display_name, transform=main_ax.transAxes,
            ha="right", va="bottom", fontsize=9, fontweight="bold", color="#272727"
        )
        axes.append(main_ax)
    if label:
        add_panel_label(axes[0], label, x=-0.33, y=1.40)
    return axes


def main():
    fig = plt.figure(figsize=(88 * MM_TO_INCH, 132 * MM_TO_INCH))
    spec = fig.add_gridspec(1, 1, left=0.12, right=0.98, bottom=0.07, top=0.98)[0]
    draw(fig, spec)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()


