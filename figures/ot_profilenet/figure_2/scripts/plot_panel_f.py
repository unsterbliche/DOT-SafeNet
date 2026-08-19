"""Figure 2f: compound-level external hERG prediction table."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.figure_style import COLORS, MM_TO_INCH, add_panel_label, save_publication_figure

DATA = PACKAGE_ROOT / "figure_2" / "data" / "panel_e_f_herg_external_validation.csv"
OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2f"


def draw(ax, data_path=DATA, label=""):
    data = pd.read_csv(data_path)
    ax.axis("off")
    rows = [[str(row.Drug).capitalize(), "{:.3f}".format(row.pIC50),
             "{:.3f}".format(row.predict_result_pAC50)] for row in data.itertuples()]
    table = ax.table(
        cellText=rows,
        colLabels=["Drug", "Measured\npIC$_{50}$", "Predicted\npAC$_{50}$"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.46, 0.27, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.32)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["neutral_light"])
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#ECECF4")
            cell.set_text_props(weight="bold")
    if label:
        add_panel_label(ax, label, x=-0.05, y=0.98)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(89 * MM_TO_INCH, 72 * MM_TO_INCH))
    draw(ax)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.94)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

