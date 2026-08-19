"""Figure 2a: numbers of safety targets in seven protein families."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.figure_style import MM_TO_INCH, add_panel_label, save_publication_figure

DATA = PACKAGE_ROOT / "figure_2" / "data" / "panel_a_target_counts.csv"
OUTPUT = PACKAGE_ROOT / "figure_2" / "outputs" / "panels" / "figure_2a"


def draw(ax, data_path=DATA, label=""):
    data = pd.read_csv(data_path)
    expected = {"target_family", "target_count"}
    if not expected.issubset(data.columns) or data.empty:
        raise ValueError("Figure 2a source data are incomplete")
    data = data.sort_values("target_count", ascending=False)
    palette = {
        "GPCR": "#F2CF4A",
        "Ion channel": "#EF899D",
        "Enzyme": "#7CC7D0",
        "Kinases": "#32B9B5",
        "NR": "#F4B860",
        "Transporter": "#E9A267",
        "Other": "#D7B7DD",
    }
    colors = [palette[name] for name in data["target_family"]]
    labels = [
        f"{name} {int(count)}"
        for name, count in zip(data["target_family"], data["target_count"])
    ]
    wedges, texts = ax.pie(
        data["target_count"],
        labels=labels,
        colors=colors,
        startangle=86,
        counterclock=False,
        labeldistance=1.12,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    for family, wedge, text in zip(data["target_family"], wedges, texts):
        text.set_color(wedge.get_facecolor())
        text.set_fontweight("bold")
        text.set_bbox(
            {
                "facecolor": "white",
                "edgecolor": wedge.get_facecolor(),
                "linewidth": 0.7,
                "pad": 1.4,
            }
        )
        if family == "Transporter":
            x, y = text.get_position()
            text.set_position((x, y + 0.12))
    ax.set_aspect("equal")
    if label:
        add_panel_label(ax, label)
    return ax


def main():
    fig, ax = plt.subplots(figsize=(78 * MM_TO_INCH, 78 * MM_TO_INCH))
    draw(ax)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

