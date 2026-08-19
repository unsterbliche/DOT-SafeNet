"""Supplementary Figure 3c: classification F1 score."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt

from common.figure_style import MM_TO_INCH, save_publication_figure
from metric_panel import draw_metric

DATA = PACKAGE_ROOT / "supplementary_figure_3" / "data" / "panel_c_f1_metrics.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_3" / "outputs" / "panels" / "supplementary_figure_3c"


def draw(ax, label="", show_legend=True):
    return draw_metric(ax, DATA, "F1", "F1 score", label, show_legend)


def main():
    fig, ax = plt.subplots(figsize=(82 * MM_TO_INCH, 62 * MM_TO_INCH))
    draw(ax, show_legend=False)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.96)
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()

