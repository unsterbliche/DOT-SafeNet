"""Supplementary Figure 4f: transporter pK regression."""

from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT)); sys.path.insert(0, str(SCRIPT_DIR))
import matplotlib.pyplot as plt
from common.figure_style import MM_TO_INCH, save_publication_figure
from regression_panel import draw_family
DATA = PACKAGE_ROOT / "supplementary_figure_4" / "data" / "panel_f_transporter_pk_predictions.csv"
OUTPUT = PACKAGE_ROOT / "supplementary_figure_4" / "outputs" / "panels" / "supplementary_figure_4f"
def draw(fig, spec, label=""): return draw_family(fig, spec, DATA, "Transporter", label)
def main():
    fig = plt.figure(figsize=(58 * MM_TO_INCH, 58 * MM_TO_INCH)); spec = fig.add_gridspec(1, 1, left=0.18, right=0.95, bottom=0.14, top=0.91)[0]
    draw(fig, spec); save_publication_figure(fig, OUTPUT); plt.close(fig)
if __name__ == "__main__": main()

