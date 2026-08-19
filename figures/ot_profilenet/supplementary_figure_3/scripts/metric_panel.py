"""Shared grouped-bar renderer for Supplementary Figure 3b–d."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import pandas as pd

from common.figure_style import COLORS, METHOD_COLORS, add_panel_label, clean_axis

TARGETS = ["GPCR", "Ion channel", "Enzyme", "Kinases", "NR", "Transporter", "Other"]
TARGET_LABELS = ["GPCR", "Ion\nchannel", "Enzyme", "Kinase", "NR", "Transporter", "Other"]
MODELS = ["MTGNN", "OT-ProfileNet (from scratch)", "OT-ProfileNet (fine-tuned)"]


def prepare(data_path: Path, metric: str) -> pd.DataFrame:
    data = pd.read_csv(data_path).copy()
    data["model_display"] = data["model"].replace({
        "PreMOTA (Train only)": "OT-ProfileNet (from scratch)",
        "PreMOTA (Pretrain + FT)": "OT-ProfileNet (fine-tuned)",
    })
    data = data.loc[data["model_display"].isin(MODELS)]
    if data.empty or metric not in data.columns:
        raise ValueError("Missing {} classification metrics".format(metric))
    return data


def draw_metric(ax, data_path: Path, metric: str, ylabel: str, label: str,
                show_legend: bool = True):
    import numpy as np

    data = prepare(data_path, metric)
    x = np.arange(len(TARGETS))
    width = 0.24
    for index, model in enumerate(MODELS):
        current = data.loc[data["model_display"].eq(model)].set_index("target")
        values = current.loc[TARGETS, metric].to_numpy(float)
        ax.bar(x + (index - 1) * width, values, width, color=METHOD_COLORS[model],
               edgecolor="white", linewidth=0.3, label=model, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    clean_axis(ax, grid_axis="y")
    ax.tick_params(labelsize=8, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(COLORS["black"])
    if show_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=8)
    if label:
        add_panel_label(ax, label)
    return ax

