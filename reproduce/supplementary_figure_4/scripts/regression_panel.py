# Academic Figure Skill Asset Confirmation (verified against local Figure 2c)
# (a-f) marginal regression -> figure_2/scripts/plot_panel_c.py -> param inherit
# RULE: The observed/predicted data are unchanged; only the Figure 2c visual system is reused.
"""Shared Figure 2c-style renderer for Supplementary Figure 4."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.figure_style import add_panel_label, draw_joint_regression


def draw_family(fig, subplot_spec, data_path, family, label=""):
    import pandas as pd

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("No pK observations for {}".format(family))
    main_ax, top_ax, _ = draw_joint_regression(
        fig,
        subplot_spec,
        data,
        family,
        "pK",
        marginal_gap=0.0,
    )
    top_ax.set_title("", loc="left")
    main_ax.set_box_aspect(1)
    main_ax.tick_params(labelsize=7.5, pad=2)
    main_ax.xaxis.label.set_size(8.5)
    main_ax.yaxis.label.set_size(8.5)
    main_ax.texts[0].set_fontsize(7.5)
    for spine in main_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#272727")
    main_ax.text(
        0.96,
        0.06,
        family,
        transform=main_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#272727",
    )
    if label:
        add_panel_label(main_ax, label, x=-0.28, y=1.42)
    return main_ax