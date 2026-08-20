from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cmax_case_curves" / "cmax_case6_scaffold_5seed_predictions.csv"
OUT = ROOT / "outputs" / "cmax_case_curves"
CASES = [
    ("Sildenafil citrate", "Sildenafil"),
    ("Alogliptin benzoate", "Alogliptin"),
    ("Fezolinetant", "Fezolinetant"),
    ("Torsemide", "Torsemide"),
]

def full_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.tick_params(direction="out", length=3, width=0.8)

def draw(ax, frame, title, cfg):
    frame = frame.sort_values("dose")
    x = frame["dose"].to_numpy(float)
    mean = frame["predicted_activity_mean"].to_numpy(float)
    sd = frame["predicted_activity_sd"].to_numpy(float)
    obs = frame["activity"].to_numpy(float)
    ax.fill_between(x, mean - sd, mean + sd, color=cfg["colors"]["interval"], alpha=0.55, linewidth=0)
    ax.plot(x, mean, color=cfg["colors"]["prediction"], lw=1.8, label="DoseExpoNet")
    ax.scatter(x, obs, s=24, color=cfg["colors"]["observed"], edgecolor="white", linewidth=0.55,
               zorder=3, label="Observed")
    ax.set_xscale("log", base=cfg["axes"]["x_log_base"])
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel(r"$\log_{10}$ Cmax ($\mu$g mL$^{-1}$)")
    ax.text(cfg["text"]["drug_name_x"], cfg["text"]["drug_name_y"], title,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=cfg["text"]["title_size"], fontweight="bold")
    ax.grid(False)
    full_frame(ax)

def save(fig, stem, cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in cfg["export"]["formats"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=cfg["figure"]["dpi"], bbox_inches="tight",
                    transparent=cfg["export"]["transparent"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params_cmax_case_curves.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cfg["text"]["font_family"], "Helvetica", "DejaVu Sans"],
        "font.size": cfg["text"]["font_size"],
        "axes.titlesize": cfg["text"]["title_size"],
        "axes.labelsize": cfg["text"]["label_size"],
        "xtick.labelsize": cfg["text"]["tick_size"],
        "ytick.labelsize": cfg["text"]["tick_size"],
        "legend.fontsize": cfg["text"]["legend_size"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    data = pd.read_csv(DATA)
    for panel, (case, title) in zip(("c", "d", "e", "f"), CASES):
        frame = data.loc[data["case_name"].eq(case)]
        if frame.empty:
            raise ValueError(f"Missing case: {case}")
        fig, ax = plt.subplots(figsize=(cfg["figure"]["panel_width_in"], cfg["figure"]["panel_height_in"]))
        draw(ax, frame, title, cfg)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
                   frameon=False, handlelength=2.0)
        fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.29)
        save(fig, f"supplementary_figure_7{panel}_{title.lower().replace(' ', '_')}", cfg)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(cfg["figure"]["composite_width_in"], cfg["figure"]["composite_height_in"]))
    for ax, (case, title) in zip(axes.flat, CASES):
        draw(ax, data.loc[data["case_name"].eq(case)], title, cfg)
    handles = [
        Line2D([0], [0], color=cfg["colors"]["prediction"], lw=1.8, label="DoseExpoNet"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cfg["colors"]["observed"],
               markeredgecolor="white", markersize=5, label="Observed"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, cfg["layout"]["legend_y"]),
               ncol=2, frameon=False, handlelength=2.0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.15,
                        wspace=cfg["layout"]["wspace"], hspace=cfg["layout"]["hspace"])
    save(fig, "supplementary_figure_7c_f", cfg)
    plt.close(fig)

if __name__ == "__main__":
    main()
