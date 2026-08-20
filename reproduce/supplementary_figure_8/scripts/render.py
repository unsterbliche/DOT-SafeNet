# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# all panels: UMAP scatter -> cross-type inherit (Manifold asset is a 3D surface and
# is semantically incompatible); preserve the established Figure 4D visual system.

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PANELS = OUT / "panels"
LEGENDS = OUT / "legends"
TASKS = ["BLO", "CAR", "EAR", "END", "EYE", "GAS",
         "HEP", "IMM", "INF", "MET", "MUS", "NER",
         "PSY", "REN", "REP", "RES", "SKI", "VAS"]

def frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#2A2A2A")
    ax.tick_params(length=0)

def draw_soc(ax, source, abbr, xlim, ylim, cfg):
    selected = source.loc[source["adr_abbr"].eq(abbr)]
    predicted = selected.loc[selected["ensemble_probability"].gt(cfg["threshold"])]
    ax.scatter(source["UMAP_1"], source["UMAP_2"], s=1.2,
               c=cfg["colors"]["background"], alpha=0.075,
               edgecolors="none", rasterized=True)
    ax.scatter(selected["UMAP_1"], selected["UMAP_2"], s=5.0,
               c=cfg["colors"]["observed"], alpha=0.60,
               edgecolors="none", rasterized=True)
    ax.scatter(predicted["UMAP_1"], predicted["UMAP_2"], s=6.0,
               c=cfg["colors"]["predicted"], alpha=0.88,
               edgecolors="none", rasterized=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.93, abbr, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=8)
    frame(ax)

def legend_handles(cfg):
    return [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["background"], markeredgecolor="none",
               markersize=4, label="Other observed-positive pairs"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["observed"], markeredgecolor="none",
               markersize=4, label="Observed positive"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["predicted"], markeredgecolor="none",
               markersize=4, label="Observed + predicted positive"),
    ]

def render_model_comparison(cfg):
    table = pd.read_csv(DATA / "model_comparison_source_data.csv")
    stats = pd.read_csv(DATA / "model_comparison_statistics.csv")
    colors = ["#BDBDBD", "#D8D8D8", "#C44E52"]
    fig, ax = plt.subplots(figsize=cfg["model_figure_size"])
    x = np.arange(len(table))
    ax.bar(x, table["auroc_mean"], yerr=table["auroc_sd"], width=0.66,
           color=colors, edgecolor="white", linewidth=0.5, capsize=2.5,
           error_kw={"lw": 0.8, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(table["display_name"], rotation=25, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.64, 0.755)
    for i, value in enumerate(table["auroc_mean"]):
        ax.text(i, value + 0.006, f"{value:.3f}", ha="center",
                va="bottom", fontsize=7.2)
    y0 = 0.742
    for offset, (_, row) in enumerate(stats.iterrows()):
        left = 0 if "MLDNN" in row["comparison"] else 1
        right = 2
        y = y0 - offset * 0.012
        ax.plot([left, left, right, right],
                [y - 0.002, y, y, y - 0.002],
                color="#444444", lw=0.7)
        ax.text((left + right) / 2, y + 0.001, row["significance"],
                ha="center", va="bottom", fontsize=8)
    frame(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.29)
    PANELS.mkdir(parents=True, exist_ok=True)
    stem = PANELS / "supplementary_figure_8a_model_comparison"
    save_cns_figure(fig, str(stem))
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def render_umap(cfg):
    source = pd.read_csv(DATA / "all_soc_umap_source_data.csv.gz")
    dx = source["UMAP_1"].max() - source["UMAP_1"].min()
    dy = source["UMAP_2"].max() - source["UMAP_2"].min()
    xlim = (source["UMAP_1"].min() - 0.075 * dx,
            source["UMAP_1"].max() + 0.075 * dx)
    ylim = (source["UMAP_2"].min() - 0.10 * dy,
            source["UMAP_2"].max() + 0.10 * dy)

    fig, axes = plt.subplots(3, 6, figsize=cfg["figure_size"],
                             sharex=True, sharey=True)
    for ax, abbr in zip(axes.flat, TASKS):
        draw_soc(ax, source, abbr, xlim, ylim, cfg)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.02,
                        wspace=cfg["wspace"], hspace=cfg["hspace"])
    OUT.mkdir(parents=True, exist_ok=True)
    save_cns_figure(fig, str(PANELS / "supplementary_figure_8b_all_soc_umap"))
    fig.savefig(PANELS / "supplementary_figure_8b_all_soc_umap.svg",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    LEGENDS.mkdir(parents=True, exist_ok=True)
    legend_fig = plt.figure(figsize=(6.4, 0.35))
    legend_fig.legend(handles=legend_handles(cfg), loc="center", ncol=3,
                      handletextpad=0.35, columnspacing=1.1)
    save_cns_figure(
        legend_fig,
        str(LEGENDS / "supplementary_figure_8b_legend")
    )
    legend_fig.savefig(
        LEGENDS / "supplementary_figure_8b_legend.svg",
        bbox_inches="tight", dpi=300
    )
    plt.close(legend_fig)


def trim(image):
    rgb = image.convert("RGB")
    diff = ImageChops.difference(
        rgb, Image.new("RGB", rgb.size, "white")
    ).convert("L")
    box = diff.point(lambda value: 255 if value > 8 else 0).getbbox()
    return rgb.crop(box) if box else rgb

def composite(cfg):
    model = trim(Image.open(PANELS / "supplementary_figure_8a_model_comparison.png"))
    umap = trim(Image.open(PANELS / "supplementary_figure_8b_all_soc_umap.png"))
    width = 2160
    margin = 34
    gap = 36
    model_h = 680
    umap_h = 1050
    height = margin * 2 + model_h + umap_h + gap
    canvas = Image.new("RGB", (width, height), "white")

    model_fit = ImageOps.contain(model, (760, model_h), Image.LANCZOS)
    model_x = (width - model_fit.width) // 2
    model_y = margin + (model_h - model_fit.height) // 2
    canvas.paste(model_fit, (model_x, model_y))

    umap_fit = ImageOps.contain(
        umap, (width - margin * 2, umap_h), Image.LANCZOS
    )
    umap_x = (width - umap_fit.width) // 2
    umap_y = margin + model_h + gap + (umap_h - umap_fit.height) // 2
    canvas.paste(umap_fit, (umap_x, umap_y))

    canvas.save(
        OUT / "supplementary_figure_8.png",
        dpi=(300, 300),
    )
    canvas.save(
        OUT / "supplementary_figure_8.pdf",
        "PDF", resolution=300
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    render_model_comparison(cfg)
    render_umap(cfg)
    composite(cfg)

if __name__ == "__main__":
    main()
