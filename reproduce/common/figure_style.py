"""Publication style and reusable plot primitives.

The visual system follows the open-source ``nature-figure`` guidance from
Yuan1z0825/nature-skills (Apache-2.0) and retains the numerical plotting logic
of the original OT-ProfileNet notebooks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


# Nature-style typography and editable vector text.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)


# Low-saturation method-family palette adapted from nature-skills/nature-figure.
COLORS: Dict[str, str] = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "model_soft": "#E4E4F0",
    "model_mid": "#E4CCD8",
    "model_main": "#C77C8D",
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#A9C9E2",
    "orange": "#D98B5F",
    "teal": "#42949E",
    "red": "#B64342",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "black": "#272727",
    "grid": "#E7E7E7",
}

METHOD_COLORS = {
    "GraphDTA": "#D8D8D8",
    "BACPI": "#BFC3CC",
    "PMF-CPI": "#9DA6BB",
    "MTGNN": "#A8A8A8",
    "OT-ProfileNet (from scratch)": "#E4CCD8",
    "OT-ProfileNet (fine-tuned)": "#484878",
}

MM_TO_INCH = 1.0 / 25.4


def save_publication_figure(
    fig: plt.Figure,
    stem: Path,
    dpi: int = 300,
    write_tiff: bool = False,
) -> List[Path]:
    """Save editable SVG/PDF plus a PNG preview and optional 600-dpi TIFF."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    outputs = [svg_path, pdf_path, png_path]
    if write_tiff:
        path = stem.with_suffix(".tiff")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    return outputs


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.13,
    y: float = 1.04,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color=COLORS["black"],
        ha="left",
        va="bottom",
    )


def clean_axis(ax: plt.Axes, grid_axis: Optional[str] = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], lw=0.45, zorder=0)
        ax.set_axisbelow(True)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted arrays must be aligned one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression arrays contain non-finite values")
    return {
        "n": int(observed.size),
        "RMSE": float(mean_squared_error(observed, predicted) ** 0.5),
        "PCC": float(pearsonr(observed, predicted)[0]),
        "Spearman": float(spearmanr(observed, predicted)[0]),
        "R2": float(r2_score(observed, predicted)),
    }


def draw_joint_regression(
    fig: plt.Figure,
    subplot_spec,
    data: pd.DataFrame,
    title: str,
    endpoint: str,
    marginal_gap: float = 0.03,
) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Draw a scatter plot with observed/predicted marginal histograms."""
    required = {"observed", "predicted"}
    if not required.issubset(data.columns):
        raise ValueError("Regression data must contain observed and predicted columns")
    observed = data["observed"].to_numpy(float)
    predicted = data["predicted"].to_numpy(float)
    current = regression_metrics(observed, predicted)

    grid = subplot_spec.subgridspec(
        2,
        2,
        width_ratios=[4.3, 1.0],
        height_ratios=[1.0, 4.3],
        wspace=marginal_gap,
        hspace=marginal_gap,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])

    lo = math.floor(min(float(observed.min()), float(predicted.min())))
    hi = math.ceil(max(float(observed.max()), float(predicted.max())))
    pad = max((hi - lo) * 0.03, 0.15)
    limits = (lo - pad, hi + pad)
    bins = np.linspace(limits[0], limits[1], 24)

    ax_main.scatter(
        observed,
        predicted,
        s=2.2,
        color=COLORS["blue_secondary"],
        alpha=0.20,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    ax_main.plot(limits, limits, color=COLORS["neutral_mid"], lw=0.7, ls=(0, (3, 2)))
    slope, intercept = np.polyfit(observed, predicted, 1)
    xx = np.linspace(limits[0], limits[1], 100)
    ax_main.plot(xx, slope * xx + intercept, color=COLORS["red"], lw=1.0, zorder=3)
    ax_main.text(
        0.04,
        0.96,
        "r = {:.3f}\nRMSE = {:.3f}\nn = {:,}".format(
            current["PCC"], current["RMSE"], current["n"]
        ),
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
    )
    ax_main.set_xlim(limits)
    ax_main.set_ylim(limits)
    ax_main.set_xlabel("Observed {}".format(endpoint), labelpad=1)
    ax_main.set_ylabel("Predicted {}".format(endpoint), labelpad=1)
    clean_axis(ax_main)

    ax_top.hist(observed, bins=bins, color=COLORS["blue_light"], alpha=0.65, edgecolor="none")
    ax_top.hist(predicted, bins=bins, histtype="step", color=COLORS["blue_main"], lw=0.75)
    ax_right.hist(
        observed,
        bins=bins,
        orientation="horizontal",
        color=COLORS["blue_light"],
        alpha=0.65,
        edgecolor="none",
    )
    ax_right.hist(
        predicted,
        bins=bins,
        orientation="horizontal",
        histtype="step",
        color=COLORS["blue_main"],
        lw=0.75,
    )
    ax_top.set_xlim(limits)
    ax_right.set_ylim(limits)
    ax_top.set_title(title, loc="left", pad=2, fontweight="bold", fontsize=7)
    ax_top.axis("off")
    ax_right.axis("off")
    return ax_main, ax_top, ax_right


def draw_grouped_metric_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    category_col: str,
    model_col: str,
    value_col: str,
    category_order: Sequence[str],
    model_order: Sequence[str],
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.arange(len(category_order), dtype=float)
    width = 0.78 / len(model_order)
    for index, model in enumerate(model_order):
        current = data.loc[data[model_col].eq(model)].set_index(category_col)
        missing = [name for name in category_order if name not in current.index]
        if missing:
            raise ValueError("Missing categories for {}: {}".format(model, missing))
        values = current.loc[list(category_order), value_col].to_numpy(float)
        offset = (index - (len(model_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width,
            color=METHOD_COLORS.get(model, COLORS["neutral_mid"]),
            edgecolor="white",
            linewidth=0.3,
            label=model,
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(category_order, rotation=38, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0 if ylim is None else ylim[0], top=None if ylim is None else ylim[1])
    clean_axis(ax, grid_axis="y")


