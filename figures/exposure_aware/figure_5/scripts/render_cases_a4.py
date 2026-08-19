# Academic Figure Skill Asset Confirmation (verified against project production figures)
# (a-d) Dose-response line plots -> project production script -> param inherit
# (a-d) Target-attribution lollipop plots -> project production script -> param inherit
# RULE: Stored fold predictions and target-occlusion summaries are the quantitative source.

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.65,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SOURCE_SCRIPT = ROOT / "scripts" / "case_plotting_source.py"
MANIFEST = ROOT / "data" / "selected_case_manifest.csv"
PREDICTIONS = ROOT / "data" / "selected_case_dose_predictions_by_fold.csv"
OCCLUSION = ROOT / "data" / "selected_case_target_occlusion_robust_summary.csv"
MEMBERSHIP = ROOT / "data" / "soc_target_membership.csv"

BLUE = "#2166AC"
PURPLE = "#762A83"
RED = "#B2182B"
ORANGE = "#F1A340"
GREY = "#B5BCC3"
BLACK = "#222222"
GRID = "#DCE1E5"
CASE_IDS = ["meclofenamic_acid", "citalopram", "spironolactone", "candesartan"]
CASE_TITLES = {
    "meclofenamic_acid": "Meclofenamic acid",
    "citalopram": "Citalopram",
    "spironolactone": "Spironolactone",
    "candesartan": "Candesartan cilexetil",
}
CASE_SOC = {
    "meclofenamic_acid": "Gastrointestinal disorders",
    "citalopram": "Cardiac disorders",
    "spironolactone": "Reproductive system and breast disorders",
    "candesartan": "Vascular and renal disorders",
}


def load_plotter():
    spec = importlib.util.spec_from_file_location("case_plotter_a4", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boxed_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#50565C")
        spine.set_linewidth(0.65)
    ax.tick_params(length=2.5, width=0.65, pad=2)


def draw_case(fig, slot, case_id, manifest, predictions, occlusion, membership, plotter):
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=[0.86, 1.14], hspace=0.39)
    ax_curve = fig.add_subplot(inner[0])
    ax_target = fig.add_subplot(inner[1])

    case_manifest = manifest[manifest["figure_id"].eq(case_id)].copy()
    case_pred = predictions[predictions["figure_id"].eq(case_id)].copy()
    task_colors = {"Vascular": BLUE, "Renal": PURPLE}
    for task_index, row in enumerate(case_manifest.itertuples(index=False)):
        data = case_pred[case_pred["task_name"].eq(row.task_name)].copy()
        color = task_colors.get(row.adr_short, BLUE)
        for _, fold_data in data.groupby("fold"):
            fold_data = fold_data.sort_values("dose_mg_day")
            ax_curve.plot(fold_data["dose_mg_day"], fold_data["adr_score"], color=color, alpha=0.15, lw=0.55, marker="o", ms=1.8)
        summary = data.groupby("dose_mg_day", as_index=False).agg(mean=("adr_score", "mean"), low=("adr_score", "min"), high=("adr_score", "max")).sort_values("dose_mg_day")
        ax_curve.fill_between(summary["dose_mg_day"].to_numpy(), summary["low"].to_numpy(), summary["high"].to_numpy(), color=color, alpha=0.09, lw=0)
        ax_curve.plot(summary["dose_mg_day"], summary["mean"], color=color, lw=1.55, marker="o", ms=3.5, mec="white", mew=0.45)
        for dose in plotter.parse_number_list(row.clinical_doses_mg_day):
            hit = summary[np.isclose(summary["dose_mg_day"], dose)]
            if not hit.empty:
                score = float(hit.iloc[0]["mean"])
                ax_curve.axvline(dose, color=RED, ls=(0, (2, 2)), lw=0.65, alpha=0.65)
                ax_curve.scatter(dose, score, marker="D", s=25, color=RED, edgecolor="white", lw=0.5, zorder=5)
        if len(case_manifest) > 1:
            last = summary.iloc[-1]
            ax_curve.text(float(last["dose_mg_day"]) * 0.94, float(last["mean"]) - 0.035 - 0.055 * task_index, row.adr_short.upper()[:3], color=color, fontsize=5.7, fontweight="bold", ha="right", va="top")

    doses = sorted(set(case_pred["dose_mg_day"].astype(float)))
    ax_curve.set_xscale("log", base=2)
    ax_curve.set_xticks(doses)
    ax_curve.set_xticklabels([plotter.format_dose(x) for x in doses])
    ax_curve.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_yticks([0, 0.5, 1.0])
    ax_curve.set_xlabel("Daily dose (mg)", labelpad=2)
    ax_curve.set_ylabel("ADR score", labelpad=2)
    ax_curve.set_title("Dose response", loc="left", pad=3, fontweight="bold")
    ax_curve.grid(axis="y", color=GRID, lw=0.4)
    boxed_axis(ax_curve)
    ax_curve.text(0.97, 0.13, CASE_TITLES[case_id], transform=ax_curve.transAxes,
                  ha="right", va="bottom", fontsize=7.2, fontweight="bold", color=BLACK,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8})
    ax_curve.text(0.97, 0.045, CASE_SOC[case_id], transform=ax_curve.transAxes,
                  ha="right", va="bottom", fontsize=5.8, color="#55606C",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6})

    case_occ = occlusion[occlusion["figure_id"].eq(case_id)].copy()
    targets = plotter.annotate_soc_association(plotter.select_targets(case_manifest, case_occ), membership)
    targets = targets.sort_values(["task_label", "robust_delta_mean"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(targets))
    for index, target in targets.iterrows():
        if target["soc_association_class"] == "direct":
            color, marker, size = BLUE, "o", 23
        elif target["soc_association_class"] == "secondary":
            color, marker, size = ORANGE, "o", 23
        else:
            color, marker, size = GREY, "o", 20
        mean = float(target["robust_delta_mean"])
        low = float(target["robust_delta_min"])
        high = float(target["robust_delta_max"])
        ax_target.hlines(index, 0, mean, color=color, lw=1.15, alpha=0.92)
        ax_target.hlines(index, low, high, color=color, lw=3.0, alpha=0.20)
        ax_target.scatter(mean, index, color=color, marker=marker, s=size, edgecolor="white", lw=0.4, zorder=4)
    labels = []
    for target in targets.itertuples(index=False):
        label = str(target.target_gene_label)
        if len(case_manifest) > 1:
            label += " [VAS]" if str(target.task_label).lower().startswith("vascular") else " [REN]"
        labels.append(label)
    ax_target.set_yticks(y)
    ax_target.set_yticklabels(labels, fontsize=6.0)
    low_x = min(0, float(targets["robust_delta_min"].min()) * 1.08)
    high_x = max(float(targets["robust_delta_max"].max()), float(targets["robust_delta_mean"].max())) * 1.10
    ax_target.set_xlim(low_x, high_x)
    ax_target.axvline(0, color="#8F979F", lw=0.55)
    ax_target.set_xlabel("ΔADR score after target replacement", labelpad=2)
    attr_doses = sorted(set(case_manifest["attribution_dose_mg_day"].astype(float)))
    dose_text = ", ".join(plotter.format_dose(x) for x in attr_doses)
    ax_target.set_title(f"Target attribution ({dose_text} mg/day)", loc="left", pad=3, fontweight="bold")
    ax_target.grid(axis="x", color=GRID, lw=0.4)
    ax_target.tick_params(axis="y", length=0, pad=2)
    boxed_axis(ax_target)
    return targets


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plotter = load_plotter()
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    predictions = pd.read_csv(PREDICTIONS)
    occlusion = pd.read_csv(OCCLUSION)
    membership = pd.read_csv(MEMBERSHIP)
    target_tables = []

    for case_id in CASE_IDS:
        fig = plt.figure(figsize=(89 / 25.4, 108 / 25.4))
        outer = GridSpec(1, 1, figure=fig, left=0.23, right=0.98, bottom=0.10, top=0.97)
        targets = draw_case(fig, outer[0], case_id, manifest, predictions, occlusion, membership, plotter)
        target_tables.append(targets)
        stem = OUT / f"figure5_{case_id}"
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.5, marker="o", ms=3.5, mec="white", label="5-fold ensemble mean"),
        Line2D([0], [0], color="none", marker="D", mfc=RED, mec="white", ms=4.5, label="Clinical dose"),
        Line2D([0], [0], color="none", marker="o", mfc=BLUE, mec="white", ms=4.5, label="Direct ADR evidence"),
        Line2D([0], [0], color="none", marker="o", mfc=ORANGE, mec="white", ms=4.5, label="Secondary evidence"),
        Line2D([0], [0], color="none", marker="o", mfc=GREY, mec="white", ms=4.5, label="Other"),
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(183 / 25.4, 10 / 25.4))
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=5, frameon=False,
                     fontsize=6.2, handletextpad=0.35, columnspacing=0.85)
    for ext in ["png", "pdf", "svg"]:
        legend_fig.savefig(OUT / f"figure5_case_legend.{ext}", dpi=300, bbox_inches="tight")
    plt.close(legend_fig)
    pd.concat(target_tables, ignore_index=True).to_csv(OUT / "figure5_case_target_source.csv", index=False)


if __name__ == "__main__":
    main()
