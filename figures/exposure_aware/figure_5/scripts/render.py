# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Panel A: MarkerGeneDotPlot asset -> parameter-inheritance adaptation for paired evidence counts.
# Existing project ADR–target membership and OT-ProfileNet target-family assignments are preserved.

from __future__ import annotations

import argparse
from pathlib import Path

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

# Academic Figure Skill Nature/Cell/Science Color Palette — COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SOC_ORDER = [
    "BLO", "CAR", "EAR", "END", "EYE", "GAS", "HEP", "IMM", "INF",
    "MET", "MUS", "NER", "PSY", "REN", "REP", "RES", "SKI", "VAS",
]
FAMILY_ORDER = ["GPCR", "IonChannel", "Enzyme", "Kinase", "NHR", "Transporter", "Other"]
FAMILY_LABELS = {
    "GPCR": "GPCR", "IonChannel": "Ion\nchannel", "Enzyme": "Enzyme",
    "Kinase": "Kinase", "NHR": "Nuclear\nreceptor", "Transporter": "Transporter",
    "Other": "Other",
}


def classify_evidence(value: str) -> str:
    return "secondary" if "secondary" in str(value).lower() else "direct"


def prepare_counts() -> pd.DataFrame:
    membership = pd.read_csv(DATA / "soc_target_membership.csv")
    family_map = pd.read_csv(DATA / "target_family_map.csv")
    membership["evidence_type"] = membership["association_class"].map(classify_evidence)
    membership = membership.merge(family_map, on="target_uniprot", how="left", validate="many_to_one")
    if membership["target_family"].isna().any():
        missing = membership.loc[membership["target_family"].isna(), "target_uniprot"].unique()
        raise ValueError(f"Missing target-family assignments: {missing.tolist()}")

    # Direct evidence supersedes secondary evidence for duplicate SOC–target pairs.
    membership["evidence_priority"] = membership["evidence_type"].map({"direct": 0, "secondary": 1})
    unique_pairs = (
        membership.sort_values("evidence_priority")
        .drop_duplicates(["soc_abbr", "target_uniprot"], keep="first")
    )
    counts = (
        unique_pairs.groupby(["soc_abbr", "target_family", "evidence_type"], as_index=False)
        ["target_uniprot"].nunique()
        .rename(columns={"target_uniprot": "n_targets"})
    )
    full = pd.MultiIndex.from_product(
        [SOC_ORDER, FAMILY_ORDER, ["direct", "secondary"]],
        names=["soc_abbr", "target_family", "evidence_type"],
    ).to_frame(index=False)
    counts = full.merge(counts, how="left").fillna({"n_targets": 0})
    counts["n_targets"] = counts["n_targets"].astype(int)
    counts.to_csv(DATA / "figure5a_soc_target_family_counts.csv", index=False)
    return counts


def bubble_size(n: int) -> float:
    if n <= 0:
        return 0
    return 34 + 24 * np.sqrt(n)


def panel_a(cfg: dict) -> None:
    counts = prepare_counts()
    direct_color = cfg["colors"]["direct"]
    secondary_color = cfg["colors"]["secondary"]
    grid_color = cfg["colors"]["grid"]
    text_color = cfg["colors"]["text"]

    fig, ax = plt.subplots(figsize=cfg["figure"]["panel_a_size"])
    x_positions = np.arange(len(FAMILY_ORDER))
    y_positions = np.arange(len(SOC_ORDER))

    # Sparse row banding improves tracing across the wide matrix.
    for y in y_positions:
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F7F8FA", zorder=0)
    for x in np.arange(-0.5, len(FAMILY_ORDER) + 0.5, 1):
        ax.axvline(x, color=grid_color, lw=0.45, zorder=0)

    offsets = {"direct": -0.17, "secondary": 0.17}
    colors = {"direct": direct_color, "secondary": secondary_color}
    for _, row in counts.iterrows():
        n = int(row["n_targets"])
        if n == 0:
            continue
        x = FAMILY_ORDER.index(row["target_family"]) + offsets[row["evidence_type"]]
        y = SOC_ORDER.index(row["soc_abbr"])
        ax.scatter(
            x, y, s=bubble_size(n), color=colors[row["evidence_type"]],
            edgecolor="white", linewidth=0.65, alpha=0.94, zorder=3,
        )
        ax.text(
            x, y, str(n), ha="center", va="center", fontsize=5.5,
            color="white" if n >= 3 else text_color, fontweight="bold", zorder=4,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([FAMILY_LABELS[x] for x in FAMILY_ORDER], ha="center")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True,
                   labelbottom=False, length=0, pad=7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(SOC_ORDER, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(-0.55, len(FAMILY_ORDER) - 0.45)
    ax.set_ylim(len(SOC_ORDER) - 0.45, -0.55)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#AEB5BC")
        spine.set_linewidth(0.65)

    # Evidence legend is outside the data region.
    evidence_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=direct_color,
               markeredgecolor="white", markersize=7, label="Direct evidence"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=secondary_color,
               markeredgecolor="white", markersize=7, label="Secondary evidence"),
    ]
    legend1 = ax.legend(handles=evidence_handles, loc="upper left", bbox_to_anchor=(0, -0.055),
                        ncol=2, columnspacing=1.4, handletextpad=0.45)
    ax.add_artist(legend1)

    size_counts = [1, 5, 15, 30]
    size_handles = [
        plt.scatter([], [], s=bubble_size(n), facecolor="#B8BEC5", edgecolor="white", linewidth=0.6)
        for n in size_counts
    ]
    ax.legend(size_handles, [str(n) for n in size_counts], title="Number of targets",
              loc="upper right", bbox_to_anchor=(1, -0.045), ncol=4,
              columnspacing=0.8, handletextpad=0.2, title_fontsize=7.5)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.13)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in cfg["export"]["formats"]:
        fig.savefig(
            OUT / f"figure5a_soc_target_family_evidence.{ext}",
            dpi=cfg["figure"]["dpi"], bbox_inches="tight",
            transparent=cfg["export"]["transparent"],
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    panel_a(cfg)


if __name__ == "__main__":
    main()
