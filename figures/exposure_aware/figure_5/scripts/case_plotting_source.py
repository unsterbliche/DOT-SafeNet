#!/usr/bin/env python3
# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) Dose-response line plot -> project production script + assets/figures/LineTrend -> param inherit
# (b) Target attribution lollipop plot -> project production script -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
"""Draw one exact-dose ADR and target-attribution figure for each selected drug."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib as mpl

# Academic Figure Skill Typography Baseline – COPY VERBATIM, place at TOP of script
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

# Academic Figure Skill Export Baseline – COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


DISPLAY_TITLES = {
    "meclofenamic_acid": "Meclofenamic acid",
    "citalopram": "Citalopram",
    "spironolactone": "Spironolactone",
    "sunitinib": "Sunitinib",
    "candesartan": "Candesartan cilexetil",
    "clozapine": "Clozapine",
    "cysteamine": "Cysteamine",
    "enzalutamide": "Enzalutamide",
}

FILE_ORDER = list(DISPLAY_TITLES)
TASK_COLORS = {"Vascular": CATEGORICAL[0], "Renal": CATEGORICAL[4]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions-by-fold", required=True, type=Path)
    parser.add_argument("--occlusion-summary", required=True, type=Path)
    parser.add_argument("--visualization-membership", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def parse_number_list(value: object) -> list[float]:
    return [float(item) for item in str(value).split(";") if str(item).strip()]


def format_dose(value: float) -> str:
    return f"{value:g}"


def select_targets(case_manifest: pd.DataFrame, case_occlusion: pd.DataFrame) -> pd.DataFrame:
    selected = []
    n_tasks = len(case_manifest)
    n_per_task = 5 if n_tasks > 1 else 10
    for row in case_manifest.itertuples(index=False):
        data = case_occlusion[
            case_occlusion["task_name"].eq(row.task_name)
            & np.isclose(case_occlusion["dose_mg_day"], float(row.attribution_dose_mg_day))
        ].copy()
        if data.empty:
            raise ValueError(f"No target attribution for {row.figure_id} / {row.task_name}")

        data["target_gene_label"] = data["target_gene"].replace("", np.nan).fillna(data["target_id"])
        data = data.sort_values(
            ["robust_positive", "robust_delta_mean", "min_positive_fold_fraction"],
            ascending=[False, False, False],
        )
        top = data[data["robust_delta_mean"] > 0].head(n_per_task).copy()
        if len(top) < n_per_task:
            fill = data[~data["target_id"].isin(top["target_id"])].head(n_per_task - len(top))
            top = pd.concat([top, fill], ignore_index=True)

        key_target_id = str(row.key_target_id).strip()
        if key_target_id and key_target_id not in set(top["target_id"]):
            key = data[data["target_id"].eq(key_target_id)].head(1)
            if not key.empty:
                top = pd.concat([top.head(max(n_per_task - 1, 0)), key], ignore_index=True)

        top = top.drop_duplicates("target_gene_label").head(n_per_task).copy()
        if len(top) < n_per_task:
            used_labels = set(top["target_gene_label"])
            extra = data[~data["target_gene_label"].isin(used_labels)].drop_duplicates(
                "target_gene_label"
            ).head(n_per_task - len(top))
            top = pd.concat([top, extra], ignore_index=True)
        top["is_key_target"] = top["target_id"].eq(key_target_id) if key_target_id else False
        top["attribution_dose_mg_day"] = float(row.attribution_dose_mg_day)
        top["task_label"] = row.adr_short
        top["task_name_full"] = row.task_name
        selected.append(top)
    return pd.concat(selected, ignore_index=True)


def annotate_soc_association(targets: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    annotated = targets.copy()
    classes = []
    for target in annotated.itertuples(index=False):
        subset = membership[
            membership["target_uniprot"].astype(str).eq(str(target.target_id))
            & membership["soc_name"].eq(target.task_name_full)
        ]
        association_classes = set(subset["association_class"].astype(str))
        if "direct source-supported association" in association_classes:
            classes.append("direct")
        elif "network visualization secondary" in association_classes:
            classes.append("secondary")
        else:
            classes.append("other")
    annotated["soc_association_class"] = classes
    return annotated


def plot_case(
    figure_id: str,
    case_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    occlusion: pd.DataFrame,
    membership: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, list[dict], pd.DataFrame]:
    title_drug = DISPLAY_TITLES[figure_id]
    task_labels = list(case_manifest["adr_short"])
    title_task = " and ".join(task_labels) + (" disorders" if len(task_labels) > 1 else "")
    if len(task_labels) == 1:
        title_task = case_manifest.iloc[0]["task_name"]

    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.12, 1.0], left=0.08, right=0.985, bottom=0.18, top=0.76, wspace=0.36
    )
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[0, 1])

    fig.text(0.03, 0.95, title_drug, fontsize=11, fontweight="bold", color=BLACK, ha="left", va="top")
    fig.text(0.03, 0.855, title_task, fontsize=8, color="#4A5561", ha="left", va="top")

    case_pred = predictions[predictions["figure_id"].eq(figure_id)].copy()
    colors = [TASK_COLORS.get(label, CATEGORICAL[index]) for index, label in enumerate(task_labels)]
    clinical_handle_added = False
    stats_rows: list[dict] = []

    for color, row in zip(colors, case_manifest.itertuples(index=False)):
        task_data = case_pred[case_pred["task_name"].eq(row.task_name)].copy()
        for _, fold_data in task_data.groupby("fold"):
            fold_data = fold_data.sort_values("dose_mg_day")
            ax_curve.plot(
                fold_data["dose_mg_day"], fold_data["adr_score"],
                color=color, alpha=0.17, linewidth=0.7, marker="o", markersize=2,
            )
        summary = (
            task_data.groupby("dose_mg_day", as_index=False)
            .agg(mean=("adr_score", "mean"), low=("adr_score", "min"), high=("adr_score", "max"))
            .sort_values("dose_mg_day")
        )
        label = row.adr_short if len(case_manifest) > 1 else "5-fold ensemble mean"
        ax_curve.fill_between(
            summary["dose_mg_day"].to_numpy(), summary["low"].to_numpy(), summary["high"].to_numpy(),
            color=color, alpha=0.10, linewidth=0,
        )
        ax_curve.plot(
            summary["dose_mg_day"], summary["mean"], color=color, linewidth=1.8,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.5, label=label,
        )

        for dose in parse_number_list(row.clinical_doses_mg_day):
            hit = summary[np.isclose(summary["dose_mg_day"], dose)]
            if hit.empty:
                continue
            y = float(hit.iloc[0]["mean"])
            ax_curve.axvline(dose, color=ACCENT_RED, linestyle=(0, (2, 2)), linewidth=0.7, alpha=0.55, zorder=0)
            ax_curve.scatter(
                [dose], [y], marker="D", s=31, color=ACCENT_RED, edgecolor="white", linewidth=0.5,
                zorder=5, label="Clinical dose" if not clinical_handle_added else None,
            )
            clinical_handle_added = True

        ranks = summary["dose_mg_day"].rank().to_numpy()
        rho = float(pd.Series(ranks).corr(pd.Series(summary["mean"].to_numpy()), method="spearman"))
        stats_rows.append(
            {
                "figure_id": figure_id,
                "display_drug": title_drug,
                "task_name": row.task_name,
                "dose_min_mg_day": float(summary.iloc[0]["dose_mg_day"]),
                "score_at_min_dose": float(summary.iloc[0]["mean"]),
                "dose_max_mg_day": float(summary.iloc[-1]["dose_mg_day"]),
                "score_at_max_dose": float(summary.iloc[-1]["mean"]),
                "score_change_max_minus_min": float(summary.iloc[-1]["mean"] - summary.iloc[0]["mean"]),
                "spearman_dose_score": rho,
            }
        )

    all_doses = sorted(set(case_pred["dose_mg_day"]))
    ax_curve.set_xscale("log", base=2)
    ax_curve.set_xticks(all_doses)
    ax_curve.set_xticklabels([format_dose(value) for value in all_doses], rotation=0)
    ax_curve.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_curve.set_xlabel("Daily dose (mg)")
    ax_curve.set_ylabel("ADR model score")
    ax_curve.set_title("Dose–response", loc="left", fontweight="bold", pad=6)
    ax_curve.grid(axis="y", color="#D8DDE3", linewidth=0.35, alpha=0.65)
    ax_curve.legend(loc="lower right", handlelength=1.5, borderaxespad=0.25, fontsize=6.5)

    case_occ = occlusion[occlusion["figure_id"].eq(figure_id)].copy()
    targets = annotate_soc_association(select_targets(case_manifest, case_occ), membership)
    targets = targets.sort_values(["task_label", "robust_delta_mean"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(targets))
    max_delta = max(float(targets["robust_delta_max"].max()), float(targets["robust_delta_mean"].max()), 0.005)
    for index, target in targets.iterrows():
        if bool(target["is_key_target"]):
            color, marker, size = ACCENT_RED, "D", 34
        elif target["soc_association_class"] == "direct":
            color, marker, size = CATEGORICAL[0], "o", 30
        elif target["soc_association_class"] == "secondary":
            color, marker, size = CATEGORICAL[3], "o", 30
        else:
            color, marker, size = "#B8BEC5", "o", 25
        mean = float(target["robust_delta_mean"])
        low = float(target["robust_delta_min"])
        high = float(target["robust_delta_max"])
        ax_target.hlines(index, 0, mean, color=color, linewidth=1.4, alpha=0.85)
        ax_target.hlines(index, low, high, color=color, linewidth=3.5, alpha=0.22)
        ax_target.scatter(mean, index, color=color, marker=marker, s=size, edgecolor="white", linewidth=0.45, zorder=4)

    labels = []
    for target in targets.itertuples(index=False):
        label = str(target.target_gene_label)
        if len(case_manifest) > 1:
            label = f"{label}  [{target.task_label}]"
        labels.append(label)
    ax_target.set_yticks(y)
    ax_target.set_yticklabels(labels)
    ax_target.set_xlim(min(0, float(targets["robust_delta_min"].min()) * 1.1), max_delta * 1.18)
    ax_target.axvline(0, color="#AAB2BA", linewidth=0.6)
    ax_target.set_xlabel("ΔADR score after target replacement")

    unique_attr_doses = sorted(set(case_manifest["attribution_dose_mg_day"].astype(float)))
    compact_dose = ", ".join(format_dose(value) for value in unique_attr_doses)
    ax_target.set_title(f"Target attribution ({compact_dose} mg)", loc="left", fontweight="bold", pad=6)
    ax_target.grid(axis="x", color="#D8DDE3", linewidth=0.35, alpha=0.65)
    ax_target.tick_params(axis="y", length=0)

    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=ACCENT_RED, markeredgecolor="white", markersize=5, label="Key target"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORICAL[0], markeredgecolor="white", markersize=5, label="Direct ADR evidence"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORICAL[3], markeredgecolor="white", markersize=5, label="Network secondary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8BEC5", markeredgecolor="white", markersize=5, label="Other"),
    ]
    fig.legend(
        handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.025), ncol=4,
        fontsize=6.2, handletextpad=0.25, columnspacing=0.75, borderaxespad=0,
    )

    out_base = out_dir / f"selected_case_{figure_id}_exact_dose"
    save_cns_figure(fig, str(out_base))
    plt.close(fig)
    targets = targets.copy()
    targets["figure_id"] = figure_id
    return out_base.with_suffix(".png"), stats_rows, targets


def build_overview(image_paths: list[Path], out_dir: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width * 2, height * 4), "white")
    for index, image in enumerate(images):
        x = (index % 2) * width + (width - image.width) // 2
        y = (index // 2) * height + (height - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(out_dir / "selected_case_exact_dose_overview.png", dpi=(150, 150))
    images[0].close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    predictions = pd.read_csv(args.predictions_by_fold)
    occlusion = pd.read_csv(args.occlusion_summary)
    membership = pd.read_csv(args.visualization_membership)

    if manifest["figure_id"].nunique() != 8:
        raise ValueError("The selected-case manifest must contain eight unique drugs")
    if not np.isfinite(predictions["adr_score"]).all():
        raise ValueError("ADR predictions contain non-finite values")
    if predictions["adr_score"].min() < 0 or predictions["adr_score"].max() > 1:
        raise ValueError("ADR predictions fall outside [0, 1]")

    image_paths = []
    statistics = []
    target_tables = []
    for figure_id in FILE_ORDER:
        case_manifest = manifest[manifest["figure_id"].eq(figure_id)].copy()
        if case_manifest.empty:
            raise ValueError(f"Manifest lacks {figure_id}")
        image_path, rows, selected_targets = plot_case(
            figure_id, case_manifest, predictions, occlusion, membership, args.out_dir
        )
        image_paths.append(image_path)
        statistics.extend(rows)
        target_tables.append(selected_targets)

    build_overview(image_paths, args.out_dir)
    selected_targets_df = pd.concat(target_tables, ignore_index=True)
    selected_targets_df.to_csv(args.out_dir / "selected_case_top10_targets.csv", index=False)
    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(args.out_dir / "selected_case_exact_dose_statistics.csv", index=False)

    qa_rows = []
    for path in image_paths:
        with Image.open(path) as image:
            extrema = image.convert("L").getextrema()
            qa_rows.append(
                {
                    "file": path.name,
                    "width_px": image.width,
                    "height_px": image.height,
                    "nonblank": extrema[0] < extrema[1],
                    "pdf_exists": path.with_suffix(".pdf").exists(),
                }
            )
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(args.out_dir / "selected_case_exact_dose_figure_qa.csv", index=False)
    report = [
        "# Selected-case exact-dose figure QA",
        "",
        f"- Figures: {len(image_paths)} drug-specific PNG files and {len(image_paths)} vector PDF files.",
        f"- Prediction rows: {len(predictions):,}; all ADR scores are finite and within [0, 1].",
        f"- Target-attribution summary rows: {len(occlusion):,}.",
        f"- Nonblank PNG files: {int(qa['nonblank'].sum())}/{len(qa)}.",
        f"- Matching PDF files: {int(qa['pdf_exists'].sum())}/{len(qa)}.",
        "- Visual inspection is required for text overlap, target-label spacing and clinical-dose markers.",
        "",
        "Dose-response statistics are saved in `selected_case_exact_dose_statistics.csv`.",
    ]
    (args.out_dir / "selected_case_exact_dose_figure_qa.md").write_text("\n".join(report), encoding="utf-8")
    print(statistics_df.to_string(index=False))


if __name__ == "__main__":
    main()
