# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# ROC: AUROC asset; heatmap: heatmap asset; UMAP: established project visual system.
# Existing source data and analysis logic are preserved.

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.stats import ttest_rel, wilcoxon

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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
LEGENDS = OUT / "legends"
TASKS = [
    ("BLO", "Blood and lymphatic system disorders"), ("CAR", "Cardiac disorders"),
    ("EAR", "Ear and labyrinth disorders"), ("END", "Endocrine disorder"),
    ("EYE", "Eye disorders"), ("GAS", "Gastrointestinal disorders"),
    ("HEP", "Hepatobiliary disorders"), ("IMM", "Immune system disorders"),
    ("INF", "Infections and infestations"), ("MET", "Metabolism and nutrition disorders"),
    ("MUS", "Musculoskeletal and connective tissue disorders"), ("NER", "Nervous system disorders"),
    ("PSY", "Psychiatric disorders"), ("REN", "Renal and urinary disorders"),
    ("REP", "Reproductive system and breast disorders"),
    ("RES", "Respiratory, thoracic and mediastinal disorders"),
    ("SKI", "Skin and subcutaneous tissue disorders"), ("VAS", "Vascular disorders"),
]

def frame(ax, ticks=True):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#222222")
    if ticks:
        ax.tick_params(direction="out", length=2.5, width=0.7)

def save_to(fig, directory, stem, cfg):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in cfg["export"]["formats"]:
        fig.savefig(directory / f"{stem}.{ext}", dpi=cfg["figure"]["dpi"],
                    bbox_inches="tight", transparent=cfg["export"]["transparent"])

def save(fig, stem, cfg):
    save_to(fig, OUT, stem, cfg)

def panel_a(cfg):
    table = pd.read_csv(DATA / "figure4a_random_source_data.csv")
    stats = pd.read_csv(DATA / "figure4a_random_statistics.csv")
    colors = [cfg["colors"]["comparator"], "#9FD3C7", "#2A8C8D"]
    fig, ax = plt.subplots(figsize=cfg["figure"]["panel_a_size"])
    x = np.arange(len(table))
    ax.bar(x, table["auroc_mean"], yerr=table["auroc_sd"], width=0.66, color=colors,
           edgecolor="white", linewidth=0.5, capsize=2.5,
           error_kw={"lw": 0.8, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(table["display_name"], rotation=25, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.64, 0.760)
    for i, (value, sd) in enumerate(zip(table["auroc_mean"], table["auroc_sd"])):
        ax.text(i, value + sd + 0.0025, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
    y0 = 0.750
    for offset, (_, row) in enumerate(stats.iterrows()):
        left = 0 if "MLDNN" in row["comparison"] else 1
        right = 2
        y = y0 - offset * 0.013
        ax.plot([left, left, right, right], [y - 0.002, y, y, y - 0.002], color="#444444", lw=0.7)
        ax.text((left + right) / 2, y + 0.001, row["significance"], ha="center", va="bottom", fontsize=8)
    frame(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.29)
    save(fig, "figure4a_model_comparison", cfg)
    plt.close(fig)


def _draw_model_comparison(ax, cfg):
    table = pd.read_csv(DATA / "figure4a_random_source_data.csv")
    stats = pd.read_csv(DATA / "figure4a_random_statistics.csv")
    colors = [cfg["colors"]["comparator"], "#9FD3C7", "#2A8C8D"]
    x = np.arange(len(table))
    ax.bar(x, table["auroc_mean"], yerr=table["auroc_sd"], width=0.66,
           color=colors, edgecolor="white", linewidth=0.5, capsize=2.5,
           error_kw={"lw": 0.8, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(table["display_name"], rotation=25, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.64, 0.760)
    for i, (value, sd) in enumerate(zip(table["auroc_mean"], table["auroc_sd"])):
        ax.text(i, value + sd + 0.0025, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
    y0 = 0.750
    for offset, (_, row) in enumerate(stats.iterrows()):
        left = 0 if "MLDNN" in row["comparison"] else 1
        right = 2
        y = y0 - offset * 0.013
        ax.plot([left, left, right, right], [y - 0.002, y, y, y - 0.002], color="#444444", lw=0.7)
        ax.text((left + right) / 2, y + 0.001, row["significance"], ha="center", va="bottom", fontsize=8)
    frame(ax)

def _draw_finetune_metric(ax, metric, cfg, source_path, xlabel):
    source = pd.read_csv(source_path)
    if set(source["stage"]) == {"Before", "After"}:
        source = source.assign(stage=source["stage"].map({"Before": "baseline", "After": "finetuned"}))
    stage_order = ["baseline", "finetuned"]
    means = source.groupby("stage", sort=False)[metric].mean().reindex(stage_order)
    sds = source.groupby("stage", sort=False)[metric].std(ddof=1).reindex(stage_order)
    paired = source.pivot(index="fold", columns="stage", values=metric).sort_index()
    before = paired["baseline"].to_numpy(float)
    after = paired["finetuned"].to_numpy(float)
    p_value = float(ttest_rel(after, before).pvalue)
    x = np.arange(2)
    ax.bar(x, means, yerr=sds, width=0.62,
           color=[cfg["colors"]["before_finetune"], cfg["colors"]["after_finetune"]],
           alpha=0.82, edgecolor="white", linewidth=0.5, capsize=2.5,
           error_kw={"lw": 0.8, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(["Before", "After"])
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(metric)
    values = np.concatenate([before, after])
    span = max(float(values.max() - values.min()), 0.004)
    lower = max(0.0, float(values.min() - 0.30 * span))
    error_tops = (means + sds).to_numpy(float)
    label_ys = error_tops + 0.12 * span
    bracket_y = float(label_ys.max() + 0.32 * span)
    bracket_h = 0.08 * span
    upper = min(1.0, float(bracket_y + 0.30 * span))
    ax.set_ylim(lower, upper)
    for i, (value, label_y) in enumerate(zip(means, label_ys)):
        ax.text(i, label_y, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7.2)
    ax.plot([0, 0, 1, 1],
            [bracket_y - bracket_h, bracket_y, bracket_y, bracket_y - bracket_h],
            color="#333333", lw=0.65, clip_on=False)
    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    ax.text(0.5, bracket_y + bracket_h * 0.35, stars,
            ha="center", va="bottom", fontsize=8.0)
    frame(ax)

def panel_training_comparison(cfg):
    original_source = DATA / "figure4_clinical_finetune_fivefold_metrics.csv"
    clinical_source = DATA / "clinical_test_57" / "clinical_test_57_before_after_metrics_by_fold.csv"
    original = pd.read_csv(original_source)
    original.groupby("stage", sort=False)[["AUROC", "AUPRC"]].agg(["mean", "std"]).to_csv(
        OUT / "figure4_original_test_finetuning_summary.csv"
    )
    clinical = pd.read_csv(clinical_source)
    clinical.groupby("stage", sort=False)[["AUROC", "AUPRC"]].agg(["mean", "std"]).to_csv(
        OUT / "figure4_clinical57_finetuning_summary.csv"
    )
    pd.read_csv(DATA / "clinical_test_57" / "clinical_test_57_before_after_statistics.csv").to_csv(
        OUT / "figure4_clinical_dose_finetuning_statistics.csv", index=False
    )
    for metric, stem in (("AUROC", "figure4_finetune_auroc"), ("AUPRC", "figure4_finetune_auprc")):
        fig, ax = plt.subplots(figsize=cfg["figure"]["finetune_metric_size"])
        _draw_finetune_metric(ax, metric, cfg, clinical_source, "Clinical-dose test\n(57 drugs)")
        fig.subplots_adjust(left=0.25, right=0.98, top=0.94, bottom=0.23)
        save(fig, stem, cfg)
        plt.close(fig)
    for metric, stem in (("AUROC", "figure4_original_test_finetune_auroc"),
                         ("AUPRC", "figure4_original_test_finetune_auprc")):
        fig, ax = plt.subplots(figsize=cfg["figure"]["finetune_metric_size"])
        _draw_finetune_metric(ax, metric, cfg, original_source, "Original test (921 drugs)")
        fig.subplots_adjust(left=0.25, right=0.98, top=0.94, bottom=0.23)
        save(fig, stem, cfg)
        plt.close(fig)
    fig = plt.figure(figsize=cfg["figure"]["training_comparison_size"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.0, 1.0], wspace=0.46)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    _draw_model_comparison(axes[0], cfg)
    _draw_finetune_metric(axes[1], "AUROC", cfg, original_source, "Test set\n(921 drugs)")
    _draw_finetune_metric(axes[2], "AUROC", cfg, clinical_source, "Clinical-dose test\n(57 drugs)")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.94, bottom=0.28)
    save(fig, "figure4_model_comparison_and_clinical_finetuning", cfg)
    plt.close(fig)

def roc_data():
    scores = pd.read_csv(DATA / "per_adr_roc_a2mono_clinical_scores.csv")
    methods = [
        ("DOT-SafeNet", "a2mono_clinical_score"),
        ("AC50 heuristic", "ac50_score"),
        ("Free-margin heuristic", "free_margin_score"),
    ]
    curves, rows = {}, []
    for abbr, name in TASKS:
        subset = scores.loc[scores["task_name"].eq(name)]
        y = subset["label_true"].to_numpy(int)
        curves[abbr] = {}
        for label, column in methods:
            v = subset[column].to_numpy(float)
            valid = np.isfinite(v)
            if valid.sum() == 0:
                curves[abbr][label] = None
                score = np.nan
            else:
                fpr, tpr, _ = roc_curve(y[valid], v[valid])
                score = auc(fpr, tpr)
                curves[abbr][label] = (fpr, tpr, score)
            rows.append({"adr_abbr": abbr, "method": label, "auroc": score})
    return curves, pd.DataFrame(rows)

def panel_b(cfg):
    curves, metrics = roc_data()
    colors = {
        "DOT-SafeNet": cfg["colors"]["dotsafenet"],
        "AC50 heuristic": cfg["colors"]["ac50"],
        "Free-margin heuristic": cfg["colors"]["free_margin"],
    }
    styles = {"DOT-SafeNet": "-", "AC50 heuristic": "-", "Free-margin heuristic": "--"}
    fig, axes = plt.subplots(
        cfg["layout"]["roc_rows"], cfg["layout"]["roc_columns"],
        figsize=cfg["figure"]["panel_b_size"], sharex=True, sharey=True
    )
    for index, (ax, (abbr, _)) in enumerate(zip(axes.flat, TASKS)):
        row, col = divmod(index, cfg["layout"]["roc_columns"])
        ax.plot([0, 1], [0, 1], color="#C7C7C7", lw=0.55, ls=":")
        for label in colors:
            curve = curves[abbr][label]
            if curve is not None:
                ax.plot(
                    curve[0], curve[1], color=colors[label], ls=styles[label],
                    lw=1.15 if label == "DOT-SafeNet" else 0.8
                )
        score = metrics.loc[
            (metrics.adr_abbr == abbr) & (metrics.method == "DOT-SafeNet"), "auroc"
        ].iloc[0]
        ax.text(
            0.94, 0.055, f"{abbr}\n{score:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontweight="bold", fontsize=7.0,
            linespacing=1.05
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if col != 0:
            ax.tick_params(labelleft=False)
        if row != cfg["layout"]["roc_rows"] - 1:
            ax.tick_params(labelbottom=False)
        frame(ax)
    for ax in axes[-1]:
        ax.set_xlabel("FPR", labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("TPR", labelpad=1)
    fig.subplots_adjust(
        left=0.065, right=0.995, top=0.985, bottom=0.14,
        wspace=0.16, hspace=0.15
    )
    save(fig, "figure4a_roc_18soc", cfg)
    metrics.to_csv(OUT / "figure4a_roc_metrics.csv", index=False)
    plt.close(fig)

    means = metrics.groupby("method", sort=False)["auroc"].mean()
    handles = [
        Line2D(
            [0], [0], color=colors[k], ls=styles[k], lw=1.3,
            label=f"{k}  mean AUROC {means[k]:.3f}"
        )
        for k in colors
    ]
    handles.append(
        Line2D([0], [0], color="#C7C7C7", ls=":", lw=0.9, label="Chance")
    )
    legend_fig = plt.figure(figsize=(7.2, 0.38))
    legend_fig.legend(
        handles=handles, loc="center", ncol=4, frameon=False,
        handlelength=1.8, columnspacing=1.0
    )
    save_to(legend_fig, LEGENDS, "figure4a_roc_legend", cfg)
    plt.close(legend_fig)

def load_matrix(name):
    data = pd.read_csv(DATA / name)
    return data.drop(columns=["smiles", "drug"], errors="ignore")

def panel_c(cfg):
    label_full = pd.read_csv(DATA / "dotsafenet_heatmap_ordered_label_matrix.csv")
    order_smiles = label_full["smiles"].astype(str).tolist()
    soc_order = [column for column in label_full.columns if column not in {"smiles", "drug"}]
    label = label_full[soc_order]

    dose_long = pd.read_csv(DATA / "dotsafenet_full_test_18soc_dose_predictions.csv.gz")
    predictions = {}
    for dose in (20.0, 250.0):
        matrix = (
            dose_long.loc[np.isclose(dose_long["dose_mg_day"].astype(float), dose)]
            .pivot(index="smiles", columns="adr_abbr", values="ensemble_probability")
            .reindex(index=order_smiles, columns=soc_order)
        )
        if matrix.isna().any().any():
            raise ValueError(f"Incomplete DOT-SafeNet dose matrix at {dose:g} mg/day")
        predictions[dose] = matrix

    fig = plt.figure(figsize=cfg["figure"]["panel_c_size"])
    gs = fig.add_gridspec(
        1, 3, left=0.065, right=0.995, top=0.985, bottom=0.085, wspace=0.14
    )
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    pred_cmap = LinearSegmentedColormap.from_list(
        "probability", ["#F7FBFF", "#6BAED6", "#08306B"]
    )
    label_cmap = ListedColormap(["#F7F7F7", cfg["colors"]["positive"]])
    label_cmap.set_bad("#BDBDBD")

    axes[0].imshow(
        np.ma.masked_invalid(label.to_numpy(float)),
        cmap=label_cmap, vmin=0, vmax=1, aspect="auto",
        interpolation="nearest", rasterized=True
    )
    for ax, dose in zip(axes[1:], (20.0, 250.0)):
        ax.imshow(
            predictions[dose].to_numpy(float), cmap=pred_cmap, vmin=0, vmax=1,
            aspect="auto", interpolation="nearest", rasterized=True
        )
    for ax, title in zip(axes, ("Observed label", "20 mg/day", "250 mg/day")):
        ax.set_xticks(np.arange(len(soc_order)))
        ax.set_xticklabels(soc_order, fontsize=5.0, rotation=90)
        ax.set_yticks([])
        ax.set_title(title, loc="left", pad=3.0, fontweight="bold", fontsize=8)
        ax.tick_params(axis="x", length=0, pad=2)
        frame(ax, ticks=False)
    save(fig, "figure4b_observed_low_high_dose_heatmap", cfg)
    save(fig, "figure4b_observed_prediction_heatmap", cfg)
    plt.close(fig)

    low = predictions[20.0]
    high = predictions[250.0]
    dose_stats = []
    for soc in soc_order:
        delta = high[soc].to_numpy(float) - low[soc].to_numpy(float)
        dose_stats.append({
            "adr_abbr": soc,
            "mean_probability_20mg": float(low[soc].mean()),
            "mean_probability_250mg": float(high[soc].mean()),
            "mean_delta_250_minus_20": float(delta.mean()),
            "median_delta_250_minus_20": float(np.median(delta)),
            "fraction_increased": float(np.mean(delta > 0)),
            "fraction_crossing_0.5": float(np.mean((low[soc] < 0.5) & (high[soc] >= 0.5))),
        })
    pd.DataFrame(dose_stats).to_csv(OUT / "figure4b_low_high_dose_statistics.csv", index=False)

    legend_fig = plt.figure(figsize=(7.2, 0.62))
    cax = legend_fig.add_axes([0.08, 0.48, 0.30, 0.18])
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=pred_cmap
    )
    cb = legend_fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.set_label("DOT-SafeNet probability", labelpad=1)
    cb.outline.set_linewidth(0.6)
    handles = [
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="#F7F7F7", markeredgecolor="#B0B0B0",
               markersize=5, label="Negative"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor=cfg["colors"]["positive"], markeredgecolor="none",
               markersize=5, label="Positive"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="#BDBDBD", markeredgecolor="none",
               markersize=5, label="Missing label"),
    ]
    legend_fig.legend(
        handles=handles, loc="center right", bbox_to_anchor=(0.98, 0.55),
        ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.9
    )
    save_to(legend_fig, LEGENDS, "figure4b_heatmap_legend", cfg)
    plt.close(legend_fig)
def draw_soc(ax, source, abbr, xlim, ylim, cfg):
    selected = source.loc[source["adr_abbr"].eq(abbr)]
    predicted = selected.loc[
        selected["ensemble_probability"].gt(cfg["axes"]["threshold"])
    ]
    pair = "gas_vas" if abbr in {"GAS", "VAS"} else "ner_psy"
    light = cfg["colors"][f"{pair}_light"]
    dark = cfg["colors"][f"{pair}_dark"]
    ax.scatter(
        source["UMAP_1"], source["UMAP_2"], s=1.5,
        c=cfg["colors"]["background"], alpha=0.075,
        edgecolors="none", rasterized=True
    )
    ax.scatter(
        selected["UMAP_1"], selected["UMAP_2"], s=6.2,
        c=light, alpha=0.72,
        edgecolors="none", rasterized=True
    )
    ax.scatter(
        predicted["UMAP_1"], predicted["UMAP_2"], s=7.4,
        c=dark, alpha=0.92,
        edgecolors="none", rasterized=True
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.035, 0.93, abbr, transform=ax.transAxes,
        ha="left", va="top", fontweight="bold", fontsize=8
    )
    frame(ax, ticks=False)
def panel_d(cfg):
    source = pd.read_csv(DATA / "selected_soc_relationship_umap_source_data.csv.gz")
    dx = source["UMAP_1"].max() - source["UMAP_1"].min()
    dy = source["UMAP_2"].max() - source["UMAP_2"].min()
    xlim = (
        source["UMAP_1"].min() - 0.075 * dx,
        source["UMAP_1"].max() + 0.075 * dx,
    )
    ylim = (
        source["UMAP_2"].min() - 0.10 * dy,
        source["UMAP_2"].max() + 0.10 * dy,
    )
    fig, axes = plt.subplots(
        1, 4, figsize=cfg["figure"]["panel_d_size"], sharex=True, sharey=True
    )
    for ax, abbr in zip(np.ravel(axes), ("GAS", "VAS", "NER", "PSY")):
        draw_soc(ax, source, abbr, xlim, ylim, cfg)
    fig.subplots_adjust(
        left=0.018, right=0.995, top=0.98, bottom=0.045,
        wspace=cfg["layout"]["umap_wspace"],
        hspace=cfg["layout"]["umap_hspace"],
    )
    save(fig, "figure4c_soc_relationship_umap", cfg)
    plt.close(fig)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["background"], markeredgecolor="none",
               markersize=4, label="Other observed-positive pairs"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["gas_vas_light"], markeredgecolor="none",
               markersize=4, label="GAS-VAS observed positive"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["gas_vas_dark"], markeredgecolor="none",
               markersize=4, label="GAS-VAS observed + predicted"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["ner_psy_light"], markeredgecolor="none",
               markersize=4, label="NER-PSY observed positive"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cfg["colors"]["ner_psy_dark"], markeredgecolor="none",
               markersize=4, label="NER-PSY observed + predicted"),
    ]
    legend_fig = plt.figure(figsize=(7.2, 0.48))
    legend_fig.legend(
        handles=handles, loc="center", ncol=5, frameon=False,
        handletextpad=0.30, columnspacing=0.75, fontsize=6.4
    )
    save_to(legend_fig, LEGENDS, "figure4c_umap_legend", cfg)
    plt.close(legend_fig)
def trim(image):
    rgb = image.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).convert("L")
    box = diff.point(lambda v: 255 if v > 8 else 0).getbbox()
    return rgb.crop(box) if box else rgb

def composite(cfg):
    roc = trim(Image.open(OUT / "figure4a_roc_18soc.png"))
    heatmap = trim(Image.open(OUT / "figure4b_observed_prediction_heatmap.png"))
    umap = trim(Image.open(OUT / "figure4c_soc_relationship_umap.png"))
    width = 2160
    margin = 36
    gap = 34
    top_h, umap_h = 1520, 520
    height = margin * 2 + top_h + umap_h + gap
    canvas = Image.new("RGB", (width, height), "white")
    left_w = 1170
    boxes = [
        (roc, (margin, margin, margin + left_w, margin + top_h)),
        (heatmap, (margin + left_w + gap, margin, width - margin, margin + top_h)),
        (umap, (margin, margin + top_h + gap, width - margin, height - margin)),
    ]
    for im, box in boxes:
        fitted = ImageOps.contain(
            im, (box[2] - box[0], box[3] - box[1]), Image.LANCZOS
        )
        x = box[0] + (box[2] - box[0] - fitted.width) // 2
        y = box[1] + (box[3] - box[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))
    canvas.save(
        OUT / "figure_4.png",
        dpi=(cfg["figure"]["dpi"], cfg["figure"]["dpi"]),
    )
    canvas.save(OUT / "figure_4.pdf", "PDF", resolution=cfg["figure"]["dpi"])
    canvas.save(
        OUT / "latest.png",
        dpi=(cfg["figure"]["dpi"], cfg["figure"]["dpi"]),
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cfg["text"]["font_family"], "Helvetica", "DejaVu Sans"],
        "font.size": cfg["text"]["font_size"], "axes.labelsize": cfg["text"]["label_size"],
        "xtick.labelsize": cfg["text"]["tick_size"], "ytick.labelsize": cfg["text"]["tick_size"],
        "legend.fontsize": cfg["text"]["legend_size"], "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    panel_training_comparison(cfg)
    panel_b(cfg); panel_c(cfg); panel_d(cfg); composite(cfg)

if __name__ == "__main__":
    main()
