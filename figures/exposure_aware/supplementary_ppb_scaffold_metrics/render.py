from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "supplementary_table_2_plasmabindnet_and_fu.csv"
OUT = ROOT / "outputs"
MODELS = ["RF", "XGBoost", "LightGBM", "AttentiveFP", "MolSetRep", "PlasmaBindNet", "PlasmaBindNet-Fu"]
LABELS = ["RF", "XGBoost", "LightGBM", "AttentiveFP", "MolSetRep", "PlasmaBindNet", "PlasmaBindNet-Fu"]

PUBLISHED = {
    "RF": {"RMSE": (0.588, 0.0015), "MAE": (0.468, 0.0012)},
    "XGBoost": {"RMSE": (0.600, 0.0100), "MAE": (0.471, 0.0060)},
    "LightGBM": {"RMSE": (0.587, 0.0020), "MAE": (0.468, 0.0040)},
    "AttentiveFP": {"RMSE": (0.513, 0.0080), "MAE": (0.388, 0.0060)},
    "MolSetRep": {"RMSE": (0.512, 0.0090), "MAE": (0.384, 0.0080)},
    "PlasmaBindNet": {"RMSE": (0.486, 0.0050), "MAE": (0.365, 0.0060)},
}

def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def full_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.tick_params(direction="out", length=3, width=0.8)

def summarize() -> pd.DataFrame:
    table = pd.read_csv(DATA).set_index("model")
    return table.loc[MODELS].reset_index()

def draw(ax, table, metric, cfg):
    colors = [cfg["colors"]["comparator"]] * len(table)
    colors[-2] = cfg["colors"]["original"]
    colors[-1] = cfg["colors"]["bias_aware"]
    x = np.arange(len(table))
    ax.bar(x, table[f"{metric}_mean"], yerr=table[f"{metric}_sd"],
           width=0.72, color=colors, edgecolor="white", linewidth=0.5,
           capsize=2.5, error_kw={"lw": 0.8, "capthick": 0.8})
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=cfg["layout"]["rotation"], ha="right")
    ax.set_ylim(*cfg["axes"][f"{metric.lower()}_limits"])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    full_frame(ax)

def save(fig, stem: str, cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in cfg["export"]["formats"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=cfg["figure"]["dpi"],
                    bbox_inches="tight", transparent=cfg["export"]["transparent"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cfg["text"]["font_family"], "Helvetica", "DejaVu Sans"],
        "font.size": cfg["text"]["font_size"],
        "axes.labelsize": cfg["text"]["label_size"],
        "xtick.labelsize": cfg["text"]["tick_size"],
        "ytick.labelsize": cfg["text"]["tick_size"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    table = summarize()
    OUT.mkdir(parents=True, exist_ok=True)
    for metric in ("RMSE", "MAE"):
        fig, ax = plt.subplots(figsize=(cfg["figure"]["panel_width_in"], cfg["figure"]["panel_height_in"]))
        draw(ax, table, metric, cfg)
        fig.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.36)
        save(fig, metric.lower(), cfg)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(cfg["figure"]["composite_width_in"], cfg["figure"]["composite_height_in"]))
    draw(axes[0], table, "RMSE", cfg)
    draw(axes[1], table, "MAE", cfg)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.36, wspace=cfg["layout"]["wspace"])
    save(fig, "supplementary_ppb_scaffold_metrics", cfg)
    table.to_csv(OUT / "source_data_summary.csv", index=False)

if __name__ == "__main__":
    main()
