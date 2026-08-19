#!/usr/bin/env python3
"""Recalculate the DOT-SafeNet metrics used in manuscript Figure 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "exposure_aware" / "figure_4" / "data"


def clinical_57_metrics() -> tuple[pd.DataFrame, float]:
    expected = pd.read_csv(DATA / "clinical_test_57" / "clinical_test_57_before_after_metrics_by_fold.csv")
    rows = []
    for fold in range(1, 6):
        table = pd.read_csv(DATA / "clinical_test_57" / f"fold_{fold}_clinical_test_before_after_57.csv")
        y = table["label_true"].to_numpy(dtype=int)
        for stage, column in (("Before", "before_pred"), ("After", "after_pred")):
            pred = table[column].to_numpy(dtype=float)
            rows.append({
                "fold": fold,
                "stage": stage,
                "AUROC": roc_auc_score(y, pred),
                "AUPRC": average_precision_score(y, pred),
                "n_drugs": table["smiles"].nunique(),
                "n_drug_soc_pairs": len(table),
            })
    calculated = pd.DataFrame(rows)
    merged = calculated.merge(expected, on=["fold", "stage"], suffixes=("_calculated", "_reported"))
    metric_columns = ["AUROC", "AUPRC", "n_drugs", "n_drug_soc_pairs"]
    max_difference = max(
        float(np.max(np.abs(merged[f"{metric}_calculated"] - merged[f"{metric}_reported"])))
        for metric in metric_columns
    )
    return calculated, max_difference


def per_soc_auroc() -> pd.DataFrame:
    table = pd.read_csv(DATA / "per_adr_roc_a2mono_clinical_scores.csv")
    rows = []
    for (abbr, task), group in table.groupby(["adr_abbr", "task_name"], sort=False):
        group = group.dropna(subset=["a2_label_true", "a2mono_clinical_score"])
        rows.append({
            "soc": abbr,
            "task_name": task,
            "n_pairs": len(group),
            "n_positive": int(group["a2_label_true"].sum()),
            "auroc": roc_auc_score(group["a2_label_true"].astype(int), group["a2mono_clinical_score"]),
        })
    return pd.DataFrame(rows)


def base_and_finetune_summary() -> dict:
    table = pd.read_csv(DATA / "figure4_clinical_finetune_fivefold_metrics.csv")
    summary = {}
    for stage, group in table.groupby("stage"):
        summary[stage] = {
            "AUROC_mean": float(group["AUROC"].mean()),
            "AUROC_sd": float(group["AUROC"].std(ddof=1)),
            "AUPRC_mean": float(group["AUPRC"].mean()),
            "AUPRC_sd": float(group["AUPRC"].std(ddof=1)),
            "folds": int(len(group)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "qc" / "reproduced_metrics")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clinical, max_difference = clinical_57_metrics()
    soc = per_soc_auroc()
    summary = {
        "status": "passed" if max_difference <= args.tolerance else "failed",
        "clinical_57_max_absolute_difference": max_difference,
        "clinical_57": {
            stage: {
                metric: float(group[metric].mean())
                for metric in ("AUROC", "AUPRC")
            }
            for stage, group in clinical.groupby("stage")
        },
        "base_test_fivefold": base_and_finetune_summary(),
        "per_soc_mean_auroc": float(soc["auroc"].mean()),
        "per_soc_tasks": int(len(soc)),
    }
    clinical.to_csv(args.output_dir / "clinical_57_metrics_recalculated.csv", index=False)
    soc.to_csv(args.output_dir / "per_soc_auroc_recalculated.csv", index=False)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
