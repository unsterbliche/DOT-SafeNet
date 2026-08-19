#!/usr/bin/env python3
"""Evaluate baseline and clinically fine-tuned DOT-SafeNet weights on CT-ADE.

The script performs inference only. It evaluates the same clinical-dose test
set with the matched baseline and fine-tuned checkpoint for each fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from train_hetsia_soft_pseudo_monotone_loss_task_head import (
    ViewDataMO,
    parse_dataset,
    siamese_model_sider_adr_emb_task_head,
)


def load_clinical_test(dataset_dir: Path):
    x_df = pd.read_csv(dataset_dir / "test_X.csv")
    y_df = pd.read_csv(dataset_dir / "test_y.csv")
    dim_x = len(x_df.columns[2:])
    x, y, task, smiles, task_names = parse_dataset(ViewDataMO(x_df, y_df), dim_x)
    return dim_x, x, y, task, smiles, task_names


def predict(model, weights: Path, x: np.ndarray, task: np.ndarray) -> np.ndarray:
    model.load_weights(str(weights))
    return model.predict([x, task], batch_size=512, verbose=0).reshape(-1)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(y, prob)),
        "AUPRC": float(average_precision_score(y, prob)),
    }


def paired_tests(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["AUROC", "AUPRC"]:
        before = values.loc[values["stage"] == "Before", metric].to_numpy()
        after = values.loc[values["stage"] == "After", metric].to_numpy()
        t = ttest_rel(after, before)
        try:
            w = wilcoxon(after, before)
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        rows.append(
            {
                "metric": metric,
                "n_folds": len(before),
                "before_mean": before.mean(),
                "before_sd": before.std(ddof=1),
                "after_mean": after.mean(),
                "after_sd": after.std(ddof=1),
                "mean_change": (after - before).mean(),
                "paired_t_statistic": float(t.statistic),
                "paired_t_pvalue": float(t.pvalue),
                "wilcoxon_statistic": w_stat,
                "wilcoxon_pvalue": w_p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-test-dataset-dir", required=True, type=Path)
    parser.add_argument("--baseline-weights-dir", required=True, type=Path)
    parser.add_argument("--finetuned-result-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dim_x, x, y, task, smiles, task_names = load_clinical_test(args.clinical_test_dataset_dir)
    fold_metrics = []

    for fold in range(1, 6):
        model = siamese_model_sider_adr_emb_task_head(dim_X=dim_x, lr=7e-6)
        before = predict(
            model,
            args.baseline_weights_dir / f"best_model_fold_{fold}_weights",
            x,
            task,
        )
        after = predict(
            model,
            args.finetuned_result_dir / f"fold_{fold}" / "best_finetuned_weights",
            x,
            task,
        )

        result = pd.DataFrame(
            {
                "smiles": smiles,
                "task_name": task_names,
                "label_true": y,
                "before_pred": before,
                "after_pred": after,
            }
        )
        result.to_csv(args.out_dir / f"fold_{fold}_clinical_test_before_after.csv", index=False)
        for stage, probability in [("Before", before), ("After", after)]:
            fold_metrics.append({"fold": fold, "stage": stage, **metrics(y, probability)})

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(args.out_dir / "clinical_test_before_after_metrics_by_fold.csv", index=False)
    tests = paired_tests(fold_metrics_df)
    tests.to_csv(args.out_dir / "clinical_test_before_after_statistics.csv", index=False)
    metadata = {
        "n_unique_drugs": int(pd.Series(smiles).nunique()),
        "n_drug_soc_pairs": int(len(y)),
        "clinical_test_dataset_dir": str(args.clinical_test_dataset_dir),
        "baseline_weights_dir": str(args.baseline_weights_dir),
        "finetuned_result_dir": str(args.finetuned_result_dir),
    }
    (args.out_dir / "clinical_test_before_after_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(fold_metrics_df.to_string(index=False))
    print(tests.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
