#!/usr/bin/env python3
"""Predict 18 SOC scores and target-replacement effects with DOT-SafeNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--features-csv", required=True, type=Path)
    parser.add_argument("--target-reference-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    sys.path.insert(0, str(release_root / "src/dotsafenet"))
    sys.path.insert(0, str(release_root / "src"))
    from dotsafenet.train_hetsia_soft_pseudo_monotone_loss_task_head import siamese_model_sider_adr_emb_task_head
    from safety_profiler.core import load_contract, read_target_reference, validate_feature_table

    contract = load_contract(release_root)
    feature_df = pd.read_csv(args.features_csv)
    validate_feature_table(feature_df, contract)
    x = feature_df[list(contract.feature_columns)].to_numpy(np.float32)
    manifest = pd.read_csv(release_root / "weights/weights_manifest.tsv", sep="\t")
    selected = manifest[(manifest["model"] == "DOT-SafeNet") & (manifest["role"] == "clinical_index")].sort_values("member")
    if len(selected) != 5:
        raise ValueError(f"Expected five clinical DOT-SafeNet checkpoints, found {len(selected)}")
    prefixes = [args.checkpoint_root / str(path).removesuffix(".index") for path in selected["path"]]
    for prefix in prefixes:
        if not prefix.with_suffix(".index").exists():
            raise FileNotFoundError(prefix.with_suffix(".index"))

    n_rows = len(feature_df)
    n_tasks = len(contract.soc_codes)
    repeated_x = np.repeat(x, n_tasks, axis=0)
    task = np.tile(np.arange(n_tasks, dtype=np.int32), n_rows).reshape(-1, 1)
    fold_predictions = []
    reference = read_target_reference(args.target_reference_csv, contract)
    attr_by_fold = []

    for fold, prefix in enumerate(prefixes, start=1):
        model = siamese_model_sider_adr_emb_task_head(dim_X=389, lr=7e-6)
        model.load_weights(str(prefix))
        pred = model.predict([repeated_x, task], batch_size=args.batch_size, verbose=0).reshape(n_rows, n_tasks)
        fold_predictions.append(pred)

        if not args.skip_attribution:
            fold_attr = np.empty((n_rows, n_tasks, 194), dtype=np.float32)
            for row_index in range(n_rows):
                base = x[row_index]
                masked = np.repeat(base[None, :], n_tasks * 194, axis=0)
                target_index = np.tile(np.arange(194), n_tasks)
                masked[np.arange(n_tasks * 194), 1 + target_index] = np.tile(reference, n_tasks)
                exposure = float(base[0])
                masked[np.arange(n_tasks * 194), 1 + 194 + target_index] = np.tile(reference * exposure, n_tasks)
                masked_task = np.repeat(np.arange(n_tasks, dtype=np.int32), 194).reshape(-1, 1)
                masked_pred = model.predict([masked, masked_task], batch_size=args.batch_size, verbose=0).reshape(n_tasks, 194)
                fold_attr[row_index] = pred[row_index, :, None] - masked_pred
            attr_by_fold.append(fold_attr)
        del model
        tf.keras.backend.clear_session()

    prediction = np.stack(fold_predictions, axis=0)
    rows = []
    by_fold_rows = []
    for i, record in feature_df.iterrows():
        for task_index, (soc_code, soc_name) in enumerate(zip(contract.soc_codes, contract.soc_names)):
            values = prediction[:, i, task_index]
            rows.append({
                "compound_id": record["compound_id"], "name": record["Drug"], "smiles": record["smiles"],
                "soc_code": soc_code, "soc_name": soc_name,
                "probability_mean": float(values.mean()), "probability_sd": float(values.std(ddof=1)),
                "probability_min": float(values.min()), "probability_max": float(values.max()),
            })
            for fold, value in enumerate(values, start=1):
                by_fold_rows.append({"compound_id": record["compound_id"], "soc_code": soc_code, "fold": fold, "probability": float(value)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "adr_predictions.csv", index=False)
    pd.DataFrame(by_fold_rows).to_csv(args.output_dir / "adr_predictions_by_fold.csv", index=False)

    if attr_by_fold:
        attribution = np.stack(attr_by_fold, axis=0)
        attr_rows = []
        for i, record in feature_df.iterrows():
            for task_index, soc_code in enumerate(contract.soc_codes):
                for target_index, target in enumerate(contract.target_ids):
                    values = attribution[:, i, task_index, target_index]
                    attr_rows.append({
                        "compound_id": record["compound_id"], "name": record["Drug"],
                        "soc_code": soc_code, "target_uniprot": target,
                        "delta_probability": float(values.mean()), "delta_probability_sd": float(values.std(ddof=1)),
                        "positive_fold_fraction": float((values > 0).mean()),
                    })
        pd.DataFrame(attr_rows).to_csv(args.output_dir / "target_attribution.csv", index=False)
    print(f"Wrote DOT-SafeNet outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
