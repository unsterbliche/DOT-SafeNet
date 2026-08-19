#!/usr/bin/env python3
"""Rebuild HetSia-SafeNet AC50_Cmax split files from a PreMOTA prediction JSON.

The existing HetSia feature contract is:

    Drug, smiles,
    cmax_free(uM),
    194 off-target features,
    194 off-target*Cmax features

The original notebooks build the published AC50_Cmax target features as:

    feature = log10(AC50_uM) = 6 - raw_pAC50

and the interaction feature is:

    feature * log10(cmax_free_uM)

This script preserves the original train/valid/test rows, Cmax values, target
order, and y files, while replacing only the off-target-derived columns from a
new PreMOTA JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")


def load_premota_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def p_ac50_for(predictions: dict, smiles: str, target: str) -> float:
    try:
        value = predictions[smiles][target]
    except KeyError as exc:
        raise KeyError(f"Missing PreMOTA prediction for smiles={smiles!r}, target={target!r}") from exc
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Expected [pK, pAC50] for smiles={smiles!r}, target={target!r}; got {value!r}")
    return float(value[1])


def transform_pac50(value: float, transform: str, offset: float, pac50_floor: float) -> float:
    if transform == "log10_pAC50":
        value = max(value, pac50_floor)
        return float(np.log10(value))
    if transform == "offset_minus_pAC50":
        return float(offset - value)
    if transform == "raw_pAC50":
        return float(value)
    raise ValueError(f"Unknown transform: {transform}")


def rebuild_x(
    df: pd.DataFrame,
    predictions: dict,
    target_order: list[str],
    transform: str,
    transform_offset: float,
    pac50_floor: float,
) -> pd.DataFrame:
    out = df[["Drug", "smiles", "cmax_free(uM)"]].copy()
    missing_smiles = sorted(set(out["smiles"]) - set(predictions))
    if missing_smiles:
        preview = ", ".join(missing_smiles[:5])
        raise KeyError(f"{len(missing_smiles)} SMILES are missing in PreMOTA JSON. Examples: {preview}")

    cmax = out["cmax_free(uM)"].to_numpy(float)
    feature_matrix = np.empty((len(out), len(target_order)), dtype=float)
    for row_idx, smiles in enumerate(out["smiles"].tolist()):
        for col_idx, target in enumerate(target_order):
            raw_pac50 = p_ac50_for(predictions, smiles, target)
            feature_matrix[row_idx, col_idx] = transform_pac50(
                raw_pac50,
                transform,
                transform_offset,
                pac50_floor,
            )

    feature_df = pd.DataFrame(feature_matrix, columns=target_order)
    inter_df = feature_df.mul(cmax, axis=0)
    inter_df.columns = [f"{target}*Cmax" for target in target_order]
    return pd.concat([out, feature_df, inter_df], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dataset-dir", required=True, help="Existing AC50_Cmax split dir.")
    parser.add_argument("--premota-json", required=True, help="New PreMOTA JSON with SMILES -> target -> [pK,pAC50].")
    parser.add_argument("--out-dataset-dir", required=True, help="Output AC50_Cmax split dir.")
    parser.add_argument(
        "--transform",
        choices=["log10_pAC50", "offset_minus_pAC50", "raw_pAC50"],
        default="offset_minus_pAC50",
        help="Off-target feature transform. Published notebooks use offset_minus_pAC50.",
    )
    parser.add_argument("--transform-offset", type=float, default=6.0, help="Offset for offset_minus_pAC50.")
    parser.add_argument(
        "--pac50-floor",
        type=float,
        default=1.0,
        help="Floor used before log10_pAC50 so rare non-positive predictions do not create NaN values.",
    )
    args = parser.parse_args()

    old_dir = Path(args.old_dataset_dir)
    out_dir = Path(args.out_dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_premota_json(Path(args.premota_json))

    first_x = pd.read_csv(old_dir / "train_X.csv", nrows=1)
    target_order = list(first_x.columns[3:197])
    adr_task_order = list(pd.read_csv(old_dir / "train_y.csv", nrows=1).columns[1:])

    metadata = {
        "source_dataset_dir": str(old_dir),
        "premota_json": str(Path(args.premota_json)),
        "feature_transform": {
            "name": args.transform,
            "offset": args.transform_offset,
            "pac50_floor": args.pac50_floor,
            "formula": "feature = transform(raw_pAC50); interaction = feature * existing cmax_free(uM)",
        },
        "target_order": target_order,
        "adr_task_order": adr_task_order,
    }

    for split in SPLITS:
        x_path = old_dir / f"{split}_X.csv"
        y_path = old_dir / f"{split}_y.csv"
        x_df = pd.read_csv(x_path)
        rebuilt = rebuild_x(
            x_df,
            predictions,
            target_order,
            args.transform,
            args.transform_offset,
            args.pac50_floor,
        )
        rebuilt.to_csv(out_dir / f"{split}_X.csv", index=False)
        shutil.copy2(y_path, out_dir / f"{split}_y.csv")

    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote rebuilt HetSia dataset to {out_dir}")
    print(f"Targets: {len(target_order)}; ADR tasks: {len(adr_task_order)}")


if __name__ == "__main__":
    main()
