#!/usr/bin/env python3
"""Rebuild HetSia split X files with new PPB-derived log10 free Cmax."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--old-cmax-csv", required=True, help="MotifAttnNet ADR cmax file containing old ppb/cmax/MW, usually 100mg.")
    parser.add_argument("--new-ppb-csv", required=True)
    parser.add_argument("--out-dataset-dir", required=True)
    parser.add_argument("--new-ppb-col", default="pred_ppb_mean")
    parser.add_argument("--cmax-col", default="cmax(ug/ml)")
    parser.add_argument("--mw-col", default="molecular_weight")
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    base_dir = Path(args.base_dataset_dir)
    out_dir = Path(args.out_dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cmax = pd.read_csv(args.old_cmax_csv)
    new_ppb = pd.read_csv(args.new_ppb_csv)
    required_old = {"smiles", args.cmax_col, args.mw_col}
    missing_old = required_old - set(old_cmax.columns)
    if missing_old:
        raise ValueError(f"{args.old_cmax_csv} missing columns: {sorted(missing_old)}")
    if "smiles" not in new_ppb.columns or args.new_ppb_col not in new_ppb.columns:
        raise ValueError(f"{args.new_ppb_csv} must contain smiles and {args.new_ppb_col}")

    exposure = old_cmax[["smiles", args.cmax_col, args.mw_col]].drop_duplicates("smiles")
    exposure = exposure.merge(new_ppb[["smiles", args.new_ppb_col]], on="smiles", how="left")
    if exposure[args.new_ppb_col].isna().any():
        examples = exposure.loc[exposure[args.new_ppb_col].isna(), "smiles"].head().tolist()
        raise KeyError(f"Missing new PPB predictions for {exposure[args.new_ppb_col].isna().sum()} SMILES: {examples}")

    exposure["new_fu"] = np.clip(1.0 - 0.999 * exposure[args.new_ppb_col].to_numpy(float), args.eps, None)
    exposure["new_cmax_free_uM_raw"] = (
        exposure[args.cmax_col].to_numpy(float) * exposure["new_fu"].to_numpy(float) / exposure[args.mw_col].to_numpy(float) * 1000.0
    )
    exposure["new_log10_cmax_free_uM"] = np.log10(np.clip(exposure["new_cmax_free_uM_raw"].to_numpy(float), args.eps, None))
    exposure_map = exposure.set_index("smiles")["new_log10_cmax_free_uM"].to_dict()

    diagnostics = []
    for split in SPLITS:
        x_path = base_dir / f"{split}_X.csv"
        y_path = base_dir / f"{split}_y.csv"
        x_df = pd.read_csv(x_path)
        target_cols = list(x_df.columns[3:197])
        missing = sorted(set(x_df["smiles"].astype(str)) - set(exposure_map))
        if missing:
            raise KeyError(f"{split}: missing exposure for {len(missing)} SMILES, examples: {missing[:5]}")

        old_log = x_df["cmax_free(uM)"].to_numpy(float)
        new_log = x_df["smiles"].astype(str).map(exposure_map).to_numpy(float)
        out = x_df.copy()
        out["cmax_free(uM)"] = new_log
        for target in target_cols:
            out[f"{target}*Cmax"] = out[target].to_numpy(float) * new_log
        out.to_csv(out_dir / f"{split}_X.csv", index=False)
        shutil.copy2(y_path, out_dir / f"{split}_y.csv")

        diagnostics.append(
            {
                "split": split,
                "n": int(len(out)),
                "old_log_cmax_free_mean": float(np.mean(old_log)),
                "new_log_cmax_free_mean": float(np.mean(new_log)),
                "delta_log_cmax_free_mean": float(np.mean(new_log - old_log)),
                "old_log_cmax_free_std": float(np.std(old_log)),
                "new_log_cmax_free_std": float(np.std(new_log)),
            }
        )

    pd.DataFrame(diagnostics).to_csv(out_dir / "new_ppb_exposure_diagnostics.csv", index=False)
    metadata = {
        "base_dataset_dir": str(base_dir),
        "old_cmax_csv": str(args.old_cmax_csv),
        "new_ppb_csv": str(args.new_ppb_csv),
        "new_ppb_col": args.new_ppb_col,
        "formula": "new_log10_cmax_free_uM = log10(old_cmax_ug_ml * (1 - 0.999 * new_ppb) / MW * 1000)",
        "note": "Column name cmax_free(uM) is preserved for HetSia compatibility, but values are log10(cmax_free_uM).",
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote rebuilt HetSia dataset to {out_dir}")


if __name__ == "__main__":
    main()
