#!/usr/bin/env python3
"""Build dose-study HetSia X files from PreMOTA JSON and existing Cmax files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def transform_pac50(value: float, transform: str, floor: float) -> float:
    if transform == "log10_pAC50":
        return float(np.log10(max(value, floor)))
    if transform == "offset_minus_pAC50":
        return float(6.0 - value)
    if transform == "raw_pAC50":
        return float(value)
    raise ValueError(f"Unknown transform: {transform}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-info-csv", required=True)
    parser.add_argument("--cmax-dir", required=True)
    parser.add_argument("--premota-json", required=True)
    parser.add_argument("--target-order-csv", required=True, help="A HetSia X CSV whose target columns define order.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--doses", default="5mg,10mg,25mg,50mg,75mg,100mg,125mg,150mg,200mg,250mg,300mg,350mg")
    parser.add_argument("--log10-cmax", action="store_true", help="Use log10(cmax_free(uM)) before interactions.")
    parser.add_argument(
        "--target-transform",
        choices=["log10_pAC50", "offset_minus_pAC50", "raw_pAC50"],
        default="offset_minus_pAC50",
        help="Published notebooks use offset_minus_pAC50, equivalent to log10(AC50_uM).",
    )
    parser.add_argument("--pac50-floor", type=float, default=1.0)
    args = parser.parse_args()

    drug_info = pd.read_csv(args.drug_info_csv)
    if "canonical_smiles" in drug_info.columns:
        smiles_col = "canonical_smiles"
    elif "smiles" in drug_info.columns:
        smiles_col = "smiles"
    else:
        raise ValueError("drug info must contain canonical_smiles or smiles")
    name_col = "drug_name" if "drug_name" in drug_info.columns else "Drug"
    smiles_to_name = dict(zip(drug_info[smiles_col].astype(str), drug_info[name_col].astype(str)))

    target_order_df = pd.read_csv(args.target_order_csv, nrows=1)
    cols = list(target_order_df.columns)
    target_cols = cols[3:197]
    if len(target_cols) != 194:
        raise ValueError(f"Expected 194 target columns from {args.target_order_csv}, found {len(target_cols)}")

    premota = json.loads(Path(args.premota_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dose in [item.strip() for item in args.doses.split(",") if item.strip()]:
        cmax_path = Path(args.cmax_dir) / f"drug_cmax_predict_{dose}.csv"
        cmax_df = pd.read_csv(cmax_path)
        if "smiles" not in cmax_df.columns or "cmax_free(uM)" not in cmax_df.columns:
            raise ValueError(f"{cmax_path} must contain smiles and cmax_free(uM)")
        cmax_df = cmax_df[cmax_df["cmax_free(uM)"] > 0].copy()

        rows = []
        missing = []
        for _, cmax_row in cmax_df.iterrows():
            smiles = str(cmax_row["smiles"])
            targets = premota.get(smiles)
            if targets is None:
                missing.append(smiles)
                continue
            cmax_free_raw = float(cmax_row["cmax_free(uM)"])
            cmax_free = float(np.log10(cmax_free_raw)) if args.log10_cmax else cmax_free_raw
            row = {
                "Drug": smiles_to_name.get(smiles, cmax_row.get("Drug", "")),
                "smiles": smiles,
                "cmax_free(uM)": cmax_free,
            }
            for target in target_cols:
                row[target] = transform_pac50(
                    float(targets[target][1]),
                    args.target_transform,
                    args.pac50_floor,
                )
            for target in target_cols:
                row[f"{target}*Cmax"] = row[target] * cmax_free
            rows.append(row)

        if missing:
            sample = missing[:5]
            raise KeyError(f"{dose}: missing {len(missing)} SMILES in PreMOTA JSON, examples: {sample}")
        out_df = pd.DataFrame(rows)
        out_df = out_df[["Drug", "smiles", "cmax_free(uM)"] + target_cols + [f"{t}*Cmax" for t in target_cols]]
        out_path = out_dir / f"predict_X_{dose}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}: {out_df.shape}")


if __name__ == "__main__":
    main()
