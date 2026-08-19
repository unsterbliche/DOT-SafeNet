#!/usr/bin/env python3
"""Build compact public reference tables used by the safety-profiler skill."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]

CASES = [
    {"compound_id": "PAPER_MECLOFENAMIC_150", "name": "Meclofenamic acid", "drug_key": "SIDER767", "dose_mg_day": 150.0, "route": "oral"},
    {"compound_id": "PAPER_CITALOPRAM_20", "name": "Citalopram", "drug_key": "SIDER433", "dose_mg_day": 20.0, "route": "oral"},
    {"compound_id": "PAPER_SPIRONOLACTONE_25", "name": "Spironolactone", "drug_key": "spironolactone", "dose_mg_day": 25.0, "route": "oral"},
    {"compound_id": "PAPER_CANDESARTAN_CILEXETIL_32", "name": "Candesartan cilexetil", "drug_key": "SIDER372", "dose_mg_day": 32.0, "route": "oral"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-x", type=Path, help="Original DOT-SafeNet training X used to calculate target references")
    args = parser.parse_args()
    reference = ROOT / "data/reference"
    examples = ROOT / "data/examples"
    reference.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)

    target_order = pd.read_csv(ROOT / "data/manifests/target_order.csv")
    target_ids = target_order["uniprot_accession"].astype(str).tolist()
    soc = pd.read_csv(ROOT / "data/manifests/soc_order.csv")
    soc_name_to_code = dict(zip(soc["soc_name"], soc["soc_code"]))

    gene_source = PROJECT / "09_results/adr_film_exposure_clinical/soc_target_network_20260712/protein_to_human_gene_map.csv"
    gene = pd.read_csv(gene_source)[["target_uniprot", "human_gene"]].drop_duplicates("target_uniprot")
    gene.set_index("target_uniprot").loc[target_ids].reset_index().to_csv(reference / "target_gene_map.csv", index=False)

    evidence_source = ROOT / "figures/exposure_aware/figure_5/data/figure5a_module_ordered_evidence_matrix.csv"
    evidence = pd.read_csv(evidence_source, index_col=0).reindex(index=soc["soc_code"], columns=target_ids).fillna(0).astype(int)
    evidence.to_csv(reference / "soc_target_evidence_matrix.csv")

    grid_source = PROJECT / "09_results/adr_film_exposure_clinical/adr_score_background_ctade_dev_20260714/soc_background_quantile_grid.csv.gz"
    grid = pd.read_csv(grid_source)
    grid = grid[(grid["label_subset"] == "all") & (grid["score_source"] == "ensemble_mean_probability")]
    grid.to_csv(reference / "soc_background_quantiles.csv.gz", index=False, compression="gzip")

    x = pd.read_csv(ROOT / "data/dotsafenet_test/test_X.csv")
    ref_x = pd.read_csv(args.training_x) if args.training_x else x
    target_ref = pd.DataFrame({
        "target_uniprot": target_ids,
        "training_median": ref_x[target_ids].median(axis=0).to_numpy(float),
        "training_q90": ref_x[target_ids].quantile(0.90, axis=0).to_numpy(float),
        "reference_source": "training_X" if args.training_x else "held_out_test_X_temporary",
    })
    target_ref.to_csv(reference / "target_feature_reference.csv", index=False)

    input_rows = []
    exposure_rows = []
    margin_rows = []
    key_to_case = {}
    for case in CASES:
        matched = x[x["Drug"].astype(str).str.casefold() == case["drug_key"].casefold()]
        if len(matched) != 1:
            raise ValueError(f"Expected one feature row for {case['drug_key']}, found {len(matched)}")
        row = matched.iloc[0]
        key_to_case[str(row["smiles"])] = case
        input_rows.append({k: case[k] for k in ["compound_id", "name", "dose_mg_day", "route"]} | {"smiles": row["smiles"]})
        free_cmax = 10.0 ** float(row["cmax_free(uM)"])
        exposure_rows.append({
            "compound_id": case["compound_id"], "name": case["name"], "smiles": row["smiles"],
            "dose_mg_day": case["dose_mg_day"], "route": case["route"], "pred_ppb_mean": np.nan,
            "predicted_cmax_ug_ml_mean": np.nan, "free_cmax_uM": free_cmax,
            "log10_free_cmax_uM": float(row["cmax_free(uM)"]),
        })
        for target in target_ids:
            feature = float(row[target])
            margin_rows.append({
                "compound_id": case["compound_id"], "name": case["name"], "dose_mg_day": case["dose_mg_day"],
                "target_uniprot": target, "predicted_pAC50": 6.0 - feature,
                "predicted_AC50_uM": 10.0 ** feature, "free_cmax_uM": free_cmax,
                "margin_log10_AC50_over_free_Cmax": feature - float(row["cmax_free(uM)"]),
            })
    pd.DataFrame(input_rows).to_csv(examples / "paper_case_inputs.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(examples / "paper_case_exposure_reference.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(examples / "paper_case_margin_reference.csv.gz", index=False, compression="gzip")

    fold_frames = []
    for fold in range(1, 6):
        path = ROOT / f"figures/exposure_aware/figure_4/data/clinical_test_57/fold_{fold}_clinical_test_before_after_57.csv"
        frame = pd.read_csv(path)
        frame["fold"] = fold
        fold_frames.append(frame)
    predictions = pd.concat(fold_frames, ignore_index=True)
    predictions = predictions[predictions["smiles"].astype(str).isin(key_to_case)].copy()
    predictions["compound_id"] = predictions["smiles"].map(lambda value: key_to_case[str(value)]["compound_id"])
    predictions["name"] = predictions["smiles"].map(lambda value: key_to_case[str(value)]["name"])
    predictions["soc_code"] = predictions["task_name"].map(soc_name_to_code)
    summary = predictions.groupby(["compound_id", "name", "smiles", "soc_code", "task_name"], as_index=False).agg(
        probability_mean=("after_pred", "mean"), probability_sd=("after_pred", "std"),
        probability_min=("after_pred", "min"), probability_max=("after_pred", "max"), label_true=("label_true", "first"),
    ).rename(columns={"task_name": "soc_name"})
    summary.to_csv(examples / "paper_case_soc_reference.csv", index=False)

    attribution_source = pd.read_csv(ROOT / "figures/exposure_aware/figure_5/data/selected_case_target_occlusion_robust_summary.csv")
    display_to_case = {case["name"].casefold(): case for case in CASES}
    attribution_source["case_key"] = attribution_source["display_drug"].astype(str).str.casefold()
    selected_parts = []
    for key, case in display_to_case.items():
        subset = attribution_source[(attribution_source["case_key"] == key) & np.isclose(attribution_source["dose_mg_day"], case["dose_mg_day"])].copy()
        subset["compound_id"] = case["compound_id"]
        subset["name"] = case["name"]
        selected_parts.append(subset)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["soc_code"] = selected["task_name"].map(soc_name_to_code)
    selected = selected.rename(columns={
        "target_id": "target_uniprot", "robust_delta_mean": "delta_probability",
        "delta_sd_across_folds_and_backgrounds": "delta_probability_sd",
        "min_positive_fold_fraction": "positive_fold_fraction",
    })
    selected[["compound_id", "name", "soc_code", "target_uniprot", "delta_probability", "delta_probability_sd", "positive_fold_fraction"]].to_csv(
        examples / "paper_case_attribution_reference.csv.gz", index=False, compression="gzip"
    )
    print(f"Wrote profiler references to {reference} and {examples}")


if __name__ == "__main__":
    main()
