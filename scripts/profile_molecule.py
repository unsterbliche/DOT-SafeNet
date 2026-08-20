#!/usr/bin/env python3
"""Stable command-line entrypoint for prospective DOT-SafeNet profiling."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety_profiler.core import (  # noqa: E402
    assemble_dotsafenet_features,
    attach_target_annotations,
    empirical_percentile,
    load_contract,
    read_ot_profile,
    standardize_input_table,
    validate_feature_table,
)
from safety_profiler.report import render_report  # noqa: E402


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def checkpoint_files(checkpoint_root: Path, model: str, role: str) -> list[Path]:
    manifest = pd.read_csv(ROOT / "weights/weights_manifest.tsv", sep="\t")
    selected = manifest[(manifest["model"] == model) & (manifest["role"] == role)].sort_values("member")
    paths = [checkpoint_root / value for value in selected["path"].astype(str)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing deposited checkpoints:\n" + "\n".join(missing[:10]))
    return paths


def add_percentiles(adr: pd.DataFrame) -> pd.DataFrame:
    grid = pd.read_csv(ROOT / "data/reference/soc_background_quantiles.csv.gz")
    adr = adr.copy()
    adr["background_percentile"] = [
        empirical_percentile(score, grid, code)
        for score, code in zip(adr["probability_mean"], adr["soc_code"])
    ]
    return adr


def finish_report(output: Path, exposure: pd.DataFrame, margins: pd.DataFrame, adr: pd.DataFrame, attribution: pd.DataFrame, mode: str) -> None:
    gene = pd.read_csv(ROOT / "data/reference/target_gene_map.csv").rename(columns={"human_gene": "target_gene"})
    margins = margins.merge(gene, on="target_uniprot", how="left")
    attribution = attach_target_annotations(attribution, ROOT)
    adr = add_percentiles(adr)
    adr.to_csv(output / "adr_predictions.csv", index=False)
    margins.to_csv(output / "target_margins.csv", index=False)
    attribution.to_csv(output / "target_attribution.csv", index=False)
    metadata = {
        "mode": mode,
        "models": ["OT-ProfileNet", "PlasmaBindNet-Fu", "DoseExpoNet", "DOT-SafeNet clinical five-fold ensemble"],
        "feature_contract": "log10 free Cmax + 194 (6-pAC50) target features + 194 target-exposure products",
        "soc_tasks": 18,
        "target_count": 194,
        "score_interpretation": "Raw model score and within-SOC empirical development-background percentile",
    }
    render_report(output, metadata, exposure, adr, margins, attribution)


def replay(args: argparse.Namespace) -> None:
    source = pd.read_csv(ROOT / "data/examples/paper_case_inputs.csv")
    requested = pd.read_csv(args.input) if args.input else source
    ids = set(requested["compound_id"].astype(str))
    unknown = ids - set(source["compound_id"].astype(str))
    if unknown:
        raise ValueError(f"Replay mode accepts deposited sample IDs only: {sorted(unknown)}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    standardized = source[source["compound_id"].astype(str).isin(ids)].copy()
    exposure = pd.read_csv(ROOT / "data/examples/paper_case_exposure_reference.csv")
    exposure = exposure[exposure["compound_id"].astype(str).isin(ids)]
    adr = pd.read_csv(ROOT / "data/examples/paper_case_soc_reference.csv")
    adr = adr[adr["compound_id"].astype(str).isin(ids)]
    margins = pd.read_csv(ROOT / "data/examples/paper_case_margin_reference.csv.gz")
    margins = margins[margins["compound_id"].astype(str).isin(ids)]
    attribution = pd.read_csv(ROOT / "data/examples/paper_case_attribution_reference.csv.gz")
    attribution = attribution[attribution["compound_id"].astype(str).isin(ids)]
    standardized.to_csv(output / "input_standardized.csv", index=False)
    finish_report(output, exposure, margins, adr, attribution, "paper-case replay")
    print(f"Wrote sample reproduction to {output}")


def analyze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(ROOT)
    standardized = standardize_input_table(pd.read_csv(args.input), canonicalize=True)
    standardized.to_csv(output / "input_standardized.csv", index=False)
    checkpoint_root = args.checkpoint_root.resolve()
    asset_root = args.asset_root.resolve()
    shared_python = args.python or sys.executable
    ot_python = args.ot_python or shared_python
    exposure_python = args.exposure_python or shared_python
    adr_python = args.adr_python or shared_python

    ot_json = output / "ot_profile.json"
    run([
        ot_python,
        str(ROOT / "src/ot_profilenet/finetuning/predict_drugs.py"),
        "--data-root", str(asset_root / "PreMOTA/dataset_reg_multitask"),
        "--checkpoint-root", str(checkpoint_root / "PreMOTA/regression_multitask/model_fintune_save"),
        "--input-csv", str(output / "input_standardized.csv"),
        "--smiles-column", "smiles", "--output-json", str(ot_json), "--device", args.device,
    ], ROOT / "src/ot_profilenet/finetuning")

    ppb_files = checkpoint_files(checkpoint_root, "PlasmaBindNet-Fu", "ensemble_checkpoint")
    ppb_dirs = [path.parent.parent for path in ppb_files]
    ppb_csv = output / "ppb_predictions.csv"
    run([
        exposure_python,
        str(ROOT / "src/plasmabindnet_fu/predict_ppb_ensemble_for_smiles.py"),
        "--motif-root", str(ROOT / "src/plasmabindnet_fu"),
        "--input-csv", str(output / "input_standardized.csv"), "--smiles-col", "smiles",
        "--checkpoint-dirs", ",".join(map(str, ppb_dirs)), "--out-csv", str(ppb_csv),
        "--config-path", str(ROOT / "configs/exposure_model_architecture.json"),
        "--target", "logitfu", "--set-layer", "SetRep", "--device", args.device,
    ], ROOT / "src/plasmabindnet_fu")

    cmax_files = checkpoint_files(checkpoint_root, "DoseExpoNet", "ensemble_checkpoint")
    cmax_csv = output / "cmax_predictions.csv"
    run([
        exposure_python,
        str(ROOT / "src/safety_profiler/predict_cmax_ensemble.py"),
        "--model-root", str(ROOT / "src/dose_exponet"),
        "--input-csv", str(output / "input_standardized.csv"),
        "--checkpoint-files", ",".join(map(str, cmax_files)),
        "--config-json", str(ROOT / "configs/exposure_model_architecture.json"),
        "--output-csv", str(cmax_csv), "--device", args.device,
    ], ROOT / "src/dose_exponet")

    ot = read_ot_profile(ot_json, contract.target_ids)
    features, exposure, margins = assemble_dotsafenet_features(
        standardized, ot, pd.read_csv(ppb_csv), pd.read_csv(cmax_csv), contract
    )
    features.to_csv(output / "dotsafenet_features.csv", index=False)
    exposure.to_csv(output / "exposure_predictions.csv", index=False)
    margins.to_csv(output / "target_margins_unannotated.csv", index=False)

    adr_dir = output / "dotsafenet"
    run([
        adr_python,
        str(ROOT / "src/safety_profiler/predict_dotsafenet_ensemble.py"),
        "--release-root", str(ROOT), "--checkpoint-root", str(checkpoint_root),
        "--features-csv", str(output / "dotsafenet_features.csv"),
        "--target-reference-csv", str(ROOT / "data/reference/target_feature_reference.csv"),
        "--output-dir", str(adr_dir),
    ])
    shutil.copy2(adr_dir / "adr_predictions_by_fold.csv", output / "adr_predictions_by_fold.csv")
    finish_report(
        output, exposure, margins, pd.read_csv(adr_dir / "adr_predictions.csv"),
        pd.read_csv(adr_dir / "target_attribution.csv"), "prospective full-model inference",
    )
    print(f"Wrote prospective profile to {output}")


def validate(args: argparse.Namespace) -> None:
    contract = load_contract(ROOT)
    report = {"status": "passed", "targets": len(contract.target_ids), "soc_tasks": len(contract.soc_codes), "errors": []}
    required = [
        ROOT / "data/examples/paper_case_inputs.csv",
        ROOT / "data/reference/target_gene_map.csv",
        ROOT / "data/reference/target_feature_reference.csv",
        ROOT / "data/reference/soc_target_evidence_matrix.csv",
        ROOT / "data/reference/soc_background_quantiles.csv.gz",
    ]
    report["missing_files"] = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if report["missing_files"]:
        report["status"] = "failed"
        report["errors"].append("required profiler data are missing")
    if args.checkpoint_root:
        for model, role, expected in [
            ("OT-ProfileNet", "family_checkpoint", 7),
            ("PlasmaBindNet-Fu", "ensemble_checkpoint", 5),
            ("DoseExpoNet", "ensemble_checkpoint", 5),
        ]:
            paths = checkpoint_files(args.checkpoint_root.resolve(), model, role)
            if len(paths) != expected:
                report["status"] = "failed"
                report["errors"].append(f"{model}: expected {expected} checkpoints, found {len(paths)}")
        for role in ["clinical_index", "clinical_data"]:
            paths = checkpoint_files(args.checkpoint_root.resolve(), "DOT-SafeNet", role)
            if len(paths) != 5:
                report["status"] = "failed"
                report["errors"].append(f"DOT-SafeNet {role}: expected 5 files, found {len(paths)}")
    if args.asset_root:
        families = ["GPCR", "IonChannel", "Enzyme", "Kinase", "NHR", "Transporter", "Other"]
        protein_count = 0
        missing_assets = []
        for family in families:
            family_root = args.asset_root.resolve() / "PreMOTA/dataset_reg_multitask" / family
            protein_file = family_root / "proteins.txt"
            embedding_dir = family_root / "esm2"
            if not protein_file.exists() or not embedding_dir.is_dir():
                missing_assets.append(family)
                continue
            proteins = ast.literal_eval(protein_file.read_text(encoding="utf-8"))
            protein_count += len(proteins)
        report["ot_profile_target_count"] = protein_count
        report["missing_ot_profile_families"] = missing_assets
        if missing_assets or protein_count != 194:
            report["status"] = "failed"
            report["errors"].append("OT-ProfileNet inference assets are incomplete")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--checkpoint-root", type=Path)
    v.add_argument("--asset-root", type=Path)
    v.set_defaults(func=validate)
    r = sub.add_parser("replay")
    r.add_argument("--input", type=Path, help="Subset of deposited sample compound_id values")
    r.add_argument("--output", required=True, type=Path)
    r.set_defaults(func=replay)
    a = sub.add_parser("analyze")
    a.add_argument("--input", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    a.add_argument("--checkpoint-root", required=True, type=Path)
    a.add_argument("--asset-root", required=True, type=Path)
    a.add_argument(
        "--python",
        help="Shared Python interpreter for all model stages (default: current interpreter)",
    )
    a.add_argument("--ot-python", help="Optional OT-ProfileNet interpreter override")
    a.add_argument("--exposure-python", help="Optional exposure-model interpreter override")
    a.add_argument("--adr-python", help="Optional DOT-SafeNet interpreter override")
    a.add_argument("--device", default="cuda:0")
    a.set_defaults(func=analyze)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
