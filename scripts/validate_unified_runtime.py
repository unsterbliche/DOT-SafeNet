#!/usr/bin/env python3
"""Run one-molecule numerical validation in the shared model environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    site_packages = Path(sys.prefix) / "lib" / "python3.9" / "site-packages"
    candidates = [
        site_packages / "nvidia/cudnn/lib",
        site_packages / "nvidia/cublas/lib",
        site_packages / "nvidia/cuda_runtime/lib",
        Path("/usr/local/cuda-11.8/targets/x86_64-linux/lib"),
    ]
    existing = [str(path) for path in candidates if path.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        existing.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(existing)
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    return env


def framework_probe(module: str, expression: str, env: dict[str, str]) -> dict:
    code = f"import json,{module}; print(json.dumps({expression}))"
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True,
        text=True, env=env,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def maximum_difference(new_path: Path, reference_path: Path, keys: list[str], columns: list[str]) -> dict[str, float]:
    new = pd.read_csv(new_path)
    reference = pd.read_csv(reference_path)
    if keys:
        merged = new.merge(reference[keys + columns], on=keys, suffixes=("_new", "_reference"))
        if len(merged) != len(reference):
            raise ValueError(f"Row-key mismatch for {new_path.name}: {len(merged)} versus {len(reference)}")
        return {
            column: float(np.nanmax(np.abs(merged[f"{column}_new"] - merged[f"{column}_reference"])))
            for column in columns
        }
    return {column: float(abs(new.loc[0, column] - reference.loc[0, column])) for column in columns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = runtime_environment()
    package_names = [
        "numpy", "pandas", "scipy", "scikit-learn", "rdkit-pypi",
        "torch", "torch-geometric", "tensorflow",
    ]
    versions = {name: importlib.metadata.version(name) for name in package_names}
    torch_probe = framework_probe(
        "torch",
        "{'version': torch.__version__, 'cuda': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}",
        env,
    )
    tensorflow_probe = framework_probe(
        "tensorflow as tf",
        "{'version': tf.__version__, 'gpu_count': len(tf.config.list_physical_devices('GPU'))}",
        env,
    )

    inputs = pd.read_csv(ROOT / "data/examples/paper_case_inputs.csv")
    citalopram = inputs.loc[inputs["name"].str.casefold().eq("citalopram")].copy()
    if len(citalopram) != 1:
        raise ValueError("Expected one deposited Citalopram input")
    citalopram["compound_id"] = "SMOKE_CITALOPRAM_20"
    input_path = args.output_dir / "citalopram_input.csv"
    citalopram.to_csv(input_path, index=False)
    prediction_dir = args.output_dir / "citalopram_prediction"
    log_path = args.output_dir / "inference.log"
    command = [
        sys.executable, str(ROOT / "scripts/profile_molecule.py"), "analyze",
        "--input", str(input_path), "--output", str(prediction_dir),
        "--checkpoint-root", str(args.checkpoint_root.resolve()),
        "--asset-root", str(args.asset_root.resolve()),
        "--python", sys.executable, "--device", args.device,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)

    reference = ROOT / "data/examples/citalopram_full_inference"
    differences = {
        "exposure": maximum_difference(
            prediction_dir / "exposure_predictions.csv", reference / "exposure_predictions.csv", [],
            ["pred_ppb_mean", "pred_fu_mean", "predicted_log10_cmax_ug_ml_mean", "log10_free_cmax_uM"],
        ),
        "adr": maximum_difference(
            prediction_dir / "adr_predictions.csv", reference / "adr_predictions.csv", ["soc_code"],
            ["probability_mean", "probability_sd", "probability_min", "probability_max", "background_percentile"],
        ),
        "margins": maximum_difference(
            prediction_dir / "target_margins.csv", reference / "target_margins.csv", ["target_uniprot"],
            ["predicted_pAC50", "margin_log10_AC50_over_free_Cmax"],
        ),
        "attribution": maximum_difference(
            prediction_dir / "target_attribution.csv", reference / "target_attribution.csv",
            ["soc_code", "target_uniprot"], ["delta_probability", "delta_probability_sd"],
        ),
    }
    limits = {"continuous": 2e-5, "background_percentile": 1e-3}
    failures = []
    for group, values in differences.items():
        for column, value in values.items():
            limit = limits["background_percentile"] if column == "background_percentile" else limits["continuous"]
            if value > limit:
                failures.append(f"{group}.{column}: {value:.6g} > {limit:.6g}")
    if args.device.startswith("cuda") and (not torch_probe["cuda"] or tensorflow_probe["gpu_count"] < 1):
        failures.append("CUDA was requested but one or more frameworks did not detect a GPU")

    report = {
        "status": "passed" if not failures else "failed",
        "python": sys.version,
        "packages": versions,
        "torch": torch_probe,
        "tensorflow": tensorflow_probe,
        "input": "Citalopram 20 mg/day",
        "targets": len(pd.read_csv(prediction_dir / "target_margins.csv")),
        "soc_tasks": len(pd.read_csv(prediction_dir / "adr_predictions.csv")),
        "attribution_rows": len(pd.read_csv(prediction_dir / "target_attribution.csv")),
        "maximum_absolute_differences": differences,
        "limits": limits,
        "failures": failures,
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
