"""Prepare panel-specific source-data files from the audited OT-ProfileNet archive."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "09_results" / "manuscript_otprofilenet_source_package_20260804"
DERIVED = SOURCE_ROOT / "derived_source_data"
AUDIT_SOURCE = SOURCE_ROOT / "audit"


def write_csv(frame: pd.DataFrame, relative_path: str) -> Path:
    path = PACKAGE_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def copy_file(source: Path, relative_path: str) -> Path:
    destination = PACKAGE_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def prepare_figure_2() -> None:
    family = pd.read_csv(DERIVED / "figure2_ab_target_family_statistics.csv")
    regression = pd.read_csv(DERIVED / "regression_predictions_row_level.csv")
    benchmark = pd.read_csv(DERIVED / "supplementary_table1_reported.csv")
    fine_tuned = pd.read_csv(DERIVED / "regression_family_mean_metrics.csv")
    herg = pd.read_csv(DERIVED / "herg_external7_predictions.csv")

    write_csv(family[["target_family", "target_count"]], "figure_2/data/panel_a_target_counts.csv")
    write_csv(family, "figure_2/data/panel_b_dataset_statistics.csv")
    selected = ["GPCR", "Ion channel", "Enzyme", "Kinases", "NR", "Transporter"]
    panel_c = regression.loc[
        regression["endpoint"].eq("pAC50") & regression["target_family"].isin(selected)
    ].copy()
    write_csv(panel_c, "figure_2/data/panel_c_pac50_predictions.csv")

    family_name = {"Kinase": "Kinases"}
    for row in fine_tuned.itertuples(index=False):
        target = family_name.get(row.target_family, row.target_family)
        mask = benchmark["model"].eq("PreMOTA (fine-tuned)") & benchmark[
            "target_family"
        ].eq(target)
        benchmark.loc[mask, ["RMSE", "PCC", "Spearman", "R2"]] = [
            row.RMSE,
            row.PCC,
            row.Spearman,
            row.R2,
        ]
    write_csv(benchmark, "figure_2/data/panel_d_model_comparison.csv")
    write_csv(herg, "figure_2/data/panel_e_f_herg_external_validation.csv")


def prepare_supplementary_figure_2() -> None:
    lengths = pd.read_csv(DERIVED / "suppfig2a_protein_sequence_lengths.csv")
    summary = pd.read_csv(DERIVED / "suppfig2cd_pretraining_summary.csv")
    write_csv(lengths, "supplementary_figure_2/data/panel_a_protein_sequence_lengths.csv")
    write_csv(
        summary.loc[summary["measure"].isin(["positive_pairs", "negative_pairs"])],
        "supplementary_figure_2/data/panel_c_class_balance.csv",
    )
    write_csv(summary, "supplementary_figure_2/data/panel_d_dataset_summary.csv")

    expected = PACKAGE_ROOT / "supplementary_figure_2" / "data" / "panel_b_compounds_per_target.csv"
    optional = os.environ.get("SFIG2B_SOURCE")
    if optional:
        source = Path(optional)
        frame = pd.read_csv(source)
        if not {"target_id", "compound_count"}.issubset(frame.columns):
            raise ValueError("SFIG2B_SOURCE must contain target_id and compound_count")
        write_csv(frame[["target_id", "compound_count"]], str(expected.relative_to(PACKAGE_ROOT)))
    elif not expected.exists():
        write_csv(
            pd.DataFrame(columns=["target_id", "compound_count"]),
            "supplementary_figure_2/data/panel_b_expected_schema.csv",
        )


def prepare_supplementary_figure_3() -> None:
    loss = pd.read_csv(DERIVED / "suppfig3_pretraining_loss.csv")
    metrics = pd.read_csv(DERIVED / "suppfig3_classification_metrics.csv")
    write_csv(loss, "supplementary_figure_3/data/panel_a_pretraining_loss.csv")
    for panel, metric in [("b", "AUROC"), ("c", "F1"), ("d", "MCC")]:
        write_csv(
            metrics[["target", "model", metric]],
            "supplementary_figure_3/data/panel_{}_{}_metrics.csv".format(
                panel, metric.lower()
            ),
        )


def prepare_supplementary_figure_4() -> None:
    regression = pd.read_csv(DERIVED / "regression_predictions_row_level.csv")
    family_panels = [
        ("a", "GPCR"),
        ("b", "Ion channel"),
        ("c", "Enzyme"),
        ("d", "Kinases"),
        ("e", "NR"),
        ("f", "Transporter"),
    ]
    for panel, family in family_panels:
        frame = regression.loc[
            regression["endpoint"].eq("pK") & regression["target_family"].eq(family)
        ].copy()
        write_csv(
            frame,
            "supplementary_figure_4/data/panel_{}_{}_pk_predictions.csv".format(
                panel, family.lower().replace(" ", "_")
            ),
        )


def prepare_audit_files() -> None:
    for name in [
        "consistency_report.md",
        "figure_table_source_manifest.csv",
        "supplementary_table1_consistency_audit.csv",
        "supplementary_table12_hyperparameter_audit.csv",
        "supplementary_table13_hyperparameter_audit.csv",
    ]:
        copy_file(AUDIT_SOURCE / name, "qa/" + name)


def write_checksums() -> None:
    records = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.name == "sha256_checksums.csv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": path.relative_to(PACKAGE_ROOT).as_posix(), "sha256": digest})
    write_csv(pd.DataFrame(records), "qa/sha256_checksums.csv")


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError("Audited source package is unavailable: {}".format(SOURCE_ROOT))
    prepare_figure_2()
    prepare_supplementary_figure_2()
    prepare_supplementary_figure_3()
    prepare_supplementary_figure_4()
    prepare_audit_files()
    write_checksums()
    print("Prepared panel-specific source data under {}".format(PACKAGE_ROOT))


if __name__ == "__main__":
    main()



