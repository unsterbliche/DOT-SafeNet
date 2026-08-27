#!/usr/bin/env python3
"""Validate the public DOT-SafeNet code, data contracts, figures and weights."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = [f"Figure_{i}.tiff" for i in range(1, 7)] + [
    f"Supplementary_Figure_{i}.tiff" for i in range(1, 9)
]
FIGURE_ENTRYPOINTS = [
    f"reproduce/figure_{i}/run.py" for i in range(1, 7)
] + [
    f"reproduce/supplementary_figure_{i}/run.py" for i in (2, 3, 4, 6, 7, 8)
]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def validate_package(check_hashes: bool = False) -> dict:
    # check_hashes is retained for compatibility with earlier test commands.
    errors: list[str] = []
    required = [
        "README.md", "LICENSE", "CITATION.cff", "requirements-figures.txt",
        "requirements-unified.txt",
        "configs/pipeline.yaml", "configs/safety_profiler.yaml",
        "data/model/target_order.csv", "data/model/soc_order.csv",
        "data/test/dotsafenet_test_X.csv", "data/test/dotsafenet_test_y.csv",
        "data/reference/target_gene_map.csv", "data/reference/target_feature_reference.csv",
        "data/reference/soc_target_evidence_matrix.csv",
        "data/reference/soc_background_quantiles.csv.gz",
        "data/examples/paper_case_inputs.csv", "data/examples/paper_case_soc_reference.csv",
        "scripts/profile_molecule.py", "scripts/create_unified_environment.sh",
        "scripts/run_unified_profile.sh",
        "scripts/validate_unified_runtime.py", "scripts/render_all_figures.py",
        "scripts/reproduce_metrics.py", "scripts/check_weights.py",
        "scripts/run_tests.py",
        "skills/dotsafenet-paper-reproducer/SKILL.md",
        "skills/dotsafenet-safety-profiler/SKILL.md",
        "weights/weights_manifest.tsv",
    ]
    errors.extend(f"missing: {name}" for name in required if not (ROOT / name).is_file())
    errors.extend(f"missing: {name}" for name in FIGURE_ENTRYPOINTS if not (ROOT / name).is_file())
    errors.extend(f"missing: figures/{name}" for name in FIGURE_NAMES if not (ROOT / "figures" / name).is_file())
    if errors:
        return {"status": "failed", "errors": errors, "warnings": []}

    figure_requirements = {
        line.split(";", 1)[0].strip().lower()
        for line in (ROOT / "requirements-figures.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not any(requirement.startswith("rdkit") for requirement in figure_requirements):
        errors.append("requirements-figures.txt does not include RDKit required by Supplementary Figure 6")

    targets = read_csv(ROOT / "data/model/target_order.csv")
    socs = read_csv(ROOT / "data/model/soc_order.csv")
    x_header = header(ROOT / "data/test/dotsafenet_test_X.csv")
    y_header = header(ROOT / "data/test/dotsafenet_test_y.csv")
    if len(targets) != 194:
        errors.append(f"target count: {len(targets)}")
    if len(socs) != 18:
        errors.append(f"SOC count: {len(socs)}")
    if len(x_header) != 391:
        errors.append(f"test feature columns: {len(x_header)}")
    if len(y_header) != 19:
        errors.append(f"test label columns: {len(y_header)}")

    activity = [row["activity_feature"] for row in targets]
    interaction = [row["interaction_feature"] for row in targets]
    if x_header[3:197] != activity or x_header[197:] != interaction:
        errors.append("DOT-SafeNet test feature order differs from target_order.csv")
    if y_header[1:] != [row["soc_name"] for row in socs]:
        errors.append("DOT-SafeNet test label order differs from soc_order.csv")

    target_ids = [row["uniprot_accession"] for row in targets]
    soc_codes = [row["soc_code"] for row in socs]
    gene_map = read_csv(ROOT / "data/reference/target_gene_map.csv")
    target_reference = read_csv(ROOT / "data/reference/target_feature_reference.csv")
    evidence = read_csv(ROOT / "data/reference/soc_target_evidence_matrix.csv")
    examples = read_csv(ROOT / "data/examples/paper_case_inputs.csv")
    example_soc = read_csv(ROOT / "data/examples/paper_case_soc_reference.csv")
    if [row["target_uniprot"] for row in gene_map] != target_ids:
        errors.append("target_gene_map.csv differs from the fixed target order")
    if [row["target_uniprot"] for row in target_reference] != target_ids:
        errors.append("target_feature_reference.csv differs from the fixed target order")
    if [row["soc_code"] for row in evidence] != soc_codes:
        errors.append("soc_target_evidence_matrix.csv differs from the fixed SOC order")
    if list(evidence[0].keys())[1:] != target_ids:
        errors.append("soc_target_evidence_matrix.csv differs from the fixed target order")
    if len(examples) != 4 or len(example_soc) != 4 * 18:
        errors.append(f"paper examples: inputs={len(examples)}, SOC rows={len(example_soc)}")

    weights = read_csv(ROOT / "weights/weights_manifest.tsv", delimiter="\t")
    counts = Counter(row["model"] for row in weights)
    expected_weights = {"OT-ProfileNet": 7, "PlasmaBindNet-Fu": 5, "DoseExpoNet": 5, "DOT-SafeNet": 20}
    if dict(counts) != expected_weights:
        errors.append(f"weight manifest members: {dict(counts)}")
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    if any(not sha_pattern.fullmatch(row["sha256"]) for row in weights):
        errors.append("invalid checkpoint SHA256 value")

    return {
        "status": "failed" if errors else "passed",
        "targets": len(targets),
        "soc_tasks": len(socs),
        "manuscript_figures": len(FIGURE_NAMES),
        "code_figure_entrypoints": len(FIGURE_ENTRYPOINTS),
        "weight_files": len(weights),
        "profiler_examples": len(examples),
        "errors": errors,
        "warnings": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figure-hashes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = validate_package()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
