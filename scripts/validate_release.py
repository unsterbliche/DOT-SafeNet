#!/usr/bin/env python3
"""Validate the publication package, data contracts, figures, and weight manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_package(check_hashes: bool) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "README.md", "requirements-figures.txt", "configs/pipeline.yaml",
        "configs/safety_profiler.yaml",
        "data/manifests/target_order.csv", "data/manifests/soc_order.csv",
        "data/manifests/model_lineage.csv", "data/manifests/figure_manifest.csv",
        "data/manifests/final_figure_files.csv", "data/dotsafenet_test/test_X.csv",
        "data/dotsafenet_test/test_y.csv", "scripts/render_all_figures.py",
        "scripts/reproduce_metrics.py", "scripts/check_weights.py", "paper_figures/figure.pptx",
        "scripts/profile_molecule.py", "data/examples/paper_case_inputs.csv",
        "data/examples/paper_case_soc_reference.csv",
        "data/reference/target_gene_map.csv",
        "data/reference/target_feature_reference.csv",
        "data/reference/soc_target_evidence_matrix.csv",
        "data/reference/soc_background_quantiles.csv.gz",
        "skill/dotsafenet-safety-profiler/SKILL.md",
        "dist/dotsafenet-safety-profiler.skill.zip", "dist/SHA256SUMS.txt",
    ]
    errors.extend(f"missing: {name}" for name in required if not (ROOT / name).is_file())
    if errors:
        return {"status": "failed", "errors": errors, "warnings": warnings}

    targets = read_csv(ROOT / "data/manifests/target_order.csv")
    socs = read_csv(ROOT / "data/manifests/soc_order.csv")
    figures = read_csv(ROOT / "data/manifests/figure_manifest.csv")
    final_files = read_csv(ROOT / "data/manifests/final_figure_files.csv")
    x_header = header(ROOT / "data/dotsafenet_test/test_X.csv")
    y_header = header(ROOT / "data/dotsafenet_test/test_y.csv")

    if len(targets) != 194:
        errors.append(f"target count: {len(targets)}")
    if len(socs) != 18:
        errors.append(f"SOC count: {len(socs)}")
    if len(x_header) != 391:
        errors.append(f"test_X columns: {len(x_header)}")
    if len(y_header) != 19:
        errors.append(f"test_y columns: {len(y_header)}")
    if len(figures) != 14 or len(final_files) != 14:
        errors.append(f"figure count: manifest={len(figures)}, final={len(final_files)}")
    names = [row["figure"] for row in figures]
    expected = [f"Figure {i}" for i in range(1, 7)] + [f"Supplementary Figure {i}" for i in range(1, 9)]
    if names != expected:
        errors.append("figure order differs from the six main and eight supplementary figures")
    if len(set(names)) != len(names):
        errors.append("duplicate manuscript figure names")

    activity = [row["activity_feature"] for row in targets]
    interaction = [row["interaction_feature"] for row in targets]
    if x_header[3:197] != activity or x_header[197:] != interaction:
        errors.append("DOT-SafeNet test_X feature order differs from target_order.csv")
    if y_header[1:] != [row["soc_name"] for row in socs]:
        errors.append("DOT-SafeNet test_y order differs from soc_order.csv")

    target_ids = [row["uniprot_accession"] for row in targets]
    soc_codes = [row["soc_code"] for row in socs]
    gene_map = read_csv(ROOT / "data/reference/target_gene_map.csv")
    target_reference = read_csv(ROOT / "data/reference/target_feature_reference.csv")
    evidence = read_csv(ROOT / "data/reference/soc_target_evidence_matrix.csv")
    examples = read_csv(ROOT / "data/examples/paper_case_inputs.csv")
    example_soc = read_csv(ROOT / "data/examples/paper_case_soc_reference.csv")
    if [row["target_uniprot"] for row in gene_map] != target_ids:
        errors.append("target_gene_map.csv differs from the fixed 194-target order")
    if [row["target_uniprot"] for row in target_reference] != target_ids:
        errors.append("target_feature_reference.csv differs from the fixed 194-target order")
    if set(row["reference_source"] for row in target_reference) != {"base_random_train_X"}:
        errors.append("target replacement reference does not identify the deposited training matrix")
    if [row["soc_code"] for row in evidence] != soc_codes:
        errors.append("soc_target_evidence_matrix.csv differs from the fixed 18-SOC order")
    if list(evidence[0].keys())[1:] != target_ids:
        errors.append("soc_target_evidence_matrix.csv differs from the fixed 194-target order")
    if len(examples) != 4 or len(example_soc) != 4 * 18:
        errors.append(f"paper profiler examples: inputs={len(examples)}, SOC rows={len(example_soc)}")

    for row in figures:
        for field in ("entrypoint", "source_data", "final_reference", "editable_source"):
            value = row[field]
            if value and not (ROOT / value).exists():
                errors.append(f"{row['figure']} missing {field}: {value}")
    if check_hashes:
        for row in final_files:
            path = ROOT / row["file"]
            if not path.is_file():
                errors.append(f"missing final figure: {row['file']}")
            elif sha256(path) != row["sha256"]:
                errors.append(f"checksum mismatch: {row['file']}")

    weights = read_csv(ROOT / "weights/weights_manifest.tsv", delimiter="\t")
    counts = Counter(row["model"] for row in weights)
    expected_weights = {"OT-ProfileNet": 7, "PlasmaBindNet-Fu": 5, "DoseExpoNet": 5, "DOT-SafeNet": 20}
    if dict(counts) != expected_weights:
        errors.append(f"weight manifest members: {dict(counts)}")
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    if any(not sha_pattern.fullmatch(row["sha256"]) for row in weights):
        errors.append("invalid SHA256 value in weights_manifest.tsv")

    sums = {}
    for line in (ROOT / "dist/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            value, filename = line.split(maxsplit=1)
            sums[filename.strip()] = value
    profiler_archive = ROOT / "dist/dotsafenet-safety-profiler.skill.zip"
    if sums.get(profiler_archive.name) != sha256(profiler_archive):
        errors.append("dotsafenet-safety-profiler archive checksum mismatch")
    with zipfile.ZipFile(profiler_archive) as archive:
        required_members = {
            "dotsafenet-safety-profiler/SKILL.md",
            "dotsafenet-safety-profiler/agents/openai.yaml",
            "dotsafenet-safety-profiler/scripts/profile.py",
        }
        if not required_members.issubset(archive.namelist()):
            errors.append("dotsafenet-safety-profiler archive lacks required skill files")

    return {
        "status": "failed" if errors else "passed",
        "targets": len(targets),
        "soc_tasks": len(socs),
        "manuscript_figures": len(figures),
        "code_figure_entrypoints": sum(bool(row["entrypoint"]) for row in figures),
        "weight_files": len(weights),
        "profiler_examples": len(examples),
        "profiler_reference_targets": len(target_reference),
        "profiler_skill_archive": profiler_archive.name,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figure-hashes", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = validate_package(check_hashes=not args.skip_figure_hashes)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
