"""Validate source-data counts, plotting outputs and public package completeness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent


def add(checks: List[Dict[str, object]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def expected_panel_stems(include_sfig2b: bool) -> List[Path]:
    stems = []
    for panel in "abcdef":
        stems.append(ROOT / "figure_2" / "outputs" / "panels" / ("figure_2" + panel))
    stems.append(ROOT / "figure_2" / "outputs" / "figure_2")
    for panel in "acd":
        stems.append(ROOT / "supplementary_figure_2" / "outputs" / "panels" /
                     ("supplementary_figure_2" + panel))
    if include_sfig2b:
        stems.append(ROOT / "supplementary_figure_2" / "outputs" / "panels" /
                     "supplementary_figure_2b")
        stems.append(ROOT / "supplementary_figure_2" / "outputs" / "supplementary_figure_2")
    for panel in "abcd":
        stems.append(ROOT / "supplementary_figure_3" / "outputs" / "panels" /
                     ("supplementary_figure_3" + panel))
    stems.append(ROOT / "supplementary_figure_3" / "outputs" / "supplementary_figure_3")
    for panel in "abcdef":
        stems.append(ROOT / "supplementary_figure_4" / "outputs" / "panels" /
                     ("supplementary_figure_4" + panel))
    stems.append(ROOT / "supplementary_figure_4" / "outputs" / "supplementary_figure_4")
    return stems


def validate_png(path: Path) -> Dict[str, object]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path).convert("RGB") as image:
        array = np.asarray(image)
        nonwhite = np.any(array < 248, axis=2)
        density = float(nonwhite.mean())
        return {"width": int(array.shape[1]), "height": int(array.shape[0]), "content_density": density}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-sfig2b", action="store_true")
    args = parser.parse_args()
    checks: List[Dict[str, object]] = []

    family = pd.read_csv(ROOT / "figure_2" / "data" / "panel_a_target_counts.csv")
    add(checks, "Figure 2a target total", int(family["target_count"].sum()) == 194,
        "sum={}".format(int(family["target_count"].sum())))

    pac50 = pd.read_csv(ROOT / "figure_2" / "data" / "panel_c_pac50_predictions.csv")
    add(checks, "Figure 2c finite values", np.isfinite(pac50[["observed", "predicted"]]).all().all(),
        "rows={}".format(len(pac50)))

    lengths = pd.read_csv(ROOT / "supplementary_figure_2" / "data" /
                          "panel_a_protein_sequence_lengths.csv")
    add(checks, "Supplementary Figure 2a proteins", len(lengths) == 7258,
        "rows={}".format(len(lengths)))

    class_counts = pd.read_csv(ROOT / "supplementary_figure_2" / "data" /
                               "panel_c_class_balance.csv").set_index("measure")
    pair_total = int(class_counts.loc["positive_pairs", "value"] +
                     class_counts.loc["negative_pairs", "value"])
    add(checks, "Supplementary Figure 2c pair total", pair_total == 2102767,
        "positive+negative={}".format(pair_total))

    loss = pd.read_csv(ROOT / "supplementary_figure_3" / "data" / "panel_a_pretraining_loss.csv")
    add(checks, "Supplementary Figure 3a epochs", len(loss) == 100 and loss["epoch"].iloc[-1] == 100,
        "rows={}, last_epoch={}".format(len(loss), int(loss["epoch"].iloc[-1])))

    include_sfig2b = (ROOT / "supplementary_figure_2" / "data" /
                      "panel_b_compounds_per_target.csv").exists()
    if not include_sfig2b:
        add(checks, "Supplementary Figure 2b source data", args.allow_missing_sfig2b,
            "missing; release package remains incomplete for this panel")

    for stem in expected_panel_stems(include_sfig2b):
        for suffix in ["svg", "pdf", "png"]:
            path = stem.with_suffix("." + suffix)
            add(checks, "output {}".format(path.relative_to(ROOT)),
                path.exists() and path.stat().st_size > 0,
                "{} bytes".format(path.stat().st_size if path.exists() else 0))
        png_path = stem.with_suffix(".png")
        if png_path.exists():
            info = validate_png(png_path)
            density_ok = 0.01 < info["content_density"] < 0.65
            add(checks, "PNG content {}".format(png_path.relative_to(ROOT)), density_ok,
                "{}x{}, density={:.4f}".format(info["width"], info["height"], info["content_density"]))
        svg_path = stem.with_suffix(".svg")
        if svg_path.exists():
            text = svg_path.read_text(encoding="utf-8", errors="ignore")
            add(checks, "editable SVG {}".format(svg_path.relative_to(ROOT)), "<text" in text,
                "contains SVG text nodes={}".format("<text" in text))

    failures = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "allow_missing_sfig2b": args.allow_missing_sfig2b,
        "checks": checks,
        "failure_count": len(failures),
    }
    qa_dir = ROOT / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "reproduction_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# Reproduction validation", "", "Status: **{}**".format(report["status"]), ""]
    for item in checks:
        lines.append("- [{}] {} — {}".format(item["status"], item["check"], item["detail"]))
    (qa_dir / "reproduction_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "sha256_checksums.csv":
            records.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    pd.DataFrame(records).to_csv(qa_dir / "sha256_checksums.csv", index=False)
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failures": len(failures)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

