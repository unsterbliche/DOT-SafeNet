# Academic Figure Skill Asset Confirmation (verified against project production figures)
# (a-d) Dose-response and target-attribution cases -> project production script -> param inherit
# RULE: Stored fold predictions and occlusion summaries are the quantitative source.

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
OUT = ROOT / "outputs"
WORK = OUT / "case_panels"
SOURCE_SCRIPT = PROJECT / "03_code" / "adr_film_exposure_clinical" / "scripts" / "plot_selected_case_exact_dose_figures_top10_secondary.py"
MANIFEST = PROJECT / "03_code" / "adr_film_exposure_clinical" / "manifests" / "selected_case_exact_dose_20260713.csv"
RESULTS = PROJECT / "09_results" / "adr_film_exposure_clinical" / "selected_case_exact_dose_a2mono_oldpremota_20260714"
PREDICTIONS = RESULTS / "selected_case_dose_predictions_by_fold.csv"
OCCLUSION = RESULTS / "selected_case_target_occlusion_robust_summary.csv"
MEMBERSHIP = ROOT / "data" / "soc_target_membership.csv"
OUTPUT_NAMES = {
    "meclofenamic_acid": "figure5a_meclofenamic_acid",
    "citalopram": "figure5b_citalopram",
    "spironolactone": "figure5c_spironolactone",
    "candesartan": "figure5d_candesartan_cilexetil",
}


def load_plotter():
    spec = importlib.util.spec_from_file_location("case_plotter", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose_preview(paths: list[Path], labels: list[str], dpi: int) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    gap = 28
    canvas = Image.new("RGB", (2 * width + gap, 2 * height + gap), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % 2) * (width + gap)
        y = (index // 2) * (height + gap)
        canvas.paste(image, (x, y))
        draw.text((x + 10, y + 6), label, fill="#222222", font=font)
    canvas.save(OUT / "figure5_clinical_cases_overview.png", dpi=(dpi, dpi))
    for image in images:
        image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    plotter = load_plotter()
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    predictions = pd.read_csv(PREDICTIONS)
    occlusion = pd.read_csv(OCCLUSION)
    membership = pd.read_csv(MEMBERSHIP)
    selected = cfg["figure"]["selected_cases"]
    labels = cfg["figure"]["panel_labels"]
    preview_paths = []
    target_tables = []
    stats_rows = []

    for case_id in selected:
        case_manifest = manifest[manifest["figure_id"].eq(case_id)].copy()
        if case_manifest.empty:
            raise ValueError(f"Manifest lacks {case_id}")
        image_path, case_stats, targets = plotter.plot_case(
            case_id, case_manifest, predictions, occlusion, membership, WORK
        )
        output_stem = OUTPUT_NAMES[case_id]
        png_path = OUT / f"{output_stem}.png"
        pdf_path = OUT / f"{output_stem}.pdf"
        shutil.copy2(image_path, png_path)
        shutil.copy2(image_path.with_suffix(".pdf"), pdf_path)
        preview_paths.append(png_path)
        target_tables.append(targets)
        stats_rows.extend(case_stats)

    pd.concat(target_tables, ignore_index=True).to_csv(OUT / "figure5_case_target_source.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(OUT / "figure5_case_dose_statistics.csv", index=False)
    compose_preview(preview_paths, [labels[x] for x in selected], int(cfg["figure"]["dpi"]))


if __name__ == "__main__":
    main()