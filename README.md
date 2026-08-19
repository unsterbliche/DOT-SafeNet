# DOT-SafeNet

This package contains the source code, fixed model parameters, evaluation tables, figure source data, editable figure source, and final 600 dpi TIFF files used in the manuscript.

## Models

- OT-ProfileNet predicts pK and pAC50 values for 194 safety targets.
- PlasmaBindNet-Fu predicts plasma protein binding and free fraction.
- DoseExpoNet predicts dose-dependent total Cmax.
- DOT-SafeNet combines free Cmax, 194 pAC50 values, and 194 target–exposure products to predict 18 MedDRA system organ classes.

Model parameters are stored in `configs/`. `data/manifests/target_order.csv` and `data/manifests/soc_order.csv` define the fixed feature and task order. The selected checkpoints and fixed OT-ProfileNet inference assets are deposited at [Zenodo](https://doi.org/10.5281/zenodo.22010299).

## Package validation

Create a figure environment and run the release checks:

```bash
python -m pip install -r requirements-figures.txt
python scripts/validate_release.py
python scripts/render_all_figures.py --check-only
python scripts/reproduce_metrics.py
python scripts/run_tests.py
```

## Figure reproduction

`data/manifests/figure_manifest.csv` lists all six main figures and eight supplementary figures in manuscript order. Ten figures are generated from fixed source tables. Figure 1 and Figure 4 combine generated panels with editable schematic elements. Supplementary Figures 1 and 5 are editable model schematics. The complete editable source is `paper_figures/figure.pptx`; the submitted TIFF files are in `paper_figures/final_tiff/`. Large intermediate TIFF exports are omitted from Git because they can be regenerated from the panel source data.

Render every code-derived figure:

```bash
python scripts/render_all_figures.py --report docs/qc/figure_render_report.json
```

Render one manuscript figure:

```bash
python scripts/render_all_figures.py --figure "Figure 5"
```

## Training and prediction

Model-specific source code is in `src/`. Public training commands are in `scripts/train/`; prediction commands are in `scripts/predict/`. All data, checkpoint, output, and device locations are passed by command-line arguments or environment variables. The selected checkpoint members and SHA256 values are recorded in `weights/weights_manifest.tsv`. Large checkpoint and protein-embedding files are distributed through Zenodo and are not stored in Git.

Check a deposited checkpoint directory with:

```bash
python scripts/check_weights.py --checkpoint-root /path/to/checkpoints
```

Complete training datasets are listed in `data/manifests/data_manifest.csv`. Files that require separate redistribution permission are not copied into this package.

## Publication skill

The reproducibility skill is stored in `skill/dotsafenet-paper-reproducer/`. It audits the package, recalculates reported metrics, renders selected figures, and verifies deposited checkpoints.

## Prospective molecule profiling

`skill/dotsafenet-safety-profiler/` analyzes new small molecules at supplied daily doses. The public command produces 194-target activity and margin tables, PPB, total and free Cmax, 18-SOC DOT-SafeNet scores, development-background percentiles, target-replacement attribution, and HTML/JSON reports.

The lightweight installation test reconstructs the reports for four manuscript cases from the deposited panel source tables. It verifies the input, output, SOC, target-margin and attribution contracts without loading model checkpoints:

```bash
python scripts/profile_molecule.py validate
python scripts/profile_molecule.py replay --output results/paper_cases
```

The four examples are Meclofenamic acid at 150 mg/day, Citalopram at 20 mg/day, Spironolactone at 25 mg/day and Candesartan cilexetil at 32 mg/day.

Full new-molecule inference starts from SMILES and total daily dose and reruns OT-ProfileNet, PlasmaBindNet-Fu, DoseExpoNet and the clinically fine-tuned DOT-SafeNet ensemble. It requires the deposited checkpoint archive and fixed OT-ProfileNet protein resources:

```bash
python scripts/profile_molecule.py analyze \
  --input compounds.csv \
  --output results/new_compounds \
  --checkpoint-root /path/to/checkpoints \
  --asset-root /path/to/inference_assets \
  --ot-python /path/to/ot_env/bin/python \
  --exposure-python /path/to/exposure_env/bin/python \
  --adr-python /path/to/adr_env/bin/python \
  --device cuda:0
```

The generated HTML and JSON reports contain PPB, total and free Cmax, 194 predicted target activities and exposure margins, 18 SOC scores, within-SOC development-background percentiles, and target-replacement attribution. Command syntax and field definitions are documented in `skill/dotsafenet-safety-profiler/references/`.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Source code is released under the Apache License 2.0. Model checkpoints and inference assets in the associated Zenodo record are released under CC BY 4.0. Adapted MIT-licensed components and the Nature figure-style resource are documented in `THIRD_PARTY_NOTICES.md` and `licenses/`.
