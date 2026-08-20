# DOT-SafeNet

DOT-SafeNet predicts dose-dependent adverse drug reaction (ADR) risk across 18 MedDRA system organ classes. The complete workflow combines:

- **OT-ProfileNet**: activity prediction for 194 safety targets;
- **PlasmaBindNet-Fu**: plasma protein binding and unbound-fraction prediction;
- **DoseExpoNet**: dose-dependent total Cmax prediction;
- **DOT-SafeNet**: integration of free Cmax, target activity and target–exposure features for 18-SOC prediction.

Model checkpoints and fixed OT-ProfileNet inference resources are deposited at [Zenodo](https://doi.org/10.5281/zenodo.22010299).

## Repository contents

```text
configs/      Training and inference parameters
data/         Model definitions, test data and four paper examples
figures/      Final 600 dpi TIFF files
reproduce/    Figure code and fixed source tables in manuscript order
scripts/      Training, prediction, evaluation and validation commands
skills/       Installable Codex skills for reproduction and molecule profiling
src/          Model implementations
tests/        Data-contract and inference tests
weights/      Checkpoint manifest and Zenodo download instructions
```

## Validate the release

```bash
python -m pip install -r requirements-figures.txt
python scripts/validate_release.py
python scripts/reproduce_metrics.py
python scripts/render_all_figures.py --check-only
python scripts/run_tests.py
```

## Reproduce figures

The directories under `reproduce/` follow the final manuscript numbering. Generated files are written to an untracked `outputs/` directory inside each figure directory.

```bash
python scripts/render_all_figures.py --list
python scripts/render_all_figures.py --figure "Figure 5"
python scripts/render_all_figures.py
```

Supplementary Figures 1 and 5 are model schematics and are supplied as final TIFF files. The remaining 12 figures have code-derived entrypoints.

## Reproduce paper examples

The lightweight replay command rebuilds reports for Meclofenamic acid, Citalopram, Spironolactone and Candesartan cilexetil from the fixed paper outputs:

```bash
python scripts/profile_molecule.py validate
python scripts/profile_molecule.py replay --output results/paper_cases
```

## Predict a new molecule

Prepare a CSV with `smiles` and `dose_mg_day`; optional columns are `compound_id`, `name` and `route`. Full inference requires the Zenodo checkpoints and OT-ProfileNet inference resources.

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

The output includes PPB, total and free Cmax, 194 predicted target activities and exposure margins, 18 SOC scores, background percentiles, target-replacement attribution, and HTML/JSON reports.

## Training and model evaluation

Model implementations are stored in `src/`. Public commands are named by model and operation:

```text
scripts/train_ot_profilenet.sh
scripts/train_plasmabindnet_fu.sh
scripts/train_dose_exponet.sh
scripts/train_dotsafenet.sh
scripts/predict_ot_profilenet.sh
scripts/predict_plasmabindnet_fu.sh
scripts/predict_dose_exponet.sh
scripts/predict_dotsafenet_clinical.sh
```

Complete training datasets remain subject to their original database licenses. The repository includes the fixed DOT-SafeNet test table, model feature order, SOC order, target annotations and figure source tables needed for the reported evaluations.

## Codex skills

The two directories under `skills/` are directly installable; separate archive copies are unnecessary.

```bash
cp -r skills/dotsafenet-paper-reproducer ~/.codex/skills/
cp -r skills/dotsafenet-safety-profiler ~/.codex/skills/
```

The paper-reproducer skill validates the release, recalculates reported metrics and renders figures. The safety-profiler skill runs paper-case replay or full prospective molecule analysis.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Source code is released under the Apache License 2.0. Model checkpoints and inference assets in the associated Zenodo record are released under CC BY 4.0. Third-party components are listed in `THIRD_PARTY_NOTICES.md`.
