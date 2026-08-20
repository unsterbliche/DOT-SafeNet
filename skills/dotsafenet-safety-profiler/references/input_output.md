# Input and output

## Input CSV

Required columns:

- `smiles`: molecular structure.
- `dose_mg_day`: total daily dose in milligrams.

Recommended columns:

- `compound_id`: unique identifier for one molecule–regimen row.
- `name`: display name.
- `route`: administration route; the public examples use `oral`.

Represent a dose series as multiple rows with distinct `compound_id` values. A sample file is stored at `data/examples/paper_case_inputs.csv`.

## Commands

Validate the lightweight package:

```bash
python scripts/profile_molecule.py validate
```

Reconstruct all deposited paper-case reports from fixed manuscript source tables:

```bash
python scripts/profile_molecule.py replay --output results/paper_cases
```

Analyze new molecules:

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

The `replay` and `analyze` commands serve different validation purposes. `replay` verifies the released paper-case tables and report generation without checkpoints. `analyze` recomputes the target, exposure and ADR predictions from SMILES and dose.

## Output files

- `input_standardized.csv`: canonical structures and molecular weights.
- `ot_profile.json`: pK and pAC50 predictions for 194 targets.
- `ppb_predictions.csv`: ensemble PPB and free-fraction estimates.
- `cmax_predictions.csv`: member-level and ensemble total-Cmax estimates.
- `exposure_predictions.csv`: total and free exposure calculations.
- `dotsafenet_features.csv`: exact 389-feature matrix.
- `adr_predictions_by_fold.csv`: five member predictions.
- `adr_predictions.csv`: 18-SOC mean, standard deviation and background percentile.
- `target_margins.csv`: pAC50, predicted AC50 and free-exposure margin for 194 targets.
- `target_attribution.csv`: target-replacement effect, fold consistency and evidence level.
- `report.json` and `report.html`: machine-readable and human-readable reports.
