---
name: dotsafenet-paper-reproducer
description: Validate, audit, and reproduce the DOT-SafeNet manuscript release. Use when checking the 194-target and 18-SOC data contracts, recalculating DOT-SafeNet metrics, rendering any of the six main or eight supplementary figures, verifying deposited model checkpoints, or testing whether a distributed release reproduces the reported paper content.
---

# DOT-SafeNet Paper Reproducer

Use the numbered entrypoints under `reproduce/`; their directory names match the final manuscript numbering.

## Workflow

1. Locate the release root. It must contain `data/model/target_order.csv`, `scripts/validate_release.py`, and `configs/pipeline.yaml`.
2. Read `references/release_contract.md` before selecting a test.
3. Run the requested command from the release root:
   - `python scripts/validate_release.py` audits the release package and fixed data contracts.
   - `python scripts/reproduce_metrics.py` recalculates the reported DOT-SafeNet results from fixed prediction tables.
   - `python scripts/render_all_figures.py --check-only` validates all manuscript figure entrypoints.
   - `python scripts/render_all_figures.py --figure "Figure 5"` renders one exact manuscript figure.
   - `python scripts/render_all_figures.py` renders every code-derived figure.
   - `python scripts/run_tests.py` runs syntax, data-contract, figure and metric tests.
   - `python scripts/check_weights.py --checkpoint-root <path>` checks all 37 selected checkpoint files.
4. Report the executed command, Python executable, pass/fail state, and generated report paths.

## Reproduction levels

- Package audit verifies 194 targets, 18 SOC tasks, 389 model features, 14 manuscript figures and the checkpoint manifest.
- Metric reproduction recalculates the 57-drug clinical before/after AUROC and AUPRC values and the 18 per-SOC AUROC values.
- Figure reproduction executes 12 code-derived entrypoints. Supplementary Figures 1 and 5 are model schematics supplied as final TIFF files.
- Model prediction requires the deposited checkpoints listed in `weights/weights_manifest.tsv`.
- Full training requires the datasets described in the manuscript and the model-specific environments in `src/*/requirements.txt`.

Do not report full retraining as reproduced when only fixed prediction tables or figure source data were tested.

## Resources

- Read `references/release_contract.md` for the final figure map and expected counts.
- Read `references/environment.md` when creating an environment or diagnosing imports.
- Read `references/model_data_contracts.md` for feature construction, SOC order, model parameters, and checkpoint membership.
- Use the release-root commands listed above; their names match the public README and validation tests.
