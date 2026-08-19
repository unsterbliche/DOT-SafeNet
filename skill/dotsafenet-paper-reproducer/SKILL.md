---
name: dotsafenet-paper-reproducer
description: Validate, audit, and reproduce the DOT-SafeNet manuscript release. Use when checking the 194-target and 18-SOC data contracts, recalculating DOT-SafeNet metrics, rendering any of the six main or eight supplementary figures, verifying deposited model checkpoints, or testing whether a distributed release reproduces the reported paper content.
---

# DOT-SafeNet Paper Reproducer

Use the release’s own manifests and entrypoints. Do not infer figure numbering from directory names: the final manuscript map is stored in `data/manifests/figure_manifest.csv`.

## Workflow

1. Locate the release root. It must contain `data/manifests/figure_manifest.csv`, `scripts/validate_release.py`, and `configs/pipeline.yaml`.
2. Read `references/release_contract.md` before selecting a test.
3. Run `scripts/reproduce.py --release-root <path> audit`.
4. Run the requested operation:
   - `metrics` recalculates the reported DOT-SafeNet results from fixed prediction tables.
   - `figures --check-only` validates all manuscript figure entrypoints.
   - `figures --figure "Figure 5"` renders one exact manuscript figure.
   - `figures` renders every code-derived figure.
   - `tests` runs syntax, data-contract, checksum, and metric tests.
   - `weights --checkpoint-root <path>` checks all 37 selected checkpoint files.
5. Report the executed command, Python executable, pass/fail state, and generated report paths.

## Reproduction levels

- Package audit verifies 194 targets, 18 SOC tasks, 389 model features, 14 manuscript figures, final TIFF checksums, and the checkpoint manifest.
- Metric reproduction recalculates the 57-drug clinical before/after AUROC and AUPRC values and the 18 per-SOC AUROC values.
- Figure reproduction executes 12 code-derived entrypoints. Figure 1 and Figure 4 include editable schematic assembly. Supplementary Figures 1 and 5 are editable schematics. Their source is `paper_figures/figure.pptx`; submitted TIFFs are checksum-verified.
- Model prediction requires the deposited checkpoints listed in `weights/weights_manifest.tsv`.
- Full training requires the datasets listed in `data/manifests/data_manifest.csv` and the model-specific environments in `src/*/requirements.txt`.

Do not report full retraining as reproduced when only fixed prediction tables or figure source data were tested.

## Resources

- Read `references/release_contract.md` for the final figure map and expected counts.
- Read `references/environment.md` when creating an environment or diagnosing imports.
- Read `references/model_data_contracts.md` for feature construction, SOC order, model parameters, and checkpoint membership.
- Use `scripts/reproduce.py` as the stable entrypoint; do not duplicate its subprocess logic in ad hoc shell commands.
