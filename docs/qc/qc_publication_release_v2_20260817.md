# Publication release validation, 17 August 2026

## Package checks

- Target manifest: 194 activity features and 194 target–exposure product features.
- DOT-SafeNet input: 389 numerical features in the expected order.
- SOC manifest: 18 tasks in the label-table order.
- Manuscript figures: six main figures and eight supplementary figures.
- Code-derived figure entrypoints: 12.
- Final TIFF files: 14 files at 600 dpi; all SHA256 values match `data/manifests/final_figure_files.csv`.
- Checkpoint manifest: 37 files. All files on the controlled checkpoint store matched the recorded byte counts and SHA256 values.

## Metric reproduction

The 57-drug clinical test contains 907 observed drug–SOC pairs. Recalculation from the five fold-level prediction tables gave:

| Stage | AUROC | AUPRC |
|---|---:|---:|
| Before clinical fine-tuning | 0.612017847 | 0.908314917 |
| After clinical fine-tuning | 0.622208789 | 0.912599708 |

The largest absolute difference between recalculated and stored fold metrics was `1.11e-16`. All 18 per-SOC AUROC values were recalculated; their mean was 0.721146334.

The base five-fold test results were AUROC 0.724005685 and AUPRC 0.696274772. The corresponding clinically fine-tuned values were 0.726494464 and 0.701788472.

## Figure reproduction

All 12 code-derived entrypoints completed with return code 0. The run report is `figure_render_report_20260817.json`. Figure 1, Figure 4, Supplementary Figure 1, and Supplementary Figure 5 include editable schematic elements stored in `paper_figures/figure.pptx`.

## Skill test

`dotsafenet-paper-reproducer` passed `quick_validate.py`. A clean copied installation located the release through `--release-root`, completed the package audit, checked all figure entrypoints, and recalculated the reported metrics.

## Author-supplied deposition fields

The remaining fields are listed in `RELEASE_BLOCKERS.md`: final citation authors, software license, checkpoint archive DOI, and training-dataset deposition records.
