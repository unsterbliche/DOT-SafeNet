---
title: Publication release changelog
document_type: changelog
status: active
version: v1
date: 2026-08-05
scope: Publication release
source_of_truth: 09_results/manuscript_publication_release_20260805
---

# 2026-08-05 release candidate 1

- Copied selected training and prediction sources into model-specific directories.
- Recorded final parameters for four models.
- Added fixed target and SOC manifests.
- Added the complete 57-drug DOT-SafeNet test matrices.
- Collected current figure programs and panel-level source data.
- Added training, prediction, validation, and figure-rendering commands.

# 2026-08-17 release candidate 2

- Replaced the old figure numbering with the final six-main-figure and eight-supplementary-figure map.
- Added the editable 14-slide figure source and the submitted 600 dpi TIFF files with SHA256 values.
- Added package, metric, figure, and checkpoint validation programs.
- Recalculated the clinical-dose and 18-SOC DOT-SafeNet metrics from fixed prediction tables.
- Replaced machine-specific training and prediction paths with required command-line parameters.
- Added OT-ProfileNet, DoseExpoNet, and DOT-SafeNet prediction commands.
- Added the distributable `dotsafenet-paper-reproducer` skill and an installation test.

# 2026-08-18 release candidate 3

- Added the `dotsafenet-safety-profiler` skill for prospective SMILES-and-dose analysis.
- Added stable commands for OT-ProfileNet, PlasmaBindNet-Fu, DoseExpoNet and clinically fine-tuned DOT-SafeNet ensemble inference.
- Added the fixed 389-feature construction rule, 194 target margins, development-background SOC percentiles and target-replacement attribution.
- Added four manuscript-case inputs and fixed source tables for report reproduction.
- Added a complete Citalopram 20 mg/day run from SMILES through all four models as an inference test record.
- Added unit tests for the feature transformation, empirical percentiles, report reproduction, full-run output dimensions and installed-skill wrapper.
- Added deterministic skill archive generation and SHA256 files.
