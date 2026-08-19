---
title: Publication release validation
document_type: qc
status: completed
version: v1
date: 2026-08-05
scope: Publication release
source_of_truth: 09_results/manuscript_publication_release_20260805
---

# Validation summary

Status: passed with one environment warning.

# Structural checks

- Target manifest: 194 targets.
- DOT-SafeNet input: 389 numerical features plus drug and SMILES identifiers.
- SOC manifest: 18 tasks.
- Clinical fine-tuning test set: 921 drugs.
- Selected checkpoint files: 37.
- Files recorded in the release checksum manifest: 574.
- Release size: 318.5 MB.

# Source and syntax checks

The selected model source code and release scripts passed Python compileall. validate_release.py passed the feature, target, SOC, and required-file checks. render_all_figures.py found 12 figure entrypoints.

# Figure rendering

All current figure entrypoints were executed from the release directory. Output counts were:

- .csv: 6
- .gz: 1
- .md: 2
- .pdf: 32
- .png: 35
- .svg: 28
- .tiff: 4


Figure 5 initially referenced project-level case files. The release copy now contains its case manifest, five-fold dose predictions, target-occlusion summary, and shared plotting function under the Figure 5 directory.

Supplementary Figure 6 emitted a NumPy C-API warning from an optional compiled dependency in the local Anaconda environment. The plotting program returned exit code 0 and generated its outputs. A clean publication environment should install dependencies from a consistent NumPy build.

# Weight verification

weights/weights_manifest.tsv stores station-relative paths, byte counts, and SHA256 values for the seven OT-ProfileNet checkpoints, five PlasmaBindNet-Fu checkpoints, five DoseExpoNet checkpoints, and twenty DOT-SafeNet TensorFlow files.

# Items requiring author approval

- Select a software license.
- Confirm redistribution permissions for training and test datasets.
- Deposit large checkpoint files and complete training data in a public archive.
- Replace the placeholder author entry in CITATION.cff.
