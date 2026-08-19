---
title: Model training and inference
document_type: sop
status: active
version: v1
date: 2026-08-05
scope: Publication release
source_of_truth: 09_results/manuscript_publication_release_20260805
---

# Prerequisites

Create model-specific environments from the requirements files. Place training datasets in a controlled data directory. Run scripts/validate_release.py. Set project, dataset, output, and GPU environment variables.

# Training

Run OT-ProfileNet pretraining and family-specific fine-tuning from src/ot_profilenet. Run the selected PPB, Cmax, and ADR commands in scripts/train. Retain commands, configurations, metrics, and checkpoints with every run.

# Prediction

Standardize drug names, canonical SMILES, and daily dose units. Predict 194 pAC50 values, PPB, and total Cmax. Calculate free Cmax and assemble 389 features in manifest order. Predict 18 SOC probabilities with all five clinically fine-tuned checkpoints and report the ensemble mean and fold dispersion.

# Figure generation

Run python scripts/render_all_figures.py. Each figure directory contains parameters, source tables, panel programs, and an output directory.

# Quality control

Confirm row identity after table merges, retain missing ADR labels, verify exposure units, and reject feature tables whose columns differ from target_order.csv.
