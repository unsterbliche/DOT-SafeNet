---
title: Model lineage and data contracts
document_type: method
status: active
version: v1
date: 2026-08-05
scope: Publication release
source_of_truth: 09_results/manuscript_publication_release_20260805
---

# Data sources

OT-ProfileNet uses target-family activity data and produces pK and pAC50 values. PlasmaBindNet-Fu and DoseExpoNet use the PPB and Cmax datasets stored with Supplementary Figure 6. DOT-SafeNet uses 389-feature matrices generated from target activity and free exposure.

# Feature construction

Each drug is represented by log10 free Cmax, 194 pAC50 values, and 194 target-by-exposure products. target_order.csv defines the required schema.

# ADR prediction

DOT-SafeNet contains a drug encoder, an SOC encoder, an eight-dimensional pair representation, and one sigmoid task head per SOC. Base training uses real labels, soft pseudo-dose labels, and a paired monotonicity loss. Clinical fine-tuning updates SOC task heads with CT-ADE dose records.

# Evaluation

The clinical comparison contains 57 test drugs. Mean five-fold AUROC changed from 0.724005685 to 0.726494464; mean AUPRC changed from 0.696274772 to 0.701788472. Complete per-SOC ROC source data are stored with Figure 4.

# Limits

The public archive location for large training data and weights has not been assigned. Dataset redistribution terms require author review.
