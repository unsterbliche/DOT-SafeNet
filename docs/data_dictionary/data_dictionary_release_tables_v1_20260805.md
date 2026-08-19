---
title: Release data dictionary
document_type: data_dictionary
status: active
version: v1
date: 2026-08-05
scope: Publication release
source_of_truth: 09_results/manuscript_publication_release_20260805
---

# DOT-SafeNet feature table

| Column group | Count | Description |
|---|---:|---|
| Drug | 1 | Drug name |
| smiles | 1 | Molecular SMILES |
| cmax_free(uM) | 1 | Free systemic Cmax in micromolar units |
| UniProt accessions | 194 | OT-ProfileNet target activity features |
| accession*Cmax | 194 | Target-by-exposure product features |

# ADR label table

The first column is SMILES. The following 18 columns use soc_order.csv. Values are 0, 1, or missing.

# Figure tables

Every figure directory contains a data subdirectory. These tables store plotted values separately from manuscript layouts.
