# Supplementary Figure 8 guide

## Scientific content

The supplementary package contains the random-split comparison of MLDNN,
Multi-AttentiveFP, and DOT-SafeNet together with the complete 18-SOC UMAP.
All UMAP panels use the same fixed coordinates and prediction threshold.

## Rendering

Run `python run.py` from this directory.

## Inputs

- `data/model_comparison_source_data.csv`
- `data/model_comparison_statistics.csv`
- `data/all_soc_umap_source_data.csv.gz`

## Outputs

- `outputs/panels/supplementary_figure_8a_model_comparison.*`
- `outputs/panels/supplementary_figure_8b_all_soc_umap.*`
- `outputs/supplementary_figure_8.png`
- `outputs/supplementary_figure_8.pdf`
- `outputs/legends/supplementary_figure_8b_legend.*`

Colors, dimensions, threshold, and spacing are controlled by `params.yaml`.
Model metrics, UMAP coordinates, and predictions are not recalculated during
style revisions.
