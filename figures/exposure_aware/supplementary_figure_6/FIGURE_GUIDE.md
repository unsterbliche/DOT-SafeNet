# Supplementary Figure 6 guide

## Contents

- Parameters: params.yaml
- Source tables: data/
- Rendering code: scripts/render.py
- Independent panels: outputs/supplementary_figure_6a_* through
  outputs/supplementary_figure_6h_*
- Composite output: outputs/supplementary_figure_6.png and
  outputs/supplementary_figure_6.pdf
- Cached t-SNE coordinates: outputs/chemical_space_tsne_coordinates.csv.gz

## Rendering

Run `python run.py` from this directory.

Deleting the cached coordinate file reruns t-SNE. Ordinary typographic and
layout revisions must retain the cached coordinates.

Histogram panels use axis labels without internal titles. The common t-SNE
legend is exported separately under `outputs/legends/`.
