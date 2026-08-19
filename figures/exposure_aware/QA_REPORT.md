# Figure QA report

## Data checks

- PPB rows: 7,826 (train 6,260; validation 782; test 784).
- Cmax rows: 5,642 (train 4,516; validation 564; test 562).
- Normalized-dose interval counts sum to 5,642.
- Chemical-space coordinates: 17,984 structures, including all 7,826 PPB rows after unique-structure labeling and 1,235 unique Cmax structures.
- PPB scaffold metric table: 7 displayed models; PlasmaBindNet-Fu RMSE = 0.522729 and MAE = 0.386713.
- Dose-Cmax panels: Sildenafil, Alogliptin, Fezolinetant, and Torsemide; 44 dose records.
- Figure 4 heatmaps: 921 drugs x 18 SOCs, with identical row and SOC order in the prediction and observed-label matrices.
- Figure 4 UMAP source: 3,806 observed-positive drug-SOC pairs.

## Visual checks

- All plotted axes use four visible spines.
- Standalone panels contain no panel letters.
- Figure legends are placed outside the data regions.
- Figure 4 uses an asymmetric layout; panels A-D are also exported independently.
- PNG previews were inspected for clipped labels, overlapping SOC names, and empty panels.

## Export checks

- Standalone panels: PNG, PDF, and SVG at 300 dpi for raster previews.
- Composite previews: PNG and PDF.
- All required composite files exceed 10 kB.

## Reproducibility

The canonical directories are figure_4/, supplementary_figure_6/, and
supplementary_figure_7/. Each contains source tables, rendering scripts,
parameter files, a fixed run.py entry point, and independently exported panels.
Supplementary Figure 7 contains separate PPB-metric and dose-Cmax parameter
files. The chemical-space coordinates are stored in
chemical_space_tsne_coordinates.csv.gz; later style changes do not rerun t-SNE.
