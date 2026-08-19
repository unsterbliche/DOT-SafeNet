# QA report: revised Figure 4

## Data integrity

- ROC scores and curves use the existing test-set prediction table.
- Heatmap column order and all 18 SOC rows are unchanged.
- Both observed-label and predicted-probability heatmaps display all SOC names.
- UMAP coordinates are unchanged; Figure 4C includes GAS, VAS, NER, and PSY.
- The UMAP prediction threshold remains 0.5.
- The complete supplementary UMAP contains all 3,806 observed-positive drug-SOC pairs.

## Visual verification

- ROC SOC abbreviations and AUROC values are placed inside the lower-right
  corner of each axis on separate lines.
- The observed-label heatmap precedes the predicted-probability heatmap.
- ROC, heatmap, and UMAP legends are excluded from the main panels and exported
  separately.
- UMAP limits include 7.5% horizontal and 10% vertical padding.
- UMAP SOC abbreviations are placed inside the upper-left corner.
- Four-sided frames follow the project plotting standard and the user's
  explicit preference.
- No labels, legends, or data marks are clipped in the rendered PNGs.

## Export verification

All main panels and legends were generated as PNG, PDF, and SVG. The main
composite and supplementary composite were generated as PNG and PDF at 300 dpi.
