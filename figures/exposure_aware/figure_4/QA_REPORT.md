# QA report: revised Figure 4

## Data integrity

- ROC scores and curves use the existing test-set prediction table.
- Drug order and all 18 SOC columns are unchanged; the matrices are rotated only for the vertical layout.
- Observed-label, 20 mg/day, and 250 mg/day heatmaps display the same 921 drugs and 18 SOCs in identical order.
- UMAP coordinates are unchanged; Figure 4C includes GAS, VAS, NER, and PSY.
- The UMAP prediction threshold remains 0.5.
- The complete supplementary UMAP contains all 3,806 observed-positive drug-SOC pairs.

## Visual verification

- ROC SOC abbreviations and AUROC values are placed inside the lower-right
  corner of each axis on separate lines.
- Observed labels, 20 mg/day predictions, and 250 mg/day predictions are displayed as adjacent vertical strips.
- Dose-dependent inference uses the same five A2-mono-lite clinical weights; log10 free Cmax and target-by-exposure interactions are recalculated from the 100 mg/day reference.
- GAS-VAS use matched teal shades; NER-PSY use matched purple shades.
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
