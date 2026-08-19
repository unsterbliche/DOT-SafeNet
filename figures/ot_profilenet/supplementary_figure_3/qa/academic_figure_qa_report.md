# Supplementary Figure 3 QA and reproducibility report

## Data

Panel a contains 100 epochs of training and validation binary cross-entropy loss. Panels b-d contain AUROC, F1 score and MCC for seven target families. The displayed comparison includes MTGNN, OT-ProfileNet trained from scratch and fine-tuned OT-ProfileNet, giving 21 bars per metric.

The displayed ranges are AUROC 0.966-0.995, F1 score 0.768-0.922 and MCC 0.742-0.907. No displayed value is below the requested y-axis minimum of 0.5.

## Statistics

Each bar is an archived model-performance point estimate for one model and one target family. No replicate distribution, error interval or statistical test is encoded in these panels. Panel a plots the archived per-epoch loss values without smoothing.

## Figure checks

- All four panels use four-sided frames.
- Axis-label, tick-label and legend sizes are 9, 8 and 8 pt, respectively.
- Individual files and the composed figure contain no panel letters.
- Panels b-d use y limits 0.5-1.0.
- SVG and PDF vector files and 300 dpi PNG previews were regenerated.
- PNG content densities are 0.063, 0.290, 0.230 and 0.216 for panels a-d; no panel is empty or saturated.