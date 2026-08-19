# Figure 3 guide

## Contents

- Panels a-b: PPB and log10 unbound-fraction distributions transferred from Supplementary Figure 6.
- Panel c: PPB error by measured-PPB interval.
- Panel d: PPB model comparison.
- Panel e: external-test Pearson correlation; external-test RMSE is excluded from the main figure.
- Panel f: a compact 3 × 2 grid comparing PlasmaBindNet-Fu with six online PPB predictors on the external set. Each inset is approximately square and the complete grid is shorter than one standard panel. Online predictors use blue open circles, PlasmaBindNet-Fu uses coral filled circles, and predictor names are printed at the upper left of each inset. PPB values are transformed as -log10(max(1 - PPB/100, 1e-4)). Point types and the background bands corresponding to measured PPB intervals of 0–40%, 40–80%, and 80–100% are defined in a separate legend.
- Panel g: Cmax model comparison.
- Panels h-i: dose-Cmax curves for cevimeline hydrochloride and ciprofloxacin hydrochloride.

## Rendering

Run `python run.py` from this directory.

Each panel and each legend is exported separately. `params.yaml` controls fonts, colors, dimensions, and spacing. Source values are stored under `data/` and are not recalculated during style revisions.

Panels h-i use the Supplementary Figure 7 dose-Cmax drawing code for the five-seed mean, standard-deviation band, observed points, axes, and frame. Predicted and observed total Cmax use blue lines and points. Predicted and observed free Cmax use coral dashed lines and open diamonds. The y-axis reports log10 concentration so that both exposure measures are represented correctly.

The composite is arranged as a compact 3 x 3 grid at 7.2 inches wide for an A4 manuscript page.
