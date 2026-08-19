# Supplementary Figure 7 guide

## Panel assignment

- Panels a-b: PPB scaffold-test RMSE and MAE.
- Panels c-f: DoseExpoNet dose-Cmax curves for sildenafil, alogliptin,
  fezolinetant, and torsemide.

## Contents

- PPB parameters: params_ppb_scaffold_metrics.yaml
- Cmax parameters: params_cmax_case_curves.yaml
- Source tables: data/ppb_scaffold_metrics/ and data/cmax_case_curves/
- Rendering code: scripts/render_ppb_scaffold_metrics.py and
  scripts/render_cmax_case_curves.py
- Outputs: outputs/ppb_scaffold_metrics/ and outputs/cmax_case_curves/

## Rendering

Run `python run.py` from this directory.

Each panel is exported independently. The a-b and c-f composites are retained
for inspection; final manuscript assembly is performed after the individual
panels pass visual review.

Drug-name placement inside the lower-right corner of each Cmax panel is
controlled by `text.drug_name_x` and `text.drug_name_y`.
