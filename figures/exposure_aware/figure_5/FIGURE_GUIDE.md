# Figure 6: clinical dose and target-attribution case panels

Figure 6 contains four separately rendered clinical case panels generated with the DOT-SafeNet clinical model and OT-ProfileNet target predictions:

- Meclofenamic acid-GAS
- Citalopram-CAR
- Spironolactone-REP
- Candesartan cilexetil-VAS/REN

Each file contains a dose-response plot above a target-replacement attribution plot. The drug name and ADR category are written inside the lower-right corner of the dose-response plot. No panel letters or legends are embedded in the case files. A common legend is exported separately.

Direct SOC evidence is blue, secondary evidence is orange and targets without mapped SOC evidence are gray. The clinical dose remains marked by a red diamond in the dose-response panel; target-attribution panels do not use a separate key-target category.

Render with `python run.py`.

Primary outputs:

- `outputs/figure5_meclofenamic_acid.png/.pdf/.svg`
- `outputs/figure5_citalopram.png/.pdf/.svg`
- `outputs/figure5_spironolactone.png/.pdf/.svg`
- `outputs/figure5_candesartan.png/.pdf/.svg`
- `outputs/figure5_case_legend.png/.pdf/.svg`
- `outputs/figure5_case_target_source.csv`

The plotting code is `scripts/render_cases_a4.py`. Stored fold predictions and target-occlusion summaries are the quantitative sources. The former M1-M9 ADR-target matrix is retained as Supplementary Figure 7B under `../supplementary_figure_7_dotsafenet/`.
