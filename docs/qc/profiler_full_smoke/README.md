# Prospective profiler full-pipeline smoke test

This directory records the full Citalopram 20 mg/day smoke test executed on the V100 inference environment. The copied tables are retained as release quality-control evidence and are not used as training data.

The run completed the following calculations from the supplied SMILES and dose:

- OT-ProfileNet predictions for 194 targets;
- five-member PlasmaBindNet-Fu and DoseExpoNet ensembles;
- five clinically fine-tuned DOT-SafeNet members for 18 SOC tasks;
- 194 target margins and 3,492 SOC–target replacement effects.

The output contained 18 SOC rows, 194 target-margin rows and 3,492 attribution rows. `report.json` is the complete machine-readable report. The CSV files retain the main intermediate and final values used to inspect the model contract.

This prospective rerun is recorded separately from the four fixed manuscript-case source tables. The latter reproduce the submitted figure values; the prospective run recomputes all upstream predictions from molecular structure.
