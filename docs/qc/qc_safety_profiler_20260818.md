# DOT-SafeNet safety profiler validation, 18 August 2026

## Data and feature checks

- The target manifest contains 194 unique UniProt accessions.
- The SOC manifest contains 18 MedDRA system organ classes.
- The DOT-SafeNet input contains one log10 free-Cmax value, 194 values of `6 - pAC50` and 194 target–exposure products.
- The target-replacement reference contains the training-set median and 90th percentile for all 194 target features.
- The ADR–target evidence matrix contains 18 rows and 194 ordered target columns.
- The development-background table contains empirical score percentiles for all 18 SOC tasks.

## Paper-case reproduction

The lightweight `replay` command generated reports for four manuscript cases:

| Compound | Daily dose |
|---|---:|
| Meclofenamic acid | 150 mg/day |
| Citalopram | 20 mg/day |
| Spironolactone | 25 mg/day |
| Candesartan cilexetil | 32 mg/day |

The output contained 72 SOC rows together with the deposited target-margin and target-attribution tables. This test uses fixed manuscript source tables and does not load checkpoints.

## Prospective full-model inference

Citalopram at 20 mg/day was processed from SMILES through OT-ProfileNet, PlasmaBindNet-Fu, DoseExpoNet and the clinically fine-tuned DOT-SafeNet ensemble on the V100 inference environment. The run generated:

- 194 target activity predictions;
- PPB, free fraction, total Cmax and free Cmax;
- 18 SOC ensemble scores and development-background percentiles;
- 194 target margins;
- 3,492 SOC–target replacement effects;
- HTML and JSON reports.

The copied output tables are stored in `docs/qc/profiler_full_smoke/`. This test recomputes all upstream predictions from molecular structure and is kept separate from the fixed manuscript-case source tables.

## Skill checks

The final `dotsafenet-safety-profiler.skill.zip` archive passed `quick_validate.py` after extraction. Its wrapper located a separately stored release package through `--release-root` and generated the four paper-case reports. `profile_molecule.py validate` found all 37 checkpoint files, all seven OT-ProfileNet protein-family directories and 194 protein inference resources on the controlled V100 installation.

Seven profiler unit tests passed. They cover the fixed target and SOC contracts, input validation, the published feature transformation, empirical percentiles, paper-case reports, the distributable wrapper and the stored full-pipeline result dimensions.

## Public deposition fields

The checkpoint and OT-ProfileNet inference-asset archives are stored in the Zenodo draft reserved as DOI `10.5281/zenodo.22010299`. The 48 MiB archive parts, reconstruction commands and SHA256 verification procedure are documented in `weights/README.md` and in the deposited `README.md`.
