# Model and data contracts

## Feature table

The DOT-SafeNet feature table contains `Drug`, `smiles`, `cmax_free(uM)`, 194 UniProt activity columns, and 194 `accession*Cmax` columns. The numerical model input has 389 values. `data/manifests/target_order.csv` defines the exact column order.

## SOC table

The label table contains SMILES followed by 18 MedDRA SOC columns. Values are 0, 1, or missing. Missing labels remain excluded from loss and evaluation. `data/manifests/soc_order.csv` defines the exact task order.

## DOT-SafeNet training

Base training uses observed labels, soft pseudo-dose labels, and paired monotonicity loss. The pseudo-dose schedule and weights are stored in `configs/dotsafenet_base.yaml`. Clinical fine-tuning updates the SOC task heads using CT-ADE clinical-dose records; parameters are stored in `configs/dotsafenet_clinical_finetune.yaml`.

## Reported metrics

Base five-fold test AUROC is 0.724005685 and AUPRC is 0.696274772. The corresponding fine-tuned values are 0.726494464 and 0.701788472.

For the independent 57-drug clinical-dose test table, mean five-fold AUROC changes from 0.612017847 to 0.622208789 and AUPRC changes from 0.908314917 to 0.912599708. `scripts/reproduce_metrics.py` recalculates these values from the ten fold-stage prediction tables.

## Checkpoints

`weights/weights_manifest.tsv` records model, role, ensemble member, relative checkpoint path, byte count, and SHA256. Run `scripts/check_weights.py` against the root of a deposited checkpoint archive.
