# Installation and deposited resources

The release uses separate environments because OT-ProfileNet and the exposure models use PyTorch, whereas DOT-SafeNet uses TensorFlow.

Required deposited resources:

- 37 checkpoint files listed in `weights/weights_manifest.tsv`.
- `PreMOTA/dataset_reg_multitask/{family}/proteins.txt` for seven target families.
- `PreMOTA/dataset_reg_multitask/{family}/esm2/` containing the fixed protein embeddings.

Download the checkpoint and inference-asset archives from `https://doi.org/10.5281/zenodo.22010299`, verify `SHA256SUMS.txt`, and extract both archives under a common resource directory while preserving their internal paths.

Expected target families are `GPCR`, `IonChannel`, `Enzyme`, `Kinase`, `NHR`, `Transporter` and `Other`. The protein inference resources contain 194 targets and occupy approximately 590 MB.

Install model-specific dependencies from:

- `src/ot_profilenet/requirements.txt`
- `src/plasmabindnet_fu/requirements.txt`
- `src/dose_exponet/requirements.txt`
- `src/dotsafenet/requirements.txt`

Use Python 3.9 or newer. Verify the checkpoint archive with `scripts/check_weights.py`, then run:

```bash
python scripts/profile_molecule.py validate \
  --checkpoint-root /path/to/checkpoints \
  --asset-root /path/to/inference_assets
```

The paper-case `replay` command requires only the lightweight figure environment. A full installation should pass both of the following checks:

```bash
python scripts/profile_molecule.py replay --output results/paper_cases
python scripts/profile_molecule.py validate \
  --checkpoint-root /path/to/checkpoints \
  --asset-root /path/to/inference_assets
```

The release quality-control record includes a complete Citalopram 20 mg/day run covering all four models, 194 targets, 18 SOC tasks and target attribution.

When the skill is installed outside the publication-package directory, set the package location with either method below:

```bash
export DOTSAFENET_RELEASE_ROOT=/path/to/manuscript_publication_release
python scripts/profile.py validate
```

or:

```bash
python scripts/profile.py --release-root /path/to/manuscript_publication_release validate
```
