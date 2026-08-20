# Installation and deposited resources

The release uses one Python 3.9 environment for OT-ProfileNet, PlasmaBindNet-Fu, DoseExpoNet and DOT-SafeNet. PyTorch and TensorFlow inference run in separate child processes from the same environment. `requirements-unified.txt` fixes the compatible PyTorch, PyG, TensorFlow, numerical and chemistry packages.

Required deposited resources:

- 37 checkpoint files listed in `weights/weights_manifest.tsv`.
- `PreMOTA/dataset_reg_multitask/{family}/proteins.txt` for seven target families.
- `PreMOTA/dataset_reg_multitask/{family}/esm2/` containing the fixed protein embeddings.

Download the checkpoint and inference-asset archives from `https://doi.org/10.5281/zenodo.22010299`, verify `SHA256SUMS.txt`, and extract both archives under a common resource directory while preserving their internal paths.

Expected target families are `GPCR`, `IonChannel`, `Enzyme`, `Kinase`, `NHR`, `Transporter` and `Other`. The protein inference resources contain 194 targets and occupy approximately 590 MB.

Create the shared environment from the repository root:

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-unified.txt
```

The model-specific requirement files point to this shared definition.

Use Python 3.9 or newer. Download all `*.part-*` files from the Zenodo record, reconstruct the two archives as described in `weights/README.md`, extract them while retaining their directory structure, and verify the checkpoint directory with `scripts/check_weights.py`. Then run:

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

Run the numerical end-to-end check after installation:

```bash
.venv/bin/python scripts/validate_unified_runtime.py \
  --checkpoint-root /path/to/checkpoints \
  --asset-root /path/to/inference_assets \
  --output-dir results/runtime_validation \
  --device cuda:0
```

The validation runs Citalopram at 20 mg/day through all model stages and compares exposure, 194-target, 18-SOC and target-attribution outputs with the deposited reference tables.

`data/examples/citalopram_full_inference/` contains a complete Citalopram 20 mg/day reference run covering all four models, 194 targets, 18 SOC tasks and target attribution.

When the skill is installed outside the publication-package directory, set the package location with either method below:

```bash
export DOTSAFENET_RELEASE_ROOT=/path/to/manuscript_publication_release
python scripts/profile.py validate
```

or:

```bash
python scripts/profile.py --release-root /path/to/manuscript_publication_release validate
```
