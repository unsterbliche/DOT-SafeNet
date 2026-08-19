# Model weights

`weights_manifest.tsv` records the selected checkpoint members, relative archive paths, byte counts, and SHA256 values.

The archive contains:

- seven OT-ProfileNet target-family checkpoints;
- five PlasmaBindNet-Fu checkpoints (seeds 2044, 2038, 2037, 2022, and 2032);
- five DoseExpoNet checkpoints (seeds 1998, 2022, 2023, 2024, and 2025);
- five DOT-SafeNet base TensorFlow checkpoint pairs;
- five clinically fine-tuned DOT-SafeNet checkpoint pairs.

Place deposited files under one checkpoint root while retaining the relative paths in the manifest. Verify the archive with:

```bash
python scripts/check_weights.py --checkpoint-root /path/to/checkpoint/root
```

Model checkpoints and fixed inference assets are deposited at:

- DOI: `10.5281/zenodo.22010299`
- URL: `https://doi.org/10.5281/zenodo.22010299`

Prospective molecule prediction also requires the fixed OT-ProfileNet protein files and ESM2 embeddings under `PreMOTA/dataset_reg_multitask/`. Deposit these inference assets with the checkpoint archive or as a second archive while retaining that relative directory structure.

The Zenodo record contains two archives:

- `dotsafenet-model-checkpoints-v1.0.0.tar.gz`
- `dotsafenet-ot-profilenet-inference-assets-v1.0.0.tar.gz`

Archive-level SHA256 values are stored in `SHA256SUMS.txt` in the same record. File-level checkpoint hashes remain in `weights_manifest.tsv`.

The checkpoint archive and OT-ProfileNet inference-asset archive are separate release requirements. `scripts/profile_molecule.py validate` checks both before prospective inference.
