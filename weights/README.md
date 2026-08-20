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

The Zenodo record stores the two archives as consecutive 48 MiB parts:

- `dotsafenet-model-checkpoints-v1.0.0.tar.gz.part-*`
- `dotsafenet-ot-profilenet-inference-assets-v1.0.0.tar.gz.part-*`

Reconstruct and verify the archives with:

```bash
sha256sum -c PARTS_SHA256SUMS.txt
cat dotsafenet-model-checkpoints-v1.0.0.tar.gz.part-* > dotsafenet-model-checkpoints-v1.0.0.tar.gz
cat dotsafenet-ot-profilenet-inference-assets-v1.0.0.tar.gz.part-* > dotsafenet-ot-profilenet-inference-assets-v1.0.0.tar.gz
sha256sum -c SHA256SUMS.txt
```

`multipart_manifest.json` records part order, byte counts and SHA256 values. File-level checkpoint hashes remain in `weights_manifest.tsv`.

The reconstructed checkpoint and OT-ProfileNet inference-asset archives are separate release requirements. `scripts/profile_molecule.py validate` checks both before prospective inference.
