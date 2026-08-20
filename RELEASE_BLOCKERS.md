# Release status

Completed before public deposition:

1. The final author list is recorded in `CITATION.cff`.
2. Original source code is assigned Apache-2.0; retained MIT and Apache notices are stored in `licenses/` and `THIRD_PARTY_NOTICES.md`.
3. Zenodo DOI `10.5281/zenodo.22010299` is reserved for the checkpoint and inference-asset record.
4. Complete training datasets are not redistributed; `data/manifests/data_manifest.csv` records their status.
5. The 37 checkpoints and the fixed OT-ProfileNet protein resources are packaged separately for Zenodo.
6. The Zenodo draft contains 30 archive parts and seven manifests or documentation files; all 37 remote MD5 values match the local files, and the reconstructed archive SHA256 values match the source archives.
7. The distributable skill passed the lightweight paper-case tests and a complete Citalopram inference from SMILES through all four models on the controlled V100 environment.

Final release actions:

1. Confirm the final manuscript title and publication citation.
2. Publish the Zenodo record and change the GitHub repository to public in the coordinated release step.
3. A clean CPU-only installation test can be added if CPU execution will be advertised as a supported configuration; the recorded complete prospective run used the V100 inference environment.
