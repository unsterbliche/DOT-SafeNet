# Release status

Completed before public deposition:

1. The final author list is recorded in `CITATION.cff`.
2. Original source code is assigned Apache-2.0; retained MIT and Apache notices are stored in `licenses/` and `THIRD_PARTY_NOTICES.md`.
3. Zenodo DOI `10.5281/zenodo.22010299` is reserved for the checkpoint and inference-asset record.
4. Complete training datasets are not redistributed; `data/manifests/data_manifest.csv` records their status.
5. The 37 checkpoints and the fixed OT-ProfileNet protein resources are packaged separately for Zenodo.

Final release actions:

1. Verify the SHA256 values after upload to Zenodo.
2. Test a CPU new-molecule inference run using files downloaded from the draft record.
3. Confirm the manuscript title and publication citation before publishing Zenodo and changing the GitHub repository to public.
