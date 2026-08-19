# Figure guide

Purpose: reproduce the PPB and Cmax data-characteristic panels.

## Files

- Parameters: `params.yaml`
- Source data: `data/`
- Rendering code: `render.py`
- Main output: `outputs/supplementary_ppb_cmax_dataset_characteristics.png`
- Cached coordinates: `outputs/chemical_space_tsne_coordinates.csv.gz`

## Rendering

```powershell
python render.py --config params.yaml
```

Change typography, colors, panel sizes, fingerprint settings, and t-SNE settings in `params.yaml`. Deleting the cached coordinate file forces a new t-SNE calculation.
