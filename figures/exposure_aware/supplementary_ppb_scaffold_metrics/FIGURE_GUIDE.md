# Figure guide

Purpose: reproduce the scaffold-test RMSE and MAE panels for the PPB models.

## Files

- Parameters: `params.yaml`
- Source data: `data/supplementary_table_2_plasmabindnet_and_fu.csv`
- Rendering code: `render.py`
- Main output: `outputs/supplementary_ppb_scaffold_metrics.png`

## Rendering

```powershell
python render.py --config params.yaml
```

Change typography, colors, axis limits, rotation, and panel size in `params.yaml`. Statistical values are read directly from the source table.
