# Supplementary PPB scaffold metrics

## Data source

`data/supplementary_table_2_plasmabindnet_and_fu.csv` contains the scaffold-test mean and standard deviation for every displayed model. Comparator and original PlasmaBindNet values are retained from the reported manuscript; PlasmaBindNet-Fu uses the five current scaffold seeds.

## Method

RMSE and MAE are shown as mean plus or minus standard deviation. The displayed RMSE and MAE values use the fractional scale reported in Supplementary Table 2.

## Rendering

```bash
python render.py --config params.yaml
```

The script exports separate RMSE and MAE panels and a two-panel preview in PNG, PDF, and SVG.
