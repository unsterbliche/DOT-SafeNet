# PPB and Cmax data characteristics

## Data source

PPB and Cmax train, validation, and test CSV files were copied from the V100 MotifAttnNet archive. The ChEMBL reference set is the union of the seven OT-ProfileNet target-family activity tables.

## Inclusion

All PPB and Cmax rows in the archived random splits are included in the distributions and sample counts. The chemical-space plot includes all unique PPB and Cmax structures and a fixed random sample of 10,000 ChEMBL structures.

## Method

PPB is calculated as 100 x (1 - 10^activity), where activity is log10 fu. Cmax is calculated as 10^activity. Morgan fingerprints use radius 2 and 1,024 bits. Fingerprints are reduced to 40 principal components and then to two dimensions with t-SNE. The random seed and t-SNE settings are stored in params.yaml. Derived t-SNE coordinates are saved and reused on later renders.

## Rendering

```bash
python render.py --config params.yaml
```

Each histogram, table, and chemical-space plot is exported independently in PNG, PDF, and SVG. A compact PNG preview is also generated.
