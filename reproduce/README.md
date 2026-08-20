# Figure reproduction

Directory names follow the final manuscript numbering. Run an individual figure from the repository root:

```bash
python scripts/render_all_figures.py --figure "Figure 3"
```

Run all 12 code-derived figures:

```bash
python scripts/render_all_figures.py
```

Each entrypoint reads its local `data/` and `params.yaml` files and writes generated panels to `outputs/`. Supplementary Figures 1 and 5 are model schematics supplied under `figures/` as final TIFF files.
