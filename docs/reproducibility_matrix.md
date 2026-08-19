# Manuscript reproduction matrix

| Manuscript item | Source | Reproduction method |
|---|---|---|
| Figure 1 | generated ADR-output insert and editable framework | run Figure 1 entrypoint; assemble from `paper_figures/figure.pptx` |
| Figure 2 | OT-ProfileNet fixed tables | Python entrypoint |
| Figure 3 | PPB and Cmax fixed tables | Python entrypoint |
| Figure 4 | generated ADR–target panels and editable DOT-SafeNet strategy diagram | run Figure 4 entrypoint; assemble from `paper_figures/figure.pptx` |
| Figure 5 | DOT-SafeNet predictions, labels, metrics, and UMAP coordinates | Python entrypoint and `scripts/reproduce_metrics.py` |
| Figure 6 | four clinical-dose case tables | Python entrypoint |
| Supplementary Figure 1 | editable OT-ProfileNet schematic | `paper_figures/figure.pptx` |
| Supplementary Figures 2–4 | OT-ProfileNet source tables | Python entrypoints |
| Supplementary Figure 5 | editable PPB/Cmax schematic | `paper_figures/figure.pptx` |
| Supplementary Figures 6–8 | fixed dataset, case, and UMAP tables | Python entrypoints |
| Four Figure 6 clinical cases | deposited case inputs and fixed SOC, margin and attribution tables | `python scripts/profile_molecule.py replay --output results/paper_cases` |
| Prospective molecule profile | SMILES, total daily dose, deposited checkpoints and OT-ProfileNet inference assets | `python scripts/profile_molecule.py analyze ...` |

`paper_figures/final_tiff/` stores the submitted 600 dpi output for checksum comparison. `data/manifests/final_figure_files.csv` records pixel dimensions and SHA256 values.
