# Figure directory migration status

## Canonical directories

- Figure 4: `figure_4/`
- Supplementary Figure 6: `supplementary_figure_6/`
- Supplementary Figure 7: `supplementary_figure_7/`

All three render entry points completed successfully on 2026-08-05. Figure
paths in the figure workbench and figure manifest now reference these
directories.

## Retained obsolete copies

The former directories under
`03_code/manuscript_figures_reproducible_20260804/` and the four former
subdirectories under this result package are retained temporarily. They are not
referenced by the figure manifest or workbench. Removal requires a separate
confirmed cleanup operation because the old Supplementary Figure 6 source
directory contains a 114 MB tensor that is not used by the plotting script.
