# Figure 2 Guide

## Scope

This directory contains independent Figure 2 panel scripts. The current manual-assembly set comprises panels a, c, d and e plus a standalone legend for panel d. Panels b and f are retained without modification.

## Render command

```powershell
python scripts/render_selected_panels.py
```

The command generates editable SVG and PDF files and 300-dpi PNG previews under `outputs/panels/`.

Independent outputs omit panel letters. IC50, AC50, pIC50 and pAC50 are displayed with 50 as a subscript.

## Current panel geometry

The agreed dimensions and axis settings are recorded in `params.yaml`. The reproduction is intentionally static: each panel script contains its explicit layout values, and `params.yaml` documents those values for later edits.

- Panel a: seven-family target-count pie chart with direct labels.
- Panel c: compact 3 rows x 2 columns; square scatter areas; marginal distributions touch the main frame; x-axis titles appear only on the bottom row and y-axis titles only in the left column.
- Panel d: RMSE and Pearson r stacked vertically with a shared x-axis; panel height is reduced by 20%; full frames and a compact separate 3+2-row legend are retained.
- Panel e: full-frame square plotting area within a canvas whose height is reduced by 20%; hERG is labeled inside the lower-right corner and no external title is used.

## Edit boundary

Visual edits may change figure dimensions, font sizes, colors, line widths, marker sizes, axis ranges, labels, legends and annotations. Data loading, regression metrics and benchmark values must remain unchanged unless the manuscript analysis is revised.

## Verification

Run the selected-panel renderer, then confirm:

- panel a contains seven labeled wedges and the counts sum to 194;
- panel c contains six family labels, six square scatter areas and four visible spines per main axis;
- panel d uses RMSE 0.50–0.95 and Pearson r 0.60–1.02;
- all four spines are visible in panels d and e;
- SVG files retain editable text.
