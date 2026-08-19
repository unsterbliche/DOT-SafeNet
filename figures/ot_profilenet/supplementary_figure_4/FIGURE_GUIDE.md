# Supplementary Figure 4 Guide

## Scope

Supplementary Figure 4 contains six observed-versus-predicted pK regression panels: GPCR, ion channel, enzyme, kinase, NR and transporter. The archived source-data CSV files and all numerical calculations remain unchanged.

The visual implementation is inherited from `figure_2/scripts/plot_panel_c.py`: scatter plot, identity line, regression line, marginal histograms, metrics, square main axes, four-sided frame and direct family labels.

## Render command

```powershell
python run.py
```

The command generates six independent panels and the 3 x 2 composition under `outputs/`.

## Layout

- Composition: 88 x 132 mm, 3 rows x 2 columns.
- Marginal distributions touch the main plotting frame.
- X-axis titles appear only on the bottom row.
- Y-axis titles appear only in the left column.
- Panel letters are omitted.
- Each independent panel is 58 x 58 mm.

## Edit boundary

Visual settings may be adjusted in the panel renderer and composition script. Source CSV files, observed and predicted values, regression metrics and family assignments must not be changed for a visual revision.

## Verification

After rendering, confirm that all six datasets are non-empty, all six family labels are present, every main axis has four visible spines, and SVG text remains editable.
