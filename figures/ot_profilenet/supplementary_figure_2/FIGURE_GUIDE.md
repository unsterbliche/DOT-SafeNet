# Supplementary Figure 2 figure guide

## Purpose

Panel a shows the protein-sequence length distribution for the 7,258 pretraining targets. Panel b shows the number of compound-target pairs assigned to each target.

## Reproduction

Run `python scripts/render_selected_panels.py`. The script writes editable SVG and PDF files and 300 dpi PNG previews to `outputs/panels/`.

## Panel b data provenance

The counts were calculated from the OT-ProfileNet pretraining table. Grouping its 2,102,767 rows by `uniprot_id` yields 7,258 targets. The intermediate count list is retained as `data/panel_b_counts_raw.txt`; `scripts/prepare_panel_b_data.py` converts it to the plotting table and verifies both totals.

## Panel c

Run `python scripts/plot_panel_c.py`. The output is `outputs/panels/supplementary_figure_2c.png`, with matching PDF and SVG files.
