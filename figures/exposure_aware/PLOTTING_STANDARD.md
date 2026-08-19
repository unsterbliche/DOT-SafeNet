# Manuscript figure plotting standard

## Directory structure

Each manuscript figure is stored in one directory under this result package:

```text
figure_or_supplementary_figure/
  data/
  scripts/
  outputs/
    panels/
    legends/
  FIGURE_GUIDE.md
  params.yaml
  README.md
  run.py
```

All source tables required for rendering are stored under `data/`. Plotting
code is stored under `scripts/`. Generated files are stored only under
`outputs/`. A panel must be independently renderable before it is included in
a composite figure.

## Typography and axes

- Arial, Helvetica, or Liberation Sans.
- Axis labels: 9-10 pt.
- Tick labels: 8-9 pt.
- Legend text: 8-9 pt.
- No panel letters in independently exported panels.
- Cartesian plots use four visible spines, 0.6-0.8 pt.
- Tick marks point outward.
- Scientific notation uses proper superscripts and subscripts.
- Titles are omitted when the panel identity is clear from the axis labels or
  adjacent manuscript text.

## Layout

- Single-column panels use 82-89 mm width.
- Double-column composites use 183 mm width.
- Margins and panel gaps are minimized without clipping text.
- Shared legends are exported separately and are not placed over data.
- Repeated axis titles are removed when panels share an axis.

## Color

- Comparator models: neutral grey.
- Original model: muted blue.
- Revised or selected model: muted red or purple.
- Background observations: low-saturation blue or grey.
- The same model or biological meaning keeps the same color across figures.

## Export

- Each panel: editable SVG, vector PDF, and 300 dpi PNG.
- Composite figures: PDF and 300 dpi PNG; SVG is included when all elements
  remain editable.
- Output filenames begin with the manuscript figure and panel identifier.
- `latest.png` may be retained only as a convenience preview.

## Reproducibility

- `FIGURE_GUIDE.md` records the render command, inputs, outputs, and safe
  style parameters.
- `params.yaml` contains visual settings only.
- Statistical calculations and data transformations remain in scripts.
- Cached stochastic coordinates record the seed and are retained with the
  source data or outputs.
- Every render is checked for non-empty output, clipped text, overlapping
  annotations, and agreement with the source tables.
