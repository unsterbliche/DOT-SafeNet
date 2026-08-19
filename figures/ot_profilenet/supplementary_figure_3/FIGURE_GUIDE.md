# Figure Guide

Purpose: keep recurring figure edits small-scope and fast.

## Codex Rules

- Start by reading this file and `params.yaml` only.
- Do not inspect the whole project unless rendering fails or the requested change cannot be handled through parameters or a small local rendering edit.
- Prefer parameter edits for visual changes: titles, labels, colors, font sizes, line widths, marker sizes, axis ranges, legend placement, annotations, figure size, DPI, export format, and output path.
- Small local code edits are allowed when parameters are not enough, but keep the render command, project layout, data-loading path, and analysis logic unchanged.
- Do not rewrite plotting, data-loading, or analysis logic for ordinary style/layout tweaks.
- If a new visual control is needed, add the smallest useful parameter or helper and document it here.

## Files

- Parameter file: `params.yaml`
- Render command: `python run.py`
- Main output: `outputs\supplementary_figure_3.png`

## First-Pass Edit Boundary

Safe to change first:

- `figure`, `text`, `colors`, `axes`, `legend`, `layout`, `annotations`, and `export` settings in `params.yaml`

The current scripts record the display values listed in `params.yaml`; ordinary changes are made in the corresponding plotting helper while retaining the same data inputs.

Safe small code edits:

- Add a visual parameter and wire it into the existing render path.
- Adjust local style mappings, annotation placement, legend formatting, axis formatting, or export details.
- Add a tiny helper for display formatting when it does not affect data or calculations.

Inspect code only when:

- The requested change has no matching parameter.
- The render command fails.
- The output does not reflect the parameter change.
- The user asks for data, model, calculation, or plotting logic changes.

Do not change without explicit instruction:

- Data schema, statistical calculations, model logic, dependency stack, project layout, or render entrypoint.

## Verification

After edits, run:

```bash
python run.py
```

Then inspect `outputs\supplementary_figure_3.png` before reporting completion.
- Separate legends: outputs\legends\supplementary_figure_3a_legend.* and outputs\legends\supplementary_figure_3b-d_legend.*
