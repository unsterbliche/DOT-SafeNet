# Figure 1 ADR-output schematic guide

## Purpose

The schematic fills the space between the DOT-SafeNet model block and the human-body ADR illustration. It shows two model outputs: exposure-dependent SOC risk curves and target-level attribution.

## Files

- `scripts/render_adr_output_schematic.py`: drawing and preview composition.
- `params.yaml`: dimensions, colors, and placement.
- `assets/figure1_framework_reference.png`: user-provided Figure 1 draft.
- `assets/figure5_case_reference.png`: user-provided case-figure reference.

## Rendering

Run `python run.py` from this directory.

## Outputs

- `outputs/figure1_adr_output_schematic.*`: standalone insert.
- `outputs/figure1_framework_with_adr_output_preview.png`: placement preview.

The miniature curves and bars are schematic and do not encode measured values.
