"""Render the compact ADR-model output schematic used in Figure 1.

Asset confirmation
------------------
The framework and case-plot images supplied by the author were inspected before
design.  The output insert reuses their coral/blue/purple visual language.  All
curves and attribution magnitudes in this file are schematic and carry no
quantitative claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
from PIL import Image, ImageDraw
import yaml


# Typography, palette, and export baselines follow the project figure standard.
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def mm_to_inch(value: float) -> float:
    return value / 25.4


def rounded_card(ax, xy, width, height, *, facecolor, edgecolor, radius=0.035,
                 linewidth=1.0, shadow=False, zorder=1):
    x, y = xy
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x + 0.012, y - 0.015), width, height,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            transform=ax.transAxes, facecolor="#D7B8AE", edgecolor="none",
            alpha=0.22, zorder=zorder - 0.5,
        ))
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_arrowhead(ax, x, y, color, size=0.015, zorder=5):
    ax.add_patch(Polygon(
        [[x, y], [x - size, y + size * 0.55], [x - size, y - size * 0.55]],
        closed=True, transform=ax.transAxes, facecolor=color,
        edgecolor=color, zorder=zorder,
    ))


def render_insert(config: dict, root: Path) -> Path:
    c = config["colors"]
    fig_cfg = config["figure"]
    fig = plt.figure(
        figsize=(mm_to_inch(fig_cfg["width_mm"]), mm_to_inch(fig_cfg["height_mm"])),
        dpi=fig_cfg["dpi"], facecolor="none",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_card(
        ax, (0.025, 0.025), 0.95, 0.95,
        facecolor=c["peach"], edgecolor=c["coral"],
        radius=0.045, linewidth=1.2, shadow=True,
    )

    # Upper card: several SOC-specific dose-response curves.
    rounded_card(ax, (0.075, 0.545), 0.85, 0.385,
                 facecolor="white", edgecolor="#E4C8C1", radius=0.025,
                 linewidth=0.8)
    ax.text(0.115, 0.885, "Dose-dependent SOC risk", fontsize=8.3,
            fontweight="bold", color="#303236", va="center")
    x0, x1 = 0.16, 0.80
    y0, y1 = 0.615, 0.835
    axis_color = "#7F858B"
    ax.plot([x0, x1], [y0, y0], color=axis_color, lw=0.8, clip_on=False)
    ax.plot([x0, x0], [y0, y1], color=axis_color, lw=0.8, clip_on=False)
    draw_arrowhead(ax, x1 + 0.015, y0, axis_color, size=0.013)
    ax.add_patch(Polygon([[x0, y1 + 0.018], [x0 - 0.009, y1 + 0.003], [x0 + 0.009, y1 + 0.003]], closed=True, transform=ax.transAxes, facecolor=axis_color, edgecolor=axis_color, zorder=5))

    xs = np.linspace(x0 + 0.02, x1 - 0.02, 6)
    curves = [
        (np.array([0.18, 0.30, 0.49, 0.63, 0.72, 0.78]), c["blue"], "CAR"),
        (np.array([0.30, 0.44, 0.61, 0.76, 0.87, 0.94]), c["purple"], "NER"),
        (np.array([0.10, 0.18, 0.31, 0.46, 0.57, 0.65]), c["coral"], "GAS"),
    ]
    for values, color, label in curves:
        ys = y0 + values * (y1 - y0)
        ax.plot(xs, ys, color=color, lw=1.7, solid_capstyle="round", zorder=4)
        ax.scatter(xs, ys, s=9, color=color, edgecolor="white", linewidth=0.35,
                   zorder=5)
        ax.text(x1 + 0.035, ys[-1], label, color=color, fontsize=7.2,
                fontweight="bold", va="center")
    ax.text((x0 + x1) / 2, y0 - 0.055, "Free exposure", ha="center",
            va="center", fontsize=7.1, color="#4B4F54")
    ax.text(x0 + 0.01, y0 - 0.025, "low", ha="left", va="top",
            fontsize=6.2, color="#8A9096")
    ax.text(x1 - 0.01, y0 - 0.025, "high", ha="right", va="top",
            fontsize=6.2, color="#8A9096")
    ax.text(x0 - 0.055, (y0 + y1) / 2, "ADR score", rotation=90,
            ha="center", va="center", fontsize=7.1, color="#4B4F54")

    # Lower card: target-replacement attribution summary.
    rounded_card(ax, (0.075, 0.105), 0.85, 0.335,
                 facecolor="white", edgecolor="#E4C8C1", radius=0.025,
                 linewidth=0.8)
    ax.text(0.115, 0.400, "Target attribution", fontsize=8.3,
            fontweight="bold", color="#303236", va="center")
    labels = ["Key target", "ADR-linked", "Other"]
    colors = [c["coral_dark"], c["blue"], c["grey"]]
    lengths = [0.62, 0.44, 0.25]
    ys = [0.335, 0.273, 0.211]
    bar_x0, bar_x1 = 0.36, 0.83
    for label, color, length, yy in zip(labels, colors, lengths, ys):
        ax.text(0.115, yy, label, fontsize=7.0, color="#4B4F54", va="center")
        ax.plot([bar_x0, bar_x1], [yy, yy], color=c["light_grey"], lw=4.2,
                solid_capstyle="round", zorder=2)
        x_end = bar_x0 + length * (bar_x1 - bar_x0)
        ax.plot([bar_x0, x_end], [yy, yy], color=color, lw=2.1,
                solid_capstyle="round", zorder=3)
        if label == "Key target":
            ax.scatter([x_end], [yy], marker="D", s=25, color=color,
                       edgecolor="white", linewidth=0.5, zorder=4)
        else:
            ax.scatter([x_end], [yy], s=14, color=color, edgecolor="white",
                       linewidth=0.4, zorder=4)
    ax.text(0.595, 0.145, "Change in ADR score after target replacement",
            ha="center", va="center", fontsize=6.5, color="#666B70")

    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "figure1_adr_output_schematic"
    fig.savefig(stem.with_suffix(".png"), dpi=fig_cfg["dpi"], transparent=True,
                bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".pdf"), transparent=True,
                bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".svg"), transparent=True,
                bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return stem.with_suffix(".png")


def compose_preview(config: dict, root: Path, insert_path: Path) -> Path:
    source = Image.open(root / "assets" / "figure1_framework_reference.png").convert("RGBA")
    insert = Image.open(insert_path).convert("RGBA")
    p = config["preview"]
    lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
    insert.thumbnail((int(p["width_px"]), int(p["height_px"])), lanczos)
    x = int(p["x_px"] + (p["width_px"] - insert.width) / 2)
    y = int(p["y_px"] + (p["height_px"] - insert.height) / 2)
    source.alpha_composite(insert, (x, y))

    # A restrained output arrow links the card to the anatomical ADR summary.
    draw = ImageDraw.Draw(source)
    coral = config["colors"]["coral"]
    start_x = x + insert.width + 3
    cy = y + insert.height // 2
    end_x = min(source.width - 185, start_x + 32)
    if end_x > start_x:
        draw.line((start_x, cy, end_x, cy), fill=coral, width=4)
        draw.polygon([(end_x + 9, cy), (end_x, cy - 6), (end_x, cy + 6)], fill=coral)

    output = root / "outputs" / "figure1_framework_with_adr_output_preview.png"
    source.save(output)
    return output


def write_qa(root: Path, insert: Path, preview: Path) -> None:
    text = """# QA report

- The insert contains no subplot letter and uses a compact two-level layout.
- Dose-response curves and attribution bars are schematic; they are not model estimates.
- The standalone SVG and PDF retain editable vector text and geometry.
- The transparent PNG can be placed directly in the existing Figure 1 artwork.
- The placement preview preserves the supplied framework artwork and only adds the insert and output arrow.
"""
    (root / "outputs" / "QA_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.config.resolve().parent
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    insert = render_insert(config, root)
    preview = compose_preview(config, root, insert)
    write_qa(root, insert, preview)
    print(insert)
    print(preview)


if __name__ == "__main__":
    main()
