# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) model schematic -> user-provided prior HetSia architecture -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import yaml


def mm(value):
    return value / 25.4


def box(ax, x, y, w, h, fc, ec, text="", *, fs=7.5, weight="normal",
        radius=0.018, lw=0.9, color=BLACK, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=color, linespacing=1.15,
                zorder=z + 1)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#737A80", lw=1.25, mutation=10, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z,
    ))


def feature_strip(ax, x, y, w, h, colors):
    widths = [0.27, 0.365, 0.365]
    labels = [r"log C$_{free}$", r"pAC$_{50}$ ×194", "logratio ×194"]
    fills = ["#F1A340", colors["drug"], "#7BC7C5"]
    pos = x
    for frac, label, fill in zip(widths, labels, fills):
        ww = w * frac
        ax.add_patch(Rectangle((pos, y), ww, h, facecolor=fill,
                               edgecolor="white", linewidth=0.65, zorder=3))
        ax.text(pos + ww / 2, y + h / 2, label, ha="center", va="center",
                fontsize=5.7, color="white", fontweight="bold", zorder=4)
        pos += ww


def output_stack(ax, x, y, w, h, colors):
    for dx, dy, alpha in [(0.014, 0.014, 0.36), (0.007, 0.007, 0.62), (0, 0, 1.0)]:
        box(ax, x + dx, y + dy, w, h, colors["head_light"], colors["head"],
            radius=0.012, lw=0.9, z=2)
    ax.text(x + w / 2, y + h * 0.61, r"P(ADR$_a$)", ha="center",
            va="center", fontsize=7.5, fontweight="bold", color=colors["head"])
    ax.text(x + w / 2, y + h * 0.30, "18 SOCs", ha="center", va="center",
            fontsize=6.6, color=colors["grey"])


def render(config_path: Path):
    root = config_path.resolve().parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fcfg, c = cfg["figure"], cfg["colors"]
    fig = plt.figure(figsize=(mm(fcfg["width_mm"]), mm(fcfg["height_mm"])),
                     dpi=fcfg["dpi"], facecolor="white")
    ax = fig.add_axes([0.012, 0.035, 0.976, 0.93])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Branch bands preserve the visual identity of the supplied architecture.
    ax.add_patch(Rectangle((0.005, 0.545), 0.018, 0.375,
                           facecolor=c["drug"], edgecolor="none"))
    ax.add_patch(Rectangle((0.005, 0.145), 0.018, 0.285,
                           facecolor=c["soc"], edgecolor="none"))
    ax.text(0.014, 0.732, "Drug d", rotation=90, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="white")
    ax.text(0.014, 0.287, "SOC a", rotation=90, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="white")

    # Drug-profile input.
    box(ax, 0.045, 0.575, 0.225, 0.295, "#F7FCFC", c["drug"], radius=0.022, lw=1.0)
    ax.text(0.058, 0.835, "Drug profile", fontsize=8.2, fontweight="bold",
            color=c["drug"], ha="left")
    feature_strip(ax, 0.060, 0.728, 0.195, 0.065, c)
    ax.text(0.157, 0.690, "389 input features", fontsize=6.7,
            color=c["grey"], ha="center")
    ax.text(0.157, 0.635,
            r"logratio$_i$ = log$_{10}$ C$_{max,free}$ + pAC$_{50,i}$ − 6",
            fontsize=6.0, color=c["grey"], ha="center")
    ax.text(0.157, 0.594, "OT-ProfileNet + exposure models", fontsize=6.3,
            color=c["drug"], ha="center", fontweight="bold")

    # Drug encoder.
    arrow(ax, 0.278, 0.722, 0.295, 0.722, c["drug"])
    box(ax, 0.300, 0.630, 0.145, 0.185, c["drug_light"], c["drug"],
        "Drug encoder\n389–512–256–128", fs=7.5, weight="bold")
    ax.text(0.452, 0.722, r"$h_d$", fontsize=8.2, color=c["drug"],
            fontweight="bold", va="center")

    # SOC branch.
    box(ax, 0.050, 0.205, 0.105, 0.105, c["soc_light"], c["soc"],
        "SOC index\n1–18", fs=7.3, weight="bold")
    arrow(ax, 0.162, 0.257, 0.198, 0.257, c["soc"])
    box(ax, 0.204, 0.205, 0.110, 0.105, c["soc_light"], c["soc"],
        "Trainable\nembedding 512", fs=7.0)
    arrow(ax, 0.322, 0.257, 0.332, 0.257, c["soc"])
    box(ax, 0.338, 0.175, 0.135, 0.165, c["soc_light"], c["soc"],
        "ADR encoder\n512–256–128", fs=7.5, weight="bold")
    ax.text(0.480, 0.257, r"$h_a$", fontsize=8.2, color=c["soc"],
            fontweight="bold", va="center")

    # Pair representation.
    arrow(ax, 0.473, 0.708, 0.517, 0.565, c["drug"], lw=1.1)
    arrow(ax, 0.496, 0.270, 0.517, 0.445, c["soc"], lw=1.1)
    box(ax, 0.513, 0.430, 0.120, 0.150, "#EEEAF8", c["pair"],
        r"Pair representation" + "\n" + r"$|h_d-h_a|$", fs=7.5, weight="bold")
    arrow(ax, 0.638, 0.505, 0.660, 0.505, c["pair"])

    # Pair MLP and task-specific head.
    box(ax, 0.666, 0.425, 0.105, 0.160, c["training_light"], "#C8A933",
        "Pair MLP\n64–32–16–8", fs=7.4, weight="bold")
    arrow(ax, 0.776, 0.505, 0.795, 0.505, c["head"])
    box(ax, 0.800, 0.425, 0.095, 0.160, c["head_light"], c["head"],
        r"SOC head" + "\n" + r"$w_a^Th+b_a$" + "\nsigmoid", fs=6.7, weight="bold")
    arrow(ax, 0.900, 0.505, 0.920, 0.505, c["head"])
    output_stack(ax, 0.925, 0.437, 0.060, 0.136, c)

    # Training supervision: separate from inference architecture.
    box(ax, 0.513, 0.085, 0.268, 0.245, c["training_light"], "#D2B64B",
        radius=0.020, lw=0.9)
    ax.text(0.533, 0.294, "Training objective", fontsize=7.6,
            fontweight="bold", color="#78650F")
    ax.text(0.533, 0.245, "BCE: real + soft pseudo-dose labels", fontsize=6.7,
            color=BLACK)
    ax.text(0.533, 0.195, r"Monotone: 0.5·ReLU($p_{low}-p_{high}$)", fontsize=6.7,
            color=BLACK)
    ax.plot([0.541, 0.589], [0.134, 0.134], color=c["head"], lw=1.6)
    ax.scatter([0.541, 0.589], [0.134, 0.134], s=12,
               color=[c["head_light"], c["head"]], edgecolor=c["head"], linewidth=0.6)
    ax.text(0.603, 0.134, "same drug/SOC, low → high dose", fontsize=6.2,
            color=c["grey"], va="center")
    arrow(ax, 0.718, 0.420, 0.680, 0.333, "#C8A933", lw=0.8, mutation=7)

    box(ax, 0.805, 0.115, 0.175, 0.165, "#FDECEF", c["soc"],
        "CT-ADE fine-tuning\nSOC heads only\n\n5-fold ensemble", fs=6.9,
        weight="bold", radius=0.018, lw=0.9)
    arrow(ax, 0.850, 0.288, 0.850, 0.417, c["soc"], lw=0.9, mutation=8)

    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "figure_4b_dotsafenet_architecture"
    save_cns_figure(fig, str(stem))
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    qa = """# DOT-SafeNet architecture schematic QA

- The architecture follows the selected A2-mono-lite clinical model: 389 drug features, 18-SOC embedding branch, absolute-difference pair representation, 64-32-16-8 pair MLP, and SOC-specific sigmoid head.
- Soft pseudo-dose labels, pairwise monotonicity and CT-ADE fine-tuning are displayed as training procedures outside the inference architecture.
- Curves or quantitative values are not shown; the diagram carries no simulated result.
- Vector PDF and SVG and a 300-dpi PNG preview were exported.
"""
    (out / "figure_4b_dotsafenet_architecture_QA.md").write_text(qa, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    render(args.config)


if __name__ == "__main__":
    main()
