# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) module-ordered target evidence matrix -> CorrHeatmap -> param inherit
# (a, top) per-target SOC counts -> Bar -> param inherit
# RULE: "param inherit" = drawing functions below copy Class A/B/C values.

from pathlib import Path
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "figure.titlesize": 9, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.dpi": 300,
})
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DIRECT = "#075AA6"
SECONDARY = "#E9A04F"
TEXT = "#252525"
CASE = "#B2182B"
MODULE_ORDER = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","Unassigned"]
MODULE_LABEL = {**{f"M{i}": f"M{i}" for i in range(1,10)}, "Unassigned": "U"}
SOC_ORDER = ["CAR","VAS","REN","NER","PSY","GAS","HEP","MET","END","IMM","BLO","INF","RES","REP","EYE","EAR","SKI","MUS"]
SELECTED_SOC = set()
CASE_GENES = set()

m = pd.read_csv(DATA / "soc_target_membership.csv")
m["etype"] = np.where(m["association_class"].str.contains("secondary", case=False, na=False), "secondary", "direct")
m["priority"] = m["etype"].map({"direct":0,"secondary":1})
m = m.sort_values("priority").drop_duplicates(["soc_abbr","target_uniprot"], keep="first")
m["value"] = m["etype"].map({"secondary":1,"direct":2})
target_info = m.drop_duplicates("target_uniprot").set_index("target_uniprot")[["human_gene","community_module"]]
targets = target_info.index.tolist()
mat = pd.DataFrame(0, index=SOC_ORDER, columns=targets, dtype=int)
for row in m.itertuples():
    mat.loc[row.soc_abbr, row.target_uniprot] = row.value

# Preserve M1-M9+U blocks and cluster target profiles only within each module.
ordered_targets = []
module_ranges = []
start = 0
for module in MODULE_ORDER:
    members = target_info.index[target_info["community_module"].eq(module)].tolist()
    if len(members) > 1:
        profile = mat[members].T.to_numpy(dtype=float)
        profile[profile == 1] = 0.38
        profile[profile == 2] = 1.0
        distance = pdist(profile, metric="cosine")
        distance[~np.isfinite(distance)] = 1.0
        members = [members[i] for i in leaves_list(linkage(distance, method="average"))]
    ordered_targets.extend(members)
    module_ranges.append((module, start, start + len(members)))
    start += len(members)
mat = mat[ordered_targets]

genes = [str(target_info.loc[t,"human_gene"]) for t in ordered_targets]
order_table = pd.DataFrame({
    "column_index": np.arange(len(ordered_targets)),
    "target_uniprot": ordered_targets,
    "human_gene": genes,
    "community_module": [target_info.loc[t,"community_module"] for t in ordered_targets],
    "direct_soc_count": (mat == 2).sum(axis=0).to_numpy(),
    "secondary_soc_count": (mat == 1).sum(axis=0).to_numpy(),
})
OUT.mkdir(parents=True, exist_ok=True)
order_table.to_csv(OUT / "figure_4a_target_columns.csv", index=False)
mat.to_csv(OUT / "figure_4a_evidence_matrix.csv")

fig = plt.figure(figsize=(7.8, 2.60))
gs = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.30], hspace=0.04)
ax_module = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_module)

# Alternating neutral module shading preserves grouping without a rainbow palette.
module_fills = ["#E9F0F5", "#F5F3EE"]
for idx, (module, lo, hi) in enumerate(module_ranges):
    fill = module_fills[idx % 2]
    ax_module.axvspan(lo-0.5, hi-0.5, color=fill, lw=0)
    ax.axvspan(lo-0.5, hi-0.5, color=fill, alpha=0.15, lw=0, zorder=0)
    ax_module.text((lo+hi-1)/2, 0.5, MODULE_LABEL[module], ha="center", va="center",
                   fontsize=7.5, fontweight="bold", color="#40566B")
    if hi < len(ordered_targets):
        ax.axvline(hi-0.5, color="#B8C0C7", lw=0.45, zorder=5)
ax_module.set_ylim(0,1); ax_module.axis("off")

# Highlight the ADR rows used to define each M1-M9 module.
adr_block_rows = {
    "M1": (0, 1), "M2": (1, 3), "M3": (3, 5),
    "M4": (5, 7), "M5": (7, 9), "M6": (9, 12),
    "M7": (12, 13), "M8": (13, 14), "M9": (14, 18),
}
for module, lo, hi in module_ranges:
    if module not in adr_block_rows:
        continue
    row_lo, row_hi = adr_block_rows[module]
    ax.add_patch(Rectangle(
        (lo - 0.5, row_lo - 0.5), hi - lo, row_hi - row_lo,
        facecolor="#C9DDEA", edgecolor="#4F7896", alpha=0.55,
        linewidth=1.00, zorder=1,
    ))

x = np.arange(len(ordered_targets))
# Direct evidence is saturated; secondary evidence uses a translucent block.
evidence_rgba = np.zeros((mat.shape[0], mat.shape[1], 4), dtype=float)
evidence_rgba[mat.to_numpy() == 2] = mpl.colors.to_rgba(DIRECT, alpha=1.0)
evidence_rgba[mat.to_numpy() == 1] = mpl.colors.to_rgba(SECONDARY, alpha=0.48)
ax.imshow(evidence_rgba, aspect="auto", interpolation="nearest", zorder=2)

ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=5.4)
for label in ax.get_yticklabels():
    if label.get_text() in SELECTED_SOC:
        label.set_color("#7B2D36"); label.set_fontweight("bold")
ax.set_xticks(x); ax.set_xticklabels(genes, rotation=90, ha="center", va="top", fontsize=3.7)
for label in ax.get_xticklabels():
    if label.get_text() in CASE_GENES:
        label.set_color(CASE); label.set_fontweight("bold"); label.set_fontsize(4.4)
ax.tick_params(axis="x", length=0, pad=1.5)
ax.tick_params(axis="y", length=0, pad=4)
for y in np.arange(0.5, len(mat.index), 1):
    ax.axhline(y, color="#DCE1E5", lw=0.35, zorder=4)
for y in [0.5, 2.5, 4.5, 6.5, 8.5, 11.5, 12.5, 13.5]:
    ax.axhline(y, color="#C4CBD1", lw=0.45, zorder=5)
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(0.65); spine.set_color("#555555")
ax.set_xlim(-0.5, len(ordered_targets)-0.5)
ax.set_ylim(len(mat.index)-0.5, -0.5)

fig.subplots_adjust(left=0.075, right=0.995, top=0.975, bottom=0.405)
for ext in ["png","pdf","svg"]:
    fig.savefig(OUT / f"figure_4a_adr_target_association.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# Keep the dense panel clean by exporting its evidence legend separately.
legend, lax = plt.subplots(figsize=(3.0,0.34)); lax.axis("off")
handles = [
    Line2D([0],[0],marker="s",ls="",mfc=DIRECT,mec="none",ms=7,label="Direct evidence"),
    Line2D([0],[0],marker="s",ls="",mfc=SECONDARY,alpha=0.48,mec="none",ms=7,label="Secondary evidence"),
]
lax.legend(handles=handles, loc="center", ncol=2, columnspacing=1.5, handletextpad=0.45)
for ext in ["png","pdf","svg"]:
    legend.savefig(OUT / f"figure_4a_adr_target_legend.{ext}", dpi=300, bbox_inches="tight")
plt.close(legend)
