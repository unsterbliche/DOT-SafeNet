# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) clustered target-level evidence heatmap -> CorrHeatmap -> param inherit
# RULE: "param inherit" = drawing function below that copies Class A/B/C values.

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SOC_ORDER = ["BLO","CAR","EAR","END","EYE","GAS","HEP","IMM","INF","MET","MUS","NER","PSY","REN","REP","RES","SKI","VAS"]
SELECTED = ["GAS", "CAR", "REP", "VAS", "REN"]
DIRECT, SECONDARY = "#2A6FBB", "#E69F54"
CASE_GENES = {"PTGS1", "KCNH2", "AR", "AGTR1"}

m = pd.read_csv(DATA / "soc_target_membership.csv")
m["etype"] = np.where(m["association_class"].str.contains("secondary", case=False, na=False), "secondary", "direct")
m["priority"] = m["etype"].map({"direct": 0, "secondary": 1})
m = m.sort_values("priority").drop_duplicates(["soc_abbr", "target_uniprot"], keep="first")
m["value"] = m["etype"].map({"secondary": 1, "direct": 2})
gene_map = m.drop_duplicates("target_uniprot").set_index("target_uniprot")["human_gene"].to_dict()
targets = sorted(m["target_uniprot"].unique())
mat = pd.DataFrame(0, index=SOC_ORDER, columns=targets, dtype=int)
for row in m.itertuples():
    mat.loc[row.soc_abbr, row.target_uniprot] = row.value

profiles = mat.T.to_numpy(dtype=float)
profiles[profiles == 1] = 0.38
profiles[profiles == 2] = 1.0
d = pdist(profiles, metric="cosine")
d[~np.isfinite(d)] = 1.0
mat = mat.iloc[:, leaves_list(linkage(d, method="average"))]
rd = pdist((mat.to_numpy() > 0).astype(float), metric="jaccard")
mat = mat.iloc[leaves_list(linkage(rd, method="average")), :]
mat.to_csv(DATA / "figure5a_clustered_complete_target_matrix.csv")

cmap = ListedColormap(["#FFFFFF", SECONDARY, DIRECT])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

def draw(matrix, stem, selected=False):
    fig, ax = plt.subplots(figsize=(7.2, 2.7 if selected else 4.2))
    ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, rasterized=True)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for label in ax.get_yticklabels():
        if label.get_text() in set(SELECTED):
            label.set_fontweight("bold")
            label.set_color("#7B2D36")
    support = (matrix > 0).sum(axis=0)
    threshold = 3 if selected else 5
    labels = []
    for i, accession in enumerate(matrix.columns):
        gene = str(gene_map.get(accession, accession))
        if gene in CASE_GENES or int(support.iloc[i]) >= threshold:
            labels.append((i, gene))
    ax.set_xticks([i for i, _ in labels])
    ax.set_xticklabels([g for _, g in labels], rotation=90, ha="center", va="top", fontsize=5.2 if selected else 4.7)
    for label in ax.get_xticklabels():
        if label.get_text() in CASE_GENES:
            label.set_color("#B2182B")
            label.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=4)
    for y in np.arange(0.5, len(matrix.index), 1):
        ax.axhline(y, color="#E1E4E8", lw=0.35, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#555555"); spine.set_linewidth(0.65)
    ax.set_xlim(-0.5, matrix.shape[1]-0.5)
    ax.set_ylim(matrix.shape[0]-0.5, -0.5)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.985, bottom=0.29 if selected else 0.24)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Full association atlas.
draw(mat, "figure5a_target_bicluster_all_soc", selected=False)
# Enlarged manuscript candidate for the selected clinical SOCs.
selected_mat = mat.loc[SELECTED]
selected_mat = selected_mat.loc[:, (selected_mat > 0).any(axis=0)]
sp = selected_mat.T.to_numpy(dtype=float)
sp[sp == 1] = 0.38
sp[sp == 2] = 1.0
sd = pdist(sp, metric='cosine')
sd[~np.isfinite(sd)] = 1.0
selected_mat = selected_mat.iloc[:, leaves_list(linkage(sd, method='average'))]
selected_mat.to_csv(DATA / "figure5a_clustered_selected_soc_target_matrix.csv")
draw(selected_mat, "figure5a_target_bicluster_selected_soc", selected=True)

fig, ax = plt.subplots(figsize=(3.0, 0.35)); ax.axis("off")
handles = [Line2D([0],[0],marker="s",linestyle="",markerfacecolor=DIRECT,markeredgecolor="none",markersize=7,label="Direct evidence"), Line2D([0],[0],marker="s",linestyle="",markerfacecolor=SECONDARY,markeredgecolor="none",markersize=7,label="Secondary evidence")]
ax.legend(handles=handles, loc="center", ncol=2, columnspacing=1.6, handletextpad=0.5)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"figure5a_target_matrix_legend.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)