from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

def read_splits(folder):
    frames = []
    counts = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(folder / f"{split}.csv")
        frame["split"] = split
        frames.append(frame)
        counts.append((split, len(frame)))
    return pd.concat(frames, ignore_index=True), counts

def full_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.tick_params(direction="out", length=3, width=0.8)

def save(fig, stem, cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in cfg["export"]["formats"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=cfg["figure"]["dpi"],
                    bbox_inches="tight", transparent=cfg["export"]["transparent"])

def histogram(values, xlabel, title, color, stem, cfg, bins=10):
    fig, ax = plt.subplots(figsize=cfg["figure"]["panel_size"])
    ax.hist(values, bins=bins, color=color, edgecolor="#333333", linewidth=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel("Number of compounds")
    ax.set_title(title, pad=5)
    full_frame(ax)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.20)
    save(fig, stem, cfg)
    plt.close(fig)

def dose_histogram(cmax, cfg):
    edges = np.array([0, 10, 50, 100, 250, 500, 1000, 2000, 5000, 22000], float)
    labels = ["[0, 10)", "[10, 50)", "[50, 100)", "[100, 250)", "[250, 500)",
              "[500, 1000)", "[1000, 2000)", "[2000, 5000)", "[5000, 22000]"]
    categories = pd.cut(cmax["dose"], bins=edges, right=False, include_lowest=True, labels=labels)
    counts = categories.value_counts(sort=False)
    fig, ax = plt.subplots(figsize=(4.5, 2.55))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts.to_numpy(), width=0.74, color="#8CBBD2",
                  edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("Count"); ax.set_xlabel("Normalized dose (mg)")
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(counts) * 0.015,
                str(int(value)), ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(counts) * 1.13)
    full_frame(ax)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.96, bottom=0.34)
    save(fig, "normalized_dose_intervals", cfg)
    plt.close(fig)
    return pd.DataFrame({"interval": labels, "count": counts.to_numpy()})

def dataset_table(ppb_counts, cmax_counts, cfg):
    rows = [["PPB", split.capitalize(), n] for split, n in ppb_counts]
    rows += [["Cmax", split.capitalize(), n] for split, n in cmax_counts]
    table = pd.DataFrame(rows, columns=["Data set", "Split", "N samples"])
    fig, ax = plt.subplots(figsize=cfg["figure"]["table_size"])
    ax.axis("off")
    artist = ax.table(cellText=table.values, colLabels=table.columns, loc="center",
                      cellLoc="center", colLoc="center", colWidths=[0.28, 0.34, 0.30])
    artist.auto_set_font_size(False); artist.set_fontsize(9.5); artist.scale(1, 1.42)
    for (row, col), cell in artist.get_celld().items():
        cell.set_edgecolor("#333333"); cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#D9D9D9"); cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F3F3")
    save(fig, "dataset_sample_counts", cfg)
    plt.close(fig)
    return table

def load_chembl_smiles():
    files = sorted((DATA / "ot_profile").glob("*/*.csv"))
    if not files:
        raise FileNotFoundError("OT-ProfileNet activity CSV files were not found")
    values = []
    for path in files:
        frame = pd.read_csv(path, usecols=["canonical_smiles"])
        values.extend(frame["canonical_smiles"].dropna().astype(str).tolist())
    return pd.Index(values).drop_duplicates().to_numpy()

def fingerprints(smiles, cfg):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg["tsne"]["radius"], fpSize=cfg["tsne"]["fingerprint_bits"])
    arrays, valid = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((cfg["tsne"]["fingerprint_bits"],), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arrays.append(arr); valid.append(str(smi))
    return np.asarray(arrays, dtype=np.float32), valid

def tsne_coordinates(ppb, cmax, cfg):
    cache = OUT / "chemical_space_tsne_coordinates.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng(cfg["tsne"]["seed"])
    chembl = load_chembl_smiles()
    if len(chembl) > cfg["tsne"]["chembl_sample"]:
        chembl = rng.choice(chembl, size=cfg["tsne"]["chembl_sample"], replace=False)
    ppb_smiles = pd.Index(ppb["smiles"].dropna().astype(str)).drop_duplicates().to_numpy()
    cmax_smiles = pd.Index(cmax["smiles"].dropna().astype(str)).drop_duplicates().to_numpy()
    all_smiles = pd.Index(np.concatenate([chembl, ppb_smiles, cmax_smiles])).drop_duplicates().to_numpy()
    matrix, valid = fingerprints(all_smiles, cfg)
    components = min(cfg["tsne"]["pca_components"], matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=components, svd_solver="randomized",
                  random_state=cfg["tsne"]["seed"]).fit_transform(matrix)
    coords = TSNE(n_components=2, perplexity=cfg["tsne"]["perplexity"],
                  n_iter=cfg["tsne"]["iterations"], init="pca", learning_rate="auto",
                  random_state=cfg["tsne"]["seed"], method="barnes_hut").fit_transform(reduced)
    chembl_set, ppb_set, cmax_set = set(chembl), set(ppb_smiles), set(cmax_smiles)
    result = pd.DataFrame({
        "smiles": valid, "TSNE_1": coords[:, 0], "TSNE_2": coords[:, 1],
        "in_chembl": [s in chembl_set for s in valid],
        "in_ppb": [s in ppb_set for s in valid],
        "in_cmax": [s in cmax_set for s in valid],
    })
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False, compression="gzip")
    return result

def tsne_panel(coords, task_col, task_label, stem, cfg):
    background = coords.loc[coords["in_chembl"]]
    task = coords.loc[coords[task_col]]
    fig, ax = plt.subplots(figsize=cfg["figure"]["tsne_size"])
    ax.scatter(background["TSNE_1"], background["TSNE_2"], s=3.2,
               color=cfg["colors"]["chembl"], alpha=0.52, edgecolors="none", rasterized=True)
    ax.scatter(task["TSNE_1"], task["TSNE_2"], s=5.0,
               color=cfg["colors"]["task"], alpha=0.76, edgecolors="none", rasterized=True)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    full_frame(ax)
    fig.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cfg["colors"]["chembl"],
               markeredgecolor="none", markersize=4.5, label="ChEMBL"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cfg["colors"]["task"],
               markeredgecolor="none", markersize=4.5, label=task_label),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.25)
    save(fig, stem, cfg)
    plt.close(fig)

def trim(image):
    rgb = image.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).convert("L")
    box = diff.point(lambda v: 255 if v > 8 else 0).getbbox()
    return rgb.crop(box) if box else rgb

def make_composite(cfg):
    names = [
        ("ppb_distribution", "unbound_fraction_log10_distribution"),
        ("cmax_distribution", "cmax_log10_distribution"),
        ("normalized_dose_intervals", "dataset_sample_counts"),
        ("chemical_space_ppb", "chemical_space_cmax"),
    ]
    width, row_h, margin, gap = 2160, 620, 40, 28
    canvas = Image.new("RGB", (width, margin*2 + len(names)*row_h + (len(names)-1)*gap), "white")
    tile_w = (width - 2*margin - gap)//2
    for row, pair in enumerate(names):
        for col, name in enumerate(pair):
            image = trim(Image.open(OUT / f"{name}.png"))
            fitted = ImageOps.contain(image, (tile_w, row_h), Image.LANCZOS)
            x0 = margin + col*(tile_w+gap)
            y0 = margin + row*(row_h+gap)
            canvas.paste(fitted, (x0+(tile_w-fitted.width)//2, y0+(row_h-fitted.height)//2))
    canvas.save(OUT / "supplementary_ppb_cmax_dataset_characteristics.png", dpi=(cfg["figure"]["dpi"],)*2)
    canvas.save(OUT / "supplementary_ppb_cmax_dataset_characteristics.pdf", "PDF", resolution=cfg["figure"]["dpi"])
    canvas.save(OUT / "latest.png", dpi=(cfg["figure"]["dpi"],)*2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cfg["text"]["font_family"], "Helvetica", "DejaVu Sans"],
        "font.size": cfg["text"]["font_size"], "axes.titlesize": cfg["text"]["title_size"],
        "axes.labelsize": cfg["text"]["label_size"], "xtick.labelsize": cfg["text"]["tick_size"],
        "ytick.labelsize": cfg["text"]["tick_size"], "legend.fontsize": cfg["text"]["legend_size"],
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    ppb, ppb_counts = read_splits(DATA / "ppb" / "ppb_random_split")
    cmax, cmax_counts = read_splits(DATA / "cmax" / "cmax_random_split")
    ppb_percent = (1 - np.power(10.0, ppb["activity"].to_numpy(float))) * 100
    cmax_value = np.power(10.0, cmax["activity"].to_numpy(float))
    histogram(ppb_percent, "PPB (%)", "PPB distribution", cfg["colors"]["ppb"],
              "ppb_distribution", cfg, bins=np.arange(0, 110, 10))
    histogram(ppb["activity"], r"$\log_{10} f_{u}$", "Unbound fraction distribution",
              cfg["colors"]["ppb"], "unbound_fraction_log10_distribution", cfg, bins=10)
    histogram(cmax_value, r"Cmax ($\mu$g mL$^{-1}$)", "Cmax distribution",
              cfg["colors"]["cmax"], "cmax_distribution", cfg, bins=10)
    histogram(cmax["activity"], r"$\log_{10}$ Cmax ($\mu$g mL$^{-1}$)",
              "Log-transformed Cmax distribution", cfg["colors"]["cmax"],
              "cmax_log10_distribution", cfg, bins=10)
    dose_counts = dose_histogram(cmax, cfg)
    sample_counts = dataset_table(ppb_counts, cmax_counts, cfg)
    coords = tsne_coordinates(ppb, cmax, cfg)
    tsne_panel(coords, "in_ppb", "PPB", "chemical_space_ppb", cfg)
    tsne_panel(coords, "in_cmax", "Cmax", "chemical_space_cmax", cfg)
    dose_counts.to_csv(OUT / "normalized_dose_interval_counts.csv", index=False)
    sample_counts.to_csv(OUT / "dataset_sample_counts.csv", index=False)
    make_composite(cfg)

if __name__ == "__main__":
    main()
