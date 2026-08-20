from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROFILE_COLUMNS = ["compound_id", "name", "smiles", "dose_mg_day", "route"]


@dataclass(frozen=True)
class Contract:
    target_ids: tuple[str, ...]
    soc_codes: tuple[str, ...]
    soc_names: tuple[str, ...]

    @property
    def interaction_columns(self) -> tuple[str, ...]:
        return tuple(f"{target}*Cmax" for target in self.target_ids)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return ("cmax_free(uM)",) + self.target_ids + self.interaction_columns


def load_contract(release_root: Path) -> Contract:
    target = pd.read_csv(release_root / "data/model/target_order.csv")
    soc = pd.read_csv(release_root / "data/model/soc_order.csv")
    target_ids = tuple(target["uniprot_accession"].astype(str))
    soc_codes = tuple(soc["soc_code"].astype(str))
    soc_names = tuple(soc["soc_name"].astype(str))
    if len(target_ids) != 194 or len(set(target_ids)) != 194:
        raise ValueError("target_order.csv must contain 194 unique targets")
    if len(soc_codes) != 18 or len(set(soc_codes)) != 18:
        raise ValueError("soc_order.csv must contain 18 unique SOC tasks")
    return Contract(target_ids, soc_codes, soc_names)


def _canonicalize(smiles: str) -> tuple[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise RuntimeError("RDKit is required for full prediction") from exc
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonical, float(Descriptors.MolWt(mol))


def standardize_input_table(frame: pd.DataFrame, canonicalize: bool = True) -> pd.DataFrame:
    aliases = {
        "drug": "name",
        "drug_name": "name",
        "canonical_smiles": "smiles",
        "daily_dose_mg": "dose_mg_day",
        "dose": "dose_mg_day",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns and v not in frame.columns}).copy()
    missing = {"smiles", "dose_mg_day"} - set(frame.columns)
    if missing:
        raise ValueError(f"Input table lacks required columns: {sorted(missing)}")
    if "name" not in frame:
        frame["name"] = [f"compound_{i + 1}" for i in range(len(frame))]
    if "compound_id" not in frame:
        frame["compound_id"] = [f"CMPD{i + 1:04d}" for i in range(len(frame))]
    if "route" not in frame:
        frame["route"] = "oral"
    frame["dose_mg_day"] = pd.to_numeric(frame["dose_mg_day"], errors="raise")
    if frame["dose_mg_day"].isna().any() or (frame["dose_mg_day"] <= 0).any():
        raise ValueError("dose_mg_day must contain positive numbers")
    if canonicalize:
        values = frame["smiles"].astype(str).map(_canonicalize)
        frame["input_smiles"] = frame["smiles"].astype(str)
        frame["smiles"] = [item[0] for item in values]
        frame["molecular_weight"] = [item[1] for item in values]
    elif "molecular_weight" not in frame:
        frame["molecular_weight"] = np.nan
    if frame["compound_id"].duplicated().any():
        duplicates = frame.loc[frame["compound_id"].duplicated(), "compound_id"].tolist()
        raise ValueError(f"compound_id must identify one regimen row; duplicates: {duplicates[:5]}")
    return frame[[*PROFILE_COLUMNS, *[c for c in frame.columns if c not in PROFILE_COLUMNS]]]


def read_ot_profile(path: Path, target_ids: tuple[str, ...]) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[str, dict[str, float]] = {}
    for smiles, target_map in payload.items():
        missing = [target for target in target_ids if target not in target_map]
        if missing:
            raise ValueError(f"OT-ProfileNet output for {smiles} lacks {len(missing)} targets")
        parsed[smiles] = {target: float(target_map[target][1]) for target in target_ids}
    return parsed


def assemble_dotsafenet_features(
    standardized: pd.DataFrame,
    ot_profile: dict[str, dict[str, float]],
    ppb: pd.DataFrame,
    cmax: pd.DataFrame,
    contract: Contract,
    exposure_floor: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ppb_cols = ["smiles", "pred_ppb_mean", "pred_fu_mean", "pred_ppb_std", "pred_fu_std"]
    merged = standardized.merge(ppb[ppb_cols].drop_duplicates("smiles"), on="smiles", how="left")
    cmax_cols = [
        "compound_id", "predicted_log10_cmax_ug_ml_mean", "predicted_log10_cmax_ug_ml_sd",
        "predicted_cmax_ug_ml_mean", "predicted_cmax_ug_ml_sd",
    ]
    merged = merged.merge(cmax[cmax_cols], on="compound_id", how="left")
    required = ["pred_fu_mean", "predicted_cmax_ug_ml_mean", "molecular_weight"]
    if merged[required].isna().any().any():
        raise ValueError("PPB, Cmax, or molecular-weight prediction is missing for at least one regimen")
    merged["free_cmax_ug_ml"] = merged["predicted_cmax_ug_ml_mean"] * merged["pred_fu_mean"]
    merged["free_cmax_uM"] = merged["free_cmax_ug_ml"] / merged["molecular_weight"] * 1000.0
    merged["log10_free_cmax_uM"] = np.log10(np.clip(merged["free_cmax_uM"], exposure_floor, None))

    rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        smiles = str(record["smiles"])
        if smiles not in ot_profile:
            raise KeyError(f"OT-ProfileNet output lacks canonical SMILES: {smiles}")
        exposure = float(record["log10_free_cmax_uM"])
        feature: dict[str, Any] = {
            "Drug": record["name"],
            "smiles": smiles,
            "compound_id": record["compound_id"],
            "cmax_free(uM)": exposure,
        }
        for target in contract.target_ids:
            pac50 = float(ot_profile[smiles][target])
            log10_ac50_uM = 6.0 - pac50
            feature[target] = log10_ac50_uM
            feature[f"{target}*Cmax"] = log10_ac50_uM * exposure
            margin_rows.append({
                "compound_id": record["compound_id"],
                "name": record["name"],
                "dose_mg_day": record["dose_mg_day"],
                "target_uniprot": target,
                "predicted_pAC50": pac50,
                "predicted_AC50_uM": 10.0 ** log10_ac50_uM,
                "free_cmax_uM": record["free_cmax_uM"],
                "margin_log10_AC50_over_free_Cmax": log10_ac50_uM - exposure,
            })
        rows.append(feature)
    features = pd.DataFrame(rows)
    expected = ["Drug", "smiles", "compound_id", *contract.feature_columns]
    features = features[expected]
    return features, merged, pd.DataFrame(margin_rows)


def empirical_percentile(score: float, grid: pd.DataFrame, soc_code: str) -> float:
    subset = grid[(grid["adr_abbr"] == soc_code) & (grid["label_subset"] == "all")]
    if "score_source" in subset:
        subset = subset[subset["score_source"] == "ensemble_mean_probability"]
    subset = subset.sort_values("score").drop_duplicates("score")
    if subset.empty:
        return float("nan")
    return float(np.interp(score, subset["score"].to_numpy(float), subset["percentile"].to_numpy(float)))


def validate_feature_table(frame: pd.DataFrame, contract: Contract) -> None:
    missing = set(contract.feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"DOT-SafeNet feature table lacks {len(missing)} columns: {sorted(missing)[:5]}")
    numeric = frame[list(contract.feature_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("DOT-SafeNet feature table contains non-finite values")
    if numeric.shape[1] != 389:
        raise ValueError(f"Expected 389 numerical features, found {numeric.shape[1]}")


def read_target_reference(path: Path, contract: Contract, column: str = "training_median") -> np.ndarray:
    table = pd.read_csv(path).set_index("target_uniprot")
    if column not in table:
        raise ValueError(f"Target reference file lacks {column}")
    missing = set(contract.target_ids) - set(table.index.astype(str))
    if missing:
        raise ValueError(f"Target reference file lacks {len(missing)} targets")
    return table.loc[list(contract.target_ids), column].to_numpy(np.float32)


def attach_target_annotations(frame: pd.DataFrame, release_root: Path) -> pd.DataFrame:
    gene_path = release_root / "data/reference/target_gene_map.csv"
    evidence_path = release_root / "data/reference/soc_target_evidence_matrix.csv"
    gene = pd.read_csv(gene_path).rename(columns={"human_gene": "target_gene"})
    evidence = pd.read_csv(evidence_path, index_col=0)
    long = evidence.stack().rename("evidence_code").reset_index()
    long.columns = ["soc_code", "target_uniprot", "evidence_code"]
    long["evidence_level"] = long["evidence_code"].map({2: "direct", 1: "secondary", 0: "none"}).fillna("none")
    out = frame.merge(gene[["target_uniprot", "target_gene"]], on="target_uniprot", how="left")
    return out.merge(long[["soc_code", "target_uniprot", "evidence_level"]], on=["soc_code", "target_uniprot"], how="left")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value
