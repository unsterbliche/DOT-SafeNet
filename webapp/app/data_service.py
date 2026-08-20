"""Tabular export for DOT-SafeNet web results."""

from __future__ import annotations

import csv
from io import StringIO


def build_csv(result: dict) -> str:
    """Flatten one or more full-model reports into a dose-level CSV."""
    rows: list[dict] = []
    for single_result in result.get("results", [result]):
        compound = single_result["compound"]
        for dose in single_result["dose_results"]:
            row = {
                "compound": compound["name"],
                "smiles": compound["smiles"],
                "canonical_smiles": compound["canonical_smiles"],
                "dose_mg_day": dose["dose_mg"],
                "ppb_percent": dose["pk"]["ppb"],
                "fu": dose["pk"].get("fu"),
                "cmax_total_ug_ml": dose["pk"]["cmax_ug_ml"],
                "cmax_total_ug_ml_sd": dose["pk"].get("cmax_ug_ml_sd"),
                "cmax_free_ug_ml": dose["pk"]["cmax_free_ug_ml"],
                "cmax_free_uM": dose["pk"]["cmax_free_uM"],
            }
            for adr in dose["adr"]:
                abbr = adr["abbr"]
                row[f"{abbr}_mean_probability"] = adr["mean_probability"]
                row[f"{abbr}_std_probability"] = adr["std_probability"]
                row[f"{abbr}_background_percentile"] = adr.get("background_percentile")
                row[f"{abbr}_positive_label"] = adr["positive_label"]
            for target in dose["offtarget_affinity"]:
                name = target["target"]
                row[f"pAC50_{name}"] = target["value"]
                row[f"margin_log10_{name}"] = target.get("margin_log10")
                row[f"max_delta_ADR_{name}"] = target.get("max_delta_probability")
            rows.append(row)

    if not rows:
        return ""

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
