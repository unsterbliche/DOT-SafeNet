from __future__ import annotations

from collections import defaultdict
from typing import Any

from .domain import ADR_AXES


SOC_BY_CODE = {axis["abbr"].upper(): axis for axis in ADR_AXES}
EVIDENCE_RANK = {"direct": 2, "secondary": 1, "none": 0, None: 0}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_rows(margins: list[dict], attribution: list[dict]) -> list[dict]:
    attribution_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in attribution:
        attribution_by_target[str(row.get("target_uniprot"))].append(row)

    targets = []
    for margin in margins:
        uniprot = str(margin.get("target_uniprot"))
        target_attribution = attribution_by_target.get(uniprot, [])
        positive = [row for row in target_attribution if (_as_float(row.get("delta_probability")) or 0.0) > 0]
        strongest = max(
            positive or target_attribution or [{}],
            key=lambda row: (
                _as_float(row.get("delta_probability")) or -1e9,
                EVIDENCE_RANK.get(row.get("evidence_level"), 0),
            ),
        )
        targets.append({
            "target": uniprot,
            "target_name": margin.get("target_gene") or uniprot,
            "gene_name": margin.get("target_gene"),
            "uniprot_id": uniprot,
            "value": _as_float(margin.get("predicted_pAC50")),
            "predicted_ac50_uM": _as_float(margin.get("predicted_AC50_uM")),
            "free_cmax_uM": _as_float(margin.get("free_cmax_uM")),
            "margin_log10": _as_float(margin.get("margin_log10_AC50_over_free_Cmax")),
            "max_delta_probability": _as_float(strongest.get("delta_probability")),
            "max_delta_probability_sd": _as_float(strongest.get("delta_probability_sd")),
            "max_delta_soc_code": strongest.get("soc_code"),
            "evidence_level": strongest.get("evidence_level") or "none",
        })
    targets.sort(
        key=lambda row: (
            row["max_delta_probability"] if row["max_delta_probability"] is not None else -1e9,
            row["value"] if row["value"] is not None else -1e9,
        ),
        reverse=True,
    )
    return targets


def _network(targets: list[dict], attribution: list[dict], adr: list[dict]) -> dict:
    target_by_id = {row["target"]: row for row in targets}
    adr_by_code = {row["abbr"].upper(): row for row in adr}
    ranked_edges = sorted(
        (
            row for row in attribution
            if (_as_float(row.get("delta_probability")) or 0.0) > 0
            and str(row.get("target_uniprot")) in target_by_id
            and str(row.get("soc_code")).upper() in adr_by_code
        ),
        key=lambda row: (
            _as_float(row.get("delta_probability")) or 0.0,
            EVIDENCE_RANK.get(row.get("evidence_level"), 0),
        ),
        reverse=True,
    )[:18]

    target_ids = []
    soc_codes = []
    edges = []
    for row in ranked_edges:
        target = str(row["target_uniprot"])
        soc = str(row["soc_code"]).upper()
        if target not in target_ids:
            target_ids.append(target)
        if soc not in soc_codes:
            soc_codes.append(soc)
        edges.append({
            "source": f"target:{target}",
            "target": f"adr:{soc}",
            "score": _as_float(row.get("delta_probability")) or 0.0,
            "evidence_level": row.get("evidence_level") or "none",
        })

    nodes = []
    for target in target_ids[:10]:
        row = target_by_id[target]
        nodes.append({
            "id": f"target:{target}",
            "type": "target",
            "label": row["target_name"],
            "target_name": row["target_name"],
            "accession": target,
            "score": row.get("max_delta_probability") or 0.0,
        })
    retained_targets = {node["id"] for node in nodes}
    edges = [edge for edge in edges if edge["source"] in retained_targets]
    retained_soc = {edge["target"].split(":", 1)[1] for edge in edges}
    for soc in soc_codes:
        if soc not in retained_soc:
            continue
        row = adr_by_code[soc]
        nodes.append({
            "id": f"adr:{soc}",
            "type": "adr",
            "label": row["abbr"],
            "full_label": row["task_name"],
            "score": row.get("mean_probability") or 0.0,
        })
    return {"nodes": nodes, "edges": edges}


def build_profile_result(report: dict) -> dict:
    exposure_rows = report.get("exposure") or []
    adr_rows = report.get("adr_predictions") or []
    margin_rows = report.get("target_margins") or []
    attribution_rows = report.get("target_attribution") or []

    adr_by_compound: dict[str, list[dict]] = defaultdict(list)
    margin_by_compound: dict[str, list[dict]] = defaultdict(list)
    attribution_by_compound: dict[str, list[dict]] = defaultdict(list)
    for row in adr_rows:
        adr_by_compound[str(row.get("compound_id"))].append(row)
    for row in margin_rows:
        margin_by_compound[str(row.get("compound_id"))].append(row)
    for row in attribution_rows:
        attribution_by_compound[str(row.get("compound_id"))].append(row)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for exposure in exposure_rows:
        grouped[(str(exposure.get("name")), str(exposure.get("smiles")))].append(exposure)

    results = []
    for (name, smiles), exposures in grouped.items():
        dose_results = []
        for exposure in sorted(exposures, key=lambda row: float(row.get("dose_mg_day") or 0)):
            compound_id = str(exposure.get("compound_id"))
            raw_adr = {str(row.get("soc_code")).upper(): row for row in adr_by_compound[compound_id]}
            adr = []
            for axis in ADR_AXES:
                row = raw_adr.get(axis["abbr"].upper(), {})
                probability = _as_float(row.get("probability_mean"))
                adr.append({
                    "abbr": axis["abbr"],
                    "task_name": axis["task_name"],
                    "mean_probability": probability,
                    "std_probability": _as_float(row.get("probability_sd")),
                    "background_percentile": _as_float(row.get("background_percentile")),
                    "positive_label": bool(probability is not None and probability >= 0.5),
                })

            targets = _target_rows(margin_by_compound[compound_id], attribution_by_compound[compound_id])
            ranked_attribution = sorted(
                attribution_by_compound[compound_id],
                key=lambda row: _as_float(row.get("delta_probability")) or -1e9,
                reverse=True,
            )
            dose_results.append({
                "compound_id": compound_id,
                "dose_mg": float(exposure.get("dose_mg_day")),
                "route": exposure.get("route") or "oral",
                "pk": {
                    "ppb": 100.0 * float(exposure.get("pred_ppb_mean")),
                    "ppb_fraction": _as_float(exposure.get("pred_ppb_mean")),
                    "ppb_std": 100.0 * float(exposure.get("pred_ppb_std")),
                    "fu": _as_float(exposure.get("pred_fu_mean")),
                    "cmax_ug_ml": _as_float(exposure.get("predicted_cmax_ug_ml_mean")),
                    "cmax_ug_ml_sd": _as_float(exposure.get("predicted_cmax_ug_ml_sd")),
                    "cmax_free_ug_ml": _as_float(exposure.get("free_cmax_ug_ml")),
                    "cmax_free_uM": _as_float(exposure.get("free_cmax_uM")),
                    "molecular_weight": _as_float(exposure.get("molecular_weight")),
                },
                "adr": adr,
                "top_risks": sorted(
                    (row for row in adr if row["mean_probability"] is not None),
                    key=lambda row: row["mean_probability"],
                    reverse=True,
                )[:8],
                "top_offtargets": targets[:20],
                "offtarget_affinity": targets,
                "offtarget_target_count": len(targets),
                "target_attribution": ranked_attribution[:60],
                "target_adr_network": _network(targets, attribution_by_compound[compound_id], adr),
            })

        latest = dose_results[-1]
        result = {
            "compound": {
                "name": name,
                "smiles": exposures[0].get("input_smiles") or smiles,
                "canonical_smiles": smiles,
            },
            "paper_case_group": "dotsafenet_v1_full_inference",
            "dose_panel_mg": [row["dose_mg"] for row in dose_results],
            "adr_axes": ADR_AXES,
            "dose_results": dose_results,
            "target_adr_network": latest["target_adr_network"],
            "model_metadata": report.get("metadata") or {},
            "inference_engine": "DOT-SafeNet v1.0.0 clinical five-fold ensemble",
        }
        results.append(result)

    if not results:
        raise ValueError("DOT-SafeNet report contained no regimen rows")
    if len(results) == 1:
        return results[0]
    batch = dict(results[0])
    batch["results"] = results
    batch["batch_size"] = len(results)
    batch["paper_case_group"] = "dotsafenet_v1_full_inference_batch"
    return batch
