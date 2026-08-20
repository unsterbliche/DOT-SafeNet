from __future__ import annotations

import os
import sys
from pathlib import Path


WEBAPP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEBAPP_ROOT.parent
RELEASE_ROOT = PROJECT_ROOT / "09_results/dotsafenet_github_release_20260819"
sys.path.insert(0, str(WEBAPP_ROOT))

from app.data_service import build_csv
from app.domain import ADR_AXES
from app.live_service import _regimen_rows, model_health
from app.profile_adapter import build_profile_result
from app.samples import SAMPLES


EXPECTED_DOSES = {
    "meclofenamic_acid": [75, 150, 300],
    "citalopram": [10, 20, 40],
    "spironolactone": [12.5, 25, 50],
    "candesartan_cilexetil": [8, 16, 32],
}


def test_sample_contract() -> None:
    assert set(SAMPLES) == set(EXPECTED_DOSES)
    for key, doses in EXPECTED_DOSES.items():
        sample = SAMPLES[key]
        assert list(sample.dose_panel_mg) == doses
        assert sample.highlighted_soc
        assert sample.highlighted_target


def test_regimen_rows_support_decimal_doses() -> None:
    rows = _regimen_rows(
        [{"name": "Spironolactone", "smiles": SAMPLES["spironolactone"].smiles, "dose_panel_mg": [12.5, 25]}],
        [10],
    )
    assert [row["dose_mg_day"] for row in rows] == [12.5, 25.0]
    assert len({row["compound_id"] for row in rows}) == 2


def test_full_smoke_report_adapter() -> None:
    compound_id = "TEST_CITALOPRAM_20"
    targets = [("P35414", "APLNR", 6.86), ("Q12809", "KCNH2", 6.22)]
    report = {
        "metadata": {"mode": "test", "target_count": 2, "soc_tasks": 18},
        "exposure": [{
            "compound_id": compound_id,
            "name": "Citalopram",
            "smiles": SAMPLES["citalopram"].smiles,
            "input_smiles": SAMPLES["citalopram"].smiles,
            "dose_mg_day": 20,
            "route": "oral",
            "molecular_weight": 324.399,
            "pred_ppb_mean": 0.64,
            "pred_ppb_std": 0.11,
            "pred_fu_mean": 0.36,
            "predicted_cmax_ug_ml_mean": 0.021,
            "predicted_cmax_ug_ml_sd": 0.003,
            "free_cmax_ug_ml": 0.0076,
            "free_cmax_uM": 0.0234,
        }],
        "adr_predictions": [
            {
                "compound_id": compound_id,
                "soc_code": axis["abbr"].upper(),
                "soc_name": axis["task_name"],
                "probability_mean": 0.60 if axis["abbr"] == "Car" else 0.10,
                "probability_sd": 0.02,
                "background_percentile": 85.0 if axis["abbr"] == "Car" else 20.0,
            }
            for axis in ADR_AXES
        ],
        "target_margins": [
            {
                "compound_id": compound_id,
                "target_uniprot": uniprot,
                "target_gene": gene,
                "predicted_pAC50": pac50,
                "predicted_AC50_uM": 10 ** (6 - pac50),
                "free_cmax_uM": 0.0234,
                "margin_log10_AC50_over_free_Cmax": 0.8,
            }
            for uniprot, gene, pac50 in targets
        ],
        "target_attribution": [
            {
                "compound_id": compound_id,
                "soc_code": axis["abbr"].upper(),
                "target_uniprot": uniprot,
                "delta_probability": 0.22 if axis["abbr"] == "Car" and gene == "KCNH2" else 0.001,
                "delta_probability_sd": 0.01,
                "evidence_level": "direct" if axis["abbr"] == "Car" and gene == "KCNH2" else "none",
            }
            for uniprot, gene, _ in targets
            for axis in ADR_AXES
        ],
    }
    result = build_profile_result(report)
    dose = result["dose_results"][0]
    assert result["compound"]["name"] == "Citalopram"
    assert len(dose["adr"]) == 18
    assert dose["offtarget_target_count"] == 2
    assert 0 <= dose["pk"]["ppb"] <= 100
    assert dose["top_offtargets"]
    assert dose["top_offtargets"][0]["max_delta_probability"] is not None
    assert dose["target_adr_network"]["nodes"]
    csv_text = build_csv(result)
    assert "ppb_percent" in csv_text
    assert "Car_background_percentile" in csv_text
    assert "pAC50_" in csv_text
    assert "max_delta_ADR_" in csv_text


def test_health_reports_release_contract() -> None:
    os.environ["DOTSAFENET_RELEASE_ROOT"] = str(RELEASE_ROOT)
    health = model_health()
    assert health["backend"] == "DOT-SafeNet v1.0.0"
    if (RELEASE_ROOT / "data/model/target_order.csv").exists():
        assert all(health["required_paths"].values())


if __name__ == "__main__":
    test_sample_contract()
    test_regimen_rows_support_decimal_doses()
    test_full_smoke_report_adapter()
    test_health_reports_release_contract()
    print("DOT-SafeNet web contract ok")
