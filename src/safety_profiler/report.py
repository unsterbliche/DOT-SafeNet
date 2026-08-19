from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from .core import json_ready


CSS = """
body{font-family:Arial,Helvetica,sans-serif;color:#26343d;max-width:1180px;margin:32px auto;padding:0 24px;line-height:1.45}
h1,h2{color:#143f52} h1{font-size:27px} h2{font-size:19px;margin-top:30px;border-bottom:1px solid #d8e1e5;padding-bottom:6px}
.meta{color:#5d6d74}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.card{border:1px solid #d9e3e6;border-radius:8px;padding:13px;background:#f8fbfc}.big{font-size:23px;font-weight:700;color:#1f6a88}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e2e8eb;padding:6px 7px;text-align:left}th{background:#eef5f7}
.bar{height:8px;background:#e7eef1;border-radius:4px;overflow:hidden}.fill{height:100%;background:#2d75b8}.direct{color:#155f9e}.secondary{color:#c67928}.none{color:#7d888d}
.foot{margin-top:34px;color:#66757c;font-size:12px}.warn{background:#fff6df;border-left:4px solid #dfa842;padding:10px 12px}
"""


def _table(frame: pd.DataFrame, columns: list[str], n: int = 20) -> str:
    view = frame[columns].head(n).copy()
    for col in view.select_dtypes(include="number"):
        view[col] = view[col].map(lambda x: f"{x:.3f}")
    return view.to_html(index=False, escape=True, border=0)


def _number(value, suffix: str = "", digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}g}{suffix}"


def render_report(
    output: Path,
    metadata: dict,
    exposure: pd.DataFrame,
    adr: pd.DataFrame,
    margins: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "exposure": exposure.to_dict(orient="records"),
        "adr_predictions": adr.to_dict(orient="records"),
        "target_margins": margins.to_dict(orient="records"),
        "target_attribution": attribution.to_dict(orient="records"),
    }
    (output / "report.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")

    sections = []
    for row in exposure.itertuples(index=False):
        cid = str(row.compound_id)
        adr_one = adr[adr["compound_id"].astype(str) == cid].sort_values("background_percentile", ascending=False)
        margin_one = margins[margins["compound_id"].astype(str) == cid].sort_values("margin_log10_AC50_over_free_Cmax")
        attr_one = attribution[attribution["compound_id"].astype(str) == cid].sort_values("delta_probability", ascending=False)
        cards = f"""
        <div class='cards'>
          <div class='card'><div class='meta'>Daily dose</div><div class='big'>{_number(row.dose_mg_day, ' mg')}</div></div>
          <div class='card'><div class='meta'>Predicted PPB</div><div class='big'>{_number(100*float(row.pred_ppb_mean), '%') if pd.notna(row.pred_ppb_mean) else 'N/A'}</div></div>
          <div class='card'><div class='meta'>Predicted total Cmax</div><div class='big'>{_number(row.predicted_cmax_ug_ml_mean, ' µg mL⁻¹')}</div></div>
          <div class='card'><div class='meta'>Predicted free Cmax</div><div class='big'>{_number(row.free_cmax_uM, ' µM')}</div></div>
        </div>"""
        sections.append(
            f"<h2>{html.escape(str(row.name))}</h2>{cards}"
            + "<h3>18-SOC predictions</h3>"
            + _table(adr_one, ["soc_code", "soc_name", "probability_mean", "probability_sd", "background_percentile"], 18)
            + "<h3>Lowest predicted target margins</h3>"
            + _table(margin_one, ["target_gene", "target_uniprot", "predicted_pAC50", "margin_log10_AC50_over_free_Cmax"], 12)
            + "<h3>Largest positive target-replacement effects</h3>"
            + _table(attr_one, ["soc_code", "target_gene", "target_uniprot", "delta_probability", "evidence_level"], 20)
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>DOT-SafeNet safety profile</title><style>{CSS}</style></head>
    <body><h1>Exposure-aware computational safety profile</h1>
    <p class='meta'>OT-ProfileNet · PlasmaBindNet-Fu · DoseExpoNet · DOT-SafeNet</p>
    <p class='warn'>Results are model predictions for comparative research use. Dose, formulation, metabolites, drug–drug interactions and population pharmacokinetics require separate assessment.</p>
    {''.join(sections)}
    <p class='foot'>The SOC probability is accompanied by its empirical percentile in the DOT-SafeNet development background. The percentile is not a calibrated incidence estimate. Target attribution is the decrease in SOC score after replacing one target feature with its training reference value.</p>
    </body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")
