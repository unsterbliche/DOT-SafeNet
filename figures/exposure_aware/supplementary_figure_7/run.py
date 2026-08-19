"""Render the two panel groups assigned to Supplementary Figure 7."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
JOBS = [
    ("render_ppb_scaffold_metrics.py", "params_ppb_scaffold_metrics.yaml"),
    ("render_cmax_case_curves.py", "params_cmax_case_curves.yaml"),
]
for script, config in JOBS:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--config", str(ROOT / config)],
        check=True,
    )
