"""Render the revised Supplementary Figure 2a and 2b panels."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
for name in ["prepare_panel_b_data.py", "plot_panel_a.py", "plot_panel_b.py"]:
    subprocess.run([sys.executable, str(ROOT / name)], check=True)