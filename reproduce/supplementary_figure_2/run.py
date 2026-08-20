"""Generate Supplementary Figure 2; panel b source data are required."""
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parent
for name in ["plot_panel_a.py", "plot_panel_b.py", "plot_panel_c.py", "plot_panel_d.py",
             "compose_supplementary_figure_2.py"]:
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)

