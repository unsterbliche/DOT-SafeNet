"""Generate all Supplementary Figure 4 panels and the complete composition."""
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parent
for name in ["plot_panel_a.py", "plot_panel_b.py", "plot_panel_c.py", "plot_panel_d.py",
             "plot_panel_e.py", "plot_panel_f.py", "compose_supplementary_figure_4.py"]:
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)

