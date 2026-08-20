"""Render the generated source panels used in manuscript Figure 4."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "scripts" / "render_dotsafenet_architecture.py"), "--config", str(ROOT / "params.yaml")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts" / "render_adr_target_matrix.py")], check=True)
