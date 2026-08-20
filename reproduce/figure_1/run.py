"""Render the Figure 1 ADR-output insert and placement preview."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "scripts" / "render_adr_output_schematic.py"), "--config", str(ROOT / "params.yaml")], check=True)