"""Render the A4-ready four-case manuscript Figure 6 source panels."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "scripts" / "render_cases_a4.py")], check=True)
