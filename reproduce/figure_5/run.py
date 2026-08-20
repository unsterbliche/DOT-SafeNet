"""Render manuscript Figure 5 from its fixed source tables."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "render.py"), "--config", str(ROOT / "params.yaml")],
    check=True,
)
