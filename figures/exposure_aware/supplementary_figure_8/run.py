from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "scripts" / "render.py")], cwd=ROOT))
