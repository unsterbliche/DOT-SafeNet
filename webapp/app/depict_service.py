from __future__ import annotations

import os
import json
import subprocess
import tempfile
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
DEFAULT_INFERENCE_PYTHON = PROJECT_ROOT / "inference_env" / "bin" / "python"
DEPICT_SCRIPT = APP_DIR / "inference" / "depict_smiles.py"
PARSE_SCRIPT = APP_DIR / "inference" / "parse_structures.py"


def _inference_python() -> str:
    return os.environ.get("DOTSAFENET_INFERENCE_PYTHON", str(DEFAULT_INFERENCE_PYTHON))


def depict_smiles(smiles: str) -> str:
    if not smiles.strip():
        raise ValueError("SMILES is required.")
    with tempfile.TemporaryDirectory(prefix="dotsafenet_depict_", dir=APP_DIR / "data") as tmp_dir:
        output_path = Path(tmp_dir) / "molecule.svg"
        completed = subprocess.run(
            [_inference_python(), str(DEPICT_SCRIPT), "--smiles", smiles, "--output", str(output_path)],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "Unable to render molecule.").strip()
            raise ValueError(message[-500:])
        return output_path.read_text(encoding="utf-8")


def parse_structure_file(filename: str, content: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="dotsafenet_parse_", dir=APP_DIR / "data") as tmp_dir:
        input_path = Path(tmp_dir) / filename
        input_path.write_text(content, encoding="utf-8", errors="ignore")
        completed = subprocess.run(
            [_inference_python(), str(PARSE_SCRIPT), "--input", str(input_path), "--filename", filename],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "Unable to parse structure file.").strip()
            raise ValueError(message[-500:])
        payload = json.loads(completed.stdout)
        return payload.get("items") or []
