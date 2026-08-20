from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from .profile_adapter import build_profile_result


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


def _map_remote(path: Path) -> Path:
    local_root = os.environ.get("DOTSAFENET_SHARED_LOCAL_ROOT")
    remote_root = os.environ.get("DOTSAFENET_SHARED_REMOTE_ROOT")
    if not local_root or not remote_root:
        return path
    try:
        relative = path.resolve().relative_to(Path(local_root).resolve())
    except ValueError:
        return path
    return Path(remote_root) / relative


def _settings() -> dict:
    release_root = _path_env("DOTSAFENET_RELEASE_ROOT", PROJECT_ROOT / "DOT-SafeNet-v1.0.0")
    checkpoint_root = _path_env("DOTSAFENET_CHECKPOINT_ROOT", PROJECT_ROOT)
    asset_root = _path_env("DOTSAFENET_ASSET_ROOT", PROJECT_ROOT)
    return {
        "release_root": release_root,
        "checkpoint_root": checkpoint_root,
        "asset_root": asset_root,
        "ot_python": os.environ.get("DOTSAFENET_OT_PYTHON", str(PROJECT_ROOT / "mse_unified_env/bin/python")),
        "exposure_python": os.environ.get(
            "DOTSAFENET_EXPOSURE_PYTHON", str(PROJECT_ROOT / "mse_unified_env/bin/python")
        ),
        "adr_python": os.environ.get("DOTSAFENET_ADR_PYTHON", str(PROJECT_ROOT / "inference_env/bin/python")),
        "device": os.environ.get("DOTSAFENET_DEVICE", "cuda:0"),
        "ot_device": os.environ.get("DOTSAFENET_OT_DEVICE", ""),
        "exposure_device": os.environ.get("DOTSAFENET_EXPOSURE_DEVICE", ""),
        "host": os.environ.get("DOTSAFENET_INFERENCE_HOST", "").strip(),
    }


def _regimen_rows(items: list[dict], default_doses: list[float]) -> list[dict]:
    rows = []
    for item_index, item in enumerate(items, start=1):
        doses = item.get("dose_panel_mg") or default_doses
        for dose_index, dose in enumerate(doses, start=1):
            dose_value = float(dose)
            if dose_value <= 0:
                raise ValueError("All daily doses must be positive")
            rows.append({
                "compound_id": f"WEB{item_index:03d}_D{dose_index:03d}",
                "name": item.get("name") or f"compound_{item_index}",
                "dose_mg_day": dose_value,
                "route": item.get("route") or "oral",
                "smiles": item["smiles"],
            })
    return rows


def _write_input(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["compound_id", "name", "dose_mg_day", "route", "smiles"])
        writer.writeheader()
        writer.writerows(rows)


def _command(input_path: Path, output_dir: Path, settings: dict) -> list[str]:
    release_root = _map_remote(settings["release_root"])
    checkpoint_root = _map_remote(settings["checkpoint_root"])
    asset_root = _map_remote(settings["asset_root"])
    command = [
        str(_map_remote(Path(settings["adr_python"]))),
        str(release_root / "scripts/profile_molecule.py"),
        "analyze",
        "--input",
        str(_map_remote(input_path)),
        "--output",
        str(_map_remote(output_dir)),
        "--checkpoint-root",
        str(checkpoint_root),
        "--asset-root",
        str(asset_root),
        "--ot-python",
        str(_map_remote(Path(settings["ot_python"]))),
        "--exposure-python",
        str(_map_remote(Path(settings["exposure_python"]))),
        "--adr-python",
        str(_map_remote(Path(settings["adr_python"]))),
        "--device",
        settings["device"],
    ]
    if settings["ot_device"]:
        command.extend(["--ot-device", settings["ot_device"]])
    if settings["exposure_device"]:
        command.extend(["--exposure-device", settings["exposure_device"]])
    return command


def run_live_inference(items: list[dict], dose_panel_mg: list[float] | None = None, device: str | None = None) -> dict:
    if not items:
        raise ValueError("At least one molecule is required")
    settings = _settings()
    if device:
        settings["device"] = device
    default_doses = [float(value) for value in (dose_panel_mg or [10, 20, 40])]
    rows = _regimen_rows(items, default_doses)

    data_root = APP_DIR / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dotsafenet_web_", dir=data_root) as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.csv"
        output_dir = tmp_path / "output"
        _write_input(input_path, rows)
        command = _command(input_path, output_dir, settings)
        if settings["host"]:
            command = ["ssh", settings["host"], shlex.join(command)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(settings["release_root"] if not settings["host"] else APP_DIR),
                text=True,
                capture_output=True,
                timeout=int(os.environ.get("DOTSAFENET_INFERENCE_TIMEOUT", "7200")),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"DOT-SafeNet inference command is unavailable: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("DOT-SafeNet inference exceeded the configured time limit") from exc

        if completed.returncode != 0:
            stderr = completed.stderr[-6000:] if completed.stderr else ""
            stdout = completed.stdout[-2000:] if completed.stdout else ""
            raise RuntimeError(f"DOT-SafeNet inference failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        report_path = output_dir / "report.json"
        if not report_path.exists():
            raise RuntimeError("DOT-SafeNet completed without report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    result = build_profile_result(report)
    result["inference_engine"] = "DOT-SafeNet v1.0.0 clinical five-fold ensemble"
    return result


def model_health() -> dict:
    settings = _settings()
    required = {
        "release_root": settings["release_root"] / "scripts/profile_molecule.py",
        "checkpoint_manifest": settings["release_root"] / "weights/weights_manifest.tsv",
        "target_contract": settings["release_root"] / "data/model/target_order.csv",
        "soc_contract": settings["release_root"] / "data/model/soc_order.csv",
    }
    return {
        "backend": "DOT-SafeNet v1.0.0",
        "models": ["OT-ProfileNet", "PlasmaBindNet-Fu", "DoseExpoNet", "DOT-SafeNet clinical five-fold ensemble"],
        "release_root": str(settings["release_root"]),
        "inference_host": settings["host"] or "local",
        "device": settings["device"],
        "ot_device": settings["ot_device"] or settings["device"],
        "exposure_device": settings["exposure_device"] or settings["device"],
        "required_paths": {name: path.exists() for name, path in required.items()},
        "checkpoint_root": str(settings["checkpoint_root"]),
        "asset_root": str(settings["asset_root"]),
    }
