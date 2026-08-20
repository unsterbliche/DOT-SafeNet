from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parents[1]
JOB_DIR = APP_DIR / "data" / "jobs"
RETENTION_SECONDS = 5 * 24 * 60 * 60
CURRENT_ENGINES = {"dotsafenet_v1_clinical_ensemble", "DOT-SafeNet v1.0.0 clinical five-fold ensemble"}


def cleanup_jobs(now: float | None = None) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = (now or time.time()) - RETENTION_SECONDS
    for path in JOB_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
            continue
        finished = payload.get("finished_at") or payload.get("created_at") or path.stat().st_mtime
        if finished < cutoff:
            path.unlink(missing_ok=True)


def save_job(job: dict[str, Any]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = JOB_DIR / f"{job['job_id']}.json"
    path.write_text(json.dumps(job, ensure_ascii=True), encoding="utf-8")


def load_job(job_id: str) -> dict[str, Any] | None:
    path = JOB_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    cleanup_jobs()
    jobs = []
    for path in JOB_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("engine") not in CURRENT_ENGINES:
            continue
        result = payload.get("result") or {}
        compound = result.get("compound") or {}
        jobs.append({
            "job_id": payload.get("job_id"),
            "status": payload.get("status"),
            "stage": payload.get("stage"),
            "engine": payload.get("engine"),
            "created_at": payload.get("created_at"),
            "finished_at": payload.get("finished_at"),
            "compound_name": compound.get("name"),
            "batch_size": result.get("batch_size") or len(result.get("results") or []) or 1,
            "dose_panel_mg": payload.get("dose_panel_mg") or result.get("dose_panel_mg"),
            "warnings": payload.get("warnings") or [],
        })
    jobs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return jobs[:limit]
