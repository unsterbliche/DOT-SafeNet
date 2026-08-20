from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data_service import build_csv
from .depict_service import depict_smiles, parse_structure_file
from .job_store import cleanup_jobs, list_jobs, load_job, save_job
from .live_service import model_health, run_live_inference
from .samples import get_sample, list_samples


APP_DIR = Path(__file__).resolve().parents[1]


class JobItem(BaseModel):
    name: str | None = None
    smiles: str
    dose_panel_mg: list[float] | None = None


class JobRequest(BaseModel):
    sample_key: str | None = None
    items: list[JobItem] | None = None
    dose_panel_mg: list[float] | None = None


class DepictRequest(BaseModel):
    smiles: str


class StructureParseRequest(BaseModel):
    filename: str
    content: str


app = FastAPI(title="DOT-SafeNet Web App", version="1.0.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

JOBS: dict[str, dict[str, Any]] = {}


@app.on_event("startup")
def _startup() -> None:
    cleanup_jobs()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/samples")
def samples() -> list[dict]:
    return list_samples()


@app.post("/api/depict")
def depict(request: DepictRequest) -> Response:
    try:
        svg = depict_smiles(request.smiles)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/api/structures/parse")
def parse_structures(request: StructureParseRequest) -> dict:
    try:
        items = parse_structure_file(request.filename, request.content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": items}


@app.get("/api/health")
def app_health() -> dict:
    payload = model_health()
    payload["jobs_in_memory"] = len(JOBS)
    payload["job_retention_days"] = 5
    return payload


@app.post("/api/jobs")
def create_job(request: JobRequest, background_tasks: BackgroundTasks) -> dict:
    if request.sample_key:
        sample = get_sample(request.sample_key)
        if not sample:
            raise HTTPException(status_code=404, detail=f"Unknown sample: {request.sample_key}")
        job_id = _new_job()
        JOBS[job_id].update({
            "stage": "queued sample prediction",
            "compound_name": sample.name,
            "dose_panel_mg": list(sample.dose_panel_mg),
        })
        _save_current(job_id)
        background_tasks.add_task(_run_sample_job, job_id, sample)
        return {"job_id": job_id}

    items = request.items or []
    if not items:
        raise HTTPException(status_code=400, detail="Provide sample_key or items.")
    warnings = []
    if len(items) > 10:
        warnings.append(f"Received {len(items)} molecules; only the first 10 will be predicted.")
        items = items[:10]

    job_id = _new_job()
    payload_items = [item.dict() for item in items]
    JOBS[job_id].update({
        "warnings": warnings,
        "stage": "queued prediction",
        "items": payload_items,
        "compound_name": payload_items[0].get("name") if payload_items else None,
        "dose_panel_mg": request.dose_panel_mg,
    })
    _save_current(job_id)
    background_tasks.add_task(_run_manual_job, job_id, payload_items, request.dose_panel_mg)
    return {"job_id": job_id}


@app.get("/api/jobs")
def get_jobs() -> list[dict]:
    saved = list_jobs()
    saved_ids = {item.get("job_id") for item in saved}
    current = [
        _job_summary(job)
        for job in JOBS.values()
        if job["job_id"] not in saved_ids
    ]
    payload = saved + current
    payload.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return payload


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {key: value for key, value in job.items() if key != "csv"}


@app.get("/api/jobs/{job_id}/results")
def get_results(job_id: str) -> dict:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    return job["result"]


@app.get("/api/jobs/{job_id}/results.csv")
def get_results_csv(job_id: str) -> Response:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    filename = f"{job['result']['compound']['name']}_dotsafenet_results.csv"
    return Response(
        content=job["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "created_at": time.time(),
        "engine": "dotsafenet_v1_clinical_ensemble",
    }
    save_job(JOBS[job_id])
    return job_id


def _get_job(job_id: str) -> dict[str, Any] | None:
    return JOBS.get(job_id) or load_job(job_id)


def _save_current(job_id: str) -> None:
    save_job(JOBS[job_id])


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    compound = result.get("compound") or {}
    stored_items = job.get("items") or []
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "engine": job.get("engine"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "compound_name": compound.get("name") or job.get("compound_name"),
        "batch_size": result.get("batch_size") or len(result.get("results") or []) or len(stored_items) or 1,
        "dose_panel_mg": job.get("dose_panel_mg") or result.get("dose_panel_mg"),
        "warnings": job.get("warnings") or [],
    }


def _run_sample_job(job_id: str, sample) -> None:
    job = JOBS[job_id]
    job.update({"status": "running", "progress": 20, "stage": "running DOT-SafeNet v1.0.0 full-model inference"})
    try:
        result = run_live_inference([
            {"name": sample.name, "smiles": sample.smiles, "dose_panel_mg": list(sample.dose_panel_mg)}
        ])
        job.update({
            "status": "succeeded",
            "progress": 100,
            "stage": "complete",
            "engine": result.get("inference_engine", "DOT-SafeNet v1.0.0 clinical five-fold ensemble"),
            "result": result,
            "csv": build_csv(result),
            "finished_at": time.time(),
        })
        _save_current(job_id)
    except Exception as exc:
        job.update({"status": "failed", "progress": 100, "error": str(exc), "finished_at": time.time()})
        _save_current(job_id)


def _run_manual_job(job_id: str, items: list[dict], dose_panel_mg: list[float] | None) -> None:
    job = JOBS[job_id]
    job.update({
        "status": "running",
        "progress": 20,
        "stage": "running OT-ProfileNet, PlasmaBindNet-Fu, DoseExpoNet and DOT-SafeNet",
        "items": items,
        "dose_panel_mg": dose_panel_mg,
    })
    _save_current(job_id)
    try:
        result = run_live_inference(items, dose_panel_mg)
        engine = result.get("inference_engine", "DOT-SafeNet v1.0.0 clinical five-fold ensemble")
        if job.get("warnings"):
            result["warnings"] = job["warnings"]
        job.update({
            "status": "succeeded",
            "progress": 100,
            "stage": "complete",
            "engine": engine,
            "result": result,
            "csv": build_csv(result),
            "finished_at": time.time(),
        })
        _save_current(job_id)
    except Exception as exc:
        job.update({"status": "failed", "progress": 100, "error": str(exc), "finished_at": time.time()})
        _save_current(job_id)
