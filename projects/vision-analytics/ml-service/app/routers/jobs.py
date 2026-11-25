"""Video jobs: the Express server starts one, then polls it until it is done.

    POST   /jobs/video      {run_id, video_path, model, conf, line}  -> 202, job queued
    GET    /jobs/{run_id}   status + progress; when done also the full result
    DELETE /jobs/{run_id}   forget a finished job (the server calls this after saving the result)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import video_jobs
from ..config import get_settings
from ..schemas import JobOut, VideoJobIn
from .detect import get_detector

router = APIRouter(prefix="/jobs", tags=["video jobs"])


@router.post("/video", response_model=JobOut, status_code=202)
def start_video_job(body: VideoJobIn):
    if body.line is not None and not (len(body.line) == 4 and all(0 <= v <= 1 for v in body.line)):
        raise HTTPException(422, "line must be a list of 4 numbers in [0, 1]")
    name, _ = get_detector(body.model)                   # 404 now rather than a failed job later
    uploads = (get_settings().data_dir / "uploads").resolve()
    path = Path(body.video_path).resolve()
    if not path.is_relative_to(uploads):                 # only read files the server saved for us
        raise HTTPException(400, "video_path must be inside the uploads folder")
    if not path.is_file():
        raise HTTPException(400, "video file not found")
    if video_jobs.get(body.run_id) is not None:
        raise HTTPException(409, "a job for this run already exists")
    return video_jobs.submit(body.run_id, path, name, body.conf, body.line)


@router.get("/{run_id}", response_model=JobOut)
def job_status(run_id: str):
    job = video_jobs.get(run_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.delete("/{run_id}", status_code=204)
def forget_job(run_id: str):
    video_jobs.forget(run_id)
