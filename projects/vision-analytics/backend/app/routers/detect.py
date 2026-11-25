"""Image detection, model comparison and video upload endpoints."""
from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import video_jobs
from ..config import get_settings
from ..db import Detection, Run, get_db
from ..models_registry import get_registry
from ..schemas import CompareResult, ImageResult, RunOut

router = APIRouter(prefix="/api", tags=["detection"])

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _detector(name: str | None):
    name = name or get_settings().default_model
    try:
        return name, get_registry().get(name)
    except KeyError:
        raise HTTPException(404, f"model '{name}' not found; available: {get_registry().available()}")


async def _read_image(file: UploadFile) -> np.ndarray:
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(415, f"unsupported image type {file.content_type}")
    data = await file.read()
    if len(data) > get_settings().max_upload_mb * 2**20:
        raise HTTPException(413, "file too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return img


def _check_thresholds(conf: float, iou: float) -> None:
    if not (0 < conf < 1 and 0 < iou < 1):
        raise HTTPException(422, "conf and iou must be in (0, 1)")


def _detect(img: np.ndarray, name: str, det, conf: float, iou: float) -> ImageResult:
    dets, timing = det(img, conf=conf, iou=iou)
    return ImageResult(run_id=None, model=name, provider=det.provider, width=img.shape[1], height=img.shape[0],
                       detections=[d.__dict__ for d in dets], counts=dict(Counter(d.label for d in dets)),
                       timing=timing)


@router.post("/detect/image", response_model=ImageResult)
async def detect_image(file: UploadFile = File(...), model: str | None = Form(None),
                       conf: float = Form(0.25), iou: float = Form(0.45), save: bool = Form(True),
                       db: Session = Depends(get_db)):
    _check_thresholds(conf, iou)
    img = await _read_image(file)
    name, det = _detector(model)
    res = _detect(img, name, det, conf, iou)
    if save:
        run = Run(kind="image", model=name, source_name=file.filename or "image", width=res.width,
                  height=res.height, frames=1, avg_latency_ms=res.timing.total_ms, conf=conf,
                  summary={"counts": res.counts, "timing": res.timing.model_dump()})
        run.detections = [Detection(label=d.label, score=d.score, x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3])
                          for d in res.detections]
        db.add(run)
        db.commit()
        res.run_id = run.id
    return res


@router.post("/compare", response_model=CompareResult)
async def compare(file: UploadFile = File(...), models: str = Form(...),
                  conf: float = Form(0.25), iou: float = Form(0.45)):
    """Run the same image through several models (comma-separated) - accuracy vs latency side by side."""
    _check_thresholds(conf, iou)
    img = await _read_image(file)
    names = [m.strip() for m in models.split(",") if m.strip()]
    if not 1 <= len(names) <= 4:
        raise HTTPException(422, "pass 1-4 model names")
    results = []
    for n in names:
        name, det = _detector(n)
        det(img, conf=conf, iou=iou)                 # one untimed run so caches are warm for a fair comparison
        results.append(_detect(img, name, det, conf, iou))
    return CompareResult(results=results)


@router.post("/detect/video", response_model=RunOut, status_code=202)
async def detect_video(file: UploadFile = File(...), model: str | None = Form(None),
                       conf: float = Form(0.25), line: str | None = Form(None),
                       db: Session = Depends(get_db)):
    """Queue a video for tracking + analytics. `line` = JSON [x1,y1,x2,y2] in 0..1 image coords."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"unsupported video type '{ext}'")
    if not 0 < conf < 1:
        raise HTTPException(422, "conf must be in (0, 1)")
    line_v = None
    if line:
        try:
            line_v = [float(v) for v in json.loads(line)]
            assert len(line_v) == 4 and all(0 <= v <= 1 for v in line_v)
        except Exception:
            raise HTTPException(422, "line must be a JSON list of 4 numbers in [0, 1]")
    name, _ = _detector(model)
    s = get_settings()
    path = s.data_dir / "uploads" / f"{uuid.uuid4().hex}{ext}"
    size = 0
    with path.open("wb") as fh:                       # stream to disk in chunks - never hold a video in RAM
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > s.max_upload_mb * 2**20:
                fh.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            fh.write(chunk)
    run = Run(kind="video", model=name, source_name=file.filename or path.name, status="queued",
              progress=0.0, conf=conf, line=line_v, media_path=str(path))
    db.add(run)
    db.commit()
    video_jobs.submit(run.id, path)
    return run
