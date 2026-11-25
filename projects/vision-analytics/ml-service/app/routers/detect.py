"""Image detection and model comparison.

These endpoints only run the model and return the result - nothing is stored here.
The Express server forwards the upload to us, then saves the result as a run in MongoDB.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import get_settings
from ..models_registry import get_registry
from ..schemas import CompareResult, ImageResult

router = APIRouter(tags=["detection"])

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def get_detector(name: str | None):
    """(name, detector) for a model name, or 404 with the list of available models."""
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
    return ImageResult(model=name, provider=det.provider, width=img.shape[1], height=img.shape[0],
                       detections=[d.__dict__ for d in dets], counts=dict(Counter(d.label for d in dets)),
                       timing=timing)


@router.post("/detect/image", response_model=ImageResult)
async def detect_image(file: UploadFile = File(...), model: str | None = Form(None),
                       conf: float = Form(0.25), iou: float = Form(0.45)):
    _check_thresholds(conf, iou)
    img = await _read_image(file)
    name, det = get_detector(model)
    return _detect(img, name, det, conf, iou)


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
        name, det = get_detector(n)
        det(img, conf=conf, iou=iou)                 # one untimed run so caches are warm for a fair comparison
        results.append(_detect(img, name, det, conf, iou))
    return CompareResult(results=results)
