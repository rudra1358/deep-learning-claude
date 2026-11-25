"""Pydantic models = the contract between the Express server and this ML service (shown at /docs)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    box: list[float]
    score: float
    cls: int
    label: str


class Timing(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class ImageResult(BaseModel):
    model: str
    provider: str
    width: int
    height: int
    detections: list[DetectionOut]
    counts: dict[str, int]
    timing: Timing


class CompareResult(BaseModel):
    results: list[ImageResult]


class VideoJobIn(BaseModel):
    """Sent by the Express server after it saved an upload and created the run in MongoDB."""
    run_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    video_path: str                      # absolute path inside VA_DATA_DIR/uploads
    model: str | None = None
    conf: float = Field(0.25, gt=0, lt=1)
    line: list[float] | None = None      # [x1, y1, x2, y2] normalised 0..1


class JobOut(BaseModel):
    run_id: str
    status: str                          # queued | running | done | failed
    progress: float
    error: str | None = None
    model: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    result: dict | None = None           # only when done: frames, avg_latency_ms, summary, tracks, events, samples
