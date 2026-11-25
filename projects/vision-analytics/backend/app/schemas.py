"""Pydantic response models = the API contract (also rendered at /docs by FastAPI)."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_validator


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
    run_id: int | None
    model: str
    provider: str
    width: int
    height: int
    detections: list[DetectionOut]
    counts: dict[str, int]
    timing: Timing


class CompareResult(BaseModel):
    results: list[ImageResult]


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: str
    model: str
    source_name: str
    status: str
    progress: float
    error: str | None
    width: int
    height: int
    fps: float
    frames: int
    avg_latency_ms: float
    conf: float
    line: list[float] | None
    summary: dict
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        # SQLite drops tz info; values are always stored in UTC, so mark them explicitly
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


class RunPage(BaseModel):
    total: int
    items: list[RunOut]


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    track_id: int
    label: str
    direction: str
    t_seconds: float


class TimeseriesOut(BaseModel):
    labels: list[str]
    t: list[float]
    series: dict[str, list[int]]


class OverviewOut(BaseModel):
    runs_by_kind: dict[str, int]
    objects_by_label: dict[str, int]
    crossings: dict[str, dict[str, int]]
    avg_latency_ms: float | None
