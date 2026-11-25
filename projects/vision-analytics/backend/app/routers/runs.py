"""History, analytics and export endpoints."""
from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import CountSample, Detection, EventRow, Run, TrackRow, get_db
from ..schemas import EventOut, OverviewOut, RunOut, RunPage, TimeseriesOut

router = APIRouter(prefix="/api", tags=["history"])


def _run_or_404(db: Session, run_id: int) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@router.get("/runs", response_model=RunPage)
def list_runs(kind: str | None = None, label: str | None = None, q: str | None = None,
              limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
              db: Session = Depends(get_db)):
    """Search history: filter by kind, by a detected label, or by file name."""
    stmt = select(Run)
    if kind:
        stmt = stmt.where(Run.kind == kind)
    if q:
        stmt = stmt.where(Run.source_name.ilike(f"%{q}%"))
    if label:
        has_det = select(Detection.run_id).where(Detection.label == label)
        has_trk = select(TrackRow.run_id).where(TrackRow.label == label)
        stmt = stmt.where(or_(Run.id.in_(has_det), Run.id.in_(has_trk)))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(stmt.order_by(Run.created_at.desc(), Run.id.desc()).limit(limit).offset(offset)).all()
    return RunPage(total=total, items=items)


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    return _run_or_404(db, run_id)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: int, db: Session = Depends(get_db)):
    run = _run_or_404(db, run_id)
    if run.status in ("queued", "running"):
        raise HTTPException(409, "run is still processing")
    if run.media_path:
        Path(run.media_path).unlink(missing_ok=True)
    shutil.rmtree(get_settings().data_dir / "runs" / str(run.id), ignore_errors=True)
    db.delete(run)
    db.commit()


@router.get("/runs/{run_id}/detections")
def run_detections(run_id: int, db: Session = Depends(get_db)):
    run = _run_or_404(db, run_id)
    return [{"label": d.label, "score": d.score, "box": [d.x1, d.y1, d.x2, d.y2]} for d in run.detections]


@router.get("/runs/{run_id}/frames")
def run_frames(run_id: int, db: Session = Depends(get_db)):
    run = _run_or_404(db, run_id)
    p = get_settings().data_dir / "runs" / str(run.id) / "frames.json"
    if run.status != "done" or not p.exists():
        raise HTTPException(409, f"run status is '{run.status}'")
    return FileResponse(p, media_type="application/json")


@router.get("/runs/{run_id}/video")
def run_video(run_id: int, db: Session = Depends(get_db)):
    """Browser-playable copy (WebM made by the worker) if present, else the original upload.
    FileResponse supports HTTP Range requests, so seeking works in the <video> element."""
    run = _run_or_404(db, run_id)
    playback = get_settings().data_dir / "runs" / str(run.id) / "playback.webm"
    if playback.exists() and run.status == "done":
        return FileResponse(playback, media_type="video/webm")
    if not run.media_path or not Path(run.media_path).exists():
        raise HTTPException(404, "no stored video for this run")
    return FileResponse(run.media_path)


@router.get("/runs/{run_id}/events", response_model=list[EventOut])
def run_events(run_id: int, db: Session = Depends(get_db)):
    _run_or_404(db, run_id)
    return db.scalars(select(EventRow).where(EventRow.run_id == run_id).order_by(EventRow.t_seconds)).all()


@router.get("/runs/{run_id}/timeseries", response_model=TimeseriesOut)
def run_timeseries(run_id: int, db: Session = Depends(get_db)):
    """Per-second peak count of each class - drives the 'counts over time' chart."""
    _run_or_404(db, run_id)
    rows = db.scalars(select(CountSample).where(CountSample.run_id == run_id)).all()
    ts = sorted({r.t_seconds for r in rows})
    labels = sorted({r.label for r in rows})
    idx = {t: i for i, t in enumerate(ts)}
    series = {lb: [0] * len(ts) for lb in labels}
    for r in rows:
        series[r.label][idx[r.t_seconds]] = r.count
    return TimeseriesOut(labels=labels, t=ts, series=series)


@router.get("/runs/{run_id}/export.csv")
def export_csv(run_id: int, db: Session = Depends(get_db)):
    run = _run_or_404(db, run_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    if run.kind == "image":
        w.writerow(["label", "score", "x1", "y1", "x2", "y2"])
        for d in run.detections:
            w.writerow([d.label, d.score, d.x1, d.y1, d.x2, d.y2])
    else:
        w.writerow(["track_id", "label", "first_frame", "last_frame", "seconds_visible", "max_score"])
        fps = run.fps or 30.0
        for t in sorted(run.tracks, key=lambda t: t.track_id):
            w.writerow([t.track_id, t.label, t.first_frame, t.last_frame,
                        round((t.last_frame - t.first_frame + 1) / fps, 2), t.max_score])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="run-{run.id}.csv"'})


@router.get("/analytics/overview", response_model=OverviewOut)
def overview(db: Session = Depends(get_db)):
    """Dashboard totals across every run (aggregation done in SQL, not Python)."""
    kinds = dict(db.execute(select(Run.kind, func.count()).group_by(Run.kind)).all())
    det_labels = dict(db.execute(select(Detection.label, func.count()).group_by(Detection.label)).all())
    trk_labels = dict(db.execute(select(TrackRow.label, func.count()).group_by(TrackRow.label)).all())
    objects = {k: det_labels.get(k, 0) + trk_labels.get(k, 0) for k in set(det_labels) | set(trk_labels)}
    crossings: dict[str, dict[str, int]] = {"in": {}, "out": {}}
    for label, direction, n in db.execute(
            select(EventRow.label, EventRow.direction, func.count()).group_by(EventRow.label, EventRow.direction)):
        crossings[direction][label] = n
    avg = db.scalar(select(func.avg(Run.avg_latency_ms)).where(Run.status == "done", Run.avg_latency_ms > 0))
    return OverviewOut(runs_by_kind=kinds, objects_by_label=dict(sorted(objects.items(), key=lambda kv: -kv[1])),
                       crossings=crossings, avg_latency_ms=round(avg, 2) if avg else None)
