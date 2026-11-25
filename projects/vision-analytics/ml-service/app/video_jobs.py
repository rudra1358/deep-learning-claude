"""Background processing of uploaded videos.

An HTTP request must return quickly, but a 2-minute video takes a while to analyse. So the
Express server saves the upload, creates a run in MongoDB with status "queued" and calls
POST /jobs/video. We put the job on a single-worker thread pool and return immediately.
The server then polls GET /jobs/{run_id} and, when the job is done, copies the result
(tracks, line-crossing events, per-second counts, summary) into MongoDB.

Job state lives in memory (a dict). That is fine because MongoDB - not this service - is the
system of record: if this process restarts, the server sees "job not found" and marks the
run as failed. (At larger scale the same design maps onto a real queue - Celery/RQ + Redis.)

Files written by a job (big, so they stay on disk instead of in the database):
  data/runs/<run_id>/frames.json   - per-frame tracks, used by the frontend to draw boxes
                                     over the video while it plays
  data/runs/<run_id>/playback.webm - browser-playable copy of exactly the analysed frames
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from .config import get_settings
from .models_registry import get_registry
from .pipeline import Pipeline, peak_counts

log = logging.getLogger("video_jobs")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()                      # the worker thread writes, request threads read


def submit(run_id: str, video_path: Path, model: str, conf: float, line: list[float] | None) -> dict:
    job = {"run_id": run_id, "status": "queued", "progress": 0.0, "error": None, "model": model,
           "conf": conf, "line": line, "width": 0, "height": 0, "fps": 0.0, "result": None}
    with _lock:
        _jobs[run_id] = job
    _executor.submit(_process, run_id, video_path)
    return get(run_id)


def get(run_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(run_id)
        return dict(job) if job else None     # a copy, so the caller never sees a half-written update


def forget(run_id: str) -> None:
    with _lock:
        _jobs.pop(run_id, None)


def _update(run_id: str, **fields) -> None:
    with _lock:
        if run_id in _jobs:
            _jobs[run_id].update(fields)


def _process(run_id: str, video_path: Path) -> None:
    try:
        _process_run(run_id, video_path)
    except Exception as exc:  # noqa: BLE001 - any failure must be reported on the job
        log.error("run %s failed: %s", run_id, traceback.format_exc())
        _update(run_id, status="failed", error=str(exc)[:500])


def _process_run(run_id: str, video_path: Path) -> None:
    s = get_settings()
    job = get(run_id)
    if job is None:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("could not open video (unsupported codec?)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(s.max_video_seconds * fps)
    _update(run_id, status="running", width=width, height=height, fps=fps)

    pipe = Pipeline(get_registry().get(job["model"]), width, height, fps, job["line"], job["conf"])
    frames_out: list[dict] = []
    second_peak: dict[int, Counter] = {}
    events: list[dict] = []

    out_dir = s.data_dir / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Browsers can't play many upload formats (AVI, MPEG-4 Part 2, ...). Re-encode the exact
    # frames we analyse to WebM/VP8 (plays in every modern browser) so the overlay in the
    # dashboard lines up frame-for-frame with frames.json.
    writer = cv2.VideoWriter(str(out_dir / "playback.webm"), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        writer = None

    while len(frames_out) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is not None:
            writer.write(frame)
        r = pipe.process(frame)
        t = (r["frame"] - 1) / fps
        frames_out.append({"t": round(t, 3), "tracks": r["tracks"]})
        sec = int(t)
        peak = second_peak.setdefault(sec, Counter())
        for label, n in r["occupancy"].items():
            peak[label] = max(peak[label], n)
        for e in r["events"]:
            events.append({"track_id": e["track_id"], "label": e["label"], "direction": e["direction"],
                           "t_seconds": round(t, 2)})
        if r["frame"] % 25 == 0 and total:
            _update(run_id, progress=min(r["frame"] / min(total, max_frames), 0.99))
    cap.release()
    if writer is not None:
        writer.release()
    if not frames_out:
        raise ValueError("video contains no readable frames")

    (out_dir / "frames.json").write_text(json.dumps({"fps": fps, "width": width, "height": height,
                                                     "frames": frames_out}, separators=(",", ":")))
    summary = pipe.summary()
    summary["peak_occupancy"] = peak_counts(second_peak.values())
    result = {
        "frames": len(frames_out),
        "avg_latency_ms": pipe.avg_latency,
        "summary": summary,
        "tracks": track_rows(pipe),
        "events": events,
        "samples": sample_rows(second_peak),
    }
    _update(run_id, status="done", progress=1.0, result=result)
    log.info("run %s done: %d frames, %.1f ms/frame", run_id, len(frames_out), pipe.avg_latency)


def track_rows(pipe: Pipeline) -> list[dict]:
    """One row per tracked object: when it was first/last seen and its best score."""
    return [{"track_id": tid, "label": m["label"], "first_frame": m["first"], "last_frame": m["last"],
             "max_score": round(m["max_score"], 4)} for tid, m in pipe.track_meta.items()]


def sample_rows(second_peak: dict[int, Counter]) -> list[dict]:
    """Per-second peak count of each class -> the 'counts over time' chart."""
    return [{"t_seconds": float(sec), "label": label, "count": n}
            for sec, c in sorted(second_peak.items()) for label, n in c.items()]
