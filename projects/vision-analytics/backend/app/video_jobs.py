"""Background processing of uploaded videos.

An HTTP request must return quickly, but a 2-minute video takes a while to analyse. So the
upload endpoint only saves the file, creates a `runs` row with status "queued" and submits
a job to a single-worker thread pool. The client polls GET /api/runs/{id} for progress.

(At larger scale the same design maps onto a real queue - Celery/RQ + Redis - with the
worker in its own container; the DB row is the contract between API and worker.)

Output of a job:
  * data/runs/<id>/frames.json  - per-frame tracks, used by the frontend to draw boxes
                                  over the original video while it plays
  * DB rows: tracks, events, per-second count samples, and a summary on the run
"""
from __future__ import annotations

import json
import logging
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from .config import get_settings
from .db import CountSample, EventRow, Run, TrackRow, session_factory
from .models_registry import get_registry
from .pipeline import Pipeline, peak_counts

log = logging.getLogger("video_jobs")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker")


def submit(run_id: int, video_path: Path) -> None:
    _executor.submit(_process, run_id, video_path)


def _process(run_id: int, video_path: Path) -> None:
    Session = session_factory()
    with Session() as db:
        run = db.get(Run, run_id)
        if run is None:
            return
        try:
            _process_run(db, run, video_path)
        except Exception as exc:  # noqa: BLE001 - any failure must be recorded on the run
            log.error("run %s failed: %s", run_id, traceback.format_exc())
            db.rollback()
            run = db.get(Run, run_id)
            run.status, run.error = "failed", str(exc)[:500]
            db.commit()


def _process_run(db, run: Run, video_path: Path) -> None:
    s = get_settings()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("could not open video (unsupported codec?)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(s.max_video_seconds * fps)
    run.status, run.width, run.height, run.fps, run.progress = "running", width, height, fps, 0.0
    db.commit()

    pipe = Pipeline(get_registry().get(run.model), width, height, fps, run.line, run.conf)
    frames_out: list[dict] = []
    second_peak: dict[int, Counter] = {}
    events: list[EventRow] = []

    out_dir = s.data_dir / "runs" / str(run.id)
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
            events.append(EventRow(track_id=e["track_id"], label=e["label"], direction=e["direction"],
                                   t_seconds=round(t, 2)))
        if r["frame"] % 25 == 0 and total:
            run.progress = min(r["frame"] / min(total, max_frames), 0.99)
            db.commit()
    cap.release()
    if writer is not None:
        writer.release()
    if not frames_out:
        raise ValueError("video contains no readable frames")

    (out_dir / "frames.json").write_text(json.dumps({"fps": fps, "width": width, "height": height,
                                                     "frames": frames_out}, separators=(",", ":")))

    run.tracks = [TrackRow(track_id=tid, label=m["label"], first_frame=m["first"], last_frame=m["last"],
                           max_score=round(m["max_score"], 4)) for tid, m in pipe.track_meta.items()]
    run.events = events
    run.samples = [CountSample(t_seconds=float(sec), label=label, count=n)
                   for sec, c in sorted(second_peak.items()) for label, n in c.items()]
    summary = pipe.summary()
    summary["peak_occupancy"] = peak_counts(second_peak.values())
    run.summary = summary
    run.frames = len(frames_out)
    run.avg_latency_ms = pipe.avg_latency
    run.status, run.progress = "done", 1.0
    db.commit()
    log.info("run %s done: %d frames, %.1f ms/frame", run.id, run.frames, run.avg_latency_ms)
