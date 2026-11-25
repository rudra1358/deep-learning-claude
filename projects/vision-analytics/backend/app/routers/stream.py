"""Live webcam analysis over a WebSocket.

Protocol (one connection = one camera session):
  client -> server  text  {"type":"config","model":"...","conf":0.3,"line":[x1,y1,x2,y2]|null,
                           "width":W,"height":H}
  client -> server  bytes one JPEG-encoded frame
  server -> client  text  {"type":"result","frame":n,"tracks":[...],"occupancy":{...},
                           "events":[...],"timing":{...},"server_fps":..}
  client -> server  text  {"type":"stop"}  -> server saves the session as a run and replies
                          {"type":"saved","run_id":id,"summary":{...}}

Back-pressure: the browser sends the next frame only after it receives the result of the
previous one. So a slow server simply lowers the frame rate instead of building up an
ever-growing queue of stale frames (which would make the overlay lag behind reality).
Inference is CPU/GPU-bound, so it runs in a worker thread (run_in_threadpool) to keep the
asyncio event loop free for other connections.
"""
from __future__ import annotations

import json
import time
from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from ..config import get_settings
from ..db import CountSample, EventRow, Run, TrackRow, session_factory
from ..models_registry import get_registry
from ..pipeline import Pipeline, peak_counts

router = APIRouter()
MAX_FRAME_BYTES = 4 * 2**20


def _save(pipe: Pipeline, model: str, conf: float, line, w: int, h: int, fps: float,
          samples: dict[int, Counter], events: list[dict]) -> tuple[int, dict]:
    summary = pipe.summary()
    summary["peak_occupancy"] = peak_counts(samples.values())
    with session_factory()() as db:
        run = Run(kind="stream", model=model, source_name="webcam", width=w, height=h, fps=round(fps, 2),
                  frames=pipe.scene.frames, avg_latency_ms=pipe.avg_latency, conf=conf, line=line,
                  summary=summary)
        run.tracks = [TrackRow(track_id=tid, label=m["label"], first_frame=m["first"], last_frame=m["last"],
                               max_score=round(m["max_score"], 4)) for tid, m in pipe.track_meta.items()]
        run.events = [EventRow(**e) for e in events]
        run.samples = [CountSample(t_seconds=float(s), label=k, count=v)
                       for s, c in sorted(samples.items()) for k, v in c.items()]
        db.add(run)
        db.commit()
        return run.id, summary


@router.websocket("/api/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    pipe: Pipeline | None = None
    cfg: dict = {}
    t_start = time.perf_counter()
    samples: dict[int, Counter] = {}
    events: list[dict] = []
    recent: list[float] = []
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "config":
                    name = data.get("model") or get_settings().default_model
                    try:
                        det = await run_in_threadpool(get_registry().get, name)
                    except KeyError:
                        await ws.send_json({"type": "error", "message": f"model '{name}' not found"})
                        continue
                    line = data.get("line")
                    conf = data.get("conf", 0.3)
                    if line is not None and not (isinstance(line, list) and len(line) == 4 and all(
                            isinstance(v, (int, float)) and 0 <= v <= 1 for v in line)):
                        await ws.send_json({"type": "error", "message": "line must be 4 numbers in [0, 1]"})
                        continue
                    if not (isinstance(conf, (int, float)) and 0 < conf < 1):
                        await ws.send_json({"type": "error", "message": "conf must be in (0, 1)"})
                        continue
                    cfg = {"model": name, "conf": float(conf), "line": line,
                           "w": int(data.get("width", 640)), "h": int(data.get("height", 480))}
                    pipe = Pipeline(det, cfg["w"], cfg["h"], fps=15.0, line=cfg["line"], conf=cfg["conf"])
                    t_start, samples, events = time.perf_counter(), {}, []
                    await ws.send_json({"type": "ready", "model": name, "provider": det.provider})
                elif data.get("type") == "stop":
                    if pipe and pipe.scene.frames:
                        elapsed = time.perf_counter() - t_start
                        pipe.scene.fps = pipe.scene.frames / max(elapsed, 1e-6)  # real fps -> correct dwell times
                        run_id, summary = await run_in_threadpool(
                            _save, pipe, cfg["model"], cfg["conf"], cfg["line"], cfg["w"], cfg["h"],
                            pipe.scene.fps, samples, events)
                        await ws.send_json({"type": "saved", "run_id": run_id, "summary": summary})
                    else:                              # nothing processed -> nothing to save, but always answer
                        await ws.send_json({"type": "saved", "run_id": None, "summary": None})
                    pipe = None
                continue

            frame_bytes = msg.get("bytes")
            if pipe is None or not frame_bytes:
                await ws.send_json({"type": "error", "message": "send a config message first"})
                continue
            if len(frame_bytes) > MAX_FRAME_BYTES:
                await ws.send_json({"type": "error", "message": "frame too large"})
                continue
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"type": "error", "message": "could not decode frame"})
                continue
            result = await run_in_threadpool(pipe.process, frame)
            now = time.perf_counter()
            t = now - t_start
            c = samples.setdefault(int(t), Counter())
            for k, v in result["occupancy"].items():
                c[k] = max(c[k], v)
            events += [{**e, "t_seconds": round(t, 2)} for e in result["events"]]
            recent = [x for x in recent if now - x < 2.0] + [now]
            await ws.send_json({"type": "result", **result, "server_fps": round(len(recent) / 2.0, 1)})
    except WebSocketDisconnect:
        pass
