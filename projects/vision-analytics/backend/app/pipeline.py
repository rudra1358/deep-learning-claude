"""The per-stream pipeline shared by video uploads and live webcam streams:

    frame -> detector (ONNX) -> tracker (ByteTrack) -> analytics (occupancy, line crossings)

A Pipeline object holds the *state* of one stream (tracker, counters), so every video
job and every WebSocket connection gets its own instance. The detector itself is shared.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from inference.analytics import LineCounter, SceneAnalytics
from inference.detector import OnnxDetector
from inference.tracker import ByteTracker, TrackerConfig


class Pipeline:
    def __init__(self, detector: OnnxDetector, width: int, height: int, fps: float,
                 line: list[float] | None = None, conf: float = 0.25) -> None:
        self.detector = detector
        # Low detection threshold on purpose: ByteTrack needs low-score boxes for its 2nd round.
        # Only boxes >= `conf` are *displayed* / counted as new objects.
        self.det_conf = min(conf, 0.1)
        self.tracker = ByteTracker(TrackerConfig(high_thresh=max(conf, 0.3), new_track_thresh=max(conf, 0.3) + 0.1))
        lc = None
        if line:
            x1, y1, x2, y2 = line                         # normalised -> pixels
            lc = LineCounter((x1 * width, y1 * height), (x2 * width, y2 * height))
        self.scene = SceneAnalytics(fps=fps or 30.0, line=lc)
        self.latencies: list[float] = []
        self.track_meta: dict[int, dict] = {}

    def process(self, frame_bgr: np.ndarray) -> dict:
        dets, timing = self.detector(frame_bgr, conf=self.det_conf)
        if dets:
            boxes = np.array([d.box for d in dets], float)
            scores = np.array([d.score for d in dets])
            classes = np.array([d.cls for d in dets])
        else:
            boxes, scores, classes = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        tracks = self.tracker.update(boxes, scores, classes)
        frame_stats = self.scene.update(tracks, self.detector.names)
        self.latencies.append(timing["total_ms"])

        out_tracks = []
        for t in tracks:
            label = self.detector.names.get(t.cls, str(t.cls))
            meta = self.track_meta.setdefault(t.track_id, {"label": label, "first": self.scene.frames,
                                                           "max_score": 0.0})
            meta["last"] = self.scene.frames
            meta["max_score"] = max(meta["max_score"], t.score)
            out_tracks.append({"id": t.track_id, "label": label, "score": round(t.score, 3),
                               "box": [round(float(v), 1) for v in t.xyxy],
                               "trail": [[round(x, 1), round(y, 1)] for x, y in t.history[-15:]]})
        return {"frame": self.scene.frames, "tracks": out_tracks, "timing": timing, **frame_stats}

    @property
    def avg_latency(self) -> float:
        return round(float(np.mean(self.latencies)), 2) if self.latencies else 0.0

    def summary(self) -> dict:
        s = self.scene.summary()
        s["avg_latency_ms"] = self.avg_latency
        return s


def peak_counts(samples) -> dict:
    """Highest count ever seen per class, e.g. [{"car": 2}, {"car": 5, "bus": 1}] -> {"car": 5, "bus": 1}."""
    peak: Counter = Counter()
    for frame in samples:
        for k, v in frame.items():
            peak[k] = max(peak[k], v)
    return dict(peak)
