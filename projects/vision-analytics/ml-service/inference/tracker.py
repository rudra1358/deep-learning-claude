"""Multi-object tracker in the style of ByteTrack (Zhang et al., 2022).

Problem: the detector looks at every frame independently, so it cannot tell that the
car in frame 10 is the same car as in frame 11. The tracker links detections across
frames and gives each object a persistent ID.

Key idea of ByteTrack: do not throw low-confidence detections away. An object that is
partly occluded often still gets a *low* score. So matching runs in two rounds:
  round 1: existing tracks  <->  high-score detections
  round 2: tracks still unmatched <-> low-score detections (recovers occluded objects)

Each round:
  1. a Kalman filter predicts where every track should be in this frame
  2. cost matrix = 1 - IoU(predicted box, detection box)   (+ large cost if classes differ)
  3. Hungarian algorithm (linear_sum_assignment) finds the cheapest one-to-one matching
  4. pairs whose cost is above a threshold are rejected

Track lifecycle:  new (tentative) -> tracked -> lost (kept `track_buffer` frames, can be
re-found) -> removed. A new track is only confirmed after it is matched in a second frame,
which suppresses one-frame false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah
from .postprocess import box_iou


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    track_id: int
    cls: int
    score: float
    mean: np.ndarray
    cov: np.ndarray
    state: TrackState = TrackState.NEW
    is_activated: bool = False
    start_frame: int = 0
    frame_id: int = 0            # last frame this track was matched
    hits: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)  # centre points

    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def centre(self) -> tuple[float, float]:
        return float(self.mean[0]), float(self.mean[1])


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5     # detections above this take part in round 1
    low_thresh: float = 0.1      # detections between low and high take part in round 2
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8    # max cost (1 - IoU) accepted in round 1
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    track_buffer: int = 30       # frames a lost track is kept alive
    history_len: int = 30


def _match(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return (matches,
            [r for r in range(cost.shape[0]) if r not in mr],
            [c for c in range(cost.shape[1]) if c not in mc])


class ByteTracker:
    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []        # NEW + TRACKED
        self.lost: list[Track] = []
        self.frame_id = 0
        self._next_id = 1

    def reset(self) -> None:
        self.__init__(self.cfg)

    # ----------------------------------------------------------------- helpers
    def _cost(self, tracks: list[Track], boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if not tracks or len(boxes) == 0:
            return np.zeros((len(tracks), len(boxes)))
        tboxes = np.stack([t.xyxy for t in tracks])
        iou = box_iou(tboxes, boxes)
        cost = 1.0 - iou * scores[None, :]          # "fuse score": prefer confident detections
        tcls = np.array([t.cls for t in tracks])
        cost[tcls[:, None] != classes[None, :]] = 1e6  # never match across classes
        return cost

    def _update_track(self, t: Track, box: np.ndarray, score: float) -> None:
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(box))
        t.score = score
        t.frame_id = self.frame_id
        t.hits += 1
        t.state = TrackState.TRACKED
        t.is_activated = True
        t.history.append(t.centre)
        del t.history[:-self.cfg.history_len]

    # ----------------------------------------------------------------- main step
    def update(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[Track]:
        """Feed one frame of detections; returns the confirmed, currently-visible tracks."""
        cfg = self.cfg
        self.frame_id += 1
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        classes = np.asarray(classes, np.int64).reshape(-1)

        high = scores >= cfg.high_thresh
        low = (scores >= cfg.low_thresh) & ~high
        hb, hs, hc = boxes[high], scores[high], classes[high]
        lb, ls, lc = boxes[low], scores[low], classes[low]

        confirmed = [t for t in self.tracks if t.is_activated]
        unconfirmed = [t for t in self.tracks if not t.is_activated]
        pool = confirmed + self.lost

        for t in pool + unconfirmed:                     # 1. Kalman predict
            if t.state != TrackState.TRACKED:
                t.mean[7] = 0                           # freeze height velocity while not seen
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        # 2. round 1: all confirmed + lost tracks vs high-score detections
        matches, u_trk, u_det = _match(self._cost(pool, hb, hc, hs), cfg.match_thresh)
        for ti, di in matches:
            self._update_track(pool[ti], hb[di], hs[di])

        # 3. round 2: still-unmatched *tracked* tracks vs low-score detections (plain IoU)
        remain = [pool[i] for i in u_trk if pool[i].state == TrackState.TRACKED]
        cost2 = self._cost(remain, lb, lc, np.ones(len(lb)))
        matches2, u_trk2, _ = _match(cost2, cfg.second_match_thresh)
        for ti, di in matches2:
            self._update_track(remain[ti], lb[di], ls[di])
        for i in u_trk2:
            remain[i].state = TrackState.LOST

        # 4. tentative tracks vs leftover high detections
        hb2, hs2, hc2 = hb[u_det], hs[u_det], hc[u_det]
        matches3, u_unc, u_det2 = _match(self._cost(unconfirmed, hb2, hc2, hs2), cfg.unconfirmed_match_thresh)
        for ti, di in matches3:
            self._update_track(unconfirmed[ti], hb2[di], hs2[di])
        for i in u_unc:
            unconfirmed[i].state = TrackState.REMOVED

        # 5. start new tentative tracks from confident unmatched detections
        new_tracks: list[Track] = []
        for di in u_det2:
            if hs2[di] < cfg.new_track_thresh:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(hb2[di]))
            t = Track(self._next_id, int(hc2[di]), float(hs2[di]), mean, cov,
                      start_frame=self.frame_id, frame_id=self.frame_id)
            if self.frame_id == 1:                       # first frame: trust detections immediately
                t.is_activated, t.state = True, TrackState.TRACKED
            t.history.append(t.centre)
            self._next_id += 1
            new_tracks.append(t)

        # 6. bookkeeping: move tracks between the active and lost lists
        everything = pool + unconfirmed + new_tracks
        for t in everything:
            if t.state == TrackState.LOST and self.frame_id - t.frame_id > cfg.track_buffer:
                t.state = TrackState.REMOVED
        self.tracks = [t for t in everything if t.state in (TrackState.NEW, TrackState.TRACKED)]
        self.lost = [t for t in everything if t.state == TrackState.LOST]

        return [t for t in self.tracks if t.is_activated and t.frame_id == self.frame_id]
