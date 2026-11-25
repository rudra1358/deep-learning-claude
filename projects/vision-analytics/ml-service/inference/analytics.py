"""Turning tracks into business numbers: line crossings, occupancy and dwell time.

Line crossing: a counting line goes from A to B. For a point P the sign of the 2-D cross
product  (B - A) x (P - A)  says which side of the line P is on. When a track's centre
changes side between two frames AND the movement segment actually intersects the line
segment (not just its infinite extension), the object crossed. Direction:
  negative -> positive side = "in",  positive -> negative = "out".
Each track is counted at most once per direction, so jitter on the line does not
inflate the count.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .tracker import Track

Point = tuple[float, float]


def _side(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    d1, d2 = _side(a, b, p1), _side(a, b, p2)
    d3, d4 = _side(p1, p2, a), _side(p1, p2, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class LineCounter:
    a: Point
    b: Point
    counts_in: Counter = field(default_factory=Counter)
    counts_out: Counter = field(default_factory=Counter)
    _last: dict[int, Point] = field(default_factory=dict)
    _counted: set[tuple[int, str]] = field(default_factory=set)

    def update(self, tracks: list[Track], names: dict[int, str]) -> list[dict]:
        """Returns the crossing events produced in this frame."""
        events = []
        for t in tracks:
            p = t.centre
            prev = self._last.get(t.track_id)
            self._last[t.track_id] = p
            if prev is None or not segments_intersect(prev, p, self.a, self.b):
                continue
            direction = "in" if _side(self.a, self.b, prev) < 0 else "out"
            key = (t.track_id, direction)
            if key in self._counted:
                continue
            self._counted.add(key)
            label = names.get(t.cls, str(t.cls))
            (self.counts_in if direction == "in" else self.counts_out)[label] += 1
            events.append({"track_id": t.track_id, "label": label, "direction": direction})
        return events

    def summary(self) -> dict:
        return {"in": dict(self.counts_in), "out": dict(self.counts_out)}


@dataclass
class SceneAnalytics:
    """Per-stream accumulator: current occupancy, unique objects and dwell time."""
    fps: float = 30.0
    line: LineCounter | None = None
    first_seen: dict[int, int] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)
    track_label: dict[int, str] = field(default_factory=dict)
    frames: int = 0

    def update(self, tracks: list[Track], names: dict[int, str]) -> dict:
        self.frames += 1
        occupancy: Counter = Counter()
        for t in tracks:
            label = names.get(t.cls, str(t.cls))
            occupancy[label] += 1
            self.first_seen.setdefault(t.track_id, self.frames)
            self.last_seen[t.track_id] = self.frames
            self.track_label[t.track_id] = label
        events = self.line.update(tracks, names) if self.line else []
        return {"occupancy": dict(occupancy), "events": events}

    def summary(self) -> dict:
        unique: Counter = Counter(self.track_label.values())
        dwell: dict[str, list[float]] = defaultdict(list)
        for tid, label in self.track_label.items():
            dwell[label].append((self.last_seen[tid] - self.first_seen[tid] + 1) / self.fps)
        return {
            "frames": self.frames,
            "unique_objects": dict(unique),
            "avg_dwell_seconds": {k: round(sum(v) / len(v), 2) for k, v in dwell.items()},
            "line": self.line.summary() if self.line else None,
        }
