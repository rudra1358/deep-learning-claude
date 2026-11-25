"""Loads every .onnx file in MODELS_DIR once and keeps it in memory.

Loading an ONNX session takes ~0.1-1 s and allocates GPU memory, so we do it lazily on
first use and cache it - never per request. ONNX Runtime sessions are thread-safe for
`run()`, so one session can serve concurrent requests.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from inference.detector import OnnxDetector

from .config import get_settings


class ModelRegistry:
    def __init__(self, models_dir: Path, device: str) -> None:
        self.models_dir = models_dir
        self.device = device
        self._cache: dict[str, OnnxDetector] = {}
        self._lock = threading.Lock()

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.models_dir.glob("*.onnx"))

    def get(self, name: str) -> OnnxDetector:
        if name not in self._cache:
            with self._lock:                     # double-checked locking: load once even under concurrency
                if name not in self._cache:
                    path = self.models_dir / f"{name}.onnx"
                    if not path.exists():
                        raise KeyError(name)
                    det = OnnxDetector(path, device=self.device)
                    det.warmup()
                    self._cache[name] = det
        return self._cache[name]

    def describe(self) -> list[dict]:
        metrics = {}
        mf = get_settings().metrics_file
        if mf.exists():
            metrics = json.loads(mf.read_text()).get("models", {})
        out = []
        for name in self.available():
            info = {"name": name, "loaded": name in self._cache, "metrics": metrics.get(name)}
            if name in self._cache:
                d = self._cache[name]
                info.update(provider=d.provider, imgsz=d.imgsz, classes=d.names)
            out.append(info)
        return out


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        s = get_settings()
        _registry = ModelRegistry(s.models_dir, s.device)
    return _registry
