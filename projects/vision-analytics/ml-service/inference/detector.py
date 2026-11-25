"""ONNX Runtime detector: the only place the neural network is executed at serving time.

Why ONNX Runtime instead of PyTorch at serving time?
  * the server does not need PyTorch or Ultralytics installed (much smaller Docker image)
  * graph optimisations (operator fusion, constant folding) make CPU inference faster
  * the same .onnx file runs on CPU, CUDA, TensorRT, DirectML ... by switching "providers"
"""
from __future__ import annotations

import ast
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .postprocess import decode_yolo
from .preprocess import to_input_tensor

log = logging.getLogger("detector")


@dataclass
class Detection:
    box: list[float]   # x1, y1, x2, y2 in original-image pixels
    score: float
    cls: int
    label: str


def _providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    if device in ("cuda", "auto") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError(f"CUDA requested but ONNX Runtime only has {available}")
    return ["CPUExecutionProvider"]


class OnnxDetector:
    def __init__(self, path: str | Path, device: str = "auto", conf: float = 0.25, iou: float = 0.45) -> None:
        self.path = Path(path)
        if device in ("cuda", "auto"):
            try:  # onnxruntime-gpu >= 1.21 can load CUDA/cuDNN DLLs shipped with the torch / nvidia wheels
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - not available / not needed on CPU builds
                pass
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        requested = _providers(device)
        self.session = ort.InferenceSession(str(self.path), opts, providers=requested)
        self.provider = self.session.get_providers()[0]
        if requested[0] != self.provider:
            # ORT silently falls back to CPU when CUDA/cuDNN DLLs are missing - make it visible
            msg = f"{self.path.name}: wanted {requested[0]} but running on {self.provider} (missing CUDA/cuDNN libs?)"
            if device == "cuda":
                raise RuntimeError(msg)
            log.warning(msg)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        meta = self.session.get_modelmeta().custom_metadata_map
        # Ultralytics writes the class names into the ONNX metadata as a Python dict literal
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.conf, self.iou = conf, iou

    def warmup(self, runs: int = 2) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        for _ in range(runs):
            self(dummy)

    def __call__(self, img_bgr: np.ndarray, conf: float | None = None, iou: float | None = None
                 ) -> tuple[list[Detection], dict[str, float]]:
        """Run detection. Returns detections and a timing breakdown in milliseconds."""
        t0 = time.perf_counter()
        x, info = to_input_tensor(img_bgr, self.imgsz)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x.astype(self.input_dtype)})[0].astype(np.float32)
        t2 = time.perf_counter()
        boxes, scores, cls = decode_yolo(out, info, conf if conf is not None else self.conf,
                                         iou if iou is not None else self.iou)
        t3 = time.perf_counter()
        dets = [Detection([round(float(v), 1) for v in b], round(float(s), 4), int(c),
                          self.names.get(int(c), str(int(c))))
                for b, s, c in zip(boxes, scores, cls)]
        timing = {"preprocess_ms": (t1 - t0) * 1e3, "inference_ms": (t2 - t1) * 1e3,
                  "postprocess_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}
        return dets, {k: round(v, 2) for k, v in timing.items()}
