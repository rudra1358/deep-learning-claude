"""End-to-end throughput of the serving pipeline on a real video (decode -> letterbox -> ONNX ->
NMS -> ByteTrack -> analytics), CPU vs GPU. Writes training/reports/end_to_end.json.

    python scripts/fetch_demo_assets.py && python scripts/benchmark_pipeline.py --model yolo11n-coco
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import Pipeline  # noqa: E402
from inference.detector import OnnxDetector  # noqa: E402


def run(model: Path, video: Path, device: str, frames: int) -> dict:
    det = OnnxDetector(model, device=device)
    det.warmup(5)
    cap = cv2.VideoCapture(str(video))
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    pipe = Pipeline(det, w, h, fps, line=[0, 0.55, 1, 0.55], conf=0.3)
    n, t0 = 0, time.perf_counter()
    while n < frames:
        ok, f = cap.read()
        if not ok:
            break
        pipe.process(f)
        n += 1
    dt = time.perf_counter() - t0
    s = pipe.summary()
    return {"provider": det.provider, "frames": n, "resolution": f"{w}x{h}", "end_to_end_fps": round(n / dt, 1),
            "avg_detector_ms": pipe.avg_latency, "unique_objects": s["unique_objects"], "line": s["line"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-coco")
    ap.add_argument("--video", default=str(ROOT / "demo" / "pedestrians.avi"))
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    res = {"model": a.model, "video": Path(a.video).name}
    for dev in ("cuda", "cpu"):
        try:
            res[dev] = run(ROOT / "models" / f"{a.model}.onnx", Path(a.video), dev, a.frames)
        except RuntimeError as e:
            res[dev] = {"error": str(e)}
        print(dev, res[dev])
    (ROOT / "training" / "reports" / "end_to_end.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
