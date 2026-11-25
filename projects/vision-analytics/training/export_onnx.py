"""Step 5: export to ONNX, verify the export is correct, and measure speed-ups.

For each model:
  1. export .pt -> .onnx (opset 17, static 1x3x640x640 input, graph simplified)
  2. PARITY CHECK - the most important step (see parity() for details):
       raw: same tensor through PyTorch and ONNX Runtime, compare raw outputs
       end-to-end: Ultralytics predict() vs OUR serving code (letterbox -> ONNX Runtime ->
       our NMS), boxes matched by IoU. A bug in our pre/post-processing (wrong padding,
       RGB/BGR swap, wrong scaling) would show up as a low match rate / low IoU.
  3. latency of the serving pipeline: CPU FP32, CPU INT8 (dynamic quantisation),
     GPU FP32 and GPU FP16.
Results go to training/reports/metrics.json -> "models" (shown in the dashboard).

    python training/export_onnx.py                 # all available models (+ parity, latency)
    python training/export_onnx.py --export-only   # just create models/*.onnx (fresh clone, no datasets)
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import MODELS, ROOT, RUNS, pretrained, update_metrics

sys.path.insert(0, str(ROOT / "ml-service"))
from inference.detector import OnnxDetector  # noqa: E402
from inference.postprocess import box_iou  # noqa: E402

SOURCES = {
    "yolo11n-coco": lambda: pretrained("yolo11n.pt"),
    "yolo11s-coco": lambda: pretrained("yolo11s.pt"),
    "yolo11n-visdrone": lambda: str(RUNS / "yolo11n-visdrone" / "weights" / "best.pt"),
    "yolo11s-visdrone": lambda: str(RUNS / "yolo11s-visdrone" / "weights" / "best.pt"),
}


def sample_images(name: str, n: int = 30) -> list[Path]:
    root = ROOT / "datasets" / ("VisDrone/images/val" if "visdrone" in name else "coco-val500/images")
    return sorted(root.glob("*.jpg"))[:n]


def export(name: str, pt: str, half: bool = False) -> Path:
    from ultralytics import YOLO
    out = YOLO(pt).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, half=half,
                          device=0 if half else "cpu", verbose=False)
    dst = MODELS / f"{name}{'-fp16' if half else ''}.onnx"
    shutil.move(out, dst)
    return dst


def parity(pt: str, onnx_path: Path, images: list[Path]) -> dict:
    """Two checks.

    raw:  the SAME input tensor through PyTorch and through ONNX Runtime -> compare the raw
          (1, 4+C, 8400) outputs. Proves the exported graph computes the same function.
    end-to-end: Ultralytics' full predict() vs OUR serving code (letterbox, decode, NMS, rescale).
          Ultralytics' predict letterboxes to a minimal *rectangle* (auto=True) while a static
          ONNX model needs a 640x640 square, so we hand Ultralytics our square letterboxed image
          (its own letterbox is then a no-op) and map its boxes back with our letterbox info.
          Both sides now see identical pixels, so any disagreement = a bug in our post-processing.
    """
    import torch
    from ultralytics import YOLO

    from inference.postprocess import scale_boxes
    from inference.preprocess import letterbox, to_input_tensor

    ref = YOLO(pt)
    ours = OnnxDetector(onnx_path, device="cpu", conf=0.25, iou=0.45)
    net = ref.model.float().eval()
    raw_box_diff, raw_score_diff = 0.0, 0.0
    matched, total_ref, total_ours, ious, sdiff = 0, 0, 0, [], []
    for p in images:
        img = cv2.imread(str(p))
        x, info = to_input_tensor(img, 640)
        with torch.no_grad():
            t_out = net(torch.from_numpy(x))
        t_out = (t_out[0] if isinstance(t_out, (list, tuple)) else t_out).numpy()
        o_out = ours.session.run(None, {ours.input_name: x})[0]
        raw_box_diff = max(raw_box_diff, float(np.abs(t_out[0, :4] - o_out[0, :4]).max()))
        raw_score_diff = max(raw_score_diff, float(np.abs(t_out[0, 4:] - o_out[0, 4:]).max()))

        boxed, _ = letterbox(img, 640)
        r = ref.predict(boxed, conf=0.25, iou=0.45, imgsz=640, verbose=False, device="cpu")[0]
        rb = scale_boxes(r.boxes.xyxy.numpy(), info)
        rs, rc = r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int)
        dets, _ = ours(img)
        ob = np.array([d.box for d in dets]).reshape(-1, 4)
        os_ = np.array([d.score for d in dets])
        oc = np.array([d.cls for d in dets])
        total_ref += len(rb); total_ours += len(ob)
        if len(rb) and len(ob):
            iou = box_iou(rb, ob)
            iou[rc[:, None] != oc[None, :]] = 0
            for i in range(len(rb)):
                j = int(iou[i].argmax())
                if iou[i, j] > 0.9:
                    matched += 1
                    ious.append(iou[i, j]); sdiff.append(abs(rs[i] - os_[j]))
    return {"raw_max_abs_diff_box_px": round(raw_box_diff, 5), "raw_max_abs_diff_score": round(raw_score_diff, 6),
            "reference_boxes": total_ref, "onnx_boxes": total_ours,
            "match_rate": round(matched / max(total_ref, 1), 4),
            "mean_iou_of_matches": round(float(np.mean(ious)), 4) if ious else None,
            "max_score_diff": round(float(np.max(sdiff)), 5) if sdiff else None}


def latency(onnx_path: Path, device: str, images: list[Path], runs: int = 40) -> float | None:
    try:
        det = OnnxDetector(onnx_path, device=device)
    except Exception as e:  # noqa: BLE001
        print("  skip", device, e)
        return None
    if device == "cuda" and not det.provider.startswith("CUDA"):
        return None
    imgs = [cv2.imread(str(p)) for p in images[:10]]
    det.warmup(5)
    t0 = time.perf_counter()
    for i in range(runs):
        det(imgs[i % len(imgs)])
    return round((time.perf_counter() - t0) / runs * 1e3, 2)


def quantize_int8(src: Path) -> Path:
    """Dynamic INT8 quantisation: weights stored as int8, activations quantised on the fly.
    No calibration data needed. Mainly a CPU optimisation."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = src.with_name(src.stem + "-int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return dst


def main() -> None:
    for name, src in SOURCES.items():
        pt = src()
        if not Path(pt).exists():
            print("skip", name, "(weights missing - train first)")
            continue
        print("==", name)
        fp32 = export(name, pt)
        imgs = sample_images(name)
        if not imgs or "--export-only" in sys.argv:
            # fresh clone without the evaluation datasets: still produce the model the API serves,
            # just skip the parity / latency measurements (they need sample images)
            print("  exported", fp32.name, "(no sample images - parity/latency skipped; run prepare_data.py for them)")
            continue
        res = {"onnx_mb": round(fp32.stat().st_size / 2**20, 1), "parity": parity(pt, fp32, imgs)}
        print("  parity", res["parity"])
        res["latency_ms"] = {"cpu_fp32": latency(fp32, "cpu", imgs), "cuda_fp32": latency(fp32, "cuda", imgs)}
        int8 = quantize_int8(fp32)
        res["latency_ms"]["cpu_int8"] = latency(int8, "cpu", imgs)
        res["int8_parity"] = parity(pt, int8, imgs[:10])
        int8.unlink()          # measured only; INT8 dynamic quant of convs is slow on many CPUs, not served
        try:
            fp16 = export(name, pt, half=True)
            res["latency_ms"]["cuda_fp16"] = latency(fp16, "cuda", imgs)
            fp16.unlink()
        except Exception as e:  # noqa: BLE001
            print("  fp16 export skipped:", e)
        update_metrics("models", name, res)
        print(" ", res)


if __name__ == "__main__":
    main()
