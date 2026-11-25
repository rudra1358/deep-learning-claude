"""Steps 2 and 4: evaluation (baselines, fine-tuned models, error analysis).

Metric: mAP (mean Average Precision).
  * For one class, sort all predictions by confidence; walking down the list gives a
    precision-recall curve. AP = area under that curve.
  * A prediction is a true positive if IoU with an unmatched ground-truth box >= threshold.
  * mAP@0.5      = AP at IoU 0.5, averaged over classes (lenient on localisation)
  * mAP@0.5:0.95 = averaged over IoU thresholds 0.50, 0.55 ... 0.95 (COCO's main metric,
                   rewards tight boxes)

Subcommands
  coco-baseline   pretrained YOLO11n / YOLO11s and torchvision Faster R-CNN on the
                  500-image COCO val subset (same images, same pycocotools evaluator)
  domain-gap      COCO-pretrained YOLO on VisDrone val, on the classes both datasets share
  finetuned       fine-tuned VisDrone models: mAP, per-class AP, confusion matrix
  sizes           AP for small / medium / large objects on VisDrone val (pycocotools)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from common import (
    DATASETS,
    ROOT,
    RUNS,
    VISDRONE_NAMES,
    VISDRONE_TO_COCO,
    pretrained,
    update_metrics,
)

CFG = ROOT / "training" / "configs"


def _device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- pycocotools helpers
def coco_eval(gt_json: dict, dets: list[dict], cat_ids: list[int] | None = None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = gt_json
    gt.createIndex()
    if not dets:
        return {"mAP50_95": 0.0, "mAP50": 0.0}
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    if cat_ids is not None:
        ev.params.catIds = cat_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP50_95": round(float(s[0]), 4), "mAP50": round(float(s[1]), 4),
            "AP_small": round(float(s[3]), 4), "AP_medium": round(float(s[4]), 4),
            "AP_large": round(float(s[5]), 4)}


def _latency(fn, imgs, warmup: int = 3) -> float:
    for im in imgs[:warmup]:
        fn(im)
    t0 = time.perf_counter()
    for im in imgs:
        fn(im)
    return round((time.perf_counter() - t0) / len(imgs) * 1e3, 1)


# --------------------------------------------------------------------------- 1. COCO baseline
def coco_baseline(max_images: int) -> None:
    import torch
    import torchvision
    from ultralytics import YOLO

    root = DATASETS / "coco-val500"
    gt = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    cats = sorted(c["id"] for c in gt["categories"])          # contiguous idx -> COCO id
    images = gt["images"][:max_images]
    gt["images"] = images
    keep = {im["id"] for im in images}
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    paths = [str(root / "images" / im["file_name"]) for im in images]

    for name in ("yolo11n", "yolo11s"):
        model = YOLO(pretrained(f"{name}.pt"))
        dets = []
        for im, p in zip(images, paths):
            r = model.predict(p, conf=0.001, iou=0.7, max_det=300, verbose=False, device=_device())[0]
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append({"image_id": im["id"], "category_id": cats[int(c)],
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
        res = coco_eval(gt, dets)
        res["latency_ms"] = _latency(lambda p: model.predict(p, verbose=False, device=_device()), paths[:50])
        res["params_M"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
        update_metrics("coco_baseline", name, res)
        print(name, res)

    # Two-stage reference: Faster R-CNN (ResNet-50 FPN v2), torchvision COCO weights.
    # torchvision labels use the original COCO category ids directly.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001)
    frcnn.eval().to(dev)
    tf = weights.transforms()

    @torch.inference_mode()
    def run(p):
        x = tf(Image.open(p).convert("RGB")).to(dev)
        return frcnn([x])[0]

    dets = []
    for im, p in zip(images, paths):
        o = run(p)
        for (x1, y1, x2, y2), s, c in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
            dets.append({"image_id": im["id"], "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": s})
    res = coco_eval(gt, dets)
    res["latency_ms"] = _latency(run, paths[:30])
    res["params_M"] = round(sum(p.numel() for p in frcnn.parameters()) / 1e6, 2)
    update_metrics("coco_baseline", "faster_rcnn_r50_fpn_v2", res)
    print("faster_rcnn", res)


# --------------------------------------------------------------------------- VisDrone helpers
def visdrone_val() -> tuple[dict, list[Path]]:
    """VisDrone val split as a COCO-style ground-truth dict (built from the YOLO labels)."""
    import yaml
    cfg = yaml.safe_load((CFG / "visdrone.yaml").read_text(encoding="utf-8"))
    img_dir = Path(cfg["path"]) / cfg["val"]
    lab_dir = Path(str(img_dir).replace("images", "labels"))
    images, anns, paths = [], [], sorted(img_dir.glob("*.jpg"))
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": i, "file_name": p.name, "width": w, "height": h})
        lf = lab_dir / (p.stem + ".txt")
        for ln in (lf.read_text(encoding="utf-8").splitlines() if lf.exists() else []):
            c, cx, cy, bw, bh = map(float, ln.split())
            bw, bh = bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c), "iscrowd": 0,
                         "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh], "area": bw * bh})
    cats = [{"id": i, "name": n} for i, n in enumerate(VISDRONE_NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}, paths


def _predict_dets(model, paths: list[Path], cls_map=None) -> list[dict]:
    dets = []
    for i, p in enumerate(paths):
        r = model.predict(str(p), conf=0.001, iou=0.7, max_det=300, verbose=False, device=_device(), imgsz=640)[0]
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            c = int(c)
            if cls_map is not None:
                if c not in cls_map:
                    continue
                c = cls_map[c]
            dets.append({"image_id": i, "category_id": c, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return dets


# --------------------------------------------------------------------------- 2. domain gap
def domain_gap() -> None:
    """How good is a COCO model on drone footage *before* fine-tuning?

    Evaluated on the 5 classes with a COCO equivalent (pedestrian+people both -> person are
    merged, so ground truth is remapped the same way)."""
    from ultralytics import YOLO
    gt, paths = visdrone_val()
    # merge VisDrone classes into COCO concepts on the ground-truth side
    vd_to_shared = {0: 0, 1: 0, 2: 1, 3: 2, 5: 7, 8: 5, 9: 3}        # -> COCO class index
    gt["annotations"] = [{**a, "category_id": vd_to_shared[a["category_id"]]}
                         for a in gt["annotations"] if a["category_id"] in vd_to_shared]
    shared = sorted(set(vd_to_shared.values()))
    gt["categories"] = [{"id": c, "name": str(c)} for c in shared]
    for name in ("yolo11n", "yolo11s"):
        model = YOLO(pretrained(f"{name}.pt"))
        dets = _predict_dets(model, paths, cls_map={c: c for c in shared})
        res = coco_eval(gt, dets, cat_ids=shared)
        update_metrics("domain_gap", f"{name}-coco-on-visdrone", res)
        print(name, res)
    assert VISDRONE_TO_COCO == vd_to_shared


# --------------------------------------------------------------------------- 3. fine-tuned
def finetuned() -> None:
    from ultralytics import YOLO
    gt, paths = visdrone_val()
    for exp in ("yolo11n-visdrone", "yolo11s-visdrone"):
        best = RUNS / exp / "weights" / "best.pt"
        if not best.exists():
            print("skip", exp, "(not trained)")
            continue
        model = YOLO(str(best))
        # Ultralytics' own validator: mAP + per-class AP + confusion matrix plot
        m = model.val(data=str(CFG / "visdrone.yaml"), split="val", imgsz=640, batch=8, device=_device(),
                      plots=True, project=str(RUNS), name=f"{exp}-val", exist_ok=True, verbose=False)
        per_class = {VISDRONE_NAMES[i]: round(float(ap), 4) for i, ap in enumerate(m.box.maps)}
        # pycocotools on the same images: adds small / medium / large breakdown
        sizes = coco_eval(gt, _predict_dets(model, paths))
        shared = coco_eval({**gt, "annotations": [{**a, "category_id": VISDRONE_TO_COCO[a["category_id"]]}
                                                   for a in gt["annotations"] if a["category_id"] in VISDRONE_TO_COCO],
                            "categories": [{"id": c, "name": str(c)} for c in sorted(set(VISDRONE_TO_COCO.values()))]},
                           [{**d, "category_id": VISDRONE_TO_COCO[d["category_id"]]}
                            for d in _predict_dets(model, paths) if d["category_id"] in VISDRONE_TO_COCO],
                           cat_ids=sorted(set(VISDRONE_TO_COCO.values())))
        res = {"mAP50_95": round(float(m.box.map), 4), "mAP50": round(float(m.box.map50), 4),
               "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
               "per_class_AP50_95": per_class, "by_size": sizes, "shared_classes_for_domain_gap": shared,
               "confusion_matrix_png": (RUNS / f"{exp}-val" / "confusion_matrix_normalized.png").relative_to(ROOT).as_posix()}
        update_metrics("finetuned", exp, res)
        print(exp, json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["coco-baseline", "domain-gap", "finetuned"])
    ap.add_argument("--max-images", type=int, default=500)
    a = ap.parse_args()
    {"coco-baseline": lambda: coco_baseline(a.max_images), "domain-gap": domain_gap, "finetuned": finetuned}[a.what]()
