"""Paths and helpers shared by the training scripts."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]             # projects/vision-analytics
DATASETS = ROOT / "datasets"
REPORTS = ROOT / "training" / "reports"
RUNS = ROOT / "training" / "runs"
MODELS = ROOT / "models"
WEIGHTS = ROOT / "training" / "weights"                 # downloaded pretrained .pt files
for p in (DATASETS, REPORTS, RUNS, MODELS, WEIGHTS):
    p.mkdir(parents=True, exist_ok=True)

VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
                  "awning-tricycle", "bus", "motor"]

# VisDrone class -> COCO class id, used to measure how well a COCO-only model does on drone
# footage *before* fine-tuning (the "domain gap" baseline). Classes with no COCO
# equivalent (van, tricycle, awning-tricycle) are left out of that comparison.
VISDRONE_TO_COCO = {0: 0, 1: 0, 2: 1, 3: 2, 5: 7, 8: 5, 9: 3}   # person, bicycle, car, truck, bus, motorcycle


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def update_metrics(section: str, key: str, value: dict) -> None:
    """Merge results into training/reports/metrics.json (read by the API's /api/metrics)."""
    path = REPORTS / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault(section, {})[key] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def pretrained(name: str) -> str:
    """Path of a pretrained checkpoint inside training/weights (downloaded on first use)."""
    from ultralytics import YOLO
    target = WEIGHTS / name
    if not target.exists():
        YOLO(name)                                      # Ultralytics downloads into the CWD
        Path(name).replace(target)
    return str(target)
