"""Step 3: fine-tune a COCO-pretrained YOLO model on VisDrone (transfer learning).

Why fine-tune instead of training from scratch?  The backbone already learned edges,
textures and object parts from 118k COCO images. We keep all of that and adapt it to the
new domain (top-down drone/CCTV views, tiny objects, new classes like "van" and
"tricycle"). The new 10-class head is initialised fresh; everything else starts from COCO.

Everything comes from configs/experiments.yaml -> one command reproduces a run:
    python training/train.py --exp yolo11n-visdrone
Outputs (Ultralytics): training/runs/<exp>/  results.csv (per-epoch losses + mAP = our
experiment log), weights/best.pt, confusion matrix, PR curves, labelled val batches.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from common import REPORTS, ROOT, RUNS, pretrained, seed_everything


def load_experiment(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "training" / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    if name not in cfg["experiments"]:
        raise SystemExit(f"unknown experiment {name}; choose from {list(cfg['experiments'])}")
    params = {**cfg["common"], **cfg["experiments"][name]}
    params["data"] = str(ROOT / params["data"])
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, help="override (e.g. 1 for a smoke test)")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction of the data (smoke test)")
    args = ap.parse_args()

    from ultralytics import YOLO

    params = load_experiment(args.exp)
    if args.epochs:
        params["epochs"] = args.epochs
    seed_everything(params["seed"])
    model = YOLO(pretrained(params.pop("model")))
    t0 = time.time()
    model.train(**params, project=str(RUNS), name=args.exp, exist_ok=True, fraction=args.fraction,
                device=0 if _cuda() else "cpu", plots=True, verbose=False)
    run_dir = RUNS / args.exp
    best = run_dir / "weights" / "best.pt"
    params["data"] = Path(params["data"]).relative_to(ROOT).as_posix()      # no machine-specific paths in reports
    log = {"experiment": args.exp, "params": params, "minutes": round((time.time() - t0) / 60, 1),
           "best_weights": str(best.relative_to(ROOT))}
    (REPORTS / f"train_{args.exp}.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, default=str))


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
