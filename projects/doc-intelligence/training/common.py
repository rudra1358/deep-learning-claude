"""Shared paths and a small, explicit training loop used by both fine-tuning scripts."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]                 # projects/doc-intelligence
MODELS = ROOT / "models"                                   # our fine-tuned models (served by the API)
REPORTS = ROOT / "training" / "reports"
DATA = ROOT / "data"
for p in (MODELS, REPORTS, DATA):
    p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_metrics(section: str, key: str, value) -> None:
    path = REPORTS / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault(section, {})[key] = value
    # scikit-learn / seqeval return NumPy numbers, which the json module cannot write -> convert
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def train_loop(model, train_loader, epochs: int, lr: float, evaluate, log_name: str,
               warmup_frac: float = 0.1, weight_decay: float = 0.01) -> list[dict]:
    """The standard recipe for fine-tuning a pretrained Transformer:

    * AdamW optimiser, small learning rate (2e-5 .. 5e-5) - big steps would destroy what the
      pretrained weights already know ("catastrophic forgetting")
    * linear warm-up for the first 10 % of steps, then linear decay to 0 - avoids huge, noisy
      updates at the start while the new randomly-initialised head is still wrong
    * mixed precision (fp16 autocast + GradScaler) on GPU: ~2x faster, half the memory
    * gradient clipping at 1.0 - a safety net against rare exploding updates
    * evaluate after every epoch on held-out data; keep the history for the report
    """
    device = next(model.parameters()).device
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    total = epochs * len(train_loader)
    warm = int(warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss                 # HF models compute cross-entropy when labels are given
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"[{log_name}] epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}", flush=True)
        metrics = evaluate(model)
        rec = {"epoch": epoch, "train_loss": round(running / len(train_loader), 4),
               "minutes": round((time.time() - t0) / 60, 1), **metrics}
        history.append(rec)
        print(f"[{log_name}] {rec}", flush=True)
        if not math.isfinite(rec["train_loss"]):
            raise RuntimeError("loss became NaN/inf")
    return history
