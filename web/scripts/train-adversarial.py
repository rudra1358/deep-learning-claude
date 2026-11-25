"""
train-adversarial.py -- trains the three small digit classifiers used by the "Adversarial examples:
FGSM and PGD" simulator (components/sims/AdversarialSim.tsx) and exports them to
    web/public/models/adversarial.json

DATA
  scikit-learn's bundled 8x8 handwritten digits (sklearn.datasets.load_digits: 1797 images, grey
  levels 0..16). No download. Pixels are divided by 16 so they lie in [0, 1]; every attack keeps the
  perturbed image inside [0, 1]. Split: train_test_split(test_size=0.2, stratify=labels,
  random_state=0) -> 1437 training and 360 test images (the same split as the other digit sims).

MODELS
  * linear    softmax regression 64 -> 10 (650 parameters), cross-entropy + small L2 penalty.
  * mlp       64 -> 32 (ReLU) -> 10 (2,410 parameters), standard training.
  * mlp_adv   the same architecture trained adversarially (Madry et al., 2018): every mini-batch is
              replaced by its PGD adversarial version (L-infinity ball of radius 0.1, 7 steps of size
              0.025, uniform random start) before the gradient step.
  Optimiser: Adam lr 1e-2 (linear, full batch, 300 steps) / Adam lr 3e-3, batch 64 (MLPs, 60 epochs).

EXPORT FORMAT (JSON; reader: lib/sim/adversarial.ts)
  Weights are rounded to 6 decimals and stored in "x @ W" orientation (W[in][out]). Everything that is
  exported as a reference (test accuracies, accuracy-vs-epsilon curves, verification vectors) is
  recomputed in float64 NumPy with exactly these rounded weights, so the browser (float64) must agree
  to ~1e-9. Attacks (FGSM; PGD without random start, so they are deterministic):
      FGSM:  x' = clip_[0,1]( x + eps * sign(grad_x L) )
      PGD:   x_{t+1} = clip_[0,1]( clip_[x-eps, x+eps]( x_t + alpha * sign(grad_x L(x_t)) ) ),  x_0 = x
  Test images are stored as integer grey levels 0..16.

HOW TO RUN (from the repository root or from web/)
    python web/scripts/train-adversarial.py
  Requires PyTorch (CPU), NumPy and scikit-learn. Deterministic: fixed seeds, one thread,
  torch.use_deterministic_algorithms(True). Runtime: about 15 s on a laptop CPU (training ~5 s).

RESULT OF THE COMMITTED RUN (360 test digits; PGD = 20 steps of size 2.5*eps/20, no random start)
  linear : 650 params,  clean 95.3%;  eps 0.1 -> FGSM 65.0%, PGD 64.4%;  eps 0.2 -> FGSM 12.2%, PGD 8.9%
  mlp    : 2410 params, clean 97.2%;  eps 0.1 -> FGSM 36.9%, PGD 30.6%;  eps 0.2 -> FGSM 1.7%,  PGD 0.3%
  mlp_adv: 2410 params, clean 96.7%;  eps 0.1 -> FGSM 78.6%, PGD 76.7%;  eps 0.2 -> FGSM 31.1%, PGD 21.9%
  adversarial.json ~129 KB. lib/sim/adversarial.ts reproduces the verification vectors to ~1e-15 and
  the accuracy curves exactly; its analytic input gradient matches central finite differences (~4e-10).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "models" / "adversarial.json"

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
t0 = time.time()

digits = load_digits()
X_all = digits.data.astype(np.float32)
y_all = digits.target
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=0)
Xtr = torch.tensor(Xtr_raw / 16.0)
Xte = torch.tensor(Xte_raw / 16.0)
Ytr = torch.tensor(ytr)

ADV_EPS, ADV_ALPHA, ADV_STEPS = 0.1, 0.025, 7


def pgd_torch(model, x, y, eps, alpha, steps, gen):
    """PGD with a uniform random start, used only during adversarial training."""
    x_adv = (x + (torch.rand(x.shape, generator=gen) * 2 - 1) * eps).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (g,) = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + alpha * g.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def train_linear():
    torch.manual_seed(0)
    model = nn.Linear(64, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss = F.cross_entropy(model(Xtr), Ytr) + 1e-3 * model.weight.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [model]


def train_mlp(adversarial):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        for idx in torch.randperm(len(Xtr), generator=gen).split(64):
            x, y = Xtr[idx], Ytr[idx]
            if adversarial:
                x = pgd_torch(model, x, y, ADV_EPS, ADV_ALPHA, ADV_STEPS, gen)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return [model[0], model[2]]


def export_layers(layers):
    out = []
    for lin in layers:
        W = np.round(lin.weight.detach().numpy().T.astype(np.float64), 6)   # [in][out]
        b = np.round(lin.bias.detach().numpy().astype(np.float64), 6)
        out.append((W, b))
    return out


# ------------------------------------------------------------- float64 NumPy reference
def forward(layers, x):
    h = x
    pre = []
    for i, (W, b) in enumerate(layers):
        z = h @ W + b
        pre.append(z)
        h = np.maximum(z, 0) if i < len(layers) - 1 else z
    return h, pre


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def loss_grad(layers, x, y):
    """Cross-entropy loss and its gradient with respect to the input (x: [N, 64], y: [N])."""
    logits, pre = forward(layers, x)
    p = softmax(logits)
    n = len(x)
    loss = -np.log(p[np.arange(n), y])
    g = p.copy()
    g[np.arange(n), y] -= 1                     # dL/dlogits = p - onehot(y)
    for i in range(len(layers) - 1, -1, -1):
        W, _ = layers[i]
        g = g @ W.T                               # dL/d(input of layer i)
        if i > 0:
            g = g * (pre[i - 1] > 0)              # ReLU derivative of the previous layer
    return loss, g, p


def fgsm(layers, x, y, eps):
    _, g, _ = loss_grad(layers, x, y)
    return np.clip(x + eps * np.sign(g), 0, 1)


def pgd(layers, x, y, eps, alpha, steps):
    xa = x.copy()
    for _ in range(steps):
        _, g, _ = loss_grad(layers, xa, y)
        xa = xa + alpha * np.sign(g)
        xa = np.clip(np.minimum(np.maximum(xa, x - eps), x + eps), 0, 1)
    return xa


def accuracy(layers, x, y):
    return float(np.mean(forward(layers, x)[0].argmax(axis=1) == y))


models = {
    "linear": export_layers(train_linear()),
    "mlp": export_layers(train_mlp(False)),
    "mlp_adv": export_layers(train_mlp(True)),
}
print(f"trained in {time.time() - t0:.1f} s")

Xte64 = Xte_raw.astype(np.float64) / 16.0
EPS_GRID = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]
PGD_STEPS, PGD_REL_ALPHA = 20, 2.5          # curve attacks: 20 steps of size 2.5 * eps / 20
out_models = {}
for name, layers in models.items():
    curves = {"fgsm": [], "pgd": []}
    for eps in EPS_GRID:
        curves["fgsm"].append(round(accuracy(layers, fgsm(layers, Xte64, yte, eps), yte), 6))
        curves["pgd"].append(round(accuracy(layers, pgd(layers, Xte64, yte, eps, PGD_REL_ALPHA * eps / PGD_STEPS, PGD_STEPS), yte), 6))
    n_params = int(sum(W.size + b.size for W, b in layers))
    out_models[name] = {
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers],
        "n_params": n_params,
        "test_acc": round(accuracy(layers, Xte64, yte), 6),
        "train_acc": round(accuracy(layers, Xtr_raw.astype(np.float64) / 16.0, ytr), 6),
        "curves": curves,
    }
    print(f"{name:8s} params {n_params:5d}  train acc {out_models[name]['train_acc']:.4f}  test acc {out_models[name]['test_acc']:.4f}")
    print(f"         FGSM acc vs eps {curves['fgsm']}")
    print(f"         PGD  acc vs eps {curves['pgd']}")

# verification vectors: loss, gradient, FGSM and PGD results for a few test images
verify = []
for name, layers in models.items():
    for i in (0, 1, 7):
        x = Xte64[i:i + 1]
        y = yte[i:i + 1]
        loss, g, p = loss_grad(layers, x, y)
        xf = fgsm(layers, x, y, 0.1)
        xp = pgd(layers, x, y, 0.1, 0.02, 10)
        verify.append({
            "model": name, "index": i, "label": int(y[0]),
            "loss": float(loss[0]), "probs": p[0].tolist(), "grad": g[0].tolist(),
            "fgsm_eps": 0.1, "fgsm": xf[0].tolist(), "fgsm_probs": softmax(forward(layers, xf)[0])[0].tolist(),
            "pgd_eps": 0.1, "pgd_alpha": 0.02, "pgd_steps": 10, "pgd": xp[0].tolist(),
            "pgd_probs": softmax(forward(layers, xp)[0])[0].tolist(),
        })

doc = {
    "format": "adversarial-v1",
    "description": "Three 8x8-digit classifiers (softmax regression, MLP, PGD-adversarially-trained MLP) for the FGSM/PGD simulator.",
    "data": {"name": "sklearn load_digits / 16", "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
             "split": "train_test_split(test_size=0.2, stratify=y, random_state=0)"},
    "adv_training": {"eps": ADV_EPS, "alpha": ADV_ALPHA, "steps": ADV_STEPS, "random_start": True},
    "curve_attack": {"eps": EPS_GRID, "pgd_steps": PGD_STEPS, "pgd_alpha": "2.5 * eps / 20", "random_start": False},
    "models": out_models,
    "test": {"images": Xte_raw.astype(int).tolist(), "labels": [int(v) for v in yte]},
    "verify": verify,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(doc, separators=(",", ":")))
print(f"wrote {OUT.name}: {OUT.stat().st_size / 1024:.0f} KB in {time.time() - t0:.1f} s")
