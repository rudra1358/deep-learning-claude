"""
train-diffusion.py -- trains the two small DDPM noise-prediction networks used by the
"Diffusion Model Explorer (trained models)" simulator (components/sims/DiffusionSim.tsx)
and exports them to web/public/models/diffusion.json.

WHAT IT TRAINS
  Both models are epsilon-prediction networks eps_theta(x_t, t[, c]) trained with the simplified
  DDPM loss  E || eps - eps_theta(sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, t) ||^2   (Ho et al. 2020,
  Algorithm 1) on a shared noise schedule:
    * T = 100 steps, LINEAR beta schedule beta_1 = 1e-3 ... beta_T = 0.2 (Ho et al.'s 1e-4 ... 0.02
      for T = 1000, rescaled by 1000/T as in Nichol & Dhariwal 2021), giving abar_T ~ 2e-5.
    * time embedding: 32-d sinusoidal [sin(t f_k), cos(t f_k)], f_k = 200^(-k/16), k = 0..15,
      fed with the 0-based step index t = 0..T-1.
    * network: MLP, input = concat(x_t, emb(t)[, one-hot(class)]), SiLU hidden layers, linear output.

  (a) "moons": unconditional, 2-D. Data = sklearn make_moons(20000, noise=0.05, random_state=0),
      standardised per axis. MLP 34 -> 64 -> 64 -> 64 -> 2.
  (b) "digits": class-conditional with classifier-free guidance, 64-D. Data = sklearn load_digits
      (1,797 8x8 greyscale digits, pixel values 0..16 rescaled to [-1, 1]); the class label is replaced
      by the "null" class 10 with probability 0.1 so the same network also learns eps(x_t, null)
      (Ho & Salimans 2022). MLP (64 + 32 + 11) -> 256 -> 256 -> 64.
  Adam, cosine learning-rate decay, an exponential moving average (decay 0.995) of the weights is
  exported (standard practice for diffusion models; it gives visibly cleaner samples).

EXPORT FORMAT (see lib/sim/diffusion.ts for the reader)
  Weights are stored as IEEE half floats (float16), base64-encoded, row-major in "x @ W"
  orientation (shape [in, out]). After rounding to float16 the weights are loaded back into the
  PyTorch model and ALL reported numbers (metrics, verification vectors) are computed with exactly
  the weights the browser receives, in float64 (the browser computes in float64 too), so the
  browser can be checked against PyTorch to ~1e-12.
  Verification vectors: eps outputs for fixed (x_t, t, c) and complete deterministic DDIM
  trajectories (with and without guidance) from fixed x_T.

METRICS (all printed, all exported)
  * moons: energy distance between 4,000 generated points and 4,000 held-out real points,
    for DDPM (100 steps) and DDIM with 5/10/20/50/100 steps; reference values: a second fresh real
    sample (the noise floor of the estimate) and N(0, I) noise.
  * digits: a logistic-regression classifier trained on the REAL digits labels the generated images;
    "accuracy" = fraction classified as the class they were conditioned on, for several guidance
    scales w, plus the mean distance to the nearest training image (memorisation check).

HOW TO RUN (from the repository root or from web/)
    python web/scripts/train-diffusion.py
  Requires Python 3.10+, PyTorch (CPU), NumPy, scikit-learn. Deterministic: fixed seeds and
  torch.set_num_threads(4) (two runs on the same machine gave identical losses). Runtime: about
  5 minutes on a busy laptop CPU (8,000 + 15,000 training steps plus evaluation).

RESULT OF THE COMMITTED RUN
  moons : 10,690 parameters, final mean training loss 0.224. Energy distance (unbiased estimator,
          4,000 vs 4,000 points) to held-out real data: real-vs-real -0.0008 (noise floor),
          N(0, I) 0.0472, DDPM-100 0.0002, DDIM 5/10/20/50/100 steps: 0.0930 / 0.0109 / 0.0024 /
          0.0011 / 0.0002.
  digits: 109,888 parameters, final mean training loss 0.107. Logistic-regression judge (98.5%
          on the real digits) labels generated images as the requested class: DDPM w=0 97.3%,
          w=1 100%, w=3 99.7%; DDIM w=1 with 10/25/50 steps 96.7% / 98.3% / 99.0%. Mean distance
          to the nearest training image 2.21 (w=1) vs 2.05 between a real image and its nearest
          other real image: samples are close to, but mostly not copies of, training digits.
  JSON: 348 KB (float16 weights). The browser engine (lib/sim/diffusion.ts) reproduces the
  stored float64 PyTorch eps outputs to 1.4e-11 and full DDIM trajectories to 7.5e-12.
"""

import base64
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits, make_moons
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "models" / "diffusion.json"

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

# ----------------------------------------------------------------- schedule
T = 100
BETA_START, BETA_END = 1e-3, 0.2
betas = torch.linspace(BETA_START, BETA_END, T, dtype=torch.float64)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, 0)
EMB_DIM, MAX_PERIOD = 32, 200.0
NULL = 10  # class index used for the unconditional ("null") token of the digits model


def time_emb(t):
    half = EMB_DIM // 2
    f = torch.exp(-math.log(MAX_PERIOD) * torch.arange(half, dtype=torch.float64) / half)
    a = t.double()[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class EpsMLP(nn.Module):
    def __init__(self, d, hidden, n_classes=0):
        super().__init__()
        self.d, self.n_classes = d, n_classes
        dims = [d + EMB_DIM + (n_classes + 1 if n_classes else 0)] + hidden + [d]
        self.lin = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x, t, c=None):
        h = [x, time_emb(t).to(x.dtype)]
        if self.n_classes:
            h.append(F.one_hot(c, self.n_classes + 1).to(x.dtype))
        h = torch.cat(h, 1)
        for i, l in enumerate(self.lin):
            h = l(h)
            if i < len(self.lin) - 1:
                h = F.silu(h)
        return h


def train(model, data, labels, steps, batch, lr, p_uncond=0.1, log_every=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ema = [p.detach().clone() for p in model.parameters()]
    ab = abar.float()
    history, run, t0 = [], 0.0, time.time()
    for it in range(1, steps + 1):
        idx = torch.randint(0, len(data), (batch,))
        x0 = data[idx]
        t = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x0)
        xt = ab[t].sqrt()[:, None] * x0 + (1 - ab[t]).sqrt()[:, None] * eps
        c = None
        if labels is not None:
            c = labels[idx].clone()
            c[torch.rand(batch) < p_uncond] = NULL
        loss = F.mse_loss(model(xt, t, c), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for q, p in zip(ema, model.parameters()):
                q.mul_(0.995).add_(p, alpha=0.005)
        run += loss.item()
        if it % log_every == 0:
            history.append({"step": it, "loss": round(run / log_every, 4)})
            print(f"  step {it:6d}  mean loss {run / log_every:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            run = 0.0
    with torch.no_grad():
        for q, p in zip(ema, model.parameters()):
            p.copy_(q)
    return history


# ----------------------------------------------------------------- samplers (float64)
def guided_eps(model, x, t, c, w):
    tt = torch.full((len(x),), int(t))
    e = model(x, tt, c)
    if c is not None and w != 0:
        e = (1 + w) * e - w * model(x, tt, torch.full((len(x),), NULL))
    return e


@torch.no_grad()
def ddpm_sample(model, xT, gen, c=None, w=0.0):
    x = xT
    for t in range(T - 1, -1, -1):
        e = guided_eps(model, x, t, c, w)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
        if t > 0:
            var = betas[t] * (1 - abar[t - 1]) / (1 - abar[t])
            x = mean + var.sqrt() * torch.randn(x.shape, generator=gen, dtype=torch.float64)
        else:
            x = mean
    return x


def ddim_timesteps(S):
    if S <= 1:
        return [T - 1]
    return [int(math.floor((T - 1) - (T - 1) * i / (S - 1) + 0.5)) for i in range(S)]


@torch.no_grad()
def ddim_sample(model, xT, S, c=None, w=0.0):
    x = xT
    ts = ddim_timesteps(S)
    for i, t in enumerate(ts):
        e = guided_eps(model, x, t, c, w)
        x0 = (x - (1 - abar[t]).sqrt() * e) / abar[t].sqrt()
        ap = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, dtype=torch.float64)
        x = ap.sqrt() * x0 + (1 - ap).sqrt() * e
    return x


def energy(a, b):
    """Unbiased (U-statistic) energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'|; ~0 for equal distributions."""
    exact = "donot_use_mm_for_euclid_dist"
    n, m = len(a), len(b)
    within = lambda p, k: torch.cdist(p, p, compute_mode=exact).sum() / (k * (k - 1))  # diagonal is exactly 0
    return (2 * torch.cdist(a, b, compute_mode=exact).mean() - within(a, n) - within(b, m)).item()


# ----------------------------------------------------------------- export helpers
def f16_round(model):
    """Round every parameter to float16 in place (what the browser receives); switch to float64."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())
    return model.double().eval()


def f16_tensor(t):
    a = t.detach().cpu().numpy().astype("<f2")
    return {"shape": list(a.shape), "f16": base64.b64encode(a.tobytes()).decode("ascii")}


def export_net(name, model, hidden, training, metrics):
    layers = [{"W": f16_tensor(l.weight.T.contiguous()), "b": f16_tensor(l.bias)} for l in model.lin]
    return {
        "name": name,
        "data_dim": model.d,
        "n_classes": model.n_classes,
        "hidden": hidden,
        "activation": "silu",
        "n_params": sum(p.numel() for p in model.parameters()),
        "layers": layers,
        "training": training,
        "metrics": metrics,
    }


r9 = lambda v: [float(f"{x:.12g}") for x in v]
t_start = time.time()

# ================================================================= (a) two moons
print("== moons ==")
Xm, _ = make_moons(n_samples=20000, noise=0.05, random_state=0)
MU, SD = Xm.mean(0), Xm.std(0)
Xm = (Xm - MU) / SD
data_m = torch.tensor(Xm, dtype=torch.float32)
H_M = [64, 64, 64]
moons = EpsMLP(2, H_M)
STEPS_M, BATCH_M, LR_M = 8000, 512, 2e-3
hist_m = train(moons, data_m, None, STEPS_M, BATCH_M, LR_M)
moons = f16_round(moons)

g = torch.Generator().manual_seed(1)
held = torch.tensor((make_moons(n_samples=4000, noise=0.05, random_state=1)[0] - MU) / SD)
fresh = torch.tensor((make_moons(n_samples=4000, noise=0.05, random_state=2)[0] - MU) / SD)
noise = torch.randn(4000, 2, generator=g, dtype=torch.float64)
m_metrics = {"energy_real_vs_real": energy(fresh, held), "energy_noise": energy(noise, held)}
m_metrics["energy_ddpm_100"] = energy(ddpm_sample(moons, torch.randn(4000, 2, generator=g, dtype=torch.float64), g), held)
for S in [5, 10, 20, 50, 100]:
    m_metrics[f"energy_ddim_{S}"] = energy(ddim_sample(moons, torch.randn(4000, 2, generator=g, dtype=torch.float64), S), held)
m_metrics = {k: round(v, 5) for k, v in m_metrics.items()}
for k, v in m_metrics.items():
    print(f"  {k:22s} {v:.5f}")

# ================================================================= (b) digits
print("== digits ==")
dg = load_digits()
Xd = torch.tensor(dg.data / 16.0 * 2 - 1, dtype=torch.float32)
yd = torch.tensor(dg.target)
H_D = [256, 256]
digits = EpsMLP(64, H_D, n_classes=10)
STEPS_D, BATCH_D, LR_D = 15000, 256, 1e-3
hist_d = train(digits, Xd, yd, STEPS_D, BATCH_D, LR_D, log_every=1500)
digits = f16_round(digits)

clf = LogisticRegression(max_iter=5000).fit(dg.data / 16.0, dg.target)
cls = torch.arange(10).repeat(30)  # 300 images, 30 per class
d_metrics = {}
for w in [0.0, 1.0, 3.0]:
    s = ddpm_sample(digits, torch.randn(300, 64, generator=g, dtype=torch.float64), g, cls, w).clamp(-1, 1)
    d_metrics[f"acc_ddpm_w{w:g}"] = float((clf.predict(((s + 1) / 2).numpy()) == cls.numpy()).mean())
    d_metrics[f"nn_dist_ddpm_w{w:g}"] = round(torch.cdist(s, Xd.double()).min(1).values.mean().item(), 4)
for S in [10, 25, 50]:
    s = ddim_sample(digits, torch.randn(300, 64, generator=g, dtype=torch.float64), S, cls, 1.0).clamp(-1, 1)
    d_metrics[f"acc_ddim{S}_w1"] = float((clf.predict(((s + 1) / 2).numpy()) == cls.numpy()).mean())
# reference: nearest-neighbour distance from a real image to a DIFFERENT real image
dd = torch.cdist(Xd.double(), Xd.double())
dd.fill_diagonal_(float("inf"))
d_metrics["nn_dist_real_to_other_real"] = round(dd.min(1).values.mean().item(), 4)
d_metrics["classifier_train_acc_on_real"] = float(clf.score(dg.data / 16.0, dg.target))
for k, v in d_metrics.items():
    print(f"  {k:30s} {v}")

# ----------------------------------------------------------------- verification vectors
gv = torch.Generator().manual_seed(123)
verify = {"eps": [], "ddim": []}
xv = torch.randn(6, 2, generator=gv, dtype=torch.float64) * 1.3
tv = torch.tensor([0, 10, 33, 50, 75, 99])
with torch.no_grad():
    verify["eps"].append({"net": "moons", "x": r9(xv.flatten().tolist()), "t": tv.tolist(), "c": None,
                          "out": r9(moons(xv, tv).flatten().tolist())})
    xv = torch.randn(4, 64, generator=gv, dtype=torch.float64)
    tv = torch.tensor([0, 20, 60, 99])
    cv = torch.tensor([3, 7, NULL, 0])
    verify["eps"].append({"net": "digits", "x": r9(xv.flatten().tolist()), "t": tv.tolist(), "c": cv.tolist(),
                          "out": r9(digits(xv, tv, cv).flatten().tolist())})
xT = torch.randn(8, 2, generator=gv, dtype=torch.float64)
verify["ddim"].append({"net": "moons", "steps": 100, "w": 0, "c": None, "xT": r9(xT.flatten().tolist()),
                       "x0": r9(ddim_sample(moons, xT, 100).flatten().tolist())})
verify["ddim"].append({"net": "moons", "steps": 10, "w": 0, "c": None, "xT": r9(xT.flatten().tolist()),
                       "x0": r9(ddim_sample(moons, xT, 10).flatten().tolist())})
xT = torch.randn(2, 64, generator=gv, dtype=torch.float64)
cv = torch.tensor([4, 8])
verify["ddim"].append({"net": "digits", "steps": 20, "w": 2.0, "c": cv.tolist(), "xT": r9(xT.flatten().tolist()),
                       "x0": r9(ddim_sample(digits, xT, 20, cv, 2.0).flatten().tolist())})

# ----------------------------------------------------------------- export
real_m = held[:1000]
pick = torch.cat([torch.nonzero(yd == k).flatten()[:6] for k in range(10)])
elapsed = time.time() - t_start
export = {
    "format": "diffusion-v1",
    "description": (
        f"Two DDPM epsilon-prediction MLPs trained with PyTorch by scripts/train-diffusion.py "
        f"(T={T}, linear betas {BETA_START}..{BETA_END}). moons: unconditional, 2-D two-moons data. "
        f"digits: class-conditional with classifier-free guidance, sklearn 8x8 digits."
    ),
    "schedule": {"T": T, "kind": "linear", "beta_start": BETA_START, "beta_end": BETA_END,
                 "betas": r9(betas.tolist()), "alpha_bar": r9(abar.tolist())},
    "time_embedding": {"dim": EMB_DIM, "max_period": MAX_PERIOD},
    "nets": {
        "moons": export_net("moons", moons, H_M, {
            "data": "sklearn make_moons(20000, noise=0.05, random_state=0), standardised per axis",
            "steps": STEPS_M, "batch": BATCH_M, "lr": LR_M, "optimizer": "Adam + cosine decay, EMA 0.995",
            "final_loss": hist_m[-1]["loss"], "history": hist_m, "mean": MU.tolist(), "std": SD.tolist()}, m_metrics),
        "digits": export_net("digits", digits, H_D, {
            "data": "sklearn load_digits (1797 images, 8x8, rescaled to [-1, 1]); label dropped with p=0.1",
            "steps": STEPS_D, "batch": BATCH_D, "lr": LR_D, "optimizer": "Adam + cosine decay, EMA 0.995",
            "final_loss": hist_d[-1]["loss"], "history": hist_d}, d_metrics),
    },
    "data": {
        "moons_real": f16_tensor(real_m),
        "digits_real": f16_tensor(Xd[pick]),
        "digits_labels": yd[pick].tolist(),
    },
    "verify": verify,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
print(f"moons params {export['nets']['moons']['n_params']}, digits params {export['nets']['digits']['n_params']}")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB) in {elapsed:.0f}s")
