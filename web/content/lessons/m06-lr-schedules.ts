import type { Lesson } from "@/lib/types";

const MATH_PY = String.raw`import numpy as np

# ---------------------------------------------------------------
# 1) Why a constant learning rate is a compromise.
#    SGD on f(x) = lam/2 x^2 with gradient noise of std sigma:
#    x <- x - eta*(lam*x + noise). Measure speed and the noise floor.
# ---------------------------------------------------------------
lam, sigma, x0 = 1.0, 1.0, 10.0
rng = np.random.default_rng(0)
noise = rng.normal(size=(2000, 3000)) * sigma          # 2000 independent runs, 3000 steps

def sgd(etas):
    x = np.full(noise.shape[0], x0)
    for t, eta in enumerate(etas):
        x = x - eta * (lam * x + noise[:, t])
    return 0.5 * lam * np.mean(x ** 2)                   # expected loss over the 2000 runs

print("constant eta: loss after 100 and 3000 steps  vs  predicted floor eta*sigma^2/(2(2-eta*lam))")
for eta in [0.5, 0.1, 0.01]:
    print(f"  eta={eta:<5}: {sgd(np.full(100, eta)):.5f}  {sgd(np.full(3000, eta)):.5f}  "
          f"floor {eta * sigma ** 2 / (2 * (2 - eta * lam)):.5f}")
decay = [0.5 if t < 100 else 0.5 / (1 + 0.5 * lam * (t - 100)) for t in range(3000)]  # ~1/t after a fast phase
print(f"  eta=0.5 then ~1/t decay: {sgd(decay[:100]):.5f}  {sgd(decay):.5f}")

# ---------------------------------------------------------------
# 2) Schedule values at a few steps (base lr 0.1, T = 1000 steps)
# ---------------------------------------------------------------
eta0, T, W = 0.1, 1000, 100
def step_decay(t, s=300, g=0.1):       return eta0 * g ** (t // s)
def exp_decay(t, g=0.01 ** (1 / T)):  return eta0 * g ** t          # reaches 0.01*eta0 at t = T
def linear(t):                         return eta0 * (1 - t / T)
def cosine(t, eta_min=0.0):            return eta_min + 0.5 * (eta0 - eta_min) * (1 + np.cos(np.pi * t / T))
def warm_cos(t):
    return eta0 * (t + 1) / W if t < W else 0.5 * eta0 * (1 + np.cos(np.pi * (t - W) / (T - W)))
def inv_sqrt(t):                       return eta0 * min((t + 1) / W, np.sqrt(W / (t + 1)))

print("\n step   step    exp     linear  cosine  warm+cos inv-sqrt")
for t in [0, 50, 99, 100, 250, 500, 750, 999]:
    print(f"{t:5d}  " + "  ".join(f"{f(t):.5f}" for f in [step_decay, exp_decay, linear, cosine, warm_cos, inv_sqrt]))

# Vaswani et al. (2017): lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
d_model, warm = 512, 4000
vas = lambda s: d_model ** -0.5 * min(s ** -0.5, s * warm ** -1.5)
print(f"\nTransformer schedule: step 1 {vas(1):.3e}, step 4000 (peak) {vas(4000):.3e}, "
      f"step 16000 {vas(16000):.3e}, step 100000 {vas(100000):.3e}")

# ---------------------------------------------------------------
# 3) Gradient clipping by norm vs by value
# ---------------------------------------------------------------
g = np.array([3.0, 4.0, 12.0])
c = 5.0
by_norm = g * min(1.0, c / np.linalg.norm(g))
by_value = np.clip(g, -c, c)
cosang = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print(f"\ng={g}, ||g||={np.linalg.norm(g):.1f}, threshold {c}")
print(f"  clip by norm : {np.round(by_norm, 4)}  norm {np.linalg.norm(by_norm):.4f}  cos(angle to g) {cosang(by_norm, g):.4f}")
print(f"  clip by value: {by_value}  norm {np.linalg.norm(by_value):.4f}  cos(angle to g) {cosang(by_value, g):.4f}")`;
const MATH_OUT = String.raw`constant eta: loss after 100 and 3000 steps  vs  predicted floor eta*sigma^2/(2(2-eta*lam))
  eta=0.5  : 0.16605  0.15970  floor 0.16667
  eta=0.1  : 0.02686  0.02605  floor 0.02632
  eta=0.01 : 6.70028  0.00250  floor 0.00251
  eta=0.5 then ~1/t decay: 0.16605  0.00017

 step   step    exp     linear  cosine  warm+cos inv-sqrt
    0  0.10000  0.10000  0.10000  0.10000  0.00100  0.00100
   50  0.10000  0.07943  0.09500  0.09938  0.05100  0.05100
   99  0.10000  0.06339  0.09010  0.09760  0.10000  0.10000
  100  0.10000  0.06310  0.09000  0.09755  0.10000  0.09950
  250  0.10000  0.03162  0.07500  0.08536  0.09330  0.06312
  500  0.01000  0.01000  0.05000  0.05000  0.05868  0.04468
  750  0.00100  0.00316  0.02500  0.01464  0.01786  0.03649
  999  0.00010  0.00100  0.00010  0.00000  0.00000  0.03162

Transformer schedule: step 1 1.747e-07, step 4000 (peak) 6.988e-04, step 16000 3.494e-04, step 100000 1.398e-04

g=[ 3.  4. 12.], ||g||=13.0, threshold 5.0
  clip by norm : [1.1538 1.5385 4.6154]  norm 5.0000  cos(angle to g) 1.0000
  clip by value: [3. 4. 5.]  norm 7.0711  cos(angle to g) 0.9247`;
const SCRATCH_PY = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# Schedules: each maps a step t in [0, T) to a learning rate
# ---------------------------------------------------------------
def make_schedule(kind, base, T, warmup=0, step_size=None, gamma=0.1, milestones=(), t0=None, t_mult=2,
                  max_lr=None, pct_start=0.3, div=25.0, final_div=1e4):
    def cos_interp(a, b, frac):                    # cosine from a (frac=0) to b (frac=1)
        return b + 0.5 * (a - b) * (1 + np.cos(np.pi * frac))
    def f(t):
        if kind == "constant":
            return base
        if kind == "step":                         # multiply by gamma every step_size steps
            return base * gamma ** (t // step_size)
        if kind == "multistep":                    # multiply by gamma at each milestone
            return base * gamma ** sum(t >= m for m in milestones)
        if kind == "exponential":                  # gamma per step
            return base * gamma ** t
        if kind == "cosine":
            return cos_interp(base, 0.0, t / T)
        if kind == "cosine_restarts":              # SGDR: periods t0, t0*t_mult, ...
            period, start = t0, 0
            while t >= start + period:
                start, period = start + period, period * t_mult
            return cos_interp(base, 0.0, (t - start) / period)
        if kind == "warmup_cosine":
            if t < warmup:
                return base * (t + 1) / warmup     # linear warm-up, reaches base at t = warmup-1
            return cos_interp(base, 0.0, (t - warmup) / (T - warmup))
        if kind == "one_cycle":                    # PyTorch OneCycleLR (two phases, cosine)
            peak = max_lr or base
            lo, end = peak / div, peak / div / final_div
            up_end = pct_start * T - 1
            if t <= up_end:
                return cos_interp(lo, peak, t / up_end)
            return cos_interp(peak, end, (t - up_end) / (T - 1 - up_end))
        if kind == "inv_sqrt":                     # Transformer: warm-up then 1/sqrt(t), peak = base
            return base * min((t + 1) / warmup, np.sqrt(warmup / (t + 1)))
        raise ValueError(kind)
    return f

# ---------------------------------------------------------------
# Data and a small MLP (64 -> 64 -> 10, ReLU, softmax cross-entropy)
# ---------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X = X / 16.0                                       # pixel values 0..16 -> 0..1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Y = np.eye(10)[ytr]
noisy = ytr.copy()                                 # a second training set with 30% random labels
flip = np.random.default_rng(999).random(len(noisy)) < 0.3
noisy[flip] = np.random.default_rng(998).integers(0, 10, flip.sum())

def init(seed):
    r = np.random.default_rng(seed)
    return [r.normal(0, np.sqrt(2 / 64), (64, 64)), np.zeros(64),   # He init for ReLU
            r.normal(0, np.sqrt(1 / 64), (64, 10)), np.zeros(10)]

def forward(P, A):
    h = np.maximum(0, A @ P[0] + P[1])
    z = h @ P[2] + P[3]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return h, p

def loss_grads(P, A, Yb):
    h, p = forward(P, A)
    loss = -np.mean(np.sum(Yb * np.log(p + 1e-12), 1))
    dz = (p - Yb) / len(A)
    dh = (dz @ P[2].T) * (h > 0)
    return loss, [A.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]

def clip_by_norm(grads, c):
    norm = np.sqrt(sum(np.sum(g * g) for g in grads))   # ONE global norm over all tensors
    return [g * min(1.0, c / (norm + 1e-12)) for g in grads], norm

def train(schedule, seed, labels, epochs=30, batch=32, beta=0.9, clip=None):
    P, V = init(seed), None
    Y = np.eye(10)[labels]                         # one-hot targets
    order = np.random.default_rng(100 + seed)
    t = 0
    for _ in range(epochs):
        idx = order.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            b = idx[s:s + batch]
            _, G = loss_grads(P, Xtr[b], Y[b])
            if clip is not None:
                G, _ = clip_by_norm(G, clip)
            V = G if V is None else [beta * v + g for v, g in zip(V, G)]   # SGD + momentum
            lr = schedule(t)                       # one scheduler step per optimizer step
            P = [p - lr * v for p, v in zip(P, V)]
            t += 1
    p_test = forward(P, Xte)[1]
    test_nll = -np.mean(np.log(p_test[np.arange(len(yte)), yte] + 1e-12))
    return test_nll, np.mean(p_test.argmax(1) == yte)

epochs, batch = 30, 32
T = epochs * int(np.ceil(len(Xtr) / batch))
base = 0.2
print(f"digits: {len(Xtr)} train / {len(Xte)} test images, {T} optimizer steps, SGD momentum 0.9, base lr {base}")

# ---------------------------------------------------------------
# 1) LR range test: raise lr exponentially for 100 steps, watch the loss
# ---------------------------------------------------------------
P, V, order = init(0), None, np.random.default_rng(0)
lrs = np.logspace(-4, 1, 100)
smooth, rows = None, []
for i, lr in enumerate(lrs):
    b = order.choice(len(Xtr), batch, replace=False)
    loss, G = loss_grads(P, Xtr[b], Y[b])
    smooth = loss if smooth is None else 0.8 * smooth + 0.2 * loss   # smoothed loss
    rows.append((lr, smooth))
    V = G if V is None else [0.9 * v + g for v, g in zip(V, G)]
    P = [p - lr * v for p, v in zip(P, V)]
    if not np.isfinite(loss) or smooth > 4 * rows[0][1]:
        break
lr_min_loss = min(rows, key=lambda r: r[1])[0]
print("LR range test (lr: smoothed loss):", ", ".join(f"{lr:.0e}: {l:.2f}" for lr, l in rows[::12]))
print(f"  minimum smoothed loss at lr={lr_min_loss:.3f}; stopped at lr={rows[-1][0]:.2f}; rule of thumb: use ~1/10 -> {lr_min_loss / 10:.3f}")

# ---------------------------------------------------------------
# 2) The real experiment: same model, data, seeds; only the schedule changes
# ---------------------------------------------------------------
schedules = {
    "constant":            make_schedule("constant", base, T),
    "step (x0.1 @50,75%)": make_schedule("multistep", base, T, milestones=(T // 2, 3 * T // 4)),
    "cosine":              make_schedule("cosine", base, T),
    "warmup 5% + cosine":  make_schedule("warmup_cosine", base, T, warmup=T // 20),
    "one-cycle":           make_schedule("one_cycle", base, T),
}
for title, labels in [("clean labels", ytr), ("30% of training labels randomised", noisy)]:
    print()
    print(f"{title}: test NLL and test accuracy, mean over 5 seeds (min-max)")
    for name, sch in schedules.items():
        res = np.array([train(sch, seed, labels) for seed in range(5)])
        print(f"  {name:20s} NLL {res[:, 0].mean():.3f}   acc {res[:, 1].mean():.4f} ({res[:, 1].min():.3f}-{res[:, 1].max():.3f})")

# ---------------------------------------------------------------
# 3) Too-high peak lr (0.6, cosine): clipping the global gradient norm
# ---------------------------------------------------------------
print()
print("cosine from lr=0.6 (clean labels), test accuracy over 5 seeds")
for clip in [None, 2.0, 1.0, 0.5]:
    res = np.array([train(make_schedule("cosine", 0.6, T), seed, ytr, clip=clip) for seed in range(5)])
    print(f"  clip={str(clip):4s}: mean {res[:, 1].mean():.4f}  per seed {np.round(res[:, 1], 3)}")`;
const SCRATCH_OUT = String.raw`digits: 1257 train / 540 test images, 1200 optimizer steps, SGD momentum 0.9, base lr 0.2
LR range test (lr: smoothed loss): 1e-04: 2.37, 4e-04: 2.37, 2e-03: 2.32, 7e-03: 2.23, 3e-02: 1.97, 1e-01: 1.23, 4e-01: 0.59, 2e+00: 1.53
  minimum smoothed loss at lr=0.343; stopped at lr=3.51; rule of thumb: use ~1/10 -> 0.034

clean labels: test NLL and test accuracy, mean over 5 seeds (min-max)
  constant             NLL 0.090   acc 0.9781 (0.976-0.983)
  step (x0.1 @50,75%)  NLL 0.088   acc 0.9778 (0.974-0.983)
  cosine               NLL 0.095   acc 0.9778 (0.976-0.981)
  warmup 5% + cosine   NLL 0.083   acc 0.9756 (0.972-0.978)
  one-cycle            NLL 0.082   acc 0.9778 (0.972-0.981)

30% of training labels randomised: test NLL and test accuracy, mean over 5 seeds (min-max)
  constant             NLL 0.736   acc 0.8119 (0.757-0.867)
  step (x0.1 @50,75%)  NLL 0.544   acc 0.9170 (0.883-0.933)
  cosine               NLL 0.525   acc 0.9226 (0.911-0.935)
  warmup 5% + cosine   NLL 0.514   acc 0.9256 (0.919-0.937)
  one-cycle            NLL 0.508   acc 0.9289 (0.922-0.939)

cosine from lr=0.6 (clean labels), test accuracy over 5 seeds
  clip=None: mean 0.7193  per seed [0.656 0.9   0.843 0.581 0.617]
  clip=2.0 : mean 0.9022  per seed [0.889 0.893 0.926 0.893 0.911]
  clip=1.0 : mean 0.9663  per seed [0.972 0.969 0.957 0.965 0.969]
  clip=0.5 : mean 0.9715  per seed [0.974 0.969 0.978 0.969 0.969]`;
const TORCH_PY = String.raw`import math
import numpy as np
import torch
torch.set_num_threads(1)                               # tiny matrices: one thread is fastest
from torch.optim import lr_scheduler as S
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def lrs_of(make_sched, steps, base=0.1):
    """Learning rate used at each of the first "steps" optimizer steps (scheduler.step() once per step)."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base)
    sched = make_sched(opt)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()                                     # optimizer first ...
        sched.step()                                   # ... then the scheduler
    return np.array(out)

T, W, base = 1000, 100, 0.1
cos = lambda a, b, f: b + 0.5 * (a - b) * (1 + math.cos(math.pi * f))
def sgdr(t, t0=200, mult=2):                           # closed form of warm restarts
    start, period = 0, t0
    while t >= start + period:
        start, period = start + period, period * mult
    return cos(base, 0.0, (t - start) / period)
def one_cycle(t, peak=base, pct=0.3, div=25.0, final_div=1e4):
    lo, up_end = peak / div, pct * T - 1
    return cos(lo, peak, t / up_end) if t <= up_end else cos(peak, lo / final_div, (t - up_end) / (T - 1 - up_end))
warm_lambda = lambda t: min(1.0, (t + 1) / W) if t < W else cos(1.0, 0.0, (t - W) / (T - W))

checks = {
    "StepLR(step_size=300, gamma=0.1)": (lambda o: S.StepLR(o, step_size=300, gamma=0.1),
                                         lambda t: base * 0.1 ** (t // 300)),
    "ExponentialLR(gamma=0.995)":       (lambda o: S.ExponentialLR(o, gamma=0.995), lambda t: base * 0.995 ** t),
    "CosineAnnealingLR(T_max=1000)":    (lambda o: S.CosineAnnealingLR(o, T_max=T), lambda t: cos(base, 0, t / T)),
    "CosineAnnealingWarmRestarts(200,2)": (lambda o: S.CosineAnnealingWarmRestarts(o, T_0=200, T_mult=2), sgdr),
    "OneCycleLR(max_lr=0.1, total=1000)": (lambda o: S.OneCycleLR(o, max_lr=base, total_steps=T), one_cycle),
    "LambdaLR(warm-up 100 + cosine)":   (lambda o: S.LambdaLR(o, warm_lambda), lambda t: base * warm_lambda(t)),
}
print("PyTorch scheduler vs closed-form formula (1000 steps)")
for name, (make, formula) in checks.items():
    got = lrs_of(make, T)
    ref = np.array([formula(t) for t in range(T)])
    print(f"  {name:36s} max diff {np.abs(got - ref).max():.1e}   lr at t=0,100,500,999: "
          + ", ".join(f"{got[t]:.5f}" for t in (0, 100, 500, 999)))

seq = lrs_of(lambda o: S.SequentialLR(o, [S.LinearLR(o, start_factor=0.01, end_factor=1.0, total_iters=W),
                                         S.CosineAnnealingLR(o, T_max=T - W)], milestones=[W]), T)
print("  SequentialLR(LinearLR -> Cosine)     lr at t=0,50,100,550,999: "
      + ", ".join(f"{seq[t]:.5f}" for t in (0, 50, 100, 550, 999)))

# ---------------------------------------------------------------
# The classic bug: T_max given in EPOCHS but scheduler.step() called every BATCH
# ---------------------------------------------------------------
epochs, batches_per_epoch = 10, 50
bug = lrs_of(lambda o: S.CosineAnnealingLR(o, T_max=epochs), epochs * batches_per_epoch)
print("Bug: CosineAnnealingLR(T_max=10 epochs) stepped per batch, lr at the start of epochs 0..4:",
      np.round(bug[::batches_per_epoch][:5], 4), " batches 0..20:", np.round(bug[:21:4], 4))

# ---------------------------------------------------------------
# Real run: 8-layer MLP on digits, Adam lr=1e-2, with and without warm-up
# ---------------------------------------------------------------
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)

def train(seed, warmup_frac, clip=None, lr=1e-2, epochs=15, batch=64):
    torch.manual_seed(seed)
    layers, d = [], 64
    for _ in range(8):                                 # deep, no normalisation: early steps are fragile
        layers += [torch.nn.Linear(d, 128), torch.nn.ReLU()]
        d = 128
    model = torch.nn.Sequential(*layers, torch.nn.Linear(d, 10))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total = epochs * math.ceil(len(Xtr) / batch)
    w = int(warmup_frac * total)
    if w:
        sched = S.SequentialLR(opt, [S.LinearLR(opt, start_factor=1 / w, total_iters=w),
                                     S.CosineAnnealingLR(opt, T_max=total - w)], milestones=[w])
    else:
        sched = S.CosineAnnealingLR(opt, T_max=total)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=gen)
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            loss = torch.nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            sched.step()                               # per batch, because T_max is in batches
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()

print("8-layer MLP, Adam lr=1e-2, cosine decay, test accuracy over 5 seeds")
for label, wf, clip in [("no warm-up", 0.0, None), ("10% linear warm-up", 0.1, None), ("no warm-up, clip 1.0", 0.0, 1.0)]:
    accs = np.array([train(s, wf, clip) for s in range(5)])
    print(f"  {label:22s} mean {accs.mean():.4f}  per seed {np.round(accs, 3)}")`;
const TORCH_OUT = String.raw`PyTorch scheduler vs closed-form formula (1000 steps)
  StepLR(step_size=300, gamma=0.1)     max diff 0.0e+00   lr at t=0,100,500,999: 0.10000, 0.10000, 0.01000, 0.00010
  ExponentialLR(gamma=0.995)           max diff 4.2e-17   lr at t=0,100,500,999: 0.10000, 0.06058, 0.00816, 0.00067
  CosineAnnealingLR(T_max=1000)        max diff 1.1e-16   lr at t=0,100,500,999: 0.10000, 0.09755, 0.05000, 0.00000
  CosineAnnealingWarmRestarts(200,2)   max diff 2.8e-17   lr at t=0,100,500,999: 0.10000, 0.05000, 0.01464, 0.05020
  OneCycleLR(max_lr=0.1, total=1000)   max diff 0.0e+00   lr at t=0,100,500,999: 0.00400, 0.02815, 0.08100, 0.00000
  LambdaLR(warm-up 100 + cosine)       max diff 0.0e+00   lr at t=0,100,500,999: 0.00100, 0.10000, 0.05868, 0.00000
  SequentialLR(LinearLR -> Cosine)     lr at t=0,50,100,550,999: 0.00100, 0.05050, 0.10000, 0.05000, 0.00000
Bug: CosineAnnealingLR(T_max=10 epochs) stepped per batch, lr at the start of epochs 0..4: [0.1 0.  0.1 0.  0.1]  batches 0..20: [0.1    0.0655 0.0095 0.0095 0.0655 0.1   ]
8-layer MLP, Adam lr=1e-2, cosine decay, test accuracy over 5 seeds
  no warm-up             mean 0.8274  per seed [0.852 0.817 0.7   0.893 0.876]
  10% linear warm-up     mean 0.9000  per seed [0.954 0.935 0.839 0.883 0.889]
  no warm-up, clip 1.0   mean 0.8552  per seed [0.902 0.907 0.656 0.904 0.907]`;
const EX_PY = String.raw`import numpy as np
import torch

def clip_global_norm(grads, max_norm, eps=1e-6):
    """Scale ALL gradient arrays by one common factor so their joint L2 norm is <= max_norm."""
    total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
    coef = min(1.0, max_norm / (total + eps))         # PyTorch adds eps=1e-6 to the norm
    return [g * coef for g in grads], total

def clip_per_tensor(grads, max_norm):
    """The common mistake: clip each tensor separately (changes the overall direction)."""
    return [g * min(1.0, max_norm / np.linalg.norm(g)) for g in grads]

rng = np.random.default_rng(0)
grads = [rng.normal(size=(4, 3)) * 5, rng.normal(size=3) * 0.1]   # a big weight grad, a small bias grad

clipped, total = clip_global_norm(grads, max_norm=1.0)
print(f"total norm before: {total:.4f}, after: {np.sqrt(sum(np.sum(g ** 2) for g in clipped)):.4f}")

# Compare with PyTorch
params = [torch.nn.Parameter(torch.zeros(g.shape, dtype=torch.float64)) for g in grads]
for p, g in zip(params, grads):
    p.grad = torch.tensor(g)
returned = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
print(f"torch returned norm {returned.item():.4f}; max |ours - torch| = "
      f"{max(np.abs(c - p.grad.numpy()).max() for c, p in zip(clipped, params)):.1e}")

per = clip_per_tensor(grads, 1.0)
flat = lambda gs: np.concatenate([g.ravel() for g in gs])
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print(f"cos(angle) to the raw gradient: global {cos(flat(clipped), flat(grads)):.4f}, per-tensor {cos(flat(per), flat(grads)):.4f}")
print(f"bias-gradient norm: raw {np.linalg.norm(grads[1]):.4f}, global {np.linalg.norm(clipped[1]):.4f}, per-tensor {np.linalg.norm(per[1]):.4f}")`;
const EX_OUT = String.raw`total norm before: 12.2055, after: 1.0000
torch returned norm 12.2055; max |ours - torch| = 0.0e+00
cos(angle) to the raw gradient: global 1.0000, per-tensor 0.9720
bias-gradient norm: raw 0.2647, global 0.0217, per-tensor 0.2647`;

const lesson: Lesson = {
  id: "m06/lr-schedules",
  title: "Learning-Rate Schedules, Warm-up and Gradient Clipping",
  summary:
    "Why one fixed learning rate forces a trade-off between speed and precision, and the schedules that remove it: step, exponential, polynomial/linear, cosine annealing and warm restarts, linear warm-up, the Transformer inverse-square-root schedule, one-cycle, ReduceLROnPlateau and the LR range test; the linear scaling rule for batch size; gradient clipping by norm and by value; and PyTorch's lr_scheduler API with its most common bugs — all backed by real training runs.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m06/gradient-descent", "m06/momentum-adaptive", "m03/training-basics"],
  next: ["m07/initialization", "m07/normalization", "m07/regularization", "m07/residual-connections"],
  terms: ["learning-rate-schedule", "warmup", "cosine-annealing", "warm-restarts", "one-cycle", "lr-range-test", "linear-scaling-rule", "gradient-clipping", "noise-floor"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "The learning rate η is the most important setting in almost every training run, and there is no reason it should be the same on step 1 and step 1 000 000. Early on, the weights are far from a good solution, so big steps make fast progress. Late in training, the weights are close to a minimum and the leftover error comes mostly from mini-batch noise, which only small steps can average away. A **learning-rate schedule** makes η change with the step, $\\eta_t$, so one run gets both.",
      },
      {
        t: "p",
        text: "Robbins and Monro (1951) already showed that noisy optimisation needs a shrinking step size. Deep learning first used **step decay** (divide η by 10 when validation error stops improving, as in AlexNet and ResNet), then smooth **cosine annealing** (Loshchilov & Hutter, SGDR, 2016), **linear warm-up** for large batches (Goyal et al., 2017) and Transformers (Vaswani et al., 2017), and **one-cycle** (Smith, 2017–2018). Today's standard for large models is linear warm-up, then cosine (or linear) decay, plus **gradient clipping**, which caps any single update when a rare batch produces a huge gradient.",
      },
      {
        t: "callout",
        kind: "key",
        title: "Main results",
        text: String.raw`With a constant $\eta$, SGD settles on a **noise floor** proportional to $\eta$ — the speed/precision trade-off. Decay removes it. On digits with 30% wrong labels, a constant rate reached 81.2% test accuracy while step, cosine, warm-up + cosine and one-cycle reached 91.7–92.9% (same model, data and seeds). Warm-up raised an 8-layer Adam network from 82.7% to 90.0%. Clipping the gradient norm at 1.0 turned an unstable peak rate (72% accuracy) into a stable one (96.6%).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Gradient Descent and Its Descendants](/learn/m06/gradient-descent) — the update rule, the stability condition $\eta < 2/\lambda_{\max}$ on a quadratic, mini-batch gradient noise with variance $\propto 1/B$, and the Robbins–Monro conditions.`,
          String.raw`[Momentum and Adaptive Optimizers in Depth](/learn/m06/momentum-adaptive) — momentum's effective step $\eta/(1-\beta)$, and Adam's normalised step and bias correction, which explain why warm-up helps Adam.`,
          "[Training basics: epochs, batches and iterations](/learn/m03/training-basics) — schedules count in iterations or in epochs; mixing the two up is the most common bug.",
          "[Classification losses](/learn/m05/classification-losses) — the cross-entropy loss used in the experiments.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Parking a car",
        text: "To park, you drive fast on the open road, slow down as you get close, and creep the last metre. Driving the whole way at parking speed takes forever; parking at highway speed overshoots every time. A learning-rate schedule is that speed plan. **Warm-up** is pulling out of the driveway gently before speeding up — at the start the optimizer knows nothing yet and the network is fragile. **Gradient clipping** is a speed limiter: one bad reading of the road can't launch you into a wall.",
      },
      {
        t: "p",
        text: "Why can't one fixed rate do both jobs? Each step moves the weights by η times a **noisy** gradient. Far from the minimum, the true gradient is big and the noise is small next to it, so a big η makes fast progress. Near the minimum, the true gradient is tiny but the noise is not. Every step kicks the weights by about η times the noise, so they jitter around the minimum in a cloud whose size grows with η. A smaller η shrinks the cloud — but starting small wastes most of the run getting there.",
      },
      {
        t: "list",
        items: [
          "**Schedule** — the rule step → learning rate.",
          "**Peak (base) learning rate** — the largest η used; most schedules are written as a multiplier of it.",
          "**Warm-up** — a short first phase where η ramps up from near 0 to the peak.",
          "**Decay / annealing** — the phase where η falls toward 0 or a small floor.",
          "**Gradient clipping** — capping the gradient's size (or each component) before the optimizer step.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "Where the schedule and clipping sit in the training loop" },
      {
        t: "steps",
        items: [
          { title: "Forward and backward pass", text: "Compute the mini-batch loss and call `loss.backward()` to fill every parameter's `.grad`." },
          { title: "Optional: unscale (mixed precision)", text: "With a gradient scaler, call `scaler.unscale_(optimizer)` first — otherwise you would clip the scaled gradients." },
          { title: "Clip", text: String.raw`Compute the global norm $\lVert g\rVert$ over all parameters; if it is above $c$, multiply every gradient by $c/\lVert g\rVert$ (PyTorch: clip_grad_norm_).` },
          { title: "Optimizer step with the current η", text: "The optimizer reads `param_group['lr']`, which the scheduler set." },
          { title: "Scheduler step", text: "`scheduler.step()` computes the learning rate for the **next** step and writes it into the optimizer. Call it once per iteration for iteration-based schedules, once per epoch for epoch-based ones — never both." },
        ],
      },
      { t: "h", text: "The parts of a modern schedule" },
      {
        t: "table",
        head: ["Phase", "Typical length", "What happens", "Why"],
        rows: [
          ["Warm-up", "0.5–10% of training (e.g. 2000 steps for LLMs, 5 epochs in Goyal et al.)", "η rises linearly from ≈0 to the peak", "optimizer statistics are empty, early curvature and gradients are large; large batches need a gentle start"],
          ["Plateau (optional)", "0–80%", "η stays at the peak", "fast exploring; used by step schedules and warm-up-stable-decay"],
          ["Decay", "the rest", "η falls (cosine, linear, steps, 1/√t)", "shrink the noise cloud, settle into a minimum"],
          ["Floor", "end", String.raw`$\eta_{\min}$ = 0 or ≈1–10% of the peak`, "a small non-zero floor keeps learning if training is extended"],
        ],
      },
      { t: "h", text: "Choosing the peak learning rate: the LR range test" },
      {
        t: "steps",
        items: [
          { title: "Start tiny", text: String.raw`Begin with a very small η (e.g. $10^{-4}$ or less).` },
          { title: "Grow it by a fixed factor every step", text: String.raw`Multiply η by a constant each mini-batch so it covers 5–7 orders of magnitude in about 100 steps.` },
          { title: "Record a smoothed loss", text: "Mini-batch losses are noisy; use an EMA (running average) of the loss." },
          { title: "Stop when the loss explodes", text: "For example when it passes 4× its starting value." },
          { title: "Read off η", text: "The loss falls fastest well below the point where it bottoms out. Common rules: the η where the curve drops most steeply, or about a tenth of the η at the minimum. For one-cycle, the η at the minimum is often used as the maximum." },
        ],
      },
      {
        t: "p",
        text: "The test came with cyclical learning rates (Smith, 2017) and was made popular by fastai's `lr_find`. One short run replaces a grid search. It finds the **largest stable** η for the network right now; the schedule then decides how η changes.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Why a constant learning rate is a compromise: the noise floor" },
      {
        t: "p",
        text: String.raw`Model one direction of the loss near a minimum: $L(x) = \tfrac{\lambda}{2}x^2$ with curvature $\lambda$, and a noisy gradient $\lambda x_t + \xi_t$ whose noise $\xi_t$ has mean 0 and variance $\sigma^2$ (mini-batch noise). SGD with constant $\eta$ gives`,
      },
      { t: "math", tex: String.raw`x_{t+1} = (1 - \eta\lambda)\,x_t - \eta\,\xi_t` },
      {
        t: "p",
        text: String.raw`On average, $\mathbb{E}[x_t] = (1-\eta\lambda)^t x_0$ shrinks geometrically — the part a large $\eta$ removes fast. For the spread: $\xi_t$ is independent of $x_t$, so $\operatorname{Var}[x_{t+1}] = (1-\eta\lambda)^2 \operatorname{Var}[x_t] + \eta^2\sigma^2$. It settles where both sides are equal:`,
      },
      {
        t: "math",
        tex: String.raw`\operatorname{Var}_\infty = \frac{\eta^2\sigma^2}{1 - (1-\eta\lambda)^2} = \frac{\eta\,\sigma^2}{\lambda\,(2 - \eta\lambda)}, \qquad \mathbb{E}[L]_\infty = \frac{\lambda}{2}\operatorname{Var}_\infty = \frac{\eta\,\sigma^2}{2\,(2-\eta\lambda)} \approx \frac{\eta\,\sigma^2}{4}`,
        caption: String.raw`$\mathbb{E}[L]_\infty$ — the expected loss after convergence with a constant $\eta$: the noise floor.`,
      },
      {
        t: "p",
        text: String.raw`Two forces pull against each other. The time to reach the floor grows like $1/(\eta\lambda)$, which favours a large $\eta$. The floor itself grows like $\eta\sigma^2$, which favours a small $\eta$. No fixed value wins both. With $\lambda = \sigma = 1$, $x_0 = 10$, averaged over 2000 runs: $\eta = 0.5$ reaches its floor within 100 steps but stays at 0.160 (prediction 0.1667); $\eta = 0.01$ has a floor of 0.0025 but is still at 6.70 after 100 steps. Starting at 0.5 and then decaying like $1/t$ reaches **0.00017** after 3000 steps — below the floor of every constant rate tried. That is why schedules exist, and why the Robbins–Monro conditions ($\sum\eta_t = \infty$, $\sum\eta_t^2 < \infty$) need a decaying rate for exact convergence.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "When the floor doesn't matter",
        text: String.raw`If the model can fit every training example exactly (a big network on clean data), every example's gradient is zero at the solution, so $\sigma^2 \to 0$ there and a constant rate can converge too. The experiment below shows this: on clean digits the schedule barely matters, but with 30% wrong labels — which can't all be fitted, so gradient noise stays — decaying schedules win by more than 10 percentage points.`,
      },

      { t: "h", text: "2. The schedules" },
      {
        t: "p",
        text: String.raw`Notation: $t = 0, 1, \dots, T-1$ is the step (or epoch), $T$ the total, $\eta_0$ the peak, $\eta_{\min}$ the final floor (often 0), $W$ the warm-up length.`,
      },
      {
        t: "table",
        head: ["Schedule", "Formula", "Parameters", "Notes"],
        rows: [
          ["Constant", String.raw`$\eta_t = \eta_0$`, "—", "baseline; noise floor ∝ η"],
          ["Step decay", String.raw`$\eta_t = \eta_0\,\gamma^{\lfloor t/s\rfloor}$`, String.raw`step size $s$, factor $\gamma$ (often 0.1)`, "classic ImageNet recipe; MultiStep uses explicit milestones"],
          ["Exponential", String.raw`$\eta_t = \eta_0\,\gamma^{t}$`, String.raw`$\gamma$ slightly below 1`, String.raw`smooth; $\gamma = (\eta_T/\eta_0)^{1/T}$ hits a chosen final value`],
          ["Polynomial / linear", String.raw`$\eta_t = (\eta_0 - \eta_{\min})(1 - t/T)^p + \eta_{\min}$`, String.raw`power $p$ ($p = 1$: linear)`, "linear decay to 0 is common for BERT-style fine-tuning"],
          ["Cosine annealing", String.raw`$\eta_t = \eta_{\min} + \tfrac12(\eta_0-\eta_{\min})\big(1+\cos(\pi t/T)\big)$`, String.raw`$T$, $\eta_{\min}$`, "slow decay at first, fastest in the middle, gentle at the end"],
          ["Cosine with warm restarts (SGDR)", String.raw`cosine with $t \to t - t_i$ inside cycle $i$ of length $T_i = T_0\,T_{\text{mult}}^{\,i}$`, String.raw`$T_0$, $T_{\text{mult}}$`, "jumps back to the peak each cycle; any cycle end gives a usable model"],
          ["Linear warm-up", String.raw`$\eta_t = \eta_0\,(t+1)/W$ for $t < W$`, String.raw`$W$`, "always paired with a decay"],
          ["Warm-up + cosine", String.raw`warm-up, then cosine over the remaining $T - W$ steps`, String.raw`$W$, $T$, $\eta_{\min}$`, "the standard Transformer / LLM recipe"],
          ["Inverse square root (Transformer)", String.raw`$\eta_t = d_{\text{model}}^{-1/2}\min\!\big(t^{-1/2},\ t\,W^{-3/2}\big)$`, String.raw`$d_{\text{model}}$, $W$ (= 4000)`, String.raw`Vaswani et al. 2017; $t$ counts from 1; peak at $t = W$`],
          ["One-cycle", String.raw`cosine from $\eta_{\max}/25$ up to $\eta_{\max}$ over 30% of training, then cosine down to $\eta_{\max}/(25\cdot 10^4)$`, "max_lr, pct_start, div_factor, final_div_factor", "PyTorch defaults shown; momentum moves the opposite way (0.95 → 0.85 → 0.95)"],
          ["ReduceLROnPlateau", String.raw`$\eta \leftarrow \gamma\,\eta$ if the watched metric has not improved for $p$ epochs`, String.raw`factor $\gamma$ = 0.1, patience $p$ = 10 (PyTorch defaults)`, "reacts to the run, not a function of t; needs a validation metric"],
        ],
      },
      {
        t: "p",
        text: String.raw`**Cosine annealing up close.** At $t = 0$, $\cos 0 = 1$ gives $\eta_0$; at $t = T/2$, $\cos(\pi/2) = 0$ gives the midpoint $(\eta_0+\eta_{\min})/2$; at $t = T$, $\cos\pi = -1$ gives $\eta_{\min}$. Its slope $-\tfrac{\pi}{2T}(\eta_0-\eta_{\min})\sin(\pi t/T)$ is zero at both ends, so the rate leaves the peak slowly (long exploring) and nears the floor slowly (long settling). Loshchilov & Hutter ("SGDR: Stochastic Gradient Descent with Warm Restarts", 2016) added **restarts**: after each cycle the rate jumps back to $\eta_0$, with cycle lengths $T_0, T_0 T_{\text{mult}}, T_0 T_{\text{mult}}^2, \dots$ The jump can kick the weights out of a narrow valley into a possibly better one, and every cycle end gives a well-settled model.`,
      },
      {
        t: "p",
        text: String.raw`**The original Transformer schedule.** Vaswani et al. ("Attention Is All You Need", 2017) used Adam with $\beta_2 = 0.98$, $\epsilon = 10^{-9}$ and $\eta_t = d_{\text{model}}^{-1/2}\min(t^{-1/2}, t\cdot W^{-3/2})$ with $W = 4000$. For $t \le W$ the second term is smaller, so $\eta_t = d_{\text{model}}^{-1/2}\,t\,W^{-3/2}$ grows linearly. For $t \ge W$ the first is smaller, so $\eta_t \propto 1/\sqrt t$. They meet at $t = W$, where $\eta = (d_{\text{model}}W)^{-1/2}$. With $d_{\text{model}} = 512$: $1.747\times10^{-7}$ at step 1, peak $6.988\times10^{-4}$ at step 4000, half the peak ($3.494\times10^{-4}$) at step 16 000 (4× the warm-up, and $\sqrt{4} = 2$), and $1.398\times10^{-4}$ at step 100 000. The $1/\sqrt t$ tail never hits zero, which suits runs of unknown length.`,
      },
      { t: "h", text: "Worked example: every schedule at the same steps" },
      {
        t: "p",
        text: String.raw`Peak $\eta_0 = 0.1$, $T = 1000$ steps, warm-up $W = 100$. Step decay: $\gamma = 0.1$ every 300 steps. Exponential: $\gamma = 0.01^{1/1000} = 0.995405$, so that $\eta_{1000} = 0.001$. Inverse-sqrt is scaled to peak at $\eta_0$: $\eta_t = \eta_0\min\big((t+1)/W, \sqrt{W/(t+1)}\big)$.`,
      },
      {
        t: "table",
        head: ["$t$", "step", "exponential", "linear", "cosine", "warm-up + cosine", "inverse sqrt"],
        rows: [
          ["0", "0.10000", "0.10000", "0.10000", "0.10000", "0.00100", "0.00100"],
          ["50", "0.10000", "0.07943", "0.09500", "0.09938", "0.05100", "0.05100"],
          ["99", "0.10000", "0.06339", "0.09010", "0.09760", "0.10000", "0.10000"],
          ["250", "0.10000", "0.03162", "0.07500", "0.08536", "0.09330", "0.06312"],
          ["500", "0.01000", "0.01000", "0.05000", "0.05000", "0.05868", "0.04468"],
          ["750", "0.00100", "0.00316", "0.02500", "0.01464", "0.01786", "0.03649"],
          ["999", "0.00010", "0.00100", "0.00010", "0.00000", "0.00000", "0.03162"],
        ],
      },
      {
        t: "p",
        text: String.raw`Check a few by hand. Cosine at $t = 250$: $\tfrac12(0.1)(1 + \cos(\pi/4)) = 0.05 \times 1.70711 = 0.085355$. Warm-up + cosine at $t = 250$: the cosine part has run $250 - 100 = 150$ of its $900$ steps, so $\eta = 0.05\,(1 + \cos(\pi \cdot 150/900)) = 0.05\,(1 + \cos 30°) = 0.05 \times 1.86603 = 0.093301$. Inverse sqrt at $t = 999$: $0.1\sqrt{100/1000} = 0.031623$.`,
      },

      { t: "h", text: "3. Warm-up: what it fixes" },
      {
        t: "list",
        items: [
          String.raw`**Adam's statistics start empty.** At $t = 1$ the bias-corrected second moment is just $g_1^2$, so Adam's first step is $\pm\eta$ in **every** coordinate, however unreliable the gradient. For the next tens of steps $\hat v$ averages only a handful of samples and jumps around. Liu et al. ("On the Variance of the Adaptive Learning Rate and Beyond", ICLR 2020, the RAdam paper) named this a main reason Adam needs warm-up: a small $\eta$ early keeps these shaky updates small.`,
          String.raw`**High curvature at the start.** The stability rule $\eta < 2/\lambda_{\max}$ uses the curvature right now. Early on, curvature and gradient sizes can be much larger than later, so a peak rate that is fine after a few hundred steps can blow up on step 1. Warm-up lets the network first move somewhere the peak rate is safe.`,
          String.raw`**Large-batch training.** The linear scaling rule (next section) makes the peak rate large. Goyal et al. (2017) found that using it from the first step hurt accuracy, so they warmed up gradually from $\eta$ to $k\eta$ over 5 epochs; with that, ResNet-50 with batch 8192 matched the accuracy of batch 256.`,
          "**Post-LayerNorm Transformers.** Xiong et al. (2020, \"On Layer Normalization in the Transformer Architecture\") showed that in the original Post-LN Transformer, gradients near the output are large at the start, so warm-up is essential, while the Pre-LN layout trains reasonably without it.",
        ],
      },
      {
        t: "p",
        text: String.raw`In our own run (framework section), an 8-layer MLP without normalisation, trained with Adam at $\eta = 10^{-2}$ and cosine decay, reached 82.7% mean test accuracy over 5 seeds. Adding a 10% linear warm-up raised it to 90.0%, with the same peak and decay.`,
      },

      { t: "h", text: "4. The linear scaling rule for batch size" },
      {
        t: "p",
        text: String.raw`Say $k$ small-batch SGD steps use gradients $g_1, \dots, g_k$ at nearly the same weights. Together they move the weights by $-\eta\sum_j g_j$. One step with a $k$ times bigger batch uses the average $\bar g = \tfrac1k\sum_j g_j$ and moves by $-\eta'\bar g$. For the two to match:`,
      },
      {
        t: "math",
        tex: String.raw`\eta' = k\,\eta \qquad\text{e.g. Goyal et al.: } \eta = 0.1 \times \frac{B}{256} \;\Rightarrow\; \eta = 3.2 \text{ for } B = 8192`,
      },
      {
        t: "p",
        text: String.raw`"Nearly the same weights" fails when weights change fast — at the start of training (hence warm-up). The rule also stops working past a batch size that depends on the problem, where noise no longer limits progress and the bigger rate just becomes unstable. For Adam-type optimizers, whose step doesn't grow with gradient size, a square-root rule $\eta' = \sqrt k\,\eta$ is a common rule of thumb. Treat both as starting points for tuning, not laws.`,
      },

      { t: "h", text: "5. Gradient clipping" },
      {
        t: "p",
        text: String.raw`Let $g$ be all parameter gradients joined into one vector and $c > 0$ a threshold.`,
      },
      {
        t: "math",
        tex: String.raw`\text{by norm: } \; g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g\rVert_2}\right), \qquad \text{by value: } \; g_i \leftarrow \operatorname{clip}(g_i, -c, c) = \max(-c, \min(c, g_i))`,
      },
      {
        t: "p",
        text: String.raw`**By norm** (Pascanu, Mikolov & Bengio, 2013, for recurrent networks) shrinks the whole vector by one factor. The direction stays the same and the update length is capped at $\eta c$ for SGD ($\eta c/(1-\beta)$ once momentum settles). The norm is measured **globally**, over all parameters at once, as PyTorch's clip_grad_norm_ does. **By value** clamps each component on its own, which changes the direction.`,
      },
      {
        t: "p",
        text: String.raw`Example: $g = (3, 4, 12)$, $\lVert g\rVert = 13$, $c = 5$. By norm: $g \cdot 5/13 = (1.1538, 1.5385, 4.6154)$, norm exactly 5, same direction (cosine similarity 1). By value: $(3, 4, 5)$, norm 7.07, and the cosine similarity with the original falls to 0.9247 — the largest component was cut much more than the rest.`,
      },
      {
        t: "list",
        items: [
          "**When to use it.** Recurrent networks (gradients exploding through time), Transformers and LLMs (clipping at a global norm of 1.0 is almost universal), reinforcement learning, and any run with occasional loss spikes. It is cheap insurance: if the norm is usually below $c$, it changes nothing.",
          "**How to pick c.** Log the global gradient norm for a few hundred steps and set $c$ a little above its usual value, so only outliers get clipped. If almost every step is clipped, clipping has turned into a hidden learning-rate cut (every update has length η·c whatever the gradient).",
          "**With Adam** clipping matters less, because Adam already normalises each coordinate — but it still stops one huge gradient from distorting m and v. In our Adam run, clipping alone did not replace warm-up (85.5% vs 90.0%).",
        ],
      },
      {
        t: "code",
        lang: "python",
        title: "noise_floor_and_schedule_values.py — every number in sections 1, 2 and 5",
        code: MATH_PY,
        output: MATH_OUT,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator draws the chosen schedules on one plot and trains a real model with each: logistic regression on 400 fixed examples with two badly scaled features (scales 1 and 3), 15% flipped labels (so gradient noise never goes away) and 3% outliers with 8× larger features (so a few mini-batches give very large gradients). The optimizer is SGD with momentum 0.9 and batch size 8, with the same mini-batches for every schedule. The lower plot shows training loss minus its minimum (found with Newton's method), log scale, smoothed over 10 steps.",
      },
      { t: "sim", id: "lr-schedule", caption: "Learning-rate schedules and their effect on a real noisy training run. Select schedules, press Play, and watch the gold marker move along the LR curves while the loss curves grow below." },
      {
        t: "list",
        items: [
          "**Constant vs cosine (defaults, η₀ = 0.05).** For the first third both runs look alike, because the cosine rate is still near the peak. Then the cosine run keeps improving as its rate shrinks, while the constant run jitters on its noise floor. Over the last 50 steps the excess loss is about 0.13 for constant and 0.0035 for cosine — almost 40× from the schedule alone.",
          "**Raise η₀ to 0.1.** The constant and cosine runs blow up within the first 60 steps (average excess loss about 0.86): the starting curvature (about 10, shown under the table) makes η = 0.1 too big for momentum SGD there. Warm-up + cosine averages 0.06 over those steps thanks to its tiny early rate. Once warm-up reaches the peak, outlier batches still cause spikes.",
          "**Now tick Clip gradient norm (c = 1).** The spikes shrink about 10×; the table counts the clipped steps (roughly 60–80 of 600). The constant run still ends on its noise floor (about 0.13): clipping limits single steps, it does not shrink the rate.",
          "**Step decay.** The loss drops like a staircase right after each decay point (steps 150, 300, 450 by default) — each drop is the noise cloud shrinking with the rate.",
          "**Cosine + restarts.** With T₀ = 100 the cycles last 100, 200 and 400 steps. Each restart pushes the loss up (from 0.035 at step 100 to about 0.2 twenty steps later), and the next cycle brings it back down. Set T = 700 so the run ends exactly at a cycle end; at the default T = 600 it stops mid-cycle, with a higher loss.",
          "**Inverse square root and one-cycle.** Inverse sqrt never decays to zero, so its final loss (about 0.014) sits between constant and cosine. One-cycle starts at η₀/25, climbs to the peak and anneals to almost zero; it ends among the lowest losses, and at η₀ = 0.1 it is the lowest of all eight.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "All in NumPy: one function that builds every schedule (including SGDR restarts and a PyTorch-identical one-cycle), a two-layer MLP with SGD + momentum and global-norm clipping, an LR range test, and the real experiment — handwritten digits (8×8 images, 1257 training and 540 test), once with clean labels and once with 30% of the training labels swapped for random ones. Every schedule uses the same peak (0.2), 30 epochs and 5 seeds.",
      },
      {
        t: "code",
        lang: "python",
        title: "schedules_from_scratch.py",
        code: SCRATCH_PY,
        output: SCRATCH_OUT,
      },
      { t: "h", text: "What the runs show" },
      {
        t: "list",
        items: [
          String.raw`**LR range test.** The smoothed loss is flat up to about $10^{-2}$, falls steeply between 0.03 and 0.4, bottoms out at 0.343 and explodes by 3.5. The peak used in the experiment, 0.2, is in the steep part; the "tenth of the minimum" rule (0.034) would be a cautious choice.`,
          "**Clean labels: the schedule barely matters.** All five schedules give 97.6–97.8% mean test accuracy, within the seed-to-seed spread. The network fits the clean training set almost exactly, so gradient noise vanishes near the solution and a constant rate has no noise floor to fight (the note in Math section 1).",
          "**Noisy labels: decay is essential.** With 30% random labels the gradient noise never goes away. The constant rate ends at 81.2% (test NLL 0.736) with a wide spread across seeds (75.7–86.7%) — the final weights are a random point in the noise cloud. Every decaying schedule lands at 91.7–92.9% with a much tighter spread; one-cycle and warm-up + cosine are best, but by little compared with the gap to constant.",
          "**Clipping.** A peak of 0.6 with cosine decay is unstable: 72% mean accuracy, one seed at 58%. Clipping the global norm at 2.0, 1.0 and 0.5 gives 90.2%, 96.6% and 97.2%. Clipping caps each step at η·c (divided by 1 − β with momentum), which removes the rare disastrous step without touching the many normal ones.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "`torch.optim.lr_scheduler` has every schedule in this lesson. The program checks each against its formula over 1000 steps, reproduces the most common scheduler bug, and runs a real experiment: an 8-layer MLP on digits trained with Adam (η = 1e-2) and cosine decay, with and without a linear warm-up built from `SequentialLR(LinearLR, CosineAnnealingLR)`, and with `clip_grad_norm_` instead of warm-up.",
              },
              {
                t: "code",
                lang: "python",
                title: "torch_lr_schedulers.py",
                code: TORCH_PY,
                output: TORCH_OUT,
              },
              {
                t: "list",
                items: [
                  "Every PyTorch scheduler matches its formula to machine precision. `LambdaLR(opt, f)` multiplies the optimizer's starting lr by `f(step)`, the most flexible way to write warm-up + anything.",
                  "**The bug.** `CosineAnnealingLR(T_max=10)` means 10 epochs, but `scheduler.step()` is called every batch: the rate hits 0 after 10 batches and then **rises again** (PyTorch's cosine repeats past T_max), so it swings between 0.1 and 0 every 20 batches instead of decaying once. Fix: either `T_max = epochs * len(loader)` with per-batch stepping, or per-epoch stepping — and print the lr a few times to check.",
                  "**Warm-up result.** Same peak, decay and seeds: no warm-up 82.7%, 10% warm-up 90.0%, clipping at 1.0 without warm-up 85.5%. Warm-up fixes the start-up problem directly; clipping only partly stands in for it.",
                  "Call `optimizer.step()` before `scheduler.step()`; the other order skips the schedule's first value (PyTorch warns about it).",
                  "`OneCycleLR` must be stepped every batch with `total_steps = epochs * steps_per_epoch`; stepping past `total_steps` raises an error. It also moves momentum (or Adam's β₁) opposite to the learning rate unless `cycle_momentum=False`.",
                  "`ReduceLROnPlateau` is the odd one out: call `scheduler.step(val_loss)` once per epoch, with the watched metric.",
                ],
              },
              {
                t: "code",
                lang: "python",
                title: "training_loop_pattern.py (not executed in CI)",
                code: String.raw`import math
import torch

def warmup_cosine(step, warmup, total, floor=0.1):
    """Multiplier for LambdaLR: linear warm-up, then cosine to floor * peak."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

model = torch.nn.Linear(10, 1)
loader = [(torch.randn(32, 10), torch.randn(32, 1)) for _ in range(100)]
epochs = 5
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
total = epochs * len(loader)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warmup=total // 20, total=total))
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

for epoch in range(epochs):
    for x, y in loader:
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
            loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                          # clip the TRUE gradients, not the scaled ones
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()                                  # once per batch: total was counted in batches
    print(f"epoch {epoch}: lr={sched.get_last_lr()[0]:.2e}, last grad norm={norm:.3f}")`,
                output: "Prints, for each of the 5 epochs, the current learning rate (rising during the first 25 batches, then decaying along the cosine towards 3e-5) and the global gradient norm returned by clip_grad_norm_ for the last batch.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "code",
                lang: "python",
                title: "keras_schedules.py (not executed in CI)",
                code: String.raw`import math
import tensorflow as tf

steps_per_epoch, epochs = 40, 30
total = steps_per_epoch * epochs

# Built-in schedules are called with the optimizer's iteration counter (per BATCH)
step_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[total // 2, 3 * total // 4], values=[0.2, 0.02, 0.002])
exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(0.2, decay_steps=total, decay_rate=0.01)
cosine = tf.keras.optimizers.schedules.CosineDecay(0.2, decay_steps=total)
restarts = tf.keras.optimizers.schedules.CosineDecayRestarts(0.2, first_decay_steps=200, t_mul=2.0)
# Warm-up + cosine (TF >= 2.13): warmup_target is the peak reached after warmup_steps
warm_cos = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0, decay_steps=total - total // 20,
    warmup_target=0.2, warmup_steps=total // 20)

class InverseSqrt(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Vaswani et al. (2017): d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)."""
    def __init__(self, d_model=512, warmup=4000):
        self.d_model, self.warmup = float(d_model), float(warmup)
    def __call__(self, step):
        step = tf.cast(step, tf.float32) + 1.0
        return self.d_model ** -0.5 * tf.minimum(step ** -0.5, step * self.warmup ** -1.5)

for name, s in [("step", step_decay), ("exp", exp_decay), ("cosine", cosine),
                ("restarts", restarts), ("warmup+cos", warm_cos), ("inv-sqrt", InverseSqrt())]:
    print(name, [round(float(s(t)), 5) for t in (0, total // 4, total // 2, total - 1)])

# Clipping: global_clipnorm == torch clip_grad_norm_; clipnorm is per variable; clipvalue per element
opt = tf.keras.optimizers.SGD(learning_rate=cosine, momentum=0.9, global_clipnorm=1.0)
# Plateau-based decay is a callback, evaluated once per epoch:
plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10)`,
                output: "Prints, for each schedule, the learning rate at the start, a quarter, half way and the end of training (e.g. cosine: 0.2, about 0.171, about 0.1, about 0.0).",
              },
              {
                t: "list",
                items: [
                  "Keras schedules run on the optimizer's step counter, so they are per-batch by design — the epoch/batch mix-up is less likely than in PyTorch, but `decay_steps` must still be in batches.",
                  "`clipnorm` clips each variable separately, `global_clipnorm` clips the joint norm like PyTorch's `clip_grad_norm_`, and `clipvalue` clips each element.",
                ],
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      {
        t: "table",
        head: ["Schedule", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Constant", "simplest; no need to know the run length", "noise floor ∝ η; final model is a random point in the noise cloud", "quick experiments, models that fit the data exactly, as part of other schedules"],
          ["Step / multi-step", "simple, easy to read, strong results on CNNs", "milestones need tuning; sudden changes", "ResNet-style ImageNet/CIFAR training"],
          ["Exponential", "smooth, one parameter", "decays fastest early, when big steps still help", "older papers, RL, small models"],
          ["Linear / polynomial decay", "simple, reaches exactly 0 at T", "needs T in advance", "BERT-style fine-tuning (with warm-up)"],
          ["Cosine annealing", "few settings, long high-rate phase and gentle end, robust", "needs T in advance", "default for many vision and language models"],
          ["Cosine with warm restarts (SGDR)", "usable model at every cycle end, chances to escape poor valleys, snapshot ensembles", "loss spikes at restarts; cycle lengths to tune", "long runs of unknown length, ensembling"],
          ["Warm-up + cosine / linear", "stable start for Adam and large batches + settled end", "two extra settings (W, floor)", "Transformers, LLMs, ViTs: the standard"],
          ["Inverse square root", "no fixed T; theory-backed 1/√t decay", "never gets down to a small value", "original Transformer; continual or open-ended training"],
          ["One-cycle", "fast training at a high peak; often best for a fixed short budget", "budget must be known; extra momentum cycling", "fastai-style training, small-budget vision"],
          ["ReduceLROnPlateau", "adapts to the run; no T needed", "reactive and noisy; needs a validation metric; patience wastes epochs", "small projects without a known budget"],
          ["Warm-up–stable–decay (WSD)", "peak rate for most of training, short final decay; checkpoints can be decayed later", "decay length to tune", "recent LLM pre-training variants"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          String.raw`**ImageNet ResNets.** He et al. (2016) used SGD + momentum, $\eta = 0.1$ for batch 256, divided by 10 when error stalled — the classic step schedule. Goyal et al. (2017) scaled it to batch 8192 with $\eta = 3.2$ and 5 epochs of linear warm-up, training ResNet-50 in one hour.`,
          String.raw`**The original Transformer.** Adam with $\beta_2 = 0.98$ and the inverse-square-root schedule with 4000 warm-up steps (Vaswani et al., 2017).`,
          String.raw`**Large language models.** GPT-3 (Brown et al., 2020) and LLaMA (Touvron et al., 2023) report AdamW with linear warm-up, cosine decay to 10% of the peak, and gradient clipping at 1.0 — the template most open LLM recipes follow.`,
          String.raw`**Fine-tuning.** BERT-style fine-tuning uses AdamW with a small peak ($2\times10^{-5}$–$5\times10^{-5}$ in the BERT paper's search), a short warm-up and linear decay to zero over a few epochs.`,
          "**Recurrent networks and RL.** Gradient-norm clipping (Pascanu et al., 2013) is standard for LSTMs and GRUs; policy-gradient methods often clip the gradient norm (often at 0.5) and lower the learning rate linearly.",
          "**Fast training on a budget.** One-cycle (Smith, 2018) with an LR range test is the default in fastai's `fit_one_cycle`, widely used to train image classifiers in few epochs.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Knowing the budget buys performance.** Cosine, linear and one-cycle need T in advance; extending a finished cosine run means restarting the decay. Inverse-sqrt, plateau and warm-up–stable–decay suit open-ended training better.",
          "**Most decaying schedules perform alike.** In our noisy-label run, step, cosine, warm-up + cosine and one-cycle were within about 1.2 points of each other, all more than 10 points above constant. Picking *some* sensible decay matters far more than *which*.",
          String.raw`**The peak rate matters most.** A schedule only rescales the peak; a peak that is too high (0.6 in our run) or too low can't be saved by the shape. Find it first (range test, small sweep).`,
          "**Warm-up costs a little time and one more setting**, but prevents many early blow-ups, especially with Adam, large batches and deep un-normalised or Post-LN networks.",
          "**Clipping is insurance, not optimisation.** It costs one norm per step and does nothing when gradients are normal. If it fires all the time, it quietly changes the effective learning rate and hides the real problem (η too high, bad data, numerical issues).",
          "**It depends on the optimizer.** Momentum multiplies the effective step by 1/(1−β); Adam's step is already normalised. So the same schedule shape acts differently on SGD and Adam — re-tune the peak when you switch.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Stepping the scheduler at the wrong rate.** T_max or step_size given in epochs but `scheduler.step()` called every batch (or the reverse): the schedule runs 50–1000× too fast or slow. The PyTorch output above shows the rate swinging between 0.1 and 0 every 20 batches. Fix: count T in the unit you step in, and log the lr.",
          "**Calling `scheduler.step()` before `optimizer.step()`.** The schedule's first value is skipped (PyTorch warns). Fix: optimizer first, scheduler second.",
          "**Forgetting scheduler state when resuming.** Restarting from a checkpoint without `scheduler.load_state_dict(...)` restarts warm-up and the decay from the peak. Save `scheduler.state_dict()` with the model and optimizer.",
          "**Running a cosine schedule past T_max.** PyTorch's `CosineAnnealingLR` repeats: after T_max the rate climbs back toward the peak. If you extend training, rebuild the schedule.",
          "**Clipping per tensor when you meant the global norm.** Clipping each tensor separately changes the update direction (cosine similarity 0.972 in the exercise) and over-weights small tensors like biases. Use `clip_grad_norm_` over all parameters (or Keras's `global_clipnorm`).",
          "**Clipping scaled gradients in mixed precision.** `GradScaler` multiplies gradients by a big factor; clipping before `scaler.unscale_(optimizer)` uses the wrong threshold. Fix: unscale, then clip, then `scaler.step`.",
          "**Using the linear scaling rule without warm-up**, or far past the batch sizes where it holds. Fix: warm up over a few epochs and check that accuracy matches the small-batch baseline.",
          "**Tuning the schedule before the peak rate.** The shape is a smaller effect. Fix: LR range test or a rough sweep of the peak first.",
          "**Expecting a schedule to help a model that fits clean data exactly.** As our clean-label run shows, the gain can be zero; it appears when gradient noise stays (noisy labels, large datasets, data augmentation, regularisation).",
          "**Decaying to exactly zero when you might train longer.** A zero rate wastes the last steps of an extended run; a floor of a few percent of the peak (LLaMA-style 10%) keeps the model trainable.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "Why do we decrease the learning rate during training?", a: String.raw`Because each SGD step uses a noisy gradient. With a constant rate the weights only reach a cloud around the minimum, with expected extra loss about $\eta\sigma^2/4$ (for a quadratic with gradient-noise variance $\sigma^2$). A large rate gets close fast but leaves a big cloud; a small rate gives a small cloud but takes far longer. Starting large and decaying gets both: fast early progress and a small final error.` },
      { t: "qa", level: "beginner", q: "What is learning-rate warm-up?", a: String.raw`Starting with a small learning rate and raising it (usually linearly) to the peak over the first few hundred or thousand steps, before the decay phase. It prevents early blow-ups when the optimizer's statistics (Adam's second moment) are still poor, when the network starts in a high-curvature region, or when the peak is large because of a large batch.` },
      { t: "qa", level: "beginner", q: "What does gradient clipping by norm do, and how is it different from clipping by value?", a: String.raw`Clipping by norm measures the size of the whole gradient vector and, if it is above a threshold $c$, rescales it to size exactly $c$; the direction stays the same. Clipping by value clamps each component to $[-c, c]$ on its own, which can change the direction. For $g = (3, 4, 12)$ and $c = 5$: by norm gives $(1.15, 1.54, 4.62)$, by value gives $(3, 4, 5)$.` },
      { t: "qa", level: "intermediate", q: "Write the cosine annealing formula and explain its shape.", a: String.raw`$\eta_t = \eta_{\min} + \tfrac12(\eta_0 - \eta_{\min})(1 + \cos(\pi t/T))$. It equals $\eta_0$ at $t = 0$, the midpoint at $t = T/2$ and $\eta_{\min}$ at $t = T$. Its slope is proportional to $\sin(\pi t/T)$, zero at both ends: the rate stays near the peak for a long time (exploring), falls fastest in the middle, and lingers near the floor (fine settling). SGDR restarts it with cycle lengths $T_0, T_0T_{\text{mult}}, \dots$` },
      { t: "qa", level: "intermediate", q: "State the linear scaling rule and its limitations.", a: String.raw`When the batch size is multiplied by $k$, multiply the learning rate by $k$ (Goyal et al., 2017). Reason: $k$ small steps $-\eta\sum_j g_j$ equal one big step $-k\eta\,\bar g$ if the gradients barely change over those steps. It fails early in training, when weights change fast (hence gradual warm-up), and past a problem-dependent batch size where the bigger rate is just unstable. For Adam a square-root rule is a common rule of thumb. Either way, check it.` },
      { t: "qa", level: "intermediate", q: "Give the original Transformer learning-rate schedule and compute its peak for d_model = 512.", a: String.raw`$\eta_t = d_{\text{model}}^{-1/2}\min(t^{-1/2}, t\,W^{-3/2})$ with $W = 4000$. It rises linearly for $t \le W$ and decays like $1/\sqrt t$ after. The peak is at $t = W$: $\eta = (d_{\text{model}} W)^{-1/2} = (512 \times 4000)^{-1/2} = 6.99\times10^{-4}$. At step 16 000 it is half that, $3.49\times10^{-4}$.` },
      { t: "qa", level: "intermediate", q: "How do you choose the peak learning rate with an LR range test?", a: "Train for about 100 mini-batches while growing the learning rate exponentially from tiny (1e-6 to 1e-4) to large (1–10), record a smoothed loss, and stop when the loss explodes. Plot loss against log learning rate: flat at tiny rates, steep drop in the middle, a bottom, then a blow-up. Pick the peak in the steep part — often where it drops fastest, or about a tenth of the rate at the minimum; for one-cycle, the rate near the minimum is often used as the maximum." },
      { t: "qa", level: "advanced", q: "Why does Adam in particular benefit from warm-up, even with bias correction?", a: String.raw`Bias correction fixes the average of $v_t$ but not how much it jumps around. At $t = 1$, $\hat v_1 = g_1^2$, so the first update is exactly $\pm\eta$ in every coordinate, whether or not $g_1$ carries signal. Over the next tens of steps $\hat v$ averages only a few samples, so each coordinate's effective rate $\eta/\sqrt{\hat v}$ swings a lot — Liu et al. (2020) studied exactly this and proposed RAdam to fix it. Adam's step doesn't shrink with gradient size, so only η limits these noisy early moves; warm-up keeps η small until $\hat v$ is reliable. In our 8-layer MLP, adding 10% warm-up to Adam at η = 1e-2 raised mean test accuracy from 82.7% to 90.0%.` },
      { t: "qa", level: "advanced", q: "A cosine schedule was configured with T_max = 100 (epochs), but the loss oscillates periodically and never settles. What happened and how do you fix it?", a: "The scheduler was stepped every batch instead of every epoch, so the cosine finished its decay in 100 batches. PyTorch's CosineAnnealingLR repeats past T_max, so the rate climbed back to the peak and kept cycling every 200 batches; each climb kicks the model out of the valley it was settling into. Fix: set T_max = epochs × batches_per_epoch when stepping per batch (or step once per epoch), log `scheduler.get_last_lr()` regularly, and save and restore the scheduler state when resuming." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Warm-up + cosine: peak 0.1, T = 1000, W = 100 (cosine over the remaining 900 steps to 0). What is the learning rate at t = 300? (5 decimals)", answer: 0.0883, tolerance: 0.0001, explain: "Progress (300 − 100)/900 = 0.2222; 0.05·(1 + cos(0.2222π)) = 0.05·(1 + 0.76604) = 0.08830." },
          { kind: "numeric", q: "Goyal et al.'s linear scaling rule uses η = 0.1 × B/256. What is η for B = 8192?", answer: 3.2, tolerance: 0.001, explain: "0.1 × 8192/256 = 0.1 × 32 = 3.2." },
          { kind: "numeric", q: "Peak of the Transformer schedule for d_model = 1024 and W = 4000, (d_model·W)^(−1/2)? (give it ×10⁴, i.e. 4.94 for 4.94e-4)", answer: 4.941, tolerance: 0.01, explain: "(1024 × 4000)^(−1/2) = 1/2023.9 = 4.941e-4." },
          { kind: "numeric", q: "Constant-η SGD on L = ½x² (λ = 1) with gradient noise σ² = 1. Using E[L]∞ = ησ²/(2(2 − ηλ)), what is the noise floor for η = 0.2?", answer: 0.05556, tolerance: 0.0002, explain: "0.2/(2 × 1.8) = 0.2/3.6 = 0.05556." },
          { kind: "numeric", q: "Clip g = (6, 8) by norm with threshold c = 5. What is the first component after clipping?", answer: 3, tolerance: 0.001, explain: "‖g‖ = 10, scale 5/10 = 0.5, so g → (3, 4)." },
          { q: "A training script sets CosineAnnealingLR(opt, T_max=epochs) and calls scheduler.step() after every batch. What happens?", options: ["The schedule is exactly right", "The learning rate decays to the minimum after `epochs` batches and then oscillates back up periodically", "PyTorch raises an error", "The learning rate stays constant"], answer: 1, explain: "T_max counts calls to step(); PyTorch's cosine repeats past T_max, so the rate cycles every 2·T_max batches." },
          { q: "In our digits experiment, why did the schedule matter with 30% noisy labels but not with clean labels?", options: ["Noisy labels make the network smaller", "With clean labels the network fits the data and gradient noise vanishes near the solution; noisy labels keep gradient noise alive, so a constant rate stays on a noise floor", "Cosine schedules only work with noisy data", "Clean labels require a larger learning rate"], answer: 1, explain: "The noise floor ησ²/4 needs σ² > 0 at the solution; a model that fits clean data exactly has σ² ≈ 0 there." },
          { q: "Which statement about gradient clipping by global norm is correct?", options: ["It changes the direction of the update", "It caps the update length while keeping its direction", "It clamps every component to [−c, c]", "It must be applied after optimizer.step()"], answer: 1, explain: "Every gradient is multiplied by the same factor min(1, c/‖g‖), so the direction is kept; it must run between backward() and step()." },
        ],
      },
      {
        t: "exercise",
        title: "Numerical: design a warm-up + cosine schedule",
        prompt: "You train for 20 epochs of 500 batches with AdamW, peak η = 3e-4, 5% linear warm-up (multiplier (t+1)/W) and cosine decay to a floor of 10% of the peak over the remaining steps. (a) How many warm-up steps W and decay steps are there? (b) What is η at t = 0, t = 249 and t = W − 1? (c) What is η halfway through the decay and at the last step t = 9999? (d) If the batch size doubles (250 batches per epoch, same epochs), what peak does the square-root rule for Adam suggest, and how many warm-up steps keep warm-up at 5%?",
        hint: "Cosine with a floor: η = η_min + ½(η₀ − η_min)(1 + cos(π·progress)), with η_min = 3e-5.",
        solution: [
          { t: "p", text: String.raw`(a) $T = 20 \times 500 = 10\,000$ steps; $W = 0.05\,T = 500$; decay over $9\,500$ steps.` },
          { t: "p", text: String.raw`(b) $t = 0$: $3\times10^{-4}\times 1/500 = 6\times10^{-7}$. $t = 249$: $3\times10^{-4}\times 250/500 = 1.5\times10^{-4}$. $t = W-1 = 499$: $3\times10^{-4}$ (the peak).` },
          { t: "p", text: String.raw`(c) Halfway (progress 0.5, $t = 500 + 4750 = 5250$): $\cos(\pi/2) = 0$, so $\eta = 3\times10^{-5} + \tfrac12(2.7\times10^{-4}) = 1.65\times10^{-4}$. Last step $t = 9999$: progress $9499/9500 = 0.999895$, $\cos(0.999895\pi) = -0.99999995$, so $\eta \approx 3\times10^{-5} + 1.35\times10^{-4}\times 5\times10^{-8} \approx 3.0\times10^{-5}$, the floor.` },
          { t: "p", text: String.raw`(d) Batch ×2 → square-root rule: $\eta' = \sqrt2 \times 3\times10^{-4} = 4.24\times10^{-4}$. The run now has $20 \times 250 = 5000$ steps, so 5% warm-up is 250 steps. (A linear rule would suggest $6\times10^{-4}$; check either one.)` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: implement global-norm clipping and compare with PyTorch",
        prompt: "Implement clipping by global norm for a list of NumPy gradient arrays (one scaling factor for all), check it against torch.nn.utils.clip_grad_norm_ (which adds 1e-6 to the norm), and compare it with the common mistake of clipping each tensor separately: report the cosine similarity of each clipped update with the raw gradient and what happens to a small bias gradient.",
        hint: "Flatten and join all arrays to measure the joint norm and the angle.",
        solution: [
          { t: "code", lang: "python", title: "clip_global_norm.py", code: EX_PY, output: EX_OUT },
          { t: "p", text: String.raw`Global clipping matches PyTorch exactly and keeps the direction (cosine 1.0000). Per-tensor clipping leaves the small bias gradient untouched (0.2647) while shrinking the weight gradient 12-fold, so the bias is now over-weighted and the update direction turns by $\arccos 0.972 \approx 13.6°$. With global clipping the bias is scaled by the same factor as everything else (0.0217).` },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`Constant-η SGD settles on a noise floor $\mathbb{E}[L]_\infty = \eta\sigma^2/(2(2-\eta\lambda)) \approx \eta\sigma^2/4$; time to reach it $\propto 1/(\eta\lambda)$. Decay resolves the trade-off.`,
          String.raw`Step: $\eta_0\gamma^{\lfloor t/s\rfloor}$. Exponential: $\eta_0\gamma^t$. Polynomial: $(\eta_0-\eta_{\min})(1-t/T)^p + \eta_{\min}$. Cosine: $\eta_{\min} + \tfrac12(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$; SGDR restarts it with periods $T_0 T_{\text{mult}}^i$.`,
          String.raw`Linear warm-up $\eta_0 (t+1)/W$; Transformer schedule $d_{\text{model}}^{-1/2}\min(t^{-1/2}, tW^{-3/2})$ with peak $(d_{\text{model}}W)^{-1/2}$; one-cycle: up to max_lr over 30%, then anneal to max_lr/(2.5·10⁵).`,
          "Warm-up protects against unreliable early Adam statistics, high starting curvature, large-batch peak rates and Post-LN Transformers.",
          String.raw`Linear scaling rule: batch ×k → η ×k (with warm-up; breaks at very large batches); the square-root rule is a common Adam rule of thumb.`,
          String.raw`Clip by global norm: $g \leftarrow g\min(1, c/\lVert g\rVert)$, direction kept; by value: clamp each component, direction changes. Clip after backward (and after unscaling) and before step.`,
          "PyTorch: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR, LambdaLR, LinearLR, SequentialLR, ReduceLROnPlateau(step(metric)). Step the scheduler in the same unit as its length, after optimizer.step().",
          "Measured: schedules barely matter on clean data the model can fit; with noisy labels decay gave +10–12 points over constant; warm-up gave +7 points for deep Adam; clipping made an unstable peak rate stable.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Weight Initialization](/learn/m07/initialization) — the starting gradient size is what warm-up and clipping guard against; good initialisation reduces the need for both.",
          "[Batch, Layer and Group Normalization](/learn/m07/normalization) — normalisation changes the effective learning rate of the weights before it and makes higher peaks stable; Pre-LN vs Post-LN decides how much warm-up a Transformer needs.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — weight decay, whose per-step strength ηλ follows the schedule, and early stopping, which interacts with when the rate decays.",
          "[Residual and Skip Connections](/learn/m07/residual-connections) — another way to keep gradients well-sized in deep networks, so clipping is a safety net rather than a must.",
        ],
      },
    ],
  },
};

export default lesson;
