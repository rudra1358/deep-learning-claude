import type { Lesson } from "@/lib/types";

const THEORY_PY = String.raw`import numpy as np

# ---------------------------------------------------------------
# 1) Exponential moving average m_t = beta*m_{t-1} + (1-beta)*g_t
# ---------------------------------------------------------------
print("EMA facts")
k = np.arange(100000)                              # age of a gradient (0 = newest)
for beta in [0.9, 0.99, 0.999]:
    w = (1 - beta) * beta ** k                     # weight the EMA gives to a gradient of age k
    print(f"  beta={beta}: weights sum to {w.sum():.4f}, mean age {np.sum(k * w):7.1f}, "
          f"weight left after 1/(1-beta) steps {beta ** (1 / (1 - beta)):.4f}, "
          f"noise variance x{(1 - beta) / (1 + beta):.5f}, steps until bias < 1%: {np.log(0.01) / np.log(beta):.0f}")

# Variance reduction measured on white noise (sigma = 1)
rng = np.random.default_rng(0)
g = rng.normal(size=100000)
m, ms = 0.0, []
for x in g:
    m = 0.9 * m + 0.1 * x
    ms.append(m)
print(f"  measured variance of EMA(0.9) of unit noise: {np.var(ms[1000:]):.4f}  (theory 0.1/1.9 = {0.1 / 1.9:.4f})")

# Bias correction = renormalising the weights of a zero-started EMA
beta, gs = 0.9, np.array([2.0, 4.0, 3.0])
m = 0.0
for t, gt in enumerate(gs, 1):
    m = beta * m + (1 - beta) * gt
    wts = (1 - beta) * beta ** (t - np.arange(1, t + 1))  # weights on g_1..g_t
    print(f"  t={t}: m={m:.4f}  sum of weights={wts.sum():.4f}  m/(1-beta^t)={m / (1 - beta ** t):.4f}  "
          f"weighted mean={np.sum(wts * gs[:t]) / wts.sum():.4f}")

# Momentum as a filter: gain of v_t = beta*v_{t-1} + g_t for a gradient oscillating at frequency w
print("Frequency response of the momentum sum (beta = 0.9)")
for name, om in [("constant (w=0)", 0.0), ("period 4 (w=pi/2)", np.pi / 2), ("alternating (w=pi)", np.pi)]:
    print(f"  {name:20s} gain {1 / abs(1 - 0.9 * np.exp(-1j * om)):.4f}")

# ---------------------------------------------------------------
# 2) Heavy ball on a quadratic: per-eigendirection 2x2 linear system
# ---------------------------------------------------------------
def rate(eta, beta, lam, nesterov=False):
    """Spectral radius of the iteration matrix for one curvature lam."""
    a = 1 - eta * lam
    M = (np.array([[(1 + beta) * a, -beta * a], [1, 0]]) if nesterov
         else np.array([[1 + beta - eta * lam, -beta], [1, 0]]))
    return max(abs(np.linalg.eigvals(M)))

mu, L = 1.0, 100.0                                 # smallest / largest curvature, kappa = 100
kap = L / mu
eta_gd = 2 / (mu + L)
eta_hb = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
beta_hb = ((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)) ** 2
print(f"kappa = {kap:.0f}")
print(f"  GD   eta*={eta_gd:.5f}            rate={max(rate(eta_gd, 0, l) for l in (mu, L)):.5f}  "
      f"steps for 1e-6: {np.log(1e-6) / np.log((kap - 1) / (kap + 1)):.0f}")
print(f"  HB   eta*={eta_hb:.5f} beta*={beta_hb:.5f} rate={max(rate(eta_hb, beta_hb, l) for l in (mu, L)):.5f}  "
      f"steps for 1e-6: {np.log(1e-6) / np.log(np.sqrt(beta_hb)):.0f}")

# Brute-force check that no other (eta, beta) does better
best = min((max(rate(e, b, l) for l in (mu, L)), e, b)
           for e in np.linspace(0.005, 0.05, 91) for b in np.linspace(0.3, 0.95, 131))
print(f"  grid search best: rate={best[0]:.5f} at eta={best[1]:.4f}, beta={best[2]:.3f}")

# Every curvature inside [(1-sqrt b)^2, (1+sqrt b)^2]/eta converges at exactly sqrt(beta)
print("  beta=0.9, rate for eta*lam =", [f"{a}:{rate(a, 0.9, 1.0):.4f}" for a in (0.001, 0.01, 0.5, 3.0, 3.9)])

# Stability limits on eta*lam: heavy ball 2(1+b), Nesterov 2(1+b)/(1+2b)
a_grid = np.linspace(0.001, 5, 5000)
for b in (0.0, 0.5, 0.9):
    hb = a_grid[[rate(a, b, 1.0) < 1 for a in a_grid]].max()
    nag = a_grid[[rate(a, b, 1.0, True) < 1 for a in a_grid]].max()
    print(f"  beta={b}: largest stable eta*lam  HB {hb:.3f} (2(1+b)={2 * (1 + b):.3f})  "
          f"NAG {nag:.3f} (2(1+b)/(1+2b)={2 * (1 + b) / (1 + 2 * b):.3f})")`;
const THEORY_OUT = String.raw`EMA facts
  beta=0.9: weights sum to 1.0000, mean age     9.0, weight left after 1/(1-beta) steps 0.3487, noise variance x0.05263, steps until bias < 1%: 44
  beta=0.99: weights sum to 1.0000, mean age    99.0, weight left after 1/(1-beta) steps 0.3660, noise variance x0.00503, steps until bias < 1%: 458
  beta=0.999: weights sum to 1.0000, mean age   999.0, weight left after 1/(1-beta) steps 0.3677, noise variance x0.00050, steps until bias < 1%: 4603
  measured variance of EMA(0.9) of unit noise: 0.0525  (theory 0.1/1.9 = 0.0526)
  t=1: m=0.2000  sum of weights=0.1000  m/(1-beta^t)=2.0000  weighted mean=2.0000
  t=2: m=0.5800  sum of weights=0.1900  m/(1-beta^t)=3.0526  weighted mean=3.0526
  t=3: m=0.8220  sum of weights=0.2710  m/(1-beta^t)=3.0332  weighted mean=3.0332
Frequency response of the momentum sum (beta = 0.9)
  constant (w=0)       gain 10.0000
  period 4 (w=pi/2)    gain 0.7433
  alternating (w=pi)   gain 0.5263
kappa = 100
  GD   eta*=0.01980            rate=0.98020  steps for 1e-6: 691
  HB   eta*=0.03306 beta*=0.66942 rate=0.81818  steps for 1e-6: 69
  grid search best: rate=0.81854 at eta=0.0330, beta=0.670
  beta=0.9, rate for eta*lam = ['0.001:0.9889', '0.01:0.9487', '0.5:0.9487', '3.0:0.9487', '3.9:1.3162']
  beta=0.0: largest stable eta*lam  HB 1.999 (2(1+b)=2.000)  NAG 1.999 (2(1+b)/(1+2b)=2.000)
  beta=0.5: largest stable eta*lam  HB 2.999 (2(1+b)=3.000)  NAG 1.499 (2(1+b)/(1+2b)=1.500)
  beta=0.9: largest stable eta*lam  HB 3.800 (2(1+b)=3.800)  NAG 1.357 (2(1+b)/(1+2b)=1.357)`;
const ADAM_PY = String.raw`import numpy as np

def adam_ratios(gs, b1=0.9, b2=0.999, eps=1e-8, amsgrad=False):
    """Return the normalised step  m_hat / (sqrt(v_hat) + eps)  for a gradient sequence."""
    m = v = vmax = 0.0
    out = []
    for t, g in enumerate(gs, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        vmax = max(vmax, v)
        vv = vmax if amsgrad else v
        out.append((m / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + eps))
    return np.array(out)

# 1) The step tracks the signal-to-noise ratio: g ~ N(mu, 1)
rng = np.random.default_rng(0)
print("Noisy gradient g ~ N(mu, 1): average Adam step in units of eta")
for mu in [10.0, 1.0, 0.1, 0.0]:
    r = adam_ratios(mu + rng.normal(size=20000))[2000:]      # skip the start-up phase
    print(f"  mu={mu:4.1f}: mean step {r.mean():+.4f}  (mu/sqrt(mu^2+1) = {mu / np.sqrt(mu * mu + 1):.4f}),"
          f" mean |step| {np.abs(r).mean():.4f}")

# 2) Steps can exceed eta: a gradient burst after a quiet period
burst = adam_ratios(np.concatenate([np.full(2000, 1e-3), [1.0]]))[-1]
b1, b2 = 0.9, 0.999
print(f"Burst after 2000 quiet steps: step = {burst:.4f} eta;  worst-case bound "
      f"(1-b1)/sqrt((1-b2)(1-b1^2/b2)) = {(1 - b1) / np.sqrt((1 - b2) * (1 - b1 ** 2 / b2)):.4f}")

# 3) The role of epsilon: constant gradient of size g, step after 100 iterations
print("Effect of eps on the step (constant gradient):")
for g in [1e-2, 1e-4, 1e-6, 1e-8]:
    row = [f"eps={eps:g}: {adam_ratios(np.full(100, g), eps=eps)[-1]:.4f}" for eps in (1e-8, 1e-6, 1e-3)]
    print(f"  g={g:g}:  " + "   ".join(row))

# 4) Reddi et al. (2018) counterexample: f_t(x) = C*x if t % 3 == 1 else -x, x in [-1, 1]
#    The average loss is (C-2)/3 * x, minimised at x = -1.
C = 3.0
def online(amsgrad, b2, T=30000, alpha=0.5):
    x = 1.0; m = v = vmax = 0.0
    for t in range(1, T + 1):
        g = C if t % 3 == 1 else -1.0
        m, v = g, b2 * v + (1 - b2) * g * g          # beta1 = 0
        vmax = max(vmax, v)
        x -= alpha / np.sqrt(t) * m / (np.sqrt((vmax if amsgrad else v) / (1 - b2 ** t)) + 1e-8)
        x = min(1.0, max(-1.0, x))                    # project back onto [-1, 1]
    return x
for b2_ in [1 / (1 + C * C), 0.999]:
    print(f"beta2={b2_:.3f}: Adam ends at x={online(False, b2_):+.3f}, AMSGrad ends at x={online(True, b2_):+.3f}  (optimum -1)")

# Why: with beta2 = 0.1 the normalised steps over one period of 3
v, steps = 0.0, []
for t in range(1, 7):
    g = C if t % 3 == 1 else -1.0
    v = 0.1 * v + 0.9 * g * g
    steps.append(g / np.sqrt(v))
print("normalised gradients, period 2:", np.round(steps[3:], 4), "sum", round(sum(steps[3:]), 4))`;
const ADAM_OUT = String.raw`Noisy gradient g ~ N(mu, 1): average Adam step in units of eta
  mu=10.0: mean step +0.9951  (mu/sqrt(mu^2+1) = 0.9950), mean |step| 0.9951
  mu= 1.0: mean step +0.7043  (mu/sqrt(mu^2+1) = 0.7071), mean |step| 0.7043
  mu= 0.1: mean step +0.0963  (mu/sqrt(mu^2+1) = 0.0995), mean |step| 0.1986
  mu= 0.0: mean step -0.0042  (mu/sqrt(mu^2+1) = 0.0000), mean |step| 0.1851
Burst after 2000 quiet steps: step = 2.9662 eta;  worst-case bound (1-b1)/sqrt((1-b2)(1-b1^2/b2)) = 7.2703
Effect of eps on the step (constant gradient):
  g=0.01:  eps=1e-08: 1.0000   eps=1e-06: 0.9999   eps=0.001: 0.9091
  g=0.0001:  eps=1e-08: 0.9999   eps=1e-06: 0.9901   eps=0.001: 0.0909
  g=1e-06:  eps=1e-08: 0.9901   eps=1e-06: 0.5000   eps=0.001: 0.0010
  g=1e-08:  eps=1e-08: 0.5000   eps=1e-06: 0.0099   eps=0.001: 0.0000
beta2=0.100: Adam ends at x=+1.000, AMSGrad ends at x=-0.998  (optimum -1)
beta2=0.999: Adam ends at x=-0.997, AMSGrad ends at x=-0.997  (optimum -1)
normalised gradients, period 2: [ 1.0472 -0.7623 -0.9658] sum -0.6809`;
const SCRATCH_PY = String.raw`import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")                   # diverging learning rates in the grid overflow; we skip them


class Optimizer:
    """All methods of this lesson, PyTorch conventions. kind in
    sgd | heavyball | nesterov | adagrad | rmsprop | adam | amsgrad | adamw"""

    def __init__(self, kind, lr, beta=0.9, rho=0.9, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.kind, self.lr, self.beta, self.rho = kind, lr, beta, rho
        self.b1, self.b2, self.eps, self.wd = b1, b2, eps, wd
        self.t, self.buf, self.m, self.s, self.smax = 0, 0.0, 0.0, 0.0, 0.0

    def step(self, p, g):
        self.t += 1
        k, lr = self.kind, self.lr
        if k == "sgd":
            return p - lr * g
        if k in ("heavyball", "nesterov"):
            self.buf = self.beta * self.buf + g                     # velocity
            return p - lr * (self.buf if k == "heavyball" else g + self.beta * self.buf)
        if k == "adagrad":
            self.s = self.s + g * g                                 # sum of squares: only grows
            return p - lr * g / (np.sqrt(self.s) + self.eps)
        if k == "rmsprop":
            self.s = self.rho * self.s + (1 - self.rho) * g * g     # EMA of squares
            return p - lr * g / (np.sqrt(self.s) + self.eps)
        # Adam family
        if k == "adamw":
            p = p * (1 - lr * self.wd)                              # decoupled decay
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.s = self.b2 * self.s + (1 - self.b2) * g * g
        self.smax = np.maximum(self.smax, self.s)                   # AMSGrad keeps the max
        s = self.smax if k == "amsgrad" else self.s
        m_hat = self.m / (1 - self.b1 ** self.t)
        s_hat = s / (1 - self.b2 ** self.t)
        return p - lr * m_hat / (np.sqrt(s_hat) + self.eps)


KINDS = ["sgd", "heavyball", "nesterov", "adagrad", "rmsprop", "adam", "amsgrad"]
LR_GRID = np.logspace(-4, 0.5, 19)                  # every method gets its best lr from this grid

# ---------------------------------------------------------------
# Experiment 1: 10-D quadratic, curvatures 1..100 (kappa = 100),
# once axis-aligned and once randomly rotated. Same start in the eigenbasis.
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
d = 10
lam = np.logspace(0, 2, d)
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))        # random rotation
z0 = rng.normal(size=d)
problems = {"axis-aligned": (np.diag(lam), z0), "rotated": (Q @ np.diag(lam) @ Q.T, Q @ z0)}

def steps_to_tol(kind, lr, H, x0, tol=1e-6, max_steps=3000):
    opt, x, f0 = Optimizer(kind, lr), x0.copy(), 0.5 * x0 @ H @ x0
    for t in range(1, max_steps + 1):
        x = opt.step(x, H @ x)
        f = 0.5 * x @ H @ x
        if not np.isfinite(f):
            return None
        if f < tol * f0:
            return t
    return None

print("Quadratic, kappa=100: steps to reduce the loss 10^6-fold (best lr from grid)")
for kind in KINDS:
    row = []
    for name, (H, x0) in problems.items():
        res = [(steps_to_tol(kind, lr, H, x0), lr) for lr in LR_GRID]
        res = [r for r in res if r[0] is not None]
        best = min(res) if res else (None, None)
        row.append(f"{name}: {str(best[0]):>4s} (lr={best[1]:.3g})" if best[0] else f"{name}: >3000")
    print(f"  {kind:9s} " + "   ".join(row))

# ---------------------------------------------------------------
# Experiment 2: logistic regression on breast-cancer data,
# raw features (std from 0.002 to 577) versus standardised features.
# ---------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
mu, sd = Xtr.mean(0), Xtr.std(0)
print(f"\nBreast cancer: feature std ranges from {sd.min():.4f} to {sd.max():.1f}")

def train_logreg(kind, lr, A, At, epochs=30, batch=32):
    A = np.hstack([A, np.ones((len(A), 1))])        # bias column
    At = np.hstack([At, np.ones((len(At), 1))])
    w, opt, order = np.zeros(A.shape[1]), Optimizer(kind, lr), np.random.default_rng(1)
    for _ in range(epochs):
        idx = order.permutation(len(A))
        for s in range(0, len(A), batch):
            b = idx[s:s + batch]
            p = 1 / (1 + np.exp(-A[b] @ w))
            w = opt.step(w, A[b].T @ (p - ytr[b]) / len(b))
    z = A @ w
    return np.mean(np.logaddexp(0, z) - ytr * z), np.mean((At @ w > 0) == yte)

for tag, A, At in [("raw", Xtr, Xte), ("standardised", (Xtr - mu) / sd, (Xte - mu) / sd)]:
    print(f"  {tag} features (30 epochs, batch 32, best lr from grid):")
    for kind in KINDS:
        runs = [(*train_logreg(kind, lr, A, At), lr) for lr in np.logspace(-6, 0, 13)]
        loss, acc, lr = min(r for r in runs if np.isfinite(r[0]))
        print(f"    {kind:9s} lr={lr:8.2g}  train loss={loss:.4f}  test acc={acc:.3f}")

# ---------------------------------------------------------------
# Experiment 3: sparse binary features; feature j is "on" with probability
# from 0.5 down to 0.002. How well is each true weight (+-2) recovered in one pass?
# ---------------------------------------------------------------
n, d = 20000, 40
freq = np.logspace(np.log10(0.5), np.log10(0.002), d)
Xs = (rng.random((n, d)) < freq).astype(float)
w_true = rng.choice([-2.0, 2.0], size=d)
ys = (rng.random(n) < 1 / (1 + np.exp(-Xs @ w_true))).astype(float)
print(f"\nSparse features: frequent features seen ~{Xs[:, :10].sum(0).mean():.0f} times, rare ~{Xs[:, -10:].sum(0).mean():.0f} times")
for kind in ["sgd", "heavyball", "adagrad", "rmsprop", "adam"]:
    best = None
    for lr in np.logspace(-3, 1, 17):
        w, opt = np.zeros(d), Optimizer(kind, lr)
        for s in range(0, n, 32):
            xb, yb = Xs[s:s + 32], ys[s:s + 32]
            w = opt.step(w, xb.T @ (1 / (1 + np.exp(-xb @ w)) - yb) / len(yb))
        err = np.abs(w - w_true)
        if best is None or err.mean() < best[0]:
            best = (err.mean(), lr, err[:10].mean(), err[-10:].mean())
    print(f"  {kind:9s} lr={best[1]:7.3g}  mean |w - w_true| = {best[0]:.3f}  (10 most frequent {best[2]:.3f}, 10 rarest {best[3]:.3f})")`;
const SCRATCH_OUT = String.raw`Quadratic, kappa=100: steps to reduce the loss 10^6-fold (best lr from grid)
  sgd       axis-aligned:  184 (lr=0.0178)   rotated:  184 (lr=0.0178)
  heavyball axis-aligned:  105 (lr=0.0178)   rotated:  105 (lr=0.0178)
  nesterov  axis-aligned:   64 (lr=0.01)   rotated:   64 (lr=0.01)
  adagrad   axis-aligned:    7 (lr=1.78)   rotated:  126 (lr=3.16)
  rmsprop   axis-aligned: 1310 (lr=0.00178)   rotated:  691 (lr=0.00316)
  adam      axis-aligned:  103 (lr=0.316)   rotated:  113 (lr=0.562)
  amsgrad   axis-aligned:  103 (lr=0.316)   rotated:  113 (lr=0.562)

Breast cancer: feature std ranges from 0.0023 to 576.9
  raw features (30 epochs, batch 32, best lr from grid):
    sgd       lr= 3.2e-05  train loss=0.2515  test acc=0.912
    heavyball lr= 3.2e-06  train loss=0.2383  test acc=0.912
    nesterov  lr= 3.2e-06  train loss=0.2501  test acc=0.918
    adagrad   lr=   0.032  train loss=0.2028  test acc=0.901
    rmsprop   lr=  0.0032  train loss=0.1922  test acc=0.912
    adam      lr=  0.0032  train loss=0.1706  test acc=0.912
    amsgrad   lr=  0.0032  train loss=0.1706  test acc=0.912
  standardised features (30 epochs, batch 32, best lr from grid):
    sgd       lr=       1  train loss=0.0432  test acc=0.953
    heavyball lr=    0.32  train loss=0.0356  test acc=0.959
    nesterov  lr=       1  train loss=0.0324  test acc=0.953
    adagrad   lr=       1  train loss=0.0368  test acc=0.953
    rmsprop   lr=     0.1  train loss=0.0353  test acc=0.953
    adam      lr=     0.1  train loss=0.0351  test acc=0.959
    amsgrad   lr=     0.1  train loss=0.0352  test acc=0.959

Sparse features: frequent features seen ~5717 times, rare ~84 times
  sgd       lr=   5.62  mean |w - w_true| = 0.320  (10 most frequent 0.347, 10 rarest 0.452)
  heavyball lr=  0.562  mean |w - w_true| = 0.291  (10 most frequent 0.176, 10 rarest 0.514)
  adagrad   lr=  0.562  mean |w - w_true| = 0.203  (10 most frequent 0.104, 10 rarest 0.378)
  rmsprop   lr= 0.0316  mean |w - w_true| = 0.248  (10 most frequent 0.120, 10 rarest 0.506)
  adam      lr= 0.0316  mean |w - w_true| = 0.242  (10 most frequent 0.109, 10 rarest 0.422)`;
const DECAY_PY = String.raw`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

def adam_family(grad, w, lr, steps, wd=0.0, decoupled=False, b1=0.9, b2=0.999, eps=1e-8):
    """Adam (wd added to the gradient = 'Adam + L2') or AdamW (decoupled=True)."""
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, steps + 1):
        g = grad(w)
        if decoupled:
            w = w * (1 - lr * wd)                 # AdamW: shrink first (PyTorch order)
        else:
            g = g + wd * w                        # Adam + L2: penalty joins the gradient
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return w

# ---------------------------------------------------------------
# 1) Which objective does each optimizer actually minimise?
#    Logistic regression on the (standardised) breast-cancer data.
# ---------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)
n, d = X.shape
lam = 0.05                                         # regularisation strength

def grad_logistic(w):
    p = 1 / (1 + np.exp(-X @ w))
    return X.T @ (p - y) / n

# Exact minimiser of  mean log-loss + lam/2 ||w||^2  (sklearn's C = 1/(n*lam))
w_ref = LogisticRegression(C=1 / (n * lam), fit_intercept=False, tol=1e-12,
                           max_iter=10000).fit(X, y).coef_.ravel()
w_l2 = adam_family(grad_logistic, np.zeros(d), lr=0.01, steps=20000, wd=lam)
w_dw = adam_family(grad_logistic, np.zeros(d), lr=0.01, steps=20000, wd=lam, decoupled=True)
for name, w in [("L2 optimum (sklearn)", w_ref), ("Adam + L2", w_l2), ("AdamW", w_dw)]:
    kkt = np.linalg.norm(grad_logistic(w) + lam * w)  # zero only at the L2 optimum
    print(f"{name:21s} ||w||={np.linalg.norm(w):7.3f}  max|w|={np.abs(w).max():6.3f}  "
          f"||grad of L2 objective||={kkt:.1e}")
print(f"1/lam = {1 / lam:.1f}  (AdamW keeps every |w_i| below about this)")

# ---------------------------------------------------------------
# 2) Implicit bias: under-determined least squares (more weights than examples)
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
A = rng.normal(size=(20, 50))
b = rng.normal(size=20)
w_minnorm = np.linalg.pinv(A) @ b                 # smallest-norm exact solution
grad_ls = lambda w: A.T @ (A @ w - b) / len(b)

w_gd = np.zeros(50)
for _ in range(20000):
    w_gd -= 0.05 * grad_ls(w_gd)
w_adam = adam_family(grad_ls, np.zeros(50), lr=0.002, steps=20000)
for name, w in [("min-norm (pinv)", w_minnorm), ("gradient descent", w_gd), ("Adam", w_adam)]:
    print(f"{name:17s} train MSE={np.mean((A @ w - b) ** 2):.1e}  ||w||={np.linalg.norm(w):.4f}  "
          f"dist to min-norm={np.linalg.norm(w - w_minnorm):.4f}")`;
const DECAY_OUT = String.raw`L2 optimum (sklearn)  ||w||=  1.468  max|w|= 0.425  ||grad of L2 objective||=1.6e-08
Adam + L2             ||w||=  1.468  max|w|= 0.425  ||grad of L2 objective||=1.3e-05
AdamW                 ||w||= 47.784  max|w|=19.605  ||grad of L2 objective||=2.4e+00
1/lam = 20.0  (AdamW keeps every |w_i| below about this)
min-norm (pinv)   train MSE=1.3e-30  ||w||=0.5982  dist to min-norm=0.0000
gradient descent  train MSE=2.2e-31  ||w||=0.5982  dist to min-norm=0.0000
Adam              train MSE=2.4e-08  ||w||=0.6481  dist to min-norm=0.2494`;
const TORCH_PY = String.raw`import numpy as np
import torch
torch.set_num_threads(1)
from sklearn.datasets import load_breast_cancer

torch.manual_seed(0)
X, y = load_breast_cancer(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)
X = np.hstack([X, np.ones((len(X), 1))])           # bias column
n, d = X.shape

def grad_np(w, idx):
    p = 1 / (1 + np.exp(-X[idx] @ w))
    return X[idx].T @ (p - y[idx]) / len(idx)

# ----- compact NumPy reference (same equations as the from-scratch section) -----
def numpy_run(kind, lr, steps, batches, wd=0.0, beta=0.9, rho=0.99, b1=0.9, b2=0.999):
    eps = 1e-10 if kind == "adagrad" else 1e-8       # PyTorch's defaults
    w, buf, m, s, smax = np.zeros(d), 0.0, 0.0, 0.0, 0.0
    for t in range(1, steps + 1):
        g = grad_np(w, batches[t - 1])
        if kind in ("sgd_l2",):
            g = g + wd * w
        if kind in ("momentum", "nesterov", "sgd_l2"):
            buf = beta * buf + g
            w = w - lr * (g + beta * buf if kind == "nesterov" else buf)
        elif kind == "adagrad":
            s = s + g * g
            w = w - lr * g / (np.sqrt(s) + eps)
        elif kind == "rmsprop":
            s = rho * s + (1 - rho) * g * g
            w = w - lr * g / (np.sqrt(s) + eps)
        else:                                        # adam / amsgrad / adamw
            if kind == "adamw":
                w = w * (1 - lr * wd)
            m = b1 * m + (1 - b1) * g
            s = b2 * s + (1 - b2) * g * g
            smax = np.maximum(smax, s)
            sh = (smax if kind == "amsgrad" else s) / (1 - b2 ** t)
            w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(sh) + eps)
    return w

# ----- the same runs with torch.optim, in float64 so differences are pure rounding -----
def torch_run(make_opt, steps, batches):
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    opt = make_opt([w])
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
    for t in range(steps):
        idx = torch.tensor(batches[t])
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt[idx] @ w, yt[idx])
        loss.backward()
        opt.step()
    return w.detach().numpy()

rng = np.random.default_rng(0)
steps = 300
batches = [rng.choice(n, size=32, replace=False) for _ in range(steps)]
O = torch.optim
cases = {
    "momentum":  (dict(lr=0.1),            lambda p: O.SGD(p, lr=0.1, momentum=0.9)),
    "nesterov":  (dict(lr=0.1),            lambda p: O.SGD(p, lr=0.1, momentum=0.9, nesterov=True)),
    "sgd_l2":    (dict(lr=0.1, wd=1e-2),   lambda p: O.SGD(p, lr=0.1, momentum=0.9, weight_decay=1e-2)),
    "adagrad":   (dict(lr=0.1),            lambda p: O.Adagrad(p, lr=0.1)),
    "rmsprop":   (dict(lr=0.01),           lambda p: O.RMSprop(p, lr=0.01)),
    "adam":      (dict(lr=0.01),           lambda p: O.Adam(p, lr=0.01)),
    "amsgrad":   (dict(lr=0.01),           lambda p: O.Adam(p, lr=0.01, amsgrad=True)),
    "adamw":     (dict(lr=0.01, wd=0.1),   lambda p: O.AdamW(p, lr=0.01, weight_decay=0.1)),
}
print(f"NumPy from scratch vs torch.optim, {steps} mini-batch steps on breast cancer")
for kind, (kw, make) in cases.items():
    diff = np.abs(numpy_run(kind, steps=steps, batches=batches, **kw) - torch_run(make, steps, batches)).max()
    print(f"  {kind:9s} max |w_numpy - w_torch| = {diff:.1e}")

# ----- optimizer-state memory, measured -----
model = torch.nn.Sequential(torch.nn.Linear(784, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10))
n_params = sum(p.numel() for p in model.parameters())
for name, opt in [("SGD", O.SGD(model.parameters(), lr=0.1)),
                  ("SGD+momentum", O.SGD(model.parameters(), lr=0.1, momentum=0.9)),
                  ("Adam", O.Adam(model.parameters())),
                  ("Adam amsgrad", O.Adam(model.parameters(), amsgrad=True))]:
    model.zero_grad()
    model(torch.randn(8, 784)).sum().backward()
    opt.step()                                       # state is created lazily on the first step
    state = sum(v.numel() for st in opt.state.values() for v in st.values()
                if torch.is_tensor(v) and v.numel() > 1)
    print(f"  {name:13s} state = {state / n_params:.0f} x parameters ({state * 4 / 2**20:.2f} MiB in fp32)")

# ----- what that means for a GPT-2-small-sized Transformer (counted from its shapes) -----
V, T, D, Lyr = 50257, 1024, 768, 12
per_layer = 2 * 2 * D + (D * 3 * D + 3 * D) + (D * D + D) + (D * 4 * D + 4 * D) + (4 * D * D + D)
P = V * D + T * D + Lyr * per_layer + 2 * D
print(f"GPT-2 small parameters: {P:,}")
for what, nbytes in [("fp32 weights", 4), ("fp32 grads", 4), ("Adam m + v (fp32)", 8),
                     ("total fp32 training state", 16)]:
    print(f"  {what:26s} {P * nbytes / 1e9:6.2f} GB")
print(f"7e9-parameter model, Adam with fp32 m, v, master weights + bf16 weights/grads: "
      f"{7e9 * (4 + 4 + 4 + 2 + 2) / 1e9:.0f} GB before activations")`;
const TORCH_OUT = String.raw`NumPy from scratch vs torch.optim, 300 mini-batch steps on breast cancer
  momentum  max |w_numpy - w_torch| = 5.6e-16
  nesterov  max |w_numpy - w_torch| = 2.2e-16
  sgd_l2    max |w_numpy - w_torch| = 1.7e-16
  adagrad   max |w_numpy - w_torch| = 3.3e-16
  rmsprop   max |w_numpy - w_torch| = 3.3e-16
  adam      max |w_numpy - w_torch| = 2.2e-16
  amsgrad   max |w_numpy - w_torch| = 3.3e-16
  adamw     max |w_numpy - w_torch| = 1.1e-16
  SGD           state = 0 x parameters (0.00 MiB in fp32)
  SGD+momentum  state = 1 x parameters (1.55 MiB in fp32)
  Adam          state = 2 x parameters (3.11 MiB in fp32)
  Adam amsgrad  state = 3 x parameters (4.66 MiB in fp32)
GPT-2 small parameters: 124,439,808
  fp32 weights                 0.50 GB
  fp32 grads                   0.50 GB
  Adam m + v (fp32)            1.00 GB
  total fp32 training state    1.99 GB
7e9-parameter model, Adam with fp32 m, v, master weights + bf16 weights/grads: 112 GB before activations`;
const EX_PY = String.raw`import numpy as np

def heavy_ball_tuning(mu, L):
    """Optimal constant (eta, beta) of heavy-ball momentum on a quadratic with curvatures in [mu, L]."""
    eta = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    return eta, beta

def run(eta, beta, lam, x0, steps):
    """Heavy ball (PyTorch form) on f(x) = 0.5 * sum(lam * x^2); returns final distance to 0."""
    x, v = x0.copy(), np.zeros_like(x0)
    for _ in range(steps):
        v = beta * v + lam * x
        x = x - eta * v
    return np.linalg.norm(x)

mu, L = 1.0, 1000.0
lam = np.linspace(mu, L, 50)                      # 50 curvatures filling [mu, L]
x0 = np.ones(50)
eta_hb, beta_hb = heavy_ball_tuning(mu, L)
print(f"kappa={L / mu:.0f}: eta*={eta_hb:.6f}, beta*={beta_hb:.4f}, "
      f"predicted rate {np.sqrt(beta_hb):.4f} vs GD {(L - mu) / (L + mu):.4f}")

configs = {"GD, eta=2/(mu+L)": (2 / (mu + L), 0.0),
           "HB, optimal": (eta_hb, beta_hb)}
# For fixed beta, search the best eta on a fine grid
for b in (0.9, 0.99):
    etas = np.logspace(-5, np.log10(2 * (1 + b) / L), 400)
    best_eta = min(etas, key=lambda e: run(e, b, lam, x0, 300))
    configs[f"HB, beta={b}, best eta"] = (best_eta, b)
for name, (e, b) in configs.items():
    print(f"  {name:22s} eta={e:.6f}  |x_300| = {run(e, b, lam, x0, 300):.3e}  |x_1000| = {run(e, b, lam, x0, 1000):.3e}")`;
const EX_OUT = String.raw`kappa=1000: eta*=0.003759, beta*=0.8811, predicted rate 0.9387 vs GD 0.9980
  GD, eta=2/(mu+L)       eta=0.001998  |x_300| = 7.761e-01  |x_1000| = 1.914e-01
  HB, optimal            eta=0.003759  |x_300| = 3.333e-06  |x_1000| = 6.480e-25
  HB, beta=0.9, best eta eta=0.002950  |x_300| = 9.174e-07  |x_1000| = 9.295e-23
  HB, beta=0.99, best eta eta=0.000741  |x_300| = 9.631e-01  |x_1000| = 3.436e-02`;
const PG_PY = String.raw`import torch

model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True), num_layers=4)

# Decay only weight matrices; never biases or LayerNorm gains (shrinking them does not regularise usefully)
decay, no_decay = [], []
for name, p in model.named_parameters():
    (decay if p.ndim >= 2 else no_decay).append(p)

opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# Other switches worth knowing
sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
ams = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)          # AMSGrad
multi = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=True)       # multi-tensor kernels: faster on GPU
print(len(decay), "decayed tensors,", len(no_decay), "not decayed")
print(sum(p.numel() for p in decay), "decayed parameters,", sum(p.numel() for p in no_decay), "not decayed")`;
const PG_OUT = String.raw`16 decayed tensors, 32 not decayed
5242880 decayed parameters, 17408 not decayed`;

const lesson: Lesson = {
  id: "m06/momentum-adaptive",
  title: "Momentum and Adaptive Optimizers in Depth: EMAs, Heavy Ball, Nesterov, AdaGrad to AdamW",
  summary:
    "Why the popular optimizers work, when they fail and how to set them. We cover moving averages and bias correction, momentum as a noise filter and as a rolling heavy ball, the exact math that gives the best momentum for a problem, Nesterov's extra braking, why per-parameter step sizes help sparse and badly scaled data, Adam's step as a signal-to-noise ratio, AMSGrad, L2 versus decoupled weight decay, the SGD-vs-Adam debate, optimizer memory and a practical recipe per domain.",
  minutes: 90,
  level: "intermediate",
  prerequisites: ["m06/gradient-descent", "m02/eigen-svd", "m02/optimization-math", "m04/backpropagation"],
  next: ["m06/lr-schedules", "m07/regularization", "m07/normalization", "m07/initialization"],
  terms: ["momentum", "exponential-moving-average", "bias-correction", "nesterov-momentum", "adagrad", "rmsprop", "adam", "adamw", "amsgrad", "condition-number", "weight-decay"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "In [Gradient Descent and Its Descendants](/learn/m06/gradient-descent) you met the update rules of Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW and Nadam. This lesson asks the next questions: **why** do they work, **when** do they fail, and **how** should you set their knobs? We answer each one on a small model problem and then check it with code.",
      },
      {
        t: "p",
        text: "Two ideas power almost every modern optimizer:",
      },
      {
        t: "list",
        items: [
          "**Averaging over time.** Momentum and Adam keep a running average of recent gradients (an exponential moving average, or EMA). Noise cancels out, and speed builds up in directions that stay the same.",
          "**A separate step size per parameter.** AdaGrad, RMSProp and Adam divide each parameter's step by how big its gradients usually are. Parameters with huge and tiny gradients then move at a similar pace.",
        ],
      },
      {
        t: "p",
        text: "History in one line: Polyak's heavy ball (1964), Nesterov (1983), AdaGrad by Duchi, Hazan and Singer (2011), Adam by Kingma and Ba (2015), AMSGrad by Reddi, Kale and Kumar (2018, a fix for a flaw in Adam's proof), AdamW by Loshchilov and Hutter (2019, a fix for weight decay).",
      },
      {
        t: "p",
        text: "SGD with momentum still trains many image models, AdamW trains nearly every large Transformer, and AdaGrad-style methods are common for sparse embedding tables. Picking one and setting its knobs is full of folklore; we replace it with reasons you can work out yourself.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The headline result",
        text: String.raw`With the right momentum, $\beta^\star = \big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$, the error shrinks by $\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$ per step, so steps grow like $\sqrt\kappa$ instead of $\kappa$. Adaptive methods add a per-parameter step of about $\eta$ times the gradient's signal-to-noise ratio.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Gradient Descent and Its Descendants](/learn/m06/gradient-descent) — **essential**. You need all the update rules, the fact that gradient descent on a quadratic converges only if $0 < \eta < 2/\lambda_{\max}$, and the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ (how much steeper the steepest direction is than the flattest).`,
          String.raw`[Eigenvalues and the SVD](/learn/m02/eigen-svd) — writing $H = Q\Lambda Q^\top$ splits a quadratic into separate one-dimensional problems. The **spectral radius** (largest absolute eigenvalue) of an update matrix decides whether a repeated update shrinks the error.`,
          String.raw`[Optimization math](/learn/m02/optimization-math) — convexity, smoothness and the Hessian as curvature.`,
          "[Backpropagation](/learn/m04/backpropagation) — where the gradients every optimizer uses come from.",
          String.raw`Basic complex numbers: the roots of $r^2 - br + c = 0$ are complex when $b^2 < 4c$, and then $|r|^2 = c$. We use this once, in the momentum analysis.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A car's suspension and a noise-cancelling headset",
        text: "A car's suspension lets the car follow the slow shape of the road (a hill, a bend) while ignoring fast bumps. Momentum does the same to gradients: the slow part — the direction the loss keeps going down — passes through and even gets stronger, while the fast part — zig-zagging across a narrow valley, or mini-batch noise — is smoothed away. An adaptive optimizer is like a headset that sets the volume of every channel on its own: loud channels get turned down, quiet ones up, so every parameter moves at a similar pace. Where it stops: real optimizers only see numbers, not a road.",
      },
      {
        t: "p",
        text: "Picture a long, narrow valley: steep walls across, a gentle slope along the floor. Plain gradient descent must take small steps so it doesn't bounce off the walls, so it crawls along the floor. A heavy ball rolling down the same valley keeps building speed along the floor (each push adds to the last). Its sideways sloshing dies out, because the sideways pushes keep changing direction and cancel. That is all momentum does — and the math below turns it into exact numbers: with the right settings, the number of steps grows like $\\sqrt{\\kappa}$ instead of $\\kappa$.",
      },
      {
        t: "p",
        text: "Adaptive methods solve a different problem. In a network, the gradient for the embedding of a rare word is almost always zero, while the gradient for a last-layer bias is large on every step. One shared learning rate cannot suit both. Dividing each parameter's gradient by its own typical size makes the step about the same for everyone. Adam's step is close to $\\pm\\eta$ per parameter when the gradient keeps the same sign, and shrinks on its own when the gradient is mostly noise.",
      },
      {
        t: "list",
        items: [
          String.raw`**EMA** — a running average that slowly forgets old values; $\beta = 0.9$ remembers roughly the last 10 values, $\beta = 0.999$ the last 1000.`,
          "**Low-pass filter** — something that lets slow changes through and weakens fast ones.",
          "**Preconditioning** — rescaling the gradient (here: per parameter) so the problem looks evenly shaped to the optimizer.",
          "**Decoupled weight decay** — shrinking the weights directly instead of through the gradient, so the adaptive rescaling cannot distort it.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "One Adam(W) step, with everything it stores" },
      {
        t: "steps",
        items: [
          { title: "Receive the gradient", text: String.raw`Backprop gives $g_t$ for every parameter (after optional gradient clipping). The step counter goes up by one: $t = 1$ on the first update.` },
          { title: "Decoupled weight decay (AdamW only)", text: String.raw`PyTorch does this first, on the current weights: $\theta \leftarrow \theta\,(1 - \eta\lambda)$. In plain Adam with weight_decay (L2), $\lambda\theta$ is instead added to $g_t$ here.` },
          { title: "Update the first moment (running average of gradients)", tex: String.raw`m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t` },
          { title: "Update the second moment (running average of squared gradients)", tex: String.raw`v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2` },
          { title: "AMSGrad only: keep the running maximum", tex: String.raw`v^{\max}_t = \max(v^{\max}_{t-1},\, v_t)\ \text{ is used instead of } v_t` },
          { title: "Fix the zero-start bias of both", tex: String.raw`\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}` },
          { title: "Take the normalised step", tex: String.raw`\theta \leftarrow \theta - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}` },
        ],
      },
      {
        t: "p",
        text: String.raw`Every operation works on each number separately, so the cost is a few vector operations per parameter. Adam stores $m$ and $v$ (two tensors the size of the model), plus $v^{\max}$ for AMSGrad, plus the counter $t$.`,
      },
      { t: "h", text: "Heavy ball: two ways to write the same momentum" },
      {
        t: "p",
        text: String.raw`PyTorch writes momentum as $v_t = \beta v_{t-1} + g_t,\ \theta_{t+1} = \theta_t - \eta v_t$. We can remove $v$: since $\theta_t - \theta_{t-1} = -\eta v_{t-1}$,`,
      },
      { t: "math", tex: String.raw`\theta_{t+1} = \theta_t - \eta\, g_t + \beta\,(\theta_t - \theta_{t-1})`, caption: "Polyak's heavy-ball form: a normal gradient step plus a fraction β of the last move." },
      {
        t: "p",
        text: "We use this form in all the math below. The last term is inertia: the weights keep moving the way they were already going, slowed by β (like friction).",
      },
      { t: "h", text: "Nesterov: from the look-ahead form to PyTorch's form" },
      {
        t: "steps",
        items: [
          { title: "Look-ahead form (Sutskever et al., 2013)", tex: String.raw`v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta\beta v_{t-1}), \quad \theta_{t+1} = \theta_t - \eta v_t` },
          { title: "Store the look-ahead point instead", tex: String.raw`\phi_t := \theta_t - \eta\beta v_{t-1}, \quad g_t := \nabla L(\phi_t)` },
          { title: "Substitute", tex: String.raw`\phi_{t+1} = \theta_{t+1} - \eta\beta v_t = \theta_t - \eta v_t - \eta\beta v_t = \phi_t + \eta\beta v_{t-1} - \eta(1+\beta) v_t` },
          { title: "Use v_t = β v_{t-1} + g_t", tex: String.raw`\phi_{t+1} = \phi_t - \eta\,(g_t + \beta v_t)` },
        ],
      },
      {
        t: "p",
        text: "The last line only needs the gradient at the stored weights, so it fits the usual `loss.backward(); opt.step()` loop. This is exactly what `torch.optim.SGD(..., momentum=β, nesterov=True)` does.",
      },
      { t: "h", text: "What each knob controls" },
      {
        t: "table",
        head: ["Knob", "Controls", "Typical value", "Raise it when", "Lower it when"],
        rows: [
          [String.raw`$\beta$ (momentum)`, String.raw`memory of about $1/(1-\beta)$ steps; effective step $\eta/(1-\beta)$`, "0.9", "very unevenly shaped problem, noisy gradients", "training oscillates or diverges; very short runs"],
          [String.raw`$\beta_1$ (Adam)`, "smoothing of the gradient direction", "0.9", "noisy gradients", String.raw`GANs and other two-player games that keep changing (0.5 or 0 is common)`],
          [String.raw`$\beta_2$ (Adam)`, String.raw`memory of the squared-gradient average, $\approx 1/(1-\beta_2)$ steps`, "0.999 (0.95–0.98 for large Transformers)", "gradients are very noisy and steady over time", "loss spikes from out-of-date second moments; large-batch Transformer training"],
          [String.raw`$\epsilon$`, String.raw`floor under the denominator; caps the step at $\eta\,|\hat m|/\epsilon$`, String.raw`$10^{-8}$`, "updates too jumpy for tiny-gradient parameters, low-precision arithmetic", "almost never"],
          [String.raw`$\lambda$ (weight decay)`, String.raw`shrinking per step $\eta\lambda$`, "0.01–0.1 (AdamW), 1e-4–5e-4 (SGD)", "overfitting", "underfitting; never on biases and normalisation gains"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Exponential moving averages" },
      {
        t: "math",
        tex: String.raw`m_t = \beta\, m_{t-1} + (1-\beta)\, g_t,\quad m_0 = 0 \qquad\Longrightarrow\qquad m_t = \sum_{k=0}^{t-1} (1-\beta)\beta^{k}\, g_{t-k}`,
        caption: String.raw`$g_t$ — the value being averaged (a gradient or a squared gradient); $\beta \in [0,1)$ — how slowly it forgets; $k$ — how many steps old a value is.`,
      },
      {
        t: "p",
        text: String.raw`In words: each new average keeps a fraction $\beta$ of the old one and mixes in a bit of the new value. Unrolled, a value $k$ steps old gets weight $w_k = (1-\beta)\beta^k$ — older values count less and less.`,
      },
      {
        t: "p",
        text: String.raw`**How far back it looks.** Over a long history the weights sum to 1, and the average age is $\sum_k k\,w_k = \beta/(1-\beta)$ (9 steps for $\beta = 0.9$). After $1/(1-\beta)$ steps a weight has dropped to $\beta^{1/(1-\beta)} \approx e^{-1}$, because $\ln \beta \approx -(1-\beta)$. So the rule of thumb: an EMA averages roughly the last $1/(1-\beta)$ values — 10, 100 and 1000 for $\beta = 0.9, 0.99, 0.999$.`,
      },
      {
        t: "p",
        text: String.raw`**How much noise it removes.** If the $g_t$ are independent with variance $\sigma^2$, the EMA settles to variance $\sigma^2(1-\beta)^2\sum_k \beta^{2k} = \sigma^2\,\frac{1-\beta}{1+\beta}$. A plain average of $N$ samples has variance $\sigma^2/N$, so an EMA is as good as averaging $N_{\text{eff}} = \frac{1+\beta}{1-\beta}$ samples — 19 for $\beta = 0.9$, about twice its window.`,
      },
      {
        t: "p",
        text: String.raw`**The zero-start problem.** With $m_0 = 0$, only $t$ values have been seen, and their weights add up to $\sum_{k=0}^{t-1}(1-\beta)\beta^k = 1 - \beta^t < 1$. The missing weight $\beta^t$ sits on the made-up starting value 0, which drags the average toward zero. Dividing by $1 - \beta^t$ **rescales the weights so they sum to one**:`,
      },
      {
        t: "math",
        tex: String.raw`\hat m_t = \frac{m_t}{1-\beta^t} = \frac{\sum_{k=0}^{t-1} \beta^k g_{t-k}}{\sum_{k=0}^{t-1}\beta^k}`,
        caption: "Bias correction turns the zero-started EMA into a proper weighted average of the values seen so far.",
      },
      {
        t: "p",
        text: String.raw`This view needs no assumption that gradients are steady (the previous lesson's $\mathbb{E}[m_t] = (1-\beta^t)\mathbb{E}[g]$ follows from it). It matters while $\beta^t$ is not tiny: $\beta^t < 0.01$ needs $t > \ln 0.01/\ln\beta$, which is **44** steps for $\beta = 0.9$ but **4603** steps for $\beta_2 = 0.999$. Adam's second moment stays biased for thousands of steps, so the correction is not a cosmetic detail.`,
      },
      { t: "h", text: "Worked example: an EMA of three gradients" },
      {
        t: "table",
        head: ["$t$", "$g_t$", String.raw`$m_t = 0.9\,m_{t-1} + 0.1\,g_t$`, String.raw`$1 - 0.9^t$`, String.raw`$\hat m_t$`, "normalised weighted mean"],
        rows: [
          ["1", "2", "0.9·0 + 0.2 = 0.2000", "0.1000", "2.0000", "2/1 = 2.0000"],
          ["2", "4", "0.18 + 0.40 = 0.5800", "0.1900", "3.0526", "(0.9·2 + 4)/1.9 = 3.0526"],
          ["3", "3", "0.522 + 0.30 = 0.8220", "0.2710", "3.0332", "(0.81·2 + 0.9·4 + 3)/2.71 = 3.0332"],
        ],
      },
      {
        t: "p",
        text: String.raw`The raw EMA (0.2, 0.58, 0.822) is nowhere near the data (2, 4, 3). The corrected values are exactly the weighted averages. The program below prints these numbers.`,
      },

      { t: "h", text: "2. Momentum as a low-pass filter" },
      {
        t: "p",
        text: String.raw`Treat the gradients as a signal and feed in a pure wave $g_t = e^{i\omega t}$ with frequency $\omega$ ($\omega = 0$: constant; $\omega = \pi$: flips sign every step). Once settled, the velocity $v_t = \beta v_{t-1} + g_t$ is the same wave times a factor: $v_t = G(\omega)\,e^{i\omega t}$. Plugging in gives $G = \beta G e^{-i\omega} + 1$, so`,
      },
      {
        t: "math",
        tex: String.raw`G(\omega) = \frac{1}{1 - \beta e^{-i\omega}}, \qquad |G(0)| = \frac{1}{1-\beta}, \qquad |G(\pi)| = \frac{1}{1+\beta}`,
        caption: String.raw`$|G(\omega)|$ — the gain: how much momentum boosts a gradient part that repeats at frequency $\omega$.`,
      },
      {
        t: "p",
        text: String.raw`For $\beta = 0.9$, a constant gradient is boosted 10×, a wave with period 4 is multiplied by 0.743, and a sign-flipping one by 0.526. The slowest part beats the fastest by $(1+\beta)/(1-\beta) = 19$. That is the exact meaning of "speeds up steady directions and damps oscillations". Adam's EMA form $(1-\beta)v$ has the same shape scaled by $(1-\beta)$: gain 1 at $\omega = 0$ and $0.0526$ at $\omega = \pi$.`,
      },

      { t: "h", text: "3. The heavy ball as a physical system" },
      {
        t: "p",
        text: String.raw`The heavy-ball form $\theta_{t+1} - \theta_t = \beta(\theta_t - \theta_{t-1}) - \eta\nabla L(\theta_t)$ is a step-by-step version of a ball with mass $m$ rolling on the surface $L$ with friction $\gamma$: $m\ddot\theta + \gamma\dot\theta + \nabla L(\theta) = 0$. With time step $h$, use $\ddot\theta \approx (\theta_{t+1} - 2\theta_t + \theta_{t-1})/h^2$ and $\dot\theta \approx (\theta_t - \theta_{t-1})/h$; matching terms gives $\beta = 1 - \gamma h/m$ and $\eta = h^2/m$. So $1 - \beta$ is the friction per step. With no friction ($\beta \to 1$) the ball swings forever; with lots of friction ($\beta \to 0$) it creeps (plain gradient descent). Too little friction is a real failure — you will see it with $\beta = 0.99$ in the exercise.`,
      },

      { t: "h", text: "4. Momentum on a quadratic: the exact analysis" },
      {
        t: "p",
        text: String.raw`Take a bowl-shaped loss $L(\theta) = \tfrac12(\theta-\theta^\star)^\top H(\theta-\theta^\star)$ whose curvatures (eigenvalues) run from $\mu = \lambda_{\min}$ to $\lambda_{\max} = L$. As with gradient descent, each eigen-direction behaves on its own. For one direction with curvature $\lambda$ and error $e_t$ (distance from the minimum), heavy ball gives a recursion that looks two steps back:`,
      },
      {
        t: "math",
        tex: String.raw`e_{t+1} = (1 + \beta - \eta\lambda)\, e_t - \beta\, e_{t-1} \quad\Longleftrightarrow\quad \begin{pmatrix} e_{t+1} \\ e_t \end{pmatrix} = \underbrace{\begin{pmatrix} 1+\beta-\eta\lambda & -\beta \\ 1 & 0 \end{pmatrix}}_{M_\lambda} \begin{pmatrix} e_{t} \\ e_{t-1} \end{pmatrix}`,
      },
      {
        t: "p",
        text: String.raw`The error shrinks like $\rho(M_\lambda)^t$, where $\rho$ is the spectral radius (the size of the largest eigenvalue). The eigenvalues of $M_\lambda$ are the roots of`,
      },
      { t: "math", tex: String.raw`r^2 - (1+\beta-\eta\lambda)\, r + \beta = 0, \qquad r_1 r_2 = \beta` },
      {
        t: "list",
        items: [
          String.raw`**Complex roots** when $(1+\beta-\eta\lambda)^2 < 4\beta$, i.e. $(1-\sqrt\beta)^2 < \eta\lambda < (1+\sqrt\beta)^2$. Then both roots have the same size and $|r|^2 = r_1 r_2 = \beta$, so the rate is exactly $\sqrt{\beta}$, **whatever the curvature**. The error spirals in (a damped wobble).`,
          String.raw`**Real roots, low curvature** ($\eta\lambda < (1-\sqrt\beta)^2$): the slow root is $r \approx 1 - \eta\lambda/(1-\beta)$ — gradient descent with an effective learning rate $\eta/(1-\beta)$. This is how momentum speeds up flat directions.`,
          String.raw`**Blow-up** when a root leaves the unit circle. Putting $r = -1$ into the polynomial gives $2 + 2\beta - \eta\lambda = 0$, so heavy ball is stable only if $0 < \eta\lambda < 2(1+\beta)$ — plain gradient descent's limit of 2 becomes 3.8 for $\beta = 0.9$.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Best settings.** To make every curvature in $[\mu, L]$ shrink at the same rate $\sqrt\beta$, place the whole range inside the complex zone, with its ends on the edges: $\eta\mu = (1-\sqrt\beta)^2$ and $\eta L = (1+\sqrt\beta)^2$. Take square roots and add: $\sqrt\eta(\sqrt L + \sqrt\mu) = 2$. Subtract: $\sqrt\eta(\sqrt L - \sqrt\mu) = 2\sqrt\beta$. So`,
      },
      {
        t: "math",
        tex: String.raw`\boxed{\;\eta^\star = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \qquad \beta^\star = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2, \qquad \rho^\star = \sqrt{\beta^\star} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\;}`,
        caption: String.raw`$\kappa = L/\mu$. Compare gradient descent's best rate $(\kappa-1)/(\kappa+1)$.`,
      },
      {
        t: "p",
        text: String.raw`**Numbers for $\kappa = 100$** ($\mu = 1$, $L = 100$): gradient descent's best $\eta = 2/101 = 0.0198$ gives rate $99/101 = 0.9802$, so shrinking the error $10^6$-fold takes $\ln 10^{-6}/\ln 0.9802 = 691$ steps. Heavy ball with $\eta^\star = 4/121 = 0.03306$ and $\beta^\star = (9/11)^2 = 0.6694$ has rate $9/11 = 0.8182$ and needs **69** steps — a $\sqrt\kappa = 10$-fold speed-up. A grid search over $(\eta, \beta)$ finds nothing better (best rate 0.81854 at $\eta = 0.033$, $\beta = 0.670$). This is a worst-case guarantee; with only a few curvatures, other settings can win over a short run (see the exercise).`,
      },
      {
        t: "callout",
        kind: "note",
        title: "What this says about β = 0.9 in practice",
        text: String.raw`With $\beta = 0.9$, every direction with $0.00263 < \eta\lambda < 3.787$ shrinks at exactly $\sqrt{0.9} = 0.9487$ per step. Raising $\eta$ does not speed these up; it only pulls more flat directions into that zone. Directions flatter than $(1-\sqrt\beta)^2/\eta$ still behave like gradient descent with rate $\eta/(1-\beta)$. Real losses are not quadratics, but this picture — a 10× bigger effective step on flat directions, paid for by wobbling on steep ones — explains a lot of practice.`,
      },

      { t: "h", text: "5. Nesterov: extra braking on steep directions" },
      {
        t: "p",
        text: String.raw`For Nesterov, the same analysis (with $a = \eta\lambda$) gives $e_{t+1} = (1+\beta)(1-a)\,e_t - \beta(1-a)\,e_{t-1}$, with polynomial $r^2 - (1+\beta)(1-a)r + \beta(1-a) = 0$. Now the roots multiply to $\beta(1-a)$ instead of $\beta$. In the complex zone the rate is $\sqrt{\beta(1-\eta\lambda)}$: **steeper directions get braked more**. For $\beta = 0.9$ and $\eta\lambda = 0.5$, heavy ball shrinks the error by 0.9487 per step, Nesterov by $\sqrt{0.45} = 0.6708$. That is the number behind "the look-ahead gradient corrects the overshoot before it happens".`,
      },
      {
        t: "p",
        text: String.raw`The price: putting $r = -1$ in gives $1 + (1+\beta)(1-a) + \beta(1-a) = 0$, so Nesterov is stable only for $\eta\lambda < \frac{2(1+\beta)}{1+2\beta}$ — **1.357** for $\beta = 0.9$, versus 3.8 for heavy ball. Nesterov's step always includes the fresh gradient with weight $1+\beta$, so it reacts more strongly to curvature. So it needs a smaller $\eta$ but is often faster at its best $\eta$ (64 vs 105 steps on the 10-D quadratic in the code section). On smooth convex (not strongly convex) problems, Nesterov's error falls like $O(1/t^2)$ versus $O(1/t)$ for gradient descent — the classic acceleration result (Nesterov, 1983).`,
      },
      {
        t: "code",
        lang: "python",
        title: "ema_and_momentum_theory.py — every number in sections 1–5",
        code: THEORY_PY,
        output: THEORY_OUT,
      },

      { t: "h", text: "6. Adaptive step sizes: AdaGrad, sparse features, RMSProp" },
      {
        t: "p",
        text: String.raw`AdaGrad gives parameter $i$ the learning rate $\eta_{i,t} = \eta/\sqrt{G_{i,t}}$, where $G_{i,t} = \sum_{s \le t} g_{i,s}^2$ is the running total of its squared gradients. Two things follow.`,
      },
      {
        t: "list",
        items: [
          String.raw`**The rate keeps shrinking.** If a gradient has a roughly constant size $|g|$, then $G_{i,t} = t g^2$ and the step is $\eta |g|/\sqrt{t g^2} = \eta/\sqrt t$. Online convex optimisation theory recommends exactly this schedule (it gives regret $O(\sqrt T)$), but in deep learning, which needs millions of steps, it stalls training.`,
          String.raw`**Sparse features.** If feature $j$ is non-zero in only $n_j$ of the examples seen so far, it only gets gradients on those steps, so $G_j \approx n_j\,\overline{g^2}$ and its rate is $\propto 1/\sqrt{n_j}$. A feature seen 100 times gets a 10× larger rate than one seen 10 000 times. That is right for a rare feature: each of its few sightings says a lot about its weight. With one shared rate, either rare weights barely move or frequent ones oscillate.`,
        ],
      },
      {
        t: "p",
        text: String.raw`RMSProp swaps the total for an EMA: $s_{i,t} \approx \mathbb{E}_{\text{recent}}[g_i^2]$, so the rate is $\eta/\text{RMS}(g_i)$ over about the last $1/(1-\rho)$ steps. It stays steady when gradients are steady and adapts when their size drifts. The cost: the rare-feature advantage is weaker. If a rare feature has not appeared for a few hundred steps, its $s$ fades toward 0, and its next appearance gives a big, badly sized step. The sparse-feature experiment in the code section shows both effects.`,
      },

      { t: "h", text: "7. Adam's step is a normalised, sign-like step" },
      {
        t: "p",
        text: String.raw`Write each parameter's step as $\Delta_t = -\eta\, r_t$ with $r_t = \hat m_t/(\sqrt{\hat v_t} + \epsilon)$. Three cases explain most of how Adam behaves.`,
      },
      {
        t: "list",
        items: [
          String.raw`**No averaging.** With $\beta_1 = \beta_2 = 0$, $r_t = g_t/|g_t| = \operatorname{sign}(g_t)$: Adam becomes **sign descent**, and every parameter moves by exactly $\eta$.`,
          String.raw`**Signal-to-noise ratio (SNR).** Say a parameter's gradient has steady mean $\mu_g$ and variance $\sigma^2$. Over long windows $\hat m \approx \mu_g$ and $\hat v \approx \mathbb{E}[g^2] = \mu_g^2 + \sigma^2$, so $r \approx \mu_g/\sqrt{\mu_g^2 + \sigma^2}$. If the gradient is consistent ($|\mu_g| \gg \sigma$), the step is $\approx \pm\eta$. If it is mostly noise ($|\mu_g| \ll \sigma$), the step shrinks in line with the SNR $\mu_g/\sigma$. So Adam slows down by itself near a minimum, where the true gradient is small but mini-batch noise is not — a built-in cooling that SGD lacks. In a simulation with unit noise, $\mu_g = 10, 1, 0.1$ give average steps $0.995\eta,\ 0.704\eta,\ 0.096\eta$, against predictions of $0.995, 0.707, 0.0995$.`,
          String.raw`**Upper limit.** By Cauchy–Schwarz, $|\sum_k a_k g_k| \le \sqrt{\sum_k a_k^2/b_k}\,\sqrt{\sum_k b_k g_k^2}$ with $a_k = (1-\beta_1)\beta_1^k$, $b_k = (1-\beta_2)\beta_2^k$. Summing the series (valid when $\beta_1^2 < \beta_2$) gives, for large $t$, $|r_t| \le \frac{1-\beta_1}{\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)}}$ — **7.27** for the defaults. This worst case needs gradients that grow geometrically; a realistic bad case — a gradient of size 1 after 2000 steps of size $10^{-3}$ — gives $r = 2.97$. So "Adam moves each parameter by at most about $\eta$" is a rule of thumb, not a promise — one reason gradient clipping and warm-up are still used with Adam.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**What $\epsilon$ does.** For a constant gradient $g$, $r = |g|/(|g| + \epsilon)$. With $\epsilon = 10^{-8}$ the step is $\eta$ for any $|g| \gg 10^{-8}$; at $|g| = \epsilon$ it is halved; below that, Adam turns into SGD with a huge learning rate $\eta/\epsilon$. Raising $\epsilon$ to $10^{-3}$ means gradients smaller than about $10^{-3}$ no longer get full-size steps ($r = 0.09$ at $|g| = 10^{-4}$). So a large $\epsilon$ slides Adam toward momentum SGD with learning rate $\eta/\epsilon$. Larger values are used in practice for stability, e.g. $\epsilon = 10^{-6}$ in RoBERTa (Liu et al., 2019); in float16, $10^{-8}$ can even round down to zero.`,
      },
      {
        t: "table",
        head: ["Setting", "Common default", "When to change it, and why"],
        rows: [
          [String.raw`$\eta$`, String.raw`$10^{-3}$ (from scratch), $10^{-5}$–$10^{-4}$ (fine-tuning), $3\times10^{-4}$ (many RL and LLM runs)`, "Always tune it first (with a schedule, see the next lesson). Since each parameter moves about η per step, η is a step length in weight units: shrink it when weights are small (fine-tuning) or the model is large."],
          [String.raw`$\beta_1$`, "0.9", String.raw`0.5 (or 0) for GANs, where the loss surface keeps changing as the other network trains, so a long memory of old gradients hurts (DCGAN used Adam with $\beta_1 = 0.5$, $\eta = 2\times10^{-4}$).`],
          [String.raw`$\beta_2$`, "0.999", String.raw`0.98 (original Transformer, RoBERTa) or 0.95 (GPT-3, LLaMA). A shorter memory lets $v$ react fast when gradient size jumps; with 0.999 a sudden large gradient is divided by an old, too-small $\sqrt{v}$ and the step explodes (see the limit above).`],
          [String.raw`$\epsilon$`, String.raw`$10^{-8}$`, String.raw`$10^{-6}$ or larger for stability in some large-scale or low-precision runs; much larger values (up to 0.1–1) turn Adam into momentum SGD and have been used in some vision and RL settings.`],
          ["weight decay (AdamW)", "0.01 (PyTorch default); 0.1 for LLM pre-training", "Tune together with η: the shrink per step is ηλ."],
        ],
      },

      { t: "h", text: "8. AMSGrad: when Adam fails to converge" },
      {
        t: "p",
        text: String.raw`Reddi, Kale and Kumar ("On the Convergence of Adam and Beyond", ICLR 2018) found a mistake in Adam's original convergence proof and built simple convex problems where Adam settles on the wrong answer. The cause: the effective rate $\eta/\sqrt{\hat v_t}$ can **go back up** when recent gradients are small. If rare large gradients carry the key information, the EMA forgets them within about $1/(1-\beta_2)$ steps, and the frequent small gradients then get large steps that outweigh them.`,
      },
      {
        t: "p",
        text: String.raw`Their simplest example: an online problem on $x \in [-1, 1]$ whose loss is $C x$ on every third step and $-x$ on the others, with $C > 2$. The average gradient is $(C - 2)/3 > 0$, so the best point is $x = -1$. With $C = 3$, $\beta_1 = 0$ and $\beta_2 = 1/(1 + C^2) = 0.1$, the normalised gradients over one cycle settle at $+1.0472, -0.7623, -0.9658$. Their sum, $-0.68$, is **negative**, so Adam drifts toward $+1$ — the worst point. AMSGrad divides by $v^{\max}_t = \max(v^{\max}_{t-1}, v_t)$ instead, so once a large gradient appears its size is never forgotten and the rate can only fall. It finds the right answer. With the default $\beta_2 = 0.999$ both work on this problem, but the paper also proves that for any fixed $\beta_1 < \sqrt{\beta_2}$ there are stochastic convex problems where Adam does not converge.`,
      },
      {
        t: "p",
        text: "In deep-learning practice AMSGrad has rarely beaten Adam with a good β₂ and a decaying learning rate, so it stays an option (`amsgrad=True`), not the default. Its lesson is about ideas: Adam needs a long enough memory, and decaying the learning rate is part of what makes Adam converge.",
      },
      {
        t: "code",
        lang: "python",
        title: "adam_step_anatomy.py — SNR, bounds, ε and the AMSGrad counterexample",
        code: ADAM_PY,
        output: ADAM_OUT,
      },

      { t: "h", text: "9. L2 regularisation versus decoupled weight decay: different end points" },
      {
        t: "p",
        text: String.raw`Let $\ell(\theta)$ be the data loss. **Adam + L2** runs Adam on $\ell(\theta) + \tfrac{\lambda}{2}\lVert\theta\rVert^2$. Whatever the rescaling does on the way, it stops where $\hat m \to 0$, i.e.`,
      },
      { t: "math", tex: String.raw`\nabla\ell(\theta) + \lambda\theta = 0 \qquad \text{(the minimiser of the L2-regularised loss)}` },
      {
        t: "p",
        text: String.raw`**AdamW** builds $m, v$ from $\nabla\ell$ alone and shrinks the weights separately. It stops where $\eta\,\hat m/(\sqrt{\hat v}+\epsilon) + \eta\lambda\theta = 0$, i.e.`,
      },
      {
        t: "math",
        tex: String.raw`\theta_i = -\frac{1}{\lambda}\,\frac{\hat m_i}{\sqrt{\hat v_i} + \epsilon} \qquad\Longrightarrow\qquad |\theta_i| \lesssim \frac{1}{\lambda}\ \text{ when } |\hat m_i| \le \sqrt{\hat v_i}`,
      },
      {
        t: "p",
        text: String.raw`So AdamW is **not** minimising any L2-regularised loss: at rest, the decay balances the normalised gradient, not the raw one. Since the normalised gradient is at most about 1 per parameter, AdamW acts like a soft fence $|\theta_i| \lesssim 1/\lambda$, plus the same relative shrink for every weight while the gradient signal is weak.`,
      },
      {
        t: "p",
        text: String.raw`Test: logistic regression on the standardised breast-cancer data with $\lambda = 0.05$. Adam + L2 lands on the exact L2 optimum found by scikit-learn ($\lVert\theta\rVert = 1.468$, KKT residual $1.3\times10^{-5}$). AdamW ends at $\lVert\theta\rVert = 47.8$ with its largest weight 19.6 — just under $1/\lambda = 20$ (the data are nearly separable, so the plain loss keeps pushing weights outward). Neither is "wrong": they are different regularisers, and AdamW's has worked better for deep networks (Loshchilov & Hutter, 2019). For SGD the two are identical, because there is no rescaling.`,
      },

      { t: "h", text: "10. Implicit bias and the generalisation debate" },
      {
        t: "p",
        text: String.raw`When a model can fit the training data in many ways, **which** fit the optimizer finds affects test results. A clean example: least squares with 20 equations and 50 unknowns. Gradient descent started at 0 only ever adds combinations of the data rows to $\theta$, so it stays in their span and ends at the **minimum-norm** solution that fits exactly. Adam's per-parameter rescaling leaves that span and finds a different exact fit (in our run: norm 0.648 versus 0.598, distance 0.249 from the minimum-norm solution; code section).`,
      },
      {
        t: "p",
        text: "Wilson, Roelofs, Stern, Srebro and Recht (\"The Marginal Value of Adaptive Gradient Methods in Machine Learning\", NeurIPS 2017) built a linear problem where adaptive methods generalise much worse than SGD, and in their experiments well-tuned SGD with momentum matched or beat them on test error. Since then the evidence has been mixed:",
      },
      {
        t: "list",
        items: [
          "Part of the gap came from coupled L2 regularisation, which AdamW fixed.",
          "Careful tuning studies (e.g. Choi et al., 2019, \"On Empirical Comparisons of Optimizers for Deep Learning\") found that when every knob — including ε — is tuned, Adam-type methods do not lose to momentum SGD.",
          "Transformers are trained almost only with Adam-type optimizers, because SGD trains them poorly.",
        ],
      },
      {
        t: "p",
        text: "The honest summary: on convolutional image classifiers with standard recipes, SGD + momentum often generalises as well as or slightly better than Adam; on Transformers AdamW is the norm; and in any one project the difference is usually smaller than the effect of the learning-rate schedule and regularisation.",
      },

      { t: "h", text: "11. Memory cost of optimizer state" },
      {
        t: "p",
        text: String.raw`For $P$ parameters, SGD stores nothing extra, momentum stores $v$ ($P$ numbers), Adam $m$ and $v$ ($2P$), AMSGrad $3P$ (measured in PyTorch on a 407 050-parameter MLP). In 32-bit floats each copy costs $4P$ bytes. For GPT-2 small ($P$ = 124 439 808, counted from its layer shapes in the framework code), the weights take 0.50 GB, the gradients 0.50 GB and Adam's $m$ and $v$ another 1.00 GB — as much as weights and gradients together.`,
      },
      {
        t: "p",
        text: String.raw`In mixed-precision training the usual count (made popular by the ZeRO paper, Rajbhandari et al., 2020) is 16 bytes per parameter: 2 + 2 for half-precision weights and gradients, 4 + 4 + 4 for the fp32 master weights, $m$ and $v$. For a 7-billion-parameter model that is **112 GB before a single activation is stored** — more than one 80 GB accelerator holds. Hence memory-saving optimizers (Adafactor, 8-bit states, splitting state across devices).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The optimizer race below runs this lesson's exact update rules (PyTorch conventions, β = 0.9, Adam defaults) on 2-D surfaces. Use it to watch the math in action.",
      },
      { t: "sim", id: "optimizers", caption: "Optimizer race: SGD, Momentum, Nesterov, AdaGrad, RMSProp and Adam on the same surface. Click the map to move the start point." },
      {
        t: "list",
        items: [
          String.raw`**Elongated valley, SGD-family η = 0.2.** The curvatures are 0.2 and 4, so $\eta\lambda$ = 0.04 and 0.8. For Momentum both lie in the complex zone $0.0026 < \eta\lambda < 3.79$, so both directions spiral in at rate $\sqrt{0.9} = 0.949$ — watch Momentum overshoot the valley floor and swing back. SGD's flat direction shrinks only by $1 - 0.04 = 0.96$ per step.`,
          String.raw`**Same surface, raise η to about 0.45.** SGD is near its limit $2/4 = 0.5$ and zig-zags. Momentum is still stable (its limit is $3.8/4 = 0.95$). Nesterov's limit is $1.357/4 = 0.34$, so above that it blows up first, as the formula predicts.`,
          String.raw`**Lower η to 0.05.** Now $\eta\lambda$ is 0.01 (flat) and 0.2 (steep). Heavy ball spirals in both directions at $\sqrt{0.9} = 0.949$ per step, while Nesterov brakes the steep direction at $\sqrt{0.9 \times 0.8} = 0.849$, so it overshoots the floor less.`,
          "**Adaptive methods on the valley.** Adam and RMSProp first move diagonally (steps of about η per coordinate). Then RMSProp hovers around the minimum, because its normalised step never shrinks, while Adam's momentum averages out the sign flips and settles. AdaGrad's steps shrink like 1/√t.",
          "**Saddle.** The escape direction has a tiny gradient (y = 0.01). SGD and Momentum need many steps for it to grow; Adam turns it into a full-size step and escapes almost at once — the trick that helps rare features.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "One `Optimizer` class implements every method here with PyTorch's conventions (checked against `torch.optim` to 1e-16 in the framework section). Three experiments, each method using its best learning rate from the same grid: (1) a 10-D quadratic with κ = 100, once lined up with the axes and once randomly rotated; (2) logistic regression on the real breast-cancer data, raw versus standardised features; (3) logistic regression with sparse 0/1 features of very different frequencies.",
      },
      {
        t: "code",
        lang: "python",
        title: "optimizers_deep_dive.py",
        code: SCRATCH_PY,
        output: SCRATCH_OUT,
      },
      { t: "h", text: "Reading the results" },
      {
        t: "list",
        items: [
          String.raw`**Rotation doesn't matter to SGD-type methods.** SGD, heavy ball and Nesterov need exactly the same steps on the aligned and rotated quadratic (184, 105, 64): their updates are linear in the gradient, so the choice of axes is irrelevant. Nesterov beats heavy ball at β = 0.9 thanks to its extra braking (section 5). Momentum gains less than the ideal $\sqrt\kappa$ because β = 0.9 is fixed instead of $\beta^\star = 0.669$.`,
          String.raw`**Rotation does matter to adaptive methods.** AdaGrad needs 7 steps when the curvature directions line up with the axes (its per-axis rescaling then captures the curvature), but 126 when rotated. Adam is much less affected (103 vs 113), because its step follows the gradient's sign more than its size. RMSProp is slowest: with a fixed η its normalised step never shrinks, so it needs a tiny η to meet a tight target — the "hovering" from the simulator.`,
          String.raw`**Badly scaled real data.** The raw features have standard deviations from 0.0023 to 577, a ratio of about 250 000. SGD and momentum must use $\eta \approx 10^{-5}$ to $10^{-6}$ to stay stable on the largest features, and only reach a training loss of 0.24–0.25. The adaptive methods reach 0.17–0.20. But all are far behind **every** optimizer on standardised features (0.032–0.043, test accuracy 95–96%). Adaptivity softens bad scaling; it does not replace preprocessing.`,
          String.raw`**Sparse features.** In one pass over 20 000 examples, frequent features appear about 5 700 times and rare ones about 84 times. AdaGrad recovers the weights best overall (mean error 0.203) and is the best adaptive method on rare features (0.378): its total stays small for them, so their rate stays large. Momentum SGD learns frequent features well (0.176) but leaves rare ones far off (0.514); RMSProp, whose EMA forgets rare features between sightings, also does poorly on them (0.506).`,
        ],
      },
      { t: "h", text: "Weight decay, end points and implicit bias" },
      {
        t: "code",
        lang: "python",
        title: "decay_and_implicit_bias.py",
        code: DECAY_PY,
        output: DECAY_OUT,
      },
      {
        t: "p",
        text: String.raw`Adam + L2 reaches scikit-learn's L2-regularised optimum (same norm 1.468 and largest weight 0.425; the gradient of the regularised objective is $1.3\times10^{-5}$). AdamW does not: its gradient of that objective stays at 2.4 because it solves a different problem, and its largest weight sits just under $1/\lambda = 20$, as section 9 predicts. In the under-determined regression, gradient descent lands exactly on the minimum-norm solution, while Adam fits the data (training MSE $2.4\times10^{-8}$) with a different, larger-norm solution — the implicit-bias effect behind the generalisation debate.`,
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
                text: "This program runs the same 300 mini-batch steps in NumPy and with `torch.optim` in float64. Agreement to about 1e-16 proves this lesson's equations are exactly PyTorch's (including Nesterov's rewritten form, AMSGrad's maximum of the uncorrected v, and AdamW's decay-first order). It then measures optimizer memory, also for a GPT-2-small-sized model.",
              },
              {
                t: "code",
                lang: "python",
                title: "torch_optim_vs_scratch.py",
                code: TORCH_PY,
                output: TORCH_OUT,
              },
              {
                t: "code",
                lang: "python",
                title: "param_groups.py — the production pattern",
                code: PG_PY,
                output: PG_OUT,
              },
              {
                t: "list",
                items: [
                  "`SGD(momentum=β, dampening=0)` keeps the velocity in `state['momentum_buffer']`; the first step sets it to the gradient itself.",
                  "`Adam(weight_decay=λ)` is coupled L2; `AdamW(weight_decay=λ)` is decoupled and multiplies the weights by `1 - lr*λ` before the Adam step.",
                  "`Adam(amsgrad=True)` adds a third state tensor `max_exp_avg_sq`: the maximum of the *uncorrected* v, which is then bias-corrected.",
                  "Optimizer state lives in `opt.state[p]`. Save `opt.state_dict()` with the model checkpoint, or a resumed run restarts m and v from zero (and, with Adam, the bias correction from t = 1).",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "code",
                lang: "python",
                title: "keras_momentum_adaptive.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
y = y.astype("float32")

optimizers = {
    "SGD+momentum": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    "Nesterov":     tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
    "Adagrad":      tf.keras.optimizers.Adagrad(learning_rate=0.1, initial_accumulator_value=0.0),
    "RMSprop":      tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.99),
    "Adam":         tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8),
    "AMSGrad":      tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, amsgrad=True),
    "AdamW":        tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.1, epsilon=1e-8),
}
for name, opt in optimizers.items():
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential([tf.keras.Input(shape=(30,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(X, y, batch_size=32, epochs=10, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"{name:13s} loss={loss:.4f} acc={acc:.3f}")`,
                output: "Prints the training loss and accuracy of each optimizer on the standardised breast-cancer data after 10 epochs; all reach roughly 98% training accuracy, with the momentum and Adam variants converging fastest.",
              },
              {
                t: "list",
                items: [
                  "Keras's `SGD(momentum=β)` uses v ← βv − ηg, θ ← θ + v. This equals PyTorch's form for a constant learning rate; they differ only when η changes, because Keras folds η into v.",
                  "Keras defaults: `epsilon=1e-7` for Adam and RMSprop, and Adagrad starts its total at 0.1 — set them explicitly to reproduce PyTorch numbers.",
                  "`tf.keras.optimizers.AdamW` (TF ≥ 2.11) is decoupled; use `exclude_from_weight_decay(var_names=[...])` to skip biases and normalisation layers.",
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
        head: ["Method", "Key idea", "Advantages", "Disadvantages", "Use cases"],
        rows: [
          ["Heavy ball (Polyak 1964)", String.raw`$\theta_{t+1} = \theta_t - \eta g_t + \beta(\theta_t - \theta_{t-1})$`, String.raw`$\sqrt\kappa$ speed-up on quadratics; larger stability limit $2(1+\beta)/\lambda$`, "no speed-up guarantee on general convex problems; can oscillate", "SGD training of CNNs"],
          ["Nesterov (1983)", "gradient at the look-ahead point", String.raw`extra braking $\sqrt{\beta(1-\eta\lambda)}$; $O(1/t^2)$ on smooth convex problems`, String.raw`smaller stability limit $2(1+\beta)/((1+2\beta)\lambda)$`, "default choice when using SGD + momentum"],
          ["AdaGrad (2011)", String.raw`divide by $\sqrt{\sum g^2}$`, "strong theory for convex online learning; ideal for sparse features", String.raw`rate shrinks like $1/\sqrt t$`, "sparse linear models, embedding tables"],
          ["RMSProp (2012)", String.raw`divide by $\sqrt{\mathrm{EMA}(g^2)}$`, "adaptive rate that doesn't shrink", "no bias correction; hovers without decay", "RNNs and RL historically"],
          ["Adam (2015)", "EMA momentum ÷ RMS, bias-corrected", "robust defaults; scale-free step ≈ η·SNR", "2 state copies; can fail to converge (AMSGrad example)", "general default"],
          ["AMSGrad (2018)", String.raw`use $\max_s v_s$`, "rate never increases; fixes the counterexamples", "3 state copies; rarely better in practice", "when Adam's loss oscillates late in training"],
          ["AdamW (2019)", "decoupled shrinking", "weight decay independent of the rescaling", "a different regulariser from L2", "Transformers, LLMs, ViTs"],
          ["Adamax (Kingma & Ba 2015)", String.raw`$\ell_\infty$ norm: $u_t = \max(\beta_2 u_{t-1}, |g_t|)$`, "no bias correction needed for u; bounded step", "rarely better than Adam", "embeddings, occasionally"],
          ["Nadam (Dozat 2016)", "Nesterov momentum inside Adam", "sometimes slightly faster early on", "small gains", "drop-in Adam replacement"],
          ["RAdam (Liu et al. 2020)", "turns off the adaptive part while v is still poorly estimated", "removes the need for warm-up in some settings", "extra complexity", "when warm-up is hard to tune"],
          ["Adafactor (Shazeer & Stern 2018)", "stores row and column averages instead of a full second moment for matrices", "much less memory for v", "less precise per-parameter scaling", "very large models (e.g. T5)"],
          ["Lion (Chen et al. 2023)", "update = sign of a blended momentum", "1 state copy; sign steps", String.raw`needs smaller $\eta$ and larger $\lambda$ than AdamW`, "large vision and language models"],
          ["8-bit optimizers (Dettmers et al. 2022)", "store m and v in 8 bits, block by block", "~4× less optimizer memory", "harder to implement", "fine-tuning large models on limited memory"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "table",
        head: ["Domain", "Common recipe", "Why"],
        rows: [
          ["Image classification CNNs (from scratch)", String.raw`SGD + momentum 0.9 (often Nesterov), $\eta = 0.1$ per 256 examples of batch, weight decay $10^{-4}$ (ImageNet ResNets) to $5\times10^{-4}$ (CIFAR), step or cosine schedule`, "Well understood, generalises well; BatchNorm keeps the problem fairly evenly shaped, so adaptivity matters less."],
          ["Transformers / LLM pre-training", String.raw`AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1, warm-up then cosine decay to about 10% of the peak, gradient clipping at 1.0 (the GPT-3 and LLaMA papers report settings of this form)`, String.raw`Gradient sizes differ hugely between embeddings, attention and MLP weights; a shorter $\beta_2$ memory reacts to loss spikes.`],
          ["Fine-tuning pretrained Transformers", String.raw`AdamW, $\eta$ between $10^{-5}$ and $10^{-4}$ (BERT's paper searched 2e-5 to 5e-5), short linear warm-up and decay, few epochs`, "The pretrained weights are already good; Adam's η is a per-parameter step length, so it must be small not to wreck them."],
          ["GANs", String.raw`Adam with $\beta_1 = 0.5$ (or 0), $\eta = 2\times10^{-4}$ (DCGAN)`, "The goal shifts as the opponent trains; a long momentum memory follows out-of-date directions."],
          ["Reinforcement learning", String.raw`Adam, $\eta \approx 3\times10^{-4}$ (PPO's MuJoCo setting), gradient clipping`, "Moving targets, very noisy gradients."],
          ["Recommender systems / sparse embeddings", "AdaGrad or sparse Adam variants (`torch.optim.SparseAdam`) for embedding tables; a dense optimizer for the rest", "Rare items' rows get gradients rarely; per-row totals give them well-sized, larger steps, and only touched rows need updating."],
          ["Small tabular / quick baselines", String.raw`Adam, $\eta = 10^{-3}$`, "Works with almost no tuning; the problem is small, so memory doesn't matter."],
          ["Memory-limited training of huge models", "AdamW with sharded state (ZeRO/FSDP), Adafactor, or 8-bit states", "Adam's two fp32 copies dominate memory at billions of parameters."],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          String.raw`**Momentum: speed versus stability.** It turns $\kappa$ into $\sqrt\kappa$ on quadratics and raises the stability limit to $2(1+\beta)/\lambda$, but the effective step $\eta/(1-\beta)$ is 10× larger. Too much momentum ($\beta \to 1$) means too little friction: rate $\sqrt\beta \to 1$.`,
          "**Nesterov: braking versus range.** Faster on steep directions at its best η, but a stability limit almost 3× lower than heavy ball's for β = 0.9.",
          "**Adaptivity works per axis.** It fixes scale differences along the axes, not rotated unevenness (126 vs 7 steps for AdaGrad). Normalisation layers and good parameterisation still matter.",
          "**Adaptivity roughly ignores scale.** Robust to badly scaled losses and layers, but Adam keeps taking η-sized steps on parameters whose gradients are tiny noise — hence ε and a decaying η.",
          "**Memory.** 0, 1, 2 and 3 state copies for SGD, momentum, Adam and AMSGrad.",
          "**Generalisation is an empirical question.** Different optimizers find different solutions. Tune before you conclude.",
          "**Tuning sensitivity.** Adam's defaults work surprisingly often; SGD needs η and its schedule tuned with care. But Adam's robustness can hide a poor η — tune it too.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          String.raw`**Turning on momentum without lowering η.** The steady step becomes $\eta/(1-\beta)$: 10× for $\beta = 0.9$, 100× for 0.99. Fix: divide η by about $1/(1-\beta)$ to start, then re-tune.`,
          String.raw`**Using β = 0.99 "for more speed".** On the quadratic every complex-zone direction then shrinks at only $\sqrt{0.99} = 0.995$ per step; the exercise shows it is far slower than β = 0.9. Momentum should match the problem's shape ($\beta^\star$ from $\kappa$), not be maxed out.`,
          String.raw`**Assuming Nesterov is "momentum, but always better".** Its stability limit is lower ($1.357/\lambda$ vs $3.8/\lambda$ at β = 0.9). A learning rate that is fine for heavy ball can make Nesterov blow up.`,
          String.raw`**Dropping Adam's bias correction "because it only matters early".** For $\beta_2 = 0.999$ the bias is above 1% for the first 4603 steps. Without correction $\sqrt{v}$ is too small and early steps are too big ($3.16\eta$ at $t = 1$).`,
          String.raw`**Believing Adam's step can never exceed η.** After a quiet period, a gradient burst gives about $3\eta$ (and up to $7.27\eta$ in the worst case for the defaults). Fix: gradient clipping, warm-up, or a smaller $\beta_2$ (0.95–0.98) for spiky problems.`,
          String.raw`**Treating ε as a harmless detail.** It sets the gradient size below which Adam stops normalising. In half precision, $10^{-8}$ can round to zero; for tiny-gradient parameters, a larger ε changes training a lot.`,
          "**Using `Adam(weight_decay=λ)` and calling it AdamW.** That is L2: weights with very noisy gradients are barely regularised. Fix: use `AdamW`, and skip biases and normalisation parameters.",
          String.raw`**Reusing λ across Adam + L2, AdamW and SGD.** They are different regularisers with different end points. AdamW's shrink per step is $\eta\lambda$, so changing η changes the real decay. Re-tune λ whenever the optimizer or η changes.`,
          "**Not saving optimizer state in checkpoints.** Resuming Adam with empty m, v and t = 0 causes a burst of η-sized steps on every parameter. Fix: save and load `optimizer.state_dict()`.",
          "**Expecting Adam to fix badly scaled inputs.** It helps (loss 0.17 vs 0.25 on raw breast-cancer features), but standardising the inputs helped every optimizer far more (0.035).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What does the momentum coefficient β control?", a: String.raw`β sets how long the optimizer remembers past gradients. The velocity is a weighted sum with weights $\beta^k$, so the memory is about $1/(1-\beta)$ steps (10 for β = 0.9). Where gradients agree, the velocity grows to $g/(1-\beta)$, so the effective learning rate becomes $\eta/(1-\beta)$. Where gradients flip sign, they cancel and the gain drops to $1/(1+\beta)$. In the physical picture, $1-\beta$ is the friction on the rolling ball.` },
      { t: "qa", level: "beginner", q: "Why does Adam need bias correction?", a: String.raw`Its running averages start at zero. After $t$ steps the weights on the real gradients add up to $1-\beta^t$, not 1; the rest sits on the fake zero start, so the averages are too small. Dividing by $1-\beta^t$ fixes the weights so the estimate is a proper weighted average. This matters most for $v$: with $\beta_2 = 0.999$ the bias is above 1% for the first 4603 steps, and without correction the first steps would be about 3× too large.` },
      { t: "qa", level: "beginner", q: "Why are adaptive methods good for sparse features?", a: String.raw`AdaGrad divides each parameter's step by the square root of its total squared gradients. A rare feature only gets gradients when it appears, so its total stays small and its learning rate stays large — roughly $1/\sqrt{n_j}$, where $n_j$ is how often it has been seen. So each rare sighting moves its weight a lot, while frequent features get small, careful steps. One shared learning rate cannot do both.` },
      { t: "qa", level: "intermediate", q: "Derive the optimal heavy-ball momentum for a quadratic with condition number κ.", a: String.raw`For each eigen-direction with curvature λ, the error follows $e_{t+1} = (1+\beta-\eta\lambda)e_t - \beta e_{t-1}$, with polynomial $r^2 - (1+\beta-\eta\lambda)r + \beta$. The roots are complex when $(1-\sqrt\beta)^2 < \eta\lambda < (1+\sqrt\beta)^2$, and then $|r| = \sqrt\beta$ whatever λ is. Choose η and β so that $[\eta\mu, \eta L]$ exactly fills this zone: $\sqrt{\eta\mu} = 1-\sqrt\beta$ and $\sqrt{\eta L} = 1+\sqrt\beta$. That gives $\eta = 4/(\sqrt L + \sqrt\mu)^2$ and $\sqrt\beta = (\sqrt\kappa - 1)/(\sqrt\kappa+1)$. The rate is $(\sqrt\kappa-1)/(\sqrt\kappa+1)$, versus $(\kappa-1)/(\kappa+1)$ for gradient descent: for κ = 100, 69 instead of 691 steps per factor $10^6$.` },
      { t: "qa", level: "intermediate", q: "How does Nesterov momentum differ from heavy ball on a quadratic?", a: String.raw`Nesterov's per-direction recursion is $e_{t+1} = (1+\beta)(1-\eta\lambda)e_t - \beta(1-\eta\lambda)e_{t-1}$. Its roots multiply to $\beta(1-\eta\lambda)$ instead of β, so in the wobbling zone it shrinks at $\sqrt{\beta(1-\eta\lambda)}$ — steep directions are braked more than with heavy ball, which reduces overshoot. The price is a smaller stable range: $\eta\lambda < 2(1+\beta)/(1+2\beta)$ (1.357 for β = 0.9) instead of $2(1+\beta)$ (3.8). PyTorch uses the rewritten form $\phi \leftarrow \phi - \eta(g + \beta v)$, which only needs the gradient at the stored weights.` },
      { t: "qa", level: "intermediate", q: "Explain Adam's update as a signal-to-noise ratio. What happens near a minimum?", a: String.raw`If a parameter's gradient has mean $\mu_g$ and variance $\sigma^2$, then $\hat m \approx \mu_g$ and $\hat v \approx \mu_g^2 + \sigma^2$, so the step is $\eta\,\mu_g/\sqrt{\mu_g^2+\sigma^2}$. Far from a minimum, gradients are consistent and the step is about $\pm\eta$ (sign descent). Near a minimum, the mean gradient is small compared with mini-batch noise, so the step shrinks in line with $\mu_g/\sigma$ — Adam cools itself down. It never fully stops because m keeps jittering, which is why a decaying learning rate is still used.` },
      { t: "qa", level: "intermediate", q: "What is the role of ε in Adam, and when would you change it?", a: String.raw`The step is $\eta\hat m/(\sqrt{\hat v}+\epsilon)$. For gradients much larger than ε it does nothing. For gradients near ε it weakens the normalisation (step $\eta|g|/(|g|+\epsilon)$ for a constant gradient). For gradients far below ε, Adam becomes momentum SGD with learning rate $\eta/\epsilon$. Raising ε (e.g. to 1e-6, as RoBERTa did, or much larger) steadies training when $\hat v$ is tiny or poorly estimated, makes Adam more SGD-like, and avoids 1e-8 rounding to zero in low precision.` },
      { t: "qa", level: "advanced", q: "Why did Reddi et al. propose AMSGrad, and what does it change?", a: String.raw`They showed that Adam's effective rate $\eta/\sqrt{\hat v_t}$ can grow over time, because the EMA of squared gradients forgets large but rare gradients within about $1/(1-\beta_2)$ steps. In their online example (loss $Cx$ every third step, $-x$ otherwise, $C = 3$, $\beta_2 = 0.1$), the rare large gradient is normalised to about 1.05 while the two small ones get −0.76 and −0.97. The net move goes the wrong way and Adam ends at the worst point. AMSGrad divides by the running maximum of $v_t$, so the rate can never increase, which restores convergence. It costs a third state tensor and in deep learning rarely beats Adam with a decaying learning rate.` },
      { t: "qa", level: "advanced", q: "Adam + L2 and AdamW: are they minimising the same objective?", a: String.raw`No. Adam + L2 runs Adam on $\ell + \tfrac\lambda2\lVert\theta\rVert^2$, so it stops where $\nabla\ell + \lambda\theta = 0$ — the L2-regularised optimum (it matches scikit-learn's answer numerically). AdamW stops where $\hat m/(\sqrt{\hat v}+\epsilon) = -\lambda\theta$: the decay balances the normalised gradient, which is at most about 1. So it acts like a soft fence $|\theta_i| \lesssim 1/\lambda$ plus the same relative shrink for all weights. On nearly separable logistic regression with λ = 0.05, Adam + L2 ends at norm 1.47 while AdamW ends at 47.8 with weights just below 20. For SGD the two are identical.` },
      { t: "qa", level: "advanced", q: "How much memory does Adam need for a 7B-parameter model, and what can be done about it?", a: String.raw`Adam keeps two extra tensors per parameter. In fp32 that alone is $2 \times 4 \times 7\times10^9 = 56$ GB. In standard mixed precision each parameter costs 2 (bf16 weights) + 2 (bf16 gradients) + 4 (fp32 master weights) + 8 (fp32 m and v) = 16 bytes, i.e. 112 GB before activations. Fixes: split optimizer state across data-parallel workers (ZeRO / FSDP), factor the second moment (Adafactor), store the states in 8 bits, or use an optimizer with less state (SGD + momentum, or Lion with one copy).` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "An EMA uses β = 0.99. What is the variance of the EMA of independent unit-variance noise, (1−β)/(1+β)? (4 decimals)", answer: 0.005, tolerance: 0.0002, explain: "0.01/1.99 = 0.005025 — like averaging about 199 samples." },
          { kind: "numeric", q: "How many steps until the bias factor β₂ᵗ of Adam's second moment falls below 0.5, for β₂ = 0.999? (round to the nearest step)", answer: 693, tolerance: 1, explain: "t > ln 0.5 / ln 0.999 = 692.8, so t = 693." },
          { kind: "numeric", q: "Heavy ball on a quadratic with μ = 1 and L = 25. What is the optimal β* = ((√κ − 1)/(√κ + 1))²?", answer: 0.4444, tolerance: 0.001, explain: "√κ = 5, (4/6)² = 0.4444; the rate is 2/3 per step versus 24/26 = 0.923 for gradient descent." },
          { kind: "numeric", q: "For the same problem (μ = 1, L = 25), what is the optimal heavy-ball learning rate η* = 4/(√L + √μ)²?", answer: 0.1111, tolerance: 0.001, explain: "4/(5 + 1)² = 4/36 = 0.1111." },
          { kind: "numeric", q: "With β = 0.9, what is the largest ηλ for which Nesterov's method is stable on a quadratic direction, 2(1+β)/(1+2β)?", answer: 1.357, tolerance: 0.002, explain: "3.8/2.8 = 1.357, versus 2(1+β) = 3.8 for heavy ball." },
          { q: "Momentum with β = 0.9 gets a gradient that flips sign every step. Compared with a constant gradient of the same size, the settled velocity is…", options: ["the same", "about 19× smaller", "about 10× smaller", "zero"], answer: 1, explain: "Gains are 1/(1−β) = 10 at ω = 0 and 1/(1+β) = 0.526 at ω = π; the ratio is (1+β)/(1−β) = 19." },
          { q: "Which statement about AdaGrad and a 45°-rotated, unevenly shaped quadratic is correct?", options: ["Rotation does not matter for AdaGrad", "AdaGrad's per-axis rescaling cannot capture rotated curvature, so it loses most of its advantage", "AdaGrad diverges on rotated problems", "AdaGrad becomes Newton's method"], answer: 1, explain: "AdaGrad rescales each coordinate on its own. In the experiment it needed 7 steps on the aligned problem and 126 on the rotated one; SGD and momentum don't care about rotation." },
          { q: "Adam's steady gradient for a parameter has mean 1 and standard deviation 1. Its average step is closest to…", options: ["η", "0.71 η", "0.5 η", "0.1 η"], answer: 1, explain: "μ/√(μ²+σ²) = 1/√2 ≈ 0.707; the simulation gave 0.704." },
          { q: "Where AdamW (weight decay λ) comes to rest, which relation holds?", options: ["∇ℓ(θ) + λθ = 0", "m̂/(√v̂+ε) = −λθ", "θ = 0", "v̂ = λ"], answer: 1, explain: "The decay balances the normalised gradient, so |θᵢ| is at most about 1/λ; Adam + L2 instead satisfies ∇ℓ + λθ = 0." },
        ],
      },
      {
        t: "exercise",
        title: "Numerical: momentum regimes by hand",
        prompt: "A quadratic direction has curvature λ = 4, and you use heavy-ball momentum with β = 0.81 and η = 0.05. (a) Compute ηλ and decide whether the roots of r² − (1+β−ηλ)r + β are real or complex. (b) Give the shrink factor per step. (c) Is the method stable for this direction, and up to which η would it stay stable? (d) Repeat (a)–(b) for Nesterov with the same η and β.",
        hint: "Complex if (1−√β)² < ηλ < (1+√β)². Stability: ηλ < 2(1+β) for heavy ball.",
        solution: [
          { t: "p", text: String.raw`(a) $\eta\lambda = 0.2$. With $\sqrt\beta = 0.9$: $(1-0.9)^2 = 0.01$ and $(1+0.9)^2 = 3.61$. Since $0.01 < 0.2 < 3.61$, the roots are complex. Check: the polynomial is $r^2 - 1.61 r + 0.81$, discriminant $1.61^2 - 3.24 = 2.5921 - 3.24 = -0.6479 < 0$.` },
          { t: "p", text: String.raw`(b) $|r| = \sqrt{\beta} = 0.9$ per step (the error spirals in, losing 10% per step). Plain gradient descent with the same η would shrink it by $|1 - 0.2| = 0.8$ — faster for this steep direction; momentum pays off on the flat ones.` },
          { t: "p", text: String.raw`(c) Stable if $\eta\lambda < 2(1+\beta) = 3.62$, i.e. $\eta < 3.62/4 = 0.905$.` },
          { t: "p", text: String.raw`(d) Nesterov: polynomial $r^2 - (1+\beta)(1-\eta\lambda) r + \beta(1-\eta\lambda) = r^2 - 1.448 r + 0.648$. Discriminant $1.448^2 - 4(0.648) = 2.0967 - 2.592 = -0.4953 < 0$, complex, so $|r| = \sqrt{0.648} = 0.805$. Its stability limit is $\eta\lambda < 2(1.81)/(2.62) = 1.382$, i.e. $\eta < 0.345$.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: tune heavy ball for κ = 1000 and check the theory",
        prompt: "Write a function that returns the optimal (η, β) for heavy ball given μ and L. On a quadratic with 50 curvatures spread evenly over [1, 1000] and start x₀ = (1, …, 1), compare after 300 and 1000 steps: gradient descent with η = 2/(μ+L), heavy ball with the optimal pair, and heavy ball with β fixed at 0.9 and at 0.99 (pick the best η on a grid by the 300-step result). Explain the ranking.",
        hint: "Use the PyTorch form v ← βv + λx, x ← x − ηv; the gradient of ½Σλᵢxᵢ² is λ ⊙ x.",
        solution: [
          { t: "code", lang: "python", title: "tune_heavy_ball.py", code: EX_PY, output: EX_OUT },
          {
            t: "p",
            text: String.raw`Gradient descent's rate $999/1001 = 0.998$ leaves the error at 0.78 after 300 steps. The optimal pair ($\eta^\star = 0.00376$, $\beta^\star = 0.881$) has long-run rate 0.9387 and is best after 1000 steps ($6.5\times10^{-25}$). After only 300 steps, β = 0.9 with a tuned η is slightly ahead: the optimal choice puts the extreme curvatures exactly on a double root, whose error shrinks like $t\,\rho^t$ instead of $\rho^t$, and that extra factor $t$ matters over short runs. β = 0.99 has far too little friction: every wobbling direction shrinks only by $\sqrt{0.99} = 0.995$ per step, barely better than gradient descent. More momentum is not better; the right momentum depends on κ.`,
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`EMA $m_t = \beta m_{t-1} + (1-\beta)g_t$: window $\approx 1/(1-\beta)$, mean age $\beta/(1-\beta)$, noise variance factor $(1-\beta)/(1+\beta)$.`,
          String.raw`Bias correction $\hat m_t = m_t/(1-\beta^t)$ fixes the weights of a zero-started EMA; for $\beta_2 = 0.999$ it matters for thousands of steps.`,
          String.raw`Momentum is a low-pass filter: gain $1/(1-\beta)$ for constant gradients, $1/(1+\beta)$ for flipping ones; equally, a heavy ball with friction $1-\beta$.`,
          String.raw`Heavy ball on a quadratic: rate $\sqrt\beta$ for $(1-\sqrt\beta)^2 < \eta\lambda < (1+\sqrt\beta)^2$, stable only if $\eta\lambda < 2(1+\beta)$; best $\eta^\star = 4/(\sqrt L+\sqrt\mu)^2$, $\beta^\star = ((\sqrt\kappa-1)/(\sqrt\kappa+1))^2$, rate $(\sqrt\kappa-1)/(\sqrt\kappa+1)$.`,
          String.raw`Nesterov: rate $\sqrt{\beta(1-\eta\lambda)}$ (extra braking on steep directions), stable only if $\eta\lambda < 2(1+\beta)/(1+2\beta)$; PyTorch form $\phi \leftarrow \phi - \eta(g + \beta v)$.`,
          String.raw`AdaGrad: per-parameter rate $\eta/\sqrt{\sum g^2}$ — $\propto 1/\sqrt{n_j}$ for sparse features, $\propto 1/\sqrt t$ for dense ones. RMSProp: $\eta/\mathrm{RMS}_{\text{recent}}(g)$.`,
          String.raw`Adam's step $\approx \eta\,\mu_g/\sqrt{\mu_g^2+\sigma^2}$: sign-like when the signal is strong, shrinking with SNR; worst case $\approx 7.27\eta$ for default betas; ε sets where normalisation stops.`,
          String.raw`AMSGrad uses $\max_s v_s$ so the rate never increases; fixes Reddi et al.'s counterexamples.`,
          String.raw`Adam + L2 ends at $\nabla\ell + \lambda\theta = 0$; AdamW at $\hat m/\sqrt{\hat v} = -\lambda\theta$ (soft fence $|\theta_i| \lesssim 1/\lambda$).`,
          "Optimizers have implicit biases (GD → minimum-norm solution); SGD-vs-Adam generalisation gaps are empirical and depend on the task.",
          "State memory: momentum 1×, Adam 2×, AMSGrad 3× parameters; mixed-precision Adam ≈ 16 bytes/parameter (112 GB for 7B).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — everything here assumed a fixed η; warm-up and decay make Adam stable at the start and precise at the end.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — puts weight decay and optimizers' implicit bias next to the other tools against overfitting.",
          "[Batch, Layer and Group Normalization](/learn/m07/normalization) — normalisation evens out the problem's shape, which limits every optimizer here, including the rotated unevenness adaptive methods cannot fix.",
          "[Weight Initialization](/learn/m07/initialization) — sets the size of the first gradients, which decides how big early Adam steps and momentum velocities are.",
          "[Gradient Descent and Its Descendants](/learn/m06/gradient-descent) — revisit the update rules and the two-step worked example with this lesson's theory in mind.",
        ],
      },
    ],
  },
};

export default lesson;
