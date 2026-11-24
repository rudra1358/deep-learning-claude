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
    "The theory behind the optimizer update rules: exponential moving averages and bias correction, momentum as a low-pass filter and as a heavy ball, the exact quadratic analysis that gives the optimal momentum for a condition number, Nesterov's extra damping, why per-parameter step sizes help sparse and badly scaled problems, Adam as a signal-to-noise-normalised step, AMSGrad, L2 versus decoupled weight decay, the SGD-vs-Adam generalisation debate, optimizer memory and a practical recipe per domain.",
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
        text: "The lesson [Gradient Descent and Its Descendants](/learn/m06/gradient-descent) gives the update equations of Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW and Nadam, with a two-step worked example of each. This lesson assumes those equations and asks the questions that come next: **why** do they work, **when** do they fail, and **how** should their hyperparameters be chosen? Every answer is derived on a model problem and then checked numerically.",
      },
      {
        t: "p",
        text: "Two ideas carry almost all modern optimizers. The first is **averaging over time**: momentum and Adam's first moment are exponential moving averages (EMAs) of gradients, which smooth out noise and accumulate speed along consistent directions. The second is **per-parameter normalisation**: AdaGrad, RMSProp and Adam divide each coordinate's step by a running estimate of that coordinate's gradient size, which equalises progress across parameters whose gradients differ by orders of magnitude. Polyak introduced heavy-ball momentum in 1964, Nesterov his accelerated method in 1983; Duchi, Hazan and Singer published AdaGrad in 2011; Kingma and Ba published Adam in 2015; Reddi, Kale and Kumar identified a convergence flaw in Adam and proposed AMSGrad in 2018; Loshchilov and Hutter's AdamW (2019) fixed the interaction with weight decay.",
      },
      {
        t: "p",
        text: "These optimizers are used everywhere: SGD with momentum still trains many image models, AdamW trains essentially every large Transformer, AdaGrad-style methods remain common for sparse embedding tables in recommender systems. Choosing between them, and setting their four or five hyperparameters, is one of the most consequential decisions in a training run — and one where folklore is common. The goal here is to replace folklore with mechanisms you can derive.",
      },
      {
        t: "callout",
        kind: "key",
        title: "What this lesson adds",
        text: String.raw`EMAs in depth (window $1/(1-\beta)$, variance reduction, bias correction as weight renormalisation); momentum as a low-pass filter and a discretised heavy ball; the eigen-analysis of momentum on a quadratic, giving $\beta^\star = \big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$ and rate $\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$; Nesterov's extra damping and smaller stability limit; AdaGrad's decaying rate and why it suits sparse features; Adam's step as a signal-to-noise ratio, its bounds and the role of $\epsilon$; AMSGrad; the different fixed points of Adam + L2 and AdamW; implicit bias and the generalisation debate; optimizer memory; a recipe per domain.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Gradient Descent and Its Descendants](/learn/m06/gradient-descent) — **essential**. You need the update equations of all optimizers, the result that gradient descent on a quadratic converges iff $0 < \eta < 2/\lambda_{\max}$, and the definition of the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.`,
          String.raw`[Eigenvalues and the SVD](/learn/m02/eigen-svd) — a symmetric matrix $H = Q\Lambda Q^\top$ decouples a quadratic into independent one-dimensional problems; the spectral radius (largest absolute eigenvalue) of an iteration matrix decides whether a linear recursion converges.`,
          String.raw`[Optimization math](/learn/m02/optimization-math) — convexity, smoothness and the Hessian as curvature.`,
          "[Backpropagation](/learn/m04/backpropagation) — the gradients that every optimizer consumes.",
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
        text: "A car's suspension lets the body follow the slow shape of the road (a hill, a bend) while ignoring the fast bumps. Momentum does the same to the gradient sequence: the slow component — the direction the loss keeps decreasing in — passes through and even gets amplified, while the fast component — the sign-flipping oscillation across a narrow valley, or mini-batch noise — is filtered out. An adaptive optimizer is more like a headset with automatic gain control on every channel: loud channels are turned down and quiet ones turned up, so every parameter moves at a comparable pace regardless of how large its raw gradients are.",
      },
      {
        t: "p",
        text: "Think of an elongated valley: steep walls across, a gentle slope along the floor. Plain gradient descent must take steps small enough not to bounce off the walls, so it crawls along the floor. A heavy ball rolling down the same valley keeps its speed along the floor (each push adds to the last) while its sideways sloshing is damped because the sideways pushes alternate in direction and cancel. That is the entire geometric content of momentum — and the quadratic analysis below turns it into exact numbers: with the right settings the number of steps scales with $\\sqrt{\\kappa}$ instead of $\\kappa$.",
      },
      {
        t: "p",
        text: "Adaptive methods attack a different problem. In a network, the gradient for an embedding row of a rare word is zero almost always; the gradient for a bias in the last layer is large on every step. One global learning rate cannot suit both. Dividing each parameter's gradient by its own typical magnitude makes the step size roughly the same for every parameter — Adam's step is close to $\\pm\\eta$ per coordinate when the gradient's sign is consistent, and automatically shrinks when the gradient is mostly noise.",
      },
      {
        t: "list",
        items: [
          String.raw`**EMA** — a running average that forgets old values geometrically; $\beta = 0.9$ remembers roughly the last 10 values, $\beta = 0.999$ the last 1000.`,
          "**Low-pass filter** — something that passes slowly varying signals and attenuates quickly varying ones.",
          "**Preconditioning** — rescaling the gradient (here: per coordinate) so that the problem looks better conditioned to the optimizer.",
          "**Decoupled weight decay** — shrinking the weights directly instead of through the gradient, so the adaptive rescaling does not distort it.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "One Adam(W) step, with every piece of state" },
      {
        t: "steps",
        items: [
          { title: "Receive the gradient", text: String.raw`Backprop produces $g_t$ for every parameter (after optional gradient clipping). The step counter $t$ is incremented: $t = 1$ on the first update.` },
          { title: "Decoupled weight decay (AdamW only)", text: String.raw`PyTorch applies it first, to the current weights: $\theta \leftarrow \theta\,(1 - \eta\lambda)$. In Adam with weight_decay (L2), instead $\lambda\theta$ is added to $g_t$ here.` },
          { title: "Update the first moment", tex: String.raw`m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t` },
          { title: "Update the second moment", tex: String.raw`v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2` },
          { title: "AMSGrad only: keep the running maximum", tex: String.raw`v^{\max}_t = \max(v^{\max}_{t-1},\, v_t)\ \text{ is used instead of } v_t` },
          { title: "Bias-correct both moments", tex: String.raw`\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}` },
          { title: "Normalised step", tex: String.raw`\theta \leftarrow \theta - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}` },
        ],
      },
      {
        t: "p",
        text: String.raw`All operations are element-wise, so the cost is a handful of vector operations per parameter. The state is $m$ and $v$ (two tensors the size of the model), plus $v^{\max}$ for AMSGrad, plus the scalar $t$.`,
      },
      { t: "h", text: "Heavy ball: two equivalent ways to write momentum" },
      {
        t: "p",
        text: String.raw`PyTorch's form $v_t = \beta v_{t-1} + g_t,\ \theta_{t+1} = \theta_t - \eta v_t$ can be rewritten without $v$. Since $\theta_t - \theta_{t-1} = -\eta v_{t-1}$,`,
      },
      { t: "math", tex: String.raw`\theta_{t+1} = \theta_t - \eta\, g_t + \beta\,(\theta_t - \theta_{t-1})`, caption: "Polyak's heavy-ball form: a gradient step plus a fraction β of the previous displacement." },
      {
        t: "p",
        text: "This is the form used in every theoretical analysis below. The last term is literally inertia: the parameters keep moving in the direction they were already moving, scaled down by β (friction).",
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
        text: "The final line only needs the gradient at the stored parameters, so it fits the standard `loss.backward(); opt.step()` loop. It is exactly what `torch.optim.SGD(..., momentum=β, nesterov=True)` implements.",
      },
      { t: "h", text: "What each hyperparameter controls" },
      {
        t: "table",
        head: ["Knob", "Controls", "Typical value", "Raise it when", "Lower it when"],
        rows: [
          [String.raw`$\beta$ (momentum)`, String.raw`averaging window $\approx 1/(1-\beta)$; effective step $\eta/(1-\beta)$`, "0.9", "very ill-conditioned problem, noisy gradients", "training oscillates or diverges; very short runs"],
          [String.raw`$\beta_1$ (Adam)`, "smoothing of the gradient direction", "0.9", "noisy gradients", String.raw`GANs and other non-stationary games (0.5 or 0 is common)`],
          [String.raw`$\beta_2$ (Adam)`, String.raw`window of the squared-gradient estimate, $\approx 1/(1-\beta_2)$ steps`, "0.999 (0.95–0.98 for large Transformers)", "gradients are very noisy and stationary", "loss spikes from stale second moments; large-batch Transformer training"],
          [String.raw`$\epsilon$`, String.raw`floor on the denominator; caps the step at $\eta\,|\hat m|/\epsilon$`, String.raw`$10^{-8}$`, "updates are too jumpy for tiny-gradient parameters, low-precision arithmetic", "almost never"],
          [String.raw`$\lambda$ (weight decay)`, String.raw`shrinkage per step $\eta\lambda$`, "0.01–0.1 (AdamW), 1e-4–5e-4 (SGD)", "overfitting", "underfitting; never on biases and normalisation gains"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Exponential moving averages" },
      {
        t: "math",
        tex: String.raw`m_t = \beta\, m_{t-1} + (1-\beta)\, g_t,\quad m_0 = 0 \qquad\Longrightarrow\qquad m_t = \sum_{k=0}^{t-1} (1-\beta)\beta^{k}\, g_{t-k}`,
        caption: String.raw`$g_t$ — the value being averaged (a gradient or a squared gradient); $\beta \in [0,1)$ — decay; $k$ — age of a value in steps.`,
      },
      {
        t: "p",
        text: String.raw`**Effective window.** The weight on a value of age $k$ is $w_k = (1-\beta)\beta^k$. For an infinitely long history the weights sum to 1, and the mean age is $\sum_k k\,w_k = \beta/(1-\beta)$ (9 steps for $\beta = 0.9$). After $1/(1-\beta)$ steps a weight has fallen by $\beta^{1/(1-\beta)} \approx e^{-1}$, because $\ln \beta \approx -(1-\beta)$. Hence the rule of thumb: an EMA averages over roughly the last $1/(1-\beta)$ values — 10, 100 and 1000 for $\beta = 0.9, 0.99, 0.999$.`,
      },
      {
        t: "p",
        text: String.raw`**Noise reduction.** If the $g_t$ are independent with variance $\sigma^2$, the stationary variance of the EMA is $\sigma^2(1-\beta)^2\sum_k \beta^{2k} = \sigma^2\,\frac{1-\beta}{1+\beta}$. A plain average of $N$ samples has variance $\sigma^2/N$, so an EMA is as good as averaging $N_{\text{eff}} = \frac{1+\beta}{1-\beta}$ samples — 19 for $\beta = 0.9$, about twice the window.`,
      },
      {
        t: "p",
        text: String.raw`**Bias of the zero start.** With $m_0 = 0$, only $t$ values have been seen, and their weights sum to $\sum_{k=0}^{t-1}(1-\beta)\beta^k = 1 - \beta^t < 1$. The missing weight $\beta^t$ sits on the fictitious initial value 0, pulling the average towards zero. Dividing by $1 - \beta^t$ **renormalises the weights to sum to one**:`,
      },
      {
        t: "math",
        tex: String.raw`\hat m_t = \frac{m_t}{1-\beta^t} = \frac{\sum_{k=0}^{t-1} \beta^k g_{t-k}}{\sum_{k=0}^{t-1}\beta^k}`,
        caption: "Bias correction turns the zero-started EMA into an exactly normalised weighted average of the values seen so far.",
      },
      {
        t: "p",
        text: String.raw`This is the cleanest way to understand Adam's correction: it needs no assumption about stationarity (the expectation argument $\mathbb{E}[m_t] = (1-\beta^t)\mathbb{E}[g]$ from the previous lesson is a consequence). The correction matters for about as long as $\beta^t$ is not negligible: $\beta^t < 0.01$ requires $t > \ln 0.01/\ln\beta$, i.e. **44** steps for $\beta = 0.9$ but **4603** steps for $\beta_2 = 0.999$. Adam's second moment is biased for thousands of steps, which is why the correction is not a cosmetic detail.`,
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
        text: String.raw`The raw EMA (0.2, 0.58, 0.822) is nowhere near the data (2, 4, 3); the corrected values are exactly the weighted averages. These numbers are printed by the verification program below.`,
      },

      { t: "h", text: "2. Momentum as a low-pass filter" },
      {
        t: "p",
        text: String.raw`Treat the gradient sequence as a signal and feed it a pure oscillation $g_t = e^{i\omega t}$ of angular frequency $\omega$ ($\omega = 0$: constant; $\omega = \pi$: sign flips every step). In steady state the velocity $v_t = \beta v_{t-1} + g_t$ is also an oscillation, $v_t = G(\omega)\,e^{i\omega t}$. Substituting, $G = \beta G e^{-i\omega} + 1$, so`,
      },
      {
        t: "math",
        tex: String.raw`G(\omega) = \frac{1}{1 - \beta e^{-i\omega}}, \qquad |G(0)| = \frac{1}{1-\beta}, \qquad |G(\pi)| = \frac{1}{1+\beta}`,
        caption: String.raw`$|G(\omega)|$ — the gain: by how much momentum amplifies a gradient component of frequency $\omega$.`,
      },
      {
        t: "p",
        text: String.raw`For $\beta = 0.9$ a constant gradient is amplified 10×, a period-4 oscillation is multiplied by 0.743, and a sign-flipping one by 0.526: the ratio between the slowest and fastest components is $(1+\beta)/(1-\beta) = 19$. This is the precise meaning of "accelerates consistent directions and damps oscillations". The EMA form $(1-\beta)v$ used by Adam has the same shape scaled by $(1-\beta)$: gain 1 at $\omega = 0$ and $0.0526$ at $\omega = \pi$.`,
      },

      { t: "h", text: "3. The heavy ball as a physical system" },
      {
        t: "p",
        text: String.raw`The heavy-ball form $\theta_{t+1} - \theta_t = \beta(\theta_t - \theta_{t-1}) - \eta\nabla L(\theta_t)$ is a finite-difference discretisation of a ball of mass $m$ moving in the potential $L$ with friction $\gamma$: $m\ddot\theta + \gamma\dot\theta + \nabla L(\theta) = 0$. With time step $h$, $\ddot\theta \approx (\theta_{t+1} - 2\theta_t + \theta_{t-1})/h^2$ and $\dot\theta \approx (\theta_t - \theta_{t-1})/h$; matching coefficients gives $\beta = 1 - \gamma h/m$ and $\eta = h^2/m$. So $1 - \beta$ is the friction per step. Without friction ($\beta \to 1$) the ball oscillates forever; with very high friction ($\beta \to 0$) it creeps (plain gradient descent). Too little friction is a real failure mode — it is what happens with $\beta = 0.99$ in the exercise below.`,
      },

      { t: "h", text: "4. Momentum on a quadratic: the exact analysis" },
      {
        t: "p",
        text: String.raw`Take $L(\theta) = \tfrac12(\theta-\theta^\star)^\top H(\theta-\theta^\star)$ with eigenvalues $\mu = \lambda_{\min} \le \lambda_i \le \lambda_{\max} = L$. As for gradient descent, the eigen-coordinates decouple. For one direction with curvature $\lambda$ and error $e_t$, the heavy-ball update is a **second-order** linear recursion:`,
      },
      {
        t: "math",
        tex: String.raw`e_{t+1} = (1 + \beta - \eta\lambda)\, e_t - \beta\, e_{t-1} \quad\Longleftrightarrow\quad \begin{pmatrix} e_{t+1} \\ e_t \end{pmatrix} = \underbrace{\begin{pmatrix} 1+\beta-\eta\lambda & -\beta \\ 1 & 0 \end{pmatrix}}_{M_\lambda} \begin{pmatrix} e_{t} \\ e_{t-1} \end{pmatrix}`,
      },
      {
        t: "p",
        text: String.raw`The error shrinks like $\rho(M_\lambda)^t$, where $\rho$ is the spectral radius. The eigenvalues of $M_\lambda$ are the roots of the characteristic polynomial`,
      },
      { t: "math", tex: String.raw`r^2 - (1+\beta-\eta\lambda)\, r + \beta = 0, \qquad r_1 r_2 = \beta` },
      {
        t: "list",
        items: [
          String.raw`**Complex roots** when $(1+\beta-\eta\lambda)^2 < 4\beta$, i.e. $(1-\sqrt\beta)^2 < \eta\lambda < (1+\sqrt\beta)^2$. Then $|r_1| = |r_2|$ and $|r|^2 = r_1 r_2 = \beta$, so the rate is exactly $\sqrt{\beta}$, **independent of the curvature**. The error spirals in (a damped oscillation).`,
          String.raw`**Real roots, small curvature** ($\eta\lambda < (1-\sqrt\beta)^2$): the slow root is $r \approx 1 - \eta\lambda/(1-\beta)$ — exactly gradient descent with the effective learning rate $\eta/(1-\beta)$. This is the acceleration of flat directions.`,
          String.raw`**Instability** when a root leaves the unit circle. Evaluating the polynomial at $r = -1$ gives $2 + 2\beta - \eta\lambda = 0$, so heavy ball is stable iff $0 < \eta\lambda < 2(1+\beta)$ — the limit 2 of plain gradient descent becomes 3.8 for $\beta = 0.9$.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Optimal tuning.** To make every curvature in $[\mu, L]$ converge at the same rate $\sqrt\beta$, put the whole interval into the complex regime with the end-points on its boundary: $\eta\mu = (1-\sqrt\beta)^2$ and $\eta L = (1+\sqrt\beta)^2$. Taking square roots and adding: $\sqrt\eta(\sqrt L + \sqrt\mu) = 2$. Subtracting: $\sqrt\eta(\sqrt L - \sqrt\mu) = 2\sqrt\beta$. Hence`,
      },
      {
        t: "math",
        tex: String.raw`\boxed{\;\eta^\star = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \qquad \beta^\star = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2, \qquad \rho^\star = \sqrt{\beta^\star} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\;}`,
        caption: String.raw`$\kappa = L/\mu$. Compare gradient descent's best rate $(\kappa-1)/(\kappa+1)$.`,
      },
      {
        t: "p",
        text: String.raw`**Numbers for $\kappa = 100$** ($\mu = 1$, $L = 100$): gradient descent's best $\eta = 2/101 = 0.0198$ gives rate $99/101 = 0.9802$; reducing the error $10^6$-fold takes $\ln 10^{-6}/\ln 0.9802 = 691$ steps. Heavy ball with $\eta^\star = 4/121 = 0.03306$ and $\beta^\star = (9/11)^2 = 0.6694$ has rate $9/11 = 0.8182$ and needs **69** steps — a $\sqrt\kappa = 10$-fold speed-up. A brute-force grid search over $(\eta, \beta)$ finds no better pair (best grid rate 0.81854 at $\eta = 0.033$, $\beta = 0.670$). The derivation guarantees this for the worst case over all curvatures in $[\mu, L]$; on a problem with only a few eigenvalues, other settings can do better for a finite number of steps (see the exercise).`,
      },
      {
        t: "callout",
        kind: "note",
        title: "What this says about β = 0.9 in practice",
        text: String.raw`With $\beta = 0.9$ every direction with $0.00263 < \eta\lambda < 3.787$ converges at exactly $\sqrt{0.9} = 0.9487$ per step. Raising $\eta$ does not speed these directions up; it only pulls more of the flat directions into that regime. Directions flatter than $(1-\sqrt\beta)^2/\eta$ still converge like gradient descent with rate $\eta/(1-\beta)$. Deep-learning losses are not quadratics, but this picture — momentum buys a 10× larger effective step on flat directions at the cost of oscillation on steep ones — explains much of practice.`,
      },

      { t: "h", text: "5. Nesterov: extra damping on steep directions" },
      {
        t: "p",
        text: String.raw`For Nesterov's method the same eigen-analysis (with $a = \eta\lambda$) gives the recursion $e_{t+1} = (1+\beta)(1-a)\,e_t - \beta(1-a)\,e_{t-1}$, with characteristic polynomial $r^2 - (1+\beta)(1-a)r + \beta(1-a) = 0$. Now the product of the roots is $\beta(1-a)$ instead of $\beta$. In the complex regime the rate is $\sqrt{\beta(1-\eta\lambda)}$: **steeper directions are damped more**. For $\beta = 0.9$ and $\eta\lambda = 0.5$, heavy ball contracts by 0.9487 per step, Nesterov by $\sqrt{0.45} = 0.6708$. This is the quantitative version of "the look-ahead gradient corrects the overshoot before it happens".`,
      },
      {
        t: "p",
        text: String.raw`The price: evaluating at $r = -1$ gives $1 + (1+\beta)(1-a) + \beta(1-a) = 0$, i.e. Nesterov is stable only for $\eta\lambda < \frac{2(1+\beta)}{1+2\beta}$ — **1.357** for $\beta = 0.9$, versus 3.8 for heavy ball. Nesterov's step always contains the fresh gradient with weight $1+\beta$, so it reacts more strongly to curvature. In practice that is why Nesterov tolerates a somewhat smaller $\eta$ but is often faster at the best $\eta$ (64 vs 105 steps on the 10-dimensional quadratic in the code section). On smooth convex (not strongly convex) problems, Nesterov's method attains the error bound $O(1/t^2)$ versus $O(1/t)$ for gradient descent — the classical acceleration result (Nesterov, 1983).`,
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
        text: String.raw`AdaGrad's effective learning rate for coordinate $i$ is $\eta_{i,t} = \eta/\sqrt{G_{i,t}}$ with $G_{i,t} = \sum_{s \le t} g_{i,s}^2$. Two consequences follow directly.`,
      },
      {
        t: "list",
        items: [
          String.raw`**Decay.** If a coordinate's gradient has a roughly constant magnitude $|g|$, then $G_{i,t} = t g^2$ and the step is $\eta |g|/\sqrt{t g^2} = \eta/\sqrt t$. The schedule $\eta_t \propto 1/\sqrt t$ is exactly what online convex optimisation theory prescribes (it gives regret $O(\sqrt T)$), but in non-convex deep learning, where the optimizer must keep moving for millions of steps, it shuts training down.`,
          String.raw`**Sparse features.** If feature $j$ is non-zero in only $n_j$ of the examples seen so far, its gradient is non-zero only on those steps, so $G_j \approx n_j\,\overline{g^2}$ and its effective rate is $\propto 1/\sqrt{n_j}$. A feature seen 100 times gets a 10× larger rate than one seen 10 000 times. This is exactly right for a rare feature: each of its few observations carries a lot of information about its weight. With a single global rate, either the rare weights barely move or the frequent ones oscillate.`,
        ],
      },
      {
        t: "p",
        text: String.raw`RMSProp replaces the sum by an EMA: $s_{i,t} \approx \mathbb{E}_{\text{recent}}[g_i^2]$, so the effective rate is $\eta/\text{RMS}(g_i)$ over the last $\approx 1/(1-\rho)$ steps — it stays constant when gradients are stationary and adapts when their scale drifts. The trade-off is that the rare-feature advantage is weaker: once a rare feature has not appeared for a few hundred steps its $s$ decays towards 0, and the next occurrence produces a large normalised step instead of a calibrated one. The sparse-feature experiment in the code section shows both effects.`,
      },

      { t: "h", text: "7. Adam's step is a normalised, sign-like step" },
      {
        t: "p",
        text: String.raw`Write the per-coordinate step as $\Delta_t = -\eta\, r_t$ with $r_t = \hat m_t/(\sqrt{\hat v_t} + \epsilon)$. Three regimes explain most of Adam's behaviour.`,
      },
      {
        t: "list",
        items: [
          String.raw`**No averaging.** With $\beta_1 = \beta_2 = 0$, $r_t = g_t/|g_t| = \operatorname{sign}(g_t)$: Adam degenerates into **sign descent**, every coordinate moving by exactly $\eta$.`,
          String.raw`**Signal-to-noise ratio.** Suppose a coordinate's gradient is stationary with mean $\mu_g$ and variance $\sigma^2$. For long windows $\hat m \approx \mu_g$ and $\hat v \approx \mathbb{E}[g^2] = \mu_g^2 + \sigma^2$, so $r \approx \mu_g/\sqrt{\mu_g^2 + \sigma^2}$. When the gradient is consistent ($|\mu_g| \gg \sigma$) the step is $\approx \pm\eta$; when it is mostly noise ($|\mu_g| \ll \sigma$) the step shrinks in proportion to the SNR $\mu_g/\sigma$. Adam therefore slows down automatically near a minimum, where the true gradient is small but the mini-batch noise is not — an implicit annealing that SGD does not have. Simulation with unit noise: $\mu_g = 10, 1, 0.1$ give average steps $0.995\eta,\ 0.704\eta,\ 0.096\eta$ against the predictions $0.995, 0.707, 0.0995$.`,
          String.raw`**Bounds.** By Cauchy–Schwarz, $|\sum_k a_k g_k| \le \sqrt{\sum_k a_k^2/b_k}\,\sqrt{\sum_k b_k g_k^2}$ with $a_k = (1-\beta_1)\beta_1^k$, $b_k = (1-\beta_2)\beta_2^k$. Summing the geometric series (valid when $\beta_1^2 < \beta_2$) gives, for large $t$, $|r_t| \le \frac{1-\beta_1}{\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)}}$ — **7.27** for the defaults. The bound is reached only when gradients grow geometrically; a more realistic bad case — a gradient burst of size 1 after 2000 steps of size $10^{-3}$ — gives $r = 2.97$. So "Adam moves each parameter by at most about $\eta$" is a good rule of thumb, not a guarantee: sudden gradient bursts after quiet periods produce larger steps, one reason gradient clipping and warm-up are still used with Adam.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**The role of $\epsilon$.** For a constant gradient $g$, $r = |g|/(|g| + \epsilon)$. With $\epsilon = 10^{-8}$ the step is $\eta$ for any $|g| \gg 10^{-8}$; at $|g| = \epsilon$ it is halved; below, Adam turns into SGD with learning rate $\eta/\epsilon$ (huge). Increasing $\epsilon$ to $10^{-3}$ means coordinates with gradients smaller than about $10^{-3}$ are no longer normalised up to full steps ($r = 0.09$ at $|g| = 10^{-4}$). A large $\epsilon$ therefore interpolates between Adam and momentum SGD with learning rate $\eta/\epsilon$. Larger values than the default are used in practice for stability, e.g. $\epsilon = 10^{-6}$ in RoBERTa (Liu et al., 2019); in reduced precision, $10^{-8}$ can even underflow to zero in float16.`,
      },
      {
        t: "table",
        head: ["Setting", "Common default", "When to change it, and why"],
        rows: [
          [String.raw`$\eta$`, String.raw`$10^{-3}$ (from scratch), $10^{-5}$–$10^{-4}$ (fine-tuning), $3\times10^{-4}$ (many RL and LLM runs)`, "Always tune it first (with a schedule, see the next lesson). Because steps are about η per coordinate, η is a step length in parameter units: shrink it when weights are small (fine-tuning) or the model is large."],
          [String.raw`$\beta_1$`, "0.9", String.raw`0.5 (or 0) in GAN training, where the loss landscape changes as the other network trains, so a long memory of old gradients is harmful (DCGAN used Adam with $\beta_1 = 0.5$, $\eta = 2\times10^{-4}$).`],
          [String.raw`$\beta_2$`, "0.999", String.raw`0.98 (original Transformer, RoBERTa) or 0.95 (GPT-3, LLaMA). A shorter window lets $v$ react quickly when gradient scale jumps; with 0.999 a sudden large gradient is divided by a stale, too-small $\sqrt{v}$ and the step explodes (see the bound above).`],
          [String.raw`$\epsilon$`, String.raw`$10^{-8}$`, String.raw`$10^{-6}$ or larger for stability in some large-scale or reduced-precision runs; much larger values (up to 0.1–1) turn Adam into momentum SGD and have been used in some vision and RL settings.`],
          ["weight decay (AdamW)", "0.01 (PyTorch default); 0.1 for LLM pre-training", "Tune jointly with η: the per-step shrinkage is ηλ."],
        ],
      },

      { t: "h", text: "8. AMSGrad: when Adam fails to converge" },
      {
        t: "p",
        text: String.raw`Reddi, Kale and Kumar ("On the Convergence of Adam and Beyond", ICLR 2018) found an error in Adam's original convergence proof and built explicit convex problems on which Adam converges to the wrong point. The mechanism: the effective learning rate $\eta/\sqrt{\hat v_t}$ is **not monotone** — it grows again whenever recent gradients are small. If rare large gradients carry the important information, the EMA forgets them within $\approx 1/(1-\beta_2)$ steps, and the frequent small gradients then receive large normalised steps that outweigh them.`,
      },
      {
        t: "p",
        text: String.raw`Their simplest example is an online problem on $x \in [-1, 1]$: the loss is $C x$ on every third step and $-x$ on the others, with $C > 2$. The average gradient is $(C - 2)/3 > 0$, so the optimum is $x = -1$. With $C = 3$, $\beta_1 = 0$ and $\beta_2 = 1/(1 + C^2) = 0.1$, the steady-state normalised gradients over one period are $+1.0472, -0.7623, -0.9658$: their sum $-0.68$ is **negative**, so Adam moves towards $+1$ — the worst point. AMSGrad uses $v^{\max}_t = \max(v^{\max}_{t-1}, v_t)$ in the denominator, so after the large gradient appears once its scale is never forgotten and the effective rate is non-increasing; it recovers the optimum. With the default $\beta_2 = 0.999$ both converge on this particular problem; the paper also proves that for any fixed $\beta_1 < \sqrt{\beta_2}$ there are stochastic convex problems on which Adam does not converge.`,
      },
      {
        t: "p",
        text: "In deep-learning practice AMSGrad has rarely given clear improvements over Adam with a well-chosen β₂ and learning-rate decay, which is why it remains an option (`amsgrad=True`) rather than the default. Its value is conceptual: Adam's adaptivity needs a long enough memory, and decaying the learning rate is part of what makes Adam converge.",
      },
      {
        t: "code",
        lang: "python",
        title: "adam_step_anatomy.py — SNR, bounds, ε and the AMSGrad counterexample",
        code: ADAM_PY,
        output: ADAM_OUT,
      },

      { t: "h", text: "9. L2 regularisation versus decoupled weight decay: different fixed points" },
      {
        t: "p",
        text: String.raw`Let $\ell(\theta)$ be the data loss. **Adam + L2** runs Adam on $\ell(\theta) + \tfrac{\lambda}{2}\lVert\theta\rVert^2$. Whatever the preconditioner does along the way, a stationary point satisfies $\hat m \to 0$, i.e.`,
      },
      { t: "math", tex: String.raw`\nabla\ell(\theta) + \lambda\theta = 0 \qquad \text{(the minimiser of the L2-regularised loss)}` },
      {
        t: "p",
        text: String.raw`**AdamW** computes $m, v$ from $\nabla\ell$ alone and shrinks separately. A stationary point satisfies $\eta\,\hat m/(\sqrt{\hat v}+\epsilon) + \eta\lambda\theta = 0$, i.e.`,
      },
      {
        t: "math",
        tex: String.raw`\theta_i = -\frac{1}{\lambda}\,\frac{\hat m_i}{\sqrt{\hat v_i} + \epsilon} \qquad\Longrightarrow\qquad |\theta_i| \lesssim \frac{1}{\lambda}\ \text{ when } |\hat m_i| \le \sqrt{\hat v_i}`,
      },
      {
        t: "p",
        text: String.raw`AdamW is therefore **not** minimising any L2-regularised loss: in its equilibrium the decay balances the normalised gradient, not the raw one. Since the normalised gradient is at most about 1 per coordinate, AdamW behaves like a soft box constraint $|\theta_i| \lesssim 1/\lambda$, plus an equal relative shrinkage of every weight while the gradient signal is weak. On logistic regression for the standardised breast-cancer data with $\lambda = 0.05$, Adam + L2 lands on the exact L2 optimum found by scikit-learn ($\lVert\theta\rVert = 1.468$, KKT residual $1.3\times10^{-5}$), whereas AdamW ends at $\lVert\theta\rVert = 47.8$ with its largest weight 19.6 — just under $1/\lambda = 20$ (the data are nearly separable, so the unregularised loss keeps pushing weights outwards). Neither is "wrong"; they are different regularisers, and AdamW's version has empirically been the better one for deep networks (Loshchilov & Hutter, 2019). For SGD the two coincide exactly, because there is no preconditioner.`,
      },

      { t: "h", text: "10. Implicit bias and the generalisation debate" },
      {
        t: "p",
        text: String.raw`When a model can fit the training data in many ways, **which** solution an optimizer finds matters for test performance. A clean example: under-determined least squares (20 equations, 50 unknowns). Gradient descent started at 0 only ever adds multiples of rows of the data matrix to $\theta$, so it stays in their span and converges to the **minimum-norm** interpolating solution. Adam's per-coordinate rescaling leaves that subspace, and it converges to a different interpolating solution (in our run: norm 0.648 versus 0.598, distance 0.249 from the minimum-norm solution; code section).`,
      },
      {
        t: "p",
        text: "Wilson, Roelofs, Stern, Srebro and Recht (\"The Marginal Value of Adaptive Gradient Methods in Machine Learning\", NeurIPS 2017) built on this kind of argument: they constructed a linear problem where adaptive methods generalise much worse than SGD, and reported experiments (image classification, language modelling, parsing) in which well-tuned SGD with momentum matched or beat the adaptive methods on test error. The debate since then is empirical and has several threads: part of the gap was traced to coupled L2 regularisation (fixed by AdamW); careful tuning studies (e.g. Choi et al., 2019, \"On Empirical Comparisons of Optimizers for Deep Learning\") found that when all hyperparameters — including ε — are tuned, Adam-type methods do not underperform momentum SGD; and Transformer models are in practice trained almost exclusively with Adam-type optimizers because SGD trains them poorly. The honest summary: on convolutional image classifiers with standard recipes, SGD + momentum often generalises as well as or slightly better than Adam; on Transformers AdamW is the norm; in any single project the difference is usually smaller than the effect of the learning-rate schedule and regularisation.",
      },

      { t: "h", text: "11. Memory cost of optimizer state" },
      {
        t: "p",
        text: String.raw`For $P$ parameters, SGD stores nothing extra, momentum stores $v$ ($P$ numbers), Adam stores $m$ and $v$ ($2P$), AMSGrad $3P$. Measured in PyTorch on a 407 050-parameter MLP: 0×, 1×, 2× and 3× the parameter count. In 32-bit floats every copy costs $4P$ bytes. For GPT-2 small ($P$ = 124 439 808, counted from its layer shapes in the framework code) the weights take 0.50 GB, the gradients 0.50 GB and Adam's $m$ and $v$ another 1.00 GB — the optimizer state is as large as weights and gradients combined. In mixed-precision training the usual accounting (popularised by the ZeRO paper, Rajbhandari et al., 2020) is 16 bytes per parameter: 2 + 2 for half-precision weights and gradients, 4 + 4 + 4 for the fp32 master weights, $m$ and $v$. For a 7-billion-parameter model that is **112 GB before a single activation is stored** — more than one 80 GB accelerator holds. This is why memory-saving optimizers exist (Adafactor's factored second moment, 8-bit optimizer states, sharding the state across devices).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The optimizer race below runs the exact update rules of this lesson (PyTorch conventions, β = 0.9, Adam defaults) on two-dimensional surfaces. Use it to see the quadratic analysis in action.",
      },
      { t: "sim", id: "optimizers", caption: "Optimizer race: SGD, Momentum, Nesterov, AdaGrad, RMSProp and Adam on the same surface. Click the map to move the start point." },
      {
        t: "list",
        items: [
          String.raw`**Elongated valley, SGD-family η = 0.2.** The curvatures are 0.2 and 4, so $\eta\lambda$ = 0.04 and 0.8. For Momentum both lie inside the complex regime $0.0026 < \eta\lambda < 3.79$, so both directions spiral in at rate $\sqrt{0.9} = 0.949$ — watch Momentum overshoot the valley floor and swing back. SGD's flat direction contracts only by $1 - 0.04 = 0.96$ per step.`,
          String.raw`**Same surface, raise η to about 0.45.** SGD is close to its limit $2/4 = 0.5$ and zig-zags. Momentum is still stable (its limit is $3.8/4 = 0.95$), Nesterov's limit is $1.357/4 = 0.34$ — above that Nesterov diverges first, exactly as the stability formula predicts.`,
          String.raw`**Lower η to 0.05.** Now $\eta\lambda$ is 0.01 (flat) and 0.2 (steep). Heavy ball spirals in both directions at $\sqrt{0.9} = 0.949$ per step, while Nesterov damps the steep direction at $\sqrt{0.9 \times 0.8} = 0.849$, so its path overshoots the valley floor less.`,
          "**Adaptive methods on the valley.** Adam and RMSProp start by moving diagonally (equal steps of about η per coordinate, the sign-like regime), then RMSProp keeps hovering around the minimum because its normalised step never shrinks, while Adam's momentum averages the sign flips and settles. AdaGrad's steps shrink like 1/√t.",
          "**Saddle.** The escape direction has a tiny gradient (y = 0.01). SGD and Momentum need many steps for it to grow; Adam normalises the tiny gradient into a full-size step and escapes almost immediately — the same mechanism that helps rare features.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "One `Optimizer` class implements every method of this lesson with PyTorch's conventions (they are checked against `torch.optim` to 1e-16 in the framework section). Three experiments follow: (1) a 10-dimensional quadratic with κ = 100, once axis-aligned and once randomly rotated, each method using its best learning rate from the same grid; (2) logistic regression on the real breast-cancer dataset with raw versus standardised features; (3) logistic regression with sparse binary features of very different frequencies.",
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
          String.raw`**Rotation invariance.** SGD, heavy ball and Nesterov need exactly the same number of steps on the aligned and the rotated quadratic (184, 105, 64): their updates are linear in the gradient, so they do not care about the coordinate system. Nesterov beats heavy ball at β = 0.9 because of its extra damping of steep directions (section 5). Momentum's gain here is below the theoretical $\sqrt\kappa$ factor because β = 0.9 is fixed rather than set to $\beta^\star = 0.669$.`,
          String.raw`**Adaptive methods are not rotation invariant.** AdaGrad needs 7 steps when the curvature directions coincide with the coordinate axes (its diagonal preconditioner then captures the curvature), but 126 when they are rotated. Adam is much less affected here (103 vs 113), because its step is driven by the sign of the gradient more than by its scale. RMSProp is slowest by far: with a constant η its normalised step never shrinks, so it can only meet a tight tolerance with a tiny η — the "hovering" seen in the simulator.`,
          String.raw`**Badly scaled real data.** The raw breast-cancer features have standard deviations from 0.0023 to 577, a ratio of about 250 000. SGD and momentum must use $\eta \approx 10^{-5}$ to $10^{-6}$ to stay stable for the largest features, and reach a training loss of only 0.24–0.25. The adaptive methods, which rescale each weight separately, get to 0.17–0.20. But all of them are far behind **every** optimizer on standardised features (0.032–0.043, test accuracy 95–96%). Per-parameter adaptation softens bad scaling; it does not replace preprocessing.`,
          String.raw`**Sparse features.** In one pass over 20 000 examples, frequent features are seen about 5 700 times and rare ones about 84 times. AdaGrad recovers the weights best overall (mean error 0.203) and is the best adaptive method on the rare features (0.378): its accumulator stays small for rare features, giving them a large effective rate. Momentum SGD learns the frequent features well (0.176) but leaves the rare ones far off (0.514); RMSProp, whose EMA forgets rare features between occurrences, also does poorly on them (0.506).`,
        ],
      },
      { t: "h", text: "Weight decay, fixed points and implicit bias" },
      {
        t: "code",
        lang: "python",
        title: "decay_and_implicit_bias.py",
        code: DECAY_PY,
        output: DECAY_OUT,
      },
      {
        t: "p",
        text: String.raw`Adam + L2 converges to the L2-regularised optimum computed by scikit-learn (identical norm 1.468 and largest weight 0.425; the gradient of the regularised objective is $1.3\times10^{-5}$). AdamW does not: its gradient of that objective stays at 2.4 because it is solving a different problem, and its largest weight sits just under $1/\lambda = 20$, as predicted by the fixed-point equation of section 9. In the under-determined regression, gradient descent lands exactly on the minimum-norm interpolating solution, while Adam interpolates the data (training MSE $2.4\times10^{-8}$) with a different, larger-norm solution — the implicit-bias effect behind the generalisation debate.`,
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
                text: "The program re-implements the updates compactly in NumPy and runs the same 300 mini-batch steps with `torch.optim` in float64. Agreement to about 1e-16 proves the equations of this lesson are exactly PyTorch's (including Nesterov's re-parameterised form, AMSGrad's maximum of the uncorrected v, and AdamW's decay-before-update order). It then measures optimizer-state memory and computes it for a GPT-2-small-sized model.",
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
                  "`SGD(momentum=β, dampening=0)` stores the velocity in `state['momentum_buffer']`; the first step sets it to the gradient itself.",
                  "`Adam(weight_decay=λ)` is coupled L2; `AdamW(weight_decay=λ)` is decoupled and multiplies the weights by `1 - lr*λ` before the Adam step.",
                  "`Adam(amsgrad=True)` adds a third state tensor `max_exp_avg_sq`; it is the maximum of the *uncorrected* v, which is then bias-corrected.",
                  "Optimizer state lives in `opt.state[p]`; `opt.state_dict()` must be saved with the model checkpoint, otherwise a resumed run restarts m and v from zero (and, with Adam, the bias correction from t = 1).",
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
                  "Keras's `SGD(momentum=β)` uses the form v ← βv − ηg, θ ← θ + v, which equals PyTorch's for a constant learning rate (they differ only when η changes, because Keras folds η into v).",
                  "Keras defaults: `epsilon=1e-7` for Adam and RMSprop, and Adagrad starts its accumulator at 0.1 — set them explicitly to reproduce PyTorch numbers.",
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
          ["Heavy ball (Polyak 1964)", String.raw`$\theta_{t+1} = \theta_t - \eta g_t + \beta(\theta_t - \theta_{t-1})$`, String.raw`$\sqrt\kappa$ speed-up on quadratics; larger stability limit $2(1+\beta)/\lambda$`, "no acceleration guarantee on general convex problems; can oscillate", "SGD training of CNNs"],
          ["Nesterov (1983)", "gradient at the look-ahead point", String.raw`extra damping $\sqrt{\beta(1-\eta\lambda)}$; $O(1/t^2)$ on smooth convex problems`, String.raw`smaller stability limit $2(1+\beta)/((1+2\beta)\lambda)$`, "default choice when using SGD + momentum"],
          ["AdaGrad (2011)", String.raw`divide by $\sqrt{\sum g^2}$`, "strong theory for convex online learning; ideal for sparse features", String.raw`rate decays like $1/\sqrt t$`, "sparse linear models, embedding tables"],
          ["RMSProp (2012)", String.raw`divide by $\sqrt{\mathrm{EMA}(g^2)}$`, "non-decaying adaptive rate", "no bias correction; hovers without decay", "RNNs and RL historically"],
          ["Adam (2015)", "EMA momentum ÷ RMS, bias-corrected", "robust defaults; scale-free step ≈ η·SNR", "2 state copies; can fail to converge (AMSGrad example)", "general default"],
          ["AMSGrad (2018)", String.raw`use $\max_s v_s$`, "non-increasing effective rate; fixes the counterexamples", "3 state copies; rarely better in practice", "when Adam's loss oscillates late in training"],
          ["AdamW (2019)", "decoupled shrinkage", "weight decay independent of the preconditioner", "a different regulariser from L2", "Transformers, LLMs, ViTs"],
          ["Adamax (Kingma & Ba 2015)", String.raw`$\ell_\infty$ norm: $u_t = \max(\beta_2 u_{t-1}, |g_t|)$`, "no bias correction needed for u; bounded step", "rarely better than Adam", "embeddings, occasionally"],
          ["Nadam (Dozat 2016)", "Nesterov momentum inside Adam", "slightly faster early progress at times", "small gains", "drop-in Adam replacement"],
          ["RAdam (Liu et al. 2020)", "switches off the adaptive term while v is too poorly estimated", "removes the need for warm-up in some settings", "extra complexity", "when warm-up is hard to tune"],
          ["Adafactor (Shazeer & Stern 2018)", "factored row/column second moments for matrices", "sub-linear memory for v", "less precise per-parameter scaling", "very large models (e.g. T5)"],
          ["Lion (Chen et al. 2023)", "update = sign of an interpolated momentum", "1 state copy; sign steps", String.raw`needs smaller $\eta$ and larger $\lambda$ than AdamW`, "large vision and language models"],
          ["8-bit optimizers (Dettmers et al. 2022)", "store m and v quantised to 8 bits block-wise", "~4× less optimizer memory", "implementation complexity", "fine-tuning large models on limited memory"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "table",
        head: ["Domain", "Common recipe", "Why"],
        rows: [
          ["Image classification CNNs (from scratch)", String.raw`SGD + momentum 0.9 (often Nesterov), $\eta = 0.1$ per 256 examples of batch, weight decay $10^{-4}$ (ImageNet ResNets) to $5\times10^{-4}$ (CIFAR), step or cosine schedule`, "Well-understood, good generalisation; BatchNorm keeps the problem reasonably conditioned so adaptivity matters less."],
          ["Transformers / LLM pre-training", String.raw`AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1, warm-up then cosine decay to about 10% of the peak, gradient clipping at 1.0 (the GPT-3 and LLaMA papers report settings of this form)`, String.raw`Gradient scales differ enormously between embeddings, attention and MLP weights; a shorter $\beta_2$ window reacts to loss spikes.`],
          ["Fine-tuning pretrained Transformers", String.raw`AdamW, $\eta$ between $10^{-5}$ and $10^{-4}$ (BERT's paper searched 2e-5 to 5e-5), short linear warm-up and decay, few epochs`, "The pretrained weights are already good; Adam's η is a per-coordinate step length, so it must be small to avoid destroying them."],
          ["GANs", String.raw`Adam with $\beta_1 = 0.5$ (or 0), $\eta = 2\times10^{-4}$ (DCGAN)`, "The objective changes as the opponent trains; a long momentum memory follows stale directions."],
          ["Reinforcement learning", String.raw`Adam, $\eta \approx 3\times10^{-4}$ (PPO's MuJoCo setting), gradient clipping`, "Non-stationary targets, high-variance gradients."],
          ["Recommender systems / sparse embeddings", "AdaGrad or sparse Adam variants (`torch.optim.SparseAdam`) for embedding tables; a dense optimizer for the rest", "Rows of rare items get gradients rarely; per-row accumulators give them calibrated, larger steps and only touched rows need updating."],
          ["Small tabular / quick baselines", String.raw`Adam, $\eta = 10^{-3}$`, "Works with almost no tuning; the problem is small enough that memory is irrelevant."],
          ["Memory-limited training of huge models", "AdamW with sharded state (ZeRO/FSDP), Adafactor, or 8-bit states", "Adam's two fp32 copies dominate memory at billions of parameters."],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          String.raw`**Momentum: speed versus stability.** It turns $\kappa$ into $\sqrt\kappa$ on quadratics and raises the stability limit to $2(1+\beta)/\lambda$, but the effective step $\eta/(1-\beta)$ is 10× larger, so a learning rate tuned without momentum diverges with it. Too much momentum ($\beta \to 1$) is under-damped: rate $\sqrt\beta \to 1$.`,
          "**Nesterov: damping versus range.** Faster on steep directions at the best η, but a stability limit almost 3× lower than heavy ball's for β = 0.9.",
          "**Adaptivity is diagonal.** AdaGrad/RMSProp/Adam fix axis-aligned scale differences, not rotated ill-conditioning (126 vs 7 steps for AdaGrad in the experiment). Normalisation layers and good parameterisation still matter.",
          "**Adaptivity is approximately scale-free.** Adam's step depends on the gradient's sign and SNR, not its magnitude: robust to badly scaled losses and layers, but it also means Adam keeps taking η-sized steps on parameters whose gradients are tiny noise — the role of ε and of decaying η.",
          "**Memory.** 0, 1, 2 and 3 state copies for SGD, momentum, Adam and AMSGrad. At 7 billion parameters Adam's mixed-precision footprint is 112 GB before activations.",
          "**Generalisation is empirical.** Different optimizers find different solutions (implicit bias). SGD + momentum often matches or beats Adam on CNN image classification; AdamW is the norm for Transformers. Tune before concluding.",
          "**Hyperparameter sensitivity.** Adam's defaults work surprisingly often; SGD needs η and its schedule tuned carefully. But Adam's apparent robustness can hide a suboptimal η — it should be tuned too.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          String.raw`**Switching on momentum without lowering η.** The steady-state step becomes $\eta/(1-\beta)$: 10× for $\beta = 0.9$, 100× for 0.99. Fix: divide η by about $1/(1-\beta)$ as a starting point, then re-tune.`,
          String.raw`**Using β = 0.99 "for more acceleration".** On the quadratic every complex-regime direction then converges at $\sqrt{0.99} = 0.995$ per step: the exercise shows it is far slower than β = 0.9. Momentum must match the conditioning ($\beta^\star$ from $\kappa$), not be maximised.`,
          String.raw`**Assuming Nesterov is "momentum but strictly better".** Its stability limit is lower ($1.357/\lambda$ vs $3.8/\lambda$ at β = 0.9). A learning rate that is fine for heavy ball can make Nesterov diverge.`,
          String.raw`**Dropping Adam's bias correction "because it only matters early".** For $\beta_2 = 0.999$ the bias exceeds 1% for the first 4603 steps; without correction $\sqrt{v}$ is too small and the early steps are inflated ($3.16\eta$ at $t = 1$).`,
          String.raw`**Believing Adam's step can never exceed η.** After a quiet period, a gradient burst gives about $3\eta$ (and up to $7.27\eta$ in the worst case for the defaults). Fix: gradient clipping, warm-up, or a smaller $\beta_2$ (0.95–0.98) for spiky problems.`,
          String.raw`**Treating ε as numerically irrelevant.** It sets the gradient scale below which Adam stops normalising. With half-precision states, $10^{-8}$ can underflow; for tiny-gradient parameters, a larger ε changes the dynamics substantially.`,
          "**Using `Adam(weight_decay=λ)` and calling it AdamW.** It is L2: weights with large gradient variance are barely regularised. Fix: `AdamW`, and exclude biases and normalisation parameters.",
          String.raw`**Transferring λ between Adam + L2, AdamW and SGD.** They are different regularisers with different fixed points; AdamW's per-step shrinkage is $\eta\lambda$, so changing η changes the effective decay. Re-tune λ whenever the optimizer or η changes.`,
          "**Not saving optimizer state in checkpoints.** Resuming Adam with empty m, v and t = 0 produces a burst of η-sized steps in every coordinate. Fix: save and load `optimizer.state_dict()`.",
          "**Expecting Adam to fix bad input scaling.** It helps (loss 0.17 vs 0.25 on raw breast-cancer features) but standardising the inputs helped every optimizer far more (0.035).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What does the momentum coefficient β control?", a: String.raw`β sets how long the optimizer remembers past gradients: the velocity is an exponentially weighted sum with weights $\beta^k$, so the memory is about $1/(1-\beta)$ steps (10 for β = 0.9). Along a direction where gradients agree, the velocity grows to $g/(1-\beta)$, so the effective learning rate becomes $\eta/(1-\beta)$. Along a direction where gradients alternate in sign, contributions cancel and the gain falls to $1/(1+\beta)$. β is also one minus the friction of the equivalent physical heavy ball.` },
      { t: "qa", level: "beginner", q: "Why does Adam need bias correction?", a: String.raw`Its moving averages start at zero. After $t$ steps the weights on the observed gradients sum to $1-\beta^t$, not 1; the rest of the weight sits on the artificial zero start, so the averages are too small. Dividing by $1-\beta^t$ renormalises the weights so the estimate is a proper weighted average. This matters for $v$ in particular: with $\beta_2 = 0.999$ the bias is above 1% for the first 4603 steps, and without correction the first steps would be about 3× too large.` },
      { t: "qa", level: "beginner", q: "Why are adaptive methods good for sparse features?", a: String.raw`AdaGrad divides each parameter's step by the root of the sum of its squared gradients. A rare feature produces gradients only when it appears, so its sum stays small and its effective learning rate stays large — roughly proportional to $1/\sqrt{n_j}$, where $n_j$ is how often it has been seen. Each rare observation therefore moves its weight substantially, while frequent features get small, careful steps. A single global learning rate cannot do both.` },
      { t: "qa", level: "intermediate", q: "Derive the optimal heavy-ball momentum for a quadratic with condition number κ.", a: String.raw`Per eigen-direction with curvature λ, the error obeys $e_{t+1} = (1+\beta-\eta\lambda)e_t - \beta e_{t-1}$, with characteristic polynomial $r^2 - (1+\beta-\eta\lambda)r + \beta$. The roots are complex when $(1-\sqrt\beta)^2 < \eta\lambda < (1+\sqrt\beta)^2$, and then $|r| = \sqrt\beta$ regardless of λ. Choosing η and β so that the interval $[\eta\mu, \eta L]$ exactly fills this region, $\sqrt{\eta\mu} = 1-\sqrt\beta$ and $\sqrt{\eta L} = 1+\sqrt\beta$, gives $\eta = 4/(\sqrt L + \sqrt\mu)^2$ and $\sqrt\beta = (\sqrt\kappa - 1)/(\sqrt\kappa+1)$. The rate is $(\sqrt\kappa-1)/(\sqrt\kappa+1)$, versus $(\kappa-1)/(\kappa+1)$ for gradient descent: for κ = 100, 69 instead of 691 steps per factor $10^6$.` },
      { t: "qa", level: "intermediate", q: "How does Nesterov momentum differ from heavy ball on a quadratic?", a: String.raw`Nesterov's per-direction recursion is $e_{t+1} = (1+\beta)(1-\eta\lambda)e_t - \beta(1-\eta\lambda)e_{t-1}$. The product of its roots is $\beta(1-\eta\lambda)$ instead of β, so in the oscillatory regime it contracts at $\sqrt{\beta(1-\eta\lambda)}$ — steep directions are damped more than with heavy ball, which reduces overshoot. The price is a smaller stability region, $\eta\lambda < 2(1+\beta)/(1+2\beta)$ (1.357 for β = 0.9) instead of $2(1+\beta)$ (3.8). In PyTorch it is implemented in the re-parameterised form $\phi \leftarrow \phi - \eta(g + \beta v)$, which needs only the gradient at the stored parameters.` },
      { t: "qa", level: "intermediate", q: "Explain Adam's update as a signal-to-noise ratio. What happens near a minimum?", a: String.raw`For a coordinate whose gradient has mean $\mu_g$ and variance $\sigma^2$, $\hat m \approx \mu_g$ and $\hat v \approx \mu_g^2 + \sigma^2$, so the step is $\eta\,\mu_g/\sqrt{\mu_g^2+\sigma^2}$. Far from a minimum gradients are consistent and the step is about $\pm\eta$ (sign descent). Near a minimum the mean gradient is small compared with the mini-batch noise, so the step shrinks in proportion to $\mu_g/\sigma$ — Adam anneals itself. It never fully stops because m fluctuates, which is why a decaying learning rate is still used.` },
      { t: "qa", level: "intermediate", q: "What is the role of ε in Adam, and when would you change it?", a: String.raw`The step is $\eta\hat m/(\sqrt{\hat v}+\epsilon)$. For gradients much larger than ε it is irrelevant; for gradients comparable to ε it reduces the normalisation (step $\eta|g|/(|g|+\epsilon)$ for a constant gradient); for gradients far below ε Adam becomes momentum SGD with learning rate $\eta/\epsilon$. Increasing ε (e.g. to 1e-6, as RoBERTa did, or much larger) stabilises training when $\hat v$ is tiny or poorly estimated and makes Adam more SGD-like; it also avoids issues where 1e-8 is not representable in reduced precision.` },
      { t: "qa", level: "advanced", q: "Why did Reddi et al. propose AMSGrad, and what does it change?", a: String.raw`They showed that Adam's effective learning rate $\eta/\sqrt{\hat v_t}$ can increase over time, because the EMA of squared gradients forgets large but rare gradients within about $1/(1-\beta_2)$ steps. In their online example (loss $Cx$ every third step, $-x$ otherwise, $C = 3$, $\beta_2 = 0.1$) the rare large gradient is normalised to about 1.05 while the two small ones get −0.76 and −0.97, so the net movement goes in the wrong direction and Adam converges to the worst point. AMSGrad uses the running maximum of $v_t$, which makes the effective rate non-increasing and restores convergence; it costs a third state tensor and in deep-learning practice rarely beats Adam with a decaying learning rate.` },
      { t: "qa", level: "advanced", q: "Adam + L2 and AdamW: are they minimising the same objective?", a: String.raw`No. Adam + L2 runs Adam on $\ell + \tfrac\lambda2\lVert\theta\rVert^2$, so its fixed points satisfy $\nabla\ell + \lambda\theta = 0$ — the L2-regularised optimum (numerically, it matches scikit-learn's solution). AdamW's fixed points satisfy $\hat m/(\sqrt{\hat v}+\epsilon) = -\lambda\theta$: the decay balances the normalised gradient, which is bounded by about 1, so it acts like a soft constraint $|\theta_i| \lesssim 1/\lambda$ together with uniform relative shrinkage. On nearly separable logistic regression with λ = 0.05, Adam + L2 ends at norm 1.47 while AdamW ends at 47.8 with weights just below 20. For SGD the two are identical.` },
      { t: "qa", level: "advanced", q: "How much memory does Adam need for a 7B-parameter model, and what can be done about it?", a: String.raw`Adam keeps two extra tensors per parameter. In fp32 that alone is $2 \times 4 \times 7\times10^9 = 56$ GB. In standard mixed precision the per-parameter cost is 2 (bf16 weights) + 2 (bf16 gradients) + 4 (fp32 master weights) + 8 (fp32 m and v) = 16 bytes, i.e. 112 GB before activations. Remedies: shard optimizer state across data-parallel workers (ZeRO / FSDP), factor the second moment (Adafactor), quantise the states (8-bit optimizers), or use an optimizer with less state (SGD + momentum, Lion with one copy).` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "An EMA uses β = 0.99. What is the variance of the EMA of independent unit-variance noise, (1−β)/(1+β)? (4 decimals)", answer: 0.005, tolerance: 0.0002, explain: "0.01/1.99 = 0.005025 — equivalent to averaging about 199 samples." },
          { kind: "numeric", q: "How many steps until the bias factor β₂ᵗ of Adam's second moment falls below 0.5, for β₂ = 0.999? (round to the nearest step)", answer: 693, tolerance: 1, explain: "t > ln 0.5 / ln 0.999 = 692.8, so t = 693." },
          { kind: "numeric", q: "Heavy ball on a quadratic with μ = 1 and L = 25. What is the optimal β* = ((√κ − 1)/(√κ + 1))²?", answer: 0.4444, tolerance: 0.001, explain: "√κ = 5, (4/6)² = 0.4444; the rate is 2/3 per step versus 24/26 = 0.923 for gradient descent." },
          { kind: "numeric", q: "For the same problem (μ = 1, L = 25), what is the optimal heavy-ball learning rate η* = 4/(√L + √μ)²?", answer: 0.1111, tolerance: 0.001, explain: "4/(5 + 1)² = 4/36 = 0.1111." },
          { kind: "numeric", q: "With β = 0.9, what is the largest ηλ for which Nesterov's method is stable on a quadratic direction, 2(1+β)/(1+2β)?", answer: 1.357, tolerance: 0.002, explain: "3.8/2.8 = 1.357, versus 2(1+β) = 3.8 for heavy ball." },
          { q: "Momentum with β = 0.9 is fed a gradient that flips sign every step. Compared with a constant gradient of the same size, the steady-state velocity is…", options: ["the same", "about 19× smaller", "about 10× smaller", "zero"], answer: 1, explain: "Gains are 1/(1−β) = 10 at ω = 0 and 1/(1+β) = 0.526 at ω = π; the ratio is (1+β)/(1−β) = 19." },
          { q: "Which statement about AdaGrad and a 45°-rotated ill-conditioned quadratic is correct?", options: ["Rotation does not matter for AdaGrad", "AdaGrad's diagonal preconditioner cannot capture rotated curvature, so it loses most of its advantage", "AdaGrad diverges on rotated problems", "AdaGrad becomes Newton's method"], answer: 1, explain: "AdaGrad rescales coordinates independently. In the experiment it needed 7 steps on the aligned problem and 126 on the rotated one; SGD and momentum are rotation invariant." },
          { q: "Adam's stationary gradient for a coordinate has mean 1 and standard deviation 1. Its average step is closest to…", options: ["η", "0.71 η", "0.5 η", "0.1 η"], answer: 1, explain: "μ/√(μ²+σ²) = 1/√2 ≈ 0.707; the simulation gave 0.704." },
          { q: "At a fixed point of AdamW (weight decay λ), which relation holds?", options: ["∇ℓ(θ) + λθ = 0", "m̂/(√v̂+ε) = −λθ", "θ = 0", "v̂ = λ"], answer: 1, explain: "The decay balances the normalised gradient, so |θᵢ| is at most about 1/λ; Adam + L2 instead satisfies ∇ℓ + λθ = 0." },
        ],
      },
      {
        t: "exercise",
        title: "Numerical: momentum regimes by hand",
        prompt: "A quadratic direction has curvature λ = 4 and you use heavy-ball momentum with β = 0.81 and η = 0.05. (a) Compute ηλ and decide whether the roots of r² − (1+β−ηλ)r + β are real or complex. (b) Give the per-step contraction factor. (c) Is the method stable for this direction, and up to which η would it remain stable? (d) Repeat (a)–(b) for Nesterov with the same η and β.",
        hint: "Complex iff (1−√β)² < ηλ < (1+√β)². Stability: ηλ < 2(1+β) for heavy ball.",
        solution: [
          { t: "p", text: String.raw`(a) $\eta\lambda = 0.2$. With $\sqrt\beta = 0.9$: $(1-0.9)^2 = 0.01$ and $(1+0.9)^2 = 3.61$. Since $0.01 < 0.2 < 3.61$ the roots are complex. Check: the polynomial is $r^2 - 1.61 r + 0.81$, discriminant $1.61^2 - 3.24 = 2.5921 - 3.24 = -0.6479 < 0$.` },
          { t: "p", text: String.raw`(b) $|r| = \sqrt{\beta} = 0.9$ per step (the error spirals in, losing 10% per step). Plain gradient descent with the same η would contract by $|1 - 0.2| = 0.8$ — faster for this steep direction; momentum pays off on the flat directions.` },
          { t: "p", text: String.raw`(c) Stable iff $\eta\lambda < 2(1+\beta) = 3.62$, i.e. $\eta < 3.62/4 = 0.905$.` },
          { t: "p", text: String.raw`(d) Nesterov: polynomial $r^2 - (1+\beta)(1-\eta\lambda) r + \beta(1-\eta\lambda) = r^2 - 1.448 r + 0.648$. Discriminant $1.448^2 - 4(0.648) = 2.0967 - 2.592 = -0.4953 < 0$, complex, so $|r| = \sqrt{0.648} = 0.805$. Its stability limit is $\eta\lambda < 2(1.81)/(2.62) = 1.382$, i.e. $\eta < 0.345$.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: tune heavy ball for κ = 1000 and check the theory",
        prompt: "Write a function returning the optimal (η, β) for heavy ball given μ and L. On a quadratic with 50 curvatures spread uniformly over [1, 1000] and start x₀ = (1, …, 1), compare after 300 and 1000 steps: gradient descent with η = 2/(μ+L), heavy ball with the optimal pair, and heavy ball with β fixed at 0.9 and at 0.99 (choose the best η on a grid by the 300-step result). Explain the ranking.",
        hint: "Use the PyTorch form v ← βv + λx, x ← x − ηv; the gradient of ½Σλᵢxᵢ² is λ ⊙ x.",
        solution: [
          { t: "code", lang: "python", title: "tune_heavy_ball.py", code: EX_PY, output: EX_OUT },
          {
            t: "p",
            text: String.raw`Gradient descent's rate $999/1001 = 0.998$ leaves the error at 0.78 after 300 steps. The optimal pair ($\eta^\star = 0.00376$, $\beta^\star = 0.881$) has asymptotic rate 0.9387 and is best after 1000 steps ($6.5\times10^{-25}$). After only 300 steps, β = 0.9 with a tuned η is slightly ahead: the optimal choice puts the extreme curvatures exactly on a double root, whose error decays like $t\,\rho^t$ instead of $\rho^t$, and this polynomial factor matters at short horizons. β = 0.99 is badly under-damped: every oscillating direction contracts only by $\sqrt{0.99} = 0.995$ per step, so it is barely better than gradient descent. More momentum is not better; the right momentum depends on κ.`,
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
          String.raw`Bias correction $\hat m_t = m_t/(1-\beta^t)$ renormalises the weights of a zero-started EMA; for $\beta_2 = 0.999$ it matters for thousands of steps.`,
          String.raw`Momentum is a low-pass filter: gain $1/(1-\beta)$ for constant gradients, $1/(1+\beta)$ for alternating ones; equivalently a heavy ball with friction $1-\beta$.`,
          String.raw`Heavy ball on a quadratic: rate $\sqrt\beta$ for $(1-\sqrt\beta)^2 < \eta\lambda < (1+\sqrt\beta)^2$, stable iff $\eta\lambda < 2(1+\beta)$; optimal $\eta^\star = 4/(\sqrt L+\sqrt\mu)^2$, $\beta^\star = ((\sqrt\kappa-1)/(\sqrt\kappa+1))^2$, rate $(\sqrt\kappa-1)/(\sqrt\kappa+1)$.`,
          String.raw`Nesterov: rate $\sqrt{\beta(1-\eta\lambda)}$ (extra damping of steep directions), stable iff $\eta\lambda < 2(1+\beta)/(1+2\beta)$; PyTorch form $\phi \leftarrow \phi - \eta(g + \beta v)$.`,
          String.raw`AdaGrad: per-coordinate rate $\eta/\sqrt{\sum g^2}$ — $\propto 1/\sqrt{n_j}$ for sparse features, $\propto 1/\sqrt t$ for dense ones. RMSProp: $\eta/\mathrm{RMS}_{\text{recent}}(g)$.`,
          String.raw`Adam's step $\approx \eta\,\mu_g/\sqrt{\mu_g^2+\sigma^2}$: sign-like when the signal is strong, shrinking with SNR; worst case $\approx 7.27\eta$ for default betas; ε sets where normalisation stops.`,
          String.raw`AMSGrad uses $\max_s v_s$ so the effective rate never increases; fixes Reddi et al.'s counterexamples.`,
          String.raw`Adam + L2 converges to $\nabla\ell + \lambda\theta = 0$; AdamW to $\hat m/\sqrt{\hat v} = -\lambda\theta$ (soft box $|\theta_i| \lesssim 1/\lambda$).`,
          "Optimizers have implicit biases (GD → minimum-norm solution); SGD-vs-Adam generalisation differences are empirical and task-dependent.",
          "State memory: momentum 1×, Adam 2×, AMSGrad 3× parameters; mixed-precision Adam ≈ 16 bytes/parameter (112 GB for 7B).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — every result here assumed a constant η; warm-up and decay are what make Adam stable at the start and precise at the end.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — places weight decay and the implicit bias of optimizers among the other tools against overfitting.",
          "[Batch, Layer and Group Normalization](/learn/m07/normalization) — normalisation improves the conditioning that limits every optimizer here, including the rotated ill-conditioning adaptive methods cannot fix.",
          "[Weight Initialization](/learn/m07/initialization) — the scale of the first gradients, which decides how large the early Adam steps and momentum velocities are.",
          "[Gradient Descent and Its Descendants](/learn/m06/gradient-descent) — revisit the update equations and the two-step worked example with the theory from this lesson in mind.",
        ],
      },
    ],
  },
};

export default lesson;
