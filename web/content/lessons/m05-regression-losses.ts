import type { Lesson } from "@/lib/types";

const SCRATCH = String.raw`import numpy as np
from sklearn.datasets import load_diabetes

# ---------------- losses on the residual r = y_hat - y: (per-example loss, dloss/dr) ----------------
def mse(r):
    return r ** 2, 2 * r

def mae(r):
    return np.abs(r), np.sign(r)                       # subgradient 0 at r = 0

def huber(r, delta=1.0):
    small = np.abs(r) <= delta
    loss = np.where(small, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))
    return loss, np.clip(r, -delta, delta)             # gradient = clipped residual

def log_cosh(r):
    a = np.abs(r)
    return a + np.log1p(np.exp(-2 * a)) - np.log(2), np.tanh(r)   # stable log(cosh r)

def pinball(r, tau):
    # quantile loss written on u = y - y_hat = -r: tau*u if u >= 0 else (tau - 1)*u
    u = -r
    return np.maximum(tau * u, (tau - 1) * u), np.where(u > 0, -tau, 1 - tau)

# ---------------- gradient check of every smooth loss ----------------
rng = np.random.default_rng(0)
r = rng.normal(0, 2, size=7)
h = 1e-6
for name, f in [("MSE", mse), ("Huber", huber), ("log-cosh", log_cosh)]:
    num = (f(r + h)[0] - f(r - h)[0]) / (2 * h)
    print(f"{name:8s} max |analytic - numeric| = {np.abs(f(r)[1] - num).max():.1e}")

# ---------------- 1. the constant that minimises each loss ----------------
y = np.array([2.0, 3.0, 4.0, 5.0, 26.0])               # one outlier
grid = np.linspace(0, 30, 300001)                      # candidate constants, step 1e-4
for name, f in [("MSE", mse), ("MAE", mae), ("Huber", huber), ("log-cosh", log_cosh)]:
    total = np.array([f(c - y)[0].mean() for c in grid[::100]])     # coarse search ...
    c0 = grid[::100][total.argmin()]
    fine = grid[(grid > c0 - 0.02) & (grid < c0 + 0.02)]            # ... then refine
    best = fine[np.argmin([f(c - y)[0].mean() for c in fine])]
    print(f"{name:8s} best constant = {best:.4f}")
print(f"mean = {y.mean():.4f}, median = {np.median(y):.4f}")

# ---------------- 2. linear regression on real data with corrupted targets ----------------
X, t = load_diabetes(return_X_y=True)                 # 442 patients, 10 features, disease progression
idx = rng.permutation(len(t))
tr, te = idx[:330], idx[330:]
mu, sd = X[tr].mean(0), X[tr].std(0)
Xs = np.c_[np.ones(len(t)), (X - mu) / sd]             # standardised features + bias column

t_train = t[tr].copy()
bad = rng.choice(len(tr), size=33, replace=False)      # corrupt 10% of the training targets
t_train[bad] += 300                                    # e.g. a unit/entry error
y_mean, y_sd = t_train.mean(), t_train.std()           # normalise targets with training statistics
y_train = (t_train - y_mean) / y_sd

def fit(loss, steps=20000, lr=0.15):
    w = np.zeros(Xs.shape[1])
    for k in range(steps):
        rr = Xs[tr] @ w - y_train
        g = Xs[tr].T @ loss(rr)[1] / len(tr)           # chain rule: dL/dw = X^T dL/dr / N
        w -= lr / np.sqrt(1 + k / 4000) * g             # slowly decaying step: needed for the MAE kink
    return w

print(f"\ncorrupted {len(bad)} of {len(tr)} training targets (+300); test set is clean")
print(f"{'loss':10s} {'test MAE':>9s} {'test RMSE':>10s} {'mean test residual':>19s}")
for name, f in [("MSE", mse), ("MAE", mae), ("Huber 0.5", lambda r: huber(r, 0.5)), ("log-cosh", log_cosh)]:
    w = fit(f)
    pred = (Xs[te] @ w) * y_sd + y_mean                 # back to the original units
    res = pred - t[te]
    print(f"{name:10s} {np.abs(res).mean():9.2f} {np.sqrt((res ** 2).mean()):10.2f} {res.mean():19.2f}")

clean = t[tr]
w_ref = np.linalg.lstsq(Xs[tr], clean, rcond=None)[0]   # least squares on the CLEAN targets, for reference
res = Xs[te] @ w_ref - t[te]
print(f"{'MSE clean':10s} {np.abs(res).mean():9.2f} {np.sqrt((res ** 2).mean()):10.2f} {res.mean():19.2f}")

# ---------------- 3. quantile regression: a 90% prediction interval ----------------
n = 2000
x = rng.uniform(0, 4, n)
yq = 1.0 + 0.5 * x + rng.normal(0, 0.2 + 0.3 * x)      # noise grows with x (heteroscedastic)
A = np.c_[np.ones(n), x]
xtr, xte = slice(0, 1500), slice(1500, None)
lines = {}
for tau in (0.05, 0.5, 0.95):
    w = np.zeros(2)
    for k in range(20000):
        g = A[xtr].T @ pinball(A[xtr] @ w - yq[xtr], tau)[1] / 1500
        w -= 0.5 / np.sqrt(1 + k / 50) * g
    lines[tau] = w
    below = np.mean(yq[xte] <= A[xte] @ w)
    print(f"tau={tau:.2f}: y_hat = {w[0]:.3f} + {w[1]:.3f} x   fraction of test points below = {below:.3f}")
lo, hi = A[xte] @ lines[0.05], A[xte] @ lines[0.95]
print(f"90% interval coverage on test data: {np.mean((yq[xte] >= lo) & (yq[xte] <= hi)):.3f}")`;

const SCRATCH_OUT = String.raw`MSE      max |analytic - numeric| = 5.9e-10
Huber    max |analytic - numeric| = 1.4e-10
log-cosh max |analytic - numeric| = 2.8e-10
MSE      best constant = 8.0000
MAE      best constant = 4.0000
Huber    best constant = 4.0000
log-cosh best constant = 4.0188
mean = 8.0000, median = 4.0000

corrupted 33 of 330 training targets (+300); test set is clean
loss        test MAE  test RMSE  mean test residual
MSE            49.46      60.87               25.82
MAE            45.00      56.54                4.74
Huber 0.5      44.64      56.56                2.91
log-cosh       44.94      56.52               10.21
MSE clean      44.72      56.89               -5.19
tau=0.05: y_hat = 0.627 + 0.040 x   fraction of test points below = 0.036
tau=0.50: y_hat = 0.993 + 0.497 x   fraction of test points below = 0.512
tau=0.95: y_hat = 1.275 + 1.026 x   fraction of test points below = 0.934
90% interval coverage on test data: 0.898`;

const TORCH = String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Part 1: the built-in regression losses and their gradients ----
y_hat = torch.tensor([0.0, 0.5, 2.0, 10.0], requires_grad=True)
y = torch.zeros(4)                                  # so the residuals are 0, 0.5, 2, 10

def log_cosh(pred, target):                         # not built in: stable form |r| + softplus(-2|r|) - log 2
    r = (pred - target).abs()
    return r + F.softplus(-2 * r) - math.log(2)

losses = {
    "MSELoss":          nn.MSELoss(reduction="none"),
    "L1Loss":           nn.L1Loss(reduction="none"),
    "HuberLoss(d=1)":   nn.HuberLoss(delta=1.0, reduction="none"),
    "SmoothL1(beta=2)": nn.SmoothL1Loss(beta=2.0, reduction="none"),
    "HuberLoss(d=2)":   nn.HuberLoss(delta=2.0, reduction="none"),
    "log-cosh":         log_cosh,
}
print(f"{'residual':18s}", "  ".join(f"{v:7.1f}" for v in [0, 0.5, 2, 10]))
for name, fn in losses.items():
    y_hat.grad = None
    loss = fn(y_hat, y)
    loss.sum().backward()
    print(f"{name:18s}", "  ".join(f"{v:7.3f}" for v in loss.tolist()), " | grad", [round(g, 3) for g in y_hat.grad.tolist()])

# ---- Part 2: predicting the variance with the Gaussian negative log-likelihood ----
torch.manual_seed(0)
n = 2000
x = torch.rand(n, 1) * 2 - 1                         # inputs in [-1, 1]
sigma_true = 0.05 + 0.45 * (x + 1) / 2              # noise std grows from 0.05 to 0.5
yv = torch.sin(3 * x) + sigma_true * torch.randn(n, 1)
xtr, ytr, xte, yte = x[:1500], yv[:1500], x[1500:], yv[1500:]

def mlp(out):
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, out))

def train(model, loss_fn, steps=2000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(xtr)).backward()
        opt.step()
    return model

# (a) MSE: predicts the mean only; one constant sigma estimated afterwards from training residuals
m_mse = train(mlp(1), lambda out: F.mse_loss(out, ytr))
# (b) Gaussian NLL: two outputs, the mean and log-variance (exp keeps the variance positive)
gnll = nn.GaussianNLLLoss()                          # 0.5 * (log var + (y - mu)^2 / var), mean over batch
m_nll = train(mlp(2), lambda out: gnll(out[:, :1], ytr, out[:, 1:].exp()))

with torch.no_grad():
    mu_a = m_mse(xte)
    sig_a = torch.full_like(mu_a, (m_mse(xtr) - ytr).pow(2).mean().sqrt().item())
    out = m_nll(xte)
    mu_b, sig_b = out[:, :1], out[:, 1:].exp().sqrt()
    z90 = 1.6449                                     # 90% two-sided normal interval: mu +- 1.645 sigma
    for name, mu, sig in [("MSE + constant sigma", mu_a, sig_a), ("Gaussian NLL", mu_b, sig_b)]:
        mse = (mu - yte).pow(2).mean().item()
        nll = gnll(mu, yte, sig ** 2).item() + 0.5 * math.log(2 * math.pi)   # add the constant for a true NLL
        inside = ((yte - mu).abs() <= z90 * sig).float()
        lo, hi = xte < -0.5, xte > 0.5
        print(f"{name:21s} test MSE {mse:.4f}  RMSE {math.sqrt(mse):.4f}  NLL {nll:7.4f}  "
              f"90% coverage: all {inside.mean():.3f}, x<-0.5 {inside[lo].mean():.3f}, x>0.5 {inside[hi].mean():.3f}")
    probe = torch.tensor([[-0.9], [0.0], [0.9]])
    print("predicted sigma at x = -0.9, 0, 0.9:", [round(v, 3) for v in m_nll(probe)[:, 1].exp().sqrt().tolist()])
    print("true sigma      at x = -0.9, 0, 0.9:", [round(0.05 + 0.45 * (v + 1) / 2, 3) for v in (-0.9, 0.0, 0.9)])`;

const TORCH_OUT = String.raw`residual               0.0      0.5      2.0     10.0
MSELoss              0.000    0.250    4.000  100.000  | grad [0.0, 1.0, 4.0, 20.0]
L1Loss               0.000    0.500    2.000   10.000  | grad [0.0, 1.0, 1.0, 1.0]
HuberLoss(d=1)       0.000    0.125    1.500    9.500  | grad [0.0, 0.5, 1.0, 1.0]
SmoothL1(beta=2)     0.000    0.062    1.000    9.000  | grad [0.0, 0.25, 1.0, 1.0]
HuberLoss(d=2)       0.000    0.125    2.000   18.000  | grad [0.0, 0.5, 2.0, 2.0]
log-cosh             0.000    0.120    1.325    9.307  | grad [0.0, 0.462, 0.964, 1.0]
MSE + constant sigma  test MSE 0.0865  RMSE 0.2940  NLL  0.1951  90% coverage: all 0.898, x<-0.5 1.000, x>0.5 0.767
Gaussian NLL          test MSE 0.0868  RMSE 0.2947  NLL -0.0328  90% coverage: all 0.910, x<-0.5 0.942, x>0.5 0.905
predicted sigma at x = -0.9, 0, 0.9: [0.072, 0.258, 0.442]
true sigma      at x = -0.9, 0, 0.9: [0.072, 0.275, 0.477]`;

const KERAS = String.raw`import numpy as np
import tensorflow as tf

y_true = tf.constant([[0.0], [0.0], [0.0], [0.0]])
y_pred = tf.constant([[0.0], [0.5], [2.0], [10.0]])      # residuals 0, 0.5, 2, 10

for loss in [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanAbsoluteError(),
             tf.keras.losses.Huber(delta=1.0), tf.keras.losses.LogCosh()]:
    print(type(loss).__name__, float(loss(y_true, y_pred)))   # mean over the 4 examples

def pinball_loss(tau):
    def loss(y_true, y_pred):
        u = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * u, (tau - 1.0) * u), axis=-1)
    return loss

def gaussian_nll(y_true, out):
    # out[:, 0] = mean, out[:, 1] = log-variance
    mu, log_var = out[:, :1], out[:, 1:]
    return tf.reduce_mean(0.5 * (log_var + tf.square(y_true - mu) / tf.exp(log_var)), axis=-1)

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, (2000, 1)).astype("float32")
sigma = 0.05 + 0.45 * (x + 1) / 2
y = (np.sin(3 * x) + sigma * rng.normal(size=x.shape)).astype("float32")

model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(32, activation="tanh"),
                             tf.keras.layers.Dense(32, activation="tanh"), tf.keras.layers.Dense(2)])
model.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss=gaussian_nll)
model.fit(x, y, epochs=200, batch_size=256, verbose=0)
out = model.predict(np.array([[-0.9], [0.0], [0.9]], dtype="float32"), verbose=0)
print("predicted sigma:", np.sqrt(np.exp(out[:, 1])))

q90 = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(32, activation="tanh"), tf.keras.layers.Dense(1)])
q90.compile(optimizer="adam", loss=pinball_loss(0.9))          # a 90th-percentile regressor
q90.fit(x, y, epochs=100, batch_size=256, verbose=0)
print("fraction of y below the 0.9-quantile prediction:", float(np.mean(y <= q90.predict(x, verbose=0))))`;

const EXERCISE = String.raw`import numpy as np

def pinball(y, q, tau):
    u = y - q                                          # positive when we under-predict
    return np.mean(np.maximum(tau * u, (tau - 1) * u))

rng = np.random.default_rng(1)
y = rng.exponential(scale=2.0, size=1001)              # skewed data: mean, median and tails differ

grid = np.linspace(0, 12, 120001)                      # candidate constant predictions, step 1e-4
for tau in (0.1, 0.5, 0.9):
    losses = np.array([pinball(y, q, tau) for q in grid])
    q_best = grid[losses.argmin()]
    print(f"tau={tau}: argmin of pinball = {q_best:.3f}   np.quantile = {np.quantile(y, tau):.3f}   "
          f"fraction of y below = {np.mean(y < q_best):.3f}")

# subgradient check: d/dq mean pinball = mean( (1 - tau) if y < q else -tau ) = P(y < q) - tau
q, tau = 3.0, 0.9
h = 1e-6
numeric = (pinball(y, q + h, tau) - pinball(y, q - h, tau)) / (2 * h)
print(f"at q=3, tau=0.9: numeric slope {numeric:.4f}, formula P(y<q) - tau = {np.mean(y < q) - tau:.4f}")`;

const EXERCISE_OUT = String.raw`tau=0.1: argmin of pinball = 0.212   np.quantile = 0.212   fraction of y below = 0.100
tau=0.5: argmin of pinball = 1.353   np.quantile = 1.353   fraction of y below = 0.500
tau=0.9: argmin of pinball = 4.670   np.quantile = 4.670   fraction of y below = 0.900
at q=3, tau=0.9: numeric slope -0.1248, formula P(y<q) - tau = -0.1248`;

const lesson: Lesson = {
  id: "m05/regression-losses",
  title: "Regression Losses: MSE, MAE, Huber, Log-Cosh, Quantile and Gaussian NLL",
  summary:
    "How to measure error on continuous targets: residuals; MSE and its link to Gaussian maximum likelihood; MAE and the median; outlier sensitivity on real data; Huber, log-cosh and quantile (pinball) losses; RMSE versus MSE; units and target normalisation; and predicting the noise level with a heteroscedastic Gaussian negative log-likelihood.",
  minutes: 70,
  level: "beginner",
  prerequisites: ["m02/probability", "m02/mle-map", "m02/derivatives-chain-rule", "m03/linear-logistic-regression", "m04/backpropagation"],
  next: ["m05/classification-losses", "m05/advanced-losses", "m06/gradient-descent", "m17/regression-metrics"],
  terms: ["loss", "mse", "residual", "mae", "huber-loss", "log-cosh-loss", "quantile-loss", "gaussian-nll", "empirical-risk", "negative-log-likelihood"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`A network that predicts a house price, a temperature, a patient's disease progression or the coordinates of a bounding box outputs a real number $\hat y$. Training needs a way to say how wrong that number is, as a single differentiable scalar, so that backpropagation can compute how each weight should change. That scalar is the **loss**. For regression it is built from the **residual** $r = \hat y - y$, the signed difference between prediction and target.`,
      },
      {
        t: "p",
        text: String.raw`The choice of loss is not a technicality. It decides **which prediction is "best"** when the target is uncertain: squared error makes the network predict the conditional **mean** of $y$ given $x$, absolute error the conditional **median**, the pinball loss any chosen **quantile**. It decides how much a single wildly wrong label can drag the model (outlier sensitivity). It fixes the units and scale of the gradient, which interacts with the learning rate. And, read as a negative log-likelihood, it states an assumption about the noise: squared error is the Gaussian assumption, absolute error the Laplace assumption, and the heteroscedastic Gaussian NLL lets the network predict **how uncertain** it is for each input.`,
      },
      {
        t: "p",
        text: "This lesson derives each regression loss, proves which statistic it estimates, shows its gradient, demonstrates outlier sensitivity on the real *diabetes* dataset with 10% of the training targets corrupted, and ends with quantile regression for prediction intervals and variance prediction with the Gaussian NLL. Classification losses (cross-entropy, focal, hinge) are covered in [Classification Losses](/learn/m05/classification-losses).",
      },
      {
        t: "callout",
        kind: "key",
        title: "Where these losses are used",
        text: "MSE: the default for regression heads, value functions, autoencoder reconstructions and denoising. MAE / L1: image restoration, forecasting of medians, data with heavy-tailed noise. Huber / smooth L1: bounding-box regression in object detectors, Q-learning targets. Quantile loss: probabilistic forecasting and prediction intervals. Gaussian NLL: models that must report their own uncertainty.",
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Probability and distributions](/learn/m02/probability) — expectations, the Gaussian density $\mathcal{N}(y;\mu,\sigma^2)$, and the meaning of mean, median and quantile of a distribution.`,
          String.raw`[Maximum likelihood estimation](/learn/m02/mle-map) — we derive MSE and MAE as negative log-likelihoods: maximising $\prod_i p(y_i \mid x_i)$ is the same as minimising $-\sum_i \log p(y_i \mid x_i)$.`,
          String.raw`[Derivatives and the chain rule](/learn/m02/derivatives-chain-rule) — every loss here is differentiated with respect to the residual $r$ and then chained into the weights; you also need one-sided derivatives for the kink of $|r|$.`,
          String.raw`[Linear and logistic regression](/learn/m03/linear-logistic-regression) — the linear model $\hat y = w^\top x + b$ used in the experiments, and least squares.`,
          String.raw`[Backpropagation](/learn/m04/backpropagation) — the loss derivative $\partial \mathcal{L}/\partial \hat y$ computed in this lesson is the first signal that backprop sends into the network.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Where should five friends meet on a straight road?",
        text: String.raw`Five friends live along one road at kilometre marks 2, 3, 4, 5 and 26. If the meeting point must minimise the **total distance** driven, the answer is kilometre 4, the median: moving the point by one kilometre towards the majority shortens more journeys than it lengthens. If instead it must minimise the total of **squared** distances, the answer is kilometre 8, the mean: the friend at 26 is so expensive when squared that the whole group is dragged towards him. Squared error lets the farthest person dictate; absolute error gives everyone one vote. That is the whole outlier story of regression losses.`,
      },
      {
        t: "p",
        text: String.raw`Every regression loss is a function $\ell(r)$ of the residual, with $\ell(0) = 0$ and $\ell$ growing as $|r|$ grows. What matters for learning is its **derivative** $\ell'(r)$, because that is how hard each example pulls on the prediction. For squared error, $\ell'(r) = 2r$: an error ten times larger pulls ten times harder. For absolute error, $\ell'(r) = \pm 1$: every wrong example pulls with the same force, only the direction differs. Huber and log-cosh behave like squared error for small residuals (smooth, precise near the optimum) and like absolute error for large ones (a cap on how hard one example can pull).`,
      },
      {
        t: "p",
        text: String.raw`The pinball loss tilts absolute error: under-predicting costs $\tau$ per unit and over-predicting $1 - \tau$. With $\tau = 0.9$ under-prediction is nine times more expensive, so the best prediction moves up until only 10% of the data lies above it — the 90th percentile. Finally, a network can output **two** numbers, a mean and a variance; the Gaussian NLL then lets it say "I expect an error of about ±0.07 here but ±0.45 there", instead of using one error bar for every input.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "From prediction to training signal" },
      {
        t: "steps",
        items: [
          { title: "Forward pass", text: String.raw`The network maps each input $x_i$ to a prediction $\hat y_i = f(x_i;\theta)$ — one number per target, usually from a final linear layer with **no** activation, so the output can take any real value.` },
          { title: "Residual", text: String.raw`Compute $r_i = \hat y_i - y_i$ for every example in the mini-batch. Its shape must match exactly: (B,) against (B,), or (B, 1) against (B, 1).` },
          { title: "Per-example loss", text: String.raw`Apply the loss function elementwise: $\ell(r_i)$ — e.g. $r_i^2$, $|r_i|$, $\mathrm{Huber}_\delta(r_i)$.` },
          { title: "Reduction", text: String.raw`Average over the batch (and over output dimensions): $\mathcal{L} = \frac{1}{B}\sum_i \ell(r_i)$. The mean keeps the gradient scale independent of batch size; a sum would multiply the effective learning rate by $B$.` },
          { title: "Loss derivative", text: String.raw`Backprop starts with $\partial\mathcal{L}/\partial\hat y_i = \ell'(r_i)/B$, because $\partial r_i/\partial \hat y_i = 1$. This single number per example is where the losses differ.` },
          { title: "Chain into the weights", text: String.raw`For a linear model $\hat y = Xw$, $\nabla_w\mathcal{L} = X^\top \ell'(r)/B$. For a deep network, backpropagation carries $\ell'(r_i)/B$ through every layer.` },
          { title: "Undo any target normalisation for reporting", text: String.raw`If the targets were standardised as $\tilde y = (y - m)/s$, the network predicts $\tilde y$; report $\hat y = s\,\hat{\tilde y} + m$ and compute metrics in the original units.` },
        ],
      },
      { t: "h", text: "What each loss makes the network predict" },
      {
        t: "table",
        head: ["Loss", String.raw`$\ell(r)$`, String.raw`$\ell'(r)$`, "Optimal prediction", "Noise model (as NLL)"],
        rows: [
          ["MSE", String.raw`$r^2$`, String.raw`$2r$`, String.raw`mean $\mathbb{E}[y\mid x]$`, "Gaussian, fixed variance"],
          ["MAE", String.raw`$|r|$`, String.raw`$\operatorname{sign}(r)$`, "median", "Laplace, fixed scale"],
          [String.raw`Huber$_\delta$`, String.raw`$\tfrac12 r^2$ / $\delta(|r| - \tfrac12\delta)$`, String.raw`$\operatorname{clip}(r, -\delta, \delta)$`, "between mean and median (an M-estimator)", "Gaussian centre, Laplace tails"],
          ["log-cosh", String.raw`$\log\cosh r$`, String.raw`$\tanh r$`, "between mean and median", String.raw`density $\propto 1/\cosh(y-\mu)$`],
          [String.raw`pinball$_\tau$`, String.raw`$\max(\tau u, (\tau-1)u)$, $u = -r$`, String.raw`$-\tau$ or $1-\tau$`, String.raw`$\tau$-quantile`, "asymmetric Laplace"],
          ["Gaussian NLL", String.raw`$\tfrac12\log\sigma^2 + \tfrac{r^2}{2\sigma^2}$`, String.raw`$r/\sigma^2$ (w.r.t. $\mu$)`, String.raw`mean and variance`, String.raw`Gaussian, variance $\sigma^2(x)$ predicted`],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Residuals and empirical risk" },
      {
        t: "p",
        text: String.raw`Training set $\{(x_i, y_i)\}_{i=1}^N$, model $f(x;\theta)$ with parameters $\theta$, prediction $\hat y_i = f(x_i;\theta)$, residual $r_i = \hat y_i - y_i$. A regression loss is a function $\ell:\mathbb{R}\to[0,\infty)$ with $\ell(0) = 0$; training minimises the **empirical risk**`,
      },
      { t: "math", tex: String.raw`\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell\big(f(x_i;\theta) - y_i\big), \qquad \nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \ell'(r_i)\, \nabla_\theta f(x_i;\theta)` },
      {
        t: "p",
        text: String.raw`The second formula is the chain rule: each example contributes its model gradient $\nabla_\theta f$ weighted by $\ell'(r_i)$. The function $\ell'$ is called the **influence** (or $\psi$-function in robust statistics): it says how strongly one example can move the fit. Bounded $\ell'$ means bounded influence.`,
      },

      { t: "h", text: "2. Mean squared error and the mean" },
      { t: "math", tex: String.raw`\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^N (\hat y_i - y_i)^2, \qquad \frac{\partial\,\mathrm{MSE}}{\partial \hat y_i} = \frac{2}{N}(\hat y_i - y_i)` },
      {
        t: "p",
        text: String.raw`**The minimising constant is the mean.** Suppose the model can only predict a constant $c$. Then $\mathcal{L}(c) = \frac1N\sum_i (c - y_i)^2$ is a convex parabola with derivative $\frac2N\sum_i(c - y_i) = 2(c - \bar y)$, which vanishes exactly at $c = \bar y = \frac1N\sum_i y_i$. The same argument conditioned on $x$ shows that over all functions, the expected squared error $\mathbb{E}[(f(x) - y)^2]$ is minimised by $f(x) = \mathbb{E}[y \mid x]$: add and subtract the conditional mean $m(x)$,`,
      },
      { t: "math", tex: String.raw`\mathbb{E}\big[(f(x) - y)^2\big] = \mathbb{E}\big[(f(x) - m(x))^2\big] + \mathbb{E}\big[(m(x) - y)^2\big], \qquad m(x) = \mathbb{E}[y\mid x]` },
      {
        t: "p",
        text: String.raw`(the cross term vanishes because $\mathbb{E}[m(x) - y \mid x] = 0$). The second term is the irreducible noise; the first is zero only for $f = m$. A network trained with MSE is therefore a conditional-mean estimator.`,
      },
      { t: "h", text: "MSE is Gaussian maximum likelihood" },
      {
        t: "p",
        text: String.raw`Assume $y_i = f(x_i;\theta) + \varepsilon_i$ with independent noise $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ and fixed $\sigma$. The negative log-likelihood (see [MLE and MAP](/learn/m02/mle-map)) is`,
      },
      { t: "math", tex: String.raw`-\log \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-\frac{(y_i - \hat y_i)^2}{2\sigma^2}} = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^N (\hat y_i - y_i)^2` },
      {
        t: "p",
        text: String.raw`The first term does not depend on $\theta$ and the second is $\frac{N}{2\sigma^2}\,\mathrm{MSE}$, so maximising the Gaussian likelihood **is** minimising MSE. Using MSE is an implicit statement that the noise is Gaussian with the same variance for every input — the assumption we relax in section 9.`,
      },

      { t: "h", text: "3. Mean absolute error and the median" },
      { t: "math", tex: String.raw`\mathrm{MAE} = \frac1N\sum_{i=1}^N |\hat y_i - y_i|, \qquad \frac{\partial\,|r|}{\partial r} = \operatorname{sign}(r) = \begin{cases} +1 & r > 0\\ -1 & r < 0 \end{cases}, \qquad \partial|r|\,\big|_{r=0} = [-1, 1]` },
      {
        t: "p",
        text: String.raw`At $r = 0$, $|r|$ has a kink: the left derivative is $-1$ and the right derivative $+1$. Any value $g \in [-1, 1]$ is a **subgradient** — the slope of a line through the kink that stays below the function. Frameworks pick $g = 0$ (PyTorch's sign(0) = 0). A point minimises a convex function exactly when 0 belongs to its subdifferential.`,
      },
      {
        t: "p",
        text: String.raw`**The minimising constant is the median.** For a constant $c$ that is not equal to any $y_i$, $\frac{d}{dc}\frac1N\sum_i|c - y_i| = \frac1N\big(\#\{i: y_i < c\} - \#\{i: y_i > c\}\big)$. It is negative while fewer than half the points lie below $c$ (moving right helps) and positive once more than half do. At a data point $c = y_k$ the subdifferential is $\frac1N\big[\#\{y_i < c\} - \#\{y_i > c\} - 1,\; \#\{y_i < c\} - \#\{y_i > c\} + 1\big]$, which contains 0 exactly when at most half the points lie strictly on either side — the definition of a median. For odd $N$ the minimiser is the middle order statistic; for even $N$ every $c$ between the two middle values is a minimiser (the loss is flat there). Over functions, the same argument for each $x$ gives $f(x) = \operatorname{median}(y\mid x)$.`,
      },
      { t: "h", text: "MAE is Laplace maximum likelihood" },
      {
        t: "p",
        text: String.raw`With Laplace noise of scale $b$, $p(y\mid x) = \frac{1}{2b}e^{-|y - \hat y|/b}$, the negative log-likelihood is $N\log(2b) + \frac1b\sum_i |\hat y_i - y_i|$ — MAE up to constants. The Laplace density has exponential rather than Gaussian tails, so large residuals are "less surprising" and are charged linearly instead of quadratically.`,
      },

      { t: "h", text: "4. Worked example: one outlier" },
      {
        t: "p",
        text: String.raw`Targets $y = (2, 3, 4, 5, 26)$, constant predictions $c$. Mean $\bar y = 8$, median $4$. The gradient column is the derivative of the mean loss with respect to $c$, $\frac1N\sum_i \ell'(c - y_i)$.`,
      },
      {
        t: "table",
        head: ["", String.raw`residuals $c - y$`, "MSE / gradient", "MAE / gradient", String.raw`Huber$_{\delta=1}$ / gradient`, "log-cosh / gradient"],
        rows: [
          [String.raw`$c = 8$ (mean)`, String.raw`$(6, 5, 4, 3, -18)$`, "82.0 / 0", "7.2 / 0.6", "6.7 / 0.6", "6.507425 / 0.598856"],
          [String.raw`$c = 4$ (median)`, String.raw`$(2, 1, 0, -1, -22)$`, "98.0 / −8.0", "5.2 / 0", "4.8 / 0", "4.699883 / −0.007194"],
        ],
      },
      {
        t: "p",
        text: String.raw`At $c = 8$ the MSE gradient is zero: $2\cdot\frac15(6 + 5 + 4 + 3 - 18) = 0$. The four ordinary points push $c$ down with forces 12, 10, 8, 6 and the outlier pushes up with 36, exactly balancing them. Under MAE the same point has gradient $\frac15(1 + 1 + 1 + 1 - 1) = 0.6$, so MAE moves $c$ down to 4, where the clipped Huber gradient $\frac15(1 + 1 + 0 - 1 - 1) = 0$ also vanishes. The log-cosh minimiser, found numerically in the code below, is 4.0188 — just above the median, because $\tanh$ is not exactly $\pm1$ for the residuals 1 and 2.`,
      },

      { t: "h", text: "5. Outlier sensitivity" },
      {
        t: "p",
        text: String.raw`If one target is moved by $\Delta$, the mean moves by $\Delta/N$ — without bound as $\Delta \to \infty$. The median does not move at all until the corrupted point crosses the middle. Statisticians call the fraction of data that can be corrupted arbitrarily before the estimate breaks the **breakdown point**: $0$ for the mean (one point suffices, asymptotically), 50% for the median. In gradient terms: MSE's influence $2r$ is unbounded, MAE's $\pm 1$ is bounded. The experiment in the code section corrupts 10% of the diabetes training targets by +300 (a typical data-entry error): the MSE fit acquires a systematic bias of about +26 units on the clean test set, while MAE, Huber and log-cosh fits stay within a few units of the fit obtained on clean data.`,
      },
      {
        t: "callout",
        kind: "warning",
        title: "Outlier or signal?",
        text: "Robust losses ignore large residuals on purpose. If the rare extreme targets are real and costly to miss (insurance claims, peak electricity demand), down-weighting them is the wrong decision. Choose the loss for the quantity you need to predict, not only for the noise you want to ignore.",
      },

      { t: "h", text: "6. Huber loss (Huber, 1964)" },
      {
        t: "math",
        tex: String.raw`\mathrm{Huber}_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\big(|r| - \tfrac12\delta\big) & |r| > \delta \end{cases}, \qquad \mathrm{Huber}_\delta'(r) = \begin{cases} r & |r| \le \delta \\ \delta\,\operatorname{sign}(r) & |r| > \delta \end{cases} = \operatorname{clip}(r, -\delta, \delta)`,
      },
      {
        t: "p",
        text: String.raw`$\delta > 0$ is the threshold, in the **units of the target**. Peter Huber introduced it in "Robust Estimation of a Location Parameter" (1964). The linear branch is chosen so that the function and its first derivative are continuous at $|r| = \delta$: both branches give value $\tfrac12\delta^2$ and slope $\delta$ there. So Huber is once continuously differentiable (its second derivative jumps from 1 to 0). Its gradient is simply the **clipped residual**: residuals smaller than $\delta$ are treated as in least squares, larger ones contribute a capped force $\pm\delta$. For constant fits, the minimiser solves $\sum_i \operatorname{clip}(c - y_i, -\delta, \delta) = 0$; as $\delta\to\infty$ it becomes the mean and as $\delta\to 0$ it becomes the median (with $\mathrm{Huber}_\delta/\delta \to |r|$).`,
      },
      {
        t: "p",
        text: String.raw`**Choosing $\delta$.** $\delta$ should separate "ordinary noise" from "gross error". A classical rule for Gaussian noise with standard deviation $\sigma$ is $\delta = 1.345\,\sigma$, which keeps 95% of the efficiency of least squares when there are no outliers while bounding the influence when there are. In practice $\sigma$ is unknown and is estimated robustly, e.g. $\hat\sigma = 1.4826\cdot\mathrm{median}|r_i - \mathrm{median}(r)|$ (the scaled median absolute deviation) from a preliminary fit — or $\delta$ is tuned on validation data. Because $\delta$ has units, it must be re-chosen whenever the targets are rescaled.`,
      },
      {
        t: "p",
        text: String.raw`**Smooth L1** (used for bounding-box regression since Fast R-CNN, Girshick 2015) is Huber divided by its threshold: PyTorch's SmoothL1Loss with parameter $\beta$ equals $\mathrm{Huber}_\beta(r)/\beta$, i.e. $0.5r^2/\beta$ for $|r| < \beta$ and $|r| - 0.5\beta$ otherwise. Its large-residual slope is always 1, whereas Huber's is $\delta$.`,
      },

      { t: "h", text: "7. Log-cosh loss" },
      {
        t: "math",
        tex: String.raw`\ell(r) = \log\cosh r = \log\frac{e^{r} + e^{-r}}{2}, \qquad \ell'(r) = \tanh r, \qquad \ell''(r) = \operatorname{sech}^2 r = 1 - \tanh^2 r`,
      },
      {
        t: "p",
        text: String.raw`Log-cosh is infinitely differentiable — no kink at all — with a gradient that saturates smoothly at $\pm1$ (a "soft" Huber with $\delta = 1$). Its two regimes follow from Taylor expansion and from $\cosh r = \frac12 e^{|r|}(1 + e^{-2|r|})$:`,
      },
      {
        t: "math",
        tex: String.raw`\log\cosh r = \frac{r^2}{2} - \frac{r^4}{12} + O(r^6) \quad (r\to 0), \qquad \log\cosh r = |r| - \log 2 + \log\!\big(1 + e^{-2|r|}\big) \;\to\; |r| - \log 2 \quad (|r|\to\infty)`,
      },
      {
        t: "table",
        head: [String.raw`$r$`, String.raw`$\log\cosh r$`, String.raw`$r^2/2$`, String.raw`$|r| - \log 2$`, String.raw`$\tanh r$`],
        rows: [
          ["0.1", "0.004992", "0.005", "−0.593147", "0.099668"],
          ["0.5", "0.120115", "0.125", "−0.193147", "0.462117"],
          ["2", "1.325003", "2.0", "1.306853", "0.964028"],
          ["10", "9.306853", "50.0", "9.306853", "1.000000"],
        ],
      },
      {
        t: "p",
        text: String.raw`The second form is also the **numerically stable** implementation: computing $\cosh(r)$ directly overflows float32 for $|r| > 89$, while $|r| + \mathrm{log1p}(e^{-2|r|}) - \log 2$ never exponentiates a positive number. Log-cosh has a fixed transition scale of about 1 target unit, so it only behaves as intended when targets are normalised.`,
      },

      { t: "h", text: "8. Quantile (pinball) loss and prediction intervals" },
      {
        t: "p",
        text: String.raw`For a quantile level $\tau \in (0, 1)$ and $u = y - \hat y = -r$ (positive when we under-predict), the **pinball loss** of Koenker and Bassett ("Regression Quantiles", 1978) is`,
      },
      {
        t: "math",
        tex: String.raw`\rho_\tau(u) = \max\big(\tau u,\; (\tau - 1)u\big) = \begin{cases} \tau\, u & u \ge 0 \;\text{(under-prediction)}\\ (1 - \tau)(-u) & u < 0 \;\text{(over-prediction)} \end{cases}, \qquad \frac{\partial \rho_\tau}{\partial \hat y} = \begin{cases} -\tau & y > \hat y\\ 1 - \tau & y < \hat y \end{cases}`,
      },
      {
        t: "p",
        text: String.raw`**The minimiser is the $\tau$-quantile.** For a constant $q$, the derivative of $\frac1N\sum_i \rho_\tau(y_i - q)$ is $\frac1N\big[(1-\tau)\#\{y_i < q\} - \tau\,\#\{y_i > q\}\big] = F_N(q) - \tau$ (ignoring ties), where $F_N(q)$ is the fraction of data below $q$. It is negative while $F_N(q) < \tau$ and positive after, so the minimum is where a fraction $\tau$ of the data lies below $q$ — the empirical $\tau$-quantile. $\tau = 0.5$ gives $\rho_{0.5}(u) = \tfrac12|u|$, i.e. MAE and the median.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $\tau = 0.9$, $y = 10$. Predicting $\hat y = 8$ ($u = 2$) costs $0.9\times2 = 1.8$; predicting $\hat y = 12$ ($u = -2$) costs $0.1\times2 = 0.2$. Under-prediction is nine times more expensive, so the fitted value rises until 90% of targets lie below it. Training two heads with $\tau = 0.05$ and $\tau = 0.95$ gives a **90% prediction interval** whose width can vary with $x$; in the code below the interval covers 89.8% of held-out points from data whose noise grows with $x$.`,
      },

      { t: "h", text: "9. RMSE versus MSE" },
      {
        t: "p",
        text: String.raw`$\mathrm{RMSE} = \sqrt{\mathrm{MSE}}$ has the **units of the target** (MSE has squared units), which is why it is the usual reporting metric: a test MSE of $3705$ on the diabetes target is hard to read; the equivalent RMSE of $60.87$ units is directly comparable with the target's standard deviation (about 77). Because the square root is monotone, MSE and RMSE have the **same minimiser**. Their gradients differ by a data-dependent factor,`,
      },
      { t: "math", tex: String.raw`\frac{\partial\,\mathrm{RMSE}}{\partial \hat y_i} = \frac{1}{2\sqrt{\mathrm{MSE}}}\,\frac{\partial\,\mathrm{MSE}}{\partial \hat y_i} = \frac{\hat y_i - y_i}{N\cdot\mathrm{RMSE}}` },
      {
        t: "p",
        text: String.raw`so training on RMSE acts like MSE with an adaptive learning rate $\eta/(2\,\mathrm{RMSE})$ that grows as the error shrinks (and is undefined at zero error). Also note $\mathrm{MAE} \le \mathrm{RMSE} \le \sqrt{N}\,\mathrm{MAE}$; a large ratio RMSE/MAE signals a few large errors. Train on MSE, report RMSE (and MAE). Metrics are covered further in [Regression Metrics](/learn/m17/regression-metrics).`,
      },

      { t: "h", text: "10. Units, scale and target normalisation" },
      {
        t: "p",
        text: String.raw`If targets are multiplied by a factor $s$ (say, prices in cents instead of euros, $s = 100$), the MSE is multiplied by $s^2 = 10^4$ and its gradient by $s$; MAE and its gradient by $s$ and 1. With a fixed learning rate, a model that trained well in euros may diverge in cents. Huber's $\delta$ and log-cosh's built-in scale of 1 silently change meaning. Two practical consequences:`,
      },
      {
        t: "list",
        items: [
          String.raw`**Standardise targets**: train on $\tilde y = (y - m)/s$ with $m, s$ the training mean and standard deviation (or median and IQR if there are outliers), and map predictions back with $\hat y = s\,\hat{\tilde y} + m$. The initial loss is then $O(1)$, the output layer's default initialisation is appropriate, and $\delta = 1$ means "one standard deviation".`,
          String.raw`**Multi-output regression**: summing MSE over outputs with different units (metres and degrees, say) lets the output with the largest variance dominate. Normalise each output, or weight them — the heteroscedastic NLL below learns such weights automatically, one variance per output.`,
          String.raw`**Positive, skewed targets** (prices, counts) are often modelled as $\log y$: MSE on $\log y$ penalises relative rather than absolute errors and its minimiser is the conditional mean of $\log y$ (the geometric mean, not the mean, of $y$).`,
        ],
      },

      { t: "h", text: "11. Heteroscedastic Gaussian NLL: predicting the variance" },
      {
        t: "p",
        text: String.raw`**Heteroscedastic** means the noise level depends on the input. Let the network output a mean $\mu(x)$ and a variance $\sigma^2(x) > 0$, and assume $y \mid x \sim \mathcal{N}(\mu(x), \sigma^2(x))$ (Nix and Weigend, 1994, proposed exactly this mean-and-variance network; Kendall and Gal, 2017, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", popularised it as *aleatoric* uncertainty). The per-example negative log-likelihood is`,
      },
      {
        t: "math",
        tex: String.raw`\ell_i = \frac12\log\big(2\pi\sigma_i^2\big) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}, \qquad \mu_i = \mu(x_i),\; \sigma_i^2 = \sigma^2(x_i)`,
      },
      {
        t: "p",
        text: String.raw`The first term punishes claiming large uncertainty; the second punishes errors, but divided by the predicted variance, so the network can "explain away" an error by declaring a point noisy — at the price of the log term. Parameterise $s_i = \log\sigma_i^2$ (an unconstrained network output; $\sigma_i^2 = e^{s_i}$ is always positive). The gradients are`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial \ell_i}{\partial \mu_i} = \frac{\mu_i - y_i}{\sigma_i^2}, \qquad \frac{\partial \ell_i}{\partial s_i} = \frac12\left(1 - \frac{(y_i - \mu_i)^2}{\sigma_i^2}\right)`,
      },
      {
        t: "p",
        text: String.raw`Setting the second to zero gives $\sigma_i^2 = (y_i - \mu_i)^2$: at the optimum the predicted variance equals the expected squared residual at that input. The first shows that the mean is fitted by **weighted** least squares with weights $1/\sigma_i^2$ — noisy regions get less say. With a constant $\sigma$, the NLL is MSE scaled by $1/(2\sigma^2)$ plus a constant (section 2).`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $y = 3$, $\mu = 2.5$, $\sigma^2 = 0.25$: $\ell = \frac12\log(2\pi\cdot0.25) + \frac{(0.5)^2}{2\cdot0.25} = \frac12\log(1.570796) + 0.5 = 0.225791 + 0.5 = 0.725791$. The gradient with respect to $\mu$ is $(2.5 - 3)/0.25 = -2$ and with respect to $s = \log\sigma^2$ it is $\frac12(1 - 0.25/0.25) = 0$: this variance is already optimal for this residual. Predicting $\sigma^2 = 1$ instead gives $\ell = 1.043939$, and $\sigma^2 = 0.0625$ gives $1.532644$ — both worse.`,
      },
      {
        t: "callout",
        kind: "warning",
        title: "Known failure mode",
        text: String.raw`Because the mean's gradient is divided by $\sigma_i^2$, points the network has declared noisy stop influencing the mean, and the fit of the mean can get stuck where it is poor. Common remedies: warm-start the mean with MSE before switching on the variance head, clamp the log-variance to a sensible range, or use the β-NLL variant of Seitzer et al. (2022), which multiplies each example's loss by a stop-gradient factor $\sigma_i^{2\beta}$.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "sim", id: "loss-functions", caption: "Loss Function Explorer — Regression tab: MSE, MAE, Huber and log-cosh and their derivatives as functions of the residual." },
      {
        t: "list",
        items: [
          String.raw`Stay on the **Regression** tab and press **Play**: the marker sweeps the residual from −3 to 3. Watch the lower panel: MSE's derivative is the straight line $2r$, MAE's is the step $\pm1$, Huber's is the clipped line, log-cosh's is the smooth $\tanh r$.`,
          String.raw`Stop at $r = 3$ and read the table: MSE's gradient 6 is six times MAE's. In the worked example the outlier's residual was 18–22 — off this chart, with an MSE gradient near 40.`,
          String.raw`Drag **Huber δ** to 0.1: Huber's derivative becomes almost a step (MAE-like, scaled by δ). Drag it to 3: over the whole plotted range Huber equals $\tfrac12 r^2$ and its derivative equals $r$ — half of MSE's.`,
          String.raw`Compare log-cosh with Huber at δ = 1 near $r = 0$ (they coincide, both $\approx r^2/2$) and at $r = 3$ (log-cosh ≈ 2.309, Huber 2.5: both grow linearly with slope 1, offset by $\log 2 = 0.693$ versus $0.5$).`,
          String.raw`Untick MSE to rescale your attention to the robust losses: all three have bounded derivatives — that is the precise meaning of "robust to outliers".`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "Every loss from this lesson in NumPy with its derivative with respect to the residual, a finite-difference check, the minimising constant of each loss on the outlier example, a linear model fitted by gradient descent under each loss on the real *diabetes* dataset (442 patients, 10 standardised clinical features, target = disease progression one year later) with 10% of the training targets corrupted, and a linear quantile regression producing a 90% prediction interval.",
      },
      { t: "code", lang: "python", title: "regression_losses_from_scratch.py", code: SCRATCH, output: SCRATCH_OUT },
      {
        t: "list",
        items: [
          String.raw`**Gradient check.** The analytic derivatives of MSE, Huber and log-cosh agree with central differences to $\sim10^{-10}$. MAE and pinball are not checked: their derivative is undefined at the kink, and a finite difference straddling it would be meaningless.`,
          String.raw`**Constants.** MSE picks the mean 8, MAE and Huber the median 4, log-cosh 4.0188, exactly as derived in the worked example.`,
          String.raw`**Diabetes with corrupted targets.** The test set is clean. The MSE fit has a mean test residual of +25.82: the 33 inflated targets drag every prediction upwards, and its test RMSE (60.87) and MAE (49.46) are the worst. MAE, Huber ($\delta = 0.5$ standardised units) and log-cosh fits reach test MAE 44.6–45.0, essentially the 44.72 of least squares trained on the **clean** targets. Log-cosh keeps a larger bias (+10.21) because, after standardisation with the inflated standard deviation (123.5), a +300 error is only about 2.4 units — not far into its linear regime. Note that RMSE barely separates the robust fits; the bias column shows the damage most clearly.`,
          String.raw`**Decaying step size.** The update uses $\eta_k = 0.15/\sqrt{1 + k/4000}$. With a constant step, subgradient descent on MAE oscillates around the kink forever, because the gradient never shrinks; the decay makes it converge (checked separately: the final training MAE, in standardised units, is within $2\times10^{-4}$ of the exact minimum computed by linear programming).`,
          String.raw`**Quantile regression.** The 0.05- and 0.95-quantile lines fan out (slopes 0.040 and 1.026) because the noise grows with $x$; the held-out fractions below them are 3.6% and 93.4%, and the interval covers 89.8% of test points. The median line recovers the true intercept and slope (1.0, 0.5).`,
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
                text: "Part 1 evaluates the built-in regression losses on residuals 0, 0.5, 2 and 10 and reads their gradients from autograd (log-cosh is not built in, so it is written in the stable form). Part 2 trains two small MLPs on data whose noise grows with x: one with MSE (plus a single residual standard deviation estimated afterwards), one with nn.GaussianNLLLoss predicting a mean and a log-variance.",
              },
              { t: "code", lang: "python", title: "regression_losses_torch.py", code: TORCH, output: TORCH_OUT },
              {
                t: "list",
                items: [
                  String.raw`The MSE, L1 and Huber rows reproduce the table of the classification lesson; $\mathrm{SmoothL1}_{\beta=2}$ is exactly $\mathrm{Huber}_{\delta=2}/2$ (compare 9.0 with 18.0), and its large-residual gradient is 1 instead of 2. L1Loss's gradient at $r = 0$ is 0: PyTorch uses sign(0) = 0 as the subgradient.`,
                  String.raw`Both MLPs have the same test MSE (≈0.087): predicting the variance does not hurt the mean here. But the constant-σ model gives 100% coverage where the noise is small ($x < -0.5$, its intervals are far too wide) and only 76.7% where it is large ($x > 0.5$, too narrow) — the 89.8% overall average hides two failures. The NLL model's coverage is 94.2% and 90.5% in the two regions, its test NLL is lower (−0.033 vs 0.195), and its predicted σ at $x = -0.9, 0, 0.9$ (0.072, 0.258, 0.442) tracks the true values (0.072, 0.275, 0.477).`,
                  "nn.GaussianNLLLoss takes the **variance** (not the standard deviation or log-variance) as its third argument and omits the constant ½·log 2π by default (full=False); it also clamps the variance from below with eps = 1e-6 for stability.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras provides MeanSquaredError, MeanAbsoluteError, Huber and LogCosh as built-in losses; the pinball and Gaussian NLL losses are a few lines of custom code. A custom loss receives (y_true, y_pred) and returns one value per example.",
              },
              {
                t: "code",
                lang: "python",
                title: "regression_losses_keras.py (not executed in CI)",
                code: KERAS,
                output: "Prints the four built-in losses on residuals (0, 0.5, 2, 10): MSE 26.0625, MAE 3.125, Huber 2.78125 and log-cosh ≈ 2.688 (each averaged over the four examples), then the predicted noise standard deviations at x = −0.9, 0, 0.9 (increasing, close to the true 0.072, 0.275, 0.477), and the fraction of targets below the 0.9-quantile network's predictions (close to 0.9). Exact values depend on the TensorFlow version and initialisation.",
              },
              {
                t: "list",
                items: [
                  "Keras's argument order is (y_true, y_pred) — the reverse of PyTorch's (input, target). For the asymmetric pinball loss swapping them silently turns the τ-quantile into the (1 − τ)-quantile.",
                  "Keras's Huber uses the same definition as PyTorch's HuberLoss (½r² inside, δ(|r| − ½δ) outside).",
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
        head: ["Loss", "Definition", "Advantages", "Disadvantages", "Use when"],
        rows: [
          ["MSE (L2)", String.raw`$r^2$`, "smooth, convex, closed form for linear models; estimates the mean; Gaussian MLE", "unbounded influence: outliers dominate; squared units", "noise roughly Gaussian, large errors genuinely costly"],
          ["MAE (L1)", String.raw`$|r|$`, "bounded influence; estimates the median; same units as $y$", "kink at 0; constant gradient magnitude makes fine convergence slow without step decay", "heavy-tailed noise, corrupted labels, median forecasts"],
          ["Huber / smooth L1", String.raw`quadratic for $|r| \le \delta$, linear beyond`, "MSE precision near 0, MAE robustness far away; gradient = clipped residual", String.raw`extra hyperparameter $\delta$ in target units`, "default robust loss; box regression; TD targets in RL"],
          ["Log-cosh", String.raw`$\log\cosh r$`, "smooth everywhere (all derivatives), soft gradient cap", "fixed transition scale ≈ 1: needs normalised targets", "want Huber-like behaviour with second derivatives (e.g. Newton-type or gradient-boosting methods)"],
          ["Quantile / pinball", String.raw`$\max(\tau u, (\tau - 1)u)$`, "any quantile, prediction intervals, asymmetric costs", "one model (or head) per quantile; quantile lines can cross", "forecasting, risk, service-level targets"],
          ["Gaussian NLL", String.raw`$\tfrac12\log\sigma^2 + r^2/(2\sigma^2)$`, "predicts input-dependent uncertainty; automatic weighting of noisy points", "can neglect the mean in noisy regions; needs variance positivity and clamping", "uncertainty-aware regression, multi-task balancing"],
          ["Laplace NLL", String.raw`$\log b + |r|/b$`, "robust and uncertainty-aware", "kink at 0", "heavy-tailed noise with varying scale"],
          ["MSE on log targets", String.raw`$(\log\hat y - \log y)^2$`, "relative errors; positive skewed targets", "biased for the mean of $y$; needs $y > 0$", "prices, counts, durations"],
          ["Tukey biweight", "bounded loss (constant beyond a threshold)", "outliers get exactly zero influence", "non-convex: local minima, needs good initialisation", "severe gross errors, e.g. pose estimation"],
        ],
      },
      {
        t: "p",
        text: "Percentage-type losses (MAPE, |r|/|y|) are sometimes used for business forecasting; they are undefined at y = 0 and penalise over-prediction more than under-prediction in a way that biases the fit low, so they are better used as reporting metrics than as training losses.",
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Object detection.** Fast R-CNN (Girshick, 2015) introduced the smooth L1 loss for bounding-box offsets because it was less sensitive to outliers than the L2 loss used earlier; Faster R-CNN and many later detectors inherited it.",
          "**Deep Q-networks.** The Atari DQN (Mnih et al., 2015) clipped the temporal-difference error term to [−1, 1], which is equivalent to optimising the Huber loss with δ = 1 on the TD error; Huber is a common default for value learning since.",
          "**Image restoration and super-resolution.** L1 is frequently preferred over L2 for denoising, deblurring and super-resolution because L2 averages plausible outputs into blur and over-weights a few large pixel errors; Zhao, Gallo, Frosio and Kautz (2017, \"Loss Functions for Image Restoration with Neural Networks\") compare these choices systematically.",
          "**Probabilistic forecasting.** Demand, energy-load and weather forecasts are evaluated and trained with pinball losses at several quantiles so that planners get intervals, not only point forecasts.",
          "**Uncertainty-aware perception.** Depth estimation and other dense regression networks predict a per-pixel variance with the Gaussian (or Laplace) NLL, as in Kendall and Gal (2017), so downstream systems can discount unreliable pixels.",
          "**Tabular regression in practice.** House-price and insurance models often train on log targets or with Huber loss; gradient-boosting libraries expose the same choices (squared, absolute, Huber, quantile objectives).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Efficiency versus robustness.** With clean Gaussian noise MSE is the most statistically efficient choice (smallest variance of the estimate). Every robust loss gives up some efficiency to gain bounded influence; Huber with δ ≈ 1.345σ gives up only about 5%.",
          "**Smoothness versus robustness.** MAE is robust but has a kink; Huber is once-differentiable; log-cosh is infinitely differentiable but its scale is fixed. Kinks matter for second-order methods and make fixed-step gradient descent oscillate.",
          "**Which statistic you need.** Mean (MSE), median (MAE) and quantiles (pinball) are different answers for skewed targets. A delivery-time model trained with MSE predicts an average that is late more than half the time if the distribution has a long right tail.",
          "**Scale dependence.** MSE's gradient scales with the target, Huber's δ and log-cosh's transition have units; all are sensitive to target normalisation. Standardising targets removes most of this fragility.",
          "**Predicting uncertainty costs stability.** The Gaussian NLL gives calibrated intervals when it works, but adds a second output, interacts with the mean's fit and needs clamping or warm-up.",
          "**Robustness is not a cure for bad data.** Robust losses limit the damage of gross errors; they cannot recover a target that is systematically biased (e.g. all values in the wrong unit).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Shape mismatch that broadcasts.** A prediction of shape (N, 1) compared with a target of shape (N,) broadcasts to an (N, N) matrix of all pairwise differences; the loss is finite but wrong and the model learns to predict the global mean. PyTorch's MSELoss emits a warning; a hand-written ((pred - y) ** 2).mean() does not. Fix: squeeze or reshape so both have the same shape, and assert it.",
          String.raw`**Putting an activation on the output.** A ReLU or sigmoid on a regression output restricts its range (no negative values, or only $[0,1]$) and can kill gradients. Fix: a plain linear output; enforce positivity only when the target really is positive, preferably via a log target or softplus.`,
          String.raw`**Not normalising targets.** Targets in the thousands give initial MSE in the millions, huge gradients and divergence, or force a tiny learning rate. Fix: standardise $y$ with training statistics and invert the transform at prediction time.`,
          String.raw`**Forgetting to invert the normalisation** before computing metrics, or computing the normalisation statistics on the whole dataset (leaking test information). Fix: fit $m, s$ on the training split only.`,
          String.raw`**Choosing Huber's δ in the wrong units.** $\delta = 1$ is "one standard deviation" for standardised targets but "one cent" for prices in cents (where Huber becomes MAE). Fix: set δ relative to a robust estimate of the residual scale.`,
          "**Reporting MSE as if it were an error in target units.** An MSE of 3705 is an RMSE of about 61. Fix: report RMSE and MAE alongside each other.",
          "**Training MAE with a constant learning rate and expecting convergence.** The subgradient has magnitude 1 even at the optimum, so iterates keep jumping across it. Fix: decay the learning rate (e.g. $\eta_k \propto 1/\sqrt{k}$), or use Huber with a small δ, whose gradient does shrink near the optimum.",
          String.raw`**Predicting σ or σ² directly with a linear output.** It can become negative or zero, giving NaN or infinite loss. Fix: predict $\log\sigma^2$ (or use softplus) and clamp it to a reasonable range.`,
          "**Swapping arguments in an asymmetric loss.** pinball(y_pred, y_true) instead of pinball(y_true, y_pred) silently fits the (1 − τ)-quantile. Fix: test the trained model's coverage on held-out data.",
          "**Using sum reduction then tuning the learning rate.** With reduction='sum' the gradient grows with the batch size, so changing the batch size changes the effective learning rate. Fix: use the mean unless you have a reason not to.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is a residual, and why do regression losses depend only on it?", a: String.raw`The residual is $r = \hat y - y$, the signed prediction error. Standard regression losses are translation invariant — an error of 2 costs the same whether the target is 10 or 1000 — so they can be written as $\ell(r)$. Their derivative $\ell'(r)$ is the signal backpropagation sends into the network, since $\partial r/\partial\hat y = 1$. (Losses on log targets or percentages are exceptions: they measure relative error.)` },
      { t: "qa", level: "beginner", q: "MSE or MAE: which is more sensitive to outliers, and why?", a: String.raw`MSE. Its gradient $2r$ grows linearly with the error, so one point with a huge residual pulls harder than many ordinary points combined, and the optimal constant (the mean) moves without bound as one target moves. MAE's gradient has magnitude 1 regardless of the error size, and its optimum (the median) does not move when one extreme value is made more extreme.` },
      { t: "qa", level: "beginner", q: "Why report RMSE rather than MSE?", a: "RMSE is in the units of the target, so it can be compared directly with the target's spread and with MAE. MSE is in squared units. Both have the same minimiser, so training on MSE and reporting RMSE is consistent." },
      { t: "qa", level: "beginner", q: "Why should regression targets usually be standardised?", a: String.raw`So that the loss and gradients start at a scale of order 1, the default initialisation of the output layer is appropriate, the learning rate does not depend on the target's units, and scale-dependent hyperparameters (Huber's $\delta$, log-cosh's transition) have a meaning. Predictions are mapped back with $\hat y = s\hat{\tilde y} + m$ using training statistics.` },
      { t: "qa", level: "intermediate", q: "Prove that the constant minimising mean squared error is the mean and the one minimising mean absolute error is the median.", a: String.raw`MSE: $\frac{d}{dc}\frac1N\sum(c - y_i)^2 = 2(c - \bar y)$, zero only at $c = \bar y$, and the function is convex. MAE: away from data points $\frac{d}{dc}\frac1N\sum|c - y_i| = (\#\{y_i < c\} - \#\{y_i > c\})/N$, negative while fewer than half the points are below $c$ and positive when more than half are; at a data point 0 lies in the subdifferential exactly when at most half the points are on each side — the median. For even $N$ the whole interval between the middle values is optimal.` },
      { t: "qa", level: "intermediate", q: "What is the gradient of the Huber loss and why is it attractive?", a: String.raw`$\mathrm{Huber}_\delta'(r) = \operatorname{clip}(r, -\delta, \delta)$. Small residuals get the proportional least-squares gradient, which converges precisely near the optimum, and large residuals get a capped gradient $\pm\delta$, which bounds the influence of outliers. The function and its first derivative are continuous at $|r| = \delta$, so it is easier to optimise than MAE.` },
      { t: "qa", level: "intermediate", q: "How do you obtain a 90% prediction interval from a neural network without assuming Gaussian noise?", a: String.raw`Train two outputs with the pinball loss at $\tau = 0.05$ and $\tau = 0.95$. The minimiser of the expected pinball loss is the $\tau$-quantile of $y\mid x$, so the outputs estimate the 5th and 95th percentiles and the interval between them should contain 90% of new targets. Verify coverage on held-out data; also guard against crossing quantiles (e.g. predict the lower quantile and a positive width).` },
      { t: "qa", level: "intermediate", q: "Relate MSE and MAE to maximum likelihood.", a: String.raw`With Gaussian noise of fixed variance, $-\log p(y\mid x) = \frac{(y - \hat y)^2}{2\sigma^2} + \text{const}$, so maximum likelihood is least squares. With Laplace noise of fixed scale $b$, $-\log p = |y - \hat y|/b + \text{const}$, so maximum likelihood is least absolute deviations. The choice of loss is a choice of noise model; heavy-tailed noise calls for MAE-like losses.` },
      { t: "qa", level: "advanced", q: "Derive the optimal predicted variance in the heteroscedastic Gaussian NLL, and explain a failure mode.", a: String.raw`Per example $\ell = \tfrac12 s + \tfrac{r^2}{2}e^{-s}$ with $s = \log\sigma^2$. $\partial\ell/\partial s = \tfrac12(1 - r^2e^{-s})$, zero at $\sigma^2 = r^2$, so in expectation $\sigma^2(x) = \mathbb{E}[(y - \mu(x))^2\mid x]$. The mean's gradient is $r/\sigma^2$: weighted least squares. Failure mode: if the network inflates $\sigma^2$ in a region early in training, that region's residuals barely affect the mean, so a poor mean fit can be "explained" as noise and never corrected. Remedies: MSE warm-up, clamping $s$, or β-NLL weighting.` },
      { t: "qa", level: "advanced", q: "Why does subgradient descent on MAE need a decaying learning rate while gradient descent on MSE does not?", a: String.raw`For MSE the gradient is proportional to the residuals and goes to zero at the optimum, so a fixed small step converges (for a learning rate below $2/L$, $L$ the smoothness constant). The MAE subgradient does not shrink as the optimum is approached — each example contributes $\pm1$ whatever the size of its residual — so with a fixed step $\eta$ the iterates keep overshooting and oscillate in a band whose width is proportional to $\eta$. Convergence requires steps with $\sum\eta_k = \infty$ and $\eta_k\to0$, e.g. $\eta_k \propto 1/\sqrt{k}$. Huber with small δ is a smooth alternative.` },
      { t: "qa", level: "advanced", q: "You train a model on prices with MSE and then switch the target to log(price). What changes in what the model estimates?", a: String.raw`MSE on $\log y$ estimates $\mathbb{E}[\log y\mid x]$; exponentiating gives the conditional **geometric** mean, which is below the arithmetic mean for skewed data (for log-normal noise with variance $s^2$, by a factor $e^{-s^2/2}$). Errors become relative: being 10% off costs the same for cheap and expensive items. If the arithmetic mean is needed, correct the bias (e.g. multiply by $e^{\hat s^2/2}$) or train with a loss on the original scale.` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Residuals (1, −2, 3). What is the MSE?", answer: 4.6667, tolerance: 0.001, explain: "(1 + 4 + 9)/3 = 14/3 ≈ 4.6667." },
          { kind: "numeric", q: "Same residuals (1, −2, 3). What is the RMSE?", answer: 2.1602, tolerance: 0.001, explain: "√4.6667 ≈ 2.1602. (The MAE is 2, so RMSE ≥ MAE as always.)" },
          { kind: "numeric", q: "Huber loss with δ = 1 for a residual r = 3?", answer: 2.5, tolerance: 0.001, explain: "|r| > δ, so δ(|r| − δ/2) = 1 × (3 − 0.5) = 2.5. Its gradient is clipped at 1." },
          { kind: "numeric", q: "Pinball loss with τ = 0.9 for target y = 10 and prediction ŷ = 8?", answer: 1.8, tolerance: 0.001, explain: "Under-prediction (u = y − ŷ = 2 > 0) costs τu = 0.9 × 2 = 1.8. Over-predicting by 2 would cost only 0.2." },
          { kind: "numeric", q: "A Gaussian-NLL model sees residual y − μ = 0.5. Which predicted variance σ² minimises this example's loss?", answer: 0.25, tolerance: 0.001, explain: "∂ℓ/∂(log σ²) = ½(1 − r²/σ²) = 0 at σ² = r² = 0.25." },
          { kind: "numeric", q: "log cosh(10) to three decimals (use the large-residual approximation |r| − log 2)?", answer: 9.307, tolerance: 0.002, explain: "10 − 0.693147 = 9.306853; the correction log(1 + e^{−20}) is about 2e-9." },
          { q: "Targets (2, 3, 4, 5, 26). Which constant minimises the mean absolute error?", options: ["8 (the mean)", "4 (the median)", "5", "Any value between 2 and 26"], answer: 1, explain: "MAE is minimised by the median; with five values that is the third smallest, 4." },
          { q: "Your house prices are converted from thousands of euros to euros (×1000). With MSE and the same learning rate, what happens to the gradient with respect to the output?", options: ["Unchanged", "Multiplied by 1000", "Multiplied by 10⁶", "Divided by 1000"], answer: 1, explain: "∂MSE/∂ŷ = 2r/N scales linearly with r, so by 1000; the loss value itself scales by 10⁶. Standardising targets avoids this." },
          { q: "Which statement about log-cosh loss is correct?", options: ["It is non-convex", "Its gradient is tanh(r), so it behaves like r²/2 near 0 and like |r| − log 2 far away", "It equals Huber loss with δ = 1 exactly", "It cannot be computed for |r| > 20"], answer: 1, explain: "ℓ' = tanh r; Taylor gives r²/2 near 0 and the asymptote |r| − log 2. It is close to, but not equal to, Huber with δ = 1, and the stable formula works for any r." },
        ],
      },
      {
        t: "exercise",
        title: "Mean, median and Huber on a contaminated sample",
        prompt: "Targets y = (1, 2, 3, 4, 100). (a) Which constants minimise MSE and MAE? (b) Compute the MSE at the mean and the MAE at the median. (c) Show that c = 3 is also a minimiser of the Huber loss with δ = 2, and compute the mean Huber loss there. (d) What is the MSE gradient (derivative of the mean loss with respect to c) at c = 3?",
        hint: "For (c) the Huber gradient is the mean of clip(c − yᵢ, −δ, δ).",
        solution: [
          { t: "p", text: String.raw`(a) The mean $\bar y = 110/5 = 22$ minimises MSE; the median 3 minimises MAE.` },
          { t: "p", text: String.raw`(b) At $c = 22$ the residuals are $(21, 20, 19, 18, -78)$: MSE $= (441 + 400 + 361 + 324 + 6084)/5 = 7610/5 = 1522$. At $c = 3$ the absolute residuals are $(2, 1, 0, 1, 97)$: MAE $= 101/5 = 20.2$.` },
          { t: "p", text: String.raw`(c) At $c = 3$, $c - y = (2, 1, 0, -1, -97)$, clipped to $[-2, 2]$: $(2, 1, 0, -1, -2)$, mean 0 — so the Huber gradient vanishes and (by convexity) $c = 3$ is a minimiser. Loss: $\tfrac12(4) + \tfrac12(1) + 0 + \tfrac12(1) + 2(97 - 1) = 2 + 0.5 + 0.5 + 192 = 195$; mean $195/5 = 39$.` },
          { t: "p", text: String.raw`(d) $2(c - \bar y) = 2(3 - 22) = -38$: MSE would still push hard towards 22, entirely because of the single value 100.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: the pinball loss estimates quantiles",
        prompt: "Write pinball(y, q, tau) for a constant prediction q. On 1001 samples from an exponential distribution (scale 2, seed 1), find the minimiser of the mean pinball loss on a fine grid for τ = 0.1, 0.5, 0.9 and compare it with np.quantile. Then check numerically that the slope of the mean pinball loss at q = 3 equals P(y < q) − τ.",
        hint: "u = y − q; loss = max(τu, (τ − 1)u). Use a grid of step 1e-4 on [0, 12].",
        solution: [
          { t: "code", lang: "python", title: "pinball_quantiles.py", code: EXERCISE, output: EXERCISE_OUT },
          { t: "p", text: String.raw`The grid minimiser agrees with np.quantile to three decimals for every $\tau$, and exactly the fraction $\tau$ of the data lies below it. The central-difference slope at $q = 3$ equals $F_N(3) - \tau = -0.1248$: 77.52% of the samples are below 3, fewer than the target 90%, so increasing $q$ still lowers the loss. Note how far apart the three quantiles of this skewed distribution are (0.21, 1.35, 4.67): a single "best prediction" depends entirely on which loss defines "best".` },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`A regression loss is a function of the residual $r = \hat y - y$; its derivative $\ell'(r)$ is each example's pull on the model (its influence).`,
          String.raw`**MSE** $r^2$, gradient $2r$: estimates the conditional **mean**; equals Gaussian maximum likelihood with fixed variance; unbounded influence, so outliers dominate.`,
          String.raw`**MAE** $|r|$, subgradient $\operatorname{sign}(r)$ (any value in $[-1, 1]$ at 0): estimates the **median**; Laplace maximum likelihood; bounded influence; needs a decaying step size.`,
          String.raw`**Huber** $\tfrac12 r^2$ for $|r| \le \delta$, $\delta(|r| - \tfrac12\delta)$ beyond; gradient $\operatorname{clip}(r, -\delta, \delta)$; δ is in target units (≈ 1.345σ classically). Smooth L1 = Huber$_\beta/\beta$.`,
          String.raw`**Log-cosh** $\log\cosh r$, gradient $\tanh r$; $\approx r^2/2$ near 0 and $|r| - \log 2$ far away; smooth everywhere; compute as $|r| + \mathrm{log1p}(e^{-2|r|}) - \log 2$.`,
          String.raw`**Pinball** $\max(\tau u, (\tau - 1)u)$, $u = y - \hat y$: estimates the **$\tau$-quantile**; two quantiles give a prediction interval.`,
          String.raw`**RMSE** $= \sqrt{\mathrm{MSE}}$: same minimiser, target units — train on MSE, report RMSE and MAE.`,
          "**Units matter**: MSE scales with the square of the target scale. Standardise targets with training statistics, and invert for reporting.",
          String.raw`**Gaussian NLL** $\tfrac12\log\sigma^2 + r^2/(2\sigma^2)$ with predicted $\log\sigma^2$: optimal $\sigma^2 = $ expected squared residual; the mean is fitted by weighted least squares with weights $1/\sigma^2$.`,
          "On diabetes with 10% corrupted training targets, MSE's test predictions were biased by +26; MAE and Huber matched a model trained on clean data.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Classification Losses](/learn/m05/classification-losses) — the discrete-target counterpart: cross-entropy as Bernoulli/categorical maximum likelihood, focal and hinge losses.",
          "[Advanced Losses](/learn/m05/advanced-losses) — losses that compare examples with each other (contrastive, triplet, InfoNCE), and losses for segmentation and boxes (Dice, IoU, GIoU).",
          String.raw`[Gradient Descent](/learn/m06/gradient-descent) — the loss's gradient scale and smoothness determine which learning rates work; you saw why MAE needs step decay.`,
          "[Regression Metrics](/learn/m17/regression-metrics) — how to evaluate regressors (RMSE, MAE, R², calibration of intervals) once they are trained.",
          "[Variational Autoencoders](/learn/m14/vae) — the Gaussian NLL reappears as the reconstruction term of the VAE objective.",
        ],
      },
    ],
  },
};

export default lesson;
