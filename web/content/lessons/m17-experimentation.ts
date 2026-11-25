import type { Lesson } from "@/lib/types";

const EXP_NUMPY = String.raw`import numpy as np
from scipy.stats import norm

# =====================================================================================
# Part 1 - grid vs random search when only ONE of two hyperparameters matters
# (the situation analysed by Bergstra & Bengio, 2012)
# =====================================================================================
def validation_error(a, b):
    """a in [0, 1] matters a lot (sharp optimum at 0.73); b in [0, 1] barely matters."""
    return (a - 0.73) ** 2 + 0.01 * (b - 0.5) ** 2


budget = 9
g = np.linspace(0.1, 0.9, 3)                      # 3 x 3 grid -> only 3 distinct values of a
grid_best = min(validation_error(a, b) for a in g for b in g)

rng = np.random.default_rng(0)
random_best = []
for _ in range(1000):                              # repeat random search 1000 times to average out luck
    a, b = rng.random(budget), rng.random(budget)  # 9 points -> 9 distinct values of a
    random_best.append(validation_error(a, b).min())
random_best = np.array(random_best)
print(f"grid search   (9 trials): best error {grid_best:.5f}  distinct values of a tried: 3")
print(f"random search (9 trials): median best error {np.median(random_best):.5f}, "
      f"beats the grid in {np.mean(random_best < grid_best):.1%} of 1000 repetitions")

# =====================================================================================
# Part 2 - Bayesian optimization from scratch: Gaussian-process surrogate + expected improvement
# =====================================================================================
def objective(x):
    """An 'expensive' 1-D function to MINIMIZE on [0, 5] (several local minima)."""
    return np.sin(3 * x) + 0.3 * (x - 2.5) ** 2


xs = np.linspace(0, 5, 2001)                      # candidate points (and a dense grid for the true minimum)
true_min = objective(xs).min()


def rbf(A, B, length):
    return np.exp(-0.5 * (A[:, None] - B[None, :]) ** 2 / length ** 2)


def gp_posterior(X, y, Xq, length, noise=1e-6):
    """GP with unit signal variance on standardized y: posterior mean and std at Xq."""
    K = rbf(X, X, length) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = rbf(X, Xq, length)
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.clip(1.0 - np.sum(v ** 2, axis=0), 1e-12, None)
    log_ml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * np.log(2 * np.pi)
    return mu, np.sqrt(var), log_ml


def expected_improvement(mu, sd, best, xi=0.01):
    """EI for minimization: E[max(best - f - xi, 0)] under f ~ N(mu, sd^2)."""
    imp = best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def bayes_opt(n_iter, seed):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 5, 3)                         # 3 random initial evaluations
    Y = objective(X)
    for _ in range(n_iter - 3):
        m, s = Y.mean(), Y.std()
        y = (Y - m) / s                            # standardize targets
        # choose the length-scale by maximizing the GP marginal likelihood over a small grid
        length = max((0.2, 0.3, 0.5, 0.8, 1.2), key=lambda l: gp_posterior(X, y, X[:1], l)[2])
        mu, sd, _ = gp_posterior(X, y, xs, length)
        ei = expected_improvement(mu, sd, y.min())
        x_next = xs[np.argmax(ei)]                 # the point with the largest expected improvement
        X, Y = np.append(X, x_next), np.append(Y, objective(x_next))
    return X, Y


X, Y = bayes_opt(12, seed=0)
print(f"\ntrue minimum on the grid: f = {true_min:.4f} at x = {xs[np.argmin(objective(xs))]:.3f}")
print("BO evaluation order (x):", np.round(X, 2).tolist())
print(f"BO best after 12 evaluations: f = {Y.min():.4f} at x = {X[np.argmin(Y)]:.3f}")

gaps_bo, gaps_rs = [], []
for seed in range(20):                             # same budget of 12 evaluations, 20 different seeds
    _, Yb = bayes_opt(12, seed)
    Yr = objective(np.random.default_rng(100 + seed).uniform(0, 5, 12))
    gaps_bo.append(Yb.min() - true_min)
    gaps_rs.append(Yr.min() - true_min)
print(f"over 20 seeds, mean gap to the true minimum: BO {np.mean(gaps_bo):.4f}   random search {np.mean(gaps_rs):.4f}")`;

const EXP_NUMPY_OUT = String.raw`grid search   (9 trials): best error 0.02890  distinct values of a tried: 3
random search (9 trials): median best error 0.00248, beats the grid in 97.0% of 1000 repetitions

true minimum on the grid: f = -0.7572 at x = 1.630
BO evaluation order (x): [3.18, 1.35, 0.2, 1.54, 1.97, 4.18, 5.0, 3.62, 1.64, 2.67, 1.64, 1.64]
BO best after 12 evaluations: f = -0.7569 at x = 1.637
over 20 seeds, mean gap to the true minimum: BO 0.0003   random search 0.1019`;

const EXP_TORCH = String.raw`import json
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer


# ---------------- reproducibility: one function seeds every RNG we use ----------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                        # also seeds CUDA generators if a GPU is present
    torch.use_deterministic_algorithms(True)       # raise an error if a non-deterministic kernel is used


X, y = load_breast_cancer(return_X_y=True)
perm = np.random.default_rng(0).permutation(len(X))
tr, va = perm[:400], perm[400:]
mu, sd = X[tr].mean(0), X[tr].std(0)
X = torch.tensor((X - mu) / sd, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def run(cfg, epochs, seed=0):
    """Train a small MLP with mini-batches; return the validation log-loss after the given number of epochs."""
    seed_everything(seed)
    net = nn.Sequential(nn.Linear(30, cfg["width"]), nn.ReLU(), nn.Linear(cfg["width"], 1))
    opt = torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for idx in torch.randperm(len(tr)).split(32):          # shuffling uses the seeded torch RNG
            b = torch.as_tensor(tr)[idx]
            opt.zero_grad()
            loss_fn(net(X[b]), y[b]).backward()
            opt.step()
    with torch.no_grad():
        return loss_fn(net(X[va]), y[va]).item()


base = {"lr": 0.01, "wd": 1e-4, "width": 16}
a, b, c = run(base, 5, seed=0), run(base, 5, seed=0), run(base, 5, seed=1)
print(f"same seed twice: {a!r} vs {b!r} -> identical: {a == b}")
print(f"different seed:  {c!r}")

# ---------------- random search with log-uniform sampling + successive halving ----------------
rng = np.random.default_rng(42)
configs = [{"lr": float(10 ** rng.uniform(-3.5, -0.5)),          # log-uniform in [3e-4, 0.3]
            "wd": float(10 ** rng.uniform(-6, -2)),
            "width": int(rng.choice([8, 16, 64]))} for _ in range(16)]

log_path = os.path.join(tempfile.mkdtemp(), "trials.jsonl")       # a minimal experiment log
budget, survivors, rung = 2, list(range(16)), 0
with open(log_path, "w") as log:
    # first line: the environment, so the run can be reproduced later (versions, seed policy)
    log.write(json.dumps({"torch": torch.__version__, "numpy": np.__version__, "search_seed": 42}) + "\n")
    while True:
        scores = {i: run(configs[i], budget) for i in survivors}
        for i in survivors:
            log.write(json.dumps({"trial": i, "rung": rung, "epochs": budget, **configs[i], "val_loss": scores[i]}) + "\n")
        print(f"rung {rung}: {len(survivors):2d} configs x {budget:2d} epochs, best val loss {min(scores.values()):.4f}")
        if len(survivors) == 1:
            break
        survivors = sorted(survivors, key=scores.get)[: len(survivors) // 2]   # keep the best half
        budget, rung = budget * 2, rung + 1

best = survivors[0]
print(f"winner: trial {best} lr={configs[best]['lr']:.4f} wd={configs[best]['wd']:.2e} width={configs[best]['width']}")
rows = [json.loads(line) for line in open(log_path)][1:]   # skip the environment line
print(f"logged {len(rows)} evaluations; total epochs spent {sum(r['epochs'] for r in rows)} "
      f"(training all 16 configs for {budget} epochs would cost {16 * budget})")`;

const EXP_TORCH_OUT = String.raw`same seed twice: 0.17353136837482452 vs 0.17353136837482452 -> identical: True
different seed:  0.1330755650997162
rung 0: 16 configs x  2 epochs, best val loss 0.0930
rung 1:  8 configs x  4 epochs, best val loss 0.0819
rung 2:  4 configs x  8 epochs, best val loss 0.0773
rung 3:  2 configs x 16 epochs, best val loss 0.0840
rung 4:  1 configs x 32 epochs, best val loss 0.0828
winner: trial 12 lr=0.0542 wd=7.41e-03 width=16
logged 31 evaluations; total epochs spent 160 (training all 16 configs for 32 epochs would cost 512)`;

const EXP_KERAS = String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

tf.keras.utils.set_random_seed(0)                 # seeds Python, NumPy and TensorFlow together
tf.config.experimental.enable_op_determinism()    # deterministic kernels (slower on GPU)

X, y = load_breast_cancer(return_X_y=True)
X = X.astype("float32")
perm = np.random.default_rng(0).permutation(len(X))
tr, va = perm[:400], perm[400:]

def build(lr, wd):
    reg = tf.keras.regularizers.L2(wd)
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(30,)),
        tf.keras.layers.Normalization(),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.layers[0].adapt(X[tr])
    m.compile(tf.keras.optimizers.SGD(lr, momentum=0.9), "binary_crossentropy")
    return m

rng = np.random.default_rng(42)
results = []
for trial in range(8):                            # random search, log-uniform sampling
    lr, wd = 10 ** rng.uniform(-3.5, -0.5), 10 ** rng.uniform(-6, -2)
    stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    h = build(lr, wd).fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=100,
                          batch_size=32, callbacks=[stop], verbose=0)
    results.append((min(h.history["val_loss"]), lr, wd))
    print(f"trial {trial}: lr={lr:.4f} wd={wd:.1e} best val_loss={results[-1][0]:.4f}")
print("best:", min(results))`;

const lesson: Lesson = {
  id: "m17/experimentation",
  title: "Hyperparameter Tuning, Tracking and Reproducibility",
  summary: "How to search hyperparameters efficiently (grid vs random search, Bayesian optimization with a Gaussian-process surrogate and expected improvement, early stopping and successive halving), and how to run experiments you can trust: tracking, seeds and determinism, ablation studies and error analysis.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m17/data-splits", "m06/lr-schedules", "m07/regularization"],
  next: ["m18/mlops", "m19/reading-papers", "m18/serving"],
  terms: ["hyperparameter", "grid-search", "random-search", "bayesian-optimization", "gaussian-process", "expected-improvement", "successive-halving", "experiment-tracking", "reproducibility", "ablation-study", "error-analysis"],
  sections: {
    intro: [
      { t: "p", text: "A **hyperparameter** is a setting chosen before training rather than learned by it: learning rate, batch size, weight decay, dropout rate, width and depth, number of epochs, augmentation strength. Training finds the weights *given* the hyperparameters; **hyperparameter tuning** searches over the hyperparameters themselves, using the validation score as the objective. Each evaluation of that objective is a full (or partial) training run, so the search must be economical." },
      { t: "p", text: "The classical approach, **grid search**, tries every combination of a few values per hyperparameter. Bergstra and Bengio (2012, *Random Search for Hyper-Parameter Optimization*, JMLR) showed that **random search** is usually more efficient, because typically only a few hyperparameters matter and a grid wastes most trials re-testing the same values of the important ones. **Bayesian optimization** goes further: it fits a probabilistic model of the score as a function of the hyperparameters and chooses each next trial where improvement is most likely (Snoek, Larochelle and Adams, 2012, *Practical Bayesian Optimization of Machine Learning Algorithms*). **Early-stopping-based** methods such as successive halving and Hyperband save compute by abandoning bad configurations early." },
      { t: "p", text: "Tuning is only half of experimentation. Results are useless if you cannot say which code, data and settings produced them, or if rerunning gives a different answer. **Experiment tracking**, **reproducibility** practices (seeds, deterministic kernels, pinned versions), **ablation studies** (removing one component at a time to measure its contribution) and **error analysis** (inspecting what the model gets wrong) turn trial-and-error into an engineering process." },
      { t: "callout", kind: "key", text: "Search on a log scale with random or Bayesian search, stop bad runs early, tune on validation data only, log everything that could change the result, and verify improvements across several seeds before believing them." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Train/validation/test splits](/learn/m17/data-splits) — tuning optimizes the validation score; the test set (or nested CV) must stay outside the search.",
        "[Learning-rate schedules](/learn/m06/lr-schedules) and [Gradient descent](/learn/m06/gradient-descent) — the learning rate is the single most important hyperparameter, and its effect spans orders of magnitude.",
        "[Regularization](/learn/m07/regularization) — weight decay and dropout are typical tuned hyperparameters; early stopping is both a regularizer and a tuning tool.",
        "[Probability](/learn/m02/probability) — Gaussian distributions, their CDF $\\Phi$ and density $\\varphi$; Bayesian optimization's acquisition function is an expectation under a Gaussian.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Prospecting for oil", text: "You can afford a few dozen expensive test wells. Drilling on a rigid grid wastes wells if the oil lies along a narrow band the grid keeps missing. Drilling at random spots covers more distinct positions along every direction. A geologist does better still: after each well, she updates a map of where oil probably is **and how uncertain that map is**, then drills either where the map predicts the most oil (**exploitation**) or where it is most uncertain and might hide a big field (**exploration**). That is Bayesian optimization. Abandoning a well as soon as the first metres look dry is successive halving." },
      { t: "p", text: "Why does random search beat a grid? With a 3×3 grid over (learning rate, dropout), you try only **3 distinct learning rates**. If dropout barely matters, six of your nine runs merely repeat learning rates already tested. Nine random points give **nine distinct learning rates**. Since you rarely know in advance which hyperparameters matter, random search hedges automatically." },
      { t: "p", text: "Reproducibility is about removing hidden variables. A training run depends on the code version, the data version, every hyperparameter, the random seeds (initialization, shuffling, dropout, augmentation), library versions and even the hardware's floating-point kernels. If any of these is unrecorded, a \"0.5% improvement\" may just be a lucky seed — differences between seeds are often as large as the improvements people report." },
    ],
    mechanism: [
      { t: "h", text: "A tuning workflow" },
      { t: "steps", items: [
        { title: "Fix the protocol first", text: "Data split, metric, training budget, and what counts as one trial. Keep the test set out of the loop." },
        { title: "Choose the search space", text: "Ranges and **scales**: learning rate and weight decay log-uniform (e.g. $10^{-5}$ to $10^{-1}$); dropout uniform in $[0, 0.5]$; width from a discrete set. Start wide; narrow later." },
        { title: "Pick a search strategy", text: "Random search as a strong default; Bayesian optimization when trials are expensive and the space is low-dimensional (≲ 20); successive halving / Hyperband when partial training is informative; combine them (e.g. BOHB)." },
        { title: "Run trials, log everything", text: "Configuration, seed, code commit, data version, metrics per epoch, runtime, hardware." },
        { title: "Inspect, then refine", text: "Plot score vs each hyperparameter. If the best values sit on a boundary of the range, extend it. Narrow around good regions." },
        { title: "Confirm", text: "Re-run the top few configurations with several seeds; pick by mean (and spread), not by the single luckiest run. Only then evaluate on the test set." },
      ] },
      { t: "h", text: "Bayesian optimization loop" },
      { t: "steps", items: [
        { title: "Initial design", text: "Evaluate a few random configurations." },
        { title: "Fit a surrogate", text: "A **Gaussian process** (GP) gives, at every untested configuration $x$, a predicted mean $\\mu(x)$ and uncertainty $\\sigma(x)$ of the score — small near tested points, large far away." },
        { title: "Maximize an acquisition function", text: "**Expected improvement** (EI) scores each candidate by how much it is expected to beat the best result so far, combining low $\\mu$ (exploitation) and high $\\sigma$ (exploration). Maximizing EI is cheap: it only queries the surrogate." },
        { title: "Evaluate and repeat", text: "Train with the chosen configuration, add the result to the data, refit the surrogate." },
      ] },
      { t: "h", text: "Successive halving" },
      { t: "steps", items: [
        { title: "Start many configurations on a small budget", text: "E.g. 16 configurations × 2 epochs." },
        { title: "Keep the best fraction", text: "Discard the worse half (or keep the top $1/\\eta$, typically $\\eta = 3$)." },
        { title: "Multiply the budget", text: "Survivors continue with twice (or $\\eta$ times) as many epochs." },
        { title: "Repeat until one remains", text: "Total cost grows like (number of rungs) × (configs × initial budget) instead of configs × full budget. Hyperband (Li et al., 2018, JMLR) runs several halving brackets with different starting budgets to hedge against misleading early results." },
      ] },
      { t: "h", text: "Experiment tracking: what to log" },
      { t: "list", items: [
        "**Identity**: run id, timestamp, code commit hash (and uncommitted diff), command line.",
        "**Inputs**: all hyperparameters (the full resolved config, not just the ones you changed), data version/hash, train/val/test split definition, random seeds.",
        "**Environment**: library versions (Python, PyTorch/TF, CUDA/cuDNN), hardware (GPU type and count).",
        "**Outputs**: metrics per step/epoch (train and validation), final metrics, runtime, checkpoints or their paths, sample predictions, plots.",
        "**Notes**: why the run exists (the hypothesis) and what you concluded.",
      ] },
      { t: "p", text: "Tools such as **MLflow** (open source; logs parameters, metrics and artifacts to a local or server-backed store and has a model registry) and **Weights & Biases** (a hosted service with dashboards, sweeps and artifact versioning) automate this; TensorBoard covers metric curves. They are not installed in this course's environment, so the code below writes a plain JSON-lines log, which captures the same idea: one structured record per evaluation. See [MLOps](/learn/m18/mlops) for production use." },
      { t: "h", text: "Reproducibility checklist" },
      { t: "list", items: [
        "Seed every RNG: Python `random`, NumPy, PyTorch (CPU and CUDA), and data-loader workers (`worker_init_fn`, a seeded `torch.Generator`).",
        "Request deterministic kernels: `torch.use_deterministic_algorithms(True)` (some CUDA ops additionally need the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`), and `torch.backends.cudnn.benchmark = False`. In TensorFlow: `tf.keras.utils.set_random_seed` and `tf.config.experimental.enable_op_determinism()`.",
        "Pin versions (`requirements.txt` / lock file, container image). Bit-identical results are generally only guaranteed on the same hardware and software versions.",
        "Version the data and the split, not just the code.",
        "Even with everything fixed, report variability across **several seeds** — determinism makes a run repeatable, not representative.",
      ] },
      { t: "h", text: "Ablations and error analysis" },
      { t: "list", items: [
        "**Ablation study**: starting from the full system, remove or replace one component at a time (an augmentation, a loss term, a layer type) keeping everything else — including the tuning budget — fixed, and measure the change over several seeds. It attributes the gain to components and catches components that do nothing.",
        "**Error analysis**: collect validation errors, inspect a sample by hand, tag them by cause (label noise, ambiguous input, rare subgroup, preprocessing bug), and count. The largest bucket tells you whether to collect data, clean labels, change the model or fix the pipeline. Also compute metrics per slice (class, data source, input length).",
      ] },
    ],
    math: [
      { t: "h", text: "Tuning as optimization" },
      { t: "p", text: "Let $\\lambda \\in \\Lambda$ be a hyperparameter configuration, $w^*(\\lambda)$ the weights training produces with it, and $\\mathcal{L}_{\\text{val}}$ the validation loss. Tuning solves the bilevel problem" },
      { t: "math", tex: String.raw`\lambda^* = \arg\min_{\lambda\in\Lambda} \mathcal{L}_{\text{val}}\big(w^*(\lambda)\big), \qquad w^*(\lambda) = \arg\min_{w} \mathcal{L}_{\text{train}}(w;\lambda)` },
      { t: "p", text: "The outer function $f(\\lambda) = \\mathcal{L}_{\\text{val}}(w^*(\\lambda))$ is expensive (one training run per evaluation), noisy (seeds), usually non-differentiable with respect to $\\lambda$, and may mix continuous and discrete variables — so we treat it as a **black box**." },
      { t: "h", text: "Why random search wins: a counting argument" },
      { t: "p", text: "With $n$ trials and $d$ hyperparameters, a full grid has $n^{1/d}$ distinct values per hyperparameter; random search has $n$ distinct values of each. If only one hyperparameter matters, the grid explores it at $n^{1/d}$ points and random search at $n$. Independently of dimension, if a fraction $q$ of the space is \"good\", the probability that at least one of $n$ random trials lands in it is" },
      { t: "math", tex: String.raw`P(\text{hit}) = 1 - (1-q)^n, \qquad q = 0.05:\ \ n = 20 \Rightarrow 0.642,\quad n = 60 \Rightarrow 0.954` },
      { t: "h", text: "Log-uniform sampling" },
      { t: "p", text: "Sample $u \\sim \\mathcal{U}(\\log_{10} a, \\log_{10} b)$ and set $\\lambda = 10^u$. Each decade then receives equal probability: for a learning rate in $[10^{-5}, 10^{-1}]$, uniform sampling would put 90% of trials in $[10^{-2}, 10^{-1}]$, while log-uniform puts 25% in each decade." },
      { t: "h", text: "Gaussian-process surrogate" },
      { t: "p", text: "A GP prior says that the scores at any finite set of configurations are jointly Gaussian, with covariance given by a **kernel**. The RBF kernel with length-scale $\\ell$ and signal variance $\\sigma_f^2$:" },
      { t: "math", tex: String.raw`k(x, x') = \sigma_f^2 \exp\!\Big(-\frac{(x - x')^2}{2\ell^2}\Big)` },
      { t: "p", text: "Given observations $y = (y_1..y_t)$ at $X = (x_1..x_t)$ with noise variance $\\sigma_n^2$, let $K_{ij} = k(x_i, x_j)$ and $k_* = (k(x_1, x_*), \\dots, k(x_t, x_*))^\\top$. The posterior at a new point $x_*$ is Gaussian with" },
      { t: "math", tex: String.raw`\mu(x_*) = k_*^\top (K + \sigma_n^2 I)^{-1} y, \qquad \sigma^2(x_*) = k(x_*, x_*) - k_*^\top (K + \sigma_n^2 I)^{-1} k_*` },
      { t: "p", text: "The length-scale is chosen by maximizing the log marginal likelihood $\\log p(y\\mid X) = -\\tfrac12 y^\\top(K+\\sigma_n^2I)^{-1}y - \\tfrac12\\log|K+\\sigma_n^2I| - \\tfrac t2\\log 2\\pi$, which trades data fit against complexity. The code computes all of this with one Cholesky factorization $K + \\sigma_n^2 I = LL^\\top$." },
      { t: "h", text: "Expected improvement" },
      { t: "p", text: "For minimization with best observed value $f^+$, the improvement at $x$ is $I(x) = \\max(f^+ - f(x), 0)$. Under the GP posterior $f(x) \\sim \\mathcal{N}(\\mu, \\sigma^2)$, its expectation has a closed form (Jones, Schonlau and Welch, 1998, *Efficient Global Optimization of Expensive Black-Box Functions*):" },
      { t: "math", tex: String.raw`\text{EI}(x) = (f^+ - \mu)\,\Phi(z) + \sigma\,\varphi(z), \qquad z = \frac{f^+ - \mu}{\sigma}` },
      { t: "p", text: "Here $\\Phi$ and $\\varphi$ are the standard normal CDF and density. The first term rewards a low predicted mean (exploitation), the second a large uncertainty (exploration). A small margin $\\xi$ (the code uses 0.01) is often subtracted from $f^+ - \\mu$ to encourage exploration." },
      { t: "h", text: "Worked example — comparing two candidates by EI" },
      { t: "p", text: "Best value so far $f^+ = 0$. Candidate A: $\\mu = 0.2$, $\\sigma = 0.5$ (predicted worse, but very uncertain). Candidate B: $\\mu = -0.1$, $\\sigma = 0.1$ (predicted better, confident)." },
      { t: "steps", items: [
        { title: "A: standardized improvement", tex: String.raw`z_A = \frac{0 - 0.2}{0.5} = -0.4,\quad \Phi(-0.4) = 0.3446,\quad \varphi(-0.4) = 0.3683` },
        { title: "A: EI", tex: String.raw`\text{EI}_A = (-0.2)(0.3446) + 0.5(0.3683) = -0.0689 + 0.1841 = 0.1152` },
        { title: "B: standardized improvement", tex: String.raw`z_B = \frac{0 - (-0.1)}{0.1} = 1.0,\quad \Phi(1) = 0.8413,\quad \varphi(1) = 0.2420` },
        { title: "B: EI", tex: String.raw`\text{EI}_B = (0.1)(0.8413) + 0.1(0.2420) = 0.0841 + 0.0242 = 0.1083` },
      ] },
      { t: "p", text: "EI prefers A (0.1152 > 0.1083) even though its predicted mean is worse: its large uncertainty gives a real chance of a big improvement. Halve A's uncertainty and B would win — that is the exploration–exploitation trade-off made quantitative." },
      { t: "h", text: "Cost of successive halving" },
      { t: "p", text: "With $n$ configurations, initial budget $r$ and halving ($\\eta = 2$), every rung costs $n r$ (half as many configs, twice the budget) and there are $\\log_2 n + 1$ rungs:" },
      { t: "math", tex: String.raw`\text{cost} = (\log_2 n + 1)\, n\, r = (4 + 1)\times 16 \times 2 = 160 \text{ epochs} \quad \text{vs}\quad n \times r_{\max} = 16 \times 32 = 512` },
      { t: "h", text: "Seed variance and the winner's curse" },
      { t: "p", text: "If each trial's score is the true score plus independent noise with standard deviation $s$ (from seeds), the best of many configurations is biased optimistically, and a difference $\\Delta$ between two configurations measured with $k$ seeds each has standard error $s\\sqrt{2/k}$. With $s = 0.3$ points of accuracy and $k = 1$, a 0.5-point difference is barely more than one standard error ($0.3\\sqrt2 = 0.42$) — not convincing; with $k = 5$ the SE is 0.19." },
    ],
    visual: [
      { t: "p", text: "The learning rate is the hyperparameter that matters most, and its effect is easiest to see in motion. Use the simulators below as a *manual* hyperparameter search." },
      { t: "sim", id: "lr-schedule", caption: "A real training run under different learning-rate schedules — each setting is one trial of a search." },
      { t: "list", items: [
        "Try the peak learning rate at values spread over orders of magnitude (e.g. 0.001, 0.01, 0.1, 1). Notice that the interesting changes happen per *factor of 10*, which is why we sample learning rates log-uniformly.",
        "Compare schedules at the same peak learning rate: the schedule is itself a hyperparameter whose effect interacts with the peak value — one reason one-at-a-time tuning can mislead.",
        "Watch the early part of each loss curve: configurations that are clearly bad after a few steps rarely recover. That observation is what successive halving exploits.",
      ] },
      { t: "sim", id: "gradient-descent", caption: "Gradient descent in 1-D: one hyperparameter, four regimes." },
      { t: "list", items: [
        "Sweep the learning rate slowly: convergence, crawling, oscillation, divergence. The validation score as a function of a hyperparameter is often this kind of sharply peaked, asymmetric curve — a grid with coarse spacing can step right over the good region.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Part 1 reproduces the grid-vs-random argument on a function where only one of two hyperparameters matters. Part 2 implements Bayesian optimization from scratch — a Gaussian-process posterior via Cholesky, length-scale selection by marginal likelihood, and expected improvement — and compares it with random search at the same budget." },
      { t: "code", lang: "python", title: "search_from_scratch.py", code: EXP_NUMPY, output: EXP_NUMPY_OUT },
      { t: "list", items: [
        "The 3×3 grid tests $a \\in \\{0.1, 0.5, 0.9\\}$; the closest to the optimum 0.73 is 0.9, error $0.17^2 = 0.0289$. Random search with the same 9 trials tries 9 distinct values of $a$ and beats the grid in 97% of repetitions.",
        "Bayesian optimization spends its first points exploring (3.18, 1.35, 0.2, …, 5.0), then concentrates around the global minimum near $x = 1.63$, reaching $f = -0.7569$ against the true $-0.7572$ in 12 evaluations. The repeated proposals at 1.64 show EI exploiting once uncertainty elsewhere is low — a practical implementation would stop or add exploration there.",
        "Across 20 seeds with 12 evaluations each, BO's average gap to the true minimum is 0.0003 versus 0.1019 for random search. On a cheap 1-D function this is easy; BO's advantage shrinks in high dimensions and with noisy objectives, where its surrogate is harder to fit.",
        "Standardizing $y$ before fitting matches the GP's unit signal variance; `noise=1e-6` is a tiny jitter that keeps the Cholesky factorization numerically stable for a noise-free objective.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Reproducibility and an efficient search in one program: a `seed_everything` helper with deterministic algorithms, a check that the same seed reproduces the validation loss bit-for-bit, then random search over 16 log-uniformly sampled configurations pruned by successive halving, logged as JSON lines (a stand-in for MLflow / W&B)." },
          { t: "code", lang: "python", title: "tuning_pytorch.py", code: EXP_TORCH, output: EXP_TORCH_OUT },
          { t: "list", items: [
            "Two runs with seed 0 give the identical float `0.17353136837482452`; seed 1 gives 0.133. The *size* of that seed-to-seed difference after 5 epochs is a reminder to compare configurations over several seeds.",
            "Successive halving spent 160 epochs instead of 512 for training all 16 configurations to 32 epochs. Note the best validation loss is not monotone across rungs (0.0773 at rung 2, 0.0828 at the end): with 169 validation examples, losses are noisy, and longer training can overfit — the final choice should be confirmed with more seeds or CV.",
            "The JSON-lines log has one environment record (library versions, search seed) followed by one record per evaluation: enough to rebuild any table or plot later and to rerun any trial exactly.",
          ] },
          { t: "callout", kind: "note", title: "Libraries", text: "In practice you would use a tuning library rather than hand-written loops: **Optuna** (define-by-run search spaces, TPE sampler, pruners including successive halving/Hyperband), **Ray Tune**, **scikit-learn**'s `GridSearchCV`, `RandomizedSearchCV` and `HalvingRandomSearchCV`, or W&B Sweeps. None of them is needed to understand the algorithms above." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same ideas in Keras: one call seeds Python, NumPy and TensorFlow; `enable_op_determinism` requests deterministic kernels; random search with log-uniform sampling; `EarlyStopping` ends each trial when validation loss stops improving." },
          { t: "code", lang: "python", title: "tuning_keras.py (not executed in CI)", code: EXP_KERAS, output: "Prints one line per trial (sampled lr, weight decay and best validation loss) and the best (val_loss, lr, wd) tuple; rerunning gives identical numbers on the same machine and TensorFlow version." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Method", "Advantages", "Disadvantages", "Use when"], rows: [
        ["Manual tuning", "Uses expert intuition; cheap for a few knobs", "Not reproducible; biased; does not scale", "Early exploration, sanity ranges"],
        ["Grid search", "Simple, exhaustive, parallel, reproducible", "Exponential in dimensions; wastes trials on unimportant knobs", "1–2 hyperparameters, cheap models"],
        ["Random search", "Embarrassingly parallel; robust to unimportant dimensions; any budget", "No learning from past trials", "Strong default, any dimension"],
        ["Bayesian optimization (GP + EI)", "Sample-efficient; models uncertainty", "Sequential; GP scales as $O(t^3)$; struggles in high dimensions and with categorical variables", "Expensive trials, ≲ 20 continuous hyperparameters"],
        ["TPE (tree-structured Parzen estimator)", "Handles conditional/categorical spaces; cheap", "Less principled uncertainty", "Mixed search spaces (Optuna, Hyperopt)"],
        ["Successive halving / Hyperband", "Large compute savings; simple", "Early performance may mislead (e.g. small learning rates start slow)", "Many configurations, partial training informative"],
        ["BOHB / BO with pruning", "Combines model-based proposals with early stopping", "More moving parts", "Large-scale tuning"],
        ["Population-based training", "Adapts hyperparameters during training; finds schedules", "Needs many parallel workers; less reproducible", "Large compute clusters, RL, schedules"],
        ["Gradient-based HPO", "Uses hypergradients through training", "Memory-heavy; only continuous hyperparameters", "Research, many continuous hyperparameters"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Model development everywhere**: random search plus early stopping is the everyday workhorse for tuning learning rates, regularization and architecture sizes.",
        "**AutoML systems** combine Bayesian optimization, bandit-based early stopping and meta-learning to tune whole pipelines; Google Vizier is a well-known internal black-box optimization service described in a 2017 KDD paper by Golovin et al.",
        "**Large-model training** tunes on small proxy models and transfers the settings, because each full run is too expensive to repeat (see [Scaling Laws](/learn/m19/scaling-efficiency)).",
        "**Research papers** rely on ablation tables to justify each component and on multiple seeds with error bars to support claims; reproducing them is the topic of [Reading and Reproducing Papers](/learn/m19/reading-papers).",
        "**Regulated industries** (finance, healthcare) require reproducible, auditable training runs — tracked data versions, code and parameters — as part of model governance.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Sample efficiency vs parallelism**: BO uses each result to choose the next trial but is inherently sequential; random search parallelizes perfectly.",
        "**Early stopping saves compute but can prune slow starters** (small learning rates, heavy regularization) that would have won with a full budget.",
        "**Larger search spaces** find better optima in principle but need more trials and increase the optimistic bias of the best validation score.",
        "**Full determinism costs speed** (deterministic GPU kernels can be slower) and is tied to hardware/software versions; seeding alone gives repeatability on one setup, not across setups.",
        "**Tracking has overhead** (setup, storage) but repays it the first time you need to explain or reproduce a result.",
        "**Ablations multiply compute** (each removed component × several seeds) but are the only reliable way to know what matters.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Tuning on the test set** (or picking the final model by test score). The reported number becomes optimistic. Tune on validation; test once.",
        "**Sampling learning rate or weight decay uniformly.** Almost all trials land in the top decade. Sample log-uniformly.",
        "**Declaring victory from one seed.** Seed-to-seed variation can exceed the improvement. Run the top configurations with 3–5 seeds and compare means with their spread.",
        "**Best value on the edge of the range.** The optimum may lie outside; extend the range and search again.",
        "**One-at-a-time tuning.** Hyperparameters interact (learning rate with batch size, weight decay with learning rate); tune them jointly.",
        "**Unfair comparisons.** Tuning your method extensively and the baseline barely (or not at all) inflates the apparent gain. Give every method the same tuning budget.",
        "**Seeding only PyTorch.** NumPy/Python RNGs (augmentation, sampling) and data-loader workers also need seeds; non-deterministic GPU kernels can still differ.",
        "**Not logging the configuration and code version.** A great result you cannot reproduce is not a result.",
        "**Ablating without retuning concerns.** Removing a component can change the optimal learning rate; at minimum, keep the tuning budget equal across ablation variants and state what was retuned.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between a parameter and a hyperparameter?", a: "Parameters (weights, biases) are learned by the training algorithm from data. Hyperparameters (learning rate, batch size, weight decay, architecture size, epochs) are set before training and control how learning happens; they are chosen by search using validation performance." },
      { t: "qa", level: "beginner", q: "Why is random search often better than grid search?", a: "Usually only a few hyperparameters matter. A grid with $n$ trials over $d$ dimensions tests only $n^{1/d}$ distinct values of each, so most trials repeat values of the important ones. Random search tests $n$ distinct values of every hyperparameter, and it can stop at any budget (Bergstra and Bengio, 2012)." },
      { t: "qa", level: "beginner", q: "How do you make a PyTorch training run reproducible?", a: "Seed Python's `random`, NumPy and PyTorch; seed data-loader workers and use a seeded generator for shuffling; enable `torch.use_deterministic_algorithms(True)` and disable cuDNN benchmarking; pin library versions and record the code commit, data version and full config. Expect bit-identical results only on the same hardware and software stack." },
      { t: "qa", level: "intermediate", q: "Explain Bayesian optimization in a few sentences.", a: "Treat validation loss as an expensive black-box function of the hyperparameters. Fit a probabilistic surrogate (typically a Gaussian process) to the trials so far, giving a mean and uncertainty everywhere. Choose the next trial by maximizing an acquisition function such as expected improvement, which balances low predicted loss against high uncertainty. Evaluate, update the surrogate, repeat." },
      { t: "qa", level: "intermediate", q: "What is successive halving and what is its main risk?", a: "Start many configurations with a small budget, keep the best fraction (e.g. half), give survivors more budget, repeat. It spends most compute on promising configurations. Risk: early performance may not predict final performance — e.g. a small learning rate or strong regularization starts slowly but ends best — so good configurations can be pruned. Hyperband mitigates this by running brackets with different starting budgets." },
      { t: "qa", level: "intermediate", q: "What is an ablation study and how do you run a good one?", a: "Removing or replacing one component at a time from the full system to measure its contribution. Keep everything else fixed (data, budget, tuning effort), run several seeds per variant, report mean ± spread, and include the full model and a baseline. It shows which parts actually matter and catches components whose apparent benefit was noise." },
      { t: "qa", level: "advanced", q: "Derive the expected improvement formula.", a: "With $f(x) \\sim \\mathcal{N}(\\mu, \\sigma^2)$ and $I = \\max(f^+ - f, 0)$, write $f = \\mu + \\sigma u$ with $u \\sim \\mathcal{N}(0,1)$. Improvement is positive when $u < z = (f^+ - \\mu)/\\sigma$. Then $\\mathbb{E}[I] = \\int_{-\\infty}^{z} (f^+ - \\mu - \\sigma u)\\varphi(u)\\,du = (f^+ - \\mu)\\Phi(z) - \\sigma\\int_{-\\infty}^{z} u\\varphi(u)\\,du = (f^+ - \\mu)\\Phi(z) + \\sigma\\varphi(z)$, using $\\int_{-\\infty}^{z} u\\varphi(u)du = -\\varphi(z)$." },
      { t: "qa", level: "advanced", q: "Why is the best validation score from a large search biased, and how do you report performance honestly?", a: "Each configuration's validation score is its true performance plus noise (finite validation set, seeds). Selecting the maximum also selects favourable noise, so the winner's validation score overestimates its true performance — more so with more configurations and smaller validation sets. Report performance on an untouched test set, or with nested CV, and confirm the winner with additional seeds." },
      { t: "qa", level: "advanced", q: "When does GP-based Bayesian optimization struggle, and what are alternatives?", a: "High-dimensional spaces (the surrogate needs many points to be informative), many categorical or conditional hyperparameters (kernels are awkward), large numbers of trials (exact GP inference is $O(t^3)$), very noisy objectives, and massively parallel settings (BO is sequential). Alternatives: TPE (Optuna, Hyperopt), random forests as surrogates (SMAC), random search with Hyperband, BOHB, batch acquisition functions for parallelism, or population-based training." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A grid search uses 4 values for each of 3 hyperparameters. How many training runs is that?", answer: 64, tolerance: 0, explain: "$4^3 = 64$ — grids grow exponentially with the number of hyperparameters." },
        { kind: "numeric", q: "With the same 64 runs, how many distinct values of the first hyperparameter does random search try (with continuous sampling)?", answer: 64, tolerance: 0, explain: "Every random trial has its own value of every hyperparameter (with probability 1 for continuous distributions); the grid tries only 4." },
        { kind: "numeric", q: "If 5% of the search space is \"good\", what is the probability that at least one of 30 random trials lands in it? (3 decimals)", answer: 0.785, tolerance: 0.001, explain: "$1 - 0.95^{30} = 1 - 0.2146 = 0.785$." },
        { kind: "numeric", q: "Expected improvement for minimization with $f^+ = 1.0$, $\\mu = 1.0$, $\\sigma = 0.4$? (3 decimals; $\\varphi(0) = 0.3989$)", answer: 0.16, tolerance: 0.001, explain: "$z = 0$: EI $= 0\\cdot\\Phi(0) + 0.4\\,\\varphi(0) = 0.4\\times0.3989 = 0.160$ — pure exploration value." },
        { kind: "numeric", q: "Successive halving with 27 configurations, η = 3 (keep a third each rung), initial budget 1 epoch: how many total epochs until one configuration remains (rungs: 27×1, 9×3, 3×9, 1×27)?", answer: 108, tolerance: 0, explain: "Each rung costs 27 epochs and there are 4 rungs: $4\\times27 = 108$, versus $27\\times27 = 729$ to train all to 27 epochs." },
        { q: "A learning rate should be searched in [1e-5, 1e-1]. Which sampling is appropriate?", options: ["Uniform on [1e-5, 1e-1]", "Log-uniform: 10^u with u uniform in [-5, -1]", "A grid with spacing 0.01", "Always 3e-4"], answer: 1, explain: "Effects scale multiplicatively; log-uniform gives each decade equal probability. Uniform sampling puts 90% of trials above 1e-2." },
        { q: "Which practice does NOT improve reproducibility?", options: ["Logging the code commit hash", "Seeding NumPy, Python and PyTorch RNGs", "Selecting the best of 10 seeds and reporting only it", "Pinning library versions"], answer: 2, explain: "Cherry-picking the best seed makes the result optimistic and unrepresentative; report mean and spread across seeds." },
      ] },
      { t: "exercise", title: "Expected improvement by hand", prompt: "Best observed validation loss $f^+ = 0.30$. The GP predicts candidate C: $\\mu = 0.32$, $\\sigma = 0.05$; candidate D: $\\mu = 0.40$, $\\sigma = 0.20$. Compute EI (no margin) for both and say which BO would evaluate next. Use $\\Phi(-0.4) = 0.3446$, $\\varphi(-0.4) = 0.3683$, $\\Phi(-0.5) = 0.3085$, $\\varphi(-0.5) = 0.3521$.", hint: "Compute $z = (f^+ - \\mu)/\\sigma$ first.", solution: [
        { t: "steps", items: [
          { title: "C", tex: String.raw`z_C = \frac{0.30 - 0.32}{0.05} = -0.4,\quad \text{EI}_C = (-0.02)(0.3446) + 0.05(0.3683) = -0.00689 + 0.01842 = 0.01152` },
          { title: "D", tex: String.raw`z_D = \frac{0.30 - 0.40}{0.20} = -0.5,\quad \text{EI}_D = (-0.10)(0.3085) + 0.20(0.3521) = -0.03085 + 0.07042 = 0.03957` },
        ] },
        { t: "p", text: "BO evaluates D next: its mean is worse, but its uncertainty is four times larger, so the chance of a substantial improvement is higher. (Notice that C is exactly example A from the Math section scaled by 0.1, so its EI is 0.1 × 0.1152.)" },
      ] },
      { t: "exercise", title: "Add Hyperband-style brackets and multiple seeds", prompt: "Modify `tuning_pytorch.py`: (1) make `run` accept a list of seeds and return the mean validation loss over them; (2) run two successive-halving brackets — one starting with 16 configs at 2 epochs, one with 4 configs at 8 epochs — and pick the overall winner; (3) log the per-seed losses too.", hint: "Factor the halving loop into a function `halving(config_ids, budget)` returning `(winner, score)`.", solution: [
        { t: "code", lang: "python", code: String.raw`def run_seeds(cfg, epochs, seeds=(0, 1, 2)):
    losses = [run(cfg, epochs, seed=s) for s in seeds]
    return float(np.mean(losses)), losses


def halving(ids, budget, log):
    while True:
        res = {i: run_seeds(configs[i], budget) for i in ids}
        for i in ids:
            log.write(json.dumps({"trial": i, "epochs": budget, "mean": res[i][0], "per_seed": res[i][1]}) + "\n")
        if len(ids) == 1:
            return ids[0], res[ids[0]][0]
        ids = sorted(ids, key=lambda i: res[i][0])[: len(ids) // 2]
        budget *= 2


with open(log_path, "w") as log:
    a = halving(list(range(16)), 2, log)        # aggressive bracket: many configs, short start
    b = halving(list(range(4)), 8, log)         # conservative bracket: few configs, long start
winner = min([a, b], key=lambda r: r[1])
print("winner trial", winner[0], "mean val loss", round(winner[1], 4))` },
        { t: "p", text: "Averaging over seeds reduces the chance that a lucky run survives a rung; the second bracket protects configurations that start slowly. The cost is roughly 3× (seeds) plus the second bracket — the usual trade of compute for reliability." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Tuning solves $\\min_\\lambda \\mathcal{L}_{\\text{val}}(w^*(\\lambda))$ — an expensive, noisy black-box problem; keep the test set out of it.",
        "Random search beats grids when few hyperparameters matter: $n$ distinct values per dimension vs $n^{1/d}$; $P(\\text{hit}) = 1 - (1-q)^n$. Sample scale parameters log-uniformly.",
        "Bayesian optimization: GP posterior $\\mu = k_*^\\top(K+\\sigma_n^2I)^{-1}y$, $\\sigma^2 = k_{**} - k_*^\\top(K+\\sigma_n^2I)^{-1}k_*$; acquisition EI $= (f^+-\\mu)\\Phi(z) + \\sigma\\varphi(z)$.",
        "Successive halving/Hyperband prune bad runs early: cost $(\\log_2 n + 1)nr$ instead of $n\\,r_{\\max}$.",
        "Track code commit, data version, full config, seeds, environment and per-epoch metrics for every run (MLflow, W&B, or a structured log).",
        "Reproducibility = seeds for every RNG + deterministic kernels + pinned versions; still report variability over several seeds.",
        "Ablations attribute gains to components; error analysis tells you what to fix next.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Monitoring, Drift, CI/CD and Versioning](/learn/m18/mlops) — experiment tracking grows into model registries, CI for training and monitoring in production.",
        "[Reading and Reproducing Research Papers](/learn/m19/reading-papers) — ablations, baselines and seeds from the reader's side.",
        "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — taking the tuned, reproducible model to deployment.",
      ] },
    ],
  },
};

export default lesson;
