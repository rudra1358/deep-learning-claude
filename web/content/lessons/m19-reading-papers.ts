import type { Lesson } from "@/lib/types";

const REPRO = String.raw`import numpy as np
from scipy import stats
from sklearn.datasets import make_circles

# Claim to reproduce (qualitatively, after He et al., 2016, "Deep Residual Learning for Image Recognition"):
#   "a very deep plain network is hard to optimise; the same network with identity shortcuts trains well."
# Protocol fixed BEFORE running: same data, width, optimiser (Adam, lr 0.01), 150 full-batch steps and
# the same 5 seeds for every variant; we report mean +- std over seeds and a 95% t-interval.
X, y = make_circles(200, noise=0.08, factor=0.5, random_state=0)     # two concentric rings (non-linear)
WIDTH, STEPS, LR, SEEDS = 16, 150, 0.01, range(5)


def train(depth, residual, branch_scale, seed):
    rng = np.random.default_rng(seed)
    P = {"W_in": rng.normal(0, 1.0, (2, WIDTH)), "b_in": np.zeros(WIDTH),
         "w_out": rng.normal(0, np.sqrt(1 / WIDTH), WIDTH), "b_out": np.zeros(1)}
    for l in range(depth):                                     # He initialisation, optionally scaled
        P[f"W{l}"] = rng.normal(0, np.sqrt(2 / WIDTH), (WIDTH, WIDTH)) * branch_scale
        P[f"b{l}"] = np.zeros(WIDTH)
    m = {k: np.zeros_like(v) for k, v in P.items()}            # Adam moment estimates
    v = {k: np.zeros_like(v) for k, v in P.items()}
    for step in range(1, STEPS + 2):
        # forward:  h_{l+1} = relu(h_l W_l + b_l)        (plain)
        #           h_{l+1} = h_l + relu(h_l W_l + b_l)  (residual: identity shortcut)
        hs, pres = [X @ P["W_in"] + P["b_in"]], []
        for l in range(depth):
            pre = hs[-1] @ P[f"W{l}"] + P[f"b{l}"]
            pres.append(pre)
            a = np.maximum(pre, 0)
            hs.append(hs[-1] + a if residual else a)
        logit = hs[-1] @ P["w_out"] + P["b_out"]
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        if step == STEPS + 1 or not np.isfinite(loss):
            return loss, np.mean((p > 0.5) == y)
        # backward
        g = (p - y) / len(y)
        G = {"w_out": hs[-1].T @ g, "b_out": np.array([g.sum()])}
        gh = np.outer(g, P["w_out"])
        for l in range(depth - 1, -1, -1):
            ga = gh * (pres[l] > 0)
            G[f"W{l}"], G[f"b{l}"] = hs[l].T @ ga, ga.sum(0)
            gh = ga @ P[f"W{l}"].T + (gh if residual else 0)   # the shortcut adds an identity path
        G["W_in"], G["b_in"] = X.T @ gh, gh.sum(0)
        for k in P:                                            # Adam update
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
            P[k] -= LR * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)


def mean_std_ci(vals):
    vals = np.asarray(vals, dtype=float)
    mean, std = vals.mean(), vals.std(ddof=1)                  # sample std across seeds
    half = stats.t.ppf(0.975, len(vals) - 1) * std / np.sqrt(len(vals))
    return mean, std, half


variants = [("plain, depth 4 (control)", 4, False, 1.0),
            ("plain, depth 30", 30, False, 1.0),
            ("plain, depth 60", 60, False, 1.0),
            ("residual, depth 60, branch x1.0", 60, True, 1.0),
            ("residual, depth 60, branch x0.1", 60, True, 0.1)]
print(f"{'variant':<33} {'final loss':>15} {'accuracy':>15}   95% CI (acc)     failed runs")
acc_by = {}
with np.errstate(over="ignore", invalid="ignore"):
    for name, depth, res, scale in variants:
        out = [train(depth, res, scale, s) for s in SEEDS]
        loss = [o[0] for o in out]
        acc = [o[1] for o in out]
        acc_by[name] = acc
        ma, sa, ha = mean_std_ci(acc)
        loss_txt = "diverged (inf)" if not np.all(np.isfinite(loss)) else "%.3f +- %.3f" % mean_std_ci(loss)[:2]
        print(f"{name:<33} {loss_txt:>15} {ma:7.3f} +- {sa:.3f}   [{ma - ha:.3f}, {ma + ha:.3f}]   {sum(a < 0.9 for a in acc)}/{len(acc)}")

print("per-seed accuracy, plain depth 60:          ", np.round(acc_by["plain, depth 60"], 3).tolist())
print("per-seed accuracy, residual depth 60 (x0.1):", np.round(acc_by["residual, depth 60, branch x0.1"], 3).tolist())
# Accuracies are identical within each group (std 0), so a t-test is degenerate. Compare the COUNTS of
# failed runs instead with Fisher's exact test on the 2x2 table [failed, succeeded] x [plain, residual].
fail_plain = sum(a < 0.9 for a in acc_by["plain, depth 60"])
fail_res = sum(a < 0.9 for a in acc_by["residual, depth 60, branch x0.1"])
n = len(SEEDS)
_, pval = stats.fisher_exact([[fail_plain, n - fail_plain], [fail_res, n - fail_res]])
print(f"Fisher exact test, failures {fail_plain}/{n} (plain) vs {fail_res}/{n} (residual x0.1): p = {pval:.4f}")`;
const REPRO_OUT = String.raw`variant                                final loss        accuracy   95% CI (acc)     failed runs
plain, depth 4 (control)           0.000 +- 0.000   1.000 +- 0.000   [1.000, 1.000]   0/5
plain, depth 30                    0.025 +- 0.055   1.000 +- 0.000   [1.000, 1.000]   0/5
plain, depth 60                    0.692 +- 0.005   0.500 +- 0.000   [0.500, 0.500]   5/5
residual, depth 60, branch x1.0   13.771 +- 0.000   0.500 +- 0.000   [0.500, 0.500]   5/5
residual, depth 60, branch x0.1    0.000 +- 0.000   1.000 +- 0.000   [1.000, 1.000]   0/5
per-seed accuracy, plain depth 60:           [0.5, 0.5, 0.5, 0.5, 0.5]
per-seed accuracy, residual depth 60 (x0.1): [1.0, 1.0, 1.0, 1.0, 1.0]
Fisher exact test, failures 5/5 (plain) vs 0/5 (residual x0.1): p = 0.0079`;
const BENCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Claim under test (after Ioffe & Szegedy, 2015): "BatchNorm speeds up training".
# Metric fixed in advance: accuracy after a SHORT budget of 3 epochs of SGD with momentum.
torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.2, stratify=y_trv, random_state=0)
T = lambda a: torch.tensor(a)
X_tr, X_va, X_te, y_tr, y_va, y_te = map(T, (X_tr, X_va, X_te, y_tr, y_va, y_te))
EPOCHS, LR_GRID = 3, [0.003, 0.01, 0.03, 0.1, 0.3]


def make_model(bn=False, dropout=0.0):
    layers, d = [], 64
    for _ in range(3):
        layers += [nn.Linear(d, 128)] + ([nn.BatchNorm1d(128)] if bn else []) + [nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = 128
    return nn.Sequential(*layers, nn.Linear(d, 10))


def run(cfg, lr, seed, X_eval, y_eval):
    torch.manual_seed(seed)
    model = make_model(cfg.get("bn", False), cfg.get("dropout", 0.0))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.get("wd", 0.0))
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(X_tr)).split(32):
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_eval).argmax(1) == y_eval).float().mean().item()
    return acc if np.isfinite(acc) else 0.0


def tune(cfg, seeds=(100, 101, 102)):
    """Pick the learning rate by mean VALIDATION accuracy over 3 tuning seeds (never the test set)."""
    scores = {lr: np.mean([run(cfg, lr, s, X_va, y_va) for s in seeds]) for lr in LR_GRID}
    return max(scores, key=scores.get)


def test_stats(cfg, lr, seeds=range(5)):
    accs = np.array([run(cfg, lr, s, X_te, y_te) for s in seeds])
    half = stats.t.ppf(0.975, len(accs) - 1) * accs.std(ddof=1) / np.sqrt(len(accs))
    return accs, half


def show(name, lr, accs, half):
    print(f"  {name:<28} lr={lr:<6} test acc {accs.mean():.3f} +- {accs.std(ddof=1):.3f}  95% CI [{accs.mean() - half:.3f}, {accs.mean() + half:.3f}]")


# ---- 1. an UNFAIR comparison: proposed method tuned, baseline run at a single default lr ----
print("1) unfair: BatchNorm tuned on the grid, baseline at lr = 0.01")
lr_bn = tune({"bn": True})
base_unfair = test_stats({}, 0.01)
bn_res = test_stats({"bn": True}, lr_bn)
show("baseline (untuned)", 0.01, *base_unfair)
show("BatchNorm (tuned)", lr_bn, *bn_res)

# ---- 2. tuning parity: the same grid and the same selection rule for both ----
print("2) fair: both tuned on the same grid by validation accuracy")
lr_base = tune({})
base_fair = test_stats({}, lr_base)
show("baseline (tuned)", lr_base, *base_fair)
show("BatchNorm (tuned)", lr_bn, *bn_res)
t, p = stats.ttest_rel(bn_res[0], base_fair[0])        # paired: run i of both uses seed i
print(f"  paired t-test over 5 seeds: mean gain {np.mean(bn_res[0] - base_fair[0]):+.3f}, t = {t:.2f}, p = {p:.3f}")

# ---- 3. ablation of a "full" recipe: remove one component at a time, re-tune each variant ----
print("3) ablation (each row re-tuned on the same grid)")
full = {"bn": True, "dropout": 0.2, "wd": 5e-4}
rows = [("full: BN + dropout + wd", full),
        ("  - BatchNorm", {**full, "bn": False}),
        ("  - dropout", {**full, "dropout": 0.0}),
        ("  - weight decay", {**full, "wd": 0.0})]
for name, cfg in rows:
    lr = tune(cfg)
    show(name, lr, *test_stats(cfg, lr))`;
const BENCH_OUT = String.raw`1) unfair: BatchNorm tuned on the grid, baseline at lr = 0.01
  baseline (untuned)           lr=0.01   test acc 0.621 +- 0.048  95% CI [0.562, 0.681]
  BatchNorm (tuned)            lr=0.01   test acc 0.971 +- 0.004  95% CI [0.966, 0.976]
2) fair: both tuned on the same grid by validation accuracy
  baseline (tuned)             lr=0.1    test acc 0.836 +- 0.056  95% CI [0.766, 0.905]
  BatchNorm (tuned)            lr=0.01   test acc 0.971 +- 0.004  95% CI [0.966, 0.976]
  paired t-test over 5 seeds: mean gain +0.136, t = 5.12, p = 0.007
3) ablation (each row re-tuned on the same grid)
  full: BN + dropout + wd      lr=0.03   test acc 0.953 +- 0.015  95% CI [0.934, 0.971]
    - BatchNorm                lr=0.1    test acc 0.867 +- 0.032  95% CI [0.827, 0.907]
    - dropout                  lr=0.01   test acc 0.971 +- 0.004  95% CI [0.966, 0.976]
    - weight decay             lr=0.01   test acc 0.963 +- 0.006  95% CI [0.956, 0.970]`;

const KERAS = String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)


def run(seed, batchnorm):
    tf.keras.utils.set_random_seed(seed)              # seeds Python, NumPy and TensorFlow together
    tf.config.experimental.enable_op_determinism()    # deterministic kernels (slower)
    layers = []
    for _ in range(3):
        layers.append(tf.keras.layers.Dense(128))
        if batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.ReLU())
    model = tf.keras.Sequential(layers + [tf.keras.layers.Dense(10)])
    model.compile(tf.keras.optimizers.SGD(0.01, momentum=0.9),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=3, batch_size=32, verbose=0)
    return model.evaluate(X_te, y_te, verbose=0)[1]


for bn in (False, True):
    accs = np.array([run(s, bn) for s in range(5)])
    print(f"batchnorm={bn}: {accs.mean():.3f} +- {accs.std(ddof=1):.3f} over 5 seeds")`;

const lesson: Lesson = {
  id: "m19/reading-papers",
  title: "Reading and Reproducing Research Papers",
  summary:
    "A working method for research: the three-pass way to read a paper, a decoder for the notation papers use, how to reproduce a claim with seeds, multiple runs and confidence intervals (executed on a real claim — identity shortcuts make a 60-layer MLP trainable), how to benchmark fairly (equal budgets, tuning parity, paired tests), how to design ablations, and how to propose new architectures responsibly.",
  minutes: 80,
  level: "advanced",
  prerequisites: ["m01/dl-workflow", "m07/residual-connections", "m07/normalization", "m17/experimentation"],
  next: ["m19/scaling-efficiency", "m19/interpretability", "m17/experimentation"],
  terms: ["three-pass-reading", "reproducibility", "ablation-study", "tuning-parity", "confidence-interval", "paired-test", "seed-variance"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Deep learning moves through papers: a new idea appears on arXiv, a few groups reproduce it, and it either becomes standard practice (dropout, batch normalisation, residual connections, attention) or quietly disappears. Being able to read a paper quickly, extract the claim, check whether the evidence supports it, and reproduce the key result yourself is the core skill that turns a user of deep learning into a practitioner who can evaluate and extend it.",
      },
      {
        t: "p",
        text: "This is harder than it sounds. Papers compress months of work into eight pages with dense notation; important details (learning-rate schedules, data preprocessing, the number of runs) live in appendices or are missing. Well-documented studies have shown how fragile reported gains can be: Henderson et al. (2018, \"Deep Reinforcement Learning that Matters\") showed that results of deep RL algorithms vary substantially with random seeds and implementation details; Melis, Dyer and Blunsom (2018, \"On the State of the Art of Evaluation in Neural Language Models\") found that carefully tuned standard LSTMs outperformed several more recent architectures; Lucic et al. (2018, \"Are GANs Created Equal?\") found that with equal computational budgets most GAN variants reached similar scores. In each case the gap came from **evaluation practice**, not from the ideas.",
      },
      {
        t: "p",
        text: "This lesson gives a method in four parts: **reading** (the three-pass approach and a notation decoder), **reproducing** (seeds, multiple runs, mean ± standard deviation, confidence intervals), **benchmarking fairly** (same budget, tuning parity, paired comparisons), and **ablating** (removing one component at a time to find what actually matters). It ends with guidance on proposing new architectures responsibly. Both programs in this lesson run real experiments and report the statistics exactly as a careful paper should.",
      },
      {
        t: "callout",
        kind: "key",
        text: "A claim is supported only by an experiment designed *before* looking at the result: fixed metric, fixed budget, equal tuning effort for every method, several seeds, and an uncertainty estimate. Report mean ± std over seeds and a confidence interval $\\bar x \\pm t_{0.975,\\,n-1}\\, s/\\sqrt n$.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[The deep learning workflow](/learn/m01/dl-workflow) — train/validation/test splits and why the test set is touched only once.",
          "[Residual connections](/learn/m07/residual-connections) — the claim reproduced in the From-Scratch program.",
          "[Normalization](/learn/m07/normalization) — the BatchNorm claim benchmarked in the PyTorch program.",
          "[Probability](/learn/m02/probability) — mean, variance and the standard error of a mean.",
          "[Experimentation](/learn/m17/experimentation) — experiment tracking and hyper-parameter search, which this lesson uses to compare methods fairly.",
          "[Backpropagation](/learn/m04/backpropagation) — the NumPy reproduction writes the backward pass of a residual MLP by hand.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Reading a paper like inspecting a house",
        text: "You don't inspect a house by reading every page of the building report first. You walk past it (title, abstract, figures: is it worth a visit?), then you go inside and look at each room (the method and the main results: what is actually here?), and only if you are about to buy do you check the wiring and plumbing behind the walls (the derivations, the appendix, the code: does it really work as advertised?). Reproducing a result is moving in for a week before signing: you find out what the brochure didn't mention.",
      },
      {
        t: "p",
        text: "A benchmark comparison is a race. It is only informative if both runners run the same distance, start at the same time, and both had a coach. Comparing your tuned method with someone else's method at its default settings is like timing yourself on a downhill course and your rival on an uphill one. The PyTorch program shows this concretely: an untuned baseline makes BatchNorm look 35 percentage points better; with equal tuning the gain is 14 points — still real, but less than half.",
      },
      {
        t: "p",
        text: "Seeds matter because training is random: initial weights, data order and dropout masks all change from run to run. One run is one sample from a distribution of outcomes. Two methods whose single runs differ by 1% may be indistinguishable once you see the spread across seeds.",
      },
    ],
    mechanism: [
      { t: "h", text: "The three-pass method" },
      {
        t: "p",
        text: "S. Keshav's short note \"How to Read a Paper\" (2007) describes reading in three passes of increasing depth. Adapted to deep learning papers:",
      },
      {
        t: "steps",
        items: [
          { title: "Pass 1 — the survey (5–10 minutes)", text: "Read the title, abstract, introduction, section headings, every figure and table with its caption, and the conclusion. Skim the references you recognise. Afterwards you should be able to answer the **five Cs**: *Category* (new architecture? training method? analysis? benchmark?), *Context* (which prior work it builds on), *Correctness* (do the assumptions look valid?), *Contributions* (what exactly is claimed to be new), *Clarity*. Decide whether to continue." },
          { title: "Pass 2 — the content (about an hour)", text: "Read the whole paper, skipping proofs. Write the method as pseudocode. For every table, identify the baseline, the metric, the number of runs and whether error bars are reported. Note what is compared with what under which budget. List the terms and references you don't know." },
          { title: "Pass 3 — the reconstruction (several hours)", text: "Re-derive the key equations, check dimensions of every tensor, and mentally (or actually) re-implement the method. Challenge every assumption: what would happen if this component were removed? Is the improvement within seed noise? Is the baseline tuned? At the end you should be able to explain the paper's weak points and reproduce its main figure." },
          { title: "Then reproduce", text: "Pick the smallest experiment that tests the central claim, fix the protocol before running, use several seeds, and compare against the paper's numbers or qualitative trend." },
        ],
      },
      { t: "h", text: "Reproducing a result" },
      {
        t: "steps",
        items: [
          { title: "State the claim precisely", text: "Turn \"residual connections help deep networks\" into a falsifiable statement: \"with the same width, optimiser, learning rate and step budget, a 60-layer MLP with identity shortcuts reaches lower training loss than the same MLP without them\"." },
          { title: "Fix the protocol in advance", text: "Data, model sizes, optimiser, budget, metric, number of seeds and the statistical test. Write them down (in code, as constants) before the first run, so the result cannot steer the choices." },
          { title: "Include a control", text: "A condition where the effect should be absent — here a shallow plain network that trains perfectly, which shows the task is solvable and the code works." },
          { title: "Run several seeds", text: "Change only the seed. Report each per-seed number, the mean, the sample standard deviation and a confidence interval." },
          { title: "Compare with the paper", text: "Exact numbers rarely match (different hardware, libraries, data versions); the direction and rough size of the effect should. If they do not, check the implementation details first, then report the discrepancy honestly." },
        ],
      },
      { t: "h", text: "Fair benchmarking checklist" },
      {
        t: "list",
        items: [
          "**Same data and splits** for every method, with the test set used once at the end; hyper-parameters chosen on a validation split only.",
          "**Same budget** — epochs, optimiser steps, wall-clock or FLOPs — stated explicitly. A method that trains twice as long is not comparable.",
          "**Tuning parity** — the same search space size and selection rule for the baseline and the new method. Tuning only your own method is the most common source of inflated gains.",
          "**Several seeds** and an uncertainty estimate; use **paired** comparisons when runs can share seeds and data order.",
          "**Strong, up-to-date baselines** — the published numbers of a baseline from five years ago are not a baseline for a model trained with today's tricks.",
          "**Report everything**: parameter count, compute, failed runs, and negative results.",
        ],
      },
      { t: "h", text: "Ablation studies" },
      {
        t: "p",
        text: "An **ablation** removes (or replaces with a neutral alternative) one component of a method at a time while keeping everything else fixed, to measure that component's contribution. Each ablated variant should be re-tuned with the same search budget — otherwise you measure \"component removed *and* hyper-parameters now wrong\". Ablations answer \"which parts matter?\" and often reveal that a paper's headline gain comes from an incidental change (a longer schedule, a different augmentation) rather than the proposed module.",
      },
    ],
    math: [
      { t: "h", text: "1. Decoding notation" },
      {
        t: "p",
        text: "Notation varies between papers, but most deep learning papers draw on a common set of conventions. When a paper defines a symbol, its definition wins; the table below is the default reading.",
      },
      {
        t: "table",
        head: ["Notation", "Usual meaning", "Example"],
        rows: [
          ["$x, \\mathbf x, \\boldsymbol x$", "lower case (often bold): a vector; plain italic often a scalar", "$\\mathbf x \\in \\mathbb R^d$ an input"],
          ["$W, \\mathbf W$", "upper case: a matrix (weights)", "$W \\in \\mathbb R^{d_{out}\\times d_{in}}$"],
          ["$\\theta, \\phi$", "all trainable parameters of a model (encoder/decoder often $\\phi$/$\\theta$)", "$f_\\theta(x)$"],
          ["$\\mathcal D, \\mathcal L, \\mathcal X$", "calligraphic: sets, datasets, loss functionals, spaces", "$\\mathcal D = \\{(x_i, y_i)\\}_{i=1}^N$"],
          ["$\\hat y, \\tilde x, \\bar x$", "hat: prediction/estimate; tilde: corrupted or alternative version; bar: average", "$\\hat y = f_\\theta(x)$"],
          ["$x^{(i)}, x_i, x_t, x^{[l]}$", "superscript (i) or subscript i: example index; t: time step; [l]: layer", "$h_t = f(h_{t-1}, x_t)$"],
          ["$\\mathbb E_{x\\sim p}[\\cdot]$", "expectation over $x$ drawn from $p$", "$\\mathbb E_{(x,y)\\sim\\mathcal D}[L]$"],
          ["$\\nabla_\\theta L$, $\\frac{\\partial L}{\\partial\\theta}$", "gradient / partial derivative w.r.t. θ", "$\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta L$"],
          ["$\\odot$, $\\otimes$, $\\oplus$, $[a; b]$", "element-wise product; outer/Kronecker product; often concatenation or direct sum; concatenation", "$c_t = f_t\\odot c_{t-1} + i_t\\odot\\tilde c_t$"],
          ["$\\|v\\|_p$, $\\|A\\|_F$", "$p$-norm of a vector; Frobenius norm of a matrix", "$\\|\\delta\\|_\\infty \\le \\epsilon$"],
          ["$\\sigma(\\cdot)$, $\\mathrm{softmax}$", "sigmoid (sometimes any activation, or a standard deviation — check!)", "$\\sigma(z) = 1/(1+e^{-z})$"],
          ["$\\mathcal N(\\mu, \\Sigma)$, $\\mathcal U(a, b)$", "Gaussian and uniform distributions", "$\\epsilon\\sim\\mathcal N(0, I)$"],
          ["$\\mathrm{KL}(p\\,\\|\\,q)$, $H(p)$, $I(X;Y)$", "KL divergence, entropy, mutual information", "VAE regulariser"],
          ["$\\mathcal O(\\cdot)$, $\\propto$, $\\approx$, $\\triangleq$", "asymptotic order; proportional to; approximately; equal by definition", "attention costs $\\mathcal O(n^2 d)$"],
          ["$\\arg\\max$, $\\arg\\min$", "the argument that maximises/minimises", "$\\hat y = \\arg\\max_k p_k$"],
          ["$\\mathbb 1[\\cdot]$", "indicator: 1 if the condition holds, else 0", "accuracy $= \\frac1N\\sum_i \\mathbb 1[\\hat y_i = y_i]$"],
          ["$B, T, d, h, L$", "batch size, sequence length, model width, number of heads, number of layers", "tensor of shape $B\\times T\\times d$"],
        ],
      },
      {
        t: "p",
        text: "A practical decoding habit: for every equation, write the **shape** of each symbol next to it ($Q \\in \\mathbb R^{T\\times d_k}$, $QK^\\top \\in \\mathbb R^{T\\times T}$). Shape mismatches reveal misreadings immediately, and the shapes tell you how to write the code.",
      },
      { t: "h", text: "2. Mean, standard deviation and standard error" },
      {
        t: "p",
        text: "With $n$ runs (seeds) giving scores $x_1, \\dots, x_n$, report the sample mean $\\bar x$ and the sample standard deviation $s$ (with $n - 1$, *Bessel's correction*, because the mean was estimated from the same data). The uncertainty of the *mean* is the **standard error** $s/\\sqrt n$ — it shrinks with more runs, while $s$ (the spread of individual runs) does not.",
      },
      { t: "math", tex: String.raw`\bar x = \frac1n\sum_{i=1}^n x_i,\qquad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar x)^2},\qquad \mathrm{SE} = \frac{s}{\sqrt n}` },
      { t: "h", text: "3. Confidence interval for the mean" },
      {
        t: "p",
        text: "Assuming roughly normal run-to-run variation, a 95% confidence interval for the true mean uses Student's $t$ distribution with $n - 1$ degrees of freedom (wider than the normal $\\pm1.96\\,\\mathrm{SE}$ for small $n$):",
      },
      { t: "math", tex: String.raw`\bar x \pm t_{0.975,\,n-1}\,\frac{s}{\sqrt n},\qquad t_{0.975,4} = 2.776,\quad t_{0.975,9} = 2.262,\quad t_{0.975,\infty} = 1.960` },
      {
        t: "p",
        text: "**Worked example** (illustrative numbers). Method A over five seeds: accuracies 0.91, 0.93, 0.92, 0.95, 0.90. Then $\\bar x = 0.922$, $s = 0.019235$, $\\mathrm{SE} = 0.019235/\\sqrt5 = 0.008602$, and the 95% CI is $0.922 \\pm 2.776\\times0.008602 = 0.922 \\pm 0.023884$, i.e. [0.898, 0.946]. Five seeds pin the mean down only to about ±2.4 points.",
      },
      { t: "h", text: "4. Paired versus unpaired comparisons" },
      {
        t: "p",
        text: "Method B on the *same five seeds* (same initialisation stream and data order): 0.90, 0.91, 0.91, 0.92, 0.89, mean 0.906. Two ways to test whether A beats B:",
      },
      {
        t: "steps",
        items: [
          { title: "Unpaired (Welch's t-test)", text: "Treats the two sets as independent. With $s_A = 0.019235$ and $s_B = 0.011402$ the statistic is $t = 0.016/\\sqrt{s_A^2/5 + s_B^2/5} = 1.60$, $p = 0.157$: no evidence of a difference." },
          { title: "Paired t-test", text: "Uses the per-seed differences $d = (0.01, 0.02, 0.01, 0.03, 0.01)$: $\\bar d = 0.016$, $s_d = 0.008944$, $t = \\bar d/(s_d/\\sqrt5) = 4.00$, $p = 0.016$.", tex: String.raw`t = \frac{\bar d}{s_d/\sqrt n} = \frac{0.016}{0.008944/2.236} = 4.00` },
        ],
      },
      {
        t: "p",
        text: "Pairing removes the variation that both methods share (a \"lucky\" seed is lucky for both), so the same data give a much more sensitive test. Pairing is valid only if run $i$ of both methods really shares the source of randomness. With many comparisons, correct for multiple testing (e.g. Bonferroni: divide the significance level by the number of comparisons), and remember that statistical significance is not practical importance — always report the effect size $\\bar d$ with its interval.",
      },
      { t: "h", text: "5. How many seeds?" },
      {
        t: "p",
        text: "To detect a mean difference $\\Delta$ when per-run differences have standard deviation $\\sigma_d$, a paired test with 5% significance and 80% power needs roughly $n \\approx \\big((z_{0.975} + z_{0.8})\\,\\sigma_d/\\Delta\\big)^2 = (2.80\\,\\sigma_d/\\Delta)^2$ runs (normal approximation; add one or two for the $t$ correction). With $\\sigma_d = 0.009$ and a hoped-for $\\Delta = 0.005$: $(2.80 \\times 1.8)^2 \\approx 25$ seeds. Small effects need many runs — which is why single-run comparisons of sub-1% improvements are uninformative.",
      },
      { t: "h", text: "6. Counting outcomes: Fisher's exact test" },
      {
        t: "p",
        text: "When the outcome is binary per run (trained / failed), compare counts with **Fisher's exact test** on the $2\\times2$ table. In the From-Scratch program all 5 plain 60-layer runs failed and 0 of 5 residual runs did; the probability of a split at least this extreme under \"no difference\" is $2\\times\\binom55\\binom50/\\binom{10}{5} = 2/252 = 0.0079$.",
      },
    ],
    visual: [
      {
        t: "p",
        text: "The claim reproduced in the From-Scratch program is about how gradients travel through depth. Use the gradient-flow simulator to see the mechanism behind the numbers before reading them.",
      },
      { t: "sim", id: "gradient-flow", caption: "Gradient magnitudes per layer in a deep network, with and without the ingredients that keep them healthy." },
      {
        t: "list",
        items: [
          "Make the network deep and observe how the gradient norm changes from the last layer to the first. In a plain deep ReLU network small per-layer factors compound multiplicatively.",
          "Compare with the setting that keeps gradients near constant across depth, and connect it to the identity path $\\partial h_{l+1}/\\partial h_l = I + \\partial F/\\partial h_l$ of a residual block — the shortcut contributes the identity term regardless of what the branch does.",
          "Then read the From-Scratch table: the 60-layer plain MLP's loss stays at $\\ln 2 \\approx 0.693$ (it never learns), the residual version reaches 0.000.",
        ],
      },
      { t: "sim", id: "batchnorm", caption: "BatchNorm, the component benchmarked in the PyTorch program: normalise per feature over the batch, then scale and shift." },
      {
        t: "p",
        text: "In the BatchNorm simulator, look at how the normalised activations keep a stable scale regardless of the input scale. That stability is one reason a larger range of learning rates works with BatchNorm — which is exactly why comparing it with an *untuned* baseline exaggerates its benefit.",
      },
    ],
    scratch: [
      {
        t: "p",
        text: "A complete, pre-registered mini-reproduction in NumPy of the central claim of He et al. (2016, \"Deep Residual Learning for Image Recognition\"): very deep plain networks are hard to optimise, and identity shortcuts fix this. Five variants, five seeds each, the same data, width, optimiser (Adam, written by hand), learning rate and step budget; a shallow control; an ablation of the residual branch scale; mean ± std, 95% intervals and an exact test.",
      },
      { t: "code", lang: "python", title: "reproduce_residual_claim.py", code: REPRO, output: REPRO_OUT },
      {
        t: "list",
        items: [
          "**Control works**: the depth-4 plain network solves the task in every run (loss 0.000), so the task and the code are fine.",
          "**Depth 30 still trains** (5/5 runs), **depth 60 fails in 5/5 runs**: the loss stays at 0.692 ≈ ln 2, the loss of always predicting 50%. The claim concerns *very* deep plain networks; the depth at which failure starts depends on the width, initialisation and optimiser, which is why a reproduction must state them.",
          "**Shortcuts fix it**: with identity shortcuts and residual branches scaled by 0.1 at initialisation, every 60-layer run reaches loss 0.000.",
          "**The ablation matters**: identity shortcuts *without* down-scaling the branches fail too — the forward signal grows at every block (each block adds a He-initialised branch to its input), the logits saturate and the loss is huge (13.771 is the clipped log-loss of confidently wrong predictions). Practical residual networks control this with BatchNorm, small or zero initialisation of the last layer in each branch, or explicit scaling. Without this ablation one might wrongly conclude that \"shortcuts alone\" are sufficient.",
          "**Statistics**: the outcomes are identical within each group (std 0), so a t-test is meaningless; Fisher's exact test on failure counts gives $p = 0.0079$. With only five seeds, an effect must be very large to be convincing — this one is.",
        ],
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "Testing the claim \"BatchNorm speeds up training\" (Ioffe and Szegedy, 2015) on the digits with a fixed short budget (3 epochs), first unfairly, then with tuning parity (same learning-rate grid, chosen on a validation split with separate tuning seeds), then with an ablation of a three-component recipe in which every ablated variant is re-tuned.",
              },
              { t: "code", lang: "python", title: "fair_benchmark_and_ablation.py", code: BENCH, output: BENCH_OUT },
              {
                t: "list",
                items: [
                  "**Unfair**: baseline at a single \"default\" lr = 0.01 → 0.621; BatchNorm tuned → 0.971. Apparent gain: +0.350.",
                  "**Fair**: the baseline's best lr on the same grid is 0.1 → 0.836. The gain shrinks to +0.136 — still large and significant (paired t-test over seeds, $p = 0.007$), so the claim survives, but the unfair version overstated it by a factor of 2.6. Note also the baseline's much larger seed variance (± 0.056 vs ± 0.004): BatchNorm makes this short-budget training more reliable as well as faster.",
                  "**Ablation**: removing BatchNorm costs 8.6 points; removing dropout *improves* accuracy (0.953 → 0.971) because dropout slows learning in a 3-epoch budget; removing weight decay changes little (0.963). A paper presenting \"BN + dropout + weight decay\" as one recipe would hide that one component hurts under this budget.",
                  "Tuning seeds (100–102) are different from evaluation seeds (0–4), and the learning rate is selected on validation data, so the reported test numbers are not optimistically biased by selection.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "The reproducibility essentials in Keras: one call seeds Python, NumPy and TensorFlow, and op determinism makes GPU kernels deterministic (at some speed cost). Results are reported over five seeds." },
              { t: "code", lang: "python", title: "seeds_keras.py (not executed in CI)", code: KERAS, output: "Prints, for batchnorm=False and batchnorm=True, the mean ± standard deviation of test accuracy over 5 seeds after 3 epochs." },
            ],
          },
        ],
      },
    ],
    variations: [
      { t: "h", text: "Kinds of reproduction" },
      {
        t: "table",
        head: ["Kind", "What is re-done", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["Re-run released code", "same code, same data, new hardware", "fast; tests the artefact", "inherits any bugs; tests little about the idea", "first sanity check"],
          ["Re-implementation", "new code from the paper's description", "tests whether the paper fully specifies the method", "slow; discrepancies hard to attribute", "core of serious reproduction"],
          ["Qualitative / small-scale reproduction", "the claimed trend on a smaller problem", "cheap; isolates the mechanism", "may not transfer to the paper's scale", "this lesson's programs; course projects"],
          ["Replication on new data", "same method, different data or task", "tests generality", "a failure may be due to the new setting", "robustness of conclusions"],
          ["Re-evaluation with stronger baselines", "same comparison with tuned baselines", "exposes inflated gains", "compute for tuning", "e.g. Melis et al. (2018) for language models"],
        ],
      },
      { t: "h", text: "Comparison and uncertainty methods" },
      {
        t: "table",
        head: ["Method", "When", "Notes"],
        rows: [
          ["mean ± std over seeds", "always", "std describes run-to-run spread"],
          ["t confidence interval", "uncertainty of a mean from a few runs", "$\\bar x \\pm t_{0.975,n-1}s/\\sqrt n$"],
          ["paired t-test / Wilcoxon signed-rank", "two methods on shared seeds or the same test examples", "much more sensitive than unpaired; Wilcoxon avoids the normality assumption"],
          ["bootstrap CI over test examples", "uncertainty due to a finite test set", "resample the test set with replacement; complements seed variation"],
          ["McNemar's test", "two classifiers on the same test set, one run each", "uses the discordant pairs (A right/B wrong vs A wrong/B right)"],
          ["Fisher's exact test", "binary outcomes per run (diverged/converged)", "exact for small counts"],
          ["multiple-comparison correction", "many methods or datasets", "Bonferroni or Holm adjustments"],
        ],
      },
      { t: "h", text: "Designing novel architectures responsibly" },
      {
        t: "list",
        items: [
          "**Start from a hypothesis about a failure mode** (\"attention cost grows quadratically\", \"gradients vanish with depth\") rather than from a module to try. State what the change should improve and what it should not affect.",
          "**Match compute and parameters** with the baseline, or plot accuracy against FLOPs/parameters so the comparison is visible across scales.",
          "**Ablate every new component** and re-tune each variant; keep the baseline training recipe identical (schedule, augmentation, regularisation), or improve it for the baseline too.",
          "**Test on more than one dataset and scale**; check whether the gain shrinks as the baseline gets stronger.",
          "**Report negative and null results**, failed seeds, and the compute used (Dodge et al., 2019, \"Show Your Work\", argue for reporting expected performance as a function of tuning budget).",
          "**Release code, configs and seeds**; checklists such as the one in Pineau et al. (2021, \"Improving Reproducibility in Machine Learning Research\") summarise what to include.",
          "**Avoid overclaiming**: Lipton and Steinhardt (2018, \"Troubling Trends in Machine Learning Scholarship\") discuss explanation vs speculation, unclear sources of gains and misuse of language.",
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Adopting a technique at work** — before switching an internal model to a new optimiser or architecture from a paper, reproduce the gain on your data with seeds and tuned baselines; many published gains do not transfer.",
          "**Model selection for production** — a 0.3% offline gain from a single run is noise; paired tests on shared seeds or McNemar's test on the same test set decide whether a change is real.",
          "**Reviewing papers and pull requests** — the fair-benchmarking checklist is what reviewers check: budgets, tuning parity, seeds, ablations.",
          "**Reproducibility challenges and course projects** — re-implementing a recent paper is a standard way to learn a subfield and frequently uncovers missing details.",
          "**Research** — every new method needs ablations and fair baselines to be accepted; scaling studies ([Scaling Laws](/learn/m19/scaling-efficiency)) are reproductions across many model sizes.",
          "**Science and medicine** — preregistration (fixing the analysis before seeing results) is standard in clinical trials; the same discipline makes ML results trustworthy.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Small-scale reproductions are cheap and isolate mechanisms**, but effects can depend on scale; a trend that holds on digits may vanish (or appear) on ImageNet.",
          "**More seeds give tighter intervals**, at linear compute cost; the standard error only falls as $1/\\sqrt n$.",
          "**Tuning parity costs compute** (here 15 tuning runs per variant), but it is the only way to make a comparison meaningful.",
          "**Preregistering the protocol prevents self-deception**, but exploratory work is still valuable — label it as exploratory and confirm on fresh seeds/data.",
          "**Ablations multiply runs** (one per component, each re-tuned) but reveal which parts matter; unablated multi-component methods are hard to build on.",
          "**Exact numerical reproduction is often impossible** across hardware and library versions (non-deterministic GPU kernels); aim for statistical reproduction: the same conclusion within the reported uncertainty.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Reporting a single run.** One seed is one sample. Fix: ≥ 3–5 seeds (more for small effects), mean ± std and a CI.",
          "**Tuning hyper-parameters on the test set** (or picking the best epoch by test accuracy). This leaks information and biases results upward. Fix: select on validation data; evaluate on test once.",
          "**Tuning only the proposed method.** Fix: identical search space and budget for every method (+0.350 vs +0.136 in our example).",
          "**Comparing different budgets** — more epochs, larger models or extra data for the new method. Fix: fix the budget, or plot performance against compute.",
          "**Using std when you mean standard error (or vice versa).** Std is the spread of individual runs; SE = std/√n is the uncertainty of the mean. Fix: say which one, and give n.",
          "**Unpaired tests on paired data** — throwing away sensitivity (p = 0.157 vs 0.016 on the same numbers). Fix: pair by seed when the randomness is shared.",
          "**Ablating without re-tuning.** Removing BatchNorm but keeping the learning rate tuned for BatchNorm measures the wrong thing. Fix: re-tune each ablated variant with the same budget.",
          "**Misreading notation** — e.g. σ as standard deviation instead of sigmoid, or $W$ stored as $d_{in}\\times d_{out}$ vs $d_{out}\\times d_{in}$. Fix: write shapes next to every symbol and check against the paper's code if available.",
          "**Comparing your reproduction's number with the paper's number from a different setting** (different data split, preprocessing or metric definition). Fix: match the setting, or compare trends.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Describe the three-pass method for reading a paper.", a: "Pass 1 (minutes): title, abstract, introduction, headings, figures and conclusion — decide what kind of paper it is and whether it is relevant (the five Cs: category, context, correctness, contributions, clarity). Pass 2 (about an hour): read everything except proofs, write the method as pseudocode, note baselines, metrics and runs for each result. Pass 3 (hours): re-derive and virtually re-implement, challenge assumptions, identify weaknesses — after which you could reproduce the main result (Keshav, 2007)." },
      { t: "qa", level: "beginner", q: "Why should results be reported over several random seeds?", a: "Training is stochastic (initialisation, data order, dropout, non-deterministic kernels), so one run is one sample of a distribution of outcomes. Differences between methods are meaningful only relative to the run-to-run spread, which requires several runs; report mean ± std and a confidence interval." },
      { t: "qa", level: "beginner", q: "What is an ablation study?", a: "An experiment that removes or neutralises one component of a method at a time, keeping everything else (and the tuning budget) the same, to measure each component's contribution. In our example, removing BatchNorm cost 8.6 points while removing dropout improved accuracy by 1.8 points." },
      { t: "qa", level: "intermediate", q: "What is tuning parity and why does it matter?", a: "Giving every compared method the same hyper-parameter search space, budget and selection rule (on validation data). Without it, a tuned new method is compared with an untuned baseline and the gain is inflated. In the PyTorch program the apparent BatchNorm gain fell from +0.350 to +0.136 when the baseline's learning rate was tuned too." },
      { t: "qa", level: "intermediate", q: "Five seeds give accuracies 0.91, 0.93, 0.92, 0.95, 0.90. Give a 95% confidence interval for the mean.", a: "$\\bar x = 0.922$, $s = 0.019235$, SE $= s/\\sqrt5 = 0.008602$, $t_{0.975,4} = 2.776$, so $0.922 \\pm 0.0239$ = [0.898, 0.946]." },
      { t: "qa", level: "intermediate", q: "When should you use a paired test instead of an unpaired one?", a: "When each run of method A has a natural partner run of method B sharing the same source of randomness (same seed, data order or test examples). The paired test works with per-pair differences, removing shared variation; on the same numbers our paired test gave p = 0.016 while Welch's unpaired test gave p = 0.157." },
      { t: "qa", level: "intermediate", q: "A reproduction gives 2 points lower accuracy than the paper. What do you check?", a: "That the setting matches: data version and split, preprocessing and augmentation, model size, training length and schedule, evaluation protocol (e.g. single-crop vs multi-crop, test-time augmentation), metric definition. Then implementation details often omitted (initialisation, weight decay on biases/norm layers, warm-up, EMA of weights). Then statistical noise: does the paper report error bars? Contact the authors or check released code; report the discrepancy honestly if unresolved." },
      { t: "qa", level: "advanced", q: "How many seeds would you need to detect a 0.5-point improvement if paired differences have a standard deviation of 0.9 points?", a: "For 5% two-sided significance and 80% power, $n \\approx ((1.96 + 0.84)\\sigma_d/\\Delta)^2 = (2.80\\times0.9/0.5)^2 \\approx 25$ runs (slightly more with the $t$ correction). Hence single-run sub-point comparisons are not evidence." },
      { t: "qa", level: "advanced", q: "How would you design the evaluation of a new architecture you invented?", a: "Formulate the hypothesis and the failure mode it addresses; choose strong, current baselines and train them with the same recipe; match parameters and FLOPs or report accuracy-vs-compute curves; tune every method with the same budget on validation data; use several seeds with paired comparisons and CIs; ablate each new component (re-tuned); test on several datasets and scales; report compute, failed runs and negative results; release code and configs." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "In the first pass of the three-pass method you should…", options: ["re-derive every equation", "read the title, abstract, introduction, headings, figures and conclusion", "re-implement the method", "read only the related work"], answer: 1, explain: "Pass 1 is a 5–10 minute survey to classify the paper and decide whether to continue." },
          { kind: "numeric", q: "Four seeds give losses 0.50, 0.54, 0.52, 0.56. What is the sample standard deviation (with n − 1)? (3 decimals)", answer: 0.026, tolerance: 0.001, explain: "Mean 0.53; squared deviations 0.0009, 0.0001, 0.0001, 0.0009, sum 0.002; $s = \\sqrt{0.002/3} = 0.0258$." },
          { kind: "numeric", q: "Ten runs have $s = 0.02$. What is the standard error of the mean? (4 decimals)", answer: 0.0063, tolerance: 0.0002, explain: "$0.02/\\sqrt{10} = 0.006325$." },
          { q: "Your method is tuned over 20 configurations, the baseline uses the paper's defaults. The comparison is…", options: ["fair, because the defaults were tuned by the original authors", "biased in favour of your method — tuning parity is violated", "biased against your method", "fair if you use 5 seeds"], answer: 1, explain: "The baseline's defaults were tuned for a different setting; both need the same search budget on your setup." },
          { q: "Removing a component in an ablation, you should…", options: ["keep all hyper-parameters fixed at the full model's best values", "re-tune the ablated variant with the same budget", "evaluate it on the test set to choose its learning rate", "train it for fewer epochs"], answer: 1, explain: "Otherwise you measure the component's removal combined with mismatched hyper-parameters." },
          { kind: "numeric", q: "Paired differences over 5 seeds have mean 0.02 and standard deviation 0.01. What is the paired t statistic? (2 decimals)", answer: 4.47, tolerance: 0.02, explain: "$t = 0.02/(0.01/\\sqrt5) = 0.02/0.004472 = 4.47$." },
          { q: "In the From-Scratch reproduction, why was the depth-4 plain network included?", options: ["To make the table longer", "As a control showing the task is solvable and the code is correct", "Because He et al. used depth 4", "To tune the learning rate"], answer: 1, explain: "Without the control, the depth-60 failure could be blamed on a bug or an impossible task." },
          { q: "In the paper notation $h_t = f(h_{t-1}, x_t)$, the subscript $t$ most likely denotes…", options: ["the layer", "the training example", "the time step", "the transpose"], answer: 2, explain: "Subscript $t$ conventionally indexes time steps in sequence models; the paper's own definitions take precedence." },
        ],
      },
      {
        t: "exercise",
        title: "Statistics of a comparison by hand",
        prompt: "Method A on 4 seeds: 0.80, 0.84, 0.82, 0.86. Method B on the same seeds: 0.79, 0.81, 0.81, 0.83. (a) Mean and sample std of each. (b) 95% CI for A's mean ($t_{0.975,3} = 3.182$). (c) Paired t statistic for A − B. (d) Is the difference significant at 5% (critical value 3.182)?",
        hint: "Paired differences: 0.01, 0.03, 0.01, 0.03.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) A", tex: String.raw`\bar x_A = 0.83,\quad s_A = \sqrt{\tfrac{0.0009 + 0.0001 + 0.0001 + 0.0009}{3}} = \sqrt{0.000667} = 0.025820` },
              { title: "(a) B", tex: String.raw`\bar x_B = 0.81,\quad s_B = \sqrt{\tfrac{0.0004 + 0 + 0 + 0.0004}{3}} = 0.016330` },
              { title: "(b) CI for A", tex: String.raw`0.83 \pm 3.182\times\frac{0.025820}{2} = 0.83 \pm 0.041080 \;\Rightarrow\; [0.789,\ 0.871]` },
              { title: "(c) paired t", tex: String.raw`\bar d = 0.02,\quad s_d = \sqrt{\tfrac{4\times0.0001}{3}} = 0.011547,\quad t = \frac{0.02}{0.011547/2} = 3.464` },
              { title: "(d) decision", text: "$3.464 > 3.182$: significant at the 5% level (p ≈ 0.04). An unpaired test on the same numbers would give $t = 0.02/\\sqrt{(0.025820^2 + 0.016330^2)/4} = 1.31$ — not significant." },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Coding: reproduce \"BatchNorm allows higher learning rates\"",
        prompt: "Ioffe and Szegedy (2015) also claimed that BatchNorm makes training work at higher learning rates. Using `fair_benchmark_and_ablation.py` as a starting point, design and run a pre-registered experiment: for lr in {0.01, 0.03, 0.1, 0.3, 1.0}, train the baseline and the BatchNorm model for 3 epochs over 5 seeds and report mean ± std of validation accuracy, plus the number of diverged runs (accuracy ≤ 0.15). What would count as support for the claim?",
        hint: "Reuse `run(cfg, lr, seed, X_va, y_va)`; count a run as diverged if its accuracy is near chance (0.1).",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "for lr in [0.01, 0.03, 0.1, 0.3, 1.0]:\n    for name, cfg in [(\"baseline\", {}), (\"batchnorm\", {\"bn\": True})]:\n        accs = np.array([run(cfg, lr, s, X_va, y_va) for s in range(5)])\n        print(f\"lr={lr:<5} {name:<10} {accs.mean():.3f} +- {accs.std(ddof=1):.3f}  diverged {np.sum(accs <= 0.15)}/5\")",
          },
          { t: "p", text: "Fix before running: support means that at the largest learning rates the BatchNorm model keeps high accuracy with few or no diverged runs while the baseline degrades or diverges, consistently across seeds. If both degrade at similar learning rates, the claim is not supported in this setting. Either way, report all five rows — a learning-rate sweep is more informative than a single best number." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Three passes: survey (five Cs) → content (pseudocode, check baselines/metrics/runs) → reconstruction (re-derive, challenge, could reproduce).",
          "Decode notation by writing the shape of every symbol; a paper's own definitions override conventions.",
          "Reproduce: precise falsifiable claim, protocol fixed in advance, a control, several seeds, compare the trend with the paper.",
          "Statistics: $\\bar x$, $s$ (n − 1), SE $= s/\\sqrt n$, 95% CI $\\bar x \\pm t_{0.975,n-1}\\,s/\\sqrt n$; paired tests for shared seeds; Fisher's exact test for counts; seeds needed $\\approx (2.8\\sigma_d/\\Delta)^2$.",
          "Fair benchmarking: same data, budget, tuning space and selection rule; test set once; strong baselines; report compute and failures.",
          "Ablations: remove one component at a time and re-tune each variant.",
          "Results here: 60-layer plain MLP failed 5/5, residual (branch ×0.1) succeeded 5/5, p = 0.0079; unscaled shortcuts also failed. BatchNorm's gain: +0.350 unfair vs +0.136 fair.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency) — scaling studies are systematic reproductions across model sizes; fit your own power law.",
          "[Interpretability](/learn/m19/interpretability) — methods whose evaluation especially needs sanity checks and careful baselines.",
          "[Experimentation](/learn/m17/experimentation) — tooling for tracking runs, seeds and hyper-parameter searches.",
          "[Adversarial Examples and Robustness](/learn/m19/robustness-safety) — a field where weak evaluation repeatedly produced false claims.",
        ],
      },
    ],
  },
};

export default lesson;
