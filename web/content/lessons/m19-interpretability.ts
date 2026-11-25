import type { Lesson } from "@/lib/types";

const TAB = String.raw`import itertools
import math
import numpy as np
from sklearn.datasets import load_breast_cancer

# ---------------- a small tabular model: 6 features -> MLP (tanh, 16 units) -> P(benign) ----------------
data = load_breast_cancer()
cols = [0, 1, 4, 6, 20, 27]                              # mean radius, mean texture, mean smoothness,
names = [data.feature_names[c] for c in cols]           # mean concavity, worst radius, worst concave points
X = data.data[:, cols]
X = (X - X.mean(0)) / X.std(0)                          # standardise
y = data.target.astype(float)                           # 1 = benign
rng = np.random.default_rng(0)
W1, b1 = rng.normal(0, 0.5, (6, 16)), np.zeros(16)
w2, b2 = rng.normal(0, 0.5, 16), 0.0


def f(Z):
    """Model output: the LOGIT of P(benign). Accepts (n, 6) or (6,)."""
    return np.tanh(Z @ W1 + b1) @ w2 + b2


def grad_f(z):
    h = np.tanh(z @ W1 + b1)
    return W1 @ ((1 - h ** 2) * w2)                      # d logit / d input


for _ in range(3000):                                   # full-batch gradient descent on cross-entropy
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ w2 + b2)))
    g = (p - y) / len(y)
    gH = np.outer(g, w2) * (1 - H ** 2)
    w2 -= 0.5 * H.T @ g
    b2 -= 0.5 * g.sum()
    W1 -= 0.5 * X.T @ gH
    b1 -= 0.5 * gH.sum(0)
print(f"training accuracy {np.mean((f(X) > 0) == y):.3f}")

x = X[0]                                                # explain patient 0 (malignant)
background = X[rng.choice(len(X), 100, replace=False)]  # reference distribution for "absent" features
print(f"explained example: label {int(y[0])}, logit f(x) = {f(x):.4f}, mean background logit = {f(background).mean():.4f}\n")

# ---------------- 1. exact Shapley values by enumerating all 2^6 coalitions ----------------
M = 6


def v(S):
    """Value of coalition S: average output when features in S take x's values and the others come from background."""
    Z = background.copy()
    Z[:, list(S)] = x[list(S)]
    return f(Z).mean()


vals = {S: v(S) for k in range(M + 1) for S in itertools.combinations(range(M), k)}
phi = np.zeros(M)
for i in range(M):
    for S, vS in vals.items():
        if i in S:
            continue
        Si = tuple(sorted(S + (i,)))
        weight = math.factorial(len(S)) * math.factorial(M - len(S) - 1) / math.factorial(M)
        phi[i] += weight * (vals[Si] - vS)               # weighted marginal contribution of feature i
print("efficiency check: sum(phi) = {:.4f}  vs  f(x) - E[f] = {:.4f}".format(phi.sum(), vals[tuple(range(M))] - vals[()]))

# ---------------- 2. gradient x input and integrated gradients (baseline = background mean) ----------------
base = background.mean(0)
gxi = grad_f(x) * (x - base)
alphas = (np.arange(200) + 0.5) / 200                    # midpoint Riemann sum of the path integral
ig = (x - base) * np.mean([grad_f(base + a * (x - base)) for a in alphas], axis=0)
print("IG completeness: sum(IG) = {:.4f}  vs  f(x) - f(baseline) = {:.4f}".format(ig.sum(), f(x) - f(base)))

# ---------------- 3. LIME: weighted linear surrogate on random perturbations ----------------
n_samples, width = 2000, 0.75
masks = rng.integers(0, 2, (n_samples, M))               # 1 = keep x's value, 0 = replace with background mean
Z = np.where(masks == 1, x, base)
dist = np.sqrt(((1 - masks) ** 2).sum(1) / M)            # distance from x in the interpretable space
w = np.exp(-dist ** 2 / width ** 2)                      # locality kernel
A = np.c_[np.ones(n_samples), masks]
coef = np.linalg.solve(A.T @ (A * w[:, None]) + 1e-3 * np.eye(M + 1), A.T @ (w * f(Z)))
lime = coef[1:]

# ---------------- 4. Kernel SHAP: the same regression with the Shapley kernel recovers phi exactly ----------------
rows, targets, kw = [], [], []
for S, vS in vals.items():
    z = np.zeros(M)
    z[list(S)] = 1
    k = len(S)
    kw.append(1e6 if k in (0, M) else (M - 1) / (math.comb(M, k) * k * (M - k)))   # Shapley kernel
    rows.append(np.r_[1, z])
    targets.append(vS)
A, kw, targets = np.array(rows), np.array(kw), np.array(targets)
kshap = np.linalg.solve(A.T @ (A * kw[:, None]), A.T @ (kw * targets))[1:]

print(f"\n{'feature':<22}{'Shapley':>9}{'KernelSHAP':>12}{'grad*input':>12}{'IG':>9}{'LIME':>9}")
for i in np.argsort(-np.abs(phi)):
    print(f"{names[i]:<22}{phi[i]:9.4f}{kshap[i]:12.4f}{gxi[i]:12.4f}{ig[i]:9.4f}{lime[i]:9.4f}")
print(f"max |KernelSHAP - exact Shapley| = {np.abs(kshap - phi).max():.1e}")
rank = lambda a: np.argsort(np.argsort(-np.abs(a)))
print("Spearman rank correlation with exact Shapley:  IG {:.3f}   LIME {:.3f}   grad*input {:.3f}".format(
    *[np.corrcoef(rank(phi), rank(a))[0, 1] for a in (ig, lime, gxi)]))`;
const TAB_OUT = String.raw`training accuracy 0.984
explained example: label 0, logit f(x) = -7.9950, mean background logit = 0.1928

efficiency check: sum(phi) = -8.1877  vs  f(x) - E[f] = -8.1877
IG completeness: sum(IG) = -7.3530  vs  f(x) - f(baseline) = -7.3528

feature                 Shapley  KernelSHAP  grad*input       IG     LIME
worst radius           -12.9850    -12.9850     -5.0007 -22.4932 -12.8875
mean radius              3.8200      3.8200      3.1909   8.1625   4.5516
mean texture             3.4423      3.4423     -0.4742   5.7255   3.6186
worst concave points    -2.1024     -2.1024     -0.2326  -2.1704  -2.0194
mean smoothness         -0.6124     -0.6124      0.2017  -0.2512  -0.1451
mean concavity           0.2499      0.2499      2.6316   3.6738  -0.3335
max |KernelSHAP - exact Shapley| = 2.7e-07
Spearman rank correlation with exact Shapley:  IG 0.829   LIME 0.943   grad*input 0.657`;
const CNN = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)
X_tr = torch.tensor(X_tr, dtype=torch.float32).view(-1, 1, 8, 8)
X_te = torch.tensor(X_te, dtype=torch.float32).view(-1, 1, 8, 8)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)     # Grad-CAM target layer: 32 maps of 8x8
        self.fc = nn.Linear(32, 10)

    def forward(self, x, keep=False):
        a = F.relu(self.conv2(F.relu(self.conv1(x))))
        if keep:
            self.maps = a                                # keep the activations for Grad-CAM
            a.retain_grad()
        return self.fc(a.mean(dim=(2, 3)))               # global average pooling -> logits


torch.manual_seed(0)
model = CNN()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(60):
    for idx in torch.randperm(len(X_tr)).split(64):
        loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
model.eval()
with torch.no_grad():
    print(f"test accuracy {(model(X_te).argmax(1) == y_te).float().mean():.3f}")


def input_grad(m, x, c):
    x = x.clone().requires_grad_(True)
    m(x)[:, c].sum().backward()
    return x.grad


def saliency(m, x, c):                                   # Simonyan et al. (2014): |d logit_c / d x|
    return input_grad(m, x, c).abs()


def grad_x_input(m, x, c):
    return input_grad(m, x, c) * x


def integrated_gradients(m, x, c, steps=64):            # Sundararajan et al. (2017), baseline = black image
    base = torch.zeros_like(x)
    alphas = (torch.arange(steps) + 0.5) / steps
    path = base + alphas.view(-1, 1, 1, 1) * (x - base)  # (steps, 1, 8, 8)
    g = input_grad(m, path, c).mean(0, keepdim=True)
    return (x - base) * g


def grad_cam(m, x, c):                                   # Selvaraju et al. (2017)
    m.zero_grad()
    logit = m(x, keep=True)[0, c]
    logit.backward()
    weights = m.maps.grad.mean(dim=(2, 3), keepdim=True)  # alpha_k = spatial mean of d y_c / d A_k
    return F.relu((weights * m.maps).sum(1, keepdim=True)).detach()


def show(name, att):
    a = att[0, 0].numpy()
    a = a / (np.abs(a).max() + 1e-12)
    print(f"{name}:")
    for row in a:
        print("   " + " ".join("#" if v > 0.5 else "+" if v > 0.2 else "-" if v < -0.2 else "." for v in row))


x, c = X_te[:1], int(y_te[0])
print(f"\nexplaining test image 0 (label {c}); predicted {model(x).argmax(1).item()}")
show("input (ink > 0.5 '#', > 0.2 '+')", x)
ig = integrated_gradients(model, x, c)
with torch.no_grad():
    gap = (model(x)[0, c] - model(torch.zeros_like(x))[0, c]).item()
print(f"IG completeness: sum of attributions {ig.sum():.4f} vs f(x) - f(baseline) {gap:.4f}")
show("integrated gradients ('-' = negative evidence)", ig)
show("Grad-CAM (conv2, 8x8)", grad_cam(model, x, c))


# ---------------- faithfulness: delete the top-k pixels by attribution and watch the logit fall ----------------
def deletion_drop(method, k=8, n=200):
    drops = []
    for i in range(n):
        xi, ci = X_te[i:i + 1], int(y_te[i])
        att = method(model, xi, ci).flatten()
        xd = xi.clone().flatten()
        xd[att.argsort(descending=True)[:k]] = 0.0       # set the k most important pixels to 0 (black)
        with torch.no_grad():
            drops.append((model(xi)[0, ci] - model(xd.view(1, 1, 8, 8))[0, ci]).item())
    return np.mean(drops)


gen = torch.Generator().manual_seed(0)
methods = {"random pixels": lambda m, x, c: torch.rand(x.shape, generator=gen),
           "saliency |grad|": saliency, "gradient x input": grad_x_input,
           "integrated gradients": integrated_gradients,
           "Grad-CAM": grad_cam}
print("\nmean drop of the true-class logit after zeroing the top-8 pixels (200 test images):")
for name, mth in methods.items():
    print(f"   {name:<22}{deletion_drop(mth):7.3f}")


# ---------------- sanity check (Adebayo et al., 2018): randomise the weights ----------------
def rank_corr(a, b):
    """Spearman rank correlation of the absolute maps (ties get their average rank)."""
    r = spearmanr(a.abs().flatten().numpy(), b.abs().flatten().numpy())[0]
    return 0.0 if np.isnan(r) else r


torch.manual_seed(1)
random_model = CNN().eval()                              # same architecture, random (untrained) weights
print("\nrank correlation of maps: trained model vs randomly re-initialised model (mean over 50 images)")
print("(a method that explains the MODEL should change a lot -> low correlation)")
for name in ("saliency |grad|", "gradient x input", "integrated gradients", "Grad-CAM"):
    mth = methods[name]
    r = [rank_corr(mth(model, X_te[i:i + 1], int(y_te[i])), mth(random_model, X_te[i:i + 1], int(y_te[i]))) for i in range(50)]
    r_in = [rank_corr(mth(model, X_te[i:i + 1], int(y_te[i])), X_te[i:i + 1]) for i in range(50)]
    print(f"   {name:<22} vs random model {np.mean(r):6.3f}    vs the input image itself {np.mean(r_in):6.3f}")`;
const CNN_OUT = String.raw`test accuracy 0.964

explaining test image 0 (label 7); predicted 7
input (ink > 0.5 '#', > 0.2 '+'):
   . . . # # # . .
   . . # # + # . .
   . . # . . # + .
   . . . + # # + .
   . . . # # # # .
   . . . + # . . .
   . . . # # . . .
   . . . # + . . .
IG completeness: sum of attributions 12.5673 vs f(x) - f(baseline) 12.5418
integrated gradients ('-' = negative evidence):
   . . . # # + . .
   . . # # + + . .
   . . - . . - . .
   . . . . . # + .
   . . . # # # + .
   . . . . . . . .
   . . . - - . . .
   . . . - - . . .
Grad-CAM (conv2, 8x8):
   . . . . . . . .
   . . . . . . . .
   . . . . . . . .
   . . + . . . . .
   . . . . . + + .
   . . . . + # # +
   . . . + # # # +
   . . . . . . . .

mean drop of the true-class logit after zeroing the top-8 pixels (200 test images):
   random pixels           1.227
   saliency |grad|         2.064
   gradient x input        9.474
   integrated gradients   10.151
   Grad-CAM                2.391

rank correlation of maps: trained model vs randomly re-initialised model (mean over 50 images)
(a method that explains the MODEL should change a lot -> low correlation)
   saliency |grad|        vs random model  0.078    vs the input image itself  0.069
   gradient x input       vs random model  0.905    vs the input image itself  0.931
   integrated gradients   vs random model  0.906    vs the input image itself  0.940
   Grad-CAM               vs random model -0.023    vs the input image itself  0.105`;

const KERAS = String.raw`import tensorflow as tf


def integrated_gradients(model, x, target, steps=64):
    """x: (1, H, W, C) image; baseline = black image."""
    baseline = tf.zeros_like(x)
    alphas = (tf.range(steps, dtype=tf.float32) + 0.5) / steps
    path = baseline + alphas[:, None, None, None] * (x - baseline)
    with tf.GradientTape() as tape:
        tape.watch(path)
        logits = model(path)[:, target]
    grads = tape.gradient(logits, path)
    return (x - baseline) * tf.reduce_mean(grads, axis=0, keepdims=True)


def grad_cam(model, x, target, conv_layer_name):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        maps, logits = grad_model(x)
        score = logits[:, target]
    grads = tape.gradient(score, maps)                     # (1, h, w, K)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    return tf.nn.relu(tf.reduce_sum(weights * maps, axis=-1))`;

const lesson: Lesson = {
  id: "m19/interpretability",
  title: "Interpretability: Saliency, Grad-CAM, SHAP, LIME",
  summary:
    "Methods that explain individual predictions — gradient saliency, gradient × input, integrated gradients, Grad-CAM, Shapley values/SHAP and LIME — derived from their axioms, implemented from scratch (exact Shapley values by enumerating every coalition, a hand-written LIME surrogate, Kernel SHAP recovering the exact values) and on a CNN trained on digits, followed by faithfulness tests and the sanity checks that reveal when an explanation describes the input rather than the model.",
  minutes: 90,
  level: "advanced",
  prerequisites: ["m04/backpropagation", "m08/convolution", "m08/pooling-receptive-field", "m03/linear-logistic-regression"],
  next: ["m19/robustness-safety", "m19/reading-papers", "m18/mlops"],
  terms: ["interpretability", "saliency-map", "integrated-gradients", "completeness-axiom", "grad-cam", "shapley-value", "shap", "lime", "attribution", "sanity-check-saliency"],
  sections: {
    intro: [
      {
        t: "p",
        text: "A trained network is a function with millions of parameters; its accuracy tells us *how often* it is right, not *why*. **Interpretability** (or explainable AI, XAI) is the set of methods that make a model's behaviour understandable to people. This lesson covers **post-hoc local attribution**: for one input and one output, assign each input feature (pixel, word, tabular column) a score for how much it contributed. Such explanations are used to debug models (is the classifier looking at the object or at a watermark?), to satisfy users and regulators who need reasons for decisions, and to generate scientific hypotheses about what a model has learned.",
      },
      {
        t: "p",
        text: "The methods form families. **Gradient-based**: saliency maps (Simonyan, Vedaldi and Zisserman, 2014, \"Deep Inside Convolutional Networks\") use $\\partial f/\\partial x$; **gradient × input** multiplies by the input; **integrated gradients** (Sundararajan, Taly and Yan, 2017, \"Axiomatic Attribution for Deep Networks\") average gradients along a path from a baseline and satisfy a completeness axiom. **Grad-CAM** (Selvaraju et al., 2017, \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\") weights the feature maps of a convolutional layer by their pooled gradients. **Perturbation-based**: **SHAP** (Lundberg and Lee, 2017, \"A Unified Approach to Interpreting Model Predictions\") uses Shapley values from cooperative game theory; **LIME** (Ribeiro, Singh and Guestrin, 2016, \"Why Should I Trust You?\") fits a locally weighted linear surrogate model.",
      },
      {
        t: "p",
        text: "Explanations can be wrong. Adebayo et al. (2018, \"Sanity Checks for Saliency Maps\") showed that some popular maps barely change when the model's weights are randomised — they reflect the input image, not what the model learned. This lesson therefore treats evaluation (faithfulness tests and sanity checks) as part of every method, and runs those checks on real models.",
      },
      {
        t: "callout",
        kind: "key",
        text: "Integrated gradients: $\\mathrm{IG}_i(x) = (x_i - x'_i)\\int_0^1 \\frac{\\partial f(x' + \\alpha(x - x'))}{\\partial x_i}d\\alpha$, with $\\sum_i \\mathrm{IG}_i = f(x) - f(x')$. Shapley value: $\\phi_i = \\sum_{S\\subseteq N\\setminus\\{i\\}} \\frac{|S|!\\,(M - |S| - 1)!}{M!}\\big[v(S\\cup\\{i\\}) - v(S)\\big]$, with $\\sum_i\\phi_i = v(N) - v(\\varnothing)$.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Backpropagation](/learn/m04/backpropagation) — every gradient method is one backward pass to the input (or to a convolutional layer).",
          "[The convolution operation](/learn/m08/convolution) and [pooling](/learn/m08/pooling-receptive-field) — feature maps, channels and global average pooling, which Grad-CAM relies on.",
          "[Linear and logistic regression](/learn/m03/linear-logistic-regression) — LIME's surrogate is a weighted linear regression; for linear models most attribution methods agree exactly.",
          "[Probability](/learn/m02/probability) — expectations over a background distribution define what \"removing\" a feature means for SHAP.",
          "[Adversarial examples](/learn/m19/robustness-safety) — the same input gradient used for saliency is the one FGSM exploits.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Splitting a team bonus fairly",
        text: "A project team earns a bonus. How much did each person contribute? You could ask what happens if one person leaves — but the answer depends on who else is still there (two people may be redundant: removing either one changes nothing, removing both is a disaster). The Shapley value answers this by averaging each person's marginal contribution over every possible order in which the team could have been assembled. SHAP treats input features as team members and the prediction as the bonus.",
      },
      {
        t: "p",
        text: "**Gradients** answer a different, more local question: \"if I nudge this pixel slightly, how much does the score move?\" That is cheap (one backward pass) but can mislead: a feature can be very important yet have zero gradient because the model is saturated (the score would not change for a small nudge, only for removing it). **Integrated gradients** fixes this by accumulating the gradient along the whole path from a neutral baseline (e.g. a black image) to the input — measuring the effect of *bringing in* each feature rather than nudging it.",
      },
      {
        t: "p",
        text: "**LIME** asks the model many \"what if\" questions near one example — what if these features were absent? — and fits a simple linear model to the answers, weighting nearby variations more. **Grad-CAM** looks one level up: instead of pixels, it asks which regions of the last convolutional feature maps support the class, giving a coarse heat map of *where* the evidence is.",
      },
    ],
    mechanism: [
      { t: "h", text: "Gradient methods" },
      {
        t: "steps",
        items: [
          { title: "Choose the output", text: "Explain the **logit** of the class of interest (not the softmax probability, whose gradient also depends on all other classes and saturates near 1)." },
          { title: "Saliency", text: "Backpropagate $\\partial f_c/\\partial x$ to the input; the map is $|\\partial f_c/\\partial x_i|$ (per pixel; for colour images, the max over channels)." },
          { title: "Gradient × input", text: "$x_i\\,\\partial f_c/\\partial x_i$: signed, and exactly the contribution of each feature for a linear model $f = w^\\top x$." },
          { title: "Integrated gradients", text: "Pick a baseline $x'$ (black image, blurred image, mean input). Evaluate gradients at $m$ points $x' + \\alpha_j(x - x')$, average them, multiply by $(x - x')$. Check completeness: the attributions must sum to $f(x) - f(x')$; if not, increase $m$." },
        ],
      },
      { t: "h", text: "Grad-CAM" },
      {
        t: "steps",
        items: [
          { title: "Pick a convolutional layer", text: "Usually the last one: its feature maps $A^k \\in \\mathbb R^{u\\times v}$ ($k = 1..K$ channels) still have spatial layout but encode high-level features." },
          { title: "Backpropagate to it", text: "Compute $\\partial y^c/\\partial A^k_{ij}$ for the class score $y^c$." },
          { title: "Channel weights", text: "Global-average-pool the gradients: $\\alpha^c_k = \\frac{1}{uv}\\sum_{i,j}\\partial y^c/\\partial A^k_{ij}$ — how much the class cares about channel $k$ overall." },
          { title: "Weighted sum and ReLU", text: "$L^c = \\mathrm{ReLU}(\\sum_k \\alpha^c_k A^k)$: keep only regions with positive evidence. Upsample to the image size to overlay it." },
        ],
      },
      { t: "h", text: "SHAP (exact, by enumeration)" },
      {
        t: "steps",
        items: [
          { title: "Define \"absent\"", text: "A coalition $S$ of features is *present*; the others are replaced by values from a background dataset. The value of the coalition is the average prediction: $v(S) = \\mathbb E_{x^b}[f(x_S, x^b_{\\bar S})]$." },
          { title: "Evaluate every coalition", text: "$2^M$ subsets ($M$ features): 64 for $M = 6$. Exact enumeration is only feasible for small $M$; Kernel SHAP, sampling and model-specific algorithms (e.g. TreeSHAP for tree ensembles) approximate it for larger $M$." },
          { title: "Average marginal contributions", text: "For each feature, sum $v(S \\cup \\{i\\}) - v(S)$ over all $S$ not containing $i$, weighted by $|S|!\\,(M - |S| - 1)!/M!$." },
          { title: "Check efficiency", text: "$\\sum_i\\phi_i = v(N) - v(\\varnothing) = f(x) - \\mathbb E[f]$." },
        ],
      },
      { t: "h", text: "LIME" },
      {
        t: "steps",
        items: [
          { title: "Interpretable representation", text: "Binary vector $z \\in \\{0,1\\}^M$: feature (or super-pixel, or word) present or absent." },
          { title: "Sample perturbations", text: "Draw many random $z$, build the corresponding inputs (absent features replaced by a reference value) and query the model." },
          { title: "Weight by proximity", text: "$\\pi(z) = \\exp(-D(z)^2/\\sigma^2)$, with $D$ the distance from the original (all ones)." },
          { title: "Fit a sparse/regularised linear model", text: "Weighted ridge or lasso regression of the model outputs on $z$; its coefficients are the explanation." },
        ],
      },
    ],
    math: [
      { t: "h", text: "1. Saliency and gradient × input" },
      { t: "math", tex: String.raw`S_i(x) = \left|\frac{\partial f_c(x)}{\partial x_i}\right|,\qquad G_i(x) = x_i\,\frac{\partial f_c(x)}{\partial x_i}` },
      {
        t: "p",
        text: "$f_c$ is the logit of class $c$ and $x_i$ the $i$-th input feature. For a linear model $f_c = w^\\top x + b$, $G_i = w_i x_i$ is exactly feature $i$'s additive contribution, and $\\sum_i G_i = f_c(x) - b$. For non-linear models gradient × input is a first-order approximation and can be badly off.",
      },
      { t: "h", text: "2. Integrated gradients and its axioms" },
      { t: "math", tex: String.raw`\mathrm{IG}_i(x) = (x_i - x'_i)\int_0^1 \frac{\partial f\big(x' + \alpha(x - x')\big)}{\partial x_i}\,d\alpha\ \approx\ (x_i - x'_i)\,\frac1m\sum_{j=1}^{m}\frac{\partial f\big(x' + \tfrac{j - 1/2}{m}(x - x')\big)}{\partial x_i}` },
      {
        t: "p",
        text: "Sundararajan et al. motivate IG by axioms. **Completeness**: by the fundamental theorem of calculus for path integrals, $\\sum_i\\mathrm{IG}_i(x) = f(x) - f(x')$. **Sensitivity**: if $x$ and $x'$ differ in one feature and the outputs differ, that feature gets non-zero attribution (plain gradients fail this when the model saturates). **Implementation invariance**: two networks computing the same function get the same attributions (because only gradients of the function are used). The baseline $x'$ is a modelling choice: it defines \"absence\"; a black image means \"no ink\".",
      },
      {
        t: "p",
        text: "**Worked example.** $f(x_1, x_2) = x_1x_2$, input $x = (2, 3)$, baseline $x' = (0, 0)$. On the path $\\alpha x$: $\\partial f/\\partial x_1 = \\alpha x_2 = 3\\alpha$, so $\\mathrm{IG}_1 = 2\\int_0^1 3\\alpha\\,d\\alpha = 3$; likewise $\\mathrm{IG}_2 = 3\\int_0^1 2\\alpha\\,d\\alpha = 3$. Sum $= 6 = f(x) - f(x')$. Gradient × input gives $(2\\cdot3, 3\\cdot2) = (6, 6)$, summing to 12 — twice the actual change, because it linearises a product. Saliency gives $(|x_2|, |x_1|) = (3, 2)$: it ranks $x_1$ higher only because the *other* feature is larger.",
      },
      { t: "h", text: "3. Grad-CAM" },
      { t: "math", tex: String.raw`\alpha^c_k = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A^k_{ij}},\qquad L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\Big(\sum_{k}\alpha^c_k A^k\Big)` },
      {
        t: "p",
        text: "$A^k$ is the $k$-th feature map of the chosen layer, $Z = u\\cdot v$ its number of positions, $y^c$ the class score. **Worked example** with two 2×2 maps: $A^1 = \\begin{pmatrix}1&0\\\\2&1\\end{pmatrix}$, $A^2 = \\begin{pmatrix}0&3\\\\1&0\\end{pmatrix}$, gradients $\\partial y/\\partial A^1 = 0.5$ everywhere and $\\partial y/\\partial A^2 = \\begin{pmatrix}-1&0\\\\0&-1\\end{pmatrix}$. Then $\\alpha_1 = 0.5$, $\\alpha_2 = -0.5$, $0.5A^1 - 0.5A^2 = \\begin{pmatrix}0.5&-1.5\\\\0.5&0.5\\end{pmatrix}$ and after ReLU $\\begin{pmatrix}0.5&0\\\\0.5&0.5\\end{pmatrix}$: the top-right region, driven by a channel that argues *against* the class, is suppressed. For a network ending in global average pooling and one linear layer (as in our CNN), $\\alpha^c_k$ is proportional to the class weight $w^c_k$ and Grad-CAM reduces to the earlier CAM method (Zhou et al., 2016).",
      },
      { t: "h", text: "4. Shapley values" },
      { t: "math", tex: String.raw`\phi_i(v) = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!\,(M - |S| - 1)!}{M!}\Big[v\big(S\cup\{i\}\big) - v(S)\Big]` },
      {
        t: "p",
        text: "$N = \\{1, \\dots, M\\}$ is the set of features, $v(S)$ the value (expected model output) of coalition $S$. The weight is the fraction of the $M!$ orderings of the features in which exactly the members of $S$ come before $i$. Shapley (1953) proved this is the *unique* attribution satisfying **efficiency** ($\\sum_i\\phi_i = v(N) - v(\\varnothing)$), **symmetry** (interchangeable features get equal values), **dummy** (a feature that never changes $v$ gets 0) and **additivity** (values of a sum of games add). Lundberg and Lee showed that several earlier methods (LIME with particular settings, DeepLIFT, layer-wise relevance propagation) are special cases or approximations of these \"SHAP values\".",
      },
      {
        t: "p",
        text: "**Worked example** with two features: $v(\\varnothing) = 0$, $v(\\{1\\}) = 4$, $v(\\{2\\}) = 2$, $v(\\{1,2\\}) = 10$. Each ordering has probability ½. $\\phi_1 = \\tfrac12(4 - 0) + \\tfrac12(10 - 2) = 6$ and $\\phi_2 = \\tfrac12(2 - 0) + \\tfrac12(10 - 4) = 4$; $6 + 4 = 10 = v(N) - v(\\varnothing)$. The interaction (10 is more than 4 + 2) is split equally. Applied to $f = x_1x_2$ at $(2, 3)$ with the single baseline $(0, 0)$: $v(\\{1\\}) = f(2, 0) = 0$, $v(\\{2\\}) = 0$, $v(N) = 6$, so $\\phi = (3, 3)$ — the same as integrated gradients here.",
      },
      { t: "h", text: "5. LIME and Kernel SHAP" },
      { t: "math", tex: String.raw`\hat g = \arg\min_{g\in G}\ \sum_{z}\pi_x(z)\,\big(f(h_x(z)) - g(z)\big)^2 + \Omega(g),\qquad g(z) = \beta_0 + \sum_{i=1}^{M}\beta_i z_i` },
      {
        t: "p",
        text: "$z \\in \\{0,1\\}^M$ is the interpretable representation, $h_x(z)$ maps it back to an input, $\\pi_x$ is the locality kernel, $\\Omega$ a complexity penalty (ridge or a limit on non-zero coefficients). Lundberg and Lee proved that with the **Shapley kernel** $\\pi(z) = \\frac{M - 1}{\\binom{M}{|z|}\\,|z|\\,(M - |z|)}$ (and the constraints $g(\\mathbf 0) = v(\\varnothing)$, $g(\\mathbf 1) = f(x)$), no penalty, and $f(h_x(z))$ replaced by $v(S)$, the regression coefficients are exactly the Shapley values — **Kernel SHAP**. Our From-Scratch program confirms it: over all 64 coalitions, the weighted regression reproduces the enumerated Shapley values to $2.7\\times10^{-7}$.",
      },
      { t: "h", text: "6. Results on the tabular model" },
      {
        t: "p",
        text: "For a malignant tumour (logit −7.995, background mean +0.193), the exact Shapley values sum to −8.1877, exactly $f(x) - \\mathbb E[f]$. \"Worst radius\" dominates (−12.985), while \"mean radius\" gets +3.820 — a positive contribution although a large radius indicates malignancy. The two features are highly correlated, and the network has learned a large negative weight on one partly cancelled by a positive weight on the other; the interventional value function evaluates them on independent combinations that never occur in real data. This is a known limitation: with correlated features, attributions describe the *model*, including its arbitrary split of shared information, not causal effects in the world.",
      },
    ],
    visual: [
      {
        t: "p",
        text: "Grad-CAM explains a CNN through its feature maps, so it helps to see what feature maps are. The CNN pipeline simulator applies convolution, ReLU and pooling to an image and shows each map.",
      },
      { t: "sim", id: "cnn-pipeline", caption: "Feature maps of a two-layer convolutional pipeline: Grad-CAM weights maps like these by the class gradient." },
      {
        t: "list",
        items: [
          "Step through the pipeline and note that each feature map responds to one pattern (an edge orientation, a blob) at every location — it keeps spatial layout. Grad-CAM's channel weight $\\alpha_k$ says how much the class cares about the pattern of map $k$; the weighted sum says *where* those patterns occur.",
          "Observe how pooling reduces resolution: Grad-CAM maps are as coarse as the chosen layer (8×8 in our CNN, 7×7 for a ResNet's last block on a 224×224 image), which is why they are smooth blobs rather than pixel-precise maps.",
        ],
      },
      { t: "sim", id: "adversarial", caption: "The input gradient as an explanation and as an attack: the second panel shows sign(∇ₓL), the same gradient that saliency maps visualise." },
      {
        t: "p",
        text: "In the adversarial simulator, compare the gradient-sign panel of the **MLP** with that of the **adversarially trained MLP** on the same digit: the robust model's gradient tends to concentrate on and around the strokes, while the standard model's is more scattered. Tiny gradient-guided changes flip the standard model — a reminder that a gradient map shows local sensitivity, which need not match the features a person would consider important.",
      },
    ],
    scratch: [
      {
        t: "p",
        text: "A small neural network on six standardised breast-cancer features (scikit-learn's Wisconsin dataset), explained five ways: exact Shapley values by enumerating all $2^6 = 64$ coalitions, Kernel SHAP (weighted regression with the Shapley kernel), gradient × input, integrated gradients, and a hand-written LIME surrogate. Everything is NumPy; the axioms (efficiency, completeness) are checked numerically.",
      },
      { t: "code", lang: "python", title: "shap_lime_ig_numpy.py", code: TAB, output: TAB_OUT },
      {
        t: "list",
        items: [
          "**Efficiency** holds exactly (−8.1877 = −8.1877) and **completeness** of IG to $2\\times10^{-4}$ (the midpoint Riemann sum with 200 steps).",
          "Kernel SHAP equals exact Shapley to $2.7\\times10^{-7}$, confirming Lundberg and Lee's equivalence.",
          "IG and Shapley use different notions of absence (a single baseline, the background mean, vs averaging over 100 background samples), so their totals differ (−7.353 vs −8.188) and their values differ in size, though the ranking largely agrees (rank correlation 0.829).",
          "LIME, using the same \"replace with the mean\" perturbation, is closest to Shapley here (0.943) but is a random, kernel-width-dependent estimate; \"mean concavity\" even changes sign between methods. Gradient × input agrees least (0.657): it ignores the non-linearity between the background and the input.",
          "The positive attribution to \"mean radius\" for a malignant case is a consequence of strongly correlated features (see the math section) — an explanation of the model, which may not match intuition about the disease.",
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
                text: "A small CNN with global average pooling trained on scikit-learn's 8×8 digits, explained with saliency, gradient × input, integrated gradients and Grad-CAM. Then two evaluations: a **deletion test** (zero the 8 highest-attributed pixels and measure the drop of the class logit, compared with 8 random pixels) and the **model-randomisation sanity check** of Adebayo et al. (2018).",
              },
              { t: "code", lang: "python", title: "saliency_ig_gradcam.py", code: CNN, output: CNN_OUT },
              {
                t: "list",
                items: [
                  "IG on the 7 highlights the top bar and the diagonal stroke and gives negative evidence ('−') to the bottom of the stem and to two ink pixels in the third row (a 7 whose strokes extended there would look more like another digit); completeness holds to 0.03 with 64 steps.",
                  "Grad-CAM is coarse and here highlights the lower-right region where the diagonal stroke meets the stem — it localises evidence at the resolution of the conv2 maps (its upsampled version is what papers overlay on photos).",
                  "**Deletion test**: removing IG's top-8 pixels lowers the logit by 10.15, gradient × input by 9.47, Grad-CAM 2.39, saliency 2.06, random pixels 1.23. But note the confound: setting a pixel to 0 changes nothing if it is already 0, and methods that multiply by the input only select inked pixels. The deletion test rewards agreement with the deletion baseline as much as faithfulness — evaluation metrics need careful design too.",
                  "**Sanity check**: re-initialising the weights barely changes the absolute gradient × input and IG maps (rank correlation 0.905 and 0.906 with the trained model's maps), and those maps correlate 0.93–0.94 with the input image itself: with a black baseline on sparse digits, their magnitude is dominated by *where the ink is*, whatever the model. Saliency (0.078) and Grad-CAM (−0.023) change completely with the weights, so they at least depend on the model. This reproduces, at small scale, the central warning of Adebayo et al. — the signed IG map above still carries model information (its negative pixels), but its magnitude pattern alone would pass for an explanation of a random network.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "Integrated gradients and Grad-CAM with `tf.GradientTape`; Grad-CAM uses a helper model that outputs both the chosen convolutional layer and the logits." },
              { t: "code", lang: "python", title: "ig_gradcam_keras.py (not executed in CI)", code: KERAS, output: "Defines two functions; nothing is printed. integrated_gradients returns an attribution map with the shape of x, grad_cam a (1, h, w) heat map at the resolution of the chosen layer." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Method", "Needs", "Cost", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Saliency (Simonyan et al., 2014)", "gradients", "1 backward pass", "fast; model-dependent", "noisy; saturation; local only", "quick look at sensitivity"],
          ["Gradient × input", "gradients", "1 backward pass", "exact for linear models", "dominated by the input; fails sanity checks", "baseline method"],
          ["SmoothGrad (Smilkov et al., 2017)", "gradients", "n backward passes", "averages saliency over noisy copies: less noise", "extra hyper-parameters", "cleaner saliency maps"],
          ["Integrated gradients (Sundararajan et al., 2017)", "gradients, baseline", "m backward passes (20–300)", "completeness, sensitivity, implementation invariance", "baseline choice changes the answer", "images, text, tabular"],
          ["Grad-CAM (Selvaraju et al., 2017)", "a conv layer", "1 backward pass", "class-discriminative localisation; passes model randomisation", "coarse resolution; CNN-specific", "vision debugging"],
          ["SHAP (Lundberg & Lee, 2017)", "black-box access, background data", "$2^M$ exact; sampled otherwise", "unique axiomatic solution; consistent", "expensive; feature-independence assumption in the value function", "tabular models; TreeSHAP for tree ensembles"],
          ["LIME (Ribeiro et al., 2016)", "black-box access", "hundreds–thousands of queries", "model-agnostic; any data type via super-pixels/words", "unstable across runs; kernel width matters", "quick local explanations"],
          ["Attention weights", "an attention model", "free", "available at no cost", "not reliable as importance (attention is not explanation by default)", "exploratory analysis"],
          ["Mechanistic interpretability", "model internals", "high (manual analysis)", "explains circuits and algorithms (e.g. induction heads)", "labour-intensive, mostly small models", "research on Transformers"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Debugging \"clever Hans\" models** — attribution maps reveal shortcut features such as text tags, rulers in dermatology images or hospital-specific markers in X-rays that correlate with the label in the training data.",
          "**Credit and insurance decisions** — SHAP values for gradient-boosted tree models (TreeSHAP) are widely used to give per-decision reasons and to audit which features drive scores.",
          "**Medical imaging review** — Grad-CAM overlays let clinicians check whether a model's evidence lies in the relevant anatomy; they support, not replace, validation studies.",
          "**Scientific discovery** — attributions on models of genomic sequences or materials highlight candidate motifs or structures for experimental follow-up.",
          "**Model monitoring** — tracking the distribution of SHAP values over time detects changes in what drives predictions, complementing data-drift monitoring ([MLOps](/learn/m18/mlops)).",
          "**Regulation and documentation** — explanations and their limitations belong in model cards (see [Responsible AI](/learn/m19/robustness-safety)).",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Gradient methods are cheap and model-specific** but local; saturation and noise make raw saliency unreliable.",
          "**Integrated gradients has clean axioms**, but its answer depends on the baseline, and its magnitude can be dominated by the input (our sanity check).",
          "**Grad-CAM is class-discriminative and passes the randomisation test**, but is low-resolution and limited to convolutional feature maps.",
          "**Shapley values are the unique fair attribution** for a given value function — but the choice of value function (how absent features are filled in) is itself a modelling decision, exact computation is exponential, and correlated features produce attributions on unrealistic inputs.",
          "**LIME is model-agnostic and flexible**, but results vary with the random sample, the kernel width and the perturbation scheme.",
          "**Explanations are not causal.** They describe the model's computation, not the data-generating process, and can be manipulated: models can be built whose explanations look benign while their behaviour is not.",
          "**Human factors**: plausible-looking maps increase trust whether or not they are faithful; evaluation must test faithfulness, not just plausibility.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Explaining the softmax probability instead of the logit.** Near probability 1 the gradient vanishes and attributions shrink to noise. Fix: attribute the logit (or log-probability).",
          "**Not checking completeness for IG.** Too few steps give attributions that do not sum to $f(x) - f(x')$. Fix: verify the sum and increase the number of steps.",
          "**Using an arbitrary baseline without saying so.** A black baseline gives zero attribution to black pixels by construction. Fix: state the baseline, try several (blurred, uniform noise, mean image).",
          "**Trusting a map because it looks like the object.** Edge-detector-like maps can come from the input alone. Fix: run the model-randomisation and data-randomisation sanity checks (our gradient × input maps correlated 0.905 with a random model's).",
          "**Interpreting attributions causally.** \"Feature X raised the risk\" is a statement about the model. Fix: phrase explanations as model behaviour; use causal methods for causal questions.",
          "**Ignoring correlated features in SHAP/LIME.** Replacing one feature independently of a correlated one creates unrealistic inputs and splits shared credit arbitrarily (our +3.82 for mean radius). Fix: group correlated features, or use conditional value functions, and report the choice.",
          "**Reporting a single LIME run.** It is a random estimate. Fix: fix the seed, use more samples, report stability across runs.",
          "**Evaluating explanations with a confounded metric.** Deletion tests depend on the replacement value. Fix: use several deletion baselines, or retrain-based evaluations (e.g. remove-and-retrain).",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is a saliency map and how is it computed?", a: "A per-pixel map of $|\\partial f_c/\\partial x_i|$: how sensitive the class-$c$ logit is to each input pixel. One forward pass and one backward pass to the input; for colour images take the maximum (or sum) over channels." },
      { t: "qa", level: "beginner", q: "What problem does Grad-CAM solve compared with pixel gradients?", a: "It produces a class-discriminative, spatially coherent heat map by weighting the last convolutional layer's feature maps with their spatially averaged gradients and applying ReLU. It shows where the evidence for a class lies, at the layer's (coarse) resolution, and is much less noisy than raw pixel gradients." },
      { t: "qa", level: "beginner", q: "What does LIME do?", a: "It explains one prediction by sampling perturbations of the input (turning interpretable features on/off), querying the model, weighting samples by proximity to the original, and fitting a simple (sparse) linear model to the outputs; the surrogate's coefficients are the explanation." },
      { t: "qa", level: "intermediate", q: "State the completeness axiom of integrated gradients and show it on $f = x_1x_2$ at (2, 3) with baseline 0.", a: "Attributions sum to $f(x) - f(x')$. $\\mathrm{IG}_1 = 2\\int_0^1 3\\alpha\\,d\\alpha = 3$, $\\mathrm{IG}_2 = 3\\int_0^1 2\\alpha\\,d\\alpha = 3$, sum 6 = $f(2,3) - f(0,0)$. Gradient × input would give (6, 6), summing to 12." },
      { t: "qa", level: "intermediate", q: "Define the Shapley value and list its axioms.", a: "$\\phi_i = \\sum_{S\\subseteq N\\setminus\\{i\\}}\\frac{|S|!(M-|S|-1)!}{M!}[v(S\\cup\\{i\\}) - v(S)]$, the average marginal contribution over all feature orderings. It is the unique attribution with efficiency (sum = $v(N) - v(\\varnothing)$), symmetry, dummy and additivity." },
      { t: "qa", level: "intermediate", q: "Why can exact SHAP not be computed for an image, and what is done instead?", a: "Exact values need $2^M$ coalition values; with thousands of pixels that is impossible. Practitioners group pixels into super-pixels, use sampling-based estimators such as Kernel SHAP (weighted regression on sampled coalitions), gradient-based approximations (e.g. DeepSHAP/expected gradients) or model-specific exact algorithms (TreeSHAP for tree ensembles)." },
      { t: "qa", level: "intermediate", q: "Explain the connection between LIME and SHAP.", a: "Both fit an additive model over binary feature-presence variables. Lundberg and Lee showed that with the Shapley kernel as the weighting, no regularisation and the coalition value function, the weighted regression's coefficients are exactly the Shapley values (Kernel SHAP); LIME's heuristic kernel generally gives different values without the Shapley guarantees. Our program reproduces exact Shapley values to $2.7\\times10^{-7}$ this way." },
      { t: "qa", level: "advanced", q: "What are the sanity checks of Adebayo et al. (2018) and what did they find?", a: "Model-parameter randomisation (re-initialise the weights, progressively or all at once) and data randomisation (train on permuted labels): a faithful explanation should change substantially. Several methods, notably guided backpropagation and gradient × input, produced visually similar maps for trained and randomised models — behaving like edge detectors of the input. Plain gradients and Grad-CAM passed. In our run gradient × input and IG magnitudes correlated ≈ 0.9 with a random model's, saliency and Grad-CAM ≈ 0." },
      { t: "qa", level: "advanced", q: "How do correlated features affect SHAP explanations, and what can you do?", a: "Interventional SHAP fills absent features from the marginal background distribution, breaking correlations, so the model is evaluated on unrealistic combinations and credit between correlated features is split according to arbitrary model weights (e.g. a positive attribution for a risk factor). Options: group correlated features into one player, use conditional/observational value functions (with their own problems, e.g. crediting features the model does not use), and report the choice and its consequences." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which method's attributions are guaranteed to sum to $f(x) - f(x')$?", options: ["Saliency", "Gradient × input (non-linear model)", "Integrated gradients", "Grad-CAM"], answer: 2, explain: "Completeness is IG's defining axiom, a consequence of the path integral." },
          { kind: "numeric", q: "For $f(x_1, x_2) = x_1x_2$ at $x = (4, 1)$ with baseline 0, what is $\\mathrm{IG}_1$?", answer: 2, tolerance: 0.001, explain: "$\\mathrm{IG}_1 = 4\\int_0^1 \\alpha\\cdot1\\,d\\alpha = 2$; $\\mathrm{IG}_2 = 2$ as well, summing to $f = 4$." },
          { kind: "numeric", q: "Two-player game: $v(\\varnothing) = 0$, $v(\\{1\\}) = 3$, $v(\\{2\\}) = 1$, $v(\\{1,2\\}) = 6$. What is $\\phi_1$?", answer: 4, tolerance: 0.001, explain: "$\\tfrac12(3 - 0) + \\tfrac12(6 - 1) = 1.5 + 2.5 = 4$; $\\phi_2 = 2$; sum 6." },
          { kind: "numeric", q: "How many coalitions must be evaluated for exact Shapley values with 10 features?", answer: 1024, tolerance: 0.5, explain: "$2^{10} = 1024$." },
          { q: "In Grad-CAM, the weight $\\alpha_k^c$ of feature map $k$ is…", options: ["the maximum activation of map k", "the spatial average of $\\partial y^c/\\partial A^k$", "the softmax of the class scores", "the number of positive pixels"], answer: 1, explain: "Global average pooling of the gradients over the map's positions." },
          { q: "A saliency method gives nearly identical maps for a trained model and the same model with random weights. This means…", options: ["the method is very robust", "the maps mostly reflect the input, not what the model learned", "the model is underfitting", "the random model is accurate"], answer: 1, explain: "Model-randomisation sanity check (Adebayo et al., 2018)." },
          { q: "Why should you attribute the logit rather than the softmax probability?", options: ["The logit is always positive", "The probability saturates near 1, making gradients vanish, and it mixes in all other classes", "Softmax has no gradient", "Logits are easier to plot"], answer: 1, explain: "Saturation and coupling across classes distort probability gradients." },
          { kind: "numeric", q: "Grad-CAM: two channels with $\\alpha = (1, -2)$ and, at one position, activations $A^1 = 3$, $A^2 = 1$. What is the Grad-CAM value at that position?", answer: 1, tolerance: 0.001, explain: "$\\mathrm{ReLU}(1\\cdot3 - 2\\cdot1) = \\mathrm{ReLU}(1) = 1$." },
        ],
      },
      {
        t: "exercise",
        title: "Shapley values of a three-feature game by hand",
        prompt: "A model's coalition values are $v(\\varnothing) = 0$, $v(\\{1\\}) = 2$, $v(\\{2\\}) = 2$, $v(\\{3\\}) = 0$, $v(\\{1,2\\}) = 6$, $v(\\{1,3\\}) = 2$, $v(\\{2,3\\}) = 2$, $v(\\{1,2,3\\}) = 6$. Compute $\\phi_1, \\phi_2, \\phi_3$ and check efficiency. Which axiom explains $\\phi_3$?",
        hint: "Weights for $M = 3$: $|S| = 0$: 1/3, $|S| = 1$: 1/6, $|S| = 2$: 1/3.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "φ₁", tex: String.raw`\tfrac13(2 - 0) + \tfrac16(6 - 2) + \tfrac16(2 - 0) + \tfrac13(6 - 2) = 0.6667 + 0.6667 + 0.3333 + 1.3333 = 3` },
              { title: "φ₂ (symmetric to feature 1)", tex: String.raw`\tfrac13(2) + \tfrac16(6 - 2) + \tfrac16(2 - 0) + \tfrac13(6 - 2) = 3` },
              { title: "φ₃", tex: String.raw`\tfrac13(0 - 0) + \tfrac16(2 - 2) + \tfrac16(2 - 2) + \tfrac13(6 - 6) = 0` },
              { title: "efficiency", tex: String.raw`3 + 3 + 0 = 6 = v(\{1,2,3\}) - v(\varnothing)` },
            ],
          },
          { t: "p", text: "Feature 3 never changes the value of any coalition, so the **dummy** axiom gives it 0; features 1 and 2 are interchangeable, so **symmetry** gives them equal values." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: SmoothGrad and a baseline study for IG",
        prompt: "Extend `saliency_ig_gradcam.py`: (a) implement SmoothGrad — average the saliency over 50 copies of the input with Gaussian noise of standard deviation 0.15 added — and run the deletion test and the model-randomisation check on it; (b) compute IG with three baselines (black image, the mean training image, uniform noise averaged over 10 draws) for test image 0 and report how strongly the three maps correlate. What do you conclude about the baseline's influence?",
        hint: "For (a) build a batch `x + 0.15 * torch.randn(50, 1, 8, 8, generator=g)` and average `saliency` over it. For (b) pass `base` into a modified `integrated_gradients` and use `rank_corr`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "def smoothgrad(m, x, c, n=50, sigma=0.15):\n    g = torch.Generator().manual_seed(0)\n    noisy = x + sigma * torch.randn((n, 1, 8, 8), generator=g)\n    return input_grad(m, noisy, c).abs().mean(0, keepdim=True)\n\ndef ig_with(m, x, c, base, steps=64):\n    alphas = (torch.arange(steps) + 0.5) / steps\n    path = base + alphas.view(-1, 1, 1, 1) * (x - base)\n    return (x - base) * input_grad(m, path, c).mean(0, keepdim=True)\n\nmean_img = X_tr.mean(0, keepdim=True)\nx, c = X_te[:1], int(y_te[0])\nmaps = {\"black\": ig_with(model, x, c, torch.zeros_like(x)),\n        \"mean\": ig_with(model, x, c, mean_img),\n        \"noise\": sum(ig_with(model, x, c, torch.rand(x.shape, generator=torch.Generator().manual_seed(s))) for s in range(10)) / 10}\nfor a in maps:\n    for b in maps:\n        if a < b:\n            print(a, b, rank_corr(maps[a], maps[b]))",
          },
          { t: "p", text: "SmoothGrad usually lowers the visual noise of saliency while keeping its dependence on the model (low correlation with the random model). The IG maps for different baselines are typically only partly correlated: the baseline defines what \"absence\" means (no ink vs an average digit vs noise), so it changes the question being answered. Report the baseline with every IG explanation, and prefer baselines that are meaningful for the domain." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Saliency $|\\partial f_c/\\partial x|$ (sensitivity), gradient × input $x\\odot\\nabla f$ (exact only for linear models).",
          "Integrated gradients: $(x - x')\\odot\\int_0^1\\nabla f(x' + \\alpha(x - x'))d\\alpha$; completeness $\\sum\\mathrm{IG} = f(x) - f(x')$; depends on the baseline.",
          "Grad-CAM: $\\alpha_k^c = \\mathrm{mean}_{ij}\\,\\partial y^c/\\partial A^k_{ij}$, map $= \\mathrm{ReLU}(\\sum_k\\alpha_k^c A^k)$; coarse, class-discriminative.",
          "Shapley/SHAP: average marginal contribution over all orderings; unique under efficiency, symmetry, dummy, additivity; $2^M$ coalitions; the value function defines \"absence\".",
          "LIME: locally weighted linear surrogate on perturbations; with the Shapley kernel it becomes Kernel SHAP (exact here to $2.7\\times10^{-7}$).",
          "Evaluate explanations: completeness/efficiency checks, deletion tests (with care about the replacement value), and sanity checks — our gradient × input and IG magnitudes correlated ≈ 0.9 with a random model's maps; saliency and Grad-CAM did not.",
          "Explanations describe the model, not causality; correlated features and baselines strongly affect them.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Adversarial Examples, Robustness and AI Safety](/learn/m19/robustness-safety) — the same input gradients, used to attack; robust models tend to have more interpretable gradients.",
          "[Reading and Reproducing Papers](/learn/m19/reading-papers) — interpretability claims need the same fair baselines and sanity checks as any other result.",
          "[MLOps](/learn/m18/mlops) — monitoring attributions and data in production.",
        ],
      },
    ],
  },
};

export default lesson;
