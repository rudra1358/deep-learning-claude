import type { Lesson } from "@/lib/types";

const SCRATCH = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- data: 8x8 digits, pixels scaled to [0, 1] ----------------
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
K = 10                                     # number of classes


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def onehot(y):
    return np.eye(K)[y]


def input_grad(W, b, x, y):
    """Cross-entropy loss per example and dL/dx for softmax regression logits = x W + b."""
    p = softmax(x @ W + b)
    loss = -np.log(p[np.arange(len(y)), y])
    return loss, (p - onehot(y)) @ W.T            # chain rule: dL/dz = p - y,  dz/dx = W^T


def fgsm(W, b, x, y, eps):
    _, g = input_grad(W, b, x, y)
    return np.clip(x + eps * np.sign(g), 0.0, 1.0)


def pgd(W, b, x, y, eps, alpha, steps, rng=None):
    xa = x.copy()
    if rng is not None:                           # optional uniform random start inside the box
        xa = np.clip(x + rng.uniform(-eps, eps, x.shape), 0.0, 1.0)
    for _ in range(steps):
        _, g = input_grad(W, b, xa, y)
        xa = xa + alpha * np.sign(g)
        xa = np.clip(np.clip(xa, x - eps, x + eps), 0.0, 1.0)   # project: eps-box, then valid pixels
    return xa


def train(adv_eps=0.0, epochs=300, lr=0.5, l2=1e-4, seed=0):
    """Full-batch gradient descent; if adv_eps > 0 each step trains on FGSM examples of the current model."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (64, K))
    b = np.zeros(K)
    for _ in range(epochs):
        x = fgsm(W, b, X_tr, y_tr, adv_eps) if adv_eps > 0 else X_tr
        p = softmax(x @ W + b)
        d = (p - onehot(y_tr)) / len(x)           # gradient of the mean loss w.r.t. the logits
        W -= lr * (x.T @ d + l2 * W)
        b -= lr * d.sum(axis=0)
    return W, b


def acc(W, b, x, y):
    return np.mean((x @ W + b).argmax(axis=1) == y)


W, b = train()
print(f"clean test accuracy: {acc(W, b, X_te, y_te):.4f}")

# ---- 1. the analytic input gradient matches central finite differences ----
x0, y0 = X_te[:1], y_te[:1]
_, g = input_grad(W, b, x0, y0)
h = 1e-6
num = np.zeros(64)
for i in range(64):
    e = np.zeros((1, 64))
    e[0, i] = h
    num[i] = (input_grad(W, b, x0 + e, y0)[0][0] - input_grad(W, b, x0 - e, y0)[0][0]) / (2 * h)
print(f"max |analytic - numeric| gradient: {np.abs(g[0] - num).max():.2e}")

# ---- 2. one image: FGSM at eps = 0.15 ----
p_clean = softmax(x0 @ W + b)[0]
xa = fgsm(W, b, x0, y0, 0.15)
p_adv = softmax(xa @ W + b)[0]
print(f"test image 0: label {y0[0]}, clean prediction {p_clean.argmax()} (p = {p_clean.max():.3f}), "
      f"FGSM eps=0.15 prediction {p_adv.argmax()} (p = {p_adv.max():.3f}), "
      f"max pixel change {np.abs(xa - x0).max():.3f}")

# ---- 3. accuracy vs eps: FGSM, PGD, and PGD on an FGSM-adversarially-trained model ----
W_adv, b_adv = train(adv_eps=0.1)
print(f"adversarially trained model, clean test accuracy: {acc(W_adv, b_adv, X_te, y_te):.4f}")
print("  eps   FGSM    PGD-20   | adv-trained: FGSM   PGD-20")
for eps in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    alpha = 2.5 * eps / 20
    row = [acc(W, b, fgsm(W, b, X_te, y_te, eps), y_te), acc(W, b, pgd(W, b, X_te, y_te, eps, alpha, 20), y_te),
           acc(W_adv, b_adv, fgsm(W_adv, b_adv, X_te, y_te, eps), y_te),
           acc(W_adv, b_adv, pgd(W_adv, b_adv, X_te, y_te, eps, alpha, 20), y_te)]
    print(f"  {eps:4.2f}  {row[0]:.3f}   {row[1]:.3f}    |              {row[2]:.3f}   {row[3]:.3f}")

# ---- 4. random noise of the same size is almost harmless ----
rng = np.random.default_rng(0)
noise = np.clip(X_te + 0.15 * rng.choice([-1.0, 1.0], X_te.shape), 0, 1)
print(f"random +-0.15 sign noise: accuracy {acc(W, b, noise, y_te):.3f}   "
      f"vs FGSM eps=0.15: {acc(W, b, fgsm(W, b, X_te, y_te, 0.15), y_te):.3f}")

# ---- 5. why linear models are fragile: logit change under FGSM = eps * ||w_y - w_j||_1 ----
d = W[:, y0[0]] - W[:, p_adv.argmax()]
print(f"||w_true - w_adv||_1 = {np.abs(d).sum():.2f}; eps = 0.15 can shift that logit margin by up to {0.15 * np.abs(d).sum():.2f}")`;
const SCRATCH_OUT = String.raw`clean test accuracy: 0.9444
max |analytic - numeric| gradient: 1.96e-10
test image 0: label 7, clean prediction 7 (p = 0.817), FGSM eps=0.15 prediction 9 (p = 0.599), max pixel change 0.150
adversarially trained model, clean test accuracy: 0.9333
  eps   FGSM    PGD-20   | adv-trained: FGSM   PGD-20
  0.00  0.944   0.944    |              0.933   0.933
  0.05  0.864   0.858    |              0.886   0.883
  0.10  0.678   0.669    |              0.753   0.747
  0.15  0.439   0.394    |              0.564   0.517
  0.20  0.172   0.139    |              0.319   0.286
  0.30  0.003   0.003    |              0.008   0.003
random +-0.15 sign noise: accuracy 0.925   vs FGSM eps=0.15: 0.439
||w_true - w_adv||_1 = 28.85; eps = 0.15 can shift that logit margin by up to 4.33`;
const TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)
X_tr = torch.tensor(X_tr, dtype=torch.float32).view(-1, 1, 8, 8)
X_te = torch.tensor(X_te, dtype=torch.float32).view(-1, 1, 8, 8)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 16 x 4 x 4
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 32 x 2 x 2
        nn.Flatten(), nn.Linear(128, 10))


def pgd(model, x, y, eps, alpha, steps, random_start=True):
    """L-infinity PGD (Madry et al., 2018). Returns adversarial images inside [x-eps, x+eps] and [0, 1]."""
    x_adv = x.clone()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


def fgsm(model, x, y, eps):
    return pgd(model, x, y, eps, alpha=eps, steps=1, random_start=False)


def train(adv_eps=0.0, epochs=20):
    torch.manual_seed(0)
    model = make_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(X_tr)).split(64):
            x, yb = X_tr[idx], y_tr[idx]
            if adv_eps > 0:                       # min over weights of max over the eps-ball
                model.eval()
                x = pgd(model, x, yb, adv_eps, alpha=adv_eps / 4, steps=7)
                model.train()
            loss = F.cross_entropy(model(x), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(1) == y).float().mean().item()


std = train()
adv = train(adv_eps=0.1)
torch.manual_seed(1)
print("  eps  | standard CNN: FGSM   PGD-20 | PGD-trained CNN: FGSM   PGD-20")
for eps in [0.0, 0.05, 0.1, 0.15, 0.2]:
    r = []
    for m in (std, adv):
        r.append(accuracy(m, fgsm(m, X_te, y_te, eps), y_te) if eps else accuracy(m, X_te, y_te))
        r.append(accuracy(m, pgd(m, X_te, y_te, eps, 2.5 * eps / 20, 20), y_te) if eps else r[-1])
    print(f"  {eps:4.2f} |              {r[0]:.3f}   {r[1]:.3f}  |                 {r[2]:.3f}   {r[3]:.3f}")

# sanity checks for a robustness claim (Carlini et al., 2019; Athalye et al., 2018):
# a stronger attack must never do better for the model; accuracy must reach ~0 for a huge eps.
eps = 0.1
for steps, restarts in [(1, 1), (20, 1), (100, 1), (100, 5)]:
    correct = torch.ones(len(X_te), dtype=torch.bool)
    for _ in range(restarts):                     # an example counts as robust only if it survives every restart
        x_adv = pgd(adv, X_te, y_te, eps, 2.5 * eps / steps if steps > 1 else eps, steps, random_start=steps > 1)
        with torch.no_grad():
            correct &= adv(x_adv).argmax(1) == y_te
    print(f"PGD-trained CNN, eps=0.1, {steps:3d} steps x {restarts} restart(s): robust accuracy {correct.float().mean():.3f}")
print(f"PGD-trained CNN, eps=0.5 (can paint any image): {accuracy(adv, pgd(adv, X_te, y_te, 0.5, 0.05, 50), y_te):.3f}")`;
const TORCH_OUT = String.raw`eps  | standard CNN: FGSM   PGD-20 | PGD-trained CNN: FGSM   PGD-20
  0.00 |              0.975   0.975  |                 0.975   0.975
  0.05 |              0.886   0.864  |                 0.942   0.939
  0.10 |              0.592   0.533  |                 0.853   0.808
  0.15 |              0.275   0.156  |                 0.653   0.511
  0.20 |              0.067   0.014  |                 0.392   0.206
PGD-trained CNN, eps=0.1,   1 steps x 1 restart(s): robust accuracy 0.853
PGD-trained CNN, eps=0.1,  20 steps x 1 restart(s): robust accuracy 0.817
PGD-trained CNN, eps=0.1, 100 steps x 1 restart(s): robust accuracy 0.806
PGD-trained CNN, eps=0.1, 100 steps x 5 restart(s): robust accuracy 0.800
PGD-trained CNN, eps=0.5 (can paint any image): 0.000`;
const SHIFT = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ======================= Part A: distribution shift =======================
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)

rng = np.random.default_rng(0)
imgs = X_te.reshape(-1, 8, 8)
shifts = {
    "in-distribution test set": X_te,
    "Gaussian noise, sigma=0.2": np.clip(X_te + rng.normal(0, 0.2, X_te.shape), 0, 1),
    "contrast halved": X_te * 0.5,
    "shifted right by 1 pixel": np.roll(imgs, 1, axis=2).reshape(len(X_te), 64),
    "shifted right by 2 pixels": np.roll(imgs, 2, axis=2).reshape(len(X_te), 64),
}
print("condition                     accuracy   mean confidence")
for name, xs in shifts.items():
    p = clf.predict_proba(xs)
    print(f"{name:<29} {np.mean(p.argmax(1) == y_te):.3f}      {p.max(1).mean():.3f}")

# ======================= Part B: group fairness metrics =======================
# Synthetic screening task: a binary outcome y, one informative score, two groups whose
# base rates differ (40% vs 20% positives). The model never sees the group attribute.
n = 20000
group = rng.integers(0, 2, n)                         # 0 = group A, 1 = group B
y_true = (rng.random(n) < np.where(group == 0, 0.4, 0.2)).astype(int)
score = rng.normal(1.2 * y_true, 1.0)                 # same score distribution given y in both groups
model = LogisticRegression().fit(score[:, None], y_true)
prob = model.predict_proba(score[:, None])[:, 1]


def report(pred, label):
    rows = []
    for g in (0, 1):
        m = group == g
        sel = pred[m].mean()                                     # selection rate  P(yhat=1 | g)
        tpr = pred[m & (y_true == 1)].mean()                     # true positive rate
        fpr = pred[m & (y_true == 0)].mean()                     # false positive rate
        ppv = y_true[m & (pred == 1)].mean()                     # precision P(y=1 | yhat=1, g)
        rows.append((sel, tpr, fpr, ppv))
    (sa, ta, fa, pa), (sb, tb, fb, pb) = rows
    acc = np.mean(pred == y_true)
    print(f"{label}\n   group A: selection {sa:.3f}  TPR {ta:.3f}  FPR {fa:.3f}  precision {pa:.3f}"
          f"\n   group B: selection {sb:.3f}  TPR {tb:.3f}  FPR {fb:.3f}  precision {pb:.3f}"
          f"\n   demographic parity diff {sa - sb:+.3f}   equal opportunity (TPR) diff {ta - tb:+.3f}"
          f"   FPR diff {fa - fb:+.3f}   accuracy {acc:.3f}")


report((prob > 0.5).astype(int), "one threshold 0.5 for everyone:")
# Equalising selection rates requires group-specific thresholds (here: select the top 30% in each group).
thr = {g: np.quantile(prob[group == g], 0.7) for g in (0, 1)}
report((prob > np.where(group == 0, thr[0], thr[1])).astype(int), "group thresholds giving equal selection rates (30% each):")`;
const SHIFT_OUT = String.raw`condition                     accuracy   mean confidence
in-distribution test set      0.967      0.903
Gaussian noise, sigma=0.2     0.933      0.844
contrast halved               0.875      0.639
shifted right by 1 pixel      0.397      0.690
shifted right by 2 pixels     0.058      0.667
one threshold 0.5 for everyone:
   group A: selection 0.242  TPR 0.462  FPR 0.095  precision 0.765
   group B: selection 0.167  TPR 0.463  FPR 0.095  precision 0.542
   demographic parity diff +0.076   equal opportunity (TPR) diff -0.001   FPR diff +0.000   accuracy 0.773
group thresholds giving equal selection rates (30% each):
   group A: selection 0.300  TPR 0.550  FPR 0.132  precision 0.736
   group B: selection 0.300  TPR 0.664  FPR 0.212  precision 0.432
   demographic parity diff +0.000   equal opportunity (TPR) diff -0.114   FPR diff -0.080   accuracy 0.752`;

const KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import load_digits

(X, y) = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 8, 8, 1).astype("float32")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(), tf.keras.layers.Dense(10)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(3e-3)


def pgd(x, y, eps=0.1, alpha=0.025, steps=7):
    x_adv = tf.clip_by_value(x + tf.random.uniform(tf.shape(x), -eps, eps), 0.0, 1.0)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(y, model(x_adv, training=False))
        g = tape.gradient(loss, x_adv)
        x_adv = tf.clip_by_value(tf.clip_by_value(x_adv + alpha * tf.sign(g), x - eps, x + eps), 0.0, 1.0)
    return x_adv


ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(2000, seed=0).batch(64)
for epoch in range(20):
    for xb, yb in ds:
        xb_adv = pgd(xb, yb)                       # inner maximisation
        with tf.GradientTape() as tape:            # outer minimisation on the adversarial batch
            loss = loss_fn(yb, model(xb_adv, training=True))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
    print(epoch, float(loss))`;

const lesson: Lesson = {
  id: "m19/robustness-safety",
  title: "Adversarial Examples, Robustness and AI Safety",
  summary:
    "Why tiny, well-aimed changes to an input can fool an accurate network; FGSM and PGD derived and coded; adversarial training as a min–max problem; how to test robustness honestly (accuracy-vs-ε curves, adaptive attacks, sanity checks); distribution shift; and practical responsible AI — fairness metrics on real numbers, privacy, model cards, and the alignment problem in plain words.",
  minutes: 95,
  level: "advanced",
  prerequisites: ["m04/backpropagation", "m05/classification-losses", "m08/convolution", "m17/classification-metrics"],
  next: ["m19/interpretability", "m18/mlops", "m20/alignment"],
  terms: ["adversarial-example", "fgsm", "pgd", "adversarial-training", "threat-model", "gradient-masking", "distribution-shift", "demographic-parity", "equal-opportunity", "model-card"],
  sections: {
    intro: [
      {
        t: "p",
        text: "An **adversarial example** is an input changed so slightly that a person often cannot see the difference, yet the model now makes a confident mistake. Szegedy et al. first reported them for deep image classifiers (\"Intriguing properties of neural networks\", 2014), and the same changed images often fooled other networks too.",
      },
      {
        t: "list",
        items: [
          "**Goodfellow, Shlens and Szegedy (2015)** argued the cause is not strange non-linearity but the *opposite*: models act too linearly in high dimensions. They introduced the one-step **Fast Gradient Sign Method (FGSM)**.",
          "**Madry et al. (2018)** framed robustness as a min–max problem. Training on examples from the many-step **Projected Gradient Descent (PGD)** attack — **adversarial training** — is still the most reliable defence in practice.",
        ],
      },
      {
        t: "p",
        text: "Why care? **Security**: anyone who picks inputs (spam, malware, fraud, objects in front of a camera) can search for ones that fool a model. **Reliability**: models also fail under **distribution shift** (real data that differ from training data), often while staying confident; adversarial examples are the worst case. **Understanding**: tiny decision-flipping changes show networks use features humans don't, linking robustness to [interpretability](/learn/m19/interpretability).",
      },
      {
        t: "p",
        text: "The second half covers **responsible AI** as engineering: fairness metrics (do errors fall unequally on groups of people?), training-data privacy, model cards (documentation), and the **alignment problem** — making a system pursue what its designers meant, not a stand-in for it.",
      },
      {
        t: "callout",
        kind: "key",
        text: "FGSM: $x' = \\mathrm{clip}_{[0,1]}\\big(x + \\epsilon\\,\\mathrm{sign}(\\nabla_x L(f(x), y))\\big)$. PGD repeats a smaller step $\\alpha$ and pulls the result back into the $\\epsilon$-ball after each step. Adversarial training solves $\\min_\\theta \\mathbb E_{(x,y)}\\big[\\max_{\\|\\delta\\|_\\infty \\le \\epsilon} L(f_\\theta(x+\\delta), y)\\big]$.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Backpropagation](/learn/m04/backpropagation) — attacks need the gradient of the loss with respect to the *input*, which the same backward pass gives you.",
          "[Classification losses](/learn/m05/classification-losses) — softmax cross-entropy and its gradient $p - \\mathrm{onehot}(y)$ with respect to the logits.",
          "[Norms and distances](/learn/m02/norms-spaces) — the $\\ell_\\infty$, $\\ell_2$ and $\\ell_1$ norms measure how big a change may be. The fact $\\max_{\\|\\delta\\|_\\infty\\le\\epsilon} g^\\top\\delta = \\epsilon\\|g\\|_1$ is the heart of FGSM.",
          "[The convolution operation](/learn/m08/convolution) — the framework example attacks a small CNN.",
          "[Classification metrics](/learn/m17/classification-metrics) — true/false positive rates, precision and selection rates are what fairness metrics are built from.",
          "[Regularization](/learn/m07/regularization) — adversarial training can be seen as a data-dependent regulariser.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Many small pushes add up",
        text: "Picture 64 judges voting. Whispering to one judge changes nothing. Whisper to *all 64*, each in the direction that hurts the current verdict, and the small nudges add up until the verdict flips — though no judge moved far. An image classifier is a committee of pixels. FGSM finds, for each pixel, whether brighter or darker argues against the correct class, and moves every pixel a tiny amount ε that way. With $d$ pixels the total effect on a linear score grows like $\\epsilon\\cdot d$, so in high dimensions an invisible per-pixel change can swing the output a lot.",
      },
      {
        t: "p",
        text: "Random noise of the same size does almost nothing: random pushes mostly cancel, so their effect grows only like $\\epsilon\\sqrt d$, with no consistent sign. In the From-Scratch program, random ±0.15 noise leaves digit accuracy at 92.5%, while FGSM with the same per-pixel budget drops it to 43.9%. The danger is not noise; it is *aligned* noise, found with the model's own gradient.",
      },
      {
        t: "p",
        text: "The natural fix is adversarial training: keep showing the model the worst inputs near each training example and teach it to get those right. It learns features small changes can't move — at some cost in clean accuracy and compute.",
      },
    ],
    mechanism: [
      { t: "h", text: "Threat model first" },
      {
        t: "p",
        text: "A robustness claim means nothing without a **threat model**: what the attacker knows and may change. In the usual **white-box $\\ell_p$ threat model**, the attacker knows the weights and may add any $\\delta$ with $\\|\\delta\\|_p \\le \\epsilon$, as long as the input stays valid (pixels in [0, 1]). A **black-box** attacker can only query the model, or attack a stand-in model and hope the result transfers. Other threat models cover patches, rotations, physical objects, poisoned training data or stolen models.",
      },
      { t: "h", text: "FGSM step by step" },
      {
        t: "steps",
        items: [
          { title: "Forward pass", text: "Compute logits $f(x)$, probabilities $p = \\mathrm{softmax}(f(x))$ and the loss $L = -\\log p_y$ for the true label $y$." },
          { title: "Backward pass to the input", text: "Backpropagate as usual, but keep going past the first layer to get $g = \\nabla_x L$: one number per pixel saying how fast the loss grows if that pixel gets brighter." },
          { title: "Take the sign", text: "$\\mathrm{sign}(g_i) \\in \\{-1, 0, +1\\}$. We throw away the gradient's size and keep only each pixel's direction. That is the steepest uphill direction under an $\\ell_\\infty$ budget." },
          { title: "Step and clip", text: "Move every pixel by exactly ε in that direction, then clip to the valid range.", tex: String.raw`x' = \mathrm{clip}_{[0,1]}\big(x + \epsilon\,\mathrm{sign}(g)\big)` },
          { title: "Evaluate", text: "If $\\arg\\max f(x') \\ne y$, the attack worked. Total cost: one forward and one backward pass." },
        ],
      },
      { t: "h", text: "PGD step by step" },
      {
        t: "steps",
        items: [
          { title: "Start", text: "$x_0 = x$, or (as in Madry et al.) a random point in the ε-box around $x$. Random starts let restarts explore different local maxima." },
          { title: "Gradient step", text: "Compute $g_t = \\nabla_x L(f(x_t), y)$ and step $\\tilde x = x_t + \\alpha\\,\\mathrm{sign}(g_t)$, with a step size $\\alpha < \\epsilon$." },
          { title: "Project", text: "Clip each pixel to $[x_i - \\epsilon, x_i + \\epsilon]$ (the $\\ell_\\infty$ ball around the *original* image), then to [0, 1].", tex: String.raw`x_{t+1} = \mathrm{clip}_{[0,1]}\Big(\mathrm{clip}_{[x-\epsilon,\ x+\epsilon]}\big(x_t + \alpha\,\mathrm{sign}(\nabla_x L(f(x_t), y))\big)\Big)` },
          { title: "Repeat", text: "Usually 10–100 steps with $\\alpha \\approx 2.5\\epsilon/T$, so the point can travel across the whole box. Keep the point with the highest loss (or stop once the prediction flips)." },
          { title: "Restarts", text: "Run several random starts. An example counts as robust only if *every* restart fails." },
        ],
      },
      { t: "h", text: "Adversarial training" },
      {
        t: "steps",
        items: [
          { title: "Sample a mini-batch", text: "$\\{(x_n, y_n)\\}_{n=1}^B$." },
          { title: "Inner maximisation", text: "Replace each $x_n$ with a PGD adversarial example $x_n^{adv}$ against the *current* model. Put the model in eval mode so attack steps don't update BatchNorm statistics." },
          { title: "Outer minimisation", text: "Take a normal optimiser step on $\\frac1B\\sum_n L(f_\\theta(x_n^{adv}), y_n)$. Some variants mix clean and adversarial loss, or use TRADES (Zhang et al., 2019), which adds a KL term between predictions on clean and perturbed inputs." },
          { title: "Cost", text: "A $K$-step PGD inner loop makes training roughly $K + 1$ times as expensive. \"Fast\" adversarial training (Wong, Rice and Kolter, 2020) uses FGSM from a random start to cut this." },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        text: "Test with an attack at least as strong as the one used in training, plus attacks the defender didn't plan for. A defence that only beats FGSM, or that works by making gradients useless (**gradient masking**), is not robust. Athalye, Carlini and Wagner (2018, \"Obfuscated Gradients Give a False Sense of Security\") adapted their attacks and broke most ICLR 2018 defences that relied on obfuscated gradients.",
      },
    ],
    math: [
      { t: "h", text: "1. The attack as an optimisation problem" },
      {
        t: "p",
        text: "Take a classifier $f_\\theta$, a loss $L$ (cross-entropy), an input $x \\in [0,1]^d$ with label $y$, and a budget $\\epsilon$. The **untargeted** attack looks for the allowed change that makes the loss as large as possible:",
      },
      { t: "math", tex: String.raw`\delta^\star = \arg\max_{\|\delta\|_\infty \le \epsilon,\ x + \delta \in [0,1]^d} L\big(f_\theta(x + \delta), y\big)` },
      {
        t: "p",
        text: "Here $\\|\\delta\\|_\\infty = \\max_i |\\delta_i|$ is the largest change to any single pixel. A **targeted** attack toward class $t$ instead minimises $L(f_\\theta(x+\\delta), t)$. Other norms give other threat models: $\\ell_2$ limits the total energy of the change, $\\ell_0$ the number of changed pixels.",
      },
      { t: "h", text: "2. FGSM solves the linearised problem exactly" },
      {
        t: "p",
        text: "Approximate the loss by a straight line near $x$: $L(x + \\delta) \\approx L(x) + g^\\top\\delta$ with $g = \\nabla_x L(x)$. Over the box $|\\delta_i| \\le \\epsilon$ this splits into one tiny problem per pixel: push $\\delta_i$ to the end of its range that matches the sign of $g_i$:",
      },
      { t: "math", tex: String.raw`\max_{\|\delta\|_\infty\le\epsilon} g^\top\delta = \sum_{i=1}^d \max_{|\delta_i|\le\epsilon} g_i\delta_i = \epsilon\sum_{i=1}^d |g_i| = \epsilon\,\|g\|_1,\qquad \delta^\star = \epsilon\,\mathrm{sign}(g)`, caption: "The $\\ell_1$ norm shows up because it is the dual norm of $\\ell_\\infty$. With an $\\ell_2$ budget the same reasoning gives $\\delta^\\star = \\epsilon\\,g/\\|g\\|_2$ and a gain of $\\epsilon\\|g\\|_2$." },
      { t: "h", text: "3. The linearity explanation" },
      {
        t: "p",
        text: "For a linear score $s(x) = w^\\top x$, the change $\\epsilon\\,\\mathrm{sign}(w)$ moves the score by $\\epsilon\\|w\\|_1$. If the $d$ weights have average size $m$, that is $\\epsilon\\, m\\, d$ — it grows *linearly with dimension* while each pixel still moves only ε. A random sign vector moves it by a sum of $d$ terms with random signs, with standard deviation $\\epsilon\\|w\\|_2 \\approx \\epsilon\\, m\\sqrt d$.",
      },
      {
        t: "p",
        text: "Goodfellow et al. argued that ReLU networks, LSTMs and maxout networks act nearly linearly by design (to be easy to optimise), so they inherit this weakness. In our softmax regression, the difference between the weight vectors of the true class (7) and the adversarial class (9) has $\\ell_1$ norm 28.85. So ε = 0.15 can move that logit margin by up to $0.15 \\times 28.85 = 4.33$ — plenty to flip a prediction made with 82% confidence.",
      },
      { t: "h", text: "4. Gradient of cross-entropy with respect to the input" },
      {
        t: "p",
        text: "For softmax regression $z = W^\\top x + b$ ($W \\in \\mathbb R^{d\\times K}$), the chain rule gives $\\nabla_x L = W(p - e_y)$, where $e_y$ is the one-hot label. For an MLP $z = W_2^\\top \\mathrm{ReLU}(W_1^\\top x + b_1) + b_2$ it is $\\nabla_x L = W_1\\big(\\mathbb 1[h > 0] \\odot W_2(p - e_y)\\big)$, with $h$ the hidden pre-activations: the usual backward pass, carried one layer further. The program checks this against central finite differences: largest difference $1.96\\times10^{-10}$.",
      },
      { t: "h", text: "5. Worked example (binary logistic regression)" },
      {
        t: "p",
        text: "Weights $w = (1.5, -2.0, 0.5)$, bias $b = 0.1$, input $x = (0.4, 0.3, 0.8)$, label $y = 1$, budget $\\epsilon = 0.1$. Binary cross-entropy $L = -\\log\\sigma(z)$ has $\\nabla_x L = (\\sigma(z) - y)\\,w$.",
      },
      {
        t: "steps",
        items: [
          { title: "Clean logit and probability", tex: String.raw`z = 1.5(0.4) - 2.0(0.3) + 0.5(0.8) + 0.1 = 0.6 - 0.6 + 0.4 + 0.1 = 0.5,\quad \sigma(0.5) = 0.622459,\quad L = 0.474077` },
          { title: "Input gradient", tex: String.raw`\nabla_x L = (0.622459 - 1)\,w = -0.377541\,(1.5, -2.0, 0.5) = (-0.566311,\ 0.755081,\ -0.188770)` },
          { title: "FGSM perturbation", tex: String.raw`\delta = 0.1\cdot\mathrm{sign}(\nabla_x L) = (-0.1,\ +0.1,\ -0.1),\quad x' = (0.3,\ 0.4,\ 0.7)` },
          { title: "Attacked logit", tex: String.raw`z' = z - \epsilon\|w\|_1 = 0.5 - 0.1\times 4.0 = 0.1,\quad \sigma(0.1) = 0.524979,\quad L' = 0.644397` },
          { title: "Budget needed to flip", tex: String.raw`z - \epsilon\|w\|_1 < 0 \iff \epsilon > \frac{0.5}{4.0} = 0.125` },
        ],
      },
      {
        t: "p",
        text: "For a linear model FGSM is already the best attack (ignoring [0, 1] clipping). PGD with $\\alpha = 0.06$ reaches $x_1 = (0.34, 0.36, 0.74)$ with $z_1 = 0.26$, then $x_2 = (0.3, 0.4, 0.7)$ with $z_2 = 0.1$ — exactly the FGSM point, because the gradient's sign never changes. In non-linear networks the gradient direction changes along the way, so PGD finds stronger attacks. In the PyTorch program at ε = 0.2, the standard CNN keeps 6.7% accuracy against FGSM but only 1.4% against 20-step PGD.",
      },
      { t: "h", text: "6. Adversarial training as robust optimisation" },
      { t: "math", tex: String.raw`\min_\theta\ \rho(\theta),\qquad \rho(\theta) = \mathbb E_{(x,y)\sim\mathcal D}\Big[\max_{\|\delta\|_\infty\le\epsilon} L\big(f_\theta(x+\delta), y\big)\Big]` },
      {
        t: "p",
        text: "$\\rho(\\theta)$ is the **adversarial risk**: the average loss when an attacker picks the worst allowed change for every example. By Danskin's theorem, the gradient of the inner maximum with respect to $\\theta$ is (under regularity conditions) the loss gradient taken *at* the worst change $\\delta^\\star$. That justifies the recipe: find $\\delta^\\star$ roughly with PGD, then take a normal gradient step on $L(f_\\theta(x + \\delta^\\star), y)$. Madry et al. found that PGD from many random starts reaches loss maxima of very similar value, so it became the standard inner solver.",
      },
      { t: "h", text: "7. Certified robustness (brief)" },
      {
        t: "p",
        text: "Measured robustness is only an upper bound on true robustness: a better attack may exist. **Certified** methods *prove* that no $\\delta$ within the budget changes the prediction. For a linear binary classifier the certificate is exact: the prediction holds for all $\\|\\delta\\|_\\infty \\le \\epsilon$ iff $|z| > \\epsilon\\|w\\|_1$ (0.125 in the example).",
      },
      {
        t: "p",
        text: "For networks, interval bound propagation and linear relaxations bound the logits over the ball. **Randomized smoothing** (Cohen, Rosenfeld and Kolter, 2019) certifies an $\\ell_2$ radius for the smoothed classifier $g(x) = \\arg\\max_c P(f(x + \\eta) = c)$, $\\eta \\sim \\mathcal N(0, \\sigma^2 I)$ — the class the model picks most often under added Gaussian noise. Certified radii are still much smaller than empirical ones.",
      },
      { t: "h", text: "8. Fairness metrics" },
      {
        t: "p",
        text: "Let $\\hat Y$ be a yes/no decision, $Y$ the true outcome and $A$ a group attribute. Three common criteria:",
      },
      { t: "math", tex: String.raw`\underbrace{P(\hat Y = 1 \mid A = a) = P(\hat Y = 1 \mid A = b)}_{\text{demographic parity}}\qquad \underbrace{P(\hat Y = 1 \mid Y = 1, A = a) = P(\hat Y = 1 \mid Y = 1, A = b)}_{\text{equal opportunity (equal TPR)}}` },
      { t: "math", tex: String.raw`\underbrace{P(Y = 1 \mid \hat Y = 1, A = a) = P(Y = 1 \mid \hat Y = 1, A = b)}_{\text{predictive parity (equal precision)}}` },
      {
        t: "list",
        items: [
          "**Demographic parity**: both groups are selected at the same rate. **Equal opportunity**: truly positive people are found at the same rate (equal TPR). **Predictive parity**: a \"yes\" is right equally often (equal precision).",
          "**Equalized odds** (Hardt, Price and Srebro, 2016) asks for equal TPR *and* equal FPR.",
          "These criteria clash whenever base rates $P(Y = 1 \\mid A)$ differ. Kleinberg, Mullainathan and Raghavan (2016) and Chouldechova (2017) showed that a non-trivial classifier cannot in general satisfy calibration/predictive parity and equal error rates across such groups.",
        ],
      },
      {
        t: "p",
        text: "In our synthetic example (base rates 40% and 20%, a score equally informative in both groups), one threshold gives equal TPR (0.462 vs 0.463) and FPR (0.095 vs 0.095) but unequal precision (0.765 vs 0.542) and selection rates (0.242 vs 0.167). Forcing equal selection rates (30% each) breaks equal opportunity (TPR 0.550 vs 0.664). Picking a criterion is a decision about the application, not a technical detail.",
      },
    ],
    visual: [
      {
        t: "p",
        text: "The simulator attacks **three real classifiers trained with PyTorch** on scikit-learn's 8×8 digits: softmax regression (650 parameters), an MLP 64→32→10 (2,410 parameters), and the same MLP adversarially trained with PGD at ε = 0.1. Everything runs in your browser, checked against a NumPy reference.",
      },
      { t: "sim", id: "adversarial", caption: "Pick a digit, model, attack and budget ε; step through PGD and watch the probabilities and the decision change." },
      {
        t: "list",
        items: [
          "Start with **MLP**, **PGD**, ε = 0.1, 10 steps, α = 0.02 and press **Play**. The loss climbs, and the clay bar of a wrong class overtakes the accent bar of the true class. δ is a pattern, not noise: it adds ink where another digit has strokes and removes it from the true digit's strokes.",
          "Switch to **FGSM**: one step of exactly ε per pixel. For the MLP, PGD at the same ε usually ends with a higher loss. For the **Linear** model the two are nearly the same, as the math section predicts.",
          "Lower ε until your digit stops flipping, then try **MLP (adv. trained)** at ε = 0.1 — it often survives where the standard one fails. In the robustness curve the accent line (adversarially trained) stays far above the clay line (standard MLP): at ε = 0.1 with PGD, 76.7% vs 30.6% of the 360 test digits survive.",
          "Check the **clean** column: adversarial training cost 0.5 percentage points of clean accuracy here (96.7% vs 97.2%). Now slide ε to 0.3: every model fails, because the attacker can partly redraw the digit. Robustness always depends on the budget.",
          "Pixels **on the edge of the ε-box**: after about ε/α PGD steps nearly all have used their full budget; later steps only shuffle signs.",
        ],
      },
      { t: "sim", id: "gradient-flow", caption: "Related: the input gradient is the same backward pass that trains the network, carried one layer further. This simulator shows how gradient sizes travel through depth." },
    ],
    scratch: [
      {
        t: "p",
        text: "Softmax regression on the digits in NumPy: full-batch gradient-descent training, the input gradient checked against finite differences, FGSM and PGD, an accuracy-vs-ε table, adversarial training with FGSM examples, a random-noise comparison and the linearity calculation.",
      },
      { t: "code", lang: "python", title: "fgsm_pgd_numpy.py", code: SCRATCH, output: SCRATCH_OUT },
      {
        t: "list",
        items: [
          "The input gradient is $W(p - e_y)$ — the same formula as the weight gradient, with $x$ and $W$ swapping roles.",
          "For this linear model PGD is barely stronger than FGSM (0.669 vs 0.678 at ε = 0.1): with fixed weights, only the softmax probabilities and clipping change the gradient direction along the path.",
          "FGSM adversarial training raises PGD accuracy at ε = 0.15 from 39.4% to 51.7% and costs 1.1 points of clean accuracy (94.4% → 93.3%). A linear model has little capacity to be both accurate and robust.",
          "Random sign noise of size 0.15 costs only 1.9 points; FGSM of the same size costs 50.5. Direction matters, not size.",
        ],
      },
      {
        t: "p",
        text: "**Distribution shift and fairness.** The second program measures a digit classifier's accuracy and confidence under natural shifts (noise, contrast, a 1–2 pixel move). It then computes group fairness metrics for a made-up screening task with two groups whose base rates differ.",
      },
      { t: "code", lang: "python", title: "shift_and_fairness.py", code: SHIFT, output: SHIFT_OUT },
      {
        t: "list",
        items: [
          "Moving the image one pixel — nothing to a person — drops accuracy from 96.7% to 39.7%, and two pixels to 5.8%, yet mean confidence stays at 0.67–0.69. Confidence is no warning under shift. (A CNN with pooling copes better with shifts; this is the inductive-bias case for convolution.)",
          "Halving the contrast costs 9 points and lowers confidence much more: a linear model's logits scale with the input.",
          "The fairness numbers show the incompatibility result: one threshold gives equal TPR and FPR but unequal precision and selection rates; equal selection rates need group thresholds, which make TPR and FPR unequal.",
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
                text: "A small CNN on the digits, trained normally and with PGD adversarial training (ε = 0.1, 7 steps, random start). Both face FGSM and 20-step PGD across budgets, then the sanity checks every robustness evaluation needs.",
              },
              { t: "code", lang: "python", title: "pgd_adversarial_training.py", code: TORCH, output: TORCH_OUT },
              {
                t: "list",
                items: [
                  "Adversarial training cost no clean accuracy here (97.5% both) and raised PGD accuracy at ε = 0.1 from 53.3% to 80.8%. On large datasets a clean-accuracy drop is typical (Tsipras et al., 2019, \"Robustness May Be at Odds with Accuracy\").",
                  "The FGSM–PGD gap is bigger for the robust model (0.653 vs 0.511 at ε = 0.15). Reporting only FGSM would overstate its robustness.",
                  "Sanity check 1 — **more attack effort must not help the model**: 1 step 0.853 → 20 steps 0.817 → 100 steps 0.806 → 100 steps × 5 restarts 0.800. The numbers settle, which suggests (but does not prove) the attack is near its best; if accuracy kept falling with more steps, the evaluation would be unreliable. (The table's 20-step value, 0.808, used a different random start — showing how much a single-restart estimate varies.)",
                  "Sanity check 2 — **an unbounded attack must succeed**: at ε = 0.5 accuracy is 0.000. If not, gradients would be masked or the attack broken.",
                  "`model.eval()` during the inner PGD loop keeps BatchNorm/Dropout behaviour fixed (this CNN has neither, but the habit matters).",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "PGD adversarial training with `tf.GradientTape`. The inner tape watches the *input* to get $\\nabla_x L$; the outer tape computes weight gradients on the adversarial batch." },
              { t: "code", lang: "python", title: "pgd_training_keras.py (not executed in CI)", code: KERAS, output: "Prints the epoch number and the adversarial training loss of the last mini-batch for each of 20 epochs." },
            ],
          },
        ],
      },
    ],
    variations: [
      { t: "h", text: "Attacks" },
      {
        t: "table",
        head: ["Attack", "Idea", "Knowledge / cost", "Notes"],
        rows: [
          ["FGSM (Goodfellow et al., 2015)", "one step $\\epsilon\\,\\mathrm{sign}(\\nabla_x L)$", "white-box; 1 gradient", "fast; weak against non-linear or adversarially trained models"],
          ["PGD (Madry et al., 2018)", "repeated sign-gradient steps + projection, random restarts", "white-box; $T$ gradients per restart", "standard first-order benchmark attack"],
          ["Carlini–Wagner (2017)", "optimise $\\|\\delta\\|_2 + c\\cdot$margin loss on logits", "white-box; many iterations", "finds small $\\ell_2$ changes; broke defensive distillation"],
          ["DeepFool (Moosavi-Dezfooli et al., 2016)", "repeatedly step to the linearised decision boundary", "white-box", "estimates the smallest change that flips the label"],
          ["AutoAttack (Croce & Hein, 2020)", "a bundle of attacks with no settings to tune", "white-box + black-box part", "a standard for reliable robustness reporting"],
          ["Transfer / black-box", "attack a stand-in model, or estimate gradients from queries", "no weights needed", "adversarial examples often transfer between models"],
          ["Physical attacks", "printed patches or stickers optimised over viewpoints", "real world", "e.g. Kurakin et al. (2016) printed adversarial images; Eykholt et al. (2018) stop-sign stickers"],
        ],
      },
      { t: "h", text: "Defences" },
      {
        t: "table",
        head: ["Defence", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["PGD adversarial training", "most reliable measured robustness", "$(K{+}1)\\times$ training cost; clean accuracy drops on large data; covers only the trained threat model", "default when robustness is required"],
          ["TRADES (Zhang et al., 2019)", "a knob for the clean-vs-robust trade-off", "one more hyper-parameter", "robust benchmarks"],
          ["Fast (FGSM + random start) training", "close to normal training cost", "can suffer \"catastrophic overfitting\" to FGSM", "tight compute budgets"],
          ["Certified (IBP, randomized smoothing)", "provable guarantees", "small certified radii; extra inference cost (smoothing needs many samples)", "safety-critical claims"],
          ["Input preprocessing, gradient obfuscation", "cheap", "usually broken by adaptive attacks (Athalye et al., 2018)", "not recommended as the only defence"],
          ["Detecting adversarial inputs", "can flag attacks", "detectors can be attacked too", "monitoring, alongside other defences"],
        ],
      },
      { t: "h", text: "Beyond $\\ell_p$: distribution shift" },
      {
        t: "table",
        head: ["Shift", "What changes", "Example", "Mitigation"],
        rows: [
          ["Covariate shift", "$p(x)$ changes, $p(y\\mid x)$ does not", "new camera, lighting, hospital scanner", "augmentation, domain adaptation, collect target data"],
          ["Label shift", "$p(y)$ changes", "disease prevalence differs by region", "re-weight / re-calibrate priors"],
          ["Concept drift", "$p(y\\mid x)$ changes", "fraud patterns adapt to the detector", "monitoring and retraining ([MLOps](/learn/m18/mlops))"],
          ["Corruptions", "noise, blur, weather, compression", "ImageNet-C (Hendrycks & Dietterich, 2019)", "augmentation, robust pretraining"],
          ["Out-of-distribution inputs", "inputs from no training class", "a cat shown to a digit classifier", "OOD detection (max-softmax baseline, Hendrycks & Gimpel, 2017), abstaining"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Security-sensitive classifiers** — malware and spam detection, fraud scoring and content moderation face attackers who adapt, so robustness tests and adversarial training belong in their evaluation.",
          "**Autonomous systems and perception** — physical adversarial examples (printed patches, stickers on signs) are why vision in cars and robots gets robustness testing.",
          "**Robustness audits** — accuracy-vs-ε curves, corruption benchmarks and shift tests before releasing high-stakes models.",
          "**Model cards and datasheets** — short documents (Mitchell et al., 2019, \"Model Cards for Model Reporting\") giving a model's intended use, evaluation data, metrics split by relevant groups and conditions, and limitations; Gebru et al.'s \"Datasheets for Datasets\" do the same for training data. Many model hubs now publish such cards.",
          "**Fairness reviews** — lending, hiring and medical models are checked with per-group error rates. Several jurisdictions require this for high-risk uses.",
          "**Privacy-preserving training** — DP-SGD (Abadi et al., 2016) clips each example's gradient and adds noise, giving differential-privacy guarantees (Dwork et al., 2006). Membership-inference attacks (Shokri et al., 2017), which guess whether an example was in the training set, are used to audit leakage.",
          "**Aligning large models** — reinforcement learning from human feedback (Christiano et al., 2017) and its successors teach language models to follow instructions and avoid harmful outputs; see [Instruction Tuning, RLHF and DPO](/learn/m20/alignment).",
        ],
      },
      { t: "h", text: "AI safety and the alignment problem, in plain words" },
      {
        t: "p",
        text: "A learning system optimises the objective it is given, not the intention behind it. Push hard on a stand-in objective (a reward function, click-through rate, human ratings) and you can get behaviour nobody wanted: **specification gaming** or **reward hacking**, like a student who games the test instead of learning the subject.",
      },
      {
        t: "p",
        text: "Amodei et al. (2016, \"Concrete Problems in AI Safety\") listed concrete research problems: avoiding negative side effects, avoiding reward hacking, scalable oversight (supervising behaviour that is costly to check), safe exploration, and robustness to distributional shift. Adversarial examples are a small, measurable case of the same pattern: a model that excels on its training objective behaves badly on inputs the objective never covered.",
      },
      {
        t: "p",
        text: "So practical safety work is mostly evaluation and process: red-teaming (hunting for failure-causing inputs as an attacker would), staged rollout with monitoring, documented limitations, human oversight for high-stakes decisions, and the ability to roll back.",
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Adversarial training works — within its threat model.** It has survived adaptive attacks best, but robustness to $\\ell_\\infty$ at ε = 0.1 says little about rotations, patches, $\\ell_2$ attacks or natural shift.",
          "**Cost.** PGD training with $K$ steps costs about $K + 1$ times standard training.",
          "**Accuracy trade-off.** On large, complex datasets robust models usually lose clean accuracy and need more capacity and data. On our small digits the drop was 0.0–0.5 points.",
          "**Robust models have more readable gradients** — they tend to look like object features.",
          "**Evaluation is hard.** Measured robustness is only an upper bound; certified methods give guarantees, but only for small radii.",
          "**Fairness criteria conflict.** With different base rates you must choose which error rates to equalise. There is no metric-free meaning of \"fair\".",
          "**Privacy costs accuracy.** Differential privacy adds noise to training and usually lowers accuracy, more so for small datasets and rare groups.",
          "**Documentation is cheap and valuable** — but only if it states real limitations with numbers, not generic disclaimers.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Testing only with FGSM.** A model can resist this weak one-step attack and still fall to PGD (0.653 vs 0.511 for our CNN at ε = 0.15). Fix: PGD with many steps and restarts, plus AutoAttack-style bundles.",
          "**Not clipping to the valid range, or projecting around the current point.** The ε-ball is around the *original* image $x$, and pixels must stay in [0, 1] (or the normalised equivalent). Fix: `clip(x_adv, x - eps, x + eps)` then `clip(0, 1)`.",
          "**Using a pixel-space ε on normalised inputs.** After mean/std normalisation, ε = 8/255 in pixel space is no longer 8/255. Fix: put normalisation inside the model and attack raw pixels.",
          "**Mistaking gradient masking for robustness.** Non-differentiable preprocessing, randomness or saturated softmax make gradients useless, so gradient attacks fail though adversarial examples exist. Warning signs: FGSM beats PGD, black-box beats white-box, accuracy doesn't reach 0 at huge ε. Fix: adaptive attacks (BPDA, expectation over transformation) and sanity checks.",
          "**Leaving the model in training mode during the attack.** Dropout turns the target into a moving model, and BatchNorm updates its running statistics with adversarial inputs. Fix: `model.eval()` for the inner loop.",
          "**Reporting one number.** Robustness depends on ε, the norm and the attack. Fix: report accuracy-vs-ε curves, the attack settings and clean accuracy.",
          "**\"The model doesn't use the sensitive attribute, so it's fair.\"** Other features can stand in for it, and different base rates give different error profiles anyway. Fix: measure per-group metrics.",
          "**Reading high confidence as \"this input is normal\".** Our classifier was 67% confident on images it got 94% wrong. Fix: dedicated shift/OOD detection and calibration checks on shifted data.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is an adversarial example?", a: "An input changed by a small, deliberate perturbation — say, every pixel moved by at most ε — so a model that gets the original right gets it wrong, often confidently. The change is found by optimisation, usually with the loss gradient with respect to the input." },
      { t: "qa", level: "beginner", q: "Write down FGSM and explain each part.", a: "$x' = \\mathrm{clip}_{[0,1]}(x + \\epsilon\\,\\mathrm{sign}(\\nabla_x L(f(x), y)))$. $\\nabla_x L$ says how the loss changes as each pixel changes; its sign gives each pixel's direction that raises the loss; ε is the per-pixel budget ($\\ell_\\infty$ norm); clipping keeps a valid image. Cost: one forward and one backward pass." },
      { t: "qa", level: "beginner", q: "Why does random noise of the same size not fool the model?", a: "Random signs mostly cancel. The change in a linear score is a sum of $d$ terms with random signs, typically about $\\epsilon\\|w\\|_2 \\approx \\epsilon m\\sqrt d$, versus $\\epsilon\\|w\\|_1 \\approx \\epsilon m d$ for the aligned FGSM direction. On our digits, ±0.15 random noise cost 1.9 accuracy points; FGSM cost 50.5." },
      { t: "qa", level: "intermediate", q: "How does PGD differ from FGSM, and why is it stronger?", a: "PGD takes several smaller steps of size α, recomputing the gradient each time and projecting back into the ε-ball around the original input and the valid range, usually with random starts and restarts. In non-linear models the gradient direction changes along the path, so FGSM's single straight-line guess misses higher-loss points. For a linear model the two give the same result." },
      { t: "qa", level: "intermediate", q: "Explain adversarial training as an optimisation problem.", a: "$\\min_\\theta \\mathbb E[\\max_{\\|\\delta\\|\\le\\epsilon} L(f_\\theta(x+\\delta), y)]$. PGD approximates the inner max for each mini-batch. By Danskin's theorem, the gradient of the inner max is the loss gradient at the worst point, so we take an ordinary optimiser step on the adversarial examples. Cost ≈ (PGD steps + 1) × standard training." },
      { t: "qa", level: "intermediate", q: "What is gradient masking and how do you detect it?", a: "A defence that makes gradients useless (non-differentiable operations, randomness, saturation), so gradient attacks fail although the model is not robust. Signs: FGSM beats PGD, black-box or transfer attacks beat white-box, more iterations don't lower accuracy, an unbounded attack doesn't reach 0% accuracy. Remedies: adaptive attacks such as BPDA and expectation over transformation (Athalye et al., 2018)." },
      { t: "qa", level: "intermediate", q: "Name three group fairness criteria and explain why they can conflict.", a: "Demographic parity (equal selection rates), equal opportunity (equal TPR; equalized odds adds equal FPR), predictive parity (equal precision / calibration). When base rates differ between groups, a non-trivial classifier cannot satisfy calibration and equal error rates at once (Kleinberg et al., 2016; Chouldechova, 2017). In our example, equal TPR/FPR gave precision 0.765 vs 0.542, and equalising selection broke equal TPR." },
      { t: "qa", level: "advanced", q: "Why do adversarial examples transfer between models, and what does that imply?", a: "Models trained on the same data learn similar non-robust but predictive features, and adversarial directions often line up with them, so a change crafted on one model often fools another. So black-box attacks work without the weights (attack a stand-in), and hiding the weights is not a defence." },
      { t: "qa", level: "advanced", q: "How would you convince a sceptical reviewer that your defence is robust?", a: "State the threat model precisely. Attack with strong white-box attacks adapted to the defence (PGD with many steps and restarts, AutoAttack, attacks through any preprocessing), plus black-box and transfer attacks. Show accuracy-vs-ε curves that fall steadily with ε and reach ~0 for unbounded ε, and show results converge as attack effort grows. Release code and models; add certified bounds where possible (following Carlini et al. (2019, \"On Evaluating Adversarial Robustness\"))." },
      { t: "qa", level: "advanced", q: "What is randomized smoothing and what does it certify?", a: "Define $g(x) = \\arg\\max_c P_{\\eta\\sim\\mathcal N(0,\\sigma^2 I)}(f(x+\\eta) = c)$: the class $f$ picks most often under Gaussian noise. Cohen et al. (2019) proved that if the top class has probability $p_A$ and the runner-up at most $p_B$, then $g$ stays the same within an $\\ell_2$ radius $\\frac\\sigma2(\\Phi^{-1}(p_A) - \\Phi^{-1}(p_B))$. Probabilities are estimated by Monte Carlo with confidence bounds, so the certificate holds with high probability; each prediction costs many forward passes, and radii are modest." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "FGSM changes each pixel by…", options: ["a random amount in [−ε, ε]", "exactly ±ε (before clipping), following the sign of the input gradient", "ε times the gradient", "the amount that flips the label"], answer: 1, explain: "$\\delta = \\epsilon\\,\\mathrm{sign}(\\nabla_x L)$; the gradient's size is thrown away." },
          { kind: "numeric", q: "A linear binary classifier has $\\|w\\|_1 = 4$ and logit $z = 0.5$ on a correctly classified input. What is the smallest $\\ell_\\infty$ budget ε that can flip it (ignore clipping)?", answer: 0.125, tolerance: 0.001, explain: "The worst case lowers the logit by $\\epsilon\\|w\\|_1$; $0.5/4 = 0.125$." },
          { kind: "numeric", q: "For the same model, how much can FGSM with ε = 0.1 lower the logit?", answer: 0.4, tolerance: 0.001, explain: "$\\epsilon\\|w\\|_1 = 0.1 \\times 4 = 0.4$, from 0.5 to 0.1." },
          { q: "In PGD, the projection step clips each pixel to…", options: ["$[x_t - \\epsilon, x_t + \\epsilon]$ around the current point", "$[x - \\epsilon, x + \\epsilon]$ around the original input, and to the valid range", "$[0, \\epsilon]$", "nothing — PGD is unconstrained"], answer: 1, explain: "The budget is always measured from the original image $x$." },
          { q: "Which sign points to gradient masking rather than real robustness?", options: ["PGD is stronger than FGSM", "Accuracy falls as ε grows", "A black-box transfer attack succeeds more often than a white-box gradient attack", "Accuracy reaches 0% for very large ε"], answer: 2, explain: "White-box attacks know strictly more. If they do worse, the gradients carry no useful information." },
          { kind: "numeric", q: "Training uses 7-step PGD for the inner maximisation. Roughly how many forward+backward passes per mini-batch does adversarial training need, versus 1 for standard training?", answer: 8, tolerance: 0.01, explain: "7 for the attack plus 1 for the weight update." },
          { q: "Group A has a 40% positive rate and group B 20%. A classifier has equal TPR and FPR in both groups. What about precision?", options: ["It is equal too", "It will generally differ: higher in the group with the higher base rate", "It is zero in group B", "Nothing can be said"], answer: 1, explain: "Precision depends on base rates; our simulation gave 0.765 vs 0.542 with equal TPR and FPR." },
          { q: "What is the main purpose of a model card?", options: ["To store the model weights", "To document intended use, evaluation data, metrics (including by subgroup and condition) and limitations", "To certify adversarial robustness", "To encrypt the model"], answer: 1, explain: "Mitchell et al. (2019) proposed model cards as short, structured reports so models are described openly." },
        ],
      },
      {
        t: "exercise",
        title: "FGSM on a logistic-regression model by hand",
        prompt: "A logistic model has $w = (2, -1, 0.5, -0.5)$, $b = -0.2$ and input $x = (0.5, 0.2, 0.6, 0.4)$ with label $y = 1$. (a) Compute $z$, $p = \\sigma(z)$ and $\\nabla_x L$ for binary cross-entropy. (b) Apply FGSM with ε = 0.1 (ignore clipping) and find the new logit and probability. (c) What ε flips the prediction?",
        hint: "$\\nabla_x L = (p - y)w$, and for a linear model FGSM lowers the logit by exactly $\\epsilon\\|w\\|_1$.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) logit", tex: String.raw`z = 2(0.5) - 1(0.2) + 0.5(0.6) - 0.5(0.4) - 0.2 = 1.0 - 0.2 + 0.3 - 0.2 - 0.2 = 0.7,\quad p = \sigma(0.7) = 0.668188` },
              { title: "(a) gradient", tex: String.raw`\nabla_x L = (0.668188 - 1)\,w = -0.331812\,(2, -1, 0.5, -0.5) = (-0.663624,\ 0.331812,\ -0.165906,\ 0.165906)` },
              { title: "(b) FGSM", tex: String.raw`\delta = 0.1\,\mathrm{sign}(\nabla_x L) = (-0.1, 0.1, -0.1, 0.1),\quad z' = 0.7 - 0.1\cdot\|w\|_1 = 0.7 - 0.1\cdot 4 = 0.3,\quad \sigma(0.3) = 0.574443` },
              { title: "(c) flip", tex: String.raw`0.7 - 4\epsilon < 0 \iff \epsilon > 0.175` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a targeted PGD attack and a robustness curve",
        prompt: "Change `pgd_adversarial_training.py` so `pgd` takes an optional target class $t$ and *minimises* the cross-entropy toward $t$ (a targeted attack). For the standard CNN, report the targeted success rate (prediction equals $t$) at ε ∈ {0.05, 0.1, 0.2} with target $t = (y + 1) \\bmod 10$. Should the targeted success rate be higher or lower than the untargeted error rate at the same ε, and why?",
        hint: "Targeted: step in the direction $-\\mathrm{sign}(\\nabla_x L(f(x_t), t))$. Success = `model(x_adv).argmax(1) == t`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "def pgd(model, x, y, eps, alpha, steps, random_start=True, target=None):\n    x_adv = x.clone()\n    if random_start:\n        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)\n    for _ in range(steps):\n        x_adv.requires_grad_(True)\n        lbl = y if target is None else target\n        loss = F.cross_entropy(model(x_adv), lbl)\n        grad, = torch.autograd.grad(loss, x_adv)\n        direction = grad.sign() if target is None else -grad.sign()   # ascend own loss / descend target loss\n        with torch.no_grad():\n            x_adv = x_adv + alpha * direction\n            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)\n    return x_adv.detach()\n\nt = (y_te + 1) % 10\nfor eps in (0.05, 0.1, 0.2):\n    xa = pgd(std, X_te, y_te, eps, 2.5 * eps / 20, 20, target=t)\n    with torch.no_grad():\n        print(eps, (std(xa).argmax(1) == t).float().mean().item())",
          },
          { t: "p", text: "The targeted success rate must be *lower* than (or at most equal to) the untargeted error rate at the same ε. Every targeted success is also a misclassification. But an untargeted attack can land in whichever wrong class is closest, while a targeted one must reach one specific class, which may be far away (turning a 1 into a 2 is harder than turning it into a 7)." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Adversarial example: $x + \\delta$ with $\\|\\delta\\|_p \\le \\epsilon$, chosen to maximise the loss. Always state the threat model (norm, ε, what the attacker knows).",
          "FGSM: $x' = \\mathrm{clip}(x + \\epsilon\\,\\mathrm{sign}(\\nabla_x L))$ — the exact best change for the linearised loss over the $\\ell_\\infty$ ball, with gain $\\epsilon\\|\\nabla_x L\\|_1$.",
          "Linearity explanation: an aligned change moves a linear score by $\\epsilon\\|w\\|_1 \\propto \\epsilon d$; random noise only by $\\propto \\epsilon\\sqrt d$.",
          "PGD: $x_{t+1} = \\mathrm{clip}_{[0,1]}(\\mathrm{clip}_{[x-\\epsilon, x+\\epsilon]}(x_t + \\alpha\\,\\mathrm{sign}(\\nabla_x L)))$, many steps, random restarts.",
          "Adversarial training: $\\min_\\theta \\mathbb E[\\max_{\\|\\delta\\|\\le\\epsilon} L]$; digits CNN at ε = 0.1: PGD accuracy 53.3% → 80.8%; cost ≈ (K + 1)×.",
          "Test honestly: accuracy-vs-ε curves, strong adaptive attacks, results that settle as attack effort grows, 0% at huge ε; watch for gradient masking.",
          "Distribution shift: a 2-pixel move dropped a linear digit classifier to 5.8% accuracy at 67% confidence — confidence is not a safety signal.",
          "Fairness: demographic parity, equal opportunity/equalized odds and predictive parity conflict when base rates differ; choose per application and report per-group metrics.",
          "Responsible AI in practice: model cards and datasheets, privacy (DP-SGD, membership-inference audits), red-teaming, monitoring and human oversight. Alignment = getting the intended objective, not a stand-in.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Interpretability](/learn/m19/interpretability) — saliency maps use the same input gradient as FGSM; knowing which features a model uses is the flip side of robustness.",
          "[MLOps](/learn/m18/mlops) — watching for distribution shift and drift in production.",
          "[Instruction Tuning, RLHF and DPO](/learn/m20/alignment) — how alignment is done in practice for large language models.",
          "[Reading and Reproducing Papers](/learn/m19/reading-papers) — fair benchmarks and ablations, which robustness research needs more than most fields.",
        ],
      },
    ],
  },
};

export default lesson;
