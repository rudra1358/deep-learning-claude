import type { Lesson } from "@/lib/types";

const REG_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_digits

# ---------------- 1. inverted dropout preserves the expectation ----------------
rng = np.random.default_rng(0)
a = np.array([0.8, 0.0, 1.5, 0.3, 2.0])          # activations of one hidden layer
p = 0.5                                           # drop probability
masks = rng.random((100_000, a.size)) >= p        # keep each unit with prob 1 - p
inverted = (masks * a) / (1 - p)                  # train-time inverted dropout
classic = masks * a                               # classic dropout: no scaling at train time
print("a                      :", a)
print("mean inverted (train)  :", inverted.mean(0).round(3))
print("mean classic  (train)  :", classic.mean(0).round(3), "-> eval must use (1-p)*a =", (1 - p) * a)

# ---------------- 2. data: a small training set so a wide MLP overfits ----------------
X, y = load_digits(return_X_y=True)               # 1797 8x8 images, pixel values 0..16
X = X / 16.0
perm = np.random.default_rng(0).permutation(len(X))
tr, va, te = perm[:200], perm[200:700], perm[700:]  # 200 train / 500 val / 1097 test
Y = np.eye(10)[y]                                  # one-hot targets


class MLP:
    """64 -> H -> 10, ReLU hidden layer, softmax + cross-entropy, inverted dropout on the hidden layer."""

    def __init__(self, hidden=256, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(2 / 64), (64, hidden)); self.b1 = np.zeros(hidden)   # He init
        self.W2 = r.normal(0, np.sqrt(2 / hidden), (hidden, 10)); self.b2 = np.zeros(10)
        self.rng = np.random.default_rng(seed + 1)                                    # dropout masks

    def forward(self, X, p_drop=0.0, train=False):
        self.X = X
        self.h = np.maximum(0, X @ self.W1 + self.b1)
        if train and p_drop > 0:
            self.mask = (self.rng.random(self.h.shape) >= p_drop) / (1 - p_drop)  # inverted dropout
        else:
            self.mask = np.ones_like(self.h)                                      # eval: identity
        self.hd = self.h * self.mask
        z = self.hd @ self.W2 + self.b2
        z -= z.max(1, keepdims=True)                                              # stable softmax
        self.P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return self.P

    def step(self, Yb, lr, l2=0.0, l1=0.0):
        n = len(Yb)
        dz = (self.P - Yb) / n                                    # softmax + CE gradient
        dW2 = self.hd.T @ dz; db2 = dz.sum(0)
        dh = (dz @ self.W2.T) * self.mask * (self.h > 0)          # dropout mask is part of the graph
        dW1 = self.X.T @ dh; db1 = dh.sum(0)
        for W, dW in ((self.W1, dW1), (self.W2, dW2)):
            W *= (1 - lr * l2)                                    # weight decay: shrink by (1 - lr*l2)
            W -= lr * dW                                          # data-gradient step
            if l1 > 0:                                            # L1 proximal step = soft-threshold
                np.copyto(W, np.sign(W) * np.maximum(np.abs(W) - lr * l1, 0))
        self.b1 -= lr * db1; self.b2 -= lr * db2                  # biases are not regularised


def evaluate(net, idx):
    P = net.forward(X[idx])
    return -np.mean(np.log(P[np.arange(len(idx)), y[idx]] + 1e-12)), np.mean(P.argmax(1) == y[idx])


def train(l2=0.0, l1=0.0, p_drop=0.0, patience=None, epochs=300, lr=0.3, batch=25):
    net, order = MLP(), np.random.default_rng(42)
    best = (np.inf, 0, None)
    for ep in range(epochs):
        idx = order.permutation(tr)
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            net.forward(X[j], p_drop, train=True)
            net.step(Y[j], lr, l2, l1)
        if patience is not None:                                 # early stopping on validation loss
            v = evaluate(net, va)[0]
            if v < best[0]:
                best = (v, ep, [w.copy() for w in (net.W1, net.b1, net.W2, net.b2)])
            elif ep - best[1] >= patience:
                break
    if patience is not None:                                     # restore the best weights
        net.W1, net.b1, net.W2, net.b2 = best[2]
    return net, ep + 1, best[1] + 1


print("\nmethod            epochs  train acc  val loss  val acc  test acc  zero weights")
configs = [("baseline", {}), ("L2 (1e-3)", {"l2": 1e-3}), ("L1 prox (1e-3)", {"l1": 1e-3}),
           ("dropout p=0.5", {"p_drop": 0.5}), ("early stopping", {"patience": 20})]
for name, kw in configs:
    net, ran, best_ep = train(**kw)
    zeros = np.mean(np.concatenate([net.W1.ravel(), net.W2.ravel()]) == 0)
    tl, ta = evaluate(net, tr); vl, vacc = evaluate(net, va); _, tacc = evaluate(net, te)
    eps = f"{best_ep}/{ran}" if "patience" in kw else str(ran)
    print(f"{name:16s}  {eps:>6s}  {ta:9.3f}  {vl:8.3f}  {vacc:7.3f}  {tacc:8.3f}  {zeros:11.1%}")

# ---------------- 3. choose lambda on the validation set, report test once ----------------
print("\nL2 strength   train acc  val loss  val acc")
results = []
for lam in [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
    net, _, _ = train(l2=lam)
    results.append((evaluate(net, va)[1], lam, net))
    print(f"{lam:9.0e}   {evaluate(net, tr)[1]:9.3f}  {evaluate(net, va)[0]:8.3f}  {results[-1][0]:7.3f}")
best_val, best_lam, best_net = max(results, key=lambda r: r[0])
print(f"best lambda {best_lam:.0e} (val acc {best_val:.3f}) -> test acc {evaluate(best_net, te)[1]:.3f}")`;

const REG_NUMPY_OUT = String.raw`a                      : [0.8 0.  1.5 0.3 2. ]
mean inverted (train)  : [0.796 0.    1.503 0.301 2.003]
mean classic  (train)  : [0.398 0.    0.751 0.15  1.001] -> eval must use (1-p)*a = [0.4  0.   0.75 0.15 1.  ]

method            epochs  train acc  val loss  val acc  test acc  zero weights
baseline             300      1.000     0.221    0.932     0.942         0.0%
L2 (1e-3)            300      1.000     0.209    0.938     0.946         0.0%
L1 prox (1e-3)       300      1.000     0.254    0.920     0.934        89.1%
dropout p=0.5        300      1.000     0.206    0.940     0.947         0.0%
early stopping    134/154      1.000     0.220    0.932     0.940         0.0%

L2 strength   train acc  val loss  val acc
    0e+00       1.000     0.221    0.932
    1e-04       1.000     0.216    0.932
    1e-03       1.000     0.209    0.938
    1e-02       1.000     0.317    0.932
    3e-02       0.985     0.544    0.898
    1e-01       0.770     1.359    0.676
best lambda 1e-03 (val acc 0.938) -> test acc 0.946`;

const REG_TORCH = String.raw`import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.set_num_threads(1)                      # single thread -> bit-reproducible on CPU

# ---- Part 1: nn.Dropout is inverted dropout, and model.eval() switches it off ----
torch.manual_seed(3)
drop = nn.Dropout(p=0.5)
h = torch.tensor([0.8, 0.1, 1.5, 0.3, 2.0, 0.6, 1.2, 0.4])
drop.train(); print("train mode:", [round(v, 2) for v in drop(h).tolist()])  # survivors x 1/(1-p) = 2
drop.eval();  print("eval mode :", [round(v, 2) for v in drop(h).tolist()])  # identity
big = torch.full((200_000,), 1.0)
drop.train(); print("mean of 200k dropped ones (train):", round(drop(big).mean().item(), 3))

# ---- Part 2: Adam + L2 is not weight decay; AdamW is ----
# Pure penalty, no data loss: the only gradient is lambda * w.
for lam in (1e-4, 1e-2):
    for Opt in (torch.optim.Adam, torch.optim.AdamW):
        w = nn.Parameter(torch.tensor([2.0]))
        opt = Opt([w], lr=0.01, weight_decay=lam)
        for _ in range(100):
            opt.zero_grad()
            (0.0 * w).sum().backward()          # zero data gradient
            opt.step()
        print(f"{Opt.__name__:5s} weight_decay={lam:.0e}: w after 100 steps = {w.item():.4f}")

# ---- Part 3: regularisers on a small digits training set ----
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
perm = torch.tensor(np.random.default_rng(0).permutation(len(X)))
tr, va, te = perm[:200], perm[200:700], perm[700:]       # 200 train / 500 val / 1097 test


def augment(xb, g, max_deg=12.0, max_scale=0.10, max_shift=0.12):
    """Random small rotation, scaling and translation of each 8x8 image (no flips: a flipped 2 is not a 2)."""
    n = xb.shape[0]
    ang = (torch.rand(n, generator=g) * 2 - 1) * math.radians(max_deg)
    s = 1 + (torch.rand(n, generator=g) * 2 - 1) * max_scale
    t = (torch.rand(n, 2, generator=g) * 2 - 1) * max_shift      # in [-1,1] image coordinates
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = torch.cos(ang) / s; theta[:, 0, 1] = -torch.sin(ang) / s
    theta[:, 1, 0] = torch.sin(ang) / s; theta[:, 1, 1] = torch.cos(ang) / s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, (n, 1, 8, 8), align_corners=False)
    return F.grid_sample(xb.view(n, 1, 8, 8), grid, align_corners=False).view(n, 64)


def evaluate(model, idx):
    model.eval()                                           # dropout off
    with torch.no_grad():
        out = model(X[idx])
        return F.cross_entropy(out, y[idx]).item(), (out.argmax(1) == y[idx]).float().mean().item()


def run(seed, p=0.0, wd=0.0, aug=False, patience=None, epochs=100):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 100)          # shuffling + augmentation randomness
    model = nn.Sequential(nn.Linear(64, 512), nn.ReLU(), nn.Dropout(p),
                          nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=wd)
    best_loss, best_ep, best_state = float("inf"), 0, None
    for ep in range(epochs):
        model.train()                                      # dropout on
        order = tr[torch.randperm(len(tr), generator=g)]
        for b in range(0, len(order), 25):
            j = order[b:b + 25]
            xb = augment(X[j], g) if aug else X[j]
            loss = F.cross_entropy(model(xb), y[j])
            opt.zero_grad(); loss.backward(); opt.step()
        if patience is not None:
            vl = evaluate(model, va)[0]
            if vl < best_loss:
                best_loss, best_ep = vl, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= patience:
                break
    if patience is not None:
        model.load_state_dict(best_state)                  # restore best weights
    return [evaluate(model, tr)[1], *evaluate(model, va), evaluate(model, te)[1], ep + 1]


configs = [("baseline", {}), ("L2 / weight decay 5e-3", {"wd": 5e-3}), ("dropout p=0.5", {"p": 0.5}),
           ("early stopping (pat. 20)", {"patience": 20}), ("augmentation", {"aug": True}),
           ("augmentation + dropout 0.3", {"aug": True, "p": 0.3})]
print("\nmean of 3 seeds              epochs  train acc  val loss  val acc  test acc")
for name, kw in configs:
    r = np.mean([run(s, **kw) for s in range(3)], axis=0)
    print(f"{name:27s}  {r[4]:6.1f}  {r[0]:9.3f}  {r[1]:8.3f}  {r[2]:7.3f}  {r[3]:8.3f}")`;

const REG_TORCH_OUT = String.raw`train mode: [1.6, 0.2, 0.0, 0.6, 0.0, 0.0, 2.4, 0.8]
eval mode : [0.8, 0.1, 1.5, 0.3, 2.0, 0.6, 1.2, 0.4]
mean of 200k dropped ones (train): 1.004
Adam  weight_decay=1e-04: w after 100 steps = 1.0985
AdamW weight_decay=1e-04: w after 100 steps = 1.9998
Adam  weight_decay=1e-02: w after 100 steps = 1.0984
AdamW weight_decay=1e-02: w after 100 steps = 1.9801

mean of 3 seeds              epochs  train acc  val loss  val acc  test acc
baseline                      100.0      1.000     0.280    0.914     0.940
L2 / weight decay 5e-3        100.0      1.000     0.242    0.922     0.945
dropout p=0.5                 100.0      1.000     0.267    0.933     0.949
early stopping (pat. 20)       39.0      1.000     0.257    0.917     0.939
augmentation                  100.0      1.000     0.246    0.945     0.955
augmentation + dropout 0.3    100.0      1.000     0.300    0.930     0.949`;

const EX_DROPOUT = String.raw`import numpy as np


def dropout_forward(h, p, rng):
    """Inverted dropout. Returns the output and the cache needed by backward."""
    mask = (rng.random(h.shape) >= p) / (1 - p)     # 0 or 1/(1-p)
    return h * mask, mask


def dropout_backward(dout, mask):
    return dout * mask                               # same mask, same scaling


rng = np.random.default_rng(0)
h = rng.normal(size=(4, 6))
w = rng.normal(size=(6,))
p = 0.3

out, mask = dropout_forward(h, p, rng)
loss = np.sum((out @ w) ** 2)                        # any scalar function of the output
dout = 2 * (out @ w)[:, None] * w[None, :]           # dloss/dout
dh = dropout_backward(dout, mask)

# finite-difference check with the SAME mask held fixed
num = np.zeros_like(h)
eps = 1e-6
for idx in np.ndindex(h.shape):
    hp, hm = h.copy(), h.copy()
    hp[idx] += eps; hm[idx] -= eps
    num[idx] = (np.sum(((hp * mask) @ w) ** 2) - np.sum(((hm * mask) @ w) ** 2)) / (2 * eps)
print("max |analytic - numeric| =", f"{np.max(np.abs(dh - num)):.2e}")
print("dropped units get zero gradient:", bool(np.all(dh[mask == 0] == 0)))
print("kept fraction:", np.mean(mask > 0).round(3), " scale of kept units:", np.unique(mask[mask > 0]).round(4))`;

const EX_DROPOUT_OUT = String.raw`max |analytic - numeric| = 2.64e-09
dropped units get zero gradient: True
kept fraction: 0.792  scale of kept units: [1.4286]`;

const REG_KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")
perm = np.random.default_rng(0).permutation(len(X))
tr, va = perm[:200], perm[200:700]

# Data augmentation as layers: active in training, identity at inference
augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(12 / 360),        # fraction of a full turn
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.06, 0.06),
])

l2 = tf.keras.regularizers.L2(2.5e-3)                # adds 2.5e-3 * sum(w^2) to the loss
inputs = tf.keras.Input(shape=(8, 8, 1))
x = augment(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=l2)(x)
x = tf.keras.layers.Dropout(0.5)(x)                  # inverted dropout; off in predict/evaluate
x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=l2)(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.SGD(0.05, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X[tr].reshape(-1, 8, 8, 1), y[tr], batch_size=25, epochs=300,
          validation_data=(X[va].reshape(-1, 8, 8, 1), y[va]), callbacks=[stop], verbose=0)
print("stopped after", len(model.history.history["loss"]), "epochs")
print(model.evaluate(X[va].reshape(-1, 8, 8, 1), y[va], verbose=0))`;

const lesson: Lesson = {
  id: "m07/regularization",
  title: "Regularization: L1, L2, Dropout, Early Stopping",
  summary:
    "The toolbox against overfitting: weight penalties (L2 / weight decay, L1 with proximal soft-thresholding, elastic net, max-norm), dropout and its expectation-preserving 1/(1−p) scaling, early stopping, data augmentation, noise injection and label smoothing — derived, implemented from scratch, and compared on a real overfitting experiment.",
  minutes: 95,
  level: "intermediate",
  prerequisites: ["m03/training-basics", "m02/mle-map", "m02/norms-spaces", "m04/backpropagation", "m06/gradient-descent"],
  next: ["m07/residual-connections", "m07/normalization", "m06/momentum-adaptive", "m17/experimentation"],
  terms: ["regularization", "overfitting", "generalization-gap", "weight-decay", "l2-regularization", "l1-regularization", "soft-thresholding", "proximal-gradient", "elastic-net", "max-norm", "dropout", "inverted-dropout", "early-stopping", "data-augmentation", "mixup", "label-smoothing", "adamw"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      { t: "p", text: String.raw`A trained network is judged on data it has **not** seen. The difference between its error on new data and its error on the training set is the **generalization gap**. [Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) showed how a high-capacity model drives training error to zero while validation error stalls or rises — overfitting. **Regularization** is the collective name for every modification of the learning procedure that is meant to reduce the generalization error, *possibly at the cost of a slightly higher training error*.` },
      { t: "p", text: "Modern networks routinely have more parameters than training examples, so some form of regularization is present in essentially every serious training run. The techniques fall into four families:" },
      { t: "list", items: [
        String.raw`**Penalties on the weights** — add a term to the loss that prefers small or sparse weights: **L2** (weight decay), **L1** (lasso), their mixture the **elastic net**, and hard **max-norm** constraints. They encode a prior belief that simple functions are more likely ([MAP estimation](/learn/m02/mle-map)).`,
        String.raw`**Noise during training** — **dropout** randomly removes hidden units on every forward pass; relatives drop weights (DropConnect), whole feature maps (spatial dropout) or whole residual blocks (DropPath). Adding noise to inputs and softening the targets (**label smoothing**) belong here too.`,
        String.raw`**More (synthetic) data** — **data augmentation** applies label-preserving transformations (flips, crops, rotations, colour changes, audio masking) so the network sees a larger, more varied training set; **mixup** and **CutMix** blend examples.`,
        String.raw`**Limiting the optimisation** — **early stopping** halts training when the validation loss stops improving and restores the best weights.`,
      ] },
      { t: "p", text: String.raw`Dropout was introduced by Hinton et al. (2012) and analysed in depth by Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (*JMLR*, 2014). Weight decay is decades older; its correct form for adaptive optimizers was clarified by Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019), which gave us **AdamW** — the default optimizer for training transformers today.` },
      { t: "callout", kind: "key", text: String.raw`Regularization trades a little training fit for a smaller generalization gap. Penalties shrink the weights ($w \leftarrow (1-\eta\lambda)w$ for L2; soft-threshold to exact zeros for L1), dropout trains an exponentially large ensemble of weight-sharing sub-networks with an expectation-preserving $1/(1-p)$ rescale, augmentation enlarges the data, and early stopping limits how far the optimizer travels. Every strength ($\lambda$, $p$, patience, augmentation magnitude) is a hyperparameter chosen on the **validation** set.` },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      { t: "list", items: [
        String.raw`[Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) — train/validation/test splits, learning curves, the bias–variance picture and the diagnosis "large gap = overfitting". This lesson is the cure for the disease diagnosed there.`,
        String.raw`[Maximum Likelihood and MAP Estimation](/learn/m02/mle-map) — why an L2 penalty is a Gaussian prior and an L1 penalty a Laplace prior. We use that result and do not re-derive it.`,
        String.raw`[Vector Spaces, Norms and Distances](/learn/m02/norms-spaces) — the L1 and L2 norms, their unit balls (diamond vs circle) and the one-coordinate argument for why L1 gives exact zeros. Here we extend it to proximal gradient descent, the elastic net and networks.`,
        String.raw`[Backpropagation](/learn/m04/backpropagation) — dropout is just another node in the computational graph; you need to see that its backward pass multiplies by the same mask.`,
        String.raw`[Gradient Descent](/learn/m06/gradient-descent) and [Momentum and Adaptive Optimizers](/learn/m06/momentum-adaptive) — the SGD update, and Adam's per-parameter division by $\sqrt{\hat v}$, which is exactly why "L2 penalty" and "weight decay" stop being the same thing under Adam.`,
        String.raw`[Probability](/learn/m02/probability) — Bernoulli random variables, expectation and variance (for the dropout derivation). [Entropy and Cross-Entropy](/learn/m02/information-theory) for the label-smoothing pointer.`,
      ] },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      { t: "callout", kind: "analogy", title: "Studying for an exam you have not seen", text: "A student who memorises last year's answer sheet scores 100% on last year's paper and fails this year's. Good teachers counter this in several ways. They insist that explanations be **simple** (a weight penalty: no elaborate story when a short one fits). They run **study groups where random members are absent each session**, so nobody can rely on the one person who knows topic 7 — every student must understand enough on their own (dropout prevents co-adaptation). They give **reworded versions of the same questions** — numbers changed, diagrams rotated (data augmentation). And they **stop drilling once mock-exam scores stop improving**, because further hours only add memorisation (early stopping). The mock exam is the validation set; the real exam is the test set, taken once." },
      { t: "p", text: String.raw`**Weight penalties.** Large weights let a network build very sharp, wiggly functions that pass through every training point. Adding $\lambda\|w\|^2$ to the loss charges a price for each unit of weight, so the optimizer only keeps a large weight if it buys a correspondingly large reduction in the data loss. L2 shrinks every weight proportionally — like a spring pulling each weight towards zero. L1 pulls with a *constant* force, so weights whose data-gradient is weaker than that force end up at exactly zero: L1 performs feature selection.` },
      { t: "p", text: String.raw`**Dropout.** On every training step each hidden unit is switched off with probability $p$. The network can never count on a specific partner unit being present, so each unit must learn features that are useful on their own and in many combinations. Because the surviving units are scaled up by $1/(1-p)$, the *average* signal flowing into the next layer is unchanged, so at test time we can simply use all units with no scaling. A network with $n$ droppable units contains $2^n$ sub-networks that share weights; dropout trains a random one at each step, and the full network at test time behaves approximately like their average — an ensemble at the cost of one model.` },
      { t: "p", text: String.raw`**Early stopping.** Training from small initial weights, the optimizer first fits the broad, strong patterns and only later the idiosyncratic details of individual training points. Stopping when the validation loss bottoms out keeps the first and discards the second. **Augmentation** attacks the problem at its root — too little data — by generating new, plausible training examples from the ones we have.` },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "p", text: "Here is one epoch of a training loop that uses every technique in this lesson, in the order the operations actually happen:" },
      { t: "steps", items: [
        { title: "Sample a mini-batch and augment it", text: "Draw a batch from the **training** set and apply random, label-preserving transformations (a different random draw every time the example is seen). Validation and test data are never augmented." },
        { title: "Forward pass in training mode", text: String.raw`For every dropout layer, draw a fresh Bernoulli mask $m$ with $P(m_i = 1) = 1 - p$, multiply the activations by $m$ and by $1/(1-p)$. Store the mask — the backward pass needs it.`, tex: String.raw`\tilde h = \frac{m \odot h}{1-p}` },
        { title: "Loss (+ explicit penalty, if used)", text: String.raw`Compute the data loss (with label-smoothed targets if used). If you use an explicit L2 or L1 penalty in the loss, add $\frac{\lambda}{2}\|W\|_2^2$ or $\lambda\|W\|_1$ — usually over weight matrices only, not biases or normalization parameters.` },
        { title: "Backward pass", text: String.raw`Backpropagate as usual. Through a dropout node the upstream gradient is multiplied by the same $m/(1-p)$, so dropped units receive exactly zero gradient and nothing flows into the weights feeding them on this step.`, tex: String.raw`\frac{\partial\mathcal L}{\partial h} = \frac{m}{1-p}\odot\frac{\partial\mathcal L}{\partial\tilde h}` },
        { title: "Parameter update with decay / proximal / projection", text: String.raw`Apply the optimizer step. Decoupled weight decay multiplies weights by $(1-\eta\lambda)$ separately from the adaptive gradient step (AdamW). An L1 proximal step soft-thresholds the weights after the gradient step. A max-norm constraint rescales any unit's incoming weight vector whose norm exceeds $c$.` },
        { title: "Evaluate in eval mode", text: "At the end of the epoch switch to evaluation mode: dropout becomes the identity (inverted dropout needs no test-time scaling), augmentation is off, BatchNorm uses running statistics. Compute the validation loss and accuracy." },
        { title: "Early-stopping bookkeeping", text: "If the validation loss improved, save a copy of the weights and reset the patience counter; otherwise increment it. When the counter reaches the patience (e.g. 10–20 epochs), stop." },
        { title: "Restore and report", text: "Load the best saved weights. Hyperparameters (λ, p, patience, augmentation strength) are chosen by repeating this on a grid and picking the best **validation** score; the **test** set is evaluated once, at the very end." },
      ] },
      { t: "h", text: "Dropout: where and how much" },
      { t: "list", items: [
        String.raw`**Fully-connected hidden layers**: $p = 0.5$ was the default in Srivastava et al. (2014) and in AlexNet's classifier layers. On inputs, much smaller rates ($p \approx 0.2$) — dropping half of the pixels destroys the signal.`,
        String.raw`**Transformers**: $p = 0.1$ on attention weights, after each sub-layer before the residual addition, and on embeddings (the original Transformer used $P_{drop} = 0.1$). Very large language models trained on huge corpora for about one epoch often use $p = 0$ — they barely overfit.`,
        String.raw`**Convolutional layers**: ordinary element-wise dropout is weak there, because neighbouring pixels of a feature map are strongly correlated and the dropped value is effectively reconstructed from its neighbours. Use **spatial dropout** (drop entire channels) or rely on augmentation, weight decay and BatchNorm instead.`,
        String.raw`**Never on the output layer** (it would randomly delete logits) and never at test time unless you deliberately want **Monte Carlo dropout** for uncertainty estimates.`,
        String.raw`**Dropout + BatchNorm**: a dropout layer placed *before* BatchNorm makes the variance BN sees during training ($\propto 1/(1-p)$ larger for the kept units) differ from the variance at test time, so BN's running statistics are wrong at inference — the "variance shift" analysed by Li et al. (2019). Put dropout after the last BN layer (e.g. just before the classifier) or use small $p$.`,
      ] },
      { t: "h", text: "Early stopping: the procedure" },
      { t: "list", items: [
        String.raw`Monitor the **validation loss** (smoother than accuracy) once per epoch or every few hundred steps.`,
        String.raw`**Patience** $P$: tolerate $P$ evaluations without improvement before stopping, because validation curves are noisy; optionally require an improvement of at least $\delta$ (a *min-delta*).`,
        String.raw`**Restore the best weights** — without this, you return the model from $P$ evaluations *after* the optimum.`,
        String.raw`Alternative when data are scarce: record the best number of epochs $t^\star$, then retrain on train + validation for $t^\star$ (or a slightly scaled) epochs.`,
      ] },
      { t: "h", text: "Data augmentation: what is label-preserving?" },
      { t: "p", text: "The transformation must not change the correct answer. Horizontal flips are safe for natural photos (a flipped cat is a cat) but not for digits or text (a mirrored 2 is not a 2, and 6 vs 9 under rotation). Typical choices:" },
      { t: "list", items: [
        "**Images**: random resized crops and translations, horizontal flips (when the class is mirror-symmetric), small rotations and scalings, colour jitter (brightness, contrast, saturation, hue), Gaussian blur; **Cutout** (DeVries & Taylor, 2017) masks a random square; **mixup** (Zhang et al., 2018) trains on convex combinations of two images and their labels; **CutMix** pastes a patch of one image into another and mixes labels by area. For detection and segmentation the boxes/masks must be transformed with the image.",
        "**Text**: synonym replacement, random token deletion or masking, back-translation (translate to another language and back). Much harder to make label-preserving than images; for large language models the usual answer is simply more data.",
        "**Audio / speech**: time shifts, speed and pitch perturbation, added background noise, reverberation, and **SpecAugment** (Park et al., 2019): masking random blocks of time steps and frequency channels in the spectrogram.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "The generalization gap" },
      { t: "p", text: String.raw`For a model $f$ with loss $\ell$, the **true risk** is $R(f) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}[\ell(f(x), y)]$ and the **empirical risk** on $n$ training examples is $\hat R(f) = \frac1n\sum_{i=1}^n \ell(f(x_i), y_i)$. Training minimises $\hat R$; we care about $R$. The **generalization gap** is $R(f) - \hat R(f)$, estimated by (validation loss − training loss). A regularized objective is` },
      { t: "math", tex: String.raw`J(\theta) = \hat R(\theta) + \lambda\,\Omega(\theta)`, caption: String.raw`$\Omega$ is the penalty (regularizer) and $\lambda \ge 0$ its strength. Implicit regularizers (dropout, early stopping, augmentation) change the procedure rather than adding an explicit $\Omega$.` },
      { t: "h", text: "L2 regularization and weight decay" },
      { t: "p", text: String.raw`With $\Omega(w) = \frac12\|w\|_2^2$ the gradient gains a term $\lambda w$, and one gradient-descent step with learning rate $\eta$ becomes` },
      { t: "math", tex: String.raw`w_{t+1} = w_t - \eta\big(\nabla\hat R(w_t) + \lambda w_t\big) = \underbrace{(1-\eta\lambda)}_{\text{shrinkage factor}}\,w_t - \eta\,\nabla\hat R(w_t)` },
      { t: "p", text: String.raw`Every step first **decays** each weight by the factor $(1-\eta\lambda)$, then takes the ordinary data step — hence the name *weight decay*. **Worked numbers**: $\eta = 0.1$, $\lambda = 0.01$ gives a factor $0.999$ per step. A weight $w = 2$ that receives no data gradient becomes $2 \times 0.999^{1000} = 0.7354$ after 1000 steps, close to the continuous-time value $2e^{-\eta\lambda t} = 2e^{-1} = 0.7358$. A weight survives only if the data gradient keeps pushing it back.` },
      { t: "p", text: String.raw`**What L2 does to the solution.** Approximate the data loss near its unregularized minimum $w^\star$ by a quadratic with Hessian $H = Q\,\mathrm{diag}(h_1,\dots,h_d)\,Q^\top$ (eigenvalues $h_i \ge 0$ = curvatures). Setting the gradient of $\frac12(w-w^\star)^\top H (w - w^\star) + \frac\lambda2\|w\|^2$ to zero gives` },
      { t: "math", tex: String.raw`\tilde w = (H + \lambda I)^{-1} H\, w^\star \quad\Longrightarrow\quad Q^\top\tilde w = \mathrm{diag}\!\Big(\frac{h_i}{h_i+\lambda}\Big)\,Q^\top w^\star`, caption: "Along each eigen-direction of the Hessian, L2 multiplies the solution by h/(h + λ)." },
      { t: "p", text: String.raw`Directions with high curvature ($h_i \gg \lambda$) — the ones the data constrain strongly — are barely changed; directions with low curvature ($h_i \ll \lambda$), along which the data loss hardly cares, are shrunk almost to zero. L2 removes exactly the parameter directions that the data cannot determine, which are the ones most likely to be fitting noise. The Bayesian reading — $\lambda = \sigma^2/\tau^2$ for a Gaussian prior $\mathcal N(0,\tau^2)$ — is derived in [MLE and MAP](/learn/m02/mle-map).` },
      { t: "h", text: "Weight decay ≠ L2 under Adam" },
      { t: "p", text: String.raw`For SGD the two are identical (above). For Adam ([Momentum and Adaptive Optimizers](/learn/m06/momentum-adaptive)) the L2 gradient $\lambda w$ is added to $g$ **before** the moment estimates, so it is divided by $\sqrt{\hat v} + \epsilon$ like everything else:` },
      { t: "math", tex: String.raw`\text{Adam}+L_2:\; w \leftarrow w - \eta\,\frac{\hat m(g + \lambda w)}{\sqrt{\hat v(g+\lambda w)}+\epsilon} \qquad\quad \text{AdamW}:\; w \leftarrow w - \eta\Big(\frac{\hat m(g)}{\sqrt{\hat v(g)}+\epsilon} + \lambda w\Big)` },
      { t: "p", text: String.raw`With Adam + L2, a weight whose gradients have been large (big $\hat v$) is decayed *less*, and in the extreme case of no data gradient the normalised step is $\approx \eta\,\mathrm{sign}(w)$ **regardless of** $\lambda$. AdamW decouples the decay so every weight shrinks by the same factor $(1-\eta\lambda)$. The PyTorch program below measures this: starting from $w = 2$ with zero data gradient and $\eta = 0.01$, after 100 steps Adam with weight_decay $10^{-4}$ and $10^{-2}$ both reach $w \approx 1.098$ (the penalty strength is irrelevant), while AdamW gives $2(1-10^{-6})^{100} = 1.9998$ and $2(1-10^{-4})^{100} = 1.9801$ — exactly the intended decay.` },
      { t: "h", text: "L1 regularization, subgradients and soft-thresholding" },
      { t: "p", text: String.raw`With $\Omega(w) = \|w\|_1 = \sum_j |w_j|$ the penalty is not differentiable at $w_j = 0$. Its **subdifferential** is $\partial|w_j| = \{\mathrm{sign}(w_j)\}$ for $w_j \neq 0$ and the whole interval $[-1, 1]$ at $w_j = 0$. The optimality condition $0 \in \nabla_j\hat R(w) + \lambda\,\partial|w_j|$ therefore says:` },
      { t: "math", tex: String.raw`w_j = 0 \text{ is optimal} \iff \Big|\frac{\partial \hat R}{\partial w_j}\Big| \le \lambda`, caption: "A coordinate stays at exactly zero whenever the data's pull on it is weaker than λ. This is the source of sparsity." },
      { t: "p", text: String.raw`Plain (sub)gradient descent never lands exactly on 0 — it oscillates around it by about $\eta\lambda$ per step. The right tool is the **proximal gradient** method: take a gradient step on the smooth part, then solve a tiny problem that handles the penalty exactly:` },
      { t: "math", tex: String.raw`w_{t+1} = \operatorname{prox}_{\eta\lambda\|\cdot\|_1}\!\big(w_t - \eta\nabla\hat R(w_t)\big), \qquad \operatorname{prox}_{\tau|\cdot|}(v) = \arg\min_w\ \tfrac12(w - v)^2 + \tau|w|` },
      { t: "p", text: String.raw`**Deriving the prox.** For $w > 0$ the objective's derivative is $w - v + \tau = 0 \Rightarrow w = v - \tau$, valid if $v > \tau$. For $w < 0$: $w = v + \tau$, valid if $v < -\tau$. Otherwise ($|v| \le \tau$) neither branch is consistent, and $0 \in (0 - v) + \tau[-1,1]$ holds, so $w = 0$. Together:` },
      { t: "math", tex: String.raw`\operatorname{prox}_{\tau|\cdot|}(v) = S_\tau(v) = \mathrm{sign}(v)\max(|v| - \tau,\ 0)`, caption: "Soft-thresholding: shrink every coordinate towards 0 by τ = ηλ and clamp at 0. Iterating gradient step + soft-threshold is ISTA." },
      { t: "p", text: String.raw`Example: $S_{0.1}(0.8, -0.05, 0.3) = (0.7,\ 0,\ 0.2)$ — the small coordinate is set to **exactly** zero. The geometric picture (contours touching the corners of the diamond) is in [Norms](/learn/m02/norms-spaces); the simulator below lets you move the contours yourself.` },
      { t: "h", text: "Worked example: L1 vs L2 vs elastic net in two dimensions" },
      { t: "p", text: String.raw`Take the data loss $L(w) = \frac12(w-c)^\top A(w-c)$ with $A = \mathrm{diag}(1, 2)$, unregularized optimum $c = (2, 0.5)$, and $\lambda = 1.2$. Because $A$ is diagonal the coordinates decouple, and each one is a scalar problem $\frac{a}{2}(w - c)^2 + \text{penalty}$:` },
      { t: "steps", items: [
        { title: "L2: closed form", tex: String.raw`w_i = \frac{a_i c_i}{a_i + \lambda}:\quad w_1 = \frac{2}{2.2} = 0.9091,\quad w_2 = \frac{2 \cdot 0.5}{2 + 1.2} = 0.3125` },
        { title: "L1: soft-threshold the scaled target", tex: String.raw`w_i = \frac{S_\lambda(a_i c_i)}{a_i}:\quad w_1 = \frac{2 - 1.2}{1} = 0.8,\quad w_2 = \frac{S_{1.2}(1.0)}{2} = 0 \ \text{(exactly)}` },
        { title: "Elastic net, α = 0.5", text: String.raw`Penalty $\lambda\big(\alpha\|w\|_1 + \frac{1-\alpha}{2}\|w\|_2^2\big)$:`, tex: String.raw`w_i = \frac{S_{\lambda\alpha}(a_i c_i)}{a_i + \lambda(1-\alpha)}:\quad w_1 = \frac{2 - 0.6}{1.6} = 0.875,\quad w_2 = \frac{1.0 - 0.6}{2.6} = 0.1538` },
        { title: "Price paid in data loss", tex: String.raw`L(w_{L2}) = \tfrac12(1.0909^2 \cdot 1 + 0.1875^2 \cdot 2) = 0.6302,\qquad L(w_{L1}) = \tfrac12(1.2^2 + 0.5^2\cdot 2) = 0.97` },
      ] },
      { t: "p", text: String.raw`Coordinate 2 has a weaker data pull ($|a_2c_2| = 1.0 < \lambda = 1.2$), so L1 zeroes it while L2 only shrinks it by the factor $2/3.2$. The elastic net sits between: it shrinks like L2 but would still produce zeros for larger $\lambda$. For non-diagonal $A$ (correlated features) there is no coordinate-wise formula, and the simulator solves the L1 problem exactly by enumerating sign patterns and cross-checks with ISTA. The L1 solution is exactly $0$ for every $\lambda \ge \lambda_{\max} = \|Ac\|_\infty$ (here $2$), the point where the KKT condition above holds at $w = 0$ for both coordinates.` },
      { t: "p", text: String.raw`**Why the elastic net?** With strongly correlated features the lasso tends to pick one of them arbitrarily and drop the rest, and it can select at most $n$ features when $d > n$. Adding the L2 part makes the objective strictly convex and encourages correlated features to share weight (the "grouping effect"), as argued by Zou and Hastie (2005).` },
      { t: "h", text: "Dropout: deriving the 1/(1 − p) scale" },
      { t: "p", text: String.raw`Let $h \in \mathbb R^n$ be a layer's activations, $p$ the drop probability and $m_i \sim \text{Bernoulli}(1-p)$ independent ($m_i = 1$ means *kept*). **Inverted dropout** outputs $\tilde h_i = m_i h_i/(1-p)$. Since $\mathbb E[m_i] = 1 - p$:` },
      { t: "math", tex: String.raw`\mathbb E[\tilde h_i] = \frac{\mathbb E[m_i]\,h_i}{1-p} = \frac{(1-p)\,h_i}{1-p} = h_i, \qquad \operatorname{Var}[\tilde h_i] = \frac{h_i^2\operatorname{Var}[m_i]}{(1-p)^2} = \frac{p}{1-p}\,h_i^2` },
      { t: "p", text: String.raw`The next layer computes $z = w^\top\tilde h + b$, which is **linear** in $\tilde h$, so by linearity of expectation $\mathbb E_m[z] = w^\top h + b$: the expected pre-activation during training equals the deterministic pre-activation with all units present. That is why, at test time, inverted dropout is simply switched off. If we did **not** scale during training ("classic" dropout as in the original papers), $\mathbb E[\tilde h_i] = (1-p)h_i$, and every activation — equivalently every outgoing weight — must be multiplied by $(1-p)$ at test time (the *weight scaling inference rule*). Both conventions give the same expected value; inverted dropout puts the extra work in training so inference code is untouched. With independent masks the variance of $z$ is $\operatorname{Var}[z] = \frac{p}{1-p}\sum_i w_i^2 h_i^2$.` },
      { t: "steps", items: [
        { title: "Set-up", text: String.raw`$h = (2,\ 1,\ 0.5,\ 3)$, outgoing weights $w = (0.5,\ -1,\ 2,\ 0.25)$, $b = 0$, $p = 0.5$, sampled mask $m = (1, 0, 1, 1)$.` },
        { title: "Train-mode forward", tex: String.raw`\tilde h = \frac{(2,\,0,\,0.5,\,3)}{0.5} = (4,\ 0,\ 1,\ 6),\qquad z = 0.5\cdot4 + 2\cdot1 + 0.25\cdot6 = 5.5` },
        { title: "Eval-mode forward", tex: String.raw`z_{\text{eval}} = 0.5\cdot2 - 1\cdot1 + 2\cdot0.5 + 0.25\cdot3 = 1.75` },
        { title: "Average over all 2⁴ = 16 masks", text: "Enumerating every mask (each has probability 1/16) gives a mean of exactly 1.75 — the eval output — and a variance of 3.5625:", tex: String.raw`\frac{p}{1-p}\sum_i w_i^2h_i^2 = 1\cdot(0.25\cdot4 + 1\cdot1 + 4\cdot0.25 + 0.0625\cdot9) = 3.5625` },
        { title: "Classic dropout, same mask", tex: String.raw`z = 0.5\cdot2 + 2\cdot0.5 + 0.25\cdot3 = 2.75, \qquad z_{\text{eval}} = (1-p)\cdot1.75 = 0.875` },
      ] },
      { t: "p", text: String.raw`A single sample (5.5) can be far from the mean (1.75): the standard deviation is $\sqrt{3.5625} = 1.89$. Dropout injects a lot of noise, which is precisely the regularizing effect — and why dropout networks typically need more epochs to converge.` },
      { t: "h", text: "Dropout as an ensemble — and where the approximation breaks" },
      { t: "p", text: String.raw`With $n$ droppable units there are $2^n$ masks, i.e. $2^n$ thinned sub-networks that **share** their weights. Each training step trains one randomly chosen sub-network on one mini-batch. The ideal prediction would average all $2^n$ sub-networks' outputs — impossible to compute. Using the full network (with the expectation-preserving scale) is a one-pass approximation. It is **exact** for anything linear in the dropped activations (as above). For a single dropout layer feeding a softmax, Hinton et al. (2012) showed it equals the renormalised **geometric mean** of the sub-networks' predicted distributions. For deeper non-linear networks it is only approximate, because $\mathbb E[f(z)] \neq f(\mathbb E[z])$ for non-linear $f$. **Monte Carlo dropout** (Gal & Ghahramani, 2016) keeps dropout on at test time, averages $T$ stochastic passes, and uses their spread as an uncertainty estimate.` },
      { t: "p", text: String.raw`**Dropout as adaptive L2.** For linear regression with inverted dropout on the inputs, averaging the squared error over masks gives (using $\mathbb E[\tilde x_j] = x_j$, $\operatorname{Var}[\tilde x_j] = \frac{p}{1-p}x_j^2$):` },
      { t: "math", tex: String.raw`\mathbb E_m\Big[\big(y - w^\top\tilde x\big)^2\Big] = \big(y - w^\top x\big)^2 + \frac{p}{1-p}\sum_j w_j^2 x_j^2`, caption: "Bias² + variance: dropout adds a ridge penalty whose strength per weight is scaled by the squared size of that feature." },
      { t: "p", text: String.raw`Numerically: $x = (1, 2, -1)$, $w = (0.5, 0.3, 1)$, $y = 1$, $p = 0.2$: $(1 - 0.1)^2 + 0.25\,(0.25 + 0.36 + 1) = 0.81 + 0.4025 = 1.2125$, which matches the exact average over all 8 masks. Srivastava et al. (2014) derive this marginalisation; it shows dropout is a *data-dependent* L2 penalty.` },
      { t: "h", text: "Noise injection" },
      { t: "p", text: String.raw`Adding Gaussian noise $\varepsilon \sim \mathcal N(0, \sigma^2 I)$ to the inputs of a linear model, $\mathbb E_\varepsilon[(y - w^\top(x+\varepsilon))^2] = (y - w^\top x)^2 + \sigma^2\|w\|_2^2$ — exactly ridge regression with $\lambda = \sigma^2$ per example (Bishop, 1995, "Training with noise is equivalent to Tikhonov regularization"). For deep networks the equivalence is only approximate, but noise on inputs, hidden units, weights or gradients all act as regularizers.` },
      { t: "h", text: "Early stopping as implicit L2" },
      { t: "p", text: String.raw`Run gradient descent from $w_0 = 0$ on the quadratic approximation above. In the Hessian's eigenbasis each component evolves independently: after $t$ steps component $i$ equals $\big[1 - (1-\eta h_i)^t\big]w^\star_i$. Compare L2's factor $h_i/(h_i + \lambda)$. Both are $\approx 1$ for large $h_i$ and small for small $h_i$; for small $\eta h_i$ they match when` },
      { t: "math", tex: String.raw`\lambda \approx \frac{1}{\eta\, t}`, caption: "Training for t steps with learning rate η behaves like L2 with strength about 1/(ηt): stopping early = stronger regularization." },
      { t: "p", text: "Early stopping therefore regularizes for free — it picks the effective λ automatically by monitoring validation loss, in a single training run." },
      { t: "h", text: "Max-norm, label smoothing and mixup" },
      { t: "p", text: String.raw`**Max-norm constraint**: after every update, for each unit $j$ with incoming weight vector $w_j$, if $\|w_j\|_2 > c$ set $w_j \leftarrow c\,w_j/\|w_j\|_2$ (projection onto a ball). Example: $w_j = (3, 4, 12)$ has norm 13; with $c = 3$ it becomes $(0.6923,\ 0.9231,\ 2.7692)$. Srivastava et al. (2014) found max-norm ($c \approx 3$–$4$) combined with dropout allows large learning rates without weights blowing up.` },
      { t: "p", text: String.raw`**Label smoothing** replaces the one-hot target $y$ by $(1-\varepsilon)y + \varepsilon/K$ for $K$ classes, e.g. $\varepsilon = 0.1$, $K = 10$: the true class gets 0.91 and every other class 0.01. The cross-entropy minimum is then at finite logits, discouraging over-confident predictions (Szegedy et al., 2016). See [cross-entropy](/learn/m02/information-theory) and [Classification Losses](/learn/m05/classification-losses).` },
      { t: "p", text: String.raw`**Mixup** trains on $\tilde x = \beta x_i + (1-\beta)x_j$, $\tilde y = \beta y_i + (1-\beta)y_j$ with $\beta \sim \mathrm{Beta}(\alpha, \alpha)$, encouraging the network to behave linearly between training examples.` },
      { t: "h", text: "Choosing λ (and every other strength) by validation" },
      { t: "p", text: String.raw`Regularization strengths are hyperparameters: they cannot be learned on the training set (the training loss is always lowest at $\lambda = 0$). Search them on a log grid (e.g. $\lambda \in \{0, 10^{-5}, 10^{-4}, \dots, 10^{-1}\}$), train one model per value, pick the value with the best **validation** metric, and report the **test** metric of that single model. Validation error versus $\lambda$ is U-shaped: too small overfits (variance), too large underfits (bias). The NumPy experiment below traces this curve.` },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "p", text: String.raw`**Dropout simulator.** A fixed 8 → 10 → 1 network with ReLU hidden units; dropout acts on the hidden layer. Each **Step** draws a new Bernoulli mask from a seeded sequence. Dropped units turn grey with a cross; each surviving unit shows $h \to \tilde h$ (its activation before and after the $1/(1-p)$ scale), and the table lists every number that goes into the output.` },
      { t: "sim", id: "dropout", caption: "Sampled sub-network outputs (dots), their running mean (line) and the deterministic eval-mode output (dashed)." },
      { t: "list", items: [
        String.raw`Press **Play** with $p = 0.5$. Individual outputs scatter widely, but the running mean settles onto the dashed eval-mode line — expectation preservation, observed. The side panel shows the exact average over all $2^{10} = 1024$ masks: it equals the eval output to every printed digit.`,
        String.raw`Set $p = 0.1$ and then $p = 0.8$ and press **New masks** + Play each time. The spread of the dots (the std readout) grows like $\sqrt{p/(1-p)}$: heavier dropout = noisier training signal.`,
        String.raw`Switch scaling to **Classic**: survivors are no longer scaled, so sample outputs are centred on $(1-p)\times$ the full-network output. Eval mode must then multiply by $(1-p)$; the clay dashed line shows what you would get if you forgot — a systematic bias, not noise.`,
        String.raw`Switch the output to **Sigmoid**. Now the running mean converges to the exact mask average, which is visibly *different* from the eval output: $\mathbb E[\sigma(z)] \ne \sigma(\mathbb E[z])$. The weight-scaling rule is exact only for parts of the network that are linear in the dropped units.`,
        "Toggle **Eval** mode: all units are active with the eval-time scale and the output is deterministic.",
      ] },
      { t: "p", text: String.raw`**L1 vs L2 simulator.** Each penalised optimum is drawn together with the norm ball that passes through it and the loss contour through it; the contour is tangent to the ball there. The lower chart is the **regularisation path**: each coefficient as a function of $\lambda$.` },
      { t: "sim", id: "regularization", caption: "Contours of the data loss, the L1 diamond and L2 circle at the current λ, and the full paths as λ grows." },
      { t: "list", items: [
        String.raw`Preset **Axis-aligned**, press Play. Around $\lambda \approx 1.25$ the L1 solution jumps onto the $w_1$-axis and its $w_2$ readout shows **0 exact**; the L2 point keeps both coordinates non-zero all the way to the end of the sweep. In the path chart the solid clay curve for $w_2$ hits zero and stays there; the dashed accent curves only bend towards zero.`,
        String.raw`To reproduce the worked example, choose **Axis-aligned** (centre $c = (2, 0.5)$, rotation 0°), set the elongation to 2 so that $A = \mathrm{diag}(1, 2)$, and move the λ slider to the grid value closest to 1.2 (it shows 1.211). The table reads L1 $= (0.789,\ 0)$ and L2 $= (0.905,\ 0.311)$ — the worked numbers, shifted slightly because $\lambda$ is 1.211 rather than 1.2.`,
        String.raw`**Correlated features** (the loss is 16 times stiffer along the $(1,1)$ direction, as when two input features move together): the lasso removes equal amounts from both coefficients until $w_2$ hits exactly 0 near $\lambda = 16$, then shrinks $w_1$ alone. Ridge behaves very differently — $w_2$ first *grows* from 1.0 to about 1.17 and the two coefficients converge towards each other (0.86 vs 0.79 at $\lambda \approx 13$): L2 spreads weight across correlated features, L1 picks one. Tick **elastic net** and slide α between 1 and 0 to morph between the two behaviours.`,
        String.raw`**Round bowl**: with $A = I$, L2 shrinks $c$ straight towards the origin ($w = c/(1+\lambda)$), while L1 subtracts $\lambda$ from each coordinate — the smaller one reaches zero first.`,
        "Drag the centre into another quadrant: the diamond's corners are on all four half-axes, so sparsity appears wherever the contours point.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      { t: "p", text: "Everything from scratch in NumPy: (1) a Monte-Carlo check that inverted dropout preserves the mean while classic dropout scales it by 1 − p; (2) a 64 → 256 → 10 ReLU network trained with mini-batch SGD on only **200** of the 1797 scikit-learn 8×8 digit images, with switches for L2 weight decay, an L1 proximal step, inverted dropout (mask reused in the backward pass) and early stopping with best-weight restoration; (3) a λ sweep chosen on the validation set. 500 images are used for validation and the remaining 1097 as a test set touched only for reporting." },
      { t: "code", lang: "python", title: "regularization_from_scratch.py", code: REG_NUMPY, output: REG_NUMPY_OUT },
      { t: "p", text: "Reading the output:" },
      { t: "list", items: [
        String.raw`Part 1: the inverted-dropout average over 100 000 masks matches $a$ to about $\pm 0.005$ (Monte-Carlo error); the classic version averages $(1-p)a$, which is why it needs test-time scaling.`,
        String.raw`Every model reaches **100% training accuracy** while validation accuracy is 92–94%: a 6–8 point generalization gap, the overfitting regime. L2 ($\lambda = 10^{-3}$) and dropout ($p = 0.5$) both lower the validation loss (0.221 → 0.209 and 0.206) and raise validation and test accuracy by 0.4–0.8 points. These are honest, modest gains — on a clean, easy dataset the baseline is already decent.`,
        String.raw`**L1** with $\lambda = 10^{-3}$ sets **89.1% of all weights exactly to zero** (soft-thresholding at work) for a cost of about one point of accuracy: a much smaller model nearly as accurate. Plain subgradient descent would leave those weights at tiny non-zero values.`,
        String.raw`**Early stopping** (patience 20 on validation loss) stopped after 154 of 300 epochs and restored epoch 134: the same accuracy as the baseline for half the compute. Here the validation loss is flat rather than rising, so early stopping saves time rather than accuracy — a common outcome when other regularization is light.`,
        String.raw`The **λ sweep** is the U-curve in action: validation accuracy peaks at $\lambda = 10^{-3}$ (0.938); at $\lambda = 10^{-1}$ even the training accuracy collapses to 0.770 — underfitting. Only after choosing $\lambda$ on validation do we look at the test set (0.946).`,
        "Biases are excluded from the decay (they do not cause overfitting and decaying them can hurt), and the learning rate 0.3 / 300 epochs were chosen so the baseline converges to zero training error.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: String.raw`Three parts. **Part 1**: nn.Dropout is inverted dropout, and model.eval() turns it into the identity. **Part 2**: the Adam-vs-AdamW measurement quoted in the maths section. **Part 3**: the main experiment — a wider 64 → 512 → 512 → 10 MLP trained with SGD + momentum on the same 200-image training set, comparing a baseline with L2 weight decay, dropout, early stopping, **data augmentation** (random rotations up to ±12°, scaling ±10% and translations, implemented with affine_grid / grid_sample — no flips, because a mirrored digit is a different symbol) and augmentation + dropout. Each row is the mean of 3 seeds.` },
          { t: "code", lang: "python", title: "regularization_pytorch.py", code: REG_TORCH, output: REG_TORCH_OUT },
          { t: "table", head: ["Method (mean of 3 seeds)", "Train acc", "Val acc", "Test acc", "Gap (train − val)"], rows: [
            ["Baseline", "1.000", "0.914", "0.940", "8.6 pts"],
            ["L2 / weight decay 5e-3", "1.000", "0.922", "0.945", "7.8 pts"],
            ["Dropout p = 0.5", "1.000", "0.933", "0.949", "6.7 pts"],
            ["Early stopping (patience 20, ≈ 39 epochs)", "1.000", "0.917", "0.939", "8.3 pts"],
            ["Augmentation (rotate / scale / shift)", "1.000", "**0.945**", "**0.955**", "5.5 pts"],
            ["Augmentation + dropout 0.3", "1.000", "0.930", "0.949", "7.0 pts"],
          ] },
          { t: "list", items: [
            String.raw`**Augmentation wins** (+3.1 points validation, +1.5 test over the baseline): it addresses the actual cause — too few distinct training images. Note its training accuracy is measured on the un-augmented images.`,
            String.raw`**Dropout** is second (+1.9 val), **L2** gives a smaller but consistent improvement and the lowest validation loss (0.242).`,
            String.raw`**Early stopping** stopped after ≈ 39 epochs instead of 100 with roughly baseline accuracy; its validation numbers are slightly optimistic because the validation set was used to choose the epoch — the test column is the honest one.`,
            String.raw`**Regularizers do not simply add up**: augmentation + dropout is *worse* than augmentation alone. With augmentation already supplying noise, extra dropout pushes the model towards underfitting within 100 epochs. Combinations must be re-tuned together on validation data.`,
            "Part 2 shows why you should use torch.optim.AdamW (or a framework's decoupled decay) when you want weight decay with Adam: with torch.optim.Adam the weight_decay argument is an L2 penalty that Adam's normalisation largely cancels.",
          ] },
          { t: "callout", kind: "warning", title: "Excluding biases and norm parameters from decay", text: "The usual practice is to apply weight decay to weight matrices only. In PyTorch pass parameter groups to the optimizer, e.g. `[{'params': weights, 'weight_decay': 0.05}, {'params': biases_and_norm_params, 'weight_decay': 0.0}]`, choosing the groups by `p.ndim > 1`." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras exposes every technique as a layer or argument: augmentation layers (active only in training), `kernel_regularizer` for L1/L2 penalties, `Dropout` layers (inverted dropout, disabled automatically by `predict`/`evaluate`), and the `EarlyStopping` callback with `restore_best_weights=True`. Note the convention difference: `regularizers.L2(l)` adds $l\\sum w^2$ to the loss (gradient $2lw$), whereas PyTorch's `weight_decay=λ` corresponds to $\\frac{\\lambda}{2}\\|w\\|^2$ (gradient $\\lambda w$); the value 2.5e-3 below therefore matches PyTorch's 5e-3. For decoupled decay with Adam use `tf.keras.optimizers.AdamW`." },
          { t: "code", lang: "python", title: "regularization_keras.py (not executed in CI)", code: REG_KERAS, output: "Prints the number of epochs actually run before early stopping triggered (at most 300) and then [validation loss, validation accuracy] of the restored best weights. Numbers are not reproduced here because TensorFlow is not available in the verification environment." },
        ] },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "table", head: ["Technique", "What it does", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["L2 penalty / weight decay", String.raw`Adds $\frac\lambda2\|w\|^2$; shrinks weights by $(1-\eta\lambda)$ each step`, "Simple, smooth, cheap; Gaussian-prior interpretation; works with any architecture", "Never produces exact zeros; interacts with adaptive optimizers and with normalization layers", String.raw`Almost everything: $10^{-4}$ for ResNets with SGD; 0.01–0.1 with AdamW for transformers`],
        ["Decoupled weight decay (AdamW)", "Applies the decay directly to the weights, outside the adaptive step", "Decay strength independent of gradient history; easier to tune separately from η", "In PyTorch the decay is multiplied by the (scheduled) learning rate", "Default for transformers and most Adam-trained models"],
        ["L1 penalty (lasso)", String.raw`Adds $\lambda\|w\|_1$; soft-thresholding gives exact zeros`, "Sparse weights, feature selection, smaller models", "Non-differentiable at 0 (needs prox or subgradient); unstable choice among correlated features", "Sparse linear models, feature selection, sparsity-inducing pruning"],
        ["Elastic net", "Mix of L1 and L2 with ratio α", "Sparsity plus grouping of correlated features; strictly convex", "Two hyperparameters", "High-dimensional regression with correlated features"],
        ["Max-norm", String.raw`Projects each unit's incoming weights onto $\|w_j\| \le c$`, "Bounds weights even with large learning rates; pairs well with dropout", "Extra hyperparameter c; projection step after each update", "Dropout MLPs (Srivastava et al. 2014)"],
        ["Dropout", "Zeroes each activation with prob. p, scales survivors by 1/(1−p)", "Strong, cheap, architecture-agnostic; approximate ensemble", "Slower convergence; train/test discrepancy; awkward with BatchNorm", "MLP heads, transformers (p ≈ 0.1), RNN outputs"],
        ["DropConnect (Wan et al. 2013)", "Drops individual weights instead of activations", "Finer-grained noise", "More expensive; rarely better in practice", "Research, some RNN variants"],
        ["Spatial dropout (Tompson et al. 2015)", "Drops whole feature-map channels", "Effective for correlated conv activations", "Coarse noise", "CNNs, segmentation, 1-D conv on sequences"],
        ["DropPath / stochastic depth (Huang et al. 2016)", "Skips whole residual blocks at random during training", "Regularizes very deep nets and speeds training", "Only for residual architectures", "Vision transformers, deep ResNets, ConvNeXt"],
        ["Early stopping", "Stops when validation loss stops improving; restores best weights", "Free (saves compute), no hyperparameter except patience; implicit L2", "Uses validation data for selection; noisy curves need patience", "Nearly every training run"],
        ["Data augmentation", "Label-preserving random transforms of the inputs", "Usually the largest gain; encodes invariances", "Must be label-preserving; domain knowledge required; more epochs", "Vision (crops, flips, colour), audio (SpecAugment), some NLP"],
        ["Mixup / CutMix", "Trains on blends of two examples and their labels", "Smoother decision boundaries, better calibration", "Less interpretable training samples; can hurt on some tasks", "Image classification"],
        ["Label smoothing", String.raw`Targets $(1-\varepsilon)y + \varepsilon/K$`, "Discourages over-confidence; trivial to add", "Can hurt knowledge distillation and calibration-sensitive uses", "Image classification, machine translation (ε = 0.1)"],
        ["Noise injection", "Gaussian noise on inputs/weights/activations", "Equals ridge for linear models; simple", "Noise scale must be tuned", "Small data, denoising, robustness"],
      ] },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      { t: "list", items: [
        String.raw`**AlexNet** (Krizhevsky, Sutskever & Hinton, 2012) won ImageNet 2012 using dropout ($p = 0.5$) in its fully-connected layers, weight decay $5\times10^{-4}$ and heavy augmentation (random crops, horizontal flips, colour perturbation) — without them the 60-million-parameter network overfitted badly.`,
        String.raw`**ResNets** on ImageNet (He et al., 2016) are trained with SGD + momentum, weight decay $10^{-4}$, random crops and flips and essentially no dropout; BatchNorm and augmentation do most of the regularizing.`,
        String.raw`**Transformers**: the original Transformer (Vaswani et al., 2017) used dropout 0.1 and label smoothing 0.1; GPT-3 reports a weight decay of 0.1. Vision transformers rely on strong augmentation, mixup/CutMix and stochastic depth because they lack convolutional inductive biases.`,
        "**Speech recognition**: SpecAugment's time and frequency masking became a standard ingredient of end-to-end ASR training.",
        "**Sparse and interpretable models**: L1-regularized logistic regression and the lasso are used for feature selection in high-dimensional problems such as genomics and text classification with large vocabularies; L1-style penalties also drive weight pruning for compressed networks.",
        "**Uncertainty estimation**: Monte-Carlo dropout gives cheap approximate predictive uncertainty for active learning and safety-relevant predictions.",
        "**Every production training pipeline** uses early stopping or its equivalent — checkpoint selection by validation metric.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "list", items: [
        "**Regularization buys generalization with training fit.** Too much of it turns an overfitting model into an underfitting one (see the λ = 0.1 row of the sweep: training accuracy 0.77). It never fixes high bias.",
        "**Data beats penalties.** Augmentation (and more real data) improved results the most in our experiment because it addresses the cause of overfitting; penalties and dropout only constrain the model.",
        "**Effects are often modest and dataset-dependent.** In our runs L2 and dropout moved accuracy by 0.5–2 points; on larger, noisier problems their effect can be much bigger or negligible. Always measure on validation data with several seeds.",
        "**Interactions are real.** Dropout + BatchNorm (variance shift), L2 + BatchNorm (scale-invariant weights make L2 act mainly as an effective learning-rate increase), L2 + Adam (needs AdamW), augmentation + dropout (can over-regularize).",
        "**Dropout costs convergence speed**: the gradient is noisier, so more epochs are needed; it also complicates reproducibility and gradient checking (fix the mask).",
        "**Early stopping is nearly free** but couples the choice of epoch to the validation set; if the same set is also used for many other decisions, its estimate becomes optimistic.",
        "**L1 gives sparsity, but unstructured**: scattered zeros rarely speed up dense GPU kernels; structured sparsity (whole neurons or channels, e.g. group lasso) is needed for real speed-ups.",
        "**Hyperparameters multiply**: λ, p, patience, augmentation magnitudes and their interactions all need tuning; good defaults (weight decay 0.01–0.1 with AdamW, dropout 0.1 in transformers, standard augmentation recipes) save most of the search.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      { t: "list", items: [
        "**Forgetting `model.eval()` before validation or inference.** Why wrong: dropout keeps zeroing units, so predictions are random and on average worse, and BatchNorm updates its statistics with validation data. Fix: `model.eval()` + `torch.no_grad()` for evaluation, `model.train()` before the next training step.",
        "**Scaling twice (or not at all).** Writing `nn.Dropout(p)(h) / (1 - p)` double-scales, and multiplying by `(1 - p)` at test time with inverted dropout under-scales. Why wrong: the expectation is no longer preserved, so the test network sees systematically different activations. Fix: frameworks already implement inverted dropout — use them as-is.",
        "**Using `torch.optim.Adam(weight_decay=...)` and expecting weight decay.** Why wrong: that is an L2 gradient term divided by Adam's adaptive scale (the Part 2 output shows its strength becomes nearly irrelevant). Fix: `torch.optim.AdamW`.",
        "**Decaying biases and normalization parameters.** Why wrong: biases do not increase capacity in the way weights do, and decaying LayerNorm/BatchNorm gains pushes them towards zero, harming the scale the network needs. Fix: put them in a parameter group with `weight_decay=0`.",
        "**Expecting exact zeros from L1 trained with plain (sub)gradient descent / Adam.** Why wrong: the iterates oscillate around zero by about ηλ and never land on it. Fix: proximal (soft-threshold) steps, or threshold after training.",
        "**Augmentations that change the label**: horizontal flips on digits or text, vertical flips on scenes, large rotations that turn 6 into 9, crops that cut out the object. Fix: only use transformations under which the true label is invariant, and look at augmented samples.",
        "**Augmenting the validation or test set**, or computing normalization statistics on them. Why wrong: you no longer measure performance on the real data distribution. Fix: augmentation is a training-only operation (test-time augmentation is a separate, deliberate technique).",
        "**Placing dropout right before BatchNorm** (or using large p in conv layers followed by BN). Why wrong: the variance shift between train and test makes BN's running statistics wrong. Fix: dropout after the last BN layer, or small p.",
        "**Early stopping without restoring the best weights**, or with patience 0 on a noisy curve. Fix: `restore_best_weights=True` / save-best checkpoints, patience of several evaluations, monitor the loss rather than accuracy.",
        "**Tuning λ, p or the stopping epoch on the test set.** Why wrong: the test score becomes an optimistically biased estimate. Fix: tune on validation (or cross-validation), test once.",
        "**Comparing λ values across frameworks without converting.** Keras `L2(l)` adds $l\\|w\\|^2$; PyTorch `weight_decay=λ` corresponds to $\\frac{\\lambda}{2}\\|w\\|^2$; and in PyTorch's AdamW the per-step decay is `lr × weight_decay`. The same number means different strengths.",
        "**Regularizing an underfitting model.** If training error is high, adding dropout or decay makes it worse. Fix: diagnose with the learning curves from [Training basics](/learn/m03/training-basics) first.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is regularization and why is it needed?", a: "Any change to the training procedure intended to reduce generalization error (test error) rather than training error. High-capacity models can fit noise and peculiarities of the training set; regularization constrains the model (penalties), adds noise (dropout), adds data (augmentation) or limits optimization (early stopping) so that what is learned transfers to new data." },
      { t: "qa", level: "beginner", q: "What is the difference between L1 and L2 regularization?", a: String.raw`L2 adds $\frac\lambda2\sum w^2$; its gradient $\lambda w$ shrinks every weight proportionally and almost never makes one exactly zero. L1 adds $\lambda\sum|w|$; its (sub)gradient has constant magnitude $\lambda$, so weights whose data gradient is smaller than $\lambda$ become exactly zero — L1 produces sparse models. Geometrically, the L1 ball is a diamond with corners on the axes where loss contours tend to touch it.` },
      { t: "qa", level: "beginner", q: "How does dropout behave differently during training and inference?", a: String.raw`During training each unit is zeroed with probability $p$ and (inverted dropout) survivors are multiplied by $1/(1-p)$; a new mask is drawn for every example and step. During inference dropout is turned off and nothing is scaled, because the training-time scaling already makes the expected activation equal to the full activation. In PyTorch this switch is model.train() / model.eval().` },
      { t: "qa", level: "beginner", q: "What is early stopping and what does 'patience' mean?", a: "Monitoring the validation loss during training and stopping once it has not improved for a number of evaluations (the patience), then restoring the weights from the best evaluation. Patience is needed because validation curves are noisy; a single non-improving epoch does not mean the optimum has passed." },
      { t: "qa", level: "intermediate", q: "Derive why inverted dropout scales by 1/(1 − p).", a: String.raw`With $m_i \sim \text{Bernoulli}(1-p)$, $\mathbb E[m_ih_i] = (1-p)h_i$. To make the training-time activation unbiased for the test-time (full) activation we divide by $(1-p)$: $\mathbb E[m_ih_i/(1-p)] = h_i$. Because the next layer is linear in these activations, the expected pre-activation is unchanged, so the test network needs no modification. The cost is variance $\frac{p}{1-p}h_i^2$ per unit.` },
      { t: "qa", level: "intermediate", q: "Why is weight decay not the same as L2 regularization in Adam, and what does AdamW change?", a: String.raw`In SGD, adding $\lambda w$ to the gradient multiplies weights by $(1-\eta\lambda)$ — identical to decay. Adam divides the whole gradient, including $\lambda w$, by $\sqrt{\hat v}+\epsilon$, so weights with large historical gradients are regularized less and the effective strength depends on gradient statistics (with no data gradient the step is ≈ η·sign(w) regardless of λ). AdamW applies $w \leftarrow w - \eta\lambda w$ separately from the adaptive step, restoring uniform proportional decay and decoupling λ from the learning-rate tuning.` },
      { t: "qa", level: "intermediate", q: "Why does L1 produce exact zeros only with a proximal step, not with plain gradient descent?", a: String.raw`A subgradient step moves a weight by $\eta\lambda\,\mathrm{sign}(w)$; near zero it overshoots back and forth and lands on exactly 0 only by coincidence. The proximal step solves $\min_w \frac12(w-v)^2 + \eta\lambda|w|$ exactly, which is soft-thresholding $\mathrm{sign}(v)\max(|v|-\eta\lambda, 0)$: any coordinate with $|v| \le \eta\lambda$ is set to exactly zero.` },
      { t: "qa", level: "intermediate", q: "How do you choose the regularization strength?", a: "Treat it as a hyperparameter: train models over a log-spaced grid (or random/Bayesian search) and select the value with the best validation metric (or cross-validation score when data are scarce); then evaluate the selected model once on the test set. Re-tune when changing model size, data size, optimizer or other regularizers, since they interact." },
      { t: "qa", level: "advanced", q: "In what sense is dropout an ensemble method, and when is the test-time approximation exact?", a: String.raw`A network with $n$ droppable units defines $2^n$ thinned sub-networks sharing weights; each training step trains one sampled sub-network. The ideal predictor averages all of them. Using the full network with expectation-preserving scaling (weight-scaling rule) is exact for quantities linear in the dropped activations, and for a single dropout layer before a softmax it equals the normalised geometric mean of the sub-networks' predictions. With further non-linearities it is an approximation because $\mathbb E[f(z)] \ne f(\mathbb E[z])$; MC dropout estimates the true average by sampling.` },
      { t: "qa", level: "advanced", q: "Show that early stopping behaves like L2 regularization for a quadratic loss.", a: String.raw`Approximate the loss by $\frac12(w-w^\star)^\top H(w-w^\star)$ with eigenvalues $h_i$, start GD at 0 with step $\eta$. In the eigenbasis $w_i^{(t)} = [1-(1-\eta h_i)^t]w^\star_i$. L2 gives $\frac{h_i}{h_i+\lambda}w^\star_i$. Both pass high-curvature directions almost unchanged and suppress low-curvature ones. Match the *remaining* fractions: $(1-\eta h_i)^t = \lambda/(h_i+\lambda)$. Taking logs, $t\log(1-\eta h_i) = -\log(1 + h_i/\lambda)$, and for small $\eta h_i$ and $h_i/\lambda$ both logs are linear: $t\,\eta h_i \approx h_i/\lambda$, i.e. $\lambda \approx 1/(\eta t)$. The number of steps acts as an inverse regularization strength.` },
      { t: "qa", level: "advanced", q: "Why can L2 regularization on weights followed by BatchNorm not constrain capacity, yet still matter?", a: String.raw`BatchNorm makes the layer's output invariant to the scale of the preceding weights: $\mathrm{BN}(\alpha Wx) = \mathrm{BN}(Wx)$ for $\alpha > 0$. Shrinking $W$ therefore does not change the function, so L2 does not restrict what the network can represent. But the gradient of a scale-invariant function scales as $1/\|W\|$, so smaller weights mean larger effective steps: weight decay acts as an effective learning-rate increase that keeps training dynamics in a useful regime. Tuning it still matters, just for a different reason.` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "With inverted dropout and drop probability p = 0.25, by what factor is a kept activation multiplied during training?", answer: 1.3333, tolerance: 0.001, explain: "1/(1 − p) = 1/0.75 = 1.3333." },
        { kind: "numeric", q: String.raw`SGD with learning rate $\eta = 0.1$ and L2 strength $\lambda = 10^{-3}$: what is the per-step shrinkage factor applied to each weight?`, answer: 0.9999, tolerance: 0.00001, explain: String.raw`$1 - \eta\lambda = 1 - 10^{-4} = 0.9999$.` },
        { kind: "numeric", q: String.raw`Compute the soft-thresholding $S_{0.5}(-1.2)$ (the L1 proximal step with threshold 0.5).`, answer: -0.7, tolerance: 0.001, explain: String.raw`$\mathrm{sign}(-1.2)\max(1.2 - 0.5, 0) = -0.7$. A value with $|v| \le 0.5$ would have become exactly 0.` },
        { kind: "numeric", q: String.raw`Label smoothing with $\varepsilon = 0.1$ and $K = 5$ classes: what is the target probability of the true class?`, answer: 0.92, tolerance: 0.001, explain: String.raw`$(1-\varepsilon) + \varepsilon/K = 0.9 + 0.02 = 0.92$; every other class gets 0.02.` },
        { q: "A dropout model gets much worse validation accuracy than expected, and the numbers change every time you evaluate. The most likely cause?", options: ["λ is too small", "The model is evaluated in training mode, so dropout is still active", "The learning rate is too low", "Inverted dropout needs test-time scaling"], answer: 1, explain: "Random, degraded evaluations are the signature of dropout left on. Call model.eval() (and torch.no_grad()) for evaluation." },
        { q: "Which penalty can set weights exactly to zero at the optimum?", options: ["L2 only", "L1 (and elastic net with α > 0)", "Neither — only pruning can", "Both, equally often"], answer: 1, explain: "The L1 subdifferential at 0 is [−λ, λ], so 0 is optimal whenever the data gradient is at most λ in magnitude; L2's gradient λw vanishes at 0 and only shrinks." },
        { q: "You train with torch.optim.Adam(params, lr=1e-3, weight_decay=0.01). What are you actually getting?", options: ["Decoupled weight decay identical to AdamW", "An L2 penalty gradient that is rescaled by Adam's adaptive denominator", "No regularization at all", "L1 regularization"], answer: 1, explain: "torch.optim.Adam adds weight_decay·w to the gradient before the moment estimates — L2, not decoupled decay. Use AdamW for the latter." },
        { q: "For an 8×8 handwritten-digit dataset, which augmentation is NOT label-preserving?", options: ["Translation by one pixel", "Rotation by ±10°", "Horizontal flip", "Small scaling"], answer: 2, explain: "A mirrored digit is a different (or no) symbol; flips are only safe when the class is mirror-symmetric, as for most natural photos." },
        { kind: "numeric", q: "A layer has 10 units subject to dropout. How many distinct thinned sub-networks does it define?", answer: 1024, tolerance: 0, explain: "Each unit is either kept or dropped: 2¹⁰ = 1024 masks, all sharing the same weights." },
      ] },
      { t: "exercise", title: "Dropout, weight decay and soft-thresholding by hand", prompt: String.raw`(a) A hidden layer has activations $h = (1.5, 0.4, 2.0)$, $p = 0.2$, and the mask $m = (1, 0, 1)$ is drawn. Give $\tilde h$ for inverted dropout and the eval-mode output of the same layer. (b) With $\eta = 0.01$ and $\lambda = 5\times10^{-4}$ (SGD weight decay) and no data gradient, how many steps does it take for a weight to halve? (c) One ISTA step with $\eta = 0.5$, $\lambda = 0.4$ starts at $w = (0.3, -1.0)$ with data gradient $g = (0.2, -0.6)$. Compute the new $w$.`, hint: String.raw`(b) solve $(1-\eta\lambda)^n = 1/2$. (c) gradient step first, then soft-threshold with $\tau = \eta\lambda$.`, solution: [
        { t: "steps", items: [
          { title: "(a) Inverted dropout", tex: String.raw`\tilde h = \frac{(1.5,\ 0,\ 2.0)}{0.8} = (1.875,\ 0,\ 2.5),\qquad \text{eval: } (1.5,\ 0.4,\ 2.0)\ \text{unchanged}` },
          { title: "(b) Half-life of weight decay", tex: String.raw`n = \frac{\ln 0.5}{\ln(1 - 0.01\cdot5\times10^{-4})} = \frac{-0.693147}{-5.0000\times10^{-6}} \approx 138{,}629 \text{ steps}` },
          { title: "(c) Gradient step", tex: String.raw`v = w - \eta g = (0.3 - 0.1,\ -1.0 + 0.3) = (0.2,\ -0.7)` },
          { title: "(c) Soft-threshold with τ = 0.2", tex: String.raw`w_{\text{new}} = \big(\mathrm{sign}(0.2)\max(0.2-0.2, 0),\ \mathrm{sign}(-0.7)\max(0.7-0.2, 0)\big) = (0,\ -0.5)` },
        ] },
        { t: "p", text: "Part (b) shows why weight decay is a gentle force over a single step but a strong one over a full training run: a decay that removes only 0.0005% per step halves an unsupported weight in about 139 thousand steps, which is a typical training length." },
      ] },
      { t: "exercise", title: "Implement and gradient-check a dropout layer", prompt: "Write `dropout_forward(h, p, rng)` (inverted dropout, returning the output and the mask) and `dropout_backward(dout, mask)`. Verify the backward pass with central finite differences on a small loss, holding the mask fixed, and confirm that dropped units receive zero gradient.", hint: "The backward pass of y = h ⊙ mask is just dout ⊙ mask. For the gradient check the random mask must not be redrawn between the two perturbed evaluations.", solution: [
        { t: "code", lang: "python", title: "dropout_layer_check.py", code: EX_DROPOUT, output: EX_DROPOUT_OUT },
        { t: "p", text: "The analytic and numerical gradients agree to about 1e-9 (finite-difference rounding error). Holding the mask fixed is essential: if each forward call drew a new mask, the finite difference would measure the change caused by the new mask rather than by the perturbation. This is also why frameworks disable dropout (or fix its seed) when gradient checking. The kept fraction 0.792 differs from 1 − p = 0.7 because only 24 activations were masked: the sample fraction has standard deviation √(0.3·0.7/24) ≈ 0.09, and only its expectation is 0.7." },
      ] },
      { t: "exercise", title: "Where does the regularization go?", prompt: "You are fine-tuning a 12-layer transformer classifier with AdamW on 5,000 labelled sentences. Training accuracy reaches 99% after two epochs while validation accuracy peaks at 86% in epoch 1 and then declines. List, in order of what you would try first, four regularization changes and explain what each targets.", solution: [
        { t: "list", ordered: true, items: [
          "**Early stopping / checkpoint selection** on validation loss — the validation peak at epoch 1 says the model starts memorising quickly; restoring the best checkpoint is free.",
          "**Weight decay** in AdamW (e.g. 0.01–0.1, excluding biases and LayerNorm parameters) and a lower learning rate, which limit how far the weights drift from the pretrained solution.",
          "**Dropout** at the usual 0.1 in attention and feed-forward sub-layers (check it is not accidentally disabled), possibly a higher rate on the classification head.",
          "**Data-side regularization**: text augmentation (back-translation, token masking) or label smoothing 0.1 to reduce over-confidence; collecting more labelled data if possible. Re-tune each change on the validation set, not the test set.",
        ] },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      { t: "list", items: [
        String.raw`**Generalization gap** = validation/test error − training error; regularization reduces it, possibly at a small cost in training error. Regularized objective: $J = \hat R + \lambda\Omega$.`,
        String.raw`**L2 / weight decay**: gradient $+\lambda w$; SGD update $w \leftarrow (1-\eta\lambda)w - \eta\nabla\hat R$. Shrinks each Hessian eigen-direction by $h_i/(h_i+\lambda)$. Gaussian prior.`,
        String.raw`**Adam vs AdamW**: L2 inside Adam is rescaled by $1/(\sqrt{\hat v}+\epsilon)$; AdamW applies the decay directly. Use AdamW for weight decay with adaptive optimizers.`,
        String.raw`**L1**: zero is optimal iff $|\partial\hat R/\partial w_j| \le \lambda$; proximal step = soft-thresholding $S_\tau(v) = \mathrm{sign}(v)\max(|v|-\tau, 0)$ with $\tau = \eta\lambda$ — exact zeros, sparsity. Laplace prior.`,
        String.raw`**Elastic net**: $\lambda(\alpha\|w\|_1 + \frac{1-\alpha}2\|w\|_2^2)$ — sparsity plus grouping of correlated features.`,
        String.raw`**Dropout**: $\tilde h = m\odot h/(1-p)$, $m_i \sim \text{Bernoulli}(1-p)$; $\mathbb E[\tilde h] = h$, $\operatorname{Var}[\tilde h_i] = \frac{p}{1-p}h_i^2$. Off at test time. Ensemble of $2^n$ weight-sharing sub-networks; exact averaging only for linear parts. Typical $p$: 0.5 (MLP hidden), 0.2 (inputs), 0.1 (transformers); prefer spatial dropout in CNNs; beware dropout before BatchNorm.`,
        String.raw`**Early stopping**: monitor validation loss, patience, restore best weights; behaves like L2 with $\lambda \approx 1/(\eta t)$.`,
        "**Data augmentation**: label-preserving transforms (crops, flips when valid, rotations, colour; SpecAugment for audio; back-translation for text); mixup/CutMix/Cutout. Largest gain in our digits experiment (val 0.914 → 0.945).",
        String.raw`**Others**: noise injection (= ridge for linear models), max-norm projection $\|w_j\| \le c$, label smoothing $(1-\varepsilon)y + \varepsilon/K$.`,
        "**Choose every strength on validation data**, report test once; regularizers interact, so re-tune combinations together.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      { t: "p", text: "Regularization is one of three pillars that make deep networks trainable and reliable; the other two are covered next in this module, and the third link closes the loop on the optimizer interaction:" },
      { t: "list", items: [
        "[Residual and Skip Connections](/learn/m07/residual-connections) — how very deep networks are made trainable at all; stochastic depth (DropPath) from this lesson operates on residual blocks.",
        "[Batch, Layer and Group Normalization](/learn/m07/normalization) — normalization has its own mild regularizing effect and interacts with dropout and weight decay as discussed here.",
        "[Momentum and Adaptive Optimizers](/learn/m06/momentum-adaptive) — the Adam update in detail, to fully understand why AdamW decouples weight decay.",
        "[Convolution](/learn/m08/convolution) and [Transfer Learning](/learn/m12/transfer-learning) — where data augmentation and pretrained weights become the dominant regularizers in vision.",
        "[Experimentation](/learn/m17/experimentation) — running the validation sweeps and multi-seed comparisons this lesson relied on, rigorously.",
      ] },
    ],
  },
};

export default lesson;
