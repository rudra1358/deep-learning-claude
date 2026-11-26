import type { Lesson } from "@/lib/types";

const BPTT_NUMPY = String.raw`import numpy as np


class VanillaRNN:
    """h_t = tanh(W_xh x_t + W_hh h_(t-1) + b),  y_t = W_hy h_t + b_y,
    loss L = sum_t 0.5 * mean_over_batch ||y_t - target_t||^2  (a loss at EVERY step)."""

    def __init__(self, n_in, n_h, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W_xh": rng.normal(0, 1 / np.sqrt(n_in), (n_h, n_in)),
            "W_hh": rng.normal(0, 1 / np.sqrt(n_h), (n_h, n_h)),   # spectral radius ~ 1 at init
            "b": np.zeros(n_h),
            "W_hy": rng.normal(0, 1 / np.sqrt(n_h), (n_out, n_h)),
            "b_y": np.zeros(n_out),
        }

    def forward(self, xs, h0=None):
        """xs: (T, B, n_in). Caches every h_t: the backward pass needs all of them (O(T) memory)."""
        T, B, _ = xs.shape
        p = self.p
        self.xs = xs
        self.hs = [np.zeros((B, p["W_hh"].shape[0])) if h0 is None else h0]   # hs[0] = h_0
        ys = []
        for t in range(T):
            h = np.tanh(xs[t] @ p["W_xh"].T + self.hs[-1] @ p["W_hh"].T + p["b"])
            self.hs.append(h)
            ys.append(h @ p["W_hy"].T + p["b_y"])
        self.ys = np.stack(ys)
        return self.ys

    def loss(self, targets):
        return 0.5 * np.sum((self.ys - targets) ** 2) / targets.shape[1]

    def backward(self, targets, k=None):
        """Backpropagation through time. k = None: full BPTT.
        k = int: truncated — the loss at step s sends gradient back through at most k steps (s-k+1 .. s)."""
        p, xs, hs = self.p, self.xs, self.hs
        T, B = targets.shape[:2]
        g = {name: np.zeros_like(v) for name, v in p.items()}
        e = (self.ys - targets) / B                              # dL/dy_t for every t, shape (T, B, n_out)
        g["W_hy"] = np.einsum("tbo,tbh->oh", e, np.stack(hs[1:]))
        g["b_y"] = e.sum(axis=(0, 1))
        dh_from_y = e @ p["W_hy"]                               # direct gradient of each loss term on its own h_t

        def accumulate(t, dh):                                   # local gradients at step t; returns dL/dh_(t-1)
            dz = dh * (1 - hs[t + 1] ** 2)                       # through tanh: diag(1 - h_t^2)
            g["W_hh"] += dz.T @ hs[t]                            # shared weights: contributions ADD over time
            g["W_xh"] += dz.T @ xs[t]
            g["b"] += dz.sum(axis=0)
            return dz @ p["W_hh"]                                # dL/dh_(t-1) = W_hh^T dz  (row-vector form)

        if k is None:                                            # one right-to-left sweep, O(T)
            dh_next = np.zeros_like(hs[0])
            for t in reversed(range(T)):
                dh_next = accumulate(t, dh_from_y[t] + dh_next)  # gradient from y_t PLUS from h_(t+1)
            self.dh0 = dh_next
        else:                                                    # each loss term separately, k steps deep
            for s in range(T):
                dh = dh_from_y[s]
                for t in range(s, max(-1, s - k), -1):
                    dh = accumulate(t, dh)
        return g


def gradient_check(model, xs, targets, eps=1e-6):
    model.forward(xs)
    grads = model.backward(targets)
    worst = 0.0
    for name, W in model.p.items():
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + eps; model.forward(xs); lp = model.loss(targets)
            W[idx] = old - eps; model.forward(xs); lm = model.loss(targets)
            W[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - grads[name][idx]) / max(1e-12, abs(num) + abs(grads[name][idx])))
    return worst


def delay_task(rng, T, B, delay):
    """Target at step t is the input from "delay" steps earlier (0 before that)."""
    xs = rng.normal(size=(T, B, 1))
    ys = np.zeros_like(xs)
    ys[delay:] = xs[:-delay]
    return xs, ys


# ---- 1. gradient check of full BPTT against central differences ----
rng = np.random.default_rng(0)
small = VanillaRNN(n_in=1, n_h=4, n_out=1, seed=1)
xs, ys = delay_task(rng, T=6, B=3, delay=2)
print(f"gradient check (full BPTT), worst relative error: {gradient_check(small, xs, ys):.1e}")

# ---- 2. truncation drops terms: compare truncated gradients with the full one ----
model = VanillaRNN(1, 16, 1, seed=2)
xs, ys = delay_task(rng, T=30, B=8, delay=3)
model.forward(xs)
full = model.backward(ys)["W_hh"]
for k in [1, 2, 4, 8, 30]:
    model.forward(xs)
    trunc = model.backward(ys, k=k)["W_hh"]
    print(f"k={k:2d}: ||dW_hh(trunc) - dW_hh(full)|| / ||dW_hh(full)|| = {np.linalg.norm(trunc - full) / np.linalg.norm(full):.2e}")


# ---- 3. training on a delay-3 echo task: full BPTT vs truncated windows ----
def adam_train(k, steps=1500, lr=0.01, clip=1.0, seed=0):
    rng = np.random.default_rng(seed)
    net = VanillaRNN(1, 16, 1, seed=seed)
    m = {n: np.zeros_like(v) for n, v in net.p.items()}
    v = {n: np.zeros_like(w) for n, w in net.p.items()}
    for step in range(1, steps + 1):
        xs, ys = delay_task(rng, T=20, B=32, delay=3)
        net.forward(xs)
        g = net.backward(ys, k=k)
        norm = np.sqrt(sum(np.sum(x ** 2) for x in g.values()))
        scale = min(1.0, clip / (norm + 1e-6))                   # clip the global gradient norm
        for n in net.p:
            gi = g[n] * scale
            m[n] = 0.9 * m[n] + 0.1 * gi
            v[n] = 0.999 * v[n] + 0.001 * gi ** 2
            mh, vh = m[n] / (1 - 0.9 ** step), v[n] / (1 - 0.999 ** step)
            net.p[n] -= lr * mh / (np.sqrt(vh) + 1e-8)
    xs, ys = delay_task(np.random.default_rng(99), T=20, B=500, delay=3)
    pred = net.forward(xs)
    return np.mean((pred[3:] - ys[3:]) ** 2)                    # test MSE after the warm-up steps


print("delay-3 echo task, test MSE (predicting 0 gives 1.0):")
for k in [None, 4, 3, 2]:
    print(f"  {'full BPTT' if k is None else f'truncated k={k}':16s} {adam_train(k):.4f}")`;

const BPTT_NUMPY_OUT = String.raw`gradient check (full BPTT), worst relative error: 3.8e-09
k= 1: ||dW_hh(trunc) - dW_hh(full)|| / ||dW_hh(full)|| = 6.05e-01
k= 2: ||dW_hh(trunc) - dW_hh(full)|| / ||dW_hh(full)|| = 3.57e-01
k= 4: ||dW_hh(trunc) - dW_hh(full)|| / ||dW_hh(full)|| = 7.15e-02
k= 8: ||dW_hh(trunc) - dW_hh(full)|| / ||dW_hh(full)|| = 1.15e-02
k=30: ||dW_hh(trunc) - dW_hh(full)|| / ||dW_hh(full)|| = 8.00e-16
delay-3 echo task, test MSE (predicting 0 gives 1.0):
  full BPTT        0.0017
  truncated k=4    0.0016
  truncated k=3    0.9558
  truncated k=2    0.9736`;

const BPTT_AUTOGRAD = String.raw`import torch

torch.set_printoptions(precision=6)
# The 2-unit network of the worked example (same numbers as the forward-pass lesson)
W_xh = torch.tensor([[0.5], [-0.3]], dtype=torch.float64, requires_grad=True)
W_hh = torch.tensor([[0.4, -0.2], [0.1, 0.6]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.0, 0.1], dtype=torch.float64, requires_grad=True)
W_hy = torch.tensor([[0.8, -0.5]], dtype=torch.float64, requires_grad=True)
b_y = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
xs = torch.tensor([[1.0], [0.5], [-1.0]], dtype=torch.float64)
targets = torch.tensor([0.5, 0.0, -0.5], dtype=torch.float64)

h = torch.zeros(2, dtype=torch.float64, requires_grad=True)    # h_0 (a leaf, so we can read dL/dh_0)
h0, hs, loss = h, [], 0.0
for t in range(3):
    h = torch.tanh(W_xh @ xs[t] + W_hh @ h + b)
    h.retain_grad()                                  # keep dL/dh_t for inspection (non-leaf tensors drop it)
    hs.append(h)
    y = W_hy @ h + b_y
    loss = loss + 0.5 * (y - targets[t]).pow(2).sum()
loss.backward()                                      # autograd = backpropagation through time

print("L =", round(loss.item(), 6))
for t, ht in enumerate(hs, 1):
    print(f"dL/dh_{t} =", ht.grad.numpy().round(6))
print("dL/dh_0 =", h0.grad.numpy().round(6))
print("dL/dW_hh =", W_hh.grad.numpy().round(6).tolist())
print("dL/dW_xh =", W_xh.grad.numpy().round(6).ravel().tolist())
print("dL/db    =", b.grad.numpy().round(6).tolist())
print("dL/dW_hy =", W_hy.grad.numpy().round(6).tolist(), " dL/db_y =", b_y.grad.numpy().round(6).tolist())`;

const BPTT_AUTOGRAD_OUT = String.raw`L = 0.152594
dL/dh_1 = [ 0.169417 -0.305943]
dL/dh_2 = [ 0.458731 -0.334791]
dL/dh_3 = [ 0.152126 -0.095079]
dL/dh_0 = [ 0.023893 -0.203062]
dL/dW_hh = [[0.232157, -0.089818], [-0.189126, 0.075207]]
dL/dW_xh = [0.178561, -0.375851]
dL/db    = [0.641773, -0.706958]
dL/dW_hy = [[0.203471, -0.008508]]  dL/db_y = [0.772691]`;

const BPTT_NORMS = String.raw`import numpy as np

H, T, N_SEQ = 32, 100, 20          # hidden size, sequence length, sequences averaged over
KS = [1, 5, 10, 20, 50, 99]        # distances k at which we report ||dL/dh_(T-k)|| / ||dL/dh_T||


def recurrent_matrix(kind, rho, rng):
    if kind == "orthogonal":
        q, r = np.linalg.qr(rng.normal(size=(H, H)))
        return q * np.sign(np.diag(r))             # a random orthogonal matrix: all |eigenvalues| = 1
    W = rng.normal(0, 1 / np.sqrt(H), (H, H))
    return W * (rho / max(abs(np.linalg.eigvals(W))))   # rescale to the requested spectral radius


def profile(act, kind, rho, x_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    W_hh = recurrent_matrix(kind, rho, rng)
    W_xh = rng.normal(0, 1, (H, 1))
    w_out = rng.normal(0, 1 / np.sqrt(H), H)
    f = {"tanh": np.tanh, "linear": lambda z: z}[act]
    log_ratios, slopes, factors = [], [], []
    for _ in range(N_SEQ):
        xs = rng.normal(0, x_scale, (T, 1))          # input sequence
        h, hs, zs = np.zeros(H), [], []
        for x in xs:                                     # forward pass
            z = W_hh @ h + W_xh @ x
            h = f(z)
            zs.append(z); hs.append(h)
        y = w_out @ hs[-1]
        g = w_out * (y - 1.0)                            # dL/dh_T for L = 0.5 (y_T - 1)^2
        norms = [np.linalg.norm(g)]
        for t in range(T - 1, 0, -1):                    # BPTT: dL/dh_(t-1) = W_hh^T diag(f'(z_t)) dL/dh_t
            d = {"tanh": 1 - hs[t] ** 2, "linear": np.ones(H)}[act]
            slopes.append(d.mean())
            factors.append(np.linalg.norm(W_hh.T * d, 2))   # ||W_hh^T diag(d)||_2, the per-step factor
            g = W_hh.T @ (d * g)
            norms.append(np.linalg.norm(g))
        log_ratios.append(np.log10(np.array(norms) / norms[0]))
    mean_log = np.mean(log_ratios, axis=0)               # average of log10 ratios = geometric mean
    rho_eff = max(abs(np.linalg.eigvals(W_hh)))
    return rho_eff, np.mean(slopes), np.mean(factors), [mean_log[k] for k in KS]


print("geometric-mean ratio ||dL/dh_(T-k)|| / ||dL/dh_T|| over 20 random sequences, T = 100, hidden 32")
print(f"{'activation / W_hh / inputs':30s} {'rho':>4s} {'mean slope':>10s} {'mean factor':>11s}  " + " ".join(f"{'k=' + str(k):>7s}" for k in KS))
configs = [("tanh", "random", r, 1.0) for r in (0.5, 0.9, 1.0, 1.5, 2.5, 5.0)] + [
    ("tanh", "random", 1.0, 0.1),          # small inputs: tanh stays in its linear region
    ("tanh", "orthogonal", 1.0, 1.0),
    ("tanh", "orthogonal", 1.0, 0.1),
    ("linear", "random", 0.9, 1.0),        # no activation: the product is exactly W_hh^k
    ("linear", "random", 1.1, 1.0),
]
for act, kind, rho, x_scale in configs:
    r, s, fac, logs = profile(act, kind, rho, x_scale)
    name = f"{act} / {kind} / x~N(0,{x_scale:g})"
    print(f"{name:30s} {r:4.2f} {s:10.3f} {fac:11.3f}  " + " ".join(f"{10 ** v:7.0e}" for v in logs))`;

const BPTT_NORMS_OUT = String.raw`geometric-mean ratio ||dL/dh_(T-k)|| / ||dL/dh_T|| over 20 random sequences, T = 100, hidden 32
activation / W_hh / inputs      rho mean slope mean factor      k=1     k=5    k=10    k=20    k=50    k=99
tanh / random / x~N(0,1)       0.50      0.676       0.696    4e-01   4e-03   2e-05   3e-10   9e-25   2e-48
tanh / random / x~N(0,1)       0.90      0.617       1.208    6e-01   5e-02   3e-03   7e-06   6e-14   5e-27
tanh / random / x~N(0,1)       1.00      0.599       1.327    7e-01   7e-02   6e-03   4e-05   3e-12   2e-23
tanh / random / x~N(0,1)       1.50      0.505       1.878    9e-01   3e-01   8e-02   6e-03   3e-06   1e-11
tanh / random / x~N(0,1)       2.50      0.358       2.868    1e+00   2e+00   2e+00   5e+00   5e+01   6e+02
tanh / random / x~N(0,1)       5.00      0.187       4.960    2e+00   5e+00   3e+01   1e+03   4e+07   7e+14
tanh / random / x~N(0,0.1)     1.00      0.959       1.604    9e-01   4e-01   2e-01   8e-02   2e-02   2e-03
tanh / orthogonal / x~N(0,1)   1.00      0.587       0.996    6e-01   1e-01   2e-02   3e-04   9e-10   8e-19
tanh / orthogonal / x~N(0,0.1) 1.00      0.932       1.000    9e-01   7e-01   5e-01   3e-01   4e-02   2e-03
linear / random / x~N(0,1)     0.90      1.000       1.510    8e-01   3e-01   1e-01   3e-02   1e-03   7e-06
linear / random / x~N(0,1)     1.10      1.000       1.846    1e+00   9e-01   8e-01   2e+00   3e+01   3e+03`;

const BPTT_REMEMBER = String.raw`import torch
import torch.nn as nn

torch.set_num_threads(1)          # tiny model: extra threads only add overhead


def make_batch(B, T, gen):
    """The bit to remember is x_1 = +1 or -1; x_2..x_T are N(0, 1) distractors.
    The label (read only at the LAST step) is 1 if x_1 was +1."""
    x = torch.randn(B, T, 1, generator=gen)
    bit = torch.randint(0, 2, (B,), generator=gen)
    x[:, 0, 0] = bit.float() * 2 - 1
    return x, bit.float()


class Remember(nn.Module):
    def __init__(self, cell, hidden=32):
        super().__init__()
        self.rnn = cell(1, hidden, batch_first=True)
        if cell is nn.LSTM:                       # standard LSTM trick: forget-gate bias = 1
            with torch.no_grad():
                self.rnn.bias_ih_l0[hidden:2 * hidden].fill_(1.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)                      # out: (B, T, hidden)
        return self.head(out[:, -1]).squeeze(-1)  # classify from the last hidden state only


def train(cell, T, seed, steps=500, B=64):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Remember(cell)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(B, T, gen)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against exploding steps
        opt.step()
    x, y = make_batch(1000, T, torch.Generator().manual_seed(123))   # fresh test set
    with torch.no_grad():
        return ((model(x) > 0).float() == y).float().mean().item()


CELLS = [nn.RNN]                  # add nn.LSTM here to compare (see text)
print("test accuracy after 500 Adam steps (3 seeds); chance = 0.5")
for cell in CELLS:
    for T in [5, 10, 20, 40]:
        accs = [train(cell, T, seed) for seed in range(3)]
        print(f"{cell.__name__:4s} T={T:2d}: " + "  ".join(f"{a:.3f}" for a in accs))`;

const BPTT_REMEMBER_OUT = String.raw`test accuracy after 500 Adam steps (3 seeds); chance = 0.5
RNN  T= 5: 1.000  1.000  1.000
RNN  T=10: 1.000  1.000  0.998
RNN  T=20: 0.522  0.528  0.481
RNN  T=40: 0.488  0.529  0.501`;

const BPTT_TBPTT = String.raw`import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)

# One long stream (e.g. a sensor log): 4 parallel streams of 20,000 steps, target = input 3 steps earlier.
DELAY, LENGTH, STREAMS, CHUNK = 3, 20_000, 4, 25
x = torch.randn(STREAMS, LENGTH, 1)
y = torch.zeros_like(x)
y[:, DELAY:] = x[:, :-DELAY]


class Echo(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        return self.head(out), h


# ---- 1. carrying the state WITHOUT detach: the second backward() fails ----
model = Echo()
h = None
try:
    for start in range(0, 2 * CHUNK, CHUNK):
        pred, h = model(x[:, start:start + CHUNK], h)
        ((pred - y[:, start:start + CHUNK]) ** 2).mean().backward()
except RuntimeError as err:
    print("without detach:", str(err).split(".")[0])


# ---- 2. truncated BPTT: carry the state forward, cut the graph with detach() ----
def train_tbptt(clip, lr, init_scale=None, epochs=2):
    torch.manual_seed(0)
    model = Echo()
    if init_scale is not None:                       # deliberately large recurrent weights
        with torch.no_grad():
            model.rnn.weight_hh_l0.mul_(init_scale)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    norms, losses = [], []
    for epoch in range(epochs):
        h = None                                     # reset the state at the start of the stream
        for start in range(0, LENGTH, CHUNK):
            xb, yb = x[:, start:start + CHUNK], y[:, start:start + CHUNK]
            if h is not None:
                h = h.detach()                       # keep the VALUE of h, drop its history
            pred, h = model(xb, h)
            loss = ((pred[:, DELAY:] - yb[:, DELAY:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()                          # backprop only through this chunk (<= CHUNK steps)
            total = nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf"))
            norms.append(total.item())
            opt.step()
            losses.append(loss.item())
    tail = losses[-200:]
    return sum(tail) / len(tail), max(norms), sum(n > 1.0 for n in norms) / len(norms)


print(f"{'setting (SGD+momentum, lr 0.03)':34s} {'final MSE':>9s} {'max raw norm':>12s} {'steps norm>1':>13s}")
for name, clip, lr, scale in [("TBPTT k=25, default init", None, 0.03, None),
                              ("TBPTT k=25, W_hh x5, no clipping", None, 0.03, 5.0),
                              ("TBPTT k=25, W_hh x5, clip 1.0", 1.0, 0.03, 5.0)]:
    mse, mx, frac = train_tbptt(clip, lr, scale)
    print(f"{name:34s} {mse:9.4f} {mx:12.2f} {100 * frac:12.1f}%")`;

const BPTT_TBPTT_OUT = String.raw`without detach: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
setting (SGD+momentum, lr 0.03)    final MSE max raw norm  steps norm>1
TBPTT k=25, default init              0.0035         2.66         10.5%
TBPTT k=25, W_hh x5, no clipping      0.9566       189.80         69.0%
TBPTT k=25, W_hh x5, clip 1.0         0.0360       761.56         84.4%`;

const BPTT_KERAS = String.raw`import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
T, B = 20, 64


def batch():
    x = rng.normal(size=(B, T, 1)).astype("float32")
    x[:, 0, 0] = np.sign(x[:, 0, 0])                       # the bit to remember
    y = (x[:, 0, 0] > 0).astype("float32")
    return x, y


model = tf.keras.Sequential([
    tf.keras.Input(shape=(T, 1)),
    tf.keras.layers.SimpleRNN(32),                         # recurrent kernel: orthogonal init (Keras default)
    tf.keras.layers.Dense(1),
])
opt = tf.keras.optimizers.Adam(learning_rate=3e-3, global_clipnorm=1.0)   # clip the joint gradient norm
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

for step in range(500):
    x, y = batch()
    with tf.GradientTape() as tape:                        # records the unrolled forward pass
        loss = loss_fn(y, model(x)[:, 0])
    grads = tape.gradient(loss, model.trainable_variables) # BPTT through all T steps
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if step % 100 == 0:
        print(step, float(loss), float(tf.linalg.global_norm(grads)))

# How far back does the learning signal reach? Gradient of the loss w.r.t. every input x_t.
x, y = batch()
x = tf.constant(x)
with tf.GradientTape() as tape:
    tape.watch(x)
    loss = loss_fn(y, model(x)[:, 0])
g = tape.gradient(loss, x)                                 # shape (B, T, 1)
print(np.abs(g.numpy()).mean(axis=(0, 2)))                 # mean |dL/dx_t| for t = 1..T`;

const lesson: Lesson = {
  id: "m09/bptt",
  title: "Backpropagation Through Time",
  summary:
    "How a recurrent network learns: backpropagation on the unrolled network, the δ recursion that runs backwards in time, why shared-weight gradients are sums over steps, a three-step backward pass checked with autograd, the product of Jacobians behind vanishing and exploding gradients (measured on real runs), and the practical fixes — clipping, truncated BPTT, careful initialisation and gating.",
  minutes: 85,
  level: "intermediate",
  prerequisites: ["m09/rnn-basics", "m04/backpropagation", "m04/gradient-problems", "m02/jacobians-hessians"],
  next: ["m09/lstm", "m09/gru-variants", "m06/lr-schedules"],
  terms: ["bptt", "truncated-bptt", "vanishing-gradient", "exploding-gradient", "gradient-clipping", "spectral-radius", "long-range-dependency", "orthogonal-init"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "You train an RNN like any other network: loss, gradient, optimizer step. The only puzzle is the gradient, because the **same weights are used at every time step**. **Backpropagation through time** (BPTT) solves it in the plainest way. Unroll the RNN over the $T$ steps of the sequence into a deep feed-forward network — one layer per step, all layers sharing the same weights — and run ordinary [backpropagation](/learn/m04/backpropagation) on it. Paul Werbos named and explained it (1990); the unrolling idea already appears in Rumelhart, Hinton and Williams (1986)." },
      { t: "p", text: "Two things are new compared with a feed-forward network:" },
      { t: "list", items: [
        "Every hidden state affects the loss along **two** routes: through its own output, and through the next hidden state. So the backward step adds two terms.",
        "Each weight matrix is used $T$ times, so its gradient is the **sum** of $T$ per-step pieces.",
      ] },
      { t: "p", text: "Both come straight from the chain rule. We derive them line by line and check them with numbers." },
      { t: "p", text: "The derivation also shows the big weakness of plain (\"vanilla\") RNNs. A gradient travelling $k$ steps back gets multiplied by $k$ matrices $W_{hh}^\\top\\operatorname{diag}(1-h^2)$. A product of many matrices almost always shrinks or grows exponentially. Hochreiter (1991) and Bengio, Simard and Frasconi (1994) showed this makes long-range links very hard to learn. Pascanu, Mikolov and Bengio (2013) analysed both cases and made **gradient-norm clipping** the standard fix for the exploding one. The vanishing case is what the [LSTM](/learn/m09/lstm) was built to solve." },
      { t: "callout", kind: "key", text: "BPTT = backprop on the unrolled network. Recursion: $\\partial\\mathcal L/\\partial h_t = W_{hy}^\\top e_t + W_{hh}^\\top\\delta_{t+1}$, $\\delta_t = \\partial\\mathcal L/\\partial h_t \\odot (1-h_t^2)$, and $\\partial\\mathcal L/\\partial W_{hh} = \\sum_t \\delta_t h_{t-1}^\\top$. The signal from step $t$ back to step $k$ is multiplied by $\\prod W_{hh}^\\top\\operatorname{diag}(1-h_j^2)$ — the source of vanishing and exploding gradients." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Sequential data and vanilla RNNs](/learn/m09/rnn-basics) — the update $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$, the output $\\hat y_t = W_{hy}h_t + b_y$, unrolling, and the three-step worked forward pass we now run backwards.",
        "[Backpropagation](/learn/m04/backpropagation) — the error $\\delta = \\partial\\mathcal L/\\partial z$, the rules $\\delta^{[l-1]} = (W^\\top\\delta^{[l]})\\odot f'(z^{[l-1]})$ and $\\partial\\mathcal L/\\partial W = \\delta\\,a^\\top$, and the fact that gradients along different paths **add**.",
        "[Vanishing and exploding gradients](/learn/m04/gradient-problems) — the same problem in deep feed-forward nets, spectral-norm bounds, and a first short RNN measurement. Here we go deeper along time.",
        "[Jacobians](/learn/m02/jacobians-hessians) — the matrix of all partial derivatives of a vector function, the chain rule as a product of Jacobians, and vector–Jacobian products.",
        "[Eigenvalues and SVD](/learn/m02/eigen-svd) — spectral radius $\\rho(W) = \\max|\\lambda_i|$ (largest eigenvalue size) and spectral norm $\\|W\\|_2 = \\sigma_{\\max}(W)$ (largest stretch), with $\\rho(W) \\le \\|W\\|_2$.",
        "[Gradient checking](/learn/m04/gradient-checking) — central differences and relative error, used to test our from-scratch BPTT.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "Passing blame back through a week", text: "A shop's stock on Friday was wrong (the loss). Friday's stock depends on Thursday's, which depends on Wednesday's, and so on. The same clerk, with the same rules, placed an order every day. To fix the rules, you ask for each day: \"how much did that day's order add to Friday's error?\" Start on Friday and pass the blame back one day at a time. Each hand-over multiplies the blame by how strongly that day's stock carried into the next. The clerk's total blame is the **sum** over all days, because the same rules acted every day. If the carry-over is 0.5 per day, Monday gets only $0.5^4 \\approx 6\\%$ of Friday's blame — and after a month, almost nothing. That is the vanishing gradient." },
      { t: "p", text: "In an RNN the carry-over factor is a matrix, $W_{hh}^\\top\\operatorname{diag}(1-h_t^2)$. It has two parts: the recurrent weights (how much state $t$ fed state $t+1$) and the tanh slopes (how responsive each unit was then). A **saturated** unit — stuck near $\\pm1$ — has a slope near 0 and blocks the backward flow." },
      { t: "list", items: [
        "**Factors usually below 1:** the learning signal for long-range effects fades exponentially. The information may well be there in the forward pass; the gradient just never tells the weights to keep it.",
        "**Factors usually above 1:** the backward signal grows exponentially, and one unlucky sequence causes a gigantic update.",
      ] },
      { t: "p", text: "Two practical fixes follow. Every state must be stored for the backward pass, so long streams are trained in chunks, cutting the backward pass after $k$ steps (**truncated BPTT**). Explosions are rare but violent, so we cap each update's size (**gradient clipping**). Vanishing needs a new architecture: an additive memory path whose factor can stay near 1 — the LSTM's cell state." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "p", text: "One training step of full BPTT on a sequence (or a batch of sequences) of length $T$:" },
      { t: "steps", items: [
        { title: "Forward pass, storing every state", text: "Run the recurrence from $h_0$ and **keep** $h_1, \\dots, h_T$ (and the inputs). Compute each output $\\hat y_t$ and the loss terms $\\ell_t$ at steps that have targets; $\\mathcal L = \\sum_t \\ell_t$." },
        { title: "Output-layer gradients", text: "For each step with a loss, $e_t = \\partial\\ell_t/\\partial\\hat y_t$ ($\\hat y_t - y_t$ for squared error; $p_t - \\text{onehot}(y_t)$ for softmax + cross-entropy). Add up $\\partial\\mathcal L/\\partial W_{hy} \\mathrel{+}= e_t h_t^\\top$ and $\\partial\\mathcal L/\\partial b_y \\mathrel{+}= e_t$." },
        { title: "Start with nothing from the future", text: "Set $\\bar g = \\mathbf 0$: no gradient comes from beyond the last step." },
        { title: "At step t (going t = T, T−1, …, 1): total gradient on the state", text: "The state affected the loss through its own output **and** through $h_{t+1}$. Add both.", tex: String.raw`g_t = \frac{\partial\mathcal L}{\partial h_t} = W_{hy}^\top e_t + \bar g` },
        { title: "Through the tanh", text: "Multiply element by element by the slope $1 - h_t^2$, using the stored state.", tex: String.raw`\delta_t = \frac{\partial\mathcal L}{\partial z_t} = g_t \odot (1 - h_t^2)` },
        { title: "Add up the shared-weight gradients", text: "This step's use of each shared weight gets its local gradient; the pieces from all steps **add up**.", tex: String.raw`\frac{\partial\mathcal L}{\partial W_{hh}} \mathrel{+}= \delta_t h_{t-1}^\top,\quad \frac{\partial\mathcal L}{\partial W_{xh}} \mathrel{+}= \delta_t x_t^\top,\quad \frac{\partial\mathcal L}{\partial b} \mathrel{+}= \delta_t` },
        { title: "Send the gradient one step back", text: "Through the transposed recurrent matrix. This becomes the \"from the future\" term for step $t-1$.", tex: String.raw`\bar g \leftarrow W_{hh}^\top \delta_t` },
        { title: "Update", text: "After the loop, optionally clip the global gradient norm, then let the optimizer update all parameters at once." },
      ] },
      { t: "callout", kind: "warning", text: "The weight gradient at step $t$ uses $h_{t-1}$, the state *entering* the step. The slope uses $h_t$, the state *leaving* it. Mixing them up is the most common bug in hand-written BPTT; a gradient check catches it at once." },
      { t: "h", text: "Truncated BPTT" },
      { t: "p", text: "Full BPTT on a stream of a million steps would store a million states and make just one update per million steps. **Truncated BPTT** (described by Williams and Peng, 1990) runs the forward pass without stopping but limits how far each backward pass reaches. General form, TBPTT($k_1$, $k_2$): every $k_1$ steps, backpropagate through the latest $k_2$ steps, then update. Almost everyone uses $k_1 = k_2 = k$: cut the stream into chunks of $k$ steps and pass each chunk's final state into the next as its starting state — but **detach** it, so the backward pass stops at the chunk boundary." },
      { t: "list", items: [
        "Inside a chunk, the loss at position $i$ backpropagates through $i$ steps (1 to $k$). Dependencies longer than $k$ steps get no gradient at all. In the scratch code, a window of 3 cannot learn a delay of 3, while a window of 4 can.",
        "The forward state still carries information across chunks, so the model can *use* long context it happened to keep. It just never gets a learning signal telling it to keep it.",
        "Memory drops from $O(T)$ to $O(k)$ states, and updates come every $k$ steps instead of once per sequence.",
        "In PyTorch: `h = h.detach()` before each chunk. Forget it and you get an error (\"Trying to backward through the graph a second time\") or, with `retain_graph=True`, a silent backward pass through the whole history with ever-growing memory.",
      ] },
      { t: "h", text: "Gradient clipping" },
      { t: "p", text: "After the backward pass, compute the norm (overall size) of all gradients together, $\\|g\\|$. If it is above a threshold $c$, scale every gradient by $c/\\|g\\|$. The direction stays the same; only the step length is capped. So one exploding sequence cannot throw the weights somewhere far away. It does nothing for vanishing gradients. Typical thresholds are 0.25–5; see also [Learning-rate schedules](/learn/m06/lr-schedules)." },
      { t: "h", text: "Cost" },
      { t: "p", text: "**Time:** the backward loop does the forward loop's matrix products plus one more per step. So BPTT costs about 2–3× the forward pass, $O(T(n_h^2 + n_h n_{in}))$ — linear in $T$ and, like the forward pass, strictly one step after another." },
      { t: "p", text: "**Memory:** all $T$ hidden states ($T \\times B \\times n_h$ numbers for a batch) must be kept. For $T = 1000$, $B = 32$, $n_h = 512$ in float32 that is $1000\\cdot32\\cdot512\\cdot4$ bytes ≈ 65.5 MB for one layer's states — and an LSTM's gates multiply that by about five. This linear growth is why we truncate (or use gradient checkpointing, which stores only some states and recomputes the rest)." },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Setting" },
      { t: "math", tex: String.raw`z_t = W_{xh}x_t + W_{hh}h_{t-1} + b,\quad h_t = \tanh(z_t),\quad \hat y_t = W_{hy}h_t + b_y,\quad \mathcal L = \sum_{t=1}^{T}\ell_t(\hat y_t, y_t)` },
      { t: "p", text: "Symbols: $x_t \\in \\mathbb R^{n_{in}}$ is the input, $h_t, z_t \\in \\mathbb R^{n_h}$ the state and its value before tanh, $\\hat y_t \\in \\mathbb R^{n_{out}}$ the output. $\\ell_t$ is the loss at step $t$ (zero if there is no target). We also name:" },
      { t: "list", items: [
        "$e_t = \\partial\\ell_t/\\partial\\hat y_t$, the output error (for $\\ell_t = \\tfrac12\\|\\hat y_t - y_t\\|^2$, $e_t = \\hat y_t - y_t$);",
        "$g_t = \\partial\\mathcal L/\\partial h_t$, the **total** gradient on the state, through all later steps;",
        "$\\delta_t = \\partial\\mathcal L/\\partial z_t$, the same gradient before the tanh;",
        "$D_t = \\operatorname{diag}(1-h_t^2)$, the Jacobian of $\\tanh$ at step $t$.",
      ] },
      { t: "h", text: "Deriving the backward recursion" },
      { t: "p", text: "In the unrolled network, $h_t$ is used in exactly two places: in $\\hat y_t$ (so in $\\ell_t$) and in $z_{t+1}$ (so in everything after step $t$). The chain rule adds the two paths:" },
      { t: "math", tex: String.raw`g_t = \frac{\partial\mathcal L}{\partial h_t} = \underbrace{\Big(\frac{\partial \hat y_t}{\partial h_t}\Big)^{\!\top}\frac{\partial\ell_t}{\partial\hat y_t}}_{\text{from the output at } t} + \underbrace{\Big(\frac{\partial z_{t+1}}{\partial h_t}\Big)^{\!\top}\frac{\partial\mathcal L}{\partial z_{t+1}}}_{\text{from the future}} = W_{hy}^\top e_t + W_{hh}^\top\delta_{t+1}` },
      { t: "p", text: "with $\\delta_{T+1} = \\mathbf 0$. Next, $h_t = \\tanh(z_t)$ element by element and $\\tanh'(z) = 1 - \\tanh^2(z)$, so:" },
      { t: "math", tex: String.raw`\delta_t = \Big(\frac{\partial h_t}{\partial z_t}\Big)^{\!\top} g_t = D_t\,g_t = g_t \odot (1 - h_t^2)` },
      { t: "p", text: "Putting the two lines together gives a recursion in $\\delta$ alone — the time version of BP2 from the backpropagation lesson:" },
      { t: "math", tex: String.raw`\boxed{\;\delta_t = \left(W_{hy}^\top e_t + W_{hh}^\top\delta_{t+1}\right)\odot(1-h_t^2),\qquad t = T, T-1, \dots, 1\;}` },
      { t: "h", text: "Shared-weight gradients are sums over time" },
      { t: "p", text: "Pretend each step $t$ has its own copy $W_{hh}^{(t)}$ of the recurrent matrix, all equal to $W_{hh}$. The loss depends on $W_{hh}$ through every copy, so the chain rule gives $\\partial\\mathcal L/\\partial W_{hh} = \\sum_t \\partial\\mathcal L/\\partial W_{hh}^{(t)}$. Each copy appears only in $z_t = \\dots + W_{hh}^{(t)}h_{t-1}$, so its gradient is the same outer product as in a feed-forward layer:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal L}{\partial W_{hh}} = \sum_{t=1}^{T}\delta_t\,h_{t-1}^\top,\qquad \frac{\partial\mathcal L}{\partial W_{xh}} = \sum_{t=1}^{T}\delta_t\,x_t^\top,\qquad \frac{\partial\mathcal L}{\partial b} = \sum_{t=1}^{T}\delta_t,\qquad \frac{\partial\mathcal L}{\partial W_{hy}} = \sum_{t=1}^{T}e_t\,h_t^\top,\qquad \frac{\partial\mathcal L}{\partial b_y} = \sum_{t=1}^{T}e_t` },
      { t: "h", text: "Unrolled form: products of Jacobians" },
      { t: "p", text: "Unrolling the recursion shows *where* each piece comes from. The gradient that loss term $\\ell_t$ sends to an earlier state $h_k$ ($k < t$) is a product of per-step factors:" },
      { t: "math", tex: String.raw`\frac{\partial\ell_t}{\partial h_k} = \Big(\frac{\partial h_t}{\partial h_k}\Big)^{\!\top}\frac{\partial\ell_t}{\partial h_t},\qquad \frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}} = D_t W_{hh}\, D_{t-1}W_{hh}\cdots D_{k+1}W_{hh}` },
      { t: "math", tex: String.raw`\frac{\partial\mathcal L}{\partial W_{hh}} = \sum_{t=1}^{T}\;\sum_{k=1}^{t}\;\underbrace{D_k\Big(\frac{\partial h_t}{\partial h_k}\Big)^{\!\top}\frac{\partial \ell_t}{\partial h_t}}_{\text{signal from loss } t \text{ arriving at step } k}\;h_{k-1}^\top`, caption: "Each loss term ℓ_t reaches every earlier step k ≤ t; the piece from step k is weighted by the Jacobian product over the t − k steps in between (Pascanu, Mikolov and Bengio, 2013)." },
      { t: "p", text: "Terms with small $t - k$ are **short-range**; terms with large $t-k$ are **long-range**. Whether an RNN can learn long dependencies depends entirely on the size of $\\partial h_t/\\partial h_k$." },
      { t: "h", text: "Worked example: three steps backward, by hand" },
      { t: "p", text: "We reuse the network and forward pass from [the RNN lesson](/learn/m09/rnn-basics): $W_{xh} = (0.5, -0.3)^\\top$, $W_{hh} = \\begin{pmatrix}0.4 & -0.2\\\\0.1 & 0.6\\end{pmatrix}$, $b = (0, 0.1)$, $W_{hy} = (0.8, -0.5)$, $b_y = 0.1$, inputs $x = (1, 0.5, -1)$. That gave $h_1 = (0.462117, -0.197375)$, $h_2 = (0.441685, -0.121609)$, $h_3 = (-0.290401, 0.355044)$ and $\\hat y = (0.568381, 0.514152, -0.309843)$. Targets are $y = (0.5, 0, -0.5)$, with $\\ell_t = \\tfrac12(\\hat y_t - y_t)^2$ at every step." },
      { t: "steps", items: [
        { title: "Errors and loss", tex: String.raw`e = \hat y - y = (0.068381,\ 0.514152,\ 0.190157),\qquad \mathcal L = 0.002338 + 0.132176 + 0.018080 = 0.152594` },
        { title: "t = 3: only the output path (no future)", tex: String.raw`g_3 = W_{hy}^\top e_3 = \begin{pmatrix}0.8\\-0.5\end{pmatrix}0.190157 = \begin{pmatrix}0.152126\\-0.095079\end{pmatrix},\quad 1-h_3^2 = \begin{pmatrix}0.915667\\0.873944\end{pmatrix},\quad \delta_3 = \begin{pmatrix}0.139297\\-0.083093\end{pmatrix}` },
        { title: "t = 3: contributions", tex: String.raw`\delta_3 h_2^\top = \begin{pmatrix}0.061525 & -0.016940\\ -0.036701 & 0.010105\end{pmatrix},\qquad \delta_3 x_3 = \begin{pmatrix}-0.139297\\0.083093\end{pmatrix}` },
        { title: "Send back to step 2", tex: String.raw`W_{hh}^\top\delta_3 = \begin{pmatrix}0.4(0.139297) + 0.1(-0.083093)\\ -0.2(0.139297) + 0.6(-0.083093)\end{pmatrix} = \begin{pmatrix}0.047409\\-0.077715\end{pmatrix}` },
        { title: "t = 2: both paths add", tex: String.raw`g_2 = \underbrace{\begin{pmatrix}0.411322\\-0.257076\end{pmatrix}}_{W_{hy}^\top e_2} + \underbrace{\begin{pmatrix}0.047409\\-0.077715\end{pmatrix}}_{W_{hh}^\top\delta_3} = \begin{pmatrix}0.458731\\-0.334791\end{pmatrix},\quad 1-h_2^2 = \begin{pmatrix}0.804915\\0.985211\end{pmatrix},\quad \delta_2 = \begin{pmatrix}0.369239\\-0.329840\end{pmatrix}` },
        { title: "t = 2: contributions", tex: String.raw`\delta_2 h_1^\top = \begin{pmatrix}0.170632 & -0.072879\\ -0.152425 & 0.065102\end{pmatrix},\qquad \delta_2 x_2 = \begin{pmatrix}0.184620\\-0.164920\end{pmatrix}` },
        { title: "Send back to step 1", tex: String.raw`W_{hh}^\top\delta_2 = \begin{pmatrix}0.114712\\-0.271752\end{pmatrix}` },
        { title: "t = 1", tex: String.raw`g_1 = \begin{pmatrix}0.054705\\-0.034191\end{pmatrix} + \begin{pmatrix}0.114712\\-0.271752\end{pmatrix} = \begin{pmatrix}0.169417\\-0.305943\end{pmatrix},\quad 1-h_1^2 = \begin{pmatrix}0.786448\\0.961043\end{pmatrix},\quad \delta_1 = \begin{pmatrix}0.133237\\-0.294024\end{pmatrix}` },
        { title: "t = 1: contributions", text: "$h_0 = 0$, so step 1 adds nothing to $\\partial\\mathcal L/\\partial W_{hh}$ — but it does add to $W_{xh}$ and $b$.", tex: String.raw`\delta_1 h_0^\top = \mathbf 0,\qquad \delta_1 x_1 = \begin{pmatrix}0.133237\\-0.294024\end{pmatrix}` },
        { title: "Sum over time", tex: String.raw`\frac{\partial\mathcal L}{\partial W_{hh}} = \begin{pmatrix}0.232157 & -0.089818\\ -0.189126 & 0.075207\end{pmatrix},\quad \frac{\partial\mathcal L}{\partial W_{xh}} = \begin{pmatrix}0.178561\\-0.375851\end{pmatrix},\quad \frac{\partial\mathcal L}{\partial b} = \begin{pmatrix}0.641773\\-0.706958\end{pmatrix}` },
        { title: "Output layer", tex: String.raw`\frac{\partial\mathcal L}{\partial W_{hy}} = \sum_t e_t h_t^\top = (0.203471,\ -0.008508),\qquad \frac{\partial\mathcal L}{\partial b_y} = \sum_t e_t = 0.772691` },
      ] },
      { t: "p", text: "PyTorch autograd in the Framework section reproduces every number to six decimals, including $\\partial\\mathcal L/\\partial h_0 = W_{hh}^\\top\\delta_1 = (0.023893, -0.203062)$ — the gradient you would use to learn a starting state. Notice how much of $g_2$ and $g_1$ comes from the future: at $t = 1$ the term passed back from step 2 is about five times the step's own output term." },
      { t: "p", text: "To see the decay on its own, keep only the last loss term ($\\mathcal L = \\ell_3$). Then $\\|g_3\\| = 0.179394$, $\\|g_2\\| = 0.091035$, $\\|g_1\\| = 0.054109$: each step back multiplies the norm by about 0.5–0.6. The per-step factors are $\\|W_{hh}^\\top D_t\\|_2 = 0.598, 0.613, 0.558$ for $t = 1, 2, 3$, and $\\rho(W_{hh}) = |0.5 \\pm 0.1i| = 0.510$. Twenty steps at 0.55 would leave $0.55^{20} \\approx 6\\times10^{-6}$ of the signal." },
      { t: "h", text: "Spectral analysis: vanishing and exploding" },
      { t: "p", text: "Take norms of the Jacobian product. We use $\\|AB\\| \\le \\|A\\|\\|B\\|$, and $\\|D_j\\|_2 = \\max_i(1 - h_{j,i}^2) \\le \\gamma$, where $\\gamma$ bounds the activation's slope ($\\gamma = 1$ for tanh):" },
      { t: "math", tex: String.raw`\Big\|\frac{\partial h_t}{\partial h_k}\Big\|_2 \;\le\; \prod_{j=k+1}^{t}\|D_j\|_2\,\|W_{hh}\|_2 \;\le\; \big(\gamma\,\sigma_{\max}(W_{hh})\big)^{t-k}` },
      { t: "list", items: [
        "**Enough to guarantee vanishing:** if $\\sigma_{\\max}(W_{hh}) < 1/\\gamma = 1$, long-range pieces shrink at least geometrically, for *every* input sequence (Pascanu, Mikolov and Bengio, 2013).",
        "**Needed for exploding:** by the same bound, explosion is impossible unless $\\sigma_{\\max}(W_{hh}) > 1$. But that is not enough on its own: saturated units ($D_j \\approx 0$) can still make the product shrink.",
        "**Linear RNN** ($h_t = W_{hh}h_{t-1} + \\dots$): the product is exactly $W_{hh}^{t-k}$, whose norm behaves like $\\rho(W_{hh})^{t-k}$ — vanishing for $\\rho < 1$, exploding for $\\rho > 1$. The tanh only adds factors $\\le 1$.",
        "**Spectral radius vs norm:** for a random Gaussian matrix, $\\|W\\|_2 \\approx 2\\rho(W)$, so the bound is loose. The real decay is measured below.",
      ] },
      { t: "p", text: "Bengio, Simard and Frasconi (1994) made a sharper point. To *store* a bit robustly (so input noise does not erase it), an RNN's dynamics must pull nearby states together around the stored state — and that is exactly when gradients vanish. In a vanilla RNN, robust memory and learnable long-range gradients pull in opposite directions. Here is how big the effect is at initialisation (32 units, 100 steps, loss at the last step):" },
      { t: "code", lang: "python", title: "gradient_norms_through_time.py", code: BPTT_NORMS, output: BPTT_NORMS_OUT },
      { t: "list", items: [
        "With tanh and $\\mathcal N(0, 1)$ inputs and $\\rho \\in [0.5, 1.5]$, the gradient 20 steps back is $3\\times10^{-10}$ to $6\\times10^{-3}$ of its size at the loss. 99 steps back it is between $10^{-11}$ and $10^{-48}$ — numerically nothing.",
        "Even $\\rho = 1$ vanishes. The inputs push units into saturation, so the mean slope $1-h^2$ is only about 0.6. With small inputs ($\\sigma = 0.1$) the slope is 0.96, and the same $W_{hh}$ keeps $2\\times10^{-3}$ of the signal after 99 steps instead of $2\\times10^{-23}$.",
        "Large $\\rho$ (2.5, 5.0) explodes despite heavy saturation (mean slope 0.19 at $\\rho = 5$): the few unsaturated units, times large weights, take over. 99 steps back the gradient is $7\\times10^{14}$ times larger.",
        "An **orthogonal** $W_{hh}$ (all singular values 1, $\\|W\\|_2 = 1$) is the best case for the matrix part. But the tanh slopes still make the product vanish unless units stay in their linear region.",
        "The \"mean factor\" column (average $\\|W_{hh}^\\top D_t\\|_2$) is often above 1 even when the product vanishes: the per-step bound is loose.",
      ] },
      { t: "h", text: "Truncation as an approximation" },
      { t: "p", text: "Truncated BPTT with window $k$ keeps only the terms with $t - j < k$ from the double sum above:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal L}{\partial W_{hh}} \approx \sum_{t=1}^{T}\ \sum_{j=\max(1,\,t-k+1)}^{t} D_j\Big(\frac{\partial h_t}{\partial h_j}\Big)^{\!\top}\frac{\partial\ell_t}{\partial h_t}\,h_{j-1}^\top` },
      { t: "p", text: "It is a **biased** estimate: it ignores dependencies longer than $k$. If gradients vanish anyway, the dropped terms are tiny and truncation costs little. But a dependency of length $d$ is invisible to training when $k \\le d$. The scratch code measures the bias (relative error 0.61 for $k=1$, 0.07 for $k=4$, 0.01 for $k=8$, machine precision for $k = T$) and its effect (a delay-3 task is learned with $k=4$, not with $k=3$)." },
      { t: "h", text: "Gradient clipping" },
      { t: "math", tex: String.raw`\hat g \leftarrow \begin{cases} g & \text{if } \|g\| \le c\\[2pt] \dfrac{c}{\|g\|}\,g & \text{if } \|g\| > c\end{cases}\qquad \|g\| = \sqrt{\textstyle\sum_{\text{all parameters}}\|g_\theta\|^2}` },
      { t: "p", text: "Pascanu and colleagues' picture: exploding gradients are steep \"cliffs\" in the loss surface. A full gradient step at a cliff jumps far away; the rescaled step moves a limited distance in the same direction." },
      { t: "h", text: "Initialisation and architecture" },
      { t: "list", items: [
        "**Orthogonal initialisation** of $W_{hh}$ makes all singular values 1, so the linear part of each factor keeps norms exactly (Saxe, McClelland and Ganguli, 2014, studied it for deep linear networks). Keras recurrent layers use it by default.",
        "**Identity initialisation with ReLU** (IRNN; Le, Jaitly and Hinton, 2015): $W_{hh} = I$, $b = 0$, ReLU activations. At the start the state is copied forward unchanged, so the gradient along active units flows back with factor 1.",
        "**Gating** — the real fix. An LSTM updates its cell state by addition, $c_t = f_t\\odot c_{t-1} + i_t\\odot g_t$. Along that path $\\partial c_t/\\partial c_{t-1} = \\operatorname{diag}(f_t)$: no weight matrix and no tanh slope, just a learned forget factor that can stay near 1. See [LSTM](/learn/m09/lstm).",
      ] },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "p", text: "The simulator runs a 4-unit vanilla RNN forward over $T$ random inputs, then steps the backward pass from right to left. For each step it shows $\\|\\partial\\mathcal L/\\partial h_t\\|$ (log scale), the per-step factor $\\|W_{hh}^\\top D_t\\|_2$ (with a line at 1), and that step's piece $\\delta_t h_{t-1}^\\top$ of $\\partial\\mathcal L/\\partial W_{hh}$ — outlined for full BPTT, filled for the part truncation keeps. The side panel prints the recursion with the real numbers." },
      { t: "sim", id: "bptt", caption: "Everything is computed exactly (double precision) from the unrolled network; the recurrent matrix is rescaled to the chosen spectral radius." },
      { t: "list", items: [
        "**Vanishing**: T = 40, tanh, loss at the last step, ρ = 0.9. Step through the whole backward pass and read the ratio ‖∂L/∂h₁‖/‖∂L/∂h_T‖ — many orders of magnitude. The bars for early steps are invisible: $W_{hh}$ learns almost nothing from them.",
        "**Saturation**: keep ρ = 0.9 and lower the input scale to 0.1. The slopes $1-h^2$ get close to 1 and the gradient decays much more slowly. Raise the input scale to 2: faster decay.",
        "**Exploding**: switch to *linear* and drag ρ above 1 (try 1.3). The bars grow geometrically to the left and the factor row sits above the line at 1. Tick clipping with c = 1 and compare the raw and clipped global norms.",
        "**Truncation**: with loss at the last step, set k = 5. Steps more than 5 from the loss are dashed and their pieces dropped (outline only). With vanishing gradients (tanh, ρ = 0.9, T = 40), full and truncated ‖∂L/∂W_hh‖ are of the same order (0.330 vs 0.402): older steps carried almost no signal. In the exploding setting (linear, ρ = 1.3) they differ by almost 10× ($9.2\\times10^{9}$ vs $1.1\\times10^{9}$), because the dropped far-away steps are the largest terms.",
        "**Loss at every step**: each step now adds its own gradient, so ‖∂L/∂h_t‖ no longer shrinks steadily — short-range terms dominate. With truncation, the kept part of each step's piece is the sum over loss terms at most k steps to its right.",
      ] },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "A full NumPy BPTT for a batched vanilla RNN with a loss at every step: one right-to-left sweep for full BPTT, a windowed version for truncated BPTT, a gradient check, the truncation bias, and Adam training with clipping on a task with a known dependency length." },
      { t: "code", lang: "python", title: "bptt_numpy.py", code: BPTT_NUMPY, output: BPTT_NUMPY_OUT },
      { t: "list", items: [
        "`backward` with `k=None` is the mechanism section line for line: `dh_from_y[t] + dh_next` is $g_t$; `dh * (1 - hs[t + 1] ** 2)` is $\\delta_t$ (the cache `hs` holds $h_0$ at index 0, so `hs[t + 1]` is $h_t$ for 0-based `t`); `dz.T @ hs[t]` is $\\delta_t h_{t-1}^\\top$ summed over the batch; `dz @ W_hh` is $W_{hh}^\\top\\delta_t$ in row-vector form.",
        "Gradient check: worst relative error $3.8\\times10^{-9}$ over all parameters, so the formula-based gradient is right. ($\\varepsilon = 10^{-6}$ in float64.)",
        "Truncation bias on a 30-step sequence: the relative error of $\\partial\\mathcal L/\\partial W_{hh}$ is 0.61 for $k = 1$ (only each step's own loss), 0.012 for $k = 8$, and $8\\times10^{-16}$ for $k = 30 = T$ — so the windowed code equals full BPTT when nothing is cut.",
        "Delay-3 echo task ($y_t = x_{t-3}$): full BPTT and $k = 4$ reach a test MSE of about 0.002; $k = 3$ and $k = 2$ stay near 1.0, the error of always predicting 0. Why 4? The input $x_{t-3}$ enters at step $t-3$, so the loss at step $t$ must flow back through steps $t, t-1, t-2, t-3$ — four steps. With $k \\le 3$ that term is never computed, so learning cannot see the dependency.",
      ] },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**1. Autograd reproduces the worked example.** Autograd records the unrolled loop as it runs, so `loss.backward()` is BPTT. `retain_grad()` keeps $\\partial\\mathcal L/\\partial h_t$ on the in-between states so we can compare them with $g_t$." },
          { t: "code", lang: "python", title: "bptt_autograd_check.py", code: BPTT_AUTOGRAD, output: BPTT_AUTOGRAD_OUT },
          { t: "p", text: "**2. A long-range dependency a vanilla RNN cannot learn.** The first input is $+1$ or $-1$; the next $T-1$ inputs are random noise. At the last step the model must say which sign came first. The information is there in the forward pass — a perfect memory would solve it for any $T$ — but the gradient linking the final loss to $x_1$ must pass back through $T-1$ Jacobians." },
          { t: "code", lang: "python", title: "remember_first_input.py", code: BPTT_REMEMBER, output: BPTT_REMEMBER_OUT },
          { t: "list", items: [
            "Up to $T = 10$ every seed solves the task (accuracy ≥ 0.998). At $T = 20$ and $T = 40$ all three seeds are at chance: training never finds the dependency within 500 steps.",
            "With `CELLS = [nn.RNN, nn.LSTM]` (forget-gate bias 1), the LSTM scores 1.000/1.000/1.000 at $T = 10$ and 0.995/1.000/1.000 at $T = 20$, where the vanilla RNN fails. At $T = 40$ one of three LSTM seeds reaches 0.888 within 500 steps. Gating pushes the limit out; it does not remove it — see [LSTM](/learn/m09/lstm).",
            "Clipping at 1.0 is on for every run, so the failure is not an explosion. It is the vanishing learning signal measured in the Math section.",
          ] },
          { t: "p", text: "**3. Truncated BPTT with `detach()`, and clipping.** A long stream (4 parallel streams × 20,000 steps, target = input 3 steps earlier) is processed in chunks of 25. The first block shows what happens without detaching the carried state. The second trains with TBPTT: once with normal initialisation, and twice with $W_{hh}$ multiplied by 5 (deliberately exploding), without and with clipping. We use SGD with momentum because its step size, unlike Adam's, grows with the gradient — so explosions and clipping show up clearly." },
          { t: "code", lang: "python", title: "tbptt_and_clipping.py", code: BPTT_TBPTT, output: BPTT_TBPTT_OUT },
          { t: "list", items: [
            "Without `detach()`, the second chunk's `backward()` tries to go through the first chunk's graph, whose buffers are already freed — PyTorch raises the error shown.",
            "With TBPTT and normal initialisation the model learns the delay-3 echo (MSE 0.0035; skipping the loss on each chunk's first 3 positions only keeps the code short). The raw gradient norm stays at or below 2.66.",
            "With $W_{hh} \\times 5$ and no clipping, raw norms reach 190 and training fails (MSE 0.96 ≈ predicting 0). With clipping at 1.0 the same run reaches MSE 0.036. The raw norms are even larger (up to 762, as the weights wander into steeper regions), but every applied step is capped.",
            "`clip_grad_norm_` returns the norm **before** clipping — log it. The share of steps above the threshold is a useful health signal: 10.5% in the healthy run, 84.4% in the badly initialised one.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "In TensorFlow, `tf.GradientTape` records the unrolled computation of a `SimpleRNN`, and `tape.gradient` does BPTT. Keras optimizers clip by global norm with `global_clipnorm` (or per variable with `clipnorm`). Watching the *input* tensor gives $\\partial\\mathcal L/\\partial x_t$ for every step — a direct view of how far back the learning signal reaches." },
          { t: "code", lang: "python", title: "bptt_keras.py (not executed in CI)", code: BPTT_KERAS, output: "Prints the step, loss and global gradient norm every 100 steps, then the mean |∂L/∂x_t| for t = 1…20. Expect the values for early t to be orders of magnitude smaller than for the last steps unless the network has learned to carry the first input forward." },
          { t: "p", text: "Truncated BPTT in Keras usually uses `stateful=True` recurrent layers fed with consecutive chunks. Each batch's final state becomes the next batch's starting state, while each `train_on_batch` call only differentiates through the current chunk." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "table", head: ["Method", "Idea", "Pros", "Cons", "When used"], rows: [
        ["Full BPTT", "Backprop through all $T$ steps of each sequence", "Exact gradient", "Memory $O(T)$; one update per sequence", "Medium-length sequences (sentences, short clips)"],
        ["Truncated BPTT, $k_1 = k_2 = k$", "Cut the stream into chunks; carry the state, detach it at chunk edges", "Memory $O(k)$; frequent updates; simple", "Biased: no gradient for dependencies longer than $k$", "Language modelling on long text, long sensor streams"],
        ["TBPTT($k_1$, $k_2$), $k_2 > k_1$", "Every $k_1$ steps, backprop $k_2$ steps", "Every loss term sees at least $k_2 - k_1$ steps of history", "More compute per update (backward passes overlap)", "When short chunks would cut too much"],
        ["Real-time recurrent learning (RTRL)", "Forward-mode derivatives: carry $\\partial h_t/\\partial\\theta$ along with $h_t$ (Williams and Zipser, 1989)", "Exact, online, no stored history", "$O(n_h^4)$ time per step for a fully connected RNN", "Research on online learning and its approximations"],
        ["Gradient checkpointing", "Store only every $m$-th state; recompute the rest during backward", "Memory $O(T/m + m)$, exact gradient", "About one extra forward pass of compute", "Long sequences that must not be truncated"],
        ["Gradient clipping (norm / value)", "Rescale the whole gradient to norm $\\le c$ / clamp each entry", "Stops exploding updates; norm clipping keeps the direction", "Nothing for vanishing; value clipping changes the direction", "Almost always when training RNNs"],
        ["Orthogonal / identity / unitary $W_{hh}$", "Make the linear part of each Jacobian keep norms (unitary RNNs keep it that way during training — Arjovsky, Shah and Bengio, 2016)", "Better gradient flow; cheap (initialisation only)", "tanh slopes still shrink the product; constraints limit what the model can express", "Vanilla RNNs on long tasks, research models"],
        ["Gated cells (LSTM, GRU)", "Additive memory path with learned factors near 1", "Remembers over hundreds of steps", "More parameters and compute per step", "The default — see [LSTM](/learn/m09/lstm), [GRU](/learn/m09/gru-variants)"],
      ] },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Training every recurrent model.** BPTT — as autograd through the unrolled loop, or inside cuDNN's fused RNN kernels — is how vanilla RNNs, LSTMs and GRUs were trained for speech recognition, translation, handwriting and time series.",
        "**Language modelling on long text** is the classic use of truncated BPTT: the text is one long stream cut into fixed chunks (often 35–200 tokens), with the hidden state carried across chunks and detached.",
        "**Streaming learning** (sensors, control): TBPTT lets a model learn on an endless stream with constant memory.",
        "**Recurrent policies in reinforcement learning** are trained by BPTT over stored pieces of trajectories, with the same truncation and state-carrying issues.",
        "**Beyond RNNs:** any time loop that autograd differentiates — physics simulators, unrolled optimisation, spiking networks trained with surrogate gradients — is BPTT under another name, with the same memory cost and Jacobian products.",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "table", head: ["Strength", "Limitation"], rows: [
        ["Exact gradients for any differentiable recurrence, computed automatically by autograd", "Memory grows linearly with the number of steps you backpropagate through"],
        ["Costs only about 2–3× the forward pass", "Sequential in time: no parallelism across steps, unlike Transformers"],
        ["Truncation gives a simple memory-vs-accuracy dial ($k$)", "Truncation is biased: dependencies longer than $k$ get no learning signal"],
        ["Clipping makes exploding gradients a manageable nuisance", "No training trick fixes vanishing gradients: vanilla RNNs stay limited to short dependencies"],
      ] },
      { t: "p", text: "Rules of thumb: always clip (start with a global-norm threshold around 1). Make the truncation length comfortably longer than the dependencies you need. Initialise $W_{hh}$ orthogonally. If the task needs memory over more than about 10–20 steps, use an LSTM or GRU instead of fighting the vanilla RNN." },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Forgetting the \"from the future\" term.** Computing $\\delta_t$ from $W_{hy}^\\top e_t$ alone gives the gradient of a network with no recurrence. The loss still falls a little, which hides the bug; a gradient check exposes it.",
        "**Off-by-one in the cache.** The weight gradient uses $h_{t-1}$; the slope uses $h_t$. With $h_0$ stored at index 0, these are `hs[t]` and `hs[t + 1]` for 0-based `t`.",
        "**Overwriting instead of adding** the shared-weight gradients (`dW = ...` inside the loop instead of `dW += ...`): only step 1's piece survives.",
        "**Not detaching the carried state** in stateful training — an error on the second chunk, or with `retain_graph=True` a graph and backward pass that grow with every batch.",
        "**Detaching when you should not.** Calling `detach()` on the state *inside* a sequence (say, every step) turns BPTT into one-step truncation: the model learns no dependencies at all.",
        "**Truncation window shorter than the dependency.** As the delay task shows, no amount of training recovers a dependency the backward pass never spans.",
        "**Clipping per parameter when you meant global norm**, or clipping by value: both change the gradient's direction. Use `clip_grad_norm_` over all parameters, after `backward()` and before `step()`.",
        "**Taking a falling training loss as proof that long-range structure is learned.** Short-range terms dominate the gradient, so the loss can improve steadily while the long dependency is never learned. Test long-range behaviour directly (for example, gradient or accuracy as a function of distance).",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is backpropagation through time?", a: "Ordinary backpropagation on the RNN unrolled over the time steps of a sequence. The unrolled network is a deep feed-forward net whose layers share weights. The backward pass runs from the last step to the first, and each shared weight's gradient is the sum of its per-step gradients." },
      { t: "qa", level: "beginner", q: "Why is the gradient of $W_{hh}$ a sum over time steps?", a: "The same matrix is used at every step, so the loss depends on it through every step. Treat each use as a separate copy; the multivariable chain rule then adds them: $\\partial\\mathcal L/\\partial W_{hh} = \\sum_t \\delta_t h_{t-1}^\\top$." },
      { t: "qa", level: "beginner", q: "What does gradient clipping do, and which problem does it address?", a: "If the global gradient norm is above a threshold $c$, the gradient is rescaled to norm $c$, keeping its direction. This stops exploding gradients from causing huge updates. It does nothing for vanishing gradients." },
      { t: "qa", level: "intermediate", q: "Write the BPTT recursion for a tanh RNN with a loss at every step.", a: "$g_t = W_{hy}^\\top e_t + W_{hh}^\\top\\delta_{t+1}$ (with $\\delta_{T+1} = 0$), $\\delta_t = g_t \\odot (1 - h_t^2)$; then $\\partial\\mathcal L/\\partial W_{hh} = \\sum_t\\delta_t h_{t-1}^\\top$, $\\partial\\mathcal L/\\partial W_{xh} = \\sum_t\\delta_t x_t^\\top$, $\\partial\\mathcal L/\\partial b = \\sum_t\\delta_t$, $\\partial\\mathcal L/\\partial W_{hy} = \\sum_t e_t h_t^\\top$." },
      { t: "qa", level: "intermediate", q: "What is truncated BPTT, and what does it cost you?", a: "You process the stream in chunks of $k$ steps. The hidden state is carried forward but detached at chunk edges, so each backward pass covers at most $k$ steps. Memory is bounded to $O(k)$ and updates are frequent. The cost: the gradient is biased — dependencies longer than $k$ steps get no learning signal, even if the forward state carries the information." },
      { t: "qa", level: "intermediate", q: "Why do vanilla RNN gradients vanish even when $\\rho(W_{hh}) = 1$?", a: "The per-step Jacobian is $D_tW_{hh}$ with $D_t = \\operatorname{diag}(1-h_t^2)$. Unless every unit stays in tanh's linear region, the slopes are below 1 (about 0.6 on average in our measurement with unit-variance inputs), so the product shrinks geometrically. Also, $\\rho = 1$ does not make a non-normal matrix keep norms over finite products." },
      { t: "qa", level: "advanced", q: "State the conditions of Pascanu, Mikolov and Bengio (2013) for vanishing and exploding gradients in a tanh RNN.", a: "Let $\\gamma$ bound $|f'|$ ($\\gamma = 1$ for tanh) and $\\lambda_1$ be the largest singular value of $W_{hh}$. Then $\\lambda_1 < 1/\\gamma$ is **sufficient** for long-term gradient pieces to vanish, because $\\|\\partial h_t/\\partial h_k\\| \\le (\\gamma\\lambda_1)^{t-k}$. And $\\lambda_1 > 1/\\gamma$ is **necessary** for them to explode. They proposed gradient-norm clipping for explosion, and a regulariser pushing Jacobians to keep norms for vanishing." },
      { t: "qa", level: "advanced", q: "Why does the LSTM mitigate vanishing gradients, in terms of Jacobians?", a: "Its cell state is updated as $c_t = f_t\\odot c_{t-1} + i_t\\odot g_t$. Along the direct cell path, $\\partial c_t/\\partial c_{t-1} = \\operatorname{diag}(f_t)$ — no weight matrix and no saturating slope. If the forget gates stay near 1, the product over many steps stays near 1, so a long-range gradient path exists. Other paths through the gates still go through weight matrices and can vanish, so gating *reduces* the problem rather than removing it." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "An RNN's per-step factor is bounded by $\\|W_{hh}\\|_2 = 0.8$ (tanh slopes ≤ 1). What is the upper bound on $\\|\\partial h_{t}/\\partial h_{t-30}\\|_2$? (5 decimals)", answer: 0.00124, tolerance: 0.00001, explain: "$0.8^{30} = 0.0012379$ — after 30 steps at most about 0.1% of the signal is left." },
        { kind: "numeric", q: "The global gradient norm is 50 and the clipping threshold is $c = 5$. One gradient entry is 20 before clipping. What is it after?", answer: 2, tolerance: 0.0001, explain: "Scale factor $c/\\|g\\| = 5/50 = 0.1$, applied to every entry: $20 \\times 0.1 = 2$." },
        { kind: "numeric", q: "How many megabytes (10^6 bytes) does BPTT need to store the hidden states for $T = 1000$, batch 32, $n_h = 512$, float32 (one layer, states only)?", answer: 65.536, tolerance: 0.01, explain: "$1000 \\times 32 \\times 512 \\times 4 = 65{,}536{,}000$ bytes = 65.536 MB. It grows linearly with $T$." },
        { q: "In $g_t = W_{hy}^\\top e_t + W_{hh}^\\top\\delta_{t+1}$, what does the second term mean?", options: ["The gradient from the output at step t", "The gradient flowing back from all later time steps through h_{t+1}", "The weight-decay term", "The gradient with respect to the input x_t"], answer: 1, explain: "$h_t$ affects the loss through its own output and through $z_{t+1}$. The second term is the latter; it carries all future loss terms." },
        { q: "You train with truncated BPTT, chunk length k = 20. The target depends on an input 50 steps earlier. What happens?", options: ["It is learned, just more slowly", "The gradient for this dependency is never computed, so it cannot be learned by gradient descent (only by luck through the forward state)", "Training diverges", "Clipping compensates for the truncation"], answer: 1, explain: "No backward pass spans 50 steps, so the long-range term is missing from every gradient. The truncation length must be longer than the dependency." },
        { q: "Which statement about gradient clipping is correct?", options: ["It solves the vanishing gradient problem", "Clipping by global norm preserves the gradient direction", "It must be applied before backward()", "It changes the loss function"], answer: 1, explain: "Norm clipping multiplies the whole gradient by one number ≤ 1, so the direction stays the same. It goes after backward() and before the optimizer step, and does nothing against vanishing gradients." },
        { q: "A linear RNN ($h_t = W_{hh}h_{t-1} + W_{xh}x_t$) has $\\rho(W_{hh}) = 1.1$. How does $\\|\\partial h_t/\\partial h_{t-k}\\|$ behave for large k?", options: ["Decays like $1.1^{-k}$", "Stays constant", "Grows roughly like $1.1^{k}$", "Oscillates with bounded norm"], answer: 2, explain: "The Jacobian is exactly $W_{hh}^k$, whose norm grows like $\\rho^k$. The measured table shows about $3\\times10^3$ after 99 steps for a random matrix with ρ = 1.1." },
      ] },
      { t: "exercise", title: "Scalar BPTT by hand", prompt: "A scalar RNN has $h_t = \\tanh(w_x x_t + w_h h_{t-1})$ with $w_x = 1$, $w_h = 0.5$, $h_0 = 0$, inputs $x = (1, -1)$ and one loss at the end, $\\mathcal L = \\tfrac12 h_2^2$. Use BPTT to compute $\\partial\\mathcal L/\\partial w_h$ and $\\partial\\mathcal L/\\partial w_x$, showing each step's piece.", hint: "Forward: $h_1 = \\tanh(1)$, $h_2 = \\tanh(-1 + 0.5h_1)$. Backward: $g_2 = h_2$, $\\delta_2 = g_2(1-h_2^2)$, $g_1 = w_h\\delta_2$, $\\delta_1 = g_1(1-h_1^2)$.", solution: [
        { t: "steps", items: [
          { title: "Forward", tex: String.raw`h_1 = \tanh(1) = 0.761594,\quad h_2 = \tanh(-0.619203) = -0.550573,\quad \mathcal L = \tfrac12 h_2^2 = 0.151565` },
          { title: "Step 2", tex: String.raw`g_2 = h_2 = -0.550573,\quad 1-h_2^2 = 0.696870,\quad \delta_2 = -0.383677` },
          { title: "Step 1", tex: String.raw`g_1 = w_h\,\delta_2 = -0.191839,\quad 1-h_1^2 = 0.419974,\quad \delta_1 = -0.080567` },
          { title: "Recurrent-weight gradient (sum over steps)", tex: String.raw`\frac{\partial\mathcal L}{\partial w_h} = \delta_2 h_1 + \delta_1 h_0 = (-0.383677)(0.761594) + 0 = -0.292206` },
          { title: "Input-weight gradient", tex: String.raw`\frac{\partial\mathcal L}{\partial w_x} = \delta_2 x_2 + \delta_1 x_1 = (-0.383677)(-1) + (-0.080567)(1) = 0.303110` },
        ] },
        { t: "p", text: "Step 1's piece of $\\partial\\mathcal L/\\partial w_x$ ($-0.080567$) is about a fifth of step 2's. One step back, the signal was multiplied by $w_h(1-h_1^2) = 0.5 \\times 0.419974 = 0.21$ — vanishing in miniature." },
      ] },
      { t: "exercise", title: "TBPTT(k1, k2) from scratch", prompt: "Add a method `tbptt_grads(xs, ys, k1, k2)` to `VanillaRNN` in `bptt_numpy.py` that implements general truncated BPTT. Walk through the sequence in blocks of $k_1$ steps. At the end of each block, backpropagate the loss terms **of that block** through at most $k_2$ steps (so with $k_2 > k_1$ the backward pass reaches into the previous block). Return the summed gradients. Check that `k1 = k2 = T` reproduces `backward(ys)` exactly.", hint: "For a loss term at step s in the block ending at step e, the backward pass may go down to step max(0, e − k2 + 1). Reuse the inner `accumulate` logic.", solution: [
        { t: "code", lang: "python", code: String.raw`def tbptt_grads(self, xs, ys, k1, k2):
    self.forward(xs)                                   # cache all states once (for the demo)
    p, hs = self.p, self.hs
    T, B = ys.shape[:2]
    g = {n: np.zeros_like(v) for n, v in p.items()}
    e = (self.ys - ys) / B
    g["W_hy"] = np.einsum("tbo,tbh->oh", e, np.stack(hs[1:]))
    g["b_y"] = e.sum(axis=(0, 1))
    dh_y = e @ p["W_hy"]
    for start in range(0, T, k1):                      # first step of each block
        end = min(start + k1, T) - 1                   # last step of the block
        stop = max(0, end - k2 + 1)                    # how far back this backward pass may go
        dh = np.zeros_like(hs[0])
        for t in range(end, stop - 1, -1):             # one sweep: the block's losses enter as we pass them
            if t >= start:
                dh = dh + dh_y[t]
            dz = dh * (1 - hs[t + 1] ** 2)
            g["W_hh"] += dz.T @ hs[t]
            g["W_xh"] += dz.T @ xs[t]
            g["b"] += dz.sum(axis=0)
            dh = dz @ p["W_hh"]
    return g

VanillaRNN.tbptt_grads = tbptt_grads                  # attach to the class from bptt_numpy.py
T = xs.shape[0]
model.forward(xs)
full = model.backward(ys)
tb = model.tbptt_grads(xs, ys, k1=T, k2=T)
print(max(np.abs(tb[n] - full[n]).max() for n in full))   # 0.0` },
        { t: "p", text: "With $k_1 = k_2 = T$ there is one block and one full sweep, the same as `backward`. With $k_2 > k_1$, each block's loss terms reach $k_2 - k_1$ steps into the previous block, so even a block's first loss term sees at least $k_2 - k_1 + 1$ steps of history. The price: the overlapping steps are backpropagated more than once." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "BPTT = backpropagation on the RNN unrolled over $T$ steps; the layers share weights.",
        "Recursion: $g_t = W_{hy}^\\top e_t + W_{hh}^\\top\\delta_{t+1}$, $\\delta_t = g_t \\odot (1-h_t^2)$, with $\\delta_{T+1} = 0$.",
        "Shared-weight gradients are sums over time: $\\partial\\mathcal L/\\partial W_{hh} = \\sum_t\\delta_t h_{t-1}^\\top$, $\\partial\\mathcal L/\\partial W_{xh} = \\sum_t\\delta_t x_t^\\top$, $\\partial\\mathcal L/\\partial b = \\sum_t\\delta_t$.",
        "Long-range terms are weighted by $\\partial h_t/\\partial h_k = \\prod D_jW_{hh}$, with $\\|\\cdot\\| \\le (\\gamma\\sigma_{\\max})^{t-k}$: vanishing if $\\sigma_{\\max}(W_{hh}) < 1$; exploding requires $\\sigma_{\\max} > 1$; linear RNNs behave like $\\rho^{t-k}$.",
        "Measured: 32-unit tanh RNNs at initialisation lose about 5 to 24 orders of magnitude of gradient over 50 steps (ρ between 0.5 and 1.5); a vanilla RNN fails the remember-the-first-input task at T = 20.",
        "Truncated BPTT: chunk, carry the state, `detach()`. Memory $O(k)$, but no gradient for dependencies longer than $k$.",
        "Clip the global gradient norm after `backward()`. It fixes explosions, not vanishing.",
        "Orthogonal or identity initialisation helps the matrix part; gating (LSTM/GRU) adds an additive path with factors near 1.",
        "Cost: time $O(T(n_h^2 + n_hn_{in}))$, about 2–3× the forward pass; memory $O(T\\,B\\,n_h)$ for the stored states.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Long Short-Term Memory (LSTM)](/learn/m09/lstm) — the gated cell built to keep the BPTT signal alive: forget, input and output gates and the additive cell-state path.",
        "[GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants) — a lighter gated cell, and how depth and two-way reading interact with BPTT.",
        "[Learning-rate schedules](/learn/m06/lr-schedules) — warm-up, decay and clipping as ways to control step size.",
        "[Attention: the core idea](/learn/m10/attention-intro) — connecting every output straight to every input, which removes the long chain of Jacobians entirely.",
      ] },
    ],
  },
};

export default lesson;
