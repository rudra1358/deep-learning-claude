import type { Lesson } from "@/lib/types";

const GP_NUMPY = String.raw`import numpy as np

# ---------------- 1. scalar intuition: a product of L numbers ----------------
for factor, L in [(0.9, 30), (1.1, 30), (0.25, 10)]:
    print(f"{factor}^{L} = {factor ** L:.3e}")

# ---------------- 2. activations and their derivatives ----------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

ACTS = {   # name: (f, f' expressed through z and the cached activation a)
    "sigmoid": (sigmoid,                     lambda z, a: a * (1 - a)),
    "tanh":    (np.tanh,                     lambda z, a: 1 - a ** 2),
    "relu":    (lambda z: np.maximum(z, 0),  lambda z, a: (z > 0).astype(float)),
}
INITS = {  # name: standard deviation of each weight, as a function of fan-in / fan-out
    "N(0,0.05^2)": lambda n_in, n_out: 0.05,
    "Xavier":      lambda n_in, n_out: np.sqrt(2 / (n_in + n_out)),
    "He":          lambda n_in, n_out: np.sqrt(2 / n_in),
    "N(0,0.5^2)":  lambda n_in, n_out: 0.5,
}


def gradient_flow(act, init, L=30, n=64, B=128, residual=False, seed=0):
    """One forward + backward pass through an L-layer MLP (width n, batch B, zero biases).
    Loss: 0.5 * mean squared error of a linear read-out against random targets.
    Returns activation std per layer, ||delta^[l]|| and ||dL/dW^[l]|| per layer (l = 1..L)."""
    f, fprime = ACTS[act]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(B, n))                     # inputs ~ N(0, 1)
    y = rng.normal(size=(B, 1))                     # random regression targets
    Ws = [rng.normal(0, INITS[init](n, n), (n, n)) for _ in range(L)]
    w_out = rng.normal(0, np.sqrt(1 / n), (1, n))   # linear read-out layer

    h, zs, hs = X, [], [X]                          # cache pre-activations and layer outputs
    for W in Ws:
        z = h @ W.T
        a = f(z)
        h = h + a if residual else a                # residual: h^[l] = h^[l-1] + f(W h^[l-1])
        zs.append(z); hs.append(h)
    pred = h @ w_out.T

    dh = (pred - y) / B @ w_out                     # dL/dh^[L]
    delta_norm, gW_norm = np.zeros(L), np.zeros(L)
    for l in reversed(range(L)):
        a_l = hs[l + 1] - hs[l] if residual else hs[l + 1]
        delta = dh * fprime(zs[l], a_l)             # BP2: delta^[l] = (upstream) * f'(z^[l])
        delta_norm[l] = np.linalg.norm(delta)
        gW_norm[l] = np.linalg.norm(delta.T @ hs[l])   # BP3 summed over the batch
        dh = delta @ Ws[l] + (dh if residual else 0)   # residual adds the identity path
    act_std = np.array([hl.std() for hl in hs[1:]])
    return act_std, delta_norm, gW_norm


print("\nL = 30 layers, width 64, batch 128")
print(f"{'activation':10s} {'init':12s} {'std a[30]':>10s} {'|d1|/|d30|':>11s} {'|dW1|':>9s} {'|dW30|':>9s}")
for act in ACTS:
    for init in INITS:
        s, d, g = gradient_flow(act, init)
        print(f"{act:10s} {init:12s} {s[-1]:10.2e} {d[0] / d[-1]:11.2e} {g[0]:9.2e} {g[-1]:9.2e}")

# ---------------- 3. per-layer profile for one healthy and one vanishing network ----------------
for act, init in [("relu", "He"), ("sigmoid", "Xavier")]:
    _, _, g = gradient_flow(act, init)
    print(f"\n{act}/{init}: ||dL/dW|| at layers 1,5,10,15,20,25,30:")
    print("  " + "  ".join(f"{g[l - 1]:.1e}" for l in (1, 5, 10, 15, 20, 25, 30)))

# ---------------- 4. residual connections keep the backward signal alive ----------------
for act, init in [("tanh", "N(0,0.05^2)"), ("sigmoid", "Xavier")]:
    for res in (False, True):
        _, d, _ = gradient_flow(act, init, residual=res)
        print(f"{act:8s} {init:12s} residual={res!s:5s}  |delta1|/|delta30| = {d[0] / d[-1]:.2e}")`;

const GP_NUMPY_OUT = String.raw`0.9^30 = 4.239e-02
1.1^30 = 1.745e+01
0.25^10 = 9.537e-07

L = 30 layers, width 64, batch 128
activation init          std a[30]  |d1|/|d30|     |dW1|    |dW30|
sigmoid    N(0,0.05^2)    5.02e-02    7.72e-30  1.23e-30  2.98e-01
sigmoid    Xavier         1.22e-01    3.64e-19  5.70e-20  3.15e-01
sigmoid    He             1.69e-01    1.32e-15  2.01e-16  3.42e-01
sigmoid    N(0,0.5^2)     3.53e-01    1.03e-06  1.36e-07  5.12e-01
tanh       N(0,0.05^2)    9.16e-13    2.91e-12  1.72e-12  9.00e-13
tanh       Xavier         1.23e-01    2.06e-01  1.28e-01  6.67e-02
tanh       He             5.83e-01    2.76e+00  1.67e+00  1.04e+00
tanh       N(0,0.5^2)     8.84e-01    1.26e+05  4.24e+04  4.05e-01
relu       N(0,0.05^2)    1.94e-17    1.54e-16  5.23e-17  2.96e-17
relu       Xavier         1.68e-05    5.33e-05  1.81e-05  1.03e-05
relu       He             5.51e-01    1.25e+00  4.81e-01  1.31e+00
relu       N(0,0.5^2)     1.94e+13    1.49e+13  1.33e+26  6.69e+26

relu/He: ||dL/dW|| at layers 1,5,10,15,20,25,30:
  4.8e-01  1.1e+00  1.4e+00  2.2e+00  2.8e+00  1.6e+00  1.3e+00

sigmoid/Xavier: ||dL/dW|| at layers 1,5,10,15,20,25,30:
  5.7e-20  5.0e-17  8.0e-14  1.4e-10  2.2e-07  2.7e-04  3.1e-01
tanh     N(0,0.05^2)  residual=False  |delta1|/|delta30| = 2.91e-12
tanh     N(0,0.05^2)  residual=True   |delta1|/|delta30| = 4.07e+00
sigmoid  Xavier       residual=False  |delta1|/|delta30| = 3.64e-19
sigmoid  Xavier       residual=True   |delta1|/|delta30| = 3.42e+00`;

const GP_RNN = String.raw`import numpy as np

n, T = 32, 60                                     # hidden size, sequence length
for name, gain, act in [("tanh", 0.8, np.tanh), ("tanh", 1.0, np.tanh), ("tanh", 1.5, np.tanh), ("linear", 1.1, None)]:
    rng = np.random.default_rng(1)
    W_hh = rng.normal(0, gain / np.sqrt(n), (n, n))   # recurrent weights, spectral radius ~ gain
    W_xh = rng.normal(0, 1 / np.sqrt(n), (n, n))
    xs = rng.normal(size=(T, n))
    h, hs = np.zeros(n), []
    for t in range(T):                                # forward: h_t = f(W_hh h_{t-1} + W_xh x_t)
        z = W_hh @ h + W_xh @ xs[t]
        h = act(z) if act else z
        hs.append(h)
    # dh_T/dh_{T-k} = prod over the last k steps of diag(f'(z_t)) W_hh  (BP2 through time)
    J, report = np.eye(n), []
    for k in range(1, T):
        fprime = 1 - hs[T - k] ** 2 if act else np.ones(n)
        J = J @ (fprime[:, None] * W_hh)
        if k in (1, 10, 20, 40, 59):
            report.append(f"k={k}: {np.linalg.norm(J, 2):.1e}")
    rho = max(abs(np.linalg.eigvals(W_hh)))
    print(f"{name:6s} spectral radius {rho:.2f} | ||dh_T/dh_(T-k)||_2  " + "  ".join(report))`;

const GP_RNN_OUT = String.raw`tanh   spectral radius 0.82 | ||dh_T/dh_(T-k)||_2  k=1: 1.2e+00  k=10: 6.8e-03  k=20: 8.8e-06  k=40: 1.2e-11  k=59: 3.9e-17
tanh   spectral radius 1.02 | ||dh_T/dh_(T-k)||_2  k=1: 1.5e+00  k=10: 3.8e-02  k=20: 3.1e-04  k=40: 2.1e-08  k=59: 2.7e-12
tanh   spectral radius 1.54 | ||dh_T/dh_(T-k)||_2  k=1: 2.2e+00  k=10: 1.1e+00  k=20: 9.3e-02  k=40: 6.0e-04  k=59: 9.4e-06
linear spectral radius 1.13 | ||dh_T/dh_(T-k)||_2  k=1: 2.1e+00  k=10: 9.5e+00  k=20: 3.3e+01  k=40: 2.6e+02  k=59: 2.5e+03`;

const GP_TORCH_HOOKS = String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
DEPTH, WIDTH, BATCH = 30, 64, 128


class Block(nn.Module):
    """Linear -> (optional LayerNorm) -> activation, optionally wrapped in a residual connection."""

    def __init__(self, width, act, residual=False, layernorm=False):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width) if layernorm else nn.Identity()
        self.act = act()
        self.residual = residual

    def forward(self, h):
        out = self.act(self.norm(self.linear(h)))
        return h + out if self.residual else out


def build(act, init, residual=False, layernorm=False):
    blocks = [Block(WIDTH, act, residual, layernorm) for _ in range(DEPTH)]
    for b in blocks:
        if init == "xavier":
            nn.init.xavier_normal_(b.linear.weight)
        elif init == "he":
            nn.init.kaiming_normal_(b.linear.weight, nonlinearity="relu")
        else:                                   # a plain normal with the given std
            nn.init.normal_(b.linear.weight, 0.0, init)
    return nn.Sequential(*blocks, nn.Linear(WIDTH, 1))


def layer_grad_norms(model, X, y):
    """Log ||dL/dW|| of every hidden Linear layer using tensor hooks on the weights."""
    norms = {}
    handles = []
    hidden = [m.linear for m in model if isinstance(m, Block)]
    for i, lin in enumerate(hidden):
        # the hook runs during backward() with the gradient of this weight
        handles.append(lin.weight.register_hook(lambda g, i=i: norms.__setitem__(i, g.norm().item())))
    loss = 0.5 * ((model(X) - y) ** 2).mean()
    loss.backward()
    for h in handles:
        h.remove()                              # always remove hooks you no longer need
    return [norms[i] for i in range(len(hidden))]


X = torch.randn(BATCH, WIDTH)
y = torch.randn(BATCH, 1)
configs = [
    ("sigmoid, Xavier",            nn.Sigmoid, "xavier", False, False),
    ("tanh, Xavier",               nn.Tanh,    "xavier", False, False),
    ("ReLU, He",                   nn.ReLU,    "he",     False, False),
    ("ReLU, N(0, 0.5^2)",          nn.ReLU,    0.5,      False, False),
    ("sigmoid, Xavier, residual",  nn.Sigmoid, "xavier", True,  False),
    ("ReLU, N(0, 0.5^2), LayerNorm", nn.ReLU,  0.5,      False, True),
]
print(f"{'configuration':30s} {'|dW1|':>9s} {'|dW15|':>9s} {'|dW30|':>9s} {'first/last':>11s}")
for name, act, init, res, ln in configs:
    torch.manual_seed(1)
    model = build(act, init, res, ln)
    g = layer_grad_norms(model, X, y)
    print(f"{name:30s} {g[0]:9.2e} {g[14]:9.2e} {g[-1]:9.2e} {g[0] / g[-1]:11.2e}")`;

const GP_TORCH_HOOKS_OUT = String.raw`configuration                      |dW1|    |dW15|    |dW30|  first/last
sigmoid, Xavier                 5.66e-20  3.53e-12  9.07e-03    6.23e-18
tanh, Xavier                    7.19e-02  9.06e-02  7.14e-02    1.01e+00
ReLU, He                        2.54e-01  2.12e-01  8.00e-02    3.18e+00
ReLU, N(0, 0.5^2)                    inf       inf       inf         nan
sigmoid, Xavier, residual       5.28e-01  1.11e+01  2.03e+01    2.61e-02
ReLU, N(0, 0.5^2), LayerNorm    2.09e-01  2.01e-01  7.45e-02    2.80e+00`;

const GP_TORCH_TRAIN = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import make_moons

torch.set_num_threads(1)                       # small model: threading overhead only slows it down
X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
DEPTH, WIDTH, STEPS, LR = 20, 32, 400, 0.1


def deep_mlp(act, init):
    layers, d = [], 2
    for _ in range(DEPTH):
        lin = nn.Linear(d, WIDTH)
        if init == "xavier":
            nn.init.xavier_normal_(lin.weight)
        elif init == "he":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        else:
            nn.init.normal_(lin.weight, 0.0, init)
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = WIDTH
    return nn.Sequential(*layers, nn.Linear(d, 1))


def train(act, init, clip=None):
    torch.manual_seed(0)
    model = deep_mlp(act, init)
    first, last = model[0].weight, model[2 * DEPTH - 2].weight   # first and 20th hidden layer
    w1_0, w20_0 = first.detach().clone(), last.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    curve = []
    for step in range(STEPS + 1):
        loss = loss_fn(model(X), y)
        opt.zero_grad()
        loss.backward()
        if clip is not None:                   # rescale the whole gradient if its norm exceeds clip
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step()
        if step in (0, 1, 2, 100, 200, 400):
            curve.append(loss.item())
    with torch.no_grad():
        acc = ((model(X) > 0).float() == y).float().mean().item()
    rel = lambda w, w0: ((w.detach() - w0).norm() / w0.norm()).item()
    return curve, acc, rel(first, w1_0), rel(last, w20_0)


runs = [
    ("sigmoid, Xavier",        nn.Sigmoid, "xavier", None),
    ("tanh, Xavier",           nn.Tanh,    "xavier", None),
    ("ReLU, He",               nn.ReLU,    "he",     None),
    ("ReLU, N(0,0.5^2)",       nn.ReLU,    0.5,      None),
    ("ReLU, N(0,0.5^2), clip", nn.ReLU,    0.5,      1.0),
]
print("loss at steps 0, 1, 2, 100, 200, 400 | train acc | relative change of W[1], W[20]")
for name, act, init, clip in runs:
    curve, acc, r1, r20 = train(act, init, clip)
    print(f"{name:23s} " + " ".join(f"{c:9.4g}" for c in curve) + f" | {acc:.3f} | {r1:.1e}, {r20:.1e}")`;

const GP_TORCH_TRAIN_OUT = String.raw`loss at steps 0, 1, 2, 100, 200, 400 | train acc | relative change of W[1], W[20]
sigmoid, Xavier            0.7031    0.6991    0.6967    0.6931    0.6931    0.6931 | 0.500 | 0.0e+00, 2.4e-03
tanh, Xavier               0.6929    0.6859    0.6779    0.3252    0.2184    0.0987 | 0.960 | 6.4e-01, 1.0e-01
ReLU, He                   0.7445    0.6973     0.697    0.1471   0.08828   0.05471 | 0.980 | 4.1e-02, 4.0e-02
ReLU, N(0,0.5^2)        5.286e+04       nan       nan       nan       nan       nan | 0.500 | nan, nan
ReLU, N(0,0.5^2), clip  5.286e+04 1.473e+04      2426     16.04     6.711     1.451 | 0.880 | 1.4e-01, 2.1e-02`;

const GP_CLIP = String.raw`import numpy as np
import torch


def clip_by_global_norm(grads, max_norm, eps=1e-6):
    """Rescale a list of gradient arrays so that their joint L2 norm is at most max_norm."""
    total = np.sqrt(sum(np.sum(g ** 2) for g in grads))     # norm of all gradients as one vector
    scale = min(1.0, max_norm / (total + eps))              # only ever shrink, never enlarge
    return [g * scale for g in grads], total


def clip_by_value(grads, c):
    """Clamp every gradient entry to [-c, c] (changes the direction of the gradient)."""
    return [np.clip(g, -c, c) for g in grads]


rng = np.random.default_rng(0)
grads = [rng.normal(0, 5, (4, 3)), rng.normal(0, 5, (3,))]   # a deliberately large "gradient"
clipped, total = clip_by_global_norm(grads, max_norm=1.0)
print(f"norm before {total:.4f}, after {np.sqrt(sum(np.sum(g ** 2) for g in clipped)):.4f}")

# compare with PyTorch's implementation
params = [torch.zeros(4, 3, dtype=torch.float64, requires_grad=True), torch.zeros(3, dtype=torch.float64, requires_grad=True)]
for p, g in zip(params, grads):
    p.grad = torch.tensor(g)
torch_total = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
diff = max(np.abs(p.grad.numpy() - c).max() for p, c in zip(params, clipped))
print(f"torch total norm {torch_total.item():.4f}, max difference after clipping {diff:.1e}")

# clipping by norm keeps the direction, clipping by value does not
v = [np.array([3.0, 0.1])]
by_norm, _ = clip_by_global_norm(v, 1.0)
by_value = clip_by_value(v, 1.0)
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("by norm :", by_norm[0].round(4), " cosine with original", round(cos(by_norm[0], v[0]), 4))
print("by value:", by_value[0].round(4), " cosine with original", round(cos(by_value[0], v[0]), 4))`;

const GP_CLIP_OUT = String.raw`norm before 18.0015, after 1.0000
torch total norm 18.0015, max difference after clipping 0.0e+00
by norm : [0.9994 0.0333]  cosine with original 1.0
by value: [1.  0.1]  cosine with original 0.9978`;

const GP_TENSORBOARD = String.raw`from torch.utils.tensorboard import SummaryWriter   # needs: pip install tensorboard

writer = SummaryWriter("runs/gradient-flow")
for step, (xb, yb) in enumerate(loader):
    loss = loss_fn(model(xb), yb)
    opt.zero_grad()
    loss.backward()
    if step % 100 == 0:                       # histograms are expensive: log occasionally
        for name, p in model.named_parameters():
            writer.add_histogram(f"weights/{name}", p, step)
            writer.add_histogram(f"grads/{name}", p.grad, step)
            writer.add_scalar(f"grad_norm/{name}", p.grad.norm(), step)
    opt.step()
writer.close()                                # then: tensorboard --logdir runs`;

const GP_KERAS = String.raw`import numpy as np
import tensorflow as tf

DEPTH, WIDTH = 30, 64


def deep_mlp(activation, initializer):
    layers = [tf.keras.Input(shape=(WIDTH,))]
    for _ in range(DEPTH):
        layers.append(tf.keras.layers.Dense(WIDTH, activation=activation, use_bias=False,
                                            kernel_initializer=initializer))
    layers.append(tf.keras.layers.Dense(1))
    return tf.keras.Sequential(layers)


rng = np.random.default_rng(0)
X = tf.constant(rng.normal(size=(128, WIDTH)), dtype=tf.float32)
y = tf.constant(rng.normal(size=(128, 1)), dtype=tf.float32)

for name, act, init in [("sigmoid, Glorot", "sigmoid", "glorot_normal"),
                        ("tanh, Glorot", "tanh", "glorot_normal"),
                        ("relu, He", "relu", "he_normal")]:
    model = deep_mlp(act, init)
    with tf.GradientTape() as tape:
        loss = 0.5 * tf.reduce_mean((model(X) - y) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    norms = [float(tf.norm(g)) for g in grads[:DEPTH]]   # hidden-layer kernels, first to last
    print(f"{name:16s} |dW1| {norms[0]:.2e}  |dW30| {norms[-1]:.2e}  ratio {norms[0] / norms[-1]:.2e}")

# Gradient clipping is a constructor argument of every Keras optimizer:
opt = tf.keras.optimizers.SGD(learning_rate=0.1, clipnorm=1.0)        # per-variable norm
opt_global = tf.keras.optimizers.SGD(learning_rate=0.1, global_clipnorm=1.0)  # joint norm`;

const lesson: Lesson = {
  id: "m04/gradient-problems",
  title: "Vanishing and Exploding Gradients",
  summary: "Why the error signal shrinks or blows up as it travels back through many layers — shown by unrolling BP2, bounded with spectral norms, measured in real 30-layer networks — and how activations, initialisation, normalisation, residual connections, clipping and gating fix it.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m04/backpropagation", "m02/eigen-svd", "m03/activation-functions"],
  next: ["m07/initialization", "m07/normalization", "m07/residual-connections", "m09/bptt"],
  terms: ["vanishing-gradient", "exploding-gradient", "saturation", "spectral-norm", "xavier-init", "he-init", "gradient-clipping", "dead-relu", "residual-connection"],
  sections: {
    intro: [
      { t: "p", text: "In [Backpropagation](/learn/m04/backpropagation) we found $\\delta^{[l-1]} = (W^{[l]\\top}\\delta^{[l]})\\odot f'(z^{[l-1]})$: each layer's error signal comes from the next layer's, multiplied once by a transposed weight matrix and once by the activation slopes. With two or three layers that is harmless. With thirty, the error reaching layer 1 has been multiplied by about thirty matrices and thirty sets of slopes — and a product of many factors almost never stays a sensible size." },
      { t: "p", text: "If the factors are usually below 1, the gradient shrinks exponentially with depth: the **vanishing gradient problem**. Early layers get almost no learning signal and stay near their random starting values, while the last layers do all the learning. If the factors are usually above 1, the gradient grows exponentially: the **exploding gradient problem**. One update throws the weights far away, the loss jumps, and soon the numbers become `inf` and then `NaN`." },
      { t: "p", text: "It was first studied in recurrent networks, which reuse one weight matrix at every time step: Sepp Hochreiter's 1991 diploma thesis (TU München) and Bengio, Simard and Frasconi (1994, *\"Learning long-term dependencies with gradient descent is difficult\"*) showed gradients over long spans vanish or explode exponentially. Glorot and Bengio (2010) found the same in deep feedforward networks and proposed a variance-preserving initialisation; He, Zhang, Ren and Sun (2015, *\"Delving deep into rectifiers\"*) adapted it for ReLU." },
      { t: "p", text: "Many modern habits are partly fixes for this: ReLU-family activations, Xavier/He initialisation, batch and layer normalisation, residual connections, gated units (LSTM, GRU) and gradient clipping. Knowing *why* gradients vanish or explode tells you what each tool does and how to diagnose a network that won't train." },
      { t: "callout", kind: "key", text: "The gradient at layer 1 is the output gradient times a **product of $L-1$ layer Jacobians** $D^{[l]}W^{[l+1]\\top}$. A product of many factors shrinks or grows exponentially unless every factor stays close to norm 1 — and that is exactly what good activations, initialisation, normalisation and skip connections are for." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Backpropagation](/learn/m04/backpropagation) — this lesson continues it. You need BP1–BP4, especially BP2 $\\delta^{[l-1]} = (W^{[l]\\top}\\delta^{[l]})\\odot f'(z^{[l-1]})$ and BP3 $\\partial\\mathcal{L}/\\partial W^{[l]} = \\delta^{[l]}a^{[l-1]\\top}$, plus the notation $W^{[l]}, z^{[l]}, a^{[l]}, \\delta^{[l]}$.",
        "[Activation functions](/learn/m03/activation-functions) — the shapes of sigmoid, tanh and ReLU and their derivatives, especially where each derivative is small.",
        "[Eigenvalues, eigenvectors and SVD](/learn/m02/eigen-svd) — the **spectral norm** $\\|W\\|_2$ (largest singular value): the most a matrix can stretch a vector, $\\|Wv\\| \\le \\|W\\|_2\\|v\\|$. Also the spectral radius (largest $|\\lambda|$), which controls repeated multiplication by the *same* matrix.",
        "[Probability](/learn/m02/probability) — mean, variance, and two rules: the variance of a sum of independent terms is the sum of their variances, and $\\mathrm{Var}(XY) = \\mathrm{Var}(X)\\mathrm{Var}(Y)$ for independent zero-mean $X, Y$. Used for initialisation.",
        "[Gradient descent](/learn/m06/gradient-descent) — the update $W \\leftarrow W - \\eta\\,\\partial\\mathcal{L}/\\partial W$, so you can see why a gradient $10^{-20}$ (or $10^{20}$) times too big is useless.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A message through a chain of amplifiers", text: "A signal goes through 30 amplifiers in a row. If each has a gain of 0.9, the output is $0.9^{30} \\approx 0.042$ of the input — barely audible. If each has a gain of 1.1, the output is $1.1^{30} \\approx 17.4$ times the input — distorted and clipping. Only a gain very close to 1 at every stage keeps the signal usable. No single amplifier is the problem; the **length of the chain** is." },
      { t: "p", text: "In backpropagation, layer 1's learning signal travels through every layer above it, and each layer acts like one amplifier. Its \"gain\" has two parts: the weights (through $W^\\top$) and the activation's slope at the current input ($f'(z)$). Sigmoid's slope is never above $0.25$, so a sigmoid layer has a gain of at most 0.25 unless the weights make up for it. A **saturated** unit — one whose input is far from zero, where the curve is flat — has a slope near 0 and blocks the signal almost completely." },
      { t: "p", text: "In the 30-layer sigmoid network measured below, the first layer's weight gradient is about $10^{-19}$ times the last layer's. With any learning rate safe for the last layer, the first layer never really changes. And when weights are too large, activations and gradients grow layer by layer until they overflow." },
      { t: "p", text: "Every fix makes each stage's gain close to 1: activations with slope 1 (ReLU), weights scaled to preserve variance (Xavier/He), normalisation that resets the scale at each layer, and residual connections that add a direct path with gain exactly 1 around each layer." },
    ],
    mechanism: [
      { t: "p", text: "What happens, step by step, when you train a deep plain MLP $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$, $a^{[l]} = f(z^{[l]})$ with $L$ layers:" },
      { t: "steps", items: [
        { title: "Forward pass: the signal's scale changes layer by layer", text: "Each layer multiplies the activations by $W^{[l]}$ and applies $f$. Weights too small: activations shrink towards 0 (tanh/ReLU) or towards the constant $\\sigma(0) = 0.5$ (sigmoid). Weights too large: pre-activations grow, so tanh and sigmoid saturate at their limits and ReLU outputs grow without bound." },
        { title: "Loss and output error", text: "The loss gives $\\delta^{[L]} = \\nabla_{a^{[L]}}\\mathcal{L}\\odot f'(z^{[L]})$ (BP1). Its size is reasonable: it only depends on how wrong the prediction is." },
        { title: "One backward step = one Jacobian multiplication", text: "BP2 gets $\\delta^{[l-1]}$ from $\\delta^{[l]}$ by multiplying by $W^{[l]\\top}$ and then by the slopes $f'(z^{[l-1]})$. Each step rescales the error by some factor — below 1, above 1, or (rarely, by design) close to 1." },
        { title: "The factors compound", text: "After $L-1$ backward steps, layer 1's error is the output error times the product of all these factors. A factor of 0.5 over 29 steps gives $0.5^{29} \\approx 1.9\\times10^{-9}$; a factor of 2 gives $5.4\\times10^{8}$." },
        { title: "Weight gradients combine both directions", text: "BP3 gives $\\partial\\mathcal{L}/\\partial W^{[l]} = \\delta^{[l]}a^{[l-1]\\top}$: a weight gradient's size is (backward error at layer $l$) × (forward activation entering layer $l$). Shrinking activations and shrinking errors both make weight gradients small." },
        { title: "The update uses one learning rate", text: "Gradient descent multiplies every gradient by the same $\\eta$. If layer 1's gradient is $10^{-19}$ times layer 30's, a learning rate that moves layer 30 sensibly moves layer 1 by nothing (in float32, literally zero — see the training run below). If gradients explode, one step can multiply the weights hugely; the next forward pass overflows to `inf`, `inf - inf` gives `NaN`, and the run is dead." },
      ] },
      { t: "h", text: "Symptoms you can observe" },
      { t: "list", items: [
        "**Vanishing**: training loss stalls early, often at a constant-prediction value (e.g. $\\ln 2 \\approx 0.6931$ for balanced binary cross-entropy). First-layer weights barely move. Per-layer gradient norms fall steadily from output to input. Sigmoid/tanh activations pile up at their limits (saturation), or all activations collapse towards 0.",
        "**Exploding**: loss spikes or becomes `inf`/`NaN`; gradient norms are huge or grow from output to input; weights grow fast; training is extremely sensitive to the learning rate.",
        "**Dead ReLUs**: a ReLU unit whose pre-activation is negative for every input outputs 0 and has $f' = 0$, so it gets no gradient and never recovers. Large learning rates, a big negative bias update or poor initialisation can kill many units; the activation histogram then shows a spike at exactly 0.",
      ] },
      { t: "h", text: "Diagnosing it in practice" },
      { t: "steps", items: [
        { title: "Log per-layer gradient norms", text: "After `loss.backward()`, record $\\|\\partial\\mathcal{L}/\\partial W^{[l]}\\|$ for every layer (from `p.grad`, or with tensor hooks — see the PyTorch code). Plot against layer index on a log scale. In a healthy network the norms are within one or two orders of magnitude; a steady slope across layers means vanishing or exploding." },
        { title: "Log the update-to-weight ratio", text: "$\\eta\\|\\partial\\mathcal{L}/\\partial W\\|/\\|W\\|$ says how much one step changes a layer compared with its size. A common rule of thumb is about $10^{-3}$. Near $10^{-7}$ means the layer is frozen; near 1 means each step rewrites it." },
        { title: "Look at activation histograms", text: "TensorBoard's `add_histogram` (or Weights & Biases) shows each layer's activations and gradients over time. Spikes at $\\pm1$ (tanh saturation), at 0 and 1 (sigmoid saturation) or at exactly 0 (dead ReLUs) stand out clearly." },
        { title: "Track the total gradient norm", text: "`clip_grad_norm_` returns the total norm before clipping; logging it every step shows the spikes that come before a divergence." },
        { title: "Check for non-finite values early", text: "Use `torch.isfinite(loss)` before `backward()`. `torch.autograd.set_detect_anomaly(True)` (slow, debugging only) reports the operation that first produced a `NaN`." },
      ] },
    ],
    math: [
      { t: "h", text: "1. Unrolling BP2 across L layers" },
      { t: "p", text: "Write layer $l$'s activation slopes as a diagonal matrix $D^{[l]} = \\mathrm{diag}\\big(f'(z^{[l]})\\big)$. Multiplying entry by entry by $f'(z^{[l]})$ is the same as multiplying by $D^{[l]}$, so BP2 becomes a plain matrix-vector product:" },
      { t: "math", tex: String.raw`\delta^{[l]} = D^{[l]}\,W^{[l+1]\top}\,\delta^{[l+1]}, \qquad l = L-1, \dots, 1` },
      { t: "p", text: "Here $\\delta^{[l]} = \\partial\\mathcal{L}/\\partial z^{[l]} \\in \\mathbb{R}^{n_l}$, $W^{[l+1]} \\in \\mathbb{R}^{n_{l+1}\\times n_l}$, and $J^{[l]} = D^{[l]}W^{[l+1]\\top}$ is the transpose of the Jacobian $\\partial z^{[l+1]}/\\partial z^{[l]} = W^{[l+1]}D^{[l]}$. Applying it again and again from the output down to layer 1:" },
      { t: "math", tex: String.raw`\delta^{[1]} = D^{[1]}W^{[2]\top}\,D^{[2]}W^{[3]\top}\cdots D^{[L-1]}W^{[L]\top}\,\delta^{[L]} = \Bigg(\prod_{l=1}^{L-1} D^{[l]}W^{[l+1]\top}\Bigg)\delta^{[L]}`, caption: "The product runs left to right from l = 1 to l = L − 1. The first layer's weight gradient is then ∂L/∂W[1] = δ[1] x^T (BP3)." },
      { t: "p", text: "Everything in this lesson follows from this formula: layer 1's backward signal is the output signal times $L-1$ matrices, so its size depends on the sizes of those matrices." },
      { t: "h", text: "2. The scalar picture: a product of L numbers" },
      { t: "p", text: "With one neuron per layer, every matrix is a number and the product is" },
      { t: "math", tex: String.raw`\delta^{[1]} = \Bigg(\prod_{l=1}^{L-1} w^{[l+1]}\, f'\big(z^{[l]}\big)\Bigg)\,\delta^{[L]}` },
      { t: "p", text: "If every factor $|w f'|$ is about $0.9$, thirty layers give $0.9^{30} = 4.239\\times10^{-2}$; if every factor is $1.1$, they give $1.1^{30} = 17.45$. With sigmoid and $|w| \\le 1$, each factor is at most $0.25$, and ten layers give $0.25^{10} = 9.537\\times10^{-7}$. The change is **exponential in depth**, so a small mismatch per layer becomes huge end to end." },
      { t: "h", text: "Worked example: a 4-layer sigmoid chain" },
      { t: "p", text: "One neuron per layer, sigmoid everywhere, no biases, input $x = 1$, weights $w = (0.8,\\ 1.5,\\ -1.2,\\ 2.0)$, target $y = 1$, loss $\\mathcal{L} = (a^{[4]} - y)^2$. All numbers in float64, rounded to six decimals." },
      { t: "steps", items: [
        { title: "Forward pass", tex: String.raw`z = (0.8,\ 1.034962,\ -0.885452,\ 0.584099),\qquad a = \sigma(z) = (0.689974,\ 0.737877,\ 0.292049,\ 0.642010)` },
        { title: "Slopes σ′(z) = a(1 − a)", tex: String.raw`f'(z) = (0.213910,\ 0.193415,\ 0.206756,\ 0.229833),\qquad \mathcal{L} = (0.642010 - 1)^2 = 0.128157` },
        { title: "Output error (BP1)", tex: String.raw`\delta^{[4]} = 2(a^{[4]} - y)\,f'(z^{[4]}) = 2(-0.357990)(0.229833) = -0.164556` },
        { title: "Backward steps (BP2, scalar form δ[l] = w[l+1] f′(z[l]) δ[l+1])", tex: String.raw`\delta^{[3]} = 2.0 \times 0.206756 \times (-0.164556) = -0.068046,\quad \delta^{[2]} = -1.2\times0.193415\times(-0.068046) = 0.015793,\quad \delta^{[1]} = 1.5\times0.213910\times0.015793 = 0.005068` },
        { title: "The three factors and their product", tex: String.raw`w^{[2]}f'(z^{[1]}) = 0.320865,\quad w^{[3]}f'(z^{[2]}) = -0.232098,\quad w^{[4]}f'(z^{[3]}) = 0.413513;\qquad \frac{\delta^{[1]}}{\delta^{[4]}} = 0.320865 \times (-0.232098) \times 0.413513 = -0.030795` },
        { title: "Weight gradients (BP3: δ[l] times the input of layer l)", tex: String.raw`\frac{\partial\mathcal{L}}{\partial w^{[1]}} = 0.005068,\quad \frac{\partial\mathcal{L}}{\partial w^{[2]}} = 0.010897,\quad \frac{\partial\mathcal{L}}{\partial w^{[3]}} = -0.050210,\quad \frac{\partial\mathcal{L}}{\partial w^{[4]}} = -0.048058` },
      ] },
      { t: "p", text: "Two weights are bigger than 1 in size, yet every factor $|w f'|$ is below 0.42, because the sigmoid slopes are all near their ceiling of 0.25. After only three backward steps the error is down to 3% of its output value, and the first weight's gradient is about ten times smaller than the last one's. Thirty such layers would multiply it by roughly $0.3^{29} \\approx 10^{-15}$." },
      { t: "h", text: "3. A bound with spectral norms" },
      { t: "p", text: "For matrices, \"size\" means the **spectral norm** $\\|A\\|_2 = \\sigma_{\\max}(A)$, the largest singular value — the most $A$ can stretch any vector (see [Eigenvalues and SVD](/learn/m02/eigen-svd)). Two facts give a bound: $\\|Av\\| \\le \\|A\\|_2\\|v\\|$ and $\\|AB\\|_2 \\le \\|A\\|_2\\|B\\|_2$. A diagonal matrix's spectral norm is its largest absolute entry, so $\\|D^{[l]}\\|_2 = \\max_j |f'(z^{[l]}_j)| \\le \\|f'\\|_\\infty$, the activation's largest possible slope. Also $\\|W^\\top\\|_2 = \\|W\\|_2$. So" },
      { t: "math", tex: String.raw`\big\|\delta^{[1]}\big\| \;\le\; \prod_{l=1}^{L-1} \big\|D^{[l]}\big\|_2\,\big\|W^{[l+1]}\big\|_2 \;\big\|\delta^{[L]}\big\| \;\le\; \Big(\|f'\|_\infty\Big)^{L-1}\prod_{l=2}^{L}\big\|W^{[l]}\big\|_2\;\big\|\delta^{[L]}\big\|` },
      { t: "p", text: "For sigmoid $\\|f'\\|_\\infty = 1/4$; for tanh and ReLU it is 1. **So:** in a sigmoid network where every weight matrix has $\\|W\\|_2 < 4$, the backward signal is *guaranteed* to shrink at least geometrically, by a factor $\\|W\\|_2/4 < 1$ per layer. Pascanu, Mikolov and Bengio (2013, *\"On the difficulty of training recurrent neural networks\"*) used this sufficient condition for RNNs." },
      { t: "p", text: "Numerical check on a 30-layer sigmoid network of width 64 with Xavier weights (each $\\|W\\|_2 \\approx 1.95$): for one example, the true ratio $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\|$ is $6.6\\times10^{-19}$, and the bound $\\prod\\|D\\|_2\\|W\\|_2$ is $8.4\\times10^{-10}$. The bound holds but is loose: real error vectors don't line up with every matrix's top singular direction, and most slopes are below their maximum." },
      { t: "callout", kind: "note", text: "The bound works one way only. Large norms ($\\|W\\|_2\\|f'\\|_\\infty > 1$) make explosion *possible* but don't guarantee it; small norms *guarantee* vanishing. To predict typical behaviour instead of the worst case, we track variances — Section 5." },
      { t: "h", text: "4. Saturation: why sigmoid and tanh are prone to vanishing" },
      { t: "math", tex: String.raw`\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) \le \tfrac14,\qquad \tanh'(z) = 1 - \tanh^2(z) \le 1,\qquad \mathrm{ReLU}'(z) = \begin{cases}1 & z > 0\\ 0 & z < 0\end{cases}` },
      { t: "p", text: "Sigmoid's slope is $0.25$ at best (at $z = 0$), $0.105$ at $z = 2$ and $0.0066$ at $z = 5$. Tanh's slope is 1 at $z = 0$ but $0.0099$ at $z = 3$. A unit is **saturated** when $|z|$ is big enough that its slope is tiny; it then passes back almost nothing, whatever its weights. Sigmoid has two more problems. Its top slope is 1/4, so even unsaturated sigmoid layers need weights of spectral norm about 4 just to break even. And its outputs are always positive (mean ≈ 0.5), which shifts the next layer's pre-activations and pushes it towards saturation. Tanh is zero-centred with slope 1 at the origin, so with good initialisation it behaves much better, as the measurements below show." },
      { t: "h", text: "5. Variance propagation and the Xavier / He initialisations" },
      { t: "p", text: "Take one layer $z_j = \\sum_{k=1}^{n_{in}} W_{jk}a_k$ (bias 0). The weights are drawn independently with mean 0 and variance $\\mathrm{Var}(W)$, independent of the inputs $a_k$ (which are i.i.d.). Each term $W_{jk}a_k$ has mean 0 and variance $\\mathrm{Var}(W)\\,\\mathbb{E}[a_k^2]$ (a product of independent variables, one zero-mean). Adding $n_{in}$ independent terms:" },
      { t: "math", tex: String.raw`\mathrm{Var}\big(z^{[l]}\big) = n_{in}\,\mathrm{Var}\big(W^{[l]}\big)\;\mathbb{E}\Big[\big(a^{[l-1]}\big)^2\Big]`, caption: "n_in = fan-in of layer l (inputs per neuron)." },
      { t: "p", text: "**Tanh (roughly linear near 0).** For small pre-activations $\\tanh(z) \\approx z$, so $\\mathbb{E}[a^2] \\approx \\mathrm{Var}(z^{[l-1]})$ and $\\mathrm{Var}(z^{[l]}) \\approx n_{in}\\mathrm{Var}(W)\\,\\mathrm{Var}(z^{[l-1]})$. The forward variance stays constant exactly when $n_{in}\\mathrm{Var}(W) = 1$. The same reasoning on BP2, where $\\delta^{[l-1]}_k = f'(z_k)\\sum_j W_{jk}\\delta^{[l]}_j$ sums over the $n_{out}$ neurons the unit feeds, gives $\\mathrm{Var}(\\delta^{[l-1]}) \\approx n_{out}\\mathrm{Var}(W)\\,\\mathrm{Var}(\\delta^{[l]})$, kept constant when $n_{out}\\mathrm{Var}(W) = 1$. Both can't hold unless $n_{in} = n_{out}$, so Glorot and Bengio (2010) took the middle ground — the **Xavier (Glorot)** initialisation:" },
      { t: "math", tex: String.raw`\mathrm{Var}(W) = \frac{2}{n_{in} + n_{out}}\qquad\text{(normal: } W \sim \mathcal{N}(0, \tfrac{2}{n_{in}+n_{out}})\text{; uniform: } W \sim \mathcal{U}\big[-\sqrt{\tfrac{6}{n_{in}+n_{out}}},\ \sqrt{\tfrac{6}{n_{in}+n_{out}}}\big]\text{)}` },
      { t: "p", text: "**ReLU.** If $z$ is symmetric around 0, half of it is zeroed, so $\\mathbb{E}[\\mathrm{ReLU}(z)^2] = \\tfrac12\\mathrm{Var}(z)$ (a Monte-Carlo check with $10^6$ samples gives $0.5007$). Then $\\mathrm{Var}(z^{[l]}) = \\tfrac12 n_{in}\\mathrm{Var}(W)\\,\\mathrm{Var}(z^{[l-1]})$, kept constant when $\\tfrac12 n_{in}\\mathrm{Var}(W) = 1$ — the **He (Kaiming)** initialisation of He et al. (2015):" },
      { t: "math", tex: String.raw`\mathrm{Var}(W) = \frac{2}{n_{in}}\qquad\Longrightarrow\qquad \mathrm{Var}\big(z^{[l]}\big) = \mathrm{Var}\big(z^{[l-1]}\big)` },
      { t: "p", text: "Worked numbers for $n_{in} = 256$, $n_{out} = 128$: Xavier gives $\\mathrm{Var}(W) = 2/384 = 0.005208$ (std $0.07217$); He gives $\\mathrm{Var}(W) = 2/256 = 0.0078125$ (std $0.08839$). Xavier in a deep ReLU network with square layers ($n_{in} = n_{out} = n$, so $\\mathrm{Var}(W) = 1/n$) halves the variance every layer: after 29 transitions the std should shrink by $2^{-29/2} = 4.3\\times10^{-5}$. The NumPy measurement below gives $1.68\\times10^{-5}/0.578 = 2.9\\times10^{-5}$ — same order; the gap comes from finite width and batch. Full details (fan modes, uniform vs normal, gains for other activations, orthogonal initialisation) are in [Weight Initialization](/learn/m07/initialization)." },
      { t: "callout", kind: "tip", text: "Initialisation fixes the scale only **at the start** of training. It can't stop the scale drifting as weights change, or remove saturation that appears later. That is why normalisation and residual connections are added on top." },
      { t: "h", text: "6. Residual connections: the +I term" },
      { t: "p", text: "A residual block computes $h^{[l]} = h^{[l-1]} + F(h^{[l-1]})$, where $F$ is a small sub-network (e.g. linear → activation). Its Jacobian contains the identity:" },
      { t: "math", tex: String.raw`\frac{\partial h^{[l]}}{\partial h^{[l-1]}} = I + \frac{\partial F}{\partial h^{[l-1]}}\qquad\Longrightarrow\qquad \frac{\partial\mathcal{L}}{\partial h^{[1]}} = \prod_{l=2}^{L}\Big(I + J_F^{[l]\top}\Big)\frac{\partial\mathcal{L}}{\partial h^{[L]}} = \frac{\partial\mathcal{L}}{\partial h^{[L]}} + \sum_{l} J_F^{[l]\top}\frac{\partial\mathcal{L}}{\partial h^{[L]}} + \cdots` },
      { t: "p", text: "Expand the product: one term is the output gradient itself, carried unchanged by the chain of identities; the other terms pass through some of the blocks. Even if every $J_F$ is tiny, that first term stops the gradient at layer 1 from vanishing. In the measurements below, adding residual connections to a 30-layer sigmoid network changes $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\|$ from $3.6\\times10^{-19}$ to $3.4$. Residuals alone don't stop *forward* growth (each block adds to $h$), so they are paired with normalisation or a scaled initialisation of each block's last layer. See [Residual Connections](/learn/m07/residual-connections)." },
      { t: "h", text: "7. The same problem through time: RNNs" },
      { t: "p", text: "A vanilla RNN computes $h_t = f(W_{hh}h_{t-1} + W_{xh}x_t)$. [Backpropagation through time](/learn/m09/bptt) applies BP2 along the time axis, and since the *same* $W_{hh}$ is used at every step," },
      { t: "math", tex: String.raw`\frac{\partial h_T}{\partial h_{T-k}} = \prod_{t=T-k+1}^{T} \mathrm{diag}\big(f'(z_t)\big)\,W_{hh}` },
      { t: "p", text: "For a linear RNN the product is $W_{hh}^k$, which grows or decays like $\\rho(W_{hh})^k$, where $\\rho$ is the spectral radius (largest $|\\lambda|$). With tanh, the slopes multiply in and push towards vanishing. The program below measures $\\|\\partial h_T/\\partial h_{T-k}\\|_2$ for a 32-unit RNN. With tanh, an input 59 steps back has $10^{-17}$ to $10^{-5}$ of the influence of the latest one, even when $\\rho = 1.54$; a linear RNN with $\\rho = 1.13$ explodes to $2.5\\times10^{3}$. This is the difficulty Hochreiter (1991) and Bengio et al. (1994) found, and the reason for the gated cell state of the [LSTM](/learn/m09/lstm)." },
      { t: "details", summary: "Code: measuring ∂h_T/∂h_(T−k) in a vanilla RNN", blocks: [
        { t: "code", lang: "python", title: "rnn_gradient_through_time.py", code: GP_RNN, output: GP_RNN_OUT },
      ] },
    ],
    visual: [
      { t: "p", text: "The simulator runs a real forward and backward pass through an MLP of 2–50 layers on a fixed random batch of 32 inputs. Top chart: each layer's output standard deviation. Bottom chart: $\\|\\partial\\mathcal{L}/\\partial W^{[l]}\\|$ (bars) and $\\|\\delta^{[l]}\\|$ (dashed), on a log scale. Press **Play**: forward bars fill left → right, then gradient bars right → left, with the real first/last ratio in the narration. The histogram shows one layer's activations (click a bar to pick it)." },
      { t: "sim", id: "gradient-flow", caption: "Depth, width, activation, initialisation, residual connections and LayerNorm are all live: every change recomputes both passes." },
      { t: "list", items: [
        "**Sigmoid, Xavier (vanishing)** (default, 30 layers): the gradient bars form a straight ramp falling from right to left. With the default sample the narration reports $\\|\\partial\\mathcal{L}/\\partial W^{[1]}\\|/\\|\\partial\\mathcal{L}/\\partial W^{[30]}\\| \\approx 2.3\\times10^{-19}$ — about 0.23 per layer, close to sigmoid's slope ceiling of 0.25.",
        "**ReLU, He (healthy)**: activation std stays around 1 and the gradient bars are nearly flat (first/last ≈ 1.5). Now change *only* the init to **Xavier**: the forward std falls by about $\\sqrt2$ per layer (to about $10^{-5}$ at layer 30) and $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\|$ drops to about $10^{-5}$ — the factor $\\tfrac12$ per layer the variance analysis predicts.",
        "**ReLU, σ = 0.5 (exploding)**: activations reach about $10^{13}$ at layer 30 and $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\| \\approx 4\\times10^{12}$. The weight-gradient *ratio* is near 1 — growing $\\delta$ and backwards-shrinking activations cancel in $\\delta a^\\top$ — but every weight gradient is about $10^{24}$ times the weights' size, so any learning rate wrecks the network. The table's \"max ‖∂L/∂W‖/‖W‖\" row shows this.",
        "**Tanh, σ = 0.05 (collapse)**: the forward signal decays to about $10^{-12}$ and weight gradients are tiny *everywhere*. Vanishing isn't only a front-versus-back imbalance.",
        "**Sigmoid + residual**: the same sigmoid network now has $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\| \\approx 3.8$ — the identity path carries the gradient. Check the last layers' histogram: their sigmoid units are heavily saturated (the residual stream has grown), yet the gradient still arrives.",
        "**ReLU σ = 0.5 + LayerNorm**: normalising each example's pre-activations keeps the forward std between 0.53 and 0.68 at every layer and brings gradients back to a healthy range, despite the bad initialisation.",
        "With ReLU, look at layer 30's histogram: about 44% of units are zero for all 32 examples. In a deep, bias-free ReLU network, different inputs look more and more alike with depth, so a unit tends to be on for all of them or off for all of them; the \"dead for this batch\" units pass no gradient. The fraction is the same for every ReLU init scale: multiplying all weights by a constant never flips the sign of a ReLU pre-activation.",
        "Set depth to 2 or 3 with sigmoid: the problem disappears. It is a property of **depth**.",
      ] },
      { t: "p", text: "The [neural network visualizer](/learn/m04/backpropagation) from the previous lesson shows the same effect neuron by neuron: build a network with four sigmoid hidden layers and compare the $\\delta$ under the first hidden layer with the output layer's." },
      { t: "sim", id: "network", caption: "Add hidden layers with sigmoid activations and step through the backward pass: each layer's δ is smaller than the one after it." },
    ],
    scratch: [
      { t: "p", text: "A NumPy experiment that measures gradient flow directly. It builds a 30-layer MLP of width 64 and runs one forward and one backward pass (BP1–BP3 written by hand) on 128 random examples. For each activation/initialisation pair it reports the last layer's activation std, the ratio $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\|$, and the first- and last-layer weight-gradient norms. Then it prints the per-layer profile of a healthy and a vanishing network, and the effect of residual connections." },
      { t: "code", lang: "python", title: "gradient_flow_numpy.py", code: GP_NUMPY, output: GP_NUMPY_OUT },
      { t: "h", text: "Reading the table" },
      { t: "table", head: ["activation", "init", "std a[30]", "‖δ¹‖/‖δ³⁰‖", "‖∂L/∂W¹‖", "‖∂L/∂W³⁰‖", "diagnosis"], rows: [
        ["sigmoid", "Xavier", "0.122", "3.6e-19", "5.7e-20", "0.315", "vanishing; no initialisation rescues sigmoid (best: 1.0e-6 with σ = 0.5)"],
        ["tanh", "N(0, 0.05²)", "9.2e-13", "2.9e-12", "1.7e-12", "9.0e-13", "forward collapse: everything tiny"],
        ["tanh", "Xavier", "0.123", "0.21", "0.128", "0.067", "healthy (slow forward decay)"],
        ["tanh", "N(0, 0.5²)", "0.884", "1.3e5", "4.2e4", "0.405", "exploding backwards, saturated forward"],
        ["ReLU", "Xavier", "1.7e-5", "5.3e-5", "1.8e-5", "1.0e-5", "variance halves each layer"],
        ["ReLU", "He", "0.551", "1.25", "0.481", "1.31", "healthy"],
        ["ReLU", "N(0, 0.5²)", "1.9e13", "1.5e13", "1.3e26", "6.7e26", "exploding (forward and backward)"],
      ] },
      { t: "list", items: [
        "**Sigmoid vanishes under every initialisation.** Bigger weights help (σ = 0.5 gives $10^{-6}$ instead of $10^{-19}$) but can't fix it: slopes are at most 0.25 and outputs aren't zero-centred.",
        "**The right initialisation depends on the activation**: Xavier for tanh, He for ReLU. ReLU with Xavier loses half its variance each layer. Tanh with σ = 0.5 (gain 4) is *chaotic*: pre-activations are large and units saturate forward, yet the backward signal explodes.",
        "**Forward and backward scales interact.** For ReLU with N(0, 0.05²), $\\|\\delta\\|$ shrinks backwards while activations shrink *forwards*, so layers 1 and 30 have similar weight-gradient norms — both about $10^{-17}$. Always look at absolute sizes, not only ratios.",
        "In the per-layer profile, sigmoid/Xavier loses about three orders of magnitude every five layers: a straight line on a log scale, i.e. exponential decay with a constant per-layer factor.",
        "The residual rows: the identity path lifts $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\|$ from $2.9\\times10^{-12}$ to $4.1$ (tanh, tiny init) and from $3.6\\times10^{-19}$ to $3.4$ (sigmoid).",
        "Implementation detail: `fprime` receives the saved activation `a`, so sigmoid and tanh derivatives are $a(1-a)$ and $1-a^2$, as in the backpropagation lesson. For residual layers the branch output is recovered as `hs[l + 1] - hs[l]`, and `dh` gets the extra identity term.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**1. Diagnosing with hooks.** `tensor.register_hook(fn)` calls `fn(grad)` when that tensor's gradient is computed in `backward()`. One hook per layer weight records per-layer gradient norms without touching the training loop. Same 30-layer, width-64 setups as the NumPy experiment, plus a residual and a LayerNorm variant (float32, PyTorch's default):" },
          { t: "code", lang: "python", title: "gradient_norm_hooks.py", code: GP_TORCH_HOOKS, output: GP_TORCH_HOOKS_OUT },
          { t: "p", text: "In float32 the badly initialised ReLU network overflows: activations around $10^{13}$ give a squared error around $10^{26}$ and gradients beyond float32's maximum ($\\approx 3.4\\times10^{38}$), so the norms are `inf` and the ratio `nan`. LayerNorm before each activation repairs the same weights (first/last = 2.8). With residual connections, the sigmoid network's first-layer gradient goes from $5.7\\times10^{-20}$ to $0.53$." },
          { t: "p", text: "**2. What happens in training.** Twenty layers of width 32 on two-moons data (400 points), full-batch SGD with $\\eta = 0.1$ for 400 steps; `clip_grad_norm_` rescales the whole gradient when its norm exceeds 1:" },
          { t: "code", lang: "python", title: "deep_training.py", code: GP_TORCH_TRAIN, output: GP_TORCH_TRAIN_OUT },
          { t: "list", items: [
            "**Sigmoid**: loss settles at $0.6931 = \\ln 2$ — the network predicts 0.5 for everything — and accuracy is 50%. The first layer's relative weight change is **exactly zero**: its gradient (about $10^{-14}$) times $\\eta$ is far below the gap between neighbouring float32 numbers at the weights' size, so `w - lr * grad` rounds back to `w`.",
            "**Tanh + Xavier** and **ReLU + He** reach 96% and 98% accuracy, and their first layers move as much as or more than their last layers.",
            "**ReLU with N(0, 0.5²)**: the initial loss is 52 861, the first update is enormous, and from step 1 the loss is `nan`. The run can't recover.",
            "**Same network with clipping** survives: each update is limited to norm 1, the loss falls from 52 861 to 1.45 and accuracy reaches 88%. Clipping stops the disastrous step but doesn't fix the bad scaling — still far worse than He initialisation.",
          ] },
          { t: "p", text: "**3. Histograms in TensorBoard.** In a real training loop, log weight and gradient histograms now and then. (TensorBoard isn't installed in the verification environment, so this snippet was not run; `model`, `loader`, `loss_fn` and `opt` are your own.)" },
          { t: "code", lang: "python", title: "tensorboard_histograms.py (not executed in CI)", code: GP_TENSORBOARD, output: "Writes event files under runs/gradient-flow. In TensorBoard's Histograms and Distributions tabs, a vanishing network shows gradient histograms of the first layers collapsed to a spike at 0, while saturated tanh/sigmoid activations pile up at their limits." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "`tf.GradientTape` gives all gradients at once; their per-layer norms show the same pattern. Keras optimizers accept `clipnorm` (each variable clipped separately), `global_clipnorm` (all gradients clipped together, like PyTorch's `clip_grad_norm_`) and `clipvalue`." },
          { t: "code", lang: "python", title: "gradient_norms_keras.py (not executed in CI)", code: GP_KERAS, output: "Prints one line per configuration with the first- and last-layer kernel-gradient norms and their ratio. The sigmoid/Glorot network shows a ratio many orders of magnitude below 1, while the tanh/Glorot and ReLU/He networks show ratios of order 1 (exact values differ from the PyTorch run because of different random weights)." },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "Forms of the problem" },
      { t: "table", head: ["Form", "Where it appears", "Main cause", "Typical symptom"], rows: [
        ["Vanishing in depth", "Deep plain MLPs/CNNs, especially sigmoid/tanh", "Per-layer factors ‖D W‖ < 1: small weights, flat slopes", "Early layers stop learning; loss stalls"],
        ["Exploding in depth", "Deep nets with large init or unnormalised ReLU stacks", "Per-layer factors > 1", "Loss spikes, inf/NaN; extreme LR sensitivity"],
        ["Forward collapse", "Too-small init with tanh/ReLU", "Activations shrink to 0, so δ·aᵀ is tiny everywhere", "All gradients tiny, not just early ones"],
        ["Saturation", "Sigmoid/tanh with large pre-activations", "f′(z) ≈ 0 for |z| large", "Activation histograms piled at the limits"],
        ["Dead ReLUs", "ReLU networks after large updates or with bad bias/init", "Pre-activation negative for all inputs ⇒ f′ = 0", "Spike at exactly 0; units never recover"],
        ["Vanishing through time", "Vanilla RNNs on long sequences", "Repeated diag(f′)·W_hh with ρ(W_hh) and slopes < 1", "Can't learn long-range dependencies"],
        ["Exploding through time", "RNNs with ρ(W_hh) > 1", "Powers of W_hh", "Sudden huge gradients on some sequences"],
      ] },
      { t: "h", text: "Remedies" },
      { t: "table", head: ["Remedy", "How it keeps the per-layer factor near 1", "Limitations", "Used in"], rows: [
        ["ReLU-family activations ([activations](/learn/m03/activation-functions))", "Slope exactly 1 on the active side; no saturation for z > 0", "Dead units (slope 0 for z < 0); Leaky ReLU/GELU/SiLU soften this", "Almost all modern CNNs and MLPs; GELU/SiLU in Transformers"],
        ["Careful initialisation ([Xavier/He](/learn/m07/initialization))", "Picks Var(W) so forward/backward variance is kept at the start", "Only fixes the starting point; assumes independence", "Default initialisers in every framework"],
        ["Batch / layer normalisation ([normalization](/learn/m07/normalization))", "Rescales activations to zero mean, unit variance at every layer, every step", "Extra computation; BatchNorm depends on batch size", "BatchNorm in CNNs; LayerNorm in Transformers and RNNs"],
        ["Residual connections ([residual](/learn/m07/residual-connections))", "Adds +I to each block's Jacobian: a direct gradient path", "Forward scale can grow; needs normalisation or scaled init", "ResNets, Transformers, most deep architectures"],
        ["Gradient clipping ([clipping](/learn/m06/lr-schedules))", "Caps the update size when the gradient explodes", "Treats the symptom; does nothing for vanishing", "RNN and Transformer training"],
        ["Gated units ([LSTM](/learn/m09/lstm), GRU)", "Additive cell-state path whose gate can keep factor ≈ 1 over many steps", "More parameters and computation", "Sequence models before Transformers; still common"],
        ["Smaller learning rate / warmup", "Stops big updates from pushing units into saturation or death", "Slower training; doesn't fix vanishing", "Everywhere; warmup standard for Transformers"],
      ] },
      { t: "p", text: "**Gradient clipping in detail.** *By norm*: if $\\|g\\| > c$, replace $g$ with $c\\,g/\\|g\\|$ — same direction, limited length. Applied to all parameters together it's called *global-norm* clipping. *By value*: clamp each component to $[-c, c]$ — cheaper, but it changes the update's direction. Norm clipping is the usual choice (exercise 2 implements both)." },
    ],
    applications: [
      { t: "list", items: [
        "**Very deep CNNs.** Residual connections let He et al. train networks with over 100 layers (ResNet-152, \"Deep Residual Learning for Image Recognition\", 2016); plain networks got *higher* training error as depth grew (the degradation problem). Batch normalisation (Ioffe and Szegedy, 2015) and He initialisation are part of the same recipe.",
        "**Transformers and large language models.** Every Transformer block is wrapped in a residual connection with layer normalisation (Ba, Kiros and Hinton, 2016). Large-model recipes routinely combine learning-rate warmup with global-norm clipping to survive occasional gradient spikes.",
        "**Sequence modelling with RNNs.** The LSTM (Hochreiter and Schmidhuber, 1997) was designed to avoid vanishing gradients through time. LSTMs and GRUs, trained with gradient clipping, powered speech recognition, translation and language modelling before Transformers.",
        "**Training monitoring.** Production pipelines log the global gradient norm and per-layer gradient statistics every step (TensorBoard, Weights & Biases), because a rising gradient norm is often the first warning of divergence.",
        "**Debugging a model that won't learn.** When a new model sits at a constant loss, per-layer gradient norms and activation histograms are early checks: they separate \"no gradient reaches these layers\" from \"the data or labels are wrong\".",
      ] },
    ],
    tradeoffs: [
      { t: "p", text: "There is no single fix; each remedy has a cost and targets one part of the product of Jacobians." },
      { t: "list", items: [
        "**Saturating activations (sigmoid, tanh)** give bounded outputs, useful for gates and probabilities, but are the main cause of vanishing in deep stacks. Keep them where a bounded output is needed (LSTM gates, output layers); use ReLU-family activations in hidden layers.",
        "**ReLU** removes saturation on the positive side and is cheap, but brings dead units and isn't zero-centred. Leaky ReLU, GELU and SiLU cost a little more for a non-zero gradient everywhere.",
        "**Initialisation** is free and essential, but its guarantees are statistical and hold only at step 0. It assumes independent inputs and weights, which training breaks.",
        "**Normalisation** keeps scales in check during training and allows higher learning rates. Costs: extra computation and memory, and (for BatchNorm) dependence on batch statistics plus different train/test behaviour.",
        "**Residual connections** are the most robust fix for depth and almost free. But the forward scale grows with depth unless paired with normalisation or scaled initialisation, and block input and output shapes must match (or be projected).",
        "**Gradient clipping** only guards against explosion. A threshold set too low quietly turns training into normalised-gradient descent with a smaller effective learning rate.",
        "**Gating (LSTM/GRU)** handles vanishing through time far better than vanilla RNNs, but has about four times the parameters of a vanilla RNN cell and still runs step by step in time — one reason Transformers replaced them for long sequences.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Using sigmoid in the hidden layers of a deep network.** Its slope is at most 0.25, so the backward signal shrinks about 4× per layer even with ideal weights. Use ReLU-family or tanh hidden layers; keep sigmoid for outputs and gates.",
        "**Mismatching activation and initialisation.** Xavier with ReLU halves the variance every layer; He with tanh pushes units towards saturation. Use Xavier for tanh/linear, He for ReLU-family (with the right gain for leaky slopes).",
        "**Using a fixed standard deviation such as 0.01 or 1.** The right scale depends on fan-in; a std that works for width 64 is wrong for width 4096. Use $\\sqrt{2/n_{in}}$-style formulas.",
        "**Looking only at the first/last gradient ratio.** Both ends can be tiny (forward collapse) or both enormous (ReLU with large weights, where the ratio is about 1). Check absolute norms and the update-to-weight ratio too.",
        "**Treating a NaN loss as bad luck.** It's almost always an exploding gradient or an invalid operation (log of 0, division by 0). Lower the learning rate, add clipping, check the initialisation, and use `torch.autograd.set_detect_anomaly(True)` to find the first non-finite value.",
        "**Clipping by value when you meant by norm.** Value clipping changes the gradient's direction; norm clipping keeps it. Clip *after* `backward()` and *before* `optimizer.step()` — and with mixed-precision loss scaling, unscale the gradients first.",
        "**Expecting clipping to fix vanishing gradients.** Clipping only shrinks large gradients; it can't enlarge small ones.",
        "**Thinking residual connections alone are enough.** Without normalisation, the residual stream's variance grows with depth (each block adds to it), which can saturate later layers — see the simulator's \"Sigmoid + residual\" preset.",
        "**Ignoring dead ReLUs.** A high learning rate can push many units permanently negative. Watch the fraction of zero activations; use a smaller learning rate, Leaky ReLU/GELU, or proper initialisation.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the vanishing gradient problem?", a: "In a deep network, the loss gradient for the early layers' weights becomes extremely small. Backpropagation multiplies the output error by one factor per layer (weights and activation slopes), and a product of many factors below 1 decays exponentially. Early layers then learn very slowly or not at all." },
      { t: "qa", level: "beginner", q: "Why is sigmoid particularly prone to vanishing gradients?", a: "Its derivative $\\sigma(z)(1-\\sigma(z))$ is at most 0.25 (at $z=0$) and nearly 0 for large $|z|$ (saturation). So each sigmoid layer multiplies the backward signal by at most 0.25 times the weights' scale. Its outputs are also always positive, which shifts the next layer's inputs and encourages saturation." },
      { t: "qa", level: "beginner", q: "What are the symptoms of exploding gradients, and a quick fix?", a: "The loss jumps or becomes inf/NaN, gradient norms are huge, weights grow fast. Quick fixes: lower the learning rate and clip the gradient norm (`torch.nn.utils.clip_grad_norm_`). Then fix the cause — initialisation, normalisation, architecture." },
      { t: "qa", level: "intermediate", q: "Derive why δ at layer 1 is a product of matrices.", a: "BP2 is $\\delta^{[l]} = D^{[l]}W^{[l+1]\\top}\\delta^{[l+1]}$ with $D^{[l]} = \\mathrm{diag}(f'(z^{[l]}))$. Applying it from $l = L-1$ down to 1 gives $\\delta^{[1]} = D^{[1]}W^{[2]\\top}\\cdots D^{[L-1]}W^{[L]\\top}\\delta^{[L]}$. Taking norms, $\\|\\delta^{[1]}\\| \\le \\prod\\|D^{[l]}\\|_2\\|W^{[l+1]}\\|_2\\|\\delta^{[L]}\\|$, so if every factor is below 1 the gradient decays at least geometrically." },
      { t: "qa", level: "intermediate", q: "Derive He initialisation.", a: "For $z = \\sum_{k=1}^{n_{in}} W_k a_k$ with i.i.d. zero-mean weights independent of the inputs, $\\mathrm{Var}(z) = n_{in}\\mathrm{Var}(W)\\mathbb{E}[a^2]$. For ReLU with symmetric $z^{[l-1]}$, $\\mathbb{E}[a^2] = \\tfrac12\\mathrm{Var}(z^{[l-1]})$. Keeping variance constant needs $\\tfrac12 n_{in}\\mathrm{Var}(W) = 1$, i.e. $\\mathrm{Var}(W) = 2/n_{in}$. Xavier's $2/(n_{in}+n_{out})$ comes from the linear (tanh near 0) case, averaging the forward condition $1/n_{in}$ and the backward condition $1/n_{out}$." },
      { t: "qa", level: "intermediate", q: "How do residual connections help gradient flow?", a: "A block $h^{[l]} = h^{[l-1]} + F(h^{[l-1]})$ has Jacobian $I + \\partial F/\\partial h$. The product over blocks expands into a sum whose first term is the identity, so the output gradient reaches every layer directly and undiminished, even if each $\\partial F/\\partial h$ is small. In a 30-layer sigmoid MLP this changes $\\|\\delta^{[1]}\\|/\\|\\delta^{[30]}\\|$ from about $10^{-19}$ to about 3." },
      { t: "qa", level: "intermediate", q: "Gradient clipping by norm vs by value?", a: "By norm: if the (global) norm exceeds $c$, scale the whole gradient by $c/\\|g\\|$; the direction is kept. By value: clamp each component to $[-c, c]$; the direction changes because big components are cut more than small ones. Norm clipping is standard; it protects against explosion but does nothing for vanishing." },
      { t: "qa", level: "advanced", q: "Why do vanilla RNNs have more trouble than feedforward nets of the same depth, and how does the LSTM fix it?", a: "An RNN uses the same $W_{hh}$ at every step, so $\\partial h_T/\\partial h_{T-k}$ contains $W_{hh}$ $k$ times. Its size behaves like $\\rho(W_{hh})^k$ times the product of the slopes — nothing averages out, and you can't pick a different scale per step. The LSTM adds a cell state updated additively, $c_t = f_t\\odot c_{t-1} + i_t\\odot\\tilde c_t$, so $\\partial c_t/\\partial c_{t-1}$ contains $\\mathrm{diag}(f_t)$. With forget gates near 1 the gradient can pass many steps with little decay — the same idea as a residual connection's +I." },
      { t: "qa", level: "advanced", q: "A ReLU network with large initial weights has first-to-last weight-gradient ratio ≈ 1. Is it healthy?", a: "Not necessarily. Bias-free ReLU networks are positively homogeneous: scaling all weights by $c$ scales layer-$l$ activations by $c^l$ and $\\delta^{[l]}$ by $c^{L-l}$, so $\\|\\delta^{[l]}a^{[l-1]\\top}\\|$ scales the same way at every layer. The ratio stays near 1 while every gradient is astronomically large compared with its weights (in the simulator, about $10^{24}$ times). The right checks are the update-to-weight ratio $\\eta\\|\\nabla W\\|/\\|W\\|$ and the absolute activation scale." },
      { t: "qa", level: "advanced", q: "Does Adam solve vanishing gradients?", a: "Only partly. Adam divides each parameter's step by a running RMS of its gradient, so a small but consistent gradient still gives steps of about $\\eta$. But when the gradient is mostly noise or float underflow, or exactly zero (dead ReLUs, saturated units in float32), there is no signal to rescale, and information about early layers still passes through many small Jacobians. Architectural fixes (residuals, normalisation, good activations) are still needed." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A gradient passes back through 30 layers, each multiplying its norm by 0.9. What fraction of the original norm reaches layer 1? (3 decimals)", answer: 0.042, tolerance: 0.001, explain: "$0.9^{30} = 0.0424$ — exponential decay in depth." },
        { kind: "numeric", q: "Xavier initialisation: what is $\\mathrm{Var}(W)$ for a layer with $n_{in} = 300$, $n_{out} = 100$?", answer: 0.005, tolerance: 0.0001, explain: "$2/(300 + 100) = 0.005$." },
        { kind: "numeric", q: "He initialisation: what is the standard deviation of the weights for a layer with $n_{in} = 512$?", answer: 0.0625, tolerance: 0.0005, explain: "$\\sqrt{2/512} = \\sqrt{1/256} = 0.0625$." },
        { kind: "numeric", q: "A sigmoid network has 11 layers and every weight matrix has spectral norm 2. Using $\\|\\delta^{[1]}\\| \\le \\prod(\\|W\\|_2\\cdot 0.25)\\|\\delta^{[L]}\\|$ over the 10 backward steps, what is the upper bound on $\\|\\delta^{[1]}\\|/\\|\\delta^{[11]}\\|$? Enter the value × 1000 (3 decimals).", answer: 0.977, tolerance: 0.002, explain: "Each factor is at most $2\\times0.25 = 0.5$; $0.5^{10} = 9.766\\times10^{-4}$, i.e. 0.977 × 10⁻³." },
        { q: "Which statement about residual connections is correct?", options: ["They make the Jacobian of each block equal to the identity, so the block learns nothing", "They add an identity term to each block's Jacobian, giving the gradient a direct path to early layers", "They remove the need for any initialisation scheme or normalisation", "They prevent exploding gradients by clipping the update"], answer: 1, explain: "$\\partial h^{[l]}/\\partial h^{[l-1]} = I + \\partial F/\\partial h^{[l-1]}$. The block still learns $F$; the forward scale can still grow, so normalisation is still used." },
        { q: "A 20-layer sigmoid network sits at a training loss of exactly 0.6931 on a balanced binary task, and its first-layer weights have not changed at all after 400 steps. The most likely cause is:", options: ["The learning rate is too large", "Vanishing gradients: the first layers receive gradients too small to change float32 weights", "Exploding gradients", "Overfitting"], answer: 1, explain: "$\\ln 2 = 0.6931$ is the BCE of predicting 0.5 everywhere. With a first-layer gradient around $10^{-14}$, $\\eta g$ is smaller than the float32 spacing at the weights' size, so the weights don't change." },
        { q: "Gradient clipping by global norm…", options: ["…fixes vanishing gradients by enlarging small gradients", "…rescales the full gradient when its norm exceeds a threshold, preserving its direction", "…clamps each gradient component independently", "…is applied before backward()"], answer: 1, explain: "Norm clipping multiplies the whole gradient by $c/\\|g\\|$ when $\\|g\\| > c$. Clamping each component is clipping by value. It goes after `backward()` and before `step()`." },
        { kind: "numeric", q: "For $z \\sim \\mathcal{N}(0, 4)$, what is $\\mathbb{E}[\\mathrm{ReLU}(z)^2]$?", answer: 2, tolerance: 0.01, explain: "By symmetry, half of the second moment survives: $\\tfrac12\\mathrm{Var}(z) = 2$. This $\\tfrac12$ is why He uses $2/n_{in}$." },
      ] },
      { t: "exercise", title: "Predict the collapse of a ReLU network with Xavier initialisation", prompt: "A 20-layer ReLU MLP with square layers of width 512 (biases 0) uses Xavier, i.e. $\\mathrm{Var}(W) = 2/(512+512)$. (a) By what factor does $\\mathrm{Var}(z^{[l]})$ change from one layer to the next? (b) By what factor is the standard deviation of $z^{[20]}$ smaller than that of $z^{[1]}$? (c) What weight standard deviation would keep it constant?", hint: "Use $\\mathrm{Var}(z^{[l]}) = n_{in}\\mathrm{Var}(W)\\,\\mathbb{E}[(a^{[l-1]})^2]$ and $\\mathbb{E}[\\mathrm{ReLU}(z)^2] = \\tfrac12\\mathrm{Var}(z)$. There are 19 transitions from layer 1 to layer 20.", solution: [
        { t: "steps", items: [
          { title: "(a) Per-layer factor", tex: String.raw`\mathrm{Var}(z^{[l]}) = 512\cdot\frac{1}{512}\cdot\tfrac12\,\mathrm{Var}(z^{[l-1]}) = \tfrac12\,\mathrm{Var}(z^{[l-1]})` },
          { title: "(b) Over 19 transitions", tex: String.raw`\frac{\mathrm{Var}(z^{[20]})}{\mathrm{Var}(z^{[1]})} = 2^{-19} = 1.907\times10^{-6},\qquad \frac{\mathrm{std}(z^{[20]})}{\mathrm{std}(z^{[1]})} = 2^{-9.5} = 1.381\times10^{-3}` },
          { title: "(c) He initialisation", tex: String.raw`\tfrac12 \cdot 512\cdot \mathrm{Var}(W) = 1 \;\Rightarrow\; \mathrm{Var}(W) = \tfrac{2}{512},\quad \mathrm{std}(W) = 0.0625` },
        ] },
        { t: "p", text: "Compare with the NumPy experiment (width 64, 30 layers, ReLU/Xavier): activation std fell from 0.578 at layer 1 to $1.68\\times10^{-5}$ at layer 30, a factor of $2.9\\times10^{-5}$, against the predicted $2^{-29/2} = 4.3\\times10^{-5}$." },
      ] },
      { t: "exercise", title: "Implement gradient clipping from scratch", prompt: "Write `clip_by_global_norm(grads, max_norm)` for a list of NumPy gradient arrays, returning the clipped list and the total norm before clipping, and `clip_by_value(grads, c)`. Check that your norm clipping matches `torch.nn.utils.clip_grad_norm_` exactly, and show on the vector $(3, 0.1)$ with threshold 1 that clipping by norm keeps the direction while clipping by value doesn't.", hint: "The global norm is the L2 norm of all gradient entries joined together. Scale by $\\min(1, c/(\\|g\\| + \\epsilon))$ — PyTorch uses $\\epsilon = 10^{-6}$.", solution: [
        { t: "code", lang: "python", title: "clipping_from_scratch.py", code: GP_CLIP, output: GP_CLIP_OUT },
        { t: "p", text: "Norm clipping rescales $(3, 0.1)$ to $(0.9994, 0.0333)$ — same direction (cosine 1.0), length 1 (the tiny $\\epsilon$ keeps it a hair below 1). Value clipping gives $(1, 0.1)$: the small component is untouched while the big one is cut, so the direction rotates (cosine 0.9978). In high dimensions with many clipped components the rotation can be much larger." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Unrolled BP2: $\\delta^{[1]} = \\big(\\prod_{l=1}^{L-1} D^{[l]}W^{[l+1]\\top}\\big)\\delta^{[L]}$ with $D^{[l]} = \\mathrm{diag}(f'(z^{[l]}))$ — a product of $L-1$ Jacobians.",
        "Products of many factors shrink or grow exponentially: $0.9^{30} = 0.042$, $1.1^{30} = 17.45$, $0.25^{10} = 9.5\\times10^{-7}$.",
        "Bound: $\\|\\delta^{[1]}\\| \\le \\prod\\|W^{[l]}\\|_2\\,\\|f'\\|_\\infty^{L-1}\\|\\delta^{[L]}\\|$; for sigmoid ($\\|f'\\|_\\infty = 1/4$), $\\|W\\|_2 < 4$ everywhere guarantees vanishing.",
        "Weight gradients are $\\delta^{[l]}a^{[l-1]\\top}$: both backward errors and forward activations must keep their scale. Check absolute sizes and $\\|\\nabla W\\|/\\|W\\|$, not only first/last ratios.",
        "Variance propagation: $\\mathrm{Var}(z^{[l]}) = n_{in}\\mathrm{Var}(W)\\mathbb{E}[a^2]$ ⇒ Xavier $\\mathrm{Var}(W) = 2/(n_{in}+n_{out})$ (tanh/linear), He $\\mathrm{Var}(W) = 2/n_{in}$ (ReLU).",
        "Measured (30 layers, width 64): sigmoid/Xavier first/last weight-gradient ratio $1.8\\times10^{-19}$; tanh/Xavier 1.9; ReLU/He 0.37; ReLU/N(0, 0.5²) gradients $\\sim10^{26}$.",
        "Residual blocks add $+I$ to the Jacobian, a direct gradient path; normalisation resets the scale every layer; clipping caps exploding updates; LSTM gating keeps a near-identity path through time.",
        "Diagnose with per-layer gradient norms (hooks), update-to-weight ratios, activation/gradient histograms, and the total norm returned by `clip_grad_norm_`.",
      ] },
    ],
    next: [
      { t: "p", text: "Each fix has its own lesson:" },
      { t: "list", items: [
        "[Weight Initialization: Xavier and He](/learn/m07/initialization) — full derivations, uniform vs normal, gains for other activations, and why zero initialisation fails.",
        "[Batch, Layer and Group Normalization](/learn/m07/normalization) — how normalising activations keeps the scale in check during training.",
        "[Residual and Skip Connections](/learn/m07/residual-connections) — the +I path in depth, the degradation problem, and how ResNets and Transformers use it.",
        "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — warmup and clipping in practice.",
        "[Backpropagation Through Time](/learn/m09/bptt) and [LSTM](/learn/m09/lstm) — vanishing gradients over time and the gated fix.",
        "[Numerical Gradient Checking](/learn/m04/gradient-checking) — to be sure a vanishing gradient comes from the network, not a bug in your backward pass.",
      ] },
    ],
  },
};

export default lesson;
