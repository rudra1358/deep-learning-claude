import type { Lesson } from "@/lib/types";

const MLP_NUMPY = String.raw`import numpy as np
from sklearn.datasets import make_circles

np.set_printoptions(precision=3, suppress=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MLP:
    """Fully-connected network: tanh hidden layers, one sigmoid output, binary cross-entropy.
    sizes = [n_inputs, n_hidden_1, ..., n_hidden_k, 1]."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        # Xavier/Glorot init: std sqrt(2 / (fan_in + fan_out)) keeps tanh units out of saturation
        self.W = [rng.normal(0, np.sqrt(2 / (i + o)), (o, i)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def forward(self, X):
        self.a = [X]                                   # a[0] = input, a[l] = output of layer l
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.a[-1] @ W.T + b
            self.a.append(sigmoid(z) if l == len(self.W) - 1 else np.tanh(z))
        return self.a[-1]

    def train(self, X, y, lr, epochs):
        """Full-batch gradient descent; backprop as derived in the backpropagation lesson."""
        for _ in range(epochs):
            p = self.forward(X)
            delta = (p - y) / len(y)                   # dL/dz at the output for sigmoid + mean BCE
            for l in reversed(range(len(self.W))):
                gW, gb = delta.T @ self.a[l], delta.sum(axis=0)
                if l > 0:                              # propagate BEFORE updating W[l]
                    delta = (delta @ self.W[l]) * (1 - self.a[l] ** 2)
                self.W[l] -= lr * gW
                self.b[l] -= lr * gb
        p = np.clip(self.forward(X), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def linear_probe_accuracy(F, y, lr=0.5, epochs=3000):
    """Best accuracy a single linear classifier (logistic regression) reaches on features F."""
    F = (F - F.mean(axis=0)) / F.std(axis=0)
    probe = MLP([F.shape[1], 1], seed=0)               # no hidden layer = logistic regression
    probe.train(F, y, lr, epochs)
    return np.mean((probe.forward(F) > 0.5) == y)


# ========== Part 1: XOR with a 2-2-1 network ==========
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
print(f"XOR, logistic regression on raw inputs: accuracy {linear_probe_accuracy(X, y):.2f}")
net = MLP([2, 2, 1], seed=0)
loss = net.train(X, y, lr=1.0, epochs=5000)
p = net.forward(X)
print(f"XOR, 2-2-1 network ({net.n_params()} parameters): loss {loss:.4f}, outputs {p.ravel()}")
H = net.a[1]                                           # the 2-D hidden representation
for x_i, h_i, y_i in zip(X, H, y.ravel()):
    print(f"  x = {x_i}  ->  h = {h_i}   class {int(y_i)}")
w, b = net.W[1][0], net.b[1][0]
print(f"  output neuron: z = {w[0]:+.2f} h1 {w[1]:+.2f} h2 {b:+.2f}  (a straight line in h-space)")

# How often does plain GD solve XOR? Depends on width (20 random initialisations each).
for hidden in (2, 4):
    solved = sum(np.all((MLP([2, hidden, 1], seed=s).train(X, y, 1.0, 3000) < 0.1)) for s in range(20))
    print(f"  width {hidden}: {solved}/20 seeds reach loss < 0.1 within 3000 steps")

# ========== Part 2: concentric circles, a 2-D bottleneck layer ==========
Xc, yc = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=0)
yc = yc.reshape(-1, 1).astype(float)
Xtr, ytr, Xte, yte = Xc[:300], yc[:300], Xc[300:], yc[300:]
net = MLP([2, 8, 2, 1], seed=0)                        # second hidden layer has only 2 units
loss = net.train(Xtr, ytr, lr=0.5, epochs=4000)
print(f"\ncircles, 2-8-2-1 network ({net.n_params()} parameters): train loss {loss:.4f}")
print(f"  train acc {np.mean((net.forward(Xtr) > 0.5) == ytr):.3f}   test acc {np.mean((net.forward(Xte) > 0.5) == yte):.3f}")
net.forward(Xte)
H2 = net.a[2]                                          # test points as seen by hidden layer 2
print(f"  linear-probe accuracy on raw x:          {linear_probe_accuracy(Xte, yte):.3f}")
print(f"  linear-probe accuracy on hidden layer 2: {linear_probe_accuracy(H2, yte):.3f}")
for cls, name in ((0, "outer ring"), (1, "inner ring")):
    idx = np.where(yte.ravel() == cls)[0][:3]
    for i in idx:
        print(f"  {name}: x = {Xte[i]}  r = {np.linalg.norm(Xte[i]):.2f}  ->  h2 = {H2[i]}")`;

const MLP_NUMPY_OUT = String.raw`XOR, logistic regression on raw inputs: accuracy 0.50
XOR, 2-2-1 network (9 parameters): loss 0.0006, outputs [0.001 1.    1.    0.001]
  x = [0. 0.]  ->  h = [-1.    -0.968]   class 0
  x = [0. 1.]  ->  h = [-0.925  0.976]   class 1
  x = [1. 0.]  ->  h = [-0.925  0.976]   class 1
  x = [1. 1.]  ->  h = [0.947 1.   ]   class 0
  output neuron: z = -8.03 h1 +7.96 h2 -7.44  (a straight line in h-space)
  width 2: 12/20 seeds reach loss < 0.1 within 3000 steps
  width 4: 20/20 seeds reach loss < 0.1 within 3000 steps

circles, 2-8-2-1 network (45 parameters): train loss 0.0006
  train acc 1.000   test acc 1.000
  linear-probe accuracy on raw x:          0.570
  linear-probe accuracy on hidden layer 2: 1.000
  outer ring: x = [-0.999  0.064]  r = 1.00  ->  h2 = [ 0.308 -0.79 ]
  outer ring: x = [ 0.717 -0.611]  r = 0.94  ->  h2 = [ 0.121 -0.769]
  outer ring: x = [-0.873  0.147]  r = 0.88  ->  h2 = [ 0.162 -0.607]
  inner ring: x = [0.446 0.15 ]  r = 0.47  ->  h2 = [-0.734  0.984]
  inner ring: x = [-0.238 -0.469]  r = 0.53  ->  h2 = [-0.646  0.982]
  inner ring: x = [ 0.392 -0.246]  r = 0.46  ->  h2 = [-0.69   0.969]`;

const HINGE_CODE = String.raw`import numpy as np

relu = lambda z: np.maximum(0.0, z)


def hinge_network(f, a, b, n_units):
    """One-hidden-layer ReLU network that linearly interpolates f at n_units + 1 equally spaced knots on [a, b].
    Hidden unit j computes relu(x - t_j); its output weight is the CHANGE in slope at knot t_j."""
    t = np.linspace(a, b, n_units + 1)                 # knots t_0 < t_1 < ... < t_n
    slopes = np.diff(f(t)) / np.diff(t)                # slope of each linear piece
    c = np.diff(slopes, prepend=0.0)                   # c_0 = first slope, c_j = slope change at t_j
    knots = t[:-1]
    return lambda x: f(a) + relu(x[:, None] - knots[None, :]) @ c


f = np.sin
x = np.linspace(0, 2 * np.pi, 20001)
for n in (4, 8, 16, 32, 64):
    g = hinge_network(f, 0, 2 * np.pi, n)
    h = 2 * np.pi / n
    print(f"{n:3d} hidden ReLUs: max |sin - net| = {np.max(np.abs(f(x) - g(x))):.5f}   bound h^2/8 = {h * h / 8:.5f}")`;

const HINGE_OUT = String.raw`  4 hidden ReLUs: max |sin - net| = 0.21051   bound h^2/8 = 0.30843
  8 hidden ReLUs: max |sin - net| = 0.07038   bound h^2/8 = 0.07711
 16 hidden ReLUs: max |sin - net| = 0.01885   bound h^2/8 = 0.01928
 32 hidden ReLUs: max |sin - net| = 0.00479   bound h^2/8 = 0.00482
 64 hidden ReLUs: max |sin - net| = 0.00120   bound h^2/8 = 0.00120`;

const DEPTH_CODE = String.raw`import numpy as np

relu = lambda z: np.maximum(0.0, z)


def tent(x):
    """Tent map on [0, 1] built from TWO ReLUs: rises 0->1 on [0, 0.5], falls 1->0 on [0.5, 1]."""
    return 2 * relu(x) - 4 * relu(x - 0.5)


def count_pieces(y, x):
    """Number of linear pieces = 1 + number of slope changes on a fine grid."""
    slope = np.diff(y) / np.diff(x)
    return 1 + int(np.sum(np.abs(np.diff(slope)) > 1e-6))


x = np.linspace(0, 1, 2 ** 14 + 1)                    # grid contains every breakpoint k / 2^m
y = x.copy()
for depth in range(1, 9):
    y = tent(y)                                        # one more layer of 2 ReLUs
    print(f"depth {depth}: {2 * depth:2d} ReLUs in total -> {count_pieces(y, x):4d} linear pieces"
          f"   (a 1-hidden-layer net needs >= {count_pieces(y, x) - 1} ReLUs)")`;

const DEPTH_OUT = String.raw`depth 1:  2 ReLUs in total ->    2 linear pieces   (a 1-hidden-layer net needs >= 1 ReLUs)
depth 2:  4 ReLUs in total ->    4 linear pieces   (a 1-hidden-layer net needs >= 3 ReLUs)
depth 3:  6 ReLUs in total ->    8 linear pieces   (a 1-hidden-layer net needs >= 7 ReLUs)
depth 4:  8 ReLUs in total ->   16 linear pieces   (a 1-hidden-layer net needs >= 15 ReLUs)
depth 5: 10 ReLUs in total ->   32 linear pieces   (a 1-hidden-layer net needs >= 31 ReLUs)
depth 6: 12 ReLUs in total ->   64 linear pieces   (a 1-hidden-layer net needs >= 63 ReLUs)
depth 7: 14 ReLUs in total ->  128 linear pieces   (a 1-hidden-layer net needs >= 127 ReLUs)
depth 8: 16 ReLUs in total ->  256 linear pieces   (a 1-hidden-layer net needs >= 255 ReLUs)`;

const MLP_TORCH = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
X_tr, y_tr, X_te, y_te = X[:200], y[:200], X[200:], y[200:]     # small training set, large test set


# ---- Style 1: nn.Sequential — a plain stack of layers ----
def make_sequential(width):
    return nn.Sequential(
        nn.Linear(2, width), nn.ReLU(),        # hidden layer 1
        nn.Linear(width, width), nn.ReLU(),    # hidden layer 2
        nn.Linear(width, 1),                   # output layer: one logit (binary task)
    )


# ---- Style 2: nn.Module — same network, but forward() can be anything you write ----
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(2, width), nn.Linear(width, width)])
        self.out = nn.Linear(width, 1)

    def forward(self, x, return_hidden=False):
        h = x
        for layer in self.hidden:
            h = torch.relu(layer(h))
        return (self.out(h), h) if return_hidden else self.out(h)


def train(model, epochs=2000, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()           # sigmoid output + BCE, fused for stability
    for _ in range(epochs):
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = lambda X_, y_: ((model(X_) > 0).float() == y_).float().mean().item()
        return loss.item(), acc(X_tr, y_tr), acc(X_te, y_te)


torch.manual_seed(0)
a = make_sequential(16)
torch.manual_seed(0)
b = MLP(16)
print("same parameter count:", sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
print("Sequential: loss %.4f  train acc %.3f  test acc %.3f" % train(a))
print("Module    : loss %.4f  train acc %.3f  test acc %.3f" % train(b))

# ---- Capacity: the same architecture at different widths ----
print("\nwidth  params  train loss  train acc  test acc")
for width in (1, 2, 4, 16, 128):
    torch.manual_seed(0)
    model = make_sequential(width)
    n = sum(p.numel() for p in model.parameters())
    loss, tr, te = train(model)
    print(f"{width:5d}  {n:6d}  {loss:10.4f}  {tr:9.3f}  {te:8.3f}")`;

const MLP_TORCH_OUT = String.raw`same parameter count: 337 337
Sequential: loss 0.1032  train acc 0.955  test acc 0.933
Module    : loss 0.1032  train acc 0.955  test acc 0.933

width  params  train loss  train acc  test acc
    1       7      0.6919      0.525     0.488
    2      15      0.2620      0.890     0.875
    4      37      0.2618      0.890     0.875
   16     337      0.1032      0.955     0.933
  128   17025      0.0131      1.000     0.913`;

const COLLAPSE_CODE = String.raw`import numpy as np
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=0)
y = y.reshape(-1, 1).astype(float)


def hidden_and_logit(X, W, b, act):
    """Run the hidden layers with activation act; return all activations and the output logit."""
    a = [X]
    for l in range(len(W) - 1):
        a.append(act(a[-1] @ W[l].T + b[l]))
    return a, a[-1] @ W[-1].T + b[-1]


def train_mlp(sizes, act, act_grad, lr=0.5, epochs=4000, seed=0):
    """The lesson's MLP (sigmoid output, BCE, full-batch GD) with a pluggable hidden activation."""
    rng = np.random.default_rng(seed)
    W = [rng.normal(0, np.sqrt(2 / (i + o)), (o, i)) for i, o in zip(sizes[:-1], sizes[1:])]
    b = [np.zeros(o) for o in sizes[1:]]
    for _ in range(epochs):
        a, z = hidden_and_logit(X, W, b, act)
        a.append(1 / (1 + np.exp(-z)))
        delta = (a[-1] - y) / len(y)
        for l in reversed(range(len(W))):
            gW, gb = delta.T @ a[l], delta.sum(axis=0)
            if l > 0:
                delta = (delta @ W[l]) * act_grad(a[l])
            W[l] -= lr * gW
            b[l] -= lr * gb
    _, z = hidden_and_logit(X, W, b, act)
    return np.mean((z > 0) == y), W, b


acc, _, _ = train_mlp([2, 8, 2, 1], np.tanh, lambda a: 1 - a ** 2)
print(f"tanh hidden layers:     train accuracy {acc:.3f}")
identity = lambda z: z
acc, W, b = train_mlp([2, 8, 2, 1], identity, np.ones_like)
print(f"identity hidden layers: train accuracy {acc:.3f}")

# With identity activations the network is ONE affine map: z = w.x + c. Recover w and c from 3 probes.
probes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.3, -0.7]])
_, z = hidden_and_logit(probes, W, b, identity)
z = z.ravel()
c, w = z[0], np.array([z[1] - z[0], z[2] - z[0]])
print(f"collapsed to z = {w[0]:.4f} x1 + {w[1]:.4f} x2 + {c:.4f}")
print(f"check at (0.3, -0.7): network {z[3]:.6f}   affine formula {w @ probes[3] + c:.6f}")`;

const COLLAPSE_OUT = String.raw`tanh hidden layers:     train accuracy 1.000
identity hidden layers: train accuracy 0.495
collapsed to z = 0.0157 x1 + 0.0025 x2 + 0.0001
check at (0.3, -0.7): network 0.003092   affine formula 0.003092`;

const MLP_KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X_tr, y_tr, X_te, y_te = X[:200], y[:200], X[200:], y[200:]

# Sequential API: a plain stack of layers
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),   # hidden layer 1
    tf.keras.layers.Dense(16, activation="relu"),   # hidden layer 2
    tf.keras.layers.Dense(1),                       # one logit
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])
model.summary()                                     # prints the parameter count per layer (total 337)
model.fit(X_tr, y_tr, batch_size=len(X_tr), epochs=2000, verbose=0)
print(model.evaluate(X_te, y_te, verbose=0))         # [test loss, test accuracy]`;

const lesson: Lesson = {
  id: "m03/multilayer-networks",
  title: "Multi-Layer Networks and Model Capacity",
  summary:
    "Why one layer is not enough, what stacking layers buys (learned features that make classes linearly separable), the universal approximation theorem stated precisely with a constructive ReLU proof, why depth can be exponentially more efficient than width, counting parameters, capacity, and how to design the output layer for each task.",
  minutes: 70,
  level: "beginner",
  prerequisites: ["m03/linear-logistic-regression", "m03/artificial-neuron", "m03/activation-functions"],
  next: ["m03/training-basics", "m04/forward-propagation", "m04/backpropagation"],
  terms: [
    "multilayer-perceptron",
    "hidden-layer",
    "width",
    "depth",
    "representation-learning",
    "universal-approximation-theorem",
    "linear-region",
    "parameter",
    "hyperparameter",
    "model-capacity",
    "linear-probe",
  ],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "A single neuron — [logistic regression](/learn/m03/linear-logistic-regression), or the [perceptron](/learn/m03/artificial-neuron) — can only draw one straight boundary through its input space. That is enough for a surprising number of problems, but not for XOR, not for one class surrounded by another, and certainly not for recognising a face in raw pixels. The fix, known since the 1960s but only trainable at scale after backpropagation was popularised in 1986, is to put **layers of neurons between the input and the output**. A network with one or more such **hidden layers** is called a **multi-layer perceptron (MLP)**, or a fully connected feed-forward network.",
      },
      {
        t: "p",
        text: "Hidden layers solve the problem in a specific way: each layer computes a new **representation** of the input, and the layers are trained so that, by the last hidden layer, the problem has become one a single neuron *can* solve. In the previous lesson you had to invent features by hand ($x^2$, or $x_1^2 + x_2^2$ for circles); an MLP learns them from data. This is the sense in which deep learning is **representation learning**.",
      },
      {
        t: "p",
        text: "How powerful is this? The **universal approximation theorem** says that a network with a single hidden layer can approximate any continuous function on a bounded region as closely as you like, given enough hidden units. This lesson states that theorem precisely, proves a version of it with ReLUs in a few lines, and is equally careful about what it does *not* say: nothing about how many units are needed, whether training will find the weights, or whether the result will generalise. Those gaps lead to the practical questions of **depth versus width**, **parameters versus hyperparameters**, and **model capacity** — the vocabulary used to design every network in the rest of the course.",
      },
      {
        t: "callout",
        kind: "key",
        title: "In one sentence",
        text: String.raw`An MLP alternates affine maps and non-linearities, $\mathbf{h}^{[l]} = f(W^{[l]}\mathbf{h}^{[l-1]} + \mathbf{b}^{[l]})$; the hidden layers learn features that make the task linearly solvable for the output layer, and with enough hidden units such networks can approximate any continuous function on a compact set — which says nothing about learnability or generalisation.`,
      },
    ],

    // --------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Linear and Logistic Regression as Neural Networks](/learn/m03/linear-logistic-regression) — the output layer of an MLP *is* one of these models, applied to learned features. You need the decision boundary $\\mathbf{w}^\\top\\mathbf{x} + b = 0$ and the pairings identity + MSE, sigmoid + BCE, softmax + cross-entropy.",
          "[The Artificial Neuron and the Perceptron](/learn/m03/artificial-neuron) — linear separability and the proof that one neuron cannot compute XOR; that lesson also builds an XOR network by hand, which we will train here instead.",
          "[Activation Functions](/learn/m03/activation-functions) — why a non-linearity is required between layers (otherwise layers collapse into one linear map), and the definitions of ReLU, tanh and sigmoid.",
          "[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — a layer is a matrix–vector product; shapes $(n_{out}\\times n_{in})$ will be used to count parameters.",
          "Optional: [Backpropagation](/learn/m04/backpropagation) explains how the code below computes gradients; here we use it as a tool.",
        ],
      },
    ],

    // ------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Folding a sheet of paper",
        text: "Draw blue dots and red dots on a sheet of paper so that no straight cut separates them — say, red in two opposite corners and blue in the other two (XOR). Now fold the paper along the diagonal. The two red corners land on top of each other, and so do the two blue ones; a single straight cut now separates red from blue. A hidden layer does exactly this: it bends and folds the input space so that points of the same class are brought together, and the output neuron makes the final straight cut. A deeper network folds the already-folded sheet again, and each fold doubles the number of regions a single cut can carve out.",
      },
      {
        t: "p",
        text: "Concretely, each hidden neuron is a small logistic-regression-like detector: it computes a weighted sum of its inputs and fires (strongly, weakly or not at all) depending on which side of its own hyperplane the input lies. A hidden layer with $m$ neurons therefore re-describes every input as $m$ numbers — \"how much does this input look like pattern 1, pattern 2, …\". The output neuron then works only with that description. Training adjusts the detectors and the output neuron together, so the detectors become whatever patterns make the final decision easiest.",
      },
      {
        t: "p",
        text: "Stacking layers repeats the trick: the second hidden layer detects patterns *of patterns*. In image models this produces the familiar progression from edges to textures to object parts. Two numbers describe the shape of such a network: its **width** (neurons per layer) and its **depth** (number of layers). Wider layers can describe each input in more ways; deeper stacks can compose descriptions. Both add **capacity** — the variety of functions the network can represent — and capacity must be matched to the amount of data, which is the subject of [Training Basics](/learn/m03/training-basics).",
      },
    ],

    // ------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Anatomy of an MLP" },
      {
        t: "list",
        items: [
          "**Input layer** — not really a layer of neurons: it holds the $d$ feature values $\\mathbf{x}$. It has no parameters.",
          "**Hidden layers** $l = 1, \\dots, L-1$ — each takes the previous layer's output vector $\\mathbf{h}^{[l-1]}$ (with $\\mathbf{h}^{[0]} = \\mathbf{x}$), computes the pre-activations $\\mathbf{z}^{[l]} = W^{[l]}\\mathbf{h}^{[l-1]} + \\mathbf{b}^{[l]}$ and applies a non-linear activation element-wise: $\\mathbf{h}^{[l]} = f(\\mathbf{z}^{[l]})$. \"Hidden\" means the training data says nothing about what these values should be; the network decides.",
          "**Output layer** $L$ — an affine map followed by the activation that matches the task (identity, sigmoid, softmax). It is exactly a linear/logistic/softmax regression whose inputs are $\\mathbf{h}^{[L-1]}$.",
          "**Fully connected** — every neuron of layer $l$ receives every output of layer $l-1$, so $W^{[l]}$ has shape $n_l\\times n_{l-1}$ and layer $l$ has $n_l n_{l-1} + n_l$ parameters.",
          "**Feed-forward** — information flows strictly from input to output, with no cycles (contrast with [recurrent networks](/learn/m09/rnn-basics)).",
        ],
      },
      { t: "h", text: "What one forward pass does" },
      {
        t: "steps",
        items: [
          { title: "Start with the input", text: String.raw`$\mathbf{h}^{[0]} = \mathbf{x} \in \mathbb{R}^{d}$ (for a batch, a matrix with one example per row).` },
          { title: "Affine map", text: String.raw`$\mathbf{z}^{[1]} = W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$: each of the $n_1$ hidden neurons measures the signed distance-like score of $\mathbf{x}$ from its own hyperplane.` },
          { title: "Non-linearity", text: String.raw`$\mathbf{h}^{[1]} = f(\mathbf{z}^{[1]})$: ReLU zeroes the negative side of each hyperplane, tanh/sigmoid squash. This is the step that bends the space; without it, layer 2 would just compose two affine maps into one.` },
          { title: "Repeat", text: "Apply steps 2–3 for every hidden layer. Each layer's output is the next layer's input; the representation dimension changes from $n_{l-1}$ to $n_l$." },
          { title: "Output", text: String.raw`$\hat{\mathbf{y}} = g(W^{[L]}\mathbf{h}^{[L-1]} + \mathbf{b}^{[L]})$ with $g$ chosen for the task (table below). The loss compares $\hat{\mathbf{y}}$ with the target.` },
          { title: "Learn", text: "Backpropagation computes the gradient of the loss with respect to every $W^{[l]}$ and $\\mathbf{b}^{[l]}$; an optimizer updates them. Every layer is trained jointly, for the single purpose of lowering the final loss — this is what makes the hidden features useful rather than arbitrary." },
        ],
      },
      { t: "h", text: "Designing the output layer" },
      {
        t: "table",
        head: ["Task", "Output units", "Output activation", "Loss", "PyTorch idiom"],
        rows: [
          ["Regression (one or more real targets)", "one per target", "identity", "MSE (or MAE / Huber)", "`nn.Linear(h, k)` + `nn.MSELoss()`"],
          ["Regression, positive target", "one per target", "softplus or exp", "MSE, or a likelihood (e.g. Poisson)", "`F.softplus(...)`"],
          ["Binary classification", "1", "sigmoid", "binary cross-entropy", "`nn.Linear(h, 1)` + `nn.BCEWithLogitsLoss()`"],
          ["Multiclass, exactly one label", "K", "softmax", "categorical cross-entropy", "`nn.Linear(h, K)` + `nn.CrossEntropyLoss()`"],
          ["Multi-label (any subset of K labels)", "K", "sigmoid per unit (independent)", "BCE summed/averaged over labels", "`nn.Linear(h, K)` + `nn.BCEWithLogitsLoss()`"],
        ],
      },
      {
        t: "p",
        text: "Two rules follow from the previous lesson. First, the output activation and loss come in matched pairs whose gradient at the logits is *prediction − target*; frameworks fuse them (`BCEWithLogitsLoss`, `CrossEntropyLoss`), so the model should output **raw logits** and the loss applies the sigmoid/softmax. Second, hidden layers almost never use the output-layer activations: ReLU and its relatives are the default for hidden units in MLPs ([why](/learn/m03/activation-functions)).",
      },
      { t: "h", text: "Parameters versus hyperparameters" },
      {
        t: "table",
        head: ["", "Parameters", "Hyperparameters"],
        rows: [
          ["What", "Weights $W^{[l]}$ and biases $\\mathbf{b}^{[l]}$", "Everything that defines the model and training but is not learned by gradient descent"],
          ["Examples", "the 337 numbers in a 2-16-16-1 MLP", "number of layers, width of each layer, activation, learning rate, batch size, epochs, regularisation strength, initialisation scheme"],
          ["Set by", "the optimizer, from training data", "you (or a search procedure), using **validation** data"],
          ["Count", "can be billions", "usually a handful to a few dozen"],
        ],
      },
      {
        t: "p",
        text: "Width and depth are architectural hyperparameters; together with the activation they fix the **hypothesis class** — the set of functions the optimizer is allowed to search. The training-loop hyperparameters (learning rate, batch size, epochs) are treated in [Training Basics](/learn/m03/training-basics).",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "The MLP as a composition" },
      {
        t: "math",
        tex: String.raw`F(\mathbf{x}) = g\Big(W^{[L]} f\big(W^{[L-1]} \cdots f(W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) \cdots + \mathbf{b}^{[L-1]}\big) + \mathbf{b}^{[L]}\Big)`,
        caption: "$W^{[l]}\\in\\mathbb{R}^{n_l\\times n_{l-1}}$, $\\mathbf{b}^{[l]}\\in\\mathbb{R}^{n_l}$, $n_0 = d$ inputs, $f$ the hidden activation, $g$ the output activation.",
      },
      {
        t: "p",
        text: String.raw`If $f$ were the identity, the composition would collapse: $W^{[2]}(W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = (W^{[2]}W^{[1]})\mathbf{x} + (W^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]})$ — one affine map, no more expressive than logistic regression however many layers are stacked. The coding exercise demonstrates this numerically.`,
      },
      { t: "h", text: "Counting parameters" },
      {
        t: "math",
        tex: String.raw`P = \sum_{l=1}^{L}\big(n_{l-1}\,n_l + n_l\big) = \sum_{l=1}^{L}(n_{l-1} + 1)\,n_l`,
        caption: "Weights plus biases, layer by layer. The +1 is the bias of each neuron.",
      },
      {
        t: "steps",
        items: [
          { title: "XOR network 2-2-1", tex: String.raw`(2+1)\cdot2 + (2+1)\cdot1 = 6 + 3 = 9` },
          { title: "Circles network 2-8-2-1", tex: String.raw`(2+1)\cdot8 + (8+1)\cdot2 + (2+1)\cdot1 = 24 + 18 + 3 = 45` },
          { title: "MNIST-style classifier 784-256-128-10", tex: String.raw`785\cdot256 + 257\cdot128 + 129\cdot10 = 200{,}960 + 32{,}896 + 1{,}290 = 235{,}146` },
        ],
      },
      {
        t: "p",
        text: "In the last example 85% of the parameters sit in the first layer, because the input is wide (784 pixels). This is typical of fully connected layers on raw high-dimensional inputs, and is one motivation for weight sharing in [convolutional networks](/learn/m08/convolution).",
      },
      { t: "h", text: "What hidden layers represent: the space-folding view" },
      {
        t: "p",
        text: String.raw`The output layer is a linear classifier on $\mathbf{h}^{[L-1]}$. So the network solves a non-linear problem exactly when the last hidden layer maps the data to a space where the classes are **linearly separable**. You can test this directly with a **linear probe**: freeze the network, compute $\mathbf{h}$ for every example, and train a fresh logistic regression on $\mathbf{h}$.`,
      },
      {
        t: "p",
        text: String.raw`The scratch code does this for XOR with a 2-2-1 tanh network. After training, the hidden layer maps the four inputs to $(0,0)\mapsto(-1.000, -0.968)$, $(0,1)\mapsto(-0.925, 0.976)$, $(1,0)\mapsto(-0.925, 0.976)$, $(1,1)\mapsto(0.947, 1.000)$. The two class-1 inputs land on **the same point** — the input square has been folded along its anti-diagonal — and the output neuron $z = -8.03h_1 + 7.96h_2 - 7.44$ separates the folded points with a single line. For concentric circles with a 2-8-2-1 network, a linear probe on the raw coordinates reaches only 57.0% test accuracy, while a linear probe on the 2-D second hidden layer reaches 100%: the inner ring is mapped near $(-0.7, 0.98)$ and the outer ring near $(0.2, -0.7)$.`,
      },
      {
        t: "p",
        text: String.raw`For ReLU networks the folding picture can be made exact. Each ReLU unit splits its input space with a hyperplane into an "off" half (output 0) and an "on" half (output linear). Within any region where the on/off pattern of all units is fixed, the whole network is a single affine function; the network is therefore **piecewise linear**, and its number of linear regions measures how finely it can bend. Montúfar, Pascanu, Cho and Bengio (2014, "On the number of linear regions of deep neural networks") used this view to show that deep ReLU networks can have exponentially more linear regions than shallow ones with the same number of units, because each layer can fold the space onto itself and every later fold is replicated in every image of the earlier ones.`,
      },
      { t: "h", text: "The universal approximation theorem, stated precisely" },
      {
        t: "p",
        text: String.raw`Consider one hidden layer of $N$ units with activation $f$ and a linear output: $F_N(\mathbf{x}) = \sum_{j=1}^{N} v_j\, f(\mathbf{a}_j^\top\mathbf{x} + c_j)$, with $\mathbf{a}_j\in\mathbb{R}^d$ and $v_j, c_j\in\mathbb{R}$.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "Theorem (Cybenko 1989; Hornik 1991; Leshno et al. 1993)",
        text: String.raw`Let $K\subset\mathbb{R}^d$ be compact (closed and bounded) and $g: K\to\mathbb{R}$ continuous. For every $\varepsilon > 0$ there exist $N$ and parameters $\{\mathbf{a}_j, c_j, v_j\}$ such that $\sup_{\mathbf{x}\in K}|F_N(\mathbf{x}) - g(\mathbf{x})| < \varepsilon$. **Cybenko** ("Approximation by superpositions of a sigmoidal function", 1989) proved this for continuous sigmoidal $f$ (tending to 0 at $-\infty$ and 1 at $+\infty$) on the unit cube. **Hornik** ("Approximation capabilities of multilayer feedforward networks", 1991) showed that any bounded, non-constant $f$ suffices, and that the same networks are dense in $L^p(\mu)$ for finite measures $\mu$. **Leshno, Lin, Pinkus and Schocken** (1993) showed that for continuous activations — and more generally locally bounded, piecewise continuous ones — the property holds **if and only if $f$ is not a polynomial**, which covers ReLU. Vector-valued outputs follow by applying the result to each output coordinate.`,
      },
      {
        t: "p",
        text: "Read the quantifiers carefully. **What the theorem promises:** for each continuous target on a bounded domain and each tolerance, *some* finite one-hidden-layer network is uniformly that close. So limited expressiveness is never the fundamental obstacle for a sufficiently wide MLP.",
      },
      {
        t: "list",
        items: [
          "**Not how many units.** $N$ depends on $g$ and $\\varepsilon$ and may be astronomically large. For generic smooth functions the number of units needed to reach accuracy $\\varepsilon$ can grow exponentially with the input dimension $d$; rates that avoid this hold only for restricted function classes (e.g. Barron, 1993, for functions with a finite first moment of the Fourier transform).",
          "**Not that training finds the weights.** It is an existence result. Gradient descent on a finite network may stall in poor regions — even XOR with 2 hidden units fails from 8 of 20 random initialisations in the code below.",
          "**Not generalisation.** It concerns approximating a *known* function everywhere. A learner only sees finitely many noisy samples; a network wide enough to approximate anything can also fit the noise. Which function is learned between the samples depends on the data, the architecture, the optimizer and regularisation.",
          "**Not outside $K$.** Nothing is promised beyond the compact set; MLPs typically extrapolate poorly (a ReLU network becomes linear far from the data).",
          "**Not discontinuous targets in the uniform sense.** A step function cannot be uniformly approximated by continuous functions; only in average-error ($L^p$) senses. The approximator simulator's step target shows the error that remains at the jump.",
          "**Not that one hidden layer is a good design.** Depth changes how many units are needed (below).",
        ],
      },
      { t: "h", text: "A constructive proof for ReLU in one dimension" },
      {
        t: "p",
        text: String.raw`Take a continuous $g$ on $[a, b]$ and knots $a = t_0 < t_1 < \dots < t_N = b$. The piecewise-linear interpolant $\ell$ that joins the points $(t_k, g(t_k))$ has slope $s_k = \frac{g(t_{k+1}) - g(t_k)}{t_{k+1} - t_k}$ on piece $k$. A single hinge $\mathrm{ReLU}(x - t_k)$ is zero to the left of $t_k$ and has slope 1 to its right, so a weighted sum of hinges changes slope by its weight at each knot. Hence`,
      },
      {
        t: "math",
        tex: String.raw`\ell(x) = g(t_0) + \sum_{k=0}^{N-1} c_k\,\mathrm{ReLU}(x - t_k),\qquad c_0 = s_0,\quad c_k = s_k - s_{k-1}\ (k \ge 1)`,
        caption: "Exactly a one-hidden-layer ReLU network: hidden weights 1, hidden biases $-t_k$, output weights $c_k$, output bias $g(t_0)$.",
      },
      {
        t: "p",
        text: String.raw`Because $g$ is continuous on a closed interval it is uniformly continuous, so for fine enough knots $g$ varies by less than $\varepsilon$ on each piece, and so does $\ell$; therefore $|\ell - g| < 2\varepsilon$ everywhere. That proves the 1-D theorem for ReLU. For twice-differentiable $g$ with equally spaced knots of width $h$, the classical interpolation bound gives the rate $\max|\ell - g| \le \frac{h^2}{8}\max|g''|$, so the error falls by 4 each time the number of units doubles. In $d$ dimensions the same idea works with ridge functions, but covering a $d$-dimensional box with a grid of width $h$ needs on the order of $h^{-d}$ pieces — the curse of dimensionality behind the first "not" above.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Approximate $g(x) = x^2$ on $[0,1]$ with 2 hidden units, knots $0, 0.5, 1$. Slopes: $s_0 = (0.25 - 0)/0.5 = 0.5$ and $s_1 = (1 - 0.25)/0.5 = 1.5$, so $c_0 = 0.5$ and $c_1 = 1.0$:`,
      },
      {
        t: "math",
        tex: String.raw`F(x) = 0.5\,\mathrm{ReLU}(x) + 1.0\,\mathrm{ReLU}(x - 0.5),\qquad W^{[1]} = \begin{pmatrix}1\\1\end{pmatrix},\ \mathbf{b}^{[1]} = \begin{pmatrix}0\\-0.5\end{pmatrix},\ W^{[2]} = (0.5\ \ 1.0),\ b^{[2]} = 0`,
      },
      {
        t: "p",
        text: String.raw`Check: $F(0.25) = 0.125$ (true $0.0625$), $F(0.5) = 0.25$ (exact), $F(0.75) = 0.375 + 0.25 = 0.625$ (true $0.5625$), $F(1) = 0.5 + 0.5 = 1$ (exact). The maximum error $0.0625$ occurs at the midpoints and equals the bound $\frac{h^2}{8}\cdot\max|g''| = \frac{0.25}{8}\cdot 2$. With 4 units the error is $0.015625$; with 8 units $0.00390625$ — each doubling divides it by 4. The same construction for $\sin$ on $[0, 2\pi]$:`,
      },
      { t: "code", lang: "python", title: "hinge_construction.py", code: HINGE_CODE, output: HINGE_OUT },
      { t: "h", text: "Depth efficiency: why not just one wide layer?" },
      {
        t: "p",
        text: String.raw`A one-hidden-layer ReLU network on a 1-D input with $N$ units has at most $N$ kinks, hence at most $N + 1$ linear pieces. Now compose the tent map $T(x) = 2\,\mathrm{ReLU}(x) - 4\,\mathrm{ReLU}(x - 0.5)$ (which rises from 0 to 1 on $[0, 0.5]$ and falls back on $[0.5, 1]$) with itself $k$ times. Each composition folds $[0,1]$ in half and maps both halves onto $[0, 1]$, doubling the number of pieces: $T^{\circ k}$ is a "sawtooth" with $2^k$ linear pieces, computed by a network of depth $k$ with only $2k$ ReLUs. Any one-hidden-layer network that represents it exactly needs at least $2^k - 1$ units:`,
      },
      { t: "code", lang: "python", title: "depth_vs_width.py", code: DEPTH_CODE, output: DEPTH_OUT },
      {
        t: "p",
        text: "Exact representation is a strong requirement, but the gap survives approximation. Telgarsky (2016, \"Benefits of depth in neural networks\") built on this sawtooth to show that for every $k$ there are functions computed by ReLU networks with $\\Theta(k^3)$ layers and $\\Theta(1)$ units per layer that cannot be approximated to within a fixed constant (in average error) by networks with $O(k)$ layers unless they have $\\Omega(2^k)$ units. Depth buys compositional, repeated structure cheaply; width alone pays for each repetition separately. This is a statement about specific functions, not a proof that deeper is always better — but it matches the empirical pattern that moderately deep networks use parameters more efficiently than very wide shallow ones.",
      },
      { t: "h", text: "Capacity" },
      {
        t: "p",
        text: String.raw`Informally, a model's **capacity** is the richness of the set of functions it can represent. For MLPs it grows with width, depth and the number of parameters, but parameter count is only a proxy: a 2-2-1 network and a 1-hidden-layer network with 8 units can both fit XOR, and regularisation reduces the capacity that is effectively used. Formal measures (VC dimension, Rademacher complexity) exist but are loose for large networks. What matters in practice is the relation between capacity and data:`,
      },
      {
        t: "list",
        items: [
          "**Too little capacity → underfitting.** Training and test error are both high. In the PyTorch width sweep below, width 1 (7 parameters) stays at chance on two moons (train 52.5%, test 48.8%).",
          "**Enough capacity → good fit.** Width 16 (337 parameters): train 95.5%, test 93.3%.",
          "**More capacity than the data supports → overfitting.** Width 128 (17,025 parameters, trained on 200 points) reaches 100% training accuracy but test accuracy *drops* to 91.3%: it has started to fit noise.",
        ],
      },
      {
        t: "p",
        text: "The remedies — more data, regularisation, early stopping, choosing the size on validation data — and the bias–variance view are developed in [Training Basics](/learn/m03/training-basics) and [Regularization](/learn/m07/regularization). One modern caveat: very large networks trained to zero training error sometimes generalise well anyway (the test error can fall again past the interpolation point, a phenomenon called *double descent*), so \"more parameters than data points\" does not by itself imply overfitting.",
      },
    ],

    // ---------------------------------------------------------------- visual
    visual: [
      { t: "h", text: "1. Universal approximation, live" },
      {
        t: "p",
        text: "The first simulator trains a one-hidden-layer ReLU network $\\hat y(x) = \\sum_j v_j\\max(0, a_jx + c_j) + d$ in your browser. The gold curve is the network, the dashed curve the target, the thin lines each hidden unit's hinge $v_j\\max(0, a_jx + c_j)$, and the gold ticks on the x-axis are the kinks. The right panel tracks the training MSE and the MSE against the true function on a dense grid.",
      },
      { t: "sim", id: "approximator", caption: "Each Step runs 20 full-batch iterations of Adam (or gradient descent) on the real network; every curve is computed from the current weights." },
      {
        t: "list",
        items: [
          "**More units, better fit.** With the *Bumps* target and dense samples, train with N = 2, 4, 12 and 50. Watch the error floor fall as the number of linear pieces grows — the universal approximation theorem in action. Turn on the hinges and see how a few units with large $|v_j|$ carve each bump.",
          "**|x| is exactly two units.** Select |x| with N = 2 and press Play: $|x| - 0.5 = \\mathrm{ReLU}(x) + \\mathrm{ReLU}(-x) - 0.5$ and the error can go to essentially zero. Press *New init* a few times — some initialisations never find it (both kinks drift to the same side), a small-scale example of the theorem not promising trainability.",
          "**The step.** A ReLU network is continuous; the step is not. However many units you add, an error spike remains at the jump — uniform approximation of a discontinuous function is impossible.",
          "**Sparse samples.** Switch to *Sparse (10 points)* with N = 50. The training MSE falls to almost zero, but the MSE against the target stays large: between the points nothing constrains the curve. With *Sparse + noise* the network threads through the noise. Approximation power is not generalisation.",
          "**Optimisers.** Switch to GD: progress is slower and more sensitive to the learning rate, because the gradient's scale differs wildly between parameters ([Optimizers](/learn/m06/momentum-adaptive) explains why Adam helps).",
        ],
      },
      { t: "h", text: "2. Build and count a network" },
      {
        t: "p",
        text: "The second simulator lets you design a small MLP — add or remove neurons and layers, pick each layer's activation — and step through its forward and backward passes on one example.",
      },
      { t: "sim", id: "network", caption: "Real forward propagation, backpropagation and gradient descent on the network you design." },
      {
        t: "list",
        items: [
          "Build a 2-3-1 network and count its connections and biases; check against $P = \\sum_l (n_{l-1}+1)n_l = 9 + 4 = 13$. Add a hidden layer of 2 and recount.",
          "Set every activation to *linear* and train with *Train ×25*: the output is a linear function of the inputs whatever the depth. Switch the hidden layers to ReLU or tanh to restore non-linearity.",
          "Watch the hidden values during the forward pass: they are the network's internal representation of your input — the numbers a linear probe would read.",
        ],
      },
    ],

    // --------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "A compact, general MLP in NumPy (tanh hidden layers, sigmoid output, binary cross-entropy, full-batch gradient descent — the backward pass is the one derived in [Backpropagation](/learn/m04/backpropagation)). Part 1 trains a 2-2-1 network on XOR, prints the hidden representation of each input, and measures how often random initialisations succeed at widths 2 and 4. Part 2 trains a 2-8-2-1 network on concentric circles and uses **linear probes** to show that the 2-D second hidden layer has made the classes linearly separable. Runs in a few seconds.",
      },
      { t: "code", lang: "python", title: "mlp_from_scratch.py", code: MLP_NUMPY, output: MLP_NUMPY_OUT },
      { t: "p", text: "Reading the output:" },
      {
        t: "list",
        items: [
          "Logistic regression on raw XOR inputs is stuck at 50% — the impossibility proof from [the perceptron lesson](/learn/m03/artificial-neuron), observed.",
          "The trained 2-2-1 network outputs (0.001, 1.000, 1.000, 0.001). Its hidden layer maps $(0,1)$ and $(1,0)$ to the identical point $(-0.925, 0.976)$: the fold. In $h$-space the three distinct points are separated by the output neuron's line.",
          "**Width helps optimisation, not just expressiveness.** Two hidden units *can* represent XOR, yet only 12 of 20 initialisations reach loss < 0.1 in 3000 steps; with 4 units all 20 do. Extra units give gradient descent more routes to a solution.",
          "**Circles:** 45 parameters, 100% train and test accuracy. The linear probe on the raw coordinates scores 57.0% (barely better than guessing), on the second hidden layer 100%. The printed rows show the mechanism: inner-ring points, at radius ≈ 0.5 in every direction, are all mapped close to $(-0.7, 0.98)$; outer-ring points to around $(0.1$–$0.3, -0.6$–$-0.8)$. The network has learned a feature that behaves like the radius, without being told.",
          "`linear_probe_accuracy` is the same `MLP` class with no hidden layer: logistic regression is literally the depth-1 special case.",
        ],
      },
    ],

    // ------------------------------------------------------------- framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "PyTorch offers two ways to define an MLP. `nn.Sequential` chains layers and is ideal when the forward pass is a plain stack. Subclassing `nn.Module` lets `forward` contain arbitrary Python — here it optionally returns the last hidden representation (for a linear probe). Seeded identically, both versions create the same parameters in the same order and train to identical results. The second part trains the same architecture at widths 1 to 128 to show capacity at work.",
              },
              { t: "code", lang: "python", title: "mlp_pytorch.py", code: MLP_TORCH, output: MLP_TORCH_OUT },
              {
                t: "list",
                items: [
                  "`sum(p.numel() for p in model.parameters())` is the standard parameter count: $(2+1)16 + (16+1)16 + (16+1)1 = 48 + 272 + 17 = 337$.",
                  "Width 1 never learns: a single ReLU unit can only produce a hinge, and here it ends up inactive on most of the data. Widths 2 and 4 underfit equally (loss ≈ 0.262); 16 fits well; 128 fits the 200 training points perfectly but generalises *worse* than 16 — overfitting on a small dataset.",
                  "Adam with learning rate 0.01 is used because it is robust across widths; [Optimizers](/learn/m06/momentum-adaptive) explains the choice.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "The Keras `Sequential` API mirrors `nn.Sequential`; `Dense(n, activation=...)` is a fully connected layer with its activation. `model.summary()` prints the per-layer parameter counts (48, 272, 17).",
              },
              {
                t: "code",
                lang: "python",
                title: "mlp_keras.py (not executed in CI)",
                code: MLP_KERAS,
                output: "Prints the model summary (Dense layers with 48, 272 and 17 parameters; 337 in total), then [test loss, test accuracy]. Accuracy should be in the low-to-mid 90s, similar to the PyTorch width-16 run; exact numbers differ because Keras uses different initialisation and random streams.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Architecture", "Shape", "Advantages", "Disadvantages", "Use cases"],
        rows: [
          ["Single-layer network (no hidden layer)", "input → output", "Convex training, interpretable, fast", "Only linear boundaries / linear functions", "Baselines, linear probes, output heads"],
          ["Shallow wide MLP", "1 hidden layer, many units", "Universal approximator; simple; easy to analyse", "May need exponentially many units; prone to memorising", "Small tabular problems, theory"],
          ["Deep narrow MLP", "many layers, few units each", "Compositional functions with few parameters", "Harder to optimise (vanishing gradients) without normalisation / residuals", "Rarely used plain; motivates residual nets"],
          ["Typical deep MLP", "2–8 hidden layers of 64–1024 units, ReLU/GELU", "Good default for tabular and embedding inputs", "Ignores spatial/sequential structure; many parameters on wide inputs", "Tabular data, recommendation towers, heads on top of encoders"],
          ["Bottleneck / autoencoder MLP", "widths shrink then grow", "Learns compressed representations", "Reconstruction objective may not match the downstream task", "[Autoencoders](/learn/m14/autoencoders), dimensionality reduction"],
          ["MLP block in a Transformer", "$d \\to 4d \\to d$ with GELU, plus a residual connection", "Large capacity per token; parallel", "Most of the model's parameters", "Every [Transformer](/learn/m11/transformer-architecture) layer"],
          ["Residual MLP", "$\\mathbf{h} + F(\\mathbf{h})$ blocks", "Trains very deep stacks", "Needs equal widths (or projections)", "Deep tabular models, MLP-Mixer-style models"],
        ],
      },
    ],

    // ---------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Tabular prediction** — credit risk, demand forecasting, churn: MLPs on standardised features (often competing with gradient-boosted trees).",
          "**Classification heads** — the final layers of image, audio and text models are small MLPs mapping learned embeddings to classes.",
          "**Recommender systems** — two-tower and deep-and-wide models use MLPs to combine user and item embeddings.",
          "**Inside Transformers** — every Transformer layer contains a position-wise feed-forward MLP; in large language models these MLP blocks hold most of the parameters.",
          "**Function approximation in science and control** — MLPs approximate value functions and policies in [reinforcement learning](/learn/m15/deep-rl), and represent continuous fields (e.g. physics-informed networks, neural implicit representations of shapes and scenes).",
          "**Representation learning and probing** — hidden activations of trained networks are reused as features for other tasks, and linear probes on them are a standard analysis tool.",
        ],
      },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Universal:** with enough units an MLP can approximate any continuous function on a bounded domain — expressiveness is never the hard limit.",
          "**Learns features:** removes most hand-crafted feature engineering for non-linear problems.",
          "**Simple and general:** a few matrix multiplications; runs efficiently on GPUs; the same code works for regression and classification by changing only the output layer.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Non-convex training:** results depend on initialisation, learning rate and luck (12/20 seeds for 2-unit XOR).",
          "**No built-in structure:** an MLP treats inputs as an unordered vector; it does not know that neighbouring pixels or consecutive words are related. Convolutions, recurrence and attention add that structure and need far fewer parameters.",
          "**Parameter-hungry on wide inputs:** a single 4096→4096 layer has 16.8 million parameters.",
          "**Poor extrapolation:** outside the training distribution ReLU MLPs become linear and unreliable.",
          "**Opaque:** individual hidden units rarely correspond to human concepts; interpretation needs dedicated tools.",
        ],
      },
      { t: "h", text: "Width versus depth in practice" },
      {
        t: "list",
        items: [
          "Width increases the number of features per layer and smooths optimisation (more paths to a solution); beyond a point it mostly adds parameters.",
          "Depth increases compositional power cheaply, but plain deep MLPs are harder to train ([vanishing gradients](/learn/m04/gradient-problems)); residual connections and normalisation make depth usable.",
          "A common starting point for tabular data: 2–3 hidden layers of 128–512 units with ReLU, then tune on validation data.",
        ],
      },
    ],

    // -------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Stacking layers without non-linearities.** `nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 1))` is a linear model with extra parameters. *Fix:* put an activation between every pair of linear layers (the coding exercise shows the collapse).",
          "**Applying ReLU (or any activation) to the output of a regression network.** A ReLU output can never predict negative values and has zero gradient when it does; a sigmoid output caps predictions at 1. *Fix:* identity output for unbounded regression targets.",
          "**Adding softmax/sigmoid before a fused loss.** `CrossEntropyLoss` and `BCEWithLogitsLoss` expect logits. *Fix:* output raw logits; apply softmax/sigmoid only for reporting probabilities.",
          "**Using softmax for multi-label problems.** Softmax forces the labels to compete (probabilities sum to 1). *Fix:* one sigmoid per label with BCE.",
          "**Believing the universal approximation theorem guarantees success.** It guarantees that a good network exists, not that you can find it with your data and optimizer, nor that it generalises. *Fix:* judge models by validation performance.",
          "**Equating parameter count with overfitting.** Overfitting is a mismatch between capacity and data (plus regularisation); measure it with a validation set rather than inferring it from size alone.",
          "**Tuning hyperparameters on the test set.** Architecture choices are fitted to whatever data you evaluate them on. *Fix:* choose them on a validation set and touch the test set once.",
          "**Symmetric initialisation.** Initialising all hidden weights to the same value makes every hidden unit compute the same feature forever. *Fix:* random initialisation ([Initialization](/learn/m07/initialization)).",
        ],
      },
    ],

    // ------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "Why can't a single-layer network solve XOR, and how does a hidden layer fix it?", a: "A single neuron's decision boundary is one hyperplane, and XOR's classes — $(0,1),(1,0)$ versus $(0,0),(1,1)$ — are not separable by any line. A hidden layer re-maps the inputs: with two hidden units the network can map $(0,1)$ and $(1,0)$ to the same point (a fold), after which one line separates the classes. The output neuron is logistic regression on the hidden features." },
      { t: "qa", level: "beginner", q: "What is the difference between a parameter and a hyperparameter?", a: "Parameters (weights and biases) are learned from the training data by the optimizer. Hyperparameters are set before or around training — number of layers, units per layer, activation, learning rate, batch size, epochs, regularisation — and are chosen using validation data." },
      { t: "qa", level: "beginner", q: "How many parameters does a 784-256-128-10 fully connected network have?", a: "$\\sum (n_{l-1}+1)n_l = 785\\cdot256 + 257\\cdot128 + 129\\cdot10 = 200{,}960 + 32{,}896 + 1{,}290 = 235{,}146$." },
      { t: "qa", level: "intermediate", q: "State the universal approximation theorem and its limitations.", a: "A feed-forward network with one hidden layer of finitely many units and a non-polynomial activation (e.g. sigmoid, ReLU) can approximate any continuous function on a compact subset of $\\mathbb{R}^d$ to any uniform accuracy (Cybenko 1989; Hornik 1991; Leshno et al. 1993). It does not say how many units are needed (possibly exponential in $d$), that gradient descent will find the weights, that the network will generalise from finite data, or anything outside the compact domain; it also does not cover uniform approximation of discontinuous functions." },
      { t: "qa", level: "intermediate", q: "What happens if you remove all activation functions from a deep network?", a: "Each layer is affine, and a composition of affine maps is affine: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. The whole network is equivalent to a single linear layer, so depth adds parameters but no expressive power (and the rank of the product is limited by the narrowest layer)." },
      { t: "qa", level: "intermediate", q: "How do you choose the output layer and loss for regression, binary, multiclass and multi-label tasks?", a: "Regression: identity units with MSE (or Huber/MAE). Binary: one unit, sigmoid, binary cross-entropy. Multiclass with exactly one label: K units, softmax, categorical cross-entropy. Multi-label: K units with independent sigmoids and BCE per label. In frameworks, output logits and use the fused loss (`BCEWithLogitsLoss`, `CrossEntropyLoss`)." },
      { t: "qa", level: "intermediate", q: "What does it mean that hidden layers learn representations? How would you check it?", a: "Each hidden layer outputs a vector that re-describes the input; training shapes these vectors so the output layer (a linear model) can solve the task. Check with a linear probe: freeze the network, extract the hidden activations, and fit a logistic regression on them. On concentric circles a probe on raw coordinates scores ~57%, on the trained 2-D hidden layer 100%." },
      { t: "qa", level: "advanced", q: "Sketch a proof of universal approximation for ReLU networks in one dimension.", a: "Uniform continuity on $[a,b]$ means the piecewise-linear interpolant $\\ell$ at sufficiently fine knots $t_0<\\dots<t_N$ is within $\\varepsilon$ of $g$. And $\\ell(x) = g(t_0) + \\sum_k c_k\\,\\mathrm{ReLU}(x - t_k)$ with $c_0$ the first slope and $c_k$ the slope change at $t_k$ — a one-hidden-layer ReLU network with $N$ units. For $C^2$ functions the error is at most $\\frac{h^2}{8}\\max|g''|$." },
      { t: "qa", level: "advanced", q: "Why can depth be more efficient than width?", a: "A 1-hidden-layer ReLU network on a 1-D input with $N$ units has at most $N+1$ linear pieces, whereas composing a 2-ReLU tent map $k$ times — depth $k$, $2k$ units — produces $2^k$ pieces, so a shallow network needs at least $2^k-1$ units to represent it. Telgarsky (2016) extended this to approximation: some functions computed by deep networks with $\\Theta(k^3)$ layers of constant width cannot be approximated by $O(k)$-layer networks with fewer than $\\Omega(2^k)$ units. More generally, deep ReLU networks can have exponentially many linear regions in depth (Montúfar et al. 2014)." },
      { t: "qa", level: "advanced", q: "If a network has more parameters than training examples, must it overfit?", a: "No. Such a network *can* memorise the training set, but what it actually learns depends on the optimizer, initialisation, architecture and regularisation (implicit and explicit). Large networks trained to zero training error often generalise well, and test error can even fall again as size grows past the interpolation threshold (double descent). Overfitting must be measured on held-out data, not inferred from the parameter count." },
    ],

    // -------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "How many trainable parameters does a fully connected 3-5-2 network have (weights and biases)?", answer: 32, tolerance: 0, explain: "$(3+1)\\cdot5 + (5+1)\\cdot2 = 20 + 12 = 32$." },
          { kind: "numeric", q: "How many parameters does a 784-256-128-10 MLP have?", answer: 235146, tolerance: 0, explain: "$785\\cdot256 + 257\\cdot128 + 129\\cdot10 = 200{,}960 + 32{,}896 + 1{,}290 = 235{,}146$." },
          { kind: "numeric", q: "A one-hidden-layer ReLU network with a scalar input has 7 hidden units. At most how many linear pieces can its output have?", answer: 8, tolerance: 0, explain: "Each unit contributes at most one kink (where $a_jx + c_j = 0$), so at most 7 kinks and 8 pieces." },
          { kind: "numeric", q: "The tent map built from 2 ReLUs is composed with itself 5 times. How many linear pieces does the result have on $[0,1]$?", answer: 32, tolerance: 0, explain: "Each composition doubles the number of pieces: $2^5 = 32$, using only 10 ReLUs. A one-hidden-layer network would need at least 31." },
          { kind: "numeric", q: "Piecewise-linear interpolation of $g(x) = x^2$ on $[0,1]$ with 4 equal pieces (a 4-unit ReLU network). What is the maximum error? Use $\\frac{h^2}{8}\\max|g''|$.", answer: 0.015625, tolerance: 0.0001, explain: "$h = 0.25$, $g'' = 2$: $0.0625/8\\cdot2 = 0.015625$, attained at the midpoints of the pieces." },
          { q: "Which statement does the universal approximation theorem support?", options: ["Gradient descent will find a one-hidden-layer network that fits any dataset", "For any continuous function on a compact set and any ε > 0, some one-hidden-layer network is uniformly within ε of it", "One hidden layer is always the most parameter-efficient architecture", "A wide enough network will generalise to unseen data"], answer: 1, explain: "It is an existence result about approximation on a compact set; it says nothing about optimisation, efficiency or generalisation." },
          { q: "A model must tag photos with any subset of 20 labels (beach, dog, sunset, …). Which output layer is right?", options: ["20 units with softmax and categorical cross-entropy", "20 units with independent sigmoids and binary cross-entropy", "1 unit with sigmoid", "20 units with ReLU and MSE"], answer: 1, explain: "Labels are not mutually exclusive, so each needs its own independent probability; softmax would force them to compete." },
          { q: "Which of these is a parameter rather than a hyperparameter?", options: ["The number of hidden layers", "The learning rate", "The bias of hidden unit 3 in layer 2", "The batch size"], answer: 2, explain: "Biases (and weights) are learned by gradient descent; the others are chosen by the practitioner." },
        ],
      },
      {
        t: "exercise",
        title: "Build a network by hand from hinges",
        prompt: "Write down a one-hidden-layer ReLU network that computes, exactly on $[0, 3]$, the piecewise-linear function through the points $(0, 0), (1, 2), (2, 1), (3, 1)$. Give $W^{[1]}, \\mathbf{b}^{[1]}, W^{[2]}, b^{[2]}$, check it at $x = 1.5$ and $x = 2.5$, and count its parameters.",
        hint: "Compute the slope of each piece; the output weight of the hinge at a knot is the change of slope there.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Slopes", tex: String.raw`s_0 = \tfrac{2-0}{1} = 2,\quad s_1 = \tfrac{1-2}{1} = -1,\quad s_2 = \tfrac{1-1}{1} = 0` },
              { title: "Hinge weights", tex: String.raw`c_0 = s_0 = 2,\quad c_1 = s_1 - s_0 = -3,\quad c_2 = s_2 - s_1 = 1` },
              { title: "Network", tex: String.raw`F(x) = 2\,\mathrm{ReLU}(x) - 3\,\mathrm{ReLU}(x-1) + \mathrm{ReLU}(x-2):\quad W^{[1]} = \begin{pmatrix}1\\1\\1\end{pmatrix},\ \mathbf{b}^{[1]} = \begin{pmatrix}0\\-1\\-2\end{pmatrix},\ W^{[2]} = (2\ \ {-3}\ \ 1),\ b^{[2]} = 0` },
              { title: "Check", tex: String.raw`F(1.5) = 3 - 1.5 + 0 = 1.5\ \ (\text{midway between 2 and 1}),\qquad F(2.5) = 5 - 4.5 + 0.5 = 1` },
              { title: "Parameters", tex: String.raw`(1+1)\cdot3 + (3+1)\cdot1 = 6 + 4 = 10` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Remove the non-linearity",
        prompt: "Take the 2-8-2-1 circles network from the scratch code and replace the tanh hidden activations by the identity (derivative 1). Train it the same way, report the training accuracy, and verify numerically that the trained network is a single affine function $z = w_1x_1 + w_2x_2 + c$ of its input.",
        hint: "An affine function is determined by its values at $(0,0)$, $(1,0)$ and $(0,1)$. Evaluate the network's output logit there, then test the formula at a fourth point.",
        solution: [
          { t: "code", lang: "python", title: "linear_collapse.py", code: COLLAPSE_CODE, output: COLLAPSE_OUT },
          {
            t: "p",
            text: "With identity activations the 45-parameter network can only do what logistic regression does: 49.5% training accuracy, essentially chance on this balanced dataset (a line cannot separate a ring from its centre). The logit at an arbitrary point matches the recovered affine formula to every printed digit, confirming that the three layers collapsed into one. With tanh restored, the same architecture reaches 100%.",
          },
        ],
      },
    ],

    // --------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "An MLP: input → hidden layers $\\mathbf{h}^{[l]} = f(W^{[l]}\\mathbf{h}^{[l-1]} + \\mathbf{b}^{[l]})$ → task-specific output layer. Without non-linear $f$ it collapses to one affine map.",
          "Parameter count: $\\sum_l (n_{l-1}+1)\\,n_l$. Parameters are learned; hyperparameters (depth, width, activation, learning rate, …) are chosen on validation data.",
          "Hidden layers learn representations; the output layer is a linear/logistic/softmax regression on them. Linear probes measure linear separability of a representation (circles: 57% raw → 100% hidden).",
          "ReLU networks are piecewise linear; each layer folds space, and the number of linear regions can grow exponentially with depth.",
          "Universal approximation (Cybenko 1989, Hornik 1991, Leshno et al. 1993): one hidden layer with a non-polynomial activation can uniformly approximate any continuous function on a compact set. Constructive ReLU proof: sums of hinges = piecewise-linear interpolation, error $\\le \\frac{h^2}{8}\\max|g''|$.",
          "The theorem does not bound the number of units, guarantee trainability, imply generalisation, or cover extrapolation and discontinuities.",
          "Depth efficiency: $k$ compositions of a 2-ReLU tent give $2^k$ pieces; a shallow net needs $\\ge 2^k - 1$ units (Telgarsky 2016 for approximation).",
          "Output layers: identity + MSE (regression), sigmoid + BCE (binary), softmax + CE (multiclass), K sigmoids + BCE (multi-label). Output logits for fused losses.",
          "Capacity must match the data: too little underfits, too much (without regularisation) can overfit — measured on held-out data.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "p", text: "You know what an MLP is and what it can represent. Next, learn to train it well and to see inside it:" },
      {
        t: "list",
        items: [
          "[Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) — the training loop, and how to diagnose and control the capacity–data balance introduced here.",
          "[Forward Propagation](/learn/m04/forward-propagation) — the layer-by-layer computation, with every number worked out, in scalar and matrix form.",
          "[Backpropagation](/learn/m04/backpropagation) — how the gradients used by every training loop in this lesson are computed.",
          "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — why plain deep MLPs are hard to train, the practical counterweight to depth efficiency.",
        ],
      },
    ],
  },
};

export default lesson;
