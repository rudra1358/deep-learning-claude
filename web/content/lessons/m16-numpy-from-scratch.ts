import type { Lesson } from "@/lib/types";

const NN_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ============================ layers ============================
class Layer:
    """Base class. A layer maps x -> y in forward() and dL/dy -> dL/dx in backward().
    Layers with weights store them in self.params and their gradients in self.grads (same keys)."""

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


class Linear(Layer):
    """y = x @ W + b with x of shape (N, n_in), W of shape (n_in, n_out)."""

    def __init__(self, n_in, n_out, rng):
        super().__init__()
        # He initialisation: std = sqrt(2 / n_in) keeps activation variance stable through ReLUs
        self.params["W"] = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.params["b"] = np.zeros(n_out)

    def forward(self, x):
        self.x = x                                         # cache the input: dW needs it
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):                          # grad_out = dL/dy, shape (N, n_out)
        self.grads["W"] = self.x.T @ grad_out              # (n_in, N) @ (N, n_out): sums over the batch
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T               # dL/dx, shape (N, n_in)


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0                                  # cache where the input was positive
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask                        # derivative is 1 where x > 0, else 0


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):                # reverse order: chain rule from the loss back
            grad_out = layer.backward(grad_out)
        return grad_out

    def parameters(self):
        """(layer, key) handles let optimizers update params in place and keep per-param state."""
        return [(layer, k) for layer in self.layers for k in layer.params]


# ============================ loss ============================
class SoftmaxCrossEntropy:
    """Mean cross-entropy of softmax(logits). Fused because the combined gradient is simply (p - y)/N."""

    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)    # subtract row max: exp() cannot overflow
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        self.p, self.y = np.exp(log_p), y
        return -log_p[np.arange(len(y)), y].mean()

    def backward(self):
        g = self.p.copy()
        g[np.arange(len(self.y)), self.y] -= 1.0           # p - one_hot(y)
        return g / len(self.y)


# ============================ optimizers ============================
class SGD:
    def __init__(self, params, lr):
        self.params, self.lr = params, lr

    def step(self):
        for layer, k in self.params:
            layer.params[k] -= self.lr * layer.grads[k]


class Momentum:
    def __init__(self, params, lr, beta=0.9):
        self.params, self.lr, self.beta = params, lr, beta
        self.v = [np.zeros_like(layer.params[k]) for layer, k in params]

    def step(self):
        for (layer, k), v in zip(self.params, self.v):
            v *= self.beta
            v += layer.grads[k]                            # v <- beta * v + g
            layer.params[k] -= self.lr * v


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(layer.params[k]) for layer, k in params]
        self.s = [np.zeros_like(layer.params[k]) for layer, k in params]
        self.t = 0

    def step(self):
        self.t += 1
        for (layer, k), m, s in zip(self.params, self.m, self.s):
            g = layer.grads[k]
            m[:] = self.b1 * m + (1 - self.b1) * g         # first moment (mean of g)
            s[:] = self.b2 * s + (1 - self.b2) * g * g     # second moment (mean of g^2)
            m_hat = m / (1 - self.b1 ** self.t)            # bias correction for the zero start
            s_hat = s / (1 - self.b2 ** self.t)
            layer.params[k] -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


# ============================ gradient check ============================
def gradient_check(model, loss_fn, x, y, h=1e-5):
    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())
    worst = 0.0
    for layer, k in model.parameters():
        P, G = layer.params[k], layer.grads[k].copy()
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + h; lp = loss_fn.forward(model.forward(x), y)
            P[idx] = old - h; lm = loss_fn.forward(model.forward(x), y)
            P[idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - G[idx]) / max(1e-12, abs(num) + abs(G[idx])))
    return worst


# ============================ data ============================
digits = load_digits()                                     # 1797 8x8 grey images, pixel values 0..16
X = digits.data / 16.0                                     # scale pixels to [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(X, digits.target, test_size=0.2, random_state=0, stratify=digits.target)
print(f"train {X_tr.shape}, test {X_te.shape}, classes {len(np.unique(y_tr))}")

# gradient check on a tiny network (every parameter perturbed, so keep it small)
rng = np.random.default_rng(0)
tiny = Sequential(Linear(64, 8, rng), ReLU(), Linear(8, 10, rng))
print(f"gradient check, worst relative error: {gradient_check(tiny, SoftmaxCrossEntropy(), X_tr[:16], y_tr[:16]):.1e}")


def accuracy(model, X, y):
    return np.mean(model.forward(X).argmax(axis=1) == y)


def train(make_opt, epochs=20, batch_size=32, log=False):
    rng = np.random.default_rng(0)                         # same init + same batch order for every optimizer
    model = Sequential(Linear(64, 64, rng), ReLU(), Linear(64, 10, rng))
    loss_fn, opt = SoftmaxCrossEntropy(), make_opt(model.parameters())
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_tr))                 # reshuffle every epoch
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            loss = loss_fn.forward(model.forward(X_tr[idx]), y_tr[idx])   # 1. forward
            model.backward(loss_fn.backward())                            # 2. backward
            opt.step()                                                    # 3. update
            total += loss * len(idx)
        if log and epoch % 5 == 0:
            print(f"  epoch {epoch:2d}  train loss {total / len(X_tr):.4f}  "
                  f"train acc {accuracy(model, X_tr, y_tr):.3f}  test acc {accuracy(model, X_te, y_te):.3f}")
    return accuracy(model, X_te, y_te)


print("Adam (lr 1e-3):")
results = {"Adam lr=1e-3": train(lambda p: Adam(p, lr=1e-3), log=True)}
results["SGD lr=0.1"] = train(lambda p: SGD(p, lr=0.1))
results["Momentum lr=0.01"] = train(lambda p: Momentum(p, lr=0.01))
for name, acc in results.items():
    print(f"{name:17s} final test accuracy {acc:.3f}")`;

const NN_NUMPY_OUT = String.raw`train (1437, 64), test (360, 64), classes 10
gradient check, worst relative error: 4.2e-08
Adam (lr 1e-3):
  epoch  5  train loss 0.4919  train acc 0.928  test acc 0.919
  epoch 10  train loss 0.2277  train acc 0.955  test acc 0.933
  epoch 15  train loss 0.1532  train acc 0.974  test acc 0.958
  epoch 20  train loss 0.1138  train acc 0.982  test acc 0.964
Adam lr=1e-3      final test accuracy 0.964
SGD lr=0.1        final test accuracy 0.967
Momentum lr=0.01  final test accuracy 0.964`;

const NN_TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---- Part 1: autograd reproduces the hand-computed Linear + softmax-CE gradients ----
x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
W = torch.tensor([[0.5, -0.2, 0.1], [0.3, 0.4, -0.6]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64, requires_grad=True)
x.requires_grad_(True)
loss = F.cross_entropy(x @ W + b, torch.tensor([0]))   # fused log-softmax + NLL, like our class
loss.backward()
print("loss", round(loss.item(), 6))
print("dW  ", W.grad.numpy().round(6).tolist())
print("db  ", b.grad.numpy().round(6).tolist())
print("dx  ", x.grad.numpy().round(6).tolist())

# ---- Part 2: the same 64-64-10 digits classifier in idiomatic PyTorch ----
torch.manual_seed(0)
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.2,
                                          random_state=0, stratify=digits.target)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))   # = our Sequential
opt = torch.optim.Adam(model.parameters(), lr=1e-3)                      # = our Adam
print("parameters:", sum(p.numel() for p in model.parameters()))

for epoch in range(1, 21):
    perm = torch.randperm(len(X_tr))
    for start in range(0, len(perm), 32):
        idx = perm[start:start + 32]
        loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])   # forward
        opt.zero_grad()
        loss.backward()                                        # backward (autograd builds it for us)
        opt.step()                                             # update
    if epoch % 5 == 0:
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        print(f"epoch {epoch:2d}  last batch loss {loss.item():.4f}  test acc {acc:.3f}")`;

const NN_TORCH_OUT = String.raw`loss 0.590924
dW   [[-0.446184, 0.371234, 0.074951], [-0.892369, 0.742467, 0.149902]]
db   [-0.446184, 0.371234, 0.074951]
dx   [[-0.289844, -0.030332]]
parameters: 4810
epoch  5  last batch loss 0.4672  test acc 0.908
epoch 10  last batch loss 0.2283  test acc 0.931
epoch 15  last batch loss 0.1394  test acc 0.958
epoch 20  last batch loss 0.1081  test acc 0.969`;

const NN_KERAS = String.raw`import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

tf.random.set_seed(0)
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.2,
                                          random_state=0, stratify=digits.target)

# the same 64-64-10 network: Dense = our Linear (+ optional activation)
model = keras.Sequential([
    keras.Input(shape=(64,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),                                # logits
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # = SoftmaxCrossEntropy
              metrics=["accuracy"])
model.summary()                                            # 4,810 trainable parameters
model.fit(X_tr.astype(np.float32), y_tr, batch_size=32, epochs=20, verbose=2)
loss, acc = model.evaluate(X_te.astype(np.float32), y_te, verbose=0)
print(f"test accuracy {acc:.3f}")`;

const TANH_SOL = String.raw`import numpy as np


class Layer:
    def __init__(self):
        self.params, self.grads = {}, {}


class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)                    # cache the OUTPUT: tanh'(x) = 1 - tanh(x)^2
        return self.out

    def backward(self, grad_out):
        return grad_out * (1.0 - self.out ** 2)


# check backward() with a random upstream gradient g: L = sum(g * tanh(x)) has dL/dx = backward(g)
rng = np.random.default_rng(0)
x, g = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
layer = Tanh()
layer.forward(x)
analytic = layer.backward(g)
h, numeric = 1e-6, np.zeros_like(x)
for idx in np.ndindex(x.shape):
    xp, xm = x.copy(), x.copy()
    xp[idx] += h; xm[idx] -= h
    numeric[idx] = ((g * np.tanh(xp)).sum() - (g * np.tanh(xm)).sum()) / (2 * h)
print(f"max |analytic - numeric| = {np.abs(analytic - numeric).max():.1e}")`;

const TANH_SOL_OUT = String.raw`max |analytic - numeric| = 9.7e-11`;

const lesson: Lesson = {
  id: "m16/numpy-from-scratch",
  title: "A Complete Neural Network in NumPy",
  summary: "The capstone of the from-scratch thread: we build a small, modular deep learning library — a Layer base class, Linear, ReLU, fused softmax cross-entropy, SGD/Momentum/Adam, mini-batch training and a gradient check — and it reaches 96% test accuracy on handwritten digits.",
  minutes: 70,
  level: "intermediate",
  prerequisites: ["m04/backpropagation", "m04/gradient-checking", "m05/classification-losses", "m06/momentum-adaptive", "m07/initialization"],
  next: ["m16/pytorch", "m16/tensorflow-keras", "m17/experimentation"],
  terms: ["layer-api", "module-composition", "training-loop", "optimizer-state", "fused-softmax-cross-entropy"],
  sections: {
    intro: [
      { t: "p", text: "Earlier lessons built each ingredient of deep learning on its own: the forward pass, [backpropagation](/learn/m04/backpropagation), [cross-entropy](/learn/m05/classification-losses), [gradient descent and Adam](/learn/m06/momentum-adaptive), [He initialization](/learn/m07/initialization). Now we join them into one small **library** — a few classes that follow fixed rules, so they combine into any feed-forward network — and train it on real data." },
      { t: "p", text: "The goal is not to compete with PyTorch but to see that a framework is a surprisingly small set of ideas: (1) **layers** that compute their output and turn an output gradient into an input gradient; (2) a **container** that runs layers forward, then backward in reverse; (3) a **loss** that starts the backward pass; (4) **optimizers** that update each parameter from its gradient, sometimes keeping extra memory per parameter; (5) a **training loop** feeding mini-batches through it all. PyTorch's `nn.Module`/`torch.optim` and Keras's `Layer`/`Model.fit` are industrial versions of exactly these five pieces." },
      { t: "p", text: "Building it once pays off for years: when a framework model refuses to train, you will know which piece to suspect, what each remembers, and how to test it alone with a gradient check." },
      { t: "callout", kind: "key", text: "A framework = layers with `forward`/`backward` + a container that chains them + a loss that seeds the gradient + optimizers that own update rules and state + a loop. About 150 lines of NumPy implement all of it." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Backpropagation](/learn/m04/backpropagation) — the recurrence $\\delta^{[l-1]} = (W^{[l]\\top}\\delta^{[l]})\\odot f'(z^{[l-1]})$ and the batch formula $\\partial\\mathcal{L}/\\partial W = A^\\top\\Delta$. Here they are split between the `Linear` and `ReLU` classes.",
        "[Numerical gradient checking](/learn/m04/gradient-checking) — central differences and the relative-error test; we use it to certify every layer.",
        "[Classification losses](/learn/m05/classification-losses) — softmax, cross-entropy, and why their combined gradient is $p - y$.",
        "[Momentum and adaptive optimizers](/learn/m06/momentum-adaptive) — the momentum and Adam update rules (bias correction included), which we write as classes with state.",
        "[Weight initialization](/learn/m07/initialization) — why ReLU layers use He initialization with standard deviation $\\sqrt{2/n_{in}}$.",
        "[Vectorization](/learn/m00/vectorization) — batched matrix products and broadcasting; nothing in the library loops over examples.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "An assembly line that can run in reverse", text: "Picture a factory line of stations. Each takes a part, changes it and passes it on — that is `forward`. At the end an inspector measures how far the product is from the spec (the loss). Then the complaint travels back up the line: each station hears \"here is how your output should have differed\", uses its notes from the forward run to work out how its input should have differed, and passes that upstream — that is `backward`. Stations with knobs (weights) also note how each knob should turn. Finally a technician (the optimizer) turns every knob a little. No station needs to know about the others; it only follows the protocol." },
      { t: "p", text: "That protocol makes the code modular. A `Linear` layer does not know whether a `ReLU`, another `Linear` or the loss comes next; it just receives $\\partial\\mathcal{L}/\\partial(\\text{its output})$. So layers stack in any order, and calling `backward` on each in reverse applies the chain rule for you." },
      { t: "p", text: "Two more ideas. Each layer must **remember what it saw** in the forward pass (the input for `Linear`, the sign pattern for `ReLU`), because the backward formulas need it. And the optimizer is **separate from the layers**: SGD, momentum and Adam use the same gradients but update differently (Adam also keeps two running averages per parameter), so swapping optimizers changes one line." },
    ],
    mechanism: [
      { t: "h", text: "The rules every layer follows" },
      { t: "table", head: ["Class", "forward(x) computes and caches", "backward(g) returns", "Gradients stored"], rows: [
        ["`Linear(n_in, n_out)`", "$y = xW + b$; caches $x$", "$g\\,W^\\top$", "`grads['W']` $= x^\\top g$, `grads['b']` $=\\sum_n g_n$"],
        ["`ReLU()`", "$\\max(x, 0)$; caches the mask $x > 0$", "$g \\odot \\text{mask}$", "none"],
        ["`Sequential(*layers)`", "calls each layer's forward in order", "calls each layer's backward in **reverse** order", "collects `(layer, key)` handles"],
        ["`SoftmaxCrossEntropy`", "mean $-\\log p_{y}$; caches $p$, $y$", "$(p - \\text{onehot}(y))/N$", "none (it starts the backward pass)"],
      ] },
      { t: "p", text: "$g$ always means the loss gradient with respect to the layer's **output** (output's shape); the returned array is the gradient with respect to its **input** (input's shape). This shape rule is the library's most useful debugging check." },
      { t: "h", text: "One training step" },
      { t: "steps", items: [
        { title: "Sample a mini-batch", text: "Shuffle the training indices once per epoch and take consecutive slices of `batch_size` (32 here). Each slice gives $X_B \\in \\mathbb{R}^{B\\times 64}$ and labels $y_B$." },
        { title: "Forward through the container", text: "`Sequential.forward` calls `Linear → ReLU → Linear`, and each layer caches what its backward needs. The result is a logits matrix of shape $(B, 10)$." },
        { title: "Loss", text: "`SoftmaxCrossEntropy.forward` subtracts each row's maximum (so `exp` cannot overflow), computes log-probabilities and returns the mean negative log-probability of the correct class." },
        { title: "Seed the backward pass", text: "`loss.backward()` returns $(p - \\text{onehot}(y))/B$ — the gradient of the *mean* loss with respect to the logits." },
        { title: "Backward through the container", text: "`Sequential.backward` hands that array to the last `Linear`, which stores $\\partial\\mathcal{L}/\\partial W$ and $\\partial\\mathcal{L}/\\partial b$ and returns $\\partial\\mathcal{L}/\\partial x$; `ReLU` zeroes entries where its input was negative; the first `Linear` stores its gradients. The gradient for the raw pixels is ignored." },
        { title: "Optimizer step", text: "The optimizer loops over the `(layer, key)` handles from `parameters()`, reads `layer.grads[key]` and updates `layer.params[key]` **in place**, using and updating any per-parameter state (velocity for momentum; first and second moments for Adam)." },
        { title: "Repeat, then evaluate", text: "After each epoch (one pass over all training batches), run the forward pass on held-out data and compare the logits' arg-max with the labels. Evaluation needs no backward pass." },
      ] },
      { t: "callout", kind: "warning", title: "Forward and backward must be paired", text: "`backward` uses whatever the latest `forward` cached. Calling `forward` on test data between a training forward and its backward silently overwrites the cache, so the gradients belong to the wrong batch. PyTorch builds a fresh graph per forward call; in our library you must keep the calls paired." },
    ],
    math: [
      { t: "h", text: "Linear layer: the three gradients" },
      { t: "p", text: "Take a batch $X \\in \\mathbb{R}^{N\\times n_{in}}$, weights $W \\in \\mathbb{R}^{n_{in}\\times n_{out}}$ and bias $b \\in \\mathbb{R}^{n_{out}}$. The layer computes $Y = XW + \\mathbf{1}b^\\top$ (the bias is copied to every row). Let $G = \\partial\\mathcal{L}/\\partial Y \\in \\mathbb{R}^{N\\times n_{out}}$ be the incoming gradient. Since $Y_{nj} = \\sum_k X_{nk}W_{kj} + b_j$:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W_{kj}} = \sum_{n} G_{nj}X_{nk}\ \Rightarrow\ \frac{\partial\mathcal{L}}{\partial W} = X^\top G,\qquad \frac{\partial\mathcal{L}}{\partial b_j} = \sum_n G_{nj},\qquad \frac{\partial\mathcal{L}}{\partial X_{nk}} = \sum_j G_{nj}W_{kj}\ \Rightarrow\ \frac{\partial\mathcal{L}}{\partial X} = G\,W^\top`, caption: "The shapes check themselves: $X^\\top G$ is $(n_{in}\\times N)(N\\times n_{out})$, and $GW^\\top$ is $(N\\times n_{out})(n_{out}\\times n_{in})$." },
      { t: "p", text: "In plain words: a weight's gradient is \"input times upstream gradient\", summed over the batch; the input's gradient sends the upstream gradient back through the same weights. These are BP3/BP2 from [Backpropagation](/learn/m04/backpropagation) with examples as rows and $W$ stored as $n_{in}\\times n_{out}$, like a Keras `Dense` kernel (PyTorch's `nn.Linear` stores the transpose). The sum over $n$ in $\\partial\\mathcal{L}/\\partial b$ is the gradient of broadcasting: a value copied to $N$ rows receives the sum of the $N$ row gradients." },
      { t: "h", text: "ReLU and the fused loss" },
      { t: "math", tex: String.raw`\mathrm{ReLU}(x) = \max(x, 0),\qquad \frac{\partial\mathcal{L}}{\partial x} = G \odot \mathbb{1}[x > 0]` },
      { t: "p", text: "For logits $z_n \\in \\mathbb{R}^C$ ($C$ classes), $p_n = \\mathrm{softmax}(z_n)$ and the mean loss $\\mathcal{L} = -\\frac{1}{N}\\sum_n \\log p_{n,y_n}$, the softmax Jacobian and the log combine into a very simple result:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial z_{n}} = \frac{1}{N}\left(p_n - e_{y_n}\right)`, caption: "$e_{y_n}$ is the one-hot vector of the true class. Computing softmax and cross-entropy in one class means we never take the $\\log$ of a probability that underflowed to 0." },
      { t: "p", text: "In code we use the log-sum-exp identity $\\log p_{n,c} = (z_{n,c} - m_n) - \\log\\sum_j e^{z_{n,j} - m_n}$ with $m_n = \\max_j z_{n,j}$. Subtracting $m_n$ changes nothing mathematically but keeps every exponent $\\le 0$." },
      { t: "h", text: "The three optimizers" },
      { t: "p", text: "For a parameter $\\theta$ with gradient $g_t$ at step $t$ and learning rate $\\eta$:" },
      { t: "math", tex: String.raw`\begin{aligned} \text{SGD:}\quad & \theta \leftarrow \theta - \eta\, g_t\\ \text{Momentum:}\quad & v_t = \beta v_{t-1} + g_t,\quad \theta \leftarrow \theta - \eta\, v_t\\ \text{Adam:}\quad & m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2\\ & \hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat s_t = \frac{s_t}{1-\beta_2^t},\quad \theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon} \end{aligned}`, caption: "Defaults in the code: $\\beta = 0.9$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$. $v, m, s$ are the optimizer's state — one array per parameter, with the parameter's shape." },
      { t: "p", text: "State is why optimizers are objects, not functions: Adam needs $m$, $s$ and the step counter $t$ to survive from one step to the next, and a checkpoint must save them (see [PyTorch](/learn/m16/pytorch))." },
      { t: "h", text: "Worked example: one Linear layer + softmax cross-entropy" },
      { t: "p", text: "One example $x = (1, 2)$, true class $y = 0$, three classes, $W = \\begin{pmatrix}0.5 & -0.2 & 0.1\\\\ 0.3 & 0.4 & -0.6\\end{pmatrix}$, $b = (0, 0.1, 0.2)$. All numbers are float64, and PyTorch autograd reproduces them in the Framework section." },
      { t: "steps", items: [
        { title: "Logits", tex: String.raw`z = xW + b = (0.5 + 0.6 + 0,\ -0.2 + 0.8 + 0.1,\ 0.1 - 1.2 + 0.2) = (1.1,\ 0.7,\ -0.9)` },
        { title: "Softmax", tex: String.raw`p = \frac{(e^{1.1}, e^{0.7}, e^{-0.9})}{e^{1.1}+e^{0.7}+e^{-0.9}} = (0.553816,\ 0.371234,\ 0.074951)` },
        { title: "Loss", tex: String.raw`\mathcal{L} = -\log 0.553816 = 0.590924` },
        { title: "Gradient w.r.t. logits (N = 1)", tex: String.raw`G = p - e_0 = (-0.446184,\ 0.371234,\ 0.074951)` },
        { title: "Weight gradient", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W} = x^\top G = \begin{pmatrix}1\\2\end{pmatrix}(-0.446184,\ 0.371234,\ 0.074951) = \begin{pmatrix}-0.446184 & 0.371234 & 0.074951\\ -0.892369 & 0.742467 & 0.149902\end{pmatrix}` },
        { title: "Bias and input gradients", tex: String.raw`\frac{\partial\mathcal{L}}{\partial b} = G,\qquad \frac{\partial\mathcal{L}}{\partial x} = G W^\top = (0.5(-0.446184) - 0.2(0.371234) + 0.1(0.074951),\ \dots) = (-0.289844,\ -0.030332)` },
        { title: "SGD step with η = 0.1", text: "Every weight moves against its gradient; the loss on this example drops from 0.590924 to 0.411756.", tex: String.raw`W_{11} \leftarrow 0.5 - 0.1(-0.446184) = 0.544618` },
        { title: "Adam's first step on the same weight (η = 0.01)", tex: String.raw`m_1 = 0.1g = -0.044618,\ s_1 = 0.001g^2 = 0.000199,\ \hat m_1 = g,\ \hat s_1 = g^2\ \Rightarrow\ \Delta = 0.01\frac{g}{|g|+10^{-8}} \approx -0.01` },
      ] },
      { t: "p", text: "Worth remembering: after bias correction, Adam's **first** step is about $\\eta$ in size for every parameter, whatever the gradient's size — its step size is set by $\\eta$, not the gradient scale. Read the signs too: the correct class's logit has a negative gradient (raising it lowers the loss); wrong classes have positive gradients." },
      { t: "h", text: "Size of the digits network" },
      { t: "p", text: "The model is $64 \\to 64 \\to 10$: $64\\cdot64 + 64 = 4160$ parameters in the first `Linear` and $64\\cdot10 + 10 = 650$ in the second, $4810$ in total. Adam keeps two more arrays of the same size, so its state is $2\\times4810 = 9620$ numbers. For large models, optimizer state rather than weights often dominates memory." },
    ],
    visual: [
      { t: "p", text: "The network simulator runs exactly this lesson's loop — forward with caching, loss, backward in reverse, update — on a tiny network where every number is visible." },
      { t: "sim", id: "network", caption: "Step through forward, loss and backward; click a connection to compare its backprop gradient with a finite-difference estimate (the gradient check our library runs on every parameter)." },
      { t: "list", items: [
        "Step through one iteration and name the library class behind each value: activations (`Linear.forward`/`ReLU.forward`), $\\delta$ values (`backward` return values), weight gradients (`Linear.grads`), the update (the optimizer).",
        "Switch a hidden layer to ReLU and find a neuron with a negative pre-activation: its incoming weight gradients are exactly 0. That is the `mask` in `ReLU.backward`.",
        "Click several connections: backprop and numerical gradients agree to many digits — the test `gradient_check` runs on all 610 parameters of the scratch code's tiny network.",
      ] },
      { t: "p", text: "The optimizer race shows why the optimizer is a separate, swappable object: same gradients, three different update rules." },
      { t: "sim", id: "optimizers", caption: "SGD, Momentum and Adam (and others) on the same 2-D loss surfaces." },
      { t: "list", items: [
        "Pick the long narrow valley and race SGD against Momentum: momentum's velocity $v$ (the state our `Momentum` class stores) builds up along the valley floor.",
        "Compare Adam's first few steps with SGD's: Adam's steps are about the learning rate in size in each coordinate, even where the gradient is tiny — the first-step property from the worked example.",
      ] },
    ],
    scratch: [
      { t: "p", text: "The whole library and experiment in one file: layers, container, fused loss, three optimizers, a gradient check, and mini-batch training on scikit-learn's built-in digits (1797 grayscale $8\\times8$ images of 0–9, no download). It runs in seconds." },
      { t: "code", lang: "python", title: "mini_nn.py", code: NN_NUMPY, output: NN_NUMPY_OUT },
      { t: "p", text: "Reading the code against the maths and the design:" },
      { t: "list", items: [
        "**`Layer`** sets the rules: `params` and `grads` are dictionaries with matching keys, so any optimizer can update any layer without knowing its type. `ReLU` just has empty dictionaries.",
        "**`Linear.backward`** is the three formulas from the Math section: `self.x.T @ grad_out`, `grad_out.sum(axis=0)`, `grad_out @ W.T`. It needs the cached input `self.x` — which is why activations dominate training memory.",
        "**`Sequential.parameters`** returns `(layer, key)` handles, not arrays. The optimizer updates `layer.params[k]` in place, so the layer sees the new value, and its state lists line up with the handles by position.",
        "**`SoftmaxCrossEntropy`** returns the *mean* loss and the gradient divided by $N$. Keeping these consistent means the learning rate does not silently change with the batch size.",
        "**`gradient_check`** nudges each of the 610 weights and biases of a $64\\to8\\to10$ network by $\\pm10^{-5}$ in float64. A worst relative error around $10^{-8}$ certifies `Linear`, `ReLU` and the loss together; a bug usually gives $10^{-2}$ or worse.",
        "**`train`** uses the same seed for initialization and batch order for every optimizer, so the comparison is fair. Learning rates are typical for each: SGD $0.1$, momentum $0.01$ (its effective step is about $\\eta/(1-\\beta) = 10\\eta$), Adam $10^{-3}$. All three reach about 96–97% test accuracy on the 360 held-out images after 20 epochs.",
        "The **test** set is used here only to report progress. A real experiment would tune learning rates on a separate validation split — see [Data Splits](/learn/m17/data-splits).",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Part 1 checks our hand-derived Linear + softmax-CE gradients against autograd. Part 2 is the digits experiment in normal PyTorch style: `nn.Linear` is our `Linear` (it stores the weight as $(n_{out}, n_{in})$ and computes $xW^\\top + b$), `F.cross_entropy` is our fused loss, `torch.optim.Adam` is our `Adam`, and `loss.backward()` replaces the hand-written `backward` methods." },
          { t: "code", lang: "python", title: "digits_pytorch.py", code: NN_TORCH, output: NN_TORCH_OUT },
          { t: "p", text: "Same architecture (4810 parameters), same optimizer settings, similar accuracy (numbers differ slightly because PyTorch's default initialization and random stream differ). The loop has the same three lines — forward, backward, step — plus `opt.zero_grad()`, needed because PyTorch **adds** new gradients into `.grad`, while our layers overwrite `grads` each backward." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "In Keras the training loop is hidden inside `model.fit`. `Dense(64, activation=\"relu\")` combines our `Linear` and `ReLU`; `SparseCategoricalCrossentropy(from_logits=True)` is the fused softmax cross-entropy with integer labels." },
          { t: "code", lang: "python", title: "digits_keras.py (not executed in CI)", code: NN_KERAS, output: "Prints the model summary (two Dense layers, 4,810 trainable parameters), one line per epoch for 20 epochs with the loss and accuracy, then the test accuracy. Exact values depend on the TensorFlow version and its random initialization." },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "Design choices for a framework" },
      { t: "table", head: ["Design", "How gradients are obtained", "Strengths", "Weaknesses", "Examples"], rows: [
        ["Layer-wise manual backward (this lesson)", "Each layer class writes its own `backward`", "Transparent; no graph bookkeeping; little memory overhead", "Every new operation needs a hand-derived backward; only works for chains and simple graphs", "Early libraries (Caffe layers, Torch7 `nn` modules) worked this way"],
        ["Define-by-run autodiff", "Operations are recorded as they run; reverse-mode AD walks the record backwards", "Any Python control flow; you write only the forward", "Overhead per operation; graph rebuilt every step", "PyTorch autograd, TensorFlow eager `GradientTape`, JAX tracing"],
        ["Static graph", "The whole computation is declared first, then compiled and differentiated", "Whole-program optimization, easy deployment", "Harder debugging; control flow needs special ops", "TensorFlow 1.x graphs; today `tf.function` / `torch.compile` capture graphs from eager code"],
      ] },
      { t: "h", text: "Variations inside the library" },
      { t: "table", head: ["Component", "Alternatives", "When to choose", "Trade-off"], rows: [
        ["Loss", "Separate `Softmax` layer + `CrossEntropy` loss", "Never for training; fine for inference", "Less stable (log of an underflowed probability) and a messier gradient"],
        ["Parameter handles", "Optimizer holds direct array references", "Simpler code when arrays are only ever updated in place", "Breaks if any code points `layer.params[k]` at a new array"],
        ["Optimizer", "SGD / Momentum / Adam / AdamW", "Adam(W) as a robust default; SGD+momentum often for CNNs with a tuned schedule", "Adam stores 2 extra arrays per parameter; SGD needs more LR tuning"],
        ["Training loop", "Full-batch / mini-batch / online (batch size 1)", "Mini-batch (16–512) almost always", "Small batches are noisy but cheap per step; big batches use hardware better"],
        ["Precision", "float64 / float32 / bfloat16", "float64 for gradient checks, float32 (or lower) for training", "float64 is ~2× slower and uses more memory"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Debugging frameworks.** Knowing what `nn.Linear` caches, why `zero_grad` exists and what Adam's state is makes \"mysterious\" bugs familiar.",
        "**Writing a new layer.** When you invent an operation with a custom backward (PyTorch's `torch.autograd.Function`, TensorFlow's `tf.custom_gradient`), you write exactly a `forward`/`backward` pair and verify it with a gradient check, as here.",
        "**Inference without dependencies.** A trained network's forward pass is a few matrix products; running exported weights in NumPy (or C, or JavaScript, as this site's simulators do) avoids shipping a framework.",
        "**Teaching and interviews.** \"Implement a two-layer network with backprop in NumPy\" is a standard machine learning interview task; this library is a complete answer.",
        "**Testing framework behaviour.** A small reference implementation is a referee: if your PyTorch and NumPy models disagree on the same weights and inputs, one has a bug.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Full transparency**: every number can be printed and every formula is visible; there is no hidden state beyond what you wrote.",
        "**Tiny and dependency-free**: NumPy only; easy to embed or port.",
        "**No automatic differentiation**: every new operation needs a hand-derived, gradient-checked backward. Branching architectures (skip connections, attention) need a graph, not a `Sequential` chain — which is what autodiff frameworks provide.",
        "**CPU only and unfused**: NumPy cannot use GPUs, and every operation creates a full intermediate array; frameworks fuse operations on accelerators, often 10–100× faster for large models.",
        "**No ecosystem**: data loaders, mixed precision, checkpoints, distributed training, pretrained models and deployment tools come with real frameworks — the next two lessons.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Unpaired forward/backward.** Evaluating between a training forward and its backward overwrites the cached activations, giving gradients for the wrong inputs. Fix: call `backward` right after its `forward` (or make layers return a cache object).",
        "**Mismatched scaling of loss and gradient.** Returning the mean loss but a summed gradient (or the reverse) multiplies the effective learning rate by $N$. Fix: divide the seed gradient by $N$ exactly when the loss is a mean; the gradient check catches the mismatch.",
        "**Forgetting to sum the bias gradient over the batch.** `grads['b'] = grad_out` has shape $(N, n_{out})$, and broadcasting may hide the error until the update. Fix: `grad_out.sum(axis=0)`, and assert that gradient shapes equal parameter shapes.",
        "**Rebinding instead of updating in place** when the optimizer holds references: `W = W - lr * g` makes a new array, the layer keeps the old one, and nothing learns. Fix: update in place (`-=`) or use handles, as `Sequential.parameters` does.",
        "**Softmax without subtracting the max.** Logits around 1000 make `np.exp` return `inf` and the loss `nan`. Fix: subtract the row maximum, or better, compute log-softmax directly.",
        "**Gradient-checking in float32 or with $h$ too small.** Float32 rounding in $(f(\\theta+h) - f(\\theta-h))$ gives relative errors of $10^{-3}$ that look like bugs. Fix: check in float64 with $h\\approx10^{-5}$ on a small network and batch.",
        "**Setting all weights to zero.** All hidden units then compute the same thing and get the same gradient forever. Fix: random initialization scaled to the layer (He for ReLU).",
        "**Reporting tuned results on the test set.** Choosing the learning rate by test accuracy leaks information. Fix: tune on a validation split and look at the test set once.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What does a layer's `backward` method receive and return?", a: "It receives the gradient of the loss with respect to the layer's output (same shape as the output) and returns the gradient with respect to the layer's input (same shape as the input). Layers with parameters also store their parameters' gradients." },
      { t: "qa", level: "beginner", q: "Why does a `Linear` layer cache its input during the forward pass?", a: "Because its weight gradient is $X^\\top G$: each weight's gradient is the upstream gradient times the input that flowed through it. Without the cached $X$ the backward pass could not compute $\\partial\\mathcal{L}/\\partial W$." },
      { t: "qa", level: "beginner", q: "Why are softmax and cross-entropy implemented as one class?", a: "Their combined gradient simplifies to $(p - \\text{onehot}(y))/N$, which is cheap and stable, and computing log-softmax via log-sum-exp never takes the log of an underflowed probability. Done separately, you would compute a full softmax Jacobian and risk $\\log 0$." },
      { t: "qa", level: "intermediate", q: "Derive the gradients of $Y = XW + b$ for a batch.", a: "With $G = \\partial\\mathcal{L}/\\partial Y$: $Y_{nj} = \\sum_k X_{nk}W_{kj} + b_j$, so $\\partial\\mathcal{L}/\\partial W_{kj} = \\sum_n X_{nk}G_{nj}$, i.e. $X^\\top G$; $\\partial\\mathcal{L}/\\partial b_j = \\sum_n G_{nj}$ (the bias is copied to every row, so its gradient sums over rows); $\\partial\\mathcal{L}/\\partial X_{nk} = \\sum_j G_{nj}W_{kj}$, i.e. $GW^\\top$." },
      { t: "qa", level: "intermediate", q: "Why is the optimizer a separate object with state, rather than a method on each layer?", a: "Update rules do not depend on layer type, so separating them lets any optimizer drive any model. Momentum and Adam keep per-parameter state (velocity; moments and a step counter) that must persist between steps and be checkpointed — an object owns it cleanly." },
      { t: "qa", level: "intermediate", q: "How would you verify that a new layer's backward is correct?", a: "Gradient-check it: in float64, nudge each parameter (and the input) by $\\pm h$ with $h\\approx10^{-5}$, compute central differences of a scalar loss, and compare with the analytic gradients by relative error (below about $10^{-7}$ means correct). For a layer alone, use $L = \\sum g\\odot f(x)$ with a random $g$, so that $\\partial L/\\partial x$ equals `backward(g)`." },
      { t: "qa", level: "intermediate", q: "Adam's first update has magnitude close to the learning rate for every parameter. Why?", a: "At $t=1$, bias correction gives $\\hat m_1 = g$ and $\\hat s_1 = g^2$, so the step is $\\eta\\,g/(|g| + \\epsilon) \\approx \\eta\\,\\mathrm{sign}(g)$. Adam divides gradients by their running RMS, so step sizes are set by $\\eta$ rather than by the raw gradient scale." },
      { t: "qa", level: "advanced", q: "What is missing from this library to support a residual block $y = x + F(x)$, and how do autodiff frameworks solve it?", a: "`Sequential` assumes a chain: each output feeds only the next layer. In a residual block $x$ is used twice, so its gradient is the **sum** of the gradients from both paths. You would need a graph that records every use of a value and adds up gradients from all consumers in reverse topological order. Autodiff frameworks record each operation and its inputs (a tape or graph) during the forward pass and accumulate gradients exactly this way." },
      { t: "qa", level: "advanced", q: "Where does memory go during training of this network, and how does it scale?", a: "Parameters (4810), gradients (4810) and optimizer state (Adam: 9620) do not depend on batch size. Cached activations do: every layer's input and hidden activations for every example, growing linearly with batch size and depth. For large models at large batch sizes activations dominate, which motivates activation checkpointing (recompute instead of store) and mixed precision." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "How many parameters does a `Linear(64, 32)` layer have (weights plus biases)?", answer: 2080, tolerance: 0, explain: "$64\\times32 = 2048$ weights plus $32$ biases $= 2080$." },
        { q: "In `Linear.backward`, what shape is `grads['W']` for a batch of 32, `n_in = 64`, `n_out = 10`?", options: ["(32, 10)", "(64, 10)", "(32, 64)", "(10, 64)"], answer: 1, explain: "$X^\\top G$ is $(64\\times32)(32\\times10) = (64, 10)$ — always the same shape as $W$." },
        { kind: "numeric", q: "Softmax probabilities are $p = (0.7, 0.2, 0.1)$ for one example whose true class is 1 (the second class). What is the gradient of the cross-entropy with respect to the second logit?", answer: -0.8, tolerance: 0.0001, explain: "$\\partial\\mathcal{L}/\\partial z = p - e_y = (0.7, 0.2 - 1, 0.1)$; the second entry is $-0.8$." },
        { q: "Why does `ReLU` store a boolean mask rather than its output?", options: ["To save memory only", "The backward pass only needs to know where the input was positive", "Because the output is not needed by the next layer", "Masks make the forward pass faster"], answer: 1, explain: "ReLU's derivative is 1 where $x > 0$ and 0 elsewhere, so the sign pattern is all the backward pass needs (and a boolean array is 8× smaller than float64)." },
        { kind: "numeric", q: "With momentum $\\beta = 0.9$ and a constant gradient, the velocity approaches $g/(1-\\beta)$. By what factor is the steady-state step larger than plain SGD's with the same learning rate?", answer: 10, tolerance: 0.001, explain: "$1/(1-0.9) = 10$ — which is why the scratch code gives momentum a 10× smaller learning rate than SGD." },
        { q: "A gradient check reports a worst relative error of $3\\times10^{-2}$. What is the most likely conclusion?", options: ["The implementation is correct; float rounding explains it", "There is a bug in some backward pass (or the loss/gradient scaling)", "The learning rate is too high", "The network is overfitting"], answer: 1, explain: "Correct float64 code gives about $10^{-7}$ or less. Errors of $10^{-2}$ point to a wrong formula, a missing sum, or a mean/sum mismatch." },
        { q: "What does `Sequential.backward` do?", options: ["Runs every layer's backward in the same order as forward", "Runs every layer's backward in reverse order, passing each returned gradient to the previous layer", "Updates the weights", "Computes the loss"], answer: 1, explain: "The chain rule runs from the loss back to the input; updating is the optimizer's job." },
        { kind: "numeric", q: "Adam (with $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon \\approx 0$) takes its first step on a parameter whose gradient is $g = 250$. With $\\eta = 0.001$, what is the size of the step?", answer: 0.001, tolerance: 0.00001, explain: "After bias correction $\\hat m_1/\\sqrt{\\hat s_1} = g/|g| = 1$, so the step is $\\eta = 0.001$ whatever $g$ is." },
      ] },
      { t: "exercise", title: "Backward through Linear by hand", prompt: "A `Linear(2, 2)` layer has $W = \\begin{pmatrix}1 & 0\\\\ -1 & 2\\end{pmatrix}$, $b = (0, 0)$. A batch of two inputs $X = \\begin{pmatrix}1 & 2\\\\ 0 & 1\\end{pmatrix}$ gives outputs $Y$, and the upstream gradient is $G = \\begin{pmatrix}1 & 0\\\\ 0 & -1\\end{pmatrix}$. Compute $Y$, $\\partial\\mathcal{L}/\\partial W$, $\\partial\\mathcal{L}/\\partial b$ and $\\partial\\mathcal{L}/\\partial X$.", hint: "Use $X^\\top G$, the column sums of $G$, and $GW^\\top$.", solution: [
        { t: "steps", items: [
          { title: "Forward", tex: String.raw`Y = XW = \begin{pmatrix}1\cdot1 + 2(-1) & 1\cdot0 + 2\cdot2\\ 0\cdot1 + 1(-1) & 0\cdot0 + 1\cdot2\end{pmatrix} = \begin{pmatrix}-1 & 4\\ -1 & 2\end{pmatrix}` },
          { title: "Weight gradient", tex: String.raw`X^\top G = \begin{pmatrix}1 & 0\\ 2 & 1\end{pmatrix}\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix} = \begin{pmatrix}1 & 0\\ 2 & -1\end{pmatrix}` },
          { title: "Bias gradient (column sums of G)", tex: String.raw`\frac{\partial\mathcal{L}}{\partial b} = (1 + 0,\ 0 - 1) = (1,\ -1)` },
          { title: "Input gradient", tex: String.raw`G W^\top = \begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}\begin{pmatrix}1 & -1\\ 0 & 2\end{pmatrix} = \begin{pmatrix}1 & -1\\ 0 & -2\end{pmatrix}` },
        ] },
        { t: "p", text: "Shape check: $\\partial\\mathcal{L}/\\partial W$ is $2\\times2$ like $W$, $\\partial\\mathcal{L}/\\partial X$ is $2\\times2$ like $X$, and $\\partial\\mathcal{L}/\\partial b$ has one entry per output unit." },
      ] },
      { t: "exercise", title: "Add a Tanh layer and certify it", prompt: "Add a `Tanh` layer to the library (same rules as `ReLU`) and verify its backward pass numerically, without a loss: for a random upstream gradient $g$, the scalar $L = \\sum g \\odot \\tanh(x)$ has $\\partial L/\\partial x = $ `backward(g)`. Then explain how you would use it in `mini_nn.py`.", hint: "$\\tanh'(x) = 1 - \\tanh^2(x)$, so caching the output is enough.", solution: [
        { t: "code", lang: "python", title: "tanh_layer.py", code: TANH_SOL, output: TANH_SOL_OUT },
        { t: "p", text: "The analytic and numerical gradients agree to about $10^{-10}$, so the layer is correct. To use it, replace `ReLU()` with `Tanh()` in the `Sequential`, and switch the preceding `Linear` to Xavier initialization ($\\sigma = \\sqrt{1/n_{in}}$ or $\\sqrt{2/(n_{in}+n_{out})}$), which suits tanh rather than ReLU." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "A framework is five pieces: layers (`forward`/`backward`), a container, a loss that seeds the gradient, optimizers with state, and a training loop.",
        "Layer rule: `backward` maps $\\partial\\mathcal{L}/\\partial(\\text{output})$ to $\\partial\\mathcal{L}/\\partial(\\text{input})$, with the shapes of output and input; parameter gradients have parameter shapes.",
        "Linear: $\\partial\\mathcal{L}/\\partial W = X^\\top G$, $\\partial\\mathcal{L}/\\partial b = \\sum_n G_n$, $\\partial\\mathcal{L}/\\partial X = GW^\\top$. ReLU: $G\\odot\\mathbb{1}[x>0]$.",
        "Fused softmax cross-entropy: mean loss via log-sum-exp; gradient $(p - \\text{onehot}(y))/N$.",
        "SGD $\\theta \\leftarrow \\theta - \\eta g$; momentum keeps velocity $v$; Adam keeps $m$, $s$, $t$, and its first step is $\\approx\\eta\\,\\mathrm{sign}(g)$.",
        "Gradient-check every new layer in float64; keep forward and backward paired; update parameters in place.",
        "The 4810-parameter $64\\to64\\to10$ network reaches about 96% test accuracy on digits with any of the three optimizers.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[PyTorch: Tensors to Distributed Training](/learn/m16/pytorch) — the industrial version of all this: autograd replaces hand-written backward passes, and you gain GPUs, data loaders, mixed precision, checkpointing and multi-GPU training.",
        "[TensorFlow and Keras](/learn/m16/tensorflow-keras) — the other major framework; its `GradientTape` records a backward pass like ours, and its callbacks hook into a training loop like ours.",
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — the scratch code compared three optimizers on one seed; learn how to make such comparisons trustworthy.",
      ] },
    ],
  },
};

export default lesson;
