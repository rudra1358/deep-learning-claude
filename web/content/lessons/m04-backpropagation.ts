import type { Lesson } from "@/lib/types";

const BP_NUMPY = String.raw`import numpy as np
from sklearn.datasets import make_moons

# ---------------- data: two interleaving half-moons (not linearly separable) ----------------
X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
y = y.reshape(-1, 1).astype(float)                 # (N, 1) column of 0/1 labels
X_train, y_train, X_val, y_val = X[:300], y[:300], X[300:], y[300:]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """Fully-connected network: tanh hidden layers, sigmoid output, binary cross-entropy."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        # Xavier init keeps activation variance roughly constant through tanh layers
        self.W = [rng.normal(0, np.sqrt(2 / (i + o)), (o, i)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def forward(self, X):
        self.a = [X]                                   # cache activations for the backward pass
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.a[-1] @ W.T + b
            last = l == len(self.W) - 1
            self.a.append(sigmoid(z) if last else np.tanh(z))
        return self.a[-1]

    @staticmethod
    def loss(p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)                   # avoid log(0)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def backward(self, y):
        """Return (dW, db) lists. Must be called right after forward()."""
        N = y.shape[0]
        dW, db = [None] * len(self.W), [None] * len(self.b)
        # sigmoid + BCE simplify beautifully: dL/dz_out = (p - y) / N
        delta = (self.a[-1] - y) / N                   # (N, n_out)
        for l in reversed(range(len(self.W))):
            dW[l] = delta.T @ self.a[l]                # (n_out, N) @ (N, n_in) -> (n_out, n_in)
            db[l] = delta.sum(axis=0)                  # sum over the batch
            if l > 0:
                da = delta @ self.W[l]                 # gradient wrt previous activation
                delta = da * (1 - self.a[l] ** 2)      # tanh'(z) = 1 - tanh(z)^2
        return dW, db

    def step(self, dW, db, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]


def gradient_check(net, X, y, h=1e-5):
    """Compare every analytic gradient with a central finite difference."""
    net.forward(X)
    dW, _ = net.backward(y)
    worst = 0.0
    for l, W in enumerate(net.W):
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + h; lp = net.loss(net.forward(X), y)
            W[idx] = old - h; lm = net.loss(net.forward(X), y)
            W[idx] = old
            num = (lp - lm) / (2 * h)
            rel = abs(num - dW[l][idx]) / max(1e-12, abs(num) + abs(dW[l][idx]))
            worst = max(worst, rel)
    return worst


net = MLP([2, 16, 16, 1], seed=0)
print(f"gradient check, worst relative error: {gradient_check(net, X_train[:20], y_train[:20]):.2e}")

LR, EPOCHS = 0.5, 2000
for epoch in range(EPOCHS + 1):
    p = net.forward(X_train)
    if epoch % 400 == 0:
        acc = np.mean((p > 0.5) == y_train)
        val_acc = np.mean((net.forward(X_val) > 0.5) == y_val)
        net.forward(X_train)                           # restore the training cache
        print(f"epoch {epoch:4d}  loss {net.loss(p, y_train):.4f}  train acc {acc:.3f}  val acc {val_acc:.3f}")
    dW, db = net.backward(y_train)
    net.step(dW, db, LR)`;

const BP_NUMPY_OUT = String.raw`gradient check, worst relative error: 8.07e-08
epoch    0  loss 0.6411  train acc 0.637  val acc 0.710
epoch  400  loss 0.0914  train acc 0.950  val acc 0.970
epoch  800  loss 0.0771  train acc 0.967  val acc 0.970
epoch 1200  loss 0.0732  train acc 0.967  val acc 0.970
epoch 1600  loss 0.0701  train acc 0.970  val acc 0.970
epoch 2000  loss 0.0673  train acc 0.977  val acc 0.970`;

const BP_TORCH = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# ---- Part 1: autograd reproduces the hand-computed gradients of the worked example ----
x = torch.tensor([1.0, 0.5])
y = torch.tensor([1.0])
W1 = torch.tensor([[0.2, -0.4], [0.7, 0.1]], requires_grad=True)
b1 = torch.tensor([0.1, -0.1], requires_grad=True)
W2 = torch.tensor([[0.6, -0.3]], requires_grad=True)
b2 = torch.tensor([0.05], requires_grad=True)

a1 = torch.sigmoid(W1 @ x + b1)       # PyTorch records every operation in a graph...
a2 = torch.sigmoid(W2 @ a1 + b2)
loss = ((a2 - y) ** 2).sum()
loss.backward()                        # ...and backward() applies the chain rule through it
print("loss :", round(loss.item(), 6))
print("dL/dW2:", W2.grad.numpy().round(6), " dL/db2:", b2.grad.numpy().round(6))
print("dL/dW1:", [[round(v, 6) for v in row] for row in W1.grad.tolist()])
print("dL/db1:", b1.grad.numpy().round(6))

# ---- Part 2: the same moons classifier, trained with autograd + an optimizer ----
torch.manual_seed(0)
X, yv = make_moons(n_samples=400, noise=0.2, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
yv = torch.tensor(yv, dtype=torch.float32).unsqueeze(1)
X_train, y_train, X_val, y_val = X[:300], yv[:300], X[300:], yv[300:]

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()       # sigmoid + BCE fused: numerically stable
opt = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(2001):
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()                    # gradients ACCUMULATE by default — clear them
    loss.backward()                    # backpropagation
    opt.step()                         # W <- W - lr * dL/dW for every parameter
    if epoch % 400 == 0:
        with torch.no_grad():
            val_acc = ((model(X_val) > 0) == y_val).float().mean().item()
        print(f"epoch {epoch:4d}  loss {loss.item():.4f}  val acc {val_acc:.3f}")`;

const BP_TORCH_OUT = String.raw`loss : 0.209881
dL/dW2: [[-0.11941  -0.149442]]  dL/db2: [-0.227457]
dL/dW1: [[-0.034033, -0.017017], [0.015377, 0.007689]]
dL/db1: [-0.034033  0.015377]
epoch    0  loss 0.7170  val acc 0.630
epoch  400  loss 0.0885  val acc 0.960
epoch  800  loss 0.0778  val acc 0.960
epoch 1200  loss 0.0741  val acc 0.960
epoch 1600  loss 0.0722  val acc 0.960
epoch 2000  loss 0.0709  val acc 0.960`;

const BP_KERAS = String.raw`import numpy as np
import tensorflow as tf

x = tf.constant([[1.0, 0.5]])
y = tf.constant([[1.0]])
W1 = tf.Variable([[0.2, 0.7], [-0.4, 0.1]])   # Keras orientation (n_in, n_out) = our W1.T
b1 = tf.Variable([0.1, -0.1])
W2 = tf.Variable([[0.6], [-0.3]])
b2 = tf.Variable([0.05])

with tf.GradientTape() as tape:               # the tape records operations for autodiff
    a1 = tf.sigmoid(x @ W1 + b1)
    a2 = tf.sigmoid(a1 @ W2 + b2)
    loss = tf.reduce_sum((a2 - y) ** 2)

grads = tape.gradient(loss, [W1, b1, W2, b2])  # backpropagation
for name, g in zip(["dW1 (transposed)", "db1", "dW2 (transposed)", "db2"], grads):
    print(name, np.round(g.numpy(), 6))`;

const lesson: Lesson = {
  id: "m04/backpropagation",
  title: "Backpropagation",
  summary: "How a network computes the gradient of its loss with respect to every weight — derived from the chain rule, computed by hand on a real example, verified numerically, and implemented from scratch.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m04/forward-propagation", "m02/derivatives-chain-rule", "m06/gradient-descent"],
  next: ["m04/gradient-problems", "m04/gradient-checking", "m06/gradient-descent"],
  terms: ["backpropagation", "chain-rule", "gradient", "computational-graph", "learning-rate"],
  sections: {
    intro: [
      { t: "p", text: "A network learns by nudging each of its parameters in the direction that reduces the loss. To do that we need the **gradient**: for every weight $w$, the derivative $\\partial \\mathcal{L}/\\partial w$, which says how much the loss would change if $w$ changed slightly. **Backpropagation** (short for *backward propagation of errors*) is the algorithm that computes all of these derivatives." },
      { t: "p", text: "Why is a special algorithm needed? Modern networks have millions to billions of weights. The naive approach — wiggle each weight, re-run the network, see how the loss changes — needs one extra forward pass **per weight**. Backpropagation computes **every** gradient for roughly the cost of **one** extra pass, by applying the chain rule backwards through the network and reusing shared intermediate results. That efficiency is what makes training deep networks possible at all." },
      { t: "p", text: "The idea was developed by several researchers (notably in Paul Werbos's 1974 thesis), and was popularized for neural networks by Rumelhart, Hinton and Williams in their 1986 *Nature* paper \"Learning representations by back-propagating errors\", which showed that it lets hidden layers learn useful internal representations. Today it runs inside every deep learning framework as **reverse-mode automatic differentiation** — the `loss.backward()` you call in PyTorch." },
      { t: "callout", kind: "key", text: "Backpropagation = the chain rule, applied from the loss backwards, layer by layer, storing one quantity per neuron ($\\delta = \\partial\\mathcal{L}/\\partial z$) so nothing is computed twice." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Forward Propagation](/learn/m04/forward-propagation) — this lesson continues its worked example; you need $z = Wa + b$ and $a = f(z)$, and the notation $W^{[l]}, a^{[l]}, z^{[l]}$.",
        "[Derivatives and the chain rule](/learn/m02/derivatives-chain-rule) — in particular the multivariable chain rule: when a variable affects the output through several paths, the contributions **add**.",
        "[Activation functions](/learn/m03/activation-functions) — their derivatives, e.g. $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$ and $\\tanh'(z) = 1-\\tanh^2(z)$.",
        "The gradient descent update $\\theta \\leftarrow \\theta - \\eta\\,\\partial\\mathcal{L}/\\partial\\theta$ — covered fully in [Gradient Descent](/learn/m06/gradient-descent); here we only use it to apply the gradients we compute.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Assigning blame in a relay of decisions", text: "A company's quarterly result (the loss) is bad. The CEO asks each division head: \"how much did your numbers move the result?\" Each division head, in turn, asks each of their team leads how much **they** moved the division's numbers, and multiplies that by how much the division moved the result. The question passes backwards through the organisation; at each level, blame is **multiplied** by how sensitive the next level up was — and if a person reports to two managers, the blame from both is **added**. Nobody needs to re-run the whole quarter to find their share." },
      { t: "p", text: "That is backpropagation. The \"blame\" of a neuron is $\\delta = \\partial\\mathcal{L}/\\partial z$: how much the loss changes per unit change of the neuron's pre-activation. The output neuron's blame comes straight from the loss. Every earlier neuron's blame is the sum, over the neurons it feeds, of (their blame × the weight connecting them), scaled by how responsive this neuron's activation is at its current input ($f'(z)$)." },
      { t: "p", text: "Once a neuron knows its blame, the gradient of each of its incoming weights is just **blame × the input that flowed along that weight**: a weight carrying a large signal into a highly-blamed neuron gets a large gradient; a weight whose input was zero gets no gradient at all." },
    ],
    mechanism: [
      { t: "p", text: "One full training step on one example (or one mini-batch):" },
      { t: "steps", items: [
        { title: "Forward pass, caching values", text: "Compute and **store** $z^{[l]}$ and $a^{[l]}$ for every layer, then the loss $\\mathcal{L}$." },
        { title: "Start at the loss", text: "Compute $\\partial\\mathcal{L}/\\partial a^{[L]}$, the derivative of the loss with respect to the network's output. For squared error $(a-y)^2$ it is $2(a-y)$." },
        { title: "Output-layer error", text: "Multiply element-wise by the derivative of the output activation, evaluated at the cached $z^{[L]}$.", tex: String.raw`\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \odot f'^{[L]}\!\left(z^{[L]}\right)` },
        { title: "Gradients of this layer's parameters", text: "Weight gradient = this layer's $\\delta$ times the previous layer's activations (an outer product). Bias gradient = $\\delta$ itself.", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W^{[l]}} = \delta^{[l]}\, a^{[l-1]\top}, \qquad \frac{\partial\mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}` },
        { title: "Send the error one layer back", text: "Push $\\delta$ backwards through the **transposed** weight matrix (this sums over all paths), then multiply by the local activation derivative.", tex: String.raw`\delta^{[l-1]} = \left(W^{[l]\top}\delta^{[l]}\right) \odot f'^{[l-1]}\!\left(z^{[l-1]}\right)` },
        { title: "Repeat down to layer 1", text: "Alternate steps 4 and 5 until every layer has its gradients. The input layer needs no $\\delta$ (inputs are not parameters)." },
        { title: "Update all parameters", text: "Apply gradient descent with learning rate $\\eta$. Only now do the weights change — every gradient was computed with the *old* weights.", tex: String.raw`W^{[l]} \leftarrow W^{[l]} - \eta\,\frac{\partial\mathcal{L}}{\partial W^{[l]}}, \qquad b^{[l]} \leftarrow b^{[l]} - \eta\,\frac{\partial\mathcal{L}}{\partial b^{[l]}}` },
      ] },
      { t: "callout", kind: "warning", text: "Do not update a layer's weights before propagating $\\delta$ through them. Step 5 uses $W^{[l]}$ — if you've already modified it, the earlier layers receive gradients for a network that never existed." },
    ],
    math: [
      { t: "h", text: "Why the recurrence is true: deriving it from the chain rule" },
      { t: "p", text: "Fix a layer $l$. The weight $W^{[l]}_{jk}$ affects the loss **only** through the pre-activation $z^{[l]}_j = \\sum_k W^{[l]}_{jk}a^{[l-1]}_k + b^{[l]}_j$. By the chain rule:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W^{[l]}_{jk}} = \frac{\partial\mathcal{L}}{\partial z^{[l]}_j}\cdot\frac{\partial z^{[l]}_j}{\partial W^{[l]}_{jk}} = \delta^{[l]}_j\, a^{[l-1]}_k, \qquad \text{where } \delta^{[l]}_j \equiv \frac{\partial\mathcal{L}}{\partial z^{[l]}_j}` },
      { t: "p", text: "Similarly $\\partial z^{[l]}_j/\\partial b^{[l]}_j = 1$, so $\\partial\\mathcal{L}/\\partial b^{[l]}_j = \\delta^{[l]}_j$. Everything therefore reduces to computing $\\delta$ for every neuron." },
      { t: "p", text: "**Output layer.** $z^{[L]}_j$ affects the loss only through $a^{[L]}_j = f(z^{[L]}_j)$:" },
      { t: "math", tex: String.raw`\delta^{[L]}_j = \frac{\partial\mathcal{L}}{\partial a^{[L]}_j}\, f'\!\left(z^{[L]}_j\right)` },
      { t: "p", text: "**Hidden layers.** A hidden pre-activation $z^{[l-1]}_k$ affects the loss through its activation $a^{[l-1]}_k$, which feeds **every** neuron $j$ of layer $l$. The multivariable chain rule sums over all these paths:" },
      { t: "math", tex: String.raw`\delta^{[l-1]}_k = \sum_{j} \underbrace{\frac{\partial\mathcal{L}}{\partial z^{[l]}_j}}_{\delta^{[l]}_j}\, \underbrace{\frac{\partial z^{[l]}_j}{\partial a^{[l-1]}_k}}_{W^{[l]}_{jk}}\, \underbrace{\frac{\partial a^{[l-1]}_k}{\partial z^{[l-1]}_k}}_{f'(z^{[l-1]}_k)} = \Big(\sum_j W^{[l]}_{jk}\,\delta^{[l]}_j\Big) f'\!\left(z^{[l-1]}_k\right)` },
      { t: "p", text: "The sum $\\sum_j W^{[l]}_{jk}\\delta^{[l]}_j$ is entry $k$ of $W^{[l]\\top}\\delta^{[l]}$ — which is why the backward pass multiplies by the **transpose** of each weight matrix. These four equations are the whole algorithm:" },
      { t: "math", tex: String.raw`\begin{aligned} \delta^{[L]} &= \nabla_{a^{[L]}}\mathcal{L} \odot f'(z^{[L]}) && \text{(BP1)}\\ \delta^{[l-1]} &= (W^{[l]\top}\delta^{[l]}) \odot f'(z^{[l-1]}) && \text{(BP2)}\\ \partial\mathcal{L}/\partial W^{[l]} &= \delta^{[l]} a^{[l-1]\top} && \text{(BP3)}\\ \partial\mathcal{L}/\partial b^{[l]} &= \delta^{[l]} && \text{(BP4)} \end{aligned}` },
      { t: "h", text: "Worked example — continuing the forward-propagation lesson" },
      { t: "p", text: "Network 2–2–1, sigmoid everywhere, $x = (1.0, 0.5)$, $y = 1$, $\\mathcal{L} = (a^{[2]} - y)^2$. From the forward pass: $a^{[1]} = (0.524979,\\ 0.657010)$, $a^{[2]} = 0.541873$, $\\mathcal{L} = 0.209881$. Recall $\\sigma'(z) = \\sigma(z)(1-\\sigma(z)) = a(1-a)$, so we can compute it from the cached activations." },
      { t: "steps", items: [
        { title: "Loss derivative", tex: String.raw`\frac{\partial\mathcal{L}}{\partial a^{[2]}} = 2(a^{[2]} - y) = 2(0.541873 - 1) = -0.916254` },
        { title: "Output δ (BP1)", tex: String.raw`\delta^{[2]} = -0.916254 \times \underbrace{0.541873(1-0.541873)}_{\sigma'(z^{[2]}) = 0.248247} = -0.227457` },
        { title: "Output-layer gradients (BP3, BP4)", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W^{[2]}} = \delta^{[2]}a^{[1]\top} = -0.227457\,(0.524979,\ 0.657010) = (-0.119410,\ -0.149442),\quad \frac{\partial\mathcal{L}}{\partial b^{[2]}} = -0.227457` },
        { title: "Error reaching the hidden layer", tex: String.raw`W^{[2]\top}\delta^{[2]} = \begin{pmatrix}0.6\\-0.3\end{pmatrix}(-0.227457) = \begin{pmatrix}-0.136474\\ 0.068237\end{pmatrix}` },
        { title: "Hidden δ (BP2)", text: "$\\sigma'(z^{[1]}) = a^{[1]}(1-a^{[1]}) = (0.249376,\\ 0.225348)$", tex: String.raw`\delta^{[1]} = \begin{pmatrix}-0.136474 \times 0.249376\\ 0.068237 \times 0.225348\end{pmatrix} = \begin{pmatrix}-0.034033\\ 0.015377\end{pmatrix}` },
        { title: "Hidden-layer gradients", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W^{[1]}} = \delta^{[1]}x^\top = \begin{pmatrix}-0.034033\\0.015377\end{pmatrix}(1.0,\ 0.5) = \begin{pmatrix}-0.034033 & -0.017017\\ 0.015377 & 0.007689\end{pmatrix},\quad \frac{\partial\mathcal{L}}{\partial b^{[1]}} = \begin{pmatrix}-0.034033\\0.015377\end{pmatrix}` },
        { title: "Update with η = 0.5", tex: String.raw`W^{[2]} \leftarrow (0.6,\,-0.3) - 0.5(-0.119410,\,-0.149442) = (0.659705,\ -0.225279),\quad b^{[2]} \leftarrow 0.163729` },
        { title: "…and the hidden layer", tex: String.raw`W^{[1]} \leftarrow \begin{pmatrix}0.217017 & -0.391492\\ 0.692311 & 0.096156\end{pmatrix},\quad b^{[1]} \leftarrow (0.117017,\ -0.107689)` },
        { title: "Check: run the forward pass again", tex: String.raw`\hat y_{\text{new}} = 0.591271,\qquad \mathcal{L}_{\text{new}} = 0.167059 < 0.209881 \ \checkmark` },
      ] },
      { t: "p", text: "Read the signs: $\\delta^{[2]} < 0$ means increasing $z^{[2]}$ would *decrease* the loss (the prediction is too low), so the update increases both output weights' contributions. Hidden neuron 2 is connected through a *negative* weight ($-0.3$), so its $\\delta$ is positive: making it fire **less** helps. Every one of these numbers is reproduced to six decimals by PyTorch's autograd in the Framework section below." },
      { t: "h", text: "Vectorized form for a mini-batch" },
      { t: "p", text: "With $N$ examples stored as rows ($A^{[l]} \\in \\mathbb{R}^{N\\times n_l}$) and the loss averaged over the batch, $\\Delta^{[l]} \\in \\mathbb{R}^{N\\times n_l}$ holds one row of $\\delta$ per example:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial W^{[l]}} = \Delta^{[l]\top} A^{[l-1]}, \qquad \frac{\partial\mathcal{L}}{\partial b^{[l]}} = \sum_{n=1}^{N} \Delta^{[l]}_{n,:}, \qquad \Delta^{[l-1]} = \big(\Delta^{[l]} W^{[l]}\big) \odot f'(Z^{[l-1]})`, caption: "The matrix product $\\Delta^\\top A$ sums the per-example outer products — gradients of a batch are the sum (or mean) of per-example gradients." },
      { t: "p", text: "A particularly clean special case: with a **sigmoid output and binary cross-entropy** loss (or softmax + categorical cross-entropy), the $f'$ factor cancels against the loss derivative and $\\Delta^{[L]} = (A^{[L]} - Y)/N$. The from-scratch code below uses exactly this." },
      { t: "h", text: "Cost: why backprop is efficient" },
      { t: "p", text: "The backward pass performs two matrix products per layer ($\\Delta^\\top A$ for the weight gradient and $\\Delta W$ for the next $\\delta$), each the same size as the forward product. So one backward pass costs about **twice** a forward pass, and a full training step about **three** forward passes — regardless of the number of parameters. The finite-difference alternative needs $P+1$ forward passes for $P$ parameters: for a model with $10^8$ parameters that is a factor of tens of millions." },
      { t: "h", text: "Connection to vanishing and exploding gradients" },
      { t: "p", text: "Unrolling BP2 across many layers multiplies together many factors $W^{[l]\\top}$ and $f'(z^{[l]})$. Since $\\sigma'(z) \\le 0.25$, ten sigmoid layers can shrink $\\delta$ by up to $0.25^{10} \\approx 9.5\\times10^{-7}$ — the **vanishing gradient** problem; large weights can instead make it grow exponentially. This is studied in [Vanishing and Exploding Gradients](/learn/m04/gradient-problems)." },
    ],
    visual: [
      { t: "p", text: "Press **Step** until the loss appears, then keep stepping: the backward phase begins at the output and moves left. Clay-coloured dots carry gradients backwards; each neuron shows its $\\delta$ underneath, and the edges entering the active neuron show their weight gradients." },
      { t: "sim", id: "network", caption: "Click any connection to compare its backprop gradient with a numerical central-difference gradient (a gradient check)." },
      { t: "list", items: [
        "Reproduce the worked example (set the weights and biases from the Math section) and confirm every $\\delta$ and gradient matches.",
        "At the update step, read the list of old → new parameters and the loss before and after. Press **Apply update** a few times — or tick *keep training* — and watch the loss history fall.",
        "Set the learning rate to 3 and train: the loss may oscillate or increase. The gradient is only a *local* direction; too large a step overshoots.",
        "Use ReLU in a hidden layer and make one neuron's pre-activation negative: its $\\delta$ is exactly 0, and so is the gradient of every weight into it. That neuron receives no learning signal for this input.",
        "Build a deep network (4 hidden layers, sigmoid) and compare the size of $\\delta$ in the first hidden layer with the output layer — vanishing gradients in action.",
      ] },
    ],
    scratch: [
      { t: "p", text: "A complete, vectorized implementation: forward pass with caching, backward pass using BP1–BP4 for a batch, a **gradient check** against finite differences on every weight, and full-batch training on the non-linearly-separable *two moons* dataset. It runs in a few seconds on a CPU (and in your browser with the Run button)." },
      { t: "code", lang: "python", title: "backprop_from_scratch.py", code: BP_NUMPY, output: BP_NUMPY_OUT },
      { t: "p", text: "How the important parts map to the maths:" },
      { t: "list", items: [
        "`delta = (self.a[-1] - y) / N` is $\\Delta^{[L]}$ for sigmoid + mean BCE (the $f'$ cancels).",
        "`dW[l] = delta.T @ self.a[l]` is BP3 summed over the batch: $(n_{out}\\times N)(N\\times n_{in})$. Because `self.a[0]` is the input, `self.a[l]` is the *previous* layer's activation for weight matrix `l`.",
        "`da = delta @ self.W[l]` followed by `* (1 - self.a[l] ** 2)` is BP2 with $\\tanh'(z) = 1-\\tanh^2(z)$, computed from the cached activation instead of $z$.",
        "`gradient_check` perturbs each weight by $\\pm h$ ($h = 10^{-5}$ balances truncation error against floating-point cancellation) and reports the worst relative error. Values around $10^{-7}$ or smaller mean the backward pass is correct; $10^{-2}$ would indicate a bug.",
        "The learning rate 0.5 and 2000 epochs were chosen so the run converges in seconds; the validation accuracy (97%) shows the network generalizes beyond the 300 training points.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Part 1 recomputes the worked example with **autograd**: tensors created with `requires_grad=True` are leaves of a recorded computational graph, and `loss.backward()` runs reverse-mode differentiation through it, filling each `.grad`. The numbers match the hand derivation. Part 2 trains the same moons classifier with `nn.Module`, a fused loss and an optimizer." },
          { t: "code", lang: "python", title: "backprop_pytorch.py", code: BP_TORCH, output: BP_TORCH_OUT },
          { t: "callout", kind: "warning", title: "Gradients accumulate", text: "`backward()` **adds** into `.grad`. If you forget `opt.zero_grad()` each step, gradients from all previous steps pile up and training diverges. (Accumulation is deliberate: it lets you simulate a larger batch by calling `backward()` on several small batches before one `step()`.)" },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "TensorFlow records operations executed inside a `tf.GradientTape` context and differentiates them with `tape.gradient`. Variables use Keras's $(n_{in}, n_{out})$ orientation, so the printed weight gradients are the transposes of ours." },
          { t: "code", lang: "python", title: "backprop_keras.py (not executed in CI)", code: BP_KERAS, output: "Prints dW1 (transposed) ≈ [[-0.034033 0.015377] [-0.017017 0.007689]], db1 ≈ [-0.034033 0.015377], dW2 (transposed) ≈ [[-0.11941] [-0.149442]], db2 ≈ [-0.227457] — the same values as the hand derivation." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Variant", "What differs", "When it's used"], rows: [
        ["Reverse-mode autodiff (backprop)", "One backward sweep gives the gradient of one scalar w.r.t. all inputs", "Training any network — always, since the loss is a scalar"],
        ["Forward-mode autodiff", "Propagates derivatives forward alongside values; one pass per input direction", "Few inputs, many outputs (e.g. Jacobian-vector products, some scientific computing)"],
        ["Backpropagation through time (BPTT)", "Backprop on an RNN unrolled over time steps; shared weights accumulate gradients across steps", "Recurrent networks — see [BPTT](/learn/m09/bptt)"],
        ["Truncated BPTT", "Backpropagate only a fixed number of steps", "Long sequences, to bound memory"],
        ["Gradient checkpointing", "Store fewer activations, recompute them during the backward pass", "Large models limited by memory"],
        ["Mixed-precision backprop", "Forward/backward in fp16/bf16 with loss scaling, master weights in fp32", "Modern GPU training for speed and memory"],
        ["Straight-through estimator", "Uses a surrogate gradient for non-differentiable ops (e.g. rounding)", "Quantization-aware training, discrete latents"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**All neural network training** — image classifiers, language models, speech recognizers, recommender systems — uses backpropagation (as reverse-mode autodiff) to compute gradients.",
        "**Fine-tuning and transfer learning**: backprop through a pretrained model, often with most layers frozen (their parameters simply receive no update).",
        "**Saliency maps and Grad-CAM**: backpropagating to the *input* image ($\\partial\\text{score}/\\partial x$) shows which pixels influenced a prediction — see [Interpretability](/learn/m19/interpretability).",
        "**Adversarial examples**: the same input gradient, followed in the direction that *increases* the loss, produces small perturbations that fool a model (FGSM).",
        "**Differentiable programming** beyond neural networks: physics simulators, rendering and optimization layers trained end-to-end with the same machinery.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Exact and efficient**: gradients are exact (up to floating point), at ~2× the cost of a forward pass, independent of the number of parameters.",
        "**Fully automatic in practice**: you only write the forward computation; frameworks derive the backward pass.",
        "**Memory-hungry**: requires storing intermediate activations from the forward pass.",
        "**Needs differentiability**: discrete operations (argmax, sampling, rounding) have zero or undefined gradients and need workarounds (reparameterization, straight-through, REINFORCE).",
        "**Local information only**: the gradient tells you the best direction for an infinitesimal step; it can't escape poor regions, and its usefulness depends on the optimizer and learning rate.",
        "**Gradient pathologies in deep or recurrent networks**: products of many Jacobians vanish or explode, motivating careful initialization, normalization, residual connections and gated units.",
        "**Not biologically plausible** in its standard form (it needs symmetric backward weights and a separate backward phase), which is a research topic, not a practical limitation.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Evaluating $f'$ at the wrong value.** The activation derivative belongs to the *same* layer as the $\\delta$ being computed, evaluated at that layer's cached $z$ (or, for sigmoid/tanh, its cached $a$). Using the previous layer's activation is a classic off-by-one bug — a gradient check catches it immediately.",
        "**Forgetting to sum over the batch for the bias gradient**, or summing the weight gradient while *averaging* the loss (or vice versa). The gradient's scale must match the loss's scale, otherwise the effective learning rate silently changes with batch size.",
        "**Updating weights in place before backpropagating through them** (see the warning in the mechanism section).",
        "**Not zeroing gradients in PyTorch** between steps — gradients accumulate.",
        "**Applying sigmoid in the model and then using a with-logits loss** — the gradient becomes $\\sigma'$-scaled twice and much smaller than it should be.",
        "**Initialising all weights to the same value** (e.g. zero). Every neuron in a layer then computes the same output and receives the same gradient, so they stay identical forever — symmetry is never broken.",
        "**Believing a falling loss proves the gradients are right.** A slightly wrong gradient often still points downhill. Always gradient-check a new layer implementation.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What does backpropagation compute, and what does it not do?", a: "It computes the gradient of the loss with respect to every parameter (and intermediate value). It does **not** update the weights — that is the optimizer's job (gradient descent, Adam, …), which uses the gradients backprop produced." },
      { t: "qa", level: "beginner", q: "Why is it called *back*propagation?", a: "The error signal $\\delta$ is computed first at the output layer and then propagated backwards, layer by layer, towards the input, because each layer's $\\delta$ depends on the next layer's $\\delta$ via the chain rule." },
      { t: "qa", level: "beginner", q: "What is $\\delta$ for a neuron?", a: "$\\delta = \\partial\\mathcal{L}/\\partial z$, the derivative of the loss with respect to the neuron's pre-activation. The gradient of an incoming weight is $\\delta$ times the activation entering along that weight, and the bias gradient equals $\\delta$." },
      { t: "qa", level: "intermediate", q: "Why does the backward pass use $W^\\top$?", a: "Neuron $k$ of layer $l-1$ influences every neuron $j$ of layer $l$ through weight $W_{jk}$. The chain rule sums those contributions: $\\sum_j W_{jk}\\delta_j$, which is the $k$-th entry of $W^\\top\\delta$." },
      { t: "qa", level: "intermediate", q: "Derive the gradient of sigmoid + binary cross-entropy with respect to the logit.", a: "$\\mathcal{L} = -[y\\log p + (1-y)\\log(1-p)]$, $p = \\sigma(z)$. $\\partial\\mathcal{L}/\\partial p = -y/p + (1-y)/(1-p) = (p-y)/(p(1-p))$ and $\\partial p/\\partial z = p(1-p)$, so $\\partial\\mathcal{L}/\\partial z = p - y$. The saturating $\\sigma'$ cancels, which is why this pairing trains well even when the prediction is confidently wrong." },
      { t: "qa", level: "intermediate", q: "How do you verify a backpropagation implementation?", a: "Gradient checking: for each parameter compute $(\\mathcal{L}(\\theta+h) - \\mathcal{L}(\\theta-h))/2h$ with $h\\approx10^{-5}$ in float64 and compare to the analytic gradient using relative error $|g_a - g_n|/(|g_a|+|g_n|)$. Errors below ~$10^{-7}$ are good; above ~$10^{-3}$ indicate a bug. Disable dropout and other randomness while checking, and beware of kinks (ReLU at 0)." },
      { t: "qa", level: "intermediate", q: "What is the computational cost of backprop relative to the forward pass?", a: "Roughly 2× the forward pass (two matrix products per layer instead of one), so a training step is about 3× a forward pass. Memory is the larger cost: all activations needed for the backward pass must be kept." },
      { t: "qa", level: "advanced", q: "Why is reverse-mode differentiation preferred over forward-mode for neural networks?", a: "Forward mode computes a Jacobian-vector product per pass — one pass per input dimension to get a full gradient. Reverse mode computes a vector-Jacobian product per pass — one pass per *output*. Training has millions of inputs (parameters) and one scalar output (the loss), so reverse mode gets the whole gradient in one pass." },
      { t: "qa", level: "advanced", q: "What happens to gradients in a network with a ReLU whose input is negative for every training example?", a: "Its output and derivative are 0 for all inputs, so $\\delta = 0$ and every incoming weight and its bias get zero gradient: the neuron is \"dead\" and can never recover under gradient descent. Causes include large learning rates or poor initialization; remedies include Leaky ReLU/GELU, lower learning rates and proper (He) initialization." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "What is the maximum value of the sigmoid derivative $\\sigma'(z)$?", answer: 0.25, tolerance: 0.0001, explain: "$\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$ is maximised at $z=0$ where $\\sigma = 0.5$: $0.5\\times0.5 = 0.25$." },
        { kind: "numeric", q: "A neuron has $\\partial\\mathcal{L}/\\partial a = 3$ and $f'(z) = 0.5$. One of its incoming weights carries activation $a_{prev} = 2$. What is that weight's gradient?", answer: 3, tolerance: 0.0001, explain: "$\\delta = 3\\times0.5 = 1.5$; gradient $= \\delta\\,a_{prev} = 1.5\\times2 = 3$." },
        { kind: "numeric", q: "A sigmoid output with binary cross-entropy predicts $p = 0.8$ for a negative example ($y=0$). What is $\\partial\\mathcal{L}/\\partial z$ at the output?", answer: 0.8, tolerance: 0.0001, explain: "For sigmoid + BCE, $\\partial\\mathcal{L}/\\partial z = p - y = 0.8$." },
        { q: "A hidden ReLU neuron has pre-activation $z = -2$ for an example. What is the gradient of its incoming weights for that example?", options: ["It depends on the learning rate", "Exactly zero", "Equal to the input values", "Equal to the bias"], answer: 1, explain: "$\\mathrm{ReLU}'(z) = 0$ for $z<0$, so $\\delta = 0$ and all incoming-weight gradients ($\\delta\\cdot a_{prev}$) are zero." },
        { q: "In BP2, $\\delta^{[l-1]} = (W^{[l]\\top}\\delta^{[l]})\\odot f'(z^{[l-1]})$, which weights are used?", options: ["The weights after the gradient-descent update", "The weights before the update (those used in the forward pass)", "Randomly re-initialised weights", "The weights of layer $l-1$"], answer: 1, explain: "Gradients must be computed for the network that produced the loss — the forward-pass weights. Updates happen only after all gradients are computed." },
        { q: "Why is backpropagation far cheaper than estimating each gradient by finite differences?", options: ["It uses lower precision", "It reuses each neuron's δ for all gradients that depend on it, needing about one backward pass in total", "It skips the hidden layers", "It only computes gradients for the output layer"], answer: 1, explain: "Finite differences need one extra forward pass per parameter; backprop computes all gradients in one sweep by sharing intermediate results." },
        { kind: "numeric", q: "Approximately how much smaller can $\\delta$ become after passing back through 10 sigmoid layers if the weights are 1? Give $0.25^{10}$ in scientific notation's mantissa×1e-7 form — i.e. enter $0.25^{10}\\times10^{7}$ (2 decimals).", answer: 9.54, tolerance: 0.02, explain: "$0.25^{10} = 9.54\\times10^{-7}$ — gradients reaching early layers can be a millionth of those at the output: the vanishing gradient problem." },
      ] },
      { t: "exercise", title: "One neuron, by hand", prompt: "A single neuron computes $a = \\sigma(wx + b)$ with loss $\\mathcal{L} = (a - y)^2$. With $x = 2$, $w = 0.5$, $b = -1$, $y = 1$: compute $\\partial\\mathcal{L}/\\partial w$ and $\\partial\\mathcal{L}/\\partial b$, then the updated $w$ and $b$ for $\\eta = 0.1$.", hint: "First compute $z$ — it comes out to a convenient value.", solution: [
        { t: "steps", items: [
          { title: "Forward", tex: String.raw`z = 0.5\cdot2 - 1 = 0,\quad a = \sigma(0) = 0.5,\quad \mathcal{L} = (0.5-1)^2 = 0.25` },
          { title: "Backward", tex: String.raw`\frac{\partial\mathcal{L}}{\partial a} = 2(0.5-1) = -1,\quad \sigma'(0) = 0.25,\quad \delta = -0.25` },
          { title: "Gradients", tex: String.raw`\frac{\partial\mathcal{L}}{\partial w} = \delta x = -0.5,\qquad \frac{\partial\mathcal{L}}{\partial b} = \delta = -0.25` },
          { title: "Update", tex: String.raw`w \leftarrow 0.5 - 0.1(-0.5) = 0.55,\qquad b \leftarrow -1 - 0.1(-0.25) = -0.975` },
        ] },
      ] },
      { t: "exercise", title: "Switch the hidden activation to ReLU", prompt: "Modify `backprop_from_scratch.py` to use **ReLU** in the hidden layers (and He initialization, std $\\sqrt{2/n_{in}}$). Which two lines must change in `forward`/`backward`? Run the gradient check — why might it occasionally report a larger error with ReLU?", hint: "ReLU's derivative is 1 where the pre-activation is positive and 0 elsewhere. You can test positivity using the cached *activation*.", solution: [
        { t: "list", items: [
          "In `forward`: `self.a.append(sigmoid(z) if last else np.maximum(0.0, z))`.",
          "In `backward`: replace `(1 - self.a[l] ** 2)` with `(self.a[l] > 0)` — the ReLU derivative computed from the cached activation.",
          "In `__init__`: use `np.sqrt(2 / i)` as the standard deviation (He initialization).",
        ] },
        { t: "p", text: "ReLU has a **kink** at 0. If some pre-activation lies within $h$ of zero, the finite difference straddles the kink and averages the slopes 0 and 1, while the analytic gradient uses one side — producing a spuriously large relative error for that parameter. This is a limitation of the check, not a bug; it disappears if you re-check with a different seed or smaller $h$." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Backpropagation computes $\\partial\\mathcal{L}/\\partial\\theta$ for **every** parameter in one backward sweep, at ~2× the cost of a forward pass.",
        "Central quantity: $\\delta^{[l]} = \\partial\\mathcal{L}/\\partial z^{[l]}$, one per neuron.",
        "Output: $\\delta^{[L]} = \\nabla_a\\mathcal{L}\\odot f'(z^{[L]})$. Hidden: $\\delta^{[l-1]} = (W^{[l]\\top}\\delta^{[l]})\\odot f'(z^{[l-1]})$.",
        "Parameter gradients: $\\partial\\mathcal{L}/\\partial W^{[l]} = \\delta^{[l]}a^{[l-1]\\top}$, $\\partial\\mathcal{L}/\\partial b^{[l]} = \\delta^{[l]}$; for a batch, $\\Delta^\\top A$ and a sum over rows.",
        "Sigmoid + BCE (and softmax + CE) give the simple output error $p - y$.",
        "Compute all gradients with the forward-pass weights, then update. Verify new code with a gradient check.",
        "Repeated multiplication by $W^\\top$ and $f'$ causes vanishing/exploding gradients in deep networks.",
      ] },
    ],
    next: [
      { t: "p", text: "Three directions follow naturally:" },
      { t: "list", items: [
        "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — what goes wrong when BP2 is applied through many layers, and the remedies.",
        "[Numerical Gradient Checking](/learn/m04/gradient-checking) — the debugging technique used above, in depth.",
        "[Gradient Descent and Optimizers](/learn/m06/gradient-descent) — now that you can compute gradients, learn how to use them well: learning rates, momentum, Adam.",
      ] },
    ],
  },
};

export default lesson;
