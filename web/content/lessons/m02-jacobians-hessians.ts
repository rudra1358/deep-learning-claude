import type { Lesson } from "@/lib/types";

const AD_CODE = String.raw`import math
import random


class Value:
    """A scalar that remembers how it was computed, so gradients can flow back through it."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0                 # dL/d(self), filled in by backward()
        self._parents = parents         # the Values this one was computed from
        self._op = op                   # for printing the graph
        self._backward = lambda: None   # pushes self.grad to the parents (one VJP)

    # ---------------- primitive operations: value + local derivative ----------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():                # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():                # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):               # k is a plain number (constant exponent)
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():                # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():                # tanh'(a) = 1 - tanh(a)^2, reusing the forward value
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward():                # exp'(a) = exp(a)
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # ---------------- derived operations (built from the primitives) ----------------
    def __neg__(self):            return self * -1.0
    def __sub__(self, other):     return self + (-other if isinstance(other, Value) else -other)
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)
    def __radd__(self, other):    return self + other
    def __rmul__(self, other):    return self * other
    def __rsub__(self, other):    return Value(other) - self

    # ---------------- reverse-mode sweep ----------------
    def backward(self):
        order, seen = [], set()
        def build(v):                   # depth-first post-order = topological order
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    build(p)
                order.append(v)
        build(self)
        for v in order:
            v.grad = 0.0                # fresh gradients for this call
        self.grad = 1.0                 # dL/dL = 1 (the seed)
        for v in reversed(order):       # every node is processed after all its consumers
            v._backward()
        return order


# ---------- test 1: a small expression with a shared sub-expression ----------
a, b = Value(2.0), Value(-3.0)
c = a * b                               # -6
d = (c + a ** 2).tanh()                 # a is used twice: gradients must add up
e = (d * b).exp() + c / a
order = e.backward()
print(f"e = {e.data:.6f}   nodes in graph = {len(order)}")
print(f"de/da = {a.grad:.6f}   de/db = {b.grad:.6f}")

# ---------- test 2: a 3-2-1 tanh network with squared-error loss ----------
random.seed(0)
x = [Value(v) for v in (0.5, -1.2, 2.0)]
W1 = [[Value(random.uniform(-1, 1)) for _ in range(3)] for _ in range(2)]
b1 = [Value(0.1), Value(-0.1)]
W2 = [Value(random.uniform(-1, 1)) for _ in range(2)]
h = [(sum((w * xi for w, xi in zip(row, x)), Value(0.0)) + bj).tanh() for row, bj in zip(W1, b1)]
y_hat = sum((w * hj for w, hj in zip(W2, h)), Value(0.0))
loss = (y_hat - 1.0) ** 2
loss.backward()
print(f"loss = {loss.data:.6f}")
print("dL/dW1 =", [[round(w.grad, 6) for w in row] for row in W1])
print("dL/dW2 =", [round(w.grad, 6) for w in W2])

# ---------- referee: PyTorch autograd on exactly the same computations ----------
import torch
A = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
Bt = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
C = A * Bt
E = (torch.tanh(C + A ** 2) * Bt).exp() + C / A
E.backward()
print(f"torch : e = {E.item():.6f}   de/da = {A.grad.item():.6f}   de/db = {Bt.grad.item():.6f}")

X = torch.tensor([0.5, -1.2, 2.0], dtype=torch.float64)
TW1 = torch.tensor([[w.data for w in row] for row in W1], dtype=torch.float64, requires_grad=True)
Tb1 = torch.tensor([0.1, -0.1], dtype=torch.float64)
TW2 = torch.tensor([w.data for w in W2], dtype=torch.float64, requires_grad=True)
TL = (TW2 @ torch.tanh(TW1 @ X + Tb1) - 1.0) ** 2
TL.backward()
err = max((TW1.grad - torch.tensor([[w.grad for w in r] for r in W1], dtype=torch.float64)).abs().max().item(),
          (TW2.grad - torch.tensor([w.grad for w in W2], dtype=torch.float64)).abs().max().item())
print(f"torch : loss = {TL.item():.6f}   max |grad difference| = {err:.1e}")`;

const AD_OUT = String.raw`e = 15.030824   nodes in graph = 11
de/da = -3.821678   de/db = -24.025567
loss = 1.018903
dL/dW1 = [[-0.453747, 1.088994, -1.81499], [0.237369, -0.569686, 0.949477]]
dL/dW2 = [0.920792, 1.280326]
torch : e = 15.030824   de/da = -3.821678   de/db = -24.025567
torch : loss = 1.018903   max |grad difference| = 0.0e+00`;

const NP_CODE = String.raw`import numpy as np

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(0)


def numerical_jacobian(f, x, h=1e-6):
    """J[i, j] = d f_i / d x_j by central differences: one column per input coordinate."""
    fx = f(x)
    J = np.zeros((fx.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        J[:, j] = (f(x + e) - f(x - e)) / (2 * h)
    return J


# ---------- 1. the worked example: f : R^2 -> R^3 ----------
f = lambda x: np.array([x[0] ** 2 * x[1], 5 * x[0] + np.sin(x[1]), x[0] * x[1]])
x = np.array([1.0, 2.0])
J_exact = np.array([[2 * x[0] * x[1], x[0] ** 2], [5.0, np.cos(x[1])], [x[1], x[0]]])
print("Jacobian shape (m, n):", J_exact.shape)
print("max |numerical - analytic| :", f"{np.abs(numerical_jacobian(f, x) - J_exact).max():.1e}")

# ---------- 2. Jacobians of three common layers ----------
n_in, n_out = 4, 3
W, b = rng.normal(size=(n_out, n_in)), rng.normal(size=n_out)
v = rng.normal(size=n_in)
print("linear layer   : J == W ?", np.allclose(numerical_jacobian(lambda u: W @ u + b, v), W))
print("tanh (diagonal): J == diag(1 - tanh^2) ?",
      np.allclose(numerical_jacobian(np.tanh, v), np.diag(1 - np.tanh(v) ** 2)))

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.array([1.0, 2.0, 0.5])
p = softmax(z)
J_soft = np.diag(p) - np.outer(p, p)
print("softmax        : J == diag(p) - p p^T ?", np.allclose(numerical_jacobian(softmax, z), J_soft))
print(f"softmax Jacobian, largest |row sum|: {np.abs(J_soft.sum(axis=1)).max():.1e}")

# ---------- 3. VJP and JVP never need the full Jacobian ----------
u = rng.normal(size=n_out)                  # an "upstream gradient" dL/dy
s = rng.normal(size=n_in)                   # a "tangent" direction for forward mode
a = np.tanh(W @ v + b)                      # layer y = tanh(Wv + b)
J_layer = np.diag(1 - a ** 2) @ W           # full (n_out x n_in) Jacobian, for checking only
vjp = W.T @ (u * (1 - a ** 2))              # reverse mode: u^T J  without forming J
jvp = (1 - a ** 2) * (W @ s)                # forward mode: J s    without forming J
print("VJP matches u^T J:", np.allclose(vjp, u @ J_layer), "  JVP matches J s:", np.allclose(jvp, J_layer @ s))

# ---------- 4. Hessian, eigenvalues and the type of a critical point ----------
g = lambda q: np.array([3 * q[0] ** 2 - 3 * q[1], 3 * q[1] ** 2 - 3 * q[0]])   # grad of x^3 + y^3 - 3xy
for point in ([1.0, 1.0], [0.0, 0.0]):
    q = np.array(point)
    H = numerical_jacobian(g, q)            # the Hessian is the Jacobian of the gradient
    lam = np.linalg.eigvalsh((H + H.T) / 2)
    kind = "minimum" if lam.min() > 0 else "maximum" if lam.max() < 0 else "saddle"
    print(f"point {point}: |grad| = {np.linalg.norm(g(q)):.1e}  eigenvalues {lam.round(4)}  -> {kind}")

# ---------- 5. Hessian-vector product from two gradients (no Hessian stored) ----------
q, w_dir, eps = np.array([1.2, 0.9]), np.array([0.6, -0.8]), 1e-5
hvp_fd = (g(q + eps * w_dir) - g(q - eps * w_dir)) / (2 * eps)
H_exact = np.array([[6 * q[0], -3.0], [-3.0, 6 * q[1]]])
print("Hv by finite differences of the gradient:", hvp_fd, " exact:", H_exact @ w_dir)

# ---------- 6. Newton's method: fast near a minimum, but attracted to saddles ----------
for start in ([1.5, 0.8], [0.2, 0.1]):
    q = np.array(start)
    for _ in range(5):
        H = np.array([[6 * q[0], -3.0], [-3.0, 6 * q[1]]])
        q = q - np.linalg.solve(H, g(q))    # solve H d = g instead of inverting H
    print(f"Newton from {start} -> {q.round(6) + 0.0}")`;

const NP_OUT = String.raw`Jacobian shape (m, n): (3, 2)
max |numerical - analytic| : 1.9e-10
linear layer   : J == W ? True
tanh (diagonal): J == diag(1 - tanh^2) ? True
softmax        : J == diag(p) - p p^T ? True
softmax Jacobian, largest |row sum|: 2.1e-17
VJP matches u^T J: True   JVP matches J s: True
point [1.0, 1.0]: |grad| = 0.0e+00  eigenvalues [3. 9.]  -> minimum
point [0.0, 0.0]: |grad| = 0.0e+00  eigenvalues [-3.  3.]  -> saddle
Hv by finite differences of the gradient: [ 6.72 -6.12]  exact: [ 6.72 -6.12]
Newton from [1.5, 0.8] -> [1. 1.]
Newton from [0.2, 0.1] -> [0. 0.]`;

const TORCH_CODE = String.raw`import torch
from torch.autograd.functional import jacobian, hessian, hvp
from torch.func import jacrev, jacfwd, jvp, vjp, grad

torch.set_printoptions(precision=6, sci_mode=False)
torch.manual_seed(0)

# ---------- 1. full Jacobians: torch.autograd.functional and torch.func ----------
def f(x):                                    # R^2 -> R^3, the lesson's worked example
    return torch.stack([x[0] ** 2 * x[1], 5 * x[0] + torch.sin(x[1]), x[0] * x[1]])

x = torch.tensor([1.0, 2.0], dtype=torch.float64)
print("jacobian(f, x) =\n", jacobian(f, x))
print("jacrev == jacfwd:", torch.allclose(jacrev(f)(x), jacfwd(f)(x)))

z = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
p = torch.softmax(z, dim=0)
print("softmax Jacobian == diag(p) - p p^T:",
      torch.allclose(jacrev(lambda t: torch.softmax(t, 0))(z), torch.diag(p) - torch.outer(p, p)))

# ---------- 2. products instead of matrices: vjp (reverse) and jvp (forward) ----------
layer = torch.nn.Linear(4, 3).double()
v = torch.randn(4, dtype=torch.float64)
g_fn = lambda u: torch.tanh(layer(u))
J = jacrev(g_fn)(v)                          # (3, 4), materialised only to check the products
u = torch.randn(3, dtype=torch.float64)      # upstream gradient
s = torch.randn(4, dtype=torch.float64)      # tangent direction
_, vjp_fn = vjp(g_fn, v)
(uJ,) = vjp_fn(u)                            # one reverse pass
_, Js = jvp(g_fn, (v,), (s,))                # one forward pass
print("vjp == u^T J:", torch.allclose(uJ, u @ J), "  jvp == J s:", torch.allclose(Js, J @ s))

# ---------- 3. Hessian and its eigenvalues ----------
def h(q):                                    # x^3 + y^3 - 3xy
    return q[0] ** 3 + q[1] ** 3 - 3 * q[0] * q[1]

for pt in ([1.0, 1.0], [0.0, 0.0]):
    H = hessian(h, torch.tensor(pt, dtype=torch.float64))
    print(f"Hessian at {pt}: {H.tolist()}  eigenvalues {torch.linalg.eigvalsh(H).tolist()}")

# ---------- 4. Hessian-vector products three ways ----------
q = torch.tensor([1.2, 0.9], dtype=torch.float64)
w = torch.tensor([0.6, -0.8], dtype=torch.float64)

_, hv1 = hvp(h, q, w)                                   # (a) functional API

qq = q.clone().requires_grad_(True)                     # (b) double backprop by hand
(g1,) = torch.autograd.grad(h(qq), qq, create_graph=True)   # keep the graph of the gradient
(hv2,) = torch.autograd.grad(g1 @ w, qq)                # d/dq (grad . w) = H w

_, hv3 = jvp(grad(h), (q,), (w,))                       # (c) forward-over-reverse
print("Hv (functional, double backprop, forward-over-reverse):", *[[round(c, 6) for c in t.tolist()] for t in (hv1, hv2, hv3)])

# ---------- 5. cost check: HVP of a real network loss without forming the Hessian ----------
net = torch.nn.Sequential(torch.nn.Linear(10, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)).double()
X, Y = torch.randn(256, 10, dtype=torch.float64), torch.randn(256, 1, dtype=torch.float64)
params = list(net.parameters())
n_params = sum(t.numel() for t in params)
loss = torch.nn.functional.mse_loss(net(X), Y)
grads = torch.autograd.grad(loss, params, create_graph=True)
vec = [torch.randn_like(t) for t in params]
Hv = torch.autograd.grad(sum((g * v_).sum() for g, v_ in zip(grads, vec)), params)
print(f"parameters: {n_params}, Hessian would hold {n_params ** 2} numbers; "
      f"Hv holds {sum(t.numel() for t in Hv)}")`;

const TORCH_OUT = String.raw`jacobian(f, x) =
 tensor([[ 4.000000,  1.000000],
        [ 5.000000, -0.416147],
        [ 2.000000,  1.000000]], dtype=torch.float64)
jacrev == jacfwd: True
softmax Jacobian == diag(p) - p p^T: True
vjp == u^T J: True   jvp == J s: True
Hessian at [1.0, 1.0]: [[6.0, -3.0], [-3.0, 6.0]]  eigenvalues [3.0, 9.0]
Hessian at [0.0, 0.0]: [[0.0, -3.0], [-3.0, 0.0]]  eigenvalues [-3.0, 3.0]
Hv (functional, double backprop, forward-over-reverse): [6.72, -6.12] [6.72, -6.12] [6.72, -6.12]
parameters: 769, Hessian would hold 591361 numbers; Hv holds 769`;

const EX_CODE = String.raw`import torch
from torch.func import jacrev, jvp, vjp

torch.manual_seed(0)
n, m = 5, 2                                   # 5 inputs, 2 outputs
A = torch.randn(m, n, dtype=torch.float64)
f = lambda x: torch.tanh(A @ x) ** 2          # R^5 -> R^2
x = torch.randn(n, dtype=torch.float64)

# reverse mode: one VJP per OUTPUT gives one ROW of J
_, pullback = vjp(f, x)
rows = [pullback(torch.eye(m, dtype=torch.float64)[i])[0] for i in range(m)]
J_rev = torch.stack(rows)

# forward mode: one JVP per INPUT gives one COLUMN of J
cols = [jvp(f, (x,), (torch.eye(n, dtype=torch.float64)[j],))[1] for j in range(n)]
J_fwd = torch.stack(cols, dim=1)

print("passes needed: reverse =", len(rows), " forward =", len(cols))
print("both equal jacrev:", torch.allclose(J_rev, jacrev(f)(x)), torch.allclose(J_fwd, jacrev(f)(x)))
print("J shape:", tuple(J_rev.shape))`;

const EX_OUT = String.raw`passes needed: reverse = 2  forward = 5
both equal jacrev: True True
J shape: (2, 5)`;

const TF_CODE = String.raw`import tensorflow as tf

# ---------- full Jacobian with a GradientTape ----------
x = tf.Variable([1.0, 2.0], dtype=tf.float64)
with tf.GradientTape() as tape:
    y = tf.stack([x[0] ** 2 * x[1], 5 * x[0] + tf.sin(x[1]), x[0] * x[1]])
print("Jacobian:\n", tape.jacobian(y, x).numpy())

# ---------- Hessian with nested tapes ----------
q = tf.Variable([1.2, 0.9], dtype=tf.float64)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        f = q[0] ** 3 + q[1] ** 3 - 3 * q[0] * q[1]
    g = inner.gradient(f, q)            # gradient, recorded by the outer tape
print("Hessian:\n", outer.jacobian(g, q).numpy())

# ---------- Hessian-vector product: d/dq (g . v) ----------
v = tf.constant([0.6, -0.8], dtype=tf.float64)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        f = q[0] ** 3 + q[1] ** 3 - 3 * q[0] * q[1]
    g = inner.gradient(f, q)
    gv = tf.reduce_sum(g * v)
print("Hv (nested tapes):", outer.gradient(gv, q).numpy())

# ---------- forward-over-reverse HVP with ForwardAccumulator ----------
with tf.autodiff.ForwardAccumulator(primals=q, tangents=v) as acc:
    with tf.GradientTape() as tape:
        f = q[0] ** 3 + q[1] ** 3 - 3 * q[0] * q[1]
    g = tape.gradient(f, q)
print("Hv (forward-over-reverse):", acc.jvp(g).numpy())

# ---------- a JVP (forward mode) of the R^2 -> R^3 function ----------
with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.constant([1.0, 0.0], dtype=tf.float64)) as acc:
    y = tf.stack([x[0] ** 2 * x[1], 5 * x[0] + tf.sin(x[1]), x[0] * x[1]])
print("JVP J @ (1, 0):", acc.jvp(y).numpy())`;


const lesson: Lesson = {
  id: "m02/jacobians-hessians",
  title: "Jacobians, Hessians and Computational Graphs",
  summary:
    "Derivatives of vector-valued functions (the Jacobian), the multivariable chain rule as a product of Jacobians, why autodiff computes vector-Jacobian products instead of Jacobians, a reverse-mode autodiff engine built from scratch, and second derivatives (the Hessian): curvature, critical points, Hessian-vector products and Newton's method.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m02/derivatives-chain-rule", "m02/vectors-matrices", "m02/eigen-svd"],
  next: ["m02/optimization-math", "m04/backpropagation", "m04/gradient-checking", "m06/momentum-adaptive"],
  terms: ["jacobian", "hessian", "vector-jacobian-product", "jacobian-vector-product", "computational-graph", "automatic-differentiation", "topological-order", "hessian-vector-product", "newtons-method", "saddle-point"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`The [derivatives lesson](/learn/m02/derivatives-chain-rule) dealt with functions that output one number: the gradient $\nabla L$ of a scalar loss. But almost every piece of a neural network maps a **vector to a vector**: a linear layer maps $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$, an activation maps $\mathbb{R}^{m}$ to $\mathbb{R}^{m}$, softmax maps logits to probabilities. The derivative of such a function is a matrix, the **Jacobian**, which records how every output responds to every input.`,
      },
      {
        t: "p",
        text: String.raw`With Jacobians, the chain rule becomes a clean statement: the derivative of a composition is the **product of the Jacobians** of its pieces. Backpropagation is that product, evaluated from the loss end. The practical twist is that nobody ever forms those Jacobian matrices — they would be enormous. Frameworks instead compute **vector-Jacobian products** (reverse mode) or **Jacobian-vector products** (forward mode), organised on a **computational graph** that is traversed in topological order. In this lesson you will build such an engine yourself in about 80 lines of Python and check it against PyTorch.`,
      },
      {
        t: "p",
        text: String.raw`The second half moves to **second derivatives**. The **Hessian** of a scalar function describes its curvature: whether a critical point is a minimum, a maximum or a saddle, how large a learning rate can be, and how Newton's method could take a perfectly-scaled step. Full Hessians are as impractical as full Jacobians, but **Hessian-vector products** can be computed for the price of about two gradients — the basis of curvature analysis, sharpness measures and Hessian-free optimisation.`,
      },
      {
        t: "callout",
        kind: "key",
        title: "The two ideas to keep",
        text: String.raw`(1) $J_{g\circ f}(x) = J_g(f(x))\,J_f(x)$, and reverse-mode autodiff evaluates $\bar{y}^\top J_K J_{K-1}\cdots J_1$ from the left, one vector-matrix product at a time, so a scalar loss gets its whole gradient in one backward pass. (2) The Hessian $H = \nabla^2 f$ is the Jacobian of the gradient; its eigenvalues are the curvatures along its eigenvectors, and $Hv$ costs about as much as a gradient.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Derivatives, Partial Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — partial derivatives, the gradient vector, the scalar and multivariable chain rule ("contributions along different paths add"), and the scalar computational graph. This lesson generalises all of these to vectors.`,
          String.raw`[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — matrix-vector and matrix-matrix products, the transpose, outer products $uv^\top$, and the cost of multiplying an $m\times k$ by a $k\times n$ matrix ($mkn$ multiply-adds).`,
          String.raw`[Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd) — a symmetric matrix has real eigenvalues and orthonormal eigenvectors ($H = Q\Lambda Q^\top$); we use this to read curvature from a Hessian.`,
          String.raw`It helps to have seen [Backpropagation](/learn/m04/backpropagation): its rule $\delta^{[l-1]} = (W^{[l]\top}\delta^{[l]})\odot f'(z^{[l-1]})$ is exactly a vector-Jacobian product, as we will show.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A mixing desk",
        text: "Picture a sound engineer's mixing desk with $n$ input faders and $m$ output speakers. Nudge one fader slightly and every speaker's volume changes a little, each by its own amount. Write those sensitivities in a table — one row per speaker, one column per fader — and you have the Jacobian. Asking \"if I nudge the faders in this particular pattern, how do the speakers change?\" is a Jacobian-vector product: you only need one test, not the whole table. Asking \"the audience wants this speaker louder and that one quieter — which faders should I move?\" is a vector-Jacobian product, the question backpropagation answers.",
      },
      {
        t: "p",
        text: String.raw`**Jacobian = best linear approximation.** Zoom far enough into any smooth function and it looks linear. For a scalar function that linear piece is a slope; for $f:\mathbb{R}^n\to\mathbb{R}^m$ it is a matrix $J$, so that $f(x+\Delta) \approx f(x) + J\Delta$ for small $\Delta$. Column $j$ of $J$ says where the outputs move when you push input $j$; row $i$ is the gradient of output $i$.`,
      },
      {
        t: "p",
        text: String.raw`**Chain rule = composing linear maps.** If $f$ is locally the matrix $J_f$ and $g$ is locally $J_g$, then running $f$ then $g$ is locally $J_g J_f$. A deep network is a long chain of such local linear maps, and its sensitivity is their product.`,
      },
      {
        t: "p",
        text: String.raw`**Why products, not matrices.** The loss at the end of the chain is a single number. Its sensitivity to anything is a single row vector. Multiplying a row vector by one Jacobian after another (right to left through the network) only ever needs vector-matrix products — cheap — while multiplying the Jacobians together first would create huge matrix-matrix products that are then thrown away.`,
      },
      {
        t: "p",
        text: String.raw`**Hessian = how the slope itself changes.** On a hiking map, the gradient tells you which way is downhill; the Hessian tells you whether you are in a bowl (curves up in every direction), on a dome (curves down in every direction) or on a mountain pass (up along the ridge, down across it — a **saddle**). Its eigenvectors are the principal directions of curvature and its eigenvalues are how sharply the surface bends along them.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "How an autodiff engine computes a gradient" },
      {
        t: "steps",
        items: [
          {
            title: "Record the graph during the forward pass",
            text: "Every primitive operation ($+$, $\\times$, `tanh`, `exp`, power, matrix product, …) creates an output node that stores its value, references to its input nodes, and a small function that knows the operation's local derivative. This is the **computational graph**: a directed acyclic graph (DAG) whose edges point from inputs to outputs. PyTorch builds it on the fly as your Python code runs (*define-by-run*); `torch.compile`, JAX's `jit` and TensorFlow's `tf.function` trace it once into a static graph.",
          },
          {
            title: "Order the nodes topologically",
            text: "A **topological order** lists every node after all of the nodes it depends on. A depth-first search that appends a node only after visiting all of its inputs (post-order) produces one. Reversed, it lists every node *before* the nodes it depends on — exactly the order the backward sweep needs.",
          },
          {
            title: "Seed the output",
            text: String.raw`Set $\bar{L} = \partial L/\partial L = 1$ and every other node's gradient to $0$. The notation $\bar{v} = \partial L/\partial v$ (the **adjoint** of $v$) is standard in the autodiff literature.`,
          },
          {
            title: "Sweep backwards, pushing gradients to inputs",
            text: String.raw`Visit nodes in reverse topological order. For a node $y = \phi(x_1,\dots,x_k)$, add $\bar{y}\,\partial\phi/\partial x_i$ to each $\bar{x}_i$. For vector nodes this is a vector-Jacobian product: $\bar{x} \mathrel{+}= J_\phi^\top \bar{y}$. Because the reverse topological order visits a node only after **all** of its consumers, $\bar{y}$ is complete by the time it is used.`,
          },
          {
            title: "Accumulate, never overwrite",
            text: "A value used in several places (a weight shared across time steps, a residual connection, $x$ in $x\\cdot x$) receives one contribution per use. They must be **added** — this is the multivariable chain rule (\"sum over paths\") in code form, and the reason PyTorch's `.grad` accumulates.",
          },
          {
            title: "Read off leaf gradients",
            text: String.raw`After the sweep, every leaf (parameter or input) holds $\partial L/\partial(\text{leaf})$. The whole sweep costs a small constant multiple of the forward pass, independent of the number of leaves.`,
          },
        ],
      },
      { t: "h", text: "Tracing one example by hand" },
      {
        t: "p",
        text: String.raw`Take $a = 2$, $b = -3$ and $e = \exp\!\big(\tanh(ab + a^2)\,b\big) + ab/a$. The variable $a$ enters through three paths (inside $ab$, inside $a^2$, and as the divisor), $b$ through two, and $c = ab$ is itself used twice. The forward pass, one primitive per node:`,
      },
      {
        t: "table",
        head: ["Node", "Operation", "Value"],
        rows: [
          ["$c$", "$a\\cdot b$", "$-6$"],
          ["$q$", "$a^2$", "$4$"],
          ["$s$", "$c + q$", "$-2$"],
          ["$d$", "$\\tanh(s)$", "$-0.964028$"],
          ["$m$", "$d\\cdot b$", "$2.892083$"],
          ["$x$", "$\\exp(m)$", "$18.030824$"],
          ["$i$", "$a^{-1}$", "$0.5$"],
          ["$r$", "$c\\cdot i$", "$-3$"],
          ["$e$", "$x + r$", "$15.030824$"],
        ],
      },
      {
        t: "p",
        text: String.raw`Backward, in reverse topological order (each line uses only adjoints that are already complete):`,
      },
      {
        t: "steps",
        items: [
          { title: "Seed and the final sum", tex: String.raw`\bar e = 1,\qquad \bar x = \bar e\cdot 1 = 1,\qquad \bar r = \bar e \cdot 1 = 1` },
          { title: "Through the division branch", tex: String.raw`\bar c \mathrel{+}= \bar r\, i = 0.5,\qquad \bar i = \bar r\, c = -6` },
          { title: "Through exp and the product with b", tex: String.raw`\bar m = \bar x\, e^{m} = 18.030824,\qquad \bar d = \bar m\, b = -54.092472,\qquad \bar b \mathrel{+}= \bar m\, d = -17.382212` },
          { title: "Through tanh", tex: String.raw`\bar s = \bar d\,(1 - d^2) = -54.092472 \times 0.070651 = -3.821678` },
          { title: "Through the sum s = c + q", tex: String.raw`\bar c \mathrel{+}= \bar s \;\Rightarrow\; \bar c = 0.5 - 3.821678 = -3.321678,\qquad \bar q = -3.821678` },
          { title: "Contributions to a (three paths)", tex: String.raw`\bar a = \underbrace{2a\,\bar q}_{-15.286711} + \underbrace{(-a^{-2})\,\bar i}_{+1.5} + \underbrace{b\,\bar c}_{+9.965033} = -3.821678` },
          { title: "Contributions to b (two paths)", tex: String.raw`\bar b = \underbrace{-17.382212}_{\text{via } m} + \underbrace{a\,\bar c}_{-6.643356} = -24.025567` },
        ],
      },
      {
        t: "p",
        text: "Notice that $\\bar c$ had to wait for **both** of its consumers ($r$ and $s$) before it could be propagated into $a$ and $b$ — that is precisely what the reverse topological order guarantees. The engine in the Code section reproduces $\\partial e/\\partial a = -3.821678$ and $\\partial e/\\partial b = -24.025567$, and so does PyTorch.",
      },
      { t: "h", text: "What frameworks store for each primitive" },
      {
        t: "table",
        head: ["Primitive $y = \\phi(x)$", "Saved in forward", "Backward rule (VJP) $\\bar x \\mathrel{+}= J^\\top\\bar y$"],
        rows: [
          ["$y = Wx + b$", "$x$ (and $W$)", "$\\bar x = W^\\top\\bar y$, $\\bar W = \\bar y\\, x^\\top$, $\\bar b = \\bar y$"],
          ["$y = \\sigma(x)$ element-wise", "$y$", "$\\bar x = \\bar y \\odot y \\odot (1-y)$"],
          ["$y = \\tanh(x)$", "$y$", "$\\bar x = \\bar y \\odot (1-y^2)$"],
          ["$y = \\mathrm{ReLU}(x)$", "mask $x > 0$", "$\\bar x = \\bar y \\odot \\mathbb{1}[x>0]$"],
          ["$p = \\mathrm{softmax}(z)$", "$p$", "$\\bar z = p \\odot (\\bar p - (p^\\top\\bar p)\\mathbf{1})$"],
          ["$y = x_1 \\odot x_2$", "$x_1, x_2$", "$\\bar x_1 = \\bar y\\odot x_2$, $\\bar x_2 = \\bar y \\odot x_1$"],
          ["$y = \\sum_i x_i$", "nothing", "$\\bar x = \\bar y\\,\\mathbf{1}$ (broadcast)"],
        ],
      },
      {
        t: "callout",
        kind: "note",
        text: "Every rule in the right-hand column is computed **without** building $J$. The softmax rule, for instance, is $O(n)$ work, whereas the explicit $n\\times n$ Jacobian would cost $O(n^2)$ memory and time.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Functions from ℝⁿ to ℝᵐ and the Jacobian" },
      {
        t: "p",
        text: String.raw`A vector-valued function $f:\mathbb{R}^n\to\mathbb{R}^m$ is a stack of $m$ scalar functions $f(x) = (f_1(x),\dots,f_m(x))$, each depending on all $n$ inputs $x = (x_1,\dots,x_n)$. Its **Jacobian** at $x$ is the $m\times n$ matrix of all first partial derivatives:`,
      },
      {
        t: "math",
        tex: String.raw`J_f(x) = \frac{\partial f}{\partial x} = \begin{pmatrix} \dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m\times n}, \qquad (J_f)_{ij} = \frac{\partial f_i}{\partial x_j}`,
        caption: "Shape convention (numerator layout): rows = outputs, columns = inputs. This is the convention of torch.autograd.functional.jacobian and jax.jacobian.",
      },
      {
        t: "p",
        text: String.raw`Row $i$ is $\nabla f_i(x)^\top$, the transposed gradient of output $i$. For a scalar function ($m = 1$) the Jacobian is the $1\times n$ row $\nabla f^\top$, so the gradient is the transpose of the Jacobian. The defining property is the first-order approximation`,
      },
      {
        t: "math",
        tex: String.raw`f(x + \Delta) = f(x) + J_f(x)\,\Delta + o(\lVert\Delta\rVert)`,
        caption: String.raw`$o(\lVert\Delta\rVert)$ means an error that vanishes faster than $\lVert\Delta\rVert$ as $\Delta \to 0$.`,
      },
      { t: "h", text: "Worked example" },
      {
        t: "p",
        text: String.raw`Let $f(x_1, x_2) = \big(x_1^2 x_2,\; 5x_1 + \sin x_2,\; x_1x_2\big)$, so $n = 2$, $m = 3$, and evaluate at $x = (1, 2)$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Value", tex: String.raw`f(1,2) = (1\cdot 2,\; 5 + \sin 2,\; 2) = (2,\; 5.909297,\; 2)` },
          { title: "Partial derivatives, row by row", tex: String.raw`J_f = \begin{pmatrix} 2x_1x_2 & x_1^2 \\ 5 & \cos x_2 \\ x_2 & x_1 \end{pmatrix} = \begin{pmatrix} 4 & 1 \\ 5 & -0.416147 \\ 2 & 1 \end{pmatrix}` },
          { title: "Use it: predict f at (1.01, 1.98), i.e. Δ = (0.01, −0.02)", tex: String.raw`f(x) + J\Delta = (2 + 0.04 - 0.02,\; 5.909297 + 0.05 + 0.008323,\; 2 + 0.02 - 0.02) = (2.02,\; 5.967620,\; 2.0)` },
          { title: "Compare with the true value", tex: String.raw`f(1.01, 1.98) = (2.019798,\; 5.967438,\; 1.9998)\quad\Rightarrow\quad \text{errors} \approx (-2.0,\,-1.8,\,-2.0)\times 10^{-4} = O(\lVert\Delta\rVert^2)` },
        ],
      },
      { t: "h", text: "2. Jacobians of the layers you meet every day" },
      {
        t: "p",
        text: String.raw`**Linear (affine) layer** $y = Wx + b$ with $W\in\mathbb{R}^{m\times n}$: $y_i = \sum_k W_{ik}x_k + b_i$, so $\partial y_i/\partial x_j = W_{ij}$. The Jacobian with respect to the input is the weight matrix itself: $J = W$.`,
      },
      {
        t: "p",
        text: String.raw`**Element-wise activation** $y = \phi(z)$ with $y_i = \phi(z_i)$: output $i$ depends only on input $i$, so all off-diagonal entries vanish and $J = \operatorname{diag}(\phi'(z_1),\dots,\phi'(z_m))$. Multiplying by a diagonal matrix is the same as an element-wise product, which is why backprop writes $\odot\,\phi'(z)$.`,
      },
      {
        t: "p",
        text: String.raw`**Softmax** $p_i = e^{z_i}/\sum_k e^{z_k}$. Differentiate the quotient. For $i = j$: $\partial p_i/\partial z_i = p_i - p_i^2$. For $i \ne j$: $\partial p_i/\partial z_j = -e^{z_i}e^{z_j}/(\sum_k e^{z_k})^2 = -p_ip_j$. Together, with $\delta_{ij}$ the Kronecker delta ($1$ if $i=j$, else $0$):`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j) \qquad\Longleftrightarrow\qquad J_{\text{softmax}} = \operatorname{diag}(p) - pp^\top`,
      },
      {
        t: "p",
        text: String.raw`For $z = (1, 2, 0.5)$: $p = (0.231224,\ 0.628532,\ 0.140244)$ and`,
      },
      {
        t: "math",
        tex: String.raw`J = \begin{pmatrix} 0.177759 & -0.145332 & -0.032428 \\ -0.145332 & 0.233480 & -0.088148 \\ -0.032428 & -0.088148 & 0.120576 \end{pmatrix}, \qquad \text{eigenvalues } (0,\ 0.168127,\ 0.363688)`,
      },
      {
        t: "p",
        text: String.raw`Three properties are worth remembering. It is **symmetric**. Its rows sum to zero, because $\sum_i p_i = 1$ cannot change when $z$ changes, so $J\mathbf{1} = 0$ — adding the same constant to every logit leaves the probabilities unchanged (eigenvalue $0$ along $\mathbf{1}$). And it is positive semidefinite, which is the curvature fact behind the convexity of cross-entropy in the logits. Combined with cross-entropy $L = -\log p_y$, whose gradient is $\bar p = -e_y/p_y$, the vector-Jacobian product collapses to the famous $\bar z = p - e_y$ (with $e_y$ the one-hot target).`,
      },
      { t: "h", text: "3. The multivariable chain rule as a product of Jacobians" },
      {
        t: "p",
        text: String.raw`Let $f:\mathbb{R}^n\to\mathbb{R}^k$ and $g:\mathbb{R}^k\to\mathbb{R}^m$, and $h = g\circ f$. Composing the two linear approximations, $g(f(x+\Delta)) \approx g(f(x) + J_f\Delta) \approx g(f(x)) + J_g J_f\Delta$, so`,
      },
      {
        t: "math",
        tex: String.raw`J_{g\circ f}(x) = J_g\big(f(x)\big)\,J_f(x), \qquad (m\times n) = (m\times k)(k\times n)`,
      },
      {
        t: "p",
        text: String.raw`Written out entry by entry, $\partial h_i/\partial x_j = \sum_{r=1}^{k} (\partial g_i/\partial y_r)(\partial y_r/\partial x_j)$ — the "sum over paths" rule from the [derivatives lesson](/learn/m02/derivatives-chain-rule), now packaged as a matrix product. For a network $L = \ell(f_K(\cdots f_1(x)))$ with intermediate values $h_0 = x, h_1, \dots, h_K$:`,
      },
      {
        t: "math",
        tex: String.raw`\nabla_x L^\top = \underbrace{\nabla_{h_K}\ell^\top}_{1\times d_K}\; J_{f_K}\; J_{f_{K-1}} \cdots J_{f_1}`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** One tanh layer $h = \tanh(Wx + b)$ followed by $L = \tfrac12\lVert h - y\rVert^2$, with $W = \begin{pmatrix}0.5 & -1\\ 2 & 0.3\end{pmatrix}$, $b = (0.1, -0.2)$, $x = (1, 0.5)$, $y = (0, 1)$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Forward", tex: String.raw`z = Wx + b = (0.1,\ 1.95),\quad h = \tanh z = (0.099668,\ 0.960319),\quad L = 0.005754` },
          { title: "Jacobian of the layer: diag(1 − h²) W", tex: String.raw`J_h = \begin{pmatrix}0.990066 & 0\\ 0 & 0.077787\end{pmatrix}\begin{pmatrix}0.5 & -1\\ 2 & 0.3\end{pmatrix} = \begin{pmatrix}0.495033 & -0.990066\\ 0.155573 & 0.023336\end{pmatrix}` },
          { title: "Gradient of the loss w.r.t. h", tex: String.raw`\nabla_h L = h - y = (0.099668,\ -0.039681)` },
          { title: "Chain rule as a row vector times the Jacobian", tex: String.raw`\nabla_x L^\top = (0.099668,\ -0.039681)\,J_h = (0.043166,\ -0.099604)` },
          { title: "The same result without forming J_h (a VJP)", tex: String.raw`W^\top\big((h-y)\odot(1-h^2)\big) = W^\top(0.098678,\ -0.003087) = (0.043166,\ -0.099604)` },
        ],
      },
      {
        t: "p",
        text: String.raw`The last line is backprop's $\delta = (h - y)\odot\tanh'(z)$ followed by $W^\top\delta$: [backpropagation](/learn/m04/backpropagation) is a sequence of vector-Jacobian products. The weight gradient is $\partial L/\partial W = \delta\,x^\top = \begin{pmatrix}0.098678 & 0.049339\\ -0.003087 & -0.001543\end{pmatrix}$.`,
      },
      { t: "h", text: "4. VJPs, JVPs and why full Jacobians are never materialised" },
      {
        t: "p",
        text: String.raw`Two products can be computed without ever forming $J\in\mathbb{R}^{m\times n}$:`,
      },
      {
        t: "math",
        tex: String.raw`\textbf{JVP (forward mode):}\; \dot y = J\,\dot x \in \mathbb{R}^m \qquad\qquad \textbf{VJP (reverse mode):}\; \bar x^\top = \bar y^\top J \in \mathbb{R}^{1\times n}`,
        caption: String.raw`$\dot x$ is a tangent (an input direction); $\bar y$ is a cotangent (an output weighting, e.g. $\partial L/\partial y$).`,
      },
      {
        t: "p",
        text: String.raw`**Forward mode** carries a tangent alongside every value: when $y = \phi(x)$ is computed, also compute $\dot y = J_\phi\dot x$ using the same local rules. One forward-mode pass gives one JVP, i.e. one **column** of $J$ (with $\dot x = e_j$), at a small constant multiple of the cost of evaluating $f$, and extra memory proportional only to the current values. **Reverse mode** runs $f$ forward, stores the intermediates, then pushes a cotangent backwards: one pass gives one VJP, i.e. one **row** of $J$ (with $\bar y = e_i$), also at a small constant multiple of the cost of $f$, but with memory proportional to all the stored intermediates.`,
      },
      {
        t: "table",
        head: ["Goal", "Forward mode", "Reverse mode"],
        rows: [
          ["Full Jacobian of $f:\\mathbb{R}^n\\to\\mathbb{R}^m$", "$n$ passes (one per column)", "$m$ passes (one per row)"],
          ["Gradient of a scalar loss ($m = 1$), $n = 10^8$ parameters", "$10^8$ passes", "**1 pass**"],
          ["Directional derivative $J\\dot x$", "1 pass", "$m$ passes"],
          ["Extra memory", "$O(1)$ per live value", "all intermediates needed by the backward rules"],
        ],
      },
      {
        t: "p",
        text: String.raw`**Matrix-chain view.** Consider $K$ layers of width $d$ and a scalar loss. Evaluating $\bar y^\top J_K\cdots J_1$ from the left costs $K$ vector-matrix products: about $Kd^2$ multiply-adds. Building the Jacobian product from the right, $J_K(\cdots(J_2J_1))$, costs $K$ matrix-matrix products: about $Kd^3$. For $d = 1000$ and $K = 10$ that is $10^7$ versus $10^{10}$ — a factor of $d$. The order of evaluation is the whole difference between forward and reverse mode.`,
      },
      {
        t: "p",
        text: String.raw`**Memory.** A linear layer applied to a batch of $B = 64$ vectors with $n = m = 4096$ maps $\mathbb{R}^{B\times n}\to\mathbb{R}^{B\times m}$. Its full Jacobian has $(Bm)(Bn) = 68{,}719{,}476{,}736$ entries — about $275$ GB in float32 — and almost all of them are zero (example $i$'s output does not depend on example $j$'s input). The VJP needs one tensor of the input's shape: $Bn = 262{,}144$ numbers. Likewise an element-wise activation has a diagonal Jacobian: storing $n^2$ numbers to represent $n$ useful ones would be absurd. This is why every framework implements, for each primitive, a VJP rule (and, increasingly, a JVP rule) rather than a Jacobian.`,
      },
      { t: "h", text: "5. Computational graphs and topological order" },
      {
        t: "p",
        text: String.raw`A **computational graph** is a DAG $G = (V, E)$ whose nodes are intermediate values and whose edge $u\to v$ means "$v$ was computed from $u$". The forward pass evaluates nodes in a topological order $v_1,\dots,v_N$ (every edge goes from an earlier to a later node). Reverse mode computes the adjoints $\bar v = \partial L/\partial v$ by`,
      },
      {
        t: "math",
        tex: String.raw`\bar v_N = 1, \qquad \bar v_i = \sum_{j\,:\,v_i \to v_j} \Big(\frac{\partial v_j}{\partial v_i}\Big)^{\!\top} \bar v_j \quad\text{for } i = N-1, \dots, 1`,
        caption: "Each adjoint is a sum over the node's consumers; processing in reverse topological order guarantees every consumer is finished first.",
      },
      {
        t: "p",
        text: String.raw`**Why the order matters.** If node $c$ were processed before one of its consumers had added its contribution, $\bar c$ would be incomplete and every gradient upstream of $c$ would be wrong. A topological order exists exactly because the graph has no cycles; recurrent networks are handled by unrolling them in time into a (longer) DAG. **Total cost**: every edge is visited once forward and once backward, and each backward rule costs about as much as its forward operation (a matrix product's VJP is two matrix products of the same size), so a backward pass costs roughly 1–2 forward passes. This is the "cheap gradient principle" of reverse-mode automatic differentiation.`,
      },
      { t: "h", text: "6. The Hessian" },
      {
        t: "p",
        text: String.raw`For a twice-differentiable scalar function $f:\mathbb{R}^n\to\mathbb{R}$, the **Hessian** is the $n\times n$ matrix of second partial derivatives, equivalently the Jacobian of the gradient:`,
      },
      {
        t: "math",
        tex: String.raw`H_f(x) = \nabla^2 f(x) = J_{\nabla f}(x), \qquad H_{ij} = \frac{\partial^2 f}{\partial x_i\,\partial x_j}`,
      },
      {
        t: "p",
        text: String.raw`**Symmetry.** If the second partial derivatives are continuous, the order of differentiation does not matter (Schwarz's / Clairaut's theorem), so $H_{ij} = H_{ji}$ and $H = H^\top$. A real symmetric matrix has real eigenvalues $\lambda_1\le\dots\le\lambda_n$ and an orthonormal eigenbasis $q_1,\dots,q_n$: $H = Q\Lambda Q^\top$. The number $q^\top H q$ for a unit vector $q$ is the **curvature** of $f$ along direction $q$ (the second derivative of $t\mapsto f(x + tq)$ at $t = 0$), so $\lambda_{\min}$ and $\lambda_{\max}$ are the smallest and largest curvatures in any direction.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $f(x, y) = x^3 + y^3 - 3xy$. Gradient: $\nabla f = (3x^2 - 3y,\ 3y^2 - 3x)$. Hessian: $H = \begin{pmatrix}6x & -3\\ -3 & 6y\end{pmatrix}$ — symmetric, as promised. Setting the gradient to zero gives $y = x^2$ and $x = y^2$, so $x^4 = x$: the critical points are $(0,0)$ and $(1,1)$.`,
      },
      {
        t: "table",
        head: ["Point", "$f$", "$H$", "Eigenvalues", "Type"],
        rows: [
          ["$(1,1)$", "$-1$", "$\\begin{pmatrix}6 & -3\\\\ -3 & 6\\end{pmatrix}$", "$3,\\ 9$", "local minimum (all positive)"],
          ["$(0,0)$", "$0$", "$\\begin{pmatrix}0 & -3\\\\ -3 & 0\\end{pmatrix}$", "$-3,\\ 3$", "saddle (mixed signs)"],
        ],
      },
      { t: "h", text: "7. Second-order Taylor expansion and curvature" },
      {
        t: "math",
        tex: String.raw`f(x + \Delta) = f(x) + \nabla f(x)^\top\Delta + \tfrac12\,\Delta^\top H(x)\,\Delta + o(\lVert\Delta\rVert^2)`,
      },
      {
        t: "p",
        text: String.raw`Continuing the example at $x_0 = (1.2,\ 0.9)$ with step $\Delta = (0.1,\ -0.2)$: $f(x_0) = -0.783$, $\nabla f(x_0) = (1.62,\ -1.17)$, $H(x_0) = \begin{pmatrix}7.2 & -3\\ -3 & 5.4\end{pmatrix}$, so $\nabla f^\top\Delta = 0.396$ and $\Delta^\top H\Delta = 0.408$.`,
      },
      {
        t: "table",
        head: ["Model", "Prediction of $f(1.3,\\ 0.7)$", "Error"],
        rows: [
          ["first order: $f + \\nabla f^\\top\\Delta$", "$-0.387$", "$0.197$"],
          ["second order: $\\ldots + \\tfrac12\\Delta^\\top H\\Delta$", "$-0.183$", "$-0.007$"],
          ["exact", "$-0.190$", "—"],
        ],
      },
      {
        t: "p",
        text: String.raw`The second-order model is 28 times more accurate here. Its remaining error is exactly the third-order term $\Delta_1^3 + \Delta_2^3 = 0.001 - 0.008 = -0.007$, since $f$ is a cubic. The quadratic term is what the first-order picture behind gradient descent ignores; the [gradient descent lesson](/learn/m06/gradient-descent) uses it to derive the stability condition $\eta < 2/\lambda_{\max}$.`,
      },
      { t: "h", text: "8. Eigenvalues of the Hessian classify critical points" },
      {
        t: "p",
        text: String.raw`At a critical point ($\nabla f(x^\star) = 0$) the expansion becomes $f(x^\star + \Delta) \approx f(x^\star) + \tfrac12\Delta^\top H\Delta$. Writing $\Delta = \sum_i c_iq_i$ in the eigenbasis gives $\tfrac12\Delta^\top H\Delta = \tfrac12\sum_i\lambda_ic_i^2$. Hence:`,
      },
      {
        t: "list",
        items: [
          String.raw`all $\lambda_i > 0$ ($H$ **positive definite**): $f$ increases in every direction — strict local **minimum**;`,
          String.raw`all $\lambda_i < 0$ (negative definite): strict local **maximum**;`,
          String.raw`some positive and some negative (**indefinite**): a **saddle point** — downhill along eigenvectors with $\lambda_i < 0$;`,
          String.raw`some $\lambda_i = 0$ and the rest of one sign (semidefinite): the test is **inconclusive**; higher-order terms decide (e.g. $x^4$ vs $-x^4$ vs $x^3$ at $0$, all with $f'' = 0$).`,
        ],
      },
      {
        t: "p",
        text: String.raw`For a $2\times2$ Hessian, $\det H = \lambda_1\lambda_2$ and $\operatorname{tr}H = \lambda_1 + \lambda_2$: $\det H < 0$ means a saddle; $\det H > 0$ with $\operatorname{tr}H > 0$ means a minimum. Why saddles, not bad local minima, dominate in high dimensions is discussed in [Optimization: Objectives, Convexity and Minima](/learn/m02/optimization-math).`,
      },
      { t: "h", text: "9. Hessian-vector products via double backprop" },
      {
        t: "p",
        text: String.raw`A network with $d$ parameters has a $d\times d$ Hessian. For $d = 10^8$ that is $10^{16}$ entries — $4\times10^{16}$ bytes (40 petabytes) in float32 — and inverting it would take on the order of $d^3 = 10^{24}$ operations. But the product $Hv$ with any vector $v$ is cheap, because`,
      },
      {
        t: "math",
        tex: String.raw`H v = \nabla_x\big(\nabla f(x)^\top v\big) \qquad (v \text{ held constant})`,
        caption: "The derivative of the scalar ⟨∇f, v⟩ is Hv, because ∂/∂xᵢ Σⱼ (∂f/∂xⱼ) vⱼ = Σⱼ Hᵢⱼ vⱼ.",
      },
      {
        t: "p",
        text: String.raw`So: (1) compute the gradient with reverse mode **while recording a graph of that computation** (`+"`create_graph=True`"+String.raw` in PyTorch); (2) form the scalar $g^\top v$; (3) backpropagate again. This **double backprop** (reverse-over-reverse) costs a small constant multiple of one gradient — roughly two to four. Equivalently, $Hv$ is the JVP of the gradient function in direction $v$ (**forward-over-reverse**), usually the most efficient option. Pearlmutter (1994, "Fast exact multiplication by the Hessian") introduced this exact, cheap $Hv$ for neural networks. With $Hv$ you can run power iteration or the Lanczos method to estimate the top eigenvalues of the Hessian (sharpness), solve $Hd = -g$ with conjugate gradients (Hessian-free optimisation), or estimate $\operatorname{tr}H$ with random probes (Hutchinson's estimator: $\mathbb{E}[v^\top Hv] = \operatorname{tr}H$ when $\mathbb{E}[vv^\top] = I$).`,
      },
      {
        t: "p",
        text: String.raw`**Check on the example:** at $x_0 = (1.2,\ 0.9)$ with $v = (0.6,\ -0.8)$, $Hv = (7.2\cdot0.6 + 3\cdot0.8,\ -3\cdot0.6 - 5.4\cdot0.8) = (6.72,\ -6.12)$, which both the finite-difference and the double-backprop computations below reproduce.`,
      },
      { t: "h", text: "10. Newton's method" },
      {
        t: "p",
        text: String.raw`Minimise the second-order model $m(\Delta) = f + g^\top\Delta + \tfrac12\Delta^\top H\Delta$ (with $g = \nabla f$). Setting $\nabla_\Delta m = g + H\Delta = 0$ gives the **Newton step**`,
      },
      {
        t: "math",
        tex: String.raw`x_{t+1} = x_t - H(x_t)^{-1}\nabla f(x_t) \qquad(\text{in practice: solve } H\Delta = -g \text{, never invert } H)`,
      },
      {
        t: "p",
        text: String.raw`In the eigenbasis the step divides each gradient component by its curvature, $c_i \mapsto c_i - g_i/\lambda_i$: steep directions take small steps, flat directions large ones, and there is no learning rate to tune. Near a minimum with positive definite $H$, Newton's method converges **quadratically** (the number of correct digits roughly doubles per step). On $x^3 + y^3 - 3xy$ from $(1.5,\ 0.8)$ the gradient norm goes $0.516 \to 0.0410 \to 3.7\times10^{-4} \to 3.3\times10^{-8} \to 0$ in five steps.`,
      },
      {
        t: "p",
        text: String.raw`**Why it is impractical for large networks.** (i) Memory and time: $O(d^2)$ storage and $O(d^3)$ solve, impossible for $d$ in the millions or billions. (ii) It finds **critical points, not minima**: where $H$ has negative eigenvalues the step $-H^{-1}g$ moves *uphill* along those directions. From $(0.2,\ 0.1)$, Newton converges to the saddle $(0,0)$ in a few steps, while plain gradient descent from the same start (η = 0.05, 200 steps) reaches the minimum $(1,1)$. Deep-learning loss surfaces are full of saddles. (iii) Stochasticity: a mini-batch Hessian is a noisy estimate, and dividing by noisy small eigenvalues amplifies noise. (iv) Non-smooth activations such as ReLU have zero second derivative almost everywhere, so most curvature comes from the interaction between layers and the loss.`,
      },
      { t: "h", text: "11. Gauss–Newton and Fisher, briefly" },
      {
        t: "p",
        text: String.raw`For a least-squares objective $L(\theta) = \tfrac12\lVert r(\theta)\rVert^2$ with residuals $r:\mathbb{R}^d\to\mathbb{R}^N$ and residual Jacobian $J_r$, the chain rule gives`,
      },
      {
        t: "math",
        tex: String.raw`\nabla L = J_r^\top r, \qquad \nabla^2 L = \underbrace{J_r^\top J_r}_{\text{Gauss--Newton } G} + \sum_{i=1}^{N} r_i\,\nabla^2 r_i`,
      },
      {
        t: "p",
        text: String.raw`The **Gauss–Newton** matrix $G$ drops the second term, which is small when residuals are small or the model is nearly linear. $G$ is always positive semidefinite ($v^\top G v = \lVert J_rv\rVert^2\ge0$), so Gauss–Newton steps never head for saddles or maxima. Example: $r(\theta) = (\theta_1\theta_2 - 1,\ \theta_1 - 0.5)$ at $\theta = (1, 2)$ gives $G = \begin{pmatrix}5 & 2\\ 2 & 1\end{pmatrix}$ with eigenvalues $0.171573,\ 5.828427$, while the true Hessian $\begin{pmatrix}5 & 3\\ 3 & 1\end{pmatrix}$ has eigenvalues $-0.605551,\ 6.605551$ (indefinite). The **generalised Gauss–Newton** extends this to any convex loss on the network output, $G = J^\top H_\ell J$. The **Fisher information matrix** $F = \mathbb{E}\big[\nabla_\theta\log p(y\mid x,\theta)\,\nabla_\theta\log p(y\mid x,\theta)^\top\big]$, with $y$ sampled from the model, coincides with the generalised Gauss–Newton matrix for the common output/loss pairs (softmax + cross-entropy, Gaussian + squared error). Preconditioning by $F$ is **natural gradient descent**; K-FAC (Martens & Grosse, 2015) approximates $F$ by Kronecker-factored blocks per layer. Note that the "empirical Fisher", which uses the observed labels instead of sampled ones, is a different matrix and a poorer curvature estimate.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "h", text: "1. A computational graph, forward and backward" },
      {
        t: "p",
        text: String.raw`The graph below computes $L = (\sigma(wx + b) - y)^2$ one primitive per node. Every edge's local derivative is a $1\times1$ Jacobian, and every backward step multiplies the incoming adjoint by it — a VJP in the scalar case.`,
      },
      { t: "sim", id: "chain-rule", caption: "Values flow forward (accent), adjoints flow backward (clay), in reverse topological order." },
      {
        t: "list",
        items: [
          "Step through the forward pass and note that each node is evaluated only after its inputs: that is a topological order.",
          "Step through the backward pass and check that each node's gradient equals (gradient of the node it feeds) × (local derivative on the edge between them).",
          String.raw`Set $x = 0$. The local derivative on the $w$ edge is $x = 0$, so $\partial L/\partial w = 0$ while $\partial L/\partial b$ is not: a zero entry in a Jacobian blocks the gradient along that path.`,
          String.raw`Drive $z = wx + b$ far from 0 (e.g. $w = 3$, $x = 3$). The sigmoid's local derivative $\sigma'(z)$ becomes tiny, and every adjoint to its left shrinks with it — the diagonal activation Jacobian is nearly zero.`,
        ],
      },
      { t: "h", text: "2. Layers as Jacobians" },
      {
        t: "p",
        text: String.raw`In a full network each backward step multiplies the $\delta$ vector by $W^\top$ (the transpose of the linear layer's Jacobian) and then by the diagonal Jacobian of the activation.`,
      },
      { t: "sim", id: "network", caption: "Backpropagation through a small network; click a connection to compare the backprop gradient with a finite-difference estimate." },
      {
        t: "list",
        items: [
          String.raw`Pause at a hidden layer during the backward phase and verify by hand one entry of $W^\top\delta$: it is a weighted sum of the $\delta$ values of the neurons that the hidden neuron feeds.`,
          String.raw`Switch a hidden layer to ReLU and find a neuron with negative pre-activation: its diagonal Jacobian entry is 0, so its $\delta$ and all its incoming weight gradients are exactly 0.`,
        ],
      },
      { t: "h", text: "3. Hessian eigenvalues on a 2-D surface" },
      {
        t: "p",
        text: String.raw`Choose the **Saddle** surface $f = x^2 + y^4/4 - y^2/2$. Its Hessian at the origin is $\operatorname{diag}(2, -1)$: positive curvature along $x$, negative along $y$. The minima at $(0, \pm1)$ have Hessian $\operatorname{diag}(2, 2)$.`,
      },
      { t: "sim", id: "optimizers", caption: "Optimizer trajectories on 2-D surfaces; pick the Saddle or the Elongated valley surface." },
      {
        t: "list",
        items: [
          String.raw`With the default start $(-1.8,\ 0.01)$, watch SGD run quickly along $x$ to the ridge $x = 0$ and then linger near the saddle before sliding off along $y$. Its slow escape is governed by the negative eigenvalue: the $y$-coordinate is multiplied by $1 + \eta\cdot1$ per step.`,
          String.raw`Switch to the **Elongated valley** $0.1x^2 + 2y^2$: the Hessian is $\operatorname{diag}(0.2, 4)$, condition number 20. The steep eigen-direction limits the learning rate, the flat one limits the speed — the problem Newton's method (dividing by curvature) and adaptive optimizers address.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      { t: "h", text: "A reverse-mode autodiff engine from scratch" },
      {
        t: "p",
        text: "The `Value` class wraps a Python float. Each operation returns a new `Value` that remembers its parents and a closure `_backward` implementing that operation's VJP. `backward()` sorts the graph topologically with a depth-first post-order traversal and calls the closures in reverse. Subtraction and division are built from the primitives, so they need no backward rule of their own. The engine is then checked against PyTorch on the hand-traced expression and on a tiny 3–2–1 tanh network.",
      },
      { t: "code", lang: "python", title: "tiny_autodiff.py", code: AD_CODE, output: AD_OUT },
      {
        t: "list",
        items: [
          "`self.grad += ...` (not `=`) in every closure is the sum-over-paths rule: `a` and `c` each receive contributions from several consumers.",
          "`build` appends a node only after all of its parents: a post-order DFS, which is a topological order. Reversed, every node runs after all nodes that consume it.",
          "`tanh` and `exp` reuse their forward outputs `t` and `e` in the backward rule — the same \"save for backward\" trick real frameworks use.",
          "The graph for `e` has 11 nodes, matching the 9 intermediate nodes in the hand trace plus the two leaves.",
          "The engine is scalar-valued and therefore slow; real frameworks apply the same algorithm to tensor-valued nodes, where each VJP is a vectorised kernel.",
        ],
      },
      { t: "h", text: "Jacobians, VJPs, Hessians and Newton in NumPy" },
      {
        t: "p",
        text: "The second program verifies the formulas of the Math section numerically: a finite-difference Jacobian (one column per input), the Jacobians of a linear layer, tanh and softmax, VJP and JVP computed without the Jacobian, the Hessian as the Jacobian of the gradient with eigenvalue classification, a Hessian-vector product from two gradient evaluations, and Newton's method converging to a minimum from one start and to a saddle from another.",
      },
      { t: "code", lang: "python", title: "jacobians_hessians_numpy.py", code: NP_CODE, output: NP_OUT },
      {
        t: "p",
        text: String.raw`The step $h = 10^{-6}$ for the central difference keeps truncation error ($O(h^2)$) and floating-point round-off ($O(\epsilon/h)$, with $\epsilon\approx2.2\times10^{-16}$) both small, which is why the worst error is around $10^{-10}$. The finite-difference Hessian-vector product, $(\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v))/2\epsilon$, is the numerical twin of double backprop: two gradient evaluations instead of $n$.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "PyTorch offers two APIs. `torch.autograd.functional` (`jacobian`, `hessian`, `vjp`, `jvp`, `hvp`) wraps ordinary autograd. `torch.func` (formerly functorch) provides composable transforms in the style of JAX: `grad`, `jacrev` (Jacobian by reverse mode, one VJP per output, vectorised with `vmap`), `jacfwd` (one JVP per input), `vjp`, `jvp`, and `hessian`. Composing them gives forward-over-reverse Hessian-vector products. The last part computes an exact $Hv$ for an MLP's loss by double backprop with `create_graph=True`.",
              },
              { t: "code", lang: "python", title: "jacobians_pytorch.py", code: TORCH_CODE, output: TORCH_OUT },
              {
                t: "callout",
                kind: "tip",
                title: "jacrev or jacfwd?",
                text: "Use `jacrev` when the function has fewer outputs than inputs (wide Jacobian) and `jacfwd` when it has fewer inputs than outputs (tall Jacobian). For Hessians, `jacfwd(jacrev(f))` (forward-over-reverse) is usually fastest, which is what `torch.func.hessian` does.",
              },
              {
                t: "callout",
                kind: "warning",
                title: "create_graph=True",
                text: "Plain `loss.backward()` frees the graph and records nothing about the gradient computation itself, so the gradient cannot be differentiated again. Second derivatives require `torch.autograd.grad(..., create_graph=True)`, which keeps (and grows) the graph — remember that it costs extra memory.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow records operations on a `tf.GradientTape`. `tape.jacobian` builds full Jacobians (and `tape.batch_jacobian` per-example ones), nested tapes give second derivatives, and `tf.autodiff.ForwardAccumulator` implements forward-mode JVPs.",
              },
              { t: "code", lang: "python", title: "jacobians_tf.py (not executed in CI)", code: TF_CODE, output: "Prints the 3×2 Jacobian [[4, 1], [5, -0.416147], [2, 1]], the Hessian of x³ + y³ − 3xy at (1.2, 0.9) as [[7.2, -3], [-3, 5.4]], the Hessian-vector product [6.72, -6.12] (twice: nested tapes and forward-over-reverse), and the JVP J·(1, 0) = [4, 5, 2]." },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Derivative objects compared" },
      {
        t: "table",
        head: ["Object", "Function type", "Shape", "Cost to compute", "Typical use"],
        rows: [
          ["Gradient $\\nabla f$", "$\\mathbb{R}^n\\to\\mathbb{R}$", "$n$", "1 reverse pass", "every training step"],
          ["Jacobian $J$", "$\\mathbb{R}^n\\to\\mathbb{R}^m$", "$m\\times n$", "$\\min(m, n)$ passes", "small models, sensitivity analysis, normalising flows (log-determinant)"],
          ["JVP $J\\dot x$", "$\\mathbb{R}^n\\to\\mathbb{R}^m$", "$m$", "1 forward-mode pass", "directional derivatives, forward-mode training research, Hv"],
          ["VJP $\\bar y^\\top J$", "$\\mathbb{R}^n\\to\\mathbb{R}^m$", "$n$", "1 reverse pass", "backpropagation"],
          ["Hessian $H$", "$\\mathbb{R}^n\\to\\mathbb{R}$", "$n\\times n$", "$n$ Hessian-vector products", "tiny problems, theory"],
          ["HVP $Hv$", "$\\mathbb{R}^n\\to\\mathbb{R}$", "$n$", "≈ 2–4 gradients", "sharpness, Lanczos spectra, Hessian-free / CG"],
          ["Gauss–Newton / Fisher $G$, $F$", "loss of a model", "$d\\times d$ (approximated)", "via $J$, $J^\\top$ products or Kronecker factors", "natural gradient, K-FAC, Shampoo-style preconditioners"],
        ],
      },
      { t: "h", text: "Ways to build and traverse the graph" },
      {
        t: "table",
        head: ["Approach", "How", "Advantages", "Disadvantages", "Examples"],
        rows: [
          ["Define-by-run (dynamic tape)", "record operations as Python executes them", "ordinary control flow, easy debugging", "graph rebuilt each step; Python overhead", "PyTorch eager, TF GradientTape, the Value engine above"],
          ["Trace and compile (static graph)", "trace once, optimise, then run many times", "operator fusion, memory planning, speed", "data-dependent control flow is harder; retracing", "JAX jit, torch.compile, tf.function, XLA"],
          ["Source transformation", "generate derivative code from the program text", "no runtime tape; optimisable", "complex to implement for a full language", "Tapenade, Zygote (Julia)"],
          ["Forward mode with dual numbers", "carry $(v, \\dot v)$ through every operation", "no tape, low memory", "one pass per input direction", "torch.func.jvp, jax.jvp"],
        ],
      },
      { t: "h", text: "Second-order and quasi-second-order methods" },
      {
        t: "table",
        head: ["Method", "Curvature used", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["Newton", "exact $H$, solved exactly", "quadratic convergence, scale invariant", "$O(d^2)$ memory, $O(d^3)$ time; attracted to saddles", "small convex problems (e.g. logistic regression via IRLS)"],
          ["Gauss–Newton / Levenberg–Marquardt", "$J^\\top J$ (+ damping $\\mu I$)", "PSD, no second derivatives of the model", "still $d\\times d$ unless approximated", "least squares, curve fitting, small networks"],
          ["Hessian-free (truncated Newton)", "$Hv$ or $Gv$ inside conjugate gradients", "no matrix stored", "many HVPs per step; tuning damping", "research on RNNs and deep autoencoders"],
          ["Quasi-Newton (BFGS, L-BFGS)", "low-rank estimate from gradient differences", "cheap, good on smooth deterministic problems", "struggles with mini-batch noise", "full-batch fine-tuning, scientific ML"],
          ["Diagonal / Kronecker preconditioners", "per-parameter or per-layer approximations", "scalable", "approximate", "Adam (diagonal of squared gradients, not a true Hessian), K-FAC, Shampoo"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Training every network.** `loss.backward()` in PyTorch and `jax.grad` are reverse-mode VJP engines over a computational graph; the from-scratch engine in this lesson has the same structure as Karpathy's educational *micrograd*.",
          "**Normalising flows.** Change-of-variables likelihoods need $\\log|\\det J_f|$; flows such as RealNVP use triangular Jacobians so the determinant is a product of diagonal entries.",
          "**Sharpness and training-dynamics research.** The top Hessian eigenvalue, computed by power iteration on HVPs, is tracked to study learning-rate stability; Cohen et al. (2021, \"Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability\") observed it hovering near $2/\\eta$ during full-batch training.",
          "**Second-order optimisation and preconditioning.** K-FAC and Shampoo-style optimisers approximate curvature per layer; L-BFGS is used for full-batch problems such as style transfer and physics-informed networks.",
          "**Influence functions and model pruning.** Classical pruning methods (Optimal Brain Damage, LeCun et al. 1990; Optimal Brain Surgeon, Hassibi & Stork 1993) use Hessian information to decide which weights can be removed with least increase in loss; influence functions use inverse-Hessian-vector products to estimate the effect of a training point.",
          "**Physics-informed and scientific ML.** Loss terms containing derivatives of the network output with respect to its inputs (gradients, Laplacians = traces of Hessians) are computed with nested autodiff.",
          "**Jacobian regularisation and robustness.** Penalising $\\lVert J_f(x)\\rVert_F$ of the input-output Jacobian (estimated with random JVPs/VJPs) smooths a model; contractive autoencoders use this penalty on the encoder.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Reverse mode is ideal for training**: one backward pass gives the gradient of a scalar loss with respect to all parameters, at a small constant multiple of a forward pass.",
          "**Reverse mode costs memory**: every intermediate that a backward rule needs must be stored until the backward pass reaches it. Gradient checkpointing trades recomputation for memory.",
          "**Forward mode is memory-light** but needs one pass per input direction, so it is used for JVPs, few-input functions and as the outer layer of forward-over-reverse HVPs.",
          "**Full Jacobians and Hessians do not scale**: $O(mn)$ and $O(n^2)$ storage. Work with products ($J\\dot x$, $\\bar y^\\top J$, $Hv$) and structure (diagonal, block-diagonal, Kronecker, low rank).",
          "**Curvature is informative but local**: the Hessian describes a quadratic model valid only nearby; for ReLU networks it is piecewise and can change abruptly.",
          "**Second-order methods** converge in far fewer iterations on smooth, well-specified problems, but their per-step cost, sensitivity to noise and attraction to saddles keep first-order methods dominant in deep learning.",
          "**Double backprop needs differentiable backward rules**: some custom operations implement backward functions that are not themselves differentiable, which breaks HVPs.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          String.raw`**Mixing Jacobian layouts.** Some texts use the "denominator layout" ($n\times m$, the transpose). Mixing the two silently transposes results. Fix: decide that $J\in\mathbb{R}^{m\times n}$ (outputs × inputs), check shapes at every step, and remember gradients are $J^\top$ of a $1\times n$ Jacobian.`,
          String.raw`**Writing softmax's Jacobian as diag(p(1 − p)).** That keeps only the diagonal; softmax outputs are coupled through the shared denominator, so the off-diagonal $-p_ip_j$ terms are essential. Fix: use $\operatorname{diag}(p) - pp^\top$, or better, fuse softmax with cross-entropy so that $\bar z = p - e_y$.`,
          "**Overwriting instead of accumulating gradients** in a hand-written engine (`grad = ...` instead of `grad += ...`). Any value used twice then gets only its last contribution. A gradient check on $f(x) = x\\cdot x$ catches it: the correct answer is $2x$, the buggy one $x$.",
          "**Running the backward sweep in the wrong order** (e.g. creation order without a topological sort, or a breadth-first order). A node can then be propagated before all its consumers contribute. Fix: reverse of a DFS post-order.",
          "**Materialising Jacobians for batched inputs.** Calling `jacobian` on a batched function returns a $(B, m, B, n)$ tensor that is mostly zeros. Fix: use `vmap(jacrev(f))` for per-example Jacobians or, better, compute only the VJP/JVP you need.",
          "**Forgetting `create_graph=True`** when differentiating a gradient. The first `grad` call then returns a tensor with no history and the second call fails or returns zeros.",
          String.raw`**Concluding "minimum" from a zero gradient.** $\nabla f = 0$ also holds at maxima and saddles. Check the Hessian's eigenvalues (or at least that the loss cannot decrease along any direction you test).`,
          "**Trusting Newton's method on non-convex losses.** With indefinite $H$ the Newton step can point uphill and converges to saddles. Fix: damp the Hessian ($H + \\mu I$), use Gauss–Newton/Fisher (PSD), or use first-order methods.",
          "**Assuming the Hessian is symmetric for any function.** Symmetry requires continuous second partial derivatives; numerically, finite-difference Hessians are only approximately symmetric. Symmetrise with $(H + H^\\top)/2$ before calling a symmetric eigen-solver.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is a Jacobian and what is its shape?", a: "For $f:\\mathbb{R}^n\\to\\mathbb{R}^m$ the Jacobian is the $m\\times n$ matrix of first partial derivatives, $J_{ij} = \\partial f_i/\\partial x_j$. Row $i$ is the gradient of output $i$; column $j$ is how all outputs respond to input $j$. It is the matrix of the best linear approximation, $f(x+\\Delta)\\approx f(x) + J\\Delta$." },
      { t: "qa", level: "beginner", q: "What is the Jacobian of a linear layer $y = Wx + b$ with respect to $x$, and of an element-wise ReLU?", a: "For the linear layer it is $W$ itself. For ReLU it is diagonal, $\\operatorname{diag}(\\mathbb{1}[z_i > 0])$, because each output depends only on its own input." },
      { t: "qa", level: "beginner", q: "What is the Hessian, and why is it symmetric?", a: "The $n\\times n$ matrix of second partial derivatives of a scalar function, $H_{ij} = \\partial^2 f/\\partial x_i\\partial x_j$ — the Jacobian of the gradient. If the second partials are continuous, mixed partials can be taken in either order (Schwarz/Clairaut), so $H_{ij} = H_{ji}$." },
      { t: "qa", level: "beginner", q: "How do the Hessian's eigenvalues tell you what kind of critical point you are at?", a: "At a point with zero gradient: all eigenvalues positive → local minimum; all negative → local maximum; mixed signs → saddle point; some zero → inconclusive, higher-order terms decide." },
      { t: "qa", level: "intermediate", q: "State the chain rule for vector functions and connect it to backpropagation.", a: "$J_{g\\circ f}(x) = J_g(f(x))J_f(x)$. For a network ending in a scalar loss, $\\nabla_x L^\\top = \\nabla_{h_K}\\ell^\\top J_K\\cdots J_1$. Backprop evaluates this product from the left, one vector-Jacobian product per layer: through a linear layer the VJP is $W^\\top\\bar y$, through an activation it is $\\bar y\\odot\\phi'(z)$ — exactly the $\\delta$ recursion." },
      { t: "qa", level: "intermediate", q: "Derive the softmax Jacobian and simplify the gradient of cross-entropy with respect to the logits.", a: "$\\partial p_i/\\partial z_j = p_i(\\delta_{ij} - p_j)$, i.e. $J = \\operatorname{diag}(p) - pp^\\top$. With $L = -\\log p_y$, $\\bar p = -e_y/p_y$; then $\\bar z = J^\\top\\bar p = \\operatorname{diag}(p)\\bar p - p(p^\\top\\bar p) = -e_y + p$, so $\\partial L/\\partial z = p - e_y$." },
      { t: "qa", level: "intermediate", q: "Why do frameworks never materialise full Jacobians?", a: "They are huge and mostly structured: a batched linear layer's Jacobian is block-diagonal over examples, an activation's is diagonal, and a 4096-wide layer over a batch of 64 would need about 275 GB. Training only needs the VJP $\\bar y^\\top J$, which each primitive can compute directly in time and memory comparable to its forward pass." },
      { t: "qa", level: "intermediate", q: "Why is a topological order required for the backward pass?", a: "A node's adjoint is the sum of contributions from all of its consumers. Processing nodes in reverse topological order guarantees every consumer has already added its contribution before the node propagates its adjoint further. In any other order some adjoints would be used before they are complete." },
      { t: "qa", level: "advanced", q: "Compare forward and reverse mode autodiff. When would you choose forward mode?", a: "Forward mode computes a JVP per pass (one Jacobian column) with memory proportional to live values; reverse mode computes a VJP per pass (one row) but must store intermediates. For $f:\\mathbb{R}^n\\to\\mathbb{R}^m$, a full Jacobian costs $n$ forward or $m$ reverse passes. Choose forward mode when $n \\ll m$ (few inputs, many outputs), for directional derivatives, or as the outer transform in forward-over-reverse Hessian-vector products." },
      { t: "qa", level: "advanced", q: "How do you compute a Hessian-vector product for a network with $10^8$ parameters, and what does it cost?", a: "Use $Hv = \\nabla_\\theta(\\nabla_\\theta L^\\top v)$: compute the gradient with `create_graph=True`, take the dot product with $v$, and backpropagate again (or take a JVP of the gradient function — forward-over-reverse). Cost is a small constant multiple (about 2–4) of one gradient and $O(d)$ memory beyond the graph, versus $O(d^2) = 10^{16}$ numbers for the explicit Hessian." },
      { t: "qa", level: "advanced", q: "Why is Newton's method rarely used to train deep networks, and what do Gauss–Newton and natural gradient change?", a: "The Hessian is too large to form or invert ($O(d^2)$ memory, $O(d^3)$ time), mini-batch estimates are noisy, and on non-convex losses the Newton step converges to any critical point, including saddles, because negative-curvature directions are followed uphill. Gauss–Newton replaces $H$ with $J^\\top H_\\ell J$, which is positive semidefinite; natural gradient uses the Fisher matrix, which coincides with the generalised Gauss–Newton for standard output/loss pairs. Practical methods such as K-FAC approximate these with per-layer Kronecker factors and add damping." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "How many entries does the Jacobian of $f:\\mathbb{R}^5\\to\\mathbb{R}^3$ have?", answer: 15, tolerance: 0, explain: "It is $3\\times5$: one row per output, one column per input." },
          { kind: "numeric", q: "A softmax output is $p = (0.5, 0.3, 0.2)$. What is $\\partial p_1/\\partial z_2$?", answer: -0.15, tolerance: 0.0001, explain: "Off-diagonal entries are $-p_ip_j = -0.5\\times0.3 = -0.15$." },
          { kind: "numeric", q: "For the same $p$, what is $\\partial p_1/\\partial z_1$?", answer: 0.25, tolerance: 0.0001, explain: "Diagonal entries are $p_i(1-p_i) = 0.5\\times0.5 = 0.25$." },
          { q: "You need the full Jacobian of $f:\\mathbb{R}^{1000}\\to\\mathbb{R}^{10}$. What is cheapest?", options: ["1000 forward-mode passes", "10 reverse-mode passes", "One reverse-mode pass", "Finite differences with 1001 evaluations"], answer: 1, explain: "Reverse mode yields one row per pass, and there are only 10 rows." },
          { kind: "numeric", q: "$f(x,y) = x^2 + 3xy + 2y^2$ has a critical point at the origin. What is the determinant of its Hessian?", answer: -1, tolerance: 0.0001, explain: "$H = \\begin{pmatrix}2 & 3\\\\ 3 & 4\\end{pmatrix}$, $\\det H = 8 - 9 = -1 < 0$: eigenvalues of opposite signs ($6.162$ and $-0.162$), so the origin is a saddle." },
          { q: "In a hand-written reverse-mode engine, why must `grad` be accumulated with `+=`?", options: ["To make the code faster", "Because a value used by several operations receives one gradient contribution per use, and the chain rule sums them", "Because gradients are always positive", "To avoid needing a topological sort"], answer: 1, explain: "The multivariable chain rule sums over all paths from a variable to the output." },
          { q: "Newton's method is started near a saddle point of a non-convex function. What typically happens?", options: ["It escapes along the negative-curvature direction", "It converges to the saddle, since it seeks any zero of the gradient", "It diverges to infinity", "It behaves exactly like gradient descent"], answer: 1, explain: "The Newton step divides by the (negative) curvature, which reverses the direction along negative-curvature eigenvectors, so it moves toward the saddle. In the lesson's example it reaches (0, 0) from (0.2, 0.1)." },
          { kind: "numeric", q: "One Newton step on $f(x) = (x-3)^2 + 1$ from $x_0 = 10$. What is $x_1$?", answer: 3, tolerance: 0.0001, explain: "$x_1 = x_0 - f'(x_0)/f''(x_0) = 10 - 14/2 = 3$: Newton minimises a quadratic exactly in one step." },
        ],
      },
      {
        t: "exercise",
        title: "Gradient, Hessian and a Taylor prediction by hand",
        prompt: "For $f(x, y) = x^2y + y^3$ at $(1, 2)$: compute $f$, $\\nabla f$, the Hessian and its eigenvalues. Is the Hessian positive definite? Use the second-order Taylor expansion to predict $f(1.1,\\ 1.9)$ and compare with the exact value.",
        hint: "For a symmetric 2×2 matrix with entries a, b, b, c, the eigenvalues are (a + c)/2 ± sqrt(((a − c)/2)² + b²).",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Value and gradient", tex: String.raw`f(1,2) = 2 + 8 = 10,\qquad \nabla f = (2xy,\ x^2 + 3y^2) = (4,\ 13)` },
              { title: "Hessian", tex: String.raw`H = \begin{pmatrix}2y & 2x\\ 2x & 6y\end{pmatrix} = \begin{pmatrix}4 & 2\\ 2 & 12\end{pmatrix}` },
              { title: "Eigenvalues", tex: String.raw`\lambda = 8 \pm\sqrt{16 + 4} = 8 \pm 4.472136 = 3.527864,\ 12.472136 > 0 \;\Rightarrow\; \text{positive definite}` },
              { title: "Taylor prediction with Δ = (0.1, −0.1)", tex: String.raw`10 + \underbrace{(4)(0.1) + (13)(-0.1)}_{-0.9} + \tfrac12\underbrace{\Delta^\top H\Delta}_{0.12} = 9.16` },
              { title: "Exact", tex: String.raw`f(1.1, 1.9) = 1.21\times1.9 + 6.859 = 9.158 \quad(\text{error } 0.002)` },
            ],
          },
          {
            t: "p",
            text: "The Hessian is positive definite, so $f$ is locally convex around $(1,2)$; but $(1,2)$ is not a minimum, because the gradient is not zero. Curvature and stationarity are separate questions.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Build a Jacobian from VJPs and from JVPs",
        prompt: "For $f(x) = \\tanh(Ax)^2$ (element-wise square) with $A\\in\\mathbb{R}^{2\\times5}$, build the full $2\\times5$ Jacobian twice using `torch.func`: once from VJPs with the standard basis vectors of the output space (rows), once from JVPs with the standard basis vectors of the input space (columns). Check both against `jacrev` and count the passes each approach needs.",
        hint: "`vjp(f, x)` returns `(f(x), pullback)`; `pullback(u)` returns a tuple containing $u^\\top J$. `jvp(f, (x,), (s,))` returns `(f(x), J s)`.",
        solution: [
          { t: "code", lang: "python", title: "jacobian_from_products.py", code: EX_CODE, output: EX_OUT },
          {
            t: "p",
            text: "Reverse mode needs one pass per output (2), forward mode one per input (5). For a scalar loss of a network with millions of parameters the counts become 1 and millions — the reason training uses reverse mode.",
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`Jacobian of $f:\mathbb{R}^n\to\mathbb{R}^m$: $J\in\mathbb{R}^{m\times n}$, $J_{ij} = \partial f_i/\partial x_j$; $f(x+\Delta)\approx f(x) + J\Delta$.`,
          String.raw`Common layers: linear $J = W$; element-wise activation $J = \operatorname{diag}(\phi'(z))$; softmax $J = \operatorname{diag}(p) - pp^\top$ (symmetric, rows sum to 0).`,
          String.raw`Chain rule: $J_{g\circ f} = J_g J_f$. Backprop evaluates $\bar y^\top J_K\cdots J_1$ from the left as a sequence of VJPs.`,
          String.raw`JVP $J\dot x$ (forward mode, one column per pass) vs VJP $\bar y^\top J$ (reverse mode, one row per pass, needs stored intermediates). Scalar loss ⇒ reverse mode: whole gradient in one pass.`,
          "Autodiff = record a DAG, sort topologically, sweep in reverse, **accumulate** adjoints.",
          String.raw`Hessian $H = \nabla^2 f$, symmetric; $f(x+\Delta)\approx f + g^\top\Delta + \tfrac12\Delta^\top H\Delta$; eigenvalues = principal curvatures; signs classify minima, maxima, saddles.`,
          String.raw`$Hv = \nabla(\nabla f^\top v)$ costs a few gradients (double backprop or forward-over-reverse); the full Hessian costs $O(d^2)$ memory.`,
          String.raw`Newton: $x \leftarrow x - H^{-1}g$ — quadratic convergence near minima, but $O(d^3)$, noise-sensitive and attracted to saddles. Gauss–Newton $J^\top J$ and Fisher are PSD alternatives.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Optimization: Objectives, Convexity and Minima](/learn/m02/optimization-math) — uses the Hessian to define convexity, analyse saddle points and smoothness, and derive safe learning rates.",
          "[Backpropagation](/learn/m04/backpropagation) — the VJP chain of this lesson applied layer by layer to a full network, with a complete worked example.",
          "[Numerical Gradient Checking](/learn/m04/gradient-checking) — the finite-difference Jacobians used above, turned into a systematic debugging tool for autodiff code.",
          "[Momentum, Adam and adaptive optimizers](/learn/m06/momentum-adaptive) — how practical optimisers approximate curvature information cheaply instead of using the Hessian.",
          "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — what happens to a long product of Jacobians when their singular values are consistently below or above 1.",
        ],
      },
    ],
  },
};

export default lesson;
