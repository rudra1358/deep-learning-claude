import type { Lesson } from "@/lib/types";

const FD_ERROR = String.raw`import numpy as np

# Differentiate f(x) = sin(x) at x = 1, where the exact derivative is cos(1).
x0, exact = 1.0, np.cos(1.0)
hs = 10.0 ** -np.arange(1, 16)                    # h = 1e-1, 1e-2, ..., 1e-15

def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

err_f = np.array([abs(forward_diff(np.sin, x0, h) - exact) for h in hs])
err_c = np.array([abs(central_diff(np.sin, x0, h) - exact) for h in hs])

print("    h       forward err   central err")
for h, ef, ec in zip(hs, err_f, err_c):
    print(f"  1e{int(np.log10(h)):+03d}    {ef:10.2e}    {ec:10.2e}")

# ---- an ASCII log-log plot: column = log10(h), row = log10(error) ----
rows = range(0, -13, -1)                          # log10(error) from 1e0 down to 1e-12
print("\nlog10(err)   F = forward, C = central, * = both")
for r in rows:
    line = ""
    for ef, ec in zip(err_f, err_c):
        f_here = round(np.log10(max(ef, 1e-30))) == r
        c_here = round(np.log10(max(ec, 1e-30))) == r
        line += "   *" if (f_here and c_here) else "   F" if f_here else "   C" if c_here else "   ."
    print(f"   {r:4d} {line}")
print("  log10 h" + "".join(f"{int(np.log10(h)):4d}" for h in hs))

eps = np.finfo(np.float64).eps
print(f"\nbest h forward: 1e{int(np.log10(hs[err_f.argmin()]))}  (theory 2*sqrt(eps*|f|/|f''|) = {2 * np.sqrt(eps):.1e})")
print(f"best h central: 1e{int(np.log10(hs[err_c.argmin()]))}  (theory (3*eps*|f|/|f'''|)^(1/3) = {(3 * eps * np.sin(1) / np.cos(1)) ** (1 / 3):.1e})")
print(f"smallest errors: forward {err_f.min():.1e}, central {err_c.min():.1e}")

# ---- the same experiment in float32 ----
x32, exact32 = np.float32(1.0), np.cos(1.0)
err32 = [abs(float((np.sin(x32 + np.float32(h)) - np.sin(x32 - np.float32(h))) / np.float32(2 * h)) - exact32)
         for h in hs[:7]]
k = int(np.argmin(err32))
print(f"float32 central: best h = 1e{int(np.log10(hs[k]))}, smallest error {err32[k]:.1e}")`;

const FD_ERROR_OUT = String.raw`    h       forward err   central err
  1e-01      4.29e-02      9.00e-04
  1e-02      4.22e-03      9.00e-06
  1e-03      4.21e-04      9.01e-08
  1e-04      4.21e-05      9.00e-10
  1e-05      4.21e-06      1.11e-11
  1e-06      4.21e-07      2.77e-11
  1e-07      4.18e-08      1.94e-10
  1e-08      2.97e-09      2.58e-09
  1e-09      5.25e-08      2.97e-09
  1e-10      5.85e-08      5.85e-08
  1e-11      1.17e-06      1.17e-06
  1e-12      4.32e-05      1.23e-05
  1e-13      7.34e-04      1.79e-04
  1e-14      3.71e-03      3.71e-03
  1e-15      1.48e-02      1.48e-02

log10(err)   F = forward, C = central, * = both
      0    .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
     -1    F   .   .   .   .   .   .   .   .   .   .   .   .   .   .
     -2    .   F   .   .   .   .   .   .   .   .   .   .   .   *   *
     -3    C   .   F   .   .   .   .   .   .   .   .   .   F   .   .
     -4    .   .   .   F   .   .   .   .   .   .   .   F   C   .   .
     -5    .   C   .   .   F   .   .   .   .   .   .   C   .   .   .
     -6    .   .   .   .   .   F   .   .   .   .   *   .   .   .   .
     -7    .   .   C   .   .   .   F   .   F   *   .   .   .   .   .
     -8    .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
     -9    .   .   .   C   .   .   .   *   C   .   .   .   .   .   .
    -10    .   .   .   .   .   .   C   .   .   .   .   .   .   .   .
    -11    .   .   .   .   C   C   .   .   .   .   .   .   .   .   .
    -12    .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
  log10 h  -1  -2  -3  -4  -5  -6  -7  -8  -9 -10 -11 -12 -13 -14 -15

best h forward: 1e-8  (theory 2*sqrt(eps*|f|/|f''|) = 3.0e-08)
best h central: 1e-5  (theory (3*eps*|f|/|f'''|)^(1/3) = 1.0e-05)
smallest errors: forward 3.0e-09, central 1.1e-11
float32 central: best h = 1e-2, smallest error 1.0e-05`;

const GC_MLP = String.raw`import numpy as np

rng = np.random.default_rng(0)
N, D, H, C = 64, 4, 8, 3                       # examples, inputs, hidden units, classes
X = rng.normal(size=(N, D))
y = np.argmax(X @ rng.normal(size=(D, C)), axis=1)   # labels from a random linear "teacher"
LAM = 0.05                                     # L2 strength: loss includes LAM * sum(W**2)


def init_params(seed=1, dtype=np.float64):
    r = np.random.default_rng(seed)
    return {"W1": (r.normal(size=(D, H)) * 0.5).astype(dtype), "b1": np.zeros(H, dtype),
            "W2": (r.normal(size=(H, C)) * 0.5).astype(dtype), "b2": np.zeros(C, dtype)}


def loss_fn(P, X, y):
    """Mean softmax cross-entropy + L2 penalty on the weight matrices (not the biases)."""
    z1 = X @ P["W1"] + P["b1"]
    h = np.tanh(z1)
    s = h @ P["W2"] + P["b2"]
    s = s - s.max(axis=1, keepdims=True)                     # stable log-softmax
    logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    data = -logp[np.arange(len(y)), y].mean()
    return data + LAM * (np.sum(P["W1"] ** 2) + np.sum(P["W2"] ** 2))


def grads(P, X, y, buggy):
    n = len(y)
    z1 = X @ P["W1"] + P["b1"]
    h = np.tanh(z1)
    s = h @ P["W2"] + P["b2"]
    prob = np.exp(s - s.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    ds = prob.copy()
    ds[np.arange(n), y] -= 1
    ds /= n                                                    # dL/ds for the mean loss
    dh = ds @ P["W2"].T
    if buggy:
        dz1 = dh * h * (1 - h)        # BUG A: sigmoid's derivative a(1-a) pasted in for tanh
        reg = 1.0                     # BUG B: d(LAM*W^2)/dW written as LAM*W
    else:
        dz1 = dh * (1 - h ** 2)       # tanh'(z) = 1 - tanh(z)^2
        reg = 2.0
    return {"W1": X.T @ dz1 + reg * LAM * P["W1"], "b1": dz1.sum(0),
            "W2": h.T @ ds + reg * LAM * P["W2"], "b2": ds.sum(0)}


def rel_error(a, b):
    """Element-wise |a - b| / (|a| + |b|), guarded against 0/0."""
    return np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)


def norm_rel_error(a, b):
    """One number per tensor: ||a - b|| / (||a|| + ||b||)."""
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def gradient_check(P, X, y, buggy, h=1e-5):
    """Central differences on EVERY parameter; report the worst relative error per tensor."""
    G = grads(P, X, y, buggy)
    print(f"  {'tensor':6s} {'shape':8s} {'norm rel err':>12s} {'worst entry':>12s}  verdict")
    for name, W in P.items():
        num = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + h; lp = loss_fn(P, X, y)
            W[idx] = old - h; lm = loss_fn(P, X, y)
            W[idx] = old                                         # always restore!
            num[idx] = (lp - lm) / (2 * h)
        nrel, worst = norm_rel_error(G[name], num), rel_error(G[name], num).max()
        verdict = "ok" if nrel < 1e-7 else "SUSPICIOUS" if nrel < 1e-4 else "WRONG"
        print(f"  {name:6s} {str(W.shape):8s} {nrel:12.2e} {worst:12.2e}  {verdict}")


def directional_check(P, X, y, buggy, h=1e-6, seed=0):
    """One random direction v: compare (L(P+hv) - L(P-hv)) / 2h with <grad, v>. Two loss evaluations."""
    r = np.random.default_rng(seed)
    V = {k: r.normal(size=W.shape) for k, W in P.items()}
    G = grads(P, X, y, buggy)
    plus = {k: P[k] + h * V[k] for k in P}
    minus = {k: P[k] - h * V[k] for k in P}
    num = (loss_fn(plus, X, y) - loss_fn(minus, X, y)) / (2 * h)
    ana = sum(np.sum(G[k] * V[k]) for k in P)
    print(f"  directional: numeric {num:.8f}  analytic {ana:.8f}  rel err {rel_error(ana, num):.2e}")


def train(buggy, steps=300, lr=0.5):
    P = init_params()
    first = loss_fn(P, X, y)
    for _ in range(steps):
        G = grads(P, X, y, buggy)
        for k in P:
            P[k] -= lr * G[k]
    return first, loss_fn(P, X, y)


# 1. the buggy implementation still "trains"
for buggy in (True, False):
    a, b = train(buggy)
    print(f"{'buggy' if buggy else 'fixed'} backprop: loss {a:.4f} -> {b:.4f} after 300 steps")

# 2. the gradient check exposes it
print("\ngradient check, BUGGY backward:")
gradient_check(init_params(), X, y, buggy=True)
directional_check(init_params(), X, y, buggy=True)

print("\nsame check with the L2 term switched off (LAM = 0) to isolate the bugs:")
LAM = 0.0
gradient_check(init_params(), X, y, buggy=True)
LAM = 0.05

print("\ngradient check, FIXED backward:")
gradient_check(init_params(), X, y, buggy=False)
directional_check(init_params(), X, y, buggy=False)

# 3. the same (correct) check in float32 is far noisier
print()
P32, X32 = init_params(dtype=np.float32), X.astype(np.float32)
G32 = grads(P32, X32, y, buggy=False)
for h32 in (1e-1, 1e-2, 1e-3, 1e-4):
    num = np.zeros_like(P32["W1"])
    for idx in np.ndindex(P32["W1"].shape):
        old = P32["W1"][idx]
        P32["W1"][idx] = old + np.float32(h32); lp = loss_fn(P32, X32, y)
        P32["W1"][idx] = old - np.float32(h32); lm = loss_fn(P32, X32, y)
        P32["W1"][idx] = old
        num[idx] = (lp - lm) / np.float32(2 * h32)
    print(f"float32 check of W1 (correct code), h={h32:.0e}: norm rel err {norm_rel_error(G32['W1'], num):.2e}")`;

const GC_MLP_OUT = String.raw`buggy backprop: loss 1.6260 -> 0.8956 after 300 steps
fixed backprop: loss 1.6260 -> 0.7184 after 300 steps

gradient check, BUGGY backward:
  tensor shape    norm rel err  worst entry  verdict
  W1     (4, 8)       8.38e-01     1.00e+00  WRONG
  b1     (8,)         8.77e-01     1.00e+00  WRONG
  W2     (8, 3)       1.10e-01     1.00e+00  WRONG
  b2     (3,)         4.73e-11     3.13e-10  ok
  directional: numeric 0.34859372  analytic 0.22500912  rel err 2.15e-01

same check with the L2 term switched off (LAM = 0) to isolate the bugs:
  tensor shape    norm rel err  worst entry  verdict
  W1     (4, 8)       8.74e-01     1.00e+00  WRONG
  b1     (8,)         8.77e-01     1.00e+00  WRONG
  W2     (8, 3)       3.99e-11     1.91e-09  ok
  b2     (3,)         4.73e-11     3.13e-10  ok

gradient check, FIXED backward:
  tensor shape    norm rel err  worst entry  verdict
  W1     (4, 8)       7.40e-11     1.85e-08  ok
  b1     (8,)         7.62e-11     9.84e-09  ok
  W2     (8, 3)       4.87e-11     9.70e-10  ok
  b2     (3,)         4.73e-11     3.13e-10  ok
  directional: numeric 0.34859372  analytic 0.34859372  rel err 2.48e-10

float32 check of W1 (correct code), h=1e-01: norm rel err 2.77e-03
float32 check of W1 (correct code), h=1e-02: norm rel err 4.15e-05
float32 check of W1 (correct code), h=1e-03: norm rel err 2.66e-04
float32 check of W1 (correct code), h=1e-04: norm rel err 3.17e-03`;

const GC_TORCH = String.raw`import warnings
import torch
import torch.nn as nn
from torch.autograd import gradcheck, gradgradcheck

torch.manual_seed(0)


class SiLUBuggy(torch.autograd.Function):
    """y = x * sigmoid(x), with a hand-written backward that forgets the product rule."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.sigmoid(x)                    # BUG: missing x * s * (1 - s)


class SiLUFixed(SiLUBuggy):
    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_out * (s + x * s * (1 - s))               # d/dx [x s(x)] = s + x s (1 - s)


x = torch.randn(3, dtype=torch.float64, requires_grad=True)   # gradcheck needs float64

# ---- 1. gradcheck catches the buggy backward ----
try:
    gradcheck(SiLUBuggy.apply, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)
except Exception as e:                                         # torch raises GradcheckError
    print("buggy :", type(e).__name__)
    print(str(e).strip())
print("fixed : gradcheck ->", gradcheck(SiLUFixed.apply, (x,), eps=1e-6, atol=1e-5, rtol=1e-3))
# second derivatives: works because the fixed backward is built from differentiable torch ops
print("fixed : gradgradcheck ->", gradgradcheck(SiLUFixed.apply, (x,)))

# ---- 2. float32 inputs: the same correct function fails the default tolerances ----
x32 = x.detach().float().requires_grad_()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        print("float32:", gradcheck(SiLUFixed.apply, (x32,), eps=1e-6, atol=1e-5, rtol=1e-3))
    except Exception as e:
        print("float32:", type(e).__name__, "-", str(e).splitlines()[0])
for w in caught:
    print("         warning:", str(w.message).strip())

# ---- 3. a whole module: dropout must be off, everything in float64 ----
model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Dropout(0.5), nn.Linear(8, 1)).double()
inp = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
model.train()
print("\nMLP with dropout, train mode:", gradcheck(lambda t: model(t), (inp,), raise_exception=False))
model.eval()
print("MLP with dropout, eval mode :", gradcheck(lambda t: model(t), (inp,), raise_exception=False))

# ---- 4. ReLU exactly at its kink ----
k = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
print("\nrelu at x = (0, 1, -1):", gradcheck(torch.relu, (k,), raise_exception=False))
k2 = torch.tensor([0.3, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
print("relu at x = (0.3, 1, -1):", gradcheck(torch.relu, (k2,), raise_exception=False))

# ---- 5. batch norm in train mode: correct, but every probe mutates the running statistics ----
bn = nn.BatchNorm1d(4).double().train()
xb = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
print("\nbatchnorm train mode:", gradcheck(lambda t: bn(t), (xb,), raise_exception=False),
      "| forward passes recorded in running stats:", bn.num_batches_tracked.item())

# ---- 6. directional check of ALL parameters of a model: 2 extra forward passes ----
from torch.nn.utils import parameters_to_vector, vector_to_parameters

net = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, 3)).double()
xs, ys = torch.randn(16, 10, dtype=torch.float64), torch.randint(0, 3, (16,))
loss_of = lambda: nn.functional.cross_entropy(net(xs), ys)

net.zero_grad()
loss_of().backward()
g = torch.cat([p.grad.flatten() for p in net.parameters()])     # analytic gradient
theta = parameters_to_vector(net.parameters()).detach()
v = torch.randn_like(theta)                                       # one random direction
h = 1e-6
with torch.no_grad():
    vector_to_parameters(theta + h * v, net.parameters()); lp = loss_of()
    vector_to_parameters(theta - h * v, net.parameters()); lm = loss_of()
    vector_to_parameters(theta, net.parameters())                 # restore
num, ana = ((lp - lm) / (2 * h)).item(), (g @ v).item()
print(f"\n{theta.numel()} parameters, directional derivative: numeric {num:.10f}, analytic {ana:.10f}, "
      f"rel err {abs(num - ana) / (abs(num) + abs(ana)):.1e}")`;

const GC_TORCH_OUT = String.raw`buggy : GradcheckError
Jacobian mismatch for output 0 with respect to input 0,
numerical:tensor([[ 1.0475,  0.0000,  0.0000],
        [ 0.0000,  0.3554,  0.0000],
        [ 0.0000,  0.0000, -0.0973]], dtype=torch.float64)
analytical:tensor([[0.8236, 0.0000, 0.0000],
        [0.0000, 0.4272, 0.0000],
        [0.0000, 0.0000, 0.1017]], dtype=torch.float64)
fixed : gradcheck -> True
fixed : gradgradcheck -> True
float32: GradcheckError - Jacobian mismatch for output 0 with respect to input 0,
         warning: Input #0 requires gradient and is not a double precision floating point or complex. This check will likely fail if all the inputs are not of double precision floating point or complex.

MLP with dropout, train mode: False
MLP with dropout, eval mode : True

relu at x = (0, 1, -1): False
relu at x = (0.3, 1, -1): True

batchnorm train mode: True | forward passes recorded in running stats: 50

451 parameters, directional derivative: numeric 0.1157288327, analytic 0.1157288328, rel err 2.8e-10`;

const GC_TF = String.raw`import tensorflow as tf

# tf.test.compute_gradient returns (theoretical, numerical) Jacobians for each input
def silu(x):
    return x * tf.sigmoid(x)

x = tf.constant([[-1.2, 0.3, 2.0]], dtype=tf.float64)     # float64, as with any gradient check
theoretical, numerical = tf.test.compute_gradient(silu, [x])
err = tf.reduce_max(tf.abs(theoretical[0] - numerical[0]))
print("max |analytic - numerical| =", float(err))`;

const EX_GC = String.raw`import numpy as np

rng = np.random.default_rng(0)


def silu_forward(x):
    return x / (1.0 + np.exp(-x))                     # x * sigmoid(x)


def silu_backward(x, grad_out):
    s = 1.0 / (1.0 + np.exp(-x))
    return grad_out * (s + x * s * (1.0 - s))


def check_layer(forward, backward, x, h=1e-6):
    """Turn the vector output into a scalar with a fixed random projection r, then check d(sum(r*y))/dx."""
    r = rng.normal(size=forward(x).shape)            # random weights make every output element matter
    analytic = backward(x, r)                         # chain rule: dL/dy = r
    numeric = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h; lp = np.sum(r * forward(x))
        x[i] = old - h; lm = np.sum(r * forward(x))
        x[i] = old
        numeric[i] = (lp - lm) / (2 * h)
    return np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))


x = rng.normal(size=(4, 5))
print(f"correct backward: rel err {check_layer(silu_forward, silu_backward, x):.2e}")
print(f"buggy backward  : rel err {check_layer(silu_forward, lambda x, g: g / (1 + np.exp(-x)), x):.2e}")`;

const EX_GC_OUT = String.raw`correct backward: rel err 6.00e-11
buggy backward  : rel err 1.32e-01`;

const lesson: Lesson = {
  id: "m04/gradient-checking",
  title: "Numerical Gradient Checking",
  summary:
    "How to prove a backward pass is correct: finite differences, the Taylor-series and round-off error analysis behind choosing h, relative-error thresholds, float64, directional checks, the traps (kinks, dropout, batch norm, regularisation), torch.autograd.gradcheck — and a buggy backprop that trains anyway until a gradient check exposes it.",
  minutes: 60,
  level: "intermediate",
  prerequisites: ["m04/backpropagation", "m02/derivatives-chain-rule"],
  next: ["m04/gradient-problems", "m06/gradient-descent", "m16/pytorch"],
  terms: ["gradient-checking", "finite-difference", "central-difference", "relative-error", "machine-epsilon", "directional-derivative"],
  sections: {
    intro: [
      { t: "p", text: "**Gradient checking** compares the gradient from your backward pass (the **analytic** gradient) with an estimate built only from forward passes (the **numerical** gradient, from finite differences). If they agree to many digits, the backward pass is almost certainly right. If they disagree, you have a bug — and the pattern of disagreement usually points to where." },
      { t: "p", text: "Why bother when the loss is clearly going down? Because a wrong gradient is usually still a **descent direction** — a direction that lowers the loss. If your code returns $\\tilde g$ instead of the true gradient $g$, a small step $-\\eta\\tilde g$ changes the loss by about $-\\eta\\, g^\\top\\tilde g$. That is negative whenever the wrong vector is within 90° of the right one ($g^\\top\\tilde g > 0$) — which missing factors, wrong activation derivatives and half-done regularisers usually keep. So training still *works*, just worse. In this lesson's demo, a backward pass with two bugs takes the loss from 1.626 to 0.896, while the correct one reaches 0.718 — nothing about the buggy curve looks broken." },
      { t: "p", text: "[Backpropagation](/learn/m04/backpropagation) already ran a short check. Here we go deeper: *why* central differences, *how* to choose the step $h$ (a trade-off between two errors), which threshold to trust, why float64, how to check millions of parameters cheaply, and when correct code fails or wrong code passes (kinks, randomness, batch statistics, regularisers)." },
      { t: "callout", kind: "key", text: String.raw`Compare $\dfrac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}$ with $\partial L/\partial\theta_i$ in float64 with $h \approx 10^{-5}$–$10^{-6}$, using the relative error $\dfrac{|g_a - g_n|}{|g_a| + |g_n|}$. About $10^{-7}$ or below: correct. Above $10^{-3}$: a bug (or a kink).` },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Backpropagation](/learn/m04/backpropagation) — what an analytic gradient is and how it's computed; this lesson debugs exactly that code. You should be able to read its from-scratch NumPy network.",
        "[Derivatives and the chain rule](/learn/m02/derivatives-chain-rule) — the limit definition $f'(x) = \\lim_{h\\to0}(f(x+h) - f(x))/h$, partial derivatives, and **Taylor's theorem**: $f(x+h) = f(x) + hf'(x) + \\frac{h^2}{2}f''(x) + \\frac{h^3}{6}f'''(x) + O(h^4)$ for smooth $f$.",
        "Floating-point numbers — a float64 holds about 16 significant digits. Its **machine epsilon** $\\varepsilon = 2^{-52} \\approx 2.22\\times10^{-16}$ is the gap between 1 and the next number it can store. float32 has $\\varepsilon = 2^{-23} \\approx 1.19\\times10^{-7}$ (about 7 digits). Each arithmetic step can add a relative error of about $\\varepsilon$.",
        "Basic PyTorch autograd (`requires_grad`, `.backward()`, `.grad`) for the framework section — see the PyTorch part of [Backpropagation](/learn/m04/backpropagation).",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Checking a speedometer with a stopwatch", text: "Your speedometer is the analytic gradient: it claims to show the rate of change directly. To check it, time how long you take between two distance markers — distance over time is the numerical gradient. Markers too far apart (large $h$): you sped up in between, so the average differs from the speed at that moment — **truncation error**. Too close (tiny $h$): a 0.1 s stopwatch error over 1 m ruins the estimate — **round-off error**. The best spacing is in between. One marker before the point and one after cancels most of the speeding-up effect: the **central difference**." },
      { t: "p", text: "A numerical gradient is slow — two forward passes **per parameter** — but it can't share the backward pass's bugs: it only uses the forward pass and the definition of a derivative. Backpropagation is fast but built from many hand-written pieces. Comparing the two on a small problem gives you confidence to use the fast one on the big problem." },
      { t: "p", text: "Compare them with a **relative** measure. A gradient of $10^{-8}$ that should be $2\\times10^{-8}$ is completely wrong, even though the absolute difference is tiny. $1000$ versus $1000.01$ is excellent agreement, despite a difference of $0.01$." },
    ],
    mechanism: [
      { t: "p", text: "A standard gradient check for a model with loss $L(\\theta)$:" },
      { t: "steps", items: [
        { title: "Make the loss deterministic", text: "Turn off dropout and any other sampling, fix the mini-batch (a few examples are enough), turn off data augmentation, and decide what batch norm does (see Common Mistakes). The function must be the *same* function on every call." },
        { title: "Switch to float64", text: "Cast inputs and parameters to double precision. In float32 a finite difference is accurate to about $10^{-5}$ relative at best — too coarse to tell subtle bugs from noise." },
        { title: "Move away from special points", text: "Use random parameters (or take a few training steps) instead of all-zero initialisation, so every code path runs and no unit sits exactly on a kink." },
        { title: "Compute the analytic gradient once", text: "Run forward and backward, and store $g_a = \\nabla_\\theta L$." },
        { title: "Perturb one parameter at a time", text: "For each parameter $\\theta_i$ (or a random subset): set $\\theta_i \\leftarrow \\theta_i + h$ and compute $L_+$; set $\\theta_i \\leftarrow \\theta_i - h$ and compute $L_-$; then **restore** $\\theta_i$.", tex: String.raw`g_{n,i} = \frac{L_+ - L_-}{2h}` },
        { title: "Compare with a relative error", text: "Per entry, and as one number per tensor (using norms).", tex: String.raw`r_i = \frac{|g_{a,i} - g_{n,i}|}{\max(|g_{a,i}| + |g_{n,i}|,\ \tau)}, \qquad r_{\text{tensor}} = \frac{\|g_a - g_n\|_2}{\|g_a\|_2 + \|g_n\|_2}` },
        { title: "Read the report per tensor", text: "Which tensors fail tells you where the bug is. Only the first layer fails? The code sending $\\delta$ into it is wrong. Every weight fails but biases pass? Suspect the weight-decay term." },
        { title: "Optionally: a directional check for the whole model", text: "Pick one random direction $v$ over all parameters and compare $(L(\\theta + hv) - L(\\theta - hv))/2h$ with $g_a^\\top v$: two forward passes for the whole network." },
      ] },
      { t: "callout", kind: "warning", text: "The most common gradient-check bug is in the checker itself: forgetting to restore $\\theta_i$ after perturbing it, or perturbing a *copy* the loss function never reads. Always restore, and test the checker on a function whose derivative you know." },
    ],
    math: [
      { t: "h", text: "Truncation error: why central differences" },
      { t: "p", text: "Expand $f$ around $x$ with Taylor's theorem (assume $f$ has three continuous derivatives):" },
      { t: "math", tex: String.raw`f(x \pm h) = f(x) \pm h f'(x) + \frac{h^2}{2}f''(x) \pm \frac{h^3}{6}f'''(x) + O(h^4)` },
      { t: "p", text: "The **forward difference** uses one extra evaluation:" },
      { t: "math", tex: String.raw`D_+(h) = \frac{f(x+h) - f(x)}{h} = f'(x) + \underbrace{\frac{h}{2}f''(x) + \frac{h^2}{6}f'''(x) + O(h^3)}_{\text{truncation error } = O(h)}` },
      { t: "p", text: "The **central difference** subtracts the two expansions, so the even-order terms ($f$ and $f''$) cancel:" },
      { t: "math", tex: String.raw`D_c(h) = \frac{f(x+h) - f(x-h)}{2h} = f'(x) + \underbrace{\frac{h^2}{6}f'''(x) + O(h^4)}_{\text{truncation error } = O(h^2)}` },
      { t: "p", text: "Make $h$ ten times smaller and the forward error drops tenfold, but the central error drops a hundredfold. The central difference costs two evaluations per parameter instead of one (forward differences share one $f(x)$), and that is almost always worth it." },
      { t: "h", text: "Worked example: f(x) = x³ at x = 2" },
      { t: "p", text: "The exact derivative is $f'(2) = 3\\cdot2^2 = 12$; $f''(2) = 12$, $f'''(x) = 6$, and all higher derivatives are zero, so the error formulas above are exact." },
      { t: "steps", items: [
        { title: "Evaluations, h = 0.1", tex: String.raw`f(2.1) = 9.261, \quad f(2) = 8, \quad f(1.9) = 6.859` },
        { title: "Forward difference", tex: String.raw`D_+ = \frac{9.261 - 8}{0.1} = 12.61, \qquad \text{error } 0.61 = \tfrac{h}{2}f'' + \tfrac{h^2}{6}f''' = 0.6 + 0.01` },
        { title: "Central difference", tex: String.raw`D_c = \frac{9.261 - 6.859}{0.2} = 12.01, \qquad \text{error } 0.01 = \tfrac{h^2}{6}f''' = \tfrac{0.01\cdot 6}{6}` },
        { title: "h = 0.01", tex: String.raw`D_+ = \frac{8.120601 - 8}{0.01} = 12.0601\ (\text{error } 0.0601), \qquad D_c = \frac{8.120601 - 7.880599}{0.02} = 12.0001\ (\text{error } 10^{-4})` },
      ] },
      { t: "p", text: "Ten times smaller $h$: forward error $0.61 \\to 0.0601$ (about $\\div 10$), central error $0.01 \\to 0.0001$ ($\\div 100$)." },
      { t: "h", text: "Round-off error and the optimal step size" },
      { t: "p", text: "In floating point, each computed value is $\\tilde f(x) = f(x)(1 + \\delta)$ with $|\\delta|$ about machine epsilon $\\varepsilon$ (more if the loss has many operations, but it scales the same). So the numerator $\\tilde f(x+h) - \\tilde f(x-h)$ can be off by about $2\\varepsilon|f|$, and dividing by $2h$ makes that an error of about $\\varepsilon |f| / h$ — which **grows** as $h$ shrinks. Subtracting two nearly equal numbers wipes out their leading digits (**catastrophic cancellation**) and leaves mostly noise. The total error of the central difference is about" },
      { t: "math", tex: String.raw`E_c(h) \approx \underbrace{\frac{M_3}{6}h^2}_{\text{truncation}} + \underbrace{\frac{\varepsilon |f|}{h}}_{\text{round-off}}, \qquad M_3 = |f'''(x)|` },
      { t: "p", text: "Setting its derivative to zero, $\\frac{M_3}{3}h - \\frac{\\varepsilon|f|}{h^2} = 0$, gives the best step and the best possible error:" },
      { t: "math", tex: String.raw`h^\ast_c = \left(\frac{3\varepsilon|f|}{M_3}\right)^{1/3} \sim \varepsilon^{1/3}, \qquad E_c(h^\ast_c) = \frac{M_3}{2}\,(h^\ast_c)^2 \sim \varepsilon^{2/3}` },
      { t: "p", text: "The same reasoning for the forward difference, $E_+(h) \\approx \\frac{M_2}{2}h + \\frac{2\\varepsilon|f|}{h}$ with $M_2 = |f''(x)|$, gives" },
      { t: "math", tex: String.raw`h^\ast_+ = 2\sqrt{\frac{\varepsilon|f|}{M_2}} \sim \varepsilon^{1/2}, \qquad E_+(h^\ast_+) = 2\sqrt{\varepsilon|f|M_2} \sim \varepsilon^{1/2}` },
      { t: "table", head: ["", "float64 (ε = 2.22e−16)", "float32 (ε = 1.19e−7)"], rows: [
        ["Forward: optimal h, scale ε^(1/2)", "1.5e−8", "3.5e−4"],
        ["Forward: best error, scale ε^(1/2)", "1.5e−8", "3.5e−4"],
        ["Central: optimal h, scale ε^(1/3)", "6.1e−6", "4.9e−3"],
        ["Central: best error, scale ε^(2/3)", "3.7e−11", "2.4e−5"],
      ] },
      { t: "p", text: "**Checked numerically** (From-Scratch section, $f = \\sin$ at $x = 1$, where $|f| = 0.841$, $|f''| = 0.841$, $|f'''| = 0.540$): predicted $h^\\ast_+ = 3.0\\times10^{-8}$ and $h^\\ast_c = 1.0\\times10^{-5}$; measured best steps (powers of ten) $10^{-8}$ and $10^{-5}$, with errors $3.0\\times10^{-9}$ and $1.1\\times10^{-11}$. Error against $h$ on a log-log plot is V-shaped: slope $+1$ (forward) or $+2$ (central) on the right (truncation), slope $-1$ on the left (round-off). In float32 the best central error is $1.0\\times10^{-5}$, at $h = 10^{-2}$." },
      { t: "p", text: "That's why gradient checks use **float64** and $h$ between $10^{-4}$ and $10^{-6}$. The minimum is wide, and for a loss around 1 you can reach $10^{-8}$–$10^{-11}$ relative accuracy — many orders of magnitude between \"correct\" and \"buggy\". In float32, *correct* code already shows relative errors around $10^{-4}$–$10^{-5}$, overlapping the range of small bugs." },
      { t: "h", text: "The relative error and its thresholds" },
      { t: "math", tex: String.raw`r = \frac{|g_a - g_n|}{\max\left(|g_a| + |g_n|,\ \tau\right)}`, caption: "$g_a$: analytic gradient entry; $g_n$: numerical; $\\tau$ (e.g. $10^{-12}$) avoids $0/0$ when both are zero. Some versions divide by $\\max(|g_a|, |g_n|)$, giving up to twice the value." },
      { t: "p", text: "Example: $g_a = 0.3001$, $g_n = 0.3$ gives $r = 10^{-4}/0.6001 = 1.67\\times10^{-4}$ (or $3.33\\times10^{-4}$ with the max version). Rules of thumb for float64 checks of a smooth loss around 1 — the scale depends on network depth and loss size, so treat them as guidelines, not laws:" },
      { t: "table", head: ["Relative error", "Interpretation"], rows: [
        ["> 1e−2", "Almost certainly a bug (or the check crosses a kink)"],
        ["1e−2 … 1e−4", "Suspicious — investigate unless the loss has kinks (ReLU, max, hinge)"],
        ["1e−4 … 1e−7", "Acceptable for networks with kinks; borderline for fully smooth ones"],
        ["< 1e−7", "Correct; typical for smooth functions in float64 (often 1e−9 to 1e−11)"],
      ] },
      { t: "p", text: "Per-entry errors are dominated by entries whose true gradient is near zero: there $|g_a| + |g_n|$ is tiny, so even round-off noise looks large. The **norm-based** error $\\|g_a - g_n\\|/(\\|g_a\\| + \\|g_n\\|)$ of a whole tensor is far more stable; report both." },
      { t: "h", text: "Directional derivative check" },
      { t: "p", text: "Checking every parameter costs $2P$ forward passes: a million parameters at 10 ms per pass is $2\\times10^6 \\times 0.01$ s $\\approx 5.6$ hours. Two cheaper options are standard. One: check a **random subset** of entries in each tensor. Two: a **directional check** — for a random vector $v$ over *all* parameters," },
      { t: "math", tex: String.raw`\nabla L(\theta)^\top v = \lim_{h\to0}\frac{L(\theta + hv) - L(\theta - hv)}{2h}` },
      { t: "p", text: "— two forward passes for the whole model. If the backward pass returns $\\tilde g = g + e$, the mismatch is $e^\\top v$. For $v \\sim \\mathcal N(0, I)$, $e^\\top v$ has standard deviation $\\|e\\|$ and $g^\\top v$ has $\\|g\\|$, so the relative error is typically about $\\|e\\|/\\|g\\|$ — like the norm-based error, *however many parameters there are*. A bug hides only if $e^\\top v$ is small by chance or $g^\\top v$ is small and inflates noise, so use two or three directions. The catch: it tells you *that* something is wrong, not *where* — follow up with per-tensor checks." },
      { t: "h", text: "Kinks: when a correct gradient fails the check" },
      { t: "p", text: "Taylor's theorem needs $f$ to be smooth on $[x-h, x+h]$. ReLU, $\\max$, $|x|$, hinge loss and max-pooling have **kinks** — points where the slope jumps. If a nudge of size $h$ moves a pre-activation across 0, the central difference averages the two slopes. Example: ReLU at $z = 10^{-6}$ with $h = 10^{-5}$:" },
      { t: "math", tex: String.raw`\frac{\mathrm{ReLU}(10^{-6} + 10^{-5}) - \mathrm{ReLU}(10^{-6} - 10^{-5})}{2\cdot10^{-5}} = \frac{1.1\times10^{-5} - 0}{2\times10^{-5}} = 0.55, \qquad \mathrm{ReLU}'(10^{-6}) = 1, \qquad r = \frac{0.45}{1.55} = 0.29` },
      { t: "p", text: "So one large error in a ReLU network doesn't prove a bug. To diagnose, record whether any pre-activation has $|z| < h\\cdot(\\text{its sensitivity})$ during the two nudged passes (\"kink crossings\"), and re-check with a smaller $h$ or at another random point. A real bug fails every time; a kink failure moves around." },
    ],
    visual: [
      { t: "p", text: "The **Neural Network Visualizer** computes every weight's backprop gradient. Click a connection to also see a live central-difference gradient ($h = 10^{-5}$, double precision) and their relative error. Below $10^{-5}$ it marks the check as passed." },
      { t: "sim", id: "network", caption: "Click any connection: the inspector shows the backprop gradient, the numerical gradient and their relative error." },
      { t: "list", items: [
        "Click several connections in the default sigmoid network: relative errors are about $10^{-14}$ to $10^{-10}$, far below the threshold. This is what correct code looks like in float64.",
        "Set $x_1$ to $0.00001$ and click the weight from $x_1$ into the first hidden neuron: its gradient shrinks to about $2\\times10^{-7}$ and the relative error rises to about $4\\times10^{-6}$. The code is still correct; round-off noise is just large *compared with* a tiny gradient — why per-tensor norm errors beat the worst single entry.",
        "**Make a kink.** Switch the hidden layer to ReLU. Set $x_1 = 1$ and $x_2 = 0$, then select the first hidden neuron: set its weight from $x_1$ to $0.5$ and its bias to $-0.5$, so its pre-activation is exactly $z = 0$. Click that weight: the backprop gradient is 0 (the code uses $\\mathrm{ReLU}'(0) = 0$), but the numerical gradient is half the positive-side slope. The check fails although nothing is wrong. Nudge the bias to $-0.49$ and it passes again.",
        "Change the loss from MSE to BCE and confirm every gradient still checks — the check doesn't care how the gradient was derived.",
      ] },
      { t: "p", text: "The **Chain Rule** simulator shows the same comparison on one computational graph. Step to its final *gradient check* phase: the chain-rule values $\\partial L/\\partial w$ and $\\partial L/\\partial b$ are compared with central differences using $h = 10^{-4}$, and the absolute difference is about $10^{-10}$ or smaller — matching a truncation error $\\frac{h^2}{6}|f'''|$ with $h^2/6 \\approx 1.7\\times10^{-9}$." },
      { t: "sim", id: "chain-rule", caption: "Step to the end to see the chain-rule gradient compared against a finite difference." },
    ],
    scratch: [
      { t: "h", text: "1. Error versus step size, measured" },
      { t: "p", text: "This program differentiates $\\sin$ at $x = 1$ with forward and central differences for $h = 10^{-1}, \\dots, 10^{-15}$. It prints the error table, draws an ASCII log-log plot, compares the measured best $h$ with the formulas above, then repeats the central difference in float32." },
      { t: "code", lang: "python", title: "finite_difference_error.py", code: FD_ERROR, output: FD_ERROR_OUT },
      { t: "list", items: [
        "Right side of the plot (large $h$): forward errors drop one decade per decade of $h$ (slope 1), central errors two (slope 2) — the $O(h)$ and $O(h^2)$ truncation terms.",
        "Left side (small $h$): both rise one decade per decade (slope −1) — round-off $\\propto \\varepsilon/h$. Below $h \\approx 10^{-9}$ both methods give nearly the same (bad) answers, because round-off dominates.",
        "The best central difference ($1.1\\times10^{-11}$) is about 300 times more accurate than the best forward one ($3.0\\times10^{-9}$), matching $\\varepsilon^{2/3}$ vs $\\varepsilon^{1/2}$.",
        "In float32 nothing better than $10^{-5}$ is possible, at the much larger $h = 10^{-2}$.",
      ] },
      { t: "h", text: "2. A buggy backward pass, caught" },
      { t: "p", text: "A one-hidden-layer tanh network with softmax cross-entropy and an L2 penalty $\\lambda(\\|W_1\\|^2 + \\|W_2\\|^2)$, $\\lambda = 0.05$. The `buggy=True` backward pass has two realistic mistakes. **Bug A** pastes the sigmoid derivative $a(1-a)$ where the tanh derivative $1-a^2$ belongs. **Bug B** writes the gradient of $\\lambda W^2$ as $\\lambda W$ instead of $2\\lambda W$. The program trains both versions, runs a full per-tensor check and a one-direction check, isolates the bugs by switching off the penalty, checks the fixed code, and finally runs the check in float32." },
      { t: "code", lang: "python", title: "gradient_check_mlp.py", code: GC_MLP, output: GC_MLP_OUT },
      { t: "p", text: "Reading the report like a debugger:" },
      { t: "list", items: [
        "**The buggy network trains.** Loss 1.626 → 0.896. Only the fixed code's 0.718 hints that something is off — without it you'd never know.",
        "**`b2` passes, everything else fails.** `b2`'s gradient is `ds.sum(0)`, which touches neither the hidden derivative nor the penalty, so both bugs are elsewhere.",
        "**Switching off the penalty** ($\\lambda = 0$): now `W2` passes, so its remaining error came from the regulariser (bug B). `W1` and `b1` still fail — exactly the tensors after the hidden-layer derivative (bug A). Biases aren't regularised, so `b1` failing at $\\lambda = 0$ points straight at the activation derivative.",
        "**The fixed code** gives norm-based errors around $10^{-11}$ for every tensor and worst single entries around $10^{-8}$. The directional check agrees to $2.5\\times10^{-10}$ with just two loss evaluations.",
        "**The directional check also caught the bugs** ($r = 0.215$) with two forward passes — but couldn't say which tensor was wrong.",
        "**float32** with correct code: the error is smallest ($4.2\\times10^{-5}$) at $h = 10^{-2}$ and grows to $3.2\\times10^{-3}$ at $h = 10^{-4}$ — the V-shape again, with a floor high enough to hide small bugs.",
      ] },
      { t: "p", text: "The fix is the two lines in the `else` branch: `dz1 = dh * (1 - h ** 2)` and `reg = 2.0`." },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "`torch.autograd.gradcheck(fn, inputs)` builds the full numerical Jacobian of `fn` with central differences (default `eps=1e-6`) and compares it entry by entry with autograd's Jacobian, using `|analytic − numerical| ≤ atol + rtol·|numerical|` (defaults `atol=1e-5`, `rtol=1e-3`). `gradgradcheck` does the same for second derivatives (the backward of the backward), which matters for custom functions used in Hessian-vector products, gradient penalties or meta-learning. Both expect **double-precision** inputs. The program tests a custom `autograd.Function` with a buggy backward, its fix, float32, dropout, a ReLU kink, train-mode batch norm, and a directional check of an `nn.Module`'s parameters." },
          { t: "code", lang: "python", title: "gradcheck_pytorch.py", code: GC_TORCH, output: GC_TORCH_OUT },
          { t: "list", items: [
            "The buggy SiLU backward returns $\\sigma(x)$, missing the product-rule term $x\\sigma(x)(1-\\sigma(x))$. gradcheck prints the numerical and analytical Jacobians side by side (diagonal, because SiLU acts on each element separately).",
            "float32: gradcheck warns that the inputs aren't double precision, and the correct function fails.",
            "Dropout in train mode fails because each call draws a new mask — the \"function\" changes between the $+h$ and $-h$ evaluations. In eval mode it passes.",
            "ReLU exactly at 0 fails (the numerical derivative there is 0.5, autograd uses 0); moving the point to 0.3 passes.",
            "Batch norm in train mode **passes** — normalising with batch statistics is a deterministic, differentiable function of the whole batch. But the check ran 50 forward passes, each updating the running mean and variance, so the module's state afterwards isn't what it was before.",
          ] },
          { t: "callout", kind: "tip", text: "For large modules, gradcheck's full Jacobian is expensive. Pass `fast_mode=True` (it checks random projections of the Jacobian, the same idea as the directional check), wrap the model so it returns a scalar or small output, or use the parameter-vector directional check in part 6." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "TensorFlow offers `tf.test.compute_gradient(f, x)`, which returns the autodiff and finite-difference Jacobians of `f` at `x`; you compare them yourself. As in PyTorch, use float64 tensors. For Keras models, set the float type to float64 before building the model so the weights are double precision too." },
          { t: "code", lang: "python", title: "gradient_check_tf.py (not executed in CI)", code: GC_TF, output: "Prints the maximum absolute difference between the autodiff and finite-difference Jacobians of SiLU — a very small number (far below 1e-6) for this correct, float64 function." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Method", "Cost (forward passes)", "Truncation error", "Advantages", "Disadvantages / use"], rows: [
        ["Forward difference", "P + 1", "O(h)", "Cheapest; reuses L(θ)", "Best accuracy only ~√ε; rarely worth it for checking"],
        ["Central difference", "2P", "O(h²)", "Best accuracy ~ε^(2/3); standard choice", "Twice the cost of forward"],
        ["Higher-order stencils (e.g. 5-point)", "4P", "O(h⁴)", "Even more accurate for smooth f", "More cost; rarely needed in float64"],
        ["Complex-step derivative", "P (complex arithmetic)", "O(h²), no cancellation", "Machine-precision accuracy with tiny h", "Every op must accept complex numbers; uncommon in DL code"],
        ["Random subset of entries", "2k", "as central", "Scales to large models; shows which tensor fails", "Can miss bugs that affect few entries (e.g. one index)"],
        ["Directional (random v)", "2 per direction", "as central", "Whole model in seconds; sensitivity ~‖e‖/‖g‖", "Doesn't show where the bug is"],
        ["gradcheck / gradgradcheck", "2 × #inputs (slow mode)", "as central", "Checks full Jacobians, second derivatives", "Expensive for large inputs; use fast_mode"],
        ["Comparison with a trusted autodiff", "1 backward", "none", "Exact reference when re-implementing a layer by hand", "Needs a trusted reference implementation"],
      ] },
      { t: "p", text: "The **complex-step** method uses $f'(x) \\approx \\mathrm{Im}\\,f(x + ih)/h$: no subtraction, so no cancellation, and $h$ can be $10^{-20}$ with machine-precision accuracy — but only for analytic code that accepts complex inputs. In deep learning the common \"variation\" is a different *target*: checking a layer's Jacobian through a random scalar projection $L = r^\\top f(x)$ (coding exercise below), which turns any vector-valued layer into a scalar function without losing sensitivity." },
    ],
    applications: [
      { t: "list", items: [
        "**Writing a custom layer or `autograd.Function`** with a hand-written backward (for speed or memory, e.g. a fused activation or custom attention kernel): gradcheck/gradgradcheck it in float64 before using it.",
        "**Implementing backprop from scratch** — in courses, NumPy prototypes, or a new framework's operator library. Frameworks test their own operators' gradients with finite-difference checks like this.",
        "**Debugging a model that trains poorly**: if a network of standard layers underperforms, a directional check on the full loss quickly rules gradient bugs in or out (e.g. a detached tensor or an accidental in-place operation that blocks gradient flow).",
        "**Verifying hand-derived gradients** in research code: new losses, differentiable simulators, implicit layers, custom regularisers.",
        "**Teaching and interviews**: it's the standard answer to \"how do you know your backprop is right?\"",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Strength — independence:** the numerical gradient uses only the forward pass and the definition of a derivative, so it can't share the backward pass's bugs.",
        "**Strength — localisation:** per-tensor (or per-layer) reports point at the faulty code; switching terms on and off (regulariser, extra losses) narrows it further.",
        "**Limitation — cost:** $2P$ forward passes; only practical on tiny models, random subsets, or with directional checks.",
        "**Limitation — precision:** needs float64; correct float32 code already shows errors around $10^{-4}$–$10^{-5}$.",
        "**Limitation — non-smoothness:** kinks cause false alarms; thresholds must be looser for ReLU networks.",
        "**Limitation — it checks consistency, not whether the model is right:** if the *forward* pass has a bug (wrong loss formula, wrong mask), the gradient of that wrong function checks perfectly. Test the forward pass separately against known values.",
        "**Limitation — the point matters:** a check at a special point (all-zero weights, saturated units) can pass while bugs hide in code paths that are inactive there.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Running the check in float32.** Round-off limits accuracy to about $10^{-5}$ at best, and default tolerances fail correct code. Fix: cast the model and inputs to float64 (`model.double()`).",
        "**Leaving dropout or other randomness on.** $L(\\theta + h)$ and $L(\\theta - h)$ then use different masks, so the difference means nothing. Fix: `model.eval()` for dropout, or fix the mask/seed so both evaluations see the same noise; also turn off data augmentation and stochastic depth.",
        "**Batch norm in train mode, handled inconsistently.** In train mode each example's loss depends on the whole batch. Checking a per-example loss, or comparing a train-mode analytic gradient with an eval-mode numerical one, gives false failures. Also, every probe forward updates the running statistics. Fix: check the full-batch loss in one consistent mode (train mode is fine) and restore or deep-copy the module.",
        "**Including the regulariser on one side only.** If the checker's loss leaves out weight decay but the analytic gradient includes it (or the reverse), every weight fails while biases pass. Fix: make both sides compute the same objective — and check the data term and regulariser separately when debugging.",
        "**Using an absolute error.** $|g_a - g_n| = 10^{-6}$ is excellent for a gradient of 10 and a total failure for a gradient of $10^{-6}$. Fix: relative error with a small floor in the denominator.",
        "**$h$ too small.** $h = 10^{-10}$ in float64 already gives errors around $10^{-7}$ from round-off, which looks like a bug. Fix: $h \\approx 10^{-5}$–$10^{-6}$ for central differences.",
        "**Forgetting to restore the parameter**, or perturbing a copy the loss never reads (then every numerical gradient is exactly 0). Fix: restore in the loop and test the checker on a function with a known derivative.",
        "**Mistaking a kink failure for a bug — or a bug for a kink.** Fix: re-check with a smaller $h$ and at another random point; real bugs persist, kink failures move.",
        "**Checking only at initialisation with zero biases and symmetric weights.** Some code paths are inactive or degenerate there. Fix: random parameters, or check after a few training steps.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is gradient checking and why do we do it?", a: String.raw`It compares the analytic gradient from backpropagation with a numerical estimate from finite differences, $(L(\theta + h e_i) - L(\theta - h e_i))/2h$. We do it because backward-pass bugs rarely stop training — a wrong gradient is often still a descent direction — so a falling loss proves nothing.` },
      { t: "qa", level: "beginner", q: "Why use the central difference instead of the forward difference?", a: String.raw`Its truncation error is $O(h^2)$ instead of $O(h)$, because the $f''$ terms of the two Taylor expansions cancel. In float64 that gives best-case accuracy around $10^{-10}$–$10^{-11}$ versus about $10^{-8}$ for forward differences, at twice the cost per parameter.` },
      { t: "qa", level: "beginner", q: "Why is a relative error used, and what values are acceptable?", a: String.raw`Gradients span many orders of magnitude, so an absolute difference means nothing without the scale. With $r = |g_a - g_n|/(|g_a| + |g_n|)$ in float64: below about $10^{-7}$ is correct for smooth functions, up to $10^{-4}$ can be fine with kinks, and above $10^{-3}$–$10^{-2}$ means a bug.` },
      { t: "qa", level: "intermediate", q: "Why not make h as small as possible?", a: String.raw`Round-off. The computed values $f(x \pm h)$ each carry relative errors of about $\varepsilon$; their difference loses leading digits (catastrophic cancellation), and the error $\sim \varepsilon|f|/h$ grows as $h$ shrinks. Balancing it against truncation $\sim |f'''|h^2/6$ gives $h^\ast \sim \varepsilon^{1/3} \approx 6\times10^{-6}$ in float64.` },
      { t: "qa", level: "intermediate", q: "Why must gradient checks run in double precision?", a: String.raw`In float32 ($\varepsilon \approx 1.2\times10^{-7}$) the best central difference reaches only about $\varepsilon^{2/3} \approx 2\times10^{-5}$ relative accuracy, at $h \approx 5\times10^{-3}$. Correct code then shows errors of $10^{-5}$–$10^{-3}$, overlapping the errors small bugs produce, so the check can't tell them apart. float64 pushes the floor below $10^{-9}$.` },
      { t: "qa", level: "intermediate", q: "Your gradient check fails for a few entries in a ReLU network but passes everywhere else. Is there a bug?", a: String.raw`Not necessarily. If a nudge of $\pm h$ moves some pre-activation across 0, the central difference averages the slopes 0 and 1 (e.g. 0.55 instead of 1), giving a big error for that entry. Check whether any pre-activation is within the nudge range, and re-run with a smaller $h$ or at another random point. A real bug fails consistently and usually affects a whole tensor.` },
      { t: "qa", level: "intermediate", q: "How do you check a model with millions of parameters?", a: String.raw`Check a random subset of entries per tensor, use a tiny batch and a smaller copy of the architecture, or use directional checks: compare $(L(\theta + hv) - L(\theta - hv))/2h$ with $\nabla L^\top v$ for one to three random vectors $v$. Each costs two forward passes and has relative sensitivity about $\|e\|/\|g\|$, whatever the number of parameters. Then find any failure with per-tensor checks.` },
      { t: "qa", level: "advanced", q: "What does torch.autograd.gradgradcheck test, and when do you need it?", a: String.raw`It checks second derivatives: it numerically differentiates the backward function itself and compares with autograd's double-backward. You need it when a custom autograd.Function will be differentiated twice — Hessian-vector products, gradient-penalty losses (e.g. WGAN-GP), meta-learning such as MAML. A backward written with non-differentiable operations (e.g. NumPy calls or in-place ops on saved tensors) can pass gradcheck but fail gradgradcheck.` },
      { t: "qa", level: "advanced", q: "Can a gradient check pass while the model is still wrong?", a: String.raw`Yes. It only checks that the backward pass matches the forward pass — it differentiates whatever the forward computes. A wrong loss formula, wrong mask or data-pipeline bug gives a perfectly checked gradient of the wrong objective. It can also pass at a point where buggy code paths are inactive (e.g. all units in one regime), or miss a bug outside the checked subset. You also need forward-pass unit tests and checks at several random points.` },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Estimate the derivative of $f(x) = x^2$ at $x = 3$ with a forward difference and $h = 0.1$. What is the absolute error?", answer: 0.1, tolerance: 0.001, explain: "$(3.1^2 - 9)/0.1 = 0.61/0.1 = 6.1$, true value 6, error $0.1 = \\frac{h}{2}f''(x) = 0.05\\cdot 2$." },
        { kind: "numeric", q: "Central difference for $f(x) = x^3$ at $x = 2$ with $h = 0.1$: what value do you get?", answer: 12.01, tolerance: 0.0005, explain: "$(9.261 - 6.859)/0.2 = 12.01$; the error $0.01 = \\frac{h^2}{6}f'''$ with $f''' = 6$." },
        { kind: "numeric", q: "Analytic gradient 0.3001, numerical 0.3000. Compute the relative error $|g_a - g_n|/(|g_a| + |g_n|)$ and enter it multiplied by $10^4$ (2 decimals).", answer: 1.67, tolerance: 0.01, explain: "$10^{-4}/0.6001 = 1.666\\times10^{-4}$ → 1.67." },
        { kind: "numeric", q: "With float64's $\\varepsilon = 2.22\\times10^{-16}$, compute $\\varepsilon^{1/3}$ (the scale of the optimal central-difference step). Enter it in units of $10^{-6}$ (2 decimals).", answer: 6.06, tolerance: 0.02, explain: "$(2.22\\times10^{-16})^{1/3} = 6.06\\times10^{-6}$." },
        { kind: "numeric", q: "How many forward passes does a full central-difference check of a model with 1,000 parameters need (ignoring the one analytic forward/backward)?", answer: 2000, tolerance: 0, explain: "Two per parameter: $L(\\theta + he_i)$ and $L(\\theta - he_i)$." },
        { q: "A correct implementation fails a gradient check with relative errors around 1e−3 for every parameter. Which cause is most likely?", options: ["The check is running in float32", "h is 1e−5 in float64", "The network uses tanh", "The batch has 8 examples"], answer: 0, explain: "In float32 the best possible relative accuracy is ~1e−5, and typical values are 1e−4 to 1e−3 — a precision problem, not a bug." },
        { q: "Why can a gradient check fail on a correct dropout network in train mode?", options: ["Dropout has no gradient", "Each forward pass samples a different mask, so L(θ+h) and L(θ−h) are evaluated on different functions", "Dropout makes the loss convex", "Dropout requires float16"], answer: 1, explain: "A numerical derivative only makes sense for a fixed function; turn off dropout (eval mode) or fix the mask." },
        { q: "In a per-tensor report, every weight matrix fails but every bias passes. What is the most likely bug?", options: ["The softmax derivative", "The weight-decay (L2) gradient term", "The learning rate", "The data loader"], answer: 1, explain: "Weight decay usually applies only to weights. A passing bias rules out errors in that layer's shared δ computation; a regulariser mismatch hits exactly the weights." },
      ] },
      { t: "exercise", title: "Forward vs central difference by hand", prompt: "For $f(x) = e^x$ at $x = 0$ (exact derivative 1), compute the forward and central differences with $h = 0.1$ and $h = 0.01$, and their errors. Check that the errors match $\\frac{h}{2}f''(0)$ and $\\frac{h^2}{6}f'''(0)$ to leading order.", hint: "$e^{0.1} = 1.105171$, $e^{-0.1} = 0.904837$, $e^{0.01} = 1.010050$, $e^{-0.01} = 0.990050$. All derivatives of $e^x$ at 0 equal 1.", solution: [
        { t: "table", head: ["h", "forward D₊", "error", "h/2", "central D_c", "error", "h²/6"], rows: [
          ["0.1", "1.051709", "5.17e−2", "5.0e−2", "1.001668", "1.67e−3", "1.67e−3"],
          ["0.01", "1.005017", "5.02e−3", "5.0e−3", "1.000017", "1.67e−5", "1.67e−5"],
        ] },
        { t: "p", text: "The forward error shrinks tenfold with $h$ and is a little above $h/2$ (the next term, $h^2/6$, adds to it). The central error shrinks a hundredfold and matches $h^2/6$ to three digits. At these step sizes round-off doesn't matter yet." },
      ] },
      { t: "exercise", title: "Check a vector-valued layer with a random projection", prompt: "A layer maps $x \\in \\mathbb{R}^{4\\times5}$ to $y = x\\,\\sigma(x)$ element-wise (SiLU). Write `check_layer(forward, backward, x)` that verifies `backward(x, grad_out)` — which must return $\\partial L/\\partial x$ given $\\partial L/\\partial y$ — by defining the scalar $L = \\sum r \\odot y$ with a fixed random $r$, so that $\\partial L/\\partial y = r$. Report the norm-based relative error for the correct backward and for a buggy one that returns `grad_out * sigmoid(x)`.", hint: "With $L = \\sum_i r_i y_i$, the analytic gradient is `backward(x, r)`. Nudge each entry of x by ±h and restore it.", solution: [
        { t: "code", lang: "python", title: "check_layer.py", code: EX_GC, output: EX_GC_OUT },
        { t: "p", text: "The random projection gives every output element a random weight in the scalar, so an error anywhere in the Jacobian-vector product shows up (with probability 1). It's the same trick as `gradcheck`'s fast mode and the directional check, applied to outputs instead of parameters. The correct backward agrees to $6\\times10^{-11}$; the buggy one (missing the product-rule term) is off by 13%." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "A falling loss doesn't prove the gradient is right: any $\\tilde g$ with $g^\\top\\tilde g > 0$ is a descent direction.",
        "Forward difference: $(f(x+h)-f(x))/h$, error $\\frac{h}{2}f'' = O(h)$. Central: $(f(x+h)-f(x-h))/2h$, error $\\frac{h^2}{6}f''' = O(h^2)$.",
        "Round-off adds $\\sim\\varepsilon|f|/h$. Best central step $h^\\ast = (3\\varepsilon|f|/|f'''|)^{1/3} \\sim \\varepsilon^{1/3} \\approx 10^{-5}$ in float64, best error $\\sim\\varepsilon^{2/3}$; forward: $h^\\ast \\sim \\sqrt\\varepsilon$, error $\\sim\\sqrt\\varepsilon$.",
        "Relative error $|g_a - g_n|/(|g_a| + |g_n|)$; also report the norm-based version per tensor. Float64, smooth: $< 10^{-7}$ is correct; $> 10^{-3}$ is a bug or a kink.",
        "Always check in float64 (float32 bottoms out at ~$10^{-5}$).",
        "Scale up with random subsets or directional checks: $\\nabla L^\\top v$ vs $(L(\\theta+hv)-L(\\theta-hv))/2h$, two passes, sensitivity $\\sim\\|e\\|/\\|g\\|$.",
        "Traps: kinks (ReLU, max, abs), dropout/randomness, batch norm mode and state, regulariser on one side only, forgetting to restore parameters, special points.",
        "PyTorch: `torch.autograd.gradcheck` / `gradgradcheck` on double inputs; `fast_mode=True` for big inputs.",
        "Find bugs by which tensors fail, and by switching loss terms on and off.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — once gradients are *correct*, the next question is whether they're *useful* in a deep network.",
        "[Gradient Descent](/learn/m06/gradient-descent) — how verified gradients become parameter updates.",
        "[PyTorch: Tensors to Distributed Training](/learn/m16/pytorch) — autograd, custom functions and where gradcheck fits into testing real code.",
        "[Backpropagation](/learn/m04/backpropagation) — revisit the from-scratch network and its short gradient check with this lesson's tools.",
      ] },
    ],
  },
};

export default lesson;
