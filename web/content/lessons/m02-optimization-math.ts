import type { Lesson } from "@/lib/types";

const LAND_CODE = String.raw`import numpy as np

rng = np.random.default_rng(0)


def gd(grad, w0, lr, steps):
    w = np.array(w0, dtype=float)
    for _ in range(steps):
        w = w - lr * grad(w)
    return w


# ---------- 1. convex vs non-convex in 1-D ----------
convex = lambda w: np.log1p(np.exp(-w)) + 0.1 * w ** 2          # logistic loss + L2 penalty
convex_grad = lambda w: -1 / (1 + np.exp(w)) + 0.2 * w
nonconvex = lambda w: w ** 4 - 3 * w ** 2 + w                     # the "two minima" curve
nonconvex_grad = lambda w: 4 * w ** 3 - 6 * w + 1


def chord_violations(f, trials=100_000):
    """Count pairs (a, b, t) with f(t a + (1-t) b) > t f(a) + (1-t) f(b): impossible if f is convex."""
    a, b = rng.uniform(-2, 2, trials), rng.uniform(-2, 2, trials)
    t = rng.uniform(0, 1, trials)
    return int(np.sum(f(t * a + (1 - t) * b) > t * f(a) + (1 - t) * f(b) + 1e-12))


print("chord-inequality violations: convex", chord_violations(convex), "| non-convex", chord_violations(nonconvex))
starts = [-2.0, -0.5, 0.5, 2.0]
print("GD end points, convex    :", [round(float(gd(convex_grad, s, 0.5, 500)), 4) for s in starts])
ends = [float(gd(nonconvex_grad, s, 0.02, 500)) for s in starts]
print("GD end points, non-convex:", [round(e, 4) for e in ends], "losses", [round(nonconvex(e), 4) for e in ends])

# ---------- 2. the simplest non-convex "network": f(w1, w2) = (w1 * w2 - 1)^2 ----------
f2 = lambda w: (w[0] * w[1] - 1) ** 2
A, B = np.array([1.0, 1.0]), np.array([-1.0, -1.0])               # two global minima (loss 0)
print(f"f(A) = {f2(A)}, f(B) = {f2(B)}, f(midpoint) = {f2((A + B) / 2)}  -> not convex")
H0 = np.array([[0.0, -2.0], [-2.0, 0.0]])                          # Hessian of f2 at the origin
print("Hessian eigenvalues at the origin:", np.linalg.eigvalsh(H0), "-> saddle")

# ---------- 3. permutation symmetry of a trained 1-hidden-layer network ----------
X = np.linspace(-3, 3, 64).reshape(-1, 1)
Y = np.sin(X)
H = 8                                                              # hidden units
W1, b1 = rng.normal(0, 1, (1, H)), np.zeros(H)
W2, b2 = rng.normal(0, 1 / np.sqrt(H), (H, 1)), np.zeros(1)


def mlp_loss(W1, b1, W2, b2):
    return float(np.mean((np.tanh(X @ W1 + b1) @ W2 + b2 - Y) ** 2))


for step in range(3000):                                           # plain full-batch gradient descent
    Hd = np.tanh(X @ W1 + b1)
    err = 2 * (Hd @ W2 + b2 - Y) / len(X)
    gW2, gb2 = Hd.T @ err, err.sum(0)
    dz = (err @ W2.T) * (1 - Hd ** 2)
    gW1, gb1 = X.T @ dz, dz.sum(0)
    W1, b1, W2, b2 = W1 - 0.1 * gW1, b1 - 0.1 * gb1, W2 - 0.1 * gW2, b2 - 0.1 * gb2

perm = rng.permutation(H)                                          # relabel the hidden units
theta = (W1, b1, W2, b2)
theta_p = (W1[:, perm], b1[perm], W2[perm, :], b2)
print(f"trained loss {mlp_loss(*theta):.6f} | permuted loss {mlp_loss(*theta_p):.6f}")
for t in (0.0, 0.25, 0.5, 0.75, 1.0):                              # straight line between the two
    mix = [(1 - t) * p + t * q for p, q in zip(theta, theta_p)]
    print(f"  t = {t:4.2f}  loss = {mlp_loss(*mix):.6f}")

# ---------- 4. escaping the saddle of f = x^2 + y^4/4 - y^2/2 (Hessian diag(2, -1) at 0) ----------
saddle_grad = lambda w: np.array([2 * w[0], w[1] ** 3 - w[1]])
for y0 in (0.0, 1e-8, 1e-2):
    w, k = np.array([-1.8, y0]), 0
    while abs(w[1]) < 0.9 and k < 2000:
        w, k = w - 0.05 * saddle_grad(w), k + 1
    status = f"reached |y| > 0.9 after {k} steps" if abs(w[1]) >= 0.9 else f"stuck at {w.round(6) + 0.0} after {k} steps"
    print(f"start y0 = {y0:g}: {status}")
w = np.array([-1.8, 0.0])                                          # exactly on the ridge, plus small noise
noise_rng = np.random.default_rng(1)
for k in range(1, 2001):
    w = w - 0.05 * (saddle_grad(w) + 1e-3 * noise_rng.normal(size=2))
    if abs(w[1]) > 0.9:
        break
print(f"start y0 = 0 with gradient noise: reached |y| > 0.9 after {k} steps")

# ---------- 5. Random symmetric matrices: how often are all eigenvalues positive? ----------
for d in (1, 2, 4, 6, 8, 10):
    M = rng.normal(size=(20_000, d, d))
    S = (M + np.transpose(M, (0, 2, 1))) / 2
    frac = np.mean(np.linalg.eigvalsh(S).min(axis=1) > 0)
    print(f"d = {d:2d}: fraction positive definite = {frac:.4f}")`;

const LAND_OUT = String.raw`chord-inequality violations: convex 0 | non-convex 45354
GD end points, convex    : [1.1775, 1.1775, 1.1775, 1.1775]
GD end points, non-convex: [-1.3008, -1.3008, 1.1309, 1.1309] losses [-3.5139, -3.5139, -1.0702, -1.0702]
f(A) = 0.0, f(B) = 0.0, f(midpoint) = 1.0  -> not convex
Hessian eigenvalues at the origin: [-2.  2.] -> saddle
trained loss 0.015545 | permuted loss 0.015545
  t = 0.00  loss = 0.015545
  t = 0.25  loss = 0.135151
  t = 0.50  loss = 0.259372
  t = 0.75  loss = 0.136640
  t = 1.00  loss = 0.015545
start y0 = 0: stuck at [0. 0.] after 2000 steps
start y0 = 1e-08: reached |y| > 0.9 after 392 steps
start y0 = 0.01: reached |y| > 0.9 after 109 steps
start y0 = 0 with gradient noise: reached |y| > 0.9 after 216 steps
d =  1: fraction positive definite = 0.5026
d =  2: fraction positive definite = 0.1458
d =  4: fraction positive definite = 0.0021
d =  6: fraction positive definite = 0.0000
d =  8: fraction positive definite = 0.0000
d = 10: fraction positive definite = 0.0000`;

const SMOOTH_CODE = String.raw`import numpy as np

rng = np.random.default_rng(0)

# ---------- an L2-regularised logistic regression: convex and L-smooth ----------
N, d, lam = 200, 5, 0.1
X = rng.normal(size=(N, d))
y = (X @ rng.normal(size=d) + 0.5 * rng.normal(size=N) > 0).astype(float)


def loss(w):
    z = X @ w
    # log(1 + e^z) - y z, written stably with logaddexp (see the numerical-stability program)
    return np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * w @ w


def grad(w):
    p = 1 / (1 + np.exp(-(X @ w)))
    return X.T @ (p - y) / N + lam * w


# Hessian = X^T diag(p(1-p)) X / N + lam I, and p(1-p) <= 1/4, so the smoothness constant is
L = np.linalg.eigvalsh(X.T @ X / N).max() / 4 + lam
print(f"smoothness constant L = {L:.4f}  ->  step 1/L = {1 / L:.4f}, stability limit 2/L = {2 / L:.4f}")

# reference minimiser by Newton's method (the problem is small and strongly convex)
w_star = np.zeros(d)
for _ in range(20):
    p = 1 / (1 + np.exp(-(X @ w_star)))
    Hs = X.T @ (X * (p * (1 - p))[:, None]) / N + lam * np.eye(d)
    w_star -= np.linalg.solve(Hs, grad(w_star))
f_star = loss(w_star)
print(f"f* = {f_star:.6f}, |grad f(w*)| = {np.linalg.norm(grad(w_star)):.1e}, "
      f"largest Hessian eigenvalue at w* = {np.linalg.eigvalsh(Hs).max():.4f} (<= L)")

# ---------- gradient descent with different step sizes ----------
w0 = np.zeros(d)
R2 = np.sum((w0 - w_star) ** 2)
for name, eta in [("1/L", 1 / L), ("1.9/L", 1.9 / L), ("2.5/L", 2.5 / L)]:
    w, gaps, monotone = w0.copy(), [], True
    prev = loss(w)
    for t in range(1, 101):
        w = w - eta * grad(w)
        cur = loss(w)
        monotone &= cur <= prev + 1e-15
        prev = cur
        gaps.append(cur - f_star)
    print(f"eta = {name:5s}: gap after 10/100 steps = {gaps[9]:.2e} / {gaps[99]:.2e}, monotone = {monotone}")
    if name == "1/L":
        bound_ok = all(g <= L * R2 / (2 * t) for t, g in enumerate(gaps, start=1))
        print(f"           convex-rate bound f(w_t) - f* <= L |w0 - w*|^2 / (2t) holds for all t: {bound_ok}")

# ---------- the descent lemma, checked at random points ----------
worst = -np.inf
for _ in range(1000):
    w = rng.normal(scale=3, size=d)
    g = grad(w)
    guaranteed = loss(w) - g @ g / (2 * L)              # f(w - g/L) <= f(w) - |g|^2 / (2L)
    worst = max(worst, loss(w - g / L) - guaranteed)
print(f"descent lemma: max of f(w - g/L) - [f(w) - |g|^2/(2L)] over 1000 points = {worst:.2e} (must be <= 0)")`;

const SMOOTH_OUT = String.raw`smoothness constant L = 0.4065  ->  step 1/L = 2.4600, stability limit 2/L = 4.9199
f* = 0.456450, |grad f(w*)| = 2.9e-17, largest Hessian eigenvalue at w* = 0.3361 (<= L)
eta = 1/L  : gap after 10/100 steps = 5.30e-08 / 5.55e-17, monotone = True
           convex-rate bound f(w_t) - f* <= L |w0 - w*|^2 / (2t) holds for all t: True
eta = 1.9/L: gap after 10/100 steps = 1.69e-08 / 5.55e-17, monotone = True
eta = 2.5/L: gap after 10/100 steps = 2.41e-03 / 2.47e-02, monotone = False
descent lemma: max of f(w - g/L) - [f(w) - |g|^2/(2L)] over 1000 points = -5.02e-02 (must be <= 0)`;

const STAB_CODE = String.raw`import numpy as np

np.seterr(all="ignore")                     # let overflow / invalid produce inf and nan silently
fmt = lambda a: "[" + ", ".join(f"{v:.6g}" for v in np.asarray(a, dtype=np.float64)) + "]"


def to_bfloat16(x):
    """Round float32 values to bfloat16 (keep the top 16 bits, round to nearest even)."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


# ---------- 1. what each format can represent ----------
print("format    max          smallest normal  machine eps")
for name, t in [("float64", np.float64), ("float32", np.float32), ("float16", np.float16)]:
    fi = np.finfo(t)
    print(f"{name:8s}  {float(fi.max):.3e}    {float(fi.tiny):.3e}        {float(fi.eps):.3e}")
print(f"bfloat16  {float(to_bfloat16(np.float32(3.3895314e38))):.3e}    {2.0 ** -126:.3e}        {2.0 ** -7:.3e}")
print("float16: 65504 + 16 =", np.float16(65504) + np.float16(16), "  smallest subnormal =", f"{float(np.float16(2 ** -24)):.3e}")
print("bfloat16: 256 + 1 =", to_bfloat16(np.float32(256) + np.float32(1)), "  1 + 2^-8 =", to_bfloat16(np.float32(1 + 2 ** -8)))

# ---------- 2. softmax: naive overflows, shifted version is exact ----------
z = np.array([1000.0, 1001.0, 1002.0])
naive = np.exp(z) / np.exp(z).sum()
stable = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print("naive softmax :", fmt(naive))
print("stable softmax:", fmt(stable))

# ---------- 3. log-sum-exp ----------
def logsumexp(v):
    m = v.max()
    return m + np.log(np.sum(np.exp(v - m)))

print("naive log(sum(exp(z)))      :", np.log(np.sum(np.exp(z))))
print("stable logsumexp(z)         :", round(logsumexp(z), 6))
print("naive for z = [-1000, -1001]:", np.log(np.sum(np.exp(np.array([-1000.0, -1001.0])))),
      "  stable:", round(logsumexp(np.array([-1000.0, -1001.0])), 6))

# ---------- 4. sigmoid, softplus and binary cross-entropy from logits ----------
def sigmoid_stable(x):
    e = np.exp(-np.abs(x))                  # always in (0, 1]: cannot overflow
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))

def softplus_stable(x):                     # log(1 + e^x)
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

x = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
print("naive sigmoid e^x/(1+e^x):", fmt(np.exp(x) / (1 + np.exp(x))))
print("stable sigmoid           :", fmt(sigmoid_stable(x)))
print("naive softplus log(1+e^x):", fmt(np.log(1 + np.exp(x))))
print("stable softplus          :", fmt(softplus_stable(x)))
logit = -120.0                              # a confident, wrong prediction for a positive example
p = np.float32(1 / (1 + np.exp(-logit)))    # the probability underflows to 0 in float32
print(f"BCE = -log(p) with float32 p = {p}: {-np.log(p)}   BCE = softplus(-logit): {softplus_stable(-logit)}")

# ---------- 5. catastrophic cancellation ----------
print("float64: (1 + 1e-16) - 1 =", (1.0 + 1e-16) - 1.0, "  log(1 + 1e-10) =", np.log(1 + 1e-10), "  log1p =", np.log1p(1e-10))
rng = np.random.default_rng(0)
data = (1e4 + rng.normal(size=10_000)).astype(np.float32)       # true variance is about 1
naive_var = np.mean(data * data) - np.mean(data) ** 2           # E[x^2] - E[x]^2 in float32
two_pass = np.mean((data - data.mean()) ** 2)                   # subtract the mean first
print(f"variance in float32: naive E[x^2]-E[x]^2 = {naive_var:.4f}   two-pass = {two_pass:.4f}   "
      f"float64 reference = {np.var(data.astype(np.float64)):.4f}")

# ---------- 6. epsilon in denominators ----------
v = np.zeros(3)
print("normalise a zero vector:", fmt(v / np.linalg.norm(v)), "  with eps:", fmt(v / (np.linalg.norm(v) + 1e-8)))

# ---------- 7. float16 underflow of small gradients, and loss scaling ----------
grads = np.array([1e-3, 1e-5, 1e-7, 1e-8, 1e-9])
print("gradients in float16           :", fmt(grads.astype(np.float16)))
scale = 1024.0
recovered = (grads * scale).astype(np.float16).astype(np.float64) / scale
print("scaled by 1024, then unscaled  :", fmt(recovered))
print("relative error after scaling   :", fmt(np.abs(recovered - grads) / grads))`;

const STAB_OUT = String.raw`format    max          smallest normal  machine eps
float64   1.798e+308    2.225e-308        2.220e-16
float32   3.403e+38    1.175e-38        1.192e-07
float16   6.550e+04    6.104e-05        9.766e-04
bfloat16  3.390e+38    1.175e-38        7.812e-03
float16: 65504 + 16 = inf   smallest subnormal = 5.960e-08
bfloat16: 256 + 1 = 256.0   1 + 2^-8 = 1.0
naive softmax : [nan, nan, nan]
stable softmax: [0.0900306, 0.244728, 0.665241]
naive log(sum(exp(z)))      : inf
stable logsumexp(z)         : 1002.407606
naive for z = [-1000, -1001]: -inf   stable: -999.686738
naive sigmoid e^x/(1+e^x): [0, 9.35762e-14, 0.5, 1, nan]
stable sigmoid           : [0, 9.35762e-14, 0.5, 1, 1]
naive softplus log(1+e^x): [0, 9.34808e-14, 0.693147, 30, inf]
stable softplus          : [0, 9.35762e-14, 0.693147, 30, 800]
BCE = -log(p) with float32 p = 0.0: inf   BCE = softplus(-logit): 120.0
float64: (1 + 1e-16) - 1 = 0.0   log(1 + 1e-10) = 1.000000082690371e-10   log1p = 9.999999999500001e-11
variance in float32: naive E[x^2]-E[x]^2 = -8.0000   two-pass = 0.9962   float64 reference = 0.9962
normalise a zero vector: [nan, nan, nan]   with eps: [0, 0, 0]
gradients in float16           : [0.0010004, 1.00136e-05, 1.19209e-07, 0, 0]
scaled by 1024, then unscaled  : [0.0010004, 9.99868e-06, 1.00001e-07, 1.00117e-08, 9.8953e-10]
relative error after scaling   : [0.000404358, 0.000132084, 7.61449e-06, 0.00117177, 0.0104698]`;

const TORCH_CODE = String.raw`import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---------- 1. floating-point formats ----------
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):15s} max {fi.max:.3e}  smallest normal {fi.tiny:.3e}  eps {fi.eps:.3e}")

# ---------- 2. built-in stable reductions ----------
z = torch.tensor([1000.0, 1001.0, 1002.0])
print("naive softmax  :", (z.exp() / z.exp().sum()).tolist())
print("torch.softmax  :", [round(v, 6) for v in torch.softmax(z, 0).tolist()])
print("log_softmax    :", [round(v, 6) for v in torch.log_softmax(z, 0).tolist()])
print("logsumexp      :", round(torch.logsumexp(z, 0).item(), 6))

# ---------- 3. sigmoid + BCE vs BCE-with-logits for a confident wrong prediction ----------
for fused in (False, True):
    logit = torch.tensor([-120.0], requires_grad=True)
    target = torch.tensor([1.0])
    if fused:
        loss = F.binary_cross_entropy_with_logits(logit, target)
    else:
        loss = F.binary_cross_entropy(torch.sigmoid(logit), target)
    loss.backward()
    name = "BCEWithLogits      " if fused else "BCE(sigmoid(logit))"
    print(f"{name}: loss = {loss.item():.4f}, dL/dlogit = {logit.grad.item():.4f}")

# ---------- 4. float16 gradient underflow and loss scaling ----------
w = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)
x = torch.tensor([1e-4], dtype=torch.float16)
loss = (w * x * x).sum()                     # d loss / d w = x^2 = 1e-8
loss.backward()
print("fp16 gradient without scaling:", w.grad.item())
w.grad = None
scale = 2.0 ** 14                            # must keep scale * gradients below the fp16 max 65504
loss = (w * x * x).sum()                     # rebuild the graph (backward freed the old one)
(loss * scale).backward()                    # what torch.amp.GradScaler does internally
print(f"fp16 gradient with scale 2^14, unscaled in fp32: {w.grad.float().item() / scale:.4e}")

# ---------- 5. L2 penalty in the loss == weight_decay in SGD ----------
X, y = torch.randn(64, 3), torch.randn(64)
w1 = torch.zeros(3, requires_grad=True)
w2 = torch.zeros(3, requires_grad=True)
lam, lr = 0.1, 0.05
opt1 = torch.optim.SGD([w1], lr=lr)
opt2 = torch.optim.SGD([w2], lr=lr, weight_decay=lam)
for _ in range(100):
    opt1.zero_grad()
    (F.mse_loss(X @ w1, y) + 0.5 * lam * (w1 ** 2).sum()).backward()   # penalty in the objective
    opt1.step()
    opt2.zero_grad()
    F.mse_loss(X @ w2, y).backward()                                    # penalty in the optimizer
    opt2.step()
print("penalty vs weight_decay, max |difference|:", f"{(w1 - w2).abs().max().item():.1e}")

# ---------- 6. constraint instead of penalty: projected gradient descent onto |w| <= r ----------
r = 0.1
w3 = torch.zeros(3, requires_grad=True)
opt3 = torch.optim.SGD([w3], lr=lr)
for _ in range(500):
    opt3.zero_grad()
    F.mse_loss(X @ w3, y).backward()
    opt3.step()
    with torch.no_grad():                    # project back onto the feasible ball
        n = w3.norm()
        if n > r:
            w3.mul_(r / n)
w_free = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
print(f"|w| unconstrained = {w_free.norm().item():.4f}, with L2 penalty = {w1.norm().item():.4f}, "
      f"projected onto |w| <= {r} = {w3.norm().item():.4f}")`;

const TORCH_OUT = String.raw`torch.float32   max 3.403e+38  smallest normal 1.175e-38  eps 1.192e-07
torch.float16   max 6.550e+04  smallest normal 6.104e-05  eps 9.766e-04
torch.bfloat16  max 3.390e+38  smallest normal 1.175e-38  eps 7.812e-03
naive softmax  : [nan, nan, nan]
torch.softmax  : [0.090031, 0.244728, 0.665241]
log_softmax    : [-2.407606, -1.407606, -0.407606]
logsumexp      : 1002.407593
BCE(sigmoid(logit)): loss = 100.0000, dL/dlogit = -0.0000
BCEWithLogits      : loss = 120.0000, dL/dlogit = -1.0000
fp16 gradient without scaling: 0.0
fp16 gradient with scale 2^14, unscaled in fp32: 1.0004e-08
penalty vs weight_decay, max |difference|: 0.0e+00
|w| unconstrained = 0.1659, with L2 penalty = 0.1572, projected onto |w| <= 0.1 = 0.1000`;

const EX_CODE = String.raw`import numpy as np

np.seterr(all="ignore")


def cross_entropy_naive(logits, label):
    p = np.exp(logits) / np.exp(logits).sum()
    return -np.log(p[label])


def cross_entropy_stable(logits, label):
    """Return (loss, dloss/dlogits) using log_softmax = z - logsumexp(z)."""
    m = logits.max()
    lse = m + np.log(np.sum(np.exp(logits - m)))
    loss = lse - logits[label]
    grad = np.exp(logits - lse)                 # softmax, computed from the stable log-probabilities
    grad[label] -= 1.0                          # p - one_hot(label)
    return loss, grad


tests = [np.array([2.0, 1.0, 0.1]), np.array([1000.0, 0.0, -1000.0]), np.array([-1000.0, 0.0, 1000.0])]
for z in tests:
    loss, g = cross_entropy_stable(z, 0)
    print(f"logits {z}: naive {cross_entropy_naive(z, 0)}  stable {loss:.6f}  grad {np.round(g, 6) + 0.0}")

# finite-difference check of the gradient on the first example
z, h = tests[0], 1e-6
num = np.array([(cross_entropy_stable(z + h * e, 0)[0] - cross_entropy_stable(z - h * e, 0)[0]) / (2 * h) for e in np.eye(3)])
print(f"max |analytic - numerical| = {np.abs(num - cross_entropy_stable(z, 0)[1]).max():.1e}")`;

const EX_OUT = String.raw`logits [2.  1.  0.1]: naive 0.4170300162778335  stable 0.417030  grad [-0.340999  0.242433  0.098566]
logits [ 1000.     0. -1000.]: naive nan  stable 0.000000  grad [0. 0. 0.]
logits [-1000.     0.  1000.]: naive inf  stable 2000.000000  grad [-1.  0.  1.]
max |analytic - numerical| = 1.2e-10`;

const TF_CODE = String.raw`import numpy as np
import tensorflow as tf

# ---------- stable primitives ----------
z = tf.constant([1000.0, 1001.0, 1002.0])
print("logsumexp:", tf.math.reduce_logsumexp(z).numpy())
print("softmax  :", tf.nn.softmax(z).numpy())
print("BCE from logits:", tf.nn.sigmoid_cross_entropy_with_logits(labels=[1.0], logits=[-120.0]).numpy())

# ---------- mixed precision: float16 compute, float32 variables, automatic loss scaling ----------
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(20,)),
    tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),          # penalty added to the objective
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),       # projection after each update
    ),
    tf.keras.layers.Dense(1, dtype="float32"),                     # keep the output layer in float32
])
model.summary()

# Under the mixed_float16 policy, Keras wraps the optimizer in a LossScaleOptimizer automatically
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))   # stable: works on logits

rng = np.random.default_rng(0)
X = rng.normal(size=(512, 20)).astype("float32")
y = (X[:, 0] + X[:, 1] > 0).astype("float32")
model.fit(X, y, epochs=3, batch_size=64, verbose=2)`;


const lesson: Lesson = {
  id: "m02/optimization-math",
  title: "Optimization: Objectives, Convexity and Minima",
  summary:
    "Training a model as an optimization problem: objectives and empirical risk, constraints vs penalties, convex sets and functions (with proofs), why neural networks are non-convex, local minima, saddle points and plateaus, optimality conditions, L-smoothness and the 1/L step size, floating-point stability, and regularization as part of the objective.",
  minutes: 85,
  level: "intermediate",
  prerequisites: ["m02/derivatives-chain-rule", "m02/jacobians-hessians", "m02/eigen-svd", "m02/norms-spaces"],
  next: ["m06/gradient-descent", "m06/momentum-adaptive", "m07/regularization", "m02/mle-map"],
  terms: ["objective-function", "empirical-risk", "convex-set", "convex-function", "jensen-inequality", "local-minimum", "global-minimum", "saddle-point", "l-smooth", "numerical-stability", "log-sum-exp", "catastrophic-cancellation", "machine-epsilon", "loss-scaling", "regularization"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`"Training a neural network" means solving an **optimization problem**: find the parameters $\theta$ (all weights and biases) that make one number, the **objective** $J(\theta)$, as small as possible. The objective is usually the average training loss plus a regularization term. Everything else — architecture, optimizer, learning-rate schedule, float16 or float32 — serves to solve that problem well enough, fast enough, and in a way that works on new data.`,
      },
      {
        t: "p",
        text: String.raw`For **convex** (single-bowl) problems, classical optimization has a complete theory: every local minimum is global, and simple algorithms provably converge at known speeds. Linear regression, logistic regression and support vector machines are convex. Neural networks are not. We show exactly why (a symmetry argument), what their landscape looks like instead (many equivalent minima, huge numbers of saddle points, flat plateaus), and which convex ideas still help — above all **smoothness**, which tells you how large a learning rate is safe.`,
      },
      {
        t: "p",
        text: String.raw`The last part is **numerical stability**. Computers store numbers with limited precision. A mathematically correct softmax can return NaN, a variance can come out negative, and a gradient of $10^{-8}$ becomes exactly zero in float16. The log-sum-exp trick, stable sigmoid and softplus, and loss scaling are part of solving the problem correctly.`,
      },
      {
        t: "callout",
        kind: "key",
        title: "What this lesson covers",
        text: String.raw`Setting up the problem and empirical risk minimisation; constraints vs penalties; convex sets and functions (definition, first- and second-order tests, Jensen) and the proof that convex problems have no false local minima; why neural networks are non-convex; local minima, saddles and plateaus; optimality conditions; $L$-smoothness and the step size $\eta \le 1/L$ ($\eta$ = learning rate); floating-point traps and fixes; regularization as part of the objective.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Derivatives, Partial Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — gradients, and the fact that $-\nabla f$ is the direction of steepest descent.`,
          String.raw`[Jacobians, Hessians and Computational Graphs](/learn/m02/jacobians-hessians) — the Hessian (the matrix of second derivatives, i.e. curvature), the second-order Taylor expansion, and classifying critical points by Hessian eigenvalues. Used throughout.`,
          String.raw`[Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd) — a symmetric matrix is positive semidefinite ($v^\top Av \ge 0$ for all $v$: it curves up or stays flat in every direction) exactly when all its eigenvalues are $\ge 0$.`,
          String.raw`[Vector Spaces, Norms and Distances](/learn/m02/norms-spaces) — the $L_1$ and $L_2$ norms used in penalties and constraints.`,
          String.raw`[Gradient Descent](/learn/m06/gradient-descent) is the companion lesson: it derives the update rule and the condition $\eta < 2/\lambda_{\max}$ on quadratics. We refer to it rather than repeat it.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Finding the lowest point in a foggy landscape",
        text: "You are dropped into mountains in thick fog and must reach the lowest point. You can only feel the slope under your feet. If the terrain is one smooth bowl (a **convex** landscape), walking downhill always gets you to the bottom, and wherever the ground is flat you have arrived. Real terrain has many valleys (local minima), mountain passes that are flat but lead downhill on two sides (saddle points), and wide flat plains where you can hardly tell which way is down (plateaus). Stride length matters too: in a narrow gully a long stride carries you up the far wall (learning rate too large).",
      },
      {
        t: "p",
        text: String.raw`**Convexity** is the maths version of "one bowl": the straight line between any two points on the graph lies on or above the graph. That rules out separate valleys, which is why convex problems count as "solved".`,
      },
      {
        t: "p",
        text: String.raw`**Neural networks are not bowls.** Swap two hidden neurons (with their incoming and outgoing weights) and the network computes exactly the same function. So every good solution has many identical twins elsewhere in parameter space. Walk straight between two twins and the loss rises in the middle — a bowl can't do that. Yet training works well in practice: in very high dimensions most flat points are saddles you slide off, not traps, and many valleys are about equally deep.`,
      },
      {
        t: "p",
        text: String.raw`**Smoothness** limits how fast the slope can change. If it changes by at most $L$ per unit distance, a step of size $1/L$ downhill is guaranteed to go down. **Numerical stability** means your arithmetic doesn't betray you: $e^{1000}$ is too large for any floating-point format, so a good program never computes it, even when the formula says to.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "From a learning task to an optimization problem" },
      {
        t: "steps",
        items: [
          { title: "Choose a model family", text: String.raw`A function $f(x;\theta)$ with parameters $\theta\in\mathbb{R}^d$ ($d$ numbers: all weights and biases).` },
          { title: "Choose a per-example loss", text: String.raw`$\ell(\hat y, y)\ge0$ says how bad prediction $\hat y$ is when the truth is $y$: squared error for regression, cross-entropy for classification. See [classification losses](/learn/m05/classification-losses).` },
          { title: "Average over the data: the empirical risk", text: String.raw`$\hat R(\theta) = \frac1N\sum_{i=1}^N \ell(f(x_i;\theta), y_i)$ — the average loss on the $N$ training examples, a stand-in for the true risk on unseen data.` },
          { title: "Add preferences: the regularized objective", text: String.raw`$J(\theta) = \hat R(\theta) + \lambda\,\Omega(\theta)$. $\Omega$ penalises unwanted parameters (large or non-sparse weights); $\lambda\ge0$ sets how much. Hard requirements become constraints $\theta\in\mathcal{C}$ instead.` },
          { title: "Pick an algorithm and its hyper-parameters", text: String.raw`Gradient descent or a variant, a step size (from the problem's smoothness, or tuned) and an initialization. For non-convex $J$ the result depends on both.` },
          { title: "Implement it in finite precision", text: String.raw`Compute $J$ and $\nabla J$ with stable formulas (log-sum-exp, fused sigmoid + cross-entropy, epsilons in denominators), in a floating-point format whose range and precision fit the values.` },
          { title: "Diagnose the landscape", text: String.raw`If training stalls: is the gradient small because of a minimum, a plateau or a saddle? Is the loss diverging because $\eta$ is past the stability limit? Do NaNs come from overflow? The tools below answer each question.` },
        ],
      },
      { t: "h", text: "Decision guide: what is happening when the gradient is (nearly) zero?" },
      {
        t: "table",
        head: ["Situation", "Gradient", "Hessian eigenvalues", "Loss nearby", "What to do"],
        rows: [
          ["Local (or global) minimum", "$0$", "all $\\ge 0$", "rises in every direction", "stop, or reach a better basin via noise, restarts, schedules"],
          ["Strict saddle point", "$0$", "at least one $< 0$", "falls along some directions", "keep going: noise, momentum or perturbations escape it"],
          ["Plateau", "tiny but non-zero", "all near $0$", "almost flat over a large region", "better initialization, normalization, non-saturating activations, adaptive optimizers"],
          ["Local maximum", "$0$", "all $\\le 0$", "falls in every direction", "unstable: gradient descent leaves immediately unless started exactly there"],
        ],
      },
      {
        t: "callout",
        kind: "note",
        text: "For a convex objective only the first row can happen, and every such point is a global minimum. That is what \"convex problems have no false local minima\" means in practice.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Formulating an optimization problem" },
      {
        t: "math",
        tex: String.raw`\min_{\theta\in\mathbb{R}^d}\; J(\theta) \quad \text{subject to}\quad g_i(\theta)\le 0\ (i = 1..p), \qquad h_j(\theta) = 0\ (j = 1..q)`,
      },
      {
        t: "p",
        text: String.raw`$J:\mathbb{R}^d\to\mathbb{R}$ (it maps $d$ numbers to one number) is the **objective function**. The $g_i$ and $h_j$ define the **feasible set** $\mathcal{C} = \{\theta: g_i(\theta)\le0,\ h_j(\theta) = 0\}$ — the allowed parameters. A point $\theta^\star\in\mathcal{C}$ is a **global minimum** if $J(\theta^\star)\le J(\theta)$ for every allowed $\theta$, and a **local minimum** if this holds for allowed $\theta$ within some distance $r > 0$ of it. The set of global minimisers, $\arg\min J$, may be empty (the lowest value is never reached, e.g. $e^{-\theta}$), one point, or a whole set. Maximising $J$ equals minimising $-J$, so we always talk about minimising.`,
      },
      {
        t: "p",
        text: String.raw`**Vocabulary.** **Loss** usually means the per-example $\ell(\hat y, y)$; **cost** the average over a dataset or batch; **objective** whatever is actually minimised, regularization included. People mix these up; what matters is knowing exactly which function your optimizer differentiates.`,
      },
      { t: "h", text: "2. Empirical risk minimisation" },
      {
        t: "p",
        text: String.raw`Data $(x, y)$ come from an unknown distribution $\mathcal{D}$. We want to minimise the **(true) risk**, the expected loss on new data. We can only compute the **empirical risk** on the $N$ training examples:`,
      },
      {
        t: "math",
        tex: String.raw`R(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\ell(f(x;\theta), y)\big], \qquad \hat R_N(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell\big(f(x_i;\theta), y_i\big)`,
      },
      {
        t: "p",
        text: String.raw`For a **fixed** $\theta$, $\hat R_N(\theta)$ is an unbiased estimate of $R(\theta)$ and approaches it as $N\to\infty$ (law of large numbers). **Empirical risk minimisation** (ERM) picks $\hat\theta = \arg\min\hat R_N$. But $\hat\theta$ is chosen *using* the training data, so $\hat R_N(\hat\theta)$ looks too good. The difference $R(\hat\theta) - \hat R_N(\hat\theta)$ is the **generalization gap**: a flexible model can drive $\hat R_N$ to zero while $R$ stays large (overfitting, see [training basics](/learn/m03/training-basics)). So we rarely minimise $\hat R_N$ alone, and we watch a validation loss: optimization is a means to low $R$, not the goal. Mini-batch SGD can also be seen as optimising $R$ directly, since in the first epoch each fresh mini-batch gives an unbiased estimate of $\nabla R$.`,
      },
      { t: "h", text: "3. Constraints versus penalties" },
      {
        t: "p",
        text: String.raw`A wish like "keep the weights small" can be a **constraint** ($\lVert\theta\rVert_2\le r$, length at most $r$) or a **penalty** added to the objective ($+\tfrac\lambda2\lVert\theta\rVert_2^2$). **Lagrange multipliers** link the two. For $\min J(\theta)$ subject to $g(\theta)\le0$, form the Lagrangian $\mathcal{L}(\theta,\mu) = J(\theta) + \mu\,g(\theta)$ with $\mu\ge0$. Under mild conditions a solution satisfies the **KKT conditions**:`,
      },
      {
        t: "math",
        tex: String.raw`\nabla J(\theta^\star) + \mu^\star\nabla g(\theta^\star) = 0,\qquad g(\theta^\star)\le0,\qquad \mu^\star\ge0,\qquad \mu^\star g(\theta^\star) = 0`,
        caption: "Stationarity, primal feasibility, dual feasibility, complementary slackness. If the constraint is inactive (g < 0), then μ* = 0 and it plays no role.",
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Minimise $(w-3)^2$ subject to $|w|\le1$. The free minimum $w = 3$ is not allowed, so the constraint is active and $w^\star = 1$. Stationarity at $w = 1$ with $g(w) = w^2 - 1$: $2(1-3) + \mu\cdot2(1) = 0$, so $\mu^\star = 2$. Now the **penalised** problem $\min (w-3)^2 + \lambda w^2$: setting the derivative $2(w-3) + 2\lambda w$ to zero gives $w = 3/(1+\lambda)$ — $3$ for $\lambda = 0$, $2$ for $\lambda = 0.5$, exactly $1$ for $\lambda = 2 = \mu^\star$. For convex problems this always works: each constraint radius $r$ matches some penalty weight $\lambda$ and vice versa, though the match depends on the data and is rarely known in advance.`,
      },
      {
        t: "p",
        text: String.raw`**In practice** deep learning mostly uses penalties: the problem stays unconstrained and gradient descent works unchanged. Constraints appear as (i) **projections** after each step — projected gradient descent, $\theta\leftarrow\Pi_{\mathcal{C}}(\theta - \eta\nabla J)$, where $\Pi_{\mathcal{C}}$ moves $\theta$ to the nearest allowed point (e.g. max-norm limits on each neuron's incoming weights, used with dropout by Srivastava et al., 2014); and (ii) **reparametrisations** that make the constraint automatic: a positive variance as $\sigma = \operatorname{softplus}(s)$ or $e^s$, a probability vector as $\operatorname{softmax}(z)$, an orthogonal matrix via a matrix exponential.`,
      },
      { t: "h", text: "4. Convex sets and convex functions" },
      {
        t: "p",
        text: String.raw`A set $\mathcal{C}$ is **convex** if the straight segment between any two of its points stays inside: $x, y\in\mathcal{C},\ t\in[0,1] \Rightarrow tx + (1-t)y\in\mathcal{C}$. Examples: hyperplanes, half-spaces $\{a^\top x\le b\}$, balls $\{\lVert x\rVert\le r\}$, the probability simplex, and any intersection of convex sets. Not convex: a ring, the vectors with at most $k$ non-zero entries, two separate balls.`,
      },
      {
        t: "p",
        text: String.raw`A function $f$ defined on a convex set is **convex** if its graph lies on or below every chord:`,
      },
      {
        t: "math",
        tex: String.raw`f\big(tx + (1-t)y\big) \;\le\; t\,f(x) + (1-t)\,f(y) \qquad \text{for all } x, y,\ t\in[0,1]`,
      },
      {
        t: "p",
        text: String.raw`In words: the graph never rises above the straight line joining two of its points. It is **strictly convex** if the inequality is strict for $x\ne y$, $t\in(0,1)$, and **$\mu$-strongly convex** if $f(x) - \tfrac\mu2\lVert x\rVert^2$ is still convex (it curves up at least as much as a parabola of curvature $\mu$). For differentiable $f$ there are two equivalent tests:`,
      },
      {
        t: "math",
        tex: String.raw`\textbf{first order: } f(y)\;\ge\; f(x) + \nabla f(x)^\top(y - x)\ \ \forall x,y \qquad\qquad \textbf{second order: } \nabla^2 f(x)\succeq 0\ \ \forall x`,
        caption: String.raw`First order: every tangent plane is a global under-estimator. Second order: the Hessian is positive semidefinite everywhere ($\nabla^2 f\succeq\mu I$ for $\mu$-strong convexity).`,
      },
      {
        t: "p",
        text: String.raw`**Examples.** Least squares $\tfrac1N\lVert Xw - y\rVert^2$ has Hessian $\tfrac2N X^\top X\succeq0$: convex. Logistic regression's loss $\tfrac1N\sum_i[\log(1+e^{z_i}) - y_iz_i]$ with $z = Xw$ has Hessian $\tfrac1N X^\top SX$, $S = \operatorname{diag}(p_i(1-p_i))$, also $\succeq0$: convex. Norms, $\max(0, 1 - yz)$ (hinge), $\log\sum_i e^{z_i}$ (log-sum-exp) and $-\log\operatorname{softmax}(z)_y$ are convex in $z$. Non-negative sums and pointwise maxima of convex functions are convex, and so is a convex function of a linear map — so every linear model with a convex loss is a convex problem. Feeding a convex loss a *non-linear* network $f(x;\theta)$ usually destroys convexity in $\theta$.`,
      },
      {
        t: "p",
        text: String.raw`**Jensen's inequality.** The same definition for many points with weights $t_i\ge0$, $\sum t_i = 1$, and in the limit for expectations — "$f$ of the average ≤ average of $f$":`,
      },
      {
        t: "math",
        tex: String.raw`f\Big(\sum_i t_i x_i\Big)\le\sum_i t_i f(x_i), \qquad f\big(\mathbb{E}[X]\big)\le\mathbb{E}\big[f(X)\big]`,
      },
      {
        t: "p",
        text: String.raw`For $f(x) = x^2$ and $X$ uniform on $\{0, 2\}$: $(\mathbb{E}X)^2 = 1\le\mathbb{E}[X^2] = 2$; the gap is the variance. Jensen is behind the non-negativity of KL divergence, the evidence lower bound of VAEs, and why averaging weights of models trained from the same start can help (loss of the average ≤ average loss, if the loss is convex between them).`,
      },
      { t: "h", text: "5. Why convex problems have no spurious local minima" },
      {
        t: "p",
        text: String.raw`**Theorem.** If $f$ is convex on a convex set $\mathcal{C}$, every local minimum of $f$ over $\mathcal{C}$ is a global minimum.`,
      },
      {
        t: "p",
        text: String.raw`**Proof.** Let $x$ be a local minimum: $f(x)\le f(z)$ for every allowed $z$ with $\lVert z - x\rVert < r$. Suppose some allowed $y$ had $f(y) < f(x)$. For $t\in(0,1]$, the point $z_t = (1-t)x + ty$ is allowed (convex set) and, by convexity,`,
      },
      {
        t: "math",
        tex: String.raw`f(z_t)\;\le\;(1-t)f(x) + t f(y)\;=\;f(x) - t\big(f(x) - f(y)\big)\;<\;f(x)`,
      },
      {
        t: "p",
        text: String.raw`Choosing $t < r/\lVert y - x\rVert$ puts $z_t$ inside the neighbourhood where $x$ should be lowest, yet $f(z_t) < f(x)$ — a contradiction. So no such $y$ exists. $\square$ In plain words: in a bowl, any lower point elsewhere would drag the ground right next to you lower too.`,
      },
      {
        t: "p",
        text: String.raw`Two consequences. (i) The global minimisers form a convex set (if $x$ and $y$ both reach the minimum, so does every point between); with **strict** convexity there is at most one. (ii) For differentiable convex $f$, $\nabla f(x) = 0$ is **enough** for a global minimum: put $\nabla f(x) = 0$ into the first-order test to get $f(y)\ge f(x)$ for all $y$.`,
      },
      { t: "h", text: "6. Neural networks are non-convex" },
      {
        t: "p",
        text: String.raw`**The smallest example.** The "network" $\hat y = w_2w_1x$ (two linear layers of width 1) fitted to one example $x = 1$, $y = 1$ has loss $f(w_1, w_2) = (w_1w_2 - 1)^2$. Both $A = (1, 1)$ and $B = (-1, -1)$ are global minima with $f = 0$, but their midpoint $(0, 0)$ has $f = 1 > 0 = \tfrac12f(A) + \tfrac12f(B)$, breaking convexity. At the origin the gradient $2(w_1w_2 - 1)(w_2, w_1)$ is zero and the Hessian $\begin{pmatrix}0 & -2\\ -2 & 0\end{pmatrix}$ has eigenvalues $\pm2$: a saddle point. Even a linear network is non-convex in its parameters.`,
      },
      {
        t: "p",
        text: String.raw`**The general argument: permutation symmetry.** Take one hidden layer of $H$ units, $f(x) = W_2\,\phi(W_1x + b_1) + b_2$ ($\phi$ is the activation). Reorder the hidden units by any permutation $\pi$ — the rows of $W_1$ and $b_1$ and the columns of $W_2$ together — and the new parameters $\pi(\theta)$ compute **exactly the same function**, so $J(\pi(\theta)) = J(\theta)$. A minimiser $\theta^\star$ with distinct hidden units thus has up to $H!$ distinct copies. Now suppose $J$ were convex. Average $\theta^\star$ over all $H!$ permutations: by Jensen, $J(\bar\theta)\le\frac{1}{H!}\sum_\pi J(\pi(\theta^\star)) = J^\star$, so $\bar\theta$ would be optimal too. But in $\bar\theta$ every hidden unit has the **same** incoming weights (the average), so it acts like a network with a **single** hidden unit. Usually one unit fits worse than $H$ units — a contradiction. So $J$ is not convex. $\square$`,
      },
      {
        t: "p",
        text: String.raw`The from-scratch code checks this: an 8-unit tanh MLP fitted to $\sin x$ reaches loss $0.015545$; a permuted copy has the same $0.015545$; on the straight line between them the loss climbs to $0.259372$ at the midpoint — about 17 times higher. More symmetries add more equal minima: ReLU units can be rescaled ($w_{\text{in}}\to\alpha w_{\text{in}}$, $w_{\text{out}}\to w_{\text{out}}/\alpha$ for $\alpha > 0$), and tanh units can flip the sign of both incoming and outgoing weights.`,
      },
      { t: "h", text: "7. Local minima, saddle points and plateaus" },
      {
        t: "p",
        text: String.raw`A **critical** (stationary) point has $\nabla J = 0$ (flat ground). By the [Hessian classification](/learn/m02/jacobians-hessians): a strict local minimum if $\nabla^2J\succ0$ (curves up every way), a **strict saddle** if $\nabla^2J$ has at least one negative eigenvalue (curves down some way), degenerate if some eigenvalues are zero. A **plateau** is a large region where $\lVert\nabla J\rVert$ is tiny but the loss is far from good — e.g. when sigmoid or tanh units saturate (derivative near 0), or early in training after a poor initialization.`,
      },
      {
        t: "p",
        text: String.raw`**Saddles dominate in high dimension — a rule of thumb.** Think of the Hessian at a "random" critical point as a random symmetric matrix. For a minimum, **all** $d$ eigenvalues must be positive. With Gaussian entries (the Gaussian orthogonal ensemble), the fraction of positive definite matrices in 20,000 samples is $0.5026$ for $d = 1$, $0.1458$ for $d = 2$, $0.0021$ for $d = 4$, and $0$ (none) for $d = 6, 8, 10$ — faster than exponential decay. Dauphin et al. (2014, "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization") argued, from statistical physics and network experiments, that high-loss critical points are overwhelmingly saddles, while local minima tend to have loss close to the global minimum. This is a heuristic, not a theorem for every network, but it matches practice: training rarely gets stuck in a bad local minimum; it slows near saddles and on plateaus.`,
      },
      {
        t: "p",
        text: String.raw`**Escaping a saddle.** Near a saddle, $J(\theta^\star + \Delta)\approx J^\star + \tfrac12\sum_i\lambda_ic_i^2$, where $c_i$ is the offset along the $i$-th Hessian eigenvector and $\lambda_i$ its curvature. Gradient descent multiplies each $c_i$ by $1 - \eta\lambda_i$ per step, so along a negative $\lambda$ the offset **grows** by $1 + \eta|\lambda|$ per step. From distance $\varepsilon$, reaching distance $r$ takes about $\ln(r/\varepsilon)/\ln(1 + \eta|\lambda|)$ steps — logarithmic in $1/\varepsilon$, but infinite if $\varepsilon = 0$ exactly. On the surface $x^2 + y^4/4 - y^2/2$ (Hessian $\operatorname{diag}(2, -1)$ at the origin) with $\eta = 0.05$, the code measures:`,
      },
      {
        t: "table",
        head: ["Start $y_0$", "Steps to reach $|y| > 0.9$", "Linear prediction $\\ln(0.9/y_0)/\\ln 1.05$"],
        rows: [
          ["$10^{-2}$", "109", "92"],
          ["$10^{-8}$", "392", "375"],
          ["$0$ exactly", "never (converges to the saddle)", "∞"],
          ["$0$, with gradient noise of std $10^{-3}$", "216", "—"],
        ],
      },
      {
        t: "p",
        text: String.raw`(Measured counts are a bit larger than predicted because the cubic term $y^3$ slows the motion as $|y|$ nears 1.) Exact saddles are reached only from a "measure-zero" set of starts (practically never by chance). Lee et al. (2016, "Gradient Descent Only Converges to Minimizers") proved gradient descent with a small fixed step and random initialization almost surely avoids strict saddles; Jin et al. (2017, "How to Escape Saddle Points Efficiently") showed small random perturbations escape them in time polylogarithmic in the dimension. Mini-batch noise plays a similar role in practice, though slow escape can still cost many steps.`,
      },
      { t: "h", text: "8. Optimality conditions" },
      {
        t: "table",
        head: ["Condition", "Statement", "Type"],
        rows: [
          ["First-order necessary", "$\\theta^\\star$ local minimum of differentiable $J$ on $\\mathbb{R}^d$ $\\Rightarrow$ $\\nabla J(\\theta^\\star) = 0$", "necessary, not sufficient (saddles, maxima also satisfy it)"],
          ["Second-order necessary", "local minimum and $J\\in C^2$ $\\Rightarrow$ $\\nabla J = 0$ and $\\nabla^2J\\succeq0$", "necessary"],
          ["Second-order sufficient", "$\\nabla J(\\theta^\\star) = 0$ and $\\nabla^2J(\\theta^\\star)\\succ0$ $\\Rightarrow$ strict local minimum", "sufficient"],
          ["Convex case", "$J$ convex and $\\nabla J(\\theta^\\star) = 0$ $\\Rightarrow$ global minimum", "necessary and sufficient"],
          ["Constrained (KKT)", "stationarity of the Lagrangian + feasibility + $\\mu\\ge0$ + complementary slackness", "necessary under constraint qualifications; sufficient for convex problems"],
        ],
      },
      {
        t: "p",
        text: String.raw`**Why $\nabla J = 0$ is necessary.** If $g = \nabla J(\theta)\ne0$, then $J(\theta - tg) = J(\theta) - t\lVert g\rVert^2 + o(t)$ is below $J(\theta)$ for small $t > 0$: a small downhill step helps, so $\theta$ is not a local minimum. In deep learning we almost never reach an exact stationary point; training stops when the (validation) loss stops improving, usually with a gradient that is small compared with its start but not zero.`,
      },
      { t: "h", text: "9. The learning rate and the quadratic model" },
      {
        t: "p",
        text: String.raw`Near a minimum, any twice-differentiable ($C^2$) objective looks like a bowl-shaped quadratic: $J(\theta)\approx J^\star + \tfrac12(\theta - \theta^\star)^\top H(\theta - \theta^\star)$, with $H$ the Hessian. The [gradient descent lesson](/learn/m06/gradient-descent) shows gradient descent then converges iff $0 < \eta < 2/\lambda_{\max}(H)$, the best fixed rate is $2/(\lambda_{\min} + \lambda_{\max})$, and speed depends on the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ (steepest over flattest curvature). The next section turns this local fact into a global guarantee, valid far from the minimum and for non-convex functions.`,
      },
      { t: "h", text: "10. Lipschitz smoothness and the step size η ≤ 1/L" },
      {
        t: "p",
        text: String.raw`$J$ is **$L$-smooth** if its gradient is $L$-Lipschitz: $\lVert\nabla J(x) - \nabla J(y)\rVert\le L\lVert x - y\rVert$ for all $x, y$ — the slope changes by at most $L$ per unit distance. For twice-differentiable $J$ this means $-LI\preceq\nabla^2J(x)\preceq LI$ everywhere: no direction ever curves more than $L$. The key result is the **descent lemma**, a quadratic upper bound:`,
      },
      {
        t: "math",
        tex: String.raw`J(y)\;\le\; J(x) + \nabla J(x)^\top(y - x) + \frac{L}{2}\lVert y - x\rVert^2`,
      },
      {
        t: "p",
        text: String.raw`**Proof.** By the fundamental theorem of calculus along the segment $x + t(y - x)$,`,
      },
      {
        t: "math",
        tex: String.raw`J(y) - J(x) - \nabla J(x)^\top(y - x) = \int_0^1\big(\nabla J(x + t(y-x)) - \nabla J(x)\big)^\top(y - x)\,dt \;\le\; \int_0^1 L\,t\,\lVert y - x\rVert^2\,dt = \frac L2\lVert y - x\rVert^2`,
        caption: "Cauchy–Schwarz and the Lipschitz condition bound the integrand by ‖∇J(x+t(y−x)) − ∇J(x)‖·‖y−x‖ ≤ L t ‖y−x‖².",
      },
      {
        t: "p",
        text: String.raw`Substitute the gradient step $y = x - \eta\nabla J(x)$:`,
      },
      {
        t: "math",
        tex: String.raw`J\big(x - \eta\nabla J(x)\big)\;\le\;J(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\lVert\nabla J(x)\rVert^2 \qquad\overset{\eta = 1/L}{\Longrightarrow}\qquad J(x^+)\;\le\;J(x) - \frac{1}{2L}\lVert\nabla J(x)\rVert^2`,
      },
      {
        t: "p",
        text: String.raw`The guaranteed decrease $\eta(1 - L\eta/2)$ is positive for $0 < \eta < 2/L$ and largest at $\eta = 1/L$. So **with $\eta\le1/L$, every gradient step lowers the objective**, anywhere, convex or not. Two convergence results follow:`,
      },
      {
        t: "list",
        items: [
          String.raw`**Non-convex, $L$-smooth, never below $J_{\inf}$.** Adding up the decreases over $T$ steps: $\min_{t<T}\lVert\nabla J(\theta_t)\rVert^2\le 2L\,(J(\theta_0) - J_{\inf})/T$. Gradient descent finds a nearly flat point at rate $O(1/T)$ — possibly a saddle; nothing more is guaranteed without extra structure.`,
          String.raw`**Convex and $L$-smooth.** $J(\theta_T) - J^\star\le L\lVert\theta_0 - \theta^\star\rVert^2/(2T)$: the gap to the best value shrinks like $1/T$. If $J$ is also $\mu$-strongly convex, the gap shrinks geometrically, by at most $1 - \mu/L$ per step with $\eta = 1/L$ — $L/\mu$ is the condition number again.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Numerical check (L2-regularised logistic regression, $N = 200$, $d = 5$, $\lambda = 0.1$).** The Hessian is $\tfrac1N X^\top SX + \lambda I$ with $S_{ii} = p_i(1-p_i)\le\tfrac14$, so $L = \lambda_{\max}(X^\top X/N)/4 + \lambda = 0.4065$: $1/L = 2.46$, $2/L = 4.92$. The real largest curvature at the optimum is $0.3361 < L$ ($L$ is worst-case). With $\eta = 1/L$ and $1.9/L$ the loss falls every step and the bound $J(\theta_t) - J^\star\le L\lVert\theta_0 - \theta^\star\rVert^2/(2t)$ always holds. With $\eta = 2.5/L$ — above $2/\lambda_{\max}$ at the optimum, $5.95$ — the steps oscillate and the gap *grows* from $2.4\times10^{-3}$ after 10 steps to $2.5\times10^{-2}$ after 100. The descent lemma's decrease was checked at 1000 random points.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "What this means for neural networks",
        text: String.raw`Network losses are not globally $L$-smooth with a known $L$ (ReLU kinks, curvature growing with weight size), so learning rates are tuned, not set to $1/L$. The theory still explains what you see: a curvature-dependent stability limit exists; warm-up helps when early curvature is high; and in full-batch training the top Hessian eigenvalue often rises until it hovers around $2/\eta$ (the "edge of stability", Cohen et al., 2021).`,
      },
      { t: "h", text: "11. Numerical stability" },
      {
        t: "p",
        text: String.raw`A floating-point number is stored as $(-1)^s\times1.m\times2^{e}$: a sign bit, an exponent $e$ (sets the **range**) and a mantissa $m$ (sets the **precision**). **Machine epsilon** $\epsilon$ is the gap between 1 and the next storable number; every result is rounded with relative error at most $\epsilon/2$. Too-large numbers **overflow** to $\pm\infty$; numbers below the smallest normal value lose precision as **subnormals** and then **underflow** to $0$.`,
      },
      {
        t: "table",
        head: ["Format", "Bits (exponent / mantissa)", "Largest finite", "Smallest normal", "Machine $\\epsilon$"],
        rows: [
          ["float64", "11 / 52", "$1.798\\times10^{308}$", "$2.225\\times10^{-308}$", "$2.220\\times10^{-16}$"],
          ["float32", "8 / 23", "$3.403\\times10^{38}$", "$1.175\\times10^{-38}$", "$1.192\\times10^{-7}$"],
          ["float16", "5 / 10", "$65504$", "$6.104\\times10^{-5}$ (subnormals down to $5.96\\times10^{-8}$)", "$9.766\\times10^{-4}$"],
          ["bfloat16", "8 / 7", "$3.390\\times10^{38}$", "$1.175\\times10^{-38}$", "$7.812\\times10^{-3}$"],
        ],
      },
      {
        t: "p",
        text: String.raw`float16 has fair precision but a tiny range: $65504 + 16$ overflows to $\infty$, and a gradient of $10^{-8}$ rounds to $0$. bfloat16 keeps float32's range but has only 8 significant bits: $256 + 1 = 256$ and $1 + 2^{-8} = 1$.`,
      },
      {
        t: "p",
        text: String.raw`**Overflow in softmax and the log-sum-exp trick.** $e^{1000}$ is beyond float64's range, so the naive softmax of $z = (1000, 1001, 1002)$ computes $\infty/\infty = $ NaN. The fix is an exact identity that holds for any constant $m$:`,
      },
      {
        t: "math",
        tex: String.raw`\log\sum_{i=1}^{n} e^{z_i} = m + \log\sum_{i=1}^{n} e^{z_i - m}, \qquad \operatorname{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_k e^{z_k - m}}, \qquad m = \max_i z_i`,
      },
      {
        t: "p",
        text: String.raw`With $m = \max z$ every exponent is $\le0$ (no overflow) and the largest term is $e^0 = 1$, so the sum is in $[1, n]$ and never gives $\log0$ (no underflow). For $z = (1000, 1001, 1002)$: $\operatorname{LSE} = 1002 + \log(e^{-2} + e^{-1} + 1) = 1002.407606$ and softmax $= (0.090031,\ 0.244728,\ 0.665241)$. For $z = (-1000, -1001)$ the naive result is $\log0 = -\infty$; the stable value is $-999.686738$. The same idea gives **log-softmax** $\log p_i = z_i - \operatorname{LSE}(z)$ — always compute cross-entropy this way.`,
      },
      {
        t: "p",
        text: String.raw`**Stable sigmoid, softplus and binary cross-entropy.** Written as $e^x/(1+e^x)$, the sigmoid returns $\infty/\infty = $ NaN at $x = 800$. Branch on the sign so only $e^{-|x|}\in(0,1]$ is computed: $\sigma(x) = 1/(1 + e^{-|x|})$ for $x\ge0$ and $e^{-|x|}/(1 + e^{-|x|})$ for $x < 0$. Similarly`,
      },
      {
        t: "math",
        tex: String.raw`\operatorname{softplus}(x) = \log(1 + e^x) = \max(x, 0) + \operatorname{log1p}\big(e^{-|x|}\big), \qquad \operatorname{BCE}(z, y) = \operatorname{softplus}(z) - yz = y\operatorname{softplus}(-z) + (1-y)\operatorname{softplus}(z)`,
      },
      {
        t: "p",
        text: String.raw`BCE computed from the probability fails on confident mistakes: with logit $z = -120$ and label 1, $\sigma(z)\approx7.7\times10^{-53}$ underflows to $0$ in float32 and $-\log 0 = \infty$; the logit form gives the correct $120$. Hence frameworks' fused "with logits" losses.`,
      },
      {
        t: "p",
        text: String.raw`**Catastrophic cancellation.** Subtracting two nearly equal numbers cancels the leading digits and leaves mostly rounding error. In float64, $(1 + 10^{-16}) - 1 = 0$, and naive $\log(1 + 10^{-10})$ is $1.000000083\times10^{-10}$ (about 7 correct digits) while $\operatorname{log1p}(10^{-10}) = 9.9999999995\times10^{-11}$ (fully accurate). Classic failure: $\mathbb{E}[x^2] - (\mathbb{E}x)^2$ on float32 data with mean $10^4$ and variance $\approx1$ subtracts two numbers near $10^8$ whose float32 spacing is $8$, and returns $-8.0$ — a *negative* variance. The two-pass formula $\mathbb{E}[(x - \bar x)^2]$ gives $0.9962$, matching float64. Normalization layers use the two-pass form (or Welford's algorithm) for this reason.`,
      },
      {
        t: "p",
        text: String.raw`**Epsilon in denominators.** $v/\lVert v\rVert$ for a zero vector gives $0/0 = $ NaN; $v/(\lVert v\rVert + \epsilon)$ gives $0$. Such constants are everywhere: Adam's $\hat m/(\sqrt{\hat v} + \epsilon)$ (PyTorch default $\epsilon = 10^{-8}$), LayerNorm and BatchNorm's $\sqrt{\sigma^2 + \epsilon}$ (default $10^{-5}$). Careful when changing precision: $10^{-8}$ itself rounds to $0$ in float16.`,
      },
      {
        t: "p",
        text: String.raw`**Mixed precision and loss scaling.** float16 training is fast, but small gradients underflow: $10^{-8}$ and $10^{-9}$ become exactly $0$, and $10^{-7}$ is stored as $1.19\times10^{-7}$. **Loss scaling** multiplies the loss by $S$ (e.g. $2^{10}$–$2^{16}$) before backpropagation, so all gradients are $S$ times larger and storable, then divides by $S$ in float32 before the update. With $S = 1024$ the code recovers $10^{-8}$ within $0.12\%$. Dynamic scalers lower $S$ after an overflow (inf/NaN) and raise it otherwise. bfloat16 has float32's range, so it usually needs no scaling, at the price of lower precision; master weights stay in float32 so small updates are not rounded away.`,
      },
      { t: "h", text: "12. Regularization as a modified objective" },
      {
        t: "math",
        tex: String.raw`J(\theta) = \hat R_N(\theta) + \lambda\,\Omega(\theta), \qquad \Omega_{L_2}(\theta) = \tfrac12\lVert\theta\rVert_2^2, \qquad \Omega_{L_1}(\theta) = \lVert\theta\rVert_1 = \sum_k|\theta_k|`,
      },
      {
        t: "p",
        text: String.raw`**$L_2$ (ridge, weight decay).** The gradient gains $\lambda\theta$, so a step is $\theta\leftarrow(1 - \eta\lambda)\theta - \eta\nabla\hat R$: weights shrink by a constant factor each step ("weight decay"; for adaptive optimizers the two differ — see AdamW in [the optimizers lesson](/learn/m06/momentum-adaptive)). The Hessian becomes $\nabla^2\hat R + \lambda I$: every eigenvalue moves up by $\lambda$, so a convex $\hat R$ becomes $\lambda$-strongly convex with one minimiser, and the condition number improves. Above, the penalty $\lambda w^2$ moved the minimiser of $(w-3)^2$ from $3$ to $3/(1+\lambda)$.`,
      },
      {
        t: "p",
        text: String.raw`**$L_1$ (lasso).** $|w|$ is convex but has a kink at $0$. In one dimension: $\arg\min_w\tfrac12(w - a)^2 + \lambda|w| = \operatorname{sign}(a)\max(|a| - \lambda, 0)$, **soft-thresholding** — move toward 0 by $\lambda$, stopping at 0. For $a = 3$: $\lambda = 1$ gives $w = 2$; any $\lambda\ge3$ gives exactly $w = 0$. So $L_1$ makes exact zeros (sparsity); $L_2$ only shrinks.`,
      },
      {
        t: "p",
        text: String.raw`**Probabilistic view.** Minimising $-\log p(\text{data}\mid\theta) - \log p(\theta)$ is MAP estimation: $L_2$ is a Gaussian prior on the weights, $L_1$ a Laplace prior — see [Maximum Likelihood and MAP](/learn/m02/mle-map). Dropout, early stopping and data augmentation regularize without a penalty term; see [Regularization](/learn/m07/regularization).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "h", text: "1. Convex vs non-convex in one dimension" },
      { t: "sim", id: "gradient-descent", caption: "Gradient descent on four 1-D functions: a convex quadratic, a non-convex function with two minima, a plateau, and a flat-bottomed quartic." },
      {
        t: "list",
        items: [
          String.raw`**Quadratic** $x^2$ (convex, $L = 2$): every start reaches the same minimum. The update is $x\leftarrow(1 - 2\eta)x$, so $\eta\le1/L = 0.5$ lowers $f$ every step; $0.5 < \eta < 1$ converges but oscillates; $\eta = 1$ bounces forever; $\eta > 1 = 2/L$ diverges.`,
          String.raw`**Two minima** $x^4 - 3x^2 + x$ (non-convex): from $x_0 = 1.9$ you reach the local minimum near $1.13$ ($f\approx-1.07$); from $x_0 = -0.5$, the global one near $-1.30$ ($f\approx-3.51$). The answer depends on the start — impossible for a convex function.`,
          String.raw`**Plateau + cliff**: from $x_0 = -2.6$ the gradient is nearly zero and the first steps barely move — a plateau, not a minimum. Raise $\eta$ to cross faster, then watch the steep wall near 0 punish the big step.`,
          String.raw`**Flat bottom** $x^4/4$: convex, but the curvature $3x^2$ is zero at the minimum, so it is not strongly convex and convergence near the bottom is slower than geometric (sub-linear).`,
        ],
      },
      { t: "h", text: "2. Saddle points and multiple basins in two dimensions" },
      { t: "sim", id: "optimizers", caption: "Choose the Saddle surface, then the Multi-modal surface." },
      {
        t: "list",
        items: [
          String.raw`**Saddle** $x^2 + y^4/4 - y^2/2$: the default start $(-1.8, 0.01)$ is almost on the ridge $y = 0$. SGD reaches the saddle fast (curvature $+2$ along $x$), then lingers: escape along $y$ depends on the negative curvature $-1$ and the tiny offset. Compare how long Momentum and Adam take to leave.`,
          String.raw`Click starts ever closer to $y = 0$: the closer, the longer SGD lingers (escape time grows like $\ln(1/|y_0|)$). A start exactly on $y = 0$ has zero $y$-gradient forever and converges to the saddle — the measure-zero exception from the theory.`,
          String.raw`**Multi-modal**: four wells of different depths. Start from several points and note which well each optimizer ends in — the non-convex situation in miniature.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      { t: "h", text: "Landscapes: convexity, symmetry, saddles" },
      {
        t: "p",
        text: "Experiments in one program: the chord inequality tested on a convex and a non-convex 1-D function, gradient descent from several starts on each, the $(w_1w_2 - 1)^2$ counter-example, permutation symmetry on a trained MLP, saddle escape on the simulator's surface, and how rare positive definite random Hessians are.",
      },
      { t: "code", lang: "python", title: "landscapes.py", code: LAND_CODE, output: LAND_OUT },
      {
        t: "list",
        items: [
          "The convex function (logistic loss + $0.1w^2$) never breaks the chord inequality in 100,000 random trials, and every start reaches $w = 1.1775$. The non-convex quartic breaks it in about 45% of trials; gradient descent ends at $-1.3008$ or $1.1309$ depending on the start.",
          "The permutation randomly relabels the 8 hidden units; the loss is equal at both ends of the segment and highest in the middle.",
          "`while abs(w[1]) < 0.9` counts the steps needed to leave the saddle; with `y0 = 0` the loop stops at the 2000-step cap with the iterate sitting on the saddle.",
          "The last experiment symmetrises Gaussian matrices, $(M + M^\\top)/2$, and checks whether the smallest eigenvalue is positive.",
        ],
      },
      { t: "h", text: "Smoothness and the step size" },
      {
        t: "p",
        text: "L2-regularised logistic regression is convex and $L$-smooth with a computable $L$. The program computes $L$, finds the exact minimiser with Newton's method, runs gradient descent with three step sizes, checks the $O(1/t)$ bound, and verifies the descent lemma at random points.",
      },
      { t: "code", lang: "python", title: "smoothness.py", code: SMOOTH_CODE, output: SMOOTH_OUT },
      { t: "h", text: "Numerical stability" },
      {
        t: "p",
        text: "Every failure from the Math section, reproduced and fixed. NumPy has no bfloat16, so `to_bfloat16` emulates it by rounding a float32 bit pattern to its top 16 bits (round-to-nearest-even) — exactly how bfloat16 is defined.",
      },
      { t: "code", lang: "python", title: "numerical_stability.py", code: STAB_CODE, output: STAB_OUT },
      {
        t: "list",
        items: [
          "`np.seterr(all=\"ignore\")` lets the naive versions produce inf and NaN silently so we can print them; in real code, NumPy's warnings and PyTorch's `torch.autograd.set_detect_anomaly(True)` help find the first NaN.",
          "The naive softplus at $x = -30$ returns $9.34808\\times10^{-14}$ instead of $9.35762\\times10^{-14}$: adding a tiny number to 1 then taking the log loses about three digits; `log1p` keeps them.",
          "After scaling by 1024, $10^{-9}$ comes back with 1% error because $1024\\times10^{-9}$ is still a float16 subnormal. A larger scale would do better, as long as the largest gradient times the scale stays below 65504.",
        ],
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
                text: "PyTorch's built-ins already use the stable formulas: `torch.softmax`, `torch.log_softmax` and `torch.logsumexp` subtract the maximum; `F.binary_cross_entropy_with_logits` and `F.cross_entropy` work from logits. The program also shows float16 gradient underflow and manual loss scaling (what `torch.amp.GradScaler` automates), that an L2 penalty equals `weight_decay` for plain SGD, and a constraint by projection.",
              },
              { t: "code", lang: "python", title: "optimization_pytorch.py", code: TORCH_CODE, output: TORCH_OUT },
              {
                t: "list",
                items: [
                  "`F.binary_cross_entropy(torch.sigmoid(logit), target)` clamps the log at $-100$, so the loss caps at 100 and the gradient through the saturated sigmoid is essentially zero (printed $-0.0000$). The fused version returns the true loss 120 and gradient $\\sigma(z) - y = -1$: the model learns it is badly wrong.",
                  "The `logsumexp` value is printed from float32, hence $1002.407593$ instead of the float64 value $1002.407606$ from the NumPy program.",
                  "In real mixed-precision training use `torch.autocast(device_type=\"cuda\", dtype=torch.float16)` with `torch.amp.GradScaler`, or bfloat16 autocast without a scaler.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow offers the same stable primitives (`tf.math.reduce_logsumexp`, `tf.nn.softmax`, `tf.nn.sigmoid_cross_entropy_with_logits`, `from_logits=True` in Keras losses), penalties via `kernel_regularizer`, constraints via `kernel_constraint`, and mixed precision via a global policy with automatic loss scaling.",
              },
              { t: "code", lang: "python", title: "optimization_keras.py (not executed in CI)", code: TF_CODE, output: "Prints a logsumexp of about 1002.4076, the softmax [0.090031 0.244728 0.665241], a sigmoid cross-entropy of 120 for the logit −120 with label 1, and the model summary; training then runs 3 epochs with an L2 penalty, a max-norm constraint and mixed-float16 precision (on hardware that supports it), printing the loss per epoch." },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Kinds of optimization problems" },
      {
        t: "table",
        head: ["Problem class", "Example", "Guarantees", "Typical algorithms"],
        rows: [
          ["Convex, smooth, unconstrained", "linear / logistic regression with L2", "global optimum; GD gap $O(1/T)$, linear rate if strongly convex", "GD, Newton, L-BFGS"],
          ["Convex, non-smooth", "lasso, SVM hinge loss", "global optimum; subgradient methods $O(1/\\sqrt T)$, proximal methods faster", "proximal gradient (ISTA/FISTA), coordinate descent"],
          ["Convex, constrained", "max-norm or simplex constraints", "KKT conditions necessary and sufficient", "projected gradient, interior-point methods"],
          ["Non-convex, smooth", "networks with tanh, GELU, softplus", "only approximate stationarity in general: $\\min\\lVert\\nabla J\\rVert^2 = O(1/T)$", "SGD, momentum, Adam"],
          ["Non-convex, non-smooth", "ReLU networks", "weaker still (Clarke stationarity)", "the same, in practice"],
          ["Stochastic (expectation objective)", "every mini-batch training run", "rates in expectation; noise floor unless $\\eta$ decays", "SGD with schedules, variance reduction"],
        ],
      },
      { t: "h", text: "Floating-point formats for training" },
      {
        t: "table",
        head: ["Format", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["float64", "huge range, 16 digits", "slow on GPUs, 2× memory of float32", "gradient checking, small reference runs"],
          ["float32", "safe default, 7 digits", "slower and larger than 16-bit formats", "master weights, optimizer state, reductions"],
          ["float16", "fast, half the memory, 3–4 digits", "max 65504; gradients underflow without loss scaling", "mixed-precision training with a dynamic loss scaler; inference"],
          ["bfloat16", "float32 range, no loss scaling needed", "only 2–3 digits; small updates to large weights are lost", "large-model training on TPUs and recent GPUs"],
          ["8-bit formats (FP8, INT8)", "even faster and smaller", "need per-tensor scaling and careful calibration", "inference quantization; emerging for training"],
        ],
      },
      { t: "h", text: "Ways to express a preference on the parameters" },
      {
        t: "table",
        head: ["Mechanism", "Form", "Effect", "Notes"],
        rows: [
          ["L2 penalty / weight decay", "$+\\tfrac\\lambda2\\lVert\\theta\\rVert_2^2$", "shrinks all weights; strong convexity for convex losses", "equals weight decay for SGD but not for Adam (use AdamW)"],
          ["L1 penalty", "$+\\lambda\\lVert\\theta\\rVert_1$", "exact zeros (sparsity)", "non-differentiable at 0: subgradients or proximal steps"],
          ["Elastic net", "$+\\lambda_1\\lVert\\theta\\rVert_1 + \\tfrac{\\lambda_2}{2}\\lVert\\theta\\rVert_2^2$", "sparse and stable with correlated features", "two hyper-parameters"],
          ["Hard constraint + projection", "$\\theta\\in\\mathcal{C}$, $\\theta\\leftarrow\\Pi_{\\mathcal{C}}(\\cdot)$", "property holds exactly", "projection must be cheap (balls, boxes, simplex)"],
          ["Reparametrisation", "$\\sigma = \\operatorname{softplus}(s)$, $p = \\operatorname{softmax}(z)$", "constraint holds automatically", "changes the landscape and the effective step sizes"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Every training run** minimises a regularized empirical risk. Knowing which parts are data fit, penalty and constraint makes hyper-parameters like weight decay meaningful.",
          "**Convex models as reliable baselines.** Logistic regression and linear SVMs (e.g. in scikit-learn) are convex, so results don't depend on initialization — a good sanity check before training a network.",
          "**Convex sub-problems inside deep learning.** Training only the last layer on frozen features (linear probing) is convex; so is fitting a temperature for probability calibration.",
          "**Learning-rate selection.** Range tests and warm-up are practical answers to the curvature-dependent stability limit; research uses Hessian eigenvalue estimates (from Hessian-vector products) to explain divergence.",
          "**Mixed-precision training** (float16 with loss scaling, or bfloat16) is standard for large models and relies on this lesson's stable formulas.",
          "**Numerically stable kernels.** Softmax attention, including memory-efficient versions like FlashAttention, is built on a running-maximum form of the log-sum-exp trick.",
          "**Model merging and weight averaging.** Averaging models fine-tuned from the same pre-trained start can work when they share a basin, but averaging independently trained networks usually fails unless their hidden units are aligned first — Section 6's permutation symmetry in action.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Convex formulations** give global optima, reproducibility and guarantees, but limit what the model can express; neural networks trade the guarantees for power.",
          "**Non-convexity hurts less than it sounds** in large networks — many minima are equal by symmetry and most bad critical points are saddles — but initialization, optimizer and noise still change which solution you get.",
          "**Penalties** keep optimization simple, but $\\lambda$ must be tuned and gives no hard bound; **constraints** guarantee the bound but need projections or reparametrisations.",
          "**Smoothness-based step sizes** guarantee descent but are cautious (worst-case $L$); tuned or adaptive rates are faster but guarantee nothing.",
          "**Lower precision** is faster and saves memory, at the cost of range (float16) or precision (bfloat16), and needs stable formulas and loss scaling.",
          "**Epsilons and clamps** prevent NaNs but bias results slightly and can hide real bugs (a clamp that is always active means something upstream is wrong).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          String.raw`**"The loss stopped decreasing, so we are at a minimum."** A small gradient can mean a minimum, a saddle or a plateau. Try another learning rate, momentum or a small nudge, and check the validation loss before declaring convergence.`,
          String.raw`**Believing a convex loss makes the problem convex.** Cross-entropy is convex in the logits, but logits are a non-linear function of the weights, so a network's training objective is non-convex. Convexity must hold in the variables you optimise.`,
          String.raw`**Writing softmax and cross-entropy separately.** $\log(\operatorname{softmax}(z))$ in two steps overflows or gives $\log0$ for large logits. Use log-softmax / cross-entropy on logits (PyTorch's cross-entropy expects raw logits, not probabilities).`,
          String.raw`**Applying a sigmoid before a "with logits" loss**, or probability-based BCE on a sigmoid output. The first applies the sigmoid twice; the second loses the gradient on confident mistakes (the loss sticks at the clamp).`,
          String.raw`**Adding epsilon in the wrong place.** $\sqrt{v} + \epsilon$ and $\sqrt{v + \epsilon}$ differ near $v = 0$ (the gradient of $\sqrt{v}$ is infinite at 0, so put $\epsilon$ inside if you differentiate through it). An $\epsilon$ below float16's smallest subnormal is silently $0$.`,
          String.raw`**Computing variance as $\mathbb{E}[x^2] - (\mathbb{E}x)^2$ in low precision.** It can even come out negative. Subtract the mean first (two-pass) or use Welford's online algorithm.`,
          String.raw`**Assuming L2 penalty and weight decay are always the same.** Equal for plain SGD (checked in the PyTorch code), not for adaptive optimizers like Adam, which rescale the penalty gradient per parameter. Use AdamW for decoupled weight decay.`,
          String.raw`**Ignoring loss scale when setting the learning rate.** Multiplying the loss by 10 (e.g. summing instead of averaging over a batch of 10) multiplies $L$ by 10 and the safe step by 1/10. Keep the reduction consistent when comparing learning rates.`,
          String.raw`**Treating NaNs as bad luck.** They almost always come from overflow (exp, large learning rate), $\log 0$, $0/0$ or $\sqrt{\text{negative}}$. Find the first NaN (anomaly detection, finiteness asserts) and fix the formula.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between a loss, a cost and an objective function?", a: "The loss usually measures the error on one example, $\\ell(\\hat y, y)$. The cost (empirical risk) is the average loss over the dataset or a batch. The objective is the full function the optimizer minimises, usually cost plus regularization. Usage is loose; what matters is knowing which function is differentiated." },
      { t: "qa", level: "beginner", q: "What is a convex function, and why do we like convex problems?", a: "$f$ is convex if $f(tx + (1-t)y)\\le tf(x) + (1-t)f(y)$ for all $x, y$ and $t\\in[0,1]$ — the graph lies below its chords. Then every local minimum is global, the minimisers form a convex set, a zero gradient proves optimality, and gradient methods have provable convergence rates." },
      { t: "qa", level: "beginner", q: "What is a saddle point?", a: "A critical point (zero gradient) that is neither a minimum nor a maximum: the function increases along some directions and decreases along others. Its Hessian has both positive and negative eigenvalues. Example: $x^2 - y^2$ at the origin." },
      { t: "qa", level: "beginner", q: "Why does a naive softmax return NaN for large logits, and how do you fix it?", a: "$e^{z}$ overflows to infinity for $z$ above about 709 in float64 (88 in float32), giving $\\infty/\\infty$. Subtract the largest logit first: $\\operatorname{softmax}(z) = \\operatorname{softmax}(z - \\max z)$ exactly, and all exponents are now $\\le 0$." },
      { t: "qa", level: "intermediate", q: "Prove that for a convex function every local minimum is a global minimum.", a: "Let $x$ be a local minimum and suppose $f(y) < f(x)$ for some $y$. For $t\\in(0,1]$, convexity gives $f((1-t)x + ty)\\le f(x) - t(f(x) - f(y)) < f(x)$. Taking $t$ small puts this point inside the neighbourhood where $x$ is minimal, a contradiction." },
      { t: "qa", level: "intermediate", q: "Why is the training loss of a neural network non-convex?", a: "Hidden units can be reordered without changing the network's function, so every minimiser has many copies with equal loss. If the loss were convex, the average of all copies would also be a minimiser (Jensen), but that average has identical hidden units — effectively a one-unit network, which usually fits worse. Even $(w_1w_2 - 1)^2$, a two-layer linear 'network', is non-convex: the midpoint of minima $(1,1)$ and $(-1,-1)$ has higher loss." },
      { t: "qa", level: "intermediate", q: "What does L-smoothness give you, and where does $\\eta\\le1/L$ come from?", a: "If $\\nabla J$ is $L$-Lipschitz, then $J(y)\\le J(x) + \\nabla J(x)^\\top(y - x) + \\tfrac L2\\lVert y - x\\rVert^2$ (descent lemma). For a gradient step, $J(x^+)\\le J(x) - \\eta(1 - L\\eta/2)\\lVert\\nabla J\\rVert^2$: a guaranteed decrease for $\\eta < 2/L$, largest at $\\eta = 1/L$, where it is $\\lVert\\nabla J\\rVert^2/(2L)$. Summing over steps gives $\\min_t\\lVert\\nabla J(\\theta_t)\\rVert^2\\le2L(J(\\theta_0) - J_{\\inf})/T$ even for non-convex $J$." },
      { t: "qa", level: "intermediate", q: "Explain the log-sum-exp trick and where it is used.", a: "$\\log\\sum_ie^{z_i} = m + \\log\\sum_ie^{z_i - m}$ for any $m$; with $m = \\max_iz_i$ all exponents are $\\le 0$ and the sum is between 1 and $n$, so neither overflow nor $\\log 0$ can happen. Used in softmax, log-softmax, cross-entropy, attention, mixture-model and HMM log-likelihoods, and anywhere probabilities are combined in log space." },
      { t: "qa", level: "advanced", q: "Why are saddle points, rather than bad local minima, considered the main obstacle in high-dimensional non-convex optimization?", a: "A local minimum needs all $d$ Hessian eigenvalues positive. Under random-matrix heuristics the chance of that falls extremely fast with $d$ (our simulation: 0.15 at $d = 2$, 0.002 at $d = 4$, none in 20,000 samples at $d\\ge6$), so most high-loss critical points are saddles; Dauphin et al. (2014) backed this with network experiments. Gradient descent escapes strict saddles from almost every start, but slowly when the negative curvature is weak or the start is near the stable manifold, so saddles mainly cost time. Noise, momentum and perturbations speed escape." },
      { t: "qa", level: "advanced", q: "How does mixed-precision training avoid gradient underflow, and why does bfloat16 usually not need loss scaling?", a: "In float16 the smallest positive number is about $6\\times10^{-8}$, so many small gradients round to 0. Loss scaling multiplies the loss by $S$ so gradients are $S$ times larger, then unscales them in float32 before the update; dynamic scalers lower $S$ on overflow. bfloat16 has an 8-bit exponent like float32, so its range reaches about $10^{-38}$ and gradients rarely underflow, though its 8-bit significand is imprecise; master weights stay in float32 so small updates are not rounded away." },
      { t: "qa", level: "advanced", q: "When are a constrained problem and a penalised problem equivalent?", a: "For convex problems, under a constraint qualification such as Slater's condition, the solution of $\\min J$ s.t. $\\Omega(\\theta)\\le r$ also solves $\\min J + \\mu^\\star\\Omega$, where $\\mu^\\star$ is the KKT multiplier; each penalty weight likewise matches some radius. The map between $r$ and $\\lambda$ depends on the data. For non-convex problems this can fail — some constrained solutions minimise no penalised objective." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which statement about a differentiable convex function $f$ on $\\mathbb{R}^d$ is true?", options: ["It has exactly one minimiser", "Any point with $\\nabla f = 0$ is a global minimiser", "Its Hessian is positive definite everywhere", "Gradient descent converges for any learning rate"], answer: 1, explain: "The first-order test $f(y)\\ge f(x) + \\nabla f(x)^\\top(y - x)$ with $\\nabla f(x) = 0$ gives $f(y)\\ge f(x)$. Uniqueness needs strict convexity, the Hessian is only positive semidefinite, and step sizes must still respect smoothness." },
          { kind: "numeric", q: "$f(w_1, w_2) = (w_1w_2 - 1)^2$. What is $f$ at the midpoint of its two minima $(1, 1)$ and $(-1, -1)$?", answer: 1, tolerance: 0.0001, explain: "The midpoint is $(0, 0)$ and $f(0,0) = (0 - 1)^2 = 1 > 0$, which violates convexity." },
          { kind: "numeric", q: "$J(w) = \\tfrac12 w^\\top Aw$ with $A = \\operatorname{diag}(1, 8)$. What is the largest step size $\\eta = 1/L$ that the descent lemma recommends?", answer: 0.125, tolerance: 0.0001, explain: "$L = \\lambda_{\\max}(A) = 8$, so $1/L = 0.125$ (and steps must stay below $2/L = 0.25$ to converge)." },
          { kind: "numeric", q: "Compute $\\operatorname{logsumexp}(1000, 1000) = \\log(e^{1000} + e^{1000})$ to 4 decimals.", answer: 1000.6931, tolerance: 0.0002, explain: "$1000 + \\log(1 + 1) = 1000 + \\ln2 = 1000.6931$ — computed without ever forming $e^{1000}$." },
          { kind: "numeric", q: "What is the largest finite float16 value?", answer: 65504, tolerance: 0, explain: "float16 has a 5-bit exponent and 10-bit mantissa; its maximum is $(2 - 2^{-10})\\times2^{15} = 65504$." },
          { q: "A gradient of magnitude $10^{-8}$ is stored in float16. What happens?", options: ["It is stored exactly", "It rounds to the nearest subnormal with ~1% error", "It underflows to exactly 0", "It overflows to infinity"], answer: 2, explain: "float16's smallest positive subnormal is $2^{-24}\\approx5.96\\times10^{-8}$; $10^{-8}$ is below half of it and rounds to 0. Loss scaling prevents this." },
          { q: "At a critical point the Hessian has eigenvalues $(4, 0.5, -0.01)$. The point is:", options: ["a local minimum", "a local maximum", "a saddle point", "impossible to classify"], answer: 2, explain: "One negative eigenvalue, however small, makes it a (strict) saddle; gradient descent eventually leaves along that eigenvector, slowly, because the curvature is weak." },
          { kind: "numeric", q: "Soft-thresholding: what is $\\arg\\min_w \\tfrac12(w - 3)^2 + 1.5|w|$?", answer: 1.5, tolerance: 0.0001, explain: "$\\operatorname{sign}(3)\\max(3 - 1.5, 0) = 1.5$." },
        ],
      },
      {
        t: "exercise",
        title: "Smoothness and a guaranteed decrease",
        prompt: "Let $J(w) = \\tfrac12 w^\\top Aw$ with $A = \\begin{pmatrix}3 & 1\\\\ 1 & 3\\end{pmatrix}$. (a) Is $J$ convex? Strongly convex? (b) Find its smoothness constant $L$ and condition number. (c) From $w_0 = (1, 0)$, take one gradient step with $\\eta = 1/L$. What decrease does the descent lemma guarantee, and what is the actual new value?",
        hint: "The eigenvalues of a symmetric 2×2 matrix with entries a, b, b, a are a ± b.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Eigenvalues", tex: String.raw`\lambda = 3\pm1 = 2,\ 4 \;\Rightarrow\; A\succ0:\ \text{convex, and } 2\text{-strongly convex}` },
              { title: "Smoothness and conditioning", tex: String.raw`L = \lambda_{\max} = 4,\qquad \mu = 2,\qquad \kappa = L/\mu = 2,\qquad \eta = 1/L = 0.25` },
              { title: "Gradient at the start", tex: String.raw`J(w_0) = \tfrac12\cdot3 = 1.5,\qquad g = Aw_0 = (3,\ 1),\qquad \lVert g\rVert^2 = 10` },
              { title: "Guarantee", tex: String.raw`J(w_1)\le J(w_0) - \frac{\lVert g\rVert^2}{2L} = 1.5 - \frac{10}{8} = 0.25` },
              { title: "Actual step", tex: String.raw`w_1 = (1, 0) - 0.25(3, 1) = (0.25,\ -0.25),\qquad J(w_1) = \tfrac12(0.1875 - 0.125 + 0.1875) = 0.125` },
            ],
          },
          { t: "p", text: "The actual $0.125$ is below the guaranteed $0.25$: the bound is worst-case, assuming the largest curvature in every direction, while this gradient is not aligned with the steepest eigenvector." },
        ],
      },
      {
        t: "exercise",
        title: "A numerically stable cross-entropy with its gradient",
        prompt: "Write `cross_entropy_stable(logits, label)` in NumPy returning the loss $-\\log\\operatorname{softmax}(z)_y$ and its gradient with respect to the logits, using log-sum-exp. Compare with the naive version on $z = (2, 1, 0.1)$, $(1000, 0, -1000)$ and $(-1000, 0, 1000)$ for label 0, and verify the gradient with central differences.",
        hint: "$-\\log p_y = \\operatorname{LSE}(z) - z_y$, and the gradient is $\\operatorname{softmax}(z) - e_y$.",
        solution: [
          { t: "code", lang: "python", title: "stable_cross_entropy.py", code: EX_CODE, output: EX_OUT },
          { t: "p", text: "The naive version returns NaN when the correct class has a huge logit ($\\infty/\\infty$) and $\\infty$ when it has a hugely negative one ($\\log 0$). The stable version returns the exact losses $0$ and $2000$, and the gradient $p - e_y$ matches to $10^{-10}$ on the ordinary example." },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`Training = minimising an objective $J(\theta) = \hat R_N(\theta) + \lambda\Omega(\theta)$; the empirical risk $\hat R_N$ is a proxy for the true risk $R$, and the gap between them is the generalization gap.`,
          String.raw`Constraints ↔ penalties via Lagrange multipliers (KKT); deep learning uses penalties, projections and reparametrisations.`,
          String.raw`Convex: $f(tx + (1-t)y)\le tf(x) + (1-t)f(y)$ ⇔ $f(y)\ge f(x) + \nabla f(x)^\top(y - x)$ ⇔ $\nabla^2f\succeq0$. Jensen: $f(\mathbb{E}X)\le\mathbb{E}f(X)$.`,
          "Convex ⇒ every local minimum is global, and $\\nabla f = 0$ certifies optimality.",
          "Neural networks are non-convex (permutation and scaling symmetries); high-loss critical points are mostly saddles, which gradient descent escapes from almost every start, maybe slowly. Plateaus slow training without being minima.",
          String.raw`Optimality: $\nabla J = 0$ necessary; $\nabla J = 0$ and $\nabla^2J\succ0$ sufficient for a strict local minimum.`,
          String.raw`$L$-smooth ⇒ descent lemma $J(y)\le J(x) + \nabla J^\top(y-x) + \tfrac L2\lVert y-x\rVert^2$ ⇒ with $\eta = 1/L$, $J$ drops by $\lVert\nabla J\rVert^2/(2L)$ per step; convex rate $L\lVert\theta_0 - \theta^\star\rVert^2/(2T)$.`,
          String.raw`Stability: $\operatorname{LSE}(z) = m + \log\sum e^{z_i - m}$; stable sigmoid/softplus via $e^{-|x|}$; compute losses from logits; avoid cancellation ($\operatorname{log1p}$, two-pass variance); $\epsilon$ in denominators; float16 max 65504 and gradients below $\sim6\times10^{-8}$ vanish — use loss scaling or bfloat16.`,
          String.raw`$L_2$ adds $\lambda\theta$ to the gradient and $\lambda I$ to the Hessian; $L_1$ soft-thresholds and creates exact zeros.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Gradient Descent](/learn/m06/gradient-descent) — the algorithm itself: the update rule, $2/\\lambda_{\\max}$ stability on quadratics, and mini-batches.",
          "[Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW](/learn/m06/momentum-adaptive) — how practical optimizers handle ill-conditioning, saddles and plateaus.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — Section 12's modified objectives and implicit regularizers in practice.",
          "[Maximum Likelihood and MAP Estimation](/learn/m02/mle-map) — where losses and penalties come from, probabilistically.",
          "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — practical tools built on this lesson's smoothness and stability ideas.",
        ],
      },
    ],
  },
};

export default lesson;
