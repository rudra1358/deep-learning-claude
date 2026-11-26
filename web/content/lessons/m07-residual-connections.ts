import type { Lesson } from "@/lib/types";

const RES_NUMPY = String.raw`import numpy as np
from math import comb

rng = np.random.default_rng(0)
relu = lambda z: np.maximum(z, 0)


# ---------------- 1. one residual block and its Jacobian ----------------
def block_forward(x, W1, W2):
    """y = x + F(x) with F(x) = W2 relu(W1 x). Returns y and the cache needed for backward."""
    z = W1 @ x
    return x + W2 @ relu(z), z


def block_backward(dy, z, W1, W2):
    """Given dL/dy, return dL/dx = dy (identity path) + W1^T (relu'(z) * (W2^T dy)) (branch path)."""
    return dy + W1.T @ ((z > 0) * (W2.T @ dy))


n = 6
x = rng.normal(size=n)
W1, W2 = rng.normal(0, np.sqrt(2 / n), (n, n)), rng.normal(0, 0.1, (n, n))
y, z = block_forward(x, W1, W2)
J_analytic = np.eye(n) + W2 @ np.diag((z > 0).astype(float)) @ W1      # I + dF/dx
eps, J_numeric = 1e-6, np.zeros((n, n))
for k in range(n):                                                   # central differences, column by column
    e = np.zeros(n); e[k] = eps
    J_numeric[:, k] = (block_forward(x + e, W1, W2)[0] - block_forward(x - e, W1, W2)[0]) / (2 * eps)
print(f"max |J_numeric - (I + W2 D W1)| = {np.abs(J_numeric - J_analytic).max():.1e}")
dy = rng.normal(size=n)
print(f"backward() matches J^T dy: {np.allclose(block_backward(dy, z, W1, W2), J_analytic.T @ dy)}")
print("singular values of dF/dx:   ", np.round(np.linalg.svd(J_analytic - np.eye(n), compute_uv=False), 3))
print("singular values of I + dF/dx:", np.round(np.linalg.svd(J_analytic, compute_uv=False), 3))


# ---------------- 2. plain vs residual: gradient norms through 50 layers ----------------
def gradient_profile(L, n, residual, branch_scale, w1_var=2.0, B=128, seed=1):
    """Forward + backward through L layers of width n (Var(W1) = w1_var / n; W2 scaled by branch_scale).
    plain:    h <- relu(W1 h)              residual: h <- h + branch_scale * W2 relu(W1 h)
    Returns std of h at the output and ||dL/dh|| at the input of every layer (L = sum of h_L * g)."""
    r = np.random.default_rng(seed)
    H = r.normal(size=(n, B))
    cache = []
    for _ in range(L):
        W1 = r.normal(0, np.sqrt(w1_var / n), (n, n))
        W2 = r.normal(0, np.sqrt(1 / n), (n, n)) * branch_scale
        Z = W1 @ H
        cache.append((W1, W2, Z))
        H = H + W2 @ relu(Z) if residual else relu(Z)
    G = r.normal(size=(n, B)) / B                 # upstream gradient dL/dh_L
    norms = []
    for W1, W2, Z in reversed(cache):
        if residual:
            G = G + W1.T @ ((Z > 0) * (W2.T @ G))
        else:
            G = W1.T @ ((Z > 0) * G)
        norms.append(np.linalg.norm(G))
    return H.std(), norms[::-1]                   # norms[0] is at the input of layer 1


L, n = 50, 64
print(f"\n{L} layers, width {n}: ||dL/dh|| at the input of layers 1, 10, 25, 50 and output std")
for name, res, s, v in [("plain, He  Var(W) = 2/n", False, 1.0, 2.0), ("plain, Xavier  Var(W) = 1/n", False, 1.0, 1.0),
                        ("residual, branch scale 1", True, 1.0, 2.0), ("residual, branch scale 1/sqrt(L)", True, 1 / np.sqrt(L), 2.0),
                        ("residual, branch scale 0", True, 0.0, 2.0)]:
    std, g = gradient_profile(L, n, res, s, v)
    print(f"{name:33s} " + " ".join(f"{g[l - 1]:9.2e}" for l in (1, 10, 25, 50)) + f"   std(h_L) {std:9.2e}")


# ---------------- 3. the unrolled view: a residual net is a sum of 2^L paths ----------------
L = 30
lengths = np.arange(L + 1)
counts = np.array([comb(L, k) for k in lengths], dtype=float)    # number of paths through exactly k branches
print(f"\n{L} blocks: 2^{L} = {2 ** L:,} paths; mean path length {np.sum(lengths * counts) / 2 ** L:.1f}; "
      f"fraction of paths with 10..20 branches {counts[10:21].sum() / 2 ** L:.3f}")

# split the backward gradient into the part carried by paths through exactly k branches:
# (I + J_L^T)...(I + J_1^T) g  =  sum_k  (sum over paths with k branches)  g
n = 64
r = np.random.default_rng(2)
h = r.normal(size=n)
Js = []
for _ in range(L):                                # linearise each block at its operating point
    W1 = r.normal(0, np.sqrt(2 / n), (n, n)); W2 = r.normal(0, np.sqrt(1 / n), (n, n)) / np.sqrt(L)
    z = W1 @ h
    Js.append(W2 @ ((z > 0)[:, None] * W1))      # dF/dh = W2 diag(relu'(z)) W1
    h = h + W2 @ relu(z)
g = r.normal(size=n)                             # dL/dh_L
parts = np.zeros((L + 1, n)); parts[0] = g        # parts[k] = gradient carried by paths with k branches
for J in reversed(Js):                            # going backwards, each block either skips or uses its branch
    parts[1:] = parts[1:] + parts[:-1] @ J        # (row vector) v J  ==  J^T v
total = parts.sum(0)
exact = g.copy()
for J in reversed(Js):
    exact = exact + exact @ J
print(f"sum over path lengths equals the full backward pass: {np.allclose(total, exact)}")
contrib = np.linalg.norm(parts, axis=1)
print("||gradient carried by paths with k branches||, k = 0..6:", " ".join(f"{c:.3f}" for c in contrib[:7]))
print(f"k with the largest contribution: {contrib.argmax()}, k = 15 (the most numerous paths): {contrib[15]:.1e}")`;

const RES_NUMPY_OUT = String.raw`max |J_numeric - (I + W2 D W1)| = 5.2e-11
backward() matches J^T dy: True
singular values of dF/dx:    [0.494 0.179 0.    0.    0.    0.   ]
singular values of I + dF/dx: [1.362 1.029 1.    1.    0.948 0.783]

50 layers, width 64: ||dL/dh|| at the input of layers 1, 10, 25, 50 and output std
plain, He  Var(W) = 2/n            1.18e+00  8.59e-01  9.66e-01  6.64e-01   std(h_L)  1.64e+00
plain, Xavier  Var(W) = 1/n        3.51e-08  5.79e-07  1.18e-04  4.69e-01   std(h_L)  4.89e-08
residual, branch scale 1           1.37e+07  6.65e+05  4.15e+03  9.87e-01   std(h_L)  1.71e+07
residual, branch scale 1/sqrt(L)   1.15e+00  1.06e+00  9.10e-01  7.15e-01   std(h_L)  1.62e+00
residual, branch scale 0           7.08e-01  7.08e-01  7.08e-01  7.08e-01   std(h_L)  9.96e-01

30 blocks: 2^30 = 1,073,741,824 paths; mean path length 15.0; fraction of paths with 10..20 branches 0.957
sum over path lengths equals the full backward pass: True
||gradient carried by paths with k branches||, k = 0..6: 8.677 8.465 5.668 3.590 1.781 0.830 0.268
k with the largest contribution: 0, k = 15 (the most numerous paths): 5.7e-07`;

const RES_TORCH_BLOCKS = String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)


class BasicBlock(nn.Module):
    """ResNet-18/34 block: 3x3 conv - BN - ReLU - 3x3 conv - BN, plus shortcut, then ReLU."""
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:          # projection shortcut: 1x1 conv (with the same stride) + BN
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    """ResNet-50/101/152 block: 1x1 reduce - 3x3 - 1x1 expand (x4), plus shortcut, then ReLU."""
    expansion = 4

    def __init__(self, c_in, width, stride=1):
        super().__init__()
        c_out = width * self.expansion
        self.conv1 = nn.Conv2d(c_in, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, c_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return torch.relu(out + self.shortcut(x))


class PreActBlock(nn.Module):
    """He et al. (2016, 'Identity Mappings'): BN - ReLU - conv - BN - ReLU - conv, added to an untouched x."""

    def __init__(self, c):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(c), nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(c), nn.Conv2d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return x + out                            # no ReLU after the addition: a clean identity path


def n_params(m, conv_only=False):
    return sum(p.numel() for name, p in m.named_parameters() if not conv_only or p.dim() == 4)


x = torch.randn(8, 64, 32, 32)
blocks = [
    ("BasicBlock 64->64", BasicBlock(64, 64), x),
    ("BasicBlock 64->128, stride 2", BasicBlock(64, 128, stride=2), x),
    ("Bottleneck 256->(64)->256", Bottleneck(256, 64), torch.randn(8, 256, 32, 32)),
    ("Plain 3x3 stack 256->256 (x2)", nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.Conv2d(256, 256, 3, padding=1, bias=False)), torch.randn(8, 256, 32, 32)),
    ("PreActBlock 64", PreActBlock(64), x),
]
print(f"{'block':31s} {'input':>16s} {'output':>16s} {'conv weights':>13s} {'all params':>11s}")
for name, blk, inp in blocks:
    out = blk(inp)
    print(f"{name:31s} {str(tuple(inp.shape)):>16s} {str(tuple(out.shape)):>16s} {n_params(blk, True):13,d} {n_params(blk):11,d}")

# the gradient of a residual block contains the identity: d(sum(y))/dx >= 1 element-wise in the pre-act block
blk = PreActBlock(64)
nn.init.zeros_(blk.conv2.weight)                  # branch output 0 -> the block is exactly the identity
xr = torch.randn(2, 64, 8, 8, requires_grad=True)
y = blk(xr)
y.sum().backward()
print(f"\nzero-initialised branch: max |y - x| = {(y - xr).abs().max().item():.1e}, "
      f"dsum(y)/dx all ones: {torch.allclose(xr.grad, torch.ones_like(xr))}")`;

const RES_TORCH_BLOCKS_OUT = String.raw`block                                      input           output  conv weights  all params
BasicBlock 64->64                (8, 64, 32, 32)  (8, 64, 32, 32)        73,728      73,984
BasicBlock 64->128, stride 2     (8, 64, 32, 32) (8, 128, 16, 16)       229,376     230,144
Bottleneck 256->(64)->256       (8, 256, 32, 32) (8, 256, 32, 32)        69,632      70,400
Plain 3x3 stack 256->256 (x2)   (8, 256, 32, 32) (8, 256, 32, 32)     1,179,648   1,179,648
PreActBlock 64                   (8, 64, 32, 32)  (8, 64, 32, 32)        73,728      73,984

zero-initialised branch: max |y - x| = 0.0e+00, dsum(y)/dx all ones: True`;

const RES_TORCH_DEGRADATION = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)                          # small model: deterministic and faster on one thread
X, y = load_digits(return_X_y=True)
X = (X / 16.0 - 0.5) / 0.5                        # pixels 0..16 -> [-1, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)
WIDTH, EPOCHS, BATCH, LR = 64, 20, 64, 0.01


class Block(nn.Module):
    """Two Linear-BN layers. Plain: relu(F(x)). Residual (He et al. 2016 layout): relu(x + F(x))."""

    def __init__(self, w, residual, zero_gamma):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(w, w, bias=False), nn.BatchNorm1d(w)
        self.fc2, self.bn2 = nn.Linear(w, w, bias=False), nn.BatchNorm1d(w)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        if zero_gamma:
            nn.init.zeros_(self.bn2.weight)       # the branch starts as F(x) = 0: block = relu(x)
        self.residual = residual

    def forward(self, x):
        out = self.bn2(self.fc2(torch.relu(self.bn1(self.fc1(x)))))
        return torch.relu(x + out) if self.residual else torch.relu(out)


def make(n_blocks, residual, zero_gamma):
    stem = nn.Linear(64, WIDTH)
    nn.init.kaiming_normal_(stem.weight, nonlinearity="relu")
    blocks = [Block(WIDTH, residual, zero_gamma) for _ in range(n_blocks)]
    return nn.Sequential(stem, nn.ReLU(), *blocks, nn.Linear(WIDTH, 10))


def grad_norms_at_init(model):
    """||dL/dh|| at the input of the first and of the last block, and ||dL/dW|| of the stem, on one batch."""
    blocks = [m for m in model if isinstance(m, Block)]
    inputs = []
    hooks = [b.register_forward_pre_hook(lambda m, inp: (inp[0].retain_grad(), inputs.append(inp[0]))[1]) for b in blocks]
    nn.CrossEntropyLoss()(model(Xtr[:256]), ytr[:256]).backward()
    for h in hooks:
        h.remove()
    out = inputs[0].grad.norm().item(), inputs[-1].grad.norm().item(), model[0].weight.grad.norm().item()
    model.zero_grad()
    return out


def run(n_blocks, residual, zero_gamma):
    torch.manual_seed(0)
    model = make(n_blocks, residual, zero_gamma)
    g_first, g_last, g_stem = grad_norms_at_init(model)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(0)
    curve = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm, total = torch.randperm(len(Xtr), generator=gen), 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch in (1, 5, 10, 20):
            curve.append(total / len(Xtr))        # mean training loss over the epoch
    model.eval()
    with torch.no_grad():
        train_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
        test_acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return g_first, g_last, g_stem, curve, train_acc, test_acc


print("                           gradient norms at init     |  mean train loss in epoch      |   accuracy")
print("layers variant            dL/dh[1] dL/dh[last] dL/dW0 |     1       5      10      20 | train   test")
for n_blocks in (1, 4, 16):
    for name, residual, zero_gamma in [("plain", False, False), ("residual", True, False), ("residual, gamma=0", True, True)]:
        g1, gL, g0, curve, tr, te = run(n_blocks, residual, zero_gamma)
        print(f"{2 * n_blocks + 2:4d}   {name:17s} {g1:9.2e} {gL:9.2e} {g0:9.2e} | "
              + " ".join(f"{c:7.3f}" if c < 100 else f"{c:7.1e}" for c in curve) + f" | {tr:.3f}  {te:.3f}")`;

const RES_TORCH_DEGRADATION_OUT = String.raw`gradient norms at init     |  mean train loss in epoch      |   accuracy
layers variant            dL/dh[1] dL/dh[last] dL/dW0 |     1       5      10      20 | train   test
   4   plain              5.60e-02  5.60e-02  3.93e-01 |   2.008   0.303   0.146   0.072 | 0.995  0.971
   4   residual           7.05e-02  7.05e-02  6.47e-01 |   1.808   0.201   0.080   0.027 | 0.999  0.980
   4   residual, gamma=0  2.39e-02  2.39e-02  4.09e-01 |   2.105   0.318   0.167   0.092 | 0.983  0.971
  10   plain              2.13e-01  5.13e-02  7.90e-01 |   2.185   1.236   0.914   1.294 | 0.575  0.564
  10   residual           2.29e-01  4.99e-02  1.28e+00 |   1.617   9.132   1.029   0.187 | 0.975  0.960
  10   residual, gamma=0  2.58e-02  2.58e-02  6.20e-01 |   2.155   0.321   0.159   0.079 | 0.986  0.969
  34   plain              1.88e+01  4.82e-02  6.55e+01 |   2.354   2.290   2.254   2.230 | 0.142  0.156
  34   residual           1.07e+00  4.19e-02  3.88e+00 |   1.873 5.3e+06 4.7e+05 9.9e+04 | 0.902  0.869
  34   residual, gamma=0  2.51e-02  2.51e-02  4.56e-01 |   1.992   0.262   0.119   0.043 | 0.996  0.971`;

const RES_TORCH_LN = String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
D, DEPTH = 64, 24


class FFBlock(nn.Module):
    """Transformer-style feed-forward sub-layer, wired post-LN or pre-LN."""

    def __init__(self, d, pre_ln):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.pre_ln = pre_ln

    def forward(self, x):
        if self.pre_ln:
            return x + self.ff(self.ln(x))        # pre-LN: x_{l+1} = x_l + F(LN(x_l)); identity path untouched
        return self.ln(x + self.ff(x))            # post-LN: x_{l+1} = LN(x_l + F(x_l)); LN sits on the main path


x = torch.randn(32, 16, D)                        # (batch, tokens, features)
with torch.no_grad():
    for pre_ln in (False, True):
        torch.manual_seed(1)
        h, stds = x, []
        for _ in range(DEPTH):
            h = FFBlock(D, pre_ln)(h)
            stds.append(h.std().item())
        print(f"{'pre-LN ' if pre_ln else 'post-LN'}: std of the residual stream after blocks 1, 6, 12, 18, 24 = "
              + "  ".join(f"{stds[i - 1]:.3f}" for i in (1, 6, 12, 18, 24)))`;

const RES_TORCH_LN_OUT = String.raw`post-LN: std of the residual stream after blocks 1, 6, 12, 18, 24 = 1.000  1.000  1.000  1.000  1.000
pre-LN : std of the residual stream after blocks 1, 6, 12, 18, 24 = 1.022  1.117  1.227  1.306  1.398`;

const RES_HIGHWAY = String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
WIDTH, DEPTH = 64, 50


class Highway(nn.Module):
    """Srivastava et al. (2015): y = T(x) * H(x) + (1 - T(x)) * x, with a learned transform gate T."""

    def __init__(self, d, gate_bias=-2.0):
        super().__init__()
        self.H = nn.Linear(d, d)
        self.T = nn.Linear(d, d)
        nn.init.constant_(self.T.bias, gate_bias)    # negative bias: start by carrying x through

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        return t * torch.tanh(self.H(x)) + (1 - t) * x


class Plain(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.H = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.H(x))


class Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.H = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.tanh(self.H(x))


def input_gradient(layers):
    x = torch.randn(128, WIDTH, requires_grad=True)
    target = torch.randn(128, WIDTH)
    out = layers(x)
    ((out - target) ** 2).mean().backward()
    return x.grad.norm().item(), out.std().item()


print(f"{DEPTH} layers of width {WIDTH} (PyTorch default Linear init)")
for name, make in [("plain tanh", lambda: Plain(WIDTH)), ("highway, gate bias -2", lambda: Highway(WIDTH, -2.0)),
                   ("highway, gate bias 0", lambda: Highway(WIDTH, 0.0)), ("residual x + tanh(Wx)", lambda: Residual(WIDTH))]:
    torch.manual_seed(1)
    net = nn.Sequential(*[make() for _ in range(DEPTH)])
    g, s = input_gradient(net)
    print(f"{name:24s} ||dL/dx|| = {g:.3e}   std(output) = {s:.3f}")

hw = Highway(WIDTH)
with torch.no_grad():
    t = torch.sigmoid(hw.T(torch.randn(1000, WIDTH)))
print(f"\nmean transform gate at init with bias -2: {t.mean():.3f} (sigmoid(-2) = {torch.sigmoid(torch.tensor(-2.0)):.3f})")
n_res = sum(p.numel() for p in Residual(WIDTH).parameters())
n_hw = sum(p.numel() for p in Highway(WIDTH).parameters())
print(f"parameters per layer: residual {n_res:,}, highway {n_hw:,}")

# DenseNet-style concatenation: every layer sees all previous feature maps
k0, k, L = 64, 32, 6                               # input channels, growth rate, layers in the dense block
channels = [k0 + l * k for l in range(L + 1)]
print("\ndense block, growth rate 32: input channels seen by layers 1..6 =", channels[:-1], "-> block output", channels[-1])`;

const RES_HIGHWAY_OUT = String.raw`50 layers of width 64 (PyTorch default Linear init)
plain tanh               ||dL/dx|| = 6.557e-15   std(output) = 0.082
highway, gate bias -2    ||dL/dx|| = 4.284e-05   std(output) = 0.019
highway, gate bias 0     ||dL/dx|| = 1.853e-14   std(output) = 0.046
residual x + tanh(Wx)    ||dL/dx|| = 1.351e+00   std(output) = 5.412

mean transform gate at init with bias -2: 0.132 (sigmoid(-2) = 0.119)
parameters per layer: residual 4,160, highway 8,320

dense block, growth rate 32: input channels seen by layers 1..6 = [64, 96, 128, 160, 192, 224] -> block output 256`;

const RES_KERAS = String.raw`import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_digits


def residual_block(x, filters, stride=1):
    """Basic ResNet block with the functional API: two 3x3 convs, BN, and an Add() shortcut."""
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization(gamma_initializer="zeros")(y)    # zero-gamma: the block starts as the identity
    if stride != 1 or x.shape[-1] != filters:                       # shapes differ: 1x1 projection shortcut
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([y, shortcut])                                 # the residual connection
    return layers.ReLU()(y)


inputs = tf.keras.Input(shape=(8, 8, 1))                            # sklearn digits as 8x8 images
x = layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
for filters, stride in [(32, 1), (32, 1), (64, 2), (64, 1)]:
    x = residual_block(x, filters, stride)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 8, 8, 1).astype("float32")
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(X, y, batch_size=64, epochs=10, validation_split=0.25, verbose=2)`;

const lesson: Lesson = {
  id: "m07/residual-connections",
  title: "Residual and Skip Connections",
  summary: "Why deeper plain networks train worse (the degradation problem), how the residual form y = x + F(x) fixes it, the gradient I + ∂F/∂x and the sum-of-paths view, projection shortcuts, basic vs bottleneck vs pre-activation blocks, highway and dense connections, residual streams in Transformers and U-Net skips — with a real plain-vs-residual experiment.",
  minutes: 85,
  level: "intermediate",
  prerequisites: ["m04/gradient-problems", "m07/initialization", "m04/backpropagation", "m08/convolution"],
  next: ["m07/normalization", "m08/cnn-architectures", "m11/transformer-architecture", "m12/segmentation"],
  terms: ["residual-connection", "skip-connection", "degradation-problem", "identity-mapping", "projection-shortcut", "bottleneck-block", "pre-activation-resnet", "highway-network", "residual-stream", "dead-relu"],
  sections: {
    intro: [
      { t: "p", text: String.raw`A **residual connection** (or **skip connection**) adds a layer's input straight to its output. Instead of $y = F(x)$, a residual block computes $y = x + F(x)$. The small sub-network $F$ (usually two or three layers) only has to learn the *change* (the residual) to apply to $x$. This one-line idea made networks with hundreds of layers trainable. It is now in almost every deep architecture: ResNets in vision, every Transformer (where the running sum of all updates is called the **residual stream**), diffusion U-Nets, and graph and speech models.` },
      { t: "p", text: String.raw`It was invented to solve a surprising problem. By 2015, batch normalisation and careful initialisation ([Weight Initialization](/learn/m07/initialization)) had mostly solved *vanishing gradients*: plain networks of 20–30 layers could be trained from scratch. Yet He, Zhang, Ren and Sun (2016, *"Deep Residual Learning for Image Recognition"*, CVPR; arXiv December 2015) saw that making such a network deeper made it *worse* — not on the test set, where you might blame overfitting, but on the **training set**. On ImageNet their 34-layer plain network had higher training error than their 18-layer one throughout training (top-1 validation error 28.54% vs 27.94%). This is the **degradation problem**. A deeper network can do everything a shallower one can (its extra layers could just copy their input), so the higher training error is an *optimisation* failure.` },
      { t: "p", text: "Residual learning fixed it. With shortcuts, ResNet-34 beat ResNet-18 (25.03% vs 27.88% top-1), and a 152-layer ResNet ensemble won the ILSVRC 2015 classification task with 3.57% top-5 test error. A follow-up (He et al., 2016, *\"Identity Mappings in Deep Residual Networks\"*) explained why the identity path matters and trained a 1001-layer *pre-activation* ResNet on CIFAR-10." },
      { t: "p", text: String.raw`[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) briefly showed that a residual block's Jacobian is $I + \partial F/\partial x$. Here is the full story, checked on real runs.` },
      { t: "callout", kind: "key", text: String.raw`A residual block $y = x + F(x)$ does two things. (1) It makes "do nothing" the *default*: the block only has to learn $F \approx 0$ to pass its input through. (2) It gives the gradient a path with Jacobian exactly $I$ around every block, so $\partial\mathcal L/\partial x$ always contains $\partial\mathcal L/\partial y$ itself. At initialisation, a deep residual network acts like a sum of many shallow networks.` },
    ],
    prerequisites: [
      { t: "list", items: [
        String.raw`[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — the backward signal as a product of layer Jacobians $\prod D^{[l]}W^{[l+1]\top}$, and the `+"`gradient-flow`"+" simulator with its residual toggle.",
        "[Weight Initialization](/learn/m07/initialization) — keeping variance steady, He initialisation, and zero-initialised residual branches (zero-γ, Fixup). This lesson's experiment depends on them.",
        String.raw`[Backpropagation](/learn/m04/backpropagation) — gradients add where a value is used twice. If $x$ feeds both the shortcut and $F$, $\partial\mathcal L/\partial x$ is the sum of the two contributions. That is the whole gradient story of residual connections.`,
        String.raw`[Convolution](/learn/m08/convolution) — channels, kernel sizes, stride and padding, needed for ResNet blocks, $1\times1$ projection shortcuts and parameter counts.`,
        String.raw`[Jacobians](/learn/m02/jacobians-hessians) — the Jacobian of a vector function and the chain rule as a product of Jacobians. Batch normalisation is covered in [Normalization](/learn/m07/normalization); here you only need to know it standardises each channel, then applies a learnable scale $\gamma$ and shift $\beta$.`,
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Editing a document instead of retyping it", text: "Picture 50 editors passing a document down a line. In the *plain* setup each editor retypes the whole document, adding their changes. One careless typist garbles everything after them, and an editor with nothing to add must still copy the text perfectly. In the *residual* setup each editor only writes tracked changes on top of what they received. An editor with nothing to say writes nothing, and the document passes through untouched. Fifty editors can no longer do worse than five." },
      { t: "p", text: String.raw`Plain deep networks are like the retyping team: even copying the input through a stack of nonlinear layers (the identity function) is hard to learn by gradient descent from a random start. A residual block makes copying free. If the best a block can do is nothing, it only needs $F(x) = 0$, which is easy: push the last layer's weights toward zero, or start them there.` },
      { t: "p", text: "The backward pass benefits too. The gradient at a block's output is copied unchanged to its input through the shortcut, plus a second part through $F$. However badly the branches behave, the copied part reaches the first layer intact. A plain network has one path, a product of fifty factors; a residual network has $2^{50}$ paths, including one with no factors at all." },
      { t: "p", text: String.raw`Two caveats. First, adding $F(x)$ to $x$ at every block makes the signal's variance grow with depth, so residual networks need normalisation or small starting branches to stay stable. Second, $x$ and $F(x)$ must have the same shape. Where they differ (more channels, lower resolution), the shortcut must be a learned projection.` },
    ],
    mechanism: [
      { t: "steps", items: [
        { title: "Split the block into a branch and a shortcut", text: String.raw`The input $x$ is used twice: it goes into the residual branch $F$ (e.g. conv → BN → ReLU → conv → BN) and, unchanged, into the shortcut.` },
        { title: "Compute the residual branch", text: String.raw`$F(x; \{W_i\})$ is an ordinary small sub-network. Its output must have the same shape as $x$ (same channels and spatial size) so the two can be added.` },
        { title: "Make shapes match if necessary", text: String.raw`If $F$ changes the number of channels or downsamples (stride 2), the shortcut becomes a **projection** $W_sx$: a $1\times1$ convolution with the same stride (usually followed by BN). Otherwise the shortcut is the identity: no parameters, no cost.` },
        { title: "Add element-wise", text: String.raw`$y = x + F(x)$ (or $W_sx + F(x)$). The shortcut adds element by element — it does not concatenate or gate.` },
        { title: "Optionally apply a nonlinearity after the sum", text: String.raw`The original ResNet applies ReLU after the addition: $y = \mathrm{ReLU}(x + F(x))$. The pre-activation ResNet moves BN and ReLU inside the branch and leaves the sum alone, $y = x + F(x)$, so the shortcut is a pure identity through the whole network.` },
        { title: "Stack blocks into stages", text: String.raw`A ResNet is a stem (e.g. a $7\times7$ convolution and pooling), then stages of blocks at constant resolution, each stage starting with a downsampling block (projection shortcut), then global average pooling and a linear classifier. ResNet-18/34 use basic blocks; ResNet-50/101/152 use bottleneck blocks.` },
        { title: "Backward pass", text: String.raw`Because $x$ fed two consumers, its gradient is the sum of both: $\partial\mathcal L/\partial x = \partial\mathcal L/\partial y + (\partial F/\partial x)^\top\partial\mathcal L/\partial y$. Automatic differentiation does this with no special code.` },
        { title: "Initialisation", text: String.raw`Initialise the branch layers as usual (He for ReLU convolutions). Often the last BN's $\gamma$ (or the last layer) of each branch is set to 0, so each block starts as the identity (or as $\mathrm{ReLU}(x)$ in the post-activation design).` },
      ] },
      { t: "h", text: "Where the shortcut sits in different architectures" },
      { t: "table", head: ["Architecture", "Shortcut form", "Combine by", "Spans"], rows: [
        ["ResNet (2015/16)", "identity or 1×1 projection", "addition, ReLU after", "2 (basic) or 3 (bottleneck) layers"],
        ["Pre-activation ResNet (2016)", "identity, BN-ReLU moved into the branch", "addition, nothing after", "2–3 layers"],
        ["Highway network (2015)", "identity scaled by a learned carry gate", "gated sum T·H + (1−T)·x", "1 layer"],
        ["DenseNet (2017)", "all earlier feature maps", "concatenation", "every earlier layer in the dense block"],
        ["Transformer (2017)", "identity around attention and around the MLP", "addition (LayerNorm before or after)", "one sub-layer"],
        ["U-Net (2015)", "encoder feature map at the same resolution", "concatenation into the decoder", "across the whole encoder–decoder"],
      ] },
    ],
    math: [
      { t: "h", text: "1. Residual learning and the identity mapping" },
      { t: "p", text: String.raw`Let $\mathcal H(x)$ be the mapping a group of layers should ideally compute. A plain network learns $\mathcal H$ directly. A residual block rewrites it as` },
      { t: "math", tex: String.raw`y = \mathcal H(x) = x + F(x;\,\theta),\qquad F(x;\theta) = \mathcal H(x) - x`, caption: "x: block input; F: residual branch with parameters θ; y: block output (same shape as x)." },
      { t: "p", text: String.raw`Both forms represent the same functions, but gradient descent does not find them equally easily. He et al. argued that in very deep networks many layers should only make small refinements, so the best $\mathcal H$ is often near the identity. The residual form then only needs $F \approx 0$. A plain stack must find weights whose composition, through nonlinearities, reproduces its input — with ReLU and typical weights, not even roughly true. In our experiment below, a 34-layer plain MLP with BatchNorm could not get its training loss below 2.23 (on a task where a 4-layer network reaches 0.07). The same depth with shortcuts reached 0.043.` },
      { t: "h", text: "2. The Jacobian of a block: I + ∂F/∂x" },
      { t: "p", text: String.raw`Differentiating $y = x + F(x)$:` },
      { t: "math", tex: String.raw`J_y = \frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x},\qquad \frac{\partial\mathcal L}{\partial x} = J_y^\top\frac{\partial\mathcal L}{\partial y} = \frac{\partial\mathcal L}{\partial y} + \Big(\frac{\partial F}{\partial x}\Big)^{\!\top}\frac{\partial\mathcal L}{\partial y}` },
      { t: "p", text: String.raw`For a two-layer branch $F(x) = W_2\,\mathrm{ReLU}(W_1x)$, $\partial F/\partial x = W_2DW_1$ with $D = \mathrm{diag}(\mathbb 1[W_1x > 0])$ (1 for active units, 0 otherwise). Adding $I$ shifts every eigenvalue of the branch Jacobian by 1. So if $\partial F/\partial x$ is small, the singular values of $J_y$ are close to 1: the block nearly preserves lengths. In the NumPy program below, a branch Jacobian with singular values $(0.494, 0.179, 0, 0, 0, 0)$ gives a block Jacobian with $(1.362, 1.029, 1, 1, 0.948, 0.783)$. No direction is lost, even though the branch alone wipes out four of the six.` },
      { t: "h", text: "Worked example: one residual block in two dimensions" },
      { t: "p", text: String.raw`$x = (1, -1)$, $W_1 = \begin{pmatrix}0.5 & -0.2\\ 0.3 & 0.8\end{pmatrix}$, $W_2 = \begin{pmatrix}0.4 & 0.1\\ -0.2 & 0.3\end{pmatrix}$, $F(x) = W_2\,\mathrm{ReLU}(W_1x)$, upstream gradient $g = \partial\mathcal L/\partial y = (1, 0.5)$. All values computed in Python.` },
      { t: "steps", items: [
        { title: "Branch forward", tex: String.raw`z = W_1x = (0.5 + 0.2,\ 0.3 - 0.8) = (0.7,\ -0.5),\quad a = \mathrm{ReLU}(z) = (0.7,\ 0),\quad F = W_2a = (0.28,\ -0.14)` },
        { title: "Block output", tex: String.raw`y = x + F = (1.28,\ -1.14)` },
        { title: "Branch Jacobian (only unit 1 is active, D = diag(1, 0))", tex: String.raw`\frac{\partial F}{\partial x} = W_2DW_1 = \begin{pmatrix}0.4\\-0.2\end{pmatrix}\begin{pmatrix}0.5 & -0.2\end{pmatrix} = \begin{pmatrix}0.2 & -0.08\\ -0.1 & 0.04\end{pmatrix},\qquad J_y = \begin{pmatrix}1.2 & -0.08\\ -0.1 & 1.04\end{pmatrix}` },
        { title: "Backward: branch path, shortcut path, total", tex: String.raw`\Big(\frac{\partial F}{\partial x}\Big)^{\!\top}g = (0.2 - 0.05,\ -0.08 + 0.02) = (0.15,\ -0.06),\qquad \frac{\partial\mathcal L}{\partial x} = g + (0.15,\ -0.06) = (1.15,\ 0.44)` },
      ] },
      { t: "p", text: String.raw`Without the shortcut the gradient at $x$ would be $(0.15, -0.06)$ — about 14% of $\|g\|$, coming through a single active unit. With the shortcut it is $(1.15, 0.44)$: the upstream gradient plus a correction.` },
      { t: "h", text: "3. Unrolling many blocks: a product becomes a sum" },
      { t: "p", text: String.raw`For blocks $h^{[l]} = h^{[l-1]} + F_l(h^{[l-1]})$, $l = 1, \dots, L$, adding up the block equations gives the forward relation used by He et al. (2016, *Identity Mappings*):` },
      { t: "math", tex: String.raw`h^{[L]} = h^{[l]} + \sum_{i=l+1}^{L} F_i\big(h^{[i-1]}\big)\qquad\Longrightarrow\qquad \frac{\partial\mathcal L}{\partial h^{[l]}} = \frac{\partial\mathcal L}{\partial h^{[L]}}\Bigg(I + \frac{\partial}{\partial h^{[l]}}\sum_{i=l+1}^{L} F_i\big(h^{[i-1]}\big)\Bigg)`, caption: "Written with row-vector gradients. The term ∂L/∂h[L] reaches every earlier block directly, with no weight layer in between." },
      { t: "p", text: String.raw`In words: the gradient at any block contains the output gradient, undiminished. For it to vanish, the second term would have to cancel it exactly, which is unlikely for a mini-batch. In Jacobian form, with $J_l = \partial F_l/\partial h^{[l-1]}$:` },
      { t: "math", tex: String.raw`\frac{\partial h^{[L]}}{\partial h^{[0]}} = \prod_{l=L}^{1}\big(I + J_l\big) = I + \sum_{l} J_l + \sum_{l > m} J_lJ_m + \sum_{l > m > k} J_lJ_mJ_k + \cdots + J_L\cdots J_1` },
      { t: "p", text: String.raw`Expanding the product gives $2^L$ terms, one per subset of blocks. Each term is a **path** that goes through the branches in that subset and skips the rest. A plain network has only the last term: the single path through every layer, whose size shrinks or grows exponentially. $\binom{L}{k}$ paths use exactly $k$ branches, so path lengths follow a binomial distribution. With $L = 30$ there are $2^{30} = 1{,}073{,}741{,}824$ paths, the mean length is 15, and 95.7% of paths use between 10 and 20 branches.` },
      { t: "p", text: String.raw`**Scalar example.** Three blocks with branch derivatives $a = (0.2, -0.5, 0.1)$. Residual: $(1 + 0.2)(1 - 0.5)(1 + 0.1) = 0.66$. Expanded: $1 + (0.2 - 0.5 + 0.1) + (0.2\cdot(-0.5) + 0.2\cdot0.1 + (-0.5)\cdot0.1) + 0.2\cdot(-0.5)\cdot0.1 = 1 - 0.2 - 0.13 - 0.01 = 0.66$. Plain: only the last term, $-0.01$. Sixty-six times larger, and dominated by the short paths.` },
      { t: "h", text: "4. Which paths carry the gradient? The ensemble view" },
      { t: "p", text: String.raw`Many paths does not mean important paths. A path through $k$ branches multiplies $k$ Jacobians, usually each of norm below 1, so long paths contribute exponentially little. The NumPy program splits a real backward pass through 30 linearised blocks (branch scale $1/\sqrt{30}$) by path length. Paths through 0, 1, 2, 3, 4 branches carry gradient norms 8.68, 8.47, 5.67, 3.59, 1.78; the 155 million paths of length 15 together carry $5.7\times10^{-7}$. All lengths summed reproduce the full backward pass exactly. Veit, Wilber and Belongie (2016, *"Residual Networks Behave Like Ensembles of Relatively Shallow Networks"*) showed the same for trained ResNets: deleting one block at test time hurts accuracy only modestly, while deleting a layer from a plain VGG-style network destroys it. A deep ResNet acts like a large ensemble of shallower, weight-sharing networks.` },
      { t: "h", text: "5. Why the shortcut must be an identity: scaled and gated shortcuts" },
      { t: "p", text: String.raw`Suppose the shortcut multiplies by a constant $\lambda$: $h^{[l]} = \lambda h^{[l-1]} + F_l(h^{[l-1]})$. The direct term from $h^{[0]}$ to $h^{[L]}$ becomes $\lambda^L h^{[0]}$, and the gradient's direct term is scaled by $\lambda^L$: $0.9^{50} = 0.0052$, $1.1^{50} = 117$. The exponential problem is back. He et al. (2016, *Identity Mappings*) also tried gated and $1\times1$-convolution shortcuts on the main path; they trained worse than the pure identity because they block or rescale the direct path. The exercise below measures this for highway layers: a carry gate of about 0.87 per layer gives a direct-path factor $0.87^{50} \approx 9\times10^{-4}$ over 50 layers.` },
      { t: "h", text: "6. Forward variance grows — and how to keep it bounded" },
      { t: "p", text: String.raw`If $F_l(h)$ is roughly uncorrelated with $h$, then $\mathrm{Var}(h^{[l]}) = \mathrm{Var}(h^{[l-1]}) + \mathrm{Var}(F_l)$. If every branch preserves variance (He-initialised, $\mathrm{Var}(F_l) \approx \mathrm{Var}(h^{[l-1]})$), the variance *doubles* per block: in the NumPy program, 50 ReLU blocks with branch scale 1 give output std $1.7\times10^7$ and input-gradient norm $1.4\times10^7$. Fixes: scale each branch by about $1/\sqrt L$ (std stays 1.6, gradients 0.7–1.2 across all 50 layers); start branches at zero (std 1.0, gradient exactly equal at every layer: 0.708); or normalise inside the branch. With branches of variance $v$, the stream variance after $L$ blocks is $\mathrm{Var}(h^{[0]}) + Lv$ — linear, not exponential, growth. That is why pre-LN Transformers end with a final LayerNorm.` },
      { t: "h", text: "7. Projection shortcuts" },
      { t: "p", text: String.raw`When $F$ changes the shape — doubling channels from $C$ to $2C$ and halving resolution with stride 2 — the identity cannot be added. He et al. (2016) compared: (A) identity with zero-padded extra channels (and stride-2 subsampling), no parameters; (B) projections $W_sx$ only where shapes change, identity elsewhere; (C) projections everywhere. B and C beat A slightly, and C was only marginally better than B at a large cost, so B became standard. A projection is a $1\times1$ convolution with stride 2:` },
      { t: "math", tex: String.raw`y = F(x) + W_s x,\qquad W_s \in \mathbb R^{2C\times C\times1\times1}\ \ (\text{stride } 2),\qquad \#\text{params}(W_s) = C\cdot 2C` },
      { t: "p", text: String.raw`For $C = 64$: $64\cdot128 = 8{,}192$ weights, small next to the block's two $3\times3$ convolutions ($9\cdot64\cdot128 + 9\cdot128\cdot128 = 221{,}184$). The program below reports 229,376 convolution weights for the whole downsampling block.` },
      { t: "h", text: "8. Basic vs bottleneck blocks: parameter counts" },
      { t: "p", text: String.raw`A **basic block** has two $3\times3$ convolutions with $C$ channels: $2\cdot9C^2$ weights. A **bottleneck block** (ResNet-50 and deeper) carries a wide $4C$-channel stream but does its $3\times3$ convolution on only $C$ channels: $1\times1$ reduce $4C\to C$, $3\times3$ $C\to C$, $1\times1$ expand $C\to4C$.` },
      { t: "math", tex: String.raw`\text{basic}(C) = 18C^2,\qquad \text{bottleneck}(C) = 4C^2 + 9C^2 + 4C^2 = 17C^2,\qquad \text{plain two } 3\times3 \text{ on } 4C = 2\cdot9\cdot(4C)^2 = 288C^2` },
      { t: "p", text: String.raw`With $C = 64$: basic block 73,728 weights; bottleneck on a 256-channel stream 69,632; two $3\times3$ convolutions straight on 256 channels 1,179,648 — 17× more. The bottleneck lets ResNet-50 be deeper and wider than ResNet-34 at a similar budget per block. (Counts checked by the PyTorch program; BatchNorm adds $2\times$channels per BN layer.)` },
      { t: "h", text: "9. Pre-activation ordering" },
      { t: "p", text: String.raw`In the original block $y = \mathrm{ReLU}(x + F(x))$, the ReLU after the addition sits on the shortcut path. The identity is broken at every block, and channels that are negative after the sum are zeroed. The pre-activation block (He et al., 2016, *Identity Mappings*) reorders the branch as BN → ReLU → conv → BN → ReLU → conv and leaves the sum alone:` },
      { t: "math", tex: String.raw`\text{original: } y = \mathrm{ReLU}\big(x + \mathrm{BN}(W_2\,\mathrm{ReLU}(\mathrm{BN}(W_1x)))\big)\qquad \text{pre-activation: } y = x + W_2\,\mathrm{ReLU}\big(\mathrm{BN}(W_1\,\mathrm{ReLU}(\mathrm{BN}(x)))\big)` },
      { t: "p", text: String.raw`Now $h^{[L]} = h^{[l]} + \sum F_i$ holds exactly (Section 3) and the branch input is always normalised. The authors trained 1001-layer CIFAR networks and 200-layer ImageNet networks that the original ordering handled less well. The same reasoning led to the pre-LN Transformer (see *Residual streams in Transformers* under Types and Variations).` },
    ],
    visual: [
      { t: "p", text: String.raw`The `+"`gradient-flow`"+String.raw` simulator implements $h^{[l]} = h^{[l-1]} + f(W^{[l]}h^{[l-1]})$ (optionally with LayerNorm before $f$) on a real batch of 32 inputs. Tick **residual connections** to switch every layer between plain and residual. The top chart shows the std of each layer's output (the residual stream), the bottom chart the per-layer gradient norms, and the narration quotes the first-to-last ratios. Numbers below come from the simulator's own code at its default sample (width 64, depth 30, MSE loss).` },
      { t: "sim", id: "gradient-flow", caption: "Toggle 'residual connections' on and off in each configuration; compare the shape of the gradient bars and the growth of the forward std." },
      { t: "list", items: [
        String.raw`**Rescue a vanishing network.** Start from the default **Sigmoid, Xavier (vanishing)**: $\|\delta^{[1]}\|/\|\delta^{[30]}\| \approx 2.9\times10^{-19}$, a straight ramp on the log scale. Tick **residual connections** (or press **Sigmoid + residual**): the ratio becomes 3.8 and the bars are nearly flat. The shortcut gave every layer a gradient path with no sigmoid slope in it.`,
        String.raw`**Same for a collapsed network.** **Tanh, σ = 0.05 (collapse)**: the forward std decays to $9\times10^{-13}$ and the ratio is $2.2\times10^{-12}$. With residual on, the forward std stays between 1.07 and 3.25 and the ratio is 3.3.`,
        String.raw`**Residual alone can explode.** Choose **ReLU, He (healthy)** and tick residual: the stream std grows from 1.3 at layer 1 to 94 at layer 10 and $4.6\times10^6$ at layer 30, and gradients become huge. Each He-initialised branch adds as much variance as the stream already has (math Section 6).`,
        String.raw`**Small branches fix it.** Keep residual on, select **Normal σ**, drag σ to 0.01: the std stays between 1.01 and 1.07 at every layer and $\|\delta^{[1]}\|/\|\delta^{[30]}\| \approx 1.2$. σ = 0.05 gives std 27 at layer 30, and 0.1 gives 4300. At depth 50 with σ = 0.01 it is still healthy (std 1.18 at layer 50); the plain network with the same weights has output std around $10^{-63}$.`,
        "**Normalisation in the branch.** With **ReLU**, **He**, residual and **layer normalisation** ticked, the stream grows only gently (1.16 → 2.09 → 3.10): each branch adds a bounded amount of variance — linear, not exponential, growth.",
        "Look at the histogram of the last layers in **Sigmoid + residual**: those sigmoid units are saturated (their input, the residual stream, has grown), yet the gradient still arrives through the identity path. Residual connections protect gradient flow, not the health of each layer.",
      ] },
    ],
    scratch: [
      { t: "p", text: String.raw`A NumPy program in three parts. (1) A residual block's forward and backward pass, with the Jacobian $I + W_2DW_1$ checked against finite differences and the singular values of branch and block compared. (2) The gradient norm $\|\partial\mathcal L/\partial h\|$ at the input of layers 1, 10, 25 and 50 of 50-layer plain and residual networks with different branch scales. (3) The unrolled view: counting paths, and splitting a real backward pass through 30 linearised blocks by path length.` },
      { t: "code", lang: "python", title: "residual_numpy.py", code: RES_NUMPY, output: RES_NUMPY_OUT },
      { t: "h", text: "Reading the output" },
      { t: "list", items: [
        String.raw`The analytic block Jacobian matches finite differences to $5\times10^{-11}$, and `+"`block_backward`"+" — just `dy + branch_backward(dy)` — equals $J^\\top dy$.",
        String.raw`**Plain He** keeps gradients of order 1 in this 50-layer net at initialisation (that is what He initialisation is for); **plain Xavier** loses them ($3.5\times10^{-8}$ at layer 1). Initialisation alone is fragile: one factor of $\sqrt2$ decides between working and not.`,
        String.raw`**Residual with full-scale branches** fails the other way: gradients at layer 1 are $1.4\times10^7$ and the output std $1.7\times10^7$.`,
        String.raw`**Residual with branch scale $1/\sqrt L$ or 0** is healthy at every depth. With scale 0 the gradient is *identical* at every layer (0.708): the network is exactly the identity and the shortcut carries everything.`,
        String.raw`In the path split, short paths (0–4 branches) carry almost all the gradient, not the most numerous ones (length 15). The DP update `+"`parts[1:] = parts[1:] + parts[:-1] @ J`"+String.raw` expands $(I + J)$ one block at a time: a path either skips the block (stays at length $k$) or uses its branch (moves to $k + 1$).`,
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**1. Residual blocks as `nn.Module`s.** A basic block with an optional projection shortcut, a bottleneck block and a pre-activation block, with output shapes and parameter counts. Finally, a check that a pre-activation block with a zero-initialised last convolution is exactly the identity and passes gradient 1 to every input element:" },
          { t: "code", lang: "python", title: "resnet_blocks.py", code: RES_TORCH_BLOCKS, output: RES_TORCH_BLOCKS_OUT },
          { t: "p", text: String.raw`The stride-2 block halves the resolution (32 → 16) and doubles the channels; its $1\times1$ projection shortcut has $64\cdot128 = 8{,}192$ weights. The "all params" column adds the BatchNorm $\gamma$ and $\beta$ (2 per channel per BN layer).` },
          { t: "p", text: String.raw`**2. The degradation problem, reproduced.** MLPs on the `+"`sklearn`"+String.raw` digits (1347 training / 450 test images): a stem layer, $N$ two-layer blocks (Linear → BN → ReLU → Linear → BN) and a classifier, so $2N + 2$ weight layers. Three variants per depth: **plain** ($\mathrm{ReLU}(F(x))$), **residual** ($\mathrm{ReLU}(x + F(x))$, the original ResNet layout), and **residual with zero-γ** (last BN of each branch starts at $\gamma = 0$). He initialisation, SGD with momentum 0.9, learning rate 0.01, batch 64, 20 epochs. It records gradient norms at initialisation (first/last block input via forward pre-hooks and `+"`retain_grad`"+"; stem weights), the training loss of epochs 1, 5, 10 and 20, and final accuracies. About 15–20 s on one CPU thread." },
          { t: "code", lang: "python", title: "degradation_experiment.py", code: RES_TORCH_DEGRADATION, output: RES_TORCH_DEGRADATION_OUT },
          { t: "list", items: [
            "**Degradation in plain networks.** Plain training loss after 20 epochs: 0.072 (4 layers) → 1.294 (10 layers) → 2.230 (34 layers, barely below the 2.303 of guessing). Training accuracy: 99.5% → 57.5% → 14.2%. The deeper networks can represent strictly more, so this is an optimisation failure — on the *training* set, exactly as He et al. saw.",
            String.raw`**It is not vanishing gradients.** The plain networks use He initialisation and BatchNorm, and at 34 layers the gradient at the first block's input is $1.88\times10^{1}$ — about 400× *larger* than at the last block ($4.8\times10^{-2}$). Deep BatchNorm networks without shortcuts are known to have gradients that grow toward the input at initialisation (Yang et al., 2019, *"A Mean Field Theory of Batch Normalization"*). Either way the optimisation landscape is badly conditioned.`,
            "**Shortcuts remove degradation — with the right initialisation.** Residual with zero-γ: training loss 0.092 → 0.079 → 0.043 and training accuracy 98.3% → 98.6% → 99.6% as depth grows from 4 to 34 layers; test accuracy 96.9–97.1% at every depth. Deeper is now at least as good as shallower. The gradient at every block's input is identical at initialisation (0.0239, 0.0258, 0.0251): every block is the identity, so the 34-layer network starts as a 2-layer one.",
            "**Shortcuts alone are not the whole fix.** With ordinary γ = 1 the residual network trains well at 4 layers (best of all: 0.027) but is unstable when deep: at 10 layers the loss jumps to 9.1 in epoch 5 before recovering; at 34 layers it explodes to $5.3\\times10^6$, ending with 90.2% training accuracy but a huge loss. Each branch adds unit-variance BN output, so the stream grows with depth, and momentum SGD becomes unstable. Zero-γ (Goyal et al., 2017) or the pre-activation layout are the standard cures — see [Weight Initialization](/learn/m07/initialization).",
          ] },
          { t: "callout", kind: "note", text: "Single runs with one seed and one learning rate; exact numbers move with the seed, and the γ = 1 residual runs are especially sensitive to the learning rate. The robust findings: plain networks get steadily worse with depth, and zero-γ residual networks get steadily better." },
          { t: "p", text: String.raw`**3. Residual streams in Transformers: post-LN vs pre-LN.** The original Transformer applies LayerNorm *after* each residual addition, $x_{l+1} = \mathrm{LN}(x_l + F(x_l))$ (post-LN). GPT-2 and most later models apply it at the *input* of each branch, $x_{l+1} = x_l + F(\mathrm{LN}(x_l))$ (pre-LN), with one final LayerNorm. The difference shows in the stream statistics of 24 feed-forward sub-layers at initialisation:` },
          { t: "code", lang: "python", title: "pre_vs_post_ln.py", code: RES_TORCH_LN, output: RES_TORCH_LN_OUT },
          { t: "p", text: "Post-LN renormalises the stream after every addition (std exactly 1.000), interrupting the identity path at every layer. Pre-LN leaves it alone, so the stream piles up branch outputs and its std grows slowly (1.02 → 1.40 after 24 blocks) — hence the final LayerNorm. Xiong et al. (2020, *\"On Layer Normalization in the Transformer Architecture\"*) showed post-LN Transformers have large gradients near the output at initialisation and need learning-rate warm-up, while pre-LN ones can train without it. See [Transformer Architecture](/learn/m11/transformer-architecture)." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: String.raw`In the Keras functional API a residual connection is literally `+"`layers.Add()([branch_output, shortcut])`"+String.raw`. This small ResNet for the $8\times8$ digits uses basic blocks, a projection shortcut where the channels double and the resolution halves, and zero-γ in the last BN of each branch:` },
          { t: "code", lang: "python", title: "keras_resnet_digits.py (not executed in CI)", code: RES_KERAS, output: "model.summary() lists the Add layers joining each branch and shortcut (and the extra Conv2D/BatchNormalization of the projection shortcut in the stride-2 block); fit() prints the training and validation loss and accuracy for each of 10 epochs. Exact values depend on TensorFlow's random initialisation and are not reproduced here." },
          { t: "p", text: "`tf.keras.applications.ResNet50` (and `ResNet50V2`, the pre-activation variant) build the full ImageNet architectures, with optional pre-trained weights." },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "Families of shortcut connections" },
      { t: "table", head: ["Variant", "Form", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Basic residual block", "ReLU(x + BN(conv(ReLU(BN(conv x)))))", "simple; identity shortcut costs nothing", "ReLU after the sum breaks the pure identity path", "ResNet-18/34, small CNNs"],
        ["Bottleneck block", "1×1 reduce → 3×3 → 1×1 expand, + x", "17C² vs 288C² weights for a 4C-wide stream", "more layers per block; 1×1 convs are memory-bound", "ResNet-50/101/152, ResNeXt"],
        ["Projection shortcut", "W_s x (1×1 conv, stride s) + F(x)", "handles channel/resolution changes", "parameters; not an identity", "first block of each stage"],
        ["Pre-activation block", "x + conv(ReLU(BN(conv(ReLU(BN x)))))", "pure identity path; trains 1000+ layers", "needs a final BN-ReLU before the head", "ResNet-v2, very deep CNNs"],
        ["Highway layer", "T(x)·H(x) + (1 − T(x))·x", "learned, input-dependent mixing", "2× parameters; carry gate < 1 shrinks the direct path", "historical; gating ideas in LSTMs/GRUs"],
        ["Dense connections", "x_l = H([x_0, …, x_{l−1}]) (concatenation)", "feature reuse; few parameters per layer", "channel count and memory grow linearly with depth", "DenseNet"],
        ["Pre-LN residual (Transformers)", "x + F(LN(x)), final LN", "stable without warm-up; clean identity path", "stream variance grows with depth", "GPT-2 onwards, most LLMs, ViT"],
        ["Post-LN residual", "LN(x + F(x))", "bounded stream; the original Transformer", "large gradients near the output at init; needs warm-up", "original Transformer, BERT"],
        ["U-Net skips", "concat(encoder feature, upsampled decoder feature)", "restores spatial detail lost by downsampling", "memory for stored encoder maps", "segmentation, diffusion models"],
        ["Stochastic depth", "drop whole residual branches at random in training", "regularises; shortens expected depth", "extra hyperparameter (drop rate)", "deep ResNets, ViTs"],
      ] },
      { t: "h", text: "Highway networks: the gated predecessor" },
      { t: "p", text: String.raw`Srivastava, Greff and Schmidhuber (2015, *"Highway Networks"* and *"Training Very Deep Networks"*) let information skip a layer through a learned gate, inspired by LSTMs: $y = T(x)\odot H(x) + (1 - T(x))\odot x$, with transform gate $T(x) = \sigma(W_Tx + b_T)$. A negative $b_T$ (they suggested about $-1$ to $-3$) makes layers carry their input at first. They trained networks of hundreds of layers. A residual block is the special case with both gates fixed open ($T = 1$ on the branch, carry $= 1$ on the shortcut). Fixing them helps because a carry gate below 1 multiplies the direct path, and over many layers that product shrinks (exercise 2 measures it).` },
      { t: "h", text: "DenseNet: concatenate instead of add" },
      { t: "p", text: String.raw`Huang, Liu, van der Maaten and Weinberger (2017, *"Densely Connected Convolutional Networks"*) connect each layer to *all* earlier layers in a dense block, concatenating instead of adding: $x_l = H_l([x_0, x_1, \dots, x_{l-1}])$. Each layer adds only $k$ new channels (the growth rate), so layer $l$ sees $k_0 + (l-1)k$ channels. With $k_0 = 64$, $k = 32$, layers 1–6 see 64, 96, 128, 160, 192, 224 channels and the block outputs 256. Addition merges features into a fixed-width stream (cheap, but overwrites earlier features); concatenation keeps them (reuse, few parameters, more activation memory). Transition layers ($1\times1$ conv + pooling) between dense blocks reset the width.` },
      { t: "h", text: "Skip connections in U-Net" },
      { t: "p", text: "Ronneberger, Fischer and Brox (2015, *\"U-Net: Convolutional Networks for Biomedical Image Segmentation\"*) link each encoder stage to the same-resolution decoder stage by **concatenating** the encoder's feature map onto the upsampled decoder features. These long skips bring back fine detail lost in downsampling; they mainly move information across scales, though they also help gradient flow. Diffusion models' denoising U-Nets use both kinds: residual blocks inside each stage and concatenation skips across the U. See [Segmentation](/learn/m12/segmentation)." },
      { t: "h", text: "Residual streams in Transformers" },
      { t: "p", text: String.raw`Every Transformer sub-layer — self-attention and the MLP — is wrapped in a residual connection. So a token's representation is a running sum $x_L = x_0 + \sum_l F_l(\cdot)$ that every sub-layer reads from and writes into. Interpretability research uses this *residual stream* view heavily. Post-LN (original Transformer, BERT) normalises after each addition; pre-LN (GPT-2, most large language models, ViT) normalises the branch input and adds one final LayerNorm. Scaled initialisation of the output projections ($1/\sqrt N$ in GPT-2) keeps the stream's variance growth in check — the Transformer version of zero-γ.` },
    ],
    applications: [
      { t: "list", items: [
        "**Image classification backbones.** ResNet-50 is still a standard baseline and feature extractor; ResNeXt, Wide ResNets, RegNets and ConvNeXt keep the residual block. Detection and segmentation systems (Faster R-CNN, Mask R-CNN, DeepLab) often use ResNet backbones. See [Classic CNN Architectures](/learn/m08/cnn-architectures).",
        "**Transformers and large language models.** Every block of BERT, GPT-style models and Vision Transformers is two residual sub-layers; without shortcuts, stacks of dozens of attention layers would not train. See [Transformer Architecture](/learn/m11/transformer-architecture).",
        "**Segmentation and image generation.** U-Net's concatenation skips are the standard for biomedical segmentation and the backbone of diffusion-model denoisers, which also use residual blocks in each stage. See [Segmentation](/learn/m12/segmentation).",
        "**Speech and audio.** WaveNet-style models stack dilated convolutions in residual blocks with skip connections to the output; Conformer-style speech encoders wrap their sub-layers in residual connections.",
        "**Graph neural networks and very deep MLPs.** Residual connections are the standard way to stack more message-passing or MLP layers without degradation.",
        "**Fine-tuning and adapters.** Adapter and LoRA-style fine-tuning add small residual branches (starting at or near zero) to a frozen pre-trained model. The residual form guarantees training starts exactly at the pre-trained function.",
      ] },
    ],
    tradeoffs: [
      { t: "h", text: "Strengths" },
      { t: "list", items: [
        "Removes the degradation problem: deeper networks train at least as well as shallow ones (in our experiment, 34-layer zero-γ residual MLP 0.043 training loss vs 2.230 for the plain one).",
        "An identity shortcut costs no parameters and one addition per element.",
        String.raw`Gives every layer a direct gradient path (Jacobian $I + J$), so gradient flow survives imperfect initialisation and saturating layers.`,
        "Makes 'do nothing' the easy default, which enables zero-initialised branches, stochastic depth, adapters and step-by-step fine-tuning.",
      ] },
      { t: "h", text: "Limitations" },
      { t: "list", items: [
        "**Forward variance grows** with depth unless branches are scaled, zero-initialised or normalised. Left alone, this caused the loss explosion of the γ = 1 residual runs above.",
        String.raw`**Shape constraint**: $x$ and $F(x)$ must match, so downsampling blocks need projection shortcuts and architectures are built in stages.`,
        "**Memory**: the block input must be kept for the addition and for backpropagation; concatenation skips (U-Net, DenseNet) cost even more activation memory.",
        "**Effective depth is less than nominal depth**: much of the gradient flows through short paths, so extra blocks give diminishing returns; beyond a point, width and data often matter more.",
        "**Not a cure-all**: residual connections do not prevent saturated or dead units inside branches, overfitting, or a bad learning rate.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Treating degradation as overfitting.** The defining symptom is higher *training* error for the deeper plain network. Regularisation will not fix it; shortcuts (or a shallower network) will.",
        "**Adding tensors of different shapes.** If a block changes channels or uses stride 2, `x + F(x)` fails or, worse, silently broadcasts (e.g. a (B, 1) and a (B, C) tensor). Use a projection shortcut (1×1 conv with the same stride, plus BN) whenever shapes differ.",
        "**Putting a nonlinearity or normalisation on the main path in very deep nets.** A ReLU after every sum (original ResNet) or a LayerNorm after every sum (post-LN) interrupts the identity path. For very deep stacks prefer pre-activation / pre-LN with a final normalisation.",
        "**Forgetting the final normalisation in pre-activation / pre-LN networks.** The stream is never normalised inside the blocks, so the head sees a raw, growing signal unless you add a final BN-ReLU or LayerNorm.",
        String.raw`**Full-scale branches in a deep network without normalisation.** The stream variance doubles per block (simulator: ReLU + He + residual reaches std $4.6\times10^6$). Zero-initialise the last layer/γ of each branch or scale branches by $\approx1/\sqrt L$.`,
        String.raw`**Scaling or gating the shortcut 'to control variance'.** $\lambda^L$ brings back exponential vanishing or explosion ($0.9^{50} = 0.005$). Control the branch, not the shortcut.`,
        String.raw`**Modifying the shortcut tensor in place.** If the branch starts with an in-place operation on its input (`+"`x.relu_()`, or `nn.ReLU(inplace=True)` applied directly to `x`"+String.raw`), the tensor the shortcut later adds has already changed. The block silently computes $\mathrm{ReLU}(x) + F(x)$, or autograd raises 'a variable needed for gradient computation has been modified by an inplace operation'. Never modify `+"`x`"+" in place inside a residual block; in-place ops on the branch's own intermediate tensors are fine.",
        "**Confusing U-Net/DenseNet concatenation with residual addition.** Concatenation doubles channels and needs the next layer sized for it; addition keeps the width fixed. Different operations, different costs.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is a residual connection?", a: String.raw`An identity shortcut that adds a block's input to its output: $y = x + F(x)$. The block's layers only learn the residual $F(x) = \mathcal H(x) - x$. If the block has nothing useful to add, $F \approx 0$ and the input passes through unchanged.` },
      { t: "qa", level: "beginner", q: "What is the degradation problem?", a: "Beyond some depth, plain (shortcut-free) networks get worse *training* error as they get deeper, even with batch normalisation and good initialisation. The deeper network could in principle copy the shallower one and make its extra layers the identity, so this is an optimisation failure, not overfitting. He et al. (2016) showed it on CIFAR-10 and ImageNet and introduced residual learning to fix it." },
      { t: "qa", level: "beginner", q: "What do you do when the input and output of a residual block have different shapes?", a: String.raw`Use a projection shortcut: a $1\times1$ convolution (same stride as the block, usually followed by BN) that maps $x$ to the right channels and resolution, $y = F(x) + W_sx$. The ResNet paper's alternative, an identity with zero-padded extra channels and strided subsampling, has no parameters but did slightly worse.` },
      { t: "qa", level: "intermediate", q: "Derive the gradient through a residual block and explain why it helps.", a: String.raw`$y = x + F(x)$ gives $\partial y/\partial x = I + \partial F/\partial x$, so $\partial\mathcal L/\partial x = \partial\mathcal L/\partial y + (\partial F/\partial x)^\top\partial\mathcal L/\partial y$. Over $L$ blocks, the gradient at block $l$ contains $\partial\mathcal L/\partial h^{[L]}$ directly (from $h^{[L]} = h^{[l]} + \sum_i F_i$). The product $\prod(I + J_i)$ expands into $2^L$ path terms including the identity, so the gradient cannot vanish exponentially the way a plain product $\prod J_i$ can.` },
      { t: "qa", level: "intermediate", q: "Compare basic and bottleneck blocks.", a: String.raw`Basic: two $3\times3$ convolutions on $C$ channels, $18C^2$ weights. Bottleneck: $1\times1$ reducing $4C \to C$, $3\times3$ on $C$, $1\times1$ expanding back to $4C$: $17C^2$ weights for a $4C$-wide stream (two $3\times3$ convs on $4C$ channels would cost $288C^2$). For $C = 64$: 73,728 vs 69,632 vs 1,179,648. Bottlenecks allow deeper/wider networks (ResNet-50+) at similar cost.` },
      { t: "qa", level: "intermediate", q: "What changed in the pre-activation ResNet and why?", a: String.raw`BN and ReLU moved from after the addition to the start of the branch: $y = x + W_2\mathrm{ReLU}(\mathrm{BN}(W_1\mathrm{ReLU}(\mathrm{BN}(x))))$. Nothing is applied after the sum, so the shortcut is a pure identity through the whole network, $h^{[L]} = h^{[l]} + \sum F_i$ holds exactly, and signal and gradient travel directly between any two blocks. This made 1000-layer CIFAR ResNets trainable and improved deep ImageNet models. A final BN-ReLU is needed before the classifier.` },
      { t: "qa", level: "intermediate", q: "How do residual connections in Transformers differ between post-LN and pre-LN, and which is used today?", a: String.raw`Post-LN (original Transformer, BERT): $x_{l+1} = \mathrm{LN}(x_l + F(x_l))$, normalising the stream after each addition. Pre-LN (GPT-2, most later LLMs, ViT): $x_{l+1} = x_l + F(\mathrm{LN}(x_l))$ plus a final LayerNorm, leaving the identity path untouched. Xiong et al. (2020) showed post-LN has large gradients near the output at initialisation and needs warm-up. Pre-LN is more stable, at the cost of a stream whose variance grows with depth (measured: std 1.0 → 1.4 over 24 sub-layers).` },
      { t: "qa", level: "advanced", q: "Explain the 'ensemble of shallow networks' view of ResNets and the evidence for it.", a: String.raw`Expanding $\prod_l(I + J_l)$ gives $2^L$ paths, $\binom Lk$ of them using $k$ branches, so most paths have length about $L/2$. But each branch Jacobian usually has norm below 1, so long paths contribute exponentially little and most gradient flows through short paths (in our split of 30 blocks, lengths 0–4 dominate; length-15 paths carry $\sim10^{-7}$). Veit, Wilber and Belongie (2016) showed that deleting single blocks of a trained ResNet at test time only modestly hurts accuracy, unlike in a plain network — consistent with an ensemble of many weight-sharing shallower networks.` },
      { t: "qa", level: "advanced", q: "Residual connections fix gradient flow; why do deep ResNets still need BatchNorm, zero-γ or Fixup?", a: String.raw`The forward signal grows. With branches uncorrelated with their input, $\mathrm{Var}(h^{[l]}) = \mathrm{Var}(h^{[l-1]}) + \mathrm{Var}(F_l)$, which doubles per block for variance-preserving branches (measured: std $1.7\times10^7$ after 50 He-initialised ReLU blocks). Large activations make the loss and gradients explode. Normalisation inside the branch bounds each increment (linear growth); zero-γ / Fixup start each branch at 0, so the network starts shallow; $1/\sqrt N$ scaling (GPT-2) balances the sum. In our digits experiment the γ = 1 residual network diverged to a loss of $5\times10^6$ at 34 layers, while the zero-γ version reached 99.6% training accuracy.` },
      { t: "qa", level: "advanced", q: "Why not learn the shortcut, e.g. with a highway gate or a scalar λ?", a: String.raw`Any multiplier on the main path compounds. A constant $\lambda$ gives a direct-path factor $\lambda^L$ ($0.9^{50} = 0.005$), and a highway carry gate $1 - T(x) < 1$ does the same, input by input (measured: 50 highway layers with gate bias −2 give an input gradient of $4\times10^{-5}$ vs 1.35 for residual layers). He et al. (2016) found gated and convolutional shortcuts trained worse than identity shortcuts. A fixed identity shortcut, with the branch doing the learning, guarantees a factor of exactly 1.` },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "How many weights do the three convolutions of a bottleneck block with a 512-channel stream and bottleneck width 128 have (1×1 512→128, 3×3 128→128, 1×1 128→512, no biases)?", answer: 278528, tolerance: 1, explain: String.raw`$512\cdot128 + 9\cdot128\cdot128 + 128\cdot512 = 65{,}536 + 147{,}456 + 65{,}536 = 278{,}528$, i.e. $17C^2$ with $C = 128$.` },
        { kind: "numeric", q: "How many distinct paths (subsets of branches) does the unrolled gradient of a network with 10 residual blocks contain?", answer: 1024, tolerance: 0, explain: String.raw`Each block is either skipped or used: $2^{10} = 1024$ paths.` },
        { kind: "numeric", q: String.raw`Three scalar residual blocks have branch derivatives 0.2, −0.5 and 0.1. What is $\partial h^{[3]}/\partial h^{[0]}$? (2 decimals)`, answer: 0.66, tolerance: 0.005, explain: String.raw`$(1.2)(0.5)(1.1) = 0.66$. The plain chain would give $0.2\cdot(-0.5)\cdot0.1 = -0.01$.` },
        { kind: "numeric", q: "A downsampling block maps 128 to 256 channels with stride 2. How many weights does its 1×1 projection shortcut have (no bias)?", answer: 32768, tolerance: 0, explain: String.raw`$128\cdot256\cdot1\cdot1 = 32{,}768$.` },
        { kind: "numeric", q: "If every shortcut multiplied its input by λ = 0.9, by what factor would the direct gradient path be scaled across 50 blocks? (4 decimals)", answer: 0.0052, tolerance: 0.0002, explain: String.raw`$0.9^{50} = 0.00515$ — exponential shrinking returns, which is why the shortcut should be an exact identity.` },
        { q: "The 'degradation problem' refers to deeper plain networks having…", options: ["…higher test error but lower training error (overfitting)", "…higher training error than shallower plain networks", "…exploding parameter counts", "…slower inference"], answer: 1, explain: "It is an optimisation failure visible on the training set. A deeper network could copy the shallower one, so capacity is not the problem." },
        { q: "What distinguishes a pre-activation residual block from the original?", options: ["It concatenates instead of adding", "BN and ReLU are moved into the branch before each convolution, and nothing is applied after the addition", "It removes BatchNorm entirely", "It uses a gated shortcut"], answer: 1, explain: String.raw`$y = x + F(x)$ with $F$ = BN → ReLU → conv → BN → ReLU → conv: a clean identity path through the network.` },
        { q: "DenseNet differs from ResNet in that each layer…", options: ["…adds its output to a fixed-width stream", "…receives the concatenation of all earlier feature maps in its block", "…uses a learned gate on the shortcut", "…has no nonlinearity"], answer: 1, explain: String.raw`$x_l = H_l([x_0, \dots, x_{l-1}])$: concatenation, so the channel count grows by the growth rate $k$ per layer.` },
        { q: "A 40-block residual MLP without normalisation, He-initialised branches, produces NaN losses after a few steps. The best first fix is…", options: ["Remove the shortcuts", "Initialise the last layer of each branch to zero (or scale branches by ~1/√L)", "Multiply each shortcut by 0.5", "Switch the activation to sigmoid"], answer: 1, explain: "Full-scale branches double the stream variance per block; starting branches at (near) zero keeps the stream bounded. Scaling the shortcut brings back exponential shrinking." },
      ] },
      { t: "exercise", title: "Gradient paths and parameter budgets by hand", prompt: String.raw`(a) A network has 4 residual blocks with scalar branch derivatives $a = (0.5, -0.4, 0.3, 0.2)$ at the current input. Compute $\partial h^{[4]}/\partial h^{[0]}$ for the residual network and for the plain chain $\prod a_l$, and give the contribution of the paths of length 0 and length 1. (b) A ResNet stage turns a $64$-channel, $32\times32$ map into a $128$-channel, $16\times16$ map with a basic block. Count the convolution weights of the branch and of the projection shortcut. (c) How many weights would a bottleneck block need on a 512-channel stream with bottleneck width 128, compared with two plain $3\times3$ convolutions on 512 channels?`, hint: String.raw`Expand $\prod(1 + a_l)$; a path of length 1 uses exactly one branch. For convolutions: $C_{out}\cdot C_{in}\cdot k\cdot k$.`, solution: [
        { t: "steps", items: [
          { title: "(a) Residual product and the plain chain", tex: String.raw`\prod_{l}(1 + a_l) = 1.5\cdot0.6\cdot1.3\cdot1.2 = 1.404,\qquad \prod_l a_l = 0.5\cdot(-0.4)\cdot0.3\cdot0.2 = -0.012` },
          { title: "(a) Short paths", tex: String.raw`\text{length }0: 1,\qquad \text{length }1: 0.5 - 0.4 + 0.3 + 0.2 = 0.6` },
          { title: "(b) Downsampling basic block", tex: String.raw`\underbrace{128\cdot64\cdot9}_{73{,}728} + \underbrace{128\cdot128\cdot9}_{147{,}456} = 221{,}184\ \text{(branch)},\qquad 128\cdot64 = 8{,}192\ \text{(projection)},\qquad \text{total } 229{,}376` },
          { title: "(c) Bottleneck vs plain", tex: String.raw`512\cdot128 + 9\cdot128^2 + 128\cdot512 = 278{,}528\qquad\text{vs}\qquad 2\cdot9\cdot512^2 = 4{,}718{,}592\ (16.9\times\text{ more})` },
        ] },
        { t: "p", text: String.raw`In (a) the two shortest path groups already give $1.6$ of the total $1.404$ (longer paths subtract $0.196$); the plain network keeps only the single longest path, $-0.012$. The (b) total matches the 229,376 convolution weights the PyTorch program reports for `+"`BasicBlock(64, 128, stride=2)`." },
      ] },
      { t: "exercise", title: "Compare highway, residual and plain stacks", prompt: String.raw`In PyTorch, implement a highway layer $y = T\odot\tanh(W_Hx + b_H) + (1 - T)\odot x$ with $T = \sigma(W_Tx + b_T)$ and a configurable gate bias, a plain layer $\tanh(W_Hx + b_H)$ and a residual layer $x + \tanh(W_Hx + b_H)$. Stack 50 of each (width 64, PyTorch default initialisation) and report $\|\partial\mathcal L/\partial x\|$ at the input and the output std for an MSE loss against random targets. Use gate biases −2 and 0. Also report the mean gate value at initialisation, parameters per layer, and the channel counts inside a DenseNet block with $k_0 = 64$, growth rate 32 and 6 layers.`, hint: String.raw`Set `+"`requires_grad=True`"+" on the input and read `x.grad` after `backward()`. With gate bias $b_T$ and small weights, $T \\approx \\sigma(b_T)$.", solution: [
        { t: "code", lang: "python", title: "highway_vs_residual.py", code: RES_HIGHWAY, output: RES_HIGHWAY_OUT },
        { t: "list", items: [
          String.raw`**Plain tanh** stacks with PyTorch's default (small) initialisation lose the gradient entirely: $\|\partial\mathcal L/\partial x\| = 6.6\times10^{-15}$.`,
          String.raw`**Highway with gate bias −2** starts with $T \approx 0.13$, so each layer mostly carries its input. The gradient improves by ten orders of magnitude to $4.3\times10^{-5}$ but is still small: the carry factor $1 - T \approx 0.87$ multiplies the direct path at every layer ($0.87^{50} \approx 10^{-3}$), and the output std shrinks to 0.019. With gate bias 0 ($T \approx 0.5$) it is as bad as plain ($1.9\times10^{-14}$).`,
          "**Residual** layers keep the gradient at 1.35 because the direct path has factor exactly 1 — at the price of a growing stream (output std 5.4). This is the identity-mappings argument in numbers.",
          String.raw`Highway layers have twice the parameters of residual ones (8,320 vs 4,160 per layer: a second weight matrix and bias for the gate). The DenseNet block's layers see $64, 96, \dots, 224$ input channels and the block outputs 256.`,
        ] },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Degradation problem: deeper *plain* networks reach higher *training* error — an optimisation failure, not overfitting (ImageNet plain-34 vs plain-18; our digits MLP: plain-34 loss 2.23 vs plain-4 0.07).",
        String.raw`Residual block: $y = x + F(x)$; the identity is the default ($F = 0$). Projection shortcut $W_sx$ (1×1 conv, stride) when shapes differ.`,
        String.raw`Gradient: $\partial y/\partial x = I + \partial F/\partial x$; across blocks $h^{[L]} = h^{[l]} + \sum F_i$, so $\partial\mathcal L/\partial h^{[l]}$ always contains $\partial\mathcal L/\partial h^{[L]}$.`,
        String.raw`$\prod(I + J_l)$ = sum over $2^L$ paths, $\binom Lk$ of length $k$; short paths carry most of the gradient (ensemble view, Veit et al. 2016).`,
        String.raw`Keep the shortcut an exact identity: scaled or gated shortcuts give $\lambda^L$ factors ($0.9^{50} = 0.005$).`,
        String.raw`Forward variance grows ($\mathrm{Var}(h^{[l]}) = \mathrm{Var}(h^{[l-1]}) + \mathrm{Var}(F)$): use normalisation in the branch, zero-γ/Fixup, or $1/\sqrt N$ branch scaling. Our 34-layer residual MLP diverged with γ = 1 and reached 99.6% training accuracy with zero-γ.`,
        String.raw`Basic block $18C^2$ weights; bottleneck $17C^2$ on a $4C$-wide stream (vs $288C^2$ plain). Pre-activation ordering keeps a pure identity path.`,
        "Relatives: highway (gated, shrinking), DenseNet (concatenation, feature reuse), U-Net (concatenation across scales), Transformer residual stream (pre-LN vs post-LN).",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Batch, Layer and Group Normalization](/learn/m07/normalization) — the other half of every ResNet and Transformer block, and why post- and pre-LN behave differently.",
        "[Classic CNN Architectures](/learn/m08/cnn-architectures) — ResNet, DenseNet, ResNeXt and ConvNeXt as complete architectures.",
        "[Transformer Architecture](/learn/m11/transformer-architecture) — residual streams around attention and MLP sub-layers.",
        "[Segmentation](/learn/m12/segmentation) — U-Net's concatenation skips in action.",
        "[Regularization](/learn/m07/regularization) — once deep networks train, the next problem is generalisation (stochastic depth is a residual-specific regulariser).",
      ] },
    ],
  },
};

export default lesson;
