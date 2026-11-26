import type { Lesson } from "@/lib/types";

const BN_NUMPY = String.raw`import numpy as np
np.set_printoptions(suppress=True)


class BatchNorm1D:
    """Batch normalization for inputs of shape (N, D): statistics per feature, over the batch."""

    def __init__(self, D, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(D)            # learnable scale, starts as identity
        self.beta = np.zeros(D)            # learnable shift
        self.momentum, self.eps = momentum, eps
        self.running_mean = np.zeros(D)    # buffers used at inference time
        self.running_var = np.ones(D)
        self.training = True

    def forward(self, x):
        if self.training:
            N = x.shape[0]
            mu = x.mean(axis=0)                    # (D,) batch mean per feature
            var = x.var(axis=0)                    # (D,) biased batch variance (divide by N)
            # exponential moving averages (PyTorch convention: new = (1-m)*old + m*batch,
            # with the UNBIASED variance N/(N-1) * var stored in the running estimate)
            if N == 1:  # unbiased variance undefined (PyTorch raises an error here too)
                raise ValueError("batch norm needs more than 1 value per feature in training mode")
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mu
            self.running_var = (1 - m) * self.running_var + m * var * N / (N - 1)
        else:
            mu, var = self.running_mean, self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mu) * inv_std
        self.cache = (x_hat, inv_std)              # what the backward pass needs
        return self.gamma * x_hat + self.beta

    def backward(self, dy):
        """dy = dL/dy (N, D). Returns dL/dx and stores dL/dgamma, dL/dbeta (training mode)."""
        x_hat, inv_std = self.cache
        N = dy.shape[0]
        self.dgamma = (dy * x_hat).sum(axis=0)
        self.dbeta = dy.sum(axis=0)
        dx_hat = dy * self.gamma
        # the three-term formula: direct path - mean path - variance path
        return inv_std / N * (N * dx_hat - dx_hat.sum(axis=0) - x_hat * (dx_hat * x_hat).sum(axis=0))


def layer_norm(x, gamma, beta, eps=1e-5):
    """Statistics per EXAMPLE, over its features (axis=-1)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def rms_norm(x, gamma, eps=1e-6):
    """No mean subtraction, no shift: divide by the root-mean-square of the features."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return gamma * x / rms


rng = np.random.default_rng(0)
# ---------------- 1. badly scaled features in, standardized features out ----------------
D = 3
shift = np.array([5.0, -40.0, 0.0])        # deliberately different offsets ...
scale = np.array([0.1, 20.0, 3.0])         # ... and spreads per feature
x = shift + scale * rng.normal(size=(64, D))
bn = BatchNorm1D(D)
y = bn.forward(x)
print("input  mean", x.mean(0).round(3), " std", x.std(0).round(3))
print("output mean", y.mean(0).round(3) + 0.0, " std", y.std(0).round(3))

# ---------------- 2. gradient check of the three-term backward pass ----------------
bn = BatchNorm1D(D)
bn.gamma = rng.normal(size=D); bn.beta = rng.normal(size=D)
xs = rng.normal(size=(5, D)) * 2 + 1
W_out = rng.normal(size=(5, D))             # loss L = sum(W_out * y): dL/dy = W_out


def loss(xv):
    b = BatchNorm1D(D); b.gamma, b.beta = bn.gamma, bn.beta
    return (W_out * b.forward(xv)).sum()


bn.forward(xs)
dx = bn.backward(W_out)
num = np.zeros_like(xs); h = 1e-6
for idx in np.ndindex(xs.shape):
    xp, xm = xs.copy(), xs.copy()
    xp[idx] += h; xm[idx] -= h
    num[idx] = (loss(xp) - loss(xm)) / (2 * h)
rel = np.abs(dx - num).max() / np.abs(num).max()
print(f"dL/dx  analytic vs numerical, max relative error: {rel:.1e}")
print("column sums of dL/dx (should be 0):", dx.sum(0).round(12) + 0.0)
naive = W_out * bn.gamma * bn.cache[1]      # pretend mu and var were constants
print(f"error if mu/var are treated as constants: {np.abs(naive - num).max() / np.abs(num).max():.2f}")

# ---------------- 3. running statistics converge to the data statistics ----------------
bn = BatchNorm1D(D)
for step in range(200):
    bn.forward(shift + scale * rng.normal(size=(32, D)))
print("running_mean", bn.running_mean.round(2), " (true", shift, ")")
print("running_var ", bn.running_var.round(2), " (true", scale ** 2, ")")

# ---------------- 4. train vs eval on a single example ----------------
x1 = shift + scale * rng.normal(size=(1, D))
try:
    bn.forward(x1)
except ValueError as e:
    print("train mode, batch of 1 -> ValueError:", e)
# even ignoring that, the mean of a 1-example batch is the example itself, so x - mu = 0:
print("batch-statistics output for 1 example:", (bn.gamma * (x1 - x1.mean(0)) / np.sqrt(x1.var(0) + bn.eps) + bn.beta)[0])
bn.training = False
print("eval  mode, batch of 1:", bn.forward(x1).round(4)[0])

# ---------------- 5. layer norm and RMSNorm normalise each ROW instead ----------------
z = np.array([[1.0, 2.0, 3.0, 6.0], [10.0, 0.0, -10.0, 4.0]])
ln = layer_norm(z, np.ones(4), np.zeros(4))
print("LayerNorm rows: mean", ln.mean(1).round(6) + 0.0, " std", ln.std(1).round(4))
rn = rms_norm(z, np.ones(4))
print("RMSNorm rows: mean", rn.mean(1).round(4), " rms", np.sqrt((rn ** 2).mean(1)).round(4))`;

const BN_NUMPY_OUT = String.raw`input  mean [  4.982 -39.214   0.53 ]  std [ 0.106 17.231  2.765]
output mean [0. 0. 0.]  std [1. 1. 1.]
dL/dx  analytic vs numerical, max relative error: 9.7e-09
column sums of dL/dx (should be 0): [0. 0. 0.]
error if mu/var are treated as constants: 8.77
running_mean [  5.   -40.69   0.14]  (true [  5. -40.   0.] )
running_var  [  0.01 359.07   9.13]  (true [  0.01 400.     9.  ] )
train mode, batch of 1 -> ValueError: batch norm needs more than 1 value per feature in training mode
batch-statistics output for 1 example: [0. 0. 0.]
eval  mode, batch of 1: [ 0.6119 -0.4813  1.6881]
LayerNorm rows: mean [0. 0.]  std [1. 1.]
RMSNorm rows: mean [0.8485 0.1361]  rms [1. 1.]`;

const BN_TORCH_CONV = String.raw`import torch
import torch.nn as nn

torch.set_printoptions(precision=4, sci_mode=False)

# ---- 1. the worked example from the Math section, through nn.BatchNorm1d ----
x = torch.tensor([[2.0, 10.0], [4.0, 30.0], [6.0, 20.0], [8.0, 60.0]])
bn = nn.BatchNorm1d(2, eps=1e-5, momentum=0.1)
with torch.no_grad():
    bn.weight.copy_(torch.tensor([1.5, 0.5]))      # gamma
    bn.bias.copy_(torch.tensor([0.2, -1.0]))       # beta
bn.train()
y = bn(x)
print("y =\n", y.detach())
print("running_mean:", bn.running_mean, " running_var:", bn.running_var)
# reproduce the buffers by hand: PyTorch's 'momentum' is the weight of the NEW batch
m, N = 0.1, x.shape[0]
print("by hand     :", (1 - m) * 0 + m * x.mean(0), "", (1 - m) * 1 + m * x.var(0, unbiased=True))
print("biased var used for normalising:", x.var(0, unbiased=False), " unbiased var stored:", x.var(0, unbiased=True))
print("num_batches_tracked:", bn.num_batches_tracked.item())

# ---- 2. eval mode uses the running buffers, not the batch ----
bn.eval()
x_new = torch.tensor([[5.0, 30.0]])
print("eval output for [5, 30]:", bn(x_new).detach())

# ---- 3. momentum=None -> cumulative (equal-weight) average over all batches seen ----
torch.manual_seed(0)
cum = nn.BatchNorm1d(1, momentum=None)
batches = [torch.randn(16, 1) * 2 + 3 for _ in range(50)]
for b in batches:
    cum(b)
print("cumulative running_mean %.4f  vs mean of batch means %.4f" % (cum.running_mean.item(), torch.stack([b.mean() for b in batches]).mean().item()))

# ---- 4. batch of one in training mode ----
bn.train()
try:
    bn(torch.tensor([[5.0, 30.0]]))
except ValueError as e:
    print("ValueError:", e)

# ---- 5. BatchNorm2d: one mean/var per CHANNEL, pooled over N, H, W ----
torch.manual_seed(0)
xc = torch.randn(8, 3, 5, 5) * torch.tensor([1.0, 4.0, 0.5]).view(1, 3, 1, 1) + torch.tensor([0.0, 10.0, -2.0]).view(1, 3, 1, 1)
bn2 = nn.BatchNorm2d(3)
out = bn2(xc)
mu = xc.mean(dim=(0, 2, 3), keepdim=True)
var = xc.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
manual = (xc - mu) / torch.sqrt(var + bn2.eps)
print("per-channel means:", mu.flatten(), " values per statistic:", xc.numel() // 3)
print("max |BatchNorm2d - manual|: %.2e" % (out - manual).abs().max().item())
print("parameters: gamma", tuple(bn2.weight.shape), "beta", tuple(bn2.bias.shape))`;

const BN_TORCH_CONV_OUT = String.raw`y =
 tensor([[-1.8125, -1.5345],
        [-0.4708, -1.0000],
        [ 0.8708, -1.2673],
        [ 2.2125, -0.1982]])
running_mean: tensor([0.5000, 3.0000])  running_var: tensor([ 1.5667, 47.5667])
by hand     : tensor([0.5000, 3.0000])  tensor([ 1.5667, 47.5667])
biased var used for normalising: tensor([  5., 350.])  unbiased var stored: tensor([  6.6667, 466.6667])
num_batches_tracked: 1
eval output for [5, 30]: tensor([[5.5928, 0.9574]])
cumulative running_mean 3.1303  vs mean of batch means 3.1303
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 2])
per-channel means: tensor([ 0.0474, 10.1969, -1.9512])  values per statistic: 200
max |BatchNorm2d - manual|: 7.15e-07
parameters: gamma (3,) beta (3,)`;

const BN_TORCH_BWD = String.raw`import numpy as np
import torch
import torch.nn.functional as F


def bn_forward(x, gamma, beta, axes, eps=1e-5):
    """Training-mode batch norm; 'axes' are the axes the statistics are pooled over."""
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv_std
    return gamma * x_hat + beta, (x_hat, inv_std)


def bn_backward(dy, gamma, cache, axes):
    """Three-term gradient. M = number of values sharing one mean/variance."""
    x_hat, inv_std = cache
    M = np.prod([dy.shape[a] for a in axes])
    dgamma = (dy * x_hat).sum(axis=axes, keepdims=True)
    dbeta = dy.sum(axis=axes, keepdims=True)
    dx_hat = dy * gamma
    dx = inv_std / M * (M * dx_hat - dx_hat.sum(axis=axes, keepdims=True)
                        - x_hat * (dx_hat * x_hat).sum(axis=axes, keepdims=True))
    return dx, dgamma, dbeta


rng = np.random.default_rng(0)
for name, shape, axes, pshape in [("BatchNorm1d (N, D)", (16, 5), (0,), (1, 5)),
                                  ("BatchNorm2d (N, C, H, W)", (4, 3, 6, 6), (0, 2, 3), (1, 3, 1, 1))]:
    x = rng.normal(2.0, 3.0, size=shape)
    gamma = rng.normal(size=pshape); beta = rng.normal(size=pshape)
    dy = rng.normal(size=shape)                  # an arbitrary upstream gradient dL/dy

    y, cache = bn_forward(x, gamma, beta, axes)
    dx, dgamma, dbeta = bn_backward(dy, gamma, cache, axes)

    # the same computation with autograd (float64), through PyTorch's own batch_norm kernel
    xt = torch.tensor(x, requires_grad=True)
    gt = torch.tensor(gamma.ravel(), requires_grad=True)
    bt = torch.tensor(beta.ravel(), requires_grad=True)
    yt = F.batch_norm(xt, None, None, gt, bt, training=True, eps=1e-5)
    yt.backward(torch.tensor(dy))                # vector-Jacobian product with dy
    print(name)
    print("  max |y  - torch|      = %.1e" % np.abs(y - yt.detach().numpy()).max())
    print("  max |dx - torch|      = %.1e" % np.abs(dx - xt.grad.numpy()).max())
    print("  max |dgamma - torch|  = %.1e" % np.abs(dgamma.ravel() - gt.grad.numpy()).max())
    print("  max |dbeta  - torch|  = %.1e" % np.abs(dbeta.ravel() - bt.grad.numpy()).max())`;

const BN_TORCH_BWD_OUT = String.raw`BatchNorm1d (N, D)
  max |y  - torch|      = 8.9e-16
  max |dx - torch|      = 4.4e-16
  max |dgamma - torch|  = 2.7e-15
  max |dbeta  - torch|  = 1.8e-15
BatchNorm2d (N, C, H, W)
  max |y  - torch|      = 1.3e-15
  max |dx - torch|      = 4.4e-16
  max |dgamma - torch|  = 7.1e-15
  max |dbeta  - torch|  = 5.3e-15`;

const NORMS_TORCH = String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
N, C, H, W = 2, 6, 4, 4
x = torch.randn(N, C, H, W) * 3 + 1


def standardize(t, dims):
    mu = t.mean(dim=dims, keepdim=True)
    var = t.var(dim=dims, unbiased=False, keepdim=True)
    return (t - mu) / torch.sqrt(var + 1e-5)


# each layer below is freshly constructed, so gamma = 1 and beta = 0 (identity affine part)
checks = {
    "BatchNorm2d    over (N,H,W)  ": (nn.BatchNorm2d(C)(x), standardize(x, (0, 2, 3))),
    "LayerNorm      over (C,H,W)  ": (nn.LayerNorm([C, H, W])(x), standardize(x, (1, 2, 3))),
    "InstanceNorm2d over (H,W)    ": (nn.InstanceNorm2d(C)(x), standardize(x, (2, 3))),
    "GroupNorm(3)   over (C/3,H,W)": (nn.GroupNorm(3, C)(x), standardize(x.view(N, 3, C // 3, H, W), (2, 3, 4)).view(N, C, H, W)),
    "GroupNorm(1)  == LayerNorm   ": (nn.GroupNorm(1, C)(x), nn.LayerNorm([C, H, W])(x)),
    "GroupNorm(C)  == InstanceNorm": (nn.GroupNorm(C, C)(x), nn.InstanceNorm2d(C)(x)),
}
for name, (a, b) in checks.items():
    print(f"{name}: max diff {(a - b).abs().max().item():.1e}")

# Transformer-style input (batch, tokens, features): LayerNorm and RMSNorm act per token
h = torch.randn(2, 5, 8) * 4 + 2
ln, rms = nn.LayerNorm(8), nn.RMSNorm(8, eps=1e-6)
manual_rms = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
print("RMSNorm vs manual: max diff %.1e" % (rms(h) - manual_rms).abs().max().item())
print("LayerNorm params:", sum(p.numel() for p in ln.parameters()), " RMSNorm params:", sum(p.numel() for p in rms.parameters()))
# a token's LayerNorm output does not depend on the other tokens or the other sequences
h2 = h.clone(); h2[1] = torch.randn(5, 8) * 10      # change the second sequence completely
print("LayerNorm output of sequence 0 unchanged:", torch.allclose(ln(h)[0], ln(h2)[0]))
bn = nn.BatchNorm1d(8)
bn_out = lambda t: bn(t.reshape(-1, 8)).reshape(2, 5, 8)
print("BatchNorm output of sequence 0 unchanged:", torch.allclose(bn_out(h)[0], bn_out(h2)[0]))

# scale invariance: BN(a * Wx) == BN(Wx), and the gradient w.r.t. W shrinks by 1/a
torch.manual_seed(1)
inp = torch.randn(32, 10)
target = torch.randn(32, 4)
for a in [1.0, 10.0]:
    torch.manual_seed(2)                                   # the same random direction each time
    Wt = (a * torch.randn(4, 10)).requires_grad_(True)
    out = nn.functional.batch_norm(inp @ Wt.T, None, None, training=True)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    print(f"W scaled by {a:>4}: loss {loss.item():.5f}  ||dL/dW|| {Wt.grad.norm().item():.6f}")`;

const NORMS_TORCH_OUT = String.raw`BatchNorm2d    over (N,H,W)  : max diff 2.4e-07
LayerNorm      over (C,H,W)  : max diff 2.4e-07
InstanceNorm2d over (H,W)    : max diff 2.4e-07
GroupNorm(3)   over (C/3,H,W): max diff 2.4e-07
GroupNorm(1)  == LayerNorm   : max diff 2.4e-07
GroupNorm(C)  == InstanceNorm: max diff 2.4e-07
RMSNorm vs manual: max diff 2.4e-07
LayerNorm params: 16  RMSNorm params: 8
LayerNorm output of sequence 0 unchanged: True
BatchNorm output of sequence 0 unchanged: False
W scaled by  1.0: loss 1.99238  ||dL/dW|| 0.230191
W scaled by 10.0: loss 1.99238  ||dL/dW|| 0.023019`;

const DIGITS_LR = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)                 # 1797 images of 8x8 pixels, values 0..16
X = torch.tensor(X / 16.0, dtype=torch.float32)     # scale to [0, 1]
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


def make_mlp(bn, depth=8, width=64):
    layers, d = [], 64
    for _ in range(depth):
        # with BN the Linear bias is redundant (the mean subtraction removes it)
        layers += [nn.Linear(d, width, bias=not bn)] + ([nn.BatchNorm1d(width)] if bn else []) + [nn.ReLU()]
        d = width
    model = nn.Sequential(*layers, nn.Linear(d, 10))
    for mod in model:
        if isinstance(mod, nn.Linear):          # He initialisation for both variants (fair comparison)
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    return model


def train(bn, lr, batch_size=32, epochs=10, seed=0):
    torch.manual_seed(seed)
    model = make_mlp(bn)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - batch_size + 1, batch_size):   # drop the ragged last batch
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()                                  # inference: use the running statistics
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
        train_loss = loss_fn(model(Xtr), ytr).item()
    return acc, train_loss


print("8 hidden layers x 64 ReLUs, He init, SGD momentum 0.9, batch 32, 10 epochs")
print(f"{'lr':>6} | {'plain: test acc':>15} {'train loss':>10} | {'BN: test acc':>12} {'train loss':>10}")
for lr in [0.001, 0.01, 0.1, 0.5, 2.0]:
    a0, l0 = train(False, lr)
    a1, l1 = train(True, lr)
    print(f"{lr:>6} | {a0:>15.3f} {l0:>10.3f} | {a1:>12.3f} {l1:>10.3f}")`;

const DIGITS_LR_OUT = String.raw`8 hidden layers x 64 ReLUs, He init, SGD momentum 0.9, batch 32, 10 epochs
    lr | plain: test acc train loss | BN: test acc train loss
 0.001 |           0.894      0.315 |        0.894      0.348
  0.01 |           0.956      0.038 |        0.975      0.013
   0.1 |           0.356      1.540 |        0.967      0.028
   0.5 |           0.097      2.321 |        0.928      0.240
   2.0 |           0.100        nan |        0.100      2.688`;

const DIGITS_BS = String.raw`import copy
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


def make_mlp(bn, depth=8, width=64):
    layers, d = [], 64
    for _ in range(depth):
        layers += [nn.Linear(d, width, bias=not bn)] + ([nn.BatchNorm1d(width)] if bn else []) + [nn.ReLU()]
        d = width
    model = nn.Sequential(*layers, nn.Linear(d, 10))
    for mod in model:
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    return model


def train(bn, lr, batch_size, epochs, seed=0):
    torch.manual_seed(seed)
    model = make_mlp(bn)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - batch_size + 1, batch_size):   # drop the ragged last batch
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def accuracy(model, train_mode=False):
    m = copy.deepcopy(model)          # copy: train-mode forward passes would overwrite the buffers
    m.train(train_mode)
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()


print("lr 0.003, 5 epochs, same seed; test accuracy")
print(f"{'batch':>5} | {'plain':>6} | {'BN (eval)':>9} | {'BN, test-set batch stats':>24}")
for bs in [2, 4, 8, 32, 128]:
    plain = train(False, 0.003, bs, epochs=5)
    bnm = train(True, 0.003, bs, epochs=5)
    print(f"{bs:>5} | {accuracy(plain):>6.3f} | {accuracy(bnm):>9.3f} | {accuracy(bnm, train_mode=True):>24.3f}")`;

const DIGITS_BS_OUT = String.raw`lr 0.003, 5 epochs, same seed; test accuracy
batch |  plain | BN (eval) | BN, test-set batch stats
    2 |  0.925 |     0.164 |                    0.172
    4 |  0.950 |     0.508 |                    0.733
    8 |  0.939 |     0.933 |                    0.947
   32 |  0.881 |     0.931 |                    0.928
  128 |  0.439 |     0.644 |                    0.656`;

const EVAL_BUG = String.raw`import copy
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                      nn.Linear(64, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                      nn.Linear(64, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for epoch in range(10):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr) - 31, 32):
        idx = perm[i:i + 32]
        loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()


def test_acc(m, batches):
    with torch.no_grad():
        return sum((m(Xte[b]).argmax(1) == yte[b]).sum().item() for b in batches) / len(yte)


every = [torch.arange(len(yte))]                                  # one batch of 360
chunks = list(torch.arange(len(yte)).split(4))                    # batches of 4
by_class = list(torch.argsort(yte, stable=True).split(36))        # each batch = one digit

model.eval()
print("correct  : model.eval(), any batching        acc %.3f / %.3f" % (test_acc(model, every), test_acc(model, chunks)))

bug = copy.deepcopy(model).train()      # the bug: evaluation without model.eval()
print("forgot eval, whole test set as one batch     acc %.3f" % test_acc(bug, every))
print("forgot eval, batches of 4                    acc %.3f" % test_acc(bug, chunks))
print("forgot eval, batches sorted by class         acc %.3f" % test_acc(bug, by_class))
try:
    bug(Xte[:1])
except ValueError as e:
    print("forgot eval, single example  ->", type(e).__name__)

# side effect: every train-mode forward pass also overwrote the running statistics
drift = (bug[1].running_mean - model[1].running_mean).abs().max().item()
print("max change of layer-1 running_mean caused by the 'evaluation': %.3f" % drift)
bug.eval()
print("eval() afterwards, with the corrupted buffers  acc %.3f" % test_acc(bug, every))`;

const EVAL_BUG_OUT = String.raw`correct  : model.eval(), any batching        acc 0.975 / 0.975
forgot eval, whole test set as one batch     acc 0.969
forgot eval, batches of 4                    acc 0.775
forgot eval, batches sorted by class         acc 0.125
forgot eval, single example  -> ValueError
max change of layer-1 running_mean caused by the 'evaluation': 0.125
eval() afterwards, with the corrupted buffers  acc 0.969`;

const EX_LN = String.raw`import numpy as np


def ln_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)              # one mean per row (example)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv_std
    return gamma * x_hat + beta, (x_hat, inv_std)


def ln_backward(dy, gamma, cache):
    x_hat, inv_std = cache
    D = dy.shape[-1]                                 # values sharing one mean = features
    dgamma = (dy * x_hat).sum(axis=0)                # gamma is shared by all rows -> sum over rows
    dbeta = dy.sum(axis=0)
    dx_hat = dy * gamma
    dx = inv_std / D * (D * dx_hat - dx_hat.sum(-1, keepdims=True)
                        - x_hat * (dx_hat * x_hat).sum(-1, keepdims=True))
    return dx, dgamma, dbeta


rng = np.random.default_rng(0)
x = rng.normal(1.0, 2.0, size=(3, 6))
gamma, beta = rng.normal(size=6), rng.normal(size=6)
dy = rng.normal(size=(3, 6))                         # loss L = sum(dy * y)
y, cache = ln_forward(x, gamma, beta)
dx, dgamma, dbeta = ln_backward(dy, gamma, cache)

num_dx = np.zeros_like(x); num_dg = np.zeros_like(gamma); h = 1e-6
for idx in np.ndindex(x.shape):
    xp, xm = x.copy(), x.copy(); xp[idx] += h; xm[idx] -= h
    num_dx[idx] = ((dy * ln_forward(xp, gamma, beta)[0]).sum() - (dy * ln_forward(xm, gamma, beta)[0]).sum()) / (2 * h)
for j in range(6):
    gp, gm = gamma.copy(), gamma.copy(); gp[j] += h; gm[j] -= h
    num_dg[j] = ((dy * ln_forward(x, gp, beta)[0]).sum() - (dy * ln_forward(x, gm, beta)[0]).sum()) / (2 * h)
print("max |dx - numerical|     = %.1e" % np.abs(dx - num_dx).max())
print("max |dgamma - numerical| = %.1e" % np.abs(dgamma - num_dg).max())
print("row sums of dx:", dx.sum(axis=1).round(12) + 0.0)`;

const EX_LN_OUT = String.raw`max |dx - numerical|     = 1.5e-09
max |dgamma - numerical| = 4.5e-10
row sums of dx: [0. 0. 0.]`;

const BN_KERAS = String.raw`import numpy as np
import tensorflow as tf

x = np.array([[2.0, 10.0], [4.0, 30.0], [6.0, 20.0], [8.0, 60.0]], dtype="float32")

# Keras convention: momentum is the weight of the OLD value (0.99 by default), epsilon defaults to 1e-3.
# momentum=0.9 here equals PyTorch's momentum=0.1.
bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
y_train = bn(x, training=True)        # batch statistics, buffers updated
y_infer = bn(x, training=False)       # moving statistics, no update
print("train-mode output:\n", y_train.numpy().round(4))
print("moving_mean:", bn.moving_mean.numpy(), " moving_variance:", bn.moving_variance.numpy())

ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)   # per example, over features
print("layer norm rows:", ln(x).numpy().round(4))

# A small CNN with BN after each convolution (no conv bias: BN's beta replaces it)
model = tf.keras.Sequential([
    tf.keras.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, use_bias=False),
    tf.keras.layers.BatchNormalization(),          # statistics per channel over (N, H, W)
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10),
])
model.summary()   # fit() runs BN in training mode; predict()/evaluate() in inference mode`;

const lesson: Lesson = {
  id: "m07/normalization",
  title: "Batch, Layer and Group Normalization",
  summary: "Normalising activations inside a network: batch normalization worked out by hand (forward pass, running statistics and the full three-term backward pass), why it acts differently in training and inference, why it helps, how it fails, and the alternatives that don't depend on the batch — layer norm, RMSNorm, group and instance norm — used by Transformers, LLMs and small-batch vision models.",
  minutes: 90,
  level: "intermediate",
  prerequisites: ["m04/backpropagation", "m04/gradient-problems", "m07/initialization", "m03/training-basics"],
  next: ["m07/residual-connections", "m07/regularization", "m11/transformer-architecture"],
  terms: ["batch-normalization", "layer-normalization", "rms-norm", "group-normalization", "instance-normalization", "running-statistics", "internal-covariate-shift", "standardization"],
  sections: {
    intro: [
      { t: "p", text: "**Normalization layers** rescale the values flowing through a network so that some chosen group of them has mean 0 and variance 1 — and then let the network learn whatever scale and offset it really wants. They are everywhere: almost every convolutional network since 2015 has **batch normalization**, every Transformer has **layer normalization**, and many recent large language models use its simpler cousin **RMSNorm**." },
      { t: "p", text: "The idea starts from a familiar trick. Before training, we **standardise the input features**: subtract each feature's mean and divide by its standard deviation. Then a feature measured in thousands doesn't drown out one measured in fractions, and gradient descent sees an evenly shaped loss surface. **Batch normalization** (Ioffe & Szegedy, 2015, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*) does the same to the **inputs of every layer**, not just the first, computing mean and variance on the fly from the current mini-batch." },
      { t: "p", text: "In the original paper, adding BN to an Inception network reached the same accuracy in 14× fewer training steps, handled much larger learning rates and cared less about initialisation. But using **batch** statistics brings problems: different behaviour in training and inference, poor results with small batches, and outputs that depend on the other examples in the batch. Layer, group and instance normalization compute the same statistics over **different axes** of the activation tensor to avoid them." },
      { t: "callout", kind: "key", text: "Every normalization layer computes $y = \\gamma\\,(x-\\mu)/\\sqrt{\\sigma^2+\\epsilon} + \\beta$. They differ only in **which values share one $\\mu$ and $\\sigma^2$**: the batch (BatchNorm), the features of one example (LayerNorm), a group of channels of one example (GroupNorm) or one channel of one example (InstanceNorm)." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Backpropagation](/learn/m04/backpropagation) — the chain rule through a graph, $\\delta = \\partial\\mathcal{L}/\\partial z$, and the rule that contributions along several paths **add**. The BN backward pass is a careful use of exactly that.",
        "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — why the size of activations and gradients drifts with depth. Normalization is one of the main fixes listed there; this lesson explains how it works.",
        "[Weight Initialization](/learn/m07/initialization) — variance-preserving initialisation (Xavier/He). Normalization makes training far less sensitive to it.",
        "[Training basics](/learn/m03/training-basics) — mini-batches, epochs, train/validation/test splits and the difference between training and inference.",
        "Mean, variance and standard deviation, and the biased ($\\frac1N$) vs unbiased ($\\frac1{N-1}$) variance — see [Probability](/learn/m02/probability).",
        "For the convolutional parts: tensors of shape $(N, C, H, W)$ — batch, channels, height, width — from [The Convolution Operation](/learn/m08/convolution).",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Grading on a curve", text: "A teacher marks several classes with different exams: one out of 10, one out of 200, and one marker was harsh. To compare fairly, each class's marks become \"how many standard deviations above the class average\" — a z-score. That is batch normalization, with the **class** as the mini-batch. Two things follow. A student's curved grade depends on who else sat the exam (batch dependence). And a student who sits alone can't be curved at all — their z-score is 0 (the batch-of-one problem). The school fixes both by curving against **historical** averages at year's end (running statistics at inference). LayerNorm is a different policy: each student's marks across their **own subjects** are standardised, which needs no classmates at all." },
      { t: "p", text: "Why do this inside every layer? Each layer's input is the output of all the layers before it. As those layers keep changing during training, the mean and spread of what a layer receives drift. Worse, the **size** of activations can grow or shrink geometrically with depth. A layer with huge inputs saturates its activation or takes far-too-big gradient steps; a layer with tiny inputs barely learns." },
      { t: "p", text: "Normalization takes mean and scale out of the picture: whatever earlier layers do, the next layer sees values centred at 0 with spread 1. The learnable scale $\\gamma$ and shift $\\beta$ then put back exactly as much scale and offset as is useful — now as **two explicit learnable numbers per feature**, instead of an accident of every weight in every earlier layer." },
      { t: "p", text: "One more idea explains later sections: since normalization divides by the spread, multiplying the weights before it by 10 changes **nothing** in the output. The layer stops caring about the scale of its incoming weights, which is a big part of why it tolerates large learning rates." },
    ],
    mechanism: [
      { t: "h", text: "Batch normalization in training mode" },
      { t: "p", text: "Write one layer's pre-activations for a mini-batch as a matrix $X\\in\\mathbb{R}^{N\\times D}$: $N$ examples (rows), $D$ features (columns). BN handles each **column** on its own:" },
      { t: "steps", items: [
        { title: "Batch mean per feature", text: "For each feature $j$, average its $N$ values over the batch.", tex: String.raw`\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}` },
        { title: "Batch variance per feature", text: "The biased (divide by $N$) variance of the same column.", tex: String.raw`\sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N} (x_{ij}-\mu_j)^2` },
        { title: "Standardise", text: "Subtract the mean, divide by the standard deviation. The small $\\epsilon$ (PyTorch default $10^{-5}$, Keras $10^{-3}$) avoids dividing by zero when a feature is constant over the batch.", tex: String.raw`\hat x_{ij} = \frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}` },
        { title: "Scale and shift", text: "Multiply by a learnable $\\gamma_j$ and add a learnable $\\beta_j$ (starting at 1 and 0), trained by backprop like any parameter.", tex: String.raw`y_{ij} = \gamma_j\,\hat x_{ij} + \beta_j` },
        { title: "Update the running statistics", text: "Keep running averages of the batch mean and variance in non-trainable **buffers**, for use at inference. In PyTorch the momentum $m$ (default 0.1) is the weight of the **new** batch, and the stored variance is the **unbiased** one.", tex: String.raw`\hat\mu_j \leftarrow (1-m)\,\hat\mu_j + m\,\mu_j, \qquad \hat\sigma^2_j \leftarrow (1-m)\,\hat\sigma^2_j + m\,\tfrac{N}{N-1}\sigma_j^2` },
      ] },
      { t: "h", text: "Why γ and β are needed" },
      { t: "p", text: "Forcing every feature to mean 0 and variance 1 would **limit** what the layer can represent. A sigmoid fed only values near 0 works in its nearly linear zone; a ReLU fed values centred at 0 is always on for about half the batch. With $\\gamma_j = \\sqrt{\\sigma_j^2+\\epsilon}$ and $\\beta_j = \\mu_j$ the layer computes the **identity** — so adding BN never shrinks the set of functions the network can express. Normalization changes how the network is parameterised; it doesn't constrain it." },
      { t: "p", text: "A practical detail: the bias $b$ of the linear or conv layer right before a BN is **useless**, because subtracting $\\mu_j$ removes any constant added to the whole column. So BN is used with `nn.Linear(..., bias=False)` / `nn.Conv2d(..., bias=False)`; $\\beta$ takes over the bias's job." },
      { t: "h", text: "Batch normalization at inference" },
      { t: "p", text: "At test time we may predict one example, and a prediction shouldn't depend on which other examples happen to come with it. So in **eval mode** BN uses the stored running statistics instead of batch statistics:" },
      { t: "math", tex: String.raw`y_j = \gamma_j\,\frac{x_j - \hat\mu_j}{\sqrt{\hat\sigma_j^2+\epsilon}} + \beta_j \;=\; \underbrace{\frac{\gamma_j}{\sqrt{\hat\sigma_j^2+\epsilon}}}_{a_j}\,x_j + \underbrace{\Big(\beta_j - \frac{\gamma_j\hat\mu_j}{\sqrt{\hat\sigma_j^2+\epsilon}}\Big)}_{c_j}`, caption: "At inference BN is a fixed per-feature map $y = a\\,x + c$, so it can be merged into the previous linear or conv layer's weights and bias for deployment." },
      { t: "p", text: "In PyTorch the switch is `model.train()` / `model.eval()`; in Keras it is the `training` argument (set automatically by `fit` and `predict`). Forgetting it is one of the most common bugs in deep learning — shown with real numbers in the Framework section." },
      { t: "h", text: "Batch normalization for convolutional layers" },
      { t: "p", text: "A conv feature map has shape $(N, C, H, W)$. A convolution applies **the same filter** at every position, so all positions of one channel count as samples of the same feature. `BatchNorm2d` therefore computes **one mean and one variance per channel**, over the batch **and** all positions — $N\\cdot H\\cdot W$ values each — and learns one $\\gamma$ and one $\\beta$ per channel ($2C$ parameters). This matches how convolution shares weights across positions, and gives far more values per statistic than the batch alone." },
      { t: "h", text: "The other normalizations: same formula, different axes" },
      { t: "table", head: ["Method", "Mean/variance computed over", "Number of statistics for an (N, C, H, W) tensor", "Depends on other examples?", "γ, β shape"], rows: [
        ["Batch norm", "(N, H, W) for each channel", "C", "Yes (in training)", "per channel"],
        ["Layer norm", "(C, H, W) for each example — or, in a Transformer, the D features of each token", "N (one per token for sequences)", "No", "per feature (elementwise)"],
        ["Instance norm", "(H, W) for each example and channel", "N · C", "No", "per channel (optional)"],
        ["Group norm (G groups)", "(C/G, H, W) for each example and channel group", "N · G", "No", "per channel"],
        ["RMSNorm", "Like layer norm, but only the root mean square; no mean subtraction", "N (per token)", "No", "γ only (no β)"],
      ] },
      { t: "p", text: "Group norm sits between the extremes: with $G = 1$ it **is** layer norm over $(C, H, W)$, with $G = C$ it **is** instance norm — both checked numerically in the PyTorch code below." },
    ],
    math: [
      { t: "h", text: "1. Definitions" },
      { t: "p", text: "For one feature (column) of a training mini-batch $\\{x_1,\\dots,x_N\\}$ (dropping the index $j$): $\\mu$ is the batch mean, $\\sigma^2$ the biased batch variance, $\\epsilon > 0$ a small constant, $s = \\sqrt{\\sigma^2+\\epsilon}$ the padded standard deviation, $\\hat x_i$ the standardised value, and $\\gamma, \\beta$ the learnable scale and shift. $\\mathcal{L}$ is the loss, and $g_i = \\partial\\mathcal{L}/\\partial y_i$ is the gradient arriving from the layers above." },
      { t: "math", tex: String.raw`\mu = \frac1N\sum_{i=1}^N x_i,\qquad \sigma^2 = \frac1N\sum_{i=1}^N (x_i-\mu)^2,\qquad s = \sqrt{\sigma^2+\epsilon},\qquad \hat x_i = \frac{x_i-\mu}{s},\qquad y_i = \gamma\hat x_i + \beta` },
      { t: "p", text: "Two facts follow at once: $\\sum_i \\hat x_i = 0$ and $\\frac1N\\sum_i \\hat x_i^2 = \\sigma^2/(\\sigma^2+\\epsilon) \\approx 1$." },
      { t: "h", text: "2. Worked example: forward pass on a batch of 4" },
      { t: "p", text: "A batch of $N = 4$ examples with $D = 2$ features, $\\gamma = (1.5, 0.5)$, $\\beta = (0.2, -1)$, $\\epsilon = 10^{-5}$. Notice the features live on very different scales." },
      { t: "math", tex: String.raw`X = \begin{pmatrix} 2 & 10\\ 4 & 30\\ 6 & 20\\ 8 & 60 \end{pmatrix}` },
      { t: "steps", items: [
        { title: "Means (per column)", tex: String.raw`\mu_1 = \frac{2+4+6+8}{4} = 5,\qquad \mu_2 = \frac{10+30+20+60}{4} = 30` },
        { title: "Biased variances", tex: String.raw`\sigma_1^2 = \frac{9+1+1+9}{4} = 5,\qquad \sigma_2^2 = \frac{400+0+100+900}{4} = 350` },
        { title: "Padded standard deviations", tex: String.raw`s_1 = \sqrt{5.00001} = 2.236070,\qquad s_2 = \sqrt{350.00001} = 18.708287` },
        { title: "Standardise", tex: String.raw`\hat X = \begin{pmatrix} -3/2.236070 & -20/18.708287\\ -1/2.236070 & 0\\ 1/2.236070 & -10/18.708287\\ 3/2.236070 & 30/18.708287 \end{pmatrix} = \begin{pmatrix} -1.341639 & -1.069045\\ -0.447213 & 0\\ 0.447213 & -0.534522\\ 1.341639 & 1.603567 \end{pmatrix}` },
        { title: "Scale and shift", text: "Column 1: $1.5\\hat x + 0.2$; column 2: $0.5\\hat x - 1$.", tex: String.raw`Y = \begin{pmatrix} -1.812459 & -1.534522\\ -0.470820 & -1.000000\\ 0.870820 & -1.267261\\ 2.212459 & -0.198216 \end{pmatrix}` },
        { title: "Running statistics (PyTorch, m = 0.1, buffers start at mean 0, var 1)", text: "The stored variance is unbiased: $\\tfrac{4}{3}\\sigma^2 = (6.666667,\\ 466.666667)$.", tex: String.raw`\hat\mu = 0.9\cdot(0,0) + 0.1\cdot(5, 30) = (0.5,\ 3.0),\qquad \hat\sigma^2 = 0.9\cdot(1,1) + 0.1\cdot(6.666667,\ 466.666667) = (1.566667,\ 47.566667)` },
        { title: "Inference on x = (5, 30)", text: "With batch statistics this input sits exactly at the mean, so the output would be $\\beta = (0.2, -1)$. With running statistics after only **one** batch it is far off — running statistics need many updates to settle, which is why a model trained for few steps behaves badly in eval mode.", tex: String.raw`y = \Big(1.5\,\tfrac{5-0.5}{\sqrt{1.56668}} + 0.2,\ \ 0.5\,\tfrac{30-3}{\sqrt{47.56668}} - 1\Big) = (5.592797,\ 0.957413)` },
      ] },
      { t: "p", text: "`nn.BatchNorm1d` reproduces every number above in the first PyTorch program of the Framework section, including the running buffers $(0.5, 3.0)$ and $(1.5667, 47.5667)$." },
      { t: "h", text: "3. The backward pass: deriving the three-term gradient" },
      { t: "p", text: "The parameter gradients are easy: $y_i = \\gamma\\hat x_i + \\beta$, and $\\gamma, \\beta$ are shared by every example, so their gradients **sum** over the batch:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial\gamma} = \sum_{i=1}^N g_i\,\hat x_i, \qquad \frac{\partial\mathcal{L}}{\partial\beta} = \sum_{i=1}^N g_i, \qquad \hat g_i \equiv \frac{\partial\mathcal{L}}{\partial\hat x_i} = \gamma\, g_i` },
      { t: "p", text: "The input gradient is where BN differs from an ordinary per-element layer. Each $x_k$ affects the loss along **three paths**: directly through its own $\\hat x_k$, through the mean $\\mu$ (used by every $\\hat x_i$), and through the variance $\\sigma^2$ (also used by every $\\hat x_i$). By the chain rule the contributions add:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial x_k} = \underbrace{\hat g_k\,\frac{1}{s}}_{\text{direct}} \;+\; \underbrace{\frac{\partial\mathcal{L}}{\partial\mu}\,\frac{1}{N}}_{\text{via }\mu} \;+\; \underbrace{\frac{\partial\mathcal{L}}{\partial\sigma^2}\,\frac{2(x_k-\mu)}{N}}_{\text{via }\sigma^2}` },
      { t: "p", text: "**Through the variance.** $\\hat x_i = (x_i-\\mu)(\\sigma^2+\\epsilon)^{-1/2}$, so $\\partial\\hat x_i/\\partial\\sigma^2 = -\\tfrac12(x_i-\\mu)(\\sigma^2+\\epsilon)^{-3/2} = -\\hat x_i/(2s^2)$:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial\sigma^2} = \sum_{i}\hat g_i\,\frac{\partial\hat x_i}{\partial\sigma^2} = -\frac{1}{2s^2}\sum_i \hat g_i\,\hat x_i` },
      { t: "p", text: "**Through the mean.** $\\mu$ enters each $\\hat x_i$ directly ($\\partial\\hat x_i/\\partial\\mu = -1/s$) and also enters $\\sigma^2$, but $\\partial\\sigma^2/\\partial\\mu = -\\tfrac2N\\sum_i(x_i-\\mu) = 0$, so only the direct part is left:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial\mu} = -\frac1s\sum_i \hat g_i` },
      { t: "p", text: "**Put it together**, writing $x_k - \\mu = s\\,\\hat x_k$:" },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial x_k} = \frac{\hat g_k}{s} - \frac{1}{Ns}\sum_i\hat g_i - \frac{\hat x_k}{Ns}\sum_i\hat g_i\hat x_i \;=\; \frac{1}{N s}\Big(N\hat g_k - \sum_{i=1}^N \hat g_i - \hat x_k\sum_{i=1}^N \hat g_i\,\hat x_i\Big)`, caption: "The three-term BN gradient. Over a whole batch it costs only two sums per feature — this is the formula in the NumPy code." },
      { t: "p", text: "**What it means.** Treat $\\hat g$ as a vector over the batch. The bracket is $N$ times $\\hat g$ minus its projection onto the all-ones vector $\\mathbf 1$ and onto $\\hat x$ (since $\\|\\mathbf 1\\|^2 = N$ and $\\|\\hat x\\|^2 \\approx N$). In plain words: shifting all inputs by a constant, or stretching them about their mean, doesn't change BN's output — so the gradient has no part in those directions. Consequences: $\\sum_k \\partial\\mathcal{L}/\\partial x_k = 0$ exactly, and $\\sum_k \\hat x_k\\,\\partial\\mathcal{L}/\\partial x_k \\approx 0$ (exactly 0 when $\\epsilon = 0$)." },
      { t: "h", text: "4. Worked example: backward pass" },
      { t: "p", text: "Continue with feature 1 above ($\\gamma_1 = 1.5$, $s_1 = 2.236070$, $\\hat x = (-1.341639, -0.447213, 0.447213, 1.341639)$) and say the upstream gradient is $g = (0.1, -0.2, 0.3, 0.4)$." },
      { t: "steps", items: [
        { title: "Parameter gradients", tex: String.raw`\frac{\partial\mathcal{L}}{\partial\gamma_1} = \sum_i g_i\hat x_i = 0.626098,\qquad \frac{\partial\mathcal{L}}{\partial\beta_1} = \sum_i g_i = 0.6` },
        { title: "Gradient w.r.t. x̂", tex: String.raw`\hat g = \gamma_1 g = (0.15,\ -0.3,\ 0.45,\ 0.6),\qquad \sum_i\hat g_i = 0.9,\qquad \sum_i\hat g_i\hat x_i = 0.939148` },
        { title: "First example", tex: String.raw`\frac{\partial\mathcal{L}}{\partial x_1} = \frac{4(0.15) - 0.9 - (-1.341639)(0.939148)}{4 \times 2.236070} = \frac{0.6 - 0.9 + 1.260000}{8.944281} = 0.107331` },
        { title: "All four", tex: String.raw`\frac{\partial\mathcal{L}}{\partial x} = (0.107331,\ -0.187830,\ 0.053666,\ 0.026833),\qquad \textstyle\sum_k \partial\mathcal{L}/\partial x_k = 0` },
        { title: "Compare: treating μ and σ² as constants (wrong)", tex: String.raw`\hat g / s_1 = (0.067082,\ -0.134164,\ 0.201246,\ 0.268328)` },
      ] },
      { t: "p", text: "The naive gradient isn't even close — the last example's is 10× too large — and a central finite difference confirms the three-term values to six decimals. Ignoring how examples are linked through the batch is the classic mistake when hand-writing BN; the gradient check in the NumPy code catches it (relative error 8.77 instead of $10^{-8}$)." },
      { t: "h", text: "5. Scale invariance and what it causes" },
      { t: "p", text: "Let BN follow a linear layer, $x = Wu$. Scaling the weights by $a > 0$ scales $x$, $\\mu$ and $\\sigma$ by $a$ (ignoring $\\epsilon$), so $\\hat x$ — and the loss — stays the same: $\\mathrm{BN}(aWu) = \\mathrm{BN}(Wu)$. Differentiating $\\mathcal{L}(aW) = \\mathcal{L}(W)$ with respect to $W$ gives:" },
      { t: "math", tex: String.raw`\nabla\mathcal{L}\big|_{aW} = \frac{1}{a}\,\nabla\mathcal{L}\big|_{W}, \qquad W^\top\nabla_W\mathcal{L} = 0` },
      { t: "p", text: "Large weights get proportionally **smaller** gradients, so the effective step $\\eta\\|\\nabla\\mathcal{L}\\|/\\|W\\|$ shrinks on its own if the weights grow — a built-in brake that helps explain why BN networks handle higher learning rates. The PyTorch code checks it: scaling $W$ by 10 leaves the loss at 1.99238 and cuts the gradient norm from 0.230191 to 0.023019. (The second identity says the gradient is at right angles to $W$; it is also why weight decay on such layers mostly changes the effective learning rate rather than the function.)" },
      { t: "h", text: "6. Why does batch normalization help? Two explanations" },
      { t: "p", text: "**Internal covariate shift (the original reason).** Ioffe & Szegedy defined *internal covariate shift* as the change in a layer's input distribution caused by updates to earlier layers. They argued it slows training, because each layer must keep chasing a moving target, and proposed BN to reduce it." },
      { t: "p", text: "**Later evidence.** Santurkar, Tsipras, Ilyas & Madry (2018, *How Does Batch Normalization Help Optimization?*) tested this. They reported that (i) injecting random, changing noise after BN layers — making input distributions *less* stable — left BN's training speed mostly intact, and (ii) by their measures BN doesn't necessarily reduce internal covariate shift. Instead they found BN makes the loss surface **smoother**: the loss and its gradients change more slowly along the training path, so the gradient stays a reliable guide over a longer step, which allows larger learning rates. This is not settled; scale invariance (above) and batch noise are also discussed as factors." },
      { t: "p", text: "The digits experiment below shows the learning-rate effect directly: an 8-layer MLP **without** BN diverges or stalls at learning rate 0.1 (test accuracy 0.356) while the same network **with** BN reaches 0.967; at 0.5 the plain network is at chance (0.097) and the BN network still gets 0.928." },
      { t: "h", text: "7. Layer normalization and RMSNorm" },
      { t: "p", text: "Layer normalization (Ba, Kiros & Hinton, 2016) computes the statistics over the $D$ features of **one** example (one token, in a Transformer), with per-feature $\\gamma, \\beta$:" },
      { t: "math", tex: String.raw`\mu = \frac1D\sum_{j=1}^D x_j,\qquad \sigma^2 = \frac1D\sum_{j=1}^D (x_j-\mu)^2,\qquad y_j = \gamma_j\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_j` },
      { t: "p", text: "Its backward pass is the same three-term formula with sums over features instead of the batch (practice exercise 2). RMSNorm (Zhang & Sennrich, 2019) drops the mean subtraction and the shift, and divides by the root mean square:" },
      { t: "math", tex: String.raw`\mathrm{RMS}(x) = \sqrt{\frac1D\sum_{j=1}^D x_j^2 + \epsilon},\qquad y_j = \gamma_j\,\frac{x_j}{\mathrm{RMS}(x)}` },
      { t: "p", text: "It keeps the rescaling invariance (the part that seems to matter most), is cheaper — one sum instead of two — and has half the parameters. The LLaMA family of language models, for example, uses it." },
    ],
    visual: [
      { t: "p", text: "The simulator draws a real mini-batch of four features whose shift and spread you control, and walks through every stage with the actual numbers. Top: one strip per statistics group (a feature for BatchNorm, an example for LayerNorm), with the mean as a line and ±1 standard deviation as a band. Bottom: the batch as a matrix, with a gold outline around the cells that share one mean and variance." },
      { t: "sim", id: "batchnorm", caption: "Stages: raw values → mean → variance → standardise → γ, β → running-statistics update. Toggle BatchNorm/LayerNorm and Train/Eval, and resample the other examples." },
      { t: "list", items: [
        "**Badly scaled features.** With the default preset, step through stages 1–4. Raw, $x_1$ is a cluster near 5 and $x_2$ spreads over dozens of units; after standardising, all four strips sit at 0 with unit bands. Read the mean and variance arithmetic in the narration.",
        "**γ and β can undo everything.** At stage 5 press \"γ = σ, β = μ (undo)\": the outputs land exactly on the raw values again — the identity argument from the Mechanism section.",
        "**Running statistics.** At stage 6 read the running-average update: the stored variance uses $N/(N-1)$. Change the momentum and the number of past batches and watch how far the buffers are from the current batch's statistics.",
        "**Batch of one.** Set $N = 1$ in Train mode: every $\\hat x$ is 0 and the output is just $\\beta$. Switch to Eval: the running statistics give a sensible output for the same example.",
        "**N = 2.** In Train mode with $N = 2$ every BatchNorm column becomes ±1 (up to $\\epsilon$) — the layer only keeps *which* of the two examples is larger. This is why tiny batches train badly (see the batch-size experiment).",
        "**Batch dependence.** In Train mode press \"Resample other examples\": example #1 is unchanged but its output changes. Repeat in Eval mode and in LayerNorm mode: the change is exactly 0.",
        "**LayerNorm axis.** Switch to LayerNorm: the gold outline turns from a column into a row, the strips become one per example, and stage 6 explains why there are no running statistics.",
      ] },
      { t: "p", text: "The gradient-flow simulator from [Vanishing and Exploding Gradients](/learn/m04/gradient-problems) has a LayerNorm switch. Pick the preset \"ReLU, σ = 0.5 (exploding)\": activations grow by orders of magnitude with depth. Tick *layer normalisation*: the activation size is pinned at every layer, and the per-layer gradient norms flatten." },
      { t: "sim", id: "gradient-flow", caption: "Compare a 30-layer ReLU network with too-large initial weights with and without LayerNorm before each activation." },
    ],
    scratch: [
      { t: "p", text: "A full batch normalization layer in NumPy — training and eval forward passes, PyTorch-compatible running statistics, and the three-term backward pass — plus layer norm and RMSNorm forward passes. The script checks the backward pass against central finite differences and shows what happens if you ignore how $\\mu$ and $\\sigma^2$ depend on the inputs." },
      { t: "code", lang: "python", title: "batchnorm_from_scratch.py", code: BN_NUMPY, output: BN_NUMPY_OUT },
      { t: "p", text: "Reading the output:" },
      { t: "list", items: [
        "Three features with standard deviations 0.106, 17.2 and 2.77 all come out with mean 0 and standard deviation 1 (γ = 1, β = 0 at the start).",
        "The analytic gradient matches the numerical one to relative error $10^{-8}$, and its column sums are 0 — the \"no part along $\\mathbf 1$\" property. Treating $\\mu$ and $\\sigma^2$ as constants gives an error of 8.77: a completely wrong gradient.",
        "After 200 batches the running mean is close to the true means $(5, -40, 0)$. The running variance is noisier (359 instead of 400 for the second feature): with momentum 0.1 the running average really only covers the last ~10 batches, so it inherits their noise.",
        "A training-mode batch of one is rejected (as in PyTorch), and even computed by hand it would map the example to exactly 0. Eval mode handles a single example correctly.",
        "Layer norm makes each **row** mean 0 / std 1; RMSNorm makes each row's RMS 1 but leaves its mean non-zero (0.85 and 0.14).",
      ] },
      { t: "callout", kind: "note", text: "The factor $\\tfrac{1}{N s}$ in the backward pass means gradients through BN are scaled by one over the batch spread: if a layer's pre-activations are large, its incoming gradient is automatically small. This is the scale invariance of Math §5 seen from the backward side." },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch: conventions", blocks: [
          { t: "p", text: "This program pushes the worked example through `nn.BatchNorm1d` and tests every convention: what `momentum` means, which variance is stored, the cumulative-average mode `momentum=None`, the batch-of-one error, and per-channel statistics in `BatchNorm2d`." },
          { t: "code", lang: "python", title: "bn_pytorch_conventions.py", code: BN_TORCH_CONV, output: BN_TORCH_CONV_OUT },
          { t: "list", items: [
            "**Momentum is the weight of the new batch**: $\\hat\\mu \\leftarrow 0.9\\hat\\mu + 0.1\\mu_B$. Keras/TensorFlow use the opposite (`momentum=0.99` means 0.99 × old + 0.01 × new), so PyTorch 0.1 ≈ Keras 0.9.",
            "**Normalising uses the biased variance** (5, 350); **the buffer stores the unbiased one** (6.667, 466.667). For large batches the gap is tiny; for small ones it isn't.",
            "`momentum=None` replaces the running average with an exact average of all batch statistics so far (useful for re-estimating statistics after training, e.g. after weight averaging).",
            "`BatchNorm2d` statistics are per channel over $N\\cdot H\\cdot W = 200$ values, matching the manual computation to float32 precision.",
          ] },
        ] },
        { label: "PyTorch: backward vs autograd", blocks: [
          { t: "p", text: "The three-term NumPy backward pass, generalised to any reduction axes, compared in float64 with PyTorch's own `F.batch_norm` differentiated by autograd — for a fully-connected input $(N, D)$ and a conv input $(N, C, H, W)$, where statistics are pooled over axes (0, 2, 3) and $M = N H W$." },
          { t: "code", lang: "python", title: "bn_backward_vs_autograd.py", code: BN_TORCH_BWD, output: BN_TORCH_BWD_OUT },
          { t: "p", text: "Agreement to ~$10^{-15}$ (float64 rounding) for the output and all three gradients confirms the derivation, and that the conv case is the same formula with $N$ replaced by $N H W$." },
        ] },
        { label: "PyTorch: digits experiment", blocks: [
          { t: "p", text: "A real comparison on scikit-learn's 8×8 handwritten digits (1437 training / 360 test images): an 8-hidden-layer ReLU MLP with He initialisation, trained with SGD + momentum, **with and without** a `BatchNorm1d` after every linear layer. Same seed and data order; only the learning rate changes." },
          { t: "code", lang: "python", title: "digits_bn_learning_rates.py", code: DIGITS_LR, output: DIGITS_LR_OUT },
          { t: "list", items: [
            "At a small learning rate (0.001) both networks behave alike — BN's gift is not magic accuracy, it is **robustness**.",
            "At 0.01 BN gets further (train loss 0.013 vs 0.038; test 0.975 vs 0.956).",
            "At 0.1 and 0.5 the plain network fails (0.356, then chance) while the BN network still trains (0.967, 0.928). At 2.0 both fail — BN widens the usable range of learning rates, it doesn't remove the limit. The plain network's loss is `nan`: its activations overflowed.",
          ] },
          { t: "p", text: "Second experiment: the **batch size** (learning rate 0.003, 5 epochs). The last column evaluates the BN model with batch statistics of the whole test set instead of running statistics, to tell \"bad running estimates\" apart from \"bad training\"." },
          { t: "code", lang: "python", title: "digits_bn_batch_size.py", code: DIGITS_BS, output: DIGITS_BS_OUT },
          { t: "list", items: [
            "With batches of 2 or 4, BN is **much worse** than no normalization (0.164 and 0.508 vs 0.925 and 0.950). Full-test-set statistics barely help at $N = 2$ (0.172): training itself is the problem — with two examples each normalised feature is ±1, so the network only learns from pairwise comparisons, with extremely noisy statistics.",
            "From batch 8 up, BN matches or beats the plain network. At batch 128 both underfit, because 5 epochs are only 55 steps at that size — but BN still helps (0.644 vs 0.439).",
            "This is why group/layer normalization is used when per-device batches are small (detection, segmentation, video, very large models).",
          ] },
        ] },
        { label: "PyTorch: the eval-mode bug", blocks: [
          { t: "p", text: "A two-hidden-layer BN network is trained on the digits, then evaluated correctly and incorrectly. The incorrect runs just forget `model.eval()`." },
          { t: "code", lang: "python", title: "forgot_model_eval.py", code: EVAL_BUG, output: EVAL_BUG_OUT },
          { t: "list", items: [
            "Correct evaluation gives 0.975 however the test set is batched, because each prediction depends only on its own input.",
            "Without `eval()`, accuracy depends on **how the test set is batched**: 0.969 as one big shuffled batch (its statistics look like the training ones), 0.775 in batches of 4, and 0.125 when each batch holds one class — the class-specific batch mean is subtracted away, removing exactly the information the classifier needs.",
            "A single-example prediction crashes. Worse, the buggy \"evaluation\" quietly **overwrote the running statistics** (layer 1's running mean moved by up to 0.125), so even a later correct `eval()` gives a different result (0.969) — evaluating the model changed it.",
          ] },
          { t: "callout", kind: "tip", text: "Use `model.eval()` together with `torch.no_grad()` (or `torch.inference_mode()`) for every validation/test pass, and `model.train()` before training again. `eval()` changes layer behaviour (BN, dropout); `no_grad()` only turns off gradient tracking — you need both." },
        ] },
        { label: "PyTorch: LN, RMSNorm, GN, IN", blocks: [
          { t: "p", text: "The axes table from the Mechanism section, checked: each built-in layer equals standardisation over the stated axes; group norm with 1 group is layer norm and with $C$ groups is instance norm; RMSNorm matches its formula; and only batch norm makes one sequence's output depend on another sequence in the batch. The last lines check the scale invariance of Math §5." },
          { t: "code", lang: "python", title: "normalization_family.py", code: NORMS_TORCH, output: NORMS_TORCH_OUT },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras's `BatchNormalization` works the same way with different defaults: `momentum=0.99` is the weight of the **old** running average, and `epsilon=1e-3`. Train/inference behaviour is set by the `training` argument. Since TensorFlow 2, setting `trainable = False` on a `BatchNormalization` layer also makes it run in inference mode — the usual recipe when fine-tuning a pretrained network." },
          { t: "code", lang: "python", title: "batchnorm_keras.py (not executed in CI)", code: BN_KERAS, output: "Prints the training-mode output (the same matrix Y as the worked example, up to rounding), the moving mean [0.5, 3.0] and the moving variance (updated from the batch variance; whether Keras applies the N/(N−1) correction differs from PyTorch's documented behaviour, so do not assume it matches 1.5667, 47.5667), the row-wise layer-normalised batch, and the model summary showing 128 BatchNormalization weights for 32 channels: 64 trainable (γ, β) and 64 non-trainable (moving mean and variance)." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Variant", "Statistics / idea", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Batch norm (Ioffe & Szegedy 2015)", "Per feature/channel over the batch (and space)", "Big speed-ups for CNNs; allows high learning rates; can be merged into conv weights at inference", "Needs fairly large batches; train/eval mismatch; links examples together; awkward for variable-length sequences", "CNNs for classification (ResNet-style), most vision backbones"],
        ["Layer norm (Ba et al. 2016)", "Per example over its features", "No batch dependence; same in training and inference; works for any sequence length and batch size 1", "Weaker than BN for many CNNs; normalises across channels that may carry very different information", "Transformers, RNNs"],
        ["RMSNorm (Zhang & Sennrich 2019)", "Per example: divide by root mean square; no centring, no β", "Cheaper (one sum), fewer parameters, similar quality", "No re-centring", "Recent LLMs (e.g. LLaMA)"],
        ["Group norm (Wu & He 2018)", "Per example over groups of C/G channels and space", "Accuracy steady across batch sizes; no running statistics", "Extra setting G (32 is common); slightly behind BN at large batch in their experiments", "Detection/segmentation with small per-GPU batches, diffusion-model U-Nets"],
        ["Instance norm (Ulyanov et al. 2016)", "Per example and channel over space", "Removes per-image contrast/style statistics", "Throws away information that may help recognition", "Style transfer, image generation (and AdaIN-style conditioning)"],
        ["Weight norm (Salimans & Kingma 2016)", "Rewrite $w = g\\,v/\\|v\\|$; normalises weights, not activations", "No batch dependence; almost free", "Doesn't control activation size by itself; less widely used", "Some generative and RL models, audio models"],
        ["Sync BN", "BN whose statistics are combined across all devices", "Gives back a large effective batch for statistics in data-parallel training", "Communication at every BN layer", "Detection/segmentation training on many GPUs (`nn.SyncBatchNorm`)"],
        ["Frozen BN", "Use fixed (pretrained) statistics and often fixed γ, β", "Removes batch dependence when fine-tuning with tiny batches", "Statistics no longer adapt to the new data", "Fine-tuning detectors (`torchvision.ops.FrozenBatchNorm2d`)"],
      ] },
      { t: "h", text: "Where to put the normalization" },
      { t: "list", items: [
        "**Before or after the activation?** The original BN paper normalises the pre-activation $Wu$ and then applies the nonlinearity: Linear/Conv → BN → ReLU, which is also the ResNet layout. BN after the activation also works in practice; the pre-activation spot has the advantage that the nonlinearity gets inputs with a controlled distribution. \"Pre-activation\" ResNets (He et al. 2016, *Identity Mappings in Deep Residual Networks*) reorder each residual branch as BN → ReLU → Conv, keeping the skip path a pure identity.",
        "**Post-LN Transformer** (the original 2017 Transformer, BERT): $x_{l+1} = \\mathrm{LN}(x_l + F(x_l))$ — normalization after the residual addition.",
        "**Pre-LN Transformer** (GPT-2 and most modern LLMs): $x_{l+1} = x_l + F(\\mathrm{LN}(x_l))$, plus a final LN before the output head. The residual stream is never normalised, so gradients flow through the identity path undamped. Xiong et al. (2020, *On Layer Normalization in the Transformer Architecture*) explained why post-LN needs learning-rate warm-up (large gradients near the output at the start) while pre-LN often trains without it; post-LN can reach slightly better quality when it does train stably, which is why new variants keep appearing.",
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Image classification backbones**: ResNets put BN after every convolution; inference engines merge the BN map into the conv weights so it costs nothing at deployment.",
        "**Transformers and LLMs**: every encoder/decoder block has layer normalization (BERT, GPT-2); LLaMA-family models use pre-normalization with RMSNorm, and T5 uses a scale-only layer norm without mean subtraction or bias.",
        "**Object detection and segmentation**: high-resolution inputs force small per-GPU batches, so these systems use Sync BN, frozen BN when fine-tuning pretrained backbones, or group normalization.",
        "**Diffusion and generative image models**: U-Net denoisers commonly use group normalization; style-transfer and style-based generators use instance normalization and its adaptive version (AdaIN), where $\\gamma, \\beta$ are predicted from a style input.",
        "**Domain adaptation**: re-estimating BN running statistics on unlabeled data from the new domain is a known, cheap way to adapt a model to shifted inputs.",
        "**Training very deep networks without normalization** is possible with careful initialisation and scaling (e.g. Fixup; NFNets by Brock et al. 2021) — normalization is a powerful convenience, not a strict requirement.",
      ] },
    ],
    tradeoffs: [
      { t: "h", text: "Strengths" },
      { t: "list", items: [
        "Faster, more reliable optimisation; a much wider range of usable learning rates (BN: 0.928 test accuracy at learning rate 0.5 where the plain network is at chance).",
        "Less sensitive to initialisation and to the scale of inputs and weights (scale invariance).",
        "BN's batch noise acts as a mild regulariser, often reducing the need for dropout in CNNs.",
        "At inference, BN is a free per-feature map (mergeable); LN/RMSNorm are cheap per-token sums.",
      ] },
      { t: "h", text: "Weaknesses and practical problems" },
      { t: "list", items: [
        "**Small batches** (BN): noisy statistics in training and poor running estimates — the batch-size experiment drops from 0.93 to 0.16 accuracy at batch 2.",
        "**Batch dependence** (BN): an example's output depends on its batch-mates. This breaks per-example gradient methods, leaks information between examples (a known pitfall in contrastive learning) and can be exploited when batches aren't shuffled.",
        "**Train/eval mismatch** (BN): running statistics may not match those seen in training — after few steps, after big learning-rate changes, after weight averaging, or when the data shifts. Re-estimating statistics with a pass over training data fixes many of these.",
        "**Distributed training** (BN): each device normalises its own sub-batch unless Sync BN is used, which costs communication.",
        "**Sequences** (BN): variable lengths and padding make per-feature batch statistics awkward; recurrent use would need separate statistics per time step. A main reason Transformers and RNNs use LN.",
        "**Cost**: extra memory for the cached $\\hat x$ and sums in every layer; in large models these memory-bound layers take a noticeable share of runtime, part of RMSNorm's appeal.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Forgetting `model.eval()` at test time.** BN then uses test-batch statistics: results depend on batching (0.975 → 0.775 with batches of 4, 0.125 with class-sorted batches in the experiment), single examples crash, and the running buffers get overwritten. Fix: `model.eval()` + `torch.no_grad()` for evaluation, `model.train()` before training.",
        "**Forgetting `model.train()` after validation.** The model then trains with frozen running statistics and no normalization of the current batch, which quietly hurts training. Validation loops must switch back.",
        "**Treating $\\mu$ and $\\sigma^2$ as constants in a hand-written backward pass.** The gradient is then wrong (relative error 8.77 in the NumPy check). Use the three-term formula and gradient-check new layers.",
        "**Keeping a bias before BN.** The mean subtraction cancels it, so it wastes parameters. Use `bias=False`; β does its job.",
        "**Mixing up momentum conventions** between PyTorch (weight of new: 0.1) and Keras (weight of old: 0.99). Using `momentum=0.99` in PyTorch makes the running statistics follow almost only the latest batch.",
        "**Using BN with tiny per-device batches** (≤ 4) — e.g. in detection or 3D segmentation. Use GroupNorm, Sync BN or frozen BN instead.",
        "**Fine-tuning with BN in train mode on small batches.** The well-estimated pretrained statistics get replaced by noisy ones. Freeze BN layers (`eval()` on them, or FrozenBatchNorm) unless the batch is large.",
        "**Claiming BN works because it \"removes internal covariate shift\".** The mechanism is still debated (Math §6); state the observed benefits — higher usable learning rates, smoother optimisation — instead.",
        "**Assuming LayerNorm normalises over the batch or the sequence.** In a Transformer `nn.LayerNorm(d_model)` normalises the $d_{model}$ features of each token separately; tokens never share statistics.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What does batch normalization compute during training?", a: "For each feature (or each channel, for convolutions) it computes the mean and variance over the mini-batch (and spatial positions), standardises to $\\hat x = (x-\\mu)/\\sqrt{\\sigma^2+\\epsilon}$, then outputs $\\gamma\\hat x + \\beta$ with learnable $\\gamma, \\beta$. It also updates running averages of the mean and variance for use at inference." },
      { t: "qa", level: "beginner", q: "Why does BN have learnable γ and β?", a: "Pure standardisation would force every feature to mean 0 and variance 1, limiting what the layer can represent (e.g. keeping a sigmoid in its linear zone). With $\\gamma = \\sqrt{\\sigma^2+\\epsilon}$ and $\\beta = \\mu$ the layer can represent the identity, so adding BN never reduces expressiveness; it just makes scale and offset two explicit learnable numbers per feature." },
      { t: "qa", level: "beginner", q: "How does BN behave differently at training and inference time?", a: "In training it uses the current batch's statistics and updates the running mean/variance. At inference (`model.eval()`) it uses the stored running statistics, so each prediction depends only on its own input and works for a batch of one. It is then a fixed per-feature map." },
      { t: "qa", level: "beginner", q: "What is the difference between batch norm and layer norm?", a: "They normalise over different axes. Batch norm computes one mean/variance per feature across the batch's examples; layer norm computes one mean/variance per example across its features. So layer norm needs no batch and no running statistics, and behaves the same in training and inference." },
      { t: "qa", level: "intermediate", q: "Why do Transformers use layer normalization rather than batch normalization?", a: "Sequence models handle variable-length, padded sequences, often with small or changing batches, and must decode one token at a time at inference. Per-token layer norm doesn't depend on batch size, sequence length or padding, has no train/inference mismatch and no running statistics. Batch statistics over tokens would mix positions and sequences, be skewed by padding, and link examples together." },
      { t: "qa", level: "intermediate", q: "How does BN work for convolutional layers, and how many parameters does BatchNorm2d(C) have?", a: "Statistics are per channel, pooled over batch and spatial positions ($N\\cdot H\\cdot W$ values each), matching convolution's weight sharing. It has $2C$ learnable parameters ($\\gamma, \\beta$) and $2C$ buffers (running mean and variance), plus a counter of batches seen." },
      { t: "qa", level: "intermediate", q: "Explain PyTorch's BN momentum and the variance stored in running_var.", a: "`running = (1 - momentum) * running + momentum * batch_stat`, default momentum 0.1, so momentum is the weight of the **new** batch (Keras uses the opposite, default 0.99 for the old value). The batch is normalised with the biased variance ($\\div N$), but running_var stores the unbiased estimate $\\tfrac{N}{N-1}\\sigma^2$. With `momentum=None` it keeps a cumulative average instead." },
      { t: "qa", level: "intermediate", q: "Why does BN perform badly with very small batches, and what are the alternatives?", a: "The batch mean and variance are noisy; with $N = 2$ each normalised feature is ±1 whatever the values. Training gets noisy and the running statistics poorly match what the network trained on. Alternatives: group norm or layer norm (no batch dependence), Sync BN (pool statistics across devices), frozen BN when fine-tuning, or Batch Renormalization (Ioffe 2017)." },
      { t: "qa", level: "advanced", q: "Derive the gradient of BN with respect to its inputs.", a: "With $\\hat g_i = \\gamma\\,\\partial\\mathcal{L}/\\partial y_i$ and $s = \\sqrt{\\sigma^2+\\epsilon}$: $\\partial\\mathcal{L}/\\partial\\sigma^2 = -\\frac{1}{2s^2}\\sum_i\\hat g_i\\hat x_i$; $\\partial\\mathcal{L}/\\partial\\mu = -\\frac1s\\sum_i\\hat g_i$ (the path through $\\sigma^2$ vanishes because $\\sum_i(x_i-\\mu)=0$); then $\\partial\\mathcal{L}/\\partial x_k = \\hat g_k/s + \\frac1N\\partial\\mathcal{L}/\\partial\\mu + \\frac{2(x_k-\\mu)}{N}\\partial\\mathcal{L}/\\partial\\sigma^2 = \\frac{1}{Ns}(N\\hat g_k - \\sum_i\\hat g_i - \\hat x_k\\sum_i\\hat g_i\\hat x_i)$. It removes from $\\hat g$ its parts along $\\mathbf 1$ and $\\hat x$, so the input gradients sum to zero." },
      { t: "qa", level: "advanced", q: "Is internal covariate shift why BN works?", a: "It was the original motivation, but Santurkar et al. (2018) reported that injecting distribution-changing noise after BN barely hurt its benefits, and that BN need not reduce internal covariate shift by their measures; they found instead that BN smooths the loss surface (loss and gradients change more slowly), making larger steps safe. Scale invariance — $\\mathrm{BN}(aWx) = \\mathrm{BN}(Wx)$ with gradients scaling as $1/a$ — and the regularising batch noise are further explanations. Honestly: the benefit is well established in practice, while the full explanation is still studied." },
      { t: "qa", level: "advanced", q: "Pre-LN vs post-LN Transformers: what is the difference and why does it matter?", a: "Post-LN applies $\\mathrm{LN}(x + F(x))$ after each residual addition (original Transformer, BERT); pre-LN applies $x + F(\\mathrm{LN}(x))$ and adds a final LN (GPT-2, most LLMs). In pre-LN the residual stream is an unnormalised identity path, so gradients reach early layers without passing through any LN. Deep stacks then train stably and depend less on learning-rate warm-up (analysed by Xiong et al. 2020). Post-LN tends to need warm-up and careful tuning at depth." },
      { t: "qa", level: "advanced", q: "How can BN be removed at deployment time?", a: "In eval mode BN is $y = a\\,x + c$ with $a = \\gamma/\\sqrt{\\hat\\sigma^2+\\epsilon}$ and $c = \\beta - a\\hat\\mu$ per channel. If it follows a convolution $x = W*u + b$, replace the weights by $a_c W_c$ (each output channel scaled) and the bias by $a_c b_c + c_c$. The merged convolution gives exactly the same output with no separate normalization layer." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A feature takes the values 1, 3, 5, 7 in a training batch. With $\\epsilon = 10^{-5}$, what is the standardised value $\\hat x$ for the example with value 7? (3 decimals)", answer: 1.342, tolerance: 0.002, explain: "$\\mu = 4$, $\\sigma^2 = (9+1+1+9)/4 = 5$, $\\hat x = 3/\\sqrt{5.00001} = 1.341639$." },
        { kind: "numeric", q: "PyTorch BN with momentum 0.1 has running_mean 2.0 for a feature. The current batch mean is 4.0. What is the new running_mean?", answer: 2.2, tolerance: 0.0001, explain: "$0.9\\times2.0 + 0.1\\times4.0 = 2.2$ — momentum is the weight of the new batch." },
        { kind: "numeric", q: "Same feature: batch of $N = 4$ with biased batch variance 5, old running_var 1.0, momentum 0.1. What is the new running_var in PyTorch? (4 decimals)", answer: 1.5667, tolerance: 0.0002, explain: "The stored variance is unbiased: $5\\times4/3 = 6.6667$. $0.9\\times1 + 0.1\\times6.6667 = 1.5667$." },
        { q: "Over which axes does `nn.BatchNorm2d` compute its mean and variance for an input of shape (N, C, H, W)?", options: ["(C, H, W), one statistic per example", "(N, H, W), one statistic per channel", "(H, W), one statistic per example and channel", "(N,), one statistic per pixel and channel"], answer: 1, explain: "BN pools over the batch and all positions for each channel, matching convolution's weight sharing. (C, H, W) is layer norm and (H, W) is instance norm." },
        { kind: "numeric", q: "How many **learnable** parameters does `nn.BatchNorm2d(64)` have?", answer: 128, tolerance: 0, explain: "One $\\gamma$ and one $\\beta$ per channel: $2\\times64 = 128$. The 128 running statistics are buffers, not parameters." },
        { q: "A Keras model uses `BatchNormalization(momentum=0.99)`. Which PyTorch setting gives the same running-average update?", options: ["momentum=0.99", "momentum=0.01", "momentum=0.1", "momentum=None"], answer: 1, explain: "Keras momentum is the weight of the old value; PyTorch momentum is the weight of the new batch, so 1 − 0.99 = 0.01." },
        { q: "You evaluate a trained BN model without calling `model.eval()`, feeding test batches that each contain a single class. What happens?", options: ["Nothing — BN is the identity at test time", "Accuracy can collapse, because the class-specific batch mean is subtracted out, and the running statistics are overwritten", "The model raises an error for every batch", "Accuracy improves because statistics match the test data"], answer: 1, explain: "In the experiment accuracy fell from 0.975 to 0.125, and the buggy pass also changed the running buffers. Only batches of size 1 raise an error." },
        { q: "Why is the bias of a linear layer placed directly before a BN layer redundant?", options: ["Because BN has no gradient for it", "Because the batch-mean subtraction removes any constant added to the whole feature; β takes over its role", "Because BN multiplies it by zero γ", "It is not redundant"], answer: 1, explain: "$(x+b) - \\mathrm{mean}(x+b) = x - \\mathrm{mean}(x)$, so $b$ has no effect on the output and gets zero gradient." },
        { q: "Which statement about RMSNorm is correct?", options: ["It normalises over the batch like BN", "It subtracts the mean and divides by the standard deviation", "It divides each token's features by their root mean square and has a scale γ but no shift β", "It needs running statistics at inference"], answer: 2, explain: "RMSNorm skips the mean subtraction and the shift: $y = \\gamma\\,x/\\mathrm{RMS}(x)$, per token." },
      ] },
      { t: "exercise", title: "BN forward pass and running statistics by hand", prompt: "A feature takes the values $x = (1, 3, 5, 7)$ in a training batch; $\\gamma = 2$, $\\beta = 1$, $\\epsilon = 10^{-5}$. (a) Compute $\\mu$, $\\sigma^2$, $\\hat x$ and $y$. (b) With PyTorch defaults (running_mean 0, running_var 1, momentum 0.1), compute both buffers after this batch. (c) In eval mode with these buffers, what is the output for an input of 4?", hint: "The stored variance uses $N/(N-1)$.", solution: [
        { t: "steps", items: [
          { title: "(a) Statistics", tex: String.raw`\mu = 4,\qquad \sigma^2 = \frac{9+1+1+9}{4} = 5,\qquad s = \sqrt{5.00001} = 2.236070` },
          { title: "(a) Standardise and scale", tex: String.raw`\hat x = (-1.341639,\ -0.447213,\ 0.447213,\ 1.341639),\qquad y = 2\hat x + 1 = (-1.683279,\ 0.105574,\ 1.894426,\ 3.683279)` },
          { title: "(b) Buffers", tex: String.raw`\hat\mu = 0.9\cdot0 + 0.1\cdot4 = 0.4,\qquad \hat\sigma^2 = 0.9\cdot1 + 0.1\cdot\tfrac43\cdot5 = 0.9 + 0.666667 = 1.566667` },
          { title: "(c) Eval-mode output for x = 4", tex: String.raw`y = 2\cdot\frac{4 - 0.4}{\sqrt{1.566677}} + 1 = 2\cdot\frac{3.6}{1.251670} + 1 = 6.752` },
        ] },
        { t: "p", text: "In training mode the input 4 equals the batch mean and would map to $\\beta = 1$; after one update the running statistics are still dominated by their starting values (0 and 1), so eval mode gives 6.752. Many batches are needed before the buffers can be trusted — about $\\ln 0.01/\\ln 0.9 \\approx 44$ updates for the starting value's weight $0.9^k$ to fall below 1%." },
      ] },
      { t: "exercise", title: "Implement and verify the layer-norm backward pass", prompt: "Layer norm standardises each **row** of an $(N, D)$ matrix over its $D$ features, with per-feature $\\gamma, \\beta$ shared by all rows. Write `ln_forward` and `ln_backward` in NumPy (returning $\\partial\\mathcal{L}/\\partial x$, $\\partial\\mathcal{L}/\\partial\\gamma$, $\\partial\\mathcal{L}/\\partial\\beta$) and check the gradients with central finite differences. Which sums in the BN formula move to the feature axis, and which don't?", hint: "The three-term formula is unchanged with $N$ replaced by $D$ and the sums taken along axis −1. But $\\gamma$ and $\\beta$ are still shared across rows, so their gradients still sum over axis 0.", solution: [
        { t: "code", lang: "python", title: "layernorm_backward.py", code: EX_LN, output: EX_LN_OUT },
        { t: "p", text: "The input-gradient sums (the mean and variance paths) move to the feature axis because the statistics are per row; the parameter-gradient sums stay on the batch axis because each $\\gamma_j, \\beta_j$ is used by every row. Each row of $\\partial\\mathcal{L}/\\partial x$ sums to zero, mirroring BN's column sums." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "All normalization layers compute $y = \\gamma(x-\\mu)/\\sqrt{\\sigma^2+\\epsilon} + \\beta$; they differ in the axes of $\\mu, \\sigma^2$: batch (BN), features of one example (LN), channel groups (GN), one channel (IN). RMSNorm drops centring: $y = \\gamma x/\\mathrm{RMS}(x)$.",
        "γ, β (starting at 1, 0) let the layer recover the identity, so normalization re-parameterises rather than restricts. A bias before BN is redundant.",
        "BN training: batch statistics (biased variance) + running average. PyTorch: $\\hat\\mu \\leftarrow (1-m)\\hat\\mu + m\\mu_B$, $m = 0.1$, running_var stores $\\tfrac{N}{N-1}\\sigma_B^2$. Keras: momentum 0.99 is the weight of the old value.",
        "BN inference: running statistics, a fixed per-feature map that can be merged into the previous layer. Always `model.eval()` for evaluation and `model.train()` for training.",
        "Backward: $\\partial\\mathcal{L}/\\partial x_k = \\frac{1}{Ns}(N\\hat g_k - \\sum_i\\hat g_i - \\hat x_k\\sum_i\\hat g_i\\hat x_i)$, $\\partial\\mathcal{L}/\\partial\\gamma = \\sum_i g_i\\hat x_i$, $\\partial\\mathcal{L}/\\partial\\beta = \\sum_i g_i$.",
        "Conv BN: one statistic per channel over $(N, H, W)$; $2C$ parameters and $2C$ buffers.",
        "Why it helps: first framed as reducing internal covariate shift; Santurkar et al. (2018) found evidence for a smoother loss surface instead. Scale invariance gives gradients $\\propto 1/\\|W\\|$. In practice: much higher usable learning rates (0.967 vs 0.356 accuracy at lr 0.1 on digits).",
        "Problems: small batches (0.164 accuracy at batch 2), batch dependence, train/eval mismatch, multi-device statistics (Sync BN).",
        "Transformers use LN (pre-LN in modern models, post-LN originally); LLMs often use RMSNorm; small-batch vision uses GN; style transfer uses IN.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Residual and Skip Connections](/learn/m07/residual-connections) — the other half of what makes very deep networks trainable, and why pre-norm placement matters.",
        "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — BN's batch noise is a mild regulariser; see how it interacts with dropout and weight decay.",
        "[Transformer Architecture](/learn/m11/transformer-architecture) — where layer normalization and RMSNorm sit inside each block.",
        "[Classic CNN Architectures](/learn/m08/cnn-architectures) — how BN opened the ResNet era of very deep convolutional networks.",
      ] },
    ],
  },
};

export default lesson;
