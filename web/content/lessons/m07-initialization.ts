import type { Lesson } from "@/lib/types";

const INIT_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_digits

# ---------------- 1. symmetry: constant initialisation keeps hidden units identical ----------------
X, y = load_digits(return_X_y=True)
X = X / 16.0                                     # pixel values 0..16 -> 0..1
Y = np.eye(10)[y]                                # one-hot targets
N, D, H, K = X.shape[0], 64, 16, 10              # samples, inputs, hidden units, classes


def train_two_layer(W1, b1, W2, b2, steps=300, lr=0.5):
    """Full-batch gradient descent on a tanh hidden layer + softmax output (cross-entropy)."""
    for _ in range(steps):
        Hh = np.tanh(X @ W1 + b1)                # (N, H)
        Z = Hh @ W2 + b2                         # (N, K)
        P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        dZ = (P - Y) / N                         # softmax + cross-entropy gradient
        dW2, db2 = Hh.T @ dZ, dZ.sum(0)
        dHh = dZ @ W2.T * (1 - Hh ** 2)          # backprop through tanh
        dW1, db1 = X.T @ dHh, dHh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    loss = -np.mean(np.log(P[np.arange(N), y] + 1e-12))
    acc = np.mean(P.argmax(1) == y)
    return W1, loss, acc


rng = np.random.default_rng(0)
inits = {
    "zeros":        (np.zeros((D, H)), np.zeros((H, K))),
    "constant 0.01": (np.full((D, H), 0.01), np.full((H, K), 0.01)),
    "random (Xavier)": (rng.normal(0, np.sqrt(2 / (D + H)), (D, H)), rng.normal(0, np.sqrt(2 / (H + K)), (H, K))),
}
print("init              loss    acc    max |W1[:, j] - W1[:, 0]|   rank(W1)")
for name, (W1, W2) in inits.items():
    W1, loss, acc = train_two_layer(W1.copy(), np.zeros(H), W2.copy(), np.zeros(K))
    spread = np.abs(W1 - W1[:, :1]).max()        # 0 means every hidden unit has the same weights
    print(f"{name:16s} {loss:6.3f}  {acc:.3f}   {spread:22.2e}   {np.linalg.matrix_rank(W1):5d}")

# ---------------- 2. forward variance through 20 layers (width 256) ----------------
n, L, B = 256, 20, 500
x0 = rng.normal(size=(B, n))
acts = {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0)}
stds = {"N(0, 0.01^2)": lambda: 0.01, "N(0, 1)": lambda: 1.0,
        "Xavier 1/n": lambda: np.sqrt(2 / (n + n)), "He 2/n": lambda: np.sqrt(2 / n)}
print(f"\nstd of pre-activations z[l], width {n}, batch {B}")
print("act   init             l=1       l=5      l=10      l=20")
for aname, f in acts.items():
    for iname, s in stds.items():
        h, rec = x0, []
        for l in range(1, L + 1):
            z = h @ rng.normal(0, s(), (n, n))
            h = f(z)
            if l in (1, 5, 10, 20):
                rec.append(z.std())
        print(f"{aname:5s} {iname:13s} " + " ".join(f"{v:9.2e}" for v in rec))
print(f"theory, ReLU + Xavier: std(z[20]) / std(z[1]) = 2^(-19/2) = {2 ** -9.5:.2e}")

# ---------------- 3. fan-in vs fan-out: a linear network that narrows 1024 -> 32 ----------------
widths = [1024, 512, 256, 128, 64, 32]
rules = {"fan_in   1/n_in": lambda i, o: 1 / i,
         "fan_out  1/n_out": lambda i, o: 1 / o,
         "Xavier 2/(n_in+n_out)": lambda i, o: 2 / (i + o)}
print("\nlinear net 1024-512-256-128-64-32: variance ratio output/input (forward) and input/output (backward)")
for name, var in rules.items():
    Ws = [rng.normal(0, np.sqrt(var(i, o)), (i, o)) for i, o in zip(widths[:-1], widths[1:])]
    h = rng.normal(size=(B, widths[0]))
    for W in Ws:
        h = h @ W
    g = rng.normal(size=(B, widths[-1]))       # upstream gradient at the output
    for W in reversed(Ws):
        g = g @ W.T
    print(f"{name:22s} forward {h.var():7.3f}   backward {g.var():7.3f}")
print(f"theory: fan_in -> 1 and {np.prod([o / i for i, o in zip(widths[:-1], widths[1:])]):.5f}; "
      f"fan_out -> {np.prod([i / o for i, o in zip(widths[:-1], widths[1:])]):.0f} and 1; "
      f"Xavier -> {(4 / 3) ** 5:.3f} and {(2 / 3) ** 5:.3f}")

# ---------------- 4. dynamical isometry: singular values of a deep linear network's Jacobian ----------------
n, L = 128, 30
for name, make in [("Gaussian N(0, 1/n)", lambda: rng.normal(0, np.sqrt(1 / n), (n, n))),
                   ("orthogonal", lambda: np.linalg.qr(rng.normal(size=(n, n)))[0])]:
    J = np.eye(n)
    for _ in range(L):
        J = make() @ J                           # Jacobian of x -> W_L ... W_1 x
    s = np.linalg.svd(J, compute_uv=False)
    print(f"{name:19s} depth {L}: largest singular value {s[0]:6.3f}, median {np.median(s):8.1e}, "
          f"fraction below 0.1: {np.mean(s < 0.1):.2f}")`;

const INIT_NUMPY_OUT = String.raw`init              loss    acc    max |W1[:, j] - W1[:, 0]|   rank(W1)
zeros             2.302  0.102                 0.00e+00       0
constant 0.01     1.490  0.318                 0.00e+00       1
random (Xavier)   0.098  0.984                 1.38e+00      16

std of pre-activations z[l], width 256, batch 500
act   init             l=1       l=5      l=10      l=20
tanh  N(0, 0.01^2)   1.60e-01  1.04e-04  1.09e-08  1.17e-16
tanh  N(0, 1)        1.60e+01  1.56e+01  1.56e+01  1.56e+01
tanh  Xavier 1/n     9.97e-01  3.54e-01  2.42e-01  1.64e-01
tanh  He 2/n         1.41e+00  8.16e-01  7.89e-01  7.66e-01
relu  N(0, 0.01^2)   1.60e-01  2.48e-05  3.93e-10  1.45e-19
relu  N(0, 1)        1.60e+01  2.38e+05  4.59e+10  1.60e+21
relu  Xavier 1/n     1.00e+00  2.20e-01  4.16e-02  1.38e-03
relu  He 2/n         1.41e+00  1.33e+00  1.58e+00  1.69e+00
theory, ReLU + Xavier: std(z[20]) / std(z[1]) = 2^(-19/2) = 1.38e-03

linear net 1024-512-256-128-64-32: variance ratio output/input (forward) and input/output (backward)
fan_in   1/n_in        forward   1.027   backward   0.033
fan_out  1/n_out       forward  32.522   backward   0.988
Xavier 2/(n_in+n_out)  forward   4.016   backward   0.124
theory: fan_in -> 1 and 0.03125; fan_out -> 32 and 1; Xavier -> 4.214 and 0.132
Gaussian N(0, 1/n)  depth 30: largest singular value  5.806, median  2.9e-05, fraction below 0.1: 0.81
orthogonal          depth 30: largest singular value  1.000, median  1.0e+00, fraction below 0.1: 0.00`;

const INIT_TORCH_DEFAULTS = String.raw`import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------- 1. what nn.Linear does by default ----------------
fan_in, fan_out = 256, 128
lin = nn.Linear(fan_in, fan_out)
w, b = lin.weight.detach(), lin.bias.detach()
print(f"nn.Linear({fan_in}, {fan_out}) default init")
print(f"  max |w| = {w.abs().max():.5f}   1/sqrt(fan_in) = {1 / math.sqrt(fan_in):.5f}")
print(f"  std(w)  = {w.std():.5f}   1/sqrt(3 fan_in) = {1 / math.sqrt(3 * fan_in):.5f}   He std sqrt(2/fan_in) = {math.sqrt(2 / fan_in):.5f}")
print(f"  max |b| = {b.abs().max():.5f}   (bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)))")

# reproduce it exactly: kaiming_uniform_ with a = sqrt(5) uses the same random numbers
torch.manual_seed(0)
ref = nn.Linear(fan_in, fan_out)
torch.manual_seed(0)
mine = torch.empty(fan_out, fan_in)
nn.init.kaiming_uniform_(mine, a=math.sqrt(5))
print(f"  identical to kaiming_uniform_(a=sqrt(5)): {torch.equal(mine, ref.weight.detach())}")
gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
print(f"  gain(leaky_relu, a=sqrt(5)) = sqrt(2/(1+5)) = {gain:.5f}, bound = gain*sqrt(3/fan_in) = {gain * math.sqrt(3 / fan_in):.5f}")

# ---------------- 2. gains for different nonlinearities ----------------
print("\ncalculate_gain:", ", ".join(f"{k} {nn.init.calculate_gain(k):.4f}" for k in ["linear", "sigmoid", "tanh", "relu", "selu"]) +
      f"; leaky_relu(0.01) {nn.init.calculate_gain('leaky_relu', 0.01):.4f}")

# ---------------- 3. the initialisers, checked against their formulas ----------------
W = torch.empty(512, 256)                              # (fan_out, fan_in) as in nn.Linear
checks = [
    ("xavier_uniform_", lambda t: nn.init.xavier_uniform_(t), math.sqrt(2 / (256 + 512))),
    ("xavier_normal_", lambda t: nn.init.xavier_normal_(t), math.sqrt(2 / (256 + 512))),
    ("kaiming_uniform_ relu", lambda t: nn.init.kaiming_uniform_(t, nonlinearity="relu"), math.sqrt(2 / 256)),
    ("kaiming_normal_ relu", lambda t: nn.init.kaiming_normal_(t, nonlinearity="relu"), math.sqrt(2 / 256)),
    ("kaiming_normal_ fan_out", lambda t: nn.init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu"), math.sqrt(2 / 512)),
    ("lecun (normal, std 1/sqrt(fan_in))", lambda t: nn.init.normal_(t, 0, 1 / math.sqrt(256)), math.sqrt(1 / 256)),
]
print("\ninitialiser                          measured std   formula std")
for name, fn, target in checks:
    fn(W)
    print(f"{name:36s} {W.std().item():12.5f} {target:13.5f}")
nn.init.xavier_uniform_(W)
print(f"xavier_uniform_ max |w| = {W.abs().max():.5f}, bound sqrt(6/(fan_in+fan_out)) = {math.sqrt(6 / 768):.5f}")
Q = torch.empty(256, 256)
nn.init.orthogonal_(Q)
print(f"orthogonal_: max |Q Q^T - I| = {(Q @ Q.T - torch.eye(256)).abs().max():.2e}, std = {Q.std():.5f} (1/sqrt(n) = {1 / 16:.5f})")

# ---------------- 4. convolutions: fan_in = C_in * k * k ----------------
conv = nn.Conv2d(16, 32, kernel_size=3)
fi, fo = nn.init._calculate_fan_in_and_fan_out(conv.weight)
print(f"\nConv2d(16, 32, 3): weight shape {tuple(conv.weight.shape)}, fan_in = {fi} (16*3*3), fan_out = {fo} (32*3*3)")
nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
print(f"  kaiming_normal_ std {conv.weight.std():.4f} vs sqrt(2/144) = {math.sqrt(2 / 144):.4f}")`;

const INIT_TORCH_DEFAULTS_OUT = String.raw`nn.Linear(256, 128) default init
  max |w| = 0.06250   1/sqrt(fan_in) = 0.06250
  std(w)  = 0.03602   1/sqrt(3 fan_in) = 0.03608   He std sqrt(2/fan_in) = 0.08839
  max |b| = 0.06151   (bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)))
  identical to kaiming_uniform_(a=sqrt(5)): True
  gain(leaky_relu, a=sqrt(5)) = sqrt(2/(1+5)) = 0.57735, bound = gain*sqrt(3/fan_in) = 0.06250

calculate_gain: linear 1.0000, sigmoid 1.0000, tanh 1.6667, relu 1.4142, selu 0.7500; leaky_relu(0.01) 1.4141

initialiser                          measured std   formula std
xavier_uniform_                           0.05107       0.05103
xavier_normal_                            0.05100       0.05103
kaiming_uniform_ relu                     0.08835       0.08839
kaiming_normal_ relu                      0.08844       0.08839
kaiming_normal_ fan_out                   0.06239       0.06250
lecun (normal, std 1/sqrt(fan_in))        0.06243       0.06250
xavier_uniform_ max |w| = 0.08839, bound sqrt(6/(fan_in+fan_out)) = 0.08839
orthogonal_: max |Q Q^T - I| = 7.15e-07, std = 0.06250 (1/sqrt(n) = 0.06250)

Conv2d(16, 32, 3): weight shape (32, 16, 3, 3), fan_in = 144 (16*3*3), fan_out = 288 (32*3*3)
  kaiming_normal_ std 0.1171 vs sqrt(2/144) = 0.1179`;

const INIT_TORCH_TRAIN = String.raw`import math
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)                         # small model: deterministic and faster on one thread
X, y = load_digits(return_X_y=True)
X = (X / 16.0 - 0.5) / 0.5                       # pixels 0..16 -> roughly [-1, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)
DEPTH, WIDTH, EPOCHS, BATCH, LR = 20, 64, 20, 32, 0.003


def make_mlp(act, init):
    layers, d = [], 64
    for _ in range(DEPTH):
        lin = nn.Linear(d, WIDTH)
        if init != "default":                    # "default" keeps PyTorch's kaiming_uniform_(a=sqrt(5))
            nn.init.zeros_(lin.bias)
            if init == "zeros":
                nn.init.zeros_(lin.weight)
            elif init == "N(0,0.01^2)":
                nn.init.normal_(lin.weight, 0.0, 0.01)
            elif init == "N(0,1)":
                nn.init.normal_(lin.weight, 0.0, 1.0)
            elif init == "xavier":
                nn.init.xavier_normal_(lin.weight)
            elif init == "he":
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            elif init == "orthogonal":
                nn.init.orthogonal_(lin.weight, gain=nn.init.calculate_gain("relu"))
        layers += [lin, act()]
        d = WIDTH
    return nn.Sequential(*layers, nn.Linear(d, 10))


@torch.no_grad()
def activation_stds(model, x):
    out = []
    for m in model:
        x = m(x)
        if not isinstance(m, nn.Linear):
            out.append(x.std().item())
    return out


def run(act, init):
    torch.manual_seed(0)
    model = make_mlp(act, init)
    stds = activation_stds(model, Xtr)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        curve = [loss_fn(model(Xtr), ytr).item()]
    g = torch.Generator().manual_seed(0)
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (1, 5, EPOCHS):
            with torch.no_grad():
                curve.append(loss_fn(model(Xtr), ytr).item())
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return stds, curve, acc


configs = [("ReLU", nn.ReLU, "zeros"), ("ReLU", nn.ReLU, "N(0,0.01^2)"), ("ReLU", nn.ReLU, "N(0,1)"),
           ("ReLU", nn.ReLU, "default"), ("ReLU", nn.ReLU, "xavier"), ("ReLU", nn.ReLU, "he"),
           ("ReLU", nn.ReLU, "orthogonal"), ("tanh", nn.Tanh, "xavier")]
print(f"{DEPTH} hidden layers of width {WIDTH}, sklearn digits, SGD momentum 0.9, lr {LR}")
print("                        activation std at layer       train loss at epoch        test")
print("act  init              1        5       10       20 |    0      1      5     20 | acc")
for name, act, init in configs:
    s, c, acc = run(act, init)
    print(f"{name:4s} {init:12s} " + " ".join(f"{s[l - 1]:8.1e}" for l in (1, 5, 10, 20)) + " | "
          + " ".join(f"{v:6.3f}" if v < 100 else f"{v:6.0e}" for v in c) + f" | {acc:.3f}")`;

const INIT_TORCH_TRAIN_OUT = String.raw`20 hidden layers of width 64, sklearn digits, SGD momentum 0.9, lr 0.003
                        activation std at layer       train loss at epoch        test
act  init              1        5       10       20 |    0      1      5     20 | acc
ReLU zeros         0.0e+00  0.0e+00  0.0e+00  0.0e+00 |  2.305  2.304  2.303  2.303 | 0.102
ReLU N(0,0.01^2)   3.8e-02  4.5e-07  1.9e-13  2.7e-26 |  2.306  2.305  2.304  2.303 | 0.102
ReLU N(0,1)        3.8e+00  4.5e+03  1.9e+07  2.7e+14 |  1e+14    nan    nan    nan | 0.100
ReLU default       3.4e-01  4.6e-02  4.9e-02  4.5e-02 |  2.304  2.304  2.303  2.303 | 0.102
ReLU xavier        4.8e-01  1.4e-01  1.7e-02  2.4e-04 |  2.306  2.305  2.303  1.742 | 0.200
ReLU he            6.8e-01  7.8e-01  5.6e-01  2.4e-01 |  2.288  1.560  0.370  0.034 | 0.942
ReLU orthogonal    7.0e-01  7.6e-01  9.6e-01  5.4e-01 |  2.317  1.798  0.395  0.016 | 0.969
tanh xavier        5.7e-01  3.2e-01  2.5e-01  1.7e-01 |  2.337  2.022  0.647  0.037 | 0.956`;

const INIT_TORCH_EXTRAS = String.raw`import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------- 1. output bias = log prior for an imbalanced binary task ----------------
p = 0.01                                          # 1% positives
y = (torch.rand(10000) < p).float()
feats = torch.randn(10000, 32)
head = nn.Linear(32, 1)
nn.init.normal_(head.weight, 0.0, 0.01)           # small weights: the bias dominates the initial logit
loss_fn = nn.BCEWithLogitsLoss()
for name, b0 in [("bias 0", 0.0), ("bias log(p/(1-p))", math.log(p / (1 - p)))]:
    nn.init.constant_(head.bias, b0)
    with torch.no_grad():
        out = head(feats).squeeze(1)
        print(f"{name:18s} b = {b0:7.3f}  mean predicted p = {torch.sigmoid(out).mean():.4f}  initial loss = {loss_fn(out, y):.4f}")
H = -(p * math.log(p) + (1 - p) * math.log(1 - p))
print(f"entropy of the label prior (best constant predictor): {H:.4f}")

# ---------------- 2. LSTM forget-gate bias = 1 ----------------
hidden = 4
lstm = nn.LSTM(input_size=3, hidden_size=hidden)
with torch.no_grad():
    for name, bias in lstm.named_parameters():
        if "bias" in name:                       # PyTorch gate order in the bias vector: input, forget, cell, output
            bias.zero_()
    lstm.bias_ih_l0[hidden:2 * hidden].fill_(1.0)  # forget gate; bias_hh_l0 adds to it, so set only one of them
print("\nLSTM bias_ih_l0 =", lstm.bias_ih_l0.detach().tolist())
print(f"forget gate at zero input: sigmoid(1) = {torch.sigmoid(torch.tensor(1.0)):.4f}  (vs sigmoid(0) = 0.5)")

# ---------------- 3. SELU + LeCun normal keeps activations near mean 0, variance 1 ----------------
def deep_stats(act, std_fn, depth=32, width=256):
    torch.manual_seed(1)
    x = torch.randn(2048, width)
    rows = []
    for l in range(1, depth + 1):
        W = torch.randn(width, width) * std_fn(width)
        x = act(x @ W.T)
        if l in (1, 8, 16, 32):
            rows.append(f"{x.mean():6.3f}/{x.var():.3f}")
    return rows


print("\n32 layers, width 256, mean/variance of activations at layers 1, 8, 16, 32")
print("SELU, LeCun N(0, 1/n):  ", "  ".join(deep_stats(torch.selu, lambda n: 1 / math.sqrt(n))))
print("ReLU, He N(0, 2/n):     ", "  ".join(deep_stats(torch.relu, lambda n: math.sqrt(2 / n))))
print("tanh, Xavier N(0, 1/n): ", "  ".join(deep_stats(torch.tanh, lambda n: 1 / math.sqrt(n))))

# ---------------- 4. zero-init the last layer of each residual branch ----------------
class ResBlock(nn.Module):
    def __init__(self, c, zero_init=False):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(c, c, bias=False), nn.BatchNorm1d(c)
        self.fc2, self.bn2 = nn.Linear(c, c, bias=False), nn.BatchNorm1d(c)
        if zero_init:
            nn.init.zeros_(self.bn2.weight)       # Goyal et al. (2017): gamma = 0 in the last BN of the branch

    def forward(self, x):
        out = self.bn2(self.fc2(torch.relu(self.bn1(self.fc1(x)))))
        return torch.relu(x + out)


x = torch.randn(256, 64)
for zero_init in (False, True):
    torch.manual_seed(2)
    net = nn.Sequential(*[ResBlock(64, zero_init) for _ in range(50)])
    out = net(x)
    print(f"\n50 residual blocks, zero-init gamma = {zero_init!s:5s}: std(out) = {out.std():.3f}, "
          f"max |out - relu(x)| = {(out - torch.relu(x)).abs().max():.3e}")
    out.pow(2).mean().backward()
    g = net[0].fc1.weight.grad.norm().item(), net[0].bn2.weight.grad.norm().item()
    print(f"  block 1: |dL/dW_fc1| = {g[0]:.3e}, |dL/dgamma_bn2| = {g[1]:.3e}")`;

const INIT_TORCH_EXTRAS_OUT = String.raw`bias 0             b =   0.000  mean predicted p = 0.5001  initial loss = 0.6939
bias log(p/(1-p))  b =  -4.595  mean predicted p = 0.0100  initial loss = 0.0662
entropy of the label prior (best constant predictor): 0.0560

LSTM bias_ih_l0 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
forget gate at zero input: sigmoid(1) = 0.7311  (vs sigmoid(0) = 0.5)

32 layers, width 256, mean/variance of activations at layers 1, 8, 16, 32
SELU, LeCun N(0, 1/n):   -0.001/1.001  -0.002/0.993   0.001/1.019  -0.005/0.973
ReLU, He N(0, 2/n):       0.564/0.682   0.603/0.744   0.447/0.449   0.480/0.499
tanh, Xavier N(0, 1/n):  -0.000/0.394  -0.000/0.065   0.000/0.033   0.000/0.014

50 residual blocks, zero-init gamma = False: std(out) = 4.509, max |out - relu(x)| = 4.856e+01
  block 1: |dL/dW_fc1| = 5.806e+01, |dL/dgamma_bn2| = 3.196e+00

50 residual blocks, zero-init gamma = True : std(out) = 0.582, max |out - relu(x)| = 0.000e+00
  block 1: |dL/dW_fc1| = 0.000e+00, |dL/dgamma_bn2| = 1.587e-02`;

const INIT_LSUV = String.raw`import numpy as np

rng = np.random.default_rng(0)


def orthogonal(n_out, n_in, gain=1.0):
    """Saxe et al. (2013): a random (semi-)orthogonal matrix from the QR decomposition of a Gaussian matrix."""
    A = rng.normal(size=(max(n_out, n_in), min(n_out, n_in)))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))                     # fix column signs so Q is uniformly distributed
    return gain * (Q if n_out >= n_in else Q.T)


def lsuv(Ws, X, act, tol=0.01, max_iter=10):
    """Mishkin & Matas (2016) LSUV: orthogonal init, then rescale each layer in turn
    so that its output on a data batch has variance 1."""
    h = X
    for l, W in enumerate(Ws):
        for _ in range(max_iter):
            v = (h @ W.T).var()
            if abs(v - 1) < tol:
                break
            W /= np.sqrt(v)                      # scale the weights by 1/std of the layer's output
        h = act(h @ W.T)
    return Ws


depth, n = 30, 128
X = rng.normal(2.0, 3.0, size=(512, n))         # deliberately NOT standardised: mean 2, std 3
relu = lambda z: np.maximum(z, 0)
for name, make in [("orthogonal, gain sqrt(2)", lambda: [orthogonal(n, n, np.sqrt(2)) for _ in range(depth)]),
                   ("orthogonal + LSUV", lambda: lsuv([orthogonal(n, n) for _ in range(depth)], X, relu))]:
    Ws = make()
    h, out = X, []
    for l, W in enumerate(Ws, 1):
        z = h @ W.T
        h = relu(z)
        if l in (1, 10, 20, 30):
            out.append(f"{z.var():8.3f}")
    print(f"{name:25s} Var(z) at layers 1, 10, 20, 30: " + " ".join(out))
W = orthogonal(64, 128)
print("semi-orthogonal 64x128: max |W W^T - I| =", f"{np.abs(W @ W.T - np.eye(64)).max():.1e}")`;

const INIT_LSUV_OUT = String.raw`orthogonal, gain sqrt(2)  Var(z) at layers 1, 10, 20, 30:   25.892   23.434   16.810   25.304
orthogonal + LSUV         Var(z) at layers 1, 10, 20, 30:    1.000    1.000    1.000    1.000
semi-orthogonal 64x128: max |W W^T - I| = 8.9e-16`;

const INIT_TORCH_APPLY = String.raw`import math
import torch
import torch.nn as nn


def init_weights(m):
    """Called once for every sub-module by model.apply(). Pick the rule by layer type."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 6 * 6, 10))
model.apply(init_weights)
for name, p in model.named_parameters():
    if p.dim() > 1:                                   # weight tensors only
        fan_in = p[0].numel()                          # inputs per output unit: C_in*k*k or n_in
        print(f"{name:9s} shape {str(tuple(p.shape)):15s} std {p.std():.4f}  He target sqrt(2/{fan_in}) = {math.sqrt(2 / fan_in):.4f}")
print("output shape on an 8x8 RGB batch:", tuple(model(torch.randn(4, 3, 8, 8)).shape))`;

const INIT_TORCH_APPLY_OUT = String.raw`0.weight  shape (32, 3, 3, 3)   std 0.2832  He target sqrt(2/27) = 0.2722
4.weight  shape (10, 1152)      std 0.0418  He target sqrt(2/1152) = 0.0417
output shape on an 8x8 RGB batch: (4, 10)`;

const INIT_KERAS = String.raw`import numpy as np
import tensorflow as tf

# Dense and Conv2D default to kernel_initializer="glorot_uniform", bias_initializer="zeros"
layer = tf.keras.layers.Dense(128)
layer.build((None, 256))
w = layer.kernel.numpy()
print("max |w| =", np.abs(w).max(), " Glorot bound =", np.sqrt(6 / (256 + 128)))


def deep_mlp(kernel_init, activation="relu", depth=20, width=64):
    model = tf.keras.Sequential([tf.keras.Input(shape=(64,))])
    for _ in range(depth):
        model.add(tf.keras.layers.Dense(width, activation=activation, kernel_initializer=kernel_init))
    model.add(tf.keras.layers.Dense(10))
    return model


x = np.random.default_rng(0).normal(size=(512, 64)).astype("float32")
for name, init in [("glorot_uniform (default)", "glorot_uniform"),
                   ("he_normal", tf.keras.initializers.HeNormal(seed=0)),
                   ("orthogonal, gain sqrt(2)", tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=0)),
                   ("VarianceScaling(2, fan_in)", tf.keras.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal", seed=0))]:
    model = deep_mlp(init)
    h = x
    for lyr in model.layers[:-1]:          # hidden layers only
        h = lyr(h)
    print(f"{name:28s} std of layer-20 activations: {float(tf.math.reduce_std(h)):.2e}")

# SELU network: LeCun normal + selu
selu_layer = tf.keras.layers.Dense(64, activation="selu", kernel_initializer="lecun_normal")
# imbalanced binary output: bias = log(p / (1 - p))
head = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(np.log(0.01 / 0.99)))
# LSTM: unit_forget_bias=True (the default) adds 1 to the forget-gate bias at initialisation
lstm = tf.keras.layers.LSTM(32, unit_forget_bias=True)`;

const lesson: Lesson = {
  id: "m07/initialization",
  title: "Weight Initialization: Xavier and He",
  summary: "Why the starting weights decide whether a deep network can learn: symmetry breaking, the forward and backward variance derivations behind LeCun, Xavier/Glorot and He/Kaiming initialisation, gains, orthogonal and data-dependent schemes, bias initialisation, framework defaults (measured), and how initialisation works together with normalisation and residual connections.",
  minutes: 85,
  level: "intermediate",
  prerequisites: ["m04/gradient-problems", "m04/backpropagation", "m02/probability", "m03/activation-functions"],
  next: ["m07/normalization", "m07/residual-connections", "m06/gradient-descent", "m08/cnn-architectures"],
  terms: ["symmetry-breaking", "fan-in", "xavier-init", "he-init", "lecun-init", "init-gain", "orthogonal-init", "dynamical-isometry", "lsuv", "self-normalizing-network", "zero-init-residual"],
  sections: {
    intro: [
      { t: "p", text: "Before the first training step, every weight in a network needs a starting value. This choice — the **initialisation** — looks like a detail, but it decides two things that later training can't easily undo:" },
      { t: "list", items: [
        "**Symmetry breaking:** can the hidden units become *different* from each other at all?",
        "**Scale:** do signals keep a sensible size as they flow forward through dozens of layers, and do gradients keep a sensible size flowing back?",
      ] },
      { t: "p", text: "In [Vanishing and Exploding Gradients](/learn/m04/gradient-problems) we saw that the gradient at layer 1 is a product of one factor per layer, so a factor a bit below or above 1 shrinks or blows up exponentially with depth. That lesson gave the one-line rule $n_{in}\\mathrm{Var}(W) = 1$ (tanh) or $= 2$ (ReLU). Here is the full story: the assumptions, the backward pass and Xavier's compromise, gains for any activation, SELU, orthogonal matrices, biases, convolutions, what PyTorch *actually* does by default (not He — we check), LSUV, and what changes with normalisation and residual connections." },
      { t: "p", text: "Good initialisation helped make deep networks trainable without layer-by-layer pre-training. Glorot and Bengio (2010, *\"Understanding the difficulty of training deep feedforward neural networks\"*) studied how activation and gradient sizes change through layers and proposed **Xavier** (or **Glorot**) initialisation. He, Zhang, Ren and Sun (2015, *\"Delving deep into rectifiers\"*) redid it for ReLU and got **He** (or **Kaiming**) initialisation, which let them train a 30-layer plain network from scratch." },
      { t: "callout", kind: "key", text: "A good initialisation (1) uses *random* weights so units differ, and (2) picks their variance so that, at step 0, activations have the same variance at every layer going forward and gradients have the same variance at every layer going backward. For a layer with fan-in $n_{in}$ that means $\\mathrm{Var}(W) = g^2/n_{in}$, where the gain $g$ depends on the activation ($g^2 = 2$ for ReLU)." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — the product-of-factors picture, the short variance argument for Xavier/He, and the `gradient-flow` simulator.",
        "[Backpropagation](/learn/m04/backpropagation) — the recurrence $\\delta^{[l-1]} = (W^{[l]\\top}\\delta^{[l]})\\odot f'(z^{[l-1]})$ and $\\partial\\mathcal{L}/\\partial W^{[l]} = \\delta^{[l]}a^{[l-1]\\top}$. The backward derivation is this recurrence with variances taken.",
        "[Probability](/learn/m02/probability) — expectation, variance, independence; $\\mathrm{Var}(\\sum_k X_k) = \\sum_k\\mathrm{Var}(X_k)$ for independent terms, $\\mathbb{E}[XY] = \\mathbb{E}[X]\\mathbb{E}[Y]$ for independent $X, Y$; the variance of a uniform distribution.",
        "[Activation functions](/learn/m03/activation-functions) — shapes and slopes of tanh, sigmoid, ReLU, Leaky ReLU and SELU.",
        "[Eigenvalues and SVD](/learn/m02/eigen-svd) — singular values and orthogonal matrices ($Q^\\top Q = I$), used for orthogonal initialisation.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A relay of microphones", text: "A message passes through twenty microphone-and-speaker stages, each feeding the next. If every stage turns the volume down a little, the last speaker is silent; if every stage turns it up a little, it screeches. Initialisation sets every volume knob before the show — not by listening to the music (that would be training), but by counting how many inputs each stage mixes. A stage that sums 500 inputs needs much quieter channels than one that sums 5. Where it stops: a real layer also reshapes the signal with its activation, which the gain accounts for." },
      { t: "p", text: "Two separate needs hide in that picture. The first is **symmetry breaking**. If every hidden unit in a layer starts with the same weights, they compute the same outputs, get the same gradients and stay the same forever — 512 units behave like one. Randomness is the cheapest way to make them differ." },
      { t: "p", text: "The second is **scale**. A pre-activation $z = \\sum_k w_k x_k$ adds up $n_{in}$ random terms, so its variance grows in proportion to $n_{in}$. To keep $z$ around size 1, each weight must shrink like $1/\\sqrt{n_{in}}$. The activation then adjusts this: ReLU throws away the negative half of $z$, halving its average square, so ReLU layers need weights $\\sqrt 2$ times larger than tanh layers. That one factor of 2 is the whole difference between Xavier and He." },
      { t: "p", text: "Finally, initialisation only fixes the *start*. Once training moves the weights, normalisation layers and residual connections keep the scale in check. With those present the question changes from \"keep the variance steady through 50 layers\" to \"make each residual branch start small, so the network begins close to a shallow one\"." },
    ],
    mechanism: [
      { t: "p", text: "What an initialiser does, step by step:" },
      { t: "steps", items: [
        { title: "Find the fan-in and fan-out of the weight tensor", text: "For a dense layer with weight shape $(n_{out}, n_{in})$ (PyTorch convention), fan-in $= n_{in}$ is how many inputs each output unit sums; fan-out $= n_{out}$ is how many units each input feeds. For a convolution with weight shape $(C_{out}, C_{in}, k_h, k_w)$, each output sums $C_{in}k_hk_w$ inputs, so fan-in $= C_{in}k_hk_w$ and fan-out $= C_{out}k_hk_w$." },
        { title: "Pick the target variance", text: "Choose which signal to keep steady (forward: fan-in; backward: fan-out; Xavier: their average) and the gain $g$ for the activation that follows. The target is $\\mathrm{Var}(W) = g^2/\\text{fan}$, or $g^2\\cdot 2/(\\text{fan}_{in} + \\text{fan}_{out})$ for Xavier." },
        { title: "Pick the distribution and turn the variance into its parameter", text: "Normal: $\\mathcal{N}(0, \\sigma^2)$ with $\\sigma = \\sqrt{\\mathrm{Var}(W)}$. Uniform: $\\mathcal{U}[-a, a]$ has variance $a^2/3$, so $a = \\sqrt{3\\,\\mathrm{Var}(W)}$. Truncated normal (Keras's `he_normal`, `lecun_normal`): a normal cut at $\\pm2\\sigma$, with $\\sigma$ adjusted so the variance still matches. Orthogonal: a random orthogonal matrix times $g$." },
        { title: "Draw each weight independently", text: "Every entry is sampled i.i.d. with mean 0. Independence makes the variance-of-a-sum argument work; randomness breaks symmetry." },
        { title: "Set the biases", text: "Usually zero (the weights already break symmetry). Exceptions: LSTM forget-gate biases are set to 1 so the cell remembers by default; a classifier's output bias can be set to the log prior so first predictions match the class frequencies." },
        { title: "Apply special rules for special layers", text: "Normalisation layers start as the identity ($\\gamma = 1$, $\\beta = 0$). In residual networks the last layer (or last $\\gamma$) of each branch is often set to zero so every block starts as the identity. Embeddings, attention projections and output heads may have their own scaled rules." },
        { title: "Check, then train", text: "Run one forward pass on a real batch and print each layer's activation std (and, after one backward pass, each layer's gradient norm). If they stay within about 10× of each other across layers, the initialisation works; steady shrinking or growth from layer to layer means it doesn't." },
      ] },
      { t: "h", text: "How it is applied in code" },
      { t: "p", text: "Every PyTorch layer initialises itself in its constructor via `reset_parameters()`. To override it, call the functions in `torch.nn.init` on specific tensors (they end in an underscore because they change the tensor in place), or write a function and pass it to `model.apply(fn)`, which visits every sub-module once:" },
      { t: "code", lang: "python", title: "init_with_apply.py", code: INIT_TORCH_APPLY, output: INIT_TORCH_APPLY_OUT },
      { t: "p", text: "The measured stds match $\\sqrt{2/\\text{fan}_{in}}$ up to sampling noise (the convolution has only $32\\cdot27 = 864$ weights, so its std is less precise). `p[0].numel()` is the fan-in for both layer types: one output unit's slice of the weight tensor." },
    ],
    math: [
      { t: "h", text: "1. Symmetry: why constant initialisation can never work" },
      { t: "p", text: "Take a hidden layer $h = f(W^{[1]}x + b^{[1]})$ followed by $\\hat y = g(W^{[2]}h + b^{[2]})$. Say two hidden units $j$ and $k$ start with the same incoming weights ($W^{[1]}_{j,:} = W^{[1]}_{k,:}$, $b^{[1]}_j = b^{[1]}_k$) and the same outgoing weights ($W^{[2]}_{:,j} = W^{[2]}_{:,k}$). Any constant initialisation does this." },
      { t: "steps", items: [
        { title: "Same forward values", text: "For every input $x$: $z_j = W^{[1]}_{j,:}x + b_j = z_k$, so $h_j = h_k$." },
        { title: "Same backward errors", tex: String.raw`\delta^{[1]}_j = f'(z_j)\sum_i W^{[2]}_{ij}\delta^{[2]}_i = f'(z_k)\sum_i W^{[2]}_{ik}\delta^{[2]}_i = \delta^{[1]}_k` },
        { title: "Same gradients", tex: String.raw`\frac{\partial\mathcal L}{\partial W^{[1]}_{j,:}} = \delta^{[1]}_j\,x^\top = \frac{\partial\mathcal L}{\partial W^{[1]}_{k,:}},\qquad \frac{\partial\mathcal L}{\partial W^{[2]}_{:,j}} = \delta^{[2]}h_j = \frac{\partial\mathcal L}{\partial W^{[2]}_{:,k}}` },
        { title: "Same update, forever", text: "Gradient descent (and any optimizer that treats each parameter on its own, like Adam) gives both the same update, so after the step they are still identical — and the same holds at every later step. The layer can only do what a single unit can. Only something that treats units differently — random initialisation, dropout masks — can separate them." },
      ] },
      { t: "p", text: "**All-zero is even worse.** With $W^{[1]} = 0$, $W^{[2]} = 0$ and tanh hidden units, $h = \\tanh(0) = 0$ for every input, so $\\partial\\mathcal L/\\partial W^{[2]} = \\delta^{[2]}h^\\top = 0$, and $\\delta^{[1]} = f'(0)\\,W^{[2]\\top}\\delta^{[2]} = 0$, so $\\partial\\mathcal L/\\partial W^{[1]} = 0$. Neither weight matrix ever changes; only the output bias learns, and the network predicts the class frequencies. On the digits data below that gives a loss of $2.302 \\approx \\ln 10$ and 10.2% accuracy. With a small constant (0.01) the units stay identical (rank of $W^{[1]}$ stays 1) and the network acts like one hidden unit: 31.8% accuracy versus 98.4% with random weights." },
      { t: "h", text: "2. Forward pass: the variance of a pre-activation" },
      { t: "p", text: "Take one unit of layer $l$: $z = \\sum_{k=1}^{n_{in}} W_k x_k$ (bias 0 at the start). Everything below rests on these assumptions:" },
      { t: "list", ordered: true, items: [
        "**(A1)** The weights $W_k$ are i.i.d. with mean 0 and variance $\\mathrm{Var}(W)$.",
        "**(A2)** The weights are independent of the inputs $x_k$ (true at initialisation, false after training).",
        "**(A3)** The inputs $x_k$ all have the same distribution (they need not be independent, because the products $W_kx_k$ are uncorrelated anyway: $\\mathbb{E}[W_kx_kW_mx_m] = \\mathbb{E}[W_k]\\mathbb{E}[W_m]\\mathbb{E}[x_kx_m] = 0$ for $k \\ne m$).",
      ] },
      { t: "p", text: "Each term has mean $\\mathbb{E}[W_k]\\mathbb{E}[x_k] = 0$ and variance $\\mathbb{E}[W_k^2x_k^2] - 0 = \\mathrm{Var}(W)\\,\\mathbb{E}[x^2]$. Uncorrelated terms' variances add:" },
      { t: "math", tex: String.raw`\mathrm{Var}(z) = n_{in}\,\mathrm{Var}(W)\,\mathbb{E}\big[x^2\big] \;\overset{\mathbb{E}[x]=0}{=}\; n_{in}\,\mathrm{Var}(W)\,\mathrm{Var}(x)`, caption: "n_in = fan-in; E[x²] is the average square of the layer's input, which equals Var(x) only when the input has mean zero." },
      { t: "p", text: "In words: the output's spread is (number of inputs) × (weight spread) × (input's average square). The difference between $\\mathbb{E}[x^2]$ and $\\mathrm{Var}(x)$ matters: ReLU outputs never have mean zero, and using $\\mathrm{Var}(x)$ there gives the wrong answer. He et al. (2015) made exactly this fix to Glorot and Bengio's analysis. Applying the formula layer after layer, with $x = f(z^{[l-1]})$:" },
      { t: "math", tex: String.raw`\mathrm{Var}\big(z^{[L]}\big) = \mathrm{Var}\big(z^{[1]}\big)\prod_{l=2}^{L} n_{l-1}\,\mathrm{Var}\big(W^{[l]}\big)\,\frac{\mathbb{E}\big[f(z^{[l-1]})^2\big]}{\mathrm{Var}\big(z^{[l-1]}\big)}`, caption: "A product of one factor per layer: unless each factor is 1, the variance changes exponentially with depth." },
      { t: "h", text: "3. What the activation contributes" },
      { t: "p", text: "The ratio $c_f = \\mathbb{E}[f(z)^2]/\\mathrm{Var}(z)$, for zero-mean symmetric $z$, measures how much the activation shrinks the signal:" },
      { t: "list", items: [
        "**Linear / tanh near 0:** $f(z) \\approx z$, so $c_f = 1$. (For larger $z$ tanh squashes, so $c_f < 1$; for $z \\sim \\mathcal{N}(0,1)$, $\\mathbb{E}[\\tanh(z)^2] = 0.395$.)",
        "**ReLU:** $\\mathbb{E}[\\mathrm{ReLU}(z)^2] = \\int_0^\\infty z^2p(z)\\,dz = \\tfrac12\\mathbb{E}[z^2]$ because $p$ is symmetric, so $c_f = \\tfrac12$.",
        "**Leaky ReLU with slope $a$:** the negative half adds $a^2\\cdot\\tfrac12\\mathbb{E}[z^2]$, so $c_f = (1 + a^2)/2$. A Monte-Carlo check with $10^6$ samples gives $0.5214$ for $a = 0.2$ against the formula's $0.52$.",
      ] },
      { t: "p", text: "Setting each layer's factor $n_{in}\\mathrm{Var}(W)\\,c_f$ to 1 gives the forward rule $\\mathrm{Var}(W) = 1/(c_f\\,n_{in})$:" },
      { t: "math", tex: String.raw`\underbrace{\mathrm{Var}(W) = \frac{1}{n_{in}}}_{\text{LeCun (linear, tanh, SELU)}}\qquad \underbrace{\mathrm{Var}(W) = \frac{2}{n_{in}}}_{\text{He / Kaiming (ReLU)}}\qquad \underbrace{\mathrm{Var}(W) = \frac{2}{(1+a^2)\,n_{in}}}_{\text{He for Leaky ReLU}}` },
      { t: "h", text: "4. Backward pass" },
      { t: "p", text: "Backprop computes $\\delta^{[l-1]}_k = f'(z^{[l-1]}_k)\\sum_{j=1}^{n_{out}} W^{[l]}_{jk}\\delta^{[l]}_j$, a sum over the $n_{out}$ units that unit $k$ feeds. With matching assumptions (weights independent of the $\\delta$s, and $f'(z)$ independent of the incoming sum), the same argument gives" },
      { t: "math", tex: String.raw`\mathrm{Var}\big(\delta^{[l-1]}\big) = n_{out}\,\mathrm{Var}\big(W^{[l]}\big)\,\mathbb{E}\big[f'(z)^2\big]\,\mathrm{Var}\big(\delta^{[l]}\big)`, caption: "n_out = fan-out of layer l. For tanh near 0, E[f′²] ≈ 1; for ReLU, f′ is 0 or 1 with probability ½ each, so E[f′²] = ½." },
      { t: "p", text: "Keeping the backward variance steady needs $\\mathrm{Var}(W) = 1/n_{out}$ (tanh) or $2/n_{out}$ (ReLU) — the **fan-out** mode. The weight gradient $\\partial\\mathcal L/\\partial W^{[l]}_{jk} = \\delta^{[l]}_j a^{[l-1]}_k$ has variance $\\mathrm{Var}(\\delta^{[l]})\\,\\mathbb{E}[(a^{[l-1]})^2]$. If both the forward and backward conditions hold, weight gradients have the same size in every layer, so one learning rate suits them all." },
      { t: "h", text: "5. Xavier / Glorot: the compromise" },
      { t: "p", text: "Forward wants $n_{in}\\mathrm{Var}(W) = 1$, backward wants $n_{out}\\mathrm{Var}(W) = 1$. Both hold only when $n_{in} = n_{out}$. Glorot and Bengio (2010) averaged the two, $\\tfrac12(n_{in} + n_{out})\\mathrm{Var}(W) = 1$:" },
      { t: "math", tex: String.raw`\mathrm{Var}(W) = \frac{2}{n_{in} + n_{out}}\qquad\Longrightarrow\qquad W \sim \mathcal{N}\Big(0,\ \frac{2}{n_{in}+n_{out}}\Big)\quad\text{or}\quad W \sim \mathcal{U}\Big[-\sqrt{\tfrac{6}{n_{in}+n_{out}}},\ \sqrt{\tfrac{6}{n_{in}+n_{out}}}\Big]` },
      { t: "p", text: "The uniform bound comes from $\\mathrm{Var}(\\mathcal U[-a, a]) = a^2/3$: setting $a^2/3 = 2/(n_{in}+n_{out})$ gives $a = \\sqrt{6/(n_{in}+n_{out})}$. The paper proposed the uniform version (calling it the *normalized initialization*) to replace the then-common rule $\\mathcal{U}[-1/\\sqrt{n_{in}}, 1/\\sqrt{n_{in}}]$, whose variance $1/(3n_{in})$ shrinks the signal 3× per linear layer. That old rule still matters: it is, numerically, PyTorch's default (Section 9)." },
      { t: "p", text: "How much does the choice of fan matter? In a linear network that narrows $1024 \\to 512 \\to 256 \\to 128 \\to 64 \\to 32$ (halving each layer), the measured variance ratios are: fan-in rule — forward 1.03, backward 0.033 (theory 1 and $1/32$); fan-out rule — forward 32.5, backward 0.99 (theory 32 and 1); Xavier — forward 4.0, backward 0.12 (theory $(4/3)^5 = 4.21$ and $(2/3)^5 = 0.132$). Xavier splits the error between both directions. Notice the fan-in rule's backward error, $1/32$, is exactly last width over first width: the factors **telescope** (cancel in a chain) rather than pile up." },
      { t: "h", text: "6. He / Kaiming initialisation and the factor 2" },
      { t: "p", text: "For ReLU, $c_f = \\tfrac12$ forward and $\\mathbb{E}[f'^2] = \\tfrac12$ backward. Keeping the forward variance steady:" },
      { t: "math", tex: String.raw`\mathrm{Var}\big(z^{[l]}\big) = n_{in}\,\mathrm{Var}(W)\cdot\tfrac12\,\mathrm{Var}\big(z^{[l-1]}\big) \overset{!}{=} \mathrm{Var}\big(z^{[l-1]}\big)\quad\Longrightarrow\quad \mathrm{Var}(W) = \frac{2}{n_{in}},\quad \sigma_W = \sqrt{\frac{2}{n_{in}}},\quad a_{\text{uniform}} = \sqrt{\frac{6}{n_{in}}}` },
      { t: "p", text: "The factor 2 exactly makes up for the half of the pre-activation that ReLU zeroes. He et al. also noted that ReLU networks need no compromise: if every layer uses the fan-in rule, the backward variance changes by $\\prod_l n_l/n_{l-1} = n_L/n_0$ over the whole network — one ratio of widths, not something that compounds with depth (the telescoping above). Fan-out everywhere works the same way. PyTorch offers `mode=\"fan_in\"` (default) or `mode=\"fan_out\"` (used by torchvision's ResNets for convolutions)." },
      { t: "p", text: "Measured on 20 layers of width 256 (NumPy program below): with He, the std of $z$ stays between 1.33 and 1.69 at layers 1–20; with Xavier ($1/n$ for square layers) it falls from 1.00 to $1.38\\times10^{-3}$ — exactly the predicted $2^{-19/2} = 1.38\\times10^{-3}$; with $\\mathcal N(0, 1)$ it explodes to $1.6\\times10^{21}$." },
      { t: "h", text: "7. Gains: one formula for every activation" },
      { t: "p", text: "Frameworks write all of these with a **gain** $g$: $\\sigma_W = g/\\sqrt{\\text{fan}}$ (He/LeCun family) or $\\sigma_W = g\\sqrt{2/(\\text{fan}_{in}+\\text{fan}_{out})}$ (Xavier family). $g^2 = 1/c_f$ undoes the activation's shrinking. PyTorch's `nn.init.calculate_gain` returns (printed by the program below):" },
      { t: "table", head: ["nonlinearity", "gain g", "where it comes from"], rows: [
        ["linear, conv, sigmoid", "1", "identity has c_f = 1; for sigmoid this is a convention, not a derivation"],
        ["tanh", "5/3 ≈ 1.6667", "chosen by experiment to keep tanh in its near-linear, non-saturated range"],
        ["relu", "√2 ≈ 1.4142", "c_f = ½ (derived above)"],
        ["leaky_relu (slope a)", "√(2/(1+a²))", "c_f = (1+a²)/2; a = 0.01 gives 1.4141"],
        ["selu", "3/4", "PyTorch's documented value; for self-normalising networks the docs recommend the linear gain (Var(W) = 1/n_in) instead"],
      ] },
      { t: "h", text: "8. LeCun initialisation and self-normalising networks" },
      { t: "p", text: "$\\mathrm{Var}(W) = 1/n_{in}$ is the oldest rule (LeCun, Bottou, Orr and Müller, *\"Efficient BackProp\"*, 1998), derived for standardised inputs and roughly linear units. It came back with the **SELU** activation of Klambauer, Unterthiner, Mayr and Hochreiter (2017, *\"Self-Normalizing Neural Networks\"*):" },
      { t: "math", tex: String.raw`\mathrm{selu}(z) = \lambda\begin{cases} z & z > 0\\ \alpha\,(e^{z} - 1) & z \le 0\end{cases},\qquad \lambda \approx 1.0507,\ \alpha \approx 1.6733` },
      { t: "p", text: "The constants are picked so that, when weights have mean 0 and $n_{in}\\mathrm{Var}(W) = 1$, a layer's (mean, variance) is pulled toward (0, 1). Activations that drift away get pulled back layer by layer — normalisation without a normalisation layer. Below, 32 SELU layers with LeCun-normal weights keep mean/variance at $-0.001/1.001$ after layer 1 and $-0.005/0.973$ after layer 32. A He-initialised ReLU network has mean around 0.5 (ReLU outputs are positive), and a tanh/Xavier network's activation variance decays from 0.394 to 0.014." },
      { t: "h", text: "9. What PyTorch does by default" },
      { t: "p", text: "`nn.Linear` and `nn.Conv2d` initialise weights with `kaiming_uniform_(w, a=math.sqrt(5))`. That is the He-uniform formula for a *Leaky ReLU with slope* $\\sqrt5$, whose gain is $\\sqrt{2/(1+5)} = \\sqrt{1/3}$, so the bound is" },
      { t: "math", tex: String.raw`a_{\text{bound}} = g\sqrt{\frac{3}{n_{in}}} = \sqrt{\frac13}\sqrt{\frac{3}{n_{in}}} = \frac{1}{\sqrt{n_{in}}},\qquad \mathrm{Var}(W) = \frac{a_{\text{bound}}^2}{3} = \frac{1}{3\,n_{in}}` },
      { t: "p", text: "Biases are drawn from $\\mathcal U[-1/\\sqrt{n_{in}}, 1/\\sqrt{n_{in}}]$ too. The program below confirms it: for `nn.Linear(256, 128)` the largest weight is $0.06250 = 1/\\sqrt{256}$, the std is $0.03602$ against $1/\\sqrt{3\\cdot256} = 0.03608$, and `kaiming_uniform_(a=sqrt(5))` from the same seed reproduces the weights bit for bit. This is the old $\\mathcal U[-1/\\sqrt{n}, 1/\\sqrt{n}]$ rule, and it is **not** He: its variance is $1/6$ of He's." },
      { t: "p", text: "In a ReLU network each layer then multiplies the forward variance by $n\\cdot\\frac{1}{3n}\\cdot\\frac12 = \\frac16$ (std $\\times0.408$). In the 20-layer digits experiment below, the default-initialised ReLU network's activation std drops from 0.34 at layer 1 to 0.046 at layer 5, then stays near 0.045 only because the random biases (std $1/\\sqrt{3\\cdot64} = 0.072$) add a floor — and the network doesn't train at all (loss stuck at 2.303). For shallow networks the default is fine; for deep plain ReLU stacks, set He explicitly." },
      { t: "h", text: "10. Orthogonal initialisation and dynamical isometry" },
      { t: "p", text: "Variance arguments control the *average* stretch of a vector. The full picture is in the singular values of the input–output Jacobian $J = \\partial z^{[L]}/\\partial x$ (how each input direction gets stretched). For a deep linear network $J = W^{[L]}\\cdots W^{[1]}$. Even if every $W$ has $\\mathrm{Var}(W) = 1/n$ (average squared stretch 1), a product of random Gaussian matrices has wildly spread singular values: for 30 layers of width 128 we measure a largest of 5.8, a median of $2.9\\times10^{-5}$, and 81% below 0.1. Most directions get crushed, a few amplified, and gradient descent faces a badly shaped problem." },
      { t: "p", text: "Saxe, McClelland and Ganguli (2013, *\"Exact solutions to the nonlinear dynamics of learning in deep linear neural networks\"*) proposed **orthogonal initialisation**: each $W$ is a random orthogonal matrix ($W^\\top W = I$, from the QR decomposition of a Gaussian matrix), times the gain. Products of orthogonal matrices stay orthogonal, so in a linear network every singular value of $J$ is exactly 1 at any depth (measured: all 128 equal 1.000). Deep linear networks started this way can learn in time independent of depth. A network whose Jacobian has all singular values near 1 has **dynamical isometry**. Pennington, Schoenholz and Ganguli (2017, *\"Resurrecting the sigmoid in deep learning through dynamical isometry\"*) showed tanh-like networks with orthogonal weights and a tuned scale can get close to it; ReLU networks cannot. Orthogonal initialisation is still standard for recurrent weights, where one matrix is applied many times." },
      { t: "h", text: "11. Worked example: three initialisations of a 784 → 256 → 256 ReLU network" },
      { t: "p", text: "Standardised inputs ($\\mathbb E[x^2] = 1$), biases 0; layer 1 has $n_{in} = 784$, $n_{out} = 256$; layer 2 is $256 \\to 256$. Theory (exact) and, in brackets, a simulation with 4000 Gaussian inputs:" },
      { t: "table", head: ["init", "Var(W¹)", "Var(z¹) = 784·Var(W¹)·1", "E[(a¹)²] = ½Var(z¹)", "Var(W²)", "Var(z²) = 256·Var(W²)·E[(a¹)²]"], rows: [
        ["He", "2/784 = 0.002551 (std 0.05051)", "2.0000 [1.9889]", "1.0000 [0.9953]", "2/256 = 0.007813", "2.0000 [2.0141]"],
        ["Xavier", "2/1040 = 0.001923 (std 0.04385)", "1.5077 [1.4973]", "0.7538 [0.7478]", "2/512 = 0.003906", "0.7538 [0.7247]"],
        ["PyTorch default", "1/2352 = 0.000425 (std 0.02062)", "0.3333 [0.3351]", "0.1667 [0.1678]", "1/768 = 0.001302", "0.0556 [0.0565]"],
      ] },
      { t: "p", text: "Only He keeps $\\mathrm{Var}(z)$ the same from layer to layer (2 → 2). Xavier loses 2× per square ReLU layer (1.51 → 0.75), and the PyTorch default loses 6× (0.333 → 0.056). The uniform bounds for layer 1 would be $\\sqrt{6/784} = 0.08748$ (He), $\\sqrt{6/1040} = 0.07596$ (Xavier) and $1/\\sqrt{784} = 0.03571$ (default)." },
      { t: "h", text: "12. Convolutions" },
      { t: "p", text: "A convolution output $z_{c,i,j} = \\sum_{c'}\\sum_{u,v} W_{c,c',u,v}\\,x_{c',i+u,j+v}$ is again a sum of $C_{in}k_hk_w$ weight-times-input terms, so every formula above applies with $n_{in} = C_{in}k_hk_w$ (and $n_{out} = C_{out}k_hk_w$ backward, because each input value is read by $C_{out}$ kernels at $k_hk_w$ positions). For `Conv2d(16, 32, 3)` the program reports weight shape $(32, 16, 3, 3)$, fan-in $144$, fan-out $288$, and He std $\\sqrt{2/144} = 0.1179$. A $3\\times3$ convolution with 64 input channels has the same fan-in (576) as a dense layer with 576 inputs." },
      { t: "h", text: "13. Biases" },
      { t: "list", items: [
        "**Default: zero.** The weights already break symmetry, and a zero bias keeps the pre-activation at mean zero, as the variance analysis assumes. (A small positive bias such as 0.01 for ReLU units is sometimes used to reduce dead units at the start; evidence that it helps is mixed.)",
        "**LSTM forget gate: 1.** The cell update is $c_t = f_t\\odot c_{t-1} + i_t\\odot\\tilde c_t$ with $f_t = \\sigma(\\cdot + b_f)$. With $b_f = 0$ the cell forgets half its content each step at the start ($\\sigma(0) = 0.5$); with $b_f = 1$, $\\sigma(1) = 0.7311$, so gradients survive more time steps early on. Gers, Schmidhuber and Cummins (2000) introduced the forget gate, and Jozefowicz, Zaremba and Sutskever (2015, *\"An Empirical Exploration of Recurrent Network Architectures\"*) found a forget bias of 1 closes much of the gap between LSTMs and GRUs. Keras does this by default (`unit_forget_bias=True`); PyTorch does not. See [LSTM](/learn/m09/lstm).",
        "**Output layer of an imbalanced classifier: the log prior.** If positives have frequency $p$, setting the sigmoid output bias to $b = \\log\\frac{p}{1-p}$ makes the first prediction $\\sigma(b) = p$. With $p = 0.01$ the measured starting loss drops from $0.6939$ (bias 0, predicting 0.5 for everything) to $0.0662$, close to the 0.0560 entropy of the labels. No early steps are wasted learning the base rate. Lin et al. (2017, *\"Focal Loss for Dense Object Detection\"*) used this with $p = 0.01$ for RetinaNet. For softmax, set $b_k = \\log\\pi_k$ for class priors $\\pi_k$.",
      ] },
      { t: "h", text: "14. Working with normalisation and residual connections" },
      { t: "p", text: "**Normalisation.** A [BatchNorm or LayerNorm](/learn/m07/normalization) layer after $Wx$ divides by the measured standard deviation, so the forward pass ignores the scale of $W$: $\\mathrm{BN}(cWx) = \\mathrm{BN}(Wx)$ for $c > 0$. So the initial scale matters much less for the forward signal — which is why badly initialised but normalised networks still train (in the simulator, LayerNorm rescues $\\sigma = 0.5$ ReLU weights). It still matters for optimisation: the gradient with respect to $W$ scales like $1/c$, so the effective learning rate depends on $\\|W\\|$, and layers *without* normalisation (embeddings, output heads) still need sensible scales." },
      { t: "p", text: "**Residual connections.** A residual block computes $h^{[l]} = h^{[l-1]} + F(h^{[l-1]})$. If $F$'s output is roughly independent of its input with variance $v$, then $\\mathrm{Var}(h^{[l]}) \\approx \\mathrm{Var}(h^{[l-1]}) + v$: the residual stream's variance grows with the number of blocks, even when each branch is perfectly initialised. So several recipes make each branch start small:" },
      { t: "list", items: [
        "**Zero-γ** (Goyal et al., 2017, *\"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour\"*): set the scale $\\gamma$ of the last BatchNorm in each residual branch to 0, so every block starts as the identity. Measured on 50 BN residual blocks: output std 4.509 with $\\gamma = 1$ versus 0.582 with $\\gamma = 0$, and the zero-$\\gamma$ network's output equals $\\mathrm{ReLU}(x)$ exactly. torchvision offers this as `zero_init_residual=True`.",
        "**Fixup** (Zhang, Dauphin and Ma, 2019, *\"Fixup Initialization: Residual Learning Without Normalization\"*): set the last layer of each branch to zero, rescale the other branch layers by a depth-dependent factor $L^{-1/(2m-2)}$ ($L$ blocks, $m$ layers per branch), and add scalar biases and multipliers; this trains deep ResNets with no normalisation layer at all.",
        "**Scaled residual projections** in Transformers: GPT-2 (Radford et al., 2019) scales residual layers' weights at initialisation by $1/\\sqrt N$, where $N$ is the number of residual layers.",
      ] },
      { t: "p", text: "All three share one idea: at the start, a deep residual network should act like a shallow one, and training \"switches on\" depth gradually. [Residual Connections](/learn/m07/residual-connections) shows what this does in a real experiment." },
      { t: "h", text: "15. Data-dependent initialisation: LSUV" },
      { t: "p", text: "Every scheme so far assumes standardised inputs and independent, zero-mean pre-activations. Real data and architectures break this. **Layer-sequential unit-variance** (LSUV, Mishkin and Matas, 2016, *\"All you need is a good init\"*) measures instead of assuming: initialise orthogonally, then, layer by layer from input to output, push a mini-batch through and divide the layer's weights by the measured std of its output until that output has variance 1. It handles any activation, layer or input. In the exercise below, an input with mean 2 and std 3 gives output variances of about 26 at every layer of a 30-layer orthogonal ReLU network with gain $\\sqrt2$ (the scheme keeps the input's raw average square $\\mathbb E[x^2] = 13$, times 2); LSUV gives exactly 1.000 at every layer. *Weight normalisation* (Salimans and Kingma, 2016) has a similar data-driven initialisation step." },
    ],
    visual: [
      { t: "p", text: "The `gradient-flow` simulator runs a real forward and backward pass through a square MLP (width 8–64, depth 2–50, a batch of 32 Gaussian inputs, zero biases) with your chosen initialisation. Its **init** row has three buttons: **Normal σ** (a fixed std, set with the log-scale **σ (Normal)** slider), **Xavier** (std $\\sqrt{2/(n+n)} = 1/\\sqrt n$) and **He** (std $\\sqrt{2/n}$); the current std is shown next to them. Top chart: each layer's output std; bottom: each layer's gradient norm. Numbers below are from the simulator's default sample (MSE loss, width 64, depth 30)." },
      { t: "sim", id: "gradient-flow", caption: "Try every initialisation with each activation; watch the top (forward std) chart first, then the gradient chart." },
      { t: "list", items: [
        "**Match init to activation.** Click the preset **ReLU, He (healthy)**: output std goes 0.87 (layer 1) → 0.70 (layer 10) → 0.42 (layer 30). Now click **Xavier**: it falls 0.62 → 0.022 → $1.3\\times10^{-5}$ — the factor $\\sqrt{1/2}$ per layer from Section 6. Select **Tanh** with **Xavier**: 0.63 → 0.21 → 0.13 (slow decay from tanh's squashing); with **He** it goes 0.73 → 0.55 → 0.56.",
        "**Linear networks are the exact case.** Select **Linear** with **Xavier**: std 1.03 → 0.90 → 0.93, flat. Switch to **He**: the same network grows to $3.1\\times10^4$, because for a linear unit $c_f = 1$ and He's factor 2 is one too many — $2^{29/2}\\approx 2.3\\times10^4$.",
        "**A fixed σ only works for one width.** With **ReLU** and **Normal σ**, set σ to about 0.177 ($= \\sqrt{2/64}$, i.e. He for width 64): layer-30 std 0.43. Now drag **width n** down to 16 without touching σ: the std collapses to about $10^{-10}$, because He for width 16 would be 0.354. A fixed std ignores fan-in; σ = 0.1 or 0.25 at width 64 gives $1.6\\times10^{-8}$ and $1.4\\times10^{4}$ — a factor of 2.5 in σ spans twelve orders of magnitude after 30 layers.",
        "**Initialisation vs normalisation.** Click **ReLU σ = 0.5 + LayerNorm**: weights that make the plain network explode to std $1.5\\times10^{13}$ now give a flat 0.57–0.59 at every layer. LayerNorm makes the forward pass ignore weight scale (Section 14).",
        "**Initialisation vs residual connections.** Select **ReLU**, **He**, and tick **residual connections**: the stream std grows from 1.3 to $4.6\\times10^6$ — residual branches add variance (Section 14). Now choose **Normal σ** with σ = 0.01 (a nearly zero branch) and keep residual on: std stays 1.0–1.1 at every layer and the gradient bars are nearly flat. This is the \"start every block near the identity\" recipe.",
        "Note the **dead units** readout for ReLU: about 44% of layer-30 units are zero for all 32 inputs under every ReLU scale (scaling weights never flips a pre-activation's sign). Initialisation scale can't fix that; it comes from depth and zero biases.",
      ] },
    ],
    scratch: [
      { t: "p", text: "A NumPy program with four experiments: (1) zero, constant and random initialisation of a one-hidden-layer tanh network trained on the 1797 `sklearn` digits, checking that constant initialisation keeps every hidden unit identical; (2) the std of pre-activations through 20 layers of width 256 for tanh and ReLU under four initialisations, against theory; (3) fan-in, fan-out and Xavier in a linear network whose width halves each layer; (4) the singular values of a 30-layer linear network's Jacobian under Gaussian and orthogonal weights." },
      { t: "code", lang: "python", title: "initialisation_numpy.py", code: INIT_NUMPY, output: INIT_NUMPY_OUT },
      { t: "h", text: "What the output shows" },
      { t: "list", items: [
        "**Symmetry.** Zero init: $W^{[1]}$ is still exactly zero after 300 steps (rank 0), loss $\\ln 10 = 2.302$. Constant 0.01: all 16 hidden units still have identical weights (spread $0$, rank 1); 31.8% accuracy is what one hidden unit can do. Random Xavier: 98.4%.",
        "**Forward scale.** Only He keeps ReLU pre-activations around size 1 through 20 layers (1.41, 1.33, 1.58, 1.69). Xavier+ReLU matches the predicted decay $2^{-9.5}$ to three digits. $\\mathcal N(0, 1)$ explodes ReLU to $10^{21}$ and pins tanh at std 15.6 — not exploding, because tanh is bounded, but every unit is stuck at $\\pm1$ with slope near 0. $\\mathcal{N}(0, 0.01^2)$ kills both.",
        "**Tanh likes slightly more than Xavier.** With Xavier the tanh pre-activation std decays (0.997 → 0.164) because tanh squashes values away from 0 ($c_f < 1$); PyTorch's tanh gain 5/3 exists to make up for this.",
        "**Fan modes.** The narrowing linear network matches the predicted ratios: fan-in keeps the forward pass (1.03) and loses $1/32$ backward, fan-out the reverse, Xavier splits the difference.",
        "**Isometry.** After 30 Gaussian layers, 81% of the Jacobian's singular values are below 0.1 although the average squared gain is 1; with orthogonal layers all are exactly 1.",
      ] },
      { t: "p", text: "Implementation notes: the hidden units are the *columns* of `W1` (inputs are rows of `X`, so `X @ W1`), so the spread is measured between columns. `np.linalg.qr` returns a matrix with orthonormal columns; any fixed sign convention is fine here (the exercise below fixes the signs to make the distribution exactly uniform)." },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**1. The initialisers, and what the defaults really are.** This program measures PyTorch's default `nn.Linear` initialisation, reproduces it with `kaiming_uniform_(a=sqrt(5))`, prints `calculate_gain`, checks every `nn.init` function against its formula, and shows fan-in for a convolution:" },
          { t: "code", lang: "python", title: "torch_init_api.py", code: INIT_TORCH_DEFAULTS, output: INIT_TORCH_DEFAULTS_OUT },
          { t: "p", text: "`mode=\"fan_out\"` uses $n_{out} = 512$, giving std $\\sqrt{2/512} = 0.0625$, and a $(32, 16, 3, 3)$ convolution kernel has fan-in $16\\cdot9 = 144$. `nn.init._calculate_fan_in_and_fan_out` is a private helper, used here only to show the numbers." },
          { t: "p", text: "**2. Training a 20-layer MLP on digits under seven initialisations.** Twenty hidden layers of width 64 (plus a linear output layer), the 1797 `sklearn` digits split 75/25, SGD with momentum 0.9, learning rate 0.003, batch 32, 20 epochs. For each initialisation it prints the activation std after hidden layers 1, 5, 10 and 20 at the start (on the training set), the training loss at epochs 0, 1, 5 and 20, and the final test accuracy. About 20 s on one CPU thread." },
          { t: "code", lang: "python", title: "init_digits_training.py", code: INIT_TORCH_TRAIN, output: INIT_TORCH_TRAIN_OUT },
          { t: "list", items: [
            "**zeros** and **N(0, 0.01²)**: the forward signal is 0 or $10^{-26}$ at layer 20; the loss never leaves $\\ln 10 \\approx 2.303$ and accuracy is chance (10.2%).",
            "**N(0, 1)**: activations reach $2.7\\times10^{14}$, the starting loss is about $10^{14}$, and the first update gives `nan`.",
            "**PyTorch default** (`kaiming_uniform_(a=√5)`): the ReLU network also stays at 2.303. Its weight variance is $1/(3n)$, a sixth of He's; the activation std falls to 0.046 by layer 5. With 20 ReLU layers this default is unusable.",
            "**Xavier** with ReLU: std $2.4\\times10^{-4}$ at layer 20; after 20 epochs the loss has only just started to fall (1.742, 20% test accuracy).",
            "**He** and **orthogonal with gain √2**: activations stay around 0.2–1 at every layer and the networks reach 94.2% and 96.9% test accuracy. **Tanh + Xavier** also trains (95.6%).",
          ] },
          { t: "callout", kind: "note", text: "These are single runs with one seed and a fixed small learning rate chosen so all well-initialised networks converge; accuracies would move a few points with other seeds or schedules. The robust result is the split — four initialisations that can't start, three that train." },
          { t: "p", text: "**3. Biases, SELU and zero-initialised residual branches.** Output bias at the log prior, LSTM forget-gate bias, a 32-layer SELU network with LeCun-normal weights, and 50 BatchNorm residual blocks with and without zero-$\\gamma$:" },
          { t: "code", lang: "python", title: "init_bias_selu_residual.py", code: INIT_TORCH_EXTRAS, output: INIT_TORCH_EXTRAS_OUT },
          { t: "list", items: [
            "PyTorch's LSTM stores two bias vectors, `bias_ih` and `bias_hh`, each laid out as [input, forget, cell, output] gates of `hidden_size` entries; their sum is the gate bias, so set the forget slice of one to 1 and leave the other at 0.",
            "SELU + LeCun keeps mean ≈ 0 and variance ≈ 1 at all 32 layers without any normalisation layer.",
            "With zero-$\\gamma$, the residual branches output exactly 0 at the start, so the first block's `fc1` gets zero gradient on the very first step; $\\gamma$ itself gets a non-zero gradient ($1.6\\times10^{-2}$), becomes non-zero after the first update, and the branch weights start learning from step 2.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras layers take `kernel_initializer` and `bias_initializer` arguments, as strings (`\"glorot_uniform\"`, `\"he_normal\"`, `\"lecun_normal\"`, `\"orthogonal\"`, `\"zeros\"`) or objects from `tf.keras.initializers`. Unlike PyTorch, Keras's default for `Dense` and `Conv2D` kernels is **Glorot uniform** with zero bias. Keras's `HeNormal` and `LecunNormal` draw from a *truncated* normal (cut at two standard deviations) with the std adjusted so the variance is still $2/n_{in}$ or $1/n_{in}$. `VarianceScaling(scale, mode, distribution)` is the general form of all of them." },
          { t: "code", lang: "python", title: "keras_initializers.py (not executed in CI)", code: INIT_KERAS, output: "Prints the largest default Dense weight (at most the Glorot bound 0.125) and, for each initialiser, the std of the 20th hidden layer's ReLU activations on 512 random inputs: tiny for glorot_uniform (variance halves each layer) and of order 1 for he_normal, orthogonal with gain √2 and VarianceScaling(2, fan_in)." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Scheme", "Var(W)", "Distributions", "Designed for", "Strengths", "Weaknesses"], rows: [
        ["Zero / constant", "0", "—", "never for weights (fine for biases, BN β)", "trivial", "no symmetry breaking: all units identical forever"],
        ["Small fixed std (e.g. 0.01)", "fixed", "normal", "historical (shallow nets)", "simple", "ignores fan-in; vanishes in deep nets, right for only one width"],
        ["Old rule / PyTorch default", "1/(3 n_in)", "U[−1/√n_in, 1/√n_in]", "shallow nets", "safe, never explodes", "shrinks ReLU variance 6× per layer; deep plain ReLU nets fail"],
        ["LeCun (1998)", "1/n_in", "normal, truncated normal, uniform", "linear, tanh near 0, SELU", "exact for linear units; required for self-normalising nets", "ReLU loses ½ per layer"],
        ["Xavier / Glorot (2010)", "2/(n_in + n_out)", "uniform ±√(6/(n_in+n_out)), normal", "tanh, sigmoid, linear", "balances forward and backward; Keras default", "too small for ReLU"],
        ["He / Kaiming (2015)", "2/n_in (or 2/n_out)", "normal, uniform ±√(6/n_in)", "ReLU (gain √(2/(1+a²)) for Leaky ReLU)", "keeps variance steady in deep ReLU nets", "assumes symmetric zero-mean pre-activations; too large for linear/tanh"],
        ["Orthogonal (Saxe et al. 2013)", "g²/n (square)", "random orthogonal × gain", "RNN recurrent weights, very deep nets", "all singular values = gain: exact norm preservation for linear maps", "QR cost at init; can't give isometry with ReLU"],
        ["LSUV (2016)", "measured", "orthogonal, rescaled on data", "any architecture", "no assumptions about activation or input statistics", "needs a data batch; repeated per-layer procedure"],
        ["Zero-init residual / Fixup", "last branch layer 0", "—", "residual networks", "blocks start as identity; deep nets start shallow", "needs residual structure; zero layer gets no gradient on step 1"],
      ] },
      { t: "p", text: "**Uniform or normal?** Same variance, so the analysis can't tell them apart. Uniform is bounded (no rare huge weights), normal has long tails, and truncated normal sits between. In practice the difference is tiny next to getting the variance right. **Fan-in or fan-out?** For ReLU networks either works (the other direction only changes by a width ratio); fan-in is PyTorch's default, while torchvision's ResNets use `kaiming_normal_(mode=\"fan_out\")` for convolutions." },
    ],
    applications: [
      { t: "list", items: [
        "**Convolutional networks.** torchvision's ResNet initialises convolutions with `kaiming_normal_(mode=\"fan_out\", nonlinearity=\"relu\")`, BatchNorm with $\\gamma = 1, \\beta = 0$, and optionally the last BN of each residual branch with $\\gamma = 0$ (`zero_init_residual=True`). He et al. used their initialisation to train a 30-layer plain network from scratch and, later, ResNets.",
        "**Language models.** GPT-2 (Radford et al., 2019) scales residual projection weights by $1/\\sqrt N$ ($N$ = number of residual layers) so the residual stream's variance doesn't grow with depth; Transformer codebases routinely use small-std normal initialisation for embeddings and linear layers.",
        "**Recurrent networks.** Orthogonal $W_{hh}$ and a forget-gate bias of 1 are standard for LSTMs; Keras applies both by default (`recurrent_initializer=\"orthogonal\"`, `unit_forget_bias=True`).",
        "**Object detection with extreme class imbalance.** RetinaNet (Lin et al., 2017) sets the classification head's bias to $-\\log((1-\\pi)/\\pi)$ with $\\pi = 0.01$, which the authors reported prevents early instability from the huge number of background anchors.",
        "**Normalisation-free networks.** Fixup trains residual networks without BatchNorm through initialisation alone, and SELU networks rely on LeCun initialisation for their self-normalising property.",
        "**Transfer learning** mostly sidesteps the question: a pre-trained backbone starts from learned weights, and only the new head needs initialising — usually at a small scale so it doesn't disturb the pre-trained features at first.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Free and essential, but only at step 0.** Initialisation costs nothing and decides whether training can start. Its guarantees are averages over random weights and assume independence that training breaks right away; it can't control scale later. That is the job of normalisation, residual connections and learning-rate control.",
        "**Activation-specific.** Each rule is derived for one activation. Swapping ReLU for GELU, SiLU or tanh without changing the gain silently changes the per-layer factor; for unusual activations, measure (or use LSUV).",
        "**Assumes standardised inputs.** Every data-independent scheme assumes $\\mathbb E[x^2] \\approx 1$ at the input. Raw inputs (pixels 0–255) multiply the first layer's variance by thousands. Standardise the data first.",
        "**Defaults are compromises.** PyTorch's default is safe for shallow networks and never explodes, but is too small for deep plain ReLU stacks; Keras's Glorot default is off for ReLU by 2× per layer. Neither matters much once BatchNorm/LayerNorm follow every layer — which is why the defaults survive.",
        "**Orthogonal and data-dependent schemes** give tighter control at the cost of extra work at initialisation and, for LSUV, dependence on the first batch.",
        "**Zero-initialised residual branches** make very deep networks start shallow and stable, but a zero layer gets no gradient on step 1 and they need a residual path.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Setting weights to zero (or any constant).** Units stay identical forever (Section 1). Always initialise weights randomly; zero is fine for biases.",
        "**Assuming PyTorch's default is He.** `nn.Linear`/`nn.Conv2d` use `kaiming_uniform_(a=√5)`, i.e. $\\mathrm{Var}(W) = 1/(3n_{in})$, a sixth of He's. In a deep plain ReLU MLP this stopped training entirely in our experiment. Call `nn.init.kaiming_normal_(w, nonlinearity=\"relu\")` explicitly.",
        "**Confusing fan-in with the tensor's first dimension.** PyTorch stores `Linear` weights as (out, in), so `weight.shape[1]` is fan-in; NumPy code that computes `x @ W` stores (in, out). For convolutions fan-in is $C_{in}k_hk_w$, not $C_{in}$ — forgetting the $k^2$ makes the weights 3× too large for $3\\times3$ kernels.",
        "**Using the wrong gain for the activation.** He with tanh or linear layers doubles the variance each layer (simulator: linear + He reaches $3\\times10^4$ at layer 30); Xavier with ReLU halves it. Match the gain to the activation that *follows* the layer.",
        "**Using Var(x) instead of E[x²] for inputs with non-zero mean.** The formula needs the average square. ReLU outputs and unnormalised data have large means, so the 'variance' version underestimates the scale.",
        "**Forgetting to standardise inputs.** A $\\sqrt{2/n_{in}}$ first layer fed raw pixels 0–255 starts with pre-activations hundreds of times too large. Scale inputs to mean 0 and variance 1 (or at least to [−1, 1]).",
        "**Overwriting initialisation after loading pre-trained weights** (calling `model.apply(init_fn)` after `load_state_dict`), or forgetting that `apply` also re-initialises layers you meant to keep frozen.",
        "**Setting both LSTM forget biases to 1 in PyTorch.** `bias_ih` and `bias_hh` are added, so setting both gives an effective forget bias of 2.",
        "**Expecting initialisation to fix instability later on.** If the loss explodes after 1000 steps, blame the learning rate, missing normalisation or the architecture — not the starting weights.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Why can't we initialise all weights to zero?", a: "Hidden units with identical incoming and outgoing weights compute identical activations, get identical gradients and identical updates, so they stay identical forever — $n$ units act like one. All-zero is even worse for tanh/ReLU: hidden activations and backpropagated errors are 0, so weight gradients are 0 and only the output bias learns. Random initialisation breaks the symmetry." },
      { t: "qa", level: "beginner", q: "What are fan-in and fan-out?", a: "Fan-in is how many inputs each unit of a layer sums; fan-out is how many units each input feeds. For a dense layer with weight shape (out, in) they are in and out. For a convolution with kernel $(C_{out}, C_{in}, k, k)$, fan-in $= C_{in}k^2$ and fan-out $= C_{out}k^2$. Initialisation variances are inversely proportional to them, because a pre-activation's variance grows linearly with the number of terms summed." },
      { t: "qa", level: "beginner", q: "What is the difference between Xavier and He initialisation, and when do you use each?", a: "Xavier: $\\mathrm{Var}(W) = 2/(n_{in}+n_{out})$, derived for activations that are roughly linear around 0 (tanh, sigmoid, linear layers). He: $\\mathrm{Var}(W) = 2/n_{in}$, derived for ReLU, where the extra factor 2 makes up for ReLU zeroing half the pre-activation. Use He for ReLU-family activations and Xavier (or LeCun) for tanh/linear/sigmoid." },
      { t: "qa", level: "intermediate", q: "Derive He initialisation, stating your assumptions.", a: "Assume weights i.i.d. with mean 0, independent of the inputs, and previous-layer pre-activations symmetric around 0. Then $z = \\sum_{k=1}^{n_{in}}W_kx_k$ has $\\mathrm{Var}(z) = n_{in}\\mathrm{Var}(W)\\mathbb E[x^2]$ (uncorrelated zero-mean terms). With $x = \\mathrm{ReLU}(z')$ and $z'$ symmetric, $\\mathbb E[x^2] = \\tfrac12\\mathrm{Var}(z')$. So $\\mathrm{Var}(z) = \\tfrac12 n_{in}\\mathrm{Var}(W)\\mathrm{Var}(z')$; setting this equal to $\\mathrm{Var}(z')$ gives $\\mathrm{Var}(W) = 2/n_{in}$, i.e. std $\\sqrt{2/n_{in}}$ or uniform bound $\\sqrt{6/n_{in}}$. The backward pass gives $2/n_{out}$; either is enough, because the other direction then changes only by the ratio of first and last widths." },
      { t: "qa", level: "intermediate", q: "Why does Xavier use 2/(n_in + n_out) rather than 1/n_in?", a: "Keeping the forward variance needs $n_{in}\\mathrm{Var}(W) = 1$; keeping the backward variance needs $n_{out}\\mathrm{Var}(W) = 1$. They clash unless the layer is square. Glorot and Bengio averaged them: $\\tfrac{n_{in}+n_{out}}{2}\\mathrm{Var}(W) = 1$. In a network that halves its width each layer, this splits the error: forward variance grows 4/3 per layer and backward shrinks 2/3, instead of one direction being exact and the other losing 2×." },
      { t: "qa", level: "intermediate", q: "What does PyTorch's nn.Linear use by default, and is it a problem?", a: "`kaiming_uniform_(weight, a=math.sqrt(5))`, which works out to $\\mathcal U[-1/\\sqrt{n_{in}}, 1/\\sqrt{n_{in}}]$, variance $1/(3n_{in})$; the bias comes from the same range. It is the old pre-2010 rule, a sixth of He's variance. For shallow nets or nets with normalisation after each layer it is harmless; in a deep plain ReLU MLP each layer multiplies the forward variance by about 1/6 and training can fail — a 20-layer digits MLP stayed at chance, while He reached 94%." },
      { t: "qa", level: "intermediate", q: "How should the output bias of a binary classifier be initialised when only 1% of examples are positive?", a: "To the log-odds of the prior, $b = \\log(0.01/0.99) = -4.595$, with small output weights. The first prediction is then $\\sigma(b) = 0.01$, the starting loss is close to the labels' entropy (0.066 vs 0.693 with $b = 0$ in our measurement), and the network doesn't spend its first steps — with large, unstable gradients from the many easy negatives — learning the base rate." },
      { t: "qa", level: "advanced", q: "What is dynamical isometry, and why does orthogonal initialisation help achieve it?", a: "A network has dynamical isometry when all singular values of its input–output Jacobian are close to 1, so every gradient direction passes through with about unit gain — not just on average. Variance-preserving Gaussian initialisation only controls the mean squared singular value; in deep products the spread widens (measured: 30 Gaussian layers give a largest singular value of 5.8 and a median of $3\\times10^{-5}$). Orthogonal matrices have all singular values 1 and their products stay orthogonal, so deep linear networks with orthogonal weights are exactly isometric. With nonlinearities, isometry also needs activation slopes near 1: Pennington et al. (2017) showed it can be approached with tanh-like units and orthogonal weights, but not with ReLU." },
      { t: "qa", level: "advanced", q: "Why do zero-initialised residual branches (zero-γ, Fixup) help, and what do they cost?", a: "In $h^{[l]} = h^{[l-1]} + F(h^{[l-1]})$ each branch adds variance to the residual stream, so variance grows with depth even when each branch is well-initialised (measured: std 4.5 after 50 BN blocks). Setting the last layer or BN scale of each branch to 0 makes every block the identity at the start: signal and gradient pass through unchanged and the network acts shallow, so large learning rates are safe from the start. The cost: the branch upstream of the zeroed parameter gets no gradient on the first step (only the zeroed parameter itself does), and it only works where a residual structure exists." },
      { t: "qa", level: "advanced", q: "Explain how SELU networks normalise themselves and which initialisation they require.", a: "SELU is $\\lambda z$ for $z > 0$ and $\\lambda\\alpha(e^z - 1)$ otherwise, with $\\lambda \\approx 1.0507$, $\\alpha \\approx 1.6733$ chosen so that a layer's (mean, variance) is pulled toward (0, 1), provided the weights have zero mean and $n_{in}\\mathrm{Var}(W) = 1$ — LeCun-normal initialisation. Activations that drift are pulled back layer after layer. Measured: 32 SELU layers keep variance between 0.97 and 1.02. He initialisation would break this (variance factor 2). Dropout must also be swapped for 'alpha dropout', which keeps the statistics." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A ReLU convolution has 32 input channels and 3×3 kernels. What is the He-normal standard deviation (fan-in mode)? (4 decimals)", answer: 0.0833, tolerance: 0.0005, explain: "Fan-in $= 32\\cdot3\\cdot3 = 288$; $\\sqrt{2/288} = \\sqrt{1/144} = 1/12 = 0.0833$." },
        { kind: "numeric", q: "Xavier-uniform for a layer with $n_{in} = n_{out} = 300$: what is the bound $a$ of $\\mathcal U[-a, a]$?", answer: 0.1, tolerance: 0.001, explain: "$a = \\sqrt{6/(300+300)} = \\sqrt{0.01} = 0.1$." },
        { kind: "numeric", q: "What is the standard deviation of the default PyTorch weights of `nn.Linear(300, 50)`? (4 decimals)", answer: 0.0333, tolerance: 0.0005, explain: "$\\mathcal U[-1/\\sqrt{300}, 1/\\sqrt{300}]$ has variance $1/(3\\cdot300) = 1/900$, std $1/30 = 0.0333$." },
        { kind: "numeric", q: "5% of training examples are positive. To what value should the sigmoid output bias be initialised? (3 decimals)", answer: -2.944, tolerance: 0.005, explain: "$\\log(0.05/0.95) = -2.944$; then $\\sigma(b) = 0.05$." },
        { kind: "numeric", q: "In a deep ReLU network with square layers using PyTorch's default ($\\mathrm{Var}(W) = 1/(3n)$), by what factor does the *standard deviation* of the pre-activations shrink per layer? (3 decimals)", answer: 0.408, tolerance: 0.002, explain: "Variance factor $n\\cdot\\frac{1}{3n}\\cdot\\frac12 = \\frac16$; std factor $\\sqrt{1/6} = 0.408$." },
        { q: "Two hidden units start with identical incoming and outgoing weights. After 1000 steps of SGD with momentum they will…", options: ["…differ, because momentum accumulates different histories", "…still be identical, because they receive identical gradients at every step", "…differ, because the mini-batches are random", "…differ only if the learning rate is large"], answer: 1, explain: "Identical units see the same mini-batch and produce identical errors, so their gradients — and any per-parameter optimizer state built from them — are identical. Only something that treats them differently (random init, dropout) breaks the symmetry." },
        { q: "Which initialisation is required for SELU's self-normalising property?", options: ["He normal, Var(W) = 2/n_in", "Xavier uniform", "LeCun normal, Var(W) = 1/n_in", "Orthogonal with gain √2"], answer: 2, explain: "SELU's pull toward (0, 1) is derived for zero-mean weights with $n_{in}\\mathrm{Var}(W) = 1$; PyTorch's docs likewise recommend the linear gain for self-normalising networks." },
        { q: "Why does He initialisation use 2/n_in instead of 1/n_in?", options: ["To compensate for ReLU setting the negative half of the pre-activation to zero, which halves its second moment", "Because ReLU's derivative is 2", "To average fan-in and fan-out", "To make the weights' mean positive"], answer: 0, explain: "$\\mathbb E[\\mathrm{ReLU}(z)^2] = \\tfrac12\\mathrm{Var}(z)$ for symmetric $z$; doubling the weight variance brings the per-layer factor back to 1." },
        { q: "A network has a BatchNorm layer after every linear layer. Compared with the same network without normalisation, initialisation scale…", options: ["…matters more, because BN amplifies errors", "…no longer affects the forward activations of normalised layers, but still affects the gradient scale and the unnormalised layers", "…is irrelevant for every layer", "…must be zero"], answer: 1, explain: "$\\mathrm{BN}(cWx) = \\mathrm{BN}(Wx)$, so the forward pass ignores scale; the gradient w.r.t. $W$ scales like $1/c$, and embeddings/output heads are not normalised." },
      ] },
      { t: "exercise", title: "Initialise a convolutional layer and predict the damage of a wrong default", prompt: "A ReLU CNN has a stack of $3\\times3$ convolutions with $C_{in} = 64$ and $C_{out} = 128$ channels (no normalisation). (a) Compute fan-in and fan-out. (b) Compute the He-normal std (fan-in mode) and the He-uniform bound. (c) If each of 10 such layers instead used PyTorch's default ($\\mathrm{Var}(W) = 1/(3\\,\\text{fan}_{in})$), by what factor would the variance and the standard deviation of the pre-activations shrink across the 10 layers, compared with He?", hint: "The per-layer variance factor is $\\text{fan}_{in}\\cdot\\mathrm{Var}(W)\\cdot\\tfrac12$ for ReLU.", solution: [
        { t: "steps", items: [
          { title: "(a) Fan-in and fan-out", tex: String.raw`\text{fan}_{in} = 64\cdot3\cdot3 = 576,\qquad \text{fan}_{out} = 128\cdot3\cdot3 = 1152` },
          { title: "(b) He normal and He uniform", tex: String.raw`\sigma = \sqrt{2/576} = 0.05893,\qquad a = \sqrt{6/576} = 0.10206` },
          { title: "(c) Per-layer factor with the default, then over 10 layers", tex: String.raw`576\cdot\frac{1}{3\cdot576}\cdot\frac12 = \frac16,\qquad \Big(\frac16\Big)^{10} = 1.654\times10^{-8}\ \text{(variance)},\qquad 6^{-5} = 1.286\times10^{-4}\ \text{(std)}` },
        ] },
        { t: "p", text: "He keeps the per-layer factor at $576\\cdot\\frac{2}{576}\\cdot\\frac12 = 1$. The default's factor $\\frac16$ compounds: after ten layers the signal is about $10^{-4}$ of its He size — enough to stall training unless BatchNorm follows each convolution." },
      ] },
      { t: "exercise", title: "Implement orthogonal initialisation and LSUV", prompt: "In NumPy, write (1) `orthogonal(n_out, n_in, gain)` that returns a random (semi-)orthogonal matrix via the QR decomposition of a Gaussian matrix, and (2) `lsuv(Ws, X, act)` that goes through the layers in order and rescales each weight matrix until the layer's pre-activation on batch `X` has variance 1. Test on a 30-layer ReLU network of width 128 fed *unstandardised* data (mean 2, std 3): compare orthogonal with gain $\\sqrt2$ against orthogonal + LSUV, and check that a $64\\times128$ semi-orthogonal matrix satisfies $WW^\\top = I$.", hint: "Multiply the columns of Q by the signs of R's diagonal so the distribution is uniform over orthogonal matrices. For LSUV, divide W by the measured std of the layer output, and repeat a few times, because with biases or nonlinearities the output variance is only roughly quadratic in the scale.", solution: [
        { t: "code", lang: "python", title: "orthogonal_lsuv.py", code: INIT_LSUV, output: INIT_LSUV_OUT },
        { t: "p", text: "Orthogonal with gain $\\sqrt2$ is perfectly stable across depth (variance 25.9 at layer 1, 25.3 at layer 30, wobbling with the data) — but stable at the *wrong level*, because the input's average square is $2^2 + 3^2 = 13$ and the scheme keeps it (×2 for the gain, ×½ for ReLU). LSUV measures instead of assuming and brings every layer to exactly variance 1.000. The semi-orthogonal check is at machine precision ($8.9\\times10^{-16}$)." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Initialise weights **randomly** (symmetry breaking): identical units get identical gradients forever; all-zero weights don't learn at all.",
        "Forward: $\\mathrm{Var}(z) = n_{in}\\mathrm{Var}(W)\\mathbb E[x^2]$ (i.i.d. zero-mean weights independent of inputs). Backward: $\\mathrm{Var}(\\delta^{[l-1]}) = n_{out}\\mathrm{Var}(W)\\mathbb E[f'^2]\\mathrm{Var}(\\delta^{[l]})$.",
        "Activation factor $c_f = \\mathbb E[f(z)^2]/\\mathrm{Var}(z)$: 1 for linear/tanh near 0, ½ for ReLU, $(1+a^2)/2$ for Leaky ReLU.",
        "LeCun $1/n_{in}$ (linear, SELU); Xavier $2/(n_{in}+n_{out})$, uniform bound $\\sqrt{6/(n_{in}+n_{out})}$ (tanh); He $2/n_{in}$, uniform bound $\\sqrt{6/n_{in}}$ (ReLU). Uniform $\\mathcal U[-a,a]$ has variance $a^2/3$.",
        "Gain: $\\sigma = g/\\sqrt{\\text{fan}}$; PyTorch gains: linear/sigmoid 1, tanh 5/3, ReLU √2, Leaky √(2/(1+a²)), SELU 3/4.",
        "Convolutions: fan-in $= C_{in}k_hk_w$, fan-out $= C_{out}k_hk_w$.",
        "PyTorch's default for Linear/Conv is `kaiming_uniform_(a=√5)` = $\\mathcal U[\\pm1/\\sqrt{n_{in}}]$, Var $= 1/(3n_{in})$ — a sixth of He. A 20-layer ReLU MLP on digits didn't train with it (2.303 loss) but reached 94% with He and 97% with orthogonal; Keras defaults to Glorot uniform.",
        "Orthogonal init preserves norms exactly in linear nets (dynamical isometry); LSUV rescales layers on real data to unit variance.",
        "Biases: 0; LSTM forget gate 1; imbalanced output $\\log(p/(1-p))$.",
        "With normalisation the forward pass ignores weight scale; with residual connections, start each branch near zero (zero-γ, Fixup, $1/\\sqrt N$ scaling) so the deep network starts shallow.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Batch, Layer and Group Normalization](/learn/m07/normalization) — keeps activations standardised throughout training, not just at step 0, and makes the forward pass ignore weight scale.",
        "[Residual and Skip Connections](/learn/m07/residual-connections) — why identity shortcuts let 100+ layer networks train, and why their branches start near zero (with a real plain-vs-residual experiment).",
        "[Gradient Descent and Optimizers](/learn/m06/gradient-descent) — once the starting scale is right, the learning rate and optimizer decide how it changes.",
        "[Classic CNN Architectures](/learn/m08/cnn-architectures) — how VGG, ResNet and later designs combined He initialisation, BatchNorm and shortcuts.",
        "[LSTM](/learn/m09/lstm) — where orthogonal recurrent weights and the forget-gate bias of 1 come from in practice.",
      ] },
    ],
  },
};

export default lesson;
