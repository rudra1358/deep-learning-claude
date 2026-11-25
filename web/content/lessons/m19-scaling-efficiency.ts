import type { Lesson } from "@/lib/types";

const SCALING = String.raw`import numpy as np

# ---------------------------------------------------------------------------------------------
# Part 1. Measure your own scaling curve: test loss of models with N parameters.
# Model family: random-feature networks  f(x) = relu(x V) w  where V (d x N) is random and fixed
# and only the N output weights w are fitted (by ridge-regularised least squares). N = model size.
# Task: regress a fixed "teacher" function of d = 8 inputs, with plenty of training data so that
# model size (not data) is the bottleneck.
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
d, n_train, n_test = 8, 8000, 4000
T1 = rng.normal(0, 1, (d, 64)) / np.sqrt(d)                 # teacher: a random 2-layer tanh network
T2 = rng.normal(0, 1, 64) / np.sqrt(64)
teacher = lambda X: np.tanh(X @ T1 * 2) @ T2
X_tr, X_te = rng.normal(0, 1, (n_train, d)), rng.normal(0, 1, (n_test, d))
y_tr, y_te = teacher(X_tr), teacher(X_te)


def test_loss(N, seed):
    r = np.random.default_rng(seed)
    V = r.normal(0, 1, (d + 1, N)) / np.sqrt(d)             # random first layer (+ bias row)
    F = lambda X: np.maximum(np.c_[X, np.ones(len(X))] @ V, 0)
    Phi = F(X_tr)
    w = np.linalg.solve(Phi.T @ Phi + 1e-6 * n_train * np.eye(N), Phi.T @ y_tr)
    return np.mean((F(X_te) @ w - y_te) ** 2)


sizes = np.array([16, 32, 64, 128, 256, 512])
loss = np.array([np.mean([test_loss(N, s) for s in range(3)]) for N in sizes])   # 3 seeds per size
for N, L in zip(sizes, loss):
    print(f"N = {N:4d} parameters   test MSE = {L:.5f}")

# Fit L(N) = a * N^(-alpha) by linear regression in log-log space: log L = log a - alpha log N
slope, intercept = np.polyfit(np.log(sizes), np.log(loss), 1)
alpha, a = -slope, np.exp(intercept)
pred = a * sizes ** (-alpha)
r2 = 1 - np.sum((np.log(loss) - np.log(pred)) ** 2) / np.sum((np.log(loss) - np.log(loss).mean()) ** 2)
print(f"fit: L(N) = {a:.3f} * N^(-{alpha:.3f})   R^2 (log space) = {r2:.4f}")

# The real test of a scaling law: predict a size you have NOT trained, then train it.
N_new = 1024
print(f"extrapolated to N = {N_new}: predicted {a * N_new ** (-alpha):.5f}, "
      f"measured {np.mean([test_loss(N_new, s) for s in range(3)]):.5f}")

# ---------------------------------------------------------------------------------------------
# Part 2. Compute-optimal allocation (Hoffmann et al., 2022): C ~= 6 N D and D_opt ~= 20 N_opt.
# ---------------------------------------------------------------------------------------------
print("\ncompute budget C (FLOPs)   N_opt (params)   D_opt (tokens)   check 6*N*D")
for C in [1e18, 1e21, 5.76e23]:
    N_opt = np.sqrt(C / (6 * 20))                            # C = 6 N (20 N) = 120 N^2
    D_opt = 20 * N_opt
    print(f"   {C:9.2e}             {N_opt:10.3e}       {D_opt:10.3e}      {6 * N_opt * D_opt:9.2e}")

# Chinchilla's fitted parametric loss (Approach 3 of the paper): L = E + A/N^alpha + B/D^beta
E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
Lhat = lambda N, D: E + A / N ** al + B / D ** be
C = 6 * 70e9 * 1.4e12                                        # Chinchilla's own budget: 70B params, 1.4T tokens
print(f"\nsame budget C = {C:.2e}: predicted loss with 70B params / 1.4T tokens = {Lhat(70e9, 1.4e12):.3f}")
print(f"                          predicted loss with 280B params / {C / (6 * 280e9):.2e} tokens = {Lhat(280e9, C / (6 * 280e9)):.3f}")
Ns = np.logspace(9, 12, 3001)                                # minimise the fitted loss along C = 6 N D
best = Ns[np.argmin(Lhat(Ns, C / (6 * Ns)))]
print(f"argmin of the fitted loss on this budget: N = {best:.3e}, D = {C / (6 * best):.3e}, tokens/param = {C / (6 * best) / best:.1f}")`;
const SCALING_OUT = String.raw`N =   16 parameters   test MSE = 0.15242
N =   32 parameters   test MSE = 0.11739
N =   64 parameters   test MSE = 0.07677
N =  128 parameters   test MSE = 0.06550
N =  256 parameters   test MSE = 0.05391
N =  512 parameters   test MSE = 0.03741
fit: L(N) = 0.441 * N^(-0.392)   R^2 (log space) = 0.9837
extrapolated to N = 1024: predicted 0.02905, measured 0.02676

compute budget C (FLOPs)   N_opt (params)   D_opt (tokens)   check 6*N*D
    1.00e+18              9.129e+07        1.826e+09       1.00e+18
    1.00e+21              2.887e+09        5.774e+10       1.00e+21
    5.76e+23              6.928e+10        1.386e+12       5.76e+23

same budget C = 5.88e+23: predicted loss with 70B params / 1.4T tokens = 1.937
                          predicted loss with 280B params / 3.50e+11 tokens = 1.983
argmin of the fitted loss on this budget: N = 3.251e+10, D = 3.015e+12, tokens/param = 92.7`;
const LORA = String.raw`import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
d_in, d_out, n = 64, 32, 4000

# A "pretrained" layer W0 and a new task whose ideal weights differ from W0 by a LOW-RANK update
# (the LoRA hypothesis, Hu et al., 2021: the change needed for adaptation has low intrinsic rank).
W0 = rng.normal(0, 1 / np.sqrt(d_in), (d_out, d_in))
U, V = rng.normal(0, 1, (d_out, 2)), rng.normal(0, 1, (2, d_in))
W_task = W0 + 0.15 * U @ V / np.sqrt(d_in)                     # rank-2 change
X = rng.normal(0, 1, (n, d_in))
Y = X @ W_task.T + 0.05 * rng.normal(0, 1, (n, d_out))
X_tr, Y_tr, X_te, Y_te = X[:3000], Y[:3000], X[3000:], Y[3000:]
mse = lambda W, Xs, Ys: np.mean((Xs @ W.T - Ys) ** 2)
print(f"frozen base W0 (no adaptation):       test MSE {mse(W0, X_te, Y_te):.5f}")
W_full = np.linalg.lstsq(X_tr, Y_tr, rcond=None)[0].T          # full fine-tuning optimum
print(f"full fine-tuning ({W0.size} params):     test MSE {mse(W_full, X_te, Y_te):.5f}")


def train_lora(W_base, r, alpha, steps=500, lr=0.03, seed=0):
    """h = W_base x + (alpha / r) * B A x.  W_base frozen; A: r x d_in (Gaussian), B: d_out x r (zeros)."""
    g = np.random.default_rng(seed)
    A = g.normal(0, 1 / np.sqrt(d_in), (r, d_in))
    B = np.zeros((d_out, r))
    s = alpha / r
    state = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in (("A", A), ("B", B))}
    for t in range(1, steps + 1):
        XA = X_tr @ A.T                                          # (n, r)  = A x for every example
        R = X_tr @ W_base.T + s * XA @ B.T - Y_tr                # residual (n, d_out)
        G = 2 * R / R.size                                       # dL/dh for the mean squared error
        grads = {"B": s * G.T @ XA,                              # dL/dB = s * G^T (A x)
                 "A": s * (G @ B).T @ X_tr}                      # dL/dA = s * B^T G^T x
        for k, P in (("A", A), ("B", B)):                        # Adam
            m, v = state[k]
            m[:] = 0.9 * m + 0.1 * grads[k]
            v[:] = 0.999 * v + 0.001 * grads[k] ** 2
            P -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return A, B, s


# zero-initialised B means the adapted model starts EXACTLY at the pretrained model
A0 = np.random.default_rng(0).normal(0, 1, (4, d_in))
print(f"output change at initialisation (B = 0): {np.abs((4 / 4) * X_te @ (np.zeros((d_out, 4)) @ A0).T).max():.1f}")

print("\nrank r  trainable params  % of full   test MSE")
for r in (1, 2, 4, 8):
    A, B, s = train_lora(W0, r, alpha=r)
    W_merged = W0 + s * B @ A                                    # merge for inference: no extra latency
    print(f"  {r:2d}       {A.size + B.size:5d}          {100 * (A.size + B.size) / W0.size:5.1f}%    {mse(W_merged, X_te, Y_te):.5f}")
A, B, s = train_lora(W0, 4, alpha=4)
h_lora = X_te @ W0.T + s * (X_te @ A.T) @ B.T
print(f"max |unmerged - merged| output difference: {np.abs(h_lora - X_te @ (W0 + s * B @ A).T).max():.1e}")

# ------------------------------- 4-bit quantisation of the frozen base -------------------------------
# NF4-style codebook (Dettmers et al., 2023): levels at quantiles of N(0, 1), with an exact zero,
# normalised to [-1, 1]; 8 positive and 7 negative levels = 16 levels = 4 bits.
offset = 0.9677
pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
nf4 = np.sort(np.r_[pos, 0.0, neg])
nf4 /= np.abs(nf4).max()
int4 = np.arange(-7, 8) / 7.0                                     # uniform symmetric 4-bit levels
print("\nNF4 levels:", np.round(nf4, 4).tolist())


def quantize(W, levels, block=64):
    """Block-wise absmax quantisation: each block of 64 weights is scaled by its max |w| into [-1, 1]."""
    flat = W.reshape(-1, block)
    scale = np.abs(flat).max(axis=1, keepdims=True)               # one constant per block
    idx = np.abs(flat[:, :, None] / scale[:, :, None] - levels[None, None, :]).argmin(axis=2)   # 4-bit codes
    return (levels[idx] * scale).reshape(W.shape)


for name, lv in (("int4 (uniform)", int4), ("NF4", nf4)):
    Wq = quantize(W0, lv)
    print(f"{name:<15} weight MSE {np.mean((Wq - W0) ** 2):.3e}   base test MSE {mse(Wq, X_te, Y_te):.5f}")

# QLoRA: freeze the 4-bit base, train LoRA in full precision on top of it
Wq = quantize(W0, nf4)
for r in (4, 16):
    A, B, s = train_lora(Wq, r, alpha=r)
    print(f"QLoRA (NF4 base + LoRA r={r:2d}): test MSE {mse(Wq + s * B @ A, X_te, Y_te):.5f}")
print(f"memory for the base: fp16 {W0.size * 16 / 8:.0f} bytes, NF4 {W0.size * 4 / 8 + W0.size / 64 * 4:.0f} bytes "
      f"(4-bit codes + one fp32 constant per 64 weights = {4 + 32 / 64} bits/param)")`;
const LORA_OUT = String.raw`frozen base W0 (no adaptation):       test MSE 0.04803
full fine-tuning (2048 params):     test MSE 0.00261
output change at initialisation (B = 0): 0.0

rank r  trainable params  % of full   test MSE
   1          96            4.7%    0.02002
   2         192            9.4%    0.00256
   4         384           18.8%    0.00260
   8         768           37.5%    0.00257
max |unmerged - merged| output difference: 4.0e-15

NF4 levels: [-1.0, -0.6962, -0.5251, -0.3949, -0.2845, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.723, 1.0]
int4 (uniform)  weight MSE 1.874e-04   base test MSE 0.06033
NF4             weight MSE 1.293e-04   base test MSE 0.05475
QLoRA (NF4 base + LoRA r= 4): test MSE 0.00879
QLoRA (NF4 base + LoRA r=16): test MSE 0.00440
memory for the base: fp16 4096 bytes, NF4 1152 bytes (4-bit codes + one fp32 constant per 64 weights = 4.5 bits/param)`;
const PEFT = String.raw`import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
X_tr, X_te, y_tr, y_te = map(torch.tensor, train_test_split(X, y, test_size=0.2, stratify=y, random_state=0))
# New "domain": the same digits mirrored along the diagonal (image transposed). Only 200 labelled
# target examples are available for adaptation, as in a typical fine-tuning scenario.
transpose = lambda t: t.view(-1, 8, 8).transpose(1, 2).reshape(-1, 64)
T_tr, T_te = transpose(X_tr[:200]), transpose(X_te)
yT_tr, yT_te = y_tr[:200], y_te


class LoRALinear(nn.Module):
    """y = x W0^T + b0 + (alpha / r) * x A^T B^T, with W0, b0 frozen (Hu et al., 2021)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))       # zero init: starts at the base model
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()

    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A


class Adapter(nn.Module):
    """Bottleneck adapter (Houlsby et al., 2019): h + up(relu(down(h))), up initialised to zero."""

    def __init__(self, d, m=4):
        super().__init__()
        self.down, self.up = nn.Linear(d, m), nn.Linear(m, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.relu(self.down(h)))


class Scale(nn.Module):
    """IA3-style learned rescaling of activations (Liu et al., 2022), initialised to ones."""

    def __init__(self, d):
        super().__init__()
        self.s = nn.Parameter(torch.ones(d))

    def forward(self, h):
        return h * self.s


def make_base():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))


def fit(model, params, Xs, ys, epochs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xs), generator=g).split(32):
            loss = F.cross_entropy(model(Xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model, Xs, ys):
    return (model(Xs).argmax(1) == ys).float().mean().item()


torch.manual_seed(0)
pretrained = make_base()
fit(pretrained, pretrained.parameters(), X_tr, y_tr, 40, 1e-3, 0)
print(f"pretrained model: source test acc {acc(pretrained, X_te, y_te):.3f}, "
      f"target (transposed) test acc before adaptation {acc(pretrained, T_te, yT_te):.3f}")
state = {k: v.clone() for k, v in pretrained.state_dict().items()}
n_total = sum(p.numel() for p in pretrained.parameters())


def method(name, seed):
    torch.manual_seed(seed)
    m = make_base()
    m.load_state_dict(state)
    for p in m.parameters():
        p.requires_grad_(False)
    if name == "full fine-tuning":
        for p in m.parameters():
            p.requires_grad_(True)
    elif name == "head only (linear probe)":
        for p in m[4].parameters():
            p.requires_grad_(True)
    elif name == "BitFit (biases only)":
        for lin in (m[0], m[2], m[4]):
            lin.bias.requires_grad_(True)
    elif name.startswith("LoRA"):
        r = int(name.split("r=")[1].rstrip(")"))
        for i in (0, 2, 4):
            m[i] = LoRALinear(m[i], r=r, alpha=8)
    elif name == "adapters (m=4)":
        m = nn.Sequential(m[0], m[1], Adapter(128), m[2], m[3], Adapter(128), m[4])
    elif name == "IA3-style scaling":
        m = nn.Sequential(m[0], m[1], Scale(128), m[2], m[3], Scale(128), m[4])
    params = [p for p in m.parameters() if p.requires_grad]
    lr = 1e-3 if name == "full fine-tuning" else 1e-2
    fit(m, params, T_tr, yT_tr, 30, lr, seed)
    return m, sum(p.numel() for p in params), acc(m, T_te, yT_te), acc(m, X_te, y_te)


names = ["head only (linear probe)", "BitFit (biases only)", "IA3-style scaling", "adapters (m=4)",
         "LoRA (r=1)", "LoRA (r=2)", "LoRA (r=8)", "full fine-tuning"]
print(f"\n{'method':<26}{'trainable':>10}{'% of model':>12}{'target acc':>14}{'source acc':>12}")
for name in names:
    runs = [method(name, s) for s in range(3)]
    k = runs[0][1]
    ta = np.array([r[2] for r in runs])
    sa = np.array([r[3] for r in runs])
    print(f"{name:<26}{k:>10}{100 * k / n_total:>11.2f}%{ta.mean():>9.3f}+-{ta.std(ddof=1):.3f}{sa.mean():>10.3f}")

# merging: a LoRA layer can be folded into one ordinary weight matrix for inference
m, *_ = method("LoRA (r=8)", 0)
lin = nn.Linear(64, 128)
with torch.no_grad():
    lin.weight.copy_(m[0].merged_weight())
    lin.bias.copy_(m[0].base.bias)
    print(f"\nmerged vs unmerged first layer, max |diff| = {(lin(T_te) - m[0](T_te)).abs().max():.1e}")`;
const PEFT_OUT = String.raw`pretrained model: source test acc 0.967, target (transposed) test acc before adaptation 0.111

method                     trainable  % of model    target acc  source acc
head only (linear probe)        1290       4.94%    0.785+-0.008     0.272
BitFit (biases only)             266       1.02%    0.326+-0.006     0.220
IA3-style scaling                256       0.98%    0.435+-0.010     0.232
adapters (m=4)                  2312       8.85%    0.841+-0.018     0.651
LoRA (r=1)                       586       2.24%    0.795+-0.017     0.362
LoRA (r=2)                      1172       4.49%    0.852+-0.020     0.407
LoRA (r=8)                      4688      17.95%    0.871+-0.025     0.457
full fine-tuning               26122     100.00%    0.856+-0.007     0.577

merged vs unmerged first layer, max |diff| = 1.2e-06`;

const KERAS = String.raw`import tensorflow as tf


class LoRADense(tf.keras.layers.Layer):
    """Wraps a trained Dense layer: y = base(x) + (alpha / r) * x A B, with base frozen and B zero-initialised."""

    def __init__(self, base: tf.keras.layers.Dense, r=4, alpha=8, **kw):
        super().__init__(**kw)
        self.base, self.r, self.scale = base, r, alpha / r
        self.base.trainable = False

    def build(self, input_shape):
        d_in, d_out = int(input_shape[-1]), self.base.units
        self.A = self.add_weight(name="A", shape=(d_in, self.r), initializer=tf.keras.initializers.RandomNormal(stddev=d_in ** -0.5))
        self.B = self.add_weight(name="B", shape=(self.r, d_out), initializer="zeros")

    def call(self, x):
        return self.base(x) + self.scale * tf.matmul(tf.matmul(x, self.A), self.B)


# usage: wrap the Dense layers of a trained model, then compile and fit as usual;
# only A and B appear in model.trainable_variables.
# new_layer = LoRADense(trained_model.layers[1], r=4, alpha=8)`;

const lesson: Lesson = {
  id: "m19/scaling-efficiency",
  title: "Scaling Laws, LoRA, QLoRA and PEFT",
  summary:
    "How loss falls as a power law of parameters, data and compute (Kaplan et al., 2020), how to spend a compute budget optimally (Hoffmann et al., 2022: C ≈ 6ND, about 20 tokens per parameter), fitting a power law to your own measured models and testing its extrapolation; then parameter-efficient fine-tuning: LoRA derived and implemented by hand and in PyTorch, QLoRA's 4-bit NF4 base simulated in NumPy, and adapters, prefix/prompt tuning, BitFit and IA³ compared on a real adaptation task.",
  minutes: 100,
  level: "research",
  prerequisites: ["m12/transfer-learning", "m02/eigen-svd", "m11/transformer-architecture", "m19/reading-papers"],
  next: ["m20/llm-pretraining", "m18/model-optimization", "m20/llm-inference"],
  terms: ["scaling-law", "power-law", "compute-optimal", "chinchilla-scaling", "peft", "lora", "qlora", "nf4", "adapter", "prefix-tuning", "prompt-tuning", "bitfit", "ia3"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Two questions dominate modern deep learning practice. **How big should the model be, and how much data should it see, for a given compute budget?** And once a large model exists, **how can it be adapted to a new task cheaply?** The first is answered empirically by **scaling laws**: Kaplan et al. (2020, \"Scaling Laws for Neural Language Models\") found that the test loss of Transformer language models falls as a smooth **power law** in the number of parameters $N$, the number of training tokens $D$ and the training compute $C$, over many orders of magnitude. Hoffmann et al. (2022, \"Training Compute-Optimal Large Language Models\" — the *Chinchilla* paper) revisited the allocation question and concluded that parameters and tokens should grow in roughly equal proportion with compute, about 20 training tokens per parameter, with training compute $C \\approx 6ND$ FLOPs.",
      },
      {
        t: "p",
        text: "The second is answered by **parameter-efficient fine-tuning (PEFT)**: freeze the pretrained weights and train a small number of new or selected parameters. **LoRA** (Hu et al., 2021, \"LoRA: Low-Rank Adaptation of Large Language Models\") adds a trainable low-rank update $\\Delta W = BA$ next to each frozen weight matrix; the paper reports, for GPT-3 175B, reducing the number of trainable parameters by 10,000 times and GPU memory by 3 times compared with full fine-tuning, without adding inference latency (the update can be merged into $W$). **QLoRA** (Dettmers et al., 2023, \"QLoRA: Efficient Finetuning of Quantized LLMs\") stores the frozen base in a 4-bit **NormalFloat (NF4)** format and backpropagates through it into LoRA adapters, which made fine-tuning a 65B-parameter model possible on a single 48 GB GPU.",
      },
      {
        t: "p",
        text: "This lesson owns the scaling-law mathematics used later in [LLM pretraining](/learn/m20/llm-pretraining). You will measure a scaling curve on models you train yourself, fit the power law by log–log regression, test whether it extrapolates, compute compute-optimal model sizes, implement LoRA from scratch and in PyTorch, simulate NF4 quantisation, and compare PEFT methods by trainable parameters and accuracy on a real adaptation task.",
      },
      {
        t: "callout",
        kind: "key",
        text: "Scaling: $L(N) \\approx (N_c/N)^{\\alpha_N}$; compute $C \\approx 6ND$; compute-optimal $D_{opt} \\approx 20 N_{opt}$, so $N_{opt} \\approx \\sqrt{C/120}$. LoRA: $h = W_0x + \\frac{\\alpha}{r}BAx$ with $W_0$ frozen, $A \\in \\mathbb R^{r\\times k}$ random, $B \\in \\mathbb R^{d\\times r}$ zero-initialised; trainable parameters $r(d + k)$ instead of $dk$.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Transfer learning](/learn/m12/transfer-learning) — pretraining then fine-tuning, linear probes, catastrophic forgetting.",
          "[Eigenvalues and SVD](/learn/m02/eigen-svd) — matrix rank and low-rank approximation (Eckart–Young), the mathematical basis of LoRA.",
          "[Transformer architecture](/learn/m11/transformer-architecture) — the weight matrices ($W_Q, W_K, W_V, W_O$, FFN) that LoRA adapts, and where parameters live.",
          "[Reading and reproducing papers](/learn/m19/reading-papers) — log–log fits, seeds and fair comparisons.",
          "[Training basics](/learn/m03/training-basics) and [Adam](/learn/m06/momentum-adaptive) — the optimiser state that dominates fine-tuning memory.",
          "[Logarithms and derivatives](/learn/m02/derivatives-chain-rule) — taking logs turns a power law into a straight line.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A student, a library and a fixed number of study hours",
        text: "Think of compute as a fixed number of study hours. You can spend them making the student smarter (more parameters) or making them read more books (more tokens). A very bright student who reads one book learns little; an average student who reads the whole library cannot absorb it. Scaling laws measure how test performance responds to each, and the compute-optimal recipe says: for every doubling of your hours, make the student about √2 times more capable *and* give them about √2 times more books. Kaplan et al. advised spending most of the extra budget on a bigger student; Chinchilla showed that the students of the time had been under-read.",
      },
      {
        t: "p",
        text: "**Power laws** are straight lines on log–log axes: every 10× increase in model size reduces the loss by the same *factor*. They imply steady but diminishing returns — the hundredth billion parameters helps far less in absolute terms than the first — and they let you predict a large model's loss from small ones, which is how labs choose their final training run.",
      },
      {
        t: "callout",
        kind: "analogy",
        title: "LoRA: editing with a thin overlay",
        text: "Instead of repainting a huge mural (full fine-tuning: change every pixel of every weight matrix), you lay a transparent sheet over it and paint only a few broad strokes (a low-rank update). The mural stays intact underneath — you can peel the sheet off, or keep several sheets for several tasks. LoRA's bet is that adapting a model to a new task needs only a few such broad strokes per layer.",
      },
    ],
    mechanism: [
      { t: "h", text: "Running a scaling study" },
      {
        t: "steps",
        items: [
          { title: "Fix the family", text: "One architecture family, one data distribution, one tokenizer; vary only the size (width/depth) — and train each model with appropriately tuned hyper-parameters, because a badly tuned large model bends the curve." },
          { title: "Choose the axis", text: "Parameters $N$ (Kaplan counted non-embedding parameters), data $D$, or compute $C$. For $L(N)$, train long enough (or on enough data) that data is not the bottleneck." },
          { title: "Measure", text: "Held-out loss for each size, ideally over several seeds; sizes spaced geometrically (×2 or ×4)." },
          { title: "Fit", text: "Linear regression of $\\log L$ on $\\log N$ (or a fit with an irreducible term $L = E + A/N^\\alpha$)." },
          { title: "Validate by extrapolation", text: "Predict a larger size *before* training it, then check. A fit that only describes the points it was fitted to proves little." },
        ],
      },
      { t: "h", text: "LoRA forward and backward" },
      {
        t: "steps",
        items: [
          { title: "Freeze", text: "Keep the pretrained weight $W_0 \\in \\mathbb R^{d\\times k}$ (and bias) fixed: no gradients, no optimizer state." },
          { title: "Add two small matrices", text: "$A \\in \\mathbb R^{r\\times k}$ initialised randomly (Gaussian), $B \\in \\mathbb R^{d\\times r}$ initialised to zero, with rank $r \\ll \\min(d, k)$." },
          { title: "Forward", text: "Compute the usual output plus a scaled low-rank correction. Because $B = 0$ initially, the adapted model starts *exactly* at the pretrained model.", tex: String.raw`h = W_0x + \frac{\alpha}{r}\,B(Ax)` },
          { title: "Backward", text: "Only $A$ and $B$ receive gradients: $\\partial L/\\partial B = \\frac{\\alpha}{r}\\,g\\,(Ax)^\\top$ and $\\partial L/\\partial A = \\frac{\\alpha}{r}\\,B^\\top g\\,x^\\top$, with $g = \\partial L/\\partial h$. The gradient still flows *through* $W_0$ to earlier layers." },
          { title: "Merge for deployment", text: "$W = W_0 + \\frac{\\alpha}{r}BA$ is an ordinary matrix: no extra inference latency. Unmerge (subtract) to switch tasks." },
        ],
      },
      { t: "h", text: "QLoRA in three ideas" },
      {
        t: "list",
        items: [
          "**4-bit NormalFloat (NF4)**: pretrained weights are roughly normally distributed, so the 16 quantisation levels are placed at quantiles of a normal distribution (with an exact zero), normalised to [−1, 1]; each block of 64 weights is scaled by its absolute maximum.",
          "**Double quantisation**: the per-block scale constants are themselves quantised to 8 bits, cutting their overhead from $32/64 = 0.5$ to $8/64 + 32/(64\\cdot256) \\approx 0.127$ bits per parameter.",
          "**Paged optimizers**: optimizer states can be paged between GPU and CPU memory to survive memory spikes.",
          "Forward/backward: the 4-bit weights are de-quantised to bfloat16 on the fly for each matrix multiplication; gradients flow through the frozen quantised base into the LoRA adapters, which are kept in 16-bit precision.",
        ],
      },
    ],
    math: [
      { t: "h", text: "1. Power laws and log–log regression" },
      { t: "math", tex: String.raw`L(N) = a\,N^{-\alpha}\quad\Longleftrightarrow\quad \log L = \log a - \alpha\log N` },
      {
        t: "p",
        text: "$a > 0$ is a scale constant and $\\alpha > 0$ the **scaling exponent**. Multiplying $N$ by $k$ multiplies the loss by $k^{-\\alpha}$, independently of $N$. Fitting is ordinary least squares on the points $(\\log N_i, \\log L_i)$: the slope is $-\\alpha$ and the intercept $\\log a$. Real curves often bend towards an irreducible loss $E$ (the entropy of the data that no model can remove), $L(N) = E + A N^{-\\alpha}$, which needs a non-linear fit.",
      },
      {
        t: "p",
        text: "**Our measurement** (From-Scratch program): random-feature networks with $N = 16, \\dots, 512$ fitted output weights give test MSE 0.15242 → 0.03741, and the fit is $L(N) = 0.441\\,N^{-0.392}$ with $R^2 = 0.984$ in log space. Extrapolated to $N = 1024$ it predicts 0.02905; training that size gives 0.02676 — within 9%, but already visibly off the line, which is typical: small-scale fits are useful guides, not exact predictions.",
      },
      { t: "h", text: "2. Kaplan et al. (2020)" },
      {
        t: "p",
        text: "For Transformer language models (loss in nats per token, $N$ = non-embedding parameters, $D$ = tokens), they reported, each when the other factors are not limiting:",
      },
      { t: "math", tex: String.raw`L(N) = \Big(\frac{N_c}{N}\Big)^{\alpha_N},\ \alpha_N \approx 0.076,\ N_c \approx 8.8\times10^{13};\qquad L(D) = \Big(\frac{D_c}{D}\Big)^{\alpha_D},\ \alpha_D \approx 0.095,\ D_c \approx 5.4\times10^{13}` },
      {
        t: "p",
        text: "Worked numbers: doubling $N$ multiplies the loss by $2^{-0.076} = 0.9487$ (a 5.1% reduction); 10× more parameters multiplies it by $10^{-0.076} = 0.8395$. At $N = 10^9$ the formula gives $(8.8\\times10^{13}/10^9)^{0.076} = 2.376$ nats, at $10^{10}$ it gives 1.994. Kaplan et al. also concluded that, for a growing compute budget, model size should grow much faster than data (roughly $N_{opt} \\propto C^{0.73}$), and that large models are more sample-efficient — advice that shaped models such as GPT-3 (175B parameters trained on about 300B tokens).",
      },
      { t: "h", text: "3. Training compute: C ≈ 6ND" },
      {
        t: "p",
        text: "A forward pass of a dense Transformer costs about $2N$ FLOPs per token (one multiply and one add per parameter); the backward pass costs about twice the forward pass ($4N$: gradients with respect to activations and to weights). Hence training on $D$ tokens costs",
      },
      { t: "math", tex: String.raw`C \approx 2ND + 4ND = 6ND\ \text{FLOPs}`, caption: "Ignores attention's sequence-length-dependent term, which is small when the model width is large relative to the context length." },
      { t: "h", text: "4. Chinchilla: compute-optimal allocation" },
      {
        t: "p",
        text: "Hoffmann et al. trained over 400 models (70M to over 16B parameters, 5B to 500B tokens) and used three approaches to find, for each compute budget, the loss-minimising $N$ and $D$. All three found exponents near 0.5: $N_{opt} \\propto C^{a}$, $D_{opt} \\propto C^{b}$ with $a \\approx b \\approx 0.5$ — parameters and tokens should be scaled **equally**. The practical rule of thumb from their results is about **20 tokens per parameter**. With $D = 20N$ and $C = 6ND = 120N^2$:",
      },
      { t: "math", tex: String.raw`N_{opt} \approx \sqrt{\frac{C}{120}},\qquad D_{opt} \approx 20\,N_{opt}` },
      {
        t: "steps",
        items: [
          { title: "Example: C = 10²² FLOPs", tex: String.raw`N_{opt} = \sqrt{10^{22}/120} = 9.13\times10^{9},\qquad D_{opt} = 1.83\times10^{11}\ \text{tokens}` },
          { title: "Chinchilla itself", tex: String.raw`N = 70\times10^9,\ D = 1.4\times10^{12}\ (20\ \text{tokens/param}),\quad C = 6\cdot 70\times10^9\cdot1.4\times10^{12} = 5.88\times10^{23}` },
        ],
      },
      {
        t: "p",
        text: "Chinchilla used the same compute as DeepMind's earlier Gopher (280B parameters, 300B tokens) and outperformed it on a large range of benchmarks. The paper's third approach fits a parametric loss $\\hat L(N, D) = E + A/N^{\\alpha} + B/D^{\\beta}$ with reported values $E = 1.69$, $A = 406.4$, $B = 410.7$, $\\alpha = 0.34$, $\\beta = 0.28$. Plugging in (From-Scratch program): 70B/1.4T gives 1.937, 280B on the same budget gives 1.983 — the smaller, longer-trained model wins. But minimising these rounded coefficients along $C = 6ND$ puts the optimum at 32.5B parameters and about 93 tokens per parameter, not 20: the three approaches in the paper do not agree exactly, and a 2024 replication attempt (Besiroglu et al., \"Chinchilla Scaling: A replication attempt\") reported problems with the Approach-3 fit. Treat \"20 tokens per parameter\" as an approximate rule, not a constant of nature.",
      },
      {
        t: "callout",
        kind: "note",
        text: "Compute-optimal is about *training* compute. If a model will serve many requests, inference cost favours a smaller model trained on many more tokens than the Chinchilla ratio (\"over-training\"), and several widely used open models are trained this way.",
      },
      { t: "h", text: "5. LoRA mathematics" },
      {
        t: "p",
        text: "For a frozen $W_0 \\in \\mathbb R^{d\\times k}$, LoRA restricts the update to rank $\\le r$: $W = W_0 + \\Delta W$, $\\Delta W = \\frac{\\alpha}{r}BA$, $B \\in \\mathbb R^{d\\times r}$, $A \\in \\mathbb R^{r\\times k}$. Symbols: $d$ output size, $k$ input size, $r$ rank, $\\alpha$ a constant scaling hyper-parameter. Dividing by $r$ keeps the update's scale roughly constant when $r$ changes, so the learning rate need not be re-tuned for every rank (the paper sets $\\alpha$ once and does not tune it).",
      },
      { t: "math", tex: String.raw`\#\text{params}_{\text{full}} = dk,\qquad \#\text{params}_{\text{LoRA}} = r(d + k),\qquad d = k = 4096,\ r = 8:\ \ 16{,}777{,}216\ \text{vs}\ 65{,}536\ (0.39\%)` },
      {
        t: "p",
        text: "Why low rank can suffice: by Eckart–Young, the best rank-$r$ approximation of any matrix keeps its $r$ largest singular directions, and Hu et al. found that the updates learned during adaptation of large language models have a small \"intrinsic rank\" — very low ranks (even $r = 1$ or 2) were competitive in their experiments on GPT-3. In our NumPy experiment the true update has rank 2: rank 1 reaches test MSE 0.02002 (it cannot represent the update), ranks 2, 4 and 8 reach 0.00256–0.00260, equal to full fine-tuning (0.00261) with 9.4% of its parameters.",
      },
      { t: "h", text: "6. Fine-tuning memory: why PEFT matters" },
      {
        t: "p",
        text: "Mixed-precision Adam needs roughly 16 bytes per *trainable* parameter (fp16 weights 2 + fp16 gradients 2 + fp32 master weights 4 + two fp32 moments 8), plus activations. Full fine-tuning of a 7B model: $7\\times10^9 \\times 16 = 112$ GB. With LoRA, the frozen base needs only its fp16 weights (14 GB) and the adapters' optimizer state is negligible. With QLoRA the base is 4-bit: $7\\times10^9 \\times (4 + 0.127)/8 \\approx 3.6$ GB with double quantisation.",
      },
      { t: "h", text: "7. NF4 quantisation" },
      {
        t: "p",
        text: "Block-wise absmax quantisation of a block $w_{1..64}$: scale $c = \\max_i |w_i|$, code $q_i = \\arg\\min_j |w_i/c - \\ell_j|$ for 16 levels $\\ell_j \\in [-1, 1]$, de-quantised value $\\hat w_i = c\\,\\ell_{q_i}$. Uniform int4 spaces the levels evenly; NF4 places them at normal quantiles, so each level is used about equally often for normally distributed weights. The program builds the NF4 levels (−1, −0.6962, …, 0, 0.0796, …, 0.7230, 1) and finds a 31% lower weight MSE than uniform int4 on a Gaussian weight matrix ($1.293\\times10^{-4}$ vs $1.874\\times10^{-4}$).",
      },
    ],
    visual: [
      {
        t: "p",
        text: "A scaling curve is a capacity curve measured carefully. The fitting simulator shows the same trade-off at toy scale: more capacity reduces training error steadily, while held-out error depends on whether data or capacity is the bottleneck.",
      },
      { t: "sim", id: "fitting", caption: "Model capacity vs data: watch training and validation error as the polynomial degree grows." },
      {
        t: "list",
        items: [
          "Increase the degree step by step and read the training error: it falls steadily — the analogue of $L(N)$ falling with $N$ when data is plentiful.",
          "With few samples, validation error turns up at high degree: the model is now **data-limited**. Scaling laws are measured in the regime where the varied factor is the bottleneck — this is why Kaplan et al. give separate $L(N)$ and $L(D)$ laws and why Chinchilla grows $N$ and $D$ together.",
          "Compare with the From-Scratch table: our random-feature models were given 8,000 training points so that the test loss is controlled by $N$ across the measured range.",
        ],
      },
      { t: "sim", id: "quantization", caption: "Quantisation: mapping continuous weights onto a few discrete levels, the operation QLoRA applies to the frozen base model." },
      {
        t: "p",
        text: "In the quantisation simulator, observe how the error depends on the number of levels and on where they are placed relative to the distribution of values — the reason NF4 puts more levels near zero, where most weights of a roughly normal distribution lie.",
      },
    ],
    scratch: [
      {
        t: "p",
        text: "**Program 1 — scaling.** Measure test loss for six model sizes (three seeds each), fit a power law by log–log regression, test the fit by extrapolating to a size not used for fitting, then compute Chinchilla-style compute-optimal sizes and evaluate the published parametric loss.",
      },
      { t: "code", lang: "python", title: "scaling_laws_numpy.py", code: SCALING, output: SCALING_OUT },
      {
        t: "list",
        items: [
          "Here $N$ counts the fitted output weights; the random first layer is fixed (not trained), which makes each \"training run\" an exact least-squares solve and the curve cheap to measure.",
          "The exponent 0.392 is specific to this toy task; language-model exponents are much smaller (≈ 0.076 for $L(N)$), meaning far slower improvement per doubling.",
          "The extrapolation error (0.02905 predicted vs 0.02676 measured) shows why labs fit on many sizes over several orders of magnitude and check predictions on intermediate runs before committing to a final model.",
          "The compute table confirms $6 N_{opt} D_{opt} = C$ and reproduces Chinchilla's scale: $5.76\\times10^{23}$ FLOPs → about 69B parameters and 1.39T tokens.",
        ],
      },
      {
        t: "p",
        text: "**Program 2 — LoRA and QLoRA from scratch.** A frozen \"pretrained\" layer, a new task whose ideal weights differ by a rank-2 update, LoRA with hand-written gradients and Adam for several ranks, merging, an NF4 codebook, block-wise 4-bit quantisation, and LoRA on top of the 4-bit base.",
      },
      { t: "code", lang: "python", title: "lora_qlora_numpy.py", code: LORA, output: LORA_OUT },
      {
        t: "list",
        items: [
          "Zero-initialising $B$ makes the initial output change exactly 0: fine-tuning starts from the pretrained function, not from a random perturbation of it.",
          "Rank 1 cannot express a rank-2 update (0.02002); rank ≥ 2 matches full fine-tuning. On real models the needed rank is unknown and is chosen by validation.",
          "Merged and unmerged outputs agree to $4\\times10^{-15}$: merging is exact algebra, so deployment costs nothing extra.",
          "NF4 has 31% lower weight error than uniform int4, and the 4-bit base alone loses accuracy (0.05475 vs 0.04803 for the fp base). LoRA on the 4-bit base adapts to the task (0.00879 at $r = 4$, 0.00440 at $r = 16$) but cannot fully cancel the quantisation error, which is full-rank noise; in this toy problem the quantisation error is large relative to the task's noise. Dettmers et al. report that on their LLM benchmarks QLoRA matched 16-bit fine-tuning — a claim about large networks, where per-weight noise matters much less.",
        ],
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "A `LoRALinear` module that wraps a trained `nn.Linear`, plus bottleneck adapters, IA³-style scaling, BitFit, a linear probe and full fine-tuning, all applied to the same pretrained digit MLP adapted to a new domain (digits mirrored along the diagonal) with only 200 labelled target examples, three seeds each. The `peft` library is not needed: LoRA is a dozen lines.",
              },
              { t: "code", lang: "python", title: "peft_comparison.py", code: PEFT, output: PEFT_OUT },
              {
                t: "list",
                items: [
                  "Before adaptation the model is at chance on the new domain (0.111). LoRA $r = 2$ (4.5% of the parameters) reaches 0.852 and $r = 8$ reaches 0.871 — on par with full fine-tuning (0.856) with 200 training examples, where full fine-tuning's 26k free parameters can overfit.",
                  "**BitFit** and **IA³-style scaling** train about 1% of the parameters and are far weaker here (0.326, 0.435): they can only shift or rescale existing features, and a transposed image needs new first-layer features. On large pretrained language models, where useful features already exist, these methods are much more competitive.",
                  "A **linear probe** (0.785) is a strong cheap baseline — always report it.",
                  "The *source acc* column shows how the adapted network now does on the original digits. A merged LoRA model forgets (0.457), but because $W_0$ is untouched, subtracting the adapter recovers the original model exactly; full fine-tuning has no such undo.",
                  "Hyper-parameters: Adam lr $10^{-2}$ for all PEFT methods, $10^{-3}$ for full fine-tuning (a large lr on all weights destroys the pretrained features); $\\alpha = 8$ for every rank, so the scale is $\\alpha/r$.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "The same idea as a Keras layer wrapping a trained `Dense`: the base is marked non-trainable and only $A$ and $B$ are trainable weights." },
              { t: "code", lang: "python", title: "lora_keras.py (not executed in CI)", code: KERAS, output: "Defines the layer only; nothing is printed. Wrapping a trained model's Dense layers and calling fit trains only A and B." },
            ],
          },
        ],
      },
    ],
    variations: [
      { t: "h", text: "The PEFT family" },
      {
        t: "table",
        head: ["Method", "What is trained", "Trainable params (typical)", "Advantages", "Disadvantages"],
        rows: [
          ["Linear probe", "the output head only", "tiny", "cheapest; strong baseline", "cannot change features"],
          ["BitFit (Ben Zaken et al., 2022)", "bias terms only", "≈ 0.1% in Transformers", "trivial to implement; no new modules", "limited expressiveness (0.326 here)"],
          ["Adapters (Houlsby et al., 2019)", "small bottleneck MLPs inserted after sub-layers", "≈ 0.5–8%", "expressive; modular per task", "extra inference latency (sequential layers)"],
          ["Prefix tuning (Li & Liang, 2021)", "trainable key/value vectors prepended in every attention layer", "≈ 0.1%", "no weight changes; one model serves many tasks", "uses context length; can be unstable to train"],
          ["Prompt tuning (Lester et al., 2021)", "trainable soft-prompt embeddings at the input only", "≈ 0.01%", "fewest parameters; competitive for very large models", "weak for small models"],
          ["LoRA (Hu et al., 2021)", "low-rank $BA$ added to chosen weight matrices", "≈ 0.1–1% (r = 4–16)", "mergeable → no latency; strong results; adapters swappable", "rank and target modules to choose"],
          ["IA³ (Liu et al., 2022)", "learned vectors rescaling keys, values and FFN activations", "≈ 0.01%", "extremely few parameters; mergeable", "rescaling only (0.435 here)"],
          ["QLoRA (Dettmers et al., 2023)", "LoRA on a 4-bit NF4-quantised frozen base", "same as LoRA", "fine-tune very large models on one GPU", "de-quantisation overhead; quantisation error in the base"],
        ],
      },
      { t: "h", text: "Scaling-law variants" },
      {
        t: "table",
        head: ["Law", "Form", "Notes"],
        rows: [
          ["Kaplan et al. (2020)", "separate power laws in $N$, $D$, $C$", "recommended growing $N$ much faster than $D$"],
          ["Chinchilla (Hoffmann et al., 2022)", "$E + A/N^\\alpha + B/D^\\beta$; $N_{opt}, D_{opt} \\propto C^{\\approx0.5}$", "≈ 20 tokens/parameter"],
          ["Inference-aware", "adds serving cost to the objective", "favours smaller models trained longer"],
          ["Data-constrained", "accounts for repeated epochs over limited unique data", "repeated tokens are worth less than fresh ones"],
          ["Downstream / emergent abilities", "task metrics vs scale", "smooth loss can coexist with sharp jumps in thresholded task metrics"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Planning large training runs** — labs fit scaling laws on small models to choose the size, token count and hyper-parameters of the final run, and to predict its loss before spending the compute (see [LLM Pretraining](/learn/m20/llm-pretraining)).",
          "**Budgeting** — $C \\approx 6ND$ converts a GPU budget into model and data sizes: a GPU sustaining $10^{14}$ FLOP/s for 30 days delivers about $2.6\\times10^{20}$ FLOPs.",
          "**Instruction-tuning and domain adaptation of open models** — LoRA and QLoRA are the standard way to adapt open-weight LLMs to a domain, a style or a task on a single GPU; QLoRA's paper introduced the Guanaco models fine-tuned this way.",
          "**Multi-tenant serving** — one frozen base model with many small LoRA adapters (one per customer or task) loaded and swapped at request time.",
          "**Diffusion models** — LoRA is widely used to fine-tune text-to-image models on a style or subject with a few images.",
          "**On-device and edge** — 4-bit quantisation of weights (as in QLoRA's NF4, and post-training quantisation methods) makes large models fit in limited memory; see [Model Optimization](/learn/m18/model-optimization).",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Scaling laws are empirical.** They hold within a family, data distribution and training recipe; exponents change with data quality, architecture and tokenizer, and extrapolation over orders of magnitude carries risk (our 9% miss after one doubling).",
          "**Loss is not capability.** Smooth loss curves can hide abrupt changes in specific task metrics, and downstream performance depends on data mixture and alignment.",
          "**Compute-optimal ≠ deployment-optimal.** Serving costs favour smaller, longer-trained models.",
          "**LoRA: big savings, small risks.** Far less memory and storage, mergeable, swappable; but the rank, $\\alpha$ and which matrices to adapt must be chosen, and very different target tasks may need higher ranks or full fine-tuning.",
          "**QLoRA: largest memory savings,** at the cost of slower steps (de-quantisation) and quantisation error in the frozen base.",
          "**Prompt/prefix tuning** use the fewest parameters but occupy context length and work best on very large models.",
          "**PEFT reduces but does not eliminate forgetting** of the merged model on the original task — keep the adapter separate if you need both behaviours.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Fitting a power law to too narrow a range** (a factor of 4 in $N$) and extrapolating 1000×. Fix: span orders of magnitude, validate predictions on held-out sizes.",
          "**Comparing sizes with untuned hyper-parameters.** A fixed learning rate across sizes bends the curve (large models usually need smaller learning rates). Fix: tune per size, or use a principled transfer of hyper-parameters across width.",
          "**Counting parameters inconsistently** — with or without embeddings. Kaplan's laws use non-embedding parameters; mixing conventions distorts small-model points.",
          "**Fitting $\\log L$ vs $\\log N$ when the loss is near its floor.** Close to the irreducible loss the curve flattens; fit $E + A N^{-\\alpha}$ instead.",
          "**Initialising both LoRA matrices randomly.** Then the adapted model starts away from the pretrained one. Fix: $B = 0$ (or $A = 0$), so $\\Delta W = 0$ at the start.",
          "**Forgetting the $\\alpha/r$ scale when changing $r$**, or merging without it. Fix: $W = W_0 + \\frac{\\alpha}{r}BA$ everywhere.",
          "**Leaving base weights trainable** (e.g. forgetting `requires_grad_(False)`), which silently turns LoRA into full fine-tuning with extra parameters. Fix: count trainable parameters and print them, as the program does.",
          "**Using the full-fine-tuning learning rate for LoRA (or vice versa).** PEFT typically needs a larger learning rate (here $10^{-2}$ vs $10^{-3}$).",
          "**Assuming 4-bit means 4 bits per parameter.** Block constants add 0.5 bits (0.127 with double quantisation), and activations, LoRA weights and optimizer state still need memory.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is a scaling law?", a: "An empirical relation, usually a power law $L \\approx (N_c/N)^{\\alpha}$, between a model's test loss and a resource — parameters, data or compute — holding over orders of magnitude. On log–log axes it is a straight line, so small models can be used to predict large ones." },
      { t: "qa", level: "beginner", q: "What does LoRA train, and why does it save memory?", a: "It freezes the pretrained weight $W_0$ and trains two small matrices $A$ ($r\\times k$) and $B$ ($d\\times r$) whose product is a low-rank update. Only $r(d+k)$ parameters need gradients and optimizer state instead of $dk$ — e.g. 65,536 instead of 16.8M for a 4096×4096 matrix at $r = 8$." },
      { t: "qa", level: "beginner", q: "Why is $B$ initialised to zero in LoRA?", a: "So that $\\Delta W = BA = 0$ at the start: the adapted model is exactly the pretrained model, and fine-tuning begins from its function rather than from a random perturbation. $A$ is random so that gradients for $B$ are non-zero ($\\partial L/\\partial B \\propto g(Ax)^\\top$)." },
      { t: "qa", level: "intermediate", q: "Derive C ≈ 6ND.", a: "A forward pass multiplies each activation by each parameter once per token: about $2N$ FLOPs (multiply + add). The backward pass computes gradients w.r.t. activations and w.r.t. weights, each about the cost of a forward pass: $4N$. Total $6N$ per token, $6ND$ for $D$ tokens (ignoring the attention term)." },
      { t: "qa", level: "intermediate", q: "What did Chinchilla change compared with Kaplan et al.?", a: "Kaplan et al. recommended growing model size much faster than data ($N_{opt} \\propto C^{0.73}$). Hoffmann et al., training many models to completion with tuned schedules, found $N_{opt}$ and $D_{opt}$ should grow about equally ($\\propto C^{0.5}$), ≈ 20 tokens per parameter, so models of the time were undertrained. Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at the same compute." },
      { t: "qa", level: "intermediate", q: "With C = 10²² FLOPs, what model size and token count does the 20-tokens-per-parameter rule suggest?", a: "$C = 6N\\cdot20N = 120N^2$ → $N = \\sqrt{10^{22}/120} \\approx 9.1\\times10^9$ parameters and $D = 20N \\approx 1.8\\times10^{11}$ tokens." },
      { t: "qa", level: "intermediate", q: "Compare LoRA, adapters and prompt tuning.", a: "LoRA adds a low-rank update to weight matrices; it can be merged, so no inference latency. Adapters insert small bottleneck layers, adding sequential computation at inference. Prompt tuning learns only input soft-prompt embeddings: fewest parameters, no change to weights, but uses context length and works well mainly for very large models. All freeze the base model and allow per-task modules." },
      { t: "qa", level: "advanced", q: "Explain NF4 and double quantisation in QLoRA.", a: "NF4 is a 4-bit data type with 16 levels at quantiles of a standard normal (including an exact zero), normalised to [−1, 1], applied block-wise (64 weights per block, scaled by the block's absmax). For normally distributed weights each level is used about equally often, giving lower error than uniform int4 (31% lower MSE in our test). Double quantisation stores the per-block fp32 constants in 8 bits with a second-level constant per 256 blocks, reducing overhead from 0.5 to about 0.127 bits per parameter." },
      { t: "qa", level: "advanced", q: "Why might a low rank be enough for fine-tuning, and when would it not be?", a: "Pretrained models already contain most needed features; adaptation mostly re-weights a few directions, so the weight change has low intrinsic rank (Hu et al. found very small ranks competitive on GPT-3). It may fail when the target task requires genuinely new features — a large domain shift, a new modality or language — where higher ranks, more adapted matrices or full fine-tuning work better. In our domain-shift experiment rank 1 was clearly worse than rank 8 (0.795 vs 0.871)." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "A power law $L = aN^{-\\alpha}$ plotted on log–log axes is…", options: ["an exponential curve", "a straight line with slope $-\\alpha$", "a parabola", "a horizontal line"], answer: 1, explain: "$\\log L = \\log a - \\alpha\\log N$." },
          { kind: "numeric", q: "With $\\alpha_N = 0.076$, by what factor does the loss shrink when $N$ is multiplied by 10? (4 decimals)", answer: 0.8395, tolerance: 0.0005, explain: "$10^{-0.076} = 0.8395$." },
          { kind: "numeric", q: "Training compute for a 1B-parameter model on 20B tokens, in units of $10^{20}$ FLOPs?", answer: 1.2, tolerance: 0.01, explain: "$6 \\times 10^9 \\times 2\\times10^{10} = 1.2\\times10^{20}$." },
          { kind: "numeric", q: "Using $N_{opt} = \\sqrt{C/120}$, what is $N_{opt}$ for $C = 1.2\\times10^{21}$ FLOPs, in billions?", answer: 3.162, tolerance: 0.01, explain: "$\\sqrt{10^{19}} = 3.162\\times10^9$." },
          { kind: "numeric", q: "How many trainable parameters does LoRA with $r = 16$ add to a $1024\\times4096$ weight matrix?", answer: 81920, tolerance: 0.5, explain: "$r(d + k) = 16 \\times (1024 + 4096) = 81{,}920$." },
          { q: "Which PEFT method can be merged into the weights so inference has no extra latency?", options: ["Bottleneck adapters", "Prefix tuning", "LoRA", "Prompt tuning"], answer: 2, explain: "$W_0 + \\frac{\\alpha}{r}BA$ is one matrix." },
          { q: "What does NF4 optimise compared with uniform int4?", options: ["Fewer bits", "Level placement matched to normally distributed weights", "Exact representation of every weight", "Faster matrix multiplication"], answer: 1, explain: "Levels at normal quantiles; lower error for bell-shaped weight distributions." },
          { kind: "numeric", q: "Mixed-precision Adam uses ≈ 16 bytes per trainable parameter. How many GB for full fine-tuning of a 3B model (optimizer + weights + grads, no activations)?", answer: 48, tolerance: 0.5, explain: "$3\\times10^9 \\times 16 = 48\\times10^9$ bytes." },
        ],
      },
      {
        t: "exercise",
        title: "Fit a power law by hand",
        prompt: "Three model sizes give losses: $N = 10^6$: 4.00, $N = 10^7$: 3.20, $N = 10^8$: 2.56. (a) Fit $L = aN^{-\\alpha}$. (b) Predict $L$ at $N = 10^9$. (c) How many parameters would be needed for $L = 1.5$?",
        hint: "Each 10× gives the same ratio, so $10^{-\\alpha} = 3.2/4.0$.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) exponent", tex: String.raw`10^{-\alpha} = 0.8\ \Rightarrow\ \alpha = -\log_{10}0.8 = 0.096910` },
              { title: "(a) scale", tex: String.raw`a = 4.00\times(10^6)^{0.096910} = 4.00\times10^{0.581460} = 4.00\times3.8147 = 15.259` },
              { title: "(b) prediction", tex: String.raw`L(10^9) = 2.56\times0.8 = 2.048` },
              { title: "(c) required size", tex: String.raw`N = \Big(\frac{a}{1.5}\Big)^{1/\alpha} = (10.173)^{10.319} = 10^{1.007429\times10.319} = 10^{10.396} \approx 2.5\times10^{10}` },
            ],
          },
          { t: "p", text: "Each further 10× buys only a 20% reduction, so going from 2.56 to 1.5 needs roughly 250× more parameters — the diminishing returns of power-law scaling (and a reminder that this extrapolation assumes the law holds over two more orders of magnitude)." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: LoRA on the attention projections only",
        prompt: "In a Transformer, LoRA is often applied only to the query and value projections. Using `LoRALinear` from `peft_comparison.py`, write a function `add_lora(model, names, r, alpha)` that replaces every `nn.Linear` whose attribute name is in `names` (e.g. {'q_proj', 'v_proj'}) with a `LoRALinear`, freezes everything else, and returns the number of trainable parameters. Test it on `nn.TransformerEncoderLayer`-style modules or on a small module tree of your own.",
        hint: "Iterate with `model.named_modules()`, collect (parent, attribute name, child) triples first, then `setattr(parent, name, LoRALinear(child, r, alpha))`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "def add_lora(model, names, r=8, alpha=16):\n    for p in model.parameters():\n        p.requires_grad_(False)\n    targets = []\n    for parent in model.modules():\n        for attr, child in parent.named_children():\n            if attr in names and isinstance(child, nn.Linear):\n                targets.append((parent, attr, child))\n    for parent, attr, child in targets:\n        setattr(parent, attr, LoRALinear(child, r=r, alpha=alpha))\n    return sum(p.numel() for p in model.parameters() if p.requires_grad)\n\n# check: a toy block with q_proj, k_proj, v_proj of size 64x64 and r = 8\n# trainable = 2 adapted layers x 8 x (64 + 64) = 2048",
          },
          { t: "p", text: "Collecting the targets before replacing avoids modifying the module tree while iterating over it. For two adapted 64×64 projections at $r = 8$ the count is $2\\times8\\times128 = 2048$, which is a good unit test of the function." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Power law $L = aN^{-\\alpha}$ ↔ straight line in log–log; fit by least squares on logs; validate by extrapolating to an unseen size (ours: 0.02905 predicted vs 0.02676 measured).",
          "Kaplan et al. (2020): $\\alpha_N \\approx 0.076$, $\\alpha_D \\approx 0.095$; 10× parameters → loss × 0.84.",
          "$C \\approx 6ND$ (2N forward + 4N backward per token).",
          "Chinchilla (2022): scale $N$ and $D$ equally, ≈ 20 tokens/param, $N_{opt} \\approx \\sqrt{C/120}$; 70B/1.4T beat 280B/300B at the same compute; the rule is approximate.",
          "LoRA: $h = W_0x + \\frac{\\alpha}{r}BAx$, $B = 0$ init, $r(d+k)$ trainable params, mergeable, swappable.",
          "QLoRA: NF4 4-bit frozen base (normal-quantile levels, block size 64), double quantisation (0.5 → 0.127 bits/param overhead), paged optimizers, 16-bit LoRA adapters.",
          "PEFT family: linear probe, BitFit, adapters, prefix/prompt tuning, LoRA, IA³ — trade trainable parameters against expressiveness; on our domain shift LoRA r = 2–8 matched full fine-tuning with 4–18% of the parameters.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — applies these scaling laws to plan real pretraining runs.",
          "[Model Optimization](/learn/m18/model-optimization) — quantisation, pruning and distillation for deployment.",
          "[Efficient Inference](/learn/m20/llm-inference) — KV caching, FlashAttention, MoE and LLM quantisation at serving time.",
          "[Instruction Tuning, RLHF and DPO](/learn/m20/alignment) — where LoRA/QLoRA are most often used in practice.",
        ],
      },
    ],
  },
};

export default lesson;
