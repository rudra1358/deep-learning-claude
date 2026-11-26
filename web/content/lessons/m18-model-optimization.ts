import type { Lesson } from "@/lib/types";

const NUMPY_CODE = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- quantizers
def qparams(lo, hi, bits, symmetric):
    """Scale and zero-point for mapping the float range [lo, hi] onto b-bit integers."""
    if symmetric:                                   # signed grid, zero-point fixed at 0
        qmax = 2 ** (bits - 1) - 1                  # e.g. 127 for 8 bits (we drop -128 to stay symmetric)
        qmin = -qmax
        amax = np.maximum(np.abs(lo), np.abs(hi))
        scale = np.where(amax > 0, amax / qmax, 1.0)
        zero = np.zeros_like(scale)
    else:                                           # unsigned grid 0..2^b-1, zero-point shifts it
        qmin, qmax = 0, 2 ** bits - 1
        lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)   # range must contain 0 so 0.0 is exact
        scale = np.where(hi > lo, (hi - lo) / (qmax - qmin), 1.0)
        zero = np.round(qmin - lo / scale)
    return scale, zero, qmin, qmax


def fake_quant(w, bits, symmetric=True, per_channel=False):
    """quantize -> integers -> dequantize. per_channel: one (scale, zero) per output row."""
    if per_channel:
        lo, hi = w.min(axis=1, keepdims=True), w.max(axis=1, keepdims=True)
    else:
        lo, hi = w.min(), w.max()
    scale, zero, qmin, qmax = qparams(lo, hi, bits, symmetric)
    q = np.clip(np.round(w / scale + zero), qmin, qmax)       # the stored integers
    return (q - zero) * scale, q, scale, zero


def sqnr_db(w, w_hat):
    return 10 * np.log10(np.sum(w ** 2) / np.sum((w - w_hat) ** 2))


# ---------------------------------------------------------------- 1. one value, step by step
w = np.array([-0.62, -0.10, 0.0, 0.31, 0.95])
w_hat, q, s, z = fake_quant(w, bits=8, symmetric=False)
print(f"asymmetric 8-bit: scale={float(s):.6f} zero_point={int(z)}")
print("  w      :", w)
print("  q (int):", q.astype(int))
print("  w_hat  :", np.round(w_hat, 6))
print(f"  max |error| = {np.abs(w - w_hat).max():.6f}  <=  scale/2 = {float(s) / 2:.6f}")

# ---------------------------------------------------------------- 2. per-tensor vs per-channel
# A weight matrix whose rows have very different magnitudes (common after training).
rng = np.random.default_rng(0)
row_scale = np.array([0.02, 0.1, 0.5, 2.0])[:, None]
Wm = rng.laplace(0, 1, size=(4, 256)) * row_scale
print("\nSQNR in dB (higher is better)      per-tensor  per-channel")
for bits in (8, 4):
    pt = sqnr_db(Wm, fake_quant(Wm, bits)[0])
    pc = sqnr_db(Wm, fake_quant(Wm, bits, per_channel=True)[0])
    print(f"  {bits}-bit symmetric                  {pt:8.2f}    {pc:8.2f}")
print("row 0 (tiny weights), 4-bit per-tensor: fraction rounded to 0 =",
      f"{np.mean(fake_quant(Wm, 4)[0][0] == 0):.3f}")

# Calibration: clipping the range below max|w| trades a few large clipping errors
# for a finer grid everywhere else. Search the clip ratio per channel for 4 bits.
best, plain = [], []
for row in Wm:
    amax = np.abs(row).max()
    results = []
    for ratio in np.linspace(0.2, 1.0, 81):
        scale = ratio * amax / 7                        # 4-bit symmetric: q in [-7, 7]
        r_hat = np.clip(np.round(row / scale), -7, 7) * scale
        results.append((sqnr_db(row, r_hat), ratio))
    best.append(max(results))
    plain.append(results[-1][0])                        # ratio = 1.0: no clipping
print("4-bit per-channel SQNR per row, clip at max|w|:", [f"{s:.2f}" for s in plain])
print("4-bit per-channel SQNR per row, best clipping :", [f"{s:.2f}" for s, _ in best],
      " clip ratios", [f"{r:.2f}" for _, r in best])


# ---------------------------------------------------------------- 3. a real model: digits MLP in NumPy
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.25, stratify=y, random_state=0)
H = 64
W1 = rng.normal(0, np.sqrt(2 / 64), (64, H)); b1 = np.zeros(H)
W2 = rng.normal(0, np.sqrt(2 / H), (H, 10)); b2 = np.zeros(10)
Y = np.eye(10)[y_tr]
for step in range(300):                                 # full-batch gradient descent
    h = np.maximum(0, X_tr @ W1 + b1)
    lg = h @ W2 + b2
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    g2 = (p - Y) / len(X_tr)
    gh = (g2 @ W2.T) * (h > 0)
    W2 -= 0.5 * h.T @ g2; b2 -= 0.5 * g2.sum(0)
    W1 -= 0.5 * X_tr.T @ gh; b1 -= 0.5 * gh.sum(0)


def accuracy(W1_, W2_):
    return np.mean((np.maximum(0, X_te @ W1_ + b1) @ W2_ + b2).argmax(1) == y_te)


print(f"\nfloat32 test accuracy: {accuracy(W1, W2):.4f}")
print("weight-only PTQ   per-tensor  per-channel   (biases kept in float)")
for bits in (8, 4, 3, 2):
    # rows of W.T are output channels; quantize each layer's weights
    pt = accuracy(fake_quant(W1.T, bits)[0].T, fake_quant(W2.T, bits)[0].T)
    pc = accuracy(fake_quant(W1.T, bits, per_channel=True)[0].T, fake_quant(W2.T, bits, per_channel=True)[0].T)
    print(f"  {bits}-bit            {pt:.4f}      {pc:.4f}")

# ---------------------------------------------------------------- 4. integer-only matmul
# y = x W with x and W quantized symmetrically: sum(q_x q_w) is an int32 dot product,
# rescaled once by s_x * s_w at the end.
x = X_te[:1]
xq_hat, qx, sx, _ = fake_quant(x, 8)
wq_hat, qw, sw, _ = fake_quant(W1, 8)
acc32 = qx.astype(np.int32) @ qw.astype(np.int32)      # what int8 hardware computes
y_int = acc32 * (sx * sw)
print(f"\nint8 x int8 -> int32 matmul vs float: max |diff| = {np.abs(y_int - x @ W1).max():.5f}"
      f" (largest |pre-activation| {np.abs(x @ W1).max():.3f})")

# ---------------------------------------------------------------- 5. magnitude pruning
print("\nunstructured magnitude pruning (global, both layers), no fine-tuning")
all_w = np.concatenate([np.abs(W1).ravel(), np.abs(W2).ravel()])
for sparsity in (0.5, 0.8, 0.9, 0.95):
    thr = np.quantile(all_w, sparsity)                  # drop the smallest |w|
    print(f"  {int(sparsity * 100)}% zeros -> accuracy {accuracy(W1 * (np.abs(W1) > thr), W2 * (np.abs(W2) > thr)):.4f}")

print("structured pruning: remove whole hidden neurons with the smallest L2 norm of incoming weights")
norms = np.linalg.norm(W1, axis=0)
for keep in (48, 32, 16, 8):
    idx = np.sort(np.argsort(norms)[-keep:])           # neurons to keep -> a genuinely smaller layer
    acc = np.mean((np.maximum(0, X_te @ W1[:, idx] + b1[idx]) @ W2[idx] + b2).argmax(1) == y_te)
    print(f"  keep {keep:2d}/64 neurons -> W1 {W1[:, idx].shape}, accuracy {acc:.4f}")`;
const NUMPY_OUT = String.raw`asymmetric 8-bit: scale=0.006157 zero_point=101
  w      : [-0.62 -0.1   0.    0.31  0.95]
  q (int): [  0  85 101 151 255]
  w_hat  : [-0.621843 -0.09851   0.        0.307843  0.948157]
  max |error| = 0.002157  <=  scale/2 = 0.003078

SQNR in dB (higher is better)      per-tensor  per-channel
  8-bit symmetric                     32.97       37.87
  4-bit symmetric                     11.36       13.23
row 0 (tiny weights), 4-bit per-tensor: fraction rounded to 0 = 1.000
4-bit per-channel SQNR per row, clip at max|w|: ['16.96', '13.61', '13.75', '13.20']
4-bit per-channel SQNR per row, best clipping : ['18.06', '14.86', '15.30', '14.24']  clip ratios ['0.87', '0.76', '0.69', '0.84']

float32 test accuracy: 0.9711
weight-only PTQ   per-tensor  per-channel   (biases kept in float)
  8-bit            0.9711      0.9711
  4-bit            0.9667      0.9689
  3-bit            0.9689      0.9511
  2-bit            0.3978      0.8400

int8 x int8 -> int32 matmul vs float: max |diff| = 0.02077 (largest |pre-activation| 2.575)

unstructured magnitude pruning (global, both layers), no fine-tuning
  50% zeros -> accuracy 0.9756
  80% zeros -> accuracy 0.9156
  90% zeros -> accuracy 0.6178
  95% zeros -> accuracy 0.5422
structured pruning: remove whole hidden neurons with the smallest L2 norm of incoming weights
  keep 48/64 neurons -> W1 (64, 48), accuracy 0.9667
  keep 32/64 neurons -> W1 (64, 32), accuracy 0.9644
  keep 16/64 neurons -> W1 (64, 16), accuracy 0.5756
  keep  8/64 neurons -> W1 (64, 8), accuracy 0.4089`;
const TORCH_CODE = String.raw`import copy
import io
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")          # torch.ao.quantization prints deprecation warnings (stderr)
torch.manual_seed(0)
torch.set_num_threads(1)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.25, stratify=y, random_state=0)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


def make_mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def train(model, steps=200, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(model(X_tr), y_tr).backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def acc(model):
    return (model(X_te).argmax(1) == y_te).float().mean().item()


def size_kb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024


fp32 = train(make_mlp())
print(f"fp32 model:            accuracy {acc(fp32):.4f}   state_dict {size_kb(fp32):6.1f} KB")

# ---------------------------------------------------------------- 1. dynamic int8 quantization (PTQ)
# Weights are quantized ahead of time (per-tensor affine int8); activations are quantized
# on the fly per batch. No calibration data needed. Targets nn.Linear / LSTM layers on CPU.
int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8)
print(f"dynamic int8 model:    accuracy {acc(int8):.4f}   state_dict {size_kb(int8):6.1f} KB")
with torch.no_grad():
    diff = (int8(X_te) - fp32(X_te)).abs()
agree = (int8(X_te).argmax(1) == fp32(X_te).argmax(1)).float().mean().item()
print(f"  logits max |diff| {diff.max().item():.4f}, top-1 agreement with fp32 {agree:.4f}")
print("  layer 0 is now:", type(int8[0]).__module__ + "." + type(int8[0]).__name__)


# ---------------------------------------------------------------- 2. PTQ vs QAT at low bit-width
def fake_quant(w, bits):
    """Symmetric per-output-channel fake quantization with a straight-through estimator."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    w_q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (w_q - w).detach()          # forward: w_q; backward: gradient passes straight to w


class QuantLinear(nn.Linear):
    bits = 8

    def forward(self, x):
        return nn.functional.linear(x, fake_quant(self.weight, self.bits), self.bias)


def to_quant(model, bits):
    q = nn.Sequential(*[QuantLinear(m.in_features, m.out_features) if isinstance(m, nn.Linear) else m for m in model])
    q.load_state_dict(model.state_dict())
    for m in q:
        if isinstance(m, QuantLinear):
            m.bits = bits
    return q


print("\nweight bits   PTQ (round trained fp32)   QAT (fine-tune 100 steps with fake quant)")
for bits in (4, 3, 2):
    ptq = to_quant(fp32, bits).eval()
    torch.manual_seed(1)
    qat = train(to_quant(fp32, bits), steps=100, lr=1e-3)
    print(f"   {bits}            {acc(ptq):.4f}                      {acc(qat):.4f}")

# ---------------------------------------------------------------- 3. pruning with torch.nn.utils.prune
linears = [(m, "weight") for m in fp32 if isinstance(m, nn.Linear)]
n_weights = sum(m.weight.numel() for m, _ in linears)
print("\nglobal unstructured L1 pruning     no fine-tune   after 50 fine-tune steps")
for amount in (0.5, 0.8, 0.9, 0.95):
    pruned = copy.deepcopy(fp32)
    params = [(m, "weight") for m in pruned if isinstance(m, nn.Linear)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    before = acc(pruned)
    torch.manual_seed(1)
    train(pruned, steps=50, lr=1e-3)            # masks stay applied: pruned weights remain exactly 0
    zeros = sum((m.weight == 0).sum().item() for m, _ in params)
    print(f"   {int(amount * 100)}% of weights zero         {before:.4f}         {acc(pruned):.4f}"
          f"   (zeros {zeros}/{n_weights})")

# structured: remove 75% of the rows (output neurons) of the first hidden layer by L2 norm
s = copy.deepcopy(fp32)
prune.ln_structured(s[0], name="weight", amount=0.75, n=2, dim=0)
dead_rows = (s[0].weight.abs().sum(dim=1) == 0).sum().item()
print(f"\nstructured L2 pruning of layer 0 rows: {dead_rows}/256 neurons zeroed, accuracy {acc(s):.4f}")
prune.remove(s[0], "weight")                     # make it permanent: drop mask, keep zeros in .weight
print("after prune.remove, parameters are plain again:", [n for n, _ in s[0].named_parameters()])`;
const TORCH_OUT = String.raw`fp32 model:            accuracy 0.9689   state_dict  334.6 KB
dynamic int8 model:    accuracy 0.9689   state_dict   89.1 KB
  logits max |diff| 0.3340, top-1 agreement with fp32 1.0000
  layer 0 is now: torch.ao.nn.quantized.dynamic.modules.linear.Linear

weight bits   PTQ (round trained fp32)   QAT (fine-tune 100 steps with fake quant)
   4            0.9711                      0.9689
   3            0.9533                      0.9711
   2            0.2978                      0.9622

global unstructured L1 pruning     no fine-tune   after 50 fine-tune steps
   50% of weights zero         0.9622         0.9689   (zeros 42240/84480)
   80% of weights zero         0.7933         0.9667   (zeros 67584/84480)
   90% of weights zero         0.4156         0.8578   (zeros 76032/84480)
   95% of weights zero         0.2178         0.4711   (zeros 80256/84480)

structured L2 pruning of layer 0 rows: 192/256 neurons zeroed, accuracy 0.4978
after prune.remove, parameters are plain again: ['bias', 'weight']`;
const KD_CODE = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.25, stratify=y, random_state=0)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)
N_LAB = 100                       # the student only gets 100 labelled images ...
X_lab, y_lab = X_tr[:N_LAB], y_tr[:N_LAB]
X_transfer = X_tr                 # ... but may see all 1347 training images WITHOUT labels


def mlp(hidden):
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def acc(m):
    return (m(X_te).argmax(1) == y_te).float().mean().item()


def fit(model, loss_fn, steps=300, lr=1e-2, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model).backward()
        opt.step()
    return model.eval()


# The teacher: a big model trained on the full labelled set (in practice: an expensive production model).
torch.manual_seed(0)
teacher = fit(mlp(512), lambda m: F.cross_entropy(m(X_tr), y_tr))


def kd_loss(student, X_soft, T=4.0, alpha=0.5):
    """alpha * CE(hard labels) + (1 - alpha) * T^2 * KL(teacher_T || student_T)   (Hinton et al., 2015)."""
    hard = F.cross_entropy(student(X_lab), y_lab)
    with torch.no_grad():
        t_prob = F.softmax(teacher(X_soft) / T, dim=1)          # frozen teacher
    soft = F.kl_div(F.log_softmax(student(X_soft) / T, dim=1), t_prob, reduction="batchmean")
    return alpha * hard + (1 - alpha) * T * T * soft


print(f"teacher (hidden 512, {n_params(teacher)} params): test accuracy {acc(teacher):.4f}")
print(f"student (hidden 16, {n_params(mlp(16))} params), test accuracy:")
print("            100 labels only   KD on the 100   KD on 1347 unlabelled transfer images")
rows = []
for seed in range(3):
    r = []
    for loss in (lambda m: F.cross_entropy(m(X_lab), y_lab),
                 lambda m: kd_loss(m, X_lab),
                 lambda m: kd_loss(m, X_transfer)):
        torch.manual_seed(100 + seed)            # identical initialisation in all three columns
        r.append(acc(fit(mlp(16), loss)))
    rows.append(r)
    print(f"   seed {seed}      {r[0]:.4f}          {r[1]:.4f}          {r[2]:.4f}")
mean = torch.tensor(rows).mean(0)
print(f"   mean        {mean[0]:.4f}          {mean[1]:.4f}          {mean[2]:.4f}")`;
const KD_OUT = String.raw`teacher (hidden 512, 38410 params): test accuracy 0.9822
student (hidden 16, 1210 params), test accuracy:
            100 labels only   KD on the 100   KD on 1347 unlabelled transfer images
   seed 0      0.9000          0.9533          0.9778
   seed 1      0.9000          0.9556          0.9711
   seed 2      0.8956          0.9533          0.9756
   mean        0.8985          0.9541          0.9748`;
const TIME_CODE = String.raw`import copy
import time
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.manual_seed(0)
torch.set_num_threads(1)              # pin threads: otherwise results depend on what else is running

fp32 = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 10)).eval()
int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8)


@torch.no_grad()
def bench(model, batch, reps):
    x = torch.randn(batch, 512)
    for _ in range(5):                        # warm-up: allocations, caches, lazy init
        model(x)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    p95 = times[int(0.95 * len(times)) - 1]
    return median * 1e3, p95 * 1e3, batch / median     # ms, ms, samples per second


print("model   batch   median latency   p95 latency   throughput")
for batch in (1, 8, 64, 256):
    for name, m in (("fp32", fp32), ("int8", int8)):
        med, p95, thr = bench(m, batch, reps=200 if batch < 64 else 40)
        print(f"{name}   {batch:5d}   {med:10.3f} ms   {p95:8.3f} ms   {thr:10.0f} /s")`;
const TIME_OUT = String.raw`model   batch   median latency   p95 latency   throughput
fp32       1        1.478 ms      1.809 ms          677 /s
int8       1        0.372 ms      0.912 ms         2690 /s
fp32       8        3.419 ms      4.541 ms         2340 /s
int8       8        1.018 ms      1.548 ms         7861 /s
fp32      64        9.159 ms     11.341 ms         6988 /s
int8      64        3.285 ms      3.977 ms        19485 /s
fp32     256       30.738 ms     35.438 ms         8329 /s
int8     256       12.130 ms     13.309 ms        21105 /s`;

const TFLITE_CODE = String.raw`import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(64,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10),
])
# ... train the model ...


def representative_data():
    # ~100 typical inputs: used to calibrate activation ranges for full-integer quantization
    for _ in range(100):
        yield [np.random.rand(1, 64).astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]          # weight quantization
converter.representative_dataset = representative_data           # + activation calibration
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_bytes = converter.convert()
print(len(tflite_bytes))`;

const lesson: Lesson = {
  id: "m18/model-optimization",
  title: "Quantization, Pruning and Distillation",
  summary:
    "Making networks smaller and faster while measuring what accuracy it costs: affine and symmetric quantization (scale, zero-point, per-tensor vs per-channel, clipping, integer matmul), PTQ vs QAT, PyTorch dynamic int8 quantization, magnitude and structured pruning with fine-tuning, a practical knowledge-distillation recipe, and honest latency and throughput measurement — all on a real digits model.",
  minutes: 85,
  level: "advanced",
  prerequisites: ["m18/serving", "m03/multilayer-networks", "m15/learning-paradigms", "m07/regularization"],
  next: ["m18/mlops", "m19/scaling-efficiency"],
  terms: ["quantization", "scale-factor", "zero-point", "per-channel-quantization", "post-training-quantization", "quantization-aware-training", "straight-through-estimator", "sqnr", "pruning", "structured-pruning", "knowledge-distillation", "latency", "throughput"],
  sections: {
    intro: [
      {
        t: "p",
        text: "A trained network is usually bigger and slower than needed: its weights are 32-bit floats though a few bits would do, many are near zero, and its capacity often exceeds the task. **Model optimization** (or model compression) exploits this with three tools:",
      },
      {
        t: "list",
        items: [
          "**Quantization** — store and compute with low-precision integers instead of floats.",
          "**Pruning** — remove weights or whole neurons.",
          "**Knowledge distillation** — train a small *student* to imitate a large *teacher*.",
        ],
      },
      {
        t: "p",
        text: "Why? Deployment cost: memory (a phone, a browser, a small GPU), latency (an interactive product), throughput per dollar (a data centre) and energy (batteries). Every technique trades some accuracy for these gains, so **measure**: held-out accuracy before and after, model size, and latency/throughput on the target hardware. We do exactly that for a small MLP on scikit-learn's digits, so every claim has a reproducible number.",
      },
      {
        t: "callout",
        kind: "key",
        text: "Quantize: $q = \\mathrm{clip}(\\mathrm{round}(x/s) + z,\\ q_{min},\\ q_{max})$, $\\hat x = s\\,(q - z)$. Prune: zero the weights (or neurons) with the smallest magnitude. Distil: train the student on $\\alpha\\,\\mathrm{CE}(y, p_s) + (1-\\alpha)T^2\\,\\mathrm{KL}(p_t^{(T)}\\,\\|\\,p_s^{(T)})$.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — what latency and throughput are, and the serving path these optimizations plug into.",
          "[Multilayer networks](/learn/m03/multilayer-networks) — weight matrices, output channels (rows) and hidden neurons: what per-channel quantization and structured pruning act on.",
          "[Distillation, Curriculum, Meta-, Few- and Zero-Shot Learning](/learn/m15/learning-paradigms) — the theory of distillation (soft targets, temperature); here we only apply the recipe.",
          "[Regularization](/learn/m07/regularization) — L1 penalties make weights sparse, the starting point of magnitude pruning.",
          "[Information theory](/learn/m02/information-theory) — the KL divergence in the distillation loss.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Rulers, gardens and apprentices",
        text: "**Quantization** is measuring with a coarser ruler: only 16 or 256 ticks across the range you care about, and every length reported as the nearest tick. If the ruler is too long (range set by one huge outlier), ticks are far apart and small lengths all round to zero — so choosing the range (per channel, with clipping) matters. **Pruning** is gardening: cut the weakest branches and the tree keeps its shape; cut too many and it needs time to regrow (fine-tuning). **Distillation** is an apprenticeship: the master says not just \"this is a 3\" but \"a 3 that looks a bit like an 8\" — graded opinions carry more information than a bare label.",
      },
      {
        t: "p",
        text: "Why do networks tolerate this? Training finds fairly flat solutions where small weight changes barely move outputs, and rounding error acts like a little noise per weight. Many weights are tiny, so removing them changes little. A big model's predictions are richer than one-hot labels, so a small model learns faster from them. Each tolerance has a breaking point; the experiments below find it.",
      },
    ],
    mechanism: [
      { t: "h", text: "Quantization" },
      {
        t: "steps",
        items: [
          { title: "1. Choose the bit-width b", text: "8 bits give 256 levels; 4 bits 16; 2 bits 4." },
          { title: "2. Choose the range [α, β] (calibration)", text: "Min/max of the weights, or a clipped range (percentile or MSE-optimal) ignoring outliers. For activations, estimate it from a few hundred calibration inputs (static) or from each batch at run time (dynamic)." },
          { title: "3. Compute scale s and zero-point z", text: "Asymmetric (affine): integer grid 0…2ᵇ−1, s = (β−α)/(2ᵇ−1), z = round(−α/s) so real 0 maps exactly to an integer. Symmetric: grid −(2ᵇ⁻¹−1)…2ᵇ⁻¹−1, z = 0, s = max|x|/(2ᵇ⁻¹−1)." },
          { title: "4. Quantize", text: "q = clip(round(x/s) + z). Inside the range, rounding error is at most s/2; values outside are clipped (bigger error)." },
          { title: "5. Compute in integers", text: "Products of int8 values add up in int32; one float multiplication by sₓ·s_w at the end rescales the result. Zero-points add correction terms, which is why weights are usually symmetric." },
          { title: "6. Dequantize (or requantize)", text: "x̂ = s(q − z) for the next float operation, or requantize straight to the next layer's int8 scale." },
          { title: "7. Granularity", text: "Per-tensor: one (s, z) for the whole weight matrix. Per-channel: one per output channel (row), so rows with small weights get a fine grid. Group-wise: one per block of e.g. 64–128 weights (used for 4-bit LLM weights)." },
          { title: "8. PTQ or QAT", text: "Post-training quantization (PTQ) rounds a trained model, optionally with calibration. Quantization-aware training (QAT) inserts fake quantization (quantize→dequantize) in the forward pass and fine-tunes, passing gradients through the rounding with the straight-through estimator." },
        ],
      },
      { t: "h", text: "Pruning" },
      {
        t: "list",
        items: [
          "**Unstructured (magnitude) pruning**: zero the individual weights with the smallest |w|, globally or per layer. High sparsity is possible, but dense hardware gets no faster without sparse kernels or storage.",
          "**Structured pruning**: remove whole rows/neurons, channels, attention heads or layers (ranked e.g. by L2 norm). You get a truly smaller dense model, faster everywhere, but accuracy drops sooner.",
          "**Fine-tune after pruning** with the mask held fixed (`torch.nn.utils.prune` keeps a `weight_mask` buffer and recomputes `weight = weight_orig * weight_mask` on every forward). Pruning and fine-tuning in small rounds reaches higher sparsity than one big cut.",
          "`prune.remove` makes pruning permanent: it deletes the mask and `weight_orig`, leaving a plain `weight` that contains zeros.",
        ],
      },
      { t: "h", text: "Distillation recipe (applied)" },
      {
        t: "list",
        ordered: true,
        items: [
          "Train (or get) an accurate teacher and freeze it.",
          "Pick a student architecture that fits the deployment budget.",
          "Compute teacher logits on a **transfer set** — the labelled training data plus, ideally, extra unlabelled inputs from the same distribution.",
          "Train the student on α·CE(hard labels) + (1−α)·T²·KL(softmax(teacher/T) ‖ softmax(student/T)), starting with T ≈ 2–8 and α ≈ 0.1–0.5. The T² factor keeps soft-target gradients the same size as T changes.",
          "Compare with a label-only student from the same initialization — that difference is distillation's real value.",
        ],
      },
      { t: "h", text: "Measuring latency and throughput" },
      {
        t: "list",
        items: [
          "Warm up first (allocations, caches, lazy initialization), then time many repetitions and report the **median** and a tail percentile (p95/p99), not the mean of a few runs.",
          "Fix the thread count and keep the machine idle. On GPUs, call `torch.cuda.synchronize()` before reading the clock — kernels run asynchronously.",
          "Measure at the batch sizes production uses: latency at batch 1 and throughput at big batches can rank two models differently.",
          "Measure on the target hardware: int8 speed-ups depend on CPU instructions (e.g. VNNI/AMX on x86, dot-product instructions on ARM) and on the back end.",
        ],
      },
    ],
    math: [
      { t: "p", text: "For a real value $x$, bit-width $b$ and range $[\\alpha, \\beta]$ (extended to contain 0), the **asymmetric (affine)** quantizer is:" },
      { t: "math", tex: String.raw`s = \frac{\beta - \alpha}{2^b - 1},\qquad z = \mathrm{round}\!\left(-\frac{\alpha}{s}\right),\qquad q = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right) + z,\ 0,\ 2^b - 1\right),\qquad \hat x = s\,(q - z)` },
      { t: "p", text: "$s$ is the **scale** (the real size of one integer step), $z$ the **zero-point** (the integer that stands for real 0), $q$ the stored integer, and $\\hat x$ the value you get back. The **symmetric** quantizer fixes $z = 0$ and uses a signed grid:" },
      { t: "math", tex: String.raw`s = \frac{\max|x|}{2^{b-1} - 1},\qquad q = \mathrm{clip}\!\left(\mathrm{round}(x/s),\ -(2^{b-1}-1),\ 2^{b-1}-1\right),\qquad \hat x = s\,q` },
      { t: "p", text: "Inside the range, rounding moves $x/s$ by at most $1/2$, so $|x - \\hat x| \\le s/2$. Modelled as uniform on $[-s/2, s/2]$, the error has variance $s^2/12$. For a signal filling the full range, this gives the classic rule: each extra bit improves the **signal-to-quantization-noise ratio** (SQNR) by about 6.02 dB:" },
      { t: "math", tex: String.raw`\mathrm{SQNR} = 10\log_{10}\frac{\sum_i x_i^2}{\sum_i (x_i - \hat x_i)^2}\ \text{dB},\qquad \mathrm{SQNR}_{\text{uniform full-scale}} \approx 6.02\,b + 1.76\ \text{dB}` },
      { t: "p", text: "Real weights are bell-shaped with heavy tails, so they use the grid poorly and reach much lower SQNR (our Laplacian rows: about 13–17 dB at 4 bits instead of 25.8). Clipping at $c < \\max|x|$ shrinks $s$ — less rounding error for the many small values — at the cost of clipping error for the few large ones. The best $c$ balances the two." },
      { t: "p", text: "**Integer matrix multiplication.** With symmetric quantization, $x_j \\approx s_x q_{x,j}$ and $W_{jk} \\approx s_w q_{w,jk}$:" },
      { t: "math", tex: String.raw`y_k = \sum_j x_j W_{jk} \approx s_x s_w \underbrace{\sum_j q_{x,j}\,q_{w,jk}}_{\text{int32 accumulator}}` },
      { t: "p", text: "In words: do the whole dot product in integers, then multiply once by the two scales. With per-channel weights $s_w$ becomes $s_{w,k}$ — still one multiplication per output. Asymmetric activations add a term $-z_x\\sum_j q_{w,jk}$ that can be precomputed." },
      { t: "p", text: "**Straight-through estimator.** $\\mathrm{round}$ has zero gradient almost everywhere, so QAT uses $\\hat w = w + \\mathrm{sg}(Q(w) - w)$, where $\\mathrm{sg}$ stops gradients. The forward pass sees $Q(w)$; the backward pass treats $\\partial \\hat w/\\partial w = 1$." },
      { t: "p", text: "**Distillation loss** with temperature $T$, teacher logits $u$, student logits $v$, labels $y$:" },
      { t: "math", tex: String.raw`\mathcal L = \alpha\,\mathrm{CE}\big(y, \mathrm{softmax}(v)\big) + (1-\alpha)\,T^2\,\mathrm{KL}\big(\mathrm{softmax}(u/T)\,\|\,\mathrm{softmax}(v/T)\big)` },
      { t: "p", text: "**Sparsity and size.** Pruning a fraction $\\rho$ of $P$ weights leaves $(1-\\rho)P$ non-zeros. Stored sparsely (value + index), this saves memory only when $\\rho$ is large, because each kept weight also needs an index." },
      { t: "h", text: "Worked example: 8-bit asymmetric quantization of five weights" },
      {
        t: "steps",
        items: [
          { title: "Weights and range", tex: String.raw`w = [-0.62,\ -0.10,\ 0,\ 0.31,\ 0.95],\quad \alpha = -0.62,\ \beta = 0.95` },
          { title: "Scale", tex: String.raw`s = \frac{0.95 - (-0.62)}{255} = \frac{1.57}{255} = 0.00615686` },
          { title: "Zero-point", tex: String.raw`z = \mathrm{round}(0.62 / 0.00615686) = \mathrm{round}(100.7006) = 101` },
          { title: "Quantize 0.31", tex: String.raw`0.31/s = 50.3503;\ \ 50.3503 + 101 = 151.3503 \to q = 151` },
          { title: "Dequantize and error", tex: String.raw`\hat w = 0.00615686\,(151 - 101) = 0.307843,\quad |0.31 - 0.307843| = 0.002157 \le s/2 = 0.003078` },
          { title: "Endpoints", tex: String.raw`-0.62 \to q = \mathrm{round}(0.2994) = 0 \to \hat w = -0.621843;\quad 0.95 \to 255.2994 \to q = 255 \to \hat w = 0.948157` },
          { title: "Real zero is exact", tex: String.raw`0 \to q = z = 101 \to \hat w = 0` },
          { title: "Same value, 4-bit symmetric (s = 0.95/7)", tex: String.raw`s = 0.135714,\quad 0.31/s = 2.2842 \to q = 2 \to \hat w = 0.271429,\ \text{error } 0.038571` },
        ],
      },
      { t: "p", text: "Because $z$ was rounded, the grid shifts slightly: $0.95$ comes back as $0.948157$ — still within $s/2$. The 4-bit symmetric error (0.0386) is about 18× the 8-bit one because the scale grew 22× (from 0.0062 to 0.1357)." },
      { t: "h", text: "Worked example: distillation soft targets" },
      {
        t: "p",
        text: "Teacher logits $u = (4, 1, 0)$. At $T = 1$ the softmax is $(0.9362, 0.0466, 0.0171)$ — almost one-hot. At $T = 4$ it is $(0.5434, 0.2567, 0.1999)$, revealing that class 2 is judged more similar than class 3. For student logits $v = (3, 1.5, 0.5)$, $\\mathrm{softmax}(v/4) = (0.4499, 0.3092, 0.2408)$ and $\\mathrm{KL} = 0.017533$ nats. Times $T^2 = 16$, the soft term contributes $0.2805$.",
      },
    ],
    visual: [
      { t: "sim", id: "quantization", caption: "Quantize a real weight matrix and watch scale, zero-point, rounding, clipping and the error — and whether a tiny classifier changes its decisions." },
      {
        t: "list",
        items: [
          "Press Step to walk one weight through range → scale → zero-point → divide → shift → round → clip → dequantize → error, checking each number against the formulas.",
          "Set 8 bits: the dequantized dots sit on the floats and all classifier decisions match. Drop to 4, 3, then 2 bits: SQNR falls by roughly 6 dB per bit and decisions start to flip.",
          "At 4 bits or fewer, switch per-tensor → per-channel. In per-tensor mode the row with small weights rounds almost entirely to zero; per-channel gives it its own fine grid and its error drops sharply.",
          "Lower the clip ratio below 1: the grid gets finer (smaller scale) and SQNR often improves — until clipping the largest weights takes over and it drops again.",
          "Toggle symmetric/asymmetric for a row of mostly positive weights: asymmetric uses the whole integer range; symmetric wastes half of it.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "Everything from first principles in NumPy: the quantizers, a step-by-step example, per-tensor vs per-channel SQNR on rows of very different scales, a clipping search, weight-only PTQ of a trained digits MLP at 8/4/3/2 bits, an integer-only matmul, and magnitude (unstructured) and neuron (structured) pruning.",
      },
      { t: "code", lang: "python", title: "compress_numpy.py", code: NUMPY_CODE, output: NUMPY_OUT },
      {
        t: "list",
        items: [
          "8-bit weight quantization costs nothing measurable (0.9711 → 0.9711). Accuracy holds down to 3–4 bits. At 2 bits, per-tensor collapses to 0.3978 while per-channel keeps 0.8400.",
          "The 3-bit row (per-channel 0.9511 below per-tensor 0.9689) is a reminder: with 450 test images, one image is 0.22 points, so gaps of one or two points between nearby settings are noise. The 2-bit gap is not.",
          "Row 0's weights are 100× smaller than row 3's. With one 4-bit scale for the whole matrix, every weight in row 0 rounds to 0 (fraction 1.000). Per-channel scales fix this.",
          "Searching the clip ratio improves 4-bit SQNR by about 1.0–1.6 dB per row (best ratios 0.69–0.87 of max|w|).",
          "Unstructured pruning without fine-tuning: 50% of weights can go (0.9756), 80% costs 5.5 points, 90% breaks the model. Structured pruning keeps accuracy down to 32 of 64 neurons, then collapses.",
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
              { t: "p", text: "A 64-256-256-10 MLP (84,480 weights): dynamic int8 quantization with `torch.ao.quantization.quantize_dynamic`, PTQ vs QAT at 4/3/2 bits with a hand-written fake-quant layer and straight-through estimator, global unstructured L1 pruning with and without fine-tuning, and structured pruning with `torch.nn.utils.prune`." },
              { t: "code", lang: "python", title: "compress_torch.py", code: TORCH_CODE, output: TORCH_OUT },
              {
                t: "list",
                items: [
                  "Dynamic int8: same accuracy and 100% top-1 agreement with fp32; state_dict 334.6 KB → 89.1 KB (3.8×, not exactly 4× because biases, scales and packing metadata stay larger).",
                  "PTQ vs QAT: at 4 bits both are fine; at 3 bits QAT recovers 1.8 points; at 2 bits PTQ collapses (0.2978) while 100 QAT steps bring the model back to 0.9622.",
                  "Pruning + 50 fine-tune steps: 80% sparsity recovers from 0.7933 to 0.9667; 90% only to 0.8578; 95% stays broken. The mask is fixed, so fine-tuning can't regrow pruned connections.",
                  "Removing 75% of the first layer's neurons in one shot without fine-tuning drops accuracy to 0.4978. Structured pruning hurts more per removed weight, but the result can be rebuilt as a truly smaller dense layer.",
                  "`torch.ao.quantization` prints a deprecation warning in this PyTorch version (2.15), pointing to the separate `torchao` library for new code; the eager dynamic API still works on this CPU (quantized engine: onednn).",
                ],
              },
              { t: "p", text: "Applied knowledge distillation. The theory (soft targets, temperature, why it works) is in [Distillation, Curriculum, Meta-, Few- and Zero-Shot Learning](/learn/m15/learning-paradigms). Here is the recipe with a fair comparison: same student, same initialization, three training signals." },
              { t: "code", lang: "python", title: "distill_torch.py", code: KD_CODE, output: KD_OUT },
              {
                t: "p",
                text: "With only 100 labels, the 1,210-parameter student reaches 0.8985. Distilling on the same 100 images gives 0.9541; adding 1,347 unlabelled transfer images gives 0.9748 — close to the 38,410-parameter teacher (0.9822) with 32× fewer parameters. Caveat: the teacher was trained on all 1,347 labels, so part of the gain is label information flowing through it. That is also the realistic setting: an expensive model trained on a big labelled set, squeezed into a cheap one using plentiful unlabelled inputs.",
              },
              { t: "p", text: "Latency and throughput, fp32 vs dynamic int8, on a larger 512-2048-2048-10 MLP. Timings depend on the machine and its load, so this block is not checked byte for byte; the output below is from one real run." },
              { t: "code", lang: "python", title: "benchmark.py (timings vary between runs)", code: TIME_CODE, output: TIME_OUT },
              { t: "p", text: "In our runs int8 was about 2.5–4× faster than fp32 at every batch size on this CPU. Throughput and per-call latency both grew with batch size — the serving lesson's trade-off. p95 sits well above the median at small batches: tail latency comes from the OS and caches, not the model." },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "Full-integer post-training quantization with the TFLite converter, calibrated on a representative dataset. Pruning and QAT in Keras live in the separate TensorFlow Model Optimization Toolkit (`tensorflow_model_optimization`)." },
              { t: "code", lang: "python", title: "tflite_int8.py (not executed in CI)", code: TFLITE_CODE, output: "Prints the size in bytes of the int8 TFLite flatbuffer, roughly a quarter of the float32 model size." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Technique", "Pros", "Cons", "Use cases"],
        rows: [
          ["Dynamic quantization (weights int8, activations quantized per batch)", "No calibration data, one line, good for Linear/LSTM on CPU", "Run-time cost of quantizing activations; limited op coverage", "Transformer/RNN inference on CPU"],
          ["Static PTQ (weights + calibrated activation ranges)", "Fastest integer inference, no training", "Needs representative calibration data; sensitive to outliers", "CNNs on CPUs, mobile, edge accelerators"],
          ["Quantization-aware training", "Best accuracy at low bit-widths", "Needs a training pipeline and data; slower", "4-bit and below, accuracy-critical int8"],
          ["Weight-only low-bit (group-wise 4-bit, e.g. GPTQ/AWQ-style)", "Big memory savings for huge models", "Compute still in fp16 after dequantization; speed-up depends on kernels", "Memory-bound LLM decoding"],
          ["Half precision (fp16 / bf16)", "2× smaller, near-lossless, native on GPUs", "Only 2×; fp16 range issues", "Default GPU inference"],
          ["Unstructured pruning", "High sparsity, small loss", "Needs sparse kernels/hardware to be faster", "Storage compression, sparse accelerators"],
          ["Structured pruning (neurons, channels, heads)", "Real speed-up on any hardware", "More accuracy lost per parameter removed", "Shrinking CNN channels, transformer heads/layers"],
          ["Knowledge distillation", "Free choice of architecture; can use unlabelled data", "Needs a teacher and a training run", "Small production models (e.g. DistilBERT-style students)"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "Mobile and embedded runtimes (TensorFlow Lite / LiteRT, PyTorch ExecuTorch, Core ML) rely on int8 or fp16 models to fit memory and battery budgets.",
          "Large language models are routinely served with 8-bit or 4-bit weights so they fit on fewer GPUs. QLoRA fine-tunes adapters on top of a frozen 4-bit base model (see [Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency)).",
          "DistilBERT (Sanh et al., 2019) is a well-known distilled student of BERT: about 40% smaller while keeping most of its accuracy on language-understanding benchmarks.",
          "NVIDIA Ampere-generation GPUs support 2:4 structured sparsity (two zeros in every group of four weights) in their tensor cores — a hardware-friendly form of pruning.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "table",
        head: ["", "Size", "Speed", "Accuracy risk", "Effort"],
        rows: [
          ["fp16/bf16", "2×", "Large on GPU", "Very low", "Minimal"],
          ["int8 dynamic", "≈4× weights", "CPU: 2.5–4× here", "Very low (0 points here)", "One line"],
          ["int8 static", "4×", "Highest on int8 hardware", "Low with good calibration", "Calibration set"],
          ["4-bit and below", "8×+", "Kernel-dependent", "Medium–high without QAT", "QAT or advanced PTQ"],
          ["Unstructured pruning 80–90%", "Only with sparse storage", "Usually none on dense hardware", "Needs fine-tuning", "Moderate"],
          ["Structured pruning", "Proportional", "Proportional", "Higher", "Moderate, needs fine-tuning"],
          ["Distillation", "Any (new architecture)", "Any", "Depends on student capacity", "Full training run"],
        ],
      },
      {
        t: "p",
        text: "The techniques stack: distil into a smaller architecture, prune, then quantize — validating each stage so you know which step cost accuracy. Fewer FLOPs or parameters don't guarantee wall-clock gains; only a benchmark on the target device does.",
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Per-tensor scales for weights whose rows differ a lot in size.** Small rows round to zero (our row 0: 100%). Fix: per-channel (or group-wise) weight scales.",
          "**Calibrating on unrepresentative data.** Ranges from a few odd inputs cause clipping or a coarse grid on real traffic. Fix: a few hundred typical inputs; consider percentile/MSE calibration.",
          "**Forgetting that real 0 must be exact.** Padding and ReLU outputs are exactly 0; a non-integer zero-point adds a systematic bias. Fix: round the zero-point and extend the range to include 0 (as our `qparams` does).",
          "**Expecting unstructured sparsity to speed up dense inference.** The zeros still get multiplied. Fix: structured pruning, sparse kernels, or count it only as a storage gain.",
          "**Pruning without fine-tuning and deciding the model can't be pruned.** Here 80% sparsity cost 17.6 points before, and 0.2 after 50 fine-tune steps.",
          "**Comparing a distilled student to a weaker baseline.** Give the label-only student the same architecture, initialization and data, or the gain can't be credited to distillation.",
          "**Benchmarking without warm-up, with a mean of 3 runs, or with unsynchronized GPU timing.** Fix: warm up, many repetitions, median and p95/p99, `torch.cuda.synchronize()`.",
          "**Measuring only accuracy.** Quantization can keep top-1 accuracy but shift confidences (calibration) or hurt rare classes. Also compare per-class metrics and output agreement.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What are the scale and zero-point in quantization?", a: "The scale s is the real size of one integer step; the zero-point z is the integer that stands for real 0. A real x maps to q = clip(round(x/s) + z) and back to x̂ = s(q − z). Symmetric quantization fixes z = 0; asymmetric picks z so the integer grid covers an off-centre range, such as post-ReLU activations." },
      { t: "qa", level: "beginner", q: "What is the difference between unstructured and structured pruning?", a: "Unstructured pruning zeroes individual weights (usually the smallest): high sparsity, little accuracy loss, but the matrix keeps its shape, so dense hardware is no faster. Structured pruning removes whole neurons, channels, heads or layers, giving a smaller dense model that is faster everywhere but usually loses more accuracy per removed parameter." },
      { t: "qa", level: "beginner", q: "What is knowledge distillation in one sentence, and what is the temperature for?", a: "A small student is trained to match the softened output distribution of a large teacher (plus the true labels). Dividing logits by T > 1 before the softmax spreads probability onto the non-top classes, exposing which classes the teacher sees as similar; the loss is multiplied by T² to keep the gradient scale comparable." },
      { t: "qa", level: "intermediate", q: "Why is per-channel quantization preferred for weights but per-tensor common for activations?", a: "Output channels of a weight matrix can differ in size by orders of magnitude, and one shared scale wastes resolution on the small ones. Per-channel weight scales are free at run time: each output is rescaled once by its own s_w,k after the integer dot product. For activations, a per-channel scale along the reduction dimension would vary inside the dot product and couldn't be factored out, so activations usually use per-tensor (or per-token) scales." },
      { t: "qa", level: "intermediate", q: "Compare PTQ and QAT.", a: "PTQ quantizes a trained model directly, optionally calibrating activation ranges on sample data: fast, no training, fine at 8 bits. QAT simulates quantization during fine-tuning with fake-quant ops and the straight-through estimator, so weights move to spots that survive rounding. It costs a training run but is far better at low bit-widths. In our experiment at 2 bits, PTQ gave 0.2978 accuracy and QAT 0.9622." },
      { t: "qa", level: "intermediate", q: "How would you benchmark whether int8 actually speeds up your model?", a: "On the target hardware, with fixed threads and an idle machine: warm up, time many repetitions at production batch sizes, and report median and p95/p99 latency plus throughput for fp32 and int8 versions of the same pipeline (pre/postprocessing included). On GPU, synchronize before reading timers. Confirm held-out accuracy and check int8 kernels are really used (the back end may fall back to float for unsupported ops)." },
      { t: "qa", level: "advanced", q: "Explain the straight-through estimator and its failure modes.", a: "Rounding has zero derivative almost everywhere, so backprop through it gives no signal. The STE uses the quantized value forward but treats the quantizer as the identity backward (ŵ = w + stop_grad(Q(w) − w)). It is biased: weights near a rounding boundary can flip between two levels, gradients ignore clipping unless zeroed outside the range, and at very low bit-widths the forward/backward mismatch can destabilize training. Remedies: learning the scale (LSQ-style), freezing oscillating weights, lower learning rates, and lowering the bit-width gradually." },
      { t: "qa", level: "advanced", q: "Your int8 model keeps accuracy on average but a few classes degrade badly. What could cause it and how do you fix it?", a: "Likely causes: activation outliers in some channels forcing a coarse per-tensor scale; calibration data under-representing those classes; or a sensitive layer (often first or last) losing too much precision. Diagnose by quantizing one layer at a time and measuring per-class accuracy or output SQNR. Fixes: per-channel or finer scales, better calibration (more representative data, percentile/MSE ranges), keeping sensitive layers in higher precision (mixed precision), outlier-smoothing transforms that shift scale between activations and weights, or QAT." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "8-bit asymmetric quantization of the range [−1, 3]. What is the scale s? (4 decimals)", answer: 0.0157, tolerance: 0.0001, explain: "s = (3 − (−1))/255 = 4/255 = 0.015686." },
          { kind: "numeric", q: "For the same range [−1, 3] at 8 bits, what is the zero-point z?", answer: 64, tolerance: 0.01, explain: "z = round(1/0.015686) = round(63.75) = 64." },
          { kind: "numeric", q: "4-bit symmetric quantization with max|w| = 0.7. What is the dequantized value of w = 0.27? (2 decimals)", answer: 0.3, tolerance: 0.0001, explain: "s = 0.7/7 = 0.1; 0.27/0.1 = 2.7 → q = 3 → ŵ = 0.3 (error 0.03 ≤ s/2 = 0.05)." },
          { q: "Which granularity gives the row with tiny weights its own fine grid?", options: ["Per-tensor", "Per-channel", "Dynamic", "Symmetric"], answer: 1, explain: "One scale per output channel (row)." },
          { q: "Unstructured 90% sparsity on a dense GPU kernel typically gives…", options: ["about 10× speed-up", "no speed-up by itself", "a 2× slow-down always", "lower memory automatically"], answer: 1, explain: "Zeros are still stored and multiplied unless sparse formats/kernels are used." },
          { kind: "numeric", q: "Using the 6.02b + 1.76 dB rule, how many dB of SQNR does a full-scale uniform signal reach at 8 bits? (2 decimals)", answer: 49.92, tolerance: 0.01, explain: "6.02 × 8 + 1.76 = 49.92 dB." },
          { q: "Why is the soft-target term in distillation multiplied by T²?", options: ["To make it a valid probability", "Because softening by T scales its gradients by about 1/T², so T² restores their magnitude", "To normalise the cross-entropy", "It has no effect"], answer: 1, explain: "Hinton et al. (2015): gradients of the softened term scale as 1/T²." },
          { q: "In our experiment, what did QAT achieve at 2 bits compared with PTQ?", options: ["Both ≈ 0.30", "PTQ 0.96, QAT 0.30", "PTQ 0.30, QAT 0.96", "Both ≈ 0.97"], answer: 2, explain: "PTQ 0.2978, QAT 0.9622 after 100 fine-tuning steps." },
        ],
      },
      {
        t: "exercise",
        title: "Quantize by hand",
        prompt: "Quantize w = [−0.38, 0.05, 0.22, 0.8] with (a) 4-bit asymmetric per-tensor and (b) 4-bit symmetric per-tensor. For each, give s, z, the integers, the dequantized values and the max error. Which is better here, and why?",
        hint: "Asymmetric: 15 integer steps across [−0.38, 0.8]. Symmetric: q ∈ [−7, 7], s = 0.8/7.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) scale, zero-point", tex: String.raw`s = 1.18/15 = 0.078667,\quad z = \mathrm{round}(0.38/0.078667) = \mathrm{round}(4.8305) = 5` },
              { title: "(a) integers", tex: String.raw`q = \mathrm{round}([-4.8305,\ 0.6356,\ 2.7966,\ 10.1695]) + 5 = [0,\ 6,\ 8,\ 15]` },
              { title: "(a) dequantized", tex: String.raw`\hat w = 0.078667\,(q - 5) = [-0.393333,\ 0.078667,\ 0.236,\ 0.786667],\quad \max|e| = 0.028667 \le s/2 = 0.039333` },
              { title: "(b) symmetric", tex: String.raw`s = 0.8/7 = 0.114286,\quad q = \mathrm{round}([-3.325,\ 0.4375,\ 1.925,\ 7]) = [-3,\ 0,\ 2,\ 7]` },
              { title: "(b) dequantized", tex: String.raw`\hat w = [-0.342857,\ 0,\ 0.228571,\ 0.8],\quad \max|e| = 0.05 \le s/2 = 0.057143` },
            ],
          },
          { t: "p", text: "Asymmetric wins here (max error 0.0287 vs 0.05) because the weights lean positive. Symmetric reserves levels −7…−4 for [−0.8, −0.4), where there are no values, so its step is coarser. Note the asymmetric grid, shifted by the rounded zero-point, no longer hits the endpoints exactly (0.8 → 0.786667, −0.38 → −0.393333)." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: iterative pruning",
        prompt: "Change `compress_torch.py` to prune in rounds: 5 rounds, each pruning 37% of the remaining weights globally (≈90% after 5 rounds, since 0.63⁵ ≈ 0.099), each followed by 20 fine-tuning steps. Compare the final accuracy with one-shot 90% pruning plus 100 fine-tuning steps (same total steps).",
        hint: "Calling `prune.global_unstructured` again on an already-pruned module combines the new mask with the old one; `amount` refers to the weights not yet pruned.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "it = copy.deepcopy(fp32)\nparams = [(m, \"weight\") for m in it if isinstance(m, nn.Linear)]\nfor r in range(5):\n    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.37)\n    torch.manual_seed(r)\n    train(it, steps=20, lr=1e-3)\n    zeros = sum((m.weight == 0).sum().item() for m, _ in params)\n    print(f\"round {r}: sparsity {zeros / n_weights:.3f}  accuracy {acc(it):.4f}\")\n\none = copy.deepcopy(fp32)\np1 = [(m, \"weight\") for m in one if isinstance(m, nn.Linear)]\nprune.global_unstructured(p1, pruning_method=prune.L1Unstructured, amount=0.9)\ntrain(one, steps=100, lr=1e-3)\nprint(\"one-shot 90% + 100 steps:\", acc(one))",
          },
          { t: "p", text: "Expect pruning in rounds to beat one-shot at the same sparsity: after each moderate cut, the network shifts its work onto surviving weights, so the magnitude criterion keeps ranking weights that really matter. Run it and compare with the one-shot 90% result (0.8578 after 50 steps in the lesson's program)." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Affine: $s = (\\beta-\\alpha)/(2^b-1)$, $z = \\mathrm{round}(-\\alpha/s)$, $q = \\mathrm{clip}(\\mathrm{round}(x/s)+z)$, $\\hat x = s(q-z)$; symmetric: $z = 0$, $s = \\max|x|/(2^{b-1}-1)$. Error $\\le s/2$ inside the range.",
          "SQNR ≈ 6.02 dB per bit for full-scale uniform signals; heavy-tailed weights do worse; clipping trades rounding error for clipping error.",
          "Per-channel weight scales rescue small rows; integer matmul accumulates in int32 and rescales once by $s_x s_w$.",
          "Measured on digits: int8 dynamic = same accuracy, 3.8× smaller; 2-bit PTQ collapses (0.30) but QAT recovers (0.96).",
          "Pruning: 50% free, 80% needs fine-tuning (0.79 → 0.97), 90%+ breaks; structured pruning is harsher but gives real speed-ups.",
          "Distillation recipe: frozen teacher, transfer set (incl. unlabelled data), loss $\\alpha\\mathrm{CE} + (1-\\alpha)T^2\\mathrm{KL}$; here 0.90 → 0.97 for a 32× smaller student.",
          "Benchmark with warm-up, many repetitions, median + p95, fixed threads, on target hardware; timings vary between runs.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Monitoring, Drift, CI/CD and Versioning](/learn/m18/mlops) — a compressed model is a new model version: test it, register it and monitor it like any other.",
          "[Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency) — 4-bit quantization and low-rank adapters for fine-tuning very large models.",
          "[Distillation, Curriculum, Meta-, Few- and Zero-Shot Learning](/learn/m15/learning-paradigms) — the theory behind the distillation recipe used here.",
        ],
      },
    ],
  },
};

export default lesson;
