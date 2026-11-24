import type { Lesson } from "@/lib/types";

const CHECK_MATH = String.raw`import numpy as np
import torch
import torch.nn.functional as F

T = lambda a: torch.tensor(np.asarray(a, dtype=float))

# Example 1: dilated 1-D, x = [3 1 4 1 5 9 2 6], w = [1 2 -1], d = 2
y = F.conv1d(T([3, 1, 4, 1, 5, 9, 2, 6]).view(1, 1, -1), T([1, 2, -1]).view(1, 1, -1), dilation=2)
print("dilated 1-D:", y.view(-1).tolist())

# Example 2: dilated 2-D, X[i, j] = (i + 2j) mod 5 (7x7), Sobel-x, d = 2
X = np.fromfunction(lambda i, j: (i + 2 * j) % 5, (7, 7))
K = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
print("dilated 2-D:", F.conv2d(T(X)[None, None], T(K)[None, None], dilation=2)[0, 0].tolist())

# Example 3: transposed 1-D, x = [1 2 3], w = [1 2 1], s = 2
print("transposed 1-D:", F.conv_transpose1d(T([1, 2, 3]).view(1, 1, -1), T([1, 2, 1]).view(1, 1, -1), stride=2).view(-1).tolist())
print("overlap count :", F.conv_transpose1d(T([1, 1, 1]).view(1, 1, -1), T([1, 1, 1]).view(1, 1, -1), stride=2).view(-1).tolist())

# Example 4: the conv matrix of w = [1 2 3], stride 2, input length 5
C = np.array([[1, 2, 3, 0, 0], [0, 0, 1, 2, 3]], dtype=float)
x, g = np.array([1, 0, 2, 1, 3.0]), np.array([1, -1.0])
print("C x =", (C @ x).tolist(), " conv1d:", F.conv1d(T(x).view(1, 1, -1), T([1, 2, 3]).view(1, 1, -1), stride=2).view(-1).tolist())
print("C^T g =", (C.T @ g).tolist(), " conv_transpose1d:", F.conv_transpose1d(T(g).view(1, 1, -1), T([1, 2, 3]).view(1, 1, -1), stride=2).view(-1).tolist())

# Example 5: depthwise-separable on two 3x3 channels
Xs = T([[[1, 2, 0], [0, 1, 3], [2, 1, 1]], [[2, 0, 1], [1, 1, 0], [0, 2, 1]]])[None]
Dk = T([[[1, -1], [0, 2]], [[0, 1], [1, 0]]])[:, None]           # one 2x2 kernel per channel
Pw = T([[1, 1], [1, -1], [2, 0]])[:, :, None, None]               # 3 outputs x 2 inputs
dw = F.conv2d(Xs, Dk, groups=2)
print("depthwise:", dw[0].tolist())
print("pointwise:", F.conv2d(dw, Pw)[0].tolist())

# Example 6: costs, C_in = 32, C_out = 64, k = 3, 56x56
std, sep = 64 * 32 * 9, 32 * 9 + 32 * 64
print("params", std, sep, " MACs", std * 56 * 56, sep * 56 * 56, f" ratio {sep / std:.6f} = 1/64 + 1/9 = {1 / 64 + 1 / 9:.6f}")`;

const CHECK_MATH_OUT = String.raw`dilated 1-D: [6.0, -6.0, 12.0, 13.0]
dilated 2-D: [[3.0, 3.0, -2.0], [-2.0, -2.0, 3.0], [3.0, -2.0, -2.0]]
transposed 1-D: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0]
overlap count : [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0]
C x = [7.0, 13.0]  conv1d: [7.0, 13.0]
C^T g = [1.0, 2.0, 2.0, -2.0, -3.0]  conv_transpose1d: [1.0, 2.0, 2.0, -2.0, -3.0]
depthwise: [[[1.0, 8.0], [1.0, 0.0]], [[1.0, 2.0], [1.0, 2.0]]]
pointwise: [[[2.0, 10.0], [2.0, 2.0]], [[0.0, 6.0], [0.0, -2.0]], [[2.0, 16.0], [2.0, 0.0]]]
params 18432 2336  MACs 57802752 7325696  ratio 0.126736 = 1/64 + 1/9 = 0.126736`;

const SCRATCH = String.raw`import numpy as np
import torch                        # used only to check our results
import torch.nn.functional as F

rng = np.random.default_rng(0)
np.set_printoptions(linewidth=120)


def check(name, ours, ref):
    print(f"{name:44s} shape={str(ours.shape):16s} max|diff|={np.abs(ours - ref).max():.1e}")


# ---------------------------------------------------------------- 1. 1x1 conv
def pointwise(x, w, b=None):
    """1x1 convolution = the same linear map applied to the channel vector of every pixel.
    x: (N, C_in, H, W), w: (C_out, C_in) -> (N, C_out, H, W)"""
    out = np.einsum("oc,nchw->nohw", w, x)
    return out if b is None else out + b[None, :, None, None]


# ------------------------------------------------------- 2. dilated convolution
def conv2d_dilated(x, w, b=None, stride=1, padding=0, dilation=1):
    """Cross-correlation with kernel taps spaced dilation pixels apart.
    x: (N, C_in, H, W), w: (C_out, C_in, k, k) -> (N, C_out, H_out, W_out)"""
    N, C_in, H, W = x.shape
    C_out, _, k, _ = w.shape
    k_eff = dilation * (k - 1) + 1                      # span of the spread-out kernel
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (H + 2 * padding - k_eff) // stride + 1
    W_out = (W + 2 * padding - k_eff) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out))
    for a in range(k):                                  # loop over the k*k kernel taps
        for c in range(k):
            r0, c0 = a * dilation, c * dilation         # tap (a, c) sits d*a rows, d*c cols from the corner
            patch = xp[:, :, r0:r0 + stride * H_out:stride, c0:c0 + stride * W_out:stride]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, a, c])
    return out if b is None else out + b[None, :, None, None]


# ----------------------------------------------------- 3. transposed convolution
def conv_transpose2d(x, w, b=None, stride=1, padding=0, output_padding=0):
    """Scatter-add: every input pixel stamps a scaled copy of the kernel into the output.
    x: (N, C_in, H, W), w: (C_in, C_out, k, k)  [PyTorch's layout for transposed conv]"""
    N, C_in, H, W = x.shape
    _, C_out, k, _ = w.shape
    Hf = (H - 1) * stride + k + output_padding          # full canvas before cropping
    Wf = (W - 1) * stride + k + output_padding
    full = np.zeros((N, C_out, Hf, Wf))
    for i in range(H):
        for j in range(W):
            r, c = i * stride, j * stride               # where pixel (i, j) lands on the canvas
            full[:, :, r:r + k, c:c + k] += np.einsum("nc,cokl->nokl", x[:, :, i, j], w)
    out = full[:, :, padding:Hf - padding, padding:Wf - padding]   # 'padding' crops the border
    return out if b is None else out + b[None, :, None, None]


# ------------------------------------------------ 4. depthwise-separable convolution
def depthwise(x, w, stride=1, padding=0):
    """One k x k filter per channel, no mixing across channels. w: (C, k, k)."""
    out = [conv2d_dilated(x[:, c:c + 1], w[c][None, None], stride=stride, padding=padding)
           for c in range(x.shape[1])]
    return np.concatenate(out, axis=1)


def separable(x, w_dw, w_pw, stride=1, padding=0):
    return pointwise(depthwise(x, w_dw, stride, padding), w_pw)


T = torch.tensor
print("== correctness against torch.nn.functional ==")
x = rng.standard_normal((2, 6, 9, 9))
w1 = rng.standard_normal((4, 6))
b1 = rng.standard_normal(4)
check("1x1 conv (pointwise)", pointwise(x, w1, b1),
      F.conv2d(T(x), T(w1)[:, :, None, None], T(b1)).numpy())
lin = (x.transpose(0, 2, 3, 1) @ w1.T + b1).transpose(0, 3, 1, 2)   # Linear on each pixel
check("1x1 conv vs per-pixel Linear", pointwise(x, w1, b1), lin)

w = rng.standard_normal((4, 6, 3, 3))
b = rng.standard_normal(4)
for s, p, d in [(1, 0, 2), (1, 2, 2), (1, 3, 3), (2, 2, 2)]:
    check(f"dilated  s={s} p={p} d={d}", conv2d_dilated(x, w, b, s, p, d),
          F.conv2d(T(x), T(w), T(b), stride=s, padding=p, dilation=d).numpy())

xt = rng.standard_normal((2, 6, 4, 4))
wt = rng.standard_normal((6, 4, 3, 3))
for s, p, op in [(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1)]:
    check(f"transposed s={s} p={p} output_padding={op}", conv_transpose2d(xt, wt, b, s, p, op),
          F.conv_transpose2d(T(xt), T(wt), T(b), stride=s, padding=p, output_padding=op).numpy())

w_dw = rng.standard_normal((6, 3, 3))
w_pw = rng.standard_normal((8, 6))
ref_dw = F.conv2d(T(x), T(w_dw)[:, None], padding=1, groups=6)        # groups = C_in
check("depthwise (groups=C_in)", depthwise(x, w_dw, padding=1), ref_dw.numpy())
check("depthwise-separable", separable(x, w_dw, w_pw, padding=1),
      F.conv2d(ref_dw, T(w_pw)[:, :, None, None]).numpy())

print("\n== transposed conv is the transpose of the conv matrix ==")
k, n, s = 3, 5, 2
K = rng.choice([-2.0, -1.0, 1.0, 2.0], size=(k, k))     # small non-zero integer kernel
m = (n - k) // s + 1                                    # conv output size: 2
C = np.zeros((m * m, n * n))                            # one row per output, one column per input pixel
for i in range(m):
    for j in range(m):
        canvas = np.zeros((n, n))
        canvas[i * s:i * s + k, j * s:j * s + k] = K    # where output (i, j) looks
        C[i * m + j] = canvas.ravel()
print("C shape", C.shape, f"| non-zeros {np.count_nonzero(C)} of {C.size}")
print("kernel K:\n", K.astype(int))
print("row 1 of C (output (0,1)) reshaped to 5x5:\n", C[1].reshape(n, n).astype(int))
X = rng.standard_normal((n, n))
G = rng.standard_normal((m, m))
conv = F.conv2d(T(X)[None, None], T(K)[None, None], stride=s).numpy().ravel()
check("conv  = C @ x", C @ X.ravel(), conv)
convT = conv_transpose2d(G[None, None], K[None, None], stride=s).ravel()
check("convT = C.T @ g", C.T @ G.ravel(), convT)
Xg = T(X, requires_grad=True)
(F.conv2d(Xg[None, None], T(K)[None, None], stride=s) * T(G)).sum().backward()
check("convT = autograd dL/dx of the conv", convT, Xg.grad.numpy().ravel())
print("adjoint identity <Cx, g> = <x, C^T g>:", round(float(conv @ G.ravel()), 6), round(float(X.ravel() @ convT), 6))

print("\n== checkerboard: how many kernel stamps cover each output pixel ==")
ones_in = np.ones((1, 1, 4, 4))
cover = conv_transpose2d(ones_in, np.ones((1, 1, 3, 3)), stride=2)[0, 0]
print("transposed k=3, s=2 (9x9), rows 0-3:\n", cover[:4].astype(int))
up = ones_in.repeat(2, axis=2).repeat(2, axis=3)        # nearest-neighbour 2x upsample -> 8x8
rc = conv2d_dilated(up, np.ones((1, 1, 3, 3)) / 9, padding=1)[0, 0]
print("resize-conv (nearest x2, then 3x3 mean), interior values:", np.unique(rc[1:-1, 1:-1].round(6)))

print("\n== gridding: which input pixels reach one output of two stacked 3x3, d=2 convs ==")
inp = torch.zeros(1, 1, 9, 9, requires_grad=True)
h = F.conv2d(F.conv2d(inp, torch.ones(1, 1, 3, 3), dilation=2), torch.ones(1, 1, 3, 3), dilation=2)
h[0, 0, 0, 0].backward()
print((inp.grad[0, 0] != 0).int().numpy())`;

const SCRATCH_OUT = String.raw`== correctness against torch.nn.functional ==
1x1 conv (pointwise)                         shape=(2, 4, 9, 9)     max|diff|=1.8e-15
1x1 conv vs per-pixel Linear                 shape=(2, 4, 9, 9)     max|diff|=1.8e-15
dilated  s=1 p=0 d=2                         shape=(2, 4, 5, 5)     max|diff|=1.1e-14
dilated  s=1 p=2 d=2                         shape=(2, 4, 9, 9)     max|diff|=1.1e-14
dilated  s=1 p=3 d=3                         shape=(2, 4, 9, 9)     max|diff|=8.9e-15
dilated  s=2 p=2 d=2                         shape=(2, 4, 5, 5)     max|diff|=1.1e-14
transposed s=1 p=0 output_padding=0          shape=(2, 4, 6, 6)     max|diff|=3.6e-15
transposed s=2 p=0 output_padding=0          shape=(2, 4, 9, 9)     max|diff|=3.6e-15
transposed s=2 p=1 output_padding=0          shape=(2, 4, 7, 7)     max|diff|=3.6e-15
transposed s=2 p=1 output_padding=1          shape=(2, 4, 8, 8)     max|diff|=3.6e-15
depthwise (groups=C_in)                      shape=(2, 6, 9, 9)     max|diff|=3.6e-15
depthwise-separable                          shape=(2, 8, 9, 9)     max|diff|=7.1e-15

== transposed conv is the transpose of the conv matrix ==
C shape (4, 25) | non-zeros 36 of 100
kernel K:
 [[-1 -1  1]
 [ 2 -2  1]
 [-2  1  1]]
row 1 of C (output (0,1)) reshaped to 5x5:
 [[ 0  0 -1 -1  1]
 [ 0  0  2 -2  1]
 [ 0  0 -2  1  1]
 [ 0  0  0  0  0]
 [ 0  0  0  0  0]]
conv  = C @ x                                shape=(4,)             max|diff|=8.9e-16
convT = C.T @ g                              shape=(25,)            max|diff|=0.0e+00
convT = autograd dL/dx of the conv           shape=(25,)            max|diff|=2.2e-16
adjoint identity <Cx, g> = <x, C^T g>: 22.141827 22.141827

== checkerboard: how many kernel stamps cover each output pixel ==
transposed k=3, s=2 (9x9), rows 0-3:
 [[1 1 2 1 2 1 2 1 1]
 [1 1 2 1 2 1 2 1 1]
 [2 2 4 2 4 2 4 2 2]
 [1 1 2 1 2 1 2 1 1]]
resize-conv (nearest x2, then 3x3 mean), interior values: [1.]

== gridding: which input pixels reach one output of two stacked 3x3, d=2 convs ==
[[1 0 1 0 1 0 1 0 1]
 [0 0 0 0 0 0 0 0 0]
 [1 0 1 0 1 0 1 0 1]
 [0 0 0 0 0 0 0 0 0]
 [1 0 1 0 1 0 1 0 1]
 [0 0 0 0 0 0 0 0 0]
 [1 0 1 0 1 0 1 0 1]
 [0 0 0 0 0 0 0 0 0]
 [1 0 1 0 1 0 1 0 1]]`;

const FW_COUNTS = String.raw`import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)


def count_macs(model, x):
    """Multiply-adds of every Conv2d and Linear layer for one input, measured with forward hooks."""
    macs = {"standard": 0, "depthwise": 0, "pointwise": 0, "grouped": 0, "linear": 0}

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            per_output = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            n = out[0].numel() * per_output                 # out[0]: drop the batch axis
            if m.groups == m.in_channels and m.groups > 1:
                macs["depthwise"] += n
            elif m.groups > 1:
                macs["grouped"] += n
            elif m.kernel_size == (1, 1):
                macs["pointwise"] += n
            else:
                macs["standard"] += n
        else:
            macs["linear"] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return macs


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---- Part 1: one layer, C_in=32 -> C_out=64, 3x3, on a 56x56 map ----
x = torch.randn(1, 32, 56, 56)
layers = {
    "standard 3x3": nn.Conv2d(32, 64, 3, padding=1, bias=False),
    "grouped 3x3, groups=4": nn.Conv2d(32, 64, 3, padding=1, groups=4, bias=False),
    "depthwise-separable": nn.Sequential(
        nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),   # depthwise: groups = C_in
        nn.Conv2d(32, 64, 1, bias=False)),                        # pointwise: 1x1
}
print("Part 1: one layer, 32 -> 64 channels, 3x3, 56x56")
for name, layer in layers.items():
    macs = sum(count_macs(layer, x).values())
    print(f"  {name:24s} params {n_params(layer):6d}   MACs {macs:11,d}   output {tuple(layer(x).shape)}")
print(f"  predicted separable/standard ratio 1/64 + 1/9 = {1/64 + 1/9:.4f}")

# ---- Part 2: real architectures at 224x224 (random weights; counts only) ----
print("\nPart 2: real models, one 3x224x224 image")
img = torch.randn(1, 3, 224, 224)
for name, ctor in [("ResNet-18", torchvision.models.resnet18),
                   ("MobileNetV2", torchvision.models.mobilenet_v2)]:
    model = ctor(weights=None).eval()
    m = count_macs(model, img)
    total = sum(m.values())
    split = ", ".join(f"{k} {v / total:.1%}" for k, v in m.items() if v)
    print(f"  {name:12s} params {n_params(model) / 1e6:5.2f} M   MACs {total / 1e6:7.1f} M   ({split})")

# ---- Part 3: transposed convolution shapes in a decoder ----
print("\nPart 3: ConvTranspose2d shapes")
z = torch.randn(1, 64, 7, 7)
for k, s, p, op in [(3, 2, 1, 0), (3, 2, 1, 1), (4, 2, 1, 0), (2, 2, 0, 0)]:
    up = nn.ConvTranspose2d(64, 32, k, stride=s, padding=p, output_padding=op)
    print(f"  k={k} s={s} p={p} output_padding={op}: 7x7 -> {tuple(up(z).shape[-2:])}   weight {tuple(up.weight.shape)}")
down = nn.Conv2d(32, 64, 3, stride=2, padding=1)
print("  Conv2d(k=3, s=2, p=1) maps 14x14 ->", tuple(down(torch.randn(1, 32, 14, 14)).shape[-2:]),
      "and 13x13 ->", tuple(down(torch.randn(1, 32, 13, 13)).shape[-2:]))`;

const FW_COUNTS_OUT = String.raw`Part 1: one layer, 32 -> 64 channels, 3x3, 56x56
  standard 3x3             params  18432   MACs  57,802,752   output (1, 64, 56, 56)
  grouped 3x3, groups=4    params   4608   MACs  14,450,688   output (1, 64, 56, 56)
  depthwise-separable      params   2336   MACs   7,325,696   output (1, 64, 56, 56)
  predicted separable/standard ratio 1/64 + 1/9 = 0.1267

Part 2: real models, one 3x224x224 image
  ResNet-18    params 11.69 M   MACs  1814.1 M   (standard 98.9%, pointwise 1.1%, linear 0.0%)
  MobileNetV2  params  3.50 M   MACs   300.8 M   (standard 3.6%, depthwise 6.9%, pointwise 89.1%, linear 0.4%)

Part 3: ConvTranspose2d shapes
  k=3 s=2 p=1 output_padding=0: 7x7 -> (13, 13)   weight (64, 32, 3, 3)
  k=3 s=2 p=1 output_padding=1: 7x7 -> (14, 14)   weight (64, 32, 3, 3)
  k=4 s=2 p=1 output_padding=0: 7x7 -> (14, 14)   weight (64, 32, 4, 4)
  k=2 s=2 p=0 output_padding=0: 7x7 -> (14, 14)   weight (64, 32, 2, 2)
  Conv2d(k=3, s=2, p=1) maps 14x14 -> (7, 7) and 13x13 -> (7, 7)`;

const FW_DIGITS = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)            # single thread -> reproducible floating-point order

digits = load_digits()
X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0   # (1797, 1, 8, 8)
y = torch.tensor(digits.target)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def cbr(c_in, c_out, k, groups=1):
    """Conv -> BatchNorm -> ReLU (bias is redundant before BatchNorm)."""
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU()]


def block(c_in, c_out, separable):
    if not separable:
        return cbr(c_in, c_out, 3)                                   # standard 3x3
    return cbr(c_in, c_in, 3, groups=c_in) + cbr(c_in, c_out, 1)     # depthwise 3x3, then pointwise 1x1


def make_cnn(separable):
    return nn.Sequential(
        *cbr(1, 16, 3),                                   # first layer: C_in = 1, nothing to separate
        *block(16, 32, separable), nn.MaxPool2d(2),       # 16x8x8 -> 32x8x8 -> 32x4x4
        *block(32, 64, separable),                        # 32x4x4 -> 64x4x4
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def conv_macs(model):
    """Multiply-adds of all conv layers for one 8x8 image (each conv here keeps its input's spatial size)."""
    total, hw = 0, 64
    for m in model:
        if isinstance(m, nn.MaxPool2d):
            hw //= 4
        if isinstance(m, nn.Conv2d):
            total += m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] ** 2 * hw
    return total


def train_and_test(separable, seed, epochs=10, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    model = make_cnn(separable)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model, (model(X_te).argmax(1) == y_te).float().mean().item()


for separable in (False, True):
    accs = []
    for seed in range(3):
        model, acc = train_and_test(separable, seed)
        accs.append(acc)
    n = sum(p.numel() for p in model.parameters())
    name = "separable" if separable else "standard"
    print(f"{name:9s} params {n:6d}  conv MACs {conv_macs(model):7,d}  "
          f"test acc per seed {[round(a, 4) for a in accs]}  mean {sum(accs) / 3:.4f}")`;

const FW_DIGITS_OUT = String.raw`standard  params  24058  conv MACs 599,040  test acc per seed [0.9889, 0.9956, 0.9822]  mean 0.9889
separable params   4106  conv MACs  88,576  test acc per seed [0.9733, 0.9822, 0.9822]  mean 0.9793`;

const CAUSAL = String.raw`import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)


def causal_conv1d(x, w, dilation=1):
    """y[t] = sum_m w[m] * x[t - dilation*(k-1-m)]: only the present and the past are used.
    x: (C_in, T), w: (C_out, C_in, k). Left-pad by dilation*(k-1) so the output also has length T."""
    C_out, C_in, k = w.shape
    pad = dilation * (k - 1)
    xp = np.concatenate([np.zeros((C_in, pad)), x], axis=1)
    T = x.shape[1]
    y = np.zeros((C_out, T))
    for m in range(k):
        y += w[:, :, m] @ xp[:, m * dilation:m * dilation + T]     # tap m looks back pad - m*d steps
    return y


def stack(x, weights, dilations):
    for w, d in zip(weights, dilations):
        x = np.tanh(causal_conv1d(x, w, d))
    return x


T, C = 32, 4
dilations = [1, 2, 4, 8]
weights = [rng.standard_normal((C, 1 if i == 0 else C, 2)) * 0.5 for i in range(len(dilations))]
x = rng.standard_normal((1, T))

# 1) one layer agrees with PyTorch (explicit left padding, then an ordinary dilated conv)
ref = F.conv1d(F.pad(torch.tensor(x)[None], (8 * 1, 0)), torch.tensor(weights[3][:, :1]), dilation=8)[0].numpy()
print("single layer vs torch, max|diff| =", f"{np.abs(causal_conv1d(x, weights[3][:, :1], 8) - ref).max():.1e}")

# 2) causality: changing x[20] must leave every output before t = 20 untouched
y0 = stack(x, weights, dilations)
x2 = x.copy(); x2[0, 20] += 1.0
changed = np.where(np.abs(stack(x2, weights, dilations) - y0).max(axis=0) > 1e-12)[0]
print("outputs changed by x[20]: t =", changed.min(), "...", changed.max())

# 3) receptive field of output t = 31: which inputs have a non-zero gradient?
xt = torch.tensor(x, requires_grad=True)
h = xt
for w, d in zip(weights, dilations):
    h = torch.tanh(F.conv1d(F.pad(h[None], (d, 0)), torch.tensor(w), dilation=d)[0])
h[:, 31].sum().backward()
nz = np.where(xt.grad[0].numpy() != 0)[0]
print("inputs reaching y[31]: t =", nz.min(), "...", nz.max(), f"({len(nz)} steps)")
print("formula 1 + (k-1)*sum(d) =", 1 + 1 * sum(dilations))`;

const CAUSAL_OUT = String.raw`single layer vs torch, max|diff| = 1.1e-16
outputs changed by x[20]: t = 20 ... 31
inputs reaching y[31]: t = 16 ... 31 (16 steps)
formula 1 + (k-1)*sum(d) = 16`;

const KERAS = String.raw`import tensorflow as tf

inp = tf.keras.Input(shape=(56, 56, 32))                       # Keras is channels-last: (H, W, C)

pointwise  = tf.keras.layers.Conv2D(64, kernel_size=1)(inp)                          # 1x1 conv
dilated    = tf.keras.layers.Conv2D(64, 3, padding="same", dilation_rate=2)(inp)     # atrous
depthwise  = tf.keras.layers.DepthwiseConv2D(3, padding="same")(inp)                 # groups = C_in
separable  = tf.keras.layers.SeparableConv2D(64, 3, padding="same")(inp)             # depthwise + pointwise
grouped    = tf.keras.layers.Conv2D(64, 3, padding="same", groups=4)(inp)            # grouped conv
upsampled  = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")(inp)  # 56 -> 112
resize_cnv = tf.keras.layers.Conv2D(32, 3, padding="same")(
                 tf.keras.layers.UpSampling2D(2, interpolation="nearest")(inp))     # resize-convolution

model = tf.keras.Model(inp, [pointwise, dilated, depthwise, separable, grouped, upsampled, resize_cnv])
model.summary()

# 1-D causal, dilated convolutions (WaveNet style) on a sequence of 100 steps with 1 feature
seq = tf.keras.Input(shape=(100, 1))
h = seq
for d in [1, 2, 4, 8]:
    h = tf.keras.layers.Conv1D(16, kernel_size=2, padding="causal", dilation_rate=d, activation="tanh")(h)
print(tf.keras.Model(seq, h).output_shape)`;

const lesson: Lesson = {
  id: "m08/conv-variants",
  title: "Dilated, Transposed, Depthwise and Pointwise Convolutions",
  summary:
    "The convolution variants that modern CNNs are built from: 1×1 convolutions that mix channels, dilated convolutions that see far without downsampling, transposed convolutions that upsample (and why they are the adjoint of a strided convolution), depthwise-separable and grouped convolutions that cut cost by an order of magnitude, plus 1-D, 3-D and causal convolutions — each implemented from scratch and checked against PyTorch.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m08/convolution", "m08/pooling-receptive-field", "m04/backpropagation", "m02/vectors-matrices"],
  next: ["m08/cnn-architectures", "m12/segmentation", "m14/gans"],
  terms: ["pointwise-convolution", "dilated-convolution", "transposed-convolution", "depthwise-convolution", "depthwise-separable-convolution", "grouped-convolution", "checkerboard-artifact", "causal-convolution"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "The [standard convolution](/learn/m08/convolution) slides a $C_{in}\\times k\\times k$ kernel over its input, sums everything under it, and does that for each of $C_{out}$ kernels. That one operation couples three things: **how far** each output looks (the kernel size), **how the resolution changes** (the stride), and **how channels interact** (every output channel reads every input channel). Almost every important CNN design since 2014 is a way of decoupling one of these three. This lesson covers the variants that do it.",
      },
      {
        t: "list",
        items: [
          "**1×1 (pointwise) convolution** — a kernel with no spatial extent. It is a fully connected layer applied independently to the channel vector of every pixel: it mixes and re-sizes channels without looking at neighbours. Used to shrink channels before expensive layers (Inception, ResNet bottlenecks) and as the mixing half of separable convolutions.",
          "**Dilated (atrous) convolution** — the kernel's taps are spread out with gaps, so a 3×3 kernel covers a 5×5, 7×7 or larger area with the same 9 weights. Receptive fields grow exponentially with depth **without** pooling, which is exactly what dense prediction (segmentation, audio generation) needs.",
          "**Transposed convolution** — the operation whose matrix is the transpose of a convolution's matrix. Where a stride-2 convolution halves the resolution, a stride-2 transposed convolution doubles it, with learnable weights. It is the upsampling layer of decoders, U-Nets and GAN generators — and the source of the well-known checkerboard artifacts.",
          "**Depthwise, pointwise and depthwise-separable convolution** — split a standard convolution into a per-channel spatial filter (depthwise) followed by a 1×1 channel mixer (pointwise). For a 3×3 layer with many output channels this costs roughly 8–9× less. It is the core of Xception, MobileNet, EfficientNet and ConvNeXt.",
          "**Grouped convolution** — the in-between case: channels are split into $g$ groups that are convolved separately (AlexNet, ResNeXt).",
          "**1-D, 3-D and causal convolutions** — the same ideas along one axis (sequences, audio), three axes (video, volumetric scans), or restricted to the past (WaveNet-style generation).",
        ],
      },
      {
        t: "p",
        text: "The variants appeared to solve concrete problems. Lin, Chen and Yan's *Network In Network* (2013) used 1×1 convolutions as tiny per-pixel MLPs; Szegedy et al.'s GoogLeNet/Inception (2014) and He et al.'s ResNet (2015) used them to make deep networks affordable. Yu and Koltun (2015, *Multi-Scale Context Aggregation by Dilated Convolutions*) and the DeepLab family (Chen et al., from 2014) used dilation to keep segmentation maps at high resolution. Long, Shelhamer and Darrell's fully convolutional networks (2015) and Radford et al.'s DCGAN (2015) popularised learned upsampling with transposed convolutions. Chollet's Xception (2017) and Howard et al.'s MobileNets (2017) made depthwise-separable convolution the default for efficient vision.",
      },
      {
        t: "callout",
        kind: "key",
        text: "All of these are still linear, locally connected, weight-shared operations — every one of them can be written as a (sparse) matrix times a vector. They differ in which entries of that matrix are allowed to be non-zero and which are tied together.",
      },
    ],

    // ---------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[The Convolution Operation](/learn/m08/convolution) — the sliding-window definition, stride, padding, the output-size formula $\\lfloor (n+2p-k)/s \\rfloor + 1$, the $C_{out}\\times C_{in}\\times k\\times k$ weight tensor and its parameter/multiply-add counts. Every variant here is a modification of that formula, so it must be familiar.",
          "[Pooling, Receptive Fields and CNN Architecture](/learn/m08/pooling-receptive-field) — the receptive-field recursion $r_\\ell = r_{\\ell-1} + (k_\\ell - 1)j_{\\ell-1}$. Dilation changes the $(k_\\ell-1)$ term; that is the whole story of its exponential receptive fields.",
          "[Backpropagation](/learn/m04/backpropagation) — in particular that a linear layer $y = Wx$ sends gradients back through $W^\\top$. Transposed convolution is literally that backward step, used as a forward layer.",
          "[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — matrix–vector products, the transpose, and the identity $\\langle Ax, y\\rangle = \\langle x, A^\\top y\\rangle$ used in the matrix view.",
          "[Multi-Layer Networks](/learn/m03/multilayer-networks) — what a fully connected (linear) layer computes, since a 1×1 convolution is one applied at every pixel.",
        ],
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Four ways to change a stencil",
        text: "Think of a 3×3 convolution as a stencil with nine holes that you slide over a picture, writing down a weighted sum of what shows through. A **1×1 convolution** is a stencil with one hole — but behind that hole you see a whole stack of transparent sheets (the channels), and you blend them into a new stack. A **dilated** convolution keeps the nine holes but spreads them apart, like reading every second seat in a stadium row: you see a wider area with the same nine numbers, at the price of ignoring the seats in between. A **transposed** convolution turns the stencil into a rubber stamp: each input pixel inks the stamp with its own value and presses it onto a larger sheet; where stamps overlap, the ink adds up. A **depthwise-separable** convolution is a kitchen with a division of labour: first each cook prepares one ingredient on their own (one spatial filter per channel), then a single plating step combines the prepared ingredients in different proportions for each dish (the 1×1 mixing).",
      },
      {
        t: "p",
        text: "**Why 1×1 is useful even though it sees nothing spatially.** A 256-channel feature map stores 256 numbers per pixel. A 1×1 convolution with 64 outputs computes 64 different weighted combinations of those 256 numbers at every pixel — it compresses, expands or re-mixes the per-pixel description. Put it before a 3×3 convolution and that 3×3 now operates on 64 channels instead of 256, about 16× cheaper.",
      },
      {
        t: "p",
        text: "**Why dilation instead of pooling.** Pooling grows the receptive field by throwing away resolution. For classification that is fine — the final answer is one label. For segmentation you need one answer **per pixel**, so after pooling you would have to upsample again and the fine detail is gone. Dilation grows the view while keeping every pixel's output. Stacking dilations 1, 2, 4, 8 makes a 3×3 network see 31 pixels wide after only four layers, where plain 3×3 layers would see 9.",
      },
      {
        t: "p",
        text: "**Why transposed convolution is called \"transposed\".** A stride-2 convolution takes, say, a 5-pixel signal to 2 outputs; you can write it as a 2×5 matrix $C$. Its transpose $C^\\top$ is 5×2: it takes 2 numbers back to 5. Multiplying by $C^\\top$ is exactly what backpropagation does to send gradients from the 2 outputs back to the 5 inputs. Use that same map as a **forward** layer with learnable kernel and you have an upsampler that is shaped like the reverse of a downsampler. It is not an inverse — $C^\\top C \\ne I$ — it only reverses the *shapes*.",
      },
      {
        t: "p",
        text: "**Why separating depthwise and pointwise is cheap.** A standard 3×3 layer from 32 to 64 channels learns a separate 3×3 filter for every (input, output) pair: 64·32 = 2,048 spatial filters. The separable version learns only 32 spatial filters (one per input channel) and then 64·32 single numbers to mix them. It bets that *what spatial pattern to look for* and *how to combine channels* can be learned separately. Empirically the bet mostly pays off — which is why phones run separable networks.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "1×1 (pointwise) convolution" },
      {
        t: "steps",
        items: [
          { title: "Take one pixel's channel vector", text: "At position (i, j) the input holds a vector $x_{:,i,j}\\in\\mathbb{R}^{C_{in}}$." },
          { title: "Apply one linear layer", text: "Multiply by the weight matrix $W\\in\\mathbb{R}^{C_{out}\\times C_{in}}$ and add the bias: $y_{:,i,j} = Wx_{:,i,j} + b$." },
          { title: "Repeat at every pixel with the same W", text: "The output keeps the spatial size and has $C_{out}$ channels. Equivalently: reshape the input to an $(HW)\\times C_{in}$ matrix, multiply by $W^\\top$, reshape back." },
          { title: "Add a non-linearity", text: "With ReLU after it, a 1×1 convolution is a one-hidden-layer MLP shared across positions — the *Network In Network* idea." },
        ],
      },
      { t: "h", text: "Dilated (atrous) convolution" },
      {
        t: "steps",
        items: [
          { title: "Choose a dilation rate d", text: "d = 1 is an ordinary convolution. With d = 2 the kernel taps sit 2 pixels apart, with d = 3 they sit 3 apart. *Atrous* comes from the French *à trous*, “with holes”, a name from wavelet signal processing." },
          { title: "Place the spread-out kernel", text: "Tap (m, n) of the kernel reads the input at offset (d·m, d·n) from the window's corner. A k×k kernel therefore spans $k_{\\text{eff}} = k + (k-1)(d-1)$ pixels per side but still has only $k^2$ weights." },
          { title: "Multiply-add the k² sampled pixels", text: "The pixels inside the span that are not on a tap are simply skipped — they do not contribute to this output." },
          { title: "Slide by the stride (usually 1)", text: "Output size uses $k_{\\text{eff}}$ in place of k. For same-size output with stride 1, pad by $p = d(k-1)/2$ — i.e. p = d for a 3×3 kernel." },
          { title: "Stack with growing rates", text: "Rates 1, 2, 4, 8, … make the receptive field grow exponentially with depth. Rates that share a common factor (e.g. 2, 2, 2) leave regular holes in the receptive field — the gridding artifact; mixing co-prime or doubling rates avoids it." },
        ],
      },
      { t: "h", text: "Transposed convolution (scatter view)" },
      {
        t: "steps",
        items: [
          { title: "Allocate a canvas", text: "For an n×n input, kernel k and stride s, the full canvas is $(n-1)s + k$ on each side (plus output_padding, see below)." },
          { title: "Stamp each input pixel", text: "Input pixel (i, j) multiplies the whole kernel by its value and **adds** the result into the canvas block whose top-left corner is (i·s, j·s). With several channels, each input channel stamps its own kernel slice into every output channel." },
          { title: "Overlaps accumulate", text: "If k > s, neighbouring stamps overlap and their contributions add. If k is not a multiple of s, some canvas pixels receive more stamps than their neighbours — the root of checkerboard artifacts." },
          { title: "Crop the padding", text: "The padding argument p of a transposed convolution **removes** p rows/columns from each border of the canvas (it is the padding of the forward convolution it mirrors). output_padding adds extra rows/columns on one side to pick between output sizes that the forward convolution would map to the same input size." },
          { title: "Add bias, then non-linearity", text: "One bias per output channel, as in an ordinary convolution." },
        ],
      },
      {
        t: "p",
        text: "An equivalent **gather view**, often drawn in textbooks: insert s − 1 zeros between input pixels, pad the result by k − 1 − p on each side, and run an ordinary stride-1 convolution with the kernel rotated by 180°. It computes exactly the same numbers; the zero-insertion is why the operation was called *fractionally strided convolution* in the DCGAN paper. (Some papers also call it *deconvolution*, but it is not a deconvolution in the signal-processing sense — it does not invert anything.)",
      },
      { t: "h", text: "Depthwise, pointwise and depthwise-separable convolution" },
      {
        t: "steps",
        items: [
          { title: "Depthwise step", text: "Give each of the $C_{in}$ input channels its own k×k kernel. Channel c is convolved only with kernel c, producing one map. No information crosses channels. In PyTorch this is nn.Conv2d with groups = in_channels; the weight has shape $(C_{in}, 1, k, k)$." },
          { title: "(Optional) BatchNorm + ReLU", text: "MobileNet v1 puts BN + ReLU after both steps; Xception puts no non-linearity between them. Both are used in practice." },
          { title: "Pointwise step", text: "Apply a 1×1 convolution $C_{in}\\to C_{out}$ to the depthwise output: at every pixel, $C_{out}$ weighted sums of the $C_{in}$ filtered channels." },
          { title: "Result", text: "Same output shape as a standard $k\\times k$ convolution from $C_{in}$ to $C_{out}$, but the spatial filtering and the channel mixing have been factorised into two cheaper layers." },
        ],
      },
      { t: "h", text: "Grouped convolution" },
      {
        t: "p",
        text: "Split the $C_{in}$ input channels into $g$ equal groups and the $C_{out}$ output channels into $g$ groups; output group $i$ only reads input group $i$. Each kernel then has depth $C_{in}/g$ instead of $C_{in}$, so weights and multiply-adds drop by a factor of $g$. $g = 1$ is a standard convolution; $g = C_{in} = C_{out}$ is a depthwise convolution. AlexNet (2012) used $g = 2$ to split the model across two GPUs; ResNeXt (Xie et al., 2017) used $g = 32$ inside bottleneck blocks and found it more accurate than a standard block of the same cost. Because groups never talk to each other, networks that stack grouped convolutions need something to mix them — a 1×1 convolution, or ShuffleNet's channel shuffle.",
      },
      { t: "h", text: "1-D, 3-D and causal convolutions" },
      {
        t: "list",
        items: [
          "**1-D convolution** slides a $C_{in}\\times k$ kernel along one axis: time for audio and sensor signals, position for text embeddings. Shapes are $(N, C, L)$ in PyTorch (nn.Conv1d). Every variant above has a 1-D version — 1-D dilated convolutions are particularly common.",
          "**3-D convolution** slides a $C_{in}\\times k_t\\times k\\times k$ kernel over (time or depth, height, width): video clips, CT and MRI volumes (nn.Conv3d). Cost grows with an extra factor $k_t$, which is why video networks often factorise a 3-D kernel into a 2-D spatial convolution followed by a 1-D temporal one — the same separability idea as depthwise-separable convolution, applied to axes instead of channels.",
          "**Causal convolution** is a 1-D convolution whose output at time $t$ depends only on inputs at times $\\le t$. It is implemented by padding $d(k-1)$ zeros on the **left only**. WaveNet (van den Oord et al., 2016) stacked causal convolutions with dilations 1, 2, 4, …, 512 to generate raw audio one sample at a time; temporal convolutional networks (TCNs) use the same block for sequence modelling.",
        ],
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "1×1 convolution as a matrix product" },
      {
        t: "p",
        text: String.raw`Let $X\in\mathbb{R}^{C_{in}\times H\times W}$, $W\in\mathbb{R}^{C_{out}\times C_{in}}$, $b\in\mathbb{R}^{C_{out}}$. The 1×1 convolution is`,
      },
      {
        t: "math",
        tex: String.raw`Y[o,i,j] = b_o + \sum_{c=0}^{C_{in}-1} W[o,c]\,X[c,i,j] \quad\Longleftrightarrow\quad \underbrace{Y_{\text{mat}}}_{C_{out}\times HW} = W\,\underbrace{X_{\text{mat}}}_{C_{in}\times HW} + b\mathbf{1}^\top`,
        caption: "A 1×1 convolution is one matrix multiplication: params $C_{out}C_{in}$ (+ $C_{out}$), multiply-adds $C_{out}C_{in}HW$.",
      },
      {
        t: "p",
        text: String.raw`**Bottleneck arithmetic.** A ResNet-50 bottleneck block on 256 channels runs 1×1 (256→64), 3×3 (64→64), 1×1 (64→256): $256\cdot64 + 64\cdot64\cdot9 + 64\cdot256 = 69{,}632$ weights. Two plain 3×3 layers at 256 channels would need $2\cdot256\cdot256\cdot9 = 1{,}179{,}648$ — 17× more. (He et al. designed the bottleneck to cost about the same as two 3×3 layers at 64 channels, $73{,}728$ weights, while working on a 256-channel representation.)`,
      },
      { t: "h", text: "Dilated convolution" },
      {
        t: "p",
        text: String.raw`With dilation $d\ge1$, stride $s$, padding $p$, a $k\times k$ kernel $K$ and the zero-padded input $X_p$:`,
      },
      {
        t: "math",
        tex: String.raw`Y[i,j] = b + \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_p[\,s\,i + d\,m,\; s\,j + d\,n\,]\;K[m,n]`,
        caption: "The only change from the standard formula: the tap offsets m, n are multiplied by d.",
      },
      {
        t: "math",
        tex: String.raw`k_{\text{eff}} = k + (k-1)(d-1) = d(k-1) + 1, \qquad H_{\text{out}} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1`,
        caption: "The output-size formula is the standard one with k replaced by the effective kernel size. Same padding at stride 1: p = d(k−1)/2.",
      },
      {
        t: "p",
        text: String.raw`**Receptive field.** In the recursion $r_\ell = r_{\ell-1} + (k_{\text{eff},\ell}-1)\,j_{\ell-1}$ with stride 1 everywhere ($j = 1$), each layer adds $(k-1)d_\ell$. For $L$ layers of 3×3 kernels with dilations $d_\ell = 2^{\ell-1}$:`,
      },
      {
        t: "math",
        tex: String.raw`r_L = 1 + (k-1)\sum_{\ell=1}^{L} d_\ell = 1 + 2\,(1 + 2 + \dots + 2^{L-1}) = 2^{L+1} - 1`,
        caption: "Exponential growth with linear depth and linear parameter count, at full resolution. Dilations (1,1,1,1) give 9; (2,2,2,2) give 17; (1,2,4,8) give 31.",
      },
      {
        t: "p",
        text: String.raw`**Worked example (1-D).** $x = (3, 1, 4, 1, 5, 9, 2, 6)$, $w = (1, 2, -1)$, $d = 2$, no padding. Then $k_{\text{eff}} = 3 + 2\cdot1 = 5$ and the output has $8 - 5 + 1 = 4$ entries, $y_i = x_i + 2x_{i+2} - x_{i+4}$:`,
      },
      {
        t: "math",
        tex: String.raw`\begin{aligned} y_0 &= 3 + 2\cdot4 - 5 = 6, & y_1 &= 1 + 2\cdot1 - 9 = -6,\\ y_2 &= 4 + 2\cdot5 - 2 = 12, & y_3 &= 1 + 2\cdot9 - 6 = 13. \end{aligned}`,
      },
      {
        t: "p",
        text: String.raw`**Worked example (2-D).** The 7×7 input $X[i,j] = (i + 2j) \bmod 5$ with the Sobel-x kernel $K = \begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}$ and $d = 2$: $k_{\text{eff}} = 5$, so the output is $7 - 5 + 1 = 3$ wide. Output $(0,0)$ reads rows 0, 2, 4 and columns 0, 2, 4:`,
      },
      {
        t: "math",
        tex: String.raw`\begin{bmatrix}X_{00}&X_{02}&X_{04}\\X_{20}&X_{22}&X_{24}\\X_{40}&X_{42}&X_{44}\end{bmatrix} = \begin{bmatrix}0&4&3\\2&1&0\\4&3&2\end{bmatrix},\quad Y_{00} = (0 - 3) + 2(2 - 0) + (4 - 2) = 3,\qquad Y = \begin{bmatrix}3&3&-2\\-2&-2&3\\3&-2&-2\end{bmatrix}`,
      },
      {
        t: "p",
        text: String.raw`**Gridding.** Two stacked $d=2$ layers combine offsets $2m_1 + 2m_2$, which are always even: the output depends only on input pixels whose row and column offsets are both even — a quarter of the 9×9 receptive field (the NumPy section prints this mask). Neighbouring outputs therefore see disjoint sets of pixels, which shows up as grid-like artifacts in segmentation maps. Wang et al. (2018) proposed *hybrid dilated convolution*: choose consecutive rates without a common factor greater than 1 (e.g. 1, 2, 3) so the holes are filled.`,
      },
      { t: "h", text: "Transposed convolution: the matrix view" },
      {
        t: "p",
        text: String.raw`Any convolution is linear in its input, so flattening the input to $x\in\mathbb{R}^{n}$ and the output to $y\in\mathbb{R}^{m}$ it can be written $y = Cx$ with a sparse $C\in\mathbb{R}^{m\times n}$: row $i$ of $C$ holds the kernel weights at the positions that output $i$ reads, and zeros elsewhere. Example: 1-D, input length 5, kernel $w = (1, 2, 3)$, stride 2, giving 2 outputs:`,
      },
      {
        t: "math",
        tex: String.raw`C = \begin{bmatrix}1&2&3&0&0\\0&0&1&2&3\end{bmatrix},\qquad x = \begin{bmatrix}1\\0\\2\\1\\3\end{bmatrix} \;\Rightarrow\; Cx = \begin{bmatrix}1 + 0 + 6\\ 2 + 2 + 9\end{bmatrix} = \begin{bmatrix}7\\13\end{bmatrix}`,
      },
      {
        t: "p",
        text: String.raw`The **transposed convolution** with the same kernel and stride is the linear map $C^\top\in\mathbb{R}^{5\times2}$. For $g = (1, -1)$:`,
      },
      {
        t: "math",
        tex: String.raw`C^\top g = \begin{bmatrix}1&0\\2&0\\3&1\\0&2\\0&3\end{bmatrix}\begin{bmatrix}1\\-1\end{bmatrix} = \begin{bmatrix}1\\2\\2\\-2\\-3\end{bmatrix} = 1\cdot\underbrace{(1,2,3,0,0)}_{\text{stamp at }0} \;+\; (-1)\cdot\underbrace{(0,0,1,2,3)}_{\text{stamp at }2}`,
        caption: "Column j of Cᵀ is the kernel placed at position j·s: multiplying by Cᵀ is exactly the scatter-add of stamps.",
      },
      {
        t: "p",
        text: String.raw`Why this is the gradient: if $y = Cx$ and $L$ is any loss with upstream gradient $g = \partial L/\partial y$, then $\partial L/\partial x = C^\top g$ — the same rule as for a dense layer in [backpropagation](/learn/m04/backpropagation). So **the forward pass of a transposed convolution is the backward (input-gradient) pass of the corresponding convolution**, and vice versa. Frameworks implement one with the other's kernels. The defining property is the adjoint identity $\langle Cx, g\rangle = \langle x, C^\top g\rangle$ for all $x, g$; the NumPy section checks it numerically on a 2-D example, together with autograd.`,
      },
      {
        t: "p",
        text: String.raw`**Scatter form.** For a single channel, stride $s$, kernel $K$ and before cropping:`,
      },
      {
        t: "math",
        tex: String.raw`Z[u,v] = \sum_{i,j} X[i,j]\,K[u - s\,i,\; v - s\,j] \quad(\text{terms with } K \text{ indices outside } [0,k) \text{ are } 0),\qquad Y = Z[p : -p,\; p : -p]`,
      },
      {
        t: "math",
        tex: String.raw`H_{\text{out}} = (H_{\text{in}} - 1)\,s - 2p + d\,(k-1) + \text{output\_padding} + 1`,
        caption: "PyTorch's ConvTranspose2d output size (d = dilation, usually 1). It inverts the forward size formula when (H_out + 2p − k) is divisible by s.",
      },
      {
        t: "p",
        text: String.raw`**Why output_padding exists.** A convolution with $k=3, s=2, p=1$ maps both 14 and 13 to $\lfloor(14+2-3)/2\rfloor+1 = \lfloor(13+2-3)/2\rfloor + 1 = 7$. The transposed layer, going from 7 back, gives $(7-1)\cdot2 - 2 + 3 = 13$ by default; output_padding = 1 selects 14. Common decoder choices that exactly double the size: $(k, s, p) = (4, 2, 1)$ or $(2, 2, 0)$, or $(3, 2, 1)$ with output_padding 1.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example (1-D scatter).** $x = (1, 2, 3)$, $w = (1, 2, 1)$, $s = 2$: canvas length $(3-1)\cdot2 + 3 = 7$. Stamps: $1\cdot w$ at positions 0–2, $2\cdot w = (2,4,2)$ at 2–4, $3\cdot w = (3,6,3)$ at 4–6. Summing, $y = (1,\ 2,\ 1{+}2,\ 4,\ 2{+}3,\ 6,\ 3) = (1, 2, 3, 4, 5, 6, 3)$.`,
      },
      {
        t: "p",
        text: String.raw`**Checkerboard artifacts** (Odena, Dumoulin and Olah, 2016, *Deconvolution and Checkerboard Artifacts*). Feed all-ones input and an all-ones kernel: each output equals the number of stamps covering it. With $k = 3, s = 2$ in 1-D that is $(1, 1, 2, 1, 2, 1, 1)$ — the interior alternates 1, 2, 1, 2. In 2-D the counts multiply, giving a 1–2–4 pattern. Whenever $k$ is not divisible by $s$, a stride-$s$ transposed convolution has this uneven overlap; the network can learn weights that compensate, but in practice it often doesn't fully, and generated images show a faint checkerboard. Two standard fixes: choose $k$ divisible by $s$ (e.g. $k=4, s=2$), or use **resize-convolution** — nearest-neighbour or bilinear upsampling followed by an ordinary stride-1 convolution, whose overlap is uniform by construction.`,
      },
      { t: "h", text: "Depthwise-separable convolution: cost" },
      {
        t: "p",
        text: String.raw`For a stride-1, same-padded layer on an $H\times W$ map (biases ignored), a standard convolution costs $C_{out}C_{in}k^2$ weights and $C_{out}C_{in}k^2HW$ multiply-adds (MACs). The separable pair costs $C_{in}k^2$ (depthwise) plus $C_{in}C_{out}$ (pointwise):`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\text{separable}}{\text{standard}} = \frac{C_{in}k^2 + C_{in}C_{out}}{C_{out}C_{in}k^2} = \frac{1}{C_{out}} + \frac{1}{k^2}`,
        caption: "Same ratio for weights and for MACs (both scale by HW). With many output channels the cost is ≈ 1/k² — about 9× cheaper for 3×3.",
      },
      {
        t: "p",
        text: String.raw`**Numbers.** $C_{in}=32, C_{out}=64, k=3, 56\times56$: standard $64\cdot32\cdot9 = 18{,}432$ weights and $18{,}432\cdot3136 = 57{,}802{,}752$ MACs; separable $32\cdot9 + 32\cdot64 = 288 + 2{,}048 = 2{,}336$ weights and $7{,}325{,}696$ MACs. Ratio $2336/18432 = 0.126736 = 1/64 + 1/9$, a 7.9× reduction. Note that 2,048 of the 2,336 weights (88%) are in the pointwise layer — in separable networks the 1×1 convolutions dominate the cost (89% of MobileNetV2's MACs, measured in the PyTorch tab).`,
      },
      {
        t: "p",
        text: String.raw`**Grouped convolution** with $g$ groups costs $C_{out}\,(C_{in}/g)\,k^2$ weights — exactly $1/g$ of the standard layer. Depthwise is the extreme $g = C_{in}$ (with $C_{out} = C_{in}$, or a channel multiplier $C_{out} = mC_{in}$).`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Two 3×3 input channels $X_0 = \begin{bmatrix}1&2&0\\0&1&3\\2&1&1\end{bmatrix}$, $X_1 = \begin{bmatrix}2&0&1\\1&1&0\\0&2&1\end{bmatrix}$; depthwise 2×2 kernels $D_0 = \begin{bmatrix}1&-1\\0&2\end{bmatrix}$, $D_1 = \begin{bmatrix}0&1\\1&0\end{bmatrix}$ (no padding); pointwise matrix $P = \begin{bmatrix}1&1\\1&-1\\2&0\end{bmatrix}$ (3 outputs).`,
      },
      {
        t: "steps",
        items: [
          { title: "Depthwise, channel 0, position (0,0)", tex: String.raw`1\cdot1 + 2\cdot(-1) + 0\cdot0 + 1\cdot2 = 1;\quad \text{all positions: } \tilde X_0 = \begin{bmatrix}1&8\\1&0\end{bmatrix}` },
          { title: "Depthwise, channel 1", text: "D₁ adds the top-right and bottom-left pixel of each window.", tex: String.raw`(0,0):\; 2\cdot0 + 0\cdot1 + 1\cdot1 + 1\cdot0 = 1;\quad \text{all positions: } \tilde X_1 = \begin{bmatrix}1&2\\1&2\end{bmatrix}` },
          { title: "Pointwise mixes per pixel", tex: String.raw`Y_0 = \tilde X_0 + \tilde X_1 = \begin{bmatrix}2&10\\2&2\end{bmatrix},\; Y_1 = \tilde X_0 - \tilde X_1 = \begin{bmatrix}0&6\\0&-2\end{bmatrix},\; Y_2 = 2\tilde X_0 = \begin{bmatrix}2&16\\2&0\end{bmatrix}` },
          { title: "Cost of this toy layer", text: "Separable: 2·4 + 2·3 = 14 weights. A standard 2×2 convolution 2 → 3 channels: 3·2·4 = 24 weights; ratio 14/24 = 1/3 + 1/4." },
        ],
      },
      {
        t: "details",
        summary: "Check every worked number with PyTorch",
        blocks: [
          { t: "code", lang: "python", title: "check_worked_examples.py", code: CHECK_MATH, output: CHECK_MATH_OUT },
        ],
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "conv-variants", caption: "Three tabs: a dilated kernel sliding over an input, a transposed convolution stamping kernels onto a canvas, and a depthwise-separable layer with a live cost calculator." },
      { t: "h", text: "Dilated tab" },
      {
        t: "list",
        items: [
          "Start at d = 2 with the 9×9 input and press **Step**. The dashed square is the 5×5 span; only the 9 gold cells are read. Compare the multiply-add in the narration with the cell values.",
          "Move **Dilation** from 1 to 4 and watch the readout: effective kernel 3, 5, 7, 9; the weights stay at 9; the valid output shrinks from 7×7 to 1×1. Switch padding to **same** (p = d) and the output stays 9×9 for every d.",
          "Read the two receptive-field lines: three layers all at d = 2 give 13, while d = 1, 2, 4 give 15 and leave no holes.",
          "With d = 4 and valid padding on a 7×7 input, the kernel no longer fits — the narration tells you why.",
        ],
      },
      { t: "h", text: "Transposed tab" },
      {
        t: "list",
        items: [
          "Default: 3×3 input of ones, 3×3 kernel of ones, stride 2. Step through the 9 stamps; the gold block is where the current stamp lands, and the corner numbers count overlaps. The final 7×7 canvas shows 1s, 2s and 4s — the checkerboard.",
          "Toggle **overlap count**. Then set k = 4 (divisible by s = 2): the interior becomes uniform. Set k = 2: no overlaps at all.",
          "Set stride 1: stamps overlap heavily but evenly in the interior — a stride-1 transposed convolution is just a full-padded convolution with the flipped kernel.",
          "Increase padding p and watch the dashed border being cropped at the last step; with stride 2 raise output_padding to 1 to get the even output size the readout predicts from $(n-1)s - 2p + k + \\text{op}$.",
          "Switch to **random ints** to see the same mechanism with real values: each output is the sum of the products from all stamps that covered it.",
        ],
      },
      { t: "h", text: "Depthwise-separable tab" },
      {
        t: "list",
        items: [
          "Step once per stage: each depthwise step connects one input channel to one output map only; each pointwise step draws lines from **all** depthwise maps to one output. The narration computes the outlined centre pixel at each stage.",
          "In the calculator, set C_in = 32, C_out = 64, k = 3, H = W = 56: you get 18,432 vs 2,336 weights and 57,802,752 vs 7,325,696 MACs — the numbers from the Math section and the PyTorch run.",
          "Push C_out to 1024: the ratio approaches 1/9 = 0.111. Set k = 7: the ratio for large C_out approaches 1/49 — large depthwise kernels are cheap, which is why ConvNeXt can afford 7×7.",
          "Watch the *share of MACs in pointwise* line: with k = 3 and 64 or more output channels it is above 85% (it equals $C_{out}/(C_{out} + k^2)$). Making separable networks faster mostly means making their 1×1 convolutions faster.",
        ],
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "NumPy implementations of the 1×1, dilated, transposed (as a scatter-add of kernel stamps), depthwise and depthwise-separable convolutions, each compared with `torch.nn.functional` on random data for several stride/padding/dilation settings. The second part builds the explicit convolution matrix C for a 5×5 input and shows that the transposed convolution is C transposed and equals the autograd input-gradient. The last two parts reproduce the checkerboard overlap pattern (and its absence with resize-convolution) and the gridding mask of stacked dilated layers.",
      },
      { t: "code", lang: "python", title: "conv_variants_from_scratch.py", code: SCRATCH, output: SCRATCH_OUT },
      {
        t: "list",
        items: [
          "`conv2d_dilated` loops over the k² **taps** rather than over output positions: for tap (a, c) it takes one strided slice of the padded input, starting at (a·d, c·d), that lines up with every output position at once. That is the formula $X_p[si + dm, sj + dn]$ read literally.",
          "`conv_transpose2d` is the scatter view: each input pixel adds `x[:, :, i, j]` times the kernel into the canvas block at (i·s, j·s); `padding` then crops the border and `output_padding` adds one extra row/column at the bottom/right. Note PyTorch's weight layout for transposed convolution is (C_in, C_out, k, k) — the transpose of Conv2d's (C_out, C_in, k, k).",
          "`depthwise` runs a separate single-channel convolution per channel and concatenates; PyTorch gets the same result with `groups=6`. The separable result equals a depthwise layer followed by a 1×1 layer.",
          "The matrix check: each row of C is the kernel placed where one output looks (row 1 is printed). `C @ x` equals the convolution, `C.T @ g` equals our transposed convolution, and both equal the gradient autograd computes for the convolution's input. The adjoint identity holds to all printed digits.",
          "The coverage map has 1–2–4 in the 2-D interior for k = 3, s = 2; the resize-convolution (nearest ×2 then 3×3 mean) is exactly 1 everywhere in the interior.",
          "The gridding mask shows that two stacked d = 2 layers read only every other row and column of their 9×9 receptive field.",
        ],
      },
    ],

    // -------------------------------------------------------------- framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "Every variant is a constructor argument of `nn.Conv2d` / `nn.ConvTranspose2d`: `kernel_size=1` (pointwise), `dilation=d`, `groups=g` (`groups=in_channels` for depthwise). The first program counts weights and multiply-adds with forward hooks — for a single layer and for two real torchvision architectures (random weights; nothing is downloaded) — and prints the shapes produced by common transposed-convolution settings.",
              },
              { t: "code", lang: "python", title: "conv_variant_costs.py", code: FW_COUNTS, output: FW_COUNTS_OUT },
              {
                t: "list",
                items: [
                  "The grouped layer costs exactly 1/4 of the standard one; the separable pair costs 2,336/18,432 = 0.1267 of it, matching $1/C_{out} + 1/k^2$.",
                  "ResNet-18 (1.81 G MACs, 11.69 M parameters) is almost entirely standard 3×3 convolutions. MobileNetV2 (Sandler et al., 2018) needs 300.8 M MACs — 6× fewer — and 89% of them are in 1×1 convolutions; the depthwise 3×3 layers are only 6.9%.",
                  "ConvTranspose2d's weight has shape (in, out, k, k). k = 3, s = 2, p = 1 gives 13 from 7 unless output_padding = 1; (4, 2, 1) and (2, 2, 0) give 14 directly.",
                ],
              },
              {
                t: "p",
                text: "The second program trains the same small CNN on scikit-learn's 8×8 digits twice: once with standard 3×3 convolutions in layers 2 and 3, once with depthwise-separable ones (Conv → BatchNorm → ReLU after both the depthwise and pointwise steps, as in MobileNet). Three seeds each, 10 epochs of Adam.",
              },
              { t: "code", lang: "python", title: "standard_vs_separable_digits.py", code: FW_DIGITS, output: FW_DIGITS_OUT },
              {
                t: "p",
                text: "With 5.9× fewer parameters and 6.8× fewer convolution multiply-adds, the separable network loses about one percentage point of test accuracy on average (97.9% vs 98.9%) in this run. That is the typical trade: a small accuracy cost for a large efficiency gain, which can be won back by spending part of the savings on width or depth (the strategy of MobileNetV2 and EfficientNet). On such a small dataset the per-seed spread is itself around one point, so treat the gap as indicative, not precise.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras has a dedicated layer for each variant: `Conv2D(dilation_rate=…)`, `Conv2D(groups=…)`, `DepthwiseConv2D`, `SeparableConv2D` (depthwise + pointwise in one layer), `Conv2DTranspose`, and `Conv1D(padding=\"causal\")`. Data is channels-last (N, H, W, C).",
              },
              { t: "code", lang: "python", title: "conv_variants_keras.py (not executed in CI)", code: KERAS, output: "Prints a model summary listing each layer's output shape and parameter count: the 1×1, dilated and grouped layers keep 56×56×64; DepthwiseConv2D keeps 56×56×32 with 32·9 + 32 = 320 parameters; SeparableConv2D gives 56×56×64 with 32·9 + 32·64 + 64 = 2,400 parameters; Conv2DTranspose with strides=2 and padding=\"same\" gives 112×112×32, as does the UpSampling2D + Conv2D path. The final line prints the causal stack's output shape (None, 100, 16): causal padding keeps the sequence length." },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      {
        t: "table",
        head: ["Variant", "Weights (no bias)", "What it changes", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Standard k×k", "$C_{out}C_{in}k^2$", "—", "Mixes space and channels jointly; most expressive per layer", "Cost ∝ C_in·C_out·k²", "Most CNN layers (ResNet, VGG)"],
          ["1×1 (pointwise)", "$C_{out}C_{in}$", "No spatial extent", "Cheap channel mixing and resizing; adds depth/non-linearity", "Sees no neighbours", "Bottlenecks (Inception, ResNet), mixing half of separable convs, projection shortcuts"],
          ["Dilated / atrous", "$C_{out}C_{in}k^2$", "Taps spaced d apart", "Large receptive field at full resolution, no extra weights", "Gridding with shared-factor rates; sparse sampling of small objects", "DeepLab ASPP, WaveNet, TCNs"],
          ["Transposed", "$C_{in}C_{out}k^2$", "Scatter/stamp; upsamples by s", "Learnable upsampling; exact adjoint of strided conv", "Checkerboard artifacts when k is not divisible by s", "FCN/U-Net decoders, DCGAN generators"],
          ["Resize-convolution", "$C_{out}C_{in}k^2$", "Fixed interpolation, then stride-1 conv", "No checkerboard; simple", "Interpolation is not learned; more MACs at high resolution", "Modern GAN and diffusion U-Net decoders"],
          ["Depthwise", "$C_{in}k^2$", "groups = C_in", "Very cheap spatial filtering", "No channel mixing; low arithmetic intensity on GPUs", "First half of separable convs; ConvNeXt 7×7"],
          ["Depthwise-separable", "$C_{in}k^2 + C_{in}C_{out}$", "Factorises space and channels", "≈ 1/k² of the cost for large C_out", "Slightly lower accuracy per layer; memory-bound depthwise step", "Xception, MobileNet v1–v3, EfficientNet"],
          ["Grouped (g groups)", "$C_{out}C_{in}k^2/g$", "Channels split into g independent groups", "1/g of the cost; works as an ensemble of paths", "Groups never communicate without extra mixing", "AlexNet (g = 2), ResNeXt (g = 32), ShuffleNet"],
          ["1-D / causal", "$C_{out}C_{in}k$", "One axis; causal = left padding only", "Parallel training over sequences; strict causality", "Fixed receptive field; needs dilation for long context", "WaveNet, TCNs, keyword spotting, ECG"],
          ["3-D", "$C_{out}C_{in}k_tk^2$", "Kernel also spans time/depth", "Models motion and volumes directly", "Expensive; often factorised into 2-D + 1-D", "Video (C3D, I3D), CT/MRI segmentation"],
        ],
      },
      {
        t: "p",
        text: "Related ideas you will meet: **inverted residual blocks** (MobileNetV2: 1×1 expand → 3×3 depthwise → 1×1 project, with a residual connection between the thin ends); **atrous spatial pyramid pooling** (DeepLab v2/v3: parallel 3×3 convolutions with several dilation rates, e.g. 6, 12, 18, concatenated to capture multiple scales); **pixel shuffle** / sub-pixel convolution (a stride-1 convolution produces $s^2$ times the channels, then a reshape moves them into an $s\\times$ larger grid — another checkerboard-free upsampler); and **deformable convolution**, where the tap offsets are themselves predicted by the network instead of fixed at multiples of d.",
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Semantic segmentation** — DeepLab (v1–v3+) uses atrous convolution to keep feature maps at 1/8 or 1/16 of the input resolution instead of 1/32, and ASPP to aggregate context at several rates. U-Net-style decoders upsample with transposed convolutions or resize-convolutions. See [Segmentation](/learn/m12/segmentation).",
          "**Mobile and embedded vision** — MobileNet v1–v3, EfficientNet and many on-device detectors are built from depthwise-separable blocks, because phone CPUs and NPUs are limited by multiply-adds and memory. See [Model Optimization](/learn/m18/model-optimization) for the complementary tools (quantization, pruning).",
          "**Image generation** — DCGAN-style generators upsample a latent vector with stacks of stride-2 transposed convolutions; later generators and the U-Nets inside [diffusion models](/learn/m14/diffusion) commonly use nearest/bilinear upsampling followed by a convolution to avoid checkerboard artifacts. See [GANs](/learn/m14/gans).",
          "**Audio generation and speech** — WaveNet's dilated causal convolutions model raw waveforms at 16 kHz with receptive fields of thousands of samples; 1-D dilated convolutions also appear in speech enhancement and keyword-spotting models.",
          "**Bottlenecks everywhere** — every ResNet-50/101/152 block, every Inception module and most modern CNNs use 1×1 convolutions to change channel counts; projection shortcuts in ResNet are 1×1 convolutions with stride 2.",
          "**Video and medical volumes** — 3-D convolutions (C3D, I3D) for action recognition; 3-D U-Nets with transposed convolutions for CT/MRI segmentation.",
          "**Modern ConvNets** — ConvNeXt (Liu et al., 2022) uses 7×7 depthwise convolutions followed by 1×1 expansions, mirroring the attention + MLP split of a Transformer block.",
        ],
      },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**1×1**: the cheapest way to change channel count and add depth; turns convolutions into a per-pixel MLP.",
          "**Dilation**: receptive field grows exponentially with depth at constant parameter count and full resolution; no information is discarded by pooling.",
          "**Transposed**: learnable upsampling with exactly the right shapes to mirror a strided encoder; implemented by the same kernels as the convolution backward pass.",
          "**Separable / grouped**: 8–9× (separable 3×3) or $g$× (grouped) fewer weights and MACs; they also act as a regulariser on small data.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**FLOPs are not latency.** Depthwise convolutions do very little arithmetic per byte of memory moved, so on GPUs they often run far below peak throughput, and a separable network's wall-clock speedup is usually much smaller than its MAC reduction. Measure latency on the target hardware.",
          "**Dilation** samples sparsely: thin structures can fall between taps, and repeated rates cause gridding. Large rates on small feature maps degenerate (most taps land in the padding), so DeepLab v3 added image-level pooling to ASPP.",
          "**Transposed convolution** produces checkerboard artifacts unless the kernel size is divisible by the stride or the network learns to compensate; its output size is ambiguous without output_padding.",
          "**Separable** convolutions have less capacity per layer than standard ones; reaching the same accuracy usually needs more channels or layers (inverted residuals expand channels by 6× inside each block).",
          "**Grouped** convolutions block information flow between groups unless something mixes them.",
        ],
      },
      { t: "h", text: "Practical trade-offs" },
      {
        t: "table",
        head: ["Decision", "Option A", "Option B"],
        rows: [
          ["Upsampling in a decoder", "Transposed conv (4, 2, 1): learnable, one layer", "Resize + 3×3 conv: no checkerboard, simpler to reason about"],
          ["Large context for dense prediction", "Dilated convs: keep resolution, more memory per layer", "Encoder–decoder with pooling: cheap at low resolution, needs skip connections to recover detail"],
          ["Efficient backbone", "Depthwise-separable: minimal MACs, memory-bound", "Grouped/standard conv: more MACs but higher hardware utilisation"],
          ["Channel reduction", "1×1 bottleneck: big savings, possible information loss", "Full width: more accurate, more expensive"],
        ],
      },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Assuming a transposed convolution inverts a convolution.** $C^\\top$ is not $C^{-1}$: running a convolution and then its transpose does not give the input back. It only reverses the *shapes*. Fix: think of it as a learnable upsampler (or as the gradient map), never as an inverse.",
          "**Using the wrong padding formula for transposed convolutions.** In ConvTranspose2d, `padding` **removes** border pixels from the output; increasing it makes the output smaller. To double the size use (k, s, p) = (4, 2, 1), (2, 2, 0), or (3, 2, 1) with `output_padding=1`.",
          "**Swapping the weight layout.** ConvTranspose2d stores weights as (C_in, C_out, k, k), Conv2d as (C_out, C_in, k, k). Copying weights between them (e.g. for tied encoder–decoder weights) without accounting for this silently mixes channels.",
          "**Forgetting to adjust padding for dilation.** Same padding for a 3×3 kernel with dilation d is p = d, not 1. With p = 1 and d = 2 the feature map shrinks by 2 every layer and skip connections no longer line up. (PyTorch also accepts padding=\"same\" at stride 1.)",
          "**Stacking the same dilation rate repeatedly** (2, 2, 2, …) and getting gridding artifacts. Fix: use rates without a common factor (1, 2, 3, or 1, 2, 5) or interleave d = 1 layers.",
          "**Writing a depthwise layer as nn.Conv2d(C, C, 3)** — without `groups=C` it is a full convolution with C× more weights. Check the weight shape: depthwise is (C, 1, k, k).",
          "**Omitting the pointwise step.** A stack of depthwise convolutions never mixes channels, so each channel is processed in isolation all the way through. Every depthwise layer needs a 1×1 convolution (or grouped conv + shuffle) somewhere after it.",
          "**Judging efficiency by parameter count alone.** Parameters do not depend on H×W; MACs do, and latency also depends on memory traffic. Report MACs at the real input resolution and time the model on the target device.",
          "**Padding causal convolutions symmetrically.** padding=(k−1)·d/2 on both sides lets output t see the future. Causal means left padding of (k−1)·d only (Keras padding=\"causal\"; in PyTorch pad manually with F.pad(x, (d·(k−1), 0)) ).",
        ],
      },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "What does a 1×1 convolution do if it cannot see any neighbours?", a: "It applies the same linear map $W\\in\\mathbb{R}^{C_{out}\\times C_{in}}$ (plus bias) to the channel vector of every pixel: a fully connected layer shared across positions. It changes the number of channels (reducing them before an expensive 3×3 convolution, or expanding them), mixes information across channels, and, followed by a non-linearity, adds depth cheaply. Cost: $C_{out}C_{in}$ weights, $C_{out}C_{in}HW$ multiply-adds." },
      { t: "qa", level: "beginner", q: "What is a dilated convolution and what is its effective kernel size?", a: "A convolution whose kernel taps are spaced $d$ pixels apart instead of adjacent. A $k\\times k$ kernel with dilation $d$ covers $k + (k-1)(d-1)$ pixels per side while keeping $k^2$ weights; a 3×3 kernel with $d = 2$ covers 5×5, with $d = 4$ it covers 9×9. It enlarges the receptive field without pooling or extra parameters." },
      { t: "qa", level: "beginner", q: "What is a transposed convolution used for?", a: "Learnable upsampling: with stride $s$ it maps an $n\\times n$ input to roughly $sn\\times sn$. Each input value scales a copy of the kernel and adds it into the output at stride-$s$ spacing. It is used in decoders of segmentation networks (FCN, U-Net), GAN generators and autoencoders. Its output size is $(n-1)s - 2p + k + \\text{output\\_padding}$." },
      { t: "qa", level: "beginner", q: "What is a depthwise-separable convolution?", a: "A standard convolution factorised into two layers: a depthwise convolution that filters each input channel with its own $k\\times k$ kernel (no channel mixing), followed by a pointwise 1×1 convolution that mixes channels. It has $C_{in}k^2 + C_{in}C_{out}$ weights instead of $C_{out}C_{in}k^2$ — about 8–9× fewer for 3×3 kernels — and is the building block of Xception and MobileNet." },
      { t: "qa", level: "intermediate", q: "Derive the cost ratio of a depthwise-separable to a standard convolution. Why is the real speedup often smaller?", a: "Standard: $C_{out}C_{in}k^2HW$ MACs. Separable: $C_{in}k^2HW + C_{in}C_{out}HW$. Ratio $= 1/C_{out} + 1/k^2$, e.g. $1/64 + 1/9 = 0.127$ for 32→64 channels with 3×3. The speedup is smaller in practice because the depthwise step has very low arithmetic intensity (each loaded value is used for only $k^2$ MACs, versus $C_{out}k^2$ in a standard conv), so it is memory-bandwidth-bound and poorly utilises GPUs and matrix units; the extra layer also adds kernel-launch and activation-memory overhead." },
      { t: "qa", level: "intermediate", q: "Why does dilation give exponential receptive-field growth, and what is gridding?", a: "At stride 1 a layer adds $(k-1)d$ to the receptive field. With $d_\\ell = 2^{\\ell-1}$ the sum is $(k-1)(2^L - 1)$, so $L$ 3×3 layers see $2^{L+1}-1$ pixels — 31 after four layers vs 9 without dilation — while parameters grow only linearly and resolution is kept. Gridding: when stacked rates share a common factor (e.g. all $d = 2$), every combined offset is a multiple of that factor, so each output depends only on a sub-lattice of the input (one quarter of the pixels for $d=2$). Adjacent outputs see disjoint pixel sets, producing grid-like artifacts. Fix: rates with no common factor (hybrid dilated convolution, e.g. 1, 2, 3) or doubling rates starting at 1." },
      { t: "qa", level: "intermediate", q: "Where do checkerboard artifacts come from, and how do you avoid them?", a: "In a stride-$s$ transposed convolution each input pixel stamps a $k\\times k$ kernel at spacing $s$. If $k$ is not divisible by $s$, output pixels are covered by different numbers of stamps (for $k=3, s=2$: 1, 2, 1, 2, … per axis, so 1, 2, 4 in 2-D), which imprints a periodic pattern unless the weights exactly compensate. Avoid by using $k$ divisible by $s$ (e.g. 4 with stride 2), or by resize-convolution: nearest/bilinear upsampling followed by a stride-1 convolution, which has uniform overlap. Odena, Dumoulin and Olah (2016) analysed this in detail." },
      { t: "qa", level: "intermediate", q: "How does grouped convolution relate to standard and depthwise convolution?", a: "With $g$ groups the input and output channels are split into $g$ blocks and block $i$ of the output only reads block $i$ of the input, so each kernel has depth $C_{in}/g$ and the layer has $C_{out}C_{in}k^2/g$ weights. $g = 1$ is standard convolution; $g = C_{in}$ (with $C_{out}$ a multiple of $C_{in}$) is depthwise. AlexNet used $g = 2$ to fit on two GPUs; ResNeXt showed that a grouped bottleneck with $g = 32$ (cardinality) was more accurate than a standard one of equal cost. Groups need a 1×1 convolution or a channel shuffle to exchange information." },
      { t: "qa", level: "advanced", q: "Show that the transposed convolution is the gradient of a convolution with respect to its input.", a: "A convolution is linear in its input, so after flattening $y = Cx$ with a sparse matrix $C$ whose row $i$ holds the kernel weights at the input positions output $i$ reads. For a loss $L$ with $g = \\partial L/\\partial y$, the chain rule gives $\\partial L/\\partial x_j = \\sum_i g_i\\,\\partial y_i/\\partial x_j = \\sum_i g_i C_{ij} = (C^\\top g)_j$. Column $j$ of $C^\\top$ lists every output that read $x_j$, weighted by the kernel entry used, which is precisely the scatter of $g$ through the kernel at stride $s$ — the transposed convolution. Equivalently, $\\langle Cx, g\\rangle = \\langle x, C^\\top g\\rangle$ for all $x, g$. This is why frameworks implement ConvTranspose's forward pass with the convolution's backward-data kernel and vice versa." },
      { t: "qa", level: "advanced", q: "Design a WaveNet-style stack for 16 kHz audio with a receptive field of at least one second. How many layers, and what padding?", a: "With kernel size 2 and dilations 1, 2, …, 512 (10 layers), one block covers $1 + \\sum d = 1024$ samples. One second at 16 kHz is 16,000 samples, so repeat the block: $\\lceil 15{,}999/1023\\rceil = 16$ blocks, i.e. 160 layers, give $1 + 16\\cdot1023 = 16{,}369$ samples. (WaveNet itself repeated the 1–512 block a few times and accepted a shorter context.) Each layer must be causal: left-pad by $d(k-1) = d$ zeros so output $t$ depends only on inputs $\\le t$; training is fully parallel because the whole target sequence is known, while generation is sequential, one sample at a time (caching past activations makes each step cheap)." },
      { t: "qa", level: "advanced", q: "Why do separable networks such as MobileNetV2 use an inverted residual (expand–depthwise–project) block rather than the ResNet bottleneck (reduce–3×3–expand)?", a: "A depthwise 3×3 is so cheap that the cost is dominated by the 1×1 layers; it is therefore affordable to run the depthwise filter on an expanded representation (typically 6× the block's input channels), where spatial filtering has many channels to work with. The block input/output (the residual path) is kept thin to save memory, and the final 1×1 projection is linear (no ReLU), because Sandler et al. (2018) argued that ReLU in a low-dimensional space destroys information. The result: thin, memory-friendly tensors between blocks and wide tensors only inside, the opposite of the ResNet bottleneck, which reduces before an expensive standard 3×3." },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "What is the effective kernel size (span in pixels) of a 3×3 kernel with dilation d = 6, as used in DeepLab's ASPP?", answer: 13, tolerance: 0, explain: "$k + (k-1)(d-1) = 3 + 2\\cdot5 = 13$, or equivalently $d(k-1)+1 = 13$." },
          { kind: "numeric", q: "Five stacked stride-1 3×3 convolutions use dilations 1, 2, 4, 8, 16. What is the receptive-field width of one output?", answer: 63, tolerance: 0, explain: "$1 + (k-1)\\sum d = 1 + 2\\cdot31 = 63 = 2^{6}-1$." },
          { kind: "numeric", q: "A ConvTranspose2d with kernel 4, stride 2, padding 1 (output_padding 0) receives a 16×16 input. What is the output width?", answer: 32, tolerance: 0, explain: "$(16-1)\\cdot2 - 2\\cdot1 + 4 = 32$. The (4, 2, 1) setting exactly doubles the size." },
          { kind: "numeric", q: "How many weights (no biases) does a depthwise-separable 3×3 layer from 128 to 256 channels have?", answer: 33920, tolerance: 0, explain: "Depthwise $128\\cdot9 = 1{,}152$ plus pointwise $128\\cdot256 = 32{,}768$ = 33,920. The standard layer has $256\\cdot128\\cdot9 = 294{,}912$; ratio $0.115 = 1/256 + 1/9$." },
          { kind: "numeric", q: "A grouped 3×3 convolution has 256 input and 256 output channels and 32 groups (ResNeXt style). How many weights does it have (no biases)?", answer: 18432, tolerance: 0, explain: "$C_{out}(C_{in}/g)k^2 = 256\\cdot8\\cdot9 = 18{,}432$, i.e. 1/32 of the standard 589,824." },
          { q: "In PyTorch, which layer is a depthwise convolution on a 64-channel input?", options: ["nn.Conv2d(64, 64, 3, padding=1)", "nn.Conv2d(64, 64, 3, padding=1, groups=64)", "nn.Conv2d(64, 64, 1)", "nn.Conv2d(64, 1, 3, padding=1)"], answer: 1, explain: "groups = in_channels gives each channel its own 3×3 kernel: weight shape (64, 1, 3, 3), 576 weights. The first option is a standard convolution with 36,864 weights; the third is pointwise; the last produces one channel by mixing all of them." },
          { q: "A transposed convolution with kernel 3 and stride 2 is fed all-ones input and an all-ones kernel. What does the 2-D interior of its output look like?", options: ["All values equal (uniform)", "A repeating 1–2–4 pattern (checkerboard)", "All zeros", "Values increasing linearly towards the centre"], answer: 1, explain: "Each output counts the stamps covering it. Along one axis the counts alternate 1, 2 because 3 is not divisible by 2; in 2-D they multiply to 1, 2 and 4 — the checkerboard artifact." },
          { q: "Which statement about the transposed convolution $C^\\top$ of a convolution matrix $C$ is correct?", options: ["$C^\\top C = I$, so it undoes the convolution", "It equals the gradient map $g \\mapsto \\partial L/\\partial x$ of the convolution", "It is only defined for stride 1", "It has no learnable parameters"], answer: 1, explain: "$\\partial L/\\partial x = C^\\top\\,\\partial L/\\partial y$. It reverses shapes but is not an inverse; it is defined for any stride and, used as a layer, its kernel is learned." },
          { q: "Why can't you build a deep network from depthwise convolutions alone?", options: ["They have too many parameters", "They never mix information across channels", "They cannot use padding", "They only work with 1×1 kernels"], answer: 1, explain: "Each output channel depends only on the matching input channel, so channels would be processed in isolation through the whole network. A pointwise (1×1) convolution or a channel shuffle must mix them." },
        ],
      },
      {
        t: "exercise",
        title: "Cost of a MobileNet-style block",
        prompt: "A feature map of size 28×28 with 192 channels goes through (a) a standard 3×3 convolution 192 → 192 (stride 1, same padding), or (b) a depthwise 3×3 followed by a pointwise 192 → 192. For each, compute the weights and the multiply-adds (ignore biases). What is the ratio, and how does it compare with $1/C_{out} + 1/k^2$? What fraction of the separable MACs is spent in the pointwise step?",
        hint: "Weights: $C_{out}C_{in}k^2$ vs $C_{in}k^2 + C_{in}C_{out}$. MACs = weights × H × W for stride-1, same-padded layers.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Standard", tex: String.raw`192\cdot192\cdot9 = 331{,}776 \text{ weights},\quad 331{,}776\cdot784 = 260{,}112{,}384 \text{ MACs}` },
              { title: "Separable", tex: String.raw`192\cdot9 + 192\cdot192 = 1{,}728 + 36{,}864 = 38{,}592 \text{ weights},\quad 38{,}592\cdot784 = 30{,}256{,}128 \text{ MACs}` },
              { title: "Ratio", tex: String.raw`\frac{38{,}592}{331{,}776} = 0.11632 = \frac{1}{192} + \frac{1}{9} \;\;(\approx 8.6\times \text{ fewer})` },
              { title: "Pointwise share", tex: String.raw`\frac{36{,}864}{38{,}592} = 95.5\%` },
            ],
          },
          { t: "p", text: "The depthwise step is under 5% of the arithmetic, which is why efficient-network research focuses on the 1×1 layers (grouping them, as ShuffleNet does, or reducing their width)." },
        ],
      },
      {
        t: "exercise",
        title: "Causal dilated 1-D convolutions from scratch",
        prompt: "Implement `causal_conv1d(x, w, dilation)` in NumPy for x of shape (C_in, T) and w of shape (C_out, C_in, k), producing an output of length T in which output t depends only on inputs ≤ t. Stack four layers with kernel size 2 and dilations 1, 2, 4, 8 (tanh after each). Verify (1) one layer against PyTorch (left-pad, then `F.conv1d` with `dilation`), (2) causality: perturbing x[20] changes no output before t = 20, and (3) the receptive field of the last output by checking which inputs have non-zero gradient, against the formula $1 + (k-1)\\sum d$.",
        hint: "Left-pad by d·(k−1) zeros. Tap m of the kernel then reads xp[:, m·d : m·d + T]: the last tap (m = k−1) is the current time step.",
        solution: [
          { t: "code", lang: "python", title: "causal_dilated_conv.py", code: CAUSAL, output: CAUSAL_OUT },
          { t: "p", text: "The receptive field of y[31] is exactly the 16 inputs t = 16 … 31, matching $1 + 1\\cdot(1+2+4+8) = 16$, and no input after 31 (there are none) or before 16 contributes. Perturbing x[20] changes outputs 20 … 31 only. Doubling the dilations to 16, 32, … would double the receptive field with each extra layer — the WaveNet recipe." },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "**1×1 convolution** = a linear layer applied to every pixel's channel vector: changes channel count, mixes channels, adds depth. Bottlenecks (Inception, ResNet) and the mixing half of separable convs.",
          "**Dilated convolution**: taps $d$ apart, effective size $k + (k-1)(d-1)$, same $k^2$ weights. Doubling rates give receptive field $2^{L+1}-1$ at full resolution; repeated rates cause gridding.",
          "**Transposed convolution** = multiplication by $C^\\top$ = scatter-add of kernel stamps = the input-gradient of a convolution. Output $(n-1)s - 2p + d(k-1) + \\text{op} + 1$. Not an inverse.",
          "**Checkerboard artifacts** come from uneven stamp overlap when $k$ is not divisible by $s$; fix with $k$ divisible by $s$ or resize-convolution.",
          "**Depthwise** (groups = $C_{in}$) filters each channel separately; **pointwise** mixes channels; together (**separable**) they cost $1/C_{out} + 1/k^2$ of a standard conv — 2,336 vs 18,432 weights for 32→64, 3×3.",
          "**Grouped** convolution costs $1/g$; depthwise is the $g = C_{in}$ extreme. Groups need mixing.",
          "In separable networks most MACs are in the 1×1 layers (89% for MobileNetV2), and MACs are not latency: depthwise layers are memory-bound.",
          "1-D convolutions for sequences, 3-D for video/volumes; causal = left padding only (WaveNet).",
        ],
      },
      {
        t: "math",
        tex: String.raw`k_{\text{eff}} = d(k-1)+1,\quad H^{\text{conv}}_{\text{out}} = \left\lfloor\tfrac{H+2p-d(k-1)-1}{s}\right\rfloor+1,\quad H^{\text{T}}_{\text{out}} = (H-1)s - 2p + d(k-1) + \text{op} + 1,\quad \tfrac{\text{sep}}{\text{std}} = \tfrac{1}{C_{out}} + \tfrac{1}{k^2}`,
        caption: "Key formulas of the lesson.",
      },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[Classic CNN Architectures: LeNet to ConvNeXt](/learn/m08/cnn-architectures) — see these variants in their natural habitat: 1×1 bottlenecks in Inception and ResNet, grouped convolutions in ResNeXt, depthwise-separable blocks in Xception, MobileNet and EfficientNet, 7×7 depthwise convolutions in ConvNeXt.",
          "[Semantic, Instance and Panoptic Segmentation](/learn/m12/segmentation) — where dilated convolutions (DeepLab, ASPP) and transposed-convolution decoders (FCN, U-Net) are essential.",
          "[Generative Adversarial Networks](/learn/m14/gans) and [Diffusion Models](/learn/m14/diffusion) — generators and denoising U-Nets that upsample with transposed or resize-convolutions, where checkerboard artifacts matter visibly.",
          "[Quantization, Pruning and Distillation](/learn/m18/model-optimization) — the other half of efficient deployment, complementary to separable architectures.",
          "[Batch, Layer and Group Normalization](/learn/m07/normalization) — group normalization shares the \"split channels into groups\" idea with grouped convolution and is common in the small-batch settings (segmentation, detection) where these variants are used.",
        ],
      },
    ],
  },
};

export default lesson;
