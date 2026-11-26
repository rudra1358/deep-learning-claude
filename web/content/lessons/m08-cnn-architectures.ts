import type { Lesson } from "@/lib/types";

const COSTS_NUMPY = String.raw`import numpy as np


def conv_cost(c_in, c_out, k, h_out, w_out, groups=1, bias=True):
    """Parameters and multiply-accumulates (MACs) of one 2-D convolution layer."""
    params = c_out * (c_in // groups) * k * k + (c_out if bias else 0)
    macs = c_out * (c_in // groups) * k * k * h_out * w_out
    return params, macs


# ---- 1) VGG-16, layer by layer, from its configuration alone ----
cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
c, h = 3, 224
params, macs = [], []
for v in cfg:
    if v == "M":
        h //= 2                                  # 2x2 max-pool, stride 2: no parameters
        continue
    p, m = conv_cost(c, v, 3, h, h)              # 3x3, padding 1: spatial size unchanged
    params.append(p); macs.append(m)
    c = v
conv_p, conv_m = sum(params), sum(macs)
fc = [(512 * 7 * 7, 4096), (4096, 4096), (4096, 1000)]
fc_p = sum(i * o + o for i, o in fc)
fc_m = sum(i * o for i, o in fc)
print(f"VGG-16 conv layers: {conv_p:,} params, {conv_m / 1e9:.3f} G MACs")
print(f"VGG-16 FC layers:   {fc_p:,} params, {fc_m / 1e9:.3f} G MACs")
print(f"VGG-16 total:       {conv_p + fc_p:,} params, {(conv_m + fc_m) / 1e9:.3f} G MACs")
print(f"share of parameters in FC layers: {fc_p / (conv_p + fc_p):.1%}; "
      f"share of MACs in conv layers: {conv_m / (conv_m + fc_m):.1%}")
top = [i + 1 for i, m in enumerate(macs) if m == max(macs)]
print(f"conv layers tied for the largest cost ({max(macs) / 1e6:.1f} M MACs each): {top}")

# ---- 2) Stacked 3x3 vs one large kernel (C channels in and out, bias ignored) ----
C = 256
for n, k_big in [(2, 5), (3, 7)]:
    stacked, big = n * 9 * C * C, k_big * k_big * C * C
    print(f"{n} x (3x3) vs one {k_big}x{k_big}: {stacked:,} vs {big:,} weights "
          f"({stacked / big:.0%}), same {k_big}x{k_big} receptive field")

# ---- 3) Inception 3a, 5x5 branch: 1x1 bottleneck before the expensive conv ----
H = W = 28
_, direct = conv_cost(192, 32, 5, H, W)
_, reduce = conv_cost(192, 16, 1, H, W)
_, after = conv_cost(16, 32, 5, H, W)
print(f"5x5 branch on 28x28x192: direct {direct / 1e6:.1f} M MACs, "
      f"with 1x1 to 16 ch {(reduce + after) / 1e6:.1f} M MACs ({direct / (reduce + after):.1f}x cheaper)")

# ---- 4) Depthwise separable vs standard 3x3 conv, 56x56, 128 -> 128 ----
H = W = 56
p_std, m_std = conv_cost(128, 128, 3, H, W, bias=False)
p_dw, m_dw = conv_cost(128, 128, 3, H, W, groups=128, bias=False)
p_pw, m_pw = conv_cost(128, 128, 1, H, W, bias=False)
print(f"standard 3x3:  {p_std:,} params, {m_std / 1e6:.1f} M MACs")
print(f"depthwise+1x1: {p_dw + p_pw:,} params, {(m_dw + m_pw) / 1e6:.1f} M MACs "
      f"(ratio {(m_dw + m_pw) / m_std:.4f}; formula 1/C_out + 1/k^2 = {1 / 128 + 1 / 9:.4f})")

# ---- 5) A depthwise separable conv IS a standard conv with a rank-1-per-channel kernel ----
rng = np.random.default_rng(0)
c_in, c_out, k, n = 3, 4, 3, 6
x = rng.standard_normal((c_in, n, n))
D = rng.standard_normal((c_in, k, k))          # one k x k filter per input channel
P = rng.standard_normal((c_out, c_in))         # 1x1 mixing matrix


def corr2d(img, ker):                           # valid cross-correlation of one channel
    m = img.shape[0] - ker.shape[0] + 1
    return np.array([[np.sum(img[i:i + k, j:j + k] * ker) for j in range(m)] for i in range(m)])


depthwise = np.stack([corr2d(x[c], D[c]) for c in range(c_in)])           # (c_in, 4, 4)
separable = np.einsum("oc,chw->ohw", P, depthwise)                         # pointwise mixing
W_equiv = P[:, :, None, None] * D[None]                                    # (c_out, c_in, k, k)
standard = np.stack([sum(corr2d(x[c], W_equiv[o, c]) for c in range(c_in)) for o in range(c_out)])
print("separable == standard conv with kernel W[o,c] = P[o,c] * D[c]:", np.allclose(separable, standard))`;

const COSTS_NUMPY_OUT = String.raw`VGG-16 conv layers: 14,714,688 params, 15.347 G MACs
VGG-16 FC layers:   123,642,856 params, 0.124 G MACs
VGG-16 total:       138,357,544 params, 15.470 G MACs
share of parameters in FC layers: 89.4%; share of MACs in conv layers: 99.2%
conv layers tied for the largest cost (1849.7 M MACs each): [2, 4, 6, 7, 9, 10]
2 x (3x3) vs one 5x5: 1,179,648 vs 1,638,400 weights (72%), same 5x5 receptive field
3 x (3x3) vs one 7x7: 1,769,472 vs 3,211,264 weights (55%), same 7x7 receptive field
5x5 branch on 28x28x192: direct 120.4 M MACs, with 1x1 to 16 ch 12.4 M MACs (9.7x cheaper)
standard 3x3:  147,456 params, 462.4 M MACs
depthwise+1x1: 17,536 params, 55.0 M MACs (ratio 0.1189; formula 1/C_out + 1/k^2 = 0.1189)
separable == standard conv with kernel W[o,c] = P[o,c] * D[c]: True`;

const BLOCKS_TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, x):
    """Multiply-accumulates of all Conv2d/Linear layers for one forward pass of x (per image)."""
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            total += out.numel() * m.in_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total // x.shape[0]


def conv_bn_relu(c_in, c_out, k, stride=1, groups=1, act=True):
    """k x k conv ('same' padding for odd k) + BatchNorm (+ ReLU). Bias is dropped: BN's shift replaces it."""
    layers = [nn.Conv2d(c_in, c_out, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out)]
    return nn.Sequential(*layers, nn.ReLU(inplace=True)) if act else nn.Sequential(*layers)


# ---------------- VGG block: n x (3x3 conv + ReLU), then 2x2 max-pool ----------------
def vgg_block(c_in, c_out, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


# ---------------- Inception module (GoogLeNet style, with 1x1 reductions) ----------------
class Inception(nn.Module):
    def __init__(self, c_in, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = conv_bn_relu(c_in, c1, 1)
        self.b2 = nn.Sequential(conv_bn_relu(c_in, c3r, 1), conv_bn_relu(c3r, c3, 3))
        self.b3 = nn.Sequential(conv_bn_relu(c_in, c5r, 1), conv_bn_relu(c5r, c5, 5))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv_bn_relu(c_in, cp, 1))

    def forward(self, x):                          # four parallel branches, concatenated on channels
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# ---------------- ResNet basic block: two 3x3 convs + identity (or 1x1 projection) shortcut ----------------
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(conv_bn_relu(c_in, c_out, 3, stride), conv_bn_relu(c_out, c_out, 3, act=False))
        self.shortcut = (nn.Identity() if stride == 1 and c_in == c_out
                         else conv_bn_relu(c_in, c_out, 1, stride, act=False))

    def forward(self, x):
        return F.relu(self.f(x) + self.shortcut(x))  # y = relu(F(x) + x)


# ---------------- DenseNet dense block: every layer sees the concatenation of all earlier maps ----------------
class DenseLayer(nn.Module):
    def __init__(self, c_in, growth, bn_size=4):     # BN-ReLU-conv1x1 (4k ch) - BN-ReLU-conv3x3 (k ch)
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm2d(c_in), nn.ReLU(inplace=True), nn.Conv2d(c_in, bn_size * growth, 1, bias=False),
                               nn.BatchNorm2d(bn_size * growth), nn.ReLU(inplace=True),
                               nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.f(x)], dim=1)       # append k new channels, keep the old ones


def dense_block(c_in, growth, n_layers):
    return nn.Sequential(*[DenseLayer(c_in + i * growth, growth) for i in range(n_layers)])


# ---------------- MobileNetV2 inverted residual: expand 1x1 -> depthwise 3x3 -> project 1x1 (linear) ----------------
class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, stride=1, expand=6):
        super().__init__()
        hidden = c_in * expand
        self.use_res = stride == 1 and c_in == c_out
        self.f = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, c_out, 1, bias=False), nn.BatchNorm2d(c_out))   # linear bottleneck: no ReLU

    def forward(self, x):
        return x + self.f(x) if self.use_res else self.f(x)


# ---------------- ConvNeXt block: dw 7x7 -> LayerNorm -> 1x1 (4x) -> GELU -> 1x1 -> layer scale -> + x ----------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)                 # normalises over channels at each pixel
        self.pw1 = nn.Linear(dim, 4 * dim)            # a 1x1 conv written as a Linear on channels-last data
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)            # N,C,H,W -> N,H,W,C
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 3, 1, 2)


bn = lambda c: 2 * c                                   # BatchNorm/LayerNorm: one scale + one shift per channel
tests = [
    ("VGG block (64->128, 2 convs)", vgg_block(64, 128, 2), (64, 56, 56),
     (64 * 9 * 128 + 128) + (128 * 9 * 128 + 128)),
    ("Inception 3a (192->256)", Inception(192, 64, 96, 128, 16, 32, 32), (192, 28, 28),
     192 * 64 + bn(64) + 192 * 96 + bn(96) + 96 * 9 * 128 + bn(128)
     + 192 * 16 + bn(16) + 16 * 25 * 32 + bn(32) + 192 * 32 + bn(32)),
    ("ResNet basic (64->64)", BasicBlock(64, 64), (64, 56, 56), 2 * (64 * 9 * 64 + bn(64))),
    ("ResNet basic (64->128, /2)", BasicBlock(64, 128, 2), (64, 56, 56),
     64 * 9 * 128 + bn(128) + 128 * 9 * 128 + bn(128) + 64 * 128 + bn(128)),
    ("Dense block (64 + 6x32)", dense_block(64, 32, 6), (64, 56, 56),
     sum(bn(64 + 32 * i) + (64 + 32 * i) * 128 + bn(128) + 128 * 9 * 32 for i in range(6))),
    ("Inverted residual (24->24, t=6)", InvertedResidual(24, 24), (24, 56, 56),
     24 * 144 + bn(144) + 144 * 9 + bn(144) + 144 * 24 + bn(24)),
    ("ConvNeXt block (96)", ConvNeXtBlock(96), (96, 56, 56),
     96 * 49 + 96 + bn(96) + 96 * 384 + 384 + 384 * 96 + 96 + 96),
]
print(f"{'block':32s} {'input C,H,W':>13s} {'output C,H,W':>13s} {'params':>8s} {'MACs (M)':>8s}  formula")
for name, block, shape, formula in tests:
    block.eval()
    x = torch.randn(2, *shape)
    with torch.no_grad():
        out = block(x)
    print(f"{name:32s} {str(shape):>13s} {str(tuple(out.shape[1:])):>13s} {n_params(block):8,d} "
          f"{count_macs(block, x) / 1e6:8.1f}  {'ok' if n_params(block) == formula else 'MISMATCH ' + str(formula)}")

# Residual blocks start close to the identity map when their last BN/scale is small:
blk = ConvNeXtBlock(96).eval()
x = torch.randn(1, 96, 14, 14)
print(f"ConvNeXt block at init: max |block(x) - x| = {(blk(x) - x).abs().max().item():.1e}")`;

const BLOCKS_TORCH_OUT = String.raw`block                              input C,H,W  output C,H,W   params MACs (M)  formula
VGG block (64->128, 2 convs)      (64, 56, 56) (128, 28, 28)  221,440    693.6  ok
Inception 3a (192->256)          (192, 28, 28) (256, 28, 28)  164,064    128.0  ok
ResNet basic (64->64)             (64, 56, 56)  (64, 56, 56)   73,984    231.2  ok
ResNet basic (64->128, /2)        (64, 56, 56) (128, 28, 28)  230,144    179.8  ok
Dense block (64 + 6x32)           (64, 56, 56) (256, 56, 56)  335,040   1040.4  ok
Inverted residual (24->24, t=6)   (24, 56, 56)  (24, 56, 56)    8,832     25.7  ok
ConvNeXt block (96)               (96, 56, 56)  (96, 56, 56)   79,296    246.0  ok
ConvNeXt block at init: max |block(x) - x| = 8.0e-07`;

const PLAIN_VS_RES = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)            # single-threaded CPU kernels: faster for tiny tensors and bit-reproducible

digits = load_digits()              # 1,797 grayscale 8x8 digits, values 0..16
X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0     # (N, 1, 8, 8)
y = torch.tensor(digits.target)
i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=digits.target)
X_tr, X_te, y_tr, y_te = X[i_tr], X[i_te], y[i_tr], y[i_te]


class Block(nn.Module):
    """conv-BN-ReLU-conv-BN, then ReLU. residual=True adds the shortcut x (projected if the shape changes)."""
    def __init__(self, c_in, c_out, stride, residual):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
                               nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.short = nn.Identity()
        if residual and (stride != 1 or c_in != c_out):
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.f(x) + self.short(x)) if self.residual else F.relu(self.f(x))


def make_net(blocks_per_stage, residual):
    """stem (16 ch, 8x8) -> stage 1 (16 ch, 8x8) -> stage 2 (32 ch, 4x4) -> global avg pool -> linear.
    Weighted layers = 1 stem + 2 stages * blocks * 2 convs + 1 linear = 4 * blocks + 2."""
    layers, c = [nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()], 16
    for stage, width in enumerate([16, 32]):
        for i in range(blocks_per_stage):
            layers.append(Block(c, width, 2 if (stage == 1 and i == 0) else 1, residual))
            c = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 10))


EPOCHS, BATCH, LR = 8, 64, 1e-3
print(f"{'model':12s} {'params':>7s} {'|grad stem| at init':>19s}  train loss per epoch (1..8)               test acc")
for blocks in [1, 5]:
    for residual in [False, True]:
        torch.manual_seed(0)                                   # identical data order for every model
        net = make_net(blocks, residual)
        name = f"{'ResNet' if residual else 'plain'}-{4 * blocks + 2}"
        # how much gradient reaches the very first conv layer before any training?
        F.cross_entropy(net(X_tr[:256]), y_tr[:256]).backward()
        g_stem = net[0].weight.grad.norm().item()
        net.zero_grad()
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        losses = []
        for epoch in range(EPOCHS):
            net.train()
            perm, total = torch.randperm(len(X_tr)), 0.0
            for i in range(0, len(X_tr), BATCH):
                b = perm[i:i + BATCH]
                loss = F.cross_entropy(net(X_tr[b]), y_tr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            losses.append(total / len(X_tr))
        net.eval()                                             # BN uses its running statistics
        with torch.no_grad():
            acc = (net(X_te).argmax(1) == y_te).float().mean().item()
        n_par = sum(p.numel() for p in net.parameters())
        print(f"{name:12s} {n_par:7,d} {g_stem:19.3f}  {' '.join(f'{l:.2f}' for l in losses)}  {acc:.3f}")`;

const PLAIN_VS_RES_OUT = String.raw`model         params |grad stem| at init  train loss per epoch (1..8)               test acc
plain-6       19,130               0.130  2.01 1.53 1.19 0.90 0.66 0.48 0.36 0.27  0.980
ResNet-6      19,706               0.158  2.00 1.42 1.01 0.70 0.50 0.35 0.28 0.22  0.980
plain-22     112,058               0.705  2.06 1.56 1.26 1.00 0.75 0.58 0.42 0.30  0.947
ResNet-22    112,634               0.483  1.20 0.29 0.11 0.05 0.03 0.07 0.05 0.09  0.951`;

const ZOO_TORCH = String.raw`import torch
import torch.nn as nn
from torchvision import models


def count_macs(model, input_shape):
    """Multiply-accumulates of every Conv2d and Linear layer for ONE image (forward hooks).
    conv:   out_elements * (C_in / groups) * kH * kW      linear: out_elements * in_features
    BatchNorm, activations, pooling and additions are ignored (they are a few % at most)."""
    total = 0

    def conv_hook(m, inp, out):
        nonlocal total
        total += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        nonlocal total
        total += out.numel() * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device="meta"))
    for h in hooks:
        h.remove()
    return total


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------- the two architectures torchvision does not ship: written by hand ----------
def lenet5():
    """LeNet-5 in its common modern form (full C3 connectivity, plain average pooling)."""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),        # 1x32x32 -> 6x28x28 -> 6x14x14
        nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),       # -> 16x10x10 -> 16x5x5
        nn.Flatten(), nn.Linear(400, 120), nn.Tanh(),
        nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))


def dw_separable(c_in, c_out, stride):
    """MobileNet-v1 unit: 3x3 depthwise conv + BN + ReLU, then 1x1 pointwise conv + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False), nn.BatchNorm2d(c_in), nn.ReLU(inplace=True),
        nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


def mobilenet_v1(width=1.0):
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
           (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]
    c = int(32 * width)
    layers = [nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
    for out, s in cfg:
        layers.append(dw_separable(c, int(out * width), s))
        c = int(out * width)
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 1000)]
    return nn.Sequential(*layers)


class SepConv(nn.Sequential):
    """Xception separable conv: depthwise 3x3 then pointwise 1x1 (no non-linearity between), then BN."""
    def __init__(self, c_in, c_out):
        super().__init__(nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False),
                         nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))


class XBlock(nn.Module):
    """Residual block of separable convs. If downsample: end with 3x3/2 max-pool and a strided 1x1 shortcut."""
    def __init__(self, c_in, chans, downsample, relu_first=True):
        super().__init__()
        layers, c = [], c_in
        for i, c_out in enumerate(chans):
            if i > 0 or relu_first:
                layers.append(nn.ReLU(inplace=False))
            layers.append(SepConv(c, c_out))
            c = c_out
        if downsample:
            layers.append(nn.MaxPool2d(3, 2, 1))
            self.skip = nn.Sequential(nn.Conv2d(c_in, c, 1, 2, bias=False), nn.BatchNorm2d(c))
        else:
            self.skip = nn.Identity()
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x) + self.skip(x)


def xception():
    entry = [nn.Conv2d(3, 32, 3, 2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
             nn.Conv2d(32, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
             XBlock(64, [128, 128], True, relu_first=False),
             XBlock(128, [256, 256], True), XBlock(256, [728, 728], True)]
    middle = [XBlock(728, [728, 728, 728], False) for _ in range(8)]
    exit_ = [XBlock(728, [728, 1024], True),
             SepConv(1024, 1536), nn.ReLU(inplace=True), SepConv(1536, 2048), nn.ReLU(inplace=True),
             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 1000)]
    return nn.Sequential(*entry, *middle, *exit_)


ZOO = [  # name, year, constructor, input (C, H, W)
    ("LeNet-5", 1998, lenet5, (1, 32, 32)),
    ("AlexNet", 2012, lambda: models.alexnet(weights=None), (3, 224, 224)),
    ("VGG-16", 2014, lambda: models.vgg16(weights=None), (3, 224, 224)),
    ("GoogLeNet", 2014, lambda: models.googlenet(weights=None, aux_logits=False, init_weights=True), (3, 224, 224)),
    ("Inception-v3", 2015, lambda: models.inception_v3(weights=None, aux_logits=False, init_weights=True), (3, 299, 299)),
    ("ResNet-18", 2015, lambda: models.resnet18(weights=None), (3, 224, 224)),
    ("ResNet-50", 2015, lambda: models.resnet50(weights=None), (3, 224, 224)),
    ("ResNet-152", 2015, lambda: models.resnet152(weights=None), (3, 224, 224)),
    ("DenseNet-121", 2016, lambda: models.densenet121(weights=None), (3, 224, 224)),
    ("Xception", 2016, xception, (3, 299, 299)),
    ("MobileNet-v1", 2017, mobilenet_v1, (3, 224, 224)),
    ("MobileNet-v2", 2018, lambda: models.mobilenet_v2(weights=None), (3, 224, 224)),
    ("MobileNet-v3-L", 2019, lambda: models.mobilenet_v3_large(weights=None), (3, 224, 224)),
    ("MobileNet-v3-S", 2019, lambda: models.mobilenet_v3_small(weights=None), (3, 224, 224)),
    ("EfficientNet-B0", 2019, lambda: models.efficientnet_b0(weights=None), (3, 224, 224)),
    ("ConvNeXt-T", 2022, lambda: models.convnext_tiny(weights=None), (3, 224, 224)),
]

print(f"{'model':16s} {'year':>4s} {'input':>8s} {'parameters':>12s} {'MACs (M)':>9s}")
for name, year, build, shape in ZOO:
    with torch.device("meta"):          # build without allocating or initialising real weights
        net = build()
    print(f"{name:16s} {year:4d} {shape[1]:>4d}x{shape[2]:<3d} {n_params(net):12,d} "
          f"{count_macs(net, shape) / 1e6:9.1f}")

print("\nEfficientNet compound scaling (each model at its own input resolution)")
for b, res in [(0, 224), (1, 240), (2, 260), (3, 300), (4, 380)]:
    with torch.device("meta"):
        net = getattr(models, f"efficientnet_b{b}")(weights=None)
    print(f"  B{b} @ {res}: {n_params(net) / 1e6:6.2f} M params, {count_macs(net, (3, res, res)) / 1e9:5.2f} G MACs")`;

const ZOO_TORCH_OUT = String.raw`model            year    input   parameters  MACs (M)
LeNet-5          1998   32x32        61,706       0.4
AlexNet          2012  224x224   61,100,840     714.2
VGG-16           2014  224x224  138,357,544   15470.3
GoogLeNet        2014  224x224    6,624,904    1498.4
Inception-v3     2015  299x299   23,834,568    5713.2
ResNet-18        2015  224x224   11,689,512    1814.1
ResNet-50        2015  224x224   25,557,032    4089.2
ResNet-152       2015  224x224   60,192,808   11513.6
DenseNet-121     2016  224x224    7,978,856    2834.2
Xception         2016  299x299   22,855,952    8357.4
MobileNet-v1     2017  224x224    4,231,976     568.7
MobileNet-v2     2018  224x224    3,504,872     300.8
MobileNet-v3-L   2019  224x224    5,483,032     216.6
MobileNet-v3-S   2019  224x224    2,542,856      56.5
EfficientNet-B0  2019  224x224    5,288,548     385.8
ConvNeXt-T       2022  224x224   28,589,128    4455.5

EfficientNet compound scaling (each model at its own input resolution)
  B0 @ 224:   5.29 M params,  0.39 G MACs
  B1 @ 240:   7.79 M params,  0.69 G MACs
  B2 @ 260:   9.11 M params,  0.99 G MACs
  B3 @ 300:  12.23 M params,  1.83 G MACs
  B4 @ 380:  19.34 M params,  4.39 G MACs`;

const BOTTLENECK_SOL = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as TVBottleneck

torch.manual_seed(0)


class Bottleneck(nn.Module):
    """ResNet-50-style block: 1x1 reduce -> 3x3 (carries the stride, as in torchvision) -> 1x1 expand (x4)."""
    expansion = 4

    def __init__(self, c_in, width, stride=1):
        super().__init__()
        c_out = width * self.expansion
        self.f = nn.Sequential(
            nn.Conv2d(c_in, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.f(x) + self.shortcut(x))


ours = Bottleneck(256, 64)
x = torch.randn(1, 256, 56, 56)
print("output shape:", tuple(ours(x).shape))
n = sum(p.numel() for p in ours.parameters())
print(f"params: {n:,}  (conv weights {256*64 + 9*64*64 + 64*256:,} + BN {2*(64+64+256)})")

# same parameter count as torchvision's block, and identical outputs once the weights are copied
tv = TVBottleneck(256, 64)
print("torchvision block params:", f"{sum(p.numel() for p in tv.parameters()):,}")
src = [m for m in tv.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
dst = [m for m in ours.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
for a, b in zip(src, dst):
    b.load_state_dict(a.state_dict())
ours.eval(); tv.eval()
with torch.no_grad():
    print("max |ours - torchvision| =", (ours(x) - tv(x)).abs().max().item())`;

const BOTTLENECK_SOL_OUT = String.raw`output shape: (1, 256, 56, 56)
params: 70,400  (conv weights 69,632 + BN 768)
torchvision block params: 70,400
max |ours - torchvision| = 0.0`;

const ZOO_KERAS = String.raw`import tensorflow as tf

A = tf.keras.applications
# weights=None: random initialisation, nothing is downloaded. Each call builds the full ImageNet classifier.
zoo = {
    "VGG16":           lambda: A.VGG16(weights=None),
    "InceptionV3":     lambda: A.InceptionV3(weights=None),            # default input 299x299
    "ResNet50":        lambda: A.ResNet50(weights=None),
    "DenseNet121":     lambda: A.DenseNet121(weights=None),
    "Xception":        lambda: A.Xception(weights=None),               # default input 299x299
    "MobileNet":       lambda: A.MobileNet(weights=None),              # v1, alpha (width multiplier) = 1.0
    "MobileNetV2":     lambda: A.MobileNetV2(weights=None),
    "MobileNetV3Large": lambda: A.MobileNetV3Large(weights=None),
    "EfficientNetB0":  lambda: A.EfficientNetB0(weights=None),
    "ConvNeXtTiny":    lambda: A.ConvNeXtTiny(weights=None),           # needs TF/Keras >= 2.10
}
for name, build in zoo.items():
    model = build()
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"{name:17s} input {model.input_shape[1:]}  params {model.count_params():>11,d}  trainable {trainable:>11,d}")
    tf.keras.backend.clear_session()

# Transfer-learning form: drop the ImageNet head and add your own
base = A.EfficientNetB0(weights=None, include_top=False, pooling="avg", input_shape=(224, 224, 3))
base.trainable = False                       # freeze the backbone (feature extraction)
model = tf.keras.Sequential([base, tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(5, activation="softmax")])
model.summary()`;

const lesson: Lesson = {
  id: "m08/cnn-architectures",
  title: "Classic CNN Architectures: LeNet to ConvNeXt",
  summary:
    "Twenty-four years of CNN design, one idea at a time: LeNet-5, AlexNet, VGG, GoogLeNet/Inception, ResNet, DenseNet, Xception, MobileNet v1–v3, EfficientNet and ConvNeXt — the problem each solved, its block, its measured parameters and multiply-adds, its pros and cons — with every block built and checked in PyTorch, a real plain-vs-residual training comparison, and a practical guide to picking a backbone today.",
  minutes: 110,
  level: "intermediate",
  prerequisites: ["m08/convolution", "m08/pooling-receptive-field", "m08/conv-variants", "m07/residual-connections", "m07/normalization"],
  next: ["m12/transfer-learning", "m11/transformer-families", "m12/object-detection", "m18/model-optimization"],
  terms: ["lenet", "alexnet", "vgg", "inception-module", "bottleneck-layer", "resnet", "densenet", "growth-rate", "depthwise-separable-convolution", "inverted-residual", "compound-scaling", "convnext", "macs", "backbone"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "A **CNN architecture** is a recipe for stacking convolutions, normalisation, non-linearities, pooling and shortcuts: how many layers, how wide each is, where the resolution shrinks, and which repeated **block** the network is built from. [The Convolution Operation](/learn/m08/convolution) covered the operations; this lesson is about the *design* choices, which turned out to matter just as much." },
      { t: "p", text: "From 1998 to 2022, a handful of architectures each brought one idea the field adopted. **LeNet-5** (1998): an end-to-end trained CNN reads handwriting. **AlexNet** (2012): that recipe scaled to ImageNet on GPUs, with ReLU and dropout. **VGG** (2014): depth from uniform 3×3 convolutions. **GoogLeNet/Inception** (2014): parallel branches and cheap 1×1 *bottlenecks*. **ResNet** (2015): identity shortcuts make 50–150+ layers trainable. **DenseNet** (2016): every layer connected to every later one. **Xception** and **MobileNet** (2016–2017): **depthwise separable** convolutions, ~8–9× less compute. **MobileNetV2**: the *inverted residual*. **EfficientNet** (2019): a searched base network scaled in depth, width and resolution together. **ConvNeXt** (2022): a ResNet modernised with Vision Transformer ideas, matching them." },
      { t: "p", text: "Why study models that are no longer state of the art? They are still the **backbones** (feature extractors) inside detectors, segmenters, medical systems and mobile apps; each idea (3×3 stacking, 1×1 bottlenecks, residuals, depthwise convs, compound scaling) is reusable; and reasoning about parameters, multiply-adds, memory and receptive field is a core skill. Every cost figure here was **measured** by building the model (torchvision, `weights=None`) and counting with forward hooks (code in the Framework section)." },
      { t: "callout", kind: "key", text: "Each landmark answered one question: *how do we get more accuracy per parameter, per multiply-add, and per unit of training difficulty?* Depth (VGG), cheap width (Inception), trainable depth (ResNet), feature reuse (DenseNet), cheap convolutions (Xception/MobileNet), balanced scaling (EfficientNet), modern training and overall design (ConvNeXt)." },
      { t: "p", text: "The site's [Architecture Explorer](/architectures) computes every layer's output shape and parameter count for LeNet-5, AlexNet and VGG-16 — keep it open alongside this lesson." },
    ],

    // ---------------------------------------------------------- prerequisites
    prerequisites: [
      { t: "list", items: [
        String.raw`[The Convolution Operation](/learn/m08/convolution) — the output-size formula, kernels of shape $C_{out}\times C_{in}\times k\times k$, and the parameter and multiply-add formulas. Every cost here applies them.`,
        "[Pooling, Receptive Fields and CNN Architecture](/learn/m08/pooling-receptive-field) — pooling (including global average pooling) and how stacking layers grows the receptive field, which we use to compare 3×3 stacks with large kernels.",
        "[Dilated, Transposed, Depthwise and Pointwise Convolutions](/learn/m08/conv-variants) — depthwise and 1×1 convolutions, the building blocks of Xception, MobileNet, EfficientNet and ConvNeXt. We re-derive their cost briefly.",
        "[Residual and Skip Connections](/learn/m07/residual-connections) — why $y = x + F(x)$ keeps gradients flowing and why very deep plain networks *degrade*. We use residual blocks without re-deriving them.",
        "[Batch, Layer and Group Normalization](/learn/m07/normalization) — BatchNorm is in every architecture after 2015; ConvNeXt swaps it for LayerNorm. Know that BN has a learnable scale and shift per channel (2 parameters each) and behaves differently in training and evaluation.",
        "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) and [Weight Initialization](/learn/m07/initialization) — the training difficulties that pushed the move from AlexNet/VGG to ResNet.",
      ] },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "A factory that keeps improving", text: "Think of a CNN as an assembly line turning pixels into a decision. **LeNet/AlexNet**: a short line of skilled generalists. **VGG**: many simple identical stations (3×3 convs) beat a few elaborate ones, but the line gets long and costly. **Inception**: parallel specialists at each station (small, medium, large windows), each handed a short summary (a 1×1 conv) instead of the full file. **ResNet**: a conveyor belt carries the product *past* every station, and stations only add corrections, so a useless station does no harm and 150 stations still work. **DenseNet**: every station gets a copy of everything made so far. **MobileNet/Xception**: each specialist inspects only their own material (depthwise), then a coordinator mixes the results (pointwise). **EfficientNet**: when the budget doubles, grow line length, workers per station *and* material resolution in a fixed ratio. **ConvNeXt**: an old ResNet factory adopts a newer rival's (the Transformer's) practices and becomes just as productive." },
      { t: "p", text: "Almost every architecture has the same three-part skeleton. A **stem** quickly cuts the resolution (e.g. a 7×7 stride-2 convolution plus pooling turns 224×224 into 56×56). Then come **stages**: inside a stage the resolution stays fixed and one block repeats; between stages the resolution halves and the channel count usually doubles. Finally a **head** — in modern designs, global average pooling plus one linear layer — turns the last feature map into class scores. Families differ in the **block** and in how they split the budget between depth, width and resolution." },
      { t: "p", text: "Two budgets rule every design. **Parameters** set model size (memory to store, data needed to fit them). **Multiply-accumulates (MACs)** — one multiply plus one add, the main work of convolutions and linear layers — set compute. They can be very different: VGG-16's three fully connected layers hold 89% of its parameters but under 1% of its compute, while its 3×3 convolutions do over 99% of the multiply-adds. Real **latency** depends on MACs *and* on memory traffic, parallelism and kernel support, so fewer MACs does not always mean faster." },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "p", text: "For each architecture: problem, key idea, block, a short layer table, measured cost, and pros, cons and uses. Costs are for one 224×224 RGB image unless stated, counting only convolution and linear multiply-adds (BatchNorm, activations, pooling and additions add a few percent)." },

      // ---- LeNet-5
      { t: "h", text: "LeNet-5 (LeCun, Bottou, Bengio, Haffner, 1998)" },
      { t: "p", text: "**Problem.** Reading handwritten digits (postcodes, bank cheques) with hand-designed features was fragile. **Idea.** Learn the features: alternate *convolution* (shared local feature detectors) and *subsampling* (local averaging), and train the whole pipeline by gradient descent (\"Gradient-Based Learning Applied to Document Recognition\")." },
      { t: "table", head: ["Layer", "Output (C×H×W)", "Parameters (modern form)"], rows: [
        ["Input", "1×32×32", "0"],
        ["C1: conv 5×5, 6 maps", "6×28×28", "6·25 + 6 = 156"],
        ["S2: 2×2 average pool", "6×14×14", "0"],
        ["C3: conv 5×5, 16 maps", "16×10×10", "16·6·25 + 16 = 2,416"],
        ["S4: 2×2 average pool", "16×5×5", "0"],
        ["C5/F5: dense 400 → 120", "120", "48,120"],
        ["F6: dense 120 → 84", "84", "10,164"],
        ["Output: dense 84 → 10", "10", "850"],
      ] },
      { t: "p", text: "Measured: **61,706 parameters, 0.4 M MACs** for the common modern form (tanh, full C3 connectivity). The 1998 original differed in details — C3 connected each map to only some S2 maps, subsampling had trainable coefficients, the output used radial-basis units — so its counts differ slightly. **Pros:** tiny, fast, the template for every later CNN (conv → non-linearity → downsample, then a classifier). **Cons:** saturating tanh and the small scale of the time (no GPUs, small datasets) limited depth; far too small for natural images. **Uses today:** teaching, MNIST-like tasks, microcontroller-sized classifiers." },

      // ---- AlexNet
      { t: "h", text: "AlexNet (Krizhevsky, Sutskever, Hinton, 2012)" },
      { t: "p", text: "**Problem.** Scale CNNs to ImageNet (1.2 million training images, 1,000 classes), then thought out of reach for neural networks. **Ideas** (\"ImageNet Classification with Deep Convolutional Neural Networks\"): **ReLU** instead of tanh (doesn't saturate, trains much faster); **GPU training**, split across two GPUs with only 3 GB each; **dropout** in the big fully connected layers; heavy **data augmentation** (random crops, flips, colour changes); overlapping max-pooling and local response normalisation (LRN, later dropped). It won ILSVRC-2012 by a wide margin and started the deep-learning era in vision." },
      { t: "table", head: ["Stage", "torchvision AlexNet", "Output"], rows: [
        ["conv1", "11×11, 64, stride 4, pad 2 → ReLU → max-pool 3×3/2", "64×27×27"],
        ["conv2", "5×5, 192, pad 2 → ReLU → max-pool 3×3/2", "192×13×13"],
        ["conv3–5", "3×3: 384, 256, 256 → ReLU each → max-pool 3×3/2", "256×6×6"],
        ["fc6, fc7", "dropout → 9,216 → 4,096 → ReLU, dropout → 4,096 → 4,096 → ReLU", "4,096"],
        ["fc8", "4,096 → 1,000", "1,000"],
      ] },
      { t: "p", text: "Measured (torchvision single-GPU variant): **61,100,840 parameters, 714 M MACs.** Only 2,469,696 parameters (4%) are in the five convolutions; the fully connected layers hold 58.6 M. **Pros:** proved that deep CNNs + data + GPUs work; introduced the ReLU/dropout/augmentation toolkit still used today. **Cons:** huge dense layers that overfit; big early kernels (11×11, stride 4) throw away detail and waste parameters; LRN costs much for little gain; ad-hoc layer sizes. **Uses today:** historical baseline, teaching; rarely deployed." },

      // ---- VGG
      { t: "h", text: "VGG-16/19 (Simonyan and Zisserman, 2014)" },
      { t: "p", text: "**Problem.** How much does *depth* matter if everything else stays simple? **Idea** (\"Very Deep Convolutional Networks for Large-Scale Image Recognition\"): use only 3×3 convolutions (stride 1, padding 1) and 2×2 max-pooling, and go to 16–19 weight layers. Two stacked 3×3 convolutions see 5×5 and three see 7×7, with fewer weights (72% and 55%, see Math) and more non-linearities. Channels double (64 → 128 → 256 → 512) each time the resolution halves, keeping the cost per layer roughly constant." },
      { t: "table", head: ["Stage", "Block", "Output"], rows: [
        ["1", "2 × conv3×3-64 → max-pool", "64×112×112"],
        ["2", "2 × conv3×3-128 → max-pool", "128×56×56"],
        ["3", "3 × conv3×3-256 → max-pool", "256×28×28"],
        ["4", "3 × conv3×3-512 → max-pool", "512×14×14"],
        ["5", "3 × conv3×3-512 → max-pool", "512×7×7"],
        ["head", "FC 25,088 → 4,096 → 4,096 → 1,000 (ReLU, dropout)", "1,000"],
      ] },
      { t: "p", text: "Measured: **138,357,544 parameters, 15.47 G MACs** (VGG-16) — the most compute in this lesson. 89.4% of the parameters sit in the three FC layers; 99.2% of the MACs in the convolutions, where six of thirteen tie as most expensive at 1.85 G MACs each (the channel-doubling rule at work). **Pros:** very simple and uniform; its rich, well-studied features are still used for style transfer and *perceptual losses*; a common detector backbone (SSD, Faster R-CNN). **Cons:** very costly in compute and memory; enormous FC layers; no normalisation in the original, so very deep plain VGG-style nets are hard to train (what ResNet fixed). **Uses:** perceptual/style losses, teaching, legacy backbones." },

      // ---- GoogLeNet / Inception
      { t: "h", text: "GoogLeNet / Inception v1 (Szegedy et al., 2014) and Inception v2/v3" },
      { t: "p", text: "**Problem.** Going deeper and wider the VGG way multiplies compute, and the right filter size for a layer is unclear — objects appear at many scales. **Idea** (\"Going Deeper with Convolutions\"): the **Inception module** runs 1×1, 3×3 and 5×5 convolutions and a 3×3 max-pool **in parallel** on the same input and concatenates the outputs along channels. To keep it affordable, a **1×1 convolution** first cuts the channels before each expensive 3×3/5×5 branch (and after the pooling branch). This *bottleneck* is the key trick: in module 3a it cuts the 5×5 branch from 120.4 M to 12.4 M MACs. The 22-layer network ends with **global average pooling** instead of big FC layers, and used two *auxiliary classifiers* on middle layers during training to help gradients reach early layers." },
      { t: "steps", items: [
        { title: "Input", text: "Feature map $X$ with $C_{in}$ channels (module 3a: 192×28×28)." },
        { title: "Branch 1", text: "1×1 conv → $c_1$ channels (64)." },
        { title: "Branch 2", text: "1×1 conv to $c_{3r}$ (96) → 3×3 conv → $c_3$ (128)." },
        { title: "Branch 3", text: "1×1 conv to $c_{5r}$ (16) → 5×5 conv → $c_5$ (32)." },
        { title: "Branch 4", text: "3×3 max-pool, stride 1 → 1×1 conv → $c_p$ (32)." },
        { title: "Concatenate", text: "Stack along channels: $64+128+32+32 = 256$ channels at the same 28×28. Every branch keeps the spatial size (\"same\" padding) so they can be stacked." },
      ] },
      { t: "p", text: String.raw`Measured: torchvision GoogLeNet **6,624,904 parameters, 1.50 G MACs**; Inception-v3 at 299×299 **23,834,568 parameters, 5.71 G MACs** (auxiliary classifiers excluded). torchvision's GoogLeNet adds BatchNorm and uses a 3×3 kernel in the "5×5" branch (for weight compatibility), so it is not exactly the 2014 network. **Inception v2/v3** (2015) split 5×5 into two 3×3s and $n\times n$ into $1\times n$ then $n\times1$, added BatchNorm, and introduced *label smoothing*; Inception-v4 and Inception-ResNet (2016) added residuals. **Pros:** high accuracy per MAC for its time; multi-scale; the GAP head removes most parameters. **Cons:** hand-crafted, irregular modules with many hyperparameters, awkward to scale or change; branchy graphs run less efficiently than one wide convolution. **Uses:** Inception-v3 features underlie the *Fréchet Inception Distance* (FID) for generative models; legacy backbones.` },

      // ---- ResNet
      { t: "h", text: "ResNet (He, Zhang, Ren, Sun, 2015)" },
      { t: "p", text: String.raw`**Problem.** The **degradation problem**: making a plain network deeper (e.g. 20 → 56 layers) made even the *training* error worse — an optimisation failure, not overfitting — even though BatchNorm stops gradients from vanishing. **Idea** ("Deep Residual Learning for Image Recognition"): each block learns a residual $F(x)$ and outputs $y = \mathrm{ReLU}(F(x) + x)$. If the identity is nearly optimal, the block just pushes $F$ toward zero, which is easy, and gradients get an identity path through every block (derived in [Residual and Skip Connections](/learn/m07/residual-connections)). An ensemble of ResNets, the deepest with 152 layers, won ILSVRC-2015 classification.` },
      { t: "table", head: ["Block", "Structure", "Used in"], rows: [
        ["Basic", "3×3 conv-BN-ReLU → 3×3 conv-BN → + shortcut → ReLU", "ResNet-18, ResNet-34"],
        ["Bottleneck", "1×1 reduce (C/4) → 3×3 → 1×1 expand (C) → + shortcut → ReLU", "ResNet-50/101/152"],
        ["Shortcut", "identity if shapes match; otherwise 1×1 conv with stride 2 + BN (\"projection\")", "first block of each stage"],
      ] },
      { t: "p", text: "Layout: 7×7/2 stem conv (64 channels) + 3×3/2 max-pool → four stages at 56, 28, 14, 7 pixels with base widths 64, 128, 256, 512 → GAP → FC. Blocks per stage: [2,2,2,2] for ResNet-18, [3,4,6,3] for ResNet-34 and -50, [3,8,36,3] for ResNet-152. Measured: **ResNet-18 11.69 M params / 1.81 G MACs; ResNet-50 25.56 M / 4.09 G; ResNet-152 60.19 M / 11.51 G.** (The paper reports about 3.8 billion multiply-adds for ResNet-50; torchvision's \"v1.5\" puts the stride-2 in the 3×3 instead of the first 1×1 of a downsampling block — costlier but better.) A bottleneck block at 256 channels has 69,632 conv weights versus 1,179,648 for two 3×3 convs — 17× fewer, which makes 50+ layers affordable. Successors: **pre-activation ResNet** (BN-ReLU before the conv), **ResNeXt** (grouped bottleneck convs — \"cardinality\"), **SE-ResNet** (squeeze-and-excitation channel attention), **Wide ResNet**. **Pros:** trains reliably at great depth; simple, regular, hardware-friendly; still a strong default baseline. **Cons:** deep 3×3 stacks at 56×56 use much activation memory; lower accuracy per MAC than later efficient designs; the original recipe is dated (modern recipes help a lot). **Uses:** backbones in Faster/Mask R-CNN, FPN, DeepLab, medical imaging; image encoder in many multimodal systems; the default research baseline." },

      // ---- DenseNet
      { t: "h", text: "DenseNet (Huang, Liu, van der Maaten, Weinberger, 2016)" },
      { t: "p", text: "**Problem.** In a ResNet features are *added*, so early information gets mixed into a running sum and later layers may recompute similar features. **Idea** (\"Densely Connected Convolutional Networks\"): inside a **dense block** each layer receives the **concatenation** of *all* earlier layers' maps in the block and adds only $k$ new channels, the **growth rate** (k = 32 for DenseNet-121). Each layer is BN-ReLU-1×1 conv (to $4k$ channels, a bottleneck) → BN-ReLU-3×3 conv ($k$ channels). Between blocks a **transition layer** (BN, a 1×1 conv halving the channels — compression θ = 0.5 — and 2×2 average pooling) shrinks the size." },
      { t: "table", head: ["DenseNet-121 stage", "Layers", "Channels in → out of block", "After transition"], rows: [
        ["Stem", "7×7/2 conv + 3×3/2 max-pool", "3 → 64", "—"],
        ["Dense block 1 (56×56)", "6", "64 → 64 + 6·32 = 256", "128, 28×28"],
        ["Dense block 2 (28×28)", "12", "128 → 128 + 12·32 = 512", "256, 14×14"],
        ["Dense block 3 (14×14)", "24", "256 → 256 + 24·32 = 1,024", "512, 7×7"],
        ["Dense block 4 (7×7)", "16", "512 → 512 + 16·32 = 1,024", "GAP → FC 1,000"],
      ] },
      { t: "p", text: "Measured: **DenseNet-121 7,978,856 parameters, 2.83 G MACs** — under a third of ResNet-50's parameters. **Pros:** strong feature reuse, very parameter-efficient, built-in deep supervision (every layer is near the loss), good on small datasets. **Cons:** naive concatenation makes activation memory grow quadratically with block depth and causes many memory reads, so DenseNets are often slower and hungrier than MACs suggest; harder to scale. **Uses:** medical imaging (CheXNet's chest X-ray model used DenseNet-121), small-data classification, feature extraction." },

      // ---- Xception
      { t: "h", text: "Xception (Chollet, 2016)" },
      { t: "p", text: "**Problem.** An Inception module partly separates *spatial* patterns (3×3/5×5 convs) from *cross-channel* patterns (1×1 convs). Why not separate them completely? **Idea** (\"Xception: Deep Learning with Depthwise Separable Convolutions\" — \"extreme Inception\"): replace every convolution with a **depthwise separable convolution** — a per-channel 3×3 *depthwise* conv followed by a 1×1 *pointwise* conv that mixes channels — inside residual blocks. Layout: an entry flow (two ordinary convs + three downsampling residual blocks of separable convs), a middle flow of 8 identical residual blocks at 728 channels and 19×19, and an exit flow ending at 2,048 channels, GAP, FC." },
      { t: "steps", items: [
        { title: "Middle-flow block (×8)", text: "ReLU → SepConv 3×3, 728 → BN → ReLU → SepConv 3×3, 728 → BN → ReLU → SepConv 3×3, 728 → BN, then add the input (identity shortcut)." },
        { title: "Downsampling block", text: "Separable convs → 3×3 max-pool stride 2; the shortcut is a strided 1×1 conv + BN so shapes match." },
        { title: "No non-linearity inside the separable conv", text: "Unlike MobileNet-v1, Xception puts no ReLU between the depthwise and pointwise steps; the paper found this worked better." },
      ] },
      { t: "p", text: "Measured (our from-scratch version at 299×299): **22,855,952 parameters, 8.36 G MACs** — about Inception-v3's parameter count, consistent with the paper's ~22.9 M. **Pros:** uses the same parameter budget better than Inception; clean, regular design. **Cons:** heavy compute at 299×299; depthwise convs do few MACs per byte moved, so real GPU speed-ups are smaller than the MAC reduction. **Uses:** a modified, aligned Xception is the DeepLabv3+ segmentation backbone; available in Keras applications." },

      // ---- MobileNet
      { t: "h", text: "MobileNet v1, v2 and v3 (Howard et al. 2017; Sandler et al. 2018; Howard et al. 2019)" },
      { t: "p", text: "**Problem.** Run CNNs on phones and embedded devices with tight compute, memory and energy budgets. **MobileNet v1** (\"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications\") is a plain stack: one standard 3×3/2 conv, then 13 depthwise separable layers (3×3 depthwise-BN-ReLU, 1×1 pointwise-BN-ReLU), GAP, FC. Two global knobs trade accuracy for cost: a **width multiplier** α scaling all channel counts and a **resolution multiplier** ρ scaling the input size. Measured (our implementation): **4,231,976 parameters, 568.7 M MACs** — matching the paper's 4.2 M and 569 M multiply-adds, a useful check of our counter." },
      { t: "p", text: String.raw`**MobileNet v2** ("MobileNetV2: Inverted Residuals and Linear Bottlenecks") introduced the **inverted residual** block. A ResNet bottleneck goes wide → narrow → wide with the shortcut between *wide* tensors. MobileNetV2 goes **narrow → wide → narrow** with the shortcut between *narrow* tensors: a 1×1 conv *expands* channels by a factor $t$ (6 by default) → a 3×3 *depthwise* conv works cheaply in the wide space → a 1×1 conv *projects* back to a narrow **linear bottleneck**, on purpose with **no ReLU**, because ReLU in a low-dimensional space destroys information. The shortcut is used when stride is 1 and channel counts match. Activations are ReLU6 ($\min(\max(0,x),6)$, robust to low-precision arithmetic). Measured: **3,504,872 parameters, 300.8 M MACs** — fewer MACs than v1, with higher reported accuracy.` },
      { t: "table", head: ["Inverted residual step (c = 24, t = 6, 56×56)", "Output", "Weights"], rows: [
        ["1×1 expand + BN + ReLU6", "144×56×56", "24·144 = 3,456"],
        ["3×3 depthwise + BN + ReLU6", "144×56×56", "144·9 = 1,296"],
        ["1×1 project + BN (linear, no activation)", "24×56×56", "144·24 = 3,456"],
        ["+ input (stride 1, same channels)", "24×56×56", "0"],
      ] },
      { t: "p", text: String.raw`**MobileNet v3** ("Searching for MobileNetV3") came from **hardware-aware neural architecture search** (NAS, plus NetAdapt tuning) optimising measured phone latency. It keeps inverted residuals, adds **squeeze-and-excitation** (SE) channel attention in some blocks, uses the cheap **hard-swish** $x\cdot\mathrm{ReLU6}(x+3)/6$, and redesigns the costly first and last layers. Measured: **v3-Large 5,483,032 params / 216.6 M MACs; v3-Small 2,542,856 / 56.5 M MACs.** **Pros (family):** very low MACs and size; mature mobile-runtime support (TFLite, Core ML, ONNX Runtime) and quantisation-friendly variants. **Cons:** lower accuracy than big models; depthwise convs are memory-bound, so on GPUs a MobileNet is often barely faster than ResNet-18 despite 6× fewer MACs; the searched v3 is harder to modify. **Uses:** on-device classification, light detectors (SSDLite) and segmenters, real-time video on phones and boards.` },

      // ---- EfficientNet
      { t: "h", text: "EfficientNet (Tan and Le, 2019)" },
      { t: "p", text: String.raw`**Problem.** With twice (or 2^φ times) the compute, how should a network grow? People scaled depth (ResNet-18 → 152), width (Wide ResNet) *or* resolution, one at a time, with quickly shrinking returns. **Idea** ("EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"): **compound scaling** — scale depth by $\alpha^\phi$, width by $\beta^\phi$ and resolution by $\gamma^\phi$ together, with $\alpha\beta^2\gamma^2 \approx 2$ so each unit of $\phi$ roughly doubles FLOPs. The constants (α = 1.2, β = 1.1, γ = 1.15) came from a small grid search on the base model. The base **EfficientNet-B0** came from a NAS (similar to MnasNet) and is built from **MBConv** blocks — MobileNetV2 inverted residuals with squeeze-and-excitation and the **swish/SiLU** activation $x\,\sigma(x)$ — plus stochastic depth in training.` },
      { t: "table", head: ["Model", "Input (paper)", "Measured params", "Measured MACs"], rows: [
        ["B0", "224", "5.29 M", "0.39 G"],
        ["B1", "240", "7.79 M", "0.69 G"],
        ["B2", "260", "9.11 M", "0.99 G"],
        ["B3", "300", "12.23 M", "1.83 G"],
        ["B4", "380", "19.34 M", "4.39 G"],
      ] },
      { t: "p", text: "Measured B0: **5,288,548 parameters, 385.8 M MACs.** Each step roughly doubles compute (the published B1–B7 round the coefficients, so the ratio isn't exactly 1.92). **Pros:** excellent accuracy per parameter and per MAC; a principled scaling rule that works for other base networks too. **Cons:** depthwise convs, SE and swish make it slower on GPUs/TPUs than MACs suggest and memory-hungry when training at high resolution (EfficientNetV2, 2021, swapped early MBConvs for fused regular convs to train faster); sensitive to the training recipe. **Uses:** transfer-learning backbone, EfficientDet detectors, Kaggle-style competitions, mobile (EfficientNet-Lite)." },

      // ---- ConvNeXt
      { t: "h", text: "ConvNeXt (Liu, Mao, Wu, Feichtenhofer, Darrell, Xie, 2022)" },
      { t: "p", text: "**Problem.** By 2021 hierarchical Vision Transformers (e.g. Swin) beat ResNets — but was it attention, or the surrounding design and training recipe? **Idea** (\"A ConvNet for the 2020s\"): modernise ResNet-50 one change at a time, staying a pure ConvNet: (1) a modern **training recipe** (AdamW, 300 epochs, Mixup/CutMix, RandAugment, label smoothing, stochastic depth); (2) **stage ratio** 3:3:9:3 like Swin-T; (3) a **patchify stem** — a 4×4 stride-4 conv; (4) **depthwise convolutions** with more width (ResNeXt-style), 96 base channels; (5) an **inverted bottleneck** (4× expansion); (6) **large 7×7 depthwise kernels**, moved to the start of the block; (7) **GELU**, fewer activations and norms, **LayerNorm** instead of BatchNorm; (8) separate 2×2 stride-2 downsampling layers. Each step was measured; the result matched Swin Transformers of similar size." },
      { t: "steps", items: [
        { title: "Depthwise 7×7 conv", text: String.raw`$C \to C$ channels, spatial mixing only (like attention's token mixing, but local).` },
        { title: "LayerNorm", text: "Over each pixel's channels (channels-last)." },
        { title: "1×1 conv (Linear) $C \\to 4C$, GELU", text: "Channel mixing in an expanded space — the Transformer's MLP." },
        { title: "1×1 conv (Linear) $4C \\to C$", text: "Project back down." },
        { title: "Layer scale + residual", text: String.raw`Multiply by a learnable per-channel $\gamma$ (initialised to $10^{-6}$), apply stochastic depth in training, and add the input.` },
      ] },
      { t: "p", text: "ConvNeXt-T: stage depths (3, 3, 9, 3), widths (96, 192, 384, 768). Measured: **28,589,128 parameters, 4.46 G MACs** — about ResNet-50's budget (25.6 M, 4.1 G). **Pros:** Transformer-level accuracy with a simple conv design; no attention cost growing quadratically with image size; works at any resolution and for dense prediction; scales to large models. **Cons:** LayerNorm and 7×7 depthwise convs are less optimised than BN + 3×3 on some hardware; gains depend on the long modern recipe; heavier than mobile models. **Uses:** general-purpose backbone for classification, detection and segmentation; ConvNeXt V2 (2023) adds masked-autoencoder pre-training." },

      // ---- timeline
      { t: "h", text: "Timeline: how the design ideas piled up" },
      { t: "list", ordered: true, items: [
        "**Learned local features (1989–1998).** Convolution + subsampling + backprop (LeNet). Weight sharing makes the parameter count independent of image size.",
        "**Scale and better non-linearities (2012).** ReLU, GPUs, dropout and augmentation (AlexNet) made it possible to train an 8-layer CNN on a million images. Big early kernels and huge FC heads were normal.",
        "**Depth through small kernels (2014).** VGG replaced large kernels with stacks of 3×3s — same receptive field, fewer parameters, more non-linearities — and doubled channels when halving resolution.",
        "**Multi-branch modules and 1×1 bottlenecks (2014–2015).** Inception: parallel filter sizes, 1×1 convs to cut channels before expensive convolutions, global average pooling instead of giant FC layers. BatchNorm (2015) made all these networks far easier to train.",
        "**Residual learning (2015).** ResNet's identity shortcuts solve degradation and take depth to 100+ layers; bottleneck blocks keep it affordable. From then on nearly every architecture has shortcuts.",
        "**Dense connectivity (2016).** DenseNet: concatenate instead of add, small growth rate, maximum feature reuse.",
        "**Depthwise separability (2016–2018).** Xception and MobileNet split convolutions into spatial (depthwise) and channel (pointwise) parts, ~8–9× fewer MACs; MobileNetV2's inverted residual with linear bottleneck becomes the standard efficient block. Squeeze-and-excitation (2017) adds cheap channel attention.",
        "**Search and compound scaling (2018–2019).** NAS (MnasNet, MobileNetV3, EfficientNet-B0) designs blocks for measured latency; EfficientNet scales depth, width and resolution together.",
        "**Modernised ConvNets (2020–2022).** After Vision Transformers, ConvNeXt shows that a ConvNet with Transformer-era design (patchify stem, stage ratios, large depthwise kernels, inverted bottlenecks, LayerNorm, GELU, fewer norms/activations) and training recipe is just as accurate. RegNet (2020) and EfficientNetV2 (2021) chase the same goal: simple, fast, well-scaled designs.",
      ] },
      { t: "callout", kind: "note", title: "Accuracy numbers", text: "This lesson deliberately skips ImageNet accuracy tables: they depend heavily on the training recipe, resolution and evaluation protocol, and torchvision's re-trained weights differ from the papers. Use each paper's tables or the model card of the checkpoint you actually load." },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "Cost of a convolution layer" },
      { t: "p", text: String.raw`For a convolution with $C_{in}$ input channels, $C_{out}$ output channels, a $k\times k$ kernel, $g$ groups and an $H_{out}\times W_{out}$ output, each output value is a dot product of length $(C_{in}/g)\,k^2$:` },
      { t: "math", tex: String.raw`P = C_{out}\,\frac{C_{in}}{g}\,k^2 + C_{out}, \qquad \text{MACs} = C_{out}\,\frac{C_{in}}{g}\,k^2\,H_{out}\,W_{out}`, caption: "Parameters P (weights + biases) and multiply-accumulates. Ordinary conv: g = 1. Depthwise conv: g = C_in = C_out." },
      { t: "p", text: String.raw`A linear layer with $n_{in}$ inputs and $n_{out}$ outputs has $n_{in}n_{out} + n_{out}$ parameters and $n_{in}n_{out}$ MACs. BatchNorm and LayerNorm have $2C$ learnable parameters (scale $\gamma$, shift $\beta$); BN's running mean and variance are buffers, not parameters (Keras counts them as non-trainable parameters, so its totals are larger). One MAC is often counted as 2 FLOPs, and papers quoting "FLOPs" sometimes mean MACs — always check.` },
      { t: "h", text: "VGG: stacked 3×3 versus one large kernel" },
      { t: "p", text: String.raw`With $C$ channels in and out and stride 1, $n$ stacked $3\times3$ convolutions have receptive field $1 + 2n$ ($r_\ell = r_{\ell-1} + (k-1)$) and $9nC^2$ weights. One $(2n+1)\times(2n+1)$ convolution has the same receptive field and $(2n+1)^2C^2$ weights:` },
      { t: "math", tex: String.raw`\frac{9nC^2}{(2n+1)^2C^2}:\qquad n=2:\ \frac{18}{25} = 0.72,\qquad n=3:\ \frac{27}{49} \approx 0.55` },
      { t: "p", text: String.raw`For $C = 256$: two 3×3 layers have 1,179,648 weights versus 1,638,400 for one 5×5; three 3×3s have 1,769,472 versus 3,211,264 for one 7×7. Doubling channels while halving resolution keeps a layer's cost the same: $C_{out}C_{in}k^2HW$ is unchanged under $C \to 2C$, $H,W \to H/2, W/2$. That is why VGG-16's conv1_2, conv2_2, conv3_2, conv3_3, conv4_2 and conv4_3 all cost exactly $224^2\cdot64^2\cdot9 = 1.85\times10^9$ MACs.` },
      { t: "h", text: "Worked example: the Inception 3a module" },
      { t: "p", text: String.raw`Input $X \in \mathbb{R}^{192\times28\times28}$, so $HW = 784$ positions. Branch channel counts $(c_1, c_{3r}, c_3, c_{5r}, c_5, c_p) = (64, 96, 128, 16, 32, 32)$. Weights (biases ignored):` },
      { t: "steps", items: [
        { title: "Branch 1 (1×1)", tex: String.raw`192\cdot64 = 12{,}288` },
        { title: "Branch 2 (1×1 → 3×3)", tex: String.raw`192\cdot96 + 96\cdot9\cdot128 = 18{,}432 + 110{,}592 = 129{,}024` },
        { title: "Branch 3 (1×1 → 5×5)", tex: String.raw`192\cdot16 + 16\cdot25\cdot32 = 3{,}072 + 12{,}800 = 15{,}872` },
        { title: "Branch 4 (pool → 1×1)", tex: String.raw`192\cdot32 = 6{,}144` },
        { title: "Total and compute", tex: String.raw`P = 163{,}328,\qquad \text{MACs} = 163{,}328 \times 784 = 128.05\times10^6` },
        { title: "Without the 1×1 reductions", text: "1×1, 3×3 and 5×5 directly on 192 channels (the pool branch passes all 192 channels through, so the output grows to 64+128+32+192 = 416 channels):", tex: String.raw`192(64 + 9\cdot128 + 25\cdot32) = 387{,}072 \text{ weights},\quad 303.46\times10^6 \text{ MACs}` },
        { title: "The 5×5 branch alone", tex: String.raw`\underbrace{784\cdot192\cdot25\cdot32}_{\text{direct}} = 120.4\text{M} \quad\text{vs}\quad \underbrace{784(192\cdot16 + 16\cdot25\cdot32)}_{\text{with }1\times1\text{ to 16 ch}} = 12.4\text{M}\ \ (9.7\times)` },
      ] },
      { t: "p", text: "With BatchNorm (2 parameters per channel of every conv) the module in our PyTorch code has 164,064 parameters and 128.0 M MACs, matching these hand counts." },
      { t: "h", text: "ResNet: basic versus bottleneck" },
      { t: "math", tex: String.raw`y = \mathrm{ReLU}\big(F(x;\{W_i\}) + x\big),\qquad y = \mathrm{ReLU}\big(F(x;\{W_i\}) + W_s x\big)\ \text{(projection when shapes differ)}` },
      { t: "p", text: String.raw`Here $F$ is the residual branch and $W_s$ a strided 1×1 convolution. At width $C = 256$ the bottleneck branch (1×1 to 64, 3×3 at 64, 1×1 to 256) has $256\cdot64 + 9\cdot64^2 + 64\cdot256 = 69{,}632$ weights, versus $2\cdot9\cdot256^2 = 1{,}179{,}648$ for two full 3×3 convolutions (17× more). At 56×56 that is 218.4 M versus 3,699.4 M MACs. A residual block's Jacobian is $I + \partial F/\partial x$, so the gradient always has a direct path (see [Residual Connections](/learn/m07/residual-connections)).` },
      { t: "h", text: "DenseNet: channel growth" },
      { t: "p", text: String.raw`Layer $\ell$ of a dense block receives $x_\ell^{in} = [x_0, x_1, \dots, x_{\ell-1}]$ (all earlier outputs, concatenated) and outputs $x_\ell = H_\ell(x_\ell^{in})$ with $k$ channels. With $c_0$ channels entering the block:` },
      { t: "math", tex: String.raw`C(x_\ell^{in}) = c_0 + (\ell-1)\,k,\qquad C_{\text{out of block}} = c_0 + L\,k` },
      { t: "p", text: String.raw`DenseNet-121 block 1: $64 + 6\cdot32 = 256$; transition halves to 128; block 2: $128 + 12\cdot32 = 512 \to 256$; block 3: $256 + 24\cdot32 = 1024 \to 512$; block 4: $512 + 16\cdot32 = 1024$. Each layer's 1×1 bottleneck sees all earlier channels, so layer $\ell$'s cost grows linearly with $\ell$ and a block's total cost grows quadratically with $L$.` },
      { t: "h", text: "Depthwise separable convolution" },
      { t: "p", text: String.raw`A depthwise separable convolution applies one $k\times k$ filter $D_c$ per input channel (depthwise, $g = C_{in}$), then a $1\times1$ convolution $P \in \mathbb{R}^{C_{out}\times C_{in}}$. Compared with a standard convolution of the same input, output and kernel size, it costs` },
      { t: "math", tex: String.raw`\frac{C_{in}k^2HW + C_{in}C_{out}HW}{C_{in}C_{out}k^2HW} = \frac{1}{C_{out}} + \frac{1}{k^2}` },
      { t: "p", text: String.raw`For $k = 3$, $C_{in} = C_{out} = 128$ at 56×56: standard 147,456 weights and 462.4 M MACs; separable $128\cdot9 + 128^2 = 17{,}536$ weights and 55.0 M MACs, ratio $1/128 + 1/9 = 0.1189$ — about 8.4× cheaper. It equals a standard convolution whose kernel is forced into the product form` },
      { t: "math", tex: String.raw`W[o, c, :, :] = P[o, c]\; D_c`, caption: "Each input-channel slice of every output filter is the same spatial pattern D_c, only rescaled. The NumPy code checks this numerically." },
      { t: "p", text: String.raw`So a separable conv is a *restricted* standard conv: it cannot learn arbitrary joint spatial-channel patterns, which is why such networks make up for it with more channels or layers. A MobileNetV2 inverted residual with $c$ input/output channels and expansion $t$ costs, per output pixel (stride 1), $c\cdot tc + 9\,tc + tc\cdot c = tc(2c + 9)$ MACs. For $c = 24, t = 6$ that is $144\cdot57 = 8{,}208$, i.e. 25.7 M MACs at 56×56 — versus 16.3 M for one standard 3×3 conv at 24 channels, but the block works in a 144-channel space.` },
      { t: "h", text: "EfficientNet compound scaling" },
      { t: "p", text: String.raw`Let $d, w, r$ multiply the base network's depth (number of layers), width (channels) and input resolution. A convolution's MACs grow linearly with depth, quadratically with width ($C_{in}C_{out}$) and quadratically with resolution ($HW$), so total FLOPs scale as $d\,w^2r^2$. EfficientNet sets` },
      { t: "math", tex: String.raw`d = \alpha^\phi,\quad w = \beta^\phi,\quad r = \gamma^\phi,\qquad \text{s.t. } \alpha\,\beta^2\gamma^2 \approx 2,\ \ \alpha,\beta,\gamma \ge 1` },
      { t: "p", text: String.raw`With the paper's $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$: $\alpha\beta^2\gamma^2 = 1.2 \times 1.21 \times 1.3225 = 1.92027$. For $\phi = 2$: depth ×1.44, width ×1.21, resolution ×1.3225 (224 → 296), and FLOPs ×$1.92027^2 = 3.687$. For $\phi = 3$: FLOPs ×7.08. Our measurements (B0 → B4: 0.39 → 4.39 G MACs) follow this trend; the released models round depth, width and resolution, so the ratios are not exact.` },
      { t: "h", text: "ConvNeXt block cost" },
      { t: "p", text: String.raw`Per pixel, with $C$ channels: the depthwise 7×7 costs $49C$ MACs and the two pointwise layers $C\cdot4C + 4C\cdot C = 8C^2$. For $C = 96$: $4{,}704 + 73{,}728 = 78{,}432$ MACs per pixel, i.e. 245.96 M at 56×56; parameters $49C + C + 2C + 4C^2 + 4C + 4C^2 + C + C = 79{,}296$ (dw weights+bias, LayerNorm, two linear layers with biases, layer scale). The 7×7 spatial mixing is only 6% of the cost — the channel MLP dominates, as in a Transformer block.` },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "p", text: "Two simulators make the ideas behind these architectures concrete. The first shows the conv → ReLU → pool pipeline that LeNet, AlexNet and VGG repeat; the second shows why depth without shortcuts was a problem and how residual connections fix it." },
      { t: "sim", id: "cnn-pipeline", caption: "A two-stage CNN (conv → ReLU → max-pool, twice) on a real image, with the tensor shape at every stage." },
      { t: "list", items: [
        "Step through all stages and note the shape at each. This is one VGG stage (with learned instead of hand-designed kernels): spatial size shrinks, channels carry the features.",
        "Set layer-1 stride to 2 and compare shapes. AlexNet's 11×11 stride-4 first layer and ResNet's 7×7 stride-2 stem use stride the same way — to cut resolution, and so compute, early.",
        "Count each stage's multiply-adds with $C_{out}C_{in}k^2H_{out}W_{out}$ from the Math section, and see which stage dominates.",
      ] },
      { t: "sim", id: "gradient-flow", caption: "Signal scale (top) and gradient size per layer (bottom) in a real forward/backward pass through a deep network." },
      { t: "list", items: [
        "Choose the preset **Sigmoid, Xavier (vanishing)** with depth 30: the gradient at the first layers is many orders of magnitude smaller than at the last. This held back pre-2012 networks and motivated ReLU (AlexNet).",
        "Now tick **Residual connections** (or choose **Sigmoid + residual**): the gradient at layer 1 becomes comparable to the last layer, because the identity path adds a term that doesn't shrink. This is the ResNet idea in its simplest form.",
        "Try **ReLU, He (healthy)** with depth 50, with and without residuals. With good initialisation gradients survive in both — yet deep plain CNNs still train worse in practice (see the training comparison in the Code section). Gradient *size* is not the whole story; residual blocks also make the loss landscape easier to optimise.",
      ] },
      { t: "callout", kind: "tip", title: "Architecture Explorer", text: "Open the [Architecture Explorer](/architectures) and select VGG-16: check that the FC layers hold most of the parameters (the first FC layer alone has 25,088 × 4,096 + 4,096 = 102,764,544). Then select AlexNet and find the layer with the most parameters. Build your own variant by swapping a 5×5 conv for two 3×3 convs and compare totals." },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      { t: "p", text: "First, a **NumPy** cost calculator that derives this lesson's numbers from configurations alone: VGG-16's parameters and MACs layer by layer (matching torchvision exactly), the 3×3-stacking and Inception-bottleneck savings, the depthwise-separable cost ratio, and a numerical proof that a depthwise separable convolution equals a standard convolution with a factored kernel." },
      { t: "code", lang: "python", title: "cnn_costs_numpy.py", code: COSTS_NUMPY, output: COSTS_NUMPY_OUT },
      { t: "p", text: "Next, every block in this lesson built from `nn.Conv2d`, `nn.BatchNorm2d` and `nn.Linear` (no torchvision blocks). For each, the script runs a forward pass, prints shapes, counts parameters and MACs, and checks the parameter count against the formula beside it. NumPy versions would add hundreds of lines of convolution and BN code without teaching anything new about the designs." },
      { t: "code", lang: "python", title: "cnn_blocks_from_scratch.py", code: BLOCKS_TORCH, output: BLOCKS_TORCH_OUT },
      { t: "list", items: [
        "The VGG block costs the most MACs (693.6 M) because both 3×3 convs run at 56×56 with up to 128 channels; the inverted residual costs 25.7 M.",
        "The dense block outputs 64 + 6·32 = 256 channels, and its cost (1,040 M MACs) is dominated by the later layers, whose 1×1 convs see up to 224 input channels.",
        "The strided basic block (64 → 128, /2) has *more* parameters but *fewer* MACs than the stride-1 one — compute follows output resolution.",
        "ConvNeXt's layer scale (γ = 10⁻⁶) makes each block start as an almost exact identity map (difference 8 × 10⁻⁷), which stabilises training of deep stacks.",
      ] },
      { t: "h", text: "A real comparison: plain vs residual on handwritten digits" },
      { t: "p", text: "Two networks that differ **only** in the shortcut are trained the same way (Adam, learning rate $10^{-3}$, batch 64, 8 epochs, same seed and data order) on scikit-learn's 8×8 digits at two depths: 6 and 22 weighted layers. Both use BatchNorm, so neither suffers from vanishing gradients. The script also prints the gradient norm reaching the first convolution at initialisation. It takes about 20 seconds on one CPU thread." },
      { t: "code", lang: "python", title: "plain_vs_resnet_digits.py", code: PLAIN_VS_RES, output: PLAIN_VS_RES_OUT },
      { t: "list", items: [
        "**Degradation in miniature.** The 22-layer plain network trains *more slowly* than the 6-layer plain one (training loss 0.30 vs 0.27 after 8 epochs), though it has 6× more parameters and could represent the shallow network.",
        "**Residuals fix optimisation.** ResNet-22 reaches training loss 0.11 after 3 epochs and 0.03 after 5, while plain-22 is still at 1.26 and 0.75. (Its final 0.09 is a small loss spike typical of Adam on a tiny dataset.)",
        "**Not a vanishing-gradient effect.** At initialisation plain-22 sends a *larger* gradient to its first layer (0.705) than ResNet-22 (0.483). With BatchNorm the gradient size is fine; residuals change the *shape* of the optimisation problem — each block starts near the identity, so the network acts shallow early in training, as the ResNet paper argued.",
        "**Test accuracy is noisier.** With 450 test images, plain-22 (0.947) and ResNet-22 (0.951) both trail the shallow nets (0.980): for this tiny 8×8 problem 22 layers are more than needed and 8 epochs are short. The question that matters at ImageNet scale — can the deep network be optimised at all — is what the training losses show.",
      ] },
    ],

    // -------------------------------------------------------------- framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Every model in the table is built with `weights=None` (random weights, nothing downloaded) and measured on PyTorch's **meta device**, which makes shape-only tensors with no storage — so even VGG-16 builds instantly and the hooks still see every shape. MobileNet-v1, Xception and LeNet-5 are not in torchvision, so they are hand-written." },
          { t: "code", lang: "python", title: "measure_cnn_zoo.py", code: ZOO_TORCH, output: ZOO_TORCH_OUT },
          { t: "list", items: [
            "The MAC counter counts only `Conv2d` and `Linear`; a conv costs output elements × $(C_{in}/g)\\,k^2$, so depthwise convolutions (groups = channels) are counted correctly.",
            "Two sanity checks against published numbers: our MobileNet-v1 gives 4.23 M parameters and 568.7 M MACs (paper: 4.2 M and 569 M multiply-adds), and our Xception gives 22,855,952 parameters (about 22.9 M in the paper).",
            "For pretrained weights in transfer learning you would pass e.g. `weights=\"IMAGENET1K_V1\"` (or the `ResNet50_Weights.DEFAULT` enum), which downloads a checkpoint; this lesson never does. For a new task, replace the head (`model.fc`, `model.classifier[-1]` or `model.head`, depending on the family) — see [Transfer Learning](/learn/m12/transfer-learning).",
            "torchvision's GoogLeNet and ResNet-50 differ slightly from the papers (BatchNorm and a 3×3 kernel in the \"5×5\" branch; stride placement in ResNet v1.5), so measured MACs can differ from the papers' figures.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras ships most of these families in `tf.keras.applications`; with `weights=None` nothing is downloaded. Keras's `count_params()` includes BatchNorm moving means and variances (non-trainable), so its totals are higher than PyTorch's for BN-based models; compare the trainable count. Keras models take channels-last (N×H×W×C) inputs, and some include their own preprocessing layers." },
          { t: "code", lang: "python", title: "cnn_zoo_keras.py (not executed in CI)", code: ZOO_KERAS, output: "Prints one line per model with its default input shape (e.g. (224, 224, 3), or (299, 299, 3) for InceptionV3 and Xception), its total parameter count and its trainable parameter count, then the summary of a frozen EfficientNetB0 backbone with a new 5-class head (only the Dense layer's 1280·5 + 5 = 6,405 parameters are trainable)." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      { t: "p", text: "The big comparison. Parameters and MACs are measured (one image, convolution + linear multiply-adds, auxiliary classifiers excluded); the year is the first public version (arXiv or publication)." },
      { t: "table", head: ["Model", "Year", "Input", "Params (measured)", "MACs (measured)", "Key idea"], rows: [
        ["LeNet-5", "1998", "32×32", "61.7 K", "0.4 M", "Conv + subsampling learned end-to-end"],
        ["AlexNet", "2012", "224", "61.1 M", "0.71 G", "ReLU, GPU training, dropout, augmentation"],
        ["VGG-16", "2014", "224", "138.4 M", "15.47 G", "Deep stacks of 3×3 convs"],
        ["GoogLeNet", "2014", "224", "6.6 M", "1.50 G", "Inception module, 1×1 bottlenecks, GAP head"],
        ["Inception-v3", "2015", "299", "23.8 M", "5.71 G", "Factorised convolutions, BN, label smoothing"],
        ["ResNet-18", "2015", "224", "11.7 M", "1.81 G", "Residual basic blocks"],
        ["ResNet-50", "2015", "224", "25.6 M", "4.09 G", "Residual bottleneck blocks"],
        ["ResNet-152", "2015", "224", "60.2 M", "11.51 G", "Very deep residual network"],
        ["DenseNet-121", "2016", "224", "8.0 M", "2.83 G", "Dense (concatenative) connectivity"],
        ["Xception", "2016", "299", "22.9 M", "8.36 G", "Depthwise separable convs + residuals"],
        ["MobileNet-v1", "2017", "224", "4.2 M", "0.57 G", "Plain depthwise separable stack, width multiplier"],
        ["MobileNet-v2", "2018", "224", "3.5 M", "0.30 G", "Inverted residual, linear bottleneck"],
        ["MobileNet-v3-L", "2019", "224", "5.5 M", "0.22 G", "Hardware-aware NAS, SE, hard-swish"],
        ["MobileNet-v3-S", "2019", "224", "2.5 M", "0.06 G", "Small variant for low-end devices"],
        ["EfficientNet-B0", "2019", "224", "5.3 M", "0.39 G", "NAS base (MBConv + SE) + compound scaling"],
        ["ConvNeXt-T", "2022", "224", "28.6 M", "4.46 G", "ResNet modernised with Transformer-era design"],
      ] },
      { t: "p", text: "How the block designs compare:" },
      { t: "table", head: ["Block", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Plain conv stack (VGG)", "Simple, uniform, rich features", "Expensive; hard to train very deep without BN/shortcuts", "Perceptual losses, teaching"],
        ["Inception module", "Multi-scale, cheap via 1×1 reductions", "Many hand-tuned hyperparameters; branchy", "Legacy backbones, FID features"],
        ["Residual (basic / bottleneck)", "Trains at any depth; regular; fast on GPUs", "Activation memory; dated without modern recipe", "Default backbone, baselines"],
        ["Dense block", "Feature reuse, few parameters", "Memory traffic from concatenation; slower than MACs suggest", "Medical imaging, small data"],
        ["Depthwise separable (Xception, MobileNet-v1)", "~8–9× fewer MACs", "Memory-bound depthwise step; less expressive per layer", "Segmentation backbones, mobile"],
        ["Inverted residual / MBConv", "Very efficient; good with quantisation", "Many small ops; GPU-inefficient", "Mobile, EfficientNet family"],
        ["ConvNeXt block", "Transformer-level accuracy, pure conv", "LayerNorm and 7×7 dw less optimised on some runtimes", "Modern general-purpose backbone"],
      ] },
      { t: "p", text: "Related families worth knowing by name: **ResNeXt** (grouped bottlenecks), **SENet** (squeeze-and-excitation), **Wide ResNet**, **ShuffleNet** (channel shuffle for grouped convs), **RegNet** (design spaces of simple regular networks), **EfficientNetV2** (fused MBConv, progressive resizing), and the hybrid or attention-based backbones in [BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families)." },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      { t: "h", text: "Where these architectures are used" },
      { t: "list", items: [
        "**Backbones for detection and segmentation**: ResNet-50/101 with feature pyramids in Faster/Mask R-CNN; VGG-16 in the original SSD; modified Xception in DeepLabv3+; EfficientNet in EfficientDet; ConvNeXt in modern detectors and segmenters. See [Object Detection](/learn/m12/object-detection) and [Segmentation](/learn/m12/segmentation).",
        "**On-device vision**: MobileNet-v2/v3 and EfficientNet-Lite run classification, detection (SSDLite) and segmentation on phones and embedded boards via TFLite, Core ML or ONNX Runtime.",
        "**Medical imaging**: ResNet and DenseNet backbones fine-tuned for X-ray, CT, dermatology and pathology, often starting from ImageNet weights.",
        "**Generative modelling**: VGG features for perceptual and style losses; Inception-v3 features for the FID metric; ResNet-style blocks inside diffusion-model U-Nets.",
        "**Image encoders in multimodal systems**: ResNet and ConvNeXt variants have served as image encoders in contrastive image-text models such as CLIP (which released both ResNet and ViT encoders).",
      ] },
      { t: "h", text: "How to choose an architecture today" },
      { t: "steps", items: [
        { title: "1. Start from a pretrained backbone, not from scratch", text: "Unless you have millions of labelled images, fine-tune an ImageNet-pretrained model. Having a *good, available checkpoint* matters more than small architectural differences. See [Transfer Learning](/learn/m12/transfer-learning)." },
        { title: "2. Fix your deployment constraint first", text: "Latency budget on the target device, memory (parameters + activations at your batch size and resolution), and energy. Measure on the real hardware — MACs are only a rough guide: depthwise convs, SE blocks and concatenations are slower on GPUs than their MAC count suggests, while plain 3×3 convs are very well optimised." },
        { title: "3. Pick a family by budget", text: "Microcontroller / very low-end phone: MobileNet-v3-Small or smaller. Phone, real time: MobileNet-v2/v3-Large, EfficientNet-B0/Lite. Server GPU, strong baseline: ResNet-50 (with a modern recipe) or ConvNeXt-T/S. Accuracy first, plenty of compute: larger ConvNeXt/EfficientNet or a Vision Transformer, especially with large-scale pretraining." },
        { title: "4. Consider the task's resolution and output", text: "Dense prediction (segmentation, detection) needs multi-scale features: ResNet, ConvNeXt and EfficientNet backbones have natural stages for feature pyramids. High input resolution favours convolutional backbones, whose cost grows linearly with pixels (global attention grows quadratically)." },
        { title: "5. Consider data size and domain shift", text: "Small datasets favour smaller or strongly pretrained models and heavy augmentation; very different domains (medical, satellite) benefit from fine-tuning more layers or domain-specific pretraining." },
        { title: "6. Check tooling", text: "Quantisation, pruning and export support (TensorRT, TFLite, ONNX, Core ML) for your ops; BatchNorm folds into convolutions at inference, LayerNorm does not. See [Model Optimization](/learn/m18/model-optimization)." },
        { title: "7. Benchmark two or three candidates", text: "Fine-tune each with the same recipe, then measure validation accuracy and latency on the target device. Pick the smallest model that meets the accuracy target." },
      ] },
    ],

    // -------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "table", head: ["Axis", "What drives it", "Architectures that do well", "Watch out for"], rows: [
        ["Accuracy", "Model capacity, training recipe, pretraining data", "ConvNeXt, EfficientNet-B3+, ResNet-152 with modern recipe", "Comparisons across different recipes are unfair"],
        ["Compute (MACs)", "Channels², kernel², resolution², depth", "MobileNet-v3, EfficientNet-B0", "Low MACs ≠ low latency"],
        ["Latency on GPU", "Parallelism, memory bandwidth, kernel fusion", "ResNet, VGG-style plain 3×3 convs, RegNet", "Depthwise and SE ops are memory-bound"],
        ["Latency on mobile CPU/NPU", "MACs, supported ops, quantisation", "MobileNet-v2/v3, EfficientNet-Lite", "Unsupported ops fall back to slow paths"],
        ["Parameter memory", "Weights (FC heads, wide layers)", "MobileNet, DenseNet, GAP heads", "VGG/AlexNet FC layers"],
        ["Training activation memory", "Resolution × channels × depth × batch", "Bottleneck designs, smaller inputs", "DenseNet concatenations, EfficientNet at high resolution"],
        ["Ease of modification", "Regularity of the design", "ResNet, ConvNeXt, VGG", "Inception modules, NAS-found networks"],
      ] },
      { t: "list", items: [
        "**Depth vs width vs resolution**: each raises accuracy with diminishing returns; EfficientNet's lesson is to raise them together.",
        "**Parameters vs compute**: they are separate — VGG's FC layers are parameter-heavy but cheap; its conv layers are compute-heavy but parameter-light. GAP heads (GoogLeNet onward) removed the parameter problem.",
        "**Built-in assumptions vs data**: convolution's locality and weight sharing make CNNs data-efficient; Transformers need more data or stronger pretraining but scale flexibly. ConvNeXt shows much of the gap comes from design and recipe rather than attention itself.",
        "**Efficiency tricks cost generality**: depthwise separable convs and inverted residuals cut MACs but limit what each layer can represent, and run less efficiently on hardware tuned for dense matrix multiplication.",
        "**Simplicity has value**: ResNet's regularity explains its long life — easy to scale, prune, quantise, export and reason about.",
      ] },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      { t: "list", items: [
        "**Equating MACs with speed.** A MobileNet with 6× fewer MACs than ResNet-18 can run about as fast on a GPU, because depthwise convolutions are limited by memory bandwidth. Fix: benchmark latency on the target hardware and batch size.",
        "**Confusing FLOPs and MACs.** One multiply-accumulate is often counted as 2 FLOPs, and papers and tools differ. Fix: say which you report; compare like with like.",
        "**Comparing architectures trained with different recipes.** A ResNet-50 with a modern recipe is far stronger than with the 2015 one; ConvNeXt's own ablation shows the recipe explains a large part of the gap to Transformers. Fix: compare at equal recipe, or use each paper's numbers with caution.",
        "**Feeding the wrong input size or normalisation to a pretrained model.** Inception-v3 and Xception expect 299×299; torchvision models expect ImageNet mean/std normalisation, while some Keras applications expect [-1, 1] scaling or include preprocessing. Fix: use the preprocessing that comes with the checkpoint.",
        "**Adding a ReLU after an inverted residual's projection.** MobileNetV2's projection is linear on purpose; a ReLU there destroys information in the narrow bottleneck and hurts accuracy. Fix: no activation after the block's final 1×1.",
        "**Forgetting the projection shortcut when shapes change.** Adding $x$ to $F(x)$ fails (or silently broadcasts wrongly) when the stride or channel count changes. Fix: a 1×1 strided conv + BN on the shortcut of each stage's first block.",
        "**Leaving BatchNorm in training mode at inference (or fine-tuning with tiny batches).** Outputs depend on the batch and running statistics drift. Fix: `model.eval()` for inference; freeze BN statistics or use GroupNorm/LayerNorm when fine-tuning with very small batches (see [Normalization](/learn/m07/normalization)).",
        "**Treating Keras and PyTorch parameter counts as the same thing.** Keras includes BN moving statistics as non-trainable parameters. Fix: compare trainable parameters.",
        "**Believing deeper is always better.** Our digits experiment shows a 22-layer plain net training worse than a 6-layer one; even with residuals, depth beyond what the data and resolution support wastes compute. Fix: scale depth together with width and resolution, and validate.",
      ] },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "Why did VGG use only 3×3 convolutions?", a: "Stacked small kernels give the same receptive field as one large kernel (two 3×3 → 5×5, three → 7×7) with fewer parameters ($18C^2$ vs $25C^2$, $27C^2$ vs $49C^2$) and a non-linearity after each layer, making the function more expressive. It also keeps the design uniform and easy to deepen." },
      { t: "qa", level: "beginner", q: "What is a 1×1 convolution and why is it useful?", a: "A convolution whose kernel covers one pixel: at each position it computes a learned mix of the input channels — a fully connected layer applied per pixel. It changes the channel count cheaply. Inception and ResNet bottlenecks use it to cut channels before expensive 3×3/5×5 convolutions; MobileNet uses it as the pointwise half of a separable convolution." },
      { t: "qa", level: "beginner", q: "What problem do residual connections solve in ResNet?", a: "The degradation problem: deeper plain networks had *higher training error* than shallower ones because they were harder to optimise. With $y = F(x) + x$ each block only learns a correction to the identity, the identity is free to represent, and gradients have a direct additive path to early layers — so 100+ layer networks train well." },
      { t: "qa", level: "beginner", q: "Why do modern CNNs end with global average pooling instead of large fully connected layers?", a: "Averaging each final feature map to one number removes the spatial dimensions with zero parameters, so the classifier is one small linear layer. VGG-16's FC layers hold 89% of its 138 M parameters; GAP heads (GoogLeNet onward) avoid that, reduce overfitting, and allow any input resolution." },
      { t: "qa", level: "intermediate", q: "Derive the cost ratio of a depthwise separable convolution to a standard one.", a: "Standard: $C_{in}C_{out}k^2HW$ MACs. Separable: depthwise $C_{in}k^2HW$ + pointwise $C_{in}C_{out}HW$. Ratio $= 1/C_{out} + 1/k^2$. For $k = 3$ and large $C_{out}$ it approaches $1/9$, i.e. 8–9× cheaper; e.g. 0.1189 for 128 channels." },
      { t: "qa", level: "intermediate", q: "How does MobileNetV2's inverted residual differ from a ResNet bottleneck, and why is its projection linear?", a: "ResNet: wide → narrow (1×1) → 3×3 → wide (1×1), shortcut between wide tensors. MobileNetV2: narrow → expand ×t (1×1) → 3×3 depthwise → project to narrow (1×1), shortcut between narrow tensors, which saves memory. The projection has no ReLU because ReLU in a low-dimensional space zeroes out information that can't be recovered; a linear bottleneck keeps it." },
      { t: "qa", level: "intermediate", q: "What is a dense block's growth rate, and how many channels leave a block of L layers?", a: String.raw`Each layer adds $k$ new feature maps (the growth rate) and receives all earlier maps concatenated. A block with $c_0$ input channels outputs $c_0 + Lk$; e.g. DenseNet-121 block 1: $64 + 6\cdot32 = 256$. Transition layers then halve the channels and the resolution.` },
      { t: "qa", level: "intermediate", q: "Why does the number of channels usually double when the spatial resolution halves?", a: "A conv layer's MACs are $C_{in}C_{out}k^2HW$. Doubling both channel counts multiplies that by 4 and halving H and W divides by 4, so each layer's compute stays roughly constant across stages, while the representation trades spatial detail for more, richer features. VGG-16's six equal-cost layers (1.85 G MACs each) show this exactly." },
      { t: "qa", level: "advanced", q: "Explain EfficientNet's compound scaling and why the constraint is $\\alpha\\beta^2\\gamma^2 \\approx 2$.", a: String.raw`Depth, width and resolution are scaled as $\alpha^\phi, \beta^\phi, \gamma^\phi$. FLOPs grow linearly with depth and quadratically with width (both $C_{in}$ and $C_{out}$ grow) and with resolution ($H$ and $W$ grow), so FLOPs scale as $(\alpha\beta^2\gamma^2)^\phi$. Fixing the product at ≈2 means each step of $\phi$ doubles compute; α, β, γ come from a small grid search at φ = 1 and are then reused. Intuition: higher resolution needs more depth (bigger receptive field) and more width (finer patterns), so balanced scaling beats scaling one dimension.` },
      { t: "qa", level: "advanced", q: "If MACs drop by 8× with depthwise separable convolutions, why is the GPU speed-up often much smaller?", a: "Depthwise convolutions have very low arithmetic intensity: each weight is used on one channel only, so the operation moves almost as much data as it computes and is limited by memory bandwidth. GPUs are built for large dense matrix multiplications (which standard and 1×1 convs map onto well). Extra small ops (BN, activations, SE) add kernel-launch and memory overhead. So models are compared by measured latency, and designs like EfficientNetV2 bring regular convolutions back into early stages." },
      { t: "qa", level: "advanced", q: "What did ConvNeXt change relative to ResNet-50, and what does its result suggest?", a: "A modern training recipe (AdamW, long schedule, Mixup/CutMix, RandAugment, stochastic depth, label smoothing); 3:3:9:3 stage ratio; 4×4 stride-4 patchify stem; depthwise convolutions with wider channels; inverted bottleneck; 7×7 depthwise kernels first in the block; GELU; fewer activations and norms; LayerNorm instead of BN; separate downsampling layers; layer scale. It matched hierarchical Vision Transformers of similar size, suggesting much of the Transformers' edge in vision came from overall design and training recipe rather than self-attention itself." },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "How many parameters (weights + biases) does a 3×3 convolution from 64 to 128 channels have?", answer: 73856, tolerance: 0, explain: "$128\\cdot64\\cdot9 + 128 = 73{,}728 + 128 = 73{,}856$." },
        { kind: "numeric", q: "What is the receptive field (in pixels, one side) of three stacked 3×3 convolutions with stride 1?", answer: 7, tolerance: 0, explain: "$r = 1 + 3\\cdot(3-1) = 7$ — the same as one 7×7 convolution." },
        { kind: "numeric", q: "Cost ratio of a depthwise separable 3×3 conv to a standard 3×3 conv with 64 output channels (3 decimals)?", answer: 0.127, tolerance: 0.001, explain: "$1/64 + 1/9 = 0.015625 + 0.111111 = 0.1267$." },
        { kind: "numeric", q: "A dense block receives 128 channels and has 12 layers with growth rate 32. How many channels does it output?", answer: 512, tolerance: 0, explain: "$128 + 12\\cdot32 = 512$ (DenseNet-121's second block)." },
        { kind: "numeric", q: "With EfficientNet's α = 1.2, β = 1.1, γ = 1.15, by what factor do FLOPs grow for φ = 1 (2 decimals)?", answer: 1.92, tolerance: 0.01, explain: "$\\alpha\\beta^2\\gamma^2 = 1.2\\times1.21\\times1.3225 = 1.920$." },
        { q: "Which statement about MobileNetV2's inverted residual block is correct?", options: ["It reduces channels with a 1×1 conv, applies a 3×3 conv, then expands", "It expands channels, applies a depthwise 3×3 conv, then projects linearly to a narrow output with the shortcut between narrow tensors", "It concatenates all previous layers' outputs", "It uses parallel 1×1, 3×3 and 5×5 branches"], answer: 1, explain: "Narrow → wide → narrow, with a linear (no-ReLU) projection. Option 1 is the ResNet bottleneck, 3 is DenseNet, 4 is Inception." },
        { q: "Where are most of VGG-16's parameters, and where is most of its compute?", options: ["Both in the convolutions", "Both in the FC layers", "Parameters in the FC layers; compute in the convolutions", "Parameters in the convolutions; compute in the FC layers"], answer: 2, explain: "Measured: 89.4% of parameters in the FC layers, 99.2% of MACs in the convolutions." },
        { q: "What was the main purpose of the 1×1 convolutions placed before the 3×3 and 5×5 convolutions in GoogLeNet's Inception module?", options: ["To add non-linearity only", "To reduce the number of channels so the expensive convolutions are cheap", "To downsample spatially", "To normalise the activations"], answer: 1, explain: "They are bottlenecks: in module 3a they cut the 5×5 branch from 120.4 M to 12.4 M MACs." },
        { q: "In the digits experiment, the 22-layer plain network trained more slowly than the 6-layer one even though its stem gradient was larger. What does this illustrate?", options: ["Overfitting", "The degradation problem is an optimisation difficulty, not just vanishing gradients", "BatchNorm is harmful", "Adam does not work with CNNs"], answer: 1, explain: "With BatchNorm gradients don't vanish, yet the deeper plain net is harder to optimise; residual connections fix this." },
      ] },
      { t: "exercise", title: "Bottleneck versus basic: count it by hand", prompt: "A ResNet stage runs at 56×56 with 256 channels. Compare (a) a residual branch of two 3×3 convolutions at 256 channels and (b) a bottleneck branch 1×1 (256 → 64), 3×3 (64 → 64), 1×1 (64 → 256). Ignore biases and BatchNorm. Give each one's weights and MACs, and the ratio.", hint: "MACs = weights × H × W for stride-1 convolutions with 'same' padding.", solution: [
        { t: "steps", items: [
          { title: "(a) Two 3×3 convs", tex: String.raw`2\cdot9\cdot256\cdot256 = 1{,}179{,}648 \text{ weights};\quad \times 56^2 = 3{,}699.4\text{M MACs}` },
          { title: "(b) Bottleneck", tex: String.raw`256\cdot64 + 9\cdot64\cdot64 + 64\cdot256 = 16{,}384 + 36{,}864 + 16{,}384 = 69{,}632;\quad \times 56^2 = 218.4\text{M MACs}` },
          { title: "Ratio", tex: String.raw`1{,}179{,}648 / 69{,}632 \approx 16.9` },
        ] },
        { t: "p", text: "The bottleneck runs the 3×3 on a quarter of the channels (a 16× saving on that layer) and pays only for two cheap 1×1 projections — this is what makes ResNet-50/101/152 affordable." },
      ] },
      { t: "exercise", title: "Implement a ResNet-50 bottleneck block and verify it", prompt: "Write a PyTorch `Bottleneck` module (1×1 reduce → BN-ReLU → 3×3 (with the stride) → BN-ReLU → 1×1 expand ×4 → BN, plus an identity or projection shortcut, then ReLU). Check the output shape and parameter count for input 256×56×56 with width 64, and verify that it computes exactly the same function as torchvision's `Bottleneck` once the weights are copied.", hint: "With bias-free convs, parameters = conv weights + 2 per BN channel. Copy weights by zipping the Conv2d/BatchNorm2d modules of the two blocks in order.", solution: [
        { t: "code", lang: "python", title: "bottleneck_solution.py", code: BOTTLENECK_SOL, output: BOTTLENECK_SOL_OUT },
        { t: "p", text: "The output difference is exactly zero because both blocks do the same operations in the same order. The 768 BN parameters are γ and β for 64 + 64 + 256 channels; the running statistics are buffers, copied by `load_state_dict` but not counted as parameters." },
      ] },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      { t: "list", items: [
        "CNNs share a stem → stages → head skeleton; families differ in the block and in how they spend depth, width and resolution.",
        "Conv cost: $P = C_{out}(C_{in}/g)k^2 + C_{out}$, MACs $= C_{out}(C_{in}/g)k^2H_{out}W_{out}$. Parameters and compute are separate budgets; latency is a third.",
        "**LeNet-5** (1998): learned conv features, 61.7 K params. **AlexNet** (2012): ReLU, GPUs, dropout, augmentation; 61.1 M params, mostly FC.",
        "**VGG** (2014): 3×3 stacks — $9nC^2$ vs $(2n+1)^2C^2$ weights for the same receptive field; 138.4 M params, 15.5 G MACs.",
        "**Inception** (2014): parallel multi-scale branches with 1×1 bottlenecks (≈10× cheaper 5×5 branch), GAP head.",
        "**ResNet** (2015): $y = \\mathrm{ReLU}(F(x) + x)$ solves degradation; bottlenecks (≈17× fewer weights than two 3×3s at 256 channels) make 50–152 layers affordable.",
        "**DenseNet** (2016): concatenate all previous maps, $c_0 + Lk$ channels; very parameter-efficient, memory-hungry.",
        "**Xception / MobileNet-v1**: depthwise separable convs, cost ratio $1/C_{out} + 1/k^2$. **MobileNet-v2**: inverted residual with linear bottleneck. **v3**: NAS, SE, hard-swish.",
        "**EfficientNet** (2019): $d = \\alpha^\\phi$, $w = \\beta^\\phi$, $r = \\gamma^\\phi$ with $\\alpha\\beta^2\\gamma^2 \\approx 2$; B0 has 5.3 M params, 0.39 G MACs.",
        "**ConvNeXt** (2022): ResNet modernised with patchify stem, 7×7 depthwise, inverted bottleneck, LayerNorm, GELU and a modern recipe; ~ResNet-50 budget, Transformer-level accuracy.",
        "Choosing today: a pretrained checkpoint first, then the deployment constraint (measured latency/memory), then the smallest family that meets the accuracy target.",
      ] },
    ],

    // ------------------------------------------------------------------- next
    next: [
      { t: "list", items: [
        "[Image Classification, Transfer Learning and Fine-Tuning](/learn/m12/transfer-learning) — put these backbones to work: feature extraction vs fine-tuning, freezing layers, picking learning rates.",
        "[BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families) — the Vision Transformers ConvNeXt was measured against, and how attention replaces convolution for mixing information across positions.",
        "[Object Detection](/learn/m12/object-detection) and [Segmentation](/learn/m12/segmentation) — how these backbones become detectors and segmenters with feature pyramids and decoders.",
        "[Model Optimization](/learn/m18/model-optimization) — quantisation, pruning and distillation, the next step after picking an efficient architecture.",
      ] },
    ],
  },
};

export default lesson;
