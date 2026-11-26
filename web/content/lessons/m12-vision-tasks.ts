import type { Lesson } from "@/lib/types";

const VT_METRICS = String.raw`import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # small matrices: one BLAS thread is faster and bit-reproducible
import numpy as np
from sklearn.datasets import load_digits


# ======================= 1. PSNR and SSIM from scratch =======================
def psnr(x, y, data_range=1.0):
    mse = np.mean((x - y) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(data_range ** 2 / mse)


def gaussian_window(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-ax ** 2 / (2 * sigma ** 2))
    return g / g.sum()                                   # 1-D; the 2-D window is its outer product (separable)


def filt(img, g):
    """'valid' separable Gaussian filtering: rows then columns."""
    tmp = np.apply_along_axis(lambda r: np.convolve(r, g, mode="valid"), 1, img)
    return np.apply_along_axis(lambda c: np.convolve(c, g, mode="valid"), 0, tmp)


def ssim(x, y, data_range=1.0, size=11, sigma=1.5, K1=0.01, K2=0.03):
    """Mean SSIM with the standard settings of Wang et al. (2004): 11x11 Gaussian window, sigma 1.5."""
    g = gaussian_window(size, sigma)
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    mu_x, mu_y = filt(x, g), filt(y, g)                  # local means
    var_x = filt(x * x, g) - mu_x ** 2                   # local variances  E[x^2] - E[x]^2
    var_y = filt(y * y, g) - mu_y ** 2
    cov = filt(x * y, g) - mu_x * mu_y                   # local covariance
    s_map = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return s_map.mean()


# A smooth 64x64 test image built from sklearn digits (8x8 digits tiled 2x2, each upsampled 4x)
d = load_digits().images / 16.0
img = np.block([[np.kron(d[0], np.ones((4, 4))), np.kron(d[1], np.ones((4, 4)))],
                [np.kron(d[2], np.ones((4, 4))), np.kron(d[3], np.ones((4, 4)))]])
rng = np.random.default_rng(0)

# Distortion A: additive Gaussian noise.  Distortion B: blur (5x5 box), then rescaled to the SAME MSE as A.
noisy = img + rng.normal(0, 0.1, img.shape)
k = np.ones(5) / 5
blurred = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, img)
blurred = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, blurred)
err = blurred - img
blurred = img + err * np.sqrt(np.mean((noisy - img) ** 2) / np.mean(err ** 2))   # match MSE exactly
shifted = img + 0.1                                                               # constant brightness shift

print("image", img.shape, "range [%.2f, %.2f]" % (img.min(), img.max()))
print(f"{'distortion':18s}{'MSE':>9s}{'PSNR dB':>9s}{'SSIM':>8s}")
for name, z in [("identical", img), ("Gaussian noise", noisy), ("blur (same MSE)", blurred), ("brightness +0.1", shifted)]:
    print(f"{name:18s}{np.mean((z - img) ** 2):9.5f}{psnr(img, z):9.2f}{ssim(img, z):8.4f}")
print("symmetry  SSIM(a,b) - SSIM(b,a) =", f"{ssim(img, noisy) - ssim(noisy, img):.1e}")
print("PSNR check: MSE 0.01 on [0,1] data ->", psnr(np.zeros(4), np.full(4, 0.1)), "dB")

# ======================= 2. Keypoint heatmaps: encode, decode, sub-pixel refinement =======================
H = W = 16                                               # heatmap resolution (e.g. 64x64 input, stride 4)
ys_, xs_ = np.mgrid[0:H, 0:W]


def encode(kx, ky, sigma=1.0):
    return np.exp(-((xs_ - kx) ** 2 + (ys_ - ky) ** 2) / (2 * sigma ** 2))   # Gaussian centred on the keypoint


def decode_argmax(hm):
    iy, ix = np.unravel_index(hm.argmax(), hm.shape)
    return float(ix), float(iy)


def decode_subpixel(hm):
    """Fit a parabola to the log-heatmap through the peak and its two neighbours, per axis.
    For an exact Gaussian the log is an exact parabola, so this recovers the true centre."""
    ix, iy = map(int, decode_argmax(hm))
    lh = np.log(np.maximum(hm, 1e-12))
    dx = 0.5 * (lh[iy, ix - 1] - lh[iy, ix + 1]) / (lh[iy, ix - 1] - 2 * lh[iy, ix] + lh[iy, ix + 1])
    dy = 0.5 * (lh[iy - 1, ix] - lh[iy + 1, ix]) / (lh[iy - 1, ix] - 2 * lh[iy, ix] + lh[iy + 1, ix])
    return ix + dx, iy + dy


def soft_argmax(hm, beta=10.0):
    p = np.exp(beta * (hm - hm.max())); p /= p.sum()   # softmax over all pixels
    return float((p * xs_).sum()), float((p * ys_).sum())


true_pts = rng.uniform(3, 12, size=(200, 2))           # 200 random sub-pixel keypoints
errs = {"argmax": [], "parabola": [], "soft-argmax": []}
for kx, ky in true_pts:
    hm = encode(kx, ky) + rng.normal(0, 0.01, (H, W))   # a slightly noisy "predicted" heatmap
    hm = np.clip(hm, 1e-6, None)
    for name, f in [("argmax", decode_argmax), ("parabola", decode_subpixel), ("soft-argmax", soft_argmax)]:
        px, py = f(hm)
        errs[name].append(np.hypot(px - kx, py - ky))
print("\nkeypoint decoding error on a 16x16 heatmap (pixels; x4 for a stride-4 network):")
for name, e in errs.items():
    print(f"  {name:12s} mean {np.mean(e):.3f}   max {np.max(e):.3f}")

# ======================= 3. Optical flow: endpoint error (EPE) =======================
gt = np.stack([np.full((H, W), 1.5), np.full((H, W), -0.5)], axis=-1)        # true flow (u, v) per pixel
pred = gt + rng.normal(0, 0.3, gt.shape)                                    # a noisy prediction
epe = np.linalg.norm(pred - gt, axis=-1)                                    # per-pixel Euclidean error
print(f"\nflow EPE: mean {epe.mean():.3f} px, fraction of pixels with EPE > 1 px: {np.mean(epe > 1):.3f}")
print(f"zero-flow baseline EPE: {np.linalg.norm(gt, axis=-1).mean():.3f} px")

# Brightness constancy, linearised: I_x u + I_y v + I_t = 0 at every pixel. Lucas-Kanade solves it by least squares
# over a window (here: the whole image, since the true motion is a single global shift).
yy, xx = np.mgrid[0:32, 0:32].astype(float)
scene = lambda x, y: np.sin(0.35 * x) + np.cos(0.25 * y) + 0.5 * np.sin(0.2 * (x + y))
u_true, v_true = 0.6, -0.3
I1, I2 = scene(xx, yy), scene(xx - u_true, yy - v_true)        # frame 2 = frame 1 moved by (u, v)
Iy, Ix = np.gradient((I1 + I2) / 2)                             # spatial derivatives (rows = y, cols = x)
It = I2 - I1                                                    # temporal derivative
A = np.stack([Ix.ravel(), Iy.ravel()], axis=1)
u_est, v_est = np.linalg.lstsq(A, -It.ravel(), rcond=None)[0]
print(f"Lucas-Kanade on a smooth scene: estimated (u, v) = ({u_est:.3f}, {v_est:.3f}), true ({u_true}, {v_true}),"
      f" EPE {np.hypot(u_est - u_true, v_est - v_true):.3f} px")`;

const VT_METRICS_OUT = String.raw`image (64, 64) range [0.00, 1.00]
distortion              MSE  PSNR dB    SSIM
identical           0.00000      inf  1.0000
Gaussian noise      0.00995    20.02  0.7103
blur (same MSE)     0.00995    20.02  0.8221
brightness +0.1     0.01000    20.00  0.7376
symmetry  SSIM(a,b) - SSIM(b,a) = 0.0e+00
PSNR check: MSE 0.01 on [0,1] data -> 20.0 dB

keypoint decoding error on a 16x16 heatmap (pixels; x4 for a stride-4 network):
  argmax       mean 0.374   max 0.650
  parabola     mean 0.017   max 0.060
  soft-argmax  mean 0.146   max 0.298

flow EPE: mean 0.371 px, fraction of pixels with EPE > 1 px: 0.000
zero-flow baseline EPE: 1.581 px
Lucas-Kanade on a smooth scene: estimated (u, v) = (0.611, -0.305), true (0.6, -0.3), EPE 0.012 px`;

const VT_SR = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.set_num_threads(1)
torch.manual_seed(0)

# ---------------- 1. PixelShuffle is just a reshape + permute ----------------
r = 2
x = torch.arange(1 * 8 * 3 * 3, dtype=torch.float32).reshape(1, 8, 3, 3)   # (N, C*r^2, H, W) with C = 2
ps = nn.PixelShuffle(r)(x)                                                   # -> (N, C, H*r, W*r)
N, Cr2, H, W = x.shape
C = Cr2 // r ** 2
manual = x.reshape(N, C, r, r, H, W).permute(0, 1, 4, 2, 5, 3).reshape(N, C, H * r, W * r)
print("PixelShuffle", tuple(x.shape), "->", tuple(ps.shape), " equal to reshape/permute:", torch.equal(ps, manual))
print("channel 0 of the output (top-left 4x4):\n", ps[0, 0, :4, :4].int().numpy())
print("round trip with PixelUnshuffle:", torch.equal(nn.PixelUnshuffle(r)(ps), x))

# ---------------- 2. A tiny ESPCN-style x2 super-resolution net on sklearn digits ----------------
hr = torch.tensor(load_digits().images / 16.0, dtype=torch.float32).unsqueeze(1)   # (1797, 1, 8, 8) HR targets
lr = F.avg_pool2d(hr, 2)                                                           # (1797, 1, 4, 4) LR inputs
perm = torch.randperm(len(hr))
tr, te = perm[:1400], perm[1400:]


def psnr(a, b):                                       # per-image PSNR (data range 1), averaged over images
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def up(x, mode):
    if mode == "nearest":
        return F.interpolate(x, scale_factor=2, mode="nearest")
    return F.interpolate(x, scale_factor=2, mode=mode, align_corners=False).clamp(0, 1)


class ESPCN(nn.Module):
    """Features are computed at LOW resolution; one conv produces r^2 channels per output channel,
    and PixelShuffle rearranges them into an r-times larger image (sub-pixel convolution)."""
    def __init__(self, r=2, width=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, r * r, 3, padding=1))
        self.shuffle = nn.PixelShuffle(r)

    def forward(self, x):
        return self.shuffle(self.body(x)) + up(x, "bicubic")   # predict the residual over bicubic


net = ESPCN()
print("\nESPCN parameters:", sum(p.numel() for p in net.parameters()))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for epoch in range(1, 41):
    order = tr[torch.randperm(len(tr))]                         # reshuffle every epoch
    for i in range(0, len(tr), 100):
        b = order[i:i + 100]
        loss = F.mse_loss(net(lr[b]), hr[b])                    # pixel (L2) loss
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            print(f"epoch {epoch:2d}  train MSE {loss.item():.5f}  test PSNR {psnr(net(lr[te]).clamp(0, 1), hr[te]):.2f} dB")

with torch.no_grad():
    print(f"\nx2 upsampling of {len(te)} held-out 4x4 digits to 8x8 (mean PSNR, dB):")
    for name, out in [("nearest", up(lr[te], "nearest")), ("bilinear", up(lr[te], "bilinear")),
                      ("bicubic", up(lr[te], "bicubic")), ("ESPCN (learned)", net(lr[te]).clamp(0, 1))]:
        print(f"  {name:16s}{psnr(out, hr[te]):6.2f}")`;

const VT_SR_OUT = String.raw`PixelShuffle (1, 8, 3, 3) -> (1, 2, 6, 6)  equal to reshape/permute: True
channel 0 of the output (top-left 4x4):
 [[ 0  9  1 10]
 [18 27 19 28]
 [ 3 12  4 13]
 [21 30 22 31]]
round trip with PixelUnshuffle: True

ESPCN parameters: 10724
epoch 10  train MSE 0.01968  test PSNR 17.46 dB
epoch 20  train MSE 0.01612  test PSNR 18.28 dB
epoch 30  train MSE 0.01510  test PSNR 18.67 dB
epoch 40  train MSE 0.01353  test PSNR 18.89 dB

x2 upsampling of 397 held-out 4x4 digits to 8x8 (mean PSNR, dB):
  nearest          13.27
  bilinear         12.88
  bicubic          13.52
  ESPCN (learned)  18.89`;

const VT_VERIF = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score

torch.set_num_threads(1)
torch.manual_seed(0)
d = load_digits()
X = torch.tensor(d.data / 16.0, dtype=torch.float32)
y = torch.tensor(d.target)

TRAIN, TEST = slice(0, 1200), slice(1200, None)       # images 0-1199 for training, 1200-1796 for evaluation


class ArcFaceHead(nn.Module):
    """Logits = s * cos(theta_j), with an additive angular margin m on the true class: s * cos(theta_y + m)."""
    def __init__(self, dim, n_classes, s=16.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.1)
        self.s, self.m = s, m

    def forward(self, emb, labels):
        cos = F.normalize(emb) @ F.normalize(self.W).T                      # cosine to every class centre
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)                                  # penalised true-class logit
        onehot = F.one_hot(labels, cos.shape[1]).bool()
        return self.s * torch.where(onehot, target, cos)


def train_embedder(Xtr, ytr, margin):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))  # 32-D embedding
    head = ArcFaceHead(32, int(ytr.max()) + 1, m=margin)
    opt = torch.optim.Adam([*net.parameters(), *head.parameters()], lr=3e-3)
    for epoch in range(20):
        order = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = order[i:i + 64]
            loss = F.cross_entropy(head(net(Xtr[b]), ytr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def roc(scores, same):
    """Sweep the threshold over every score: TAR = true accept rate, FAR = false accept rate."""
    order = np.argsort(-scores)
    s = same[order]
    tar = np.concatenate([[0], np.cumsum(s) / s.sum()])
    far = np.concatenate([[0], np.cumsum(~s) / (~s).sum()])
    return far, tar


def evaluate(name, emb, labels, pairs):
    e = F.normalize(emb, dim=1).numpy()
    lab = labels.numpy()
    same = lab[pairs[:, 0]] == lab[pairs[:, 1]]
    scores = (e[pairs[:, 0]] * e[pairs[:, 1]]).sum(1)                  # cosine similarity per pair
    far, tar = roc(scores, same)
    auc = np.trapezoid(tar, far)                                        # area under the ROC curve
    tar_at = lambda f: tar[np.searchsorted(far, f, side="right") - 1]
    # identification: enrol ONE gallery image per identity; every other image is a probe
    ids = np.unique(lab)
    gallery = [int(np.flatnonzero(lab == c)[0]) for c in ids]
    probes = np.setdiff1d(np.arange(len(e)), gallery)
    rank1 = np.mean(ids[(e[probes] @ e[gallery].T).argmax(1)] == lab[probes])
    print(f"  {name:22s} AUC {auc:.4f} (sklearn {roc_auc_score(same, scores):.4f})  "
          f"TAR@FAR=1% {tar_at(0.01):.3f}  TAR@FAR=0.1% {tar_at(0.001):.3f}  rank-1 {rank1:.3f}")


Xte, yte = X[TEST], y[TEST]
for title, train_mask, test_mask in [
        ("A) closed set: identities 0-9 seen in training, new images", y[TRAIN] >= 0, yte >= 0),
        ("B) open set: train on identities 0-6, test on UNSEEN 7-9", y[TRAIN] < 7, yte >= 7)]:
    Xe, ye = Xte[test_mask], yte[test_mask]
    pairs = np.stack(np.triu_indices(len(Xe), k=1), axis=1)            # every unordered pair once
    print(f"{title}: {len(Xe)} test images, {len(pairs)} pairs")
    evaluate("raw pixels (cosine)", Xe, ye, pairs)
    for m in (0.0, 0.3):
        net = train_embedder(X[TRAIN][train_mask], y[TRAIN][train_mask], m)
        with torch.no_grad():
            evaluate(f"embedding, margin {m}", net(Xe), ye, pairs)`;

const VT_VERIF_OUT = String.raw`A) closed set: identities 0-9 seen in training, new images: 597 test images, 177906 pairs
  raw pixels (cosine)    AUC 0.8619 (sklearn 0.8619)  TAR@FAR=1% 0.452  TAR@FAR=0.1% 0.219  rank-1 0.700
  embedding, margin 0.0  AUC 0.9669 (sklearn 0.9669)  TAR@FAR=1% 0.736  TAR@FAR=0.1% 0.531  rank-1 0.818
  embedding, margin 0.3  AUC 0.9792 (sklearn 0.9792)  TAR@FAR=1% 0.837  TAR@FAR=0.1% 0.617  rank-1 0.852
B) open set: train on identities 0-6, test on UNSEEN 7-9: 174 test images, 15051 pairs
  raw pixels (cosine)    AUC 0.8193 (sklearn 0.8193)  TAR@FAR=1% 0.342  TAR@FAR=0.1% 0.186  rank-1 0.743
  embedding, margin 0.0  AUC 0.8137 (sklearn 0.8137)  TAR@FAR=1% 0.343  TAR@FAR=0.1% 0.156  rank-1 0.649
  embedding, margin 0.3  AUC 0.8032 (sklearn 0.8032)  TAR@FAR=1% 0.334  TAR@FAR=0.1% 0.135  rank-1 0.591`;

const VT_CAPTION = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(0)
VOCAB, EMB, HID, FEAT = 12, 16, 32, 64       # toy sizes; real systems: ~10k words, 512-dim states, 2048-dim features
BOS, EOS = 1, 2


class Encoder(nn.Module):
    """CNN that keeps a spatial grid: every grid cell becomes one 'region' vector the decoder can attend to."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, FEAT, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, img):                              # (N, 3, 32, 32)
        f = self.cnn(img)                                # (N, FEAT, 8, 8)
        return f.flatten(2).transpose(1, 2)              # (N, 64 regions, FEAT)


class AttnDecoder(nn.Module):
    """GRU decoder with additive (Bahdanau-style) attention over image regions, as in Show, Attend and Tell."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, EMB)
        self.init_h = nn.Linear(FEAT, HID)
        self.W_f, self.W_h, self.v = nn.Linear(FEAT, HID), nn.Linear(HID, HID), nn.Linear(HID, 1)
        self.gru = nn.GRUCell(EMB + FEAT, HID)
        self.out = nn.Linear(HID, VOCAB)

    def step(self, word, h, regions):
        scores = self.v(torch.tanh(self.W_f(regions) + self.W_h(h).unsqueeze(1))).squeeze(-1)   # (N, R)
        alpha = scores.softmax(dim=1)                                                           # attention weights
        context = (alpha.unsqueeze(-1) * regions).sum(1)                                        # (N, FEAT)
        h = self.gru(torch.cat([self.embed(word), context], dim=1), h)
        return self.out(h), h, alpha

    def forward(self, regions, captions):                # teacher forcing: feed the ground-truth previous word
        h = torch.tanh(self.init_h(regions.mean(1)))
        logits = []
        for t in range(captions.shape[1] - 1):
            lg, h, alpha = self.step(captions[:, t], h, regions)
            logits.append(lg)
        return torch.stack(logits, 1), alpha


enc, dec = Encoder(), AttnDecoder()
images = torch.rand(2, 3, 32, 32)
captions = torch.tensor([[BOS, 5, 7, 9, EOS], [BOS, 4, 6, EOS, 0]])        # 0 = padding
regions = enc(images)
logits, alpha = dec(regions, captions)
loss = F.cross_entropy(logits.reshape(-1, VOCAB), captions[:, 1:].reshape(-1), ignore_index=0)
print("regions", tuple(regions.shape), "| logits", tuple(logits.shape), "| last attention map", tuple(alpha.shape))
print("attention weights sum to 1:", torch.allclose(alpha.sum(1), torch.ones(2)))
print(f"teacher-forced cross-entropy {loss.item():.3f}  (uniform guess = ln {VOCAB} = {torch.log(torch.tensor(float(VOCAB))).item():.3f})")

# greedy decoding at inference: feed back the model's own previous word (weights are untrained here)
with torch.no_grad():
    h = torch.tanh(dec.init_h(regions.mean(1)))
    word, out = torch.full((2,), BOS), []
    for t in range(6):
        lg, h, alpha = dec.step(word, h, regions)
        word = lg.argmax(1)
        out.append(word)
print("greedy token ids (untrained):", torch.stack(out, 1).tolist())
print("most-attended region of image 0 at the last step:", divmod(alpha[0].argmax().item(), 8), "(row, col) of the 8x8 grid")`;

const VT_CAPTION_OUT = String.raw`regions (2, 64, 64) | logits (2, 4, 12) | last attention map (2, 64)
attention weights sum to 1: True
teacher-forced cross-entropy 2.416  (uniform guess = ln 12 = 2.485)
greedy token ids (untrained): [[9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9]]
most-attended region of image 0 at the last step: (7, 0) (row, col) of the 8x8 grid`;

const VT_KERAS = String.raw`import tensorflow as tf

R = 2   # upscaling factor

def espcn(r=R, width=32):
    lr = tf.keras.Input(shape=(None, None, 1))                   # any LR size, 1 channel
    x = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu")(lr)
    x = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(r * r, 3, padding="same")(x)      # r^2 sub-pixel channels
    sr = tf.keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, r))(x)   # Keras/TF's PixelShuffle (NHWC)
    return tf.keras.Model(lr, sr)

model = espcn()
model.compile(optimizer="adam", loss="mse")
hr = tf.random.uniform((4, 8, 8, 1))
lr_img = tf.nn.avg_pool2d(hr, 2, 2, "VALID")
print(model(lr_img).shape)                                        # (4, 8, 8, 1)
print("PSNR:", tf.image.psnr(model(lr_img), hr, max_val=1.0).numpy())
print("SSIM:", tf.image.ssim(hr, hr, max_val=1.0, filter_size=3).numpy())   # identical images -> 1.0`;

const lesson: Lesson = {
  id: "m12/vision-tasks",
  title: "Faces, Pose, Optical Flow, Super-Resolution and Captioning",
  summary:
    "A guided tour of vision tasks beyond classification, detection and segmentation: face detection and recognition, human pose estimation, optical flow, super-resolution and image captioning (plus a pointer to image generation). For each you get the problem, the input and output tensors, a standard architecture, the loss, the metric and the pitfalls. PSNR/SSIM, heatmap decoding, endpoint error, sub-pixel convolution, a trained super-resolution network and a face-verification ROC are all implemented and run.",
  minutes: 110,
  level: "advanced",
  prerequisites: ["m12/transfer-learning", "m08/conv-variants", "m05/advanced-losses", "m10/attention-intro", "m05/regression-losses"],
  next: ["m12/object-detection", "m12/segmentation", "m14/gans", "m14/diffusion", "m15/multimodal"],
  terms: ["face-verification", "face-identification", "arcface", "keypoint-heatmap", "part-affinity-field", "optical-flow", "brightness-constancy", "endpoint-error", "super-resolution", "pixel-shuffle", "psnr", "ssim", "perceptual-loss", "image-captioning", "tar-at-far"],
  sections: {
    intro: [
      { t: "p", text: "Image classification answers *what is in this image?* with one label. Many apps need other answers: *whose* face, *where* are the joints, *how* did each pixel move, *what* would it look like sharper, *how* would a person describe it? Each task has its own output, loss and metric, but they share building blocks: pretrained backbones ([Transfer Learning](/learn/m12/transfer-learning)), dense prediction heads, upsampling layers ([Transposed and Sub-Pixel Convolution](/learn/m08/conv-variants)), embedding losses ([Advanced Losses](/learn/m05/advanced-losses)) and attention ([Attention](/learn/m10/attention-intro))." },
      { t: "p", text: "For each task: the **problem**, **input → output tensors**, a **standard architecture**, the **loss**, the **metric** and the **pitfalls**. The code builds metrics and key operations from scratch and trains small real models, so every number shown is measured." },
      { t: "table", head: ["Task", "Input", "Output", "Typical loss", "Metric"], rows: [
        ["Face detection", "Image $3\\times H\\times W$", "Boxes + 5 landmarks per face", "Classification + box/landmark regression", "AP at IoU 0.5 (easy/medium/hard subsets)"],
        ["Face recognition", "Aligned face crop, e.g. $3\\times112\\times112$", "Unit embedding $e \\in \\mathbb{R}^{512}$", "Margin softmax (ArcFace) or triplet", "TAR@FAR (verification), rank-1 (identification)"],
        ["Pose estimation", "Image or person crop", "$K$ heatmaps $K\\times H'\\times W'$ (+ PAFs)", "MSE on heatmaps", "PCK, OKS-based AP"],
        ["Optical flow", "Two frames $2\\times3\\times H\\times W$", "Flow field $2\\times H\\times W$", "L1/L2 endpoint error over iterations", "EPE, outlier rate (Fl)"],
        ["Super-resolution", "LR image $C\\times H\\times W$", "HR image $C\\times rH\\times rW$", "L1/L2 pixel, perceptual, adversarial", "PSNR, SSIM, perceptual studies"],
        ["Image captioning", "Image", "Token sequence $w_1\\dots w_T$", "Cross-entropy (teacher forcing), then CIDEr-optimised RL", "BLEU, METEOR, CIDEr, SPICE"],
      ] },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Transfer Learning](/learn/m12/transfer-learning) — every task here starts from a pretrained backbone; know feature extraction vs fine-tuning.",
        "[Dilated, Transposed, Depthwise Convolutions](/learn/m08/conv-variants) — the upsampling layers in pose, flow and super-resolution networks.",
        "[Advanced Losses: Contrastive, Triplet](/learn/m05/advanced-losses) — metric learning is the heart of face recognition.",
        "[Regression Losses](/learn/m05/regression-losses) — MSE/L1 are the pixel losses for heatmaps, flow and super-resolution.",
        "[Attention](/learn/m10/attention-intro) and [Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) — captioning decoders attend over image regions.",
        "[Classification Metrics, ROC](/learn/m17/classification-metrics) — face verification is judged with ROC curves and operating points.",
        "[Object Detection](/learn/m12/object-detection) — face detection is detection of one class, with the same anchors, IoU and NMS.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "One camera, six questions", text: "A security guard watching a camera does several jobs at once. They notice *there is a face* (detection). They recognise *it is Alice* by comparing with faces they remember (recognition — by a sense of similarity, not one brain cell per person). They see *she is raising her arm* (pose), notice *the car is moving right* (optical flow), squint at a blurry licence plate (super-resolution — guessing plausible detail), and log *woman in red coat enters lobby* (captioning). Each answer has a different shape." },
      { t: "p", text: "The trick in each task is picking an **output representation** a convolutional network can predict well:" },
      { t: "list", items: [
        "Joints are drawn as blobs on heatmaps, not regressed as raw numbers, because convolutions are good at producing images.",
        "Faces are mapped to points in an embedding space, not sorted into a fixed list of people, because new people appear every day.",
        "Motion is an image with two channels $(u, v)$.",
        "A high-resolution image is predicted in pieces at low resolution, then shuffled into place.",
        "A sentence is produced one word at a time, each word looking at the image regions it needs.",
      ] },
    ],
    mechanism: [
      { t: "h", text: "1. Face detection" },
      { t: "p", text: "**Problem:** find every face (often tiny, hidden or rotated), usually with five landmarks (eyes, nose tip, mouth corners) used to align the face for recognition. **Architecture:** classic Viola–Jones (2001) used boosted cascades of Haar features. MTCNN (Zhang et al., 2016) chains three small CNNs (propose → refine → output with landmarks). Modern detectors such as RetinaFace are single-stage and anchor-based on a feature pyramid, predicting per anchor a face score, box offset and landmark offsets. **Loss:** classification (cross-entropy/focal) + smooth-L1 box regression + landmark regression. **Metric:** AP at IoU 0.5, on benchmarks split into easy/medium/hard by face size and occlusion. **Pitfalls:** small faces need high-resolution maps and small anchors; the NMS threshold trades missed faces in crowds against duplicates. See [Object Detection](/learn/m12/object-detection)." },
      { t: "h", text: "2. Face recognition: verification vs identification" },
      { t: "p", text: "**Verification** (1:1): are these two faces the same person? (Phone unlock, selfie vs passport.) **Identification** (1:N): which of the $N$ enrolled people is this, or none of them? Both use an **embedding network** $f$ that maps an aligned crop to a unit vector $e = f(x)/\\lVert f(x)\\rVert$. Similarity is the cosine $s = e_1^\\top e_2$. Verification puts a threshold on $s$. Identification takes the nearest gallery embedding, and rejects it if similarity is below a threshold (the *open-set* case). People at test time were **never seen in training**, so a classifier with one output per training person is useless at inference; only the embedding is kept." },
      { t: "steps", items: [
        { title: "Detect and align", text: "Warp the detected face (a similarity transform) so the five landmarks land on fixed spots in, e.g., a 112×112 crop. This removes pose and scale changes the network would otherwise have to learn." },
        { title: "Embed", text: "A ResNet-style backbone outputs a 512-dimensional feature, L2-normalised (scaled to length 1)." },
        { title: "Train with a metric-learning loss", text: "FaceNet (Schroff, Kalenichenko and Philbin, 2015) used a triplet loss on (anchor, positive, negative) triplets. Margin-based softmax losses later took over: a classifier over the training people, on normalised features and weights with a margin, so classes are separated by an angle. ArcFace (Deng, Guo, Xue and Zafeiriou, 2019) adds an angular margin $m$ to the true class." },
        { title: "Evaluate on unseen people", text: "Draw the ROC curve over genuine (same) and impostor (different) pairs. Report TAR (true accept rate) at a fixed FAR (false accept rate), e.g. TAR@FAR=$10^{-4}$, because a real system is set to a tolerable false-accept rate." },
      ] },
      { t: "p", text: "**Pitfalls:** training and test people must not overlap (otherwise you measure closed-set classification). Unbalanced demographics cause unequal error rates across groups. A threshold tuned on one dataset does not transfer to another camera or population. Privacy and consent are legal requirements in many places." },
      { t: "h", text: "3. Human pose estimation" },
      { t: "p", text: "**Problem:** find $K$ body keypoints for every person (COCO uses 17: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). **Regression vs heatmaps:** DeepPose (Toshev and Szegedy, 2014) regressed the $2K$ coordinates directly. That throws away spatial structure: a dense layer must turn 'feature at position p' into the number p. The winning approach predicts one **heatmap** per keypoint — an image with a Gaussian bump centred on the joint. The output lines up with the input, which suits convolutions. The location is read by argmax plus **sub-pixel refinement**, because heatmaps are 4× smaller than the input and a whole-pixel argmax would round positions." },
      { t: "list", items: [
        "**Top-down**: run a person detector, crop each person, predict $K$ heatmaps per crop (Stacked Hourglass, Newell, Yang and Deng, 2016; HRNet, Sun et al., 2019). Accurate, but cost grows with the crowd and it fails when the detector fails.",
        "**Bottom-up**: predict every keypoint of every person in one pass, then group them into people. OpenPose (Cao, Simon, Wei and Sheikh, 2017) predicts **Part Affinity Fields** — 2-channel arrow fields along each limb, pointing from one joint to the next. Joints are grouped by summing the field along candidate limbs. Cost barely depends on crowd size.",
        "**Loss:** MSE between predicted and target heatmaps (target Gaussian, $\\sigma$ ≈ 2 px at heatmap resolution), often only on visible/labelled joints. **Metrics:** PCK (share of keypoints within $\\alpha$ × a reference length, e.g. PCKh@0.5 relative to head size) and COCO's OKS-based AP.",
        "**Pitfalls:** horizontal flips must swap left/right labels; rounding error without sub-pixel decoding; hidden joints; crowded scenes for top-down methods.",
      ] },
      { t: "h", text: "4. Optical flow" },
      { t: "p", text: "**Problem:** for every pixel of frame 1, find the 2-D move $(u, v)$ to its spot in frame 2. Output: a $2\\times H\\times W$ tensor. Classical methods assume **brightness constancy** — a moving point keeps its brightness. Linearising gives the constraint $I_x u + I_y v + I_t = 0$, plus a smoothness prior (Horn and Schunck, 1981) or constant flow in a window (Lucas and Kanade, 1981). One equation per pixel for two unknowns is the **aperture problem**: locally you can only see motion along the image gradient." },
      { t: "list", items: [
        "**FlowNet** (Dosovitskiy et al., 2015): an encoder–decoder CNN on the two frames, with a correlation layer comparing feature patches. Trained on synthetic FlyingChairs, because dense true flow cannot be hand-labelled.",
        "**PWC-Net** (Sun et al., 2018): coarse-to-fine pyramid. At each level it warps frame-2 features by the current flow guess and computes a local cost volume.",
        "**RAFT** (Teed and Deng, 2020): computes an all-pairs correlation volume between every pixel of frame 1 and every pixel of frame 2 (pooled at several scales). A GRU then repeatedly updates one high-resolution flow field, looking up correlations around its current guess. The loss is a weighted sum of L1 errors over all iterations, with weights growing toward the last.",
        "**Metric:** average **endpoint error** (EPE: distance between predicted and true flow vectors, in pixels) and outlier percentages (KITTI's Fl counts pixels with error above 3 px and 5% of the true magnitude). Benchmarks: MPI-Sintel, KITTI.",
        "**Pitfalls:** occlusions (pixels with no match), large moves (beyond the correlation range), lighting changes, motion blur, and the synthetic-to-real gap.",
      ] },
      { t: "h", text: "5. Single-image super-resolution (SR)" },
      { t: "p", text: "**Problem:** given a low-resolution image $y = D(x)$ (blur + downsample by $r$, maybe noise and compression), predict the high-resolution $x$. It is **ill-posed**: many HR images shrink to the same LR image, so the model must pick a plausible one. **Architectures:** SRCNN (Dong, Loy, He and Tang, 2014) upsampled with bicubic first, then refined with three conv layers — all work at high resolution. ESPCN (Shi et al., 2016, *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*) computes features at **low** resolution. Its last convolution outputs $C r^2$ channels, and a **pixel shuffle** rearranges them into a $C\\times rH\\times rW$ image — $r^2$ times fewer operations in the body. Later models (VDSR, EDSR) are deeper residual networks; SRGAN (Ledig et al., 2017) adds an adversarial loss." },
      { t: "list", items: [
        "**Pixel losses** (L2 or L1) maximise PSNR. But the average of many plausible HR images is blurry, so textures come out too smooth.",
        "**Perceptual loss** (Johnson, Alahi and Fei-Fei, 2016): distance between feature maps of a fixed pretrained network (e.g. VGG) for output and target. It rewards matching edges and textures rather than exact pixels.",
        "**Adversarial loss**: a discriminator pushes outputs toward realistic natural images. Sharper, but lower PSNR and possibly invented detail. See [GANs](/learn/m14/gans) and, for state-of-the-art generative SR, [Diffusion Models](/learn/m14/diffusion).",
        "**Metrics:** PSNR and SSIM (formulas in the Math section). For GAN/diffusion SR, use perceptual metrics and human studies, because PSNR punishes plausible-but-not-identical detail.",
        "**Pitfalls:** training on bicubic-downsampled images but testing on real camera images (different degradation). PSNR on RGB vs Y channel, with or without border cropping — numbers only compare under the same protocol. Invented detail is dangerous in medical or forensic use.",
      ] },
      { t: "h", text: "6. Image captioning" },
      { t: "p", text: "**Problem:** write a sentence describing the image. **Architecture:** an encoder–decoder. Show and Tell (Vinyals et al., 2015) fed one CNN feature vector as the first input of an LSTM language model. Show, Attend and Tell (Xu et al., 2015) kept the **spatial grid** of CNN features (e.g. 14×14 = 196 region vectors) and, for every word, computed attention weights over the regions. This is the additive attention of [From Seq2Seq to Attention](/learn/m10/attention-intro), over image regions instead of source words. Later systems attended over detected object regions (Anderson et al., 2018) and used Transformer decoders with cross-attention ([Transformer Architecture](/learn/m11/transformer-architecture)). Modern vision–language models connect an image encoder to a large language model ([Multimodal Models](/learn/m15/multimodal))." },
      { t: "list", items: [
        "**Training:** teacher forcing — the decoder is given the true previous word and trained with token-level cross-entropy. A second stage often optimises the metric (CIDEr) directly with reinforcement learning, e.g. self-critical sequence training (Rennie et al., 2017).",
        "**Inference:** greedy or beam search, feeding back the model's own words. This mismatch with teacher forcing is called **exposure bias**.",
        "**Metrics:** n-gram overlap with several human reference captions — BLEU, METEOR, CIDEr (consensus-weighted TF-IDF n-grams), SPICE (scene-graph overlap).",
        "**Pitfalls:** bland captions ('a man standing in a room') score well; object hallucination (naming things that are not there); dataset biases (gender, context).",
      ] },
      { t: "h", text: "7. Image generation (pointer)" },
      { t: "p", text: "Creating new images (text-to-image, inpainting, image-to-image translation and more) has its own module: [Autoencoders](/learn/m14/autoencoders), [VAEs](/learn/m14/vae), [GANs](/learn/m14/gans) and [Diffusion Models](/learn/m14/diffusion). SR and captioning are its closest neighbours: SR is generation given a small image, and text-to-image is captioning run backwards." },
    ],
    math: [
      { t: "h", text: "PSNR" },
      { t: "math", tex: String.raw`\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat x_i)^2, \qquad \mathrm{PSNR} = 10\log_{10}\frac{L^2}{\mathrm{MSE}}\ \text{dB}`, caption: "$x$ is the reference image, $\\hat x$ the estimate, $N$ the number of pixel values, $L$ the data range (1 for images in [0,1], 255 for 8-bit)." },
      { t: "p", text: "Worked: on [0,1] images, MSE $0.01$ gives $10\\log_{10}(1/0.01) = 20$ dB. On 8-bit images, MSE 100 (RMS error 10 grey levels) gives $10\\log_{10}(65025/100) = 28.13$ dB. Halving the RMS error to 5 (MSE 25) adds $10\\log_{10}4 = 6.02$ dB, giving $34.15$ dB. PSNR depends only on MSE, so all distortions with the same MSE get the same PSNR." },
      { t: "h", text: "SSIM" },
      { t: "p", text: "SSIM (Wang, Bovik, Sheikh and Simoncelli, 2004, *Image quality assessment: from error visibility to structural similarity*) compares, in a sliding window, local means (brightness), standard deviations (contrast) and correlation (structure):" },
      { t: "math", tex: String.raw`\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},\qquad C_1 = (K_1L)^2,\ C_2 = (K_2L)^2`, caption: "$\\mu$: local (Gaussian-weighted) means; $\\sigma_x^2, \\sigma_y^2$: local variances; $\\sigma_{xy}$: local covariance; $K_1 = 0.01$, $K_2 = 0.03$; standard window 11×11 Gaussian with $\\sigma = 1.5$. SSIM is the mean over all window positions and equals 1 only for identical images." },
      { t: "p", text: "Worked example with one window over four pixels, $x = (0.2, 0.4, 0.6, 0.8)$, $y = (0.3, 0.4, 0.5, 0.9)$, $L = 1$ ($C_1 = 10^{-4}$, $C_2 = 9\\times10^{-4}$), using population (divide-by-$N$) moments:" },
      { t: "steps", items: [
        { title: "Means and moments", tex: String.raw`\mu_x = 0.5,\ \ \mu_y = 0.525,\ \ \sigma_x^2 = 0.05,\ \ \sigma_y^2 = 0.051875,\ \ \sigma_{xy} = 0.0475` },
        { title: "Luminance term", tex: String.raw`\frac{2(0.5)(0.525) + 10^{-4}}{0.25 + 0.275625 + 10^{-4}} = \frac{0.5251}{0.525725} = 0.99881` },
        { title: "Contrast–structure term", tex: String.raw`\frac{2(0.0475) + 9\times10^{-4}}{0.05 + 0.051875 + 9\times10^{-4}} = \frac{0.0959}{0.102775} = 0.93311` },
        { title: "SSIM and PSNR", tex: String.raw`\mathrm{SSIM} = 0.99881\times0.93311 = 0.93200,\qquad \mathrm{MSE} = 0.0075,\ \mathrm{PSNR} = 21.25\ \text{dB}` },
      ] },
      { t: "h", text: "Sub-pixel convolution (pixel shuffle)" },
      { t: "p", text: "A tensor $T$ of shape $(C r^2, H, W)$ is rearranged into $S$ of shape $(C, rH, rW)$ by" },
      { t: "math", tex: String.raw`S[c,\ hr + i,\ wr + j] = T[c\,r^2 + i\,r + j,\ h,\ w],\qquad 0 \le i, j < r` },
      { t: "p", text: "In words: the $r^2$ channels for output channel $c$ fill the $r\\times r$ block of output pixels at each input position. It is a pure reshape and transpose — no arithmetic; the convolution before it is the learned part. Shi et al. showed it is equivalent to a stride-$r$ transposed convolution, but cheaper and free of the uneven-overlap checkerboard pattern. Example ($r = 2$): output pixel $(0, 1)$ of channel 0 comes from input channel $0\\cdot4 + 0\\cdot2 + 1 = 1$ at position $(0,0)$ — the program below prints exactly this." },
      { t: "h", text: "Perceptual loss" },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{perc}}(\hat x, x) = \sum_{j \in J}\frac{1}{C_jH_jW_j}\big\lVert \phi_j(\hat x) - \phi_j(x)\big\rVert_2^2`, caption: "$\\phi_j$: activations of layer $j$ of a fixed pretrained network, shape $C_j\\times H_j\\times W_j$; $J$: the chosen layers. Compare features, not pixels." },
      { t: "h", text: "Keypoint heatmaps and sub-pixel decoding" },
      { t: "p", text: "The target heatmap for a joint at $(k_x, k_y)$ is $H(p) = \\exp\\big(-\\lVert p - k\\rVert^2/(2\\sigma^2)\\big)$ over heatmap pixels $p$. Its log is a parabola in each coordinate. So a parabola through the log-values at the peak pixel $i$ and its two neighbours recovers the centre exactly:" },
      { t: "math", tex: String.raw`\hat k_x = i + \frac{1}{2}\,\frac{\ell_{i-1} - \ell_{i+1}}{\ell_{i-1} - 2\ell_i + \ell_{i+1}},\qquad \ell_j = \log H(j)` },
      { t: "p", text: "Worked: $k_x = 4.3$, $\\sigma = 1$. Values at $x = 3, 4, 5$: $0.42956, 0.95600, 0.78270$; logs $-0.845, -0.045, -0.245$. Offset $= \\tfrac12(-0.845 + 0.245)/(-0.845 + 0.090 - 0.245) = \\tfrac12(-0.6)/(-1.0) = 0.3$, so $\\hat k_x = 4.3$ exactly. Argmax says 4 (error 0.3 heatmap px = 1.2 input px at stride 4). **Soft-argmax** $\\hat k = \\sum_p \\mathrm{softmax}(\\beta H)_p\\, p$ is differentiable (handy for end-to-end training) but pulled toward the centre by the low-valued background." },
      { t: "p", text: "**Evaluation.** PCK@$\\alpha$ counts a joint as correct if $\\lVert \\hat k - k\\rVert < \\alpha\\, d_{\\text{ref}}$. COCO's **object keypoint similarity** for one person is" },
      { t: "math", tex: String.raw`\mathrm{OKS} = \frac{\sum_i \exp\!\big(-d_i^2 / (2 s^2 \kappa_i^2)\big)\,\delta(v_i > 0)}{\sum_i \delta(v_i > 0)}`, caption: "$d_i$: distance between predicted and true keypoint $i$; $s^2$: the person's area; $\\kappa_i$: a per-keypoint constant for how much labellers disagree; $v_i$: visibility flag. OKS plays the role of IoU when computing AP." },
      { t: "p", text: "Example: $d = 4$ px, $s^2 = 1600$, $\\kappa = 0.1$ gives $\\exp(-16/(2\\cdot1600\\cdot0.01)) = \\exp(-0.5) = 0.607$." },
      { t: "h", text: "Optical flow: brightness constancy and EPE" },
      { t: "math", tex: String.raw`I(x + u,\ y + v,\ t + 1) = I(x, y, t)\ \ \Rightarrow\ \ I_x u + I_y v + I_t \approx 0` , caption: "First-order Taylor expansion; $I_x, I_y$ are spatial derivatives, $I_t$ the time derivative. Lucas–Kanade solves $\\min_{u,v}\\sum_{\\text{window}}(I_xu + I_yv + I_t)^2$, a 2×2 least-squares problem." },
      { t: "math", tex: String.raw`\mathrm{EPE} = \frac{1}{HW}\sum_{p}\sqrt{(\hat u_p - u_p)^2 + (\hat v_p - v_p)^2}` },
      { t: "p", text: "Worked: predicted $(1.2, -0.1)$, true $(1.5, -0.5)$: $\\sqrt{0.3^2 + 0.4^2} = 0.5$ px." },
      { t: "h", text: "ArcFace: the additive angular margin" },
      { t: "math", tex: String.raw`\mathcal{L} = -\log\frac{e^{s\cos(\theta_{y} + m)}}{e^{s\cos(\theta_{y} + m)} + \sum_{j\ne y} e^{s\cos\theta_j}},\qquad \cos\theta_j = \frac{W_j^\top f(x)}{\lVert W_j\rVert\,\lVert f(x)\rVert}`, caption: "$W_j$: learned centre of training identity $j$; $\\theta_j$: angle between the embedding and that centre; $s$: scale; $m$: angular margin." },
      { t: "p", text: "Worked: $s = 16$, $m = 0.3$; the embedding has $\\cos\\theta_y = 0.8$ to its own centre and $0.6$ to four other centres. Without margin the true logit is $12.8$ vs $9.6$ for the others, so $p_y = 0.860$. With margin, $\\theta_y = 0.6435 \\to 0.9435$ rad, $\\cos = 0.5870$, logit $9.391$ and $p_y = 0.169$. The loss stays high until the embedding is a full $0.3$ rad closer to its centre than plain softmax demands — packing identities tightly with angular gaps between them." },
      { t: "h", text: "Verification metrics" },
      { t: "p", text: "For a similarity threshold $\\tau$: $\\mathrm{TAR}(\\tau) = P(s > \\tau \\mid \\text{same})$, $\\mathrm{FAR}(\\tau) = P(s > \\tau \\mid \\text{different})$. Sweeping $\\tau$ traces the ROC curve. Its area is the chance that a random genuine pair scores higher than a random impostor pair. Real systems fix FAR (e.g. $10^{-4}$, one false unlock in 10,000 impostor attempts) and report the TAR there." },
      { t: "h", text: "Captioning loss" },
      { t: "math", tex: String.raw`\mathcal{L} = -\sum_{t=1}^{T}\log p_\theta(w_t \mid w_{<t}, I),\qquad \alpha_{t,r} = \mathrm{softmax}_r\big(v^\top\tanh(W_f a_r + W_h h_{t-1})\big),\ \ c_t = \sum_r \alpha_{t,r}a_r`, caption: "$a_r$: CNN feature of region $r$; $h_{t-1}$: decoder state; $\\alpha_{t,r}$: attention weight on region $r$ for word $t$; $c_t$: context vector fed to the decoder." },
    ],
    visual: [
      { t: "p", text: "Face recognition is metric learning. Here a linear map is trained in your browser to pull same-class points together and push classes apart." },
      { t: "sim", id: "metric-learning", caption: "Contrastive and triplet embeddings in 2-D — a face embedding network in miniature." },
      { t: "list", items: [
        "Train with the triplet loss and watch clusters form. Then raise the margin: clusters get tighter and further apart — what ArcFace's angular margin does on the sphere.",
        "A new, never-trained class lands somewhere in the space; verification works only if the map generalises. The open-set experiment below measures this.",
      ] },
      { t: "sim", id: "detection", caption: "Anchors, IoU and non-max suppression — the machinery of face detectors such as RetinaFace." },
      { t: "list", items: [
        "Lower the NMS threshold and watch overlapping boxes merge: in a crowd photo this deletes faces that really do overlap.",
      ] },
      { t: "sim", id: "seq2seq-attention", caption: "Additive attention in a trained encoder–decoder: swap the source tokens for image regions and this is the Show, Attend and Tell captioner." },
      { t: "list", items: [
        "Step through decoding: each output step puts its attention on a few source positions. In captioning, the attention map over the region grid can be upsampled onto the image to show where the model 'looked' for each word.",
      ] },
      { t: "sim", id: "conv-variants", caption: "Transposed convolution — the alternative to pixel shuffle for upsampling in SR, pose and flow decoders." },
      { t: "list", items: [
        "Set stride 2 with kernel 3: the stamped kernels overlap unevenly, making the checkerboard pattern that sub-pixel convolution avoids.",
      ] },
    ],
    scratch: [
      { t: "p", text: "PSNR and SSIM from scratch with standard SSIM settings, tested on distortions of a 64×64 image; heatmap encoding and three decoders; endpoint error; and a Lucas–Kanade flow estimate. scikit-image is not installed in the test environment, so we check properties instead: SSIM of an image with itself is exactly 1, SSIM is symmetric, and PSNR of a known MSE matches the formula." },
      { t: "code", lang: "python", title: "vision_metrics_numpy.py", code: VT_METRICS, output: VT_METRICS_OUT },
      { t: "list", items: [
        "**Same PSNR, different SSIM.** Noise, blur and a brightness shift with (almost) the same MSE all give ≈20.0 dB, but SSIM ranges from 0.710 (noise) to 0.822 (blur). On this blocky, high-contrast image, blur keeps more local structure than noise. The +0.1 shift is punished by the luminance term in the dark background, where $\\mu_x \\approx 0$. The point: PSNR measures only how much error; SSIM measures *what kind*.",
        "**Sub-pixel decoding matters.** Plain argmax has mean error 0.374 heatmap px (≈1.5 input px at stride 4). The log-parabola fit cuts it to 0.017 even with noise. Soft-argmax (0.146) is biased by the noise floor across the whole map.",
        "**EPE** of a noisy prediction (0.371 px) vs a zero-flow baseline (1.581 px = the true flow size) — always report a trivial baseline for scale.",
        "**Lucas–Kanade** recovers a sub-pixel global shift (0.6, −0.3) with 0.012 px EPE. Brightness constancy + linearisation work for small motion on smooth images, and fail for large motion — which is why learned methods use multi-scale correlation volumes.",
        "To compare with scikit-image when available: `skimage.metrics.structural_similarity(x, y, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` uses the same settings (not run here).",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch: super-resolution", blocks: [
          { t: "p", text: "Part 1 checks that `nn.PixelShuffle` is a reshape/permute (the Math section formula) and that `PixelUnshuffle` undoes it. Part 2 trains a 10.7k-parameter ESPCN-style network for ×2 super-resolution of sklearn digits (HR 8×8, LR 4×4 made by 2×2 average pooling). It predicts a correction on top of bicubic upsampling; we compare PSNR on 397 held-out digits with nearest, bilinear and bicubic interpolation." },
          { t: "code", lang: "python", title: "superres_pixelshuffle.py", code: VT_SR, output: VT_SR_OUT },
          { t: "list", items: [
            "The learned model beats bicubic by about 5.4 dB (18.89 vs 13.52 dB). It has learned what digits look like — the whole point of learned SR. Interpolation only knows smoothness.",
            "Bilinear scores below nearest here: at 4×4 → 8×8 the images are tiny and high-contrast, and bilinear smears ink into the background. Classical rankings depend on the content.",
            "The whole body runs at 4×4; only the final shuffle makes 8×8 — ESPCN's efficiency argument.",
            "Absolute PSNR is low because the task is extreme (each LR pixel replaces four HR pixels of a 64-pixel digit); papers report ~30+ dB on smoother natural photos at ×2.",
          ] },
        ] },
        { label: "PyTorch: face-verification ROC", blocks: [
          { t: "p", text: "A face-recognition pipeline in miniature, with digits as 'identities'. An MLP maps each image to a 32-D embedding, trained with an ArcFace head ($m = 0$, a normalised-softmax 'cosine' classifier, vs $m = 0.3$). We score every pair of test images: cosine similarity, an ROC curve from a threshold sweep (checked against `sklearn.metrics.roc_auc_score`), TAR at FAR = 1% and 0.1%, and rank-1 identification with one gallery image per identity. Protocol A tests on new images of identities seen in training. Protocol B — the real face-recognition setting — trains on identities 0–6 and tests on **unseen** identities 7–9." },
          { t: "code", lang: "python", title: "verification_roc.py", code: VT_VERIF, output: VT_VERIF_OUT },
          { t: "list", items: [
            "**Closed set (A):** the embedding clearly beats raw pixels (AUC 0.967 vs 0.862), and the angular margin helps more: TAR@FAR=1% rises from 0.736 to 0.837, rank-1 from 0.818 to 0.852.",
            "**Open set (B):** trained on only 7 identities, the embedding does **not** generalise to new ones — AUC 0.814 / 0.803 vs 0.819 for raw pixels, and the margin even hurts rank-1. The network learned to separate *those 7 digits*, not a general sense of digit identity.",
            "So real face recognition trains on tens of thousands of identities and tests on different people: training must cover variation *between* identities, not just within them. Protocol A numbers would hugely overstate real-world performance.",
          ] },
        ] },
        { label: "PyTorch: captioning shapes", blocks: [
          { t: "p", text: "The data flow of an attention captioner (encoder grid → additive attention → GRU decoder) with tiny random weights, so it runs instantly. It shows the tensor shapes, that attention weights form a distribution over the 64 regions, the teacher-forced cross-entropy (near the uniform-guess value $\\ln 12$ because the model is untrained), and greedy decoding feeding back its own outputs. An untrained model repeats one token; training on (image, caption) pairs is what makes attention meaningful." },
          { t: "code", lang: "python", title: "caption_attention_shapes.py", code: VT_CAPTION, output: VT_CAPTION_OUT },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "TensorFlow has the same pieces built in: `tf.nn.depth_to_space` is pixel shuffle for NHWC tensors, and `tf.image.psnr` / `tf.image.ssim` compute the metrics. SSIM defaults to an 11×11 Gaussian window with $\\sigma = 1.5$, and images must be at least window-sized — hence `filter_size=3` for 8×8 images." },
          { t: "code", lang: "python", title: "espcn_keras.py (not executed in CI)", code: VT_KERAS, output: "Prints the SR output shape (4, 8, 8, 1), four PSNR values for the untrained model, and four SSIM values of 1.0 for identical images." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Task / approach", "Representative models", "Pros", "Cons", "Use cases"], rows: [
        ["Face detection: cascade", "Viola–Jones, MTCNN", "Fast on CPU; landmarks (MTCNN)", "Weaker on small/hidden faces", "Embedded cameras, legacy systems"],
        ["Face detection: single-stage", "RetinaFace-style detectors", "Accurate on tiny faces; landmarks", "Heavier backbone", "Phone cameras, photo apps, face pipelines"],
        ["Face embedding: triplet", "FaceNet", "Optimises distances directly", "Needs hard-triplet mining; slow to converge", "Research, small identity sets"],
        ["Face embedding: margin softmax", "SphereFace, CosFace, ArcFace", "Stable training, state of the art", "Classifier weights grow with #identities", "Large-scale face recognition"],
        ["Pose: direct regression", "DeepPose", "Simple; no decoding step", "Loses spatial structure; less accurate", "Low-resolution or fast settings"],
        ["Pose: top-down heatmaps", "Stacked Hourglass, HRNet", "Most accurate per person", "Cost ∝ number of people; depends on detector", "Sports analytics, fitness, motion capture"],
        ["Pose: bottom-up", "OpenPose (PAFs)", "Constant cost in crowds", "Grouping errors; lower per-person accuracy", "Crowded scenes, real time"],
        ["Flow: variational", "Horn–Schunck, Lucas–Kanade", "No training data; interpretable", "Small motions only; slow; aperture problem", "Classical tracking, video stabilisation"],
        ["Flow: learned", "FlowNet, PWC-Net, RAFT", "Accurate, handles large motion", "Needs synthetic training data; memory (RAFT's all-pairs volume)", "Video interpolation, action recognition, driving"],
        ["SR: pixel loss", "SRCNN, ESPCN, EDSR", "High PSNR; stable", "Over-smooth textures", "Measurement, medical, video upscaling"],
        ["SR: perceptual / GAN / diffusion", "SRGAN, ESRGAN, diffusion SR", "Sharp, realistic detail", "Invented detail; lower PSNR", "Photo enhancement, games, media restoration"],
        ["Captioning: CNN + RNN (+attention)", "Show and Tell, Show Attend and Tell", "Simple; attention you can inspect", "Limited fluency; exposure bias", "Teaching, small systems"],
        ["Captioning: Transformer / VLM", "Region features + Transformer; image encoder + LLM", "Fluent, open vocabulary", "Large; hallucination", "Accessibility (alt text), search, assistants"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Faces**: phone unlock (verification at very low FAR), grouping photo libraries (clustering embeddings), access control, passport matching; camera autofocus uses face detection.",
        "**Pose**: fitness apps counting reps, sports analytics, sign-language research, marker-free motion capture, fall detection.",
        "**Optical flow**: frame interpolation (slow motion), video compression and stabilisation, action recognition (two-stream networks take flow as a second input), driver assistance and robot odometry.",
        "**Super-resolution**: upscaling old film and video, rendering games at lower resolution then upscaling, satellite images, phone camera zoom. Be careful in medical and forensic use because of invented detail.",
        "**Captioning**: automatic alt text for accessibility, image search indexing, visual question answering and assistants built on vision–language models.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Dense outputs suit convolutions**: heatmaps, flow fields and HR images keep the input's spatial layout. That is why image-to-image prediction beats coordinate regression for these tasks.",
        "**Labels are expensive or impossible**: dense flow cannot be hand-labelled (hence synthetic data and a sim-to-real gap); keypoints and captions are slow to label; SR gets free labels by downsampling — but the fake degradation may not match real cameras.",
        "**Metrics are imperfect stand-ins**: PSNR prefers blurry SR; BLEU/CIDEr reward bland captions; AUC hides the low-FAR region that matters for security; PCK hides how wrong the misses are.",
        "**Open-set generalisation** (faces) needs data covering many identities — the experiment above shows a model failing to generalise from 7.",
        "**Compute**: RAFT's all-pairs correlation takes $O((HW)^2)$ memory at the finest level before pooling; top-down pose scales with the number of people; high-resolution SR is memory-bound.",
        "**Societal risk**: face recognition raises privacy, surveillance and bias concerns; SR and captioning can invent detail or content that looks trustworthy.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Testing face recognition on people seen in training.** That is closed-set classification and overstates performance (protocol A vs B above). Fix: keep training and test identities separate.",
        "**Reporting AUC or accuracy for a security application.** What matters is TAR at the deployment FAR (e.g. $10^{-4}$ or lower). Fix: report TAR@FAR at the relevant operating points.",
        "**Skipping face alignment**, or aligning differently in training and testing. Fix: same detector, landmarks and template for both.",
        "**Flipping images for pose augmentation without swapping left/right joint labels** — the network learns a left wrist can be on either side. Fix: swap the channels of symmetric keypoints when flipping.",
        "**Whole-pixel argmax on low-resolution heatmaps.** Up to half a heatmap pixel of rounding error (×stride in the image). Fix: parabola/Taylor refinement or soft-argmax.",
        "**Comparing SR PSNR across papers with different protocols** (RGB vs Y channel, border cropping, the bicubic kernel used to make LR). Fix: state the protocol and recompute baselines under it.",
        "**Using only a pixel loss and being surprised by blurry SR.** The MSE-optimal output is the average of all plausible answers. Fix: add perceptual/adversarial terms if perceived quality matters — and accept lower PSNR.",
        "**Upsampling with stride-2 transposed convolutions whose kernel size is not divisible by the stride**, causing checkerboard artefacts. Fix: pixel shuffle or resize-then-convolve.",
        "**Optical flow sign and channel conventions**: flow from frame 1 to 2 vs 2 to 1, $(u, v)$ vs $(dy, dx)$, pixels at which resolution. Fix: warp one frame with the flow and compare with the other.",
        "**Evaluating captioning with teacher forcing.** Teacher-forced loss says little about generated captions. Fix: evaluate generated text (beam search) against several references.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between face verification and face identification?", a: "Verification is 1:1 — are these two faces the same person? (threshold on similarity). Identification is 1:N — which enrolled person is this? (nearest neighbour in a gallery), possibly rejecting if no one is similar enough (open-set identification)." },
      { t: "qa", level: "beginner", q: "Why do pose estimators predict heatmaps instead of coordinates?", a: "A heatmap lines up with the image, so a fully convolutional network can predict it with translation equivariance and a dense training signal around the joint. Regressing coordinates forces a dense layer to turn spatial features into numbers, and it generalises worse." },
      { t: "qa", level: "beginner", q: "What do PSNR and SSIM measure?", a: "PSNR is a log-scaled inverse of the mean squared error relative to the data range ($10\\log_{10}(L^2/\\mathrm{MSE})$ dB). SSIM compares local means, contrasts and correlation in sliding windows, so it reacts more to structural damage (noise, blur), and equals 1 only for identical images." },
      { t: "qa", level: "beginner", q: "What is optical flow and what is endpoint error?", a: "Optical flow is a per-pixel 2-D movement field between two frames. Endpoint error is the distance between predicted and true movement vectors, averaged over pixels." },
      { t: "qa", level: "intermediate", q: "Explain pixel shuffle and why ESPCN uses it.", a: "A conv layer outputs $Cr^2$ channels at low resolution; pixel shuffle rearranges each group of $r^2$ channels into an $r\\times r$ block of output pixels, giving $C\\times rH\\times rW$. All convolutions then run at low resolution ($r^2$ fewer operations than upsampling first, as SRCNN did), and it avoids the uneven overlap of strided transposed convolutions." },
      { t: "qa", level: "intermediate", q: "Why do pixel-loss super-resolution models produce blurry results?", a: "Many HR images fit one LR input. The L2-optimal prediction is their average (conditional mean), which washes out fine detail. Perceptual losses (distances in feature space) and adversarial losses push toward one sharp, plausible answer instead, at the cost of PSNR." },
      { t: "qa", level: "intermediate", q: "Describe top-down vs bottom-up pose estimation.", a: "Top-down detects people, crops each and predicts keypoint heatmaps per crop — accurate, but cost grows with the number of people and it depends on the detector. Bottom-up predicts all keypoints at once plus grouping cues (OpenPose's part affinity fields) and assembles people — cost hardly depends on crowd size, but grouping can fail." },
      { t: "qa", level: "intermediate", q: "How does an attention-based captioning model work?", a: "A CNN encoder produces a grid of region features. At each step, a recurrent or Transformer decoder computes attention weights over the regions from its current state, takes their weighted sum as a context vector, and predicts the next word from the context, previous word and state. It is trained with teacher-forced cross-entropy and decoded with beam search." },
      { t: "qa", level: "advanced", q: "What does ArcFace's additive angular margin do geometrically, and why is it preferred over triplet loss at scale?", a: "Features and class weights are normalised, so logits are $s\\cos\\theta_j$. Replacing $\\theta_y$ by $\\theta_y + m$ for the true class forces each embedding to be at least $m$ radians closer to its identity centre than the decision boundary needs. Identities become compact, with angular gaps between them. Every softmax compares all identities, with no triplet mining, so it trains more stably and efficiently on millions of images than triplet loss, whose useful triplets become rare." },
      { t: "qa", level: "advanced", q: "How does RAFT estimate flow, and why does it iterate?", a: "It extracts features for both frames and builds a 4-D correlation volume of dot products between all pixel pairs (pooled to several scales). It keeps one flow field at 1/8 resolution that a convolutional GRU updates repeatedly: each iteration looks up correlations around the current match and predicts a correction. Iterating refines large and small motions with shared weights, like an optimiser. The loss supervises every iteration with growing weights, and the final flow is upsampled with a learned convex combination." },
      { t: "qa", level: "advanced", q: "Your face-verification model has AUC 0.98 on a validation split but performs poorly in deployment. What would you check?", a: "Whether validation identities overlap with training ones (closed-set leakage — our experiment shows AUC can fall to the pixel-baseline level on unseen identities); whether you looked at the deployment FAR (AUC hides low-FAR performance); consistent alignment and preprocessing; domain shift (camera, lighting, demographics); threshold calibration on deployment-like data." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "8-bit images, MSE = 100. What is the PSNR in dB? (2 decimals)", answer: 28.13, tolerance: 0.01, explain: "$10\\log_{10}(255^2/100) = 10\\log_{10}650.25 = 28.13$ dB." },
        { kind: "numeric", q: "A pixel-shuffle layer upsamples by $r = 3$ to produce a 3-channel RGB image. How many channels must the convolution before it output?", answer: 27, tolerance: 0, explain: "$C r^2 = 3\\cdot9 = 27$." },
        { kind: "numeric", q: "Predicted flow $(2.0, 1.0)$, true flow $(2.6, 1.8)$. What is the endpoint error in pixels? (2 decimals)", answer: 1.0, tolerance: 0.01, explain: "$\\sqrt{0.6^2 + 0.8^2} = 1.0$." },
        { kind: "numeric", q: "Heatmap log-values at the peak pixel $i = 7$ and its neighbours are $\\ell_6 = -0.5$, $\\ell_7 = 0$, $\\ell_8 = -1.5$. What is the refined coordinate? (2 decimals)", answer: 6.75, tolerance: 0.01, explain: "Offset $= \\tfrac12(-0.5 - (-1.5))/(-0.5 - 0 - 1.5) = \\tfrac12(1)/(-2) = -0.25$, so $7 - 0.25 = 6.75$ (the left neighbour is higher, so the centre is left of the peak)." },
        { q: "Which split protocol correctly evaluates a face-recognition model?", options: ["Random split of images, same identities in train and test", "Disjoint identities between training and test", "Test on the training images", "Any split, as long as AUC is reported"], answer: 1, explain: "In real use the system meets people never seen in training; only separate identities measure that (protocol B in the experiment)." },
        { q: "Two SR outputs have equal PSNR. One is blurred, the other noisy. What can SSIM tell you that PSNR cannot?", options: ["Nothing — SSIM is a monotone function of MSE", "Which distortion better preserves local structure, since SSIM compares local means, contrasts and correlations", "The resolution of the image", "Whether a GAN was used"], answer: 1, explain: "PSNR depends only on MSE; SSIM separates brightness, contrast and structure (0.710 vs 0.822 in our run)." },
        { q: "What problem do OpenPose's part affinity fields solve?", options: ["Sub-pixel localisation", "Assigning detected keypoints to the correct person in bottom-up pose estimation", "Detecting faces", "Occlusion of the whole body"], answer: 1, explain: "PAFs encode limb direction fields; summing them along candidate joint pairs scores which joints belong together." },
        { q: "Why is optical flow ground truth usually synthetic?", options: ["Real videos have no motion", "Dense per-pixel correspondences cannot practically be labelled by hand", "Synthetic images have higher resolution", "Flow networks cannot process real images"], answer: 1, explain: "Hence rendered datasets like FlyingChairs and Sintel, and sparse LiDAR-based labels (KITTI)." },
      ] },
      { t: "exercise", title: "ArcFace logits by hand", prompt: "An embedding has cosine 0.9 to its own identity centre and 0.5 to each of two other centres. With $s = 10$ and $m = 0.2$, compute the true-class logit and its softmax probability, with and without the margin.", hint: "$\\theta_y = \\arccos 0.9$; add $m$ in radians before taking the cosine.", solution: [
        { t: "steps", items: [
          { title: "Without margin", tex: String.raw`z_y = 10(0.9) = 9,\ z_j = 5;\quad p_y = \frac{e^{9}}{e^{9} + 2e^{5}} = \frac{1}{1 + 2e^{-4}} = 0.9647` },
          { title: "Angle with margin", tex: String.raw`\theta_y = \arccos 0.9 = 0.4510,\quad \cos(0.6510) = 0.7955,\quad z_y = 7.955` },
          { title: "Probability with margin", tex: String.raw`p_y = \frac{1}{1 + 2e^{5 - 7.955}} = \frac{1}{1 + 2(0.05210)} = 0.9056` },
        ] },
        { t: "p", text: "The margin lowers the confidence of an example that is already right, so its loss ($-\\ln 0.9056 = 0.099$ vs $0.036$) keeps pushing the embedding toward its centre." },
      ] },
      { t: "exercise", title: "Train a ×2 SR model with an L1 loss and report SSIM", prompt: "Change `superres_pixelshuffle.py` to (a) use `F.l1_loss` instead of MSE and (b) also report mean SSIM on the test digits with a 3×3 uniform window (an 11×11 window does not fit 8×8 images). Does the L1 model reach higher or lower PSNR than the L2 model? Why might it?", hint: "Implement SSIM with `F.avg_pool2d(x, 3, 1)` for the local means, variances and covariance.", solution: [
        { t: "code", lang: "python", title: "ssim_torch sketch", code: String.raw`def ssim_torch(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    mu_a, mu_b = F.avg_pool2d(a, 3, 1), F.avg_pool2d(b, 3, 1)
    var_a = F.avg_pool2d(a * a, 3, 1) - mu_a ** 2
    var_b = F.avg_pool2d(b * b, 3, 1) - mu_b ** 2
    cov = F.avg_pool2d(a * b, 3, 1) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))
    return s.mean().item()

# in the training loop: loss = F.l1_loss(net(lr[b]), hr[b])`, output: "" },
        { t: "p", text: "PSNR is a function of MSE, so an MSE-trained model optimises it directly. An L1-trained model usually gets slightly *lower* PSNR, since it was not trained for that metric. Yet many SR papers (EDSR, for instance) train with L1 because it converges more reliably and gives sharper edges — L1's optimum is the conditional median, not the mean. Measure both; on 8×8 digits the gap may be a few tenths of a dB." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Each task is defined by its output: boxes + landmarks (face detection), unit embeddings (recognition), heatmaps/PAFs (pose), 2-channel flow fields, $r\\times$ larger images (SR), token sequences (captioning).",
        "Face recognition = detect → align → embed → cosine similarity; trained with margin softmax (ArcFace: $s\\cos(\\theta_y + m)$) or triplet loss; tested on unseen people with TAR@FAR and rank-1.",
        "Pose: Gaussian heatmaps + sub-pixel decoding ($\\hat k = i + \\tfrac12\\frac{\\ell_{i-1}-\\ell_{i+1}}{\\ell_{i-1}-2\\ell_i+\\ell_{i+1}}$); top-down vs bottom-up (PAFs); PCK and OKS.",
        "Flow: brightness constancy $I_xu + I_yv + I_t = 0$, aperture problem; FlowNet → PWC-Net → RAFT; EPE.",
        "SR: pixel shuffle $S[c, hr+i, wr+j] = T[cr^2 + ir + j, h, w]$; pixel vs perceptual vs adversarial losses; PSNR $= 10\\log_{10}(L^2/\\mathrm{MSE})$, SSIM.",
        "Captioning: CNN grid encoder + attention decoder, teacher-forced cross-entropy, beam search, BLEU/CIDEr.",
        "Measured here: learned ×2 SR 18.89 dB vs bicubic 13.52 dB; closed-set verification AUC 0.979 (margin) vs 0.862 (pixels), but no gain on unseen identities when trained on only 7.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Object Detection](/learn/m12/object-detection) and [Segmentation](/learn/m12/segmentation) — the other two dense vision tasks, including the detection machinery face detectors use.",
        "[GANs](/learn/m14/gans) and [Diffusion Models](/learn/m14/diffusion) — the generative models behind perceptual super-resolution and image synthesis.",
        "[Self-Supervised and Contrastive Learning](/learn/m15/self-supervised) — learning embeddings without labels, a broader version of the metric learning used for faces.",
        "[Multimodal Models and CLIP](/learn/m15/multimodal) — joint image–text embeddings, the modern basis of captioning and vision–language assistants.",
      ] },
    ],
  },
};

export default lesson;
