import type { Lesson } from "@/lib/types";

const SEG_NUMPY = String.raw`import numpy as np

# ---------------- 1. semantic metrics from a confusion matrix ----------------
def confusion(pred, gt, n_cls):
    """C[i, j] = number of pixels whose true class is i and predicted class is j."""
    return np.bincount(gt.ravel() * n_cls + pred.ravel(), minlength=n_cls ** 2).reshape(n_cls, n_cls)

def semantic_metrics(C):
    tp = np.diag(C).astype(float)
    pixel_acc = tp.sum() / C.sum()
    iou = tp / (C.sum(0) + C.sum(1) - tp)              # TP / (TP + FP + FN) for every class
    return pixel_acc, iou, np.nanmean(iou)

gt = np.zeros((8, 8), int); gt[1:4, 1:4] = 1; gt[5:7, 4:8] = 2         # background 0, two small objects
pred = np.zeros((8, 8), int); pred[1:4, 1:5] = 1; pred[6:7, 4:8] = 2    # class 1 too wide, class 2 too thin
C = confusion(pred, gt, 3)
acc, iou, miou = semantic_metrics(C)
print("confusion matrix (rows = truth):\n", C)
print(f"pixel accuracy {acc:.4f}  per-class IoU {iou.round(4)}  mIoU {miou:.4f}")
print(f"'all background' baseline: pixel accuracy {np.mean(gt == 0):.4f}, mIoU {semantic_metrics(confusion(np.zeros_like(gt), gt, 3))[2]:.4f}")

# ---------------- 2. panoptic quality ----------------
def panoptic_quality(pred_segs, gt_segs):
    """Segments are (class, boolean mask). A match needs the same class and IoU > 0.5 (then it is unique)."""
    matched_p, ious = set(), []
    for c_g, m_g in gt_segs:
        for k, (c_p, m_p) in enumerate(pred_segs):
            if c_p != c_g or k in matched_p:
                continue
            iou = np.logical_and(m_g, m_p).sum() / np.logical_or(m_g, m_p).sum()
            if iou > 0.5:
                matched_p.add(k); ious.append(iou); break
    tp = len(ious); fp = len(pred_segs) - tp; fn = len(gt_segs) - tp
    sq = np.mean(ious) if tp else 0.0                  # segmentation quality: mean IoU of matches
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)               # recognition quality: an F1 score
    return sq * rq, sq, rq, (tp, fp, fn)

def disc(cx, cy, r, n=32):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

gt_segs = [(1, disc(8, 8, 5)), (1, disc(22, 8, 4)), (2, disc(16, 24, 6))]          # two "cars", one "person"
pred_segs = [(1, disc(9, 8, 5)), (1, disc(22, 9, 2)), (2, disc(16, 24, 6)), (2, disc(4, 26, 3))]
pq, sq, rq, (tp, fp, fn) = panoptic_quality(pred_segs, gt_segs)
print(f"PQ: TP={tp} FP={fp} FN={fn}  SQ={sq:.4f}  RQ={rq:.4f}  PQ=SQ*RQ={pq:.4f}")

# ---------------- 3. RoIAlign from scratch (bilinear sampling, no rounding) ----------------
def bilinear(f, y, x):
    """Sample feature map f (H, W) at real-valued (y, x); pixel centres are at integer coordinates."""
    H, W = f.shape
    if y < -1 or y > H or x < -1 or x > W:
        return 0.0
    y, x = min(max(y, 0), H - 1), min(max(x, 0), W - 1)
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
    ly, lx = y - y0, x - x0
    return ((1 - ly) * (1 - lx) * f[y0, x0] + (1 - ly) * lx * f[y0, x1]
            + ly * (1 - lx) * f[y1, x0] + ly * lx * f[y1, x1])

def roi_align(f, box, out=2, scale=1.0, samples=2, aligned=True):
    """box = (x1, y1, x2, y2) in image coords; scale maps image -> feature coords."""
    off = 0.5 if aligned else 0.0                      # aligned=True: a pixel's centre sits at index + 0.5
    x1, y1, x2, y2 = [v * scale - off for v in box]
    bw, bh = (x2 - x1) / out, (y2 - y1) / out          # bin size: real-valued, never rounded
    res = np.zeros((out, out))
    for i in range(out):
        for j in range(out):
            vals = [bilinear(f, y1 + (i + (a + 0.5) / samples) * bh, x1 + (j + (b + 0.5) / samples) * bw)
                    for a in range(samples) for b in range(samples)]
            res[i, j] = np.mean(vals)                  # average the samples inside the bin
    return res

def roi_pool(f, box, out=2, scale=1.0):
    """Fast R-CNN RoIPool: round the box to the grid, split into integer bins, max-pool each bin."""
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
    h, w = max(y2 - y1 + 1, 1), max(x2 - x1 + 1, 1)
    res = np.zeros((out, out))
    for i in range(out):
        for j in range(out):
            ys = y1 + int(np.floor(i * h / out)); ye = y1 + int(np.ceil((i + 1) * h / out))
            xs = x1 + int(np.floor(j * w / out)); xe = x1 + int(np.ceil((j + 1) * w / out))
            res[i, j] = f[ys:ye, xs:xe].max()
    return res

feat = np.arange(36, dtype=float).reshape(6, 6)        # value = 6*row + col: easy to check by hand
box = (1.3, 0.7, 4.9, 3.6)
print("RoIAlign 2x2:\n", roi_align(feat, box).round(4))
print("RoIPool  2x2:\n", roi_pool(feat, box))

# ---------------- 4. U-Net (Ronneberger et al. 2015) feature-map sizes with unpadded 3x3 convs ----------------
size, skips = 572, []
for level in range(4):                                 # contracting path: conv, conv, 2x2 max-pool
    size -= 4; skips.append(size); size //= 2
size -= 4                                              # bottleneck: two more unpadded convs
print("contracting path sizes before pooling:", skips, " bottleneck:", size)
for level in reversed(range(4)):                       # expanding path: 2x2 up-conv, crop skip, concat, 2 convs
    size *= 2
    crop = (skips[level] - size) // 2
    print(f"  up to {size:3d}: crop skip {skips[level]} -> {size} ({crop} px per side), after convs {size - 4}")
    size -= 4
print("output map:", size, "x", size)

# ---------------- 5. class weights for imbalanced pixel cross-entropy ----------------
freq = np.bincount(gt.ravel(), minlength=3) / gt.size
w_inv = 1 / freq / np.sum(1 / freq) * 3                # inverse frequency, normalised to mean 1
w_med = np.median(freq) / freq                         # median-frequency balancing
print("pixel frequencies", freq.round(4), " inverse-freq weights", w_inv.round(3), " median-freq weights", w_med.round(3))

import torch
from torchvision.ops import roi_align as tv_align, roi_pool as tv_pool
rng = np.random.default_rng(0)
F = rng.normal(size=(20, 20)); worst = 0.0
for _ in range(50):
    x1, y1 = rng.uniform(0, 60, 2); w, h = rng.uniform(4, 30, 2)
    b = (x1, y1, x1 + w, y1 + h)
    mine = roi_align(F, b, out=7, scale=1 / 4, samples=2, aligned=True)
    ref = tv_align(torch.tensor(F)[None, None], [torch.tensor([b])], output_size=7,
                   spatial_scale=1 / 4, sampling_ratio=2, aligned=True)[0, 0].numpy()
    worst = max(worst, np.abs(mine - ref).max())
print(f"RoIAlign vs torchvision over 50 random boxes: max abs diff {worst:.1e}")
ref_pool = tv_pool(torch.tensor(feat)[None, None], [torch.tensor([box], dtype=torch.float64)], output_size=2)[0, 0].numpy()
print("RoIPool matches torchvision:", np.array_equal(roi_pool(feat, box), ref_pool))`;

const SEG_NUMPY_OUT = String.raw`confusion matrix (rows = truth):
 [[44  3  0]
 [ 0  9  0]
 [ 4  0  4]]
pixel accuracy 0.8906  per-class IoU [0.8627 0.75   0.5   ]  mIoU 0.7042
'all background' baseline: pixel accuracy 0.7344, mIoU 0.2448
PQ: TP=2 FP=2 FN=1  SQ=0.8804  RQ=0.5714  PQ=SQ*RQ=0.5031
RoIAlign 2x2:
 [[ 7.25  9.05]
 [15.95 17.75]]
RoIPool  2x2:
 [[15. 17.]
 [27. 29.]]
contracting path sizes before pooling: [568, 280, 136, 64]  bottleneck: 28
  up to  56: crop skip 64 -> 56 (4 px per side), after convs 52
  up to 104: crop skip 136 -> 104 (16 px per side), after convs 100
  up to 200: crop skip 280 -> 200 (40 px per side), after convs 196
  up to 392: crop skip 568 -> 392 (88 px per side), after convs 388
output map: 388 x 388
pixel frequencies [0.7344 0.1406 0.125 ]  inverse-freq weights [0.248 1.295 1.457]  median-freq weights [0.191 1.    1.125]
RoIAlign vs torchvision over 50 random boxes: max abs diff 1.2e-14
RoIPool matches torchvision: True`;

const UNET = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)                               # small model: one thread is fast and reproducible
rng = np.random.default_rng(0)
S, N_CLS = 32, 3                                       # 32x32 images; classes: 0 background, 1 blob, 2 thin line

# ---------------- synthetic data: random-colour blobs and thin lines on a noisy, cluttered background ----------------
def make_sample():
    yy, xx = np.mgrid[0:S, 0:S] + 0.5
    img = rng.uniform(0.0, 0.4, (3, 1, 1)) + rng.normal(0, 0.12, (3, S, S))   # tinted background + pixel noise
    for _ in range(4):                                  # clutter: faint smudges that are NOT objects
        cx, cy, r = rng.uniform(0, S, 2).tolist() + [rng.uniform(2, 6)]
        img[:, (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] += rng.uniform(-0.15, 0.15)
    mask = np.zeros((S, S), np.int64)
    for _ in range(rng.integers(2, 5)):
        cls = int(rng.integers(1, 3))
        if cls == 1:                                    # filled disc or square, radius 3-7 px
            r = rng.uniform(3, 7); cx, cy = rng.uniform(r, S - r, 2)
            m = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2) if rng.random() < 0.5 else                 ((np.abs(xx - cx) < r * 0.85) & (np.abs(yy - cy) < r * 0.85))
        else:                                           # thin straight line, ~1.5 px wide, 12-28 px long
            p = rng.uniform(2, S - 2, 2); ang = rng.uniform(0, np.pi); L = rng.uniform(12, 28)
            d = np.array([np.cos(ang), np.sin(ang)])
            t = np.clip((xx - p[0]) * d[0] + (yy - p[1]) * d[1], 0, L)          # projection onto the segment
            m = (xx - p[0] - t * d[0]) ** 2 + (yy - p[1] - t * d[1]) ** 2 < 0.75 ** 2
        img[:, m] = rng.uniform(0.45, 1.0, (3, 1))     # random colour: the class is in the SHAPE, not the colour
        mask[m] = cls
    return img.astype(np.float32), mask

def dataset(n):
    pairs = [make_sample() for _ in range(n)]
    return torch.tensor(np.stack([p[0] for p in pairs])), torch.tensor(np.stack([p[1] for p in pairs]))

X_tr, Y_tr = dataset(400)
X_te, Y_te = dataset(150)
freq = torch.bincount(Y_tr.flatten(), minlength=N_CLS).float() / Y_tr.numel()
print("class pixel frequencies (train):", [round(f, 3) for f in freq.tolist()])

# ---------------- U-Net (padded convs so sizes halve/double exactly) ----------------
def double_conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

class UNet(nn.Module):
    def __init__(self, skips=True, c=(12, 24, 48)):
        super().__init__()
        self.skips = skips
        self.enc1, self.enc2 = double_conv(3, c[0]), double_conv(c[0], c[1])
        self.bott = double_conv(c[1], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)          # learnable 2x upsampling
        self.dec2 = double_conv(c[1] * (2 if skips else 1), c[1])        # concat doubles the channels
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = double_conv(c[0] * (2 if skips else 1), c[0])
        self.head = nn.Conv2d(c[0], N_CLS, 1)                             # per-pixel class logits
    def forward(self, x):
        e1 = self.enc1(x)                       # (N, 12, 32, 32)  full resolution
        e2 = self.enc2(F.max_pool2d(e1, 2))     # (N, 24, 16, 16)
        b = self.bott(F.max_pool2d(e2, 2))      # (N, 48, 8, 8)    bottleneck
        d2 = self.up2(b)                        # (N, 24, 16, 16)
        d2 = self.dec2(torch.cat([d2, e2], 1) if self.skips else d2)
        d1 = self.up1(d2)                       # (N, 12, 32, 32)
        d1 = self.dec1(torch.cat([d1, e1], 1) if self.skips else d1)
        return self.head(d1)                    # (N, 3, 32, 32)

def miou(pred, gt):
    C = torch.bincount(gt.flatten() * N_CLS + pred.flatten(), minlength=N_CLS ** 2).reshape(N_CLS, N_CLS).double()
    tp = C.diag()
    iou = tp / (C.sum(0) + C.sum(1) - tp)
    return iou, (tp.sum() / C.sum()).item()

def run(skips, epochs=6, bs=32):
    torch.manual_seed(0)
    model = UNet(skips)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X_tr)); total = 0.0
        for k in range(0, len(X_tr), bs):
            b = perm[k:k + bs]
            loss = F.cross_entropy(model(X_tr[b]), Y_tr[b])               # mean over all pixels
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(1)
    iou, acc = miou(pred, Y_te)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"skips={str(skips):5s} params {n_par:,}  last-epoch train loss {total / len(X_tr):.3f}  "
          f"pixel acc {acc:.3f}  IoU per class {[round(v, 3) for v in iou.tolist()]}  mIoU {iou.mean().item():.3f}")
    return pred

pred_skip = run(True)
pred_noskip = run(False)

# where do the two models disagree? measure error rate on boundary pixels vs interior pixels
gt = Y_te
edge = torch.zeros_like(gt, dtype=torch.bool)
edge[:, 1:, :] |= gt[:, 1:, :] != gt[:, :-1, :]; edge[:, :-1, :] |= gt[:, 1:, :] != gt[:, :-1, :]
edge[:, :, 1:] |= gt[:, :, 1:] != gt[:, :, :-1]; edge[:, :, :-1] |= gt[:, :, 1:] != gt[:, :, :-1]
for name, p in [("with skips", pred_skip), ("no skips  ", pred_noskip)]:
    wrong = p != gt
    print(f"{name}: error on boundary pixels {wrong[edge].float().mean():.3f}, on interior pixels {wrong[~edge].float().mean():.3f}")`;

const UNET_OUT = String.raw`class pixel frequencies (train): [0.865, 0.106, 0.029]
skips=True  params 66,495  last-epoch train loss 0.236  pixel acc 0.994  IoU per class [0.996, 0.972, 0.84]  mIoU 0.936
skips=False params 60,015  last-epoch train loss 0.131  pixel acc 0.963  IoU per class [0.961, 0.912, 0.0]  mIoU 0.624
with skips: error on boundary pixels 0.039, on interior pixels 0.000
no skips  : error on boundary pixels 0.262, on interior pixels 0.001`;

const TVSEG = String.raw`import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn

torch.manual_seed(0)
count = lambda m: sum(p.numel() for p in m.parameters())
x = torch.rand(1, 3, 128, 160)

# Semantic segmentation: DeepLabV3 (ResNet-50, dilated last stages, ASPP head). Random init, nothing downloaded.
deeplab = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=4, aux_loss=False).eval()
print(f"DeepLabV3-ResNet50: {count(deeplab):,} parameters")
print("  ASPP atrous rates:", [m[0].dilation[0] for m in deeplab.classifier[0].convs[1:4]])
print("  last backbone stage dilation:", deeplab.backbone.layer4[0].conv2.dilation)
with torch.no_grad():
    feats = deeplab.backbone(x)["out"]
    out = deeplab(x)["out"]
print("  backbone feature map:", tuple(feats.shape), "-> output stride", 128 // feats.shape[2])
print("  logits (bilinearly upsampled to the input):", tuple(out.shape))

# Instance segmentation: Mask R-CNN = Faster R-CNN + a per-RoI mask head on RoIAlign features.
mrcnn = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=3, min_size=224, max_size=320).eval()
print(f"Mask R-CNN-ResNet50-FPN: {count(mrcnn):,} parameters, mask head {count(mrcnn.roi_heads.mask_head) + count(mrcnn.roi_heads.mask_predictor):,}")
print("  mask RoIAlign output:", mrcnn.roi_heads.mask_roi_pool.output_size, " box RoIAlign output:", mrcnn.roi_heads.box_roi_pool.output_size)
with torch.no_grad():
    det = mrcnn([x[0]])[0]
print("  eval output keys:", sorted(det), " masks tensor shape (N, 1, H, W):", tuple(det["masks"].shape[1:]))`;

const TVSEG_OUT = String.raw`DeepLabV3-ResNet50: 39,634,500 parameters
  ASPP atrous rates: [12, 24, 36]
  last backbone stage dilation: (2, 2)
  backbone feature map: (1, 2048, 16, 20) -> output stride 8
  logits (bilinearly upsampled to the input): (1, 4, 128, 160)
Mask R-CNN-ResNet50-FPN: 43,980,897 parameters, mask head 2,623,491
  mask RoIAlign output: (14, 14)  box RoIAlign output: (7, 7)
  eval output keys: ['boxes', 'labels', 'masks', 'scores']  masks tensor shape (N, 1, H, W): (1, 128, 160)`;

const SEG_KERAS = String.raw`import tensorflow as tf
from tensorflow.keras import layers

def double_conv(x, c):
    x = layers.Conv2D(c, 3, padding="same", activation="relu")(x)
    return layers.Conv2D(c, 3, padding="same", activation="relu")(x)

inp = layers.Input((32, 32, 3))                          # channels-last
e1 = double_conv(inp, 12)                                # 32x32
e2 = double_conv(layers.MaxPooling2D()(e1), 24)          # 16x16
b = double_conv(layers.MaxPooling2D()(e2), 48)           # 8x8
d2 = layers.Conv2DTranspose(24, 2, strides=2)(b)         # 16x16
d2 = double_conv(layers.Concatenate()([d2, e2]), 24)     # skip connection by concatenation
d1 = layers.Conv2DTranspose(12, 2, strides=2)(d2)        # 32x32
d1 = double_conv(layers.Concatenate()([d1, e1]), 12)
out = layers.Conv2D(3, 1)(d1)                            # per-pixel logits for 3 classes

model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.MeanIoU(num_classes=3, sparse_y_true=True, sparse_y_pred=False)])
model.summary()`;

const lesson: Lesson = {
  id: "m12/segmentation",
  title: "Semantic, Instance and Panoptic Segmentation",
  summary:
    "Labelling every pixel: the three segmentation tasks and how to score them (pixel accuracy, mIoU, panoptic quality), fully convolutional networks, U-Net, DeepLab's atrous convolution and ASPP, Mask R-CNN with RoIAlign, and losses for rare classes. You build mIoU, PQ and RoIAlign from scratch and train a U-Net with and without skip connections.",
  minutes: 100,
  level: "advanced",
  prerequisites: ["m12/object-detection", "m08/conv-variants", "m05/advanced-losses", "m07/residual-connections"],
  next: ["m12/vision-tasks", "m11/transformer-families", "m05/advanced-losses"],
  terms: ["semantic-segmentation", "instance-segmentation", "panoptic-segmentation", "miou", "panoptic-quality", "fcn", "u-net", "atrous-convolution", "aspp", "roi-align", "skip-connection"],
  sections: {
    intro: [
      { t: "p", text: "**Segmentation** gives a label to every pixel. A detector says \"there is a car in this rectangle\". A segmentation model says exactly which pixels are car. There are three tasks, and they differ in *what* the label tells you:" },
      { t: "list", items: [
        "**Semantic segmentation**: each pixel gets a class (road, car, sky). Two touching cars melt into one \"car\" region. Output: an $H \\times W$ map of class numbers (from $C \\times H \\times W$ logits).",
        "**Instance segmentation**: each *object* gets its own mask, class and confidence score. Background like sky or road is not segmented. Output: a list of (binary mask, class, score) — detection with masks instead of boxes.",
        "**Panoptic segmentation** (Kirillov et al., 2019) combines both: every pixel gets a class *and* an instance id. Countable **things** (cars, people) get one id each; shapeless **stuff** (road, sky) gets one region per class. No overlaps.",
      ] },
      { t: "p", text: "You need it whenever shape or area matters, not only location: tumour volume in a scan, the drivable road for a self-driving car, background removal in video calls." },
      { t: "p", text: "The hard part: classification CNNs *throw away* resolution. A ResNet shrinks a 224×224 image to a 7×7 map (stride 32). Segmentation needs a full-size output. Each famous model answers \"how do we get wide context *and* fine detail?\":" },
      { t: "list", items: [
        "**FCN** (Long, Shelhamer and Darrell, 2015) upsamples and mixes in earlier layers.",
        "**U-Net** (Ronneberger, Fischer and Brox, 2015) mirrors the encoder with a decoder, linked by skip connections.",
        "**DeepLab** (Chen et al., 2015–2018) avoids shrinking with atrous (dilated) convolution and gathers multi-scale context with ASPP.",
        "**Mask R-CNN** (He et al., 2017) adds a mask branch to a two-stage detector, made pixel-accurate by **RoIAlign**.",
      ] },
      { t: "callout", kind: "key", text: "Segmentation = classifying every pixel, with a network whose output is as large as its input. The core tension: context needs shrinking, exact location needs high resolution. Skips, dilated convolutions and RoIAlign are the main fixes. Measure with IoU per class, not pixel accuracy." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Object detection](/learn/m12/object-detection) — boxes, IoU, anchors, NMS and Faster R-CNN. Mask R-CNN builds straight on it.",
        "[Dilated, transposed, depthwise convolutions](/learn/m08/conv-variants) — transposed convolution (learned upsampling) and dilated convolution are used everywhere here and not re-explained.",
        "[Pooling and receptive fields](/learn/m08/pooling-receptive-field) — why context needs downsampling, and how dilation grows the receptive field.",
        "[Residual connections](/learn/m07/residual-connections) — U-Net's skips *concatenate* instead of adding; that lesson compares the two.",
        "[Advanced losses](/learn/m05/advanced-losses) — Dice and IoU (Jaccard) losses for rare classes are derived there.",
        "[Classification losses](/learn/m05/classification-losses) — the default segmentation loss is cross-entropy at every pixel.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Colouring a map from an aerial photo", text: "Zoomed out, you see this area is a *town* and that one a *lake* — but not the exact shoreline. Zoomed in, every edge is sharp — but a grey patch could be a road or a roof. A good map-maker uses the zoomed-out view to decide *what* a region is, and the zoomed-in view to decide *where* its border runs." },
      { t: "p", text: "The **encoder** (a classification-style CNN) is the zoomed-out view: deep, low-resolution features that know *what*. **Skip connections** hand the **decoder** the zoomed-in view: early, high-resolution features that know *where*. The decoder combines both. DeepLab instead keeps the map at medium resolution and uses dilated filters — filters with holes — to look far without zooming out." },
      { t: "p", text: "Instance segmentation works differently: a detector proposes boxes, then the object inside each box is cut out. That only works if the feature-map crop lines up exactly with the box — which RoIAlign guarantees and the older RoI pooling did not." },
    ],
    mechanism: [
      { t: "h", text: "Fully convolutional networks (FCN)" },
      { t: "steps", items: [
        { title: "Turn a classifier into convolutions", text: "Replace the fully connected layers with convolutions: a $7 \\times 7$ FC layer becomes a $7 \\times 7$ conv, later FCs become $1 \\times 1$ convs. Now any input size works, and the output is a coarse **score map**: one class-score vector per stride-32 cell." },
        { title: "Upsample", text: "Bring the score map back to input size with a transposed convolution (started as bilinear interpolation). Upsampling by 32 in one go gives **FCN-32s**: right classes, blobby edges." },
        { title: "Skip fusion", text: "Also predict scores from the stride-16 map (with a $1 \\times 1$ conv), add them to the stride-32 scores upsampled 2×, then upsample by 16: **FCN-16s**. Adding stride-8 scores too gives **FCN-8s**, with clearly sharper edges." },
        { title: "Train", text: "Softmax cross-entropy averaged over all labelled pixels; 'void' pixels are skipped." },
      ] },
      { t: "h", text: "Encoder–decoder and U-Net" },
      { t: "steps", items: [
        { title: "Contracting path (encoder)", text: "Two 3×3 convs + ReLU, then 2×2 max-pooling, repeated. Channels double each level (64 → 128 → 256 → 512 → 1024 in the original). Resolution drops, context grows." },
        { title: "Expanding path (decoder)", text: "At each level, a 2×2 **up-convolution** (transposed conv) doubles the resolution and halves the channels." },
        { title: "Skip connections", text: "The encoder map from the same level is **concatenated** with the upsampled decoder map along the channel axis, then two 3×3 convs follow. (The original cropped the encoder map to fit, because its convolutions were unpadded.) So the decoder sees both 'what' and 'where'." },
        { title: "Output", text: "A 1×1 conv turns the final features into $C$ class logits per pixel." },
      ] },
      { t: "p", text: "U-Net was built for biomedical images, with very few training images. It used heavy elastic-deformation augmentation and a per-pixel **weight map** that makes thin borders between touching cells count more. Unpadded convolutions turn a 572×572 input into a 388×388 output, so big images are processed as overlapping tiles. Modern versions mostly use padded convolutions, so sizes match. **SegNet** (Badrinarayanan, Kendall and Cipolla) is a cousin that passes only max-pooling *positions* (indices) to the decoder." },
      { t: "h", text: "DeepLab: atrous convolution and ASPP" },
      { t: "steps", items: [
        { title: "Stop shrinking, keep the view", text: "In the backbone's last stages, set the stride to 1 and dilate the following convolutions instead (rate 2, then 4). The **output stride** (input size / feature size) becomes 16 or 8 instead of 32. Weights and receptive field stay the same; bigger maps cost more compute." },
        { title: "Atrous Spatial Pyramid Pooling (ASPP)", text: "On the same feature map, run in parallel: several 3×3 convolutions with different dilation rates (e.g. 6, 12, 18 at output stride 16; doubled at output stride 8), a 1×1 conv, and an image-level branch using global average pooling. Concatenate and mix with a 1×1 conv. Each branch sees a different scale." },
        { title: "Upsample / decode", text: "DeepLabv3 bilinearly upsamples the logits to input size. DeepLabv3+ adds a light decoder that mixes in stride-4 encoder features for sharper edges. DeepLab v1/v2 also cleaned up the result with a fully connected CRF." },
      ] },
      { t: "h", text: "Mask R-CNN" },
      { t: "steps", items: [
        { title: "Detect", text: "A Faster R-CNN with FPN proposes regions, then classifies and refines them, as in the [detection lesson](/learn/m12/object-detection)." },
        { title: "RoIAlign", text: "For every region (RoI), read a fixed grid of features (7×7 for the box head, 14×14 for the mask head) from the right FPN level. Each value is a **bilinear interpolation** at an exact, non-integer position — nothing is rounded." },
        { title: "Mask head", text: "A small FCN (four 3×3 convs and a 2× transposed conv) outputs one 28×28 mask **per class**. Each pixel uses a sigmoid with binary cross-entropy, applied only to the mask of the true class. Classes do not compete; choosing the class is the box head's job." },
        { title: "Paste", text: "At inference, the predicted class's 28×28 mask is resized to the box and thresholded at 0.5, giving the object's mask in the image." },
      ] },
      { t: "h", text: "RoI pooling versus RoIAlign" },
      { t: "p", text: "RoI pooling (Fast R-CNN) rounds twice: it snaps the box to the feature grid, then splits it into bins with whole-number edges and max-pools each. At stride 16, half a cell is 8 input pixels — fine for classifying a box, bad for a pixel-accurate mask. RoIAlign never rounds: it places $s \\times s$ evenly spaced points in each bin, computes each by bilinear interpolation from the four nearest cells, and averages them." },
      { t: "h", text: "Panoptic models" },
      { t: "p", text: "Early panoptic systems (e.g. Panoptic FPN) ran a semantic head (stuff) and an instance head (things) on one backbone and merged them with hand-made rules. Later **mask-classification** models such as MaskFormer and Mask2Former predict a set of (mask, class) pairs, matched to the truth with DETR-style Hungarian matching. One architecture handles all three tasks." },
    ],
    math: [
      { t: "h", text: "1. Per-pixel cross-entropy and class weighting" },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{CE}} = -\frac{1}{\sum_{i} w_{y_i}} \sum_{i=1}^{HW} w_{y_i} \log \frac{e^{z_{i, y_i}}}{\sum_{c=1}^{C} e^{z_{i,c}}}` },
      { t: "p", text: String.raw`In words: a weighted average, over pixels, of minus the log-probability of the right class. $i$ counts pixels, $y_i$ is pixel $i$'s true class, $z_{i,c}$ its logit for class $c$, and $w_c$ a class weight ($w_c = 1$ gives the plain mean). With 86% background pixels, the plain loss is mostly about background. **Inverse-frequency** weights $w_c \propto 1/f_c$, or **median-frequency balancing** $w_c = \mathrm{median}(f)/f_c$ (used by SegNet), fix that. In the 8×8 toy image below, frequencies $(0.734, 0.141, 0.125)$ give median-frequency weights $(0.191, 1, 1.125)$. Overlap losses (Dice, Jaccard — [derived here](/learn/m05/advanced-losses)) are another option and do not care how big the background is; CE + Dice is a common mix.` },
      { t: "h", text: "2. Semantic metrics: pixel accuracy and mIoU" },
      { t: "p", text: String.raw`Count the confusion matrix $N_{ij}$ = number of pixels of true class $i$ predicted as $j$, over the **whole** dataset. Then:` },
      { t: "math", tex: String.raw`\text{pixel acc} = \frac{\sum_i N_{ii}}{\sum_{i,j} N_{ij}},\qquad \mathrm{IoU}_c = \frac{N_{cc}}{\sum_j N_{cj} + \sum_i N_{ic} - N_{cc}} = \frac{TP_c}{TP_c + FP_c + FN_c},\qquad \mathrm{mIoU} = \frac{1}{C}\sum_c \mathrm{IoU}_c` },
      { t: "p", text: "IoU of a class is overlap over union: correct pixels divided by correct + false alarms + misses. Worked example (8×8 image; class 1 predicted one column too wide, class 2 one row too thin):" },
      { t: "table", head: ["true \\ predicted", "0", "1", "2"], rows: [["0 (background)", "44", "3", "0"], ["1", "0", "9", "0"], ["2", "4", "0", "4"]] },
      { t: "steps", items: [
        { title: "Pixel accuracy", tex: String.raw`(44 + 9 + 4)/64 = 57/64 = 0.8906` },
        { title: "IoU of class 0", tex: String.raw`\frac{44}{(44+3+0) + (44+0+4) - 44} = \frac{44}{51} = 0.8627` },
        { title: "IoU of class 1", tex: String.raw`\frac{9}{9 + 12 - 9} = \frac{9}{12} = 0.75` },
        { title: "IoU of class 2", tex: String.raw`\frac{4}{8 + 4 - 4} = \frac{4}{8} = 0.5` },
        { title: "mIoU", tex: String.raw`(0.8627 + 0.75 + 0.5)/3 = 0.7042` },
      ] },
      { t: "p", text: "Predicting *everything* as background gives pixel accuracy 0.7344 but mIoU only 0.2448 ($= 0.7344/3$: background IoU 0.7344, both object classes 0). Pixel accuracy rewards ignoring small classes; mIoU does not. (Some benchmarks also report frequency-weighted IoU.)" },
      { t: "h", text: "3. Panoptic quality" },
      { t: "p", text: String.raw`Kirillov et al. (2019) pair predicted and true segments of the same class; a pair is a match (TP) if IoU > 0.5. Segments in one image never overlap, so each match is **unique** — no assignment algorithm needed. Unmatched predictions are FP; unmatched true segments are FN. Then:` },
      { t: "math", tex: String.raw`\mathrm{PQ} = \frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP| + \tfrac12|FP| + \tfrac12|FN|} = \underbrace{\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP|}}_{\text{SQ (segmentation quality)}} \times \underbrace{\frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}}_{\text{RQ (recognition quality)}}` },
      { t: "p", text: String.raw`SQ (mean IoU of matches) says how well segments are drawn; RQ (a detection F1 score) says how well they are found. PQ is computed per class and averaged over classes, things and stuff alike. Worked example from the code: the truth has two \"cars\" and one \"person\". Predictions: the first car slightly shifted (IoU 0.7608 → TP); a much too small second car (IoU < 0.5 → FP, and the true car → FN); the person exactly (IoU 1 → TP); and a made-up extra person (FP). So $TP = 2, FP = 2, FN = 1$:` },
      { t: "math", tex: String.raw`\mathrm{SQ} = \frac{0.7608 + 1}{2} = 0.8804,\qquad \mathrm{RQ} = \frac{2}{2 + 1 + 0.5} = 0.5714,\qquad \mathrm{PQ} = 0.8804 \times 0.5714 = 0.5031`, caption: "Both classes pooled here for illustration; the official metric averages PQ per class." },
      { t: "h", text: "4. Receptive field of atrous convolution" },
      { t: "p", text: String.raw`A $k \times k$ kernel with dilation rate $r$ spans $k_{\text{eff}} = k + (k-1)(r-1)$ positions per side, still with only $k^2$ weights. ASPP's $3 \times 3$ branches at rates 6, 12, 18 span 13, 25 and 37 feature cells — at output stride 16, that is 208, 400 and 592 input pixels. For input size $n$ at output stride $OS$, the feature map is $\lceil n/OS \rceil$ wide. Keeping $OS = 8$ instead of 32 makes the last feature maps 16× larger in area, so 16× the compute there.` },
      { t: "h", text: "5. U-Net feature-map sizes" },
      { t: "p", text: String.raw`Each unpadded 3×3 convolution trims 2 pixels; each level has two (−4). Pooling halves, up-convolution doubles. From a $572 \times 572$ input: $572 \to 568 \xrightarrow{\text{pool}} 284 \to 280 \to 140 \to 136 \to 68 \to 64 \to 32 \to 28$ (bottleneck). Going up: $28 \to 56$, concatenate with the encoder's 64-pixel map **cropped** by 4 px per side, two convs $\to 52$; then $104 \to 100$, $200 \to 196$, $392 \to 388$. The $388 \times 388$ output covers the centre of the input; the 92-pixel border on each side is only context.` },
      { t: "h", text: "6. Bilinear interpolation and RoIAlign" },
      { t: "p", text: String.raw`To read a feature map at a non-integer position $(y, x)$, let $y_0 = \lfloor y \rfloor$, $x_0 = \lfloor x \rfloor$ (round down), $l_y = y - y_0$, $l_x = x - x_0$ (the leftover fractions). Then:` },
      { t: "math", tex: String.raw`f(y,x) = (1-l_y)(1-l_x)f_{y_0,x_0} + (1-l_y)l_x f_{y_0,x_0+1} + l_y(1-l_x) f_{y_0+1,x_0} + l_y l_x f_{y_0+1,x_0+1}` },
      { t: "p", text: "In words: a weighted average of the four surrounding cells, where closer cells get more weight." },
      { t: "p", text: String.raw`RoIAlign: map the box $(x_1, y_1, x_2, y_2)$ to feature coordinates: multiply by the spatial scale $1/\text{stride}$, and with the \"aligned\" convention subtract 0.5 so pixel centres sit at integer + 0.5. Split it into $P \times P$ bins of exact size $b_w = (x_2 - x_1)/P$, $b_h = (y_2 - y_1)/P$. Sample $s \times s$ points per bin at $x_1 + (j + (b + 0.5)/s)\,b_w$ (same for $y$), and average them. It is a weighted sum of features with weights that change smoothly with the box, so it is differentiable with respect to the features (and almost everywhere with respect to the box coordinates).` },
      { t: "p", text: "Worked example: feature map $f_{r,c} = 6r + c$ (6×6), box $(1.3, 0.7, 4.9, 3.6)$, stride 1, output 2×2, $s = 2$." },
      { t: "steps", items: [
        { title: "Shift by 0.5 (aligned)", tex: String.raw`(x_1, y_1, x_2, y_2) = (0.8,\ 0.2,\ 4.4,\ 3.1),\quad b_w = 3.6/2 = 1.8,\quad b_h = 2.9/2 = 1.45` },
        { title: "Sample points of bin (0, 0)", tex: String.raw`x \in \{0.8 + 0.25(1.8),\ 0.8 + 0.75(1.8)\} = \{1.25,\ 2.15\},\quad y \in \{0.2 + 0.25(1.45),\ 0.2 + 0.75(1.45)\} = \{0.5625,\ 1.2875\}` },
        { title: "One bilinear sample", text: "At $(y, x) = (0.5625, 1.25)$: neighbours $f_{0,1} = 1, f_{0,2} = 2, f_{1,1} = 7, f_{1,2} = 8$, with $l_y = 0.5625$, $l_x = 0.25$.", tex: String.raw`0.4375(0.75)(1) + 0.4375(0.25)(2) + 0.5625(0.75)(7) + 0.5625(0.25)(8) = 4.625` },
        { title: "Average the four samples", text: "$f$ is linear, so the average equals $f$ at the bin centre $(0.925, 1.7)$:", tex: String.raw`6(0.925) + 1.7 = 7.25` },
        { title: "RoI pooling for comparison", text: "Round the box to $(1, 1, 5, 4)$ and max-pool whole-number bins:", tex: String.raw`\begin{pmatrix}15 & 17\\ 27 & 29\end{pmatrix}\quad\text{vs. RoIAlign}\quad\begin{pmatrix}7.25 & 9.05\\ 15.95 & 17.75\end{pmatrix}` },
      ] },
      { t: "p", text: "RoI pooling's output is shifted (by rounding) and pulled toward each bin's bottom-right corner (by the max). RoIAlign's values are exactly the features at the bin centres." },
    ],
    visual: [
      { t: "p", text: "Learned upsampling and dilated convolution turn a classifier into a segmentation network. Try both:" },
      { t: "sim", id: "conv-variants", caption: "Dilated tab: DeepLab's atrous convolution. Transposed tab: the learned upsampling in FCN and U-Net decoders." },
      { t: "list", items: [
        "**Dilated tab.** Set dilation to 2, 3, 4 and watch the kernel's span grow ($3 + 2(d-1)$) with the same 9 weights. Stack rates 1, 2, 4 to see how DeepLab looks far without pooling.",
        "**Dilated tab, padding.** Switch to 'same' padding ($p = d$): the output keeps the input size — just what you want after removing the backbone's last strides.",
        "**Transposed tab.** Use stride 2: each input value is spread over a 2×2 (or larger) patch of the output — the decoder's 'up-convolution'. With kernel 4 and stride 2, look for uneven overlap: that causes checkerboard artefacts.",
      ] },
      { t: "p", text: "Masks are scored with the same IoU the detection simulator computes for boxes, just counted in mask pixels." },
      { t: "sim", id: "detection", caption: "IoU tab: the same intersection-over-union used for mIoU and for matching segments in panoptic quality." },
    ],
    scratch: [
      { t: "p", text: "Segmentation's scoring and geometry tools in NumPy: a confusion matrix with pixel accuracy and per-class IoU / mIoU, panoptic quality with SQ and RQ, bilinear RoIAlign and RoI pooling (checked against `torchvision.ops.roi_align` and `roi_pool`), the original U-Net's size arithmetic, and class weights for imbalanced cross-entropy." },
      { t: "code", lang: "python", title: "segmentation_from_scratch.py", code: SEG_NUMPY, output: SEG_NUMPY_OUT },
      { t: "list", items: [
        "`confusion` fills the whole matrix in one `bincount(gt * C + pred)` call. In real use, add up the matrix over the whole dataset before dividing — averaging per-image mIoUs gives a different, noisier number.",
        "`panoptic_quality` uses the fact that IoU > 0.5 matches are unique: once a prediction matches a true segment, we stop searching.",
        "`roi_align` matches torchvision to $10^{-14}$ over 50 random boxes with spatial scale 1/4 (a stride-4 feature map), sampling ratio 2 and `aligned=True`. `roi_pool` matches exactly.",
        "The U-Net arithmetic reproduces the paper's 572 → 388 sizes and each skip's crop.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch: U-Net with and without skips", blocks: [
          { t: "p", text: "A small U-Net trained on synthetic 32×32 images. Class 1 ('blob') is randomly coloured filled discs and squares; class 2 ('line') is thin straight lines about 1.5 px wide. Backgrounds are noisy, with faint smudges; the class depends on *shape*, not colour. We train the same network twice with the same seeds — once with concatenation skips, once without (then the decoder sees only the upsampled bottleneck) — and test on 150 unseen images." },
          { t: "code", lang: "python", title: "unet_skip_ablation.py", code: UNET, output: UNET_OUT },
          { t: "list", items: [
            "**Without skips the thin lines disappear**: line IoU is 0.0 — never predicted. A 1.5-px line cannot survive two 2× poolings down to an 8×8 bottleneck and then be redrawn in the right place by upsampling alone. With skips, full-resolution encoder features carry it straight to the output: line IoU 0.84, mIoU 0.936 vs 0.624.",
            "Both models find the blobs (IoU 0.972 vs 0.912); the difference is at the edges: boundary-pixel error is 3.9% with skips vs 26.2% without, while inside pixels are almost perfect for both.",
            "Pixel accuracy hides the failure: 0.963 for a model missing a whole class (lines are only 2.9% of pixels).",
            "Lower cross-entropy is not better segmentation: the no-skip model ends with a *lower* average training loss (0.131 vs 0.236) yet much lower mIoU. The loss is dominated by easy background and depends on calibration. Always evaluate with IoU per class.",
            "Removing the skips saves only 6,480 parameters (66,495 vs 60,015). The gain is the information path, not capacity.",
          ] },
          { t: "h", text: "torchvision's DeepLabV3 and Mask R-CNN (random init, no download)" },
          { t: "code", lang: "python", title: "torchvision_segmentation_models.py", code: TVSEG, output: TVSEG_OUT },
          { t: "p", text: "torchvision's DeepLabV3 swaps the strides of ResNet-50's last two stages for dilation, giving output stride 8: a 128×160 input yields a 16×20 map with 2048 channels. Its ASPP uses rates 12, 24, 36 — doubled to fit output stride 8. Mask R-CNN adds a 2.6M-parameter mask head fed by 14×14 RoIAlign features. In eval mode each detection has a soft mask already pasted into the full image, shape (N, 1, H, W); threshold it at 0.5." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same small U-Net in Keras (channels-last). `Concatenate` makes the skip connections. `MeanIoU` with `sparse_y_pred=False` takes the argmax of the logits for you." },
          { t: "code", lang: "python", title: "unet_keras.py (not executed in CI)", code: SEG_KERAS, output: "Prints the model summary: input (32, 32, 3), two encoder levels, an 8×8 bottleneck with 48 channels, two Conv2DTranspose + Concatenate decoder levels and a (32, 32, 3) logit output. Train with model.fit(images, masks) where masks are integer class maps of shape (N, 32, 32)." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Model", "Task", "Key idea", "Pros", "Cons"], rows: [
        ["FCN-32s/16s/8s (Long et al., 2015)", "Semantic", "Classifier made fully convolutional + upsampling + adding earlier scores", "Simple; any input size; trained end to end", "Coarse edges (especially 32s)"],
        ["U-Net (Ronneberger et al., 2015)", "Semantic (esp. biomedical)", "Mirror-image encoder–decoder with concatenation skips", "Sharp edges; works with little data; easy to adapt (3D, nnU-Net)", "Full-resolution decoder uses a lot of memory"],
        ["SegNet", "Semantic", "Decoder upsamples using saved max-pool positions", "Saves memory", "Less detail than full skip features"],
        ["DeepLab v1–v3+ (Chen et al.)", "Semantic", "Atrous convolution, ASPP, (v1/v2) CRF, (v3+) decoder", "Wide context at medium resolution; strong benchmark scores", "Large dilated maps are expensive"],
        ["PSPNet (Zhao et al., 2017)", "Semantic", "Pyramid pooling module for whole-image context", "Consistent labels across the scene", "Pooled context is coarse"],
        ["Mask R-CNN (He et al., 2017)", "Instance", "Faster R-CNN + per-RoI mask head + RoIAlign", "Accurate instances; extends to keypoints", "Masks only 28×28 per RoI; two-stage cost"],
        ["Panoptic FPN, Mask2Former", "Panoptic (and all)", "Merge semantic + instance heads / predict a set of masks", "One model for all three tasks", "Complex training; heavy compute"],
      ] },
      { t: "list", items: [
        "**Upsampling options**: bilinear interpolation (no parameters, followed by a conv), transposed convolution (learned, may cause checkerboard artefacts), unpooling with indices (SegNet), pixel shuffle.",
        "**Skip fusion options**: concatenation (U-Net; the decoder learns how to combine) vs addition (FCN, FPN; cheaper, needs equal channel counts).",
        "**Promptable segmentation**: the Segment Anything Model (Kirillov et al., 2023) draws a mask for a given point or box, trained on a huge automatically labelled mask dataset.",
        "**3D segmentation** (CT/MRI volumes) uses 3D convolutions in the same U-Net pattern.",
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Medical imaging**: organs and tumours in CT and MRI, cells in microscopy (U-Net's original use), retinal vessels. Volumes and borders matter, so masks — not boxes.",
        "**Self-driving**: drivable area, lane markings and full scene labelling (semantic), plus separate vehicles and pedestrians (panoptic).",
        "**Remote sensing**: land cover, building outlines, floods and crop fields from satellite images.",
        "**Photo and video editing**: background removal, portrait mode, cut-out tools.",
        "**Robotics and industry**: grasp planning, surface defects, conveyor-belt sorting.",
        "**Agriculture**: plants, weeds and fruit for targeted spraying and yield estimates.",
      ] },
    ],
    tradeoffs: [
      { t: "h", text: "Strengths" },
      { t: "list", items: [
        "Pixel-precise output: shape, area and borders, not just location.",
        "Fully convolutional models take any input size and share computation across all pixels.",
        "Encoder–decoders with skips train well even on small datasets (with augmentation).",
      ] },
      { t: "h", text: "Limitations and trade-offs" },
      { t: "list", items: [
        "**Labelling cost**: pixel masks cost far more than boxes or image labels; label errors cluster at borders.",
        "**Resolution vs compute**: high-resolution decoders and dilated backbones multiply memory and FLOPs (output stride 8 vs 32 → 16× the area).",
        "**Class imbalance**: background swamps small structures in the loss; you need weighting, Dice, or sampling.",
        "**Semantic models cannot count**: touching objects of one class merge. You need instance or panoptic methods.",
        "**Mask R-CNN masks are low resolution** (28×28 per object), so large objects get soft, blobby edges.",
        "**Metrics differ by task**: pixel accuracy, mIoU, mask AP and PQ are not interchangeable.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Reporting pixel accuracy on imbalanced data.** A model that ignores a whole class can score 96% (like our no-skip run). Report per-class IoU and mIoU.",
        "**Averaging mIoU per image instead of adding up one confusion matrix over the dataset.** Images missing a class give 0/0 or noise. Add up counts, then divide once.",
        "**Resizing masks with bilinear interpolation.** Blending class numbers creates fake labels (class 1.5). Use nearest-neighbour for label maps (bilinear is fine for images and soft masks).",
        "**Augmenting image and mask differently** (e.g. a random crop drawn twice). Use one transform on both.",
        "**Forgetting the ignore label** (e.g. 255 for 'void' pixels). Use `ignore_index` in the loss and leave them out of the metrics.",
        "**Using softmax over classes for Mask R-CNN masks.** The paper uses a per-class sigmoid with BCE on the true class only; softmax makes classes compete and hurts masks.",
        "**Mismatched skip sizes** with odd inputs: pooling 33 → 16 then upsampling → 32 breaks concatenation. Pad inputs to a multiple of $2^{\\text{levels}}$, or crop/interpolate.",
        "**Using RoI pooling for masks** — rounding misaligns features by up to one stride. Use RoIAlign.",
        "**Judging training by the loss alone**: CE can fall while a rare class is never predicted. Track per-class IoU on validation data.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Distinguish semantic, instance and panoptic segmentation.", a: "Semantic: one class per pixel, no separate objects. Instance: a mask (with class and score) per countable object; background stuff ignored; masks may overlap. Panoptic: every pixel gets a class and, for 'things', an instance id; each 'stuff' class is one region; no overlaps." },
      { t: "qa", level: "beginner", q: "Why is pixel accuracy a poor metric for segmentation?", a: "Background often covers most pixels, so a model predicting only background still scores high. mIoU computes IoU = TP/(TP+FP+FN) for each class and averages over classes, so every class counts equally." },
      { t: "qa", level: "beginner", q: "What makes a network 'fully convolutional' and why does that matter for segmentation?", a: "It has no fully connected layers that need a fixed input size — only convolutions, pooling and upsampling. So any image size maps to a grid of predictions, all pixels computed in one pass with shared weights." },
      { t: "qa", level: "intermediate", q: "What do U-Net's skip connections provide, and why concatenate rather than add?", a: "They pass high-resolution encoder features (edges, exact positions) to the decoder, which otherwise has only coarse upsampled bottleneck features. Essential for thin structures and borders (in our run, thin lines went from IoU 0 to 0.84). Concatenation keeps both feature sets separate and lets the next convolutions learn how to combine them; addition forces them into one shared channel space." },
      { t: "qa", level: "intermediate", q: "How does atrous convolution help semantic segmentation? What does ASPP add?", a: "Dilating later-stage kernels instead of striding keeps feature maps at output stride 8 or 16, with the same receptive field and pretrained weights, so predictions are less coarse. ASPP runs parallel atrous convolutions with several rates, plus image-level pooling, on the same features. That captures context at several scales." },
      { t: "qa", level: "intermediate", q: "Define panoptic quality and explain SQ and RQ.", a: "PQ = Σ_TP IoU / (|TP| + ½|FP| + ½|FN|). A predicted and a true segment of the same class match if IoU > 0.5; the match is unique because segments do not overlap. It splits into SQ = mean IoU of matches (how well they are drawn) and RQ = |TP| / (|TP| + ½|FP| + ½|FN|), an F1 score (how well segments are found). PQ is averaged over classes." },
      { t: "qa", level: "advanced", q: "Explain RoIAlign and why Mask R-CNN needed it.", a: "RoI pooling rounds the RoI to the feature grid and splits it into whole-number bins, so features can be misaligned by up to a stride (e.g. 16 px). RoIAlign keeps coordinates exact: each bin samples a few evenly spaced points by bilinear interpolation of the 4 nearest cells and averages (or maxes) them. Pixel masks are very sensitive to misalignment; He et al. report large mask-AP gains from this change alone." },
      { t: "qa", level: "advanced", q: "Your segmentation model has high mIoU on large classes but misses thin structures. What would you change?", a: "Add resolution where it matters: add or strengthen skips from early layers, lower the output stride (dilation instead of stride), add a decoder (DeepLabv3+ style), train at higher input resolution. Rebalance the loss (class weights, Dice/Tversky or boundary-aware losses), oversample images with the structure, and check labels were resized with nearest-neighbour. Evaluate with per-class IoU and boundary metrics." },
      { t: "qa", level: "advanced", q: "Why does Mask R-CNN predict one binary mask per class with a sigmoid instead of a multi-class softmax mask?", a: "To separate masks from classification. The box head picks the class; the mask head only answers 'which pixels in this box belong to an object of class k', without competing against other classes. He et al. found a per-pixel softmax over classes clearly worse. The loss is applied only to the true class's mask." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "For one class, TP = 30, FP = 10, FN = 20 pixels. What is its IoU? (2 decimals)", answer: 0.5, tolerance: 0.005, explain: "IoU = TP/(TP+FP+FN) = 30/60 = 0.5." },
        { kind: "numeric", q: "TP = 6 matched segments with IoUs summing to 4.8, FP = 2, FN = 4. What is PQ? (3 decimals)", answer: 0.533, tolerance: 0.002, explain: "SQ = 4.8/6 = 0.8; RQ = 6/(6 + 1 + 2) = 0.667; PQ = 0.8 × 0.667 = 0.533 (= 4.8/9)." },
        { kind: "numeric", q: "A 3×3 kernel with dilation rate 12 covers how many input positions per side?", answer: 25, tolerance: 0, explain: "$k + (k-1)(r-1) = 3 + 2 \\times 11 = 25$." },
        { q: "Which method should you use to resize a ground-truth label map?", options: ["Bilinear interpolation", "Bicubic interpolation", "Nearest-neighbour interpolation", "Average pooling"], answer: 2, explain: "Label numbers are categories; blending them creates labels that do not exist." },
        { q: "What is the main reason RoIAlign gives better masks than RoI pooling?", options: ["It uses more parameters", "It avoids rounding the RoI and bin boundaries, sampling features bilinearly at exact positions", "It uses average instead of max", "It increases the output size to 28×28"], answer: 1, explain: "Rounding misaligns features by up to one stride; RoIAlign removes that." },
        { q: "In our experiment, why did the U-Net without skip connections get IoU 0 on the 'line' class?", options: ["Lines were absent from the test set", "The 1.5-px lines cannot be reconstructed from an 8×8 bottleneck by upsampling alone", "The model had too few parameters", "Cross-entropy cannot learn thin structures"], answer: 1, explain: "Downsampling loses where exactly the line is; skips bring it back (IoU 0.84) for only 6.5k extra parameters." },
        { kind: "numeric", q: "Unpadded U-Net: a level's feature map is 136×136 and the upsampled decoder map is 104×104. How many pixels must be cropped from each side of the encoder map?", answer: 16, tolerance: 0, explain: "(136 − 104)/2 = 16." },
        { q: "Which statement about panoptic segmentation is correct?", options: ["Stuff classes get one instance id per object", "Segments may overlap", "Every pixel receives a class label and, for thing classes, an instance id", "It ignores background classes"], answer: 2, explain: "Panoptic output splits the image into non-overlapping parts: things are split into instances, stuff is one region per class." },
      ] },
      { t: "exercise", title: "Metrics by hand", prompt: "A 4×4 image has 10 background pixels (class 0) and 6 object pixels (class 1) in the ground truth. The prediction marks 8 pixels as class 1, and 5 of them really are class 1. Compute pixel accuracy, the IoU of both classes and mIoU. Then compute PQ, treating this as a single 'thing' segment on each side.", hint: "Build the 2×2 confusion matrix first.", solution: [
        { t: "steps", items: [
          { title: "Confusion matrix", tex: String.raw`N = \begin{pmatrix} 7 & 3 \\ 1 & 5 \end{pmatrix}\ (\text{rows = truth})` },
          { title: "Pixel accuracy", tex: String.raw`(7 + 5)/16 = 0.75` },
          { title: "IoUs", tex: String.raw`\mathrm{IoU}_0 = \frac{7}{10 + 8 - 7} = 0.6364,\quad \mathrm{IoU}_1 = \frac{5}{6 + 8 - 5} = 0.5556,\quad \mathrm{mIoU} = 0.5960` },
          { title: "PQ", tex: String.raw`\mathrm{IoU} = 0.5556 > 0.5 \Rightarrow TP = 1, FP = FN = 0,\ \mathrm{SQ} = 0.5556,\ \mathrm{RQ} = 1,\ \mathrm{PQ} = 0.5556` },
        ] },
      ] },
      { t: "exercise", title: "Class-weighted loss and a Dice term", prompt: "Change `unet_skip_ablation.py`: (a) compute median-frequency class weights from `freq` and pass them as `weight=` to `F.cross_entropy`; (b) add a soft Dice loss averaged over the two foreground classes. Train the *no-skip* model with each loss and report the line-class IoU. Does re-weighting bring back the missing class?", hint: "Soft Dice per class: $1 - (2\\sum p\\,g + \\epsilon)/(\\sum p + \\sum g + \\epsilon)$ with $p$ = softmax probabilities, $g$ = one-hot targets.", solution: [
        { t: "code", lang: "python", code: String.raw`w = (freq.median() / freq)                          # median-frequency balancing
def dice_loss(logits, y, eps=1.0):
    p = logits.softmax(1)
    g = F.one_hot(y, N_CLS).permute(0, 3, 1, 2).float()
    inter = (p * g).sum((0, 2, 3)); denom = p.sum((0, 2, 3)) + g.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps))[1:].mean()   # foreground classes only

loss = F.cross_entropy(out, Y_tr[b], weight=w) + dice_loss(out, Y_tr[b])` },
        { t: "p", text: "Re-weighting makes the no-skip model *predict* lines (ignoring them is now expensive), so recall rises. But without high-resolution information it cannot place them precisely, so line IoU stays far below the U-Net with skips. The loss sets priorities; the architecture decides what information gets through. Run it to see the exact numbers on your machine." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Semantic: a class per pixel. Instance: a mask per object. Panoptic: class + instance id per pixel, things and stuff, no overlaps.",
        String.raw`Metrics: pixel accuracy (misleading when classes are imbalanced); $\mathrm{IoU}_c = TP/(TP+FP+FN)$ and mIoU from one dataset-wide confusion matrix; $\mathrm{PQ} = \mathrm{SQ} \times \mathrm{RQ}$ with matches at IoU > 0.5.`,
        "FCN: classifier made fully convolutional + upsampling + skip fusion (32s → 16s → 8s).",
        "U-Net: encoder–decoder with concatenation skips; original 572 → 388 with unpadded convs. Skips carry the 'where' (our run: thin-line IoU 0 → 0.84, mIoU 0.624 → 0.936).",
        String.raw`DeepLab: atrous convolution keeps output stride 8/16; $k_{\text{eff}} = k + (k-1)(r-1)$; ASPP gathers context at several scales.`,
        "Mask R-CNN: Faster R-CNN + per-class 28×28 sigmoid masks on RoIAlign features. RoIAlign = bilinear sampling, no rounding.",
        "Losses: pixel CE (with class weights / ignore index), Dice or Jaccard for imbalance, often combined.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Faces, Pose, Optical Flow, Super-Resolution and Captioning](/learn/m12/vision-tasks) — keypoint heatmaps, flow fields and super-resolution reuse encoder–decoder designs.",
        "[Transformer families](/learn/m11/transformer-families) — ViT and Swin backbones power modern segmentation models such as Mask2Former.",
        "[Advanced losses](/learn/m05/advanced-losses) — revisit Dice, Jaccard and their gradients now that you have seen where they are used.",
      ] },
    ],
  },
};

export default lesson;
