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
    "Pixel-level prediction: the three segmentation tasks and their metrics (pixel accuracy, mIoU, panoptic quality), fully convolutional networks, encoder–decoders and U-Net, atrous convolution and ASPP in DeepLab, Mask R-CNN with RoIAlign, losses for imbalanced pixels — with mIoU, PQ and RoIAlign implemented from scratch and a U-Net trained with and without skip connections.",
  minutes: 100,
  level: "advanced",
  prerequisites: ["m12/object-detection", "m08/conv-variants", "m05/advanced-losses", "m07/residual-connections"],
  next: ["m12/vision-tasks", "m11/transformer-families", "m05/advanced-losses"],
  terms: ["semantic-segmentation", "instance-segmentation", "panoptic-segmentation", "miou", "panoptic-quality", "fcn", "u-net", "atrous-convolution", "aspp", "roi-align", "skip-connection"],
  sections: {
    intro: [
      { t: "p", text: "**Segmentation** assigns a label to every pixel. Where a detector says \"there is a car in this rectangle\", a segmentation model says exactly which pixels are car. Three tasks are distinguished by *what* the label identifies:" },
      { t: "list", items: [
        "**Semantic segmentation**: each pixel gets a class (road, car, sky). Two touching cars form one undifferentiated \"car\" region. Output: an $H \\times W$ map of class indices (from $C \\times H \\times W$ logits).",
        "**Instance segmentation**: each *object* gets its own mask, class and score; background classes like sky or road are not segmented. Output: a variable-size set of (binary mask, class, score) — detection with masks instead of boxes.",
        "**Panoptic segmentation** (Kirillov et al., 2019) unifies both: every pixel gets a class *and* an instance id. Countable **things** (cars, people) get separate ids; amorphous **stuff** (road, sky, grass) gets one region per class. Output: an $H \\times W$ map of (class, instance id) pairs with no overlaps.",
      ] },
      { t: "p", text: "Segmentation is needed wherever shape or area matters, not just location: tumour volumes in medical scans, drivable road surface for autonomous vehicles, land cover in satellite images, background removal in video calls, and precise object outlines for robotic grasping or photo editing." },
      { t: "p", text: "The architectural problem is that classification CNNs *throw away* spatial resolution — a ResNet reduces a 224×224 image to a 7×7 map (stride 32) — while segmentation needs a full-resolution output. The history of the field is a sequence of answers to the question \"how do we get both large context and fine detail?\": **FCN** (Long, Shelhamer and Darrell, 2015) upsamples and fuses earlier layers; **U-Net** (Ronneberger, Fischer and Brox, 2015) mirrors the encoder with a decoder linked by skip connections; **DeepLab** (Chen et al., 2015–2018) avoids downsampling with atrous (dilated) convolution and gathers multi-scale context with ASPP; **Mask R-CNN** (He et al., 2017) adds a mask branch to a two-stage detector, made pixel-accurate by **RoIAlign**." },
      { t: "callout", kind: "key", text: "Segmentation = per-pixel classification with a network whose output has the input's resolution. The core tension is context (needs downsampling, large receptive fields) versus localisation (needs high resolution); skip connections, dilated convolutions and RoIAlign are the main tools for resolving it. Measure with IoU per class, not pixel accuracy." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Object detection](/learn/m12/object-detection) — boxes, IoU, anchors, NMS and Faster R-CNN; Mask R-CNN extends it directly.",
        "[Dilated, transposed, depthwise convolutions](/learn/m08/conv-variants) — transposed convolution (learned upsampling) and dilated convolution are used throughout and are not re-derived here.",
        "[Pooling and receptive fields](/learn/m08/pooling-receptive-field) — why context requires downsampling and how dilation grows the receptive field.",
        "[Residual connections](/learn/m07/residual-connections) — U-Net's skip connections *concatenate* rather than add; that lesson compares the two.",
        "[Advanced losses](/learn/m05/advanced-losses) — Dice and IoU (Jaccard) losses for imbalanced masks are derived there.",
        "[Classification losses](/learn/m05/classification-losses) — segmentation's default loss is cross-entropy applied at every pixel.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Colouring a map from an aerial photo", text: "To colour every pixel of an aerial photo by land use you need two kinds of knowledge. Zoomed out, you recognise that this region is a *town* and that one is a *lake* — but you cannot see the exact shoreline. Zoomed in, you see every edge sharply — but a patch of grey could be a road or a roof. A good cartographer looks at the zoomed-out view to decide *what* each region is and at the zoomed-in view to decide *where exactly* its border runs." },
      { t: "p", text: "An encoder (a classification-style CNN) provides the zoomed-out view: deep, low-resolution features that know *what*. Skip connections hand the decoder the zoomed-in view: early, high-resolution features that know *where*. The decoder combines them to paint a sharp, correctly labelled map. DeepLab takes a different route: it keeps the map at moderate resolution throughout and uses dilated filters — filters with holes — to look far without zooming out." },
      { t: "p", text: "For instance segmentation, the analogy changes: first find each object (a detector proposes boxes), then for each box, carefully cut out the object inside it. Cutting requires that the crop of the feature map line up exactly with the box — which is what RoIAlign guarantees and RoI pooling did not." },
    ],
    mechanism: [
      { t: "h", text: "Fully convolutional networks (FCN)" },
      { t: "steps", items: [
        { title: "Convolutionalise a classifier", text: "Replace the classifier's fully connected layers by convolutions (a $7 \\times 7$ FC layer becomes a $7 \\times 7$ conv, later FCs become $1 \\times 1$ convs). The network now accepts any input size and outputs a coarse **score map** — one class-score vector per stride-32 cell — instead of one vector." },
        { title: "Upsample", text: "Bring the score map back to input resolution with a transposed convolution (initialised to bilinear interpolation). Upsampling stride 32 directly gives **FCN-32s** — correct classes, blobby boundaries." },
        { title: "Skip fusion", text: "Predict scores from the stride-16 feature map too (a $1 \\times 1$ conv), add them to the 2× upsampled stride-32 scores, then upsample by 16: **FCN-16s**. Adding stride-8 scores gives **FCN-8s**, with markedly sharper boundaries." },
        { title: "Train", text: "Pixel-wise softmax cross-entropy, averaged over all labelled pixels (unlabelled 'void' pixels are ignored)." },
      ] },
      { t: "h", text: "Encoder–decoder and U-Net" },
      { t: "steps", items: [
        { title: "Contracting path (encoder)", text: "Repeated blocks of two 3×3 convs + ReLU, then 2×2 max-pooling; channels double at each level (64 → 128 → 256 → 512 → 1024 in the original). Resolution falls, context grows." },
        { title: "Expanding path (decoder)", text: "At each level: a 2×2 **up-convolution** (transposed conv) doubles the resolution and halves the channels." },
        { title: "Skip connections", text: "The encoder feature map of the same level is **concatenated** with the upsampled decoder map along the channel axis (cropped to size in the original, which used unpadded convolutions), followed by two 3×3 convs. The decoder therefore sees both the high-level 'what' and the high-resolution 'where'." },
        { title: "Output", text: "A 1×1 conv maps the final features to $C$ class logits per pixel." },
      ] },
      { t: "p", text: "Ronneberger et al. designed U-Net for biomedical images with very few training images, relying on heavy elastic-deformation augmentation and a pixel-wise **weight map** that up-weights the thin borders between touching cells. With unpadded convolutions, a 572×572 input gives a 388×388 output; large images are processed as overlapping tiles. Most modern implementations use padded convolutions so input and output sizes match. **SegNet** (Badrinarayanan, Kendall and Cipolla) is a related encoder–decoder that passes max-pooling *indices* instead of whole feature maps to the decoder." },
      { t: "h", text: "DeepLab: atrous convolution and ASPP" },
      { t: "steps", items: [
        { title: "Remove downsampling, keep receptive field", text: "In the last stages of the backbone, set the stride to 1 and dilate the subsequent convolutions (rate 2, then 4) instead. The **output stride** (input size / feature size) becomes 16 or 8 instead of 32, with the same weights and the same receptive field — at a higher compute cost because the maps are larger." },
        { title: "Atrous Spatial Pyramid Pooling (ASPP)", text: "Apply several parallel 3×3 convolutions with different dilation rates (e.g. 6, 12, 18 at output stride 16; doubled at output stride 8), a 1×1 conv and an image-level global-average-pooling branch to the same feature map; concatenate and fuse with a 1×1 conv. Each branch sees context at a different scale." },
        { title: "Upsample / decode", text: "DeepLabv3 bilinearly upsamples the logits to input size; DeepLabv3+ adds a light decoder that fuses stride-4 encoder features for sharper boundaries. DeepLab v1/v2 also post-processed with a fully connected CRF." },
      ] },
      { t: "h", text: "Mask R-CNN" },
      { t: "steps", items: [
        { title: "Detect", text: "A Faster R-CNN with FPN proposes regions and classifies/refines them as in the [detection lesson](/learn/m12/object-detection)." },
        { title: "RoIAlign", text: "For every RoI, sample a fixed grid of features (7×7 for the box head, 14×14 for the mask head) from the appropriate FPN level by **bilinear interpolation** at real-valued positions — no rounding anywhere." },
        { title: "Mask head", text: "A small FCN (four 3×3 convs, a 2× transposed conv) outputs a 28×28 mask **per class**; a per-pixel sigmoid with binary cross-entropy is applied only to the mask of the ground-truth class. Masks of different classes do not compete — classification is left to the box head." },
        { title: "Paste", text: "At inference, the predicted class's 28×28 mask is resized to the detected box and thresholded at 0.5 to give the instance mask in image coordinates." },
      ] },
      { t: "h", text: "RoI pooling versus RoIAlign" },
      { t: "p", text: "RoI pooling (Fast R-CNN) quantises twice: the RoI's real-valued coordinates are rounded to the feature grid, and the RoI is split into bins with integer boundaries; each bin is max-pooled. At stride 16 a rounding error of half a cell is 8 input pixels — harmless for classifying the box, harmful for a pixel-accurate mask. RoIAlign keeps every coordinate real-valued, places $s \\times s$ regularly spaced sample points in each bin, computes each by bilinear interpolation from the four nearest feature cells, and averages them." },
      { t: "h", text: "Panoptic models" },
      { t: "p", text: "Early panoptic systems ran a semantic head (for stuff) and an instance head (for things) on a shared backbone and merged their outputs with heuristics (e.g. Panoptic FPN). Later **mask-classification** models such as MaskFormer and Mask2Former predict a set of (mask, class) pairs with DETR-style Hungarian matching and handle semantic, instance and panoptic segmentation with one architecture." },
    ],
    math: [
      { t: "h", text: "1. Per-pixel cross-entropy and class weighting" },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{CE}} = -\frac{1}{\sum_{i} w_{y_i}} \sum_{i=1}^{HW} w_{y_i} \log \frac{e^{z_{i, y_i}}}{\sum_{c=1}^{C} e^{z_{i,c}}}` },
      { t: "p", text: String.raw`Here $i$ indexes pixels, $y_i$ is the true class of pixel $i$, $z_{i,c}$ the logit for class $c$ and $w_c$ a class weight ($w_c = 1$ gives the plain mean). With 86% background pixels, the unweighted loss is dominated by background. **Inverse-frequency** weights $w_c \propto 1/f_c$ or **median-frequency balancing** $w_c = \mathrm{median}(f)/f_c$ (used by SegNet) compensate. In the 8×8 toy image below, frequencies $(0.734, 0.141, 0.125)$ give median-frequency weights $(0.191, 1, 1.125)$. Overlap losses (Dice, Jaccard — [derived here](/learn/m05/advanced-losses)) are an alternative that is insensitive to background size; CE + Dice is a common combination.` },
      { t: "h", text: "2. Semantic metrics: pixel accuracy and mIoU" },
      { t: "p", text: String.raw`Accumulate the confusion matrix $N_{ij}$ = number of pixels of true class $i$ predicted as $j$ over the **whole** dataset. Then` },
      { t: "math", tex: String.raw`\text{pixel acc} = \frac{\sum_i N_{ii}}{\sum_{i,j} N_{ij}},\qquad \mathrm{IoU}_c = \frac{N_{cc}}{\sum_j N_{cj} + \sum_i N_{ic} - N_{cc}} = \frac{TP_c}{TP_c + FP_c + FN_c},\qquad \mathrm{mIoU} = \frac{1}{C}\sum_c \mathrm{IoU}_c` },
      { t: "p", text: "Worked example (8×8 image; class 1 predicted one column too wide, class 2 predicted one row too thin):" },
      { t: "table", head: ["true \\ predicted", "0", "1", "2"], rows: [["0 (background)", "44", "3", "0"], ["1", "0", "9", "0"], ["2", "4", "0", "4"]] },
      { t: "steps", items: [
        { title: "Pixel accuracy", tex: String.raw`(44 + 9 + 4)/64 = 57/64 = 0.8906` },
        { title: "IoU of class 0", tex: String.raw`\frac{44}{(44+3+0) + (44+0+4) - 44} = \frac{44}{51} = 0.8627` },
        { title: "IoU of class 1", tex: String.raw`\frac{9}{9 + 12 - 9} = \frac{9}{12} = 0.75` },
        { title: "IoU of class 2", tex: String.raw`\frac{4}{8 + 4 - 4} = \frac{4}{8} = 0.5` },
        { title: "mIoU", tex: String.raw`(0.8627 + 0.75 + 0.5)/3 = 0.7042` },
      ] },
      { t: "p", text: "Predicting *everything* as background scores a pixel accuracy of 0.7344 but an mIoU of only 0.2448 ($= 0.7344/3$: background IoU 0.7344, the two object classes 0). Pixel accuracy rewards ignoring small classes; mIoU does not. (Some benchmarks also report frequency-weighted IoU.)" },
      { t: "h", text: "3. Panoptic quality" },
      { t: "p", text: String.raw`Kirillov et al. (2019) match predicted and ground-truth segments of the same class; a pair is a match (TP) if their IoU exceeds 0.5. Because segments within one image do not overlap, IoU > 0.5 makes the matching **unique** — no assignment algorithm is needed. Unmatched predictions are FP, unmatched ground truth FN. Then` },
      { t: "math", tex: String.raw`\mathrm{PQ} = \frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP| + \tfrac12|FP| + \tfrac12|FN|} = \underbrace{\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP|}}_{\text{SQ (segmentation quality)}} \times \underbrace{\frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}}_{\text{RQ (recognition quality)}}` },
      { t: "p", text: String.raw`SQ is the mean IoU of matched segments; RQ is the F1 score of detection. PQ is computed per class and averaged over classes (things and stuff alike). Worked example (from the code): ground truth = two \"cars\" and one \"person\"; predictions = a slightly shifted first car (IoU 0.7608 → TP), a far-too-small second car (IoU < 0.5 → FP, and the GT car → FN), the person exactly (IoU 1 → TP), and a spurious person (FP). So $TP = 2, FP = 2, FN = 1$:` },
      { t: "math", tex: String.raw`\mathrm{SQ} = \frac{0.7608 + 1}{2} = 0.8804,\qquad \mathrm{RQ} = \frac{2}{2 + 1 + 0.5} = 0.5714,\qquad \mathrm{PQ} = 0.8804 \times 0.5714 = 0.5031`, caption: "Pooled over both classes for illustration; the official metric averages per-class PQ." },
      { t: "h", text: "4. Receptive field of atrous convolution" },
      { t: "p", text: String.raw`A $k \times k$ kernel with dilation rate $r$ covers $k_{\text{eff}} = k + (k-1)(r-1)$ input positions per side with only $k^2$ weights. For ASPP's $3 \times 3$ branches at rates 6, 12, 18 this is 13, 25 and 37 feature cells — at output stride 16, that is 208, 400 and 592 input pixels. The feature-map size for input size $n$ at output stride $OS$ is $\lceil n/OS \rceil$; keeping $OS = 8$ instead of 32 multiplies the area of the last feature maps, and hence their compute, by 16.` },
      { t: "h", text: "5. U-Net feature-map sizes" },
      { t: "p", text: String.raw`Each unpadded 3×3 convolution shrinks a map by 2 pixels; each level applies two (−4), pooling halves, and the up-convolution doubles. From a $572 \times 572$ input: $572 \to 568 \xrightarrow{\text{pool}} 284 \to 280 \to 140 \to 136 \to 68 \to 64 \to 32 \to 28$ (bottleneck). Going up: $28 \to 56$, concatenate with the encoder's 64-pixel map **cropped** by 4 px per side, two convs $\to 52$; then $104 \to 100$, $200 \to 196$, $392 \to 388$. The output ($388 \times 388$) covers the central part of the input; the 92-pixel margin on each side is context. With padded convolutions every level simply halves and doubles.` },
      { t: "h", text: "6. Bilinear interpolation and RoIAlign" },
      { t: "p", text: String.raw`For a real-valued position $(y, x)$ with $y_0 = \lfloor y \rfloor$, $x_0 = \lfloor x \rfloor$, $l_y = y - y_0$, $l_x = x - x_0$:` },
      { t: "math", tex: String.raw`f(y,x) = (1-l_y)(1-l_x)f_{y_0,x_0} + (1-l_y)l_x f_{y_0,x_0+1} + l_y(1-l_x) f_{y_0+1,x_0} + l_y l_x f_{y_0+1,x_0+1}` },
      { t: "p", text: String.raw`RoIAlign maps the box $(x_1, y_1, x_2, y_2)$ to feature coordinates (multiply by the spatial scale $1/\text{stride}$; with the \"aligned\" convention subtract 0.5 so that pixel centres sit at integer + 0.5), divides it into $P \times P$ bins of real-valued size $b_w = (x_2 - x_1)/P$, $b_h = (y_2 - y_1)/P$, samples $s \times s$ points per bin at $x_1 + (j + (b + 0.5)/s)\,b_w$ (similarly for $y$), and averages them. Because the operation is a weighted sum of feature values with weights that depend continuously on the box, it is differentiable with respect to the features (and almost everywhere with respect to the box coordinates).` },
      { t: "p", text: "Worked example: feature map $f_{r,c} = 6r + c$ (6×6), box $(1.3, 0.7, 4.9, 3.6)$, stride 1, output 2×2, $s = 2$." },
      { t: "steps", items: [
        { title: "Shift by 0.5 (aligned)", tex: String.raw`(x_1, y_1, x_2, y_2) = (0.8,\ 0.2,\ 4.4,\ 3.1),\quad b_w = 3.6/2 = 1.8,\quad b_h = 2.9/2 = 1.45` },
        { title: "Sample points of bin (0, 0)", tex: String.raw`x \in \{0.8 + 0.25(1.8),\ 0.8 + 0.75(1.8)\} = \{1.25,\ 2.15\},\quad y \in \{0.2 + 0.25(1.45),\ 0.2 + 0.75(1.45)\} = \{0.5625,\ 1.2875\}` },
        { title: "One bilinear sample", text: "At $(y, x) = (0.5625, 1.25)$: neighbours $f_{0,1} = 1, f_{0,2} = 2, f_{1,1} = 7, f_{1,2} = 8$, $l_y = 0.5625$, $l_x = 0.25$.", tex: String.raw`0.4375(0.75)(1) + 0.4375(0.25)(2) + 0.5625(0.75)(7) + 0.5625(0.25)(8) = 4.625` },
        { title: "Average the four samples", text: "Because $f$ is linear, the average equals $f$ at the bin centre $(0.925, 1.7)$:", tex: String.raw`6(0.925) + 1.7 = 7.25` },
        { title: "RoI pooling for comparison", text: "Rounding the box to $(1, 1, 5, 4)$ and max-pooling integer bins gives", tex: String.raw`\begin{pmatrix}15 & 17\\ 27 & 29\end{pmatrix}\quad\text{vs. RoIAlign}\quad\begin{pmatrix}7.25 & 9.05\\ 15.95 & 17.75\end{pmatrix}` },
      ] },
      { t: "p", text: "RoI pooling's output is both shifted (by rounding) and biased toward each bin's bottom-right corner (by the max); RoIAlign's values are exactly the features at the bin centres." },
    ],
    visual: [
      { t: "p", text: "The two building blocks that turn a classifier into a segmentation network are learned upsampling and dilated convolution. Explore both in the convolution-variants simulator:" },
      { t: "sim", id: "conv-variants", caption: "Dilated tab: DeepLab's atrous convolution. Transposed tab: the learned upsampling used by FCN and U-Net decoders." },
      { t: "list", items: [
        "**Dilated tab.** Set dilation to 2, 3, 4 and watch the effective kernel grow ($3 + 2(d-1)$) with the same 9 weights. Read off the receptive field of three stacked layers with rates 1, 2, 4 — this is how DeepLab keeps a large field of view without pooling.",
        "**Dilated tab, padding.** Switch to 'same' padding ($p = d$): the output keeps the input size — exactly what you want when you have removed the backbone's last strides.",
        "**Transposed tab.** Use stride 2: each input value is scattered into a 2×2 (or larger) footprint of the output — the decoder's 'up-convolution'. With kernel 4 and stride 2, look for uneven overlap, the origin of checkerboard artefacts.",
      ] },
      { t: "p", text: "Mask quality is judged with the same IoU that the detection simulator computes for boxes — the only difference is that areas are counted in pixels of the mask rather than rectangles:" },
      { t: "sim", id: "detection", caption: "IoU tab: the same intersection-over-union formula used for mIoU and for matching segments in panoptic quality." },
    ],
    scratch: [
      { t: "p", text: "The evaluation and geometry primitives of segmentation in NumPy: a confusion matrix with pixel accuracy and per-class IoU / mIoU, panoptic quality with SQ and RQ, bilinear RoIAlign and RoI pooling (checked against `torchvision.ops.roi_align` and `roi_pool`), the original U-Net's shape arithmetic, and class weights for imbalanced cross-entropy." },
      { t: "code", lang: "python", title: "segmentation_from_scratch.py", code: SEG_NUMPY, output: SEG_NUMPY_OUT },
      { t: "list", items: [
        "`confusion` uses the trick `bincount(gt * C + pred)` to fill the whole confusion matrix in one vectorised call; mIoU is then three lines of arithmetic. In practice accumulate the matrix over the whole dataset before dividing — averaging per-image mIoUs gives a different (and noisier) number.",
        "`panoptic_quality` exploits the uniqueness of IoU > 0.5 matches: once a prediction matches a ground-truth segment we can stop searching.",
        "`roi_align` reproduces torchvision to $10^{-14}$ over 50 random boxes with spatial scale 1/4 (a stride-4 feature map), sampling ratio 2 and `aligned=True`; `roi_pool` matches exactly.",
        "The U-Net arithmetic reproduces the 572 → 388 sizes of the original paper and the crop needed at each skip connection.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch: U-Net with and without skips", blocks: [
          { t: "p", text: "A small U-Net trained from scratch on synthetic 32×32 images: randomly coloured filled discs and squares (class 1, 'blob') and thin straight lines about 1.5 px wide (class 2, 'line') on noisy, tinted backgrounds with faint distractor smudges. The class is determined by *shape*, not colour. The same network is trained twice with identical seeds — once with concatenation skip connections, once with the skips removed (the decoder then sees only the upsampled bottleneck) — and evaluated on 150 held-out images." },
          { t: "code", lang: "python", title: "unet_skip_ablation.py", code: UNET, output: UNET_OUT },
          { t: "list", items: [
            "**Without skips the thin lines vanish**: their IoU is 0.0 — the model never predicts the class. A line 1.5 px wide cannot survive two 2× poolings to an 8×8 bottleneck and be redrawn at the right place by upsampling alone. With skips, the full-resolution encoder features carry it straight to the output: line IoU 0.84, mIoU 0.936 vs 0.624.",
            "Blobs (large regions) are found by both models (IoU 0.972 vs 0.912); the difference there is in the boundaries: the boundary-pixel error rate is 3.9% with skips vs 26.2% without, while interior pixels are almost perfect for both.",
            "Pixel accuracy hides the failure: 0.963 for the model that misses an entire class. Lines are only 2.9% of pixels.",
            "Lower cross-entropy does not mean better segmentation: the no-skip model ends with a *lower* average training loss (0.131 vs 0.236) yet far lower mIoU. The loss is an average over all pixels dominated by easy background, and its value depends on calibration; always evaluate with IoU per class.",
            "Removing the skips saves only 6,480 parameters (66,495 vs 60,015): the gain comes from the information pathway, not from capacity.",
          ] },
          { t: "h", text: "torchvision's DeepLabV3 and Mask R-CNN (random init, no download)" },
          { t: "code", lang: "python", title: "torchvision_segmentation_models.py", code: TVSEG, output: TVSEG_OUT },
          { t: "p", text: "torchvision's DeepLabV3 replaces the strides of ResNet-50's last two stages by dilation, giving output stride 8 (a 128×160 input yields a 16×20 map with 2048 channels), and its ASPP uses rates 12, 24, 36 — the doubled rates appropriate for output stride 8. Mask R-CNN adds a 2.6M-parameter mask head fed by 14×14 RoIAlign features; in eval mode each detection carries a soft mask already pasted into full-image coordinates, shape (N, 1, H, W), to be thresholded at 0.5." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same small U-Net in Keras (channels-last). `Concatenate` implements the skip connections; `MeanIoU` with `sparse_y_pred=False` takes the argmax of the logits for you." },
          { t: "code", lang: "python", title: "unet_keras.py (not executed in CI)", code: SEG_KERAS, output: "Prints the model summary: input (32, 32, 3), two encoder levels, an 8×8 bottleneck with 48 channels, two Conv2DTranspose + Concatenate decoder levels and a (32, 32, 3) logit output. Train with model.fit(images, masks) where masks are integer class maps of shape (N, 32, 32)." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Model", "Task", "Key idea", "Advantages", "Disadvantages"], rows: [
        ["FCN-32s/16s/8s (Long et al., 2015)", "Semantic", "Convolutionalised classifier + upsampling + skip fusion of scores", "Simple; any input size; end-to-end", "Coarse boundaries (especially 32s)"],
        ["U-Net (Ronneberger et al., 2015)", "Semantic (esp. biomedical)", "Symmetric encoder–decoder with concatenation skips", "Sharp boundaries; works with little data; easy to adapt (3D, nnU-Net)", "Full-resolution decoder is memory-hungry"],
        ["SegNet", "Semantic", "Decoder upsamples with stored max-pool indices", "Memory-efficient", "Less detail than full skip features"],
        ["DeepLab v1–v3+ (Chen et al.)", "Semantic", "Atrous convolution, ASPP, (v1/v2) CRF, (v3+) decoder", "Large context at moderate resolution; strong benchmarks", "Dilated high-res maps are expensive"],
        ["PSPNet (Zhao et al., 2017)", "Semantic", "Pyramid pooling module for global context", "Good scene-level consistency", "Coarse pooled context"],
        ["Mask R-CNN (He et al., 2017)", "Instance", "Faster R-CNN + per-RoI mask head + RoIAlign", "Accurate instances; extends to keypoints", "Masks limited to 28×28 per RoI; two-stage cost"],
        ["Panoptic FPN, Mask2Former", "Panoptic (and all)", "Merge semantic + instance heads / set prediction of masks", "One model for all three tasks", "Complex training; heavy compute"],
      ] },
      { t: "list", items: [
        "**Upsampling choices**: bilinear interpolation (no parameters, then a conv), transposed convolution (learned, may produce checkerboard artefacts), unpooling with indices (SegNet), pixel shuffle.",
        "**Skip fusion choices**: concatenation (U-Net; the decoder learns how to combine) vs addition (FCN, FPN; cheaper, requires equal channels).",
        "**Promptable segmentation**: the Segment Anything Model (Kirillov et al., 2023) predicts masks for points or boxes given as prompts, trained on a very large automatically annotated mask dataset.",
        "**3D segmentation** (volumetric CT/MRI) uses 3D convolutions in the same U-Net pattern.",
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Medical imaging**: organ and tumour segmentation in CT and MRI, cell and nucleus segmentation in microscopy (U-Net's original domain), retinal vessel segmentation. Volumes and boundaries matter clinically, so masks — not boxes — are required.",
        "**Autonomous driving**: drivable area, lane markings and per-pixel scene parsing (semantic), plus instances of vehicles and pedestrians (panoptic).",
        "**Remote sensing**: land-cover maps, building footprints, flood extent, crop fields from satellite and aerial imagery.",
        "**Photo and video editing**: background removal and portrait mode, object cut-out and selection tools.",
        "**Robotics and industry**: grasp planning from object masks, defect segmentation on surfaces, sorting on conveyor belts.",
        "**Agriculture**: segmenting plants, weeds and fruit for targeted spraying and yield estimation.",
      ] },
    ],
    tradeoffs: [
      { t: "h", text: "Strengths" },
      { t: "list", items: [
        "Pixel-precise output: shape, area and boundaries, not just location.",
        "Fully convolutional models handle arbitrary input sizes and share computation across all pixels.",
        "Encoder–decoders with skips train well even on small datasets (with augmentation).",
      ] },
      { t: "h", text: "Limitations and trade-offs" },
      { t: "list", items: [
        "**Annotation cost**: per-pixel masks are far more expensive to label than boxes or image labels; label noise concentrates at boundaries.",
        "**Resolution vs compute**: high-resolution decoders and dilated backbones multiply memory and FLOPs (output stride 8 vs 32 → 16× the area).",
        "**Class imbalance**: small structures are swamped by background in the loss; needs weighting, Dice, or sampling.",
        "**Semantic models cannot count**: touching objects of one class merge; instance or panoptic methods are needed.",
        "**Mask R-CNN masks are low resolution** (28×28 per object), so large objects get soft, blobby edges.",
        "**Metrics differ by task**: pixel accuracy, mIoU, mask AP, PQ are not interchangeable.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Reporting pixel accuracy on imbalanced data.** A model that ignores a whole class can score 96% (as in our no-skip run). Report per-class IoU and mIoU.",
        "**Averaging mIoU per image instead of accumulating the confusion matrix over the dataset.** Images where a class is absent give 0/0 or inflate variance. Accumulate counts, then divide once.",
        "**Resizing masks with bilinear interpolation.** Interpolating class indices creates invalid in-between labels (class 1.5). Use nearest-neighbour for label maps (bilinear for images and soft masks).",
        "**Applying different geometric augmentation to image and mask** (e.g. a random crop drawn twice). Use one transform call on both.",
        "**Forgetting the ignore label** (e.g. 255 for 'void' pixels). Use `ignore_index` in the loss and exclude them from metrics.",
        "**Using softmax over classes for Mask R-CNN masks.** The paper's design is a per-class sigmoid with BCE on the ground-truth class only; a softmax couples classes and hurts mask quality.",
        "**Mismatched sizes at skip connections** with odd input sizes: after pooling 33 → 16 and upsampling → 32, concatenation fails. Pad inputs to a multiple of $2^{\\text{levels}}$ or crop/interpolate.",
        "**Using RoI pooling for masks** — quantisation misaligns features by up to a stride; use RoIAlign.",
        "**Judging training by the loss value alone**: CE can fall while a rare class is never predicted. Track per-class IoU on validation data.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Distinguish semantic, instance and panoptic segmentation.", a: "Semantic: a class label per pixel, no notion of individual objects. Instance: a separate mask (with class and score) for each countable object, background stuff ignored, masks may overlap. Panoptic: every pixel gets a class and, for 'things', an instance id; 'stuff' classes form one region each; no overlaps." },
      { t: "qa", level: "beginner", q: "Why is pixel accuracy a poor metric for segmentation?", a: "Because classes are highly imbalanced: background often covers most pixels, so a model predicting only background gets high pixel accuracy. mIoU computes IoU = TP/(TP+FP+FN) per class and averages over classes, so every class counts equally." },
      { t: "qa", level: "beginner", q: "What makes a network 'fully convolutional' and why does that matter for segmentation?", a: "It contains no fully connected layers that require a fixed input size — only convolutions, pooling and upsampling — so it maps an image of any size to a spatial map of predictions, one per location, computing all pixels in one pass with shared weights." },
      { t: "qa", level: "intermediate", q: "What do U-Net's skip connections provide, and why concatenate rather than add?", a: "They pass high-resolution encoder features (edges, precise locations) to the decoder, which otherwise only has coarse upsampled bottleneck features — essential for thin structures and boundaries (in our run, thin lines go from IoU 0 to 0.84). Concatenation keeps both feature sets separate and lets the following convolutions learn how to combine them; addition forces them into the same channel space." },
      { t: "qa", level: "intermediate", q: "How does atrous convolution help semantic segmentation? What does ASPP add?", a: "Dilating the kernels of the later stages (instead of striding) keeps feature maps at output stride 8 or 16 while preserving the receptive field and the pretrained weights, so predictions are less coarse. ASPP applies parallel atrous convolutions with several rates plus image-level pooling to the same features, capturing context at multiple scales for objects of different sizes." },
      { t: "qa", level: "intermediate", q: "Define panoptic quality and explain SQ and RQ.", a: "PQ = Σ_TP IoU / (|TP| + ½|FP| + ½|FN|), where a predicted and a ground-truth segment of the same class match if IoU > 0.5 (unique since segments do not overlap). It factors into SQ = mean IoU of matched segments (how well matched segments are delineated) and RQ = |TP| / (|TP| + ½|FP| + ½|FN|), an F1 score (how well segments are detected). PQ is averaged over classes." },
      { t: "qa", level: "advanced", q: "Explain RoIAlign and why Mask R-CNN needed it.", a: "RoI pooling rounds the RoI to the feature grid and splits it into integer bins, misaligning the extracted features by up to a stride (e.g. 16 px). RoIAlign keeps coordinates continuous: for each output bin it samples a few regular points, computes each by bilinear interpolation of the 4 nearest feature cells, and averages (or maxes) them. Pixel-level masks are sensitive to such misalignments; He et al. report large mask-AP improvements from this change alone." },
      { t: "qa", level: "advanced", q: "Your segmentation model has high mIoU on large classes but misses thin structures. What would you change?", a: "Increase resolution where it matters: add or strengthen skip connections from early layers, reduce output stride (dilation instead of stride), use a decoder (DeepLabv3+ style), train at higher input resolution. Rebalance the loss (class weights, Dice/Tversky or boundary-aware losses), oversample images containing the structure, and check that label resizing used nearest-neighbour. Evaluate with per-class IoU and boundary metrics." },
      { t: "qa", level: "advanced", q: "Why does Mask R-CNN predict one binary mask per class with a sigmoid instead of a multi-class softmax mask?", a: "Decoupling mask prediction from classification: the box head decides the class, and the mask head only has to answer 'which pixels in this box belong to an object of class k' without competing against other classes. He et al. found a per-pixel softmax over classes clearly worse; the loss is applied only to the ground-truth class's mask." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "For one class, TP = 30, FP = 10, FN = 20 pixels. What is its IoU? (2 decimals)", answer: 0.5, tolerance: 0.005, explain: "IoU = TP/(TP+FP+FN) = 30/60 = 0.5." },
        { kind: "numeric", q: "TP = 6 matched segments with IoUs summing to 4.8, FP = 2, FN = 4. What is PQ? (3 decimals)", answer: 0.533, tolerance: 0.002, explain: "SQ = 4.8/6 = 0.8; RQ = 6/(6 + 1 + 2) = 0.667; PQ = 0.8 × 0.667 = 0.533 (= 4.8/9)." },
        { kind: "numeric", q: "A 3×3 kernel with dilation rate 12 covers how many input positions per side?", answer: 25, tolerance: 0, explain: "$k + (k-1)(r-1) = 3 + 2 \\times 11 = 25$." },
        { q: "Which operation should be used to resize a ground-truth label map?", options: ["Bilinear interpolation", "Bicubic interpolation", "Nearest-neighbour interpolation", "Average pooling"], answer: 2, explain: "Label indices are categorical; interpolating them creates invalid labels." },
        { q: "What is the main reason RoIAlign improves masks over RoI pooling?", options: ["It uses more parameters", "It avoids rounding the RoI and bin boundaries, sampling features bilinearly at exact positions", "It uses average instead of max", "It increases the output size to 28×28"], answer: 1, explain: "Quantisation misaligns features by up to one stride; RoIAlign removes it." },
        { q: "In our ablation, why did the U-Net without skip connections get IoU 0 on the 'line' class?", options: ["Lines were absent from the test set", "The 1.5-px lines cannot be reconstructed from an 8×8 bottleneck by upsampling alone", "The model had too few parameters", "Cross-entropy cannot learn thin structures"], answer: 1, explain: "The information about exact line position is lost by downsampling; skips restore it (IoU 0.84) with only 6.5k extra parameters." },
        { kind: "numeric", q: "Unpadded U-Net: a level's feature map is 136×136 and the upsampled decoder map is 104×104. How many pixels must be cropped from each side of the encoder map?", answer: 16, tolerance: 0, explain: "(136 − 104)/2 = 16." },
        { q: "Which statement about panoptic segmentation is correct?", options: ["Stuff classes get one instance id per object", "Segments may overlap", "Every pixel receives a class label and, for thing classes, an instance id", "It ignores background classes"], answer: 2, explain: "Panoptic output is a non-overlapping partition: things are separated into instances, stuff is one region per class." },
      ] },
      { t: "exercise", title: "Metrics by hand", prompt: "A 4×4 image has ground truth with 10 background pixels (class 0) and 6 object pixels (class 1). The prediction marks 8 pixels as class 1, of which 5 are truly class 1. Compute pixel accuracy, IoU of both classes and mIoU. Then compute PQ if this were a single 'thing' segment on each side.", hint: "Build the 2×2 confusion matrix first.", solution: [
        { t: "steps", items: [
          { title: "Confusion matrix", tex: String.raw`N = \begin{pmatrix} 7 & 3 \\ 1 & 5 \end{pmatrix}\ (\text{rows = truth})` },
          { title: "Pixel accuracy", tex: String.raw`(7 + 5)/16 = 0.75` },
          { title: "IoUs", tex: String.raw`\mathrm{IoU}_0 = \frac{7}{10 + 8 - 7} = 0.6364,\quad \mathrm{IoU}_1 = \frac{5}{6 + 8 - 5} = 0.5556,\quad \mathrm{mIoU} = 0.5960` },
          { title: "PQ", tex: String.raw`\mathrm{IoU} = 0.5556 > 0.5 \Rightarrow TP = 1, FP = FN = 0,\ \mathrm{SQ} = 0.5556,\ \mathrm{RQ} = 1,\ \mathrm{PQ} = 0.5556` },
        ] },
      ] },
      { t: "exercise", title: "Class-weighted loss and a Dice term", prompt: "Modify `unet_skip_ablation.py`: (a) compute median-frequency class weights from `freq` and pass them as `weight=` to `F.cross_entropy`; (b) add a soft Dice loss averaged over the two foreground classes. Train the *no-skip* model with each loss and report the line-class IoU. Does re-weighting fix the missing class?", hint: "Soft Dice per class: $1 - (2\\sum p\\,g + \\epsilon)/(\\sum p + \\sum g + \\epsilon)$ with $p$ = softmax probabilities, $g$ = one-hot targets.", solution: [
        { t: "code", lang: "python", code: String.raw`w = (freq.median() / freq)                          # median-frequency balancing
def dice_loss(logits, y, eps=1.0):
    p = logits.softmax(1)
    g = F.one_hot(y, N_CLS).permute(0, 3, 1, 2).float()
    inter = (p * g).sum((0, 2, 3)); denom = p.sum((0, 2, 3)) + g.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps))[1:].mean()   # foreground classes only

loss = F.cross_entropy(out, Y_tr[b], weight=w) + dice_loss(out, Y_tr[b])` },
        { t: "p", text: "Re-weighting makes the no-skip model *predict* the line class (it is now expensive to ignore), so recall on lines rises; but without high-resolution information it cannot place them precisely, so the line IoU stays far below the U-Net with skips. Loss design fixes priorities; architecture fixes information flow. Run it to see the exact numbers on your machine." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Semantic: class per pixel. Instance: mask per object. Panoptic: class + instance id per pixel, things and stuff, no overlaps.",
        String.raw`Metrics: pixel accuracy (misleading under imbalance); $\mathrm{IoU}_c = TP/(TP+FP+FN)$ and mIoU from a dataset-level confusion matrix; $\mathrm{PQ} = \mathrm{SQ} \times \mathrm{RQ}$ with matches at IoU > 0.5.`,
        "FCN: convolutionalised classifier + upsampling + skip fusion (32s → 16s → 8s).",
        "U-Net: encoder–decoder with concatenation skips; original 572 → 388 with unpadded convs. Skips carry 'where' (our run: thin-line IoU 0 → 0.84, mIoU 0.624 → 0.936).",
        String.raw`DeepLab: atrous convolution keeps output stride 8/16; $k_{\text{eff}} = k + (k-1)(r-1)$; ASPP gathers multi-scale context.`,
        "Mask R-CNN: Faster R-CNN + per-class 28×28 sigmoid masks on RoIAlign features. RoIAlign = bilinear sampling, no rounding.",
        "Losses: pixel CE (with class weights / ignore index), Dice or Jaccard for imbalance, often combined.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Faces, Pose, Optical Flow, Super-Resolution and Captioning](/learn/m12/vision-tasks) — dense prediction beyond class labels: keypoint heatmaps, flow fields and high-resolution reconstruction reuse encoder–decoder designs.",
        "[Transformer families](/learn/m11/transformer-families) — ViT and Swin backbones underpin modern segmentation models such as Mask2Former.",
        "[Advanced losses](/learn/m05/advanced-losses) — revisit Dice, Jaccard and their gradients now that you have seen where they are used.",
      ] },
    ],
  },
};

export default lesson;
