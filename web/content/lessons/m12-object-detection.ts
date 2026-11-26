import type { Lesson } from "@/lib/types";

const DET_NUMPY = String.raw`import numpy as np

# ---------------- 1. box formats: corners (x1, y1, x2, y2) <-> centre/size (cx, cy, w, h) ----------------
def xyxy_to_cxcywh(b):
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)

def cxcywh_to_xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)

# ---------------- 2. pairwise IoU, fully vectorised: (N,4) x (M,4) -> (N,M) ----------------
def box_iou(a, b):
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])     # top-left of the intersection
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])     # bottom-right of the intersection
    wh = np.clip(rb - lt, 0, None)                       # zero if the boxes do not overlap
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)

# ---------------- 3. greedy non-maximum suppression ----------------
def nms(boxes, scores, thr):
    order = np.argsort(-scores, kind="stable")          # highest score first
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))                              # the best remaining box survives
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= thr]                   # drop everything overlapping it too much
    return np.array(keep)

# ---------------- 4. anchors on a feature-map grid ----------------
def make_anchors(fm_h, fm_w, stride, scales, ratios):
    """One anchor per (cell, scale, ratio); ratio r = h / w, area = scale^2."""
    shapes = [(s / np.sqrt(r), s * np.sqrt(r)) for s in scales for r in ratios]   # (w, h)
    ys, xs = np.meshgrid(np.arange(fm_h), np.arange(fm_w), indexing="ij")
    centres = np.stack([(xs.ravel() + 0.5) * stride, (ys.ravel() + 0.5) * stride], axis=1)
    out = [[cx, cy, w, h] for cx, cy in centres for (w, h) in shapes]
    return cxcywh_to_xyxy(np.array(out))

def match_anchors(anchors, gt, pos_thr=0.5, neg_thr=0.4):
    """Label each anchor: 1 positive, 0 negative, -1 ignore; return matched GT index."""
    iou = box_iou(anchors, gt)                           # (A, G)
    best_gt = iou.argmax(axis=1)
    best_iou = iou.max(axis=1)
    labels = np.full(len(anchors), -1)
    labels[best_iou < neg_thr] = 0
    labels[best_iou >= pos_thr] = 1
    # every GT gets at least its single best anchor, even if that IoU is below pos_thr
    for g, a in enumerate(iou.argmax(axis=0)):
        labels[a] = 1
        best_gt[a] = g
    return labels, best_gt, best_iou

# ---------------- 5. regression targets relative to an anchor ----------------
def encode(gt, anc):
    g, a = xyxy_to_cxcywh(gt), xyxy_to_cxcywh(anc)
    return np.stack([(g[:, 0] - a[:, 0]) / a[:, 2], (g[:, 1] - a[:, 1]) / a[:, 3],
                     np.log(g[:, 2] / a[:, 2]), np.log(g[:, 3] / a[:, 3])], axis=1)

def decode(t, anc):
    a = xyxy_to_cxcywh(anc)
    return cxcywh_to_xyxy(np.stack([a[:, 0] + t[:, 0] * a[:, 2], a[:, 1] + t[:, 1] * a[:, 3],
                                    a[:, 2] * np.exp(t[:, 2]), a[:, 3] * np.exp(t[:, 3])], axis=1))

# ---------------- 6. average precision (all-point interpolation) for one class ----------------
def average_precision(det_boxes, det_scores, det_img, gt_boxes, gt_img, iou_thr=0.5):
    order = np.argsort(-det_scores, kind="stable")
    used = np.zeros(len(gt_boxes), bool)                # each GT may be matched only once
    tp = np.zeros(len(order))
    for k, d in enumerate(order):
        cand = np.where(gt_img == det_img[d])[0]
        if cand.size:
            ious = box_iou(det_boxes[d:d + 1], gt_boxes[cand])[0]
            j = ious.argmax()
            if ious[j] >= iou_thr and not used[cand[j]]:
                used[cand[j]] = True
                tp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / len(gt_boxes)
    precision = ctp / (ctp + cfp)
    # precision envelope: make it monotonically non-increasing from the right
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), precision, recall

# ======================= demo =======================
gt = np.array([[20., 20., 60., 70.], [70., 30., 110., 60.]])          # two objects in a 128x128 image
print("IoU(gt0, gt shifted by 10px):", round(float(box_iou(gt[:1], gt[:1] + 10)[0, 0]), 4))

anchors = make_anchors(4, 4, stride=32, scales=[32, 64], ratios=[0.5, 1.0, 2.0])
labels, matched, best = match_anchors(anchors, gt)
print(f"anchors: {len(anchors)}  positive {np.sum(labels == 1)}  negative {np.sum(labels == 0)}  ignored {np.sum(labels == -1)}")
pos = np.where(labels == 1)[0]
t = encode(gt[matched[pos]], anchors[pos])
for a, tt in zip(pos, t):
    print(f"  anchor {a:2d} {anchors[a].round(1)}  IoU {best[a]:.3f} -> gt {matched[a]}  t = {tt.round(3)}")
print("decode(encode(gt)) == gt:", np.allclose(decode(t, anchors[pos]), gt[matched[pos]]))

rng = np.random.default_rng(0)
cand = np.repeat(gt, 5, axis=0) + rng.normal(0, 4, (10, 4))           # 5 noisy candidates per object
scores = rng.uniform(0.3, 1.0, 10)
keep = nms(cand, scores, 0.5)
print("NMS keeps", keep.tolist(), "with scores", scores[keep].round(3).tolist())

# AP on a toy set: 3 images, 4 GT boxes, 6 scored detections
gt_b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 10, 10], [5, 5, 15, 15]], float)
gt_i = np.array([0, 0, 1, 2])
dt_b = np.array([[1, 1, 10, 10], [0, 0, 9, 11], [21, 19, 30, 31], [30, 30, 40, 40], [0, 1, 10, 10], [40, 40, 50, 50]], float)
dt_s = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
dt_i = np.array([0, 0, 0, 1, 1, 2])
ap, p, r = average_precision(dt_b, dt_s, dt_i, gt_b, gt_i)
print("precision:", p.round(3).tolist())
print("recall:   ", r.round(3).tolist())
print(f"AP@0.5 = {ap:.4f}")

# ---------------- check against torchvision's reference implementations ----------------
import torch
from torchvision.ops import box_iou as tv_iou, nms as tv_nms
xy = rng.uniform(0, 100, (200, 2)); wh = rng.uniform(5, 40, (200, 2))
B = np.concatenate([xy, xy + wh], axis=1); S = rng.uniform(0, 1, 200)
print("max |IoU - torchvision| :", f"{np.abs(box_iou(B, B) - tv_iou(torch.tensor(B), torch.tensor(B)).numpy()).max():.1e}")
print("NMS identical to torchvision:", np.array_equal(nms(B, S, 0.5), tv_nms(torch.tensor(B), torch.tensor(S), 0.5).numpy()))`;

const DET_NUMPY_OUT = String.raw`IoU(gt0, gt shifted by 10px): 0.4286
anchors: 96  positive 2  negative 88  ignored 6
  anchor 32 [36.7 25.4 59.3 70.6]  IoU 0.501 -> gt 0  t = [-0.354 -0.066  0.57   0.1  ]
  anchor 36 [ 57.4  36.7 102.6  59.3]  IoU 0.497 -> gt 1  t = [ 0.221 -0.133 -0.123  0.282]
decode(encode(gt)) == gt: True
NMS keeps [5, 2] with scores [0.923, 0.716]
precision: [1.0, 0.5, 0.667, 0.5, 0.6, 0.5]
recall:    [0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
AP@0.5 = 0.5667
max |IoU - torchvision| : 0.0e+00
NMS identical to torchvision: True`;

const DET_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms

torch.manual_seed(0)
torch.set_num_threads(1)                              # tiny model: one thread is fastest and deterministic
rng = np.random.default_rng(0)
S, CLASSES = 48, ["rect", "circle"]                   # 48x48 canvases, 2 object classes (+ background)

def overlap(a, b):                                    # IoU of two xyxy boxes (numpy)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)

# ---------------- synthetic dataset: coloured rectangles and circles on noisy backgrounds ----------------
def make_image():
    img = rng.uniform(0.0, 0.35, (3, S, S)) + rng.normal(0, 0.05, (3, S, S))
    boxes, labels = [], []
    yy, xx = np.mgrid[0:S, 0:S] + 0.5
    for _ in range(rng.integers(1, 4)):                # 1-3 objects per image
        for _try in range(20):                         # rejection-sample a box that barely overlaps the others
            cls = int(rng.integers(0, 2))
            w = rng.uniform(8, 22); h = w if cls == 1 else rng.uniform(8, 22)
            x1, y1 = rng.uniform(0, S - w), rng.uniform(0, S - h)
            b = np.array([x1, y1, x1 + w, y1 + h])
            if not boxes or all(overlap(b, o) < 0.1 for o in boxes):
                break
        colour = rng.uniform(0.5, 1.0, 3)
        if cls == 0:
            mask = (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
        else:
            mask = (xx - (b[0] + b[2]) / 2) ** 2 + (yy - (b[1] + b[3]) / 2) ** 2 <= (w / 2) ** 2
        img[:, mask] = colour[:, None]
        boxes.append(b); labels.append(cls + 1)         # label 0 is reserved for background
    return img.astype(np.float32), np.array(boxes, np.float32), np.array(labels)

train = [make_image() for _ in range(800)]
test = [make_image() for _ in range(300)]

# ---------------- anchors: 6x6 grid (stride 8), 2 scales x 3 aspect ratios = 6 per cell ----------------
STRIDE, FM = 8, 6
shapes = [(s / r ** 0.5, s * r ** 0.5) for s in (11, 19) for r in (0.5, 1.0, 2.0)]   # (w, h), r = h/w
A = len(shapes)
cy, cx = torch.meshgrid(torch.arange(FM) + 0.5, torch.arange(FM) + 0.5, indexing="ij")
anc_c = torch.tensor([[x * STRIDE, y * STRIDE, w, h] for x, y in zip(cx.flatten().tolist(), cy.flatten().tolist()) for w, h in shapes])
anchors = torch.cat([anc_c[:, :2] - anc_c[:, 2:] / 2, anc_c[:, :2] + anc_c[:, 2:] / 2], 1)   # (216, 4) xyxy

def encode(gt, anc_c):                                  # gt in xyxy, anchors in cxcywh
    g = torch.cat([(gt[:, :2] + gt[:, 2:]) / 2, gt[:, 2:] - gt[:, :2]], 1)
    return torch.cat([(g[:, :2] - anc_c[:, :2]) / anc_c[:, 2:], torch.log(g[:, 2:] / anc_c[:, 2:])], 1)

def decode(t, anc_c):
    c = anc_c[:, :2] + t[:, :2] * anc_c[:, 2:]; wh = anc_c[:, 2:] * torch.exp(t[:, 2:])
    return torch.cat([c - wh / 2, c + wh / 2], 1)

def targets(boxes, labels, pos_thr=0.5, neg_thr=0.4):
    iou = box_iou(anchors, torch.tensor(boxes))         # (216, n_objects)
    best, idx = iou.max(1)
    idx[iou.argmax(0)] = torch.arange(len(boxes)); best[iou.argmax(0)] = 1.0   # force-match each GT
    cls = torch.tensor(labels)[idx]
    cls[best < pos_thr] = 0                             # background
    cls[(best >= neg_thr) & (best < pos_thr)] = -1      # ignored in the loss
    return cls, encode(torch.tensor(boxes)[idx], anc_c)

X_tr = torch.tensor(np.stack([d[0] for d in train]))
T_tr = [targets(d[1], d[2]) for d in train]
C_tr = torch.stack([t[0] for t in T_tr]); R_tr = torch.stack([t[1] for t in T_tr])

# ---------------- model: tiny CNN backbone + one conv head predicting (classes, box offsets) per anchor ----------------
class TinySSD(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
        def block(i, o, k=3): return nn.Sequential(nn.Conv2d(i, o, k, stride=2, padding=k // 2), nn.BatchNorm2d(o), nn.ReLU())
        self.backbone = nn.Sequential(block(3, 24, 5), block(24, 32), block(32, 64))     # 48 -> 24 -> 12 -> 6 (stride 8)
        self.ctx = nn.Sequential(nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU())  # widen receptive field
        self.cls_head = nn.Conv2d(64, A * n_cls, 3, padding=1)
        self.box_head = nn.Conv2d(64, A * 4, 3, padding=1)
        self.n_cls = n_cls
        # start by predicting "background" with probability ~0.99 everywhere (prior init, as in RetinaNet):
        # avoids a huge, unstable classification loss on the first iterations
        with torch.no_grad():
            self.cls_head.bias.view(A, n_cls)[:, 0] = 4.6
    def forward(self, x):
        f = self.ctx(self.backbone(x))
        N = x.shape[0]
        cls = self.cls_head(f).permute(0, 2, 3, 1).reshape(N, -1, self.n_cls)   # (N, 216, 3), anchor order = (y, x, shape)
        box = self.box_head(f).permute(0, 2, 3, 1).reshape(N, -1, 4)            # (N, 216, 4)
        return cls, box

def loss_fn(cls_logits, box_pred, cls_t, box_t):
    pos = cls_t > 0
    ce = F.cross_entropy(cls_logits.reshape(-1, 3), cls_t.clamp(min=0).reshape(-1), reduction="none").view_as(cls_t)
    # hard-negative mining (SSD): keep the 3*n_pos highest-loss background anchors per image
    neg_loss = ce.clone(); neg_loss[cls_t != 0] = 0
    rank = neg_loss.argsort(1, descending=True).argsort(1)
    neg = (rank < 3 * pos.sum(1, keepdim=True)) & (cls_t == 0)
    n_pos = pos.sum().clamp(min=1)
    l_cls = (ce[pos].sum() + ce[neg].sum()) / n_pos
    l_box = F.smooth_l1_loss(box_pred[pos], box_t[pos], beta=1 / 9, reduction="sum") / n_pos
    return l_cls, l_box

model = TinySSD()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
EPOCHS, BS = 15, 32
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * (len(train) // BS))
for epoch in range(EPOCHS):
    model.train(); perm = torch.randperm(len(train)); tot = [0.0, 0.0]
    for k in range(len(train) // BS):
        b = perm[k * BS:(k + 1) * BS]
        l_cls, l_box = loss_fn(*model(X_tr[b]), C_tr[b], R_tr[b])
        opt.zero_grad(); (l_cls + l_box).backward(); opt.step(); sched.step()
        tot[0] += l_cls.item(); tot[1] += l_box.item()
    if epoch % 3 == 2 or epoch == 0:
        n = len(train) // BS
        print(f"epoch {epoch + 1:2d}  cls loss {tot[0] / n:.3f}  box loss {tot[1] / n:.3f}")

# ---------------- inference: softmax -> decode -> per-class NMS ----------------
@torch.no_grad()
def detect(x, score_thr=0.05, iou_thr=0.45):
    model.eval()
    logits, t = model(x)
    out = []
    for i in range(len(x)):
        probs = logits[i].softmax(-1); boxes = decode(t[i], anc_c).clamp(0, S)
        dets = []
        for c in (1, 2):
            m = probs[:, c] > score_thr
            keep = nms(boxes[m], probs[m, c], iou_thr)
            dets += [(c, float(probs[m, c][k]), boxes[m][k]) for k in keep]
        out.append(dets)
    return out

def ap_for_class(dets, gts, c, thr):
    """All-point-interpolated AP. dets: per image list of (cls, score, box); gts: per image (boxes, labels)."""
    flat = sorted([(s, i, b) for i, d in enumerate(dets) for (cc, s, b) in d if cc == c], key=lambda z: -z[0])
    gt_c = [torch.tensor(g[0][g[1] == c]) for g in gts]
    used = [torch.zeros(len(g), dtype=torch.bool) for g in gt_c]
    n_gt = sum(len(g) for g in gt_c); tp = []
    for s, i, b in flat:
        hit = False
        if len(gt_c[i]):
            ious = box_iou(b[None], gt_c[i])[0]; j = int(ious.argmax())
            if ious[j] >= thr and not used[i][j]:
                used[i][j] = True; hit = True
        tp.append(hit)
    tp = np.array(tp, float); ctp = np.cumsum(tp)
    rec = np.concatenate([[0], ctp / n_gt, [1]]); pre = np.concatenate([[1], ctp / np.arange(1, len(tp) + 1), [0]])
    pre = np.maximum.accumulate(pre[::-1])[::-1]
    i = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[i + 1] - rec[i]) * pre[i + 1]))

X_te = torch.tensor(np.stack([d[0] for d in test]))
dets = detect(X_te)
gts = [(d[1], d[2]) for d in test]
ap50 = [ap_for_class(dets, gts, c, 0.5) for c in (1, 2)]
coco = np.mean([np.mean([ap_for_class(dets, gts, c, t) for c in (1, 2)]) for t in np.arange(0.5, 0.96, 0.05)])
print(f"test AP@0.5  rect {ap50[0]:.3f}  circle {ap50[1]:.3f}  ->  mAP@0.5 = {np.mean(ap50):.3f}")
print(f"COCO-style mAP@[.5:.95] = {coco:.3f}")
print(f"objects in test set: {sum(len(g[0]) for g in gts)}, detections kept: {sum(len(d) for d in dets)}")`;

const DET_TORCH_OUT = String.raw`parameters: 88666
epoch  1  cls loss 4.013  box loss 0.524
epoch  3  cls loss 2.034  box loss 0.275
epoch  6  cls loss 0.725  box loss 0.246
epoch  9  cls loss 0.223  box loss 0.138
epoch 12  cls loss 0.088  box loss 0.055
epoch 15  cls loss 0.066  box loss 0.026
test AP@0.5  rect 0.837  circle 0.890  ->  mAP@0.5 = 0.864
COCO-style mAP@[.5:.95] = 0.452
objects in test set: 604, detections kept: 934`;

const FRCNN = String.raw`import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

torch.manual_seed(0)
# weights=None, weights_backbone=None: random initialisation, nothing is downloaded
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"COCO configuration (91 classes): {count(fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)):,} parameters")
# our own 2-class problem (+ background); small min/max_size keeps this CPU demo fast (default resizes to 800 px)
model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=3, min_size=224, max_size=320)
print(f"total parameters      {count(model):>11,}")
print(f"  backbone (ResNet-50) {count(model.backbone.body):>11,}")
print(f"  FPN                  {count(model.backbone.fpn):>11,}")
print(f"  RPN                  {count(model.rpn):>11,}")
print(f"  RoI heads            {count(model.roi_heads):>11,}")
print("anchor sizes per FPN level:", model.rpn.anchor_generator.sizes)
print("aspect ratios per level:   ", model.rpn.anchor_generator.aspect_ratios[0])

images = [torch.rand(3, 240, 320), torch.rand(3, 200, 200)]            # a list: images may differ in size
targets = [{"boxes": torch.tensor([[30., 40., 120., 160.]]), "labels": torch.tensor([1])},
           {"boxes": torch.tensor([[10., 10., 80., 90.], [100., 50., 180., 150.]]), "labels": torch.tensor([2, 1])}]
model.train()
losses = model(images, targets)                     # training mode: a dict of the four losses
print("training-mode output keys:", sorted(losses))
model.eval()
with torch.no_grad():
    out = model(images)                             # eval mode: one dict per image
print("eval-mode output per image:", {k: tuple(v.shape) for k, v in out[0].items()})
print("box format: absolute (x1, y1, x2, y2) pixels of the original image; scores sorted descending:",
      bool((out[0]["scores"][:-1] >= out[0]["scores"][1:]).all()) if len(out[0]["scores"]) > 1 else True)`;

const FRCNN_OUT = String.raw`COCO configuration (91 classes): 41,808,406 parameters
total parameters       41,357,406
  backbone (ResNet-50)  23,508,032
  FPN                    3,344,384
  RPN                      593,935
  RoI heads             13,911,055
anchor sizes per FPN level: ((32,), (64,), (128,), (256,), (512,))
aspect ratios per level:    (0.5, 1.0, 2.0)
training-mode output keys: ['loss_box_reg', 'loss_classifier', 'loss_objectness', 'loss_rpn_box_reg']
eval-mode output per image: {'boxes': (100, 4), 'labels': (100,), 'scores': (100,)}
box format: absolute (x1, y1, x2, y2) pixels of the original image; scores sorted descending: True`;

const DET_KERAS = String.raw`import tensorflow as tf

# boxes for tf.image ops are (y1, x1, y2, x2) — note the different coordinate order
boxes = tf.constant([[10., 10., 60., 60.], [12., 14., 64., 62.], [40., 40., 90., 90.],
                     [100., 100., 150., 150.], [8., 12., 62., 58.]])
scores = tf.constant([0.9, 0.8, 0.7, 0.6, 0.5])

keep = tf.image.non_max_suppression(boxes, scores, max_output_size=10, iou_threshold=0.5)
print("hard NMS keeps:", keep.numpy())

# soft-NMS (Gaussian decay): returns indices and the decayed scores
keep_s, new_scores = tf.image.non_max_suppression_with_scores(
    boxes, scores, max_output_size=10, iou_threshold=1.0, score_threshold=0.05, soft_nms_sigma=0.5)
print("soft-NMS keeps:", keep_s.numpy(), "scores:", new_scores.numpy().round(3))`;

const lesson: Lesson = {
  id: "m12/object-detection",
  title: "Object Detection: R-CNN Family, SSD, YOLO, RetinaNet and DETR",
  summary:
    "Finding and labelling every object in an image: box formats, IoU, anchors and target assignment, box encoding, non-maximum suppression, the two-stage (R-CNN → Faster R-CNN), one-stage (YOLO, SSD, RetinaNet, anchor-free) and set-prediction (DETR) designs, detection losses, and mAP. All built from scratch and checked against torchvision, plus a single-shot detector trained on synthetic data.",
  minutes: 110,
  level: "advanced",
  prerequisites: ["m08/cnn-architectures", "m05/classification-losses", "m05/advanced-losses", "m12/transfer-learning"],
  next: ["m12/segmentation", "m12/vision-tasks", "m11/transformer-families"],
  terms: ["object-detection", "bounding-box", "iou", "anchor-box", "non-max-suppression", "region-proposal-network", "roi-pooling", "average-precision", "map", "hungarian-matching", "focal-loss", "feature-pyramid-network"],
  sections: {
    intro: [
      { t: "p", text: "**Image classification** says *what* is in an image, with one label. **Object localization** adds *where* for one object: the network also outputs a **bounding box** — four numbers for the tightest upright rectangle around it. **Object detection** drops the \"one object\" assumption. The image may hold any number of objects of any classes — zero, one, or a hundred — and the model must output a *set* of (box, class, confidence) triples: one per object, and nothing else." },
      { t: "p", text: "That small change creates three problems a classifier never meets:" },
      { t: "list", ordered: true, items: [
        "**Variable-size output**: the number of objects is unknown, but a network has a fixed number of outputs.",
        "**Matching**: during training, which prediction is responsible for which true object?",
        "**Duplicates**: many nearby predictions fire on the same object, and all but one must go.",
      ] },
      { t: "p", text: "Every detector in this lesson is a different answer to these three questions." },
      { t: "p", text: "A short history. **R-CNN** (Girshick et al., 2014) ran a CNN on ~2000 externally proposed regions. **Fast R-CNN** (Girshick, 2015) and **Faster R-CNN** (Ren, He, Girshick and Sun, 2015) learned the proposals with a region proposal network built on **anchor boxes**. One-stage detectors then predicted boxes densely in one pass — **YOLO** (Redmon et al., 2015/2016), **SSD** (Liu et al., 2016), **RetinaNet** with the focal loss (Lin et al., 2017) — followed by **anchor-free** designs (FCOS, CenterNet). Finally **DETR** (Carion et al., 2020) treats detection as direct *set prediction* with a transformer, needing neither anchors nor non-maximum suppression." },
      { t: "p", text: "Detection is used wherever a system must count, track or act on single objects (see applications below). It is also the first stage of instance segmentation ([next lesson](/learn/m12/segmentation)), pose estimation and multi-object tracking." },
      { t: "callout", kind: "key", text: "A detector = many predictions tied to reference positions (anchors, grid cells, points or learned queries) + a rule for assigning true objects to them in training + a classification loss and a box-regression loss + a step that removes duplicates (NMS, or one-to-one matching). Quality is measured by mean average precision (mAP) at one or several IoU thresholds." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Classic CNN Architectures](/learn/m08/cnn-architectures) — detectors reuse ResNet/VGG-style **backbones**. You need the idea of a feature map whose cells match regions of the input at a given **stride**.",
        "[Pooling, receptive fields](/learn/m08/pooling-receptive-field) — a cell of a stride-16 feature map \"sees\" a large patch of the image; anchors sit at these cell centres.",
        "[Classification losses](/learn/m05/classification-losses) — softmax/sigmoid cross-entropy and the **focal loss**, used for each anchor's class prediction.",
        "[Regression losses](/learn/m05/regression-losses) — the Huber / **smooth L1** loss used for box offsets.",
        "[Advanced losses](/learn/m05/advanced-losses) — where the IoU and **GIoU** box losses are derived; here we just use them.",
        "[Transfer learning](/learn/m12/transfer-learning) — real detectors start from an ImageNet-pretrained backbone (here we train from scratch on synthetic data, so nothing is downloaded).",
        "For DETR: [the Transformer architecture](/learn/m11/transformer-architecture) (encoder–decoder, cross-attention).",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A room full of spotters", text: "Picture a stadium photo cut into a grid, with a spotter behind each square holding a few cardboard frames (tall, square, wide). For each frame, a spotter answers: \"is an object of roughly this shape centred near my square?\" and \"how should I stretch and slide my frame to fit it?\". Many neighbours raise hands for the same person, so a referee goes from the most confident down and, whenever two frames cover mostly the same area, keeps only the more confident one." },
      { t: "p", text: "The spotters are **anchors** (or grid cells); their frames are **anchor boxes** of several sizes and shapes; \"stretch and slide\" is the **regression target** $(t_x, t_y, t_w, t_h)$; the referee is **non-maximum suppression** (NMS). In training, which spotters *should* raise a hand is decided by each frame's overlap with a true object, **Intersection over Union (IoU)**: good overlap → **positive**, poor → **negative** (background), borderline → **ignored**." },
      { t: "p", text: "Two-stage detectors split the job: a class-blind first stage proposes a few hundred regions that probably contain *something*; a second stage classifies each and refines its box. One-stage detectors do it all in one dense pass — faster, but swamped by background positions, the problem the focal loss solves. DETR replaces the spotters with a fixed number of learned \"queries\" that negotiate (via self-attention) so each object is claimed by exactly one. No referee needed." },
    ],
    mechanism: [
      { t: "h", text: "Box formats" },
      { t: "table", head: ["Format", "Numbers", "Used by"], rows: [
        ["xyxy (corners)", "$(x_1, y_1, x_2, y_2)$: top-left and bottom-right", "PASCAL VOC labels, torchvision ops (`box_iou`, `nms`), IoU computation"],
        ["xywh", "$(x_1, y_1, w, h)$: top-left plus size", "COCO label files"],
        ["cxcywh", "$(c_x, c_y, w, h)$: centre plus size", "Regression targets, YOLO, DETR outputs (scaled to [0, 1])"],
      ] },
      { t: "p", text: "Each conversion is one line ($c_x = (x_1+x_2)/2$, $w = x_2 - x_1$, …), but silently mixing formats is one of the most common detection bugs. Coordinates are in pixels unless stated; YOLO and DETR divide by image width and height." },
      { t: "h", text: "One training and inference step of an anchor-based detector" },
      { t: "steps", items: [
        { title: "Backbone", text: "A CNN (often with a **feature pyramid network**, FPN) turns the image into one or more feature maps with strides such as 8, 16, 32." },
        { title: "Anchors", text: "At every cell centre of every feature map, place $A$ anchor boxes (e.g. 3 sizes × 3 shapes). Their positions are fixed — not learned." },
        { title: "Head", text: "A small conv head predicts, for every anchor, $C$ (or $C+1$ with background) class scores and 4 box offsets $(t_x, t_y, t_w, t_h)$. One 3×3 conv with $A(C+4)$ output channels does this for all anchors at once." },
        { title: "Target assignment (training)", text: "Compute every anchor's IoU with every true box. Best IoU ≥ $\\tau_+$: positive for that object; < $\\tau_-$: negative; in between: ignored. Each object's highest-IoU anchor is usually forced positive, so no object lacks a predictor." },
        { title: "Encode targets", text: "For each positive anchor, describe its true box relative to the anchor: centre offsets scaled by anchor size, and log size ratios." },
        { title: "Loss", text: "Classification loss over positives and (sampled or focal-weighted) negatives, plus a box loss (smooth L1 on the offsets, or GIoU on decoded boxes) over positives only." },
        { title: "Decode (inference)", text: "Reverse the encoding to turn predicted offsets into boxes; drop boxes scoring below a small threshold (e.g. 0.05)." },
        { title: "Non-maximum suppression", text: "Per class: keep the top-scoring box, remove all remaining boxes with IoU > threshold (e.g. 0.5) against it, and repeat. Return the top-$k$ survivors." },
      ] },
      { t: "h", text: "Greedy NMS, step by step" },
      { t: "steps", items: [
        { title: "Sort", text: "Order the candidate boxes $B$ by score, highest first. Start the output list $K = \\emptyset$." },
        { title: "Pick", text: "Move the highest-scoring remaining box $b^*$ from $B$ to $K$." },
        { title: "Suppress", text: "Compute IoU($b^*$, $b$) for every remaining $b \\in B$ and delete those with IoU > $N_t$." },
        { title: "Repeat", text: "Until $B$ is empty. Worst case $O(n^2)$. Run it per class (or with class offsets) so a dog box cannot suppress an overlapping person box." },
      ] },
      { t: "p", text: "**Soft-NMS** (Bodla et al., 2017) lowers scores instead of deleting boxes — e.g. $s_b \\leftarrow s_b\\,e^{-\\mathrm{IoU}(b^*, b)^2/\\sigma}$. A heavily overlapping box that is really a *second* object (two people hugging) is down-weighted, not thrown away. It changes only inference and usually helps in crowded scenes." },
      { t: "h", text: "Evaluating a detector" },
      { t: "steps", items: [
        { title: "Match", text: "For one class, sort all detections over the test set by score and walk down the list. A detection is a **true positive** (TP) if its IoU with a not-yet-matched true box of that class in the same image is ≥ the threshold (0.5 for PASCAL VOC); otherwise a **false positive** (FP) — duplicates included. True boxes never matched are **false negatives**." },
        { title: "Precision–recall curve", text: "After each detection in the list: precision = TP/(TP+FP) so far, and recall = TP/(number of true boxes)." },
        { title: "Average precision", text: "Make precision never increase with recall (replace each value by the highest precision at any higher recall) and take the area under this envelope — the **all-point interpolated AP**." },
        { title: "mAP", text: "Average AP over classes. **mAP@0.5** uses IoU threshold 0.5. The **COCO metric** AP@[.5:.95] averages AP over ten thresholds 0.50, 0.55, …, 0.95, rewarding tight boxes." },
      ] },
    ],
    math: [
      { t: "h", text: "1. Intersection over Union" },
      { t: "math", tex: String.raw`\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A\cap B|}`, caption: "$|\\cdot|$ is area. IoU is 1 for identical boxes, 0 for boxes that don't touch, and does not depend on scale." },
      { t: "p", text: String.raw`For corner boxes, the overlap is the rectangle with corners $(\max(x_1^A, x_1^B), \max(y_1^A, y_1^B))$ and $(\min(x_2^A, x_2^B), \min(y_2^A, y_2^B))$; its width and height are clipped at 0 when the boxes do not overlap.` },
      { t: "steps", items: [
        { title: "Boxes", tex: String.raw`G = (50, 40, 150, 160),\qquad P = (80, 60, 170, 180)` },
        { title: "Areas", tex: String.raw`|G| = 100 \times 120 = 12000,\qquad |P| = 90 \times 120 = 10800` },
        { title: "Intersection", tex: String.raw`(\max(50,80), \max(40,60), \min(150,170), \min(160,180)) = (80, 60, 150, 160) \Rightarrow 70 \times 100 = 7000` },
        { title: "Union", tex: String.raw`12000 + 10800 - 7000 = 15800` },
        { title: "IoU", tex: String.raw`7000 / 15800 = 0.4430` },
      ] },
      { t: "p", text: "The boxes look well aligned (same height, centres 25 px apart), yet IoU is below 0.5: at mAP@0.5 this prediction is a false positive *and* the object is missed. IoU punishes misalignment fast — shifting a box by 1/3 of its width already drops IoU to 0.5." },
      { t: "h", text: "2. Anchors" },
      { t: "p", text: String.raw`An anchor with scale $s$ and aspect ratio $r = h/w$ has $w = s/\sqrt{r}$ and $h = s\sqrt{r}$, so its area is $s^2$ for every ratio. On an $H_f \times W_f$ feature map with stride $\sigma$, anchor centres are at $((j + \tfrac12)\sigma, (i + \tfrac12)\sigma)$, and there are $H_f W_f A$ anchors in total. Faster R-CNN used $A = 9$ (3 scales × 3 ratios) on a stride-16 map: for a $600 \times 1000$ image, $37 \times 62 \times 9 = 20{,}646$ anchors (about 20k). SSD300 uses six feature maps ($38^2, 19^2, 10^2, 5^2, 3^2, 1^2$ cells with 4 or 6 default boxes each): $8{,}732$ boxes.` },
      { t: "p", text: String.raw`**Matching rule.** With the IoU matrix $M_{ag} = \mathrm{IoU}(\text{anchor}_a, \text{gt}_g)$, each anchor gets its best object $g^*(a) = \arg\max_g M_{ag}$ with overlap $m_a = \max_g M_{ag}$, and the label` },
      { t: "math", tex: String.raw`\ell_a = \begin{cases} \text{positive (class of } g^*(a)) & m_a \ge \tau_+ \\ \text{ignore} & \tau_- \le m_a < \tau_+ \\ \text{background} & m_a < \tau_- \end{cases}\qquad\text{plus } \ell_{\arg\max_a M_{ag}} = \text{positive for every } g`, caption: "RetinaNet uses τ₊ = 0.5, τ₋ = 0.4; Faster R-CNN's RPN uses 0.7 and 0.3." },
      { t: "h", text: "3. Box encoding relative to an anchor" },
      { t: "p", text: String.raw`Let the anchor be $(x_a, y_a, w_a, h_a)$ and the true box $(x, y, w, h)$, both in centre format. Faster R-CNN's targets are` },
      { t: "math", tex: String.raw`t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a}` },
      { t: "math", tex: String.raw`\hat x = x_a + \hat t_x w_a,\quad \hat y = y_a + \hat t_y h_a,\quad \hat w = w_a e^{\hat t_w},\quad \hat h = h_a e^{\hat t_h}`, caption: "Decoding reverses the encoding exactly." },
      { t: "p", text: "Why these forms? Dividing by anchor size makes targets **scale-invariant**: a 10-px shift means far more for a 20-px box than a 400-px one. The **log** makes size changes symmetric (doubling and halving give $\\pm 0.693$), keeps sizes positive through $\\exp$, and keeps targets roughly zero-mean with similar spread — good for an L1/L2-type loss. Implementations often also divide targets by fixed standard deviations (torchvision's RoI heads use weights $(10, 10, 5, 5)$)." },
      { t: "steps", items: [
        { title: "Anchor and true box", tex: String.raw`\text{anchor }(60,40,140,140) \to (x_a,y_a,w_a,h_a) = (100, 90, 80, 100);\quad G = (50,40,150,160) \to (100, 100, 100, 120)` },
        { title: "Targets", tex: String.raw`t_x = \tfrac{100-100}{80} = 0,\quad t_y = \tfrac{100-90}{100} = 0.1,\quad t_w = \ln\tfrac{100}{80} = 0.2231,\quad t_h = \ln\tfrac{120}{100} = 0.1823` },
        { title: "Decoding a prediction", text: "Say the head outputs $\\hat t = (0.05, -0.1, 0.1, 0.2)$:", tex: String.raw`\hat x = 100 + 0.05 \cdot 80 = 104,\ \ \hat y = 90 - 0.1 \cdot 100 = 80,\ \ \hat w = 80e^{0.1} = 88.41,\ \ \hat h = 100 e^{0.2} = 122.14` },
        { title: "Back to corners", tex: String.raw`(104 - 44.21,\ 80 - 61.07,\ 104 + 44.21,\ 80 + 61.07) = (59.79,\ 18.93,\ 148.21,\ 141.07)` },
      ] },
      { t: "h", text: "4. Detection losses" },
      { t: "p", text: String.raw`A typical anchor-based loss over the positive anchors $\mathcal{P}$ and sampled negatives $\mathcal{N}$:` },
      { t: "math", tex: String.raw`\mathcal{L} = \frac{1}{|\mathcal{P}|}\Big[\sum_{a \in \mathcal{P}\cup\mathcal{N}} \mathcal{L}_{\text{cls}}(p_a, c_a) + \lambda \sum_{a\in\mathcal{P}}\sum_{k \in \{x,y,w,h\}} \mathrm{smooth}_{L1}(\hat t_{a,k} - t_{a,k})\Big]` },
      { t: "math", tex: String.raw`\mathrm{smooth}_{L1}(d) = \begin{cases} 0.5\,d^2/\beta & |d| < \beta \\ |d| - 0.5\beta & \text{otherwise}\end{cases}`, caption: "Quadratic near zero (smooth gradients for small errors), linear for big errors (robust to outliers). Fast R-CNN used β = 1." },
      { t: "p", text: String.raw`Worked example with $\beta = 1$: target $t = (0, 0.1, 0.2231, 0.1823)$, prediction $\hat t = (0.1, 0.3, 0.5, -1.2)$. Differences $d = (0.1, 0.2, 0.2769, -1.3823)$; losses $0.005 + 0.02 + 0.0383 + (1.3823 - 0.5 = 0.8823) = 0.9456$. The one big error counts linearly, not quadratically ($0.5 \times 1.3823^2 = 0.955$ under L2).` },
      { t: "p", text: String.raw`**Background imbalance.** A dense detector scores $10^4$–$10^5$ anchors per image, and only a handful are positive. SSD keeps only the highest-loss negatives, at a 3:1 negative:positive ratio (**hard negative mining**). RetinaNet instead applies the **focal loss** ([derivation](/learn/m05/classification-losses)) to every anchor:` },
      { t: "math", tex: String.raw`\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log p_t,\qquad \gamma = 2,\ \alpha = 0.25` },
      { t: "p", text: String.raw`Here $p_t$ is the predicted probability of the true class. With $\gamma = 2$, an easy background anchor at $p_t = 0.9$ has cross-entropy $0.1054$ but focal loss $(0.1)^2 \times 0.1054 = 0.00105$ — 100× smaller — while a hard one at $p_t = 0.1$ keeps $0.81 \times 2.3026 = 1.865$. The thousands of easy negatives no longer drown the gradient. RetinaNet also sets the classifier bias to $b = -\log((1-\pi)/\pi)$ with $\pi = 0.01$, so training starts out predicting \"background\" everywhere.` },
      { t: "p", text: String.raw`**IoU-based regression.** Instead of four separate offsets, you can decode the box and minimise $1 - \mathrm{GIoU}$ ([derived here](/learn/m05/advanced-losses)). That directly optimises what is evaluated and is scale-invariant. DETR combines L1 on normalised $(c_x, c_y, w, h)$ with GIoU.` },
      { t: "h", text: "5. NMS worked example" },
      { t: "p", text: "Five boxes (score): $b_0 = (10,10,60,60)$ 0.9, $b_1 = (14,12,62,64)$ 0.8, $b_2 = (40,40,90,90)$ 0.7, $b_3 = (100,100,150,150)$ 0.6, $b_4 = (12,8,58,62)$ 0.5; threshold $N_t = 0.5$." },
      { t: "steps", items: [
        { title: "Pick b0 (0.9)", tex: String.raw`\mathrm{IoU}(b_0,b_1) = 0.792,\ \mathrm{IoU}(b_0,b_2) = 0.087,\ \mathrm{IoU}(b_0,b_3) = 0,\ \mathrm{IoU}(b_0,b_4) = 0.857 \Rightarrow \text{suppress } b_1, b_4` },
        { title: "Pick b2 (0.7)", tex: String.raw`\mathrm{IoU}(b_2, b_3) = 0 \Rightarrow \text{nothing suppressed}` },
        { title: "Pick b3 (0.6)", tex: String.raw`K = \{b_0, b_2, b_3\}` },
      ] },
      { t: "p", text: "The framework section runs the same example with TensorFlow's `tf.image.non_max_suppression`. Note that $b_2$ survives although it touches $b_0$: an IoU of 0.087 is far below the threshold." },
      { t: "h", text: "6. Average precision" },
      { t: "p", text: String.raw`With detections sorted by score, let $P_k$ and $R_k$ be precision and recall after the $k$-th detection. The interpolated precision is $P_{\text{interp}}(r) = \max_{k:\,R_k \ge r} P_k$ — the best precision at recall $r$ or beyond — and` },
      { t: "math", tex: String.raw`\mathrm{AP} = \sum_{k} (R_k - R_{k-1})\, P_{\text{interp}}(R_k), \qquad \mathrm{mAP} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{AP}_c, \qquad \mathrm{AP}^{\text{COCO}} = \frac{1}{10}\sum_{\tau \in \{0.5, 0.55, \ldots, 0.95\}} \mathrm{mAP}_\tau` },
      { t: "p", text: "Worked example (the toy case in the NumPy code): 4 true boxes in 3 images; six detections in score order are TP, FP (duplicate), TP, FP, TP, FP." },
      { t: "table", head: ["k", "score", "TP/FP", "total TP", "precision", "recall"], rows: [
        ["1", "0.95", "TP", "1", "1/1 = 1.000", "1/4 = 0.25"],
        ["2", "0.90", "FP (duplicate)", "1", "1/2 = 0.500", "0.25"],
        ["3", "0.80", "TP", "2", "2/3 = 0.667", "0.50"],
        ["4", "0.70", "FP", "2", "2/4 = 0.500", "0.50"],
        ["5", "0.60", "TP", "3", "3/5 = 0.600", "0.75"],
        ["6", "0.50", "FP", "3", "3/6 = 0.500", "0.75"],
      ] },
      { t: "p", text: String.raw`Recall rises at $k = 1, 3, 5$ to $0.25, 0.5, 0.75$. Interpolated precision at those recalls: $\max(1, 0.5, \ldots) = 1$, $\max(0.667, 0.5, 0.6, 0.5) = 0.667$, $\max(0.6, 0.5) = 0.6$. So $\mathrm{AP} = 0.25(1) + 0.25(0.667) + 0.25(0.6) = 0.5667$. The fourth object is never found, so recall stops at 0.75 and the area beyond it is 0. (PASCAL VOC 2007 used an older 11-point approximation; COCO samples the envelope at 101 recall values.)` },
      { t: "h", text: "7. DETR: set prediction with Hungarian matching" },
      { t: "p", text: String.raw`DETR outputs a fixed set of $N$ predictions $\hat y_i = (\hat p_i, \hat b_i)$ ($N = 100$ in the paper, far more than the number of objects) and pads the true objects with \"no object\" $\varnothing$ up to size $N$. Training first finds the pairing $\hat\sigma$ with the lowest total matching cost` },
      { t: "math", tex: String.raw`\hat\sigma = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat y_{\sigma(i)}),\qquad \mathcal{L}_{\text{match}} = -\mathbb{1}_{\{c_i \neq \varnothing\}}\,\hat p_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}}\,\mathcal{L}_{\text{box}}(b_i, \hat b_{\sigma(i)})` },
      { t: "p", text: String.raw`solved exactly in $O(N^3)$ by the Hungarian algorithm. Then it applies cross-entropy to every prediction (matched ones to their class, the rest to $\varnothing$) and $\lambda_{L1}\|b - \hat b\|_1 + \lambda_{\text{GIoU}}(1 - \mathrm{GIoU})$ to matched ones. Each object is matched to exactly one prediction, so the model learns to produce **no duplicates** — and needs no NMS.` },
    ],
    visual: [
      { t: "p", text: "The simulator has three tabs. Every box can be dragged (drag the body to move it, the corner square to resize); the \"Adjust boxes with sliders\" panel gives the same control without a pointer." },
      { t: "sim", id: "detection", caption: "IoU, anchor assignment and greedy NMS computed live on boxes you control." },
      { t: "list", items: [
        "**IoU tab.** Step through the computation. Slide P sideways until IoU crosses 0.5, then shrink P to half its size inside G: IoU equals the area ratio (at most 0.5 for half the area), though P is exactly right about *where* the object is.",
        "**Anchors tab.** With stride 32, scales {48, 96} and three ratios, count positives. Untick \"each GT keeps its best anchor\" and raise the positive threshold to 0.7 (the RPN value): small or oddly shaped objects lose every positive anchor — the reason for the forced match. Add scale 32 or 160 and see which GT gains positives.",
        "**Anchors tab, targets.** Step through the positives: anchors whose centre is left of the GT centre have $t_x > 0$; anchors smaller than the GT have $t_w > 0$. Check the numbers against the encoding formula.",
        "**NMS tab.** Step through the decisions, reading each remaining box's IoU with the current pick. Threshold 0.9: duplicates survive, precision drops. Drag two GT boxes to overlap heavily, threshold 0.3: one object is suppressed, recall drops — the classic crowd failure.",
        "Raise the score threshold: background false positives vanish first (they score low), raising precision without hurting recall — until the threshold reaches the scores of real objects.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Every core piece of a detector in NumPy: format conversion, vectorised pairwise IoU, greedy NMS, an anchor generator, max-IoU matching with forced best matches, box encoding/decoding, and all-point interpolated AP. The last lines compare IoU and NMS with `torchvision.ops` on 200 random boxes." },
      { t: "code", lang: "python", title: "detection_from_scratch.py", code: DET_NUMPY, output: DET_NUMPY_OUT },
      { t: "list", items: [
        "`box_iou` broadcasts `(N,1,2)` against `(1,M,2)` to get all $N \\times M$ intersection corners at once; `np.clip(..., 0, None)` handles boxes that don't touch.",
        "The 10-pixel shift example: a $40 \\times 50$ box shifted by 10 px both ways has intersection $30 \\times 40 = 1200$ and union $2000 + 2000 - 1200 = 2800$: IoU $0.4286$.",
        "Matching: on this coarse 4×4 grid only 2 of 96 anchors are positive, and anchor 36 (IoU 0.497) is positive only thanks to the forced best-anchor rule — exactly the case the rule exists for. `decode(encode(gt))` recovers the boxes exactly.",
        "NMS keeps one box per object (indices 5 and 2) out of ten noisy candidates, and the toy-case AP equals the hand calculation (0.5667).",
        "IoU matches `torchvision.ops.box_iou` to machine precision, and NMS keeps the same indices as `torchvision.ops.nms`.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch: train a detector", blocks: [
          { t: "p", text: "A complete single-shot (SSD-style) detector trained from scratch on synthetic data made on the fly: 48×48 canvases with 1–3 randomly coloured rectangles and circles on noisy backgrounds; classes are set by shape, not colour. The backbone shrinks the image to a 6×6 map (stride 8) with 6 anchors per cell (216 anchors); the head predicts 3 class logits and 4 offsets per anchor. Training: cross-entropy with 3:1 hard negative mining plus smooth L1. Inference: decode → per-class NMS. Evaluation: per-class AP at IoU 0.5 and the COCO-style average over 0.5:0.95. Nothing is downloaded." },
          { t: "code", lang: "python", title: "tiny_ssd_synthetic.py", code: DET_TORCH, output: DET_TORCH_OUT },
          { t: "list", items: [
            "mAP@0.5 ≈ 0.86 but AP@[.5:.95] ≈ 0.45: the model finds the objects, but its boxes are often not tight enough for IoU ≥ 0.8. This gap is typical — and why COCO reports the averaged metric.",
            "Training loss keeps falling while test AP levels off: with only 800 training images the model overfits. More (synthetic) data, flips and crops would help — detection is data-hungry.",
            "On a 6×6 grid, two small objects in the same cell compete for the same anchors; real detectors use finer strides and feature pyramids for small objects.",
            "The head's background bias starts at 4.6 (probability ≈ 0.99 of background) — RetinaNet's prior-initialisation trick.",
          ] },
          { t: "h", text: "Faster R-CNN in torchvision (random init, no download)" },
          { t: "p", text: "`fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)` builds the full two-stage model with random weights. In training mode it takes a list of images plus a list of target dicts and returns four losses (RPN objectness and box regression, RoI-head classification and box regression). In eval mode it returns, per image, up to 100 detections — boxes in absolute xyxy pixel coordinates of the original image, labels and scores." },
          { t: "code", lang: "python", title: "faster_rcnn_structure.py", code: FRCNN, output: FRCNN_OUT },
          { t: "p", text: "Most of the 41.8M parameters sit in the ResNet-50 backbone (23.5M) and the RoI box head (two fully connected layers of width 1024 on 256×7×7 RoIAlign features). The RPN is tiny: one 3×3 conv plus two 1×1 convs shared across the five FPN levels, each with one anchor size and three aspect ratios. With random weights the eval scores mean nothing; real use starts from pretrained weights, which this course does not download." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "TensorFlow provides NMS and soft-NMS as image ops. Boxes use the order $(y_1, x_1, y_2, x_2)$. The example repeats the NMS worked example from the Math section (swapping coordinates changes nothing here, so the result is the same)." },
          { t: "code", lang: "python", title: "nms_keras.py (not executed in CI)", code: DET_KERAS, output: "Prints the kept indices of hard NMS — [0 2 3], as in the worked example — and the indices and Gaussian-decayed scores kept by soft-NMS, where the suppressed duplicates survive with reduced scores." },
          { t: "p", text: "Full Keras detectors (RetinaNet, YOLO variants) are in KerasCV. Their APIs have changed between versions, so check the current docs rather than copying signatures." },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "Two-stage: the R-CNN family" },
      { t: "list", items: [
        "**R-CNN** (Girshick, Donahue, Darrell, Malik, 2014): ~2000 class-blind proposals from *selective search*, each warped to a fixed size and run through a CNN; per-class SVMs classify the features and a linear regressor refines the box. Accurate for its time but slow: one CNN pass per proposal.",
        "**Fast R-CNN** (Girshick, 2015): run the CNN **once** per image. For each proposal, **RoI pooling** crops its feature-map region and max-pools it to a fixed grid (e.g. 7×7); fully connected layers give two outputs (softmax over $C+1$ classes, per-class box offsets), trained together with smooth L1. Proposals still come from selective search.",
        "**Faster R-CNN** (Ren, He, Girshick, Sun, 2015): a **Region Proposal Network** — a small conv head on the shared feature map — predicts an objectness score and box offsets for $k = 9$ anchors per position. Its top proposals (after NMS) feed the Fast R-CNN head. Proposals become nearly free and learned end-to-end.",
        "**FPN** (Lin et al., 2017) adds a top-down path with side connections, so every pyramid level gets semantically strong features; each RoI is pooled from the level that matches its size.",
        "**Mask R-CNN** (He, Gkioxari, Dollár, Girshick, 2017) adds a mask branch per RoI and replaces RoI pooling's rounding with **RoIAlign** — covered in [segmentation](/learn/m12/segmentation).",
      ] },
      { t: "h", text: "One-stage and set-prediction detectors" },
      { t: "list", items: [
        String.raw`**YOLO** (Redmon, Divvala, Girshick, Farhadi; arXiv 2015, CVPR 2016): splits the image into an $S \times S$ grid; the cell holding an object's centre is responsible for it. Each cell predicts $B$ boxes $(x, y, w, h, \text{confidence})$ and one set of $C$ class probabilities, so the output is $S \times S \times (5B + C)$ — $7 \times 7 \times 30$ for $S=7, B=2, C=20$. The loss is sum-squared error with weights $\lambda_{\text{coord}} = 5$, $\lambda_{\text{noobj}} = 0.5$, and regresses $\sqrt{w}, \sqrt{h}$ so errors on small boxes count more. Very fast; weak on small, clustered objects (one class set per cell).`,
        "**Later YOLO versions** (v2 onward, plus many community successors) added anchors, multi-scale prediction, stronger backbones and augmentation, later moving to anchor-free heads. Designs vary; read each paper or repository.",
        "**SSD** (Liu et al., 2016): predicts class scores and offsets for **default boxes** on *several* feature maps of falling resolution — fine maps for small objects, coarse ones for large — in one pass, with 3:1 hard negative mining and heavy data augmentation.",
        "**RetinaNet** (Lin, Goyal, Girshick, He, Dollár, 2017): FPN backbone + two dense conv subnets (classification with sigmoid focal loss, box regression), 9 anchors per level. Showed a one-stage detector can match two-stage accuracy once class imbalance is handled.",
        "**Anchor-free**: **FCOS** (Tian et al., 2019) predicts, at every feature-map location inside an object, the distances $(l, t, r, b)$ to the four box sides plus a \"centerness\" score. **CenterNet** (Zhou, Wang, Krähenbühl, 2019, *Objects as Points*) predicts a heatmap of object centres and regresses size at each peak. No anchor hyperparameters to tune.",
        "**DETR** (Carion et al., 2020): CNN backbone + transformer encoder–decoder with $N$ learned **object queries**; one-to-one (Hungarian) matching loss; no anchors, no NMS. Simple and elegant, but slow to train and originally weaker on small objects — which motivated follow-ups such as Deformable DETR.",
      ] },
      { t: "table", head: ["Family", "Proposals / references", "Duplicates removed by", "Strengths", "Weaknesses"], rows: [
        ["R-CNN / Fast R-CNN", "Selective search (external)", "NMS", "Accurate per-region features", "Slow, not end-to-end"],
        ["Faster R-CNN (+FPN)", "RPN on anchors", "NMS (twice: RPN and final)", "High accuracy, flexible RoI heads (masks, keypoints)", "Two stages, more complex, slower"],
        ["YOLO (v1)", "Grid cells", "NMS", "Very fast, sees global context", "Small/clustered objects, rough localisation"],
        ["SSD", "Multi-scale default boxes", "NMS", "Fast, multi-scale", "Small objects rely on weak early features"],
        ["RetinaNet", "Dense anchors + FPN", "NMS", "One-stage at two-stage accuracy (focal loss)", "Many anchors, anchor tuning"],
        ["FCOS / CenterNet", "Points / centres", "NMS (or peak picking)", "No anchor hyperparameters", "Ambiguous for overlapping objects"],
        ["DETR", "Learned queries", "Hungarian matching (none at inference)", "End-to-end, no hand-designed parts", "Long training, small objects (original)"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Self-driving and driver assistance**: cars, pedestrians, cyclists, traffic signs and lights — often in 3D (bird's-eye-view boxes from camera or LiDAR) and tracked over time.",
        "**Retail and logistics**: shelf checks, checkout-free stores, finding packages and barcodes in warehouses.",
        "**Factory inspection**: locating defects (scratches, missing parts) on production lines, where objects are tiny and rare — a natural fit for focal loss and high-resolution feature maps.",
        "**Medical imaging**: nodules in CT, cells and mitoses in histopathology, polyps in endoscopy video.",
        "**Remote sensing**: ships, aircraft, vehicles and buildings in satellite images (often with rotated boxes).",
        "**Face detection** in cameras and photo apps, and as the first step of face recognition ([vision tasks](/learn/m12/vision-tasks)).",
        "**As a building block**: instance segmentation (Mask R-CNN), human pose estimation (find people, then keypoints), multi-object tracking (detect per frame, then link).",
      ] },
    ],
    tradeoffs: [
      { t: "h", text: "Strengths" },
      { t: "list", items: [
        "One network outputs *how many*, *what* and *where*, trainable end-to-end with standard losses.",
        "Modular: backbones, necks (FPN), heads and losses can be swapped; the two-stage design extends to masks and keypoints.",
        "A speed–accuracy dial: one-stage detectors run in real time on edge hardware; two-stage detectors and large DETR-style models localise best.",
      ] },
      { t: "h", text: "Limitations and trade-offs" },
      { t: "list", items: [
        "**Hand-designed parts**: anchor sizes/shapes and the IoU, NMS and score thresholds need tuning per dataset; anchor-free and DETR-style models remove some.",
        "**NMS is not learned** and fails in crowds (real neighbours get suppressed) — soft-NMS or set prediction help.",
        "**Small objects** are hard: they cover few cells of a strided feature map. Remedies — FPN, higher input resolution, tiling — cost compute.",
        "**Upright boxes** fit long, rotated or oddly shaped objects poorly (use rotated boxes or masks).",
        "**Labelling cost**: boxes are cheaper than masks but still expensive, and noisy box edges limit the AP you can reach at high IoU thresholds.",
        "**mAP hides behaviour**: one number mixes classification, localisation and duplicate errors; inspect PR curves, AP by object size (COCO reports AP-small/medium/large) and class confusions.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Mixing box formats** (xyxy vs xywh vs cxcywh, or (x, y) vs TensorFlow's (y, x)). IoU silently turns into nonsense. Fix: convert once when loading data, and assert $x_2 > x_1$, $y_2 > y_1$.",
        "**Not rescaling boxes when resizing images** (augmentation, letterboxing), so targets are misaligned. Fix: transform boxes exactly like pixels, and look at a batch.",
        "**Encoding targets with one anchor convention and decoding with another** (e.g. different variance weights, or centre vs corner). Fix: unit-test `decode(encode(gt, a), a) == gt`.",
        "**Running NMS across classes** — a confident \"person\" box suppresses the overlapping \"bicycle\". Fix: per-class NMS (or `batched_nms` with class indices).",
        "**Letting several detections match one true box in evaluation.** Every extra detection must be a false positive, or duplicates inflate precision. Fix: mark the true box as used.",
        "**Computing AP with a high score threshold** (e.g. 0.5). AP covers all thresholds; dropping low-score detections cuts the PR curve short and lowers AP. Use ~0.05 for evaluation, higher for deployment.",
        "**Treating ignored anchors as negatives** — anchors overlapping an object at IoU 0.45 get pushed toward \"background\", fighting nearby positives. Fix: leave them out of the loss.",
        "**Dividing the loss by the number of anchors** instead of positives: the box loss then shrinks with image size and the many negatives dominate. Divide by the number of positives.",
        "**Comparing mAP@0.5 with COCO AP**: they differ a lot (0.86 vs 0.45 in our run). Always say which metric and IoU thresholds.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between classification, localization and detection?", a: "Classification gives one label for the whole image. Localization gives a label plus one bounding box (assuming one object). Detection gives a variable-size set of (box, class, score) for every object of every class present — possibly none." },
      { t: "qa", level: "beginner", q: "Define IoU and give its range. Why is it used instead of, e.g., the distance between box centres?", a: "IoU = area of overlap / area of union, between 0 and 1. It measures overlap relative to size, so it is scale-invariant and covers both position and size. A 10-px centre distance is excellent for a large box and terrible for a tiny one." },
      { t: "qa", level: "beginner", q: "What does non-maximum suppression do and what is its main hyperparameter?", a: "It removes duplicate detections of one object: sort by score, keep the best box, delete remaining boxes whose IoU with it exceeds a threshold (typically 0.45–0.7), and repeat. The IoU threshold trades duplicates (too high) against suppressing real neighbouring objects (too low)." },
      { t: "qa", level: "intermediate", q: "Why are box offsets encoded as (Δx/w_a, Δy/h_a, log(w/w_a), log(h/h_a)) rather than raw pixel coordinates?", a: "Dividing by anchor size makes targets scale-invariant, so one regressor serves small and large anchors. The log makes size errors symmetric (×2 and ×½ give ±0.69), guarantees positive sizes after exp, and keeps targets small and roughly zero-mean, suiting smooth L1. Each anchor only predicts a small correction." },
      { t: "qa", level: "intermediate", q: "How does Faster R-CNN differ from Fast R-CNN?", a: "Fast R-CNN still uses outside proposals (selective search), which dominate run time. Faster R-CNN adds a Region Proposal Network on the shared backbone features: at each position it scores k anchors for objectness and regresses their offsets; the top proposals after NMS feed the RoI head. Proposals become learned and nearly free." },
      { t: "qa", level: "intermediate", q: "What problem does the focal loss solve and how?", a: "Dense one-stage detectors have ~10⁴–10⁵ anchors per image, almost all easy background, whose small losses add up and dominate training. FL = −α(1−p_t)^γ log p_t multiplies the loss by (1−p_t)^γ, so well-classified examples (p_t → 1) add almost nothing and training focuses on hard ones. With γ = 2, an example at p_t = 0.9 is down-weighted 100×." },
      { t: "qa", level: "intermediate", q: "Explain mAP@0.5 versus COCO AP@[.5:.95].", a: "AP is the area under the interpolated precision–recall curve for one class at a fixed IoU threshold; mAP averages it over classes. mAP@0.5 uses threshold 0.5 (PASCAL VOC). COCO averages mAP over thresholds 0.50 to 0.95 in steps of 0.05, rewarding tight boxes — so a model can have high mAP@0.5 and much lower COCO AP." },
      { t: "qa", level: "advanced", q: "How does DETR avoid NMS, and why does it need Hungarian matching?", a: "DETR predicts a fixed set of N boxes and uses the Hungarian algorithm to find the one-to-one pairing of predictions with true objects (padded with 'no object') that minimises a class + L1 + GIoU cost. Each matched prediction is trained toward its object, all others toward 'no object'. A duplicate would be paired with 'no object' and penalised, so the network (with self-attention between queries) learns to output each object once — no NMS. Without one-to-one matching the loss would be ambiguous and depend on prediction order." },
      { t: "qa", level: "advanced", q: "Why does RoIAlign improve over RoI pooling, and why does it matter more for masks than for boxes?", a: "RoI pooling rounds the RoI and its bin edges to the feature grid; at stride 16 each rounding can shift features by up to 16 input pixels, misaligning features and RoI. RoIAlign keeps real-valued coordinates and samples bilinearly. Classification tolerates small shifts, but pixel-accurate masks (and high-IoU boxes) do not — Mask R-CNN reported large mask-accuracy gains from RoIAlign." },
      { t: "qa", level: "advanced", q: "Your detector has high recall but low precision at IoU 0.5. List likely causes and diagnostics.", a: "Duplicates (NMS threshold too high, or not per class), a too-low deployment score threshold, background confusions (too few hard negatives), or class confusion. Diagnose by splitting FPs into duplicates / background / wrong class / poor localisation (IoU 0.1–0.5), checking the PR curve, and viewing high-scoring false positives." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Box A = (0, 0, 4, 4), box B = (2, 2, 6, 6). What is their IoU? (4 decimals)", answer: 0.1429, tolerance: 0.0005, explain: "Intersection (2,2,4,4) has area 4; union = 16 + 16 − 4 = 28; IoU = 4/28 = 0.1429." },
        { kind: "numeric", q: "An anchor has centre (50, 50) and size 40×40. The true box has centre (60, 50) and size 80×40. What is $t_w$? (3 decimals)", answer: 0.693, tolerance: 0.002, explain: "$t_w = \\ln(80/40) = \\ln 2 = 0.693$. (Also $t_x = 10/40 = 0.25$, $t_y = 0$, $t_h = 0$.)" },
        { kind: "numeric", q: "A feature map of 20×30 cells with 3 scales × 3 aspect ratios per cell. How many anchors?", answer: 5400, tolerance: 0, explain: "$20 \\times 30 \\times 9 = 5400$." },
        { q: "In greedy NMS with threshold 0.5, box X (score 0.9) and box Y (score 0.8) have IoU 0.6 and belong to different classes. With per-class NMS:", options: ["Y is suppressed", "X is suppressed", "Both are kept", "Both are suppressed"], answer: 2, explain: "Per-class NMS only compares boxes of the same class, so both survive." },
        { q: "What does the focal loss's $(1 - p_t)^\\gamma$ factor do?", options: ["Upweights easy, well-classified examples", "Downweights easy examples so hard ones dominate the gradient", "Normalises the box regression loss", "Replaces non-maximum suppression"], answer: 1, explain: "As $p_t \\to 1$ the factor goes to 0, so the flood of easy negatives adds little." },
        { q: "Why does DETR not need NMS?", options: ["It uses very high score thresholds", "Its one-to-one Hungarian matching loss penalises duplicate predictions", "It predicts only one object per image", "It uses anchors of every possible size"], answer: 1, explain: "Each true object is matched to exactly one query; the others learn to predict 'no object', so duplicates are trained away." },
        { kind: "numeric", q: "Detections sorted by score are TP, TP, FP, TP and there are 5 true objects. What is the all-point interpolated AP? (3 decimals)", answer: 0.55, tolerance: 0.002, explain: "(precision, recall) after each detection: (1, 0.2), (1, 0.4), (0.667, 0.4), (0.75, 0.6). Recall rises at detections 1, 2 and 4, each by 0.2; interpolated precisions there are 1, 1 and 0.75. AP = 0.2·1 + 0.2·1 + 0.2·0.75 = 0.55. The two objects never found add nothing." },
        { q: "Which statement about YOLO v1 is correct?", options: ["It uses a region proposal network", "Each grid cell predicts B boxes but only one set of class probabilities", "It uses RoIAlign", "It has no grid"], answer: 1, explain: "Output $S \\times S \\times (5B + C)$: B boxes with confidences and one class distribution per cell, which limits it on clusters of small objects." },
      ] },
      { t: "exercise", title: "IoU, encoding and NMS by hand", prompt: "True box $G = (20, 20, 100, 80)$. Anchor $a = (30, 20, 90, 80)$. (a) Compute IoU(G, a). (b) Compute the targets $(t_x, t_y, t_w, t_h)$. (c) Three detections: $d_1 = (20,20,100,80)$ score 0.9, $d_2 = (30,20,90,80)$ score 0.8, $d_3 = (90,20,150,80)$ score 0.7. Run NMS with threshold 0.5.", hint: "The anchor lies entirely inside G.", solution: [
        { t: "steps", items: [
          { title: "(a) IoU", tex: String.raw`|G| = 80 \times 60 = 4800,\ |a| = 60 \times 60 = 3600,\ |G \cap a| = 3600 \Rightarrow \mathrm{IoU} = 3600/4800 = 0.75` },
          { title: "(b) Targets", tex: String.raw`G: (60, 50, 80, 60),\ a: (60, 50, 60, 60) \Rightarrow t = (0,\ 0,\ \ln\tfrac{80}{60} = 0.2877,\ 0)` },
          { title: "(c) NMS", tex: String.raw`\text{keep } d_1;\ \mathrm{IoU}(d_1,d_2) = 0.75 > 0.5 \Rightarrow \text{suppress } d_2;\ \mathrm{IoU}(d_1,d_3) = \tfrac{10\times60}{4800 + 3600 - 600} = 0.0769 \Rightarrow \text{keep } d_3` },
        ] },
      ] },
      { t: "exercise", title: "Add soft-NMS and measure AP", prompt: "In `detection_from_scratch.py`, write `soft_nms(boxes, scores, sigma=0.5, score_thr=0.001)` with Gaussian decay $s_b \\leftarrow s_b\\, e^{-\\mathrm{IoU}^2/\\sigma}$. Build an image with two heavily overlapping true objects (IoU ≈ 0.6) and noisy candidates around each, and compare AP@0.5 after hard NMS (threshold 0.5) and after soft-NMS.", hint: "Soft-NMS never deletes boxes above `score_thr`; it re-sorts after each decay. The second object's best candidate overlaps the first object's best by ≈ 0.6, so hard NMS deletes it.", solution: [
        { t: "code", lang: "python", code: String.raw`def soft_nms(boxes, scores, sigma=0.5, score_thr=1e-3):
    scores = scores.astype(float).copy()
    idx = np.arange(len(boxes)); keep = []
    while idx.size:
        i = idx[np.argmax(scores[idx])]
        keep.append(i)
        idx = idx[idx != i]
        if idx.size == 0:
            break
        ious = box_iou(boxes[i:i + 1], boxes[idx])[0]
        scores[idx] *= np.exp(-ious ** 2 / sigma)       # decay instead of delete
        idx = idx[scores[idx] > score_thr]
    return np.array(keep), scores` },
        { t: "p", text: "With hard NMS the first pick deletes all the second object's candidates, so recall is capped at 0.5 and AP ≈ 0.5. With soft-NMS they survive with lowered scores ($e^{-0.36/0.5} \\approx 0.49$ times the original), ranking after the first object's best box but ahead of most noise, and AP rises a lot. Exact values depend on your random candidates — run it and compare." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Detection = a variable-size set of (box, class, score). Core problems: fixed outputs, matching in training, removing duplicates.",
        String.raw`Formats: xyxy, xywh, cxcywh. $\mathrm{IoU} = |A\cap B| / (|A| + |B| - |A \cap B|)$.`,
        String.raw`Anchors: $H_f W_f A$ reference boxes; labelled by max IoU with thresholds $\tau_+$, $\tau_-$, plus a forced best match.`,
        String.raw`Encoding: $t_x = (x - x_a)/w_a$, $t_y = (y-y_a)/h_a$, $t_w = \log(w/w_a)$, $t_h = \log(h/h_a)$; decoding reverses it.`,
        "Loss: classification (CE with hard negative mining, or focal loss) + smooth L1 / GIoU on positives, divided by the number of positives.",
        "NMS: keep the best, suppress IoU > threshold, repeat; per class. Soft-NMS lowers scores instead of deleting.",
        "Two-stage: R-CNN → Fast R-CNN (RoI pooling) → Faster R-CNN (RPN + anchors) → Mask R-CNN. One-stage: YOLO, SSD, RetinaNet, FCOS/CenterNet. Set prediction: DETR (Hungarian matching, no NMS).",
        "Evaluation: AP = area under the non-increasing PR envelope; mAP@0.5 vs COCO AP@[.5:.95]. Our tiny SSD: 0.86 vs 0.45.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Semantic, Instance and Panoptic Segmentation](/learn/m12/segmentation) — from boxes to per-pixel masks: FCN, U-Net, DeepLab, and Mask R-CNN with RoIAlign.",
        "[Faces, Pose, Optical Flow, Super-Resolution and Captioning](/learn/m12/vision-tasks) — tasks built on detectors (face detection, top-down pose estimation).",
        "[Transformer families](/learn/m11/transformer-families) — ViT and Swin backbones used in modern detectors, and the attention machinery behind DETR.",
      ] },
    ],
  },
};

export default lesson;
