import type { Lesson } from "@/lib/types";

const TL_NUMPY = String.raw`import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # small matrices: one BLAS thread is faster and bit-reproducible
import numpy as np
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)
digits = load_digits()


def jitter(images):
    """Each 8x8 digit at a random position on a 12x12 canvas, plus pixel noise -> 144 features in [0, 1]."""
    out = np.zeros((len(images), 12, 12))
    for i, (r, c) in enumerate(rng.integers(0, 5, size=(len(images), 2))):
        out[i, r:r + 8, c:c + 8] = images[i] / 16.0
    return np.clip(out + rng.normal(0, 0.2, out.shape), 0, 1).reshape(len(images), 144)


# SOURCE: digits 0-4 in 4 rotations = 20 classes.  TARGET: upright digits 5-9 (relabelled 0-4), few labels.
base, y = digits.images, digits.target
Xs = np.concatenate([jitter(np.rot90(base[y < 5], r, axes=(1, 2))) for r in range(4)])
ys = np.concatenate([y[y < 5] + 5 * r for r in range(4)])
Xt, yt = jitter(base[y >= 5]), y[y >= 5] - 5
perm = rng.permutation(len(Xt))
test_idx, pool_idx = perm[:450], perm[450:]


def init(sizes, g):
    """He-initialised weights and zero biases: [W1, b1, W2, b2, ...]."""
    p = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        p += [g.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)), np.zeros(n_out)]
    return p


def forward(p, x):
    """Backbone = all layers but the last (ReLU); head = last layer (logits). Returns every activation."""
    acts = [x]
    for i in range(0, len(p) - 2, 2):
        acts.append(np.maximum(acts[-1] @ p[i] + p[i + 1], 0))
    return acts, acts[-1] @ p[-2] + p[-1]


def step(p, x, y, lrs):
    """One full-batch gradient step with softmax cross-entropy; lrs holds one learning rate per tensor."""
    acts, logits = forward(p, x)
    pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    delta = pr; delta[np.arange(len(y)), y] -= 1; delta /= len(y)        # dL/dlogits = (p - onehot) / N
    for i in range(len(p) - 2, -1, -2):                                  # walk the layers backwards
        a_in = acts[i // 2]
        gW, gb = a_in.T @ delta, delta.sum(0)
        if i > 0:
            delta = (delta @ p[i].T) * (a_in > 0)                        # back through the ReLU
        p[i] -= lrs[i] * gW
        p[i + 1] -= lrs[i + 1] * gb


def acc(p, x, y):
    return np.mean(forward(p, x)[1].argmax(1) == y)


# ---------------- 1. pretrain a 144-128-64-20 MLP on the source task ----------------
src = init([144, 128, 64, 20], rng)
order = rng.permutation(len(Xs)); tr_s, te_s = order[:3200], order[3200:]
for epoch in range(30):
    for b in np.array_split(rng.permutation(tr_s), 50):                 # mini-batches of 64
        step(src, Xs[b], ys[b], [0.1] * 6)
print(f"source (20 classes) held-out accuracy: {acc(src, Xs[te_s], ys[te_s]):.3f}")


# ---------------- 2. transfer to the target with k labels per class ----------------
def run(method, k, seed, steps=200):
    g = np.random.default_rng(seed)
    tr = np.concatenate([g.choice(pool_idx[yt[pool_idx] == c], k, replace=False) for c in range(5)])
    head = init([64, 5], g)                                  # the new task always gets a fresh head
    if method == "scratch":
        p, lrs = init([144, 128, 64], g) + head, [0.1] * 6
    else:
        p = [w.copy() for w in src[:4]] + head               # copy the pretrained backbone
        backbone_lr = {"linear probe": 0.0, "fine-tune all": 0.1, "low backbone LR": 0.01}[method]
        lrs = [backbone_lr] * 4 + [0.1] * 2
    for _ in range(steps):
        step(p, Xt[tr], yt[tr], lrs)
    return acc(p, Xt[test_idx], yt[test_idx])


print(f"\ntarget accuracy on {len(test_idx)} test images (mean over 5 seeds)")
print(f"{'method':17s}{'k=1':>7s}{'k=5':>7s}{'k=20':>7s}")
for method in ["scratch", "linear probe", "fine-tune all", "low backbone LR"]:
    print(f"{method:17s}" + "".join(f"{np.mean([run(method, k, s) for s in range(5)]):7.3f}" for k in (1, 5, 20)))


# ---------------- 3. which layer transfers best? linear probe on each layer's features ----------------
def probe(feats, k=5, seed=0, steps=300):
    g = np.random.default_rng(seed)
    tr = np.concatenate([g.choice(pool_idx[yt[pool_idx] == c], k, replace=False) for c in range(5)])
    p = init([feats.shape[1], 5], g)
    for _ in range(steps):
        step(p, feats[tr], yt[tr], [0.1, 0.1])
    return acc(p, feats[test_idx], yt[test_idx])


acts, _ = forward(src, Xt)
for name, f in [("raw pixels (144)", acts[0]), ("hidden 1 (128)", acts[1]), ("hidden 2 (64)", acts[2])]:
    print(f"5-shot linear probe on {name:17s}: {np.mean([probe(f, seed=s) for s in range(5)]):.3f}")`;

const TL_NUMPY_OUT = String.raw`source (20 classes) held-out accuracy: 0.490

target accuracy on 450 test images (mean over 5 seeds)
method               k=1    k=5   k=20
scratch            0.247  0.282  0.364
linear probe       0.284  0.344  0.459
fine-tune all      0.288  0.340  0.454
low backbone LR    0.287  0.351  0.453
5-shot linear probe on raw pixels (144) : 0.290
5-shot linear probe on hidden 1 (128)   : 0.324
5-shot linear probe on hidden 2 (64)    : 0.340`;

const TL_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

torch.set_num_threads(1)                      # tiny tensors: one thread is fastest and reproducible
gen = torch.Generator().manual_seed(0)
digits = load_digits()
labels = torch.tensor(digits.target)


def jitter(images):
    """Drop each 8x8 digit at a random position on a 12x12 canvas and add pixel noise:
    a harder variant of sklearn digits whose solution needs translation-robust features."""
    x = torch.tensor(images / 16.0, dtype=torch.float32)
    out = torch.zeros(len(x), 1, 12, 12)
    for i, (r, c) in enumerate(torch.randint(0, 5, (len(x), 2), generator=gen).tolist()):
        out[i, 0, r:r + 8, c:c + 8] = x[i]
    return (out + 0.2 * torch.randn(out.shape, generator=gen)).clamp(0, 1)


# ---------------- SOURCE: digits 0-4 in 4 rotations = 20 classes (a richer label space, like ImageNet's 1000) ----
base = digits.images[digits.target < 5]
Xs = torch.cat([jitter(np.rot90(base, r, axes=(1, 2)).copy()) for r in range(4)])
ys = torch.cat([labels[labels < 5] + 5 * r for r in range(4)])
order = torch.randperm(len(Xs), generator=gen)
Xs, ys = Xs[order], ys[order]
N_SRC = 3200                                  # training images; the remaining ~400 are a held-out check

# ---------------- TARGET: upright digits 5-9 (relabelled 0-4), only k labels per class ----------------
Xt, yt = jitter(digits.images[digits.target >= 5]), labels[labels >= 5] - 5
rng = np.random.default_rng(0)
perm = rng.permutation(len(Xt))
test_idx, pool_idx = perm[:450], perm[450:]   # fixed target test set; few-shot sets are drawn from the pool
PIX = torch.tensor(rng.permutation(144))      # one fixed pixel permutation -> the "far" source domain


def shuffle_pixels(x):
    return x.reshape(len(x), 144)[:, PIX].reshape(-1, 1, 12, 12)


def make_backbone():                          # conv blocks + global average pooling -> 32 features
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())


def pretrain(Xsrc, epochs=6):
    torch.manual_seed(0)
    model = nn.Sequential(make_backbone(), nn.Linear(32, 20))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        perm_s = torch.randperm(N_SRC)
        for i in range(0, N_SRC, 64):
            b = perm_s[i:i + 64]
            loss = nn.functional.cross_entropy(model(Xsrc[b]), ys[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xsrc[N_SRC:]).argmax(1) == ys[N_SRC:]).float().mean().item()
    return model[0].state_dict(), acc


near_state, near_acc = pretrain(Xs)
far_state, far_acc = pretrain(shuffle_pixels(Xs))
print(f"source held-out accuracy (20 classes): near {near_acc:.3f}, far (shuffled pixels) {far_acc:.3f}")


def transfer(state, method, k, seed, steps=60):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    tr = np.concatenate([g.choice(pool_idx[yt[pool_idx].numpy() == c], k, replace=False) for c in range(5)])
    bb, head = make_backbone(), nn.Linear(32, 5)             # the target task always gets a NEW head
    if state is not None:
        bb.load_state_dict(state)
    if method == "linear probe":
        bb.requires_grad_(False)                             # frozen weights ...
        bb.eval()                                            # ... and frozen BatchNorm statistics
        with torch.no_grad():
            feats, test_feats = bb(Xt[tr]), bb(Xt[test_idx])   # features never change: compute once
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(3 * steps):
            loss = nn.functional.cross_entropy(head(feats), yt[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        return (head(test_feats).argmax(1) == yt[test_idx]).float().mean().item()
    if method == "low backbone LR":                          # discriminative learning rates (param groups)
        groups = [{"params": bb.parameters(), "lr": 1e-3}, {"params": head.parameters(), "lr": 1e-2}]
    else:                                                    # scratch / fine-tune all: one LR everywhere
        groups = [{"params": [*bb.parameters(), *head.parameters()], "lr": 1e-2}]
    opt = torch.optim.Adam(groups)
    bb.train()
    for _ in range(steps):
        loss = nn.functional.cross_entropy(head(bb(Xt[tr])), yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    bb.eval()
    with torch.no_grad():
        return (head(bb(Xt[test_idx])).argmax(1) == yt[test_idx]).float().mean().item()


configs = [("from scratch", None, "scratch"),
           ("linear probe", near_state, "linear probe"),
           ("fine-tune all, one LR", near_state, "fine-tune all"),
           ("fine-tune, low backbone LR", near_state, "low backbone LR"),
           ("FAR source: linear probe", far_state, "linear probe"),
           ("FAR source: fine-tune low LR", far_state, "low backbone LR")]
SHOTS, SEEDS = (1, 5, 20), range(3)
print(f"\ntarget = digits 5-9, {len(test_idx)} test images, accuracy % (mean +- std over {len(SEEDS)} seeds)")
print(f"{'labels per class':29s}" + "".join(f"{k:>12d}" for k in SHOTS))
for name, state, method in configs:
    accs = [np.array([transfer(state, method, k, s) for s in SEEDS]) * 100 for k in SHOTS]
    print(f"{name:29s}" + "".join(f"{a.mean():7.1f} +-{a.std():3.1f}" for a in accs))`;

const TL_TORCH_OUT = String.raw`source held-out accuracy (20 classes): near 0.666, far (shuffled pixels) 0.099

target = digits 5-9, 450 test images, accuracy % (mean +- std over 3 seeds)
labels per class                        1           5          20
from scratch                    23.6 +-1.0   43.4 +-5.3   67.5 +-4.5
linear probe                    35.6 +-2.9   55.4 +-0.9   67.6 +-2.8
fine-tune all, one LR           39.0 +-5.4   62.4 +-7.1   81.9 +-0.4
fine-tune, low backbone LR      38.1 +-1.4   61.3 +-3.5   81.1 +-1.3
FAR source: linear probe        23.4 +-1.1   30.8 +-1.1   37.4 +-1.4
FAR source: fine-tune low LR    25.5 +-2.7   42.7 +-1.3   57.2 +-2.7`;

const TL_RECIPE = String.raw`import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)
NUM_CLASSES = 5

# In a real project:  weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
#                     model = torchvision.models.resnet18(weights=weights)   # downloads ~45 MB once
#                     preprocess = weights.transforms()                       # the matching resize/crop/normalise
# This lesson never downloads weights, so we build the same architecture with random weights.
model = torchvision.models.resnet18(weights=None)
print("original head:", model.fc)

# 1. Replace the task-specific head (1000 ImageNet classes -> our classes)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)


def count(m, trainable_only=True):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


# 2. Feature extraction: freeze everything except the new head
model.requires_grad_(False)
model.fc.requires_grad_(True)
print(f"feature extraction: {count(model):,} trainable of {count(model, False):,} parameters")


# 3. Keep BatchNorm statistics frozen too: put BN layers in eval mode after every model.train() call
def freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


model.train(); freeze_bn(model)
x = torch.randn(4, 3, 224, 224)
before = model.bn1.running_mean.clone()
model(x)
print("BN running mean unchanged by a forward pass:", torch.equal(before, model.bn1.running_mean))

# 4. Gradual unfreezing: unfreeze from the top (task-specific) down to the bottom (generic)
stages = [model.layer4, model.layer3, model.layer2, model.layer1]
for i, stage in enumerate(stages[:2], start=1):
    stage.requires_grad_(True)
    print(f"after unfreezing stage {i}: {count(model):,} trainable")

# 5. Discriminative learning rates with parameter groups (only trainable parameters are passed)
base_lr = 1e-3
groups = [{"params": model.fc.parameters(), "lr": base_lr},
          {"params": model.layer4.parameters(), "lr": base_lr / 3},
          {"params": model.layer3.parameters(), "lr": base_lr / 9}]
opt = torch.optim.AdamW(groups, weight_decay=0.01)
for g in opt.param_groups:
    print(f"group: {sum(p.numel() for p in g['params']):>10,} params  lr {g['lr']:.2e}")

# One training step to show that frozen layers really do not move
y = torch.randint(0, NUM_CLASSES, (4,))
w_frozen, w_l4 = model.conv1.weight.clone(), model.layer4[0].conv1.weight.clone()
loss = nn.functional.cross_entropy(model(x), y)
opt.zero_grad(); loss.backward(); opt.step()
print("conv1 (frozen) changed:", not torch.equal(w_frozen, model.conv1.weight),
      "| layer4 (unfrozen) changed:", not torch.equal(w_l4, model.layer4[0].conv1.weight),
      "| conv1.weight.grad is None:", model.conv1.weight.grad is None)`;

const TL_RECIPE_OUT = String.raw`original head: Linear(in_features=512, out_features=1000, bias=True)
feature extraction: 2,565 trainable of 11,179,077 parameters
BN running mean unchanged by a forward pass: True
after unfreezing stage 1: 8,396,293 trainable
after unfreezing stage 2: 10,496,005 trainable
group:      2,565 params  lr 1.00e-03
group:  8,393,728 params  lr 3.33e-04
group:  2,099,712 params  lr 1.11e-04
conv1 (frozen) changed: False | layer4 (unfrozen) changed: True | conv1.weight.grad is None: True`;

const TL_KERAS = String.raw`import tensorflow as tf

NUM_CLASSES = 5
# weights="imagenet" would download a checkpoint; weights=None builds the same architecture untrained.
base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3), pooling="avg")

# --- Stage 1: feature extraction (frozen backbone, new head) ---
base.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)   # the preprocessing the backbone was trained with
x = base(x, training=False)          # training=False keeps BatchNorm in inference mode even after unfreezing
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
print("trainable weights, stage 1:", len(model.trainable_weights))

# --- Stage 2: fine-tune the top of the backbone with a much smaller learning rate ---
base.trainable = True
for layer in base.layers[:-30]:      # keep all but the last 30 layers frozen
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),     # recompile after changing trainable flags
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
print("trainable weights, stage 2:", len(model.trainable_weights))
# model.fit(train_ds, validation_data=val_ds, epochs=...) in each stage`;

const lesson: Lesson = {
  id: "m12/transfer-learning",
  title: "Image Classification, Transfer Learning and Fine-Tuning",
  summary:
    "Why features learned on one image task can be reused on another, and how to do it well: feature extraction vs fine-tuning, linear probes, discriminative learning rates, gradual unfreezing, BatchNorm handling, matching the pretraining preprocessing, and picking a strategy from dataset size and domain similarity. Includes a real source→target experiment (scratch vs linear probe vs fine-tuning, near vs far source) over several seeds.",
  minutes: 95,
  level: "intermediate",
  prerequisites: ["m08/cnn-architectures", "m08/images-as-tensors", "m07/normalization", "m07/regularization", "m06/momentum-adaptive"],
  next: ["m12/object-detection", "m12/segmentation", "m15/self-supervised", "m19/scaling-efficiency"],
  terms: ["transfer-learning", "fine-tuning", "feature-extraction", "linear-probe", "discriminative-learning-rates", "gradual-unfreezing", "catastrophic-forgetting", "domain-shift", "negative-transfer"],
  sections: {
    intro: [
      { t: "p", text: "**Image classification** gives a whole image one label from a fixed set. A tensor $x \\in \\mathbb{R}^{3\\times H\\times W}$ goes in; a probability for each of $K$ classes comes out. The usual pipeline: preprocess the image (resize, crop, normalise); pass it through a convolutional or transformer **backbone** that turns pixels into a feature vector; then a small **head** (usually one linear layer) turns that vector into $K$ logits, trained with softmax cross-entropy." },
      { t: "p", text: "Training a backbone from random weights needs lots of labelled data — ImageNet-1k has about 1.28 million training images — and lots of compute. Most real problems have neither: a hospital may have 2,000 labelled X-rays, a factory 300 photos of faulty parts. **Transfer learning** bridges the gap: take a network already trained on a big **source** task, keep what it learned, and adapt it to the small **target** task. There are two basic ways: **feature extraction** (freeze the pretrained backbone, train only a new head) and **fine-tuning** (keep training some or all pretrained weights on target data, usually with a smaller learning rate)." },
      { t: "p", text: "It is the default start for almost every applied vision system — medical, satellite, retail, factory inspection — and for the backbones inside detectors and segmenters (see [Object Detection](/learn/m12/object-detection) and [Segmentation](/learn/m12/segmentation)). The same idea — pretrain big, adapt small — later became the basis of language models and parameter-efficient fine-tuning ([LoRA and PEFT](/learn/m19/scaling-efficiency))." },
      { t: "callout", kind: "key", text: "Transfer learning reuses a representation, not a decision rule. The backbone's features are kept (frozen or gently updated). The head that maps features to classes is always new, because the target classes are different." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Classic CNN Architectures](/learn/m08/cnn-architectures) — what a backbone (ResNet, EfficientNet, ConvNeXt) is, its stages, and where the classifier head sits. Here we reuse backbones instead of designing them.",
        "[Images as Tensors](/learn/m08/images-as-tensors) — channel order, value range and per-channel mean/std normalisation. A pretrained model only works if you copy its exact preprocessing.",
        "[Batch, Layer and Group Normalization](/learn/m07/normalization) — BatchNorm's running statistics and train vs eval behaviour, the most common source of silent fine-tuning bugs.",
        "[Regularization](/learn/m07/regularization) — weight decay, dropout, early stopping and data augmentation. With few target labels, regularization decides whether fine-tuning helps or overfits.",
        "[Momentum and Adam](/learn/m06/momentum-adaptive) and [Learning-Rate Schedules](/learn/m06/lr-schedules) — fine-tuning is mostly about *how far* the optimizer may move pretrained weights.",
        "[Backpropagation](/learn/m04/backpropagation) — why a frozen layer (`requires_grad=False`) gets no gradient and costs no optimizer memory.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A trained musician learning a new instrument", text: "A pianist taking up the violin does not relearn notes, rhythm or chords — that knowledge carries over; only the fingering is new. Feature extraction lets the pianist learn just the fingering chart (the head) while their musical understanding stays fixed. Fine-tuning lets their habits shift a little too, which helps because some piano habits are wrong for the violin. But practise too hard with too few lessons, and they may lose what they knew (catastrophic forgetting) or copy quirks of the few examples they saw (overfitting)." },
      { t: "p", text: "A CNN trained on many everyday images learns a **hierarchy** of features. First layers react to edges, colour blobs and simple textures. Middle layers react to corners, repeated patterns and object parts. Last layers react to combinations specific to the training classes. Edges and textures help almost any image task, so early layers are **generic**; last layers are **specific** to the source labels, so they transfer less well. Yosinski, Clune, Bengio and Lipson (2014, *How transferable are features in deep neural networks?*) measured this by moving the first $n$ layers of an ImageNet network into a second network: transfer got worse as $n$ grew and as the two tasks grew more different." },
      { t: "p", text: "Two things decide what to do: **how much labelled target data** you have, and **how similar** target images are to source images. Little data, similar images: freeze nearly everything and train a head — the features already fit, and there is too little data to safely change millions of weights. Lots of data or very different images: fine-tune more layers, since the features must change and the data can support it." },
    ],
    mechanism: [
      { t: "h", text: "The image classification pipeline, with a pretrained backbone" },
      { t: "steps", items: [
        { title: "1. Load a pretrained backbone and note its preprocessing", text: "Input size, resize/crop rule, channel order (RGB vs BGR), value range and per-channel mean/std (torchvision ImageNet models: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) on [0,1] RGB). The weights only make sense on inputs with these statistics." },
        { title: "2. Cut off the source head", text: "Remove the last layer(s) that map features to the 1,000 source classes (`model.fc` in ResNet, `model.classifier` in VGG/EfficientNet/MobileNet/ConvNeXt). What remains turns an image into a feature vector $h = f_\\theta(x) \\in \\mathbb{R}^d$ (512 for ResNet-18, 2048 for ResNet-50)." },
        { title: "3. Attach a new head", text: "Usually one linear layer $g_\\phi(h) = W h + b$ with $W \\in \\mathbb{R}^{K\\times d}$, randomly initialised, maybe with dropout before it." },
        { title: "4. Decide what is trainable", text: "Feature extraction: freeze all of $\\theta$ (`requires_grad_(False)`), train only $\\phi$. Fine-tuning: unfreeze all of $\\theta$, or only the top $k$ stages. Frozen parameters get no gradient and need no optimizer state." },
        { title: "5. Decide BatchNorm behaviour", text: "Either keep BN layers in eval mode (use the pretrained running mean/variance — needed when batches are small), or let them update their statistics on target batches (reasonable with large batches and a different domain)." },
        { title: "6. Choose learning rates", text: "The new head starts random and needs a normal learning rate. Pretrained layers need a smaller one (typically 3–100× smaller), and lower layers often smaller still (discriminative learning rates)." },
        { title: "7. Train with augmentation and early stopping", text: "Random resized crops, flips and colour jitter make the dataset effectively bigger. Validation accuracy decides when to stop, because a small target set overfits fast." },
        { title: "8. Optionally unfreeze more and continue", text: "Gradual unfreezing: once the head has settled, unfreeze the top stage, train a bit, unfreeze the next, and so on. This keeps big gradients from a random head away from well-trained features." },
      ] },
      { t: "h", text: "Feature extraction vs fine-tuning" },
      { t: "p", text: "In **feature extraction** the backbone is a fixed function, so you can run it **once** over the dataset, save $h$ per image, and train the head on the saved vectors — as cheap as logistic regression. The price: features cannot adapt, and augmentation means recomputing them. When the head is one linear layer trained this way, it is called a **linear probe**. It is also the standard way to *measure* how good a representation is (self-supervised methods are compared by linear-probe accuracy — see [Self-Supervised Learning](/learn/m15/self-supervised))." },
      { t: "p", text: "In **fine-tuning** gradients flow into the backbone and its weights change. With enough data this usually wins on accuracy, but it costs a full backward pass and optimizer state per unfrozen parameter, and can overfit or overwrite useful features. A common middle path is **LP-FT**: train a linear probe until it converges, then fine-tune everything starting from that head. Kumar et al. (2022, *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*) explained why: with a random head, the early large gradients bend the features before the head is any good; a trained head avoids that." },
      { t: "h", text: "Discriminative learning rates and gradual unfreezing" },
      { t: "p", text: "Different depths need different step sizes. Early generic layers should barely move; top layers may need to change more; the new head most of all. **Discriminative (layer-wise) learning rates** give each parameter group its own rate, often shrinking geometrically with depth: $\\eta_l = \\eta_{\\text{top}}/\\xi^{L-l}$. This and **gradual unfreezing** (unfreeze one stage per epoch, top down) were made popular for language-model fine-tuning by Howard and Ruder (2018, ULMFiT), who used $\\xi = 2.6$; both work for CNNs too. In PyTorch they are just parameter groups and `requires_grad` flags." },
      { t: "h", text: "BatchNorm when fine-tuning" },
      { t: "p", text: "A BN layer stores running estimates $\\hat\\mu, \\hat\\sigma^2$ of each channel's mean and variance, built up over millions of pretraining images. In `train()` mode it normalises with the **current batch's** statistics and blends them into the running estimates (an exponential moving average). In `eval()` mode it uses the stored ones. Fine-tuning with batches of 8–32 images gives noisy batch statistics, and the stored estimates drift toward those of the small target set. Three options:" },
      { t: "list", items: [
        "**Frozen BN** — call `.eval()` on all BN layers after every `model.train()` (and optionally freeze $\\gamma, \\beta$). The standard choice for detection/segmentation fine-tuning and small batches.",
        "**Train-mode BN** — let statistics adapt. Can help when the target's pixel statistics differ a lot and batches are large.",
        "**GroupNorm** instead of BN when batches are tiny.",
      ] },
      { t: "p", text: "In Keras, calling the base model with `training=False` keeps BN in inference mode even when its weights are trainable. See [Normalization](/learn/m07/normalization) for the full train/eval details." },
      { t: "h", text: "Matching the pretraining preprocessing — and augmenting" },
      { t: "p", text: "A pretrained model's first convolution is tuned to inputs with a certain mean and scale. Feeding $[0, 255]$ values, BGR instead of RGB, or skipping mean/std normalisation shifts every later activation. Outputs still come out, so the bug is silent — just poor accuracy. Always reuse the exact preprocessing (in torchvision, `weights.transforms()`). For augmentation, choose changes that keep the label true in the target domain. Horizontal flips are fine for animals but wrong for text, or for chest X-rays where left/right anatomy matters. Colour jitter is fine for photos but may destroy the signal in stained pathology slides." },
    ],
    math: [
      { t: "h", text: "Setup and notation" },
      { t: "p", text: "Let $x$ be an image, $f_\\theta : x \\mapsto h \\in \\mathbb{R}^d$ the backbone with parameters $\\theta$, and $g_\\phi(h) = Wh + b$ the head with $W \\in \\mathbb{R}^{K\\times d}$, $b \\in \\mathbb{R}^K$, $\\phi = (W, b)$, for $K$ target classes. Pretraining gave $\\theta_0$ on the source data. With $n$ labelled target examples $(x_i, y_i)$, transfer learning minimises" },
      { t: "math", tex: String.raw`\min_{\theta \in \Theta,\ \phi}\ \frac{1}{n}\sum_{i=1}^{n} \ell\big(g_\phi(f_\theta(x_i)),\, y_i\big) \;+\; \lambda\, R(\theta, \theta_0),\qquad \theta \text{ initialised at } \theta_0`, caption: "$\\ell$ is softmax cross-entropy; $\\Theta$ is the set of allowed backbone values; $R$ is an optional penalty keeping $\\theta$ near $\\theta_0$, weighted by $\\lambda \\ge 0$." },
      { t: "p", text: "In words: fit the target data, starting from the pretrained weights, maybe with a penalty for drifting from them. The strategies differ only in what $\\Theta$ allows:" },
      { t: "list", items: [
        "**Feature extraction / linear probe**: $\\Theta = \\{\\theta_0\\}$ (nothing moves). The problem becomes multinomial logistic regression on fixed features $h_i = f_{\\theta_0}(x_i)$ — **convex** in $\\phi$, so no bad local minima and very few hyperparameters.",
        "**Partial fine-tuning**: only the top $k$ stages may change; lower stages stay at $\\theta_0$.",
        "**Full fine-tuning**: everything may change. Only the starting point, the small learning rate, the limited number of steps (implicit regularisation) and optionally $R$ keep $\\theta$ near $\\theta_0$.",
        "**L2-SP** (Li, Grandvalet and Davoine, 2018): $R(\\theta, \\theta_0) = \\tfrac{1}{2}\\lVert\\theta - \\theta_0\\rVert_2^2$ — weight decay toward the pretrained weights instead of toward zero. Plain weight decay $\\tfrac12\\lVert\\theta\\rVert^2$ pulls pretrained weights to 0, the wrong prior for transfer.",
      ] },
      { t: "h", text: "Why freezing reduces overfitting: counting parameters" },
      { t: "p", text: "A linear probe on ResNet-18 features ($d = 512$) with $K = 5$ classes trains $dK + K = 512\\cdot5 + 5 = 2{,}565$ parameters; full fine-tuning trains all $11{,}179{,}077$ (the PyTorch program below prints both). With 25 labelled images the first is well-posed; the second has ~450,000 parameters per image and relies entirely on the pretrained start and the optimizer to avoid memorising them. A frozen parameter also needs no gradient buffer and no Adam moments: about a third of the training memory per frozen weight (the weight only, instead of weight + gradient + 2 Adam moments), and no backward pass through the frozen part." },
      { t: "h", text: "Gradients when part of the network is frozen" },
      { t: "p", text: "With a frozen backbone, the chain rule stops at the features: $\\partial\\mathcal{L}/\\partial W = (p - y)\\,h^\\top$ (softmax-CE, one example) and $\\partial\\mathcal{L}/\\partial b = p - y$, where $p$ is the softmax output and $y$ the one-hot label. Nothing flows into $f_\\theta$. When fine-tuning, the gradient reaching the backbone is $\\partial\\mathcal{L}/\\partial h = W^\\top(p - y)$. At the start $W$ is random, so this signal is basically **noise as large as a real gradient**, pushed into well-trained features. That is why we train the head first (LP-FT), unfreeze gradually, or give the backbone a much smaller learning rate." },
      { t: "h", text: "Discriminative learning rates: a worked example" },
      { t: "p", text: "Four parameter groups (head, stage 4, stage 3, stage 2), top learning rate $\\eta_{\\text{top}} = 10^{-3}$, shrink factor $\\xi = 2.6$:" },
      { t: "math", tex: String.raw`\eta_{\text{head}} = 10^{-3},\quad \eta_{4} = \frac{10^{-3}}{2.6} = 3.846\times10^{-4},\quad \eta_{3} = \frac{10^{-3}}{2.6^2} = 1.479\times10^{-4},\quad \eta_{2} = \frac{10^{-3}}{2.6^3} = 5.690\times10^{-5}` },
      { t: "p", text: "Stage 2 moves about $17.6\\times$ more slowly than the head ($2.6^3 = 17.576$). Adam changes each weight by roughly $\\eta$ per step whatever the gradient's size, so the learning rate directly limits how far each group can travel from $\\theta_0$ in $T$ steps: at most about $\\eta T$ — $0.0569$ for stage 2 versus $1.0$ for the head after 1,000 steps." },
      { t: "h", text: "BatchNorm statistics drift: a worked example" },
      { t: "p", text: "A BN channel has pretrained running mean $\\hat\\mu = 0.5$. A fine-tuning batch in train mode has mean $2.0$ for that channel (different domain). With PyTorch's default momentum $0.1$, one step gives $\\hat\\mu \\leftarrow 0.9\\cdot0.5 + 0.1\\cdot2.0 = 0.65$; after $t$ such steps $\\hat\\mu = 2.0 - 1.5\\cdot0.9^t$. After 20 steps it is $1.82$: the statistics the next layers were trained with are gone, whether or not the target set is big enough to estimate them well. In eval (frozen) mode $\\hat\\mu$ stays at $0.5$." },
      { t: "h", text: "Choosing a strategy: dataset size × similarity" },
      { t: "table", head: ["", "Target similar to source", "Target different from source"], rows: [
        ["**Small target set** (tens to a few thousand images)", "Feature extraction / linear probe; maybe fine-tune the top stage with a small LR. Overfitting is the main risk.", "Hardest case. Use features from an *earlier*, more generic layer; fine-tune top stages carefully with strong augmentation; or find a closer pretrained model (e.g. one pretrained on medical or satellite images)."],
        ["**Large target set** (tens of thousands or more)", "Fine-tune the whole network with discriminative LRs; converges fast and beats training from scratch.", "Fine-tune everything (a pretrained start still usually speeds convergence), or train from scratch if the domain is extremely different and data is plentiful; compare both on validation."],
      ] },
      { t: "p", text: "Three terms to know:" },
      { t: "list", items: [
        "**Domain shift**: the input distribution $p(x)$ differs between source and target (everyday photos → X-rays; day → night; RGB → multispectral).",
        "**Catastrophic forgetting**: losing the source skill when fine-tuning on the target only. Harmless if you only care about the target, harmful if the model must keep both — and a sign the features were distorted.",
        "**Negative transfer**: starting from pretrained weights ends *worse* than starting from scratch. It happens when the source is too far from the target (the experiment below causes it on purpose).",
      ] },
    ],
    visual: [
      { t: "p", text: "The first simulator shows what a backbone computes: an image passing through convolution → ReLU → pooling stages, with the tensor shape at every step. Transfer learning reuses exactly these stages and replaces what comes after the last one." },
      { t: "sim", id: "cnn-pipeline", caption: "A two-stage CNN on a real image: the early feature maps respond to edges and simple patterns — the generic features that transfer across tasks." },
      { t: "list", items: [
        "Look at the first-stage maps: each highlights one kind of edge or contrast wherever it appears. Such detectors help with digits, faces, X-rays and cars alike — that is why early layers are generic.",
        "Follow the shapes: spatial size shrinks and channel count grows. The final pooled vector is the $h$ a new head takes in.",
        "Which stage would you freeze for satellite images (keep stage 1, adapt stage 2)? For more everyday-object photos (keep both)?",
      ] },
      { t: "sim", id: "receptive-field", caption: "Receptive field growth with depth: deeper units see larger image regions and therefore encode larger, more task-specific patterns." },
      { t: "list", items: [
        "Trace a first-layer unit back: it sees a small patch — a local edge or texture. A last-layer unit sees most of the image, so it can encode object-level patterns tied to the source classes.",
        "This is the geometric reason transfer gets worse with depth, and why fine-tuning usually starts from the top layers.",
      ] },
    ],
    scratch: [
      { t: "p", text: "A complete transfer experiment in NumPy with a multilayer perceptron, so every gradient is explicit. The **source** task is digits 0–4 in four rotations (20 classes — more labels force more general features, the same reason ImageNet's 1,000 classes help). The **target** task is *upright* digits 5–9, never seen by the source, with only $k$ labels per class. Both use a harder version of sklearn's 8×8 digits: each digit sits at a random spot on a 12×12 canvas with added noise. We compare training from scratch, a linear probe on the frozen backbone, fine-tuning everything, and fine-tuning with a 10× smaller backbone learning rate. They differ **only** in the per-tensor learning rates given to `step`." },
      { t: "code", lang: "python", title: "transfer_numpy.py", code: TL_NUMPY, output: TL_NUMPY_OUT },
      { t: "list", items: [
        "The pretrained backbone helps at every label budget: with 20 labels per class, all three transfer strategies reach about 0.45–0.46 versus 0.36 from scratch.",
        "An MLP has no notion of shifting, so absolute accuracies on jittered digits are low; the CNN in the PyTorch section does much better on the same data. The *relative* effect of transfer is the point here.",
        "Part 3 probes each layer: both pretrained hidden layers give better 5-shot linear probes (0.324, 0.340) than raw pixels (0.290). They have re-encoded the pixels into something easier to separate linearly for *new* digit classes.",
        "`lrs = [backbone_lr] * 4 + [0.1] * 2` is the whole difference between linear probe (0), discriminative rates (0.01) and full fine-tuning (0.1). Freezing is just a learning rate of zero (frameworks also skip the computation).",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch: the experiment", blocks: [
          { t: "p", text: "The same source and target tasks with a small CNN (three conv layers with BatchNorm, global average pooling, 32 features). Each setup runs with 3 seeds (different few-shot training sets and head initialisations); the table shows mean ± standard deviation of accuracy on 450 fixed target test images. A **far** source applies one fixed random shuffle to every source image's pixels: labels stay, but the spatial structure convolutions rely on is gone, so its features are of little use on real digits. Nothing is downloaded: the source model is trained in the script." },
          { t: "code", lang: "python", title: "transfer_experiment.py", code: TL_TORCH, output: TL_TORCH_OUT },
          { t: "p", text: "What the numbers show:" },
          { t: "list", items: [
            "**Near transfer helps most when labels are scarce.** With 1 label per class, fine-tuning gets 39.0% versus 23.6% from scratch (chance is 20%); with 5 labels, 62.4% vs 43.4%; with 20 labels, 81.9% vs 67.5%.",
            "**Linear probe vs fine-tuning.** The probe is a strong, cheap baseline at 1 label (35.6%), but its lead over scratch is gone by 20 labels (67.6% vs 67.5%) while fine-tuning climbs to 81.9%. Frozen features from a 20-class source cannot capture everything the new task needs; with enough data, fine-tuning adapts them.",
            "**Low backbone LR vs one LR.** Within one standard deviation here (e.g. 61.3 ± 3.5 vs 62.4 ± 7.1 at 5 labels): on a 3-layer network trained 60 steps, one learning rate of $10^{-2}$ does not wreck the features. Discriminative rates matter more for deep backbones and long runs; the low-LR runs do vary less at 1 and 5 labels.",
            "**A far source helps much less, and can hurt.** The shuffled-pixel source reached only 9.9% on its own 20-way task (chance 5%), so its features say almost nothing about strokes. Its linear probe is near chance at 1 label (23.4%) and far below scratch at 20 labels (37.4% vs 67.5%). Fine-tuning from it ends **below** scratch at 20 labels (57.2% vs 67.5%) — **negative transfer**: the far starting point is worse than a fresh random one.",
            "With only 3 seeds and stds of 1–7 points, gaps of a couple of points are not significant; the large gaps (near vs scratch, near vs far) are.",
          ] },
        ] },
        { label: "PyTorch: practical recipe", blocks: [
          { t: "p", text: "The everyday API on a real torchvision ResNet-18: replacing `fc`, freezing with `requires_grad_`, keeping BN statistics frozen, gradual unfreezing and discriminative parameter groups. Commented lines show how a real project loads pretrained weights (`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` downloads a checkpoint; `weights.transforms()` gives the matching preprocessing). The runnable code uses `weights=None` so nothing is downloaded — the mechanics are identical." },
          { t: "code", lang: "python", title: "finetune_recipe.py", code: TL_RECIPE, output: TL_RECIPE_OUT },
          { t: "list", items: [
            "Freezing is `requires_grad_(False)`: autograd stops there, `conv1.weight.grad` stays `None`, and no optimizer state is created for it.",
            "BN layers update their running statistics on every train-mode forward pass even if their weights are frozen — `requires_grad` does **not** freeze statistics. Call `freeze_bn` after every `model.train()`.",
            "Param groups: giving the optimizer only the trainable groups saves memory and makes each stage's learning rate explicit. For VGG, EfficientNet, MobileNet and ConvNeXt the head is `model.classifier` (a `Sequential`; replace its last `Linear`) instead of `model.fc`.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras's two-stage recipe: freeze the base (`base.trainable = False`) and train the head; then unfreeze the top layers and **recompile** with a much smaller learning rate. Calling `base(x, training=False)` keeps BatchNorm in inference mode in both stages." },
          { t: "code", lang: "python", title: "finetune_keras.py (not executed in CI)", code: TL_KERAS, output: "Prints the number of trainable weight tensors in stage 1 (only the Dense head's kernel and bias) and in stage 2 (the head plus the trainable weights among the last 30 layers of ResNet50). No training is run and nothing is downloaded with weights=None." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Strategy", "What is trained", "Pros", "Cons", "Use when"], rows: [
        ["Feature extraction (cached features)", "New head only; backbone run once", "Cheapest; convex if the head is linear; backbone cannot overfit", "Features cannot adapt; augmentation needs recomputing", "Tiny datasets, similar domain, quick baselines"],
        ["Linear probe", "One linear layer on frozen features", "Standard measure of representation quality; reproducible", "Capped by the features", "Evaluating pretrained / self-supervised models"],
        ["Partial fine-tuning (top-k stages)", "Head + last k stages", "Adapts specific features, keeps generic ones; less memory", "k is one more hyperparameter", "Medium data, fairly similar domain"],
        ["Full fine-tuning", "Everything", "Best accuracy with enough data", "Most compute/memory; can overfit or forget", "Large target sets, or any domain shift with enough labels"],
        ["Discriminative LRs", "Everything, lower LR for lower layers", "Protects generic features while adapting", "More hyperparameters", "Deep backbones, long fine-tuning"],
        ["Gradual unfreezing", "Top down, one stage at a time", "Keeps random-head gradients away from early layers", "Longer schedule", "Small data with a deep backbone"],
        ["LP-FT", "Linear probe first, then full fine-tune", "Less feature distortion; often better out-of-distribution", "Two stages", "When robustness to shift matters"],
        ["L2-SP regularisation", "Everything, penalised toward θ₀", "Explicit pull toward the pretrained solution", "Extra λ to tune", "Small target sets with full fine-tuning"],
        ["Parameter-efficient (adapters, LoRA)", "Small added modules", "Tiny per-task storage; many tasks share one backbone", "Slightly lower ceiling in some cases", "Large models, many tasks — see [PEFT](/learn/m19/scaling-efficiency)"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Medical imaging**: ImageNet-initialised CNNs fine-tuned for chest X-ray, skin, retina and pathology classification, where labels are scarce and costly. Raghu et al. (2019, *Transfusion: Understanding Transfer Learning for Medical Imaging*) found much of the benefit there comes from better-conditioned starting weights and faster convergence, more than from reusing high-level ImageNet features — a reminder that this domain is far from ImageNet.",
        "**Detection and segmentation**: Faster R-CNN, Mask R-CNN, U-Net variants and DETR use ImageNet-pretrained backbones (often with frozen BN) — see [Object Detection](/learn/m12/object-detection) and [Segmentation](/learn/m12/segmentation).",
        "**Industrial and retail recognition**: defect detection, product recognition and plant/animal species apps fine-tune a mobile backbone (MobileNet, EfficientNet) on a few hundred to a few thousand images per class.",
        "**Remote sensing**: satellite/aerial models fine-tuned from ImageNet, or from models pretrained on large unlabelled satellite archives (closer source = better transfer).",
        "**Representation evaluation**: self-supervised and vision–language models ([CLIP](/learn/m15/multimodal)) are routinely compared by linear-probe accuracy on many downstream datasets.",
      ] },
      { t: "p", text: "Kornblith, Shlens and Le (2019, *Do Better ImageNet Models Transfer Better?*) found ImageNet accuracy strongly predicts fine-tuning accuracy across a dozen datasets — though on some fine-grained tasks with enough data, the gain over scratch was small. He, Girshick and Dollár (2019, *Rethinking ImageNet Pre-training*) showed that for COCO detection, training from scratch can match ImageNet-pretrained models given enough iterations: with plenty of target data, pretraining mainly speeds convergence." },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Data efficiency**: the main win — good accuracy from tens or hundreds of labels per class, as in the experiment above.",
        "**Compute**: fine-tuning converges in a fraction of the steps needed from scratch; feature extraction with cached features is nearly free.",
        "**Strong default**: a well-chosen pretrained model plus a simple recipe beats most hand-designed models trained from scratch on small data.",
        "**Inherited constraints**: input size, normalisation, architecture and licence come with the checkpoint — and so do the biases of the source dataset (what kinds of images and people it contained).",
        "**Domain mismatch**: the further the target from the source, the smaller the gain; it can even turn negative (shown above with the far source).",
        "**Hyperparameter sensitivity**: learning rate, number of unfrozen stages, BN mode and number of epochs interact. With tiny validation sets it is easy to overfit the hyperparameters themselves.",
        "**Forgetting**: full fine-tuning erases the source skill. If you need both, use frozen features, adapters, or multi-task training with rehearsal.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Mismatched preprocessing** (no mean/std normalisation, BGR instead of RGB, 0–255 instead of 0–1, wrong input size). The model silently sees unfamiliar inputs. Fix: use the checkpoint's own transforms (`weights.transforms()` in torchvision, `preprocess_input` in Keras).",
        "**Freezing the weights but not BatchNorm.** `requires_grad_(False)` does not stop running statistics updating in train mode, so a \"frozen\" backbone still changes. Fix: put BN layers in eval mode after `model.train()`, or use `FrozenBatchNorm2d`.",
        "**One high learning rate for both the backbone and a new random head.** Big gradients from the random head wreck pretrained features in the first steps. Fix: train the head first (LP-FT), unfreeze gradually, or give the backbone a 10–100× smaller LR.",
        "**Passing frozen parameters to the optimizer with weight decay.** PyTorch skips `grad=None` parameters even with decoupled decay (AdamW), but custom loops or other frameworks may still shrink \"frozen\" weights. Fix: pass only trainable parameters.",
        "**Plain weight decay toward zero during fine-tuning** treats pretrained weights as if they were noise. Fix: small decay, or L2-SP (decay toward $\\theta_0$).",
        "**Forgetting to recompile in Keras** after changing `trainable` flags — the change only takes effect when `compile` is called again.",
        "**Augmentations that change the label** (flipping text or left/right-sensitive medical images, colour jitter on colour-coded data). Fix: pick augmentations from what truly does not matter in the target domain.",
        "**Judging a strategy on one seed with a tiny test set.** Few-shot results swing by several points between draws of the training set (std up to 7 points in our experiment). Fix: report mean ± std over several seeds and splits.",
        "**Evaluating in train mode.** Dropout and BN batch statistics make predictions depend on the batch. Fix: `model.eval()` before validation, `model.train()` after.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is transfer learning and why does it work for images?", a: "Using a network trained on a big source task as the starting point for a target task. It works because CNNs learn a feature hierarchy: early layers detect edges, colours and textures useful for almost any image; later layers combine them into parts and objects. Reusing these features means the target task needs far fewer labels to learn its own decision rule." },
      { t: "qa", level: "beginner", q: "What is the difference between feature extraction and fine-tuning?", a: "Feature extraction freezes the pretrained backbone and trains only a new head on its outputs: cheap, little overfitting, but features cannot adapt. Fine-tuning keeps training some or all backbone weights on target data, usually with a smaller learning rate: more accurate with enough data, but costlier, with a risk of overfitting and forgetting." },
      { t: "qa", level: "beginner", q: "Why must you replace the final layer of a pretrained classifier?", a: "Its output size and meaning are tied to the source classes (e.g. 1,000 ImageNet classes). The target has different classes, so you swap in a new randomly initialised layer of size $K$ that maps the backbone's features to the target classes." },
      { t: "qa", level: "intermediate", q: "How do you choose between freezing and fine-tuning given dataset size and domain similarity?", a: "Small and similar: freeze, train a head (maybe fine-tune the top stage). Small and different: use earlier (more generic) features, or fine-tune top stages carefully with strong augmentation, or find a closer pretrained model. Large and similar: fine-tune everything with discriminative LRs. Large and different: fine-tune everything, and compare with training from scratch." },
      { t: "qa", level: "intermediate", q: "What can go wrong with BatchNorm during fine-tuning, and what are the fixes?", a: "In train mode BN normalises with noisy small-batch statistics, and its moving average overwrites the well-estimated pretrained running statistics with ones from the small target set; `requires_grad=False` does not prevent this. Fixes: keep BN in eval mode (frozen statistics, optionally frozen affine parameters), use big enough batches if statistics must adapt to a new domain, or switch to GroupNorm." },
      { t: "qa", level: "intermediate", q: "What are discriminative learning rates and gradual unfreezing, and why do they help?", a: "Discriminative LRs give lower layers smaller learning rates than higher layers and the new head (e.g. dividing by 2.6 per group), so generic features move little while specific ones adapt. Gradual unfreezing trains the head first, then unfreezes stages one at a time from the top down, so the big, noisy gradients from a random head never reach the generic lower layers." },
      { t: "qa", level: "intermediate", q: "Why is a linear probe used to evaluate representations?", a: "It has very few hyperparameters, is convex (one best solution), and cannot change the features. So its accuracy measures how linearly separable the target classes are in the representation itself, not how well an expensive fine-tuning recipe was tuned." },
      { t: "qa", level: "advanced", q: "Why can fine-tuning underperform a linear probe out-of-distribution, and what is LP-FT?", a: "With a random head, early fine-tuning gradients $W^\\top(p-y)$ into the backbone are large and essentially arbitrary. They reshape features for the in-distribution training data while features for out-of-distribution inputs change in an uncoordinated way (Kumar et al., 2022, in theory and experiments). LP-FT trains a linear probe to convergence first, then fine-tunes everything from that head, so backbone gradients are meaningful from the start. It tends to keep both fine-tuning's in-distribution gains and the probe's robustness." },
      { t: "qa", level: "advanced", q: "What is negative transfer and how would you detect it?", a: "When starting from the source model gives worse target results than training from scratch on the same budget, because the source features are irrelevant or misleading (very different domain or task). Detect it by always running a from-scratch baseline (and a linear probe) over several seeds. Reduce it with a closer source, transferring only earlier layers, re-initialising top layers, or training longer from scratch. In this lesson's experiment the shuffled-pixel source caused it: 57.2% vs 67.5% from scratch at 20 labels per class." },
      { t: "qa", level: "advanced", q: "How does catastrophic forgetting arise and how can you limit it if the model must keep the source task?", a: "Gradient descent on the target loss alone has nothing that protects source performance, so weights move wherever the target gradient points and source-relevant directions get overwritten. Fixes: freeze the shared backbone and add task heads; parameter-efficient adapters/LoRA (original weights untouched); L2-SP or EWC-style penalties keeping important weights near $\\theta_0$; rehearsal (mixing source examples into target batches); small learning rates and few steps." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A linear probe on 2048-dimensional ResNet-50 features is trained for 10 classes. How many parameters does it have (weights + biases)?", answer: 20490, tolerance: 0.5, explain: "$dK + K = 2048\\cdot10 + 10 = 20{,}490$." },
        { kind: "numeric", q: "With discriminative learning rates, the head uses $\\eta = 10^{-3}$ and each lower group divides by 2.6. What is the learning rate of the third group below the head, in units of $10^{-5}$? (2 decimals)", answer: 5.69, tolerance: 0.01, explain: "$10^{-3}/2.6^3 = 5.690\\times10^{-5}$." },
        { kind: "numeric", q: "A BN running mean is 0.5; fine-tuning batches (train mode, momentum 0.1) all have mean 2.0. What is the running mean after 2 steps? (3 decimals)", answer: 0.785, tolerance: 0.001, explain: "Step 1: $0.9\\cdot0.5 + 0.1\\cdot2 = 0.65$. Step 2: $0.9\\cdot0.65 + 0.2 = 0.785$ — or directly $2 - 1.5\\cdot0.9^2$." },
        { q: "You have 300 labelled photos of 6 kinds of household objects and an ImageNet-pretrained ResNet. What is the most sensible first approach?", options: ["Train a ResNet from scratch with heavy augmentation", "Linear probe / feature extraction, then maybe fine-tune the top stage with a small LR", "Fine-tune all layers with LR 0.1", "Use the ImageNet classifier unchanged"], answer: 1, explain: "Small, similar target: frozen features already fit, and there is too little data to safely change millions of weights." },
        { q: "You froze a ResNet's weights with `requires_grad_(False)`, but your feature-extraction results change from epoch to epoch on the same images. Most likely cause?", options: ["Autograd still updates frozen weights", "BatchNorm running statistics are updating because the model is in train mode", "The head's learning rate is too small", "The images are normalised twice"], answer: 1, explain: "BN statistics are buffers, not parameters; they update in train mode whatever `requires_grad` says. Put BN layers (or the whole backbone) in eval mode." },
        { q: "In the lesson's experiment, why did fine-tuning from the shuffled-pixel source end below training from scratch with 20 labels per class?", options: ["The far source had more classes", "Its features carry almost no information about spatial stroke structure, so it is a worse start than random — negative transfer", "Linear probes always beat fine-tuning", "The test set was different"], answer: 1, explain: "A fixed pixel shuffle destroys the local structure convolutions rely on; the source model barely learned (9.9% on a 20-way task), and starting from it hurt." },
        { q: "Which statement about linear probes is true?", options: ["They update the backbone with a small LR", "They train one linear layer on frozen features, a convex problem", "They always beat fine-tuning", "They need BatchNorm in train mode"], answer: 1, explain: "A linear probe is logistic regression on fixed features — convex in the head's parameters." },
        { q: "Your ImageNet-pretrained model performs poorly; you load images with OpenCV and pass them as 0–255 floats. What are two fixes?", options: ["Raise the learning rate and train longer", "Convert BGR→RGB and apply the model's expected scaling and mean/std normalisation", "Remove BatchNorm layers", "Add more dropout"], answer: 1, explain: "OpenCV returns BGR uint8; torchvision ImageNet models expect RGB in [0,1] normalised with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)." },
      ] },
      { t: "exercise", title: "Memory budget for fine-tuning", prompt: "A backbone has 25 million float32 parameters. Training with Adam stores, for each trainable parameter, the weight, its gradient and two moment estimates. (a) How many megabytes is that for full fine-tuning? (b) For feature extraction where only a 2048×10 linear head is trainable? Ignore activations. (Use 1 MB = $10^6$ bytes.)", hint: "Each float32 is 4 bytes. Trainable parameters cost 4 floats; frozen ones 1 float (the weight itself).", solution: [
        { t: "steps", items: [
          { title: "Full fine-tuning", tex: String.raw`25\times10^6 \times 4\ \text{floats} \times 4\ \text{bytes} = 400\ \text{MB}` },
          { title: "Feature extraction: frozen weights", tex: String.raw`25\times10^6 \times 1 \times 4\ \text{bytes} = 100\ \text{MB}` },
          { title: "Feature extraction: trainable head", tex: String.raw`(2048\cdot10 + 10)\times 4 \times 4\ \text{bytes} = 20490\times16 = 327{,}840\ \text{bytes} \approx 0.33\ \text{MB}` },
          { title: "Total", tex: String.raw`\approx 100.3\ \text{MB}\ \text{vs}\ 400\ \text{MB}` },
        ] },
        { t: "p", text: "Activations shrink too with feature extraction: with no backward pass through the backbone, intermediate activations need not be stored (run it under `torch.no_grad()`)." },
      ] },
      { t: "exercise", title: "Add LP-FT and gradual unfreezing to the experiment", prompt: "Extend `transfer_experiment.py` with a method \"LP-FT\": train the head alone on frozen features for 60 steps, then fine-tune everything (backbone LR $10^{-3}$, head LR $10^{-2}$) for 60 more steps. Compare it with \"fine-tune, low backbone LR\" at 1, 5 and 20 labels per class over the same seeds. Then add gradual unfreezing (head only for 20 steps, then also the last conv block, then everything).", hint: "Build the optimizer twice, or build it once with all groups and set `requires_grad_(False)` on the backbone during the first phase — `Adam.step()` skips parameters with `grad=None`.", solution: [
        { t: "code", lang: "python", title: "lp_ft sketch (add to transfer())", code: String.raw`if method == "LP-FT":
    bb.requires_grad_(False); bb.eval()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(steps):                                   # phase 1: linear probe
        loss = nn.functional.cross_entropy(head(bb(Xt[tr])), yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    bb.requires_grad_(True); bb.train()
    opt = torch.optim.Adam([{"params": bb.parameters(), "lr": 1e-3},
                            {"params": head.parameters(), "lr": 1e-2}])
    for _ in range(steps):                                   # phase 2: fine-tune from a trained head
        loss = nn.functional.cross_entropy(head(bb(Xt[tr])), yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()

# gradual unfreezing: the backbone Sequential's last conv block is bb[7:]
# phase A: bb.requires_grad_(False); train head
# phase B: bb[7:].requires_grad_(True); train
# phase C: bb.requires_grad_(True); train`, output: "" },
        { t: "p", text: "What to look for: LP-FT should be at least as good as direct fine-tuning, and usually less variable across seeds at 1–5 labels, because the backbone never gets gradients from a random head. With such a small network the differences may fall within the seed-to-seed standard deviation — run 5–10 seeds before concluding anything, as the mistakes section advises." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Pipeline: preprocess exactly as in pretraining → backbone $f_\\theta$ → new head $g_\\phi(h) = Wh + b$ → softmax cross-entropy.",
        "Transfer works because early CNN layers learn generic features; later layers are source-specific (Yosinski et al. 2014).",
        "Feature extraction / linear probe: freeze $\\theta$, train $\\phi$ — convex, cheap, robust with few labels; capped by the features.",
        "Fine-tuning: update $\\theta$ too, with smaller (discriminative) learning rates $\\eta_l = \\eta_{\\text{top}}/\\xi^{L-l}$, optionally gradual unfreezing or LP-FT.",
        "BatchNorm: statistics update in train mode whatever `requires_grad` says; freeze them (eval mode) for small batches.",
        "Strategy by size × similarity: small + similar → freeze; large → fine-tune all; different domain → fine-tune more or pick a closer source.",
        "Measured here: near transfer 39.0% vs 23.6% (1 label/class), 81.9% vs 67.5% (20 labels/class); a far source gave negative transfer (57.2% vs 67.5%).",
        "Pitfalls: wrong preprocessing, BN drift, high LR meeting random-head gradients, label-changing augmentation, single-seed conclusions.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Object Detection](/learn/m12/object-detection) — pretrained backbones become feature extractors for boxes and classes; frozen BN and multi-scale features show up right away.",
        "[Segmentation](/learn/m12/segmentation) — the same backbones as encoders for pixel-level predictions.",
        "[Other Vision Tasks](/learn/m12/vision-tasks) — faces, pose, optical flow, super-resolution and captioning, all built on transferred backbones.",
        "[Self-Supervised and Contrastive Learning](/learn/m15/self-supervised) — pretraining without labels, judged with the linear probes introduced here.",
        "[LoRA and Parameter-Efficient Fine-Tuning](/learn/m19/scaling-efficiency) — fine-tuning when the backbone has billions of parameters.",
      ] },
    ],
  },
};

export default lesson;
