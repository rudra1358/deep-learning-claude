import type { Lesson } from "@/lib/types";

const NTXENT_NUMPY = String.raw`import numpy as np
from sklearn.linear_model import LogisticRegression


def nt_xent(H, tau):
    """NT-Xent loss for 2N un-normalised embeddings H (rows i and i+N are two views of image i).
    Returns the mean loss and dL/dH (backpropagated through the L2 normalisation)."""
    n2 = H.shape[0]
    N = n2 // 2
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    Z = H / norms                                   # unit vectors
    S = Z @ Z.T / tau                               # (2N, 2N) scaled cosine similarities
    np.fill_diagonal(S, -np.inf)                    # an embedding is never its own candidate
    pos = np.concatenate([np.arange(N, n2), np.arange(N)])   # index of each row's positive
    S_max = S.max(axis=1, keepdims=True)
    P = np.exp(S - S_max)
    P /= P.sum(axis=1, keepdims=True)               # softmax over the 2N - 1 candidates
    loss = -np.mean(np.log(P[np.arange(n2), pos]))
    G = P.copy()                                    # dL/dS = (softmax - one_hot(positive)) / 2N
    G[np.arange(n2), pos] -= 1.0
    G /= n2
    dZ = (G + G.T) @ Z / tau                        # S is symmetric in Z, so both slots contribute
    dH = (dZ - np.sum(dZ * Z, axis=1, keepdims=True) * Z) / norms   # through z = h / ||h||
    return loss, dH


# ---- 1. worked example: N = 2 images, 2 views each, 2-D embeddings ----
H = np.array([[1.0, 0.2], [0.1, 1.0],               # view 1 of image 0, view 1 of image 1
              [0.9, 0.5], [-0.2, 1.0]])             # view 2 of image 0, view 2 of image 1
loss, _ = nt_xent(H, tau=0.5)
Z = H / np.linalg.norm(H, axis=1, keepdims=True)
print("cosine similarities:\n", np.round(Z @ Z.T, 4) + 0.0)   # + 0.0 turns -0.0 into 0.0
print(f"NT-Xent (tau = 0.5) = {loss:.4f}")

# ---- 2. gradient check with central differences ----
rng = np.random.default_rng(0)
Hr = rng.standard_normal((8, 5))
_, g = nt_xent(Hr, 0.2)
num = np.zeros_like(Hr)
for idx in np.ndindex(*Hr.shape):
    Hp, Hm = Hr.copy(), Hr.copy()
    Hp[idx] += 1e-6
    Hm[idx] -= 1e-6
    num[idx] = (nt_xent(Hp, 0.2)[0] - nt_xent(Hm, 0.2)[0]) / 2e-6
print(f"max |analytic - numerical| gradient = {np.abs(g - num).max():.2e}")

# ---- 3. collapse: every embedding identical -> loss = log(2N - 1), whatever tau is ----
N = 128
for tau in (0.1, 0.5):
    print(f"collapsed batch, N = {N}, tau = {tau}: loss {nt_xent(np.ones((2 * N, 4)), tau)[0]:.4f}"
          f"  log(2N-1) = {np.log(2 * N - 1):.4f}")

# ---- 4. augmentations decide what is learned: a linear SimCLR on synthetic data ----
# 4 classes. Each point = 3 "content" dims (class signal) + 9 "nuisance" dims with LARGE variance,
# mixed by a random rotation so no single input axis is pure content or pure nuisance.
D_C, D_N, OUT = 3, 9, 3
means = rng.standard_normal((4, D_C)) * 1.5
Q, _ = np.linalg.qr(rng.standard_normal((D_C + D_N, D_C + D_N)))


def make(labels, nuisance):
    content = means[labels] + 0.3 * rng.standard_normal((len(labels), D_C))
    return np.hstack([content, nuisance]) @ Q.T


def augment(X):
    """An 'augmentation' that keeps the content but re-draws the nuisance (like a new crop / colour)."""
    U = X @ Q                                     # back to (content, nuisance) coordinates
    U[:, D_C:] = 3.0 * rng.standard_normal((len(X), D_N))
    return U @ Q.T


y_tr, y_te = rng.integers(0, 4, 2000), rng.integers(0, 4, 1000)
X_tr = make(y_tr, 3.0 * rng.standard_normal((2000, D_N)))
X_te = make(y_te, 3.0 * rng.standard_normal((1000, D_N)))

W = 0.1 * rng.standard_normal((D_C + D_N, OUT))   # linear encoder h = x W (no labels used!)
m, v = np.zeros_like(W), np.zeros_like(W)         # Adam state
for step in range(1, 1501):
    xb = X_tr[rng.choice(2000, 128, replace=False)]
    views = np.vstack([augment(xb), augment(xb)])  # 2N = 256 rows, row i pairs with row i + 128
    loss, dH = nt_xent(views @ W, tau=0.2)
    gW = views.T @ dH
    m, v = 0.9 * m + 0.1 * gW, 0.999 * v + 0.001 * gW ** 2
    W -= 0.01 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    if step in (1, 100, 500, 1500):
        print(f"step {step:4d}  NT-Xent {loss:.4f}")


def probe(F_tr, F_te):
    """Linear probe: logistic regression on frozen features, trained with the labels."""
    return LogisticRegression(max_iter=2000).fit(F_tr, y_tr).score(F_te, y_te)


mu = X_tr.mean(axis=0)
pcs = np.linalg.svd(X_tr - mu, full_matrices=False)[2][:OUT].T    # top-3 principal directions
R = rng.standard_normal((D_C + D_N, OUT))
print(f"linear probe on 3-D features:  random projection {probe(X_tr @ R, X_te @ R):.3f}   "
      f"PCA {probe((X_tr - mu) @ pcs, (X_te - mu) @ pcs):.3f}   SimCLR {probe(X_tr @ W, X_te @ W):.3f}")
print(f"linear probe on all 12 raw dims: {probe(X_tr, X_te):.3f}")
content_share = np.linalg.norm((Q.T @ W)[:D_C]) ** 2 / np.linalg.norm(W) ** 2
print(f"share of the encoder's weight energy on content directions: {content_share:.3f} (random: {D_C / (D_C + D_N):.3f})")`;
const NTXENT_NUMPY_OUT = String.raw`cosine similarities:
 [[1.     0.2927 0.9524 0.    ]
 [0.2927 1.     0.5702 0.9562]
 [0.9524 0.5702 1.     0.3048]
 [0.     0.9562 0.3048 1.    ]]
NT-Xent (tau = 0.5) = 0.4496
max |analytic - numerical| gradient = 3.84e-10
collapsed batch, N = 128, tau = 0.1: loss 5.5413  log(2N-1) = 5.5413
collapsed batch, N = 128, tau = 0.5: loss 5.5413  log(2N-1) = 5.5413
step    1  NT-Xent 7.4538
step  100  NT-Xent 4.3989
step  500  NT-Xent 3.8473
step 1500  NT-Xent 3.8486
linear probe on 3-D features:  random projection 0.388   PCA 0.250   SimCLR 0.987
linear probe on all 12 raw dims: 0.995
share of the encoder's weight energy on content directions: 1.000 (random: 0.250)`;
const SIMCLR_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=0)
T_tr = torch.tensor(X_tr, dtype=torch.float32)
T_te = torch.tensor(X_te, dtype=torch.float32)
ROWS, COLS = torch.arange(8).view(1, 8, 1), torch.arange(8).view(1, 1, 8)


def shift(x, max_shift=1):
    """Translate each 8x8 image by a random (dy, dx) in [-1, 1] pixels, filling with zeros."""
    B = x.shape[0]
    img = F.pad(x.view(B, 1, 8, 8), (max_shift,) * 4)[:, 0]          # (B, 10, 10)
    dy, dx = torch.randint(0, 2 * max_shift + 1, (2, B))
    r = (ROWS + dy.view(B, 1, 1)).expand(B, 8, 8)
    c = (COLS + dx.view(B, 1, 1)).expand(B, 8, 8)
    return img[torch.arange(B).view(B, 1, 1), r, c]                  # (B, 8, 8)


def augment(x):
    """A random view: shift, erase a 3x3 patch (half of the time), random contrast, pixel noise."""
    B = x.shape[0]
    img = shift(x)
    ey, ex = torch.randint(0, 6, (2, B)).view(2, B, 1, 1)
    patch = (ROWS >= ey) & (ROWS < ey + 3) & (COLS >= ex) & (COLS < ex + 3)
    img = img * ~(patch & (torch.rand(B, 1, 1) < 0.5))
    img = img * (0.6 + 0.8 * torch.rand(B, 1, 1)) + 0.1 * torch.randn_like(img)
    return img.reshape(B, 64)


def nt_xent(z1, z2, tau=0.5):
    z = F.normalize(torch.cat([z1, z2]), dim=1)                     # (2N, d) unit vectors
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                               # exclude self-similarity
    N = z1.shape[0]
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(N)])  # the other view is the "class"
    return F.cross_entropy(sim, targets)


torch.manual_seed(0)
encoder = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))            # representation h
projector = nn.Sequential(nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))  # z, used only by the loss
random_encoder = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))     # baseline: never trained
opt = torch.optim.Adam([*encoder.parameters(), *projector.parameters()], lr=1e-3)
for epoch in range(1, 151):
    for idx in torch.randperm(len(T_tr)).split(256):
        x = T_tr[idx]
        loss = nt_xent(projector(encoder(augment(x))), projector(encoder(augment(x))))   # NO labels used
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch in (1, 10, 50, 150):
        print(f"epoch {epoch:3d}  NT-Xent {loss.item():.4f}   (uniform guess: log(2N-1) = {np.log(2 * len(x) - 1):.4f})")

torch.manual_seed(123)
T_te_shifted = shift(T_te).reshape(-1, 64)                          # the same test digits, moved by <= 1 pixel


def probe(feat, per_class):
    """Linear probe: logistic regression on frozen features with per_class labels per digit (0 = all)."""
    rng = np.random.default_rng(0)
    idx = (np.concatenate([rng.choice(np.where(y_tr == c)[0], per_class, replace=False) for c in range(10)])
           if per_class else np.arange(len(y_tr)))
    with torch.no_grad():
        F_tr, F_te, F_sh = (feat(t).numpy() for t in (T_tr, T_te, T_te_shifted))
    clf = LogisticRegression(max_iter=5000).fit(F_tr[idx], y_tr[idx])
    return clf.score(F_te, y_te), clf.score(F_sh, y_te)


reps = {"raw pixels": lambda t: t, "random-init encoder h": random_encoder,
        "SimCLR projector z": lambda t: projector(encoder(t)), "SimCLR encoder h": encoder}
print("\nlinear probe, test accuracy (clean / shifted test images)")
print(f"{'features':<24}{'1 label/class':>16}{'5 labels/class':>17}{'all 1257 labels':>17}")
for name, feat in reps.items():
    cells = [probe(feat, k) for k in (1, 5, 0)]
    print(f"{name:<24}" + "".join(f"{a:9.3f} /{b:.3f}" for a, b in cells))`;
const SIMCLR_TORCH_OUT = String.raw`epoch   1  NT-Xent 6.1212   (uniform guess: log(2N-1) = 6.1420)
epoch  10  NT-Xent 5.3372   (uniform guess: log(2N-1) = 6.1420)
epoch  50  NT-Xent 4.9854   (uniform guess: log(2N-1) = 6.1420)
epoch 150  NT-Xent 4.7939   (uniform guess: log(2N-1) = 6.1420)

linear probe, test accuracy (clean / shifted test images)
features                   1 label/class   5 labels/class  all 1257 labels
raw pixels                  0.707 /0.346    0.844 /0.346    0.970 /0.394
random-init encoder h       0.581 /0.296    0.713 /0.328    0.893 /0.372
SimCLR projector z          0.744 /0.750    0.833 /0.813    0.854 /0.844
SimCLR encoder h            0.691 /0.474    0.839 /0.576    0.937 /0.656`;
const BYOL_TORCH = String.raw`import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=0)
T_tr, T_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
ROWS, COLS = torch.arange(8).view(1, 8, 1), torch.arange(8).view(1, 1, 8)


def augment(x):
    """Shift by <= 1 pixel, erase a 3x3 patch half of the time, random contrast, pixel noise."""
    B = x.shape[0]
    img = F.pad(x.view(B, 1, 8, 8), (1, 1, 1, 1))[:, 0]
    dy, dx = torch.randint(0, 3, (2, B))
    r, c = (ROWS + dy.view(B, 1, 1)).expand(B, 8, 8), (COLS + dx.view(B, 1, 1)).expand(B, 8, 8)
    img = img[torch.arange(B).view(B, 1, 1), r, c]
    ey, ex = torch.randint(0, 6, (2, B)).view(2, B, 1, 1)
    img = img * ~((ROWS >= ey) & (ROWS < ey + 3) & (COLS >= ex) & (COLS < ex + 3) & (torch.rand(B, 1, 1) < 0.5))
    return (img * (0.6 + 0.8 * torch.rand(B, 1, 1)) + 0.1 * torch.randn_like(img)).reshape(B, 64)


def mlp(i, h, o, bn=True):
    norm = nn.BatchNorm1d(h) if bn else nn.Identity()
    return nn.Sequential(nn.Linear(i, h), norm, nn.ReLU(), nn.Linear(h, o))


def byol_loss(p, z):
    """2 - 2 cos(p, z): squared distance between unit vectors; z is the target and gets no gradient."""
    return (2 - 2 * F.cosine_similarity(p, z.detach(), dim=1)).mean()


def run(variant, epochs=60, tau=0.99):
    torch.manual_seed(0)
    bn = variant != "BYOL without BatchNorm"
    online = nn.Sequential(mlp(64, 256, 64, bn), nn.ReLU(), mlp(64, 128, 32, bn))   # encoder h (block 0) + projector
    predictor = nn.Identity() if variant == "BYOL without predictor" else mlp(32, 128, 32, bn)
    target = copy.deepcopy(online)                                    # EMA copy of the online network
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([*online.parameters(), *predictor.parameters()], lr=1e-3)
    for epoch in range(epochs):
        for idx in torch.randperm(len(T_tr)).split(256):
            v1, v2 = augment(T_tr[idx]), augment(T_tr[idx])
            with torch.no_grad():
                z1, z2 = target(v1), target(v2)
            loss = byol_loss(predictor(online(v1)), z2) + byol_loss(predictor(online(v2)), z1)   # symmetrised
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():                                     # target <- tau * target + (1 - tau) * online
                for pt, po in zip(target.parameters(), online.parameters()):
                    pt.mul_(tau).add_((1 - tau) * po)
                for bt, bo in zip(target.buffers(), online.buffers()):
                    bt.copy_(bo)                                      # BatchNorm running statistics
    online.eval()
    with torch.no_grad():
        z = F.normalize(online(T_te), dim=1)
        h_tr, h_te = online[0](T_tr).numpy(), online[0](T_te).numpy()
    acc = LogisticRegression(max_iter=5000).fit(h_tr, y_tr).score(h_te, y_te)
    # collapse check: per-dimension std of unit-normalised outputs is about 1/sqrt(32) when spread out, 0 when collapsed
    print(f"{variant:<24} final loss {loss.item():.4f}   std of normalised z {z.std(0).mean().item():.4f}   probe on h {acc:.3f}")


print("BYOL on 8x8 digits, no negatives anywhere (1/sqrt(32) = 0.1768)")
for v in ("BYOL", "BYOL without predictor", "BYOL without BatchNorm"):
    run(v)`;
const BYOL_TORCH_OUT = String.raw`BYOL on 8x8 digits, no negatives anywhere (1/sqrt(32) = 0.1768)
BYOL                     final loss 1.1084   std of normalised z 0.1487   probe on h 0.954
BYOL without predictor   final loss 0.0142   std of normalised z 0.0134   probe on h 0.959
BYOL without BatchNorm   final loss 0.0003   std of normalised z 0.0007   probe on h 0.928`;
const SEMI_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=0)
rng = np.random.default_rng(0)
lab = np.concatenate([rng.choice(np.where(y_tr == c)[0], 5, replace=False) for c in range(10)])  # 50 labels
XL, YL = torch.tensor(X_tr[lab], dtype=torch.float32), torch.tensor(y_tr[lab])
XU = torch.tensor(np.delete(X_tr, lab, axis=0), dtype=torch.float32)        # 1207 images, labels thrown away
XT, YT = torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te)
ROWS, COLS = torch.arange(8).view(1, 8, 1), torch.arange(8).view(1, 1, 8)


def weak(x):
    """Weak augmentation: random shift by up to 1 pixel."""
    B = x.shape[0]
    img = F.pad(x.view(B, 1, 8, 8), (1, 1, 1, 1))[:, 0]
    dy, dx = torch.randint(0, 3, (2, B))
    r, c = (ROWS + dy.view(B, 1, 1)).expand(B, 8, 8), (COLS + dx.view(B, 1, 1)).expand(B, 8, 8)
    return img[torch.arange(B).view(B, 1, 1), r, c].reshape(B, 64)


def strong(x):
    """Strong augmentation: weak + a 3x3 erased patch + contrast change + pixel noise."""
    B = x.shape[0]
    img = weak(x).view(B, 8, 8)
    ey, ex = torch.randint(0, 6, (2, B)).view(2, B, 1, 1)
    img = img * ~((ROWS >= ey) & (ROWS < ey + 3) & (COLS >= ex) & (COLS < ex + 3))
    return (img * (0.6 + 0.8 * torch.rand(B, 1, 1)) + 0.15 * torch.randn_like(img)).reshape(B, 64)


def make_net():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))


def accuracy(net):
    with torch.no_grad():
        return (net(XT).argmax(1) == YT).float().mean().item()


def train(method, steps=2500, threshold=0.8, lam=1.0):
    torch.manual_seed(0)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(steps):
        loss = F.cross_entropy(net(weak(XL)), YL)                       # supervised term on the 50 labels
        if method != "supervised only" and step >= 200:                 # 200 warm-up steps on labels only
            xu = XU[torch.randint(0, len(XU), (256,))]
            if method == "pseudo-labelling":
                with torch.no_grad():
                    p = F.softmax(net(xu), dim=1)                        # predict on the clean image
                conf, pseudo = p.max(1)
                mask = conf >= threshold
                loss_u = (F.cross_entropy(net(xu), pseudo, reduction="none") * mask).mean()
            else:                                                       # FixMatch-style consistency
                with torch.no_grad():
                    p = F.softmax(net(weak(xu)), dim=1)                  # pseudo-label from a WEAK view
                conf, pseudo = p.max(1)
                mask = conf >= threshold
                loss_u = (F.cross_entropy(net(strong(xu)), pseudo, reduction="none") * mask).mean()  # STRONG view
            loss = loss + lam * loss_u
        opt.zero_grad()
        loss.backward()
        opt.step()
    # how many unlabelled images pass the threshold at the end, and how often their pseudo-label is right
    yu = torch.tensor(np.delete(y_tr, lab))
    with torch.no_grad():
        p = F.softmax(net(XU), dim=1)
    conf, pseudo = p.max(1)
    m = conf >= threshold
    return accuracy(net), m.float().mean().item(), (pseudo[m] == yu[m]).float().mean().item()


print("50 labelled + 1207 unlabelled training images, test accuracy on 540 images")
for method in ("supervised only", "pseudo-labelling", "consistency (FixMatch-style)"):
    acc, frac, purity = train(method)
    print(f"{method:<30} test acc {acc:.3f}   confident (p >= 0.8) unlabelled: {frac:.3f}, of which correct {purity:.3f}")`;
const SEMI_TORCH_OUT = String.raw`50 labelled + 1207 unlabelled training images, test accuracy on 540 images
supervised only                test acc 0.819   confident (p >= 0.8) unlabelled: 0.811, of which correct 0.907
pseudo-labelling               test acc 0.852   confident (p >= 0.8) unlabelled: 0.944, of which correct 0.890
consistency (FixMatch-style)   test acc 0.896   confident (p >= 0.8) unlabelled: 0.771, of which correct 0.957`;

const SIMCLR_KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import load_digits

X = load_digits().data.astype("float32") / 16.0


def augment(x):
    x = tf.reshape(x, (-1, 8, 8, 1))
    x = tf.image.resize_with_crop_or_pad(x, 10, 10)
    x = tf.image.random_crop(x, (tf.shape(x)[0], 8, 8, 1))        # random shift by <= 1 pixel
    x = x * tf.random.uniform((tf.shape(x)[0], 1, 1, 1), 0.6, 1.4) + tf.random.normal(tf.shape(x), stddev=0.1)
    return tf.reshape(x, (-1, 64))


encoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dense(64)])
projector = tf.keras.Sequential([tf.keras.layers.ReLU(), tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(32)])
opt = tf.keras.optimizers.Adam(1e-3)


def nt_xent(z1, z2, tau=0.5):
    z = tf.math.l2_normalize(tf.concat([z1, z2], 0), axis=1)
    n = tf.shape(z1)[0]
    sim = tf.matmul(z, z, transpose_b=True) / tau - 1e9 * tf.eye(2 * n)   # mask self-similarity
    targets = tf.concat([tf.range(n, 2 * n), tf.range(n)], 0)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, sim, from_logits=True))


ds = tf.data.Dataset.from_tensor_slices(X).shuffle(2000).batch(256)
for epoch in range(150):
    for x in ds:
        with tf.GradientTape() as tape:
            loss = nt_xent(projector(encoder(augment(x))), projector(encoder(augment(x))))
        variables = encoder.trainable_variables + projector.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
    if epoch % 50 == 0:
        print(epoch, float(loss))`;

const lesson: Lesson = {
  id: "m15/self-supervised",
  title: "Self-Supervised and Contrastive Learning",
  summary:
    "Learning useful features without labels: pretext tasks, the contrastive InfoNCE / NT-Xent loss and its gradient, SimCLR, MoCo's momentum encoder and queue, BYOL's negative-free training and the collapse problem, masked modelling, and semi-supervised learning with pseudo-labels and consistency — derived, built from scratch and trained on real digits.",
  minutes: 85,
  level: "advanced",
  prerequisites: ["m05/advanced-losses", "m14/autoencoders", "m12/transfer-learning", "m02/norms-spaces"],
  next: ["m15/learning-paradigms", "m15/multimodal", "m11/transformer-families"],
  terms: ["self-supervised-learning", "pretext-task", "contrastive-learning", "infonce", "simclr", "projection-head", "moco", "momentum-encoder", "byol", "representation-collapse", "linear-probe", "semi-supervised-learning", "pseudo-labeling", "consistency-regularization"],
  sections: {
    intro: [
      { t: "p", text: "Labels are expensive (a radiologist's notes, a hand-drawn box); raw data is cheap (billions of unlabelled images, hours of audio, trillions of words). **Self-supervised learning** (SSL) trains on unlabelled data by inventing a task whose answers come from the data itself — predict a hidden word, fill in a missing patch, or spot that two distorted crops come from the same photo. The task itself doesn't matter. Solving it forces the network to learn a **representation** (a feature vector per input) that helps the tasks we care about, which we then learn from only a few labels." },
      { t: "p", text: String.raw`The main families: **Pretext tasks** (predict rotation, solve jigsaws, colourise) came first. **Contrastive learning** — SimCLR (Chen, Kornblith, Norouzi and Hinton, 2020) and MoCo (He, Fan, Wu, Xie and Girshick, 2020) — learns an embedding where two augmented views of one image are close and views of different images are far apart, using the InfoNCE loss. **Non-contrastive** methods such as BYOL (Grill et al., 2020, "Bootstrap Your Own Latent") drop the negatives and avoid the trivial constant solution through an asymmetric architecture. **Masked modelling** — BERT for text, masked autoencoders for images — is how every large language model is pretrained (next-token prediction is self-supervision too).` },
      { t: "p", text: "A close cousin is **semi-supervised learning**: a few labelled examples plus many unlabelled ones, used *together* in one training run. Two common tricks: train on the model's own confident predictions (**pseudo-labelling**), or demand the same prediction for different augmentations of an unlabelled input (**consistency regularisation**)." },
      { t: "callout", kind: "key", text: String.raw`Contrastive loss for an embedding $z_i$ with positive $z_{j(i)}$ among $2N$ embeddings: $\ell_i = -\log \dfrac{\exp(\mathrm{sim}(z_i, z_{j(i)})/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(z_i, z_k)/\tau)}$. The augmentations decide what the representation ignores; the negatives (or an architectural trick) stop it collapsing to a constant.` },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Advanced Losses](/learn/m05/advanced-losses) — where the InfoNCE / NT-Xent loss, its temperature and its gradient were introduced; here we build full self-supervised systems with them.",
        "[Classification Losses](/learn/m05/classification-losses) — softmax cross-entropy and its gradient $p - y$. InfoNCE is a cross-entropy whose \"classes\" are the other examples in the batch.",
        "[Norms and Spaces](/learn/m02/norms-spaces) — cosine similarity and L2 normalisation.",
        "[Autoencoders](/learn/m14/autoencoders) — the oldest label-free way to learn features (reconstruction), the baseline SSL improves on.",
        "[Transfer Learning](/learn/m12/transfer-learning) — linear probing and fine-tuning a pretrained backbone, which is how SSL features get used.",
        "[Data augmentation](/learn/m07/regularization) — the random transformations that define the positive pairs.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Spot the same person in different photos", text: "Imagine learning to recognise people without ever hearing their names. You see pairs of photos and must say \"same person or not?\" — where 'same' pairs are one snapshot cropped differently, recoloured, blurred or flipped. You can't rely on the background or the framing, since those change; you must notice face shape, hair, build. After thousands of questions you have a description of people that ignores lighting and framing — exactly what you need when someone finally tells you a few names." },
      { t: "p", text: "That is contrastive learning. The 'same person' pairs are two random **augmentations** (views) of one image: the positive pair. Every other image in the batch is a negative. The loss rewards picking out the positive among all candidates. The augmentations are the teacher: whatever they change (position, colour, a missing patch), the network learns to ignore; whatever they keep (the digit's shape), it must encode." },
      { t: "p", text: "Why negatives? Without them there is a cheat: map *every* image to the same vector. All views then agree and the 'pull together' loss is zero — but the features are useless. This is **representation collapse**. Contrastive methods prevent it by also pushing different images apart. BYOL showed a carefully asymmetric design (a predictor on one side, a slowly moving copy of the network on the other) can avoid collapse without negatives." },
    ],
    mechanism: [
      { t: "h", text: "SimCLR, step by step" },
      { t: "steps", items: [
        { title: "Sample a mini-batch", text: String.raw`Take $N$ unlabelled images $x_1, \dots, x_N$.` },
        { title: "Make two views of each", text: String.raw`Apply two independently drawn random augmentations $t, t'$: $\tilde x_i = t(x_i)$, $\tilde x_{i+N} = t'(x_i)$, giving $2N$ inputs. SimCLR used random resized crops, flips, colour jitter, grayscale and blur, and found **crop plus colour distortion** the most important combination.` },
        { title: "Encode", text: String.raw`A backbone $f$ (a ResNet in the paper; an MLP in our digit example) gives representations $h_k = f(\tilde x_k)$. These are what you keep.` },
        { title: "Project", text: String.raw`A small MLP **projection head** $g$ maps $h_k$ to $z_k = g(h_k)$, typically 128-D; the loss uses $z$. SimCLR found the features **before** the head transfer much better: the head throws away information (e.g. colour) that the loss treats as noise but other tasks may need.` },
        { title: "Normalise and compare", text: String.raw`Cosine similarities $s_{kl} = z_k^\top z_l / (\lVert z_k\rVert\lVert z_l\rVert)$ for all $2N \times 2N$ pairs, divided by the temperature $\tau$.` },
        { title: "NT-Xent loss", text: String.raw`Each of the $2N$ embeddings gets a $(2N-1)$-way softmax over all others, with its partner view as the answer. Average the $2N$ cross-entropies.` },
        { title: "Update and repeat", text: "Backpropagate through head and backbone. SimCLR used huge batches (up to 8192, LARS optimiser) because more negatives per step helped." },
        { title: "Use the representation", text: String.raw`Drop $g$. Evaluate with a **linear probe** (a linear classifier on frozen $h$), by fine-tuning $f$ with a few labels, or by $k$-nearest-neighbour lookup.` },
      ] },
      { t: "h", text: "MoCo: many negatives without huge batches" },
      { t: "steps", items: [
        { title: "Two encoders", text: String.raw`A **query encoder** $f_q$ (weights $\theta_q$, trained by backprop) and a **key encoder** $f_k$ (weights $\theta_k$, no gradients).` },
        { title: "Momentum update", text: String.raw`After each step, $\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q$ ($m = 0.999$ in the paper). The key encoder is a slow moving average of the query encoder, so its outputs stay consistent over many steps.` },
        { title: "Queue of negatives", text: String.raw`Keys from earlier batches wait in a first-in-first-out queue (65,536 in the paper). Each query is compared with its positive key and every queued key — far more negatives than one batch holds.` },
        { title: "Enqueue / dequeue", text: "Current keys join the queue; the oldest leave. Because the key encoder moves slowly, old keys still match fresh queries — the reason for the momentum." },
      ] },
      { t: "h", text: "BYOL: no negatives at all" },
      { t: "steps", items: [
        { title: "Online and target networks", text: String.raw`The **online** network (encoder + projector, weights $\theta$) is trained. The **target** network has the same shape, with weights $\xi$ that are a moving average of $\theta$: $\xi \leftarrow \tau\xi + (1-\tau)\theta$, $\tau$ starting at 0.996 in the paper and rising toward 1.` },
        { title: "Predictor", text: String.raw`Only the online branch has an extra MLP **predictor** $q_\theta$, which predicts the target's projection of the *other* view.` },
        { title: "Loss", text: String.raw`$\lVert \bar q_\theta(z_\theta) - \bar z'_\xi\rVert^2 = 2 - 2\cos(q_\theta(z_\theta), z'_\xi)$ (bars mean L2 normalisation), made symmetric by swapping views.` },
        { title: "Stop-gradient", text: "No gradient flows into the target; it moves only by the moving average. This asymmetry — predictor on one side, stop-gradient and moving average on the other — keeps training away from the constant solution in practice." },
      ] },
      { t: "h", text: "Semi-supervised learning: pseudo-labels and consistency" },
      { t: "steps", items: [
        { title: "Supervised term", text: "Normal cross-entropy on the few labelled examples." },
        { title: "Pseudo-labelling", text: String.raw`For unlabelled $x$, predict $p(y\mid x)$. If $\max_c p_c \ge$ a threshold, treat $\hat y = \arg\max_c p_c$ as the label and add its cross-entropy: the model teaches itself on examples it is sure about.` },
        { title: "Consistency (FixMatch-style)", text: "Make the pseudo-label from a **weakly** augmented view and train the prediction for a **strongly** augmented view to match it — the contrastive invariance idea, applied to class predictions." },
      ] },
    ],
    math: [
      { t: "h", text: "1. The NT-Xent loss" },
      { t: "p", text: String.raw`Let $z_1, \dots, z_{2N} \in \mathbb R^d$ be the projections of the $2N$ views, where $z_i$ and $z_{i+N}$ come from the same image; $j(i)$ is the index of $i$'s partner. With cosine similarity $s_{ik} = z_i^\top z_k/(\lVert z_i\rVert\lVert z_k\rVert) \in [-1, 1]$ and temperature $\tau > 0$:` },
      { t: "math", tex: String.raw`\ell_i = -\log\frac{\exp(s_{i\,j(i)}/\tau)}{\sum_{k=1,\,k\ne i}^{2N}\exp(s_{ik}/\tau)},\qquad \mathcal L = \frac{1}{2N}\sum_{i=1}^{2N}\ell_i` },
      { t: "p", text: String.raw`In plain words, $\ell_i$ is the cross-entropy of a $(2N-1)$-way guess with logits $s_{ik}/\tau$ and correct answer $j(i)$. If all embeddings are identical (collapse), every logit is equal and $\ell_i = \log(2N-1)$ for any $\tau$ — the from-scratch program prints 5.5413 = log 255 for $N = 128$. A useful representation must get **below** $\log(2N-1)$.` },
      { t: "h", text: "2. Gradient: a pull and weighted pushes" },
      { t: "p", text: String.raw`Let $p_{ik}$ be the softmax weight of candidate $k$ in row $i$. Since $\ell_i$ is a softmax cross-entropy, $\partial \ell_i/\partial (s_{ik}/\tau) = p_{ik} - \mathbb 1[k = j(i)]$. With unit vectors $u_k = z_k/\lVert z_k\rVert$:` },
      { t: "math", tex: String.raw`\frac{\partial \ell_i}{\partial u_i} = \frac{1}{\tau}\Big(\underbrace{-(1 - p_{i\,j(i)})\,u_{j(i)}}_{\text{pull towards the positive}} + \underbrace{\sum_{k\ne i,\,j(i)} p_{ik}\,u_k}_{\text{push from negatives, weighted by } p_{ik}}\Big)` },
      { t: "p", text: String.raw`Negatives that already look like the anchor have large $p_{ik}$ and get pushed hardest — automatic hard-negative mining, sharper for smaller $\tau$. The gradient then passes through the normalisation: $\partial/\partial z = (I - u u^\top)\,(\partial/\partial u)/\lVert z\rVert$, which removes the part along $u$ (length does not matter for cosine similarity). The from-scratch code does exactly this and matches finite differences to $4\times10^{-10}$.` },
      { t: "h", text: "3. Worked example" },
      { t: "p", text: String.raw`$N = 2$ images, two views each, 2-D embeddings: $z_1 = (1, 0.2)$, $z_2 = (0.1, 1)$ (first views of images A and B), $z_3 = (0.9, 0.5)$, $z_4 = (-0.2, 1)$ (second views). Positives: $1\leftrightarrow3$, $2\leftrightarrow4$. $\tau = 0.5$.` },
      { t: "steps", items: [
        { title: "Normalise", tex: String.raw`u_1 = (0.9806, 0.1961),\ u_2 = (0.0995, 0.9950),\ u_3 = (0.8742, 0.4856),\ u_4 = (-0.1961, 0.9806)` },
        { title: "Cosine similarities of row 1", tex: String.raw`s_{12} = 0.2927,\quad s_{13} = 0.9524,\quad s_{14} = 0` },
        { title: "Divide by τ and exponentiate", tex: String.raw`e^{0.5854} = 1.7958,\quad e^{1.9048} = 6.7184,\quad e^{0} = 1;\quad \text{sum} = 9.5141` },
        { title: "Row-1 loss", tex: String.raw`p_{13} = 6.7184/9.5141 = 0.7061,\qquad \ell_1 = -\ln 0.7061 = 0.3479` },
        { title: "Other rows (same recipe)", tex: String.raw`\ell_2 = 0.5466,\quad \ell_3 = 0.5536,\quad \ell_4 = 0.3503` },
        { title: "Batch loss", tex: String.raw`\mathcal L = \tfrac14(0.3479 + 0.5466 + 0.5536 + 0.3503) = 0.4496` },
      ] },
      { t: "p", text: String.raw`Rows 2 and 3 have larger losses because their strongest negative is close: $s_{23} = 0.5702$ (B's first view looks like A's second view). A network that cannot tell candidates apart would score $\log(2N-1) = \log 3 = 1.0986$.` },
      { t: "h", text: "4. Why it learns something: InfoNCE and mutual information" },
      { t: "p", text: String.raw`van den Oord, Li and Vinyals (2018) showed that with $K$ negatives, $I(v; v') \ge \log(K+1) - \mathcal L_{\text{InfoNCE}}$, where $I$ is the mutual information (shared information) between the two views' representations. Lowering the loss raises a lower bound on what the embeddings share — exactly what the augmentations leave unchanged. So the augmentations, not the loss, decide what the representation means. Wang and Isola (2020) split the loss into **alignment** (positives close) and **uniformity** (embeddings spread evenly over the unit sphere, enforced by the denominator).` },
      { t: "h", text: "5. MoCo's momentum update" },
      { t: "math", tex: String.raw`\theta_k \leftarrow m\,\theta_k + (1 - m)\,\theta_q,\qquad 0 \le m < 1` },
      { t: "p", text: String.raw`$\theta_q$ are the query-encoder weights (trained by backprop) and $\theta_k$ the key-encoder weights. This moving average remembers about the last $1/(1-m)$ steps: 100 for $m = 0.99$, 1000 for $m = 0.999$. Example: $\theta_k = 1.0$, $\theta_q = 2.0$, $m = 0.99$ gives $0.99 \cdot 1 + 0.01 \cdot 2 = 1.01$. The MoCo paper reported that training failed with $m = 0$ (key encoder = query encoder): queued keys then come from fast-changing networks and do not match. With $K = 65{,}536$ queued negatives, the uniform-guess loss is $\log(K+1) = 11.09$.` },
      { t: "h", text: "6. BYOL's objective" },
      { t: "math", tex: String.raw`\mathcal L_{\theta,\xi} = \Big\lVert \frac{q_\theta(z_\theta)}{\lVert q_\theta(z_\theta)\rVert} - \frac{z'_\xi}{\lVert z'_\xi\rVert}\Big\rVert_2^2 = 2 - 2\,\frac{\langle q_\theta(z_\theta), z'_\xi\rangle}{\lVert q_\theta(z_\theta)\rVert\,\lVert z'_\xi\rVert},\qquad \xi \leftarrow \tau\xi + (1-\tau)\theta` },
      { t: "p", text: String.raw`$z_\theta$ is the online projection of view 1, $z'_\xi$ the target projection of view 2, $q_\theta$ the predictor. Example: $q = (0.6, 0.8)$, $z' = (1, 0)$ gives $\cos = 0.6$ and loss $2 - 1.2 = 0.8$. Only $\theta$ is trained; $\xi$ follows by moving average. A constant output would give zero loss, so nothing in the *loss* forbids collapse — the training dynamics do. Our BYOL program shows what happens without the predictor or BatchNorm.` },
      { t: "h", text: "7. Semi-supervised objective" },
      { t: "math", tex: String.raw`\mathcal L = \frac{1}{B}\sum_{b=1}^{B}\mathrm{CE}\big(y_b,\ p_\theta(y\mid \alpha(x_b))\big) + \lambda\,\frac{1}{\mu B}\sum_{u=1}^{\mu B}\mathbb 1\big[\max_c \hat p_{uc} \ge \gamma\big]\,\mathrm{CE}\big(\hat y_u,\ p_\theta(y\mid \mathcal A(u_u))\big)` },
      { t: "p", text: String.raw`$x_b, y_b$: labelled examples; $u_u$: unlabelled ones; $\alpha$ / $\mathcal A$: weak / strong augmentation; $\hat p_u = p_\theta(y\mid\alpha(u_u))$, computed without gradient; $\hat y_u = \arg\max_c \hat p_{uc}$; $\gamma$: confidence threshold; $\lambda$: weight of the unlabelled term; $\mu$: unlabelled-to-labelled batch ratio. This is FixMatch (Sohn et al., 2020). Plain pseudo-labelling is the special case $\alpha = \mathcal A = $ identity.` },
    ],
    visual: [
      { t: "p", text: "No simulator trains SimCLR itself, but the **metric-learning simulator** shows the same geometry: an encoder trained only from *pairs* (which points belong together, which do not) arranges an embedding so related points cluster. Contrastive SSL is this, with 'same class' replaced by 'two augmentations of the same image'." },
      { t: "sim", id: "metric-learning", caption: "Contrastive and triplet losses shaping a 2-D embedding from pairwise relations only." },
      { t: "list", items: [
        "Pick the **contrastive** loss and press Play. Positives pull together and negatives inside the margin push apart — the two forces of the NT-Xent gradient (math §2).",
        "Switch negatives to **hard**: the closest negatives drive the updates. InfoNCE does a soft version of this automatically, with the temperature setting how hard.",
        "The nearest-neighbour accuracy curve rises although no classifier is trained — the same check ($k$-NN on frozen embeddings) is standard for SSL.",
        "Lower the margin toward its minimum: the push weakens and clusters stay crowded, like having too few or too weak negatives.",
      ] },
      { t: "p", text: "For contrast, the autoencoder simulator learns features by reconstruction, the older self-supervised signal. Its codes must keep every pixel detail needed to redraw the image, whereas a contrastive embedding keeps only what survives the augmentations." },
      { t: "sim", id: "autoencoder", caption: "Reconstruction-based self-supervision: a trained autoencoder on 8×8 digits." },
    ],
    scratch: [
      { t: "p", text: "NT-Xent with its full analytic gradient (through the L2 normalisation), checked on the worked example and by finite differences. We then train a **linear SimCLR** on synthetic data where each point has 3 'content' dimensions (the class signal) and 9 high-variance 'nuisance' dimensions. The augmentation re-draws the nuisance and keeps the content, so we can see exactly what the contrastive loss learns." },
      { t: "code", lang: "python", title: "ntxent_from_scratch.py", code: NTXENT_NUMPY, output: NTXENT_NUMPY_OUT },
      { t: "list", items: [
        "The loss 0.4496 and the cosine-similarity matrix reproduce the math section's worked example.",
        "The analytic gradient matches central differences to $4\\times10^{-10}$. The formula `dZ = (G + G.T) @ Z / tau` counts each $u_k$ both as an anchor and as a candidate, and the last line removes the radial direction.",
        "A collapsed batch scores exactly $\\log(2N-1)$ at every temperature — the loss itself punishes collapse.",
        "**The headline**: PCA keeps the directions of largest variance — the nuisance — so its 3-D features give 25% probe accuracy (chance). A random projection gives 39%. The contrastive encoder, which never saw a label, puts 100.0% of its weight energy on the 3 content directions and reaches 98.7%, close to the 99.5% of all 12 raw dimensions. What it learned was decided entirely by what the augmentation changed.",
        "The loss levels off near 3.85, well below $\\log 255 = 5.54$ but far from 0: two different points of the same class have nearly the same content, so they act as hard 'false negatives' the loss cannot separate.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**SimCLR on scikit-learn's 8×8 digits.** MLP backbone (64→256→64), projection head (64→64→32), and augmentations: shifts of up to 1 pixel, random 3×3 erasing, contrast change and noise. We evaluate with linear probes on frozen features, on clean test digits and on the same digits shifted by up to one pixel." },
          { t: "code", lang: "python", title: "simclr_digits.py", code: SIMCLR_TORCH, output: SIMCLR_TORCH_OUT },
          { t: "list", items: [
            "The loss falls from the uniform-guess value (log 465 = 6.14 for the last, 233-image batch) to 4.79. It stays far above 0 partly because $\\tau = 0.5$ limits how peaked the softmax can be.",
            "**On clean digits, SSL does not beat raw pixels** here: a linear classifier on the 64 raw pixels is already very strong on this tiny, centred dataset (0.970 with all labels). An honest result, and a reminder that SSL pays off when raw inputs are a poor feature space.",
            "**On shifted digits it does**: raw-pixel probes drop to 0.35–0.39 (a one-pixel shift moves every stroke to different inputs), while SimCLR features hold 0.47–0.66 (encoder $h$) and 0.75–0.84 (projector $z$). The network became shift-invariant *because the augmentation included shifts*.",
            "The projector output $z$ is the most invariant (clean and shifted scores nearly equal) but loses information (0.854 clean with all labels vs 0.937 for $h$) — SimCLR's reported trade-off: the head drops what augmentations treat as noise, so $h$ is usually the better general-purpose representation.",
            "A randomly initialised encoder is worse than raw pixels everywhere: the gains come from training, not from the architecture.",
          ] },
          { t: "p", text: "**BYOL and collapse.** The same backbone trained with BYOL (no negatives), plus two ablations. We measure collapse as the per-dimension standard deviation of the L2-normalised outputs: about $1/\\sqrt{d}$ when spread over the sphere, near 0 when every output is the same vector." },
          { t: "code", lang: "python", title: "byol_digits.py", code: BYOL_TORCH, output: BYOL_TORCH_OUT },
          { t: "list", items: [
            "Full BYOL keeps its outputs spread (std 0.149 vs 0.177 for perfectly uniform) and its loss stays near 1 — the predictor has no shortcut for matching a moving target.",
            "Without the predictor the outputs collapse (std 0.013, loss 0.014). The online network just copies the target, and the easiest way to agree with a moving-average copy of yourself on every view is to output a constant.",
            "Without BatchNorm, *in this small setup*, even full BYOL collapses (std 0.0007). BatchNorm's role in BYOL is debated: its authors later replaced it with other normalisation plus suitable initialisation, and SimSiam (Chen and He, 2021) showed a predictor plus stop-gradient suffices even without a moving average. Lesson: collapse prevention in non-contrastive methods is fragile and depends on details.",
            "The probe on $h$ stays high in all three runs (0.93–0.95): the 64-D encoder output can keep information even when the 32-D projector output has collapsed, and a random encoder already scores 0.89 on clean digits. Check collapse directly, not only through a downstream score.",
          ] },
          { t: "p", text: "**Semi-supervised learning** with 5 labels per class (50 in total) and 1,207 unlabelled training images." },
          { t: "code", lang: "python", title: "semi_supervised.py", code: SEMI_TORCH, output: SEMI_TORCH_OUT },
          { t: "list", items: [
            "Supervised training on the 50 labels alone reaches 0.819. Pseudo-labelling lifts it to 0.852 and FixMatch-style weak/strong consistency to 0.896 — with exactly the same 50 labels.",
            "Pseudo-labelling lets the most unlabelled images past the threshold (94%) but with the lowest purity (89% correct). This is **confirmation bias** in action: the model trains on its own mistakes and grows more confident in them.",
            "Consistency is pickier (77%) and more accurate (95.7%): demanding agreement across a strong augmentation filters out shaky predictions.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "SimCLR with a custom training loop and `tf.GradientTape`; self-similarities are masked by subtracting a large constant on the diagonal." },
          { t: "code", lang: "python", title: "simclr_keras.py (not executed in CI)", code: SIMCLR_KERAS, output: "Prints the epoch number and the NT-Xent loss every 50 epochs; the loss starts near log(2N − 1) and decreases." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Method", "Signal", "Prevents collapse by", "Advantages", "Disadvantages"], rows: [
        ["Pretext tasks (rotation prediction, jigsaw, colourisation)", "predict a transformation or missing property", "the task has a real target", "simple; no pairs needed", "features specialise to the pretext task; mostly replaced"],
        ["Autoencoders / denoising AEs", "reconstruct the input", "reconstruction target", "simple, generative", "keeps pixel detail that doesn't matter for meaning"],
        ["SimCLR (Chen et al. 2020)", "two views, NT-Xent over in-batch negatives", "negatives", "simple, strong", "needs large batches; false negatives"],
        ["MoCo v1/v2 (He et al. 2020)", "query vs momentum-encoded keys + queue", "negatives from a queue", "many negatives with small batches", "extra encoder and queue; stale keys"],
        ["BYOL (Grill et al. 2020)", "predict the target network's projection of the other view", "predictor + stop-gradient + EMA", "no negatives; robust to batch size", "collapse prevention is implicit and fragile"],
        ["SimSiam (Chen & He 2021)", "like BYOL without EMA", "predictor + stop-gradient", "minimal", "same fragility"],
        ["Barlow Twins / VICReg", "decorrelate embedding dimensions across views", "explicit variance/covariance terms", "no negatives, explicit anti-collapse", "high-dimensional projectors"],
        ["DINO", "self-distillation: student matches a centred, sharpened EMA teacher", "centring + sharpening", "excellent ViT features; segmentation emerges", "many hyper-parameters"],
        ["Masked modelling (BERT, MAE)", "predict masked tokens or patches", "reconstruction target", "scales extremely well; standard for text and ViTs", "linear-probe quality can trail contrastive methods without fine-tuning"],
        ["Semi-supervised (pseudo-labels, FixMatch)", "own confident predictions + consistency", "the labelled term", "uses labels and unlabelled data together", "confirmation bias; threshold tuning"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Language pretraining** — BERT's masked language modelling and GPT's next-token prediction are self-supervised; every modern LLM is pretrained this way. See [Transformer Families](/learn/m11/transformer-families).",
        "**Vision backbones** — SSL-pretrained ViTs and ResNets (DINO, MAE, SimCLR, MoCo) serve as feature extractors where labels are scarce.",
        "**Medical imaging** — pretrain on large unlabelled scan archives, then fine-tune on the few expert-labelled cases.",
        "**Speech** — wav2vec 2.0 (Baevski et al., 2020) learns speech features contrastively from unlabelled audio, enabling recognisers with little transcribed data.",
        "**Multimodal models** — CLIP uses the same contrastive loss across images and their captions; see [Multimodal Models](/learn/m15/multimodal).",
        "**Retrieval and deduplication** — contrastive embeddings power near-duplicate detection and similarity search without labelled pairs.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Uses unlabelled data**, which is plentiful; downstream tasks need far fewer labels.",
        "**You control the invariances** through augmentations — which is also a risk: an augmentation that destroys task-relevant information (colour jitter for flower species, flips for left/right tasks) makes the representation blind to it.",
        "**Compute-hungry**: many epochs, two forward passes per image, and (for SimCLR) very large batches.",
        "**False negatives**: in-batch 'negatives' may share the anchor's class, and the loss pushes them apart anyway. Bigger datasets with less class overlap reduce the effect.",
        "**Collapse** must be prevented explicitly (negatives, predictor/EMA asymmetry, variance terms) and monitored.",
        "**Evaluation is indirect**: the SSL loss says little about downstream quality, so you need linear probes, $k$-NN and fine-tuning — and on easy data raw inputs may already compete (as on our digits).",
        "**Semi-supervised** methods are cheaper than full SSL pretraining but suffer from confirmation bias and depend on the threshold and augmentations.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Not masking the diagonal** of the similarity matrix. Each embedding has similarity 1 with itself, the easiest 'candidate', so a meaningless term dominates the loss. Set the diagonal to $-\\infty$ before the softmax.",
        "**Not normalising embeddings.** Without L2 normalisation the network can lower the loss by inflating norms, and the temperature loses its meaning. Use cosine similarity.",
        "**Evaluating the projector output.** SimCLR's head throws information away; probe the backbone output $h$ (our run: 0.937 vs 0.854 with all labels).",
        "**Using identical augmentations for both views** (e.g. the same random seed). The positive pair becomes trivial and nothing is learned.",
        "**Augmentations that erase the label signal**, such as flips for digits like 2/5, or heavy colour jitter when colour is the class. Pick augmentations that keep what downstream tasks need.",
        "**Letting gradients flow into the target / key encoder** in MoCo or BYOL. They must change only through the moving average (use `torch.no_grad()` and `detach()`).",
        "**Judging collapse by the loss alone.** A collapsed BYOL has a very *low* loss (0.0142 in our ablation). Monitor the spread of normalised embeddings.",
        "**Pseudo-labelling with a low threshold from the start.** Early wrong predictions get reinforced. Warm up on labels first, and use a high threshold or a consistency term.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is self-supervised learning?", a: "Training on unlabelled data with a supervised-style task whose targets come from the data itself — e.g. predict masked words, predict a missing patch, or find which crop came from the same image. The goal is a representation that transfers to downstream tasks learned from few labels." },
      { t: "qa", level: "beginner", q: "In contrastive learning, what are positives and negatives?", a: "A positive pair is two inputs that should map close together — in SimCLR, two random augmentations of one image. Negatives are inputs that should map far apart — the other images in the batch (or in MoCo's queue)." },
      { t: "qa", level: "beginner", q: "How do you evaluate a self-supervised representation?", a: "Freeze the encoder and train a linear classifier on its features (linear probe), run k-nearest-neighbour classification on the features, or fine-tune the whole network on a small labelled set. Results are often reported at 1% and 10% of labels." },
      { t: "qa", level: "intermediate", q: "Why does SimCLR use a projection head and then throw it away?", a: "The loss forces invariance to the augmentations, and the head absorbs this by discarding information like colour or position. The backbone output before the head keeps more information and transfers better. In our digit run, the probe on h reached 0.937 vs 0.854 on the projector output z, while z was more shift-invariant." },
      { t: "qa", level: "intermediate", q: "What problem does MoCo's momentum encoder solve?", a: "MoCo wants many negatives without huge batches, so it keeps keys from past batches in a queue. If the key encoder changed as fast as the query encoder, old keys would not match new queries. Making the key encoder a slow moving average ($m = 0.999$) keeps queued keys comparable; the paper reported training failure with $m = 0$." },
      { t: "qa", level: "intermediate", q: "What is the effect of the temperature τ in NT-Xent?", a: "Similarities are divided by τ before the softmax. Small τ makes the softmax peaked, focusing the gradient on the hardest negatives and scaling gradients by 1/τ; too small is unstable and over-punishes false negatives. Large τ spreads the weight evenly and weakens the push from informative negatives. Typical values are 0.05–0.5." },
      { t: "qa", level: "intermediate", q: "How does pseudo-labelling differ from consistency regularisation?", a: "Pseudo-labelling trains on the model's own confident predictions for unlabelled data. Consistency regularisation demands predictions that stay stable when the input is perturbed. FixMatch combines them: pseudo-label from a weak augmentation, and train the strong-augmentation prediction to match it. In our run: 0.819 supervised, 0.852 pseudo-labels, 0.896 consistency." },
      { t: "qa", level: "advanced", q: "BYOL has no negatives. Why doesn't it collapse, and does it always avoid collapse?", a: "Nothing in its loss forbids a constant output; the dynamics do. The online network must predict, through an extra predictor, a slowly moving stop-gradient target, and this asymmetry makes the constant solution unstable or hard to reach in practice. It is not guaranteed: removing the predictor collapses, and in our small digit experiment removing BatchNorm also collapsed (std of normalised outputs 0.0007). SimSiam showed predictor plus stop-gradient suffices even without the moving average." },
      { t: "qa", level: "advanced", q: "Relate InfoNCE to mutual information and explain the role of augmentations.", a: "With K negatives, $I(v; v') \\ge \\log(K+1) - \\mathcal L$, so minimising InfoNCE maximises a lower bound on the mutual information between the two views' representations. What they share is what the augmentations preserve, so the augmentations define the learned invariances. Our synthetic experiment shows it: the encoder put all its weight on the dimensions the augmentations left alone." },
      { t: "qa", level: "advanced", q: "What are false negatives and how can they be mitigated?", a: "In-batch negatives from the anchor's own semantic class. The loss pushes them apart, hurting class structure (why our synthetic loss levelled off at 3.85). Fixes: larger, more diverse datasets; supervised contrastive loss when labels exist (all same-class items are positives); dropping likely false negatives by similarity; clustering-based methods; or non-contrastive methods without negatives." },
    ],
    practice: [
      { t: "quiz", questions: [
        { q: "In SimCLR with a batch of N images, how many candidates does each embedding's softmax cover?", options: ["N", "N − 1", "2N − 1", "2N"], answer: 2, explain: "2N views in total, minus the embedding itself." },
        { kind: "numeric", q: "A SimCLR batch has N = 256 images. What is the NT-Xent loss of a fully collapsed encoder (natural log, 3 decimals)?", answer: 6.236, tolerance: 0.002, explain: "$\\log(2N-1) = \\log 511 = 6.2364$, whatever τ is." },
        { q: "What does MoCo's queue provide?", options: ["Labels for the unlabelled data", "Many negatives without needing a huge batch", "Data augmentation", "A learning-rate schedule"], answer: 1, explain: "Keys from earlier batches are reused as negatives." },
        { kind: "numeric", q: "MoCo key-encoder weight 0.50, query-encoder weight 0.80, momentum m = 0.9. What is the value after one momentum update?", answer: 0.53, tolerance: 0.0001, explain: "$0.9\\cdot0.5 + 0.1\\cdot0.8 = 0.53$." },
        { kind: "numeric", q: "BYOL: predictor output (3, 4), target projection (0, 1). What is the loss 2 − 2cos?", answer: 0.4, tolerance: 0.0001, explain: "$\\cos = 4/5 = 0.8$, so $2 - 1.6 = 0.4$." },
        { q: "With BatchNorm present, which component's removal made BYOL collapse in our experiment?", options: ["The EMA update", "The predictor", "The second view", "The Adam optimiser"], answer: 1, explain: "Without the predictor, the std of normalised outputs dropped to 0.013." },
        { q: "SimCLR features beat raw pixels on shifted test digits mainly because…", options: ["the MLP is deeper", "the augmentations included shifts, so the encoder learned shift invariance", "it used more labels", "NT-Xent is a better classifier loss"], answer: 1, explain: "The representation becomes invariant to whatever the augmentations change." },
        { kind: "numeric", q: "One row of NT-Xent has scaled logits 2.0 (positive), 1.0 and 0.0 (negatives). What is its loss (3 decimals)?", answer: 0.408, tolerance: 0.002, explain: "$-\\ln\\frac{e^2}{e^2+e^1+e^0} = -\\ln\\frac{7.389}{11.107} = 0.4076$." },
      ] },
      { t: "exercise", title: "NT-Xent by hand", prompt: "Unit embeddings: $u_1 = (1, 0)$, $u_2 = (0, 1)$, $u_3 = (0.8, 0.6)$, $u_4 = (-0.6, 0.8)$, with positives $1\\leftrightarrow3$ and $2\\leftrightarrow4$, and $\\tau = 1$. Compute $\\ell_1$ and the softmax weight on row 1's hardest negative.", hint: "Row 1's similarities: $s_{12} = 0$, $s_{13} = 0.8$, $s_{14} = -0.6$.", solution: [
        { t: "steps", items: [
          { title: "Exponentials", tex: String.raw`e^{0} = 1,\quad e^{0.8} = 2.2255,\quad e^{-0.6} = 0.5488;\quad \text{sum} = 3.7744` },
          { title: "Loss", tex: String.raw`\ell_1 = -\ln\frac{2.2255}{3.7744} = -\ln 0.5896 = 0.5282` },
          { title: "Hardest negative", tex: String.raw`p_{12} = 1/3.7744 = 0.2649\ \ (u_2,\ \text{similarity } 0 > -0.6)` },
        ] },
        { t: "p", text: "The gradient pushes $u_1$ away from $u_2$ with weight 0.2649 and from $u_4$ with weight $0.5488/3.7744 = 0.1454$, and pulls it toward $u_3$ with weight $1 - 0.5896 = 0.4104$." },
      ] },
      { t: "exercise", title: "Coding: add a MoCo queue", prompt: "Turn `simclr_digits.py` into MoCo: a key encoder that is a moving-average copy (m = 0.99) of encoder+projector, a queue of 1024 normalised keys, and an InfoNCE loss where each query's positive is its own key and the negatives are the queue. Train with batch size 64 and compare the linear probe with SimCLR at batch size 64.", hint: "Logits: `torch.cat([(q * k).sum(1, keepdim=True), q @ queue.T], 1) / tau`; the target class is always 0. Update the queue with `queue = torch.cat([k, queue])[:1024]` after each step.", solution: [
        {
          t: "code",
          lang: "text",
          code: "net = nn.Sequential(encoder, projector)\nkey_net = copy.deepcopy(net)\nfor p in key_net.parameters():\n    p.requires_grad_(False)\nqueue = F.normalize(torch.randn(1024, 32), dim=1)\nfor epoch in range(150):\n    for idx in torch.randperm(len(T_tr)).split(64):\n        x = T_tr[idx]\n        q = F.normalize(net(augment(x)), dim=1)\n        with torch.no_grad():\n            k = F.normalize(key_net(augment(x)), dim=1)\n        logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.T], 1) / 0.2\n        loss = F.cross_entropy(logits, torch.zeros(len(x), dtype=torch.long))\n        opt.zero_grad(); loss.backward(); opt.step()\n        with torch.no_grad():\n            for pk, pq in zip(key_net.parameters(), net.parameters()):\n                pk.mul_(0.99).add_(0.01 * pq)\n        queue = torch.cat([k, queue])[:1024]",
        },
        { t: "p", text: "Each query now sees 1024 negatives instead of 2·64 − 2 = 126. Expect MoCo at batch 64 to do at least as well as SimCLR at batch 64, but report your run's actual numbers rather than assuming. Setting the momentum to 0 should make training worse, as the MoCo paper reported." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Self-supervised learning makes targets from the data itself (views, masks, next tokens) to learn representations; downstream tasks then need few labels.",
        "NT-Xent: $\\ell_i = -\\log\\frac{e^{s_{ij}/\\tau}}{\\sum_{k\\ne i}e^{s_{ik}/\\tau}}$ over $2N-1$ candidates; collapse gives $\\log(2N-1)$; worked example 0.4496.",
        "Gradient: pull toward the positive, push negatives weighted by their softmax probability (soft hard-negative mining); τ sets how hard.",
        "SimCLR: two augmentations, encoder + projection head, in-batch negatives; keep the features before the head.",
        "MoCo: momentum key encoder $\\theta_k \\leftarrow m\\theta_k + (1-m)\\theta_q$ and a queue of negatives.",
        "BYOL: online network + predictor predicts a moving-average target, loss $2 - 2\\cos$; no negatives, collapse avoided by asymmetry (fragile: removing the predictor collapsed it).",
        "Augmentations define the invariances (synthetic demo: 98.7% vs 25% for PCA; digits: robust to shifts).",
        "Semi-supervised: pseudo-labels and weak/strong consistency (50 labels: 0.819 → 0.852 → 0.896).",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Distillation, Curriculum, Meta- and Few-Shot Learning](/learn/m15/learning-paradigms) — more ways to learn with few labels: from a teacher, from episodes, and from embeddings of unseen classes.",
        "[Multimodal Models, CLIP and RAG](/learn/m15/multimodal) — contrastive learning across modalities: images and text in one embedding space, and retrieval built on it.",
        "[Transformer Families](/learn/m11/transformer-families) — masked and causal language modelling, the self-supervised objectives behind BERT and GPT.",
      ] },
    ],
  },
};

export default lesson;
