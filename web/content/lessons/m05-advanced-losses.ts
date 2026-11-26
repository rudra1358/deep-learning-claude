import type { Lesson } from "@/lib/types";

const SCRATCH_METRIC = String.raw`import numpy as np
from sklearn.datasets import load_digits

# ---------------- losses on embeddings, each returning (loss, gradients w.r.t. the embeddings) ----------------
def contrastive(e1, e2, dissimilar, m=1.0):
    """Hadsell, Chopra & LeCun (2006): (1-Y) 1/2 D^2 + Y 1/2 max(0, m - D)^2, with Y = 1 for a dissimilar pair."""
    diff = e1 - e2
    D = np.linalg.norm(diff)
    if not dissimilar:
        return 0.5 * D ** 2, diff, -diff                      # pull: gradient is the difference vector
    h = max(0.0, m - D)
    g = -h * diff / max(D, 1e-12)                             # push, with force (m - D), only inside the margin
    return 0.5 * h ** 2, g, -g

def triplet(a, p, n, m=0.5, squared=False):
    """max(0, d(a,p) - d(a,n) + m); squared=True is the FaceNet form with squared distances."""
    if squared:
        L = np.sum((a - p) ** 2) - np.sum((a - n) ** 2) + m
        if L <= 0:
            return 0.0, 0 * a, 0 * p, 0 * n
        return L, 2 * (n - p), -2 * (a - p), 2 * (a - n)
    dap, dan = np.linalg.norm(a - p), np.linalg.norm(a - n)
    L = dap - dan + m
    if L <= 0:
        return 0.0, 0 * a, 0 * p, 0 * n
    u, v = (a - p) / dap, (a - n) / dan                       # unit vectors
    return L, u - v, -u, v

# ---------------- finite-difference check of every gradient ----------------
rng = np.random.default_rng(0)
def check(f, args, h=1e-6):
    worst = 0.0
    grads = f(*args)[1:]
    for k, x in enumerate(args):
        for i in range(x.size):
            xp = [a.copy() for a in args]; xm = [a.copy() for a in args]
            xp[k].flat[i] += h; xm[k].flat[i] -= h
            num = (f(*xp)[0] - f(*xm)[0]) / (2 * h)
            worst = max(worst, abs(num - grads[k].flat[i]))
    return worst

a, p, n = rng.normal(size=(3, 4))
print(f"contrastive similar    grad error {check(lambda x, y: contrastive(x, y, False), [a, p]):.1e}")
print(f"contrastive dissimilar grad error {check(lambda x, y: contrastive(x, y, True, m=5.0), [a, n]):.1e}")
print(f"triplet (Euclidean)    grad error {check(lambda x, y, z: triplet(x, y, z, m=2.0), [a, p, n]):.1e}")
print(f"triplet (squared)      grad error {check(lambda x, y, z: triplet(x, y, z, m=5.0, squared=True), [a, p, n]):.1e}")

# ---------------- training a linear embedding of handwritten digits with the triplet loss ----------------
X, y = load_digits(return_X_y=True)                       # 1797 images of 8x8 pixels -> 64-D vectors
X = X / 16.0                                              # pixel intensities in [0, 1]
idx = rng.permutation(len(y))
tr, te = idx[:1200], idx[1200:]
X = X - X[tr].mean(0)                                     # centre with training statistics

def knn_accuracy(W):
    Etr, Ete = X[tr] @ W.T, X[te] @ W.T
    d = ((Ete[:, None, :] - Etr[None, :, :]) ** 2).sum(-1)
    return np.mean(y[tr][d.argmin(1)] == y[te])

def train(mining, steps=600, batch=64, m=0.5, lr=0.02, dim=2, seed=1):
    r = np.random.default_rng(seed)
    W = r.normal(0, 0.1, size=(dim, 64))                    # embedding e = W x
    Xt, yt = X[tr], y[tr]
    active_hist = []
    for step in range(steps):
        E = Xt @ W.T
        ai = r.choice(len(tr), size=batch, replace=False)   # anchors of this mini-batch
        same = (yt[ai, None] == yt[None, :]) & (ai[:, None] != np.arange(len(tr))[None, :])
        other = yt[ai, None] != yt[None, :]
        pi = np.where(same, r.random(same.shape), -1).argmax(1)        # a random positive per anchor
        D = np.linalg.norm(E[ai, None, :] - E[None, :, :], axis=2)      # (batch, N) anchor-to-all distances
        dap = D[np.arange(batch), pi]
        rand_neg = np.where(other, r.random(other.shape), -1).argmax(1)
        if mining == "random":
            ni = rand_neg
        elif mining == "hard":                                          # closest point of another class
            ni = np.where(other, D, np.inf).argmin(1)
        else:                                                           # semi-hard: dap < dan < dap + m
            ok = other & (D > dap[:, None]) & (D < dap[:, None] + m)
            ni = np.where(ok.any(1), np.where(ok, D, np.inf).argmin(1), rand_neg)
        a, p, n = E[ai], E[pi], E[ni]
        dan = D[np.arange(batch), ni]
        on = (dap - dan + m > 0)[:, None]                               # only violating triplets have a gradient
        u = (a - p) / np.maximum(dap, 1e-12)[:, None]
        v = (a - n) / np.maximum(dan, 1e-12)[:, None]
        ga, gp, gn = on * (u - v), on * -u, on * v                      # gradients w.r.t. a, p, n (see triplet())
        # chain rule through e = W x: dL/dW = sum over the three embeddings of g x^T
        G = ga.T @ Xt[ai] + gp.T @ Xt[pi] + gn.T @ Xt[ni]
        W -= lr * G / batch
        active_hist.append(on.mean())
    return W, active_hist

W0 = np.random.default_rng(1).normal(0, 0.1, size=(2, 64))
print(f"\nrandom W (step 0)  : 1-NN acc {knn_accuracy(W0):.3f}  mean |e| {np.linalg.norm(X[te] @ W0.T, axis=1).mean():.3f}")
for mining in ["random", "semi-hard", "hard"]:
    W, act = train(mining)
    spread = np.linalg.norm(X[te] @ W.T, axis=1).mean()                # average embedding size
    print(f"{mining:9s} negatives: 1-NN acc {knn_accuracy(W):.3f}  mean |e| {spread:.3f}  "
          f"active triplets: first 50 steps {np.mean(act[:50]):.2f}, last 50 {np.mean(act[-50:]):.2f}")`;

const SCRATCH_METRIC_OUT = String.raw`contrastive similar    grad error 1.2e-10
contrastive dissimilar grad error 7.2e-10
triplet (Euclidean)    grad error 1.4e-10
triplet (squared)      grad error 6.2e-10

random W (step 0)  : 1-NN acc 0.164  mean |e| 0.180
random    negatives: 1-NN acc 0.635  mean |e| 0.830  active triplets: first 50 steps 0.89, last 50 0.35
semi-hard negatives: 1-NN acc 0.645  mean |e| 0.580  active triplets: first 50 steps 1.00, last 50 0.99
hard      negatives: 1-NN acc 0.362  mean |e| 0.006  active triplets: first 50 steps 1.00, last 50 1.00`;

const SCRATCH_OVERLAP = String.raw`import numpy as np

rng = np.random.default_rng(0)

def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

# ================= 1. NT-Xent (InfoNCE over a batch of positive pairs) =================
def nt_xent(Z, tau):
    """Z: (2N, d) raw embeddings; rows i and i+N are two views of the same example.
    Returns the mean loss and dL/dZ (through the L2 normalisation)."""
    n2 = len(Z)
    N = n2 // 2
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    U = Z / norms                                        # points on the unit sphere
    S = U @ U.T / tau                                    # cosine similarities / temperature
    np.fill_diagonal(S, -np.inf)                         # an embedding is never its own negative
    pos = np.r_[np.arange(N, n2), np.arange(N)]          # index of each row's positive
    S_max = S.max(1, keepdims=True)
    logP = S - S_max - np.log(np.exp(S - S_max).sum(1, keepdims=True))   # row-wise log-softmax
    loss = -logP[np.arange(n2), pos].mean()
    P = np.exp(logP)                                     # softmax weights over the 2N-1 candidates
    G = P.copy()
    G[np.arange(n2), pos] -= 1                           # dL/dS = (P - onehot(pos)) / 2N, like cross-entropy
    G /= n2
    dU = (G + G.T) @ U / tau                             # S is symmetric in U
    dZ = (dU - U * (dU * U).sum(1, keepdims=True)) / norms   # project out the radial direction
    return loss, dZ

Z = rng.normal(size=(8, 5))
loss, dZ = nt_xent(Z, 0.5)
print(f"NT-Xent loss {loss:.4f}, gradient error vs finite differences {np.abs(dZ - num_grad(lambda z: nt_xent(z, 0.5)[0], Z)).max():.1e}")

# temperature: how the gradient is shared among negatives for one anchor
s_pos, s_neg = 0.8, np.array([0.5, 0.1, -0.3])
for tau in (1.0, 0.5, 0.1):
    logits = np.r_[s_pos, s_neg] / tau
    p = np.exp(logits - logits.max()); p /= p.sum()
    print(f"tau={tau:<4}: loss {-np.log(p[0]):.4f}  softmax weights of negatives {np.round(p[1:], 4)}"
          f"  share on the hardest {p[1] / p[1:].sum():.3f}")

# ================= 2. KL divergence: distillation and the VAE prior term =================
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def kl(p, q):
    return np.sum(p * (np.log(p) - np.log(q)))

t_logits, s_logits, T = np.array([3.0, 1.0, 0.2]), np.array([2.0, 1.5, 0.0]), 2.0
pt, ps = softmax(t_logits / T), softmax(s_logits / T)
distill = T ** 2 * kl(pt, ps)
grad = num_grad(lambda z: T ** 2 * kl(pt, softmax(z / T)), s_logits)
print(f"\nteacher p^T {np.round(pt, 4)}  student p^T {np.round(ps, 4)}")
print(f"T^2 KL(teacher||student) = {distill:.4f}; gradient {np.round(grad, 4)} = T (p_s - p_t) {np.round(T * (ps - pt), 4)}")

mu, logvar = np.array([0.5, -1.0]), np.array([0.0, -1.0])
kl_closed = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1 - logvar)
z = mu + np.exp(0.5 * logvar) * rng.normal(size=(200000, 2))     # samples from q(z|x)
log_q = -0.5 * np.sum(np.log(2 * np.pi) + logvar + (z - mu) ** 2 / np.exp(logvar), 1)
log_p = -0.5 * np.sum(np.log(2 * np.pi) + z ** 2, 1)
print(f"VAE KL(q||N(0,I)): closed form {kl_closed:.4f}, Monte Carlo {np.mean(log_q - log_p):.4f}")

# ================= 3. Segmentation: Dice versus pixel-wise cross-entropy =================
H = W = 64
yy, xx = np.mgrid[:H, :W]
target = ((yy - 20) ** 2 + (xx - 40) ** 2 <= 5 ** 2).astype(float)   # a small lesion
fg = target.mean()

def bce(p, g, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(g * np.log(p) + (1 - g) * np.log(1 - p))

def soft_dice_loss(p, g, eps=1.0):
    inter = np.sum(p * g)
    return 1 - (2 * inter + eps) / (np.sum(p) + np.sum(g) + eps)

def soft_dice_grad(p, g, eps=1.0):
    inter, denom = np.sum(p * g), np.sum(p) + np.sum(g) + eps
    return -(2 * g * denom - (2 * inter + eps)) / denom ** 2

def soft_iou_loss(p, g, eps=1.0):
    inter = np.sum(p * g)
    return 1 - (inter + eps) / (np.sum(p) + np.sum(g) - inter + eps)

p_test = rng.uniform(0.05, 0.95, size=(6, 6)); g_test = (rng.uniform(size=(6, 6)) < 0.3).astype(float)
print(f"\nforeground fraction {fg:.4f}; soft-Dice gradient error "
      f"{np.abs(soft_dice_grad(p_test, g_test) - num_grad(lambda q: soft_dice_loss(q, g_test), p_test)).max():.1e}")

shifted = np.roll(target, 6, axis=1)                                 # right shape, 6 pixels off
preds = {
    "all background (p=0.01)": np.full((H, W), 0.01),
    "prior everywhere (p=fg)": np.full((H, W), fg),
    "lesion, p=0.9 / 0.1":     np.where(target == 1, 0.9, 0.1),
    "shifted lesion, p=0.9":   np.where(shifted == 1, 0.9, 0.01),
    "perfect, p=0.99 / 0.01":  np.where(target == 1, 0.99, 0.01),
}
print(f"{'prediction':25s} {'BCE':>7s} {'Dice loss':>10s} {'IoU loss':>9s} {'hard Dice':>10s}")
for name, p in preds.items():
    hard = (p > 0.5).astype(float)
    hard_dice = 2 * np.sum(hard * target) / max(np.sum(hard) + np.sum(target), 1e-9)
    print(f"{name:25s} {bce(p, target):7.4f} {soft_dice_loss(p, target):10.4f} {soft_iou_loss(p, target):9.4f} {hard_dice:10.4f}")

p = np.full((H, W), 0.1)
gd = soft_dice_grad(p, target)
gb = (p - target) / (p * (1 - p)) / target.size                     # dBCE/dp per pixel
print(f"at p=0.1 everywhere: |dDice/dp| foreground {abs(gd[target == 1][0]):.2e}, background {abs(gd[target == 0][0]):.2e}"
      f" (ratio {abs(gd[target == 1][0] / gd[target == 0][0]):.1f});"
      f" |dBCE/dp| ratio {abs(gb[target == 1][0] / gb[target == 0][0]):.1f}")

# ================= 4. Boxes: IoU loss versus GIoU loss =================
def iou_giou(b, g):
    """boxes as (x1, y1, x2, y2)."""
    iw = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    ih = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    inter = iw * ih
    union = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    hull = (max(b[2], g[2]) - min(b[0], g[0])) * (max(b[3], g[3]) - min(b[1], g[1]))
    iou = inter / union
    return iou, iou - (hull - union) / hull

gt = np.array([0.0, 0.0, 2.0, 2.0])
print()
for b in ([1, 1, 3, 3], [2, 0, 4, 2], [3, 0, 5, 2], [6, 0, 8, 2]):
    iou, giou = iou_giou(np.array(b, float), gt)
    print(f"pred {b}: IoU {iou:.4f}  GIoU {giou:+.4f}  -> IoU loss {1 - iou:.4f}, GIoU loss {1 - giou:.4f}")

for name, k in [("IoU loss ", 0), ("GIoU loss", 1)]:
    b = np.array([4.0, 3.0, 5.0, 4.0])                               # starts with no overlap
    for step in range(300):
        g = num_grad(lambda q: 1 - iou_giou(q, gt)[k], b)           # 4 numbers: finite differences suffice
        b -= 0.5 * g
    print(f"{name}: final box {np.round(b, 3)}  IoU with target {iou_giou(b, gt)[0]:.3f}")`;

const SCRATCH_OVERLAP_OUT = String.raw`NT-Xent loss 2.3384, gradient error vs finite differences 3.4e-10
tau=1.0 : loss 0.9440  softmax weights of negatives [0.2882 0.1932 0.1295]  share on the hardest 0.472
tau=0.5 : loss 0.6451  softmax weights of negatives [0.2879 0.1294 0.0581]  share on the hardest 0.606
tau=0.1 : loss 0.0495  softmax weights of negatives [0.0474 0.0009 0.    ]  share on the hardest 0.982

teacher p^T [0.6194 0.2279 0.1527]  student p^T [0.4658 0.3628 0.1714]
T^2 KL(teacher||student) = 0.2117; gradient [-0.3071  0.2699  0.0373] = T (p_s - p_t) [-0.3071  0.2699  0.0373]
VAE KL(q||N(0,I)): closed form 0.8089, Monte Carlo 0.8095

foreground fraction 0.0198; soft-Dice gradient error 4.1e-11
prediction                    BCE  Dice loss  IoU loss  hard Dice
all background (p=0.01)    0.1009     0.9787    0.9852     0.0000
prior everywhere (p=fg)    0.0972     0.9742    0.9839     0.0000
lesion, p=0.9 / 0.1        0.1054     0.7362    0.8472     1.0000
shifted lesion, p=0.9      0.1048     0.7584    0.8601     0.3086
perfect, p=0.99 / 0.01     0.0101     0.2024    0.3353     1.0000
at p=0.1 everywhere: |dDice/dp| foreground 4.00e-03, background 7.12e-05 (ratio 56.2); |dBCE/dp| ratio 9.0

pred [1, 1, 3, 3]: IoU 0.1429  GIoU -0.0794  -> IoU loss 0.8571, GIoU loss 1.0794
pred [2, 0, 4, 2]: IoU 0.0000  GIoU +0.0000  -> IoU loss 1.0000, GIoU loss 1.0000
pred [3, 0, 5, 2]: IoU 0.0000  GIoU -0.2000  -> IoU loss 1.0000, GIoU loss 1.2000
pred [6, 0, 8, 2]: IoU 0.0000  GIoU -0.5000  -> IoU loss 1.0000, GIoU loss 1.5000
IoU loss : final box [4. 3. 5. 4.]  IoU with target 0.000
GIoU loss: final box [-0.089 -0.02   2.088  2.069]  IoU with target 0.880`;

const TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.manual_seed(0)
DIM = 4                                               # a small embedding, so the loss has to choose what to keep

# ---------------- data: 8x8 digits; metric learning is trained on digits 0-6 only ----------------
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
perm = torch.randperm(len(y))
tr, te = perm[:1200], perm[1200:]
tr_seen = tr[y[tr] < 7]
SEEN, UNSEEN = torch.arange(7), torch.arange(7, 10)

def recall_at_1(f, classes):
    """Retrieval: for each test image of these classes, is its nearest gallery (train) image the same digit?"""
    with torch.no_grad():
        g, q = tr[torch.isin(y[tr], classes)], te[torch.isin(y[te], classes)]
        Eg, Eq = F.normalize(f(X[g]), dim=1), F.normalize(f(X[q]), dim=1)
        return (y[g][(Eq @ Eg.T).argmax(1)] == y[q]).float().mean().item()

def report(name, f):
    print(f"{name:30s} recall@1 seen digits {recall_at_1(f, SEEN):.3f}   unseen digits 7-9 {recall_at_1(f, UNSEEN):.3f}")

def embedder():
    torch.manual_seed(0)                              # same initialisation for every loss
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, DIM))

def pk_batch(per_class=12):
    """P x K batch: 7 classes x 12 images, so every anchor has positives and negatives."""
    return torch.cat([tr_seen[y[tr_seen] == c][torch.randperm(int((y[tr_seen] == c).sum()))[:per_class]] for c in range(7)])

def fit(loss_fn, make_batch, steps=400):
    net = embedder()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = loss_fn(net, make_batch())
        opt.zero_grad(); loss.backward(); opt.step()
    return net

# baseline: PCA to the same dimension
mu = X[tr_seen].mean(0)
V = torch.linalg.svd(X[tr_seen] - mu, full_matrices=False).Vh[:DIM].T
report("PCA (4-D)", lambda x: (x - mu) @ V)

# (a) contrastive loss (Hadsell et al. 2006) over all pairs of a P x K batch
def contrastive_loss(E, labels, m=0.5):
    D = torch.cdist(E, E)
    same = (labels[:, None] == labels[None, :]).float()
    off_diag = 1 - torch.eye(len(labels))
    L = (same * off_diag) * D ** 2 + (1 - same) * F.relu(m - D) ** 2
    return 0.5 * L.sum() / off_diag.sum()

net = fit(lambda f, i: contrastive_loss(F.normalize(f(X[i]), dim=1), y[i]), pk_batch)
report("contrastive (m=0.5)", net)

# (b) triplet loss with batch-hard mining: hardest positive and hardest negative per anchor
def batch_hard_triplet(E, labels, m=0.2):
    D = torch.cdist(E, E)
    same = labels[:, None] == labels[None, :]
    hardest_pos = (D * same).max(1).values
    hardest_neg = D.masked_fill(same, float("inf")).min(1).values
    return F.relu(hardest_pos - hardest_neg + m).mean()

net = fit(lambda f, i: batch_hard_triplet(F.normalize(f(X[i]), dim=1), y[i]), pk_batch)
report("triplet, batch-hard", net)

a, p, n = torch.randn(3, 5, 16)                       # the built-in loss takes explicit triplets
print("nn.TripletMarginLoss == mean max(0, d(a,p) - d(a,n) + m):",
      torch.allclose(nn.TripletMarginLoss(margin=1.0)(a, p, n),
                     F.relu(F.pairwise_distance(a, p) - F.pairwise_distance(a, n) + 1.0).mean()))

# (c) NT-Xent / InfoNCE (SimCLR-style): two augmented views per image, NO labels used
def augment(x):
    imgs = x.view(-1, 1, 8, 8)
    dx, dy = torch.randint(-1, 2, (2,)).tolist()      # shift by up to one pixel ...
    imgs = torch.roll(imgs, shifts=(dy, dx), dims=(2, 3))
    return (imgs + 0.15 * torch.randn_like(imgs)).view(-1, 64)   # ... and add pixel noise

def nt_xent(z1, z2, tau):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    S = z @ z.T / tau                                 # cosine similarities / temperature
    S.fill_diagonal_(float("-inf"))                   # never contrast an embedding with itself
    N = len(z1)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(N)])   # row i's positive is its other view
    return F.cross_entropy(S, targets)               # InfoNCE = cross-entropy over 2N - 1 candidates

for tau in (0.1, 0.2, 0.5):
    def loss_fn(f, i, tau=tau):
        xb = X[i]
        return nt_xent(f(augment(xb)), f(augment(xb)), tau)
    net = fit(loss_fn, lambda: tr_seen[torch.randperm(len(tr_seen))[:256]])
    report(f"NT-Xent tau={tau} (no labels)", net)

# ---------------- KL divergence for distillation ----------------
t_logits, T = torch.tensor([[3.0, 1.0, 0.2]]), 2.0
s_logits = torch.tensor([[2.0, 1.5, 0.0]], requires_grad=True)
# kl_div expects the student's LOG-probabilities and the teacher's probabilities
kd = F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1), reduction="batchmean") * T * T
kd.backward()
print(f"\ndistillation T^2 KL = {kd.item():.4f}, gradient w.r.t. student logits {s_logits.grad.numpy().round(4)}")

# ---------------- segmentation: soft Dice + cross-entropy on logits ----------------
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2))
    return (1 - (2 * inter + eps) / (p.sum((1, 2)) + target.sum((1, 2)) + eps)).mean()

def dice_ce_loss(logits, target):
    return dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits, target)

target = torch.zeros(1, 64, 64)
target[0, 15:25, 35:45] = 1.0                         # 100 foreground pixels out of 4096
logits = torch.full((1, 64, 64), -3.0, requires_grad=True)
loss = dice_ce_loss(logits, target)
loss.backward()
g = logits.grad[0]
print(f"Dice+CE {loss.item():.4f}; |gradient| on a foreground pixel {g[20, 40].abs():.2e}, background pixel {g[0, 0].abs():.2e}")

# ---------------- boxes: IoU loss vs GIoU loss for boxes that do not overlap ----------------
def iou_and_giou(b, g):                               # boxes (x1, y1, x2, y2)
    lt, rb = torch.max(b[:2], g[:2]), torch.min(b[2:], g[2:])
    inter = (rb - lt).clamp(min=0).prod()
    union = (b[2:] - b[:2]).prod() + (g[2:] - g[:2]).prod() - inter
    hull = (torch.max(b[2:], g[2:]) - torch.min(b[:2], g[:2])).prod()   # smallest enclosing box C
    iou = inter / union
    return iou, iou - (hull - union) / hull

gt = torch.tensor([0.0, 0.0, 2.0, 2.0])
for k, name in [(0, "IoU loss "), (1, "GIoU loss")]:
    box = torch.tensor([3.0, 0.0, 5.0, 2.0], requires_grad=True)   # 1 unit to the right of gt
    (1 - iou_and_giou(box, gt)[k]).backward()
    print(f"{name} gradient w.r.t. (x1, y1, x2, y2): {box.grad.numpy().round(4)}")`;

const TORCH_OUT = String.raw`PCA (4-D)                      recall@1 seen digits 0.927   unseen digits 7-9 0.788
contrastive (m=0.5)            recall@1 seen digits 0.988   unseen digits 7-9 0.620
triplet, batch-hard            recall@1 seen digits 0.976   unseen digits 7-9 0.603
nn.TripletMarginLoss == mean max(0, d(a,p) - d(a,n) + m): True
NT-Xent tau=0.1 (no labels)    recall@1 seen digits 0.896   unseen digits 7-9 0.717
NT-Xent tau=0.2 (no labels)    recall@1 seen digits 0.908   unseen digits 7-9 0.783
NT-Xent tau=0.5 (no labels)    recall@1 seen digits 0.884   unseen digits 7-9 0.696

distillation T^2 KL = 0.2117, gradient w.r.t. student logits [[-0.3071  0.2699  0.0373]]
Dice+CE 1.0863; |gradient| on a foreground pixel 5.33e-04, background pixel 1.70e-05
IoU loss  gradient w.r.t. (x1, y1, x2, y2): [0. 0. 0. 0.]
GIoU loss gradient w.r.t. (x1, y1, x2, y2): [ 0.2   0.   -0.04 -0.  ]`;

const KERAS = String.raw`import tensorflow as tf

# ---- triplet loss on explicit triplets (embeddings L2-normalised, squared distances as in FaceNet) ----
def triplet_loss(a, p, n, margin=0.2):
    a, p, n = (tf.math.l2_normalize(t, axis=-1) for t in (a, p, n))
    d_ap = tf.reduce_sum(tf.square(a - p), axis=-1)
    d_an = tf.reduce_sum(tf.square(a - n), axis=-1)
    return tf.reduce_mean(tf.maximum(d_ap - d_an + margin, 0.0))

# ---- NT-Xent: cross-entropy over cosine similarities ----
def nt_xent(z1, z2, tau=0.2):
    z = tf.math.l2_normalize(tf.concat([z1, z2], 0), axis=1)
    n = tf.shape(z1)[0]
    sim = tf.matmul(z, z, transpose_b=True) / tau
    sim = sim - 1e9 * tf.eye(2 * n)                          # mask self-similarity
    targets = tf.concat([tf.range(n, 2 * n), tf.range(n)], 0)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, sim, from_logits=True))

# ---- distillation: KL(teacher || student) at temperature T, scaled by T^2 ----
T = 2.0
kl = tf.keras.losses.KLDivergence()                           # expects probabilities: (y_true, y_pred)
teacher, student = tf.constant([[3.0, 1.0, 0.2]]), tf.constant([[2.0, 1.5, 0.0]])
print("T^2 KL:", float(T * T * kl(tf.nn.softmax(teacher / T), tf.nn.softmax(student / T))))

# ---- segmentation: soft Dice + binary cross-entropy (model outputs logits) ----
def dice_ce(y_true, logits, eps=1.0):
    p = tf.sigmoid(logits)
    inter = tf.reduce_sum(p * y_true, axis=[1, 2])
    dice = (2 * inter + eps) / (tf.reduce_sum(p, axis=[1, 2]) + tf.reduce_sum(y_true, axis=[1, 2]) + eps)
    bce = tf.keras.losses.binary_crossentropy(y_true, logits, from_logits=True)
    return (1 - dice) + tf.reduce_mean(bce, axis=[1, 2])

# ---- GIoU loss for boxes (x1, y1, x2, y2) ----
def giou_loss(b, g):
    lt, rb = tf.maximum(b[..., :2], g[..., :2]), tf.minimum(b[..., 2:], g[..., 2:])
    inter = tf.reduce_prod(tf.maximum(rb - lt, 0.0), axis=-1)
    area = lambda x: tf.reduce_prod(x[..., 2:] - x[..., :2], axis=-1)
    union = area(b) + area(g) - inter
    hull = tf.reduce_prod(tf.maximum(b[..., 2:], g[..., 2:]) - tf.minimum(b[..., :2], g[..., :2]), axis=-1)
    return 1 - (inter / union - (hull - union) / hull)

print("GIoU loss:", giou_loss(tf.constant([[3.0, 0.0, 5.0, 2.0]]), tf.constant([[0.0, 0.0, 2.0, 2.0]])).numpy())`;

const EXERCISE = String.raw`import numpy as np

def classify_negatives(E, labels, margin):
    """For every (anchor, positive, negative) triplet in a batch, count easy / semi-hard / hard negatives.
    Euclidean distances; hard: d_an < d_ap, semi-hard: d_ap <= d_an < d_ap + m, easy: d_an >= d_ap + m."""
    D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
    same = labels[:, None] == labels[None, :]
    pos = same & ~np.eye(len(labels), dtype=bool)           # (anchor, positive) pairs
    neg = ~same                                            # (anchor, negative) pairs
    d_ap = D[:, :, None]                                   # axis 1 = positive index
    d_an = D[:, None, :]                                   # axis 2 = negative index
    valid = pos[:, :, None] & neg[:, None, :]              # all valid triplets (a, p, n)
    hard = valid & (d_an < d_ap)
    semi = valid & (d_an >= d_ap) & (d_an < d_ap + margin)
    easy = valid & (d_an >= d_ap + margin)
    loss = np.maximum(d_ap - d_an + margin, 0)[valid]      # "batch-all" triplet losses
    return hard.sum(), semi.sum(), easy.sum(), loss

def loop_version(E, labels, margin):                       # slow reference implementation
    counts = [0, 0, 0]
    for a in range(len(E)):
        for p in range(len(E)):
            if p == a or labels[p] != labels[a]:
                continue
            for n in range(len(E)):
                if labels[n] == labels[a]:
                    continue
                dap, dan = np.linalg.norm(E[a] - E[p]), np.linalg.norm(E[a] - E[n])
                counts[0 if dan < dap else 1 if dan < dap + margin else 2] += 1
    return counts

rng = np.random.default_rng(0)
labels = np.repeat(np.arange(4), 6)                        # a P x K batch: 4 classes x 6 samples
centres = rng.normal(0, 2.0, size=(4, 8))
for spread in (3.0, 1.5, 0.8):                             # from "untrained" to "well trained" embeddings
    E = centres[labels] + rng.normal(0, spread, size=(24, 8))
    h, s, e, loss = classify_negatives(E, labels, margin=1.0)
    assert [h, s, e] == loop_version(E, labels, 1.0)
    print(f"spread {spread}: {h + s + e} triplets  hard {h:4d}  semi-hard {s:4d}  easy {e:4d}  "
          f"mean loss over all {loss.mean():.3f}, over non-zero only {loss[loss > 0].mean() if (loss > 0).any() else 0:.3f}")`;

const EXERCISE_OUT = String.raw`spread 3.0: 2160 triplets  hard  472  semi-hard  191  easy 1497  mean loss over all 0.687, over non-zero only 2.237
spread 1.5: 2160 triplets  hard  279  semi-hard  227  easy 1654  mean loss over all 0.359, over non-zero only 1.534
spread 0.8: 2160 triplets  hard    8  semi-hard   52  easy 2100  mean loss over all 0.014, over non-zero only 0.509`;

const lesson: Lesson = {
  id: "m05/advanced-losses",
  title: "Advanced Losses: Contrastive, Triplet, InfoNCE, KL, Dice, IoU and GIoU",
  summary:
    "Losses that do not compare one prediction with one label: contrastive and triplet losses for metric learning (with margins and negative mining), InfoNCE / NT-Xent and its temperature, KL divergence for distillation and VAEs, Dice and IoU losses for imbalanced segmentation, IoU and GIoU losses for bounding boxes, and reconstruction losses for autoencoders — each derived, differentiated and tested in code.",
  minutes: 90,
  level: "advanced",
  prerequisites: ["m05/classification-losses", "m05/regression-losses", "m02/information-theory", "m02/norms-spaces", "m04/backpropagation"],
  next: ["m15/self-supervised", "m12/segmentation", "m12/object-detection", "m14/autoencoders", "m14/vae"],
  terms: ["metric-learning", "embedding", "contrastive-loss", "triplet-loss", "hard-negative-mining", "infonce", "temperature", "kl-divergence", "knowledge-distillation", "dice-loss", "jaccard-index", "giou", "reconstruction-loss", "cosine-similarity"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`The last two lessons' losses compare **one prediction with one target**: probabilities with a class label, or a number with a number. Many important problems do not fit that shape:`,
      },
      {
        t: "list",
        items: [
          "Face verification must decide whether two photos show the **same person**, including people never seen in training — so it cannot have one output per person.",
          "Image search must rank millions of images by **similarity** to a query.",
          "Segmentation must outline a tumour covering 2% of an image, where per-pixel accuracy rewards answering \"no tumour\" everywhere.",
          "A detector is judged by how much its boxes **overlap** the true boxes, not by coordinate errors.",
        ],
      },
      {
        t: "p",
        text: String.raw`**Metric-learning losses** — contrastive (Hadsell et al., 2006), triplet (made popular by FaceNet, 2015) and InfoNCE / NT-Xent (behind modern self-supervised learning) — train an **embedding** (a list of numbers per input) where distance means dissimilarity. The **KL divergence** compares whole probability distributions and drives knowledge distillation and VAEs. **Dice and IoU losses** optimise overlap directly and do not care how big the background is. The **GIoU loss** fixes the IoU loss's zero gradient for non-overlapping boxes. **Reconstruction losses** bring back MSE and cross-entropy for autoencoders.`,
      },
      {
        t: "callout",
        kind: "key",
        title: "The common idea",
        text: "Match the loss to what the system will be judged on: relative distances for retrieval and verification, agreement between distributions for distillation, overlap for segmentation and detection. A loss that measures something else (per-pixel accuracy, coordinate error, a fixed list of classes) can go down while the real goal stays poor.",
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Classification Losses](/learn/m05/classification-losses) — softmax cross-entropy and its gradient $p - y$: InfoNCE is exactly a softmax cross-entropy over similarities, and the hinge $\max(0, \cdot)$ returns in the contrastive and triplet losses.`,
          String.raw`[Regression Losses](/learn/m05/regression-losses) — MSE as Gaussian maximum likelihood, the basis of reconstruction losses.`,
          String.raw`[Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) — the definition $\mathrm{KL}(p\,\|\,q) = \sum_k p_k\log(p_k/q_k)$, why it is never negative, and why it is not symmetric.`,
          String.raw`[Norms and inner-product spaces](/learn/m02/norms-spaces) — Euclidean distance, unit vectors and cosine similarity; for unit vectors $\|u - v\|^2 = 2 - 2\cos(u, v)$.`,
          "[Backpropagation](/learn/m04/backpropagation) — every loss here is differentiated with respect to embeddings, probabilities or box coordinates, then backpropagated into the network.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Arranging guests at a reception",
        text: "You must seat guests so friends sit close and strangers are at least a table apart. The **contrastive** rule looks at one pair at a time: friends are pulled together; strangers are pushed apart only if they are closer than one table-width (the **margin**) — past that, nobody cares how far. The **triplet** rule is relative: each guest's friend must be closer than any stranger by at least the margin; absolute positions do not matter. Useful corrections come from awkward cases — a stranger sitting closer than a friend (a **hard negative**).",
      },
      {
        t: "p",
        text: String.raw`A metric-learning network maps each input to a point (an **embedding**) in a space of, say, 128 dimensions. Training never says "this is person #4127"; it only says "these two are the same, those two are different". Afterwards, checking a pair is a distance against a threshold and search is nearest neighbours — both work for unseen people.`,
      },
      {
        t: "p",
        text: String.raw`InfoNCE turns the same idea into a multiple-choice question: "of these 512 candidates, which is the other view of my image?" — a softmax over similarities, trained with cross-entropy. The **temperature** sets how much the single most confusing wrong candidate drives the answer.`,
      },
      {
        t: "p",
        text: String.raw`For segmentation, imagine grading a student's drawing of a small lesion on a big scan. Counting correctly coloured pixels gives a blank page 98%. The **Dice** score asks instead "how much do the drawn and true regions overlap, relative to their sizes?" — a blank page scores 0, however big the scan. For boxes, IoU asks the same about rectangles; GIoU adds "and if they don't overlap at all, how much empty space lies between them?", so a far-away box learns which way to move.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "Training an embedding with a metric-learning loss" },
      {
        t: "steps",
        items: [
          { title: "Build a batch with structure", text: String.raw`Sample $P$ classes and $K$ examples of each (a "$P\times K$" batch), so every example has positives (same class) and negatives (other classes) in the batch. For self-supervised InfoNCE, take $N$ examples and make two random augmentations ("views") of each.` },
          { title: "Embed", text: String.raw`Run the encoder: $e_i = f_\theta(x_i) \in \mathbb{R}^d$. Usually L2-normalise, $u_i = e_i/\|e_i\|$, so all embeddings sit on the unit sphere and distance and cosine similarity carry the same information.` },
          { title: "Compute all pairwise distances or similarities", text: String.raw`One matrix product gives the $B\times B$ matrix of $\|u_i - u_j\|$ or $u_i^\top u_j$ for the batch.` },
          { title: "Form the training tuples (mining)", text: String.raw`Pairs for the contrastive loss, (anchor, positive, negative) triplets for the triplet loss. Choosing **which** negatives to use is called mining: random, semi-hard, hardest-in-batch. InfoNCE uses every other example in the batch as a negative.` },
          { title: "Evaluate the loss", text: String.raw`Margin-based hinge for contrastive/triplet; softmax cross-entropy over similarities for InfoNCE. Tuples that already meet the margin contribute exactly zero.` },
          { title: "Backpropagate", text: String.raw`Each embedding's gradient is a sum of unit vectors toward positives and away from negatives (derived below), carried back through the normalisation and encoder.` },
          { title: "Evaluate by retrieval", text: String.raw`Measure recall@k (is a correct match among the $k$ nearest neighbours?) or verification accuracy at a distance threshold on **held-out identities** — not the training loss.` },
        ],
      },
      { t: "h", text: "The other families at a glance" },
      {
        t: "table",
        head: ["Family", "Compares", "Typical output layer", "Loss"],
        rows: [
          ["Metric learning", "pairs / triplets / batches of embeddings", "embedding, L2-normalised", "contrastive, triplet, InfoNCE / NT-Xent"],
          ["Distribution matching", "two probability distributions", "logits → softmax, or (μ, log σ²)", "KL divergence (distillation, VAE prior term)"],
          ["Segmentation overlap", "predicted mask vs true mask", "per-pixel logits → sigmoid / softmax", "soft Dice, soft IoU (Jaccard), often + cross-entropy"],
          ["Box overlap", "predicted box vs true box", "4 box coordinates", "IoU loss, GIoU (DIoU, CIoU)"],
          ["Reconstruction", "output vs input", "same shape as input", "MSE (Gaussian), BCE (Bernoulli), L1, perceptual"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Embeddings and distances" },
      {
        t: "p",
        text: String.raw`An encoder $f_\theta: \mathcal{X}\to\mathbb{R}^d$ maps inputs to embeddings. The learned distance is $D_\theta(x, x') = \|f_\theta(x) - f_\theta(x')\|_2$. For unit-length embeddings, $\|u - v\|_2^2 = 2 - 2\,u^\top v$, so squared distance and cosine similarity $u^\top v$ say the same thing. Two gradients appear throughout:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial \|a - b\|}{\partial a} = \frac{a - b}{\|a - b\|} \;(\text{a unit vector}), \qquad \frac{\partial \|a - b\|^2}{\partial a} = 2(a - b), \qquad \frac{\partial \|a - b\|}{\partial b} = -\frac{a - b}{\|a - b\|}`,
      },

      { t: "h", text: "2. Contrastive loss (Hadsell, Chopra and LeCun, 2006)" },
      {
        t: "p",
        text: String.raw`In "Dimensionality Reduction by Learning an Invariant Mapping" (CVPR 2006), each training example is a pair $(x_1, x_2)$ with label $Y = 0$ if similar and $Y = 1$ if dissimilar. With $D = \|f(x_1) - f(x_2)\|$ and margin $m > 0$:`,
      },
      {
        t: "math",
        tex: String.raw`\mathcal{L}(D, Y) = (1 - Y)\,\tfrac12 D^2 \;+\; Y\,\tfrac12\big[\max(0,\; m - D)\big]^2, \qquad \frac{\partial\mathcal{L}}{\partial D} = \begin{cases} D & Y = 0 \text{ (pull)}\\ -(m - D) & Y = 1,\; D < m \text{ (push)}\\ 0 & Y = 1,\; D \ge m \end{cases}`,
      },
      {
        t: "p",
        text: String.raw`By the chain rule, the first embedding's gradient is $\frac{\partial\mathcal{L}}{\partial D}\cdot\frac{e_1 - e_2}{D}$. Similar pair: $e_1 - e_2$, a spring pulling them together. Dissimilar pair inside the margin: $-(m - D)\frac{e_1 - e_2}{D}$, a push of strength $m - D$ that drops to zero at the margin. The second embedding gets the opposite gradient. Two design points:`,
      },
      {
        t: "list",
        items: [
          String.raw`**Why the push term is needed.** With only the similar-pair term, the best solution is $f(x) = \text{const}$: every distance 0, loss 0. The push term makes this collapse cost $\tfrac12 m^2$ per dissimilar pair.`,
          String.raw`**Why a margin.** Without it, dissimilar pairs could be pushed apart forever by scaling up the embeddings. With it, only pairs closer than $m$ count — far-apart negatives waste no effort, and $m$ fixes the embedding's scale.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Worked example** ($m = 1$): a similar pair at $D = 0.3$ costs $\tfrac12(0.09) = 0.045$, with $\partial\mathcal{L}/\partial D = 0.3$. A dissimilar pair at $D = 0.3$ costs $\tfrac12(0.7)^2 = 0.245$ with $\partial\mathcal{L}/\partial D = -0.7$; at $D = 1.2$ it costs 0 with zero gradient.`,
      },

      { t: "h", text: "3. Triplet loss (FaceNet, Schroff et al., 2015)" },
      {
        t: "p",
        text: String.raw`A **triplet** is an anchor $a$, a positive $p$ (same identity) and a negative $n$ (different identity). FaceNet ("FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015) embeds faces on the unit sphere in 128 dimensions and minimises`,
      },
      {
        t: "math",
        tex: String.raw`\mathcal{L} = \sum_{(a,p,n)} \Big[\, \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha \,\Big]_+, \qquad [z]_+ = \max(0, z)`,
        caption: String.raw`$\alpha$ is the margin (0.2 in the paper). A common variant (PyTorch's TripletMarginLoss) uses non-squared distances: $[\,d(a,p) - d(a,n) + m\,]_+$.`,
      },
      {
        t: "p",
        text: String.raw`Unlike the contrastive loss, the triplet loss is **relative**: each anchor must be closer to its positive than to its negative by the margin, with no rule on absolute distances. Classes may have different spreads. For the Euclidean variant, with unit vectors $u = (a - p)/d(a,p)$ and $v = (a - n)/d(a,n)$, an active triplet ($d(a,p) - d(a,n) + m > 0$) has gradients`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\mathcal{L}}{\partial a} = u - v, \qquad \frac{\partial\mathcal{L}}{\partial p} = -u, \qquad \frac{\partial\mathcal{L}}{\partial n} = v`,
      },
      {
        t: "p",
        text: String.raw`So gradient descent moves $p$ along $u$ (toward $a$), $n$ along $-v$ (away from $a$), and $a$ along $v - u$ (toward $p$, away from $n$). Every active triplet gives unit-size forces whatever its loss (the hinge is fully on or off). In the squared (FaceNet) form the gradients are $2(n - p)$, $-2(a - p)$ and $2(a - n)$: bigger distances, bigger forces.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example** (Euclidean, $m = 0.5$): $a = (0, 0)$, $p = (0.6, 0.8)$, $n = (1.2, 0)$. Then $d(a,p) = 1$, $d(a,n) = 1.2$ and $\mathcal{L} = \max(0, 1 - 1.2 + 0.5) = 0.3$. $u = (-0.6, -0.8)$, $v = (-1, 0)$, so $\partial\mathcal{L}/\partial a = (0.4, -0.8)$, $\partial\mathcal{L}/\partial p = (0.6, 0.8)$, $\partial\mathcal{L}/\partial n = (-1, 0)$. In the FaceNet form with $\alpha = 0.2$, the same triplet has $1 - 1.44 + 0.2 < 0$: already satisfied, zero loss.`,
      },
      { t: "h", text: "Easy, semi-hard and hard negatives" },
      {
        t: "p",
        text: String.raw`For a fixed anchor–positive pair at distance $d_{ap}$, negatives fall into three groups:`,
      },
      {
        t: "table",
        head: ["Negative", "Condition", "Loss", "Effect"],
        rows: [
          ["easy", String.raw`$d_{an} \ge d_{ap} + m$`, "0", "no gradient — wasted computation"],
          ["semi-hard", String.raw`$d_{ap} < d_{an} < d_{ap} + m$`, String.raw`in $(0, m)$`, "right order but inside the margin: useful, stable signal"],
          ["hard", String.raw`$d_{an} < d_{ap}$`, String.raw`$> m$`, "wrong order: biggest signal, but often label noise or outliers"],
        ],
      },
      {
        t: "p",
        text: String.raw`With $N$ examples there are $O(N^3)$ triplets, and after a little training almost all are easy, so random sampling soon stops giving gradient. **Mining** picks informative triplets, usually **online** inside each mini-batch. FaceNet used **semi-hard** negatives in large batches and reported that picking the hardest negatives early in training can collapse the model to $f(x) = 0$.`,
      },
      {
        t: "p",
        text: String.raw`Why? If for many anchors the closest negative is nearer than a random positive, the loss $d_{ap} - d_{an} + m$ is above $m$. Shrinking every embedding toward one point brings it down to $m$ — so collapse is a stable resting point. Hermans et al. (2017) proposed **batch-hard** mining — per anchor, the farthest positive and closest negative in a $P\times K$ batch — which works well when batches are well structured. The from-scratch code reproduces the collapse: a 2-D linear embedding trained with the hardest negatives shrinks to mean norm 0.006.`,
      },

      { t: "h", text: "4. InfoNCE and NT-Xent" },
      {
        t: "p",
        text: String.raw`InfoNCE (van den Oord, Li and Vinyals, 2018, "Representation Learning with Contrastive Predictive Coding") scores one positive against many negatives at once. With anchor $u$, positive $u^+$, negatives $u^-_1,\dots,u^-_K$, similarity $s(\cdot,\cdot)$ (cosine for unit vectors) and temperature $\tau > 0$:`,
      },
      {
        t: "math",
        tex: String.raw`\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp\big(s(u, u^+)/\tau\big)}{\exp\big(s(u, u^+)/\tau\big) + \sum_{k=1}^{K}\exp\big(s(u, u^-_k)/\tau\big)}`,
      },
      {
        t: "p",
        text: String.raw`This is the previous lesson's softmax cross-entropy, with logits $s/\tau$ and the positive as the correct class. So the gradient with respect to the similarities is "probability minus target", with $q_j$ the softmax weights:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\mathcal{L}}{\partial s(u, u^+)} = \frac{q_+ - 1}{\tau}, \qquad \frac{\partial\mathcal{L}}{\partial s(u, u^-_k)} = \frac{q_k}{\tau}, \qquad q_j = \frac{e^{s_j/\tau}}{\sum_i e^{s_i/\tau}}`,
      },
      {
        t: "p",
        text: String.raw`Each negative is pushed in proportion to its softmax weight: **InfoNCE does soft hard-negative mining by itself**, and $\tau$ sets how hard. Example with $s^+ = 0.8$ and negatives $0.5, 0.1, -0.3$ (computed in the code):`,
      },
      {
        t: "table",
        head: [String.raw`$\tau$`, "loss", "softmax weights of the negatives", "share of the push on the hardest negative"],
        rows: [
          ["1.0", "0.9440", "0.2882, 0.1932, 0.1295", "0.472"],
          ["0.5", "0.6451", "0.2879, 0.1294, 0.0581", "0.606"],
          ["0.1", "0.0495", "0.0474, 0.0009, 0.0000", "0.982"],
        ],
      },
      {
        t: "p",
        text: String.raw`Small $\tau$ puts almost all the gradient on the closest negative (and multiplies its size by $1/\tau$); large $\tau$ spreads it evenly. Too small is unstable and over-sensitive to false negatives; too large ignores informative ones. van den Oord et al. showed InfoNCE bounds the mutual information between the two views: $I \ge \log(K + 1) - \mathcal{L}_{\text{InfoNCE}}$, so more negatives allow a tighter bound.`,
      },
      {
        t: "p",
        text: String.raw`**NT-Xent** ("normalised temperature-scaled cross-entropy", SimCLR — Chen, Kornblith, Norouzi and Hinton, 2020) is InfoNCE applied symmetrically to $N$ images with two augmented views each. For each of the $2N$ embeddings, its other view is the positive and the other $2N - 2$ are negatives, so the softmax runs over $2N - 1$ candidates. A network that cannot tell them apart has loss $\log(2N - 1)$, e.g. $\log 511 = 6.2364$ for $N = 256$. CLIP (Radford et al., 2021) uses the same loss between image and text embeddings, in both directions, with a learned temperature. More in [Self-Supervised Learning](/learn/m15/self-supervised).`,
      },

      { t: "h", text: "5. KL divergence as a loss" },
      {
        t: "math",
        tex: String.raw`\mathrm{KL}(p\,\|\,q) = \sum_k p_k \log\frac{p_k}{q_k} = \underbrace{-\sum_k p_k\log q_k}_{H(p,\,q)} - \underbrace{\Big(-\sum_k p_k\log p_k\Big)}_{H(p)} \;\ge 0`,
      },
      {
        t: "p",
        text: String.raw`(see [Information Theory](/learn/m02/information-theory)). When $p$ is a fixed target and $q = q_\theta$ is the model, $H(p)$ is a constant, so minimising $\mathrm{KL}(p\,\|\,q_\theta)$ and the cross-entropy $H(p, q_\theta)$ give identical gradients. The values differ by $H(p)$, which is why a KL loss can reach 0 but a soft-target cross-entropy cannot. KL is **not symmetric**: forward $\mathrm{KL}(p\,\|\,q)$ punishes $q_k \approx 0$ where $p_k > 0$, so $q$ spreads to cover all of $p$; reverse $\mathrm{KL}(q\,\|\,p)$ punishes mass where $p$ has none, so $q$ settles on one mode (peak).`,
      },
      {
        t: "p",
        text: String.raw`**Knowledge distillation** (Hinton, Vinyals and Dean, 2015, "Distilling the Knowledge in a Neural Network") trains a small student to match a large teacher's softened outputs. With teacher logits $t$, student logits $z$ and temperature $T$, $p^T = \mathrm{softmax}(t/T)$, $q^T = \mathrm{softmax}(z/T)$:`,
      },
      {
        t: "math",
        tex: String.raw`\mathcal{L}_{\text{KD}} = T^2\,\mathrm{KL}\big(p^T\,\|\,q^T\big), \qquad \frac{\partial\mathcal{L}_{\text{KD}}}{\partial z_k} = T\,\big(q^T_k - p^T_k\big)`,
      },
      {
        t: "p",
        text: String.raw`A softmax at $T > 1$ reveals the teacher's small probabilities ("a 2 looks a bit like a 7"). The $T^2$ factor makes up for gradients shrinking like $1/T^2$ as $T$ grows, keeping this term comparable to the ordinary hard-label cross-entropy usually added to it. **Worked example:** teacher logits $(3, 1, 0.2)$, student $(2, 1.5, 0)$, $T = 2$: $p^T = (0.6194, 0.2279, 0.1527)$, $q^T = (0.4658, 0.3628, 0.1714)$, $\mathcal{L}_{\text{KD}} = 0.2117$, gradient $2(q^T - p^T) = (-0.3071, 0.2699, 0.0373)$ — nudging the student toward the teacher's ranking.`,
      },
      {
        t: "p",
        text: String.raw`**Variational autoencoders** (Kingma and Welling, "Auto-Encoding Variational Bayes", 2013) add $\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,\mathcal{N}(0, I)\big)$ to the reconstruction loss. For a diagonal Gaussian encoder output $(\mu, \log\sigma^2)$ it has a closed form:`,
      },
      {
        t: "math",
        tex: String.raw`\mathrm{KL}\big(\mathcal{N}(\mu, \operatorname{diag}\sigma^2)\,\|\,\mathcal{N}(0, I)\big) = \frac12\sum_{j=1}^{d}\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big)`,
      },
      {
        t: "p",
        text: String.raw`For $\mu = (0.5, -1)$, $\log\sigma^2 = (0, -1)$: $\tfrac12(0.25 + 1 - 1 - 0) + \tfrac12(1 + e^{-1} - 1 + 1) = 0.125 + 0.683940 = 0.808940$; a Monte-Carlo estimate with 200,000 samples in the code gives 0.8095. It pulls each latent toward $\mu = 0$, $\sigma = 1$ (where it is zero). See [Variational Autoencoders](/learn/m14/vae).`,
      },

      { t: "h", text: "6. Dice loss for segmentation" },
      {
        t: "p",
        text: String.raw`For a binary mask, let $g_i \in \{0, 1\}$ be pixel $i$'s true label and $p_i \in [0, 1]$ its predicted foreground probability. The **Dice coefficient** of sets $A$ (prediction) and $B$ (truth) is $2|A\cap B|/(|A| + |B|)$ — the same as the foreground F1 score. Replacing set sizes by sums of probabilities gives the differentiable **soft Dice**:`,
      },
      {
        t: "math",
        tex: String.raw`\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon}`,
        caption: String.raw`$\varepsilon$ (often 1) avoids 0/0 on empty masks and smooths the loss. Milletari, Navab and Ahmadi's V-Net (2016) used squared terms $\sum p_i^2 + \sum g_i^2$ in the denominator.`,
      },
      {
        t: "p",
        text: String.raw`With $I = \sum_i p_i g_i$ and $S = \sum_i p_i + \sum_i g_i + \varepsilon$, the quotient rule gives the per-pixel gradient`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\mathcal{L}_{\text{Dice}}}{\partial p_i} = -\frac{2 g_i S - (2I + \varepsilon)}{S^2} = \begin{cases} -\dfrac{2S - (2I + \varepsilon)}{S^2} & g_i = 1 \text{ (raise } p_i)\\[2mm] +\dfrac{2I + \varepsilon}{S^2} & g_i = 0 \text{ (lower } p_i) \end{cases}`,
      },
      {
        t: "p",
        text: String.raw`Two things to notice. (1) The gradient depends on the pixel only through $g_i$: every foreground pixel gets the same push, every background pixel the same pull, with sizes set by **global** sums. (2) $I$ and $S$ grow with the object's size, not the image's, so adding background pixels with $p_i \approx 0$ changes nothing. **Worked example:** four pixels, $p = (0.9, 0.6, 0.2, 0.1)$, $g = (1, 1, 0, 0)$, $\varepsilon = 0$: $I = 1.5$, $S = 1.8 + 2 = 3.8$, Dice $= 3/3.8 = 0.789474$, loss $0.210526$; gradients $-4.6/14.44 = -0.318560$ on foreground pixels and $+3/14.44 = +0.207756$ on background pixels.`,
      },
      { t: "h", text: "Why Dice helps with class imbalance — and where pixel-wise cross-entropy fails" },
      {
        t: "p",
        text: String.raw`Pixel-wise binary cross-entropy **averages over pixels**, so with 2% foreground the background dominates loss and gradient. In the code (a 64×64 image with an 81-pixel lesion, 1.98% foreground), predicting a constant 0.01 everywhere gets BCE 0.1009 — **lower** than a prediction that finds the lesion with $p = 0.9$ inside and $p = 0.1$ outside (BCE 0.1054). Soft Dice ranks them the other way (0.9787 vs 0.7362). At a uniform prediction of 0.1, Dice's gradient on a foreground pixel is 56 times that on a background pixel; BCE's ratio is 9. Dice re-weights the rare class by itself.`,
      },

      { t: "h", text: "7. IoU (Jaccard) loss" },
      {
        t: "math",
        tex: String.raw`J(A, B) = \frac{|A\cap B|}{|A\cup B|}, \qquad \mathcal{L}_{\text{IoU}} = 1 - \frac{\sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i - \sum_i p_i g_i + \varepsilon}, \qquad \text{Dice} = \frac{2J}{1 + J},\;\; J = \frac{\text{Dice}}{2 - \text{Dice}}`,
      },
      {
        t: "p",
        text: String.raw`The Jaccard index (intersection over union) is the standard segmentation metric (mean IoU over classes). **For hard masks**, Dice and $J$ always move together, so they rank predictions the same; as soft losses they weight errors slightly differently ($J \le$ Dice, and IoU punishes partial overlap more). In the worked example, soft IoU is $1.5/(3.8 - 1.5) = 0.652174$ (loss 0.347826) against Dice 0.789474. The Lovász-softmax loss (Berman et al., 2018) is a tractable convex surrogate of IoU for multi-class segmentation.`,
      },
      { t: "h", text: "8. Dice versus cross-entropy, and the combination" },
      {
        t: "table",
        head: ["", "Pixel-wise cross-entropy", "Soft Dice / IoU"],
        rows: [
          ["Normalised by", "number of pixels", "size of the object (and prediction)"],
          ["Class imbalance", "dominated by the majority class unless re-weighted", "insensitive to background size"],
          ["Gradient", String.raw`local, $(p_i - g_i)$ per logit, smooth and well-behaved`, "global, depends on batch/image sums; noisy for tiny or empty objects"],
          ["Calibration", "proper scoring rule: calibrated probabilities", "pushes probabilities toward 0/1; not calibrated"],
          ["Empty ground truth", "well defined", String.raw`ill-defined without $\varepsilon$; any false positive is punished hard`],
        ],
      },
      {
        t: "p",
        text: String.raw`Their weaknesses complement each other, so many segmentation systems minimise $\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Dice}}$ — the default of the popular nnU-Net framework (Isensee et al., 2021). Cross-entropy gives stable per-pixel gradients early on; Dice keeps small structures from being ignored. Multi-class versions average Dice over classes (optionally weighting each class by the inverse square of its volume, the "generalised Dice" of Sudre et al., 2017).`,
      },

      { t: "h", text: "9. Bounding boxes: IoU loss and GIoU" },
      {
        t: "p",
        text: String.raw`Detectors are scored by the IoU between predicted box $B$ and true box $G$ (e.g. a detection counts if IoU ≥ 0.5). L1 or smooth L1 on the four coordinates treats them as independent and ignores scale: 5 pixels off is huge for a 10-pixel box, nothing for a 500-pixel one. The **IoU loss** $1 - \mathrm{IoU}(B, G)$ (UnitBox, Yu et al., 2016, used $-\ln\mathrm{IoU}$) optimises the evaluation measure directly and is scale-invariant. Its flaw: if the boxes do not overlap, IoU is 0 however far apart they are, so the loss is constant and **its gradient is exactly zero**.`,
      },
      {
        t: "p",
        text: String.raw`**Generalised IoU** (Rezatofighi et al., CVPR 2019) subtracts the part of the smallest enclosing box $C$ not covered by the union:`,
      },
      {
        t: "math",
        tex: String.raw`\mathrm{GIoU}(B, G) = \mathrm{IoU}(B, G) - \frac{|C \setminus (B\cup G)|}{|C|}, \qquad \mathcal{L}_{\text{GIoU}} = 1 - \mathrm{GIoU} \in [0, 2)`,
      },
      {
        t: "p",
        text: String.raw`Properties: $-1 < \mathrm{GIoU} \le \mathrm{IoU} \le 1$; GIoU equals IoU when the union fills its enclosing box, and tends to $-1$ for tiny boxes far apart. **Worked examples** with $G = [0, 0, 2, 2]$ (corners $x_1, y_1, x_2, y_2$):`,
      },
      {
        t: "table",
        head: ["Prediction B", "IoU", "enclosing box C", "GIoU", "IoU loss", "GIoU loss"],
        rows: [
          ["[1, 1, 3, 3] (overlap 1)", "1/7 = 0.1429", "[0,0,3,3], area 9", "1/7 − 2/9 = −0.0794", "0.8571", "1.0794"],
          ["[2, 0, 4, 2] (touching)", "0", "[0,0,4,2], area 8", "0", "1", "1"],
          ["[3, 0, 5, 2] (gap 1)", "0", "[0,0,5,2], area 10", "−2/10 = −0.2", "1", "1.2"],
          ["[6, 0, 8, 2] (gap 4)", "0", "[0,0,8,2], area 16", "−8/16 = −0.5", "1", "1.5"],
        ],
      },
      {
        t: "p",
        text: String.raw`The IoU loss cannot tell the last three apart; the GIoU loss grows with the gap. Autograd gives the box at gap 1 the gradient $(0.2, 0, -0.04, 0)$ with respect to $(x_1, y_1, x_2, y_2)$: moving the left edge toward $G$ helps most. The small negative entry for $x_2$ shows GIoU's known weakness: enlarging the box also shrinks the empty part of $C$, so for non-overlapping boxes GIoU tends to **grow** the box before moving it. DIoU and CIoU (Zheng et al., 2020) add a penalty on the normalised centre distance (and, for CIoU, aspect ratio) to converge faster.`,
      },

      { t: "h", text: "10. Reconstruction losses" },
      {
        t: "p",
        text: String.raw`An [autoencoder](/learn/m14/autoencoders) rebuilds its input $x$ as $\hat x = g(f(x))$. The reconstruction loss is again a negative log-likelihood under a noise model for each output dimension:`,
      },
      {
        t: "list",
        items: [
          String.raw`**MSE** $\frac1D\sum_j(\hat x_j - x_j)^2$ — Gaussian decoder with fixed variance; for real-valued data (standardised features, audio). It favours blurry outputs, because the average of several plausible reconstructions minimises it (see [Regression Losses](/learn/m05/regression-losses)).`,
          String.raw`**BCE** $-\frac1D\sum_j[x_j\log\hat x_j + (1 - x_j)\log(1 - \hat x_j)]$ with $\hat x_j = \sigma(\text{logit}_j)$ — Bernoulli decoder; standard for images with intensities in $[0, 1]$. For non-binary targets (e.g. $x_j = 0.3$) the minimum is at $\hat x_j = x_j$ but equals the entropy $H(x_j) > 0$, not 0 — so its value is not an error measure; compare models by MSE or PSNR.`,
          String.raw`**L1** — Laplace decoder; less blur than MSE, common in image restoration.`,
          String.raw`**Perceptual losses** (Johnson, Alahi and Fei-Fei, 2016) compare feature maps of a fixed pretrained network instead of pixels, judging structure and texture rather than exact pixel values.`,
        ],
      },
      {
        t: "p",
        text: String.raw`In a VAE the full objective is reconstruction NLL + KL term (section 5). Their balance depends on whether the reconstruction term is summed or averaged over dimensions: mixing a mean MSE with a summed KL quietly changes the balance by a factor of $D$.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "sim", id: "metric-learning", caption: "Contrastive & Triplet Embeddings: a linear map from 8-D to 2-D trained in your browser on five classes." },
      {
        t: "p",
        text: "Five classes in 8 dimensions; only the first four coordinates carry class information, the other four are large noise. The embedding e = Wx (W is 2×8) is trained by mini-batch SGD (16 anchors per step). The highlighted a, p, n are the batch's first tuple; the dashed gold circle is the margin; arrows show −∂ℓ/∂e. All of it is the real computation.",
      },
      {
        t: "list",
        items: [
          "**Step 0.** The random projection mixes the classes: 1-NN test accuracy is 62%. Press **Play** (triplet loss, semi-hard negatives, margin 1): within about 25 steps the classes separate and 1-NN accuracy reaches 98%.",
          String.raw`**Watch the active count.** "Active triplets k/16" counts triplets with non-zero loss. With semi-hard mining it starts near 12/16 and falls to about 3/16 by step 250–300; only these produce a gradient.`,
          String.raw`**Switch negatives to random.** Most random triplets are already satisfied (the narration says "margin satisfied, zero gradient"): on average only about 3.5 of 16 are active in the first 20 steps and fewer than 1 near the end, so learning is slower (96% at step 100 instead of 100%). This is why mining exists.`,
          String.raw`**Switch to hard negatives.** The negative is now the closest point of another class — often inside the circle, sometimes closer than the positive (loss above m). Here it trains as fast as semi-hard; compare the from-scratch code, where hardest-negative mining collapses a 2-D embedding of real digits.`,
          String.raw`**Contrastive loss.** Each step shows one pair. Positive pairs (accent line) are always pulled, with force proportional to distance; negative pairs (dashed clay line) are pushed only inside the circle of radius m. The embedding's overall size is set by m, whereas with the triplet loss only differences of distances matter.`,
          String.raw`**Change the margin.** A larger margin (2) keeps more tuples active longer (about 4 of 16 near step 300, versus about 1.6 with margin 0.3); a small margin is met quickly and stops shaping the embedding early.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: String.raw`**Part 1 — metric learning.** The contrastive loss and both triplet variants with analytic gradients, checked against finite differences; then a linear 64→2 embedding of the 8×8 *digits* dataset (1797 images, 1200 for training) trained with the triplet loss and random, semi-hard or hardest negatives. All mining is vectorised over the batch.`,
      },
      { t: "code", lang: "python", title: "metric_learning_from_scratch.py", code: SCRATCH_METRIC, output: SCRATCH_METRIC_OUT },
      {
        t: "list",
        items: [
          String.raw`All four analytic gradients match central differences to $\sim10^{-10}$.`,
          String.raw`A random 2-D projection gives 16.4% nearest-neighbour accuracy (chance is 10%). Random and semi-hard negatives both reach about 64% — impressive for a **linear** map to only 2 dimensions. With random negatives the active fraction falls from 0.89 to 0.35: two thirds of the work is wasted at the end. Semi-hard keeps nearly all selected triplets active (0.99).`,
          String.raw`**Hardest negatives collapse the embedding**: mean norm shrinks from 0.180 to 0.006 and accuracy drops to 36%, with every triplet still active. This is FaceNet's warned-of failure: in 2-D most anchors have a negative closer than their positive, and pulling everything together is the easiest way to bring $d_{ap} - d_{an} + m$ down toward $m$.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Part 2 — InfoNCE, KL, Dice/IoU and GIoU.** NT-Xent with its full gradient through the L2 normalisation (checked numerically), the temperature table from the Math section, distillation and VAE KL terms, Dice versus cross-entropy on an imbalanced mask, and gradient descent on a box with the IoU and GIoU losses.`,
      },
      { t: "code", lang: "python", title: "overlap_and_distribution_losses.py", code: SCRATCH_OVERLAP, output: SCRATCH_OVERLAP_OUT },
      {
        t: "list",
        items: [
          String.raw`The NT-Xent gradient is the cross-entropy gradient $(P - \text{onehot})/2N$ for the similarity matrix, pushed through $S = UU^\top/\tau$ (hence $(G + G^\top)U/\tau$) and through the normalisation (removing the part along each $u_i$). It matches finite differences to $3\times10^{-10}$.`,
          String.raw`The numerical distillation gradient equals $T(q^T - p^T)$, and the Monte-Carlo VAE KL (0.8095) matches the closed form (0.8089).`,
          String.raw`**Segmentation.** BCE prefers the all-background prediction (0.1009) to a correct lesion with 10% background noise (0.1054); Dice and IoU losses rank them correctly. Even the "perfect" prediction with $p = 0.01$ on the background has Dice loss 0.2024: the 4015 background pixels add $\sum p_i \approx 40$ to the denominator. This pushes networks toward very confident outputs. The shifted lesion has hard Dice 0.3086.`,
          String.raw`**Boxes.** Starting at $[4, 3, 5, 4]$, which misses the target, the IoU-loss box never moves (zero gradient); the GIoU-loss box converges to $[-0.089, -0.020, 2.088, 2.069]$, IoU 0.880 with the target after 300 steps.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "A 64→128→4 MLP embedding of the digits, trained only on digits 0–6 with three losses (contrastive over all pairs of a P×K batch, triplet with batch-hard mining, and self-supervised NT-Xent at three temperatures), scored by recall@1 retrieval on test images of seen digits and unseen digits 7–9, against PCA to the same 4 dimensions. Then KL distillation with F.kl_div, a Dice + cross-entropy segmentation loss, and IoU vs GIoU gradients for a non-overlapping box.",
              },
              { t: "code", lang: "python", title: "advanced_losses_torch.py", code: TORCH, output: TORCH_OUT },
              {
                t: "list",
                items: [
                  String.raw`**Seen classes:** both supervised metric losses clearly beat PCA (0.988 and 0.976 vs 0.927 recall@1 in 4 dimensions). **Unseen classes:** they are worse than PCA (0.620 and 0.603 vs 0.788). With only 7 training classes, the embedding specialises in those 7 and throws away directions that would separate 7, 8 and 9. Real systems train on thousands of identities. Always evaluate metric learning on **held-out classes**.`,
                  String.raw`NT-Xent uses no labels, so it is a bit worse on seen digits (0.908 at $\tau = 0.2$) but the best learned embedding on unseen digits (0.783): augmentation invariance is less tied to the training classes. Both $\tau = 0.1$ and $\tau = 0.5$ do worse than 0.2 — temperature really is a hyperparameter.`,
                  "F.kl_div(input, target) expects the model's **log**-probabilities as input and probabilities as target (the reverse of the maths KL(p‖q) order). reduction='batchmean' gives the true KL per example; the default 'mean' also divides by the number of elements. The result matches the hand computation (0.2117).",
                  "With logits −3 everywhere (p ≈ 0.047), the combined loss's gradient is about 31 times larger on a foreground pixel than on a background one.",
                  "For the box one unit right of the target, the IoU loss gradient is exactly zero; GIoU gives a usable direction.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "code",
                lang: "python",
                title: "advanced_losses_keras.py (not executed in CI)",
                code: KERAS,
                output: "Prints the distillation loss T²·KL ≈ 0.2117 (the same value as the NumPy and PyTorch versions) and the GIoU loss [1.2] for the box [3, 0, 5, 2] against [0, 0, 2, 2]. The triplet, NT-Xent and Dice + CE functions are defined for use in a custom training loop or model.compile.",
              },
              {
                t: "list",
                items: [
                  "tf.keras.losses.KLDivergence takes (y_true, y_pred) as **probabilities** and computes KL(y_true ‖ y_pred) — unlike PyTorch's kl_div, which takes log-probabilities for the prediction.",
                  "TensorFlow Addons used to ship triplet losses with semi-hard and hard mining, but it is no longer maintained, so writing the few lines yourself, as above, is safer.",
                ],
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      {
        t: "table",
        head: ["Loss", "Key idea", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Contrastive (pairs)", "pull similar pairs, push dissimilar ones beyond margin m", "simple; needs only pair labels", "one absolute margin for all classes; most negative pairs easy", "siamese verification, dimensionality reduction"],
          ["Triplet", "anchor closer to positive than to negative by m", "relative constraint; classes may spread differently", String.raw`$O(N^3)$ triplets; needs mining; collapse with hardest negatives`, "face recognition, person re-identification, retrieval"],
          ["Batch-hard / batch-all triplet", "online mining within a P×K batch", "no offline mining; efficient", "sensitive to label noise (batch-hard)", "re-identification"],
          ["N-pair / InfoNCE / NT-Xent", "softmax over one positive and many negatives", "uses all negatives in the batch; built-in soft hard-negative weighting", "needs large batches (or a memory bank); temperature to tune", "self-supervised learning, CLIP-style multimodal training"],
          ["Supervised contrastive", "InfoNCE with all same-class samples as positives (Khosla et al., 2020)", "label-aware, robust", "large batches", "representation learning with labels"],
          ["Margin-softmax (ArcFace etc.)", "classification with an angular margin on normalised features (Deng et al., 2019)", "no mining; stable", "needs a class list at training time", "large-scale face recognition"],
          ["KL divergence", "match a target distribution", "uses full soft targets; closed form for Gaussians", "not symmetric: direction matters", "distillation, VAE prior, policy regularisation"],
          ["Soft Dice / soft IoU", "differentiable overlap", "insensitive to background size", "noisy gradients for tiny or empty masks; uncalibrated", "medical and small-object segmentation"],
          ["Tversky / focal variants", "weight false positives vs false negatives differently", "tunable precision/recall trade-off", "extra hyperparameters", "highly imbalanced segmentation"],
          ["Lovász-softmax", "convex surrogate of IoU", "directly targets mean IoU", "more complex to implement", "semantic segmentation benchmarks"],
          ["IoU / GIoU / DIoU / CIoU", "box overlap, plus enclosing-box or centre-distance terms", "scale-invariant; matches evaluation metric", "IoU: zero gradient without overlap; GIoU: slow, enlarges boxes", "bounding-box regression"],
          ["Reconstruction (MSE / BCE / L1 / perceptual)", "compare output with input", "simple, likelihood-based", "pixel losses favour blur", "autoencoders, restoration"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Face verification and recognition.** FaceNet trained a 128-D face embedding with the triplet loss and semi-hard online mining: verification thresholds a distance, recognition is nearest-neighbour search, clustering groups photos of one person. Later systems mostly moved to margin-based softmax losses such as ArcFace.",
          "**Person re-identification.** Matching pedestrians across cameras with triplet losses and batch-hard mining (Hermans et al., 2017) is a standard baseline.",
          "**Image and product retrieval.** Embeddings trained with contrastive or triplet losses power \"find similar items\" search over large catalogues with approximate nearest-neighbour indexes.",
          "**Self-supervised and multimodal pretraining.** SimCLR (NT-Xent between augmented views) and CLIP (InfoNCE between images and captions) learn general representations without class labels; CLIP's embeddings allow zero-shot classification by comparing an image with text prompts.",
          "**Model compression.** Knowledge distillation with a KL term trains small students from large teachers — DistilBERT (Sanh et al., 2019) is a well-known distilled language model.",
          "**Medical image segmentation.** Organs and lesions are small relative to scans; Dice + cross-entropy is nnU-Net's default loss, strong across many medical challenges.",
          "**Object detection.** IoU-family losses (GIoU, DIoU, CIoU) handle box regression in many modern detectors, instead of or alongside smooth L1.",
          "**Generative models.** VAEs add a KL term to a reconstruction loss; many image-to-image models use L1/MSE plus perceptual losses.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Works for new classes, but harder to train.** Metric losses can generalise to unseen classes, but are harder to optimise than softmax classification: most tuples are uninformative, and progress depends on batch construction and mining.",
          "**Mining is a bias–variance knob.** Harder negatives give bigger, more informative gradients but amplify label noise and can collapse the embedding; easier ones are stable but slow. Semi-hard and soft weighting (InfoNCE temperature) sit between.",
          "**Batch size matters.** InfoNCE's negatives come from the batch, so small batches mean few negatives and a weaker signal; memory banks or queues of past embeddings are a common fix.",
          "**Margins and temperatures depend on normalisation.** On the unit sphere distances are at most 2, so a margin of 1 is huge and 0.2 is typical; without normalisation the network can meet any margin by scaling up its outputs.",
          "**Overlap losses match the metric but have global, noisier gradients.** Dice/IoU need care with empty masks and tiny objects and give poorly calibrated probabilities; adding cross-entropy is the usual compromise.",
          "**GIoU fixes zero gradients but converges slowly** for distant boxes because it first enlarges the prediction; DIoU/CIoU add explicit centre-distance terms.",
          "**KL direction is a modelling choice**, not a detail: forward KL averages over modes, reverse KL picks one.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Batches without positives.** A random batch from thousands of identities has almost no same-identity pairs, so nothing gets pulled together. Fix: P×K sampling (P classes, K examples each).",
          "**Forgetting to normalise embeddings** (or normalising after the loss). Without L2 normalisation the network can meet a margin by scaling, and cosine-based InfoNCE assumes unit vectors. Fix: normalise inside the model before computing distances, and pick margins for the normalised geometry.",
          "**Taking the norm of a zero vector.** The gradient of ‖a − b‖ is undefined when a = b (duplicate images, or an anchor compared with itself). Fix: exclude the diagonal, add a small ε inside the square root, or use squared distances.",
          "**Using the anchor itself as a negative or positive.** In NT-Xent the diagonal of the similarity matrix must be masked; in batch-hard mining the anchor must not be its own hardest positive (distance 0 is harmless there but wrong as a negative).",
          "**Mining the hardest negatives from the start.** Can collapse the embedding (shown above) and over-fits mislabelled pairs. Fix: semi-hard mining, warm-up with easier negatives, or InfoNCE with a moderate temperature.",
          "**Evaluating metric learning on training classes only.** The embedding's whole point is to work for new identities; recall@k on seen classes can look great while unseen-class retrieval is poor (as in the PyTorch example).",
          "**Calling F.kl_div with probabilities as input, or with swapped arguments.** PyTorch expects the model's log-probabilities first and target probabilities second; swapping computes a different divergence. Use reduction='batchmean' for the correct value, and multiply by T² in distillation.",
          "**Computing Dice per pixel or over the wrong axes.** Dice is a ratio of sums: one Dice over the whole batch and per-image Dice averaged over images are different losses (the latter weights small objects more). Choose on purpose, and handle images with empty ground truth with ε.",
          "**Applying sigmoid twice** (in the model and again in the Dice/BCE code), or feeding logits to a loss that expects probabilities.",
          "**Using IoU loss alone from random initial boxes.** Non-overlapping boxes get zero gradient. Fix: GIoU/DIoU, or add an L1 term on coordinates.",
          "**Reporting BCE as a reconstruction error for non-binary images.** Its minimum is the targets' entropy, not 0. Report MSE/PSNR.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "Why use metric learning instead of a softmax classifier for face verification?", a: "A softmax classifier has one output per identity known at training time and cannot handle new people. Verification asks whether two images show the same person, including people never seen before. Metric learning trains an embedding where distance reflects identity, so any two faces can be compared by thresholding their distance, and new identities need no retraining." },
      { t: "qa", level: "beginner", q: "What does the margin in the contrastive and triplet losses do?", a: String.raw`In the contrastive loss, dissimilar pairs are penalised only if closer than $m$; beyond it they contribute nothing. This prevents collapse and wastes no effort on pairs already far apart. In the triplet loss, the margin demands the negative be farther than the positive by at least $m$, so the network cannot satisfy the constraint with all distances equal (or zero).` },
      { t: "qa", level: "beginner", q: "What is the Dice coefficient and why is it used for segmentation?", a: String.raw`Dice $= 2|A\cap B|/(|A| + |B|)$: the overlap of predicted and true masks relative to their sizes (equal to the foreground F1 score). Unlike pixel accuracy or pixel-averaged cross-entropy, it does not reward predicting the dominant background, so it suits small structures such as lesions.` },
      { t: "qa", level: "intermediate", q: "Explain easy, semi-hard and hard negatives, and why mining matters.", a: String.raw`For anchor–positive distance $d_{ap}$ and margin $m$: easy negatives have $d_{an} \ge d_{ap} + m$ (zero loss and gradient), semi-hard have $d_{ap} < d_{an} < d_{ap} + m$ (right order, inside the margin), hard have $d_{an} < d_{ap}$. Most random triplets become easy early, so random sampling wastes computation. Mining picks informative triplets; semi-hard mining (FaceNet) is stable, while always using the hardest negatives can amplify noise and collapse the embedding.` },
      { t: "qa", level: "intermediate", q: "Show that InfoNCE is a cross-entropy, and explain the role of the temperature.", a: String.raw`With logits $s_j/\tau$ for the positive and $K$ negatives, InfoNCE is $-\log\mathrm{softmax}(s/\tau)_+$ — cross-entropy with the positive as the correct class. Its gradient is $(q_+ - 1)/\tau$ for the positive and $q_k/\tau$ for each negative, where $q$ are the softmax weights. Small $\tau$ makes the softmax peaky, focusing the push on the hardest negatives (soft hard-negative mining) and scaling gradients by $1/\tau$; large $\tau$ spreads it evenly. Tune it: too small is unstable and sensitive to false negatives, too large is uninformative.` },
      { t: "qa", level: "intermediate", q: "Why multiply the distillation loss by T²?", a: String.raw`The gradient of $\mathrm{KL}(p^T\|q^T)$ with respect to the student logits is $(q^T - p^T)/T$, and for large $T$ the difference $q^T - p^T$ itself shrinks roughly like $1/T$, so gradients scale like $1/T^2$. Multiplying by $T^2$ keeps the soft-target term's contribution roughly independent of $T$, so its weight relative to the hard-label cross-entropy stays put when $T$ is tuned.` },
      { t: "qa", level: "intermediate", q: "Why does the IoU loss fail for non-overlapping boxes, and how does GIoU fix it?", a: String.raw`Without overlap, IoU is 0 wherever the prediction is, so $1 - \mathrm{IoU}$ is constant and its gradient is zero: the box gets no signal. GIoU subtracts $|C\setminus(B\cup G)|/|C|$, the empty fraction of the smallest enclosing box, which grows as the boxes move apart; so its gradient points toward the target even without overlap. GIoU $\le$ IoU, with equality when the union fills its enclosing box.` },
      { t: "qa", level: "advanced", q: "Derive the gradient of the soft Dice loss and explain how it handles class imbalance.", a: String.raw`With $I = \sum p_ig_i$ and $S = \sum p_i + \sum g_i + \varepsilon$, $\mathcal{L} = 1 - (2I + \varepsilon)/S$ and $\partial\mathcal{L}/\partial p_i = -(2g_iS - 2I - \varepsilon)/S^2$: one negative constant for every foreground pixel, the positive constant $(2I + \varepsilon)/S^2$ for every background pixel. Both scale with the object's size, not the image's, so the many background pixels cannot dominate: adding background pixels with $p \approx 0$ changes nothing. The cost: gradients are global (all pixels of a class get the same value) and noisy when the object is tiny or absent, so Dice is usually combined with cross-entropy.` },
      { t: "qa", level: "advanced", q: "Why can hardest-negative triplet mining collapse an embedding, and what are the remedies?", a: String.raw`If for many anchors the hardest negative is closer than the (random or hardest) positive, the loss $d_{ap} - d_{an} + m$ exceeds $m$. Shrinking all embeddings toward one point drives every distance to 0 and the loss to exactly $m$ — lower than now — so gradient descent can slide into the collapsed solution $f(x) = $ const, a stationary point where gradients are undefined or uninformative. Hardest negatives are also disproportionately mislabelled. Remedies: semi-hard mining, warm-up with random negatives, larger well-structured P×K batches, L2 normalisation (stops shrinking to 0, not to one direction), soft weighting such as InfoNCE, or margin-based softmax losses.` },
      { t: "qa", level: "advanced", q: "When would you choose forward KL and when reverse KL as a training objective?", a: String.raw`Forward $\mathrm{KL}(p\,\|\,q_\theta)$ (equivalent to maximum likelihood / cross-entropy with targets $p$) is used when samples or probabilities of $p$ are available, as in distillation. It is mass-covering: $q$ must put probability wherever $p$ does, and averages modes it cannot represent. Reverse $\mathrm{KL}(q_\theta\,\|\,p)$ needs only an unnormalised $p$ and samples from $q$, as in variational inference (the VAE's ELBO contains a reverse KL to the posterior). It is mode-seeking: $q$ fits one mode well and ignores others. The choice decides whether the model over-generalises (forward) or under-covers (reverse).` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Triplet loss (Euclidean) with d(a,p) = 1.0, d(a,n) = 1.2 and margin 0.5?", answer: 0.3, tolerance: 0.001, explain: "max(0, 1.0 − 1.2 + 0.5) = 0.3: the negative is semi-hard (farther than the positive but inside the margin)." },
          { kind: "numeric", q: "Contrastive loss with margin m = 1 for a dissimilar pair at distance D = 0.3?", answer: 0.245, tolerance: 0.001, explain: "½·max(0, 1 − 0.3)² = ½·0.49 = 0.245." },
          { kind: "numeric", q: "Predicted probabilities p = (0.9, 0.6, 0.2, 0.1), ground truth g = (1, 1, 0, 0). What is the soft Dice coefficient (ε = 0)?", answer: 0.7895, tolerance: 0.001, explain: "2·(0.9 + 0.6)/(1.8 + 2) = 3/3.8 = 0.7895; the Dice loss is 0.2105." },
          { kind: "numeric", q: "Two hard masks have IoU 0.5. What is their Dice coefficient?", answer: 0.6667, tolerance: 0.001, explain: "Dice = 2J/(1 + J) = 1/1.5 = 0.6667." },
          { kind: "numeric", q: "Boxes B = [1, 1, 3, 3] and G = [0, 0, 2, 2]. What is GIoU(B, G)?", answer: -0.0794, tolerance: 0.001, explain: "IoU = 1/7; the enclosing box [0,0,3,3] has area 9 and the union 7, so GIoU = 1/7 − 2/9 = −0.0794." },
          { kind: "numeric", q: "NT-Xent with N = 256 images (512 embeddings). What is the loss of a network that gives every candidate the same similarity?", answer: 6.2364, tolerance: 0.001, explain: "Each softmax runs over 2N − 1 = 511 candidates; uniform probabilities give −log(1/511) = log 511 = 6.2364." },
          { kind: "numeric", q: "VAE KL term for a single latent with μ = 0.5 and σ² = 1?", answer: 0.125, tolerance: 0.001, explain: "½(μ² + σ² − 1 − log σ²) = ½(0.25 + 1 − 1 − 0) = 0.125." },
          { q: "Lowering the InfoNCE temperature τ from 1 to 0.1 mainly…", options: ["makes all negatives contribute equally", "concentrates the gradient on the most similar negatives", "removes the need for negatives", "turns the loss into the triplet loss"], answer: 1, explain: "The softmax becomes peaky: in the lesson's example the hardest negative's share of the push rose from 47% to 98%." },
          { q: "Why can the IoU loss not train a box that does not overlap its target?", options: ["IoU is negative there", "IoU is constant (0), so the gradient is zero", "IoU is undefined for non-overlapping boxes", "Because IoU is not scale-invariant"], answer: 1, explain: "Moving a non-overlapping box changes nothing in IoU; GIoU adds an enclosing-box term that does change." },
          { q: "A segmentation model predicts background everywhere on images where 2% of pixels are foreground. Which statement is true?", options: ["Pixel accuracy and BCE look good while Dice is near 0", "Dice is about 0.98", "BCE is maximal", "All three metrics agree the model is bad"], answer: 0, explain: "Pixel-averaged metrics are dominated by the background; Dice measures overlap with the foreground and is ≈ 0." },
        ],
      },
      {
        t: "exercise",
        title: "Triplet gradients and GIoU by hand",
        prompt: "(a) For a = (0, 0), p = (0, 2), n = (1, 0) and margin m = 0.5 (Euclidean triplet loss), classify the negative (easy / semi-hard / hard), compute the loss and the gradients with respect to a, p and n. (b) Boxes G = [0, 0, 4, 4] and B = [2, 2, 6, 6]: compute IoU, GIoU and both losses.",
        hint: "u = (a − p)/d(a,p) and v = (a − n)/d(a,n); ∂L/∂a = u − v, ∂L/∂p = −u, ∂L/∂n = v.",
        solution: [
          { t: "p", text: String.raw`(a) $d(a,p) = 2$, $d(a,n) = 1 < 2$: a **hard** negative (wrong order). $\mathcal{L} = 2 - 1 + 0.5 = 1.5$. $u = (0, -2)/2 = (0, -1)$, $v = (-1, 0)/1 = (-1, 0)$. Gradients: $\partial\mathcal{L}/\partial a = u - v = (1, -1)$, $\partial\mathcal{L}/\partial p = (0, 1)$, $\partial\mathcal{L}/\partial n = (-1, 0)$. A descent step moves $a$ toward $(-1, +1)$ (up toward $p$, left away from $n$), $p$ down toward $a$, and $n$ right, away from $a$.` },
          { t: "p", text: String.raw`(b) Intersection $[2,2,4,4]$, area 4; each box has area 16, so the union is $16 + 16 - 4 = 28$ and IoU $= 4/28 = 0.142857$. The enclosing box $[0,0,6,6]$ has area 36 and empty part $36 - 28 = 8$, so GIoU $= 0.142857 - 8/36 = 0.142857 - 0.222222 = -0.079365$. IoU loss $0.857143$, GIoU loss $1.079365$ — the same as the quiz example, because this is that example scaled by 2: both measures are scale-invariant.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: count easy, semi-hard and hard triplets in a batch",
        prompt: "Write a vectorised NumPy function that, for a P×K batch of embeddings, counts the easy, semi-hard and hard (anchor, positive, negative) triplets for a margin m and returns all the batch-all triplet losses. Check it against a triple loop, then generate embeddings around 4 class centres with shrinking spread (from \"untrained\" to \"well trained\") and report how the counts and the mean loss change.",
        hint: "Build the B×B distance matrix; broadcast d_ap as D[:, :, None] and d_an as D[:, None, :] to get a B×B×B tensor indexed (a, p, n), and mask invalid triplets.",
        solution: [
          { t: "code", lang: "python", title: "count_triplets.py", code: EXERCISE, output: EXERCISE_OUT },
          { t: "p", text: String.raw`The vectorised counts match the loop exactly (the assert passes). There are $24\times5\times18 = 2160$ valid triplets (24 anchors, 5 positives, 18 negatives each). As the classes tighten, hard triplets fall from 472 to 8 and easy ones rise to 97%. The mean over **all** triplets (0.014 at the end) is mostly zeros; over non-zero triplets it is 0.509 — which is why batch-all code usually averages only over active triplets.` },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`**Metric learning** trains an embedding $f_\theta(x)$ where distance encodes similarity; it supports verification, retrieval and clustering of classes never seen in training. Evaluate on held-out classes.`,
          String.raw`**Contrastive** (Hadsell et al., 2006): $(1 - Y)\tfrac12D^2 + Y\tfrac12\max(0, m - D)^2$; similar pairs are pulled with force $D$, dissimilar pairs pushed with force $m - D$ only inside the margin.`,
          String.raw`**Triplet** (FaceNet, 2015): $[\,d(a,p) - d(a,n) + m\,]_+$ (squared distances on the unit sphere in FaceNet); gradients $u - v$, $-u$, $v$ with unit vectors $u, v$. Negatives are easy / semi-hard / hard; mine semi-hard or batch-hard; hardest-only can collapse.`,
          String.raw`**InfoNCE / NT-Xent**: softmax cross-entropy over similarities $s/\tau$; gradient $q_k/\tau$ per negative = automatic soft hard-negative mining; chance loss $\log(2N - 1)$; temperature must be tuned.`,
          String.raw`**KL divergence**: $\mathrm{KL}(p\|q) = H(p, q) - H(p)$; distillation $T^2\mathrm{KL}(p^T\|q^T)$ with gradient $T(q^T - p^T)$; VAE term $\tfrac12\sum(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$. Mind the direction.`,
          String.raw`**Soft Dice** $1 - (2\sum p_ig_i + \varepsilon)/(\sum p_i + \sum g_i + \varepsilon)$ and **soft IoU**: overlap losses insensitive to background size; Dice $= 2J/(1 + J)$. Combine with cross-entropy for stable gradients.`,
          String.raw`**Boxes**: IoU loss has zero gradient without overlap; **GIoU** $= \mathrm{IoU} - |C\setminus(B\cup G)|/|C|$ fixes it (DIoU/CIoU converge faster).`,
          "**Reconstruction**: MSE (Gaussian), BCE (Bernoulli, for [0,1] data; minimum is the entropy, not 0), L1, perceptual losses.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Self-Supervised Learning](/learn/m15/self-supervised) — InfoNCE in context: augmentations, SimCLR, momentum encoders and non-contrastive methods.",
          "[Semantic Segmentation](/learn/m12/segmentation) — the architectures (U-Net and successors) that the Dice + cross-entropy loss trains.",
          "[Object Detection](/learn/m12/object-detection) — where box losses (smooth L1, IoU, GIoU) sit inside a detector.",
          "[Autoencoders](/learn/m14/autoencoders) and [Variational Autoencoders](/learn/m14/vae) — reconstruction losses and the KL term in full models.",
          "[Gradient Descent](/learn/m06/gradient-descent) — how the optimiser uses these gradients; batch size and learning rate interact strongly with mining and temperature.",
        ],
      },
    ],
  },
};

export default lesson;
