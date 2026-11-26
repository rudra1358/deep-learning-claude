import type { Lesson } from "@/lib/types";

const SCRATCH_CODE = String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------- 1. Vector norms written out by hand ----------------
def lp_norm(x, p):
    """||x||_p = (sum |x_i|^p)^(1/p) for p >= 1; p = inf gives max |x_i|; p = 0 counts non-zeros."""
    x = np.asarray(x, dtype=float)
    if p == 0:
        return float(np.count_nonzero(x))          # not a true norm (fails homogeneity)
    if p == np.inf:
        return float(np.max(np.abs(x)))
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))

x = np.array([3.0, -4.0, 0.0, 12.0])
for p in [0, 1, 2, 3, 10, np.inf]:
    print(f"||x||_{p:<4} = {lp_norm(x, p):8.4f}")
print("matches np.linalg.norm (p=1,2,inf):",
      all(np.isclose(lp_norm(x, p), np.linalg.norm(x, p)) for p in [1, 2, np.inf]))

# ---------------- 2. Rank and linear independence by Gaussian elimination ----------------
def rank(A, tol=1e-10):
    """Row-reduce a copy of A and count the pivots."""
    M = np.array(A, dtype=float)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = r + np.argmax(np.abs(M[r:, c])) if r < rows else None
        if pivot is None or abs(M[pivot, c]) < tol:
            continue                                # no pivot in this column
        M[[r, pivot]] = M[[pivot, r]]               # swap the pivot row up
        M[r + 1:] -= np.outer(M[r + 1:, c] / M[r, c], M[r])   # eliminate below
        r += 1
        if r == rows:
            break
    return r

V = np.array([[1, 0, 2],
              [0, 1, 1],
              [2, 3, 7]], dtype=float).T             # columns v1, v2, v3
print("rank of [v1 v2 v3]:", rank(V), "| numpy:", np.linalg.matrix_rank(V))
print("v3 - (2 v1 + 3 v2) =", V[:, 2] - (2 * V[:, 0] + 3 * V[:, 1]))

# ---------------- 3. Projection onto a line and onto a subspace ----------------
a = np.array([1.0, 1.0, 0.0])
b = np.array([2.0, 3.0, 4.0])
proj_line = (a @ b) / (a @ a) * a
print("proj of b onto a:", proj_line, "| residual . a =", (b - proj_line) @ a)

A = V[:, :2]                                        # basis of the plane span{v1, v2}
coef = np.linalg.solve(A.T @ A, A.T @ b)            # normal equations  A^T A c = A^T b
p_plane = A @ coef
print("proj of b onto span{v1,v2}:", p_plane, "| residual . columns:", A.T @ (b - p_plane))

# ---------------- 4. Gram-Schmidt ----------------
def gram_schmidt(vectors, tol=1e-10):
    """Return an orthonormal basis (as columns) for the span of the given columns."""
    Q = []
    for v in vectors.T:
        w = v.astype(float).copy()
        for q in Q:                                 # modified Gram-Schmidt: subtract one
            w -= (q @ w) * q                        # projection at a time, using updated w
        n = np.linalg.norm(w)
        if n > tol:                                 # skip dependent vectors
            Q.append(w / n)
    return np.array(Q).T

B = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float).T            # columns (1,1,0), (1,0,1), (0,1,1)
Q = gram_schmidt(B)
print("Q =\n", Q)
print("Q^T Q == I:", np.allclose(Q.T @ Q, np.eye(3)))
print("dependent set keeps", gram_schmidt(V).shape[1], "vectors")

# ---------------- 5. Matrix norms: Frobenius and spectral (power iteration) ----------------
W = np.array([[3.0, 0.0],
              [4.0, 5.0]])
fro = np.sqrt(np.sum(W ** 2))
v = np.array([1.0, 0.0])
for _ in range(50):                                 # power iteration on W^T W
    v = W.T @ (W @ v)
    v /= np.linalg.norm(v)
spec = np.linalg.norm(W @ v)
print(f"Frobenius {fro:.4f} (numpy {np.linalg.norm(W, 'fro'):.4f}) | "
      f"spectral {spec:.4f} (numpy {np.linalg.norm(W, 2):.4f})")

# ---------------- 6. Cosine vs Euclidean on count vectors ----------------
docs = {"A: 'cat sat'      ": np.array([1.0, 1.0, 0.0]),
        "B: A repeated x10 ": np.array([10.0, 10.0, 0.0]),
        "C: 'cat ran'      ": np.array([1.0, 0.0, 1.0])}
q = docs["A: 'cat sat'      "]
for name, d in docs.items():
    cos = q @ d / (np.linalg.norm(q) * np.linalg.norm(d))
    print(f"{name} euclid {np.linalg.norm(q - d):6.3f}   cosine sim {cos:.3f}")

# ---------------- 7. Distance concentration in high dimensions ----------------
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000]:
    pts = rng.random((1000, d))                     # 1000 uniform points in [0,1]^d
    query = rng.random(d)
    dist = np.linalg.norm(pts - query, axis=1)
    print(f"d={d:5d}  min {dist.min():7.3f}  max {dist.max():7.3f}  (max-min)/min = {(dist.max() - dist.min()) / dist.min():7.3f}")

# ---------------- 8. Clipping a gradient by its L2 norm ----------------
g = np.array([3.0, 4.0, 12.0])
max_norm = 1.0
g_clipped = g * min(1.0, max_norm / np.linalg.norm(g))
print("grad norm", np.linalg.norm(g), "-> clipped", g_clipped, "norm", round(np.linalg.norm(g_clipped), 6))

# ---------------- 9. L1 gives exact zeros, L2 only shrinks ----------------
n, d = 100, 20
X = rng.normal(size=(n, d))
w_true = np.zeros(d); w_true[[0, 5, 12]] = [3.0, -2.0, 1.5]   # only 3 features matter
y = X @ w_true + 0.5 * rng.normal(size=n)

lam = 0.1
w_ridge = np.linalg.solve(X.T @ X / n + lam * np.eye(d), X.T @ y / n)   # min MSE/2 + lam/2 ||w||^2

def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

step = n / np.linalg.norm(X, 2) ** 2               # 1 / Lipschitz constant of the MSE/2 gradient
w_lasso = np.zeros(d)
for _ in range(2000):                               # ISTA: gradient step, then soft-threshold
    grad = X.T @ (X @ w_lasso - y) / n
    w_lasso = soft_threshold(w_lasso - step * grad, step * lam)

print("exact zeros  ridge:", int(np.sum(w_ridge == 0)), " lasso:", int(np.sum(w_lasso == 0)), "of", d)
print("lasso non-zero indices:", np.flatnonzero(w_lasso), "values:", w_lasso[w_lasso != 0])
print("smallest |ridge weight|:", f"{np.min(np.abs(w_ridge)):.4f}")`;

const SCRATCH_OUTPUT = String.raw`||x||_0    =   3.0000
||x||_1    =  19.0000
||x||_2    =  13.0000
||x||_3    =  12.2071
||x||_10   =  12.0000
||x||_inf  =  12.0000
matches np.linalg.norm (p=1,2,inf): True
rank of [v1 v2 v3]: 2 | numpy: 2
v3 - (2 v1 + 3 v2) = [0. 0. 0.]
proj of b onto a: [2.5 2.5 0. ] | residual . a = 0.0
proj of b onto span{v1,v2}: [1.  2.5 4.5] | residual . columns: [0. 0.]
Q =
 [[ 0.7071  0.4082 -0.5774]
 [ 0.7071 -0.4082  0.5774]
 [ 0.      0.8165  0.5774]]
Q^T Q == I: True
dependent set keeps 2 vectors
Frobenius 7.0711 (numpy 7.0711) | spectral 6.7082 (numpy 6.7082)
A: 'cat sat'       euclid  0.000   cosine sim 1.000
B: A repeated x10  euclid 12.728   cosine sim 1.000
C: 'cat ran'       euclid  1.414   cosine sim 0.500
d=    2  min   0.010  max   1.334  (max-min)/min = 134.618
d=   10  min   0.471  max   2.111  (max-min)/min =   3.484
d=  100  min   3.349  max   4.653  (max-min)/min =   0.389
d= 1000  min  12.166  max  13.553  (max-min)/min =   0.114
grad norm 13.0 -> clipped [0.2308 0.3077 0.9231] norm 1.0
exact zeros  ridge: 0  lasso: 15 of 20
lasso non-zero indices: [ 0  1  3  5 12] values: [ 2.9208  0.0231 -0.0011 -1.9129  1.386 ]
smallest |ridge weight|: 0.0012`;

const TORCH_CODE = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---- 1. Vector and matrix norms ----
x = torch.tensor([3.0, -4.0, 0.0, 12.0])
for p in [0, 1, 2, float("inf")]:
    print(f"vector_norm ord={p}: {torch.linalg.vector_norm(x, ord=p).item():.4f}")
W = torch.tensor([[3.0, 0.0], [4.0, 5.0]])
print("fro:", round(torch.linalg.matrix_norm(W, ord="fro").item(), 4),
      "| spectral:", round(torch.linalg.matrix_norm(W, ord=2).item(), 4))

# ---- 2. Embedding similarity: normalize once, then dot products are cosines ----
emb = torch.randn(5, 8)                               # 5 "embeddings" of dimension 8
emb[1] = 3.0 * emb[0] + 0.1 * torch.randn(8)          # item 1: same direction as 0, 3x longer
unit = F.normalize(emb, p=2, dim=1)                   # divide each row by its L2 norm
cos = unit @ unit.T                                   # (5, 5) matrix of cosine similarities
print("cos(0,1) via normalize:", round(cos[0, 1].item(), 4),
      "| F.cosine_similarity:", round(F.cosine_similarity(emb[0], emb[1], dim=0).item(), 4))
d = torch.cdist(emb[:1], emb)                         # Euclidean distances from item 0
print("nearest to item 0 by Euclidean:", d[0, 1:].argmin().item() + 1,
      "| by cosine:", cos[0, 1:].argmax().item() + 1)

# ---- 3. Gradient clipping by global L2 norm ----
model = nn.Linear(4, 1)
loss = (model(torch.randn(16, 4) * 50).squeeze() ** 2).mean()   # deliberately huge loss
loss.backward()
total = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # returns the norm BEFORE clipping
after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
print(f"grad norm before {total.item():.2f} -> after {after.item():.4f}")

# ---- 4. L1 penalty vs weight decay on a sparse regression problem ----
# ---- 4. L1 penalty vs weight decay on a sparse regression problem ----
X = torch.randn(200, 20)
w_true = torch.zeros(20); w_true[[0, 5, 12]] = torch.tensor([3.0, -2.0, 1.5])
y = X @ w_true + 0.5 * torch.randn(200)
LR, LAM = 0.05, 0.1

def fit(mode, steps=2000):
    torch.manual_seed(1)
    lin = nn.Linear(20, 1, bias=False)
    wd = LAM if mode == "weight decay" else 0.0
    opt = torch.optim.SGD(lin.parameters(), lr=LR, weight_decay=wd)  # adds wd * w to the gradient
    for _ in range(steps):
        loss = F.mse_loss(lin(X).squeeze(1), y)
        if mode == "L1 subgradient":
            loss = loss + LAM * lin.weight.abs().sum()     # autograd uses sign(w) as the gradient
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == "L1 proximal":                          # soft-threshold after each step
            with torch.no_grad():
                w = lin.weight
                w.copy_(torch.sign(w) * torch.clamp(w.abs() - LR * LAM, min=0.0))
    return lin.weight.detach().squeeze()

for mode in ["weight decay", "L1 subgradient", "L1 proximal"]:
    w = fit(mode)
    print(f"{mode:15s} exact zeros {(w == 0).sum().item():2d}/20 | |w|<1e-3 {(w.abs() < 1e-3).sum().item():2d}/20 | "
          f"w[0,5,12] = {[round(v, 3) for v in w[[0, 5, 12]].tolist()]}")`;

const TORCH_OUTPUT = String.raw`vector_norm ord=0: 3.0000
vector_norm ord=1: 19.0000
vector_norm ord=2: 13.0000
vector_norm ord=inf: 12.0000
fro: 7.0711 | spectral: 6.7082
cos(0,1) via normalize: 0.9995 | F.cosine_similarity: 0.9995
nearest to item 0 by Euclidean: 3 | by cosine: 1
grad norm before 2914.29 -> after 1.0000
weight decay    exact zeros  0/20 | |w|<1e-3  1/20 | w[0,5,12] = [2.883, -1.906, 1.371]
L1 subgradient  exact zeros  0/20 | |w|<1e-3  3/20 | w[0,5,12] = [2.98, -1.966, 1.406]
L1 proximal     exact zeros 14/20 | |w|<1e-3 14/20 | w[0,5,12] = [2.98, -1.968, 1.407]`;

const TF_CODE = String.raw`import tensorflow as tf

x = tf.constant([3.0, -4.0, 0.0, 12.0])
print(tf.norm(x, ord=1).numpy(), tf.norm(x, ord=2).numpy(), tf.norm(x, ord=float("inf")).numpy())

W = tf.constant([[3.0, 0.0], [4.0, 5.0]])
print(tf.norm(W, ord="fro", axis=[-2, -1]).numpy())       # Frobenius norm of a matrix
print(tf.linalg.svd(W, compute_uv=False)[0].numpy())      # spectral norm = largest singular value

emb = tf.random.normal((5, 8), seed=0)
unit = tf.math.l2_normalize(emb, axis=1)                  # rows scaled to length 1
cos = tf.matmul(unit, unit, transpose_b=True)             # pairwise cosine similarities
# Keras' cosine "loss" is the NEGATIVE cosine similarity (so that minimising it aligns vectors)
print(tf.keras.losses.cosine_similarity(emb[0:1], emb[1:2]).numpy(), -cos[0, 1].numpy())

# Gradient clipping by global norm
grads = [tf.constant([3.0, 4.0]), tf.constant([12.0])]
clipped, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
print(global_norm.numpy(), [g.numpy() for g in clipped])

# L1 / L2 penalties attached to a layer
layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L1(0.1))
layer2 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(0.1))`;

const EX_CODE = String.raw`import numpy as np

def top_k_cosine(queries, database, k=3):
    """Indices and scores of the k most cosine-similar database rows for each query."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)     # unit-normalize once
    d = database / np.linalg.norm(database, axis=1, keepdims=True)
    sims = q @ d.T                                                    # (n_q, n_db) cosines
    idx = np.argsort(-sims, axis=1)[:, :k]                            # largest first
    return idx, np.take_along_axis(sims, idx, axis=1)

rng = np.random.default_rng(0)
db = rng.normal(size=(10_000, 64))
db *= rng.uniform(0.5, 5.0, size=(10_000, 1))      # random lengths: magnitude carries no meaning here
queries = 0.1 * db[:2] + 0.03 * rng.normal(size=(2, 64))   # shrunken, slightly noisy copies of items 0 and 1

idx, scores = top_k_cosine(queries, db)
print("cosine top-3:", idx.tolist(), np.round(scores, 3).tolist())

# Euclidean on RAW vectors is fooled by length; on UNIT vectors it ranks exactly like cosine
raw = np.argsort(np.linalg.norm(db[None] - queries[:, None], axis=2), axis=1)[:, :3]
dn = db / np.linalg.norm(db, axis=1, keepdims=True)
qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
unit = np.argsort(np.linalg.norm(dn[None] - qn[:, None], axis=2), axis=1)[:, :3]
print("euclidean (raw) top-3:", raw.tolist())
print("euclidean (unit) top-3:", unit.tolist())
print("identity ||a-b||^2 = 2 - 2cos holds:",
      np.allclose(np.linalg.norm(qn[0] - dn[5]) ** 2, 2 - 2 * qn[0] @ dn[5]))`;

const EX_OUTPUT = String.raw`cosine top-3: [[0, 9496, 2562], [1, 3833, 5889]] [[0.988, 0.443, 0.421], [0.995, 0.438, 0.408]]
euclidean (raw) top-3: [[9755, 1314, 9193], [1314, 639, 9755]]
euclidean (unit) top-3: [[0, 9496, 2562], [1, 3833, 5889]]
identity ||a-b||^2 = 2 - 2cos holds: True`;

const lesson: Lesson = {
  id: "m02/norms-spaces",
  title: "Vector Spaces, Norms and Distances",
  summary:
    "Span, independence, basis, dimension and rank; inner products, projections and Gram–Schmidt; the L0/L1/L2/L∞/Lp, Frobenius and spectral norms and their unit balls; Euclidean, Manhattan and cosine distances in high dimensions; and why an L1 penalty gives exact zeros while L2 only shrinks.",
  minutes: 70,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices"],
  next: ["m02/eigen-svd", "m07/regularization", "m06/lr-schedules", "m13/word-embeddings"],
  terms: ["vector-space", "subspace", "span", "linear-independence", "basis", "rank", "inner-product", "norm", "l1-norm", "l2-norm", "frobenius-norm", "spectral-norm", "cosine-similarity", "projection", "gram-schmidt", "orthonormal"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`The [previous lesson](/learn/m02/vectors-matrices) introduced vectors and the dot product. This one asks three follow-up questions: **which vectors can a set of vectors build** (span, basis, dimension, rank), **how long is a vector** (norms), and **how far apart are two vectors** (distances and similarities). They sound basic, but almost every design choice about embeddings, regularization and training stability comes down to one of them.`,
      },
      {
        t: "p",
        text: String.raw`A **norm**, written with double bars $\|\mathbf{x}\|$, gives every vector a length of zero or more. There is more than one fair way to measure length: the straight-line length ($L_2$), the sum of absolute coordinates ($L_1$), or the largest coordinate ($L_\infty$, "L-infinity"). They behave differently enough to change what an algorithm prefers. An $L_1$ penalty on weights drives many to exactly zero; an $L_2$ penalty shrinks them all smoothly. Clipping gradients by their $L_2$ norm keeps the update's direction but caps its size; clipping each coordinate (an $L_\infty$ limit) changes the direction.`,
      },
      {
        t: "p",
        text: String.raw`A **distance** is the norm of the difference, $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$, and **cosine similarity** comes from the dot product. Search, recommendation, face verification and attention all boil down to "find the stored vectors closest to this one", so the distance you pick (and whether you normalise first) decides what counts as similar. High dimensions add a twist: for random data, all distances become nearly equal, so raw nearest-neighbour search there means less than it seems.`,
      },
      {
        t: "callout",
        kind: "key",
        title: "The ideas to keep",
        text: String.raw`A basis is a coordinate system; rank counts the independent directions a matrix can output. A norm is a choice of what "size" means, visible in the shape of its unit ball (all vectors of length ≤ 1): a diamond ($L_1$), a circle ($L_2$), a square ($L_\infty$). Cosine similarity compares directions and ignores length. After scaling vectors to length 1, Euclidean distance and cosine rank neighbours the same way, since $\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2\cos\theta$.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — the dot product $\mathbf{u}\cdot\mathbf{v} = \sum_i u_iv_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$, matrix-vector products as mixes of columns ($A\mathbf{x} = \sum_j x_j \mathbf{a}_j$), the transpose ($\top$), and the inverse of a $2\times 2$ matrix.`,
          String.raw`**Solving small linear systems** by elimination (subtract multiples of one equation from another). We use it to test linear independence and compute rank.`,
          String.raw`**Absolute value, square roots and the Pythagorean theorem.** The $L_2$ norm is Pythagoras in $n$ dimensions.`,
          String.raw`**Optional:** derivatives ([Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule)) help in the regularization part, where we compare the slope of $w^2$ (which fades near zero) with that of $|w|$ (which does not).`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Three ways to measure a trip across a city",
        text: "A café is 3 blocks east and 4 blocks north of you. A bird flies straight there: 5 blocks (Euclidean, L2). A taxi follows the grid: 3 + 4 = 7 blocks (Manhattan, L1). A chess king, moving one square per move in any direction including diagonally, needs max(3, 4) = 4 moves (Chebyshev, L∞). All three honestly answer 'how far?', each suited to some purpose. Now ignore distance and ask only which compass direction the café is in: that is cosine similarity. A café 30 east and 40 north lies in exactly the same direction — cosine 1 — though ten times farther.",
      },
      {
        t: "p",
        text: "Think of vectors as building blocks:",
      },
      {
        t: "list",
        items: [
          "A **vector space** is a collection you can add together and stretch by numbers without ever leaving it — arrows in the plane, lists of 768 numbers, even functions.",
          "The **span** of some vectors is everything you can build from them by stretching and adding. Two non-parallel arrows span the whole plane; two parallel arrows span only a line, because the second adds nothing new.",
          "Vectors that each add a truly new direction are **linearly independent**.",
          "An independent set that spans the whole space is a **basis** — a coordinate system. The number of vectors in any basis is the **dimension**.",
          "The **rank** of a matrix is the number of independent directions among its columns: how many dimensions its outputs can really fill.",
        ],
      },
      {
        t: "p",
        text: "A **projection** is a shadow: the closest point to a vector on a line or plane. The leftover (vector minus shadow) is perpendicular to that line or plane. **Gram–Schmidt** turns any basis into a perpendicular, unit-length (**orthonormal**) one by removing each new vector's shadow on the directions already chosen.",
      },
      {
        t: "p",
        text: "Why do norms matter for learning? Penalising the **L2** norm of the weights ties each weight to zero with a spring: big weights are pulled hard, small ones barely, so weights get small but rarely exactly zero. Penalising the **L1** norm is a constant pull toward zero: even a tiny weight feels the full force, so weights that do not earn their keep are pushed all the way to zero. That is the difference between 'shrinkage' and 'sparsity' (many exact zeros).",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "Computing the common vector norms" },
      {
        t: "steps",
        items: [
          { title: "Take absolute values", text: String.raw`For $\mathbf{x} = (3, -4, 0, 12)$: $(3, 4, 0, 12)$. Signs never matter to a norm.` },
          { title: "L0: count non-zeros", text: String.raw`Three entries are non-zero, so $\|\mathbf{x}\|_0 = 3$ (not a true norm — see the math section).` },
          { title: "L1: add the magnitudes", tex: String.raw`\|\mathbf{x}\|_1 = 3 + 4 + 0 + 12 = 19` },
          { title: "L2: square, add, square-root", tex: String.raw`\|\mathbf{x}\|_2 = \sqrt{9 + 16 + 0 + 144} = \sqrt{169} = 13` },
          { title: "L∞: take the largest magnitude", tex: String.raw`\|\mathbf{x}\|_\infty = \max(3, 4, 0, 12) = 12` },
          { title: "Lp in general: raise to p, add, take the p-th root", text: String.raw`The bigger $p$ is, the more the largest entry takes over: $\|\mathbf{x}\|_3 = 1819^{1/3} = 12.2071$, $\|\mathbf{x}\|_{10} = 12.0000$ (four decimals), heading to $\|\mathbf{x}\|_\infty = 12$.` },
        ],
      },
      { t: "h", text: "Testing independence and computing rank by elimination" },
      {
        t: "steps",
        items: [
          { title: "Put the vectors in the columns of a matrix", text: String.raw`$V = [\mathbf{v}_1\ \mathbf{v}_2\ \mathbf{v}_3]$. The vectors are independent only if the one way to get $V\mathbf{c} = \mathbf{0}$ is $\mathbf{c} = \mathbf{0}$ (all weights zero).` },
          { title: "Row-reduce", text: "Pick a pivot (the largest entry in the column, for safety), swap it to the top, subtract multiples of its row to zero everything below, move to the next column." },
          { title: "Count pivots", text: "A column with no usable pivot (all remaining entries ≈ 0) adds no new direction. The number of pivots is the rank." },
          { title: "Interpret", text: String.raw`Rank = number of vectors ⇒ independent. Smaller ⇒ dependent: some vector is a combination of the others. In floating point "zero" means "below a tolerance", so libraries compute rank from singular values instead (see [Eigenvalues and SVD](/learn/m02/eigen-svd)).` },
        ],
      },
      { t: "h", text: "Gram–Schmidt: building an orthonormal basis" },
      {
        t: "steps",
        items: [
          { title: "Normalise the first vector", tex: String.raw`\mathbf{q}_1 = \mathbf{v}_1 / \|\mathbf{v}_1\|` },
          { title: "Remove the next vector's shadow on every earlier q", text: "Take the next vector and subtract its shadow (projection) on each $\\mathbf{q}_j$ found so far. What is left points in a new, perpendicular direction.", tex: String.raw`\mathbf{w}_k = \mathbf{v}_k - \sum_{j<k} (\mathbf{q}_j^\top \mathbf{v}_k)\,\mathbf{q}_j` },
          { title: "Normalise (or discard)", text: String.raw`If $\|\mathbf{w}_k\| > 0$, set $\mathbf{q}_k = \mathbf{w}_k / \|\mathbf{w}_k\|$. If $\mathbf{w}_k \approx \mathbf{0}$, $\mathbf{v}_k$ depended on earlier vectors; skip it.` },
          { title: "Repeat for every vector", text: String.raw`The $\mathbf{q}$'s are orthonormal and span the same space as the $\mathbf{v}$'s. As columns they form $Q$ with $Q^\top Q = I$: the QR factorisation $V = QR$.` },
        ],
      },
      { t: "h", text: "Nearest-neighbour search with cosine similarity" },
      {
        t: "steps",
        items: [
          { title: "Normalise every stored vector once", text: "Divide each database row by its L2 norm." },
          { title: "Normalise the query", text: "Same operation on the query vector." },
          { title: "One matrix-vector product", text: "Unit vectors' dot product is their cosine, so `database_unit @ query_unit` gives every similarity at once." },
          { title: "Take the top-k", text: "Sort the scores. Approximate nearest-neighbour libraries use indexes instead of scanning everything, but the score is the same." },
        ],
      },
      { t: "h", text: "Gradient clipping by norm" },
      {
        t: "p",
        text: String.raw`Before an optimizer step, compute the $L_2$ norm of all gradients together, $\|\mathbf{g}\|_2$. If it exceeds a threshold $c$, rescale every gradient by $c/\|\mathbf{g}\|_2$. Direction stays; only step length is capped. For $\mathbf{g} = (3, 4, 12)$ and $c = 1$: $\|\mathbf{g}\|_2 = 13$, so $\mathbf{g} \leftarrow \mathbf{g}/13 = (0.2308, 0.3077, 0.9231)$. More in [Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Vector spaces and subspaces" },
      {
        t: "p",
        text: String.raw`A **(real) vector space** is a set $\mathcal{V}$ where you can add vectors ($\mathbf{u} + \mathbf{v}$) and scale them by real numbers ($\alpha\mathbf{v}$, with $\alpha \in \mathbb{R}$, "$\alpha$ is a real number") and always stay inside $\mathcal{V}$. The usual arithmetic rules hold (there is a zero vector $\mathbf{0}$, every $\mathbf{v}$ has a negative $-\mathbf{v}$, $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$, and so on). $\mathbb{R}^n$ is the example that matters here; functions and fixed-shape matrices are vector spaces too.`,
      },
      {
        t: "p",
        text: String.raw`A **subspace** is a part $\mathcal{S} \subseteq \mathcal{V}$ ("$\mathcal{S}$ sits inside $\mathcal{V}$") that is itself a vector space: it contains $\mathbf{0}$, and adding or scaling never leads out of it. In $\mathbb{R}^3$ the subspaces are $\{\mathbf{0}\}$, lines through the origin, planes through the origin, and $\mathbb{R}^3$ itself. A line that misses the origin is **not** a subspace — which is why a layer $W\mathbf{x} + \mathbf{b}$ with $\mathbf{b} \neq \mathbf{0}$ is called *affine* (linear plus a shift), not linear.`,
      },
      { t: "h", text: "Span, linear independence, basis, dimension" },
      {
        t: "math",
        tex: String.raw`\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_k\} = \Big\{\textstyle\sum_{i=1}^k c_i\mathbf{v}_i \;:\; c_i \in \mathbb{R}\Big\}, \qquad \text{independent} \iff \Big(\textstyle\sum_i c_i\mathbf{v}_i = \mathbf{0} \Rightarrow c_1 = \dots = c_k = 0\Big)`,
      },
      {
        t: "p",
        text: String.raw`In words: the span is every **linear combination** $\sum c_i\mathbf{v}_i$ (each $c_i$ any real number); it is always a subspace. Vectors are **independent** if the only way to combine them into zero is all $c_i = 0$ ($\iff$ means "exactly when", $\Rightarrow$ "implies"). Equivalently, none lies in the span of the others. A **basis** of $\mathcal{V}$ is an independent set that spans it; every vector then has **unique** coordinates in it. All bases of a space have the same size, the **dimension** $\dim\mathcal{V}$. In $\mathbb{R}^n$, any $n+1$ vectors are dependent and any $n$ independent vectors form a basis.`,
      },
      { t: "h", text: "Rank and the four subspaces of a matrix" },
      {
        t: "p",
        text: String.raw`For $A \in \mathbb{R}^{m\times n}$ ($m$ rows, $n$ columns), the **column space** $\mathcal{C}(A) = \{A\mathbf{x}\}$ is the span of the columns — every possible output. The **null space** $\mathcal{N}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ (read ":" as "such that") is every input sent to zero. The **rank** is $\operatorname{rank}A = \dim\mathcal{C}(A)$, which always equals the row space's dimension (column rank = row rank). The **rank–nullity theorem** links them:`,
      },
      { t: "math", tex: String.raw`\operatorname{rank}(A) + \dim\mathcal{N}(A) = n, \qquad \operatorname{rank}(A) \le \min(m, n), \qquad \operatorname{rank}(AB) \le \min(\operatorname{rank}A, \operatorname{rank}B)` },
      {
        t: "p",
        text: String.raw`In plain words: every input direction either survives into the output (counted by the rank) or is destroyed (counted by the null space's dimension, the **nullity**). The last inequality explains **bottlenecks**: a network $W_2W_1$ with hidden width $r$ can represent at most a rank-$r$ linear map — the idea behind autoencoders and low-rank adapters.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $\mathbf{v}_1 = (1, 0, 2)$, $\mathbf{v}_2 = (0, 1, 1)$, $\mathbf{v}_3 = (2, 3, 7)$. Since $2\mathbf{v}_1 + 3\mathbf{v}_2 = (2, 3, 4 + 3) = (2, 3, 7) = \mathbf{v}_3$, the set is dependent; $\mathbf{v}_1, \mathbf{v}_2$ are independent (neither is a multiple of the other), so $\operatorname{rank}[\mathbf{v}_1\ \mathbf{v}_2\ \mathbf{v}_3] = 2$. The three vectors span only a plane in $\mathbb{R}^3$, and the null space is the line spanned by $(2, 3, -1)$, so $2 + 1 = 3 = n$ as rank–nullity requires.`,
      },
      { t: "h", text: "Inner products" },
      {
        t: "p",
        text: String.raw`An **inner product** $\langle\mathbf{u}, \mathbf{v}\rangle$ is a generalised dot product: it takes two vectors, gives a number, doesn't care about order, is linear in each input, and gives a positive value for any non-zero vector paired with itself. The dot product $\mathbf{u}^\top\mathbf{v}$ is the standard one; $\mathbf{u}^\top M\mathbf{v}$ for a symmetric positive-definite matrix $M$ is another (behind the Mahalanobis distance). Every inner product obeys the **Cauchy–Schwarz inequality**`,
      },
      { t: "math", tex: String.raw`|\langle\mathbf{u},\mathbf{v}\rangle| \le \|\mathbf{u}\|\,\|\mathbf{v}\|, \qquad \|\mathbf{v}\| = \sqrt{\langle\mathbf{v},\mathbf{v}\rangle}`, caption: "Equality holds exactly when one vector is a multiple of the other. This is why cosine similarity lies in [−1, 1]." },
      { t: "h", text: "Norms" },
      {
        t: "p",
        text: String.raw`A **norm** $\|\cdot\|$ turns each vector into a number in $[0, \infty)$ (zero or more) and obeys three rules for all vectors $\mathbf{x}, \mathbf{y}$ and numbers $\alpha$: only the zero vector has length 0; scaling a vector scales its length; and a detour is never shorter than the direct path.`,
      },
      {
        t: "math",
        tex: String.raw`\underbrace{\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}}_{\text{definiteness}}, \qquad \underbrace{\|\alpha\mathbf{x}\| = |\alpha|\,\|\mathbf{x}\|}_{\text{homogeneity}}, \qquad \underbrace{\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|}_{\text{triangle inequality}}`,
      },
      { t: "math", tex: String.raw`\|\mathbf{x}\|_p = \Big(\sum_{i=1}^n |x_i|^p\Big)^{1/p}\ (p \ge 1), \qquad \|\mathbf{x}\|_\infty = \max_i |x_i| = \lim_{p\to\infty}\|\mathbf{x}\|_p, \qquad \|\mathbf{x}\|_0 = \#\{i : x_i \ne 0\}` },
      {
        t: "p",
        text: String.raw`Read the $L_p$ formula as: raise each $|x_i|$ to the power $p$, add, take the $p$-th root. $p = 1$ gives the **L1 (Manhattan)** norm, $p = 2$ the **L2 (Euclidean)** norm. $\max_i$ means "largest over all $i$", $\lim_{p\to\infty}$ means "what $\|\mathbf{x}\|_p$ approaches as $p$ grows", and $\#$ means "count". The "**L0 norm**" counts non-zeros; it is *not* a norm because scaling breaks it: $\|2\mathbf{x}\|_0 = \|\mathbf{x}\|_0$, not $2\|\mathbf{x}\|_0$. For $0 < p < 1$ the formula also fails the triangle inequality. In $\mathbb{R}^n$ all norms agree up to constant factors; the most useful bounds are`,
      },
      { t: "math", tex: String.raw`\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\,\|\mathbf{x}\|_2 \le n\,\|\mathbf{x}\|_\infty` },
      {
        t: "p",
        text: String.raw`For $\mathbf{x} = (3, -4, 0, 12)$ ($n = 4$): $12 \le 13 \le 19 \le \sqrt{4}\cdot 13 = 26 \le 48$. The constants grow with $n$, which matters when $n$ is in the millions: a gradient whose every coordinate is small ($\|\mathbf{g}\|_\infty$ small) can still have a large $L_2$ norm.`,
      },
      { t: "h", text: "Unit balls: the geometry of a norm" },
      {
        t: "p",
        text: String.raw`The **unit ball** $\{\mathbf{x} : \|\mathbf{x}\| \le 1\}$ — all vectors of length at most 1 — shows a norm's character. In 2-D: $L_1$ gives a diamond with corners at $(\pm1, 0)$, $(0, \pm1)$; $L_2$ a circle; $L_\infty$ a square with corners $(\pm1, \pm1)$; other $L_p$ fall in between, squarer as $p$ grows. Every norm ball is convex (no dents) and symmetric. The $L_1$ ball's corners sit **on the axes**, where all but one coordinate is zero. That one geometric fact is the source of L1 sparsity.`,
      },
      { t: "h", text: "Matrix norms" },
      {
        t: "math",
        tex: String.raw`\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}, \qquad \|A\|_2 = \max_{\mathbf{x}\ne\mathbf{0}}\frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_{\max}(A)`,
      },
      {
        t: "p",
        text: String.raw`The **Frobenius norm** treats the matrix as one long vector ($\operatorname{tr}$, the trace, is the sum of the diagonal). The **spectral norm** is the most $A$ can stretch any vector; it equals the largest **singular value** $\sigma_{\max}$, the square root of the largest eigenvalue of $A^\top A$ (see [Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd)). It is **submultiplicative**, $\|AB\|_2 \le \|A\|_2\|B\|_2$, so a stack of $L$ layers stretches a signal (or a backpropagated gradient) by at most the product of their spectral norms — the link to [exploding and vanishing gradients](/learn/m04/gradient-problems). Always $\|A\|_2 \le \|A\|_F \le \sqrt{\operatorname{rank}A}\,\|A\|_2$.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $W = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}$: $\|W\|_F = \sqrt{9 + 0 + 16 + 25} = \sqrt{50} = 7.0711$. $W^\top W = \begin{pmatrix} 25 & 20 \\ 20 & 25 \end{pmatrix}$ has eigenvalues $45$ and $5$, so $\|W\|_2 = \sqrt{45} = 6.7082$. Check: $7.0711^2 = 50 = 45 + 5$ — the squared Frobenius norm is the sum of all squared singular values.`,
      },
      { t: "h", text: "Distances and similarities" },
      {
        t: "math",
        tex: String.raw`d_2(\mathbf{x},\mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2, \quad d_1(\mathbf{x},\mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1, \quad \cos(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|_2\|\mathbf{y}\|_2}, \quad d_{\cos} = 1 - \cos(\mathbf{x},\mathbf{y})`,
      },
      {
        t: "p",
        text: String.raw`Every norm gives a proper distance, or **metric** (never negative, zero only for equal points, same both ways, obeys the triangle inequality). **Cosine distance** is not a metric (it fails the triangle inequality) and ignores length. For unit vectors (marked with a hat, $\hat{\mathbf{a}}$) the two views agree:`,
      },
      { t: "math", tex: String.raw`\|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|_2^2 = \|\hat{\mathbf{a}}\|^2 + \|\hat{\mathbf{b}}\|^2 - 2\,\hat{\mathbf{a}}^\top\hat{\mathbf{b}} = 2 - 2\cos\theta \qquad (\|\hat{\mathbf{a}}\| = \|\hat{\mathbf{b}}\| = 1)`, caption: "Euclidean distance between normalised vectors is a decreasing function of cosine similarity, so both produce the same nearest-neighbour ranking." },
      {
        t: "p",
        text: String.raw`**When cosine beats Euclidean.** Use cosine when direction carries the meaning and length is noise: word counts of documents of different lengths, embeddings whose norm tracks frequency, or embeddings trained with a cosine-based loss. Example with word counts over (cat, sat, ran): $A = (1, 1, 0)$ "cat sat", $B = (10, 10, 0)$ the same sentence repeated ten times, $C = (1, 0, 1)$ "cat ran". Euclidean: $d(A, B) = \sqrt{81 + 81} = 12.728$ but $d(A, C) = \sqrt{0 + 1 + 1} = 1.414$, so $C$ looks closer. Cosine: $\cos(A, B) = 20/(\sqrt2\cdot\sqrt{200}) = 1$, $\cos(A, C) = 1/(\sqrt2\cdot\sqrt2) = 0.5$, so $B$ is the better match. Use Euclidean (or plain dot product) when length means something — e.g. the model was trained with an L2 or dot-product objective, or length encodes confidence or popularity.`,
      },
      { t: "h", text: "Distance concentration in high dimensions" },
      {
        t: "p",
        text: String.raw`Pick $\mathbf{x}, \mathbf{y}$ at random, independently and uniformly, in the cube $[0, 1]^d$ ($d$ coordinates, each between 0 and 1). Write $\mathbb{E}$ for the expected (average) value and $\operatorname{Var}$ for the variance (spread). Each coordinate gives $\mathbb{E}[(x_i - y_i)^2] = 1/6$ and $\operatorname{Var}[(x_i - y_i)^2] = 1/15 - 1/36 = 7/180$, so adding $d$ independent coordinates gives`,
      },
      { t: "math", tex: String.raw`\mathbb{E}\,\|\mathbf{x}-\mathbf{y}\|^2 = \frac{d}{6}, \qquad \frac{\operatorname{std}\,\|\mathbf{x}-\mathbf{y}\|^2}{\mathbb{E}\,\|\mathbf{x}-\mathbf{y}\|^2} = \frac{\sqrt{7d/180}}{d/6} = \frac{\sqrt{1.4}}{\sqrt d} \approx \frac{1.18}{\sqrt d}` },
      {
        t: "p",
        text: String.raw`In plain words: distances grow like $\sqrt{d/6}$, but their **relative spread shrinks like** $1/\sqrt d$, so the nearest and farthest points end up almost equally far. The program below measures it for 1000 random points: (max − min)/min falls from 134.6 at $d = 2$ to 3.48 at $d = 10$, 0.389 at $d = 100$ and 0.114 at $d = 1000$ (distances 12.17–13.55 around $\sqrt{1000/6} = 12.91$). Real embeddings lie near low-dimensional structure and escape the worst of it, so nearest-neighbour search works far better on **learned** representations than on raw features.`,
      },
      { t: "h", text: "Projections and orthogonality" },
      {
        t: "p",
        text: String.raw`Vectors are **orthogonal** if $\mathbf{u}^\top\mathbf{v} = 0$; a set is **orthonormal** if each also has norm 1. The **projection** of $\mathbf{b}$ onto the line through $\mathbf{a}$ is the point $\hat c\,\mathbf{a}$ (the best multiple $\hat c$) that minimises $\|\mathbf{b} - c\,\mathbf{a}\|^2$. Setting the derivative $-2\mathbf{a}^\top(\mathbf{b} - c\,\mathbf{a})$ to zero gives`,
      },
      { t: "math", tex: String.raw`\operatorname{proj}_{\mathbf{a}}\mathbf{b} = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a}, \qquad \mathbf{a}^\top(\mathbf{b} - \operatorname{proj}_{\mathbf{a}}\mathbf{b}) = 0` },
      {
        t: "p",
        text: String.raw`The leftover (residual) is orthogonal to $\mathbf{a}$ — the mark of the closest point. For a subspace spanned by the independent columns of $A$, the same argument (minimise $\|\mathbf{b} - A\mathbf{c}\|^2$, set the gradient $-2A^\top(\mathbf{b} - A\mathbf{c})$ to zero) gives the **normal equations** and the projection matrix $P$`,
      },
      { t: "math", tex: String.raw`A^\top A\,\hat{\mathbf{c}} = A^\top\mathbf{b}, \qquad \mathbf{p} = A\hat{\mathbf{c}} = \underbrace{A(A^\top A)^{-1}A^\top}_{P}\,\mathbf{b}, \qquad P^2 = P,\ P^\top = P` },
      {
        t: "p",
        text: String.raw`This is exactly least-squares linear regression: the fitted values are the projection of the targets onto the column space of the data matrix. With orthonormal columns ($A = Q$), $Q^\top Q = I$ and the formula shrinks to $P = QQ^\top$ — the practical reason to orthonormalise.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $\mathbf{b} = (2, 3, 4)$, $\mathbf{a} = (1, 1, 0)$: $\mathbf{a}^\top\mathbf{b} = 5$, $\mathbf{a}^\top\mathbf{a} = 2$, so $\operatorname{proj}_{\mathbf{a}}\mathbf{b} = 2.5\,(1, 1, 0) = (2.5, 2.5, 0)$ and the residual $(-0.5, 0.5, 4)$ has dot product $-0.5 + 0.5 + 0 = 0$ with $\mathbf{a}$. Onto the plane spanned by $\mathbf{v}_1 = (1, 0, 2)$, $\mathbf{v}_2 = (0, 1, 1)$: $A^\top A = \begin{pmatrix} 5 & 2 \\ 2 & 2\end{pmatrix}$, $A^\top\mathbf{b} = (10, 7)$, solving gives $\hat{\mathbf{c}} = (1, 2.5)$ and $\mathbf{p} = 1\cdot\mathbf{v}_1 + 2.5\cdot\mathbf{v}_2 = (1, 2.5, 4.5)$. The residual $(1, 0.5, -0.5)$ is orthogonal to both: $1 + 0 - 1 = 0$ and $0 + 0.5 - 0.5 = 0$.`,
      },
      { t: "h", text: "Gram–Schmidt, worked" },
      {
        t: "steps",
        items: [
          { title: "Input", tex: String.raw`\mathbf{v}_1 = (1, 1, 0),\quad \mathbf{v}_2 = (1, 0, 1),\quad \mathbf{v}_3 = (0, 1, 1)` },
          { title: "q₁", tex: String.raw`\mathbf{q}_1 = \tfrac{1}{\sqrt2}(1, 1, 0) = (0.7071,\ 0.7071,\ 0)` },
          { title: "w₂ and q₂", tex: String.raw`\mathbf{q}_1^\top\mathbf{v}_2 = 0.7071,\quad \mathbf{w}_2 = (1,0,1) - 0.7071\,\mathbf{q}_1 = (0.5,\ -0.5,\ 1),\quad \|\mathbf{w}_2\| = 1.2247,\quad \mathbf{q}_2 = (0.4082,\ -0.4082,\ 0.8165)` },
          { title: "w₃: remove the q₁ part", tex: String.raw`\mathbf{q}_1^\top\mathbf{v}_3 = 0.7071 \;\Rightarrow\; (0,1,1) - (0.5, 0.5, 0) = (-0.5,\ 0.5,\ 1)` },
          { title: "w₃: remove the q₂ part", tex: String.raw`\mathbf{q}_2^\top(-0.5, 0.5, 1) = 0.4082 \;\Rightarrow\; \mathbf{w}_3 = (-0.6667,\ 0.6667,\ 0.6667),\quad \|\mathbf{w}_3\| = 1.1547` },
          { title: "q₃", tex: String.raw`\mathbf{q}_3 = (-0.5774,\ 0.5774,\ 0.5774), \qquad Q^\top Q = I` },
        ],
      },
      {
        t: "p",
        text: String.raw`The program subtracts each projection from the *updated* vector (modified Gram–Schmidt): same result in exact arithmetic, far less damage from rounding. Libraries use Householder reflections (as in NumPy's linalg.qr) for full stability.`,
      },
      { t: "h", text: "Why L1 gives sparsity and L2 gives shrinkage" },
      {
        t: "p",
        text: String.raw`**Geometric argument.** Minimising loss plus penalty, $\min_\mathbf{w} \mathcal{L}(\mathbf{w}) + \lambda\|\mathbf{w}\|$ ($\lambda$, lambda, sets the penalty strength), is equivalent (for a matching budget $t$) to minimising $\mathcal{L}(\mathbf{w})$ while keeping $\|\mathbf{w}\| \le t$. The answer is where the loss's contour lines, growing outward, first touch the norm ball. An ellipse growing toward a **circle** touches it at a typical point, where both coordinates are non-zero. Toward a **diamond**, it very likely hits a **corner** — and corners lie on the axes, where coordinates are exactly zero. In higher dimensions the $L_1$ ball has $2n$ corners plus many sparse edges and faces, so the effect gets stronger.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Find the closest point to $\mathbf{a} = (2, 0.5)$ inside each unit ball (loss $\|\mathbf{w} - \mathbf{a}\|^2$). For $L_2$: scale $\mathbf{a}$ back onto the circle, $\mathbf{a}/\|\mathbf{a}\| = (2, 0.5)/2.0616 = (0.9701, 0.2425)$ — both non-zero. For $L_1$: the answer is $\operatorname{sign}(a_i)\max(|a_i| - \tau, 0)$ ($\operatorname{sign}$ is +1 or −1, the sign of the number), with $\tau$ chosen so the result has $L_1$ norm 1. With $\tau = 1$: $(2 - 1, \max(0.5 - 1, 0)) = (1, 0)$, $L_1$ norm 1. The second coordinate is **exactly zero**: a corner.`,
      },
      {
        t: "p",
        text: String.raw`**Algebra (one coordinate).** Minimise $\tfrac12(w - z)^2 + \lambda\,\Omega(w)$, where $z$ is what the data alone would pick and $\Omega$ (omega) is the penalty.`,
      },
      {
        t: "math",
        tex: String.raw`\Omega(w) = \tfrac12 w^2:\;\; w^\star = \frac{z}{1+\lambda} \qquad\qquad \Omega(w) = |w|:\;\; w^\star = \operatorname{sign}(z)\max(|z| - \lambda,\ 0)`,
        caption: "Left: L2 multiplies by a constant factor — never exactly zero unless z = 0. Right: L1 soft-thresholding — every |z| ≤ λ is set exactly to zero.",
      },
      {
        t: "p",
        text: String.raw`The reason is the slope near zero. The gradient of $\tfrac12\lambda w^2$ is $\lambda w$, which **fades** as $w \to 0$, so the pull weakens and never quite finishes. The slope (subgradient, since $|w|$ has a kink at 0) of $\lambda|w|$ is $\lambda\operatorname{sign}(w)$: a **constant** pull. At $w = 0$ any slope in $[-\lambda, \lambda]$ is allowed, so $w = 0$ is optimal whenever the data's pull $|z|$ is at most $\lambda$. In the program below, ISTA (gradient step + soft-threshold) on a 20-feature regression with 3 true non-zero weights sets 15 of 20 weights exactly to zero; ridge regression with the same $\lambda$ sets none (smallest weight 0.0012).`,
      },
      {
        t: "callout",
        kind: "note",
        title: "Weight decay is an L2 penalty — with one caveat",
        text: String.raw`For plain SGD, adding $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ to the loss equals shrinking weights a little each step: $\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta\nabla\mathcal{L}$ ($\eta$ = learning rate, $\nabla\mathcal{L}$ = loss gradient). For Adam the two differ, because Adam divides the penalty gradient by its per-parameter scale; AdamW applies the decay directly to the weights instead. See [Regularization](/learn/m07/regularization) and [Momentum and adaptive optimizers](/learn/m06/momentum-adaptive); L2 as a Gaussian prior and L1 as a Laplace prior are in [MLE and MAP](/learn/m02/mle-map).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator applies a 2×2 matrix to the plane. It is built for the next lesson, but shows several ideas from this one: column space and rank, the image of the L2 unit ball, the spectral norm, and projection.",
      },
      { t: "sim", id: "linear-transform", caption: "The black curve is the image of the L2 unit circle; the shaded parallelogram is the image of the unit square; drag on the plane to move the input vector x." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "**Span and rank.** Choose the *Projection (singular)* preset. Both columns point along the x-axis, so the plane collapses onto a line: column space 1-dimensional, rank 1, det 0. Drag x — Ax always lands on that line, and every x on the vertical axis maps to 0 (the null space; 1 + 1 = 2, as rank–nullity says).",
          String.raw`**Independence.** From *General*, drag the sliders until column 2, $(b, d)$, is a multiple of column 1, $(a, c)$ — e.g. $a = 1, c = 2, b = 2, d = 4$. The parallelogram flattens and det hits 0 exactly when the columns become dependent.`,
          String.raw`**The image of the L2 ball is an ellipse; its longest semi-axis is the spectral norm.** Switch to **SVD** mode and step to the end: the arrows $\sigma_1\mathbf{u}_1$ and $\sigma_2\mathbf{u}_2$ are the semi-axes. With the preset $[[3,0],[4,5]]$ the panel shows $\sigma_1 = 6.71 = \|W\|_2$ and $\sigma_2 = 2.24$; the worked example above gave $\sqrt{45}$ and $\sqrt5$.`,
          String.raw`**Spectral norm = maximum stretch.** Keep that preset and drag x around the unit circle ($\|\mathbf{x}\| = 1$). $A\mathbf{x}$ never gets longer than 6.71 or shorter than 2.24; the extremes are at x along $\mathbf{v}_1$ or $\mathbf{v}_2$.`,
          "**Orthogonal maps keep lengths.** Pick *Rotation 45°* or *Reflection*: the circle maps onto itself and |Ax| = |x|. These are the matrices Q with QᵀQ = I that Gram–Schmidt produces.",
        ],
      },
      {
        t: "callout",
        kind: "tip",
        title: "A thought experiment for the unit balls",
        text: String.raw`Draw the diamond $|w_1| + |w_2| \le 1$ and the circle $w_1^2 + w_2^2 \le 1$, then draw growing ellipses (loss contours) centred at $(2, 0.5)$ until they touch each ball. They touch the circle at $(0.97, 0.24)$ and the diamond at its corner $(1, 0)$ — the math section's example. Move the centre to $(1, 1)$: by symmetry the contour now touches the diamond along an edge, the one case where L1 does not pick a single coordinate.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "One program, nine experiments: norms by hand, rank by elimination, projections, modified Gram–Schmidt, Frobenius and spectral norms, cosine vs Euclidean, distance concentration, clipping by norm, and L1 sparsity vs L2 shrinkage. Every number in the math section comes from this output.",
      },
      { t: "code", lang: "python", title: "norms_spaces_from_scratch.py", code: SCRATCH_CODE, output: SCRATCH_OUTPUT },
      { t: "p", text: "Notes on the implementation:" },
      {
        t: "list",
        items: [
          "`rank` uses **partial pivoting** (largest remaining entry as pivot) and a tolerance of 1e-10; without it, rounding error would make a dependent column look independent.",
          "`gram_schmidt` skips vectors whose residual falls below the tolerance, so the dependent set `V` yields only 2 basis vectors.",
          "The spectral norm uses **power iteration**: repeatedly multiplying v by WᵀW and rescaling it to length 1 lines v up with the most-stretched direction, and |Wv| is then σ_max. PyTorch's `spectral_norm` uses the same idea.",
          "In the Lasso part, `step` is 1/L, where L = σ_max(X)²/n bounds how fast the half-MSE gradient can change; this step size keeps ISTA from diverging. The soft-threshold `step * lam` is the L1 proximal step.",
          "Lasso kept two small false weights (indices 1 and 3, sizes 0.0231 and 0.0011) besides the three true ones; a larger λ removes them but shrinks the true weights more. Ridge has no exact zeros.",
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
                text: "`torch.linalg.vector_norm` and `torch.linalg.matrix_norm` compute every norm here; `F.normalize` and `F.cosine_similarity` handle embeddings; `clip_grad_norm_` clips by global L2 norm and returns the pre-clipping norm (worth logging as a health signal). The last part compares weight decay, an L1 penalty via plain autograd (gradient sign(w), never exactly zero), and a proximal L1 step that soft-thresholds after each update.",
              },
              { t: "code", lang: "python", title: "norms_torch.py", code: TORCH_CODE, output: TORCH_OUTPUT, runnable: false },
              {
                t: "callout",
                kind: "warning",
                title: "L1 + SGD does not give exact zeros",
                text: "Autograd treats the slope of |w| as sign(w), so a weight near zero bounces across zero by ±lr·λ each step and never settles. You get tiny weights (3 below 1e-3 above), not exact zeros. Proximal updates (soft-thresholding, as in ISTA) or pruning small weights after training give true sparsity (14 exact zeros above).",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow's equivalents: `tf.norm`, `tf.math.l2_normalize`, `tf.clip_by_global_norm` and `tf.keras.regularizers.L1/L2`. Note that `tf.keras.losses.cosine_similarity` returns the **negative** cosine, so minimising it raises similarity.",
              },
              {
                t: "code",
                lang: "python",
                title: "norms_tf.py (not executed in CI)",
                code: TF_CODE,
                output: "Prints 19.0 13.0 12.0; the Frobenius norm 7.071; the spectral norm 6.708; the Keras cosine loss and the negated cosine from the normalised matrix (equal values); the global norm 13.0 and the clipped gradients [0.2308, 0.3077] and [0.9231].",
                runnable: false,
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Norms" },
      {
        t: "table",
        head: ["Norm", "Formula", "Unit ball (2-D)", "Advantages", "Disadvantages / use"],
        rows: [
          ["L0 (not a norm)", String.raw`$\#\{x_i \ne 0\}$`, "the two axes", "counts non-zeros, which is exactly sparsity", "jumpy, no useful gradient; minimising it means trying combinations (NP-hard). Used to *measure* sparsity and for pruning"],
          ["L1 (Manhattan)", String.raw`$\sum_i|x_i|$`, "diamond", "convex stand-in for L0; as a loss (MAE) it shrugs off outliers", "has a kink at 0; Lasso, L1 regularization, sparse coding"],
          ["L2 (Euclidean)", String.raw`$\sqrt{\sum_i x_i^2}$`, "circle", "smooth, same in every rotation, comes from the dot product", "squared, it is hit hard by outliers; weight decay, clipping, normalising embeddings"],
          ["L∞ (Chebyshev)", String.raw`$\max_i|x_i|$`, "square", "caps every coordinate", "looks only at the largest entry; adversarial-attack budgets, clip-by-value"],
          ["Lp, 1 < p < ∞", String.raw`$(\sum_i|x_i|^p)^{1/p}$`, "rounded square", "sits between L1 and L∞", "rarely used directly; Lp pooling, theory"],
          ["Frobenius", String.raw`$\sqrt{\sum_{ij}A_{ij}^2}$`, "—", "cheap; L2 of the matrix flattened into a vector", "misses the worst-case stretch; weight decay on matrices"],
          ["Spectral", String.raw`$\sigma_{\max}(A)$`, "—", "exactly the most a linear layer can stretch an input (its Lipschitz constant)", "needs SVD or power iteration; spectral normalization, stability checks"],
          ["Nuclear", String.raw`$\sum_i\sigma_i(A)$`, "—", "convex stand-in for rank", "needs an SVD; filling in low-rank matrices"],
        ],
      },
      { t: "h", text: "Distances and similarities" },
      {
        t: "table",
        head: ["Measure", "Definition", "Scale-invariant?", "Typical use"],
        rows: [
          ["Euclidean", String.raw`$\|\mathbf{x}-\mathbf{y}\|_2$`, "no", "k-means, k-NN on standardised features, embeddings trained with L2"],
          ["Squared Euclidean", String.raw`$\|\mathbf{x}-\mathbf{y}\|_2^2$`, "no", "MSE, k-means; same ranking, cheaper (no square root)"],
          ["Manhattan", String.raw`$\|\mathbf{x}-\mathbf{y}\|_1$`, "no", "grid-like features; less thrown by a few big coordinate gaps"],
          ["Cosine similarity", String.raw`$\mathbf{x}^\top\mathbf{y}/(\|\mathbf{x}\|\|\mathbf{y}\|)$`, "yes", "text and sentence embeddings, contrastive learning, search"],
          ["Dot product", String.raw`$\mathbf{x}^\top\mathbf{y}$`, "no", "attention scores, recommenders that look for the largest dot product"],
          ["Mahalanobis", String.raw`$\sqrt{(\mathbf{x}-\mathbf{y})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$`, "affine-invariant", "correlated features, spotting unusual (out-of-distribution) inputs"],
          ["Hamming", "number of differing positions", "—", "binary codes, hashing-based retrieval"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Semantic search and retrieval-augmented generation.** Documents and queries are embedded, usually scaled to length 1, and matched by cosine similarity. Libraries such as FAISS offer both L2 and inner-product indexes.",
          "**Image–text models.** CLIP normalises its image and text embeddings and trains on scaled cosine similarities, so zero-shot classification is a nearest-neighbour search by cosine.",
          "**Face verification.** Systems compare normalised face embeddings by cosine, often trained with angular-margin losses (e.g. ArcFace); a cosine threshold decides 'same person'.",
          String.raw`**Gradient clipping** by global $L_2$ norm is standard for RNNs and Transformers, where occasional huge gradients would destabilise training.`,
          String.raw`**Weight decay** (an $L_2$ penalty, or decoupled decay in AdamW) appears in nearly every large training recipe.`,
          String.raw`**Spectral normalization** (Miyato et al., 2018) divides each weight matrix by its spectral norm, bounding the Lipschitz constant of a GAN discriminator.`,
          String.raw`**Sparse models.** L1 regularization (the Lasso) selects features in linear models; pruning removes a network's small weights — a rough route to an $L_0$ goal.`,
          String.raw`**Least squares.** Linear regression is a projection, solved reliably with QR factorisation (Gram–Schmidt's stable cousin); orthogonal weight initialisation ([Weight Initialization](/learn/m07/initialization)) uses the Q of a QR factorisation.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Choosing a norm" },
      {
        t: "list",
        items: [
          "**L2** is smooth, unchanged by rotation and cheap to differentiate, so it dominates as penalty and distance. But squaring makes it sensitive to outliers, and it never gives exact zeros.",
          "**L1** gives sparsity and resists outliers, but has a kink at 0, so plain gradient descent jitters; exact zeros need proximal steps or pruning. It changes under rotation: sparsity depends on the chosen basis.",
          "**L∞** limits bound every coordinate — right when each has its own physical limit (pixel values). Clipping by value changes a gradient's direction; clipping by L2 norm does not.",
          "**Spectral norm** measures exactly how much a layer can amplify signals or gradients, but needs an SVD; power iteration gives a cheap, slightly lagging estimate.",
        ],
      },
      { t: "h", text: "Choosing a distance" },
      {
        t: "list",
        items: [
          "**Cosine** ignores length, so it is robust to document length or norm drift — but it discards anything stored in the length (confidence, frequency, popularity).",
          "**Euclidean** depends on feature scales: a feature in thousands swamps one in fractions. Standardise first.",
          "**High dimensions** squeeze the range of distances for unstructured data; the cure is learned structure, dimensionality reduction (PCA) or metric learning, not a different formula.",
          "**Normalising embeddings** makes inner-product search equal cosine search, at the cost of losing length information.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Calling L0 a norm and optimising it with gradients.** Its gradient is zero almost everywhere. **Fix:** use L1 as a convex surrogate, or prune after training.",
          "**Expecting negative cosines from non-negative features.** If all features are ≥ 0 (counts, ReLU outputs), all cosines lie in [0, 1] and differences squeeze together. **Fix:** subtract the mean first if direction relative to the mean matters.",
          "**Comparing Euclidean distances across embeddings of different scale.** Norm-~30 embeddings and norm-~1 embeddings give incomparable distances. **Fix:** normalise, or compare rankings.",
          "**Normalising along the wrong axis.** `F.normalize(x, dim=0)` on a `(batch, features)` tensor normalises each feature across the batch, not each embedding. **Fix:** use `dim=1` or `dim=-1` and assert row norms equal 1.",
          "**Dividing by a zero norm.** Normalising an all-zero vector gives NaNs that spread silently. **Fix:** add a small epsilon to the denominator (`F.normalize` does by default).",
          "**Clipping each tensor separately when you meant global clipping.** That changes the layers' relative gradient sizes, and so the update direction. **Fix:** `clip_grad_norm_` over all parameters at once, after `backward()` and before `optimizer.step()`.",
          "**Classical Gram–Schmidt in floating point.** Computing all projections from the original vector loses orthogonality for nearly dependent inputs. **Fix:** modified Gram–Schmidt, or better `np.linalg.qr`.",
          "**Inverting AᵀA to project.** Forming and inverting AᵀA squares the condition number. **Fix:** `np.linalg.lstsq` or a QR factorisation.",
          "**Weight-decaying biases and normalisation parameters.** They do not cause overfitting the same way, and decaying them can hurt. **Fix:** exclude them via optimizer parameter groups (common in Transformer training).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What are the L1, L2 and L∞ norms of (3, −4)?", a: "L1 = |3| + |−4| = 7; L2 = √(9 + 16) = 5; L∞ = max(3, 4) = 4. Always L∞ ≤ L2 ≤ L1." },
      { t: "qa", level: "beginner", q: "What does it mean for vectors to be linearly independent, and how do you check it?", a: "No vector is a combination of the others; equivalently, Σ cᵢvᵢ = 0 only when all cᵢ = 0. Put the vectors in a matrix's columns and compute its rank (elimination or SVD): independent exactly when rank equals the number of vectors." },
      { t: "qa", level: "beginner", q: "Why is cosine similarity often preferred to Euclidean distance for text embeddings?", a: "It compares direction only, so two texts on the same topic match even if their vectors differ in length (from text length, frequency or the model). Euclidean distance mixes direction with length. After L2-normalisation both give the same ranking, since ‖a − b‖² = 2 − 2 cos θ." },
      { t: "qa", level: "intermediate", q: "Explain geometrically why L1 regularization produces sparse solutions.", a: "Penalised regression equals minimising the loss inside a norm ball: the loss contours grow until they touch the ball. The L1 ball's corners and low-dimensional faces lie where some coordinates are exactly zero, and a typical contour is far more likely to touch there. The round L2 ball is usually touched where all coordinates are non-zero. Algebraically: L1's proximal step is soft-thresholding, mapping every |z| ≤ λ to zero, while L2 only rescales z by 1/(1 + λ)." },
      { t: "qa", level: "intermediate", q: "What is the rank of a matrix and what does a low-rank weight matrix imply?", a: "The dimension of its column space — the number of independent output directions. A d_out × d_in weight of rank r maps every input into an r-dimensional subspace and factors as BA (B is d_out × r, A is r × d_in), using r(d_in + d_out) instead of d_in·d_out parameters. LoRA learns such a low-rank update to a frozen weight." },
      { t: "qa", level: "intermediate", q: "How does gradient clipping by norm work and why is it preferable to clipping by value?", a: "Compute the global L2 norm of all gradients; if it exceeds c, multiply all gradients by c/‖g‖. Direction is unchanged; only length is capped. Clipping by value clamps each coordinate to [−c, c], cutting large coordinates more than small ones, so the direction changes and may no longer reduce the loss as fast." },
      { t: "qa", level: "intermediate", q: "Derive the projection of b onto the column space of A.", a: "Minimise ‖b − Ac‖². The gradient is −2Aᵀ(b − Ac); setting it to zero gives the normal equations AᵀAc = Aᵀb, so c = (AᵀA)⁻¹Aᵀb (columns independent) and p = A(AᵀA)⁻¹Aᵀb. The residual b − p is orthogonal to every column. With orthonormal columns Q this simplifies to p = QQᵀb." },
      { t: "qa", level: "advanced", q: "What is the curse of dimensionality for nearest-neighbour search, quantitatively?", a: "With independent, identically distributed coordinates, ‖x − y‖² is a sum of d independent terms: its mean grows like d, its standard deviation like √d, so the relative spread is O(1/√d). For the uniform cube it is √1.4/√d ≈ 1.18/√d. Nearest and farthest neighbours become nearly equidistant, and small noise can reorder them. Learned embeddings, lying near low-dimensional manifolds, suffer less — an argument for metric learning or dimensionality reduction over raw features." },
      { t: "qa", level: "advanced", q: "How are the spectral norm, the Frobenius norm and Lipschitz continuity of a network related?", a: "The Lipschitz constant is the most a function can stretch distances. For a linear layer (in L2) it is ‖W‖₂ = σ_max. With 1-Lipschitz activations (ReLU, tanh), a network's constant is at most the product Π‖W_l‖₂. Since ‖W‖₂ ≤ ‖W‖_F, bounding Frobenius norms bounds it too, but loosely (up to √rank). Spectral normalization divides each W by an estimate of σ_max (power iteration) to make each layer 1-Lipschitz — used for GAN discriminators and certified robustness." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Compute the L1 norm of (2, −3, 6).", answer: 11, tolerance: 0.001, explain: "|2| + |−3| + |6| = 11." },
          { kind: "numeric", q: "Compute the L2 norm of (2, −3, 6).", answer: 7, tolerance: 0.001, explain: "√(4 + 9 + 36) = √49 = 7." },
          { q: "Which of these is NOT a norm?", options: ["L1", "L2", "L∞", "L0 (number of non-zeros)"], answer: 3, explain: "L0 fails homogeneity: scaling a vector by 2 does not change its count of non-zeros." },
          { kind: "numeric", q: "What is the cosine similarity between (1, 0) and (1, 1)? (4 decimals)", answer: 0.7071, tolerance: 0.001, explain: "Dot product 1; norms 1 and √2; cos = 1/√2 ≈ 0.7071 (an angle of 45°)." },
          { kind: "numeric", q: "Two unit vectors have cosine similarity 0.5. What is their Euclidean distance?", answer: 1, tolerance: 0.001, explain: "‖a − b‖² = 2 − 2·0.5 = 1, so the distance is 1." },
          { kind: "numeric", q: "A gradient g = (6, 8) is clipped to max L2 norm 5. What is the first component afterwards?", answer: 3, tolerance: 0.001, explain: "‖g‖ = 10 > 5, so g ← g·5/10 = (3, 4)." },
          { q: "What is the rank of the matrix with rows (1, 2, 3), (2, 4, 6), (1, 0, 1)?", options: ["1", "2", "3", "0"], answer: 1, explain: "Row 2 = 2 × row 1, so it adds nothing; rows 1 and 3 are independent. Rank 2." },
          { q: "Why does L1 regularization produce exactly-zero weights while L2 does not?", options: ["L1 has a larger penalty for large weights", "The L1 penalty's (sub)gradient keeps constant magnitude λ near zero, and its proximal step thresholds small values to zero; L2's gradient λw vanishes near zero", "L2 is not convex", "L1 is only applied to biases"], answer: 1, explain: "L2 pulls proportionally to w, weakening near zero; L1 pulls with constant force, so weights whose data-driven pull is weaker than λ end exactly at zero." },
          { q: "As the dimension d of uniformly random data grows, the ratio (farthest distance − nearest distance)/nearest distance…", options: ["grows like d", "stays constant", "shrinks roughly like 1/√d", "becomes negative"], answer: 2, explain: "Distances concentrate: their mean grows like √d but their relative spread falls like 1/√d." },
        ],
      },
      {
        t: "exercise",
        title: "Gram–Schmidt and a projection by hand",
        prompt: String.raw`(a) Apply Gram–Schmidt to $\mathbf{v}_1 = (3, 4)$ and $\mathbf{v}_2 = (1, 0)$. (b) Project $\mathbf{b} = (1, 0)$ onto the line spanned by $(3, 4)$ and verify that the residual is orthogonal to $(3, 4)$.`,
        hint: "‖(3, 4)‖ = 5. In (b), the projection coefficient is aᵀb / aᵀa.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "q₁", tex: String.raw`\mathbf{q}_1 = (3, 4)/5 = (0.6,\ 0.8)` },
              { title: "w₂", tex: String.raw`\mathbf{q}_1^\top\mathbf{v}_2 = 0.6,\quad \mathbf{w}_2 = (1, 0) - 0.6\,(0.6, 0.8) = (0.64,\ -0.48),\quad \|\mathbf{w}_2\| = \sqrt{0.4096 + 0.2304} = 0.8` },
              { title: "q₂", tex: String.raw`\mathbf{q}_2 = (0.64, -0.48)/0.8 = (0.8,\ -0.6), \qquad \mathbf{q}_1^\top\mathbf{q}_2 = 0.48 - 0.48 = 0` },
              { title: "(b) projection", tex: String.raw`\operatorname{proj} = \frac{3}{25}(3, 4) = (0.36,\ 0.48),\quad \text{residual} = (0.64,\ -0.48),\quad (3,4)\cdot(0.64,-0.48) = 1.92 - 1.92 = 0` },
            ],
          },
          { t: "p", text: String.raw`The residual in (b) is exactly $\mathbf{w}_2$ from (a): Gram–Schmidt *is* "subtract the projection onto what you already have".` },
        ],
      },
      {
        t: "exercise",
        title: "Top-k cosine search, and when Euclidean disagrees",
        prompt: String.raw`Write a NumPy function that returns the indices and scores of the $k$ most cosine-similar database rows for each query, using one matrix product. Build a database of 10,000 random 64-d vectors with random lengths between 0.5 and 5, and queries that are shrunken (×0.1), slightly noisy copies of items 0 and 1. Compare the top-3 results with Euclidean distance on the raw vectors and on unit-normalised vectors, and verify $\|\hat{\mathbf{a}}-\hat{\mathbf{b}}\|^2 = 2 - 2\cos\theta$.`,
        hint: "Normalise rows with x / np.linalg.norm(x, axis=1, keepdims=True); np.argsort(-sims, axis=1)[:, :k] gives the top-k.",
        solution: [
          { t: "code", lang: "python", title: "cosine_search.py", code: EX_CODE, output: EX_OUTPUT },
          {
            t: "p",
            text: "Cosine search finds items 0 and 1 with similarity ≈ 0.99. Euclidean on raw vectors returns unrelated items (9755, 1314, …): the shrunken query sits near the origin, so the shortest vectors look 'closest' whatever their direction. On unit vectors, Euclidean gives exactly the cosine ranking, as the identity predicts.",
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`**Span** = everything you can build from the vectors; **independent** = none of them can be built from the others; **basis** = independent and spans the space; **dimension** = how many vectors any basis has.`,
          String.raw`**Rank** = how many truly different directions the columns give (= number of pivots); $\operatorname{rank}A + \dim\mathcal{N}(A) = n$; $\operatorname{rank}(AB) \le \min(\operatorname{rank}A, \operatorname{rank}B)$.`,
          String.raw`**A norm** is zero only for the zero vector, scales with the vector, and obeys the triangle inequality. $\|\mathbf{x}\|_p = (\sum|x_i|^p)^{1/p}$; $L_\infty = \max|x_i|$; $L_0$ counts non-zeros and is not a norm. $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt n\|\mathbf{x}\|_2$.`,
          String.raw`**Matrix norms:** $\|A\|_F = \sqrt{\sum A_{ij}^2}$; $\|A\|_2 = \sigma_{\max}$ = the biggest stretch, and $\|AB\| \le \|A\|\|B\|$.`,
          String.raw`**Cosine** $= \mathbf{x}^\top\mathbf{y}/(\|\mathbf{x}\|\|\mathbf{y}\|)$ looks only at direction; for unit vectors $\|\hat{\mathbf{a}}-\hat{\mathbf{b}}\|^2 = 2 - 2\cos\theta$.`,
          String.raw`**High dimensions:** random distances bunch together; their relative spread shrinks like $1/\sqrt d$.`,
          String.raw`**Projection:** $\operatorname{proj}_\mathbf{a}\mathbf{b} = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\mathbf{a}$; onto $\mathcal{C}(A)$: $A(A^\top A)^{-1}A^\top\mathbf{b}$, or $QQ^\top\mathbf{b}$ with orthonormal $Q$; the leftover is perpendicular. Gram–Schmidt builds an orthonormal basis by subtracting projections.`,
          String.raw`**L1 vs L2:** L1 soft-thresholds, $\operatorname{sign}(z)\max(|z|-\lambda, 0)$, giving exact zeros; L2 only shrinks, $z/(1+\lambda)$. In pictures: the diamond's corners sit on the axes.`,
          "**In practice:** normalise embeddings for cosine search; clip gradients by global L2 norm; weight decay (AdamW) for L2; proximal steps or pruning for true L1 sparsity.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd) — explains the spectral norm, rank and the simulator's ellipse; also gives PCA and low-rank compression.",
          "[Regularization](/learn/m07/regularization) — L1, L2/weight decay, dropout and early stopping in real training.",
          "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — where clipping by norm fits in a training loop.",
          "[Word Embeddings](/learn/m13/word-embeddings) — cosine similarity and normalisation applied to learned representations of words.",
          "[MLE and MAP](/learn/m02/mle-map) — why an L2 penalty is a Gaussian prior and an L1 penalty a Laplace prior.",
        ],
      },
    ],
  },
};

export default lesson;
