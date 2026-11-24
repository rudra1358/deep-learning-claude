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
    "Span, independence, basis, dimension and rank; inner products, projections and Gram–Schmidt; the L0/L1/L2/L∞/Lp, Frobenius and spectral norms with their unit balls; Euclidean, Manhattan and cosine distances and their behaviour in high dimensions; and why an L1 penalty produces exact zeros while L2 only shrinks.",
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
        text: String.raw`The [previous lesson](/learn/m02/vectors-matrices) introduced vectors as arrays and the dot product as a way to combine two of them. This lesson asks the next questions: **which vectors can a set of vectors produce** (span, basis, dimension, rank), **how long is a vector** (norms), and **how far apart are two vectors** (distances and similarities). These sound elementary, but nearly every design decision about embeddings, regularization and training stability is a decision about one of them.`,
      },
      {
        t: "p",
        text: String.raw`A **norm** $\|\mathbf{x}\|$ is a function that assigns a non-negative length to every vector. There is not one correct notion of length: the straight-line (Euclidean, $L_2$) length, the sum of absolute coordinates ($L_1$) and the largest coordinate ($L_\infty$) are all legitimate norms, and they behave differently enough that switching between them changes what an algorithm prefers. An $L_1$ penalty on weights drives many of them to exactly zero; an $L_2$ penalty shrinks all of them smoothly. Clipping gradients by their $L_2$ norm keeps the update direction but bounds its size; clipping each coordinate (an $L_\infty$ constraint) changes the direction.`,
      },
      {
        t: "p",
        text: String.raw`**Distances** follow from norms ($d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$), and **cosine similarity** follows from the inner product. Retrieval systems, recommendation engines, face verification and the attention mechanism all reduce to "find the stored vectors closest to this one", so the choice of distance and whether vectors are normalised first directly determines what counts as similar. High dimensions add a twist: for random data, all pairwise distances become nearly equal, which is why raw Euclidean nearest-neighbour search on unstructured high-dimensional data is less meaningful than it looks.`,
      },
      {
        t: "callout",
        kind: "key",
        title: "The ideas to keep",
        text: String.raw`A basis is a coordinate system; rank counts the independent directions a matrix can output. A norm is a choice of what "size" means, visible in the shape of its unit ball: a diamond ($L_1$), a circle ($L_2$), a square ($L_\infty$). Cosine similarity compares directions and ignores length; after normalising to unit length, Euclidean distance and cosine similarity rank neighbours identically, since $\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2\cos\theta$.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — you need the dot product $\mathbf{u}\cdot\mathbf{v} = \sum_i u_iv_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$, matrix-vector products as combinations of columns ($A\mathbf{x} = \sum_j x_j \mathbf{a}_j$), the transpose, and the inverse of a $2\times 2$ matrix. This lesson builds directly on all four and does not repeat them.`,
          String.raw`**Solving small linear systems** by elimination (subtract multiples of one equation from another). We use it to test linear independence and compute rank.`,
          String.raw`**Absolute value, square roots and the Pythagorean theorem.** The $L_2$ norm is Pythagoras in $n$ dimensions.`,
          String.raw`**Optional:** the idea of a derivative from [Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) helps in the regularization section, where we compare the gradient of $w^2$ (which vanishes near zero) with the subgradient of $|w|$ (which does not).`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Three ways to measure a trip across a city",
        text: "You are at a street corner and a café is 3 blocks east and 4 blocks north. A bird flies straight there: 5 blocks (Euclidean, L2). A taxi must follow the grid: 3 + 4 = 7 blocks (Manhattan, L1). A chess king, which moves one square in any direction including diagonally per move, needs max(3, 4) = 4 moves (Chebyshev, L∞). All three are honest answers to 'how far?', and each is the natural one for some purpose. Now ignore distance altogether and ask only which compass direction the café lies in: that is what cosine similarity measures. A café 30 blocks east and 40 north is in exactly the same direction, cosine similarity 1, even though it is ten times farther away.",
      },
      {
        t: "p",
        text: "A **vector space** is a collection of objects you can add together and stretch by numbers without ever leaving the collection — arrows in the plane, lists of 768 numbers, even functions. The **span** of some vectors is everything you can build from them by stretching and adding. Two non-parallel arrows in the plane span the whole plane; two parallel arrows span only a line, because the second adds nothing new. Vectors that each add a genuinely new direction are **linearly independent**. A set of independent vectors that spans the whole space is a **basis** — a coordinate system — and the number of vectors in any basis is the **dimension**. The **rank** of a matrix is the number of independent directions among its columns: how many dimensions its outputs can actually fill.",
      },
      {
        t: "p",
        text: "A **projection** is a shadow: the closest point to a vector within a line or plane. The leftover part (vector minus shadow) is perpendicular to the line or plane. **Gram–Schmidt** turns any basis into a perpendicular, unit-length (orthonormal) one by repeatedly removing each new vector's shadow on the directions already chosen.",
      },
      {
        t: "p",
        text: "Why do norms matter for learning? Penalising the **L2** norm of the weights is like attaching each weight to zero with a spring: big weights are pulled back hard, small ones barely at all, so weights become small but rarely exactly zero. Penalising the **L1** norm is like a constant friction force toward zero: even a tiny weight feels the full pull, so weights that do not earn their keep are pushed all the way to zero and stay there. That is the whole difference between 'shrinkage' and 'sparsity'.",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "Computing the common vector norms" },
      {
        t: "steps",
        items: [
          { title: "Take absolute values", text: String.raw`For $\mathbf{x} = (3, -4, 0, 12)$ the magnitudes are $(3, 4, 0, 12)$. Signs never matter to a norm.` },
          { title: "L0: count non-zeros", text: String.raw`Three entries are non-zero, so $\|\mathbf{x}\|_0 = 3$. (Not a true norm — see the math section.)` },
          { title: "L1: add the magnitudes", tex: String.raw`\|\mathbf{x}\|_1 = 3 + 4 + 0 + 12 = 19` },
          { title: "L2: square, add, square-root", tex: String.raw`\|\mathbf{x}\|_2 = \sqrt{9 + 16 + 0 + 144} = \sqrt{169} = 13` },
          { title: "L∞: take the largest magnitude", tex: String.raw`\|\mathbf{x}\|_\infty = \max(3, 4, 0, 12) = 12` },
          { title: "Lp in general: raise to p, add, take the p-th root", text: String.raw`As $p$ grows the largest entry dominates: $\|\mathbf{x}\|_3 = 1819^{1/3} = 12.2071$ and $\|\mathbf{x}\|_{10} = 12.0000$ to four decimals, approaching $\|\mathbf{x}\|_\infty = 12$.` },
        ],
      },
      { t: "h", text: "Testing independence and computing rank by elimination" },
      {
        t: "steps",
        items: [
          { title: "Put the vectors in the columns of a matrix", text: String.raw`$V = [\mathbf{v}_1\ \mathbf{v}_2\ \mathbf{v}_3]$. The vectors are independent exactly when $V\mathbf{c} = \mathbf{0}$ has only the solution $\mathbf{c} = \mathbf{0}$.` },
          { title: "Row-reduce", text: "Pick a pivot (largest entry in the current column for numerical safety), swap it to the top, subtract multiples of the pivot row to zero out everything below it, move to the next column." },
          { title: "Count pivots", text: "A column with no usable pivot (all remaining entries ≈ 0) contributes no new direction. The number of pivots is the rank." },
          { title: "Interpret", text: String.raw`Rank = number of vectors ⇒ independent. Rank smaller ⇒ dependent: some vector is a combination of the others. In floating point, "zero" means "below a tolerance", which is why libraries compute rank from singular values instead (see [Eigenvalues and SVD](/learn/m02/eigen-svd)).` },
        ],
      },
      { t: "h", text: "Gram–Schmidt: building an orthonormal basis" },
      {
        t: "steps",
        items: [
          { title: "Normalise the first vector", tex: String.raw`\mathbf{q}_1 = \mathbf{v}_1 / \|\mathbf{v}_1\|` },
          { title: "Remove the shadow of the next vector on every earlier q", text: "Subtract its projection on each $\\mathbf{q}_j$ already found. What remains is perpendicular to all of them.", tex: String.raw`\mathbf{w}_k = \mathbf{v}_k - \sum_{j<k} (\mathbf{q}_j^\top \mathbf{v}_k)\,\mathbf{q}_j` },
          { title: "Normalise (or discard)", text: String.raw`If $\|\mathbf{w}_k\| > 0$, set $\mathbf{q}_k = \mathbf{w}_k / \|\mathbf{w}_k\|$. If $\mathbf{w}_k \approx \mathbf{0}$, then $\mathbf{v}_k$ was dependent on earlier vectors; skip it.` },
          { title: "Repeat for every vector", text: String.raw`The $\mathbf{q}$'s are orthonormal and span the same space as the $\mathbf{v}$'s. Stacking them as columns gives $Q$ with $Q^\top Q = I$; this is the QR factorisation $V = QR$.` },
        ],
      },
      { t: "h", text: "Nearest-neighbour search with cosine similarity" },
      {
        t: "steps",
        items: [
          { title: "Normalise every stored vector once", text: "Divide each row of the database by its L2 norm and store the result." },
          { title: "Normalise the query", text: "Same operation on the query vector." },
          { title: "One matrix-vector product", text: "The dot product of unit vectors is their cosine, so `database_unit @ query_unit` gives every similarity at once." },
          { title: "Take the top-k", text: "Sort (or partially sort) the scores. Approximate nearest-neighbour libraries replace this exhaustive scan by indexes, but the score is the same." },
        ],
      },
      { t: "h", text: "Gradient clipping by norm" },
      {
        t: "p",
        text: String.raw`Before an optimizer step, compute the $L_2$ norm of all gradients concatenated, $\|\mathbf{g}\|_2$. If it exceeds a threshold $c$, rescale every gradient by $c/\|\mathbf{g}\|_2$. Direction is preserved, only the step length is capped. For $\mathbf{g} = (3, 4, 12)$ and $c = 1$: $\|\mathbf{g}\|_2 = 13$, so $\mathbf{g} \leftarrow \mathbf{g}/13 = (0.2308, 0.3077, 0.9231)$. Learning-rate schedules and clipping are covered in [Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Vector spaces and subspaces" },
      {
        t: "p",
        text: String.raw`A **(real) vector space** is a set $\mathcal{V}$ with an addition $\mathbf{u} + \mathbf{v}$ and a scalar multiplication $\alpha\mathbf{v}$ ($\alpha \in \mathbb{R}$) that stay inside $\mathcal{V}$ and obey the familiar rules: addition is associative and commutative, there is a zero vector $\mathbf{0}$, every $\mathbf{v}$ has a negative $-\mathbf{v}$, $1\mathbf{v} = \mathbf{v}$, $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$, and both distributive laws hold. $\mathbb{R}^n$ is the example that matters here, but polynomials, functions and matrices of a fixed shape are vector spaces too.`,
      },
      {
        t: "p",
        text: String.raw`A **subspace** is a subset $\mathcal{S} \subseteq \mathcal{V}$ that is itself a vector space: it contains $\mathbf{0}$ and is closed under addition and scaling. In $\mathbb{R}^3$ the subspaces are $\{\mathbf{0}\}$, lines through the origin, planes through the origin and $\mathbb{R}^3$ itself. A line that misses the origin is **not** a subspace (it does not contain $\mathbf{0}$) — the reason a layer $W\mathbf{x} + \mathbf{b}$ with $\mathbf{b} \neq \mathbf{0}$ is called *affine* rather than linear.`,
      },
      { t: "h", text: "Span, linear independence, basis, dimension" },
      {
        t: "math",
        tex: String.raw`\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_k\} = \Big\{\textstyle\sum_{i=1}^k c_i\mathbf{v}_i \;:\; c_i \in \mathbb{R}\Big\}, \qquad \text{independent} \iff \Big(\textstyle\sum_i c_i\mathbf{v}_i = \mathbf{0} \Rightarrow c_1 = \dots = c_k = 0\Big)`,
      },
      {
        t: "p",
        text: String.raw`The span is the set of all **linear combinations**; it is always a subspace. Vectors are **linearly independent** if the only combination giving zero is the trivial one — equivalently, none of them lies in the span of the others. A **basis** of $\mathcal{V}$ is an independent set that spans $\mathcal{V}$; every vector then has **unique** coordinates in that basis. All bases of a space have the same number of elements, the **dimension** $\dim\mathcal{V}$. In $\mathbb{R}^n$, any $n+1$ vectors are dependent and any $n$ independent vectors form a basis.`,
      },
      { t: "h", text: "Rank and the four subspaces of a matrix" },
      {
        t: "p",
        text: String.raw`For $A \in \mathbb{R}^{m\times n}$, the **column space** $\mathcal{C}(A) = \{A\mathbf{x}\}$ is the span of the columns (all possible outputs) and the **null space** $\mathcal{N}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ is the set of inputs that are annihilated. The **rank** is $\operatorname{rank}A = \dim\mathcal{C}(A)$, which always equals the dimension of the row space (column rank = row rank). The **rank–nullity theorem** links them:`,
      },
      { t: "math", tex: String.raw`\operatorname{rank}(A) + \dim\mathcal{N}(A) = n, \qquad \operatorname{rank}(A) \le \min(m, n), \qquad \operatorname{rank}(AB) \le \min(\operatorname{rank}A, \operatorname{rank}B)` },
      {
        t: "p",
        text: String.raw`Here $n$ is the number of columns (the input dimension). Every input direction either survives into the output (counted by the rank) or is destroyed (counted by the nullity). The last inequality explains **bottlenecks**: a network $W_2W_1$ whose hidden layer has width $r$ can represent at most a rank-$r$ linear map — the idea behind autoencoders and low-rank adapters.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** $\mathbf{v}_1 = (1, 0, 2)$, $\mathbf{v}_2 = (0, 1, 1)$, $\mathbf{v}_3 = (2, 3, 7)$. Since $2\mathbf{v}_1 + 3\mathbf{v}_2 = (2, 3, 4 + 3) = (2, 3, 7) = \mathbf{v}_3$, the set is dependent; $\mathbf{v}_1, \mathbf{v}_2$ are independent (neither is a multiple of the other), so $\operatorname{rank}[\mathbf{v}_1\ \mathbf{v}_2\ \mathbf{v}_3] = 2$. The three vectors span only a plane in $\mathbb{R}^3$, and the null space is the line spanned by $(2, 3, -1)$, so $2 + 1 = 3 = n$ as rank–nullity requires.`,
      },
      { t: "h", text: "Inner products" },
      {
        t: "p",
        text: String.raw`An **inner product** $\langle\mathbf{u}, \mathbf{v}\rangle$ is a function of two vectors that is symmetric ($\langle\mathbf{u},\mathbf{v}\rangle = \langle\mathbf{v},\mathbf{u}\rangle$), linear in each argument, and positive definite ($\langle\mathbf{v},\mathbf{v}\rangle > 0$ for $\mathbf{v} \ne \mathbf{0}$). The dot product $\mathbf{u}^\top\mathbf{v}$ is the standard one; $\langle\mathbf{u},\mathbf{v}\rangle_M = \mathbf{u}^\top M\mathbf{v}$ for a symmetric positive-definite matrix $M$ is another (it underlies the Mahalanobis distance). Every inner product satisfies the **Cauchy–Schwarz inequality**`,
      },
      { t: "math", tex: String.raw`|\langle\mathbf{u},\mathbf{v}\rangle| \le \|\mathbf{u}\|\,\|\mathbf{v}\|, \qquad \|\mathbf{v}\| = \sqrt{\langle\mathbf{v},\mathbf{v}\rangle}`, caption: "Equality holds exactly when one vector is a multiple of the other. This is why cosine similarity lies in [−1, 1]." },
      { t: "h", text: "Norms" },
      {
        t: "p",
        text: String.raw`A **norm** is a function $\|\cdot\| : \mathcal{V} \to [0, \infty)$ with three properties for all vectors $\mathbf{x}, \mathbf{y}$ and scalars $\alpha$:`,
      },
      {
        t: "math",
        tex: String.raw`\underbrace{\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}}_{\text{definiteness}}, \qquad \underbrace{\|\alpha\mathbf{x}\| = |\alpha|\,\|\mathbf{x}\|}_{\text{homogeneity}}, \qquad \underbrace{\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|}_{\text{triangle inequality}}`,
      },
      { t: "math", tex: String.raw`\|\mathbf{x}\|_p = \Big(\sum_{i=1}^n |x_i|^p\Big)^{1/p}\ (p \ge 1), \qquad \|\mathbf{x}\|_\infty = \max_i |x_i| = \lim_{p\to\infty}\|\mathbf{x}\|_p, \qquad \|\mathbf{x}\|_0 = \#\{i : x_i \ne 0\}` },
      {
        t: "p",
        text: String.raw`$p = 1$ gives the **L1 (Manhattan)** norm, $p = 2$ the **L2 (Euclidean)** norm. The "**L0 norm**" counts non-zeros; it is *not* a norm because it violates homogeneity: $\|2\mathbf{x}\|_0 = \|\mathbf{x}\|_0$, not $2\|\mathbf{x}\|_0$. For $0 < p < 1$ the formula also fails the triangle inequality. In $\mathbb{R}^n$ all norms are **equivalent** up to constants; the most useful bounds are`,
      },
      { t: "math", tex: String.raw`\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\,\|\mathbf{x}\|_2 \le n\,\|\mathbf{x}\|_\infty` },
      {
        t: "p",
        text: String.raw`For $\mathbf{x} = (3, -4, 0, 12)$ ($n = 4$): $12 \le 13 \le 19 \le \sqrt{4}\cdot 13 = 26 \le 48$. The constants grow with $n$, which matters in deep learning, where $n$ is in the millions: a gradient whose every coordinate is small ($\|\mathbf{g}\|_\infty$ small) can still have a large $L_2$ norm.`,
      },
      { t: "h", text: "Unit balls: the geometry of a norm" },
      {
        t: "p",
        text: String.raw`The **unit ball** $\{\mathbf{x} : \|\mathbf{x}\| \le 1\}$ shows a norm's character. In 2-D: $L_1$ gives a diamond with corners at $(\pm1, 0)$, $(0, \pm1)$; $L_2$ a circle; $L_\infty$ a square with corners $(\pm1, \pm1)$; general $L_p$ interpolates between them, becoming more square as $p$ grows. Every norm ball is convex and symmetric; the corners of the $L_1$ ball sit **on the coordinate axes**, where all but one coordinate is zero. That single geometric fact is the source of L1 sparsity.`,
      },
      { t: "h", text: "Matrix norms" },
      {
        t: "math",
        tex: String.raw`\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}, \qquad \|A\|_2 = \max_{\mathbf{x}\ne\mathbf{0}}\frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_{\max}(A)`,
      },
      {
        t: "p",
        text: String.raw`The **Frobenius norm** treats the matrix as one long vector. The **spectral (operator) norm** is the largest factor by which $A$ can stretch any vector; it equals the largest **singular value** $\sigma_{\max}$, the square root of the largest eigenvalue of $A^\top A$ (derived in [Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd)). It is **submultiplicative**, $\|AB\|_2 \le \|A\|_2\|B\|_2$, so a stack of $L$ layers stretches a signal (or a backpropagated gradient) by at most the product of their spectral norms — the quantitative link to [exploding and vanishing gradients](/learn/m04/gradient-problems). Always $\|A\|_2 \le \|A\|_F \le \sqrt{\operatorname{rank}A}\,\|A\|_2$.`,
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
        text: String.raw`Every norm induces a **metric** (non-negative, zero only for equal points, symmetric, triangle inequality). **Cosine distance** is not a metric (it fails the triangle inequality) and ignores length entirely. For unit vectors the two views coincide:`,
      },
      { t: "math", tex: String.raw`\|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|_2^2 = \|\hat{\mathbf{a}}\|^2 + \|\hat{\mathbf{b}}\|^2 - 2\,\hat{\mathbf{a}}^\top\hat{\mathbf{b}} = 2 - 2\cos\theta \qquad (\|\hat{\mathbf{a}}\| = \|\hat{\mathbf{b}}\| = 1)`, caption: "Euclidean distance between normalised vectors is a decreasing function of cosine similarity, so both produce the same nearest-neighbour ranking." },
      {
        t: "p",
        text: String.raw`**When cosine beats Euclidean.** Use cosine when the direction carries the meaning and the length is a nuisance: word-count vectors of documents of different lengths, sentence embeddings whose norm tracks frequency or token count, embeddings trained with a cosine-based contrastive loss. Example with term counts over the vocabulary (cat, sat, ran): $A = (1, 1, 0)$ "cat sat", $B = (10, 10, 0)$ the same sentence repeated ten times, $C = (1, 0, 1)$ "cat ran". Euclidean: $d(A, B) = \sqrt{81 + 81} = 12.728$ but $d(A, C) = \sqrt{0 + 1 + 1} = 1.414$, so $C$ looks closer. Cosine: $\cos(A, B) = 20/(\sqrt2\cdot\sqrt{200}) = 1$, $\cos(A, C) = 1/(\sqrt2\cdot\sqrt2) = 0.5$, so $B$ is the better match. Use Euclidean (or plain dot product) when magnitude is meaningful — for example when a model was trained with an L2 or dot-product objective, or when vector length encodes confidence or popularity.`,
      },
      { t: "h", text: "Distance concentration in high dimensions" },
      {
        t: "p",
        text: String.raw`Let $\mathbf{x}, \mathbf{y}$ be independent and uniform in $[0, 1]^d$. Each coordinate difference satisfies $\mathbb{E}[(x_i - y_i)^2] = 1/6$ and $\operatorname{Var}[(x_i - y_i)^2] = 1/15 - 1/36 = 7/180$, so by independence`,
      },
      { t: "math", tex: String.raw`\mathbb{E}\,\|\mathbf{x}-\mathbf{y}\|^2 = \frac{d}{6}, \qquad \frac{\operatorname{std}\,\|\mathbf{x}-\mathbf{y}\|^2}{\mathbb{E}\,\|\mathbf{x}-\mathbf{y}\|^2} = \frac{\sqrt{7d/180}}{d/6} = \frac{\sqrt{1.4}}{\sqrt d} \approx \frac{1.18}{\sqrt d}` },
      {
        t: "p",
        text: String.raw`Distances grow like $\sqrt{d/6}$, but their **relative spread shrinks like** $1/\sqrt d$: the nearest and farthest points become almost equally far. The NumPy program below measures it for 1000 random points: the ratio (max − min)/min falls from 134.6 at $d = 2$ to 3.48 at $d = 10$, 0.389 at $d = 100$ and 0.114 at $d = 1000$ (distances 12.17–13.55 around $\sqrt{1000/6} = 12.91$). Real embeddings escape the worst of this because they lie near low-dimensional structure, but the effect is why nearest-neighbour search works on **learned** representations far better than on raw high-dimensional features.`,
      },
      { t: "h", text: "Projections and orthogonality" },
      {
        t: "p",
        text: String.raw`Vectors are **orthogonal** if $\mathbf{u}^\top\mathbf{v} = 0$; a set is **orthonormal** if additionally each has norm 1. The **projection** of $\mathbf{b}$ onto the line spanned by $\mathbf{a}$ is the point $\hat c\,\mathbf{a}$ that minimises $\|\mathbf{b} - c\,\mathbf{a}\|^2$. Setting the derivative $-2\mathbf{a}^\top(\mathbf{b} - c\,\mathbf{a})$ to zero gives`,
      },
      { t: "math", tex: String.raw`\operatorname{proj}_{\mathbf{a}}\mathbf{b} = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a}, \qquad \mathbf{a}^\top(\mathbf{b} - \operatorname{proj}_{\mathbf{a}}\mathbf{b}) = 0` },
      {
        t: "p",
        text: String.raw`The residual is orthogonal to $\mathbf{a}$: that is the defining property of the closest point. Onto a subspace spanned by the independent columns of $A$, the same argument (minimise $\|\mathbf{b} - A\mathbf{c}\|^2$, set the gradient $-2A^\top(\mathbf{b} - A\mathbf{c})$ to zero) gives the **normal equations** and the projection matrix`,
      },
      { t: "math", tex: String.raw`A^\top A\,\hat{\mathbf{c}} = A^\top\mathbf{b}, \qquad \mathbf{p} = A\hat{\mathbf{c}} = \underbrace{A(A^\top A)^{-1}A^\top}_{P}\,\mathbf{b}, \qquad P^2 = P,\ P^\top = P` },
      {
        t: "p",
        text: String.raw`This is exactly least-squares linear regression: the fitted values are the projection of the targets onto the column space of the design matrix. If the columns are orthonormal ($A = Q$), $Q^\top Q = I$ and the formula collapses to $P = QQ^\top$ — the practical reason to orthonormalise.`,
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
        text: String.raw`The program subtracts each projection from the *updated* vector (modified Gram–Schmidt), which gives the same result in exact arithmetic but loses far less orthogonality to rounding error. Libraries use Householder reflections (as in NumPy's linalg.qr) for full stability.`,
      },
      { t: "h", text: "Why L1 gives sparsity and L2 gives shrinkage" },
      {
        t: "p",
        text: String.raw`**Geometric argument.** A penalised problem $\min_\mathbf{w} \mathcal{L}(\mathbf{w}) + \lambda\|\mathbf{w}\|$ is equivalent (for a matching budget $t$) to the constrained problem $\min \mathcal{L}(\mathbf{w})$ subject to $\|\mathbf{w}\| \le t$. The solution is where the smallest level set (contour) of $\mathcal{L}$ first touches the norm ball. A smooth elliptical contour approaching a **circle** touches it at a generic point, where both coordinates are non-zero. Approaching a **diamond**, it is very likely to touch a **corner**, and corners lie on the axes, where coordinates are exactly zero. In higher dimensions the $L_1$ ball has $2n$ corners and many low-dimensional edges and faces, all of which are sparse, so the effect becomes stronger.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Find the closest point to $\mathbf{a} = (2, 0.5)$ inside each unit ball (the loss is $\|\mathbf{w} - \mathbf{a}\|^2$). For $L_2$: scale $\mathbf{a}$ back onto the circle, $\mathbf{a}/\|\mathbf{a}\| = (2, 0.5)/2.0616 = (0.9701, 0.2425)$ — both coordinates non-zero. For $L_1$: the solution is $\operatorname{sign}(a_i)\max(|a_i| - \tau, 0)$ with $\tau$ chosen so the result has $L_1$ norm 1. With $\tau = 1$: $(2 - 1, \max(0.5 - 1, 0)) = (1, 0)$, whose $L_1$ norm is 1. The second coordinate is **exactly zero**: the solution is a corner.`,
      },
      {
        t: "p",
        text: String.raw`**Analytic argument (one coordinate).** Minimise $\tfrac12(w - z)^2 + \lambda\,\Omega(w)$, where $z$ is what the data alone would choose.`,
      },
      {
        t: "math",
        tex: String.raw`\Omega(w) = \tfrac12 w^2:\;\; w^\star = \frac{z}{1+\lambda} \qquad\qquad \Omega(w) = |w|:\;\; w^\star = \operatorname{sign}(z)\max(|z| - \lambda,\ 0)`,
        caption: "Left: L2 multiplies by a constant factor — never exactly zero unless z = 0. Right: L1 soft-thresholding — every |z| ≤ λ is set exactly to zero.",
      },
      {
        t: "p",
        text: String.raw`The reason is the derivative near zero. The gradient of $\tfrac12\lambda w^2$ is $\lambda w$, which **vanishes** as $w \to 0$, so the pull toward zero weakens and never quite finishes. The subgradient of $\lambda|w|$ is $\lambda\operatorname{sign}(w)$, a **constant-size** pull that does not weaken; at $w = 0$ the subdifferential is the whole interval $[-\lambda, \lambda]$, so $w = 0$ is optimal whenever the data's pull $|z|$ is at most $\lambda$. In the program below, ISTA (gradient step + soft-threshold) on a 20-feature regression with 3 true non-zero weights sets 15 of 20 weights exactly to zero; ridge regression with the same $\lambda$ sets none (its smallest weight is 0.0012).`,
      },
      {
        t: "callout",
        kind: "note",
        title: "Weight decay is an L2 penalty — with one caveat",
        text: String.raw`For plain SGD, adding $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ to the loss is identical to "decaying" weights each step: $\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta\nabla\mathcal{L}$. For adaptive optimizers (Adam) the two differ, because the penalty gradient gets divided by Adam's per-parameter scale; AdamW applies the decay directly to the weights instead. See [Regularization](/learn/m07/regularization) and [Momentum and adaptive optimizers](/learn/m06/momentum-adaptive). The Bayesian reading — L2 is a Gaussian prior, L1 a Laplace prior — is in [MLE and MAP](/learn/m02/mle-map).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator applies a 2×2 matrix to the plane. Although it is built for the next lesson, it makes several ideas from this one visible: the column space and rank of a matrix, the image of the L2 unit ball, the spectral norm, and what a projection does.",
      },
      { t: "sim", id: "linear-transform", caption: "The black curve is the image of the L2 unit circle; the shaded parallelogram is the image of the unit square; drag on the plane to move the input vector x." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "**Span and rank.** Choose the *Projection (singular)* preset. Both columns of the matrix point along the x-axis, so the whole plane collapses onto a line: the column space is 1-dimensional, rank 1, det 0. Drag x around — Ax always lands on that line, and every x on the vertical axis maps to 0 (that is the null space; 1 + 1 = 2 as rank–nullity says).",
          String.raw`**Independence.** Start from *General* and drag the sliders until column 2, $(b, d)$, is a multiple of column 1, $(a, c)$ — e.g. $a = 1, c = 2, b = 2, d = 4$. The parallelogram flattens and det reaches 0 exactly when the columns become dependent.`,
          String.raw`**The image of the L2 ball is an ellipse; its longest semi-axis is the spectral norm.** Switch to **SVD** mode and step to the end: the arrows $\sigma_1\mathbf{u}_1$ and $\sigma_2\mathbf{u}_2$ are the semi-axes. With the preset $[[3,0],[4,5]]$ the panel shows $\sigma_1 = 6.71 = \|W\|_2$ and $\sigma_2 = 2.24$; the worked example above gave $\sqrt{45}$ and $\sqrt5$.`,
          String.raw`**Spectral norm as the maximum stretch.** Keep that preset and drag x around the unit circle ($\|\mathbf{x}\| = 1$). The readout $A\mathbf{x}$ never gets longer than 6.71 or shorter than 2.24; the extremes occur when x points along $\mathbf{v}_1$ or $\mathbf{v}_2$.`,
          "**Orthogonal maps preserve norms.** Pick *Rotation 45°* or *Reflection*: the circle maps onto itself and every |Ax| equals |x|. These are the matrices Q with QᵀQ = I that Gram–Schmidt produces.",
        ],
      },
      {
        t: "callout",
        kind: "tip",
        title: "A thought experiment for the unit balls",
        text: String.raw`Draw the diamond $|w_1| + |w_2| \le 1$ and the circle $w_1^2 + w_2^2 \le 1$ on paper, then draw a family of ellipses (loss contours) centred at $(2, 0.5)$ and expand them until they first touch each ball. They touch the circle at $(0.97, 0.24)$ and the diamond at its corner $(1, 0)$ — the worked example of the math section. Move the centre to $(1, 1)$: by symmetry the contour now touches the diamond along an edge, the one case where L1 does not select a single coordinate.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "One program, nine experiments: hand-written norms, rank by Gaussian elimination, projections, modified Gram–Schmidt, Frobenius and spectral norms (by power iteration), cosine versus Euclidean, distance concentration, clipping by norm, and the sparsity of L1 versus the shrinkage of L2. Every number quoted in the math section comes from this output.",
      },
      { t: "code", lang: "python", title: "norms_spaces_from_scratch.py", code: SCRATCH_CODE, output: SCRATCH_OUTPUT },
      { t: "p", text: "Notes on the implementation:" },
      {
        t: "list",
        items: [
          "`rank` uses **partial pivoting** (largest remaining entry as pivot) and a tolerance of 1e-10; without the tolerance, rounding error would make a dependent column look independent.",
          "`gram_schmidt` skips vectors whose residual norm falls below the tolerance, which is how the dependent set `V` yields only 2 basis vectors.",
          "The spectral norm uses **power iteration** on WᵀW: repeatedly multiplying by WᵀW and renormalising aligns v with the top right-singular vector, and |Wv| is then σ_max. The same idea underlies PyTorch's `spectral_norm`.",
          "In the Lasso part, `step` is 1/L, where L = σ_max(X)²/n is the Lipschitz constant of the gradient of the half-MSE; this step size guarantees ISTA does not diverge. The soft-threshold `step * lam` is the L1 proximal operator.",
          "Lasso kept two small spurious weights (indices 1 and 3, magnitudes 0.0231 and 0.0011) besides the three true ones; a larger λ removes them but shrinks the true weights more. Ridge has no exact zeros at all.",
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
                text: "`torch.linalg.vector_norm` and `torch.linalg.matrix_norm` compute every norm in this lesson; `F.normalize` and `F.cosine_similarity` handle embeddings; `clip_grad_norm_` implements clipping by global L2 norm and returns the norm measured before clipping (a useful training-health signal to log). The last part compares weight decay, an L1 penalty optimised with plain autograd (which uses sign(w) as the gradient and never reaches exact zero), and a proximal L1 step that soft-thresholds after each update.",
              },
              { t: "code", lang: "python", title: "norms_torch.py", code: TORCH_CODE, output: TORCH_OUTPUT, runnable: false },
              {
                t: "callout",
                kind: "warning",
                title: "L1 + SGD does not give exact zeros",
                text: "Autograd differentiates |w| as sign(w), so a weight near zero is pushed back and forth across zero by ±lr·λ each step and never settles. You get many tiny weights (3 below 1e-3 above) rather than exact zeros. Proximal updates (soft-thresholding, as in ISTA), or pruning small weights after training, give true sparsity (14 exact zeros above).",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow's equivalents are `tf.norm`, `tf.math.l2_normalize`, `tf.clip_by_global_norm` and the `tf.keras.regularizers.L1/L2` objects. Note the sign convention of `tf.keras.losses.cosine_similarity`: it returns the **negative** cosine so that minimising it increases similarity.",
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
          ["L0 (not a norm)", String.raw`$\#\{x_i \ne 0\}$`, "the two axes", "counts exactly what sparsity means", "non-convex, discontinuous; optimising it is combinatorial (NP-hard in general). Used to *measure* sparsity, pruning"],
          ["L1 (Manhattan)", String.raw`$\sum_i|x_i|$`, "diamond", "convex surrogate for L0; robust to outliers when used as a loss (MAE)", "not differentiable at 0; Lasso, L1 regularization, sparse coding"],
          ["L2 (Euclidean)", String.raw`$\sqrt{\sum_i x_i^2}$`, "circle", "smooth, rotation-invariant, induced by the dot product", "sensitive to outliers when squared; weight decay, clipping, embedding normalisation"],
          ["L∞ (Chebyshev)", String.raw`$\max_i|x_i|$`, "square", "bounds every coordinate", "ignores all but the largest entry; adversarial-robustness budgets, clip-by-value"],
          ["Lp, 1 < p < ∞", String.raw`$(\sum_i|x_i|^p)^{1/p}$`, "rounded square", "interpolates between L1 and L∞", "rarely used directly; Lp pooling, theory"],
          ["Frobenius", String.raw`$\sqrt{\sum_{ij}A_{ij}^2}$`, "—", "cheap; equals L2 of the flattened matrix", "does not measure the worst-case stretch; weight decay on matrices"],
          ["Spectral", String.raw`$\sigma_{\max}(A)$`, "—", "the exact Lipschitz constant of a linear layer", "needs an SVD or power iteration; spectral normalization, stability analysis"],
          ["Nuclear", String.raw`$\sum_i\sigma_i(A)$`, "—", "convex surrogate for rank", "needs an SVD; low-rank matrix completion"],
        ],
      },
      { t: "h", text: "Distances and similarities" },
      {
        t: "table",
        head: ["Measure", "Definition", "Scale-invariant?", "Typical use"],
        rows: [
          ["Euclidean", String.raw`$\|\mathbf{x}-\mathbf{y}\|_2$`, "no", "k-means, k-NN on standardised features, L2-trained embeddings"],
          ["Squared Euclidean", String.raw`$\|\mathbf{x}-\mathbf{y}\|_2^2$`, "no", "MSE, k-means objective; cheaper (no square root) and same ranking"],
          ["Manhattan", String.raw`$\|\mathbf{x}-\mathbf{y}\|_1$`, "no", "grid-like features, robustness to a few large coordinate differences"],
          ["Cosine similarity", String.raw`$\mathbf{x}^\top\mathbf{y}/(\|\mathbf{x}\|\|\mathbf{y}\|)$`, "yes", "text and sentence embeddings, contrastive learning, retrieval"],
          ["Dot product", String.raw`$\mathbf{x}^\top\mathbf{y}$`, "no", "attention scores, maximum-inner-product search in recommenders"],
          ["Mahalanobis", String.raw`$\sqrt{(\mathbf{x}-\mathbf{y})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$`, "affine-invariant", "correlated features, out-of-distribution scoring"],
          ["Hamming", "number of differing positions", "—", "binary codes, hashing-based retrieval"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Semantic search and retrieval-augmented generation.** Documents and queries are embedded, usually L2-normalised, and retrieved by cosine similarity (equivalently, inner product of unit vectors). Vector-search libraries such as FAISS support both L2 and inner-product indexes for this reason.",
          "**Contrastive image–text models.** CLIP normalises its image and text embeddings and trains with a softmax over scaled cosine similarities, so at inference time zero-shot classification is a nearest-neighbour search by cosine.",
          "**Face verification.** Modern face-recognition systems compare L2-normalised embeddings with cosine similarity and train with angular-margin losses (ArcFace is a well-known example); a threshold on the cosine decides 'same person'.",
          String.raw`**Gradient clipping** by global $L_2$ norm is standard when training RNNs and Transformers, where occasional very large gradients would otherwise destabilise training; logging the pre-clipping norm is a common health metric.`,
          String.raw`**Weight decay** (an $L_2$ penalty, or decoupled decay in AdamW) is used in essentially every large-scale training recipe to control weight growth.`,
          String.raw`**Spectral normalization** (Miyato et al., 2018) divides each weight matrix by its spectral norm, bounding the Lipschitz constant of a GAN discriminator.`,
          String.raw`**Sparse models.** L1 regularization (the Lasso) selects features in linear models; magnitude pruning of neural networks removes small weights — an $L_0$ goal approached heuristically.`,
          String.raw`**Least squares and orthogonalisation.** Linear regression is a projection; QR factorisation (Gram–Schmidt's stable cousin) is how it is solved reliably, and orthogonal initialisation of weight matrices (see [Weight Initialization](/learn/m07/initialization)) uses the Q of a QR factorisation.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Choosing a norm" },
      {
        t: "list",
        items: [
          "**L2** is smooth, rotation-invariant and cheap to differentiate, which is why it dominates as a penalty and as a distance. Its weakness: squaring makes it sensitive to outliers and it never produces exact zeros.",
          "**L1** yields sparsity and is more robust to outliers, but is non-differentiable at 0, so naive gradient descent oscillates; proximal methods or pruning are needed for exact zeros. It is not rotation-invariant: sparsity depends on the basis you choose.",
          "**L∞** constraints bound every coordinate, appropriate when each coordinate has its own physical limit (pixel intensities); clipping by value changes the gradient's direction, clipping by L2 norm does not.",
          "**Spectral norm** is the right measure of how much a layer can amplify signals or gradients, but costs an SVD; power iteration gives a cheap, slightly lagging estimate.",
        ],
      },
      { t: "h", text: "Choosing a distance" },
      {
        t: "list",
        items: [
          "**Cosine** is invariant to vector length, making it robust to document length or embedding-norm drift — but it throws away any information stored in the length (confidence, frequency, popularity).",
          "**Euclidean** depends on feature scales: one feature measured in thousands dominates one measured in fractions. Standardise features first.",
          "**High dimensions** compress the range of distances for unstructured data; learned low-dimensional structure, dimensionality reduction (PCA) or metric learning are the remedies, not a different formula.",
          "**Normalising embeddings** makes inner-product search equal to cosine search and keeps magnitudes from dominating, at the price of losing magnitude information and of an extra operation per vector.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Calling L0 a norm and optimising it with gradients.** Its gradient is zero almost everywhere. **Fix:** use L1 as a convex surrogate, or prune after training.",
          "**Using cosine similarity on un-centred non-negative features and expecting negatives.** If all features are non-negative (counts, ReLU outputs), all cosines lie in [0, 1] and differences compress. **Fix:** centre the data if direction relative to the mean is what matters.",
          "**Comparing Euclidean distances across embeddings of different scale.** Distances from a model whose embeddings have norm ~30 are not comparable to one with norm ~1. **Fix:** normalise, or compare rankings rather than raw distances.",
          "**Normalising along the wrong axis.** `F.normalize(x, dim=0)` on a `(batch, features)` tensor normalises each feature across the batch, not each embedding. **Fix:** normalise along the feature axis (`dim=1` or `dim=-1`) and assert row norms equal 1.",
          "**Dividing by a zero norm.** Normalising an all-zero vector produces NaNs that propagate silently. **Fix:** add a small epsilon to the denominator (`F.normalize` does this by default).",
          "**Clipping each parameter tensor separately when you meant global clipping.** Per-tensor clipping changes the relative sizes of different layers' gradients and therefore the update direction. **Fix:** `clip_grad_norm_` over all parameters at once — and call it after `backward()` and before `optimizer.step()`.",
          "**Classical Gram–Schmidt in floating point.** Subtracting all projections computed from the original vector loses orthogonality for nearly dependent inputs. **Fix:** modified Gram–Schmidt, or better, `np.linalg.qr`.",
          "**Inverting AᵀA to project.** Forming and inverting AᵀA squares the condition number. **Fix:** `np.linalg.lstsq` or a QR factorisation.",
          "**Regularising biases and normalisation parameters with weight decay.** These parameters do not cause overfitting in the same way, and decaying them can hurt. **Fix:** exclude them via optimizer parameter groups (a common practice in Transformer training).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What are the L1, L2 and L∞ norms of (3, −4)?", a: "L1 = |3| + |−4| = 7; L2 = √(9 + 16) = 5; L∞ = max(3, 4) = 4. Always L∞ ≤ L2 ≤ L1." },
      { t: "qa", level: "beginner", q: "What does it mean for vectors to be linearly independent, and how do you check it?", a: "No vector in the set can be written as a combination of the others; equivalently the only solution of Σ cᵢvᵢ = 0 is all cᵢ = 0. Put the vectors in the columns of a matrix and compute its rank (by elimination or SVD): they are independent exactly when the rank equals the number of vectors." },
      { t: "qa", level: "beginner", q: "Why is cosine similarity often preferred to Euclidean distance for text embeddings?", a: "It compares direction only, so two texts about the same thing match even if their vectors have different lengths (because of length, frequency or the model's norm behaviour). Euclidean distance mixes direction with magnitude. After L2-normalisation, the two produce the same ranking since ‖a − b‖² = 2 − 2 cos θ." },
      { t: "qa", level: "intermediate", q: "Explain geometrically why L1 regularization produces sparse solutions.", a: "Penalised regression is equivalent to minimising the loss within a norm ball. The loss contours expand until they touch the ball. The L1 ball is a cross-polytope whose corners and low-dimensional faces lie on coordinate subspaces, where some coordinates are exactly zero; a generic contour is far more likely to touch there than in the interior of a face. The L2 ball is round, so the contact point generically has all coordinates non-zero. Analytically: the proximal step of L1 is soft-thresholding, which maps every |z| ≤ λ to zero, while L2 merely rescales z by 1/(1 + λ)." },
      { t: "qa", level: "intermediate", q: "What is the rank of a matrix and what does a low-rank weight matrix imply?", a: "The dimension of its column space — the number of independent output directions. A d_out × d_in weight of rank r maps every input into an r-dimensional subspace and can be factored as BA with B ∈ ℝ^{d_out×r}, A ∈ ℝ^{r×d_in}, using r(d_in + d_out) instead of d_in·d_out parameters. LoRA exploits this by learning a low-rank update to a frozen weight." },
      { t: "qa", level: "intermediate", q: "How does gradient clipping by norm work and why is it preferable to clipping by value?", a: "Compute the global L2 norm of all gradients; if it exceeds c, multiply all gradients by c/‖g‖. The update direction is unchanged, only its length is capped. Clipping by value clamps each coordinate to [−c, c], which changes the direction (large coordinates are cut more than small ones) and so can push the update in a direction that no longer decreases the loss as quickly." },
      { t: "qa", level: "intermediate", q: "Derive the projection of b onto the column space of A.", a: "Minimise ‖b − Ac‖². The gradient is −2Aᵀ(b − Ac); setting it to zero gives the normal equations AᵀAc = Aᵀb, so c = (AᵀA)⁻¹Aᵀb (columns independent) and p = A(AᵀA)⁻¹Aᵀb. The residual b − p is orthogonal to every column. With orthonormal columns Q this simplifies to p = QQᵀb." },
      { t: "qa", level: "advanced", q: "What is the curse of dimensionality for nearest-neighbour search, quantitatively?", a: "For i.i.d. coordinates, ‖x − y‖² is a sum of d independent terms, so its mean grows like d while its standard deviation grows like √d; the relative spread is O(1/√d). For the uniform cube the ratio is √1.4/√d ≈ 1.18/√d. Nearest and farthest neighbours therefore become nearly equidistant, and small noise can reorder them. It does not doom learned embeddings, whose data lie near low-dimensional manifolds, but it does argue for metric learning or dimensionality reduction over raw high-dimensional features." },
      { t: "qa", level: "advanced", q: "How are the spectral norm, the Frobenius norm and Lipschitz continuity of a network related?", a: "For a linear layer, the smallest Lipschitz constant w.r.t. L2 is ‖W‖₂ = σ_max. Composing layers with 1-Lipschitz activations (ReLU, tanh) gives a network Lipschitz constant at most Π‖W_l‖₂. Since ‖W‖₂ ≤ ‖W‖_F, bounding Frobenius norms also bounds this, but loosely (by up to √rank). Spectral normalization divides each W by an estimate of σ_max (power iteration) to make each layer 1-Lipschitz, used for GAN discriminators and certified robustness." },
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
            text: "Cosine search recovers items 0 and 1 with similarity ≈ 0.99. Euclidean distance on raw vectors returns unrelated items (9755, 1314, …): the shrunken query sits near the origin, so the shortest database vectors are 'closest' regardless of direction. On unit vectors, Euclidean distance gives exactly the cosine ranking, as the identity predicts.",
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`**Span** = all linear combinations; **independent** = no vector is a combination of the others; **basis** = independent + spanning; **dimension** = size of any basis.`,
          String.raw`**Rank** = dimension of the column space = number of pivots; $\operatorname{rank}A + \dim\mathcal{N}(A) = n$; $\operatorname{rank}(AB) \le \min(\operatorname{rank}A, \operatorname{rank}B)$.`,
          String.raw`**Norm axioms:** definiteness, homogeneity, triangle inequality. $\|\mathbf{x}\|_p = (\sum|x_i|^p)^{1/p}$; $L_\infty = \max|x_i|$; $L_0$ counts non-zeros and is not a norm. $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt n\|\mathbf{x}\|_2$.`,
          String.raw`**Matrix norms:** $\|A\|_F = \sqrt{\sum A_{ij}^2}$; $\|A\|_2 = \sigma_{\max}$ = maximum stretch, submultiplicative.`,
          String.raw`**Cosine** $= \mathbf{x}^\top\mathbf{y}/(\|\mathbf{x}\|\|\mathbf{y}\|)$ ignores length; for unit vectors $\|\hat{\mathbf{a}}-\hat{\mathbf{b}}\|^2 = 2 - 2\cos\theta$.`,
          String.raw`**Distance concentration:** relative spread of random distances $\sim 1/\sqrt d$.`,
          String.raw`**Projection:** $\operatorname{proj}_\mathbf{a}\mathbf{b} = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\mathbf{a}$; onto $\mathcal{C}(A)$: $A(A^\top A)^{-1}A^\top\mathbf{b}$, or $QQ^\top\mathbf{b}$ with orthonormal $Q$; residual ⟂ subspace. Gram–Schmidt orthonormalises by subtracting projections.`,
          String.raw`**L1 vs L2:** soft-threshold $\operatorname{sign}(z)\max(|z|-\lambda, 0)$ (exact zeros) vs shrink $z/(1+\lambda)$; geometrically, corners of the diamond lie on the axes.`,
          "**In practice:** normalise embeddings for cosine search; clip gradients by global L2 norm; use weight decay (AdamW) for L2; use proximal steps or pruning for true L1 sparsity.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd) — the spectral norm, rank and the ellipse in the simulator are all explained by the singular value decomposition; it also gives PCA and low-rank compression.",
          "[Regularization](/learn/m07/regularization) — L1, L2/weight decay, dropout and early stopping in real training, building on the sparsity-vs-shrinkage argument here.",
          "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — where clipping by norm fits in a training loop.",
          "[Word Embeddings](/learn/m13/word-embeddings) — cosine similarity and normalisation applied to learned representations of words.",
          "[MLE and MAP](/learn/m02/mle-map) — why an L2 penalty is a Gaussian prior and an L1 penalty a Laplace prior.",
        ],
      },
    ],
  },
};

export default lesson;
