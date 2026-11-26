import type { Lesson } from "@/lib/types";

const SCRATCH_CODE = String.raw`import numpy as np
from sklearn.datasets import load_digits

np.set_printoptions(precision=4, suppress=True)

# ---------------- 1. 2x2 eigenvalues from the characteristic polynomial ----------------
def eig2x2(A):
    """Roots of  lambda^2 - tr(A) lambda + det(A) = 0  and one eigenvector per root."""
    tr, det = np.trace(A), np.linalg.det(A)
    disc = tr ** 2 / 4 - det
    if disc < 0:
        raise ValueError("complex eigenvalues: the map rotates every direction")
    lams = [tr / 2 + np.sqrt(disc), tr / 2 - np.sqrt(disc)]
    vecs = []
    for lam in lams:
        a, b = A[0]
        v = np.array([b, lam - a]) if abs(b) > 1e-12 else np.array([lam - A[1, 1], A[1, 0]])
        vecs.append(v / np.linalg.norm(v))          # (A - lam I) v = 0 from the first row
    return np.array(lams), np.array(vecs).T

A = np.array([[4.0, 1.0],
              [2.0, 3.0]])
lams, P = eig2x2(A)
print("eigenvalues:", lams, "| trace", np.trace(A), "= sum", lams.sum(), "| det", round(np.linalg.det(A), 6), "= product", round(lams.prod(), 6))
for lam, v in zip(lams, P.T):
    print(f"A v = {A @ v}  lambda v = {lam * v}")

# ---------------- 2. Diagonalization and matrix powers ----------------
D = np.diag(lams)
print("P D P^-1 == A:", np.allclose(P @ D @ np.linalg.inv(P), A))
A10 = P @ np.diag(lams ** 10) @ np.linalg.inv(P)     # A^10 needs only the eigenvalues to the 10th power
print("A^10 via eigen == matrix_power:", np.allclose(A10, np.linalg.matrix_power(A, 10)))

# ---------------- 3. Power iteration: the dominant eigenvector emerges ----------------
v = np.array([1.0, 0.0])
for k in range(1, 26):
    v = A @ v
    v /= np.linalg.norm(v)
    if k in (1, 2, 5, 25):
        print(f"iter {k:2d}: v = {v}, Rayleigh quotient v^T A v = {v @ A @ v:.6f}")

# ---------------- 4. Symmetric matrices: real eigenvalues, orthogonal eigenvectors ----------------
S = np.array([[2.0, 1.0],
              [1.0, 2.0]])
w, Q = np.linalg.eigh(S)                             # eigh: for symmetric/Hermitian matrices, ascending order
print("eigh eigenvalues:", w, "| Q^T Q == I:", np.allclose(Q.T @ Q, np.eye(2)),
      "| Q diag(w) Q^T == S:", np.allclose(Q @ np.diag(w) @ Q.T, S))

# ---------------- 5. SVD built from the eigen-decomposition of M^T M ----------------
M = np.array([[3.0, 0.0],
              [4.0, 5.0]])
evals, V = np.linalg.eigh(M.T @ M)                   # M^T M = V Sigma^2 V^T
order = np.argsort(evals)[::-1]
evals, V = evals[order], V[:, order]
sigma = np.sqrt(evals)
U = (M @ V) / sigma                                  # u_i = M v_i / sigma_i
print("M^T M =\n", M.T @ M, "\nsigma =", sigma)
print("U Sigma V^T == M:", np.allclose(U @ np.diag(sigma) @ V.T, M),
      "| matches np.linalg.svd:", np.allclose(sigma, np.linalg.svd(M, compute_uv=False)))
print("|det M| =", abs(np.linalg.det(M)), "= sigma1*sigma2 =", round(sigma.prod(), 6),
      "| condition number =", round(sigma[0] / sigma[-1], 4))

# ---------------- 6. Low-rank approximation and Eckart-Young ----------------
rng = np.random.default_rng(0)
B = rng.normal(size=(60, 8)) @ rng.normal(size=(8, 40)) + 0.1 * rng.normal(size=(60, 40))  # nearly rank 8
Ub, sb, Vbt = np.linalg.svd(B, full_matrices=False)
print("singular values:", np.round(sb[:10], 2))
for k in (2, 8, 20):
    Bk = Ub[:, :k] * sb[:k] @ Vbt[:k]                # sum of the top-k rank-1 pieces
    print(f"rank {k:2d}: ||B-Bk||_2 = {np.linalg.norm(B - Bk, 2):.4f} (sigma_k+1 = {sb[k]:.4f}), "
          f"||B-Bk||_F = {np.linalg.norm(B - Bk):.4f} (sqrt tail = {np.sqrt(np.sum(sb[k:] ** 2)):.4f}), "
          f"storage {k * (60 + 40 + 1)} vs {60 * 40}")

# ---------------- 7. PCA via SVD on the 8x8 digits dataset ----------------
X = load_digits().data                               # (1797, 64) pixel intensities 0..16
Xc = X - X.mean(axis=0)                              # PCA requires centred data
_, s, Vt = np.linalg.svd(Xc, full_matrices=False)
var = s ** 2 / (len(X) - 1)                          # eigenvalues of the covariance matrix
ratio = var / var.sum()
cum = np.cumsum(ratio)
print("explained variance ratio, first 5:", np.round(ratio[:5], 4))
for k in (2, 10, 21):
    Xk = Xc @ Vt[:k].T @ Vt[:k]                      # project onto k components and back
    print(f"k={k:2d}: cumulative variance {cum[k - 1]:.4f}, mean squared reconstruction error {np.mean((Xc - Xk) ** 2):.4f}")
print("components for 90% variance:", int(np.searchsorted(cum, 0.90) + 1))
cov_evals = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1]
print("SVD route == covariance eigenvalues:", np.allclose(var, cov_evals))`;

const SCRATCH_OUTPUT = String.raw`eigenvalues: [5. 2.] | trace 7.0 = sum 7.0 | det 10.0 = product 10.0
A v = [3.5355 3.5355]  lambda v = [3.5355 3.5355]
A v = [ 0.8944 -1.7889]  lambda v = [ 0.8944 -1.7889]
P D P^-1 == A: True
A^10 via eigen == matrix_power: True
iter  1: v = [0.8944 0.4472], Rayleigh quotient v^T A v = 5.000000
iter  2: v = [0.7894 0.6139], Rayleigh quotient v^T A v = 5.076923
iter  5: v = [0.7125 0.7016], Rayleigh quotient v^T A v = 5.007521
iter 25: v = [0.7071 0.7071], Rayleigh quotient v^T A v = 5.000000
eigh eigenvalues: [1. 3.] | Q^T Q == I: True | Q diag(w) Q^T == S: True
M^T M =
 [[25. 20.]
 [20. 25.]] 
sigma = [6.7082 2.2361]
U Sigma V^T == M: True | matches np.linalg.svd: True
|det M| = 15.0 = sigma1*sigma2 = 15.0 | condition number = 3.0
singular values: [71.13 58.78 56.02 46.05 39.77 38.43 35.7  28.21  1.27  1.17]
rank  2: ||B-Bk||_2 = 56.0226 (sigma_k+1 = 56.0226), ||B-Bk||_F = 102.0050 (sqrt tail = 102.0050), storage 202 vs 2400
rank  8: ||B-Bk||_2 = 1.2687 (sigma_k+1 = 1.2687), ||B-Bk||_F = 4.0588 (sqrt tail = 4.0588), storage 808 vs 2400
rank 20: ||B-Bk||_2 = 0.7300 (sigma_k+1 = 0.7300), ||B-Bk||_F = 2.2343 (sqrt tail = 2.2343), storage 2020 vs 2400
explained variance ratio, first 5: [0.1489 0.1362 0.1179 0.0841 0.0578]
k= 2: cumulative variance 0.2851, mean squared reconstruction error 13.4210
k=10: cumulative variance 0.7382, mean squared reconstruction error 4.9143
k=21: cumulative variance 0.9032, mean squared reconstruction error 1.8173
components for 90% variance: 21
SVD route == covariance eigenvalues: True`;

const TORCH_CODE = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

# ---- 1. The three decompositions ----
A = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
evals, evecs = torch.linalg.eig(A)                  # general matrices: complex dtype output
print("eig  :", evals, "| imag parts all 0:", bool((evals.imag == 0).all()))
R = torch.tensor([[0.0, -1.0], [1.0, 0.0]])          # 90-degree rotation
print("eig of rotation:", torch.linalg.eig(R)[0])    # +-i: no real eigenvector
S = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
w, Q = torch.linalg.eigh(S)                          # symmetric: real, ascending, orthonormal Q
print("eigh :", w, "| Q^T Q = I:", torch.allclose(Q.T @ Q, torch.eye(2)))
M = torch.tensor([[3.0, 0.0], [4.0, 5.0]])
U, s, Vh = torch.linalg.svd(M)
print("svd  : sigma =", s, "| U diag(s) Vh = M:", torch.allclose(U @ torch.diag(s) @ Vh, M, atol=1e-6))

# ---- 2. Train a small MLP on 8x8 digits ----
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)

model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(60):
    for i in range(0, len(Xtr), 64):
        loss = F.cross_entropy(model(Xtr[i:i + 64]), ytr[i:i + 64])
        opt.zero_grad(); loss.backward(); opt.step()

def accuracy(m):
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()

# ---- 3. Replace the 256x256 hidden layer by a rank-r factorisation W ~ (U_r S_r) V_r^T ----
def low_rank(linear, r):
    U, s, Vh = torch.linalg.svd(linear.weight.detach(), full_matrices=False)
    first = nn.Linear(linear.in_features, r, bias=False)       # x -> V_r^T x     (r x in)
    second = nn.Linear(r, linear.out_features)                 # -> U_r S_r (.)+b (out x r)
    with torch.no_grad():
        first.weight.copy_(Vh[:r])
        second.weight.copy_(U[:, :r] * s[:r])
        second.bias.copy_(linear.bias)
    return nn.Sequential(first, second)

W = model[2].weight.detach()
s_all = torch.linalg.svdvals(W)
print(f"\nfull model test acc {accuracy(model):.4f} | hidden layer params {W.numel() + 256}")
print("top singular values:", [round(v, 2) for v in s_all[:5].tolist()], "... smallest:", round(s_all[-1].item(), 4))
for r in [4, 8, 16, 32, 64]:
    compressed = nn.Sequential(*model[:2], low_rank(model[2], r), *model[3:])
    params = r * 256 + 256 * r + 256
    energy = (s_all[:r] ** 2).sum() / (s_all ** 2).sum()
    print(f"rank {r:3d}: params {params:6d} ({params / (W.numel() + 256):5.1%})  "
          f"energy kept {energy:.3f}  test acc {accuracy(compressed):.4f}")

# ---- 4. Spectral norm and signal growth through depth ----
n, depth = 256, 50
h0 = torch.randn(n)
for gain in [0.9, 1.0, 1.1]:
    h = h0.clone()
    for _ in range(depth):
        Wl = torch.randn(n, n) * gain / n ** 0.5             # entries with std gain/sqrt(n)
        h = Wl @ h
    print(f"gain {gain}: ||W||_2 of one layer ~ {torch.linalg.matrix_norm(Wl, 2).item():.2f}, "
          f"||h_50|| / ||h_0|| = {(h.norm() / h0.norm()).item():.3e}  (gain^50 = {gain ** depth:.3e})")

sn = nn.utils.parametrizations.spectral_norm(nn.Linear(256, 256))
for _ in range(10):
    sn(torch.randn(8, 256))                                   # each training-mode call runs one power-iteration step
print("spectral_norm-wrapped layer: largest singular value =",
      round(torch.linalg.matrix_norm(sn.weight.detach(), 2).item(), 4))`;

const TORCH_OUTPUT = String.raw`eig  : tensor([5.0000+0.j, 2.0000+0.j]) | imag parts all 0: True
eig of rotation: tensor([0.+1.j, 0.-1.j])
eigh : tensor([1., 3.]) | Q^T Q = I: True
svd  : sigma = tensor([6.7082, 2.2361]) | U diag(s) Vh = M: True

full model test acc 0.9741 | hidden layer params 65792
top singular values: [6.52, 5.87, 5.54, 5.31, 4.84] ... smallest: 0.0019
rank   4: params   2304 ( 3.5%)  energy kept 0.443  test acc 0.7037
rank   8: params   4352 ( 6.6%)  energy kept 0.660  test acc 0.9574
rank  16: params   8448 (12.8%)  energy kept 0.745  test acc 0.9741
rank  32: params  16640 (25.3%)  energy kept 0.803  test acc 0.9741
rank  64: params  33024 (50.2%)  energy kept 0.883  test acc 0.9741
gain 0.9: ||W||_2 of one layer ~ 1.76, ||h_50|| / ||h_0|| = 5.106e-03  (gain^50 = 5.154e-03)
gain 1.0: ||W||_2 of one layer ~ 1.98, ||h_50|| / ||h_0|| = 6.487e-01  (gain^50 = 1.000e+00)
gain 1.1: ||W||_2 of one layer ~ 2.15, ||h_50|| / ||h_0|| = 8.142e+01  (gain^50 = 1.174e+02)
spectral_norm-wrapped layer: largest singular value = 1.0002`;

const TF_CODE = String.raw`import numpy as np
import tensorflow as tf

A = tf.constant([[4.0, 1.0], [2.0, 3.0]])
print(tf.linalg.eig(A)[0].numpy())                 # complex dtype, eigenvalues 5 and 2

S = tf.constant([[2.0, 1.0], [1.0, 2.0]])
w, Q = tf.linalg.eigh(S)                            # ascending eigenvalues, orthonormal columns
print(w.numpy())

M = tf.constant([[3.0, 0.0], [4.0, 5.0]])
s, u, v = tf.linalg.svd(M)                          # NOTE the order (s, u, v) and that v is NOT transposed
print(s.numpy())
print(np.allclose((u @ tf.linalg.diag(s) @ tf.transpose(v)).numpy(), M.numpy(), atol=1e-5))

# Low-rank replacement of a trained Dense kernel (Keras stores it as (in, out), i.e. W transposed)
def factorize_dense(dense, r):
    kernel, bias = dense.get_weights()                       # kernel: (in, out)
    U, s, Vt = np.linalg.svd(kernel, full_matrices=False)
    first = tf.keras.layers.Dense(r, use_bias=False)
    second = tf.keras.layers.Dense(kernel.shape[1])
    first.build((None, kernel.shape[0])); second.build((None, r))
    first.set_weights([U[:, :r] * s[:r]])                    # (in, r)
    second.set_weights([Vt[:r], bias])                       # (r, out)
    return tf.keras.Sequential([first, second])`;

const EX_CODE = String.raw`import numpy as np
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg").mean(axis=2)         # (427, 640) grayscale, values 0..255
m, n = img.shape
U, s, Vt = np.linalg.svd(img, full_matrices=False)
energy = np.cumsum(s ** 2) / np.sum(s ** 2)

for k in [5, 20, 50, 100]:
    approx = (U[:, :k] * s[:k]) @ Vt[:k]                    # best rank-k approximation (Eckart-Young)
    rel_err = np.linalg.norm(img - approx) / np.linalg.norm(img)
    stored = k * (m + n + 1)
    print(f"k={k:3d}  energy {energy[k - 1]:.4f}  relative Frobenius error {rel_err:.4f}  "
          f"storage {stored:6d} numbers = {stored / (m * n):.1%} of the image")
k99 = int(np.searchsorted(energy, 0.99) + 1)
print("rank needed for 99% of the energy:", k99, "of", min(m, n))`;

const EX_OUTPUT = String.raw`k=  5  energy 0.9661  relative Frobenius error 0.1841  storage   5340 numbers = 2.0% of the image
k= 20  energy 0.9814  relative Frobenius error 0.1364  storage  21360 numbers = 7.8% of the image
k= 50  energy 0.9894  relative Frobenius error 0.1028  storage  53400 numbers = 19.5% of the image
k=100  energy 0.9946  relative Frobenius error 0.0736  storage 106800 numbers = 39.1% of the image
rank needed for 99% of the energy: 54 of 427`;

const lesson: Lesson = {
  id: "m02/eigen-svd",
  title: "Eigenvalues, Eigenvectors and SVD",
  summary:
    "Take a matrix apart to see what it does to space: the determinant as area scaling, eigenvectors as directions a matrix only stretches, diagonalization and the spectral theorem, positive definite matrices, and the SVD as rotate–scale–rotate. Then use it: low-rank approximation, PCA on real data, condition numbers, LoRA and compressing weight matrices.",
  minutes: 85,
  level: "intermediate",
  prerequisites: ["m02/vectors-matrices", "m02/norms-spaces"],
  next: ["m02/jacobians-hessians", "m02/optimization-math", "m04/gradient-problems", "m19/scaling-efficiency"],
  terms: ["determinant", "eigenvalue", "eigenvector", "characteristic-polynomial", "diagonalization", "spectral-theorem", "positive-definite", "singular-value-decomposition", "singular-value", "low-rank-approximation", "eckart-young", "pca", "condition-number"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`A matrix is a **linear map**: it moves every vector of the plane (or of $\mathbb{R}^n$, lists of $n$ numbers) somewhere new, by rotating, stretching, shearing, mirroring or flattening space. Staring at the numbers of a $4096\times4096$ weight matrix tells you almost nothing about which. **Matrix decompositions** fix this: they rewrite a matrix as a product of simple pieces with clear geometric meaning. Two matter most:`,
      },
      {
        t: "list",
        items: [
          String.raw`The **eigendecomposition** $A = P\Lambda P^{-1}$ finds the directions a square matrix only stretches (the **eigenvectors**) and by how much (the **eigenvalues**). It explains repeated application ($A^k$), whether a recurrence blows up, and how curved a loss surface is.`,
          String.raw`The **singular value decomposition (SVD)** $A = U\Sigma V^\top$ works for every matrix: any linear map is a rotation, a stretch along perpendicular axes, then another rotation. It gives the rank, the spectral norm, the best low-rank approximation, PCA and the condition number (how sensitive a problem is to small errors).`,
        ],
      },
      {
        t: "p",
        text: String.raw`In deep learning they are everywhere: PCA to visualise and compress data; spectral norms in [exploding and vanishing gradients](/learn/m04/gradient-problems); LoRA, which fine-tunes big models with **low-rank** weight updates ([Scaling and efficiency](/learn/m19/scaling-efficiency)); and shrinking trained layers with truncated SVD. The simulator below animates it all in 2-D.`,
      },
      {
        t: "callout",
        kind: "key",
        title: "The ideas to keep",
        text: String.raw`Eigenvector: $A\mathbf{v} = \lambda\mathbf{v}$ — a direction that stays on its own line. SVD: $A = U\Sigma V^\top$ — rotate ($V^\top$), stretch the axes by $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ ($\Sigma$), rotate ($U$). The unit circle becomes an ellipse whose half-axes have lengths $\sigma_i$. $\sigma_1$ is the spectral norm, the number of non-zero $\sigma_i$ is the rank, $|\det A| = \prod\sigma_i$, and keeping the $k$ largest gives the best rank-$k$ approximation.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — matrix products, $(AB)^\top = B^\top A^\top$, the inverse, the $2\times2$ determinant $ad - bc$, and that the columns of $A$ show where the basis vectors land.`,
          String.raw`[Vector Spaces, Norms and Distances](/learn/m02/norms-spaces) — span, basis, rank, orthonormal vectors ($Q^\top Q = I$), projections, Frobenius and spectral norms (here you learn *why* $\|A\|_2 = \sigma_{\max}$).`,
          String.raw`**Roots of a quadratic**: $\lambda = \tfrac{-b \pm\sqrt{b^2 - 4ac}}{2a}$; if the discriminant (the part under the root) is negative, the roots are complex.`,
          String.raw`**Variance and covariance** (for PCA): how spread out one feature is, and how two features move together ([Probability](/learn/m02/probability)).`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Kneading dough",
        text: "Stamp an ink circle on dough and knead it in one steady motion. The circle becomes an ellipse. The SVD redoes any such motion in three clean moves: turn the dough so the most- and least-stretched directions line up with the table edges (V transpose), stretch along those edges (Σ), then turn it into place (U). The ellipse's long and short axes are the singular values. Eigenvectors are lines on the dough that still point the same way afterwards — stretched or flipped, never turned.",
      },
      {
        t: "p",
        text: String.raw`An **eigenvector** of a square matrix is a direction the map does not rotate. The arrow $\mathbf{v}$ lands on its own line, just scaled by a number $\lambda$ (the Greek letter "lambda"), called its **eigenvalue**. If $\lambda = 2$ the arrow doubles; if $\lambda = 0.5$ it halves; if $\lambda = -1$ it flips.`,
      },
      {
        t: "p",
        text: String.raw`Not every map has such directions: a 45° rotation turns every line, so its eigenvalues are complex. But with a full set of eigenvectors, a matrix just scales each eigen-axis on its own. Apply it ten times and each eigen-direction is scaled by $\lambda^{10}$, so the largest $|\lambda|$ soon takes over — the same mechanism that makes signals explode or vanish through many layers.`,
      },
      {
        t: "p",
        text: String.raw`The **determinant** is how much the map scales areas (volumes in higher dimensions). 3 triples every area; a negative value also mirrors; zero squashes the plane onto a line, losing information for good. **Symmetric** matrices (equal to their transpose, like covariance matrices and Hessians) are the friendliest: real eigenvalues, perpendicular eigenvectors, and eigenvectors equal to singular vectors.`,
      },
      {
        t: "p",
        text: String.raw`**PCA** (principal component analysis) runs the SVD on mean-subtracted data. The right singular vectors are the directions where the data is most spread out; the singular values say how much. Keeping the first few gives the best low-dimensional summary (smallest squared error) — the same theorem (Eckart–Young) that justifies compressing a weight matrix by dropping small singular values.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "Finding eigenvalues and eigenvectors of a 2×2 matrix" },
      {
        t: "steps",
        items: [
          { title: "Write the condition", text: String.raw`$A\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \ne \mathbf{0}$ means $(A - \lambda I)\mathbf{v} = \mathbf{0}$ has a non-zero solution ($I$ is the identity), so $A - \lambda I$ must be singular (not invertible).` },
          { title: "Characteristic polynomial", text: "Set its determinant to zero. For 2×2 this is a quadratic in λ built from the trace (sum of the diagonal) and determinant.", tex: String.raw`\det(A - \lambda I) = \lambda^2 - \operatorname{tr}(A)\,\lambda + \det(A) = 0` },
          { title: "Solve for λ", text: "You get two roots (a repeated root counts twice). A negative discriminant means complex eigenvalues: no real eigenvector exists." },
          { title: "Find each eigenvector", text: String.raw`For each $\lambda$, solve $(A - \lambda I)\mathbf{v} = \mathbf{0}$. In 2-D one row is enough: if the first row is $(a - \lambda,\ b)$, then $\mathbf{v} = (b,\ \lambda - a)$ is perpendicular to it, so it solves the equation.` },
          { title: "Normalise (optional)", text: "An eigenvector has no fixed length or sign — any multiple works. Libraries return unit-length vectors." },
        ],
      },
      { t: "h", text: "Computing the SVD (conceptually)" },
      {
        t: "steps",
        items: [
          { title: "Form the Gram matrix AᵀA", tex: String.raw`A^\top A = V\Sigma^\top U^\top U\Sigma V^\top = V\,\Sigma^\top\Sigma\,V^\top` },
          { title: "Eigendecompose it", text: String.raw`$A^\top A$ is symmetric and positive semi-definite, so it has perpendicular unit eigenvectors $\mathbf{v}_i$ (the **right singular vectors**) and eigenvalues $\lambda_i \ge 0$.` },
          { title: "Singular values", tex: String.raw`\sigma_i = \sqrt{\lambda_i(A^\top A)}, \quad \text{sorted } \sigma_1 \ge \sigma_2 \ge \dots \ge 0` },
          { title: "Left singular vectors", text: String.raw`For each $\sigma_i > 0$: $\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$, automatically orthonormal; if $A$ is rank-deficient, add vectors to complete a basis.` },
          { title: "Assemble", tex: String.raw`A = U\Sigma V^\top = \sum_{i=1}^{r} \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top` },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        title: "This is how to understand the SVD, not how to compute it",
        text: String.raw`Forming $A^\top A$ squares the condition number and wipes out small singular values in floating point. Real routines (LAPACK, behind NumPy and PyTorch) reduce $A$ to bidiagonal form with orthogonal steps and never form $A^\top A$. Likewise, polynomial roots are numerically fragile, so beyond tiny sizes libraries find eigenvalues with the QR algorithm.`,
      },
      { t: "h", text: "PCA via the SVD" },
      {
        t: "steps",
        items: [
          { title: "Centre", text: String.raw`Subtract each feature's mean: $X_c = X - \mathbf{1}\boldsymbol{\mu}^\top$. Here $X \in \mathbb{R}^{n\times d}$ holds $n$ examples (rows) with $d$ features, and $\boldsymbol{\mu}$ is the vector of feature means.` },
          { title: "Thin SVD", tex: String.raw`X_c = U\Sigma V^\top, \qquad U \in \mathbb{R}^{n\times d},\ \Sigma, V \in \mathbb{R}^{d\times d}` },
          { title: "Read off the components", text: String.raw`The columns of $V$ (rows of $V^\top$) are the principal directions. The variance along direction $i$ is $\sigma_i^2/(n-1)$.` },
          { title: "Project", text: String.raw`The $k$-dimensional version of the data is $Z = X_cV_k = U_k\Sigma_k$ (keep the first $k$ columns).` },
          { title: "Reconstruct (optional)", text: String.raw`$\hat X = ZV_k^\top + \mathbf{1}\boldsymbol{\mu}^\top$. The reconstruction error equals the variance in the directions you threw away.` },
        ],
      },
      { t: "h", text: "Compressing a dense layer with truncated SVD" },
      {
        t: "p",
        text: String.raw`Take a trained weight $W \in \mathbb{R}^{m\times n}$, compute $W = U\Sigma V^\top$, and keep the top $r$ singular values with their vectors. Replace the layer $\mathbf{x} \mapsto W\mathbf{x} + \mathbf{b}$ by two thinner layers: first $\mathbf{x} \mapsto V_r^\top\mathbf{x}$ (no bias, $r\times n$), then $\mathbf{h} \mapsto (U_r\Sigma_r)\mathbf{h} + \mathbf{b}$ ($m\times r$). Parameters drop from $mn$ to $r(m+n)$ — a saving whenever $r < mn/(m+n)$. A short fine-tune afterwards can win back accuracy.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Linear maps, matrices and the determinant" },
      {
        t: "p",
        text: String.raw`A linear map $T:\mathbb{R}^n\to\mathbb{R}^m$ is fixed by where it sends the basis vectors: column $j$ of its matrix is $T(\mathbf{e}_j)$, and chaining maps multiplies matrices. In a new basis (columns of an invertible $P$) the **same** map is $P^{-1}AP$, a **similarity transform**; eigenvalues, trace and determinant stay the same.`,
      },
      {
        t: "p",
        text: String.raw`The **determinant** is the signed factor by which a square matrix scales $n$-dimensional volume. For $A = \begin{pmatrix} a & b\\ c & d\end{pmatrix}$, the unit square lands on the parallelogram spanned by the columns $(a, c)$ and $(b, d)$, with signed area $ad - bc$: positive if turning from $(a,c)$ to $(b,d)$ is counter-clockwise (like $\mathbf{e}_1$ to $\mathbf{e}_2$), negative if the map mirrors the plane. Key rules:`,
      },
      { t: "math", tex: String.raw`\det(AB) = \det A\,\det B, \qquad \det A^{-1} = \frac{1}{\det A}, \qquad \det A^\top = \det A, \qquad \det A = 0 \iff A \text{ singular} \iff \operatorname{rank}A < n` },
      {
        t: "p",
        text: String.raw`Volume scalings multiply when you chain maps, so stacked layers' Jacobian determinants multiply — the basis of normalising flows. For the matrices below: $\det\begin{pmatrix}4&1\\2&3\end{pmatrix} = 12 - 2 = 10$ and $\det\begin{pmatrix}3&0\\4&5\end{pmatrix} = 15$.`,
      },
      { t: "h", text: "Eigenvalues and eigenvectors" },
      {
        t: "math",
        tex: String.raw`A\mathbf{v} = \lambda\mathbf{v},\ \ \mathbf{v}\ne\mathbf{0} \quad\Longleftrightarrow\quad \det(A - \lambda I) = 0, \qquad \sum_i \lambda_i = \operatorname{tr}A, \qquad \prod_i \lambda_i = \det A`,
      },
      {
        t: "p",
        text: String.raw`$A \in \mathbb{R}^{n\times n}$ is square, $\lambda \in \mathbb{C}$ an eigenvalue (possibly complex), $\mathbf{v}$ its eigenvector. $p(\lambda) = \det(A - \lambda I)$ is the **characteristic polynomial**, of degree $n$, so there are exactly $n$ eigenvalues counting repeats (complex ones come in conjugate pairs for real $A$). They always add up to the trace and multiply to the determinant — compare the coefficients of $p$ with $\prod_i(\lambda_i - \lambda)$.`,
      },
      { t: "h", text: "Worked example: a 2×2 eigendecomposition" },
      {
        t: "steps",
        items: [
          { title: "Matrix", tex: String.raw`A = \begin{pmatrix} 4 & 1\\ 2 & 3\end{pmatrix}, \qquad \operatorname{tr}A = 7,\quad \det A = 10` },
          { title: "Characteristic polynomial", tex: String.raw`\lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0 \;\Rightarrow\; \lambda_1 = 5,\ \lambda_2 = 2` },
          { title: "Eigenvector for λ₁ = 5", tex: String.raw`A - 5I = \begin{pmatrix} -1 & 1\\ 2 & -2\end{pmatrix} \Rightarrow -v_1 + v_2 = 0 \Rightarrow \mathbf{v}_1 = (1, 1) \;\to\; (0.7071,\ 0.7071)` },
          { title: "Eigenvector for λ₂ = 2", tex: String.raw`A - 2I = \begin{pmatrix} 2 & 1\\ 2 & 1\end{pmatrix} \Rightarrow 2v_1 + v_2 = 0 \Rightarrow \mathbf{v}_2 = (1, -2) \;\to\; (0.4472,\ -0.8944)` },
          { title: "Check", tex: String.raw`A\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}5\\5\end{pmatrix} = 5\begin{pmatrix}1\\1\end{pmatrix}, \qquad A\begin{pmatrix}1\\-2\end{pmatrix} = \begin{pmatrix}2\\-4\end{pmatrix} = 2\begin{pmatrix}1\\-2\end{pmatrix}, \qquad 5 + 2 = 7,\ 5\cdot2 = 10` },
        ],
      },
      {
        t: "p",
        text: String.raw`The eigenvectors $(1,1)$ and $(1,-2)$ are **not** perpendicular ($1 - 2 = -1 \ne 0$), because $A$ is not symmetric. Two warnings: a rotation $R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$ has $p(\lambda) = \lambda^2 - 2\cos\theta\,\lambda + 1$ with discriminant $4\cos^2\theta - 4 < 0$ for $\theta \notin \{0, \pi\}$, so its eigenvalues $e^{\pm i\theta}$ are complex. A shear $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ has $\lambda = 1$ twice but only one eigen-direction, $(1, 0)$: it is **defective**.`,
      },
      { t: "h", text: "Diagonalization" },
      {
        t: "p",
        text: String.raw`If $A$ has $n$ independent eigenvectors, put them as the columns of $P$ and the eigenvalues on the diagonal of $\Lambda$ (capital lambda). Then $AP = P\Lambda$, so`,
      },
      { t: "math", tex: String.raw`A = P\Lambda P^{-1}, \qquad A^k = P\Lambda^kP^{-1}, \qquad A^k\mathbf{x} = \sum_i c_i\lambda_i^k\mathbf{v}_i \ \text{ where } \mathbf{x} = \sum_i c_i\mathbf{v}_i` },
      {
        t: "p",
        text: String.raw`In words: switch to eigen-coordinates, scale each, switch back; powers only touch the diagonal. For the example, $P = \begin{pmatrix}1&1\\1&-2\end{pmatrix}$, $P^{-1} = \frac{1}{-3}\begin{pmatrix}-2&-1\\-1&1\end{pmatrix} = \begin{pmatrix}0.6667&0.3333\\0.3333&-0.3333\end{pmatrix}$, and $A^{10} = P\,\operatorname{diag}(5^{10}, 2^{10})\,P^{-1}$ with $5^{10} = 9765625$, $2^{10} = 1024$:`,
      },
      { t: "math", tex: String.raw`A^{10} = \begin{pmatrix} \frac{2\cdot5^{10} + 2^{10}}{3} & \frac{5^{10} - 2^{10}}{3}\\[4pt] \frac{2\cdot5^{10} - 2\cdot2^{10}}{3} & \frac{5^{10} + 2\cdot2^{10}}{3}\end{pmatrix} = \begin{pmatrix} 6510758 & 3254867\\ 6509734 & 3255891\end{pmatrix}` },
      {
        t: "p",
        text: String.raw`The formula for $A^k\mathbf{x}$ is the key to **power iteration** and to deep networks. After many steps the part along the eigenvector with largest $|\lambda|$ dominates; the others shrink relative to it like $|\lambda_2/\lambda_1|^k$ (here $0.4^k$). The program below starts at $(1, 0)$ and reaches $(0.7071, 0.7071)$ within 25 steps. If $|\lambda_{\max}| > 1$ the vector grows exponentially; if all $|\lambda| < 1$ it decays to zero — the exploding/vanishing behaviour of a linear recurrence $\mathbf{h}_{t+1} = W\mathbf{h}_t$, like an RNN unrolled through time.`,
      },
      { t: "h", text: "Symmetric matrices and the spectral theorem" },
      {
        t: "math",
        tex: String.raw`A = A^\top \in \mathbb{R}^{n\times n} \;\Longrightarrow\; A = Q\Lambda Q^\top = \sum_{i=1}^n \lambda_i\,\mathbf{q}_i\mathbf{q}_i^\top, \qquad Q^\top Q = I,\ \ \lambda_i \in \mathbb{R}`,
      },
      {
        t: "p",
        text: String.raw`In words: a symmetric matrix is a sum of pieces, one per perpendicular unit direction $\mathbf{q}_i$ (the columns of orthogonal $Q$), each scaled by a real $\lambda_i$. **Why real?** If $A\mathbf{v} = \lambda\mathbf{v}$ with possibly complex $\mathbf{v}$, then $\bar{\mathbf{v}}^\top A\mathbf{v} = \lambda\|\mathbf{v}\|^2$ (the bar means complex conjugate). Taking the conjugate transpose and using that $A = A^\top$ is real gives the same quantity equal to $\bar\lambda\|\mathbf{v}\|^2$, so $\lambda = \bar\lambda$. **Why perpendicular?** For $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$, $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$ with $\lambda_1 \ne \lambda_2$: $\lambda_1\mathbf{v}_1^\top\mathbf{v}_2 = (A\mathbf{v}_1)^\top\mathbf{v}_2 = \mathbf{v}_1^\top A\mathbf{v}_2 = \lambda_2\mathbf{v}_1^\top\mathbf{v}_2$, so $(\lambda_1 - \lambda_2)\mathbf{v}_1^\top\mathbf{v}_2 = 0$ and $\mathbf{v}_1 \perp \mathbf{v}_2$. (Even with repeated eigenvalues you can still pick a perpendicular unit eigenbasis; that is the full spectral theorem.)`,
      },
      {
        t: "p",
        text: String.raw`Example: $S = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ has $p(\lambda) = \lambda^2 - 4\lambda + 3$, eigenvalues $3$ and $1$, and eigenvectors $(1,1)/\sqrt2$ and $(1,-1)/\sqrt2$, which are perpendicular. Geometrically, a symmetric matrix stretches space along perpendicular axes with no rotation. So its SVD and eigendecomposition coincide when all $\lambda_i \ge 0$ (in general $\sigma_i = |\lambda_i|$).`,
      },
      { t: "h", text: "Positive (semi-)definite matrices" },
      {
        t: "math",
        tex: String.raw`A \succeq 0 \;\iff\; \mathbf{x}^\top A\mathbf{x} \ge 0\ \ \forall\mathbf{x} \;\iff\; \lambda_i(A) \ge 0\ \forall i, \qquad A \succ 0 \;\iff\; \mathbf{x}^\top A\mathbf{x} > 0\ \ \forall\mathbf{x}\ne\mathbf{0} \;\iff\; \lambda_i > 0`,
      },
      {
        t: "p",
        text: String.raw`Read $A \succeq 0$ as "**positive semi-definite** (PSD)", $A \succ 0$ as "**positive definite** (PD)", $\forall$ as "for all". The tests agree because, for symmetric $A$ and $\mathbf{x} = Q\mathbf{y}$, $\mathbf{x}^\top A\mathbf{x} = \sum_i\lambda_iy_i^2$. Picture this **quadratic form** as a surface: a bowl (PD), a flat-bottomed trough (PSD with a zero eigenvalue), or a saddle (mixed signs). Every Gram matrix $B^\top B$ is PSD ($\mathbf{x}^\top B^\top B\mathbf{x} = \|B\mathbf{x}\|^2 \ge 0$), as is every covariance matrix and the **Hessian** (matrix of second derivatives) of a convex function.`,
      },
      {
        t: "p",
        text: String.raw`Why it matters for training: at a critical point of a loss, a PD Hessian means a local minimum, an indefinite one a saddle. Its largest-to-smallest eigenvalue ratio says how stretched the valley is, so how much gradient descent zig-zags. More in [Jacobians and Hessians](/learn/m02/jacobians-hessians) and [Optimization](/learn/m02/optimization-math).`,
      },
      { t: "h", text: "The singular value decomposition" },
      {
        t: "math",
        tex: String.raw`A = U\Sigma V^\top, \quad A \in \mathbb{R}^{m\times n},\ U \in \mathbb{R}^{m\times m},\ V \in \mathbb{R}^{n\times n} \text{ orthogonal},\ \Sigma = \operatorname{diag}(\sigma_1,\dots,\sigma_p) \in \mathbb{R}^{m\times n},\ \sigma_1 \ge \dots \ge \sigma_p \ge 0`,
      },
      {
        t: "p",
        text: String.raw`Here $p = \min(m, n)$; every real matrix has an SVD. The columns $\mathbf{v}_i$ of $V$ are the **right singular vectors**, the columns $\mathbf{u}_i$ of $U$ the **left singular vectors**, the $\sigma_i$ (sigma) the **singular values**, and $A\mathbf{v}_i = \sigma_i\mathbf{u}_i$: $A$ sends each $\mathbf{v}_i$ to $\mathbf{u}_i$, stretched by $\sigma_i$. Link to eigendecompositions:`,
      },
      { t: "math", tex: String.raw`A^\top A = V(\Sigma^\top\Sigma)V^\top, \qquad AA^\top = U(\Sigma\Sigma^\top)U^\top, \qquad \sigma_i = \sqrt{\lambda_i(A^\top A)}` },
      {
        t: "p",
        text: String.raw`**Geometric meaning.** Read $A\mathbf{x} = U(\Sigma(V^\top\mathbf{x}))$ right to left: $V^\top$ rotates (or reflects) the $\mathbf{v}_i$ onto the axes; $\Sigma$ stretches axis $i$ by $\sigma_i$ (dropping or adding dimensions if $m \ne n$); $U$ rotates the axes onto the $\mathbf{u}_i$. So the unit sphere becomes an ellipsoid with half-axes $\sigma_i\mathbf{u}_i$. Consequences: $\|A\|_2 = \sigma_1$ (the longest half-axis), $\|A\|_F = \sqrt{\sum_i\sigma_i^2}$, $\operatorname{rank}A$ = number of non-zero $\sigma_i$, and for square $A$, $|\det A| = \prod_i\sigma_i$ (rotations keep volume; only $\Sigma$ changes it).`,
      },
      { t: "h", text: "Worked example: SVD of a 2×2 matrix" },
      {
        t: "steps",
        items: [
          { title: "Matrix and Gram matrix", tex: String.raw`M = \begin{pmatrix}3&0\\4&5\end{pmatrix}, \qquad M^\top M = \begin{pmatrix}3&4\\0&5\end{pmatrix}\begin{pmatrix}3&0\\4&5\end{pmatrix} = \begin{pmatrix}25&20\\20&25\end{pmatrix}` },
          { title: "Eigenvalues of MᵀM", tex: String.raw`\lambda^2 - 50\lambda + (625 - 400) = \lambda^2 - 50\lambda + 225 = (\lambda - 45)(\lambda - 5) = 0` },
          { title: "Singular values", tex: String.raw`\sigma_1 = \sqrt{45} = 3\sqrt5 = 6.7082, \qquad \sigma_2 = \sqrt5 = 2.2361` },
          { title: "Right singular vectors (eigenvectors of MᵀM)", tex: String.raw`\mathbf{v}_1 = \tfrac{1}{\sqrt2}(1, 1), \quad \mathbf{v}_2 = \tfrac{1}{\sqrt2}(-1, 1) \qquad (V = \text{rotation by } 45^\circ)` },
          { title: "Left singular vectors", tex: String.raw`\mathbf{u}_1 = \frac{M\mathbf{v}_1}{\sigma_1} = \frac{(3, 9)/\sqrt2}{3\sqrt5} = \tfrac{1}{\sqrt{10}}(1, 3) = (0.3162,\ 0.9487), \quad \mathbf{u}_2 = \frac{M\mathbf{v}_2}{\sigma_2} = \frac{(-3, 1)/\sqrt2}{\sqrt5} = (-0.9487,\ 0.3162)` },
          { title: "Interpretation", text: String.raw`$V^\top$ rotates by $-45°$, $\Sigma$ stretches by 6.7082 and 2.2361, $U$ rotates by $\arctan(3) = 71.57°$. Checks: $\sigma_1\sigma_2 = \sqrt{225} = 15 = |\det M|$; $\sigma_1^2 + \sigma_2^2 = 50 = \|M\|_F^2$; condition number $\sigma_1/\sigma_2 = 3$.` },
        ],
      },
      { t: "h", text: "Low-rank approximation and the Eckart–Young theorem" },
      {
        t: "math",
        tex: String.raw`A_k = \sum_{i=1}^k\sigma_i\mathbf{u}_i\mathbf{v}_i^\top, \qquad \min_{\operatorname{rank}B\le k}\|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}, \qquad \min_{\operatorname{rank}B\le k}\|A - B\|_F = \|A - A_k\|_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2}`,
      },
      {
        t: "p",
        text: String.raw`The **Eckart–Young(–Mirsky) theorem**: keeping the $k$ largest singular pieces gives the best rank-$k$ approximation $A_k$ in both spectral and Frobenius norm, and the error is exactly the dropped singular values. Storing $A_k$ takes $k(m + n + 1)$ numbers instead of $mn$.`,
      },
      {
        t: "p",
        text: String.raw`For $M$ above, $A_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top = 6.7082\cdot\frac{1}{\sqrt{20}}\begin{pmatrix}1&1\\3&3\end{pmatrix} = \begin{pmatrix}1.5&1.5\\4.5&4.5\end{pmatrix}$, and $M - A_1 = \begin{pmatrix}1.5&-1.5\\-0.5&0.5\end{pmatrix}$ has spectral and Frobenius norm $2.2361 = \sigma_2$ (equal because the error has rank 1). The program below checks the theorem on a $60\times40$ matrix that is rank 8 plus noise: the first 8 singular values are 28–71, the 9th is 1.27, and the rank-8 error is exactly $\sigma_9 = 1.2687$ in spectral norm.`,
      },
      { t: "h", text: "PCA via SVD, worked on the Iris data" },
      {
        t: "p",
        text: String.raw`The Iris dataset has $n = 150$ flowers and $d = 4$ features (sepal length, sepal width, petal length, petal width, in cm). The feature means are $\boldsymbol\mu = (5.8433, 3.0573, 3.7580, 1.1993)$. After centring, the covariance matrix $C = X_c^\top X_c/(n-1)$ is`,
      },
      { t: "math", tex: String.raw`C = \begin{pmatrix} 0.6857 & -0.0424 & 1.2743 & 0.5163\\ -0.0424 & 0.1900 & -0.3297 & -0.1216\\ 1.2743 & -0.3297 & 3.1163 & 1.2956\\ 0.5163 & -0.1216 & 1.2956 & 0.5810\end{pmatrix}, \qquad \operatorname{tr}C = 4.5730` },
      {
        t: "p",
        text: String.raw`The SVD of $X_c$ gives singular values $(25.100, 6.013, 3.414, 1.885)$. Squared and divided by $n - 1 = 149$, they give each component's variance — the covariance eigenvalues — $(4.2282, 0.2427, 0.0782, 0.0238)$, summing to the trace $4.5730$. The **explained variance ratios** (each variance divided by the total) are $(0.9246, 0.0531, 0.0171, 0.0052)$: one direction carries 92.5% of the variance, two carry 97.8%.`,
      },
      {
        t: "p",
        text: String.raw`The first principal direction is $\mathbf{v}_1 = (0.3614, -0.0845, 0.8567, 0.3583)$, dominated by petal length. The first flower, $(5.1, 3.5, 1.4, 0.2)$, centres to $(-0.7433, 0.4427, -2.3580, -0.9993)$ and gets PC1 score $\mathbf{v}_1^\top\mathbf{x}_c = -0.2686 - 0.0374 - 2.0201 - 0.3580 = -2.6841$. (Each component's sign is arbitrary: a library may return $-\mathbf{v}_1$, flipping every score.)`,
      },
      {
        t: "math",
        tex: String.raw`\text{why it works:}\quad \operatorname{Var}(X_c\mathbf{w}) = \frac{\mathbf{w}^\top X_c^\top X_c\mathbf{w}}{n-1} = \mathbf{w}^\top C\mathbf{w} \;\le\; \lambda_1(C)\ \text{ for } \|\mathbf{w}\| = 1, \text{ with equality at } \mathbf{w} = \mathbf{v}_1`,
        caption: "The spread of the data along a unit direction w is wᵀCw. It is largest — equal to C's top eigenvalue — when w is the top right singular vector of the centred data.",
      },
      { t: "h", text: "Condition number" },
      {
        t: "math",
        tex: String.raw`\kappa(A) = \|A\|_2\,\|A^{-1}\|_2 = \frac{\sigma_{\max}}{\sigma_{\min}}, \qquad \frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(A)\,\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|} \ \text{ for } A\mathbf{x} = \mathbf{b}`,
      },
      {
        t: "p",
        text: String.raw`The **condition number** $\kappa$ (kappa) limits how much a small relative error in the data ($\delta\mathbf{b}$) can grow in the answer ($\delta\mathbf{x}$). $\kappa = 1$ for orthogonal matrices (perfect), $\infty$ for singular ones. Example: $C = \begin{pmatrix}1&1\\1&1.001\end{pmatrix}$ has $\sigma = (2.0005, 0.0005)$, so $\kappa = 4002$. $C\mathbf{x} = (2, 2.001)$ gives $\mathbf{x} = (1, 1)$, but $(2, 2.002)$ gives $\mathbf{x} = (0, 2)$ — a 0.05% change in $\mathbf{b}$ moves $\mathbf{x}$ by 100%. The Hessian's condition number controls how slowly plain gradient descent converges; normalisation layers and adaptive optimizers can be seen as ways to improve it.`,
      },
      { t: "h", text: "Spectral norm and exploding gradients" },
      {
        t: "p",
        text: String.raw`Backpropagation through $L$ linear layers multiplies the gradient by $W_L^\top, \dots, W_1^\top$ (and activation derivatives). Since $\|W^\top\mathbf{g}\| \le \sigma_1(W)\|\mathbf{g}\|$, the gradient norm is at most $\prod_l\sigma_1(W_l)$. But for random, roughly direction-neutral layers, typical growth per layer is about the initialisation **gain**, not $\sigma_1$. The PyTorch program below pushes a vector through 50 random $256\times256$ layers with entries of standard deviation $g/\sqrt{256}$: the norm ratio is $5.1\times10^{-3}$ for $g = 0.9$, $0.65$ for $g = 1.0$ and $81$ for $g = 1.1$ — close to $g^{50}$, though each layer's spectral norm is about $2g$. The linear-algebra answer to [vanishing and exploding gradients](/learn/m04/gradient-problems): initialisations that keep the gain near 1 ([Weight Initialization](/learn/m07/initialization)) and orthogonal matrices (all $\sigma_i = 1$).`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator applies an editable 2×2 matrix A to the plane. The grey grid, black curve and shaded parallelogram show where the square grid, unit circle and unit square land (accent when det > 0, clay when det < 0). In **I → A** mode, Play morphs the identity into A along M(t) = (1 − t)I + tA, with A's eigen-directions drawn as dotted lines. In **SVD** mode the stepper applies Vᵀ, Σ and U one at a time.",
      },
      { t: "sim", id: "linear-transform", caption: "Linear transformations, eigenvectors and the SVD in two dimensions." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          String.raw`**Eigen-lines stay put.** On *General* ($\lambda = 2.5, 1$), press **Play**. Every grid line moves, but the dotted eigen-lines stay and the black arrows slide along them, since $M(t)\mathbf{v} = (1 - t + t\lambda)\mathbf{v}$ whenever $A\mathbf{v} = \lambda\mathbf{v}$. They are not perpendicular.`,
          String.raw`**Reproduce the worked example.** *Lesson: [[4,1],[2,3]]* shows $\lambda_1 = 5$ along $(0.71, 0.71)$, $\lambda_2 = 2$ along $(0.45, -0.89)$, det 10. Drag x onto an eigen-line: Ax stays on it.`,
          String.raw`**Complex eigenvalues.** *Rotation 45°* reports $0.71 \pm 0.71i$ and no dotted lines — no direction escapes rotation. *Shear*: only one eigen-line (defective).`,
          String.raw`**Determinant as area.** On *General*, drag d from 1.5 toward 0.25: area and det fall together. Go negative and the parallelogram turns clay (orientation flipped). *Projection* gives det 0: the plane collapses to a line.`,
          String.raw`**SVD as rotate–scale–rotate.** *Lesson: [[3,0],[4,5]]*, **SVD** mode, Step: $V^\top$ rotates by $-45°$ (circle unchanged), $\Sigma$ stretches the axes to $6.708$ and $2.236$, $U$ rotates by $71.6°$ — matching the worked example.`,
          String.raw`**Symmetric matrices.** *Symmetric* ($[[2,1],[1,2]]$): the eigen-lines are perpendicular, and in SVD mode the arrows $\sigma_i\mathbf{u}_i$ lie along them: SVD = eigendecomposition. *Reflection* in SVD mode: det < 0, so U contains a reflection (the second axis passes through zero).`,
          String.raw`**Conditioning.** Set $a = 1, b = 1, c = 1, d = 1.1$ (nearly singular). The ellipse becomes a needle and the condition number grows large: tiny output changes match huge input changes.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "A from-scratch tour: eigenvalues from the characteristic polynomial, diagonalization and matrix powers, power iteration, the spectral theorem, the SVD built from the eigendecomposition of MᵀM, Eckart–Young checked numerically, and PCA via SVD on the 1797 images of the 8×8 digits dataset (bundled with scikit-learn, no download needed).",
      },
      { t: "code", lang: "python", title: "eigen_svd_from_scratch.py", code: SCRATCH_CODE, output: SCRATCH_OUTPUT },
      { t: "p", text: "Reading the output:" },
      {
        t: "list",
        items: [
          "The Rayleigh quotient vᵀAv estimates the eigenvalue from the current unit vector. It hits exactly 5 at step 1 only by luck of this start; normally it converges as the direction does.",
          "Use `np.linalg.eigh` for symmetric matrices: real eigenvalues, perpendicular unit eigenvectors, ascending order, and faster and more accurate than `np.linalg.eig`.",
          "Eckart–Young holds to every printed digit: the rank-k error is σ_{k+1} in spectral norm and the root of the sum of the dropped σ² in Frobenius norm.",
          "Digits PCA: two components hold only 28.5% of the variance (handwriting is not two-dimensional), 90% needs 21, and the mean squared reconstruction error drops from 13.42 to 1.82 grey-level units² between k = 2 and k = 21. Squared singular values over n − 1 equal the covariance eigenvalues, so both routes to PCA agree.",
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
                text: "`torch.linalg.eig` handles any square matrix and always returns complex tensors (a rotation really has complex eigenvalues); `torch.linalg.eigh` is for symmetric/Hermitian ones; `torch.linalg.svd` returns `U, S, Vh`, with `Vh` already transposed. The program then trains a 64→256→256→10 MLP on digits, swaps its 256×256 hidden layer for rank-r factorisations, and measures signal growth through 50 random layers and the effect of `spectral_norm`.",
              },
              { t: "code", lang: "python", title: "eigen_svd_torch.py", code: TORCH_CODE, output: TORCH_OUTPUT, runnable: false },
              {
                t: "p",
                text: "Result: rank 16 keeps the full test accuracy (0.9741) with 12.8% of the layer's parameters, though it keeps only 74.5% of the squared-singular-value 'energy'. What counts is the directions the next layer uses, not a perfect matrix copy. At rank 4 accuracy collapses to 0.70. In practice, pick r per layer by validation accuracy and often fine-tune briefly.",
              },
              {
                t: "callout",
                kind: "note",
                title: "LoRA in one line",
                text: String.raw`LoRA freezes a pretrained $W_0 \in \mathbb{R}^{m\times n}$ and learns $W = W_0 + \tfrac{\alpha}{r}BA$, with $B \in \mathbb{R}^{m\times r}$ (started at zero) and $A \in \mathbb{R}^{r\times n}$, where $r \ll \min(m, n)$ (Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"). The update has rank at most $r$ and only $r(m+n)$ trainable parameters: for $m = n = 4096$ and $r = 8$ that is 65,536 instead of 16,777,216 (0.39%). Unlike truncating after training, the low-rank factors are trained directly. See [Scaling and efficiency](/learn/m19/scaling-efficiency).`,
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow has `tf.linalg.eig`, `tf.linalg.eigh` and `tf.linalg.svd`. Two porting traps: `tf.linalg.svd` returns `(s, u, v)` in that order, with `v` **not** transposed (A = u·diag(s)·vᵀ), unlike NumPy/PyTorch's `Vh`. And Keras kernels are (in, out), the transpose of PyTorch's weight, which swaps the factor shapes when you factorise a layer.",
              },
              {
                t: "code",
                lang: "python",
                title: "eigen_svd_tf.py (not executed in CI)",
                code: TF_CODE,
                output: "Prints the eigenvalues [5.+0.j 2.+0.j] (order may vary); the eigh eigenvalues [1. 3.]; the singular values [6.7082 2.2361]; and True for the reconstruction check. factorize_dense returns a two-layer Sequential equivalent to the original Dense layer up to the rank-r truncation.",
                runnable: false,
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Decompositions compared" },
      {
        t: "table",
        head: ["Decomposition", "Form", "Exists for", "Strengths", "Limits / uses"],
        rows: [
          ["Eigendecomposition", String.raw`$A = P\Lambda P^{-1}$`, "square, diagonalizable", "explains powers $A^k$, recurrences, dynamics", "P may be ill-conditioned; complex values for real A; defective matrices have none"],
          ["Spectral (symmetric)", String.raw`$A = Q\Lambda Q^\top$`, "symmetric", "real λ, orthonormal Q, numerically stable", "Hessians, covariance matrices, graph Laplacians"],
          ["SVD", String.raw`$A = U\Sigma V^\top$`, "every matrix", "rank, norms, best low-rank approximation, pseudo-inverse", "cost O(mn·min(m,n)); compression, PCA, LoRA analysis"],
          ["Truncated / randomized SVD", String.raw`$A \approx U_k\Sigma_kV_k^\top$`, "every matrix", "only the top k triplets; randomized methods scale to huge matrices", "approximate; PCA on large data (scikit-learn's randomized solver)"],
          ["QR", String.raw`$A = QR$`, "every matrix", "stable least squares, orthogonal bases", "no direct spectral information; orthogonal init"],
          ["Cholesky", String.raw`$A = LL^\top$`, "symmetric positive definite", "half the cost of LU; tests PD-ness", "sampling Gaussians, Gaussian processes, second-order methods"],
          ["Jordan form", String.raw`$A = PJP^{-1}$`, "every square matrix (over ℂ)", "handles defective matrices in theory", "numerically unstable; theory only"],
        ],
      },
      { t: "h", text: "PCA variants" },
      {
        t: "table",
        head: ["Variant", "Idea", "When to use"],
        rows: [
          ["PCA (covariance)", "SVD of centred data", "features on similar scales"],
          ["PCA (correlation / standardised)", "SVD after dividing each feature by its standard deviation", "features in different units"],
          ["Randomized PCA", "random projection to a small subspace, then exact SVD there", "large n and d when you need only a few components"],
          ["Incremental PCA", "update the decomposition batch by batch", "data that does not fit in memory"],
          ["Kernel PCA", "PCA in a feature space defined by a kernel", "non-linear structure"],
          ["Linear autoencoder", "encoder/decoder trained with MSE", "spans the same subspace as PCA; see [Autoencoders](/learn/m14/autoencoders)"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Dimensionality reduction and visualisation.** PCA is the standard first look at embeddings and activations, often before t-SNE or UMAP.",
          String.raw`**Parameter-efficient fine-tuning.** LoRA (Hu et al., 2021) trains rank-$r$ updates $BA$ on frozen weights, because useful fine-tuning updates tend to have low effective rank.`,
          "**Model compression.** Truncated SVD splits large dense or embedding layers into two thin ones, alongside quantization and pruning ([Model Optimization](/learn/m18/model-optimization)).",
          String.raw`**Training stability.** Spectral normalization (Miyato et al., 2018) limits $\sigma_1(W)$ in GAN discriminators; orthogonal initialisation sets all singular values to 1; the largest Hessian eigenvalue explains why too-large learning rates $\eta$ diverge ($\eta > 2/\lambda_{\max}$ for gradient descent on a quadratic).`,
          "**Recurrent dynamics.** The eigenvalues of an RNN's recurrent weights decide whether its hidden state (and gradient) explodes or vanishes over time; see [Backpropagation Through Time](/learn/m09/bptt).",
          "**Recommender systems.** Factorising a user × item matrix, close to truncated SVD, was central in the Netflix Prize.",
          "**Latent semantic analysis.** Truncated SVD of a term × document matrix was an early way to capture word similarity, a forerunner of word embeddings.",
          "**Graph learning.** Graph Laplacian eigenvectors define spectral clustering and the spectral view of graph convolutions ([Graph Neural Networks](/learn/m15/graph-neural-networks)).",
          String.raw`**Whitening.** Multiplying centred data by $V\Sigma^{-1}$ removes feature correlations and equalises variances, a classic image preprocessing step.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**The SVD always exists and is numerically stable**; one computation gives rank, norms, condition number, pseudo-inverse and best low-rank approximations.",
          "**Eigendecompositions explain dynamics**: repeated application, stability of recurrences and gradient descent, Hessian curvature.",
          "**Optimality**: Eckart–Young makes truncated SVD and PCA provably the best linear compression in squared error.",
          "**Interpretable directions** you can inspect: singular vectors and principal components.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          String.raw`**Cost.** A full SVD of an $m\times n$ matrix costs $O(mn\min(m,n))$. For large matrices use truncated, randomized or iterative (Lanczos, power-iteration) methods.`,
          "**Linearity.** PCA finds only linear structure; curved data needs kernel PCA, autoencoders or other non-linear methods.",
          "**Scale sensitivity.** On raw features, PCA is dominated by features with the largest units; standardise when units differ.",
          "**Sign and order ambiguity.** Eigenvectors and singular vectors are fixed only up to sign (and any rotation within repeated eigenvalues), so components from two runs or libraries may not match element by element.",
          "**Non-normal matrices.** For non-symmetric matrices, eigenvalues can mislead about short-term behaviour: with all |λ| < 1, vectors can still grow for many steps if the eigenvectors are nearly parallel. Singular values give the worst-case one-step stretch.",
          "**Energy is not accuracy.** Keeping 90% of a weight matrix's squared singular values does not guarantee 90% of model accuracy — measure the task metric.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Confusing eigenvalues with singular values.** They agree (up to sign) only for symmetric (more generally, normal) matrices: the shear [[1,1],[0,1]] has eigenvalues 1 and 1 but singular values 1.618 and 0.618. **Fix:** singular values for norms, rank and conditioning; eigenvalues for powers and dynamics.",
          "**Forgetting to centre data before PCA.** The first component then points at the mean, not along the greatest spread. **Fix:** subtract column means (scikit-learn's PCA does; a hand-written SVD does not).",
          "**Mixing up SVD output conventions.** NumPy/PyTorch return `Vh` (= Vᵀ, rows are singular vectors); TensorFlow returns `v` (columns). The wrong one silently rebuilds garbage. **Fix:** check `U @ diag(S) @ Vh ≈ A` once.",
          "**Using `eig` on a symmetric matrix.** Rounding can give tiny imaginary parts, unsorted values and non-perpendicular vectors. **Fix:** use `eigh`.",
          "**Computing the SVD via eig(AᵀA) in practice.** This squares the condition number; small singular values lose half their accurate digits. **Fix:** call the SVD routine.",
          "**Reading meaning into component signs.** A PCA score of −2.68 versus +2.68 is a library convention. **Fix:** compare subspaces or absolute loadings, or fix a sign convention.",
          "**Inverting an ill-conditioned matrix.** κ = 10¹² can cost about 12 of float64's ~16 digits. **Fix:** check κ, regularise (add λI), or use a least-squares solver / pseudo-inverse that cuts off small σ.",
          "**Truncating a weight matrix without checking the task metric.** **Fix:** sweep r and measure accuracy, as in the PyTorch example; fine-tune after factorising.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is an eigenvector, intuitively and formally?", a: "Formally, a non-zero v with Av = λv; λ is its eigenvalue. Intuitively, a direction the linear map does not rotate — vectors along it are only stretched by λ (or flipped if λ < 0). The eigenvalues are the roots of det(A − λI) = 0." },
      { t: "qa", level: "beginner", q: "What does the determinant tell you geometrically?", a: "How much the map scales areas (volumes). |det A| = 2 doubles areas; a negative sign means orientation is flipped (a reflection is involved); det A = 0 means space is flattened into a lower dimension, so A is not invertible." },
      { t: "qa", level: "beginner", q: "What is the SVD and why is it useful?", a: "Every matrix factors as A = UΣVᵀ (orthogonal U, V; non-negative diagonal Σ): rotate, stretch along perpendicular axes, rotate. It reveals the rank (number of non-zero σ), spectral norm (σ₁) and condition number (σ₁/σ_min), and keeping the largest σ gives the best low-rank approximation — the basis of PCA and compression." },
      { t: "qa", level: "intermediate", q: "How are the SVD of A and the eigendecomposition of AᵀA related?", a: "AᵀA = VΣᵀΣVᵀ, so the right singular vectors of A are the eigenvectors of AᵀA, and σᵢ² are its eigenvalues. Likewise AAᵀ = UΣΣᵀUᵀ gives the left singular vectors. This is the standard way to derive the SVD, but not how it is computed numerically." },
      { t: "qa", level: "intermediate", q: "Why do symmetric matrices have real eigenvalues and orthogonal eigenvectors, and why does it matter in deep learning?", a: "If Av₁ = λ₁v₁ and Av₂ = λ₂v₂, then λ₁v₁ᵀv₂ = (Av₁)ᵀv₂ = v₁ᵀAv₂ = λ₂v₁ᵀv₂, so distinct eigenvalues force v₁ ⟂ v₂; a similar argument with complex conjugates shows λ is real. Hessians and covariance matrices are symmetric, so they split into independent curvature or variance directions along perpendicular axes — the basis of second-order analysis, PCA and conditioning." },
      { t: "qa", level: "intermediate", q: "Explain how PCA is computed with the SVD and how you choose the number of components.", a: "Centre the data and compute the thin SVD X_c = UΣVᵀ. The principal directions are the rows of Vᵀ, the component variances are σᵢ²/(n−1), and the projected data are U_kΣ_k. Choose k from the cumulative explained-variance ratio (e.g. 90–95%), from an elbow in the scree plot, or by the downstream metric." },
      { t: "qa", level: "intermediate", q: "What is the condition number and what happens when it is large?", a: "κ = σ_max/σ_min: it bounds how much relative errors grow when solving Ax = b, and how stretched the unit sphere's ellipse image is. Large κ means small data changes cause large solution changes. For Hessians, gradient descent needs a step small enough for the steepest direction, so progress along the flattest one is slow." },
      { t: "qa", level: "advanced", q: "State the Eckart–Young theorem and connect it to LoRA and to compressing weights.", a: "For any matrix with SVD UΣVᵀ, the truncation A_k = Σ_{i≤k} σᵢuᵢvᵢᵀ minimises ‖A − B‖ over rank-k B in both spectral and Frobenius norm, with errors σ_{k+1} and √(Σ_{i>k}σᵢ²). Post-hoc compression truncates a trained weight's SVD this way. LoRA instead limits the *update* ΔW = BA to rank r and trains B, A directly, betting the needed change is low-dimensional; it uses r(m+n) trainable parameters instead of mn." },
      { t: "qa", level: "advanced", q: "Why can a matrix whose eigenvalues all have |λ| < 1 still cause large transient growth, and what does this mean for RNNs?", a: "For non-normal matrices the eigenvectors can be nearly parallel, so a vector written in the eigenbasis needs huge coefficients that cancel. Each power Aᵏ shrinks the components one by one, the cancellation breaks, and ‖Aᵏx‖ can grow by orders of magnitude before it decays. The one-step bound is the spectral norm σ₁, not the spectral radius. So eigenvalue-based stability arguments for RNNs are incomplete; orthogonal or unitary recurrent matrices (all σ = |λ| = 1) avoid the problem by design." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "What is the determinant of [[3, 1], [2, 4]]?", answer: 10, tolerance: 0.001, explain: "3·4 − 1·2 = 10: the map multiplies areas by 10 and keeps orientation." },
          { kind: "numeric", q: "The matrix [[4, 1], [2, 3]] has eigenvalues 5 and one other. What is the other? (Use the trace.)", answer: 2, tolerance: 0.001, explain: "The eigenvalues sum to the trace 7, so the other is 2 (check: 5·2 = 10 = det)." },
          { kind: "numeric", q: "What is the largest singular value of [[0, −2], [1, 0]]?", answer: 2, tolerance: 0.001, explain: "AᵀA = [[1, 0], [0, 4]], whose eigenvalues are 1 and 4, so σ = 2 and 1." },
          { kind: "numeric", q: "What is the condition number of diag(10, 0.1)?", answer: 100, tolerance: 0.01, explain: "σ_max/σ_min = 10/0.1 = 100." },
          { kind: "numeric", q: "A covariance matrix has eigenvalues 6, 3 and 1. What fraction of the variance does the first principal component explain?", answer: 0.6, tolerance: 0.001, explain: "6/(6 + 3 + 1) = 0.6." },
          { kind: "numeric", q: "What is the spectral-norm error of the best rank-1 approximation of diag(5, 3, 1)?", answer: 3, tolerance: 0.001, explain: "By Eckart–Young the error is σ₂ = 3." },
          { q: "Which matrix has no real eigenvectors?", options: ["[[2, 0], [0, 3]]", "[[0, −1], [1, 0]] (rotation by 90°)", "[[1, 1], [0, 1]] (shear)", "[[1, 0], [0, 0]] (projection)"], answer: 1, explain: "A 90° rotation turns every line; its eigenvalues are ±i. The shear has one real eigen-direction, the others two." },
          { q: "For a real symmetric matrix, which statement is always true?", options: ["Its eigenvalues are positive", "Its eigenvectors for distinct eigenvalues are orthogonal and its eigenvalues are real", "It is invertible", "Its singular values equal its eigenvalues including sign"], answer: 1, explain: "That is the spectral theorem. Eigenvalues can be negative or zero; singular values are |λ|." },
          { q: "In PCA via SVD of the centred data X_c = UΣVᵀ, the principal directions are…", options: ["the columns of U", "the rows of Vᵀ (columns of V)", "the diagonal of Σ", "the rows of X_c"], answer: 1, explain: "Right singular vectors live in feature space (dimension d); U holds per-example coordinates." },
        ],
      },
      {
        t: "exercise",
        title: "Spectral decomposition by hand",
        prompt: String.raw`Let $S = \begin{pmatrix}5&2\\2&2\end{pmatrix}$. (a) Find its eigenvalues and unit eigenvectors. (b) Check that $S = \lambda_1\mathbf{q}_1\mathbf{q}_1^\top + \lambda_2\mathbf{q}_2\mathbf{q}_2^\top$. (c) Is $S$ positive definite? What are its singular values and condition number?`,
        hint: "trace 7, determinant 6.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Eigenvalues", tex: String.raw`\lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1) = 0 \Rightarrow \lambda_1 = 6,\ \lambda_2 = 1` },
              { title: "Eigenvectors", tex: String.raw`S - 6I = \begin{pmatrix}-1&2\\2&-4\end{pmatrix} \Rightarrow \mathbf{q}_1 = \tfrac{1}{\sqrt5}(2, 1); \qquad S - I = \begin{pmatrix}4&2\\2&1\end{pmatrix} \Rightarrow \mathbf{q}_2 = \tfrac{1}{\sqrt5}(1, -2)` },
              { title: "Orthogonality", tex: String.raw`\mathbf{q}_1^\top\mathbf{q}_2 = \tfrac15(2 - 2) = 0` },
              { title: "Reconstruction", tex: String.raw`\frac{6}{5}\begin{pmatrix}4&2\\2&1\end{pmatrix} + \frac{1}{5}\begin{pmatrix}1&-2\\-2&4\end{pmatrix} = \frac15\begin{pmatrix}25&10\\10&10\end{pmatrix} = \begin{pmatrix}5&2\\2&2\end{pmatrix}` },
              { title: "Definiteness, SVD, conditioning", text: String.raw`Both eigenvalues are positive, so $S \succ 0$ (e.g. $\mathbf{x}^\top S\mathbf{x} = 5x_1^2 + 4x_1x_2 + 2x_2^2 > 0$ for $\mathbf{x}\ne\mathbf{0}$). For a symmetric PD matrix the SVD equals the eigendecomposition: $\sigma = (6, 1)$, $\kappa = 6$.` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Compress an image with truncated SVD",
        prompt: "Load scikit-learn's bundled sample image `china.jpg`, convert it to grayscale, and compute its SVD. For k = 5, 20, 50 and 100, report the fraction of squared-singular-value energy kept, the relative Frobenius error of the rank-k approximation, and the storage cost compared with the original. How many components keep 99% of the energy? Explain how energy and error are related.",
        hint: "Rank-k reconstruction: (U[:, :k] * s[:k]) @ Vt[:k]. Storing the factors takes k(m + n + 1) numbers.",
        solution: [
          { t: "code", lang: "python", title: "svd_image_compression.py", code: EX_CODE, output: EX_OUTPUT },
          {
            t: "p",
            text: String.raw`By Eckart–Young, $\|A - A_k\|_F^2 = \sum_{i>k}\sigma_i^2$, so the relative error is exactly $\sqrt{1 - \text{energy}}$: for $k = 50$, $\sqrt{1 - 0.9894} = 0.103$. Natural images have singular values that fall off fast (a few smooth large-scale patterns carry most of the energy), so 54 of 427 components keep 99% of the energy at about a fifth of the storage. The last 1% is fine texture and edges — which is why low-rank images look blurry even when the energy fraction is high.`,
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`**Determinant:** signed volume scaling; $\det(AB) = \det A\det B$; zero ⇔ singular. For 2×2: $ad - bc$.`,
          String.raw`**Eigen:** $A\mathbf{v} = \lambda\mathbf{v}$; roots of $\det(A - \lambda I)$; for 2×2, $\lambda^2 - \operatorname{tr}A\,\lambda + \det A = 0$; $\sum\lambda = \operatorname{tr}$, $\prod\lambda = \det$. Rotations have complex λ; shears can be defective.`,
          String.raw`**Diagonalization:** $A = P\Lambda P^{-1}$, $A^k = P\Lambda^kP^{-1}$; the largest $|\lambda|$ dominates repeated application (power iteration, exploding/vanishing).`,
          String.raw`**Spectral theorem:** symmetric $A = Q\Lambda Q^\top$, real λ, orthonormal eigenvectors. **PSD:** $\mathbf{x}^\top A\mathbf{x} \ge 0$ ⇔ all $\lambda \ge 0$; PD Hessian at a critical point ⇔ local minimum.`,
          String.raw`**SVD:** $A = U\Sigma V^\top$ for every matrix; rotate–scale–rotate; $\sigma_i = \sqrt{\lambda_i(A^\top A)}$; $\|A\|_2 = \sigma_1$, $\|A\|_F^2 = \sum\sigma_i^2$, rank = #(σ > 0), $|\det A| = \prod\sigma_i$.`,
          String.raw`**Eckart–Young:** $A_k = \sum_{i\le k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ is the best rank-$k$ approximation; errors $\sigma_{k+1}$ (spectral) and $\sqrt{\sum_{i>k}\sigma_i^2}$ (Frobenius).`,
          String.raw`**PCA:** centre, SVD, directions = rows of $V^\top$, variances $\sigma_i^2/(n-1)$, scores $U_k\Sigma_k$.`,
          String.raw`**Condition number** $\kappa = \sigma_{\max}/\sigma_{\min}$ limits how much errors grow.`,
          "**Deep learning:** LoRA learns rank-r updates; truncated SVD compresses layers (rank 16 kept full accuracy at 12.8% of the parameters in our example); spectral norms and initialisation gain control signal and gradient growth through depth.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Jacobians, Hessians and Computational Graphs](/learn/m02/jacobians-hessians) — the Hessian is a symmetric matrix whose eigenvalues describe the loss's curvature; this lesson's spectral theorem and definiteness tests are what you need to read it.",
          "[Optimization: Objectives, Convexity and Minima](/learn/m02/optimization-math) — convexity, saddle points and why the condition number controls gradient descent.",
          "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — products of Jacobians and their singular values in real networks.",
          "[Scaling and Efficiency](/learn/m19/scaling-efficiency) — LoRA and other parameter-efficient methods built on low-rank structure.",
        ],
      },
    ],
  },
};

export default lesson;
