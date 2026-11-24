import type { Lesson } from "@/lib/types";

const SPEED = String.raw`import time
import numpy as np

def best_of(fn, repeats=3):
    """Fastest of several runs, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 32))      # 200 points with 32 features
Y = rng.normal(size=(300, 32))      # 300 other points

# ---- Pairwise squared distances D[i, j] = sum_k (X[i,k] - Y[j,k])^2 ----
def dist_python(X, Y):              # three nested Python loops
    Xl, Yl = X.tolist(), Y.tolist()
    return [[sum((a - b) ** 2 for a, b in zip(x, y)) for y in Yl] for x in Xl]

def dist_row_loop(X, Y):            # one Python loop, NumPy inside
    D = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        D[i] = ((X[i] - Y) ** 2).sum(axis=1)
    return D

def dist_broadcast(X, Y):           # no Python loop: (200,1,32) - (1,300,32) -> (200,300,32)
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)

def dist_matmul(X, Y):              # ||x||^2 + ||y||^2 - 2 x.y : one matrix product
    return (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T

ref = np.array(dist_python(X, Y))
for name, fn in [("python loops", dist_python), ("row loop", dist_row_loop),
                 ("broadcast", dist_broadcast), ("matmul trick", dist_matmul)]:
    ok = np.allclose(np.asarray(fn(X, Y)), ref)
    print(f"distances  {name:13s}: {best_of(lambda: fn(X, Y)):8.2f} ms  correct={ok}")

# ---- Softmax over a batch of logits (1000 examples, 100 classes) ----
Z = rng.normal(size=(1000, 100))

def softmax_loop(Z):
    out = np.empty_like(Z)
    for i in range(Z.shape[0]):               # one row at a time
        e = np.exp(Z[i] - Z[i].max())
        out[i] = e / e.sum()
    return out

def softmax_vec(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))   # (1000,100) - (1000,1)
    return e / e.sum(axis=1, keepdims=True)

print("softmax same result:", np.allclose(softmax_loop(Z), softmax_vec(Z)))
t_loop, t_vec = best_of(lambda: softmax_loop(Z)), best_of(lambda: softmax_vec(Z))
print(f"softmax    row loop     : {t_loop:8.2f} ms")
print(f"softmax    vectorized   : {t_vec:8.2f} ms  ({t_loop / t_vec:.0f}x faster)")`;

const SPEED_OUT = String.raw`distances  python loops :   181.87 ms  correct=True
distances  row loop     :     2.45 ms  correct=True
distances  broadcast    :     9.84 ms  correct=True
distances  matmul trick :     0.91 ms  correct=True
softmax same result: True
softmax    row loop     :     2.96 ms
softmax    vectorized   :     1.02 ms  (3x faster)`;

const TOUR = String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ---------- 1. ufuncs: element-wise functions that run in compiled code ----------
x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
print("abs  :", np.abs(x))
print("exp  :", np.exp(x))
print("relu :", np.maximum(x, 0.0))          # a binary ufunc: compares x with 0 element by element
print("where:", np.where(x > 0, x, 0.1 * x))  # leaky ReLU without a loop

# ---------- 2. Reductions: collapse one or more axes ----------
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])                  # shape (2, 3)
print("sum all:", A.sum(), "| sum axis 0:", A.sum(axis=0), "| max axis 1:", A.max(axis=1),
      "| argmax axis 1:", A.argmax(axis=1))
print("keepdims=False:", A.sum(axis=1).shape, "| keepdims=True:", A.sum(axis=1, keepdims=True).shape)

# ---------- 3. Broadcasting patterns used in deep learning ----------
X = np.arange(12, dtype=float).reshape(4, 3)  # 4 examples x 3 features
b = np.array([10., 20., 30.])                 # one bias per feature, shape (3,)
print("bias add (4,3)+(3,) ->", (X + b).shape, "\n", X + b)

mu = X.mean(axis=0, keepdims=True)            # (1, 3): one mean per column
sd = X.std(axis=0, keepdims=True)             # (1, 3)
Xn = (X - mu) / sd                            # (4,3) - (1,3) -> (4,3)
print("column means:", mu, "| normalised column means:", Xn.mean(axis=0), "| stds:", Xn.std(axis=0))

u = np.array([1., 2., 3., 4.])[:, None]       # (4, 1) column
v = np.array([1., 10., 100.])[None, :]        # (1, 3) row
print("outer product (4,1)*(1,3) ->\n", u * v)
print("matches np.outer:", np.array_equal(u * v, np.outer(u, v)))

P = rng.normal(size=(5, 2))                   # 5 points in 2-D
Q = rng.normal(size=(3, 2))                   # 3 other points
diff = P[:, None, :] - Q[None, :, :]          # (5,1,2) - (1,3,2) -> (5,3,2)
D = np.sqrt((diff ** 2).sum(axis=-1))         # (5, 3) Euclidean distances
loopD = np.array([[np.sqrt(((p - q) ** 2).sum()) for q in Q] for p in P])
print("diff shape:", diff.shape, "| D shape:", D.shape, "| equals loops:", np.allclose(D, loopD))

# ---------- 4. The (n,1) + (n,) trap ----------
pred = np.array([[2.5], [0.0], [2.0]])        # (3, 1): a model's output column
target = np.array([3.0, -0.5, 2.0])           # (3,)
wrong = pred - target                         # (3,1) - (3,) -> (3,3) !
print("pred - target shape:", wrong.shape)
print("wrong MSE:", ((pred - target) ** 2).mean(), "| right MSE:", ((pred[:, 0] - target) ** 2).mean())

# ---------- 5. Incompatible shapes raise an error ----------
try:
    np.ones((3, 4)) + np.ones(3)
except ValueError as e:
    print("ValueError:", e)

# ---------- 6. einsum: name every axis, say which ones survive ----------
a, c = rng.normal(size=4), rng.normal(size=4)
M, N = rng.normal(size=(2, 3)), rng.normal(size=(3, 4))
print("dot    :", np.isclose(np.einsum("i,i->", a, c), a @ c))
print("matmul :", np.allclose(np.einsum("ik,kj->ij", M, N), M @ N))
print("transp.:", np.array_equal(np.einsum("ij->ji", M), M.T))
Bm, Cm = rng.normal(size=(5, 2, 3)), rng.normal(size=(5, 3, 4))
print("batch matmul:", np.einsum("bik,bkj->bij", Bm, Cm).shape,
      np.allclose(np.einsum("bik,bkj->bij", Bm, Cm), Bm @ Cm))
Qh = rng.normal(size=(2, 4, 6, 8))            # batch 2, 4 heads, 6 tokens, 8 dims per head
Kh = rng.normal(size=(2, 4, 6, 8))
scores = np.einsum("bhqd,bhkd->bhqk", Qh, Kh) / np.sqrt(8)
print("attention scores:", scores.shape, np.allclose(scores, Qh @ Kh.swapaxes(-1, -2) / np.sqrt(8)))

# ---------- 7. Memory: broadcast views vs materialised arrays ----------
row = np.arange(3.0)
view = np.broadcast_to(row, (1000, 3))        # looks like (1000, 3), stores 3 numbers
full = np.tile(row, (1000, 1))                # really stores 3000 numbers
print("broadcast_to strides:", view.strides, "| shares memory:", np.shares_memory(view, row))
print("tile nbytes:", full.nbytes, "| underlying row nbytes:", row.nbytes)
print("(200,1,32)-(1,300,32) materialises", np.empty((200, 300, 32)).nbytes, "bytes")`;

const TOUR_OUT = String.raw`abs  : [2.  0.5 0.  1.  3. ]
exp  : [ 0.1353  0.6065  1.      2.7183 20.0855]
relu : [0. 0. 0. 1. 3.]
where: [-0.2  -0.05  0.    1.    3.  ]
sum all: 21.0 | sum axis 0: [5. 7. 9.] | max axis 1: [3. 6.] | argmax axis 1: [2 2]
keepdims=False: (2,) | keepdims=True: (2, 1)
bias add (4,3)+(3,) -> (4, 3)
 [[10. 21. 32.]
 [13. 24. 35.]
 [16. 27. 38.]
 [19. 30. 41.]]
column means: [[4.5 5.5 6.5]] | normalised column means: [0. 0. 0.] | stds: [1. 1. 1.]
outer product (4,1)*(1,3) ->
 [[  1.  10. 100.]
 [  2.  20. 200.]
 [  3.  30. 300.]
 [  4.  40. 400.]]
matches np.outer: True
diff shape: (5, 3, 2) | D shape: (5, 3) | equals loops: True
pred - target shape: (3, 3)
wrong MSE: 3.3333333333333335 | right MSE: 0.16666666666666666
ValueError: operands could not be broadcast together with shapes (3,4) (3,)
dot    : True
matmul : True
transp.: True
batch matmul: (5, 2, 4) True
attention scores: (2, 4, 6, 6) True
broadcast_to strides: (0, 8) | shares memory: True
tile nbytes: 24000 | underlying row nbytes: 24
(200,1,32)-(1,300,32) materialises 15360000 bytes`;

const TORCH = String.raw`import torch

torch.manual_seed(0)

# The broadcasting rules are identical to NumPy's
print(torch.broadcast_shapes((4, 3), (3,)), torch.broadcast_shapes((4, 1), (1, 3)),
      torch.broadcast_shapes((8, 1, 6, 1), (7, 1, 5)))
try:
    torch.broadcast_shapes((3, 4), (3,))
except RuntimeError as e:
    print("RuntimeError:", e)

# nn.Linear adds its bias by broadcasting: (N, out) + (out,)
layer = torch.nn.Linear(3, 2)
X = torch.randn(4, 3)
manual = X @ layer.weight.T + layer.bias
print("Linear == X @ W.T + b:", torch.allclose(layer(X), manual))

# expand = broadcast view (stride 0, no copy); repeat = real copy
row = torch.arange(3.0)
e = row.expand(1000, 3)
r = row.repeat(1000, 1)
print("expand stride:", e.stride(), "same storage:", e.data_ptr() == row.data_ptr())
print("repeat stride:", r.stride(), "bytes:", r.numel() * r.element_size())

# einsum works the same way; here: attention scores for 2 sequences, 4 heads, 6 tokens, 8 dims
Q = torch.randn(2, 4, 6, 8)
K = torch.randn(2, 4, 6, 8)
scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / 8 ** 0.5
print("scores:", tuple(scores.shape), torch.allclose(scores, Q @ K.transpose(-1, -2) / 8 ** 0.5, atol=1e-6))
weights = scores.softmax(dim=-1)                       # each row of every head sums to 1
print("row sums all 1:", torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 6)))

# The (n,1)-(n,) trap: PyTorch's MSELoss warns about it, then broadcasts anyway
import warnings
pred, target = torch.tensor([[2.5], [0.0], [2.0]]), torch.tensor([3.0, -0.5, 2.0])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bad = torch.nn.functional.mse_loss(pred, target)
print("mse with (3,1) vs (3,):", round(bad.item(), 4), "| warning raised:", len(w) > 0)
print("mse after squeeze:", round(torch.nn.functional.mse_loss(pred.squeeze(1), target).item(), 4))`;

const TORCH_OUT = String.raw`torch.Size([4, 3]) torch.Size([4, 3]) torch.Size([8, 7, 6, 5])
RuntimeError: Attempting to broadcast a dimension of length 3 at -1! Mismatching argument at index 1 had (3,); but expected shape should be broadcastable to [3, 4]
Linear == X @ W.T + b: True
expand stride: (0, 1) same storage: True
repeat stride: (3, 1) bytes: 12000
scores: (2, 4, 6, 6) True
row sums all 1: True
mse with (3,1) vs (3,): 3.3333 | warning raised: True
mse after squeeze: 0.1667`;

const TF_CODE = String.raw`import tensorflow as tf

X = tf.reshape(tf.range(12, dtype=tf.float32), (4, 3))
b = tf.constant([10.0, 20.0, 30.0])
print((X + b).shape)                                  # broadcasting: (4, 3) + (3,) -> (4, 3)

mu = tf.reduce_mean(X, axis=0, keepdims=True)         # (1, 3)
print(((X - mu) / tf.math.reduce_std(X, axis=0, keepdims=True)).shape)

Q = tf.random.normal((2, 4, 6, 8))
K = tf.random.normal((2, 4, 6, 8))
scores = tf.einsum("bhqd,bhkd->bhqk", Q, K) / 8 ** 0.5   # attention scores
print(scores.shape)
print(tf.broadcast_to(b, (1000, 3)).shape)`;

const EX_CODE = String.raw`import numpy as np

def cosine_loop(E):
    """Cosine similarity of every pair of rows, with two Python loops."""
    n = len(E)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = E[i] @ E[j] / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))
    return S

def cosine_vec(E):
    """Same result with no loops: normalise rows, then one matrix product."""
    U = E / np.linalg.norm(E, axis=1, keepdims=True)   # (n, d) / (n, 1)
    return np.einsum("id,jd->ij", U, U)                # = U @ U.T

E = np.random.default_rng(0).normal(size=(6, 4))      # 6 embeddings of size 4
S1, S2 = cosine_loop(E), cosine_vec(E)
print("same:", np.allclose(S1, S2), "| shape:", S2.shape)
print("diagonal:", S2.diagonal().round(6))
i, j = np.unravel_index(np.argmax(S2 - 2 * np.eye(6)), S2.shape)   # ignore the diagonal
print(f"most similar pair: ({i}, {j}) with cosine {S2[i, j]:.4f}")`;

const EX_OUT = String.raw`same: True | shape: (6, 6)
diagonal: [1. 1. 1. 1. 1. 1.]
most similar pair: (1, 4) with cosine 0.7622`;

const lesson: Lesson = {
  id: "m00/vectorization",
  title: "Vectorization and Broadcasting",
  summary:
    "How to replace Python loops with whole-array operations (ufuncs, reductions, matrix products and einsum), measured on real computations, and the exact rules NumPy and PyTorch use to combine arrays of different shapes — with the deep-learning patterns that rely on them and the silent bugs they can cause.",
  minutes: 60,
  level: "beginner",
  prerequisites: ["m00/numpy-arrays"],
  next: ["m02/vectors-matrices", "m04/forward-propagation", "m10/scaled-dot-product-attention"],
  terms: ["vectorization", "ufunc", "reduction", "broadcasting", "keepdims", "einsum", "stride"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "**Vectorization** means writing a computation as operations on whole arrays instead of as a Python loop over individual numbers. Instead of `for i in range(n): c[i] = a[i] + b[i]` you write `c = a + b`. The loop still happens, but inside NumPy's compiled code, where it runs one to two orders of magnitude faster.",
      },
      {
        t: "p",
        text: "Real numbers from this lesson's benchmark (one CPU, fastest of 3 runs): computing all distances between 200 points and 300 other points (32 features each) took about 180–200 ms with nested Python loops, about 9–11 ms with broadcasting, and about 0.8–1.2 ms with a matrix-product formulation — roughly 170 to 270 times faster than the loops, with identical results. Deep-learning code is written this way throughout: a layer processes a whole batch with one matrix product, a softmax normalises every row of a batch at once, and attention compares every query with every key in one call.",
      },
      {
        t: "p",
        text: "Vectorization quickly runs into arrays of *different* shapes: adding one bias vector to every row of a batch, subtracting one mean per column, comparing every point with every other point. **Broadcasting** is the set of rules that says how NumPy (and PyTorch, TensorFlow, JAX — they all use the same rules) lines up such arrays and virtually repeats the smaller one without copying memory. The rules are short, but applying them by accident is one of the most common sources of silent bugs in deep-learning code, so this lesson treats them exactly.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The two ideas to keep",
        text: "(1) Push loops into array operations: ufuncs for element-wise maths, reductions for sums/means/maxima, `@` and `einsum` for products. (2) Broadcasting: right-align the shapes; on each axis the sizes must be equal or one of them must be 1; a size-1 (or missing) axis is stretched to match, without copying. Always know the shape you expect and check it.",
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          "[NumPy Arrays, Shapes and Indexing](/learn/m00/numpy-arrays) — you need `shape`, `ndim`, `reshape`, slicing, views versus copies, the `axis` argument of reductions, and the idea of **strides** (bytes to skip per axis). Broadcasting is explained here in terms of these.",
          "Basic Python loops and functions — we start from loop code and rewrite it.",
          "Optional: [Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) gives the mathematical view of dot products, matrix products and a short statement of the broadcasting rule. This lesson goes deeper on the programming side and does not repeat the maths of matrix multiplication.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A stamp instead of a pen",
        text: "Imagine filling in a 1,000-row spreadsheet column by hand: you write each cell with a pen, checking each one (that is a Python loop). Vectorization is having a stamp the height of the whole column: one press fills every cell. Broadcasting is the stamp's clever extra: if you need the *same* number added to every row, you do not carve 1,000 copies of it onto the stamp — the machine just re-reads the one number as it moves down. Nothing is duplicated; the one value is reused.",
      },
      {
        t: "p",
        text: "Why are loops slow? Each pass of a Python loop does a lot of bookkeeping: find the next object, check its type, look up how to add those types, create a new Python object for the result. The actual addition is a tiny fraction of the work. A NumPy operation does that bookkeeping **once** for the whole array, then runs a tight compiled loop over raw numbers.",
      },
      {
        t: "p",
        text: "Broadcasting is how NumPy decides what 'the same operation on every element' means when the two arrays are different shapes. Think of each array's shape as a row of labelled boxes, lined up **from the right** (the last axes always face each other). Each pair of boxes must either agree, or one of them must say 1 — meaning 'I have just one; reuse me'. If a pair disagrees and neither is 1, there is no sensible way to pair the elements, and NumPy refuses.",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "1. Universal functions (ufuncs)" },
      {
        t: "p",
        text: "A **ufunc** (*universal function*) is a NumPy function that works element by element on whole arrays, in compiled code: `np.add` (`+`), `np.multiply` (`*`), `np.exp`, `np.log`, `np.abs`, `np.sqrt`, `np.maximum`, comparison operators such as `>`, and many more. ReLU is `np.maximum(x, 0)`; a leaky ReLU is `np.where(x > 0, x, 0.1 * x)` — `np.where(cond, a, b)` picks from `a` where the condition is True and from `b` elsewhere. Binary ufuncs (two inputs) broadcast their inputs.",
      },
      { t: "h", text: "2. Reductions" },
      {
        t: "p",
        text: "A **reduction** combines many values into one along an axis: `sum`, `mean`, `max`, `min`, `argmax` (the *position* of the maximum), `std`, `prod`, `any`, `all`. As in the previous lesson, the named axis disappears from the shape. The argument **`keepdims=True`** keeps it as a size-1 axis instead: `A.sum(axis=1)` on a `(2, 3)` array has shape `(2,)`, but `A.sum(axis=1, keepdims=True)` has shape `(2, 1)`. That size-1 axis is exactly what broadcasting needs to divide each row by its own total.",
      },
      { t: "h", text: "3. Measuring the speed-up on real computations" },
      {
        t: "p",
        text: "The benchmark computes all pairwise squared distances $D_{ij} = \\sum_k (X_{ik} - Y_{jk})^2$ between 200 and 300 points with 32 features in four ways, and a softmax over a batch of 1,000 rows of 100 logits in two ways. Every version is checked against the loop result.",
      },
      { t: "code", lang: "python", title: "vectorization_benchmark.py (timings vary between runs)", code: SPEED, output: SPEED_OUT },
      {
        t: "list",
        items: [
          "**Python loops → one NumPy loop**: removing the inner two loops (`row loop`) gave about 75–85× in our runs. Most of the gain comes from moving the innermost loops into NumPy.",
          "**Broadcasting is not automatically the fastest**: the fully broadcast version built a `(200, 300, 32)` temporary array — 15,360,000 bytes — and was *slower* than the row loop. Creating and reading large temporaries costs memory bandwidth.",
          "**Reformulate into a matrix product when you can**: the identity $\\lVert x-y\\rVert^2 = \\lVert x\\rVert^2 + \\lVert y\\rVert^2 - 2x\\cdot y$ turns the work into one `X @ Y.T`, handled by an optimised, multi-threaded BLAS library. It was the fastest and needs only the `(200, 300)` output.",
          "**Softmax**: 1,000 short rows gave about 3× — the loop body was already NumPy, so only the per-row Python overhead was saved. Speed-ups are largest when the Python loop does little work per iteration.",
        ],
      },
      { t: "h", text: "4. The broadcasting algorithm" },
      {
        t: "steps",
        items: [
          { title: "Right-align the shapes", text: "Write both shapes with their last axes in the same column. `(4, 3)` and `(3,)` become `(4, 3)` over `(·, 3)`." },
          { title: "Treat missing axes as 1", text: "The shorter shape is padded on the left with 1s: `(3,)` behaves as `(1, 3)`." },
          { title: "Compare each column (any order; we go right to left)", text: "Equal sizes → keep that size. One of them is 1 → the result takes the other size, and the size-1 array is **stretched**. Different and neither is 1 → stop with `ValueError: operands could not be broadcast together with shapes ...`." },
          { title: "Result shape", text: "The larger size in every column: `(4, 3)`." },
          { title: "Stretch without copying", text: "NumPy gives every stretched axis a stride of 0, so moving along it re-reads the same element. `np.broadcast_to(row, (1000, 3))` has strides `(0, 8)` and shares memory with the 3-element `row`." },
          { title: "Apply the operation element-wise", text: "Result element `[i, j]` combines A at `[i, j]` with B at `[j]` — each array's index is taken from the trailing positions of the result index, with 0 on every axis where that array has size 1." },
        ],
      },
      { t: "h", text: "Worked shape tables" },
      {
        t: "table",
        head: ["A", "B", "Aligned (A over B)", "Column checks, right to left", "Result"],
        rows: [
          ["`(4, 3)`", "`(3,)`", "`4 3` / `· 3`", "3 = 3 ✓; 4 vs missing (1) → 4 ✓", "`(4, 3)` — bias add"],
          ["`(4, 3)`", "`(1, 3)`", "`4 3` / `1 3`", "3 = 3 ✓; 4 vs 1 → 4 ✓", "`(4, 3)` — per-column normalise"],
          ["`(4, 1)`", "`(1, 3)`", "`4 1` / `1 3`", "1 vs 3 → 3 ✓; 4 vs 1 → 4 ✓", "`(4, 3)` — outer product"],
          ["`(3, 1)`", "`(3,)`", "`3 1` / `· 3`", "1 vs 3 → 3 ✓; 3 vs missing → 3 ✓", "`(3, 3)` — the trap"],
          ["`(3, 4)`", "`(3,)`", "`3 4` / `· 3`", "4 vs 3 ✗ — neither is 1", "**error**"],
          ["`(5, 1, 2)`", "`(1, 3, 2)`", "`5 1 2` / `1 3 2`", "2 = 2; 1 vs 3 → 3; 5 vs 1 → 5", "`(5, 3, 2)` — pairwise differences"],
          ["`(8, 1, 6, 1)`", "`(7, 1, 5)`", "`8 1 6 1` / `· 7 1 5`", "1→5; 6 vs 1 → 6; 1 vs 7 → 7; 8 vs missing → 8", "`(8, 7, 6, 5)`"],
        ],
      },
      { t: "h", text: "5. Broadcasting patterns in deep learning" },
      {
        t: "list",
        items: [
          "**Adding a bias row**: activations `(N, out)` + bias `(out,)` → `(N, out)`. Every dense layer (`X @ W.T + b`, see [Forward Propagation](/learn/m04/forward-propagation)) does this.",
          "**Normalising features per column**: `(X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)` — `(N, D)` op `(1, D)`. Image batches use `(N, C, H, W)` op `(1, C, 1, 1)` for per-channel statistics.",
          "**Outer products**: a column `(n, 1)` times a row `(1, m)` gives the `(n, m)` table of all products — the shape of a dense layer's weight gradient.",
          "**Pairwise differences and distances**: `x[:, None, :] - y[None, :, :]` turns `(n, d)` and `(m, d)` into `(n, 1, d)` and `(1, m, d)`, giving `(n, m, d)`; summing the squares over the last axis gives the `(n, m)` distance matrix used by k-nearest neighbours, clustering and contrastive losses. Indexing with `None` (also spelled `np.newaxis`) inserts a size-1 axis.",
          "**Masks**: attention masks of shape `(1, 1, T, T)` are added to scores of shape `(N, heads, T, T)` so one mask serves every example and head.",
          "**Row-wise softmax**: `z - z.max(axis=1, keepdims=True)` and `e / e.sum(axis=1, keepdims=True)` — `(N, K)` op `(N, 1)`.",
        ],
      },
      { t: "h", text: "6. The (n, 1) + (n,) trap" },
      {
        t: "p",
        text: "A model often outputs predictions of shape `(n, 1)` (one column), while targets are stored as `(n,)`. They look like 'the same n numbers', but broadcasting right-aligns them: `(n, 1)` against `(·, n)` gives **(n, n)** — every prediction minus every target. A mean-squared error on that matrix is a valid-looking number that is simply wrong (3.33 instead of 0.167 in the scratch program), and nothing crashes. Fix it with `pred[:, 0]`, `pred.squeeze(1)` or `target[:, None]`, and assert shapes before losses. PyTorch's `mse_loss` prints a warning for this case; NumPy says nothing.",
      },
      { t: "h", text: "7. einsum: index notation as code" },
      {
        t: "p",
        text: "`np.einsum` (*Einstein summation*) describes an operation by naming the axes of each input with letters and listing which letters the output keeps. **Rule: a letter that appears in the inputs but not in the output is summed over; letters in the output appear in that order.** So `\"i,i->\"` is a dot product, `\"ik,kj->ij\"` a matrix product, `\"ij->ji\"` a transpose, `\"bik,bkj->bij\"` a batch of matrix products, `\"i,j->ij\"` an outer product, and `\"bhqd,bhkd->bhqk\"` the attention scores of every head (`b` batch, `h` head, `q` query position, `k` key position, `d` feature). The scratch program checks every one against the explicit operation.",
      },
      { t: "h", text: "8. Memory: broadcast views versus materialised arrays" },
      {
        t: "p",
        text: "The stretched *input* of a broadcast is free (stride 0), but the *output* is a real array of the full broadcast shape. `np.broadcast_to(row, (1000, 3))` costs nothing; `np.tile(row, (1000, 1))` stores 24,000 bytes; and an intermediate like `(200, 1, 32) - (1, 300, 32)` allocates the full 15.36 MB. When an intermediate is much larger than the final answer, reduce early, reorganise into a matrix product, or process in chunks.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Broadcast shape" },
      {
        t: "p",
        text: "Let A have shape $(a_1, \\dots, a_r)$ and B shape $(b_1, \\dots, b_s)$. Let $R = \\max(r, s)$ and pad the shorter shape on the left with 1s so both have $R$ entries. Then:",
      },
      {
        t: "math",
        tex: String.raw`\text{compatible} \iff \forall k:\; a_k = b_k \;\lor\; a_k = 1 \;\lor\; b_k = 1, \qquad c_k = \max(a_k, b_k)`,
        caption: "$c = (c_1, \\dots, c_R)$ is the result shape. ([Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) states the same rule; here we use it to index elements.)",
      },
      { t: "h", text: "Which elements are combined" },
      {
        t: "math",
        tex: String.raw`C[i_1,\dots,i_R] = A[\tilde\imath_{R-r+1},\dots,\tilde\imath_R] \;\circ\; B[\hat\imath_{R-s+1},\dots,\hat\imath_R], \qquad \tilde\imath_k = \begin{cases} 0 & a_k = 1 \\ i_k & \text{otherwise}\end{cases}`,
        caption: "$\\circ$ is the element-wise operation (+, −, ×, …). Each input uses the trailing part of the output index, with 0 on its size-1 axes; $\\hat\\imath_k$ is defined the same way with $b_k$.",
      },
      {
        t: "p",
        text: "In stride terms, a size-1 axis that is stretched gets stride $t_k = 0$, so $\\text{offset} = \\sum_k i_k t_k$ ignores $i_k$ on that axis. This is the whole implementation of 'repeat without copying'.",
      },
      { t: "h", text: "Reductions and keepdims" },
      {
        t: "math",
        tex: String.raw`\Big(\sum_{\text{axis}=1} A\Big)_i = \sum_j A_{ij} \;\text{ has shape } (n,), \qquad \text{with keepdims: shape } (n, 1)`,
      },
      { t: "h", text: "Batch softmax" },
      {
        t: "math",
        tex: String.raw`\operatorname{softmax}(Z)_{ik} = \frac{e^{Z_{ik} - m_i}}{\sum_{k'} e^{Z_{ik'} - m_i}}, \qquad m_i = \max_{k} Z_{ik}`,
        caption: "$Z$ is $(N, K)$: N examples, K class scores (logits). Subtracting the row maximum $m_i$ does not change the result but prevents overflow of $e^{Z}$; $m$ has shape $(N, 1)$ with keepdims, so it broadcasts across each row.",
      },
      { t: "h", text: "Pairwise squared distances" },
      {
        t: "math",
        tex: String.raw`D_{ij} = \lVert x_i - y_j\rVert^2 = \sum_{k=1}^{d}(X_{ik} - Y_{jk})^2 = \lVert x_i\rVert^2 + \lVert y_j\rVert^2 - 2\,x_i \cdot y_j`,
        caption: "$X$ is $(n, d)$, $Y$ is $(m, d)$, $D$ is $(n, m)$. The right-hand form is $(n,1) + (1,m) - 2XY^\\top$: two broadcasts and one matrix product.",
      },
      { t: "h", text: "einsum" },
      {
        t: "math",
        tex: String.raw`\texttt{"ik,kj->ij"}: \; C_{ij} = \sum_k A_{ik}B_{kj}, \qquad \texttt{"bhqd,bhkd->bhqk"}: \; S_{bhqk} = \sum_d Q_{bhqd}K_{bhkd}`,
        caption: "Sum over every letter missing from the output.",
      },
      { t: "h", text: "Memory of a materialised broadcast" },
      { t: "math", tex: String.raw`\text{bytes} = b \prod_{k=1}^{R} c_k`, caption: "$b$ = bytes per element; the inputs' sizes do not matter, only the broadcast shape $c$." },
      { t: "h", text: "Worked example — every number" },
      {
        t: "steps",
        items: [
          { title: "Bias add, (4, 3) + (3,)", text: "With $X$ = `np.arange(12.).reshape(4, 3)` and $b = (10, 20, 30)$: aligned `4 3` / `· 3`, result `(4, 3)`. Element $[2, 1]$ uses $X[2, 1] = 7$ and $b[1] = 20$ (B has no row axis), so $C[2, 1] = 27$. Row 0 of the result is $(0{+}10, 1{+}20, 2{+}30) = (10, 21, 32)$." },
          { title: "Column normalise, (4, 3) − (1, 3)", tex: String.raw`\mu = (4.5, 5.5, 6.5),\quad \sigma_0 = \sqrt{\tfrac{(-4.5)^2 + (-1.5)^2 + 1.5^2 + 4.5^2}{4}} = \sqrt{\tfrac{45}{4}} = 3.3541,\quad \frac{X_{00}-\mu_0}{\sigma_0} = \frac{-4.5}{3.3541} = -1.3416` },
          { title: "Outer product, (4, 1) × (1, 3)", tex: String.raw`u = (1,2,3,4)^\top,\ v = (1, 10, 100):\quad (uv)_{21} = u_2 v_1 = 3 \times 10 = 30` },
          { title: "The trap, (3, 1) − (3,)", tex: String.raw`\hat y = (2.5, 0, 2),\ y = (3, -0.5, 2):\ \text{right MSE} = \tfrac{(-0.5)^2 + 0.5^2 + 0^2}{3} = \tfrac{0.5}{3} = 0.1667` },
          { title: "Trap, continued: the (3, 3) matrix", tex: String.raw`\hat y_i - y_j = \begin{pmatrix} -0.5 & 3 & 0.5 \\ -3 & 0.5 & -2 \\ -1 & 2.5 & 0 \end{pmatrix},\quad \text{"MSE"} = \tfrac{0.25+9+0.25+9+0.25+4+1+6.25+0}{9} = \tfrac{30}{9} = 3.3333` },
          { title: "Softmax of z = (1, 3, 0)", tex: String.raw`m = 3:\ (e^{-2}, e^{0}, e^{-3}) = (0.1353, 1, 0.0498),\ \text{sum} = 1.1851 \Rightarrow (0.1142, 0.8438, 0.0420)` },
          { title: "Distance identity, x = (1, 2), y = (3, 5)", tex: String.raw`(1-3)^2 + (2-5)^2 = 4 + 9 = 13 \quad=\quad \underbrace{5}_{\lVert x\rVert^2} + \underbrace{34}_{\lVert y\rVert^2} - 2\cdot\underbrace{13}_{x\cdot y} = 13` },
          { title: "einsum", tex: String.raw`\texttt{"i,i->"}\big((1,2,3),(4,5,6)\big) = 4 + 10 + 18 = 32, \qquad \texttt{"ik,kj->ij"}\Big(\begin{pmatrix}1&2\\3&4\end{pmatrix},\begin{pmatrix}5&6\\7&8\end{pmatrix}\Big) = \begin{pmatrix}19&22\\43&50\end{pmatrix}` },
          { title: "Memory", tex: String.raw`(200,1,32) - (1,300,32) \to 200\cdot300\cdot32\cdot 8 = 15{,}360{,}000 \text{ bytes};\qquad \text{the } (200, 300) \text{ answer: } 480{,}000 \text{ bytes}` },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator walks through broadcasting exactly as NumPy does it. Choose a preset or edit the shapes of A and B (1–3 axes, sizes 1–5) and the operation, then press **Step** (or **Play**). Step 1 right-aligns the shapes; step 2 checks the columns one at a time from the right; step 3 shows the result shape; step 4 draws both arrays stretched to that shape; step 5 shows the result — hover over or tap any result cell to see which elements of A and B produced it.",
      },
      { t: "sim", id: "broadcasting", caption: "Broadcasting step by step: faded dashed cells are virtual repeats (stride 0), not copies." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "**Bias add (4,3) + (3,)**: at step 2 the left column compares 4 with a missing axis. In step 4, B's single stored row is solid and rows 1–3 are faded: one bias row serves every example.",
          "**Column normalise (4,3) − (1,3)**: the same picture, but here B really has a size-1 axis (from `keepdims=True`). The result is identical to the bias case — a missing axis and a size-1 axis broadcast the same way.",
          "**Outer product (4,1) × (1,3)**: both arrays stretch, in different directions. Hover over the result: each cell is one A value times one B value.",
          "**The trap (4,1) + (4,)**: watch step 3 announce a (4, 4) result — 16 elements from two arrays of 4. This is the bug from the mechanism section, made visible.",
          "**Incompatible (3,4) + (3,)**: the walk stops at the first column with the clash and shows NumPy's real error message. Now change B to 1 axis of size 4, or to 2 axes (3, 1) — both fix it, with different meanings (add per column vs per row).",
          "**3-D (2,1,3) + (4,1)**: B is missing the first axis *and* has size 1 on the last. Count how many numbers are stored (6 and 4) versus how many results appear (24).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "One program covering every idea of the lesson, each checked against an explicit computation: ufuncs, reductions and keepdims, the four deep-learning broadcasting patterns, the (n,1)/(n,) trap, NumPy's error for incompatible shapes, five einsum expressions verified against `@`, `.T` and loops, and the memory difference between a broadcast view and a materialised array.",
      },
      { t: "code", lang: "python", title: "vectorization_and_broadcasting.py", code: TOUR, output: TOUR_OUT },
      { t: "p", text: "Reading the output:" },
      {
        t: "list",
        items: [
          "`np.where(x > 0, x, 0.1 * x)` computed a leaky ReLU for all five numbers in one call. `argmax(axis=1)` returned the *position* of each row's maximum (column 2 in both rows).",
          "The bias add printed the full `(4, 3)` result whose element `[2, 1]` is 27, as in the worked example. After column normalisation every column has mean 0 and standard deviation 1.",
          "`P[:, None, :] - Q[None, :, :]` produced a `(5, 3, 2)` array of differences; summing squares over the last axis gave the `(5, 3)` distance matrix, equal to the double loop.",
          "The trap produced shape `(3, 3)` and an MSE of 3.33 instead of 0.167.",
          "All einsum checks print True, including the attention scores `(2, 4, 6, 6)`: 2 sequences × 4 heads × 6 queries × 6 keys.",
          "`np.broadcast_to` has stride 0 on the stretched axis and shares memory with the 24-byte row; `np.tile` stores all 24,000 bytes.",
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
                text: "PyTorch uses the same broadcasting rules (`torch.broadcast_shapes` computes a result shape without any data), the same einsum notation, and has `expand` (a stride-0 broadcast view, like `np.broadcast_to`) versus `repeat` (a real copy, like `np.tile`). `nn.Linear` adds its bias by broadcasting, and `F.mse_loss` warns about the (n,1)/(n,) trap.",
              },
              { t: "code", lang: "python", title: "broadcasting_torch.py", code: TORCH, output: TORCH_OUT },
              {
                t: "callout",
                kind: "tip",
                text: "Use `expand` when you only need to *read* a repeated tensor; use `repeat` (or `.contiguous()` on the expanded tensor) only if you must write into each copy separately. Writing in place into an expanded tensor is an error because many positions share one memory location.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow follows the same broadcasting rules; reductions take `axis` and `keepdims` arguments (`tf.reduce_mean`, `tf.math.reduce_std`), and `tf.einsum` and `tf.broadcast_to` match their NumPy counterparts.",
              },
              {
                t: "code",
                lang: "python",
                title: "broadcasting_tf.py (not executed in CI)",
                code: TF_CODE,
                output: "Prints the shapes (4, 3), (4, 3), (2, 4, 6, 6) and (1000, 3).",
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Ways to vectorize" },
      {
        t: "table",
        head: ["Technique", "Example", "Advantages", "Disadvantages / when to avoid"],
        rows: [
          ["ufuncs", "`np.exp(x)`, `np.maximum(x, 0)`", "simple, fast, no extra memory beyond the output", "only element-wise work"],
          ["Reductions", "`x.sum(axis=1, keepdims=True)`", "one call per statistic", "choosing the wrong axis is a silent bug"],
          ["Broadcasting", "`x[:, None, :] - y[None]`", "expresses 'every pair' with no loop", "can materialise huge temporaries; accidental (n, n) results"],
          ["Matrix products", "`X @ W.T`, `X @ Y.T`", "uses optimised multi-threaded BLAS; usually fastest", "needs the problem rewritten as a product"],
          ["`np.einsum`", "`einsum('bhqd,bhkd->bhqk', Q, K)`", "readable, explicit axes; any contraction", "can be slower than `@` unless `optimize=True` is used for multi-operand expressions"],
          ["Masks / `np.where`", "`np.where(x > 0, x, 0)`", "replaces if/else inside loops", "computes both branches"],
          ["`np.vectorize`", "`np.vectorize(f)(x)`", "convenient syntax", "**not fast**: it is a Python loop internally"],
          ["JIT compilation (Numba, `torch.compile`, JAX `jit`)", "decorate a loop function", "fast loops for sequential logic", "extra dependency, compile time"],
        ],
      },
      { t: "h", text: "Broadcast view versus materialised copy" },
      {
        t: "table",
        head: ["NumPy", "PyTorch", "Memory", "Writable per element?"],
        rows: [
          ["`np.broadcast_to(a, shape)`", "`t.expand(*shape)`", "none (stride 0)", "no (read-only / shared)"],
          ["`np.tile(a, reps)`, `np.repeat`", "`t.repeat(*reps)`", "full copy", "yes"],
          ["implicit in `a + b`", "implicit in `a + b`", "only the output", "—"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Every neural-network layer**: dense layers are a matrix product plus a broadcast bias; activations are ufuncs; normalisation layers subtract and divide by broadcast statistics (see [Normalization](/learn/m07/normalization)).",
          "**Attention**: scores for all heads and positions are one batched product (`einsum('bhqd,bhkd->bhqk')` or `Q @ K.transpose(-1, -2)`), a broadcast mask is added, and a row-wise softmax with `keepdims` normalises them (see [Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention)).",
          "**Losses and metrics over a batch**: cross-entropy, accuracy (`(logits.argmax(1) == y).mean()`) and confusion matrices are reductions over vectorised element-wise results.",
          "**Similarity search and metric learning**: nearest-neighbour retrieval and contrastive losses compute all pairwise distances or cosine similarities with the matrix-product identity.",
          "**Data preprocessing**: per-channel image normalisation with `(1, C, 1, 1)` statistics, one-hot encoding with `labels[:, None] == np.arange(K)[None, :]`, and padding masks.",
          "**Gradient computations**: the weight gradient of a dense layer is a sum of outer products over the batch, computed as one matrix product `delta.T @ X`.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Speed**: 10× to several hundred times over Python loops in this lesson's measurements, with the same results.",
          "**Shorter, clearer code** once you read shapes fluently: a whole batch in one line.",
          "**Portability**: the same expressions run in PyTorch, TensorFlow and JAX on GPUs, where vectorised code is the only fast code.",
          "**Broadcasting avoids copies**: repeated operands cost no memory.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Memory blow-up**: broadcasting intermediates can be orders of magnitude larger than the result — the pairwise example allocated 32× more than its answer and ran slower than a partial loop.",
          "**Silent shape bugs**: broadcasting accepts many shapes you did not intend, like (n,1)+(n,).",
          "**Sequential dependencies cannot be vectorised directly**: when step t needs the result of step t−1 (an RNN's hidden state, a cumulative process), you loop over time and vectorise over the batch inside each step.",
          "**Numerical side effects of reformulation**: the matrix-product distance formula can return tiny negative numbers from rounding (catastrophic cancellation) when two points are almost equal; clip with `np.maximum(D, 0)` before a square root.",
          "**Readability can suffer**: long chains of `None` indexing and reshapes are hard to read; einsum or comments with shapes help.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**The (n,1) + (n,) explosion.** Predictions `(n, 1)` minus targets `(n,)` gives `(n, n)`; the loss is computed over all pairs. **Fix:** squeeze or index the column, and `assert pred.shape == target.shape` before computing a loss.",
          "**Forgetting `keepdims` and broadcasting along the wrong axis.** `Z / Z.sum(axis=1)` on a `(4, 3)` array fails (`(4, 3)` vs `(4,)`), and on a square `(3, 3)` array it *runs* but divides column j by row j's sum. **Fix:** `Z.sum(axis=1, keepdims=True)`.",
          "**Assuming broadcasting matches from the left.** `(3, 4) + (3,)` fails because shapes align from the right. To add one number per row, make the vector a column: `b[:, None]` (shape `(3, 1)`).",
          "**Materialising giant intermediates.** `X[:, None, :] - Y[None, :, :]` with 10,000 × 10,000 points of 128 float32 features needs 51.2 GB. **Fix:** use the matrix-product identity or process in chunks.",
          "**Using `np.vectorize` for speed.** It is a convenience wrapper around a Python loop. **Fix:** rewrite with ufuncs, `np.where` and reductions.",
          "**Reducing over the wrong axis in einsum or sum.** A misplaced letter silently sums over the wrong axis and may still produce a valid shape (e.g. with square matrices). **Fix:** verify against a slow explicit version on small random data with `np.allclose`, as the scratch program does.",
          "**Writing into a broadcast view.** `np.broadcast_to` returns a read-only array, and PyTorch forbids in-place writes into an `expand`ed tensor with overlapping memory. **Fix:** materialise with `.copy()` / `.clone()` / `repeat` first.",
          "**Timing without care.** The first call of a function can be slower (memory allocation, caches), and other programs disturb timings. **Fix:** repeat several times and keep the best, as the benchmark does; never compare a single run.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "What is vectorization and why is it faster than a Python loop?",
        a: "Expressing a computation as whole-array operations instead of an element-by-element Python loop. The interpreter's per-element overhead (fetching objects, type checks, dispatch, creating result objects) is paid once per operation instead of once per element, and the actual loop runs in compiled code over contiguous typed memory, often with SIMD instructions or multi-threaded BLAS for matrix products.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "State the broadcasting rules.",
        a: "Right-align the two shapes and treat missing leading axes as size 1. For each axis the sizes must be equal, or one of them must be 1; otherwise NumPy raises an error. The result size on each axis is the larger of the two, and an array with size 1 on an axis is virtually repeated along it (stride 0, no copy).",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What do (4, 3) + (3,), (4, 1) * (1, 3) and (3, 4) + (3,) produce?",
        a: "(4, 3) + (3,) → (4, 3): the vector is added to every row (a bias add). (4, 1) * (1, 3) → (4, 3): an outer product. (3, 4) + (3,) → error: the trailing sizes 4 and 3 differ and neither is 1. To add per row you would use a (3, 1) column.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why does `keepdims=True` matter for softmax over a batch?",
        a: "For logits `Z` of shape (N, K), `Z.max(axis=1)` has shape (N,), which right-aligns against K, not N: for N ≠ K it errors; for N = K it silently subtracts the wrong maxima. With `keepdims=True` the maxima and sums have shape (N, 1), which broadcasts across each row as intended.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "How would you compute all pairwise Euclidean distances between X (n, d) and Y (m, d) without loops? Compare two methods.",
        a: "Broadcasting: `np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(-1))` — simple and exact, but it materialises an (n, m, d) array. Matrix identity: `D2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T`, then `np.sqrt(np.maximum(D2, 0))` — memory O(nm), uses BLAS and is usually much faster (about 7–14× faster than the broadcast version in this lesson's benchmark runs), but can produce small negative values from cancellation, hence the clip.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Write attention scores for a batch of multi-head queries and keys with einsum.",
        a: "For Q and K of shape (B, H, T, d): `scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d)`, giving (B, H, T, T). `d` is missing from the output, so it is summed: each score is a dot product of one query with one key in the same batch element and head. It equals `Q @ K.swapaxes(-1, -2) / np.sqrt(d)`.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Broadcasting is said to be 'free'. When is it not, and what do you do about it?",
        a: "The stretched inputs are free (stride-0 views), but the output and any intermediate arrays are materialised at the full broadcast shape. An expression like `(X[:, None] - Y[None])**2` creates two (n, m, d) temporaries; for large n, m, d this costs gigabytes and memory bandwidth dominates, so it can be slower than a partial loop — as measured here (about 10 ms broadcast vs about 2.4 ms row loop vs about 1 ms matrix product). Remedies: reduce early, reformulate as a matrix product or einsum (which can avoid the intermediate), use in-place ufuncs with `out=`, process in chunks, or use a fused kernel (e.g. via `torch.compile`).",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Which computations cannot be vectorised across time, and how are they still made fast?",
        a: "Recurrences in which step t depends on step t−1, such as an RNN hidden state $h_t = f(W h_{t-1} + U x_t)$ or a cumulative process with non-linear updates. You cannot remove the loop over t, but you vectorise everything else: process the whole batch at each step, precompute the input projections $U x_t$ for all t in one matrix product, and run the loop in compiled code (cuDNN kernels, `torch.compile`, JAX `lax.scan`). Linear recurrences can sometimes be parallelised with a scan algorithm; Transformers avoid the recurrence entirely, which is part of why they train faster on GPUs.",
      },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "What is the shape of `np.ones((5, 1)) + np.ones((1, 4))`?",
            options: ["(5, 1)", "(1, 4)", "(5, 4)", "Error"],
            answer: 2,
            explain: "Right-aligned: 1 vs 4 → 4, 5 vs 1 → 5. Result (5, 4).",
          },
          {
            kind: "numeric",
            q: "How many elements does the result of broadcasting shapes (8, 1, 6, 1) and (7, 1, 5) have?",
            answer: 1680,
            tolerance: 0,
            explain: "The result shape is (8, 7, 6, 5), so 8·7·6·5 = 1,680 elements.",
          },
          {
            q: "Which pair of shapes **cannot** be broadcast together?",
            options: ["(3, 4) and (4,)", "(3, 4) and (3, 1)", "(3, 4) and (3,)", "(2, 3, 4) and (1, 4)"],
            answer: 2,
            explain: "(3,) aligns with the last axis 4; 3 ≠ 4 and neither is 1. The others are compatible.",
          },
          {
            kind: "numeric",
            q: "`pred` has shape (4, 1) and `target` has shape (4,). How many elements does `pred - target` have?",
            answer: 16,
            tolerance: 0,
            explain: "(4, 1) against (·, 4) broadcasts to (4, 4): 16 elements — the trap.",
          },
          {
            q: "What does `np.einsum('bij,bjk->bik', A, B)` compute?",
            options: ["An element-wise product", "A batch of matrix products", "A dot product of flattened arrays", "The transpose of A times B"],
            answer: 1,
            explain: "j is summed (missing from the output) for every batch index b: one matrix product per batch element, i.e. `A @ B`.",
          },
          {
            kind: "numeric",
            q: "How many MiB (2²⁰ bytes) does a float32 array of the broadcast shape of (256, 1, 128) and (1, 512, 128) occupy?",
            answer: 64,
            tolerance: 0.01,
            explain: "Result (256, 512, 128) = 16,777,216 elements × 4 bytes = 67,108,864 bytes = 64 MiB.",
          },
          {
            q: "Why do we write `Z - Z.max(axis=1, keepdims=True)` in a batch softmax?",
            options: [
              "keepdims makes the computation faster",
              "The (N, 1) maxima broadcast across each row; (N,) would align with the class axis instead",
              "keepdims sorts the logits",
              "It is required by np.exp",
            ],
            answer: 1,
            explain: "Shapes align from the right: (N, K) − (N,) compares K with N. The (N, 1) shape stretches along the K axis, one maximum per row.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Broadcast shapes and memory by hand",
        prompt: "For each pair, give the result shape or say that it fails: (a) (2, 3, 4) and (3, 1); (b) (6,) and (6, 1); (c) (5, 4) and (5,); (d) (1, 3, 1) and (2, 1, 4). Then (e): a batch of 512 query embeddings and 1,024 key embeddings, 64 float32 features each, is compared with `q[:, None, :] - k[None, :, :]`. How many bytes does the difference array occupy, and how many does the final (512, 1024) distance matrix need?",
        hint: "Right-align, compare from the right, take the larger size; for memory multiply the result shape by 4 bytes.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) (2, 3, 4) and (3, 1)", text: "Align `2 3 4` / `· 3 1`: 4 vs 1 → 4; 3 = 3; 2 vs missing → 2. Result **(2, 3, 4)**." },
              { title: "(b) (6,) and (6, 1)", text: "Align `· 6` / `6 1`: 6 vs 1 → 6; missing vs 6 → 6. Result **(6, 6)** — the trap again." },
              { title: "(c) (5, 4) and (5,)", text: "Align `5 4` / `· 5`: 4 vs 5, neither is 1 → **error**. Use `v[:, None]` to add per row." },
              { title: "(d) (1, 3, 1) and (2, 1, 4)", text: "1 vs 4 → 4; 3 vs 1 → 3; 1 vs 2 → 2. Result **(2, 3, 4)**." },
              { title: "(e) Memory", tex: String.raw`512 \cdot 1024 \cdot 64 \cdot 4 = 134{,}217{,}728 \text{ bytes} = 128 \text{ MiB}; \qquad 512 \cdot 1024 \cdot 4 = 2{,}097{,}152 \text{ bytes} = 2 \text{ MiB}` },
            ],
          },
          { t: "p", text: "The intermediate is 64 times larger than the answer — the matrix-product identity avoids it entirely." },
        ],
      },
      {
        t: "exercise",
        title: "Vectorize a cosine-similarity matrix",
        prompt: "The function `cosine_loop(E)` below fills an (n, n) matrix with the cosine similarity of every pair of rows of E using two Python loops. Write `cosine_vec(E)` with no Python loops, verify it gives the same matrix for 6 random embeddings of size 4 (seed 0), and report the most similar pair of distinct rows.",
        hint: "Divide each row by its length, using `np.linalg.norm(E, axis=1, keepdims=True)` (shape (n, 1)). The cosine of two unit vectors is their dot product, so all of them are one matrix product — or `einsum('id,jd->ij', U, U)`.",
        solution: [
          { t: "code", lang: "python", title: "cosine_similarity.py", code: EX_CODE, output: EX_OUT },
          {
            t: "p",
            text: "`keepdims=True` makes the norms `(6, 1)` so each row is divided by its own length. The diagonal is 1 (every vector is identical in direction to itself); subtracting `2 * np.eye(6)` pushes the diagonal below every real similarity so `argmax` finds the best distinct pair. This exact computation, on L2-normalised embeddings, is the core of similarity search and of contrastive losses.",
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          "**Vectorize**: replace Python loops with ufuncs, reductions, `@` and `einsum`. Measured here: roughly 170–270× faster for pairwise distances, ~3× for a row-wise softmax whose loop was already short.",
          "**ufuncs** work element by element; **reductions** collapse an axis; `keepdims=True` keeps it as size 1 so the result broadcasts back.",
          "**Broadcasting rules**: right-align, missing axes are 1, each pair must be equal or contain a 1, result = per-axis maximum, otherwise `ValueError`.",
          "Stretched axes have **stride 0** — no copy — but outputs and intermediates are full-size arrays.",
          "DL patterns: bias `(N, D) + (D,)`, normalise `(N, D) − (1, D)`, outer `(n, 1) × (1, m)`, pairwise `x[:, None, :] - y[None, :, :]`, softmax with `(N, 1)` statistics.",
          "The **(n, 1) + (n,) trap** silently produces (n, n). Assert shapes before losses.",
          "**einsum**: letters name axes; letters missing from the output are summed. Verify against explicit operations.",
          "Prefer matrix-product formulations for pairwise computations; watch memory: bytes = itemsize × product of the broadcast shape.",
        ],
      },
      { t: "math", tex: String.raw`c_k = \max(a_k, b_k) \text{ if } a_k = b_k \lor 1 \in \{a_k, b_k\}, \qquad \lVert x-y\rVert^2 = \lVert x\rVert^2 + \lVert y\rVert^2 - 2x\cdot y` },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "p",
        text: "With arrays, vectorization and broadcasting in hand you can read any deep-learning code. Next, [Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) gives the mathematics behind the matrix products used throughout this lesson. Then [Forward Propagation](/learn/m04/forward-propagation) uses exactly these patterns — a batch matrix product plus a broadcast bias — to run a neural network, and [Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) is the einsum `bhqd,bhkd->bhqk` plus a keepdims softmax you have already written.",
      },
    ],
  },
};

export default lesson;
