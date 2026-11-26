import type { Lesson } from "@/lib/types";

const SCRATCH_CODE = String.raw`import numpy as np

# ---------- 1. Building blocks written with explicit loops ----------
def dot(u, v):
    """Dot product: sum of element-wise products. Requires equal lengths."""
    assert len(u) == len(v), f"length mismatch: {len(u)} vs {len(v)}"
    total = 0.0
    for a, b in zip(u, v):
        total += a * b
    return total

def transpose(M):
    """Swap rows and columns: result[j][i] = M[i][j]."""
    rows, cols = M.shape
    T = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            T[j, i] = M[i, j]
    return T

def matmul(A, B):
    """C[i, j] = dot(row i of A, column j of B). Shapes (n, k) @ (k, m) -> (n, m)."""
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"inner dimensions differ: ({n}x{k}) @ ({k2}x{m})")
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = dot(A[i, :], B[:, j])
    return C

# ---------- 2. The worked example from the lesson ----------
A = np.array([[2., -1., 0.],
              [1.,  3., -2.]])          # shape (2, 3)
B = np.array([[1.,  2.],
              [0., -1.],
              [3.,  1.]])               # shape (3, 2)
C = matmul(A, B)
print("A @ B (loops):\n", C)
print("matches NumPy @ :", np.allclose(C, A @ B))
print("(AB)^T == B^T A^T:", np.allclose(transpose(C), transpose(B) @ transpose(A)))

try:
    matmul(A, A)                         # (2x3) @ (2x3) is undefined
except ValueError as e:
    print("shape error:", e)

# ---------- 3. Geometry of the dot product ----------
u = np.array([3., 4.])
v = np.array([4., 3.])
cos_theta = dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
print(f"u.v = {dot(u, v):.1f}, cos(theta) = {cos_theta:.2f}, theta = {np.degrees(np.arccos(cos_theta)):.2f} deg")

# ---------- 4. A dense layer y = W x + b, for one sample and for a batch ----------
W = np.array([[0.5, -1.0,  2.0],
              [1.5,  0.0, -0.5]])        # (out_features=2, in_features=3)
b = np.array([0.1, -0.2])                # (2,)
x = np.array([2.0, 1.0, 0.5])            # (3,)
print("W x + b =", W @ x + b)

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))              # batch of 4 samples, 3 features each
Y = X @ W.T + b                          # (4,3) @ (3,2) -> (4,2); b broadcasts over the batch
print("batch output shape:", Y.shape)
print("row 0 equals single-sample formula:", np.allclose(Y[0], W @ X[0] + b))

# ---------- 5. Identity and inverse ----------
M = np.array([[2., 1.],
              [1., 1.]])
M_inv = np.linalg.inv(M)
print("M^-1 =\n", M_inv)
print("M M^-1 == I:", np.allclose(M @ M_inv, np.eye(2)))

# ---------- 6. Broadcasting ----------
P = np.array([[1, 2, 3],
              [4, 5, 6]])                # (2, 3)
print("(2,3) + (3,)   ->\n", P + np.array([10, 20, 30]))
print("(2,3) + (2,1)  ->\n", P + np.array([[100], [200]]))

# ---------- 7. Tensors in deep learning ----------
images = rng.random(size=(8, 3, 32, 32))  # N x C x H x W: 8 RGB images of 32x32
print("image batch shape:", images.shape, "ndim:", images.ndim, "elements:", images.size)
channel_mean = images.mean(axis=(0, 2, 3))                     # one mean per channel -> (3,)
normalized = images - channel_mean.reshape(1, 3, 1, 1)         # broadcast over N, H, W
print("channel means are now ~0:", np.allclose(normalized.mean(axis=(0, 2, 3)), 0.0))
`;

const SCRATCH_OUTPUT = String.raw`A @ B (loops):
 [[ 2.  5.]
 [-5. -3.]]
matches NumPy @ : True
(AB)^T == B^T A^T: True
shape error: inner dimensions differ: (2x3) @ (2x3)
u.v = 24.0, cos(theta) = 0.96, theta = 16.26 deg
W x + b = [1.1  2.55]
batch output shape: (4, 2)
row 0 equals single-sample formula: True
M^-1 =
 [[ 1. -1.]
 [-1.  2.]]
M M^-1 == I: True
(2,3) + (3,)   ->
 [[11 22 33]
 [14 25 36]]
(2,3) + (2,1)  ->
 [[101 102 103]
 [204 205 206]]
image batch shape: (8, 3, 32, 32) ndim: 4 elements: 24576
channel means are now ~0: True`;

const TORCH_CODE = String.raw`import torch

torch.manual_seed(0)

# Same matrices as the worked example
A = torch.tensor([[2., -1., 0.],
                  [1.,  3., -2.]])
B = torch.tensor([[1.,  2.],
                  [0., -1.],
                  [3.,  1.]])
print("A @ B =\n", torch.matmul(A, B))          # identical to A @ B
print("shapes:", tuple(A.shape), "@", tuple(B.shape), "->", tuple((A @ B).shape))

# Dot product and angle
u = torch.tensor([3., 4.]); v = torch.tensor([4., 3.])
cos = torch.dot(u, v) / (u.norm() * v.norm())
print(f"dot = {torch.dot(u, v).item():.1f}, cos = {cos.item():.2f}")

# nn.Linear stores W with shape (out_features, in_features) and computes x W^T + b
layer = torch.nn.Linear(in_features=3, out_features=2)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[0.5, -1.0, 2.0],
                                     [1.5,  0.0, -0.5]]))
    layer.bias.copy_(torch.tensor([0.1, -0.2]))
x = torch.tensor([2.0, 1.0, 0.5])
print("nn.Linear(x)  =", layer(x).detach())
print("W @ x + b     =", (layer.weight @ x + layer.bias).detach())

# Batched input: (N, in) -> (N, out); the bias broadcasts across the batch
X = torch.randn(4, 3)
print("batch out shape:", tuple(layer(X).shape))

# Batched matrix multiply: a stack of 5 independent (2x3)@(3x4) products
P = torch.randn(5, 2, 3); Q = torch.randn(5, 3, 4)
print("bmm shape:", tuple(torch.bmm(P, Q).shape), "| matmul broadcasts too:", tuple((P @ Q[0]).shape))

# Identity and inverse
M = torch.tensor([[2., 1.], [1., 1.]])
print("inverse =\n", torch.linalg.inv(M))
print("M @ inv(M) == I:", torch.allclose(M @ torch.linalg.inv(M), torch.eye(2)))

# Image tensors are N x C x H x W in PyTorch
images = torch.rand(8, 3, 32, 32)
print("images:", tuple(images.shape), "| one image:", tuple(images[0].shape),
      "| flattened per sample:", tuple(images.flatten(start_dim=1).shape))
`;

const TORCH_OUTPUT = String.raw`A @ B =
 tensor([[ 2.,  5.],
        [-5., -3.]])
shapes: (2, 3) @ (3, 2) -> (2, 2)
dot = 24.0, cos = 0.96
nn.Linear(x)  = tensor([1.1000, 2.5500])
W @ x + b     = tensor([1.1000, 2.5500])
batch out shape: (4, 2)
bmm shape: (5, 2, 4) | matmul broadcasts too: (5, 2, 4)
inverse =
 tensor([[ 1., -1.],
        [-1.,  2.]])
M @ inv(M) == I: True
images: (8, 3, 32, 32) | one image: (3, 32, 32) | flattened per sample: (8, 3072)`;

const TF_CODE = String.raw`import numpy as np
import tensorflow as tf

A = tf.constant([[2., -1., 0.],
                 [1.,  3., -2.]])
B = tf.constant([[1.,  2.],
                 [0., -1.],
                 [3.,  1.]])
print(tf.matmul(A, B).numpy())                 # same as A @ B

# Keras Dense stores its kernel as (in_features, out_features) and computes x @ kernel + bias,
# i.e. the TRANSPOSE of PyTorch's nn.Linear weight layout.
layer = tf.keras.layers.Dense(2)
layer.build((None, 3))
W = np.array([[0.5, -1.0, 2.0],
              [1.5,  0.0, -0.5]], dtype=np.float32)   # (out, in), as in the lesson
layer.set_weights([W.T, np.array([0.1, -0.2], dtype=np.float32)])
x = tf.constant([[2.0, 1.0, 0.5]])             # a batch containing one sample
print(layer(x).numpy())

# Keras image layers default to channels_last: N x H x W x C
images = tf.random.uniform((8, 32, 32, 3))
print(images.shape, tf.transpose(images, perm=[0, 3, 1, 2]).shape)   # -> N x C x H x W
`;

const EX_CODE = String.raw`import numpy as np

def pairwise_sq_dists(X, Y):
    """D[i, j] = ||X[i] - Y[j]||^2 for X (n, d), Y (m, d), using only broadcasting."""
    diff = X[:, None, :] - Y[None, :, :]     # (n, 1, d) - (1, m, d) -> (n, m, d)
    return (diff ** 2).sum(axis=-1)          # (n, m)

def pairwise_sq_dists_matmul(X, Y):
    """Same result via ||x||^2 + ||y||^2 - 2 x.y — one matrix multiply, no (n, m, d) tensor."""
    x2 = (X ** 2).sum(axis=1)[:, None]       # (n, 1)
    y2 = (Y ** 2).sum(axis=1)[None, :]       # (1, m)
    return x2 + y2 - 2 * X @ Y.T             # (n, 1) + (1, m) - (n, m) -> (n, m)

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))
Y = rng.normal(size=(4, 3))

loops = np.array([[np.sum((x - y) ** 2) for y in Y] for x in X])
D1 = pairwise_sq_dists(X, Y)
D2 = pairwise_sq_dists_matmul(X, Y)
print("shape:", D1.shape)
print("broadcast == loops:", np.allclose(D1, loops))
print("matmul trick == loops:", np.allclose(D2, loops))
print("D[0] =", np.round(D1[0], 4))
`;

const EX_OUTPUT = String.raw`shape: (5, 4)
broadcast == loops: True
matmul trick == loops: True
D[0] = [1.8213 2.0528 1.9071 2.5712]`;

const lesson: Lesson = {
  id: "m02/vectors-matrices",
  title: "Scalars, Vectors, Matrices and Tensors",
  summary:
    "The objects deep learning computes with — scalars, vectors, matrices and tensors — and the operations on them: transpose, dot product, matrix multiplication, inverses and broadcasting. Matrix multiplication is the heart of every neural-network layer.",
  minutes: 55,
  level: "beginner",
  prerequisites: ["m00/numpy-arrays"],
  next: ["m02/norms-spaces", "m02/derivatives-chain-rule", "m02/eigen-svd", "m04/forward-propagation"],
  terms: ["scalar", "vector", "matrix", "tensor", "transpose", "dot-product", "matrix-multiplication", "identity-matrix", "inverse-matrix", "broadcasting"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "Everything a neural network touches — an input image, a word's embedding, a layer's weights, the gradient of the loss — is an array of numbers. **Linear algebra** is the maths of those arrays: how to name them, combine them, and what the results mean geometrically. This lesson covers the four kinds of array (scalars, vectors, matrices, tensors) and the few operations that do almost all of a model's arithmetic.",
      },
      {
        t: "p",
        text: String.raw`The star operation is **matrix multiplication**. A fully connected layer with 512 inputs and 256 outputs does $256 \times 512 = 131{,}072$ multiply-adds per example. Linear algebra writes the whole layer as one short expression, $\mathbf{h} = W\mathbf{x} + \mathbf{b}$ (input $\mathbf{x}$, weights $W$, bias $\mathbf{b}$, output $\mathbf{h}$). We can check its shape at a glance and hand it to hardware built for exactly this pattern (BLAS libraries on CPUs, Tensor Cores on GPUs).`,
      },
      {
        t: "p",
        text: "Matrices come from 19th-century work on systems of linear equations. Deep learning uses them because a layer *is* such a system (one equation per output neuron) followed by a non-linear bend. The same operations appear in dense layers, attention's $QK^\\top$ score matrix, embedding look-ups, convolutions, and batching, which lets one GPU call process hundreds of inputs.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The one idea to keep",
        text: String.raw`A matrix product $C = AB$ is a grid of dot products: entry $C_{ij}$ (row $i$, column $j$) is row $i$ of $A$ dotted with column $j$ of $B$. Another way to see it: $A$ is a transformation, and $AB$ applies it to every column of $B$. A neural-network layer is this operation plus a bias and a non-linearity.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      { t: "p", text: "This is the first maths lesson, so you need very little:" },
      {
        t: "list",
        items: [
          String.raw`**Arithmetic and the sum sign.** $\sum_{i=1}^{n} a_i b_i$ means "for $i$ from 1 to $n$, multiply matching pairs $a_i$ and $b_i$, then add them all up". ($\sum$ is the Greek capital sigma, "sum".) Every formula here is built from that pattern.`,
          "**Coordinate geometry.** Points as pairs $(x, y)$, Pythagoras for lengths, and the cosine of an angle. These give the dot product its geometric meaning.",
          "**Basic NumPy** ([NumPy Arrays, Shapes and Indexing](/learn/m00/numpy-arrays)): `np.array`, `.shape`, slicing like `A[i, :]`, and `@`. Only the code needs it.",
          "**Optional:** [Vectorization and Broadcasting](/learn/m00/vectorization) covers broadcasting from the coding side.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Shopping lists and price lists",
        text: "Three friends list how many apples, loaves and litres of milk they need: one row per friend gives matrix $A$ (3 friends × 3 items). Two shops list their prices: one column per shop gives matrix $B$ (3 items × 2 shops). Friend $i$'s bill at shop $j$ is (apples × apple price) + (bread × bread price) + (milk × milk price): row $i$ of $A$ dotted with column $j$ of $B$. All the bills form a 3 × 2 table: $C = AB$. The items are the shared 'inner' dimension — both tables must list the same items, or the product makes no sense.",
      },
      {
        t: "list",
        items: [
          "A **scalar** is a single number: a learning rate, a loss value, one pixel's brightness.",
          "A **vector** is an ordered list of numbers describing one thing: the 784 pixels of a small image, or a house's 3 features (size, age, rooms). Geometrically it is an arrow from the origin to a point.",
          "A **matrix** is a rectangular table of numbers: many vectors stacked as rows (a batch of houses), or a rule that transforms vectors (a layer's weights).",
          "A **tensor** is the same idea with more axes. A batch of colour images needs four: which image, which colour channel, which row, which column.",
        ],
      },
      {
        t: "p",
        text: "The **dot product** of two vectors measures how much they point the same way: large and positive if alike (e.g. embeddings of words used in similar contexts), zero if at right angles, negative if opposite. A neuron computes a dot product of its weights with its input, so it 'fires' strongly when the input looks like its weight pattern.",
      },
      {
        t: "p",
        text: "A matrix is also a machine that moves every point of space: it can rotate, stretch, shear or flatten. A layer is such a machine followed by a bend (the activation function); stacking layers chains the machines.",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "Objects and their shapes" },
      {
        t: "table",
        head: ["Object", "Axes (rank)", "Shape example", "Deep-learning example"],
        rows: [
          ["Scalar", "0", "`()`", "loss value, learning rate"],
          ["Vector", "1", "`(3,)`", "one example's features; a bias vector"],
          ["Matrix", "2", "`(64, 784)`", "a batch of 64 flattened 28×28 images; a weight matrix"],
          ["3-D tensor", "3", "`(32, 128, 768)`", "a batch of 32 sentences × 128 tokens × 768-number embeddings"],
          ["4-D tensor", "4", "`(16, 3, 224, 224)`", "16 RGB images, channels first (N × C × H × W)"],
        ],
      },
      {
        t: "p",
        text: "The **shape** lists each axis's length; the **rank** (or `ndim`) counts the axes. Careful: a vector with 3 entries is called '3-dimensional' (it lives in 3-D space), yet as an array it has one axis. Reading shapes fluently is the most useful skill here — most deep-learning bugs are shape bugs.",
      },
      { t: "h", text: "How a matrix product is computed" },
      {
        t: "steps",
        items: [
          { title: "Check the shapes", text: "For $A$ of shape $(n, k)$ and $B$ of shape $(k', m)$, the product exists only if $k = k'$ — the inner sizes match. If not, stop with an error." },
          { title: "Make room for the output", text: "The result $C$ takes the outer sizes: shape $(n, m)$." },
          { title: "Pick an output cell (i, j)", text: "Take row $i$ of $A$ (length $k$) and column $j$ of $B$ (also length $k$)." },
          { title: "Multiply matching entries", text: "Form the $k$ products $A_{i1}B_{1j}, A_{i2}B_{2j}, \\dots, A_{ik}B_{kj}$." },
          { title: "Add them up", text: "Keep a running sum; the final sum is $C_{ij}$. One multiply plus one add is called a **multiply-accumulate (MAC)**." },
          { title: "Repeat for every cell", text: "Do this for all $n \\cdot m$ cells: $n \\cdot m \\cdot k$ multiplications in total. The cells do not depend on each other, which is why GPUs can compute them in parallel." },
        ],
      },
      {
        t: "p",
        text: "Real libraries do not run three nested loops. Optimised **GEMM** (general matrix-matrix multiply) routines split matrices into tiles that fit in fast memory and reuse each tile for many multiply-adds. Same arithmetic, smarter order. That is why `A @ B` in NumPy can be thousands of times faster than a Python triple loop.",
      },
      { t: "h", text: "How the transpose is stored" },
      {
        t: "p",
        text: "In memory, a 2-D array is one flat block, stored **row-major** by default in NumPy and PyTorch: row 0, then row 1, and so on. Each array records **strides** — how many bytes to jump to move one step along each axis. Transposing copies nothing; it returns a view with shape and strides swapped. So `A.T` is free, but no longer **contiguous** (stored in order), which some operations need (e.g. PyTorch's `.view`); `.contiguous()` or `.reshape` copies when needed.",
      },
      { t: "h", text: "How broadcasting decides shapes" },
      { t: "p", text: "**Broadcasting** lets you combine arrays of different shapes, element by element, by stretching the smaller one. The steps:" },
      {
        t: "steps",
        items: [
          { title: "Line shapes up on the right", text: "Write both shapes right-aligned, e.g. `(4, 3)` and `(3,)` become `(4, 3)` and `(·, 3)`." },
          { title: "Fill missing axes with 1", text: "The shorter shape gets size-1 axes on the left: `(3,)` → `(1, 3)`." },
          { title: "Compare each axis", text: "Two sizes fit if they are equal or one of them is 1. Otherwise, error." },
          { title: "Stretch size-1 axes", text: "A size-1 axis is (virtually) repeated to match. No memory is copied; its stride is set to 0." },
          { title: "Apply the operation", text: "The result shape is the larger size on each axis: `(4, 3) + (3,)` → `(4, 3)`. This is exactly how a bias vector gets added to every row of a batch." },
        ],
      },
      { t: "h", text: "A layer as one matrix product" },
      {
        t: "p",
        text: "A dense layer has weights `W` of shape `(out, in)` and bias `b` of shape `(out,)`. For a batch `X` of shape `(N, in)` it computes `X @ W.T + b`. The product `(N, in) @ (in, out)` gives `(N, out)`, and broadcasting adds `b` to each of the N rows. Row $n$ of the result is exactly $W\\mathbf{x}_n + \\mathbf{b}$ for example $n$. In plain words: the batch is just $N$ separate matrix-vector products done in one call.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Notation" },
      {
        t: "list",
        items: [
          String.raw`A **scalar** $a \in \mathbb{R}$ is a real number. Read $\in$ as "is in" and $\mathbb{R}$ as "the real numbers". Written as lower-case italic letters.`,
          String.raw`A **vector** $\mathbf{x} \in \mathbb{R}^n$ is a list of $n$ real numbers $x_1, \dots, x_n$ ($\mathbb{R}^n$ = all such lists). By convention vectors are **column vectors** (shape $n \times 1$); the small $\top$ in $\mathbf{x}^\top$ means "transpose", turning it into a row. Written in bold lower case.`,
          String.raw`A **matrix** $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns; $A_{ij}$ is the entry in row $i$, column $j$. Written in upper case.`,
          String.raw`A **tensor** of order $r$ is an array indexed by $r$ integers, e.g. $T_{nchw}$ for an image batch: example $n$, channel $c$, row $h$, column $w$.`,
        ],
      },
      { t: "h", text: "Transpose" },
      { t: "math", tex: String.raw`(A^\top)_{ij} = A_{ji}, \qquad (A^\top)^\top = A, \qquad (AB)^\top = B^\top A^\top` },
      {
        t: "p",
        text: String.raw`The transpose flips a matrix across its main diagonal (top-left to bottom-right): rows become columns, and an $m \times n$ matrix becomes $n \times m$. Transposing twice gets you back. For a product, the order reverses: $((AB)^\top)_{ij} = (AB)_{ji} = \sum_p A_{jp} B_{pi} = \sum_p (B^\top)_{ip} (A^\top)_{pj} = (B^\top A^\top)_{ij}$. A matrix with $A^\top = A$ is **symmetric**.`,
      },
      { t: "h", text: "Dot product: algebraic definition" },
      {
        t: "math",
        tex: String.raw`\mathbf{u} \cdot \mathbf{v} \;=\; \mathbf{u}^\top \mathbf{v} \;=\; \sum_{i=1}^{n} u_i v_i \;=\; u_1 v_1 + u_2 v_2 + \dots + u_n v_n`,
        caption: "Multiply matching entries and add. Both vectors must have the same length $n$; the result is one number (a scalar).",
      },
      {
        t: "p",
        text: String.raw`The order does not matter ($\mathbf{u}\cdot\mathbf{v} = \mathbf{v}\cdot\mathbf{u}$), and it is **linear**: scaling or adding inputs scales or adds the result, $(\alpha\mathbf{u} + \mathbf{w})\cdot\mathbf{v} = \alpha\,\mathbf{u}\cdot\mathbf{v} + \mathbf{w}\cdot\mathbf{v}$ ($\alpha$ is any number). A vector dotted with itself gives its squared length: $\mathbf{u}\cdot\mathbf{u} = \|\mathbf{u}\|^2$. Here $\|\mathbf{u}\| = \sqrt{\sum_i u_i^2}$ is the **Euclidean norm** — the ordinary length of the arrow, by Pythagoras.`,
      },
      { t: "h", text: "Dot product: geometric meaning" },
      { t: "math", tex: String.raw`\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta`, caption: "$\\theta$ (theta) is the angle between the two arrows. In words: length × length × how aligned they are." },
      {
        t: "p",
        text: String.raw`**Why it is true.** The vectors $\mathbf{u}$, $\mathbf{v}$ and $\mathbf{u} - \mathbf{v}$ form a triangle. The law of cosines says $\|\mathbf{u}-\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$. Expanding the left side with the dot product gives $(\mathbf{u}-\mathbf{v})\cdot(\mathbf{u}-\mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\,\mathbf{u}\cdot\mathbf{v}$. Set the two equal and cancel: $\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$.`,
      },
      {
        t: "list",
        items: [
          String.raw`**The sign tells the angle**: positive means under 90°, zero means exactly 90° (the vectors are **orthogonal**), negative means over 90°.`,
          String.raw`**Cosine similarity**: divide out the lengths, $\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} \in [-1, 1]$. It is the standard similarity for embeddings.`,
          String.raw`**Projection**: if $\|\mathbf{v}\| = 1$, then $\mathbf{u}\cdot\mathbf{v}$ is the signed length of $\mathbf{u}$'s shadow on the direction of $\mathbf{v}$.`,
        ],
      },
      { t: "h", text: "Matrix multiplication" },
      {
        t: "math",
        tex: String.raw`A \in \mathbb{R}^{n \times k},\; B \in \mathbb{R}^{k \times m} \;\Longrightarrow\; C = AB \in \mathbb{R}^{n \times m}, \qquad C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}`,
        caption: "The inner size $k$ must match and disappears; the outer sizes $n$ and $m$ remain. $\\Longrightarrow$ means \"then\".",
      },
      {
        t: "list",
        items: [
          String.raw`**As many dot products:** $C_{ij} = (\text{row } i \text{ of } A) \cdot (\text{column } j \text{ of } B)$. Cost: $n \cdot m \cdot k$ multiplications and about as many additions.`,
          String.raw`**Column view:** column $j$ of $C$ is $A$ times column $j$ of $B$: $C = [\,A\mathbf{b}_1 \;\; A\mathbf{b}_2 \;\cdots\; A\mathbf{b}_m\,]$. The same transformation $A$ is applied to every column of $B$.`,
          String.raw`**Matrix times vector mixes columns:** $A\mathbf{x} = x_1\mathbf{a}_1 + x_2\mathbf{a}_2 + \dots + x_k\mathbf{a}_k$, where $\mathbf{a}_p$ is column $p$ of $A$. So the columns of $A$ show where the basis vectors $\mathbf{e}_1, \mathbf{e}_2, \dots$ (the unit arrows along each axis) land.`,
          String.raw`**Rules:** associative $(AB)C = A(BC)$; distributive $A(B + C) = AB + AC$; **not commutative** — usually $AB \neq BA$, and $BA$ may not even exist.`,
        ],
      },
      { t: "h", text: "Matrices as linear transformations" },
      {
        t: "p",
        text: String.raw`A function $f$ is **linear** if $f(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha f(\mathbf{x}) + \beta f(\mathbf{y})$: scaling and adding inputs scales and adds outputs. Every linear map from $\mathbb{R}^k$ to $\mathbb{R}^n$ is multiplication by some $n \times k$ matrix, and doing two maps in a row is multiplying their matrices: $A(B\mathbf{x}) = (AB)\mathbf{x}$. Example: rotation by 90° is $R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$. It sends $(1, 0)$ to $(0, 1)$ and $(2, 1)$ to $(-1, 2)$. The scaling $S = \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix}$ stretches the first axis and squashes the second: $(1, 1) \mapsto (2, 0.5)$ ($\mapsto$ means "goes to").`,
      },
      { t: "h", text: "A neural-network layer" },
      {
        t: "math",
        tex: String.raw`\mathbf{h} = \phi(W\mathbf{x} + \mathbf{b}), \qquad W \in \mathbb{R}^{d_\text{out} \times d_\text{in}},\; \mathbf{x} \in \mathbb{R}^{d_\text{in}},\; \mathbf{b},\mathbf{h} \in \mathbb{R}^{d_\text{out}}`,
      },
      {
        t: "p",
        text: String.raw`$\mathbf{x}$ is the input vector ($d_\text{in}$ numbers), $W$ the weight matrix (row $r$ holds output neuron $r$'s weights), $\mathbf{b}$ the bias vector, and $\phi$ (phi) an activation function applied to each entry. Each output $h_r = \phi(\mathbf{w}_r \cdot \mathbf{x} + b_r)$ is one neuron. For a batch stacked as rows of $X \in \mathbb{R}^{N \times d_\text{in}}$, the whole layer is $H = \phi(XW^\top + \mathbf{1}\mathbf{b}^\top)$. Here $\mathbf{1}$ is a column of ones, so $\mathbf{1}\mathbf{b}^\top$ is the bias copied onto every row — exactly what broadcasting does. The layer has $d_\text{out} \cdot d_\text{in} + d_\text{out}$ parameters. Without $\phi$, stacking layers would be pointless: $W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x}$ is again one linear layer.`,
      },
      { t: "h", text: "Identity and inverse" },
      {
        t: "math",
        tex: String.raw`I_n = \begin{pmatrix} 1 & & 0 \\ & \ddots & \\ 0 & & 1 \end{pmatrix}, \quad AI = IA = A, \qquad A A^{-1} = A^{-1} A = I`,
      },
      {
        t: "p",
        text: String.raw`The **identity matrix** $I_n$ has ones on the diagonal and zeros elsewhere; it leaves every vector unchanged (like multiplying by 1). A square matrix $A$ is **invertible** if some $A^{-1}$ undoes it. For a $2 \times 2$ matrix, $\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$, which exists only when the **determinant** $ad - bc \neq 0$. A zero determinant means the matrix flattens space onto a line or point, losing information nothing can bring back. Non-square matrices — almost all weight matrices — have no inverse at all.`,
      },
      { t: "h", text: "Broadcasting rule (formal)" },
      {
        t: "p",
        text: String.raw`Right-align the shapes $(s_1, \dots, s_r)$ and $(t_1, \dots, t_r)$, padding the shorter one with leading 1s. They fit if on every axis $s_a = t_a$ or $\min(s_a, t_a) = 1$. The result has shape $(\max(s_1,t_1), \dots, \max(s_r,t_r))$, and a size-1 axis acts as if its single entry were repeated.`,
      },

      { t: "h", text: "Worked example (the same numbers as the simulator)" },
      {
        t: "math",
        tex: String.raw`A = \begin{pmatrix} 2 & -1 & 0 \\ 1 & 3 & -2 \end{pmatrix}_{2 \times 3}, \qquad B = \begin{pmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 1 \end{pmatrix}_{3 \times 2}`,
      },
      {
        t: "steps",
        items: [
          { title: "Shape check", text: "$(2 \\times 3)(3 \\times 2)$: inner sizes 3 = 3, so $C$ is $2 \\times 2$ and needs $2 \\cdot 2 \\cdot 3 = 12$ multiplications." },
          { title: "C₁₁ = row 1 · column 1", tex: String.raw`2\cdot 1 + (-1)\cdot 0 + 0 \cdot 3 = 2 + 0 + 0 = 2` },
          { title: "C₁₂ = row 1 · column 2", tex: String.raw`2\cdot 2 + (-1)\cdot(-1) + 0\cdot 1 = 4 + 1 + 0 = 5` },
          { title: "C₂₁ = row 2 · column 1", tex: String.raw`1\cdot 1 + 3\cdot 0 + (-2)\cdot 3 = 1 + 0 - 6 = -5` },
          { title: "C₂₂ = row 2 · column 2", tex: String.raw`1\cdot 2 + 3\cdot(-1) + (-2)\cdot 1 = 2 - 3 - 2 = -3` },
          { title: "Result", tex: String.raw`C = AB = \begin{pmatrix} 2 & 5 \\ -5 & -3 \end{pmatrix}, \qquad C^\top = \begin{pmatrix} 2 & -5 \\ 5 & -3 \end{pmatrix} = B^\top A^\top` },
        ],
      },
      {
        t: "p",
        text: String.raw`**Dot product and angle.** For $\mathbf{u} = (3, 4)$ and $\mathbf{v} = (4, 3)$: $\mathbf{u}\cdot\mathbf{v} = 3\cdot 4 + 4 \cdot 3 = 24$ and $\|\mathbf{u}\| = \|\mathbf{v}\| = \sqrt{9 + 16} = 5$. So $\cos\theta = 24/25 = 0.96$ and $\theta = \arccos 0.96 \approx 16.26°$ ($\arccos$ turns a cosine back into an angle). They point in nearly the same direction.`,
      },
      {
        t: "p",
        text: String.raw`**A layer.** Take $W = \begin{pmatrix} 0.5 & -1 & 2 \\ 1.5 & 0 & -0.5 \end{pmatrix}$, $\mathbf{x} = (2, 1, 0.5)^\top$, $\mathbf{b} = (0.1, -0.2)^\top$. Neuron 1: $0.5\cdot 2 + (-1)\cdot 1 + 2 \cdot 0.5 = 1 - 1 + 1 = 1$, plus bias $0.1$ gives $1.1$. Neuron 2: $1.5 \cdot 2 + 0 \cdot 1 + (-0.5)\cdot 0.5 = 3 + 0 - 0.25 = 2.75$, plus bias $-0.2$ gives $2.55$. So $W\mathbf{x} + \mathbf{b} = (1.1, 2.55)^\top$.`,
      },
      {
        t: "p",
        text: String.raw`**An inverse.** $M = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$ has determinant $2\cdot 1 - 1 \cdot 1 = 1$, so $M^{-1} = \frac{1}{1}\begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}$. Check entries of $MM^{-1}$: row 1 · column 1 $= 2\cdot 1 + 1 \cdot(-1) = 1$, and row 1 · column 2 $= 2 \cdot(-1) + 1 \cdot 2 = 0$, as the identity requires.`,
      },
      {
        t: "p",
        text: String.raw`**Broadcasting.** $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ (shape $(2,3)$) plus $(10, 20, 30)$ (shape $(3,)$ → $(1,3)$) adds the row to each row: $\begin{pmatrix} 11 & 22 & 33 \\ 14 & 25 & 36 \end{pmatrix}$. Adding a column of shape $(2,1)$ with entries $100, 200$ instead adds 100 to row 1 and 200 to row 2: $\begin{pmatrix} 101 & 102 & 103 \\ 204 & 205 & 206 \end{pmatrix}$. The NumPy program below reproduces all these numbers.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator multiplies two small editable matrices one multiply-add at a time. The current row of A and column of B glow gold, the running sum is listed beside the grid, and each cell of C fills in when done.",
      },
      { t: "sim", id: "matmul", caption: "Matrix multiplication as a grid of row-by-column dot products." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "Press **Step** repeatedly with the default matrices. Check the first cell is $2\\cdot 1 + (-1)\\cdot 0 + 0\\cdot 3 = 2$ and that C ends as the worked example's $\\begin{pmatrix} 2 & 5 \\\\ -5 & -3 \\end{pmatrix}$.",
          "Switch **Step by** to 'output cell'. Each step now finishes a whole dot product. Count the steps: one per entry of C, each costing as many multiplications as A has columns.",
          "Set B's **rows** to 2 while A still has 3 columns. The product becomes undefined and the inner sizes turn clay-coloured: a row of length 3 cannot be dotted with a column of length 2.",
          "Make A a 3×3 identity (1s on the diagonal, 0s elsewhere) and B any 3×2 matrix. C equals B: the identity is the 'do nothing' transformation.",
          "Make A 2×2 with rows (0, −1) and (1, 0) — a 90° rotation — and B 2×3 with columns (1, 0), (0, 1), (2, 1). Each column of C is a rotated point; (2, 1) becomes (−1, 2): 'A acts on every column of B'.",
          "Press **Randomize**, then predict one entry of C before stepping to it. Then swap roles: is 2×3 times 3×2 the same shape as 3×2 times 2×3? (No — 2×2 versus 3×3, a direct demonstration that $AB \\neq BA$.)",
        ],
      },
      {
        t: "callout",
        kind: "tip",
        text: "In term mode, note that adding the products in any order gives the same exact answer, so hardware may reorder and parallelise. (In floating point, the last few bits can change — a small source of randomness on GPUs.)",
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "The program writes the dot product, transpose and matrix multiplication as loops, checks them against NumPy's `@`, then reproduces every worked-example number, plus per-channel normalisation of an image tensor.",
      },
      { t: "code", lang: "python", title: "linear_algebra_from_scratch.py", code: SCRATCH_CODE, output: SCRATCH_OUTPUT },
      {
        t: "p",
        text: "Two details. The batched layer uses `X @ W.T + b`: `(4, 3)` times `(3, 2)`, with the `(2,)` bias broadcast over 4 rows. And the image normalisation reshapes the per-channel means to `(1, 3, 1, 1)` so broadcasting lines them up with axis 1 (channels) of the `(8, 3, 32, 32)` tensor. Without the reshape, NumPy would align the 3 means with the last axis (width 32) and raise an error.",
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
                text: "PyTorch has the same operations. `torch.matmul` (or `@`) handles vectors, matrices and stacks of matrices. `nn.Linear` stores its weight as `(out_features, in_features)` and computes `x @ W.T + b`, checked here against the hand computation.",
              },
              { t: "code", lang: "python", title: "linear_algebra_torch.py", code: TORCH_CODE, output: TORCH_OUTPUT, runnable: false },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow's `tf.matmul` matches `torch.matmul`. Two layout differences from PyTorch cause frequent porting bugs: a Keras `Dense` kernel is stored as `(in_features, out_features)`, and Keras image layers default to channels-last `(N, H, W, C)`.",
              },
              {
                t: "code",
                lang: "python",
                title: "linear_algebra_tf.py (not executed in CI)",
                code: TF_CODE,
                output:
                  "Prints the 2×2 product [[2, 5], [-5, -3]]; then the Dense layer output [[1.1, 2.55]] for the single-sample batch; then the shapes (8, 32, 32, 3) and (8, 3, 32, 32).",
                runnable: false,
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Kinds of products" },
      {
        t: "table",
        head: ["Operation", "Shapes", "Result", "Where it appears"],
        rows: [
          ["Dot (inner) product $\\mathbf{u}^\\top\\mathbf{v}$", "$(n), (n)$", "scalar", "one neuron; attention score of one query with one key"],
          ["Outer product $\\mathbf{u}\\mathbf{v}^\\top$", "$(n), (m)$", "$n \\times m$ matrix", "a dense layer's weight gradient: $\\partial L/\\partial W = \\boldsymbol{\\delta}\\mathbf{x}^\\top$ ($\\boldsymbol{\\delta}$ = error signal at the output)"],
          ["Element-wise (Hadamard) product $A \\odot B$", "same shape (or broadcastable)", "same shape", "LSTM gates, masks, dropout, $\\sigma'(z)$ in backprop"],
          ["Matrix-vector product $A\\mathbf{x}$", "$(n,k), (k)$", "vector $(n)$", "one example through a dense layer"],
          ["Matrix-matrix product $AB$", "$(n,k), (k,m)$", "$(n,m)$", "a batch through a dense layer; $QK^\\top$ in attention"],
          ["Batched matmul", "$(b,n,k), (b,k,m)$", "$(b,n,m)$", "per-head attention, per-sequence operations"],
        ],
      },
      { t: "h", text: "Special matrices" },
      { t: "p", text: "Below, $O(n)$ (\"big-O of n\") means the cost grows in proportion to $n$." },
      {
        t: "table",
        head: ["Type", "Definition", "Advantage", "Limitation / use"],
        rows: [
          ["Diagonal", "non-zero only on the diagonal", "multiply in $O(n)$; easy to invert", "per-feature scaling (e.g. the scale in normalisation layers)"],
          ["Identity $I$", "diagonal of ones", "leaves vectors unchanged", "residual connections add the identity path: $\\mathbf{x} + F(\\mathbf{x})$"],
          ["Symmetric", "$A^\\top = A$", "real eigenvalues, orthogonal eigenvectors", "covariance matrices, Hessians"],
          ["Orthogonal", "$Q^\\top Q = I$", "keeps lengths and angles; $Q^{-1} = Q^\\top$", "orthogonal weight initialisation; rotations"],
          ["Sparse", "mostly zeros", "memory and compute grow with the non-zeros only", "graph adjacency matrices, pruned networks; poor fit for dense GPU kernels"],
          ["Low-rank $UV^\\top$", "product of thin matrices", "$(n + m)r$ parameters instead of $nm$", "LoRA fine-tuning, compression (see [Eigenvalues and SVD](/learn/m02/eigen-svd))"],
        ],
      },
      { t: "h", text: "Tensor layouts" },
      {
        t: "table",
        head: ["Layout", "Shape", "Default in", "Notes"],
        rows: [
          ["Batch × features", "$(N, D)$", "all frameworks", "tables, flattened inputs, dense-layer activations"],
          ["Batch × time × features", "$(N, T, D)$", "PyTorch with `batch_first=True`, Keras", "sequences and token embeddings"],
          ["Channels-first NCHW", "$(N, C, H, W)$", "PyTorch", "channels come right after the batch axis"],
          ["Channels-last NHWC", "$(N, H, W, C)$", "TensorFlow / Keras", "often faster on GPUs with Tensor Cores; PyTorch supports it via `channels_last` memory format"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Dense layers** — every MLP, and the feed-forward blocks in every Transformer, are matrix products $XW^\\top + \\mathbf{b}$ plus an activation. In large language models they do a large share of all the arithmetic.",
          "**Attention** — the score matrix $QK^\\top$ is a grid of query-key dot products, and the output is another matrix product with $V$ (see [Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention)).",
          "**Embeddings** — looking up token $t$'s embedding equals a one-hot vector times the embedding matrix; frameworks just read row $t$ to skip the zeros.",
          "**Convolutions** — many implementations turn a convolution into one big matrix product by unrolling image patches into columns (im2col), so they can reuse fast GEMM routines.",
          "**Similarity search** — comparing one embedding with millions of stored ones by dot products or cosine similarity is a matrix-vector product over the database.",
          "**Matrix factorisation** — approximating a users × items rating matrix as a product of two thin matrices was central to the Netflix Prize and is still a standard recommendation baseline.",
          "**Hardware design** — GPU Tensor Cores (introduced with NVIDIA's Volta) and Google's TPUs are built around fast matrix multiply-accumulate units, because this operation dominates deep learning.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Compact notation.** One expression describes millions of multiply-adds; shape rules catch errors early.",
          String.raw`**Hardware efficiency.** A matrix product does $O(nmk)$ arithmetic on only $O(nk + km)$ numbers, so each loaded number is reused many times. This lets GPUs run near full speed.`,
          "**Composability.** Linear maps combine by multiplication, transpose rules give backprop formulas mechanically, and batching is one more dimension.",
          "**Geometric meaning.** Dot products and transformations build intuition for layers and embeddings.",
        ],
      },
      { t: "h", text: "Limitations and trade-offs" },
      {
        t: "list",
        items: [
          "**Linear maps alone are weak.** Any stack of matrix multiplications collapses into one matrix; non-linear activations are what make depth useful.",
          String.raw`**Cubic cost.** Multiplying two $n \times n$ matrices takes $n^3$ multiplications with the standard method. Attention's $QK^\top$ grows with the square of sequence length, a major limit on context size.`,
          "**Memory-bound small ops.** Skinny products (batch size 1) and element-wise operations are limited by memory traffic, not arithmetic, so they waste the hardware.",
          "**Broadcasting is permissive.** It silently creates huge tensors or wrong shapes when axes line up by accident.",
          String.raw`**Inverses are fragile.** Inverting a nearly-singular matrix (one close to flattening space) blows up rounding errors. Solving $A\mathbf{x} = \mathbf{b}$ directly (LU or Cholesky factorisation) is more accurate and cheaper.`,
          "**Floating-point sums depend on order.** Different GPUs, batch sizes or kernels can change the last bits, which matters for exact reproducibility.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Using `*` when you mean `@`.** In NumPy and PyTorch, `A * B` is the element-wise product (and broadcasts); `A @ B` is the matrix product. With two `(3, 3)` matrices both run without error and give different answers. **Fix:** use `@` for matrix products and check the shapes of intermediate results.",
          "**The `(N,)` versus `(N, 1)` trap.** If predictions are `(N, 1)` and targets `(N,)`, then `pred - target` broadcasts to `(N, N)`, and a mean-squared error is silently computed over all pairs. **Fix:** `pred.squeeze(-1)` or `target[:, None]`, and assert shapes before reductions.",
          "**Assuming $AB = BA$.** Matrix multiplication is not commutative; swapping factors changes the result or makes it undefined. Likewise $(AB)^\\top = B^\\top A^\\top$, with the order reversed.",
          "**Wrong weight orientation when porting.** PyTorch's `nn.Linear.weight` is `(out, in)`; a Keras `Dense` kernel is `(in, out)`. Copying without transposing gives garbage (a shape error only if the layer is not square). **Fix:** transpose explicitly; test on a known input.",
          "**Transposing a 1-D array.** A NumPy 1-D array has no row/column direction: `x.T` returns the same `(n,)` array. **Fix:** `x[:, None]` or `x.reshape(-1, 1)` for a column, or let `@` handle 1-D arrays.",
          "**Computing an inverse to solve a system.** `np.linalg.inv(A) @ b` is slower and less accurate than `np.linalg.solve(A, b)`, and fails badly for ill-conditioned (nearly flattening) $A$. Neural networks almost never need inverses.",
          "**Mixing NCHW and NHWC.** Feeding a `(N, H, W, C)` batch to a PyTorch convolution expecting `(N, C, H, W)` either errors (if sizes differ) or, worse, runs on scrambled data when sizes happen to match. **Fix:** `x.permute(0, 3, 1, 2)` and assert the channel axis.",
          "**Thinking a 'vector of dimension 768' is a 768-axis tensor.** The number of entries (the vector-space dimension) and the number of axes (the rank) are different things. An embedding vector is a rank-1 tensor with 768 entries.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "What is the difference between a dot product, an element-wise product and a matrix product?",
        a: "A dot product takes two equal-length vectors and returns a scalar: the sum of pairwise products. An element-wise (Hadamard) product multiplies matching entries of two same-shaped arrays and returns that shape — no summing. A matrix product $AB$ of an $(n, k)$ and a $(k, m)$ matrix returns an $(n, m)$ matrix whose entries are dot products of rows of $A$ with columns of $B$. In NumPy/PyTorch these are `np.dot`/`@`, `*`, and `@` respectively.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "A batch has shape (32, 784) and a weight matrix has shape (784, 128). What is the shape of their product, and why?",
        a: "(32, 128). The inner sizes (784 and 784) match and are summed over; the outer sizes remain. Each of the 32 rows (examples) is dotted with each of the 128 columns (output neurons), giving 32 × 128 numbers, each a sum of 784 products.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What is a tensor in deep learning, and what does an image batch of shape (16, 3, 224, 224) mean?",
        a: "In deep learning a tensor is a multi-dimensional array with a shape and a data type; scalars, vectors and matrices are tensors of rank 0, 1 and 2. In PyTorch's NCHW convention, (16, 3, 224, 224) means 16 images, 3 colour channels (RGB), 224 rows and 224 columns of pixels — 2,408,448 numbers in total.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Explain the broadcasting rules and give an example of a silent bug they can cause.",
        a: "Right-align the shapes and pad with leading 1s; on each axis sizes must be equal or one must be 1, and the result takes the larger size (size-1 axes are virtually repeated). Classic bug: predictions of shape (N, 1) minus targets of shape (N,) — the target becomes (1, N), so the difference is (N, N). The MSE is then averaged over all N² pairs, looks plausible, and the model trains poorly with no error. Asserting shapes or using `squeeze` prevents it.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "What does the dot product measure geometrically, and why is it used for attention scores?",
        a: "$\\mathbf{u}\\cdot\\mathbf{v} = \\|\\mathbf{u}\\|\\|\\mathbf{v}\\|\\cos\\theta$, so it is large when the vectors are long and aligned, zero when orthogonal and negative when opposed. Attention needs a cheap, differentiable measure of how well a query matches each key; all query-key scores come from one fast matrix product $QK^\\top$. Scores are divided by $\\sqrt{d_k}$ because the variance of a dot product of random vectors grows with the dimension.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why are GPUs so much faster than CPUs at matrix multiplication?",
        a: "Every output entry is independent, so thousands can be computed in parallel, and each input value is reused across a whole row or column of outputs. GPUs have thousands of arithmetic units and fast on-chip memory for reused tiles, plus dedicated matrix units (Tensor Cores) that do small matrix multiply-accumulates in one instruction, often in reduced precision. The high arithmetic per byte moved keeps them busy.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Why is a deep network with no activation functions no more expressive than a single linear layer? What about the biases?",
        a: "Each layer computes $W_\\ell\\mathbf{h} + \\mathbf{b}_\\ell$. Two layers give $W_2(W_1\\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2 = (W_2W_1)\\mathbf{x} + (W_2\\mathbf{b}_1 + \\mathbf{b}_2)$ — again an affine map (linear plus a shift) with weight $W_2W_1$ and bias $W_2\\mathbf{b}_1 + \\mathbf{b}_2$. By repeating this, any depth collapses to one affine layer. Depth can only add a rank limit: if a hidden layer is narrower than the input and output, $W_2W_1$ has rank at most that width, so the stack is a restricted linear map (as in linear autoencoders, which recover PCA). Non-linear activations break the collapse.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Given A and B of size n×n and a vector x, which is cheaper: (AB)x or A(Bx)? Why does this matter in backpropagation?",
        a: "$(AB)\\mathbf{x}$ costs $n^3$ multiplications for $AB$ plus $n^2$ for the matrix-vector product; $A(B\\mathbf{x})$ costs two matrix-vector products, $2n^2$. Associativity guarantees the same answer, but the cost differs by about a factor of $n/2$. Reverse-mode automatic differentiation uses exactly this: it multiplies the gradient vector by one Jacobian at a time (vector-Jacobian products) instead of multiplying full Jacobian matrices together. That is why backpropagation costs about as much as a forward pass.",
      },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "What is the shape of the product of a (64, 128) matrix and a (128, 32) matrix?",
            options: ["(128, 128)", "(64, 32)", "(32, 64)", "Undefined"],
            answer: 1,
            explain: "The inner sizes (128) match and vanish; the outer sizes (64, 32) remain.",
          },
          {
            kind: "numeric",
            q: "How many scalar multiplications does the standard algorithm use to multiply a (64, 128) matrix by a (128, 32) matrix?",
            answer: 262144,
            tolerance: 0.5,
            explain: "$n \\cdot m \\cdot k = 64 \\times 32 \\times 128 = 262{,}144$: one multiplication per term of each of the 2,048 dot products of length 128.",
          },
          {
            kind: "numeric",
            q: "Compute the dot product of (1, 2, 3) and (4, −5, 6).",
            answer: 12,
            tolerance: 0.001,
            explain: "$1\\cdot 4 + 2 \\cdot (-5) + 3 \\cdot 6 = 4 - 10 + 18 = 12$.",
          },
          {
            kind: "numeric",
            q: "How many trainable parameters does a dense layer mapping 784 inputs to 128 outputs (with bias) have?",
            answer: 100480,
            tolerance: 0.5,
            explain: "Weights: $128 \\times 784 = 100{,}352$; biases: 128; total 100,480.",
          },
          {
            q: "Arrays of shapes (8, 1, 5) and (3, 1) are added with broadcasting. What is the result shape?",
            options: ["(8, 3, 5)", "Error: incompatible shapes", "(8, 1, 5)", "(8, 3, 1)"],
            answer: 0,
            explain: "Right-align: (8, 1, 5) and (1, 3, 1). Per axis: max(8,1)=8, max(1,3)=3, max(5,1)=5. Every pair is equal or contains a 1, so the result is (8, 3, 5).",
          },
          {
            q: "Two non-zero embedding vectors have a dot product of exactly 0. What does this mean geometrically?",
            options: ["They are identical", "They point in opposite directions", "They are orthogonal (at 90°)", "One of them must be the zero vector"],
            answer: 2,
            explain: "$\\mathbf{u}\\cdot\\mathbf{v} = \\|\\mathbf{u}\\|\\|\\mathbf{v}\\|\\cos\\theta$. With both lengths non-zero, it is 0 only when $\\cos\\theta = 0$, i.e. $\\theta = 90°$.",
          },
          {
            q: "Which identity is always true for matrices whose products are defined?",
            options: ["$AB = BA$", "$(AB)^\\top = A^\\top B^\\top$", "$(AB)^\\top = B^\\top A^\\top$", "$(A + B)^{-1} = A^{-1} + B^{-1}$"],
            answer: 2,
            explain: "Transposing a product reverses the order of the factors. The other three fail in general.",
          },
          {
            q: "In PyTorch, what is the conventional shape of a batch of 16 RGB images of size 224×224?",
            options: ["(16, 224, 224, 3)", "(3, 16, 224, 224)", "(16, 3, 224, 224)", "(224, 224, 3, 16)"],
            answer: 2,
            explain: "PyTorch uses NCHW: batch, channels, height, width. (16, 224, 224, 3) is the NHWC layout Keras uses by default.",
          },
        ],
      },
      {
        t: "exercise",
        title: "A layer by hand",
        prompt: String.raw`Let $W = \begin{pmatrix} 1 & 0 & 2 \\ -1 & 3 & 1 \end{pmatrix}$, $\mathbf{x} = (3, 1, 2)^\top$ and $\mathbf{b} = (1, 0)^\top$. (a) Compute $W\mathbf{x} + \mathbf{b}$. (b) If a batch $X$ of 50 such inputs is stored as a $(50, 3)$ matrix, write the batched expression and give its output shape.`,
        hint: "Each output is a row of W dotted with x. In the batch the examples are rows, so the weights must be transposed.",
        solution: [
          {
            t: "p",
            text: String.raw`(a) Row 1: $1\cdot 3 + 0 \cdot 1 + 2 \cdot 2 = 3 + 0 + 4 = 7$, plus $b_1 = 1$ gives $8$. Row 2: $(-1)\cdot 3 + 3 \cdot 1 + 1 \cdot 2 = -3 + 3 + 2 = 2$, plus $b_2 = 0$ gives $2$. So $W\mathbf{x} + \mathbf{b} = (8, 2)^\top$.`,
          },
          {
            t: "p",
            text: String.raw`(b) $H = XW^\top + \mathbf{b}$, with shapes $(50, 3)(3, 2) \to (50, 2)$ and the $(2,)$ bias broadcast across the 50 rows. Row $n$ of $H$ equals $W\mathbf{x}_n + \mathbf{b}$.`,
          },
          { t: "code", lang: "python", title: "check.py", code: "import numpy as np\nW = np.array([[1, 0, 2], [-1, 3, 1]]); x = np.array([3, 1, 2]); b = np.array([1, 0])\nprint(W @ x + b)", output: "[8 2]" },
        ],
      },
      {
        t: "exercise",
        title: "Pairwise distances with broadcasting and one matmul",
        prompt: String.raw`Given $X$ of shape $(n, d)$ and $Y$ of shape $(m, d)$, write two NumPy functions that return the $(n, m)$ matrix $D_{ij} = \|\mathbf{x}_i - \mathbf{y}_j\|^2$ (squared distance between row $i$ of $X$ and row $j$ of $Y$) with no Python loops. One uses broadcasting on a 3-D difference tensor. The other uses $\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\,\mathbf{x}\cdot\mathbf{y}$ and a single matrix product. Check both against a double loop.`,
        hint: "Insert size-1 axes with X[:, None, :] and Y[None, :, :]. For the second version, all cross dot products at once are X @ Y.T.",
        solution: [
          { t: "code", lang: "python", title: "pairwise_distances.py", code: EX_CODE, output: EX_OUTPUT },
          {
            t: "p",
            text: String.raw`The broadcasting version builds an $(n, m, d)$ tensor, so memory grows with $n \cdot m \cdot d$. The matmul version only creates $(n, m)$ arrays and does the heavy work in one fast GEMM, which is why libraries use it. The price is a little floating-point accuracy for nearly identical points, where subtracting large similar numbers cancels digits.`,
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          "**Scalars, vectors, matrices, tensors** are arrays with 0, 1, 2 and $r$ axes. Read and check shapes constantly.",
          String.raw`**Transpose:** $(A^\top)_{ij} = A_{ji}$ and $(AB)^\top = B^\top A^\top$. In NumPy/PyTorch it is a free view.`,
          String.raw`**Dot product:** $\mathbf{u}\cdot\mathbf{v} = \sum_i u_iv_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ — measures alignment; zero means orthogonal; divided by the lengths it is cosine similarity.`,
          String.raw`**Matrix product:** $(n \times k)(k \times m) \to (n \times m)$, $C_{ij} = \sum_p A_{ip}B_{pj}$, costing $nmk$ multiply-adds. A grid of dot products, or equally the map $A$ applied to every column of $B$. Not commutative.`,
          String.raw`**Layer:** $\mathbf{h} = \phi(W\mathbf{x} + \mathbf{b})$ with $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$; for a batch, $H = \phi(XW^\top + \mathbf{b})$. Without $\phi$, stacked layers collapse into one.`,
          String.raw`**Identity and inverse:** $AI = A$; $AA^{-1} = I$ exists only for square matrices with non-zero determinant; prefer solving systems to forming inverses.`,
          "**Broadcasting:** right-align shapes; each axis must match or be 1; size-1 axes are stretched. Powerful and dangerous — `(N,1) - (N,)` gives `(N,N)`.",
          "**Tensors in DL:** tables `(N, D)`, sequences `(N, T, D)`, images `(N, C, H, W)` in PyTorch or `(N, H, W, C)` in Keras.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Vector Spaces, Norms and Distances](/learn/m02/norms-spaces) — length and distance (L1, L2, cosine) made precise; needed for regularisation and embeddings.",
          "[Derivatives, Partial Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — the other half of training maths: how the loss changes when the weights inside these products change.",
          "[Eigenvalues, Eigenvectors and SVD](/learn/m02/eigen-svd) — a matrix broken into rotations and stretches; the basis of PCA and low-rank adaptation.",
          "[Forward Propagation](/learn/m04/forward-propagation) — applies $W\\mathbf{x} + \\mathbf{b}$ layer after layer to compute a whole network's output.",
        ],
      },
    ],
  },
};

export default lesson;
