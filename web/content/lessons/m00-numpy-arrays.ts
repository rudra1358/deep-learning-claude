import type { Lesson } from "@/lib/types";

const SPEED = String.raw`import time
import numpy as np

n = 1_000_000                      # one million numbers
py_list = list(range(n))           # a plain Python list of ints
arr = np.arange(n)                 # a NumPy array of the same ints (int64)

def best_of(fn, repeats=5):
    """Run fn several times and keep the fastest time (least disturbed by other programs)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)

# Task: the sum of squares 0^2 + 1^2 + ... + (n-1)^2
t_loop = best_of(lambda: sum(x * x for x in py_list))   # Python loop over a list
t_np = best_of(lambda: np.sum(arr * arr))               # NumPy: one call, loop runs in C

print(f"Python list loop : {t_loop * 1000:7.2f} ms")
print(f"NumPy array      : {t_np * 1000:7.2f} ms")
print(f"speed-up         : {t_loop / t_np:5.0f}x")
print("same answer:", sum(x * x for x in py_list) == int(np.sum(arr * arr)))

# What "+" means for each type
print([1, 2, 3] + [10, 20, 30])            # list: concatenation
print(np.array([1, 2, 3]) + np.array([10, 20, 30]))   # array: element-wise addition

# Memory: a list stores pointers to full Python objects; an array stores raw numbers
import sys
small = list(range(1000))
bytes_list = sys.getsizeof(small) + sum(sys.getsizeof(x) for x in small)
print("list of 1000 ints :", bytes_list, "bytes")
print("array of 1000 int64:", np.arange(1000).nbytes, "bytes")`;

const SPEED_OUT = String.raw`Python list loop :   44.05 ms
NumPy array      :    3.21 ms
speed-up         :    14x
same answer: True
[1, 2, 3, 10, 20, 30]
[11 22 33]
list of 1000 ints : 36056 bytes
array of 1000 int64: 8000 bytes`;

const TOUR = String.raw`import numpy as np

# ---------- 1. Creating arrays ----------
a = np.array([1, 2, 3, 4, 5, 6])                 # from a Python list
print("a:", a, "| dtype:", a.dtype, "| shape:", a.shape)
print("zeros:", np.zeros(3), "| ones:", np.ones((2, 2), dtype=np.int32).tolist())
print("arange(0, 10, 2):", np.arange(0, 10, 2))   # start, stop (excluded), step
print("linspace(0, 1, 5):", np.linspace(0, 1, 5)) # 5 evenly spaced points, both ends included
rng = np.random.default_rng(0)                    # a seeded random generator
print("integers:", rng.integers(0, 10, size=5))
print("normal  :", rng.normal(size=3).round(3))

# ---------- 2. dtype: the type of every element ----------
x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
print("float32 itemsize:", x.itemsize, "bytes | float64 itemsize:", x.astype(np.float64).itemsize, "bytes")
counts = np.array([120, 7], dtype=np.int8)       # int8 holds -128..127 only
print("int8 overflow: 120 + 10 =", (counts + np.int8(10))[0])
big = np.float32(16_777_216)                     # 2**24
print("float32: 2**24 + 1 =", big + np.float32(1), "| float64:", np.float64(2**24) + 1)
mask = np.array([3, -1, 0, 7]) > 0
print("bool array:", mask, "| dtype:", mask.dtype)

# ---------- 3. Shape, ndim, size, reshape ----------
m = np.arange(12)
print("m.shape:", m.shape, "ndim:", m.ndim, "size:", m.size)
M = m.reshape(3, 4)                              # 3 rows, 4 columns
print("M = m.reshape(3, 4):\n", M)
print("reshape(2, -1) ->", m.reshape(2, -1).shape, "| reshape(-1, 3) ->", m.reshape(-1, 3).shape)
try:
    m.reshape(5, -1)                             # 12 is not divisible by 5
except ValueError as e:
    print("reshape error:", e)

# ---------- 4. Indexing and slicing ----------
print("M[1, 2] =", M[1, 2], "| M[-1, -1] =", M[-1, -1])
print("row 1:", M[1], "| column 2:", M[:, 2])
print("M[0:2, 1:3] =\n", M[0:2, 1:3])
print("every other column:\n", M[:, ::2])
T = np.arange(24).reshape(2, 3, 4)               # 3-D: 2 blocks of 3x4
print("T[1, 2, 3] =", T[1, 2, 3], "| T[1].shape =", T[1].shape, "| T[:, 0, :].shape =", T[:, 0, :].shape)

# ---------- 5. Fancy and boolean indexing ----------
print("fancy M[[0, 2]] rows:\n", M[[0, 2]])
print("pairs M[[0, 1, 2], [3, 2, 1]] =", M[[0, 1, 2], [3, 2, 1]])
print("M[M % 2 == 0] =", M[M % 2 == 0])
clipped = M.copy()
clipped[clipped > 8] = 8                         # boolean mask assignment
print("clip above 8:\n", clipped)

# ---------- 6. Views vs copies ----------
row = M[0]                                       # basic slice -> a VIEW (shares memory)
picked = M[[0]]                                  # fancy index -> a COPY
print("slice shares memory:", np.shares_memory(M, row), "| fancy shares memory:", np.shares_memory(M, picked))
row[0] = 100                                     # writing through the view changes M
print("after row[0] = 100, M[0] =", M[0])
safe = M[0].copy()
safe[1] = -1
print("after editing a copy, M[0] =", M[0])
M[0, 0] = 0                                      # restore

# ---------- 7. Axis semantics ----------
S = np.array([[1, 2, 3],
              [4, 5, 6]])
print("S.sum() =", S.sum(), "| sum(axis=0) =", S.sum(axis=0), "| sum(axis=1) =", S.sum(axis=1))
print("mean(axis=0) =", S.mean(axis=0), "| mean(axis=1) =", S.mean(axis=1))

# ---------- 8. Joining arrays ----------
p, q = np.array([1, 2, 3]), np.array([4, 5, 6])
print("concatenate:", np.concatenate([p, q]), "| stack:", np.stack([p, q]).shape,
      "| stack(axis=1):", np.stack([p, q], axis=1).shape)
print("vstack:", np.vstack([S, S]).shape, "| hstack:", np.hstack([S, S]).shape)

# ---------- 9. A deep-learning tensor: a batch of images (N, C, H, W) ----------
images = rng.integers(0, 256, size=(8, 3, 32, 32), dtype=np.uint8)
print("batch:", images.shape, images.dtype, "| bytes:", images.nbytes)
first = images[0]                                # one image: (3, 32, 32)
red = images[:, 0]                               # red channel of every image: (8, 32, 32)
x = images.astype(np.float32) / 255.0            # the usual conversion before training
print("one image:", first.shape, "| red channels:", red.shape, "| as float32:", x.dtype, x.nbytes, "bytes")
print("value range:", x.min(), "to", x.max())
print("flattened for a dense layer:", x.reshape(8, -1).shape)`;

const TOUR_OUT = String.raw`a: [1 2 3 4 5 6] | dtype: int64 | shape: (6,)
zeros: [0. 0. 0.] | ones: [[1, 1], [1, 1]]
arange(0, 10, 2): [0 2 4 6 8]
linspace(0, 1, 5): [0.   0.25 0.5  0.75 1.  ]
integers: [8 6 5 2 3]
normal  : [ 0.105 -0.536  0.362]
float32 itemsize: 4 bytes | float64 itemsize: 8 bytes
int8 overflow: 120 + 10 = -126
float32: 2**24 + 1 = 1.6777216e+07 | float64: 16777217.0
bool array: [ True False False  True] | dtype: bool
m.shape: (12,) ndim: 1 size: 12
M = m.reshape(3, 4):
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
reshape(2, -1) -> (2, 6) | reshape(-1, 3) -> (4, 3)
reshape error: cannot reshape array of size 12 into shape (5,newaxis)
M[1, 2] = 6 | M[-1, -1] = 11
row 1: [4 5 6 7] | column 2: [ 2  6 10]
M[0:2, 1:3] =
 [[1 2]
 [5 6]]
every other column:
 [[ 0  2]
 [ 4  6]
 [ 8 10]]
T[1, 2, 3] = 23 | T[1].shape = (3, 4) | T[:, 0, :].shape = (2, 4)
fancy M[[0, 2]] rows:
 [[ 0  1  2  3]
 [ 8  9 10 11]]
pairs M[[0, 1, 2], [3, 2, 1]] = [3 6 9]
M[M % 2 == 0] = [ 0  2  4  6  8 10]
clip above 8:
 [[0 1 2 3]
 [4 5 6 7]
 [8 8 8 8]]
slice shares memory: True | fancy shares memory: False
after row[0] = 100, M[0] = [100   1   2   3]
after editing a copy, M[0] = [100   1   2   3]
S.sum() = 21 | sum(axis=0) = [5 7 9] | sum(axis=1) = [ 6 15]
mean(axis=0) = [2.5 3.5 4.5] | mean(axis=1) = [2. 5.]
concatenate: [1 2 3 4 5 6] | stack: (2, 3) | stack(axis=1): (3, 2)
vstack: (4, 3) | hstack: (2, 6)
batch: (8, 3, 32, 32) uint8 | bytes: 24576
one image: (3, 32, 32) | red channels: (8, 32, 32) | as float32: float32 98304 bytes
value range: 0.0 to 1.0
flattened for a dense layer: (8, 3072)`;

const LAYOUT = String.raw`import numpy as np

M = np.arange(12, dtype=np.int64).reshape(3, 4)
print("shape:", M.shape, "strides:", M.strides, "itemsize:", M.itemsize)
i, j = 2, 1
offset = i * M.strides[0] + j * M.strides[1]          # bytes from the start of the buffer
print("offset of M[2, 1]:", offset, "bytes = element", offset // M.itemsize, "-> value", M[i, j])
print("M.T strides:", M.T.strides, "| M.T shares memory:", np.shares_memory(M, M.T))
print("M[:, ::2] strides:", M[:, ::2].strides)
print("contiguous? M:", M.flags["C_CONTIGUOUS"], "| M.T:", M.T.flags["C_CONTIGUOUS"])

batch = np.zeros((8, 3, 32, 32), dtype=np.float32)
print("image batch strides:", batch.strides, "| nbytes:", batch.nbytes)
n, c, h, w = 5, 2, 10, 7
flat = ((n * 3 + c) * 32 + h) * 32 + w             # row-major flat index
print("flat index of [5, 2, 10, 7]:", flat, "| check:", np.ravel_multi_index((n, c, h, w), batch.shape))

# float32 precision
print("float32 eps:", np.finfo(np.float32).eps, "| float64 eps:", np.finfo(np.float64).eps)
print("float32 max:", np.finfo(np.float32).max, "| float16 max:", np.finfo(np.float16).max)
print("int8 range:", np.iinfo(np.int8).min, np.iinfo(np.int8).max, "| uint8:", np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
print("0.1 as float32:", f"{np.float32(0.1):.20f}")
print("0.1 as float64:", f"{np.float64(0.1):.20f}")
print("uint8 200 + 100 =", np.array([200], dtype=np.uint8) + np.uint8(100))
s = np.float32(0)
for _ in range(10):
    s += np.float32(0.1)
print("sum of ten 0.1 in float32:", s, "| == 1.0:", s == np.float32(1.0))`;

const LAYOUT_OUT = String.raw`shape: (3, 4) strides: (32, 8) itemsize: 8
offset of M[2, 1]: 72 bytes = element 9 -> value 9
M.T strides: (8, 32) | M.T shares memory: True
M[:, ::2] strides: (32, 16)
contiguous? M: True | M.T: False
image batch strides: (12288, 4096, 128, 4) | nbytes: 98304
flat index of [5, 2, 10, 7]: 17735 | check: 17735
float32 eps: 1.1920929e-07 | float64 eps: 2.220446049250313e-16
float32 max: 3.4028235e+38 | float16 max: 6.55e+04
int8 range: -128 127 | uint8: 0 255
0.1 as float32: 0.10000000149011611938
0.1 as float64: 0.10000000000000000555
uint8 200 + 100 = [44]
sum of ten 0.1 in float32: 1.0000001 | == 1.0: False`;

const TORCH = String.raw`import numpy as np
import torch

a = np.arange(6, dtype=np.float32).reshape(2, 3)

# torch.from_numpy: NO copy. The tensor and the array are two views of one buffer.
t = torch.from_numpy(a)
print("tensor:", t.dtype, tuple(t.shape))
a[0, 0] = 99.0                     # change the NumPy array ...
print("tensor sees the change:", t[0, 0].item())
t[1, 2] = -5.0                     # ... or change the tensor
print("array sees the change :", a[1, 2])
print("same memory address:", t.data_ptr() == a.ctypes.data)

# torch.tensor: ALWAYS copies
c = torch.tensor(a)
a[0, 1] = 42.0
print("torch.tensor copy unaffected:", c[0, 1].item())

# .numpy() goes the other way, also without copying (CPU tensors only)
back = t.numpy()
print("t.numpy() shares memory with a:", np.shares_memory(back, a))

# dtype pitfall: NumPy defaults to float64, PyTorch layers use float32
x64 = np.random.default_rng(0).normal(size=(4, 3))
print("numpy default dtype:", x64.dtype, "-> from_numpy gives", torch.from_numpy(x64).dtype)
layer = torch.nn.Linear(3, 2)
try:
    layer(torch.from_numpy(x64))
except RuntimeError as e:
    print("RuntimeError:", str(e).split("\n")[0])
out = layer(torch.from_numpy(x64).float())  # .float() converts to float32 (a copy)
print("after .float():", out.shape, out.dtype)

# Indexing, reshape and axis reductions use the same rules as NumPy
imgs = torch.zeros(8, 3, 32, 32)
print("imgs[0].shape:", tuple(imgs[0].shape), "| imgs[:, 0].shape:", tuple(imgs[:, 0].shape),
      "| reshape(8, -1):", tuple(imgs.reshape(8, -1).shape), "| mean(dim=(0, 2, 3)):", tuple(imgs.mean(dim=(0, 2, 3)).shape))`;

const TORCH_OUT = String.raw`tensor: torch.float32 (2, 3)
tensor sees the change: 99.0
array sees the change : -5.0
same memory address: True
torch.tensor copy unaffected: 1.0
t.numpy() shares memory with a: True
numpy default dtype: float64 -> from_numpy gives torch.float64
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
after .float(): torch.Size([4, 2]) torch.float32
imgs[0].shape: (3, 32, 32) | imgs[:, 0].shape: (8, 32, 32) | reshape(8, -1): (8, 3072) | mean(dim=(0, 2, 3)): (3,)`;

const TF_CODE = String.raw`import numpy as np
import tensorflow as tf

a = np.arange(6, dtype=np.float32).reshape(2, 3)

t = tf.convert_to_tensor(a)        # NumPy -> TensorFlow (tensors are immutable)
print(t.dtype, t.shape)
print(tf.reduce_sum(t, axis=0).numpy())   # TF calls it axis, like NumPy

back = t.numpy()                   # tensor -> NumPy array
print(type(back), back.shape)

# Keras image layers expect channels-last batches: (N, H, W, C)
images = np.zeros((8, 32, 32, 3), dtype=np.float32)
print(tf.keras.layers.Conv2D(16, 3)(images).shape)`;

const EX_CODE = String.raw`import numpy as np

rng = np.random.default_rng(42)
images = rng.random(size=(6, 3, 32, 32)).astype(np.float32)   # 6 RGB images, values in [0, 1)
images[2] *= 0.2                                            # make image 2 very dark

# 1) centre crop 16x16: rows 8..23 and columns 8..23 (a slice, so a VIEW)
crop = images[:, :, 8:24, 8:24]
print("crop:", crop.shape, "| view of images:", np.shares_memory(crop, images))

# 2) horizontal flip: reverse the last axis (also a view, with a negative stride)
flipped = images[..., ::-1]
print("flip strides:", flipped.strides[-1], "| first pixel of flip == last pixel of original:",
      flipped[0, 0, 0, 0] == images[0, 0, 0, 31])

# 3) brightness per image = mean over channels, rows and columns (axes 1, 2, 3)
brightness = images.mean(axis=(1, 2, 3))
print("brightness:", brightness.round(3))

# 4) keep only images brighter than 0.3 (boolean indexing -> a COPY)
keep = brightness > 0.3
bright = images[keep]
print("kept:", keep.sum(), "of", len(images), "| shape:", bright.shape,
      "| copy:", not np.shares_memory(bright, images))`;

const EX_OUT = String.raw`crop: (6, 3, 16, 16) | view of images: True
flip strides: -4 | first pixel of flip == last pixel of original: True
brightness: [0.5   0.487 0.101 0.499 0.502 0.504]
kept: 5 of 6 | shape: (5, 3, 32, 32) | copy: True`;

const lesson: Lesson = {
  id: "m00/numpy-arrays",
  title: "NumPy Arrays, Shapes and Indexing",
  summary:
    "The NumPy array (ndarray) is the data structure behind every tensor in deep learning. This lesson builds it from the ground up: why Python lists are not enough, how to create arrays, what dtype and shape mean, how to index and slice in 1, 2 and 3 dimensions, when you get a view and when a copy, what 'axis' means, and how all of this maps onto image batches and PyTorch tensors.",
  minutes: 60,
  level: "beginner",
  prerequisites: [],
  next: ["m00/vectorization", "m02/vectors-matrices", "m08/images-as-tensors"],
  terms: ["ndarray", "dtype", "shape", "axis", "view", "stride", "fancy-indexing", "boolean-mask", "tensor", "nchw"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "**NumPy** (short for *Numerical Python*) is the Python library for working with large blocks of numbers. Its central object is the **ndarray** (*n-dimensional array*): a grid of numbers that all have the same type, stored side by side in memory. A list of 5 temperatures, a 28×28 grayscale image, a batch of 64 colour photos — each is one ndarray.",
      },
      {
        t: "p",
        text: "Why not just use Python lists? Two reasons. First, **lists are slow for maths**: every element of a list is a separate Python object somewhere in memory, and a Python loop has to look at each one, check its type and call the right addition routine, one element at a time. NumPy stores raw numbers in one block and runs the loop in compiled C code. On our test machine, summing the squares of a million integers took about 44–83 ms with a Python loop and about 3–5 ms with NumPy — roughly 14–20 times faster — and the array used 8,000 bytes per thousand integers where the list used 36,056. Second, **lists are not numeric**: `[1, 2, 3] + [10, 20, 30]` *joins* the lists into `[1, 2, 3, 10, 20, 30]`, while adding two arrays adds them element by element to give `[11 22 33]`.",
      },
      {
        t: "p",
        text: "Deep learning is almost entirely arithmetic on large arrays, so this matters everywhere. Images, audio, text embeddings, network weights and gradients are all arrays. PyTorch tensors and TensorFlow tensors were designed to look and behave like NumPy arrays — the same shapes, the same indexing, the same `axis` arguments — and they convert to and from NumPy with one call. Learning NumPy well is therefore learning the language that every later lesson in this course is written in.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The one idea to keep",
        text: "An array is **one flat block of same-typed numbers** plus a description of how to read it: the **dtype** (what type each number is), the **shape** (how many entries along each axis) and the **strides** (how many bytes to jump to move along each axis). Reshaping, slicing and transposing mostly change the description, not the numbers.",
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      { t: "p", text: "This is the first technical lesson of the course. You need only basic Python:" },
      {
        t: "list",
        items: [
          "**Lists and list indexing** — `xs[0]` is the first element, `xs[-1]` the last, and `xs[1:4]` the elements at positions 1, 2 and 3 (the stop position is excluded). NumPy reuses exactly these rules and extends them to several dimensions.",
          "**Loops and functions** — `for` loops and `def`. We will compare loops with NumPy operations and see why the loops are slow.",
          "**Nested lists** — a list of lists such as `[[1, 2], [3, 4]]` is how you write a table in plain Python. `np.array` turns it into a 2-D array.",
          "**Installing a package** — `pip install numpy` (and `torch` for the PyTorch tab). Every program here was run with NumPy 2.x and PyTorch 2.x on a CPU.",
          "No mathematics is needed beyond arithmetic. The linear algebra behind arrays (vectors, matrices, matrix multiplication) comes later in [Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices).",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "An egg tray versus a bag of parcels",
        text: "A Python list is like a bag of parcels: each parcel (number) is wrapped separately, can be a different size, and the bag only holds *slips of paper* saying where each parcel is stored. To add up the contents you must find, unwrap and inspect every parcel. A NumPy array is like a long egg tray: every slot is the same size and holds the same kind of thing, and the slots are in one continuous row. A machine can roll along the tray and process every slot in one smooth pass. **Shape** is how you fold that long tray into rows and layers — 12 slots can be read as 3 rows of 4, or 2 layers of 2 rows of 3 — without moving a single egg.",
      },
      {
        t: "p",
        text: "Keep that picture in mind and most of NumPy follows. The **dtype** is the slot size: an `int8` slot holds one byte (a whole number from −128 to 127), a `float32` slot holds four bytes (a decimal number with about 7 significant digits). The **shape** is the folding: `(3, 4)` means 3 rows of 4. An **axis** is one direction of the folding: axis 0 runs down the rows, axis 1 runs across the columns.",
      },
      {
        t: "p",
        text: "A **slice** such as `M[0:2, 1:3]` is a cardboard window laid over part of the tray: you see fewer eggs, but they are the *same* eggs. Paint one through the window and it is painted in the tray too. That window is called a **view**. A **copy** is a second tray with the eggs duplicated; painting one leaves the other unchanged. Knowing which operations give windows and which give new trays prevents a whole family of confusing bugs.",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "What an ndarray really is" },
      {
        t: "p",
        text: "Every array has four parts: (1) a **data buffer**, one contiguous block of bytes holding the numbers; (2) a **dtype**, saying how to interpret each group of bytes (for example 8 bytes as a 64-bit integer); (3) a **shape**, a tuple giving the length of each axis; and (4) **strides**, a tuple giving how many bytes to skip to move one step along each axis. Two arrays may point at the same buffer with different shapes and strides — that is exactly what a view is.",
      },
      { t: "h", text: "Why NumPy is fast — a real measurement" },
      {
        t: "p",
        text: "The program below computes the sum of squares of the integers 0 to 999,999 in two ways and times each (keeping the fastest of 5 runs, which is the least disturbed by other programs). It also shows the two meanings of `+` and compares memory use.",
      },
      { t: "code", lang: "python", title: "list_vs_array_speed.py (timings vary between runs)", code: SPEED, output: SPEED_OUT },
      {
        t: "p",
        text: "The speed-up (14× in this run; 17–20× in other runs on the same machine) comes from three things. The loop runs in compiled C instead of the Python interpreter; there is no type check per element because every element has the same dtype; and the numbers are adjacent in memory, so the CPU can load them in large chunks and use **SIMD** instructions (*single instruction, multiple data*: one instruction that adds several numbers at once). The list costs 36,056 bytes for 1,000 integers because each integer is a full Python object (28 bytes here) plus an 8-byte pointer to it; the array stores 8 raw bytes per integer.",
      },
      { t: "h", text: "Creating arrays" },
      {
        t: "table",
        head: ["Call", "Result", "Typical use"],
        rows: [
          ["`np.array([[1, 2], [3, 4]])`", "array from (nested) lists; dtype inferred", "small hand-written examples"],
          ["`np.zeros((2, 3))`, `np.ones(5)`", "filled with 0.0 or 1.0 (float64 by default)", "initialising biases, accumulators, masks"],
          ["`np.full((2, 2), 7)`, `np.empty(3)`", "filled with 7; uninitialised (whatever bytes were in memory)", "constants; fast allocation you will overwrite"],
          ["`np.arange(0, 10, 2)`", "`[0 2 4 6 8]` — start, stop (excluded), step", "index ranges, integer grids"],
          ["`np.linspace(0, 1, 5)`", "`[0. 0.25 0.5 0.75 1.]` — 5 points, both ends included", "x-axes for plots, evenly spaced values"],
          ["`rng = np.random.default_rng(0)`", "a random **Generator** with seed 0", "reproducible randomness"],
          ["`rng.normal(size=(64, 10))`, `rng.integers(0, 10, size=5)`, `rng.random(3)`", "Gaussian, integer and uniform [0, 1) samples", "weight initialisation, synthetic data, shuffling"],
        ],
      },
      {
        t: "p",
        text: "A **seed** is the starting number of a random generator; the same seed produces the same sequence every run, which makes experiments reproducible. `np.random.default_rng(seed)` is the modern interface; the older global functions such as `np.random.rand` still work but share one hidden global state, which makes code harder to reproduce.",
      },
      { t: "h", text: "dtype: the type of every element" },
      {
        t: "table",
        head: ["dtype", "Bytes", "Range / precision", "Used in deep learning for"],
        rows: [
          ["`bool`", "1", "True / False", "masks (which tokens are padding, which pixels to keep)"],
          ["`uint8`", "1", "0 … 255", "raw image pixels as they come out of a JPEG/PNG decoder"],
          ["`int8`", "1", "−128 … 127", "quantised weights for fast inference"],
          ["`int32` / `int64`", "4 / 8", "about ±2.1×10⁹ / ±9.2×10¹⁸", "labels, token ids, indices, counts"],
          ["`float16` / `bfloat16`", "2", "about 3 / 2–3 significant digits", "mixed-precision training on GPUs"],
          ["`float32`", "4", "about 7 significant digits, max ≈ 3.4×10³⁸", "**the default for weights, activations and gradients**"],
          ["`float64`", "8", "about 16 significant digits", "NumPy's default; scientific computing, numerical checks"],
        ],
      },
      {
        t: "p",
        text: "**Why float32 dominates deep learning.** A network with 100 million weights needs 400 MB in float32 but 800 MB in float64, and the same doubling applies to activations and gradients. GPUs also perform float32 (and float16) arithmetic many times faster than float64. Training with small noisy gradient steps does not need 16 significant digits, so float32 is the standard trade-off — and PyTorch layers use it by default. NumPy, by contrast, defaults to float64, which is the source of a common dtype-mismatch error shown in the PyTorch tab.",
      },
      {
        t: "p",
        text: "Two dangers come with fixed-size types. **Overflow**: an integer type wraps around when a result does not fit. `int8` 120 + 10 gives −126, and `uint8` 200 + 100 gives 44 — no error, no warning, just a wrong number. This bites image code that brightens `uint8` pixels directly. **Limited precision**: float32 stores 24 significant bits, so whole numbers above $2^{24} = 16{,}777{,}216$ cannot all be represented — in float32, 16,777,216 + 1 is still 16,777,216 — and adding 0.1 ten times gives 1.0000001, not exactly 1.",
      },
      { t: "h", text: "shape, ndim, size and reshape" },
      {
        t: "steps",
        items: [
          { title: "shape", text: "A tuple with the length of each axis. `np.arange(12).reshape(3, 4).shape` is `(3, 4)`. A 1-D array of 6 elements has shape `(6,)` — the comma makes it a one-element tuple." },
          { title: "ndim", text: "The number of axes: `len(shape)`. A vector has ndim 1, a table 2, an image batch 4." },
          { title: "size", text: "The total number of elements: the product of the shape. `(3, 4)` has size 12; `(8, 3, 32, 32)` has size 24,576." },
          { title: "reshape", text: "Gives the same elements a new shape with the same size. NumPy reads the elements in **row-major order** (last axis fastest — along a row, then the next row) and refills the new shape in that order. If the data does not need to move, the result is a view." },
          { title: "-1 in reshape", text: "One entry may be `-1`, meaning 'work this out for me': `reshape(2, -1)` on 12 elements gives `(2, 6)` because 12 / 2 = 6. If the division is not exact NumPy raises `cannot reshape array of size 12 into shape (5,newaxis)`." },
        ],
      },
      { t: "h", text: "Indexing and slicing in 1, 2 and 3 dimensions" },
      {
        t: "p",
        text: "Inside the square brackets you give **one index per axis**, separated by commas. Each index is either an integer (pick one position, and that axis disappears from the result) or a slice `start:stop:step` (keep a range, and the axis stays). Negative numbers count from the end. Missing trailing indices mean 'everything', and `...` (Ellipsis) means 'as many `:` as needed'.",
      },
      {
        t: "table",
        head: ["Expression (M is 3×4: 0…11)", "Result", "Shape", "Why"],
        rows: [
          ["`M[1, 2]`", "6", "`()` — a single number", "row 1, column 2: both axes picked"],
          ["`M[1]` (same as `M[1, :]`)", "`[4 5 6 7]`", "`(4,)`", "row 1; the row axis is removed"],
          ["`M[:, 2]`", "`[2 6 10]`", "`(3,)`", "column 2 of every row"],
          ["`M[0:2, 1:3]`", "`[[1 2] [5 6]]`", "`(2, 2)`", "rows 0–1, columns 1–2 (stops excluded)"],
          ["`M[:, ::2]`", "columns 0 and 2", "`(3, 2)`", "step 2 along axis 1"],
          ["`M[-1, -1]`", "11", "`()`", "last row, last column"],
          ["`T[1, 2, 3]` (T is 2×3×4)", "23", "`()`", "block 1, row 2, column 3"],
          ["`T[1]`, `T[:, 0, :]`", "a 3×4 block; row 0 of each block", "`(3, 4)`, `(2, 4)`", "integer index removes its axis"],
        ],
      },
      { t: "h", text: "Fancy and boolean indexing" },
      {
        t: "p",
        text: "**Fancy indexing** means indexing with a list or array of integers: `M[[0, 2]]` picks rows 0 and 2; `M[[0, 1, 2], [3, 2, 1]]` picks the *pairs* (0, 3), (1, 2), (2, 1) and returns `[3 6 9]`. **Boolean indexing** (also called **masking**) indexes with an array of True/False of the same shape: `M[M % 2 == 0]` returns the even elements as a 1-D array, and `M[M > 8] = 8` overwrites every element above 8. Combine conditions with `&` (and), `|` (or) and `~` (not), with parentheses: `(a > 1) & (a < 4)`. Python's `and` does not work on arrays.",
      },
      { t: "h", text: "Views versus copies" },
      {
        t: "table",
        head: ["Operation", "View or copy?", "Check"],
        rows: [
          ["basic slice `a[1:5]`, `M[:, 2]`, `M[::2]`", "**view**", "`np.shares_memory(a, a[1:5])` → True"],
          ["`reshape` of a contiguous array, `ravel`, `.T`, `np.broadcast_to`", "**view** (when possible)", "True"],
          ["fancy index `a[[0, 2]]`, boolean mask `a[a > 3]`", "**copy**", "False"],
          ["`.copy()`, `flatten()`, `astype(...)` to a new dtype", "**copy**", "False"],
          ["arithmetic `a + 1`, `a * b`", "new array", "False"],
          ["plain assignment `b = a`", "**the same object** (not even a new view)", "`b is a` → True"],
        ],
      },
      {
        t: "p",
        text: "A view is cheap (no data is copied, whatever the size), but writing through it changes the original. In the scratch program, `row = M[0]` followed by `row[0] = 100` changes `M[0, 0]` to 100. Call `.copy()` when you need an independent array.",
      },
      { t: "h", text: "Axis semantics: what axis=0 and axis=1 mean" },
      {
        t: "p",
        text: "A **reduction** (sum, mean, max, …) along an axis combines all the entries *along that axis* and removes it. For a 2-D array, `axis=0` runs **down the rows**, so it collapses the rows and leaves one number per column; `axis=1` runs **across the columns**, leaving one number per row. With `S = [[1, 2, 3], [4, 5, 6]]`:",
      },
      {
        t: "table",
        head: [" ", "col 0", "col 1", "col 2", "sum(axis=1) → one per row"],
        rows: [
          ["row 0", "1", "2", "3", "**6**"],
          ["row 1", "4", "5", "6", "**15**"],
          ["sum(axis=0) → one per column", "**5**", "**7**", "**9**", "S.sum() = 21"],
        ],
      },
      {
        t: "p",
        text: "A good habit: the axis you name is the one that **disappears**. `S.sum(axis=0)` has shape `(3,)`: axis 0 (length 2) is gone. For an image batch of shape `(N, C, H, W)`, `x.mean(axis=(0, 2, 3))` collapses the batch, height and width axes and leaves one mean per channel — exactly how per-channel normalisation statistics are computed.",
      },
      { t: "h", text: "Joining arrays: concatenate versus stack" },
      {
        t: "list",
        items: [
          "`np.concatenate([p, q], axis=k)` glues arrays along an **existing** axis; all other axis lengths must match. Two `(3,)` arrays give `(6,)`; two `(2, 3)` arrays along axis 0 give `(4, 3)` (`np.vstack`), along axis 1 give `(2, 6)` (`np.hstack`).",
          "`np.stack([p, q], axis=k)` creates a **new** axis at position k; all inputs must have the same shape. Two `(3,)` arrays give `(2, 3)` with `axis=0`, `(3, 2)` with `axis=1`. Stacking 64 images of shape `(3, 32, 32)` gives a batch `(64, 3, 32, 32)` — this is what a data loader does.",
        ],
      },
      { t: "h", text: "From arrays to deep-learning tensors" },
      {
        t: "p",
        text: "In deep learning a **tensor** is simply an n-dimensional array with a dtype (the word is borrowed from mathematics, but you need nothing more than this). A batch of colour images is a 4-D tensor in **NCHW** layout: **N** images, **C** channels (3 for red, green, blue), **H** rows and **W** columns. `images[0]` is one image `(3, 32, 32)`; `images[:, 0]` is the red channel of every image `(8, 32, 32)`; `images.reshape(8, -1)` flattens each image into a 3,072-long vector for a dense layer. Image decoders produce `uint8` values 0–255; before training they are converted with `astype(np.float32) / 255`, which quadruples the memory (24,576 → 98,304 bytes for this batch). [Images as Tensors](/learn/m08/images-as-tensors) covers channels and the alternative NHWC layout in depth.",
      },
      { t: "h", text: "NumPy and PyTorch share memory" },
      {
        t: "p",
        text: "`torch.from_numpy(a)` creates a tensor that **uses the same buffer** as the NumPy array — no copy. Changing one changes the other, and `tensor.numpy()` goes back the same way for CPU tensors. `torch.tensor(a)` always copies. The PyTorch tab proves this by comparing memory addresses and by editing one side and reading the other.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Notation" },
      {
        t: "table",
        head: ["Symbol", "Meaning"],
        rows: [
          ["$d$", "number of axes (`ndim`)"],
          ["$(s_0, s_1, \\dots, s_{d-1})$", "the shape: $s_k$ is the length of axis $k$"],
          ["$(i_0, \\dots, i_{d-1})$", "an index, with $0 \\le i_k < s_k$"],
          ["$b$", "bytes per element (`itemsize`): 1 for int8/uint8/bool, 4 for float32/int32, 8 for float64/int64"],
          ["$(t_0, \\dots, t_{d-1})$", "the strides in bytes: $t_k$ = bytes to skip to increase $i_k$ by one"],
        ],
      },
      { t: "h", text: "Size and memory" },
      { t: "math", tex: String.raw`\text{size} = \prod_{k=0}^{d-1} s_k, \qquad \text{nbytes} = b \cdot \text{size}`, caption: "Memory grows with the product of all axis lengths — doubling the image side length quadruples the memory." },
      { t: "h", text: "Row-major strides and the address of an element" },
      {
        t: "math",
        tex: String.raw`t_{d-1} = b, \qquad t_k = t_{k+1}\, s_{k+1} = b \prod_{j=k+1}^{d-1} s_j, \qquad \text{offset}(i_0,\dots,i_{d-1}) = \sum_{k=0}^{d-1} i_k\, t_k`,
        caption: "In row-major (C) order the last axis is adjacent in memory; each earlier axis jumps over a whole block of later axes.",
      },
      {
        t: "p",
        text: "This formula is why many operations are free. A **transpose** only swaps the strides; a slice with step 2 doubles a stride; reading a size-1 axis with stride 0 repeats one value (that is broadcasting, the subject of the [next lesson](/learn/m00/vectorization)). None of them moves data. An array whose strides equal the row-major formula is called **C-contiguous**; a transposed array is not, and operations that need contiguous memory (for example PyTorch's `.view`) must copy it first.",
      },
      { t: "h", text: "reshape with -1" },
      { t: "math", tex: String.raw`s_{\text{missing}} = \frac{\text{size}}{\prod_{k \ne \text{missing}} s_k}, \qquad \text{valid only if the division is exact}` },
      { t: "h", text: "Reductions along an axis" },
      {
        t: "math",
        tex: String.raw`\big(\textstyle\sum_{\text{axis}=0} S\big)_j = \sum_{i} S_{ij}, \qquad \big(\textstyle\sum_{\text{axis}=1} S\big)_i = \sum_{j} S_{ij}, \qquad \text{mean} = \frac{\text{sum}}{\text{length of the reduced axis}}`,
        caption: "The summed index disappears from the result; the remaining index labels the output.",
      },
      { t: "h", text: "Fixed-width numbers" },
      {
        t: "math",
        tex: String.raw`\text{int8: } x \mapsto \big((x + 128) \bmod 256\big) - 128, \qquad \text{float32: } x = (-1)^{\text{sign}} \times 1.f \times 2^{e - 127},\ f \text{ has 23 bits}`,
        caption: "Integers wrap around modulo 256 (for 8 bits). A float32 keeps 24 significant bits (the hidden leading 1 plus 23 stored), so its relative spacing is ε = 2⁻²³ ≈ 1.19 × 10⁻⁷.",
      },
      { t: "h", text: "Worked example — every number" },
      {
        t: "steps",
        items: [
          { title: "Shape, size, memory", text: "`M = np.arange(12, dtype=np.int64).reshape(3, 4)`: $d = 2$, shape $(3, 4)$, size $3 \\times 4 = 12$, $b = 8$, nbytes $= 8 \\times 12 = 96$." },
          { title: "Strides", tex: String.raw`t_1 = b = 8, \qquad t_0 = t_1 \cdot s_1 = 8 \times 4 = 32 \quad\Rightarrow\quad \text{strides} = (32, 8)` },
          { title: "Address of M[2, 1]", tex: String.raw`\text{offset} = 2 \times 32 + 1 \times 8 = 72 \text{ bytes} = 72 / 8 = \text{element } 9, \text{ and indeed } M_{2,1} = 9` },
          { title: "Transpose", text: "`M.T` has shape $(4, 3)$ and strides $(8, 32)$: the same buffer read in a different order, so `np.shares_memory(M, M.T)` is True and `M.T` is not C-contiguous." },
          { title: "reshape(2, -1)", tex: String.raw`s_{\text{missing}} = 12 / 2 = 6 \Rightarrow (2, 6); \qquad \text{reshape}(5, -1): 12/5 = 2.4 \Rightarrow \text{error}` },
          { title: "Axis sums of S = [[1, 2, 3], [4, 5, 6]]", tex: String.raw`\textstyle\sum_{\text{axis}=0} = (1{+}4,\ 2{+}5,\ 3{+}6) = (5, 7, 9), \qquad \sum_{\text{axis}=1} = (1{+}2{+}3,\ 4{+}5{+}6) = (6, 15)` },
          { title: "Means", tex: String.raw`\text{mean}_{\text{axis}=0} = (5, 7, 9)/2 = (2.5, 3.5, 4.5), \qquad \text{mean}_{\text{axis}=1} = (6, 15)/3 = (2, 5)` },
          { title: "Image batch (8, 3, 32, 32), float32", tex: String.raw`\text{size} = 8\cdot3\cdot32\cdot32 = 24{,}576,\quad \text{nbytes} = 4 \times 24{,}576 = 98{,}304,\quad \text{strides} = (12288, 4096, 128, 4)` },
          { title: "Flat position of image 5, channel 2, row 10, column 7", tex: String.raw`((5 \cdot 3 + 2)\cdot 32 + 10)\cdot 32 + 7 = (17 \cdot 32 + 10)\cdot 32 + 7 = 554 \cdot 32 + 7 = 17{,}735` },
          { title: "int8 overflow", tex: String.raw`120 + 10 = 130 \;\rightarrow\; ((130 + 128) \bmod 256) - 128 = 2 - 128 = -126` },
          { title: "uint8 overflow", tex: String.raw`200 + 100 = 300 \;\rightarrow\; 300 \bmod 256 = 44` },
          { title: "float32 limit", tex: String.raw`2^{24} = 16{,}777{,}216;\quad 2^{24} + 1 \text{ needs 25 significant bits, so it rounds back to } 16{,}777{,}216` },
        ],
      },
      { t: "p", text: "The program below checks each of these numbers." },
      { t: "code", lang: "python", title: "memory_layout.py", code: LAYOUT, output: LAYOUT_OUT },
      {
        t: "p",
        text: "Note the last lines: 0.1 cannot be stored exactly in binary, so float32 stores 0.10000000149…, and the small errors add up: ten additions give 1.0000001. This is why you compare floats with a tolerance (`np.allclose`) rather than with `==`.",
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "Two simulators help you *see* shapes. The first shows how NumPy combines arrays of different shapes — it uses exactly the axis and shape vocabulary from this lesson, and the next lesson explains its rules in full. The second shows how shapes govern matrix multiplication.",
      },
      { t: "sim", id: "broadcasting", caption: "Edit the shapes of A and B, step through the shape check, then hover over result cells." },
      {
        t: "list",
        ordered: true,
        items: [
          "Start with the **Bias add** preset and press **Step** until the arrays appear. A has shape (4, 3): 4 rows (axis 0), 3 columns (axis 1). Hover over a result cell and read its index: the first number is the row (axis 0), the second the column (axis 1).",
          "Set A to **3 axes** with shape (2, 3, 4). The grid is drawn as 2 slices labelled `[0, :, :]` and `[1, :, :]` — exactly what `A[0]` and `A[1]` would return. Each slice has 3 rows and 4 columns.",
          "Look at the stored values of A versus the result of a (4, 3) − (1, 3) operation (**Column normalise** preset): one row of B is used for every row of A. That is an axis-0 operation in the same sense as `sum(axis=0)` — it acts down the rows.",
          "Notice the narration in step 4: the faded cells are not stored. This is the stride trick from the math section — reading a size-1 axis with stride 0.",
        ],
      },
      { t: "sim", id: "matmul", caption: "Matrix multiplication: the shapes (n×k)·(k×m) decide whether the product exists and what shape it has." },
      {
        t: "list",
        items: [
          "Change A's columns and B's rows so they differ: the product becomes undefined. In NumPy, `A @ B` would raise a `ValueError` for the same reason.",
          "The full theory of matrix products is in [Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices); here just notice that the result shape is read straight off the input shapes.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "A complete tour of the ndarray in one program: creation, dtypes, shape and reshape, indexing in 1/2/3 dimensions, fancy and boolean indexing, views versus copies (proved with `np.shares_memory`), axis reductions, joining, and a realistic image batch. Every printed value is referred to in the explanations above.",
      },
      { t: "code", lang: "python", title: "numpy_arrays_tour.py", code: TOUR, output: TOUR_OUT },
      { t: "p", text: "Things to notice in the output:" },
      {
        t: "list",
        items: [
          "`np.array([1, 2, 3, 4, 5, 6])` inferred `int64` (on Windows with NumPy 2 and on Linux/macOS; NumPy 1.x on Windows used int32). `np.zeros` defaults to float64, so its values print as `0.`.",
          "The int8 overflow prints −126 silently, and float32 cannot add 1 to 2²⁴ (`1.6777216e+07` is 16,777,216 printed in scientific notation), while float64 gets 16,777,217.",
          "The reshape error message spells out what went wrong: 12 elements cannot be split into rows of 5.",
          "`M[[0, 1, 2], [3, 2, 1]]` paired the two index lists position by position: (0, 3) → 3, (1, 2) → 6, (2, 1) → 9.",
          "`row = M[0]` is a view, so `row[0] = 100` changed `M`. Editing `safe = M[0].copy()` did not: `M[0]` still shows 100, not −1 in position 1.",
          "`images.astype(np.float32) / 255.0` turned 24,576 bytes of uint8 into 98,304 bytes of float32 with values between 0.0 and 1.0.",
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
                text: "PyTorch tensors follow NumPy's indexing, reshape and reduction rules (PyTorch calls the axis argument `dim`). The program proves that `torch.from_numpy` shares memory — edits on either side are visible on the other, and the data pointers are equal — while `torch.tensor` copies. It also shows the float64/float32 mismatch that appears when NumPy data meets a PyTorch layer.",
              },
              { t: "code", lang: "python", title: "numpy_torch_bridge.py", code: TORCH, output: TORCH_OUT },
              {
                t: "callout",
                kind: "warning",
                text: "Because `from_numpy` shares memory, an in-place edit of the NumPy array after you created the tensor silently changes your training data. If you are not sure who else holds the array, use `torch.tensor(a)` (copy) or call `.clone()` on the tensor. `.float()` and `.to(device)` also copy when they change the dtype or move to a GPU.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow converts NumPy arrays with `tf.convert_to_tensor` (or implicitly when you pass an array to a layer) and back with `.numpy()`. TensorFlow tensors are **immutable** — they cannot be edited in place — so do not rely on memory sharing in either direction. Keras image layers expect **channels-last** batches `(N, H, W, C)`, unlike PyTorch's `(N, C, H, W)`.",
              },
              {
                t: "code",
                lang: "python",
                title: "numpy_tf_bridge.py (not executed in CI)",
                code: TF_CODE,
                output: "Prints the tensor dtype float32 and shape (2, 3); the column sums [3. 5. 7.]; the type numpy.ndarray with shape (2, 3); and the Conv2D output shape (8, 30, 30, 16).",
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Array-like containers you will meet" },
      {
        t: "table",
        head: ["Container", "Strengths", "Weaknesses", "Use it for"],
        rows: [
          ["Python `list`", "any mix of types; cheap `append`", "slow arithmetic; `+` concatenates; ~4.5× the memory for ints", "collecting results before converting with `np.array`"],
          ["NumPy `ndarray`", "fast vectorised maths; huge ecosystem; views", "CPU only; no automatic gradients", "data loading, preprocessing, metrics, quick experiments"],
          ["PyTorch `Tensor`", "NumPy-like API + GPU + autograd", "heavier dependency", "models and training"],
          ["TensorFlow `tf.Tensor`", "GPU/TPU + autograd + graph compilation", "immutable; different layout defaults", "Keras/TensorFlow models"],
          ["JAX `Array`", "NumPy API (`jax.numpy`) + GPU/TPU + compilation", "immutable: `x.at[i].set(v)` instead of `x[i] = v`", "research code, large-scale training"],
          ["pandas `DataFrame`", "labelled columns of different dtypes; joins, group-by", "slower; not for n-D tensors", "tabular data before it becomes a feature array"],
        ],
      },
      { t: "h", text: "Memory orders" },
      {
        t: "table",
        head: ["Order", "Which axis is adjacent in memory", "Default in", "Notes"],
        rows: [
          ["C (row-major)", "the last axis", "NumPy, PyTorch, C", "`np.ascontiguousarray` makes one"],
          ["Fortran (column-major)", "the first axis", "Fortran, MATLAB, R", "`np.asfortranarray`; `order='F'` in reshape"],
          ["Strided view", "any — described by the strides", "result of `.T`, slicing, broadcasting", "free to create; some operations copy it first"],
        ],
      },
      { t: "h", text: "Ways to index" },
      {
        t: "table",
        head: ["Kind", "Example", "Returns", "Can assign?"],
        rows: [
          ["Integer", "`M[1, 2]`", "a scalar (axis removed)", "yes"],
          ["Slice", "`M[0:2, ::2]`", "a view", "yes, writes into the original"],
          ["Fancy (integer array)", "`M[[0, 2]]`", "a copy", "yes when used directly on the left: `M[[0, 2]] = 0`"],
          ["Boolean mask", "`M[M > 8]`", "a 1-D copy of the selected elements", "yes: `M[M > 8] = 8`"],
          ["`None` / `np.newaxis`", "`v[:, None]`", "a view with a new size-1 axis", "—"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Data loading.** Image libraries such as Pillow (via `np.asarray(img)`) and OpenCV return images as NumPy arrays of shape `(H, W, C)` and dtype uint8 — OpenCV in BGR channel order. Audio files load as 1-D float arrays of samples.",
          "**Preprocessing.** Normalising, cropping (`img[y0:y1, x0:x1]`), flipping (`img[:, ::-1]`), padding and batching (`np.stack`) are all array operations, usually done in NumPy before the data becomes a tensor.",
          "**Classical machine learning.** scikit-learn takes and returns NumPy arrays: features `X` of shape `(n_samples, n_features)` and labels `y` of shape `(n_samples,)`.",
          "**Evaluation.** Accuracy is `(pred == y).mean()` on arrays; confusion matrices, precision/recall and ROC curves are computed with NumPy reductions and boolean masks.",
          "**Model interchange.** Model weights are routinely exported to and loaded from NumPy arrays (for example `.npy`/`.npz` files), and PyTorch's `DataLoader` accepts datasets that return NumPy arrays and converts them to tensors.",
          "**Masks in models.** Boolean arrays mark padding tokens in a batch of sentences or the valid pixels in a segmentation map; the model multiplies or fills by these masks.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Speed**: compiled loops over contiguous typed memory — an order of magnitude or more over Python loops, even for a simple sum.",
          "**Compactness**: 1 to 8 bytes per number instead of a full Python object plus a pointer.",
          "**Expressiveness**: slicing, masking and axis arguments replace pages of loop code with one line.",
          "**Zero-copy views**: reshaping, transposing and slicing large arrays costs almost nothing.",
          "**Lingua franca**: PyTorch, TensorFlow, JAX, scikit-learn, pandas, OpenCV and Matplotlib all read and write NumPy arrays.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**No GPU and no automatic differentiation** — that is what PyTorch/TensorFlow/JAX add.",
          "**Fixed size and fixed dtype**: `np.append` builds a brand-new array every call, so appending in a loop is slow; collect in a list and convert once.",
          "**Silent numeric surprises**: integer overflow wraps without an error, and float precision is finite.",
          "**Views can surprise**: an in-place edit through a view changes data you thought was separate.",
          "**Temporary arrays**: an expression like `a * b + c` allocates intermediate arrays, which costs memory and time for very large data.",
          "**Mostly single-threaded element-wise operations**: matrix products use multi-threaded BLAS libraries, but simple element-wise ufuncs typically run on one core.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Thinking `b = a` makes a copy.** It only gives the same array a second name; `b[0] = -1` changes `a`. **Fix:** `b = a.copy()` when you need independence.",
          "**Editing a slice and being surprised the original changed.** Slices are views. **Fix:** `.copy()` the slice, or check with `np.shares_memory(a, b)`.",
          "**Assigning through chained fancy indexing.** `a[[0, 1]][0] = 5` changes a temporary copy and leaves `a` untouched (with `a = np.arange(6)`, `a` is still `[0 1 2 3 4 5]` afterwards). **Fix:** use one indexing operation on the left: `a[[0, 1]] = ...` or `a[0] = 5`.",
          "**Confusing reshape with transpose.** `np.arange(6).reshape(2, 3).reshape(3, 2)` gives `[[0, 1], [2, 3], [4, 5]]`, but the transpose is `[[0, 3], [1, 4], [2, 5]]`. Reshape keeps row-major order; transpose swaps axes. **Fix:** use `.T` or `np.transpose`/`np.moveaxis` to reorder axes (e.g. NHWC → NCHW).",
          "**Arithmetic on uint8 images.** `img + 100` wraps around (200 + 100 = 44), making bright pixels dark. **Fix:** convert first: `img.astype(np.float32)`, then clip back with `np.clip` if needed.",
          "**Feeding float64 arrays to PyTorch layers.** `RuntimeError: mat1 and mat2 must have the same dtype`. **Fix:** create data as float32 or call `.float()` / `astype(np.float32)`.",
          "**Using `and`/`or` on arrays.** `(a > 1) and (a < 3)` raises 'The truth value of an array with more than one element is ambiguous'. **Fix:** `(a > 1) & (a < 3)` — with parentheses, because `&` binds more tightly than `>`.",
          "**Mixing up (n,) and (n, 1).** A shape `(5,)` vector and a `(5, 1)` column look alike when printed small but behave differently in arithmetic (the next lesson shows a (3,1) − (3,) subtraction silently producing a 3×3 result). **Fix:** print `.shape` often and reshape deliberately.",
          "**Choosing the wrong axis.** `x.mean(axis=1)` on a `(batch, features)` array averages over features, not over the batch. **Fix:** remember 'the named axis disappears' and check the result's shape.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "Why use NumPy arrays instead of Python lists for numerical work?",
        a: "Arrays store same-typed numbers contiguously and run their loops in compiled code, so arithmetic is much faster (14–20× for a simple sum of squares in this lesson's measurement, far more for heavier operations) and uses less memory (8 bytes per int64 versus about 36 for a list). Arrays also have numeric semantics: `+` adds element-wise instead of concatenating, and there are built-in reductions, linear algebra and random numbers.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What are shape, ndim, size and dtype of `np.zeros((8, 3, 32, 32), dtype=np.float32)`?",
        a: "shape `(8, 3, 32, 32)`; ndim 4 (number of axes); size 8·3·32·32 = 24,576 elements; dtype float32 (4 bytes each), so nbytes = 98,304. In deep learning this is a batch of 8 RGB 32×32 images in NCHW layout.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "For a 2-D array, what is the difference between `sum(axis=0)` and `sum(axis=1)`?",
        a: "`axis=0` sums down the rows, removing the row axis and leaving one total per column; `axis=1` sums across the columns, leaving one total per row. For `[[1, 2, 3], [4, 5, 6]]` they give `[5 7 9]` and `[6 15]`. The named axis is the one that disappears.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Which NumPy operations return views and which return copies? How can you check?",
        a: "Basic slicing (`a[1:5]`, `a[:, 2]`, `a[::2]`), `reshape`/`ravel` when the data is contiguous, transposes and `np.broadcast_to` return views. Fancy indexing with integer arrays, boolean masking, `.copy()`, `flatten()`, `astype` to a new type and arithmetic return new arrays. `np.shares_memory(a, b)` (or `b.base is a`) tells you whether two arrays overlap in memory.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why is float32 the default precision in deep learning, while NumPy defaults to float64?",
        a: "NumPy targets general scientific computing, where 16 significant digits are often needed. Neural network training uses noisy stochastic gradients, so about 7 significant digits are sufficient, and float32 halves memory for weights, activations and gradients and runs much faster on GPUs (16-bit formats are faster still, used in mixed precision). The mismatch means NumPy data usually has to be cast to float32 before entering a PyTorch model.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "What does `reshape(-1, 3)` do, and how is reshape different from transpose?",
        a: "`-1` asks NumPy to infer that axis: size divided by 3 (it must divide exactly). Reshape keeps the elements in row-major order and only changes how they are grouped; transpose reorders the axes, so element (i, j) moves to (j, i). Reshaping a (2, 3) array to (3, 2) is therefore not its transpose.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "What are strides, and how do they make transposes, slicing and broadcasting free?",
        a: "Strides give, per axis, the number of bytes to jump to move one index along that axis; the address of an element is the sum of index × stride. A transpose swaps the strides; `a[::2]` doubles one stride; a reversed slice uses a negative stride; broadcasting gives a size-1 axis a stride of 0 so every index reads the same element. Each is just a new header over the same buffer — O(1) work. The price is that the result may not be contiguous, so operations needing contiguous memory copy it (`np.ascontiguousarray`, PyTorch `.contiguous()`).",
      },
      {
        t: "qa",
        level: "advanced",
        q: "`torch.from_numpy(a)` shares memory with `a`. What can go wrong, and when do you lose the sharing?",
        a: "Any in-place change to `a` (data augmentation, normalising in place, reusing a buffer) silently changes the tensor and vice versa, which can corrupt a training batch or a cached dataset. Sharing is lost as soon as an operation must produce new memory: `.float()` on float64 data, `.to('cuda')`, `.contiguous()` on a non-contiguous tensor, or any arithmetic. Also, `from_numpy` requires a NumPy dtype that PyTorch supports, and a read-only array (for example from `np.frombuffer` on bytes) produces a warning because tensors assume writability. Use `torch.tensor(a)` or `.clone()` when you need isolation.",
      },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "What is the shape of `np.zeros((3, 4)).sum(axis=0)`?",
            options: ["(3,)", "(4,)", "(3, 4)", "() — a single number"],
            answer: 1,
            explain: "Axis 0 (length 3) is summed away, leaving one value per column: shape (4,).",
          },
          {
            kind: "numeric",
            q: "How many bytes does a float32 array of shape (10, 20, 3) occupy?",
            answer: 2400,
            tolerance: 0,
            explain: "size = 10·20·3 = 600 elements, × 4 bytes = 2,400 bytes.",
          },
          {
            q: "Which expression returns a **view** of `a` (shares its memory)?",
            options: ["a[[1, 2]]", "a[a > 1]", "a[1:3]", "a.copy()"],
            answer: 2,
            explain: "Basic slices are views. Fancy indexing and boolean masks build new arrays, and `.copy()` is a copy by definition.",
          },
          {
            kind: "numeric",
            q: "An array has 24 elements. In `a.reshape(4, -1)`, what size does NumPy infer for the -1 axis?",
            answer: 6,
            tolerance: 0,
            explain: "24 / 4 = 6, so the result has shape (4, 6).",
          },
          {
            kind: "numeric",
            q: "What value does `np.array([100], dtype=np.int8) + np.int8(100)` contain?",
            answer: -56,
            tolerance: 0,
            explain: "100 + 100 = 200 does not fit in int8 (max 127), so it wraps: 200 − 256 = −56.",
          },
          {
            q: "You create `t = torch.from_numpy(a)` and then run `a[0] = 7`. What is `t[0]`?",
            options: ["The old value, because from_numpy copies", "7, because the tensor shares a's memory", "An error is raised", "It depends on the GPU"],
            answer: 1,
            explain: "`torch.from_numpy` wraps the same buffer, so changes to the array are visible in the tensor.",
          },
          {
            kind: "numeric",
            q: "A C-contiguous float32 array has shape (5, 6). What is its stride along axis 0, in bytes?",
            answer: 24,
            tolerance: 0,
            explain: "Moving one row skips a whole row of 6 elements × 4 bytes = 24 bytes. The stride along axis 1 is 4.",
          },
          {
            q: "`images` has shape (16, 3, 64, 64) (NCHW). What is the shape of `images[:, 1]`?",
            options: ["(16, 64, 64)", "(3, 64, 64)", "(16, 1, 64, 64)", "(1, 64, 64)"],
            answer: 0,
            explain: "The integer 1 picks the green channel and removes the channel axis; the batch, height and width axes remain.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Shapes, bytes and addresses by hand",
        prompt: "A training batch is stored as a C-contiguous float32 array of shape (64, 3, 224, 224). (a) How many elements and how many bytes does it hold? (b) What are its strides? (c) If you converted it to float64, how many bytes would it need?",
        hint: "Multiply the shape for the size. The last stride is the item size; each earlier stride is the next stride times the next axis length.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) Size and bytes", tex: String.raw`64 \cdot 3 \cdot 224 \cdot 224 = 9{,}633{,}792 \text{ elements}, \qquad 4 \times 9{,}633{,}792 = 38{,}535{,}168 \text{ bytes} \approx 36.75 \text{ MiB}` },
              { title: "(b) Strides", tex: String.raw`t_3 = 4,\quad t_2 = 4 \cdot 224 = 896,\quad t_1 = 896 \cdot 224 = 200{,}704,\quad t_0 = 200{,}704 \cdot 3 = 602{,}112` },
              { title: "(c) float64", tex: String.raw`8 \times 9{,}633{,}792 = 77{,}070{,}336 \text{ bytes} \approx 73.5 \text{ MiB — twice as much}` },
            ],
          },
          { t: "p", text: "You can confirm (b) with `np.zeros((64, 3, 224, 224), dtype=np.float32).strides`, which prints `(602112, 200704, 896, 4)`." },
        ],
      },
      {
        t: "exercise",
        title: "Augment and filter an image batch with slicing and masks",
        prompt: "Create a reproducible batch of 6 random RGB images of shape (6, 3, 32, 32) in float32, and make image 2 dark by multiplying it by 0.2. Then: (1) take a 16×16 centre crop of every image and show it is a view; (2) flip every image horizontally using only slicing; (3) compute each image's mean brightness over channels, rows and columns; (4) keep only the images with brightness above 0.3 using a boolean mask, and show the result is a copy.",
        hint: "Centre crop of 32 pixels to 16: rows and columns 8 to 23, i.e. `8:24`. A horizontal flip reverses the last axis: `[..., ::-1]`. Use `axis=(1, 2, 3)` for the brightness.",
        solution: [
          { t: "code", lang: "python", title: "augment_batch.py", code: EX_CODE, output: EX_OUT },
          {
            t: "p",
            text: "The flip has stride −4: NumPy walks the last axis backwards over the same buffer, so the flip is free. The brightness of the darkened image is about 0.2 × 0.5 = 0.1, so the mask keeps 5 of 6 images, and because boolean indexing gathers selected elements into a new array, `bright` is a copy.",
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          "An **ndarray** = one contiguous buffer + **dtype** + **shape** + **strides**. Lists are slow and `+` concatenates them; arrays are fast and `+` adds.",
          "Create with `np.array`, `zeros`, `ones`, `arange` (stop excluded), `linspace` (both ends included) and `np.random.default_rng(seed)`.",
          "**float32** is the deep-learning default (half the memory of float64, fast on GPUs); NumPy defaults to float64 — cast before PyTorch. Integers overflow silently; float32 has about 7 significant digits.",
          "`shape`, `ndim`, `size`; `reshape` keeps row-major order and allows one `-1`.",
          "One index per axis: integers remove the axis, slices `start:stop:step` keep it. Fancy indexing picks by integer lists; boolean masks pick by condition.",
          "Slices, reshape and `.T` are **views** (shared memory); fancy/boolean indexing and `.copy()` are **copies**. Check with `np.shares_memory`.",
          "A reduction along an axis removes that axis: `axis=0` → one result per column, `axis=1` → one per row.",
          "`concatenate` joins along an existing axis; `stack` creates a new one.",
          "Image batches are 4-D tensors (N, C, H, W). `torch.from_numpy` shares memory with the array; `torch.tensor` copies.",
        ],
      },
      { t: "math", tex: String.raw`\text{nbytes} = b\prod_k s_k, \qquad t_k = b\prod_{j>k} s_j, \qquad \text{offset} = \sum_k i_k t_k` },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "p",
        text: "Next, [Vectorization and Broadcasting](/learn/m00/vectorization) turns these arrays into a way of thinking: replacing Python loops with whole-array operations, and the exact rules NumPy uses to combine arrays of different shapes (which the simulator above already previewed). After that, [Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) gives the mathematics behind arrays — dot products and matrix multiplication — and [Images as Tensors](/learn/m08/images-as-tensors) returns to the (N, C, H, W) batch when you start working with convolutional networks.",
      },
    ],
  },
};

export default lesson;
