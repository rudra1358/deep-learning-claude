import type { Lesson } from "@/lib/types";

const ANSCOMBE = String.raw`import os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # draw into memory, no window needed
import matplotlib.pyplot as plt

# Anscombe's quartet (F. J. Anscombe, 1973): four small datasets of 11 points.
x123 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
quartet = {
    "I":   (x123, np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])),
    "II":  (x123, np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])),
    "III": (x123, np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])),
    "IV":  (np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float),
            np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])),
}

print("set   mean_x  var_x  mean_y  var_y   corr   slope  intercept")
for name, (x, y) in quartet.items():
    slope, intercept = np.polyfit(x, y, deg=1)      # least-squares straight line
    corr = np.corrcoef(x, y)[0, 1]                  # Pearson correlation
    print(f"{name:>3}  {x.mean():7.2f} {x.var(ddof=1):6.2f} {y.mean():7.2f} {y.var(ddof=1):6.3f}"
          f" {corr:6.3f} {slope:7.3f} {intercept:9.3f}")

# The numbers above are (almost) identical. Now LOOK at the data.
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
line_x = np.array([3, 20])
for ax, (name, (x, y)) in zip(axes.flat, quartet.items()):
    slope, intercept = np.polyfit(x, y, deg=1)
    ax.scatter(x, y, color="tab:blue")
    ax.plot(line_x, slope * line_x + intercept, color="tab:orange", label="best-fit line")
    ax.set_title(f"Dataset {name}")
fig.suptitle("Same statistics, four different stories")
axes[0, 0].legend()

path = os.path.join(tempfile.mkdtemp(), "anscombe.png")
fig.savefig(path, dpi=100)
print("saved image shape (height, width, channels):", plt.imread(path).shape)`;

const ANSCOMBE_OUT = String.raw`set   mean_x  var_x  mean_y  var_y   corr   slope  intercept
  I     9.00  11.00    7.50  4.127  0.816   0.500     3.000
 II     9.00  11.00    7.50  4.128  0.816   0.500     3.001
III     9.00  11.00    7.50  4.123  0.816   0.500     3.002
 IV     9.00  11.00    7.50  4.123  0.817   0.500     3.002
saved image shape (height, width, channels): (600, 800, 4)`;

const LINES = String.raw`import os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# ---------- 1) The object model: one Figure that holds two Axes ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))   # 1 row, 2 columns of Axes
print("figure size in inches:", fig.get_size_inches(), "| number of Axes:", len(fig.axes))

# ---------- 2) Line plot of functions ----------
z = np.linspace(-6, 6, 200)          # 200 evenly spaced x values
ax1.plot(z, 1 / (1 + np.exp(-z)), label="sigmoid")
ax1.plot(z, np.tanh(z), linestyle="--", label="tanh")
ax1.plot(z, np.maximum(0, z), label="ReLU")
ax1.set_ylim(-1.5, 3)
ax1.set_xlabel("z")
ax1.set_ylabel("activation(z)")
ax1.set_title("Activation functions")
ax1.legend()
ax1.grid(alpha=0.3)
print("lines on ax1:", len(ax1.lines), "| points per line:", len(ax1.lines[0].get_xdata()))

# ---------- 3) A REAL training history: logistic regression by gradient descent ----------
X, y = load_breast_cancer(return_X_y=True)
idx = np.random.default_rng(0).permutation(len(X))
tr, va = idx[:400], idx[400:]
mu, sd = X[tr].mean(0), X[tr].std(0)
X = (X - mu) / sd                    # scale with training statistics

def bce(p, t):                       # binary cross-entropy loss
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

w, b, lr = np.zeros(X.shape[1]), 0.0, 0.5
history = {"train": [], "val": []}
for step in range(300):
    p = 1 / (1 + np.exp(-(X[tr] @ w + b)))
    history["train"].append(bce(p, y[tr]))
    history["val"].append(bce(1 / (1 + np.exp(-(X[va] @ w + b))), y[va]))
    w -= lr * X[tr].T @ (p - y[tr]) / len(tr)
    b -= lr * np.mean(p - y[tr])

steps = np.arange(1, 301)
ax2.plot(steps, history["train"], label="train loss")
ax2.plot(steps, history["val"], label="validation loss")
ax2.set_yscale("log")                # losses shrink by orders of magnitude -> log axis
ax2.set_xlabel("gradient-descent step")
ax2.set_ylabel("cross-entropy (log scale)")
ax2.set_title("Training curves")
ax2.legend()
for s in [1, 10, 100, 300]:
    print(f"step {s:3d}: train {history['train'][s-1]:.4f}  val {history['val'][s-1]:.4f}")
print("best val step:", int(np.argmin(history["val"])) + 1)

fig.tight_layout()
path = os.path.join(tempfile.mkdtemp(), "curves.png")
fig.savefig(path, dpi=100)
print("saved:", os.path.basename(path), "shape", plt.imread(path).shape)`;

const LINES_OUT = String.raw`figure size in inches: [10.  4.] | number of Axes: 2
lines on ax1: 3 | points per line: 200
step   1: train 0.6931  val 0.6931
step  10: train 0.1261  val 0.1284
step 100: train 0.0628  val 0.0835
step 300: train 0.0494  val 0.0803
best val step: 300
saved: curves.png shape (400, 1000, 4)`;

const SCATTER = String.raw`import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target                    # y: 0 = malignant, 1 = benign
area, smooth = X[:, 3], X[:, 4]                  # "mean area", "mean smoothness"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ---------- Scatter plot, coloured by class ----------
for cls, name, colour in [(0, "malignant", "tab:red"), (1, "benign", "tab:blue")]:
    mask = y == cls                              # boolean array: rows of this class
    ax1.scatter(area[mask], smooth[mask], s=12, alpha=0.6, color=colour, label=name)
    print(f"{name:>9}: {mask.sum():3d} points, median area {np.median(area[mask]):7.1f}")
ax1.set_xlabel("mean area")
ax1.set_ylabel("mean smoothness")
ax1.legend()

# ---------- Histogram: counts vs density ----------
counts, edges, _ = ax2.hist(area, bins=20, color="tab:gray", edgecolor="white")
print("bins:", len(counts), "| bin width:", round(edges[1] - edges[0], 2),
      "| counts sum to", int(counts.sum()))
print("tallest bin: %d rows between %.1f and %.1f" % (counts.max(), edges[counts.argmax()], edges[counts.argmax() + 1]))

dens, edges_d = np.histogram(area, bins=20, density=True)   # same bins, height = density
print("density heights x widths sum to:", round(float(np.sum(dens * np.diff(edges_d))), 6))
ax2.set_xlabel("mean area")
ax2.set_ylabel("number of samples")
ax2.set_title("20-bin histogram")
fig.tight_layout()`;

const SCATTER_OUT = String.raw`malignant: 212 points, median area   932.0
   benign: 357 points, median area   458.4
bins: 20 | bin width: 117.88 | counts sum to 569
tallest bin: 136 rows between 379.2 and 497.1
density heights x widths sum to: 1.0`;

const IMSHOW = String.raw`import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

digits = load_digits()                           # 1797 images of handwritten digits, 8x8 pixels
img = digits.images[0]
print("image shape:", img.shape, "| pixel range:", img.min(), "to", img.max(), "| label:", digits.target[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ---------- imshow of an image ----------
im1 = ax1.imshow(img, cmap="gray_r", vmin=0, vmax=16)   # fix the colour scale to the data's 0..16 range
ax1.set_title(f"label = {digits.target[0]}")
fig.colorbar(im1, ax=ax1, label="ink intensity")

# ---------- imshow of a matrix: a confusion matrix ----------
X, y = digits.data / 16.0, digits.target         # flatten to 64 features in [0, 1]
idx = np.random.default_rng(0).permutation(len(X))
tr, te = idx[:1200], idx[1200:]
model = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
pred = model.predict(X[te])

cm = np.zeros((10, 10), dtype=int)               # rows = true digit, columns = predicted digit
for t, p in zip(y[te], pred):
    cm[t, p] += 1
print("test accuracy: %.3f" % np.mean(pred == y[te]))
print("confusion matrix (rows = true, cols = predicted):")
print(cm)

im2 = ax2.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):                          # write each count in its cell
        ax2.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
ax2.set_xlabel("predicted digit")
ax2.set_ylabel("true digit")
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
fig.colorbar(im2, ax=ax2, label="count")
off = cm - np.diag(np.diag(cm))                  # zero the diagonal to find the biggest mistake
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"most common mistake: true {i} predicted as {j} ({off[i, j]} times)")
fig.tight_layout()`;

const IMSHOW_OUT = String.raw`image shape: (8, 8) | pixel range: 0.0 to 15.0 | label: 0
test accuracy: 0.978
confusion matrix (rows = true, cols = predicted):
[[60  0  0  0  1  0  0  0  0  0]
 [ 0 61  0  0  0  0  0  0  1  0]
 [ 0  1 67  0  0  0  0  0  0  0]
 [ 0  0  1 51  0  1  0  0  0  0]
 [ 0  0  0  0 63  0  0  1  0  1]
 [ 0  0  0  0  0 63  0  0  0  0]
 [ 0  1  0  0  0  0 60  0  1  0]
 [ 0  0  0  0  0  0  0 49  0  0]
 [ 0  1  0  0  0  0  0  0 53  0]
 [ 0  0  0  0  0  2  0  0  1 57]]
most common mistake: true 9 predicted as 5 (2 times)`;

const BOUNDARY = String.raw`import os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)   # two interleaved half-moons
models = {
    "logistic regression": LogisticRegression(),
    "MLP (2 x 16 hidden units)": MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=3000, random_state=0),
}

# ---------- the grid of points we will colour ----------
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 150))
grid = np.column_stack([xx.ravel(), yy.ravel()])      # (30000, 2): every grid point as a row
print("xx shape:", xx.shape, "| grid shape:", grid.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, model) in zip(axes, models.items()):
    model.fit(X, y)
    prob = model.predict_proba(grid)[:, 1].reshape(xx.shape)   # P(class 1) at each grid point
    cs = ax.contourf(xx, yy, prob, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, prob, levels=[0.5], colors="black")       # the decision boundary itself
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=15)
    ax.set_title(f"{name}\ntrain acc {model.score(X, y):.3f}")
    ax.set_xlabel("feature 1")
    print(f"{name:>26}: train acc {model.score(X, y):.3f}, grid share predicted class 1: {np.mean(prob > 0.5):.3f}")
axes[0].set_ylabel("feature 2")
fig.colorbar(cs, ax=axes, label="P(class 1)")

# ---------- saving: pixel size = inches x dpi ----------
folder = tempfile.mkdtemp()
for dpi in [72, 150]:
    path = os.path.join(folder, f"boundary_{dpi}.png")
    fig.savefig(path, dpi=dpi)
    h, w = plt.imread(path).shape[:2]
    print(f"dpi {dpi:3d}: {w} x {h} pixels")
fig.savefig(os.path.join(folder, "boundary.pdf"))           # vector format: sharp at any zoom
print("files written:", sorted(os.listdir(folder)))`;

const BOUNDARY_OUT = String.raw`xx shape: (150, 200) | grid shape: (30000, 2)
       logistic regression: train acc 0.833, grid share predicted class 1: 0.518
 MLP (2 x 16 hidden units): train acc 0.963, grid share predicted class 1: 0.493
dpi  72: 720 x 288 pixels
dpi 150: 1500 x 600 pixels
files written: ['boundary.pdf', 'boundary_150.png', 'boundary_72.png']`;

const TORCH_PLOT = String.raw`import os, tempfile
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

torch.manual_seed(0)
torch.set_num_threads(1)
X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
Xtr, ytr, Xva, yva = X[:300], y[:300], X[300:], y[300:]

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

history = {"train_loss": [], "val_loss": [], "val_acc": []}   # one entry per epoch
for epoch in range(200):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr).squeeze(1), ytr)               # full-batch training
    loss.backward()
    opt.step()
    model.eval()
    with torch.no_grad():                                     # record metrics, no gradients
        logits = model(Xva).squeeze(1)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(loss_fn(logits, yva).item())
        history["val_acc"].append(((logits > 0).float() == yva).float().mean().item())

for e in [0, 49, 199]:
    print(f"epoch {e+1:3d}: train {history['train_loss'][e]:.4f}  val {history['val_loss'][e]:.4f}  val acc {history['val_acc'][e]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
epochs = np.arange(1, 201)
axes[0].plot(epochs, history["train_loss"], label="train")
axes[0].plot(epochs, history["val_loss"], label="validation")
axes[0].set_yscale("log"); axes[0].set_title("loss (log scale)"); axes[0].legend()
axes[1].plot(epochs, history["val_acc"]); axes[1].set_title("validation accuracy")
for ax in axes[:2]:
    ax.set_xlabel("epoch")

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 150))
grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
with torch.no_grad():
    prob = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)   # tensor -> NumPy for Matplotlib
axes[2].contourf(xx, yy, prob, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.7)
axes[2].scatter(Xva[:, 0], Xva[:, 1], c=yva, cmap="RdBu_r", edgecolor="k", s=15)
axes[2].set_title("decision regions (validation points)")
fig.tight_layout()
path = os.path.join(tempfile.mkdtemp(), "torch_run.png")
fig.savefig(path, dpi=80)
print("grid prob range: %.3f to %.3f | image shape: %s" % (prob.min(), prob.max(), plt.imread(path).shape))`;

const TORCH_PLOT_OUT = String.raw`epoch   1: train 0.6972  val 0.6902  val acc 0.500
epoch  50: train 0.2774  val 0.1992  val acc 0.910
epoch 200: train 0.1567  val 0.1593  val acc 0.960
grid prob range: 0.000 to 1.000 | image shape: (320, 1120, 4)`;

const KERAS_PLOT = String.raw`import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# fit() returns a History object; history.history is a dict of per-epoch lists
history = model.fit(X[:300], y[:300], validation_data=(X[300:], y[300:]),
                    epochs=100, verbose=0)
print(sorted(history.history.keys()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="train")
ax.plot(history.history["val_loss"], label="validation")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
fig.savefig("keras_curves.png", dpi=100)`;

const EX_SOLUTION = String.raw`import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(8, 4))   # a 2 x 5 grid = 10 Axes
for d, ax in enumerate(axes.flat):
    first = np.flatnonzero(digits.target == d)[0]  # index of the first image of digit d
    ax.imshow(digits.images[first], cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(f"{d} (row {first})")
    ax.axis("off")                                 # hide ticks: they mean nothing here
print("axes array shape:", axes.shape)
print("first index of each digit:", [int(np.flatnonzero(digits.target == d)[0]) for d in range(10)])
means = [digits.images[digits.target == d].mean() for d in range(10)]
print("mean ink per digit:", np.round(means, 2))
print("most ink:", int(np.argmax(means)), "| least ink:", int(np.argmin(means)))`;

const EX_SOLUTION_OUT = String.raw`axes array shape: (2, 5)
first index of each digit: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mean ink per digit: [4.95 4.89 4.91 4.79 4.85 4.8  4.86 4.74 5.16 4.9 ]
most ink: 8 | least ink: 7`;

const lesson: Lesson = {
  id: "m00/plotting-data",
  title: "Visualizing Data with Matplotlib",
  summary:
    "Why you must look at data before modelling it (Anscombe's quartet), how Matplotlib's Figure and Axes work, and how to draw the plots deep learning uses every day: functions, training curves, scatter plots, histograms, images, confusion matrices and decision boundaries.",
  minutes: 50,
  level: "beginner",
  prerequisites: ["m00/numpy-arrays", "m00/vectorization"],
  next: ["m01/dl-workflow", "m03/training-basics"],
  terms: ["figure", "axes", "histogram", "colormap", "confusion-matrix", "decision-boundary", "learning-curve", "anscombes-quartet"],
  sections: {
    // ------------------------------------------------------------------
    intro: [
      { t: "p", text: "**Data visualization** means turning numbers into pictures so that a human can see patterns, mistakes and surprises at a glance. **Matplotlib** is the standard Python library for this: almost every other Python plotting tool is built on it, and nearly every deep learning tutorial, paper figure and debugging session uses it." },
      { t: "p", text: "Why does a deep learning course need a plotting lesson? Because a neural network is trained by looking at numbers you cannot read one by one: thousands of images, millions of losses. A few well-chosen plots answer the questions that matter: *Is my data what I think it is? Is training working? Is the model overfitting? Which classes does it confuse?* A single line of printed accuracy hides all of that." },
      { t: "p", text: "The classic demonstration is **Anscombe's quartet**, published by the statistician Francis Anscombe in 1973: four tiny datasets that have the same average, the same spread, the same correlation and the same best-fit line, yet look completely different when plotted. One is a straight-ish cloud, one is a curve, one is a perfect line with one outlier, and one is a vertical stack with one far-away point. Statistics alone would call them identical. You will compute those statistics yourself below." },
      { t: "callout", kind: "key", title: "What you will be able to do", text: "Create a Figure with several Axes; draw line plots of functions and **training curves** (train vs validation, with a log-scale axis); scatter plots coloured by class; histograms with chosen bins or as a density; images and matrices with `imshow` (colormaps, `vmin`/`vmax`, colour bars) — including an 8×8 digit and a confusion matrix; and a model's **decision boundary** using `meshgrid` + `contourf`. Then save any of it to a file with `savefig`." },
    ],

    // ------------------------------------------------------------------
    prerequisites: [
      { t: "list", items: [
        "[NumPy Arrays, Shapes and Indexing](/learn/m00/numpy-arrays) — every plotting function takes NumPy arrays. You need `shape`, slicing like `X[:, 3]` (all rows, column 3), and boolean masks like `X[y == 0]`.",
        "[Vectorization and Broadcasting](/learn/m00/vectorization) — we compute whole curves at once, e.g. `np.tanh(z)` on 200 points, and turn a grid into a list of points with `ravel` and back with `reshape`.",
        "Basic Python: lists, dictionaries, `for` loops and f-strings such as `f\"{x:.3f}\"` (print `x` with 3 decimals).",
        "No machine learning knowledge is required. When a plot shows a model (logistic regression, a small network) we only use it as a source of numbers to draw; the models themselves are taught in [Linear and Logistic Regression](/learn/m03/linear-logistic-regression).",
      ] },
    ],

    // ------------------------------------------------------------------
    intuition: [
      { t: "callout", kind: "analogy", title: "A doctor reading a chart, not a spreadsheet", text: "A doctor monitoring a patient does not read a column of 2,000 heart-rate numbers; she looks at the line on the monitor and instantly sees a spike, a flat line or a slow drift. Plots do the same for your data and your training run. The numbers are all there in the table, but the eye only finds the problem in the picture." },
      { t: "p", text: "Think of Matplotlib as a **picture frame with drawing panels**. The frame is the **Figure** — the whole image that will be shown or saved. Inside the frame you place one or more **Axes** — a single panel with its own x-axis, y-axis, title and data. (The name is confusing: *Axes* means one whole plotting panel, not the plural of axis.) You draw on an Axes by calling its methods: `ax.plot` for lines, `ax.scatter` for dots, `ax.hist` for histograms, `ax.imshow` for images." },
      { t: "p", text: "Each kind of plot answers a different question. A **line plot** shows how one quantity changes with another (loss over training steps). A **scatter plot** shows how two measurements relate and whether classes separate. A **histogram** shows how one measurement is distributed — where most values are and whether there are outliers. An **image plot** shows a grid of numbers as colours, which works both for photos and for tables like a confusion matrix. A **decision boundary plot** shows which answer a model would give at every point of the input space." },
    ],

    // ------------------------------------------------------------------
    mechanism: [
      { t: "h", text: "The object model: Figure → Axes → artists" },
      { t: "steps", items: [
        { title: "Choose a backend", text: "A **backend** is the part of Matplotlib that actually produces pixels. `matplotlib.use(\"Agg\")` selects a backend that draws into memory and files only — no window — which is what you want in scripts, servers and automated checks. It must be called before `pyplot` draws anything." },
        { title: "Create the Figure and its Axes", text: "`fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))` creates a Figure of the given size in **inches** and a grid of Axes. With one row and two columns you can unpack directly: `fig, (ax1, ax2) = plt.subplots(1, 2)`. With a 2×2 grid, `axes` is a 2×2 NumPy array of Axes and `axes.flat` iterates over all four." },
        { title: "Draw on an Axes", text: "Each call such as `ax.plot(x, y)` creates an **artist** — an object that knows how to draw itself (a line, a set of dots, a text label) — and adds it to that Axes. Calling `ax.plot` three times puts three lines on the same panel." },
        { title: "Label it", text: "`ax.set_xlabel`, `ax.set_ylabel` and `ax.set_title` describe the panel; passing `label=\"...\"` to each drawing call and then calling `ax.legend()` adds a key explaining each line or colour. `fig.suptitle` puts one title over the whole Figure." },
        { title: "Adjust scales and layout", text: "`ax.set_yscale(\"log\")` makes the y-axis logarithmic; `ax.set_ylim(a, b)` fixes its range; `fig.tight_layout()` moves panels so labels do not overlap." },
        { title: "Show or save", text: "`fig.savefig(\"name.png\", dpi=150)` writes the image file. In a Jupyter notebook the figure appears below the cell; in this website's in-browser **Run** button (which runs Python in your browser with Pyodide) every figure still open at the end of the program is captured and displayed inline under the output." },
      ] },
      { t: "callout", kind: "note", title: "Two styles: pyplot vs object-oriented", text: "You will see both `plt.plot(x, y)` and `ax.plot(x, y)`. The `plt.` style draws on whichever Axes is \"current\", which is convenient for one quick plot but confusing with several panels. This lesson uses the object-oriented style (`fig, ax = plt.subplots()` then `ax.something`) everywhere, because you always know which panel you are drawing on." },
      { t: "h", text: "How each plot type turns arrays into pictures" },
      { t: "table", head: ["Call", "Input arrays", "What is drawn"], rows: [
        ["`ax.plot(x, y)`", "two 1-D arrays of equal length $n$", "$n$ points joined by straight segments — with 200 points a smooth-looking curve"],
        ["`ax.scatter(x, y, c=...)`", "two 1-D arrays; optional colour per point", "one marker per point, no connecting lines"],
        ["`ax.hist(v, bins=k)`", "one 1-D array", "the range of $v$ is cut into $k$ equal-width bins; each bar's height is how many values fall in it"],
        ["`ax.imshow(M, cmap=..., vmin=..., vmax=...)`", "a 2-D array (or $H\\times W\\times 3$ for colour)", "one coloured square per entry; row 0 at the top"],
        ["`ax.contourf(xx, yy, Z)`", "three 2-D arrays of the same shape", "regions filled with colour according to the value of $Z$ on the grid"],
      ] },
      { t: "h", text: "How a decision-boundary plot is built" },
      { t: "steps", items: [
        { title: "Make a grid", text: "`np.meshgrid(xs, ys)` takes 200 x-positions and 150 y-positions and returns two 150×200 arrays `xx`, `yy`: together they list the coordinates of all 30,000 grid points." },
        { title: "Flatten to a list of points", text: "`np.column_stack([xx.ravel(), yy.ravel()])` turns them into a 30000×2 array — the same shape a model expects as input (one row per example, one column per feature)." },
        { title: "Ask the model", text: "`model.predict_proba(grid)[:, 1]` returns 30,000 probabilities of class 1." },
        { title: "Reshape and colour", text: "`.reshape(xx.shape)` puts the 30,000 numbers back into the 150×200 grid; `ax.contourf(xx, yy, prob)` colours each region, and `ax.contour(..., levels=[0.5])` draws the line where the probability crosses 0.5 — the **decision boundary**." },
        { title: "Overlay the data", text: "Finally `ax.scatter` draws the real training points on top, so you can see which ones fall on the wrong side." },
      ] },
    ],

    // ------------------------------------------------------------------
    math: [
      { t: "h", text: "The summary statistics that Anscombe's quartet fools" },
      { t: "p", text: "For $n$ points $(x_i, y_i)$, with $\\bar x$ and $\\bar y$ the averages:" },
      { t: "math", tex: String.raw`\bar x = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad s_x^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2, \qquad r = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i (x_i-\bar x)^2\,\sum_i (y_i-\bar y)^2}}`, caption: "Mean, sample variance (the average squared distance from the mean, with $n-1$ in the denominator; `ddof=1` in NumPy) and the Pearson correlation $r$, which runs from −1 (perfect falling line) to +1 (perfect rising line)." },
      { t: "math", tex: String.raw`\text{slope} = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2}, \qquad \text{intercept} = \bar y - \text{slope}\cdot\bar x`, caption: "The least-squares straight line $y \\approx \\text{slope}\\cdot x + \\text{intercept}$ — what `np.polyfit(x, y, deg=1)` returns." },
      { t: "h", text: "Worked example: dataset I of the quartet" },
      { t: "steps", items: [
        { title: "Means", text: "The 11 x-values are 4…14 so they sum to 99; the y-values sum to 82.51.", tex: String.raw`\bar x = 99/11 = 9.00, \qquad \bar y = 82.51/11 = 7.501` },
        { title: "Sums of squares and cross-products", text: "Computed in Python from the data above.", tex: String.raw`\sum (x_i-\bar x)^2 = 110.0, \quad \sum (y_i-\bar y)^2 = 41.2727, \quad \sum (x_i-\bar x)(y_i-\bar y) = 55.01` },
        { title: "Variances", tex: String.raw`s_x^2 = 110.0/10 = 11.00, \qquad s_y^2 = 41.2727/10 = 4.127` },
        { title: "Correlation", tex: String.raw`r = \frac{55.01}{\sqrt{110.0 \times 41.2727}} = \frac{55.01}{67.38} = 0.816` },
        { title: "Best-fit line", tex: String.raw`\text{slope} = 55.01/110.0 = 0.5001, \qquad \text{intercept} = 7.501 - 0.5001 \times 9 = 3.000` },
      ] },
      { t: "p", text: "Datasets II, III and IV give the same numbers to two or three decimals (see the program output in the next sections), yet only dataset I is a sensible linear relationship. The lesson: **a few summary numbers can be identical for very different data; always plot.**" },
      { t: "h", text: "Histograms: counts and density" },
      { t: "p", text: "With $N$ values cut into bins of width $w$, a count histogram's bar height is $c_k$ (number of values in bin $k$). A **density** histogram divides by $N w$ so that the total area of the bars is 1 — this lets you compare datasets of different sizes on the same axes, and compare a histogram with a probability density curve." },
      { t: "math", tex: String.raw`h_k = \frac{c_k}{N\,w}, \qquad \sum_k h_k\, w = \sum_k \frac{c_k}{N} = 1` },
      { t: "p", text: "Worked example: the 7 values $(1, 2, 2, 3, 3, 3, 4)$ with 3 bins over $[1, 4]$ have edges $1, 2, 3, 4$ and width $w = 1$. NumPy counts $c = (1, 2, 4)$ — the last bin includes its right edge, so it holds 3, 3, 3 and 4. Densities: $h = (1/7, 2/7, 4/7) = (0.142857, 0.285714, 0.571429)$, and $0.142857 + 0.285714 + 0.571429 = 1$." },
      { t: "h", text: "Figure size in pixels" },
      { t: "math", tex: String.raw`\text{pixels} = \text{inches} \times \text{dpi}`, caption: "dpi = dots (pixels) per inch. An 8 × 6-inch figure saved at dpi 100 is 800 × 600 pixels; a 10 × 4-inch figure is 720 × 288 at dpi 72 and 1500 × 600 at dpi 150." },
      { t: "h", text: "Why training curves use a log scale" },
      { t: "p", text: "On a logarithmic axis equal distances mean equal **ratios**: 1 → 0.1 → 0.01 are evenly spaced. A loss that falls from 0.69 to 0.05 is a factor of 14; on a normal axis everything after the first few steps looks like a flat line at the bottom, while on a log axis you can still see whether it is improving and whether train and validation are separating. In our run the loss starts at $0.6931 = \\ln 2$, exactly what cross-entropy gives when the untrained model (all weights zero) predicts probability 0.5 for every example: $-\\ln 0.5 = \\ln 2$." },
      { t: "h", text: "Grid size for a decision boundary" },
      { t: "p", text: "A grid of $n_x$ by $n_y$ points requires $n_x n_y$ model predictions. With $200\\times150$ that is 30,000 — trivial for a small model, but halving the step size quadruples the cost, so keep grids around a few hundred points per side." },
    ],

    // ------------------------------------------------------------------
    visual: [
      { t: "p", text: "The two simulators below are themselves plots of exactly the kinds taught here, driven by live computations. Use them to practise **reading** plots before you write them." },
      { t: "sim", id: "fitting", caption: "Left: a scatter plot of training (filled) and validation (hollow) points with a fitted curve. Right: a line plot of training and validation error against polynomial degree on a log-scale axis." },
      { t: "list", items: [
        "Set the degree to 1, then slowly increase it. On the right-hand plot, find the degree where the **validation** line is lowest — the training line keeps falling, the validation line turns back up. This is the same picture you will draw from a real training history.",
        "Notice the y-axis is logarithmic: the errors span several powers of ten, which a linear axis could not show.",
        "Increase the noise σ and watch the scatter plot become more spread out. Re-draw a new random sample: the curves change, which is why one plot from one run should not be over-interpreted.",
      ] },
      { t: "sim", id: "logistic-regression", caption: "A decision-boundary plot: background colour is the predicted probability on a grid, the line is where it equals 0.5, plus a line plot of loss and accuracy history." },
      { t: "list", items: [
        "Press Play and watch the background colour (a `contourf`-style probability map) sharpen as training proceeds, while the loss curve underneath falls.",
        "Click to add a few points of one class deep inside the other class's region and watch both the boundary and the loss curve react.",
        "Compare with the program in the next section, which draws the same kind of plot with `meshgrid` + `contourf`, and also draws an MLP whose boundary can bend.",
      ] },
    ],

    // ------------------------------------------------------------------
    scratch: [
      { t: "callout", kind: "tip", title: "How to run these programs", text: "Each program uses `matplotlib.use(\"Agg\")`, so it never opens a window and works anywhere, and it saves figures into a fresh temporary folder (`tempfile.mkdtemp()`). Because this page cannot compare pictures automatically, every program also **prints numbers** about what it drew (shapes, statistics, pixel sizes) — compare them with the output shown. Press **Run** to execute in your browser: the figures appear inline below the printed output." },
      { t: "h", text: "1. Anscombe's quartet: identical statistics, different data" },
      { t: "code", lang: "python", title: "anscombe.py", code: ANSCOMBE, output: ANSCOMBE_OUT },
      { t: "p", text: "`np.polyfit(x, y, deg=1)` fits a straight line by least squares; `np.corrcoef` returns a 2×2 matrix whose off-diagonal entry is $r$. `sharex=True, sharey=True` gives all four panels identical axis ranges so they can be compared fairly — without it each panel would zoom to its own data. `plt.imread` reads the saved PNG back: its shape (600, 800, 4) is height × width × 4 colour channels (red, green, blue, alpha/transparency)." },
      { t: "h", text: "2. Line plots: functions and a real training history" },
      { t: "code", lang: "python", title: "line_plots.py", code: LINES, output: LINES_OUT },
      { t: "list", items: [
        "`np.linspace(-6, 6, 200)` gives 200 evenly spaced inputs; 200 straight segments look like a smooth curve.",
        "The training loop is real: logistic regression by gradient descent on the breast-cancer dataset (bundled with scikit-learn), with 400 training rows and 169 validation rows. Each step appends one train and one validation loss to the `history` lists — exactly how you would record any network's training.",
        "Reading the curve: both losses fall fast for ~10 steps, then train keeps improving (0.0628 → 0.0494) while validation nearly stops (0.0835 → 0.0803). The widening gap between the two lines is the early warning sign of overfitting explained in [Training Basics](/learn/m03/training-basics).",
      ] },
      { t: "h", text: "3. Scatter plots and histograms" },
      { t: "code", lang: "python", title: "scatter_hist.py", code: SCATTER, output: SCATTER_OUT },
      { t: "p", text: "To colour by class, loop over the classes and draw each subset with its own colour and `label` — the legend then explains the colours. `alpha=0.6` makes points 60% opaque so overlapping points remain visible. The scatter shows malignant tumours tend to be larger (median area 932 vs 458). `ax.hist` returns the counts and bin edges it used, so you can check it: 20 bins of width 117.88 whose counts add up to all 569 rows. With `density=True` the bar areas sum to exactly 1." },
      { t: "h", text: "4. imshow: an image and a confusion matrix" },
      { t: "code", lang: "python", title: "imshow_digit_confusion.py", code: IMSHOW, output: IMSHOW_OUT },
      { t: "list", items: [
        "A **colormap** (`cmap`) maps numbers to colours. `gray_r` is reversed grey: 0 = white, high = black, like ink on paper. `Blues` goes from white to dark blue.",
        "`vmin=0, vmax=16` fixes which numbers map to the ends of the colormap. Without them Matplotlib stretches each image to its own min and max (here 0 and 15), so two images with different brightness would look the same. Fixing them makes images comparable.",
        "A **colour bar** (`fig.colorbar`) is the legend of an image plot: it shows which number each colour means.",
        "A **confusion matrix** counts, for each true class (row), how often the model predicted each class (column). Correct predictions are on the diagonal. Here 97.8% are correct and the most frequent confusion is a 9 read as a 5 (twice).",
      ] },
      { t: "h", text: "5. Subplot grid, decision boundaries and saving" },
      { t: "code", lang: "python", title: "decision_boundary.py", code: BOUNDARY, output: BOUNDARY_OUT },
      { t: "p", text: "The straight boundary of logistic regression cannot follow the two moons (83.3% training accuracy); the small neural network bends its boundary around them (96.3%). One `fig.colorbar(cs, ax=axes)` serves both panels. Saving the same Figure at dpi 72 and 150 gives 720 × 288 and 1500 × 600 pixels — the drawing is identical, only the resolution changes. A **PDF** (or SVG) is a *vector* file that stores shapes rather than pixels, so it stays sharp at any zoom — the usual choice for papers." },
    ],

    // ------------------------------------------------------------------
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Matplotlib draws NumPy arrays, so with PyTorch the only extra step is converting: `.item()` turns a one-number tensor into a Python float for the history lists, and `.numpy()` turns a tensor into an array (call it inside `torch.no_grad()` or after `.detach()`, since tensors that track gradients cannot be converted). PyTorch does not run in the browser, so run this one locally." },
          { t: "code", lang: "python", title: "plot_torch_training.py", code: TORCH_PLOT, output: TORCH_PLOT_OUT },
          { t: "p", text: "The three panels are the standard training dashboard: loss curves on a log axis, a metric curve, and (for 2-D toy data) the decision regions. Validation accuracy rises from 0.500 (guessing) to 0.960." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras records the history for you: `model.fit` returns a `History` object whose `.history` attribute is a dictionary of per-epoch lists, with `val_`-prefixed keys when you pass validation data." },
          { t: "code", lang: "python", title: "plot_keras_history.py (not executed in CI)", code: KERAS_PLOT, output: "Prints the history keys ['accuracy', 'loss', 'val_accuracy', 'val_loss'] and writes keras_curves.png with the train and validation loss curves on a log scale." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------
    variations: [
      { t: "table", head: ["Plot", "Best for", "Advantages", "Pitfalls"], rows: [
        ["Line (`plot`)", "a quantity over time or a smooth function: losses, learning-rate schedules, activations", "trends and comparisons of several series are obvious", "misleading if the x-values are unordered categories; noisy curves may need smoothing"],
        ["Scatter (`scatter`)", "relationship between two measurements; class separation; predicted vs true values", "shows every point, outliers and clusters", "overplotting with thousands of points — use small markers, `alpha`, or a 2-D histogram"],
        ["Histogram (`hist`)", "distribution of one variable: feature values, weights, activations, gradients", "shows skew, multiple peaks and outliers", "the picture depends on the number of bins; try several"],
        ["Image (`imshow`)", "images, weight matrices, attention maps, confusion matrices", "a whole 2-D array at once", "autoscaled colours hide differences between images — set `vmin`/`vmax`"],
        ["Filled contour (`contourf`)", "a function over a 2-D input: decision regions, loss landscapes", "shows the model's behaviour everywhere, not just at data points", "only possible for 2 input features (or a 2-D slice of more)"],
        ["Bar (`bar`)", "comparing a few numbers: accuracy per class, model comparison", "easy to read exact values", "do not start the y-axis above zero for bars — it exaggerates differences"],
        ["Box / violin (`boxplot`, `violinplot`)", "comparing distributions across groups", "compact; shows medians and spread", "hides how many points there are"],
      ] },
      { t: "p", text: "Related tools: **seaborn** adds statistical plots on top of Matplotlib, **pandas** has `df.plot()` built on it, and **TensorBoard** or experiment trackers draw live training curves in the browser during long runs. They all use the same plot types — knowing Matplotlib's Figure/Axes model lets you customize any of them." },
    ],

    // ------------------------------------------------------------------
    applications: [
      { t: "list", items: [
        "**Exploring a new dataset**: histograms of every feature (find impossible values, skew, units), scatter plots of feature pairs coloured by label, and a grid of example images with their labels — mislabelled images are often spotted this way.",
        "**Monitoring training**: train vs validation loss and accuracy per epoch, usually on a log scale; the learning rate over time ([Learning-Rate Schedules](/learn/m06/lr-schedules)).",
        "**Diagnosing a classifier**: confusion matrices show which classes are confused; per-class bar charts of accuracy show which classes need more data.",
        "**Debugging networks**: histograms of weights, activations and gradients reveal dead ReLUs or vanishing/exploding gradients ([Gradient Problems](/learn/m04/gradient-problems)).",
        "**Understanding models**: decision boundaries on toy data, filters of the first convolutional layer shown with `imshow`, attention maps overlaid on text or images.",
        "**Communicating results**: every figure in a paper, report or slide deck — saved as PDF/SVG for print or PNG at a chosen dpi for screens.",
      ] },
    ],

    // ------------------------------------------------------------------
    tradeoffs: [
      { t: "list", items: [
        "**Strength — universal**: works with any NumPy array, runs everywhere (scripts, notebooks, servers, the browser via Pyodide), and produces publication-quality files.",
        "**Strength — full control**: every element (tick, label, colour, margin) can be changed through the object-oriented API.",
        "**Limitation — verbose**: a nicely labelled multi-panel figure takes many lines; higher-level libraries do common plots in one call.",
        "**Limitation — static**: images do not support hovering or zooming; for interactive exploration of very large data, interactive libraries or dashboards are more convenient.",
        "**Limitation — slow for huge data**: drawing millions of individual points is slow and unreadable; plot a random sample or a 2-D histogram instead.",
        "**Trade-off — plots can mislead**: axis ranges, log vs linear scale, bin counts and colour limits all change the story. A plot is evidence only if those choices are honest and labelled.",
      ] },
    ],

    // ------------------------------------------------------------------
    mistakes: [
      { t: "list", items: [
        "**Trusting summary statistics without plotting.** Anscombe's quartet shows identical means, variances, correlation and best-fit line for four completely different datasets. Fix: plot before modelling.",
        "**Mixing `plt.` and `ax.` calls with several panels.** `plt.title` sets the title of whichever Axes was used last, often the wrong one. Fix: use `fig, axes = plt.subplots(...)` and call methods on the specific Axes.",
        "**Unlabelled axes and missing legends.** A curve without axis names and units cannot be interpreted by anyone else (or by you next month). Fix: always `set_xlabel`, `set_ylabel`, give each series a `label` and call `legend()`.",
        "**Linear axis for a loss curve.** Everything after the first epochs is squashed into a flat line. Fix: `ax.set_yscale(\"log\")`.",
        "**Plotting only the training loss.** It almost always goes down, so it cannot reveal overfitting. Fix: always plot validation alongside it.",
        "**Autoscaled colours with `imshow`.** Each image is stretched to its own range, so a faint image looks as dark as a bold one. Fix: pass the same `vmin`/`vmax` to every image you compare, and add a colour bar.",
        "**Displaying a normalized image as-is.** After normalization pixel values may be negative or above 1; `imshow` of an RGB array clips them and the image looks wrong. Fix: undo the normalization (multiply by std, add mean) before plotting.",
        "**Wrong shape for images.** PyTorch stores images as channels × height × width, but `imshow` expects height × width × channels. Fix: `img.permute(1, 2, 0).numpy()`.",
        "**Calling `savefig` after `plt.show()`.** In some environments `show` closes the figure, so the saved file is blank. Fix: save first, then show.",
        "**Creating hundreds of figures in a loop without closing them.** Memory grows and Matplotlib warns. Fix: `plt.close(fig)` after saving each one (but note that the in-browser runner only displays figures still open at the end).",
      ] },
    ],

    // ------------------------------------------------------------------
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between a Figure and an Axes in Matplotlib?", a: "The Figure is the whole image (the canvas, with a size in inches); an Axes is one plotting panel inside it, with its own x- and y-axis, title and data. One Figure can hold many Axes, e.g. a 2×2 grid created by `plt.subplots(2, 2)`." },
      { t: "qa", level: "beginner", q: "Why should you plot data before training a model on it?", a: "Summary statistics can hide very different structures — Anscombe's quartet has identical means, variances, correlation and regression line for four different datasets. Plots reveal outliers, non-linear relationships, wrong units, corrupted or mislabelled examples and class imbalance before they silently damage a model." },
      { t: "qa", level: "beginner", q: "What is a histogram and what does `density=True` change?", a: "A histogram splits the range of one variable into bins and shows how many values fall in each. With `density=True` each bar's height is divided by (number of values × bin width) so the total bar area is 1, which makes datasets of different sizes comparable." },
      { t: "qa", level: "intermediate", q: "Your training loss goes down steadily. Is training going well?", a: "Not necessarily. Training loss almost always decreases. You need the validation loss on the same plot: if it stops improving or starts rising while training loss keeps falling, the model is overfitting. A log-scale y-axis helps see small late changes. Also check the metric you care about (e.g. accuracy), not just loss." },
      { t: "qa", level: "intermediate", q: "How do you plot a classifier's decision boundary for 2-D inputs?", a: "Create a grid with `np.meshgrid` covering the data range, flatten it to an (n_points, 2) array, get the model's predicted probability for every grid point, reshape back to the grid shape, draw it with `contourf` and draw the 0.5 level with `contour`, then overlay the data points with `scatter`." },
      { t: "qa", level: "intermediate", q: "What do `vmin` and `vmax` do in `imshow`, and when must you set them?", a: "They set which data values map to the two ends of the colormap; values outside are clipped. By default each image is scaled to its own min and max. Set them whenever you compare several images or matrices, so the same colour means the same number in each panel." },
      { t: "qa", level: "advanced", q: "How would you visualize a 30-feature tabular dataset or a network's 512-dimensional embeddings?", a: "Start with per-feature histograms and a correlation matrix shown with `imshow`. For a 2-D picture of high-dimensional points, project them: PCA (a linear projection keeping the directions of largest variance) or non-linear methods such as t-SNE or UMAP, and scatter the 2-D result coloured by label. Remember that non-linear projections distort distances and cluster sizes, so they show neighbourhoods, not geometry." },
      { t: "qa", level: "advanced", q: "Validation loss is noisy from epoch to epoch, making the curve hard to read. What can you do without hiding real problems?", a: "Plot the raw curve faintly and a smoothed version (e.g. an exponential moving average) on top; evaluate on a larger validation set to reduce noise; plot several runs with different seeds as a band (mean ± standard deviation). Never only show the smoothed curve — smoothing delays and flattens real changes such as a sudden divergence." },
    ],

    // ------------------------------------------------------------------
    practice: [
      { t: "quiz", questions: [
        { q: "Which call creates a Figure containing a grid of 2 rows and 3 columns of Axes?", options: ["plt.figure(2, 3)", "plt.subplots(2, 3)", "plt.axes(2, 3)", "plt.plot(2, 3)"], answer: 1, explain: "`plt.subplots(nrows, ncols)` returns the Figure and a 2×3 array of Axes." },
        { kind: "numeric", q: "A figure of 6 × 4 inches is saved at dpi 200. How many pixels wide is the file?", answer: 1200, tolerance: 0, explain: "pixels = inches × dpi = 6 × 200 = 1200 (and 4 × 200 = 800 tall)." },
        { kind: "numeric", q: "A density histogram has bins of width 0.5. One bar has height 0.4. What fraction of all values falls in that bin?", answer: 0.2, tolerance: 0.001, explain: "Fraction = height × width = 0.4 × 0.5 = 0.2." },
        { q: "In Anscombe's quartet, which of these is the SAME for all four datasets (to 2–3 decimals)?", options: ["the shape of the scatter plot", "the correlation between x and y", "the number of outliers", "whether the relationship is linear"], answer: 1, explain: "All four have r ≈ 0.816 (and the same means, variances and regression line), while their shapes are completely different." },
        { q: "Why is the loss axis of a training curve usually logarithmic?", options: ["Matplotlib requires it", "Loss values are negative", "The loss shrinks by orders of magnitude, and a log axis keeps late improvements and train/validation gaps visible", "It makes the loss smaller"], answer: 2, explain: "Equal distances on a log axis mean equal ratios, so the whole range of the curve stays readable." },
        { kind: "numeric", q: "A meshgrid uses 300 x-values and 200 y-values. How many points must the model predict to draw the decision regions?", answer: 60000, tolerance: 0, explain: "Every combination: 300 × 200 = 60,000 points." },
        { q: "You compare two grayscale digit images with `imshow` without `vmin`/`vmax`. One has max ink 8, the other 16. What happens?", options: ["The first looks fainter", "Both look equally dark because each is scaled to its own max", "An error is raised", "The first is drawn smaller"], answer: 1, explain: "Autoscaling maps each image's own maximum to the darkest colour; set `vmin=0, vmax=16` for a fair comparison." },
      ] },
      { t: "exercise", title: "Density histogram by hand", prompt: "Ten values $(0.2, 0.4, 0.5, 1.1, 1.3, 1.4, 1.6, 1.8, 2.5, 2.9)$ are put in a histogram with 3 bins over $[0, 3]$. Give the counts and the density heights, and check that the bar areas sum to 1.", hint: "Bin width is $3/3 = 1$; the bins are $[0,1)$, $[1,2)$ and $[2,3]$.", solution: [
        { t: "steps", items: [
          { title: "Counts", text: "$[0,1)$: 0.2, 0.4, 0.5 → 3. $[1,2)$: 1.1, 1.3, 1.4, 1.6, 1.8 → 5. $[2,3]$: 2.5, 2.9 → 2. Total 10." },
          { title: "Densities", tex: String.raw`h_k = \frac{c_k}{N w} = \frac{(3,\,5,\,2)}{10 \times 1} = (0.3,\ 0.5,\ 0.2)` },
          { title: "Check", tex: String.raw`0.3 \times 1 + 0.5 \times 1 + 0.2 \times 1 = 1.0` },
        ] },
        { t: "p", text: "`np.histogram(v, bins=3, range=(0, 3), density=True)` returns exactly `[0.3, 0.5, 0.2]`." },
      ] },
      { t: "exercise", title: "A 2 × 5 grid of digits", prompt: "Using `load_digits`, draw a 2 × 5 grid showing the first image of each digit 0–9 with a common colour scale and its label as the title, no ticks. Also print the axes-array shape, the index of each digit's first image, and the average ink per digit class. Which digit uses the most ink?", hint: "`np.flatnonzero(digits.target == d)[0]` gives the first index of digit d; iterate over `axes.flat`; `ax.axis(\"off\")` hides the ticks.", solution: [
        { t: "code", lang: "python", title: "digit_grid.py", code: EX_SOLUTION, output: EX_SOLUTION_OUT },
        { t: "p", text: "The first ten images happen to be the digits 0–9 in order. Digit 8 uses the most ink on average (5.16) and 7 the least (4.74) — a sensible result, since an 8 has two closed loops and a 7 is just two strokes. The differences are small, which is exactly why a printed table is easier to misread than the picture grid." },
      ] },
    ],

    // ------------------------------------------------------------------
    summary: [
      { t: "list", items: [
        "Plot before you model: Anscombe's quartet has the same mean (9, 7.50), variance (11, ≈4.12–4.13), correlation (0.816) and line $y = 0.5x + 3$ for four very different datasets.",
        "Figure = the whole image; Axes = one panel. Use `fig, axes = plt.subplots(rows, cols, figsize=...)` and call methods on each Axes.",
        "Line plots for curves and training histories (train and validation together, `set_yscale(\"log\")`); scatter for two variables coloured by class; histograms for distributions (`bins`, `density=True`).",
        "`imshow` shows images and matrices; choose a `cmap`, fix `vmin`/`vmax` when comparing, and add a `colorbar`. A confusion matrix is a 2-D count table drawn this way.",
        "Decision boundaries: `meshgrid` → flatten → predict → reshape → `contourf` + `contour` at 0.5 → `scatter` the data.",
        "Always label axes, give series labels and call `legend()`. Save with `savefig(path, dpi=...)`; PNG for screens, PDF/SVG for print.",
        "Use `matplotlib.use(\"Agg\")` for scripts without a display; the in-browser Run button shows open figures inline.",
      ] },
      { t: "math", tex: String.raw`\text{pixels} = \text{inches}\times\text{dpi}, \qquad h_k = \frac{c_k}{N w}, \qquad r = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i (x_i-\bar x)^2 \sum_i (y_i-\bar y)^2}}` },
    ],

    // ------------------------------------------------------------------
    next: [
      { t: "list", items: [
        "[The Deep Learning Workflow](/learn/m01/dl-workflow) — where these plots are used: inspecting data, comparing baselines and diagnosing training with learning curves on a real end-to-end project.",
        "[Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) — how to interpret the train/validation curves you can now draw.",
        "[Linear and Logistic Regression](/learn/m03/linear-logistic-regression) — the model behind the straight decision boundary and the training curve in this lesson.",
        "[Classification Metrics](/learn/m17/classification-metrics) — confusion matrices in depth: precision, recall and why accuracy can mislead.",
      ] },
    ],
  },
};

export default lesson;
