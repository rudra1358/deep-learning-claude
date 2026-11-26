import type { Lesson } from "@/lib/types";

const WF_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_breast_cancer   # small built-in dataset, no download

# ---------- Stage 1: frame the problem ----------
# Input : 30 numbers measured from a microscope image of a breast-tissue sample
# Output: 1 = malignant (cancerous), 0 = benign
# Metric: accuracy, plus recall on malignant (share of cancers we catch)
data = load_breast_cancer()
X = data.data
y = 1 - data.target            # sklearn uses 0 = malignant; we flip so 1 = malignant

# ---------- Stage 2: inspect the data ----------
print("X shape:", X.shape, "| y shape:", y.shape)
print("class counts  benign:", int((y == 0).sum()), " malignant:", int((y == 1).sum()))
print("missing values:", int(np.isnan(X).sum()))
for j in [3, 4]:                # two features with very different scales
    print(f"{data.feature_names[j]:>16}: min {X[:, j].min():9.4f}  max {X[:, j].max():9.4f}")

# ---------- Stage 3: split ONCE, before looking at anything else ----------
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
n_tr, n_va = int(0.6 * len(X)), int(0.2 * len(X))
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
for name, part in [("train", tr), ("val", va), ("test", te)]:
    print(f"{name:>5}: {len(part):3d} rows, malignant share {y[part].mean():.3f}")

# ---------- Stage 4: preprocess with TRAIN statistics only ----------
mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
Xs = (X - mu) / sd            # apply the same train-derived numbers to every split

# ---------- Stage 5a: majority-class baseline ----------
majority = int(np.bincount(y[tr]).argmax())
print("majority class in train:", majority)
print(f"majority baseline  val accuracy {np.mean(y[va] == majority):.3f}")

# ---------- Stage 5b: logistic regression trained by gradient descent ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))   # clip avoids overflow warnings

def train_logreg(Xtr, ytr, lr=0.1, steps=500):
    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(steps):
        p = sigmoid(Xtr @ w + b)              # predicted probability of malignant
        grad = p - ytr                          # derivative of cross-entropy w.r.t. z
        w -= lr * Xtr.T @ grad / len(ytr)
        b -= lr * grad.mean()
    return w, b

def evaluate(w, b, Xp, yp):
    pred = (sigmoid(Xp @ w + b) >= 0.5).astype(int)
    acc = np.mean(pred == yp)
    recall = np.mean(pred[yp == 1] == 1)       # of the real cancers, how many we flag
    return acc, recall

w, b = train_logreg(Xs[tr], y[tr])
print("logistic regression  train acc %.3f | val acc %.3f, val recall %.3f"
      % (evaluate(w, b, Xs[tr], y[tr])[0], *evaluate(w, b, Xs[va], y[va])))

# ---------- what happens without scaling? ----------
w_raw, b_raw = train_logreg(X[tr], y[tr])
print("same model on UNSCALED inputs: val acc %.3f" % evaluate(w_raw, b_raw, X[va], y[va])[0])`;

const WF_NUMPY_OUT = String.raw`X shape: (569, 30) | y shape: (569,)
class counts  benign: 357  malignant: 212
missing values: 0
       mean area: min  143.5000  max 2501.0000
 mean smoothness: min    0.0526  max    0.1634
train: 341 rows, malignant share 0.370
  val: 113 rows, malignant share 0.336
 test: 115 rows, malignant share 0.417
majority class in train: 0
majority baseline  val accuracy 0.664
logistic regression  train acc 0.985 | val acc 0.973, val recall 0.947
same model on UNSCALED inputs: val acc 0.920`;

const LEAK = String.raw`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# ===== Leak 1: scaling with statistics computed on ALL rows (breast-cancer data) =====
data = load_breast_cancer()
X, y = data.data, 1 - data.target
idx = np.random.default_rng(0).permutation(len(X))
tr, va = idx[:341], idx[341:454]                  # same split as the main pipeline

def val_acc(Xs):
    model = LogisticRegression(max_iter=5000).fit(Xs[tr], y[tr])
    return model.score(Xs[va], y[va])

correct = (X - X[tr].mean(0)) / X[tr].std(0)     # train statistics only
leaky = (X - X.mean(0)) / X.std(0)               # statistics also use val/test rows
print(f"Leak 1  scaling: correct val acc {val_acc(correct):.3f} | leaky val acc {val_acc(leaky):.3f}")

# ===== Leak 2: choosing features using ALL labels, on pure noise =====
# 300 rows, 5000 random features, random 0/1 labels: the honest answer is ~50%.
rng = np.random.default_rng(2)
Xn = rng.normal(size=(300, 5000))
yn = rng.integers(0, 2, size=300)
tr_n, te_n = np.arange(100), np.arange(100, 300)

def top_features(Xa, ya, k=20):
    # |correlation| of every feature with the label; keep the k strongest
    Xc, yc = Xa - Xa.mean(0), ya - ya.mean()
    corr = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc))
    return np.argsort(corr)[-k:]

def test_acc(cols):
    model = LogisticRegression(max_iter=5000).fit(Xn[tr_n][:, cols], yn[tr_n])
    return model.score(Xn[te_n][:, cols], yn[te_n])

leaky_cols = top_features(Xn, yn)                 # MISTAKE: also looked at test labels
honest_cols = top_features(Xn[tr_n], yn[tr_n])    # correct: train rows only
print(f"Leak 2  feature selection on noise: leaky test acc {test_acc(leaky_cols):.3f} | honest test acc {test_acc(honest_cols):.3f}")

# ===== Leak 3: tuning on the test set =====
# Noise data again, so every model is truly a coin flip. We try 50 candidate models
# (each uses a random 3 of the 20 features) and keep the one with the best TEST score.
rng = np.random.default_rng(3)
Xq = rng.normal(size=(1100, 20))
yq = rng.integers(0, 2, size=1100)
tr_q, te_q, fresh_q = np.arange(100), np.arange(100, 150), np.arange(150, 1100)
best_score, best_model, best_cols = 0.0, None, None
for _ in range(50):
    cols = rng.choice(20, size=3, replace=False)
    m = LogisticRegression(max_iter=5000).fit(Xq[tr_q][:, cols], yq[tr_q])
    s = m.score(Xq[te_q][:, cols], yq[te_q])
    if s > best_score:
        best_score, best_model, best_cols = s, m, cols
print(f"Leak 3  tuning on test: best-of-50 test acc {best_score:.3f} (what a leaky report claims)")
print(f"        same model on 950 fresh rows:   {best_model.score(Xq[fresh_q][:, best_cols], yq[fresh_q]):.3f} (the truth)")`;

const LEAK_OUT = String.raw`Leak 1  scaling: correct val acc 0.973 | leaky val acc 0.973
Leak 2  feature selection on noise: leaky test acc 0.755 | honest test acc 0.510
Leak 3  tuning on test: best-of-50 test acc 0.580 (what a leaky report claims)
        same model on 950 fresh rows:   0.474 (the truth)`;

const WF_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0)
torch.set_num_threads(1)                         # makes CPU results repeatable

# ----- stages 1-4 exactly as in the NumPy version -----
data = load_breast_cancer()
X, y = data.data, 1 - data.target                # 1 = malignant
idx = np.random.default_rng(0).permutation(len(X))
tr, va, te = idx[:341], idx[341:454], idx[454:]
mu, sd = X[tr].mean(0), X[tr].std(0)
Xs = (X - mu) / sd
to_t = lambda a: torch.tensor(a, dtype=torch.float32)
Xtr, Xva, Xte = to_t(Xs[tr]), to_t(Xs[va]), to_t(Xs[te])
ytr, yva, yte = to_t(y[tr]), to_t(y[va]), to_t(y[te])

def make_mlp(hidden, depth):
    layers, n_in = [], 30
    for _ in range(depth):
        layers += [nn.Linear(n_in, hidden), nn.ReLU()]
        n_in = hidden
    layers.append(nn.Linear(n_in, 1))            # one logit: >0 means "malignant"
    return nn.Sequential(*layers)

def accuracy(model, Xp, yp):
    with torch.no_grad():
        return ((model(Xp).squeeze(1) > 0).float() == yp).float().mean().item()

def train(hidden, depth, weight_decay, epochs, log_at=()):
    torch.manual_seed(0)
    model = make_mlp(hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()             # sigmoid + cross-entropy in one stable op
    best = (float("inf"), 0, None)               # (val loss, epoch, weights)
    for epoch in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(Xtr)).split(32):   # mini-batches of 32 rows
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]).squeeze(1), ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(Xtr).squeeze(1), ytr).item()
            va_loss = loss_fn(model(Xva).squeeze(1), yva).item()
        if va_loss < best[0]:                    # early stopping: remember the best epoch
            best = (va_loss, epoch, {k: v.clone() for k, v in model.state_dict().items()})
        if epoch in log_at:
            print(f"  epoch {epoch:3d}  train loss {tr_loss:.4f}  val loss {va_loss:.4f}")
    model.load_state_dict(best[2])
    return model, best[1], best[0]

# ----- Stage 6-7: train a big network for a long time and read its learning curve -----
print("Big MLP (2 x 256 units, no regularisation), 150 epochs:")
big, best_epoch, best_loss = train(256, 2, 0.0, 150, log_at=(1, 5, 10, 25, 50, 100, 150))
print(f"  lowest val loss {best_loss:.4f} at epoch {best_epoch} -> later epochs overfit")

# ----- Stage 8: iterate - small grid, chosen ONLY by validation -----
print("Grid search (selected on validation):")
results = []
for hidden in [8, 64]:
    for wd in [0.0, 1e-2]:
        model, ep, vl = train(hidden, 1, wd, 100)
        acc = accuracy(model, Xva, yva)
        results.append((acc, -vl, hidden, wd, model))
        print(f"  hidden {hidden:3d}  weight_decay {wd:<5}  best epoch {ep:3d}  val loss {vl:.4f}  val acc {acc:.3f}")
acc, neg_vl, hidden, wd, mlp = max(results, key=lambda r: (r[0], r[1]))
print(f"chosen: hidden {hidden}, weight_decay {wd}")

# ----- Stage 9: final test evaluation, ONCE, for every candidate -----
logreg = LogisticRegression(max_iter=5000).fit(Xs[tr], y[tr])
majority = int(np.bincount(y[tr]).argmax())
n_params = sum(p.numel() for p in mlp.parameters())
print("FINAL TEST (115 rows, never used before):")
print(f"  majority baseline     acc {np.mean(y[te] == majority):.3f}")
print(f"  logistic regression   acc {logreg.score(Xs[te], y[te]):.3f}   (31 parameters)")
print(f"  MLP                   acc {accuracy(mlp, Xte, yte):.3f}   ({n_params} parameters)")`;

const WF_TORCH_OUT = String.raw`Big MLP (2 x 256 units, no regularisation), 150 epochs:
  epoch   1  train loss 0.2110  val loss 0.1765
  epoch   5  train loss 0.0495  val loss 0.0330
  epoch  10  train loss 0.0205  val loss 0.0285
  epoch  25  train loss 0.0017  val loss 0.0418
  epoch  50  train loss 0.0003  val loss 0.0704
  epoch 100  train loss 0.0001  val loss 0.1017
  epoch 150  train loss 0.0000  val loss 0.1225
  lowest val loss 0.0215 at epoch 11 -> later epochs overfit
Grid search (selected on validation):
  hidden   8  weight_decay 0.0    best epoch 100  val loss 0.0370  val acc 0.973
  hidden   8  weight_decay 0.01   best epoch 100  val loss 0.0458  val acc 0.982
  hidden  64  weight_decay 0.0    best epoch  99  val loss 0.0248  val acc 0.982
  hidden  64  weight_decay 0.01   best epoch  89  val loss 0.0407  val acc 0.991
chosen: hidden 64, weight_decay 0.01
FINAL TEST (115 rows, never used before):
  majority baseline     acc 0.583
  logistic regression   acc 0.974   (31 parameters)
  MLP                   acc 0.974   (2049 parameters)`;

const WF_KERAS = String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, 1 - data.target
idx = np.random.default_rng(0).permutation(len(X))
tr, va, te = idx[:341], idx[341:454], idx[454:]

# Normalization layer learns mean/std from the TRAIN rows only (adapt), then is part of the model
norm = tf.keras.layers.Normalization()
norm.adapt(X[tr])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(30,)),
    norm,
    tf.keras.layers.Dense(64, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]),
                    epochs=200, batch_size=32, callbacks=[stop], verbose=0)
print("epochs run:", len(history.history["loss"]))
print("test loss, test accuracy:", model.evaluate(X[te], y[te], verbose=0))   # once, at the end`;

const EX_SPLIT = String.raw`import numpy as np
from sklearn.datasets import load_breast_cancer

def stratified_split(y, fractions=(0.6, 0.2, 0.2), seed=0):
    """Split row indices so every part keeps the same class proportions."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == cls))   # shuffle this class's rows
        cut1 = int(round(fractions[0] * len(rows)))
        cut2 = cut1 + int(round(fractions[1] * len(rows)))
        for part, chunk in zip(parts, [rows[:cut1], rows[cut1:cut2], rows[cut2:]]):
            part.extend(chunk)
    return [rng.permutation(np.array(p)) for p in parts]

y = 1 - load_breast_cancer().target                       # 1 = malignant
tr, va, te = stratified_split(y)
print("sizes:", len(tr), len(va), len(te), "| total", len(tr) + len(va) + len(te))
for name, part in [("train", tr), ("val", va), ("test", te)]:
    print(f"{name:>5} malignant share {y[part].mean():.3f}")
print("overlap between parts:", len(set(tr) & set(va)) + len(set(tr) & set(te)) + len(set(va) & set(te)))`;

const EX_SPLIT_OUT = String.raw`sizes: 341 113 115 | total 569
train malignant share 0.372
  val malignant share 0.372
 test malignant share 0.374
overlap between parts: 0`;

const lesson: Lesson = {
  id: "m01/dl-workflow",
  title: "The Deep Learning Workflow",
  summary:
    "The ten stages of a real project — frame, inspect, split, preprocess, baseline, train, evaluate, iterate, test once, deploy and monitor — run end to end on a real medical dataset, with a deliberate data-leakage mistake and an honest look at when deep learning is not the right tool.",
  minutes: 60,
  level: "beginner",
  prerequisites: ["m01/ai-ml-dl", "m00/numpy-arrays", "m00/plotting-data"],
  next: ["m01/history", "m03/linear-logistic-regression", "m03/training-basics", "m17/data-splits"],
  terms: ["baseline", "data-leakage", "train-validation-test-split", "standardization", "hyperparameter", "early-stopping", "learning-curve", "data-drift", "recall"],
  sections: {
    // ------------------------------------------------------------------
    intro: [
      { t: "p", text: "A **workflow** is the ordered list of steps that takes you from a vague question (\"can a computer flag suspicious tumours?\") to a model people actually use and trust. Training the network — the part most tutorials focus on — is only one of ten steps, and rarely where projects fail. They fail because the question was badly framed, the data was misunderstood, the evaluation was accidentally cheating (**data leakage**), or a simple method would have done just as well." },
      { t: "p", text: "The central idea is older than deep learning: **judge a model only on data it has never seen, and change nothing after that judgement.** Every other step exists to keep that judgement honest and the model useful." },
      { t: "steps", items: [
        { title: "1. Frame the problem", text: "Define the input, the output, the metric that measures success, and the baseline you must beat." },
        { title: "2. Collect and inspect the data", text: "Count rows and classes; look for missing values, odd ranges and wrong labels, using plots ([Plotting Data](/learn/m00/plotting-data))." },
        { title: "3. Split into train / validation / test", text: "Do it once, early, and in a way that matches how the model will be used." },
        { title: "4. Preprocess", text: "Scale numbers and encode categories, using statistics from the training set only." },
        { title: "5. Baselines first", text: "Majority class, then a simple linear model. They tell you what \"good\" means." },
        { title: "6. Train a neural network", text: "Start small and check it can learn at all." },
        { title: "7. Evaluate on validation and diagnose", text: "Read the learning curves: is it underfitting, overfitting, or fine?" },
        { title: "8. Iterate", text: "Change hyperparameters, regularization, data — judged on validation only." },
        { title: "9. Final test, once", text: "Report the test score of the chosen model. Do not tune after this." },
        { title: "10. Deploy and monitor", text: "Serve predictions, watch for the world changing, retrain when needed." },
      ] },
      { t: "callout", kind: "key", title: "What makes this lesson different", text: "We run **one real example through all ten stages**: the breast-cancer dataset that comes with scikit-learn (569 tissue samples, 30 measurements each). We also make deliberate leakage mistakes and measure them, and compare the network honestly with logistic regression. (Spoiler: they tie.)" },
    ],

    // ------------------------------------------------------------------
    prerequisites: [
      { t: "list", items: [
        "[AI, Machine Learning and Deep Learning](/learn/m01/ai-ml-dl) — **supervised learning**: learning from examples whose correct answers (**labels**) are known.",
        "[NumPy Arrays](/learn/m00/numpy-arrays) — the code picks rows with an array of positions (`X[tr]`) and takes column means with `X.mean(axis=0)`.",
        "[Visualizing Data with Matplotlib](/learn/m00/plotting-data) — inspecting data and reading train/validation curves are plotting tasks.",
        "Helpful, not required: [Linear and Logistic Regression](/learn/m03/linear-logistic-regression) (our simple baseline) and [Training Basics](/learn/m03/training-basics) (epochs, over/underfitting, learning curves). Terms are explained when first used.",
      ] },
    ],

    // ------------------------------------------------------------------
    intuition: [
      { t: "callout", kind: "analogy", title: "Building a new medical test", text: "A hospital developing a new blood test decides what it must detect and how accurately (framing), and checks its samples' labels (inspection). It develops the test on one patient group (training), adjusts it on a second (validation), then runs a **blind trial** on a third group nobody looked at (test) — and publishes that number, even if it disappoints. A new test no better than the existing cheap one (baseline) is not worth it. After approval, it keeps watching real patients (monitoring)." },
      { t: "p", text: "Machine learning works the same way. The **training set** is what the model learns from. The **validation set** is what *you* learn from — you use it to choose between models and settings. The **test set** is the blind trial, used once at the very end. If test information leaks into any earlier decision, the final number measures how well you fitted the test set, not how the model will do on new patients." },
      { t: "p", text: "A **baseline** is the simplest reasonable answer. If 63% of tumours are benign, a \"model\" that always says *benign* is 63% accurate without learning anything. A real model must clearly beat that, and a neural network must clearly beat a simple linear model to be worth its extra cost." },
    ],

    // ------------------------------------------------------------------
    mechanism: [
      { t: "h", text: "Stage 1 — Frame the problem" },
      { t: "p", text: "Write down four things before any code:" },
      { t: "list", items: [
        "**Input**: 30 measurements (radius, texture, area, smoothness, …) from a microscope image of cells from a breast mass.",
        "**Output**: malignant (1) or benign (0) — a binary (two-class) classification.",
        "**Metric**: accuracy (the fraction of correct predictions), and also **recall** on malignant — of all real cancers, what fraction we flag. Missing a cancer is worse than a false alarm.",
        "**Baseline**: always predict the most common class.",
      ] },
      { t: "h", text: "Stage 2 — Collect and inspect" },
      { t: "p", text: "569 rows, 30 columns, 357 benign and 212 malignant (37%: **imbalanced** — one class is more common — but not extremely), no missing values. The features have wildly different scales: mean area runs from 143.5 to 2501, mean smoothness from 0.0526 to 0.1634. So scaling will matter (stage 4). In a real project, also plot every feature and ask an expert about surprises." },
      { t: "h", text: "Stage 3 — Split correctly" },
      { t: "p", text: "We shuffle the 569 row numbers with a fixed seed and cut them 60/20/20: 341 training, 113 validation and 115 test rows. We split **before** computing any statistic the model uses. Rules:" },
      { t: "list", items: [
        "**Random rows** only when rows are independent. If one patient has several images, split by **patient**. Otherwise the same person lands in train and test, and the model may recognise the person, not the disease.",
        "**By time** when predicting the future: train on January–June, validate on July, test on August. A random split would let the model \"see the future\".",
        "**Stratified** (same class proportions in each part) when classes are imbalanced or data is small. Our random split gave malignant shares of 0.370 / 0.336 / 0.417; the exercise below fixes this.",
        "**No duplicates across parts**: near-identical rows in train and test inflate the test score.",
      ] },
      { t: "h", text: "Stage 4 — Preprocess" },
      { t: "p", text: "**Standardization** rescales each feature to mean 0 and standard deviation 1 (the typical distance from the mean), using the training rows' numbers, then applies those same numbers to validation and test. Categories (e.g. a hospital name) would be **one-hot encoded**: one 0/1 column per category. Missing values would be filled in (imputed) from training-set statistics. The rule for all of it: **fit on train, apply to everything.**" },
      { t: "h", text: "Stages 5–6 — Baselines, then a neural network" },
      { t: "p", text: "Majority baseline: 0.664 validation accuracy. Logistic regression (one weighted sum + sigmoid — in effect a one-neuron network): 0.973. Only now do we train an **MLP** (multilayer perceptron — a network of fully connected layers) and check it at least matches the baseline." },
      { t: "h", text: "Stage 7 — Evaluate on validation and diagnose" },
      { t: "p", text: "A **learning curve** here plots training and validation loss (the error being minimised) after each **epoch** — one full pass over the training data. A deliberately oversized network (two hidden layers of 256 units) hits its lowest validation loss at epoch 11. After that, training loss keeps falling to 0.0000 while validation loss climbs from 0.0285 (epoch 10) to 0.1225 (epoch 150). That is textbook **overfitting**: the network is memorizing the 341 training rows. The simplest fix is **early stopping**: keep the weights from the epoch with the best validation loss." },
      { t: "h", text: "Stage 8 — Iterate" },
      { t: "p", text: "**Hyperparameters** are settings you choose instead of learn: hidden-layer size, learning rate, **weight decay** (a penalty that keeps weights small to reduce overfitting; see [Regularization](/learn/m07/regularization)). We try four combinations and pick by **validation** accuracy: 64 hidden units with weight decay 0.01 wins (0.991)." },
      { t: "h", text: "Stage 9 — Test once" },
      { t: "p", text: "On the 115 untouched test rows: majority 0.583, logistic regression 0.974, MLP 0.974. The chosen MLP scored higher on validation (0.991) than on test (0.974). Picking the best of four on validation makes its validation score slightly optimistic — exactly why a separate test set exists." },
      { t: "h", text: "Stage 10 — Deploy and monitor" },
      { t: "p", text: "Deployment means packaging the **whole pipeline** — saved scaler statistics plus the model — behind an interface (an API or an app feature) so new inputs get predictions ([Model Serving](/learn/m18/serving)). Monitoring means logging inputs, predictions and, when known, true outcomes. You watch for **data drift** (inputs changing, e.g. a new microscope with different brightness) and falling accuracy, then retrain on fresh data ([MLOps](/learn/m18/mlops))." },
      { t: "h", text: "Data leakage, measured" },
      { t: "p", text: "**Data leakage** is any way information from validation/test data (or from the future) sneaks into training or model choice. The code section measures three leaks:" },
      { t: "table", head: ["Mistake", "Reported score", "True score", "Why"], rows: [
        ["Scaling with mean/std of all 569 rows (breast cancer)", "0.973 val", "0.973 val", "Harmless **here**: all-row and train-only statistics barely differ (mean area 654.9 vs 662.3). Still a bug — with small or shifting data it matters."],
        ["Choosing the 20 \"best\" of 5000 noise features using all labels", "0.755 test", "0.510 test", "The selection step saw the test labels and picked features that match them by pure chance."],
        ["Trying 50 models, keeping the one with the best test score (noise data)", "0.580 test", "0.474 on 950 fresh rows", "With only 50 test rows, the luckiest of 50 coin-flippers looks better than chance."],
      ] },
    ],

    // ------------------------------------------------------------------
    math: [
      { t: "h", text: "Metrics" },
      { t: "math", tex: String.raw`\text{accuracy} = \frac{\#\{i : \hat y_i = y_i\}}{N}, \qquad \text{recall} = \frac{\#\{i : \hat y_i = 1,\ y_i = 1\}}{\#\{i : y_i = 1\}}`, caption: "$N$ = number of examples; $y_i$ = true label of example $i$; $\\hat y_i$ (\"y-hat\") = predicted label (1 = malignant); $\\#\\{\\dots\\}$ means \"count the examples where…\". Recall answers: of all real positives, how many did we catch?" },
      { t: "h", text: "The majority baseline" },
      { t: "p", text: "Let $\\pi$ (just a name here, not 3.14) be the share of positive rows. If we always predict the majority class (negative when $\\pi < 0.5$), accuracy is $1-\\pi$ and recall is 0." },
      { t: "steps", items: [
        { title: "Validation set", text: "113 rows, 38 malignant (share 0.336), 75 benign.", tex: String.raw`\text{acc}_\text{majority} = \frac{75}{113} = 0.664` },
        { title: "Test set", text: "115 rows, 48 malignant (share 0.417), 67 benign.", tex: String.raw`\text{acc}_\text{majority} = \frac{67}{115} = 0.583` },
        { title: "Logistic regression on validation", text: "0.973 accuracy and 0.947 recall correspond to these counts.", tex: String.raw`\frac{110}{113} = 0.973, \qquad \text{recall} = \frac{36}{38} = 0.947` },
      ] },
      { t: "h", text: "Standardization" },
      { t: "math", tex: String.raw`x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}, \qquad \mu_j = \frac{1}{N_\text{tr}}\sum_{i \in \text{train}} x_{ij}, \qquad \sigma_j = \sqrt{\frac{1}{N_\text{tr}}\sum_{i \in \text{train}} (x_{ij}-\mu_j)^2}`, caption: "$x_{ij}$ = feature $j$ of row $i$; $\\mu_j$ (mu) = that feature's mean; $\\sigma_j$ (sigma) = its standard deviation. Both come from the $N_\\text{tr}$ training rows only, then are applied to every row. In words: subtract the mean, divide by the spread." },
      { t: "p", text: "Worked example (feature *mean area*): the 341 training rows have $\\mu = 662.339$ and $\\sigma = 355.306$. The first training row has area 633.0, so" },
      { t: "math", tex: String.raw`x' = \frac{633.0 - 662.339}{355.306} = \frac{-29.339}{355.306} = -0.0826` },
      { t: "p", text: "Using all 569 rows instead gives $\\mu = 654.889$, $\\sigma = 351.605$ — close, which is why the scaling leak changed nothing measurable here. With no scaling at all, area values in the thousands dominate the weighted sum, and gradient descent with the same step size reaches only 0.920 validation accuracy instead of 0.973." },
      { t: "h", text: "The training objective" },
      { t: "p", text: "Both models output a probability $p_i$ that row $i$ is malignant. Training minimises **binary cross-entropy**, averaged over the training rows:" },
      { t: "math", tex: String.raw`\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \ln p_i + (1-y_i)\ln(1-p_i)\Big]`, caption: "$\\ln$ is the natural log. For each row only one term is active: $\\ln p_i$ if the label is 1, $\\ln(1-p_i)$ if it is 0. The loss is low when the model gives high probability to the correct class; see [Classification Losses](/learn/m05/classification-losses)." },
      { t: "h", text: "Why tuning on a small test set lies" },
      { t: "p", text: "Even a useless (coin-flip) model's accuracy on $n$ test rows wobbles around 0.5, with standard deviation" },
      { t: "math", tex: String.raw`\text{SD} = \sqrt{\frac{0.5 \times 0.5}{n}}, \qquad n = 50:\ \sqrt{0.005} = 0.0707` },
      { t: "p", text: "So any score from about 0.36 to 0.64 (±2 SD) fits pure guessing, and the best of many tries lands near the top. Our leak got 0.580 on 50 test rows; the same model scored 0.474 on 950 fresh rows. The same maths says that on our 115-row test set ($\\text{SD} \\approx \\sqrt{0.974 \\times 0.026/115} = 0.015$), each 0.974 score is uncertain by about ±3 percentage points." },
      { t: "h", text: "Parameter counts" },
      { t: "p", text: "Logistic regression: 30 weights + 1 bias = **31**. MLP with 64 hidden units: $30\\times64 + 64$ (hidden layer) $+\\ 64\\times1 + 1$ (output) $= 1920 + 64 + 64 + 1 =$ **2049** — 66 times more parameters for the same test accuracy." },
    ],

    // ------------------------------------------------------------------
    visual: [
      { t: "p", text: "Two visualizers cover the two halves of the workflow: the model, and how to judge it." },
      { t: "sim", id: "network", caption: "Stage 6: a small network trained live. Every value shown is computed." },
      { t: "list", items: [
        "Build a network with 2 inputs, one hidden layer and 1 output. Press Play and watch the loss: this is stage 6, the part most people call \"deep learning\".",
        "Add hidden neurons and layers. More neurons means more parameters — more power to fit, and to overfit. Our breast-cancer MLP has 2049 parameters for 341 training rows.",
        "This visualizer shows only training loss. In a real project that is never enough — you also need the validation curve, shown next.",
      ] },
      { t: "sim", id: "fitting", caption: "Stage 7: training vs validation error as model complexity grows." },
      { t: "list", items: [
        "Raise the polynomial degree step by step. Training error keeps falling; validation error falls, then rises. Picking the degree with the lowest validation error is stage 8 done right.",
        "Reduce the number of samples: the gap between training and validation error grows. Small data is when overfitting bites — and when simpler models (stage 5) tend to win.",
        "Imagine picking the degree using the test set instead: it would be tuned to that one test sample — the \"tuning on test\" leak from the table above.",
      ] },
    ],

    // ------------------------------------------------------------------
    scratch: [
      { t: "p", text: "Stages 1–5 in plain NumPy (scikit-learn only loads the dataset). Logistic regression is trained with the same gradient-descent loop as in [Linear and Logistic Regression](/learn/m03/linear-logistic-regression)." },
      { t: "code", lang: "python", title: "workflow_numpy.py", code: WF_NUMPY, output: WF_NUMPY_OUT },
      { t: "list", items: [
        "`y = 1 - data.target` flips the labels so 1 means malignant. Making the class you care about the positive one makes recall meaningful.",
        "`int(0.6 * 569) = 341` and `int(0.2 * 569) = 113`; the remaining 115 rows are the test set.",
        "`mu, sd` come from `X[tr]` only, and the same numbers rescale every split.",
        "`np.bincount(y[tr]).argmax()` finds the most common training label (0 = benign). Even the baseline is learned from training data only.",
        "Train accuracy 0.985 vs validation 0.973: a small gap, so logistic regression is not badly overfitting.",
        "The last line retrains on unscaled inputs: same model and learning rate, 0.920 instead of 0.973. Preprocessing matters.",
      ] },
      { t: "h", text: "Leakage lab: three mistakes, measured" },
      { t: "code", lang: "python", title: "leakage_lab.py", code: LEAK, output: LEAK_OUT },
      { t: "p", text: "Leak 1 gives a surprising result: here, scaling with all-row statistics gave the same validation accuracy. It is still wrong — in deployment you cannot use future data — and it can matter a lot on small or time-ordered data. Leaks 2 and 3 use random noise, where the true accuracy is 50%, so anything higher is pure leakage: 0.755 instead of 0.510, and 0.580 instead of 0.474." },
    ],

    // ------------------------------------------------------------------
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Stages 6–9 with a PyTorch MLP: an oversized network to show overfitting, early stopping, a small hyperparameter search judged on validation, and one final test of all three candidates. About 10 seconds on a CPU. (PyTorch cannot run in the browser; run it locally.)" },
          { t: "code", lang: "python", title: "workflow_pytorch.py", code: WF_TORCH, output: WF_TORCH_OUT },
          { t: "list", items: [
            "`BCEWithLogitsLoss` combines the sigmoid and binary cross-entropy in one numerically stable step. So the network outputs a raw score (a **logit**); a logit above 0 means probability above 0.5.",
            "`model.train()` / `model.eval()` switch layers like dropout between training and evaluation mode. Good practice even when, as here, there are no such layers.",
            "Early stopping: save a copy of the weights (`state_dict`) whenever validation loss improves, and restore the best copy at the end.",
            "The **honest conclusion**: on 569 rows of 30 well-chosen measurements, a 2049-parameter network ties a 31-parameter logistic regression (both 0.974 on test, i.e. 112 of 115 correct). For deployment you would pick logistic regression: smaller, faster, easier to explain.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same stages in Keras. A `Normalization` layer learns the mean and standard deviation from the training rows (`adapt`) and becomes part of the model, so the deployed model carries its own preprocessing. The `EarlyStopping` callback restores the weights with the best validation loss." },
          { t: "code", lang: "python", title: "workflow_keras.py (not executed in CI)", code: WF_KERAS, output: "Prints how many epochs ran before early stopping, then the test loss and test accuracy (expected to be close to the PyTorch result, about 0.95–0.98; exact values depend on the TensorFlow version and random initialization)." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------
    variations: [
      { t: "h", text: "Ways to split and validate" },
      { t: "table", head: ["Scheme", "How", "Advantages", "Disadvantages / use when"], rows: [
        ["Hold-out (train/val/test)", "one fixed split, e.g. 60/20/20", "simple, fast, the default for large data", "noisy with small data; one unlucky split misleads"],
        ["Stratified hold-out", "same class proportions in every part", "stable class balance", "classification with imbalance or few rows"],
        ["k-fold cross-validation", "split train+val into k parts; train k times, each part validates once; average", "every row is used for both training and validation; less noisy", "k times the compute — common for small data, rare for big networks"],
        ["Group split", "all rows of a patient/user/document go to the same part", "prevents identity leakage", "required whenever rows are grouped"],
        ["Time split", "train on past, validate/test on later periods", "mirrors deployment on future data", "required for forecasting, fraud, recommendations"],
      ] },
      { t: "h", text: "Baselines and models, from cheapest to most expensive" },
      { t: "table", head: ["Model", "Typical use", "Strengths", "Weaknesses"], rows: [
        ["Majority / mean predictor", "sanity-check floor", "zero effort; exposes imbalance", "learns nothing"],
        ["Linear / logistic regression", "tabular data, first real baseline", "fast, few parameters, interpretable weights", "only straight decision boundaries unless you engineer features"],
        ["Gradient-boosted decision trees", "tabular data (usually the strongest classical choice)", "handles mixed feature types and scales with little prep", "not suited to raw images, audio or text"],
        ["Small MLP", "tabular data when interactions matter; heads on top of embeddings", "flexible", "needs scaling and tuning; often no better on small tables"],
        ["CNN / Transformer / pretrained model", "images, audio, text, video", "learns features from raw data; transfer learning ([Transfer Learning](/learn/m12/transfer-learning))", "needs much more data and compute; harder to explain"],
      ] },
    ],

    // ------------------------------------------------------------------
    applications: [
      { t: "h", text: "Walkthrough: a medical-image triage system (generic)" },
      { t: "p", text: "Picture a system that looks at chest X-rays and moves likely-urgent cases to the top of a radiologist's list. It does **not** diagnose; a doctor still reads every image. Here are the ten stages for such a system (generic, not any specific product):" },
      { t: "steps", items: [
        { title: "Frame", text: "Input: one X-ray. Output: probability of an urgent finding. Metric: recall at a fixed false-alarm rate (missing an urgent case is the costly error), plus how much sooner urgent cases get read. Baseline: today's first-come-first-served order." },
        { title: "Collect and inspect", text: "Past images labelled from radiologist reports. Inspection finds different machines, rotated images, text burned into corners, and wrong labels from vague report wording." },
        { title: "Split", text: "By **patient** (never the same person in train and test), ideally also by **hospital or time**, so the test measures a new site or period." },
        { title: "Preprocess", text: "Resize, normalize pixel brightness, and remove burned-in text that might reveal the answer (e.g. a marker that only appears on urgent scans — a classic leak)." },
        { title: "Baseline", text: "Metadata rules, or a pretrained CNN fine-tuned in the simplest way." },
        { title: "Train", text: "Usually a pretrained image network adapted with [transfer learning](/learn/m12/transfer-learning), since labelled medical images are scarce." },
        { title: "Evaluate and iterate", text: "Learning curves, confusion matrices, and results per machine type and patient group to catch hidden failures." },
        { title: "Test once", text: "On a held-out site or period, often followed by a forward-looking study before real use." },
        { title: "Deploy and monitor", text: "Plug into the hospital's image system; log scores and later radiologist findings; alert when scores shift (new scanner, new population), then retrain or recalibrate." },
      ] },
      { t: "h", text: "Other systems that follow the same workflow" },
      { t: "list", items: [
        "**Spam filtering**: split by time because spammers adapt; constant monitoring and retraining because of drift.",
        "**Fraud detection**: fraud is rare, so the majority baseline tops 99% accuracy and accuracy is useless — recall and precision matter.",
        "**Demand forecasting**: time split is mandatory; the baseline is \"same as last week\".",
        "**Speech recognition, translation, image search**: raw, high-dimensional inputs and huge datasets — where deep learning clearly beats classical methods.",
      ] },
    ],

    // ------------------------------------------------------------------
    tradeoffs: [
      { t: "h", text: "Advantages of deep learning" },
      { t: "list", items: [
        "**Learns features from raw data**: pixels, audio and characters go straight in; classical methods need hand-designed features.",
        "**Keeps improving with more data and compute** on perception and language, where classical methods level off.",
        "**Transfer learning**: a network pretrained on lots of data can be adapted with few labels.",
        "**One toolbox for many data types**: the same workflow handles images, text, audio and mixes of them.",
      ] },
      { t: "h", text: "Limitations — and when NOT to use it" },
      { t: "list", items: [
        "**Small tables of data**: our own run shows it — 569 rows, 30 features: MLP 0.974, logistic regression 0.974, but with 66× more parameters, more code, more tuning and slower training. Here, linear models and gradient-boosted trees are the sensible first (and often last) choice.",
        "**Data hungry**: training from scratch usually needs thousands to millions of labelled examples, and labels can be expensive (medical experts).",
        "**Compute hungry**: GPUs for anything beyond small models; large models cost real time, money and energy.",
        "**Hard to explain**: an expert can read and check logistic regression's one weight per feature; a network's thousands of weights cannot be read directly ([Interpretability](/learn/m19/interpretability)).",
        "**Brittle**: performance can drop sharply when inputs change (new hardware, new population), so monitoring is a must.",
        "**Engineering cost**: more settings, more silent failure modes, more serving infrastructure.",
      ] },
      { t: "callout", kind: "tip", title: "Rule of thumb", text: "Raw images, audio, text or very large datasets → deep learning (usually from a pretrained model). A spreadsheet of a few thousand rows of meaningful measurements → start with logistic regression or gradient-boosted trees; use a network only if it clearly beats them on validation." },
    ],

    // ------------------------------------------------------------------
    mistakes: [
      { t: "list", items: [
        "**Skipping the baseline.** \"95% accuracy\" sounds great until you learn 95% of rows are one class. Fix: always report the majority baseline and a linear model next to your network.",
        "**Preprocessing before splitting.** Scaling, imputation, feature selection or oversampling done on all rows lets test information in. Our feature-selection leak turned a 0.510 model into a reported 0.755. Fix: split first; fit every preprocessing step on train only (scikit-learn `Pipeline` or Keras `Normalization` do this automatically).",
        "**Tuning on the test set.** Every decision based on the test score uses it up a little. Fix: tune on validation; touch test once.",
        "**Random split for grouped or time-ordered data.** The same patient, user or near-duplicate ends up in train and test, or you train on the future. Fix: group or time splits.",
        "**Wrong metric.** Accuracy on imbalanced data, or a metric that ignores the real cost of errors. Fix: choose the metric in stage 1 with the system's future users.",
        "**Reading only the training loss.** It nearly always falls. Fix: plot validation loss too; use early stopping.",
        "**Forgetting to ship the preprocessing.** The deployed model gets unscaled inputs and silently outputs nonsense. Fix: save the scaler statistics with the model, or build preprocessing into it.",
        "**Treating deployment as the end.** Data drifts and accuracy decays. Fix: monitor and plan retraining.",
        "**Over-reading small differences.** 0.974 vs 0.965 on 115 test rows is one example. Fix: estimate uncertainty (the SD formula above, or cross-validation) first.",
      ] },
    ],

    // ------------------------------------------------------------------
    interview: [
      { t: "qa", level: "beginner", q: "What are the training, validation and test sets for?", a: "The training set fits the model's parameters. The validation set is for decisions — choosing hyperparameters, comparing models, deciding when to stop. The test set is used once at the end to estimate performance on new data; since no decision was based on it, its score is unbiased." },
      { t: "qa", level: "beginner", q: "What is a baseline and why build one before a neural network?", a: "The simplest reasonable predictor — e.g. always predicting the majority class, then a linear model. It shows what score you get for free, exposes class imbalance, and tells you whether a complex model is worth its cost. Here: majority 0.583 on test, logistic regression 0.974 — same as the MLP." },
      { t: "qa", level: "beginner", q: "What is data leakage? Give an example.", a: "Information from outside the training data (validation/test rows or the future) influencing training or model choice, making evaluation too optimistic. Examples: scaling or selecting features on the full dataset before splitting, the same patient in train and test, a feature only recorded after the outcome, or tuning on the test set." },
      { t: "qa", level: "intermediate", q: "Your training loss keeps decreasing but validation loss started increasing at epoch 11. What is happening and what do you do?", a: "Overfitting: the model is memorizing training examples instead of learning patterns that generalize. Options: early stopping at the best validation epoch, regularization (weight decay, dropout), a smaller model, data augmentation, or more data — chosen using validation performance." },
      { t: "qa", level: "intermediate", q: "When would you NOT use deep learning?", a: "When the data is a small or medium table of meaningful features (logistic regression or gradient-boosted trees usually match or beat networks with far less effort), when labels are scarce and no suitable pretrained model exists, when predictions must be explained feature by feature, or when compute and latency budgets are tight. Deep learning shines on raw high-dimensional data (images, audio, text) and large datasets." },
      { t: "qa", level: "intermediate", q: "Why must preprocessing statistics come from the training set only?", a: "At deployment you only have past data, so the model must use statistics that do not depend on future inputs. Computing them on validation/test rows lets those rows influence the model, so evaluation no longer imitates new data. Fit the scaler on train, apply it everywhere, and save it with the model." },
      { t: "qa", level: "advanced", q: "Your model scores very well offline but poorly after deployment. List likely causes and how to investigate.", a: "Causes: leakage offline (features unavailable at prediction time, duplicates or same entities across splits, test-set tuning); training/serving skew (different preprocessing code, missing scaling, different units); distribution shift (new population, new hardware, seasons); different label definitions. Investigate: re-run the offline pipeline on logged production inputs, compare training vs production feature distributions, check which features dominate, and evaluate on a time- or site-based split." },
      { t: "qa", level: "advanced", q: "You tried 200 hyperparameter configurations on a 500-row validation set. How trustworthy is the best validation score, and what would you do?", a: "Too optimistic: with many configurations, the winner partly reflects noise in that particular validation set (a selection effect, like our best-of-50 test leak). Accuracy on 500 rows has standard deviation about $\\sqrt{p(1-p)/500}$, 1–2 points — similar to the gaps between configurations. Use cross-validation or a bigger validation set, prefer simpler configurations within noise of the best, and keep an untouched test set for the final number." },
    ],

    // ------------------------------------------------------------------
    practice: [
      { t: "quiz", questions: [
        { q: "In which order should these happen?", options: ["Scale all data → split → train", "Split → fit scaler on train → transform all splits → train", "Train → split → scale", "Split → fit scaler on test → train"], answer: 1, explain: "Split first; fit preprocessing on training rows only; then apply it to every split." },
        { kind: "numeric", q: "A dataset has 900 legitimate transactions and 100 fraudulent ones. What accuracy does the majority-class baseline get?", answer: 0.9, tolerance: 0.001, explain: "Always predicting \"legitimate\" is right 900/1000 = 0.9 of the time — while catching zero fraud (recall 0)." },
        { kind: "numeric", q: "Of 40 real malignant cases, a model flags 34. What is its recall?", answer: 0.85, tolerance: 0.001, explain: "Recall = 34 / 40 = 0.85." },
        { q: "You choose the learning rate that gives the best TEST accuracy, then report that test accuracy. What is wrong?", options: ["Nothing", "The test set was used for a decision, so its score is optimistically biased", "Learning rates cannot be tuned", "You should report training accuracy instead"], answer: 1, explain: "Model selection must use the validation set; the test set is only for the final, untouched estimate." },
        { q: "A dataset has 3 X-ray images per patient. How should you split it?", options: ["Randomly by image", "By patient, so each patient's images are all in one split", "Put all images in training", "By image file size"], answer: 1, explain: "Otherwise the same patient appears in train and test and the model can exploit patient identity — a group leakage." },
        { kind: "numeric", q: "A layer maps 30 inputs to 64 hidden units, followed by an output layer with 1 unit. How many parameters (weights + biases) in total?", answer: 2049, tolerance: 0, explain: "30×64 + 64 = 1984, plus 64×1 + 1 = 65: total 2049." },
        { q: "On a 600-row table of numeric measurements, an MLP and logistic regression both reach 0.97 test accuracy. Which should usually be deployed?", options: ["The MLP, because it is deep learning", "Logistic regression: same accuracy, far fewer parameters, faster and easier to explain", "Neither", "Whichever trained longer"], answer: 1, explain: "When performance ties, prefer the simpler, cheaper and more interpretable model." },
      ] },
      { t: "exercise", title: "Is the model better than the baseline?", prompt: "A validation set has 200 rows, 30 of them positive. Model A always predicts negative. Model B predicts 40 rows positive, of which 24 are truly positive. Compute the accuracy of A, and the accuracy and recall of B. Which model is more useful if missing positives is costly?", hint: "For B, count correct negatives too: true negatives = all negatives − false positives.", solution: [
        { t: "steps", items: [
          { title: "Model A", tex: String.raw`\text{acc}_A = \frac{170}{200} = 0.85, \qquad \text{recall}_A = \frac{0}{30} = 0` },
          { title: "Model B counts", text: "True positives 24; false positives 40 − 24 = 16; true negatives 170 − 16 = 154; false negatives 30 − 24 = 6." },
          { title: "Model B metrics", tex: String.raw`\text{acc}_B = \frac{24 + 154}{200} = \frac{178}{200} = 0.89, \qquad \text{recall}_B = \frac{24}{30} = 0.80` },
        ] },
        { t: "p", text: "B is only 4 points more accurate, but catches 80% of positives versus 0%. Accuracy alone hides most of the difference — which is why the metric is chosen in stage 1." },
      ] },
      { t: "exercise", title: "Write a stratified split", prompt: "Our random split gave malignant shares of 0.370 / 0.336 / 0.417. Write `stratified_split(y, fractions=(0.6, 0.2, 0.2), seed=0)` that splits each class separately so all three parts have nearly the same class proportions. Print the sizes, the malignant share of each part, and check that no row appears in two parts.", hint: "Shuffle the row indices of each class separately, cut each class 60/20/20, and concatenate the pieces for each part.", solution: [
        { t: "code", lang: "python", title: "stratified_split.py", code: EX_SPLIT, output: EX_SPLIT_OUT },
        { t: "p", text: "All three parts now have 37.2–37.4% malignant rows, matching the full dataset's 212/569 = 0.373. The final `rng.permutation` shuffles each part so classes are not in blocks (important for mini-batch training). scikit-learn's `train_test_split(..., stratify=y)` does the same for two-way splits." },
      ] },
    ],

    // ------------------------------------------------------------------
    summary: [
      { t: "list", items: [
        "Ten stages: frame → inspect → split → preprocess → baseline → train → evaluate/diagnose → iterate → test once → deploy and monitor.",
        "Framing fixes input, output, metric and baseline before any code. Pick a metric that reflects real costs (recall when misses are costly).",
        "Split before computing anything; split by group or time when rows are related; stratify when classes are imbalanced.",
        "Fit preprocessing (scaling, encoding, imputation, feature selection) on training data only, and ship it with the model.",
        "Baselines first: on breast cancer, majority 0.583 test, logistic regression 0.974, MLP 0.974 (31 vs 2049 parameters).",
        "Read learning curves: our oversized MLP's validation loss bottomed at epoch 11, then rose while training loss went to zero — overfitting, fixed by early stopping.",
        "Tune on validation; test once. Leaks measured here: 0.510 → 0.755 (feature selection on all rows), 0.474 → 0.580 (tuning on test).",
        "Deep learning wins on raw, high-dimensional data and large datasets; on small tables, simpler models are usually as good and far cheaper.",
      ] },
      { t: "math", tex: String.raw`x' = \frac{x - \mu_\text{train}}{\sigma_\text{train}}, \qquad \text{acc}_\text{majority} = 1 - \pi, \qquad \text{SD(acc)} \approx \sqrt{\frac{p(1-p)}{n}}` },
    ],

    // ------------------------------------------------------------------
    next: [
      { t: "list", items: [
        "[History of Neural Networks](/learn/m01/history) — why deep learning took off only once data and GPUs arrived.",
        "[Linear and Logistic Regression](/learn/m03/linear-logistic-regression) — the baseline model of this lesson, derived properly.",
        "[Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) — learning curves, underfitting and overfitting in depth.",
        "[Data Splits and Leakage](/learn/m17/data-splits) and [Classification Metrics](/learn/m17/classification-metrics) — stages 3 and 7 in full detail.",
        "[Model Serving](/learn/m18/serving) and [MLOps](/learn/m18/mlops) — stage 10: deployment, monitoring and retraining.",
      ] },
    ],
  },
};

export default lesson;
