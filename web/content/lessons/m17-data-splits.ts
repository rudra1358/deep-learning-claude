import type { Lesson } from "@/lib/types";

const SPLITS_NUMPY = String.raw`import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold

rng = np.random.default_rng(0)


# ---------------- splitters, from scratch ----------------
def kfold(n, k, rng):
    """Shuffle 0..n-1 once, cut into k nearly equal folds; yield (train_idx, val_idx)."""
    idx = rng.permutation(n)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1                       # first n % k folds get one extra example
    start = 0
    for s in sizes:
        val = idx[start:start + s]
        train = np.concatenate([idx[:start], idx[start + s:]])
        start += s
        yield train, val


def stratified_kfold(y, k, rng):
    """Deal each class's (shuffled) indices round-robin over the folds, so every fold keeps the class ratio."""
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == c))
        fold_of[members] = np.arange(len(members)) % k
    for f in range(k):
        yield np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)


def group_kfold(groups, k):
    """Whole groups go to one fold (largest groups first, each to the currently smallest fold)."""
    uniq, counts = np.unique(groups, return_counts=True)
    load = np.zeros(k, dtype=int)
    fold_of_group = {}
    for g, c in sorted(zip(uniq, counts), key=lambda gc: -gc[1]):
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += c
    fold_of = np.array([fold_of_group[g] for g in groups])
    for f in range(k):
        yield np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)


def time_series_split(n, k):
    """Expanding window: train on everything before the block, validate on the next block."""
    block = n // (k + 1)
    for f in range(k):
        end_train = n - (k - f) * block
        yield np.arange(end_train), np.arange(end_train, end_train + block)


# ---------------- 1) check the splitters ----------------
y = np.array([1] * 10 + [0] * 90)             # 10% positives
print("plain 5-fold, positives per validation fold:", [int(y[v].sum()) for _, v in kfold(100, 5, rng)])
print("stratified 5-fold, positives per fold:     ", [int(y[v].sum()) for _, v in stratified_kfold(y, 5, rng)])
print("sklearn StratifiedKFold, positives per fold:", [int(y[v].sum()) for _, v in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(100), y)])

ours = [v.tolist() for _, v in time_series_split(20, 4)]
theirs = [v.tolist() for _, v in TimeSeriesSplit(n_splits=4).split(np.zeros(20))]
print("time-series folds match sklearn:", ours == theirs, "| last fold trains on", len(list(time_series_split(20, 4))[-1][0]), "points")

groups = np.repeat(np.arange(8), [5, 5, 4, 4, 3, 3, 3, 3])   # 8 patients, 30 scans
ok = all(len(set(groups[tr]) & set(groups[va])) == 0 for tr, va in group_kfold(groups, 4))
sk = all(len(set(groups[tr]) & set(groups[va])) == 0 for tr, va in GroupKFold(4).split(np.zeros(30), groups=groups))
print("group k-fold: no patient in both train and validation:", ok, "(sklearn:", sk, ")")
print("every index validated exactly once:", sorted(np.concatenate([v for _, v in kfold(103, 5, rng)]).tolist()) == list(range(103)))


# ---------------- 2) leakage demo: selecting features on the full data ----------------
# 60 examples, 5000 features of PURE NOISE, random labels: no classifier can beat 50% on new data.
n, p, top = 60, 5000, 20
X = rng.normal(size=(n, p))
y = rng.permutation(np.repeat([0, 1], n // 2))


def top_features(X, y, m):
    """Indices of the m features with the largest |mean difference| between the classes."""
    diff = X[y == 1].mean(0) - X[y == 0].mean(0)
    return np.argsort(-np.abs(diff))[:m]


def nearest_centroid_acc(Xtr, ytr, Xva, yva):
    c0, c1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    pred = (((Xva - c1) ** 2).sum(1) < ((Xva - c0) ** 2).sum(1)).astype(int)
    return (pred == yva).mean()


folds = list(stratified_kfold(y, 5, np.random.default_rng(1)))

leak_cols = top_features(X, y, top)           # WRONG: the selection has seen every validation label
wrong = [nearest_centroid_acc(X[tr][:, leak_cols], y[tr], X[va][:, leak_cols], y[va]) for tr, va in folds]

right = []
for tr, va in folds:                          # RIGHT: select features inside each fold, on training rows only
    cols = top_features(X[tr], y[tr], top)
    right.append(nearest_centroid_acc(X[tr][:, cols], y[tr], X[va][:, cols], y[va]))

print(f"\nnoise data, CV accuracy with selection OUTSIDE the folds: {np.mean(wrong):.3f}  (folds {np.round(wrong, 2).tolist()})")
print(f"noise data, CV accuracy with selection INSIDE the folds:  {np.mean(right):.3f}  (folds {np.round(right, 2).tolist()})")`;

const SPLITS_NUMPY_OUT = String.raw`plain 5-fold, positives per validation fold: [3, 1, 4, 1, 1]
stratified 5-fold, positives per fold:      [2, 2, 2, 2, 2]
sklearn StratifiedKFold, positives per fold: [2, 2, 2, 2, 2]
time-series folds match sklearn: True | last fold trains on 16 points
group k-fold: no patient in both train and validation: True (sklearn: True )
every index validated exactly once: True

noise data, CV accuracy with selection OUTSIDE the folds: 0.917  (folds [1.0, 0.92, 0.92, 0.92, 0.83])
noise data, CV accuracy with selection INSIDE the folds:  0.483  (folds [0.42, 0.58, 0.58, 0.42, 0.42])`;

const SPLITS_TORCH = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer

torch.manual_seed(0)

# ---- data: 569 tumours, 30 features, label 1 = benign (bundled with scikit-learn, no download) ----
data = load_breast_cancer()
X = torch.tensor(data.data, dtype=torch.float32)
y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

# ---- one shuffled permutation -> 60% train / 20% validation / 20% test ----
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(X), generator=g)
n_tr, n_va = int(0.6 * len(X)), int(0.2 * len(X))
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
print(f"sizes: train {len(tr)}, val {len(va)}, test {len(te)}; positive rate train {y[tr].mean():.3f} val {y[va].mean():.3f} test {y[te].mean():.3f}")

# ---- standardization statistics come from the TRAINING rows only ----
mu, sd = X[tr].mean(0), X[tr].std(0)
Z = (X - mu) / sd                       # the same (train) mu/sd are applied to val and test

model = nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()


def accuracy(idx):
    with torch.no_grad():
        return ((model(Z[idx]) > 0).float() == y[idx]).float().mean().item()


best = (0.0, -1, None)                  # (val acc, epoch, weights): model selection uses VALIDATION only
for epoch in range(1, 201):
    opt.zero_grad()
    loss = loss_fn(model(Z[tr]), y[tr])
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        v = accuracy(va)
        if v > best[0]:
            best = (v, epoch, {k: t.clone() for k, t in model.state_dict().items()})
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  train loss {loss.item():.4f}  train acc {accuracy(tr):.3f}  val acc {v:.3f}")

model.load_state_dict(best[2])
print(f"selected epoch {best[1]} (val acc {best[0]:.3f})")
print(f"test accuracy, measured once at the very end: {accuracy(te):.3f}")`;

const SPLITS_TORCH_OUT = String.raw`sizes: train 341, val 113, test 115; positive rate train 0.622 val 0.664 test 0.609
epoch  50  train loss 0.0659  train acc 0.982  val acc 0.982
epoch 100  train loss 0.0386  train acc 0.988  val acc 0.973
epoch 150  train loss 0.0218  train acc 0.994  val acc 0.973
epoch 200  train loss 0.0122  train acc 0.994  val acc 0.973
selected epoch 20 (val acc 0.982)
test accuracy, measured once at the very end: 0.974`;

const SPLITS_SKLEARN = String.raw`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)

# ---- 1) leakage through feature selection, and the Pipeline fix (pure-noise data) ----
Xn = rng.normal(size=(60, 5000))
yn = rng.permutation(np.repeat([0, 1], 30))
cols = SelectKBest(f_classif, k=20).fit(Xn, yn).get_support()          # fitted on ALL rows: leak
leaky = cross_val_score(LogisticRegression(max_iter=1000), Xn[:, cols], yn, cv=cv5).mean()
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
honest = cross_val_score(pipe, Xn, yn, cv=cv5).mean()                   # re-fitted inside every fold
print(f"noise data  leaky CV acc {leaky:.3f}   pipeline CV acc {honest:.3f}")

# ---- 2) leakage through near-duplicates, and the GroupKFold fix ----
# 200 originals with 20% label noise; each is stored 3 times with tiny jitter (e.g. augmented copies).
Xo = rng.normal(size=(200, 2))
yo = (Xo[:, 0] + Xo[:, 1] > 0).astype(int)
flip = rng.random(200) < 0.2
yo[flip] = 1 - yo[flip]
Xd = np.repeat(Xo, 3, axis=0) + rng.normal(scale=0.01, size=(600, 2))
yd = np.repeat(yo, 3)
gid = np.repeat(np.arange(200), 3)                                      # which original each row came from
knn = KNeighborsClassifier(n_neighbors=1)
rand_cv = cross_val_score(knn, Xd, yd, cv=KFold(5, shuffle=True, random_state=0)).mean()
group_cv = cross_val_score(knn, Xd, yd, cv=GroupKFold(5), groups=gid).mean()
print(f"duplicates  random KFold acc {rand_cv:.3f}   GroupKFold acc {group_cv:.3f}")

# ---- 3) nested CV: tuning C and estimating performance without reusing the same folds ----
X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]}
inner = GridSearchCV(model, grid, cv=StratifiedKFold(5, shuffle=True, random_state=1))
inner.fit(X, y)
print(f"best C {inner.best_params_['logisticregression__C']}, best inner-CV score {inner.best_score_:.4f}  (optimistic: chosen on these folds)")
outer = cross_val_score(inner, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=2))
print(f"nested CV estimate {outer.mean():.4f} +/- {outer.std(ddof=1) / np.sqrt(5):.4f} (std. error over 5 outer folds)")`;

const SPLITS_SKLEARN_OUT = String.raw`noise data  leaky CV acc 0.933   pipeline CV acc 0.417
duplicates  random KFold acc 0.965   GroupKFold acc 0.702
best C 0.1, best inner-CV score 0.9789  (optimistic: chosen on these folds)
nested CV estimate 0.9754 +/- 0.0070 (std. error over 5 outer folds)`;

const SPLITS_KERAS = String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

tf.keras.utils.set_random_seed(0)
X, y = load_breast_cancer(return_X_y=True)
X = X.astype("float32")
perm = np.random.default_rng(0).permutation(len(X))
n_tr, n_va = int(0.6 * len(X)), int(0.2 * len(X))
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

norm = tf.keras.layers.Normalization()
norm.adapt(X[tr])                      # mean/variance learned from TRAINING rows only

model = tf.keras.Sequential([
    tf.keras.Input(shape=(30,)),
    norm,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"])
stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)
model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=200, batch_size=len(tr), callbacks=[stop], verbose=0)
print("test accuracy:", model.evaluate(X[te], y[te], verbose=0)[1])   # touched once, at the end`;

const lesson: Lesson = {
  id: "m17/data-splits",
  title: "Train/Validation/Test Splits and Cross-Validation",
  summary: "How to hold data out so that a reported score predicts performance on data the model has never seen: train/validation/test splits, k-fold, stratified, group and time-series cross-validation, nested CV — and the many ways information leaks from the evaluation data into training.",
  minutes: 60,
  level: "beginner",
  prerequisites: ["m03/training-basics", "m01/dl-workflow", "m02/probability"],
  next: ["m17/classification-metrics", "m17/regression-metrics", "m17/experimentation"],
  terms: ["train-validation-test-split", "cross-validation", "stratified-k-fold", "group-k-fold", "time-series-split", "nested-cross-validation", "data-leakage", "target-leakage"],
  sections: {
    intro: [
      { t: "p", text: "A model is useful only if it works on data it has **not** seen. The training loss cannot tell us that: a large enough network can memorize its training set and reach zero training error while being useless on new inputs (**overfitting**). To estimate performance on future data we must evaluate on examples that played **no part** in building the model. Deciding which examples those are, and keeping them clean, is what data splitting is about." },
      { t: "p", text: "The standard recipe uses three disjoint subsets. The **training set** fits the parameters (weights). The **validation set** (also called the development or *dev* set) is used to make decisions — choose hyperparameters, the architecture, the epoch to stop at. The **test set** is touched once, at the end, to report an unbiased estimate. When data is scarce, **cross-validation** (CV) reuses every example for both training and validation by rotating which part is held out." },
      { t: "p", text: "The rules sound simple, but in practice evaluation goes wrong constantly, almost always through **data leakage**: some information about the held-out data reaches the training process — through preprocessing fitted on all rows, duplicated examples, future information in time series, or features that secretly encode the label. Leaked scores look excellent and collapse in production. Kaufman, Rosset and Perlich (2012, *Leakage in Data Mining: Formulation, Detection, and Avoidance*) catalogue many real cases; Ambroise and McLachlan (2002, *PNAS*) showed that published gene-expression classifiers reported near-perfect error rates because genes had been selected using all samples before cross-validation." },
      { t: "callout", kind: "key", text: "A held-out score is an honest estimate of future performance **only if every decision that shaped the model — fitting, preprocessing, feature selection, hyperparameter choice — was made without looking at the held-out examples**, and the held-out examples are drawn the way future data will be." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Training basics](/learn/m03/training-basics) — what training, validation loss, overfitting and underfitting mean; here we make the *validation* part rigorous.",
        "[The deep learning workflow](/learn/m01/dl-workflow) — where evaluation sits in the project loop (data → model → train → evaluate → iterate).",
        "[Probability](/learn/m02/probability) — expectation, variance and the idea that a score measured on a finite sample is a random variable with a standard error.",
        "Basic NumPy indexing (`X[idx]` with an integer index array) from [NumPy arrays](/learn/m00/numpy-arrays) — every splitter below just produces index arrays.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Practice exams and the final exam", text: "A student preparing for an exam works through textbook problems (the **training set**). To decide whether they are ready, or which study method works better, they take practice exams (the **validation set**). If they take the same practice exam ten times they will eventually memorize it — their practice score stops meaning anything. The real **final exam** (the **test set**) is sealed and taken once. If a copy of the final leaked beforehand (**leakage**), a perfect grade would say nothing about what they actually learned." },
      { t: "p", text: "Every time you *look at* a score and *change something* because of it, that data becomes part of your training process in a weak but real sense. The validation set absorbs those decisions, which is why its score becomes slightly optimistic after many experiments, and why the test set must be used only at the end." },
      { t: "p", text: "**Cross-validation** is like a class where every student both writes and grades: split the data into $k$ parts, train on $k-1$ and evaluate on the remaining one, rotate $k$ times, and average. Every example is used for evaluation exactly once and for training $k-1$ times — no data is wasted, and the spread across folds shows how noisy the estimate is." },
      { t: "p", text: "The deepest idea is that the held-out data must **simulate the future**. If the model will see new patients, the held-out set must contain patients that never appeared in training (not just new scans of known patients). If it will predict tomorrow, it must be evaluated on days after its training data. A random split is correct only when future data really looks like a random draw from the same pool." },
    ],
    mechanism: [
      { t: "h", text: "The hold-out protocol" },
      { t: "steps", items: [
        { title: "Decide what “unseen” means", text: "New examples? New users/patients/documents (→ split by **group**)? Future time (→ split by **time**)? This choice determines the splitter, and is the most important decision." },
        { title: "Split once, before looking at anything", text: "Shuffle (unless splitting by time) with a fixed seed and set aside the test set. Typical proportions: 60/20/20 or 80/10/10 for small and medium data; with millions of examples a few thousand test examples (1% or less) already give a tight estimate. Do not explore or plot the test set." },
        { title: "Fit everything on the training set", text: "Model weights **and** every data-dependent preprocessing step: standardization means and variances, vocabularies, imputation values, PCA, feature selection, target encodings, oversampling. Apply the fitted transforms unchanged to validation and test." },
        { title: "Use the validation set for decisions", text: "Compare hyperparameters, architectures and checkpoints on it (early stopping, learning rate, regularization strength)." },
        { title: "Optionally retrain on train + validation", text: "Once settings are frozen, refitting on more data usually helps; keep the chosen number of epochs / hyperparameters fixed." },
        { title: "Evaluate on the test set once", text: "Report that number with its uncertainty. If you then change the model because of it, the test set has become a validation set and you need fresh test data for an unbiased number." },
      ] },
      { t: "h", text: "k-fold cross-validation" },
      { t: "steps", items: [
        { title: "Shuffle and partition", text: "Randomly permute the $n$ indices and cut them into $k$ folds of size $\\lfloor n/k\\rfloor$ or $\\lceil n/k\\rceil$ (typically $k = 5$ or $10$)." },
        { title: "Rotate", text: "For $i = 1..k$: train a **fresh** model on all folds except $i$ (including fitting any preprocessing), evaluate on fold $i$, store the score $s_i$." },
        { title: "Aggregate", text: "Report the mean $\\bar s$ and the spread (standard deviation, or standard error) of the $k$ scores." },
        { title: "Final model", text: "CV evaluates a *procedure*, not one model. Having chosen the procedure, train it once on all the data you are allowed to use." },
      ] },
      { t: "h", text: "Specialized splitters" },
      { t: "list", items: [
        "**Stratified k-fold** (classification): deal each class's indices round-robin over the folds so every fold has the same class proportions. Essential with rare classes — a plain fold could contain zero positives, making recall undefined.",
        "**Group k-fold**: when examples come in groups that are correlated (several images per patient, several sentences per document, several sessions per user), assign whole **groups** to folds so no group is in both training and validation.",
        "**Time-series split** (forward chaining / expanding window): fold $i$ trains on everything before a cut-off and validates on the block right after it. Never train on the future. A **gap** between train and validation is added when features use look-back windows or labels are only known after a delay.",
        "**Leave-one-out (LOO)**: $k = n$. Nearly unbiased but expensive and high-variance; mostly for tiny datasets or models with closed-form LOO shortcuts.",
        "**Repeated k-fold**: rerun k-fold with different shuffles and average, reducing the dependence on one random partition.",
        "**Nested CV**: an outer CV loop estimates performance; inside each outer training portion, an inner CV loop tunes hyperparameters. This is the correct way to report a score for a procedure that *includes* tuning.",
      ] },
      { t: "h", text: "The four classic leaks" },
      { t: "table", head: ["Leak", "How it happens", "Fix"], rows: [
        ["Preprocessing fitted on all data", "Scaler, imputer, PCA, vocabulary or feature selection computed before splitting; held-out statistics shape training", "Fit transforms inside each training fold (scikit-learn Pipeline, Keras Normalization.adapt on train)"],
        ["Duplicates / near-duplicates", "The same item (or an augmented copy, a re-upload, a sibling frame of a video) appears in train and test", "Deduplicate first (hashes, perceptual hashes, near-duplicate search); split by group"],
        ["Temporal leakage", "Random split of time-ordered data; features computed with future values (e.g. a rolling mean centred on the day)", "Split by time; compute features only from the past; add a gap"],
        ["Target leakage", "A feature is a consequence or proxy of the label (\"was given antibiotics\" for predicting infection; an ID that encodes the outcome)", "Ask \"would this value be known at prediction time?\" for every feature; drop or lag it"],
      ] },
    ],
    math: [
      { t: "h", text: "What a held-out score estimates" },
      { t: "p", text: "Let $(x, y) \\sim \\mathcal{D}$ be the distribution of future data, $f$ a trained model and $\\ell(f(x), y)$ a per-example loss (for accuracy, $\\ell = \\mathbb{1}[f(x) \\neq y]$ gives the error). The quantity we want is the **risk** (expected loss):" },
      { t: "math", tex: String.raw`R(f) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\ell(f(x), y)\big], \qquad \hat R_{\text{test}}(f) = \frac{1}{m}\sum_{j=1}^{m} \ell\big(f(x_j), y_j\big)` },
      { t: "p", text: "Here $\\hat R_{\\text{test}}$ is the average loss on $m$ test examples. If the test examples are drawn from $\\mathcal{D}$ **independently of how $f$ was built**, each term has expectation $R(f)$, so $\\mathbb{E}[\\hat R_{\\text{test}}] = R(f)$ (unbiased), and its variance is $\\mathrm{Var}(\\ell)/m$. For accuracy $a$ on $m$ examples, each term is a Bernoulli variable, giving the standard error" },
      { t: "math", tex: String.raw`\mathrm{SE}(\hat a) = \sqrt{\frac{a(1-a)}{m}}`, caption: "With $a = 0.9$ and $m = 100$ the SE is 0.03; with $m = 10{,}000$ it is 0.003. Small test sets give noisy numbers." },
      { t: "p", text: "The **training** loss is biased downward because $f$ was chosen to minimize it on exactly those examples. The same thing happens, more weakly, to the validation score once we pick the best of many configurations on it: the maximum of several noisy estimates is biased upward (the *winner's curse*). That is the reason for a separate test set, or nested CV." },
      { t: "h", text: "The k-fold estimate" },
      { t: "p", text: "Let the folds be index sets $F_1, \\dots, F_k$ partitioning $\\{1..n\\}$, and $f^{(-i)}$ the model trained on all examples not in $F_i$. Then" },
      { t: "math", tex: String.raw`\hat R_{\text{CV}} = \frac{1}{k}\sum_{i=1}^{k} s_i, \qquad s_i = \frac{1}{|F_i|}\sum_{j\in F_i}\ell\big(f^{(-i)}(x_j), y_j\big), \qquad \mathrm{SE} \approx \frac{s}{\sqrt{k}},\ \ s^2 = \frac{1}{k-1}\sum_{i=1}^{k}(s_i - \hat R_{\text{CV}})^2` },
      { t: "p", text: "Each $f^{(-i)}$ is trained on a fraction $(k-1)/k$ of the data, so CV slightly **underestimates** the performance of a model trained on all $n$ examples (pessimistic bias; smaller for larger $k$). The $s/\\sqrt{k}$ formula treats folds as independent, which they are not (their training sets overlap); Bengio and Grandvalet (2004, *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*) proved no unbiased variance estimator exists, so read it as a rough guide, not an exact interval." },
      { t: "h", text: "Worked example — 5-fold CV" },
      { t: "p", text: "Fold accuracies $s = (0.82, 0.78, 0.85, 0.80, 0.75)$:" },
      { t: "steps", items: [
        { title: "Mean", tex: String.raw`\bar s = \frac{0.82 + 0.78 + 0.85 + 0.80 + 0.75}{5} = \frac{4.00}{5} = 0.800` },
        { title: "Squared deviations", tex: String.raw`(0.02^2,\ 0.02^2,\ 0.05^2,\ 0^2,\ 0.05^2) = (0.0004,\ 0.0004,\ 0.0025,\ 0,\ 0.0025),\ \ \text{sum} = 0.0058` },
        { title: "Sample standard deviation", tex: String.raw`s = \sqrt{0.0058 / 4} = \sqrt{0.00145} = 0.03808` },
        { title: "Standard error", tex: String.raw`\mathrm{SE} = 0.03808 / \sqrt{5} = 0.01703` },
        { title: "Rough 95% interval (t distribution, 4 degrees of freedom)", tex: String.raw`0.800 \pm 2.776 \times 0.01703 = [0.753,\ 0.847]` },
      ] },
      { t: "p", text: "Fold sizes for $n = 103$, $k = 5$: $103 = 5\\times20 + 3$, so the first three folds get 21 examples and the last two get 20 ($21+21+21+20+20 = 103$)." },
      { t: "h", text: "Cost of nested CV" },
      { t: "p", text: "With $K_o$ outer folds, $K_i$ inner folds and a grid of $G$ configurations, each outer fold runs $K_i G$ inner fits plus one refit of the winner:" },
      { t: "math", tex: String.raw`\#\text{fits} = K_o\,(K_i\,G + 1) = 5\,(5\times 6 + 1) = 155 \quad (K_o = K_i = 5,\ G = 6)` },
      { t: "h", text: "Why selecting features on all data inflates the score" },
      { t: "p", text: "Take $p$ pure-noise features and random labels. For each feature, the difference in class means on $n$ examples has a spread of order $1/\\sqrt{n}$ around zero; among $p = 5000$ features, the top 20 by absolute difference are the most extreme chance correlations **with these particular labels** — including the validation labels. A classifier built on them looks accurate on every fold because the selection already saw the answers. Repeat the selection inside each fold (on training rows only) and the chance correlations no longer transfer: accuracy falls to the 50% that random labels deserve. The from-scratch code below measures exactly this: 0.917 versus 0.483." },
    ],
    visual: [
      { t: "p", text: "The polynomial-fitting simulator shows why a separate validation set is needed at all. Training error always falls as model capacity grows; only held-out error reveals overfitting." },
      { t: "sim", id: "fitting", caption: "Filled points are the training set, hollow points the validation set." },
      { t: "list", items: [
        "Step the degree up from 0. Watch training error fall monotonically while validation error falls, bottoms out, then rises: the minimum of the **validation** curve is the degree you would choose.",
        "Reduce the number of samples $n$ to 10–15 and repeat. The validation curve becomes jumpy — with few held-out points the estimate itself is noisy, which is the motivation for k-fold CV.",
        "Increase $n$ to 200: the best degree becomes clearer and overfitting sets in later, because more data supports more capacity.",
        "Notice that you are now *choosing* the degree by looking at validation error — exactly the decision that makes the validation score optimistic and calls for a final, untouched test set.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Every splitter is a few lines of index bookkeeping. The program implements k-fold, stratified, group and time-series splitting from scratch, cross-checks them against scikit-learn, and then runs the classic leakage experiment: 5000 noise features, random labels, and feature selection done either before or inside cross-validation." },
      { t: "code", lang: "python", title: "splits_from_scratch.py", code: SPLITS_NUMPY, output: SPLITS_NUMPY_OUT },
      { t: "list", items: [
        "Plain k-fold put between 1 and 4 of the 10 positives in each validation fold; stratification puts exactly 2 in each, matching scikit-learn's `StratifiedKFold` counts.",
        "`time_series_split` reproduces `TimeSeriesSplit` exactly: block size $\\lfloor n/(k+1)\\rfloor = 4$; fold $f$ trains on all indices before its block.",
        "`group_kfold` uses the same greedy idea as scikit-learn's `GroupKFold` (largest groups first, into the emptiest fold) and guarantees no patient appears on both sides.",
        "The leakage demo is the headline: **0.917 accuracy on pure noise** when the 20 features are chosen with all labels, versus **0.483** (chance) when the selection is repeated inside each fold. The model, data and folds are identical — only the order of operations changed.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "A complete hold-out workflow: one seeded permutation gives 60/20/20 index sets, standardization statistics come from the training rows only, the checkpoint is chosen on validation accuracy, and the test set is evaluated once." },
          { t: "code", lang: "python", title: "holdout_pytorch.py", code: SPLITS_TORCH, output: SPLITS_TORCH_OUT },
          { t: "p", text: "Validation accuracy peaked at epoch 20 while training loss kept falling; the checkpoint from that epoch scores 0.974 on the test set. With 115 test examples one example is worth 0.0087 accuracy, and the standard error is $\\sqrt{0.974\\times0.026/115} \\approx 0.015$ — so differences of a point or two between models on this test set are noise." },
          { t: "callout", kind: "tip", text: "`torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(0))` does the same split on a `Dataset`. It splits randomly — for grouped or temporal data, build the index lists yourself and wrap them with `torch.utils.data.Subset`." },
        ] },
        { label: "scikit-learn", blocks: [
          { t: "p", text: "scikit-learn provides the splitters and, crucially, `Pipeline`, which re-fits every preprocessing step inside each training fold. The program shows the feature-selection leak and its fix, the duplicate leak and the `GroupKFold` fix, and nested CV." },
          { t: "code", lang: "python", title: "cv_sklearn.py", code: SPLITS_SKLEARN, output: SPLITS_SKLEARN_OUT },
          { t: "list", items: [
            "Leak 1: 0.933 vs 0.417 on noise — the same effect as the from-scratch demo, with `SelectKBest` + logistic regression.",
            "Leak 2: with three jittered copies of each example, a 1-nearest-neighbour classifier finds a copy of almost every validation point in the training folds and scores 0.965 even though 20% of labels are random noise. Grouping copies together gives the honest 0.702.",
            "Nested CV: `GridSearchCV` is itself an estimator, so passing it to `cross_val_score` runs the tuning inside each outer fold. Here the optimistic inner best score (0.979) and the nested estimate (0.975) are close because only 6 settings were compared; the gap grows with the number of configurations tried.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same hold-out workflow in Keras. `Normalization.adapt` must be called on the training rows only; `validation_split=` in `fit` would take the **last** rows without shuffling, which is wrong for sorted data — pass an explicit `validation_data` instead." },
          { t: "code", lang: "python", title: "holdout_keras.py (not executed in CI)", code: SPLITS_KERAS, output: "Prints the test accuracy of the checkpoint with the best validation accuracy (restored by EarlyStopping) — a single number close to the PyTorch result; the exact value depends on the TensorFlow version." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Scheme", "Advantages", "Disadvantages", "Use when"], rows: [
        ["Single hold-out (train/val/test)", "Cheap: one training run per configuration; simple", "Noisy with small data; wastes data; depends on one random split", "Large datasets, expensive deep models"],
        ["k-fold CV (k = 5 or 10)", "Uses every example for validation; gives a spread", "k× the training cost; folds not independent", "Small/medium data, cheap models"],
        ["Stratified k-fold", "Stable class ratios per fold; avoids folds with no positives", "Classification only (regression needs binning)", "Classification, especially imbalanced"],
        ["Group k-fold", "Simulates unseen groups (patients, users, documents)", "Fewer effective samples; folds can be unequal", "Any grouped/correlated data"],
        ["Time-series split", "Respects causality; mimics deployment", "Early folds have little training data; non-stationarity", "Forecasting, any time-ordered data"],
        ["Leave-one-out", "Uses n−1 examples for training", "n fits; high variance", "Tiny datasets"],
        ["Repeated / Monte-Carlo CV", "Averages out partition luck", "More compute", "Comparing close models on small data"],
        ["Nested CV", "Unbiased estimate of a procedure that includes tuning", "Very expensive (K_o·K_i·G fits)", "Reporting results of tuned models on small data"],
        ["Bootstrap / out-of-bag", "Each bootstrap sample leaves ~36.8% of examples out to evaluate on", "Optimistic or pessimistic biases need corrections (.632 estimator)", "Bagging/random forests, confidence intervals"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Medical imaging**: splits are made per patient (group splits) because scans of the same patient are strongly correlated; Ambroise and McLachlan's 2002 analysis of gene-selection bias is a standard reference for why the selection must happen inside CV.",
        "**Forecasting** (demand, energy load, finance): rolling-origin / expanding-window evaluation, often with a gap equal to the forecast horizon.",
        "**Recommender systems**: hold out each user's *latest* interactions (temporal leave-last-out) rather than random interactions, so the model is tested on predicting the future.",
        "**Benchmarks**: datasets such as ImageNet publish fixed train/validation/test splits so results are comparable; hidden test sets and evaluation servers limit how often teams can query the test labels.",
        "**Competitions**: public/private leaderboards are a train/validation/test split in disguise — teams that tune heavily against the public leaderboard often drop on the private one.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**More validation data → more reliable decisions, less training data.** With huge datasets a small fraction suffices; with small data use CV.",
        "**CV costs k trainings.** For deep networks that take hours, a single well-sized validation set is the norm; CV is used for small models or final comparisons.",
        "**Larger k**: less pessimistic bias (each model trains on more data) but more compute and more correlated folds. 5 or 10 is the usual compromise.",
        "**Group/time splits are harder but honest.** They usually *lower* the score relative to random splits — that lower number is the one that predicts deployment.",
        "**Every look at the test set costs some of its validity.** Reusing it for many decisions turns it into a validation set; nested CV or a fresh test set fixes this at extra cost.",
        "**No split can fix distribution shift.** If future data differs from all of today's data, any held-out estimate is optimistic; monitor in production (see [MLOps](/learn/m18/mlops)).",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Fitting a scaler / imputer / PCA / tokenizer on the full dataset before splitting.** Held-out statistics leak into training; with few examples or many features the effect can be large. Fix: fit inside the training fold — use a `Pipeline`.",
        "**Selecting features (or tuning) on all data and then cross-validating.** As the demo showed, this can turn pure noise into 92% accuracy. Fix: the selection is part of the model; put it inside the CV loop.",
        "**Oversampling (e.g. duplicating or SMOTE-ing minority examples) before splitting.** Copies of the same minority point land in both train and validation. Fix: resample only the training fold.",
        "**Random splits of grouped data.** Images of the same patient, frames of the same video, or reviews by the same user on both sides measure memorization of the group. Fix: group splits.",
        "**Shuffling time series.** The model sees the future during training. Fix: forward-chaining splits; compute features from the past only.",
        "**Using the test set for early stopping or model selection.** It becomes a validation set and its score becomes optimistic. Fix: separate validation set; test once.",
        "**Reporting the best validation score as the final result.** It is the maximum of noisy estimates and biased upward. Fix: report the test score or a nested-CV estimate.",
        "**Ignoring uncertainty.** A 0.5-point improvement on 500 test examples (SE ≈ 1.3 points at 90% accuracy) is not evidence of anything. Fix: report standard errors or confidence intervals; use paired comparisons on the same examples.",
        "**Not stratifying with rare classes.** A fold with no positives makes recall/AUC undefined or wildly variable. Fix: stratified folds.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Why do we need a validation set in addition to a test set?", a: "The validation set is used to make decisions — hyperparameters, architecture, early-stopping epoch. Those decisions fit the validation data to some degree, so its score becomes optimistic. The test set is kept untouched until the end so it gives an unbiased estimate of the final model's performance." },
      { t: "qa", level: "beginner", q: "What is k-fold cross-validation?", a: "Split the data into $k$ folds; for each fold, train a fresh model on the other $k-1$ folds and evaluate on it; average the $k$ scores. Every example is used for validation once, and the spread across folds indicates the estimate's variability. It costs $k$ trainings." },
      { t: "qa", level: "beginner", q: "What is data leakage? Give two examples.", a: "Leakage is any way information about the evaluation data (or the label) reaches the model during training, making held-out scores optimistic. Examples: standardizing with the mean and variance of the whole dataset before splitting; the same customer appearing in both train and test; a feature that is only known after the outcome (e.g. \"discharge date\" for predicting length of stay)." },
      { t: "qa", level: "intermediate", q: "When should you use stratified, group or time-series splits?", a: "Stratified: classification, especially with imbalanced classes, so each fold keeps the class ratio. Group: when examples are correlated within groups (patients, users, documents, videos) and the model will face new groups — split so no group straddles train and validation. Time-series: whenever data is time-ordered and the model predicts the future — train on the past, validate on the next block, never shuffle." },
      { t: "qa", level: "intermediate", q: "Your model gets 99% in CV but 70% in production. What do you check?", a: "Leakage first: preprocessing or feature selection outside the folds, duplicates or the same entities across splits, temporal leakage, target-leaking features (check feature importances for suspiciously strong single features). Then distribution shift between the CV data and production data (different time period, population, sensor). Finally whether the metric used in CV matches what is measured in production." },
      { t: "qa", level: "intermediate", q: "How large should the test set be?", a: "Large enough that the standard error is small relative to the differences you care about. For accuracy $a$ on $m$ examples, $\\mathrm{SE} = \\sqrt{a(1-a)/m}$: at $a = 0.9$, $m = 1000$ gives SE ≈ 0.0095, $m = 10{,}000$ gives ≈ 0.003. With very large datasets 1% may be plenty; with small datasets prefer CV." },
      { t: "qa", level: "advanced", q: "Explain nested cross-validation and why non-nested CV with tuning is optimistic.", a: "In non-nested CV you tune hyperparameters by picking the configuration with the best CV score and report that score — but it is the maximum of several noisy estimates computed on the same folds, so it is biased upward (more so with more configurations and smaller data; see Cawley and Talbot, 2010, JMLR). Nested CV wraps tuning inside an outer loop: for each outer fold, run an inner CV on the outer-training part to choose hyperparameters, refit, and evaluate on the untouched outer fold. The outer average estimates the performance of the whole procedure, tuning included." },
      { t: "qa", level: "advanced", q: "Is the k-fold standard error $s/\\sqrt{k}$ trustworthy?", a: "Only roughly. The fold scores are not independent: any two training sets share a fraction $(k-2)/(k-1)$ of their examples, so scores are positively correlated and $s/\\sqrt{k}$ tends to underestimate the true variability. Bengio and Grandvalet (2004) proved there is no unbiased estimator of the variance of k-fold CV. Use it as a guide, prefer repeated CV or a large independent test set for important comparisons, and compare models with paired tests on the same folds." },
      { t: "qa", level: "advanced", q: "How do you evaluate a forecasting model whose features use a 7-day rolling window and whose labels arrive 3 days late?", a: "Use forward-chaining splits with a **gap**: training data ends at time $T$, and validation starts after $T$ plus the label delay (3 days) — and features for validation days must be computed only from data available at prediction time. Without the gap, the last training labels overlap information visible in early validation features, inflating the score. Evaluate across several cut-off points to see variation over time." },
    ],
    practice: [
      { t: "quiz", questions: [
        { q: "When must a StandardScaler's mean and variance be computed?", options: ["On the full dataset before splitting", "On the training data only (inside each CV fold), then applied to validation/test", "On the test set, so it matches deployment", "Separately on each split"], answer: 1, explain: "Statistics computed with held-out rows leak information. Fit on the training portion, apply the same transform elsewhere — separate fitting per split would also make validation features inconsistent with training." },
        { kind: "numeric", q: "5-fold CV of a dataset with 103 examples: how many examples are in the largest validation fold?", answer: 21, tolerance: 0, explain: "$103 = 5\\times 20 + 3$; the first three folds get 21." },
        { kind: "numeric", q: "Nested CV with 5 outer folds, 3 inner folds and a grid of 10 configurations (plus one refit of the winner per outer fold): how many model fits?", answer: 155, tolerance: 0, explain: "$K_o(K_i G + 1) = 5\\times(3\\times10 + 1) = 155$." },
        { kind: "numeric", q: "A model has 0.9 accuracy on a test set of 400 examples. What is the standard error of that accuracy? (3 decimals)", answer: 0.015, tolerance: 0.0005, explain: "$\\sqrt{0.9\\times0.1/400} = \\sqrt{0.000225} = 0.015$." },
        { q: "You have 3 X-rays per patient for 1000 patients. Which splitter is correct?", options: ["KFold with shuffle", "StratifiedKFold", "GroupKFold with patient id as the group", "TimeSeriesSplit"], answer: 2, explain: "X-rays of the same patient are correlated; the model will be used on new patients, so whole patients must be held out." },
        { q: "Which feature is most likely target leakage for predicting whether a loan will default?", options: ["Applicant income at application time", "Number of collection calls made to the borrower", "Loan amount requested", "Applicant age"], answer: 1, explain: "Collection calls happen *because* the borrower is already in arrears — the value is not known at decision time and encodes the label." },
        { q: "Why is the best validation score among 200 tried configurations an optimistic estimate?", options: ["Because the validation set is smaller than the training set", "Because the maximum of many noisy estimates is biased upward", "Because validation data is used for gradient updates", "It is not optimistic"], answer: 1, explain: "Each score = true performance + noise; selecting the maximum also selects favourable noise (winner's curse)." },
      ] },
      { t: "exercise", title: "Confidence interval for a CV estimate", prompt: "10-fold CV gives fold accuracies 0.71, 0.74, 0.69, 0.75, 0.72, 0.70, 0.73, 0.76, 0.68, 0.72. Compute the mean, the sample standard deviation, the standard error and a rough 95% interval using $t_{0.975, 9} = 2.262$.", hint: "The values sum to 7.20.", solution: [
        { t: "steps", items: [
          { title: "Mean", tex: String.raw`\bar s = 7.20/10 = 0.720` },
          { title: "Deviations", text: "(−0.01, 0.02, −0.03, 0.03, 0, −0.02, 0.01, 0.04, −0.04, 0); squares sum to 0.0001+0.0004+0.0009+0.0009+0+0.0004+0.0001+0.0016+0.0016+0 = 0.0060." },
          { title: "Standard deviation and SE", tex: String.raw`s = \sqrt{0.0060/9} = 0.02582,\qquad \mathrm{SE} = 0.02582/\sqrt{10} = 0.00816` },
          { title: "Interval", tex: String.raw`0.720 \pm 2.262\times0.00816 = [0.702,\ 0.738]` },
        ] },
        { t: "p", text: "Remember the caveat: CV folds are correlated, so the true uncertainty is somewhat larger than this interval suggests." },
      ] },
      { t: "exercise", title: "Find and fix the leak", prompt: "A colleague writes: `X = StandardScaler().fit_transform(X); X = SelectKBest(k=50).fit_transform(X, y); X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0); model.fit(X_tr, y_tr); print(model.score(X_te, y_te))`. Identify both leaks and rewrite it so the test score is honest, then extend it to 5-fold CV.", hint: "Which two lines see the test rows before the split?", solution: [
        { t: "p", text: "Both `StandardScaler().fit_transform(X)` and `SelectKBest(...).fit_transform(X, y)` are fitted on all rows; the second one even uses the test **labels** to choose features — a severe leak. Split first and wrap the steps in a pipeline:" },
        { t: "code", lang: "python", code: String.raw`from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=50), LogisticRegression(max_iter=1000))
pipe.fit(X_tr, y_tr)                       # scaler and selector see training rows only
print("test accuracy:", pipe.score(X_te, y_te))

cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X_tr, y_tr, cv=cv)   # the whole pipeline is re-fitted in every fold
print("CV accuracy: %.3f +/- %.3f" % (scores.mean(), scores.std(ddof=1) / 5 ** 0.5))` },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Training set fits parameters; validation set makes decisions; test set is used once for the final, unbiased estimate.",
        "A held-out score estimates the risk $R(f) = \\mathbb{E}[\\ell(f(x), y)]$ only if the held-out data played no role in building $f$; SE of accuracy $= \\sqrt{a(1-a)/m}$.",
        "k-fold CV: $\\hat R_{\\text{CV}} = \\frac1k\\sum_i s_i$, SE $\\approx s/\\sqrt{k}$ (rough — folds are correlated); slightly pessimistic since each model sees $(k-1)/k$ of the data.",
        "Choose the splitter to simulate deployment: stratified (class ratios), group (new entities), time-series (future), nested CV (tuning included).",
        "Leakage = held-out information reaching training: preprocessing/feature selection on all data, duplicates, temporal leaks, target leakage. Fit every data-dependent step inside the training fold.",
        "Feature selection on noise before CV produced 0.917 accuracy; inside CV, 0.483 — same data, different order of operations.",
        "Nested CV cost: $K_o(K_iG + 1)$ fits.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Classification Metrics, ROC and PR Curves](/learn/m17/classification-metrics) — now that the held-out set is honest, choose *what* to measure on it: accuracy is rarely enough.",
        "[Regression Metrics](/learn/m17/regression-metrics) — MAE, RMSE, R² and residual analysis for continuous targets.",
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — how to use the validation set efficiently (random and Bayesian search, early stopping) and keep experiments reproducible.",
      ] },
    ],
  },
};

export default lesson;
