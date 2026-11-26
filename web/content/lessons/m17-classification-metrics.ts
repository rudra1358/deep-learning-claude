import type { Lesson } from "@/lib/types";

const CLS_NUMPY = String.raw`import numpy as np
from sklearn import metrics as skm

# ---------------- data: 10% positives; negatives ~ N(0, I), positives ~ N(mu, I) in 10-D ----------------
rng = np.random.default_rng(0)
n, n_pos = 4000, 400
mu = np.r_[np.full(4, 0.8), np.zeros(6)]          # class means differ in 4 of the 10 features
X = rng.normal(size=(n, 10))
y = np.zeros(n, dtype=int)
y[:n_pos] = 1
X[:n_pos] += mu
idx = rng.permutation(n)                            # shuffle, then split 50/25/25 into train / val / test
tr, va, te = idx[:2000], idx[2000:3000], idx[3000:]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------- model: logistic regression trained by gradient descent ----------------
w, b = np.zeros(X.shape[1]), 0.0
for _ in range(2000):
    p = sigmoid(X[tr] @ w + b)
    w -= 0.1 * X[tr].T @ (p - y[tr]) / len(tr)
    b -= 0.1 * np.mean(p - y[tr])
logit = X @ w + b                              # one score per example


# ---------------- metrics from scratch ----------------
def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def prf(y, pred, beta=1.0):
    tp, fp, fn, tn = confusion(y, pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    b2 = beta ** 2
    f = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
    return prec, rec, f


def roc_points(y, s):
    """Sweep the threshold down through the sorted scores; one (FPR, TPR) point per distinct score."""
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]      # last index of each run of tied scores
    tps, fps = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]], tps, fps


def auc_trapezoid(y, s):
    fpr, tpr, _, _ = roc_points(y, s)
    return np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)


def auc_rank(y, s):
    """P(score of random positive > score of random negative): Mann-Whitney U / (n_pos * n_neg)."""
    ranks = np.empty(len(s)); ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                         # average ranks of tied scores
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, s):
    _, tpr, tps, fps = roc_points(y, s)
    precision = tps / (tps + fps)
    return np.sum(np.diff(tpr) * precision)        # sum over thresholds of (R_k - R_{k-1}) * P_k


def ece(prob, y, bins=10):
    """Expected calibration error of P(y=1): bin by predicted probability, weight |mean prob - frequency|."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, bins - 1)
    return sum(np.mean(which == k) * abs(prob[which == k].mean() - y[which == k].mean())
               for k in range(bins) if np.any(which == k))


def nll(prob, y, eps=1e-12):
    prob = np.clip(prob, eps, 1 - eps)
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))


yt, st = y[te], logit[te]
pred = (sigmoid(st) >= 0.5).astype(int)
tp, fp, fn, tn = confusion(yt, pred)
print(f"test set: {yt.sum()} positives / {len(yt)}   confusion at p>=0.5: TP={tp} FP={fp} FN={fn} TN={tn}")
P, R, F1 = prf(yt, pred)
print(f"accuracy {np.mean(pred == yt):.4f} (always-negative baseline {1 - yt.mean():.4f})")
print(f"precision {P:.4f}  recall {R:.4f}  F1 {F1:.4f}  F2 {prf(yt, pred, 2)[2]:.4f}")
print(f"ROC AUC trapezoid {auc_trapezoid(yt, st):.6f}  rank {auc_rank(yt, st):.6f}  sklearn {skm.roc_auc_score(yt, st):.6f}")
print(f"average precision {average_precision(yt, st):.6f}  sklearn {skm.average_precision_score(yt, st):.6f}")
print("sklearn P/R/F1 agree:", np.allclose([P, R, F1], skm.precision_recall_fscore_support(yt, pred, average='binary')[:3]))

# ---------------- threshold selection on VALIDATION, applied to test ----------------
cands = np.unique(sigmoid(logit[va]))
f1s = [prf(y[va], (sigmoid(logit[va]) >= c).astype(int))[2] for c in cands]
t_best = cands[int(np.argmax(f1s))]
P2, R2, F2b = prf(yt, (sigmoid(st) >= t_best).astype(int))
print(f"\nbest-F1 threshold on validation: {t_best:.3f} -> test precision {P2:.4f} recall {R2:.4f} F1 {F2b:.4f}")

# ---------------- calibration and temperature scaling ----------------
over = 3.0 * logit          # simulate an over-confident network: same ranking, logits 3x too large
Ts = np.exp(np.linspace(np.log(0.2), np.log(10), 2001))
T = Ts[int(np.argmin([nll(sigmoid(over[va] / t), y[va]) for t in Ts]))]   # fit T on validation NLL
print(f"\nwell-scaled model    test ECE {ece(sigmoid(st), yt):.4f}  NLL {nll(sigmoid(st), yt):.4f}")
print(f"over-confident model test ECE {ece(sigmoid(over[te]), yt):.4f}  NLL {nll(sigmoid(over[te]), yt):.4f}")
print(f"temperature fitted on validation T = {T:.3f}")
print(f"after scaling        test ECE {ece(sigmoid(over[te] / T), yt):.4f}  NLL {nll(sigmoid(over[te] / T), yt):.4f}")
print(f"AUC unchanged by T: {auc_rank(yt, over[te]):.6f} -> {auc_rank(yt, over[te] / T):.6f}")`;

const CLS_NUMPY_OUT = String.raw`test set: 97 positives / 1000   confusion at p>=0.5: TP=27 FP=10 FN=70 TN=893
accuracy 0.9200 (always-negative baseline 0.9030)
precision 0.7297  recall 0.2784  F1 0.4030  F2 0.3176
ROC AUC trapezoid 0.894521  rank 0.894521  sklearn 0.894521
average precision 0.598601  sklearn 0.598601
sklearn P/R/F1 agree: True

best-F1 threshold on validation: 0.173 -> test precision 0.3886 recall 0.7010 F1 0.5000

well-scaled model    test ECE 0.0204  NLL 0.2061
over-confident model test ECE 0.0631  NLL 0.3348
temperature fitted on validation T = 3.235
after scaling        test ECE 0.0250  NLL 0.2093
AUC unchanged by T: 0.894521 -> 0.894521`;

const CLS_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- data: 20% positives, two overlapping Gaussian classes in 10-D; a SMALL training set invites overfitting ----
def make(n, pos_frac=0.2):
    y = (rng.random(n) < pos_frac).astype(np.float32)
    X = rng.normal(size=(n, 10)).astype(np.float32)
    X[:, :4] += 0.8 * y[:, None]
    return torch.tensor(X), torch.tensor(y)

Xtr, ytr = make(600)
Xva, yva = make(2000)
Xte, yte = make(4000)


def ece(prob, y, bins=10):
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi)
        if m.any():
            total += m.float().mean() * (prob[m].mean() - y[m].mean()).abs()
    return float(total)


def train(pos_weight=None):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(1500):                          # far past the best epoch: the net becomes over-confident
        opt.zero_grad()
        loss_fn(net(Xtr).squeeze(1), ytr).backward()
        opt.step()
    return net


net = train()
with torch.no_grad():
    z_va, z_te = net(Xva).squeeze(1), net(Xte).squeeze(1)
bce = nn.BCEWithLogitsLoss()
p_te = torch.sigmoid(z_te)
print(f"train acc {((net(Xtr).squeeze(1) > 0).float() == ytr).float().mean():.3f}   test acc {((z_te > 0).float() == yte).float().mean():.3f}")
print(f"test ROC AUC {roc_auc_score(yte, z_te):.4f}  AP {average_precision_score(yte, z_te):.4f}  F1@0.5 {f1_score(yte, (p_te >= 0.5).int()):.4f}")
print(f"before scaling: test NLL {bce(z_te, yte):.4f}  ECE {ece(p_te, yte):.4f}  mean confidence {torch.maximum(p_te, 1 - p_te).mean():.3f}")

# ---- temperature scaling: one scalar T > 0 (parameterized as exp(log_t)) fitted on VALIDATION logits ----
log_t = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)


def closure():
    opt.zero_grad()
    loss = bce(z_va / log_t.exp(), yva)
    loss.backward()
    return loss


opt.step(closure)
T = log_t.exp().item()
p_cal = torch.sigmoid(z_te / T)
print(f"fitted temperature T = {T:.3f}")
print(f"after scaling:  test NLL {bce(z_te / T, yte):.4f}  ECE {ece(p_cal, yte):.4f}  mean confidence {torch.maximum(p_cal, 1 - p_cal).mean():.3f}")
print(f"ranking metrics unchanged: AUC {roc_auc_score(yte, z_te / T):.4f}, predictions at 0.5 unchanged: {bool(((z_te / T) > 0).eq(z_te > 0).all())}")

# ---- class imbalance: up-weight the positive class in the loss (pos_weight = n_neg / n_pos) ----
pw = (ytr == 0).sum() / (ytr == 1).sum()
net_w = train(pos_weight=pw)
with torch.no_grad():
    zw = net_w(Xte).squeeze(1)
pred_w = (zw > 0).int()
print(f"\npos_weight = {pw:.2f}: recall {(pred_w[yte == 1] == 1).float().mean():.3f} vs {((z_te > 0)[yte == 1]).float().mean():.3f} unweighted; "
      f"precision {yte[pred_w == 1].mean():.3f} vs {yte[z_te > 0].mean():.3f}")`;

const CLS_TORCH_OUT = String.raw`train acc 1.000   test acc 0.807
test ROC AUC 0.7898  AP 0.5296  F1@0.5 0.4847
before scaling: test NLL 1.5499  ECE 0.1645  mean confidence 0.970
fitted temperature T = 7.603
after scaling:  test NLL 0.4287  ECE 0.0574  mean confidence 0.810
ranking metrics unchanged: AUC 0.7898, predictions at 0.5 unchanged: True

pos_weight = 4.22: recall 0.471 vs 0.440 unweighted; precision 0.515 vs 0.540`;

const CLS_KERAS = String.raw`import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
y = (rng.random(5000) < 0.2).astype("float32")
X = rng.normal(size=(5000, 10)).astype("float32")
X[:, :4] += 0.8 * y[:, None]

tf.keras.utils.set_random_seed(0)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),                       # logits
])
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", from_logits=True, name="roc_auc"),
        tf.keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc"),
        tf.keras.metrics.Precision(thresholds=0.0, name="precision"),   # threshold on logits: 0 <=> p = 0.5
        tf.keras.metrics.Recall(thresholds=0.0, name="recall"),
    ],
)
model.fit(X[:3000], y[:3000], validation_data=(X[3000:4000], y[3000:4000]), epochs=20, verbose=0)
print(dict(zip(model.metrics_names, model.evaluate(X[4000:], y[4000:], verbose=0))))`;

const lesson: Lesson = {
  id: "m17/classification-metrics",
  title: "Classification Metrics, ROC and PR Curves",
  summary: "How to measure a classifier properly: the confusion matrix, accuracy, precision, recall, F1/Fβ, macro/micro/weighted averages, ROC curves and AUC (and why AUC is a ranking probability), precision–recall curves and average precision, picking a threshold, calibration with reliability diagrams, ECE and temperature scaling, and what class imbalance does to all of them.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m17/data-splits", "m03/linear-logistic-regression", "m05/classification-losses"],
  next: ["m17/regression-metrics", "m17/experimentation", "m12/object-detection"],
  terms: ["confusion-matrix", "precision", "recall", "f1-score", "roc-curve", "roc-auc", "pr-curve", "average-precision", "calibration", "expected-calibration-error", "temperature-scaling", "class-imbalance"],
  sections: {
    intro: [
      { t: "p", text: "A trained classifier gives each example a **score** — a logit or a probability. Classification metrics turn scores into decisions and judge those decisions. The first metric people reach for, **accuracy**, is often the least useful: on a fraud dataset with 0.1% fraud, a model that never flags anything is 99.9% accurate — and useless." },
      { t: "p", text: "Different jobs care about different mistakes. A cancer screen must not miss patients (high **recall**). A spam filter must not bin real mail (high **precision**). A search engine needs good **ranking**. A medical risk model or a weather forecast needs probabilities you can take at face value (**calibration**). We build every standard metric from the four counts of the **confusion matrix**, move to **threshold-free** summaries — the **ROC curve** with its AUC and the **precision–recall curve** with average precision — and finally ask whether the probabilities themselves can be trusted." },
      { t: "p", text: "ROC analysis comes from signal detection theory and became popular in medicine (Hanley and McNeil, 1982, *The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve*). Davis and Goadrich (2006, *The Relationship Between Precision-Recall and ROC Curves*) showed how PR and ROC curves relate. Guo, Pleiss, Sun and Weinberger (2017, *On Calibration of Modern Neural Networks*) showed that modern deep networks are often over-confident, and that one-parameter **temperature scaling** fixes most of it." },
      { t: "callout", kind: "key", text: "Choose metrics by the **cost of each error** and **how the output is used**: decisions at a fixed threshold → precision/recall/F-scores at that threshold; ranking quality → ROC AUC (balanced data) or PR AUC / AP (rare positives); probabilities used as probabilities → calibration (ECE, NLL, Brier)." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Train/validation/test splits](/learn/m17/data-splits) — every metric here is computed on held-out data. Thresholds and temperatures are chosen on the **validation** set, never the test set.",
        "[Linear and logistic regression](/learn/m03/linear-logistic-regression) — a classifier outputs a logit $z$ and a probability $p = \\sigma(z)$; the decision is $p \\ge t$.",
        "[Classification losses](/learn/m05/classification-losses) — cross-entropy (NLL) is both the training loss and a metric that reacts to calibration.",
        "[Probability](/learn/m02/probability) — conditional probability $P(A\\mid B)$. Precision and recall are two different conditional probabilities, and Bayes' rule links them through the class prior.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A smoke alarm", text: "A smoke alarm has a sensitivity dial (the **threshold**). Turn it up and it catches every fire (**recall** near 1), but it also goes off whenever you make toast (**false positives**, low **precision**). Turn it down and it stays quiet while you cook, but might miss a real fire (**false negatives**). No setting is \"correct\" — it depends on what a missed fire and a false alarm each cost. ROC and PR curves show **every** setting at once; a better alarm has a better whole curve, not a cleverer dial position." },
      { t: "p", text: "Keep two questions apart. **Recall** asks: *of the real fires, how many did we catch?* **Precision** asks: *of the alarms, how many were real fires?* Same numerator (true positives), different denominators. When fires are rare, even a small false-alarm *rate* produces many false alarms in total, so precision collapses while recall and the false-positive rate look fine. That is why PR curves are preferred for rare positives." },
      { t: "p", text: "The **AUC** has a lovely meaning: pick one random positive and one random negative; AUC is the probability that the model scores the positive higher. It measures **ranking** only: multiply all scores by 3, or apply any increasing function, and AUC stays the same. **Calibration** asks the other question: of all examples given probability 0.8, are about 80% actually positive? A model can rank perfectly and still be badly calibrated, and vice versa." },
    ],
    mechanism: [
      { t: "h", text: "From scores to a confusion matrix" },
      { t: "steps", items: [
        { title: "Score every held-out example", text: "Logits $z_i$ or probabilities $p_i = \\sigma(z_i)$ (softmax for more than two classes)." },
        { title: "Apply a threshold", text: "Predict positive when $p_i \\ge t$ (the same as $z_i \\ge \\operatorname{logit}(t)$). The default $t = 0.5$ is only optimal when probabilities are calibrated and both errors cost the same." },
        { title: "Count the four outcomes", text: "**TP** (positive, predicted positive); **FP** (negative, predicted positive — a false alarm, type I error); **FN** (positive, predicted negative — a miss, type II error); **TN** (negative, predicted negative)." },
        { title: "Compute threshold metrics", text: "Accuracy, precision, recall (TPR), specificity (TNR), false-positive rate, F1/Fβ — all ratios of these counts." },
      ] },
      { t: "h", text: "Sweeping the threshold: ROC and PR curves" },
      { t: "steps", items: [
        { title: "Sort by score, highest first", text: "Each distinct score is a candidate threshold; lowering the threshold past a score flips that example to \"predicted positive\"." },
        { title: "Walk down the list", text: "Keep running counts: a positive adds to TP, a negative adds to FP. Handle tied scores together." },
        { title: "Emit points", text: "ROC point $(\\text{FPR}, \\text{TPR}) = (FP/N, TP/P)$, starting at $(0,0)$ (nothing predicted positive) and ending at $(1,1)$. PR point $(\\text{recall}, \\text{precision}) = (TP/P, TP/(TP+FP))$." },
        { title: "Summarize", text: "ROC AUC = area under the ROC points (trapezoids). Average precision AP = $\\sum_k (R_k - R_{k-1}) P_k$: the precision at each threshold, weighted by how much recall was gained there." },
      ] },
      { t: "h", text: "Choosing an operating threshold" },
      { t: "list", items: [
        "**Maximize a metric on validation** (F1, Fβ, Youden's $J = \\text{TPR} - \\text{FPR}$) and use that threshold on test/production.",
        "**Meet a constraint**: the lowest threshold with precision ≥ 0.95, or the highest with recall ≥ 0.99, read off the validation curves.",
        "**Minimize expected cost**: with calibrated $p$ and costs $c_{FP}$, $c_{FN}$, predict positive when $p\\,c_{FN} > (1-p)\\,c_{FP}$, i.e. $t^* = c_{FP}/(c_{FP} + c_{FN})$.",
      ] },
      { t: "h", text: "Calibration check and temperature scaling" },
      { t: "steps", items: [
        { title: "Bin by confidence", text: "Split predictions into $M$ bins by predicted probability (e.g. 10 equal-width bins)." },
        { title: "Compare", text: "In each bin, compare the mean predicted probability (confidence) with the observed frequency of positives (or the accuracy, for the predicted class). Plotting one against the other gives the **reliability diagram**; perfect calibration is the diagonal." },
        { title: "Summarize", text: "**ECE** = the average gap, weighted by bin size. Also report NLL and the Brier score, which don't depend on binning." },
        { title: "Fix (after training)", text: "**Temperature scaling**: divide every logit by one number $T > 0$, fitted by minimizing NLL on the validation set. $T > 1$ softens over-confident predictions. Ranking (AUC) and argmax predictions do not change." },
      ] },
    ],
    math: [
      { t: "h", text: "Threshold metrics" },
      { t: "p", text: "With $P = TP + FN$ actual positives and $N = FP + TN$ actual negatives:" },
      { t: "math", tex: String.raw`\text{accuracy} = \frac{TP + TN}{P + N},\quad \text{precision} = \frac{TP}{TP + FP},\quad \text{recall} = \text{TPR} = \frac{TP}{P},\quad \text{FPR} = \frac{FP}{N},\quad \text{specificity} = \frac{TN}{N}` },
      { t: "p", text: "The **F1 score** is the harmonic mean of precision $\\mathcal{P}$ and recall $\\mathcal{R}$. $F_\\beta$ weights recall $\\beta$ times as much as precision ($\\beta = 2$ favours recall, $\\beta = 0.5$ favours precision):" },
      { t: "math", tex: String.raw`F_1 = \frac{2\mathcal{P}\mathcal{R}}{\mathcal{P} + \mathcal{R}} = \frac{2\,TP}{2\,TP + FP + FN}, \qquad F_\beta = \frac{(1+\beta^2)\,\mathcal{P}\mathcal{R}}{\beta^2\mathcal{P} + \mathcal{R}}` },
      { t: "p", text: "The harmonic mean is pulled toward the smaller value: $\\mathcal{P} = 1, \\mathcal{R} = 0.01$ gives $F_1 \\approx 0.02$, while the ordinary average would be 0.505. Note that $F_1$ ignores TN completely." },
      { t: "h", text: "How the class prior enters precision" },
      { t: "p", text: "Let $\\pi = P/(P+N)$ be the **prevalence** (share of positives). By Bayes' rule, precision is" },
      { t: "math", tex: String.raw`\text{precision} = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi + \text{FPR}\cdot(1-\pi)}` },
      { t: "p", text: "TPR and FPR are rates *within* each class, so they — and the ROC curve — do not depend on $\\pi$. Precision does. Example: TPR $= 0.9$, FPR $= 0.05$. At $\\pi = 0.5$, precision $= 0.45/(0.45 + 0.025) = 0.947$. At $\\pi = 0.01$, precision $= 0.009/(0.009 + 0.0495) = 0.154$: **85% of alarms are false**, with the very same classifier." },
      { t: "h", text: "AUC is a ranking probability" },
      { t: "p", text: "Let $S^+$ and $S^-$ be the scores of a random positive and a random negative. The ROC curve plots $\\text{TPR}(t) = P(S^+ \\ge t)$ against $\\text{FPR}(t) = P(S^- \\ge t)$. Integrate TPR over FPR, with $dF$ the density of the negatives' scores:" },
      { t: "math", tex: String.raw`\text{AUC} = \int_0^1 \text{TPR}\, d\,\text{FPR} = \int P(S^+ > s)\, f_{S^-}(s)\, ds = P(S^+ > S^-)\ \ (+\tfrac12 P(S^+ = S^-))` },
      { t: "p", text: "On data this is the Mann–Whitney $U$ statistic divided by $n_+ n_-$: over all positive–negative pairs, count how often the positive wins (ties count ½). With ranks $r_i$ (1 = lowest score):" },
      { t: "math", tex: String.raw`\text{AUC} = \frac{\sum_{i:\,y_i=1} r_i - n_+(n_+ + 1)/2}{n_+\, n_-}` },
      { t: "p", text: "For two Gaussian score distributions with equal variance $\\sigma^2$ and means $d\\sigma$ apart, $S^+ - S^- \\sim \\mathcal{N}(d\\sigma, 2\\sigma^2)$, so $\\text{AUC} = \\Phi(d/\\sqrt2)$: $d = 1 \\Rightarrow 0.760$, $d = 2 \\Rightarrow 0.921$. A random scorer gives 0.5; an AUC below 0.5 means the scores are flipped." },
      { t: "h", text: "Average precision" },
      { t: "math", tex: String.raw`\text{AP} = \sum_{k} (R_k - R_{k-1})\, P_k`, caption: "$P_k, R_k$: precision and recall at the $k$-th distinct threshold (highest first), $R_0 = 0$. This is scikit-learn's definition; it avoids the optimistic straight-line interpolation of the PR curve." },
      { t: "p", text: "A random scorer has precision $\\pi$ at every recall, so its AP ≈ $\\pi$: **the PR baseline is the prevalence**, not 0.5. An AP of 0.3 is poor at $\\pi = 0.5$ and excellent at $\\pi = 0.001$." },
      { t: "h", text: "Worked example — ten scored examples" },
      { t: "p", text: "Scores (label): 0.95 (1), 0.85 (1), 0.80 (0), 0.70 (1), 0.55 (0), 0.45 (1), 0.40 (0), 0.30 (0), 0.20 (0), 0.10 (0). So $P = 4$, $N = 6$, $\\pi = 0.4$." },
      { t: "steps", items: [
        { title: "Confusion matrix at t = 0.5", text: "The top five scores are predicted positive.", tex: String.raw`TP = 3\ (0.95, 0.85, 0.70),\quad FP = 2\ (0.80, 0.55),\quad FN = 1\ (0.45),\quad TN = 4` },
        { title: "Threshold metrics", tex: String.raw`\text{acc} = \tfrac{7}{10} = 0.70,\quad \mathcal{P} = \tfrac{3}{5} = 0.60,\quad \mathcal{R} = \tfrac{3}{4} = 0.75,\quad \text{FPR} = \tfrac{2}{6} = 0.333` },
        { title: "F-scores", tex: String.raw`F_1 = \frac{2(0.6)(0.75)}{1.35} = 0.6667,\quad F_2 = \frac{5(0.45)}{4(0.6) + 0.75} = \frac{2.25}{3.15} = 0.7143,\quad F_{0.5} = \frac{1.25(0.45)}{0.25(0.6)+0.75} = 0.625` },
        { title: "ROC points (threshold lowered past each score)", tex: String.raw`(0,0)\to(0,\tfrac14)\to(0,\tfrac12)\to(\tfrac16,\tfrac12)\to(\tfrac16,\tfrac34)\to(\tfrac13,\tfrac34)\to(\tfrac13,1)\to(\tfrac12,1)\to(\tfrac23,1)\to(\tfrac56,1)\to(1,1)` },
        { title: "AUC by area", text: "Each horizontal segment adds width × height.", tex: String.raw`\text{AUC} = \tfrac16\cdot\tfrac12 + \tfrac16\cdot\tfrac34 + \tfrac23\cdot 1 = 0.0833 + 0.125 + 0.6667 = 0.875` },
        { title: "AUC by ranking", text: "Positives 0.95 and 0.85 beat all 6 negatives; 0.70 beats 5 (all but 0.80); 0.45 beats 4.", tex: String.raw`\text{AUC} = \frac{6 + 6 + 5 + 4}{4\times6} = \frac{21}{24} = 0.875\ \checkmark` },
        { title: "Average precision", text: "Recall rises by ¼ at each positive (ranks 1, 2, 4, 6), where precision is 1/1, 2/2, 3/4, 4/6.", tex: String.raw`\text{AP} = \tfrac14\left(1 + 1 + 0.75 + 0.6667\right) = 0.8542` },
      ] },
      { t: "p", text: "All values match scikit-learn's `roc_auc_score` (0.875) and `average_precision_score` (0.8542)." },
      { t: "h", text: "Multi-class averaging" },
      { t: "p", text: "With $C$ classes, compute one-vs-rest precision/recall/F1 for each class $c$, then average:" },
      { t: "list", items: [
        "**Macro**: plain mean over classes (every class counts equally).",
        "**Weighted**: mean weighted by each class's size (support) $n_c$.",
        "**Micro**: pool all TP, FP, FN over classes first. For single-label problems, micro-precision = micro-recall = micro-F1 = accuracy.",
      ] },
      { t: "p", text: "Confusion matrix (rows = true A, B, C; columns = predicted A, B, C): A = (50, 3, 2), B = (5, 30, 5), C = (1, 4, 10); supports 55, 40, 15; 110 examples." },
      { t: "table", head: ["class", "precision = TP / column sum", "recall = TP / row sum", "F1"], rows: [
        ["A", "50/56 = 0.8929", "50/55 = 0.9091", "0.9009"],
        ["B", "30/37 = 0.8108", "30/40 = 0.7500", "0.7792"],
        ["C", "10/17 = 0.5882", "10/15 = 0.6667", "0.6250"],
      ] },
      { t: "math", tex: String.raw`\text{macro-}F_1 = \tfrac13(0.9009 + 0.7792 + 0.6250) = 0.7684,\quad \text{weighted-}F_1 = \tfrac{55(0.9009) + 40(0.7792) + 15(0.6250)}{110} = 0.8190,\quad \text{micro-}F_1 = \tfrac{90}{110} = 0.8182` },
      { t: "p", text: "Macro-F1 is lowest because it gives the small, hard class C a full third of the weight — exactly what you want when minority classes matter." },
      { t: "h", text: "Calibration: ECE and temperature scaling" },
      { t: "p", text: "Split predictions into bins $B_1..B_M$ by confidence. In bin $m$, $\\text{conf}(B_m)$ is the mean predicted probability and $\\text{acc}(B_m)$ the observed frequency (fraction correct, or fraction positive for a binary $P(y=1)$ forecast); $n$ is the total number of predictions:" },
      { t: "math", tex: String.raw`\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\big|\text{acc}(B_m) - \text{conf}(B_m)\big|` },
      { t: "p", text: "Example with 10 predictions and bins (0.5, 0.7], (0.7, 0.9], (0.9, 1]: the first bin has 4 predictions with mean confidence 0.625 and 2 correct (0.5); the second has 3 with confidence 0.8167 and 2 correct (0.6667); the third has 3 with confidence 0.9467 and 2 correct (0.6667). ECE $= 0.4(0.125) + 0.3(0.15) + 0.3(0.28) = 0.05 + 0.045 + 0.084 = 0.179$." },
      { t: "p", text: "Temperature scaling replaces $p = \\sigma(z)$ with $\\sigma(z/T)$ (softmax$(z/T)$ for multi-class) and fits the single number $T$ on validation data:" },
      { t: "math", tex: String.raw`T^* = \arg\min_{T>0}\; -\frac{1}{n_{\text{val}}}\sum_{i} \Big[y_i \log\sigma(z_i/T) + (1-y_i)\log\big(1-\sigma(z_i/T)\big)\Big]` },
      { t: "p", text: "Dividing by a positive constant keeps the score order, so the ranking (AUC, AP) and each logit's sign (the decision at $p = 0.5$) are untouched — only confidence changes. In the code below, a model whose logits were inflated 3× gets $T^* = 3.235$, close to the true factor." },
    ],
    visual: [
      { t: "p", text: "The simulator draws scores for 1000 examples from two overlapping Gaussians and computes every metric live from the confusion matrix at threshold $t$." },
      { t: "sim", id: "roc-pr", caption: "Drag on the histogram (or use the threshold slider) to move t; Play sweeps t from strict to lenient and traces both curves." },
      { t: "list", items: [
        "Press **Play** from the start. At $t = 4$ nothing is predicted positive: ROC sits at (0, 0) and precision is undefined (0/0). As $t$ falls, the gold point climbs the ROC curve and slides right along the PR curve, ending at (1, 1) and precision = prevalence.",
        "Set separation $d = 0$: the ROC curve hugs the diagonal (AUC ≈ 0.5) and the PR curve hugs the dashed prevalence line. Then try $d = 1$ and $d = 2$ and compare the AUC with $\\Phi(d/\\sqrt2) = 0.760$ and $0.921$ — the sample values are close.",
        "Keep $d = 2$ and drag **positives %** from 50 down to 2. ROC AUC barely moves, but AP drops sharply and the PR curve sags: with rare positives, the same FPR produces far more false alarms than true hits.",
        "At 5% positives, find the threshold with recall ≈ 0.8 and read its precision and FPR: a small FPR still means dozens of false positives.",
        "Watch the gold *best F1 threshold* as you change the imbalance: it moves higher as positives get rarer. A fixed 0.5 cut-off is rarely the best choice.",
        "Press **Resample** a few times at 2% positives: AUC and especially AP jump around with only 20 positives — metrics on small classes have large sampling error.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Everything from scratch in NumPy — confusion matrix, precision/recall/Fβ, ROC points, AUC by trapezoids and by ranks, average precision, ECE, NLL and temperature scaling — on an imbalanced problem (10% positives), with every number checked against scikit-learn." },
      { t: "code", lang: "python", title: "classification_metrics_from_scratch.py", code: CLS_NUMPY, output: CLS_NUMPY_OUT },
      { t: "list", items: [
        "**Accuracy misleads**: 0.920 sounds good until you see the always-negative baseline is 0.903. At $p \\ge 0.5$ the model finds only 27 of 97 positives (recall 0.278).",
        "**AUC is threshold-free**: 0.8945 by trapezoids, by ranks and by scikit-learn — identical, as the derivation promised. AP (0.599) is far lower because positives are rare (baseline AP ≈ 0.1).",
        "**Threshold choice matters**: picking the F1-optimal threshold **on validation** (0.173) raises test recall from 0.278 to 0.701 and F1 from 0.403 to 0.500, at the cost of precision. The test set was not used to pick it.",
        "**Temperature scaling**: inflating logits 3× worsens ECE (0.020 → 0.063) and NLL (0.206 → 0.335) without changing the AUC. Fitting $T$ on validation recovers $T = 3.235$ and brings ECE back to ≈ 0.025 and NLL to ≈ 0.209.",
        "`roc_points` handles **ties** by emitting one point per distinct score (`np.diff(s)` finds where runs of equal scores end), which makes the trapezoid AUC match the tie-corrected rank formula.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "An MLP trained far too long on 600 examples memorizes them (train accuracy 1.0) and becomes over-confident. The program measures ranking and threshold metrics with scikit-learn, computes ECE in PyTorch, fits a temperature with L-BFGS on validation logits, and tries `pos_weight` against the class imbalance." },
          { t: "code", lang: "python", title: "calibration_pytorch.py", code: CLS_TORCH, output: CLS_TORCH_OUT },
          { t: "list", items: [
            "Mean confidence 0.970 versus test accuracy 0.807 is the classic over-confidence gap. One parameter, $T = 7.6$, cuts test NLL from 1.550 to 0.429 and ECE from 0.165 to 0.057, while AUC and the 0.5-threshold predictions stay identical.",
            "Fitting $\\log T$ instead of $T$ keeps $T$ positive without constraints; L-BFGS converges in a few iterations because the objective is one-dimensional and smooth.",
            "`pos_weight` multiplies the loss of positive examples, which shifts the decision boundary toward predicting positive: recall rises (0.440 → 0.471) and precision falls (0.540 → 0.515). Reweighting mostly moves the operating point; choosing the threshold on validation does much the same without retraining.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras has streaming metric objects that add up statistics over batches. `AUC(curve=\"PR\")` approximates the PR area using a fixed grid of thresholds (200 by default), so it can differ slightly from scikit-learn's exact AP." },
          { t: "code", lang: "python", title: "metrics_keras.py (not executed in CI)", code: CLS_KERAS, output: "Prints a dict with the test loss, roc_auc, pr_auc, precision and recall of the trained model; the exact values depend on the TensorFlow version." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Metric", "Measures", "Pros", "Cons / when to avoid"], rows: [
        ["Accuracy", "Fraction correct at one threshold", "Simple; fine for balanced classes with equal costs", "Misleading under imbalance; ignores error types"],
        ["Precision", "P(actual + | predicted +)", "Captures the cost of false alarms (spam, alerts to humans)", "Undefined when nothing is predicted positive; depends on prevalence"],
        ["Recall / TPR / sensitivity", "P(predicted + | actual +)", "Captures the cost of misses (screening, safety)", "Reaches 1 just by predicting everything positive"],
        ["Specificity / TNR", "P(predicted − | actual −)", "Medical tests reported as sensitivity + specificity", "Tells little when negatives are abundant"],
        ["F1 / Fβ", "Harmonic mean of P and R", "One number balancing both; β sets the preference", "Ignores TN; depends on the threshold; not a cost model"],
        ["Balanced accuracy", "Mean of per-class recall", "Robust to imbalance", "Hides precision"],
        ["Matthews correlation (MCC)", "Correlation between prediction and label, using all four counts", "Informative under imbalance; symmetric", "Less intuitive"],
        ["ROC AUC", "P(positive scored above negative)", "Ranking quality independent of threshold and prevalence", "Optimistic with rare positives; ignores calibration"],
        ["PR AUC / AP", "Precision averaged over recall", "Focuses on the positive class; sensitive to imbalance", "Baseline = prevalence, so not comparable across datasets"],
        ["Log loss (NLL) / Brier", "Quality of the probabilities", "Proper scoring rules: reward calibration and sharpness", "Harder to interpret; NLL explodes on confident errors"],
        ["ECE / reliability diagram", "Calibration gap", "Clear picture of over/under-confidence", "Depends on binning; a constant predictor can game it"],
        ["Top-k accuracy", "True class among the k highest scores", "Large label spaces (ImageNet top-5)", "Lenient; not a decision metric"],
      ] },
      { t: "p", text: "Calibration methods: **temperature scaling** (one parameter, keeps the ranking); **Platt scaling** (logistic regression on the logit, $\\sigma(az + b)$; Platt, 1999); **isotonic regression** (a flexible, always-increasing mapping that needs more validation data; Niculescu-Mizil and Caruana, 2005, *Predicting Good Probabilities with Supervised Learning*); and vector/matrix scaling for multi-class." },
    ],
    applications: [
      { t: "list", items: [
        "**Medical screening and diagnostics** report sensitivity and specificity (or the ROC curve) and choose operating points that keep missed cases below a target.",
        "**Fraud and anomaly detection**: positives are a tiny fraction, so teams track precision at a fixed alert budget, recall at a fixed precision, and PR AUC.",
        "**Information retrieval and search** use precision@k, recall@k and average precision. **Object detection** averages AP over classes and IoU thresholds (mAP — see [Object Detection](/learn/m12/object-detection)).",
        "**Content moderation and spam filtering** set thresholds per policy: high-precision automatic removal, a lower threshold for human review.",
        "**Weather forecasting** pioneered calibration analysis (reliability diagrams, the Brier score). Risk scores in credit and medicine need calibrated probabilities because decisions use the numbers directly.",
        "**Selective prediction**: calibrated confidence lets a system abstain or hand over to a human when it is unsure.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Single-threshold metrics** match real use but depend on an arbitrary cut-off; **curve summaries** (AUC, AP) compare models regardless of threshold but can hide poor performance where you actually operate.",
        "**ROC vs PR**: ROC ignores class balance — good for comparing classifiers across datasets, bad for predicting alarm volume. PR reflects the positive class's experience (Saito and Rehmsmeier, 2015, recommend PR plots for imbalanced data).",
        "**Partial AUC** (e.g. FPR ≤ 0.01) focuses on the clinically or operationally relevant region, at the cost of a noisier estimate.",
        "**Calibration is separate from discrimination** (telling classes apart): temperature scaling never improves AUC, and a perfectly calibrated model can still rank poorly. Report both.",
        "**ECE is handy but depends on binning**; NLL and Brier score are proper scoring rules with no settings, but harder to read.",
        "**Metrics on small classes are noisy**: with 20 positives, one example moves recall by 0.05. Report confidence intervals (bootstrap the test set).",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Reporting accuracy on imbalanced data.** Always compare with the majority-class baseline and report per-class recall/precision or PR AUC.",
        "**Choosing the threshold on the test set.** Optimizing F1 on test makes the reported F1 optimistic. Tune thresholds (and temperatures) on validation, then evaluate once.",
        "**Computing ROC/PR from hard 0/1 predictions.** You get a single point and a meaningless \"AUC\". Pass scores or probabilities.",
        "**Comparing AP across datasets with different prevalence.** AP's baseline is the prevalence; normalize or compare against that baseline.",
        "**Drawing straight lines between PR points.** Precision does not change linearly with recall between thresholds, so trapezoidal PR areas are optimistic (Davis and Goadrich, 2006). Use step-wise AP.",
        "**Assuming softmax outputs are probabilities.** Modern networks are often over-confident (Guo et al., 2017). Check a reliability diagram and calibrate on validation data.",
        "**Mixing up macro and micro.** For single-label multi-class, micro-F1 equals accuracy and hides minority-class failures. Say which average you report.",
        "**Treating precision = 0 as a measurement when nothing is predicted positive.** It is undefined (0/0); libraries put in 0 with a warning. Handle it explicitly.",
        "**Oversampling or reweighting, then reading the probabilities as calibrated.** Changing the training class balance shifts the predicted probabilities. Recalibrate on data with the deployment class balance.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Define precision and recall, and give a situation where each matters most.", a: "Precision = TP/(TP+FP): of everything flagged positive, the fraction that truly is. Recall = TP/(TP+FN): of all real positives, the fraction caught. Precision matters when false alarms are costly (a spam filter hiding real email, automated account bans); recall when misses are costly (cancer screening, detecting safety failures)." },
      { t: "qa", level: "beginner", q: "Why is accuracy a poor metric for a fraud dataset with 0.5% fraud?", a: "Predicting \"not fraud\" for everything is 99.5% accurate yet catches nothing — accuracy is dominated by the majority class. Use recall/precision for the fraud class, F1/Fβ, PR AUC, or cost-weighted metrics, and always compare with the majority baseline." },
      { t: "qa", level: "beginner", q: "What does an ROC curve plot, and what do its corners mean?", a: "True-positive rate against false-positive rate for every threshold. (0,0): nothing predicted positive. (1,1): everything predicted positive. (0,1): perfect classification. The diagonal is random guessing." },
      { t: "qa", level: "intermediate", q: "What is the probabilistic interpretation of ROC AUC?", a: "AUC = P(score of a random positive > score of a random negative), with ties counting ½. It equals the Mann–Whitney U statistic divided by $n_+n_-$. So AUC depends only on the ranking of scores: any strictly increasing transformation (including temperature scaling) leaves it unchanged." },
      { t: "qa", level: "intermediate", q: "When would you prefer a PR curve over an ROC curve?", a: "When positives are rare and you care about the positive class. TPR and FPR are normalized within each class, so a tiny FPR can still mean many more false positives than true positives — the ROC curve looks excellent while precision is poor. The PR curve exposes this; its baseline is the prevalence." },
      { t: "qa", level: "intermediate", q: "Explain macro, micro and weighted F1.", a: "Compute per-class (one-vs-rest) precision, recall and F1. Macro-F1 averages per-class F1 equally, so small classes count as much as large ones. Weighted-F1 weights by class support. Micro-F1 pools TP/FP/FN across classes before computing F1; for single-label multi-class problems it equals accuracy. Report macro when minority classes matter." },
      { t: "qa", level: "intermediate", q: "How do you choose a decision threshold?", a: "On validation data, by goal: maximize F1/Fβ or Youden's J; meet a constraint (precision ≥ X, recall ≥ Y, alert budget); or minimize expected cost. With calibrated probabilities and costs $c_{FP}, c_{FN}$, the cost-optimal threshold is $t^* = c_{FP}/(c_{FP} + c_{FN})$. Then freeze it and evaluate on test." },
      { t: "qa", level: "advanced", q: "What is temperature scaling, why does it work, and what can it not fix?", a: "It divides logits by a scalar $T$ fitted by minimizing NLL on validation data. Over-trained networks produce logits that are too large (over-confident), and $T > 1$ shrinks them. Because $z \\mapsto z/T$ keeps the order, accuracy and AUC don't change — it fixes only calibration. It cannot fix poor ranking, nor calibration that differs across regions of input space (one $T$ is global), nor distribution shift between validation data and deployment." },
      { t: "qa", level: "advanced", q: "Derive how precision depends on class prevalence, and what it implies for deploying a model trained on balanced data.", a: "By Bayes, precision $= \\text{TPR}\\,\\pi / (\\text{TPR}\\,\\pi + \\text{FPR}(1-\\pi))$. TPR and FPR belong to the classifier and threshold; $\\pi$ belongs to the population. A model evaluated on balanced data (π = 0.5) with TPR 0.9 and FPR 0.05 has precision 0.947, but deployed where π = 0.01 its precision is 0.154. Re-estimate precision at the deployment prevalence, and adjust predicted probabilities before thresholding (prior shift: multiply the odds by the ratio of deployment to training prior odds)." },
      { t: "qa", level: "advanced", q: "Why can ECE be misleading, and what would you report alongside it?", a: "ECE depends on the number and type of bins (equal-width bins can be nearly empty at low confidence). It checks only the average gap per bin, so a model that predicts the base rate for everyone is perfectly calibrated with zero discrimination. With few samples it is biased upward. Report NLL and Brier score (proper scoring rules), a reliability diagram with bin counts, and a discrimination metric (AUC)." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "TP = 40, FP = 10, FN = 20, TN = 930. What is the precision?", answer: 0.8, tolerance: 0.001, explain: "$40/(40+10) = 0.8$." },
        { kind: "numeric", q: "Same confusion matrix (TP 40, FP 10, FN 20, TN 930). What is the F1 score? (3 decimals)", answer: 0.727, tolerance: 0.001, explain: "Recall $= 40/60 = 0.6667$; $F_1 = 2\\,TP/(2\\,TP + FP + FN) = 80/110 = 0.727$." },
        { kind: "numeric", q: "A classifier has TPR 0.9 and FPR 0.05. If 1% of the population is positive, what is its precision? (3 decimals)", answer: 0.154, tolerance: 0.001, explain: "$0.9\\times0.01/(0.9\\times0.01 + 0.05\\times0.99) = 0.009/0.0585 = 0.154$." },
        { q: "You multiply all of a model's logits by 0.5. Which metric can change?", options: ["ROC AUC", "Average precision", "Expected calibration error", "Accuracy at the threshold p = 0.5"], answer: 2, explain: "Scaling by a positive constant keeps the ranking (AUC, AP) and each logit's sign (decisions at p = 0.5), but changes the probabilities and so the calibration." },
        { kind: "numeric", q: "Positives scored 0.9 and 0.4; negatives scored 0.6, 0.3 and 0.2. What is the ROC AUC? (3 decimals)", answer: 0.833, tolerance: 0.001, explain: "Pairs the positive wins: 0.9 beats all 3, 0.4 beats 0.3 and 0.2 → 5 of 6 pairs = 0.833." },
        { q: "What is the average precision of a classifier that assigns random scores, on data with 2% positives?", options: ["About 0.5", "About 0.02", "Exactly 0", "About 0.98"], answer: 1, explain: "Random scores give precision ≈ prevalence at every recall, so AP ≈ 0.02 — the PR baseline is the prevalence." },
        { kind: "numeric", q: "A false negative costs 9 times as much as a false positive. With calibrated probabilities, what threshold minimizes expected cost? (2 decimals)", answer: 0.1, tolerance: 0.005, explain: "$t^* = c_{FP}/(c_{FP} + c_{FN}) = 1/(1+9) = 0.1$." },
        { q: "For a single-label 5-class problem, micro-averaged F1 equals…", options: ["Macro-F1", "Accuracy", "Weighted-F1", "Balanced accuracy"], answer: 1, explain: "Every error is one FP (for the predicted class) and one FN (for the true class), so pooled precision = pooled recall = accuracy." },
      ] },
      { t: "exercise", title: "Compute the metrics by hand", prompt: "Scores (label): 0.9 (1), 0.8 (0), 0.7 (1), 0.6 (1), 0.4 (0), 0.3 (1), 0.2 (0), 0.1 (0). (a) Confusion matrix, precision, recall and F1 at $t = 0.5$. (b) ROC AUC using the ranking interpretation. (c) Average precision.", hint: "There are 4 positives and 4 negatives; in rank order the labels are 1, 0, 1, 1, 0, 1, 0, 0.", solution: [
        { t: "steps", items: [
          { title: "(a) t = 0.5", text: "Predicted positive: 0.9, 0.8, 0.7, 0.6.", tex: String.raw`TP = 3,\ FP = 1,\ FN = 1,\ TN = 3;\quad \mathcal{P} = 0.75,\ \mathcal{R} = 0.75,\ F_1 = 0.75` },
          { title: "(b) AUC", text: "0.9 beats 4 negatives; 0.7 and 0.6 beat 3 each (not 0.8); 0.3 beats 2 (0.2, 0.1).", tex: String.raw`\text{AUC} = \frac{4 + 3 + 3 + 2}{16} = \frac{12}{16} = 0.75` },
          { title: "(c) AP", text: "Positives at ranks 1, 3, 4, 6, with precisions 1/1, 2/3, 3/4, 4/6.", tex: String.raw`\text{AP} = \tfrac14(1 + 0.6667 + 0.75 + 0.6667) = 0.7708` },
        ] },
        { t: "code", lang: "python", title: "check_by_hand.py", code: String.raw`import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

s = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
print(f1_score(y, s >= 0.5), roc_auc_score(y, s), round(average_precision_score(y, s), 4))`, output: "0.75 0.75 0.7708" },
      ] },
      { t: "exercise", title: "Implement a reliability diagram and Platt scaling", prompt: "Extend `classification_metrics_from_scratch.py`: (1) write `reliability(prob, y, bins=10)` that returns per-bin (count, mean confidence, observed frequency), and print it for the over-confident model; (2) implement **Platt scaling** $\\sigma(a z + b)$ by gradient descent on validation NLL, and compare its test ECE/NLL with temperature scaling.", hint: "Platt scaling is logistic regression with one input — the logit $z$. Temperature scaling is the special case $a = 1/T$, $b = 0$.", solution: [
        { t: "code", lang: "python", code: String.raw`def reliability(prob, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for k in range(bins):
        m = which == k
        if m.any():
            rows.append((int(m.sum()), prob[m].mean(), y[m].mean()))
    return rows

for count, conf, freq in reliability(sigmoid(over[te]), yt):
    print(f"n={count:4d}  confidence {conf:.3f}  frequency {freq:.3f}")

a, b = 1.0, 0.0                                   # Platt scaling: p = sigmoid(a*z + b)
zv, yv = over[va], y[va]
for _ in range(5000):
    p = sigmoid(a * zv + b)
    a -= 0.05 * np.mean((p - yv) * zv)            # gradient of mean NLL wrt a
    b -= 0.05 * np.mean(p - yv)                   # ... and wrt b
p_platt = sigmoid(a * over[te] + b)
print(f"Platt a={a:.3f} b={b:.3f}  test ECE {ece(p_platt, yt):.4f}  NLL {nll(p_platt, yt):.4f}")` },
        { t: "p", text: "For the over-confident model, the reliability table shows confidence above frequency in high bins and below it in low bins (predictions pushed to the extremes). Platt scaling learns $a \\approx 1/T$ plus a bias; with plenty of validation data it matches or slightly beats temperature scaling, because the extra bias can also correct a shift in the base rate." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "The confusion matrix counts TP, FP, FN, TN at a threshold; every threshold metric is a ratio of them.",
        "Precision $= TP/(TP+FP)$, recall $= TP/(TP+FN)$, FPR $= FP/(FP+TN)$, $F_1 = 2TP/(2TP+FP+FN)$, $F_\\beta = (1+\\beta^2)\\mathcal{P}\\mathcal{R}/(\\beta^2\\mathcal{P}+\\mathcal{R})$.",
        "Precision depends on prevalence: $\\text{TPR}\\,\\pi/(\\text{TPR}\\,\\pi + \\text{FPR}(1-\\pi))$. TPR, FPR and the ROC curve do not.",
        "ROC AUC $= P(S^+ > S^-)$ = Mann–Whitney U$/(n_+n_-)$; unchanged by any increasing transformation; 0.5 = random.",
        "AP $= \\sum_k (R_k - R_{k-1})P_k$; baseline = prevalence. Prefer PR/AP when positives are rare.",
        "Macro weights classes equally, weighted uses support, micro pools counts (= accuracy for single-label).",
        "Choose thresholds on validation: maximize an F-score, meet a constraint, or use $t^* = c_{FP}/(c_{FP}+c_{FN})$.",
        "Calibration: reliability diagram, ECE $= \\sum_m \\frac{|B_m|}{n}|\\text{acc}-\\text{conf}|$, NLL, Brier. Temperature scaling $\\sigma(z/T)$ fixes over-confidence without changing the ranking.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Regression Metrics](/learn/m17/regression-metrics) — the matching toolbox for continuous targets.",
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — once you have a metric, optimize it in a systematic, trustworthy way.",
        "[Object Detection](/learn/m12/object-detection) — see how AP is extended to mAP over classes and IoU thresholds.",
        "[Robustness and Safety](/learn/m19/robustness-safety) — calibration under distribution shift and adversarial inputs.",
      ] },
    ],
  },
};

export default lesson;
