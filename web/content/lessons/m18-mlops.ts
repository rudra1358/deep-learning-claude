import type { Lesson } from "@/lib/types";

const DRIFT_CODE = String.raw`import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
EPS = 1e-4                                 # floor for empty bins so logs stay finite


def bin_edges(reference, n_bins=10):
    """Quantile bins of the REFERENCE data: each bin holds ~10% of the training distribution."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # catch production values outside the training range
    return edges


def proportions(x, edges):
    counts = np.histogram(x, bins=edges)[0]
    return np.maximum(counts / counts.sum(), EPS)


def psi(reference, current, n_bins=10):
    """Population Stability Index = sum (c - r) * ln(c / r) over bins (a symmetrised KL)."""
    e = bin_edges(reference, n_bins)
    r, c = proportions(reference, e), proportions(current, e)
    return np.sum((c - r) * np.log(c / r))


def kl(reference, current, n_bins=10):
    """KL(current || reference) on the same bins, in nats."""
    e = bin_edges(reference, n_bins)
    r, c = proportions(reference, e), proportions(current, e)
    return np.sum(c * np.log(c / r))


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov D = max |F_a(t) - F_b(t)| over all t."""
    grid = np.sort(np.concatenate([a, b]))
    Fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    Fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return np.max(np.abs(Fa - Fb))


# ---------------------------------------------------------------- 1. data drift on one feature
reference = rng.normal(0, 1, 5000)                     # feature values seen at training time
scenarios = {
    "no drift (fresh sample)": rng.normal(0, 1, 2000),
    "mean shift +0.25 sd": rng.normal(0.25, 1, 2000),
    "mean shift +1 sd": rng.normal(1.0, 1, 2000),
    "variance x2.25 (sd 1.5)": rng.normal(0, 1.5, 2000),
    "10% missing -> filled with 0": np.where(rng.random(2000) < 0.1, 0.0, rng.normal(0, 1, 2000)),
}
print(f"{'scenario':30s}   PSI      KL     KS D   KS D (scipy)  p-value")
for name, cur in scenarios.items():
    res = stats.ks_2samp(reference, cur)
    print(f"{name:30s} {psi(reference, cur):6.3f}  {kl(reference, cur):6.3f}  {ks_statistic(reference, cur):6.3f}"
          f"     {res.statistic:6.3f}     {res.pvalue:9.2e}")

# ---------------------------------------------------------------- 2. huge samples make tiny drift "significant"
small_shift = 0.03
for n in (1_000, 100_000):
    cur = rng.normal(small_shift, 1, n)
    ref = rng.normal(0, 1, n)
    print(f"shift 0.03 sd, n = {n:6d}:  KS p-value {stats.ks_2samp(ref, cur).pvalue:.2e}   PSI {psi(ref, cur):.4f}")

# ---------------------------------------------------------------- 3. concept drift: P(x) fixed, P(y|x) changes
# A classifier learned the rule y = [x1 + x2 > 0]. In production the inputs keep the SAME
# distribution, but the world changes and the true rule becomes y = [x1 - 0.5*x2 > 0.3].
X_train = rng.normal(0, 1, (5000, 2))
w_model = np.array([1.0, 1.0])                         # the deployed model (perfect on old data)

X_prod = rng.normal(0, 1, (5000, 2))
y_old = (X_prod @ np.array([1.0, 1.0]) > 0)
y_new = (X_prod @ np.array([1.0, -0.5]) > 0.3)
pred = X_prod @ w_model > 0
print("\nconcept drift:")
print(f"  input PSI, feature 1: {psi(X_train[:, 0], X_prod[:, 0]):.4f}   feature 2: {psi(X_train[:, 1], X_prod[:, 1]):.4f}")
print(f"  accuracy against the old rule: {np.mean(pred == y_old):.3f}")
print(f"  accuracy against the new rule: {np.mean(pred == y_new):.3f}   <- only visible once labels arrive")
print(f"  share of positive predictions (a label-free proxy): {pred.mean():.3f}  vs true positive rate now {y_new.mean():.3f}")`;
const DRIFT_OUT = String.raw`scenario                         PSI      KL     KS D   KS D (scipy)  p-value
no drift (fresh sample)         0.007   0.003   0.028      0.028      2.00e-01
mean shift +0.25 sd             0.093   0.046   0.126      0.126      3.82e-20
mean shift +1 sd                0.939   0.453   0.384      0.384     2.37e-188
variance x2.25 (sd 1.5)         0.194   0.102   0.108      0.108      8.11e-15
10% missing -> filled with 0    0.104   0.057   0.062      0.062      3.51e-05
shift 0.03 sd, n =   1000:  KS p-value 7.95e-01   PSI 0.0213
shift 0.03 sd, n = 100000:  KS p-value 3.42e-06   PSI 0.0008

concept drift:
  input PSI, feature 1: 0.0055   feature 2: 0.0031
  accuracy against the old rule: 1.000
  accuracy against the new rule: 0.599   <- only visible once labels arrive
  share of positive predictions (a label-free proxy): 0.502  vs true positive rate now 0.393`;
const MON_CODE = String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.set_num_threads(1)
rng = np.random.default_rng(0)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
Xt, yt = torch.tensor(X_tr / 16, dtype=torch.float32), torch.tensor(y_tr)
for _ in range(150):
    opt.zero_grad()
    nn.functional.cross_entropy(model(Xt), yt).backward()
    opt.step()
model.eval()


@torch.no_grad()
def serve(batch):
    """What the production service computes and LOGS for every request (no labels needed)."""
    p = torch.softmax(model(torch.tensor(batch / 16, dtype=torch.float32)), dim=1)
    conf, pred = p.max(dim=1)
    return pred.numpy(), conf.numpy()


def psi_from_props(r, c, eps=1e-4):
    r, c = np.maximum(r, eps), np.maximum(c, eps)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(ref, cur, n_bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return psi_from_props(r, c)


# Reference window: the validation half of the held-out data, captured at deployment time.
X_ref, y_ref = X_te[:360], y_te[:360]
ref_pred, ref_conf = serve(X_ref)
ref_brightness = X_ref.mean(axis=1)                  # one cheap summary feature per image
ref_classes = np.bincount(ref_pred, minlength=10) / len(ref_pred)

X_new, y_new = X_te[360:], y_te[360:]                # "production" traffic comes from here
windows = {
    "week 1: normal traffic": (X_new, y_new),
    "week 2: dimmer scanner (x0.6)": (np.round(X_new * 0.6), y_new),
    "week 3: sensor noise (sd 3)": (np.clip(X_new + rng.normal(0, 3, X_new.shape), 0, 16), y_new),
    "week 4: only digits 0-4": (X_new[y_new < 5], y_new[y_new < 5]),
}
print(f"{'window':30s}  PSI(brightness)  PSI(confidence)  PSI(pred. classes)  mean conf  accuracy*")
print(f"{'reference':30s}  {'-':>15s}  {'-':>15s}  {'-':>18s}  {ref_conf.mean():9.3f}  {np.mean(ref_pred == y_ref):8.3f}")
for name, (xb, yb) in windows.items():
    pred, conf = serve(xb)
    cls = np.bincount(pred, minlength=10) / len(pred)
    alarm = "  <- ALERT" if max(psi(ref_brightness, xb.mean(1)), psi(ref_conf, conf), psi_from_props(ref_classes, cls)) > 0.2 else ""
    print(f"{name:30s}  {psi(ref_brightness, xb.mean(1)):15.3f}  {psi(ref_conf, conf):15.3f}  "
          f"{psi_from_props(ref_classes, cls):18.3f}  {conf.mean():9.3f}  {np.mean(pred == yb):8.3f}{alarm}")
print("* accuracy needs ground-truth labels, which in production arrive late or never")`;
const MON_OUT = String.raw`window                          PSI(brightness)  PSI(confidence)  PSI(pred. classes)  mean conf  accuracy*
reference                                     -                -                   -      0.971     0.969
week 1: normal traffic                    0.066            0.011               0.073      0.968     0.978
week 2: dimmer scanner (x0.6)             8.393            1.852               0.052      0.925     0.967  <- ALERT
week 3: sensor noise (sd 3)               0.781            0.211               0.063      0.941     0.922  <- ALERT
week 4: only digits 0-4                   0.146            0.089               2.906      0.972     0.973  <- ALERT
* accuracy needs ground-truth labels, which in production arrive late or never`;
const REG_CODE = String.raw`import hashlib
import json
import os
import tempfile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class FileRegistry:
    """A minimal experiment tracker + model registry backed by plain files.

    root/
      runs/<run_id>.json                 params, metrics, data fingerprint, artifact hash
      artifacts/<sha256>.npz             content-addressed model weights (immutable)
      registry.json                      {model_name: {"versions": [...], "stages": {stage: version}}}
    """

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "runs"), exist_ok=True)
        os.makedirs(os.path.join(root, "artifacts"), exist_ok=True)
        self.reg_path = os.path.join(root, "registry.json")
        if not os.path.exists(self.reg_path):
            self._write_registry({})

    def _read_registry(self):
        with open(self.reg_path) as f:
            return json.load(f)

    def _write_registry(self, reg):
        tmp = self.reg_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
        os.replace(tmp, self.reg_path)            # atomic rename: readers never see half a file

    def log_run(self, params, metrics, arrays, data_fingerprint):
        blob = b"".join(arrays[k].tobytes() for k in sorted(arrays))
        digest = hashlib.sha256(blob).hexdigest()
        np.savez(os.path.join(self.root, "artifacts", digest + ".npz"), **arrays)
        run_id = f"run-{len(os.listdir(os.path.join(self.root, 'runs'))) + 1:03d}"
        record = {"run_id": run_id, "params": params, "metrics": metrics,
                  "data": data_fingerprint, "artifact": digest}
        with open(os.path.join(self.root, "runs", run_id + ".json"), "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        return run_id

    def runs(self):
        out = []
        for name in sorted(os.listdir(os.path.join(self.root, "runs"))):
            with open(os.path.join(self.root, "runs", name)) as f:
                out.append(json.load(f))
        return out

    def register(self, model_name, run_id):
        reg = self._read_registry()
        entry = reg.setdefault(model_name, {"versions": [], "stages": {}})
        version = len(entry["versions"]) + 1
        entry["versions"].append({"version": version, "run_id": run_id})
        self._write_registry(reg)
        return version

    def promote(self, model_name, version, stage):
        reg = self._read_registry()
        reg[model_name]["stages"][stage] = version
        self._write_registry(reg)

    def load(self, model_name, stage):
        reg = self._read_registry()
        version = reg[model_name]["stages"][stage]
        run_id = reg[model_name]["versions"][version - 1]["run_id"]
        run = next(r for r in self.runs() if r["run_id"] == run_id)
        with np.load(os.path.join(self.root, "artifacts", run["artifact"] + ".npz")) as z:
            arrays = {k: z[k] for k in z.files}
        check = hashlib.sha256(b"".join(arrays[k].tobytes() for k in sorted(arrays))).hexdigest()
        assert check == run["artifact"], "artifact corrupted"
        return version, arrays


# ---------------------------------------------------------------- use it
X, y = load_digits(return_X_y=True)
X_tr, X_rest, y_tr, y_rest = train_test_split(X / 16.0, y, test_size=0.4, stratify=y, random_state=0)
# validation set: used to choose C.  gate set: used ONLY for the promotion decision.
X_val, X_gate, y_val, y_gate = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)
fingerprint = {"dataset": "sklearn load_digits", "n_train": len(X_tr),
               "sha256": hashlib.sha256(X_tr.tobytes() + y_tr.tobytes()).hexdigest()[:12]}

reg = FileRegistry(tempfile.mkdtemp())
for C in (0.001, 0.01, 0.1, 1.0):                         # a tiny hyperparameter sweep, every run logged
    clf = LogisticRegression(C=C, max_iter=2000).fit(X_tr, y_tr)
    reg.log_run(params={"model": "logreg", "C": C},
                metrics={"val_accuracy": round(clf.score(X_val, y_val), 4)},
                arrays={"coef": clf.coef_, "intercept": clf.intercept_},
                data_fingerprint=fingerprint)

print("run      C       val_accuracy  artifact")
for r in reg.runs():
    print(f"{r['run_id']}  {r['params']['C']:<6}  {r['metrics']['val_accuracy']:.4f}        {r['artifact'][:10]}")

best = max(reg.runs(), key=lambda r: r["metrics"]["val_accuracy"])
worst = min(reg.runs(), key=lambda r: r["metrics"]["val_accuracy"])
v1 = reg.register("digits-classifier", worst["run_id"])   # pretend an older model was live first
reg.promote("digits-classifier", v1, "production")
v2 = reg.register("digits-classifier", best["run_id"])
reg.promote("digits-classifier", v2, "staging")
print(f"\nregistered v{v1} ({worst['run_id']}) -> production, v{v2} ({best['run_id']}) -> staging")


def predict(arrays, Xb):
    return (Xb @ arrays["coef"].T + arrays["intercept"]).argmax(1)


# promotion gate: staging must beat production on a held-out set neither was selected on
_, prod = reg.load("digits-classifier", "production")
_, stag = reg.load("digits-classifier", "staging")
acc_prod, acc_stag = np.mean(predict(prod, X_gate) == y_gate), np.mean(predict(stag, X_gate) == y_gate)
print(f"gate: staging {acc_stag:.4f} vs production {acc_prod:.4f}")
if acc_stag > acc_prod:
    reg.promote("digits-classifier", v2, "production")
print("production is now version", reg.load("digits-classifier", "production")[0])

reg.promote("digits-classifier", v1, "production")       # rollback = move the pointer back
print("after rollback, production is version", reg.load("digits-classifier", "production")[0])
print("\nregistry.json:")
print(json.dumps(reg._read_registry(), indent=1, sort_keys=True))`;
const REG_OUT = String.raw`run      C       val_accuracy  artifact
run-001  0.001   0.8301        61df52ee8a
run-002  0.01    0.8914        4abefaa04d
run-003  0.1     0.9387        54c9682e2c
run-004  1.0     0.9777        61f88c25ff

registered v1 (run-001) -> production, v2 (run-004) -> staging
gate: staging 0.9667 vs production 0.8472
production is now version 2
after rollback, production is version 1

registry.json:
{
 "digits-classifier": {
  "stages": {
   "production": 1,
   "staging": 2
  },
  "versions": [
   {
    "run_id": "run-001",
    "version": 1
   },
   {
    "run_id": "run-004",
    "version": 2
   }
  ]
 }
}`;
const TEST_CODE = String.raw`# tests/test_model.py — the kind of checks a CI pipeline runs on every commit.
# Normally executed by pytest (each test_* function is discovered automatically);
# here a tiny runner at the bottom calls them so the file runs on its own.
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def build_model():
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def train(model, steps=150):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(X_tr / 16, dtype=torch.float32), torch.tensor(y_tr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(model(Xt), yt).backward()
        opt.step()
    return model.eval()


MODEL = train(build_model())


# ---------------- data tests: catch broken inputs before they reach training
def test_schema_and_ranges():
    assert X_tr.shape[1] == 64, "feature count changed"
    assert np.isfinite(X_tr).all(), "NaN or inf in features"
    assert X_tr.min() >= 0 and X_tr.max() <= 16, "pixel values outside [0, 16]"


def test_labels_and_balance():
    counts = np.bincount(y_tr, minlength=10)
    assert set(np.unique(y_tr)) == set(range(10)), "a class is missing"
    assert counts.min() / counts.max() > 0.8, f"class imbalance changed: {counts}"


def test_no_train_test_overlap():
    train_rows = {r.tobytes() for r in X_tr}
    dup = sum(r.tobytes() in train_rows for r in X_te)
    assert dup / len(X_te) < 0.01, f"{dup} test rows also appear in training data (leakage)"


# ---------------- model tests: behaviour, not just a number
def test_output_shape_and_probabilities():
    with torch.no_grad():
        p = torch.softmax(MODEL(torch.rand(5, 64)), dim=1)
    assert p.shape == (5, 10)
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-6)


def test_can_overfit_tiny_batch():
    m = build_model()                         # a healthy model+loss+optimizer memorises 20 examples
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    xb, yb = torch.tensor(X_tr[:20] / 16, dtype=torch.float32), torch.tensor(y_tr[:20])
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(xb), yb)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"cannot overfit 20 examples, loss {loss.item():.3f}"


def test_accuracy_above_threshold():
    with torch.no_grad():
        acc = (MODEL(torch.tensor(X_te / 16, dtype=torch.float32)).argmax(1).numpy() == y_te).mean()
    assert acc >= 0.90, f"accuracy regressed to {acc:.3f}"


def test_invariance_to_tiny_noise():
    xb = torch.tensor(X_te / 16, dtype=torch.float32)
    noisy = xb + 0.01 * torch.randn_like(xb)  # imperceptible change -> predictions should not flip
    with torch.no_grad():
        same = (MODEL(xb).argmax(1) == MODEL(noisy).argmax(1)).float().mean().item()
    assert same > 0.99, f"only {same:.3f} of predictions stable"


def test_deterministic_inference():
    xb = torch.tensor(X_te[:8] / 16, dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(MODEL(xb), MODEL(xb))


def preprocess(raw):
    return raw / 16.0                         # the production preprocessing function


def test_preprocessing_contract():
    out = preprocess(X_te)
    assert out.min() >= 0.0 and out.max() <= 1.0, "preprocess must map pixels into [0, 1]"


# "Golden" outputs: probabilities recorded when this model version was approved. A real repo
# stores them in a file next to the test; here we record them once at import time.
GOLDEN_X = X_te[:16]
with torch.no_grad():
    GOLDEN_P = torch.softmax(MODEL(torch.tensor(preprocess(GOLDEN_X), dtype=torch.float32)), dim=1)


def golden_gap(preprocess_fn):
    with torch.no_grad():
        p = torch.softmax(MODEL(torch.tensor(preprocess_fn(GOLDEN_X), dtype=torch.float32)), dim=1)
    return (p - GOLDEN_P).abs().max().item(), (p.argmax(1) == GOLDEN_P.argmax(1)).float().mean().item()


def test_golden_outputs():
    gap, _ = golden_gap(preprocess)
    assert gap < 1e-5, f"outputs drifted from the approved version by {gap:.2e}"


def test_golden_test_catches_missing_scaling():
    # Mutation check: does the golden test notice a classic bug (forgetting the /16)?
    gap, top1 = golden_gap(lambda raw: raw)
    print(f"      (bug injected: top-1 agrees on {top1:.3f} of inputs, probabilities move by up to {gap:.3f})")
    assert gap > 1e-5


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASSED {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAILED {t.__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")`;
const TEST_OUT = String.raw`PASSED test_schema_and_ranges
PASSED test_labels_and_balance
PASSED test_no_train_test_overlap
PASSED test_output_shape_and_probabilities
PASSED test_can_overfit_tiny_batch
PASSED test_accuracy_above_threshold
PASSED test_invariance_to_tiny_noise
PASSED test_deterministic_inference
PASSED test_preprocessing_contract
PASSED test_golden_outputs
      (bug injected: top-1 agrees on 0.938 of inputs, probabilities move by up to 0.550)
PASSED test_golden_test_catches_missing_scaling
11 passed, 0 failed`;

const GHA_YAML = String.raw`# .github/workflows/ml-ci.yml
name: ml-ci
on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Unit, data and model tests
        run: pytest -q tests/
      - name: Train a small model and check the quality gate
        run: python train.py --quick --out artifacts/ && python evaluate.py --min-accuracy 0.90 artifacts/

  build-image:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build the serving image, tagged with the commit
        run: docker build -t registry.example.com/digits-api:$GITHUB_SHA .
      # push, then deploy to staging; promotion to production is a separate, gated step`;

const MLFLOW_CODE = String.raw`import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)

mlflow.set_experiment("digits")
for C in (0.01, 0.1, 1.0):
    with mlflow.start_run():
        clf = LogisticRegression(C=C, max_iter=2000).fit(X_tr, y_tr)
        mlflow.log_param("C", C)
        mlflow.log_metric("val_accuracy", clf.score(X_val, y_val))
        # logs the model as an artifact and registers a new version under this name
        mlflow.sklearn.log_model(clf, "model", registered_model_name="digits-classifier")

# later, in the serving process
model = mlflow.pyfunc.load_model("models:/digits-classifier/3")   # name + version
print(model.predict(X_val[:5]))`;

const WANDB_CODE = String.raw`import wandb

run = wandb.init(project="digits", config={"lr": 1e-2, "hidden": 64, "epochs": 20})
for epoch in range(run.config.epochs):
    train_loss, val_acc = train_one_epoch(), evaluate()      # your own functions
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})

# version the trained weights as an artifact linked to this run
artifact = wandb.Artifact("digits-mlp", type="model")
artifact.add_file("weights.pt")
run.log_artifact(artifact)
run.finish()`;

const lesson: Lesson = {
  id: "m18/mlops",
  title: "Monitoring, Drift, CI/CD and Versioning",
  summary:
    "How to keep a deployed model healthy: what to watch with and without labels, data drift vs concept drift, PSI, KL and the KS test built from scratch and checked against SciPy, monitoring a real PyTorch classifier through simulated production changes, CI/CD tests for data and models, a file-based experiment tracker and model registry with promotion gates and rollback, MLflow and Weights & Biases, and cloud deployment, scaling and cost.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m18/serving", "m02/information-theory", "m02/probability", "m17/experimentation"],
  next: ["m19/robustness-safety", "m17/data-splits", "m20/llm-inference"],
  terms: ["mlops", "model-monitoring", "data-drift", "concept-drift", "label-shift", "psi", "ks-test", "ci-cd", "model-registry", "model-versioning", "canary-deployment", "autoscaling"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Test accuracy measures data from the past. After deployment the world keeps changing: new users, new sensors, seasons, even reactions to the model's own decisions. And a model can fail **silently** — it still returns neat answers, just worse ones. **MLOps** (machine learning operations) keeps models correct over time: **monitor** inputs, outputs and real outcomes; detect **drift** (changes in the data); **test** data and models on every change (**CI/CD**); **version** data, code and models so any prediction can be traced and any release rolled back; and do it all at a sensible **scale and cost**.",
      },
      {
        t: "p",
        text: "The hard part: accuracy on today's traffic needs labels, which often arrive late (a loan defaults months later) or never. So monitoring uses label-free stand-ins — the distribution of inputs, confidence and predicted classes — compared with a **reference window** saved at deployment. We build these statistics from scratch, see what they can and cannot catch, then build the tests, registry and gates that turn a drift alarm into a safe retrain-and-release cycle.",
      },
      {
        t: "callout",
        kind: "key",
        text: "**Data drift**: $P(x)$ changes. **Concept drift**: $P(y\\mid x)$ changes. **Label shift**: $P(y)$ changes with $P(x\\mid y)$ fixed. Input statistics (PSI, KL, KS) only see changes in $P(x)$ or in the model's outputs; concept drift needs labels.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — the deployed service whose inputs and outputs we log, and the files we version.",
          "[Information theory](/learn/m02/information-theory) — KL divergence; PSI is KL measured in both directions on histogram bins.",
          "[Probability](/learn/m02/probability) — distributions, cumulative distribution functions (for the KS statistic) and conditional probability (for the kinds of drift).",
          "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — experiment tracking and seeds; the registry here carries tracking into deployment.",
          "[Train/Validation/Test Splits](/learn/m17/data-splits) — a promotion gate is only honest on data that was not used to pick the model.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A weather station and a flight recorder",
        text: "Picture a factory built for one climate, with a weather station beside it. The station can't tell you if products are still good — that needs inspection (labels). But it warns you when the weather is far outside what the factory was tested in (data drift), and it can watch the factory's own gauges (confidence, output mix). Concept drift is normal weather but changed customer tastes: no sensor notices; only returned products (labels) will. Versioning is the flight recorder plus a spare-parts store: you can rebuild which model, code and data made a decision, and swap back to the last good version in seconds.",
      },
      {
        t: "p",
        text: "Two lessons from the experiments below: **a drift alarm is not an accuracy alarm** (a dimmer scanner moved the inputs hugely, but this model was robust to it), and **\"statistically significant\" is not \"important\"** (with 100,000 samples, KS flags a 0.03-standard-deviation shift nobody cares about). Good monitoring uses effect sizes — how big the change is — with thresholds set by business impact.",
      },
    ],
    mechanism: [
      { t: "h", text: "The monitoring loop" },
      {
        t: "steps",
        items: [
          { title: "1. Log every request", text: "Inputs (or summaries), model version, prediction, confidence, latency, timestamp, and a request id to match with the real label later." },
          { title: "2. Capture a reference window", text: "From validation data or the first days of production: a histogram per feature (quantile bin edges), the confidence distribution and the predicted-class mix." },
          { title: "3. Compare windows", text: "Each hour, day or week, compare with the reference: PSI or KL on binned features, KS for continuous features, PSI or chi-square for categorical features and predicted classes." },
          { title: "4. Alert on effect sizes", text: "Thresholds on PSI or KS D (not raw p-values), held over several windows so noise doesn't fire alarms. Also alert on latency, error rate and missing values." },
          { title: "5. Join labels when they arrive", text: "Compute real accuracy and calibration per window and segment — the only way to see concept drift." },
          { title: "6. Respond", text: "Investigate first: a broken upstream pipeline is more common than a changing world. Fix data bugs or retrain on recent data, then test, register and roll out through the same gated pipeline." },
        ],
      },
      { t: "h", text: "CI/CD for ML" },
      {
        t: "list",
        items: [
          "**Continuous integration** runs on every change: code unit tests; **data tests** (schema, ranges, missing values, class balance, train/test leakage); **model tests** (valid shapes and probabilities, can overfit a tiny batch, accuracy above a threshold, ignores irrelevant changes, same answer twice, matches saved \"golden\" outputs).",
          "**Continuous delivery** builds a package that never changes (container image + registered model version) and deploys it to staging automatically.",
          "**Promotion** to production is gated: the new model must beat the current one on a held-out set neither was chosen on, and meet latency and size budgets. Then a **canary** (a small share of real traffic) or **shadow** deployment (gets copies of traffic; answers unused) compares it live before full rollout.",
          "**Continuous training** (sometimes called CT) retrains on a schedule or when drift is found, feeding the same pipeline.",
        ],
      },
      { t: "h", text: "Versioning and registries" },
      {
        t: "p",
        text: "Reproducing a prediction needs four versions: code (git commit), data (snapshot id or content hash), configuration (hyperparameters, seeds) and the model file (content hash). An **experiment tracker** records each run's params, metrics and files. A **model registry** gives model files names and ever-increasing versions, and maps **stages** or aliases (staging, production) to versions — so deploying or rolling back is just moving a pointer. The files never change and are named by their hash (**content-addressed**). MLflow, Weights & Biases and cloud platforms offer this as a service; the Framework section builds one from plain files to show there is no magic.",
      },
      { t: "h", text: "Cloud deployment, scalability and cost" },
      {
        t: "list",
        items: [
          "**Where to deploy**: managed endpoints on cloud ML platforms, containers on Kubernetes or serverless container services, serverless functions for small models, batch jobs for offline scoring, and edge devices.",
          "**Horizontal scaling**: identical stateless replicas behind a load balancer. **Autoscaling** adds or removes replicas based on CPU/GPU use, request rate or queue length. Keep a few always warm: loading a model makes cold starts slow.",
          "**Cutting cost**: right-size instances (measure throughput per replica first), batch requests on GPUs, use spot/preemptible machines for training and batch scoring, shrink models ([quantization, pruning, distillation](/learn/m18/model-optimization)), cache repeated predictions, scale rarely used models to zero, switch off idle dev machines.",
          "**Cost per prediction** = instance cost per hour / predictions per hour at target utilisation — the number to optimise and watch.",
        ],
      },
    ],
    math: [
      { t: "p", text: "Split the reference sample into $K$ bins using quantile edges, so each bin holds about $1/K$ of the reference. Let $r_k$ and $c_k$ be the fractions of the reference and current samples in bin $k$. Both are floored at a tiny $\\epsilon$ so an empty bin never gives $\\log 0$." },
      { t: "math", tex: String.raw`\mathrm{PSI} = \sum_{k=1}^{K} (c_k - r_k)\,\ln\frac{c_k}{r_k} = \mathrm{KL}(c\,\|\,r) + \mathrm{KL}(r\,\|\,c)`, caption: "Population Stability Index: the KL divergence between the binned distributions, added up in both directions (Jeffreys divergence)." },
      { t: "math", tex: String.raw`\mathrm{KL}(c\,\|\,r) = \sum_{k=1}^{K} c_k \ln\frac{c_k}{r_k}`, caption: "KL divergence in nats: not symmetric, always ≥ 0, and = 0 only when the binned distributions match." },
      { t: "p", text: "In words: per bin, PSI multiplies \"how much the share changed\" by \"the log of how many times bigger it got\". No term is negative, since $c_k - r_k$ and $\\ln(c_k/r_k)$ share a sign. Industry rule of thumb (from credit scoring, not a statistical test): < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 major shift." },
      { t: "p", text: "The **two-sample Kolmogorov–Smirnov (KS) statistic** needs no bins. It compares empirical cumulative distribution functions $F_a(t) = \\frac{1}{n}\\#\\{i : a_i \\le t\\}$ (the share of samples at or below $t$) and $F_b$:" },
      { t: "math", tex: String.raw`D = \sup_t \left|F_a(t) - F_b(t)\right|,\qquad \text{reject "same distribution" at level } \alpha \text{ if } D > c(\alpha)\sqrt{\frac{n+m}{n\,m}}`, caption: "n, m: sample sizes; c(0.05) ≈ 1.358 for large samples." },
      { t: "p", text: "$D$ is the biggest vertical gap between the two curves. The critical value shrinks like $1/\\sqrt{n}$, so with large samples any non-zero gap becomes \"significant\" — so threshold $D$ or PSI (effect sizes), not p-values." },
      { t: "p", text: "**Kinds of drift.** Write the joint distribution two ways: $P(x, y) = P(y\\mid x)P(x) = P(x\\mid y)P(y)$. Data (covariate) drift changes $P(x)$ with $P(y\\mid x)$ fixed; concept drift changes $P(y\\mid x)$; label (prior) shift changes $P(y)$ with $P(x\\mid y)$ fixed. A model learns $P(y\\mid x)$, so under pure covariate drift it stays right where it was trained, but may be wrong in regions it never saw." },
      { t: "h", text: "Worked example: PSI, KL and KS on four bins" },
      {
        t: "steps",
        items: [
          { title: "Reference and current bin fractions", tex: String.raw`r = (0.25,\ 0.25,\ 0.25,\ 0.25),\qquad c = (0.10,\ 0.20,\ 0.30,\ 0.40)` },
          { title: "PSI terms", tex: String.raw`(c_k - r_k)\ln\tfrac{c_k}{r_k} = (-0.15)(-0.916291),\ (-0.05)(-0.223144),\ (0.05)(0.182322),\ (0.15)(0.470004)` },
          { title: "", tex: String.raw`= 0.137444 + 0.011157 + 0.009116 + 0.070501 = 0.228217` },
          { title: "KL(c ‖ r)", tex: String.raw`0.1\ln0.4 + 0.2\ln0.8 + 0.3\ln1.2 + 0.4\ln1.6 = -0.091629 - 0.044629 + 0.054696 + 0.188001 = 0.106440` },
          { title: "KL(r ‖ c) and the check", tex: String.raw`\mathrm{KL}(r\|c) = 0.121777,\qquad 0.106440 + 0.121777 = 0.228217 = \mathrm{PSI}\ \checkmark` },
          { title: "KS on the binned CDFs", tex: String.raw`F_r = (0.25, 0.5, 0.75, 1),\ F_c = (0.1, 0.3, 0.6, 1) \Rightarrow D = \max(0.15, 0.2, 0.15, 0) = 0.2` },
        ],
      },
      { t: "p", text: "PSI = 0.228 lands in the \"moderate shift\" band. Notice that KL differs by direction (0.106 vs 0.122), while PSI is the same either way." },
    ],
    visual: [
      { t: "sim", id: "distributions", caption: "Drift means the distribution changed: move the mean or widen the spread and compare the new shape with the old one — exactly what our drift statistics measure." },
      { t: "sim", id: "entropy", caption: "KL divergence between two distributions — PSI is KL in both directions, summed over histogram bins." },
      {
        t: "list",
        items: [
          "In the distributions simulator, move the mean by a quarter of a standard deviation, then a full one. Our program measured PSI 0.093 and 0.939: the first is barely visible, the second jumps out.",
          "Widen the spread without moving the mean. The CDFs cross and KS catches it (D = 0.108 for sd 1 → 1.5); a mean-only monitor would miss it.",
          "In the entropy simulator, give one distribution almost no mass where the other has a lot, and watch KL blow up — why empty bins are floored at a small ε.",
          "Swap the two distributions: KL changes, while PSI (both directions added) would not.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "PSI, KL and KS from scratch, checked against `scipy.stats.ks_2samp` on five drift scenarios; how sample size changes p-values; and concept drift where inputs stay flat while accuracy collapses.",
      },
      { t: "code", lang: "python", title: "drift_stats.py", code: DRIFT_CODE, output: DRIFT_OUT },
      {
        t: "list",
        items: [
          "Our KS statistic matches SciPy's to three decimals in every scenario; SciPy also gives the p-value.",
          "PSI ranks the scenarios sensibly: no drift 0.007, +0.25 sd shift 0.093 (just under the 0.1 band), variance change 0.194, +1 sd 0.939.",
          "Filling 10% missing values with 0 makes a spike in the middle bins (PSI 0.104). Broken upstream pipelines look like drift — and are a common real cause of alerts.",
          "Sample size: a 0.03-sd shift is \"not significant\" at 1,000 samples (p = 0.80) and \"highly significant\" at 100,000 (p = 3.4e-06), while PSI calls it tiny. PSI on 1,000 samples is noisy too (0.0213 vs 0.0008), so small windows need higher thresholds.",
          "Concept drift: input PSI ≈ 0.005 on both features — no input monitor fires — yet accuracy fell from 1.000 to 0.599. Comparing the positive-prediction share (0.502) with the true rate (0.393) also needs labels. Only label feedback shows the problem.",
        ],
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              { t: "p", text: "A real PyTorch digits classifier over four simulated production weeks, watched with label-free signals (PSI on brightness, confidence and predicted-class mix) next to the true accuracy we would only learn later." },
              { t: "code", lang: "python", title: "monitor_torch.py", code: MON_CODE, output: MON_OUT },
              {
                t: "list",
                items: [
                  "Week 2 (dimmer scanner): the biggest input drift (PSI 8.39), yet accuracy is unchanged (0.967) — this model barely cares about overall brightness. The alert should start an investigation, not an automatic retrain.",
                  "Week 3 (sensor noise): moderate input drift, confidence PSI 0.211 and a real drop to 0.922 accuracy. Here confidence is the useful early warning.",
                  "Week 4 (label shift: only digits 0–4): inputs look almost normal (0.146) but predicted-class PSI is 2.906. Accuracy is fine; the business may still care, since the downstream population changed.",
                  "Each alert means something different: you need several signals and a readable dashboard, not one number.",
                ],
              },
            ],
          },
          {
            label: "Model registry",
            blocks: [
              { t: "p", text: "A tiny tracker and registry from JSON files and hash-named `.npz` files: log a sweep, register versions, promote through a gate scored on a held-out set not used for picking, and roll back by moving the stage pointer." },
              { t: "code", lang: "python", title: "file_registry.py", code: REG_CODE, output: REG_OUT },
              { t: "p", text: "Worth copying: model files are named by their hash, so a version can't change silently; the registry file is replaced atomically (`os.replace`); each run stores a data fingerprint, tracing the model to its exact training data; and rollback is instant because it only moves the production stage pointer." },
            ],
          },
          {
            label: "CI tests",
            blocks: [
              { t: "p", text: "Data and model tests in pytest style. pytest finds each `test_*` function by itself; a small runner at the bottom lets the file run on its own here." },
              { t: "code", lang: "python", title: "test_model.py", code: TEST_CODE, output: TEST_OUT },
              { t: "p", text: "The mutation check shows something we found while writing this lesson. Forgetting the `/16` scaling kept the top prediction on 15 of 16 golden inputs (0.938); in an earlier draft, whole-test-set accuracy without scaling was still 0.958, above the 0.90 threshold. Yet probabilities moved by up to 0.55. Golden-output tests on probabilities catch preprocessing bugs that accuracy thresholds miss." },
              { t: "p", text: "A GitHub Actions workflow that runs these tests on every pull request and builds the serving image on main:" },
              { t: "code", lang: "text", title: ".github/workflows/ml-ci.yml", code: GHA_YAML },
            ],
          },
          {
            label: "MLflow",
            blocks: [
              { t: "p", text: "MLflow offers experiment tracking, a model format and a model registry with versions and aliases. It is not installed here, so this snippet uses only its core, well-known calls and is not run." },
              { t: "code", lang: "python", title: "mlflow_tracking.py (not executed in CI)", code: MLFLOW_CODE, output: "Logs three runs with their C parameter and validation accuracy to the local ./mlruns store, registers three versions of digits-classifier, then loads version 3 and prints five predicted digit labels." },
            ],
          },
          {
            label: "Weights & Biases",
            blocks: [
              { t: "p", text: "Weights & Biases is a hosted experiment-tracking service with dashboards, sweeps and versioned files (artifacts). It is not installed here and needs an account, so this is not run." },
              { t: "code", lang: "python", title: "wandb_tracking.py (not executed in CI)", code: WANDB_CODE, output: "Streams per-epoch loss and accuracy to a W&B project dashboard and uploads weights.pt as a versioned model artifact linked to the run." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Drift statistic", "Data type", "Pros", "Cons"],
        rows: [
          ["PSI", "Binned numeric or categorical", "Symmetric, easy-to-read bands, cheap to compute from saved histograms", "Depends on the bins; needs ε for empty bins; noisy on small windows"],
          ["KL divergence", "Binned / densities", "Clear information-theory meaning", "Not symmetric; infinite when the reference has an empty bin"],
          ["Jensen–Shannon divergence", "Binned", "Symmetric and capped (≤ ln 2)", "Less sensitive to small shifts"],
          ["Two-sample KS", "Continuous, 1-D", "No bins; gives a p-value", "Too sensitive at large n; one feature at a time"],
          ["Chi-square test", "Categorical", "Standard test for counts", "Same large-n problem as KS"],
          ["Wasserstein (earth mover's) distance", "Numeric", "Measured in the feature's own units", "No fixed threshold scale"],
          ["Classifier two-sample test (train a model to tell reference from current)", "Multivariate", "Catches changes in how features move together", "Heavier; needs a held-out evaluation"],
        ],
      },
      {
        t: "table",
        head: ["Kind of change", "What changes", "Visible without labels?", "Usual response"],
        rows: [
          ["Data (covariate) drift", "P(x)", "Yes — input statistics", "Investigate; retrain on new data if accuracy suffers"],
          ["Label (prior) shift", "P(y)", "Partly — predicted-class mix", "Re-weight or recalibrate thresholds"],
          ["Concept drift", "P(y | x)", "No — needs labels", "Retrain on recent labelled data"],
          ["Upstream data bug", "Schema, units, missing values", "Yes — data tests, PSI spikes", "Fix the pipeline, not the model"],
        ],
      },
      {
        t: "table",
        head: ["Rollout strategy", "How", "Pros", "Cons"],
        rows: [
          ["Big bang", "Replace all replicas at once", "Simple", "Every user gets a bad model"],
          ["Blue/green", "Two full environments; switch traffic", "Instant rollback", "Double capacity during the switch"],
          ["Canary", "Small % of traffic to the new version", "Live comparison, limited risk", "Needs metrics per version"],
          ["Shadow", "New version gets copies of traffic; answers thrown away", "Zero risk to users", "Can't measure user-facing outcomes"],
          ["A/B test", "Random split; measure a business metric", "Shows cause and effect", "Needs traffic and time to reach significance"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "Credit scoring is where PSI began, as the standard stability check for scorecards; regulators expect lenders to monitor model stability over time.",
          "Fraud detection has concept drift built in: fraudsters adapt to the model, so it is retrained often on recently labelled cases.",
          "Recommenders shape their own training data (users can only click what was shown), so they are judged with online A/B tests, not offline accuracy alone.",
          "Open-source and managed tools — MLflow, Weights & Biases, DVC for data versioning, Evidently and similar libraries for drift reports, and cloud platforms' model registries — provide the pieces built from scratch in this lesson.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "table",
        head: ["Decision", "Benefit", "Cost / risk"],
        rows: [
          ["Low drift thresholds", "Early warnings", "Alert fatigue: people start ignoring alerts"],
          ["Short monitoring windows", "Fast detection", "Noisy statistics (PSI 0.0213 at n = 1,000 vs 0.0008 at n = 100,000 for the same shift)"],
          ["Automatic retraining on drift", "No human bottleneck", "Can learn from corrupted data or feedback loops; needs strong gates"],
          ["Logging full inputs", "Complete data for debugging and retraining", "Storage cost, privacy and legal duties"],
          ["Managed ML platform", "Less infrastructure work", "Vendor lock-in, cost at scale"],
          ["Self-hosted tools", "Control, portability", "Engineering and upkeep effort"],
        ],
      },
      {
        t: "p",
        text: "The big trade-off is maturity versus speed: a hand-deployed notebook model ships fast but can't be maintained; a full platform is slow to build. A practical order: version everything and log predictions first (cheap, enables everything else), then CI data and model tests, then drift dashboards, then automated retraining once you trust the gates.",
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Alerting on p-values.** With lots of traffic every shift is significant (p = 3.4e-06 for a 0.03-sd shift here). Fix: threshold effect sizes (PSI, KS D) and require persistence across windows.",
          "**Thinking no input drift means all is fine.** Concept drift leaves inputs unchanged (PSI 0.005, accuracy 0.599). Fix: collect labels, even a small audited sample, and track real accuracy.",
          "**Thinking drift means the model is broken.** The dimmer-scanner week: PSI 8.39, accuracy unchanged. Fix: drift means investigate; check confidence and labels before retraining.",
          "**Scoring the promotion gate on the validation set used to pick the model.** The winner's score is too optimistic. Fix: a separate gate set, as in our registry program.",
          "**Recomputing bin edges from the current window.** Its quantile bins always look even, hiding drift. Fix: freeze edges from the reference, with open outer bins.",
          "**Not versioning data and preprocessing.** A model hash alone can't reproduce a prediction. Fix: record code commit, data fingerprint, config and model hash per run.",
          "**Overwritable model files (\"model_latest.pt\").** You can't tell what is deployed or roll back. Fix: hash-named files that never change, and registry stages pointing to versions.",
          "**Testing only overall accuracy in CI.** Preprocessing bugs can keep top-1 accuracy (our missing-scaling bug). Fix: golden-output tests on probabilities, invariance and data tests.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between data drift and concept drift?", a: "Data (covariate) drift changes the inputs P(x) — say, a new camera gives darker images — while the input-label link stays the same. Concept drift changes P(y | x): the same input now has a different right answer, like a once-harmless pattern becoming fraud. Data drift shows in inputs alone; concept drift needs labels." },
      { t: "qa", level: "beginner", q: "What should you monitor for a deployed model?", a: "Operations (latency percentiles, throughput, error rates, resources); input quality and drift (schema, missing values, per-feature distributions); outputs (confidence, predicted-class mix); model version per request; real performance overall and per segment once labels arrive; and the business metrics the model should move." },
      { t: "qa", level: "beginner", q: "What is a model registry?", a: "A catalogue of trained models with names, unchangeable versions and metadata (metrics, training data, code version), plus stages or aliases like staging and production pointing to versions. Deploy and rollback become moving a pointer, and every prediction traces to the exact model that made it." },
      { t: "qa", level: "intermediate", q: "How is PSI computed and how do you interpret it?", a: "Bin the reference data (usually 10 quantile bins, open outer bins). Take the share of reference and current data per bin, floored at a small epsilon, and sum (c − r)·ln(c/r), which equals KL(c‖r) + KL(r‖c). Rule of thumb: below 0.1 stable, 0.1–0.25 moderate, above 0.25 significant shift — but tune thresholds to window size and business impact, since small windows give noisy PSI." },
      { t: "qa", level: "intermediate", q: "Why can a KS test be misleading for drift monitoring?", a: "Its p-value depends on sample size: the critical value of D shrinks like 1/sqrt(n), so heavy traffic makes tiny, harmless shifts significant, while small windows can miss real ones. It also checks one feature at a time, missing changes in how features move together. Use D as an effect size with tuned thresholds, combine it with PSI, and add multivariate methods (e.g. a domain classifier) when needed." },
      { t: "qa", level: "intermediate", q: "What tests belong in CI for an ML project?", a: "Code unit tests; data tests (schema, value ranges, missing values, class balance, duplicates/leakage between splits); model tests (output shape and valid probabilities, can overfit a tiny batch, accuracy above a threshold on a fixed set, invariance/directional tests, determinism, golden outputs on fixed inputs to catch preprocessing changes); and serving tests (API schema, error codes, batch vs single-request consistency, latency budget)." },
      { t: "qa", level: "advanced", q: "Labels for your model arrive 90 days late. How do you detect performance degradation earlier?", a: "Label-free stand-ins: input drift per feature and jointly (domain classifier); output drift (confidence, predicted-class mix, entropy); disagreement with a reference or shadow model. If the model is well calibrated, estimate performance under covariate shift with importance weighting or confidence-based estimators. Get a small, fast-labelled audit sample (human review) for direct accuracy. Split metrics by key groups, and set alert thresholds from past windows where performance was later known." },
      { t: "qa", level: "advanced", q: "Design a safe automated retraining pipeline.", a: "Trigger on a schedule or lasting drift. Pull a versioned data snapshot and validate it (schema, ranges, label sanity, leakage); stop on failure. Train with a fixed config, logged to a tracker. Evaluate on a fixed held-out gate set and recent labelled data, per segment, with latency/size budgets, against production. Register the candidate as a new unchangeable version. Deploy to staging, then shadow or canary with live comparison and automatic rollback rules. Promote by moving the registry alias; keep the old version warm for instant rollback. Guard against feedback loops (the model affecting its own training labels) and poisoned data: limit recent data's weight and require human approval for big changes." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Input feature distributions are unchanged but accuracy has dropped. This is most likely…", options: ["data drift", "concept drift", "a latency problem", "label smoothing"], answer: 1, explain: "P(y|x) changed, and input statistics can't see that (our simulation: PSI 0.005, accuracy 0.599)." },
          { kind: "numeric", q: "Two bins: reference (0.5, 0.5), current (0.3, 0.7). Compute PSI (3 decimals).", answer: 0.169, tolerance: 0.002, explain: "(−0.2)·ln(0.6) + (0.2)·ln(1.4) = 0.102165 + 0.067294 = 0.169459." },
          { kind: "numeric", q: "Empirical CDFs at four points: F_a = (0.2, 0.5, 0.8, 1.0), F_b = (0.1, 0.3, 0.7, 1.0). What is the KS statistic D?", answer: 0.2, tolerance: 0.0001, explain: "The largest gap: max(0.1, 0.2, 0.1, 0) = 0.2." },
          { q: "Why are bin edges for PSI taken from the reference window?", options: ["It is faster", "Edges from the current window would make the current histogram uniform by construction and hide drift", "KL requires it", "To avoid negative values"], answer: 1, explain: "Quantile bins built from the current data always hold equal shares of the current data, so it would always look unchanged." },
          { q: "In our monitoring experiment, which week had huge input drift but no accuracy loss?", options: ["Normal traffic", "Dimmer scanner", "Sensor noise", "Only digits 0–4"], answer: 1, explain: "PSI(brightness) 8.39, accuracy 0.967 vs 0.969 in the reference." },
          { kind: "numeric", q: "A GPU instance costs 3.00 USD per hour and serves 600 predictions per second at target utilisation. What is the cost per million predictions in USD? (2 decimals)", answer: 1.39, tolerance: 0.01, explain: "600 × 3600 = 2.16 M predictions per hour; 3.00 / 2.16 = 1.39 USD per million." },
          { q: "Which rollout sends copies of live traffic to the new model without using its answers?", options: ["Canary", "Blue/green", "Shadow", "Big bang"], answer: 2, explain: "A shadow deployment compares outputs with zero risk to users." },
        ],
      },
      {
        t: "exercise",
        title: "PSI and KL by hand",
        prompt: "Reference bin fractions r = (0.2, 0.3, 0.3, 0.2); current c = (0.1, 0.2, 0.3, 0.4). Compute (a) each PSI term and the PSI, (b) KL(c‖r) and KL(r‖c), and check that PSI equals their sum. (c) What does the rule of thumb say?",
        hint: "ln(0.5) = −0.693147, ln(2/3) = −0.405465, ln 2 = 0.693147.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) terms", tex: String.raw`(-0.1)(-0.693147) = 0.069315,\ (-0.1)(-0.405465) = 0.040547,\ 0,\ (0.2)(0.693147) = 0.138629` },
              { title: "(a) PSI", tex: String.raw`0.069315 + 0.040547 + 0 + 0.138629 = 0.248491` },
              { title: "(b) KL(c‖r)", tex: String.raw`0.1(-0.693147) + 0.2(-0.405465) + 0 + 0.4(0.693147) = -0.069315 - 0.081093 + 0.277259 = 0.126851` },
              { title: "(b) KL(r‖c)", tex: String.raw`0.2(0.693147) + 0.3(0.405465) + 0 + 0.2(-0.693147) = 0.121640,\quad 0.126851 + 0.121640 = 0.248491\ \checkmark` },
            ],
          },
          { t: "p", text: "(c) PSI ≈ 0.248 is at the top of the \"moderate shift\" band (0.1–0.25). Investigate: does it last over several windows, and did confidence or (later) accuracy move?" },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a multivariate drift detector",
        prompt: "Per-feature PSI misses changes in correlation. Build a **classifier two-sample test**: label reference rows 0 and current rows 1, train a logistic regression with cross-validation, and report ROC AUC (≈ 0.5 means the windows can't be told apart). Test it on 2-D Gaussian data where the current window keeps both marginals N(0, 1) but changes the correlation from 0 to 0.8.",
        hint: "`sklearn.model_selection.cross_val_score(LogisticRegression(), X, y, scoring='roc_auc', cv=5)`. A linear model can't see a correlation change from the raw features — add the product feature x1·x2.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n\nrng = np.random.default_rng(0)\nref = rng.normal(0, 1, (5000, 2))\ncov = np.array([[1.0, 0.8], [0.8, 1.0]])\ncur = rng.multivariate_normal([0, 0], cov, 5000)    # same marginals, new correlation\n\nX = np.vstack([ref, cur])\ny = np.r_[np.zeros(5000), np.ones(5000)]\nfeats = np.c_[X, X[:, 0] * X[:, 1]]                 # the product feature exposes correlation\nauc = cross_val_score(LogisticRegression(), feats, y, scoring=\"roc_auc\", cv=5).mean()\nprint(\"domain-classifier AUC:\", round(auc, 3))",
          },
          { t: "p", text: "Per-feature PSI stays near 0, since each feature alone is still N(0, 1). The domain classifier's AUC rises clearly above 0.5, because the product feature averages 0 in the reference and 0.8 now. Without it, a linear classifier stays at AUC ≈ 0.5; a flexible classifier (e.g. gradient-boosted trees) avoids hand-crafting such features." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Data drift: $P(x)$ changes; concept drift: $P(y\\mid x)$ changes (needs labels); label shift: $P(y)$ changes.",
          "PSI $= \\sum_k (c_k - r_k)\\ln(c_k/r_k) = \\mathrm{KL}(c\\|r) + \\mathrm{KL}(r\\|c)$ on bins from reference quantiles; rule of thumb < 0.1 / 0.1–0.25 / > 0.25.",
          "KS $D = \\sup_t |F_a - F_b|$; threshold effect sizes, not p-values (large n makes everything significant).",
          "Watch inputs, confidence, predicted-class mix, operations, and real accuracy once labels arrive. Drift ≠ damage (dimmer scanner); no drift ≠ healthy (concept drift).",
          "CI: data tests, model behaviour tests, golden outputs. CD: unchangeable images and registered versions, gated promotion on held-out data, canary/shadow rollouts; rollback = move the stage pointer.",
          "Version code, data, config and model files together; registries map stages to hash-named versions.",
          "Autoscale stateless replicas with a warm minimum; optimise cost per prediction.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Adversarial Examples, Robustness and AI Safety](/learn/m19/robustness-safety) — distribution shift pushed to the extreme: inputs crafted to fool the model.",
          "[Train/Validation/Test Splits and Cross-Validation](/learn/m17/data-splits) — honest evaluation sits under every gate and alert threshold.",
          "[Quantization, Pruning and Distillation](/learn/m18/model-optimization) — the main lever for cost per prediction.",
        ],
      },
    ],
  },
};

export default lesson;
