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
    "Keeping a deployed model healthy: what to monitor with and without labels, data drift vs concept drift, PSI, KL and the KS test implemented and checked against SciPy, monitoring a real PyTorch classifier under simulated production shifts, CI/CD tests for data and models, a file-based experiment tracker and model registry with promotion gates and rollback, MLflow and Weights & Biases, cloud deployment, scaling and cost.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m18/serving", "m02/information-theory", "m02/probability", "m17/experimentation"],
  next: ["m19/robustness-safety", "m17/data-splits", "m20/llm-inference"],
  terms: ["mlops", "model-monitoring", "data-drift", "concept-drift", "label-shift", "psi", "ks-test", "ci-cd", "model-registry", "model-versioning", "canary-deployment", "autoscaling"],
  sections: {
    intro: [
      {
        t: "p",
        text: "A model's test accuracy is a measurement on data from the past. Once deployed, the world it sees keeps changing: new users, new sensors, seasonality, competitors, and feedback from the model's own decisions. Unlike ordinary software, a model can fail **silently** — it still returns well-formed answers, just worse ones. **MLOps** (machine learning operations) is the set of practices that keep models correct over time: **monitoring** inputs, outputs and (when available) outcomes; detecting **drift**; **testing** data and models automatically on every change (**CI/CD**); **versioning** data, code and models so any prediction can be traced and any release rolled back; and running all of it at acceptable **scale and cost**.",
      },
      {
        t: "p",
        text: "The core difficulty is that the quantity we care about — accuracy on today's traffic — usually needs labels that arrive late (a loan defaults months later) or never. So monitoring relies on proxies that need no labels: the distribution of the inputs, of the model's confidence and of its predicted classes, compared with a reference window captured at deployment. This lesson builds those statistics from scratch, shows exactly what they can and cannot detect, and builds the versioning machinery (tests, a registry, promotion gates) that turns a drift alarm into a safe retrain-and-release cycle.",
      },
      {
        t: "callout",
        kind: "key",
        text: "**Data drift**: $P(x)$ changes. **Concept drift**: $P(y\\mid x)$ changes. **Label shift**: $P(y)$ changes with $P(x\\mid y)$ fixed. Input statistics (PSI, KL, KS) detect only changes in $P(x)$ or in the model's outputs; concept drift needs labels.",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — the deployed service whose inputs and outputs we log, and the artifacts we version.",
          "[Information theory](/learn/m02/information-theory) — KL divergence; PSI is a symmetrised KL on histogram bins.",
          "[Probability](/learn/m02/probability) — distributions, cumulative distribution functions (needed for the KS statistic), conditional probability (for the drift taxonomy).",
          "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — experiment tracking and seeds; the registry here extends tracking to deployment.",
          "[Train/Validation/Test Splits](/learn/m17/data-splits) — a promotion gate is only honest on data that was not used to select the model.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A weather station and a flight recorder",
        text: "Monitoring a model is like running a weather station next to a factory that was designed for a particular climate. The station cannot tell you directly whether the products are still good (that needs inspection — labels), but it can tell you that today's temperature and humidity are far outside anything the factory was tested in (data drift), and it can watch the factory's own gauges (the model's confidence and output mix). Concept drift is the case where the weather is perfectly normal but customers' tastes have changed: no sensor at the factory will notice; only returned products (labels) will. Versioning is the flight recorder plus the spare parts store: when something goes wrong you can reconstruct exactly which model, code and data produced a decision, and swap back to the last good version in seconds.",
      },
      {
        t: "p",
        text: "Two lessons from the experiments below are worth carrying in your head. First, a drift alarm is not an accuracy alarm: a dimmer scanner moved the input distribution enormously while accuracy barely changed, because the model happened to be robust to that change. Second, \"statistically significant\" is not \"important\": with 100,000 samples the KS test flags a shift of 0.03 standard deviations that no one would care about. Good monitoring combines effect-size statistics with thresholds chosen for the business impact.",
      },
    ],
    mechanism: [
      { t: "h", text: "The monitoring loop" },
      {
        t: "steps",
        items: [
          { title: "1. Log every request", text: "Inputs (or summaries of them), model version, prediction, confidence, latency, timestamp and a request id that can later be joined with the eventual label." },
          { title: "2. Capture a reference window", text: "Statistics of the validation data or the first days of production: per-feature histograms (with quantile bin edges), the confidence distribution and the predicted-class mix." },
          { title: "3. Compare windows", text: "For each time window (hour/day/week) compute drift statistics against the reference: PSI or KL on binned features, the KS statistic for continuous features, PSI or chi-square for categorical ones and for predicted classes." },
          { title: "4. Alert on effect sizes", text: "Thresholds on PSI/KS D (not raw p-values), sustained over several windows to avoid noise; plus operational alerts on latency, error rate and missing values." },
          { title: "5. Join labels when they arrive", text: "Compute real accuracy/calibration per window and per segment; this is the only way to see concept drift." },
          { title: "6. Respond", text: "Investigate (a broken upstream pipeline is more common than a changing world), fix data bugs, or retrain on recent data, then test, register and roll out the new version through the same gated pipeline." },
        ],
      },
      { t: "h", text: "CI/CD for ML" },
      {
        t: "list",
        items: [
          "**Continuous integration** runs on every change: unit tests for code, **data tests** (schema, ranges, missing values, class balance, train/test leakage), **model tests** (output shapes and probabilities, can-overfit-a-tiny-batch, accuracy above a threshold, invariance to irrelevant perturbations, determinism, golden outputs for fixed inputs).",
          "**Continuous delivery** builds an immutable artifact (container image + registered model version) and deploys it to staging automatically.",
          "**Promotion** to production is gated: the candidate must beat the current production model on a held-out set neither was selected on, and pass latency/size budgets; then a **canary** (a small fraction of traffic) or **shadow** deployment (receives copies of traffic, answers are not used) compares it live before full rollout.",
          "**Continuous training** (sometimes called CT) retrains on a schedule or when drift is detected, feeding the same pipeline.",
        ],
      },
      { t: "h", text: "Versioning and registries" },
      {
        t: "p",
        text: "Reproducing a prediction needs four versions: code (git commit), data (a snapshot id or content hash), configuration (hyperparameters, seeds) and the model artifact (content hash). An **experiment tracker** records runs (params, metrics, artifacts). A **model registry** gives artifacts names and monotonically increasing versions and maps **stages** or aliases (staging, production) to versions; deploying or rolling back is just moving a pointer, and artifacts themselves are immutable and content-addressed. MLflow, Weights & Biases and cloud ML platforms provide these as services; the program in the Framework section builds a minimal one from files to show there is no magic.",
      },
      { t: "h", text: "Cloud deployment, scalability and cost" },
      {
        t: "list",
        items: [
          "**Deployment targets**: managed endpoints (cloud ML platforms), containers on Kubernetes or serverless container services, serverless functions for small models, batch jobs for offline scoring, and edge devices.",
          "**Horizontal scaling**: stateless replicas behind a load balancer; **autoscaling** on CPU/GPU utilisation, request rate or queue length; keep a minimum number of warm replicas because model loading makes cold starts slow.",
          "**Cost levers**: right-size instances (measure throughput per replica first), batch requests on GPUs, use spot/preemptible capacity for training and batch scoring, compress models ([quantization, pruning, distillation](/learn/m18/model-optimization)), cache repeated predictions, scale to zero for rarely used models, and turn off idle development resources.",
          "**Cost per prediction** = instance cost per hour / predictions per hour at the target utilisation — the number to optimise and to monitor.",
        ],
      },
    ],
    math: [
      { t: "p", text: "Let the reference sample define $K$ bins (quantile edges, so each holds about $1/K$ of the reference). Let $r_k$ and $c_k$ be the fractions of the reference and current samples in bin $k$, floored at a small $\\epsilon$ so that empty bins do not produce $\\log 0$." },
      { t: "math", tex: String.raw`\mathrm{PSI} = \sum_{k=1}^{K} (c_k - r_k)\,\ln\frac{c_k}{r_k} = \mathrm{KL}(c\,\|\,r) + \mathrm{KL}(r\,\|\,c)`, caption: "Population Stability Index: the symmetrised (Jeffreys) KL divergence between the binned distributions." },
      { t: "math", tex: String.raw`\mathrm{KL}(c\,\|\,r) = \sum_{k=1}^{K} c_k \ln\frac{c_k}{r_k}`, caption: "KL divergence in nats; asymmetric, ≥ 0, = 0 iff the binned distributions match." },
      { t: "p", text: "Each PSI term is non-negative because $c_k - r_k$ and $\\ln(c_k/r_k)$ always have the same sign. A widely used industry rule of thumb (a convention from credit scoring, not a statistical test) reads PSI < 0.1 as stable, 0.1–0.25 as a moderate shift and > 0.25 as a major shift." },
      { t: "p", text: "The **two-sample Kolmogorov–Smirnov statistic** compares empirical cumulative distribution functions $F_a(t) = \\frac{1}{n}\\#\\{i : a_i \\le t\\}$ and $F_b$ without binning:" },
      { t: "math", tex: String.raw`D = \sup_t \left|F_a(t) - F_b(t)\right|,\qquad \text{reject "same distribution" at level } \alpha \text{ if } D > c(\alpha)\sqrt{\frac{n+m}{n\,m}}`, caption: "n, m: sample sizes; c(0.05) ≈ 1.358 for large samples." },
      { t: "p", text: "The critical value shrinks like $1/\\sqrt{n}$, so with large samples any non-zero difference becomes significant — which is why monitoring should threshold $D$ or PSI (effect sizes) rather than p-values." },
      { t: "p", text: "**Drift taxonomy.** Writing the joint distribution two ways, $P(x, y) = P(y\\mid x)P(x) = P(x\\mid y)P(y)$: data (covariate) drift changes $P(x)$ with $P(y\\mid x)$ fixed; concept drift changes $P(y\\mid x)$; label (prior) shift changes $P(y)$ with $P(x\\mid y)$ fixed. A model approximates $P(y\\mid x)$, so under pure covariate drift it stays correct where it was trained, but may be wrong in regions it never saw." },
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
      { t: "p", text: "PSI = 0.228 sits in the \"moderate shift\" band of the rule of thumb. Note that KL is asymmetric (0.106 vs 0.122) while PSI is symmetric." },
    ],
    visual: [
      { t: "sim", id: "distributions", caption: "Drift is a change of distribution: shift the mean or widen the spread and compare the new shape with the original — the same changes our drift statistics measure." },
      { t: "sim", id: "entropy", caption: "KL divergence between two distributions — PSI is the sum of KL in both directions over histogram bins." },
      {
        t: "list",
        items: [
          "In the distributions simulator, move the mean by a quarter of a standard deviation and then by a full one: our program measured PSI 0.093 and 0.939 for these shifts — the first is barely visible, the second obvious.",
          "Widen the spread without moving the mean: the CDFs cross, which KS detects (D = 0.108 for sd 1 → 1.5) even though the mean is unchanged — a mean-only monitor would miss it.",
          "In the entropy simulator, make one distribution put almost no mass where the other has a lot and watch KL explode: this is why empty histogram bins must be floored at a small ε.",
          "Swap the two distributions and note that KL changes while PSI (the sum of both directions) would not.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "PSI, KL and the KS statistic from scratch, validated against `scipy.stats.ks_2samp`, on five drift scenarios; the effect of sample size on p-values; and a concept-drift simulation where input statistics stay flat while accuracy collapses.",
      },
      { t: "code", lang: "python", title: "drift_stats.py", code: DRIFT_CODE, output: DRIFT_OUT },
      {
        t: "list",
        items: [
          "Our KS statistic matches SciPy's to three decimals in every scenario; SciPy additionally provides the p-value.",
          "PSI orders the scenarios sensibly: no drift 0.007, mean shift +0.25 sd 0.093 (below the 0.1 band), variance change 0.194, +1 sd 0.939.",
          "Imputing 10% missing values with 0 creates a spike in the middle bins (PSI 0.104): broken upstream pipelines look like drift, and are a common real cause of alerts.",
          "Sample size: a 0.03-sd shift is \"not significant\" with 1,000 samples (p = 0.80) and \"highly significant\" with 100,000 (p = 3.4e-06) while PSI says it is negligible. PSI with only 1,000 samples is itself noisy (0.0213 vs 0.0008): small windows need larger thresholds.",
          "Concept drift: input PSI ≈ 0.005 on both features — no input monitor would fire — yet accuracy fell from 1.000 to 0.599. The share of positive predictions (0.502 vs a true rate of 0.393) is not observable without labels either; only label feedback reveals the problem.",
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
              { t: "p", text: "Monitoring a real PyTorch digits classifier with label-free signals — PSI on a brightness summary feature, on the confidence distribution and on the predicted-class mix — over four simulated production weeks, compared with the true accuracy we would only learn later." },
              { t: "code", lang: "python", title: "monitor_torch.py", code: MON_CODE, output: MON_OUT },
              {
                t: "list",
                items: [
                  "Week 2 (dimmer scanner): the largest input drift (PSI 8.39) yet accuracy is unchanged (0.967) — this model is nearly invariant to a global intensity scale. An alert should trigger investigation, not automatic retraining.",
                  "Week 3 (sensor noise): moderate input drift, confidence PSI 0.211 and a real accuracy drop to 0.922 — here the confidence monitor is the useful early warning.",
                  "Week 4 (label shift: only digits 0–4): inputs look almost normal (0.146) but the predicted-class PSI is 2.906. Accuracy is fine; the business may still care (the downstream population changed).",
                  "Every alert has a different meaning: monitoring needs several signals and a human-readable dashboard, not one number.",
                ],
              },
            ],
          },
          {
            label: "Model registry",
            blocks: [
              { t: "p", text: "A tiny experiment tracker and model registry built from JSON files and content-addressed `.npz` artifacts: log a hyperparameter sweep, register versions, promote through a gate evaluated on a held-out set that was not used for model selection, and roll back by moving the stage pointer." },
              { t: "code", lang: "python", title: "file_registry.py", code: REG_CODE, output: REG_OUT },
              { t: "p", text: "Design points worth copying: artifacts are immutable and named by their hash, so a version can never silently change; the registry file is replaced atomically (`os.replace`); each run records a data fingerprint so a model can be traced to the exact training data; rollback is instant because it only changes which version the production stage points to." },
            ],
          },
          {
            label: "CI tests",
            blocks: [
              { t: "p", text: "Data and model tests in pytest style (each `test_*` function is collected by pytest; a small runner makes the file run standalone here)." },
              { t: "code", lang: "python", title: "test_model.py", code: TEST_CODE, output: TEST_OUT },
              { t: "p", text: "The mutation check exposes something we found while writing this lesson: forgetting the `/16` input scaling kept top-1 predictions on 15 of 16 golden inputs (0.938), and in an earlier draft of this program the whole-test-set accuracy without scaling was still 0.958, comfortably above the 0.90 threshold; yet probabilities moved by up to 0.55. Golden-output tests on probabilities catch preprocessing regressions that accuracy thresholds miss." },
              { t: "p", text: "A GitHub Actions workflow that runs these tests on every pull request and builds the serving image on main:" },
              { t: "code", lang: "text", title: ".github/workflows/ml-ci.yml", code: GHA_YAML },
            ],
          },
          {
            label: "MLflow",
            blocks: [
              { t: "p", text: "MLflow provides experiment tracking, a model format and a model registry with versions and aliases. It is not installed in this environment, so the snippet uses only its core, well-known calls and is not executed." },
              { t: "code", lang: "python", title: "mlflow_tracking.py (not executed in CI)", code: MLFLOW_CODE, output: "Logs three runs with their C parameter and validation accuracy to the local ./mlruns store, registers three versions of digits-classifier, then loads version 3 and prints five predicted digit labels." },
            ],
          },
          {
            label: "Weights & Biases",
            blocks: [
              { t: "p", text: "Weights & Biases is a hosted experiment-tracking service with dashboards, sweeps and artifact versioning. Not installed here; requires an account. Not executed." },
              { t: "code", lang: "python", title: "wandb_tracking.py (not executed in CI)", code: WANDB_CODE, output: "Streams per-epoch loss and accuracy to a W&B project dashboard and uploads weights.pt as a versioned model artifact linked to the run." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Drift statistic", "Data type", "Advantages", "Disadvantages"],
        rows: [
          ["PSI", "Binned numeric or categorical", "Symmetric, interpretable bands, cheap to compute from stored histograms", "Depends on binning; needs ε for empty bins; noisy for small windows"],
          ["KL divergence", "Binned / densities", "Information-theoretic meaning", "Asymmetric; infinite when the reference has an empty bin"],
          ["Jensen–Shannon divergence", "Binned", "Symmetric and bounded (≤ ln 2)", "Less sensitive to small shifts"],
          ["Two-sample KS", "Continuous, 1-D", "No binning, a p-value", "Over-sensitive at large n; one feature at a time"],
          ["Chi-square test", "Categorical", "Standard test for counts", "Same large-n issue as KS"],
          ["Wasserstein (earth mover's) distance", "Numeric", "Measured in the feature's units", "No fixed threshold scale"],
          ["Classifier two-sample test (train a model to tell reference from current)", "Multivariate", "Catches joint (multivariate) changes", "Heavier; needs a held-out evaluation"],
        ],
      },
      {
        t: "table",
        head: ["Kind of change", "What changes", "Detectable without labels?", "Typical response"],
        rows: [
          ["Data (covariate) drift", "P(x)", "Yes — input statistics", "Investigate; retrain with new data if accuracy suffers"],
          ["Label (prior) shift", "P(y)", "Partly — predicted-class mix", "Re-weight or recalibrate thresholds"],
          ["Concept drift", "P(y | x)", "No — needs labels", "Retrain on recent labelled data"],
          ["Upstream data bug", "Schema, units, missing values", "Yes — data tests, PSI spikes", "Fix the pipeline, not the model"],
        ],
      },
      {
        t: "table",
        head: ["Rollout strategy", "How", "Pros", "Cons"],
        rows: [
          ["Big bang", "Replace all replicas", "Simple", "All users exposed to a bad model"],
          ["Blue/green", "Two full environments, switch traffic", "Instant rollback", "Double capacity during switch"],
          ["Canary", "Small % of traffic to the new version", "Live comparison, limited risk", "Needs per-version metrics"],
          ["Shadow", "New version gets copies of traffic, answers discarded", "Zero user risk", "Cannot measure user-facing outcomes"],
          ["A/B test", "Randomised split, measure business metric", "Causal evidence", "Needs traffic and time for significance"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "Credit scoring is where PSI originated as a standard stability metric for scorecards, and regulators expect lenders to monitor model stability over time.",
          "Fraud detection faces concept drift by design: fraudsters adapt to the model, so models are retrained frequently on recently labelled cases.",
          "Recommendation and ranking systems change their own training data through feedback loops (users only click what was shown), so they are monitored with online A/B tests rather than offline accuracy alone.",
          "Open-source and managed tools — MLflow, Weights & Biases, DVC for data versioning, Evidently and similar libraries for drift reports, and the model registries of cloud ML platforms — implement the components built from scratch in this lesson.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "table",
        head: ["Decision", "Benefit", "Cost / risk"],
        rows: [
          ["Low drift thresholds", "Early warnings", "Alert fatigue; people start ignoring alerts"],
          ["Short monitoring windows", "Fast detection", "Noisy statistics (PSI 0.0213 at n = 1,000 vs 0.0008 at n = 100,000 for the same shift)"],
          ["Automatic retraining on drift", "No human bottleneck", "Can learn from corrupted data or feedback loops; needs strong gates"],
          ["Logging full inputs", "Complete debugging and retraining data", "Storage cost, privacy and compliance obligations"],
          ["Managed ML platform", "Less infrastructure work", "Vendor lock-in, cost at scale"],
          ["Self-hosted tooling", "Control, portability", "Engineering and maintenance effort"],
        ],
      },
      {
        t: "p",
        text: "The overall trade-off is operational maturity against speed: a notebook model deployed by hand is fast to ship and impossible to maintain, a full platform is slow to build. A pragmatic order is: version everything and log predictions first (cheap, enables everything else), then add data/model tests in CI, then drift dashboards, then automated retraining once the gates are trusted.",
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Alerting on p-values.** With large traffic every shift is significant (p = 3.4e-06 for a 0.03-sd shift here). Fix: threshold effect sizes (PSI, KS D) and require persistence across windows.",
          "**Assuming no input drift means the model is fine.** Concept drift leaves inputs unchanged (PSI 0.005 while accuracy fell to 0.599). Fix: collect labels (even a small audited sample) and track real accuracy.",
          "**Assuming drift means the model is broken.** The dimmer-scanner week had PSI 8.39 and unchanged accuracy. Fix: treat drift as a trigger for investigation, check confidence and labels before retraining.",
          "**Evaluating a promotion gate on the validation set used for model selection.** The winner's validation score is optimistically biased. Fix: a separate gate/test set, as in our registry program.",
          "**Recomputing bin edges from the current window.** Quantile bins of the current data always look uniform and hide drift. Fix: freeze bin edges from the reference and add open-ended outer bins.",
          "**Not versioning data and preprocessing.** A model hash alone cannot reproduce a prediction. Fix: record code commit, data fingerprint, config and artifact hash for every run.",
          "**Mutable model files (\"model_latest.pt\").** You cannot know what is deployed or roll back. Fix: immutable, content-addressed artifacts and registry stages that point to versions.",
          "**Testing only aggregate accuracy in CI.** Preprocessing bugs can keep top-1 accuracy (our missing-scaling bug). Fix: golden-output tests on probabilities, invariance tests, data tests.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between data drift and concept drift?", a: "Data (covariate) drift is a change in the input distribution P(x) — for example a new camera produces darker images — while the relationship between inputs and labels stays the same. Concept drift is a change in P(y | x): the same input now has a different correct answer, for example a fraud pattern that used to be benign. Data drift can be detected from inputs alone; concept drift requires labels." },
      { t: "qa", level: "beginner", q: "What should you monitor for a deployed model?", a: "Operational metrics (latency percentiles, throughput, error rates, resource usage); input data quality and drift (schema, missing values, per-feature distributions); output distributions (confidence, predicted-class mix); model version per request; and, when labels arrive, real performance overall and per segment. Plus business metrics the model is meant to move." },
      { t: "qa", level: "beginner", q: "What is a model registry?", a: "A catalogue of trained model artifacts with names, immutable versions and metadata (metrics, training data, code version), plus stages or aliases such as staging and production that point to specific versions. Deployment and rollback become changing a pointer; every prediction can be traced to the exact model that produced it." },
      { t: "qa", level: "intermediate", q: "How is PSI computed and how do you interpret it?", a: "Bin the reference data (usually 10 quantile bins with open outer bins), compute the fraction of reference and current data in each bin (floored at a small epsilon), and sum (c − r)·ln(c/r). It equals KL(c‖r) + KL(r‖c). A common rule of thumb: below 0.1 stable, 0.1–0.25 moderate shift, above 0.25 significant shift — but thresholds should be tuned to window size and business impact, since small windows give noisy PSI." },
      { t: "qa", level: "intermediate", q: "Why can a KS test be misleading for drift monitoring?", a: "Its p-value depends on sample size: the critical value of D shrinks like 1/sqrt(n), so with large traffic tiny, irrelevant shifts are flagged as significant, while with small windows real shifts may not be. It is also univariate, so correlated changes across features can be missed. Use D itself as an effect size with tuned thresholds, combine with PSI, and use multivariate methods (e.g. a domain classifier) when needed." },
      { t: "qa", level: "intermediate", q: "What tests belong in CI for an ML project?", a: "Code unit tests; data tests (schema, value ranges, missing values, class balance, duplicates/leakage between splits); model tests (output shape and valid probabilities, ability to overfit a tiny batch, accuracy above a threshold on a fixed set, invariance/directional tests, determinism, golden outputs on fixed inputs to catch preprocessing changes); and serving tests (API schema, error codes, batch vs single consistency, latency budget)." },
      { t: "qa", level: "advanced", q: "Labels for your model arrive 90 days late. How do you detect performance degradation earlier?", a: "Use label-free proxies: input drift per feature and multivariate (domain classifier), output drift (confidence distribution, predicted-class mix, entropy), and disagreement between the production model and a reference or shadow model. Estimate performance under covariate shift with importance weighting or confidence-based estimators if the model is well calibrated. Collect a small fast-labelled audit sample (human review) for direct accuracy. Segment metrics by key populations, and set alert thresholds using historical windows where performance was later known." },
      { t: "qa", level: "advanced", q: "Design a safe automated retraining pipeline.", a: "Trigger on schedule or sustained drift. Pull a versioned data snapshot, run data validation (schema, ranges, label sanity, leakage checks) and stop on failure. Train with fixed config and log to a tracker. Evaluate on a fixed, held-out gate set and on recent labelled data, per segment, with latency/size budgets; compare with production. Register the candidate as a new immutable version. Deploy to staging, then shadow or canary with live metric comparison and automatic rollback criteria. Promote by moving the registry alias; keep the previous version warm for instant rollback. Guard against feedback loops (the model influencing its own training labels) and poisoned data by limiting the weight of recent data and requiring human approval for large changes." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Input feature distributions are unchanged but accuracy has dropped. This is most likely…", options: ["data drift", "concept drift", "a latency problem", "label smoothing"], answer: 1, explain: "P(y|x) changed; input statistics cannot see it (our simulation: PSI 0.005, accuracy 0.599)." },
          { kind: "numeric", q: "Two bins: reference (0.5, 0.5), current (0.3, 0.7). Compute PSI (3 decimals).", answer: 0.169, tolerance: 0.002, explain: "(−0.2)·ln(0.6) + (0.2)·ln(1.4) = 0.102165 + 0.067294 = 0.169459." },
          { kind: "numeric", q: "Empirical CDFs at four points: F_a = (0.2, 0.5, 0.8, 1.0), F_b = (0.1, 0.3, 0.7, 1.0). What is the KS statistic D?", answer: 0.2, tolerance: 0.0001, explain: "max |difference| = max(0.1, 0.2, 0.1, 0) = 0.2." },
          { q: "Why are bin edges for PSI taken from the reference window?", options: ["It is faster", "Edges from the current window would make the current histogram uniform by construction and hide drift", "KL requires it", "To avoid negative values"], answer: 1, explain: "Quantile bins of the current data always contain equal fractions of the current data." },
          { q: "In our monitoring experiment, which week had huge input drift but no accuracy loss?", options: ["Normal traffic", "Dimmer scanner", "Sensor noise", "Only digits 0–4"], answer: 1, explain: "PSI(brightness) 8.39, accuracy 0.967 vs 0.969 reference." },
          { kind: "numeric", q: "A GPU instance costs 3.00 USD per hour and serves 600 predictions per second at target utilisation. What is the cost per million predictions in USD? (2 decimals)", answer: 1.39, tolerance: 0.01, explain: "600 × 3600 = 2.16 M predictions/hour; 3.00 / 2.16 = 1.39 USD per million." },
          { q: "Which rollout sends copies of live traffic to the new model without using its answers?", options: ["Canary", "Blue/green", "Shadow", "Big bang"], answer: 2, explain: "Shadow deployments compare outputs with zero user risk." },
        ],
      },
      {
        t: "exercise",
        title: "PSI and KL by hand",
        prompt: "Reference bin fractions r = (0.2, 0.3, 0.3, 0.2); current c = (0.1, 0.2, 0.3, 0.4). Compute (a) each PSI term and PSI, (b) KL(c‖r) and KL(r‖c), and check PSI = their sum. (c) What does the rule of thumb say?",
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
          { t: "p", text: "(c) PSI ≈ 0.248 is at the top of the 0.1–0.25 \"moderate shift\" band: investigate, and check whether it persists over several windows and whether confidence or (eventually) accuracy moved." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a multivariate drift detector",
        prompt: "Per-feature PSI misses changes in correlations. Implement a **classifier two-sample test**: label reference rows 0 and current rows 1, train a logistic regression with cross-validation, and report the ROC AUC. AUC ≈ 0.5 means the windows are indistinguishable. Test it on 2-D Gaussian data where the current window keeps both marginals N(0, 1) but changes the correlation from 0 to 0.8.",
        hint: "`sklearn.model_selection.cross_val_score(LogisticRegression(), X, y, scoring='roc_auc', cv=5)`. A linear model cannot see a correlation change from raw features — add the product feature x1·x2.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n\nrng = np.random.default_rng(0)\nref = rng.normal(0, 1, (5000, 2))\ncov = np.array([[1.0, 0.8], [0.8, 1.0]])\ncur = rng.multivariate_normal([0, 0], cov, 5000)    # same marginals, new correlation\n\nX = np.vstack([ref, cur])\ny = np.r_[np.zeros(5000), np.ones(5000)]\nfeats = np.c_[X, X[:, 0] * X[:, 1]]                 # the product feature exposes correlation\nauc = cross_val_score(LogisticRegression(), feats, y, scoring=\"roc_auc\", cv=5).mean()\nprint(\"domain-classifier AUC:\", round(auc, 3))",
          },
          { t: "p", text: "Per-feature PSI stays near 0 because both marginals are still N(0, 1), while the domain classifier's AUC rises clearly above 0.5 because the product feature has mean 0 in the reference and 0.8 in the current window. Without the product feature a linear classifier stays at AUC ≈ 0.5 — a flexible classifier (e.g. gradient-boosted trees) avoids hand-crafting such features." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Data drift: $P(x)$ changes; concept drift: $P(y\\mid x)$ changes (needs labels); label shift: $P(y)$ changes.",
          "PSI $= \\sum_k (c_k - r_k)\\ln(c_k/r_k) = \\mathrm{KL}(c\\|r) + \\mathrm{KL}(r\\|c)$ on reference-quantile bins; rule of thumb < 0.1 / 0.1–0.25 / > 0.25.",
          "KS $D = \\sup_t |F_a - F_b|$; threshold effect sizes, not p-values (large n makes everything significant).",
          "Monitor inputs, confidence, predicted-class mix, operations, and real accuracy when labels arrive; drift ≠ degradation (dimmer scanner) and no drift ≠ health (concept drift).",
          "CI: data tests, model behaviour tests, golden outputs; CD: immutable images and registered versions; gated promotion on held-out data; canary/shadow rollouts; rollback = move the stage pointer.",
          "Version code, data, config and artifacts together; registries map stages to immutable, content-addressed versions (MLflow, W&B, cloud platforms, or a few JSON files).",
          "Scale stateless replicas with autoscaling and warm minimums; optimise cost per prediction (right-sizing, batching, spot capacity, compression, caching).",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Adversarial Examples, Robustness and AI Safety](/learn/m19/robustness-safety) — distribution shift taken to the extreme: inputs crafted to fool the model.",
          "[Train/Validation/Test Splits and Cross-Validation](/learn/m17/data-splits) — honest evaluation underlies every gate and alert threshold.",
          "[Quantization, Pruning and Distillation](/learn/m18/model-optimization) — the main lever for cost per prediction.",
        ],
      },
    ],
  },
};

export default lesson;
