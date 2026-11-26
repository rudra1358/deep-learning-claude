import type { Lesson } from "@/lib/types";

const NUMPY_CODE = String.raw`import hashlib
import json
import os
import pickle
import tempfile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- 1. train a tiny model
# Softmax regression on 8x8 digits (64 pixel features, 10 classes), trained with plain
# gradient descent. The model itself is not the point: the point is everything around it.
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

mean = X_tr.mean(axis=0)              # preprocessing statistics are PART of the model:
std = X_tr.std(axis=0) + 1e-6         # they must be saved and shipped with the weights

rng = np.random.default_rng(0)
W = rng.normal(0, 0.01, size=(64, 10))
b = np.zeros(10)
Z_tr = (X_tr - mean) / std
Y = np.eye(10)[y_tr]
LR, EPOCHS = 0.5, 200
for _ in range(EPOCHS):
    logits = Z_tr @ W + b
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    g = (p - Y) / len(Z_tr)            # gradient of mean cross-entropy wrt logits
    W -= LR * (Z_tr.T @ g)
    b -= LR * g.sum(axis=0)
acc = ((((X_te - mean) / std) @ W + b).argmax(axis=1) == y_te).mean()
print(f"trained: test accuracy {acc:.4f}")

# ---------------------------------------------------------------- 2. serialise SAFELY
# Arrays go into an .npz (pure data, no code); metadata goes into JSON.
# A checksum of the array bytes lets the loader detect a corrupted or swapped file.
workdir = tempfile.mkdtemp()
weights_path = os.path.join(workdir, "model.npz")
arrays = {"W": W, "b": b, "mean": mean, "std": std}
np.savez(weights_path, **arrays)
digest = hashlib.sha256(b"".join(arrays[k].tobytes() for k in sorted(arrays))).hexdigest()
meta = {
    "name": "digits-softmax",
    "version": "1.0.0",
    "input_shape": [64],
    "input_range": [0, 16],
    "classes": list(range(10)),
    "sha256": digest,
}
with open(os.path.join(workdir, "model.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"saved {sorted(arrays)}  sha256 {digest[:16]}...")


# ---------------------------------------------------------------- 3. an inference pipeline
class DigitsPredictor:
    """Loads the artifact once; serves preprocess -> model -> postprocess."""

    def __init__(self, folder):
        with open(os.path.join(folder, "model.json")) as f:
            self.meta = json.load(f)
        with np.load(os.path.join(folder, "model.npz"), allow_pickle=False) as z:   # refuse pickled objects
            self.arr = {k: z[k] for k in z.files}
        check = hashlib.sha256(b"".join(self.arr[k].tobytes() for k in sorted(self.arr))).hexdigest()
        if check != self.meta["sha256"]:
            raise ValueError("checksum mismatch: artifact corrupted or swapped")

    def preprocess(self, batch):
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim == 1:                                   # accept a single example too
            x = x[None, :]
        if x.shape[1:] != tuple(self.meta["input_shape"]):
            raise ValueError(f"expected shape (n, 64), got {x.shape}")
        lo, hi = self.meta["input_range"]
        if x.min() < lo or x.max() > hi:
            raise ValueError(f"pixel values must lie in [{lo}, {hi}]")
        return (x - self.arr["mean"]) / self.arr["std"]

    def model(self, z):
        return z @ self.arr["W"] + self.arr["b"]

    def postprocess(self, logits):
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        top = p.argmax(axis=1)
        return [{"label": int(k), "confidence": round(float(p[i, k]), 4)} for i, k in enumerate(top)]

    def predict(self, batch):
        return self.postprocess(self.model(self.preprocess(batch)))


server = DigitsPredictor(workdir)          # a "fresh process": only the files are shared
print("first three predictions:", server.predict(X_te[:3]))
print("true labels:            ", y_te[:3].tolist())

# batching must not change answers: one call with 450 rows == 450 calls with 1 row
batched = [r["label"] for r in server.predict(X_te)]
single = [server.predict(x)[0]["label"] for x in X_te]
print("batched == one-by-one:", batched == single, f"| served accuracy {np.mean(np.array(batched) == y_te):.4f}")

for bad in (np.zeros(10), np.full(64, 99.0)):
    try:
        server.predict(bad)
    except ValueError as e:
        print("rejected:", e)

# ---------------------------------------------------------------- 4. why pickle is unsafe
# pickle stores *instructions* for rebuilding an object. __reduce__ lets an object choose
# any callable to run at load time. Here it is a harmless print; an attacker would pick os.system.
class Payload:
    def __reduce__(self):
        return (print, ("!! this line was executed by pickle.loads, not by our code !!",))


blob = pickle.dumps(Payload())
print("loading an untrusted pickle ...")
pickle.loads(blob)`;
const NUMPY_OUT = String.raw`trained: test accuracy 0.9667
saved ['W', 'b', 'mean', 'std']  sha256 31f3afb3a7a4fc41...
first three predictions: [{'label': 2, 'confidence': 0.8101}, {'label': 0, 'confidence': 0.9796}, {'label': 4, 'confidence': 0.9996}]
true labels:             [2, 0, 4]
batched == one-by-one: True | served accuracy 0.9667
rejected: expected shape (n, 64), got (1, 10)
rejected: pixel values must lie in [0, 16]
loading an untrusted pickle ...
!! this line was executed by pickle.loads, not by our code !!`;
const TORCH_CODE = String.raw`import os
import pickle
import tempfile
import warnings

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")          # silence TorchScript deprecation/tracer warnings (they go to stderr)
torch.manual_seed(0)
torch.set_num_threads(1)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.25, stratify=y, random_state=0)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


class MLP(nn.Module):
    def __init__(self, hidden=64, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(100):                                 # full-batch Adam: 100 steps is enough here
    opt.zero_grad()
    nn.functional.cross_entropy(model(X_tr), y_tr).backward()
    opt.step()

# ---------------------------------------------------------------- 1. train mode vs eval mode
model.train()
with torch.no_grad():
    a, b = model(X_te[:1]), model(X_te[:1])
print("train mode, same input twice -> identical?", torch.equal(a, b))     # dropout is random
model.eval()
with torch.no_grad():
    a, b = model(X_te[:1]), model(X_te[:1])
    acc = (model(X_te).argmax(1) == y_te).float().mean().item()
print("eval mode,  same input twice -> identical?", torch.equal(a, b))
print(f"test accuracy {acc:.4f}")

# no_grad: no autograd graph is recorded, so activations are not kept for backward
out_grad = model(X_te[:1])
with torch.no_grad():
    out_nograd = model(X_te[:1])
print("requires_grad with / without no_grad:", out_grad.requires_grad, out_nograd.requires_grad)

work = tempfile.mkdtemp()
ref = model(X_te).detach()

# ---------------------------------------------------------------- 2. state_dict (recommended)
torch.save(model.state_dict(), os.path.join(work, "weights.pt"))
fresh = MLP()                                            # the CODE must exist on the loading side
fresh.load_state_dict(torch.load(os.path.join(work, "weights.pt"), weights_only=True))
fresh.eval()
print("state_dict keys:", list(fresh.state_dict().keys()))
print("state_dict round trip exact:", torch.equal(fresh(X_te).detach(), ref))

# ---------------------------------------------------------------- 3. whole-model pickle
torch.save(model, os.path.join(work, "whole.pt"))       # pickles the class *reference* + tensors
try:
    torch.load(os.path.join(work, "whole.pt"))           # default weights_only=True refuses arbitrary objects
except pickle.UnpicklingError:
    print("torch.load(whole model) with weights_only=True -> UnpicklingError (refused)")
whole = torch.load(os.path.join(work, "whole.pt"), weights_only=False)   # only for files you trust
print("whole-model round trip exact:", torch.equal(whole(X_te).detach(), ref))

# ---------------------------------------------------------------- 4. TorchScript: trace vs script
class Gate(nn.Module):
    """Data-dependent control flow: the branch depends on the input VALUES."""

    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return x - 100


pos, neg = torch.ones(1, 3), -torch.ones(1, 3)
traced = torch.jit.trace(Gate(), pos)                    # records the ops run for THIS example
scripted = torch.jit.script(Gate())                      # compiles the Python source, keeps the if
print("eager   (neg):", Gate()(neg).tolist())
print("traced  (neg):", traced(neg).tolist(), " <- branch frozen at trace time")
print("scripted(neg):", scripted(neg).tolist())

ts_model = torch.jit.trace(model, X_te[:1])               # a feed-forward MLP has no branches: trace is safe
ts_model.save(os.path.join(work, "model_ts.pt"))
loaded = torch.jit.load(os.path.join(work, "model_ts.pt"))   # loadable without the MLP class source
print("TorchScript round trip max |diff|:", (loaded(X_te) - ref).abs().max().item())

# ---------------------------------------------------------------- 5. torch.export (the newer path)
ep = torch.export.export(model, (X_te[:1],))
torch.export.save(ep, os.path.join(work, "model.pt2"))
ep2 = torch.export.load(os.path.join(work, "model.pt2"))
print("torch.export round trip max |diff|:", (ep2.module()(X_te[:1]) - ref[:1]).abs().max().item())`;
const TORCH_OUT = String.raw`train mode, same input twice -> identical? False
eval mode,  same input twice -> identical? True
test accuracy 0.9711
requires_grad with / without no_grad: True False
state_dict keys: ['net.0.weight', 'net.0.bias', 'net.3.weight', 'net.3.bias']
state_dict round trip exact: True
torch.load(whole model) with weights_only=True -> UnpicklingError (refused)
whole-model round trip exact: True
eager   (neg): [[-101.0, -101.0, -101.0]]
traced  (neg): [[-2.0, -2.0, -2.0]]  <- branch frozen at trace time
scripted(neg): [[-101.0, -101.0, -101.0]]
TorchScript round trip max |diff|: 0.0
torch.export round trip max |diff|: 7.152557373046875e-07`;
const FASTAPI_CODE = String.raw`import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sklearn.datasets import load_digits

torch.manual_seed(0)
torch.set_num_threads(1)

# ---------------------------------------------------------------- the model (trained quickly here;
# in production you would load a saved state_dict at startup instead)
X, y = load_digits(return_X_y=True)
Xt, yt = torch.tensor(X[:1500] / 16.0, dtype=torch.float32), torch.tensor(y[:1500])
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(100):
    opt.zero_grad()
    nn.functional.cross_entropy(model(Xt), yt).backward()
    opt.step()
model.eval()                     # once, at startup
MODEL_VERSION = "digits-mlp-1.0.0"
MAX_BATCH = 64


# ---------------------------------------------------------------- request / response schemas
class PredictRequest(BaseModel):
    # each instance: 64 pixel intensities in [0, 16]; pydantic validates types automatically
    instances: list[list[float]] = Field(min_length=1)


class Prediction(BaseModel):
    label: int
    confidence: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


app = FastAPI(title="digits classifier")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.instances) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH} instances per request")
    if any(len(row) != 64 for row in req.instances):
        raise HTTPException(status_code=422, detail="every instance needs exactly 64 values")
    x = torch.tensor(req.instances, dtype=torch.float32)
    if x.min() < 0 or x.max() > 16:
        raise HTTPException(status_code=422, detail="pixel values must lie in [0, 16]")
    with torch.no_grad():                                   # preprocess -> model -> postprocess
        probs = torch.softmax(model(x / 16.0), dim=1)
    conf, label = probs.max(dim=1)
    return {
        "model_version": MODEL_VERSION,
        "predictions": [{"label": int(l), "confidence": round(float(c), 4)} for l, c in zip(label, conf)],
    }


# ---------------------------------------------------------------- exercise the API in-process
# TestClient sends real HTTP requests through the ASGI app without opening a network port.
client = TestClient(app)
print("GET /health ->", client.get("/health").json())

r = client.post("/predict", json={"instances": X[1500:1503].tolist()})
print("POST /predict ->", r.status_code, r.json())
print("true labels    ->", y[1500:1503].tolist())

cases = {
    "wrong length": {"instances": [[0.0] * 10]},
    "out of range": {"instances": [[99.0] * 64]},
    "not numbers": {"instances": [["a"] * 64]},
    "empty batch": {"instances": []},
    "batch too big": {"instances": [[0.0] * 64] * 65},
}
for name, body in cases.items():
    print(f"{name:14s} -> HTTP {client.post('/predict', json=body).status_code}")

# served accuracy on the held-out rows, sent in batches of MAX_BATCH
correct = 0
rest = X[1500:]
for i in range(0, len(rest), MAX_BATCH):
    preds = client.post("/predict", json={"instances": rest[i:i + MAX_BATCH].tolist()}).json()["predictions"]
    correct += sum(p["label"] == t for p, t in zip(preds, y[1500 + i:1500 + i + MAX_BATCH]))
print(f"accuracy through the API on {len(rest)} held-out digits: {correct / len(rest):.4f}")`;
const FASTAPI_OUT = String.raw`GET /health -> {'status': 'ok', 'model_version': 'digits-mlp-1.0.0'}
POST /predict -> 200 {'model_version': 'digits-mlp-1.0.0', 'predictions': [{'label': 1, 'confidence': 0.5967}, {'label': 7, 'confidence': 0.9987}, {'label': 4, 'confidence': 1.0}]}
true labels    -> [1, 7, 4]
wrong length   -> HTTP 422
out of range   -> HTTP 422
not numbers    -> HTTP 422
empty batch    -> HTTP 422
batch too big  -> HTTP 413
accuracy through the API on 297 held-out digits: 0.9091`;

const ONNX_CODE = String.raw`import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)).eval()
example = torch.randn(1, 64)

# Export: run the model once on an example and record the graph in the ONNX format.
# dynamic_axes marks dimension 0 as variable so any batch size is accepted later.
torch.onnx.export(
    model, (example,), "digits.onnx",
    input_names=["pixels"], output_names=["logits"],
    dynamic_axes={"pixels": {0: "batch"}, "logits": {0: "batch"}},
)

# Inference with ONNX Runtime: no PyTorch needed on the serving side.
sess = ort.InferenceSession("digits.onnx", providers=["CPUExecutionProvider"])
x = np.random.rand(8, 64).astype(np.float32)
(logits,) = sess.run(["logits"], {"pixels": x})
print(logits.shape)
print(np.abs(logits - model(torch.from_numpy(x)).detach().numpy()).max())`;

const FLASK_CODE = String.raw`import torch
import torch.nn as nn
from flask import Flask, jsonify, request

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
model.load_state_dict(torch.load("weights.pt", weights_only=True))
model.eval()
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    rows = body.get("instances")
    # Flask does no validation for you: every check is hand-written
    if not isinstance(rows, list) or not rows or any(len(r) != 64 for r in rows):
        return jsonify(error="instances must be a non-empty list of 64-value rows"), 422
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(rows, dtype=torch.float32) / 16.0), dim=1)
    conf, label = probs.max(dim=1)
    return jsonify(predictions=[{"label": int(l), "confidence": float(c)} for l, c in zip(label, conf)])


if __name__ == "__main__":
    app.run(port=8000)          # development server only; use gunicorn/waitress in production`;

const DOCKERFILE = String.raw`# Dockerfile — a CPU image for the FastAPI service
FROM python:3.12-slim

# 1. dependencies first: this layer is cached until requirements.txt changes
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. then code and the model artifact (weights + metadata)
COPY app/ ./app/
COPY model/ ./model/

# 3. run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# uvicorn serves the ASGI app; several worker processes use several CPU cores
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# build and run:
#   docker build -t digits-api:1.0.0 .
#   docker run -p 8000:8000 digits-api:1.0.0`;

const TF_CODE = String.raw`import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(64,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10),
])
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.rand(256, 64).astype("float32")
y = np.random.randint(0, 10, 256)
model.fit(x, y, epochs=1, verbose=0)

model.save("digits.keras")                      # native Keras format: architecture + weights + optimizer
model.export("digits_savedmodel")               # SavedModel for TF Serving / TFLite conversion

reloaded = tf.keras.models.load_model("digits.keras")
print(np.abs(reloaded.predict(x[:4], verbose=0) - model.predict(x[:4], verbose=0)).max())`;

const lesson: Lesson = {
  id: "m18/serving",
  title: "Serialization, Inference Pipelines and APIs",
  summary:
    "How a trained network becomes a service: saving weights safely (state_dict vs pickle), TorchScript trace vs script, torch.export and ONNX, an inference pipeline with eval mode, no_grad and batching, a validated FastAPI endpoint tested in-process, Flask, Docker images and GPU serving — every step run and measured.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m01/dl-workflow", "m03/multilayer-networks", "m07/regularization", "m07/normalization"],
  next: ["m18/model-optimization", "m18/mlops"],
  terms: ["model-serialization", "state-dict", "pickle", "torchscript", "onnx", "inference-pipeline", "rest-api", "dynamic-batching", "docker-image", "latency", "throughput"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Training ends with numbers in memory. A **deployed model** is a file that another process (often on another machine) can copy, version and load, wrapped in code that turns requests into tensors and tensors into answers, behind an interface (usually HTTP) other software can call. This lesson walks that path: **serialization** (writing the model to disk), **export formats** that don't need the original Python class, the **inference pipeline** (preprocess → model → postprocess), **REST APIs** with FastAPI and Flask, **containers** with Docker, and what changes on a **GPU**.",
      },
      {
        t: "p",
        text: "Why? Training and serving environments differ. Training is a script with a GPU, a data loader and an optimizer. Serving needs a long-running process that loads once, answers thousands of independent requests, rejects bad inputs, and behaves the same on every machine. Most ML production incidents come not from the network but from this glue: preprocessing that differs from training, a model left in training mode, an unsafe file format, or a changed dependency version.",
      },
      {
        t: "callout",
        kind: "key",
        text: "A served model = **artifact** (weights + preprocessing constants + metadata) + **pipeline code** (validate → preprocess → model in eval mode under no_grad → postprocess) + **interface** (an HTTP API with a schema) + **environment** (a pinned container image).",
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[The deep learning workflow](/learn/m01/dl-workflow) — where deployment sits in a project's life cycle.",
          "[Multilayer networks](/learn/m03/multilayer-networks) — we save and serve plain MLPs, so know what their parameters are.",
          "[Regularization](/learn/m07/regularization) — dropout behaves differently in training and evaluation, which is why `model.eval()` matters.",
          "[Normalization](/learn/m07/normalization) — batch norm uses running statistics at inference; the same eval-mode switch controls it.",
          "Basic Python packaging and HTTP: what a JSON request body and a status code (200, 422, 500) are.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A recipe, a frozen meal and a restaurant",
        text: "A **state_dict** is a list of ingredient amounts: useless without the recipe (the model class). A **pickled whole model** is a box saying \"run these instructions to rebuild me\" — handy, but you run whatever is inside. An **exported graph** (TorchScript, torch.export, ONNX) is a frozen meal: recipe and ingredients sealed together, heatable in any compatible microwave (runtime). The **API** is the restaurant counter: customers order from a fixed menu (the schema), and off-menu orders are refused at the counter instead of breaking the kitchen.",
      },
      {
        t: "p",
        text: "The key shift: at serving time, the model is a pure function from checked input to output. Anything random or stateful is a bug — dropout left on, batch-norm statistics updating, gradients recorded, preprocessing constants recomputed from the request. Everything the function needs (weights *and* the training data's normalization mean and std) must be saved together and loaded once.",
      },
    ],
    mechanism: [
      {
        t: "steps",
        items: [
          { title: "1. Train and evaluate", text: "Produce weights. Record the exact preprocessing (scaling constants, vocabulary, image size) as data, not as code scattered across a notebook." },
          { title: "2. Serialize the artifact", text: "Save parameters as tensors (`state_dict`, `.npz`, safetensors) and metadata as JSON: version, input shape and range, class names, a checksum. Avoid formats that run code on load." },
          { title: "3. Optionally export a graph", text: "TorchScript (`trace` records the operations run on an example; `script` compiles the Python source, control flow included), `torch.export` (the newer whole-graph capture) or ONNX (a framework-neutral graph run by ONNX Runtime, TensorRT, OpenVINO…). Exported graphs load without the original class." },
          { title: "4. Load once at startup", text: "Read the artifact, verify the checksum, build the model, call `model.eval()`, move it to the device. Never load per request." },
          { title: "5. Validate the request", text: "Check types, shapes, ranges and batch size before touching the model. Reply 4xx for client errors instead of crashing with 500." },
          { title: "6. Preprocess", text: "Apply exactly the training-time transform, with the saved constants." },
          { title: "7. Run the model", text: "Inside `torch.no_grad()` (or `torch.inference_mode()`), in eval mode, on a batch. Batching spreads the per-call overhead over many inputs; results must not depend on how requests were grouped." },
          { title: "8. Postprocess and respond", text: "Softmax → label + confidence, map indices to class names, include the model version, serialize to JSON." },
          { title: "9. Package and run", text: "Pin dependencies in a container image. Run several worker processes (CPU) or one process per GPU with request batching. Expose health and readiness endpoints for the orchestrator." },
        ],
      },
      {
        t: "h",
        text: "state_dict vs whole-model pickle",
      },
      {
        t: "p",
        text: "`torch.save(model.state_dict())` stores an ordered mapping from parameter names (`net.0.weight`) to tensors. To load it, you build the same class first and call `load_state_dict`, which also checks that names and shapes match.",
      },
      {
        t: "p",
        text: "`torch.save(model)` pickles the module object: the class's *import path* plus instructions to rebuild it. So the file breaks when code is moved or renamed — and, worse, unpickling can call any Python function. Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`: it rebuilds only tensors and simple containers and refuses other objects (our program shows this). Load whole pickles only from trusted sources.",
      },
      {
        t: "h",
        text: "trace vs script",
      },
      {
        t: "p",
        text: "`torch.jit.trace(model, example)` runs the model once and records the tensor operations that happened. Python control flow runs during tracing, so an `if` that depends on tensor values gets frozen to the branch the example took. Our program shows a traced gate returning the wrong branch for a negative input. `torch.jit.script` compiles the source (a subset of Python) and keeps the branch.",
      },
      {
        t: "p",
        text: "Here (PyTorch 2.15 on Python 3.14) both functions warn that TorchScript is not supported on Python 3.14+ and recommend `torch.compile` or `torch.export`. TorchScript still runs in many existing deployments, so understand it, but new projects should prefer `torch.export` or ONNX.",
      },
    ],
    math: [
      {
        t: "p",
        text: "**Latency** $\\ell$ is the time for one request, from arrival to response. **Throughput** $X$ is requests finished per second. If the model processes a batch of $B$ inputs in time $t(B)$, a common first model is a fixed overhead plus a cost per item:",
      },
      { t: "math", tex: String.raw`t(B) = t_0 + B\,t_1, \qquad X(B) = \frac{B}{t(B)} = \frac{B}{t_0 + B t_1} \;\xrightarrow{B\to\infty}\; \frac{1}{t_1}`, caption: "t₀: overhead per call (kernel launches, Python, memory transfer); t₁: extra cost per example." },
      {
        t: "p",
        text: "In words: bigger batches pay the overhead $t_0$ once for many inputs, so throughput climbs toward $1/t_1$. The catch: requests may wait for the batch to fill. With **dynamic batching** the server waits at most $w_{max}$, so worst-case latency is about $w_{max} + t(B_{max})$. **Little's law** links averages in any stable system: requests in the system $L$ equals arrival rate $\\lambda$ times mean time spent there $W$:",
      },
      { t: "math", tex: String.raw`L = \lambda W`, caption: "L: requests in flight; λ: arrivals per second; W: mean time per request (seconds)." },
      {
        t: "p",
        text: "Memory sets the minimum hardware. A model with $P$ parameters stored at $b$ bytes each needs $P\\,b$ bytes for weights alone, plus activations that grow with batch size.",
      },
      { t: "math", tex: String.raw`M_{weights} = P \cdot b`, caption: "b = 4 for float32, 2 for float16/bfloat16, 1 for int8." },
      { t: "h", text: "Worked example" },
      {
        t: "steps",
        items: [
          { title: "Say a model costs t₀ = 5 ms per call and t₁ = 0.5 ms per example.", tex: String.raw`t(1) = 5.5\text{ ms} \Rightarrow X(1) = \frac{1}{0.0055} = 181.8\ \text{req/s}` },
          { title: "Batch of 16", tex: String.raw`t(16) = 5 + 16\cdot0.5 = 13\text{ ms} \Rightarrow X(16) = \frac{16}{0.013} = 1230.8\ \text{req/s}` },
          { title: "Upper limit", tex: String.raw`X(\infty) = 1/t_1 = 1/0.0005 = 2000\ \text{req/s}` },
          { title: "Latency cost with a 10 ms batching window", tex: String.raw`\ell_{worst} \approx 10 + 13 = 23\text{ ms}\ \text{instead of}\ 5.5\text{ ms}` },
          { title: "Little's law at λ = 800 req/s and W = 20 ms", tex: String.raw`L = 800 \times 0.020 = 16\ \text{requests in flight}` },
          { title: "Weights of a 25.6 M-parameter network (ResNet-50 size)", tex: String.raw`25.6\times10^6 \times 4\ \text{B} = 102.4\ \text{MB (fp32)},\quad 51.2\ \text{MB (fp16)},\quad 25.6\ \text{MB (int8)}` },
        ],
      },
      {
        t: "p",
        text: "Here batching buys 6.8× more throughput at the price of up to ~4× worse worst-case latency. Which matters depends on the product. An interactive API has a latency budget (often a 95th or 99th percentile); an offline scoring job only cares about throughput.",
      },
    ],
    visual: [
      { t: "sim", id: "network", caption: "Serving is a forward pass: watch the input flow through the layers to the output — exactly what the /predict endpoint runs, with no backward pass." },
      {
        t: "list",
        items: [
          "Step through the forward pass. Every number depends only on the input and the fixed weights — the determinism a served model must keep (no dropout, no statistics updates).",
          "Change the input: only activations change, never parameters. Inference is read-only, which is why `no_grad` can skip autograd's bookkeeping.",
          "Count the multiply-adds per layer. That per-example cost is $t_1$ in the latency model; the per-call overhead is $t_0$.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "A full serving path in NumPy: train softmax regression on scikit-learn's 8×8 digits, save weights and preprocessing constants as `.npz` plus JSON metadata with a SHA-256 checksum, and load them in a predictor class that validates, preprocesses, predicts and postprocesses. We check batching doesn't change answers, and show why unpickling untrusted data is dangerous (with a harmless `print` payload).",
      },
      { t: "code", lang: "python", title: "serving_numpy.py", code: NUMPY_CODE, output: NUMPY_OUT },
      {
        t: "p",
        text: "Notes: `allow_pickle=False` makes `np.load` refuse object arrays, so the `.npz` can't carry code. The checksum covers the array bytes, not the zip file (whose metadata can vary). The pickle demo prints a line during `pickle.loads` although our code never called `print` there — a real payload would call `os.system` instead.",
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              { t: "p", text: "Eval vs train mode, `no_grad`, state_dict vs whole-model saving (and the `weights_only` refusal), TorchScript trace vs script on a data-dependent branch, and `torch.export`." },
              { t: "code", lang: "python", title: "serving_torch.py", code: TORCH_CODE, output: TORCH_OUT },
              { t: "p", text: "The traced gate returns `-2` for a negative input because tracing recorded the `x * 2` branch taken by the positive example; the scripted version keeps the `if`. The torch.export round trip differs by ~5e-7 because the exported graph may order float operations differently — normal float32 rounding, not a bug." },
            ],
          },
          {
            label: "FastAPI",
            blocks: [
              { t: "p", text: "A REST service with a pydantic schema, explicit validation, a batch limit and a health endpoint. `TestClient` sends real HTTP requests through the app in-process, so no port is opened — the same tool you'd use in CI tests." },
              { t: "code", lang: "python", title: "api_fastapi.py", code: FASTAPI_CODE, output: FASTAPI_OUT },
              { t: "p", text: "Run it as a real server with `uvicorn api_fastapi:app --port 8000` (module name = file name). Status codes: 422 for bad input (pydantic produces it for wrong types and the empty list; our checks for wrong length and range), 413 for an oversized batch. Held-out accuracy (0.9091) is lower than a random split would give because the dataset's last rows come from different writers — a small, real case of distribution shift." },
            ],
          },
          {
            label: "ONNX",
            blocks: [
              { t: "p", text: "ONNX is an open graph format. ONNX Runtime runs it on CPU, CUDA, TensorRT, DirectML and other back ends, without PyTorch. The `onnx` and `onnxruntime` packages aren't installed here, so this program is shown but not run." },
              { t: "code", lang: "python", title: "export_onnx.py (not executed in CI)", code: ONNX_CODE, output: "Prints the output shape (8, 10) and the maximum absolute difference between ONNX Runtime and PyTorch logits, which should be at float32 rounding level (around 1e-6 or smaller)." },
            ],
          },
          {
            label: "Flask",
            blocks: [
              { t: "p", text: "Flask is a synchronous WSGI micro-framework with no built-in request validation, so you write every check by hand. Flask isn't installed here." },
              { t: "code", lang: "python", title: "api_flask.py (not executed in CI)", code: FLASK_CODE, output: "Starts a development server on port 8000; POST /predict with a JSON body returns predictions, malformed bodies return HTTP 422." },
            ],
          },
          {
            label: "Docker",
            blocks: [
              { t: "p", text: "A container image freezes the OS libraries, Python version, packages and model files. The service then behaves the same on a laptop, a CI runner and a cloud VM." },
              { t: "code", lang: "text", title: "Dockerfile", code: DOCKERFILE },
              { t: "p", text: "Pin exact versions in `requirements.txt` (e.g. `torch==…`, `fastapi==…`). For GPU images, start from a CUDA runtime base image whose CUDA version matches your PyTorch build, and run the container with GPU access (`docker run --gpus all …`, with the NVIDIA container toolkit installed on the host)." },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "code", lang: "python", title: "keras_save.py (not executed in CI)", code: TF_CODE, output: "Prints the maximum absolute difference between the reloaded and original model's predictions, which should be 0 or at float32 rounding level." },
              { t: "p", text: "`model.save(\"x.keras\")` is the native format for reloading in Keras. `model.export()` writes a SavedModel, the input for TensorFlow Serving and the TFLite converter." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Format / option", "Needs model code to load?", "Pros", "Cons", "Typical use"],
        rows: [
          ["PyTorch state_dict (`torch.save` of tensors)", "Yes", "Small, safe with `weights_only=True`, survives refactoring", "Must rebuild the class with matching names", "Checkpoints, Python serving"],
          ["Whole-model pickle", "Yes (import path)", "One call to save/load", "Runs code on load; breaks when code moves", "Quick experiments only"],
          ["safetensors / .npz", "Yes", "Pure data, no code execution, memory-mappable (safetensors)", "Weights only", "Sharing weights publicly"],
          ["TorchScript (trace / script)", "No", "Runs from C++ (libtorch), no Python class needed", "Trace freezes control flow; deprecated direction on Python 3.14+", "Legacy mobile/C++ deployments"],
          ["torch.export (.pt2)", "No", "Whole-graph capture, basis for newer compilers", "Code must be traceable; newer API", "New PyTorch deployments"],
          ["ONNX", "No", "Framework-neutral; many optimized runtimes", "Operator coverage/version issues; extra conversion step", "Cross-platform and accelerator serving"],
          ["Keras / SavedModel", "No (SavedModel)", "TF Serving, TFLite conversion", "TensorFlow ecosystem only", "TF production stacks"],
        ],
      },
      {
        t: "table",
        head: ["Serving style", "Latency", "Throughput", "When"],
        rows: [
          ["Online REST/gRPC, one request at a time", "Lowest per request", "Low", "Small models, low traffic"],
          ["Online with dynamic batching", "+ batching window", "High", "GPU serving, high traffic"],
          ["Batch (offline) scoring", "Minutes–hours", "Highest", "Nightly scores, embeddings, reports"],
          ["Streaming (queue consumer)", "Seconds", "High", "Event pipelines (Kafka etc.)"],
          ["On-device (mobile, browser, edge)", "No network hop", "One user", "Privacy, offline use; needs small models (see model optimization)"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "Model servers such as NVIDIA Triton Inference Server, TorchServe and TensorFlow Serving implement this pipeline generically: versioned model repositories, dynamic batching, health endpoints and metrics.",
          "ONNX Runtime runs exported models on CPUs and accelerators from a single artifact, including inside desktop and mobile apps.",
          "FastAPI is a common choice for wrapping Python models in internal microservices, because request schemas double as validation and as auto-generated OpenAPI docs.",
          "Containers orchestrated by Kubernetes (or managed cloud equivalents) are the standard way to scale such services out and roll out new model versions.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "table",
        head: ["Choice", "Gain", "Cost"],
        rows: [
          ["Larger batches", "Throughput, better GPU use", "Tail latency, memory"],
          ["Exported graph instead of Python", "No Python class needed, can be optimized", "Export fails on dynamic code; another artifact to test"],
          ["GPU instead of CPU", "Much faster for large models and batches", "Cost, cold-start time, needs batching to stay busy"],
          ["Strict input validation", "Clear 4xx errors, no silent garbage", "More code; must match training assumptions"],
          ["Many small worker processes", "Uses all CPU cores, isolates crashes", "Each holds its own copy of the weights in memory"],
        ],
      },
      {
        t: "p",
        text: "GPU deployment points: GPU time is expensive and wasted unless busy, so GPU services almost always batch. CPU↔GPU transfers add latency, so preprocessing on the GPU can help. One model per GPU process avoids memory contention. Half precision (fp16/bf16) halves weight memory and is usually accurate enough for inference. CUDA, driver and PyTorch versions must be compatible. The first call is slow (CUDA context creation, kernel selection), so warm up before marking the service ready.",
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting `model.eval()`.** Dropout stays on and batch norm uses per-batch statistics, so outputs are random and depend on other requests in the batch. Our program shows two identical inputs giving different outputs in train mode. Fix: call `eval()` once after loading.",
          "**Recomputing preprocessing constants at serving time.** Normalizing a request by its own mean/std is a different transform from training. Fix: save the training mean/std (or tokenizer vocabulary) inside the artifact.",
          "**Loading untrusted pickles.** `pickle.loads`, `torch.load(..., weights_only=False)` and `np.load(..., allow_pickle=True)` can run code. Fix: share weights as tensors/safetensors and load with `weights_only=True`.",
          "**Tracing models with data-dependent control flow.** The traced graph silently keeps one branch. Fix: `torch.jit.script`, `torch.export` with explicit control-flow ops, or restructure the code; test the exported model on inputs that hit every branch.",
          "**Loading the model inside the request handler.** Every request pays seconds of disk I/O. Fix: load once at startup (module level or a lifespan hook).",
          "**No input validation.** A wrong shape becomes a 500 error with a stack trace — or worse, a silently wrong prediction. Fix: a schema plus explicit shape/range checks returning 4xx.",
          "**Unpinned dependencies.** A new library version changes numerics or breaks loading. Fix: pinned requirements and an immutable container image tag per release.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between saving a state_dict and saving the whole model in PyTorch?", a: "A state_dict maps parameter/buffer names to tensors. To load it, you build the model class and call load_state_dict, which checks names and shapes. Saving the whole model pickles the module object, including its class's import path. The state_dict is recommended: it survives code refactoring and loads safely with weights_only=True. A pickle breaks when the class moves and can run arbitrary code when loaded." },
      { t: "qa", level: "beginner", q: "Why must you call model.eval() and use torch.no_grad() at inference?", a: "eval() switches layers that behave differently in training and inference: dropout stops dropping units, and batch norm uses running mean/variance instead of batch statistics, so outputs are deterministic and batch-independent. no_grad() stops autograd from recording the graph and saving activations, cutting memory and time. They are independent: eval doesn't disable gradients, and no_grad doesn't change layer behaviour." },
      { t: "qa", level: "beginner", q: "What are the stages of an inference pipeline?", a: "Validate the request (types, shape, range, batch size); preprocess exactly as in training with saved constants; run the model in eval mode without gradients, usually on a batch; postprocess (softmax, top-k, thresholding, mapping indices to class names); return a response with the model version. The artifact is loaded once, before any request." },
      { t: "qa", level: "intermediate", q: "Explain trace vs script in TorchScript and when each fails.", a: "Tracing runs the model on an example input and records the tensor operations executed. Python control flow and non-tensor logic run once and get baked in, so data-dependent branches, or loops whose length depends on the input, are silently frozen. Scripting compiles the Python source (a restricted subset) into TorchScript IR and keeps control flow, but fails on unsupported Python features or dynamic typing. Trace straight-line models like CNNs and MLPs, script models with control flow, and test either on inputs that hit every path." },
      { t: "qa", level: "intermediate", q: "Why is pickle unsafe and what should you use instead?", a: "Pickle is a small program for rebuilding objects. An object's __reduce__ can name any callable plus arguments to run on load, so loading a malicious file can run os.system or anything else with the process's permissions. Use data-only formats: tensors via torch.save(state_dict) loaded with weights_only=True, safetensors, .npz with allow_pickle=False, or exported graphs (ONNX) — and verify a checksum against a trusted registry." },
      { t: "qa", level: "intermediate", q: "How does dynamic batching trade latency against throughput?", a: "With per-call overhead t0 and per-item cost t1, throughput B/(t0 + B t1) rises with batch size toward 1/t1. Dynamic batching holds incoming requests until a maximum wait or a maximum batch size, then runs them together. Larger limits raise throughput and GPU use but add queueing delay, raising tail latency. You tune them against a latency SLO (e.g. p99 under 50 ms) at the expected request rate." },
      { t: "qa", level: "advanced", q: "Your exported ONNX model gives slightly different outputs from PyTorch. How do you decide whether it is a bug?", a: "Compare on a representative batch. Differences at float32 rounding level (around 1e-6 relative) come from different operation order or fused kernels and are expected. Larger differences point to real problems: opset differences in an operator's meaning, a model exported in training mode (dropout/batch norm), control flow frozen by tracing, undeclared dynamic axes so another batch size is mishandled, or preprocessing differences outside the graph. Check top-1 agreement and max logit difference over many inputs, bisect by exporting submodules, and add the comparison as a CI test." },
      { t: "qa", level: "advanced", q: "Design the serving setup for a 1 GB vision model receiving 500 requests/s with a p99 latency target of 100 ms.", a: "Export the model (TorchScript/torch.export/ONNX/TensorRT), possibly in fp16, and serve it on GPUs with a model server that supports dynamic batching. Measure t(B) to pick a maximum batch size and wait (e.g. a few ms) so wait + t(Bmax) stays well under 100 ms at p99 while the GPU stays busy. From measured per-GPU throughput, compute replicas for 500 req/s plus headroom. Autoscale on queue length or GPU use, put preprocessing (decode/resize) on separate CPU workers or the GPU, warm up replicas before marking them ready, and monitor latency percentiles, error rates and input statistics. Roll out new versions with canary traffic." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which call is the safest way to load weights from a file you downloaded?", options: ["torch.load(path, weights_only=False)", "pickle.load(open(path, 'rb'))", "model.load_state_dict(torch.load(path, weights_only=True))", "np.load(path, allow_pickle=True)"], answer: 2, explain: "weights_only=True rebuilds only tensors and simple containers; the other options can run code hidden in the file." },
          { q: "A traced model gives wrong outputs for some inputs but the eager model is correct. The most likely cause is…", options: ["float32 rounding", "data-dependent control flow frozen at trace time", "the model was saved with a state_dict", "no_grad was used"], answer: 1, explain: "Tracing records only the branch taken by the example input." },
          { q: "What does model.eval() change?", options: ["It disables gradient computation", "It switches dropout and batch norm to inference behaviour", "It converts the model to int8", "It compiles the model"], answer: 1, explain: "Gradients are controlled separately, by no_grad / inference_mode." },
          { kind: "numeric", q: "A model costs 5 ms per call plus 0.5 ms per example. What is its throughput in requests per second at batch size 16? (1 decimal)", answer: 1230.8, tolerance: 0.2, explain: "16 / (0.005 + 16·0.0005) = 16 / 0.013 = 1230.8." },
          { kind: "numeric", q: "How many megabytes do the float16 weights of a 25.6 M-parameter model take (1 MB = 10⁶ bytes)?", answer: 51.2, tolerance: 0.05, explain: "25.6e6 × 2 bytes = 51.2e6 bytes." },
          { kind: "numeric", q: "A service receives 400 requests/s and each spends 50 ms in the system on average. How many requests are in flight on average (Little's law)?", answer: 20, tolerance: 0.01, explain: "L = λW = 400 × 0.05 = 20." },
          { q: "Which HTTP status code fits a request whose JSON body has the wrong number of features?", options: ["200", "422", "500", "302"], answer: 1, explain: "422 Unprocessable Entity: the JSON was well-formed but failed validation. 500 would blame the server." },
        ],
      },
      {
        t: "exercise",
        title: "Batching arithmetic",
        prompt: "A GPU model has t(B) = 8 ms + 0.25 ms·B. (a) Throughput at B = 1, 32, 128. (b) The upper limit. (c) With a 5 ms batching window and B_max = 32, what is the approximate worst-case latency? (d) How many replicas are needed for 6000 req/s at B = 32 with 30% headroom?",
        hint: "X(B) = B / t(B). Headroom: plan for 1.3 × 6000 req/s.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a)", tex: String.raw`X(1) = \tfrac{1}{0.00825} = 121.2,\quad X(32) = \tfrac{32}{0.016} = 2000,\quad X(128) = \tfrac{128}{0.040} = 3200\ \text{req/s}` },
              { title: "(b)", tex: String.raw`X(\infty) = 1/0.00025 = 4000\ \text{req/s}` },
              { title: "(c)", tex: String.raw`\ell_{worst} \approx 5 + 16 = 21\text{ ms}` },
              { title: "(d)", tex: String.raw`\lceil 1.3 \times 6000 / 2000 \rceil = \lceil 3.9 \rceil = 4\ \text{replicas}` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Coding: add a batch endpoint test and a model-version header",
        prompt: "Extend `api_fastapi.py`: (1) add a test that sending the same 10 images as one request and as ten single-image requests gives identical labels; (2) return the model version in an HTTP response header `X-Model-Version`; (3) make `/predict` reject NaN values with 422.",
        hint: "Take a `fastapi.Response` parameter to set headers. JSON can't carry NaN by default, but Python's json module emits `NaN` — check with `math.isfinite`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "import math\nfrom fastapi import Response\n\n@app.post(\"/predict\", response_model=PredictResponse)\ndef predict(req: PredictRequest, response: Response):\n    if any(not math.isfinite(v) for row in req.instances for v in row):\n        raise HTTPException(status_code=422, detail=\"values must be finite\")\n    ...  # existing checks and inference\n    response.headers[\"X-Model-Version\"] = MODEL_VERSION\n    return {...}\n\n# test\nbatch = client.post(\"/predict\", json={\"instances\": X[1500:1510].tolist()}).json()[\"predictions\"]\nsingle = [client.post(\"/predict\", json={\"instances\": [row]}).json()[\"predictions\"][0] for row in X[1500:1510].tolist()]\nassert [p[\"label\"] for p in batch] == [p[\"label\"] for p in single]\nassert client.post(\"/predict\", json={\"instances\": [row]}).headers[\"X-Model-Version\"] == MODEL_VERSION",
          },
          { t: "p", text: "The batch test catches bugs where one example's output depends on the others (batch norm in train mode, padding mistakes). Labels should match exactly. Confidences may differ in the last float digit, because batched matrix multiplications can sum in a different order." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Artifact = weights + preprocessing constants + metadata (version, input schema, checksum). Save tensors (state_dict, .npz, safetensors), not pickled objects.",
          "`torch.load` defaults to `weights_only=True`; whole-model pickles need `weights_only=False` and must come from trusted sources.",
          "TorchScript trace records executed ops (freezes control flow); script compiles source. Prefer `torch.export` or ONNX for new deployments.",
          "Pipeline: validate → preprocess (saved constants) → model in `eval()` under `no_grad()` → postprocess → JSON with model version. Batching must not change answers.",
          "FastAPI: pydantic schemas validate automatically (422), and `TestClient` tests the app in-process. Flask needs manual validation.",
          "Latency model $t(B) = t_0 + Bt_1$; throughput $B/t(B) \\to 1/t_1$; Little's law $L = \\lambda W$; weight memory $P\\cdot b$.",
          "Docker pins the environment. GPUs need batching, warm-up, compatible CUDA versions and usually fp16.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Quantization, Pruning and Distillation](/learn/m18/model-optimization) — make the served model smaller and faster, and measure the accuracy cost.",
          "[Monitoring, Drift, CI/CD and Versioning](/learn/m18/mlops) — keep the deployed model healthy: detect drift, test and version every release.",
          "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — where the artifacts you deploy come from.",
        ],
      },
    ],
  },
};

export default lesson;
