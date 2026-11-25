import type { Lesson } from "@/lib/types";

const TAPE = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ===================== a tiny GradientTape in NumPy =====================
class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    # every operator goes through _op so the active tape can record it
    def __add__(self, other): return _op("add", self, as_tensor(other))
    def __mul__(self, other): return _op("mul", self, as_tensor(other))
    def __matmul__(self, other): return _op("matmul", self, as_tensor(other))
    def __pow__(self, k): return _op("pow", self, k=k)


class Variable(Tensor):
    """Trainable state. Like tf.Variable, it is watched by a tape automatically."""

    def assign_sub(self, delta):
        self.value -= delta


def as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def unbroadcast(g, shape):
    """Sum a gradient back down to shape (undo NumPy broadcasting, e.g. a bias added to every row)."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


# forward rule and vector-Jacobian product (VJP) rule of every primitive op
def _forward(name, xs, k):
    if name == "add": return xs[0] + xs[1]
    if name == "mul": return xs[0] * xs[1]
    if name == "matmul": return xs[0] @ xs[1]
    if name == "pow": return xs[0] ** k
    if name == "relu": return np.maximum(xs[0], 0.0)
    if name == "sum": return xs[0].sum()
    if name == "xent": return softmax_xent(xs[0], k)[0]


def _vjp(name, xs, out, g, k):
    """Given g = dL/d(out), return dL/d(input) for each input."""
    if name == "add": return [unbroadcast(g, xs[0].shape), unbroadcast(g, xs[1].shape)]
    if name == "mul": return [unbroadcast(g * xs[1], xs[0].shape), unbroadcast(g * xs[0], xs[1].shape)]
    if name == "matmul": return [g @ xs[1].T, xs[0].T @ g]
    if name == "pow": return [g * k * xs[0] ** (k - 1)]
    if name == "relu": return [g * (xs[0] > 0)]
    if name == "sum": return [g * np.ones_like(xs[0])]
    if name == "xent": return [g * softmax_xent(xs[0], k)[1]]


def softmax_xent(logits, y):
    """Mean softmax cross-entropy and its gradient wrt the logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y]).mean()
    grad = p.copy(); grad[np.arange(len(y)), y] -= 1
    return loss, grad / len(y)


class GradientTape:
    _stack = []                                            # tapes currently recording (they can nest)

    def __init__(self, persistent=False):
        self.persistent, self.records, self.watched, self.used = persistent, [], set(), False

    def __enter__(self):
        GradientTape._stack.append(self)
        return self

    def __exit__(self, *exc):
        GradientTape._stack.remove(self)

    def watch(self, t):                                    # opt a non-Variable tensor in
        self.watched.add(id(t))

    def _is_watched(self, t):
        return isinstance(t, Variable) or id(t) in self.watched

    def gradient(self, target, sources):
        if self.used and not self.persistent:
            raise RuntimeError("a non-persistent tape can only compute gradients once")
        self.used = True
        grads = {id(target): np.ones_like(target.value)}   # dL/dL = 1
        for name, inputs, out, k in reversed(self.records):    # reverse order = reverse-mode AD
            g = grads.get(id(out))
            if g is None:                                  # this op does not influence the target
                continue
            for inp, gi in zip(inputs, _vjp(name, [t.value for t in inputs], out.value, g, k)):
                if self._is_watched(inp):                  # constants get no gradient
                    grads[id(inp)] = grads.get(id(inp), 0) + gi    # paths that meet ADD up
        return [grads.get(id(s)) for s in sources]         # None = unconnected or not watched


def _op(name, *inputs, k=None):
    out = Tensor(_forward(name, [t.value for t in inputs], k))
    for tape in GradientTape._stack:
        if any(tape._is_watched(t) for t in inputs):       # only record what touches watched tensors
            tape.records.append((name, inputs, out, k))
            tape.watched.add(id(out))
    return out


def relu(x): return _op("relu", x)
def reduce_sum(x): return _op("sum", x)
def sparse_softmax_xent(logits, y): return _op("xent", logits, k=y)


# ---------------- 1. scalar example, checked against finite differences ----------------
w = Variable(3.0)
x = Tensor(2.0)
with GradientTape(persistent=True) as tape:
    y = w * w * x + w                                      # y = x w^2 + w
grad_w, grad_x = tape.gradient(y, [w, x])
print("dy/dw:", grad_w, "(exact 2xw + 1 =", 2 * 2 * 3 + 1, ")   dy/dx:", grad_x, "(x is a constant, not watched)")

with GradientTape() as tape:
    tape.watch(x)                                          # now x is recorded too
    y = w * w * x + w
print("dy/dx after tape.watch(x):", tape.gradient(y, [x])[0], "(exact w^2 = 9)")
try:
    tape.gradient(y, [x])
except RuntimeError as e:
    print("second call:", e)

# ---------------- 2. matrix example: gradient check against finite differences ----------------
rng = np.random.default_rng(0)
W = Variable(rng.normal(size=(4, 3)))
b = Variable(np.zeros(3))
Xc = Tensor(rng.normal(size=(5, 4)))
with GradientTape() as tape:
    out = reduce_sum(relu(Xc @ W + b) ** 2)
gW, gb = tape.gradient(out, [W, b])
h, num = 1e-6, np.zeros_like(W.value)
for idx in np.ndindex(W.value.shape):
    W.value[idx] += h; fp = (np.maximum(Xc.value @ W.value + b.value, 0) ** 2).sum()
    W.value[idx] -= 2 * h; fm = (np.maximum(Xc.value @ W.value + b.value, 0) ** 2).sum()
    W.value[idx] += h
    num[idx] = (fp - fm) / (2 * h)
print(f"tape vs finite differences, max |diff| in dW: {np.abs(gW - num).max():.1e}  (db shape {gb.shape})")

# ---------------- 3. a custom training loop, TensorFlow style ----------------
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.2,
                                          random_state=0, stratify=digits.target)
W1 = Variable(rng.normal(0, np.sqrt(2 / 64), (64, 32))); b1 = Variable(np.zeros(32))
W2 = Variable(rng.normal(0, np.sqrt(2 / 32), (32, 10))); b2 = Variable(np.zeros(10))
params, lr = [W1, b1, W2, b2], 0.5


def model(X):
    return relu(Tensor(X) @ W1 + b1) @ W2 + b2


for epoch in range(1, 16):
    order = rng.permutation(len(X_tr))
    for start in range(0, len(order), 64):
        idx = order[start:start + 64]
        with GradientTape() as tape:                       # 1. record the forward pass
            loss = sparse_softmax_xent(model(X_tr[idx]), y_tr[idx])
        grads = tape.gradient(loss, params)                # 2. replay it backwards
        for p, g in zip(params, grads):                    # 3. apply the update
            p.assign_sub(lr * g)
    if epoch % 5 == 0:
        acc = np.mean(model(X_te).value.argmax(1) == y_te)
        print(f"epoch {epoch:2d}  last batch loss {float(loss.value):.4f}  test acc {acc:.3f}")`;

const TAPE_OUT = String.raw`dy/dw: 13.0 (exact 2xw + 1 = 13 )   dy/dx: None (x is a constant, not watched)
dy/dx after tape.watch(x): 9.0 (exact w^2 = 9)
second call: a non-persistent tape can only compute gradients once
tape vs finite differences, max |diff| in dW: 1.8e-10  (db shape (3,))
epoch  5  last batch loss 0.1149  test acc 0.900
epoch 10  last batch loss 0.1672  test acc 0.942
epoch 15  last batch loss 0.0874  test acc 0.958`;

const CALLBACKS = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ===================== callbacks: hooks the training loop calls =====================
class Callback:
    """Same hook names as keras.callbacks.Callback. self.model is set by fit()."""
    def on_train_begin(self, logs): pass
    def on_epoch_begin(self, epoch, logs): pass
    def on_epoch_end(self, epoch, logs): pass
    def on_train_end(self, logs): pass


class History(Callback):                                  # fit() always adds this one
    def on_train_begin(self, logs):
        self.history = {}

    def on_epoch_end(self, epoch, logs):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class EarlyStopping(Callback):
    def __init__(self, monitor="val_loss", patience=3, restore_best_weights=True):
        self.monitor, self.patience, self.restore = monitor, patience, restore_best_weights

    def on_train_begin(self, logs):
        self.best, self.wait, self.best_epoch, self.best_weights = np.inf, 0, 0, None

    def on_epoch_end(self, epoch, logs):
        if logs[self.monitor] < self.best:                # improvement: remember it, reset patience
            self.best, self.wait, self.best_epoch = logs[self.monitor], 0, epoch
            self.best_weights = [w.copy() for w in self.model.weights]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True           # fit() checks this flag after each epoch

    def on_train_end(self, logs):
        if self.restore and self.best_weights is not None:
            for w, best in zip(self.model.weights, self.best_weights):
                w[...] = best                             # copy in place: the model keeps its arrays
            print(f"EarlyStopping: restored weights from epoch {self.best_epoch} "
                  f"({self.monitor} {self.best:.4f})")


class LearningRateScheduler(Callback):
    def __init__(self, schedule):
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs):
        self.model.lr = self.schedule(epoch, self.model.lr)


class PrintEvery(Callback):                               # a custom callback: 6 lines of code
    def __init__(self, n):
        self.n = n

    def on_epoch_end(self, epoch, logs):
        if epoch % self.n == 0 or self.model.stop_training:
            print(f"epoch {epoch:2d}  " + "  ".join(f"{k} {v:.4f}" for k, v in logs.items()))


# ===================== a minimal Keras-like model with fit() =====================
class MLP:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)), np.zeros(n_hidden),
                        rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)), np.zeros(n_out)]
        self.lr, self.stop_training, self.rng = 0.0, False, rng

    def forward(self, X):
        W1, b1, W2, b2 = self.weights
        self.X, self.h = X, np.maximum(X @ W1 + b1, 0)
        return self.h @ W2 + b2

    def loss_and_grads(self, X, y):
        W1, b1, W2, b2 = self.weights
        z = self.forward(X)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        loss = -np.log(p[np.arange(len(y)), y]).mean()
        dz = p; dz[np.arange(len(y)), y] -= 1; dz /= len(y)
        dh = (dz @ W2.T) * (self.h > 0)
        return loss, [self.X.T @ dh, dh.sum(0), self.h.T @ dz, dz.sum(0)]

    def evaluate(self, X, y):
        z = self.forward(X)
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean(), np.mean(z.argmax(1) == y)

    def fit(self, X, y, epochs, batch_size, validation_data, callbacks=()):
        history = History()
        callbacks = [history, *callbacks]
        for cb in callbacks:
            cb.model = self                               # callbacks can read and change the model
            cb.on_train_begin({})
        for epoch in range(1, epochs + 1):
            for cb in callbacks:
                cb.on_epoch_begin(epoch, {})
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                _, grads = self.loss_and_grads(X[idx], y[idx])
                for w, g in zip(self.weights, grads):
                    w -= self.lr * g                      # plain SGD
            loss, acc = self.evaluate(X, y)
            val_loss, val_acc = self.evaluate(*validation_data)
            logs = {"loss": loss, "acc": acc, "val_loss": val_loss, "val_acc": val_acc, "lr": self.lr}
            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)
            if self.stop_training:
                break
        for cb in callbacks:
            cb.on_train_end({})
        return history


digits = load_digits()
X, y = digits.data / 16.0, digits.target
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_tr, y_tr = X_tr[:150], y_tr[:150]                       # only 150 training images: it WILL overfit

model = MLP(64, 128, 10)
history = model.fit(X_tr, y_tr, epochs=200, batch_size=16, validation_data=(X_val, y_val),
                    callbacks=[LearningRateScheduler(lambda epoch, lr: 0.2 if epoch <= 20 else 0.05),
                               EarlyStopping(monitor="val_loss", patience=5),
                               PrintEvery(5)])
print("epochs run:", len(history.history["loss"]),
      " best val_loss:", round(min(history.history["val_loss"]), 4))
print("after restore: val_loss %.4f  val_acc %.4f" % model.evaluate(X_val, y_val))`;

const CALLBACKS_OUT = String.raw`epoch  5  loss 0.5441  acc 0.9000  val_loss 0.7207  val_acc 0.8074  lr 0.2000
epoch 10  loss 0.2407  acc 0.9600  val_loss 0.4467  val_acc 0.8685  lr 0.2000
epoch 15  loss 0.1126  acc 1.0000  val_loss 0.3093  val_acc 0.9185  lr 0.2000
epoch 20  loss 0.0747  acc 1.0000  val_loss 0.2700  val_acc 0.9204  lr 0.2000
epoch 25  loss 0.0613  acc 1.0000  val_loss 0.2697  val_acc 0.9167  lr 0.0500
epoch 30  loss 0.0583  acc 1.0000  val_loss 0.2764  val_acc 0.9185  lr 0.0500
epoch 32  loss 0.0553  acc 1.0000  val_loss 0.2673  val_acc 0.9167  lr 0.0500
EarlyStopping: restored weights from epoch 27 (val_loss 0.2652)
epochs run: 32  best val_loss: 0.2652
after restore: val_loss 0.2652  val_acc 0.9204`;

const KERAS_APIS = String.raw`import tensorflow as tf
from tensorflow import keras

# ---- 1. Sequential API: a plain stack ----
seq = keras.Sequential([
    keras.Input(shape=(64,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),
])
seq.summary()                                          # Dense: 4160 + 650 = 4810 parameters


# ---- 2. Custom layer: subclass keras.layers.Layer ----
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):                      # called once, when the input size is known
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="w")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        self.scale = self.add_weight(shape=(self.units,), initializer="ones", name="scale")

    def call(self, x):
        return (tf.matmul(x, self.w) + self.b) * self.scale

    def get_config(self):                              # lets the layer be re-created when loading
        return {**super().get_config(), "units": self.units}


# ---- 3. Functional API: two inputs and a merge (not expressible as Sequential) ----
pixels = keras.Input(shape=(64,), name="pixels")
brightness = keras.Input(shape=(1,), name="brightness")
h = keras.layers.ReLU()(ScaledDense(64)(pixels))
h = keras.layers.Add()([h, keras.layers.Dense(64, activation="relu")(brightness)])
logits = keras.layers.Dense(10)(h)
model = keras.Model(inputs=[pixels, brightness], outputs=logits)
model.summary()                                        # lists both inputs and the Add merge`;

const KERAS_TAPE = String.raw`import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split((digits.data / 16.0).astype("float32"), digits.target,
                                          test_size=0.2, random_state=0, stratify=digits.target)
train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(X_tr), seed=0).batch(32)

model = keras.Sequential([keras.Input(shape=(64,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(10)])
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


@tf.function                                           # trace once into a graph, then run it fast
def train_step(x, y):
    with tf.GradientTape() as tape:                    # record the forward pass
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)          # replay it backwards
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


for epoch in range(1, 21):
    for x, y in train_ds:
        loss = train_step(x, y)
    if epoch % 5 == 0:
        acc = tf.reduce_mean(tf.cast(tf.argmax(model(X_te), axis=1) == y_te, tf.float32))
        print(f"epoch {epoch:2d}  last batch loss {float(loss):.4f}  test acc {float(acc):.3f}")`;

const KERAS_CALLBACKS = String.raw`import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = (digits.data / 16.0).astype("float32"), digits.target
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


class PrintEvery(keras.callbacks.Callback):            # a custom callback
    def __init__(self, n):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):          # Keras counts epochs from 0
        if (epoch + 1) % self.n == 0:
            print(f"epoch {epoch + 1}: val_loss {logs['val_loss']:.4f}")


model = keras.Sequential([keras.Input(shape=(64,)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10)])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
history = model.fit(
    X_tr, y_tr, epochs=200, batch_size=16, validation_data=(X_val, y_val), verbose=0,
    callbacks=[
        keras.callbacks.LearningRateScheduler(lambda epoch, lr: 0.2 if epoch < 20 else 0.05),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss", save_best_only=True),
        PrintEvery(5),
    ])
print("epochs run:", len(history.history["loss"]))

# ---- saving ----
model.save("digits.keras")                             # native Keras format: architecture + weights + optimizer
reloaded = keras.models.load_model("digits.keras")
tf.saved_model.save(model, "digits_savedmodel")        # TensorFlow SavedModel: a language-neutral graph
served = tf.saved_model.load("digits_savedmodel")      # loadable by TF Serving, TFLite converter, C++/Java`;

const TORCH_EQUIV = String.raw`import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

# ---- "Sequential API": a plain stack of layers ----
seq = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
print("Sequential params:", sum(p.numel() for p in seq.parameters()))


# ---- "Custom layer": subclass nn.Module, register weights as nn.Parameter ----
class ScaledDense(nn.Module):
    """Dense layer followed by a learnable per-unit scale (a small, made-up example layer)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.scale = nn.Parameter(torch.ones(n_out))      # trainable, found by .parameters()

    def forward(self, x):
        return self.lin(x) * self.scale


# ---- "Functional API" territory: a non-sequential graph (two inputs + a skip connection) ----
class TwoInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = ScaledDense(64, 64)
        self.extra = nn.Linear(1, 64)                     # second input: the image's mean brightness
        self.head = nn.Linear(64, 10)

    def forward(self, pixels, brightness):
        h = F.relu(self.body(pixels))
        h = h + F.relu(self.extra(brightness))            # merge the branches by addition
        return self.head(h)


model = TwoInputNet()
print("parameter names:", [n for n, _ in model.named_parameters()])

digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.2,
                                          random_state=0, stratify=digits.target)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=0, stratify=y_tr)
to_t = lambda a: torch.tensor(a, dtype=torch.float32)
X_tr, X_va, X_te = to_t(X_tr), to_t(X_va), to_t(X_te)
y_tr, y_va, y_te = torch.tensor(y_tr), torch.tensor(y_va), torch.tensor(y_te)
inputs = lambda X: (X, X.mean(dim=1, keepdim=True))       # build both inputs from the pixels

# ---- the explicit loop = Keras's GradientTape custom loop; early stopping written by hand ----
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best, best_state, wait, patience = float("inf"), None, 0, 3
for epoch in range(1, 101):
    model.train()
    perm = torch.randperm(len(X_tr))
    for s in range(0, len(perm), 32):
        idx = perm[s:s + 32]
        loss = F.cross_entropy(model(*inputs(X_tr[idx])), y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        val_loss = F.cross_entropy(model(*inputs(X_va)), y_va).item()
    if val_loss < best:                                   # = keras.callbacks.EarlyStopping logic
        best, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
    else:
        wait += 1
        if wait >= patience:
            break
model.load_state_dict(best_state)                         # = restore_best_weights=True
with torch.no_grad():
    test_acc = (model(*inputs(X_te)).argmax(1) == y_te).float().mean().item()
print(f"stopped after epoch {epoch}, best epoch {best_epoch} (val loss {best:.4f}), test acc {test_acc:.3f}")`;

const TORCH_EQUIV_OUT = String.raw`Sequential params: 4810
parameter names: ['body.scale', 'body.lin.weight', 'body.lin.bias', 'extra.weight', 'extra.bias', 'head.weight', 'head.bias']
stopped after epoch 32, best epoch 29 (val loss 0.1482), test acc 0.964`;

const PLATEAU_SOL = String.raw`class Callback:
    def on_epoch_end(self, epoch, logs): pass


class ReduceLROnPlateau(Callback):
    """Multiply the learning rate by factor after patience epochs without improvement."""

    def __init__(self, monitor="val_loss", factor=0.5, patience=2, min_lr=1e-4):
        self.monitor, self.factor, self.patience, self.min_lr = monitor, factor, patience, min_lr
        self.best, self.wait = float("inf"), 0

    def on_epoch_end(self, epoch, logs):
        if logs[self.monitor] < self.best:
            self.best, self.wait = logs[self.monitor], 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.lr = max(self.model.lr * self.factor, self.min_lr)
                self.wait = 0                              # start counting again after a reduction


class FakeModel:                                           # stands in for the model fit() would pass
    lr = 0.1


cb = ReduceLROnPlateau(patience=2)
cb.model = FakeModel()
val_losses = [1.0, 0.8, 0.7, 0.72, 0.71, 0.69, 0.70, 0.70, 0.70, 0.70]
for epoch, v in enumerate(val_losses, start=1):
    cb.on_epoch_end(epoch, {"val_loss": v})
    print(f"epoch {epoch:2d}  val_loss {v:.2f}  lr {cb.model.lr:.4f}")`;

const PLATEAU_SOL_OUT = String.raw`epoch  1  val_loss 1.00  lr 0.1000
epoch  2  val_loss 0.80  lr 0.1000
epoch  3  val_loss 0.70  lr 0.1000
epoch  4  val_loss 0.72  lr 0.1000
epoch  5  val_loss 0.71  lr 0.0500
epoch  6  val_loss 0.69  lr 0.0500
epoch  7  val_loss 0.70  lr 0.0500
epoch  8  val_loss 0.70  lr 0.0250
epoch  9  val_loss 0.70  lr 0.0250
epoch 10  val_loss 0.70  lr 0.0125`;

const lesson: Lesson = {
  id: "m16/tensorflow-keras",
  title: "TensorFlow and Keras",
  summary: "Keras's Sequential and Functional APIs, custom layers, GradientTape custom training loops, callbacks and SavedModel — with a working NumPy re-implementation of a gradient tape and of the callback system, executed PyTorch equivalents, and a fair PyTorch-vs-TensorFlow comparison.",
  minutes: 70,
  level: "intermediate",
  prerequisites: ["m16/numpy-from-scratch", "m16/pytorch", "m04/backpropagation", "m02/jacobians-hessians"],
  next: ["m17/experimentation", "m18/serving", "m18/mlops"],
  terms: ["tensorflow", "keras", "sequential-api", "functional-api", "custom-layer", "gradient-tape", "tf-function", "keras-callback", "savedmodel"],
  sections: {
    intro: [
      { t: "p", text: "**TensorFlow** is Google's open-source machine learning platform, released in 2015. Version 1 required building a static computation graph and running it in a session; **TensorFlow 2** (2019) made eager execution the default and adopted **Keras** as its official high-level API. Keras began as an independent library by François Chollet (2015) with one design goal: make the common case — define a model, compile it with a loss and optimizer, call `fit` — take a few lines. Keras 3 (2023) became multi-backend again, running the same model code on TensorFlow, JAX or PyTorch." },
      { t: "p", text: "Where PyTorch hands you the training loop, Keras hides it inside `model.fit` and lets you customise it through **callbacks**. When you need full control, TensorFlow's **`GradientTape`** records operations so you can compute gradients yourself — the same reverse-mode automatic differentiation as PyTorch's autograd, but with an explicit recording scope. **SavedModel** packages a trained model as a language-neutral graph for TensorFlow Serving, TensorFlow Lite and other runtimes, which is one of TensorFlow's main strengths in deployment." },
      { t: "callout", kind: "note", title: "How code is verified in this lesson", text: "TensorFlow is not installed in this site's verification environment, so the Keras/TensorFlow programs are labelled \"not executed in CI\" and describe what they print instead of showing numbers. Everything else is executed: a NumPy implementation of a gradient tape and of Keras-style callbacks (Code From Scratch) and the PyTorch equivalents (Framework tab)." },
      { t: "callout", kind: "key", text: "Keras = layers + `compile(optimizer, loss, metrics)` + `fit(data, callbacks=[...])`. `GradientTape` = \"record the forward pass inside a `with` block, then `tape.gradient(loss, variables)`\". Both are thin layers over the same five framework pieces you built in NumPy." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[A Complete Neural Network in NumPy](/learn/m16/numpy-from-scratch) — layers, losses, optimizers and the training loop that `fit` hides.",
        "[PyTorch](/learn/m16/pytorch) — we compare every Keras feature with its PyTorch equivalent; you should know autograd, `nn.Module` and the canonical loop.",
        "[Backpropagation](/learn/m04/backpropagation) — a gradient tape is backpropagation over a recorded list of operations.",
        "[Jacobians and Hessians](/learn/m02/jacobians-hessians) — each recorded operation contributes a vector-Jacobian product in the backward pass.",
        "[Regularization](/learn/m07/regularization) — early stopping, which we implement as a callback.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A flight recorder and a flight plan", text: "A `GradientTape` is a flight recorder: everything that happens to the watched quantities while the recorder is on is written down, in order. Afterwards you play the tape backwards to find out how each control input affected the outcome. Keras's `fit` is an autopilot with a fixed flight plan — take off, fly every leg (batch), report at every waypoint (epoch) — and **callbacks** are the crew members you station at the waypoints: one logs the instruments (History), one lands the plane early when the fuel gauge stops improving (EarlyStopping), one adjusts the throttle on a schedule (LearningRateScheduler)." },
      { t: "p", text: "The two Keras model-building APIs differ in what shapes of network they can describe. **Sequential** is a list: each layer feeds the next. The **Functional** API treats layers as functions applied to symbolic tensors — `h = Dense(64)(x)` — so you can feed one tensor to two layers, merge branches, or have several inputs and outputs: any directed acyclic graph of layers. **Subclassing** (`keras.Model` or `keras.layers.Layer` with your own `call`) gives the full flexibility of Python, like an `nn.Module`." },
    ],
    mechanism: [
      { t: "h", text: "1. Building models: three APIs" },
      { t: "table", head: ["API", "How you write it", "Can express", "PyTorch analogue"], rows: [
        ["Sequential", "`keras.Sequential([Input, Dense, Dense])`", "Linear stacks only", "`nn.Sequential`"],
        ["Functional", "`x = Input(...)`; `h = Dense(64)(x)`; `keras.Model(inputs, outputs)`", "Any DAG: multiple inputs/outputs, skips, shared layers", "An `nn.Module` whose `forward` wires layers"],
        ["Subclassing", "`class M(keras.Model)` with `__init__` and `call`", "Anything, including Python control flow", "`nn.Module` subclass"],
      ] },
      { t: "p", text: "Keras layers create their weights lazily in **`build(input_shape)`**, the first time they see an input, so you do not have to state input sizes for every layer (PyTorch's `nn.Linear` needs `in_features` up front, except the `nn.LazyLinear` variant). A custom layer subclasses `keras.layers.Layer`, creates weights with `self.add_weight` in `build`, computes the output in `call`, and implements `get_config` so it can be re-created when a saved model is loaded." },
      { t: "h", text: "2. compile and fit" },
      { t: "steps", items: [
        { title: "compile", text: "Attach an optimizer, a loss and metrics to the model. Nothing is trained yet." },
        { title: "fit: on_train_begin", text: "Every callback's `on_train_begin` runs (a `History` callback is always added)." },
        { title: "fit: each epoch", text: "Callbacks' `on_epoch_begin` (a scheduler sets the learning rate here); then, for every batch, a train step: forward under a tape, loss, `tape.gradient`, `optimizer.apply_gradients`, metric updates." },
        { title: "fit: end of epoch", text: "Evaluate on `validation_data`, assemble a `logs` dictionary (`loss`, `accuracy`, `val_loss`, …), and pass it to every callback's `on_epoch_end`. A callback can set `self.model.stop_training = True`; `fit` checks it after the epoch." },
        { title: "fit: on_train_end", text: "Callbacks finish (EarlyStopping with `restore_best_weights=True` copies the best weights back). `fit` returns the History object, whose `.history` dictionary holds one list per logged quantity." },
      ] },
      { t: "h", text: "3. GradientTape" },
      { t: "steps", items: [
        { title: "Open a tape", text: "`with tf.GradientTape() as tape:` starts recording. Trainable `tf.Variable`s are **watched** automatically; a plain tensor is recorded only after `tape.watch(t)`." },
        { title: "Run the forward pass", text: "Every operation that touches a watched value is appended to the tape (its inputs, output and how to differentiate it); its output becomes watched too." },
        { title: "tape.gradient(target, sources)", text: "Walk the tape backwards from the target, computing vector-Jacobian products and summing contributions per tensor. Sources that the target does not depend on (or that were never watched) get `None`. A tape releases its resources after one `gradient` call unless created with `persistent=True`." },
        { title: "Apply", text: "`optimizer.apply_gradients(zip(grads, variables))` updates the variables." },
        { title: "Compile for speed", text: "Decorating the step function with `@tf.function` traces it once into a TensorFlow graph that is then executed without Python overhead — the counterpart of `torch.compile`." },
      ] },
      { t: "h", text: "4. Saving" },
      { t: "p", text: "Keras 3's native format is a single `.keras` file (`model.save(\"m.keras\")`, `keras.models.load_model`) holding the architecture (from layer configs), weights and optimizer state. A TensorFlow **SavedModel** directory (`tf.saved_model.save`, or `model.export(path)` in Keras 3) stores traced computation graphs and variables and can be loaded without the Python class definitions — by TensorFlow Serving, the TensorFlow Lite converter, TensorFlow.js tooling or the C++/Java APIs. Weight-only checkpoints use `model.save_weights` or `tf.train.Checkpoint`." },
    ],
    math: [
      { t: "h", text: "What tape.gradient computes" },
      { t: "p", text: "Let the tape record operations $v_1, \\dots, v_T$ in execution order, each $v_t = f_t(\\text{inputs}_t)$. Write $\\bar v = \\partial\\mathcal{L}/\\partial v$ (the **adjoint** of $v$). Starting from $\\bar{\\mathcal{L}} = 1$ and going $t = T, \\dots, 1$:" },
      { t: "math", tex: String.raw`\bar u \mathrel{+}= J_{f_t}(u)^\top\, \bar v_t \qquad \text{for every watched input } u \text{ of operation } t`, caption: "$J_{f_t}(u)$ is the Jacobian of operation $t$'s output with respect to input $u$; $\\bar u$ starts at 0. The $+=$ is the multivariable chain rule: a value used several times receives the sum of its paths' contributions." },
      { t: "p", text: "Two details matter in implementations. **Broadcasting**: if a bias $b \\in \\mathbb{R}^{n}$ was added to every row of an $N\\times n$ matrix, its adjoint is the incoming adjoint summed over the $N$ rows — the `unbroadcast` function in the scratch code. **Watching**: only operations that involve a watched value are recorded, so the tape costs nothing for pure data processing." },
      { t: "h", text: "Worked example: y = x·w² + w at w = 3, x = 2" },
      { t: "steps", items: [
        { title: "Forward (recorded)", tex: String.raw`t_1 = w\cdot w = 9,\qquad t_2 = t_1\cdot x = 18,\qquad y = t_2 + w = 21` },
        { title: "Seed", tex: String.raw`\bar y = 1` },
        { title: "Reverse op 3 (add)", tex: String.raw`\bar t_2 \mathrel{+}= \bar y = 1,\qquad \bar w \mathrel{+}= \bar y = 1` },
        { title: "Reverse op 2 (mul t₁·x)", tex: String.raw`\bar t_1 \mathrel{+}= \bar t_2\, x = 2,\qquad \bar x \mathrel{+}= \bar t_2\, t_1 = 9\ \ (\text{only if } x \text{ is watched})` },
        { title: "Reverse op 1 (mul w·w) — w is both inputs", tex: String.raw`\bar w \mathrel{+}= \bar t_1\, w + \bar t_1\, w = 6 + 6 \ \Rightarrow\ \bar w = 1 + 12 = 13` },
        { title: "Check against calculus", tex: String.raw`\frac{\partial y}{\partial w} = 2xw + 1 = 13,\qquad \frac{\partial y}{\partial x} = w^2 = 9` },
      ] },
      { t: "p", text: "The executed scratch program prints exactly these values: $13$ for $w$, `None` for the unwatched constant $x$, and $9$ for $x$ after `tape.watch(x)`." },
      { t: "h", text: "Early stopping as a rule" },
      { t: "p", text: "With monitored value $m_e$ at epoch $e$ (e.g. validation loss), best value $m^\\ast = \\min_{e' \\le e} m_{e'}$ attained at epoch $e^\\ast$, and patience $P$, training stops at the first epoch with $e - e^\\ast \\ge P$, and with weight restoration the returned parameters are $\\theta_{e^\\ast}$. In the scratch run the best validation loss $0.2652$ occurs at epoch $27$; with $P = 5$ training stops at epoch $27 + 5 = 32$ and epoch 27's weights are restored. (Keras also offers `min_delta`, the minimum change that counts as an improvement.)" },
      { t: "h", text: "Parameter count of the two-input Functional model" },
      { t: "p", text: "`ScaledDense(64)` on 64 inputs: $64\\cdot64$ weights $+ 64$ biases $+ 64$ scales $= 4224$; `Dense(64)` on the 1-dimensional brightness input: $1\\cdot64 + 64 = 128$; output `Dense(10)`: $64\\cdot10 + 10 = 650$. Total $4224 + 128 + 650 = 5002$ — the same in the Keras version and the executed PyTorch version." },
    ],
    visual: [
      { t: "p", text: "A gradient tape replays a recorded graph in reverse. The chain-rule simulator shows that replay on a five-node graph." },
      { t: "sim", id: "chain-rule", caption: "Forward values are what the tape records; backward gradients are what `tape.gradient` computes, one node at a time in reverse order." },
      { t: "list", items: [
        "Step through the forward pass and imagine each node being appended to a tape list. Then step backward: the nodes are visited in exactly the reverse of the order they were recorded.",
        "At the multiply node $w\\cdot x$ the adjoint of $w$ is (upstream) $\\times x$ and the adjoint of $x$ is (upstream) $\\times w$. In TensorFlow, if $x$ were a constant that the tape did not watch, `tape.gradient` would return `None` for it — the computation along that edge would simply be skipped.",
        "Change any slider at the last step: the tape-style gradient and the finite-difference estimate agree, the same check the scratch code performs on a $4\\times3$ weight matrix.",
      ] },
      { t: "p", text: "A `LearningRateScheduler` callback sets the learning rate at the start of every epoch. This simulator shows what common schedules do to a real training run." },
      { t: "sim", id: "lr-schedule", caption: "Learning-rate schedules — each corresponds to a schedule function you could pass to a scheduler callback." },
      { t: "list", items: [
        "Select the step schedule and watch the loss curve: the drop in learning rate produces a visible drop in loss — the same mechanism as the scratch code's switch from $0.2$ to $0.05$ after epoch 20.",
        "Compare cosine and warmup schedules. In Keras each is either a schedule *object* passed as the optimizer's `learning_rate` (updated every step) or a function in a `LearningRateScheduler` callback (updated every epoch).",
      ] },
    ],
    scratch: [
      { t: "p", text: "Two NumPy programs that implement what TensorFlow and Keras do internally. The first is a working **gradient tape**: `Tensor` and `Variable` classes whose operators record themselves onto any active tape, `watch` semantics, `persistent`, broadcasting-aware gradients, a finite-difference check, and a TensorFlow-style custom training loop on the digits dataset." },
      { t: "code", lang: "python", title: "gradient_tape_numpy.py", code: TAPE, output: TAPE_OUT },
      { t: "list", items: [
        "**Recording**: every operator calls `_op`, which computes the value and, for each active tape, appends `(name, inputs, out, k)` if any input is watched; the output then becomes watched. A `Variable` is always watched — exactly TensorFlow's rule for trainable variables.",
        "**Replay**: `gradient` walks `reversed(self.records)`, calls the op's VJP rule and **adds** contributions per tensor (keyed by object identity). Constants that are not watched receive nothing, so `dy/dx` is `None` until `tape.watch(x)`.",
        "**`persistent`**: a second `gradient` call on a non-persistent tape raises, mirroring TensorFlow (which frees the tape's resources after the first call).",
        "**`unbroadcast`** sums the adjoint of the bias over the batch axis; the finite-difference check on the $4\\times3$ weight matrix agrees to $10^{-10}$.",
        "**Training loop**: `with GradientTape()` around the forward pass, `tape.gradient(loss, params)`, `assign_sub` — the same three lines as a TensorFlow custom loop, reaching about 96% test accuracy on digits.",
      ] },
      { t: "p", text: "The second program implements Keras's **callback system** and a `fit` method: the hook names (`on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_train_end`), an always-present `History`, `EarlyStopping` with `restore_best_weights`, a `LearningRateScheduler`, and a custom printing callback. Only 150 training images are used so that the network overfits and early stopping has something to do." },
      { t: "code", lang: "python", title: "keras_callbacks_numpy.py", code: CALLBACKS, output: CALLBACKS_OUT },
      { t: "list", items: [
        "The scheduler sets the learning rate to $0.2$ for epochs 1–20 and $0.05$ afterwards (visible in the `lr` column).",
        "Training accuracy reaches 1.0 by epoch 15 while validation loss bottoms out around $0.265$: classic overfitting on a tiny training set.",
        "EarlyStopping records the best validation loss ($0.2652$ at epoch 27), runs out of patience 5 epochs later, sets `model.stop_training`, and in `on_train_end` copies epoch 27's weights back **in place** — so evaluating after training reproduces $0.2652$.",
        "Callbacks talk to the model only through `self.model` (reading/writing `lr`, `weights`, `stop_training`) and the `logs` dictionary. That narrow interface is why Keras callbacks are reusable across models.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "TensorFlow / Keras", blocks: [
          { t: "h", text: "Sequential, custom layer and Functional API" },
          { t: "code", lang: "python", title: "keras_model_apis.py (not executed in CI)", code: KERAS_APIS, output: "Prints two model summaries: the Sequential model (two Dense layers, 4,810 trainable parameters) and the two-input Functional model (inputs 'pixels' and 'brightness', the custom ScaledDense layer, an Add merge and the output Dense layer; 5,002 trainable parameters)." },
          { t: "h", text: "Custom training loop with GradientTape and tf.function" },
          { t: "code", lang: "python", title: "keras_gradient_tape_loop.py (not executed in CI)", code: KERAS_TAPE, output: "Prints, every 5 epochs for 20 epochs, the last batch's loss and the test accuracy. The first epoch is slower because tf.function traces the step into a graph. Exact values depend on the TensorFlow version and random initialization." },
          { t: "h", text: "Callbacks and saving" },
          { t: "code", lang: "python", title: "keras_callbacks_saving.py (not executed in CI)", code: KERAS_CALLBACKS, output: "Prints the validation loss every 5 epochs until EarlyStopping halts training (well before 200 epochs), then the number of epochs run. Writes best.keras (best checkpoint), digits.keras (native Keras file) and the digits_savedmodel directory." },
          { t: "callout", kind: "note", text: "Keras numbers epochs from 0 inside callbacks and schedule functions, whereas the NumPy re-implementation counts from 1; `lambda epoch, lr: 0.2 if epoch < 20 else 0.05` is the same schedule as the scratch code's `epoch <= 20`." },
        ] },
        { label: "PyTorch", blocks: [
          { t: "p", text: "The same three ideas in PyTorch, executed: `nn.Sequential`; a custom layer with an `nn.Parameter`; a two-input network with a merge (Functional-API territory, here simply written in `forward`); and an explicit loop that plays the role of a GradientTape custom loop, with early stopping and weight restoration written by hand." },
          { t: "code", lang: "python", title: "keras_equivalents_pytorch.py", code: TORCH_EQUIV, output: TORCH_EQUIV_OUT },
          { t: "list", items: [
            "PyTorch has no separate Functional API because `forward` is ordinary Python: branching and merging are just expressions.",
            "Weights are created eagerly in `__init__` (input sizes are required), whereas Keras creates them lazily in `build`.",
            "Early stopping is eight lines of loop code instead of a callback; `copy.deepcopy(model.state_dict())` plays the role of `restore_best_weights=True`. Libraries such as PyTorch Lightning provide a callback-based `fit` for PyTorch.",
          ] },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "PyTorch vs TensorFlow/Keras — a fair comparison" },
      { t: "table", head: ["Aspect", "PyTorch", "TensorFlow / Keras"], rows: [
        ["Default execution", "Eager (define-by-run)", "Eager since TF 2; graphs via `tf.function`"],
        ["Graph compilation", "`torch.compile` (since 2.0), `torch.export`", "`tf.function` with XLA (`jit_compile=True`)"],
        ["High-level training", "You write the loop; Lightning and similar libraries add `fit`-style APIs", "`model.fit` with callbacks built in; custom `train_step` or GradientTape when needed"],
        ["Autodiff interface", "Implicit: any op on `requires_grad` tensors is recorded; `loss.backward()` fills `.grad`", "Explicit scope: `GradientTape`; `tape.gradient` returns gradients"],
        ["Model definition", "`nn.Module` subclasses; eager weight creation", "Sequential, Functional, subclassing; lazy `build`"],
        ["Deployment", "TorchScript/`torch.export`, ExecuTorch for mobile, ONNX", "SavedModel, TF Serving, TF Lite (LiteRT), TF.js, ONNX via converters"],
        ["Accelerators", "NVIDIA/AMD GPUs, Apple MPS; TPUs via PyTorch/XLA", "GPUs; TPUs supported natively"],
        ["Distributed", "DDP, FSDP, `torch.distributed`", "`tf.distribute` strategies (Mirrored, MultiWorkerMirrored, TPU)"],
        ["Ecosystem", "Most research code; Hugging Face's primary backend", "Strong production tooling (TFX, TF Serving); Keras 3 also runs on JAX and PyTorch"],
      ] },
      { t: "p", text: "Neither is universally better. For research and custom architectures, PyTorch's implicit autograd and plain-Python models are usually more convenient; for standard models that must ship to mobile, browsers or established TensorFlow serving infrastructure, TensorFlow's export path is mature; and Keras 3 lets one high-level model definition target several backends. The concepts — tensors, autodiff, layers, optimizers, loops, checkpoints — transfer completely, which is the real skill." },
      { t: "h", text: "Ways to train in Keras" },
      { t: "table", head: ["Approach", "Control", "Effort", "Use when"], rows: [
        ["`compile` + `fit`", "Low (callbacks only)", "Minimal", "Standard supervised training"],
        ["Override `train_step` in a `keras.Model` subclass", "Medium: custom step, keeps `fit`, callbacks, distribution", "Moderate", "Custom losses, GAN-style updates with `fit` conveniences"],
        ["GradientTape loop", "Full", "Most code (metrics, logging, checkpoints yourself)", "Research, unusual training schemes"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Production pipelines**: TensorFlow Extended (TFX) and TensorFlow Serving serve SavedModels behind REST/gRPC APIs — see [Serving](/learn/m18/serving).",
        "**On-device inference**: models converted from SavedModel to TensorFlow Lite run on Android, iOS and microcontrollers.",
        "**In-browser ML**: TensorFlow.js runs converted models directly in web pages.",
        "**TPU training**: TensorFlow (and JAX) are the native frameworks for Google's TPUs, often through `tf.distribute.TPUStrategy`.",
        "**Rapid prototyping and teaching**: `Sequential` + `fit` trains a baseline in a handful of lines, and callbacks such as EarlyStopping and ModelCheckpoint add good practice for free.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Keras's `fit` is concise and batteries-included** (metrics, validation, callbacks, distribution) but hides the loop; stepping outside it means overriding `train_step` or writing a tape loop.",
        "**GradientTape is explicit**: you choose exactly what is recorded and differentiated, and nested tapes give higher-order derivatives — at the cost of a little more ceremony than `loss.backward()`.",
        "**`tf.function` gives graph speed** but introduces tracing semantics: Python side effects run only during tracing, and new input shapes or Python values can trigger retracing.",
        "**Deployment breadth** (Serving, Lite, JS, SavedModel) is a genuine strength; **research mindshare** is mostly with PyTorch, so newer model code is more often written for PyTorch first.",
        "**API churn**: TF 1 → TF 2 and Keras 2 → Keras 3 changed APIs and saving formats; older tutorials may not run unchanged.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Computing the loss outside the tape.** Only operations inside the `with tf.GradientTape()` block are recorded; `tape.gradient` then returns `None`. Fix: put the forward pass *and* the loss inside the block.",
        "**Expecting gradients for plain tensors.** Constants are not watched, so their gradient is `None`. Fix: `tape.watch(x)` or make it a `tf.Variable`.",
        "**Calling `tape.gradient` twice** on a non-persistent tape raises an error. Fix: request all gradients in one call, or use `persistent=True` (and delete the tape afterwards).",
        "**Softmax output + `from_logits=True`** (or logits + `from_logits=False`). The loss either applies softmax twice or treats logits as probabilities. Fix: make the final layer linear and use `from_logits=True`.",
        "**Forgetting `training=True` in a custom loop** (`model(x, training=True)`), so dropout and batch norm behave as in inference. `fit` sets this for you; a tape loop does not.",
        "**Python side effects inside `tf.function`**: a `print` runs only while tracing, and appending to a Python list does not happen per step. Fix: use `tf.print`, return values, or keep such code outside the function.",
        "**Combining callbacks that fight**: a `LearningRateScheduler` overwrites the learning rate every epoch, undoing `ReduceLROnPlateau`. Fix: use one learning-rate controller.",
        "**Saving a model with custom layers but no `get_config`**: loading fails because Keras cannot re-create the layer. Fix: implement `get_config` and pass `custom_objects` (or register the class) when loading.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between the Keras Sequential and Functional APIs?", a: "Sequential describes a linear stack where each layer feeds the next. The Functional API calls layers on symbolic tensors and builds a `keras.Model(inputs, outputs)`, so it can express any directed acyclic graph: multiple inputs or outputs, skip connections, shared layers and merges." },
      { t: "qa", level: "beginner", q: "What does a Keras callback do? Give three examples.", a: "A callback is an object whose hook methods (`on_epoch_end`, `on_train_begin`, `on_batch_end`, …) `fit` calls at fixed points, with access to the model and the latest logs. Examples: `EarlyStopping` (stop when a monitored metric stops improving, optionally restoring the best weights), `ModelCheckpoint` (save the model, e.g. only when validation loss improves), `LearningRateScheduler` / `ReduceLROnPlateau` (change the learning rate), `TensorBoard` (log metrics)." },
      { t: "qa", level: "beginner", q: "What is `tf.GradientTape`?", a: "A context manager that records operations on watched values (trainable variables automatically) during the forward pass; `tape.gradient(target, sources)` then runs reverse-mode autodiff over the recording to return the gradients. It is TensorFlow's counterpart of PyTorch's autograd with an explicit recording scope." },
      { t: "qa", level: "intermediate", q: "Why does `tape.gradient` sometimes return `None`?", a: "Because the target does not depend on that source through recorded operations: the source was not watched (a plain tensor without `tape.watch`), the computation happened outside the tape, or it went through a non-differentiable path such as an integer cast or `argmax`." },
      { t: "qa", level: "intermediate", q: "What does `@tf.function` do, and what is a common pitfall?", a: "It traces the Python function into a TensorFlow graph on the first call (per input signature) and runs the graph afterwards, removing Python overhead and enabling graph optimizations. Pitfalls: Python side effects (print, list appends) run only during tracing, and varying Python arguments or shapes cause retracing, which can be slow." },
      { t: "qa", level: "intermediate", q: "How do you write a custom layer in Keras, and why implement `get_config`?", a: "Subclass `keras.layers.Layer`; create weights in `build(input_shape)` with `self.add_weight` (lazily, once the input size is known); compute the output in `call`. `get_config` returns the constructor arguments so that a saved model can re-create the layer when loading; without it, loading a model containing the layer fails." },
      { t: "qa", level: "intermediate", q: "What is the difference between a `.keras` file and a SavedModel?", a: "A `.keras` file is Keras's native format: it stores the architecture as layer configurations, the weights and the optimizer state, and reloading re-creates Python layer objects (custom classes must be available). A SavedModel stores traced TensorFlow graphs plus variables and can be loaded and executed without the original Python code — by TF Serving, the TFLite converter or other language APIs — which makes it the deployment format." },
      { t: "qa", level: "advanced", q: "How would you compute a second derivative with GradientTape?", a: "Nest tapes: inside an outer tape, open an inner tape, compute $y$, take $g = $ `inner.gradient(y, x)` while still inside the outer tape (so the gradient computation itself is recorded), then `outer.gradient(g, x)`. Because the gradient computation is made of ordinary operations, the outer tape can differentiate it. PyTorch's equivalent is `torch.autograd.grad(y, x, create_graph=True)` followed by another `grad`." },
      { t: "qa", level: "advanced", q: "Compare GradientTape with PyTorch autograd in terms of design.", a: "Both are define-by-run reverse-mode autodiff. PyTorch records implicitly whenever an operand has `requires_grad=True` and stores gradients as `.grad` attributes on leaves, accumulating across `backward` calls. TensorFlow records only inside an explicit tape scope, watches trainable variables automatically, returns gradients as values (no accumulation into the variables) and by default frees the record after one `gradient` call. The explicit scope makes it obvious what is differentiated; the implicit design makes casual code shorter." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Inside a tape, $y = w^3 + 2w$ with $w = 2$ a Variable. What does `tape.gradient(y, w)` return?", answer: 14, tolerance: 0.0001, explain: "$dy/dw = 3w^2 + 2 = 12 + 2 = 14$." },
        { q: "$x$ is a plain `tf.constant` and was not passed to `tape.watch`. What does `tape.gradient(y, x)` return?", options: ["0", "None", "An error", "The same as for a Variable"], answer: 1, explain: "Unwatched constants are not recorded, so the tape has no path to them and returns `None`." },
        { q: "Which Keras API can express a model with two inputs merged by `Add`?", options: ["Only Sequential", "Functional (or subclassing)", "Neither", "Only compile/fit"], answer: 1, explain: "Sequential is limited to linear stacks; the Functional API (and subclassing) handles arbitrary DAGs." },
        { kind: "numeric", q: "EarlyStopping with patience 4 monitors val_loss, which is best at epoch 11 and never improves afterwards. At which epoch does training stop (epochs counted from 1)?", answer: 15, tolerance: 0, explain: "It stops when $e - e^\\ast \\ge 4$: $11 + 4 = 15$." },
        { kind: "numeric", q: "How many trainable parameters does `Dense(32)` have when applied to inputs with 100 features?", answer: 3232, tolerance: 0, explain: "$100\\times32 + 32 = 3232$." },
        { q: "Why is the first epoch of a `@tf.function` training step slower?", options: ["The optimizer warms up", "The function is traced into a graph on the first call", "Data is downloaded", "Callbacks run only in the first epoch"], answer: 1, explain: "Tracing converts the Python function into a graph once per input signature; later calls reuse the graph." },
        { q: "Which format can TensorFlow Serving load without your Python model classes?", options: [".keras file", "SavedModel", "A pickle of the model", "The History object"], answer: 1, explain: "A SavedModel contains the traced graphs and variables, independent of the Python code." },
        { q: "In a GradientTape loop, which step corresponds to PyTorch's `opt.step()`?", options: ["tape.gradient(...)", "optimizer.apply_gradients(zip(grads, vars))", "model.compile(...)", "tape.watch(...)"], answer: 1, explain: "`tape.gradient` is the analogue of `loss.backward()`; `apply_gradients` performs the update." },
      ] },
      { t: "exercise", title: "Run a tape backwards by hand", prompt: "A tape records, with Variables $a = 2$ and $b = -1$: $t_1 = a\\cdot b$, $t_2 = t_1 + a$, $y = t_2^2$. Compute $y$ and then $\\partial y/\\partial a$ and $\\partial y/\\partial b$ by replaying the tape in reverse, showing every adjoint.", hint: "$a$ is used twice; its adjoint is the sum of two contributions.", solution: [
        { t: "steps", items: [
          { title: "Forward", tex: String.raw`t_1 = 2\cdot(-1) = -2,\qquad t_2 = -2 + 2 = 0,\qquad y = 0^2 = 0` },
          { title: "Reverse y = t₂²", tex: String.raw`\bar y = 1,\qquad \bar t_2 = 2t_2\,\bar y = 0` },
          { title: "Reverse t₂ = t₁ + a", tex: String.raw`\bar t_1 \mathrel{+}= \bar t_2 = 0,\qquad \bar a \mathrel{+}= \bar t_2 = 0` },
          { title: "Reverse t₁ = a·b", tex: String.raw`\bar a \mathrel{+}= \bar t_1\, b = 0,\qquad \bar b \mathrel{+}= \bar t_1\, a = 0` },
          { title: "Result and check", tex: String.raw`\frac{\partial y}{\partial a} = 2(ab + a)(b + 1) = 0,\qquad \frac{\partial y}{\partial b} = 2(ab+a)\,a = 0` },
        ] },
        { t: "p", text: "Both gradients are 0 because $y = t_2^2$ is at its minimum ($t_2 = 0$): the square's local derivative kills every adjoint. With $a = 3$, $b = -1$ instead you would get $t_2 = 0$ again; try $a = 2$, $b = 1$: $t_2 = 4$, $\\bar t_2 = 8$, $\\partial y/\\partial a = 8(b + 1) = 16$, $\\partial y/\\partial b = 8a = 16$." },
      ] },
      { t: "exercise", title: "Write a ReduceLROnPlateau callback", prompt: "Using the callback interface of `keras_callbacks_numpy.py`, implement `ReduceLROnPlateau(monitor, factor, patience, min_lr)`: after `patience` epochs without improvement of the monitored value, multiply `self.model.lr` by `factor` (never below `min_lr`) and restart the patience count. Test it on a hand-made sequence of validation losses.", hint: "Track the best value and a wait counter, as EarlyStopping does; reset the counter after each reduction.", solution: [
        { t: "code", lang: "python", title: "reduce_lr_on_plateau.py", code: PLATEAU_SOL, output: PLATEAU_SOL_OUT },
        { t: "p", text: "The loss improves until epoch 3 (0.70); epochs 4 and 5 do not improve, so after the second non-improving epoch the learning rate halves to 0.05. Epoch 6 sets a new best (0.69); epochs 7–8 do not improve → 0.025; epochs 9–10 → 0.0125. Keras's built-in version additionally has `min_delta` (minimum improvement) and `cooldown` (epochs to wait after a reduction before counting again)." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Keras model APIs: Sequential (stacks), Functional (any DAG of layers via `Model(inputs, outputs)`), subclassing (full Python).",
        "Custom layers: `build` (lazy `add_weight`), `call`, `get_config` for saving.",
        "`compile(optimizer, loss, metrics)` + `fit(..., callbacks=[...])`; callbacks hook `on_train_begin/epoch_begin/epoch_end/train_end`, read `logs`, and can set `model.stop_training`.",
        "GradientTape: record inside `with`, `tape.gradient(loss, vars)`, `optimizer.apply_gradients`; Variables auto-watched, constants need `watch`, unconnected sources give `None`, non-persistent tapes allow one call.",
        "Reverse-mode rule: $\\bar u \\mathrel{+}= J^\\top\\bar v$ over the tape in reverse; broadcast inputs sum their adjoints.",
        "`@tf.function` traces to a graph for speed; mind tracing side effects.",
        "Saving: `.keras` (Keras objects) vs SavedModel (language-neutral graphs for Serving/Lite/JS).",
        "PyTorch vs TensorFlow: implicit vs explicit autodiff scope, write-your-loop vs `fit`; both eager by default with graph compilers; choose by ecosystem and deployment target.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — callbacks and checkpoints feed experiment tracking; learn to run comparisons you can trust.",
        "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — SavedModel, TorchScript and ONNX in practice: turning a trained model into a service.",
        "[Monitoring, Drift, CI/CD and Versioning](/learn/m18/mlops) — keeping deployed models healthy.",
      ] },
    ],
  },
};

export default lesson;
