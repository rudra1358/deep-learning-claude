import type { Lesson } from "@/lib/types";

const SCRATCH = String.raw`import numpy as np


# ======== 1. Dataset + DataLoader: what torch.utils.data does, in 25 lines ========
class ArrayDataset:
    """Map-style dataset: knows its length and how to fetch example i."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, seed=0):
        self.ds, self.bs, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last
        self.rng = np.random.default_rng(seed)             # its own generator: reproducible order

    def __len__(self):                                     # number of batches per epoch
        n = len(self.ds)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):                                    # a fresh order every time you loop
        order = self.rng.permutation(len(self.ds)) if self.shuffle else np.arange(len(self.ds))
        for start in range(0, len(order), self.bs):
            idx = order[start:start + self.bs]
            if self.drop_last and len(idx) < self.bs:
                return
            items = [self.ds[i] for i in idx]              # sampler -> indices -> examples
            yield tuple(np.stack(col) for col in zip(*items))   # "collate": stack into batch arrays


X = np.arange(10, dtype=float).reshape(10, 1)
ds = ArrayDataset(X, np.arange(10))
for drop_last in (False, True):
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=drop_last, seed=0)
    print(f"drop_last={drop_last}: {len(loader)} batches ->", [yb.tolist() for _, yb in loader])

# ======== 2. Data parallelism: averaging shard gradients == the full-batch gradient ========
rng = np.random.default_rng(0)
Xr = rng.normal(size=(8, 3))
yr = Xr @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=8)
w = np.zeros(3)


def mse_grad(w, X, y):                                     # gradient of mean((Xw - y)^2)
    return 2 * X.T @ (X @ w - y) / len(y)


full = mse_grad(w, Xr, yr)
shards = [mse_grad(w, Xr[r::2], yr[r::2]) for r in range(2)]   # 2 "GPUs", 4 examples each
all_reduced = sum(shards) / 2                                   # what DDP's all-reduce computes
print("full-batch grad :", full.round(6))
print("averaged shards :", all_reduced.round(6), " max diff", f"{np.abs(full - all_reduced).max():.1e}")

# ======== 3. Mixed precision: why float16 needs loss scaling and bfloat16 does not ========
def to_bf16(x):
    """Round float32 -> bfloat16 (keep the top 16 bits, round-to-nearest-even) and back."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.view(np.float32)


tiny_grad = np.float32(1e-8)                               # a typical small gradient value
print("float16(1e-8)        =", np.float16(tiny_grad))     # underflows to zero: update lost
scale = np.float32(65536.0)                                # loss scaling: multiply the loss by 2^16 ...
print("float16(1e-8 * 2^16) =", np.float16(tiny_grad * scale),
      "-> unscaled in float32:", np.float32(np.float16(tiny_grad * scale)) / scale)
print("bfloat16(1e-8)       =", to_bf16(tiny_grad), "(same exponent range as float32)")
print("1 + 1/512 in fp16, bf16:", np.float16(1 + 1 / 512), to_bf16(1 + 1 / 512), "(bf16 has only 8 significand bits)")
with np.errstate(over="ignore"):                          # (silence NumPy's overflow warning)
    print("largest finite fp16:", np.finfo(np.float16).max, " float16(70000) =", np.float16(70000.0))`;

const SCRATCH_OUT = String.raw`drop_last=False: 3 batches -> [[4, 6, 2, 7], [3, 5, 9, 0], [8, 1]]
drop_last=True: 2 batches -> [[4, 6, 2, 7], [3, 5, 9, 0]]
full-batch grad : [-1.988724  1.279556 -1.518037]
averaged shards : [-1.988724  1.279556 -1.518037]  max diff 2.2e-16
float16(1e-8)        = 0.0
float16(1e-8 * 2^16) = 0.000655 -> unscaled in float32: 9.997166e-09
bfloat16(1e-8)       = 1.0011718e-08 (same exponent range as float32)
1 + 1/512 in fp16, bf16: 1.002 1.0 (bf16 has only 8 significand bits)
largest finite fp16: 6.55e+04  float16(70000) = inf`;

const TENSORS = String.raw`import numpy as np
import torch

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------- tensors: dtype, shape, device, views ----------------
a = torch.tensor([[1, 2, 3], [4, 5, 6]])                 # Python ints -> int64
b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # Python floats -> float32 (not float64!)
print("dtypes:", a.dtype, b.dtype, torch.from_numpy(np.ones(2)).dtype)
print("shape:", tuple(b.shape), " b.T shape:", tuple(b.T.shape), " on:", b.to(device).device)
v = b.view(3, 2)                                         # a VIEW: same memory, new shape
v[0, 0] = 100.0
print("b[0,0] after writing through the view:", b[0, 0].item())
print("matmul (2,3)@(3,2) ->", tuple((b @ b.T).shape), " broadcast (2,3)+(3,) ->", tuple((b + torch.ones(3)).shape))

# ---------------- autograd: the graph is recorded as you compute ----------------
w = torch.tensor([2.0, -1.0], requires_grad=True)        # a leaf we want gradients for
x = torch.tensor([3.0, 4.0])                             # data: no gradient needed
y = (w * x).sum()                                        # y = 2*3 + (-1)*4 = 2
loss = y ** 2                                            # loss = 4
print("loss:", loss.item(), " grad_fn:", type(loss.grad_fn).__name__, "<-", type(y.grad_fn).__name__)
loss.backward()                                          # d loss/dw = 2*y*x = (12, 16)
print("w.grad:", w.grad.tolist())

loss = ((w * x).sum()) ** 2                              # forward again WITHOUT zeroing
loss.backward()
print("w.grad after a 2nd backward (accumulated):", w.grad.tolist())
w.grad = None                                            # what optimizer.zero_grad() does

with torch.no_grad():                                    # nothing recorded: inference / manual updates
    z = w * x
print("inside no_grad requires_grad:", z.requires_grad)

d = w.detach()                                           # same values and memory, cut out of the graph
print("w.detach() requires_grad:", d.requires_grad, " shares w's memory:", d.data_ptr() == w.data_ptr())

# a gradient only flows through the non-detached path
u = w.sum()
out = u * u.detach()                                     # d/dw = u.detach() * 1, NOT 2u
out.backward()
print("grad through u * u.detach():", w.grad.tolist(), "(u =", u.item(), ")")`;

const TENSORS_OUT = String.raw`device: cpu
dtypes: torch.int64 torch.float32 torch.float64
shape: (2, 3)  b.T shape: (3, 2)  on: cpu
b[0,0] after writing through the view: 100.0
matmul (2,3)@(3,2) -> (2, 2)  broadcast (2,3)+(3,) -> (2, 3)
loss: 4.0  grad_fn: PowBackward0 <- SumBackward0
w.grad: [12.0, 16.0]
w.grad after a 2nd backward (accumulated): [24.0, 32.0]
inside no_grad requires_grad: False
w.detach() requires_grad: False  shares w's memory: True
grad through u * u.detach(): [1.0, 1.0] (u = 1.0 )`;

const PIPELINE = String.raw`import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ---------------- 1. Dataset: how to fetch ONE example ----------------
class DigitsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X / 16.0, dtype=torch.float32)   # pixels 0..16 -> [0, 1]
        self.y = torch.tensor(y, dtype=torch.long)              # class indices must be int64

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.2,
                                          random_state=0, stratify=digits.target)
train_ds, test_ds = DigitsDataset(X_tr, y_tr), DigitsDataset(X_te, y_te)


# ---------------- 2. Model: an nn.Module subclass ----------------
class MLP(nn.Module):
    def __init__(self, n_in=64, n_hidden=128, n_out=10, p_drop=0.1):
        super().__init__()                     # registers parameters/submodules assigned below
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))       # raw logits; the loss applies softmax


def make_run(seed=0):
    """Model, optimizer and DataLoader created from one seed, so runs are reproducible."""
    torch.manual_seed(seed)
    model = MLP().to(device)                   # move parameters to the device ONCE, before the optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))  # seeded shuffling
    return model, opt, loader


def train_one_epoch(model, opt, loader):
    model.train()                              # dropout ON
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # every batch must live on the model's device
        # mixed precision: matmuls run in bfloat16, the loss is computed in float32
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(xb)
        loss = F.cross_entropy(logits.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()                               # no graph is recorded: less memory, faster
def evaluate(model, ds):
    model.eval()                               # dropout OFF
    logits = model(ds.X.to(device))
    return (logits.argmax(1).cpu() == ds.y).float().mean().item()


# ---------------- 3. Run A: train 6 epochs straight ----------------
model, opt, loader = make_run()
model.eval()                                   # (eval mode: this peek must not consume dropout randomness)
with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    print("dtype inside autocast:", model(train_ds.X[:2].to(device)).dtype)
for epoch in range(1, 7):
    loss = train_one_epoch(model, opt, loader)
    print(f"A epoch {epoch}  train loss {loss:.4f}  test acc {evaluate(model, test_ds):.3f}")
weights_A = model.fc1.weight.detach().clone()

# ---------------- 4. Run B: train 3 epochs, checkpoint, "crash", resume for 3 more ----------------
with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, "ckpt.pt")
    model, opt, loader = make_run()
    for epoch in range(1, 4):
        train_one_epoch(model, opt, loader)
    torch.save({"epoch": 3,
                "model": model.state_dict(),            # parameters and buffers
                "optim": opt.state_dict(),              # AdamW moments and step counts
                "rng": torch.get_rng_state(),           # dropout randomness
                "loader_rng": loader.generator.get_state()},  # shuffling order
               ckpt_path)
    del model, opt, loader                              # simulate the process dying

    model, opt, loader = make_run(seed=123)             # fresh objects (different seed on purpose)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])
    torch.set_rng_state(ckpt["rng"])
    loader.generator.set_state(ckpt["loader_rng"])
    print("checkpoint keys:", list(ckpt["model"].keys()))
    for epoch in range(ckpt["epoch"] + 1, 7):
        loss = train_one_epoch(model, opt, loader)
        print(f"B epoch {epoch}  train loss {loss:.4f}  test acc {evaluate(model, test_ds):.3f}")

print("resumed run identical to uninterrupted run:", torch.equal(weights_A, model.fc1.weight.detach()))`;

const PIPELINE_OUT = String.raw`device: cpu
dtype inside autocast: torch.bfloat16
A epoch 1  train loss 1.9137  test acc 0.819
A epoch 2  train loss 0.9463  test acc 0.881
A epoch 3  train loss 0.4664  test acc 0.911
A epoch 4  train loss 0.3091  test acc 0.936
A epoch 5  train loss 0.2437  test acc 0.944
A epoch 6  train loss 0.1995  test acc 0.956
checkpoint keys: ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
B epoch 4  train loss 0.3091  test acc 0.936
B epoch 5  train loss 0.2437  test acc 0.944
B epoch 6  train loss 0.1995  test acc 0.956
resumed run identical to uninterrupted run: True`;

const DDP_DEMO = String.raw`import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

WORLD_SIZE = 2


def make_data():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 5, generator=g)
    y = X @ torch.tensor([1.0, -2.0, 0.5, 3.0, 0.0]) + 0.1 * torch.randn(64, generator=g)
    return X, y.unsqueeze(1)


def worker(rank, store_file):
    # every process joins the same group; FileStore avoids opening a network port
    store = dist.FileStore(store_file, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    torch.manual_seed(0)                                   # identical initial weights on every rank
    model = DDP(nn.Linear(5, 1))                           # DDP broadcasts rank 0's weights, hooks backward
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = make_data()
    Xs, ys = X[rank::WORLD_SIZE], y[rank::WORLD_SIZE]       # each rank sees a different half of the data
    for step in range(50):
        loss = nn.functional.mse_loss(model(Xs), ys)
        opt.zero_grad()
        loss.backward()                                    # gradients are all-reduced (averaged) here
        opt.step()
    w = model.module.weight.detach().clone()
    gathered = [torch.zeros_like(w) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, w)
    if rank == 0:
        print("ranks hold identical weights:", torch.equal(gathered[0], gathered[1]))
        print("learned weights:", [round(v, 3) for v in w.squeeze().tolist()])
    dist.destroy_process_group()


def single_process_reference():
    torch.manual_seed(0)
    model = nn.Linear(5, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = make_data()
    for step in range(50):
        loss = nn.functional.mse_loss(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model.weight.detach().squeeze()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(os.path.join(tmp, "store"),), nprocs=WORLD_SIZE, join=True)
    ref = single_process_reference()
    print("single process, full batch:", [round(v, 3) for v in ref.tolist()])`;

const DDP_DEMO_OUT = String.raw`ranks hold identical weights: True
learned weights: [0.997, -2.009, 0.477, 3.014, 0.001]
single process, full batch: [0.997, -2.009, 0.477, 3.014, 0.001]`;

const DDP_TEMPLATE = String.raw`# Launch with:  torchrun --nproc_per_node=4 train_ddp.py
# torchrun starts 4 processes and sets RANK, LOCAL_RANK and WORLD_SIZE in each one's environment.
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def main():
    dist.init_process_group(backend="nccl")              # NCCL for NVIDIA GPUs ("gloo" for CPU)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = build_model().to(device)                      # build_model / train_ds: your own code
    model = DDP(model, device_ids=[local_rank])           # broadcast weights, hook gradient all-reduce
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    sampler = DistributedSampler(train_ds, shuffle=True)  # each rank gets a disjoint 1/WORLD_SIZE slice
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)

    for epoch in range(10):
        sampler.set_epoch(epoch)                          # otherwise every epoch uses the same shuffle
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()                               # gradients averaged across all ranks here
            opt.step()
        if dist.get_rank() == 0:                          # only one process writes checkpoints / logs
            torch.save(model.module.state_dict(), f"ckpt_{epoch}.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()`;

const FP16_TEMPLATE = String.raw`# float16 autocast on a CUDA GPU needs a GradScaler (bfloat16 does not)
scaler = torch.amp.GradScaler("cuda")
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()        # backward on loss * S, so small gradients stay representable
    scaler.unscale_(opt)                 # divide grads by S (needed before clipping)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                     # skips the step if any gradient is inf/nan
    scaler.update()                      # lower S after an overflow, raise it slowly otherwise`;

const TF_EQUIV = String.raw`import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits

digits = load_digits()
X, y = (digits.data / 16.0).astype("float32"), digits.target

# Dataset/DataLoader  ->  tf.data pipeline
ds = (tf.data.Dataset.from_tensor_slices((X, y))
      .shuffle(len(X), seed=0)
      .batch(64)
      .prefetch(tf.data.AUTOTUNE))                       # overlap input preparation with training

# torch.autocast(bfloat16)  ->  a global mixed-precision policy
keras.mixed_precision.set_global_policy("mixed_bfloat16")

# DDP  ->  a distribution strategy (one replica per visible GPU; falls back to one CPU replica)
strategy = tf.distribute.MirroredStrategy()
print("replicas:", strategy.num_replicas_in_sync)
with strategy.scope():                                   # variables created here are mirrored
    model = keras.Sequential([keras.Input(shape=(64,)),
                              keras.layers.Dense(128, activation="relu"),
                              keras.layers.Dense(10, dtype="float32")])   # keep logits in float32
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

# state_dict checkpointing  ->  tf.train.Checkpoint (+ a manager that keeps the last N)
ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
manager = tf.train.CheckpointManager(ckpt, directory="ckpts", max_to_keep=3)
for epoch in range(3):
    model.fit(ds, epochs=1, verbose=2)
    manager.save()
ckpt.restore(manager.latest_checkpoint)                  # resume: weights AND optimizer slots`;

const ACCUM_SOL = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
X, y = torch.randn(64, 10), torch.randint(0, 3, (64,))
model = nn.Linear(10, 3)

# (a) one batch of 64
model.zero_grad()
F.cross_entropy(model(X), y).backward()
g_big = model.weight.grad.clone()

# (b) 4 micro-batches of 16, gradients accumulated, each loss divided by the number of steps
ACCUM = 4
model.zero_grad()
for Xm, ym in zip(X.chunk(ACCUM), y.chunk(ACCUM)):
    (F.cross_entropy(model(Xm), ym) / ACCUM).backward()   # .grad += ...
g_acc = model.weight.grad.clone()

print("max |difference|:", f"{(g_big - g_acc).abs().max().item():.1e}")
print("allclose:", torch.allclose(g_big, g_acc, atol=1e-6))`;

const ACCUM_SOL_OUT = String.raw`max |difference|: 1.5e-08
allclose: True`;

const lesson: Lesson = {
  id: "m16/pytorch",
  title: "PyTorch: Tensors to Distributed Training",
  summary: "Everything you need to train real models in PyTorch: tensors, dtypes and devices; autograd (graphs, backward, no_grad, detach); nn.Module; Dataset and DataLoader; the canonical training loop; GPUs; mixed precision with autocast; checkpointing that resumes bit-for-bit; and data-parallel training with DDP.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m16/numpy-from-scratch", "m04/backpropagation", "m00/numpy-arrays", "m02/jacobians-hessians"],
  next: ["m16/tensorflow-keras", "m17/experimentation", "m18/serving"],
  terms: ["autograd", "nn-module", "dataloader", "device", "autocast", "mixed-precision", "bfloat16", "state-dict", "checkpoint", "data-parallelism", "ddp", "all-reduce", "gradient-accumulation"],
  sections: {
    intro: [
      { t: "p", text: "**PyTorch** is an open-source deep learning library, originally developed at Facebook AI Research and described in Paszke et al. (2019), \"PyTorch: An Imperative Style, High-Performance Deep Learning Library\". It provides three things on top of a NumPy-like array library: (1) **tensors** that can live on GPUs and other accelerators, (2) **autograd**, which records operations as they run and computes gradients automatically, and (3) a toolkit — `nn.Module`, optimizers, data loading, mixed precision, checkpointing, distributed training — for building and training models." },
      { t: "p", text: "The previous lesson built a framework by hand: every layer needed a hand-derived `backward`, it ran only on a CPU, and the training loop was ours. PyTorch removes each of those limits while keeping the same mental model. You write the forward computation as ordinary Python; autograd builds the backward pass; `.to(device)` moves the work to a GPU; and the five pieces (layers, container, loss, optimizer, loop) map one-to-one onto `nn.Module`, `nn.Sequential`, `torch.nn.functional`, `torch.optim` and a loop you still write yourself." },
      { t: "p", text: "PyTorch's \"define-by-run\" (eager) style — the graph is whatever code actually executed — made it the dominant framework in research, and it is widely used in production (often through `torch.compile`, TorchScript/ONNX export or serving tools). This lesson walks from a single tensor to multi-process training, with every program actually executed on a CPU." },
      { t: "callout", kind: "key", text: "Canonical loop: `for xb, yb in loader:` → move to device → `loss = loss_fn(model(xb), yb)` → `opt.zero_grad()` → `loss.backward()` → `opt.step()`. Everything else in this lesson — autocast, checkpoints, DDP — wraps this loop without changing its shape." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[A Complete Neural Network in NumPy](/learn/m16/numpy-from-scratch) — the five pieces of a framework; PyTorch is their industrial version, so we refer back to that library constantly.",
        "[Backpropagation](/learn/m04/backpropagation) — autograd *is* reverse-mode backpropagation over a recorded graph.",
        "[NumPy arrays](/learn/m00/numpy-arrays) — dtypes, shapes, views and broadcasting carry over to tensors almost unchanged.",
        "[Jacobians and Hessians](/learn/m02/jacobians-hessians) — the vector-Jacobian product, which is what each node of the autograd graph computes in the backward pass.",
        "[Momentum and adaptive optimizers](/learn/m06/momentum-adaptive) — so you know what state `torch.optim.AdamW` keeps and why a checkpoint must save it.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A lab notebook that writes itself", text: "Imagine a chemist who records every step of an experiment in a notebook as she performs it: \"mixed A and B, heated the result, filtered it\". If the result is wrong, she can read the notebook **backwards** and work out how each earlier quantity should have been different. Autograd is that notebook. Every operation on a tensor that requires gradients appends an entry (a node holding a pointer to how to differentiate that operation). `loss.backward()` reads the notebook backwards. Code that should not be recorded — evaluation, manual weight updates — is done with the notebook closed (`torch.no_grad()`)." },
      { t: "p", text: "The other ideas are logistics. A **device** is where a tensor's memory lives (CPU RAM or a particular GPU); operations require all operands on the same device, so you move the model once and every batch as it arrives. A **Dataset** answers \"give me example $i$\", and a **DataLoader** turns that into shuffled, batched tensors. **Mixed precision** does the heavy matrix arithmetic in 16-bit numbers because they are faster and smaller, while keeping the sensitive parts in 32-bit. A **checkpoint** is a snapshot of every piece of state needed to continue training as if nothing happened. **Distributed data parallel** runs one copy of the model per GPU on different data and averages their gradients, so all copies stay identical." },
    ],
    mechanism: [
      { t: "h", text: "1. Tensors, dtypes and devices" },
      { t: "p", text: "A `torch.Tensor` is an n-dimensional array with a `dtype` (float32 by default for Python floats, int64 for Python ints), a `shape`, strides (so `view`, `transpose` and slicing return views sharing memory, as in NumPy) and a `device`. `torch.from_numpy` shares memory with a NumPy array and keeps its dtype (often float64 — a common source of dtype-mismatch errors). `x.to(device)` returns a copy on the target device (or `x` itself if already there)." },
      { t: "h", text: "2. Autograd" },
      { t: "steps", items: [
        { title: "Mark leaves", text: "Parameters are tensors with `requires_grad=True` (`nn.Parameter` sets this automatically). Input data normally does not require gradients." },
        { title: "Record during the forward pass", text: "Each operation whose inputs require gradients produces an output with a `grad_fn` — a node such as `MulBackward0` that remembers its inputs (or whatever it needs, e.g. saved tensors) and points to the nodes that produced them. The graph is built fresh on every forward pass, so Python `if`s and loops just work." },
        { title: "backward()", text: "`loss.backward()` starts with $\\partial\\mathcal{L}/\\partial\\mathcal{L} = 1$ and visits nodes in reverse topological order, each computing a vector-Jacobian product and passing results to its inputs. Gradients reaching a leaf are **added** into its `.grad`. The graph is then freed (unless `retain_graph=True`)." },
        { title: "Optimizer step and zeroing", text: "`opt.step()` reads every parameter's `.grad` and updates the parameter in place without recording (internally under no-grad). `opt.zero_grad()` resets `.grad` (to `None` by default in recent versions) so the next backward does not accumulate onto stale gradients." },
        { title: "Opting out", text: "`with torch.no_grad():` (or `@torch.no_grad()` / `torch.inference_mode()`) stops recording — use it for evaluation. `t.detach()` returns a tensor sharing `t`'s memory but cut out of the graph, so gradients do not flow through it (used for targets in some losses, for logging, and for stopping gradients deliberately)." },
      ] },
      { t: "h", text: "3. nn.Module" },
      { t: "p", text: "Subclass `nn.Module`, call `super().__init__()`, assign layers and `nn.Parameter`s as attributes in `__init__`, and write `forward`. Assignment registers them, so `model.parameters()` finds all weights recursively, `model.to(device)` moves them, `model.state_dict()` names them (e.g. `fc1.weight`), and `model.train()` / `model.eval()` switch the behaviour of layers such as dropout and batch normalization. Call the model as `model(x)`, not `model.forward(x)`, so hooks run." },
      { t: "h", text: "4. Dataset and DataLoader" },
      { t: "p", text: "A map-style `Dataset` implements `__len__` and `__getitem__(i)`. A `DataLoader` wraps it: a sampler produces indices (shuffled each epoch if `shuffle=True`, using an optional seeded `generator`), they are grouped into batches, `__getitem__` is called for each, and a collate function stacks the results into batch tensors. With `num_workers > 0`, worker processes prepare batches in parallel; `pin_memory=True` speeds host-to-GPU copies." },
      { t: "h", text: "5. GPU usage" },
      { t: "p", text: "Pick `device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")`, call `model.to(device)` **before** creating the optimizer, and move each batch with `xb.to(device)`. GPU operations run asynchronously; calling `.item()` or `.cpu()` forces synchronization, so avoid it inside the hot loop except for occasional logging." },
      { t: "h", text: "6. Mixed precision" },
      { t: "p", text: "Inside `with torch.autocast(device_type, dtype=...)`, operations that benefit from low precision and are numerically safe (matrix multiplies, convolutions) run in float16 or bfloat16, while precision-sensitive ones (reductions, softmax, losses) are kept in float32. The parameters themselves stay float32 (\"master weights\"), so tiny updates are not lost. With **float16**, small gradients underflow, so a `GradScaler` multiplies the loss by a large factor $S$ before `backward` and divides the gradients by $S$ before the step. **bfloat16** has float32's exponent range and needs no scaler; it is the usual choice on recent GPUs and TPUs, and it also works on CPUs (as in our executed example)." },
      { t: "h", text: "7. Checkpointing" },
      { t: "p", text: "`model.state_dict()` is an ordered dictionary of parameter and buffer tensors; `opt.state_dict()` holds the optimizer's per-parameter state (Adam's moments and step counts) and hyperparameters. A resumable checkpoint saves both plus the epoch, any LR scheduler and gradient-scaler state, and — for bit-exact resumption — the random-number-generator states (for dropout and for data shuffling). Restore with `load_state_dict` into freshly constructed objects." },
      { t: "h", text: "8. Distributed data parallel (DDP)" },
      { t: "steps", items: [
        { title: "One process per device", text: "A launcher (`torchrun`) starts $K$ processes; each joins a process group (`init_process_group` with the NCCL backend for GPUs or gloo for CPUs) and knows its `rank` (0…K−1)." },
        { title: "Identical replicas", text: "Wrapping the model in `DistributedDataParallel` broadcasts rank 0's parameters to every process, so all replicas start equal." },
        { title: "Different data", text: "A `DistributedSampler` gives each rank a disjoint shard of every epoch; call `sampler.set_epoch(epoch)` so shuffles differ across epochs." },
        { title: "All-reduce during backward", text: "As gradients become ready, DDP groups them into buckets and **all-reduces** each bucket (every rank ends with the average over ranks), overlapping communication with the rest of the backward pass." },
        { title: "Identical steps", text: "Every rank applies the same optimizer step to the same averaged gradients, so replicas remain identical without ever exchanging weights again. Only rank 0 needs to save checkpoints or log." },
      ] },
      { t: "p", text: "When a model's parameters, gradients and optimizer state no longer fit on one device, **sharded** data parallelism (PyTorch's FSDP, following the ZeRO idea) splits those across ranks and gathers parameters just-in-time; tensor and pipeline parallelism split the model itself. These build on the same process-group and collective-communication primitives." },
    ],
    math: [
      { t: "h", text: "Autograd computes vector-Jacobian products" },
      { t: "p", text: "For an operation $y = f(x)$ inside the graph, with incoming gradient $\\bar y = \\partial\\mathcal{L}/\\partial y$, the node computes $\\bar x = \\bar y^\\top J_f(x)$ — a vector-Jacobian product — without ever forming the Jacobian $J_f$. A leaf used in several places receives the sum of the contributions from each path, which is why `.grad` accumulates." },
      { t: "math", tex: String.raw`\bar x = J_f(x)^\top \bar y,\qquad \bar\theta \mathrel{+}= \sum_{\text{uses } u \text{ of } \theta} J_u^\top \bar y_u` },
      { t: "h", text: "Worked example (reproduced by the executed tensor program)" },
      { t: "p", text: "Let $w = (2, -1)$ require gradients, $x = (3, 4)$ be data, $y = w\\cdot x$ and $\\mathcal{L} = y^2$." },
      { t: "steps", items: [
        { title: "Forward", tex: String.raw`y = 2\cdot3 + (-1)\cdot4 = 2,\qquad \mathcal{L} = 2^2 = 4` },
        { title: "PowBackward0 (L = y²)", tex: String.raw`\bar y = \frac{\partial\mathcal{L}}{\partial y} = 2y = 4` },
        { title: "SumBackward0 / MulBackward0 (y = Σ w ⊙ x)", tex: String.raw`\bar w = \bar y\, x = 4\,(3, 4) = (12,\ 16)` },
        { title: "A second backward without zeroing accumulates", tex: String.raw`w.\mathrm{grad} = (12, 16) + (12, 16) = (24,\ 32)` },
        { title: "Stopping a path: out = u · detach(u), u = w₁ + w₂ = 1", text: "Only the non-detached factor carries gradient, so the result is $u$ (treated as a constant) times $\\partial u/\\partial w$, not $2u$.", tex: String.raw`\frac{\partial\,\mathrm{out}}{\partial w} = \mathrm{detach}(u)\cdot(1, 1) = (1,\ 1)` },
      ] },
      { t: "h", text: "Why averaging gradients across GPUs is exact" },
      { t: "p", text: "Split a batch of $B$ examples into $K$ shards $S_k$ of equal size $B/K$. The mean loss and its gradient are" },
      { t: "math", tex: String.raw`\nabla_\theta \frac{1}{B}\sum_{i=1}^{B}\ell_i = \frac{1}{K}\sum_{k=1}^{K}\underbrace{\nabla_\theta\frac{1}{B/K}\sum_{i\in S_k}\ell_i}_{g_k\ \text{(rank } k\text{'s gradient)}}`, caption: "$\\ell_i$ is example $i$'s loss and $g_k$ the gradient computed by rank $k$ on its shard. The all-reduce average equals the full-batch gradient exactly (up to float rounding) — the scratch code measures a difference of $2.2\\times10^{-16}$." },
      { t: "p", text: "Hence the **effective batch size** is $K\\times b\\times a$ for $K$ ranks, per-rank batch $b$ and $a$ gradient-accumulation steps. Example: 8 GPUs × 32 per GPU × 4 accumulation steps $= 1024$. Changing the effective batch usually requires retuning the learning rate (see the linear scaling rule in [Learning-Rate Schedules](/learn/m06/lr-schedules)). If shards have unequal sizes or a layer uses batch statistics (batch norm), the equality no longer holds exactly." },
      { t: "h", text: "Number formats for mixed precision" },
      { t: "table", head: ["Format", "Bits (sign, exponent, significand)", "Largest finite", "Smallest positive normal", "Spacing at 1.0"], rows: [
        ["float32", "1, 8, 23", "$\\approx 3.40\\times10^{38}$", "$2^{-126}\\approx1.18\\times10^{-38}$", "$2^{-23}\\approx1.19\\times10^{-7}$"],
        ["float16", "1, 5, 10", "$65504$", "$2^{-14}\\approx6.10\\times10^{-5}$", "$2^{-10}\\approx9.77\\times10^{-4}$"],
        ["bfloat16", "1, 8, 7", "$\\approx 3.39\\times10^{38}$", "$2^{-126}\\approx1.18\\times10^{-38}$", "$2^{-7} = 0.0078125$"],
      ] },
      { t: "p", text: "Float16's smallest *subnormal* value is $2^{-24}\\approx5.96\\times10^{-8}$, so a gradient of $10^{-8}$ rounds to 0. Multiplying the loss by $S = 2^{16} = 65536$ multiplies every gradient by $S$ (backpropagation is linear in the upstream gradient): $10^{-8}\\times65536 = 6.55\\times10^{-4}$, comfortably representable; dividing by $S$ in float32 recovers $\\approx 10^{-8}$. Bfloat16 keeps float32's exponent range (so no scaling is needed) but has only 8 significant bits: $1 + 1/512$ rounds to exactly 1. That is why accumulations and master weights stay in float32." },
      { t: "h", text: "Memory of a training run" },
      { t: "p", text: "With float32 weights and Adam, each parameter costs 4 bytes (weight) + 4 (gradient) + 8 (two Adam moments) $= 16$ bytes before activations. A model with $10^9$ parameters therefore needs $16\\times10^9$ bytes $= 16$ GB just for this state — the motivation for sharded data parallelism, which divides it across $K$ ranks." },
    ],
    visual: [
      { t: "p", text: "Autograd builds a computational graph exactly like the one in this simulator: values flow forward, `grad_fn` nodes send vector-Jacobian products backward." },
      { t: "sim", id: "chain-rule", caption: "Forward values and backward gradients on a small expression graph — what `loss.backward()` does node by node." },
      { t: "list", items: [
        "Step through the forward pass and identify, for each node, the `grad_fn` PyTorch would attach (`MulBackward0`, `AddBackward0`, `SigmoidBackward0`, `SubBackward0`, `PowBackward0`).",
        "Continue into the backward pass: each clay arrow is one vector-Jacobian product. Note that the backward visits nodes in exactly the reverse of the forward order (reverse topological order).",
        "Set $x = 0$: the gradient reaching $w$ becomes 0. In PyTorch, a weight multiplied by a zero input gets `w.grad == 0` for that example — the same multiply-node \"swap\" rule.",
      ] },
      { t: "p", text: "PyTorch tensors follow NumPy's broadcasting rules exactly — the source of both convenient code and silent shape bugs (and the reason a broadcast bias's gradient is a sum over the broadcast axis)." },
      { t: "sim", id: "broadcasting", caption: "Line up two shapes and see which axes stretch." },
      { t: "list", items: [
        "Set shapes $(64, 10)$ and $(10,)$ — a batch of logits plus a bias: the bias stretches over the batch axis. In the backward pass PyTorch sums the gradient over that axis to get `bias.grad` of shape $(10,)$.",
        "Try $(64, 1)$ with $(64,)$: the result is $(64, 64)$, not $(64,)$. This is the classic bug when a model outputs shape $(N, 1)$ and targets have shape $(N,)$ in an MSE loss — PyTorch warns for some losses, but your own arithmetic will silently broadcast.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Three PyTorch mechanisms, re-implemented in NumPy so nothing is magic: a `Dataset`/`DataLoader` with shuffling, batching, collation and `drop_last`; the gradient averaging DDP performs, checked against the full-batch gradient; and the floating-point facts behind mixed precision (float16 underflow, loss scaling, and a bfloat16 rounding function)." },
      { t: "code", lang: "python", title: "pytorch_mechanisms_numpy.py", code: SCRATCH, output: SCRATCH_OUT },
      { t: "list", items: [
        "**DataLoader**: the sampler is just a permutation from the loader's own generator; batches are consecutive slices; collation is `np.stack` per field. The two runs print the same first two batches because both loaders were created with seed 0 — exactly why `DataLoader(..., generator=torch.Generator().manual_seed(0))` makes shuffling reproducible. With `drop_last=True` the short final batch (2 items) is discarded.",
        "**Data parallelism**: each \"GPU\" computes the mean-loss gradient on its 4 examples; averaging the two equals the gradient on all 8 to machine precision ($2.2\\times10^{-16}$), as derived in the Math section.",
        "**Mixed precision**: $10^{-8}$ becomes 0 in float16 but survives after scaling by $2^{16}$ and unscaling in float32 (the tiny difference, $9.997\\times10^{-9}$, is float16 rounding). `to_bf16` keeps the top 16 bits of a float32 with round-to-nearest-even — that is literally what bfloat16 is — so it represents $10^{-8}$ but rounds $1 + 1/512$ to $1$. Float16 overflows to `inf` above 65504, the other reason for dynamic loss scaling.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "h", text: "Tensors and autograd" },
          { t: "code", lang: "python", title: "tensors_autograd.py", code: TENSORS, output: TENSORS_OUT },
          { t: "p", text: "Every number matches the Math section's worked example: gradient $(12, 16)$, accumulation to $(24, 32)$, and $(1, 1)$ through the detached product. Note `b.view(3, 2)` shares memory with `b`, and `torch.from_numpy` keeps NumPy's float64." },
          { t: "h", text: "The full pipeline: Dataset, nn.Module, device, autocast, checkpoint and resume" },
          { t: "code", lang: "python", title: "train_pipeline.py", code: PIPELINE, output: PIPELINE_OUT },
          { t: "list", items: [
            "`device:` prints `cpu` here; on a machine with a CUDA GPU the same code prints `cuda` and runs there, because the model and every batch go through `.to(device)`.",
            "Inside `torch.autocast(..., dtype=torch.bfloat16)` the linear layers output bfloat16; the logits are cast back with `.float()` before the loss so the loss and its gradient start in float32. Parameters remain float32 throughout.",
            "Run B trains 3 epochs, saves model, optimizer and **both** RNG states (dropout and shuffling), discards everything, rebuilds the objects with a *different* seed, loads the checkpoint and finishes. Its epochs 4–6 and final weights equal run A's exactly (`torch.equal` is `True`).",
            "Bit-exact resumption is fragile: any extra use of a random generator breaks it. A first draft of this program called the model once in training mode to print the autocast dtype — that consumed dropout randomness and made run A differ from run B. Hence the `model.eval()` and `torch.no_grad()` around that peek.",
          ] },
          { t: "h", text: "Distributed data parallel: a real two-process run on CPU" },
          { t: "p", text: "Two processes join a gloo process group (through a `FileStore` in a temporary directory, so no network port is opened), each trains a DDP-wrapped linear model on half of the data, and the result is compared with a single process training on all the data." },
          { t: "code", lang: "python", title: "ddp_gloo_demo.py", code: DDP_DEMO, output: DDP_DEMO_OUT },
          { t: "p", text: "Both ranks finish with identical weights, and those weights equal the single-process full-batch run to three decimals: two half-batch gradients averaged by all-reduce are the full-batch gradient. On a GPU cluster the structure is the same; only the launcher, backend and sampler change:" },
          { t: "code", lang: "python", title: "train_ddp.py — multi-GPU template (not executed in CI)", code: DDP_TEMPLATE },
          { t: "h", text: "float16 on GPUs: the gradient scaler" },
          { t: "code", lang: "python", title: "fp16_grad_scaler.py — fragment (not executed in CI)", code: FP16_TEMPLATE },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same concepts in TensorFlow: `tf.data` for input pipelines, a global mixed-precision policy, `tf.distribute.MirroredStrategy` for single-machine data parallelism, and `tf.train.Checkpoint` for resumable state. See [TensorFlow and Keras](/learn/m16/tensorflow-keras) for the full treatment." },
          { t: "code", lang: "python", title: "tf_equivalents.py (not executed in CI)", code: TF_EQUIV, output: "Prints the number of replicas (1 on a machine without GPUs, otherwise the number of visible GPUs), then one line per epoch for 3 epochs with loss and accuracy. Writes up to 3 checkpoints to ./ckpts and restores the latest." },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "Ways to scale training" },
      { t: "table", head: ["Technique", "What is split / changed", "Advantages", "Disadvantages", "Use when"], rows: [
        ["Single device", "Nothing", "Simplest; easiest to debug", "Limited by one device's memory and speed", "Small and medium models"],
        ["Gradient accumulation", "Batch split over time (several backward calls per step)", "Large effective batch on one device", "No speed-up; batch-norm statistics use micro-batches", "Memory-limited batch size"],
        ["`nn.DataParallel` (legacy)", "Batch split across GPUs inside one process", "One-line change", "Single-process bottleneck; slower than DDP; not recommended", "Avoid; use DDP"],
        ["DDP", "Data split across processes; full replica per device", "Near-linear scaling; standard", "Every device must hold the full model + optimizer state", "Model fits on one device"],
        ["FSDP / ZeRO-style sharding", "Parameters, gradients and optimizer state sharded across ranks", "Trains models far larger than one device", "More communication; more complex checkpoints", "Model state does not fit on one device"],
        ["Tensor / pipeline parallelism", "Individual layers or groups of layers split across devices", "Very large models; combines with data parallelism", "Complex; pipeline bubbles; needs fast interconnect", "Largest models"],
      ] },
      { t: "h", text: "Precision choices" },
      { t: "table", head: ["Mode", "How", "Pros", "Cons"], rows: [
        ["float32", "Default", "Simple, robust", "Slowest on tensor-core hardware; 2× memory of 16-bit"],
        ["autocast bfloat16", "`torch.autocast(dtype=torch.bfloat16)`", "No loss scaling; float32 range", "Needs hardware support for speed; only 8 significant bits"],
        ["autocast float16 + GradScaler", "`torch.autocast(dtype=torch.float16)` + `torch.amp.GradScaler`", "Fast on GPUs without bf16 support", "Overflow/underflow handling via dynamic scaling"],
        ["TF32 matmuls", "`torch.backends.cuda.matmul.allow_tf32 = True` on supported NVIDIA GPUs", "Speed-up with no code changes", "Reduced matmul precision (10-bit significand)"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Research prototyping**: eager execution and Python control flow make PyTorch the default framework in most deep learning papers' public code.",
        "**Large-model training**: DDP and FSDP with bfloat16 autocast, checkpointing every N steps so that hardware failures cost minutes, not days.",
        "**Fine-tuning pretrained models**: load a `state_dict`, freeze some parameters (`requires_grad_(False)`), train the rest — see [Transfer Learning](/learn/m12/transfer-learning).",
        "**Ecosystem libraries** build on these primitives: torchvision (vision datasets and models), Hugging Face Transformers, PyTorch Lightning (organizes the training loop), and many others.",
        "**Deployment**: trained modules are exported (TorchScript, `torch.export`, ONNX) or compiled and served — covered in [Serving](/learn/m18/serving).",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Pythonic and debuggable**: you can print or step through any tensor mid-forward; the graph is whatever ran.",
        "**Explicit training loop**: full control (custom losses, multiple optimizers, GAN-style alternating updates) at the cost of more boilerplate than Keras's `fit` — libraries like Lightning trade some control back for brevity.",
        "**Eager overhead**: each operation is dispatched from Python; small operations are launch-bound. `torch.compile` captures graphs to fuse operations and recover speed, at the cost of compile time and occasional graph breaks.",
        "**Reproducibility needs care**: seeds for every generator, deterministic algorithms (`torch.use_deterministic_algorithms(True)`), and identical library versions; GPU kernels can still be nondeterministic by default.",
        "**Distributed training adds failure modes**: hangs when ranks disagree on the number of collective calls, mismatched data shards, and checkpoint coordination.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Forgetting `opt.zero_grad()`.** Gradients accumulate across steps and the effective step grows each iteration. Fix: zero once per optimizer step (or deliberately accumulate, scaling the loss).",
        "**Evaluating in training mode or with gradients on.** Dropout stays active and the graph wastes memory. Fix: `model.eval()` plus `torch.no_grad()` for evaluation; `model.train()` before training.",
        "**Device mismatch.** \"Expected all tensors to be on the same device\" means a batch, a target or a freshly created tensor (e.g. `torch.zeros(n)` inside `forward`) is on the CPU. Fix: move every batch; create tensors with `device=x.device`.",
        "**Creating the optimizer before `model.to(device)`.** Usually works because `.to` moves parameters in place for modules, but it is fragile with some wrappers. Fix: move first, then build the optimizer.",
        "**Accumulating the loss tensor for logging** (`total += loss`). Each step's graph is kept alive and memory grows until it runs out. Fix: `total += loss.item()` or `loss.detach()`.",
        "**Applying softmax before `F.cross_entropy`.** It expects raw logits (it applies log-softmax internally); a double softmax trains poorly. Fix: return logits.",
        "**Checkpointing only the model.** Resuming with a fresh optimizer resets Adam's moments and step count (and the LR schedule), causing a loss spike. Fix: save optimizer, scheduler, scaler, epoch and RNG states too.",
        "**DDP: not calling `sampler.set_epoch(epoch)`**, so every epoch uses the same shuffle; or saving checkpoints from every rank at once. Fix: set the epoch; save from rank 0 only (and load with `map_location`).",
        "**Wrong dtypes**: float64 inputs from NumPy into a float32 model, or float labels into `cross_entropy` (it needs int64 class indices). Fix: convert explicitly with `dtype=torch.float32` / `torch.long`.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Why do we call `optimizer.zero_grad()` every iteration?", a: "`backward()` adds gradients into each parameter's `.grad` rather than overwriting it. Without zeroing, the gradients of all previous steps would be summed into the current update. Accumulation is useful only when deliberate (gradient accumulation over micro-batches)." },
      { t: "qa", level: "beginner", q: "What do `model.train()` and `model.eval()` change?", a: "They set the `training` flag on every submodule. Layers such as dropout (active only in training) and batch normalization (batch statistics in training, running statistics in eval) behave differently. They do **not** turn off gradient tracking — that is `torch.no_grad()`." },
      { t: "qa", level: "beginner", q: "What is the difference between `torch.no_grad()` and `tensor.detach()`?", a: "`no_grad` is a context in which no operations are recorded at all (used for inference and manual updates). `detach()` returns a single tensor that shares memory with the original but has no graph history, so gradients will not flow back through it; the rest of the computation can still be recorded." },
      { t: "qa", level: "intermediate", q: "What should a checkpoint contain to resume training exactly?", a: "Model `state_dict`, optimizer `state_dict` (moments, step counts, hyperparameters), LR scheduler and gradient-scaler state, the epoch/step counter, and RNG states (torch CPU/CUDA, NumPy, Python, and the DataLoader generator). With all of these and deterministic kernels, resumption reproduces the uninterrupted run, as the executed pipeline demonstrates." },
      { t: "qa", level: "intermediate", q: "Why does fp16 mixed precision need loss scaling but bf16 does not?", a: "Float16 has a 5-bit exponent: values below about $3\\times10^{-8}$ (half the smallest subnormal, $2^{-24}$) round to zero, and many gradients are that small. Scaling the loss by $S$ scales all gradients by $S$ into the representable range; they are unscaled in float32 before the update, and steps with inf/nan are skipped while $S$ is reduced. Bfloat16 has float32's 8-bit exponent, so small gradients do not underflow." },
      { t: "qa", level: "intermediate", q: "Explain how DDP keeps model replicas synchronized.", a: "At construction rank 0's parameters are broadcast to all ranks. During `backward`, hooks all-reduce gradient buckets so every rank holds the average gradient over all ranks' data. Every rank then performs the same optimizer step on the same gradients starting from the same weights, so the replicas stay identical without exchanging weights again." },
      { t: "qa", level: "intermediate", q: "What is the effective batch size with 4 GPUs, per-GPU batch 16 and 8 gradient-accumulation steps?", a: "$4\\times16\\times8 = 512$. Each optimizer step averages gradients over 512 examples, which affects how the learning rate should be set." },
      { t: "qa", level: "advanced", q: "Why is DDP faster than the older `nn.DataParallel`?", a: "`DataParallel` uses one process with multiple threads: it scatters inputs, replicates the model onto each GPU every iteration, gathers outputs to one GPU and is limited by Python's GIL and the master GPU's load. DDP uses one process per GPU, replicates the model once, and overlaps bucketed gradient all-reduce with the backward computation, so communication is largely hidden." },
      { t: "qa", level: "advanced", q: "Training with autocast bf16 matches float32 loss early on but stalls later. What could be wrong?", a: "Bfloat16 has only 8 significant bits. If something accumulates in bf16 — e.g. you cast the parameters themselves to bf16 (`model.bfloat16()`) instead of using autocast, so updates smaller than about 1/256 of a weight's magnitude round away — learning stalls. Keep master weights and optimizer state in float32, compute losses and reductions in float32 (autocast does this for its op lists), and check custom operations." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "$w = (1, 2)$ requires grad, $x = (2, -1)$, $\\mathcal{L} = (w\\cdot x)^2$. What is the first component of `w.grad` after `L.backward()`?", answer: 0, tolerance: 0.0001, explain: "$w\\cdot x = 2 - 2 = 0$, so $\\partial\\mathcal{L}/\\partial w = 2(w\\cdot x)x = 0$." },
        { kind: "numeric", q: "Same setup but $x = (3, 1)$. You call `backward()` twice without zeroing. What is the first component of `w.grad`?", answer: 60, tolerance: 0.0001, explain: "$w\\cdot x = 5$; one backward gives $2\\cdot5\\cdot3 = 30$; two accumulate to 60." },
        { q: "Which call stops gradients from flowing through one tensor but leaves the rest of the computation recorded?", options: ["torch.no_grad()", "tensor.detach()", "model.eval()", "optimizer.zero_grad()"], answer: 1, explain: "`detach()` cuts a single tensor out of the graph; `no_grad` stops all recording; `eval()` only changes layer behaviour." },
        { kind: "numeric", q: "8 GPUs, per-GPU batch 64, 2 accumulation steps. What is the effective batch size?", answer: 1024, tolerance: 0, explain: "$8\\times64\\times2 = 1024$." },
        { q: "Why does bfloat16 not need a GradScaler?", options: ["It has more significand bits than float16", "It has the same exponent range as float32, so small gradients do not underflow", "PyTorch scales it automatically", "It is only used for inference"], answer: 1, explain: "Bfloat16 keeps float32's 8-bit exponent (it actually has fewer significand bits than float16)." },
        { kind: "numeric", q: "How many GB does float32 Adam training state (weights, gradients, two moments) take for a 7-billion-parameter model, ignoring activations? (1 GB = $10^9$ bytes)", answer: 112, tolerance: 0.5, explain: "$16$ bytes/parameter $\\times 7\\times10^9 = 112\\times10^9$ bytes." },
        { q: "In DDP, what does the all-reduce compute?", options: ["The sum of the models' weights", "The average of the gradients over all ranks, delivered to every rank", "The loss on rank 0 only", "A copy of rank 0's gradients to the others"], answer: 1, explain: "Every rank receives the average of all ranks' gradients, so all take the same step." },
        { q: "You resume from a checkpoint that stored only `model.state_dict()`. What typically happens with Adam?", options: ["Nothing; training continues identically", "A loss spike, because the moment estimates and step count restart from zero", "An error at load time", "The learning rate doubles permanently"], answer: 1, explain: "Fresh Adam state means bias-corrected steps of about the learning rate for every parameter regardless of history — usually a visible jump in the loss." },
      ] },
      { t: "exercise", title: "Loss scaling arithmetic", prompt: "In float16, the smallest positive subnormal is $2^{-24}$ and the largest finite value is 65504. A model's gradients range from $3\\times10^{-9}$ to $0.5$. (a) Which gradients underflow without scaling? (b) Choose a power-of-two loss scale $S$ that makes the smallest gradient representable without making the largest overflow. (c) What range of $S$ is valid?", hint: "Require $S\\cdot3\\times10^{-9} \\ge 2^{-24}$ and $S\\cdot0.5 \\le 65504$.", solution: [
        { t: "steps", items: [
          { title: "(a) Underflow threshold", text: "$2^{-24} \\approx 5.96\\times10^{-8}$; every gradient smaller than about $3\\times10^{-8}$ (half the smallest subnormal, the rounding midpoint) becomes 0 — including the smallest, $3\\times10^{-9}$." },
          { title: "(b) Lower bound", tex: String.raw`S \ge \frac{5.96\times10^{-8}}{3\times10^{-9}} \approx 19.9 \ \Rightarrow\ S \ge 2^{5} = 32` },
          { title: "Upper bound", tex: String.raw`S \le \frac{65504}{0.5} = 131008 \ \Rightarrow\ S \le 2^{16} = 65536` },
          { title: "(c) Valid range", text: "Any power of two from $2^5 = 32$ to $2^{16} = 65536$. A larger $S$ within the range also keeps small gradients more precise (they land in the normal range, above $2^{-14}$, for $S \\ge 2^{-14}/(3\\times10^{-9}) \\approx 20345$, i.e. $S \\ge 2^{15}$). PyTorch's `GradScaler` finds such a value dynamically: it starts high, halves $S$ after any overflow and grows it after a run of successful steps." },
        ] },
      ] },
      { t: "exercise", title: "Gradient accumulation equals a big batch", prompt: "Show in PyTorch that accumulating gradients over 4 micro-batches of 16 (calling `backward()` on each) produces the same gradient as one batch of 64. What must you do to each micro-batch loss, and why?", hint: "`F.cross_entropy` returns the mean over its batch.", solution: [
        { t: "code", lang: "python", title: "grad_accumulation.py", code: ACCUM_SOL, output: ACCUM_SOL_OUT },
        { t: "p", text: "Each micro-batch loss is a mean over 16 examples; the big-batch loss is a mean over 64 $=$ the average of the four micro-batch means. Since `.grad` sums the four backward calls, each loss must be divided by the number of accumulation steps (4) to turn the sum into that average. Without the division the gradient would be 4× too large — equivalent to a 4× learning rate. The remaining difference ($\\sim10^{-8}$) is float32 summation order." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Tensors = NumPy-like arrays with a dtype (float32 default), shape, strides (views) and a device; operands must share a device.",
        "Autograd records `grad_fn` nodes during the forward; `backward()` computes vector-Jacobian products in reverse order and **adds** into `.grad`. Use `zero_grad`, `no_grad`, `detach` deliberately.",
        "`nn.Module` registers parameters/submodules; `state_dict` names them; `train()`/`eval()` switch dropout and batch norm.",
        "Dataset (`__len__`, `__getitem__`) + DataLoader (sampler → batches → collate) with a seeded generator for reproducible shuffling.",
        "Canonical loop: to(device) → forward → loss → zero_grad → backward → step; evaluate under `eval()` + `no_grad()`.",
        "Mixed precision: autocast runs matmuls in bf16/fp16, keeps sensitive ops and master weights in float32; fp16 needs loss scaling ($10^{-8}\\cdot2^{16}$ survives), bf16 does not.",
        "Checkpoints: model + optimizer (+ scheduler, scaler, epoch, RNG states) → exact resumption.",
        "DDP: one process per device, broadcast initial weights, all-reduce averaged gradients during backward; effective batch $= K\\times b\\times a$.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[TensorFlow and Keras](/learn/m16/tensorflow-keras) — the other major framework, its high-level `fit` API and `GradientTape`, and an even-handed comparison with PyTorch.",
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — seeds, checkpoints and logging are the raw material for trustworthy experiments.",
        "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — what happens to a trained `state_dict` next: export, serving and deployment.",
      ] },
    ],
  },
};

export default lesson;
