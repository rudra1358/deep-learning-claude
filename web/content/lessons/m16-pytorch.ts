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
  summary: "Everything you need to train real models in PyTorch: tensors, dtypes and devices; autograd (how it records and runs the backward pass, no_grad, detach); nn.Module; Dataset and DataLoader; the standard training loop; GPUs; mixed precision with autocast; checkpoints that resume exactly; and training on many GPUs with DDP.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m16/numpy-from-scratch", "m04/backpropagation", "m00/numpy-arrays", "m02/jacobians-hessians"],
  next: ["m16/tensorflow-keras", "m17/experimentation", "m18/serving"],
  terms: ["autograd", "nn-module", "dataloader", "device", "autocast", "mixed-precision", "bfloat16", "state-dict", "checkpoint", "data-parallelism", "ddp", "all-reduce", "gradient-accumulation"],
  sections: {
    intro: [
      { t: "p", text: "**PyTorch** is a free, open-source deep learning library. It was built at Facebook AI Research and described in Paszke et al. (2019), \"PyTorch: An Imperative Style, High-Performance Deep Learning Library\". On top of a NumPy-like array library it gives you three things:" },
      { t: "list", items: [
        "**Tensors** — arrays that can live on GPUs and other accelerators.",
        "**Autograd** — records each operation as it runs, then computes gradients for you.",
        "**A toolkit**: `nn.Module`, optimizers, data loading, mixed precision, checkpointing and distributed training.",
      ] },
      { t: "p", text: "Last lesson we built a framework by hand: a hand-written `backward` for every layer, CPU only. PyTorch removes those limits but keeps the same picture. You write the forward pass as normal Python; autograd builds the backward pass; `.to(device)` moves the work to a GPU. Our five pieces (layers, container, loss, optimizer, loop) map one-to-one onto `nn.Module`, `nn.Sequential`, `torch.nn.functional`, `torch.optim` and a loop you still write yourself." },
      { t: "p", text: "PyTorch runs \"define-by-run\" (**eager**): the graph is whatever code actually ran. That made it the main research framework, and it is widely used in production (often via `torch.compile`, TorchScript/ONNX export or serving tools). Every program in this lesson was really run on a CPU." },
      { t: "callout", kind: "key", text: "The standard loop: `for xb, yb in loader:` → move to device → `loss = loss_fn(model(xb), yb)` → `opt.zero_grad()` → `loss.backward()` → `opt.step()`. Everything else here — autocast, checkpoints, DDP — wraps this loop without changing its shape." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[A Complete Neural Network in NumPy](/learn/m16/numpy-from-scratch) — the five pieces of a framework. PyTorch is the industrial version, so we refer back to it often.",
        "[Backpropagation](/learn/m04/backpropagation) — autograd *is* reverse-mode backpropagation over a recorded graph.",
        "[NumPy arrays](/learn/m00/numpy-arrays) — dtypes, shapes, views and broadcasting work almost the same for tensors.",
        "[Jacobians and Hessians](/learn/m02/jacobians-hessians) — the vector-Jacobian product, which each node of the autograd graph computes in the backward pass.",
        "[Momentum and adaptive optimizers](/learn/m06/momentum-adaptive) — so you know what state `torch.optim.AdamW` keeps and why a checkpoint must save it.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A lab notebook that writes itself", text: "A chemist writes down each step as she does it: \"mixed A and B, heated it\". If the result is wrong, she reads the notebook **backwards** to see how each earlier amount should have changed. Autograd is that notebook: each operation on a tensor that needs gradients adds an entry (a node that knows how to differentiate it). `loss.backward()` reads the notebook backwards. Work that should not be recorded — evaluation, manual weight updates — happens with the notebook closed (`torch.no_grad()`)." },
      { t: "p", text: "The rest is logistics. A **device** is where a tensor's memory lives (CPU or a particular GPU); an operation's inputs must share one. A **Dataset** answers \"give me example $i$\"; a **DataLoader** turns that into shuffled batches. **Mixed precision** does heavy matrix math in faster, smaller 16-bit numbers and keeps sensitive parts in 32-bit. A **checkpoint** snapshots all state needed to continue as if nothing happened. **Distributed data parallel** runs one model copy per GPU on different data and averages their gradients, so all copies stay identical." },
    ],
    mechanism: [
      { t: "h", text: "1. Tensors, dtypes and devices" },
      { t: "p", text: "A `torch.Tensor` is an n-dimensional array with a `dtype` (float32 by default for Python floats, int64 for ints), a `shape`, strides and a `device`. Strides let `view`, `transpose` and slicing return **views** that share memory, as in NumPy. `torch.from_numpy` shares memory and keeps NumPy's dtype — often float64, a common cause of dtype-mismatch errors. `x.to(device)` returns a copy on that device (or `x` itself if already there)." },
      { t: "h", text: "2. Autograd" },
      { t: "steps", items: [
        { title: "Mark leaves", text: "Parameters are tensors with `requires_grad=True` (`nn.Parameter` sets this for you). Input data normally does not need gradients." },
        { title: "Record during the forward pass", text: "Each operation whose inputs need gradients gives an output with a `grad_fn`: a node (e.g. `MulBackward0`) that keeps what it needs, such as saved tensors, and points to the nodes that made its inputs. The graph is rebuilt every forward pass, so Python `if`s and loops just work." },
        { title: "backward()", text: "`loss.backward()` starts from $\\partial\\mathcal{L}/\\partial\\mathcal{L} = 1$ and visits nodes in reverse topological order; each computes a vector-Jacobian product and passes it to its inputs. Gradients that reach a leaf are **added** into its `.grad`. The graph is then freed (unless `retain_graph=True`)." },
        { title: "Optimizer step and zeroing", text: "`opt.step()` updates each parameter in place from its `.grad`, without recording (it runs under no-grad internally). `opt.zero_grad()` resets `.grad` (to `None` by default in recent versions) so the next backward does not add onto old gradients." },
        { title: "Opting out", text: "`with torch.no_grad():` (or `@torch.no_grad()` / `torch.inference_mode()`) stops recording — use it for evaluation. `t.detach()` returns a tensor that shares `t`'s memory but is cut out of the graph, so no gradient flows through it. Use it for targets in some losses, logging, or stopping gradients on purpose." },
      ] },
      { t: "h", text: "3. nn.Module" },
      { t: "p", text: "Subclass `nn.Module`, call `super().__init__()`, assign layers and `nn.Parameter`s as attributes in `__init__`, and write `forward`. Assigning **registers** them, so `model.parameters()` finds all weights (even in sub-layers), `model.to(device)` moves them, and `model.state_dict()` names them (e.g. `fc1.weight`). `model.train()` / `model.eval()` switch dropout and batch normalization between their two behaviours. Call the model as `model(x)`, not `model.forward(x)`, so hooks run." },
      { t: "h", text: "4. Dataset and DataLoader" },
      { t: "p", text: "A map-style `Dataset` has `__len__` and `__getitem__(i)`. A `DataLoader` wraps it: a sampler produces indices (shuffled each epoch if `shuffle=True`, using an optional seeded `generator`); the indices are grouped into batches; `__getitem__` is called for each; and a **collate** function stacks the results into batch tensors. `num_workers > 0` prepares batches in parallel worker processes; `pin_memory=True` speeds up CPU-to-GPU copies." },
      { t: "h", text: "5. GPU usage" },
      { t: "p", text: "Pick `device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")`. Call `model.to(device)` **before** you create the optimizer, and move each batch with `xb.to(device)`. GPU work runs asynchronously; `.item()` or `.cpu()` makes Python wait for it, so avoid them in the main loop except for occasional logging." },
      { t: "h", text: "6. Mixed precision" },
      { t: "p", text: "Inside `with torch.autocast(device_type, dtype=...)`, ops that are fast and safe in low precision (matrix multiplies, convolutions) run in float16 or bfloat16; sensitive ones (reductions, softmax, losses) stay float32. Parameters stay float32 (\"master weights\") so tiny updates are not lost." },
      { t: "list", items: [
        "With **float16**, small gradients **underflow** (round to zero). A `GradScaler` multiplies the loss by a large factor $S$ before `backward` and divides the gradients by $S$ before the step.",
        "**bfloat16** has float32's exponent range, so it needs no scaler. It is the usual choice on recent GPUs and TPUs, and works on CPUs too (as in our example).",
      ] },
      { t: "h", text: "7. Checkpointing" },
      { t: "p", text: "`model.state_dict()` is an ordered dictionary of parameter and buffer tensors. `opt.state_dict()` holds per-parameter optimizer state (Adam's moments, step counts) and hyperparameters. A resumable checkpoint saves both, plus the epoch and any LR scheduler and gradient-scaler state. To resume **bit-for-bit**, also save the random-number-generator states (dropout and data shuffling). Restore with `load_state_dict` into freshly built objects." },
      { t: "h", text: "8. Distributed data parallel (DDP)" },
      { t: "steps", items: [
        { title: "One process per device", text: "A launcher (`torchrun`) starts $K$ processes. Each joins a process group (`init_process_group`; NCCL backend for GPUs, gloo for CPUs) and knows its `rank`, 0…K−1." },
        { title: "Identical replicas", text: "Wrapping the model in `DistributedDataParallel` copies (broadcasts) rank 0's parameters to every process, so all copies start equal." },
        { title: "Different data", text: "A `DistributedSampler` gives each rank its own non-overlapping slice of every epoch. Call `sampler.set_epoch(epoch)` so the shuffle changes between epochs." },
        { title: "All-reduce during backward", text: "As gradients become ready, DDP groups them into buckets and **all-reduces** each one (every rank ends up with the average over all ranks), overlapping this with the rest of the backward pass." },
        { title: "Identical steps", text: "Every rank applies the same step to the same averaged gradients, so copies stay identical without swapping weights again. Only rank 0 needs to save checkpoints or log." },
      ] },
      { t: "p", text: "When parameters, gradients and optimizer state no longer fit on one device, **sharded** data parallelism (PyTorch's FSDP, based on ZeRO) splits that state across ranks and gathers parameters just in time. Tensor and pipeline parallelism split the model itself. All use the same process groups and communication operations." },
    ],
    math: [
      { t: "h", text: "Autograd computes vector-Jacobian products" },
      { t: "p", text: "Take one operation $y = f(x)$. Its incoming gradient is $\\bar y = \\partial\\mathcal{L}/\\partial y$ (how much the loss changes when $y$ changes). The node computes $\\bar x = \\bar y^\\top J_f(x)$ — a vector-Jacobian product — without building the Jacobian $J_f$. A leaf used in several places gets the **sum** of each path's contribution, which is why `.grad` accumulates." },
      { t: "math", tex: String.raw`\bar x = J_f(x)^\top \bar y,\qquad \bar\theta \mathrel{+}= \sum_{\text{uses } u \text{ of } \theta} J_u^\top \bar y_u` },
      { t: "h", text: "Worked example (the tensor program reproduces it)" },
      { t: "p", text: "Let $w = (2, -1)$ require gradients, $x = (3, 4)$ be data, $y = w\\cdot x$ and $\\mathcal{L} = y^2$." },
      { t: "steps", items: [
        { title: "Forward", tex: String.raw`y = 2\cdot3 + (-1)\cdot4 = 2,\qquad \mathcal{L} = 2^2 = 4` },
        { title: "PowBackward0 (L = y²)", tex: String.raw`\bar y = \frac{\partial\mathcal{L}}{\partial y} = 2y = 4` },
        { title: "SumBackward0 / MulBackward0 (y = Σ w ⊙ x)", tex: String.raw`\bar w = \bar y\, x = 4\,(3, 4) = (12,\ 16)` },
        { title: "A second backward without zeroing adds up", tex: String.raw`w.\mathrm{grad} = (12, 16) + (12, 16) = (24,\ 32)` },
        { title: "Stopping a path: out = u · detach(u), u = w₁ + w₂ = 1", text: "Only the non-detached factor carries gradient. So the result is $u$ (treated as a constant) times $\\partial u/\\partial w$, not $2u$.", tex: String.raw`\frac{\partial\,\mathrm{out}}{\partial w} = \mathrm{detach}(u)\cdot(1, 1) = (1,\ 1)` },
      ] },
      { t: "h", text: "Why averaging gradients across GPUs is exact" },
      { t: "p", text: "Split a batch of $B$ examples into $K$ shards $S_k$, each of size $B/K$. The gradient of the mean loss is" },
      { t: "math", tex: String.raw`\nabla_\theta \frac{1}{B}\sum_{i=1}^{B}\ell_i = \frac{1}{K}\sum_{k=1}^{K}\underbrace{\nabla_\theta\frac{1}{B/K}\sum_{i\in S_k}\ell_i}_{g_k\ \text{(rank } k\text{'s gradient)}}`, caption: "$\\ell_i$ is example $i$'s loss and $g_k$ is the gradient rank $k$ computes on its shard. The average from all-reduce equals the full-batch gradient exactly (up to float rounding) — the scratch code measures a difference of $2.2\\times10^{-16}$." },
      { t: "p", text: "So the **effective batch size** is $K\\times b\\times a$ ($K$ ranks, per-rank batch $b$, $a$ accumulation steps). Example: 8 GPUs × 32 per GPU × 4 accumulation steps $= 1024$. Changing it usually means retuning the learning rate (see the linear scaling rule in [Learning-Rate Schedules](/learn/m06/lr-schedules)). Caveat: if shards differ in size, or a layer uses batch statistics (batch norm), the equality is no longer exact." },
      { t: "h", text: "Number formats for mixed precision" },
      { t: "table", head: ["Format", "Bits (sign, exponent, significand)", "Largest finite", "Smallest positive normal", "Gap between numbers at 1.0"], rows: [
        ["float32", "1, 8, 23", "$\\approx 3.40\\times10^{38}$", "$2^{-126}\\approx1.18\\times10^{-38}$", "$2^{-23}\\approx1.19\\times10^{-7}$"],
        ["float16", "1, 5, 10", "$65504$", "$2^{-14}\\approx6.10\\times10^{-5}$", "$2^{-10}\\approx9.77\\times10^{-4}$"],
        ["bfloat16", "1, 8, 7", "$\\approx 3.39\\times10^{38}$", "$2^{-126}\\approx1.18\\times10^{-38}$", "$2^{-7} = 0.0078125$"],
      ] },
      { t: "p", text: "Exponent bits set the **range**; significand bits set the **precision**. Float16's smallest *subnormal* value is $2^{-24}\\approx5.96\\times10^{-8}$, so a gradient of $10^{-8}$ rounds to 0. Multiplying the loss by $S = 2^{16} = 65536$ multiplies every gradient by $S$ (backpropagation is linear in the upstream gradient). $10^{-8}\\times65536 = 6.55\\times10^{-4}$ fits easily; dividing by $S$ in float32 recovers $\\approx 10^{-8}$. Bfloat16 keeps float32's range (no scaling needed) but has only 8 significant bits: $1 + 1/512$ rounds to exactly 1. That is why sums and master weights stay in float32." },
      { t: "h", text: "Memory of a training run" },
      { t: "p", text: "With float32 weights and Adam, each parameter costs 4 bytes (weight) + 4 (gradient) + 8 (two Adam moments) $= 16$ bytes, before activations. A $10^9$-parameter model needs $16\\times10^9$ bytes $= 16$ GB just for this — the motivation for sharded data parallelism, which divides it across $K$ ranks." },
    ],
    visual: [
      { t: "p", text: "Autograd builds a graph just like the one in this simulator: values flow forward, and `grad_fn` nodes send vector-Jacobian products backward." },
      { t: "sim", id: "chain-rule", caption: "Forward values and backward gradients on a small expression graph — what `loss.backward()` does, node by node." },
      { t: "list", items: [
        "Step through the forward pass and name each node's `grad_fn` (`MulBackward0`, `AddBackward0`, `SigmoidBackward0`, `SubBackward0`, `PowBackward0`).",
        "In the backward pass each clay arrow is one vector-Jacobian product, visited in exactly the reverse of the forward order.",
        "Set $x = 0$: the gradient reaching $w$ becomes 0. In PyTorch too, a weight times a zero input gets `w.grad == 0` for that example (the multiply node's \"swap\" rule).",
      ] },
      { t: "p", text: "PyTorch follows NumPy's broadcasting rules exactly — handy, but a source of silent shape bugs, and the reason a broadcast bias's gradient is a sum over the broadcast axis." },
      { t: "sim", id: "broadcasting", caption: "Line up two shapes and see which axes stretch." },
      { t: "list", items: [
        "Set shapes $(64, 10)$ and $(10,)$ — a batch of logits plus a bias. The bias stretches over the batch axis. In the backward pass PyTorch sums the gradient over that axis to get `bias.grad` of shape $(10,)$.",
        "Try $(64, 1)$ with $(64,)$: the result is $(64, 64)$, not $(64,)$. That is the classic bug when a model outputs $(N, 1)$ and targets are $(N,)$ in an MSE loss. PyTorch warns for some losses; your own arithmetic broadcasts silently.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Three PyTorch mechanisms rebuilt in NumPy, so nothing is magic: a `Dataset`/`DataLoader` with shuffling, batching, collation and `drop_last`; DDP's gradient averaging, checked against the full-batch gradient; and the float facts behind mixed precision (float16 underflow, loss scaling, a bfloat16 rounding function)." },
      { t: "code", lang: "python", title: "pytorch_mechanisms_numpy.py", code: SCRATCH, output: SCRATCH_OUT },
      { t: "list", items: [
        "**DataLoader**: the sampler is a permutation from the loader's own generator; batches are consecutive slices; collation is `np.stack` per field. Both runs print the same first two batches because both loaders use seed 0 — exactly why `DataLoader(..., generator=torch.Generator().manual_seed(0))` makes shuffling reproducible. With `drop_last=True` the short last batch (2 items) is dropped.",
        "**Data parallelism**: each \"GPU\" computes the mean-loss gradient on its 4 examples. Averaging the two equals the gradient on all 8 to machine precision ($2.2\\times10^{-16}$), as the Math section showed.",
        "**Mixed precision**: $10^{-8}$ becomes 0 in float16 but survives after scaling by $2^{16}$ and unscaling in float32 (the small difference, $9.997\\times10^{-9}$, is float16 rounding). `to_bf16` keeps a float32's top 16 bits with round-to-nearest-even — literally what bfloat16 is — so it holds $10^{-8}$ but rounds $1 + 1/512$ to $1$. Float16 overflows to `inf` above 65504, the other reason for dynamic loss scaling.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "h", text: "Tensors and autograd" },
          { t: "code", lang: "python", title: "tensors_autograd.py", code: TENSORS, output: TENSORS_OUT },
          { t: "p", text: "Every number matches the Math section: gradient $(12, 16)$, accumulated $(24, 32)$, and $(1, 1)$ through the detached product. Note that `b.view(3, 2)` shares memory with `b`, and `torch.from_numpy` keeps NumPy's float64." },
          { t: "h", text: "The full pipeline: Dataset, nn.Module, device, autocast, checkpoint and resume" },
          { t: "code", lang: "python", title: "train_pipeline.py", code: PIPELINE, output: PIPELINE_OUT },
          { t: "list", items: [
            "`device:` prints `cpu` here. With a CUDA GPU the same code prints `cuda` and runs there, since the model and every batch go through `.to(device)`.",
            "Inside `torch.autocast(..., dtype=torch.bfloat16)` the linear layers output bfloat16. The logits are cast back with `.float()` so the loss and its gradient start in float32. Parameters stay float32 throughout.",
            "Run B trains 3 epochs and saves the model, optimizer and **both** RNG states (dropout and shuffling). Then it discards everything, rebuilds with a *different* seed, loads the checkpoint and finishes. Its epochs 4–6 and final weights equal run A's exactly (`torch.equal` is `True`).",
            "Bit-exact resuming is fragile: any extra use of a random generator breaks it. A first draft called the model once in training mode to print the autocast dtype; that used up dropout randomness and made run A differ from run B. Hence the `model.eval()` and `torch.no_grad()` around that peek.",
          ] },
          { t: "h", text: "Distributed data parallel: a real two-process run on CPU" },
          { t: "p", text: "Two processes join a gloo process group via a `FileStore` in a temp folder (no network port is opened). Each trains a DDP-wrapped linear model on half the data; we compare with one process trained on all of it." },
          { t: "code", lang: "python", title: "ddp_gloo_demo.py", code: DDP_DEMO, output: DDP_DEMO_OUT },
          { t: "p", text: "Both ranks end with identical weights, equal to the single-process full-batch run to three decimals: two averaged half-batch gradients are the full-batch gradient. On a GPU cluster the structure is the same; only the launcher, backend and sampler change:" },
          { t: "code", lang: "python", title: "train_ddp.py — multi-GPU template (not executed in CI)", code: DDP_TEMPLATE },
          { t: "h", text: "float16 on GPUs: the gradient scaler" },
          { t: "code", lang: "python", title: "fp16_grad_scaler.py — fragment (not executed in CI)", code: FP16_TEMPLATE },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same ideas in TensorFlow: `tf.data` for input pipelines, a global mixed-precision policy, `tf.distribute.MirroredStrategy` for data parallelism on one machine, and `tf.train.Checkpoint` for state you can resume from. See [TensorFlow and Keras](/learn/m16/tensorflow-keras) for the full story." },
          { t: "code", lang: "python", title: "tf_equivalents.py (not executed in CI)", code: TF_EQUIV, output: "Prints the number of replicas (1 on a machine without GPUs, otherwise the number of visible GPUs), then one line per epoch for 3 epochs with loss and accuracy. Writes up to 3 checkpoints to ./ckpts and restores the latest." },
        ] },
      ] },
    ],
    variations: [
      { t: "h", text: "Ways to scale training" },
      { t: "table", head: ["Technique", "What is split or changed", "Pros", "Cons", "Use when"], rows: [
        ["Single device", "Nothing", "Simplest; easiest to debug", "Limited by one device's memory and speed", "Small and medium models"],
        ["Gradient accumulation", "Batch split over time (several backward calls per step)", "Large effective batch on one device", "No speed-up; batch-norm statistics use micro-batches", "Memory limits batch size"],
        ["`nn.DataParallel` (legacy)", "Batch split across GPUs inside one process", "One-line change", "Single-process bottleneck; slower than DDP; not recommended", "Avoid; use DDP"],
        ["DDP", "Data split across processes; a full copy per device", "Near-linear speed-up; standard", "Every device must hold the full model + optimizer state", "Model fits on one device"],
        ["FSDP / ZeRO-style sharding", "Parameters, gradients and optimizer state split across ranks", "Trains models far larger than one device", "More communication; more complex checkpoints", "Model state does not fit on one device"],
        ["Tensor / pipeline parallelism", "Single layers or groups of layers split across devices", "Very large models; combines with data parallelism", "Complex; idle \"pipeline bubbles\"; needs fast interconnect", "Largest models"],
      ] },
      { t: "h", text: "Precision choices" },
      { t: "table", head: ["Mode", "How", "Pros", "Cons"], rows: [
        ["float32", "Default", "Simple, robust", "Slowest on tensor-core hardware; 2× the memory of 16-bit"],
        ["autocast bfloat16", "`torch.autocast(dtype=torch.bfloat16)`", "No loss scaling; float32 range", "Needs hardware support to be fast; only 8 significant bits"],
        ["autocast float16 + GradScaler", "`torch.autocast(dtype=torch.float16)` + `torch.amp.GradScaler`", "Fast on GPUs without bf16 support", "Overflow/underflow handled by dynamic scaling"],
        ["TF32 matmuls", "`torch.backends.cuda.matmul.allow_tf32 = True` on supported NVIDIA GPUs", "Faster with no code changes", "Lower matmul precision (10-bit significand)"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Research prototypes**: eager execution and normal Python control flow make PyTorch the default framework in most papers' public code.",
        "**Large-model training**: DDP and FSDP with bfloat16 autocast, saving a checkpoint every N steps so a hardware failure costs minutes, not days.",
        "**Fine-tuning pretrained models**: load a `state_dict`, freeze some parameters (`requires_grad_(False)`), train the rest — see [Transfer Learning](/learn/m12/transfer-learning).",
        "**Libraries built on top**: torchvision (vision datasets and models), Hugging Face Transformers, PyTorch Lightning (organizes the training loop), and more.",
        "**Deployment**: trained modules are exported (TorchScript, `torch.export`, ONNX) or compiled and served — covered in [Serving](/learn/m18/serving).",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Pythonic and easy to debug**: you can print or step through any tensor in the middle of the forward pass; the graph is whatever ran.",
        "**You write the training loop**: full control (custom losses, several optimizers, GAN-style alternating updates) but more boilerplate than Keras's `fit`; Lightning trades some control for shorter code.",
        "**Eager overhead**: each operation is launched from Python, so small operations are launch-bound. `torch.compile` captures graphs and fuses operations to win speed back, at the cost of compile time and occasional \"graph breaks\".",
        "**Reproducibility needs care**: seed every generator, use deterministic algorithms (`torch.use_deterministic_algorithms(True)`), and keep library versions identical. Some GPU kernels are still nondeterministic by default.",
        "**Distributed training adds failure modes**: hangs when ranks disagree on the number of communication calls, mismatched data shards, checkpoint coordination.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Forgetting `opt.zero_grad()`.** Gradients pile up, so each step gets bigger. Fix: zero once per optimizer step (or accumulate on purpose and scale the loss).",
        "**Evaluating in training mode or with gradients on.** Dropout stays on and the graph wastes memory. Fix: `model.eval()` plus `torch.no_grad()` for evaluation; `model.train()` before training.",
        "**Device mismatch.** \"Expected all tensors to be on the same device\" means a batch, target or new tensor (e.g. `torch.zeros(n)` inside `forward`) is on the CPU. Fix: move every batch; create tensors with `device=x.device`.",
        "**Creating the optimizer before `model.to(device)`.** It usually works (`.to` moves module parameters in place) but is fragile with some wrappers. Fix: move first, then build the optimizer.",
        "**Adding the loss tensor to a running total for logging** (`total += loss`). Each step's graph stays alive until memory runs out. Fix: `total += loss.item()` or `loss.detach()`.",
        "**Applying softmax before `F.cross_entropy`.** It expects raw logits (it applies log-softmax itself); a double softmax trains poorly. Fix: return logits.",
        "**Saving only the model in a checkpoint.** A fresh optimizer resets Adam's moments and step count (and the LR schedule), causing a loss spike. Fix: also save optimizer, scheduler, scaler, epoch and RNG states.",
        "**DDP: not calling `sampler.set_epoch(epoch)`**, so every epoch uses the same shuffle; or saving checkpoints from every rank at once. Fix: set the epoch; save from rank 0 only (and load with `map_location`).",
        "**Wrong dtypes**: float64 inputs from NumPy into a float32 model, or float labels into `cross_entropy` (it needs int64 class indices). Fix: convert explicitly with `dtype=torch.float32` / `torch.long`.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Why do we call `optimizer.zero_grad()` every iteration?", a: "`backward()` adds gradients into each parameter's `.grad` instead of overwriting it. Without zeroing, all earlier steps' gradients would be summed into this update. That is only useful on purpose (gradient accumulation over micro-batches)." },
      { t: "qa", level: "beginner", q: "What do `model.train()` and `model.eval()` change?", a: "They set the `training` flag on every submodule. Dropout is active only in training; batch normalization uses batch statistics in training and running statistics in eval. They do **not** turn off gradient tracking — that is `torch.no_grad()`." },
      { t: "qa", level: "beginner", q: "What is the difference between `torch.no_grad()` and `tensor.detach()`?", a: "`no_grad` is a block where nothing is recorded (for inference and manual updates). `detach()` returns one tensor that shares memory with the original but has no graph history, so no gradient flows back through it; the rest of the computation is still recorded." },
      { t: "qa", level: "intermediate", q: "What should a checkpoint contain to resume training exactly?", a: "The model `state_dict`; the optimizer `state_dict` (moments, step counts, hyperparameters); LR scheduler and gradient-scaler state; the epoch/step counter; and RNG states (torch CPU/CUDA, NumPy, Python, and the DataLoader generator). With all of these and deterministic kernels, resuming reproduces the uninterrupted run, as the pipeline program shows." },
      { t: "qa", level: "intermediate", q: "Why does fp16 mixed precision need loss scaling but bf16 does not?", a: "Float16 has a 5-bit exponent. Values below about $3\\times10^{-8}$ (half the smallest subnormal, $2^{-24}$) round to zero, and many gradients are that small. Scaling the loss by $S$ scales all gradients by $S$ into float16's range; they are unscaled in float32 before the update, and steps with inf/nan are skipped while $S$ is lowered. Bfloat16 has float32's 8-bit exponent, so small gradients do not underflow." },
      { t: "qa", level: "intermediate", q: "Explain how DDP keeps model replicas synchronized.", a: "When DDP is created, rank 0's parameters are broadcast to all ranks. During `backward`, hooks all-reduce gradient buckets, so every rank holds the average gradient over all ranks' data. Every rank then takes the same step on the same gradients from the same weights, so the copies stay identical without swapping weights again." },
      { t: "qa", level: "intermediate", q: "What is the effective batch size with 4 GPUs, per-GPU batch 16 and 8 gradient-accumulation steps?", a: "$4\\times16\\times8 = 512$. Each optimizer step averages gradients over 512 examples, which affects how you should set the learning rate." },
      { t: "qa", level: "advanced", q: "Why is DDP faster than the older `nn.DataParallel`?", a: "`DataParallel` is one process with several threads. Every iteration it splits inputs, re-copies the model to each GPU and gathers outputs on one GPU, held back by Python's GIL and the overloaded main GPU. DDP uses one process per GPU, copies the model once, and overlaps the bucketed gradient all-reduce with the backward pass, so communication is mostly hidden." },
      { t: "qa", level: "advanced", q: "Training with autocast bf16 matches float32 loss early on but stalls later. What could be wrong?", a: "Bfloat16 has only 8 significant bits, so anything accumulated in bf16 can stall learning. A common cause: casting the parameters themselves to bf16 (`model.bfloat16()`) instead of using autocast, so updates smaller than about 1/256 of a weight's size round away. Keep master weights and optimizer state in float32, compute losses and reductions in float32 (autocast does this for its listed ops), and check custom operations." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "$w = (1, 2)$ requires grad, $x = (2, -1)$, $\\mathcal{L} = (w\\cdot x)^2$. What is the first component of `w.grad` after `L.backward()`?", answer: 0, tolerance: 0.0001, explain: "$w\\cdot x = 2 - 2 = 0$, so $\\partial\\mathcal{L}/\\partial w = 2(w\\cdot x)x = 0$." },
        { kind: "numeric", q: "Same setup but $x = (3, 1)$. You call `backward()` twice without zeroing. What is the first component of `w.grad`?", answer: 60, tolerance: 0.0001, explain: "$w\\cdot x = 5$. One backward gives $2\\cdot5\\cdot3 = 30$; two add up to 60." },
        { q: "Which call stops gradients flowing through one tensor but keeps recording the rest of the computation?", options: ["torch.no_grad()", "tensor.detach()", "model.eval()", "optimizer.zero_grad()"], answer: 1, explain: "`detach()` cuts one tensor out of the graph. `no_grad` stops all recording, and `eval()` only changes how layers behave." },
        { kind: "numeric", q: "8 GPUs, per-GPU batch 64, 2 accumulation steps. What is the effective batch size?", answer: 1024, tolerance: 0, explain: "$8\\times64\\times2 = 1024$." },
        { q: "Why does bfloat16 not need a GradScaler?", options: ["It has more significand bits than float16", "It has the same exponent range as float32, so small gradients do not underflow", "PyTorch scales it automatically", "It is only used for inference"], answer: 1, explain: "Bfloat16 keeps float32's 8-bit exponent. (It actually has fewer significand bits than float16.)" },
        { kind: "numeric", q: "How many GB does float32 Adam training state (weights, gradients, two moments) take for a 7-billion-parameter model, ignoring activations? (1 GB = $10^9$ bytes)", answer: 112, tolerance: 0.5, explain: "$16$ bytes/parameter $\\times 7\\times10^9 = 112\\times10^9$ bytes." },
        { q: "In DDP, what does the all-reduce compute?", options: ["The sum of the models' weights", "The average of the gradients over all ranks, delivered to every rank", "The loss on rank 0 only", "A copy of rank 0's gradients to the others"], answer: 1, explain: "Every rank gets the average of all ranks' gradients, so all take the same step." },
        { q: "You resume from a checkpoint that stored only `model.state_dict()`. What usually happens with Adam?", options: ["Nothing; training continues identically", "A loss spike, because the moment estimates and step count restart from zero", "An error at load time", "The learning rate doubles permanently"], answer: 1, explain: "With fresh Adam state, every parameter takes a bias-corrected step of about the learning rate, whatever its history — usually a visible jump in the loss." },
      ] },
      { t: "exercise", title: "Loss scaling arithmetic", prompt: "In float16, the smallest positive subnormal is $2^{-24}$ and the largest finite value is 65504. A model's gradients range from $3\\times10^{-9}$ to $0.5$. (a) Which gradients underflow without scaling? (b) Choose a power-of-two loss scale $S$ that makes the smallest gradient representable without making the largest overflow. (c) What range of $S$ works?", hint: "You need $S\\cdot3\\times10^{-9} \\ge 2^{-24}$ and $S\\cdot0.5 \\le 65504$.", solution: [
        { t: "steps", items: [
          { title: "(a) Underflow threshold", text: "$2^{-24} \\approx 5.96\\times10^{-8}$. Every gradient below about $3\\times10^{-8}$ (half the smallest subnormal, the rounding midpoint) becomes 0 — including the smallest, $3\\times10^{-9}$." },
          { title: "(b) Lower bound", tex: String.raw`S \ge \frac{5.96\times10^{-8}}{3\times10^{-9}} \approx 19.9 \ \Rightarrow\ S \ge 2^{5} = 32` },
          { title: "Upper bound", tex: String.raw`S \le \frac{65504}{0.5} = 131008 \ \Rightarrow\ S \le 2^{16} = 65536` },
          { title: "(c) Valid range", text: "Any power of two from $2^5 = 32$ to $2^{16} = 65536$. A larger $S$ also keeps small gradients more precise: they land in the normal range (above $2^{-14}$) when $S \\ge 2^{-14}/(3\\times10^{-9}) \\approx 20345$, i.e. $S \\ge 2^{15}$. PyTorch's `GradScaler` finds such a value on the fly: start high, halve $S$ after any overflow, grow it after a run of good steps." },
        ] },
      ] },
      { t: "exercise", title: "Gradient accumulation equals a big batch", prompt: "Show in PyTorch that accumulating gradients over 4 micro-batches of 16 (calling `backward()` on each) gives the same gradient as one batch of 64. What must you do to each micro-batch loss, and why?", hint: "`F.cross_entropy` returns the mean over its batch.", solution: [
        { t: "code", lang: "python", title: "grad_accumulation.py", code: ACCUM_SOL, output: ACCUM_SOL_OUT },
        { t: "p", text: "Each micro-batch loss is a mean over 16 examples. The big-batch loss is a mean over 64, which equals the average of the four micro-batch means. `.grad` **sums** the four backward calls, so you must divide each loss by the number of accumulation steps (4) to turn that sum into the average. Without it the gradient is 4× too large — like a 4× learning rate. The leftover difference ($\\sim10^{-8}$) is float32 summation order." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Tensors: NumPy-like arrays with a dtype (float32 default), shape, strides (views) and a device; operands must share a device.",
        "Autograd records `grad_fn` nodes in the forward pass; `backward()` computes vector-Jacobian products in reverse and **adds** into `.grad`. Use `zero_grad`, `no_grad`, `detach` on purpose.",
        "`nn.Module` registers parameters and submodules; `state_dict` names them; `train()`/`eval()` switch dropout and batch norm.",
        "Dataset (`__len__`, `__getitem__`) + DataLoader (sampler → batches → collate), with a seeded generator for reproducible shuffling.",
        "Standard loop: to(device) → forward → loss → zero_grad → backward → step. Evaluate under `eval()` + `no_grad()`.",
        "Mixed precision: autocast runs matmuls in bf16/fp16, keeps sensitive ops and master weights in float32; fp16 needs loss scaling ($10^{-8}\\cdot2^{16}$ survives); bf16 does not.",
        "Checkpoints: model + optimizer (+ scheduler, scaler, epoch, RNG states) → exact resumption.",
        "DDP: one process per device, broadcast starting weights, all-reduce (average) gradients during backward; effective batch $= K\\times b\\times a$.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[TensorFlow and Keras](/learn/m16/tensorflow-keras) — the other major framework, its high-level `fit` API and `GradientTape`, and a fair comparison with PyTorch.",
        "[Hyperparameter Tuning, Tracking and Reproducibility](/learn/m17/experimentation) — seeds, checkpoints and logging are the raw material for experiments you can trust.",
        "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — what happens to a trained `state_dict` next: export, serving and deployment.",
      ] },
    ],
  },
};

export default lesson;
