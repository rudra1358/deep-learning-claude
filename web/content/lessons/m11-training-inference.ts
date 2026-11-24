import type { Lesson } from "@/lib/types";

const CLM_NUMPY = String.raw`import numpy as np

# ---------------- a tiny decoder-only Transformer in NumPy (random weights) ----------------
rng = np.random.default_rng(0)
V, D, F, L, N_MAX = 12, 16, 64, 2, 32    # vocab, d_model, FFN width, layers, max positions


def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


P = {"tok": rng.normal(0, 0.5, (V, D)), "pos": rng.normal(0, 0.5, (N_MAX, D)), "layers": []}
for _ in range(L):
    P["layers"].append({k: rng.normal(0, D ** -0.5, s) for k, s in
                        [("Wq", (D, D)), ("Wk", (D, D)), ("Wv", (D, D)), ("Wo", (D, D)),
                         ("W1", (D, F)), ("W2", (F, D))]})


def forward(tokens, causal=True):
    """Pre-LN decoder: returns next-token logits for EVERY position, shape (T, V)."""
    T = len(tokens)
    x = P["tok"][tokens] + P["pos"][:T]
    for lp in P["layers"]:
        h = layer_norm(x)
        q, k, v = h @ lp["Wq"], h @ lp["Wk"], h @ lp["Wv"]
        s = q @ k.T / np.sqrt(D)
        if causal:
            s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)   # block j > i
        x = x + softmax(s) @ v @ lp["Wo"]
        x = x + np.maximum(0, layer_norm(x) @ lp["W1"]) @ lp["W2"]
    return layer_norm(x) @ P["tok"].T          # weight tying: output layer = embedding^T


seq = np.array([3, 7, 1, 4, 4, 9, 2, 5, 8])    # a training sequence of 9 token ids
inputs, targets = seq[:-1], seq[1:]             # shift by one: predict token t+1 from tokens <= t
print("inputs :", inputs)
print("targets:", targets)

# 1) ONE masked pass gives the prediction for every prefix at once ...
parallel = forward(inputs)
# 2) ... identical to running the model separately on each prefix (T passes)
sequential = np.stack([forward(inputs[:t + 1])[-1] for t in range(len(inputs))])
print(f"max |parallel - sequential| with causal mask : {np.abs(parallel - sequential).max():.2e}")
leaky = forward(inputs, causal=False)
print(f"max |parallel - sequential| without the mask : {np.abs(leaky - sequential).max():.2e}")

# 3) the causal-LM loss: mean cross-entropy over all T positions
logp = np.log(softmax(parallel))
nll = -logp[np.arange(len(targets)), targets]
print("per-position NLL:", nll.round(3))
print(f"mean cross-entropy {nll.mean():.4f} nats, perplexity {np.exp(nll.mean()):.3f} (uniform = {V})")

# 4) without the mask the model can "cheat": targets[t] = inputs[t+1] is visible
print(f"a mask-free model sees the answer: inputs[1:] == targets[:-1] -> {np.array_equal(inputs[1:], targets[:-1])}")

# 5) label smoothing: target distribution (1 - eps) * one_hot + eps / V
eps = 0.1
q = np.full((len(targets), V), eps / V)
q[np.arange(len(targets)), targets] += 1 - eps
ls_direct = -(q * logp).sum(-1).mean()
ls_formula = (1 - eps) * nll.mean() + eps * (-logp.mean(-1)).mean()
print(f"label-smoothed loss (eps=0.1): direct {ls_direct:.4f}  formula {ls_formula:.4f}")`;

const CLM_NUMPY_OUT = String.raw`inputs : [3 7 1 4 4 9 2 5]
targets: [7 1 4 4 9 2 5 8]
max |parallel - sequential| with causal mask : 2.97e-15
max |parallel - sequential| without the mask : 3.28e+00
per-position NLL: [5.566 3.42  9.875 4.562 3.005 8.037 0.371 2.576]
mean cross-entropy 4.6765 nats, perplexity 107.392 (uniform = 12)
a mask-free model sees the answer: inputs[1:] == targets[:-1] -> True
label-smoothed loss (eps=0.1): direct 4.6195  formula 4.6195`;

const KV_NUMPY = String.raw`import numpy as np

# Same tiny decoder as before (random weights), now with an incremental KV-cache path.
rng = np.random.default_rng(0)
V, D, F, L, N_MAX = 12, 16, 64, 2, 64


def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


P = {"tok": rng.normal(0, 0.5, (V, D)), "pos": rng.normal(0, 0.5, (N_MAX, D)), "layers": []}
for _ in range(L):
    P["layers"].append({k: rng.normal(0, D ** -0.5, s) for k, s in
                        [("Wq", (D, D)), ("Wk", (D, D)), ("Wv", (D, D)), ("Wo", (D, D)),
                         ("W1", (D, F)), ("W2", (F, D))]})
attn_macs = 0   # multiply-accumulates spent in q.k and weights.v (the O(n) / O(n^2) part)


def forward(tokens, cache=None, start=0):
    """Process 'tokens' (positions start..start+T-1). If 'cache' (a list of per-layer [K, V])
    is given, append the new keys/values to it and attend over everything cached so far."""
    global attn_macs
    T = len(tokens)
    x = P["tok"][tokens] + P["pos"][start:start + T]
    for l, lp in enumerate(P["layers"]):
        h = layer_norm(x)
        q, k, v = h @ lp["Wq"], h @ lp["Wk"], h @ lp["Wv"]
        if cache is not None:
            cache[l][0] = np.concatenate([cache[l][0], k])   # K grows by T rows
            cache[l][1] = np.concatenate([cache[l][1], v])
            k, v = cache[l]
        n = k.shape[0]                                       # total keys visible
        s = q @ k.T / np.sqrt(D)
        visible = np.arange(n)[None, :] <= (np.arange(T) + n - T)[:, None]
        s = np.where(visible, s, -np.inf)
        attn_macs += 2 * T * n * D
        x = x + softmax(s) @ v @ lp["Wo"]
        x = x + np.maximum(0, layer_norm(x) @ lp["W1"]) @ lp["W2"]
    return layer_norm(x) @ P["tok"].T


def generate_no_cache(prompt, n_new):
    toks = list(prompt)
    for _ in range(n_new):
        logits = forward(np.array(toks))            # recompute the WHOLE prefix every step
        toks.append(int(logits[-1].argmax()))       # greedy decoding
    return toks


def generate_with_cache(prompt, n_new):
    cache = [[np.zeros((0, D)), np.zeros((0, D))] for _ in range(L)]
    logits = forward(np.array(prompt), cache, 0)    # PREFILL: whole prompt in one pass
    toks = list(prompt) + [int(logits[-1].argmax())]
    for _ in range(n_new - 1):                      # DECODE: one new token per step
        logits = forward(np.array(toks[-1:]), cache, len(toks) - 1)
        toks.append(int(logits[-1].argmax()))
    return toks, cache


prompt, n_new = [3, 7, 1, 4], 40
attn_macs = 0
a = generate_no_cache(prompt, n_new)
macs_plain = attn_macs
attn_macs = 0
b, cache = generate_with_cache(prompt, n_new)
macs_cache = attn_macs
print("generated (no cache):", a[4:20], "...")
print("identical token sequences:", a == b)
print(f"attention MACs  no cache {macs_plain:,}   with cache {macs_cache:,}   ratio {macs_plain / macs_cache:.1f}x")
print("cache shapes per layer (K, V):", [(c[0].shape, c[1].shape) for c in cache])
print(f"cache floats per token = 2 * n_layers * d_model = {2 * L * D}")

# ---------------- memory of a KV cache for realistic shapes ----------------
def kv_bytes_per_token(n_layers, d_model, bytes_per_value=2):
    return 2 * n_layers * d_model * bytes_per_value        # K and V, every layer, fp16/bf16

for name, nl, dm, ctx in [("12 layers, d=768 ", 12, 768, 1024), ("32 layers, d=4096", 32, 4096, 4096)]:
    per_tok = kv_bytes_per_token(nl, dm)
    print(f"{name}: {per_tok:,} B/token = {per_tok / 2**10:.0f} KiB; "
          f"{ctx} tokens -> {per_tok * ctx / 2**20:,.0f} MiB per sequence")

# ---------------- FLOPs of one layer: attention-matrix part vs linear part ----------------
d = 768
print(f"\nper-layer forward FLOPs, d_model = {d} (FFN width 4d)")
for n in [256, 1024, 4608, 16384]:
    linear = 2 * n * (4 * d * d + 8 * d * d)      # Q,K,V,O projections (4d^2) + FFN (8d^2), 2 FLOPs/MAC
    attn = 2 * 2 * n * n * d                      # QK^T and AV, each n*n*d MACs
    print(f"n={n:6d}  linear {linear / 1e9:8.2f} GFLOP   attention {attn / 1e9:8.2f} GFLOP   attention share {attn / (attn + linear):.0%}")`;

const KV_NUMPY_OUT = String.raw`generated (no cache): [8, 8, 6, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8] ...
identical token sequences: True
attention MACs  no cache 1,754,880   with cache 60,928   ratio 28.8x
cache shapes per layer (K, V): [((43, 16), (43, 16)), ((43, 16), (43, 16))]
cache floats per token = 2 * n_layers * d_model = 64
12 layers, d=768 : 36,864 B/token = 36 KiB; 1024 tokens -> 36 MiB per sequence
32 layers, d=4096: 524,288 B/token = 512 KiB; 4096 tokens -> 2,048 MiB per sequence

per-layer forward FLOPs, d_model = 768 (FFN width 4d)
n=   256  linear     3.62 GFLOP   attention     0.20 GFLOP   attention share 5%
n=  1024  linear    14.50 GFLOP   attention     3.22 GFLOP   attention share 18%
n=  4608  linear    65.23 GFLOP   attention    65.23 GFLOP   attention share 50%
n= 16384  linear   231.93 GFLOP   attention   824.63 GFLOP   attention share 78%`;

const TRAIN_TORCH = String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)                                # tiny model: one thread is fastest and reproducible
# Opening of "Alice's Adventures in Wonderland" (Lewis Carroll, 1865) - public domain.
TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
In another moment down went Alice after it, never once considering how in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."""
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in TEXT])
split = int(0.9 * len(data))
train, val = data[:split], data[split:]                 # last 10% held out
V, CTX, D, H, LAYERS = len(chars), 48, 64, 4, 2
print(f"{len(TEXT)} chars, vocab {V}, train {len(train)}, val {len(val)}")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))   # (B, H, T, d_k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)            # causal mask
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.ln_f, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.ln_f(self.blocks(x)))        # (B, T, V) logits


def batch(src, B=24):
    s = torch.randint(0, len(src) - CTX - 1, (B,))
    x = torch.stack([src[i:i + CTX] for i in s])          # inputs  t_0 .. t_{T-1}
    y = torch.stack([src[i + 1:i + CTX + 1] for i in s])  # targets t_1 .. t_T (shifted by one)
    return x, y


@torch.no_grad()
def val_loss():
    xs = torch.stack([val[i:i + CTX] for i in range(0, len(val) - CTX - 1, 4)])
    ys = torch.stack([val[i + 1:i + CTX + 1] for i in range(0, len(val) - CTX - 1, 4)])
    return F.cross_entropy(model(xs).flatten(0, 1), ys.flatten()).item()


@torch.no_grad()
def sample(prompt, n=60, temperature=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):                                    # autoregressive loop
        logits = model(idx[:, -CTX:])[0, -1]
        if temperature == 0:
            nxt = logits.argmax().view(1, 1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=g).view(1, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return "".join(chars[i] for i in idx[0].tolist()).replace("\n", " ")


model = TinyGPT()
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
decay = [p for p in model.parameters() if p.dim() >= 2]          # weight decay on matrices only
no_decay = [p for p in model.parameters() if p.dim() < 2]        # not on biases / LayerNorm gains
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-3, betas=(0.9, 0.95))
STEPS, WARMUP = 600, 50
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / WARMUP if s < WARMUP else
                                          0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP))))

for step in range(STEPS + 1):
    if step % 100 == 0:
        model.eval()
        vl = val_loss()
        tr = f"{loss.item():.3f}" if step else "  -  "
        print(f"step {step:4d}  lr {sched.get_last_lr()[0]:.2e}  train {tr}  val {vl:.3f}  val ppl {math.exp(vl):6.2f}")
        if step in (0, 100, 300, STEPS):
            print("   sample:", sample("Alice ", 60))
        model.train()
    if step == STEPS:
        break
    x, y = batch(train)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())   # all 48 positions at once
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)              # gradient clipping
    opt.step()
    sched.step()
model.eval()
print("T=0.8 :", sample("The Rabbit ", 80, 0.8, seed=1))
print("T=0.8 :", sample("She found ", 80, 0.8, seed=2))`;

const TRAIN_TORCH_OUT = String.raw`1656 chars, vocab 43, train 1490, val 166
parameters: 108,672
step    0  lr 6.00e-05  train   -    val 3.881  val ppl  48.50
   sample: Alice tSS?:IrynuWro! A-S?lm?jgipywnuWr!T;euWhitystSSSSSSSSSSSSSSSS
step  100  lr 2.95e-03  train 2.164  val 2.412  val ppl  11.16
   sample: Alice the t our the the athe the abur the t the the abbbur the abu
step  200  lr 2.53e-03  train 1.743  val 2.376  val ppl  10.76
step  300  lr 1.84e-03  train 1.148  val 2.717  val ppl  15.14
   sample: Alice withe to the tought of it it a waisho to stco o o "Ohe de da
step  400  lr 1.09e-03  train 0.751  val 3.188  val ppl  24.24
step  500  lr 5.14e-04  train 0.538  val 3.498  val ppl  33.06
step  600  lr 3.00e-04  train 0.428  val 3.674  val ppl  39.41
   sample: Alice withe wout pictures or conversations in ity-pocket, and looo
T=0.8 : The Rabbit with ay toust al itThe alfte way tooy Rabbbbit ity tlay tof weelastce oook,  ont
T=0.8 : She found it itlf sis iel  The rabbit ser seen ran noun the cor the taf ther oubof se sits`;

const KV_TORCH = String.raw`import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)
V, CTX, D, H, LAYERS = 256, 512, 256, 8, 4      # random weights: we only measure speed/equality


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
        if cache is not None:                       # cache = dict with tensors (B, H, n_past, d_k)
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        n_past = k.shape[2] - T
        # query i (absolute position n_past+i) may see keys 0..n_past+i
        mask = torch.arange(k.shape[2])[None, :] <= (torch.arange(T)[:, None] + n_past)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)

    def forward(self, idx, caches=None, start=0):
        x = self.tok(idx) + self.pos(torch.arange(start, start + idx.shape[1]))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))


@torch.inference_mode()
def generate(model, prompt, n_new, use_cache):
    idx, last_logits = prompt, []
    if use_cache:
        caches = [{} for _ in range(LAYERS)]
        logits = model(idx, caches, 0)                       # prefill
        for _ in range(n_new):
            last_logits.append(logits[:, -1])
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], 1)
            logits = model(nxt, caches, idx.shape[1] - 1)    # decode: ONE token through the net
    else:
        for _ in range(n_new):
            logits = model(idx)                              # full recompute of the prefix
            last_logits.append(logits[:, -1])
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return idx, torch.stack(last_logits)


model = TinyGPT().eval()
prompt = torch.randint(0, V, (1, 16))
for n_new in [64, 128, 256]:
    t0 = time.perf_counter(); a, la = generate(model, prompt, n_new, use_cache=False); t1 = time.perf_counter()
    b, lb = generate(model, prompt, n_new, use_cache=True); t2 = time.perf_counter()
    print(f"{n_new:3d} new tokens: no cache {t1 - t0:6.2f} s, KV cache {t2 - t1:5.2f} s, "
          f"speed-up {(t1 - t0) / (t2 - t1):4.1f}x | same tokens: {torch.equal(a, b)}, "
          f"max logit diff {(la - lb).abs().max().item():.1e}")

# prefill vs decode: tokens per second in each phase
with torch.inference_mode():
    long_prompt = torch.randint(0, V, (1, 256))
    caches = [{} for _ in range(LAYERS)]
    t0 = time.perf_counter(); model(long_prompt, caches, 0); t1 = time.perf_counter()
    nxt = torch.zeros(1, 1, dtype=torch.long)
    for s in range(64):
        model(nxt, caches, 256 + s)
    t2 = time.perf_counter()
print(f"prefill: 256 tokens in {t1 - t0:.3f} s ({256 / (t1 - t0):,.0f} tok/s); "
      f"decode: 64 tokens in {t2 - t1:.3f} s ({64 / (t2 - t1):,.0f} tok/s)")`;

const KV_TORCH_OUT = String.raw` 64 new tokens: no cache   1.02 s, KV cache  0.39 s, speed-up  2.6x | same tokens: True, max logit diff 1.4e-06
128 new tokens: no cache   3.02 s, KV cache  0.59 s, speed-up  5.1x | same tokens: True, max logit diff 1.4e-06
256 new tokens: no cache   8.88 s, KV cache  1.20 s, speed-up  7.4x | same tokens: True, max logit diff 1.5e-06
prefill: 256 tokens in 0.057 s (4,504 tok/s); decode: 64 tokens in 0.317 s (202 tok/s)`;

const KERAS = String.raw`import numpy as np
import tensorflow as tf
from tensorflow import keras

text = open("alice.txt", encoding="utf-8").read()          # any plain-text corpus
chars = sorted(set(text))
ids = np.array([chars.index(c) for c in text])
V, CTX, D, H = len(chars), 64, 64, 4

# windows of CTX+1 characters -> (inputs, targets shifted by one)
win = np.stack([ids[i:i + CTX + 1] for i in range(0, len(ids) - CTX - 1, 4)])
x, y = win[:, :-1], win[:, 1:]


class Block(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = keras.layers.LayerNormalization(), keras.layers.LayerNormalization()
        self.att = keras.layers.MultiHeadAttention(num_heads=H, key_dim=D // H)
        self.ffn = keras.Sequential([keras.layers.Dense(4 * D, activation="gelu"), keras.layers.Dense(D)])

    def call(self, x):
        h = self.ln1(x)
        x = x + self.att(h, h, use_causal_mask=True)        # causal self-attention
        return x + self.ffn(self.ln2(x))


inp = keras.Input((CTX,), dtype="int32")
pos = keras.layers.Embedding(CTX, D)(tf.range(CTX))
h = keras.layers.Embedding(V, D)(inp) + pos
for _ in range(2):
    h = Block()(h)
logits = keras.layers.Dense(V)(keras.layers.LayerNormalization()(h))
model = keras.Model(inp, logits)

schedule = keras.optimizers.schedules.CosineDecay(0.0, decay_steps=1000, warmup_target=3e-3,
                                                  warmup_steps=100, alpha=0.1)
model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=0.1, global_clipnorm=1.0),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x, y, batch_size=32, epochs=5)`;

const lesson: Lesson = {
  id: "m11/training-inference",
  title: "Training, Autoregressive Generation and Complexity",
  summary:
    "How Transformers are trained (causal LM, masked LM and seq2seq teacher forcing, with shifted targets, label smoothing and the AdamW + warm-up + clipping recipe) and how they generate text (the sequential decode loop, the KV cache, decoding strategies, prefill vs decode, batching) — with the cost of each phase derived and measured on a tiny Transformer trained from scratch.",
  minutes: 90,
  level: "advanced",
  prerequisites: ["m11/transformer-architecture", "m10/scaled-dot-product-attention", "m02/information-theory", "m09/rnn-basics"],
  next: ["m11/transformer-families", "m20/decoding", "m20/llm-inference", "m13/language-modeling"],
  terms: ["causal-language-modeling", "teacher-forcing", "label-smoothing", "kv-cache", "prefill", "autoregressive-generation", "perplexity"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "A Transformer has two lives. During **training** it is shown complete pieces of text and must predict, at every position, what comes next (or what was hidden). During **inference** it is given a prompt and must produce new text one token at a time, feeding each output back as input. The two phases use the same weights but have completely different computational shapes, and most of what makes modern language models practical comes from understanding that difference." },
      { t: "p", text: "The decisive property of the Transformer is that **training is parallel over positions**. A recurrent network must process token 1 before token 2 before token 3, so a sequence of 2048 tokens means 2048 dependent steps. A Transformer decoder with a **causal mask** computes the predictions for all 2048 positions in one pass of large matrix multiplications — and those predictions are *exactly* the ones it would have made had it read each prefix separately. That equivalence (proved and checked numerically below) is what let Transformers absorb far more data per unit of compute than RNNs, and it is the main reason they replaced them." },
      { t: "p", text: "**Generation**, in contrast, is inherently sequential: token $t+1$ cannot be computed before token $t$ has been chosen. Done naively, every new token re-processes the entire prefix and the total cost grows with the cube of the length. The **KV cache** — storing each layer's keys and values for tokens already processed — removes the redundant work and is used by essentially every production inference system. It trades compute for memory, and that memory ($2\\cdot n_{\\text{layers}}\\cdot d_{\\text{model}}$ numbers per token) becomes the limiting resource when serving long contexts to many users." },
      { t: "p", text: "This lesson covers the training objectives (causal language modelling, masked language modelling, sequence-to-sequence teacher forcing), the practical details (shifting inputs and targets, label smoothing, AdamW with warm-up and decay, gradient clipping, mixed precision), a tiny Transformer trained from scratch with a real loss curve, perplexity and samples, and then inference: the decode loop, the KV cache with measured speed-up, decoding strategies, prefill vs decode, batching, and the complexity of every phase." },
      { t: "callout", kind: "key", text: "Training: one causally-masked pass scores all $T$ next-token predictions at once — cost per layer $O(T^2 d + T d^2)$, fully parallel. Inference: $N$ dependent steps; with a KV cache each step costs $O(t\\,d + d^2)$ per layer, plus $2\\cdot n_{\\text{layers}}\\cdot d_{\\text{model}}$ cached numbers per token." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[The Transformer Architecture](/learn/m11/transformer-architecture) — the decoder block (masked self-attention, feed-forward network, residuals, LayerNorm). This lesson treats the block as known and focuses on how it is trained and run.",
        "[Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) — the formula $\\mathrm{softmax}(QK^\\top/\\sqrt{d_k}+M)V$ and in particular the **causal mask** $M$ with $-\\infty$ above the diagonal. The parallel-training argument rests entirely on it.",
        "[Information theory](/learn/m02/information-theory) — cross-entropy as the expected negative log-probability of the true outcome, and perplexity $= e^{\\text{cross-entropy}}$.",
        "[Recurrent Neural Networks](/learn/m09/rnn-basics) — where **teacher forcing** and **exposure bias** were introduced. We reuse both ideas and show why Transformers make teacher forcing parallel.",
        "[Learning-rate schedules](/learn/m06/lr-schedules) — linear warm-up and cosine decay; and [Momentum and adaptive optimizers](/learn/m06/momentum-adaptive) for Adam. We only add AdamW's decoupled weight decay here.",
        "[Tokenization and embeddings](/learn/m11/tokens-embeddings) — text becomes a sequence of integer token ids; the model outputs a vector of logits over the vocabulary at every position.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "An exam sheet with a sliding cover card", text: "Imagine an exam: a paragraph of text, and under every word the question “what is the next word?”. A student who could see the whole paragraph would cheat trivially, so each question is printed with a card covering everything to its right. Crucially, all the questions are on **one sheet**: the examiner hands it out once and the student answers every question at the same time. That is causal-LM training — one forward pass, every position answered, the causal mask playing the card. Generation is a different exercise: writing a new story one word at a time, where you cannot write word 10 until you have decided word 9. The **KV cache** is your notes: instead of re-reading the whole story from the start before each new word, you keep a summary of each earlier word that you can consult instantly." },
      { t: "p", text: "Why is the one-sheet trick legitimate? Because, with the cover card in place, the answer to question $t$ depends only on words $1\\dots t$ — exactly the information it would have if it were asked alone. Nothing about answering question $t$ changes when questions $t+1, t+2,\\dots$ also sit on the page. So training on all positions at once gives $T$ honest training examples for the price of one pass." },
      { t: "p", text: "Generation cannot use the trick, because the inputs of later positions do not exist yet — they are the model's own future outputs. The model must therefore run once per new token. Each run, however, only needs to process the **one** new token: everything the earlier tokens contribute to attention (their keys and values) was already computed and can be reused. Without that reuse, writing a 1000-token answer would mean re-reading 1, 2, 3, …, 1000 tokens — about half a million token-reads instead of a thousand." },
      { t: "p", text: "The same training trick appears in other forms. A **masked language model** (BERT) covers random words instead of the future and asks for them back. An **encoder–decoder** translator is shown the true translation shifted by one position — **teacher forcing** — so it too answers every output position in parallel during training, and only becomes sequential at inference." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "One training step of a decoder-only (causal) language model" },
      { t: "steps", items: [
        { title: "Sample a window of text", text: "Take $T+1$ consecutive token ids $x_0, x_1, \\dots, x_T$ from the corpus (a batch holds $B$ such windows)." },
        { title: "Shift to form inputs and targets", text: "Inputs are $x_0\\dots x_{T-1}$, targets are $x_1\\dots x_T$: the target at position $t$ is the input at position $t+1$. In code: `x = w[:, :-1]`, `y = w[:, 1:]`." },
        { title: "Embed", text: "Look up token embeddings and add position information (learned, sinusoidal or rotary): a $(B, T, d_{\\text{model}})$ tensor." },
        { title: "Run all layers with the causal mask", text: "Every self-attention layer masks keys $j > i$ for query $i$. Feed-forward layers and LayerNorm act on each position separately, so no information flows from right to left anywhere in the network." },
        { title: "Project to logits", text: "A final LayerNorm and a linear layer (often tied to the embedding matrix) give $(B, T, |\\mathcal{V}|)$ logits — one next-token distribution per position." },
        { title: "Cross-entropy over all positions", text: "Flatten to $(BT, |\\mathcal{V}|)$ and average $-\\log p(\\text{target})$ over all $BT$ positions (ignoring padding). One loss, $BT$ supervised predictions." },
        { title: "Backward, clip, update, schedule", text: "Backpropagate, clip the global gradient norm (typically to 1.0), take an AdamW step, then advance the learning-rate schedule (warm-up, then decay)." },
      ] },
      { t: "h", text: "Masked LM and sequence-to-sequence variants of the same step" },
      { t: "list", items: [
        "**Masked language modelling (MLM)**: no shift and no causal mask. About 15% of input tokens are selected and (mostly) replaced by `[MASK]`; the loss is computed only at the selected positions, predicting the original token from **both** sides. Details and the 80/10/10 rule are in [BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families).",
        "**Encoder–decoder with teacher forcing**: the encoder reads the source (bidirectionally). The decoder input is the target sequence shifted right with a start token, `[BOS] y1 y2 … y(m-1)`, and its targets are `y1 y2 … ym [EOS]`. With a causal mask in the decoder's self-attention and cross-attention to the encoder, all $m$ target positions are trained in one pass — the parallel form of the teacher forcing you met in [RNNs](/learn/m09/rnn-basics).",
      ] },
      { t: "h", text: "Autoregressive generation with a KV cache" },
      { t: "steps", items: [
        { title: "Prefill", text: "Run the whole prompt (length $n_p$) through the network in one causally-masked pass, exactly like training. Store every layer's keys $K$ and values $V$ for all prompt positions in the cache. The logits at the last position give the distribution of the first new token." },
        { title: "Choose a token", text: "Turn the logits into a token: greedy argmax, or sample after temperature / top-k / top-p filtering (see the variations section and [Decoding](/learn/m20/decoding))." },
        { title: "Decode step", text: "Feed **only the new token** (with its absolute position index $t$). In every layer compute its $q, k, v$; append $k, v$ to that layer's cache; attend with $q$ over all $t+1$ cached keys. No mask is needed: everything in the cache is in the past." },
        { title: "Repeat until done", text: "Stop at an end-of-sequence token, a stop string, or a maximum length. The cache grows by $2\\cdot n_{\\text{layers}}\\cdot d_{\\text{model}}$ numbers per token." },
      ] },
      { t: "callout", kind: "warning", text: "Two details make cached and uncached generation identical: the new token must receive the **correct absolute position** (not position 0), and the attention of the new query must cover the **whole cache including its own key**. Getting either wrong produces fluent-looking but different text — so always test that the cached loop reproduces the uncached one, as both code sections below do." },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "The autoregressive factorization and the causal-LM loss" },
      { t: "p", text: "Any joint distribution over a sequence factorizes exactly by the chain rule of probability:" },
      { t: "math", tex: String.raw`p(x_1,\dots,x_T) = \prod_{t=1}^{T} p(x_t \mid x_1,\dots,x_{t-1})` },
      { t: "p", text: String.raw`A causal language model parameterizes each conditional with the same network $f_\theta$: the logits at position $t$ are $z_t = f_\theta(x_{\le t})_t \in \mathbb{R}^{|\mathcal V|}$ and $p_\theta(\cdot \mid x_{\le t}) = \mathrm{softmax}(z_t)$. Maximum likelihood on a corpus is minimizing the average negative log-likelihood, i.e. the token-level cross-entropy:` },
      { t: "math", tex: String.raw`\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=0}^{T-1} \log p_\theta\big(x_{t+1} \mid x_0,\dots,x_t\big), \qquad \mathrm{PPL} = e^{\mathcal L}`, caption: "$T$ = number of predicted positions, $x_t$ = token id at position $t$, $|\\mathcal V|$ = vocabulary size, PPL = perplexity." },
      { t: "p", text: "Perplexity is the exponentiated average surprise: a model with perplexity $k$ is, on average, as uncertain as a uniform choice among $k$ tokens. A uniform model over $|\\mathcal V|$ tokens has perplexity exactly $|\\mathcal V|$; a randomly initialised model is usually slightly worse, because its random logits are confidently wrong some of the time." },
      { t: "h", text: "Why one masked pass equals T separate passes" },
      { t: "p", text: String.raw`Claim: with a causal mask, the hidden state $h^{(\ell)}_t$ at every layer $\ell$ depends only on $x_{\le t}$. Proof by induction over layers. At layer 0, $h^{(0)}_t = E[x_t] + P_t$ depends only on $x_t$. Suppose it holds for layer $\ell-1$. Self-attention computes $a_t = \sum_{j} \alpha_{tj} v_j$ with $\alpha_{tj}=0$ for $j>t$ (because $e^{-\infty}=0$), so $a_t$ uses only $h^{(\ell-1)}_{\le t}$. LayerNorm, the residual additions and the feed-forward network act on position $t$ alone. Hence $h^{(\ell)}_t$ depends only on $x_{\le t}$. Consequently the output at position $t$ of a pass over $x_{0:T}$ equals the last output of a pass over the prefix $x_{0:t}$ — the extra tokens to the right are invisible.` },
      { t: "p", text: "The NumPy code below checks this on a 2-layer decoder: the largest difference between the parallel logits and eight separate prefix passes is $3\\times10^{-15}$ (floating-point round-off). Removing the mask makes the difference $3.3$ — and, worse, lets position $t$ read $x_{t+1}$, which is its own target, so the training loss could be driven to zero without learning anything useful." },
      { t: "h", text: "Worked example: cross-entropy and perplexity" },
      { t: "p", text: "Suppose for a 3-token target the model assigns probabilities $0.5$, $0.25$ and $0.8$ to the correct next tokens." },
      { t: "steps", items: [
        { title: "Per-position losses", tex: String.raw`-\ln 0.5 = 0.693147,\quad -\ln 0.25 = 1.386294,\quad -\ln 0.8 = 0.223144` },
        { title: "Mean cross-entropy", tex: String.raw`\mathcal L = \tfrac13(0.693147+1.386294+0.223144) = 0.767528 \text{ nats}` },
        { title: "Perplexity", tex: String.raw`\mathrm{PPL} = e^{0.767528} = 2.154435 = \big(0.5\cdot0.25\cdot0.8\big)^{-1/3} = 10^{1/3}` },
      ] },
      { t: "p", text: "The last identity shows that perplexity is the inverse **geometric mean** of the probabilities given to the true tokens — one very bad prediction (probability near 0) hurts far more than many mediocre ones." },
      { t: "h", text: "Label smoothing" },
      { t: "p", text: String.raw`Instead of the one-hot target $y$, train against a softened distribution $q = (1-\varepsilon)\,y + \varepsilon/|\mathcal V|$ (Szegedy et al., 2016; used with $\varepsilon = 0.1$ in the original Transformer of Vaswani et al., 2017). With $\log p$ the model's log-probabilities and $c$ the correct class:` },
      { t: "math", tex: String.raw`\mathcal L_{\text{LS}} = -\sum_{k} q_k \log p_k = (1-\varepsilon)\,\big(-\log p_c\big) + \varepsilon\,\Big(-\frac{1}{|\mathcal V|}\sum_{k}\log p_k\Big), \qquad \frac{\partial \mathcal L_{\text{LS}}}{\partial z_k} = p_k - q_k` },
      { t: "p", text: String.raw`The gradient $p-q$ vanishes when $p=q$, so the loss is minimized by predicting probability $1-\varepsilon+\varepsilon/|\mathcal V|$ on the correct token, not 1: the logits stay finite and the model is discouraged from over-confidence. It is a regularizer, and it changes the meaning of the loss — a label-smoothed loss is not a log-likelihood, so report perplexity with $\varepsilon=0$.` },
      { t: "steps", items: [
        { title: "Logits and softmax (|V| = 4, correct class 0, ε = 0.1)", tex: String.raw`z=(2,1,0,-1),\ e^{z}=(7.389056,\,2.718282,\,1,\,0.367879),\ \textstyle\sum=11.475217,\ p=(0.643914,\,0.236883,\,0.087144,\,0.032059)` },
        { title: "Log-probabilities", tex: String.raw`\log p = (-0.44019,\ -1.44019,\ -2.44019,\ -3.44019)` },
        { title: "Plain cross-entropy and the uniform term", tex: String.raw`-\log p_0 = 0.44019,\qquad -\tfrac14\textstyle\sum_k \log p_k = 1.94019` },
        { title: "Smoothed loss", tex: String.raw`\mathcal L_{\text{LS}} = 0.9(0.44019) + 0.1(1.94019) = 0.59019 \;=\; -\textstyle\sum_k q_k\log p_k \text{ with } q=(0.925,0.025,0.025,0.025)` },
        { title: "Optimal confidence", tex: String.raw`p_c^\star = 1-\varepsilon+\varepsilon/|\mathcal V| = 0.925,\qquad z_c - z_{k\neq c} = \ln(0.925/0.025) = \ln 37 \approx 3.61` },
      ] },
      { t: "h", text: "The training recipe: AdamW, warm-up + decay, clipping, mixed precision" },
      { t: "p", text: String.raw`**AdamW** (Loshchilov and Hutter, 2019) applies weight decay directly to the weights instead of adding $\lambda\theta$ to the gradient, so the decay is not rescaled by Adam's per-parameter denominator. With Adam moments $\hat m_t, \hat v_t$ (see [Momentum and adaptive optimizers](/learn/m06/momentum-adaptive)), learning rate $\eta_t$ and decay $\lambda$:` },
      { t: "math", tex: String.raw`\theta_{t} = \theta_{t-1} - \eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big)` },
      { t: "p", text: "Decay is normally applied to weight matrices only, not to biases, LayerNorm gains or (often) embeddings. Language-model recipes commonly use $\\beta_2 = 0.95$ instead of 0.999 so the second-moment estimate adapts faster to gradient spikes; the GPT-3 paper (Brown et al., 2020) reports $\\beta_2=0.95$, weight decay 0.1, global-norm clipping at 1.0 and cosine decay to 10% of the peak learning rate after a linear warm-up." },
      { t: "p", text: String.raw`**Warm-up then decay** (see [Learning-rate schedules](/learn/m06/lr-schedules)). Early in training Adam's second-moment estimates are poor and post-LN Transformers in particular are unstable, so the learning rate ramps up linearly over $W$ steps and then decays. The schedule used in our training script, with peak $\eta_{\max}=3\times10^{-3}$, $W=50$, $S=600$ total steps and a floor of 10%:` },
      { t: "math", tex: String.raw`\eta_s = \begin{cases} \eta_{\max}\,\dfrac{s+1}{W} & s < W \\[2mm] \eta_{\max}\Big(0.1 + 0.9\cdot\tfrac12\big(1+\cos\tfrac{\pi (s-W)}{S-W}\big)\Big) & s \ge W \end{cases}`, caption: "Values: $\\eta_0 = 6\\times10^{-5}$, $\\eta_{24} = 1.5\\times10^{-3}$, $\\eta_{49}=\\eta_{50}=3\\times10^{-3}$, $\\eta_{325}=1.65\\times10^{-3}$ (half-way through the decay), $\\eta_{599}\\approx3\\times10^{-4}$." },
      { t: "p", text: String.raw`**Gradient clipping by global norm.** Let $g$ be all gradients concatenated. If $\|g\|_2 > c$, rescale: $g \leftarrow g\cdot c/\|g\|_2$. The direction is kept, only the step length is bounded, which protects against rare loss spikes (a bad batch, an attention logit blowing up).` },
      { t: "p", text: "**Mixed precision.** Matrix multiplications run in 16-bit floats (fp16 or bfloat16) while a 32-bit master copy of the weights receives the updates. fp16 has 5 exponent bits (largest finite value 65504), so small gradients underflow; it needs **loss scaling** (multiply the loss by a large factor before backward, divide the gradients after). bfloat16 keeps float32's 8 exponent bits with fewer mantissa bits, so it rarely needs loss scaling and is the usual choice on hardware that supports it. Halving the bytes roughly halves activation memory and lets tensor cores run several times faster than fp32." },
      { t: "h", text: "Computational complexity of training" },
      { t: "p", text: String.raw`For one layer, sequence length $n$, width $d$ and an FFN of width $4d$, count multiply-accumulates (MACs; 1 MAC = 2 FLOPs). The four projections $W_Q,W_K,W_V,W_O$ cost $4nd^2$; the FFN costs $2\cdot n\cdot d\cdot 4d = 8nd^2$; the score matrix $QK^\top$ costs $n^2 d$ and the weighted sum $AV$ another $n^2 d$ (summed over heads, since heads split $d$).` },
      { t: "math", tex: String.raw`\text{FLOPs per layer (forward)} \approx \underbrace{24\,n\,d^2}_{\text{linear layers}} + \underbrace{4\,n^2 d}_{\text{attention}} = O(n d^2 + n^2 d), \qquad 4n^2 d = 24 n d^2 \iff n = 6d` },
      { t: "p", text: "For $d = 768$ the two terms are equal at $n = 6d = 4608$: at $n=1024$ attention is 18% of a layer's FLOPs (14.50 vs 3.22 GFLOP), at $n=16384$ it is 78%. The backward pass costs about twice the forward pass, so a training step is about 3× these numbers. Memory for the attention weights is $O(n^2)$ per head unless a fused kernel such as FlashAttention (Dao et al., 2022) avoids materializing them. All $n$ positions are processed by dense matrix products, so training saturates parallel hardware." },
      { t: "h", text: "Computational complexity of inference, and the KV cache" },
      { t: "p", text: String.raw`Generating token $t$ **without** a cache reruns the whole prefix: $O(t\,d^2 + t^2 d)$ per layer. Summed over $N$ generated tokens this is $O(N^2 d^2 + N^3 d)$. **With** a cache only the new token is processed: its projections and FFN cost $O(d^2)$ and its attention over $t$ cached keys costs $O(t\,d)$, so a step is $O(d^2 + t\,d)$ per layer and the whole generation $O(N d^2 + N^2 d)$ — a factor of about $N$ less. In the NumPy experiment below (prompt 4, 40 new tokens, 2 layers) the attention work falls from 1,754,880 to 60,928 MACs, a 28.8× reduction, with identical output tokens.` },
      { t: "p", text: String.raw`**Cache memory.** Each layer stores one key vector and one value vector of length $d_{\text{model}}$ (all heads concatenated) per token:` },
      { t: "math", tex: String.raw`\text{KV bytes per token} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot b, \qquad \text{total} = (\text{bytes per token})\times n_{\text{tokens}} \times \text{batch}`, caption: "$b$ = bytes per number (2 for fp16/bf16). With grouped-query attention only $n_{kv}$ of the $h$ heads store K/V, replacing $d_{\\text{model}}$ by $n_{kv}\\,d_{\\text{head}}$." },
      { t: "steps", items: [
        { title: "12 layers, d = 768, fp16 (the GPT-2 small shape)", tex: String.raw`2\cdot12\cdot768\cdot2 = 36{,}864 \text{ B} = 36\text{ KiB/token};\quad \times 1024 \text{ tokens} = 36 \text{ MiB per sequence}` },
        { title: "32 layers, d = 4096, fp16 (a 7-billion-parameter shape)", tex: String.raw`2\cdot32\cdot4096\cdot2 = 524{,}288 \text{ B} = 512\text{ KiB/token};\quad \times 4096 = 2048 \text{ MiB} = 2 \text{ GiB per sequence}` },
      ] },
      { t: "p", text: String.raw`**Prefill vs decode.** Prefill processes $n_p$ prompt tokens with large matrix–matrix products: like training, it is **compute-bound** and fast per token. A decode step processes one token per sequence: every weight matrix is read from memory to do a matrix–*vector* product, so it is **memory-bandwidth-bound**. As an illustration, a 7-billion-parameter model in fp16 occupies 14 GB; if memory delivers 1 TB/s, merely reading the weights takes $14\times10^{9}/10^{12} = 14$ ms per step, capping one sequence at about 70 tokens/s no matter how fast the arithmetic units are. Our PyTorch measurement shows the same asymmetry at toy scale: prefill ran at about 4,500 tokens/s, decode at about 200 tokens/s.` },
      { t: "p", text: String.raw`**Batching** is the remedy: decoding $B$ sequences together reads each weight once and uses it for $B$ tokens, raising throughput almost $B$-fold until compute becomes the limit — but each sequence brings its own KV cache, so batch size is limited by cache memory. Serving systems therefore batch continuously (requests join and leave the batch at every step) and manage cache memory in pages.` },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "p", text: "The first explorer runs a small decoder-only Transformer that was **actually trained** in PyTorch as a character-level language model on Chapter I of *Alice's Adventures in Wonderland* — the same kind of model as the one trained in the PyTorch section below — and executes it in your browser. The stage bar walks through Tokens → Embed → each layer's LN1 / QKV / Attn / W_O / Add / FFN / Add → Logits, and the **Generate** panel runs the autoregressive loop." },
      { t: "sim", id: "transformer", caption: "A trained decoder-only Transformer running in your browser: follow a prompt to its next-character distribution, then generate." },
      { t: "list", ordered: true, items: [
        "Type a prompt (e.g. `Alice was`) and jump to the **Logits** stage. Click different positions in the prompt: **every** position has its own next-character distribution. That is what training scores — all positions at once, each seeing only its prefix. Generation uses only the distribution after the last character.",
        "Go to an **Attn** stage, pick a late position and switch between heads: the weights on later characters are exactly zero. That causal mask is why the parallel training pass is legitimate.",
        "In the Generate panel press the single-step button a few times with **greedy** selected. Each new character is appended and, as the panel says, the full forward pass re-runs: this simulator deliberately recomputes the prefix every step, i.e. generation **without** a KV cache. With a cache, only the new position's column of each stage would need computing.",
        "Switch to **sample** and move the **Temperature T** slider: at low T the top character dominates and the output resembles greedy; at T near 2 the distribution flattens and the text degrades. Change the seed to get a different sample from the same distribution.",
        "Keep generating past the context length: the panel reports characters dropping out of the context window — a fixed context with learned positions cannot attend further back.",
        "The model has memorized much of its training chapter (its held-out loss is far above its training loss, as the panel states) — the same overfitting you will see in the training curve below.",
      ] },
      { t: "p", text: "The attention explorer shows the mask itself. Turn on **Causal mask** and step to the masked-scores stage: every cell above the diagonal becomes $-\\infty$, and after softmax the first row is exactly $[1, 0, \\dots, 0]$. Toggle the mask off and on and compare the last row: it is identical in both cases, because the last token sees everything anyway. That last row is exactly what a decode step with a KV cache computes." },
      { t: "sim", id: "attention", caption: "Turn on the causal mask: row i may only attend to columns j ≤ i." },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "**1. Causal-LM training objective, from scratch.** A 2-layer pre-LN decoder with random weights (weight-tied output layer). The script shifts a sequence into inputs and targets, proves the parallel/sequential equivalence numerically, shows what happens without the mask, computes the loss and perplexity, and checks the label-smoothing identity." },
      { t: "code", lang: "python", title: "causal_lm_objective.py", code: CLM_NUMPY, output: CLM_NUMPY_OUT },
      { t: "list", items: [
        "`inputs, targets = seq[:-1], seq[1:]` is the whole of the shift: 9 tokens give 8 training predictions.",
        "`parallel` is one pass; `sequential` stacks the last-position logits of 8 separate prefix passes. Their difference ($2.97\\times10^{-15}$) is round-off. Without the mask the difference is 3.28: positions now see the future.",
        "The random model's perplexity (107.4) exceeds the uniform value 12: random logits are confidently wrong (position 3 gives the target a log-probability of $-9.9$). Training first has to *un-learn* this over-confidence; it is the same reason the trained PyTorch model starts above its vocabulary size.",
        "The two label-smoothing computations agree (4.6195), confirming $\\mathcal L_{\\text{LS}} = (1-\\varepsilon)\\,\\mathrm{CE} + \\varepsilon\\cdot\\text{mean}_k(-\\log p_k)$.",
      ] },
      { t: "p", text: "**2. Generation with and without a KV cache, from scratch.** The same decoder gains a `cache` argument: a list holding, per layer, the keys and values of every token processed so far. Generation is greedy so outputs can be compared token by token. The script also counts the attention multiply-accumulates, evaluates the cache-memory formula for two realistic shapes and tabulates the attention vs linear FLOPs of one layer." },
      { t: "code", lang: "python", title: "kv_cache_from_scratch.py", code: KV_NUMPY, output: KV_NUMPY_OUT },
      { t: "list", items: [
        "`visible` generalizes the causal mask to the cached case: query $i$ of the new chunk sits at absolute position $n-T+i$ and may see keys $0\\dots n-T+i$. In prefill ($T = n$) it is the ordinary causal mask; in a decode step ($T=1$) it is all-true.",
        "Identical token sequences confirm the cache is exact, not an approximation. (The repeated token 8 is what greedy decoding of an untrained model looks like.)",
        "The cache holds 43 positions = 4 prompt + 39 fed-back tokens (the last generated token is never fed back), each storing $2\\cdot L\\cdot D = 64$ floats.",
        "The FLOP table reproduces the crossover $n = 6d = 4608$ derived in the math section.",
      ] },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**Training a tiny decoder-only Transformer from scratch.** Character-level, 2 layers, $d_{\\text{model}}=64$, 4 heads, context 48, about 109k parameters, trained for 600 steps on the opening paragraphs of *Alice's Adventures in Wonderland* (Lewis Carroll, 1865, public domain), with the full recipe: shifted targets, `F.scaled_dot_product_attention(..., is_causal=True)`, AdamW with $\\beta_2=0.95$ and decay on matrices only, linear warm-up + cosine decay, and gradient clipping. The last 10% of the text is held out. It runs in about 30 s on one CPU thread." },
          { t: "code", lang: "python", title: "train_tiny_gpt.py", code: TRAIN_TORCH, output: TRAIN_TORCH_OUT },
          { t: "p", text: "Read the curve carefully — it is an honest small-data experiment:" },
          { t: "list", items: [
            "At step 0 the validation perplexity is 48.5 for a 43-symbol vocabulary: worse than uniform, and the sample is random characters.",
            "By step 100 the model has learned character frequencies and common words (`the`), validation perplexity 11.2; at step 200 it reaches its best, **10.76**.",
            "From then on **training** loss keeps falling (to 0.428 nats, perplexity 1.53) while **validation** loss rises to 3.674 (perplexity 39.4): with only 1,490 training characters the model memorizes. The final greedy sample reproduces training phrases (`without pictures or conversations`), and sampled continuations splice memorized fragments together.",
            "This is why real language models are trained on corpora so large that each token is seen about once: there is no room to memorize, and validation loss tracks training loss. Here, early stopping at step 200 would select the best model.",
          ] },
          { t: "p", text: "**KV-cache speed-up, measured.** A randomly initialised model (4 layers, $d=256$, 8 heads) generates greedily with and without a cache. The cached path appends each layer's keys and values and passes an explicit mask so that the prefill chunk is causal. Timings depend on the machine and its load; the token equality does not." },
          { t: "code", lang: "python", title: "kv_cache_timing.py (timings vary between runs)", code: KV_TORCH, output: KV_TORCH_OUT },
          { t: "list", items: [
            "The speed-up grows with length (2.6× → 5.1× → 7.4×) because uncached cost grows quadratically in the number of generated tokens (each step re-reads the prefix) while cached cost grows roughly linearly.",
            "Tokens are identical; logits differ by about $10^{-6}$, the float32 round-off from summing in a different order. Ties in argmax could in principle flip a token after such round-off, which is why the test compares tokens *and* reports the logit difference.",
            "Prefill handled about 4,500 tokens/s; decode about 200 tokens/s — a ~20× gap from processing tokens as a matrix rather than one vector at a time.",
          ] },
          { t: "callout", kind: "tip", title: "Mixed precision in PyTorch", text: "On a GPU, wrap the forward pass and loss in `with torch.autocast(\"cuda\", dtype=torch.bfloat16):` and keep the optimizer on the fp32 parameters. For fp16, add `scaler = torch.amp.GradScaler()` and use `scaler.scale(loss).backward()`, `scaler.unscale_(opt)` (before clipping), `scaler.step(opt)`, `scaler.update()`." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same character-level causal LM in Keras. `MultiHeadAttention(..., use_causal_mask=True)` applies the causal mask; `CosineDecay` with `warmup_steps` gives linear warm-up then cosine decay to `alpha` × peak; `AdamW(..., global_clipnorm=1.0)` adds decoupled weight decay and global-norm clipping." },
          { t: "code", lang: "python", title: "train_tiny_gpt_keras.py (not executed in CI)", code: KERAS, output: "Prints a Keras progress bar with the training loss for each of 5 epochs; the loss falls from roughly ln(vocabulary size) as the model learns the text." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "h", text: "Training objectives" },
      { t: "table", head: ["Objective", "Inputs → targets", "Mask", "Advantages", "Disadvantages", "Used by"], rows: [
        ["Causal LM (next-token)", "$x_{0:T-1}\\to x_{1:T}$", "Causal", "Every position is a training example; the model can generate directly", "Each position sees only left context", "GPT family, most modern LLMs"],
        ["Masked LM", "Corrupted text → original tokens at ~15% positions", "Bidirectional", "Representations use both sides of context", "Only ~15% of positions give loss; cannot generate left-to-right; `[MASK]` absent at fine-tuning", "BERT, RoBERTa"],
        ["Seq2seq + teacher forcing", "Source; `[BOS]`+$y_{1:m-1}\\to y_{1:m}$`[EOS]`", "Bidirectional encoder, causal decoder, cross-attention", "Natural for input→output tasks", "Two stacks; exposure bias at inference", "Original Transformer, T5, BART, Whisper"],
        ["Span corruption (denoising)", "Text with spans replaced by sentinels → the missing spans", "Encoder–decoder", "Short targets, efficient pretraining", "Pretraining task differs from generation", "T5"],
        ["Prefix LM", "Prefix (bidirectional) → continuation (causal)", "Prefix-LM", "Bidirectional reading of the input inside one decoder stack", "Mask bookkeeping", "UniLM, T5 experiments"],
      ] },
      { t: "h", text: "Decoding strategies (brief — see Decoding for depth)" },
      { t: "table", head: ["Strategy", "Rule", "Advantages", "Disadvantages"], rows: [
        ["Greedy", "$\\arg\\max_k p_k$ at every step", "Deterministic, cheap", "Repetitive, bland; not the most likely *sequence*"],
        ["Beam search", "Keep the $b$ best partial sequences by total log-probability", "Good for translation / ASR with a clear right answer", "$b$× compute; generic, repetitive in open-ended text"],
        ["Temperature $\\tau$", "Sample from $\\mathrm{softmax}(z/\\tau)$", "One knob from deterministic ($\\tau\\to0$) to random", "High $\\tau$ samples nonsense from the long tail"],
        ["Top-k", "Sample among the $k$ most likely tokens (renormalized)", "Cuts the tail", "Fixed $k$ is too many for peaked, too few for flat distributions"],
        ["Top-p (nucleus)", "Sample among the smallest set with cumulative probability ≥ $p$", "Adapts the candidate set size to the distribution", "Still a heuristic; interacts with temperature"],
      ] },
      { t: "p", text: "Full treatment, with worked examples, in [Decoding: greedy, beam, temperature, top-k, top-p](/learn/m20/decoding)." },
      { t: "h", text: "Inference optimizations built on the KV cache" },
      { t: "table", head: ["Technique", "Idea", "What it saves"], rows: [
        ["Multi-query / grouped-query attention (Shazeer, 2019; Ainslie et al., 2023)", "Share K/V heads across groups of query heads", "Cache memory and bandwidth by the group factor"],
        ["Paged KV cache (Kwon et al., 2023, vLLM)", "Store the cache in fixed-size blocks like virtual memory", "Fragmentation; enables large dynamic batches"],
        ["Continuous batching", "Add and remove sequences from the running batch at every step", "Idle slots; raises throughput"],
        ["Speculative decoding (Leviathan et al., 2023; Chen et al., 2023)", "A small model drafts several tokens; the large model verifies them in one pass", "Sequential large-model steps, with an unchanged output distribution"],
        ["Fused attention kernels (FlashAttention)", "Compute softmax attention in tiles without materializing the $n\\times n$ matrix", "Memory traffic in training and prefill"],
        ["KV-cache quantization", "Store K/V in 8 or fewer bits", "Cache memory, at some accuracy risk"],
      ] },
      { t: "p", text: "These are covered in [Efficient Inference: KV Cache, FlashAttention, MoE](/learn/m20/llm-inference)." },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Pretraining large language models**: the causal-LM objective over trillions of tokens with the AdamW / warm-up / cosine / clipping / bf16 recipe described here — see [Pretraining LLMs](/learn/m20/llm-pretraining).",
        "**Chat assistants and code completion** generate token by token with a KV cache; prompt processing (prefill) and token streaming (decode) are scheduled and billed separately by many serving systems because their costs differ so much.",
        "**Machine translation and speech recognition** use encoder–decoder Transformers trained with teacher forcing (the original Transformer for translation; OpenAI's Whisper for speech, an encoder–decoder trained on audio–transcript pairs) and decoded with beam search or greedy search.",
        "**Representation models** (BERT-style encoders for search, classification and retrieval) are trained with masked LM and need no generation loop at all: inference is a single bidirectional pass.",
        "**Capacity planning for serving**: the formula $2\\cdot n_{\\text{layers}}\\cdot d_{\\text{model}}\\cdot b$ per token decides how many concurrent users and how long a context fit on one accelerator.",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "list", items: [
        "**Training parallelism** (strength): one pass yields $T$ supervised predictions per sequence, with dense matrix products that use accelerators efficiently. RNNs could not do this.",
        "**Quadratic attention** (weakness): $O(n^2 d)$ compute and, without fused kernels, $O(n^2)$ memory per head. Below $n\\approx 6d$ the linear layers dominate; beyond it attention does, which is why long-context training is expensive.",
        "**Sequential decoding** (weakness): $N$ tokens need $N$ dependent forward passes. Latency per token cannot be reduced by adding more hardware to a single sequence beyond a point; speculative decoding and smaller models are the main levers.",
        "**KV cache** (trade-off): it reduces generation compute by a factor of about $N$ but costs memory linear in context length × batch × layers × width. For long contexts the cache can exceed the model weights in size.",
        "**Teacher forcing** (trade-off): fast, stable, parallel training, at the price of **exposure bias** — the model never sees its own mistakes during training. In practice, with large models and data, this matters less than early RNN work suggested, but it explains degenerate loops in long generations.",
        "**Label smoothing** (trade-off): better calibration and often better accuracy/BLEU for classification and translation, but the loss is no longer a likelihood and perplexity gets worse; most LLM pretraining does not use it.",
      ] },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Forgetting to shift the targets** (`y = x` instead of `y = x[1:]`). With a causal mask the model then learns to copy its own input — the loss drops quickly to almost zero and generation is garbage. Fix: inputs `w[:, :-1]`, targets `w[:, 1:]`, or let the library shift (HF causal-LM heads shift internally — do not shift twice).",
        "**Missing or wrong causal mask** (e.g. `is_causal=False`, or a mask with `True` meaning *blocked* when the API expects `True` = *allowed*). The model sees the future, training loss collapses, validation generation fails. Check with the parallel-vs-prefix test from this lesson.",
        "**Including padding in the loss.** Padded positions must be excluded (`ignore_index=-100` in `F.cross_entropy`) and masked as keys; otherwise the model is trained to predict padding and the loss average is diluted.",
        "**Wrong position index in cached decoding**: feeding the new token at position 0, or recomputing positions relative to the new chunk. Fix: pass the absolute offset `start = len(prefix)`; verify cached and uncached outputs match.",
        "**Comparing perplexities across tokenizers.** Perplexity per token depends on what a token is: character-level and subword models (or two subword vocabularies) are not comparable. Compare bits per character/byte instead.",
        "**Reporting label-smoothed loss as perplexity.** The smoothed loss has a floor above zero; compute evaluation loss with $\\varepsilon=0$.",
        "**No warm-up with Adam on a Transformer.** Large early updates driven by poor second-moment estimates can push the model into divergence or a dead state; a few hundred to a few thousand warm-up steps are standard.",
        "**Evaluating in training mode.** Dropout stays active and results become noisy; call `model.eval()` (and use `torch.no_grad()`/`inference_mode()` to avoid storing activations).",
        "**Thinking the KV cache changes the model's output.** It is an exact reuse of computation; if outputs differ, the cache implementation has a bug.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the training objective of a GPT-style model?", a: "Causal language modelling: at every position, predict the next token given all previous tokens. The loss is the average cross-entropy $-\\log p_\\theta(x_{t+1}\\mid x_{\\le t})$ over all positions. Inputs are the sequence without its last token; targets are the sequence without its first token." },
      { t: "qa", level: "beginner", q: "What is teacher forcing?", a: "Feeding the ground-truth previous tokens as decoder inputs during training, instead of the model's own predictions. For Transformers it means the decoder receives the true target sequence shifted right by one; combined with the causal mask, all positions are trained in parallel. At inference the true tokens are unavailable, so the model consumes its own outputs (exposure bias)." },
      { t: "qa", level: "beginner", q: "What does a perplexity of 20 mean?", a: "The model's average per-token uncertainty is equivalent to choosing uniformly among 20 options: $\\mathrm{PPL} = e^{\\mathcal L}$, so the mean cross-entropy is $\\ln 20 \\approx 3.0$ nats. Lower is better; a uniform model over $|\\mathcal V|$ tokens has perplexity $|\\mathcal V|$." },
      { t: "qa", level: "beginner", q: "Why is text generation slower per token than training?", a: "Generation is sequential: token $t+1$ depends on token $t$, so each token needs its own forward pass, and a decode step processes a single vector per sequence — the hardware is limited by reading the weights from memory, not by arithmetic. Training processes all positions of many sequences in large matrix products." },
      { t: "qa", level: "intermediate", q: "Prove that a causally masked forward pass gives the same predictions as running each prefix separately.", a: "By induction over layers: embeddings at position $t$ depend only on $x_t$; attention at position $t$ gives zero weight to keys $j>t$, so it combines only states $\\le t$; LayerNorm, residuals and FFN are per-position. Hence every hidden state at $t$ depends only on $x_{\\le t}$, and the output at $t$ equals the last output of a pass on $x_{\\le t}$. Numerically the difference is floating-point round-off ($\\sim10^{-15}$ in float64)." },
      { t: "qa", level: "intermediate", q: "What is the KV cache, and how much memory does it need?", a: "During generation, the keys and values of past tokens never change (causal mask), so each layer stores them and a new token only computes its own $q,k,v$ and attends over the cache. Memory per token is $2\\cdot n_{\\text{layers}}\\cdot d_{\\text{model}}$ numbers (K and V in every layer), e.g. $2\\cdot12\\cdot768\\cdot2$ bytes = 36 KiB in fp16 for a GPT-2-small shape, and 512 KiB for 32 layers of width 4096. Multiply by context length and batch size." },
      { t: "qa", level: "intermediate", q: "What does label smoothing do to the optimum and to the gradient?", a: "Targets become $q=(1-\\varepsilon)y+\\varepsilon/|\\mathcal V|$; the gradient with respect to the logits is $p-q$, so the optimum predicts $1-\\varepsilon+\\varepsilon/|\\mathcal V|$ on the correct class and logits stay bounded (the gap is $\\ln((1-\\varepsilon+\\varepsilon/V)/(\\varepsilon/V))$). It regularizes over-confidence; the loss is no longer a pure log-likelihood." },
      { t: "qa", level: "intermediate", q: "Why do Transformer recipes use learning-rate warm-up and gradient clipping?", a: "At the start Adam's second-moment estimate is based on few gradients, so per-parameter step sizes can be huge, and attention logits/LayerNorm statistics are unstable (especially in post-LN models). A linear warm-up keeps early steps small. Global-norm clipping bounds the update when a rare batch produces a gradient spike, preventing divergence while preserving the gradient direction." },
      { t: "qa", level: "advanced", q: "Derive the cost of generating N tokens with and without a KV cache.", a: "Without a cache, step $t$ runs the full network on $t$ tokens: $O(t d^2 + t^2 d)$ per layer; summing $t = 1..N$ gives $O(N^2 d^2 + N^3 d)$. With a cache, step $t$ processes one token: $O(d^2)$ for projections and FFN plus $O(t d)$ for attention over the cache; summing gives $O(N d^2 + N^2 d)$. The cache therefore saves a factor of about $N$ in compute, paying $O(N\\,n_{\\text{layers}}\\,d)$ memory." },
      { t: "qa", level: "advanced", q: "Why is decoding memory-bandwidth-bound, and how do batching and GQA help?", a: "A decode step multiplies each weight matrix by one vector per sequence: ~2 FLOPs per weight read, far below what accelerators need to be compute-bound (hundreds of FLOPs per byte). Time is dominated by reading weights and the KV cache. Batching $B$ sequences reuses each weight read for $B$ tokens, raising arithmetic intensity; grouped-query attention shrinks the cache that every step must read by sharing K/V across query heads. The cache itself grows with $B$, so memory capacity bounds the batch." },
      { t: "qa", level: "advanced", q: "Where does attention start to dominate the FLOPs of a Transformer layer?", a: "Linear layers cost $24 n d^2$ FLOPs (projections $8nd^2$, FFN with width $4d$: $16nd^2$); attention scores and weighted sum cost $4n^2 d$. They are equal at $n = 6d$: 4608 tokens for $d = 768$. Larger models (bigger $d$) therefore spend a smaller fraction on attention at a given context length." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { q: "A training window has token ids [5, 9, 2, 7]. Which (inputs, targets) pair is correct for causal LM?", options: ["([5,9,2,7], [5,9,2,7])", "([5,9,2], [9,2,7])", "([9,2,7], [5,9,2])", "([5,9,2,7], [9,2,7,0])"], answer: 1, explain: "Inputs drop the last token, targets drop the first: position $t$ predicts the token at $t+1$." },
        { kind: "numeric", q: "A model assigns probabilities 0.5, 0.25 and 0.8 to three correct next tokens. What is its perplexity on them (2 decimals)?", answer: 2.15, tolerance: 0.01, explain: "Mean cross-entropy $= 0.767528$ nats; $e^{0.767528} = 2.154 = (0.5\\cdot0.25\\cdot0.8)^{-1/3}$." },
        { kind: "numeric", q: "KV-cache size per token in **KiB** for a model with 24 layers, $d_{\\text{model}} = 1024$, fp16 (2 bytes)?", answer: 96, tolerance: 0.01, explain: "$2\\cdot24\\cdot1024\\cdot2 = 98{,}304$ bytes $= 96$ KiB." },
        { kind: "numeric", q: "With label smoothing $\\varepsilon=0.1$ and a vocabulary of 10 tokens, what probability on the correct token minimizes the smoothed loss?", answer: 0.91, tolerance: 0.001, explain: "$1-\\varepsilon+\\varepsilon/|\\mathcal V| = 0.9 + 0.01 = 0.91$." },
        { q: "Why can a decoder-only Transformer be trained on all positions of a sequence in one pass?", options: ["Because the FFN is position-wise", "Because the causal mask makes each position's output depend only on earlier tokens, so it equals the prefix computation", "Because of teacher forcing in the encoder", "Because positional encodings are sinusoidal"], answer: 1, explain: "The causal mask guarantees no leakage from the right; the parallel output at $t$ equals what a separate prefix pass would produce." },
        { q: "During a decode step with a KV cache, what is computed for the new token in each layer?", options: ["Q, K and V for all previous tokens", "Only its own q, k, v; k and v are appended to the cache and q attends over all cached keys", "Only the FFN; attention is skipped", "Only the output logits"], answer: 1, explain: "Past keys and values are unchanged, so only the new token's projections are needed." },
        { kind: "numeric", q: "For $d_{\\text{model}}=1024$ (FFN width $4d$), at what sequence length do attention FLOPs ($4n^2d$) equal the linear-layer FLOPs ($24nd^2$)?", answer: 6144, tolerance: 0.5, explain: "$n = 6d = 6144$." },
        { q: "Which phase of LLM inference is typically memory-bandwidth-bound for a single sequence?", options: ["Prefill", "Decode", "Tokenization", "Both equally"], answer: 1, explain: "Decode multiplies every weight matrix by a single vector, so time is dominated by reading weights and the cache from memory. Prefill uses matrix–matrix products and is compute-bound." },
      ] },
      { t: "exercise", title: "Serving budget for a long context", prompt: "A model has 40 layers, $d_{\\text{model}} = 5120$, and uses fp16 for its KV cache. (a) How many bytes of cache per token? (b) How much memory (GiB) for one sequence of 8192 tokens? (c) If 40 GiB are free for the cache, how many such sequences can be batched? (d) How does the answer to (c) change with grouped-query attention using 8 K/V heads out of 40 heads of dimension 128?", hint: "Bytes per token $= 2\\cdot n_{\\text{layers}}\\cdot d\\cdot 2$. With GQA, replace $d$ by $n_{kv}\\cdot d_{\\text{head}}$.", solution: [
        { t: "steps", items: [
          { title: "(a) Per token", tex: String.raw`2\cdot40\cdot5120\cdot2 = 819{,}200 \text{ B} = 800 \text{ KiB}` },
          { title: "(b) One 8192-token sequence", tex: String.raw`819{,}200 \cdot 8192 = 6{,}710{,}886{,}400 \text{ B} = 6.25 \text{ GiB}` },
          { title: "(c) Batch that fits in 40 GiB", tex: String.raw`\lfloor 40 / 6.25 \rfloor = 6 \text{ sequences}` },
          { title: "(d) With 8 K/V heads", tex: String.raw`n_{kv}d_{\text{head}} = 8\cdot128 = 1024 = 5120/5 \Rightarrow 1.25\text{ GiB per sequence} \Rightarrow 32 \text{ sequences}` },
        ] },
      ] },
      { t: "exercise", title: "Add top-k / top-p sampling and early stopping to the training script", prompt: "Modify `train_tiny_gpt.py`: (1) keep a copy of the model state with the lowest validation loss and restore it at the end (early stopping); (2) extend `sample()` with `top_k` and `top_p` arguments. Which checkpoint is restored, and what is its validation perplexity?", hint: "Use `copy.deepcopy(model.state_dict())` when the validation loss improves. For top-p, sort the probabilities, take the cumulative sum, and zero everything after the first index where it exceeds p (keep at least one token).", solution: [
        { t: "code", lang: "text", code: String.raw`# (1) early stopping
best = (float("inf"), None)
...
        vl = val_loss()
        if vl < best[0]:
            best = (vl, copy.deepcopy(model.state_dict()))
...
model.load_state_dict(best[1])

# (2) filtering inside sample(), before torch.multinomial
logits = logits / temperature
if top_k:
    kth = logits.topk(top_k).values[-1]
    logits[logits < kth] = -float("inf")
if top_p < 1.0:
    probs, order = F.softmax(logits, -1).sort(descending=True)
    drop = probs.cumsum(-1) - probs > top_p      # tokens after the nucleus is complete
    logits[order[drop]] = -float("inf")
nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)` },
        { t: "p", text: "With evaluation every 100 steps, the restored checkpoint is step 200 with validation loss 2.376, i.e. perplexity $e^{2.376} = 10.76$ — the minimum of the printed curve. Its samples are less faithful copies of the training text but generalize better. Note that `drop` is computed with `cumsum - probs` so the token that crosses the threshold is kept, guaranteeing a non-empty nucleus." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Causal LM: inputs $x_{0:T-1}$, targets $x_{1:T}$; loss $\\mathcal L = -\\frac1T\\sum_t\\log p_\\theta(x_{t+1}\\mid x_{\\le t})$, perplexity $e^{\\mathcal L}$ (inverse geometric mean of the true-token probabilities).",
        "The causal mask makes position $t$ depend only on $x_{\\le t}$, so one parallel pass equals $T$ prefix passes (verified: difference $3\\times10^{-15}$; without the mask, 3.28).",
        "Masked LM predicts hidden tokens from both sides; seq2seq decoders use teacher forcing: input `[BOS]`+$y_{1:m-1}$, target $y_{1:m}$+`[EOS]`.",
        "Label smoothing: $q=(1-\\varepsilon)y+\\varepsilon/|\\mathcal V|$, gradient $p-q$, optimum $1-\\varepsilon+\\varepsilon/|\\mathcal V|$; do not report its loss as perplexity.",
        "Recipe: AdamW (decoupled decay on matrices, $\\beta_2\\approx0.95$), linear warm-up + cosine decay, global-norm clipping at 1.0, bf16/fp16 mixed precision with fp32 master weights.",
        "Training FLOPs per layer $\\approx 24nd^2 + 4n^2d$; attention dominates beyond $n = 6d$.",
        "Generation is sequential. KV cache: per-step cost $O(d^2 + t d)$ per layer instead of $O(t d^2 + t^2 d)$; memory $2\\cdot n_{\\text{layers}}\\cdot d_{\\text{model}}\\cdot b$ bytes per token. Outputs are identical with and without it.",
        "Prefill is compute-bound (matrix–matrix), decode is memory-bound (matrix–vector); batching raises decode throughput until KV-cache memory runs out.",
        "With tiny data a Transformer memorizes: watch validation loss, not training loss.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families) — how the three objectives here define the three Transformer families, and how the same machinery applies to images.",
        "[Decoding strategies](/learn/m20/decoding) — greedy, beam, temperature, top-k and top-p in depth: how the distribution computed here becomes text.",
        "[Efficient Inference: KV Cache, FlashAttention, MoE](/learn/m20/llm-inference) — the systems side of the generation loop: paged caches, fused attention kernels, speculative decoding, quantization.",
        "[Language Modelling and Perplexity](/learn/m13/language-modeling) — perplexity in context: n-gram baselines, evaluation pitfalls, BLEU and ROUGE.",
        "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — the same recipe scaled to trillions of tokens: data mixtures and compute-optimal training.",
      ] },
    ],
  },
};

export default lesson;
