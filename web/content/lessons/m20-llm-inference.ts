import type { Lesson } from "@/lib/types";

/* ------------------------------------------------------------------ code blocks (executed; outputs are real) */

const KV = String.raw`import numpy as np

rng = np.random.default_rng(0)
V, d, n_layers, n_heads = 40, 32, 2, 4           # tiny decoder: vocab 40, width 32, 2 layers, 4 heads
d_head = d // n_heads
macs = {"n": 0}                                   # multiply-accumulate counter for the projections + attention


def mm(a, b):
    macs["n"] += a.shape[0] * a.shape[1] * b.shape[1]
    return a @ b


def rms_norm(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


E = rng.normal(0, 0.5, (V, d))                    # token embeddings (also the tied output head)
P = rng.normal(0, 0.5, (64, d))                   # learned absolute positions
layers = [{k: rng.normal(0, d ** -0.5, (d, d)) for k in ("Wq", "Wk", "Wv", "Wo")} | {
    "W1": rng.normal(0, d ** -0.5, (d, 4 * d)), "W2": rng.normal(0, (4 * d) ** -0.5, (4 * d, d))} for _ in range(n_layers)]


def block(x, L, pos0, cache=None):
    """x: (t, d) new positions starting at absolute position pos0. cache: dict with K, V of earlier positions."""
    h = rms_norm(x)
    q, k, v = mm(h, L["Wq"]), mm(h, L["Wk"]), mm(h, L["Wv"])
    if cache is not None:                         # append the new keys/values to the cache
        cache["K"] = np.concatenate([cache["K"], k]) if "K" in cache else k
        cache["V"] = np.concatenate([cache["V"], v]) if "V" in cache else v
        k, v = cache["K"], cache["V"]
    t, n = x.shape[0], k.shape[0]
    out = np.zeros_like(q)
    for hd in range(n_heads):
        s = slice(hd * d_head, (hd + 1) * d_head)
        scores = mm(q[:, s], k[:, s].T) / np.sqrt(d_head)              # (t, n)
        qpos = pos0 + np.arange(t)[:, None]
        scores = np.where(np.arange(n)[None, :] > qpos, -np.inf, scores)  # causal mask by absolute position
        out[:, s] = mm(softmax(scores), v[:, s])
    x = x + mm(out, L["Wo"])
    return x + mm(np.maximum(0, mm(rms_norm(x), L["W1"])), L["W2"])


def logits_full(ids):
    x = E[ids] + P[: len(ids)]
    for L in layers:
        x = block(x, L, 0)
    return mm(rms_norm(x), E.T)


def pick(z, u):
    # sample from softmax(z) with a pre-drawn uniform u, so both loops make the same random choices
    return int(np.searchsorted(np.cumsum(softmax(z)), u, side="right"))


def generate_uncached(prompt, us):
    ids, zs = list(prompt), []
    for u in us:
        zs.append(logits_full(ids)[-1])            # re-runs the whole prefix every step
        ids.append(pick(zs[-1], u))
    return ids, zs


def generate_cached(prompt, us):
    caches = [dict() for _ in layers]
    ids = list(prompt)
    x = E[ids] + P[: len(ids)]                    # prefill: whole prompt in one pass
    for L, c in zip(layers, caches):
        x = block(x, L, 0, c)
    zs = [mm(rms_norm(x[-1:]), E.T)[0]]
    for u in us[:-1]:
        ids.append(pick(zs[-1], u))
        pos = len(ids) - 1
        x = E[[ids[-1]]] + P[[pos]]               # decode: ONLY the new token, at its absolute position
        for L, c in zip(layers, caches):
            x = block(x, L, pos, c)
        zs.append(mm(rms_norm(x), E.T)[0])
    ids.append(pick(zs[-1], us[-1]))
    return ids, zs, caches


prompt, n_new = [3, 14, 15, 9, 26, 5, 35, 8], 24
us = rng.random(n_new)                            # the same random numbers for both loops
macs["n"] = 0
a, za = generate_uncached(prompt, us)
macs_uncached = macs["n"]
macs["n"] = 0
b, zb, caches = generate_cached(prompt, us)
macs_cached = macs["n"]
print("identical tokens:", a == b, "| max |logit difference| over all steps:", f"{np.abs(np.array(za) - np.array(zb)).max():.1e}")
print("generated:", a[len(prompt):])
print(f"MACs uncached {macs_uncached:,}  cached {macs_cached:,}  ratio {macs_uncached / macs_cached:.2f}x")
n_cached = sum(c["K"].size + c["V"].size for c in caches)
print(f"cache holds {n_cached:,} numbers = 2 * layers * heads * d_head * seq = 2*{n_layers}*{n_heads}*{d_head}*{caches[0]['K'].shape[0]}"
      f" = {2 * n_layers * n_heads * d_head * caches[0]['K'].shape[0]:,}")


# ---- KV-cache memory for a large model: MHA vs GQA vs MQA ----
def kv_bytes(layers, kv_heads, d_head, seq, batch=1, bytes_per=2):
    return 2 * layers * kv_heads * d_head * seq * batch * bytes_per


cfg = dict(layers=32, d_head=128, seq=4096)   # a 7B-class shape: 32 layers, 32 query heads of 128
for name, kvh in (("MHA (32 KV heads)", 32), ("GQA (8 KV heads)", 8), ("MQA (1 KV head)", 1)):
    b1 = kv_bytes(kv_heads=kvh, **cfg)
    print(f"{name:18s}: {b1 / 2**30:6.3f} GiB per sequence, {b1 / cfg['seq'] / 2**10:7.1f} KiB per token,"
          f" batch 16 -> {kv_bytes(kv_heads=kvh, batch=16, **cfg) / 2**30:6.2f} GiB")`;

const KV_OUT = String.raw`identical tokens: True | max |logit difference| over all steps: 4.6e-15
generated: [8, 38, 34, 34, 34, 34, 34, 34, 5, 5, 5, 5, 25, 25, 22, 10, 38, 38, 32, 32, 32, 32, 32, 32]
MACs uncached 13,415,936  cached 859,648  ratio 15.61x
cache holds 3,968 numbers = 2 * layers * heads * d_head * seq = 2*2*4*8*31 = 3,968
MHA (32 KV heads) :  2.000 GiB per sequence,   512.0 KiB per token, batch 16 ->  32.00 GiB
GQA (8 KV heads)  :  0.500 GiB per sequence,   128.0 KiB per token, batch 16 ->   8.00 GiB
MQA (1 KV head)   :  0.062 GiB per sequence,    16.0 KiB per token, batch 16 ->   1.00 GiB`;

const FLASH = String.raw`import numpy as np

rng = np.random.default_rng(0)

# ---- 1. online softmax: one pass, running max m and running sum l ----
x = rng.normal(0, 3, 10)
m, l = -np.inf, 0.0
for xi in x:
    m_new = max(m, xi)
    l = l * np.exp(m - m_new) + np.exp(xi - m_new)   # rescale the old sum to the new max
    m = m_new
online = np.exp(x - m) / l
exact = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
print(f"online softmax: max |diff| = {np.abs(online - exact).max():.1e}, running max {m:.4f} = true max {x.max():.4f}")


# ---- 2. tiled attention (the FlashAttention forward pass, one head) ----
def attention_exact(Q, K, V, causal=False):
    S = Q @ K.T / np.sqrt(Q.shape[1])                # materialises the full n x n score matrix
    if causal:
        S = np.where(np.triu(np.ones_like(S, bool), 1), -np.inf, S)
    A = np.exp(S - S.max(1, keepdims=True))
    return (A / A.sum(1, keepdims=True)) @ V


def attention_tiled(Q, K, V, Br=16, Bc=16, causal=False):
    """Never forms more than a Br x Bc block of scores. Keeps, per query row: running max m,
    running denominator l, and an unnormalised output accumulator O."""
    n, dk = Q.shape
    O = np.zeros_like(Q)
    peak = 0
    for i in range(0, n, Br):                        # outer loop: a block of queries
        q = Q[i:i + Br]
        m = np.full(len(q), -np.inf)
        l = np.zeros(len(q))
        acc = np.zeros((len(q), V.shape[1]))
        for j in range(0, n, Bc):                    # inner loop: stream key/value blocks
            if causal and j > i + len(q) - 1:
                break                                # block entirely in the future: skip it
            s = q @ K[j:j + Bc].T / np.sqrt(dk)      # (Br, Bc) block of scores
            if causal:
                s = np.where(np.arange(j, j + s.shape[1])[None, :] > np.arange(i, i + len(q))[:, None], -np.inf, s)
            peak = max(peak, s.size)
            m_new = np.maximum(m, s.max(1))
            p = np.exp(s - m_new[:, None])
            scale = np.exp(m - m_new)                # correction for everything accumulated so far
            l = l * scale + p.sum(1)
            acc = acc * scale[:, None] + p @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = acc / l[:, None]               # normalise once, at the end
    return O, peak


n, dk = 200, 32
Q, K, V = (rng.normal(size=(n, dk)) for _ in range(3))
for causal in (False, True):
    ref = attention_exact(Q, K, V, causal)
    out, peak = attention_tiled(Q, K, V, causal=causal)
    print(f"causal={causal!s:5}: max |tiled - exact| = {np.abs(out - ref).max():.1e}; "
          f"largest score block {peak} numbers vs full matrix {n * n}")

# HBM traffic model for one head (numbers moved, ignoring constants): standard vs tiled
d, Br = 64, 128
for n in (1024, 8192):
    standard = 4 * n * d + 2 * n * n * 2               # read Q,K,V, write O + write and re-read S and P
    tiled = 2 * n * d + (n // Br) * 2 * n * d          # read Q, write O once; stream K,V once per query block
    print(f"n={n}: standard ~{standard / 1e6:.1f}M numbers, tiled ~{tiled / 1e6:.1f}M numbers")`;

const FLASH_OUT = String.raw`online softmax: max |diff| = 0.0e+00, running max 3.9120 = true max 3.9120
causal=False: max |tiled - exact| = 5.0e-16; largest score block 256 numbers vs full matrix 40000
causal=True : max |tiled - exact| = 5.6e-16; largest score block 256 numbers vs full matrix 40000
n=1024: standard ~4.5M numbers, tiled ~1.2M numbers
n=8192: standard ~270.5M numbers, tiled ~68.2M numbers`;

const MOE = String.raw`import numpy as np

rng = np.random.default_rng(0)

# ---- 1. mixture of experts: top-2 routing + load-balancing loss ----
n_tok, d, n_exp, k = 512, 16, 8, 2
x = rng.normal(size=(n_tok, d))
Wg = rng.normal(0, 0.5, (d, n_exp))                      # router (gating) weights
experts = [(rng.normal(0, d ** -0.5, (d, 4 * d)), rng.normal(0, (4 * d) ** -0.5, (4 * d, d))) for _ in range(n_exp)]


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def moe(x, Wg):
    probs = softmax(x @ Wg)                              # (n_tok, n_exp) router probabilities
    top = np.argsort(-probs, axis=1, kind="stable")[:, :k]  # each token's k chosen experts
    gate = np.take_along_axis(probs, top, 1)
    gate = gate / gate.sum(1, keepdims=True)             # renormalise the k gate values
    y = np.zeros_like(x)
    for e, (W1, W2) in enumerate(experts):
        rows, slot = np.nonzero(top == e)                # tokens routed to expert e
        if len(rows):
            y[rows] += gate[rows, slot, None] * (np.maximum(0, x[rows] @ W1) @ W2)
    f = np.bincount(top[:, 0], minlength=n_exp) / len(x)  # fraction of tokens whose FIRST choice is e
    P = probs.mean(0)                                    # mean router probability of e
    aux = n_exp * (f * P).sum()                          # Switch-Transformer balancing loss (1 = perfectly uniform)
    return y, f, P, aux, top


y, f, P, aux, top = moe(x, Wg)
print("tokens per expert (top-1):", np.bincount(top[:, 0], minlength=n_exp))
print(f"balance loss N*sum(f*P) = {aux:.4f} (uniform routing gives 1.0)")
# gradient descent on the router using ONLY the differentiable part of the loss (f treated as constant)
for step in range(300):
    probs = softmax(x @ Wg)
    f = np.bincount(np.argmax(probs, 1), minlength=n_exp) / n_tok
    g_probs = n_exp * f / n_tok                          # d aux / d probs[t, e] = N * f_e / n_tok
    g_logits = probs * (g_probs - (probs * g_probs).sum(1, keepdims=True))   # back through the softmax
    Wg -= 2.0 * x.T @ g_logits
y, f, P, aux, top = moe(x, Wg)
print("after balancing         :", np.bincount(top[:, 0], minlength=n_exp), f"loss {aux:.4f}")
dense_flops = 2 * d * 4 * d * 2 * n_exp
print(f"active expert FLOPs per token: {k}/{n_exp} experts = {2 * d * 4 * d * 2 * k:,} vs {dense_flops:,} if every expert ran")

# ---- 2. speculative decoding: the accept/reject rule preserves the target distribution ----
p = np.array([0.50, 0.25, 0.15, 0.07, 0.03])             # target model's next-token distribution
q = np.array([0.30, 0.40, 0.10, 0.15, 0.05])             # cheaper draft model's distribution
accept = np.minimum(1, p / q)                            # accept draft token x with prob min(1, p(x)/q(x))
residual = np.maximum(p - q, 0) / np.maximum(p - q, 0).sum()   # resample from norm(max(0, p - q)) on reject
alpha = np.minimum(p, q).sum()                           # overall acceptance rate = sum_x min(p, q)
exact = q * accept + (1 - alpha) * residual              # probability that the step outputs token x
print("exact output distribution:", exact.round(6), "== p:", np.allclose(exact, p), f"| acceptance rate {alpha:.2f}")

N = 200_000
draft = rng.choice(5, size=N, p=q)
ok = rng.random(N) < accept[draft]
out = np.where(ok, draft, rng.choice(5, size=N, p=residual))
print("Monte Carlo frequencies  :", (np.bincount(out, minlength=5) / N).round(4))
gamma = 4                                                # draft tokens proposed per target call
expected = (1 - alpha ** (gamma + 1)) / (1 - alpha)      # tokens produced per target forward pass
print(f"gamma={gamma}: expected tokens per target call = {expected:.3f}")

# ---- 3. weight-only quantization: per-output-channel int8 and group-wise int4 ----
W = rng.normal(0, 0.02, (256, 256))
W[3, 7] = 0.4                                            # one outlier weight, as real LLMs have
xin = rng.normal(size=(8, 256))


def quant(W, bits, group=None):
    qmax = 2 ** (bits - 1) - 1                           # 127 for int8, 7 for int4 (symmetric)
    Wg = W.reshape(W.shape[0], -1, group or W.shape[1])  # (out, n_groups, group)
    scale = np.abs(Wg).max(-1, keepdims=True) / qmax     # one scale per row (or per group of a row)
    q = np.clip(np.round(Wg / scale), -qmax, qmax)       # the stored integers
    return (q * scale).reshape(W.shape), scale.size


for name, bits, group in (("int8 per-row", 8, None), ("int4 per-row", 4, None), ("int4 group 64", 4, 64)):
    Wq, n_scales = quant(W, bits, group)
    err = np.abs(xin @ Wq.T - xin @ W.T).mean() / np.abs(xin @ W.T).mean()
    bits_per_w = bits + 16 * n_scales / W.size           # fp16 scales add overhead
    print(f"{name:14s}: {bits_per_w:5.3f} bits/weight, mean relative output error {err:.4f}")`;

const MOE_OUT = String.raw`tokens per expert (top-1): [34 56 52 62 65 87 63 93]
balance loss N*sum(f*P) = 1.0404 (uniform routing gives 1.0)
after balancing         : [64 64 64 64 64 64 64 64] loss 1.0000
active expert FLOPs per token: 2/8 experts = 8,192 vs 32,768 if every expert ran
exact output distribution: [0.5  0.25 0.15 0.07 0.03] == p: True | acceptance rate 0.75
Monte Carlo frequencies  : [0.5008 0.2494 0.1508 0.069  0.03  ]
gamma=4: expected tokens per target call = 3.051
int8 per-row  : 8.062 bits/weight, mean relative output error 0.0069
int4 per-row  : 4.062 bits/weight, mean relative output error 0.1267
int4 group 64 : 4.250 bits/weight, mean relative output error 0.1102`;

const TORCH = String.raw`import copy

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging

logging.set_verbosity_error()
torch.manual_seed(0)
torch.set_num_threads(1)

# ---- 1. KV cache in a real Transformer implementation (tiny random GPT-2, nothing downloaded) ----
cfg = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=4, n_head=4, initializer_range=0.1)
target = GPT2LMHeadModel(cfg).eval()
prompt = torch.randint(0, 100, (1, 12))


@torch.no_grad()
def greedy_no_cache(model, ids, n):
    for _ in range(n):
        ids = torch.cat([ids, model(ids, use_cache=False).logits[:, -1:].argmax(-1)], 1)
    return ids


@torch.no_grad()
def greedy_with_cache(model, ids, n):
    out = model(ids, use_cache=True)                    # prefill
    past, seq = out.past_key_values, ids
    for _ in range(n):
        nxt = out.logits[:, -1:].argmax(-1)
        seq = torch.cat([seq, nxt], 1)
        out = model(nxt, past_key_values=past, use_cache=True)   # decode: one token in
        past = out.past_key_values
    return seq, past


a = greedy_no_cache(target, prompt, 20)
b, past = greedy_with_cache(target, prompt, 20)
print("cached == uncached:", torch.equal(a, b))
k0 = past.layers[0].keys if hasattr(past, "layers") else past[0][0]
print("layer-0 key cache shape (batch, heads, seq, d_head):", tuple(k0.shape))
n_cache = 2 * cfg.n_layer * k0.numel()
print(f"cache numbers {n_cache:,} = 2*{cfg.n_layer}*{cfg.n_head}*{cfg.n_embd // cfg.n_head}*{k0.shape[2]}")

# ---- 2. grouped-query attention with PyTorch's fused attention ----
B, Hq, Hkv, T, dh = 2, 8, 2, 16, 32
q = torch.randn(B, Hq, T, dh)
k, v = torch.randn(B, Hkv, T, dh), torch.randn(B, Hkv, T, dh)
fused = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
k_rep, v_rep = k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1)  # each KV head serves 4 query heads
s = q @ k_rep.transpose(-1, -2) / dh ** 0.5
s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
naive = s.softmax(-1) @ v_rep
print(f"GQA fused vs naive: max |diff| = {(fused - naive).abs().max().item():.1e}; KV cache is {Hkv}/{Hq} of MHA's")

# ---- 3. greedy speculative decoding: a small draft proposes, the target verifies in ONE pass ----
draft = copy.deepcopy(target)                           # draft = the target's first 2 of 4 layers ("layer skipping"):
draft.transformer.h = draft.transformer.h[:2]           # half the cost, similar but not identical predictions
draft.config.n_layer = 2
calls = {"target": 0, "accepted": 0}


@torch.no_grad()
def speculative_greedy(ids, n, gamma=4):
    goal = ids.shape[1] + n
    while ids.shape[1] < goal:
        prop = greedy_no_cache(draft, ids, gamma)[:, ids.shape[1]:]          # gamma cheap draft tokens
        logits = target(torch.cat([ids, prop], 1)).logits                    # one target pass scores all of them
        calls["target"] += 1
        tgt = logits[:, ids.shape[1] - 1:].argmax(-1)                        # target's own choice at each slot
        n_ok = int((prop[0] == tgt[0, :gamma]).long().cumprod(0).sum())      # longest agreeing prefix
        calls["accepted"] += n_ok
        ids = torch.cat([ids, prop[:, :n_ok], tgt[:, n_ok:n_ok + 1]], 1)     # + one target token for free
    return ids[:, :goal]


spec = speculative_greedy(prompt, 20)
print("target greedy tokens:", a[0, 12:].tolist())
print("speculative output == target greedy output:", torch.equal(spec, a), f"| target calls {calls['target']} instead of 20, draft tokens accepted {calls['accepted']}")

# ---- 4. weight-only int8: store int8 + one scale per output row, compute in float ----
lin = target.transformer.h[0].mlp.c_fc                   # GPT-2 Conv1D: weight is (in, out)
W = lin.weight.detach()
scale = W.abs().amax(0, keepdim=True) / 127               # per output column
W_int8 = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
x = torch.randn(4, 64)
y_fp, y_q = x @ W + lin.bias, x @ (W_int8.float() * scale) + lin.bias
print(f"int8 weights: {W_int8.numel() + 2 * scale.numel():,} bytes vs fp32 {4 * W.numel():,};"
      f" relative error {((y_q - y_fp).norm() / y_fp.norm()).item():.4f}")`;

const TORCH_OUT = String.raw`cached == uncached: True
layer-0 key cache shape (batch, heads, seq, d_head): (1, 4, 32, 16)
cache numbers 16,384 = 2*4*4*16*32
GQA fused vs naive: max |diff| = 4.8e-07; KV cache is 2/8 of MHA's
target greedy tokens: [36, 47, 41, 41, 41, 23, 23, 23, 23, 23, 23, 23, 23, 23, 36, 36, 36, 36, 36, 47]
speculative output == target greedy output: True | target calls 8 instead of 20, draft tokens accepted 12
int8 weights: 16,896 bytes vs fp32 65,536; relative error 0.0059`;

const KERAS = String.raw`import tensorflow as tf
from tensorflow.keras import layers

# Grouped-query attention in Keras: 8 query heads share 2 key/value heads.
B, T, d_model, Hq, Hkv, dh = 2, 16, 256, 8, 2, 32
x = tf.random.normal((B, T, d_model))
q = tf.reshape(layers.Dense(Hq * dh)(x), (B, T, Hq, dh))
k = tf.reshape(layers.Dense(Hkv * dh)(x), (B, T, Hkv, dh))      # 4x fewer K/V numbers to cache
v = tf.reshape(layers.Dense(Hkv * dh)(x), (B, T, Hkv, dh))
k = tf.repeat(k, Hq // Hkv, axis=2)
v = tf.repeat(v, Hq // Hkv, axis=2)
scores = tf.einsum("bqhd,bkhd->bhqk", q, k) / dh ** 0.5
mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)               # lower triangle = allowed
scores += (1.0 - mask) * -1e9
out = tf.einsum("bhqk,bkhd->bqhd", tf.nn.softmax(scores, -1), v)
print(out.shape)`;

/* ------------------------------------------------------------------ lesson */

const lesson: Lesson = {
  id: "m20/llm-inference",
  title: "Efficient Inference: KV Cache, FlashAttention, MoE",
  summary:
    "Why LLM generation is limited by memory speed and how serving systems make it fast: prefill vs decode, the KV cache and its memory formula, multi-query and grouped-query attention, FlashAttention's tiling with an online softmax, mixture-of-experts routing with a load-balancing loss, speculative decoding (which provably keeps the output distribution) and weight-only quantization — each implemented and verified.",
  minutes: 75,
  level: "research",
  prerequisites: ["m11/training-inference", "m10/multi-head-attention", "m20/decoding", "m18/model-optimization"],
  next: ["m19/scaling-efficiency", "m18/serving", "m20/alignment"],
  terms: ["kv-cache", "prefill", "decode-phase", "memory-bound", "multi-query-attention", "grouped-query-attention", "flashattention", "online-softmax", "mixture-of-experts", "load-balancing-loss", "speculative-decoding", "weight-only-quantization", "paged-attention"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "Training is a throughput problem: huge batches of tokens flow through dense matrix multiplications and the GPUs stay busy. **Serving** is different. Generation is sequential — token $t+1$ needs token $t$ first — so each step handles one new token per sequence, yet must still read **every weight** and the **whole attention history** from GPU memory, doing very little arithmetic per byte." },
      { t: "p", text: "So generation is **memory-bandwidth bound**: the bottleneck is how fast bytes arrive, not how fast the GPU computes. Almost every inference trick reads fewer bytes, reads them fewer times, or does more useful work per read." },
      { t: "list", items: [
        "**KV cache** — never recompute the past (introduced in [Training, Autoregressive Generation and Complexity](/learn/m11/training-inference)).",
        "**Multi-query / grouped-query attention** — shrink the cache.",
        "**FlashAttention** — exact attention without ever writing the $n\\times n$ score matrix to memory.",
        "**Mixture of experts** — many parameters, few used per token.",
        "**Speculative decoding** — several tokens per expensive forward pass, with exactly the same output distribution.",
        "**Weight-only quantization** — fewer bytes per weight (basics in [Quantization, Pruning and Distillation](/learn/m18/model-optimization)).",
      ] },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — the generation loop, prefill, the basic KV cache and the $O(n^2 d)$ attention cost.",
        "[Multi-Head Attention](/learn/m10/multi-head-attention) — heads, $d_{\\text{head}}$, and the $Q, K, V$ projections the cache stores.",
        "[Decoding](/learn/m20/decoding) — sampling from $p(\\cdot\\mid\\text{context})$; speculative decoding must reproduce that distribution.",
        "[Quantization, Pruning and Distillation](/learn/m18/model-optimization) — integer formats, scales and the accuracy trade-off.",
        "Softmax and its max-subtraction trick (for numerical stability) from [Optimization Math](/learn/m02/optimization-math).",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "A chef, a pantry and a notebook", text: "A chef (the GPU's compute units) is extremely fast, but every ingredient comes from a pantry down the hall (GPU memory), and writing one word of a story means fetching the whole recipe book (all weights). **Batching**: words for 64 stories per trip. **KV cache**: a notebook of what each earlier word contributed, so the story is never reread. **GQA**: heads share notes, so the notebook is smaller. **FlashAttention**: cook at the counter in small batches instead of spreading half-made dishes across the pantry floor. **Mixture of experts**: eight specialist chefs, each dish visits two. **Speculative decoding**: an apprentice drafts a few words; the master checks them in one glance and keeps the part he agrees with. **Quantization**: a smaller recipe book, so trips are faster." },
      { t: "p", text: "The most useful number is **arithmetic intensity**: FLOPs per byte moved. A matrix-vector product (one token, decode) does about 2 FLOPs per weight read (1 FLOP per byte in fp16), while a modern accelerator can do hundreds of FLOPs in the time it reads one byte. Prefill (many prompt tokens at once) is a high-intensity matrix-matrix product, so it is **compute-bound**; decode is **memory-bound**. They need different optimisations." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "Prefill and decode" },
      { t: "steps", items: [
        { title: "Prefill", text: "The whole prompt ($n_p$ tokens) goes through the model in one causally masked pass, just like training, and every layer's keys and values are written to the cache. Cost: $\\approx 2N n_p$ FLOPs for $N$ parameters — compute-bound. It sets the **time to first token**." },
        { title: "Decode", text: "One new token per sequence per step: compute its $q, k, v$ in each layer, append $k, v$ to the cache, attend over the cache, produce logits, choose a token (see [Decoding](/learn/m20/decoding)). Cost per token: $\\approx 2N$ FLOPs plus reading all weights and the cache — memory-bound. It sets the **time per output token**." },
        { title: "Batching", text: "Decode is memory-bound, so running many sequences together is nearly free until compute becomes the limit: weights are read once per step for the whole batch. **Continuous batching** adds and removes requests at every step instead of waiting for a whole batch to finish." },
        { title: "Paged KV cache", text: "Caches grow one token per step and sequences finish at different times, so reserving full-length buffers wastes memory. PagedAttention (Kwon et al., 2023, the vLLM system) stores the cache in fixed-size blocks with a page table, like virtual memory, and lets sequences share prefix blocks." },
      ] },
      { t: "h", text: "FlashAttention: tiling plus an online softmax" },
      { t: "steps", items: [
        { title: "The problem", text: "Standard attention writes $S = QK^\\top/\\sqrt d$ ($n\\times n$) to the GPU's large but slow high-bandwidth memory (HBM), reads it back for the softmax, writes $P$, then reads it again to multiply by $V$. For long sequences this traffic, not the FLOPs, dominates run time." },
        { title: "Tiling", text: "Load a block of queries into small, fast on-chip SRAM and stream blocks of keys and values past it. Each block of scores lives only in SRAM." },
        { title: "Online softmax", text: "The softmax needs each row's maximum and sum over **all** keys, which aren't known until the last block. So keep a running max $m$, running sum $\\ell$ and an unnormalised output. When a new block raises the max, rescale what you have by $e^{m_{\\text{old}}-m_{\\text{new}}}$ (Milakov and Gimelshein, 2018, *Online normalizer calculation for softmax*)." },
        { title: "Finish", text: "Divide by $\\ell$ once at the end. The result is **exact** attention, not an approximation (Dao et al., 2022, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*). The backward pass recomputes the blocks instead of storing $P$." },
      ] },
      { t: "h", text: "Mixture of experts" },
      { t: "steps", items: [
        { title: "Router", text: "Replace the feed-forward block with $E$ expert FFNs and a router $W_g$. For token $x$, router probabilities are $\\mathrm{softmax}(xW_g)$." },
        { title: "Top-k dispatch", text: "Send the token to its $k$ most likely experts (Shazeer et al., 2017 used sparse top-k gating; Switch Transformers, Fedus et al., 2022, use $k=1$; Mixtral, Jiang et al., 2024, uses 8 experts with $k=2$)." },
        { title: "Combine", text: "Output $= \\sum_{e \\in \\text{top-}k} g_e(x)\\,\\mathrm{FFN}_e(x)$, with the gate values renormalised over the chosen experts." },
        { title: "Balance", text: "Left alone, the router collapses onto a few experts. An extra **load-balancing loss** pushes routing toward uniform, and an expert **capacity** limit drops or reroutes overflow tokens." },
      ] },
      { t: "h", text: "Speculative decoding" },
      { t: "steps", items: [
        { title: "Draft", text: "A small, cheap draft model proposes $\\gamma$ tokens, one after another." },
        { title: "Verify", text: "The big target model scores the prompt plus all $\\gamma$ drafts in **one** forward pass (prefill-like, so compute-efficient), giving its distribution at every draft position." },
        { title: "Accept/reject", text: "Walk through the drafts, accepting draft token $x$ with probability $\\min(1, p(x)/q(x))$. At the first rejection, sample a replacement from the normalised leftover $\\max(0, p - q)$ and stop. If all are accepted, sample one bonus token from the target." },
        { title: "Result", text: "Between 1 and $\\gamma+1$ tokens per target forward pass, and the output follows **exactly** the target model's distribution (Leviathan et al., 2023; Chen et al., 2023)." },
      ] },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "KV-cache memory" },
      { t: "math", tex: String.raw`M_{\text{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot n \cdot b \cdot B`, caption: "2 = keys and values; $L$ = layers; $H_{kv}$ = key/value heads ($=H$ for standard multi-head attention); $d_{\\text{head}}$ = head dimension; $n$ = cached tokens; $b$ = bytes per number (2 for fp16/bf16); $B$ = batch size." },
      { t: "p", text: String.raw`Worked example, a 7B-class shape with $L=32$, $H=32$, $d_{\text{head}}=128$, $n=4096$, fp16: $2\cdot32\cdot32\cdot128\cdot4096\cdot2 = 2{,}147{,}483{,}648$ bytes $= 2$ GiB **per sequence** (512 KiB per token). At batch 16 the cache alone needs 32 GiB — more than the 14 GB of fp16 weights.` },
      { t: "p", text: String.raw`**Grouped-query attention** (Ainslie et al., 2023) with $H_{kv}=8$ cuts this to 0.5 GiB; **multi-query attention** (Shazeer, 2019, *Fast Transformer Decoding: One Write-Head is All You Need*) with $H_{kv}=1$, to 64 MiB. The query heads stay the same; each group of $H/H_{kv}$ query heads shares one key/value head.` },
      { t: "h", text: "Compute with and without the cache" },
      { t: "p", text: String.raw`Without a cache, step $t$ reprocesses all $t$ tokens, so generating $n$ tokens costs $\sum_{t=1}^{n} O(t\,d^2 + t^2 d) = O(n^2 d^2 + n^3 d)$ per layer. With a cache, step $t$ handles one token: $O(d^2)$ projection work and $O(t\,d)$ attention work, $O(n d^2 + n^2 d)$ in total. In the NumPy demo (8-token prompt, 24 new tokens), the uncached loop does 13,415,936 multiply-accumulates and the cached loop 859,648 — 15.6× fewer — with identical logits.` },
      { t: "h", text: "Online softmax" },
      { t: "math", tex: String.raw`m_j = \max(m_{j-1}, x_j), \qquad \ell_j = \ell_{j-1}\,e^{m_{j-1}-m_j} + e^{x_j - m_j}, \qquad \mathrm{softmax}(x)_i = \frac{e^{x_i - m_n}}{\ell_n}`, caption: "$m_j$ = running maximum after $j$ elements, $\\ell_j$ = running sum of exponentials measured from $m_j$; $m_0=-\\infty$, $\\ell_0=0$." },
      { t: "steps", items: [
        { title: "x = (1, 3, 2): first element", tex: String.raw`m_1 = 1,\quad \ell_1 = 0\cdot e^{-\infty} + e^{0} = 1` },
        { title: "Second element raises the max", tex: String.raw`m_2 = 3,\quad \ell_2 = 1\cdot e^{1-3} + e^{0} = 0.135335 + 1 = 1.135335` },
        { title: "Third element", tex: String.raw`m_3 = 3,\quad \ell_3 = 1.135335 + e^{2-3} = 1.135335 + 0.367879 = 1.503215` },
        { title: "Result", tex: String.raw`\mathrm{softmax} = (e^{-2}, e^{0}, e^{-1})/1.503215 = (0.090031,\ 0.665241,\ 0.244728)` },
      ] },
      { t: "p", text: String.raw`For attention, the output accumulator gets the same correction: with $O_{j} = O_{j-1}\,e^{m_{j-1}-m_j} + \sum_{k \in \text{block } j} e^{s_k - m_j} v_k$, the final output is $O_n/\ell_n$. Peak memory per query block drops from $O(n)$ scores per row to $O(B_c)$, and HBM traffic drops because $S$ and $P$ are never written.` },
      { t: "h", text: "Load-balancing loss" },
      { t: "math", tex: String.raw`\mathcal L_{\text{aux}} = \alpha \cdot E \sum_{e=1}^{E} f_e\,P_e, \qquad f_e = \frac{1}{T}\sum_{t} \mathbb 1[\arg\max_i p_i(x_t) = e], \qquad P_e = \frac{1}{T}\sum_t p_e(x_t)`, caption: "Switch Transformers (Fedus et al., 2022). $E$ = experts, $T$ = tokens in the batch, $f_e$ = share routed to expert $e$ (not differentiable), $P_e$ = mean router probability (differentiable), $\\alpha$ ≈ 0.01." },
      { t: "p", text: String.raw`With uniform routing, $f_e = P_e = 1/E$ and $E\sum_e f_eP_e = 1$. Example with $E=4$, $f = (0.5, 0.25, 0.25, 0)$ and $P = (0.4, 0.3, 0.2, 0.1)$: $4(0.2 + 0.075 + 0.05 + 0) = 1.3$. The gradient flows only through $P_e$, weighted by $f_e$, so over-used experts get their router probabilities pushed down.` },
      { t: "h", text: "Speculative decoding preserves the distribution" },
      { t: "p", text: String.raw`Let $p$ be the target distribution and $q$ the draft distribution at one position. The draft samples $x\sim q$ and keeps it with probability $\min(1, p(x)/q(x))$. On rejection, a token is drawn from $r(x) = \max(0, p(x)-q(x)) / \sum_u \max(0,p(u)-q(u))$. The probability of outputting $x$ is` },
      { t: "math", tex: String.raw`\Pr[x] = q(x)\min\!\Big(1,\tfrac{p(x)}{q(x)}\Big) + (1-\beta)\,r(x) = \min(p(x),q(x)) + \max(0, p(x)-q(x)) = p(x)`, caption: "$\\beta = \\sum_x \\min(p(x), q(x))$ is the acceptance probability, and $\\sum_u \\max(0, p(u)-q(u)) = 1-\\beta$, which cancels the normaliser of $r$." },
      { t: "p", text: String.raw`In plain words: rejections trim over-proposed tokens down to $p$, and the leftover tops up under-proposed ones. Worked example: $p = (0.6, 0.3, 0.1)$, $q = (0.3, 0.5, 0.2)$. Acceptance probabilities $(1, 0.6, 0.5)$, $\beta = 0.3+0.3+0.1 = 0.7$, leftover $r = (1, 0, 0)$. Token 1: $0.3\cdot1 + 0.3\cdot1 = 0.6$; token 2: $0.5\cdot0.6 = 0.3$; token 3: $0.2\cdot0.5 = 0.1$. Exactly $p$.` },
      { t: "p", text: String.raw`If each draft token is accepted independently with probability $\beta$, the expected number of tokens per target call with $\gamma$ drafts is $(1-\beta^{\gamma+1})/(1-\beta)$ (Leviathan et al., 2023). For $\beta = 0.8$: 2.44 tokens at $\gamma=2$, 3.36 at $\gamma=4$, 4.33 at $\gamma=8$ — shrinking returns, while the draft cost grows linearly with $\gamma$.` },
      { t: "h", text: "Weight-only quantization" },
      { t: "p", text: String.raw`Symmetric per-channel int8: for each output row, $s = \max_j |w_j| / 127$, store $q_j = \mathrm{round}(w_j/s)\in[-127,127]$, and compute with $\hat w_j = s\,q_j$. Example row $w = (0.12, -0.50, 0.03, 0.31)$: $s = 0.5/127 = 0.003937$, $q = (30, -127, 8, 79)$, $\hat w = (0.11811, -0.5, 0.031496, 0.311024)$, max error 0.00189 $\le s/2$.` },
      { t: "p", text: "Int4 has only 15 levels, so one outlier inflates $s$ for its whole row; **group-wise** scales (e.g. one per 64 or 128 weights) limit the damage at the cost of extra scale storage. Weights are dequantized on the fly inside the matmul kernel, so decode reads half (int8) or a quarter (int4) of the bytes — almost directly a speed-up, since decode is memory-bound." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "sim", id: "transformer", caption: "The trained character Transformer: every generated character reruns the full forward pass over the context — exactly the work a KV cache avoids." },
      { t: "list", items: [
        "Press **+1 char** in the Generate panel a few times and open the **Attn** stage of layer 1. Earlier positions' keys and values don't change when a character is added (causal mask) — that's what a KV cache stores; only the new bottom row is new work.",
        "Size the cache for this model: 2 layers × 4 heads × 12 dims × 2 (K and V) = 192 numbers per character, so a full 64-character context holds 12,288 numbers.",
        "In the attention heatmap, each row needs only its own query plus the cached keys/values — which is why decode is a matrix–vector product per head.",
      ] },
      { t: "sim", id: "decoding", caption: "Beam search multiplies the decode work: width B means B sequences and B KV caches per step." },
      { t: "p", text: "In the decoding simulator, switch to **Beam search** and watch the candidate count in the narration: B beams × 37 characters scored per step, one forward pass per beam." },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "h", text: "1. KV cache: cached decode equals uncached, with 15× fewer operations" },
      { t: "p", text: "A tiny two-layer decoder with random weights. Both loops sample with the same pre-drawn random numbers, so any difference in logits would change the tokens. A counter tallies multiply-accumulates in the projections and attention." },
      { t: "code", lang: "python", title: "kv_cache_numpy.py", code: KV, output: KV_OUT },
      { t: "h", text: "2. Online softmax and tiled (FlashAttention-style) attention" },
      { t: "code", lang: "python", title: "tiled_attention_numpy.py", code: FLASH, output: FLASH_OUT },
      { t: "p", text: "The tiled loop never holds more than a 16 × 16 block of scores (256 numbers instead of 40,000), skips fully-future blocks when causal, and still matches exact attention to $10^{-16}$. The last two lines use a rough count of numbers moved to and from slow memory: the advantage grows with $n$, because standard attention moves $O(n^2)$ score entries." },
      { t: "h", text: "3. Mixture of experts, speculative decoding and weight-only quantization" },
      { t: "code", lang: "python", title: "moe_speculative_quant_numpy.py", code: MOE, output: MOE_OUT },
      { t: "list", items: [
        "Gradient descent on the differentiable part of the balancing loss turns a skewed routing (34 to 93 tokens per expert) into a perfectly even one.",
        "The accept/reject rule's exact output distribution equals $p$, and 200,000 simulated steps agree to sampling error.",
        "Int4 is far lossier than int8. Group-wise scales help only a little here because these weights are Gaussian; real LLM weights with outlier channels benefit much more.",
      ] },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "The same ideas in PyTorch and Hugging Face, on randomly initialised models (no download): the library's KV cache (`past_key_values`), PyTorch's fused `scaled_dot_product_attention` with grouped-query attention, greedy speculative decoding with a layer-skipping draft, and weight-only int8." },
          { t: "code", lang: "python", title: "inference_torch.py", code: TORCH, output: TORCH_OUT },
          { t: "p", text: "The greedy speculative version accepts a draft token when it equals the target's arg-max; it reproduces target greedy decoding exactly with 8 target passes for 20 tokens. The draft is the target's first two of four layers (self-speculation): half the cost, agreeing often but not always. On a GPU, `scaled_dot_product_attention` uses a FlashAttention kernel when shapes and dtypes allow." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "gqa_keras.py (not executed in CI)", code: KERAS, output: "Prints (2, 16, 8, 32): causal grouped-query attention output for 2 sequences, 16 positions, 8 query heads of size 32." },
          { t: "p", text: "Keras' `MultiHeadAttention` doesn't offer GQA directly. The explicit einsum version shows exactly where the savings come from: only the 2 key/value heads would be cached." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "table", head: ["Technique", "What it reduces", "Advantages", "Disadvantages", "Use"], rows: [
        ["KV cache", "Recomputing past K, V", "Turns O(n²) per step into O(n); exact", "Memory grows with batch × length", "Every autoregressive server"],
        ["MQA / GQA", "KV-cache size and bandwidth", "4–32× smaller cache, faster decode", "MQA can lose quality; must be trained (or up-trained) in", "Most recent open LLMs use GQA"],
        ["Paged KV cache", "Fragmentation, duplicate prefixes", "High batch sizes, prefix sharing", "Custom attention kernels", "vLLM and similar servers"],
        ["FlashAttention", "HBM traffic of attention", "Exact, faster, O(n) memory", "Kernel engineering per hardware", "Training and prefill everywhere"],
        ["Mixture of experts", "FLOPs per token for a given parameter count", "Large capacity at small compute", "All experts must sit in memory; routing imbalance; communication", "Large frontier and open models"],
        ["Speculative decoding", "Target forward passes per token", "Exact distribution, 2–3× faster decode", "Needs a good draft; wasted work on rejection", "Latency-sensitive serving"],
        ["Weight-only int8/int4", "Bytes of weights read", "~2–4× less memory and faster decode", "Accuracy loss at int4 without care (GPTQ, AWQ)", "Running models on smaller GPUs"],
        ["KV-cache quantization", "Bytes of cache", "Longer contexts / bigger batches", "Accuracy risk for keys with outliers", "Long-context serving"],
      ] },
      { t: "p", text: "Named methods you'll meet: **LLM.int8()** (Dettmers et al., 2022) keeps outlier features in fp16 during int8 matmuls; **GPTQ** (Frantar et al., 2023) quantizes weights layer by layer, correcting errors with second-order information; **AWQ** (Lin et al., 2024) protects the weights that matter for large activations. Sliding-window attention and attention sinks cap the cache for very long streams." },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**LLM serving engines** (vLLM, Hugging Face TGI, NVIDIA TensorRT-LLM) combine continuous batching, paged KV caches, FlashAttention kernels, quantization and speculative decoding.",
        "**On-device and single-GPU inference**: llama.cpp-style runtimes use 4-bit group-wise weight quantization so 7B-parameter models fit in a few GB.",
        "**Long-context assistants** are limited by KV-cache memory far more than weights; GQA and cache quantization make 100k-token contexts affordable.",
        "**Mixture-of-experts models** such as Mixtral 8×7B (Jiang et al., 2024) use 2 of 8 experts per token per layer, giving the per-token compute of a much smaller dense model.",
        "**Prompt caching**: a shared system prompt's KV cache is computed once and reused across requests.",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "list", items: [
        "**Latency vs throughput**: bigger batches raise the server's tokens/second but also each request's latency; cache memory caps the batch size.",
        "**Memory vs compute**: the KV cache trades memory for compute; recomputation (or MQA) trades back. FlashAttention recomputes in the backward pass to save memory.",
        "**Quality vs speed**: GQA/MQA, int4 weights and cache quantization can cost accuracy; KV cache, FlashAttention and speculative decoding are exact.",
        "**MoE**: low compute per token, but memory holds all experts, expert parallelism needs all-to-all communication, and small batches waste the sparsity.",
        "**Speculative decoding** helps only when the draft agrees often and is much cheaper; at high temperature acceptance drops, and at large batch sizes the GPU is no longer idle, so the gain shrinks.",
      ] },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Wrong position for the new token** when decoding with a cache (e.g. position 0 instead of $t$, or RoPE with the wrong offset): output looks fluent but differs from the uncached model. Always compare cached vs uncached logits, as both code sections do.",
        "**Leaving out the new token's own key** from its attention, or masking the cache: in a decode step every cached key is in the past and must be visible.",
        "**Counting KV memory with $H$ instead of $H_{kv}$**, or forgetting the factor 2 (keys and values) or the batch size.",
        "**Thinking FlashAttention approximates attention.** It is exact up to floating-point rounding; its gains come from less memory traffic, not fewer FLOPs.",
        "**Greedy acceptance in speculative decoding under sampling.** Accepting a draft only when it equals the target's sample (or arg-max) doesn't give the target's sampling distribution; use the $\\min(1, p/q)$ rule with leftover resampling.",
        "**Skipping the leftover renormalisation** after a rejection, which biases the output toward the draft model.",
        "**Judging MoE cost by total parameters** (memory) or by active parameters (FLOPs) alone: you need both.",
        "**Quantizing per tensor with outliers present**: one large weight sets the scale for millions of others. Use per-channel or group-wise scales.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is stored in a KV cache and why does it speed up generation?", a: "The key and value vectors for every layer and every past token. Thanks to the causal mask, past tokens' keys and values never change when new tokens are added. So each decode step computes q, k, v only for the new token and attends over the stored ones, instead of rerunning the whole prefix." },
      { t: "qa", level: "beginner", q: "Why is the decode phase memory-bound while prefill is compute-bound?", a: "Decode handles one token per sequence: each weight read from memory is used for only ~2 FLOPs (a matrix-vector product), so the GPU waits on memory. Prefill handles hundreds of prompt tokens per weight read (matrix-matrix products), giving enough arithmetic per byte to keep the compute units busy." },
      { t: "qa", level: "beginner", q: "Write the KV-cache memory formula and evaluate it for 32 layers, 32 KV heads of size 128, 4096 tokens, fp16.", a: "M = 2 · L · H_kv · d_head · n · bytes = 2·32·32·128·4096·2 = 2,147,483,648 bytes = 2 GiB per sequence." },
      { t: "qa", level: "intermediate", q: "How do MQA and GQA reduce inference cost, and what do they cost?", a: "They keep all query heads but share key/value heads (MQA: one; GQA: H_kv groups), shrinking the cache and bytes read per decode step by H/H_kv. The price is less expressive attention; GQA with a moderate number of groups recovers almost all of MHA's quality, so it's the common choice. Existing MHA checkpoints can be converted by mean-pooling K/V heads and briefly up-training (Ainslie et al., 2023)." },
      { t: "qa", level: "intermediate", q: "Explain how FlashAttention computes exact softmax attention without the full score matrix.", a: "It runs a block of queries against streamed key/value blocks in on-chip SRAM, keeping per query row a running max m, running sum ℓ and an unnormalised output. When a block brings a larger max, the old sum and output are rescaled by exp(m_old − m_new). After the last block it divides by ℓ. This equals softmax(QKᵀ/√d)V exactly; it cuts HBM reads/writes, and the backward pass recomputes blocks instead of storing the n×n matrix." },
      { t: "qa", level: "intermediate", q: "What is the load-balancing loss in a mixture-of-experts layer and why is it needed?", a: "Routers tend to collapse onto a few experts, which train faster and get picked even more. The Switch Transformer loss α·E·Σ f_e P_e (f = share of tokens routed to e, P = mean router probability) is smallest under uniform routing; its gradient lowers the router probabilities of over-used experts. Capacity limits also cap tokens per expert." },
      { t: "qa", level: "advanced", q: "Prove that speculative sampling outputs tokens distributed exactly according to the target model.", a: "Draft x ~ q, accept with min(1, p(x)/q(x)), else sample from r = max(0, p − q)/(1 − β) with β = Σ min(p, q). Then Pr[x] = q(x)·min(1, p(x)/q(x)) + (1 − β)·r(x) = min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x). Applied position by position (stopping at the first rejection, which conditions later positions correctly), the whole sequence follows the target's distribution." },
      { t: "qa", level: "advanced", q: "A 70B dense model and a mixture-of-experts model with 47B total and 13B active parameters: compare their decode cost.", a: "Per token, the MoE does roughly a 13B dense model's FLOPs but must hold all 47B parameters. In memory-bound decode, bytes read depend on which experts the batch touches: at batch 1 only the active experts (~13B); at large batches nearly all experts every step. So versus 70B dense, MoE saves much compute and some memory, but at large batch its edge is mostly FLOPs, plus routing/communication overhead." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "KV-cache size in GiB for 40 layers, 8 KV heads of dimension 128, 8192 tokens, bf16, batch 1?", answer: 1.25, tolerance: 0.01, explain: "2·40·8·128·8192·2 = 1,342,177,280 bytes = 1.25 GiB." },
        { kind: "numeric", q: "A model has 32 query heads and uses GQA with 4 KV heads. By what factor is the KV cache smaller than with standard MHA?", answer: 8, tolerance: 0.01, explain: "H / H_kv = 32 / 4 = 8." },
        { q: "FlashAttention is faster than standard attention mainly because it…", options: ["approximates softmax with a kernel", "uses fewer FLOPs by skipping small scores", "reduces reads and writes to GPU high-bandwidth memory by tiling and never materialising the n×n matrices", "uses int8 arithmetic"], answer: 2, explain: "It is exact and does about the same FLOPs (slightly more in the backward pass, from recomputation); the win is memory traffic." },
        { kind: "numeric", q: "Online softmax: after processing x = (1, 3), the running sum is ℓ = 1.135335 with m = 3. A third element x = 4 arrives. What is the new ℓ? (4 decimals)", answer: 1.4177, tolerance: 0.0002, explain: "ℓ = 1.135335·e^{3−4} + e^0 = 0.417668 + 1 = 1.417668." },
        { kind: "numeric", q: "Target p = (0.5, 0.5), draft q = (0.8, 0.2). What is the acceptance probability β = Σ min(p, q)?", answer: 0.7, tolerance: 0.001, explain: "min(0.5, 0.8) + min(0.5, 0.2) = 0.5 + 0.2 = 0.7." },
        { q: "Which statement about mixture-of-experts layers is FALSE?", options: ["Each token is processed by only k of the E experts", "All experts' parameters must be stored", "MoE reduces the memory needed for weights compared with a dense model of equal total parameters", "A load-balancing loss prevents routing collapse"], answer: 2, explain: "Memory depends on total parameters, which an MoE still stores; what it reduces is compute per token." },
        { kind: "numeric", q: "With acceptance rate β = 0.6 and γ = 3 draft tokens, how many tokens are produced per target forward pass on average? (3 decimals)", answer: 2.176, tolerance: 0.002, explain: "(1 − 0.6⁴)/(1 − 0.6) = (1 − 0.1296)/0.4 = 2.176." },
      ] },
      { t: "exercise", title: "Size a serving deployment", prompt: "A model has 32 layers, 32 query heads, 8 KV heads, $d_{\\text{head}} = 128$ and 8B parameters stored in bf16, served on an 80 GB GPU. Keeping 6 GB for activations and overhead, how many concurrent 8192-token sequences fit? How many if the weights are quantized to int4 (ignore scale overhead)?", hint: "Weights: bytes = parameters × bytes per weight. Cache per sequence from the formula.", solution: [
        { t: "steps", items: [
          { title: "Weights", tex: String.raw`8\times10^9 \times 2 = 16\text{ GB (bf16)},\qquad 8\times10^9\times0.5 = 4\text{ GB (int4)}` },
          { title: "KV cache per sequence", tex: String.raw`2\cdot32\cdot8\cdot128\cdot8192\cdot2 = 1{,}073{,}741{,}824 \text{ bytes} \approx 1.074\text{ GB}` },
          { title: "bf16 weights", tex: String.raw`(80 - 16 - 6)/1.074 = 58/1.074 = 54.0 \Rightarrow 54 \text{ sequences}` },
          { title: "int4 weights", tex: String.raw`(80 - 4 - 6)/1.074 = 70/1.074 = 65.2 \Rightarrow 65 \text{ sequences}` },
        ] },
        { t: "p", text: "Quantizing weights frees 12 GB, worth 11 more long sequences. With full multi-head attention (32 KV heads) the cache would be 4.3 GB per sequence and only 13 sequences would fit in bf16 — for long-context batch serving, GQA matters more than weight quantization." },
      ] },
      { t: "exercise", title: "Implement speculative sampling for one step", prompt: "Write `speculative_step(p_list, q_list, drafts, rng)` that takes the target distributions $p_1..p_{\\gamma+1}$, draft distributions $q_1..q_\\gamma$ and the draft tokens, and returns the accepted tokens plus the correction or bonus token. Check by Monte Carlo that the first output token follows $p_1$.", hint: "Loop over drafts; accept with probability min(1, p[x]/q[x]); on rejection sample from normalised max(0, p − q) and return; if all accepted, sample from p_{γ+1}.", solution: [
        { t: "p", text: "The core is the single-position rule in part 2 of the third scratch program (`accept`, `residual`, and the Monte Carlo check). Wrap it in a loop over the γ positions, return at the first rejection, and append a sample from the target at position γ+1 when everything is accepted: that is the full algorithm of Leviathan et al. (2023) and Chen et al. (2023). The Monte Carlo frequencies (0.5008, 0.2494, 0.1508, 0.069, 0.03) match $p = (0.5, 0.25, 0.15, 0.07, 0.03)$ within sampling error." },
        { t: "callout", kind: "warning", text: "The target distributions at positions 2..γ+1 must be conditioned on the draft tokens (which the single verification pass does). After a rejection, throw away all later drafts: they were conditioned on a token no longer in the sequence." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Prefill: whole prompt at once, compute-bound (time to first token). Decode: one token per step, memory-bound (time per token).",
        String.raw`KV cache: $M = 2\,L\,H_{kv}\,d_{\text{head}}\,n\,b\,B$ bytes; exact; cuts per-step work from $O(t)$ tokens to 1.`,
        "MQA/GQA share key/value heads among query heads: the cache shrinks by $H/H_{kv}$.",
        String.raw`Online softmax: $m_j=\max(m_{j-1},x_j)$, $\ell_j = \ell_{j-1}e^{m_{j-1}-m_j}+e^{x_j-m_j}$; FlashAttention = tiling + online softmax, exact, IO-efficient.`,
        String.raw`MoE: top-$k$ of $E$ experts per token; balance with $\alpha E\sum_e f_eP_e$.`,
        String.raw`Speculative decoding: accept with $\min(1,p/q)$, resample from $\max(0,p-q)$; output $\sim p$ exactly; expected tokens per call $(1-\beta^{\gamma+1})/(1-\beta)$.`,
        "Weight-only quantization: per-channel int8 is nearly lossless; int4 needs group-wise scales and careful methods (GPTQ, AWQ).",
        "Always check optimisations against the unoptimised model's outputs.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency) — QLoRA fine-tunes on top of 4-bit weights; scaling laws tell you how big a model is worth serving.",
        "[Serialization, Inference Pipelines and APIs](/learn/m18/serving) — wrapping an optimised model in a production service.",
        "[Instruction Tuning, RLHF and DPO](/learn/m20/alignment) — RLHF generates millions of rollouts, so fast inference matters for training too.",
      ] },
    ],
  },
};

export default lesson;
