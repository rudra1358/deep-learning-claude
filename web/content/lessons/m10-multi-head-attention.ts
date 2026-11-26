import type { Lesson } from "@/lib/types";

const ONE_VS_TWO = String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Vocabulary of 4 tokens and 3 positions. Each input row = [token one-hot (4) | position one-hot (3)].
TOK = {"A": 0, "B": 1, "C": 2, "D": 3}
N_TOK, N_POS = 4, 3


def embed(seq):
    X = np.zeros((len(seq), N_TOK + N_POS))
    for p, s in enumerate(seq):
        X[p, TOK[s]] = 1.0          # what the token is
        X[p, N_TOK + p] = 1.0       # where it is
    return X


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def head(X, q_vec, W_K, W_V):
    """One attention head for the query of the LAST position. Returns (weights, output)."""
    k = X @ W_K                     # keys = position one-hots
    v = X @ W_V                     # values = token one-hots
    w = softmax(k @ q_vec)          # (the 1/sqrt(d_k) factor is folded into the query scale)
    return w, w @ v


W_K = np.vstack([np.zeros((N_TOK, N_POS)), np.eye(N_POS)])   # read the position part
W_V = np.vstack([np.eye(N_TOK), np.zeros((N_POS, N_TOK))])   # read the token part
c = 12.0                                                     # query scale = how sharp the softmax is

# Task: the last position (index 2) must report the ORDERED pair (token at 0, token at 1).
for seq in (["A", "B", "C"], ["B", "A", "C"]):
    X = embed(seq)
    # One head: its query asks for positions 0 AND 1 at once
    w1, z1 = head(X, c * np.array([1.0, 1.0, 0.0]), W_K, W_V)
    # Two heads: head 1 asks for position 0, head 2 asks for position 1
    wa, za = head(X, c * np.array([1.0, 0.0, 0.0]), W_K, W_V)
    wb, zb = head(X, c * np.array([0.0, 1.0, 0.0]), W_K, W_V)
    print("sequence", "".join(seq))
    print("  one head : weights", w1, "-> output", z1)
    print("  two heads: weights", wa, "and", wb)
    print("             concat output", np.concatenate([za, zb]))

# A single head can trade one position against the other, but never give both weight ~1:
print("\nweights a single head can give positions 0 and 1 (they must sum to <= 1):")
for bias in (-4.0, -1.0, 0.0, 1.0, 4.0):
    w, _ = head(embed(["A", "B", "C"]), np.array([c + bias, c, 0.0]), W_K, W_V)
    print(f"  query bias {bias:+.0f}: w0={w[0]:.4f}  w1={w[1]:.4f}  w0+w1={w[0] + w[1]:.4f}")`;
const ONE_VS_TWO_OUT = String.raw`sequence ABC
  one head : weights [0.5 0.5 0. ] -> output [0.5 0.5 0.  0. ]
  two heads: weights [1. 0. 0.] and [0. 1. 0.]
             concat output [1. 0. 0. 0. 0. 1. 0. 0.]
sequence BAC
  one head : weights [0.5 0.5 0. ] -> output [0.5 0.5 0.  0. ]
  two heads: weights [1. 0. 0.] and [0. 1. 0.]
             concat output [0. 1. 0. 0. 1. 0. 0. 0.]

weights a single head can give positions 0 and 1 (they must sum to <= 1):
  query bias -4: w0=0.0180  w1=0.9820  w0+w1=1.0000
  query bias -1: w0=0.2689  w1=0.7311  w0+w1=1.0000
  query bias +0: w0=0.5000  w1=0.5000  w0+w1=1.0000
  query bias +1: w0=0.7311  w1=0.2689  w0+w1=1.0000
  query bias +4: w0=0.9820  w1=0.0180  w0+w1=1.0000`;
const MHA_EINSUM = String.raw`import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def init_params(d_model, n_heads, n_kv_heads, rng):
    """Weights stored per head: W_Q (d_model, H, d_k), W_K/W_V (d_model, G, d_k), W_O (H, d_k, d_model).
    n_kv_heads = n_heads -> standard MHA; 1 -> multi-query (MQA); in between -> grouped-query (GQA)."""
    d_k = d_model // n_heads
    s = 1 / np.sqrt(d_model)
    return {
        "W_Q": rng.normal(0, s, (d_model, n_heads, d_k)),
        "W_K": rng.normal(0, s, (d_model, n_kv_heads, d_k)),
        "W_V": rng.normal(0, s, (d_model, n_kv_heads, d_k)),
        "W_O": rng.normal(0, s, (n_heads, d_k, d_model)),
    }


def mha_einsum(x_q, x_kv, p, mask=None):
    """x_q: (B, n_q, d_model), x_kv: (B, n_k, d_model). mask: bool, broadcastable to (B, H, n_q, n_k),
    True = may attend. Returns output (B, n_q, d_model) and weights (B, H, n_q, n_k)."""
    H, G = p["W_Q"].shape[1], p["W_K"].shape[1]
    d_k = p["W_Q"].shape[2]
    Q = np.einsum("bnd,dhk->bhnk", x_q, p["W_Q"])          # (B, H, n_q, d_k)
    K = np.einsum("bmd,dgk->bgmk", x_kv, p["W_K"])         # (B, G, n_k, d_k)
    V = np.einsum("bmd,dgk->bgmk", x_kv, p["W_V"])
    K = np.repeat(K, H // G, axis=1)                       # share each K/V head across H/G query heads
    V = np.repeat(V, H // G, axis=1)
    S = np.einsum("bhnk,bhmk->bhnm", Q, K) / np.sqrt(d_k)  # all heads' scores in one call
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    A = softmax(S, axis=-1)
    Z = np.einsum("bhnm,bhmk->bhnk", A, V)                 # per-head outputs
    out = np.einsum("bhnk,hkd->bnd", Z, p["W_O"])          # concat + W_O in one contraction
    return out, A


def mha_loop(x, p, mask=None):
    """Reference: explicit loop over batch and heads with plain matrix products (H == G only)."""
    B, n, d_model = x.shape
    H, d_k = p["W_Q"].shape[1:]
    out = np.zeros((B, n, d_model))
    for b in range(B):
        heads = []
        for h in range(H):
            q, k, v = x[b] @ p["W_Q"][:, h], x[b] @ p["W_K"][:, h], x[b] @ p["W_V"][:, h]
            s = q @ k.T / np.sqrt(d_k)
            if mask is not None:
                s = np.where(np.broadcast_to(mask, (B, H, n, n))[b, h], s, -np.inf)
            heads.append(softmax(s) @ v)
        concat = np.concatenate(heads, axis=-1)            # (n, H*d_k)
        out[b] = concat @ p["W_O"].reshape(H * d_k, d_model)
    return out


B, n, d_model, H = 2, 6, 16, 4
x = rng.normal(size=(B, n, d_model))
p = init_params(d_model, H, H, rng)

# 1) einsum == loop, with a causal mask shared by all heads, shape (1, 1, n, n)
causal = np.tril(np.ones((n, n), dtype=bool))[None, None]
y_e, A = mha_einsum(x, x, p, causal)
y_l = mha_loop(x, p, causal)
print("einsum vs loop, causal   : max |diff| =", f"{np.abs(y_e - y_l).max():.1e}")

# 2) A DIFFERENT mask per head, shape (1, H, n, n): heads 0-1 local (window 2), heads 2-3 causal
dist = np.arange(n)[:, None] - np.arange(n)[None, :]
local = (dist >= 0) & (dist < 2)
per_head = np.stack([local, local, causal[0, 0], causal[0, 0]])[None]
y_e2, A2 = mha_einsum(x, x, p, per_head)
print("einsum vs loop, per-head : max |diff| =", f"{np.abs(y_e2 - mha_loop(x, p, per_head)).max():.1e}")
print("row 5 of head 0 (local) :", np.round(A2[0, 0, 5], 3))
print("row 5 of head 3 (causal):", np.round(A2[0, 3, 5], 3))

# 3) Grouped-query and multi-query attention reuse the same function
for G, name in [(4, "MHA"), (2, "GQA"), (1, "MQA")]:
    pg = init_params(d_model, H, G, np.random.default_rng(1))
    yg, _ = mha_einsum(x, x, pg, causal)
    n_par = sum(v.size for v in pg.values())
    print(f"{name}: {H} query heads, {G} K/V heads -> output {yg.shape}, params {n_par}")


# 4) Parameters and FLOPs do not depend on H when d_k = d_model / H
def cost(d_model, H, n):
    d_k = d_model // H
    params = 3 * d_model * H * d_k + H * d_k * d_model             # W_Q, W_K, W_V, W_O (no biases)
    proj_flops = 2 * n * params                                    # 2 FLOPs per multiply-add
    attn_flops = H * (2 * n * n * d_k) * 2                         # QK^T and A V, per head
    softmax_entries = H * n * n                                    # one score matrix per head
    return params, proj_flops, attn_flops, softmax_entries


print("\nd_model=512, n=1024:")
for H in (1, 2, 8, 16, 64):
    params, pf, af, se = cost(512, H, 1024)
    print(f"  H={H:2d} d_k={512 // H:3d}  params={params:,}  projection GFLOPs={pf / 1e9:.3f}  attention GFLOPs={af / 1e9:.3f}  score entries={se:,}")`;
const MHA_EINSUM_OUT = String.raw`einsum vs loop, causal   : max |diff| = 6.7e-16
einsum vs loop, per-head : max |diff| = 6.7e-16
row 5 of head 0 (local) : [0.    0.    0.    0.    0.313 0.687]
row 5 of head 3 (causal): [0.025 0.21  0.12  0.199 0.366 0.08 ]
MHA: 4 query heads, 4 K/V heads -> output (2, 6, 16), params 1024
GQA: 4 query heads, 2 K/V heads -> output (2, 6, 16), params 768
MQA: 4 query heads, 1 K/V heads -> output (2, 6, 16), params 640

d_model=512, n=1024:
  H= 1 d_k=512  params=1,048,576  projection GFLOPs=2.147  attention GFLOPs=2.147  score entries=1,048,576
  H= 2 d_k=256  params=1,048,576  projection GFLOPs=2.147  attention GFLOPs=2.147  score entries=2,097,152
  H= 8 d_k= 64  params=1,048,576  projection GFLOPs=2.147  attention GFLOPs=2.147  score entries=8,388,608
  H=16 d_k= 32  params=1,048,576  projection GFLOPs=2.147  attention GFLOPs=2.147  score entries=16,777,216
  H=64 d_k=  8  params=1,048,576  projection GFLOPs=2.147  attention GFLOPs=2.147  score entries=67,108,864`;
const KV = String.raw`# KV-cache size during autoregressive generation. Every layer stores K and V for every past token.
LAYERS, Q_HEADS, D_K, BYTES = 80, 64, 128, 2      # 80 layers, 64 query heads of size 128, fp16/bf16
D_MODEL = Q_HEADS * D_K                            # 8192


def kv_bytes_per_token(kv_heads):
    return 2 * LAYERS * kv_heads * D_K * BYTES     # 2 = one K and one V vector per head per layer


GiB = 2 ** 30
print(f"{'scheme':>12} {'K/V heads':>9} {'per token':>12} {'4k-token seq':>13} {'batch 16 x 4k':>14} {'W_K+W_V params/layer':>21}")
for name, g in [("MHA", 64), ("GQA-8", 8), ("MQA", 1)]:
    per_tok = kv_bytes_per_token(g)
    print(f"{name:>12} {g:>9} {per_tok / 1024:>9.0f} KiB {per_tok * 4096 / GiB:>9.2f} GiB {per_tok * 4096 * 16 / GiB:>10.1f} GiB"
          f" {2 * D_MODEL * g * D_K:>21,}")`;
const KV_OUT = String.raw`scheme K/V heads    per token  4k-token seq  batch 16 x 4k  W_K+W_V params/layer
         MHA        64      2560 KiB     10.00 GiB      160.0 GiB           134,217,728
       GQA-8         8       320 KiB      1.25 GiB       20.0 GiB            16,777,216
         MQA         1        40 KiB      0.16 GiB        2.5 GiB             2,097,152`;
const TORCH_CHECK = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def mha_einsum(x, W_Q, W_K, W_V, W_O, b, mask=None):
    """NumPy multi-head attention with per-head weight tensors (see the from-scratch section)."""
    H, G, d_k = W_Q.shape[1], W_K.shape[1], W_Q.shape[2]
    Q = np.einsum("bnd,dhk->bhnk", x, W_Q) + b["q"]
    K = np.repeat(np.einsum("bmd,dgk->bgmk", x, W_K) + b["k"], H // G, axis=1)
    V = np.repeat(np.einsum("bmd,dgk->bgmk", x, W_V) + b["v"], H // G, axis=1)
    S = np.einsum("bhnk,bhmk->bhnm", Q, K) / np.sqrt(d_k)
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    return np.einsum("bhnm,bhmk,hkd->bnd", softmax(S), V, W_O) + b["o"]


B, n, d_model, H = 2, 5, 32, 4
d_k = d_model // H
x = torch.randn(B, n, d_model)

# ---- 1) nn.MultiheadAttention: unpack its stacked weights into per-head tensors ----
mha = nn.MultiheadAttention(d_model, H, batch_first=True)
W_in = mha.in_proj_weight.detach().numpy()            # (3*d_model, d_model), used as x @ W.T
b_in = mha.in_proj_bias.detach().numpy()
per_head = lambda W: W.T.reshape(d_model, H, d_k)     # (d_model, d_model) -> (d_model, H, d_k)
W_Q, W_K, W_V = (per_head(W_in[i * d_model:(i + 1) * d_model]) for i in range(3))
W_O = mha.out_proj.weight.detach().numpy().T.reshape(H, d_k, d_model)
bias = {"q": b_in[:d_model].reshape(H, 1, d_k), "k": b_in[d_model:2 * d_model].reshape(H, 1, d_k),
        "v": b_in[2 * d_model:].reshape(H, 1, d_k), "o": mha.out_proj.bias.detach().numpy()}

# padding: sequence 2 has only 3 real tokens. PyTorch: True = IGNORE this key
key_padding = torch.tensor([[False] * 5, [False] * 3 + [True] * 2])
causal_blocked = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
y_t, w_t = mha(x, x, x, key_padding_mask=key_padding, attn_mask=causal_blocked, average_attn_weights=False)
allowed = (~key_padding.numpy())[:, None, None, :] & (~causal_blocked.numpy())[None, None]
y_np = mha_einsum(x.numpy(), W_Q, W_K, W_V, W_O, bias, allowed)
print("nn.MultiheadAttention vs NumPy einsum: max |diff| =", f"{np.abs(y_t.detach().numpy() - y_np).max():.1e}")
print("per-head weights shape:", tuple(w_t.shape), "| params:", sum(p.numel() for p in mha.parameters()))

# ---- 2) F.scaled_dot_product_attention on (B, H, n, d_k) tensors, with grouped K/V heads ----
G = 2                                                  # 4 query heads share 2 K/V heads
Wq = torch.randn(d_model, H, d_k) / d_model ** 0.5
Wk = torch.randn(d_model, G, d_k) / d_model ** 0.5
Wv = torch.randn(d_model, G, d_k) / d_model ** 0.5
Wo = torch.randn(H, d_k, d_model) / d_model ** 0.5
q = torch.einsum("bnd,dhk->bhnk", x, Wq)               # (B, 4, n, d_k)
k = torch.einsum("bnd,dgk->bgnk", x, Wk)               # (B, 2, n, d_k)
v = torch.einsum("bnd,dgk->bgnk", x, Wv)
z = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
y_sdpa = torch.einsum("bhnk,hkd->bnd", z, Wo)
zero = {"q": 0.0, "k": 0.0, "v": 0.0, "o": 0.0}
y_ref = mha_einsum(x.numpy(), Wq.numpy(), Wk.numpy(), Wv.numpy(), Wo.numpy(), zero,
                   np.tril(np.ones((n, n), dtype=bool)))
print("F.scaled_dot_product_attention (GQA, causal) vs NumPy: max |diff| =",
      f"{np.abs(y_sdpa.numpy() - y_ref).max():.1e}")
print("K/V tensor shape", tuple(k.shape), "vs query", tuple(q.shape))`;
const TORCH_CHECK_OUT = String.raw`nn.MultiheadAttention vs NumPy einsum: max |diff| = 1.6e-07
per-head weights shape: (2, 4, 5, 5) | params: 4224
F.scaled_dot_product_attention (GQA, causal) vs NumPy: max |diff| = 4.9e-07
K/V tensor shape (2, 2, 5, 8) vs query (2, 4, 5, 8)`;
const HEADS = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

# Task: at every position t >= 2, output TWO tokens: x[t-1] and x[t-2] (two separate linear classifiers).
# A single attention head produces ONE weighted average of values per position; two heads can each fetch one.
V, N, D_MODEL, STEPS, BATCH = 16, 12, 32, 400, 64
torch.set_num_threads(1)

ALLOWED = torch.tril(torch.ones(N, N, dtype=torch.bool), diagonal=-1)   # strictly earlier positions only
ALLOWED[0, 0] = True                                                     # position 0: avoid an empty row


def batch(gen, size):
    x = torch.randint(0, V, (size, N), generator=gen)
    return x, x[:, 1:-1], x[:, :-2]                  # inputs, x[t-1], x[t-2] for t = 2..N-1


class OneLayerAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, D_MODEL // n_heads
        self.tok = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(N, D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)          # same size for every head count
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)              # W_O
        self.readout = nn.Linear(D_MODEL, 2 * V)                        # two linear classifiers

    def forward(self, x, head_mask=None):
        B = x.shape[0]
        h = self.tok(x) + self.pos(torch.arange(N))
        q, k, v = self.qkv(h).view(B, N, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)  # (B, H, N, d_k)
        s = (q @ k.transpose(-1, -2) / self.d_k ** 0.5).masked_fill(~ALLOWED, float("-inf"))
        a = s.softmax(-1)                                                # (B, H, N, N)
        if head_mask is not None:                                        # (H,) of 0/1: switch heads off
            a = a * head_mask.view(1, -1, 1, 1)
        z = (a @ v).transpose(1, 2).reshape(B, N, D_MODEL)              # concatenate heads
        return self.readout(self.out(z)).view(B, N, 2, V), a


def train(n_heads, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = OneLayerAttention(n_heads)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(STEPS):
        x, y1, y2 = batch(gen, BATCH)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, 2:, 0].reshape(-1, V), y1.reshape(-1)) + \
               F.cross_entropy(logits[:, 2:, 1].reshape(-1, V), y2.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


TEST = batch(torch.Generator().manual_seed(123), 2000)                   # held-out sequences


def evaluate(model, head_mask=None):
    x, y1, y2 = TEST
    with torch.no_grad():
        logits, a = model(x, head_mask)
    acc1 = (logits[:, 2:, 0].argmax(-1) == y1).float().mean().item()
    acc2 = (logits[:, 2:, 1].argmax(-1) == y2).float().mean().item()
    return acc1, acc2, a


t = torch.arange(2, N)
models = {}
for H in (1, 2, 4):
    models[H] = model = train(H)
    acc1, acc2, a = evaluate(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"heads={H} d_k={D_MODEL // H:2d} params={n_params}  held-out acc: x[t-1] {acc1:.3f}, x[t-2] {acc2:.3f}")
    for h in range(H):                    # mean attention paid to offsets 1 and 2 (positions t >= 2)
        print(f"   head {h}: weight on t-1 = {a[:, h, t, t - 1].mean():.3f}   weight on t-2 = {a[:, h, t, t - 2].mean():.3f}")

torch.set_printoptions(precision=2, sci_mode=False)
a2 = evaluate(models[2])[2]
for h in range(2):
    print(f"\n2-head model, head {h}, first sequence, query rows t=2..6, key columns 0..6:")
    print(a2[0, h, 2:7, :7])

# Head ablation ("pruning") on the 4-head model: switch heads off at test time, no retraining
print("\n4-head model with heads removed:")
for removed in [(), (0,), (1,), (2,), (3,), (0, 1), (2, 3)]:
    mask = torch.ones(4)
    mask[list(removed)] = 0.0
    acc1, acc2, _ = evaluate(models[4], mask)
    print(f"   removed {str(removed):7s} acc x[t-1] {acc1:.3f}   acc x[t-2] {acc2:.3f}")`;
const HEADS_OUT = String.raw`heads=1 d_k=32 params=6048  held-out acc: x[t-1] 0.229, x[t-2] 0.924
   head 0: weight on t-1 = 0.428   weight on t-2 = 0.504
heads=2 d_k=16 params=6048  held-out acc: x[t-1] 1.000, x[t-2] 1.000
   head 0: weight on t-1 = 0.996   weight on t-2 = 0.001
   head 1: weight on t-1 = 0.001   weight on t-2 = 0.999
heads=4 d_k= 8 params=6048  held-out acc: x[t-1] 1.000, x[t-2] 1.000
   head 0: weight on t-1 = 0.993   weight on t-2 = 0.003
   head 1: weight on t-1 = 0.982   weight on t-2 = 0.006
   head 2: weight on t-1 = 0.008   weight on t-2 = 0.976
   head 3: weight on t-1 = 0.002   weight on t-2 = 0.997

2-head model, head 0, first sequence, query rows t=2..6, key columns 0..6:
tensor([[0.01, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00]])

2-head model, head 1, first sequence, query rows t=2..6, key columns 0..6:
tensor([[0.99, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00]])

4-head model with heads removed:
   removed ()      acc x[t-1] 1.000   acc x[t-2] 1.000
   removed (0,)    acc x[t-1] 0.798   acc x[t-2] 1.000
   removed (1,)    acc x[t-1] 0.935   acc x[t-2] 1.000
   removed (2,)    acc x[t-1] 1.000   acc x[t-2] 0.993
   removed (3,)    acc x[t-1] 1.000   acc x[t-2] 0.715
   removed (0, 1)  acc x[t-1] 0.064   acc x[t-2] 1.000
   removed (2, 3)  acc x[t-1] 1.000   acc x[t-2] 0.062`;
const EX_KV = String.raw`LAYERS, Q_HEADS, D_K, BYTES, CONTEXT = 32, 32, 128, 2, 8192
for name, kv_heads in [("MHA", 32), ("GQA-8", 8), ("MQA", 1)]:
    per_token = 2 * LAYERS * kv_heads * D_K * BYTES          # K and V, every layer, every K/V head
    per_seq = per_token * CONTEXT
    print(f"{name:6s} {per_token / 1024:6.0f} KiB/token   {per_seq / 2**30:6.3f} GiB per 8k sequence"
          f"   {8 * per_seq / 2**30:6.2f} GiB for 8 sequences")`;
const EX_KV_OUT = String.raw`MHA       512 KiB/token    4.000 GiB per 8k sequence    32.00 GiB for 8 sequences
GQA-8     128 KiB/token    1.000 GiB per 8k sequence     8.00 GiB for 8 sequences
MQA        16 KiB/token    0.125 GiB per 8k sequence     1.00 GiB for 8 sequences`;
const EX_GQA = String.raw`import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


B, H, G, n, d_k = 2, 8, 2, 5, 4
Q = rng.normal(size=(B, H, n, d_k))
K = rng.normal(size=(B, G, n, d_k))          # only G = 2 K/V heads are ever stored
V = rng.normal(size=(B, G, n, d_k))

# Reference: copy each K/V head H/G times (simple, but materialises H K/V heads)
Kr, Vr = np.repeat(K, H // G, axis=1), np.repeat(V, H // G, axis=1)
ref = softmax(Q @ Kr.swapaxes(-1, -2) / np.sqrt(d_k)) @ Vr

# No copies: view the H query heads as G groups of H/G and let einsum broadcast over the group members
Qg = Q.reshape(B, G, H // G, n, d_k)                                 # head h -> group h // (H/G)
S = np.einsum("bgrnk,bgmk->bgrnm", Qg, K) / np.sqrt(d_k)
out = np.einsum("bgrnm,bgmk->bgrnk", softmax(S), V).reshape(B, H, n, d_k)

print("max |grouped - repeated| =", np.abs(out - ref).max())
print("K/V numbers stored: repeated", Kr.size + Vr.size, "| grouped", K.size + V.size)`;
const EX_GQA_OUT = String.raw`max |grouped - repeated| = 4.440892098500626e-16
K/V numbers stored: repeated 640 | grouped 160`;

const KERAS = String.raw`import numpy as np
import keras

x = np.random.default_rng(0).normal(size=(2, 10, 64)).astype("float32")   # (batch, n, d_model)

# Standard multi-head attention: 8 heads of size 8 (key_dim is the PER-HEAD size)
mha = keras.layers.MultiHeadAttention(num_heads=8, key_dim=8)
y, w = mha(query=x, value=x, key=x, use_causal_mask=True, return_attention_scores=True)
print("MHA output", y.shape, "| per-head weights", w.shape, "| params", mha.count_params())

# Grouped-query attention (Keras 3): 8 query heads share 2 key/value heads
gqa = keras.layers.GroupedQueryAttention(head_dim=8, num_query_heads=8, num_key_value_heads=2)
y2 = gqa(query=x, value=x, use_causal_mask=True)
print("GQA output", y2.shape, "| params", gqa.count_params())`;

const lesson: Lesson = {
  id: "m10/multi-head-attention",
  title: "Multi-Head Attention",
  summary:
    "Why one attention head is not enough (shown with a hand-built example and a trained experiment), how to code all heads at once with einsum, why the number of heads does not change parameters or FLOPs, how big heads should be, what heads learn in real models (positional, previous-token, induction and syntax heads), removing heads, per-head masks, cross-attention heads, and multi-query / grouped-query attention with KV-cache arithmetic.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m10/scaled-dot-product-attention", "m00/vectorization", "m10/positional-encoding"],
  next: ["m11/transformer-architecture", "m20/llm-inference", "m19/interpretability"],
  terms: ["multi-head-attention", "attention-head", "head-dimension", "induction-head", "multi-query-attention", "grouped-query-attention", "kv-cache"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "In [Queries, Keys, Values and Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) you saw the basic recipe of **multi-head attention (MHA)**. Project the input into queries, keys and values. Cut each into $h$ slices of width $d_k = d_{\\text{model}}/h$. Run attention separately in each slice — each slice is a **head**. Then glue the head outputs together and mix them with an output matrix $W_O$. Now we ask: **why** heads, **how** to code them fast, **what** they learn, and **how** modern models make them cheaper.",
      },
      {
        t: "p",
        text: "The reason: one head gives each position exactly one softmax distribution, so its output is one weighted average of value vectors. If a word needs two things at once — say the subject of its clause *and* the word just before it — one average blurs them together. Vaswani et al. (2017, “Attention Is All You Need”) added several heads to fix this. Their base model used $h = 8$ heads of size $d_k = 64$ for $d_{\\text{model}} = 512$, at the same cost as one full-width head.",
      },
      {
        t: "p",
        text: "Three later findings matter. **Interpretability research** found heads with clear jobs: looking at the previous token, tracking grammar, and *induction heads* that copy patterns seen earlier. **Pruning studies** showed many heads can be deleted after training with little loss. And the stored keys and values of every head (the KV cache) eat most memory during generation, which led to multi-query and grouped-query attention, now standard in large decoders.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The idea of this lesson",
        text: "One head = one weighted average per position. $h$ heads = $h$ independent weighted averages, each in its own learned subspace, combined by $W_O$ — for the same parameters and the same matrix-multiply FLOPs as one wide head.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Scaled dot-product attention](/learn/m10/scaled-dot-product-attention) — essential. You need $\\mathrm{softmax}(QK^\\top/\\sqrt{d_k}+M)V$, causal and padding masks, self- vs cross-attention, and the basic split–attend–concatenate recipe with its $4d^2+4d$ parameters. We do not repeat them here.",
          "[Vectorization and broadcasting](/learn/m00/vectorization) — the fast code is all reshapes, transposes, broadcasting and `einsum`. You should be able to predict the shape `np.einsum(\"bnd,dhk->bhnk\", x, W)` returns.",
          "[Positional encoding](/learn/m10/positional-encoding) — heads that look by *position* (\"the previous token\") only work because position is in the input; ALiBi gives each head its own distance penalty.",
          "[Backpropagation](/learn/m04/backpropagation) and a PyTorch training loop — we train a small model in which heads specialise.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "One reporter or a newsroom",
        text: "One reporter must split attention in fixed percentages — 60% here, 40% there — and files one blended note. A newsroom sends several reporters with different jobs: one finds the subject, one checks the word just before, one finds the last mention of this name. The editor ($W_O$) combines their notes. It costs the same as one reporter, because each carries a smaller notebook. (Where it breaks: nobody assigns the jobs — the heads pick them up during training.)",
      },
      {
        t: "p",
        text: "Why can't one head do it all? A softmax row is a probability distribution: weights are never negative and add up to 1. Weight near 1 on one position forces weight near 0 everywhere else. A head can be **sharp** (copy one position) or **spread** (average several). But it cannot copy two positions into two separate slots — whatever it reads from two positions arrives added together, in the same coordinates.",
      },
      {
        t: "p",
        text: "Several heads fix this. Each head has its own query and key matrices, so it asks its own question (\"where is the previous token?\", \"where is my verb?\"), and its own value matrix, so what it fetches lands in its own block of numbers. Glued together, facts from different positions sit side by side, not on top of each other.",
      },
      {
        t: "p",
        text: "And it is cheap: splitting $d_{\\text{model}}$ into $h$ heads adds no parameters. You swap one wide head ($d_k = d_{\\text{model}}$) for $h$ narrow ones. The price: each head compares queries and keys in a smaller space. That is enough for most relations, as long as $d_k$ does not get too small.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "1. Where one head runs out" },
      {
        t: "p",
        text: "For query position $i$, a head outputs $z_i = \\sum_j A_{ij} v_j$ with $A_{ij}\\ge0$ and $\\sum_j A_{ij}=1$. So $z_i$ lies in the **convex hull** of the value vectors — it is some blend of them. Two things follow:",
      },
      {
        t: "list",
        items: [
          "If the task needs the full content of positions $p$ and $q$, the head can give them at most total weight 1, so at least one is weakened.",
          "What it reads is **added** in the same coordinates. $0.5\\,v_A + 0.5\\,v_B$ equals $0.5\\,v_B + 0.5\\,v_A$, so the head cannot say which token came from which position.",
        ],
      },
      {
        t: "p",
        text: "The Math section builds an example where exactly this happens. The Framework section trains a model on a task that needs two relations: one head gets only 23% accuracy on one of the two outputs, two heads get 100%.",
      },
      { t: "h", text: "2. The batched implementation, shape by shape" },
      {
        t: "p",
        text: "Real code never loops over heads; it uses a few big tensor operations. With batch size $B$, sequence length $n$, width $d$ and $h$ heads of size $d_k = d/h$:",
      },
      {
        t: "steps",
        items: [
          { title: "Fused projection", text: "One matrix multiply makes Q, K and V for all heads: (B, n, d) @ (d, 3d) → (B, n, 3d). PyTorch's nn.MultiheadAttention stores exactly this stacked matrix as in_proj_weight." },
          { title: "Split into heads (view)", text: "Reshape the last axis: (B, n, 3d) → (B, n, 3, h, d_k). This is free — no data moves, only the indexing changes. Head i owns columns i·d_k … (i+1)·d_k − 1 of each of Q, K, V." },
          { title: "Move heads next to batch (transpose)", text: "Permute to (3, B, h, n, d_k). Now the head axis acts like a second batch axis, and every later matrix multiply is a batched matmul over (B, h)." },
          { title: "Scores for all heads at once", text: "S = Q Kᵀ / √d_k with shapes (B, h, n, d_k) @ (B, h, d_k, n) → (B, h, n, n). In einsum: \"bhnk,bhmk->bhnm\".", tex: String.raw`S_{bhnm} = \frac{1}{\sqrt{d_k}}\sum_{k} Q_{bhnk}K_{bhmk}` },
          { title: "Mask and softmax", text: "Add a mask that broadcasts to (B, h, n, n) — see below — then softmax over the last axis (the keys)." },
          { title: "Weighted values", text: "Z = A V: (B, h, n, n) @ (B, h, n, d_k) → (B, h, n, d_k)." },
          { title: "Merge heads", text: "Transpose back to (B, n, h, d_k) and reshape to (B, n, d). After the transpose the data is no longer stored in order (non-contiguous), so this reshape copies. In PyTorch .view fails here; use .reshape or .contiguous().view." },
          { title: "Output projection", text: "(B, n, d) @ W_O (d, d) → (B, n, d). If you view W_O as (h, d_k, d), merging plus projection is one contraction, \"bhnk,hkd->bnd\" — literally a sum over heads." },
        ],
      },
      {
        t: "p",
        text: "Fused kernels such as `F.scaled_dot_product_attention` do steps 4–6 in one call on the step-3 tensors, often without ever building the (B, h, n, n) score tensor (FlashAttention-style tiling).",
      },
      { t: "h", text: "3. Masks, per batch element and per head" },
      {
        t: "p",
        text: "The scores have shape (B, h, n, n), so a mask only has to **broadcast** to that shape. The shape you choose says what the mask depends on:",
      },
      {
        t: "table",
        head: ["Mask shape", "Depends on", "Example"],
        rows: [
          ["(1, 1, n, n)", "positions only", "causal mask, the same for every sequence and head"],
          ["(B, 1, 1, n)", "sequence, key position", "padding mask: hide the padded keys of each sequence"],
          ["(B, 1, n, n)", "sequence, both positions", "causal AND padding together; block-diagonal masks for packed sequences"],
          ["(1, h, n, n)", "head, both positions", "per-head patterns: some heads local (a window), some global; ALiBi's per-head distance penalties (added numbers, not true/false)"],
        ],
      },
      {
        t: "p",
        text: "Per-head masks let one layer mix short-sighted and far-sighted heads (Longformer-style local+global attention, ALiBi). The from-scratch code gives two heads a window-2 mask and two a causal mask, in one call.",
      },
      { t: "h", text: "4. Cross-attention heads" },
      {
        t: "p",
        text: "In encoder–decoder models (the original Transformer, T5, Whisper), cross-attention takes queries from the decoder and keys/values from the encoder output. Each head learns its own alignment: in translation, some track word-by-word source–target alignment, others look more broadly. Two practical facts:",
      },
      {
        t: "list",
        items: [
          "The encoder's K and V are computed **once** per input and reused at every decoding step, so they form a fixed-size cache.",
          "The padding mask has shape (B, 1, 1, n_src) — it hides *source* positions. Causal masking does not apply: the whole source is visible.",
        ],
      },
      { t: "h", text: "5. Multi-query and grouped-query attention" },
      {
        t: "p",
        text: "A decoder writes one token at a time, and each new query must look at the keys and values of **all** earlier tokens, in every layer and head. These are stored, not recomputed: the **KV cache**. Its size grows with layers × K/V heads × $d_k$ × tokens. Each new token must read all of it from memory, so generation is limited by memory speed (bandwidth), not by arithmetic.",
      },
      {
        t: "list",
        items: [
          "**Multi-query attention (MQA)** — Shazeer (2019), “Fast Transformer Decoding: One Write-Head is All You Need”: keep $h$ query heads but only **one** shared key head and value head. The cache shrinks $h$ times. Queries still differ per head, so heads can still look at different positions; they just read the same keys and values.",
          "**Grouped-query attention (GQA)** — Ainslie et al. (2023), “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”: split the $h$ query heads into $G$ groups, each sharing one K/V head. $G = h$ is plain MHA, $G = 1$ is MQA. For in-between $G$ the paper reports quality close to MHA at speed close to MQA. An existing MHA model can be converted: average (mean-pool) each group's K/V heads, then train a little more (*uptraining*, a small fraction of the original compute).",
          "Llama 2's 34B and 70B models and Mistral 7B use GQA with 8 key/value heads; the full cache story and other tricks are in [Efficient Inference](/learn/m20/llm-inference).",
        ],
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Notation" },
      {
        t: "list",
        items: [
          String.raw`$X\in\mathbb{R}^{n\times d}$ — the input, one row per token; $d = d_{\text{model}}$.`,
          String.raw`$h$ — number of query heads; $G$ — number of key/value heads ($G=h$ for MHA, $G=1$ for MQA); $d_k$ — head size, usually $d/h$.`,
          String.raw`$W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}\in\mathbb{R}^{d\times d_k}$ — the matrices of head $i$; $W_O\in\mathbb{R}^{hd_k\times d}$, cut into row blocks $W_O^{(i)}\in\mathbb{R}^{d_k\times d}$.`,
          String.raw`$A^{(i)}\in\mathbb{R}^{n\times n}$ — attention weights of head $i$; $Z^{(i)} = A^{(i)}XW_V^{(i)}\in\mathbb{R}^{n\times d_k}$ — its output.`,
        ],
      },
      { t: "h", text: "Concatenate-then-project is a sum over heads" },
      {
        t: "math",
        tex: String.raw`\mathrm{MHA}(X) = \big[Z^{(1)}\,\big|\,\cdots\,\big|\,Z^{(h)}\big]\begin{bmatrix}W_O^{(1)}\\ \vdots\\ W_O^{(h)}\end{bmatrix} = \sum_{i=1}^{h} Z^{(i)}W_O^{(i)} = \sum_{i=1}^{h} A^{(i)}\,X\,\underbrace{W_V^{(i)}W_O^{(i)}}_{d\times d,\ \text{rank}\le d_k}`,
        caption: "Block matrix multiplication: a row of blocks times a column of blocks is the sum of the block products.",
      },
      {
        t: "p",
        text: String.raw`Each head is its own term in a sum. It picks **where** to read with $A^{(i)}$ (built from $W_Q^{(i)}W_K^{(i)\top}$, a $d\times d$ matrix of rank $\le d_k$). It picks **what** to write with the low-rank map $W_V^{(i)}W_O^{(i)}$. The layer output is the sum of all heads' terms, added to the residual stream. Removing a head deletes one term, so pruning is easy to code.`,
      },
      { t: "h", text: "Why one head cannot copy two positions" },
      {
        t: "p",
        text: String.raw`One head outputs $z_i=\sum_j A_{ij}v_j$ with $\sum_jA_{ij}=1$. Say the task needs the tokens at positions $p$ and $q$ in separate coordinates, and the values are one-hot token codes. Then $z_i = A_{ip}e_{x_p} + A_{iq}e_{x_q} + \dots$, so: (a) $A_{ip}+A_{iq}\le 1$ — both tokens can never be copied at full strength; (b) swapping the two tokens and their weights gives the same vector, so with equal weights the order is lost. Two heads with $A^{(1)}_{ip}\approx1$ and $A^{(2)}_{iq}\approx1$ give $[e_{x_p}\,|\,e_{x_q}]$ after concatenation: both tokens, in order.`,
      },
      {
        t: "p",
        text: "A worked example. Inputs are [token one-hot (4) | position one-hot (3)]. Keys read the position part and values read the token part, so the query $c\\,e_p$ means \"look at position $p$\", with sharpness $c = 12$. The last position must report the ordered pair (token at 0, token at 1):",
      },
      { t: "code", lang: "python", title: "one_vs_two_heads.py", code: ONE_VS_TWO, output: ONE_VS_TWO_OUT },
      {
        t: "p",
        text: String.raw`For "ABC" and "BAC" the single head outputs the same vector $[0.5, 0.5, 0, 0]$, so the order is lost. The two heads give $[e_A|e_B]$ and $[e_B|e_A]$. The last block shows the single head's trade-off: changing the query only moves weight from one position to the other, and $w_0 + w_1 = 1$ every time. (The printed $1$ and $0$ for the two heads are really $1 - 6\times10^{-6}$ and $6\times10^{-6}$, rounded, since $e^{-12}\approx6\times10^{-6}$.) Unequal weights such as 0.73/0.27 *do* store order in their sizes. A model with a non-linear read-out can sometimes use that; a linear read-out cannot (see the trained experiment).`,
      },
      { t: "h", text: "Parameters do not depend on h" },
      {
        t: "math",
        tex: String.raw`\#\text{params} = \underbrace{3\cdot h\cdot d\cdot d_k}_{W_Q,W_K,W_V} + \underbrace{h\,d_k\cdot d}_{W_O} \;\overset{d_k=d/h}{=}\; 3d^2 + d^2 = 4d^2\quad(+\,4d\ \text{biases})`,
      },
      { t: "h", text: "FLOPs do not depend on h either" },
      {
        t: "p",
        text: "Count one multiply-add as 2 FLOPs (floating-point operations). For one sequence of $n$ tokens:",
      },
      {
        t: "math",
        tex: String.raw`\underbrace{2n\cdot 4d^2}_{\text{4 projections}} \;+\; \underbrace{h\cdot 2n^2d_k}_{QK^\top} \;+\; \underbrace{h\cdot 2n^2 d_k}_{AV} \;=\; 8nd^2 + 4n^2 d \qquad (h\,d_k = d)`,
      },
      {
        t: "p",
        text: String.raw`In words: $h$ cancels because $h\,d_k = d$. What **does** grow with $h$:`,
      },
      {
        t: "list",
        items: [
          String.raw`the number of score entries, $h\,n^2$ — more softmax work and, without fused kernels, more memory;`,
          String.raw`the number of small matrix multiplies — GPUs are used less well when $d_k$ is tiny;`,
          String.raw`the rank limit — each head's score matrix $Q^{(i)}K^{(i)\top}$ has rank at most $d_k$. Bhojanapalli et al. (2020, “Low-Rank Bottleneck in Multi-head Attention Models”) argue that a too-small $d_k = d/h$ limits what each head can express, and that head size should be chosen separately from $h$.`,
        ],
      },
      {
        t: "p",
        text: String.raw`The from-scratch code prints these numbers for $d = 512$, $n = 1024$ and $h$ from 1 to 64: 1,048,576 parameters and 2.147 GFLOPs in both parts for every $h$, while score entries grow from 1,048,576 to 67,108,864.`,
      },
      { t: "h", text: "Head-size choices in published models" },
      {
        t: "table",
        head: ["Model", "d_model", "heads h", "d_k"],
        rows: [
          ["Transformer base (Vaswani et al., 2017)", "512", "8", "64"],
          ["Transformer big", "1024", "16", "64"],
          ["BERT-base / GPT-2 small", "768", "12", "64"],
          ["BERT-large", "1024", "16", "64"],
          ["GPT-3 175B", "12288", "96", "128"],
          ["Llama 2 7B", "4096", "32", "128"],
        ],
      },
      {
        t: "p",
        text: "In practice $d_k$ is almost always 64 or 128, and $h$ grows with width. Multiples of 64 also suit GPU matrix units and fused attention kernels, which are written for a fixed set of head sizes.",
      },
      { t: "h", text: "KV-cache arithmetic" },
      {
        t: "math",
        tex: String.raw`\text{KV bytes per token} = 2\cdot L\cdot G\cdot d_k\cdot b,\qquad \text{KV bytes per sequence} = 2\,L\,G\,d_k\,b\cdot n`,
        caption: "2 = one key and one value vector; L layers; G key/value heads; b bytes per number (2 for fp16/bf16); n cached tokens.",
      },
      {
        t: "p",
        text: "For a model shaped like Llama 2 70B (80 layers, 64 query heads of size 128, so $d = 8192$):",
      },
      { t: "code", lang: "python", title: "kv_cache.py", code: KV, output: KV_OUT },
      {
        t: "p",
        text: String.raw`With full MHA, one 4,096-token sequence needs 10 GiB of cache — more than the weights of a 7B model in fp16 — and a batch of 16 needs 160 GiB. Eight K/V heads (GQA-8) divide this by $64/8 = 8$; MQA divides it by 64. The last column shows $W_K$ and $W_V$ shrink by the same factors, so a GQA layer has fewer parameters than an MHA layer of the same width ($2d^2 + 2dGd_k$ instead of $4d^2$).`,
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "p",
        text: "The explorer runs real multi-head self-attention ($d_{\\text{model}} = 8$) on the sentence you type, with seeded **random** weights. Random heads do not specialise, so use it to see *how heads work*; the trained experiment in the Framework section shows specialisation.",
      },
      { t: "sim", id: "attention", caption: "Multi-head attention computed live; every matrix shown is the real computation." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "Set **Heads** to 1, go to stage 7 (Softmax) and note the heatmap. Switch to 2, then 4 heads: $d_k$ drops to 4 and 2, and you get 2 or 4 different heatmaps for the same sentence. Same input and parameters, a different pattern per head (the head buttons pick which one is shown).",
          "With 4 heads, compare one token's row across heads. Each row adds up to 1 inside its own head — the \"h independent weighted averages\" of the Math section.",
          "Go to stage 9 (Concat & W_O) and follow the shapes $(n\\times8)\\to h\\times(n\\times d_k)\\to(n\\times8)\\to(n\\times8)$. Check that the parameter count stays the same when you change the number of heads.",
          "Turn on **Causal mask**: every head gets the same mask (a (1, 1, n, n) mask), so every head's first row becomes $[1, 0, \\dots]$.",
          "Press **New weights** a few times with 4 heads. Some draws give almost-uniform heads, others one sharp head — random heads vary a lot.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "A full NumPy version built on `einsum`. Weights are stored **per head** — $W_Q$ as (d, h, d_k), $W_O$ as (h, d_k, d) — so no reshapes are needed and the code reads like the equations. One function covers self- and cross-attention, per-head masks, and MHA/GQA/MQA. We check it against a plain loop and print the Math section's cost table.",
      },
      { t: "code", lang: "python", title: "mha_einsum.py", code: MHA_EINSUM, output: MHA_EINSUM_OUT },
      {
        t: "list",
        items: [
          "`np.einsum(\"bnd,dhk->bhnk\", x, W_Q)` projects and splits into heads in one step: the (B, n, d) @ (d, h·d_k) product plus the reshape/transpose, written as one sum over $d$.",
          "`np.einsum(\"bhnk,bhmk->bhnm\", Q, K)` is the batched $QK^\\top$: $b$ and $h$ appear in both inputs and the output (batch axes), $k$ is summed away, $n$ and $m$ stay.",
          "`np.einsum(\"bhnk,hkd->bnd\", Z, W_O)` sums over **both** $h$ and $k$ — concatenation and output projection in one call, exactly the $\\sum_i Z^{(i)}W_O^{(i)}$ identity.",
          "The per-head mask is a (1, 4, n, n) true/false array. Heads 0–1 see only the current and previous token (row 5 of head 0 has weight on positions 4 and 5 only); heads 2–3 see the whole past.",
          "`np.repeat(K, H // G, axis=1)` shares each K/V head among $H/G$ query heads. Real kernels avoid this copy (see the second exercise). The parameter counts 1024 → 768 → 640 are $W_Q + W_O$ (512) plus $W_K + W_V$ (512, 256, 128).",
          "The einsum and loop versions agree to $7\\times10^{-16}$ — just float64 rounding.",
        ],
      },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              { t: "h", text: "1. Checking the einsum version against PyTorch" },
              {
                t: "p",
                text: "`nn.MultiheadAttention` stacks its three projections in `in_proj_weight`, shape (3d, d), used as `x @ W.T`. Transpose each $d\\times d$ block and reshape it to (d, h, d_k), and you get exactly our per-head tensors. The check uses a causal mask **and** a padding mask together (PyTorch rule here: True = blocked). The second part runs grouped-query attention with `F.scaled_dot_product_attention(..., enable_gqa=True)`, which accepts keys/values with fewer heads than the queries.",
              },
              { t: "code", lang: "python", title: "mha_torch_check.py", code: TORCH_CHECK, output: TORCH_CHECK_OUT },
              { t: "h", text: "2. A trained experiment: heads specialise" },
              {
                t: "p",
                text: "Task: in random token sequences, every position $t\\ge2$ must output **two** tokens, $x_{t-1}$ and $x_{t-2}$, using two separate linear classifiers. Each position may attend only to strictly earlier positions, so all useful information must come through attention. Models with 1, 2 and 4 heads have the same width $d=32$ and the same parameter count. The script then measures held-out accuracy and each head's average attention on offsets 1 and 2, prints two attention maps, and switches heads of the 4-head model off at test time (head ablation).",
              },
              { t: "code", lang: "python", title: "heads_copy2.py", code: HEADS, output: HEADS_OUT },
              {
                t: "list",
                items: [
                  "**One head fails, as predicted.** It splits its weight 0.43/0.50 over the two positions. It gets 92.4% on $x_{t-2}$ but only 22.9% on $x_{t-1}$ (chance is 1/16 = 6.25%). A linear read-out cannot separate two tokens that arrive added together.",
                  "**Two heads specialise cleanly**: head 0 puts 0.996 on $t-1$, head 1 puts 0.999 on $t-2$. The printed maps are shifted diagonals — a textbook *previous-token head* and a *two-back head*. Nothing in the code told the heads which job to take; random initialisation broke the tie.",
                  "**Four heads form redundant pairs**: heads 0 and 1 both learn offset 1, heads 2 and 3 offset 2.",
                  "**Ablation** shows what redundancy means. Removing one head of a pair costs a little (head 2: 99.3%) or more (head 3: 71.5%), because the two are not exact copies. Removing **both** heads of a relation drops that output to chance (6.4% and 6.2%), while the other output stays at 100%.",
                ],
              },
              {
                t: "callout",
                kind: "tip",
                title: "Fast and correct in your own code",
                text: "Call `F.scaled_dot_product_attention(q, k, v, attn_mask=..., is_causal=...)` on (B, h, n, d_k) tensors instead of writing softmax yourself: it picks a fused FlashAttention or memory-efficient kernel when one is available. Careful: its boolean mask means **True = may attend**, the opposite of `nn.MultiheadAttention`'s `attn_mask` and `key_padding_mask`.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "`keras.layers.MultiHeadAttention` takes `num_heads` and `key_dim` — the **per-head** size, not $d_{\\text{model}}$. Keras 3 also has `keras.layers.GroupedQueryAttention` with separate numbers of query and key/value heads.",
              },
              { t: "code", lang: "python", title: "keras_heads.py (not executed in CI)", code: KERAS, output: "Prints the MHA output shape (2, 10, 64), the per-head weight shape (2, 8, 10, 10) and its parameter count (16,640 = 4·64·64 weights + 4·64 biases); then the GQA output shape (2, 10, 64) and a smaller parameter count, because the key and value projections produce only 2 heads of size 8." },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Variant", "What changes", "Advantages", "Disadvantages", "Used in"],
        rows: [
          ["Multi-head (MHA)", "h query, h key, h value heads", "Most expressive; the baseline", "Largest KV cache", "Original Transformer, BERT, GPT-2/3"],
          ["Multi-query (MQA)", "h query heads, 1 shared K/V head", "KV cache ÷ h; fastest decoding", "Some quality loss; training can be less stable", "PaLM; many inference-optimised models"],
          ["Grouped-query (GQA)", "h query heads in G groups, one K/V head per group", "Near-MHA quality at near-MQA speed; can be converted from MHA by uptraining", "One more setting to choose (G)", "Llama 2 34B/70B, Mistral 7B"],
          ["Multi-head latent attention (MLA)", "K and V rebuilt from a small cached vector per token", "Even smaller cache than GQA", "More complex; interacts with RoPE", "DeepSeek-V2 and later DeepSeek models"],
          ["Cross-attention heads", "Q from one sequence, K/V from another", "Links encoder and decoder, or two modalities; K/V computed once", "Needs a second sequence", "T5, Whisper, text-to-image models"],
          ["Per-head local / sliding window", "Some or all heads see only a window of keys", "Cost grows linearly with n for windowed heads", "Far-away information must hop through several layers", "Longformer-style models, Mistral 7B (sliding window)"],
          ["Per-head positional bias (ALiBi)", "Each head subtracts its own slope × distance", "Mix of local and global heads; works on longer inputs than seen in training", "Built-in preference for nearby tokens", "BLOOM, MPT"],
          ["Pruned heads", "Unimportant heads removed after training", "Fewer FLOPs, smaller cache", "Must estimate importance and fine-tune again", "Compression research and deployment"],
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Every Transformer layer** — encoders (BERT), decoders (GPT-style models), encoder–decoders (T5, Whisper) and Vision Transformers — uses multi-head attention. The head count is a standard architecture setting.",
          "**LLM serving**: GQA and MQA are chosen mainly for inference. A smaller KV cache allows longer contexts and bigger batches on the same GPU, so a server generates more tokens per second. See [Efficient Inference](/learn/m20/llm-inference).",
          "**Interpretability**: researchers study single heads to find mechanisms. Clark et al. (2019, “What Does BERT Look At? An Analysis of BERT's Attention”) found BERT heads that look at the previous or next token, at separator tokens, or follow grammar relations (a verb's direct object, a noun's determiner). Voita et al. (2019, “Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned”) found positional, syntactic and rare-word heads in translation encoders.",
          "**Induction heads and in-context learning**: Elhage et al. (2021, “A Mathematical Framework for Transformer Circuits”) and Olsson et al. (2022, “In-context Learning and Induction Heads”) describe a two-head circuit. A **previous-token head** in an earlier layer writes \"the token before me was A\" into each position. Later, at a new A, an **induction head** uses that note to find the token that followed the earlier A — call it B — and copies it, predicting B. So it completes the pattern [A][B] … [A] → [B]. Olsson et al. report these heads appear suddenly early in training, together with a jump in in-context learning, and give evidence that they explain much of it.",
          "**Model compression**: Michel, Levy and Neubig (2019, “Are Sixteen Heads Really Better than One?”) showed many heads of trained models can be removed at test time with little loss, and some layers can drop to one head. They rank heads by the gradient of the loss with respect to a head mask. Our 4-head ablation is the same experiment at toy size.",
          "**Alignment in translation and speech**: cross-attention heads give source–target alignments, used for example to get word- or segment-level timestamps in speech recognition.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Several relations per layer for free**: $h$ independent attention patterns for the parameter and matmul cost of one.",
          "**Clean split**: the layer is a sum of per-head terms, so heads are easy to study, switch off, prune or share (MQA/GQA).",
          "**Hardware friendly**: heads become an extra batch axis for batched matmuls and fused kernels.",
          "**Flexible**: masks and biases can differ per head, giving one layer both local and global views.",
        ],
      },
      { t: "h", text: "Limitations and trade-offs" },
      {
        t: "list",
        items: [
          "**More heads = narrower heads.** With $d_k = d/h$, each head's query–key comparison has rank $\\le d_k$; too many heads hurts. In practice $d_k$ stays at 64–128.",
          "**Score memory grows with h**: $h\\,n^2$ entries per layer without fused kernels.",
          "**Redundancy**: trained models have many removable heads. Spare heads can help training, but cost compute at inference.",
          "**KV cache**: with full MHA the cache grows with $h$; MQA/GQA shrink it at some cost in quality.",
          "**Interpretation is partial**: a clear pattern in one head (\"previous token\") is a clue, not proof, of its job. What it writes through $W_VW_O$, and how later layers use it, matter just as much.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Splitting heads with one reshape.** `Q.reshape(B, h, n, d_k)` on a (B, n, d) tensor mixes up positions and features. The right split: reshape to (B, n, h, d_k), then transpose to (B, h, n, d_k). Merging is the reverse (transpose, then reshape — which needs a contiguous copy).",
          "**Scaling by the wrong size.** The scale is $1/\\sqrt{d_k}$, the per-head size, not $1/\\sqrt{d_{\\text{model}}}$. With 12 heads of size 64 in a 768-wide model, using $\\sqrt{768}$ makes scores $\\sqrt{12}\\approx3.5$ times too small and attention too flat.",
          "**Thinking more heads means more parameters or capacity.** With $d_k = d/h$ there are $4d^2$ parameters for any $h$. Raising $h$ at fixed $d$ trades head width for head count; it adds no capacity.",
          "**Mask shapes that broadcast wrongly.** A (B, n) padding mask used directly on (B, h, n, n) scores either errors or, worse, broadcasts over the wrong axes. Reshape it to (B, 1, 1, n) so it hides **key** columns for every head and query.",
          "**Mixing mask rules.** `nn.MultiheadAttention`: True = blocked. `F.scaled_dot_product_attention`: True = allowed. Keras: 1 = allowed. Test that the upper triangle of a causal head's weights is exactly zero.",
          "**Copying K/V heads in GQA.** `repeat_interleave` of K and V up to $h$ heads before attention is correct, but it throws away the memory saving inside the attention call. Group the queries instead (exercise 2) or use a kernel with built-in GQA.",
          "**Treating one head's map as an explanation.** A head with weight 0.9 on a token may write almost nothing useful through $W_VW_O$, and later layers can override it. Test with ablations.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "Why does the Transformer use several attention heads instead of one?", a: "One head gives each position one softmax distribution, so one weighted average of values. Several heads give several independent distributions, each with its own query/key/value matrices. So one layer can gather different kinds of information (e.g. the previous token and the grammatical head word) and keep them in separate coordinates. Because each head has size $d/h$, this costs the same parameters and matmul FLOPs as one full-width head." },
      { t: "qa", level: "beginner", q: "With d_model = 768 and 12 heads, what is the head dimension and the shape of the per-head weights for a sequence of 100 tokens?", a: "$d_k = 768/12 = 64$. Each head has a 100 × 100 weight matrix, so the full weight tensor for one sequence is (12, 100, 100) — or (B, 12, 100, 100) for a batch." },
      { t: "qa", level: "beginner", q: "What does the output projection W_O do?", a: "It maps the glued-together head outputs (width $h d_k = d$) back to model width and mixes information across heads. Put another way, it is a sum of per-head maps, $\\sum_i Z^{(i)}W_O^{(i)}$: each head writes its result into the residual stream through its own $d_k\\times d$ block." },
      { t: "qa", level: "intermediate", q: "Prove that the FLOPs of multi-head attention do not depend on the number of heads.", a: "Projections: four $(n\\times d)(d\\times d)$ products, $8nd^2$ FLOPs, no $h$. Scores: each head does $(n\\times d_k)(d_k\\times n)$, $2n^2d_k$ FLOPs; times $h$ heads gives $2n^2hd_k = 2n^2d$. Same for $AV$. Total $8nd^2 + 4n^2d$. What does depend on $h$ is the number of score entries ($hn^2$), so softmax work and score memory." },
      { t: "qa", level: "intermediate", q: "Explain multi-query and grouped-query attention and why they speed up generation.", a: "When generating, the keys and values of all earlier tokens are cached and re-read for every new token, in every layer; the cache holds $2LGd_kn$ numbers. MHA has $G = h$ K/V heads. MQA shares one K/V head across all query heads ($G = 1$); GQA shares one per group ($1 < G < h$). The cache and the memory traffic per token shrink by $h/G$. Decoding is limited by memory bandwidth, so it gets faster and fits longer contexts and bigger batches. Query heads stay separate, so heads can still attend differently." },
      { t: "qa", level: "intermediate", q: "How would you find out which heads of a trained model matter?", a: "Ablate them: multiply each head's output by a gate (1 or 0) and measure how validation loss or the task metric changes when heads are switched off. Michel et al. (2019) estimate this cheaply with the gradient of the loss with respect to the gates. Test correlated heads together: in the experiment above, removing one of two redundant heads cost a few points, removing both dropped accuracy to chance." },
      { t: "qa", level: "advanced", q: "What is an induction head and why does it need two layers?", a: "An induction head completes the pattern [A][B] … [A] → [B]: at the current A it attends to the position right after an earlier A and copies that token. One head alone can match \"tokens equal to me\", but that finds the earlier A, not the token after it. So the circuit uses a previous-token head in an earlier layer, which writes into B's residual stream \"my predecessor was A\". The later induction head's keys read that note, so its query (A) matches B's key. This stacking across layers is why it needs at least two layers (Elhage et al., 2021; Olsson et al., 2022)." },
      { t: "qa", level: "advanced", q: "Why not use h = d_model heads of size 1?", a: "Each head's score $x_i^\\top W_Q^{(i)}W_K^{(i)\\top}x_j$ has rank $d_k$. With $d_k = 1$ a head can compare tokens along only one direction, so it cannot express most relations (Bhojanapalli et al., 2020, call this the low-rank bottleneck). It would also create $d\\,n^2$ score entries per layer and thousands of tiny, hardware-unfriendly matmuls. Practice settles on $d_k$ of 64–128 and grows $h$ with $d$." },
      { t: "qa", level: "advanced", q: "How can an existing multi-head checkpoint be converted to grouped-query attention?", a: "Ainslie et al. (2023) average (mean-pool) the key and value matrices of the heads in each group into one K/V head per group. Then they continue pre-training (uptraining) for a small fraction of the original compute so the model adapts. Query matrices and $W_O$ are kept. In their tests, mean-pooling beat picking one head or starting from random." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "d_model = 1024 and h = 16. What is the head dimension d_k?", answer: 64, tolerance: 0, explain: "$d_k = d_{\\text{model}}/h = 1024/16 = 64$." },
          { q: "Holding d_model fixed and setting d_k = d_model / h, what happens to the number of weights in W_Q, W_K, W_V, W_O when h doubles?", options: ["It doubles", "It halves", "It stays the same", "It quadruples"], answer: 2, explain: "Each projection is $d\\times hd_k = d\\times d$ whatever $h$ is: total $4d^2$." },
          { kind: "numeric", q: "How many attention-score entries does one layer compute for one sequence with h = 8 and n = 1024?", answer: 8388608, tolerance: 0, explain: "$h\\,n^2 = 8\\times1024^2 = 8{,}388{,}608$. This part of the cost does grow with $h$." },
          { q: "A single head's output at position i is Σ_j A_ij v_j with each row of A summing to 1. Which statement is true?", options: ["It can copy two different positions at full strength into separate coordinates", "Its output lies in the convex hull of the value vectors, so it is one weighted average", "It can attend to at most one position", "Its weights can be negative"], answer: 1, explain: "Weights that are never negative and add to 1 make the output a convex combination: one average, with a total weight of 1 to share." },
          { kind: "numeric", q: "d_model = 4096 with 32 query heads of size 128 and 8 key/value heads (GQA). How many weights do W_K and W_V have together (no biases)?", answer: 8388608, tolerance: 0, explain: "$2\\times4096\\times(8\\times128) = 8{,}388{,}608$, versus $2\\times4096\\times4096 = 33{,}554{,}432$ for full MHA — 4× fewer." },
          { q: "Which mask shape hides padded keys for every head and every query in a (B, h, n, n) score tensor?", options: ["(B, n)", "(B, 1, 1, n)", "(1, h, n, 1)", "(n, n)"], answer: 1, explain: "Padding depends only on the sequence and the key position, so the mask must broadcast over heads and queries: (B, 1, 1, n)." },
          { kind: "numeric", q: "A model has 32 layers, 8 K/V heads of size 128, fp16 (2 bytes). How many KiB of KV cache does each token need?", answer: 128, tolerance: 0, explain: "$2\\cdot32\\cdot8\\cdot128\\cdot2 = 131{,}072$ bytes = 128 KiB." },
          { q: "In an induction circuit, what does the earlier-layer previous-token head contribute?", options: ["It predicts the next token directly", "It writes 'my predecessor was A' into each position so the later head's keys can match on it", "It removes padding", "It normalises the attention weights"], answer: 1, explain: "The induction head's query is the current token A. It can only find the token after the earlier A if that position's key carries its predecessor — which the previous-token head provides." },
        ],
      },
      {
        t: "exercise",
        title: "KV cache for MHA, GQA and MQA",
        prompt: "A decoder has 32 layers and 32 query heads of size 128 and stores its cache in fp16. Work out the KV cache per token and for one 8,192-token sequence with (a) MHA, (b) GQA with 8 K/V heads, (c) MQA. How much memory do 8 such sequences need in each case?",
        hint: "Bytes per token = 2 · layers · K/V heads · d_k · 2.",
        solution: [
          { t: "p", text: "MHA: $2\\cdot32\\cdot32\\cdot128\\cdot2 = 524{,}288$ bytes = 512 KiB per token; × 8192 tokens = 4 GiB per sequence; 32 GiB for 8 sequences. GQA-8 divides by 4 (128 KiB, 1 GiB, 8 GiB); MQA divides by 32 (16 KiB, 0.125 GiB, 1 GiB)." },
          { t: "code", lang: "python", title: "exercise_kv.py", code: EX_KV, output: EX_KV_OUT },
        ],
      },
      {
        t: "exercise",
        title: "Grouped-query attention without copying K and V",
        prompt: "The from-scratch code uses `np.repeat` to give every query head its own copy of the shared K/V heads. Write GQA for Q of shape (B, H, n, d_k) and K, V of shape (B, G, n, d_k) **without** repeating K and V, and check it matches the repeated version.",
        hint: "Reshape Q to (B, G, H/G, n, d_k). Query head h belongs to group g = h // (H/G), which matches the order np.repeat makes its copies in.",
        solution: [
          { t: "code", lang: "python", title: "exercise_gqa.py", code: EX_GQA, output: EX_GQA_OUT },
          { t: "p", text: "The group axis $g$ is in Q, K and V, but the within-group axis $r$ is only in Q. So einsum broadcasts each K/V head over the $H/G$ query heads of its group. The results agree to $4\\times10^{-16}$ while storing 4× fewer K/V numbers — the saving that GQA kernels keep inside the attention step." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "One head gives one convex combination (blend) of values per position: it cannot copy two positions at full strength, and what it reads from different positions is added in the same coordinates.",
          String.raw`MHA runs $h$ heads of size $d_k = d/h$ side by side; $\mathrm{MHA}(X) = \sum_i A^{(i)}XW_V^{(i)}W_O^{(i)}$ — a sum of independent per-head terms.`,
          String.raw`Parameters $4d^2$ (+$4d$) and FLOPs $8nd^2 + 4n^2d$ do not depend on $h$; score entries $hn^2$ and the per-head rank limit $d_k$ do. Typical $d_k$: 64–128.`,
          "Code: fused projection → view (B, n, h, d_k) → transpose (B, h, n, d_k) → batched attention → transpose back → reshape → $W_O$; or einsum with per-head weight tensors.",
          "Masks broadcast to (B, h, n, n): causal (1,1,n,n), padding (B,1,1,n), per-head (1,h,n,n).",
          "Trained heads specialise (previous-token, positional, syntactic, induction heads), are often redundant, and many can be pruned; test with ablations, not by eye.",
          String.raw`KV cache $= 2LGd_k b$ bytes per token. MQA ($G=1$) and GQA ($1<G<h$) shrink it by $h/G$; GQA is standard in large decoders.`,
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[The Transformer Architecture](/learn/m11/transformer-architecture) — put multi-head attention into a full block with residual connections, layer normalisation and feed-forward layers, and see how heads write into the residual stream.",
          "[Tokenization, Embeddings and Context Windows](/learn/m11/tokens-embeddings) — what the rows of $X$ are before the first attention layer, and why long contexts are expensive.",
          "[Efficient Inference: KV Cache, FlashAttention, MoE](/learn/m20/llm-inference) — the full story of the KV cache, fused attention kernels and serving.",
          "[Interpretability](/learn/m19/interpretability) — ways to test what heads and neurons really do.",
        ],
      },
    ],
  },
};

export default lesson;
