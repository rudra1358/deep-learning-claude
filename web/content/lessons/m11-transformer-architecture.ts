import type { Lesson } from "@/lib/types";

/* ------------------------------------------------------------------ code blocks */

const BLOCK_NUMPY = String.raw`import numpy as np
import torch
import torch.nn as nn

# A full Transformer encoder block in NumPy, checked against torch.nn.TransformerEncoderLayer.
d_model, n_heads, d_ff = 16, 4, 64           # small but complete: 4 heads of size d_k = 4
n, batch = 6, 2                               # 6 tokens per sequence, 2 sequences
d_k = d_model // n_heads
eps = 1e-5                                    # LayerNorm epsilon (PyTorch default)


def layer_norm(x, gamma, beta):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)            # biased variance (divide by d), as PyTorch does
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def softmax(s):
    s = s - s.max(-1, keepdims=True)          # subtract the row max for numerical stability
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def multi_head_attention(x, p, mask=None):
    """x: (batch, n, d_model). Weights stored PyTorch-style: (out, in), so we multiply by W.T."""
    B, T, _ = x.shape
    q = x @ p["Wq"].T + p["bq"]               # (B, T, d_model)
    k = x @ p["Wk"].T + p["bk"]
    v = x @ p["Wv"].T + p["bv"]
    split = lambda t: t.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)   # (B, h, T, d_k)
    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)                      # (B, h, T, T)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)                             # True = blocked
    weights = softmax(scores)
    z = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, d_model)          # concat heads
    return z @ p["Wo"].T + p["bo"], weights


def ffn(x, p):
    return np.maximum(0, x @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"]    # ReLU, as in the paper


def encoder_block(x, p, norm_first=False, mask=None):
    if norm_first:                            # pre-LN (modern): normalise the input of each sublayer
        x = x + multi_head_attention(layer_norm(x, p["g1"], p["be1"]), p, mask)[0]
        x = x + ffn(layer_norm(x, p["g2"], p["be2"]), p)
    else:                                     # post-LN (original paper): normalise after the residual add
        x = layer_norm(x + multi_head_attention(x, p, mask)[0], p["g1"], p["be1"])
        x = layer_norm(x + ffn(x, p), p["g2"], p["be2"])
    return x


def params_from_torch(layer):
    """Copy every weight out of a TransformerEncoderLayer into a dict of NumPy arrays."""
    sd = {k: v.detach().double().numpy() for k, v in layer.state_dict().items()}
    Wq, Wk, Wv = np.split(sd["self_attn.in_proj_weight"], 3)   # stacked [W_Q; W_K; W_V], (3d, d)
    bq, bk, bv = np.split(sd["self_attn.in_proj_bias"], 3)
    return dict(Wq=Wq, Wk=Wk, Wv=Wv, bq=bq, bk=bk, bv=bv,
                Wo=sd["self_attn.out_proj.weight"], bo=sd["self_attn.out_proj.bias"],
                W1=sd["linear1.weight"], b1=sd["linear1.bias"], W2=sd["linear2.weight"], b2=sd["linear2.bias"],
                g1=sd["norm1.weight"], be1=sd["norm1.bias"], g2=sd["norm2.weight"], be2=sd["norm2.bias"])


torch.manual_seed(0)
x = torch.randn(batch, n, d_model, dtype=torch.float64)
causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)          # True above the diagonal

for norm_first in (False, True):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True,
                                       norm_first=norm_first, dtype=torch.float64).eval()
    with torch.no_grad():                      # LayerNorm starts at gamma=1, beta=0: randomise to test them too
        for norm in (layer.norm1, layer.norm2):
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.2, 0.2)
    p = params_from_torch(layer)
    for name, mask in (("no mask", None), ("causal", causal)):
        ours = encoder_block(x.numpy(), p, norm_first, None if mask is None else mask.numpy())
        with torch.no_grad():
            ref = layer(x, src_mask=mask).numpy()
        print(f"norm_first={norm_first!s:5}  {name:7}  max |numpy - torch| = {np.abs(ours - ref).max():.2e}")

# parameter count: formula vs PyTorch
count = sum(t.numel() for t in layer.parameters())
formula = (4 * d_model**2 + 4 * d_model) + (2 * d_model * d_ff + d_ff + d_model) + 4 * d_model
print(f"parameters: torch {count}, formula 4d^2+4d + 2*d*d_ff+d_ff+d + 4d = {formula}")

# what the causal mask does to the attention weights of one head
_, w = multi_head_attention(layer_norm(x.numpy(), p["g1"], p["be1"]), p, causal.numpy())
print("causal attention weights, sequence 0, head 0:")
print(np.round(w[0, 0], 3))
print("row sums:", np.round(w[0, 0].sum(-1), 6))`;

const BLOCK_NUMPY_OUT = String.raw`norm_first=False  no mask  max |numpy - torch| = 8.88e-16
norm_first=False  causal   max |numpy - torch| = 1.33e-15
norm_first=True   no mask  max |numpy - torch| = 4.44e-16
norm_first=True   causal   max |numpy - torch| = 4.44e-16
parameters: torch 3280, formula 4d^2+4d + 2*d*d_ff+d_ff+d + 4d = 3280
causal attention weights, sequence 0, head 0:
[[1.    0.    0.    0.    0.    0.   ]
 [0.584 0.416 0.    0.    0.    0.   ]
 [0.049 0.597 0.354 0.    0.    0.   ]
 [0.296 0.073 0.43  0.201 0.    0.   ]
 [0.034 0.566 0.14  0.181 0.078 0.   ]
 [0.072 0.427 0.157 0.102 0.128 0.114]]
row sums: [1. 1. 1. 1. 1. 1.]`;

const PARAMS_FLOPS = String.raw`import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

d, h, d_ff = 512, 8, 2048                      # "base" Transformer of Vaswani et al. (2017)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---- 1. parameter count per layer: formula vs PyTorch ----
mha = 4 * d * d + 4 * d                        # W_Q, W_K, W_V, W_O and their biases
ffn = 2 * d * d_ff + d_ff + d                  # W_1, b_1, W_2, b_2
ln = 2 * d                                     # gamma and beta
enc_formula = mha + ffn + 2 * ln
dec_formula = 2 * mha + ffn + 3 * ln           # masked self-attn + cross-attn + FFN
enc = nn.TransformerEncoderLayer(d, h, d_ff, dropout=0.0, batch_first=True)
dec = nn.TransformerDecoderLayer(d, h, d_ff, batch_first=True)
print(f"encoder layer: formula {enc_formula:,}  torch {n_params(enc):,}")
print(f"decoder layer: formula {dec_formula:,}  torch {n_params(dec):,}")
print(f"share of encoder-layer parameters in the FFN: {ffn / enc_formula:.1%}, in attention: {mha / enc_formula:.1%}")

# ---- 2. the whole base model (6 + 6 layers, one shared 37,000-token embedding, tied output) ----
vocab, N = 37000, 6
emb = vocab * d
total = emb + N * enc_formula + N * dec_formula
print(f"base model: embeddings {emb:,} + encoder {N * enc_formula:,} + decoder {N * dec_formula:,} = {total:,}")

# ---- 3. FLOPs per token: count the matrix multiplications PyTorch actually performs ----
torch.manual_seed(0)
n = 128                                        # sequence length
x = torch.randn(1, n, d)
enc.train()                                    # (eval mode would use a fused kernel the counter cannot see)
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    enc(x)
measured = fc.get_total_flops() / n            # FLOPs per token
weights = 4 * d * d + 2 * d * d_ff             # weight-matrix entries (biases/LN are negligible)
predicted = 2 * weights + 4 * n * d            # 2 FLOPs per weight + attention (QK^T and AV)
print(f"encoder layer, n={n}: measured {measured:,.0f} FLOPs/token, predicted 2*weights + 4*n*d = {predicted:,}")
print(f"attention-score share at n={n}: {4 * n * d / predicted:.1%};  at n=4096: {4 * 4096 * d / (2 * weights + 4 * 4096 * d):.1%}")`;

const PARAMS_FLOPS_OUT = String.raw`encoder layer: formula 3,152,384  torch 3,152,384
decoder layer: formula 4,204,032  torch 4,204,032
share of encoder-layer parameters in the FFN: 66.6%, in attention: 33.3%
base model: embeddings 18,944,000 + encoder 18,914,304 + decoder 25,224,192 = 63,082,496
encoder layer, n=128: measured 6,553,600 FLOPs/token, predicted 2*weights + 4*n*d = 6,553,600
attention-score share at n=128: 4.0%;  at n=4096: 57.1%`;

const ENCDEC_TORCH = String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# An encoder-decoder Transformer built from nn.TransformerEncoder / nn.TransformerDecoder,
# trained on a toy "translation" task: output the source digit sequence reversed.
torch.manual_seed(0)
PAD, BOS, EOS = 0, 1, 2                      # special tokens; digits 0-9 become tokens 3..12
V, L, D, H, FF, N_LAYERS = 13, 8, 64, 4, 128, 2


class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)        # shared by source, target and (tied) output head
        nn.init.normal_(self.emb.weight, 0.0, D ** -0.5)   # small init keeps the tied logits O(1)
        pe = torch.zeros(L + 2, D)           # sinusoidal positional encoding (Vaswani et al.)
        pos = torch.arange(L + 2).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, H, FF, dropout=0.0, batch_first=True, norm_first=True),
            N_LAYERS, norm=nn.LayerNorm(D), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(D, H, FF, dropout=0.0, batch_first=True, norm_first=True),
            N_LAYERS, norm=nn.LayerNorm(D))

    def embed(self, tokens):
        return self.emb(tokens) * math.sqrt(D) + self.pe[: tokens.shape[1]]

    def forward(self, src, tgt_in):
        memory = self.encoder(self.embed(src))                        # (B, n_src, D)
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.shape[1])
        h = self.decoder(self.embed(tgt_in), memory, tgt_mask=causal)  # masked self-attn + cross-attn
        return h @ self.emb.weight.T                                  # weight-tied logits (B, n_tgt, V)


def batch(B):
    src = torch.randint(3, 13, (B, L))
    tgt = src.flip(1)                                                 # the "translation"
    tgt_in = torch.cat([torch.full((B, 1), BOS), tgt], 1)             # decoder input: BOS + target
    tgt_out = torch.cat([tgt, torch.full((B, 1), EOS)], 1)            # decoder target: target + EOS
    return src, tgt_in, tgt_out


model = Seq2SeqTransformer()
enc_p = sum(p.numel() for p in model.encoder.parameters())
dec_p = sum(p.numel() for p in model.decoder.parameters())
print(f"params: embedding {model.emb.weight.numel()}, encoder {enc_p}, decoder {dec_p}, "
      f"total {sum(p.numel() for p in model.parameters())}")

src, tgt_in, tgt_out = batch(2)
with torch.no_grad():
    memory = model.encoder(model.embed(src))
    print("shapes: src", tuple(src.shape), "-> memory", tuple(memory.shape),
          "| tgt_in", tuple(tgt_in.shape), "-> logits", tuple(model(src, tgt_in).shape))

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(1, 301):
    src, tgt_in, tgt_out = batch(64)
    loss = F.cross_entropy(model(src, tgt_in).reshape(-1, V), tgt_out.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 75 == 0:
        print(f"step {step}  loss {loss.item():.4f}")


@torch.no_grad()
def greedy_translate(src):
    """Encode once, then decode autoregressively: each step re-reads the encoder memory."""
    memory = model.encoder(model.embed(src))
    out = torch.full((src.shape[0], 1), BOS)
    for _ in range(L + 1):
        causal = nn.Transformer.generate_square_subsequent_mask(out.shape[1])
        h = model.decoder(model.embed(out), memory, tgt_mask=causal)
        nxt = (h[:, -1] @ model.emb.weight.T).argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
    return out[:, 1:]


model.eval()
src, _, tgt_out = batch(500)
pred = greedy_translate(src)
print(f"exact-match accuracy on 500 new sequences: {(pred == tgt_out).all(1).float().mean().item():.3f}")
print("source :", (src[0] - 3).tolist())
print("decoded:", (pred[0, :L] - 3).tolist(), "then EOS" if pred[0, L] == EOS else "")`;

const ENCDEC_TORCH_OUT = String.raw`params: embedding 832, encoder 67072, decoder 100608, total 168512
shapes: src (2, 8) -> memory (2, 8, 64) | tgt_in (2, 9) -> logits (2, 9, 13)
step 75  loss 1.9242
step 150  loss 0.2877
step 225  loss 0.0137
step 300  loss 0.0069
exact-match accuracy on 500 new sequences: 1.000
source : [4, 1, 0, 6, 3, 0, 4, 0]
decoded: [0, 4, 0, 3, 6, 0, 1, 4] then EOS`;

const LOAD_CHAR = String.raw`import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the character-level Transformer that powers the simulator above, rebuild it from
# standard PyTorch modules, check it against the stored reference logits, and generate text.
spec = json.load(open("public/models/char-transformer.json", encoding="utf-8"))
cfg, vocab = spec["config"], spec["vocab"]
d, H, T = cfg["d_model"], cfg["n_heads"], cfg["context"]


def tensor(name):                                # stored as integers q with value = q * scale
    t = spec["tensors"][name]
    return torch.tensor(t["q"], dtype=torch.float64).mul(t["scale"]).float().view(*t["shape"])


class Block(nn.Module):                          # pre-LN: x + MHA(LN(x)), x + FFN(LN(x))
    def __init__(self, l):
        super().__init__()
        p = f"layers.{l}."
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, H, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, cfg["d_ff"]), nn.GELU(approximate="tanh"), nn.Linear(cfg["d_ff"], d))
        with torch.no_grad():                    # the file stores x @ W matrices; nn.Linear wants W.T
            self.ln1.weight.copy_(tensor(p + "ln1.g")); self.ln1.bias.copy_(tensor(p + "ln1.b"))
            self.ln2.weight.copy_(tensor(p + "ln2.g")); self.ln2.bias.copy_(tensor(p + "ln2.b"))
            self.attn.in_proj_weight.copy_(torch.cat([tensor(p + "W" + k).T for k in "qkv"]))
            self.attn.in_proj_bias.copy_(torch.cat([tensor(p + "b" + k) for k in "qkv"]))
            self.attn.out_proj.weight.copy_(tensor(p + "Wo").T); self.attn.out_proj.bias.copy_(tensor(p + "bo"))
            self.ff[0].weight.copy_(tensor(p + "W1").T); self.ff[0].bias.copy_(tensor(p + "b1"))
            self.ff[2].weight.copy_(tensor(p + "W2").T); self.ff[2].bias.copy_(tensor(p + "b2"))

    def forward(self, x):
        n = x.shape[1]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


blocks = nn.ModuleList([Block(l) for l in range(cfg["n_layers"])]).eval()
tok, pos = tensor("tok_emb"), tensor("pos_emb")
ln_f = nn.LayerNorm(d)
with torch.no_grad():
    ln_f.weight.copy_(tensor("ln_f.g")); ln_f.bias.copy_(tensor("ln_f.b"))


@torch.no_grad()
def logits_for(text):
    ids = torch.tensor([vocab.index(c) for c in text[-T:]])
    x = (tok[ids] + pos[: len(ids)])[None]
    for b in blocks:
        x = b(x)
    return (ln_f(x) @ tok.T)[0]                  # weight-tied head: (n, vocab)


n_params = sum(t["shape"][0] * (t["shape"][1] if len(t["shape"]) > 1 else 1) for t in spec["tensors"].values())
print(f"{n_params:,} parameters, vocab {len(vocab)}, context {T}, trained {spec['training']['steps']} steps,"
      f" val loss {spec['training']['final_val_loss']} nats/char")

v = spec["verify"][0]
err = max((logits_for(v["prompt"])[p] - torch.tensor(row)).abs().max().item() for p, row in zip(v["positions"], v["logits"]))
print(f"max |logit difference| vs the stored reference: {err:.1e}")


def generate(prompt, n, temperature=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    text = prompt
    for _ in range(n):
        z = logits_for(text)[-1]
        if temperature is None:
            nxt = int(z.argmax())                                   # greedy decoding
        else:
            nxt = int(torch.multinomial(F.softmax(z / temperature, -1), 1, generator=g))
        text += vocab[nxt]
    return text


print("greedy     :", generate("the rabbit ", 100))
print("T = 0.8    :", generate("the rabbit ", 100, temperature=0.8))
print("T = 1.5    :", generate("the rabbit ", 100, temperature=1.5))
p = F.softmax(logits_for("alice was beginning to get ver")[-1], -1)
top = p.topk(3)
print("after 'alice was beginning to get ver':", [(vocab[i], round(float(q), 3)) for q, i in zip(top.values, top.indices)])`;

const LOAD_CHAR_OUT = String.raw`61,488 parameters, vocab 37, context 64, trained 3000 steps, val loss 2.5149 nats/char
max |logit difference| vs the stored reference: 5.7e-06
greedy     : the rabbit was nothing on of the long, the world the was not a moment to be lost: she looked up, but it was to
T = 0.8    : the rabbit with their say small resself, 'oh do cats eat bats?' did nah, did alice dinah me right say 'wis t ri
T = 1.5    : the rabbit wo sto of that dark wwasigh, the rabit say gaingtuct y, in what it, exnyo w, (wouldgh milk '-jut  yo
after 'alice was beginning to get ver': [('y', 0.736), ('a', 0.102), (' ', 0.075)]`;

const KERAS_BLOCK = String.raw`import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class EncoderBlock(layers.Layer):
    """One Transformer encoder block; pre_ln=False is the original post-LN arrangement."""

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, rate=0.1, pre_ln=False):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.drop1, self.drop2 = layers.Dropout(rate), layers.Dropout(rate)
        self.pre_ln = pre_ln

    def call(self, x, training=False, causal=False):
        if self.pre_ln:
            h = self.ln1(x)
            x = x + self.drop1(self.mha(h, h, use_causal_mask=causal), training=training)
            return x + self.drop2(self.ffn(self.ln2(x)), training=training)
        x = self.ln1(x + self.drop1(self.mha(x, x, use_causal_mask=causal), training=training))
        return self.ln2(x + self.drop2(self.ffn(x), training=training))


x = tf.random.normal((2, 10, 512))
block = EncoderBlock()
y = block(x)
print(y.shape, block.count_params())`;

const DECODER_EX = String.raw`import numpy as np
import torch
import torch.nn as nn

d_model, n_heads, d_ff, eps = 16, 4, 64, 1e-5
d_k = d_model // n_heads


def layer_norm(x, g, b):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def mha(x_q, x_kv, W_in, b_in, W_out, b_out, mask=None):
    """Queries come from x_q, keys and values from x_kv (equal for self-attention)."""
    Wq, Wk, Wv = np.split(W_in, 3)
    bq, bk, bv = np.split(b_in, 3)
    B, Tq, _ = x_q.shape
    Tk = x_kv.shape[1]
    q = (x_q @ Wq.T + bq).reshape(B, Tq, n_heads, d_k).transpose(0, 2, 1, 3)
    k = (x_kv @ Wk.T + bk).reshape(B, Tk, n_heads, d_k).transpose(0, 2, 1, 3)
    v = (x_kv @ Wv.T + bv).reshape(B, Tk, n_heads, d_k).transpose(0, 2, 1, 3)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    if mask is not None:
        s = np.where(mask, -np.inf, s)
    z = (softmax(s) @ v).transpose(0, 2, 1, 3).reshape(B, Tq, d_model)
    return z @ W_out.T + b_out


def decoder_block(y, memory, P, causal):
    """Post-LN decoder layer: masked self-attention, cross-attention, FFN."""
    y = layer_norm(y + mha(y, y, *P["self"], mask=causal), *P["n1"])
    y = layer_norm(y + mha(y, memory, *P["cross"]), *P["n2"])        # Q from decoder, K/V from encoder
    ff = np.maximum(0, y @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]
    return layer_norm(y + ff, *P["n3"])


torch.manual_seed(0)
layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True,
                                   dtype=torch.float64).eval()
sd = {k: v.detach().numpy() for k, v in layer.state_dict().items()}
P = {"self": [sd[f"self_attn.{k}"] for k in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias")],
     "cross": [sd[f"multihead_attn.{k}"] for k in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias")],
     "W1": sd["linear1.weight"], "b1": sd["linear1.bias"], "W2": sd["linear2.weight"], "b2": sd["linear2.bias"],
     **{f"n{i}": (sd[f"norm{i}.weight"], sd[f"norm{i}.bias"]) for i in (1, 2, 3)}}

tgt = torch.randn(2, 5, d_model, dtype=torch.float64)       # 5 target positions
memory = torch.randn(2, 7, d_model, dtype=torch.float64)    # 7 encoder outputs
causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
ours = decoder_block(tgt.numpy(), memory.numpy(), P, causal.numpy())
with torch.no_grad():
    ref = layer(tgt, memory, tgt_mask=causal).numpy()
print("output shape:", ours.shape, " matches torch:", np.allclose(ours, ref, atol=1e-12))
formula = 2 * (4 * d_model**2 + 4 * d_model) + (2 * d_model * d_ff + d_ff + d_model) + 3 * 2 * d_model
print("parameters: torch", sum(p.numel() for p in layer.parameters()), " formula", formula)`;

const DECODER_EX_OUT = String.raw`output shape: (2, 5, 16)  matches torch: True
parameters: torch 4400  formula 4400`;

/* ------------------------------------------------------------------ lesson */

const lesson: Lesson = {
  id: "m11/transformer-architecture",
  title: "The Transformer Architecture",
  summary:
    "The full Transformer of “Attention Is All You Need”: why it replaced RNNs, the encoder and decoder blocks, the position-wise feed-forward network, the residual stream, post-LN vs pre-LN, weight tying, a shape-by-shape data-flow table, parameter and FLOP counts checked against PyTorch, a NumPy block checked against nn.TransformerEncoderLayer — and a real trained character-level Transformer you can step through in the browser.",
  minutes: 95,
  level: "advanced",
  prerequisites: ["m10/scaled-dot-product-attention", "m10/multi-head-attention", "m10/positional-encoding", "m07/normalization", "m07/residual-connections"],
  next: ["m11/tokens-embeddings", "m11/training-inference", "m11/transformer-families"],
  terms: ["transformer", "encoder-block", "decoder-block", "position-wise-ffn", "cross-attention", "pre-ln", "post-ln", "weight-tying", "residual-stream", "teacher-forcing"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "The **Transformer** is a sequence network built from three ingredients, repeated in a stack: **multi-head self-attention** (every position gathers information from every other), a **position-wise feed-forward network** (each position processed on its own), and **residual connections with layer normalization** around both. Vaswani et al. (2017, “Attention Is All You Need”) introduced it for translation. It now powers nearly every large language model, speech models like Whisper, and Vision Transformers.",
      },
      { t: "h", text: "The problem it solved: RNNs are slow and forgetful" },
      {
        t: "p",
        text: String.raw`A recurrent network computes $h_t = f(h_{t-1}, x_t)$ — each step needs the previous one. Two problems follow:`,
      },
      {
        t: "list",
        items: [
          String.raw`**No parallelism along the sequence**: $h_{100}$ cannot be computed before $h_{99}$, so a layer needs $n$ steps *one after another* for $n$ tokens, even on a GPU with thousands of cores.`,
          String.raw`**Long paths**: information from token 1 reaches token 100 only through 99 applications of $f$. The gradient travels back along the same chain and is multiplied by a Jacobian at every step, so long-range links are hard to learn ([BPTT](/learn/m09/bptt)). Seq2seq models with attention ([From Seq2Seq Bottlenecks to Attention](/learn/m10/attention-intro)) shortened the path between decoder and encoder, but both were still recurrent inside.`,
        ],
      },
      {
        t: "p",
        text: String.raw`The Transformer drops recurrence completely. In self-attention, position $i$ reads position $j$ **directly**, through one weighted sum. The paper's comparison (its Table 1), per layer, for $n$ tokens of size $d$:`,
      },
      {
        t: "table",
        head: ["Layer type", "Compute per layer", "Sequential operations", "Maximum path length"],
        rows: [
          ["Self-attention", "$O(n^2 d)$", "$O(1)$", "$O(1)$"],
          ["Recurrent", "$O(n d^2)$", "$O(n)$", "$O(n)$"],
          ["Convolutional (kernel $k$)", "$O(k\\,n\\,d^2)$", "$O(1)$", "$O(\\log_k n)$"],
        ],
      },
      {
        t: "p",
        text: "In plain words: self-attention pays $n^2$ compute for a one-step path between any two tokens and full parallelism. In training, every position of every sequence in a batch is processed at once by big matrix multiplications — what GPUs do best. That trade, more than any accuracy gain, is why the design scaled.",
      },
      {
        t: "callout",
        kind: "key",
        title: "One sentence",
        text: "A Transformer is a stack of identical blocks. In each block, positions first **talk to each other** (attention) and then **think alone** (feed-forward), and each sublayer **adds** its result to a running representation (the residual stream), kept in check by LayerNorm.",
      },
      {
        t: "p",
        text: "This lesson joins pieces you have met — [scaled dot-product attention](/learn/m10/scaled-dot-product-attention), [multi-head attention](/learn/m10/multi-head-attention), [positional encoding](/learn/m10/positional-encoding), [LayerNorm](/learn/m07/normalization) and [residual connections](/learn/m07/residual-connections) — into the full encoder–decoder model, counts its parameters and FLOPs exactly, then shows the decoder-only version behind GPT, with a real trained model to inspect.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Scaled dot-product attention](/learn/m10/scaled-dot-product-attention) — $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, self- vs cross-attention, padding and **causal masks**. Here we use attention as a black box with known shapes.`,
          String.raw`[Multi-head attention](/learn/m10/multi-head-attention) — splitting $d_{\text{model}}$ into $h$ heads of size $d_k = d_{\text{model}}/h$, concatenating and projecting with $W_O$; its $4d^2 + 4d$ parameters.`,
          "[Positional encoding](/learn/m10/positional-encoding) — attention ignores order, so order must be added: sinusoidal or learned position vectors added to the embeddings.",
          String.raw`[Normalization](/learn/m07/normalization) — **LayerNorm** normalizes each token's vector over its $d$ features (not over the batch), then applies a learned scale $\gamma$ and shift $\beta$.`,
          String.raw`[Residual connections](/learn/m07/residual-connections) — $x + F(x)$ keeps a straight path for gradients, which is what makes 6, 12 or 96 stacked blocks trainable.`,
          "[Classification losses](/learn/m05/classification-losses) — next-token prediction is a softmax over the vocabulary, trained with cross-entropy.",
          "Helpful background: [RNNs](/learn/m09/rnn-basics) and [seq2seq with attention](/learn/m10/attention-intro), for the motivation and the encoder–decoder idea.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A shared whiteboard, meetings, and quiet desk work",
        text: "Picture a row of desks, one per word, each with a whiteboard strip of that word's notes (its vector). Each round has two phases. **Meeting**: each person writes a question (query), everyone shows a label (key), and each person copies a blend of the others' content (values), weighted by how well labels match the question. **Desk work**: everyone thinks alone about their own strip, with the same rulebook (the feed-forward network). Nobody erases; they **add** a correction (residual connection) after tidying the writing to a standard size (LayerNorm). After six rounds the strips hold rich, context-aware notes.",
      },
      {
        t: "p",
        text: "Hold on to two things. First, **attention is the only place positions share information**; everything else works on each position alone. Second, the representation is a **running sum**: the input embedding plus every sublayer's contribution. This *residual stream* view explains why deep stacks train, why early information survives to the top, and why you can read off predictions halfway up.",
      },
      {
        t: "p",
        text: "For translation the original model has two stacks. The **encoder** reads the whole source sentence with unrestricted self-attention and outputs one vector per source token (the *memory*). The **decoder** writes the target one token at a time. It attends to the target written so far (with a causal mask, so it cannot peek ahead), then **cross-attends** to the encoder's memory to consult the source, then predicts the next token. GPT-style models keep only the decoder stack, without cross-attention: they just continue a text.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "The full encoder–decoder data flow (translation)" },
      {
        t: "steps",
        items: [
          { title: "Tokenize", text: "Source and target sentences become integer ids from a (sub-word) vocabulary. The target fed to the decoder is **shifted right**: a begin-of-sequence token, then the target without its last token. See [Tokenization and embeddings](/learn/m11/tokens-embeddings)." },
          { title: "Embed and add positions", text: "Each id picks a row of a learned embedding matrix $E\\in\\mathbb{R}^{V\\times d}$ (the paper multiplies it by $\\sqrt{d}$), and a positional encoding $p_i$ is added: $x_i = \\sqrt{d}\\,E[\\text{id}_i] + p_i$." },
          { title: "Encoder block (× N)", text: "Sublayer 1: multi-head **self-attention** over all source positions. Sublayer 2: position-wise FFN. In the original (post-LN) design each sublayer is wrapped as **LayerNorm(x + Sublayer(x))**. Dropout is applied to each sublayer output before the add." },
          { title: "Encoder output = memory", text: "After N blocks the encoder gives $M\\in\\mathbb{R}^{n\\times d}$, one context-aware vector per source token. It is computed **once** per source sentence." },
          { title: "Decoder block (× N)", text: "Sublayer 1: **masked** multi-head self-attention over the target so far (position $i$ sees positions $\\le i$). Sublayer 2: **cross-attention** — queries from the decoder, keys and values from the memory $M$. Sublayer 3: position-wise FFN. Each with residual + LayerNorm." },
          { title: "Output projection and softmax", text: "A linear map $d\\to V$ (tied to the embedding matrix in the paper) gives logits (raw scores) for every target position; softmax turns them into a distribution over the next token." },
          { title: "Training vs inference", text: "In training the whole shifted target goes in at once, and every position is trained in parallel with cross-entropy against the true next token (**teacher forcing**). At inference the decoder runs autoregressively: predict, append, repeat — see [Training and inference](/learn/m11/training-inference)." },
        ],
      },
      { t: "h", text: "The encoder block" },
      {
        t: "p",
        text: "An encoder block maps $X\\in\\mathbb{R}^{n\\times d}$ to an output of the **same shape**, which is what lets us stack blocks. Attention mixes information *across* positions (rows); the FFN mixes information *across features* (columns) within each row. Neither changes $d$, so the residual addition always fits.",
      },
      { t: "h", text: "The decoder block" },
      {
        t: "list",
        items: [
          "**Masked self-attention**: the same as encoder self-attention, plus a causal mask that sets scores for future positions to $-\\infty$ before the softmax. This makes parallel training match left-to-right generation: position $i$'s output never depends on tokens after $i$.",
          "**Cross-attention (encoder–decoder attention)**: $Q = YW_Q$ from the decoder stream, $K = MW_K$ and $V = MW_V$ from the encoder memory. The score matrix is $m\\times n$ (target length × source length) — the Transformer's version of the alignment matrix in [Bahdanau attention](/learn/m10/attention-intro). No causal mask: the whole source is already known.",
          "**FFN**: the same position-wise network as in the encoder.",
        ],
      },
      { t: "h", text: "The position-wise feed-forward network" },
      {
        t: "p",
        text: "$\\mathrm{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$ with $W_1\\in\\mathbb{R}^{d\\times d_{ff}}$, $W_2\\in\\mathbb{R}^{d_{ff}\\times d}$ and $d_{ff} = 4d$ (2048 for $d=512$). The same weights are used at every position (like two 1×1 convolutions). It widens each vector, applies a non-linearity, and shrinks it back. With fixed weights, attention is a weighted average of linear projections; the FFN does most per-token non-linear work and holds **two thirds of each layer's parameters** (computed below). Modern models use GELU or gated versions (SwiGLU) instead of ReLU.",
      },
      { t: "h", text: "The residual stream view" },
      {
        t: "p",
        text: "In a pre-LN model, unroll the additions: $x_L = x_0 + \\sum_{\\ell}\\big(\\mathrm{MHA}_\\ell(\\cdot) + \\mathrm{FFN}_\\ell(\\cdot)\\big)$. The stream is a $d$-wide channel that starts as the token+position embedding; every sublayer *reads* it (through a LayerNorm) and *writes* an addition into it. Gradients flow from the loss straight down this identity path to every layer, avoiding the depth problem from [residual connections](/learn/m07/residual-connections). The simulator shows the norms of the stream and of the attention and FFN updates — the updates are usually smaller than the stream.",
      },
      { t: "h", text: "Post-LN (original) vs pre-LN (modern)" },
      {
        t: "p",
        text: "The paper puts LayerNorm **after** the residual add: $x \\leftarrow \\mathrm{LN}(x + F(x))$. Nothing skips the norm, so the gradient to early layers passes through every LayerNorm. Xiong et al. (2020, “On Layer Normalization in the Transformer Architecture”) showed post-LN has large gradients near the output at the start of training, so it needs a learning-rate **warm-up** (a slow start) to train stably.",
      },
      {
        t: "p",
        text: "**Pre-LN** puts the norm on the sublayer *input*, $x \\leftarrow x + F(\\mathrm{LN}(x))$, leaving a clean identity path. It trains stably with little or no warm-up and is the default in GPT-2-style and most later models. Because the stream itself is never normalized, pre-LN needs one **extra final LayerNorm** before the output head. In PyTorch: `norm_first=False` (post-LN, default) or `norm_first=True` (pre-LN).",
      },
      { t: "h", text: "Output projection, softmax and weight tying" },
      {
        t: "p",
        text: "At each position, the final hidden vector $h\\in\\mathbb{R}^d$ becomes $V$ logits via $z = hW_{\\text{out}}$ with $W_{\\text{out}}\\in\\mathbb{R}^{d\\times V}$, and $p = \\mathrm{softmax}(z)$. **Weight tying** sets $W_{\\text{out}} = E^\\top$: token $t$'s logit is the dot product of $h$ with token $t$'s own input embedding. The paper shares one matrix between source embedding, target embedding and output projection (with a shared sub-word vocabulary), following Press & Wolf (2017, “Using the Output Embedding to Improve Language Models”). Tying saves $Vd$ parameters — for $V = 37{,}000$, $d = 512$, 18.9 M, as many as the whole 6-layer encoder.",
      },
      { t: "h", text: "Decoder-only simplification" },
      {
        t: "p",
        text: "Remove the encoder and the cross-attention sublayer. What is left is a stack of blocks with masked self-attention + FFN — an encoder block with a causal mask. Trained to predict the next token of ordinary text, this **decoder-only** model is the GPT family; the prompt takes the place of the source sentence. Encoder-only models (BERT) keep the encoder with two-way attention; encoder–decoder models (T5, the original) keep both. [BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families) compares them. The simulator's trained model is decoder-only.",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Notation" },
      {
        t: "p",
        text: String.raw`$X\in\mathbb{R}^{n\times d}$: the $n$ token vectors entering a block (one row per position), $d = d_{\text{model}}$. $h$: number of heads, $d_k = d/h$. $d_{ff}$: FFN hidden width. $M\in\mathbb{R}^{n\times d}$: encoder output (memory). $Y\in\mathbb{R}^{m\times d}$: decoder stream for $m$ target positions. $V$: vocabulary size. $N$: number of layers. $\mathrm{LN}$: LayerNorm on each row, $\mathrm{LN}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$, where $\mu,\sigma^2$ are the mean and (biased) variance of the row's $d$ entries.`,
      },
      { t: "h", text: "Block equations" },
      { t: "math", tex: String.raw`\mathrm{MHA}(X_q, X_{kv}) = \mathrm{Concat}(H_1,\dots,H_h)\,W_O + b_O,\quad H_j = \mathrm{softmax}\!\Big(\tfrac{(X_qW_Q^{j})(X_{kv}W_K^{j})^\top}{\sqrt{d_k}} + \mathcal{M}\Big)X_{kv}W_V^{j}`, caption: "Multi-head attention (projection biases left out to keep it short). $\\mathcal{M}$ is 0 for allowed pairs and $-\\infty$ for masked ones." },
      { t: "math", tex: String.raw`\mathrm{FFN}(x) = \phi(xW_1 + b_1)\,W_2 + b_2,\qquad W_1\in\mathbb{R}^{d\times d_{ff}},\; W_2\in\mathbb{R}^{d_{ff}\times d},\;\phi=\mathrm{ReLU}\text{ or GELU}` },
      { t: "math", tex: String.raw`\textbf{Encoder (post-LN):}\quad A = \mathrm{LN}_1\big(X + \mathrm{MHA}(X, X)\big),\qquad \mathrm{Enc}(X) = \mathrm{LN}_2\big(A + \mathrm{FFN}(A)\big)` },
      { t: "math", tex: String.raw`\textbf{Encoder (pre-LN):}\quad A = X + \mathrm{MHA}\big(\mathrm{LN}_1(X), \mathrm{LN}_1(X)\big),\qquad \mathrm{Enc}(X) = A + \mathrm{FFN}\big(\mathrm{LN}_2(A)\big)` },
      { t: "math", tex: String.raw`\textbf{Decoder (post-LN):}\; B = \mathrm{LN}_1\big(Y + \mathrm{MHA}_{\text{causal}}(Y,Y)\big),\; C = \mathrm{LN}_2\big(B + \mathrm{MHA}(B, M)\big),\; \mathrm{Dec}(Y,M) = \mathrm{LN}_3\big(C + \mathrm{FFN}(C)\big)` },
      { t: "math", tex: String.raw`p(y_{t+1}\mid y_{\le t}, x) = \mathrm{softmax}\big(h_t E^\top\big),\qquad \mathcal{L} = -\frac{1}{m}\sum_{t=1}^{m}\log p(y_{t}^{\ast}\mid y_{<t}, x)`, caption: "Weight-tied output layer, and the teacher-forced cross-entropy loss averaged over all $m$ target positions." },
      { t: "h", text: "Parameter count per layer, derived" },
      {
        t: "steps",
        items: [
          { title: "Multi-head attention", text: "$W_Q, W_K, W_V, W_O$ are each $d\\times d$ (all heads together), plus four bias vectors of length $d$. The number of heads does not matter.", tex: String.raw`P_{\text{MHA}} = 4d^2 + 4d` },
          { title: "Feed-forward network", text: "$W_1$ ($d\\times d_{ff}$), $b_1$ ($d_{ff}$), $W_2$ ($d_{ff}\\times d$), $b_2$ ($d$).", tex: String.raw`P_{\text{FFN}} = 2d\,d_{ff} + d_{ff} + d \;\overset{d_{ff}=4d}{=}\; 8d^2 + 5d` },
          { title: "LayerNorms", text: "Each has $\\gamma$ and $\\beta$ of length $d$.", tex: String.raw`P_{\text{LN}} = 2d` },
          { title: "Encoder layer (two LNs)", tex: String.raw`P_{\text{enc}} = (4d^2+4d) + (8d^2+5d) + 4d = 12d^2 + 13d` },
          { title: "Decoder layer (extra cross-attention + LN)", tex: String.raw`P_{\text{dec}} = 2(4d^2+4d) + (8d^2+5d) + 6d = 16d^2 + 19d` },
          { title: "FFN share", text: "For large $d$ the FFN holds $8d^2/12d^2 = 2/3$ of each encoder layer's parameters, attention $1/3$. Most of a Transformer's weights live in its feed-forward networks." },
        ],
      },
      { t: "h", text: "Worked example 1: the paper's base model" },
      {
        t: "p",
        text: String.raw`Base setup: $d=512$, $h=8$, $d_k=64$, $d_{ff}=2048$, $N=6$ encoder and 6 decoder layers. Encoder layer: $12(512^2) + 13(512) = 3{,}145{,}728 + 6{,}656 = 3{,}152{,}384$. Decoder layer: $16(512^2)+19(512) = 4{,}194{,}304 + 9{,}728 = 4{,}204{,}032$. FFN share of an encoder layer: $(2\cdot512\cdot2048+2048+512)/3{,}152{,}384 = 2{,}099{,}712/3{,}152{,}384 = 66.6\%$. One shared embedding matrix for a 37,000-token vocabulary: $37{,}000\times512 = 18{,}944{,}000$. Total $= 18{,}944{,}000 + 6(3{,}152{,}384) + 6(4{,}204{,}032) = 18{,}944{,}000 + 18{,}914{,}304 + 25{,}224{,}192 = 63{,}082{,}496 \approx 63$ M. The paper reports about 65 M; the exact number depends on vocabulary size and details such as final normalization layers. The program below checks the per-layer formulas against PyTorch's own modules.`,
      },
      { t: "h", text: "FLOPs per token" },
      {
        t: "p",
        text: String.raw`Multiplying a vector by a $d_{in}\times d_{out}$ weight matrix costs $d_{in}d_{out}$ multiply-adds $= 2d_{in}d_{out}$ FLOPs. So every weight-matrix entry costs **2 FLOPs per token** in the forward pass, and a model with $N_p$ (non-embedding) matrix parameters needs about $2N_p$ FLOPs per token. Attention adds a term that grows with context length $n$ and has no parameters. Per layer and token, the scores $q_i\cdot k_j$ for $n$ keys cost $2nd$ FLOPs (over all heads), and the weighted sum of $n$ value vectors another $2nd$:`,
      },
      { t: "math", tex: String.raw`C_{\text{forward}} \approx 2N_p + 2\,d\,V_{\text{head}} + \underbrace{4\,n\,d\,N}_{\text{attention}} \quad\text{FLOPs per token},\qquad C_{\text{train}} \approx 3\,C_{\text{forward}} \approx 6N_p`, caption: "$2dV$ is the output projection (counted even when tied). The backward pass costs about 2× the forward, so training ≈ 6 FLOPs per parameter per token — the estimate Kaplan et al. (2020) also use. With causal masking only about half the scores are needed, but dense code computes them all." },
      {
        t: "p",
        text: String.raw`**Worked example 2** (base encoder layer, $n=128$): matrix weights $4(512^2) + 2(512)(2048) = 1{,}048{,}576 + 2{,}097{,}152 = 3{,}145{,}728$, so $2\times = 6{,}291{,}456$; attention $4\cdot128\cdot512 = 262{,}144$; total $6{,}553{,}600$ FLOPs per token — exactly what PyTorch's FLOP counter measures below. At $n=128$ attention is 4% of the work; at $n=4096$ it is 57%. The crossover ($4nd$ vs $24d^2$, i.e. $n \approx 6d$) is why long contexts are expensive.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example 3** (the simulator's trained model: $d=48$, $N=2$, $d_{ff}=192$, $V=37$, context 64, learned positions, tied head). Per layer $12(48^2)+13(48) = 27{,}648 + 624 = 28{,}272$; two layers $56{,}544$; token embeddings $37\times48 = 1{,}776$; position table $64\times48 = 3{,}072$; final LN $96$; total $\mathbf{61{,}488}$ — the number the simulator computes from the downloaded weights. FLOPs per token at full context: $2\cdot2\,(4\cdot48^2 + 2\cdot48\cdot192) = 2\cdot55{,}296 = 110{,}592$, plus head $2\cdot48\cdot37 = 3{,}552$, plus attention $4\cdot64\cdot48\cdot2 = 24{,}576$: $138{,}720$ FLOPs. Attention is already 18% at $n=64$ because $d$ is so small ($n > 6d$ would make it dominant).`,
      },
      { t: "code", lang: "python", title: "params_and_flops.py", code: PARAMS_FLOPS, output: PARAMS_FLOPS_OUT },
      { t: "h", text: "Shape-by-shape data flow (base model, translation)" },
      {
        t: "p",
        text: "A concrete case: batch $B=2$, source length $n=10$, target length $m=12$, $d=512$, $h=8$, $d_k=64$, $d_{ff}=2048$, $V=37{,}000$. PyTorch `batch_first=True` layout (batch, sequence, features).",
      },
      {
        t: "table",
        head: ["Stage", "Operation", "Output shape"],
        rows: [
          ["Source ids", "tokenizer", "(2, 10)"],
          ["Source input", "$\\sqrt{512}\\,E[\\text{ids}] + PE$", "(2, 10, 512)"],
          ["Enc: Q, K, V", "$XW_Q$ etc., split into heads", "(2, 8, 10, 64) each"],
          ["Enc: scores, weights", "$QK^\\top/\\sqrt{64}$, softmax over last axis", "(2, 8, 10, 10)"],
          ["Enc: heads", "weights · V", "(2, 8, 10, 64)"],
          ["Enc: concat, $W_O$", "reshape, $\\cdot W_O$, add & LN", "(2, 10, 512)"],
          ["Enc: FFN hidden", "$\\mathrm{ReLU}(xW_1+b_1)$", "(2, 10, 2048)"],
          ["Enc: FFN out", "$\\cdot W_2 + b_2$, add & LN", "(2, 10, 512)"],
          ["Memory (after 6 layers)", "—", "(2, 10, 512)"],
          ["Target ids (shifted right)", "BOS + target[:-1]", "(2, 12)"],
          ["Target input", "$\\sqrt{512}\\,E[\\text{ids}] + PE$", "(2, 12, 512)"],
          ["Dec: masked self-attn scores", "causal mask, $j>i$ set to $-\\infty$", "(2, 8, 12, 12)"],
          ["Dec: cross-attn Q / K,V", "Q from decoder; K, V from memory", "(2, 8, 12, 64) / (2, 8, 10, 64)"],
          ["Dec: cross-attn scores", "$QK^\\top/\\sqrt{64}$ (target × source)", "(2, 8, 12, 10)"],
          ["Dec: after cross-attn", "concat, $W_O$, add & LN", "(2, 12, 512)"],
          ["Dec: FFN hidden → out", "", "(2, 12, 2048) → (2, 12, 512)"],
          ["Decoder output (after 6 layers)", "—", "(2, 12, 512)"],
          ["Logits", "$\\cdot E^\\top$ (tied)", "(2, 12, 37000)"],
          ["Probabilities", "softmax over last axis", "(2, 12, 37000)"],
        ],
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "p",
        text: "This is a **real trained Transformer**, not random weights: decoder-only, pre-LN, 2 layers, $d_{\\text{model}}=48$, 4 heads of 12, $d_{ff}=192$, learned positions, GELU and a weight-tied output — 61,488 parameters. It was trained in PyTorch (`web/scripts/train-char-transformer.py`, 3000 steps, about 2 minutes on a CPU) to predict the next character of the opening of Chapter I of Lewis Carroll's *Alice's Adventures in Wonderland* (public domain, 6,759 training characters). The browser downloads the weights (~310 KB) and runs the forward pass in TypeScript; its logits match PyTorch's to about $5\\times10^{-6}$ (see the side panel).",
      },
      { t: "sim", id: "transformer", caption: "Step through the forward pass for your prompt; then press Play in the Generate box to let the model write, one character and one full forward pass at a time." },
      {
        t: "list",
        items: [
          "**Tokens → Embed**: each character is an id; layer 1's input is a row of the embedding table plus a row of the position table. Click any character to select its position — every later view follows it.",
          "**L1·Attn, head by head**: open layer 1's attention and click through the four heads. In a tiny model trained for two minutes the patterns are soft, but they differ. On the prompt `alice was beginning to get very tired of sitting by her sister`, layer-1 heads look mostly at the last few characters (average look-back 3–5; head 2 puts about a third of its weight on the character just before), while layer-2 heads reach further (average look-back 7–8), building on layer 1's local features. Click heatmap rows to see which characters that query reads. Everything above the diagonal is exactly 0 — the causal mask.",
          "**Residual add**: compare the bar heights. The attention and FFN updates are added to the stream instead of replacing it — the residual stream in action.",
          "**FFN**: the hidden layer is 4× wider than the stream. Note what fraction of hidden units have a positive pre-activation at the selected position.",
          "**Logits**: with the prompt `alice was beginning to get ver` the model gives `y` about 74% probability. Pick an earlier position to see the probability given to the character that really came next, and the cross-entropy loss there — this is how all positions train in parallel.",
          "**Generate**: greedy decoding reproduces chunks of the chapter, because the model has mostly memorised its 6.8k training characters (training loss 0.53 vs held-out loss 2.51 nats per character). Switch to *sample* and move the temperature: low $T$ repeats itself, $T\\approx1.5$ gives misspelled, word-like gibberish.",
        ],
      },
      {
        t: "p",
        text: "To see one attention layer with every intermediate matrix printed number by number (random weights), use the attention explorer from the previous module:",
      },
      { t: "sim", id: "attention", caption: "One multi-head attention layer on whole words, with random weights — every Q, K, V and score shown as numbers." },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "A full Transformer encoder block in NumPy: multi-head attention (head split, scaling, optional causal mask, concatenation, output projection), the ReLU FFN, LayerNorm, and both post-LN and pre-LN wiring. Every weight is copied from `torch.nn.TransformerEncoderLayer` and the outputs are compared. It also checks the $12d^2+13d$-style parameter formula and prints a causal attention matrix.",
      },
      { t: "code", lang: "python", title: "transformer_block_numpy.py", code: BLOCK_NUMPY, output: BLOCK_NUMPY_OUT },
      {
        t: "list",
        items: [
          "Agreement to ~$10^{-15}$ in float64 means the NumPy code computes **exactly** the same function as PyTorch, for both norm placements, with and without the mask.",
          "**Weight layout**: `nn.MultiheadAttention` stacks $W_Q, W_K, W_V$ in one `in_proj_weight` of shape $(3d, d)$, and every `nn.Linear` stores weights as (out, in) and computes $xW^\\top + b$. That is why the code splits the stacked matrix in three and multiplies by `.T`.",
          "**`batch_first`**: by default PyTorch Transformer modules expect (sequence, batch, features). The code passes `batch_first=True` so tensors are (batch, sequence, features) like the NumPy arrays. Forget it and the batch and sequence axes get silently mixed up when they have the same size.",
          "**`norm_first`**: `False` (default) is the paper's post-LN; `True` is pre-LN.",
          "**Mask convention**: in a boolean `src_mask`, `True` means *blocked*; NumPy's `np.where(mask, -inf, scores)` uses the same rule. (A float mask is instead *added* to the scores.)",
          "The two LayerNorm parameter vectors are randomized because at initialization $\\gamma=1, \\beta=0$, which would hide a bug in how they are used.",
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
              {
                t: "p",
                text: "**An encoder–decoder model from `nn.TransformerEncoder` and `nn.TransformerDecoder`.** A shared, weight-tied embedding, sinusoidal positions, 2 pre-LN layers per stack, and a toy translation task (reverse an 8-digit sequence), so the whole data flow — encode once, decode with masked self-attention + cross-attention, greedy step-by-step inference — trains and is checked in seconds.",
              },
              { t: "code", lang: "python", title: "encoder_decoder_pytorch.py", code: ENCDEC_TORCH, output: ENCDEC_TORCH_OUT },
              {
                t: "list",
                items: [
                  "The parameter counts match the formulas with $d=64$, $d_{ff}=128$: encoder layer $4(64^2)+4(64) + 2(64)(128)+128+64 + 4(64) = 33{,}472$; two layers plus a final LayerNorm (128) $= 67{,}072$. Decoder layer $2(16{,}640)+16{,}576+384 = 50{,}240$; two layers plus final LN $= 100{,}608$.",
                  "`generate_square_subsequent_mask` returns a float mask with $-\\infty$ above the diagonal; it is passed as `tgt_mask`. Cross-attention needs no mask here (no padding).",
                  "Reversal needs every output position to look at a *different* source position, so 100% exact-match accuracy shows cross-attention learned the alignment. A model without cross-attention could not solve it.",
                  "`enable_nested_tensor=False` only silences a warning about an inference shortcut that does not apply to pre-LN layers.",
                ],
              },
              {
                t: "p",
                text: "**Loading the trained character model and generating text.** This rebuilds the simulator's model from standard modules (`nn.MultiheadAttention`, `nn.LayerNorm`, `nn.Linear`) using the JSON file the browser downloads, checks its logits against the PyTorch reference in the file, and generates text greedily and by sampling. Run it from the project's `web/` folder.",
              },
              { t: "code", lang: "python", title: "load_char_model.py (run from web/; needs public/models/char-transformer.json — real output, not executed in CI)", code: LOAD_CHAR, output: LOAD_CHAR_OUT },
              {
                t: "callout",
                kind: "note",
                title: "What the output shows",
                text: "The greedy continuation stitches together real phrases from the chapter (“there was not a moment to be lost”, “she looked up, but it was”) — the model has memorised much of its tiny training text. Higher temperature flattens the distribution and soon gives non-words. A 61k-parameter model trained on 7k characters is a microscope slide, not a language model — but the architecture is exactly the one used at scale.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras gives you `layers.MultiHeadAttention` (with `key_dim` = per-head size) and `layers.LayerNormalization`; you build the block by hand. Careful: Keras masks use the **opposite** rule to PyTorch boolean masks (`True` = may attend), and `use_causal_mask=True` builds the causal mask for you.",
              },
              { t: "code", lang: "python", title: "encoder_block_keras.py (not executed in CI)", code: KERAS_BLOCK, output: "Prints the output shape (2, 10, 512) and the parameter count of the block, 3152384 — the same as nn.TransformerEncoderLayer(512, 8, 2048), since Keras's per-head kernels (512 × 8 × 64) hold the same number of weights as PyTorch's 512 × 512 projections." },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Design choice", "Options", "Advantages", "Disadvantages / when used"],
        rows: [
          ["Normalization placement", "Post-LN (original) · Pre-LN · Pre-LN with RMSNorm", "Post-LN can reach slightly better final quality when it trains; pre-LN is stable without long warm-up; RMSNorm (Zhang & Sennrich, 2019) drops the mean subtraction and $\\beta$, a bit cheaper", "Post-LN needs warm-up and a careful learning rate; pre-LN needs a final LN and its stream norm grows with depth. Most current LLMs: pre-norm"],
          ["Overall structure", "Encoder–decoder · encoder-only · decoder-only", "Enc–dec: natural for input→output tasks (translation, speech-to-text); encoder-only: two-way context for understanding tasks; decoder-only: one model for any text task via prompting", "See [Transformer families](/learn/m11/transformer-families)"],
          ["FFN activation", "ReLU · GELU · gated (GEGLU, SwiGLU)", "Gated versions (Shazeer, 2020, “GLU Variants Improve Transformer”) improve quality at equal compute", "Gated FFNs have three matrices; $d_{ff}$ is usually cut (≈ 8d/3) to keep the parameter count"],
          ["Positions", "Sinusoidal · learned absolute · relative · RoPE · ALiBi", "Learned: simple (used in the simulator model); RoPE/relative: encode distances, extrapolate better", "Learned absolute positions cannot go past the trained context length; see [positional encoding](/learn/m10/positional-encoding)"],
          ["Attention heads for K/V", "Multi-head · multi-query (MQA) · grouped-query (GQA)", "MQA/GQA share K and V across heads, shrinking the inference KV cache", "Small quality cost; GQA is the usual middle ground"],
          ["Output head", "Tied to embedding · untied", "Tying saves $Vd$ parameters and regularizes small models", "Very large models sometimes untie for flexibility"],
          ["Attention span", "Dense · local/sliding window · sparse", "Less than quadratic cost for long sequences", "Limits which positions can interact directly"],
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Machine translation** — the original use: the paper set new best BLEU scores on WMT 2014 English→German and English→French at a fraction of earlier models' training cost. Encoder–decoder Transformers are still the standard for dedicated translation systems.",
          "**Large language models** — GPT-style decoder-only Transformers trained on next-token prediction power chat assistants, code assistants and text generation. This lesson's block, scaled up (dozens of layers, $d$ in the thousands), is their core.",
          "**Search and text understanding** — BERT-family encoder-only models produce context-aware embeddings for ranking, classification and retrieval.",
          "**Speech** — Whisper (OpenAI) is an encoder–decoder Transformer: the encoder reads audio features, the decoder writes the transcript, linked by cross-attention.",
          "**Vision** — Vision Transformers cut an image into patches and treat them as tokens; the encoder block is unchanged.",
          "**Multimodal and scientific models** — the same blocks handle mixes of text and image tokens, and attention-based models are used for protein structure prediction.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Parallel training**: all positions are computed at once with big matrix multiplications, using hardware far better than RNNs.",
          "**One-step path** between any two positions, so long-range links can be learned directly.",
          "**Uniform, scalable design**: identical blocks; quality improves predictably with more parameters, data and compute.",
          "**Quadratic attention cost** in sequence length: $O(n^2 d)$ compute and $O(n^2)$ score memory per head per layer. FlashAttention-style kernels avoid storing the matrix, but not the quadratic compute.",
          "**Sequential inference**: generation is still one token at a time; each step re-reads the whole context (or a KV cache, whose memory grows linearly with context length).",
          "**No built-in sense of order or locality**: positions must be added, and assumptions CNNs/RNNs get for free must be learned, so small-scale Transformers need lots of data.",
          "**Fixed context window**: anything outside it is invisible (the simulator model sees only the last 64 characters).",
          "**Sensitive training**: post-LN needs warm-up; large models need careful initialization, learning-rate schedules and mixed-precision handling.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting the causal mask in a decoder.** Training loss gets suspiciously low because each position can read the very token it should predict; generation then fails completely. Always mask $j>i$ in decoder self-attention (and check that row $i$ of the weights is zero after column $i$).",
          "**Flipping the mask rule.** PyTorch boolean masks: `True` = blocked. Keras: `True` = allowed. Float masks are *added* ($0$ or $-\\infty$). Mix them up and you block the past and expose the future.",
          "**Wrong axis layout.** PyTorch Transformer modules default to (sequence, batch, features). Without `batch_first=True`, a (batch, seq, d) tensor is read with the axes swapped — and if batch size equals sequence length, there is no error at all.",
          "**Not shifting the decoder input.** The decoder input must be BOS + target[:-1] and the label the target (+EOS). Feeding the unshifted target with the same labels teaches the model to copy.",
          "**Pre-LN without the final LayerNorm.** A pre-LN model's residual stream is never normalized; leave out the final LN and the logits' scale drifts with depth.",
          "**Softmax over the wrong axis** of the scores (queries instead of keys). Each *row* (one query) must add up to 1.",
          "**Forgetting `model.eval()`** at inference: dropout stays on and outputs become random. (This exact bug came up while building the simulator's test vectors.)",
          "**Expecting attention to see order by itself.** Without position information the block is permutation-equivariant: shuffling the input tokens just shuffles the outputs.",
          "**Tying weights with a mismatched embedding scale.** The paper multiplies embeddings by $\\sqrt{d}$. If the shared matrix starts with unit-variance entries, the tied logits start huge and training stalls. Initialize it with standard deviation about $d^{-1/2}$ (as in the encoder–decoder example).",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What are the two sublayers of a Transformer encoder block and what does each do?", a: "Multi-head self-attention, where every position gathers information from all positions (the only place positions communicate), and a position-wise feed-forward network that transforms each position's vector on its own. Each is wrapped in a residual connection and LayerNorm." },
      { t: "qa", level: "beginner", q: "How does a decoder block differ from an encoder block?", a: "Its self-attention is masked (causal), so position $i$ only sees positions $\\le i$. It also has an extra cross-attention sublayer: queries come from the decoder, keys and values from the encoder output. That makes three sublayers instead of two." },
      { t: "qa", level: "beginner", q: "Why does a Transformer need positional encodings?", a: "Self-attention and the FFN treat the input as a set: shuffling the input tokens shuffles the outputs the same way, so word order would be invisible. Adding a position-dependent vector to each embedding breaks this symmetry." },
      { t: "qa", level: "beginner", q: "Why did Transformers replace RNNs for most sequence tasks?", a: "RNNs need $n$ steps in a row per layer and an $O(n)$ path between distant tokens. Self-attention handles all positions in parallel and links any two positions in one step. That trains much faster on GPUs and captures long-range links more easily, at the cost of $O(n^2)$ attention compute." },
      { t: "qa", level: "intermediate", q: "Derive the parameter count of an encoder layer with $d_{ff}=4d$.", a: "Attention: $W_Q,W_K,W_V,W_O\\in\\mathbb{R}^{d\\times d}$ plus 4 biases: $4d^2+4d$. FFN: $d\\cdot4d + 4d + 4d\\cdot d + d = 8d^2+5d$. Two LayerNorms: $4d$. Total $12d^2+13d$; for $d=512$ that is 3,152,384, matching `nn.TransformerEncoderLayer(512, 8, 2048)`." },
      { t: "qa", level: "intermediate", q: "Compare post-LN and pre-LN.", a: "Post-LN: $\\mathrm{LN}(x+F(x))$, as in the original paper. Every gradient path passes through the LayerNorms, gradients near the output are large at the start, and the model needs learning-rate warm-up. Pre-LN: $x+F(\\mathrm{LN}(x))$. The residual path is a pure identity and training is stable with less warm-up, but a final LN is required and the stream's norm grows with depth. Most modern LLMs use pre-norm (often RMSNorm)." },
      { t: "qa", level: "intermediate", q: "What is weight tying and why use it?", a: "Using the transposed input embedding matrix $E^\\top$ as the output projection, so logit$_t = h\\cdot e_t$. It removes $Vd$ parameters (often a big share of a small model) and reuses the meaning structure learned by the embeddings; Press & Wolf (2017) showed it improves language models. The Transformer paper also shares the matrix between encoder and decoder embeddings." },
      { t: "qa", level: "intermediate", q: "Where do most of a Transformer's FLOPs go?", a: "For context length $n$ well below $6d$, into the weight matrices: about $2$ FLOPs per parameter per token, two thirds of it in the FFN. The attention-score term $4nd$ per layer grows with context and dominates when $n \\gtrsim 6d$ (for $d=512$: 57% of an encoder layer's work at $n=4096$)." },
      { t: "qa", level: "advanced", q: "Why is training a decoder-only Transformer on a sequence of length $n$ equivalent to $n$ next-token prediction problems solved in one pass?", a: "With the causal mask, the output at position $i$ depends only on tokens $1..i$: weights to $j>i$ are exactly zero and everything else is per position. So the logits at $i$ are exactly what an autoregressive model would compute to predict token $i+1$ from that prefix. Summing the cross-entropy over positions trains all $n$ prefix problems at once (teacher forcing) — parallel training, sequential inference." },
      { t: "qa", level: "advanced", q: "Estimate the forward FLOPs per token of GPT-2 small ($d=768$, 12 layers, $d_{ff}=3072$, $V=50{,}257$, context 1024).", a: "Matrix weights per layer $4d^2 + 2d\\cdot4d = 12d^2 = 7{,}077{,}888$; ×12 = 84,934,656; ×2 = 169.9 M FLOPs. Output head $2dV = 77.2$ M. Attention at full context $4\\cdot1024\\cdot768\\cdot12 = 37.7$ M. Total ≈ 284.8 M FLOPs ≈ 0.28 GFLOPs per token; training ≈ 3× that." },
      { t: "qa", level: "advanced", q: "In cross-attention, what are the shapes of the score matrix and why is no causal mask applied?", a: "Queries come from $m$ target positions and keys from $n$ source positions, so the scores are $m\\times n$ per head. The whole source sentence is available before decoding starts, so every target position may look at every source position; only source padding needs masking (a key-padding mask)." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "How many parameters does one encoder layer with $d=256$, $d_{ff}=1024$ have (with biases and two LayerNorms)?", answer: 789760, tolerance: 0.5, explain: "$12d^2 + 13d = 12(65{,}536) + 13(256) = 786{,}432 + 3{,}328 = 789{,}760$." },
          { q: "Approximately what fraction of an encoder layer's parameters is in the feed-forward network when $d_{ff}=4d$?", options: ["About 1/4", "About 1/3", "About 2/3", "About 9/10"], answer: 2, explain: "FFN $\\approx 8d^2$ out of $\\approx 12d^2$ in total, i.e. 2/3. Attention holds the other 1/3." },
          { q: "In the decoder's cross-attention sublayer, where do the queries, keys and values come from?", options: ["All from the decoder", "All from the encoder output", "Queries from the decoder, keys and values from the encoder output", "Queries from the encoder output, keys and values from the decoder"], answer: 2, explain: "The decoder asks the questions (queries) and reads the source (keys and values from the memory)." },
          { q: "Which wiring is pre-LN?", options: ["$x \\leftarrow \\mathrm{LN}(x + F(x))$", "$x \\leftarrow x + F(\\mathrm{LN}(x))$", "$x \\leftarrow \\mathrm{LN}(F(x))$", "$x \\leftarrow F(x) + \\mathrm{LN}(x)$"], answer: 1, explain: "Pre-LN normalizes the sublayer's input and leaves the residual path untouched." },
          { kind: "numeric", q: "A decoder self-attention layer processes 5 positions with a causal mask. How many of the 25 attention weights (per head) can be non-zero?", answer: 15, tolerance: 0.5, explain: "Row $i$ (counting from 1) has $i$ allowed keys: $1+2+3+4+5 = 15$. The other 10 are exactly zero." },
          { kind: "numeric", q: "Using $C \\approx 2N_p$, how many GFLOPs does a forward pass cost per token for a model with $10^9$ non-embedding parameters (ignore the attention term)?", answer: 2, tolerance: 0.01, explain: "2 FLOPs per parameter per token: $2\\times10^9$ FLOPs = 2 GFLOPs. Training costs about three times as much (≈ 6 GFLOPs)." },
          { q: "With weight tying, the logit for vocabulary token $t$ at a position with final hidden state $h$ equals…", options: ["$h\\cdot e_t$, the dot product with token $t$'s input embedding", "the softmax probability of $t$", "the attention weight on token $t$", "$\\|h - e_t\\|$"], answer: 0, explain: "The output matrix is $E^\\top$, so $z_t = h\\cdot e_t$." },
          { q: "Why can a Transformer decoder be trained on all positions of a target sentence in parallel?", options: ["Because it has no recurrent state and the causal mask prevents any position from seeing later tokens", "Because the FFN is position-wise", "Because LayerNorm is applied per token", "It cannot; it must be trained token by token"], answer: 0, explain: "Teacher forcing feeds the true prefix, and the mask makes each position's output depend only on its prefix — so all next-token problems are computed in one pass." },
        ],
      },
      {
        t: "exercise",
        title: "Count GPT-2 small",
        prompt: "A decoder-only model has $d=768$, 12 layers, $d_{ff}=3072$, a 50,257-token vocabulary, a learned position table for 1024 positions, a weight-tied output head and one final LayerNorm. Decoder-only blocks have no cross-attention, so each block has the encoder-layer count. Work out the total number of parameters.",
        hint: "Per block $12d^2+13d$; add the token embedding, the position table and the final LN ($2d$).",
        solution: [
          {
            t: "steps",
            items: [
              { title: "One block", tex: String.raw`12(768^2) + 13(768) = 7{,}077{,}888 + 9{,}984 = 7{,}087{,}872` },
              { title: "12 blocks", tex: String.raw`12 \times 7{,}087{,}872 = 85{,}054{,}464` },
              { title: "Embeddings", tex: String.raw`50{,}257\times768 = 38{,}597{,}376,\qquad 1024\times768 = 786{,}432` },
              { title: "Total", tex: String.raw`85{,}054{,}464 + 38{,}597{,}376 + 786{,}432 + 1{,}536 = 124{,}439{,}808 \approx 124\text{ M}` },
            ],
          },
          { t: "p", text: "This is the well-known “124M” of GPT-2 small. Without weight tying, a separate $768\\times50{,}257$ output matrix would add another 38.6 M parameters." },
        ],
      },
      {
        t: "exercise",
        title: "Implement a decoder block in NumPy",
        prompt: "Extend the NumPy encoder block to a **post-LN decoder block**: masked self-attention, then cross-attention whose keys and values come from an encoder memory of a different length, then the FFN — each followed by add & LayerNorm. Check it against `nn.TransformerDecoderLayer` (`norm1`, `norm2`, `norm3`; the cross-attention module is called `multihead_attn`) and check the parameter formula $2(4d^2+4d) + (2d\\,d_{ff} + d_{ff} + d) + 6d$.",
        hint: "Change the attention function so it takes separate inputs for queries and for keys/values.",
        solution: [
          { t: "code", lang: "python", title: "decoder_block_numpy.py", code: DECODER_EX, output: DECODER_EX_OUT },
          { t: "p", text: "The only new idea is `mha(x_q, x_kv, ...)`: with $T_q=5$ target positions and $T_k=7$ memory positions the score tensor is (2, 4, 5, 7). The causal mask is used only in the self-attention sublayer." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Transformer = embeddings + positions → N blocks of [attention, FFN], each with residual + LayerNorm → linear (often tied) → softmax.",
          "Self-attention: $O(1)$ sequential steps and $O(1)$ path length, $O(n^2d)$ compute; RNN: $O(n)$ and $O(n)$, $O(nd^2)$.",
          "Encoder block: self-attention + FFN. Decoder block: **masked** self-attention + **cross-attention** (Q from decoder, K/V from encoder memory) + FFN.",
          String.raw`FFN: $\phi(xW_1+b_1)W_2+b_2$, $d_{ff}=4d$, applied to each position on its own; holds ≈ 2/3 of the layer's parameters.`,
          String.raw`Parameters: $P_{\text{enc}} = 12d^2+13d$, $P_{\text{dec}} = 16d^2+19d$ (for $d_{ff}=4d$); base model: 3,152,384 per encoder layer.`,
          String.raw`FLOPs per token: $\approx 2N_p + 4ndN$ forward; training $\approx 6N_p$. Attention dominates once $n \gtrsim 6d$.`,
          "Post-LN $\\mathrm{LN}(x+F(x))$ (original, needs warm-up) vs pre-LN $x+F(\\mathrm{LN}(x))$ (modern, stable, needs a final LN).",
          "Residual stream: the representation is a running sum of the embedding and every sublayer's update.",
          "Weight tying: $W_{\\text{out}} = E^\\top$ saves $Vd$ parameters.",
          "Decoder-only = decoder blocks without cross-attention, trained on next-token prediction with teacher forcing — the GPT recipe, and the simulator's model.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[Tokenization, Embeddings and Context Windows](/learn/m11/tokens-embeddings) — how text becomes the ids and vectors at the bottom of the stack, and what limits the context window.",
          "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — teacher forcing, decoding strategies (greedy, temperature, top-k/top-p), KV caching and the $O(n^2)$ cost in practice.",
          "[BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families) — what changes when you keep only the encoder, only the decoder, or both, and how the same block handles images.",
          "Revisit [Multi-Head Attention](/learn/m10/multi-head-attention) to see what single heads learn, now that you can inspect real trained heads in the simulator.",
        ],
      },
    ],
  },
};

export default lesson;
