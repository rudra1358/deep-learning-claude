import type { Lesson } from "@/lib/types";

const WORKED = String.raw`import numpy as np

E = np.array([[0.1, 0.2, 0.3],      # row 0
              [1.0, 0.0, -1.0],     # row 1
              [0.5, -0.5, 2.0],     # row 2
              [-0.2, 0.4, 0.0]])    # row 3   (V = 4 tokens, d = 3)
ids = np.array([2, 0, 2])
one_hot = np.eye(4)[ids]
print("one-hot rows:\n", one_hot)
print("one_hot @ E:\n", one_hot @ E)
print("E[ids]:\n", E[ids])

G = np.array([[1.0, 0.0, 2.0],      # dL/dx for position 0
              [0.5, 0.5, 0.5],      # position 1
              [-1.0, 3.0, 1.0]])    # position 2
print("dL/dE = one_hot.T @ G:\n", one_hot.T @ G)

# Vision Transformer input: a 224 x 224 RGB image cut into 16 x 16 patches, projected to d = 768
H = W = 224; P = 16; C = 3; d = 768
n_patches = (H // P) * (W // P)
print("patches:", n_patches, "| values per patch:", P * P * C,
      "| projection params:", P * P * C * d + d, "| tokens with [CLS]:", n_patches + 1)`;
const WORKED_OUT = String.raw`one-hot rows:
 [[0. 0. 1. 0.]
 [1. 0. 0. 0.]
 [0. 0. 1. 0.]]
one_hot @ E:
 [[ 0.5 -0.5  2. ]
 [ 0.1  0.2  0.3]
 [ 0.5 -0.5  2. ]]
E[ids]:
 [[ 0.5 -0.5  2. ]
 [ 0.1  0.2  0.3]
 [ 0.5 -0.5  2. ]]
dL/dE = one_hot.T @ G:
 [[0.5 0.5 0.5]
 [0.  0.  0. ]
 [0.  3.  3. ]
 [0.  0.  0. ]]
patches: 196 | values per patch: 768 | projection params: 590592 | tokens with [CLS]: 197`;
const COSTS = String.raw`# Parameter cost of the token embedding (V x d) for published model shapes
models = [  # name, vocabulary size V, model width d, tied input/output embedding?
    ("BERT-base", 30_522, 768, True),
    ("GPT-2 small", 50_257, 768, True),
    ("GPT-2 XL", 50_257, 1600, True),
    ("Llama 2 7B", 32_000, 4096, False),
    ("Llama 3 8B", 128_256, 4096, False),
]
print(f"{'model':12s} {'V':>8s} {'d':>5s} {'V*d (input)':>13s} {'input + output':>19s} {'fp16 MB':>8s}")
for name, V, d, tied in models:
    p_in = V * d
    total = p_in if tied else 2 * p_in
    print(f"{name:12s} {V:8,d} {d:5d} {p_in:13,d} {total:19,d} {total * 2 / 1e6:8.0f}")


# GPT-2 small total, rebuilt from its architecture: 12 layers, d=768, FFN 3072, 1024 positions, tied output
V, d, L, ff, ctx = 50_257, 768, 12, 3072, 1024
attn = 4 * d * d + 4 * d                 # W_Q, W_K, W_V, W_O + biases
mlp = d * ff + ff + ff * d + d           # two linear layers + biases
ln = 2 * (2 * d)                         # two LayerNorms (gain + bias) per block
total = V * d + ctx * d + L * (attn + mlp + ln) + 2 * d   # + final LayerNorm
print(f"\nGPT-2 small: total {total:,} parameters; token embedding {V * d:,} = {V * d / total:.1%}")
print(f"   an untied output layer would add {V * d:,} more ({(total + V * d) / 1e6:.1f}M total)")`;
const COSTS_OUT = String.raw`model               V     d   V*d (input)      input + output  fp16 MB
BERT-base      30,522   768    23,440,896          23,440,896       47
GPT-2 small    50,257   768    38,597,376          38,597,376       77
GPT-2 XL       50,257  1600    80,411,200          80,411,200      161
Llama 2 7B     32,000  4096   131,072,000         262,144,000      524
Llama 3 8B    128,256  4096   525,336,576       1,050,673,152     2101

GPT-2 small: total 124,439,808 parameters; token embedding 38,597,376 = 31.0%
   an untied output layer would add 38,597,376 more (163.0M total)`;
const CONTEXT = String.raw`# What grows with the context length n? A 7B-class decoder: 32 layers, d = 4096, 32 heads, fp16 (2 bytes)
LAYERS, D, HEADS, BYTES = 32, 4096, 32, 2
GiB = 2 ** 30

print(f"{'n':>7s} {'scores/head':>14s} {'score memory':>13s} {'attn FLOPs':>11s} {'linear FLOPs':>13s} {'attn share':>10s} {'KV cache':>9s}")
for n in (512, 2048, 8192, 32_768, 131_072):
    scores = n * n                                        # entries of one n x n score matrix
    score_mem = HEADS * scores * BYTES / GiB              # one layer, all heads, if materialised
    attn = LAYERS * 4 * n * n * D                         # QK^T and AV: 2 matmuls x 2 FLOPs x n^2 x d
    linear = LAYERS * 24 * n * D * D                      # W_Q,W_K,W_V,W_O (8nd^2) + FFN with 4d hidden (16nd^2)
    kv = 2 * LAYERS * n * D * BYTES / GiB                 # K and V for every layer and position
    print(f"{n:7,d} {scores:14,d} {score_mem:9.2f} GiB {attn / 1e12:8.1f} TF {linear / 1e12:9.1f} TF"
          f" {attn / (attn + linear):9.1%} {kv:5.1f} GiB")
print(f"\nattention FLOPs equal the linear-layer FLOPs when 4 n^2 d = 24 n d^2, i.e. n = 6d = {6 * D:,}")`;
const CONTEXT_OUT = String.raw`n    scores/head  score memory  attn FLOPs  linear FLOPs attn share  KV cache
    512        262,144      0.02 GiB      0.1 TF       6.6 TF      2.0%   0.2 GiB
  2,048      4,194,304      0.25 GiB      2.2 TF      26.4 TF      7.7%   1.0 GiB
  8,192     67,108,864      4.00 GiB     35.2 TF     105.6 TF     25.0%   4.0 GiB
 32,768  1,073,741,824     64.00 GiB    562.9 TF     422.2 TF     57.1%  16.0 GiB
131,072 17,179,869,184   1024.00 GiB   9007.2 TF    1688.8 TF     84.2%  64.0 GiB

attention FLOPs equal the linear-layer FLOPs when 4 n^2 d = 24 n d^2, i.e. n = 6d = 24,576`;
const NUMPY_PIPE = String.raw`import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

# ---------------- 1) Vocabulary and token ids (word-level for readability; real models use subwords) ----------------
corpus = ["the cat sat on the mat", "the dog sat on the log", "a cat saw a dog"]
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]                   # ids 0..3 by convention here
words = sorted({w for s in corpus for w in s.split()})
vocab = SPECIALS + words
tok2id = {t: i for i, t in enumerate(vocab)}
PAD, UNK, BOS, EOS = 0, 1, 2, 3
print("vocabulary size V =", len(vocab), "->", vocab)


def encode(text):
    return [BOS] + [tok2id.get(w, UNK) for w in text.split()] + [EOS]


def pad_batch(seqs):
    n = max(map(len, seqs))
    ids = np.full((len(seqs), n), PAD)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    return ids, (ids != PAD).astype(int)                          # attention mask: 1 = real token


ids, mask = pad_batch([encode("the cat sat on the mat"), encode("a zebra sat")])
print("ids:\n", ids, "\nmask:\n", mask)

# ---------------- 2) Embedding lookup == one-hot times matrix ----------------
V, d = len(vocab), 8
E = rng.normal(0, 1 / np.sqrt(d), (V, d))                         # embedding matrix, one row per token
x_lookup = E[ids]                                                 # (B, n, d): just row indexing
one_hot = np.eye(V)[ids]                                          # (B, n, V)
x_matmul = one_hot @ E                                            # (B, n, V) @ (V, d) -> (B, n, d)
print("\nlookup shape", x_lookup.shape, "| max |lookup - one_hot @ E| =", np.abs(x_lookup - x_matmul).max())
print("multiply-adds: lookup 0, one-hot matmul", one_hot.size * d)

# ---------------- 3) The gradient only touches rows that were used ----------------
G = rng.normal(size=x_lookup.shape)                               # pretend dL/dx from the layers above
dE_matmul = np.einsum("bnv,bnd->vd", one_hot, G)                  # dL/dE = one_hot^T G
dE_scatter = np.zeros_like(E)
np.add.at(dE_scatter, ids, G)                                     # what frameworks do: scatter-add rows
print("\nmax |one_hot^T G - scatter-add| =", np.abs(dE_matmul - dE_scatter).max())
touched = np.flatnonzero(np.abs(dE_scatter).sum(1))
print("rows with non-zero gradient:", [vocab[i] for i in touched], f"({len(touched)} of {V})")
# (<pad> got a gradient only because G is random here; PyTorch's padding_idx forces that row's gradient to 0)
the = tok2id["the"]
print("'the' appears twice, its gradient is the sum:",
      np.allclose(dE_scatter[the], G[0, 1] + G[0, 5]))

# ---------------- 4) Scale by sqrt(d) and add sinusoidal positions ----------------
def sinusoidal(n, d):
    pos, i = np.arange(n)[:, None], np.arange(0, d, 2)[None, :]
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(pos / 10000 ** (i / d)), np.cos(pos / 10000 ** (i / d))
    return pe

pe = sinusoidal(ids.shape[1], d)
print(f"\nmean row norm: token embedding {np.linalg.norm(x_lookup, axis=-1).mean():.3f}, "
      f"scaled by sqrt(d) {np.linalg.norm(np.sqrt(d) * x_lookup, axis=-1).mean():.3f}, "
      f"positional encoding {np.linalg.norm(pe, axis=-1).mean():.3f}")
h = np.sqrt(d) * x_lookup + pe                                    # the Transformer's input, (B, n, d)

# ---------------- 5) Weight tying: the output layer reuses E ----------------
logits = h @ E.T                                                  # (B, n, V): score of every token
print("logits shape", logits.shape, "| output parameters saved by tying:", V * d)

# ---------------- 6) Long inputs: overlapping windows ----------------
def windows(seq, max_len, stride):
    """Chunks of max_len tokens; consecutive chunks start stride tokens apart (overlap = max_len - stride)."""
    starts = range(0, max(len(seq) - max_len, 0) + stride, stride)
    return [seq[s:s + max_len] for s in starts if s < len(seq)]

long_doc = list(range(100, 125))                                  # 25 token ids
for w in windows(long_doc, max_len=10, stride=7):
    print("window:", w)
covered = sorted({t for w in windows(long_doc, 10, 7) for t in w})
print("every token covered:", covered == long_doc)`;
const NUMPY_PIPE_OUT = String.raw`vocabulary size V = 13 -> ['<pad>', '<unk>', '<bos>', '<eos>', 'a', 'cat', 'dog', 'log', 'mat', 'on', 'sat', 'saw', 'the']
ids:
 [[ 2 12  5 10  9 12  8  3]
 [ 2  4  1 10  3  0  0  0]]
mask:
 [[1 1 1 1 1 1 1 1]
 [1 1 1 1 1 0 0 0]]

lookup shape (2, 8, 8) | max |lookup - one_hot @ E| = 0.0
multiply-adds: lookup 0, one-hot matmul 1664

max |one_hot^T G - scatter-add| = 0.0
rows with non-zero gradient: ['<pad>', '<unk>', '<bos>', '<eos>', 'a', 'cat', 'mat', 'on', 'sat', 'the'] (10 of 13)
'the' appears twice, its gradient is the sum: True

mean row norm: token embedding 0.853, scaled by sqrt(d) 2.413, positional encoding 2.000
logits shape (2, 8, 13) | output parameters saved by tying: 104
window: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
window: [107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
window: [114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
window: [121, 122, 123, 124]
every token covered: True`;
const TORCH_EMB = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
V, d, PAD = 10, 4, 0

# 1) nn.Embedding is a lookup table; F.one_hot(...) @ weight gives the same result
emb = nn.Embedding(V, d, padding_idx=PAD)
ids = torch.tensor([[5, 2, 5, 7], [3, 9, PAD, PAD]])          # batch of 2, the second padded
x = emb(ids)
print("output shape", tuple(x.shape),
      "| max |emb(ids) - one_hot @ W| =", (x - F.one_hot(ids, V).float() @ emb.weight).abs().max().item())
print("padding row at init:", emb.weight[PAD].tolist())

# 2) Gradients: only used rows are non-zero, repeated ids accumulate, the padding row stays zero
x.sum().backward()
print("rows with gradient:", torch.nonzero(emb.weight.grad.abs().sum(1)).flatten().tolist(),
      "| grad row 5 (used twice):", emb.weight.grad[5].tolist(), "| grad row PAD:", emb.weight.grad[PAD].tolist())

# 3) sparse=True returns a sparse gradient holding only the touched rows
sp = nn.Embedding(50_000, 64, sparse=True)
sp(torch.tensor([[1, 7, 7, 42]])).sum().backward()
g = sp.weight.grad
print("sparse grad:", g.layout, "| stored rows:", g.coalesce().indices().flatten().tolist(),
      f"| dense equivalent {50_000 * 64:,} numbers")
opt = torch.optim.SparseAdam(sp.parameters(), lr=1e-3)        # optimizers that accept sparse gradients:
opt.step()                                                     # SGD, SparseAdam, Adagrad


# 4) Weight tying: one matrix for the input embedding and the output projection
class TinyLM(nn.Module):
    def __init__(self, V, d, tie, emb_std=None):
        super().__init__()
        self.emb = nn.Embedding(V, d)                          # default init: N(0, 1)
        if emb_std is not None:
            nn.init.normal_(self.emb.weight, std=emb_std)
        self.block = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.lm_head = nn.Linear(d, V, bias=False)
        if tie:
            self.lm_head.weight = self.emb.weight                 # same Parameter object, shape (V, d)

    def forward(self, ids):
        h = self.emb(ids) * self.emb.embedding_dim ** 0.5      # scale by sqrt(d_model)
        return self.lm_head(h + self.block(h))


for tie in (False, True):
    m = TinyLM(V=32_000, d=512, tie=tie)
    n = sum(p.numel() for p in m.parameters())                 # parameters() de-duplicates shared tensors
    print(f"tie={tie!s:5}  parameters {n:,}")

ids3, targets = torch.tensor([[1, 2, 3]]), torch.tensor([2, 3, 4])
for std in (None, 512 ** -0.5, 0.02):                           # default N(0,1), N(0,1/d), GPT-2's 0.02
    torch.manual_seed(0)
    m = TinyLM(32_000, 512, tie=True, emb_std=std)
    loss = F.cross_entropy(m(ids3).view(-1, 32_000), targets)
    loss.backward()
    rows = int((m.emb.weight.grad.abs().sum(1) > 0).sum())
    print(f"tied, emb std {'1 (default)' if std is None else f'{std:.4f}'}: initial loss {loss.item():10.2f} "
          f"(uniform guess = ln V = {torch.log(torch.tensor(32_000.)).item():.2f}), rows with gradient {rows:,}")

# 5) Loading fixed (pretrained) vectors
pretrained = torch.randn(V, d)
frozen = nn.Embedding.from_pretrained(pretrained, freeze=True, padding_idx=PAD)
print("from_pretrained: requires_grad =", frozen.weight.requires_grad)`;
const TORCH_EMB_OUT = String.raw`output shape (2, 4, 4) | max |emb(ids) - one_hot @ W| = 0.0
padding row at init: [0.0, 0.0, 0.0, 0.0]
rows with gradient: [2, 3, 5, 7, 9] | grad row 5 (used twice): [2.0, 2.0, 2.0, 2.0] | grad row PAD: [0.0, 0.0, 0.0, 0.0]
sparse grad: torch.sparse_coo | stored rows: [1, 7, 42] | dense equivalent 3,200,000 numbers
tie=False  parameters 34,867,712
tie=True   parameters 18,483,712
tied, emb std 1 (default): initial loss   11875.20 (uniform guess = ln V = 10.37), rows with gradient 4
tied, emb std 0.0442: initial loss      22.84 (uniform guess = ln V = 10.37), rows with gradient 32,000
tied, emb std 0.0200: initial loss      10.45 (uniform guess = ln V = 10.37), rows with gradient 32,000
from_pretrained: requires_grad = False`;
const GEOMETRY = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

# A toy language: "the <colour> <animal> <verb> <number> <food> ."
# Carnivores eat meat-type food, herbivores eat plants: a two-level structure for the embeddings to discover.
CATS = {
    "colour": ["red", "blue", "green", "white"],
    "carnivore": ["cat", "dog", "fox"],
    "herbivore": ["cow", "sheep", "horse"],
    "verb": ["eats", "likes", "wants"],
    "number": ["two", "three", "four", "five"],
    "meat": ["fish", "meat", "eggs"],
    "plant": ["grass", "hay", "leaves"],
}
vocab = ["<pad>", "the", "."] + [w for ws in CATS.values() for w in ws]
idx = {w: i for i, w in enumerate(vocab)}
V, D, CTX = len(vocab), 8, 3


def make_data(seed, n_sentences=3000):
    g = torch.Generator().manual_seed(seed)
    pick = lambda c: CATS[c][torch.randint(len(CATS[c]), (1,), generator=g).item()]
    stream = []
    for _ in range(n_sentences):
        kind = "carnivore" if torch.rand(1, generator=g).item() < 0.5 else "herbivore"
        food = "meat" if kind == "carnivore" else "plant"
        stream += [idx[w] for w in ["the", pick("colour"), pick(kind), pick("verb"), pick("number"), pick(food), "."]]
    data = torch.tensor(stream)
    X = torch.stack([data[i:i + CTX] for i in range(len(data) - CTX)])   # 3 previous tokens ...
    return X, data[CTX:]                                                  # ... predict the next one


class TinyLM(nn.Module):                                                  # embedding -> MLP -> next-token logits
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.mlp = nn.Sequential(nn.Linear(CTX * D, 64), nn.Tanh(), nn.Linear(64, V))

    def forward(self, x):
        return self.mlp(self.emb(x).flatten(1))


def cosine(E, a, b):
    return F.cosine_similarity(E[idx[a]], E[idx[b]], dim=0).item()


def report(E, tag):
    words = [w for ws in CATS.values() for w in ws]
    group = {w: c for c, ws in CATS.items() for w in ws}
    within = [cosine(E, a, b) for a in words for b in words if a < b and group[a] == group[b]]
    across = [cosine(E, a, b) for a in words for b in words if a < b and group[a] != group[b]]
    print(f"{tag}: mean cosine within a category {sum(within) / len(within):+.3f}, "
          f"across categories {sum(across) / len(across):+.3f}")
    for w in ["cat", "cow", "fish", "three"]:
        sims = sorted(((cosine(E, w, o), o) for o in words if o != w), reverse=True)[:4]
        print(f"   nearest to {w:5s}: " + ", ".join(f"{o} {s:.2f}" for s, o in sims))
    print("   cos(cat, dog) = %+.2f   cos(cat, cow) = %+.2f   cos(fish, grass) = %+.2f"
          % (cosine(E, "cat", "dog"), cosine(E, "cat", "cow"), cosine(E, "fish", "grass")))


X, Y = make_data(seed=0)
for weight_decay in (0.0, 0.5):
    torch.manual_seed(0)
    model = TinyLM()
    if weight_decay == 0.0:
        report(model.emb.weight.detach(), "random init      ")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(1)
    for epoch in range(10):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(model(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        final = F.cross_entropy(model(X), Y).item()
    report(model.emb.weight.detach(), f"trained, wd={weight_decay:.1f} ")
    print(f"   final loss {final:.3f}")`;
const GEOMETRY_OUT = String.raw`random init      : mean cosine within a category -0.019, across categories +0.004
   nearest to cat  : white 0.64, eats 0.55, four 0.53, meat 0.46
   nearest to cow  : eggs 0.81, blue 0.74, two 0.50, sheep 0.49
   nearest to fish : leaves 0.58, eggs 0.47, likes 0.39, red 0.37
   nearest to three: sheep 0.45, cow 0.21, eggs 0.17, five 0.16
   cos(cat, dog) = +0.30   cos(cat, cow) = +0.02   cos(fish, grass) = -0.07
trained, wd=0.0 : mean cosine within a category +0.243, across categories -0.036
   nearest to cat  : white 0.63, meat 0.53, four 0.48, dog 0.43
   nearest to cow  : sheep 0.67, blue 0.49, eggs 0.43, eats 0.24
   nearest to fish : leaves 0.64, eats 0.62, eggs 0.59, likes 0.54
   nearest to three: five 0.56, sheep 0.42, horse 0.36, meat 0.21
   cos(cat, dog) = +0.43   cos(cat, cow) = +0.01   cos(fish, grass) = +0.23
   final loss 0.978
trained, wd=0.5 : mean cosine within a category +0.901, across categories -0.051
   nearest to cat  : dog 0.92, fox 0.85, white 0.46, blue 0.32
   nearest to cow  : sheep 0.95, horse 0.88, eats 0.23, blue 0.12
   nearest to fish : leaves 0.96, eggs 0.95, grass 0.95, meat 0.89
   nearest to three: five 0.93, four 0.93, two 0.91, leaves 0.53
   cos(cat, dog) = +0.92   cos(cat, cow) = -0.13   cos(fish, grass) = +0.95
   final loss 1.001`;
const TOKMEASURE = String.raw`import inspect
import json, collections, email, http, logging, argparse, unittest, typing, pathlib, decimal, fractions, statistics
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ---- 1) A real English training corpus that ships with Python: the standard library's docstrings ----
def docstrings(module):
    out = [inspect.getdoc(module) or ""]
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            out.append(inspect.getdoc(obj) or "")
    return out

corpus = []
for m in (json, collections, email, http, logging, argparse, unittest, typing, pathlib, decimal, fractions, statistics):
    corpus += [d for d in docstrings(m) if d]
print(f"corpus: {len(corpus)} docstrings, {sum(len(d) for d in corpus):,} characters, "
      f"{sum(len(d.split()) for d in corpus):,} words")


def train_bpe(vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)   # GPT-2 style byte-level BPE
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    return tok


# ---- 2) Tokens per word on held-out English, for two vocabulary sizes ----
held_out = ("The function returns a new list containing the elements of the sequence in sorted order. "
            "If the argument is missing, the default value is used and a warning is logged.")
for vs in (1000, 4000):
    tok = train_bpe(vs)
    enc = tok.encode(held_out)
    print(f"\nvocab {tok.get_vocab_size()}: {len(held_out.split())} words -> {len(enc.ids)} tokens "
          f"({len(enc.ids) / len(held_out.split()):.2f} tokens/word, {len(held_out) / len(enc.ids):.2f} chars/token)")
    print("   ", enc.tokens[:14])

# ---- 3) The same sentence in six languages, with the 4000-token English-trained tokenizer ----
parallel = {
    "English": "The weather is very nice today, so we are going to the park with our children.",
    "French": "Il fait très beau aujourd'hui, alors nous allons au parc avec nos enfants.",
    "German": "Das Wetter ist heute sehr schön, deshalb gehen wir mit unseren Kindern in den Park.",
    "Spanish": "Hoy hace muy buen tiempo, así que vamos al parque con nuestros hijos.",
    "Hindi": "आज मौसम बहुत अच्छा है, इसलिए हम अपने बच्चों के साथ पार्क जा रहे हैं।",
    "Chinese": "今天天气很好，所以我们要和孩子们一起去公园。",
}
print()
base = len(tok.encode(parallel["English"]).ids)
for lang, s in parallel.items():
    n = len(tok.encode(s).ids)
    print(f"{lang:8s} chars={len(s):3d} utf8 bytes={len(s.encode()):3d} tokens={n:3d}  ({n / base:.1f}x English)")

# ---- 4) Numbers: how digits are split depends on what the merges happened to learn ----
print()
for s in ["2023", "3.14159", "1234567", "12345678 + 87654321"]:
    print(f"{s!r:24} -> {tok.encode(s).tokens}")
print("round trip:", tok.decode(tok.encode(parallel["Hindi"]).ids) == parallel["Hindi"])`;
const TOKMEASURE_OUT = String.raw`corpus: 303 docstrings, 145,992 characters, 21,514 words

vocab 1000: 30 words -> 47 tokens (1.57 tokens/word, 3.55 chars/token)
    ['The', 'Ġfunction', 'Ġreturns', 'Ġa', 'Ġnew', 'Ġlist', 'Ġcontain', 'ing', 'Ġthe', 'Ġe', 'lements', 'Ġof', 'Ġthe', 'Ġse']

vocab 4000: 30 words -> 33 tokens (1.10 tokens/word, 5.06 chars/token)
    ['The', 'Ġfunction', 'Ġreturns', 'Ġa', 'Ġnew', 'Ġlist', 'Ġcontaining', 'Ġthe', 'Ġelements', 'Ġof', 'Ġthe', 'Ġsequence', 'Ġin', 'Ġsorted']

English  chars= 78 utf8 bytes= 78 tokens= 29  (1.0x English)
French   chars= 74 utf8 bytes= 75 tokens= 43  (1.5x English)
German   chars= 83 utf8 bytes= 84 tokens= 42  (1.4x English)
Spanish  chars= 69 utf8 bytes= 70 tokens= 37  (1.3x English)
Hindi    chars= 68 utf8 bytes=174 tokens=174  (6.0x English)
Chinese  chars= 22 utf8 bytes= 66 tokens= 66  (2.3x English)

'2023'                   -> ['20', '23']
'3.14159'                -> ['3', '.', '14', '15', '9']
'1234567'                -> ['123456', '7']
'12345678 + 87654321'    -> ['123456', '78', 'Ġ+', 'Ġ8', '76', '54', '3', '2', '1']
round trip: True`;

const KERAS = String.raw`import numpy as np
import keras
from keras import ops

V, d = 1000, 64

# mask_zero=True: id 0 is padding; downstream mask-aware layers (RNNs, attention) skip those positions
emb = keras.layers.Embedding(input_dim=V, output_dim=d, mask_zero=True)
ids = np.array([[5, 17, 42, 0, 0], [8, 9, 10, 11, 12]])
x = emb(ids)
print("embeddings", x.shape, "| mask", emb.compute_mask(ids).numpy().astype(int).tolist())


class TiedLM(keras.Model):
    """Input embedding and output projection share one (V, d) matrix."""
    def __init__(self, V, d):
        super().__init__()
        self.emb = keras.layers.Embedding(V, d)
        self.ffn = keras.layers.Dense(d, activation="gelu")

    def call(self, ids):
        h = self.emb(ids) * d ** 0.5
        h = h + self.ffn(h)
        return ops.matmul(h, ops.transpose(self.emb.embeddings))   # logits (batch, n, V)


model = TiedLM(V, d)
print("logits", model(ids).shape, "| parameters", model.count_params())`;

const lesson: Lesson = {
  id: "m11/tokens-embeddings",
  title: "Tokenization, Embeddings and Context Windows",
  summary:
    "How a Transformer's input is built: token ids and vocabularies, the embedding matrix as a lookup (proved equal to one-hot × matrix, with its sparse gradient), parameter cost for real vocabularies, learned embedding geometry measured on a trained model, √d_model scaling, positional information, weight tying, what limits the context window (quadratic attention cost computed from 512 to 128k tokens, KV cache, position extrapolation), long-input strategies, tokenization effects measured with a real BPE tokenizer, and embeddings for images and other modalities.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m13/text-preprocessing", "m10/scaled-dot-product-attention", "m10/positional-encoding"],
  next: ["m11/transformer-architecture", "m11/training-inference", "m11/transformer-families"],
  terms: ["token", "token-id", "vocabulary", "embedding", "embedding-matrix", "weight-tying", "context-window", "fertility"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "A Transformer never sees text. It sees a matrix: one row of $d_{\\text{model}}$ numbers per **token**. This lesson is about how that matrix is produced and what limits its height. Three components are involved: a **tokenizer** that turns text into a sequence of integer **token ids** from a fixed **vocabulary**; an **embedding matrix** that turns each id into a learned vector; and **positional information** that tells the order-blind attention layers where each token sits. The number of rows the model can process at once is its **context window**.",
      },
      {
        t: "p",
        text: "Each of these choices has measurable consequences. The vocabulary size fixes the size of the embedding and output layers — over half a billion parameters each for a 128k-token vocabulary at width 4096. The tokenizer decides how many tokens a sentence costs, and the same sentence can cost several times more in Hindi than in English under an English-trained tokenizer, which means less text per context window and higher per-token cost. The context window is limited by attention's cost, which grows with the **square** of the sequence length, by the memory of the key/value cache, and by whether the positional scheme works beyond the lengths seen in training.",
      },
      {
        t: "p",
        text: "Embedding tables are older than Transformers — they are the input layer of neural language models since Bengio et al. (2003) and of [word2vec](/learn/m13/word-embeddings) — but in a Transformer they are trained end-to-end with the rest of the network. The same interface also lets Transformers read images, audio and other data: anything cut into pieces and linearly projected to $d_{\\text{model}}$ becomes a sequence of tokens.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The input interface in one line",
        text: String.raw`text $\to$ token ids $(t_1,\dots,t_n)$ $\to$ rows of $E\in\mathbb{R}^{V\times d}$ $\to$ $\times\sqrt{d}$ (original Transformer) $\to$ $+$ positions $\to$ Transformer layers $\to$ $h\,E^\top$ (tied output) $\to$ softmax over the $V$ tokens.`,
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Text Preprocessing and Subword Tokenization](/learn/m13/text-preprocessing) — how BPE, WordPiece and Unigram build a vocabulary, byte-level BPE, special tokens, padding, attention masks and truncation. This lesson **uses** tokenizers and measures their effects; it does not re-derive BPE.",
          "[Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) — the $O(n^2d)$ cost of attention and the padding mask; here we turn the complexity into concrete numbers for real context lengths.",
          "[Positional Encoding](/learn/m10/positional-encoding) — sinusoidal, learned, RoPE and ALiBi. We need the fact that positions are added to (or rotated into) the token vectors, and the extrapolation discussion.",
          "[Vectors and matrices](/learn/m02/vectors-matrices) and [norms and cosine similarity](/learn/m02/norms-spaces) — matrix products with one-hot vectors, vector norms, and cosine similarity to measure embedding geometry.",
          "[Classification losses](/learn/m05/classification-losses) — a language model's output is a softmax over the vocabulary trained with cross-entropy; $\\ln V$ is the loss of a uniform guess.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A locker per word",
        text: "Picture a wall of numbered lockers, one per vocabulary entry. Each locker holds a card with a few hundred numbers written on it. The tokenizer reads your sentence and gives you a list of locker numbers; the embedding layer simply opens those lockers and lays the cards out in order. Nothing is computed — it is retrieval. Training is what writes good numbers on the cards: every time a locker's card is used and the model makes a mistake, the card is corrected a little. Lockers that are never opened in a batch are not touched at all.",
      },
      {
        t: "p",
        text: "Why vectors and not the ids themselves? An id is a name, not a quantity: token 512 is not \"twice\" token 256, and feeding raw ids into a network would invent an order that means nothing. Giving every token its own vector lets the model place tokens freely in a $d$-dimensional space. Training then puts tokens that **behave** alike — that appear in similar contexts and predict similar continuations — near each other, because the layers above can then treat them alike. We measure this below: after training a tiny model, \"cat\" ends up closest to \"dog\" and \"fox\", not to \"cow\".",
      },
      {
        t: "p",
        text: "The context window is the size of the desk on which the cards are laid out. Attention compares every card with every other card, so a desk twice as long means four times as many comparisons. That is why context length is the expensive dimension, and why what the tokenizer does matters: if your language is split into six times as many tokens, you fit one sixth as much text on the same desk.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "1. From text to token ids (recap)" },
      {
        t: "p",
        text: "A subword tokenizer (BPE, WordPiece or Unigram — see [Text Preprocessing](/learn/m13/text-preprocessing)) normalises the text, splits it into pieces from a vocabulary of size $V$ and maps each piece to an integer in $\\{0,\\dots,V-1\\}$. Special tokens such as `<bos>`, `<eos>` and `<pad>` get ids of their own. A batch of sequences is padded to a common length and accompanied by an **attention mask** (1 = real token, 0 = padding) that becomes the key-padding mask inside attention. The ids are the only thing the model receives from the text.",
      },
      { t: "h", text: "2. The embedding layer" },
      {
        t: "steps",
        items: [
          { title: "Allocate the matrix", text: "A trainable matrix E with V rows and d columns. Row t is the vector for token id t. It is initialised randomly (e.g. normal with a small standard deviation).", tex: String.raw`E \in \mathbb{R}^{V\times d}` },
          { title: "Look up", text: "For each id in the input, copy out its row. A batch of shape (B, n) of integers becomes a tensor of shape (B, n, d). This is mathematically a one-hot vector times E, but implemented as indexing.", tex: String.raw`x_i = E_{t_i,:} = e_{t_i}^\top E` },
          { title: "Scale (original Transformer)", text: "Vaswani et al. multiply the embeddings by √d_model. Many later models (GPT-2, BERT) do not; they instead initialise E with a small fixed standard deviation.", tex: String.raw`x_i \leftarrow \sqrt{d}\,x_i` },
          { title: "Add position", text: "Add a sinusoidal or learned position vector, or leave the input as is and rotate queries and keys inside attention (RoPE). Without this step the model cannot distinguish 'dog bites man' from 'man bites dog'.", tex: String.raw`h^{(0)}_i = \sqrt{d}\,E_{t_i,:} + p_i` },
          { title: "Transformer layers", text: "The stack of attention and feed-forward blocks transforms the n × d matrix, keeping its shape." },
          { title: "Output projection", text: "The final hidden vector at each position is multiplied by a V × d matrix to give one score (logit) per vocabulary token; softmax turns them into next-token probabilities. With **weight tying** this matrix is E itself.", tex: String.raw`\text{logits}_i = h^{(L)}_i\,E^\top \in \mathbb{R}^{V}` },
        ],
      },
      { t: "h", text: "3. Learning the embeddings" },
      {
        t: "p",
        text: "During backpropagation the gradient flowing into the input $x_i$ is added to row $t_i$ of $E$, and nothing else. Rows of tokens absent from the batch receive exactly zero gradient; a token used twice receives the **sum** of its two gradients. This sparsity is why frameworks offer sparse embedding gradients, and why a `padding_idx` can keep the padding row fixed at zero. With weight tying the output projection adds a second, **dense** gradient: the softmax touches every vocabulary row at every position.",
      },
      { t: "h", text: "4. What limits the context window" },
      {
        t: "list",
        items: [
          "**Attention compute**: the score matrix has $n^2$ entries per head per layer; computing $QK^\\top$ and $AV$ costs $4n^2d$ FLOPs per layer, versus $24nd^2$ for the linear layers. For $n > 6d$ attention dominates (computed below for a 7B-class model: 2% of FLOPs at 512 tokens, 84% at 128k).",
          "**Attention memory**: materialising the scores needs $h\\,n^2$ numbers per layer — 1 TiB at 128k tokens for 32 heads in fp16. Fused kernels (FlashAttention) compute softmax in tiles and never store the full matrix, which removes this limit but not the quadratic compute.",
          "**KV cache**: generation stores keys and values for every token in every layer, growing **linearly** with $n$ (64 GiB at 128k tokens for the 7B-class shape). Grouped-query attention reduces it — see [Multi-Head Attention](/learn/m10/multi-head-attention).",
          "**Positional extrapolation**: a learned position table has no row beyond its maximum length; sinusoidal and RoPE encodings are defined for any position but models degrade on angle combinations never seen in training. Long-context models are usually trained at a shorter length and then extended with RoPE position interpolation or base scaling plus some long-sequence training, or use schemes like ALiBi — details in [Positional Encoding](/learn/m10/positional-encoding).",
          "**Training data and use**: a model only learns to use long-range information if it was trained on documents where that information matters. Liu et al. (2023, “Lost in the Middle: How Language Models Use Long Contexts”) found that models used information at the start and end of a long context better than information in the middle.",
        ],
      },
      { t: "h", text: "5. Inputs longer than the window" },
      {
        t: "table",
        head: ["Strategy", "How", "Good for", "Cost / risk"],
        rows: [
          ["Truncation", "Keep the first (or last) n tokens", "Classification where the start carries the signal", "Information beyond the cut is lost"],
          ["Chunking", "Split into non-overlapping windows, process each, aggregate (average, vote, max)", "Embedding or classifying long documents", "Dependencies across chunk borders are cut"],
          ["Sliding windows with stride", "Windows of n tokens starting every s < n tokens; overlap n − s", "Extractive QA, token tagging: every token gets context on both sides in some window", "≈ n/s times more compute; overlapping predictions must be merged"],
          ["Sliding-window attention", "Inside the model, each token attends only to the previous w tokens; layers stack the reach", "Long generation with bounded cache", "No direct access beyond w per layer"],
          ["Retrieval", "Index chunks, retrieve the relevant ones into the window", "Question answering over large corpora (RAG)", "Retrieval errors become model errors"],
          ["Summarise / compress", "Replace old context by a summary or compressed memory", "Long conversations", "Lossy"],
        ],
      },
      { t: "h", text: "6. Other modalities through the same interface" },
      {
        t: "p",
        text: "The embedding layer is just a map from *pieces of input* to $d$-dimensional vectors. For discrete tokens it is a lookup; for continuous data it is a linear projection. The Vision Transformer (Dosovitskiy et al., 2021, “An Image is Worth 16x16 Words”) cuts a 224 × 224 RGB image into 16 × 16 patches, flattens each to $16\\cdot16\\cdot3 = 768$ numbers and multiplies by a learned $768\\times d$ matrix: 196 patch tokens, plus a learned `[CLS]` token, plus learned position embeddings. Audio models first compute a spectrogram and turn short time slices into tokens with convolutions. Once the pieces are vectors, the Transformer cannot tell where they came from — which is what makes multimodal models possible ([Transformer families](/learn/m11/transformer-families)).",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Notation" },
      {
        t: "list",
        items: [
          String.raw`$V$ — vocabulary size; $d = d_{\text{model}}$ — embedding width; $n$ — sequence length in tokens.`,
          String.raw`$t_1,\dots,t_n\in\{0,\dots,V-1\}$ — token ids; $e_t\in\{0,1\}^V$ — one-hot vector with a 1 in position $t$.`,
          String.raw`$E\in\mathbb{R}^{V\times d}$ — embedding matrix; $X\in\mathbb{R}^{n\times d}$ — the embedded sequence, row $i$ = $E_{t_i,:}$.`,
          String.raw`$O\in\{0,1\}^{n\times V}$ — the one-hot matrix with rows $e_{t_i}^\top$; $p_i\in\mathbb{R}^d$ — positional vector; $\mathcal{L}$ — the loss.`,
        ],
      },
      { t: "h", text: "Lookup = one-hot × matrix (proof)" },
      {
        t: "math",
        tex: String.raw`(e_t^\top E)_j = \sum_{v=0}^{V-1}(e_t)_v\,E_{vj} = E_{tj}\quad\Longrightarrow\quad e_t^\top E = E_{t,:},\qquad X = OE`,
        caption: "Only the term v = t survives the sum, because (e_t)_v = 0 for v ≠ t.",
      },
      {
        t: "p",
        text: String.raw`So an embedding layer is a linear layer without bias applied to one-hot inputs. The lookup is the efficient way to compute it: the matrix product costs $n\cdot V\cdot d$ multiply-adds, almost all of them with zero, while indexing costs none. The gradient follows from $X = OE$:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\mathcal{L}}{\partial E} = O^\top\frac{\partial\mathcal{L}}{\partial X}\qquad\Longrightarrow\qquad \frac{\partial\mathcal{L}}{\partial E_{v,:}} = \sum_{i:\,t_i = v}\frac{\partial\mathcal{L}}{\partial x_i}`,
        caption: "Row v collects the gradients of all positions holding token v; rows of unused tokens are exactly zero.",
      },
      { t: "h", text: "Worked example" },
      {
        t: "p",
        text: "Take $V = 4$, $d = 3$, the sequence of ids $(2, 0, 2)$ and an upstream gradient $G = \\partial\\mathcal{L}/\\partial X$ with rows $(1,0,2)$, $(0.5,0.5,0.5)$, $(-1,3,1)$. Then $X$ has rows $E_{2,:}, E_{0,:}, E_{2,:}$, and the gradient of row 2 is $(1,0,2)+(-1,3,1) = (0,3,3)$, of row 0 is $(0.5,0.5,0.5)$, and rows 1 and 3 get zeros. The script also checks the Vision Transformer patch arithmetic from the Mechanism section.",
      },
      { t: "code", lang: "python", title: "worked_lookup.py", code: WORKED, output: WORKED_OUT },
      { t: "h", text: "Parameter cost of the vocabulary" },
      {
        t: "p",
        text: String.raw`The embedding has $Vd$ parameters; an untied output layer adds another $Vd$. For published model shapes:`,
      },
      { t: "code", lang: "python", title: "embedding_costs.py", code: COSTS, output: COSTS_OUT },
      {
        t: "p",
        text: String.raw`In GPT-2 small the token embedding is **31%** of all 124,439,808 parameters (rebuilt exactly from its architecture); without tying, the model would have 163.0M parameters. Llama 3's move to a 128,256-token vocabulary costs 525M parameters per matrix and 1.05B for the untied pair — about an eighth of an 8B model. Larger vocabularies make text shorter in tokens (see the tokenizer measurements in the Framework section), so the cost buys context capacity and speed.`,
      },
      { t: "h", text: "Why multiply by √d_model?" },
      {
        t: "p",
        text: String.raw`Vaswani et al. state that they multiply the embedding weights by $\sqrt{d_{\text{model}}}$ and share them with the pre-softmax projection, without further justification. A standard explanation compares magnitudes. A sinusoidal position vector has $d/2$ pairs $(\sin\theta,\cos\theta)$, so its norm is exactly $\|p_i\| = \sqrt{d/2}$ — 16 for $d=512$. If the shared matrix is initialised like an output layer, with entries of variance $1/d$, a token row has norm $\approx 1$, and the position signal would dominate the token signal 16 to 1. Multiplying by $\sqrt{d} = 22.6$ brings the token part to $\approx\sqrt d$, the same order as the position part. The from-scratch code measures this at $d = 8$: token rows 0.853, scaled 2.413, positions exactly $\sqrt{4} = 2.000$.`,
      },
      { t: "h", text: "Weight tying" },
      {
        t: "p",
        text: String.raw`Press and Wolf (2017, “Using the Output Embedding to Improve Language Models”) proposed using the same matrix for the input embedding and the output projection, $\text{logits} = hE^\top$, and showed that it reduces parameters and improves language-model perplexity; Inan et al. (2017) proposed the same idea from a different derivation. The saving is exactly $Vd$ parameters (16,384,000 for $V = 32{,}000$, $d = 512$ — the PyTorch code below counts 34,867,712 → 18,483,712). The intuition: tokens that should be predicted in similar contexts get similar output rows, and tokens that appear in similar contexts get similar input rows; tying makes each signal train the other. The gradient of the shared matrix is the sum of the sparse input-side gradient and the dense output-side gradient. GPT-2 and BERT tie; the Llama models keep separate matrices.`,
      },
      { t: "h", text: "The cost of context length" },
      {
        t: "p",
        text: String.raw`Per layer and sequence, attention matmuls cost $4n^2d$ FLOPs and the linear layers (projections $8nd^2$ + a feed-forward network with hidden width $4d$, $16nd^2$) cost $24nd^2$. They are equal at $n = 6d$. For a 7B-class shape (32 layers, $d = 4096$, 32 heads, fp16), one forward pass over $n$ tokens:`,
      },
      { t: "code", lang: "python", title: "context_costs.py", code: CONTEXT, output: CONTEXT_OUT },
      {
        t: "p",
        text: String.raw`Going from 2,048 to 131,072 tokens (×64) multiplies the linear-layer FLOPs by 64 but the attention FLOPs by $64^2 = 4096$ (2.2 → 9,007 TFLOPs), and attention's share of compute rises from 7.7% to 84.2%. Materialised scores would need 1 TiB per layer — impossible without fused kernels — and the KV cache of a single 128k-token sequence (64 GiB with full multi-head attention) is several times larger than the model's own weights (about 13 GiB for 7 billion fp16 parameters). (The FLOP counts here ignore the causal mask, which roughly halves the attention part, and count one multiply-add as 2 FLOPs.)`,
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "h", text: "1. Text → tokens" },
      {
        t: "p",
        text: "The tokenizer simulator trains byte-pair encoding on the text in its box. Use it to see the first stage of the input interface: how the number of tokens for a piece of text depends on what the tokenizer learned.",
      },
      { t: "sim", id: "tokenizer", caption: "BPE learned live: step through merges and encode new text with the merges learned so far." },
      {
        t: "list",
        ordered: true,
        items: [
          "Type a sentence in the encode box and note the token count. Step the merges forward: the same sentence costs fewer tokens as the vocabulary grows — the vocabulary-size versus sequence-length trade-off of the Math section.",
          "Encode a word made of the corpus's letters but never seen in it: it is still encoded, from smaller pieces. Then type characters that never appear in the corpus: they become id 0 (`<unk>`). Each id you see is a row number of the embedding matrix.",
          "Keep the English corpus, learn all merges, and encode a sentence in another language written in Latin letters (e.g. German). Compare its token count with an English sentence of similar length — a miniature version of the multilingual measurement in the Framework section.",
        ],
      },
      { t: "h", text: "2. Tokens → embeddings → prediction" },
      {
        t: "p",
        text: "The Transformer explorer runs a real, trained, character-level Transformer in your browser: a 37-character vocabulary (lower-case letters, space and some punctuation), $d_{\\text{model}} = 48$, 2 pre-LN layers with 4 heads each, learned position embeddings, a 64-character context window and a **weight-tied** output head. It was trained on the first chapter of *Alice's Adventures in Wonderland*, so it has memorised much of that text and knows little else. It does not scale embeddings by $\\sqrt{d}$: its input is simply $E[\\text{id}] + P[i]$, the GPT-2 convention.",
      },
      { t: "sim", id: "transformer", caption: "A small trained Transformer: characters → ids → E[id] + P[i] → 2 layers → logits = LN(x)·Eᵀ → next character." },
      {
        t: "list",
        ordered: true,
        items: [
          "**Characters → token ids.** Type a prompt such as `alice was beginning to get`. The explorer lower-cases the prompt, then every character is one token; the panel lists the whole vocabulary with its ids. Type a digit such as `7`: the explorer reports that it is not in the model's vocabulary — a character-level tokenizer has no fallback for unseen symbols, exactly the problem byte-level BPE solves.",
          "**Embeddings + positions.** The three heatmaps are $E[\\text{id}]$, $P[i]$ and their sum $x_0$. Click two positions holding the same character (e.g. two `e`s): their $E$ rows are identical — the same row of the matrix — while their $P$ rows and therefore their $x_0$ rows differ. That is how the model tells the two apart.",
          "**Logits.** Step to the final stage: the narration explains that the final LayerNorm output is multiplied by $E^\\top$ — the tied output head — giving 37 logits per position; the bars show the top-10 next characters with their probabilities. End the prompt mid-word and check that the top choices complete the word.",
          "**Generate** with `+1 char` or Play (greedy or sampling with a temperature slider). Each new character becomes a new id appended to the sequence, and the full forward pass is re-run. Generate past 64 characters: the status line reports how many characters have dropped out of the context window — the model literally cannot see them any more.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "The whole input interface in NumPy: building a vocabulary with special tokens, encoding and padding a batch with its attention mask, the embedding lookup checked against one-hot × matrix, the sparse scatter-add gradient, √d scaling with sinusoidal positions, tied output logits, and overlapping windows for a document longer than the context. For readability the vocabulary is word-level; a real model would use the subword tokenizer of [Text Preprocessing](/learn/m13/text-preprocessing) to produce the ids.",
      },
      { t: "code", lang: "python", title: "embedding_pipeline.py", code: NUMPY_PIPE, output: NUMPY_PIPE_OUT },
      {
        t: "list",
        items: [
          "`zebra` is not in the vocabulary, so it becomes `<unk>` (id 1) — the out-of-vocabulary problem that subword tokenizers remove.",
          "`E[ids]` and `one_hot @ E` agree exactly (difference 0.0); the matmul needed 1,664 multiply-adds to produce the same numbers as pure indexing.",
          "`np.add.at` is a scatter-add: it accumulates repeated ids correctly, whereas `dE[ids] += G` would keep only one of the two contributions for `the`. Its result matches $O^\\top G$ exactly.",
          "Only 10 of 13 rows receive gradient; `dog`, `log` and `saw` do not appear in the batch.",
          "The position vectors have norm exactly $\\sqrt{d/2} = 2$; the raw token rows (std $1/\\sqrt d$) have norm about 0.85 and the scaled ones about 2.4 — the comparison behind the $\\sqrt{d}$ factor.",
          "The windows of 10 tokens with stride 7 overlap by 3 tokens, and together cover all 25 tokens.",
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
              { t: "h", text: "1. nn.Embedding: lookup, padding_idx, sparse gradients, tying" },
              { t: "code", lang: "python", title: "embedding_torch.py", code: TORCH_EMB, output: TORCH_EMB_OUT },
              {
                t: "list",
                items: [
                  "`padding_idx=0` initialises that row to zeros and **masks its gradient**, so padding never learns a meaning (the NumPy version above, without it, gave the pad row a gradient).",
                  "`sparse=True` returns a `torch.sparse_coo` gradient storing only rows 1, 7 and 42 instead of 3.2 million numbers. Only some optimisers accept it (`SGD`, `SparseAdam`, `Adagrad`); with AdamW you need dense gradients.",
                  "Tying is one assignment: `lm_head.weight = emb.weight`. `parameters()` returns each tensor once, so the count drops by exactly $Vd = 16{,}384{,}000$.",
                  "**Initialisation matters with tying.** `nn.Embedding` initialises from $\\mathcal{N}(0,1)$. Tied, scaled by $\\sqrt{512}$, the logit of the *input* token itself is about $\\sqrt d\\,\\|e\\|^2 \\approx 22.6\\times512$, so the initial loss is 11,875 instead of $\\ln V = 10.37$, the softmax is saturated and only 4 rows get any gradient. With std $1/\\sqrt d$ the loss is still 22.84 (the model starts out \"predicting its own input\"); with GPT-2's std 0.02 it is 10.45, close to a uniform guess.",
                  "`from_pretrained(..., freeze=True)` loads fixed vectors (e.g. word2vec) with `requires_grad = False`.",
                ],
              },
              { t: "h", text: "2. Learned geometry: similar tokens become close" },
              {
                t: "p",
                text: "A tiny next-token model (embedding of width 8 → MLP) is trained on a toy language \"the <colour> <animal> <verb> <number> <food> .\" in which carnivores (cat, dog, fox) are always followed later by meat-type food and herbivores (cow, sheep, horse) by plants. We measure cosine similarities between embedding rows at random initialisation and after training, with and without weight decay.",
              },
              { t: "code", lang: "python", title: "embedding_geometry.py", code: GEOMETRY, output: GEOMETRY_OUT },
              {
                t: "list",
                items: [
                  "**Behaviour defines similarity.** With weight decay the mean within-category cosine rises from −0.019 to +0.901, and `cat`'s nearest neighbours are `dog` (0.92) and `fox` (0.85) while `cos(cat, cow)` is −0.13: carnivores and herbivores are separated because, as context, they predict different foods.",
                  "**Same behaviour → same vector, even when meanings differ.** `fish` and `grass` end up at cosine 0.95, because as *inputs* every food word does the same job here: it is always followed by `.`. Input embeddings encode what a token does as context, not a dictionary meaning.",
                  "**Regularisation does the clustering.** Without weight decay the loss is equally good (0.978 vs 1.001) but within-category cosine is only 0.243: nothing in the loss penalises the random differences between interchangeable words, so they persist. Weight decay removes components the loss does not need, leaving the shared ones. Large models see each token in vastly more varied contexts, which constrains far more directions.",
                ],
              },
              { t: "h", text: "3. Tokenization effects, measured with a real BPE tokenizer" },
              {
                t: "p",
                text: "We train GPT-2-style byte-level BPE with Hugging Face `tokenizers` on an English corpus that ships with every Python installation — the docstrings of a dozen standard-library modules — and measure tokens per word (**fertility**) for two vocabulary sizes, the cost of one sentence in six languages, and how numbers are split.",
              },
              { t: "code", lang: "python", title: "tokenizer_effects.py", code: TOKMEASURE, output: TOKMEASURE_OUT },
              {
                t: "list",
                items: [
                  "**Vocabulary size vs length**: 1,000 tokens give 1.57 tokens per word (`contain` + `ing`, `e` + `lements`); 4,000 give 1.10, i.e. 30% fewer tokens for the same text and correspondingly cheaper attention.",
                  "**Multilingual cost**: the same sentence costs 29 tokens in English, 37–43 in Spanish, German and French, 66 in Chinese and 174 in Hindi (6.0×). The corpus contains no Devanagari or Chinese, so no merges exist for them and every UTF-8 **byte** becomes a token (Hindi: 174 bytes → 174 tokens). Byte-level BPE never fails, but a model trained with this tokenizer would see six times less Hindi text per context window and pay six times more per sentence. Production multilingual tokenizers reduce, but rarely remove, this gap.",
                  "**Numbers**: `2023` → `20`,`23`; `3.14159` → `3`,`.`,`14`,`15`,`9`; `12345678` → `123456`,`78` but `87654321` → `8`,`76`,`54`,`3`,`2`,`1`. Digit groups depend on which strings happened to be frequent, so the same digit can sit at different positions in different tokens — one reason arithmetic is hard for such models. The LLaMA tokenizer avoids this by splitting all numbers into individual digits.",
                  "Decoding the Hindi ids returns the exact input (round trip `True`): byte-level BPE is lossless.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "`keras.layers.Embedding(input_dim=V, output_dim=d, mask_zero=True)` treats id 0 as padding and produces a mask that mask-aware downstream layers consume automatically. Keras has no built-in tying switch; reuse the layer's `embeddings` variable in the output projection.",
              },
              { t: "code", lang: "python", title: "keras_embedding.py (not executed in CI)", code: KERAS, output: "Prints the embedding shape (2, 5, 64) and the mask [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]; then the logits shape (2, 5, 1000) and the parameter count 68,160 (64,000 shared embedding weights + 4,160 for the Dense layer) — the output projection adds no parameters." },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Choice", "Options", "Advantages", "Disadvantages", "Seen in"],
        rows: [
          ["Token granularity", "Byte, character, subword, word", "Subword: no OOV, short sequences", "Bytes/chars: long sequences; words: OOV", "Subword in almost all LLMs; character-level for small demos"],
          ["Vocabulary size", "~30k – 250k", "Large: shorter sequences, better multilingual coverage", "Large: Vd parameters per matrix, rare rows under-trained", "BERT 30,522; GPT-2 50,257; Llama 3 128,256"],
          ["Embedding / output", "Tied or untied", "Tied: saves Vd params, often better perplexity for small models", "Untied: more flexibility for large models", "GPT-2, BERT tied; Llama untied"],
          ["Embedding scale", "×√d, or small-std init without scaling", "Balances token and position signals", "Easy to mix up conventions when porting", "Original Transformer ×√d; GPT-2 std 0.02"],
          ["Position", "Sinusoidal, learned, RoPE, ALiBi", "RoPE/ALiBi: relative, extendable", "Learned tables cannot extrapolate", "See the positional-encoding lesson"],
          ["Factorised embedding", "V × k lookup then k × d projection (k ≪ d)", "Parameters V k + k d instead of V d", "Extra matmul; less expressive", "ALBERT"],
          ["Continuous inputs", "Linear patch / frame projection", "Same Transformer for any modality", "Patch size trades resolution for sequence length", "ViT (16 × 16 patches), audio Transformers"],
          ["Context extension", "Sliding window attention, RoPE scaling, retrieval", "Longer effective context", "Approximations; extra training", "Long-context LLMs, RAG systems"],
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Every language model's input and output layer** is a token embedding and a projection back to the vocabulary; the vocabulary size directly determines the size of the softmax computed for every generated token.",
          "**API pricing and limits** are defined in tokens, so tokenizer efficiency on a given language or domain (code, chemistry, a non-Latin script) translates directly into cost and into how much text fits into a prompt.",
          "**Long-document systems** — legal, scientific, code-base assistants — combine a model's context window with chunking, overlapping windows and retrieval, as in the strategy table above.",
          "**Embedding-based search** reuses the same idea one level up: sentence or document encoders map text to vectors whose cosine similarity is used for retrieval and clustering ([Multimodal models and RAG](/learn/m15/multimodal)).",
          "**Vision and multimodal models** — ViT patch embeddings, and vision-language models that project image features into the language model's token-embedding space so images can be read as \"tokens\".",
          "**Recommender systems** use embedding tables for user and item ids with exactly the same lookup and sparse-gradient mechanics; these tables can be the largest part of such models, which is why sparse updates matter there.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Cheap and exact**: a lookup costs no arithmetic and its gradient touches only the used rows.",
          "**Learned geometry**: the model chooses where tokens sit, so similarity emerges from use; no hand-made features.",
          "**Uniform interface**: anything that can be cut into pieces and projected becomes a token sequence.",
          "**Tying** halves the vocabulary-related parameters and couples input and output knowledge.",
        ],
      },
      { t: "h", text: "Limitations and trade-offs" },
      {
        t: "list",
        items: [
          "**Vocabulary cost vs sequence length**: larger $V$ shortens sequences but costs $Vd$ (or $2Vd$) parameters and leaves rare rows poorly trained.",
          "**Tokenization is fixed before training**: the model can never see distinctions the tokenizer erased, and inherits its biases — e.g. 6× more tokens for Hindi with an English-trained tokenizer, inconsistent digit grouping.",
          "**Quadratic context cost**: attention compute grows with $n^2$, the KV cache with $n$; long contexts are expensive and not always used well.",
          "**Fixed positions**: a model's usable window is bounded by its positional scheme and training lengths, not only by memory.",
          "**Embedding norms and frequencies**: frequent tokens get many updates and rare ones few; with tying, the input and output roles of a token are forced to share one vector.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Using a different tokenizer at inference than in training.** The ids would index the wrong rows of $E$: the model receives fluent-looking garbage. Always save and load the tokenizer with the model and check the vocabulary size equals the number of embedding rows.",
          "**Id out of range.** Adding special tokens to a tokenizer without resizing the embedding (`model.resize_token_embeddings` in Hugging Face, or a new `nn.Embedding`) raises an index error — or, on GPU, a cryptic device-side assert.",
          "**Forgetting the attention mask with padding.** Padded positions then act as real keys and change the outputs of real tokens; with left-padded generation, it also shifts positions. Pass the mask and exclude padded targets from the loss.",
          "**Manual scatter with `E[ids] += grad`.** With repeated ids, NumPy keeps only one update. Use `np.add.at` (or the framework's embedding backward).",
          "**Default N(0, 1) embedding init with tying and √d scaling.** Logits become huge and the initial loss is in the thousands (measured: 11,875). Initialise with a small std (e.g. 0.02) or drop the scale.",
          "**Comparing token counts across models as if they were words.** A \"4k context\" means 4,096 tokens; how much text that is depends on the tokenizer and language (1.1 tokens/word for in-domain English above, 6× more for Hindi bytes).",
          "**Assuming a longer advertised context means the model uses it well.** Measure retrieval of facts placed at different depths before relying on it.",
          "**Truncating silently.** Tokenizers truncate to the model maximum when asked; for QA or tagging, use overlapping windows instead and merge predictions.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is an embedding layer and why not feed token ids directly?", a: "It is a trainable $V\\times d$ matrix; token id $t$ is replaced by row $t$. Ids are arbitrary labels without order or magnitude, so feeding them as numbers would impose a meaningless structure. Learned vectors let the model place tokens in a space where similar behaviour means nearby vectors." },
      { t: "qa", level: "beginner", q: "Show that an embedding lookup equals a one-hot vector times the embedding matrix.", a: "$(e_t^\\top E)_j = \\sum_v (e_t)_v E_{vj} = E_{tj}$ because only $v=t$ has a non-zero entry. So the layer is a bias-free linear layer on one-hot inputs; indexing computes it without the $V d$ multiply-adds per token." },
      { t: "qa", level: "beginner", q: "What is the context window?", a: "The maximum number of tokens the model processes at once (prompt plus generated tokens for a decoder). It is limited by attention's quadratic compute, the memory for scores and the KV cache, and the lengths the positional scheme and training data cover." },
      { t: "qa", level: "intermediate", q: "How many parameters does weight tying save for V = 50,257 and d = 768, and what does the gradient of a tied matrix look like?", a: "$Vd = 38{,}597{,}376$ — 31% of GPT-2 small's 124M. The tied matrix receives the sum of two gradients: a sparse one from the input side (only rows of tokens in the batch) and a dense one from the output softmax (every row, at every position)." },
      { t: "qa", level: "intermediate", q: "Why did the original Transformer multiply embeddings by √d_model?", a: "The paper only states it. The common explanation: the matrix is shared with the output projection and initialised with variance about $1/d$, so token rows have norm ≈ 1, while sinusoidal position vectors have norm $\\sqrt{d/2}$ (16 at $d=512$). Scaling by $\\sqrt d$ brings token vectors to norm ≈ $\\sqrt d$ so position does not swamp content." },
      { t: "qa", level: "intermediate", q: "A document has 20,000 tokens and your encoder accepts 512. How do you run extractive QA on it?", a: "Split it into overlapping windows of 512 tokens (minus question and special tokens) with a stride such as 384, run the model on each window with the question, and take the best-scoring answer span across windows, preferring spans not at window edges. Overlap ensures every answer appears with context in at least one window. Alternatively retrieve the most relevant chunks first." },
      { t: "qa", level: "advanced", q: "Quantify how attention cost limits context length.", a: "Per layer, $QK^\\top$ and $AV$ cost $4n^2d$ FLOPs versus $24nd^2$ for projections and FFN; attention dominates beyond $n = 6d$ (24,576 for $d = 4096$). For a 7B-class model, going from 2k to 128k tokens raises attention FLOPs 4096-fold, materialised scores would need 1 TiB per layer, and the KV cache reaches 64 GiB per sequence. FlashAttention fixes the score memory, GQA shrinks the cache, but the $n^2$ compute remains — hence sparse/sliding-window attention and retrieval." },
      { t: "qa", level: "advanced", q: "Why do tokenizers make some languages more expensive, and what are the consequences?", a: "BPE merges are learned from corpus frequencies. Languages or scripts that are rare in the tokenizer's training data get few merges and are split into short pieces — down to one token per UTF-8 byte (measured: 174 tokens for a Hindi sentence vs 29 for English). Consequences: higher API cost, less text per context window, more decoding steps, and less effective model capacity per word. Remedies: training the tokenizer on balanced multilingual data, larger vocabularies, or extending the vocabulary and fine-tuning." },
      { t: "qa", level: "advanced", q: "How does a Vision Transformer turn an image into tokens, and what controls sequence length?", a: "It cuts the image into non-overlapping $P\\times P$ patches, flattens each ($P^2C$ values) and applies a learned linear projection to $d$ (equivalently a convolution with kernel and stride $P$), then adds position embeddings and a `[CLS]` token. A 224 × 224 image with $P = 16$ gives 196 tokens; halving $P$ quadruples the token count and therefore multiplies attention cost by 16." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "How many parameters (in millions) does a 32,000 × 512 embedding matrix have?", answer: 16.384, tolerance: 0.001, explain: "$32{,}000\\times512 = 16{,}384{,}000$ = 16.384 M — exactly the saving from tying in the PyTorch example." },
          { q: "Which rows of the embedding matrix receive a non-zero gradient from the input side in one training step?", options: ["All rows", "Only rows of tokens that appear in the batch", "Only the padding row", "Only rows of the most frequent tokens"], answer: 1, explain: "$\\partial\\mathcal{L}/\\partial E = O^\\top G$: row $v$ gets the sum of the gradients at positions holding token $v$; other rows get zero." },
          { kind: "numeric", q: "What is the norm of a sinusoidal positional-encoding vector for d_model = 512?", answer: 16, tolerance: 0.001, explain: "It contains 256 pairs (sin θ, cos θ), each contributing 1 to the squared norm: $\\sqrt{256} = 16$." },
          { kind: "numeric", q: "Attention FLOPs scale with n². By what factor do they grow when the context goes from 8,192 to 32,768 tokens?", answer: 16, tolerance: 0, explain: "$(32768/8192)^2 = 4^2 = 16$ (35.2 → 562.9 TFLOPs in the table)." },
          { q: "An English-trained byte-level BPE tokenizer encodes a Hindi sentence into exactly as many tokens as it has UTF-8 bytes. Why?", options: ["Hindi words are longer", "No merges involving Devanagari bytes were learned, so every byte stays a separate token", "The tokenizer uses <unk> for each character", "Byte-level BPE always uses one token per byte"], answer: 1, explain: "Merges come from corpus frequencies; with no Devanagari in the corpus, only the 256 base byte tokens are available." },
          { kind: "numeric", q: "A ViT uses 14 × 14-pixel patches on a 224 × 224 image. How many patch tokens does it produce (without [CLS])?", answer: 256, tolerance: 0, explain: "$224/14 = 16$ patches per side, $16^2 = 256$ tokens." },
          { q: "What does weight tying share?", options: ["Query and key projections", "The input embedding matrix and the output (pre-softmax) projection", "Embeddings across layers", "Position and token embeddings"], answer: 1, explain: "Logits are computed as $hE^\\top$ with the same $E$ used for the input lookup." },
          { kind: "numeric", q: "Windows of 512 tokens with stride 384 are used on a 2,048-token document, with window starts at 0, 384, 768, … while the start is less than 2048 − 512 + 384. How many windows are there?", answer: 5, tolerance: 0, explain: "Starts 0, 384, 768, 1152, 1536 (the next, 1920, is not below 1920); the last window covers 1536–2047." },
        ],
      },
      {
        t: "exercise",
        title: "Embedding budget of a model",
        prompt: "A decoder has vocabulary 100,000, width 2,048, and 24 Transformer blocks with 12d² parameters each (ignore biases and norms). Compute (a) the embedding parameters, (b) the total with tied and untied output layers, and (c) the fraction of the untied model spent on the two vocabulary matrices.",
        hint: "12d² per block = 4d² attention + 8d² feed-forward with hidden width 4d.",
        solution: [
          { t: "code", lang: "python", title: "budget.py", code: String.raw`V, d, blocks = 100_000, 2048, 24
emb = V * d
body = blocks * 12 * d * d
tied, untied = emb + body, 2 * emb + body
print(f"(a) embedding: {emb:,}")
print(f"(b) blocks: {body:,} | tied total: {tied:,} | untied total: {untied:,}")
print(f"(c) vocabulary matrices in the untied model: {2 * emb / untied:.1%}")`, output: String.raw`(a) embedding: 204,800,000
(b) blocks: 1,207,959,552 | tied total: 1,412,759,552 | untied total: 1,617,559,552
(c) vocabulary matrices in the untied model: 25.3%` },
          { t: "p", text: "The vocabulary matrices are a sizeable share of a model this small; in much wider models the blocks grow with $d^2$ while the embedding grows with $d$, so the share falls." },
        ],
      },
      {
        t: "exercise",
        title: "Measure fertility and the effect of vocabulary size",
        prompt: "Using the `tokenizers` setup of the Framework section, train byte-level BPE on the same docstring corpus with vocabulary sizes 300, 1000 and 3000, and report tokens per word on the held-out sentence and the token count of the German sentence. Explain the trend.",
        hint: "Reuse train_bpe(vocab_size); fertility = len(ids) / number of whitespace-separated words.",
        solution: [
          { t: "code", lang: "python", title: "fertility_sweep.py", code: String.raw`import inspect
import json, collections, email, http, logging, argparse, unittest, typing, pathlib, decimal, fractions, statistics
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

corpus = []
for m in (json, collections, email, http, logging, argparse, unittest, typing, pathlib, decimal, fractions, statistics):
    docs = [inspect.getdoc(m) or ""] + [inspect.getdoc(o) or "" for _, o in inspect.getmembers(m)
                                         if inspect.isclass(o) or inspect.isfunction(o)]
    corpus += [d for d in docs if d]


def train_bpe(vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    return tok


held_out = ("The function returns a new list containing the elements of the sequence in sorted order. "
            "If the argument is missing, the default value is used and a warning is logged.")
german = "Das Wetter ist heute sehr schön, deshalb gehen wir mit unseren Kindern in den Park."
for vs in (300, 1000, 3000):
    tok = train_bpe(vs)
    n = len(tok.encode(held_out).ids)
    print(f"vocab {vs:5d}: English {n} tokens = {n / len(held_out.split()):.2f} tokens/word | "
          f"German sentence {len(tok.encode(german).ids)} tokens")`, output: String.raw`vocab   300: English 112 tokens = 3.73 tokens/word | German sentence 65 tokens
vocab  1000: English 47 tokens = 1.57 tokens/word | German sentence 46 tokens
vocab  3000: English 34 tokens = 1.13 tokens/word | German sentence 42 tokens` },
          { t: "p", text: "More merges mean longer pieces: fertility falls as the vocabulary grows, quickly at first and then slowly, because the remaining merges cover rarer strings. German improves too, but less, since merges were learned from English text — the multilingual gap narrows only when the tokenizer's corpus contains the language." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Text → subword token ids (vocabulary of size $V$) → rows of $E\\in\\mathbb{R}^{V\\times d}$ → (×$\\sqrt d$) + positions → Transformer → logits $hE^\\top$ (if tied) → softmax over $V$.",
          String.raw`Lookup $=$ one-hot $\times E$: $X = OE$; gradient $\partial\mathcal{L}/\partial E = O^\top G$ — only used rows, repeated tokens summed. PyTorch: padding_idx zeroes the pad row's gradient, sparse=True returns sparse gradients.`,
          String.raw`Cost: $Vd$ per vocabulary matrix (38.6M in GPT-2 small = 31%; 525M per matrix in Llama 3 8B). Tying (Press & Wolf, 2017) saves $Vd$.`,
          "Learned geometry reflects behaviour: tokens with similar contexts and continuations get high cosine similarity (0.92 for cat–dog in the trained toy model), helped by weight decay.",
          String.raw`$\sqrt{d}$ scaling balances token rows (norm ≈ 1 at init variance $1/d$) against sinusoidal positions (norm $\sqrt{d/2}$).`,
          String.raw`Context limits: attention $4n^2d$ FLOPs/layer (dominant past $n=6d$), $hn^2$ score memory without fused kernels, KV cache $\propto n$, positional extrapolation, and training data.`,
          "Long inputs: truncation, chunking, overlapping windows with stride, sliding-window attention, retrieval.",
          "Tokenization effects: vocabulary size trades parameters for sequence length; languages absent from the tokenizer corpus can cost several times more tokens; digit grouping is irregular in BPE.",
          "Other modalities enter through the same interface: patches or frames linearly projected to $d$ (ViT: 196 patches of 768 values).",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[The Transformer Architecture](/learn/m11/transformer-architecture) — what happens to the $n\\times d$ input matrix inside the encoder and decoder blocks.",
          "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — how the output softmax over the vocabulary is trained with teacher forcing and used to generate, and the complexity analysis in full.",
          "[BERT, GPT, T5, ViT and Swin](/learn/m11/transformer-families) — how different families use special tokens, tied embeddings and patch embeddings.",
          "[Efficient Inference](/learn/m20/llm-inference) — KV caches, FlashAttention and other techniques that push the context window further.",
        ],
      },
    ],
  },
};

export default lesson;
