"""
train-char-transformer.py -- trains the small character-level Transformer used by the
"Transformer Explorer (trained model)" simulator (components/sims/TransformerSim.tsx) and
exports it to web/public/models/char-transformer.json.

WHAT IT TRAINS
  A decoder-only (GPT-style) Transformer that predicts the next character.
    * data     : web/scripts/data/alice-chapter1.txt -- the opening of Chapter I ("Down the
                 Rabbit-Hole") of Lewis Carroll's "Alice's Adventures in Wonderland" (1865,
                 public domain). Lower-cased, all whitespace collapsed to single spaces.
                 The first 90% of characters are used for training, the last 10% for validation.
    * vocab    : every distinct character of the normalised text (one token per character)
    * d_model  = 48, n_layers = 2, n_heads = 4 (d_head = 12), d_ff = 192 (= 4 * d_model)
    * context  = 64 characters, LEARNED absolute position embeddings (a 64 x 48 table)
    * pre-LN blocks:  x = x + MHA(LN1(x));  x = x + FFN(LN2(x));  final LN before the head
    * FFN activation: GELU, tanh approximation (exactly reproducible in JavaScript)
    * causal (masked) multi-head self-attention, with biases on all projections
    * output head WEIGHT-TIED to the token embedding (logits = h @ E^T), no output bias
    * dropout 0.1 during training only; the exported model has no dropout

EXPORT FORMAT (JSON, see lib/sim/transformer.ts for the reader)
  Every tensor is stored as {"shape": [...], "scale": s, "q": [integers]} and its value is
  q * s (symmetric per-tensor quantisation with |q| <= QMAX, i.e. ~4 significant digits relative
  to the tensor's largest entry). Matrices are stored row-major in "x @ W" orientation
  (shape = [in, out]). After quantising, the de-quantised weights are loaded back into the
  PyTorch model and EVERYTHING reported (losses, verification logits, sample text) is computed
  with exactly the weights the browser receives, so the browser forward pass can be checked
  against PyTorch to floating-point precision.

HOW TO RUN (from the repository root or from web/)
    python web/scripts/train-char-transformer.py
  Requires Python 3.10+ and PyTorch (CPU is fine). Deterministic: fixed seeds, a fixed thread
  count (4) and torch.use_deterministic_algorithms(True), so a re-run on the same machine
  reproduces the file bit for bit. Runtime: about 2 minutes on a laptop CPU (3000 steps).

RESULT OF THE COMMITTED RUN
  61,488 parameters; after 3000 steps: train loss 0.533, held-out loss 2.515 nats/char
  (the held-out loss bottoms out near 2.15 around step 500 and then rises: with only ~6.8k
  training characters the model mostly memorises the chapter, which is what makes its output
  recognisable). Exported JSON: ~310 KB. The browser engine (lib/sim/transformer.ts) reproduces
  the stored PyTorch logits to a max absolute difference of ~5e-6.
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "alice-chapter1.txt"
OUT = HERE.parent / "public" / "models" / "char-transformer.json"

# ----------------------------------------------------------------- hyper-parameters
D_MODEL, N_LAYERS, N_HEADS, D_FF, CONTEXT = 48, 2, 4, 192, 64
DROPOUT = 0.1
BATCH, STEPS, LR_MAX, LR_MIN, WARMUP, WEIGHT_DECAY = 32, 3000, 3e-3, 1e-4, 200, 0.01
QMAX = 9999          # quantisation levels per sign: value = q * (max|w| / QMAX)
SEED = 0

torch.manual_seed(SEED)
torch.set_num_threads(4)                     # fixed thread count -> bit-reproducible on a given machine
torch.use_deterministic_algorithms(True)

# ----------------------------------------------------------------- data
raw = DATA.read_text(encoding="utf-8")
text = " ".join(raw.lower().split())          # lower-case, collapse whitespace
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
V = len(vocab)


def get_batch(src, gen):
    ix = torch.randint(0, len(src) - CONTEXT - 1, (BATCH,), generator=gen)
    x = torch.stack([src[i:i + CONTEXT] for i in ix])
    y = torch.stack([src[i + 1:i + CONTEXT + 1] for i in ix])
    return x, y


# ----------------------------------------------------------------- model
class Block(nn.Module):
    """Pre-LN decoder block: x + MHA(LN(x)), then x + FFN(LN(x))."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

    def attn(self, h):
        B, T, _ = h.shape
        dh = D_MODEL // N_HEADS
        split = lambda t: t.view(B, T, N_HEADS, dh).transpose(1, 2)          # (B, H, T, dh)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)                     # (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)    # True above the diagonal
        w = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        z = (self.drop(w) @ v).transpose(1, 2).reshape(B, T, D_MODEL)        # concat heads
        return self.o(z)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.ln2(x)), approximate="tanh")))
        return x


class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        for p in self.parameters():                      # GPT-2-style small init
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T)))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.tok.weight.T          # weight-tied output head


@torch.no_grad()
def eval_loss(model, src):
    """Mean next-character cross-entropy over every full window of the split (stride CONTEXT/2)."""
    was_training = model.training
    model.eval()
    losses = []
    for i in range(0, len(src) - CONTEXT - 1, CONTEXT // 2):
        x = src[i:i + CONTEXT][None]
        y = src[i + 1:i + CONTEXT + 1][None]
        losses.append(F.cross_entropy(model(x).view(-1, V), y.view(-1)).item())
    model.train(was_training)                    # restore the previous mode (dropout on/off)
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model, prompt, n):
    model.eval()
    ids = [stoi[c] for c in prompt]
    for _ in range(n):
        x = torch.tensor(ids[-CONTEXT:])[None]
        ids.append(int(model(x)[0, -1].argmax()))
    return "".join(vocab[i] for i in ids)


model = CharTransformer()
n_params = sum(p.numel() for p in model.parameters())
print(f"text: {len(text)} chars ({n_train} train / {len(text) - n_train} val), vocab {V}: {''.join(vocab)!r}")
print(f"parameters: {n_params}")

opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)
gen = torch.Generator().manual_seed(SEED)
t0 = time.time()
history = []
for step in range(STEPS + 1):
    lr = LR_MAX * step / WARMUP if step < WARMUP else LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch(train_data, gen)
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        tr, va = eval_loss(model, train_data), eval_loss(model, val_data)
        history.append({"step": step, "train": round(tr, 4), "val": round(va, 4)})
        print(f"step {step:5d}  lr {lr:.2e}  batch loss {loss.item():.4f}  train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.0f}s)")

# ----------------------------------------------------------------- quantise + export
model.eval()
tensors = {}


def q(name, t):
    """Quantise a tensor, store it, and return the de-quantised copy."""
    t = t.detach().float()
    scale = (t.abs().max().item() / QMAX) or 1.0
    qi = torch.round(t / scale).to(torch.int64)
    tensors[name] = {"shape": list(t.shape), "scale": float(f"{scale:.9e}"), "q": qi.flatten().tolist()}
    return (qi.double() * scale).float()


with torch.no_grad():
    model.tok.weight.copy_(q("tok_emb", model.tok.weight))
    model.pos.weight.copy_(q("pos_emb", model.pos.weight))
    for l, b in enumerate(model.blocks):
        p = f"layers.{l}."
        for ln in ("ln1", "ln2"):
            getattr(b, ln).weight.copy_(q(p + ln + ".g", getattr(b, ln).weight))
            getattr(b, ln).bias.copy_(q(p + ln + ".b", getattr(b, ln).bias))
        for name in ("q", "k", "v", "o"):
            lin = getattr(b, name)
            lin.weight.copy_(q(p + "W" + name, lin.weight.T).T)    # stored as (in, out)
            lin.bias.copy_(q(p + "b" + name, lin.bias))
        b.ff1.weight.copy_(q(p + "W1", b.ff1.weight.T).T)
        b.ff1.bias.copy_(q(p + "b1", b.ff1.bias))
        b.ff2.weight.copy_(q(p + "W2", b.ff2.weight.T).T)
        b.ff2.bias.copy_(q(p + "b2", b.ff2.bias))
    model.ln_f.weight.copy_(q("ln_f.g", model.ln_f.weight))
    model.ln_f.bias.copy_(q("ln_f.b", model.ln_f.bias))

tr_q, va_q = eval_loss(model, train_data), eval_loss(model, val_data)
print(f"after quantisation: train {tr_q:.4f}  val {va_q:.4f}")

# verification vectors: logits computed by PyTorch with the exported (de-quantised) weights
model.eval()                                     # no dropout: the browser runs the inference-mode model
verify = []
with torch.no_grad():
    for prompt in ["alice was beginning to get very", text[n_train - 20:n_train + 44]]:
        ids = [stoi[c] for c in prompt]
        logits = model(torch.tensor(ids)[None])[0]                 # (T, V)
        verify.append({"prompt": prompt, "positions": [0, len(ids) // 2, len(ids) - 1],
                       "logits": [[float(f"{v:.7g}") for v in logits[p].tolist()] for p in (0, len(ids) // 2, len(ids) - 1)]})

samples = []
for prompt in ["alice was ", "the rabbit ", "down, down, down. ", "she "]:
    samples.append({"prompt": prompt, "greedy": generate(model, prompt, 120)})
    print(f"greedy {prompt!r:24} -> {samples[-1]['greedy']!r}")

export = {
    "format": "char-transformer/1",
    "description": "Decoder-only character-level Transformer trained on the opening of Chapter I of Lewis Carroll's 'Alice's Adventures in Wonderland' (1865, public domain).",
    "config": {"vocab_size": V, "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS, "d_ff": D_FF,
               "context": CONTEXT, "positions": "learned", "norm": "pre-LN", "ln_eps": 1e-5,
               "activation": "gelu_tanh", "tied_head": True},
    "vocab": "".join(vocab),
    "n_params": n_params,
    "training": {"chars": len(text), "train_chars": n_train, "val_chars": len(text) - n_train, "steps": STEPS,
                 "batch": BATCH, "optimizer": f"AdamW(lr {LR_MAX} cosine to {LR_MIN}, warmup {WARMUP}, wd {WEIGHT_DECAY})",
                 "dropout": DROPOUT, "seed": SEED, "history": history,
                 "final_train_loss": round(tr_q, 4), "final_val_loss": round(va_q, 4)},
    "quantisation": f"value = q * scale per tensor, |q| <= {QMAX}",
    "tensors": tensors,
    "verify": verify,
    "samples": samples,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
print(f"wrote {OUT.name}: {OUT.stat().st_size / 1024:.1f} KB in {time.time() - t0:.0f}s")
