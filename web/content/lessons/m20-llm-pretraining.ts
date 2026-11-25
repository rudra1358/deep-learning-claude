import type { Lesson } from "@/lib/types";

/* ------------------------------------------------------------------ code blocks (executed; outputs are real) */

const SCRATCH = String.raw`import hashlib
import re
from collections import Counter

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. a miniature data pipeline
docs = [
    "Alice was beginning to get very tired of sitting by her sister on the bank.",
    "alice was beginning to get very tired of sitting by her sister on the bank. ",   # exact dup after normalising
    "Alice was beginning to get very tired of sitting by her sister on the river bank.",  # near duplicate
    "BUY NOW!!! cheap cheap cheap cheap cheap cheap",                                  # low-quality spam
    "The Rabbit took a watch out of its waistcoat-pocket, and looked at it.",
    "def f(x):\n    return x * 2  # a tiny piece of code",
    "ok",                                                                             # too short
]


def normalise(s):
    return " ".join(s.lower().split())


def quality_ok(s):
    words = s.split()
    if len(words) < 5:                                       # too short to be useful
        return False
    top_share = Counter(words).most_common(1)[0][1] / len(words)
    alpha = sum(c.isalpha() for c in s) / len(s)
    return top_share < 0.3 and alpha > 0.5                   # not repetitive, mostly letters


def shingles(s, n=5):
    return {s[i:i + n] for i in range(len(s) - n + 1)}       # character 5-grams


def minhash(sh, n_hash=64):
    """For each of n_hash salted hash functions keep the minimum hash over the shingles."""
    return np.array([min(int(hashlib.md5(f"{k}:{x}".encode()).hexdigest()[:8], 16) for x in sh) for k in range(n_hash)])


seen, kept, sigs = set(), [], []
for d in docs:
    n = normalise(d)
    h = hashlib.sha256(n.encode()).hexdigest()
    if h in seen:
        print(f"exact duplicate   : {n[:45]!r}"); continue
    seen.add(h)
    if not quality_ok(n):
        print(f"quality filter    : {n[:45]!r}"); continue
    sig = minhash(shingles(n))
    est = max(((sig == s).mean() for s in sigs), default=0.0)   # P(min-hashes agree) = Jaccard similarity
    if est > 0.7:
        print(f"near duplicate    : {n[:45]!r} (estimated Jaccard {est:.2f})"); continue
    kept.append(n); sigs.append(sig)
a, b = shingles(normalise(docs[0])), shingles(normalise(docs[2]))
print(f"kept {len(kept)} of {len(docs)} documents; true Jaccard(doc0, doc2) = {len(a & b) / len(a | b):.2f}")

# data mixture: sample sources in proportion to weights, not to their raw sizes
sources = {"web": 900_000, "books": 60_000, "code": 30_000, "wiki": 10_000}   # tokens available
weights = {"web": 0.6, "books": 0.2, "code": 0.15, "wiki": 0.05}
budget = 500_000
for s in sources:
    take = weights[s] * budget
    print(f"{s:5s}: {take:9,.0f} tokens = {take / sources[s]:.2f} epochs of the source")

# ---------------------------------------------------------------- 2. the modern block components
d, T = 8, 6
x = rng.normal(size=(T, d))


def rms_norm(x, g, eps=1e-6):
    return g * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def layer_norm(x, eps=1e-6):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


y = rms_norm(x, np.ones(d))
print("RMSNorm rows: rms =", np.sqrt((y ** 2).mean(-1)).round(4)[:3], " mean =", y.mean(-1).round(3)[:3])
xc = x - x.mean(-1, keepdims=True)
print("RMSNorm == LayerNorm on zero-mean input:", np.allclose(rms_norm(xc, np.ones(d)), layer_norm(xc)))


def rope(v, pos, base=10000.0):
    """Rotate consecutive halves (v_i, v_{i+d/2}) by angle pos * base^(-2i/d)."""
    half = v.shape[-1] // 2
    ang = pos * base ** (-np.arange(half) * 2 / v.shape[-1])
    v1, v2 = v[:half], v[half:]
    return np.concatenate([v1 * np.cos(ang) - v2 * np.sin(ang), v1 * np.sin(ang) + v2 * np.cos(ang)])


q, k = rng.normal(size=d), rng.normal(size=d)
for m, n in [(5, 3), (12, 10), (102, 100)]:                  # all with offset m - n = 2
    print(f"q at {m:3d}, k at {n:3d}: rope(q)·rope(k) = {rope(q, m) @ rope(k, n):.6f}")
print(f"norm preserved: |q| = {np.linalg.norm(q):.6f}, |rope(q, 7)| = {np.linalg.norm(rope(q, 7)):.6f}")


def silu(z):
    return z / (1 + np.exp(-z))


dm = 4096                                                    # parameter budget of the feed-forward block
gelu_mlp = 2 * dm * 4 * dm
hidden = int(8 * dm / 3)
swiglu = 3 * dm * hidden
print(f"FFN params at d={dm}: GELU 4d MLP {gelu_mlp:,} vs SwiGLU (hidden 8d/3 = {hidden}) {swiglu:,}")
W_g, W_u, W_d = rng.normal(size=(d, 12)), rng.normal(size=(d, 12)), rng.normal(size=(12, d))
print("SwiGLU output shape:", (silu(x @ W_g) * (x @ W_u) @ W_d).shape)

# ---------------------------------------------------------------- 3. schedule and compute budget
peak, floor, warm, total = 3e-4, 3e-5, 2000, 100_000


def lr(s):
    if s < warm:
        return peak * (s + 1) / warm
    return floor + 0.5 * (peak - floor) * (1 + np.cos(np.pi * (s - warm) / (total - warm)))


print("lr at steps 0, 999, 1999, 51000, 99999:", [f"{lr(s):.2e}" for s in (0, 999, 1999, 51000, 99999)])
for N, D in [(1e9, 20e9), (7e9, 140e9), (70e9, 1.4e12)]:
    C = 6 * N * D
    print(f"N = {N:.0e} params, D = {D:.1e} tokens (D/N = {D / N:.0f}): C = 6ND = {C:.2e} FLOPs"
          f" = {C / (312e12 * 0.4) / 86400:,.1f} GPU-days at 312 TFLOP/s and 40% utilisation")`;

const SCRATCH_OUT = String.raw`exact duplicate   : 'alice was beginning to get very tired of sitt'
near duplicate    : 'alice was beginning to get very tired of sitt' (estimated Jaccard 0.80)
quality filter    : 'buy now!!! cheap cheap cheap cheap cheap chea'
quality filter    : 'ok'
kept 3 of 7 documents; true Jaccard(doc0, doc2) = 0.85
web  :   300,000 tokens = 0.33 epochs of the source
books:   100,000 tokens = 1.67 epochs of the source
code :    75,000 tokens = 2.50 epochs of the source
wiki :    25,000 tokens = 2.50 epochs of the source
RMSNorm rows: rms = [1. 1. 1.]  mean = [ 0.535 -0.79   0.265]
RMSNorm == LayerNorm on zero-mean input: True
q at   5, k at   3: rope(q)·rope(k) = -3.364601
q at  12, k at  10: rope(q)·rope(k) = -3.364601
q at 102, k at 100: rope(q)·rope(k) = -3.364601
norm preserved: |q| = 2.968184, |rope(q, 7)| = 2.968184
FFN params at d=4096: GELU 4d MLP 134,217,728 vs SwiGLU (hidden 8d/3 = 10922) 134,209,536
SwiGLU output shape: (6, 8)
lr at steps 0, 999, 1999, 51000, 99999: ['1.50e-07', '1.50e-04', '3.00e-04', '1.65e-04', '3.00e-05']
N = 1e+09 params, D = 2.0e+10 tokens (D/N = 20): C = 6ND = 1.20e+20 FLOPs = 11.1 GPU-days at 312 TFLOP/s and 40% utilisation
N = 7e+09 params, D = 1.4e+11 tokens (D/N = 20): C = 6ND = 5.88e+21 FLOPs = 545.3 GPU-days at 312 TFLOP/s and 40% utilisation
N = 7e+10 params, D = 1.4e+12 tokens (D/N = 20): C = 6ND = 5.88e+23 FLOPs = 54,531.7 GPU-days at 312 TFLOP/s and 40% utilisation`;

const TORCH = String.raw`import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)                         # single thread: bit-identical results on every run

# The first ~4,200 characters of Chapter I of Lewis Carroll's "Alice's Adventures in Wonderland" (1865, public domain)
TEXT = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having "
    "nothing to do: once or twice she had peeped into the book her sister was reading, but it had no "
    "pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without "
    "pictures or conversations?' So she was considering in her own mind (as well as she could, for "
    "the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain "
    "would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit "
    "with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice "
    "think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I "
    "shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have "
    "wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually "
    "took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started "
    "to her feet, for it flashed across her mind that she had never before seen a rabbit with either "
    "a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the"
    " field after it, and fortunately was just in time to see it pop down a large rabbit-hole under "
    "the hedge. In another moment down went Alice after it, never once considering how in the world "
    "she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then "
    "dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself "
    "before she found herself falling down a very deep well. Either the well was very deep, or she "
    "fell very slowly, for she had plenty of time as she went down to look about her and to wonder "
    "what was going to happen next. First, she tried to look down and make out what she was coming "
    "to, but it was too dark to see anything; then she looked at the sides of the well, and noticed "
    "that they were filled with cupboards and book-shelves; here and there she saw maps and pictures "
    "hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "
    "'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the "
    "jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as "
    "she fell past it. 'Well!' thought Alice to herself, 'after such a fall as this, I shall think "
    "nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say "
    "anything about it, even if I fell off the top of the house!' (Which was very likely true.) Down,"
    " down, down. Would the fall never come to an end? 'I wonder how many miles I've fallen by this "
    "time?' she said aloud. 'I must be getting somewhere near the centre of the earth. Let me see: "
    "that would be four thousand miles down, I think--' (for, you see, Alice had learnt several "
    "things of this sort in her lessons in the schoolroom, and though this was not a very good "
    "opportunity for showing off her knowledge, as there was no one to listen to her, still it was "
    "good practice to say it over) '--yes, that's about the right distance--but then I wonder what "
    "Latitude or Longitude I've got to?' (Alice had no idea what Latitude was, or Longitude either, "
    "but thought they were nice grand words to say.) Presently she began again. 'I wonder if I shall "
    "fall right through the earth! How funny it'll seem to come out among the people that walk with "
    "their heads downward! The Antipathies, I think--' (she was rather glad there was no one "
    "listening, this time, as it didn't sound at all the right word) '--but I shall have to ask them "
    "what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?' "
    "(and she tried to curtsey as she spoke--fancy curtseying as you're falling through the air! Do "
    "you think you could manage it?) 'And what an ignorant little girl she'll think me for asking! "
    "No, it'll never do to ask: perhaps I shall see it written up somewhere.' Down, down, down. There"
    " was nothing else to do, so Alice soon began talking again. 'Dinah'll miss me very much "
    "to-night, I should think!' (Dinah was the cat.)"
)

# ---------------------------------------------------------------- data: char-level tokens, 90/10 split
text = TEXT.lower()
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
data = torch.tensor([stoi[c] for c in text])
n_train = int(0.9 * len(data))
train, val = data[:n_train], data[n_train:]
CTX, BATCH, STEPS, WARMUP = 64, 32, 300, 30
LR_MAX, LR_MIN = 2e-3, 2e-4


def batch(src, g):
    ix = torch.randint(0, len(src) - CTX - 1, (BATCH,), generator=g)
    x = torch.stack([src[i:i + CTX] for i in ix])
    return x, torch.stack([src[i + 1:i + CTX + 1] for i in ix])       # targets = inputs shifted by one


# ---------------------------------------------------------------- the modern decoder block
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):                        # no mean subtraction, no bias
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, d_head): rotate pairs (x_i, x_{i+d/2}) by angle t * base^(-2i/d)."""
    T, dh = x.shape[-2], x.shape[-1]
    inv = base ** (-torch.arange(0, dh // 2).float() * 2 / dh)
    ang = torch.arange(T).float()[:, None] * inv[None, :]               # (T, d_head/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        hidden = int(8 * d / 3)                  # SwiGLU width: keeps parameters ~ a 4d GELU MLP
        self.w_gate, self.w_up = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))             # pre-norm residual
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))  # SwiGLU feed-forward


class TinyGPT(nn.Module):
    def __init__(self, V, d=48, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)   # small init: tied logits start near uniform
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d)

    def forward(self, idx):
        x = self.emb(idx)                        # no position table: RoPE inside attention
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.emb.weight.T  # weight-tied output head


model = TinyGPT(len(vocab))
N = sum(p.numel() for p in model.parameters())
decay = [p for p in model.parameters() if p.dim() == 2]                 # no weight decay on norm gains
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=LR_MAX, betas=(0.9, 0.95))


def lr_at(s):                                    # linear warm-up, then cosine decay to LR_MIN
    if s < WARMUP:
        return LR_MAX * (s + 1) / WARMUP
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP)))


@torch.no_grad()
def evaluate(src):
    model.eval()
    losses = []
    for i in range(0, len(src) - CTX - 1, CTX):  # every non-overlapping window
        x, y = src[i:i + CTX][None], src[i + 1:i + CTX + 1][None]
        losses.append(F.cross_entropy(model(x).view(-1, len(vocab)), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


print(f"{len(text):,} characters, vocab {len(vocab)}, {N:,} parameters; uniform-guess loss ln {len(vocab)} = {math.log(len(vocab)):.3f}")
g = torch.Generator().manual_seed(0)
for s in range(STEPS + 1):
    if s % 50 == 0:
        tl, vl = evaluate(train), evaluate(val)
        print(f"step {s:3d}  lr {lr_at(min(s, STEPS - 1)):.5f}  train {tl:.3f} (ppl {math.exp(tl):5.2f})  val {vl:.3f} (ppl {math.exp(vl):5.2f})")
    if s == STEPS:
        break
    for group in opt.param_groups:
        group["lr"] = lr_at(s)
    x, y = batch(train, g)
    loss = F.cross_entropy(model(x).view(-1, len(vocab)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # global-norm clipping
    opt.step()

D = STEPS * BATCH * CTX
print(f"tokens processed D = {D:,}; training compute C = 6ND = {6 * N * D:.3e} FLOPs; last grad norm {gnorm:.3f}")


@torch.no_grad()
def greedy(prompt, n):
    ids = [stoi[c] for c in prompt]
    for _ in range(n):
        ids.append(int(model(torch.tensor([ids[-CTX:]]))[0, -1].argmax()))
    return "".join(vocab[i] for i in ids)


model.eval()
print("sample:", greedy("alice was ", 80))`;

const TORCH_OUT = String.raw`4,179 characters, vocab 37, 57,312 parameters; uniform-guess loss ln 37 = 3.611
step   0  lr 0.00007  train 3.654 (ppl 38.62)  val 3.645 (ppl 38.28)
step  50  lr 0.00198  train 2.717 (ppl 15.13)  val 2.851 (ppl 17.30)
step 100  lr 0.00172  train 2.245 (ppl  9.44)  val 2.533 (ppl 12.58)
step 150  lr 0.00126  train 1.996 (ppl  7.36)  val 2.441 (ppl 11.49)
step 200  lr 0.00074  train 1.803 (ppl  6.07)  val 2.460 (ppl 11.71)
step 250  lr 0.00035  train 1.670 (ppl  5.31)  val 2.463 (ppl 11.74)
step 300  lr 0.00020  train 1.603 (ppl  4.97)  val 2.496 (ppl 12.14)
tokens processed D = 614,400; training compute C = 6ND = 2.113e+11 FLOPs; last grad norm 1.229
sample: alice was the to the was the the the the to the as the was the wat the was the was the the`;

const KERAS = String.raw`import tensorflow as tf
from tensorflow import keras


class RMSNorm(keras.layers.Layer):
    def build(self, shape):
        self.g = self.add_weight(shape=(shape[-1],), initializer="ones", name="g")

    def call(self, x):
        return self.g * x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), -1, keepdims=True) + 1e-6)


class SwiGLU(keras.layers.Layer):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = keras.layers.Dense(hidden, use_bias=False)
        self.up = keras.layers.Dense(hidden, use_bias=False)
        self.down = keras.layers.Dense(d, use_bias=False)

    def call(self, x):
        return self.down(tf.nn.silu(self.gate(x)) * self.up(x))


# warm-up + cosine schedule and AdamW with gradient clipping
schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0, decay_steps=270, alpha=0.1,
                                                  warmup_target=2e-3, warmup_steps=30)
opt = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=0.1, beta_1=0.9, beta_2=0.95, global_clipnorm=1.0)
x = tf.random.normal((2, 8, 48))
print(SwiGLU(48, 128)(RMSNorm()(x)).shape)`;

/* ------------------------------------------------------------------ lesson */

const lesson: Lesson = {
  id: "m20/llm-pretraining",
  title: "Pretraining Large Language Models",
  summary:
    "How a decoder-only Transformer becomes a large language model: the next-token objective, the data pipeline (crawl, filter, deduplicate, tokenize, mix), the modern architecture recipe (pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings), the optimisation recipe (AdamW, warm-up + cosine, clipping, mixed precision), the compute budget C ≈ 6ND, and how to read loss and perplexity curves — ending with a tiny GPT pretrained for real on public-domain text.",
  minutes: 75,
  level: "advanced",
  prerequisites: ["m11/transformer-architecture", "m11/training-inference", "m11/tokens-embeddings", "m13/language-modeling", "m06/lr-schedules"],
  next: ["m20/decoding", "m20/alignment", "m20/llm-inference", "m19/scaling-efficiency"],
  terms: ["pretraining", "next-token-prediction", "data-mixture", "deduplication", "minhash", "rope", "swiglu", "compute-optimal-training", "base-model"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "**Pretraining** is the expensive first phase of building a large language model (LLM): a decoder-only Transformer is trained to predict the next token on hundreds of billions to trillions of tokens of text. Nothing else is asked of it — no labels, no task — yet to predict the next token of arbitrary documents well, the model must absorb spelling, grammar, facts, styles, code syntax and some reasoning patterns. The result is a **base model**: an excellent continuer of documents that later stages ([Alignment](/learn/m20/alignment)) turn into an assistant." },
      { t: "p", text: "The recipe was established step by step: GPT (Radford et al., 2018) showed that generative pretraining followed by fine-tuning works; GPT-2 (Radford et al., 2019, *Language Models are Unsupervised Multitask Learners*) scaled it to 1.5B parameters on web text; GPT-3 (Brown et al., 2020, *Language Models are Few-Shot Learners*) at 175B parameters showed in-context learning; scaling-law studies (Kaplan et al., 2020; Hoffmann et al., 2022) turned model and data sizing into a quantitative decision; and open models such as LLaMA (Touvron et al., 2023) fixed the now-standard architectural tweaks." },
      { t: "p", text: "The Transformer itself is taught in [Transformer Architecture](/learn/m11/transformer-architecture), the causal-LM loss and the basic optimiser recipe in [Training, Autoregressive Generation and Complexity](/learn/m11/training-inference), and scaling-law mathematics in [Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency). This lesson is about what changes at LLM scale: the **data**, the **architecture details**, the **compute budget** and the **evaluation**." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Transformer Architecture](/learn/m11/transformer-architecture) — decoder blocks, pre-LN vs post-LN, weight tying and the parameter count $\\approx 12d^2$ per layer.",
        "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — the causal-LM loss, perplexity, AdamW, warm-up + decay, gradient clipping and mixed precision.",
        "[Tokenization, Embeddings and Context Windows](/learn/m11/tokens-embeddings) — BPE tokenizers and context windows.",
        "[Language Modeling](/learn/m13/language-modeling) — perplexity and bits per character.",
        "[Learning-Rate Schedules](/learn/m06/lr-schedules) and [Normalization](/learn/m07/normalization) — warm-up/cosine schedules and LayerNorm, which RMSNorm simplifies.",
        "[Positional Encoding](/learn/m10/positional-encoding) — absolute positions, which RoPE replaces.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "Learning a language by finishing other people's sentences", text: "Imagine learning everything you know by reading an enormous library with one exercise only: cover the next word, guess it, uncover, adjust. At first you learn that 'the' is common. Then that adjectives come before nouns. To guess the next word of a chemistry textbook you must learn chemistry; to guess the next line of a program you must learn to program. Pretraining is this exercise at the scale of the internet. The **data pipeline** is the librarian deciding which books are worth reading (and throwing away the hundredth copy of the same spam page); the **data mixture** decides how much time you spend on novels, encyclopedias and code; the **compute budget** is how many hours you have — and the question of whether to grow a bigger brain or read more books with it is what compute-optimal training answers." },
      { t: "p", text: "Why does such a simple objective work? Because the loss can only keep falling if the model captures ever more subtle structure. Early in training, loss drops fast as frequency and local grammar are learned; later, every hundredth of a nat requires knowledge or reasoning that appears in a small fraction of documents. That is also why pretraining is so expensive: the easy gains come first." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "The data pipeline" },
      { t: "steps", items: [
        { title: "Collect", text: "Web crawls (Common Crawl snapshots), books, encyclopedias, scientific papers, code repositories, forums. Raw crawl data is mostly boilerplate, navigation menus, spam and duplicates." },
        { title: "Extract and identify language", text: "Strip HTML to text; detect language with a classifier; keep the target languages." },
        { title: "Quality filtering", text: "Heuristics (document length, fraction of alphabetic characters, repeated lines or words, 'lorem ipsum', bad-word lists — the C4 corpus of Raffel et al., 2020 used rules of this kind) and model-based filters (a classifier trained to distinguish reference-quality text, or the perplexity of a small language model)." },
        { title: "Deduplicate", text: "Exact duplicates by hashing normalised text; near-duplicates with **MinHash** (Broder, 1997) over character or word n-gram shingles and locality-sensitive hashing. Lee et al. (2022, *Deduplicating Training Data Makes Language Models Better*) found that deduplication reduces memorised output and wasted compute." },
        { title: "Decontaminate and clean", text: "Remove benchmark test sets (so evaluations are not inflated) and personal information where possible." },
        { title: "Tokenize", text: "Train a BPE or similar tokenizer on a sample (see [Text Preprocessing](/learn/m13/text-preprocessing)); convert everything to token ids; concatenate documents separated by an end-of-document token and cut into fixed-length training sequences ('packing')." },
        { title: "Mix", text: "Sample sources by chosen weights, not raw sizes: high-quality sources (books, encyclopedias, code) are **up-sampled** to several epochs, while the huge web source is seen less than once. The weights are tuned with small proxy models." },
      ] },
      { t: "h", text: "The modern architecture recipe (decoder-only)" },
      { t: "list", items: [
        "**Pre-norm** blocks $x \\leftarrow x + F(\\mathrm{Norm}(x))$ with a final norm before the output head — stable without long warm-ups (see [Transformer Architecture](/learn/m11/transformer-architecture)).",
        "**RMSNorm** (Zhang and Sennrich, 2019) instead of LayerNorm: rescale by the root-mean-square, no mean subtraction, no bias — cheaper, and empirically as good.",
        "**RoPE** — rotary position embedding (Su et al., 2021, *RoFormer*): rotate query and key vectors by position-dependent angles, so attention scores depend on relative position; no learned position table.",
        "**SwiGLU** feed-forward (Shazeer, 2020, *GLU Variants Improve Transformer*): $W_{\\text{down}}(\\mathrm{SiLU}(xW_{\\text{gate}}) \\odot xW_{\\text{up}})$ with hidden width $\\approx \\tfrac{8}{3}d$ to keep the parameter count of a $4d$ MLP.",
        "**No biases** in linear layers, often **tied** input/output embeddings for small models, and **grouped-query attention** for cheaper inference ([Efficient Inference](/learn/m20/llm-inference)).",
        "LLaMA (Touvron et al., 2023) combined pre-norm RMSNorm, SwiGLU and RoPE; most open LLMs since follow this template.",
      ] },
      { t: "h", text: "The optimisation recipe" },
      { t: "steps", items: [
        { title: "Optimizer", text: "AdamW (Loshchilov and Hutter, 2019) with $\\beta_2 \\approx 0.95$, weight decay $\\approx 0.1$ on matrices only (not norms/biases)." },
        { title: "Schedule", text: "Linear warm-up over the first few thousand steps, then cosine decay to about 10% of the peak learning rate at the end of the planned run." },
        { title: "Stability", text: "Global gradient-norm clipping at 1.0; loss spikes are handled by skipping batches or restarting from a checkpoint with a lower learning rate." },
        { title: "Precision and parallelism", text: "bf16 mixed precision (Micikevicius et al., 2018 introduced mixed-precision training with fp16 and loss scaling), with data, tensor and pipeline parallelism and sharded optimizer states across thousands of accelerators." },
        { title: "Batch", text: "Millions of tokens per optimizer step (sequences × context length), often ramped up during training." },
        { title: "Monitor", text: "Training loss, held-out loss on several domains, and periodic downstream evaluations; checkpoints are saved regularly." },
      ] },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "The objective" },
      { t: "math", tex: String.raw`\mathcal L(\theta) = -\frac{1}{|\mathcal D|}\sum_{x\in\mathcal D}\ \sum_{t=1}^{|x|} \log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = \exp(\mathcal L)`, caption: "$\\mathcal D$ = training tokens, $x_t$ = token at position $t$; the loss is in nats per token. Derivation and the one-pass causal-mask trick: [Training, Autoregressive Generation and Complexity](/learn/m11/training-inference)." },
      { t: "p", text: String.raw`Perplexity is only comparable between models **with the same tokenizer**: a model with a larger vocabulary predicts fewer, harder tokens. For tokenizer-independent comparisons use bits per byte (or per character): $\text{bits/char} = \mathcal L_{\text{token}} \cdot (\text{tokens}/\text{chars}) / \ln 2$. Worked example from the training run below: held-out loss 2.441 nats/char at step 150 gives perplexity $e^{2.441} = 11.48$ and $2.441/\ln 2 = 3.52$ bits/char, against $\ln 37 = 3.611$ nats ($\mathrm{PPL} = 37$) for uniform guessing over the 37-character vocabulary.` },
      { t: "h", text: "RMSNorm" },
      { t: "math", tex: String.raw`\mathrm{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\frac1d\sum_{i=1}^d x_i^2 + \epsilon}}`, caption: "$x \\in \\mathbb R^d$ one token's vector, $g$ learned gain, $\\epsilon$ small constant. Equal to LayerNorm (without bias) when $x$ already has zero mean." },
      { t: "p", text: String.raw`Example: $x = (1, 2, 3, 4)$, $\mathrm{rms} = \sqrt{30/4} = 2.738613$, so with $g = 1$ the output is $(0.365148,\ 0.730297,\ 1.095445,\ 1.460593)$, whose root-mean-square is 1.` },
      { t: "h", text: "Rotary position embedding" },
      { t: "p", text: String.raw`Split a $d_{\text{head}}$-dimensional query into $d_{\text{head}}/2$ pairs $(u_i, v_i)$ and rotate pair $i$ at position $m$ by angle $m\theta_i$ with $\theta_i = b^{-2i/d_{\text{head}}}$ ($b = 10{,}000$):` },
      { t: "math", tex: String.raw`\begin{pmatrix}u_i'\\ v_i'\end{pmatrix} = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}\begin{pmatrix}u_i\\ v_i\end{pmatrix}, \qquad \langle R_m q,\ R_n k\rangle = \langle q,\ R_{n-m} k\rangle` },
      { t: "p", text: String.raw`Because rotations are orthogonal ($R_m^\top R_n = R_{n-m}$), the attention score between a query at $m$ and a key at $n$ depends only on the offset $n-m$, and vector norms are unchanged. One-pair example with $q = k = (1, 0)$, $\theta = 0.5$, $m=3$, $n=1$: the score is $\cos((3-1)\cdot0.5) = \cos 1 = 0.540302$, the same for $m=13, n=11$. Low-index pairs rotate fast (fine position), high-index pairs slowly (coarse position). The code below confirms the relative-position property numerically.` },
      { t: "h", text: "SwiGLU" },
      { t: "math", tex: String.raw`\mathrm{FFN}_{\text{SwiGLU}}(x) = \big(\mathrm{SiLU}(xW_{g}) \odot xW_{u}\big)W_{d}, \qquad \mathrm{SiLU}(z) = z\,\sigma(z)`, caption: "$W_g, W_u \\in \\mathbb R^{d\\times h}$, $W_d \\in \\mathbb R^{h\\times d}$: $3dh$ parameters; $h = 8d/3$ matches a GELU MLP's $2d\\cdot4d = 8d^2$." },
      { t: "h", text: "Training compute: C ≈ 6ND" },
      { t: "p", text: String.raw`A forward pass costs about 2 FLOPs per parameter per token (one multiply and one add in each matrix product); the backward pass about twice that. So training a model with $N$ parameters on $D$ tokens costs` },
      { t: "math", tex: String.raw`C \approx 6\,N\,D \ \text{FLOPs}`, caption: "Ignores the attention term $\\propto n_{\\text{ctx}}$, which is small unless the context is long relative to $d$. Scaling-law details: [Scaling Laws](/learn/m19/scaling-efficiency)." },
      { t: "steps", items: [
        { title: "Worked example: 7B parameters, 140B tokens", tex: String.raw`C = 6 \times 7\times10^9 \times 1.4\times10^{11} = 5.88\times10^{21}\ \text{FLOPs}` },
        { title: "Wall-clock at 312 TFLOP/s peak (bf16) and 40% utilisation", tex: String.raw`\frac{5.88\times10^{21}}{0.4\times3.12\times10^{14}} = 4.71\times10^{7}\ \text{GPU-seconds} \approx 545\ \text{GPU-days}` },
        { title: "Compute-optimal split", text: String.raw`Hoffmann et al. (2022, *Training Compute-Optimal Large Language Models*, "Chinchilla") found that $N$ and $D$ should grow in proportion, at roughly 20 tokens per parameter. Then $C = 6N(20N) = 120N^2$: for $C = 10^{21}$ FLOPs, $N = \sqrt{10^{21}/120} \approx 2.9\times10^9$ and $D \approx 5.8\times10^{10}$ tokens.` },
      ] },
      { t: "p", text: "Compute-optimal is optimal for the *training* budget only. Models that will serve many requests are often trained on far more tokens than 20 per parameter, because a smaller model trained longer is cheaper to run (see [Efficient Inference](/learn/m20/llm-inference)). Our tiny run below uses $D/N \\approx 10.7$ tokens per parameter, but its 3,761 training characters are revisited about 163 times — far from the regime where scaling laws (which assume mostly unique data) apply." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "sim", id: "transformer", caption: "A small pretrained decoder-only Transformer (61,488 parameters, Alice in Wonderland Chapter I) running in your browser." },
      { t: "list", items: [
        "Open the **Logits** stage and step through positions of the prompt: every position predicts the next character simultaneously — one forward pass gives a training signal at all 64 positions.",
        "Note the aside: train loss 0.53 but held-out loss 2.51 nats/char. On 6,759 characters the model memorised the chapter. Real pretraining sees each token roughly once, so train and held-out loss stay close.",
        "Generate with greedy decoding and observe the recognisable phrases from the book: memorisation of repeated training data is what deduplication tries to prevent.",
      ] },
      { t: "sim", id: "lr-schedule", caption: "Warm-up, cosine and other schedules driving real training runs." },
      { t: "p", text: "Select the warm-up + cosine schedule and compare it with a constant learning rate: the warm-up avoids early instability, and the decay lets the loss settle at the end — the reason the tiny GPT below improves fastest in its first 150 steps and then flattens as the learning rate decays." },
      { t: "sim", id: "decoding", caption: "Text from a base model depends on how you decode it." },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "The pieces of the recipe that are pure arithmetic, in NumPy: a miniature data pipeline (normalisation, exact deduplication by SHA-256, heuristic quality filtering, near-duplicate detection with MinHash), the effect of mixture weights, RMSNorm, RoPE's relative-position property, SwiGLU's parameter matching, the warm-up + cosine schedule and the 6ND compute budget." },
      { t: "code", lang: "python", title: "pretraining_pieces_numpy.py", code: SCRATCH, output: SCRATCH_OUT },
      { t: "p", text: "Highlights: MinHash estimated the Jaccard similarity of the 'river bank' near-duplicate as 0.80 (true value 0.85) from 64 hashes and removed it; the mixture weights mean the small high-quality sources are repeated 1.7–2.5 times while the web source is only one-third used; the RoPE dot product is identical for positions (5, 3), (12, 10) and (102, 100); SwiGLU with hidden width $\\lfloor 8d/3 \\rfloor$ matches the GELU MLP's parameter count to 0.006%." },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Pretraining a tiny modern GPT for real: character tokens, RMSNorm pre-norm blocks, RoPE, SwiGLU, no biases, tied embeddings, AdamW ($\\beta_2 = 0.95$, decoupled weight decay on matrices only), linear warm-up + cosine decay, global-norm clipping — on the first 4,179 characters of *Alice's Adventures in Wonderland* (public domain), embedded in the program. Runs in about 20 seconds on one CPU thread." },
          { t: "code", lang: "python", title: "pretrain_tiny_gpt.py", code: TORCH, output: TORCH_OUT },
          { t: "p", text: "Read the curve like a pretraining engineer. At step 0 the loss is 3.654 — close to $\\ln 37 = 3.611$, as it should be with a small embedding initialisation (the first version of this program used PyTorch's default $\\mathcal N(0,1)$ embeddings with the tied head and started at a loss of 61, a classic initialisation bug). Held-out loss falls to 2.441 (perplexity 11.5) at step 150 and then **rises** while training loss keeps falling: with 3,761 training characters each seen about 163 times, the model starts memorising. Real pretraining avoids this regime by having more unique data than it can consume. The greedy sample — 'the was the the the' — shows a small model collapsing to its most frequent tokens; better decoding helps ([Decoding](/learn/m20/decoding)), but mostly more data and parameters do." },
          { t: "callout", kind: "tip", text: "On a GPU you would add `torch.autocast(\"cuda\", dtype=torch.bfloat16)` around the forward pass and `torch.compile(model)`; on CPU we keep float32 so that the output is bit-reproducible." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "modern_block_parts_keras.py (not executed in CI)", code: KERAS, output: "Prints (2, 8, 48): the shape after RMSNorm and a SwiGLU feed-forward block; the schedule and AdamW optimiser are configured as in the PyTorch run." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "table", head: ["Choice", "Options", "Advantages", "Disadvantages", "Typical modern default"], rows: [
        ["Objective", "Causal LM / masked LM / span corruption / prefix LM", "Causal LM directly enables generation and in-context learning", "Masked LM is better for some encoders, cannot generate", "Causal LM (decoder-only)"],
        ["Normalisation", "Post-LN / pre-LN LayerNorm / pre-RMSNorm", "Pre-RMSNorm is stable and cheap", "Pre-norm can let the residual stream grow", "Pre-RMSNorm"],
        ["Positions", "Learned absolute / sinusoidal / RoPE / ALiBi", "RoPE: relative, no table, extendable with scaling", "Extrapolation beyond training length still needs care", "RoPE"],
        ["FFN", "ReLU / GELU MLP / SwiGLU", "SwiGLU improves quality at equal parameters", "Three matrices instead of two", "SwiGLU"],
        ["Attention", "MHA / MQA / GQA", "GQA shrinks the KV cache", "Slight quality loss vs MHA", "GQA"],
        ["Dense vs MoE", "Dense / mixture of experts", "MoE: more parameters at same FLOPs", "Memory, routing, communication", "Both in use"],
        ["Tokenizer", "Byte-level BPE / SentencePiece unigram / bytes", "BPE: no unknown tokens, efficient", "Vocabulary affects multilingual fairness and arithmetic", "Byte-level BPE, 32k–256k tokens"],
        ["Training length", "Compute-optimal (~20 tokens/param) / over-trained", "Over-training gives cheaper inference", "More training compute", "Over-trained for deployed models"],
      ] },
      { t: "p", text: "Pretraining is often followed by **continued pretraining** (more tokens on a domain such as code or a new language), **long-context extension** (a short phase at longer sequence length with adjusted RoPE frequencies) and an **annealing** phase with the highest-quality data at the end of the learning-rate decay." },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Foundation models**: GPT-3 (Brown et al., 2020), LLaMA (Touvron et al., 2023) and their successors are base models produced exactly by this recipe at scale, then post-trained for chat.",
        "**Domain models**: continued pretraining on code, biomedical or legal text gives specialised base models.",
        "**Scientific sequence models**: the same next-token objective is applied to protein sequences, DNA and source code.",
        "**Evaluation proxies**: small models trained with the same pipeline are used to choose data mixtures and hyperparameters before the expensive run.",
        "**Embeddings and retrieval**: representations from pretrained decoders are fine-tuned into text-embedding models (see [Multimodal Models, CLIP and RAG Foundations](/learn/m15/multimodal)).",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "list", items: [
        "**Strength**: one simple, unsupervised objective scales with data and compute and yields broadly useful representations and in-context learning.",
        "**Cost**: training compute grows as $6ND$; frontier runs need thousands of accelerators for months, and mistakes are expensive — hence the reliance on small proxy experiments.",
        "**Data limits**: high-quality text is finite; repetition beyond a few epochs gives diminishing returns, as the overfitting in our tiny run shows in extreme form.",
        "**What a base model is not**: it does not follow instructions reliably, can reproduce biases, toxic content and memorised personal data from the corpus, and has a knowledge cut-off.",
        "**Tokenizer lock-in**: the tokenizer is fixed before pretraining; changing it later requires substantial retraining.",
        "**Evaluation is hard**: loss is smooth and predictable, but downstream abilities can appear abruptly and benchmarks can be contaminated by training data.",
      ] },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Comparing perplexities across tokenizers.** A 50k-token vocabulary and a character vocabulary give incomparable per-token losses; convert to bits per byte/character.",
        "**Evaluating on data that leaked into training** (duplicates across the split, benchmark contamination). Deduplicate across splits and decontaminate against test sets.",
        "**Bad initialisation with tied embeddings**: unit-variance embeddings make initial logits huge (our first attempt started at loss 61 instead of ≈ 3.6). Check that the step-0 loss is close to $\\ln|\\mathcal V|$.",
        "**Applying weight decay to norm gains and biases**, which pulls the gains toward 0. Decay matrices only.",
        "**No warm-up with Adam at a high learning rate**: early updates are huge because the second-moment estimate is poor, causing divergence.",
        "**Forgetting the end-of-document separator when packing**: the model learns spurious dependencies between unrelated documents. Some setups also mask attention across document boundaries.",
        "**Reading the training loss only.** With repeated data, training loss keeps falling while held-out loss rises (see the run above). Always track held-out loss.",
        "**Using C ≈ 6ND with total instead of non-embedding parameters for small models**, or ignoring the attention term for very long contexts — both distort the budget.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the pretraining objective of a GPT-style model?", a: "Next-token prediction: minimise the average negative log-likelihood −log p(x_t | x_<t) over all positions of the training text, computed in one causally masked forward pass per sequence. No labels are needed; the text supplies its own targets." },
      { t: "qa", level: "beginner", q: "Why deduplicate pretraining data?", a: "Duplicates waste compute on content the model has already seen, skew the effective data mixture toward whatever is copied most, increase verbatim memorisation (privacy and copyright risk), and can leak evaluation data. Exact dedup uses hashes of normalised text; near-dup uses MinHash/LSH over n-gram shingles." },
      { t: "qa", level: "beginner", q: "How do you convert a validation loss into perplexity, and what does perplexity 11.5 mean?", a: "PPL = exp(loss in nats per token). A perplexity of 11.5 means the model is, on average, as uncertain as if it were choosing uniformly among about 11.5 tokens at each step." },
      { t: "qa", level: "intermediate", q: "Estimate the training compute of a 13B-parameter model trained on 1T tokens.", a: "C ≈ 6ND = 6 × 1.3·10^10 × 10^12 = 7.8·10^22 FLOPs. At 312 TFLOP/s and 40% utilisation that is 7.8e22 / 1.248e14 ≈ 6.25·10^8 GPU-seconds ≈ 7,200 GPU-days." },
      { t: "qa", level: "intermediate", q: "What do RMSNorm, RoPE and SwiGLU change relative to the original Transformer, and why?", a: "RMSNorm drops LayerNorm's mean subtraction and bias, saving computation with no quality loss. RoPE replaces added position vectors with rotations of queries and keys so that attention scores depend on relative offsets and no position table is learned. SwiGLU replaces the ReLU/GELU MLP with a gated unit (SiLU(xW_g) ⊙ xW_u)W_d, which improved quality at equal parameter count in Shazeer (2020); the hidden width is set to ~8d/3 to match parameters." },
      { t: "qa", level: "intermediate", q: "What is a data mixture and how is it chosen?", a: "The sampling proportions of each source (web, code, books, encyclopedias, papers…) in the training stream, independent of their raw sizes. Small, high-quality sources are up-sampled (several epochs), huge noisy ones down-sampled. Weights are chosen by training small proxy models on candidate mixtures and comparing held-out losses and downstream evaluations." },
      { t: "qa", level: "advanced", q: "Why is 'compute-optimal' (≈20 tokens per parameter) not what most deployed models use?", a: "Compute-optimal minimises training loss for a fixed training budget. Deployment cost depends on model size (every generated token costs ≈2N FLOPs and memory for N parameters). Training a smaller model on many more tokens costs more training compute for the same loss, but much less inference compute over its lifetime, so total cost is lower. The loss keeps improving with more tokens, just less efficiently." },
      { t: "qa", level: "advanced", q: "A run shows a sudden loss spike at step 80k. What could cause it and what would you do?", a: "Possible causes: a bad batch (corrupted or extremely repetitive data), an optimizer instability (attention logits or the residual stream growing, Adam's second moment underestimating a sudden large gradient), or a learning rate that is too high for the current state. Remedies: roll back to a checkpoint before the spike and skip the offending batches, lower the learning rate, tighten gradient clipping, inspect the data, and consider stabilisers such as normalising queries/keys or lowering β2." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Training compute in FLOPs for N = 1e9 parameters and D = 2e10 tokens, using C ≈ 6ND? (enter in units of 1e20)", answer: 1.2, tolerance: 0.001, explain: "6 × 10^9 × 2·10^10 = 1.2·10^20 FLOPs." },
        { kind: "numeric", q: "Under the ≈20 tokens-per-parameter rule, how many parameters (in billions) are compute-optimal for C = 1.2e22 FLOPs?", answer: 10, tolerance: 0.05, explain: "C = 120 N² ⇒ N = √(1.2·10^22 / 120) = √(10^20) = 10^10 = 10 B parameters, trained on 200 B tokens." },
        { kind: "numeric", q: "A held-out loss of 2.0 nats/token corresponds to what perplexity? (2 decimals)", answer: 7.39, tolerance: 0.01, explain: "e² = 7.389." },
        { q: "Which property makes RoPE attention scores depend only on relative position?", options: ["The rotation angles are learned", "Rotations are orthogonal, so ⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩", "It adds a bias proportional to distance", "It normalises queries and keys"], answer: 1, explain: "R_mᵀR_n = R_{n−m}: only the offset survives." },
        { kind: "numeric", q: "RMSNorm of x = (3, 4) with unit gain (ignore ε): what is the first output component? (4 decimals)", answer: 0.8485, tolerance: 0.0002, explain: "rms = √((9+16)/2) = √12.5 = 3.5355; 3 / 3.5355 = 0.8485." },
        { q: "Why is weight decay usually not applied to RMSNorm gains?", options: ["They have no gradient", "Decay would pull them toward zero and shrink every normalised activation, which is not a meaningful regularisation", "PyTorch forbids it", "They are integers"], answer: 1, explain: "Gains set the scale of normalised activations; decaying them fights the normalisation." },
        { q: "In the tiny GPT run, held-out loss rises after step 150 while training loss falls. The best explanation is…", options: ["The learning rate is too low", "The model memorises a tiny training set seen ~163 times", "RoPE fails at long contexts", "Gradient clipping is too aggressive"], answer: 1, explain: "Severe data repetition leads to overfitting; real pretraining sees most tokens about once." },
      ] },
      { t: "exercise", title: "Plan a pretraining run", prompt: "You have 512 GPUs at 312 TFLOP/s (bf16 peak) for 30 days at 40% utilisation. (a) What compute budget C is that? (b) Using C = 6ND and 20 tokens per parameter, what N and D are compute-optimal? (c) If you instead train a 7B model, how many tokens can you afford?", hint: "C = GPUs × peak × utilisation × seconds.", solution: [
        { t: "steps", items: [
          { title: "Budget", tex: String.raw`C = 512 \times 3.12\times10^{14} \times 0.4 \times 30\times86400 = 1.656\times10^{23}\ \text{FLOPs}` },
          { title: "Compute-optimal", tex: String.raw`N = \sqrt{C/120} = \sqrt{1.380\times10^{21}} = 3.71\times10^{10},\quad D = 20N = 7.43\times10^{11}` },
          { title: "Fixed 7B model", tex: String.raw`D = C/(6N) = 1.656\times10^{23}/(4.2\times10^{10}) = 3.94\times10^{12}\ \text{tokens} \;(\approx 563 \text{ tokens/parameter})` },
        ] },
        { t: "p", text: "About a 37B model on 0.74T tokens minimises training loss; a 7B model on 3.9T tokens reaches a somewhat higher loss but is about 5× cheaper per generated token at inference." },
      ] },
      { t: "exercise", title: "Ablate the modern recipe", prompt: "Modify the PyTorch program: (1) replace RMSNorm with `nn.LayerNorm`, (2) replace RoPE by a learned position embedding, (3) replace SwiGLU by a GELU MLP with hidden width 4d. Run each change separately with the same seed and compare the best held-out loss. Why should you not over-interpret differences at this scale?", hint: "Keep the parameter count similar; record min over evaluation steps of the validation loss.", solution: [
        { t: "p", text: "Each change is a few lines: `self.n1 = nn.LayerNorm(d)`; add `self.pos = nn.Embedding(CTX, d)` and `x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))` while calling attention without `rope`; and `self.mlp = nn.Sequential(nn.Linear(d, 4*d, bias=False), nn.GELU(), nn.Linear(4*d, d, bias=False))`." },
        { t: "p", text: "Differences between variants on a 4k-character corpus are typically comparable to the run-to-run variation from changing the seed, because the run is dominated by overfitting and noise. The published advantages of RMSNorm, RoPE and SwiGLU were measured at much larger scale and averaged over many evaluations. A sound ablation reports several seeds and the spread, and uses enough unique data that the model is not memorising." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        String.raw`Objective: minimise $-\frac1{|\mathcal D|}\sum\log p_\theta(x_t\mid x_{<t})$; perplexity $= e^{\mathcal L}$ (compare only with the same tokenizer; otherwise bits/byte).`,
        "Data pipeline: extract → language ID → quality filter → exact + MinHash near-dedup → decontaminate → tokenize and pack → mix sources by weights.",
        "Architecture: pre-norm RMSNorm, RoPE, SwiGLU (hidden ≈ 8d/3), no biases, GQA, tied embeddings for small models.",
        "Optimisation: AdamW (β₂ ≈ 0.95, decay on matrices), warm-up + cosine to ~10%, clip at 1.0, bf16 mixed precision, million-token batches.",
        String.raw`Compute: $C \approx 6ND$; compute-optimal ≈ 20 tokens per parameter ($C = 120N^2$); deployed models are often trained longer.`,
        "Check step-0 loss ≈ ln|V|; track held-out loss; repeated data causes memorisation.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Decoding: Greedy, Beam, Temperature, Top-k, Top-p](/learn/m20/decoding) — turning the base model's distributions into text.",
        "[Instruction Tuning, RLHF and DPO](/learn/m20/alignment) — turning a base model into an assistant.",
        "[Efficient Inference: KV Cache, FlashAttention, MoE](/learn/m20/llm-inference) — serving what you trained.",
        "[Scaling Laws, LoRA, QLoRA and PEFT](/learn/m19/scaling-efficiency) — the power laws behind compute-optimal sizing, and cheap fine-tuning afterwards.",
      ] },
    ],
  },
};

export default lesson;
