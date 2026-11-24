import type { Lesson } from "@/lib/types";

const ATT_NUMPY = String.raw`import numpy as np

# Sizes (kept tiny so every number can be printed)
T, ENC, DEC, ATT, EMB = 5, 6, 4, 3, 2      # source length, annotation size, decoder size, attention size, embedding size
rng = np.random.default_rng(0)

# ---------------- parameters of one Bahdanau decoder step (random stand-ins for trained weights) ----------------
W_s = rng.normal(0, 0.5, (ATT, DEC))      # projects the previous decoder state s_{i-1}
W_h = rng.normal(0, 0.5, (ATT, ENC))      # projects each encoder annotation h_j
v = rng.normal(0, 0.5, ATT)               # turns the tanh features into one score
W_ih = rng.normal(0, 0.5, (3 * DEC, EMB + ENC))   # decoder GRU, input [emb(y_{i-1}); c_i]
W_hh = rng.normal(0, 0.5, (3 * DEC, DEC))
b_ih, b_hh = np.zeros(3 * DEC), np.zeros(3 * DEC)

# ---------------- inputs to the step ----------------
H = np.tanh(rng.normal(0, 1, (T, ENC)))   # encoder annotations h_1..h_T (rows), in (-1, 1) like GRU states
s_prev = np.tanh(rng.normal(0, 1, DEC))   # previous decoder state s_{i-1}
y_emb = rng.normal(0, 1, EMB)             # embedding of the previously emitted token y_{i-1}


def softmax(z):
    z = z - z.max()                       # subtract the max for numerical stability (result unchanged)
    e = np.exp(z)
    return e / e.sum()


def additive_attention(s_prev, H, W_s, W_h, v, mask=None):
    """e_j = v^T tanh(W_s s_prev + W_h h_j);  alpha = softmax(e);  c = sum_j alpha_j h_j."""
    keys = H @ W_h.T                      # (T, ATT): W_h h_j for every j -- depends only on the source
    query = W_s @ s_prev                  # (ATT,):   W_s s_{i-1} -- one vector per decoder step
    feats = np.tanh(query[None, :] + keys)        # (T, ATT): broadcasting adds the query to every key row
    e = feats @ v                         # (T,): one scalar score per source position
    if mask is not None:
        e = np.where(mask, e, -np.inf)    # padded positions get probability exactly 0
    alpha = softmax(e)
    c = alpha @ H                         # (ENC,): convex combination of the annotations
    return e, alpha, c


def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """PyTorch GRUCell convention: gates ordered (r, z, n); h' = (1 - z) * n + z * h."""
    n_h = h.shape[0]
    gi, gh = W_ih @ x + b_ih, W_hh @ h + b_hh
    r = 1 / (1 + np.exp(-(gi[:n_h] + gh[:n_h])))
    z = 1 / (1 + np.exp(-(gi[n_h:2 * n_h] + gh[n_h:2 * n_h])))
    n = np.tanh(gi[2 * n_h:] + r * gh[2 * n_h:])
    return (1 - z) * n + z * h


np.set_printoptions(precision=4, suppress=True)
e, alpha, c = additive_attention(s_prev, H, W_s, W_h, v)
print("scores e      :", e)
print("weights alpha :", alpha, " sum =", round(alpha.sum(), 6))
print("context c     :", c)

# the same scores with an explicit loop over source positions (the formula, term by term)
e_loop = np.array([v @ np.tanh(W_s @ s_prev + W_h @ H[j]) for j in range(T)])
print("loop == vectorized:", np.allclose(e_loop, e))

# one full decoder step: s_i = GRU([emb(y_{i-1}); c_i], s_{i-1})
s_new = gru_cell(np.concatenate([y_emb, c]), s_prev, W_ih, W_hh, b_ih, b_hh)
print("new decoder state s_i:", s_new)

# padding: pretend the source has only 3 real tokens; positions 4 and 5 are <pad>
mask = np.arange(T) < 3
e_m, alpha_m, c_m = additive_attention(s_prev, H, W_s, W_h, v, mask)
print("masked alpha  :", alpha_m)
print("renormalised  :", np.allclose(alpha_m[:3], alpha[:3] / alpha[:3].sum()))

# the fixed-context baseline ignores s_prev entirely: every step gets the same vector
print("context changes with s_prev:", not np.allclose(c, additive_attention(-s_prev, H, W_s, W_h, v)[2]))`;

const ATT_NUMPY_OUT = String.raw`scores e      : [-0.0317 -0.0567 -0.4861  0.0834 -0.2568]
weights alpha : [0.2207 0.2153 0.1401 0.2476 0.1762]  sum = 1.0
context c     : [ 0.1997 -0.8095 -0.0909 -0.0199  0.0635  0.0394]
loop == vectorized: True
new decoder state s_i: [-0.5315 -0.5559 -0.2142  0.1879]
masked alpha  : [0.3831 0.3737 0.2432 0.     0.    ]
renormalised  : True
context changes with s_prev: True`;

const ATT_TORCH_CHECK = String.raw`import numpy as np
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Additive attention: e_ij = v^T tanh(W_s s_{i-1} + W_h h_j), batched and masked."""

    def __init__(self, dec_size, enc_size, att_size):
        super().__init__()
        self.W_s = nn.Linear(dec_size, att_size, bias=False)
        self.W_h = nn.Linear(enc_size, att_size, bias=False)
        self.v = nn.Linear(att_size, 1, bias=False)

    def forward(self, s_prev, H, mask=None):
        # s_prev: (B, dec)   H: (B, T, enc)   mask: (B, T) bool, True = real token
        keys = self.W_h(H)                                       # (B, T, att); cache this across decoder steps
        e = self.v(torch.tanh(self.W_s(s_prev).unsqueeze(1) + keys)).squeeze(-1)   # (B, T)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)                         # (B, T)
        c = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)          # (B, enc)
        return c, alpha, e


# exactly the same random numbers, in the same order, as the NumPy from-scratch program
T, ENC, DEC, ATT, EMB = 5, 6, 4, 3, 2
rng = np.random.default_rng(0)
W_s = rng.normal(0, 0.5, (ATT, DEC))
W_h = rng.normal(0, 0.5, (ATT, ENC))
v = rng.normal(0, 0.5, ATT)
W_ih = rng.normal(0, 0.5, (3 * DEC, EMB + ENC))
W_hh = rng.normal(0, 0.5, (3 * DEC, DEC))
H = np.tanh(rng.normal(0, 1, (T, ENC)))
s_prev = np.tanh(rng.normal(0, 1, DEC))
y_emb = rng.normal(0, 1, EMB)

att = BahdanauAttention(DEC, ENC, ATT).double()
cell = nn.GRUCell(EMB + ENC, DEC).double()
with torch.no_grad():                                            # copy the weights into the modules
    att.W_s.weight.copy_(torch.from_numpy(W_s))
    att.W_h.weight.copy_(torch.from_numpy(W_h))
    att.v.weight.copy_(torch.from_numpy(v)[None, :])
    cell.weight_ih.copy_(torch.from_numpy(W_ih))
    cell.weight_hh.copy_(torch.from_numpy(W_hh))
    cell.bias_ih.zero_()
    cell.bias_hh.zero_()

Ht = torch.from_numpy(H)[None]                                   # add a batch dimension of 1
st = torch.from_numpy(s_prev)[None]
with torch.no_grad():
    c, alpha, e = att(st, Ht)
    s_new = cell(torch.cat([torch.from_numpy(y_emb)[None], c], dim=-1), st)
    mask = torch.arange(T)[None] < 3
    _, alpha_m, _ = att(st, Ht, mask)
e, alpha, c, s_new, alpha_m = (t[0].numpy() for t in (e, alpha, c, s_new, alpha_m))

np.set_printoptions(precision=4, suppress=True)
print("e     :", e)
print("alpha :", alpha)
print("c     :", c)
print("s_i   :", s_new)
print("masked alpha:", alpha_m)

# the same step written directly in NumPy (as in the from-scratch section)
keys = H @ W_h.T
e_np = np.tanh(W_s @ s_prev + keys) @ v
a_np = np.exp(e_np - e_np.max()); a_np /= a_np.sum()
c_np = a_np @ H
diff = max(abs(e - e_np).max(), abs(alpha - a_np).max(), abs(c - c_np).max())
print("PyTorch module matches NumPy (max |diff| < 1e-12):", diff < 1e-12)
print("attention parameters:", sum(p.numel() for p in att.parameters()))`;

const ATT_TORCH_CHECK_OUT = String.raw`e     : [-0.0317 -0.0567 -0.4861  0.0834 -0.2568]
alpha : [0.2207 0.2153 0.1401 0.2476 0.1762]
c     : [ 0.1997 -0.8095 -0.0909 -0.0199  0.0635  0.0394]
s_i   : [-0.5315 -0.5559 -0.2142  0.1879]
masked alpha: [0.3831 0.3737 0.2432 0.     0.    ]
PyTorch module matches NumPy (max |diff| < 1e-12): True
attention parameters: 33`;

const ATT_TORCH_TRAIN = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)                         # tiny model: one thread is fastest and fully reproducible
rng = np.random.default_rng(0)

PAD, SOS, EOS = 0, 1, 2                          # special symbols; letters a..h are ids 3..10
LETTERS = "abcdefgh"
V, EMB, ENC_H, DEC_H, ATT = 3 + len(LETTERS), 16, 24, 48, 24
MAX_LEN = 10


def batch(n, lo=3, hi=MAX_LEN):
    """Random strings, padded; the target is the reversed string followed by <eos>."""
    lens = rng.integers(lo, hi + 1, size=n)
    src = np.full((n, hi), PAD); tgt = np.full((n, hi + 1), PAD)
    for b, L in enumerate(lens):
        ids = rng.integers(3, V, size=L)
        src[b, :L], tgt[b, :L], tgt[b, L] = ids, ids[::-1], EOS
    return torch.tensor(src), torch.tensor(lens), torch.tensor(tgt)


class Seq2SeqAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(V, EMB, padding_idx=PAD)
        self.encoder = nn.GRU(EMB, ENC_H, batch_first=True, bidirectional=True)
        self.init = nn.Linear(ENC_H, DEC_H)                      # s_0 from the backward state at position 1
        self.W_s = nn.Linear(DEC_H, ATT, bias=False)
        self.W_h = nn.Linear(2 * ENC_H, ATT, bias=False)
        self.v = nn.Linear(ATT, 1, bias=False)
        self.tgt_emb = nn.Embedding(V, EMB)
        self.cell = nn.GRUCell(EMB + 2 * ENC_H, DEC_H)
        self.out = nn.Linear(DEC_H + 2 * ENC_H, V)

    def encode(self, src, lens):
        packed = nn.utils.rnn.pack_padded_sequence(self.src_emb(src), lens, batch_first=True, enforce_sorted=False)
        H, h_n = self.encoder(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=src.shape[1])
        mask = torch.arange(src.shape[1])[None] < lens[:, None]
        return H, self.W_h(H), mask, torch.tanh(self.init(h_n[1]))

    def step(self, y_prev, s, H, keys, mask):
        e = self.v(torch.tanh(self.W_s(s)[:, None] + keys)).squeeze(-1)    # additive scores (B, T)
        alpha = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=-1)
        c = torch.bmm(alpha[:, None], H).squeeze(1)                         # context (B, 2*ENC_H)
        s = self.cell(torch.cat([self.tgt_emb(y_prev), c], -1), s)
        return self.out(torch.cat([s, c], -1)), s, alpha

    def forward(self, src, lens, tgt):                                     # teacher forcing
        H, keys, mask, s = self.encode(src, lens)
        y, logits = torch.full((src.shape[0],), SOS), []
        for i in range(tgt.shape[1]):
            lg, s, _ = self.step(y, s, H, keys, mask)
            logits.append(lg)
            y = tgt[:, i]
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src, lens, steps=MAX_LEN + 1):
        H, keys, mask, s = self.encode(src, lens)
        y, out, alphas = torch.full((src.shape[0],), SOS), [], []
        for _ in range(steps):
            lg, s, a = self.step(y, s, H, keys, mask)
            y = lg.argmax(-1)
            out.append(y); alphas.append(a)
        return torch.stack(out, 1), torch.stack(alphas, 1)


model = Seq2SeqAttention()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
print("parameters:", sum(p.numel() for p in model.parameters()))
for step in range(1, 501):
    src, lens, tgt = batch(64)
    loss = F.cross_entropy(model(src, lens, tgt).flatten(0, 1), tgt.flatten(), ignore_index=PAD)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# exact-match accuracy on fresh strings, by length
src, lens, tgt = batch(800)
pred, _ = model.greedy(src, lens)
ok = [(pred[b, :L + 1] == tgt[b, :L + 1]).all().item() for b, L in enumerate(lens.tolist())]
for L in (3, 6, 10):
    sel = [o for o, l in zip(ok, lens.tolist()) if l == L]
    print(f"length {L:2d}: exact match {np.mean(sel):.3f}  ({len(sel)} strings)")

# the alignment for one input: for each output step, which source position got the most weight?
word = "abcdefgh"
ids = torch.tensor([[3 + LETTERS.index(ch) for ch in word] + [PAD] * (MAX_LEN - len(word))])
pred, alphas = model.greedy(ids, torch.tensor([len(word)]))
out = "".join(LETTERS[t - 3] if t >= 3 else "|" for t in pred[0].tolist()[:len(word)])
print("input :", word, " output:", out)
print("argmax_j alpha_ij per output step:", alphas[0, :len(word), :len(word)].argmax(-1).add(1).tolist())
print("max alpha per step:", [round(a, 2) for a in alphas[0, :len(word), :len(word)].max(-1).values.tolist()])`;

const ATT_TORCH_TRAIN_OUT = String.raw`parameters: 27411
step  100  loss 0.2598
step  200  loss 0.0366
step  300  loss 0.0131
step  400  loss 0.0037
step  500  loss 0.0106
length  3: exact match 1.000  (93 strings)
length  6: exact match 1.000  (113 strings)
length 10: exact match 0.939  (98 strings)
input : abcdefgh  output: hgfedcba
argmax_j alpha_ij per output step: [8, 7, 6, 5, 4, 3, 2, 1]
max alpha per step: [0.84, 0.66, 0.71, 0.68, 0.73, 0.79, 0.73, 0.79]`;

const ATT_KERAS = String.raw`import numpy as np
import tensorflow as tf
from tensorflow import keras

T, ENC, DEC, ATT = 5, 6, 4, 3
rng = np.random.default_rng(0)
H = np.tanh(rng.normal(0, 1, (1, T, ENC))).astype("float32")      # encoder annotations (batch, T, enc)
s_prev = np.tanh(rng.normal(0, 1, (1, 1, DEC))).astype("float32")  # one decoder query (batch, 1, dec)

# Keras' AdditiveAttention computes  score_j = sum(scale * tanh(query + key_j)).
# It does NOT project its inputs, so the Bahdanau projections W_s and W_h are explicit Dense layers;
# its learned scale vector then plays the role of v.
W_s = keras.layers.Dense(ATT, use_bias=False)
W_h = keras.layers.Dense(ATT, use_bias=False)
attend = keras.layers.AdditiveAttention(use_scale=True)

query = W_s(s_prev)                        # (1, 1, ATT)
keys = W_h(H)                              # (1, T, ATT)
mask = np.array([[True, True, True, False, False]])   # last two source positions are padding
context, weights = attend([query, H, keys], mask=[None, mask], return_attention_scores=True)

print("context shape:", context.shape)     # (1, 1, ENC): one context vector for the one query
print("weights:", np.round(weights.numpy()[0, 0], 4), "sum:", float(weights.numpy().sum()))

# --- inside a model: a decoder step that attends over the encoder outputs ---------------------
src = keras.Input(shape=(None,), dtype="int32")
x = keras.layers.Embedding(29, 16, mask_zero=True)(src)
enc = keras.layers.Bidirectional(keras.layers.GRU(24, return_sequences=True))(x)   # (B, T, 48)
dec_state = keras.Input(shape=(1, 48))                     # s_{i-1} for one decoding step
ctx = keras.layers.AdditiveAttention()([keras.layers.Dense(32, use_bias=False)(dec_state), enc,
                                        keras.layers.Dense(32, use_bias=False)(enc)])
step = keras.Model([src, dec_state], ctx)
step.summary()`;

const lesson: Lesson = {
  id: "m10/attention-intro",
  title: "From Seq2Seq Bottlenecks to Attention",
  summary:
    "Why encoder–decoder RNNs that squeeze a whole input into one fixed-size context vector break down on long inputs — measured on a real trained model — and how additive (Bahdanau) attention fixes it: full equations, a hand-computed step, alignments, soft vs hard attention, multiplicative variants, attention as a soft dictionary lookup, interpretability caveats and cost, implemented in NumPy and verified against PyTorch.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m09/rnn-basics", "m09/gru-variants", "m09/lstm", "m03/activation-functions"],
  next: ["m10/scaled-dot-product-attention", "m10/multi-head-attention", "m11/transformer-architecture"],
  terms: ["sequence-to-sequence", "context-vector", "additive-attention", "alignment", "attention-weights", "soft-attention", "hard-attention", "multiplicative-attention", "teacher-forcing", "bidirectional-rnn"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "Many problems map one sequence to another sequence of a different length: translating a sentence, transcribing speech, summarising a document, converting a date format. The **encoder–decoder** (or **sequence-to-sequence**, seq2seq) architecture handles them with two recurrent networks: an **encoder** reads the input and a **decoder** generates the output one token at a time. Two papers introduced it for machine translation in 2014: Sutskever, Vinyals and Le, *Sequence to Sequence Learning with Neural Networks* (deep LSTMs), and Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation* (the paper that introduced the GRU).",
      },
      {
        t: "p",
        text: "In both, everything the decoder knows about the input passes through a **single fixed-size vector** — the encoder's final hidden state, called the **context vector**. A 5-word sentence and a 50-word sentence must be compressed into the same number of floats. Cho, van Merriënboer, Bahdanau and Bengio (2014, *On the Properties of Neural Machine Translation: Encoder–Decoder Approaches*) showed that the translation quality of such models falls steeply as sentences get longer. This is the **fixed-length context-vector bottleneck**.",
      },
      {
        t: "p",
        text: "We measured the bottleneck directly on the task used throughout this lesson: **reversing a string of lowercase letters** (`attention` → `noitnetta`). A GRU encoder–decoder whose decoder sees one fixed context vector (27,405 parameters, trained on lengths 1–20) reverses 100.0% of 3-letter strings exactly, but only 63.5% at length 8, 30.5% at length 10, 3.5% at length 13 and 0% from length 16 on. The same model with attention (30,509 parameters, identical data, seed and 6000 training steps) reverses 100% of strings at every length from 1 to 20.",
      },
      {
        t: "p",
        text: "Bahdanau, Cho and Bengio (2014; ICLR 2015), *Neural Machine Translation by Jointly Learning to Align and Translate*, removed the bottleneck. The encoder keeps **one vector per input position** (an *annotation*), and at every output step the decoder computes a fresh context vector as a **weighted average of all annotations**. The weights — the **attention weights** — are produced by a small neural network that scores how relevant each input position is to the current decoder state, followed by a softmax. Everything is differentiable, so the model learns *where to look* from the translation loss alone, with no alignment labels. The scoring network is a one-hidden-layer MLP, which is why this is called **additive** attention.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The idea of this lesson",
        text: String.raw`Replace the one fixed context vector $c$ by a step-dependent $c_i = \sum_j \alpha_{ij} h_j$, with $\alpha_{ij} = \mathrm{softmax}_j\big(v^\top\tanh(W_s s_{i-1} + W_h h_j)\big)$. The decoder now *retrieves* what it needs at each step instead of *remembering* everything.`,
      },
      {
        t: "p",
        text: "Attention made neural machine translation competitive, and was quickly adopted for speech recognition, image captioning, summarisation and speech synthesis. Its generalisation — queries, keys and values compared by scaled dot products — became the only sequence-mixing operation in the Transformer, which is why this lesson is the bridge from recurrent networks to [Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) and [the Transformer](/learn/m11/transformer-architecture).",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Recurrent neural networks](/learn/m09/rnn-basics) — the hidden-state recurrence $h_t = f(h_{t-1}, x_t)$, unrolling, and the seq2seq/teacher-forcing ideas introduced there. We will not re-derive them.",
          "[GRUs and RNN variants](/learn/m09/gru-variants) — our encoder and decoder are GRUs (PyTorch gate convention), and the encoder is **bidirectional**: one GRU reads left→right, another right→left, and their states are concatenated.",
          "[LSTM](/learn/m09/lstm) — the Sutskever et al. seq2seq model used LSTMs; everything here works identically with LSTM cells.",
          "[Activation functions](/learn/m03/activation-functions) — **softmax** (turns scores into non-negative weights summing to 1) and **tanh**.",
          "[Vectors and matrices](/learn/m02/vectors-matrices) — matrix–vector products, the dot product as similarity, and broadcasting shapes.",
          "[Backpropagation through time](/learn/m09/bptt) — to understand why a short gradient path from each output to each input matters.",
          "[Classification losses](/learn/m05/classification-losses) — the decoder is trained with per-token cross-entropy.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Interpreting from memory vs. interpreting with the transcript",
        text: "An interpreter who must listen to a whole speech, then translate it **from memory**, does well on short speeches and badly on long ones — the memory is the bottleneck. Give the same interpreter the written transcript, and while producing each sentence they glance back at the part of the source that matters right now, keeping a finger on it. They no longer need to memorise; they need to know **where to look**. The fixed-context encoder–decoder is the first interpreter. Attention is the finger on the transcript.",
      },
      {
        t: "p",
        text: "Concretely: the encoder still reads the whole input, but instead of handing over only its final state, it leaves behind a vector $h_j$ for **every** input position $j$ — a summary of that word in its context. Before emitting each output token, the decoder asks: *given what I have produced so far (my state $s_{i-1}$), how useful is each $h_j$?* It scores every position, turns the scores into percentages with a softmax, and blends the $h_j$ with those percentages. The blend $c_i$ is a context vector made **for this step**.",
      },
      {
        t: "p",
        text: "Here is what the trained model in the simulator actually does when reversing `attention` (9 letters). For output 1 it must copy input 9 (`n`). Its scores over the nine positions are $e = (-1.47, -1.39, -1.24, -1.05, -0.45, 0.74, 2.82, 6.14, 8.35)$, which the softmax turns into $\\alpha_9 = 0.896$, $\\alpha_8 = 0.099$, $\\alpha_7 = 0.004$ and essentially 0 elsewhere. For output 2 the peak moves to position 8 ($\\alpha = 0.814$), for output 3 to position 7 ($0.864$), and so on down to output 9, which puts $0.976$ on position 1. Nobody told the model to do this — the anti-diagonal alignment emerged from training on input/output pairs alone. At the final ⟨eos⟩ step there is nothing left to copy and the weights spread out (the largest is $0.293$).",
      },
      {
        t: "p",
        text: "Two consequences follow. First, the amount of information available to the decoder grows with the input length (there are $T_x$ annotations, not one vector). Second, the gradient from output $i$ reaches input position $j$ through a **single** weighted-sum step with weight $\\alpha_{ij}$, instead of travelling backwards through every encoder step — so long-range credit assignment becomes easy.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "p", text: "One forward pass of an attention-based encoder–decoder (the model in the simulator), for an input of $T_x$ tokens:" },
      {
        t: "steps",
        items: [
          { title: "Embed the input", text: "Each input token id becomes a learned vector $x_j$ (16 numbers in our model)." },
          { title: "Encode in both directions", text: "A forward GRU produces $\\overrightarrow{h}_j$ from $x_1..x_j$; a backward GRU produces $\\overleftarrow{h}_j$ from $x_j..x_{T_x}$. Concatenate them into the annotation $h_j$ — it describes token $j$ **with context from both sides**.", tex: String.raw`h_j = \big[\overrightarrow{h}_j\,;\,\overleftarrow{h}_j\big] \in \mathbb{R}^{2n}` },
          { title: "Precompute the keys", text: "Project every annotation once: $k_j = W_h h_j$. These do not depend on the decoder, so they are reused at every output step." },
          { title: "Initialise the decoder", text: "As in Bahdanau et al., the first decoder state comes from the backward encoder state at position 1 (which has read the whole input).", tex: String.raw`s_0 = \tanh\!\big(W_{\text{init}}\overleftarrow{h}_1 + b_{\text{init}}\big)` },
          { title: "Score every input position", text: "Project the previous decoder state once, add it to every key, squash with tanh, and reduce to a scalar with $v$.", tex: String.raw`e_{ij} = v^\top \tanh\!\big(W_s s_{i-1} + W_h h_j\big),\quad j = 1..T_x` },
          { title: "Normalise into attention weights", text: "Softmax over input positions (padding positions are set to $-\\infty$ first so they get weight exactly 0).", tex: String.raw`\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}` },
          { title: "Build the context vector", text: "A convex combination of the annotations: it lies inside their convex hull, and is close to $h_j$ when $\\alpha_{ij}\\approx 1$.", tex: String.raw`c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j` },
          { title: "Update the decoder state", text: "The GRU cell reads the embedding of the previous output token together with the context.", tex: String.raw`s_i = \mathrm{GRU}\big([E\,y_{i-1}\,;\,c_i],\ s_{i-1}\big)` },
          { title: "Predict the next token", text: "A linear read-out of state and context, then softmax over the output vocabulary.", tex: String.raw`p(y_i \mid y_{<i}, x) = \mathrm{softmax}\big(W_o[s_i\,;\,c_i] + b_o\big)` },
          { title: "Choose and repeat", text: "Greedy decoding emits $\\arg\\max$; training instead feeds the ground-truth previous token (**teacher forcing**) and adds the cross-entropy of every step. Stop when the model emits the end-of-sequence token." },
        ],
      },
      {
        t: "p",
        text: "The **fixed-context baseline** is the same model with steps 3, 5, 6 and 7 deleted: $c = [\\overrightarrow{h}_{T_x};\\overleftarrow{h}_1]$ is computed once and fed unchanged at every step (as in Cho et al., 2014, where the context enters every decoder step). Our two trained models differ in exactly this.",
      },
      {
        t: "callout",
        kind: "note",
        title: "No alignment labels",
        text: "Nobody tells the model which input position corresponds to which output. The attention weights are an intermediate quantity; gradients of the output cross-entropy flow through $c_i$ into $\\alpha_{ij}$, into $e_{ij}$, into $W_s$, $W_h$ and $v$. The alignment is **learned as a by-product** of predicting the output well — hence the paper's title, *jointly learning to align and translate*.",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "The model as a probability distribution" },
      {
        t: "p",
        text: String.raw`An encoder–decoder defines a conditional distribution over output sequences by the chain rule of probability, one token at a time: $p(y_1..y_{T_y}\mid x) = \prod_{i=1}^{T_y} p(y_i\mid y_{<i}, x)$. The fixed-context model approximates each factor as $g(y_{i-1}, s_i, c)$ with **one** $c$ for all $i$; the attention model uses $g(y_{i-1}, s_i, c_i)$. Training maximises the log-likelihood $\sum_i \log p(y_i\mid y_{<i},x)$, i.e. minimises the summed per-token cross-entropy.`,
      },
      { t: "h", text: "Additive attention — every symbol defined" },
      {
        t: "math",
        tex: String.raw`e_{ij} = v^\top \tanh\!\big(W_s\, s_{i-1} + W_h\, h_j\big),\qquad \alpha_{ij} = \frac{\exp e_{ij}}{\sum_{k=1}^{T_x}\exp e_{ik}},\qquad c_i = \sum_{j=1}^{T_x}\alpha_{ij}\,h_j`,
      },
      {
        t: "list",
        items: [
          String.raw`$h_j\in\mathbb{R}^{d_h}$: encoder annotation of input position $j$ ($d_h = 2n$ for a bidirectional encoder with $n$ units per direction; $d_h = 48$ in our model).`,
          String.raw`$s_{i-1}\in\mathbb{R}^{d_s}$: decoder state *before* producing output $i$ ($d_s=48$).`,
          String.raw`$W_s\in\mathbb{R}^{d_a\times d_s}$, $W_h\in\mathbb{R}^{d_a\times d_h}$: projections into a common **attention space** of size $d_a$ ($d_a = 32$). Because each side has its own matrix, $d_s$ and $d_h$ may differ.`,
          String.raw`$v\in\mathbb{R}^{d_a}$: turns the $d_a$ tanh features into one scalar score. Together, $W_s, W_h, v$ form a one-hidden-layer MLP applied to the pair $(s_{i-1}, h_j)$ — the hidden pre-activation is a **sum** of the two projections, hence "additive". Its parameter count is $d_a(d_s + d_h + 1)$: $32\cdot(48+48+1) = 3104$ in our model.`,
          String.raw`$e_{ij}\in\mathbb{R}$: the **alignment score** (energy); $\alpha_{ij}\in(0,1)$ with $\sum_j\alpha_{ij}=1$: the **attention weight**; the $T_y\times T_x$ matrix $A=[\alpha_{ij}]$ is the **alignment matrix**.`,
          String.raw`$c_i\in\mathbb{R}^{d_h}$: the context vector for output step $i$ — the expectation of the annotation under the distribution $\alpha_{i\cdot}$: $c_i = \mathbb{E}_{j\sim\alpha_{i\cdot}}[h_j]$.`,
        ],
      },
      { t: "h", text: "Worked example (every intermediate number)" },
      {
        t: "p",
        text: String.raw`Take $T_x=3$ annotations of size 2, $h_1=(1,0)$, $h_2=(0,1)$, $h_3=(1,1)$, decoder state $s_{i-1}=(0.5,-0.5)$, attention size $d_a=2$ and`,
      },
      { t: "math", tex: String.raw`W_s=\begin{pmatrix}0.5&-1\\1&0.5\end{pmatrix},\quad W_h=\begin{pmatrix}1&0.5\\-0.5&1\end{pmatrix},\quad v=\begin{pmatrix}1\\-1\end{pmatrix}` },
      {
        t: "steps",
        items: [
          { title: "Query projection (once per step)", tex: String.raw`W_s s_{i-1} = (0.5\cdot0.5 + (-1)(-0.5),\ 1\cdot0.5+0.5\cdot(-0.5)) = (0.75,\ 0.25)` },
          { title: "Key projections (once per input)", tex: String.raw`W_h h_1 = (1,\,-0.5),\quad W_h h_2 = (0.5,\,1),\quad W_h h_3 = (1.5,\,0.5)` },
          { title: "Add and squash", tex: String.raw`\tanh(1.75,\,-0.25) = (0.941376,\,-0.244919),\ \tanh(1.25,\,1.25) = (0.848284,\,0.848284),\ \tanh(2.25,\,0.75) = (0.978026,\,0.635149)` },
          { title: "Scores e = vᵀ(·)", tex: String.raw`e_{i1} = 0.941376 + 0.244919 = 1.186294,\quad e_{i2} = 0,\quad e_{i3} = 0.978026 - 0.635149 = 0.342877` },
          { title: "Softmax", tex: String.raw`e^{e} = (3.274922,\ 1,\ 1.408996),\ \ \Sigma = 5.683918\ \Rightarrow\ \alpha = (0.576173,\ 0.175935,\ 0.247892)` },
          { title: "Context vector", tex: String.raw`c_i = 0.576173\,(1,0) + 0.175935\,(0,1) + 0.247892\,(1,1) = (0.824065,\ 0.423827)` },
        ],
      },
      {
        t: "p",
        text: String.raw`Note what a plain dot product would have said: $s_{i-1}\cdot h_j = (0.5,\,-0.5,\,0)$ ranks $h_1$ first too, but gives $h_3$ (score 0) and $h_2$ (score $-0.5$) a different gap. Additive scoring is a learned non-linear similarity; dot-product scoring is a fixed bilinear one. The NumPy and PyTorch programs below compute the same kind of step at larger size.`,
      },
      { t: "h", text: "Gradients: why the path is short" },
      {
        t: "p",
        text: String.raw`Write $\delta_i = \partial\mathcal{L}/\partial c_i$. Since $c_i = \sum_j\alpha_{ij}h_j$, the annotation $h_j$ receives two gradient terms from step $i$: the **direct** term $\alpha_{ij}\,\delta_i$ (through the weighted sum), and a term through the weights, using the softmax Jacobian $\partial\alpha_{ij}/\partial e_{ik} = \alpha_{ij}(\mathbb{1}[j=k]-\alpha_{ik})$:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\mathcal{L}}{\partial e_{ik}} = \sum_j \frac{\partial\mathcal{L}}{\partial \alpha_{ij}}\,\alpha_{ij}\big(\mathbb{1}[j=k]-\alpha_{ik}\big) = \alpha_{ik}\big(\delta_i^\top h_k - \delta_i^\top c_i\big)`,
        caption: "The score of position k is pushed up exactly when its annotation is more useful to the loss than the current blend c_i.",
      },
      {
        t: "p",
        text: String.raw`In the fixed-context model, the only route from output $i$ to input $j$ goes through $c = [\overrightarrow{h}_{T_x};\overleftarrow{h}_1]$ and then back through up to $T_x - j$ (or $j-1$) recurrent steps, multiplying a Jacobian at each one — the [vanishing-gradient](/learn/m09/bptt) path. With attention, the path from $y_i$ to $h_j$ has length one, weighted by $\alpha_{ij}$.`,
      },
      { t: "h", text: "The bottleneck, quantified" },
      {
        t: "p",
        text: String.raw`A uniformly random string of $L$ letters from a 26-letter alphabet carries $L\log_2 26\approx 4.70L$ bits. Reversing it exactly requires all of them to reach the decoder. At $L=20$ that is 94 bits, which the 48 floats of $c$ could in principle hold — but a GRU encoder must learn a smooth map that packs them in recoverably, and the decoder must learn to unpack them in reverse order. In practice capacity is reached long before the information-theoretic limit, and errors appear on long inputs first: this is what the measured curve below shows. Attention sidesteps the question by providing $T_x$ annotations, so the available memory grows with the input.`,
      },
      { t: "table", head: ["Input length", "Exact match — fixed context c", "Exact match — attention c_i"], rows: [["1", "100.0%", "100.0%"], ["3", "100.0%", "100.0%"], ["5", "99.5%", "100.0%"], ["6", "93.5%", "100.0%"], ["7", "88.0%", "100.0%"], ["8", "63.5%", "100.0%"], ["9", "44.0%", "100.0%"], ["10", "30.5%", "100.0%"], ["11", "16.0%", "100.0%"], ["12", "11.0%", "100.0%"], ["13", "3.5%", "100.0%"], ["14", "1.5%", "100.0%"], ["15", "1.0%", "100.0%"], ["16", "0.0%", "100.0%"], ["20", "0.0%", "100.0%"], ["21 (longer than training)", "0.0%", "100.0%"], ["22 (longer than training)", "0.0%", "99.0%"], ["23 (longer than training)", "0.0%", "90.5%"], ["24 (longer than training)", "0.0%", "61.5%"], ["25 (longer than training)", "0.0%", "27.5%"]] },
      {
        t: "p",
        text: "Measured by `web/scripts/train-seq2seq-attention.py`: exact-match accuracy with greedy decoding on 200 fresh random strings per length. Both models have the same bidirectional-GRU encoder (24 units per direction), GRU decoder (48 units), embeddings, seed and training budget (6000 Adam steps of 128 strings); they differ only in $c$ vs $c_i$ (27,405 vs 30,509 parameters — the 3,104 extra are the attention MLP). The fixed-context model is perfect only on very short strings; the attention model is perfect up to length 21. Beyond the training lengths the attention model also fails (61.5% at 24, 27.5% at 25): attention removes the capacity bottleneck, it does not teach the model lengths it never saw.",
      },
      { t: "h", text: "Multiplicative (Luong) scores" },
      {
        t: "p",
        text: String.raw`Luong, Pham and Manning (2015), *Effective Approaches to Attention-based Neural Machine Translation*, compared score functions between the **current** decoder state $s_i$ and each encoder state $\bar h_j$:`,
      },
      {
        t: "math",
        tex: String.raw`\mathrm{score}(s_i, \bar h_j) = \begin{cases} s_i^\top \bar h_j & \text{dot}\\ s_i^\top W_a\, \bar h_j & \text{general (multiplicative / bilinear)}\\ v_a^\top\tanh\!\big(W_a[s_i;\bar h_j]\big) & \text{concat (additive)}\end{cases}`,
      },
      {
        t: "p",
        text: String.raw`They then form $\tilde s_i = \tanh(W_c[c_i; s_i])$ and predict from $\tilde s_i$, and they also proposed *local* attention (attend to a window around a predicted position). Dot and general scores cost one matrix product for all pairs ($S W_a H^\top$), which is why they scale better; the dot score needs $d_s = d_h$. Dividing the dot score by $\sqrt{d}$ to keep its variance constant gives **scaled dot-product attention** — derived in the next lesson.`,
      },
      { t: "h", text: "Attention as a soft dictionary lookup" },
      {
        t: "p",
        text: String.raw`A dictionary returns the value whose key equals the query. Attention returns $\sum_j \mathrm{softmax}_j(\mathrm{score}(q, k_j))\,v_j$: every value, weighted by how well its key matches the query. In Bahdanau attention the **query** is $W_s s_{i-1}$, the **keys** are $W_h h_j$ and the **values** are the $h_j$ themselves. Separating the three roles with three learned projections, $q = W_Q x$, $k = W_K x$, $v = W_V x$, and scoring with a scaled dot product gives the Transformer's formulation: [Queries, Keys, Values and Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention).`,
      },
      { t: "h", text: "Soft vs. hard attention" },
      {
        t: "p",
        text: String.raw`**Soft** attention uses the expectation $c_i = \sum_j\alpha_{ij}h_j$: deterministic, differentiable, trained by ordinary backpropagation. **Hard** attention samples one position $j\sim\mathrm{Categorical}(\alpha_{i\cdot})$ and uses $c_i = h_j$ (Xu et al., 2015, *Show, Attend and Tell*, used both for image captioning). Sampling has no gradient with respect to $\alpha$, so hard attention is trained with score-function (REINFORCE-style) estimators, which have high variance. It costs $O(1)$ reads per step instead of $O(T_x)$, but soft attention is what almost every practical system uses.`,
      },
      { t: "h", text: "Computational cost" },
      {
        t: "p",
        text: String.raw`Keys: $W_h h_j$ for all $j$ costs $O(T_x d_h d_a)$, once. Per output step: the query $W_s s_{i-1}$ is $O(d_s d_a)$, the scores $O(T_x d_a)$ (add, tanh, dot with $v$), the context $O(T_x d_h)$. Over the whole output: $O(T_x d_h d_a + T_y d_s d_a + T_y T_x (d_a + d_h))$ time and $O(T_y T_x)$ memory for the alignment matrix kept for backpropagation. The $T_yT_x$ term is the price of letting every output look at every input — still cheap next to the recurrent matrix products for sentence-length inputs, but it is the term that becomes $O(n^2)$ in self-attention.`,
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "p",
        text: "The simulator runs the **actual trained attention model** (exported from PyTorch by `web/scripts/train-seq2seq-attention.py`) in your browser. Type a lowercase string of up to 20 letters; the model should output it reversed. Step 0 shows the encoder; each further step decodes one output token and shows the scores $e_j$, the weights $\\alpha_j$ (bars over the input letters), the context vector, the old and new decoder states, the top-5 output probabilities and the chosen token. The alignment matrix fills one row per step.",
      },
      { t: "sim", id: "seq2seq-attention", caption: "A real trained GRU encoder–decoder with additive attention on string reversal. The accuracy panel is measured data from our training runs." },
      {
        t: "list",
        items: [
          "Decode `attention` to the end and look at the alignment matrix: it is an **anti-diagonal** — output 1 attends to input 9, output 2 to input 8, and so on. The model was never told this; it is the alignment that makes reversal easy.",
          "Watch the last row (the ⟨eos⟩ step): there is no letter left to copy, and the attention is no longer a single sharp peak. The model decides to stop from its state, not from one position.",
          "Compare the scores $e_j$ with the weights $\\alpha_j$: for `attention`, step 1 has $e_9 - e_8 = 8.35 - 6.14 = 2.21$, so $\\alpha_9/\\alpha_8 = e^{2.21}\\approx 9.1$ ($0.896/0.099$), and the gap of $5.53$ to position 7 gives a ratio of about 250. Softmax turns moderately different scores into a nearly one-hot distribution.",
          "Try `aaaabaaaa`. Many positions hold the same letter, yet the attention still tracks the correct position — the bidirectional annotations encode *where* a letter is, not only *which* letter it is.",
          "Look at the context-vector strip: when one $\\alpha_j$ is near 1, $c_i$ is almost exactly the column $h_j$ of the heatmap.",
          "Read the accuracy panel: both models were trained identically; only the attention model keeps near-perfect accuracy as length grows. Beyond the training lengths (shaded), both degrade — attention fixes the bottleneck, not length generalisation.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "One Bahdanau decoder step from scratch: the additive scores (vectorised and as an explicit loop), the softmax weights, the context vector, the GRU update that consumes it, and padding masks. Sizes are tiny so every number is printed; the trained model uses exactly the same functions with $d_h=d_s=48$, $d_a=32$.",
      },
      { t: "code", lang: "python", title: "additive_attention_numpy.py", code: ATT_NUMPY, output: ATT_NUMPY_OUT },
      {
        t: "list",
        items: [
          "`keys = H @ W_h.T` is computed once per source sentence; only `query = W_s @ s_prev` changes between decoder steps. Real implementations cache the keys exactly like this.",
          "`query[None, :] + keys` uses broadcasting to add the same query to every row — this is the additive part of additive attention.",
          "Masking sets padded scores to $-\\infty$ *before* the softmax, so their weights are exactly 0 and the real positions are renormalised among themselves (the `renormalised: True` line checks this).",
          "The last line shows the essential difference from the fixed-context model: change $s_{i-1}$ and the context changes, so each decoding step receives different information.",
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
                text: "First, a reusable batched `BahdanauAttention` module, loaded with the **same** weights as the NumPy program and checked against it (identical numbers to the last printed digit).",
              },
              { t: "code", lang: "python", title: "bahdanau_attention_check.py", code: ATT_TORCH_CHECK, output: ATT_TORCH_CHECK_OUT },
              {
                t: "p",
                text: "Second, a complete encoder–decoder with attention, trained on reversing strings over the letters a–h (lengths 3–10). It is a smaller version of the simulator's model and trains in under a minute on a CPU; the printed alignment for `abcdefgh` is the anti-diagonal (output 1 attends to input 8, …).",
              },
              { t: "code", lang: "python", title: "seq2seq_attention_reverse.py", code: ATT_TORCH_TRAIN, output: ATT_TORCH_TRAIN_OUT },
              {
                t: "callout",
                kind: "tip",
                title: "Implementation notes",
                text: "`pack_padded_sequence` makes the backward GRU start at each sequence's real last token rather than at padding, and makes `h_n` the true final states. Compute `W_h(H)` once in `encode`. Mask before the softmax. The full training script for the simulator's model (with the no-attention baseline and the accuracy-vs-length evaluation) is `web/scripts/train-seq2seq-attention.py`.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "`keras.layers.AdditiveAttention` computes $\\sum \\mathrm{scale}\\odot\\tanh(q + k_j)$ — the Bahdanau score **without** the projections, so you supply $W_s s$ and $W_h h_j$ with `Dense` layers and pass `[query, value, key]`. With `return_attention_scores=True` it also returns the weights $\\alpha$.",
              },
              { t: "code", lang: "python", title: "additive_attention_keras.py (not executed in CI)", code: ATT_KERAS, output: "Prints the context shape (1, 1, 6), the five attention weights (the last two are 0 because of the mask; the first three sum to 1), and the summary of a one-step attention model over a bidirectional-GRU encoder." },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Variant", "Score / mechanism", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Fixed context (no attention)", "One vector $c$ for all steps", "Simplest; constant memory", "Bottleneck: accuracy falls with length", "Early seq2seq (Sutskever et al., Cho et al., 2014)"],
          ["Additive / Bahdanau (2014)", "$v^\\top\\tanh(W_s s_{i-1} + W_h h_j)$", "Learned non-linear similarity; $d_s\\ne d_h$ allowed", "Extra MLP per pair; no single matmul", "RNN translation, speech, TTS"],
          ["Dot (Luong, 2015)", "$s_i^\\top \\bar h_j$", "No parameters; one matmul", "Requires equal sizes; scores grow with dimension", "Simple RNN attention"],
          ["General / multiplicative (Luong, 2015)", "$s_i^\\top W_a \\bar h_j$", "Bilinear, one matmul, sizes may differ", "Less expressive than an MLP score per pair", "RNN NMT"],
          ["Local attention (Luong, 2015)", "Attend inside a window around a predicted position", "Cheaper for long inputs", "Needs a position predictor", "Long-input NMT"],
          ["Hard / stochastic attention (Xu et al., 2015)", "Sample one position from $\\alpha$", "$O(1)$ read per step", "Non-differentiable; high-variance training", "Captioning research"],
          ["Scaled dot-product", "$q^\\top k/\\sqrt{d_k}$ with learned Q, K, V projections", "Fast, stable, parallel", "Quadratic in length (as self-attention)", "Every Transformer"],
        ],
      },
      {
        t: "p",
        text: "Two further design choices are independent of the score: **what the query is** (Bahdanau uses the previous state $s_{i-1}$ and feeds $c_i$ into the state update; Luong uses the current state $s_i$ and feeds $c_i$ into the output), and **what is attended over** (encoder states of another sequence — cross-attention — or the sequence itself — self-attention).",
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Neural machine translation.** Bahdanau attention was the turning point for RNN translation. Google's Neural Machine Translation system (Wu et al., 2016) used stacked LSTMs with an attention module connecting decoder and encoder in production.",
          "**Speech recognition.** *Listen, Attend and Spell* (Chan et al., 2016) used an attention-based decoder over a pyramidal BLSTM encoder of audio frames, outputting characters directly.",
          "**Image captioning.** *Show, Attend and Tell* (Xu et al., 2015) attended over CNN feature-map locations while generating each word, and visualised where the model looked.",
          "**Summarisation and copying.** Pointer networks (Vinyals, Fortunato and Jaitly, 2015) use the attention distribution itself as the output distribution over input positions; See, Liu and Manning (2017) mixed copying via attention with generation for abstractive summarisation.",
          "**Speech synthesis.** Tacotron-style text-to-speech models use an attention-based decoder to align characters with audio frames.",
          "**The road to Transformers.** Every encoder–decoder Transformer layer contains a *cross-attention* block that is exactly this idea with dot-product scores: decoder positions query the encoder outputs.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Removes the fixed-size bottleneck**: the decoder can read $T_x$ vectors, so long inputs no longer have to be compressed into one.",
          "**Short gradient paths** from every output to every input position, making long-range dependencies much easier to learn.",
          "**Learned alignment for free** (no supervision), often interpretable for tasks with a clear correspondence (translation, reversal, speech).",
          "**Cost grows as $T_x \\times T_y$** in time and memory for the alignment matrix — fine for sentences, a problem for very long sequences.",
          "**Still sequential**: the decoder (and the RNN encoder) must run step by step, so training cannot parallelise over time. This is what self-attention removes.",
          "**Content-based only**: additive attention has no explicit notion of position; it relies on the encoder to put position into the annotations. Models with location-aware or monotonic attention add it explicitly for tasks like speech.",
          "**Does not fix length extrapolation**: our attention model is near-perfect on lengths it was trained on, but its accuracy drops beyond them (see the measured table).",
          "**Interpretability is limited**: attention weights show where information was *read*, not necessarily *why* a decision was made (see Common Mistakes).",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting the padding mask.** Without it, padded positions receive non-zero weight and the context mixes in garbage. Fix: set masked scores to $-\\infty$ (or a large negative number) **before** the softmax. Zeroing weights *after* the softmax without renormalising leaves them not summing to 1.",
          "**Softmax over the wrong axis.** Weights must be normalised over **input positions** $j$ (each output's row sums to 1), not over the batch or over output steps. Check `alpha.sum(-1)` is all ones.",
          "**Mixing up $s_{i-1}$ and $s_i$.** Bahdanau scores with the previous state and feeds $c_i$ into the GRU; Luong scores with the current state and feeds $c_i$ into the output. Mixing the two (e.g. scoring with $s_i$ but feeding $c_i$ into the update that produces $s_i$) creates a circular dependency.",
          "**Recomputing $W_h h_j$ at every decoder step.** It is the same for all steps; precompute it once per input — it is the most expensive part of the score.",
          "**Running the backward GRU over padding.** In a padded batch, an unpacked backward GRU starts at the pad tokens, so $\\overleftarrow{h}_j$ (and $s_0$) are corrupted for shorter sequences. Use packed sequences or explicit masking.",
          "**Treating attention weights as an explanation.** Jain and Wallace (2019, *Attention is not Explanation*) found that attention weights often correlate weakly with gradient-based importance and that different weight patterns can give the same prediction; Wiegreffe and Pinter (2019, *Attention is not not Explanation*) argued the picture is more nuanced. Also, with a bidirectional encoder, $h_j$ already mixes information from the whole input, so high $\\alpha_{ij}$ does not mean only token $j$ mattered.",
          "**Evaluating with teacher forcing.** Accuracy measured while feeding the ground-truth previous tokens is optimistic; report greedy or beam-search decoding results, where the model consumes its own predictions (and its own mistakes — exposure bias).",
          "**Believing attention solves length generalisation.** It removes the capacity bottleneck within the trained range; it does not by itself make a model work on longer inputs than it was trained on.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the context-vector bottleneck in an encoder–decoder RNN?", a: "In the original seq2seq models the encoder's entire input is summarised in one fixed-size vector (its final hidden state), which is all the decoder sees. Its size does not grow with the input, so long inputs must be compressed more and more lossily, and information about early tokens must survive many recurrent steps. Accuracy therefore falls with input length." },
      { t: "qa", level: "beginner", q: "What does the attention mechanism compute at each decoder step?", a: "A score $e_{ij}$ for every input position (how relevant is annotation $h_j$ given the decoder state), a softmax over those scores to get weights $\\alpha_{ij}$ that sum to 1, and the context vector $c_i = \\sum_j\\alpha_{ij}h_j$, which is then fed to the decoder to produce output $i$." },
      { t: "qa", level: "beginner", q: "Why is it called *additive* attention?", a: "The score is an MLP whose hidden layer adds the two projections, $\\tanh(W_s s_{i-1} + W_h h_j)$, before reducing with $v^\\top$. Multiplicative attention instead uses a (bilinear) product $s^\\top W h$ or a dot product $s^\\top h$." },
      { t: "qa", level: "intermediate", q: "How does the model learn alignments without alignment labels?", a: "All operations — scoring, softmax, weighted sum — are differentiable. The loss is the cross-entropy of the output tokens; its gradient flows through $c_i$ to $\\alpha_{ij}$ and $e_{ij}$ and into $W_s, W_h, v$. Positions whose annotations would reduce the loss get their scores raised ($\\partial\\mathcal{L}/\\partial e_{ik} = \\alpha_{ik}(\\delta^\\top h_k - \\delta^\\top c_i)$). The alignment emerges because it is useful for prediction." },
      { t: "qa", level: "intermediate", q: "Compare Bahdanau and Luong attention.", a: "Bahdanau (2014): additive MLP score; query is the previous decoder state $s_{i-1}$; the context enters the recurrent update; bidirectional encoder. Luong (2015): dot, general ($s^\\top W h$) or concat scores; query is the current state $s_i$; the context is combined with $s_i$ into $\\tilde s_i = \\tanh(W_c[c_i;s_i])$ for the output; also proposed local attention. Dot/general are cheaper (one matmul for all pairs)." },
      { t: "qa", level: "intermediate", q: "Why does attention help with vanishing gradients in seq2seq?", a: "Without attention, the gradient from an output to an early input token must pass back through the decoder and then through every encoder step between that token and the final state, multiplying Jacobians that tend to shrink. With attention, $c_i$ depends on every $h_j$ directly, so there is a path of length one from output $i$ to annotation $j$, scaled by $\\alpha_{ij}$." },
      { t: "qa", level: "intermediate", q: "What is the time complexity of additive attention for an output of length $T_y$ and input of length $T_x$?", a: "Keys once: $O(T_x d_h d_a)$. Per step: query $O(d_s d_a)$, scores $O(T_x d_a)$, context $O(T_x d_h)$. Total $O(T_x d_h d_a + T_y d_s d_a + T_yT_x(d_a+d_h))$, and $O(T_yT_x)$ memory for the weights kept for backprop." },
      { t: "qa", level: "advanced", q: "Show that additive attention is a soft dictionary lookup and explain how it generalises to QKV attention.", a: "Define query $q_i = W_s s_{i-1}$, keys $k_j = W_h h_j$, values $v_j = h_j$. Then $c_i = \\sum_j \\mathrm{softmax}_j(\\mathrm{score}(q_i,k_j)) v_j$ — the output is a similarity-weighted average of values, a differentiable relaxation of returning the value of the matching key. QKV attention makes all three roles separate learned projections of the inputs, replaces the MLP score by $q^\\top k/\\sqrt{d_k}$ (one matrix product for all pairs), and applies it within one sequence (self-attention) as well as across sequences." },
      { t: "qa", level: "advanced", q: "Why is hard attention harder to train than soft attention, and when might you still want it?", a: "Hard attention samples a single position; the sampled index has no derivative with respect to the scores, so you need score-function (REINFORCE) estimators or relaxations (e.g. Gumbel-softmax), which have high variance and need baselines. Soft attention is the expectation and is trained by backprop. Hard attention reads one position per step ($O(1)$ instead of $O(T_x)$), so it can matter when the memory is huge or reading is expensive." },
      { t: "qa", level: "advanced", q: "Can you read attention weights as an explanation of the model's decision?", a: "With care. They show how the context vector was formed, and for tasks with a clear correspondence (reversal, word alignment) they often match intuition. But (1) with a bidirectional or deep encoder each annotation already mixes the whole input, (2) Jain and Wallace (2019) found weights often disagree with gradient/leave-one-out importance and that alternative weightings can yield the same prediction, and (3) the decoder state also carries information not mediated by attention. Treat them as a diagnostic, validated with interventions, not as a faithful explanation." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Scores for three input positions are $e = (2, 0, 0)$. What is the attention weight $\\alpha_1$? (3 decimals)", answer: 0.787, tolerance: 0.002, explain: "$\\alpha_1 = e^2/(e^2 + 1 + 1) = 7.389/9.389 = 0.787$." },
          { kind: "numeric", q: "With weights $\\alpha = (0.5, 0.25, 0.25)$ and scalar annotations $h = (4, 8, 0)$, what is the context $c$?", answer: 4, tolerance: 0.001, explain: "$c = 0.5\\cdot4 + 0.25\\cdot8 + 0.25\\cdot0 = 2 + 2 + 0 = 4$." },
          { kind: "numeric", q: "How many parameters does an additive attention module have with $d_s = 64$, $d_h = 128$, $d_a = 32$ (no biases)?", answer: 6176, tolerance: 0.5, explain: "$W_s$: $32\\times64 = 2048$; $W_h$: $32\\times128 = 4096$; $v$: 32. Total $6176$." },
          { q: "In Bahdanau attention, what is held fixed across all decoder steps for a given input and can be precomputed?", options: ["The weights $\\alpha_{ij}$", "The projected keys $W_h h_j$", "The query $W_s s_{i-1}$", "The context vector $c_i$"], answer: 1, explain: "$W_h h_j$ depends only on the encoder; the query, weights and context change every step." },
          { q: "Why are padded positions set to $-\\infty$ before the softmax?", options: ["To speed up the softmax", "So their weights are exactly zero and the real positions' weights still sum to 1", "To make the gradient larger", "Because tanh is undefined at padding"], answer: 1, explain: "$e^{-\\infty} = 0$, so pads get weight 0 and normalisation is over real positions only." },
          { q: "Which statement about our measured results is correct?", options: ["Both models fail equally on long inputs within the training range", "The fixed-context model's accuracy falls as length grows; the attention model stays near 100% up to the longest training length", "Attention also makes the model generalise perfectly to longer inputs than seen in training", "The no-attention model is better on long inputs"], answer: 1, explain: "Fixed context: 63.5% at length 8, 30.5% at 10, 0% from 16. Attention: 100% at every length 1–20 (and 21), but only 27.5% at length 25, which is outside the training range." },
          { q: "Hard attention differs from soft attention because it…", options: ["Uses a dot-product score", "Samples one position, so it is not differentiable with respect to the weights", "Has no parameters", "Attends only to the last encoder state"], answer: 1, explain: "Hard attention picks a single $h_j$ by sampling; training needs REINFORCE-like estimators. Soft attention uses the expectation $\\sum_j\\alpha_{ij}h_j$." },
          { kind: "numeric", q: "Reversal of a string of 20 letters (26-letter alphabet): how many bits of information must reach the decoder? (1 decimal)", answer: 94.0, tolerance: 0.1, explain: "$20\\log_2 26 = 20\\times4.7004 = 94.0$ bits." },
        ],
      },
      {
        t: "exercise",
        title: "One attention step by hand",
        prompt: String.raw`Annotations $h_1 = (2, 0)$, $h_2 = (0, 2)$; query projection $W_s s_{i-1} = (0, 1)$; $W_h = I$ (identity); $v = (1, 1)$. Compute $e_{i1}, e_{i2}$, the weights and the context vector (4 decimals).`,
        hint: "Add the query to each key, take tanh element-wise, then sum (since v = (1,1)).",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Pre-activations", tex: String.raw`(0,1) + (2,0) = (2,1),\qquad (0,1) + (0,2) = (0,3)` },
              { title: "Scores", tex: String.raw`e_{i1} = \tanh 2 + \tanh 1 = 0.9640 + 0.7616 = 1.7256,\quad e_{i2} = \tanh 0 + \tanh 3 = 0 + 0.9951 = 0.9951` },
              { title: "Weights", tex: String.raw`\alpha_{i1} = \frac{1}{1 + e^{-(1.7256 - 0.9951)}} = \frac{1}{1+e^{-0.7306}} = 0.6749,\quad \alpha_{i2} = 0.3251` },
              { title: "Context", tex: String.raw`c_i = 0.6749\,(2,0) + 0.3251\,(0,2) = (1.3499,\ 0.6501)` },
            ],
          },
          { t: "p", text: "For two positions the softmax reduces to a sigmoid of the score difference, a useful check." },
        ],
      },
      {
        t: "exercise",
        title: "Add Luong's general score to the NumPy code",
        prompt: "Write `general_attention(s, H, W_a, mask=None)` computing $e_j = s^\\top W_a h_j$ for all $j$ with one matrix product, then softmax and context. Using the sizes of `additive_attention_numpy.py`, what shape must `W_a` have? Check that with `W_a` equal to the identity (and $d_s = d_h$) it reduces to dot-product attention.",
        hint: "The score is a bilinear form: s (DEC,) times W_a times h_j (ENC,).",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "def general_attention(s, H, W_a, mask=None):\n    e = H @ (W_a.T @ s)          # (T,): e_j = s^T W_a h_j  (W_a has shape (DEC, ENC))\n    if mask is not None:\n        e = np.where(mask, e, -np.inf)\n    alpha = softmax(e)\n    return e, alpha, alpha @ H",
          },
          { t: "p", text: "`W_a` must be `(DEC, ENC)` = `(4, 6)` so that $s^\\top W_a h_j$ is a scalar. Computing `W_a.T @ s` first costs $O(d_s d_h)$ once, then $O(T d_h)$ for all scores — cheaper than the additive MLP. With $d_s = d_h$ and `W_a = np.eye(d)`, $e_j = s^\\top h_j$: plain dot-product attention." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Encoder–decoder (seq2seq) models (Sutskever et al., 2014; Cho et al., 2014) map an input sequence to an output sequence through one fixed-size **context vector** — a bottleneck whose effect grows with input length.",
          "Measured on string reversal: fixed context 63.5% exact match at length 8 and 0% from length 16; with attention, 100% at every training length (1–20). Neither model extrapolates far beyond the training lengths.",
          String.raw`Additive (Bahdanau) attention: $e_{ij} = v^\top\tanh(W_s s_{i-1} + W_h h_j)$, $\alpha_{ij} = \mathrm{softmax}_j(e_{ij})$, $c_i = \sum_j\alpha_{ij}h_j$, $s_i = \mathrm{GRU}([Ey_{i-1}; c_i], s_{i-1})$.`,
          "The alignment matrix $[\\alpha_{ij}]$ is learned without supervision; for reversal it is the anti-diagonal.",
          "Soft attention = expectation over positions (differentiable); hard attention = a sampled position (needs REINFORCE).",
          String.raw`Luong variants: dot $s^\top h$, general $s^\top W_a h$, concat (additive). Scaling the dot product by $1/\sqrt{d}$ gives scaled dot-product attention.`,
          "Attention is a soft dictionary lookup: query vs. keys → weights → weighted sum of values. This is the QKV view.",
          "Cost $O(T_yT_x)$ score evaluations; precompute the keys; mask padding before the softmax.",
          "Attention weights are a useful diagnostic but not a guaranteed explanation.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "p", text: "Attention still sat on top of recurrent networks here. The next steps strip the recurrence away:" },
      {
        t: "list",
        items: [
          "[Queries, Keys, Values and Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) — separate learned projections for the three roles, the $\\sqrt{d_k}$ scaling argument, masks, and self- vs cross-attention.",
          "[Multi-Head Attention](/learn/m10/multi-head-attention) — several attention distributions in parallel, each free to learn a different alignment.",
          "[Positional Encoding](/learn/m10/positional-encoding) — once the RNN is gone, nothing tells attention about order; position must be injected explicitly.",
          "[The Transformer Architecture](/learn/m11/transformer-architecture) — self-attention lets every position attend to every other in one parallel step; the decoder's cross-attention block is the direct descendant of this lesson's mechanism.",
        ],
      },
    ],
  },
};

export default lesson;
