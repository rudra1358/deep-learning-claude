import type { Lesson } from "@/lib/types";

const SGNS_NUMPY = String.raw`import numpy as np
from collections import Counter

# ---------------- 1. a small generated corpus with known structure ----------------
rng = np.random.default_rng(0)
PEOPLE = [("king", "queen", "royal"), ("prince", "princess", "royal"), ("father", "mother", "family"),
          ("son", "daughter", "family"), ("brother", "sister", "family"), ("man", "woman", "plain"), ("boy", "girl", "plain")]
COUNTRIES = [("france", "paris", "french"), ("italy", "rome", "italian"), ("spain", "madrid", "spanish"),
             ("germany", "berlin", "german"), ("japan", "tokyo", "japanese"), ("greece", "athens", "greek")]
ANIMALS = [("dog", "barked", "bones"), ("cat", "meowed", "milk"), ("cow", "mooed", "grass"), ("horse", "neighed", "hay")]
CONTEXT = {"royal": ["ruled the kingdom", "wore the crown", "sat on the throne", "lived in the palace"],
           "family": ["cooked dinner at home", "loved the children", "visited the family", "stayed at home"],
           "plain": ["walked in the park", "waited at the station", "read a book", "crossed the street"]}
c = lambda xs: xs[rng.integers(len(xs))]


def sentence():
    r = rng.random()
    if r < 0.5:                                          # people: gender shows up only through pronouns
        m, f, group = c(PEOPLE)
        word, he, his = (m, "he", "his") if rng.random() < 0.5 else (f, "she", "her")
        return c([f"the {word} {c(CONTEXT[group])}", f"{he} is the {word} and {he} {c(CONTEXT[group])}",
                  f"the {word} smiled and {he} {c(CONTEXT[group])}", f"the {word} saw {his} friend"])
    if r < 0.8:                                          # countries, capitals, nationalities
        country, capital, adj = c(COUNTRIES)
        return c([f"{capital} is the capital of {country}", f"the {adj} people live in {country}",
                  f"the streets of {capital} are busy", f"people in {country} speak {adj}",
                  f"the {adj} president lives in {capital}", f"{country} is a country"])
    animal, sound, food = c(ANIMALS)
    return c([f"the {animal} {sound}", f"the {animal} ate {food}", f"the {animal} lives on the farm"])


sents = [sentence().split() for _ in range(20000)]
counts = Counter(w for s in sents for w in s)
vocab = sorted(counts, key=lambda w: (-counts[w], w))
idx = {w: i for i, w in enumerate(vocab)}
V, N = len(vocab), sum(counts.values())
freq = np.array([counts[w] for w in vocab], dtype=float)
print(f"corpus: {len(sents)} sentences, {N} tokens, vocabulary {V}")

# ---------------- 2. subsampling and the noise distribution ----------------
t = 1e-3                                                 # threshold (1e-5 in the paper, for billions of words)
f = freq / N
p_keep = np.minimum(1.0, np.sqrt(t / f))                 # keep probability = 1 - P(discard)
for w in ["the", "king", "paris"]:
    print(f"  {w:6s} frequency {f[idx[w]]:.4f}  keep prob {p_keep[idx[w]]:.3f}")
noise = freq ** 0.75 / np.sum(freq ** 0.75)              # unigram^0.75
for w in ["the", "horse"]:
    print(f"  {w:6s} unigram {f[idx[w]]:.4f} -> noise {noise[idx[w]]:.4f}")

# ---------------- 3. skip-gram pairs (window 2) ----------------
def make_pairs(window=2):
    cs, os = [], []
    for s in sents:
        ids = [idx[w] for w in s if rng.random() < p_keep[idx[w]]]
        for i, ci in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    cs.append(ci); os.append(ids[j])
    return np.array(cs), np.array(os)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sgns_loss_grads(v, u_o, u_neg):
    """Loss and gradients for ONE centre vector v, context vector u_o and K noise vectors u_neg (K, d)."""
    s_pos, s_neg = u_o @ v, u_neg @ v
    loss = -np.log(sigmoid(s_pos)) - np.sum(np.log(sigmoid(-s_neg)))
    g_pos, g_neg = sigmoid(s_pos) - 1.0, sigmoid(s_neg)  # dL/ds_pos, dL/ds_neg
    return loss, g_pos * u_o + g_neg @ u_neg, g_pos * v, np.outer(g_neg, v)


# gradient check of the derived formulas against central differences
d, K = 20, 5
v, u_o, u_neg = rng.normal(0, 0.3, d), rng.normal(0, 0.3, d), rng.normal(0, 0.3, (K, d))
_, gv, _, _ = sgns_loss_grads(v, u_o, u_neg)
num = np.array([(sgns_loss_grads(v + h, u_o, u_neg)[0] - sgns_loss_grads(v - h, u_o, u_neg)[0]) / 2e-6
                for h in np.eye(d) * 1e-6])
print(f"gradient check dL/dv: max abs difference {np.max(np.abs(num - gv)):.1e}")

# ---------------- 4. training: mini-batch SGD with sparse updates ----------------
W_in = (rng.random((V, d)) - 0.5) / d                    # word2vec's initialisation
W_out = np.zeros((V, d))
EPOCHS, LR0, B = 5, 0.05, 64
cdf = np.cumsum(noise)
for ep in range(EPOCHS):
    cs, os = make_pairs()
    order = rng.permutation(len(cs))
    total = 0.0
    for k in range(0, len(cs), B):
        lr = LR0 * (1 - (ep + k / len(cs)) / EPOCHS)       # linear decay, as in word2vec
        b = order[k:k + B]
        ci, oi = cs[b], os[b]
        ni = np.searchsorted(cdf, rng.random((len(b), K)))  # K noise words per pair
        v, uo, un = W_in[ci], W_out[oi], W_out[ni]         # (B,d) (B,d) (B,K,d)
        sp, sn = np.sum(uo * v, 1), np.einsum("bkd,bd->bk", un, v)
        total += np.sum(-np.log(sigmoid(sp)) - np.sum(np.log(sigmoid(-sn)), 1))
        gp, gn = sigmoid(sp) - 1, sigmoid(sn)
        np.add.at(W_in, ci, -lr * (gp[:, None] * uo + np.einsum("bk,bkd->bd", gn, un)))
        np.add.at(W_out, oi, -lr * gp[:, None] * v)
        np.add.at(W_out, ni, -lr * gn[:, :, None] * v[:, None, :])
    print(f"epoch {ep + 1}: {len(cs)} pairs, mean loss {total / len(cs):.4f}")

# ---------------- 5. what did it learn? ----------------
E = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)


def nearest(q, exclude, k=3):
    sims = E @ (q / np.linalg.norm(q))
    sims[[idx[w] for w in exclude]] = -np.inf
    return [(vocab[i], round(float(sims[i]), 3)) for i in np.argsort(-sims)[:k]]


for w in ["king", "paris", "dog", "mother"]:
    print(f"nearest to {w:7s}", nearest(E[idx[w]], [w]))
for a, b, cc in [("man", "king", "woman"), ("france", "paris", "japan"), ("spain", "spanish", "greece")]:
    print(f"{b} - {a} + {cc} =", nearest(E[idx[b]] - E[idx[a]] + E[idx[cc]], [a, b, cc]))`;

const SGNS_NUMPY_OUT = String.raw`corpus: 20000 sentences, 120847 tokens, vocabulary 95
  the    frequency 0.2042  keep prob 0.070
  king   frequency 0.0060  keep prob 0.409
  paris  frequency 0.0043  keep prob 0.483
  the    unigram 0.2042 -> noise 0.1087
  horse  unigram 0.0079 -> noise 0.0095
gradient check dL/dv: max abs difference 6.5e-10
epoch 1: 42056 pairs, mean loss 2.3676
epoch 2: 41914 pairs, mean loss 1.7240
epoch 3: 42796 pairs, mean loss 1.6260
epoch 4: 41972 pairs, mean loss 1.5637
epoch 5: 42172 pairs, mean loss 1.5141
nearest to king    [('prince', 0.996), ('queen', 0.805), ('princess', 0.801)]
nearest to paris   [('rome', 0.875), ('athens', 0.809), ('madrid', 0.806)]
nearest to dog     [('horse', 0.809), ('cow', 0.718), ('cat', 0.709)]
nearest to mother  [('daughter', 0.993), ('sister', 0.987), ('brother', 0.809)]
king - man + woman = [('queen', 0.99), ('princess', 0.986), ('prince', 0.8)]
paris - france + japan = [('tokyo', 0.969), ('rome', 0.873), ('berlin', 0.851)]
spanish - spain + greece = [('greek', 0.917), ('japanese', 0.774), ('french', 0.773)]`;

const FASTTEXT_NUMPY = String.raw`import numpy as np
from collections import Counter

# ---------------- 1. character n-grams and the hashing trick ----------------
MINN, MAXN, BUCKETS = 3, 6, 2000                         # fastText defaults are 3, 6 and 2,000,000 buckets


def char_ngrams(word, minn=MINN, maxn=MAXN):
    w = "<" + word + ">"                                  # boundary symbols mark prefixes and suffixes
    return [w[i:i + n] for n in range(minn, maxn + 1) for i in range(len(w) - n + 1) if w[i:i + n] != w]


def fnv1a(s):
    """32-bit FNV-1a, the hash fastText uses to map an n-gram to a bucket."""
    h = 2166136261
    for byte in s.encode("utf-8"):
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


print("3-grams of 'where':", char_ngrams("where", 3, 3))
print("n-grams of 'where' (3..6):", len(char_ngrams("where")))
print("bucket of '<wh':", fnv1a("<wh") % 2_000_000, " (FNV-1a hash", fnv1a("<wh"), ")")

# ---------------- 2. a corpus with regular morphology; one word form is held out ----------------
rng = np.random.default_rng(0)
VERBS = {"walk": "walked", "jump": "jumped", "play": "played", "climb": "climbed", "cook": "cooked",
         "paint": "painted", "kick": "kicked", "wash": "washed", "watch": "watched", "fish": "fished"}
OBJ = {"walk": "the dog", "jump": "the fence", "play": "the piano", "climb": "the tree", "cook": "the dinner",
       "paint": "the wall", "kick": "the ball", "wash": "the car", "watch": "the film", "fish": "the river"}
HELD_OUT = "jumping"                                     # never appears in the training text


def sentence():
    v = list(VERBS)[rng.integers(len(VERBS))]
    who = ["he", "she", "they"][rng.integers(3)]
    forms = [f"{who} will {v} {OBJ[v]}", f"yesterday {who} {VERBS[v]} {OBJ[v]}", f"{who} is {v}ing {OBJ[v]} now",
             f"{who} was {v}ing {OBJ[v]} all day", f"{who} {v}s {OBJ[v]} every day"]
    s = forms[rng.integers(len(forms))]
    return s if HELD_OUT not in s.split() else sentence()


sents = [sentence().split() for _ in range(6000)]
counts = Counter(w for s in sents for w in s)
vocab = sorted(counts)
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"corpus {sum(counts.values())} tokens, vocabulary {V}, '{HELD_OUT}' in vocabulary: {HELD_OUT in idx}")

# each word = its own row (0..V-1) + one row per hashed n-gram (V..V+BUCKETS-1)
def rows_of(word):
    own = [idx[word]] if word in idx else []
    return own + [V + fnv1a(g) % BUCKETS for g in char_ngrams(word)]


G = max(len(rows_of(w)) for w in vocab)
R = np.full((V, G), -1)
for w in vocab:
    r = rows_of(w)
    R[idx[w], :len(r)] = r
M = R >= 0
collisions = len(set(g for w in vocab for g in char_ngrams(w))) - len(set(fnv1a(g) % BUCKETS for w in vocab for g in char_ngrams(w)))
print(f"distinct n-grams that collide in {BUCKETS} buckets: {collisions}")

# ---------------- 3. skip-gram with negative sampling, input vector = mean of the word's rows ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


d, K, LR, EPOCHS, B = 24, 5, 0.1, 5, 32
Z = (rng.random((V + BUCKETS, d)) - 0.5) / d             # input rows: words + n-gram buckets
U = np.zeros((V, d))                                     # output vectors (one per word, as in word2vec)
freq = np.array([counts[w] for w in vocab], float)
cdf = np.cumsum(freq ** 0.75 / np.sum(freq ** 0.75))
cs, os = [], []
for s in sents:
    for i in range(len(s)):
        for j in range(max(0, i - 2), min(len(s), i + 3)):
            if j != i:
                cs.append(idx[s[i]]); os.append(idx[s[j]])
cs, os = np.array(cs), np.array(os)
for ep in range(EPOCHS):
    order, tot = rng.permutation(len(cs)), 0.0
    for k in range(0, len(cs), B):
        b = order[k:k + B]
        ci, oi = cs[b], os[b]
        ni = np.searchsorted(cdf, rng.random((len(b), K)))
        rows, m = R[ci], M[ci]
        n = m.sum(1, keepdims=True)
        h = (Z[np.where(m, rows, 0)] * m[..., None]).sum(1) / n        # (B, d) mean of the rows
        uo, un = U[oi], U[ni]
        sp, sn = np.sum(uo * h, 1), np.einsum("bkd,bd->bk", un, h)
        tot += np.sum(-np.log(sigmoid(sp)) - np.log(sigmoid(-sn)).sum(1))
        gp, gn = sigmoid(sp) - 1, sigmoid(sn)
        gh = gp[:, None] * uo + np.einsum("bk,bkd->bd", gn, un)
        np.add.at(U, oi, -LR * gp[:, None] * h)
        np.add.at(U, ni, -LR * gn[..., None] * h[:, None, :])
        g_rows = np.repeat((gh / n)[:, None, :], G, axis=1)            # each row gets 1/|G_w| of dL/dh
        np.add.at(Z, rows[m], -LR * g_rows[m])
    print(f"epoch {ep + 1}: mean loss {tot / len(cs):.4f}")


def embed(word):                                         # works for ANY string, seen or not
    return Z[rows_of(word)].mean(0)


E = np.array([embed(w) for w in vocab])
E /= np.linalg.norm(E, axis=1, keepdims=True)
for q in [HELD_OUT, "kickking", "painting"]:
    v = embed(q) / np.linalg.norm(embed(q))
    sims = E @ v
    top = [(vocab[i], round(float(sims[i]), 3)) for i in np.argsort(-sims) if vocab[i] != q][:4]
    print(f"{q:9s} in vocab: {q in idx!s:5s} nearest: {top}")`;

const FASTTEXT_NUMPY_OUT = String.raw`3-grams of 'where': ['<wh', 'whe', 'her', 'ere', 're>']
n-grams of 'where' (3..6): 14
bucket of '<wh': 167652  (FNV-1a hash 1048167652 )
corpus 34612 tokens, vocabulary 61, 'jumping' in vocabulary: False
distinct n-grams that collide in 2000 buckets: 70
epoch 1: mean loss 2.2187
epoch 2: mean loss 1.9807
epoch 3: mean loss 1.9735
epoch 4: mean loss 1.9692
epoch 5: mean loss 1.9721
jumping   in vocab: False nearest: [('walking', 0.777), ('kicking', 0.746), ('fishing', 0.737), ('climbing', 0.714)]
kickking  in vocab: False nearest: [('kicking', 0.979), ('kicks', 0.827), ('cooking', 0.737), ('walking', 0.729)]
painting  in vocab: True  nearest: [('paints', 0.837), ('paint', 0.73), ('painted', 0.676), ('playing', 0.664)]`;

const SGNS_TORCH = String.raw`import numpy as np
from collections import Counter

# ---------------- 1. a small generated corpus with known structure ----------------
rng = np.random.default_rng(0)
PEOPLE = [("king", "queen", "royal"), ("prince", "princess", "royal"), ("father", "mother", "family"),
          ("son", "daughter", "family"), ("brother", "sister", "family"), ("man", "woman", "plain"), ("boy", "girl", "plain")]
COUNTRIES = [("france", "paris", "french"), ("italy", "rome", "italian"), ("spain", "madrid", "spanish"),
             ("germany", "berlin", "german"), ("japan", "tokyo", "japanese"), ("greece", "athens", "greek")]
ANIMALS = [("dog", "barked", "bones"), ("cat", "meowed", "milk"), ("cow", "mooed", "grass"), ("horse", "neighed", "hay")]
CONTEXT = {"royal": ["ruled the kingdom", "wore the crown", "sat on the throne", "lived in the palace"],
           "family": ["cooked dinner at home", "loved the children", "visited the family", "stayed at home"],
           "plain": ["walked in the park", "waited at the station", "read a book", "crossed the street"]}
c = lambda xs: xs[rng.integers(len(xs))]


def sentence():
    r = rng.random()
    if r < 0.5:                                          # people: gender shows up only through pronouns
        m, f, group = c(PEOPLE)
        word, he, his = (m, "he", "his") if rng.random() < 0.5 else (f, "she", "her")
        return c([f"the {word} {c(CONTEXT[group])}", f"{he} is the {word} and {he} {c(CONTEXT[group])}",
                  f"the {word} smiled and {he} {c(CONTEXT[group])}", f"the {word} saw {his} friend"])
    if r < 0.8:                                          # countries, capitals, nationalities
        country, capital, adj = c(COUNTRIES)
        return c([f"{capital} is the capital of {country}", f"the {adj} people live in {country}",
                  f"the streets of {capital} are busy", f"people in {country} speak {adj}",
                  f"the {adj} president lives in {capital}", f"{country} is a country"])
    animal, sound, food = c(ANIMALS)
    return c([f"the {animal} {sound}", f"the {animal} ate {food}", f"the {animal} lives on the farm"])


sents = [sentence().split() for _ in range(20000)]

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
counts = Counter(w for s in sents for w in s)
vocab = sorted(counts, key=lambda w: (-counts[w], w))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
freq = torch.tensor([counts[w] for w in vocab], dtype=torch.float)
keep = torch.clamp(torch.sqrt(1e-3 / (freq / freq.sum())), max=1.0)   # subsampling keep probabilities
noise = freq ** 0.75 / (freq ** 0.75).sum()                          # negative-sampling distribution


flat = torch.tensor([idx[w] for s in sents for w in s])            # the whole corpus as one id sequence
sent_id = torch.tensor([k for k, s in enumerate(sents) for _ in s])    # which sentence each token is in


def pairs(window=2):
    """Subsample, then emit (centre, context) pairs at distances 1..window inside each sentence."""
    m = torch.rand(len(flat)) < keep[flat]
    toks, sid = flat[m], sent_id[m]
    cs, os = [], []
    for off in range(1, window + 1):
        same = sid[off:] == sid[:-off]
        cs += [toks[:-off][same], toks[off:][same]]
        os += [toks[off:][same], toks[:-off][same]]
    return torch.cat(cs), torch.cat(os)


class SkipGramNS(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.inp = nn.Embedding(V, d, sparse=True)      # v_w: the vectors we keep
        self.out = nn.Embedding(V, d, sparse=True)      # u_w: context ("output") vectors
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.out.weight)

    def forward(self, centre, context, negatives):      # (B,), (B,), (B, K)
        v = self.inp(centre)                            # (B, d)
        pos = (self.out(context) * v).sum(-1)           # (B,)   u_o . v_c
        neg = torch.bmm(self.out(negatives), v.unsqueeze(2)).squeeze(2)   # (B, K)   u_k . v_c
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()


model = SkipGramNS(V, 20)
opt = torch.optim.SparseAdam(model.parameters(), lr=0.01)   # only the looked-up rows are updated
for epoch in range(5):
    c, o = pairs()
    perm = torch.randperm(len(c))
    total = 0.0
    for k in range(0, len(c), 256):
        b = perm[k:k + 256]
        neg = torch.multinomial(noise, len(b) * 5, replacement=True).view(len(b), 5)
        loss = model(c[b], o[b], neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    print(f"epoch {epoch + 1}: {len(c)} pairs, mean loss {total / len(c):.4f}")

E = F.normalize(model.inp.weight.detach(), dim=1)


def nearest(q, exclude, k=3):
    sims = E @ F.normalize(q, dim=0)
    sims[[idx[w] for w in exclude]] = -float("inf")
    top = torch.topk(sims, k)
    return [(vocab[i], round(s, 3)) for s, i in zip(top.values.tolist(), top.indices.tolist())]


for w in ["queen", "tokyo", "cow"]:
    print(f"nearest to {w:6s}", nearest(E[idx[w]], [w]))
for a, b, cc in [("man", "king", "woman"), ("italy", "rome", "germany")]:
    print(f"{b} - {a} + {cc} =", nearest(E[idx[b]] - E[idx[a]] + E[idx[cc]], [a, b, cc]))`;

const SGNS_TORCH_OUT = String.raw`epoch 1: 41410 pairs, mean loss 2.6887
epoch 2: 42054 pairs, mean loss 1.8178
epoch 3: 42372 pairs, mean loss 1.6482
epoch 4: 41224 pairs, mean loss 1.5728
epoch 5: 42104 pairs, mean loss 1.5371
nearest to queen  [('princess', 0.989), ('king', 0.83), ('prince', 0.814)]
nearest to tokyo  [('berlin', 0.823), ('athens', 0.796), ('paris', 0.767)]
nearest to cow    [('horse', 0.781), ('cat', 0.743), ('dog', 0.733)]
king - man + woman = [('queen', 0.979), ('princess', 0.978), ('prince', 0.757)]
rome - italy + germany = [('berlin', 0.915), ('athens', 0.73), ('tokyo', 0.727)]`;

const SGNS_KERAS = String.raw`import numpy as np
import tensorflow as tf

V, d, K = 95, 20, 5                       # vocabulary size, embedding size, negatives per pair

centre_in = tf.keras.Input(shape=(), dtype="int32")
context_in = tf.keras.Input(shape=(1 + K,), dtype="int32")   # 1 true context word + K noise words
v = tf.keras.layers.Embedding(V, d, name="v")(centre_in)       # (B, d)
u = tf.keras.layers.Embedding(V, d, name="u")(context_in)      # (B, 1+K, d)
logits = tf.einsum("bkd,bd->bk", u, v)                         # u . v for every candidate
model = tf.keras.Model([centre_in, context_in], logits)

# label 1 for the true context word, 0 for the noise words -> sigmoid cross-entropy = the SGNS loss
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# centres: (N,) ids; contexts: (N, 1+K) ids with the true word first; labels: (N, 1+K)
# model.fit([centres, contexts], labels, batch_size=256, epochs=5)
# vectors = model.get_layer("v").get_weights()[0]`;

const lesson: Lesson = {
  id: "m13/word-embeddings",
  title: "Word Embeddings: Word2Vec, GloVe, FastText",
  summary:
    "How words become dense vectors learned from co-occurrence: the distributional hypothesis, CBOW and skip-gram, the softmax bottleneck and negative sampling (gradients derived and computed by hand), subsampling and the unigram^0.75 noise distribution, GloVe, FastText's hashed character n-grams for unseen words, evaluation (and why analogy scores mislead), bias, and why contextual embeddings replaced static ones — with a real 1,714-word model trained for the simulator.",
  minutes: 90,
  level: "intermediate",
  prerequisites: ["m11/tokens-embeddings", "m03/linear-logistic-regression", "m02/norms-spaces", "m04/backpropagation"],
  next: ["m13/language-modeling", "m13/nlp-tasks", "m11/transformer-architecture"],
  terms: ["word-embedding", "distributional-hypothesis", "word2vec", "skip-gram", "cbow", "negative-sampling", "subsampling", "glove", "fasttext", "word-analogy", "pmi", "embedding"],
  sections: {
    intro: [
      { t: "p", text: "A neural network cannot read the string \"king\". It needs numbers. The simplest encoding gives every word an index and a **one-hot vector** (all zeros except one 1). But that throws away everything we know about words: \"king\" is exactly as different from \"queen\" as from \"carrot\". A **word embedding** maps each word to a short dense vector (usually 50–300 numbers) where words with related meanings sit close together." },
      { t: "p", text: "Nobody designs these vectors by hand. They are **learned from raw text** using one idea from linguistics, the **distributional hypothesis**: words that appear in similar contexts tend to have similar meanings. Train a model to predict a word's neighbours, and the vectors it needs end up encoding meaning, grammar and even some relations — the famous $\\vec{\\text{king}} - \\vec{\\text{man}} + \\vec{\\text{woman}} \\approx \\vec{\\text{queen}}$." },
      { t: "p", text: "This lesson covers the three classic methods:" },
      { t: "list", items: [
        "**Word2Vec** (Mikolov et al., 2013: \"Efficient Estimation of Word Representations in Vector Space\" and \"Distributed Representations of Words and Phrases and their Compositionality\") introduced the CBOW and skip-gram models, plus the tricks — negative sampling, subsampling — that made training on billions of words cheap.",
        "**GloVe** (Pennington, Socher and Manning, 2014) fits vectors directly to global co-occurrence counts.",
        "**FastText** (Bojanowski, Grave, Joulin and Mikolov, 2017, \"Enriching Word Vectors with Subword Information\") builds words from character n-grams, so it can embed words it has never seen.",
      ] },
      { t: "p", text: "Static word embeddings were the standard first layer of NLP systems from 2013 until contextual models (ELMo, BERT, GPT) took over around 2018. They are still useful as cheap features, and they are the clearest place to see *why* an embedding layer learns meaningful geometry — the same thing happens in every [Transformer](/learn/m11/tokens-embeddings) embedding matrix." },
      { t: "callout", kind: "key", text: "Word2Vec is logistic regression in disguise. For each (word, context-word) pair it raises the dot product of their vectors, and for a few random \"noise\" words it lowers it. Words that share contexts get pulled toward the same context vectors — and so toward each other." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Tokenization, Embeddings and Context Windows](/learn/m11/tokens-embeddings) — the embedding matrix $E \\in \\mathbb{R}^{V\\times d}$, why a lookup equals one-hot × matrix, and its sparse gradient. This lesson is about *how to train* that matrix without a downstream task; we skip the lookup mechanics.",
        "[Linear and logistic regression](/learn/m03/linear-logistic-regression) — the sigmoid $\\sigma(z)$, binary cross-entropy, and the gradient $\\sigma(z) - y$. Negative sampling is exactly this loss applied to dot products.",
        "[Norms, distances and cosine similarity](/learn/m02/norms-spaces) — $\\cos(a,b) = a\\cdot b/(\\|a\\|\\|b\\|)$, used for every neighbour and analogy query.",
        "[Backpropagation](/learn/m04/backpropagation) and [cross-entropy](/learn/m02/information-theory) — we derive gradients with the chain rule and use softmax cross-entropy.",
        "[Text Preprocessing and Subword Tokenization](/learn/m13/text-preprocessing) — what a vocabulary is and why rare words are a problem; FastText's character n-grams are a cousin of BPE.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Judging people by the company they keep", text: "At a party where you know nobody, you never learn anyone's job, but you notice who stands with whom: one group keeps gathering at the barbecue, another by the piano. By the end you could draw a map where people with shared companions stand close — and you'd have found the cooks and the musicians without being told. Word2Vec does this with words: \"a word is characterized by the company it keeps\" (J. R. Firth, 1957)." },
      { t: "p", text: "Take four sentences: \"the **king** wore a golden crown\", \"the **queen** wore a golden crown\", \"the **king** ruled the kingdom\", \"the **queen** ruled the kingdom\". \"king\" and \"queen\" never appear side by side, but they have the same neighbours. Predicting the words around each is almost the same task, so the cheapest solution is to give both almost the same vector. Meaning comes for free." },
      { t: "p", text: "Why do analogies work? Say \"king\" appears in royal plus masculine contexts (he, his), and \"queen\" in royal plus feminine ones (she, her). If a word's vector acts roughly like a sum of the context features it predicts, king − man leaves \"royal\", and adding woman puts the feminine-person part back: the result points at queen. This only works as far as the relation is a consistent direction in the space — so some relations (capital ↔ country) work much better than others (irregular past tenses), as you will measure below." },
      { t: "callout", kind: "warning", title: "Similar is not synonymous", text: "Opposites share contexts too: \"the soup was hot\" / \"the soup was cold\". So \"hot\" and \"cold\" are close neighbours. Embedding similarity measures *how alike two words are used*, not whether they mean the same." },
    ],
    mechanism: [
      { t: "p", text: "Here is how skip-gram with negative sampling (SGNS), the most popular Word2Vec variant, trains on a corpus:" },
      { t: "steps", items: [
        { title: "Build the vocabulary", text: "Count every word and keep those appearing at least `min_count` times (e.g. 5). Number them $0..V-1$. Rare words are dropped — they have too few contexts to learn from." },
        { title: "Create two tables", text: "An **input** (centre-word) table $v_w$ and an **output** (context-word) table $u_w$, both $V\\times d$. Word2Vec starts $v$ uniformly in $[-0.5/d, 0.5/d]$ and $u$ at zero. Every word has two vectors; usually only $v$ is kept at the end." },
        { title: "Subsample frequent words", text: "Before each pass, randomly delete occurrences of very frequent words (\"the\", \"of\"), more often the more frequent they are. They carry little information and would swamp the pairs." },
        { title: "Slide a window", text: "At each remaining position $t$, pick a window size $b$ uniformly from $1..c$ (a *dynamic window*: near words are used more often) and emit a pair (centre $w_t$, context $w_{t+j}$) for each $0 < |j| \\le b$." },
        { title: "Draw negatives", text: "For each pair, draw $K$ noise words (5–20 for small corpora, 2–5 for large ones) from the word-frequency distribution raised to the power 3/4." },
        { title: "Score", text: "Compute dot products $s_o = u_o\\cdot v_c$ for the true context word and $s_k = u_k\\cdot v_c$ for each noise word, and pass them through the sigmoid: $\\sigma(s_o)$ is the model's probability that the pair is real.", tex: String.raw`\mathcal{L} = -\log\sigma(u_o\cdot v_c) - \sum_{k=1}^{K}\log\sigma(-u_k\cdot v_c)` },
        { title: "Update three kinds of vectors", text: "One SGD step on $\\mathcal{L}$ moves $u_o$ toward $v_c$, moves each $u_k$ away from $v_c$, and moves $v_c$ toward $u_o$ and away from the $u_k$. Only $K+2$ rows change — the update is sparse." },
        { title: "Lower the learning rate and repeat", text: "Word2Vec lowers the learning rate linearly from 0.025 to ~0 over all passes. After a few epochs, L2-normalise the rows of $v$ and compare words with cosine similarity." },
      ] },
      { t: "p", text: "**CBOW** (continuous bag-of-words) flips the direction: it averages the input vectors of the $2c$ context words and predicts the centre word from that. One prediction per position instead of $2c$ makes it several times faster. Skip-gram's separate prediction per (centre, context) pair gives rare words more updates, so it is usually better for them." },
      { t: "callout", kind: "note", text: "Nothing tells the model what a word means; the only signal is which words appear near which. Clusters of capitals, months ordered by season, analogies — all are side effects of those predictions." },
    ],
    math: [
      { t: "h", text: "Why one-hot vectors are not enough" },
      { t: "p", text: String.raw`With vocabulary size $V$, word $i$ is $e_i \in \{0,1\}^V$ with a single 1. Any two different one-hot vectors are orthogonal, $e_i\cdot e_j = 0$, so $\cos(e_i,e_j)=0$ for every pair: no idea of similarity, a size that grows with the vocabulary, and learning about "king" teaches nothing about "queen". An embedding replaces $e_i$ by row $i$ of a learned $V\times d$ matrix with $d \ll V$, so words with similar rows share what is learned.` },
      { t: "h", text: "The skip-gram objective and the softmax bottleneck" },
      { t: "p", text: String.raw`Given a corpus $w_1,\dots,w_T$ and window $c$, skip-gram maximises the average log-probability of the context words given the centre word, using a softmax over the vocabulary:` },
      { t: "math", tex: String.raw`J(\theta) = \frac{1}{T}\sum_{t=1}^{T}\ \sum_{\substack{-c\le j\le c\\ j\neq 0}} \log p(w_{t+j}\mid w_t), \qquad p(o\mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V}\exp(u_w^\top v_c)}`, caption: "$v_c$ — input vector of the centre word; $u_o$ — output vector of the context word; $\\theta$ — all of $u$ and $v$ ($2Vd$ numbers)." },
      { t: "p", text: String.raw`Differentiating the negative log-probability of one pair (the softmax cross-entropy from [information theory](/learn/m02/information-theory)):` },
      { t: "math", tex: String.raw`-\frac{\partial \log p(o\mid c)}{\partial v_c} = -u_o + \sum_{w=1}^{V} p(w\mid c)\, u_w, \qquad -\frac{\partial \log p(o\mid c)}{\partial u_w} = \big(p(w\mid c) - \mathbb{1}[w=o]\big)\, v_c` },
      { t: "p", text: String.raw`The sum runs over the **whole vocabulary**, so every pair costs $O(Vd)$: for $V=100{,}000$ and $d=300$, 30,000,000 multiply-adds per pair, with billions of pairs per corpus. Two fixes: **hierarchical softmax** (Morin and Bengio, 2005; used by Word2Vec with a Huffman tree), which swaps the flat softmax for $\log_2 V$ yes/no decisions, and **negative sampling**, which changes the problem altogether.` },
      { t: "h", text: "CBOW" },
      { t: "math", tex: String.raw`h_t = \frac{1}{2c}\sum_{\substack{-c\le j\le c\\ j\neq 0}} v_{w_{t+j}}, \qquad p(w_t \mid \text{context}) = \frac{\exp(u_{w_t}^\top h_t)}{\sum_{w}\exp(u_w^\top h_t)}`, caption: "CBOW predicts the centre word from the mean of the context vectors; the gradient for $h_t$ is shared equally by the $2c$ context words." },
      { t: "h", text: "Negative sampling: derivation of the gradients" },
      { t: "p", text: String.raw`Negative sampling (a simpler form of noise-contrastive estimation, Gutmann and Hyvärinen, 2010) turns "which of $V$ words is the context?" into yes/no questions: "is this pair from the corpus (label 1) or from noise (label 0)?" With the logistic model $P(\text{real}\mid c,w) = \sigma(u_w^\top v_c)$ and $K$ noise words $n_1..n_K$ drawn from $P_n$, one pair's loss is binary cross-entropy:` },
      { t: "math", tex: String.raw`\mathcal{L}(c,o) = -\log\sigma(s_o) - \sum_{k=1}^{K}\log\sigma(-s_k), \qquad s_o = u_o^\top v_c,\quad s_k = u_{n_k}^\top v_c,\quad n_k\sim P_n` },
      { t: "p", text: String.raw`Two sigmoid facts do all the work: $\frac{d}{dx}\log\sigma(x) = 1-\sigma(x)$ and $\frac{d}{dx}\log\sigma(-x) = -\sigma(x)$. So $\partial\mathcal{L}/\partial s_o = \sigma(s_o)-1$ (negative: raise $s_o$) and $\partial\mathcal{L}/\partial s_k = \sigma(s_k)$ (positive: lower $s_k$). Since $\partial s_o/\partial v_c = u_o$, $\partial s_o/\partial u_o = v_c$, and the same for $s_k$, the chain rule gives:` },
      { t: "math", tex: String.raw`\frac{\partial\mathcal{L}}{\partial v_c} = \big(\sigma(s_o)-1\big)u_o + \sum_{k=1}^{K}\sigma(s_k)\,u_{n_k},\qquad \frac{\partial\mathcal{L}}{\partial u_o} = \big(\sigma(s_o)-1\big)v_c,\qquad \frac{\partial\mathcal{L}}{\partial u_{n_k}} = \sigma(s_k)\,v_c` },
      { t: "p", text: String.raw`In words: each gradient is "prediction minus label" times the other vector. Cost per pair drops from $O(Vd)$ to $O((K+1)d)$: with $K=5$, $d=300$, 1,800 multiply-adds instead of 30,000,000, about 16,700 times cheaper. The price: no normalised distribution over words — fine, since we only want the vectors. (If a noise word happens to be the true context word, word2vec's code just skips it.)` },
      { t: "h", text: "Worked example: one SGNS update by hand" },
      { t: "p", text: String.raw`$d=3$, $K=2$, learning rate $\eta=0.5$. Centre vector $v_c = (0.2, -0.1, 0.4)$, true context $u_o = (0.3, 0.2, -0.1)$, noise words $u_1 = (-0.2, 0.4, 0.1)$, $u_2 = (0.1, -0.3, 0.2)$.` },
      { t: "steps", items: [
        { title: "Scores", tex: String.raw`s_o = 0.06 - 0.02 - 0.04 = 0,\quad s_1 = -0.04 - 0.04 + 0.04 = -0.04,\quad s_2 = 0.02 + 0.03 + 0.08 = 0.13` },
        { title: "Sigmoids", tex: String.raw`\sigma(s_o) = 0.5,\quad \sigma(s_1) = 0.490001,\quad \sigma(s_2) = 0.532454` },
        { title: "Loss", tex: String.raw`\mathcal{L} = -\log 0.5 - \log\sigma(0.04) - \log\sigma(-0.13) = 0.693147 + 0.673347 + 0.760258 = 2.126753` },
        { title: "Gradient for the centre vector", tex: String.raw`\frac{\partial\mathcal{L}}{\partial v_c} = -0.5\,u_o + 0.490001\,u_1 + 0.532454\,u_2 = (-0.194755,\ -0.063736,\ 0.205491)` },
        { title: "Gradients for the output vectors", tex: String.raw`\frac{\partial\mathcal{L}}{\partial u_o} = -0.5\,v_c = (-0.1,\ 0.05,\ -0.2),\quad \frac{\partial\mathcal{L}}{\partial u_1} = (0.098000,\ -0.049000,\ 0.196001),\quad \frac{\partial\mathcal{L}}{\partial u_2} = (0.106491,\ -0.053245,\ 0.212982)` },
        { title: "Update (all gradients use the old vectors)", tex: String.raw`v_c \leftarrow (0.297377,\ -0.068132,\ 0.297255),\quad u_o \leftarrow (0.35,\ 0.175,\ 0),\quad u_1 \leftarrow (-0.249000,\ 0.424500,\ 0.002000),\quad u_2 \leftarrow (0.046755,\ -0.273377,\ 0.093509)` },
        { title: "Check", tex: String.raw`s_o: 0 \to 0.092159,\quad s_1: -0.04\to-0.102375,\quad s_2: 0.13\to 0.060326,\qquad \mathcal{L}: 2.126753 \to 2.015163` },
      ] },
      { t: "p", text: "The true pair's score rose, both noise scores fell, and the loss went down. The gradient for $u_o$ is $-0.5\\,v_c$, so the context vector moves *toward* the centre vector. That is how all centre words seen with the same context end up pointing the same way." },
      { t: "h", text: "The noise distribution and subsampling" },
      { t: "p", text: String.raw`Mikolov et al. found by experiment that drawing negatives from the word-frequency (unigram) distribution $U(w)$ raised to the power $3/4$ beats both $U$ and the uniform distribution: $P_n(w) = U(w)^{3/4}/\sum_{w'}U(w')^{3/4}$. The power flattens things: four words with counts 900, 90, 9, 1 have unigram probabilities $(0.9, 0.09, 0.009, 0.001)$ but noise probabilities $(0.8227, 0.1463, 0.0260, 0.0050)$. The rarest word is drawn five times more often than its frequency, so rare words also get the "this is not a context" signal.` },
      { t: "p", text: String.raw`**Subsampling** throws away each occurrence of word $w$ with probability $P_{\text{discard}}(w) = 1-\sqrt{t/f(w)}$ (zero if negative), where $f(w)$ is its share of all tokens and $t\approx10^{-5}$ for billion-word corpora. A word with $f=0.05$ ("the") is discarded 98.59% of the time; one with $f=0.001$, 90%; any word with $f\le t$, never. This speeds training *and* improves rarer words' vectors, because their windows now reach further. (The released C code uses a slightly different formula, keep probability $\sqrt{t/f}+t/f$, which our training script follows.)` },
      { t: "h", text: "What SGNS is really optimising" },
      { t: "p", text: String.raw`Levy and Goldberg (2014, "Neural Word Embedding as Implicit Matrix Factorization") showed that, with unconstrained vectors, SGNS is optimal when $u_o^\top v_c = \mathrm{PMI}(c,o) - \log K$. **Pointwise mutual information**, $\mathrm{PMI}(c,o) = \log\frac{P(c,o)}{P(c)P(o)}$, measures how much more often two words appear together than chance predicts. So SGNS quietly factorises a shifted PMI matrix into $UV^\top$ of rank $d$ — which explains why it behaves like older count-based methods (PMI + SVD), and links it to GloVe.` },
      { t: "h", text: "GloVe: regression on log co-occurrence counts" },
      { t: "p", text: String.raw`GloVe starts from the co-occurrence matrix $X$: $X_{ij}$ counts how often word $j$ appears in word $i$'s window (weighted by $1/\text{distance}$). Its key observation: *ratios* of co-occurrence probabilities $P(k\mid i)/P(k\mid j)$ separate relevant from irrelevant context words. For $i$ = ice, $j$ = steam, the ratio is large for $k$ = solid, small for $k$ = gas, and near 1 for $k$ = water. Asking vector differences to encode log-ratios, $(w_i - w_j)^\top\tilde w_k = \log P(k\mid i) - \log P(k\mid j)$, leads to the weighted least-squares objective` },
      { t: "math", tex: String.raw`J = \sum_{i,j:\,X_{ij}>0} f(X_{ij})\,\big(w_i^\top\tilde w_j + b_i + \tilde b_j - \log X_{ij}\big)^2, \qquad f(x) = \begin{cases}(x/x_{\max})^{\alpha} & x < x_{\max}\\ 1 & \text{otherwise}\end{cases}`, caption: "$w_i, \\tilde w_j$ — word and context vectors; $b_i, \\tilde b_j$ — single-number biases; the paper uses $x_{\\max}=100$, $\\alpha = 3/4$ and reports $w + \\tilde w$ as the final vectors." },
      { t: "p", text: String.raw`In words: make each dot product (plus biases) match the log count, trusting reliable counts more. The weight $f$ stops rare, noisy counts dominating ($f(1) = 0.0316$, $f(10) = 0.1778$, $f(50) = 0.5946$) and caps frequent pairs at 1 ($f(100)=f(500)=1$). Pairs with $X_{ij}=0$ are skipped (their log is $-\infty$). GloVe trains with AdaGrad over non-zero entries; one pass over the matrix replaces many passes over the corpus.` },
      { t: "h", text: "FastText: words as bags of character n-grams" },
      { t: "p", text: String.raw`FastText wraps each word in boundary symbols and represents it by the set $\mathcal{G}_w$ of its character n-grams ($3\le n\le6$) plus the whole word: "where" with $n=3$ gives <wh, whe, her, ere, re> plus <where>. Each n-gram $g$ has a vector $z_g$, and the skip-gram score becomes` },
      { t: "math", tex: String.raw`s(w, c) = \sum_{g\in\mathcal{G}_w} z_g^\top u_c \qquad\text{(the fastText code uses the mean over } \mathcal{G}_w\text{ as the input vector)}` },
      { t: "p", text: String.raw`There are far too many distinct n-grams to store, so each goes into one of $B$ buckets (2,000,000 by default) via the 32-bit FNV-1a hash: $h_0 = 2166136261$, $h \leftarrow (h \oplus \text{byte})\cdot 16777619 \bmod 2^{32}$, bucket $= h \bmod B$. Collisions just share a vector. The payoff: an **out-of-vocabulary** word still has n-grams, so its vector is the mean of their trained vectors. Misspellings, inflections and compounds get sensible embeddings — most useful for languages with rich word forms.` },
      { t: "h", text: "Similarity and analogy queries" },
      { t: "p", text: String.raw`Neighbours are ranked by cosine similarity. The analogy "a is to b as c is to ?" is answered by **3CosAdd**: $d^* = \arg\max_{d\notin\{a,b,c\}} \cos(v_d,\ v_b - v_a + v_c)$. With unit vectors this equals $\arg\max_d (\cos(d,b) - \cos(d,a) + \cos(d,c))$, so one large similarity can dominate; Levy and Goldberg (2014, "Linguistic Regularities in Sparse and Explicit Word Representations") proposed the multiplicative **3CosMul** to reduce that. Note that $a, b, c$ are excluded: without that, king − man + woman very often returns "king" itself.` },
    ],
    visual: [
      { t: "p", text: "The explorer loads a **real** SGNS model trained by `web/scripts/train-word2vec.py`: 1,714 words × 50 dimensions, window 5, 5 negatives, 6 epochs. Its corpus is **generated**: 60,000 sentences (379,887 tokens, 2.0 MB) built from a hand-written English lexicon and ~150 sentence templates — royalty and family words, 53 countries with capitals and nationalities, 80 verbs in four forms, 48 adjectives with comparatives and superlatives, animals, food, colours, months. Because we know the ground truth (see the callout), we can *measure* what the model recovered. The browser recomputes every neighbour and analogy from the 50-D vectors and checks itself against the Python results in the file." },
      { t: "sim", id: "word-embeddings", caption: "Tab 1: neighbours, analogies and a 2-D map of the trained vectors. Tab 2: train a 2-D skip-gram model one update at a time." },
      { t: "list", items: [
        "Press **Play** for the guided tour, or type a word. Compare the 50-D nearest neighbours (dashed lines) with the 2-D map, switching between t-SNE and PCA. 50-D neighbours are not always 2-D neighbours — the projection keeps only 15% of the variance (PCA) or only local structure (t-SNE).",
        "Try `happy`: its nearest neighbour is `sad`. Try `january`: the other winter months come first.",
        "In the analogy box try `man king woman`, `france paris japan`, `big bigger cold`. Then `walk walked swim`: it fails — irregular past tenses are the weakest relation (39%).",
        "Open *How good is it?*: overall 3CosAdd top-1 accuracy is 62.2% on 36,702 questions, while answering with the nearest neighbour of $c$ alone gets 7.8%. Look at *opposites*: the baseline (50.4%) beats the analogy method (47.1%) — the \"offset\" adds nothing there.",
        "Type a word that is not in the vocabulary (e.g. `jumping`): Word2Vec has no vector for it. FastText would build one from subwords (see the code below).",
        "Switch to **Train a toy model**. Press **Step** a few times and read the table: one centre, one context, $K$ negatives, their dot products, sigmoids, loss and gradients. The gold arrow is the real update of $v_c$ (magnified ×5). Then press **+20 epochs**: king/queen, man/woman and milk/water end up pointing the same way, though no sentence contains both words of a pair.",
        "Raise the learning rate to 0.5 and retrain: the loss curve turns jumpy and unrelated words can collapse together — with d = 2 there is little room.",
      ] },
      { t: "callout", kind: "warning", title: "Why a generated corpus?", text: "This project trains its own models and never downloads pretrained vectors. Good Word2Vec vectors need hundreds of millions of words; the ~2 MB of public-domain novels we could include would give noisy neighbours and analogy accuracy near zero. A template corpus is far more regular than real English — so treat these numbers as an upper bound for this model size, not what Word2Vec achieves on real text — but it makes the mechanism visible and measurable." },
    ],
    scratch: [
      { t: "p", text: "Skip-gram with negative sampling in NumPy, from corpus to analogies. The small generated corpus (20,000 sentences, 95 word types) has known structure: people whose gender shows only through pronouns, three social groups (royal, family, plain), six countries, four animals. The script computes subsampling and noise probabilities, gradient-checks the formulas above, trains with sparse mini-batch updates, and queries the result in about 3 seconds." },
      { t: "code", lang: "python", title: "sgns_numpy.py", code: SGNS_NUMPY, output: SGNS_NUMPY_OUT },
      { t: "list", items: [
        "`p_keep` is the paper's $\\sqrt{t/f}$: \"the\" (20% of all tokens) survives only 7% of the time. We use $t = 10^{-3}$ because the corpus is tiny; $10^{-5}$ would delete almost everything.",
        "The noise distribution lowers \"the\" from 0.204 to 0.109 and raises \"horse\" from 0.0079 to 0.0095 — the flattening effect of the 3/4 power.",
        "The gradient check confirms $\\partial\\mathcal{L}/\\partial v_c$ to $6.5\\times10^{-10}$.",
        "`np.add.at` does the sparse update: only rows used in the batch change, and a row used twice gets both gradients (plain fancy-index assignment would keep only one).",
        "The loss falls from 2.37 to 1.51 nats per pair. Neighbours match the generator's groups — but look closer: `king`'s nearest neighbour is `prince` (0.996), closer than `queen` (0.805). In this corpus the royal/family/plain context matters more than gender, which shows only through two pronouns.",
        "All three analogies are solved; `paris - france + japan = tokyo` works even though Tokyo and France never appear together.",
      ] },
      { t: "h", text: "FastText's subword hashing, and a vector for a word never seen" },
      { t: "p", text: "The second script implements character n-grams, FNV-1a hashing into buckets, and SGNS whose input vector is the mean of the word's own row and its n-gram rows. `jumping` is removed from the corpus completely; afterwards we ask for its vector anyway, and for a misspelling, `kickking`." },
      { t: "code", lang: "python", title: "fasttext_subwords.py", code: FASTTEXT_NUMPY, output: FASTTEXT_NUMPY_OUT },
      { t: "list", items: [
        "\"where\" gives exactly the 3-grams of the FastText paper's example; with $n = 3..6$ it has 14 n-grams.",
        "With only 2,000 buckets, 70 distinct n-grams of this tiny vocabulary already collide — which is why the real default is 2 million.",
        "The unseen word `jumping` gets a vector whose neighbours are other **-ing** forms (walking 0.777, kicking 0.746): its suffix n-grams (ing, ing>, ng>) carried the information. Honest caveat: its stem did *not* pull it toward jump/jumped — the -ing n-grams are shared by ten verbs and dominate.",
        "The misspelling `kickking` lands next to `kicking` (0.979). A Word2Vec model would have no vector at all for either query.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Two `nn.Embedding` tables with `sparse=True` (gradients only for looked-up rows) and `torch.optim.SparseAdam` (updates only those rows). The loss uses `F.logsigmoid`, which stays numerically stable for large scores (`torch.log(torch.sigmoid(x))` underflows to $-\\infty$ for $x \\lesssim -100$ in float32). Negatives come from `torch.multinomial` on the unigram$^{0.75}$ distribution." },
          { t: "code", lang: "python", title: "sgns_pytorch.py", code: SGNS_TORCH, output: SGNS_TORCH_OUT },
          { t: "p", text: "Different random numbers and optimiser from the NumPy version, same conclusions: groups recovered, both analogies solved (queen 0.979 just ahead of princess 0.978 — top-1 accuracy can hinge on tiny margins). For real corpora, `gensim`'s `Word2Vec` class is the standard tool; this course avoids it so every number comes from code you can read." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "In Keras, SGNS is a two-input model: the centre id and a row of $1+K$ candidate ids (true context first), scored by dot products and trained with sigmoid cross-entropy against labels $(1, 0, \\dots, 0)$. TensorFlow also offers `tf.keras.preprocessing.sequence.skipgrams` and `make_sampling_table` for building the pairs." },
          { t: "code", lang: "python", title: "sgns_keras.py (not executed in CI)", code: SGNS_KERAS, output: "Defines the model; with real (centre, candidates, labels) arrays, fit() prints the binary cross-entropy for each of 5 epochs, and the 'v' layer's weights are the word vectors." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Method", "Training signal", "Pros", "Cons", "Use cases"], rows: [
        ["One-hot / bag of words", "None (fixed)", "Very simple, easy to read", "No similarity, size $V$", "Linear baselines, sparse features"],
        ["Count + PMI + SVD (LSA-style)", "Global co-occurrence matrix, factorised", "Same result every time, closed form", "Dense $V\\times V$ matrix is big; SVD expensive", "Small corpora, analysis"],
        ["Word2Vec CBOW", "Predict centre from mean of context", "Fastest; good for frequent words", "Averaging blurs; weaker on rare words", "Large corpora, speed"],
        ["Word2Vec skip-gram (NS)", "Predict each context word from centre; negative sampling", "Strong on rare words and analogies", "Slower than CBOW; no OOV vectors", "The default static embedding"],
        ["Skip-gram + hierarchical softmax", "Binary tree over the vocabulary", "Normalised probabilities, $O(\\log V)$", "Rare words on deep tree paths", "When a proper distribution is needed"],
        ["GloVe", "Weighted least squares on $\\log X_{ij}$", "Uses global statistics; one pass over counts", "Needs the co-occurrence matrix in memory", "Pretrained general-purpose vectors"],
        ["FastText", "Skip-gram/CBOW over character n-grams", "Vectors for unseen and misspelled words; word structure", "Bigger model (buckets); n-grams can dominate", "Languages with rich word forms, noisy text"],
        ["Contextual (ELMo, BERT, GPT)", "Deep LM; one vector per token occurrence", "Handles multiple meanings, uses the whole sentence", "Expensive; needs a model at inference", "Modern NLP — see [Transformers](/learn/m11/transformer-families)"],
      ] },
      { t: "p", text: "**Static vs contextual.** Every method above gives each word *type* one vector, so \"bank\" (river) and \"bank\" (money) share one point. In our model `orange` is both a fruit and a colour: its ten nearest neighbours are fruits, then colours (red 0.749, purple 0.742). Contextual embeddings compute a vector for each *occurrence* from the whole sentence with a deep language model ([Transformer](/learn/m11/transformer-architecture) or LSTM). Their first layer is still an embedding table trained as in [Tokenization and Embeddings](/learn/m11/tokens-embeddings); the layers above add the context." },
      { t: "p", text: "Other variants worth knowing: **doc2vec / paragraph vectors** add a learned vector per document; **phrase vectors** (\"new_york\") come from merging frequent word pairs before training (Mikolov et al., 2013); **retrofitting** nudges vectors to agree with a lexicon such as WordNet; cross-lingual embeddings align two languages' spaces with a linear map." },
    ],
    applications: [
      { t: "list", items: [
        "**Input features for downstream models**: before contextual models, almost every neural NLP system (classifiers, taggers, parsers, translators) started its embedding layer from pretrained Word2Vec, GloVe or FastText vectors, often fine-tuning them — see [NLP tasks](/learn/m13/nlp-tasks).",
        "**Search and query expansion**: finding documents with related words (\"car\" → \"vehicle\") by nearest-neighbour search in embedding space.",
        "**Item embeddings in recommender systems**: the same SGNS objective run on sequences of items a user interacted with (products, songs, listings) instead of words — \"item2vec\"-style models are common in industry.",
        "**Graph embeddings**: DeepWalk and node2vec run skip-gram on random walks over a graph to embed its nodes — see [Graph Neural Networks](/learn/m15/graph-neural-networks).",
        "**Computational social science and linguistics**: tracking how word usage changes over decades, and measuring stereotypes in text (below).",
        "**Lightweight text classification**: FastText's supervised mode (averaged n-gram embeddings + a linear classifier) is a fast, strong baseline for large-scale classification.",
      ] },
      { t: "h", text: "Bias in embeddings" },
      { t: "p", text: "Vectors summarise co-occurrence, so they also absorb the stereotypes in their training text. Bolukbasi et al. (2016, \"Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings\") showed this in Word2Vec vectors trained on Google News; Caliskan, Bryson and Narayanan (2017, *Science*) measured human-like implicit associations in GloVe vectors. Models built on them can pass the bias on, e.g. ranking CVs differently because of gendered words. Fixes like removing a \"gender direction\" shrink the measurable effect but do not remove the information — it can usually still be recovered from the neighbourhoods." },
      { t: "p", text: "Our generator gives every profession he/she with probability 0.5, so there is no profession–gender link. Yet projecting profession vectors onto the normalised $v_{he}-v_{she}$ direction gives −0.026 (nurse) to 0.245 (farmer), while king and queen differ by only 0.086 (0.010 vs −0.076). Here the he–she axis is mostly noise — check that a bias direction really means what you think." },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Cheap and effective**: SGNS trains on billions of words on CPUs; a vector lookup is the cheapest possible feature extractor.",
        "**Unsupervised**: needs only raw text — no labels.",
        "**Transfer**: vectors trained on a large corpus help models trained on small labelled sets.",
        "**One vector per word type**: different meanings get averaged together (orange the fruit vs the colour; bank).",
        "**Relatedness, not similarity**: opposites and siblings (cat, dog) are close; the geometry cannot tell \"is a\", \"is the opposite of\" and \"appears with\" apart.",
        "**No vectors for unseen words** (Word2Vec, GloVe); FastText fixes this at the cost of a bigger model.",
        "**Data hunger**: quality depends heavily on corpus size and domain; vectors trained on news are poor for medicine.",
        "**Bias**: they encode the associations, stereotypes included, of their training text.",
        "**Instability**: neighbours of rare words change between runs with different seeds; do not over-read single distances.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Not excluding the query words in analogies.** $v_b - v_a + v_c$ is usually closest to $v_b$ or $v_c$ itself; reported accuracies assume the inputs are excluded. Without exclusion, our model answers king − man + woman with king.",
        "**Using Euclidean distance on unnormalised vectors.** Frequent words get long vectors, so length reflects frequency more than meaning. Normalise and use cosine.",
        "**Treating the analogy score as a measure of embedding quality.** It depends on the exclusion rule, on how often the answer is just the nearest neighbour of $c$ (Linzen, 2016, \"Issues in evaluating semantic spaces using word analogies\"), and on vocabulary size. Always report a baseline — ours gets 7.8% overall, 50.4% on opposites.",
        "**Wrong sign on the negative term**: writing $-\\log\\sigma(s_k)$ instead of $-\\log\\sigma(-s_k)$ makes the model *raise* noise scores too; every vector collapses to one direction and the loss still falls. Gradient-check new code.",
        "**Sampling negatives from raw word frequencies or uniformly.** Raw frequencies almost always pick \"the\"; uniform sampling almost never picks common words. Use $U^{3/4}$.",
        "**Not subsampling** frequent words: most pairs then involve \"the\", training is slower, and content-word vectors are worse.",
        "**Updating rows with plain fancy indexing** (`W[idx] -= g` in NumPy) when `idx` has repeats — only one of the repeated updates survives. Use `np.add.at` or `index_add_`.",
        "**Comparing vectors from two separately trained models.** Each run's space is only defined up to a rotation; align them first (e.g. orthogonal Procrustes).",
        "**Reading meaning into 2-D plots.** t-SNE and PCA distort distances; check claims in the original space.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "Why are one-hot vectors a poor representation of words?", a: "They are as long as the vocabulary, almost all zeros, and every pair of different words is orthogonal (cosine 0). So they hold no similarity information, and nothing learned about one word transfers to another. Embeddings are short dense vectors where related words are close." },
      { t: "qa", level: "beginner", q: "What is the distributional hypothesis?", a: "Words that appear in similar contexts tend to have similar meanings (Harris 1954; Firth 1957: \"you shall know a word by the company it keeps\"). Word2Vec, GloVe and FastText all turn it into an objective: words with similar context distributions get similar vectors." },
      { t: "qa", level: "beginner", q: "What is the difference between CBOW and skip-gram?", a: "CBOW predicts the centre word from the average of its context words' vectors (one prediction per position — fast). Skip-gram predicts each context word from the centre word (one prediction per pair — slower, better for rare words)." },
      { t: "qa", level: "intermediate", q: "Why is the full softmax a problem for Word2Vec, and how does negative sampling solve it?", a: "The softmax sums over all V words, so each pair costs O(Vd) — tens of millions of operations for real vocabularies. Negative sampling swaps the many-class problem for K+1 yes/no logistic problems: the true pair should score high, K noise pairs low. Cost drops to O((K+1)d). Probabilities are no longer normalised, which does not matter for learning vectors." },
      { t: "qa", level: "intermediate", q: "Derive the gradient of the SGNS loss with respect to the centre vector.", a: "$\\mathcal L = -\\log\\sigma(s_o) - \\sum_k\\log\\sigma(-s_k)$ with $s = u\\cdot v_c$. Using $d\\log\\sigma(x)/dx = 1-\\sigma(x)$ and $d\\log\\sigma(-x)/dx = -\\sigma(x)$: $\\partial\\mathcal L/\\partial v_c = (\\sigma(s_o)-1)u_o + \\sum_k \\sigma(s_k)u_k$. Each term is a logistic-regression error (prediction − label) times the other vector." },
      { t: "qa", level: "intermediate", q: "Why raise the unigram distribution to the power 3/4 for negative sampling, and why subsample frequent words?", a: "The power flattens the distribution: very frequent words are drawn less than their frequency, rare words more. In practice this beats raw unigrams (dominated by function words) and uniform sampling (ignores common words). Subsampling discards frequent tokens with probability $1-\\sqrt{t/f}$, removing uninformative pairs, speeding training and effectively widening windows for content words." },
      { t: "qa", level: "intermediate", q: "How does FastText produce a vector for a word it has never seen?", a: "A word is represented by its character n-grams (3–6, with < and > boundary markers), each hashed with FNV-1a into one of ~2M buckets with trained vectors. An unseen word's n-grams usually exist (from other words), so its vector is the mean of theirs; misspellings and inflections land near the right words." },
      { t: "qa", level: "advanced", q: "What does SGNS implicitly factorise, and how does that relate to GloVe?", a: "Levy and Goldberg (2014) showed that with enough dimensions SGNS is optimal at $u_o\\cdot v_c = \\mathrm{PMI}(c,o) - \\log K$: an implicit low-rank factorisation of the shifted PMI matrix. GloVe explicitly fits $w_i\\cdot\\tilde w_j + b_i + \\tilde b_j \\approx \\log X_{ij}$ with weighted least squares; the biases absorb the $\\log P(i)$ and $\\log P(j)$ terms. So both model (log) co-occurrence statistics, one by sampling and one from the global count matrix." },
      { t: "qa", level: "advanced", q: "What are the main criticisms of word-analogy benchmarks?", a: "(1) Inputs must be excluded, otherwise b or c usually wins, so the method is not a pure vector offset. (2) Many questions are answered by the nearest neighbour of c alone (Linzen 2016) — the offset adds little, so a baseline must be reported. (3) 3CosAdd lets one large similarity dominate (3CosMul helps). (4) The test sets cover few relation types, mostly word forms and geography, and analogy scores correlate weakly with downstream performance. In our model, opposites are solved 47.1% of the time but the c-only baseline scores 50.4%." },
      { t: "qa", level: "advanced", q: "Why were static embeddings replaced by contextual ones?", a: "A static embedding gives each word type one vector, so all senses of a word share it and the sentence cannot be used. Contextual models (ELMo 2018, BERT 2019, GPT) compute a vector per token occurrence with a deep network trained as a language model, which separates senses and captures syntax; the gains on almost every task were large. Static vectors survive where speed or interpretability matters." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "With $V = 50{,}000$, $d = 100$ and $K = 5$ negatives, how many times fewer multiply-adds does negative sampling need per training pair than the full softmax (count only the dot products)?", answer: 8333.33, tolerance: 1, explain: "Full softmax: $Vd = 5{,}000{,}000$. Negative sampling: $(K+1)d = 600$. Ratio $50{,}000/6 = 8333.3$." },
        { kind: "numeric", q: "A word has relative frequency $f = 0.01$. With threshold $t = 10^{-5}$, what is its discard probability $1-\\sqrt{t/f}$ (3 decimals)?", answer: 0.968, tolerance: 0.001, explain: "$\\sqrt{10^{-5}/10^{-2}} = \\sqrt{10^{-3}} = 0.0316$, so $1 - 0.0316 = 0.968$." },
        { kind: "numeric", q: "Two words have counts 81 and 1 (nothing else is in the vocabulary). What is the noise probability of the rare word under $U^{3/4}$ (3 decimals)?", answer: 0.036, tolerance: 0.001, explain: "$81^{0.75} = 27$, $1^{0.75} = 1$, so $1/28 = 0.0357$ — versus $1/82 = 0.0122$ under raw word frequencies." },
        { kind: "numeric", q: "In one SGNS step the true pair has score $s_o = 0$. What is $\\partial\\mathcal{L}/\\partial s_o$?", answer: -0.5, tolerance: 0.0001, explain: "$\\sigma(0) - 1 = -0.5$: raising the score lowers the loss." },
        { q: "Which statement about 3CosAdd analogy evaluation is correct?", options: ["The answer is the nearest neighbour of $b - a + c$ over the whole vocabulary including a, b, c", "The inputs a, b, c are excluded; without exclusion the answer is usually one of them", "It needs unnormalised vectors", "It is unaffected by how close d is to c"], answer: 1, explain: "Standard evaluation excludes the three query words; otherwise $b$ or $c$ is usually the closest vector." },
        { q: "Why are 'hot' and 'cold' close in a Word2Vec space?", options: ["A bug in negative sampling", "They appear in very similar contexts", "They have similar spellings", "Subsampling removed one of them"], answer: 1, explain: "Distributional similarity measures shared contexts, and opposites share most of theirs." },
        { q: "What lets FastText embed an out-of-vocabulary word?", options: ["It retrains on the new word", "It sums/averages trained vectors of the word's character n-grams", "It uses the vector of the nearest word in edit distance", "It assigns a random vector"], answer: 1, explain: "The new word's character n-grams (hashed into buckets) usually appeared in training words." },
        { q: "GloVe's weighting function $f(X_{ij})$ is mainly there to…", options: ["make the objective convex", "down-weight rare noisy co-occurrences and cap very frequent ones", "normalise the vectors", "implement negative sampling"], answer: 1, explain: "$f(x) = (x/x_{\\max})^{3/4}$ below $x_{\\max} = 100$, 1 above: rare counts get little weight, frequent ones are capped." },
      ] },
      { t: "exercise", title: "One SGNS update by hand", prompt: "With $d = 2$, $v_c = (1, 0)$, $u_o = (0, 1)$, one negative $u_1 = (1, 1)$ and $\\eta = 1$: compute the loss, the three gradients and the new vectors. Did the loss go down?", hint: "$s_o = 0$, $s_1 = 1$; $\\sigma(1) = 0.731059$.", solution: [
        { t: "steps", items: [
          { title: "Scores and loss", tex: String.raw`s_o = 0,\ s_1 = 1;\quad \mathcal{L} = -\log 0.5 - \log\sigma(-1) = 0.693147 + 1.313262 = 2.006409` },
          { title: "Gradients", tex: String.raw`\nabla_{v_c} = -0.5\,(0,1) + 0.731059\,(1,1) = (0.731059,\ 0.231059),\quad \nabla_{u_o} = -0.5\,(1,0),\quad \nabla_{u_1} = 0.731059\,(1,0)` },
          { title: "Update", tex: String.raw`v_c = (0.268941,\ -0.231059),\quad u_o = (0.5,\ 1),\quad u_1 = (0.268941,\ 1)` },
          { title: "New loss", tex: String.raw`s_o = 0.134471 - 0.231059 = -0.096588,\quad s_1 = 0.072329 - 0.231059 = -0.158730,\quad \mathcal{L} = 0.742607 + 0.616928 = 1.359535` },
        ] },
        { t: "p", text: "The loss fell from 2.006 to 1.360. But $s_o$ actually dropped a little: with $\\eta = 1$, the big push away from the negative also moved $v_c$ away from $u_o$. The total loss still improved because the negative term dominated." },
      ] },
      { t: "exercise", title: "Add 3CosMul and a c-only baseline", prompt: "Extend `sgns_numpy.py` with (a) 3CosMul, $\\arg\\max_d \\frac{\\cos'(d,b)\\cos'(d,c)}{\\cos'(d,a)+\\epsilon}$ with $\\cos' = (\\cos+1)/2$ and $\\epsilon = 10^{-3}$, and (b) the baseline that returns the nearest neighbour of $c$. Evaluate both on all country questions (capital–country pairs, 30 questions) and compare.", hint: "Use the normalised matrix E; exclude a, b, c in all three methods.", solution: [
        { t: "code", lang: "text", code: String.raw`def answer(a, b, c, method):
    sa, sb, sc = (E @ E[idx[w]] for w in (a, b, c))
    if method == "add":
        s = sb - sa + sc
    elif method == "mul":
        s = ((sb + 1) / 2) * ((sc + 1) / 2) / ((sa + 1) / 2 + 1e-3)
    else:                                   # baseline: nearest neighbour of c
        s = sc.copy()
    s[[idx[a], idx[b], idx[c]]] = -np.inf
    return vocab[int(np.argmax(s))]

pairs = [(cap, country) for country, cap, _ in COUNTRIES]
for method in ("add", "mul", "c-only"):
    hits = sum(answer(a, b, c, method) == d for (a, b) in pairs for (c, d) in pairs if a != c)
    print(method, hits, "/", len(pairs) * (len(pairs) - 1))` },
        { t: "p", text: "Added to the script, this prints `add 30 / 30`, `mul 30 / 30`, `c-only 0 / 30`. The c-only baseline fails completely because a capital's nearest neighbour is always another capital, never a country — so here every success really comes from the offset. On this clean data 3CosAdd and 3CosMul tie; they differ on noisier relations and larger vocabularies." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Embeddings map words to dense vectors; one-hot vectors are orthogonal and carry no similarity.",
        "Distributional hypothesis: similar contexts ⇒ similar meaning. Word2Vec, GloVe and FastText all learn from co-occurrence.",
        "Skip-gram: $\\max \\sum \\log p(w_{t+j}\\mid w_t)$ with $p(o\\mid c) = \\mathrm{softmax}(u^\\top v_c)_o$ — costs $O(Vd)$ per pair. CBOW predicts the centre from the mean context vector.",
        "SGNS: $\\mathcal L = -\\log\\sigma(u_o^\\top v_c) - \\sum_k \\log\\sigma(-u_k^\\top v_c)$; gradients $(\\sigma(s_o)-1)u_o + \\sum_k\\sigma(s_k)u_k$ for $v_c$, $(\\sigma(s_o)-1)v_c$ and $\\sigma(s_k)v_c$ for the outputs.",
        "Noise distribution $\\propto U(w)^{3/4}$; subsampling discards with probability $1-\\sqrt{t/f(w)}$.",
        "SGNS ≈ factorising PMI − log K. GloVe: $\\sum f(X_{ij})(w_i^\\top\\tilde w_j + b_i + \\tilde b_j - \\log X_{ij})^2$.",
        "FastText: word = bag of hashed character n-grams (3–6, FNV-1a, 2M buckets) ⇒ vectors for unseen words.",
        "Evaluate with similarity and analogies (3CosAdd, inputs excluded), always against a baseline; our real model: 62.2% vs 7.8% baseline.",
        "Embeddings inherit their corpus's biases; static vectors cannot represent multiple meanings — contextual models can.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Language Modelling and Perplexity](/learn/m13/language-modeling) — word2vec learns by predicting neighbours; a language model predicts the *next* word with real probabilities, which is how neural LMs (and their embedding layers) are trained and evaluated.",
        "[Classification, NER, QA, Summarization, Translation](/learn/m13/nlp-tasks) — embeddings as the input layer of task models.",
        "[The Transformer Architecture](/learn/m11/transformer-architecture) and [BERT, GPT, T5](/learn/m11/transformer-families) — the contextual embeddings that replaced static ones.",
        "[Self-Supervised and Contrastive Learning](/learn/m15/self-supervised) — negative sampling is an early contrastive objective; the same idea trains modern image and text encoders.",
      ] },
    ],
  },
};

export default lesson;
