import type { Lesson } from "@/lib/types";

// Every program below was executed with the project's verification interpreter; outputs are pasted verbatim.

const TAG_QA = String.raw`import re
import string
from collections import Counter

import numpy as np

# =====================================================================
# Part 1 - sequence labelling: BIO tags -> entity spans -> entity-level F1
# =====================================================================

def bio_to_spans(tags):
    """Return a set of (type, start, end_inclusive) entity spans from BIO tags.
    An I-X that does not continue an X entity starts a new one (conlleval rule)."""
    spans, start, etype = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):          # sentinel "O" closes the last entity
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and label == etype
        if start is not None and not continues:            # current entity ends before i
            spans.append((etype, start, i - 1))
            start, etype = None, None
        if prefix == "B" or (prefix == "I" and not continues):
            start, etype = i, label
    return set(spans)


def entity_prf(gold_seqs, pred_seqs):
    """Micro-averaged precision/recall/F1 over exact (type, start, end) matches."""
    tp = n_pred = n_gold = 0
    for g, p in zip(gold_seqs, pred_seqs):
        G, P = bio_to_spans(g), bio_to_spans(p)
        tp += len(G & P); n_pred += len(P); n_gold += len(G)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def token_accuracy(gold_seqs, pred_seqs):
    pairs = [(g, p) for gs, ps in zip(gold_seqs, pred_seqs) for g, p in zip(gs, ps)]
    return sum(g == p for g, p in pairs) / len(pairs)


words = "Marie Curie moved from Warsaw to Paris on 7 November 1891 .".split()
gold  = ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "O"]
pred  = ["B-PER", "O",     "O", "O", "B-LOC", "O", "B-LOC", "O", "B-DATE", "I-DATE", "O",      "O"]
all_o = ["O"] * len(gold)

print("gold spans:", sorted(bio_to_spans(gold), key=lambda s: s[1]))
print("pred spans:", sorted(bio_to_spans(pred), key=lambda s: s[1]))
for name, p in [("model", pred), ("all-O", all_o)]:
    P, R, F = entity_prf([gold], [p])
    print(f"{name:5s}: token acc {token_accuracy([gold], [p]):.3f}  entity P {P:.3f} R {R:.3f} F1 {F:.3f}")

# ---------------------------------------------------------------------
# CRF layer: score whole tag paths = emissions + transitions; decode with Viterbi
# ---------------------------------------------------------------------
TAGS = ["O", "B-PER", "I-PER"]
emissions = np.array([[1.0, 0.8, 0.9],      # token 0 "Marie": O narrowly beats B-PER on its own
                      [0.2, 0.1, 1.5],      # token 1 "Curie"
                      [2.0, 0.0, 0.3]])     # token 2 "moved"
NEG = -1e4                                  # effectively forbids a transition
start_trans = np.array([0.0, 0.0, NEG])                 # a sentence cannot start with I-PER
trans = np.array([[0.5, 0.0, NEG],          # from O: O->I-PER forbidden
                  [0.0, -0.5, 1.0],         # from B-PER: B->I is encouraged
                  [0.0, -0.5, 0.5]])        # from I-PER


def path_score(path):
    s = start_trans[path[0]] + emissions[0, path[0]]
    for t in range(1, len(path)):
        s += trans[path[t - 1], path[t]] + emissions[t, path[t]]
    return s


def viterbi(em, start, tr):
    delta = start + em[0]                                # best score of a path ending in each tag
    back = []
    for t in range(1, len(em)):
        cand = delta[:, None] + tr + em[t][None, :]     # cand[i, j]: come from tag i, go to tag j
        back.append(cand.argmax(0)); delta = cand.max(0)
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1], delta.max()


def log_partition(em, start, tr):
    """Forward algorithm: log of the sum of exp(score) over all |tags|^n paths."""
    alpha = start + em[0]
    for t in range(1, len(em)):
        alpha = np.logaddexp.reduce(alpha[:, None] + tr + em[t][None, :], axis=0)
    return np.logaddexp.reduce(alpha)


greedy = [TAGS[i] for i in emissions.argmax(1)]
best, best_sc = viterbi(emissions, start_trans, trans)
logZ = log_partition(emissions, start_trans, trans)
gold_path = [1, 2, 0]                                    # B-PER I-PER O
print("\nper-token argmax :", greedy)
print("Viterbi (CRF)    :", [TAGS[i] for i in best], f"score {best_sc:.2f}")
print(f"log Z {logZ:.4f}   CRF loss for gold path = logZ - score(gold) = {logZ - path_score(gold_path):.4f}")

# =====================================================================
# Part 2 - extractive QA: best span from start/end logits
# =====================================================================

def best_span(start_logits, end_logits, max_len):
    """argmax over i <= j < i + max_len of start_logits[i] + end_logits[j]. O(n * max_len)."""
    n = len(start_logits)
    best, best_score = (0, 0), -np.inf
    for i in range(n):
        for j in range(i, min(n, i + max_len)):
            score = start_logits[i] + end_logits[j]
            if score > best_score:
                best, best_score = (i, j), score
    return best, best_score


def best_span_vectorized(start_logits, end_logits, max_len):
    """Same search as an n x n score matrix masked to the valid upper band."""
    n = len(start_logits)
    S = start_logits[:, None] + end_logits[None, :]        # S[i, j] = s_i + e_j
    i, j = np.indices((n, n))
    S = np.where((j >= i) & (j - i < max_len), S, -np.inf)
    flat = int(np.argmax(S))
    return (flat // n, flat % n), S.max()


context = "the eiffel tower was completed in 1889 and stands in paris".split()
start_logits = np.array([0.1, 0.3, 0.2, -1.0, 0.5, 1.2, 3.1, -0.5, -1.2, 0.4, 2.0])
end_logits   = np.array([-0.8, 0.1, 0.9, -0.6, 3.3, 0.7, 2.9, -0.4, -1.0, 0.2, 2.2])

i_ind, j_ind = int(np.argmax(start_logits)), int(np.argmax(end_logits))
print(f"\nindependent argmax: start={i_ind} ({context[i_ind]}), end={j_ind} ({context[j_ind]})  -> invalid, end < start")
(i, j), sc = best_span(start_logits, end_logits, max_len=4)
(i2, j2), sc2 = best_span_vectorized(start_logits, end_logits, max_len=4)
assert (i, j) == (i2, j2)
print(f"best valid span   : ({i}, {j}) score {sc:.1f} -> '{' '.join(context[i:j + 1])}'")

# probabilities of the chosen span under the model: softmax over starts x softmax over ends
p_start = np.exp(start_logits - start_logits.max()); p_start /= p_start.sum()
p_end = np.exp(end_logits - end_logits.max()); p_end /= p_end.sum()
print(f"P(start={i}) = {p_start[i]:.4f}, P(end={j}) = {p_end[j]:.4f}, span prob = {p_start[i] * p_end[j]:.4f}")

# =====================================================================
# Part 3 - SQuAD-style exact match and token F1
# =====================================================================

def normalize_answer(s):
    """Lower-case, strip punctuation, drop articles a/an/the, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)                        # multiset intersection
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score(pred, golds):
    """SQuAD takes the max over all reference answers for each question."""
    return max(exact_match(pred, g) for g in golds), max(token_f1(pred, g) for g in golds)


examples = [
    ("in 1889", ["1889"]),
    ("The Eiffel Tower.", ["Eiffel tower"]),
    ("Paris, France", ["Paris", "in Paris"]),
    ("Gustave Eiffel's company", ["the company of Gustave Eiffel"]),
    ("London", ["Paris"]),
]
ems, f1s = [], []
for pred, golds in examples:
    em, f1 = score(pred, golds)
    ems.append(em); f1s.append(f1)
    print(f"pred {pred!r:28s} gold {golds!s:36s} EM {em:.0f}  F1 {f1:.3f}")
print(f"dataset EM {100 * np.mean(ems):.1f}  F1 {100 * np.mean(f1s):.1f}")`;

const TAG_QA_OUT = String.raw`gold spans: [('PER', 0, 1), ('LOC', 4, 4), ('LOC', 6, 6), ('DATE', 8, 10)]
pred spans: [('PER', 0, 0), ('LOC', 4, 4), ('LOC', 6, 6), ('DATE', 8, 9)]
model: token acc 0.833  entity P 0.500 R 0.500 F1 0.500
all-O: token acc 0.417  entity P 0.000 R 0.000 F1 0.000

per-token argmax : ['O', 'I-PER', 'O']
Viterbi (CRF)    : ['B-PER', 'I-PER', 'O'] score 5.30
log Z 6.0812   CRF loss for gold path = logZ - score(gold) = 0.7812

independent argmax: start=6 (1889), end=4 (completed)  -> invalid, end < start
best valid span   : (6, 6) score 6.0 -> '1889'
P(start=6) = 0.5414, P(end=6) = 0.2878, span prob = 0.1558
pred 'in 1889'                    gold ['1889']                             EM 0  F1 0.667
pred 'The Eiffel Tower.'          gold ['Eiffel tower']                     EM 1  F1 1.000
pred 'Paris, France'              gold ['Paris', 'in Paris']                EM 0  F1 0.667
pred "Gustave Eiffel's company"   gold ['the company of Gustave Eiffel']    EM 0  F1 0.571
pred 'London'                     gold ['Paris']                            EM 0  F1 0.000
dataset EM 20.0  F1 58.1`;

const BLEU_ROUGE = String.raw`import math
from collections import Counter

import numpy as np

# =====================================================================
# Part 1 - corpus BLEU (Papineni et al., 2002), implemented from the definition
# =====================================================================

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4, verbose=False):
    """hyps: list of token lists; refs: list of lists of reference token lists.
    Clipped n-gram counts and lengths are summed over the whole corpus first."""
    matched, total = np.zeros(max_n), np.zeros(max_n)
    hyp_len = ref_len = 0
    for hyp, rs in zip(hyps, refs):
        hyp_len += len(hyp)
        # effective reference length: the reference closest in length to the hypothesis
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in rs)[1]
        for n in range(1, max_n + 1):
            h = ngrams(hyp, n)
            max_ref = Counter()
            for r in rs:
                max_ref |= ngrams(r, n)                 # max count of each n-gram over references
            matched[n - 1] += sum(min(c, max_ref[g]) for g, c in h.items())   # clipping
            total[n - 1] += max(len(hyp) - n + 1, 0)
    precisions = matched / np.maximum(total, 1)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)   # brevity penalty
    if np.any(precisions == 0):                         # log(0): unsmoothed BLEU is exactly 0
        return 0.0, precisions, bp
    log_avg = np.mean(np.log(precisions))               # uniform weights 1/N
    if verbose:
        print("  matched", matched.astype(int), "total", total.astype(int))
        print(f"  hyp_len {hyp_len}  ref_len {ref_len}  BP {bp:.4f}")
    return bp * math.exp(log_avg), precisions, bp


ref = "the cat is sitting on the mat".split()
for hyp in ["the cat is sitting on the mat", "the cat sat on the mat", "the the the the the the the",
            "on the mat the cat is sitting"]:
    b, p, bp = corpus_bleu([hyp.split()], [[ref]])
    print(f"{hyp!r:34s} p1..p4 {np.round(p, 3)}  BP {bp:.3f}  BLEU {100 * b:.1f}")

print("\ncorpus of two sentences:")
hyps = ["the cat sat on the mat".split(), "a dog is running in the park".split()]
refs = [[ref], ["a dog runs in the park".split(), "the dog is running in a park".split()]]
b, p, bp = corpus_bleu(hyps, refs, verbose=True)
print(f"  precisions {np.round(p, 4)}  BLEU {100 * b:.2f}")

# =====================================================================
# Part 2 - ROUGE-1 / ROUGE-2 / ROUGE-L (Lin, 2004) as F1 scores
# =====================================================================

def rouge_n(cand, ref, n, verbose=False):
    c, r = ngrams(cand, n), ngrams(ref, n)
    overlap = sum((c & r).values())                     # clipped n-gram matches
    if verbose:
        print(f"  ROUGE-{n}: overlap {overlap}, candidate {sum(c.values())}, reference {sum(r.values())}")
    if overlap == 0:
        return 0.0
    p, rc = overlap / sum(c.values()), overlap / sum(r.values())
    return 2 * p * rc / (p + rc)


def lcs_len(a, b):
    """Longest common subsequence by dynamic programming, O(len(a) * len(b))."""
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1, j], dp[i, j - 1])
    return dp[-1, -1]


def rouge_l(cand, ref):
    l = lcs_len(cand, ref)
    if l == 0:
        return 0.0
    p, r = l / len(cand), l / len(ref)
    return 2 * p * r / (p + r)

# =====================================================================
# Part 3 - an extractive summariser (frequency scoring) vs the lead baseline
# =====================================================================

document = [
    "the city council voted on tuesday to build a new bridge across the river",
    "the old bridge was closed last year after engineers found cracks in its supports",
    "since then commuters have faced detours of up to forty minutes",
    "the new bridge will cost about twelve million dollars and should open in two years",
    "some residents argued the money should be spent on buses instead",
    "the mayor said the bridge was essential for the local economy",
]
reference = "the council approved a new bridge to replace the old bridge closed after cracks were found it will cost twelve million dollars".split()

STOP = {"the", "a", "on", "to", "in", "of", "its", "and", "was", "have", "up", "for", "be", "should",
        "will", "after", "since", "then", "some", "said", "about", "across", "last", "instead", "on"}
sents = [s.split() for s in document]
freq = Counter(w for s in sents for w in s if w not in STOP)


def sentence_score(s):
    content = [w for w in s if w not in STOP]
    return sum(freq[w] for w in content) / len(content)  # mean content-word frequency


scores = np.array([sentence_score(s) for s in sents])
print("\nsentence scores:", np.round(scores, 3))
top2 = sorted(np.argsort(-scores, kind="stable")[:2])   # keep document order
summaries = {"lead-2": sents[0] + sents[1], "freq top-2": sents[top2[0]] + sents[top2[1]]}
print("frequency summariser picks sentences", [int(k) for k in top2])
for name, cand in summaries.items():
    print(f"{name:11s} ROUGE-1 {rouge_n(cand, reference, 1):.3f}  ROUGE-2 {rouge_n(cand, reference, 2):.3f}"
          f"  ROUGE-L {rouge_l(cand, reference):.3f}  ({len(cand)} words)")
print("lead-2 details (reference has", len(reference), "words):")
rouge_n(summaries["lead-2"], reference, 1, verbose=True)
print("  LCS length", lcs_len(summaries["lead-2"], reference))`;

const BLEU_ROUGE_OUT = String.raw`'the cat is sitting on the mat'    p1..p4 [1. 1. 1. 1.]  BP 1.000  BLEU 100.0
'the cat sat on the mat'           p1..p4 [0.833 0.6   0.25  0.   ]  BP 0.846  BLEU 0.0
'the the the the the the the'      p1..p4 [0.286 0.    0.    0.   ]  BP 1.000  BLEU 0.0
'on the mat the cat is sitting'    p1..p4 [1.    0.833 0.6   0.25 ]  BP 1.000  BLEU 59.5

corpus of two sentences:
  matched [12  9  4  1] total [13 11  9  7]
  hyp_len 13  ref_len 14  BP 0.9260
  precisions [0.9231 0.8182 0.4444 0.1429]  BLEU 43.33

sentence scores: [1.5   1.375 1.    1.444 1.    1.6  ]
frequency summariser picks sentences [0, 5]
lead-2      ROUGE-1 0.520  ROUGE-2 0.167  ROUGE-L 0.440  (28 words)
freq top-2  ROUGE-1 0.340  ROUGE-2 0.089  ROUGE-L 0.298  (25 words)
lead-2 details (reference has 22 words):
  ROUGE-1: overlap 13, candidate 28, reference 22
  LCS length 11`;

const SENTIMENT = String.raw`import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

torch.set_num_threads(1)          # tiny models: one CPU thread is fastest and reproducible

# ---------------------------------------------------------------------------
# 1. A synthetic 3-class review dataset. Labels come from the template rule,
#    so they are noise-free -- far easier and less varied than real reviews.
# ---------------------------------------------------------------------------
rng = random.Random(0)
POS = ["good", "great", "excellent", "wonderful", "enjoyable", "delightful", "superb", "pleasant"]
NEG = ["bad", "awful", "terrible", "boring", "dull", "disappointing", "poor", "mediocre"]
NOUNS = ["film", "plot", "acting", "hotel", "room", "service", "food", "staff", "book", "ending",
         "camera", "battery"]
INTENS = ["", "very ", "really ", "quite ", "truly "]
FACTS = ["released in may", "about two hours long", "on the second floor", "made in france",
         "delivered on monday", "written in english", "reviewed last week", "new this year",
         "shown to the guests", "described in the brochure"]
LABELS = ["negative", "neutral", "positive"]          # class ids 0, 1, 2


def pol(adj):
    return 2 if adj in POS else 0


def flip(label):
    return 2 - label                                    # positive <-> negative


def make_example():
    n, n2 = rng.sample(NOUNS, 2)
    a = rng.choice(POS + NEG)
    b = rng.choice(NEG if a in POS else POS)            # opposite polarity to a
    t = rng.randrange(8)
    if t == 0:
        return f"the {n} was {rng.choice(INTENS)}{a}", pol(a), "plain"
    if t == 1:
        return f"the {n} was not {a}", flip(pol(a)), "not ADJ"
    if t == 2:
        return f"i did not think the {n} was {a}", flip(pol(a)), "did not think"
    if t == 3:
        return f"i expected the {n} to be {b} but it was {a}", pol(a), "expected..but"
    if t == 4:
        return f"{a} {n} , although the {n2} was {b}", pol(a), "although"
    if t == 5:
        return f"i asked whether the {n} was {rng.choice(INTENS)}{a}", 1, "question"
    if t == 6:
        return f"will the {n} be {a} ?", 1, "question"
    return f"the {n} was {rng.choice(FACTS)}", 1, "fact"


data, seen = [], set()
while len(data) < 1600:
    s, y, kind = make_example()
    if s not in seen:                                   # no duplicate sentences across splits
        seen.add(s)
        data.append((s, y, kind))
train, test = data[:600], data[600:]                   # deliberately small training set
Xtr, ytr, _ = zip(*train)
Xte, yte, kte = zip(*test)
ytr, yte = np.array(ytr), np.array(yte)
print(f"train {len(train)}  test {len(test)}  class counts (test) {np.bincount(yte)}")

results = {}

# ---------------------------------------------------------------------------
# 2. Bag-of-words logistic regression (unigrams, then unigrams + bigrams)
# ---------------------------------------------------------------------------
for name, ngram in [("BoW unigram LR", (1, 1)), ("BoW 1+2-gram LR", (1, 2))]:
    vec = CountVectorizer(ngram_range=ngram, token_pattern=r"[^ ]+")
    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(vec.fit_transform(Xtr), ytr)
    results[name] = clf.predict(vec.transform(Xte))
    print(f"{name}: {len(vec.vocabulary_)} features")

# ---------------------------------------------------------------------------
# 3. A small neural model: embeddings -> bidirectional GRU -> max-pool -> linear
# ---------------------------------------------------------------------------
torch.manual_seed(0)
vocab = {"<pad>": 0, "<unk>": 1}
for s in Xtr:
    for w in s.split():
        vocab.setdefault(w, len(vocab))


def encode(sents):
    ids = [[vocab.get(w, 1) for w in s.split()] for s in sents]
    L = max(map(len, ids))
    return torch.tensor([x + [0] * (L - len(x)) for x in ids])


class BiGRUClassifier(nn.Module):
    def __init__(self, V, d=32, h=32, C=3):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, C)

    def forward(self, x):
        H, _ = self.gru(self.emb(x))                    # (B, T, 2h)
        H = H.masked_fill((x == 0).unsqueeze(-1), -1e4)  # ignore padding in the max
        return self.out(H.max(dim=1).values)            # (B, C) logits


Xtr_t, Xte_t, ytr_t = encode(Xtr), encode(Xte), torch.tensor(ytr)
model = BiGRUClassifier(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(1, 31):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    for k in range(0, len(perm), 32):
        idx = perm[k:k + 32]
        loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte_t).argmax(1).numpy() == yte).mean()
        print(f"  BiGRU epoch {epoch:2d}  last batch loss {loss.item():.4f}  test acc {acc:.3f}")
model.eval()
with torch.no_grad():
    results["BiGRU"] = model(Xte_t).argmax(1).numpy()
print(f"BiGRU: {sum(p.numel() for p in model.parameters())} parameters")

# ---------------------------------------------------------------------------
# 4. Evaluation: accuracy, macro-F1, per-template accuracy, confusion matrices
# ---------------------------------------------------------------------------
print("\nmajority class: acc", f"{np.mean(yte == np.bincount(ytr).argmax()):.3f}")
kinds = ["plain", "not ADJ", "did not think", "expected..but", "although", "question", "fact"]
kte = np.array(kte)
print(f"{'model':16s} {'acc':>6s} {'macroF1':>8s}  " + " ".join(f"{k[:9]:>9s}" for k in kinds))
for name, pred in results.items():
    per = [np.mean(pred[kte == k] == yte[kte == k]) for k in kinds]
    print(f"{name:16s} {accuracy_score(yte, pred):6.3f} {f1_score(yte, pred, average='macro'):8.3f}  "
          + " ".join(f"{v:9.2f}" for v in per))
for name in ["BoW unigram LR", "BiGRU"]:
    print(f"\nconfusion matrix, {name} (rows = true {LABELS}, cols = predicted)")
    print(confusion_matrix(yte, results[name]))`;

const SENTIMENT_OUT = String.raw`train 600  test 1000  class counts (test) [327 312 361]
BoW unigram LR: 75 features
BoW 1+2-gram LR: 389 features
  BiGRU epoch 10  last batch loss 0.0339  test acc 0.987
  BiGRU epoch 20  last batch loss 0.0029  test acc 0.997
  BiGRU epoch 30  last batch loss 0.0010  test acc 0.999
BiGRU: 15331 parameters

majority class: acc 0.312
model               acc  macroF1      plain   not ADJ did not t expected.  although  question      fact
BoW unigram LR    0.679    0.689       0.26      0.95      0.89      0.48      0.49      1.00      1.00
BoW 1+2-gram LR   0.913    0.916       0.67      1.00      0.91      0.92      0.93      1.00      1.00
BiGRU             0.999    0.999       0.99      1.00      1.00      1.00      1.00      1.00      1.00

confusion matrix, BoW unigram LR (rows = true ['negative', 'neutral', 'positive'], cols = predicted)
[[190   0 137]
 [  0 312   0]
 [184   0 177]]

confusion matrix, BiGRU (rows = true ['negative', 'neutral', 'positive'], cols = predicted)
[[326   0   1]
 [  0 312   0]
 [  0   0 361]]`;

const NER = String.raw`import random

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)          # tiny models: one CPU thread is fastest and reproducible

# ---------------------------------------------------------------------------
# 1. Synthetic NER data. Train and test draw entities from DISJOINT name lists,
#    so every test entity word is unseen and must be recognised from context/shape.
# ---------------------------------------------------------------------------
rng = random.Random(0)
POOLS = {
    "train": dict(
        first=["Alice", "Omar", "Priya", "Lukas", "Mei", "Carlos", "Fatima", "Ivan", "Grace", "Kofi"],
        last=["Smith", "Haddad", "Rao", "Weber", "Chen", "Silva", "Okafor", "Petrov"],
        loc=[["Paris"], ["Lagos"], ["Lima"], ["Oslo"], ["New", "York"], ["Cape", "Town"], ["Rio", "de", "Janeiro"]],
        org=[["Globex"], ["Initech"], ["Acme", "Corp"], ["Bank", "of", "Avalon"], ["Northwind", "Labs"]]),
    "test": dict(
        first=["Sofia", "Tariq", "Ananya", "Henrik", "Yuki", "Diego", "Amina", "Pavel", "Nora", "Kwame"],
        last=["Jones", "Nasser", "Iyer", "Fischer", "Tanaka", "Costa", "Mensah", "Ivanov"],
        loc=[["Berlin"], ["Nairobi"], ["Quito"], ["Dublin"], ["Los", "Angeles"], ["Hong", "Kong"], ["Buenos", "Aires"]],
        org=[["Umbrella"], ["Vandelay"], ["Stark", "Industries"], ["Bank", "of", "Lumen"], ["Blue", "Sun", "Labs"]]),
}
MONTHS = ["January", "March", "May", "June", "August", "October", "December"]
DAYS = ["Monday", "Tuesday", "Friday"]


def entity(kind, P):
    if kind == "PER":
        return [rng.choice(P["first"])] + ([rng.choice(P["last"])] if rng.random() < 0.6 else [])
    if kind == "LOC":
        return list(rng.choice(P["loc"]))
    if kind == "ORG":
        return list(rng.choice(P["org"]))
    r = rng.random()                                      # DATE: 3 surface forms
    if r < 0.4:
        return [str(rng.randint(1, 28)), rng.choice(MONTHS), str(rng.randint(1990, 2024))]
    if r < 0.7:
        return [rng.choice(MONTHS), str(rng.randint(1990, 2024))]
    return [rng.choice(DAYS)]


TEMPLATES = [
    "{PER} moved to {LOC} in {DATE} .",
    "{PER} works for {ORG} in {LOC} .",
    "on {DATE} , {ORG} opened an office in {LOC} .",
    "{PER} met {PER} in {LOC} on {DATE} .",
    "the {ORG} report was published on {DATE} .",
    "{PER} may visit {LOC} next year .",                  # "may" here is a verb, not a month
    "{PER} and {PER} founded {ORG} in {DATE} .",
    "prices in {LOC} rose by 3 percent .",                # a number that is not a date
]
TEST_ONLY = [                                             # sentence shapes never seen in training
    "{LOC} hosted {PER} and {ORG} on {DATE} .",
    "according to {PER} , {ORG} {LOC} will close .",
]


def make_sentence(P, templates):
    words, tags = [], []
    for piece in rng.choice(templates).split():
        if piece.startswith("{"):
            kind = piece[1:-1]
            ent = entity(kind, P)
            words += ent
            tags += [f"B-{kind}"] + [f"I-{kind}"] * (len(ent) - 1)
        else:
            words.append(piece); tags.append("O")
    return words, tags


train = [make_sentence(POOLS["train"], TEMPLATES) for _ in range(2000)]
test = [make_sentence(POOLS["test"], TEMPLATES) for _ in range(400)]
test += [make_sentence(POOLS["test"], TEST_ONLY) for _ in range(100)]
print("example:", list(zip(*train[3])))

# ---------------------------------------------------------------------------
# 2. Vocabularies and a word-shape feature (lower / Capitalised / digits / other)
# ---------------------------------------------------------------------------
TAGS = ["O"] + [f"{p}-{k}" for k in ["PER", "LOC", "ORG", "DATE"] for p in "BI"]
tag2id = {t: i for i, t in enumerate(TAGS)}
vocab = {"<pad>": 0, "<unk>": 1}
for w, _ in train:
    for x in w:
        vocab.setdefault(x.lower(), len(vocab))


def shape(w):
    return 1 if w.islower() else 2 if w[0].isupper() else 3 if w.isdigit() else 4


def tensorize(data, word_dropout=0.0, gen=None):
    L = max(len(w) for w, _ in data)
    W = torch.zeros(len(data), L, dtype=torch.long)
    S = torch.zeros(len(data), L, dtype=torch.long)        # shape 0 = padding
    Y = torch.full((len(data), L), -100, dtype=torch.long)  # -100 = ignored by the loss
    for i, (w, t) in enumerate(data):
        for j, (x, tag) in enumerate(zip(w, t)):
            unk = word_dropout > 0 and gen.random() < word_dropout
            W[i, j] = 1 if unk else vocab.get(x.lower(), 1)
            S[i, j] = shape(x)
            Y[i, j] = tag2id[tag]
    return W, S, Y


class BiGRUTagger(nn.Module):
    def __init__(self, V, T, d=32, ds=4, h=32):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.shape_emb = nn.Embedding(5, ds, padding_idx=0)
        self.gru = nn.GRU(d + ds, h, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, T)                     # one tag distribution per token

    def forward(self, w, s):
        H, _ = self.gru(torch.cat([self.emb(w), self.shape_emb(s)], dim=-1))
        return self.out(H)                                 # (B, L, T) emission logits

# ---------------------------------------------------------------------------
# 3. Metrics from scratch: token accuracy and entity-level (exact span) F1
# ---------------------------------------------------------------------------
def bio_to_spans(tags):
    spans, start, etype = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and label == etype
        if start is not None and not continues:
            spans.append((etype, start, i - 1)); start, etype = None, None
        if prefix == "B" or (prefix == "I" and not continues):
            start, etype = i, label
    return set(spans)


def evaluate(gold, pred):
    tok = np.mean([g == p for gs, ps in zip(gold, pred) for g, p in zip(gs, ps)])
    per_type, tp_all, np_all, ng_all = {}, 0, 0, 0
    for k in ["PER", "LOC", "ORG", "DATE"]:
        tp = npred = ngold = 0
        for g, p in zip(gold, pred):
            G = {s for s in bio_to_spans(g) if s[0] == k}
            P = {s for s in bio_to_spans(p) if s[0] == k}
            tp += len(G & P); npred += len(P); ngold += len(G)
        per_type[k] = 2 * tp / (npred + ngold) if npred + ngold else 0.0   # F1 = 2TP/(|P|+|G|)
        tp_all += tp; np_all += npred; ng_all += ngold
    return tok, 2 * tp_all / (np_all + ng_all), per_type


def predict(model, data):
    W, S, _ = tensorize(data)
    model.eval()
    with torch.no_grad():
        ids = model(W, S).argmax(-1)
    return [[TAGS[i] for i in ids[n, :len(w)].tolist()] for n, (w, _) in enumerate(data)]


gold = [t for _, t in test]
tok, f1, _ = evaluate(gold, [["O"] * len(t) for t in gold])
print(f"all-O baseline         token acc {tok:.3f}  entity F1 {f1:.3f}")

# ---------------------------------------------------------------------------
# 4. Train with and without word dropout (random <unk> replacement)
# ---------------------------------------------------------------------------
for p_drop in [0.0, 0.25]:
    torch.manual_seed(0)
    gen = random.Random(1)
    model = BiGRUTagger(len(vocab), len(TAGS))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(8):
        W, S, Y = tensorize(train, word_dropout=p_drop, gen=gen)   # fresh dropout mask each epoch
        perm = torch.randperm(len(train))
        model.train()
        for k in range(0, len(train), 32):
            idx = perm[k:k + 32]
            logits = model(W[idx], S[idx])
            loss = loss_fn(logits.reshape(-1, len(TAGS)), Y[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    pred = predict(model, test)
    tok, f1, per = evaluate(gold, pred)
    print(f"BiGRU word-dropout {p_drop:.2f} token acc {tok:.3f}  entity F1 {f1:.3f}  per type "
          + "  ".join(f"{k} {v:.3f}" for k, v in per.items()))

# show some errors of the last model
shown = 0
for (w, g), p in zip(test, pred):
    if bio_to_spans(g) != bio_to_spans(p) and shown < 3:
        shown += 1
        print("  " + " ".join(f"{x}[{pt}]" if pt != gt else x for x, gt, pt in zip(w, g, p)),
              "| gold:", " ".join(f"{x}/{gt}" for x, gt in zip(w, g) if gt != "O"))`;

const NER_OUT = String.raw`example: [('Carlos', 'B-PER'), ('Weber', 'I-PER'), ('works', 'O'), ('for', 'O'), ('Northwind', 'B-ORG'), ('Labs', 'I-ORG'), ('in', 'O'), ('Oslo', 'B-LOC'), ('.', 'O')]
all-O baseline         token acc 0.503  entity F1 0.000
BiGRU word-dropout 0.00 token acc 0.795  entity F1 0.367  per type PER 0.214  LOC 0.498  ORG 0.000  DATE 0.694
BiGRU word-dropout 0.25 token acc 0.944  entity F1 0.838  per type PER 0.874  LOC 0.766  ORG 0.730  DATE 0.979
  Hong[B-PER] Kong[I-PER] hosted Amina Mensah and Blue[B-PER] Sun[I-LOC] Labs on 11 March 1993 . | gold: Hong/B-LOC Kong/I-LOC Amina/B-PER Mensah/I-PER Blue/B-ORG Sun/I-ORG Labs/I-ORG 11/B-DATE March/I-DATE 1993/I-DATE
  Hong[B-PER] Kong[I-PER] hosted Amina and Umbrella[B-LOC] on 19 December 2000 . | gold: Hong/B-LOC Kong/I-LOC Amina/B-PER Umbrella/B-ORG 19/B-DATE December/I-DATE 2000/I-DATE
  Buenos[B-PER] Aires[I-PER] hosted Yuki and Bank of Lumen on Friday . | gold: Buenos/B-LOC Aires/I-LOC Yuki/B-PER Bank/B-ORG of/I-ORG Lumen/I-ORG Friday/B-DATE`;

const TRANSLATE = String.raw`import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)          # tiny models: one CPU thread is fastest and reproducible

# ---------------------------------------------------------------------------
# 1. A toy "translation" task: English number words -> French number words, 0..999.
#    French reorders and restructures (71 = soixante et onze, 97 = quatre vingt dix sept),
#    so this is more than word-for-word substitution. Hyphens are written as spaces.
# ---------------------------------------------------------------------------
EN_U = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN_T = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
FR_U = "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize".split()
FR_T = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}


def en(n):
    h, r = divmod(n, 100)
    words = [EN_U[h], "hundred"] if h else []
    if r or not h:
        words += [EN_U[r]] if r < 20 else [EN_T[r // 10]] + ([EN_U[r % 10]] if r % 10 else [])
    return words


def fr_below_100(r):
    if r <= 16:
        return [FR_U[r]]
    if r < 20:
        return ["dix", FR_U[r - 10]]
    t, u = divmod(r, 10)
    if t in (7, 9):                                   # 70-79 = 60 + 10..19, 90-99 = 80 + 10..19
        base = ["soixante"] if t == 7 else ["quatre", "vingt"]
        if t == 7 and u == 1:
            return base + ["et", "onze"]
        return base + fr_below_100(10 + u)
    if t == 8:
        return ["quatre", "vingts"] if u == 0 else ["quatre", "vingt", FR_U[u]]
    if u == 0:
        return [FR_T[t]]
    return [FR_T[t], "et", "un"] if u == 1 else [FR_T[t], FR_U[u]]


def fr(n):
    h, r = divmod(n, 100)
    if h == 0:
        return fr_below_100(r)
    head = ["cent"] if h == 1 else [FR_U[h], "cents" if r == 0 else "cent"]
    return head + (fr_below_100(r) if r else [])


for n in [0, 17, 71, 80, 97, 200, 381]:
    print(f"{n:3d}: {' '.join(en(n)):28s} -> {' '.join(fr(n))}")

rng = random.Random(0)
numbers = list(range(1000)); rng.shuffle(numbers)
train_n, test_n = numbers[:700], numbers[700:]         # 300 numbers never seen in training

PAD, BOS, EOS = 0, 1, 2
src_vocab = {w: i + 3 for i, w in enumerate(sorted({w for n in range(1000) for w in en(n)}))}
tgt_words = sorted({w for n in range(1000) for w in fr(n)})
tgt_vocab = {w: i + 3 for i, w in enumerate(tgt_words)}
id2tgt = {i: w for w, i in tgt_vocab.items()}


def batch(ns):
    S = [[src_vocab[w] for w in en(n)] + [EOS] for n in ns]
    T = [[BOS] + [tgt_vocab[w] for w in fr(n)] + [EOS] for n in ns]
    pad = lambda X: torch.tensor([x + [PAD] * (max(map(len, X)) - len(x)) for x in X])
    return pad(S), pad(T)

# ---------------------------------------------------------------------------
# 2. A tiny encoder-decoder Transformer built on nn.Transformer
# ---------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, Vs, Vt, d=64, heads=4, layers=2, ff=128, max_len=16):
        super().__init__()
        self.src_emb, self.tgt_emb = nn.Embedding(Vs, d), nn.Embedding(Vt, d)
        self.pos = nn.Embedding(max_len, d)              # learned positional embeddings
        self.tf = nn.Transformer(d, heads, layers, layers, ff, dropout=0.1, batch_first=True)
        self.out = nn.Linear(d, Vt)

    def embed(self, emb, x):
        return emb(x) + self.pos(torch.arange(x.shape[1]))

    def forward(self, src, tgt_in):
        L = tgt_in.shape[1]                               # True = "may not attend" (future positions)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        h = self.tf(self.embed(self.src_emb, src), self.embed(self.tgt_emb, tgt_in),
                    tgt_mask=causal, src_key_padding_mask=src == PAD,
                    tgt_key_padding_mask=tgt_in == PAD, memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src, max_len=10):
        ys = torch.full((src.shape[0], 1), BOS)
        for _ in range(max_len):
            nxt = self(src, ys)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        out = []
        for row in ys[:, 1:].tolist():
            row = row[:row.index(EOS)] if EOS in row else row
            out.append([id2tgt.get(i, "?") for i in row])
        return out


torch.manual_seed(0)
model = Seq2SeqTransformer(len(src_vocab) + 3, len(tgt_vocab) + 3)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
print(f"parameters: {sum(p.numel() for p in model.parameters())}")

for epoch in range(1, 21):
    model.train()
    order = train_n[:]; rng.shuffle(order)
    for k in range(0, len(order), 32):
        src, tgt = batch(order[k:k + 32])
        logits = model(src, tgt[:, :-1])               # teacher forcing: feed gold prefix
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  train loss {loss.item():.4f}")

# ---------------------------------------------------------------------------
# 3. Evaluate on unseen numbers: exact match and corpus BLEU (from scratch)
# ---------------------------------------------------------------------------
def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    m, tot = np.zeros(max_n), np.zeros(max_n)
    hl = sum(map(len, hyps)); rl = sum(map(len, refs))
    for h, r in zip(hyps, refs):
        for n in range(1, max_n + 1):
            hc, rc = ngrams(h, n), ngrams(r, n)
            m[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            tot[n - 1] += max(len(h) - n + 1, 0)
    p = m / np.maximum(tot, 1)
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return (bp * math.exp(np.mean(np.log(p))) if p.min() > 0 else 0.0), p, bp


model.eval()
src, _ = batch(test_n)
hyps = model.greedy(src)
refs = [fr(n) for n in test_n]
em = np.mean([h == r for h, r in zip(hyps, refs)])
bleu, p, bp = corpus_bleu(hyps, refs)
print(f"test (300 unseen numbers): exact match {em:.3f}  BLEU {100 * bleu:.1f}  "
      f"p1..p4 {np.round(p, 3)}  BP {bp:.3f}")
errors = [(n, h, r) for n, h, r in zip(test_n, hyps, refs) if h != r]
for n, h, r in errors[:4]:
    print(f"  {n:3d}: got '{' '.join(h)}'  expected '{' '.join(r)}'")`;

const TRANSLATE_OUT = String.raw`  0: zero                         -> zéro
 17: seventeen                    -> dix sept
 71: seventy one                  -> soixante et onze
 80: eighty                       -> quatre vingts
 97: ninety seven                 -> quatre vingt dix sept
200: two hundred                  -> deux cents
381: three hundred eighty one     -> trois cent quatre vingt un
parameters: 174493
epoch  5  train loss 0.4448
epoch 10  train loss 0.3881
epoch 15  train loss 0.3730
epoch 20  train loss 0.3670
test (300 unseen numbers): exact match 0.987  BLEU 99.8  p1..p4 [0.996 0.996 0.998 1.   ]  BP 1.000
   80: got 'quatre vingt'  expected 'quatre vingts'
   71: got 'soixante et un'  expected 'soixante et onze'
  400: got 'quatre cent'  expected 'quatre cents'
    1: got 'cent'  expected 'un'`;

const HEADS = String.raw`import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (BertConfig, BertForQuestionAnswering, BertForSequenceClassification,
                          BertForTokenClassification, T5Config, T5ForConditionalGeneration)

torch.set_num_threads(1)          # tiny models: one CPU thread is fastest and reproducible

# ---------------------------------------------------------------------------
# 1. One tiny randomly initialised encoder, three task heads (config only, no downloads)
# ---------------------------------------------------------------------------
def cfg(num_labels=2):
    return BertConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      intermediate_size=128, max_position_embeddings=64, num_labels=num_labels)


torch.manual_seed(0)

ids = torch.randint(5, 1000, (2, 12))                   # batch of 2 sequences, 12 tokens each
mask = torch.ones_like(ids)

heads = {
    "sequence classification (3 classes)": BertForSequenceClassification(cfg(3)),
    "token classification (9 BIO tags)": BertForTokenClassification(cfg(9)),
    "extractive QA (start/end)": BertForQuestionAnswering(cfg()),
}
for name, model in heads.items():
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    total = sum(p.numel() for p in model.parameters())
    body = sum(p.numel() for p in model.bert.parameters())
    print(f"{name:38s} outputs {shapes}  head params {total - body}")

# loss for each head: classification CE, token CE with -100 ignored, start/end CE averaged
clf = heads["sequence classification (3 classes)"]
print("seq-cls loss  :", round(clf(input_ids=ids, labels=torch.tensor([2, 0])).loss.item(), 4))
tok = heads["token classification (9 BIO tags)"]
labels = torch.randint(0, 9, (2, 12)); labels[:, 0] = -100       # e.g. [CLS] position ignored
print("token loss    :", round(tok(input_ids=ids, labels=labels).loss.item(), 4))
qa = heads["extractive QA (start/end)"]
print("QA loss       :", round(qa(input_ids=ids, start_positions=torch.tensor([3, 5]),
                                  end_positions=torch.tensor([4, 9])).loss.item(), 4))

# an encoder-decoder for translation / summarisation: teacher-forced loss over target tokens
t5 = T5ForConditionalGeneration(T5Config(vocab_size=1000, d_model=64, d_kv=16, d_ff=128,
                                         num_layers=2, num_heads=4, decoder_start_token_id=0))
tgt = torch.randint(5, 1000, (2, 7))
out = t5(input_ids=ids, labels=tgt)
print(f"T5 seq2seq    : logits {tuple(out.logits.shape)}  loss {out.loss.item():.4f}  "
      f"(ln V = {torch.log(torch.tensor(1000.0)).item():.4f})")

# ---------------------------------------------------------------------------
# 2. Aligning word-level NER labels to subword tokens (first-subword labelling)
# ---------------------------------------------------------------------------
corpus = ["Marie Curie moved from Warsaw to Paris in November 1891 .",
          "the city of Paris is large .", "she moved to the city .",
          "Pierre met Marie in Paris .", "the move was in November ."] * 20
wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wp.pre_tokenizer = pre_tokenizers.Whitespace()
wp.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=60, special_tokens=["[UNK]"], show_progress=False))

words = "Marie Curie moved to Warsaw".split()
tags = ["B-PER", "I-PER", "O", "O", "B-LOC"]
enc = wp.encode(words, is_pretokenized=True)
aligned, prev = [], None
for wid in enc.word_ids:
    aligned.append("-100" if wid == prev else tags[wid])  # label only the first piece of each word
    prev = wid
for piece, wid, lab in zip(enc.tokens, enc.word_ids, aligned):
    print(f"  {piece:8s} word {wid}  label {lab}")
print(f"{len(words)} words -> {len(enc.tokens)} subword tokens")`;

const HEADS_OUT = String.raw`sequence classification (3 classes)    outputs {'logits': (2, 3)}  head params 195
token classification (9 BIO tags)      outputs {'logits': (2, 12, 9)}  head params 585
extractive QA (start/end)              outputs {'start_logits': (2, 12), 'end_logits': (2, 12)}  head params 130
seq-cls loss  : 1.0792
token loss    : 2.1738
QA loss       : 2.5442
T5 seq2seq    : logits (2, 7, 1000)  loss 7.8151  (ln V = 6.9078)
  M        word 0  label B-PER
  ##ari    word 0  label -100
  ##e      word 0  label -100
  C        word 1  label I-PER
  ##u      word 1  label -100
  ##r      word 1  label -100
  ##i      word 1  label -100
  ##e      word 1  label -100
  move     word 2  label O
  ##d      word 2  label -100
  t        word 3  label O
  ##o      word 3  label -100
  W        word 4  label B-LOC
  ##ar     word 4  label -100
  ##s      word 4  label -100
  ##a      word 4  label -100
  ##w      word 4  label -100
5 words -> 17 subword tokens`;

const KERAS_TAGGER = String.raw`import numpy as np
import tensorflow as tf
from tensorflow import keras

# Toy tensors standing in for a tokenised, padded NER corpus:
# word ids (0 = padding) and one BIO tag id per token (9 tags: O + B/I for 4 types).
rng = np.random.default_rng(0)
N, T, V, NUM_TAGS = 512, 12, 200, 9
X = rng.integers(1, V, size=(N, T))
lengths = rng.integers(5, T + 1, size=N)
for i, L in enumerate(lengths):
    X[i, L:] = 0
Y = rng.integers(0, NUM_TAGS, size=(N, T))
sample_weight = (X != 0).astype("float32")           # padding positions do not count in the loss

inputs = keras.Input(shape=(None,), dtype="int32")
h = keras.layers.Embedding(V, 32, mask_zero=True)(inputs)            # the mask propagates to the GRU
h = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True))(h)
logits = keras.layers.Dense(NUM_TAGS)(h)                              # one tag distribution per token
model = keras.Model(inputs, logits)
model.compile(optimizer=keras.optimizers.Adam(5e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(X, Y, sample_weight=sample_weight, batch_size=32, epochs=3, verbose=2)
pred = model.predict(X[:2], verbose=0).argmax(-1)
print(pred.shape)       # (2, 12): one tag id per token; decode to spans before computing entity F1`;

const lesson: Lesson = {
  id: "m13/nlp-tasks",
  title: "Classification, NER, QA, Summarization, Translation",
  summary:
    "The five everyday NLP tasks, each treated as a concrete machine-learning problem: what goes in, what comes out, which head sits on the model, which loss trains it and which metric judges it. We build BIO tagging with entity-level F1, CRF decoding, start/end span search with SQuAD exact match and F1, and BLEU and ROUGE from scratch, then run four real experiments (bag-of-words vs BiGRU sentiment, a BiGRU NER tagger, a tiny nn.Transformer translator, and the Hugging Face task heads).",
  minutes: 110,
  level: "advanced",
  prerequisites: ["m13/text-preprocessing", "m11/transformer-families", "m10/attention-intro", "m09/gru-variants", "m17/classification-metrics"],
  next: ["m13/language-modeling", "m12/transfer-learning", "m20/alignment", "m15/multimodal"],
  terms: ["text-classification", "sentiment-analysis", "named-entity-recognition", "sequence-labelling", "bio-tagging", "entity-level-f1", "crf", "viterbi", "extractive-qa", "exact-match", "bleu", "rouge", "brevity-penalty", "extractive-summarization", "abstractive-summarization"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "Most real NLP work comes down to four **task shapes**. A model either gives **one label to a whole text** (spam or not, positive or negative), gives **one label to every token** (is this word part of a name?), **points at a span** of a given text (where is the answer?), or **writes a new sequence** (a translation, a summary). Classification and sentiment, named-entity recognition (NER), extractive question answering, translation and summarisation are the classic examples, and almost every other task (part-of-speech tagging, relation extraction, code generation…) borrows one of these shapes.",
      },
      {
        t: "p",
        text: "Each task once had its own special systems: hand-built features, conditional random fields for tagging, phrase-based statistical translation. Deep learning unified them in two steps. First, recurrent encoder–decoders (Sutskever, Vinyals and Le, 2014, *Sequence to Sequence Learning with Neural Networks*; Bahdanau et al.'s attention, see [From Seq2Seq to Attention](/learn/m10/attention-intro)) let one model family read and write sequences. Second, **pretrained Transformers** ([BERT, GPT, T5](/learn/m11/transformer-families)) turned every task into *pretrained body + small task head + fine-tuning*, or, for large decoder-only models, *pretrained body + a prompt*.",
      },
      {
        t: "p",
        text: "For each task you will see what is predicted, the input and output, the **head**, the **loss**, the **metric** and why, the **data pitfalls** that quietly ruin results, and a worked example. Then we code the metrics that are easiest to get wrong and run four small real experiments.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The one idea to remember",
        text: "Pick the task shape first: **sequence → label**, **sequence → one label per token**, **sequence → span (start, end)**, or **sequence → sequence**. The shape decides the head, the loss and the metric. The encoder or decoder underneath can mostly be swapped.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Text preprocessing and subword tokenization](/learn/m13/text-preprocessing) — token IDs, special tokens like `[CLS]`/`[SEP]`, padding and attention masks. NER and QA labels belong to *words* or *characters*, so you must map them onto subword tokens and back.",
          "[BERT, GPT, T5](/learn/m11/transformer-families) — encoder-only, decoder-only and encoder–decoder models and how they are pretrained. Here we only put heads on top of them.",
          "[From Seq2Seq bottlenecks to attention](/learn/m10/attention-intro) — encoder–decoder translation, teacher forcing and greedy decoding: where neural translation began.",
          "[GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants) — the bidirectional GRU we use as a tagger, and why reading both directions is fine for tagging a finished sentence but not for generating one.",
          String.raw`[Classification losses](/learn/m05/classification-losses) — softmax and cross-entropy $-\log p_y$, which every head in this lesson uses in some form.`,
          "[Classification metrics](/learn/m17/classification-metrics) — precision, recall, F1, macro-averaging and confusion matrices. We reuse them for entities and n-grams.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "One careful reader, five different answer sheets",
        text: "A skilled reader has studied a document. You can hand them different answer sheets: a **tick-box** for one verdict on the whole text; a **highlighter** to colour every person, place or date; a **pointer** to circle the phrase that answers a question; a **blank page** to rewrite it in another language or in three sentences. The reading (the encoder) stays the same; only the sheet (the head) and the grading change. Where the analogy stops: fine-tuning also adjusts the 'reading' for each sheet.",
      },
      {
        t: "list",
        items: [
          "**Classification** is graded right or wrong.",
          "**NER:** getting 2 of the 3 words of `7 November 1891` right is still a *wrong date*, so we grade whole entities.",
          "**Extractive QA:** `in 1889` vs `1889` is nearly right, so we grade by shared tokens.",
          "**Translation and summarisation** have many correct outputs, so we count n-grams (runs of n words) shared with references — a rough stand-in for meaning, which is why these metrics are argued about.",
        ],
      },
      {
        t: "p",
        text: "The other intuition is **what the model can see**. A bag-of-words classifier sees which words occur but not their order, so `the film was not good` and `the film was good, not bad` look almost the same. A sequence model sees order and can learn that *not* flips what follows. Experiment (a) measures exactly this difference.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "p", text: String.raw`The shared skeleton: text → tokenizer → token IDs $x_1..x_n$ → an **encoder** (BiGRU, BERT-style Transformer) that gives one vector $h_i \in \mathbb{R}^d$ per token, and/or a **decoder** that writes tokens one at a time → a **task head** → a loss during training, and decoding plus a metric at evaluation.` },

      { t: "h", text: "1. Text classification and sentiment analysis" },
      {
        t: "list",
        items: [
          "**Framing.** Map a whole text to one of $C$ classes (single-label), or to a set of labels (multi-label, e.g. several toxicity types). Sentiment analysis is classification with polarity labels (negative/neutral/positive, or 1–5 stars). *Aspect-based* sentiment gives a polarity per aspect (\"food: +, service: −\").",
          "**Input/output.** Token IDs plus attention mask in; a vector of $C$ logits (raw class scores) out.",
          String.raw`**Head.** Squash the token vectors into one vector (**pooling**): the final $h_{[\mathrm{CLS}]}$ of a BERT-style encoder, a mean over non-padding tokens, or a max over time (what our BiGRU does). Then a linear layer $z = W h + b$, $W \in \mathbb{R}^{C\times d}$. A decoder-only LM uses the last token's state instead, or is prompted to write the label word.`,
          "**Loss.** Softmax cross-entropy for single-label; one sigmoid + binary cross-entropy per label for multi-label. Class weights or resampling handle imbalance.",
          "**Metric.** Accuracy for balanced classes; **macro-F1** (the average of per-class F1) otherwise, so a rare class counts as much as a common one; the **confusion matrix** shows *which* classes get mixed up.",
          "**Data pitfalls.** Imbalance (with 99% \"not spam\", 99% accuracy means nothing); near-duplicates in train and test inflate scores; annotator disagreement on neutral or mixed texts caps reachable accuracy; accidental cues (product names, review length) that correlate with labels; domain shift (movie-review sentiment does not transfer cleanly to finance).",
        ],
      },

      { t: "h", text: "2. Named-entity recognition (sequence labelling)" },
      {
        t: "list",
        items: [
          "**Framing.** Find every mention of a person, organisation, location, date… and its type. Each entity is a *span* of one or more words. We turn this into **sequence labelling**: predict one tag per token so that the tags spell out the spans.",
          "**BIO tagging.** `B-X` begins an entity of type X, `I-X` continues it, `O` is outside any entity: `Marie/B-PER Curie/I-PER moved/O to/O Paris/B-LOC`. `B` separates neighbouring same-type entities (`[Alice] [Bob]` vs `[Alice Bob]`); **IO** drops it and cannot. **BIOES** (or BILOU) adds `E-X` for the last token and `S-X` for one-token entities, giving clearer boundary targets. With $K$ types, BIO has $2K+1$ tags and BIOES $4K+1$.",
          String.raw`**Head.** A linear layer on every token vector: $z_i = W h_i + b$ with $W \in \mathbb{R}^{T\times d}$ for $T$ tags, optionally with a **CRF layer** on top (below). With subword tokenizers, usually only a word's first piece gets a label; the other pieces are ignored by the loss (label $-100$ in PyTorch).`,
          "**Loss.** Token-level cross-entropy averaged over real (non-padding, labelled) tokens. With a CRF: the negative log-likelihood of the whole gold tag sequence.",
          "**Metric.** **Entity-level (span) F1**: turn gold and predicted tags into sets of (type, start, end) spans. A prediction counts only if the type *and both boundaries* match exactly (the CoNLL-2003 shared-task rule, Tjong Kim Sang and De Meulder, 2003). **Token accuracy** misleads because most tokens are `O`: tagging everything `O` gets 50% token accuracy on our test set and 0 entity F1.",
          "**Data pitfalls.** The same names in train and test (the model memorises names instead of learning context; our experiment avoids this on purpose); inconsistent boundaries (`the United Nations` vs `United Nations`); nested entities (`[Bank of [Scotland]]`) that flat BIO cannot express; labels misaligned after subword tokenization; invalid sequences (`O I-PER`) from picking each tag separately.",
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "The CRF layer idea",
        text: String.raw`A per-token softmax picks each tag on its own, so it can output "O I-PER" (an entity that never began) or "B-PER I-LOC". A **linear-chain conditional random field** (Lafferty, McCallum and Pereira, 2001) scores the *whole* tag sequence instead: the encoder's per-token (emission) scores plus a learned transition score $A_{y_{i-1},y_i}$ for each pair of neighbouring tags. Training raises the log-probability of the gold sequence (normalised over all $T^n$ sequences with the forward algorithm). Decoding finds the best sequence with the Viterbi algorithm. BiLSTM + CRF was the standard neural NER model before pretrained Transformers (Huang, Xu and Yu, 2015; Lample et al., 2016, *Neural Architectures for Named Entity Recognition*). On strong fine-tuned encoders it adds less accuracy, but still guarantees well-formed outputs.`,
      },

      { t: "h", text: "3. Machine translation" },
      {
        t: "list",
        items: [
          String.raw`**Framing.** Conditional generation: model $p(y_1..y_m \mid x_1..x_n)$, a target-language sentence $y$ given a source sentence $x$, one word at a time from left to right: $\prod_t p(y_t \mid y_{<t}, x)$.`,
          "**Input/output.** Source token IDs go into the encoder. Target IDs, shifted right behind a start token, go into the decoder (**teacher forcing**: feeding the true previous words in training). Out comes a distribution over the target vocabulary at every position. Both languages usually share one subword vocabulary.",
          String.raw`**Model and head.** An **encoder–decoder**: a bidirectional encoder over the source, a causal decoder that cross-attends to the encoder states, and a linear **LM head** $W_o h_t \in \mathbb{R}^{V}$ (often sharing weights with the target embeddings). Decoder-only LLMs translate when prompted, treating source + target as one sequence.`,
          "**Loss.** Token-level cross-entropy on the target tokens (padding ignored), often with label smoothing.",
          "**Inference.** Greedy or beam search (see [Decoding](/learn/m20/decoding)). Now the model reads its own predictions instead of the true words.",
          "**Metric.** **BLEU** (Papineni et al., 2002): clipped n-gram precision for $n = 1..4$, geometric mean, times a brevity penalty, over the whole corpus. chrF (character n-gram F-score) and learned metrics like COMET agree better with humans; human evaluation is still the gold standard. See also [Language Modelling](/learn/m13/language-modeling).",
          "**Data pitfalls.** Noisy or misaligned sentence pairs in web-mined data; test sentences that also appear in training; BLEU from different tokenizations is **not comparable** (report the exact tool and settings); domain mismatch; a single reference punishes valid paraphrases.",
        ],
      },

      { t: "h", text: "4. Summarisation" },
      {
        t: "list",
        items: [
          "**Framing.** Write a much shorter text that keeps a document's key content (one or several documents). **Extractive** summarisation picks sentences from the source. **Abstractive** summarisation writes new text that may paraphrase, compress and merge sentences.",
          "**Extractive model.** Encode each sentence (e.g. the vector at a per-sentence marker token), score it with a linear layer + sigmoid, and train with binary cross-entropy against oracle labels (the sentence set with the highest ROUGE against the reference). Pick the top-$k$, optionally dropping repeats.",
          "**Abstractive model.** The translation setup: an encoder–decoder (BART, T5, or See, Liu and Manning's 2017 pointer-generator, which can copy source words) or a prompted LLM, trained with token cross-entropy on reference summaries.",
          "**Metric.** **ROUGE** (Lin, 2004): ROUGE-N is n-gram overlap with the reference (first defined as recall, usually reported as F1 today); ROUGE-L uses the longest common subsequence. Always add a **faithfulness** check: abstractive models can write fluent sentences the source does not support (hallucination), and ROUGE barely punishes that.",
          "**Data pitfalls.** The **lead bias** of news: the first sentences already summarise the article, so the *lead-k* baseline (take the first k sentences) is strong — it beats our naive summariser below. Also: references with facts not in the source; long inputs silently cut to the encoder's maximum length; ROUGE rewards length if you report recall only.",
        ],
      },

      { t: "h", text: "5. Question answering" },
      {
        t: "list",
        items: [
          "**Extractive QA (SQuAD format).** Given a question and a context paragraph, the answer is a span of the context. SQuAD (Rajpurkar, Zhang, Lopyrev and Liang, 2016, *SQuAD: 100,000+ Questions for Machine Comprehension of Text*) stores each example as a context, a question, and answers given as text plus a **character** start offset. SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018) adds questions with no answer.",
          "**Input/output.** `[CLS] question [SEP] context [SEP]` goes in; two scores per token come out: a start logit $s_i$ and an end logit $e_i$. Character offsets are turned into token positions with the tokenizer's offset mapping, and long contexts are cut into overlapping windows (with a *stride*).",
          String.raw`**Head.** One linear layer $\mathbb{R}^d \to \mathbb{R}^2$ on every token vector gives $(s_i, e_i)$. A softmax over positions turns each into a start or end distribution.`,
          "**Loss.** Cross-entropy of the gold start position plus cross-entropy of the gold end position, averaged. For questions with no answer, the target is the `[CLS]` position.",
          String.raw`**Decoding.** Return the span $(i, j)$ with the highest $s_i + e_j$, where $i \le j$, $j - i + 1 \le L_{\max}$ and both lie in the context (not the question). Taking the two argmaxes separately can give $j < i$.`,
          "**Metric.** **Exact match** (EM) and **token-level F1** after normalising both answers (lower-case, remove punctuation and *a/an/the*, collapse spaces), taking the best score over the reference answers, averaged over questions.",
          String.raw`**Generative QA** writes the answer with a decoder (T5, an LLM), so the answer need not be one continuous span, at the risk of hallucination. **Open-domain QA** first retrieves passages from a large collection (dense retrievers such as DPR, Karpukhin et al., 2020), then reads them. Retrieval-augmented generation (Lewis et al., 2020) feeds retrieved passages to the generator and can return them as evidence — see [Multimodal models and RAG](/learn/m15/multimodal).`,
          "**Data pitfalls.** Off-by-one character offsets; answers cut by window edges; annotators wrote questions while reading the paragraph, so questions share words with the answer and models learn word-matching shortcuts; EM punishes harmless variations, while F1 credits partly wrong spans.",
        ],
      },

      { t: "h", text: "The five tasks side by side" },
      {
        t: "table",
        head: ["Task", "Output shape", "Typical model", "Head", "Training loss", "Main metric(s)"],
        rows: [
          ["Classification / sentiment", "1 label (or label set) per text", "Fine-tuned encoder; prompted LLM", "Pool → linear $d\\to C$", "Softmax CE (sigmoid BCE if multi-label)", "Accuracy, macro-F1, confusion matrix"],
          ["NER", "1 tag per token → spans", "Fine-tuned encoder (± CRF); BiLSTM-CRF", "Per-token linear $d\\to T$ (+ CRF transitions)", "Token CE, or CRF sequence NLL", "Entity-level P/R/F1 (exact span + type)"],
          ["Extractive QA", "(start, end) in the context", "Fine-tuned encoder", "Per-token linear $d\\to 2$", "CE(start) + CE(end)", "EM and token F1 (SQuAD normalisation)"],
          ["Machine translation", "New sequence, other language", "Encoder–decoder; LLM", "LM head $d\\to V$ per step", "Token CE with teacher forcing", "BLEU, chrF, COMET, human rating"],
          ["Summarisation", "Short sequence (picked or written)", "Encoder–decoder (abstractive); sentence scorer (extractive)", "LM head; or per-sentence sigmoid", "Token CE; or sentence BCE", "ROUGE-1/2/L + faithfulness checks"],
        ],
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Classification: pooled vector → softmax → cross-entropy" },
      { t: "math", tex: String.raw`z = W h_{\text{pool}} + b,\qquad p_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}},\qquad \mathcal{L} = -\log p_{y},\qquad \frac{\partial \mathcal{L}}{\partial z} = p - \mathbf{1}_y`, caption: String.raw`$h_{\text{pool}}\in\mathbb{R}^d$ is the pooled text vector, $W\in\mathbb{R}^{C\times d}$ and $b\in\mathbb{R}^C$ the head's weights, $y$ the true class and $\mathbf{1}_y$ its one-hot vector. The gradient says: push each logit by (predicted − true) probability.` },
      {
        t: "steps",
        items: [
          { title: "Logits for (positive, neutral, negative)", tex: String.raw`z = (2.1,\ 0.3,\ -1.2)` },
          { title: "Exponentiate and normalise", tex: String.raw`e^{z} = (8.1662,\ 1.3499,\ 0.3012),\ \ \textstyle\sum = 9.8172 \Rightarrow p = (0.8318,\ 0.1375,\ 0.0307)` },
          { title: "Loss if the true label is positive", tex: String.raw`\mathcal{L} = -\ln 0.8318 = 0.1841` },
          { title: "Loss if the true label is negative", tex: String.raw`\mathcal{L} = -\ln 0.0307 = 3.4841 \quad\text{(confidently wrong is expensive)}` },
          { title: "Gradient for the logits (true = positive)", tex: String.raw`p - \mathbf{1}_y = (-0.1682,\ 0.1375,\ 0.0307)` },
        ],
      },
      { t: "p", text: String.raw`Macro-F1 averages the F1 of each class: $\text{F1}_{\text{macro}} = \frac{1}{C}\sum_c \frac{2P_cR_c}{P_c+R_c}$. So a three-class classifier that ignores a 5% class loses a third of its macro-F1, even if accuracy barely moves.` },

      { t: "h", text: "NER: token accuracy vs entity-level F1" },
      { t: "math", tex: String.raw`P = \frac{|\mathcal{S}_{\text{pred}} \cap \mathcal{S}_{\text{gold}}|}{|\mathcal{S}_{\text{pred}}|},\qquad R = \frac{|\mathcal{S}_{\text{pred}} \cap \mathcal{S}_{\text{gold}}|}{|\mathcal{S}_{\text{gold}}|},\qquad F_1 = \frac{2PR}{P+R} = \frac{2\,|\mathcal{S}_{\text{pred}} \cap \mathcal{S}_{\text{gold}}|}{|\mathcal{S}_{\text{pred}}| + |\mathcal{S}_{\text{gold}}|}`, caption: String.raw`$\mathcal{S}$ is the set of (type, start, end) spans read off the BIO tags, pooled over the whole corpus (micro-average). Precision: how many predicted entities are right. Recall: how many true entities were found.` },
      { t: "p", text: "Sentence: `Marie Curie moved from Warsaw to Paris on 7 November 1891 .` (12 tokens). Gold spans: PER(0–1), LOC(4), LOC(6), DATE(8–10). The model tags `Curie` and `1891` as `O`; everything else is right." },
      {
        t: "steps",
        items: [
          { title: "Token accuracy", tex: String.raw`\frac{10 \text{ correct tags}}{12} = 0.833` },
          { title: "Predicted spans", text: "PER(0–0), LOC(4), LOC(6), DATE(8–9). Two spans end too early, so only the two locations match exactly." },
          { title: "Entity-level scores", tex: String.raw`P = \tfrac{2}{4} = 0.5,\quad R = \tfrac{2}{4} = 0.5,\quad F_1 = 0.5` },
          { title: "All-O baseline", tex: String.raw`\text{token acc} = \tfrac{5}{12} = 0.417,\qquad F_1 = 0` },
        ],
      },
      { t: "p", text: "Two wrong tokens out of twelve cost 17 points of token accuracy but 50 points of entity F1. Each boundary error ruins a whole entity *and* adds a false positive." },

      { t: "h", text: "The CRF layer: scoring and decoding whole tag sequences" },
      { t: "math", tex: String.raw`\mathrm{score}(x, y) = \pi_{y_1} + \sum_{i=1}^{n} E_{i, y_i} + \sum_{i=2}^{n} A_{y_{i-1}, y_i},\qquad p(y \mid x) = \frac{\exp \mathrm{score}(x,y)}{Z(x)},\quad Z(x) = \sum_{y'} \exp \mathrm{score}(x,y')`, caption: String.raw`$E_{i,t}$ is the emission score of tag $t$ at token $i$ (the encoder's per-token logits), $A_{s,t}$ the learned score for moving from tag $s$ to tag $t$, $\pi_t$ the score for starting with $t$. The sum in $Z$ runs over all $T^n$ tag sequences.` },
      { t: "p", text: String.raw`Listing all $T^n$ sequences is impossible, but dynamic programming computes both $\log Z$ and the best sequence in $O(nT^2)$. The forward algorithm uses $\alpha_i(t) = \log\sum_s \exp(\alpha_{i-1}(s) + A_{s,t}) + E_{i,t}$. Viterbi swaps the log-sum-exp for a max and remembers back-pointers. The training loss is $\log Z(x) - \mathrm{score}(x, y_{\text{gold}})$.` },
      {
        t: "steps",
        items: [
          { title: "Setup (tags O, B-PER, I-PER; tokens Marie, Curie, moved)", text: "Emissions per token: Marie (1.0, 0.8, 0.9), Curie (0.2, 0.1, 1.5), moved (2.0, 0.0, 0.3). Transitions O→I-PER and start→I-PER are forbidden ($-10^4$); B-PER→I-PER = 1.0, O→O = 0.5, I-PER→I-PER = 0.5, B→B = I→B = −0.5, others 0." },
          { title: "Per-token argmax", text: "O, I-PER, O — invalid, because an entity continues without beginning." },
          { title: "Score of the gold path B-PER I-PER O", tex: String.raw`0 + 0.8 + 1.0 + 1.5 + 0 + 2.0 = 5.30` },
          { title: "Viterbi result", text: "B-PER, I-PER, O with score 5.30. The transition scores override the first token's slight preference for O." },
          { title: "CRF loss of the gold path", tex: String.raw`\log Z = 6.0812,\qquad \mathcal{L} = 6.0812 - 5.30 = 0.7812 \;\Rightarrow\; p(y_{\text{gold}} \mid x) = e^{-0.7812} = 0.458` },
        ],
      },

      { t: "h", text: "Extractive QA: span scores and SQuAD metrics" },
      { t: "math", tex: String.raw`p_{\text{start}}(i) = \frac{e^{s_i}}{\sum_k e^{s_k}},\quad p_{\text{end}}(j) = \frac{e^{e_j}}{\sum_k e^{e_k}},\quad \mathcal{L} = -\tfrac12\big(\log p_{\text{start}}(i^*) + \log p_{\text{end}}(j^*)\big),\quad (\hat\imath, \hat\jmath) = \arg\max_{i \le j < i + L_{\max}} s_i + e_j` },
      { t: "p", text: String.raw`In words: softmax over positions for the start and the end, average the two cross-entropies of the true positions, and answer with the best valid pair. Maximising $s_i + e_j$ is the same as maximising $p_{\text{start}}(i)\,p_{\text{end}}(j)$, because the two normalisers do not depend on $(i, j)$. The constrained search costs $O(n L_{\max})$.` },
      {
        t: "steps",
        items: [
          { title: "Context (11 tokens)", text: "`the eiffel tower was completed in 1889 and stands in paris`; question: *When was the Eiffel Tower completed?*" },
          { title: "Logits", text: "Start: 0.1, 0.3, 0.2, −1.0, 0.5, 1.2, **3.1**, −0.5, −1.2, 0.4, 2.0. End: −0.8, 0.1, 0.9, −0.6, **3.3**, 0.7, 2.9, −0.4, −1.0, 0.2, 2.2." },
          { title: "Independent argmax", text: "start = 6 (`1889`), end = 4 (`completed`): the end comes before the start, so this is not a span." },
          { title: "Constrained best span", tex: String.raw`(6, 6):\ 3.1 + 2.9 = 6.0 \;>\; (6,10)\!:\ 5.3\ \text{(also too long for } L_{\max}=4),\ (10,10)\!:\ 4.2,\ (5,6)\!:\ 4.1` },
          { title: "Span probability", tex: String.raw`p_{\text{start}}(6)\,p_{\text{end}}(6) = 0.5414 \times 0.2878 = 0.1558` },
          { title: "Metrics for the prediction \"in 1889\" against gold \"1889\"", tex: String.raw`\text{EM} = 0,\quad P = \tfrac{1}{2},\ R = \tfrac{1}{1},\ F_1 = \tfrac{2 \cdot 0.5 \cdot 1}{1.5} = 0.667` },
        ],
      },
      { t: "p", text: "Normalisation matters. `The Eiffel Tower.` and `Eiffel tower` are an exact match after lower-casing and removing the article and full stop. `Gustave Eiffel's company` vs `the company of Gustave Eiffel` gets only F1 0.571, because removing the apostrophe turns `Eiffel's` into `eiffels`, which matches nothing." },

      { t: "h", text: "Machine translation: BLEU" },
      { t: "math", tex: String.raw`p_n = \frac{\sum_{\text{sent}} \sum_{g \in \text{n-grams(hyp)}} \min\!\big(\mathrm{count}_{\text{hyp}}(g),\ \max_{r} \mathrm{count}_{r}(g)\big)}{\sum_{\text{sent}} \#\text{n-grams(hyp)}},\qquad \mathrm{BP} = \begin{cases}1 & c > r\\ e^{\,1 - r/c} & c \le r\end{cases},\qquad \mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\frac{1}{4}\sum_{n=1}^{4}\log p_n\Big)`, caption: "$p_n$: the share of the hypothesis's n-grams found in a reference, each counted at most as often as it appears there (clipping). $c$ = total hypothesis length, $r$ = total effective reference length (per sentence, the reference length closest to the hypothesis length). Counts are summed over the corpus before dividing." },
      { t: "p", text: "Corpus of two sentences. Hyp 1 `the cat sat on the mat` vs ref `the cat is sitting on the mat`; hyp 2 `a dog is running in the park` vs refs `a dog runs in the park` and `the dog is running in a park`." },
      {
        t: "steps",
        items: [
          { title: "Clipped matches / totals for n = 1..4", tex: String.raw`p_1 = \tfrac{12}{13} = 0.9231,\ p_2 = \tfrac{9}{11} = 0.8182,\ p_3 = \tfrac{4}{9} = 0.4444,\ p_4 = \tfrac{1}{7} = 0.1429` },
          { title: "Log precisions and their mean", tex: String.raw`\ln p_n = (-0.0800,\ -0.2007,\ -0.8109,\ -1.9459),\quad \text{mean} = -0.7594,\quad e^{-0.7594} = 0.4680` },
          { title: "Brevity penalty", tex: String.raw`c = 6 + 7 = 13,\ r = 7 + 7 = 14\ \Rightarrow\ \mathrm{BP} = e^{1 - 14/13} = 0.9260` },
          { title: "BLEU", tex: String.raw`0.9260 \times 0.4680 = 0.4333 \ \ (43.33)` },
        ],
      },
      { t: "p", text: "Two warnings from the from-scratch run. The single sentence `the cat sat on the mat` scores BLEU **0**, because it has no matching 4-gram (sentence-level BLEU needs smoothing). The scrambled `on the mat the cat is sitting` scores **59.5**: every word is right, the order is not. BLEU measures n-gram overlap, not whether the meaning is conveyed." },

      { t: "h", text: "Summarisation: ROUGE" },
      { t: "math", tex: String.raw`\text{ROUGE-N}: \ P = \frac{\text{overlap}_n}{\#\text{n-grams(cand)}},\ R = \frac{\text{overlap}_n}{\#\text{n-grams(ref)}};\qquad \text{ROUGE-L}: \ P = \frac{\mathrm{LCS}}{|\text{cand}|},\ R = \frac{\mathrm{LCS}}{|\text{ref}|};\qquad F_1 = \frac{2PR}{P+R}` },
      { t: "p", text: "ROUGE-N counts shared n-grams; ROUGE-L uses the LCS, the longest run of words that appears in both texts in the same order (gaps allowed)." },
      {
        t: "steps",
        items: [
          { title: "Lead-2 summary vs a 22-word reference", text: "The first two sentences of the bridge article (28 words) share 13 clipped unigrams with the reference; the longest common subsequence has 11 words." },
          { title: "ROUGE-1", tex: String.raw`P = \tfrac{13}{28} = 0.464,\ R = \tfrac{13}{22} = 0.591,\ F_1 = \tfrac{2\cdot 13}{28 + 22} = 0.520` },
          { title: "ROUGE-L", tex: String.raw`F_1 = \tfrac{2\cdot 11}{28 + 22} = 0.440` },
        ],
      },

      { t: "h", text: "Sequence-to-sequence training objective" },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{seq2seq}} = -\sum_{t=1}^{m} \log p_\theta\big(y_t \mid y_{<t}, x\big)`, caption: "Teacher forcing: in training, $y_{<t}$ are the true previous tokens. At test time they are the model's own outputs, so test metrics must use real decoding." },
      { t: "p", text: "Translation and abstractive summarisation use exactly this loss. They differ only in data and metric." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "p", text: "Three simulators each show one ingredient of these tasks." },
      { t: "h", text: "1. Encoder–decoder alignment (translation, summarisation)" },
      { t: "sim", id: "seq2seq-attention", caption: "A real trained GRU encoder–decoder with additive attention (string reversal). Translation and abstractive summarisation use the same encode → attend → emit loop." },
      {
        t: "list",
        items: [
          "Decode step by step and watch each output token's attention row. In translation the alignment is near-diagonal for languages with similar word order and bends where words are reordered (in our number task, `ninety seven` → `quatre vingt dix sept` spreads one source word over several target words).",
          "The decoder reads its own previous output, so one early mistake changes everything after it. That is why BLEU must be computed on decoded output, not teacher-forced predictions.",
          "Accuracy drops beyond the training lengths. Summarisers trained on short articles suffer the same way on long documents.",
        ],
      },
      { t: "h", text: "2. Bidirectional vs causal attention (choosing an encoder or a decoder)" },
      { t: "sim", id: "attention", caption: "Type a short QA-style input such as `when was it built SEP it was built in 1889` and toggle the causal mask." },
      {
        t: "list",
        items: [
          "Without the causal mask, every token attends to every other token: the context word `1889` can look at the question words. An extractive-QA encoder needs exactly this to score start and end positions.",
          "Turn the causal mask on: each token sees only tokens to its left. A decoder-only model reading `question … context` can still connect them, but a token never sees what follows — one reason small fine-tuned bidirectional encoders stay strong for tagging and span extraction.",
          "The weights are seeded random, not trained, so look at which cells are allowed to be non-zero, not at the patterns.",
        ],
      },
      { t: "h", text: "3. Subword tokens vs word-level labels (NER, QA)" },
      { t: "sim", id: "tokenizer", caption: "BPE trained live on the text in the box." },
      {
        t: "list",
        items: [
          "After a few merges, encode a rare name such as `sklodowska`: it splits into several pieces. In NER only one piece (usually the first) carries the word's tag; the others get the ignore label, and at prediction time you read the first piece's tag.",
          "For QA the gold answer comes as character offsets. You need each token's character span (the *offset mapping*) to find which tokens hold the answer's first and last characters.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: String.raw`Two NumPy programs build the parts that are more than "a linear layer + cross-entropy": decoding and evaluation. The first does BIO → spans with token accuracy and entity F1, CRF Viterbi decoding and forward-algorithm loss, the best QA span under the $i \le j$ and length limits (loop and vectorised), and SQuAD normalisation, exact match and F1. Every number in the Math section comes from this run.` },
      { t: "code", lang: "python", title: "tagging_and_qa_from_scratch.py", code: TAG_QA, output: TAG_QA_OUT },
      {
        t: "list",
        items: [
          "`bio_to_spans` follows the conlleval rule: an `I-X` that does not continue an `X` entity starts a new one. Some libraries throw such spans away instead (\"strict\" mode), which changes scores for models that output invalid sequences. Always say which rule you used.",
          "`best_span_vectorized` builds the full $n\\times n$ score matrix and keeps only the band $0 \\le j - i < L_{\\max}$. Real QA code also masks out question and padding positions.",
          "`token_f1` uses a multiset intersection (`Counter &`), so a repeated word is credited only as many times as it appears in both answers.",
        ],
      },
      { t: "p", text: "The second implements corpus BLEU (clipping, multiple references, brevity penalty), ROUGE-1/2/L, and a tiny extractive summariser that scores sentences by the average frequency of their content words, compared with lead-2 on a six-sentence article written for this lesson." },
      { t: "code", lang: "python", title: "bleu_rouge_summariser.py", code: BLEU_ROUGE, output: BLEU_ROUGE_OUT },
      {
        t: "callout",
        kind: "warning",
        title: "The naive summariser loses to lead-2",
        text: "The frequency summariser picks sentence 5 (`the mayor said the bridge was essential…`) because it repeats the common word *bridge*. It scores ROUGE-1 0.340, against 0.520 for just taking the first two sentences. This is the lead bias of news in miniature, and why every summarisation paper reports a lead baseline.",
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
              { t: "h", text: "(a) Sentiment: bag-of-words logistic regression vs a BiGRU" },
              { t: "p", text: "A three-class (negative / neutral / positive) review dataset built from eight templates: 600 training and 1,000 test sentences, with no sentence in both. **Honest caveats:** the vocabulary has 75 words, labels follow fixed rules (no label noise), and real reviews are far more varied, so read the numbers as a demo of *what each model can represent*, not a benchmark. The templates hold the constructions that beat word counting: negation (`not ADJ`), long-range negation (`i did not think the X was ADJ`), contrast (`expected ADJ1 but it was ADJ2`, `ADJ1 X, although the Y was ADJ2`) and opinion words inside non-opinions (`i asked whether the X was great`)." },
              { t: "code", lang: "python", title: "sentiment_bow_vs_bigru.py", code: SENTIMENT, output: SENTIMENT_OUT },
              {
                t: "list",
                items: [
                  "**Unigram BoW (acc 0.679, macro-F1 0.689).** Neutral is perfect: its cue words (`asked`, `whether`, `?`, factual phrases) appear nowhere else. Between negative and positive it is near a coin flip (137 negatives called positive, 184 positives called negative). A linear model over word counts cannot express \"the adjective's polarity, *flipped* if `not` is present\" — an XOR-like interaction. Since flipped and contrast templates outnumber plain ones, it even gets most plain `the X was ADJ` sentences wrong (0.26).",
                  "**Uni+bigram BoW (acc 0.913).** Bigrams like `not good` or `be great` act as 'both words together' features, so adjacent negation (1.00) and the `to be … was` contrast (0.92) become linearly separable. Plain sentences (0.67) and long-range negation (0.91) stay imperfect because the negation is not next to the adjective.",
                  "**BiGRU (15,331 parameters, acc 0.999).** Reading words in order, it represents negation scope and contrast directly: one error in 1,000. On real data the gap is smaller and the neural model needs much more data, but the direction holds.",
                ],
              },
              { t: "h", text: "(b) NER: a BiGRU BIO tagger, token accuracy vs entity F1" },
              { t: "p", text: "Template sentences with PER, LOC, ORG and DATE entities (multi-word names like `Rio de Janeiro`, `Bank of Avalon`, `12 March 2021`), plus traps: `may` as a verb, and a number that is not a date. **Train and test use separate name lists**, and 100 of the 500 test sentences use two sentence shapes never seen in training. The tagger sees lower-cased word IDs plus a word-shape feature (lower / Capitalised / digits / other). We train it twice: normally, and with **word dropout** (each training word replaced by `<unk>` with probability 0.25), which forces it to rely on context and shape — exactly what unseen names need." },
              { t: "code", lang: "python", title: "bigru_ner_tagger.py", code: NER, output: NER_OUT },
              {
                t: "list",
                items: [
                  "**Token accuracy and entity F1 tell different stories.** The all-O baseline already gets 0.503 token accuracy. Without word dropout, the tagger reaches 0.795 token accuracy but only **0.367** entity F1 (ORG 0.000): every test name is `<unk>`, which it never saw in training. With word dropout: 0.944 token accuracy vs **0.838** entity F1 — 5.6% of tokens wrong, but 16% of the entity score lost.",
                  "**Errors cluster where the data shifted.** All printed errors come from the unseen shape `LOC hosted PER and ORG on DATE`. In training, a capitalised sentence-initial name was always a person, so `Hong Kong` becomes `B-PER I-PER`, and `Blue Sun Labs` gets the invalid sequence `B-PER I-LOC O` — the kind of output a CRF would forbid.",
                  "Real NER datasets (CoNLL-2003 has PER, ORG, LOC, MISC) are messier. Use this to understand the metric and failure modes, not to estimate real-world F1.",
                ],
              },
              { t: "h", text: "(d) A tiny nn.Transformer translator with BLEU" },
              { t: "p", text: "Toy translation: English number words → French number words for 0–999, a fixed language pair with real structure. French rebuilds 70–99 (`seventy one` → `soixante et onze`, `ninety seven` → `quatre vingt dix sept`) and has agreement quirks (`quatre vingts` but `quatre vingt un`; `deux cents` but `deux cent un`). 700 numbers are for training; the other 300 are **never seen** in training, so good test scores need compositional generalisation, not memorisation. The model is `nn.Transformer` with 2 encoder and 2 decoder layers, $d=64$, learned position embeddings and greedy decoding. BLEU is our from-scratch corpus BLEU (one reference)." },
              { t: "code", lang: "python", title: "number_translation_transformer.py", code: TRANSLATE, output: TRANSLATE_OUT },
              { t: "p", text: "The model has 174,493 parameters. On the 300 held-out numbers greedy decoding gets **98.7% exact match** and corpus **BLEU 99.8**. Training loss levels off near 0.37, not 0, mainly because label smoothing (0.05) makes zero loss impossible." },
              {
                t: "list",
                items: [
                  "Three of the four errors are rare irregular forms: `quatre vingts` (80) and `quatre cents` (400) take a plural -s only when nothing follows, and 71 is `soixante et onze`. Each pattern covers only about ten of the 1,000 numbers. The fourth error (1 translated as `cent`) is a plain mistake on the shortest input.",
                  "BLEU is forgiving: 1.3% of translations are wrong, yet BLEU is 99.8, because a missing -s costs one token in roughly a thousand. Real translation has many valid outputs per sentence, so even very good systems score far lower against a single reference, and BLEU differences of a point or less are hard to interpret.",
                ],
              },
              { t: "h", text: "Task heads on a Transformer encoder (Hugging Face, configuration only)" },
              { t: "p", text: "The `transformers` library ships these heads. Built from small configurations with **random weights** (nothing is downloaded), they show output shapes, head sizes and losses. The second part trains a 60-token WordPiece vocabulary on a few sentences and aligns word-level BIO labels to subword tokens." },
              { t: "code", lang: "python", title: "task_heads_and_label_alignment.py", code: HEADS, output: HEADS_OUT },
              { t: "p", text: String.raw`The heads are tiny: 195 parameters for three classes ($64\times3+3$; BERT's tanh pooler counts as part of the body), 585 for nine BIO tags ($64\times9+9$) and 130 for QA ($64\times2+2$). Almost all the capacity is in the shared encoder, which is why one pretrained body can serve every task.` },
              { t: "p", text: String.raw`With random weights each loss is near the uniform-guess value: $\ln 3 = 1.0986$ vs 1.0792, $\ln 9 = 2.1972$ vs 2.1738, and $\ln 12 = 2.4849$ vs 2.5442 for QA (softmax over 12 positions). The untrained T5 scores 7.8151, above $\ln 1000 = 6.9078$, because its random logits are not uniform. In the alignment only each word's first piece keeps the tag; the rest get −100. With a 60-token vocabulary even common words break apart (5 words become 17 tokens); with real 30k–50k vocabularies mostly rare words and names are split.` },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "A Keras BiGRU tagger with the same structure as experiment (b). `mask_zero=True` makes the embedding produce a padding mask that the bidirectional GRU respects, and `sample_weight` removes padding positions from the loss. The random tensors only stand in for a real tokenised corpus." },
              { t: "code", lang: "python", title: "keras_bigru_tagger.py (not executed in CI)", code: KERAS_TAGGER, output: "Prints three lines of training progress (one per epoch, loss for random labels stays near ln 9 ≈ 2.2) and then (2, 12) — one predicted tag id per token for two sentences." },
              { t: "p", text: "For pretrained Transformer heads in TensorFlow, `transformers` provides `TFAutoModelForSequenceClassification`, `TFAutoModelForTokenClassification` and `TFAutoModelForQuestionAnswering`, with the same heads as the PyTorch classes above." },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "h", text: "Model families for the same tasks" },
      {
        t: "table",
        head: ["Approach", "How it works", "Advantages", "Disadvantages", "Use it when"],
        rows: [
          ["Bag-of-words / TF-IDF + linear model", "Word (n-gram) counts → logistic regression or linear SVM", "Trains in seconds, readable weights, strong baseline for topic classification", "No word order beyond n-grams; fails on negation scope and contrast (acc 0.679 vs 0.999 in experiment a)", "Baseline; topic/spam filtering at huge scale; very little compute"],
          ["CNN / BiLSTM / BiGRU on embeddings", "Learned or pretrained word vectors → convolution or recurrence → head", "Sees word order, small, fast on CPU", "Needs labelled data per task; weaker than pretrained Transformers", "Edge devices, streaming, small specialised tasks"],
          ["BiLSTM-CRF tagger", "Recurrent emissions + CRF transitions, Viterbi decoding", "Well-formed tag sequences, models how labels depend on each other", "Sequential training; fine-tuned encoders are more accurate", "NER/POS with modest data, structured-output constraints"],
          ["Fine-tuned encoder (BERT family)", "Pretrained bidirectional encoder + task head, all weights fine-tuned", "State of the art for classification, NER, extractive QA at small cost; one forward pass", "Cannot generate text; needs labelled data (hundreds to thousands of examples); fixed max length", "Production classification/extraction with a fixed label set"],
          ["Fine-tuned encoder–decoder (T5, BART)", "Pretrained seq2seq model fine-tuned on input → output pairs", "Natural for translation and summarisation; any task as text-to-text", "Token-by-token decoding is costly; can hallucinate", "Translation, summarisation, generative QA with paired data"],
          ["Prompted decoder-only LLM", "Instruction-tuned LLM with zero/few-shot prompts, optionally with retrieval", "No task-specific training data; handles open-ended and multi-step tasks", "Expensive per call, slower, output must be parsed, less calibrated, harder to audit", "Few labels, many changing tasks, generation-heavy work"],
        ],
      },
      { t: "h", text: "Tagging schemes" },
      {
        t: "table",
        head: ["Scheme", "Tags for K types", "Example for `New York City`", "Notes"],
        rows: [
          ["IO", "K + 1", "I-LOC I-LOC I-LOC", "Cannot separate neighbouring same-type entities"],
          ["BIO (IOB2)", "2K + 1", "B-LOC I-LOC I-LOC", "Standard; only the start is marked explicitly"],
          ["BIOES / BILOU", "4K + 1", "B-LOC I-LOC E-LOC (single word: S-LOC)", "Explicit end and single-word tags: direct supervision for where entities end"],
          ["Span-based", "—", "score every (start, end, type) candidate", "Handles nested entities; $O(n^2)$ candidates, usually capped by length"],
        ],
      },
      { t: "h", text: "Task variants" },
      {
        t: "list",
        items: [
          "**Classification:** binary, multi-class, multi-label (one sigmoid per label), hierarchical, aspect-based sentiment, natural-language inference (classify a premise–hypothesis *pair*: `[CLS] premise [SEP] hypothesis [SEP]`).",
          "**Sequence labelling:** NER, part-of-speech tagging, chunking, slot filling in dialogue systems, token-level error detection.",
          "**QA:** extractive (span), multiple-choice (score each option with a classification head), generative (decoder), open-domain retrieve-then-read, retrieval-augmented generation with citations, detecting unanswerable questions (SQuAD 2.0's no-answer option).",
          "**Translation:** bilingual vs multilingual models (one model, many language pairs, the target language given by a tag token), document-level translation, speech translation.",
          "**Summarisation:** extractive vs abstractive, single- vs multi-document, query-focused, long-document (needs long-context encoders or chunking).",
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Classification:** spam and phishing filters, ticket routing and intent detection, content moderation (toxicity, hate speech), sentiment tracking in reviews and social media, sorting legal and medical documents.",
          "**NER and information extraction:** de-identifying clinical notes (names, dates, record numbers), pulling parties, amounts and dates from contracts and invoices, linking company mentions in financial news, building knowledge graphs.",
          "**Extractive and retrieval-augmented QA:** search engines that highlight the answer, company document search, support assistants that answer from a help-centre and cite their passages.",
          "**Machine translation:** translation apps, localising software and documentation, cross-language search, live subtitles.",
          "**Summarisation:** meeting and call summaries, news digests, overviews of long reports, emails or patient records — usually with a human checking faithfulness when stakes are high.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths and limits of each framing" },
      {
        t: "list",
        items: [
          "**Classification** is the cheapest, most reliable framing, with well-understood metrics. But one label loses nuance (mixed reviews, sarcasm), and label definitions drift over time.",
          "**Sequence labelling** is fast (one pass, one decision per token), and exact-span scoring is strict and objective. But flat BIO cannot show nested or broken-up entities, and quality depends on consistent boundary annotation.",
          "**Extractive QA** cannot invent an answer that is not in the text, and it points to its evidence. But it fails when the answer must be combined, counted, or is not one continuous span.",
          "**Generation (translation, summarisation, generative QA)** is the most flexible. But decoding is slower, outputs can be fluent and wrong, and BLEU and ROUGE are weak stand-ins for quality: our scrambled sentence scored BLEU 59.5.",
        ],
      },
      { t: "h", text: "How to choose a model for an NLP task today" },
      {
        t: "table",
        head: ["Criterion", "Fine-tuned encoder (BERT-style)", "Prompted decoder-only LLM", "Fine-tuned encoder–decoder / small LLM"],
        rows: [
          ["Labelled data needed", "Hundreds to thousands of examples", "None to a few examples in the prompt", "Thousands of input–output pairs"],
          ["Cost and latency per input", "Lowest: one forward pass of a ~100M-parameter model", "Highest: large model, token-by-token generation", "Medium"],
          ["Output control", "Exactly the labels / spans you defined", "Free text: must be parsed and checked; may invent labels", "Text, but shaped by fine-tuning"],
          ["Accuracy on a fixed, well-labelled task", "Usually very strong", "Often competitive zero-shot, sometimes below a fine-tuned small model", "Strong for generation tasks"],
          ["Changing requirements / many tasks", "Retrain for each change", "Edit the prompt", "Retrain"],
          ["Privacy, auditing, on-device", "Easy to host and audit", "Hard with a hosted API; large to self-host", "Moderate"],
        ],
      },
      {
        t: "list",
        items: [
          "**Start from the task shape and the volume.** Fixed labels or spans, high volume, tight latency or cost → fine-tune an encoder. Open-ended outputs, few labels, changing tasks → prompt an instruction-tuned LLM (see [Instruction tuning, RLHF and DPO](/learn/m20/alignment)).",
          "**A common, effective pattern:** use an LLM to label a few thousand examples or prototype the task, check those labels, then fine-tune a small encoder for production. Keep the LLM for the hard leftovers.",
          "**Measure, do not assume.** Build a held-out test set with the *right* metric (entity F1, not token accuracy; EM/F1; ROUGE plus a faithfulness check) and compare a fine-tuned small model, a prompted LLM and a trivial baseline (majority class, all-O, lead-k) on it.",
          "**Parameter-efficient fine-tuning** (LoRA and relatives, see [Scaling and PEFT](/learn/m19/scaling-efficiency)) narrows the gap: you can adapt a mid-sized LLM with modest compute when neither prompting nor a small encoder is enough.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Reporting token accuracy for NER.** Most tokens are `O`: our all-O tagger scores 0.503, and a tagger with 0.944 token accuracy has only 0.838 entity F1. Fix: report entity-level precision, recall and F1 (micro over entities, plus per type).",
          "**Giving every subword piece the word's tag, then scoring on pieces.** A split `Sklodowska` is counted several times, and B/I tags become inconsistent (`B-PER B-PER B-PER`). Fix: label only the first piece (−100 for the rest) and evaluate on words.",
          String.raw`**Taking separate argmaxes for the QA start and end.** You can get end < start or a huge span. Fix: search $i \le j < i + L_{\max}$ over context tokens only, maximising $s_i + e_j$.`,
          "**Comparing BLEU from different tokenizers or tools.** Subword tokens, another tokenizer or smoothing can shift BLEU by several points. Fix: score detokenized output with one standard implementation and report its settings.",
          "**Evaluating seq2seq models with teacher forcing.** Accuracy with the true previous tokens is far higher than with the model's own. Fix: decode (greedy or beam) and score the decoded text.",
          "**Trying only bag-of-words and deciding the task is hard.** Negation and contrast need word order; try n-grams or a sequence model (experiment a: 0.679 → 0.913 → 0.999).",
          "**Leaking entities or near-duplicates across splits.** Shared names, templates or reposts reward memorising. Fix: split by source, document or time, and test on unseen entities, as our NER experiment does.",
          "**Optimising only ROUGE for abstractive summaries.** A model can raise ROUGE while inventing facts. Fix: add faithfulness checks (human review, entailment- or QA-based consistency checks) and always report a lead-k baseline.",
          "**Cutting long inputs silently.** Inputs over the maximum length lose the answer or key content. Fix: sliding windows with a stride for QA, chunking or long-context models for summarisation, and track how often cutting happens.",
          "**Using accuracy on imbalanced classes.** With 95% negatives, a constant classifier scores 95%. Fix: macro-F1, per-class recall, a confusion matrix, and a threshold chosen on validation data.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What are the four basic output shapes of NLP tasks, with an example of each?", a: "One label per text (sentiment, spam, intent classification); one label per token (NER, part-of-speech tagging); a span of the input given by start and end positions (extractive QA); and a new output sequence (translation, abstractive summarisation, generative QA). The shape decides the head, the loss and the metric." },
      { t: "qa", level: "beginner", q: "What do the B, I and O tags mean in BIO tagging, and why is B needed?", a: "B-X marks the first token of an entity of type X, I-X a token inside that entity, O a token outside any entity. Without B you cannot tell two neighbouring entities of the same type (`[Alice] [Bob]`) from one two-word entity (`[Alice Bob]`). B marks where a new entity starts." },
      { t: "qa", level: "beginner", q: "What is the difference between extractive and abstractive summarisation?", a: "Extractive summarisation picks sentences (or phrases) from the source and joins them, so it keeps the source's wording but can be choppy and repetitive. Abstractive summarisation writes new text with a decoder, so it can paraphrase and compress, but it can also state things the source does not support." },
      { t: "qa", level: "beginner", q: "Why is accuracy alone a poor metric for many text-classification problems?", a: "Real label distributions are often imbalanced. A classifier that always predicts the majority class can have very high accuracy while never finding the class you care about. Macro-F1, per-class precision and recall, and the confusion matrix show how each class is handled." },
      { t: "qa", level: "intermediate", q: "Why can a NER model have high token accuracy but low entity F1?", a: "Most tokens are O, so predicting O is usually right. Entity F1 needs the exact span and type: one wrong boundary token makes the predicted entity a false positive and leaves the true entity as a false negative. In our worked example 10 of 12 tokens were right (0.833) but only 2 of 4 entities (F1 0.5)." },
      { t: "qa", level: "intermediate", q: "How does an extractive QA model produce an answer, and how is it trained?", a: String.raw`The question and context are encoded together, and a linear layer gives each token a start logit and an end logit. Training minimises the average cross-entropy of the true start and end positions (softmax over positions). At inference, pick the span with the highest $s_i + e_j$ where $i \le j$, within a maximum length, and with both positions in the context. Map token positions back to characters with the offset mapping.` },
      { t: "qa", level: "intermediate", q: "Explain the components of BLEU.", a: "Clipped n-gram precisions $p_1..p_4$ (each hypothesis n-gram counts at most as often as it appears in any one reference), combined by a geometric mean, and multiplied by a brevity penalty $e^{1-r/c}$ when the hypothesis is shorter than the reference — otherwise precision alone would reward very short outputs. Counts are summed over the corpus before dividing. BLEU measures surface overlap, not meaning, and scores are not comparable across tokenizations." },
      { t: "qa", level: "intermediate", q: "What does a CRF layer add on top of a BiLSTM or BERT tagger?", a: "Learned transition scores between neighbouring tags, so the model scores whole tag sequences instead of picking each tag alone. Training maximises the log-likelihood of the true sequence, normalised over all sequences with the forward algorithm; decoding uses Viterbi. It enforces valid sequences (no I-PER after O) and captures how labels depend on each other. With strong pretrained encoders the accuracy gain is usually small." },
      { t: "qa", level: "advanced", q: "You must classify support tickets into 30 categories, 2 million tickets per day, with 5,000 labelled examples. Fine-tune an encoder or prompt an LLM?", a: "Fine-tune a small encoder. The labels are fixed, there is enough labelled data, and at 2M a day cost and latency per ticket dominate: one forward pass of a ~100M-parameter model is orders of magnitude cheaper than LLM generation, and it always outputs a valid label. An LLM helps to bootstrap labels, handle an 'other/unclear' tail, or prototype new categories. Check on a held-out set with macro-F1 and per-class recall before deciding." },
      { t: "qa", level: "advanced", q: "Your abstractive summariser improved ROUGE-L by 2 points but users complain about wrong facts. What is happening and what do you do?", a: "ROUGE counts n-gram/LCS overlap with a reference. It barely punishes a swapped name or number and can reward copying common phrasing, so the model may be hallucinating unsupported content. Add faithfulness evaluation (human fact checks, entailment- or QA-based consistency metrics), study the errors, consider more extractive or copy-friendly models, constrain decoding, ground generation in retrieved source passages, and keep a lead-k baseline." },
      { t: "qa", level: "advanced", q: "How would you handle contexts longer than the encoder's maximum length in extractive QA, and what can go wrong?", a: "Split the context into overlapping windows (e.g. 384 tokens with a 128-token stride), each starting with the question. Score candidate spans in every window and take the best overall, mapping back to characters with the offset mapping. Pitfalls: answers cut by a window edge (the overlap reduces this); raw logits are not calibrated across windows; windows without the answer need a training label (point them at [CLS]); off-by-one character offsets." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which head is used for extractive question answering on a BERT-style encoder?", options: ["A linear layer on the [CLS] vector producing one logit per answer", "A linear layer on every token producing a start and an end logit", "A decoder that generates the answer token by token", "A CRF over BIO tags"], answer: 1, explain: String.raw`Each token gets $(s_i, e_i)$ from a $d\to2$ linear layer; the answer is the best valid (start, end) pair.` },
          { kind: "numeric", q: "Gold entities: 5. The tagger predicts 4 entities, of which 3 match exactly. What is the entity-level F1? (3 decimals)", answer: 0.667, tolerance: 0.001, explain: String.raw`$P = 3/4$, $R = 3/5$, $F_1 = 2\cdot3/(4+5) = 0.667$.` },
          { kind: "numeric", q: "SQuAD F1 of the prediction \"the treaty of Paris\" against the gold answer \"Treaty of Paris 1783\"? (3 decimals)", answer: 0.857, tolerance: 0.001, explain: "After normalising: prediction `treaty of paris` (3 tokens, the article is removed), gold `treaty of paris 1783` (4 tokens). Overlap 3: $P = 1$, $R = 0.75$, $F_1 = 2\\cdot0.75/1.75 = 0.857$." },
          { kind: "numeric", q: "A hypothesis corpus has total length 18 and effective reference length 20. What is the BLEU brevity penalty? (3 decimals)", answer: 0.895, tolerance: 0.001, explain: String.raw`$\mathrm{BP} = e^{1 - 20/18} = e^{-0.1111} = 0.895$.` },
          { q: "Start logits peak at token 7 and end logits peak at token 3. What should the decoder return?", options: ["The span 3–7", "The span 7–3, reversed", "The highest-scoring (i, j) with i ≤ j within the length limit", "No answer"], answer: 2, explain: "The two separate argmaxes do not form a valid span; search all valid pairs for the highest $s_i + e_j$." },
          { q: "Why did bigram features improve the bag-of-words sentiment model from 0.679 to 0.913 accuracy in experiment (a)?", options: ["Bigrams add more training data", "A bigram like `not good` is a conjunction feature, so negation adjacent to the adjective becomes linearly separable", "Bigrams remove stop words", "Bigrams make logistic regression non-linear"], answer: 1, explain: "Unigram counts cannot express 'polarity flipped by negation'; the bigram is itself the 'both words together' feature. Long-range negation still fails because the words are not next to each other." },
          { kind: "numeric", q: "How many tags does BIOES use for 4 entity types?", answer: 17, tolerance: 0, explain: "$4K + 1 = 17$: B, I, E, S for each type, plus O." },
          { q: "Which statement about ROUGE is correct?", options: ["It measures factual consistency with the source", "ROUGE-L uses the longest common subsequence between candidate and reference", "It is computed against the source document, not a reference", "Higher ROUGE always means a better summary"], answer: 1, explain: "ROUGE compares with reference summaries by n-gram overlap (ROUGE-N) or LCS (ROUGE-L). It does not check faithfulness." },
        ],
      },
      {
        t: "exercise",
        title: "Score a QA prediction and a tagging output by hand",
        prompt: "(1) Context tokens 0–5 have start logits (0.2, 2.5, 0.1, 1.9, −0.3, 0.4) and end logits (0.0, 0.3, 2.2, −0.5, 2.6, 0.1). With $L_{\\max} = 3$, find the best span and its score. What would the separate argmaxes give? (2) Gold tags `B-ORG I-ORG I-ORG O B-LOC`, predicted `B-ORG I-ORG O O B-LOC`. Compute token accuracy and entity F1.",
        hint: String.raw`For (1), list $s_i + e_j$ for each start with $j \in \{i, i+1, i+2\}$.`,
        solution: [
          {
            t: "steps",
            items: [
              { title: "Separate argmaxes", text: String.raw`start = 1 (2.5), end = 4 (2.6): span 1–4 has length 4 > $L_{\max}$ = 3, so it is not allowed.` },
              { title: "Candidates", tex: String.raw`(1,2): 2.5 + 2.2 = 4.7,\quad (3,4): 1.9 + 2.6 = 4.5,\quad (1,3): 2.0,\quad (2,4): 2.7,\quad (4,4): 2.3` },
              { title: "Best span", text: "(1, 2) with score 4.7." },
              { title: "Tagging", tex: String.raw`\text{token acc} = 4/5 = 0.8;\ \text{gold} = \{\mathrm{ORG}(0\text{–}2), \mathrm{LOC}(4)\},\ \text{pred} = \{\mathrm{ORG}(0\text{–}1), \mathrm{LOC}(4)\}\Rightarrow P = R = F_1 = 0.5` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Convert BIO to BIOES and evaluate both",
        prompt: "Write `bio_to_bioes(tags)`, which converts a valid BIO sequence to BIOES (last token of a multi-token entity → `E-X`, one-token entity → `S-X`), and `bioes_to_spans(tags)`. On the Marie Curie example from the scratch code, check that `bioes_to_spans(bio_to_bioes(gold)) == bio_to_spans(gold)`.",
        hint: "A tag ends its entity if the next tag is not `I-` of the same type.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "bioes.py",
            code: String.raw`def bio_to_bioes(tags):
    out = []
    for i, tag in enumerate(tags):
        if tag == "O":
            out.append(tag); continue
        prefix, label = tag.split("-", 1)
        nxt = tags[i + 1] if i + 1 < len(tags) else "O"
        ends = nxt != f"I-{label}"                     # the entity does not continue
        if prefix == "B":
            out.append(("S-" if ends else "B-") + label)
        else:
            out.append(("E-" if ends else "I-") + label)
    return out


def bioes_to_spans(tags):
    spans, start = set(), None
    for i, tag in enumerate(tags):
        prefix, _, label = tag.partition("-")
        if prefix == "S":
            spans.add((label, i, i))
        elif prefix == "B":
            start = i
        elif prefix == "E" and start is not None:
            spans.add((label, start, i)); start = None
    return spans


gold = ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "O"]
print(bio_to_bioes(gold))
print(sorted(bioes_to_spans(bio_to_bioes(gold)), key=lambda s: s[1]))`,
            output: String.raw`['B-PER', 'E-PER', 'O', 'O', 'S-LOC', 'O', 'S-LOC', 'O', 'B-DATE', 'I-DATE', 'E-DATE', 'O']
[('PER', 0, 1), ('LOC', 4, 4), ('LOC', 6, 6), ('DATE', 8, 10)]`,
          },
          { t: "p", text: "The spans match those printed by `bio_to_spans` in the scratch program. BIOES stores the same information with explicit end and single-word tags, so entity-level F1 does not change. What changes is the *training target*: the model gets direct supervision for where entities end." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Four task shapes: text → label (classification, sentiment), token → tag (NER), (start, end) span (extractive QA), sequence → sequence (translation, summarisation, generative QA).",
          String.raw`Classification: pooled vector → $W h + b$ → softmax cross-entropy. Report macro-F1 and a confusion matrix when classes are imbalanced.`,
          String.raw`NER: BIO/BIOES tags and a per-token linear head (optionally a CRF scoring $\sum E_{i,y_i} + \sum A_{y_{i-1},y_i}$, decoded with Viterbi). Evaluate with exact-span **entity-level F1**, never token accuracy alone.`,
          String.raw`Extractive QA: start/end logits per token, loss $-\tfrac12(\log p_s(i^*) + \log p_e(j^*))$, decode $\arg\max_{i\le j<i+L_{\max}} s_i + e_j$; metrics EM and token F1 after SQuAD normalisation.`,
          String.raw`Translation and abstractive summarisation: encoder–decoder with teacher-forced loss $-\sum_t \log p(y_t \mid y_{<t}, x)$. Evaluate decoded output with BLEU $= \mathrm{BP}\cdot\exp(\tfrac14\sum\log p_n)$ or ROUGE-N/L, plus human or faithfulness checks.`,
          "Experiments: BoW unigram 0.679 → bigram 0.913 → BiGRU 0.999 accuracy on compositional sentiment; NER token accuracy 0.944 vs entity F1 0.838 on unseen names; lead-2 beats a naive frequency summariser; a 174k-parameter `nn.Transformer` translates unseen numbers into French with 98.7% exact match and BLEU 99.8, failing on rare irregular forms.",
          "Choosing a model: fixed labels + high volume → fine-tuned encoder; few labels + open-ended or changing tasks → prompted LLM. Always compare against a trivial baseline with the right metric.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[Language modelling and perplexity](/learn/m13/language-modeling) — the pretraining objective behind every model used here, and BLEU/ROUGE in language-model evaluation.",
          "[Transfer learning and fine-tuning](/learn/m12/transfer-learning) — how to fine-tune the pretrained bodies we put heads on: learning rates, freezing, data sizes.",
          "[Instruction tuning, RLHF and DPO](/learn/m20/alignment) — why a prompted LLM can do these tasks without task-specific training, and where it falls short.",
          "[Multimodal models and RAG foundations](/learn/m15/multimodal) — retrieval-augmented generation, the open-domain extension of extractive QA.",
          "[Decoding: greedy, beam, top-k, top-p](/learn/m20/decoding) — how translation and summarisation models turn distributions into text.",
        ],
      },
    ],
  },
};

export default lesson;
