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
    "The five workhorse NLP tasks as concrete machine-learning problems: how each is framed, what goes in and comes out, which head sits on which encoder or decoder, which loss trains it and which metric judges it — with BIO tagging and entity-level F1, CRF decoding, start/end span search with SQuAD exact-match/F1, BLEU and ROUGE implemented from scratch, and four real experiments (bag-of-words vs BiGRU sentiment, a BiGRU NER tagger, a tiny nn.Transformer translator).",
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
        text: "Most practical natural-language processing reduces to a handful of **task shapes**. A model either assigns **one label to a whole text** (spam or not, positive or negative, which of 40 intents), assigns **one label to every token** (is this word part of a person's name?), **points at a span** of a given text (where in this paragraph is the answer?), or **generates a new sequence** (a translation, a summary, a free-form answer). Text classification and sentiment analysis, named-entity recognition (NER), extractive question answering, machine translation and summarisation are the canonical instances of these four shapes, and almost every other NLP task — intent detection, part-of-speech tagging, relation extraction, grammatical error correction, code generation — borrows one of them.",
      },
      {
        t: "p",
        text: "Each task has a long history of specialised systems: feature-engineered classifiers, conditional random fields for tagging, phrase-based statistical translation. Deep learning unified them in two steps. First, recurrent encoders and encoder–decoders (Sutskever, Vinyals and Le, 2014, *Sequence to Sequence Learning with Neural Networks*; Bahdanau et al.'s attention, covered in [From Seq2Seq to Attention](/learn/m10/attention-intro)) let one architecture family read and write sequences. Second, **pretrained Transformers** (BERT, GPT, T5 — see [BERT, GPT, T5](/learn/m11/transformer-families)) turned every task into *pretrained body + small task-specific head + fine-tuning*, or, for large decoder-only models, *pretrained body + a prompt*.",
      },
      {
        t: "p",
        text: "This lesson treats each task as an engineering problem. For every task you will see the **framing** (what exactly is predicted), the **input/output representation**, the **head** placed on an encoder or decoder, the **loss**, the **metric** and why it is chosen, the **data pitfalls** that silently ruin results, and a **worked numerical example**. We then implement the metrics that are easiest to get subtly wrong — entity-level F1, best-span search, SQuAD exact match and F1, BLEU, ROUGE — and run four small but real experiments.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The one idea to remember",
        text: "Pick the task shape first: **sequence → label**, **sequence → one label per token**, **sequence → span (start, end)**, or **sequence → sequence**. The shape fixes the head, the loss and the metric; the encoder or decoder underneath is largely interchangeable.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Text preprocessing and subword tokenization](/learn/m13/text-preprocessing) — token IDs, special tokens such as `[CLS]`/`[SEP]`, padding and attention masks. NER and QA labels live on *words* or *characters*, so you must be able to map them onto subword tokens and back.",
          "[BERT, GPT, T5](/learn/m11/transformer-families) — encoder-only, decoder-only and encoder–decoder models and their pretraining objectives. This lesson puts heads on top of them; it does not re-derive them.",
          "[From Seq2Seq bottlenecks to attention](/learn/m10/attention-intro) — encoder–decoder translation, teacher forcing and greedy decoding, the origin of neural MT.",
          "[GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants) — the bidirectional GRU we use as a tagger, and why a bidirectional model is fine for tagging a finished sentence but not for generation.",
          String.raw`[Classification losses](/learn/m05/classification-losses) — softmax and cross-entropy $-\log p_y$, which every head in this lesson uses in some form.`,
          "[Classification metrics](/learn/m17/classification-metrics) — precision, recall, F1, macro-averaging and confusion matrices. We reuse them at the level of entities and n-grams.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "One careful reader, five different answer sheets",
        text: "Imagine a skilled reader who has read a document closely. You can hand them different answer sheets. A **tick-box sheet** asks for one verdict on the whole document (positive / negative / neutral). A **highlighter sheet** asks them to colour every word that belongs to a person, a place or a date. A **pointer sheet** asks them to circle the exact phrase that answers a question. A **blank page** asks them to rewrite the document in another language or in three sentences. The reading — the encoder — is the same; only the sheet — the head — and the way you grade it differ. A tick box is graded right or wrong; a highlighter sheet is graded by whole phrases, not individual words; a rewritten page can only be graded by comparing it with reference rewrites.",
      },
      {
        t: "p",
        text: "That last point matters more than it looks. For classification, a prediction is simply correct or not. For NER, getting 2 of the 3 words of `7 November 1891` right is still a *wrong date* — so we grade whole entities. For extractive QA, `in 1889` vs `1889` is nearly right, so we grade with token overlap. For translation and summarisation there are many correct outputs, so we compare with one or more references by counting shared n-grams — a crude proxy for meaning, which is why these metrics are argued about.",
      },
      {
        t: "p",
        text: "The other intuition is about **what the model can see**. A bag-of-words classifier sees which words occur but not their order, so `the film was not good` and `the film was good, not bad` look almost the same. A sequence model sees order and can learn that *not* flips what follows. Experiment (a) below measures exactly that difference.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "p", text: "Common skeleton: text → tokenizer → token IDs $x_1..x_n$ → **encoder** (BiGRU, BERT-style Transformer) producing one vector $h_i \\in \\mathbb{R}^d$ per token, and/or a **decoder** that generates tokens one at a time → **task head** → loss during training, decoding + metric at evaluation." },

      { t: "h", text: "1. Text classification and sentiment analysis" },
      {
        t: "list",
        items: [
          "**Framing.** Map a whole text to one of $C$ classes (single-label) or to a subset of labels (multi-label, e.g. toxicity categories). Sentiment analysis is classification with polarity labels (negative/neutral/positive, or 1–5 stars); *aspect-based* sentiment assigns polarity per aspect (\"food: +, service: −\").",
          "**Input/output.** Token IDs plus attention mask in; a vector of $C$ logits out.",
          String.raw`**Head.** Pool the token vectors into one vector — the final $h_{[\mathrm{CLS}]}$ of a BERT-style encoder, a mean over non-padding tokens, or a max over time (what our BiGRU uses) — then a linear layer $z = W h + b$, $W \in \mathbb{R}^{C\times d}$. A decoder-only LM instead uses the last token's state, or is simply prompted to emit the label word.`,
          "**Loss.** Softmax cross-entropy (single-label); independent sigmoid + binary cross-entropy per label (multi-label). Class weights or resampling for imbalance.",
          "**Metric.** Accuracy when classes are balanced; **macro-F1** (average of per-class F1) when they are not, so a rare class counts as much as a frequent one; the **confusion matrix** to see *which* classes are confused.",
          "**Data pitfalls.** Label imbalance (99% \"not spam\" makes 99% accuracy trivial); near-duplicate texts across train and test (reposts, templates) inflate scores; annotator disagreement on neutral/mixed texts caps achievable accuracy; spurious cues (product names, review length) that correlate with labels; domain shift (movie-review sentiment does not transfer cleanly to finance).",
        ],
      },

      { t: "h", text: "2. Named-entity recognition (sequence labelling)" },
      {
        t: "list",
        items: [
          "**Framing.** Find every mention of a person, organisation, location, date… and its type. Each entity is a *span* of one or more words. It is cast as **sequence labelling**: predict one tag per token such that the tags spell out the spans.",
          "**BIO tagging.** `B-X` begins an entity of type X, `I-X` continues it, `O` is outside any entity: `Marie/B-PER Curie/I-PER moved/O to/O Paris/B-LOC`. The `B` is needed to separate adjacent entities of the same type (`[Alice] [Bob]` vs `[Alice Bob]`). **BIOES** (also called BILOU) adds `E-X` for the last token and `S-X` for single-token entities, giving the model more explicit boundary targets; **IO** drops `B` and cannot represent adjacent same-type entities. With $K$ entity types BIO has $2K+1$ tags, BIOES $4K+1$.",
          String.raw`**Head.** A linear layer on every token vector: $z_i = W h_i + b$ with $W \in \mathbb{R}^{T\times d}$ for $T$ tags. Optionally a **CRF layer** on top (below). With subword tokenizers, usually only the first subword of each word gets a label; the rest are ignored in the loss (label $-100$ in PyTorch).`,
          "**Loss.** Token-level cross-entropy averaged over real (non-padding, labelled) tokens; or, with a CRF, the negative log-likelihood of the whole gold tag sequence.",
          "**Metric.** **Entity-level (span) F1**: convert gold and predicted tags to sets of (type, start, end) spans; a prediction counts only if type *and both boundaries* match exactly (the CoNLL-2003 shared-task convention, Tjong Kim Sang and De Meulder, 2003). **Token accuracy** is misleading because most tokens are `O`: tagging everything `O` scores 50% token accuracy on our test set and 0 entity F1.",
          "**Data pitfalls.** Entity names overlap between train and test (the model memorises names instead of learning context — our experiment deliberately uses disjoint name lists); inconsistent annotation of boundaries (`the United Nations` vs `United Nations`); nested entities (`[Bank of [Scotland]]`) cannot be expressed with flat BIO; misaligned labels after subword tokenization; invalid tag sequences (`O I-PER`) produced by per-token argmax.",
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "The CRF layer idea",
        text: String.raw`A per-token softmax chooses each tag independently, so it can output "O I-PER" (an entity that never began) or "B-PER I-LOC". A **linear-chain conditional random field** (Lafferty, McCallum and Pereira, 2001) scores the *whole* tag sequence: emission scores from the encoder plus a learned transition score $A_{y_{i-1},y_i}$ for every adjacent tag pair. Training maximises the log-probability of the gold sequence (normalised over all $T^n$ sequences with the forward algorithm); decoding finds the best sequence with the Viterbi algorithm. BiLSTM + CRF was the standard neural NER architecture before pretrained Transformers (Huang, Xu and Yu, 2015; Lample et al., 2016, *Neural Architectures for Named Entity Recognition*). With strong fine-tuned encoders the CRF adds less, but it still guarantees well-formed outputs.`,
      },

      { t: "h", text: "3. Machine translation" },
      {
        t: "list",
        items: [
          String.raw`**Framing.** Conditional generation: model $p(y_1..y_m \mid x_1..x_n)$ for a target-language sentence $y$ given a source sentence $x$, factorised left to right as $\prod_t p(y_t \mid y_{<t}, x)$.`,
          "**Input/output.** Source token IDs into the encoder; target token IDs, shifted right with a beginning-of-sequence token, into the decoder (**teacher forcing** during training); a distribution over the target vocabulary at every output position. Usually a shared subword vocabulary for both languages.",
          "**Model and head.** An **encoder–decoder**: bidirectional encoder over the source, causal decoder with cross-attention to the encoder states, and a linear **LM head** $W_o h_t \\in \\mathbb{R}^{V}$ (often tied to the target embeddings). Decoder-only LLMs translate when prompted, treating source + target as one sequence.",
          "**Loss.** Token-level cross-entropy on the target tokens (padding ignored), often with label smoothing.",
          "**Inference.** Greedy or beam search (see [Decoding](/learn/m20/decoding)) — the model now consumes its own predictions.",
          "**Metric.** **BLEU** (Papineni et al., 2002): clipped n-gram precision for $n = 1..4$, geometric mean, times a brevity penalty, computed over the whole corpus. chrF (character n-gram F-score) and learned metrics such as COMET correlate better with human judgement; human evaluation remains the reference. BLEU is defined and also used in [Language Modelling](/learn/m13/language-modeling).",
          "**Data pitfalls.** Misaligned or noisy sentence pairs in web-mined corpora; test sentences that appear in the training data; BLEU scores computed with different tokenizations are **not comparable** (always report the exact tool and settings); translationese and domain mismatch; single references penalise valid paraphrases.",
        ],
      },

      { t: "h", text: "4. Summarisation" },
      {
        t: "list",
        items: [
          "**Framing.** Produce a much shorter text that preserves the key content of a document (single- or multi-document). **Extractive** summarisation selects sentences from the source; **abstractive** summarisation generates new text that may paraphrase, compress and fuse sentences.",
          "**Extractive model.** Encode each sentence (e.g. the vector at a per-sentence marker token), score it with a linear layer + sigmoid, train with binary cross-entropy against oracle labels (the sentence set that maximises ROUGE with the reference), select the top-$k$ sentences, optionally removing redundant ones.",
          "**Abstractive model.** Exactly the translation setup — encoder–decoder (BART, T5, pointer-generator networks of See, Liu and Manning, 2017, which can copy source words) or a prompted LLM — with token cross-entropy on reference summaries.",
          "**Metric.** **ROUGE** (Lin, 2004): ROUGE-N is n-gram overlap with the reference (recall-oriented originally, usually reported as F1 today); ROUGE-L uses the longest common subsequence. Always pair with a check of **faithfulness**: abstractive models can produce fluent sentences that the source does not support (hallucination), which ROUGE barely penalises.",
          "**Data pitfalls.** The **lead bias** of news — the first sentences already summarise the article, so a trivial *lead-k* baseline is strong (it beats our naive frequency summariser below); reference summaries that contain facts absent from the source; truncation of long inputs to the encoder's maximum length silently drops content; ROUGE rewards length if you report recall only.",
        ],
      },

      { t: "h", text: "5. Question answering" },
      {
        t: "list",
        items: [
          String.raw`**Extractive QA (SQuAD format).** Given a question and a context paragraph, the answer is a span of the context. SQuAD (Rajpurkar, Zhang, Lopyrev and Liang, 2016, *SQuAD: 100,000+ Questions for Machine Comprehension of Text*) stores each example as context, question, and answers given as text plus a **character** start offset. SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018) adds unanswerable questions.`,
          "**Input/output.** `[CLS] question [SEP] context [SEP]` into an encoder; two scores per token out: a start logit $s_i$ and an end logit $e_i$. Character offsets must be converted to token indices (with the tokenizer's offset mapping), and contexts longer than the maximum length are split into overlapping windows (a *stride*).",
          String.raw`**Head.** One linear layer $\mathbb{R}^d \to \mathbb{R}^2$ applied to every token vector gives $(s_i, e_i)$. Start and end distributions are softmaxes over positions.`,
          "**Loss.** Cross-entropy of the gold start position plus cross-entropy of the gold end position, averaged. For unanswerable questions the target is the `[CLS]` position.",
          String.raw`**Decoding.** Return the span $(i, j)$ maximising $s_i + e_j$ subject to $i \le j$, $j - i + 1 \le L_{\max}$ and both inside the context (not the question). Taking the two argmaxes independently can produce $j < i$.`,
          "**Metric.** **Exact match** (EM) and **token-level F1** after normalisation (lower-case, strip punctuation and the articles *a/an/the*, collapse whitespace), each taking the maximum over the available reference answers, averaged over questions.",
          String.raw`**Generative QA** produces the answer with a decoder (T5, an LLM), so it can answer questions whose answer is not a contiguous span, at the price of possible hallucination. **Open-domain / retrieval-augmented QA** first retrieves passages from a large collection (dense retrievers such as DPR, Karpukhin et al., 2020) and then reads them extractively or generatively; retrieval-augmented generation (Lewis et al., 2020) conditions the generator on retrieved passages and can return them as pointers to evidence — see [Multimodal models and RAG](/learn/m15/multimodal).`,
          "**Data pitfalls.** Off-by-one character offsets; answers cut by context windows; questions written by annotators who looked at the paragraph overlap lexically with the answer sentence, so models learn word-matching shortcuts; EM penalises harmless variations, F1 rewards partially wrong spans.",
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
          ["Summarisation", "Short sequence (extract or generate)", "Encoder–decoder (abstractive); sentence scorer (extractive)", "LM head; or per-sentence sigmoid", "Token CE; or sentence BCE", "ROUGE-1/2/L + faithfulness checks"],
        ],
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Classification: pooled vector → softmax → cross-entropy" },
      { t: "math", tex: String.raw`z = W h_{\text{pool}} + b,\qquad p_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}},\qquad \mathcal{L} = -\log p_{y},\qquad \frac{\partial \mathcal{L}}{\partial z} = p - \mathbf{1}_y`, caption: "$h_{\\text{pool}}\\in\\mathbb{R}^d$ pooled text vector, $W\\in\\mathbb{R}^{C\\times d}$, $b\\in\\mathbb{R}^C$, $y$ the gold class, $\\mathbf{1}_y$ its one-hot vector." },
      {
        t: "steps",
        items: [
          { title: "Logits for (positive, neutral, negative)", tex: String.raw`z = (2.1,\ 0.3,\ -1.2)` },
          { title: "Exponentiate and normalise", tex: String.raw`e^{z} = (8.1662,\ 1.3499,\ 0.3012),\ \ \textstyle\sum = 9.8172 \Rightarrow p = (0.8318,\ 0.1375,\ 0.0307)` },
          { title: "Loss if the gold label is positive", tex: String.raw`\mathcal{L} = -\ln 0.8318 = 0.1841` },
          { title: "Loss if the gold label is negative", tex: String.raw`\mathcal{L} = -\ln 0.0307 = 3.4841 \quad\text{(confidently wrong is expensive)}` },
          { title: "Gradient w.r.t. the logits (gold = positive)", tex: String.raw`p - \mathbf{1}_y = (-0.1682,\ 0.1375,\ 0.0307)` },
        ],
      },
      { t: "p", text: "Macro-F1 averages per-class F1: $\\text{F1}_{\\text{macro}} = \\frac{1}{C}\\sum_c \\frac{2P_cR_c}{P_c+R_c}$, so a classifier that ignores a 5% class loses a third of a three-class macro-F1 even if accuracy barely moves." },

      { t: "h", text: "NER: token accuracy vs entity-level F1" },
      { t: "math", tex: String.raw`P = \frac{|\mathcal{S}_{\text{pred}} \cap \mathcal{S}_{\text{gold}}|}{|\mathcal{S}_{\text{pred}}|},\qquad R = \frac{|\mathcal{S}_{\text{pred}} \cap \mathcal{S}_{\text{gold}}|}{|\mathcal{S}_{\text{gold}}|},\qquad F_1 = \frac{2PR}{P+R} = \frac{2\,|\mathcal{S}_{\text{pred}} \cap \mathcal{S}_{\text{gold}}|}{|\mathcal{S}_{\text{pred}}| + |\mathcal{S}_{\text{gold}}|}`, caption: "$\\mathcal{S}$ = set of (type, start, end) spans decoded from the BIO tags, pooled over the whole corpus (micro-average)." },
      { t: "p", text: "Sentence: `Marie Curie moved from Warsaw to Paris on 7 November 1891 .` (12 tokens). Gold spans: PER(0–1), LOC(4), LOC(6), DATE(8–10). The model tags `Curie` as `O` and `1891` as `O`; everything else is right." },
      {
        t: "steps",
        items: [
          { title: "Token accuracy", tex: String.raw`\frac{10 \text{ correct tags}}{12} = 0.833` },
          { title: "Predicted spans", text: "PER(0–0), LOC(4), LOC(6), DATE(8–9). Two spans have the wrong right boundary, so only the two locations match exactly." },
          { title: "Entity-level scores", tex: String.raw`P = \tfrac{2}{4} = 0.5,\quad R = \tfrac{2}{4} = 0.5,\quad F_1 = 0.5` },
          { title: "All-O baseline", tex: String.raw`\text{token acc} = \tfrac{5}{12} = 0.417,\qquad F_1 = 0` },
        ],
      },
      { t: "p", text: "Two wrong tokens out of twelve cost 17 points of token accuracy but 50 points of entity F1: every boundary error destroys a whole entity *and* adds a false positive." },

      { t: "h", text: "The CRF layer: scoring and decoding whole tag sequences" },
      { t: "math", tex: String.raw`\mathrm{score}(x, y) = \pi_{y_1} + \sum_{i=1}^{n} E_{i, y_i} + \sum_{i=2}^{n} A_{y_{i-1}, y_i},\qquad p(y \mid x) = \frac{\exp \mathrm{score}(x,y)}{Z(x)},\quad Z(x) = \sum_{y'} \exp \mathrm{score}(x,y')`, caption: "$E_{i,t}$ emission score of tag $t$ at token $i$ (the encoder's per-token logits), $A_{s,t}$ learned transition score from tag $s$ to tag $t$, $\\pi_t$ start score. The sum in $Z$ runs over all $T^n$ tag sequences." },
      { t: "p", text: "Both $\\log Z$ and the best sequence are computed in $O(nT^2)$ by dynamic programming. The forward algorithm uses $\\alpha_i(t) = \\log\\sum_s \\exp(\\alpha_{i-1}(s) + A_{s,t}) + E_{i,t}$; Viterbi replaces the log-sum-exp by a max and keeps back-pointers. The training loss is $\\log Z(x) - \\mathrm{score}(x, y_{\\text{gold}})$." },
      {
        t: "steps",
        items: [
          { title: "Setup (tags O, B-PER, I-PER; tokens Marie, Curie, moved)", text: "Emissions per token: Marie (1.0, 0.8, 0.9), Curie (0.2, 0.1, 1.5), moved (2.0, 0.0, 0.3). Transitions O→I-PER and start→I-PER are forbidden ($-10^4$); B-PER→I-PER = 1.0, O→O = 0.5, I-PER→I-PER = 0.5, B→B = I→B = −0.5, others 0." },
          { title: "Per-token argmax", text: "O, I-PER, O — an invalid sequence (an entity continues without beginning)." },
          { title: "Score of the gold path B-PER I-PER O", tex: String.raw`0 + 0.8 + 1.0 + 1.5 + 0 + 2.0 = 5.30` },
          { title: "Viterbi result", text: "B-PER, I-PER, O with score 5.30 — the transition scores repair the first token's narrow emission preference for O." },
          { title: "CRF loss of the gold path", tex: String.raw`\log Z = 6.0812,\qquad \mathcal{L} = 6.0812 - 5.30 = 0.7812 \;\Rightarrow\; p(y_{\text{gold}} \mid x) = e^{-0.7812} = 0.458` },
        ],
      },

      { t: "h", text: "Extractive QA: span scores and SQuAD metrics" },
      { t: "math", tex: String.raw`p_{\text{start}}(i) = \frac{e^{s_i}}{\sum_k e^{s_k}},\quad p_{\text{end}}(j) = \frac{e^{e_j}}{\sum_k e^{e_k}},\quad \mathcal{L} = -\tfrac12\big(\log p_{\text{start}}(i^*) + \log p_{\text{end}}(j^*)\big),\quad (\hat\imath, \hat\jmath) = \arg\max_{i \le j < i + L_{\max}} s_i + e_j` },
      { t: "p", text: "Maximising $s_i + e_j$ is the same as maximising $p_{\\text{start}}(i)\\,p_{\\text{end}}(j)$, since the two normalisers do not depend on $(i, j)$. The constrained search costs $O(n L_{\\max})$." },
      {
        t: "steps",
        items: [
          { title: "Context (11 tokens)", text: "`the eiffel tower was completed in 1889 and stands in paris`; question: *When was the Eiffel Tower completed?*" },
          { title: "Logits", text: "Start: 0.1, 0.3, 0.2, −1.0, 0.5, 1.2, **3.1**, −0.5, −1.2, 0.4, 2.0. End: −0.8, 0.1, 0.9, −0.6, **3.3**, 0.7, 2.9, −0.4, −1.0, 0.2, 2.2." },
          { title: "Independent argmax", text: "start = 6 (`1889`), end = 4 (`completed`): end < start, not a span." },
          { title: "Constrained best span", tex: String.raw`(6, 6):\ 3.1 + 2.9 = 6.0 \;>\; (6,10)\!:\ 5.3\ \text{(also too long for } L_{\max}=4),\ (10,10)\!:\ 4.2,\ (5,6)\!:\ 4.1` },
          { title: "Span probability", tex: String.raw`p_{\text{start}}(6)\,p_{\text{end}}(6) = 0.5414 \times 0.2878 = 0.1558` },
          { title: "Metrics for the prediction \"in 1889\" against gold \"1889\"", tex: String.raw`\text{EM} = 0,\quad P = \tfrac{1}{2},\ R = \tfrac{1}{1},\ F_1 = \tfrac{2 \cdot 0.5 \cdot 1}{1.5} = 0.667` },
        ],
      },
      { t: "p", text: "Normalisation matters: `The Eiffel Tower.` and `Eiffel tower` are an exact match after lower-casing and stripping the article and the full stop; `Gustave Eiffel's company` vs `the company of Gustave Eiffel` gets F1 0.571 because stripping the apostrophe turns `Eiffel's` into the unmatched token `eiffels`." },

      { t: "h", text: "Machine translation: BLEU" },
      { t: "math", tex: String.raw`p_n = \frac{\sum_{\text{sent}} \sum_{g \in \text{n-grams(hyp)}} \min\!\big(\mathrm{count}_{\text{hyp}}(g),\ \max_{r} \mathrm{count}_{r}(g)\big)}{\sum_{\text{sent}} \#\text{n-grams(hyp)}},\qquad \mathrm{BP} = \begin{cases}1 & c > r\\ e^{\,1 - r/c} & c \le r\end{cases},\qquad \mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\frac{1}{4}\sum_{n=1}^{4}\log p_n\Big)`, caption: "$c$ = total hypothesis length, $r$ = total effective reference length (per sentence, the reference length closest to the hypothesis length). Counts are summed over the corpus before dividing." },
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
      { t: "p", text: "Two warnings from the from-scratch run: the single sentence `the cat sat on the mat` scores BLEU **0** because it has no matching 4-gram (sentence-level BLEU needs smoothing), while the scrambled `on the mat the cat is sitting` scores **59.5** — every word is right, the order is not. BLEU measures n-gram overlap, not adequacy." },

      { t: "h", text: "Summarisation: ROUGE" },
      { t: "math", tex: String.raw`\text{ROUGE-N}: \ P = \frac{\text{overlap}_n}{\#\text{n-grams(cand)}},\ R = \frac{\text{overlap}_n}{\#\text{n-grams(ref)}};\qquad \text{ROUGE-L}: \ P = \frac{\mathrm{LCS}}{|\text{cand}|},\ R = \frac{\mathrm{LCS}}{|\text{ref}|};\qquad F_1 = \frac{2PR}{P+R}` },
      {
        t: "steps",
        items: [
          { title: "Lead-2 summary vs a 22-word reference", text: "The first two sentences of the bridge article (28 words) share 13 clipped unigrams with the reference; the longest common subsequence has 11 words." },
          { title: "ROUGE-1", tex: String.raw`P = \tfrac{13}{28} = 0.464,\ R = \tfrac{13}{22} = 0.591,\ F_1 = \tfrac{2\cdot 13}{28 + 22} = 0.520` },
          { title: "ROUGE-L", tex: String.raw`F_1 = \tfrac{2\cdot 11}{28 + 22} = 0.440` },
        ],
      },

      { t: "h", text: "Sequence-to-sequence training objective" },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{seq2seq}} = -\sum_{t=1}^{m} \log p_\theta\big(y_t \mid y_{<t}, x\big)`, caption: "Teacher forcing: during training $y_{<t}$ are the gold previous tokens. At test time they are the model's own outputs, which is why test-time metrics must use real decoding." },
      { t: "p", text: "Translation and abstractive summarisation share this loss exactly; they differ in data and in metric." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "p", text: "Three existing simulators each show one ingredient of these tasks." },
      { t: "h", text: "1. Encoder–decoder alignment (translation, summarisation)" },
      { t: "sim", id: "seq2seq-attention", caption: "A real trained GRU encoder–decoder with additive attention (string reversal). Translation and abstractive summarisation use the same encode → attend → emit loop." },
      {
        t: "list",
        items: [
          "Decode a word step by step and watch each output token's attention row. In translation the alignment matrix is close to diagonal for languages with similar word order and bends where words are reordered (in our number task, `ninety seven` → `quatre vingt dix sept` spreads one source word over several target words).",
          "Note that the decoder consumes its own previous output at every step. A single early mistake changes everything after it — the reason BLEU must be computed on decoded output, not teacher-forced predictions.",
          "Read the accuracy panel: beyond the training lengths accuracy falls. Summarisation models trained on short articles suffer the same way when given long documents.",
        ],
      },
      { t: "h", text: "2. Bidirectional vs causal attention (choosing an encoder or a decoder)" },
      { t: "sim", id: "attention", caption: "Type a short QA-style input such as `when was it built SEP it was built in 1889` and toggle the causal mask." },
      {
        t: "list",
        items: [
          "Without the causal mask every token attends to every other token: the context word `1889` can look at the question words. That is what an extractive-QA encoder needs to score start and end positions.",
          "Switch the causal mask on: each token now sees only the tokens to its left. A decoder-only model reading `question … context` can still connect them, but a token's representation never sees what comes after it — one reason fine-tuned bidirectional encoders remain strong for tagging and span extraction at small sizes.",
          "The weights are seeded random, not trained, so look at which cells are allowed to be non-zero, not at the patterns.",
        ],
      },
      { t: "h", text: "3. Subword tokens vs word-level labels (NER, QA)" },
      { t: "sim", id: "tokenizer", caption: "BPE trained live on the text in the box." },
      {
        t: "list",
        items: [
          "After a few merges, encode a rare name such as `sklodowska`: it is split into several pieces. In NER only one of those pieces (usually the first) carries the word's tag; the others get the ignore label, and at prediction time you read the tag of the first piece.",
          "For QA the gold answer is given as character offsets; you need each token's character span (the *offset mapping*) to find which tokens contain the answer's first and last characters.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "Two NumPy programs implement the parts of these tasks that are not just \"a linear layer + cross-entropy\": decoding and evaluation. The first converts BIO tags to spans and computes token accuracy and entity-level F1, decodes a CRF with Viterbi and computes its loss with the forward algorithm, finds the best QA span under the $i \\le j$ and length constraints (loop and vectorised versions), and implements SQuAD normalisation, exact match and F1. All numbers in the Math section come from this run." },
      { t: "code", lang: "python", title: "tagging_and_qa_from_scratch.py", code: TAG_QA, output: TAG_QA_OUT },
      {
        t: "list",
        items: [
          "`bio_to_spans` follows the conlleval convention: an `I-X` that does not continue an `X` entity starts a new entity. Some libraries discard such spans instead (\"strict\" mode), which changes scores for models that emit invalid sequences — always state which convention you used.",
          "`best_span_vectorized` builds the full $n\\times n$ score matrix and masks it to the band $0 \\le j - i < L_{\\max}$; real QA code additionally masks positions that belong to the question or to padding.",
          "`token_f1` uses a multiset intersection (`Counter &`), so a repeated word is only credited as many times as it appears in both answers.",
        ],
      },
      { t: "p", text: "The second program implements corpus BLEU with clipping, multiple references and the brevity penalty, ROUGE-1/2/L, and a tiny extractive summariser (score each sentence by the average corpus frequency of its content words) compared with the lead-2 baseline. The article is a six-sentence text written for this lesson." },
      { t: "code", lang: "python", title: "bleu_rouge_summariser.py", code: BLEU_ROUGE, output: BLEU_ROUGE_OUT },
      {
        t: "callout",
        kind: "warning",
        title: "The naive summariser loses to lead-2",
        text: "The frequency summariser picks sentence 5 (`the mayor said the bridge was essential…`) because it repeats the frequent word *bridge*, and scores ROUGE-1 0.340 against 0.520 for simply taking the first two sentences. This is the lead bias of news text in miniature, and the reason every summarisation paper reports a lead baseline.",
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
              { t: "p", text: "A three-class (negative / neutral / positive) review dataset generated from eight templates, 600 training and 1,000 test sentences with no duplicates across the split. **Honest caveats:** the vocabulary has 75 words, labels follow deterministic rules (so they are noise-free), and real reviews are far more varied — treat the absolute numbers as a controlled demonstration of *what each model can represent*, not as a benchmark. The templates are designed to contain the constructions that defeat word counts: negation (`not ADJ`), long-range negation (`i did not think the X was ADJ`), contrast (`expected ADJ1 but it was ADJ2`, `ADJ1 X, although the Y was ADJ2`) and opinion words inside non-opinions (`i asked whether the X was great`)." },
              { t: "code", lang: "python", title: "sentiment_bow_vs_bigru.py", code: SENTIMENT, output: SENTIMENT_OUT },
              {
                t: "list",
                items: [
                  "**Unigram BoW (acc 0.679, macro-F1 0.689).** Neutral is perfect — its cue words (`asked`, `whether`, `?`, factual phrases) never occur elsewhere. But between negative and positive the model is close to a coin flip (137 negatives called positive, 184 positives called negative). A linear model over word counts cannot express \"polarity of the adjective, *flipped* if `not` is present\" — an XOR-like interaction. Because flipped and contrastive templates outnumber plain ones, it even gets the plain sentences `the X was ADJ` mostly wrong (0.26).",
                  "**Uni+bigram BoW (acc 0.913).** Bigrams such as `not good` or `be great` are conjunction features, so adjacent negation (1.00) and the `to be … was` contrast (0.92) become linearly separable. Plain sentences (0.67) and long-range negation (0.91) remain imperfect because the negation is not adjacent to the adjective.",
                  "**BiGRU (15,331 parameters, acc 0.999).** Reading the sequence in order, it represents negation scope and contrast directly; one test error out of 1,000. On real data the gap is smaller and the neural model needs much more data, but the direction is the same: order-aware models handle compositional sentiment that word counts cannot.",
                ],
              },
              { t: "h", text: "(b) NER: a BiGRU BIO tagger, token accuracy vs entity F1" },
              { t: "p", text: "Template sentences with PER, LOC, ORG and DATE entities (multi-word names such as `Rio de Janeiro`, `Bank of Avalon`, `12 March 2021`), plus traps: `may` as a verb, a number that is not a date. **Train and test use disjoint name lists**, and 100 of the 500 test sentences use two sentence shapes never seen in training. The tagger sees lower-cased word IDs plus a 4-way word-shape feature (lower / Capitalised / digits / other). We train it twice: once normally, once with **word dropout** (each training word replaced by `<unk>` with probability 0.25), which forces it to rely on context and shape — exactly what unseen names require." },
              { t: "code", lang: "python", title: "bigru_ner_tagger.py", code: NER, output: NER_OUT },
              {
                t: "list",
                items: [
                  "**Token accuracy and entity F1 tell different stories.** The all-O baseline already gets 0.503 token accuracy. Without word dropout the tagger reaches 0.795 token accuracy but only **0.367** entity F1 (ORG 0.000): every test name is `<unk>`, which the model never saw during training. With word dropout: 0.944 token accuracy vs **0.838** entity F1 — 5.6% of tokens wrong, but 16% of the entity score lost.",
                  "**Errors concentrate where the data shifted.** All printed errors come from the unseen sentence shape `LOC hosted PER and ORG on DATE`: in training, a capitalised sentence-initial name was always a person, so `Hong Kong` becomes `B-PER I-PER`, and `Blue Sun Labs` gets the invalid sequence `B-PER I-LOC O` — the kind of output a CRF would forbid.",
                  "Real NER datasets (CoNLL-2003 has PER, ORG, LOC, MISC) are messier than this generator; use it to understand the metric and the failure modes, not as an estimate of real-world F1.",
                ],
              },
              { t: "h", text: "(d) A tiny nn.Transformer translator with BLEU" },
              { t: "p", text: "Toy translation: English number words → French number words for 0–999, a deterministic language pair with real structure — French restructures 70–99 (`seventy one` → `soixante et onze`, `ninety seven` → `quatre vingt dix sept`) and has agreement quirks (`quatre vingts` but `quatre vingt un`; `deux cents` but `deux cent un`). 700 numbers train, the other 300 are **never seen** in training, so good test scores require compositional generalisation, not memorisation. The model is `nn.Transformer` with 2 encoder and 2 decoder layers, $d=64$, learned position embeddings and greedy decoding; BLEU is the from-scratch corpus BLEU (one reference)." },
              { t: "code", lang: "python", title: "number_translation_transformer.py", code: TRANSLATE, output: TRANSLATE_OUT },
              { t: "p", text: "The model has 174,493 parameters. On the 300 held-out numbers greedy decoding gets **98.7% exact match** and corpus **BLEU 99.8**. The training loss levels off near 0.37 rather than 0 mainly because label smoothing (0.05) makes a zero loss impossible. Three of the four errors are the rare irregular forms: `quatre vingts` (80) and `quatre cents` (400) take a plural -s only when nothing follows, and 71 is `soixante et onze`. Each pattern covers only about ten of the 1,000 numbers, so the model sees very few examples of it; the fourth error (1 translated as `cent`) is a plain mistake on the shortest input. Note how forgiving BLEU is: 1.3% of the translations are wrong, yet BLEU is 99.8, because a missing -s costs one token out of roughly a thousand. Real translation has many valid outputs per sentence, so even very good systems score far lower against a single reference, and BLEU differences of a point or less are hard to interpret." },
              { t: "h", text: "Task heads on a Transformer encoder (Hugging Face, configuration only)" },
              { t: "p", text: "The `transformers` library ships the heads described in this lesson. Built from small configurations with **random weights** (nothing is downloaded), they show the output shapes, the size of each head and the loss each computes. The second part trains a 60-token WordPiece vocabulary on a few local sentences and aligns word-level BIO labels to subword tokens." },
              { t: "code", lang: "python", title: "task_heads_and_label_alignment.py", code: HEADS, output: HEADS_OUT },
              { t: "p", text: String.raw`The heads are tiny: 195 parameters for three classes ($64\times3+3$; BERT's tanh pooler counts as part of the body), 585 for nine BIO tags ($64\times9+9$) and 130 for QA ($64\times2+2$). Almost all the capacity is in the shared encoder, which is why one pretrained body can serve every task. With random weights each loss is close to the uniform-guess value: $\ln 3 = 1.0986$ vs 1.0792, $\ln 9 = 2.1972$ vs 2.1738, and $\ln 12 = 2.4849$ vs 2.5442 for QA (a softmax over the 12 positions). The untrained T5 scores 7.8151, above $\ln 1000 = 6.9078$, because its random logits are not uniform. In the alignment only the first piece of each word keeps the word's tag; the rest get −100 and are ignored by the loss. With a 60-token vocabulary even common words are fragmented (5 words become 17 tokens); with real 30k–50k vocabularies mostly rare words and names are split.` },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "A Keras BiGRU tagger with the same structure as experiment (b). `mask_zero=True` makes the embedding emit a padding mask that the bidirectional GRU respects, and `sample_weight` removes padding positions from the loss. The random tensors only stand in for a real tokenised corpus." },
              { t: "code", lang: "python", title: "keras_bigru_tagger.py (not executed in CI)", code: KERAS_TAGGER, output: "Prints three lines of training progress (one per epoch, loss for random labels stays near ln 9 ≈ 2.2) and then (2, 12) — one predicted tag id per token for two sentences." },
              { t: "p", text: "For pretrained Transformer heads in TensorFlow, the `transformers` library provides `TFAutoModelForSequenceClassification`, `TFAutoModelForTokenClassification` and `TFAutoModelForQuestionAnswering` with the same heads as the PyTorch classes above." },
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
          ["Bag-of-words / TF-IDF + linear model", "Word (n-gram) counts → logistic regression or linear SVM", "Trains in seconds, interpretable weights, strong baseline for topic classification", "No word order beyond n-grams; fails on negation scope and contrast (acc 0.679 vs 0.999 in experiment a)", "Baseline; topic/spam filtering at huge scale; very little compute"],
          ["CNN / BiLSTM / BiGRU on embeddings", "Learned or pretrained word vectors → convolution or recurrence → head", "Order-aware, small, fast on CPU", "Needs labelled data per task; weaker than pretrained Transformers", "Edge devices, streaming, small specialised tasks"],
          ["BiLSTM-CRF tagger", "Recurrent emissions + CRF transitions, Viterbi decoding", "Well-formed tag sequences, models label dependencies", "Sequential training; superseded in accuracy by fine-tuned encoders", "NER/POS with modest data, structured-output constraints"],
          ["Fine-tuned encoder (BERT family)", "Pretrained bidirectional encoder + task head, all weights fine-tuned", "State of the art for classification, NER, extractive QA at small cost; one forward pass", "Cannot generate text; needs labelled data (hundreds to thousands of examples); fixed max length", "Production classification/extraction with a fixed label set"],
          ["Fine-tuned encoder–decoder (T5, BART)", "Pretrained seq2seq model fine-tuned on input → output pairs", "Natural for translation and summarisation; any task as text-to-text", "Autoregressive decoding cost; can hallucinate", "Translation, summarisation, generative QA with paired data"],
          ["Prompted decoder-only LLM", "Instruction-tuned LLM with zero/few-shot prompts, optionally with retrieval", "No task-specific training data; handles open-ended and multi-step tasks", "Expensive per call, slower, output format must be parsed, less calibrated, harder to audit", "Few labels, many changing tasks, generation-heavy work"],
        ],
      },
      { t: "h", text: "Tagging schemes" },
      {
        t: "table",
        head: ["Scheme", "Tags for K types", "Example for `New York City`", "Notes"],
        rows: [
          ["IO", "K + 1", "I-LOC I-LOC I-LOC", "Cannot separate adjacent same-type entities"],
          ["BIO (IOB2)", "2K + 1", "B-LOC I-LOC I-LOC", "Standard; boundaries explicit at the start only"],
          ["BIOES / BILOU", "4K + 1", "B-LOC I-LOC E-LOC (single word: S-LOC)", "Explicit end and singleton tags: direct supervision for where entities end"],
          ["Span-based", "—", "score every (start, end, type) candidate", "Handles nested entities; $O(n^2)$ candidates, usually length-capped"],
        ],
      },
      { t: "h", text: "Task variants" },
      {
        t: "list",
        items: [
          "**Classification:** binary, multi-class, multi-label (sigmoid per label), hierarchical, aspect-based sentiment, natural-language inference (classify a premise–hypothesis *pair*: `[CLS] premise [SEP] hypothesis [SEP]`).",
          "**Sequence labelling:** NER, part-of-speech tagging, chunking, slot filling in dialogue systems, token-level error detection.",
          "**QA:** extractive (span), multiple-choice (score each option with a classification head), generative (decoder), open-domain retrieve-then-read, retrieval-augmented generation with citations, unanswerable-question detection (SQuAD 2.0's no-answer option).",
          "**Translation:** bilingual vs multilingual models (one model, many language pairs, target language given by a tag token), document-level translation, speech translation.",
          "**Summarisation:** extractive vs abstractive, single- vs multi-document, query-focused, long-document (needs long-context encoders or chunking).",
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Classification:** spam and phishing filters, support-ticket routing and intent detection in assistants, content moderation (toxicity, hate speech), sentiment monitoring of reviews and social media, document triage in legal and medical workflows.",
          "**NER and information extraction:** de-identifying clinical notes (finding names, dates, record numbers), extracting parties, amounts and dates from contracts and invoices, linking company mentions in financial news, building knowledge graphs, populating search filters.",
          "**Extractive and retrieval-augmented QA:** search engines that highlight the answer span in a result, enterprise document search, customer-support assistants that answer from a help-centre corpus and cite the passages they used.",
          "**Machine translation:** consumer translation services, localisation of software and documentation, cross-lingual search, real-time subtitling.",
          "**Summarisation:** meeting and call summaries, news digests, summarising long reports, emails or patient records for a quick overview — usually with a human checking faithfulness for high-stakes uses.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths and limitations of the task framings" },
      {
        t: "list",
        items: [
          "**Classification** is the cheapest and most reliable framing, with well-understood metrics — but forcing a nuanced text into one label loses information (mixed reviews, sarcasm), and label definitions drift over time.",
          "**Sequence labelling** is fast (one pass, one decision per token) and exact-span evaluation is strict and objective — but flat BIO cannot represent nested or discontinuous entities, and quality hinges on consistent boundary annotation.",
          "**Extractive QA** cannot hallucinate an answer that is not in the text and points to its evidence — but it fails when the answer must be synthesised, counted or is not a contiguous span.",
          "**Generation (translation, summarisation, generative QA)** is the most flexible framing — but decoding is slower, outputs can be fluent and wrong, and automatic metrics (BLEU, ROUGE) are weak proxies for quality: our scrambled sentence scored BLEU 59.5.",
        ],
      },
      { t: "h", text: "How to choose a model for an NLP task today" },
      {
        t: "table",
        head: ["Criterion", "Fine-tuned encoder (BERT-style)", "Prompted decoder-only LLM", "Fine-tuned encoder–decoder / small LLM"],
        rows: [
          ["Labelled data needed", "Hundreds to thousands of examples", "None to a few examples in the prompt", "Thousands of input–output pairs"],
          ["Cost and latency per input", "Lowest: one forward pass of a ~100M-parameter model", "Highest: large model, token-by-token generation", "Medium"],
          ["Output control", "Exactly the label set / spans you defined", "Free text: must be parsed and validated; may invent labels", "Text, but constrained by fine-tuning"],
          ["Accuracy on a fixed, well-labelled task", "Typically very strong", "Often competitive zero-shot, sometimes below a fine-tuned small model", "Strong for generation tasks"],
          ["Changing requirements / many tasks", "Retrain for each change", "Edit the prompt", "Retrain"],
          ["Privacy, auditing, on-device", "Easy to host and audit", "Hard if using a hosted API; large to self-host", "Moderate"],
        ],
      },
      {
        t: "list",
        items: [
          "**Start from the task shape and the volume.** Fixed labels or spans, high volume, latency or cost limits → fine-tune an encoder. Open-ended outputs, few labels, many evolving tasks → prompt an instruction-tuned LLM (see [Instruction tuning, RLHF and DPO](/learn/m20/alignment) for how base models become instruction followers).",
          "**A common, effective pattern:** use an LLM to bootstrap — label a few thousand examples or prototype the task — then check the labels, and fine-tune a small encoder for production. Keep the LLM for the hard tail.",
          "**Measure, do not assume.** Build a held-out test set with the *right* metric (entity F1, not token accuracy; EM/F1; ROUGE plus a faithfulness check) and compare a fine-tuned small model, a prompted LLM and a trivial baseline (majority class, all-O, lead-k) on it.",
          "**Parameter-efficient fine-tuning** (LoRA and relatives, see [Scaling and PEFT](/learn/m19/scaling-efficiency)) narrows the gap: you can adapt a mid-sized LLM to a task with modest compute when neither pure prompting nor a small encoder is enough.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Reporting token accuracy for NER.** Most tokens are `O`, so accuracy is dominated by the easy class: our all-O tagger scores 0.503 and a tagger with 0.944 token accuracy has only 0.838 entity F1. Fix: decode spans and report entity-level precision, recall and F1 (micro over entities, plus per type).",
          "**Labelling every subword piece with the word's tag, then evaluating on pieces.** `Sklodowska` split into several pieces is counted several times and B/I tags become inconsistent (`B-PER B-PER B-PER`). Fix: label only the first piece (ignore index −100 for the rest) and evaluate on words.",
          "**Taking independent argmaxes for the QA start and end.** It can return end < start or an absurdly long span. Fix: search $i \\le j < i + L_{\\max}$ over context tokens only, maximising $s_i + e_j$.",
          "**Comparing BLEU numbers from different tokenizers or tools.** BLEU on subword tokens, on detokenized text with a different tokenizer, or with smoothing differs by several points. Fix: compute BLEU on detokenized output with one standard implementation and report its settings.",
          "**Evaluating seq2seq models with teacher forcing.** Accuracy with gold previous tokens is far higher than with the model's own outputs. Fix: decode (greedy or beam) and score the decoded text.",
          "**Using a bag-of-words model and concluding the task is hard.** Negation and contrast need order; try n-grams or a sequence model before giving up (experiment a: 0.679 → 0.913 → 0.999).",
          "**Leaking entities or near-duplicate texts across splits.** The same names, templates or reposted reviews in train and test reward memorisation. Fix: split by source/document/time and test on unseen entities, as our NER experiment does.",
          "**Optimising ROUGE alone for abstractive summaries.** A model can increase ROUGE while inventing facts. Fix: add faithfulness checks (human review, entailment- or QA-based consistency checks) and always report a lead-k baseline.",
          "**Truncating long inputs silently.** QA contexts or documents longer than the maximum length lose the answer or key content. Fix: sliding windows with stride for QA, chunking or long-context models for summarisation, and monitor how often truncation happens.",
          "**Using accuracy on imbalanced classes.** With 95% negatives, a constant classifier scores 95%. Fix: macro-F1, per-class recall, a confusion matrix, and a threshold chosen on validation data.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What are the four basic output shapes of NLP tasks, with an example of each?", a: "One label per text (sentiment, spam, intent classification); one label per token (NER, part-of-speech tagging); a span of the input given by start and end positions (extractive QA); and a new output sequence (translation, abstractive summarisation, generative QA). The shape determines the head, the loss and the metric." },
      { t: "qa", level: "beginner", q: "What do the B, I and O tags mean in BIO tagging, and why is B needed?", a: "B-X marks the first token of an entity of type X, I-X a token inside the same entity, O a token outside any entity. Without B you cannot tell two adjacent entities of the same type (`[Alice] [Bob]`) from one two-word entity (`[Alice Bob]`); B marks where a new entity starts." },
      { t: "qa", level: "beginner", q: "What is the difference between extractive and abstractive summarisation?", a: "Extractive summarisation selects sentences (or phrases) from the source and concatenates them, so it is faithful to the wording but can be choppy and redundant. Abstractive summarisation generates new text with a decoder, so it can paraphrase and compress, but it can also state things the source does not support." },
      { t: "qa", level: "beginner", q: "Why is accuracy alone a poor metric for many text-classification problems?", a: "Real label distributions are often imbalanced. A classifier that always predicts the majority class can have very high accuracy while never detecting the class you care about. Macro-F1, per-class precision/recall and the confusion matrix show how each class is handled." },
      { t: "qa", level: "intermediate", q: "Why can a NER model have high token accuracy but low entity F1?", a: "Most tokens are O, so predicting O is right most of the time. Entity F1 requires the exact span and type: one wrong boundary token makes the predicted entity a false positive and leaves the gold entity as a false negative. In our worked example 10/12 tokens were right (0.833) but only 2 of 4 entities (F1 0.5)." },
      { t: "qa", level: "intermediate", q: "How does an extractive QA model produce an answer, and how is it trained?", a: "The question and context are encoded together; a linear layer gives each token a start logit and an end logit. Training minimises the average cross-entropy of the gold start and gold end positions (softmax over positions). At inference, choose the span maximising $s_i + e_j$ with $i \\le j$, a maximum length, and both positions inside the context; map token positions back to characters with the offset mapping." },
      { t: "qa", level: "intermediate", q: "Explain the components of BLEU.", a: "Clipped n-gram precisions $p_1..p_4$ (each hypothesis n-gram counts at most as often as in any one reference), combined by a geometric mean, multiplied by a brevity penalty $e^{1-r/c}$ when the hypothesis is shorter than the reference, because precision alone would reward very short outputs. Counts are summed over the corpus before dividing. It measures surface overlap, not meaning, and is not comparable across tokenizations." },
      { t: "qa", level: "intermediate", q: "What does a CRF layer add on top of a BiLSTM or BERT tagger?", a: "Learned transition scores between adjacent tags, so the model scores whole tag sequences instead of choosing each tag independently. Training maximises the log-likelihood of the gold sequence, normalised over all sequences with the forward algorithm; decoding uses Viterbi. It enforces valid sequences (no I-PER after O) and captures label dependencies; with strong pretrained encoders the accuracy gain is usually small." },
      { t: "qa", level: "advanced", q: "You must classify support tickets into 30 categories, 2 million tickets per day, with 5,000 labelled examples. Fine-tune an encoder or prompt an LLM?", a: "Fine-tune a small encoder: the label set is fixed, there is enough labelled data, and per-ticket cost and latency dominate at 2M/day — one forward pass of a ~100M-parameter model is orders of magnitude cheaper than LLM generation, and its outputs are guaranteed to be valid labels. An LLM is useful for bootstrapping more labels, handling an 'other/unclear' tail, or prototyping new categories. Validate on a held-out set with macro-F1 and per-class recall before deciding." },
      { t: "qa", level: "advanced", q: "Your abstractive summariser improved ROUGE-L by 2 points but users complain about wrong facts. What is happening and what do you do?", a: "ROUGE counts n-gram/LCS overlap with a reference; it barely penalises a substituted name or number, and can reward copying frequent reference phrasing. The model may be hallucinating content unsupported by the source. Add faithfulness evaluation (human checks of factual consistency, entailment- or QA-based consistency metrics), inspect errors, consider more extractive or copy-friendly models, constrain decoding, ground generation in retrieved source passages, and keep a lead-k baseline for comparison." },
      { t: "qa", level: "advanced", q: "How would you handle contexts longer than the encoder's maximum length in extractive QA, and what can go wrong?", a: "Split the context into overlapping windows (e.g. 384 tokens with a 128-token stride), each prefixed by the question; run the model on each window; score candidate spans in each window and take the best across windows, mapping back to characters via the offset mapping. Pitfalls: answers cut by a window boundary (the overlap mitigates this), comparing raw logits across windows (they are not calibrated across windows), training labels for windows that do not contain the answer (point them at [CLS]), and off-by-one character offsets." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which head is used for extractive question answering on a BERT-style encoder?", options: ["A linear layer on the [CLS] vector producing one logit per answer", "A linear layer on every token producing a start and an end logit", "A decoder that generates the answer token by token", "A CRF over BIO tags"], answer: 1, explain: "Each token gets $(s_i, e_i)$ from a $d\\to2$ linear layer; the answer is the best valid (start, end) pair." },
          { kind: "numeric", q: "Gold entities: 5. The tagger predicts 4 entities, of which 3 match exactly. What is the entity-level F1? (3 decimals)", answer: 0.667, tolerance: 0.001, explain: "$P = 3/4$, $R = 3/5$, $F_1 = 2\\cdot3/(4+5) = 0.667$." },
          { kind: "numeric", q: "SQuAD F1 of the prediction \"the treaty of Paris\" against the gold answer \"Treaty of Paris 1783\"? (3 decimals)", answer: 0.857, tolerance: 0.001, explain: "Normalised: prediction `treaty of paris` (3 tokens, the article is removed), gold `treaty of paris 1783` (4 tokens). Overlap 3: $P = 1$, $R = 0.75$, $F_1 = 2\\cdot0.75/1.75 = 0.857$." },
          { kind: "numeric", q: "A hypothesis corpus has total length 18 and effective reference length 20. What is the BLEU brevity penalty? (3 decimals)", answer: 0.895, tolerance: 0.001, explain: "$\\mathrm{BP} = e^{1 - 20/18} = e^{-0.1111} = 0.895$." },
          { q: "Start logits peak at token 7 and end logits peak at token 3. What should the decoder return?", options: ["The span 3–7", "The span 7–3, reversed", "The highest-scoring (i, j) with i ≤ j within the length limit", "No answer"], answer: 2, explain: "The independent argmaxes form no valid span; search all valid pairs for the maximal $s_i + e_j$." },
          { q: "Why did bigram features improve the bag-of-words sentiment model from 0.679 to 0.913 accuracy in experiment (a)?", options: ["Bigrams add more training data", "A bigram like `not good` is a conjunction feature, so negation adjacent to the adjective becomes linearly separable", "Bigrams remove stop words", "Bigrams make logistic regression non-linear"], answer: 1, explain: "Unigram counts cannot express polarity XOR negation; the bigram is itself the conjunction. Long-range negation still fails because the words are not adjacent." },
          { kind: "numeric", q: "How many tags does BIOES use for 4 entity types?", answer: 17, tolerance: 0, explain: "$4K + 1 = 17$: B, I, E, S for each type plus O." },
          { q: "Which statement about ROUGE is correct?", options: ["It measures factual consistency with the source", "ROUGE-L uses the longest common subsequence between candidate and reference", "It is computed against the source document, not a reference", "Higher ROUGE always means a better summary"], answer: 1, explain: "ROUGE compares with reference summaries by n-gram overlap (ROUGE-N) or LCS (ROUGE-L); it does not check faithfulness." },
        ],
      },
      {
        t: "exercise",
        title: "Score a QA prediction and a tagging output by hand",
        prompt: "(1) Context tokens 0–5 have start logits (0.2, 2.5, 0.1, 1.9, −0.3, 0.4) and end logits (0.0, 0.3, 2.2, −0.5, 2.6, 0.1). With $L_{\\max} = 3$, find the best span and its score; what would the independent argmaxes give? (2) Gold tags `B-ORG I-ORG I-ORG O B-LOC`, predicted `B-ORG I-ORG O O B-LOC`. Compute token accuracy and entity F1.",
        hint: "For (1) list $s_i + e_j$ for each start with $j \\in \\{i, i+1, i+2\\}$.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Independent argmaxes", text: "start = 1 (2.5), end = 4 (2.6): span 1–4 has length 4 > $L_{\\max}$ = 3, so it is not allowed." },
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
        prompt: "Write a function `bio_to_bioes(tags)` that converts a valid BIO sequence to BIOES (last token of a multi-token entity → `E-X`, single-token entity → `S-X`), and a function `bioes_to_spans(tags)`. Check on the Marie Curie example from the scratch code that `bioes_to_spans(bio_to_bioes(gold)) == bio_to_spans(gold)`.",
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
          { t: "p", text: "The spans are identical to those printed by `bio_to_spans` in the scratch program: BIOES is a re-encoding of the same information with explicit end and singleton tags, so entity-level F1 is unchanged by the conversion. What changes is the *training target*: the model gets direct supervision for where entities end." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Four task shapes: text → label (classification, sentiment), token → tag (NER), (start, end) span (extractive QA), sequence → sequence (translation, summarisation, generative QA).",
          String.raw`Classification: pooled vector → $W h + b$ → softmax cross-entropy; report macro-F1 and a confusion matrix when classes are imbalanced.`,
          "NER: BIO/BIOES tags, per-token linear head (optionally a CRF scoring $\\sum E_{i,y_i} + \\sum A_{y_{i-1},y_i}$, decoded with Viterbi); evaluate with exact-span **entity-level F1**, never token accuracy alone.",
          String.raw`Extractive QA: start/end logits per token, loss $-\tfrac12(\log p_s(i^*) + \log p_e(j^*))$, decode $\arg\max_{i\le j<i+L_{\max}} s_i + e_j$; metrics EM and token F1 after SQuAD normalisation.`,
          String.raw`Translation and abstractive summarisation: encoder–decoder with teacher-forced loss $-\sum_t \log p(y_t \mid y_{<t}, x)$; evaluate on decoded output with BLEU $= \mathrm{BP}\cdot\exp(\tfrac14\sum\log p_n)$ or ROUGE-N/L, plus human or faithfulness checks.`,
          "Experiments: BoW unigram 0.679 → bigram 0.913 → BiGRU 0.999 accuracy on compositional sentiment; NER token accuracy 0.944 vs entity F1 0.838 on unseen names; lead-2 beats a naive frequency summariser; a 174k-parameter `nn.Transformer` translates unseen numbers into French with 98.7% exact match and BLEU 99.8, failing on rare irregular forms.",
          "Choosing a model: fixed labels + volume → fine-tuned encoder; few labels + open-ended or changing tasks → prompted LLM; always compare against a trivial baseline with the right metric.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[Language modelling and perplexity](/learn/m13/language-modeling) — the pretraining objective behind every model used here, and BLEU/ROUGE in the context of language-model evaluation.",
          "[Transfer learning and fine-tuning](/learn/m12/transfer-learning) — how to fine-tune the pretrained bodies this lesson put heads on: learning rates, freezing, data sizes.",
          "[Instruction tuning, RLHF and DPO](/learn/m20/alignment) — why a prompted LLM can perform these tasks without task-specific training, and its limits.",
          "[Multimodal models and RAG foundations](/learn/m15/multimodal) — retrieval-augmented generation, the open-domain extension of extractive QA.",
          "[Decoding: greedy, beam, top-k, top-p](/learn/m20/decoding) — how translation and summarisation models turn distributions into text.",
        ],
      },
    ],
  },
};

export default lesson;
