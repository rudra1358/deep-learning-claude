import type { Lesson } from "@/lib/types";

const NGRAM = String.raw`# Chapter I of "Alice's Adventures in Wonderland" (Lewis Carroll, 1865; public domain), opening pages
TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled 'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.

'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)

Down, down, down. Would the fall never come to an end? 'I wonder how many miles I've fallen by this time?' she said aloud. 'I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think--' (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) '--yes, that's about the right distance--but then I wonder what Latitude or Longitude I've got to?' (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)

Presently she began again. 'I wonder if I shall fall right through the earth! How funny it'll seem to come out among the people that walk with their heads downward! The Antipathies, I think--' (she was rather glad there was no one listening, this time, as it didn't sound at all the right word) '--but I shall have to ask them what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?' (and she tried to curtsey as she spoke--fancy curtseying as you're falling through the air! Do you think you could manage it?) 'And what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it written up somewhere.'

Down, down, down. There was nothing else to do, so Alice soon began talking again. 'Dinah'll miss me very much to-night, I should think!' (Dinah was the cat.) 'I hope they'll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?' And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, 'Do cats eat bats? Do cats eat bats?' and sometimes, 'Do bats eat cats?' for, you see, as she couldn't answer either question, it didn't much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, 'Now, Dinah, tell me the truth: did you ever eat a bat?' when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over.

Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, 'Oh my ears and whiskers, how late it's getting!' She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof.

There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice's first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; 'and even if my head would go through,' thought poor Alice, 'it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin.' For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible.

There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, ('which certainly was not here before,' said Alice,) and round the neck of the bottle was a paper label, with the words 'DRINK ME' beautifully printed on it in large letters.
"""

import math
import re
from collections import Counter

text = " ".join(TEXT.lower().split())                    # lower-case, collapse whitespace
n = len(text)
train, dev, test = text[: int(0.8 * n)], text[int(0.8 * n): int(0.9 * n)], text[int(0.9 * n):]
chars = sorted(set(train))
print(f"characters: train {len(train)}, dev {len(dev)}, test {len(test)}; {len(chars)} symbols in train")
V = len(chars) + 1                                       # +1: one slot for any unseen symbol


# ---------------- word-level sparsity: why n-gram counts run out ----------------
def words(s):
    return re.findall(r"[a-z']+|[.,;:!?()-]", s)


wtr, wte = words(train), words(test)
seen = {k: set(zip(*[wtr[i:] for i in range(k)])) for k in (1, 2, 3)}
for k in (1, 2, 3):
    grams = list(zip(*[wte[i:] for i in range(k)]))
    unseen = sum(g not in seen[k] for g in grams) / len(grams)
    print(f"word {k}-grams of the test text never seen in training: {unseen:.1%}")


# ---------------- character n-gram models ----------------
class AddK:
    """P(w | h) = (c(h, w) + k) / (c(h) + k V)."""

    def __init__(self, seq, order, k):
        self.N, self.k = order, k
        self.c = Counter(seq[i:i + order] for i in range(len(seq) - order + 1))
        self.h = Counter(seq[i:i + order - 1] for i in range(len(seq) - order + 2))

    def prob(self, ctx, w):
        h = ctx[len(ctx) - self.N + 1:] if self.N > 1 else ""
        return (self.c[h + w] + self.k) / (self.h[h] + self.k * V)


class Interpolated:
    """Jelinek-Mercer: P(w | h) = sum_m lambda_m P_ML(w | last m-1 symbols), lambdas fitted by EM on dev."""

    def __init__(self, seq, order):
        self.N = order
        self.c = [Counter(seq[i:i + m] for i in range(len(seq) - m + 1)) for m in range(order + 1)]
        self.c[0] = Counter({"": len(seq)})
        self.lam = [1.0 / (order + 1)] * (order + 1)     # lambda_0 weights the uniform distribution

    def parts(self, ctx, w):
        out = [1.0 / V]
        for m in range(1, self.N + 1):
            h = ctx[len(ctx) - m + 1:] if m > 1 else ""
            out.append(self.c[m][h + w] / self.c[m - 1][h] if self.c[m - 1][h] else 0.0)
        return out

    def fit(self, held_out, iters=30):
        for _ in range(iters):                           # EM: expected responsibility of each component
            acc = [0.0] * (self.N + 1)
            for i in range(self.N, len(held_out)):
                p = [l * q for l, q in zip(self.lam, self.parts(held_out[:i], held_out[i]))]
                z = sum(p)
                acc = [a + q / z for a, q in zip(acc, p)]
            self.lam = [a / sum(acc) for a in acc]
        return self

    def prob(self, ctx, w):
        return sum(l * q for l, q in zip(self.lam, self.parts(ctx, w)))


class KneserNey:
    """Interpolated Kneser-Ney with absolute discount D; lower orders use continuation counts."""

    def __init__(self, seq, order, D=0.75):
        self.N, self.D = order, D
        self.c = {order: Counter(seq[i:i + order] for i in range(len(seq) - order + 1))}
        for m in range(order - 1, 0, -1):                # N1+(. g): in how many contexts does g follow?
            self.c[m] = Counter(g[1:] for g in self.c[m + 1])
        self.tot, self.types = {}, {}
        for m, cm in self.c.items():
            t, ty = Counter(), Counter()
            for g, x in cm.items():
                t[g[:-1]] += x
                ty[g[:-1]] += 1
            self.tot[m], self.types[m] = t, ty

    def prob(self, ctx, w, m=None):
        m = self.N if m is None else m
        h = ctx[len(ctx) - m + 1:] if m > 1 else ""
        lower = 1.0 / V if m == 1 else self.prob(ctx, w, m - 1)
        t = self.tot[m][h]
        if t == 0:
            return lower
        return max(self.c[m][h + w] - self.D, 0) / t + self.D * self.types[m][h] / t * lower


def evaluate(model, s, warm=""):
    """Average negative log-likelihood (nats/char) of s, each char conditioned on everything before it."""
    full = warm + s
    nll = -sum(math.log(model.prob(full[:i], full[i])) for i in range(len(warm), len(full)))
    return nll / len(s)


ctx = "alice was beginning to get very tire"
kn4 = KneserNey(train, 4)
total = sum(kn4.prob(ctx, ch) for ch in chars + ["#"])  # "#" never occurs: it stands for the unseen slot
best = sorted(chars, key=lambda ch: -kn4.prob(ctx, ch))[:3]
print(f"\nKN 4-gram after '...{ctx[-12:]}': probabilities sum to {total:.6f}; most likely next: {best}")

mle = AddK(train, 3, 0.0)                                # k = 0 is the unsmoothed maximum-likelihood model
zeros = sum(mle.h[(dev[-2:] + test)[i:i + 2]] == 0 or mle.prob((dev[-2:] + test)[:i + 2], test[i]) == 0
            for i in range(len(test)))
print(f"unsmoothed 3-gram MLE: {zeros} of {len(test)} test characters get probability 0 -> perplexity = infinity")

print("\nmodel                         dev PPL   test PPL   test bits/char")
results = {}
for order in (1, 2, 3, 4, 5):
    best_k = min([0.001, 0.01, 0.1, 0.5, 1.0], key=lambda k: evaluate(AddK(train, order, k), dev, train[-8:]))
    for name, model in [(f"add-k  (k={best_k})", AddK(train, order, best_k)),
                        ("interpolated (EM)", Interpolated(train, order).fit(dev)),
                        ("Kneser-Ney (D=0.75)", KneserNey(train, order))]:
        hd, ht = evaluate(model, dev, train[-8:]), evaluate(model, test, dev[-8:])
        results[(order, name.split()[0])] = ht
        print(f"{order}-gram {name:22s} {math.exp(hd):7.3f}   {math.exp(ht):7.3f}    {ht / math.log(2):.3f}")

# ---------------- sampling-based evaluation: what does the best n-gram model generate? ----------------
import random
rnd, kn5, out = random.Random(0), KneserNey(train, 5), "alice "
for _ in range(150):
    out += rnd.choices(chars, weights=[kn5.prob(out, ch) for ch in chars])[0]
print("\nsample from the KN 5-gram:", out)
grams = [out[i:i + 10] for i in range(len(out) - 9)]
print(f"10-character strings of the sample that occur verbatim in the training text: {sum(g in train for g in grams) / len(grams):.0%}")`;

const NGRAM_OUT = String.raw`characters: train 6008, dev 751, test 752; 37 symbols in train
word 1-grams of the test text never seen in training: 21.8%
word 2-grams of the test text never seen in training: 75.7%
word 3-grams of the test text never seen in training: 96.5%

KN 4-gram after '...et very tire': probabilities sum to 1.000000; most likely next: ['d', ' ', 'a']
unsmoothed 3-gram MLE: 115 of 752 test characters get probability 0 -> perplexity = infinity

model                         dev PPL   test PPL   test bits/char
1-gram add-k  (k=0.001)        17.672    19.070    4.253
1-gram interpolated (EM)       17.669    19.067    4.253
1-gram Kneser-Ney (D=0.75)     17.671    19.069    4.253
2-gram add-k  (k=0.1)          10.190    11.160    3.480
2-gram interpolated (EM)       10.163    11.217    3.488
2-gram Kneser-Ney (D=0.75)     10.194    11.199    3.485
3-gram add-k  (k=0.1)           7.541     9.449    3.240
3-gram interpolated (EM)        6.804     8.685    3.119
3-gram Kneser-Ney (D=0.75)      6.728     8.349    3.062
4-gram add-k  (k=0.1)           8.756    11.001    3.460
4-gram interpolated (EM)        6.187     8.059    3.011
4-gram Kneser-Ney (D=0.75)      5.682     7.090    2.826
5-gram add-k  (k=0.01)         12.180    14.976    3.905
5-gram interpolated (EM)        6.157     8.072    3.013
5-gram Kneser-Ney (D=0.75)      5.358     6.694    2.743

sample from the KN 5-gram: alice to do: onch goodly, and satiruns, i worl so mice went it soment on like to begurtundering up surr, ind ther the all think it at actu; be?' (abby gort,
10-character strings of the sample that occur verbatim in the training text: 4%`;

const TOKPPL = String.raw`# Chapter I of "Alice's Adventures in Wonderland" (Lewis Carroll, 1865; public domain), opening pages
TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled 'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.

'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)

Down, down, down. Would the fall never come to an end? 'I wonder how many miles I've fallen by this time?' she said aloud. 'I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think--' (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) '--yes, that's about the right distance--but then I wonder what Latitude or Longitude I've got to?' (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)

Presently she began again. 'I wonder if I shall fall right through the earth! How funny it'll seem to come out among the people that walk with their heads downward! The Antipathies, I think--' (she was rather glad there was no one listening, this time, as it didn't sound at all the right word) '--but I shall have to ask them what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?' (and she tried to curtsey as she spoke--fancy curtseying as you're falling through the air! Do you think you could manage it?) 'And what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it written up somewhere.'

Down, down, down. There was nothing else to do, so Alice soon began talking again. 'Dinah'll miss me very much to-night, I should think!' (Dinah was the cat.) 'I hope they'll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?' And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, 'Do cats eat bats? Do cats eat bats?' and sometimes, 'Do bats eat cats?' for, you see, as she couldn't answer either question, it didn't much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, 'Now, Dinah, tell me the truth: did you ever eat a bat?' when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over.

Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, 'Oh my ears and whiskers, how late it's getting!' She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof.

There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice's first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; 'and even if my head would go through,' thought poor Alice, 'it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin.' For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible.

There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, ('which certainly was not here before,' said Alice,) and round the neck of the bottle was a paper label, with the words 'DRINK ME' beautifully printed on it in large letters.
"""

import math
import re
from collections import Counter

text = " ".join(TEXT.lower().split())
n = len(text)
train, test = text[: int(0.8 * n)], text[int(0.9 * n):]   # same split as before (dev part unused here)


def pieces(s):                                            # pre-tokenise: a word keeps its leading space
    return re.findall(r" ?[^ ]+", s)


def learn_bpe(s, n_merges):
    vocab = Counter(tuple(p) for p in pieces(s))
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for word, c in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))    # most frequent pair (ties: deterministic)
        merges.append(best)
        new = Counter()
        for word, c in vocab.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1]); i += 2
                else:
                    out.append(word[i]); i += 1
            new[tuple(out)] += c
        vocab = new
    return merges


def encode(s, merges):
    rank = {m: r for r, m in enumerate(merges)}
    out = []
    for p in pieces(s):
        sym = list(p)
        while True:                                       # apply the earliest-learned merge first
            cand = [(rank[(a, b)], i) for i, (a, b) in enumerate(zip(sym, sym[1:])) if (a, b) in rank]
            if not cand:
                break
            _, i = min(cand)
            sym[i:i + 2] = [sym[i] + sym[i + 1]]
        out.extend(sym)
    return out


def kn_nll(train_toks, test_toks, order=3, D=0.75):
    """Total negative log-likelihood (nats) of test_toks under an interpolated Kneser-Ney model."""
    V = len(set(train_toks)) + 1
    c = {order: Counter(tuple(train_toks[i:i + order]) for i in range(len(train_toks) - order + 1))}
    for m in range(order - 1, 0, -1):
        c[m] = Counter(g[1:] for g in c[m + 1])
    tot = {m: Counter() for m in c}
    typ = {m: Counter() for m in c}
    for m in c:
        for g, x in c[m].items():
            tot[m][g[:-1]] += x
            typ[m][g[:-1]] += 1

    def prob(h, w, m):
        lower = 1.0 / V if m == 1 else prob(h[1:], w, m - 1)
        t = tot[m][h]
        return lower if t == 0 else max(c[m][h + (w,)] - D, 0) / t + D * typ[m][h] / t * lower

    seq = list(train_toks[-(order - 1):]) + list(test_toks)
    return -sum(math.log(prob(tuple(seq[i - order + 1:i]), seq[i], order)) for i in range(order - 1, len(seq)))


print(f"test text: {len(test)} characters")
print("tokenizer            vocab  test tokens  chars/token  PPL per token  bits per char")
for n_merges in (0, 50, 200, 600):
    merges = learn_bpe(train, n_merges)
    tr, te = encode(train, merges), encode(test, merges)
    nll = kn_nll(tr, te)
    name = "characters" if n_merges == 0 else f"BPE, {n_merges} merges"
    print(f"{name:20s} {len(set(tr)):5d}  {len(te):11d}  {len(test) / len(te):11.2f}  {math.exp(nll / len(te)):13.2f}"
          f"  {nll / len(test) / math.log(2):13.3f}")`;

const TOKPPL_OUT = String.raw`test text: 752 characters
tokenizer            vocab  test tokens  chars/token  PPL per token  bits per char
characters              37          752         1.00           8.39          3.068
BPE, 50 merges          86          498         1.51          20.69          2.894
BPE, 200 merges        222          347         2.17          68.72          2.816
BPE, 600 merges        456          287         2.62         201.34          2.921`;

const BLEU = String.raw`import math
from collections import Counter
from fractions import Fraction
from itertools import combinations


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidates, references, max_n=4):
    """Corpus BLEU (Papineni et al., 2002). candidates: list of token lists; references: list of lists of token lists."""
    match, total = [0] * max_n, [0] * max_n
    c_len = r_len = 0
    for cand, refs in zip(candidates, references):
        c_len += len(cand)
        r_len += min((abs(len(r) - len(cand)), len(r)) for r in refs)[1]   # closest reference length
        for n in range(1, max_n + 1):
            cand_counts = ngrams(cand, n)
            max_ref = Counter()
            for r in refs:                                # clip by the max count in any single reference
                max_ref |= ngrams(r, n)
            match[n - 1] += sum(min(c, max_ref[g]) for g, c in cand_counts.items())
            total[n - 1] += max(len(cand) - n + 1, 0)
    precisions = [m / t if t else 0.0 for m, t in zip(match, total)]
    if min(precisions) == 0:
        return 0.0, precisions, 1.0
    bp = 1.0 if c_len > r_len else math.exp(1 - r_len / c_len)
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n), precisions, bp


def rouge_n(cand, ref, n):
    c, r = ngrams(cand, n), ngrams(ref, n)
    overlap = sum((c & r).values())
    p, rec = overlap / max(sum(c.values()), 1), overlap / max(sum(r.values()), 1)
    return p, rec, (2 * p * rec / (p + rec) if p + rec else 0.0)


def lcs(a, b):
    """Length of the longest common subsequence, by dynamic programming in O(len(a) len(b))."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def rouge_l(cand, ref):
    m = lcs(cand, ref)
    p, r = m / len(cand), m / len(ref)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


T = str.split
# 1. the clipping example from the BLEU paper
cand = T("the the the the the the the")
refs = [T("the cat is on the mat"), T("there is a cat on the mat")]
_, p, _ = bleu([cand], [refs], max_n=1)
print(f"'the' x7: plain unigram precision {sum(w in refs[0] + refs[1] for w in cand)}/7, modified (clipped) {p[0]:.4f} = 2/7")

# 2. a near miss: one substituted word
cand, ref = T("the cat sat on the mat"), T("the cat is on the mat")
score4, p, bp = bleu([cand], [[ref]])
score2, _, _ = bleu([cand], [[ref]], max_n=2)
print("n-gram precisions:", [f"{x:.4f}" for x in p], f" BP {bp}")
print(f"sentence BLEU-4 = {score4:.4f} (no 4-gram matches),  BLEU-2 = {score2:.4f} = sqrt(5/6 * 3/5)")
for n in (1, 2):
    print(f"ROUGE-{n}: P/R/F1 = " + " / ".join(f"{x:.4f}" for x in rouge_n(cand, ref, n)))
print("ROUGE-L: P/R/F1 = " + " / ".join(f"{x:.4f}" for x in rouge_l(cand, ref)), f" (LCS = {lcs(cand, ref)}: 'the cat on the mat')")

# 3. the brevity penalty stops short, safe outputs from scoring well
score, p, bp = bleu([T("the cat")], [[ref]], max_n=2)
print(f"'the cat': precisions {p}, BP = exp(1 - 6/2) = {bp:.4f}, BLEU-2 = {score:.4f}")

# 4. corpus BLEU is NOT the mean of sentence BLEUs
C = [T("the cat sat on the mat"), T("a dog ran in the big park today")]
R = [[T("the cat is on the mat")], [T("a dog ran in the park today")]]
corpus, p, bp = bleu(C, R)
sent = [bleu([c], [r])[0] for c, r in zip(C, R)]
print(f"corpus BLEU-4 {corpus:.4f}  vs  mean of sentence BLEU-4 {sum(sent) / 2:.4f}  (sentences: {[round(s, 4) for s in sent]})")

# 5. independent checks: exact rational arithmetic for example 2, brute-force LCS over all subsequences
def is_subsequence(sub, seq):
    it = iter(seq)
    return all(tok in it for tok in sub)                  # "in" consumes the iterator up to the match


exact = math.sqrt(Fraction(5, 6) * Fraction(3, 5))
brute = max(k for k in range(len(cand) + 1) for sub in combinations(cand, k) if is_subsequence(sub, ref))
print(f"check: BLEU-2 from exact fractions {exact:.6f}, LCS by brute force {brute}")`;

const BLEU_OUT = String.raw`'the' x7: plain unigram precision 7/7, modified (clipped) 0.2857 = 2/7
n-gram precisions: ['0.8333', '0.6000', '0.2500', '0.0000']  BP 1.0
sentence BLEU-4 = 0.0000 (no 4-gram matches),  BLEU-2 = 0.7071 = sqrt(5/6 * 3/5)
ROUGE-1: P/R/F1 = 0.8333 / 0.8333 / 0.8333
ROUGE-2: P/R/F1 = 0.6000 / 0.6000 / 0.6000
ROUGE-L: P/R/F1 = 0.8333 / 0.8333 / 0.8333  (LCS = 5: 'the cat on the mat')
'the cat': precisions [1.0, 1.0], BP = exp(1 - 6/2) = 0.1353, BLEU-2 = 0.1353
corpus BLEU-4 0.4889  vs  mean of sentence BLEU-4 0.2973  (sentences: [0.0, 0.5946])
check: BLEU-2 from exact fractions 0.707107, LCS by brute force 5`;

const NEURAL = String.raw`# Chapter I of "Alice's Adventures in Wonderland" (Lewis Carroll, 1865; public domain), opening pages
TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled 'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.

'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)

Down, down, down. Would the fall never come to an end? 'I wonder how many miles I've fallen by this time?' she said aloud. 'I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think--' (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) '--yes, that's about the right distance--but then I wonder what Latitude or Longitude I've got to?' (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)

Presently she began again. 'I wonder if I shall fall right through the earth! How funny it'll seem to come out among the people that walk with their heads downward! The Antipathies, I think--' (she was rather glad there was no one listening, this time, as it didn't sound at all the right word) '--but I shall have to ask them what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?' (and she tried to curtsey as she spoke--fancy curtseying as you're falling through the air! Do you think you could manage it?) 'And what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it written up somewhere.'

Down, down, down. There was nothing else to do, so Alice soon began talking again. 'Dinah'll miss me very much to-night, I should think!' (Dinah was the cat.) 'I hope they'll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?' And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, 'Do cats eat bats? Do cats eat bats?' and sometimes, 'Do bats eat cats?' for, you see, as she couldn't answer either question, it didn't much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, 'Now, Dinah, tell me the truth: did you ever eat a bat?' when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over.

Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, 'Oh my ears and whiskers, how late it's getting!' She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof.

There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice's first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; 'and even if my head would go through,' thought poor Alice, 'it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin.' For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible.

There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, ('which certainly was not here before,' said Alice,) and round the neck of the bottle was a paper label, with the words 'DRINK ME' beautifully printed on it in large letters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)
text = " ".join(TEXT.lower().split())
n = len(text)
train, dev, test = text[: int(0.8 * n)], text[int(0.8 * n): int(0.9 * n)], text[int(0.9 * n):]
chars = sorted(set(train)) + ["<unk>"]
stoi = {ch: i for i, ch in enumerate(chars)}
V = len(chars)
enc = lambda s: torch.tensor([stoi.get(ch, V - 1) for ch in s])
CTX = 8                                                   # the model sees the previous 8 characters


def windows(prev, s):
    """All (context, next-char) pairs for s; the context may reach back into prev."""
    full = enc(prev[-CTX:] + s)
    X = torch.stack([full[i:i + CTX] for i in range(len(full) - CTX)])
    return X, full[CTX:]


class NPLM(nn.Module):
    """Bengio et al. (2003): concatenate the context embeddings, one tanh hidden layer, softmax over V."""

    def __init__(self, d=24, h=128, p_drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.hidden = nn.Linear(CTX * d, h)
        self.out = nn.Linear(h, V)
        self.direct = nn.Linear(CTX * d, V, bias=False)  # the paper's optional direct connections
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):                                 # x: (B, CTX) token ids
        e = self.drop(self.emb(x).flatten(1))             # (B, CTX*d)
        return self.out(self.drop(torch.tanh(self.hidden(e)))) + self.direct(e)


class GRULM(nn.Module):
    """A recurrent LM reading the same 8-character window (its state could carry further; kept equal for fairness)."""

    def __init__(self, d=24, h=128, p_drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.GRU(d, h, batch_first=True)
        self.out = nn.Linear(h, V)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        hs, _ = self.rnn(self.drop(self.emb(x)))
        return self.out(self.drop(hs[:, -1]))


Xtr, Ytr = windows(train[:CTX], train[CTX:])
Xdv, Ydv = windows(train, dev)
Xte, Yte = windows(dev, test)


def nats(model, X, Y):
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(model(X), Y).item()


for Model in (NPLM, GRULM):
    torch.manual_seed(0)
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.05)
    best = (float("inf"), None, 0)
    for epoch in range(1, 21):
        model.train()
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(Xtr), 128):
            b = perm[k:k + 128]
            loss = F.cross_entropy(model(Xtr[b]), Ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        d = nats(model, Xdv, Ydv)
        if d < best[0]:                                   # early stopping on the dev set
            best = (d, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    t = nats(model, Xte, Yte)
    params = sum(p.numel() for p in model.parameters())
    print(f"{Model.__name__:5s} {params:6d} params  best epoch {best[2]:2d}  train {nats(model, Xtr, Ytr):.3f}  "
          f"dev {best[0]:.3f}  test {t:.3f} nats/char  ->  test PPL {math.exp(t):.3f}, {t / math.log(2):.3f} bits/char")`;

const NEURAL_OUT = String.raw`NPLM   37814 params  best epoch 20  train 1.311  dev 2.019  test 2.277 nats/char  ->  test PPL 9.746, 3.285 bits/char
GRULM  64950 params  best epoch 11  train 1.282  dev 1.869  test 2.078 nats/char  ->  test PPL 7.987, 2.998 bits/char`;

const NPLM_KERAS = String.raw`import numpy as np
import tensorflow as tf

V, CTX, d, h = 38, 8, 24, 128          # vocabulary, context length, embedding size, hidden units

ids = tf.keras.Input(shape=(CTX,), dtype="int32")
e = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(V, d)(ids))      # concatenated context embeddings
hidden = tf.keras.layers.Dense(h, activation="tanh")(tf.keras.layers.Dropout(0.3)(e))
logits = tf.keras.layers.Add()([tf.keras.layers.Dense(V)(tf.keras.layers.Dropout(0.3)(hidden)),
                                tf.keras.layers.Dense(V, use_bias=False)(e)])   # direct connections
model = tf.keras.Model(ids, logits)
model.compile(optimizer=tf.keras.optimizers.AdamW(5e-3, weight_decay=0.05),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# X: (N, 8) context ids, y: (N,) next-character ids (built exactly as in the PyTorch version)
# model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=20, batch_size=128,
#           callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5)])
# test_nats = model.evaluate(X_test, y_test); perplexity = np.exp(test_nats); bpc = test_nats / np.log(2)`;

const lesson: Lesson = {
  id: "m13/language-modeling",
  title: "Language Modelling and Perplexity",
  summary:
    "What a language model is (the chain rule splits a text's probability into next-token guesses), n-gram models built from counts, why they run out of data, and how smoothing fixes it (add-k, interpolation, backoff, Kneser–Ney). Then perplexity and bits per character, and why perplexities from different tokenizers cannot be compared — all measured on real text. Finally, neural LMs from Bengio's NPLM to RNNs and Transformers, and how generated text is judged with BLEU, ROUGE (both built from scratch) and samples.",
  minutes: 95,
  level: "intermediate",
  prerequisites: ["m02/information-theory", "m02/probability", "m02/mle-map", "m13/text-preprocessing", "m13/word-embeddings"],
  next: ["m11/training-inference", "m20/decoding", "m20/llm-pretraining", "m13/nlp-tasks"],
  terms: ["language-model", "n-gram", "markov-assumption", "smoothing", "kneser-ney", "perplexity", "bits-per-character", "bleu", "rouge", "brevity-penalty"],
  sections: {
    intro: [
      { t: "p", text: "A **language model** (LM) gives a probability to any piece of text — or, put another way, predicts the next token from everything before it. That one skill is hugely useful. A speech recogniser uses it to prefer \"recognise speech\" over the similar-sounding \"wreck a nice beach\"; a phone keyboard uses it to suggest your next word; a translator uses it to sound fluent. And GPT *is* a language model, trained on trillions of tokens, whose next-token guesses are sampled to write text." },
      { t: "p", text: "Language modelling is older than deep learning. Shannon estimated the entropy of English with n-gram statistics and human guessing games (1948; \"Prediction and Entropy of Printed English\", 1951). From the 1980s to the 2010s, **n-gram models** with careful **smoothing** powered speech recognition and statistical translation. Bengio et al. (2003, \"A Neural Probabilistic Language Model\") swapped the count table for a neural network with word embeddings; recurrent LMs (Mikolov et al., 2010) removed the fixed context; Transformers scaled it up." },
      { t: "p", text: "Two questions run through this lesson: **how to estimate** next-token probabilities when most word sequences were never seen, and **how to evaluate** a model — by perplexity on unseen text and, for generated text, by BLEU, ROUGE and reading samples. Every number is measured on real public-domain text: the opening of *Alice's Adventures in Wonderland* (Lewis Carroll, 1865)." },
      { t: "callout", kind: "key", text: String.raw`Perplexity is the model's average surprise on unseen text, turned back into a count: $\mathrm{PPL} = \exp\big(-\tfrac1T\sum_t \log p(w_t\mid w_{<t})\big)$. Lower is better — but only between models that cut the text into the same tokens.` },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) — cross-entropy as average negative log-probability, nats vs bits, and the definition of perplexity, which we use here without re-deriving.",
        String.raw`[Probability](/learn/m02/probability) — conditional probability and the chain rule $P(A,B) = P(A)P(B\mid A)$.`,
        "[Maximum likelihood](/learn/m02/mle-map) — why relative frequencies are the maximum-likelihood estimate of a categorical distribution.",
        "[Text Preprocessing and Subword Tokenization](/learn/m13/text-preprocessing) — what tokens are and how BPE builds them; perplexity depends on that choice.",
        "[Word Embeddings](/learn/m13/word-embeddings) — the embedding table that neural LMs use as their first layer.",
        "Helpful for the neural part: [Vanilla RNNs](/learn/m09/rnn-basics) and [Transformer training](/learn/m11/training-inference).",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "A game of guessing the next letter", text: "Cover a page of a novel and reveal it one letter at a time. Before each letter, write down your betting odds for all 27 symbols. After \"Alice was beginning to get very tire\" you bet almost everything on \"d\". At the start of a new word you spread your bets. A language model plays this game, and perplexity scores how well: a perplexity of 6.7 means that, on average, it was as unsure as someone picking evenly among 6.7 symbols." },
      { t: "p", text: "The simplest player remembers only the last few symbols — an **n-gram model**: \"after 'tir' I have seen 'e' 3 times and 'i' once.\" It learns by counting. The trouble: most longer histories were never seen (in our test text, 96.5% of word trigrams never occur in training). An unseen event gets count zero, so probability zero, so infinite surprise. **Smoothing** shaves a little probability off seen events and gives it to unseen ones, ideally guided by shorter histories that *were* seen." },
      { t: "p", text: "A **neural** player turns the context into vectors, so a history it has never seen lands near similar ones it has, and the prediction carries over. With enough data this wins easily. With only 6,000 training characters, as you will see, a well-smoothed count model can still beat it." },
    ],
    mechanism: [
      { t: "steps", items: [
        { title: "Choose the tokens", text: "Characters, words or subwords (BPE). Every later number — counts, probabilities, perplexity — is per token, so this choice is part of the model." },
        { title: "Split with the chain rule", text: "This is exact, with no assumption: a sequence's probability is the product of its next-token probabilities.", tex: String.raw`P(w_1,\dots,w_T) = \prod_{t=1}^{T} P(w_t\mid w_1,\dots,w_{t-1})` },
        { title: "Limit the history (n-gram models)", text: "Assume each token depends only on the previous $n-1$ tokens (the **Markov assumption**). Pad the start with $n-1$ start symbols and end each sentence with an end symbol, so probabilities over whole sentences sum to one.", tex: String.raw`P(w_t\mid w_{<t}) \approx P(w_t\mid w_{t-n+1},\dots,w_{t-1})` },
        { title: "Count", text: "Estimate each probability from training counts (maximum likelihood): the n-gram's count divided by its history's count." },
        { title: "Smooth", text: "Save probability for unseen n-grams: add pseudo-counts (add-k), mix in shorter-history estimates (interpolation), fall back to them when the n-gram is unseen (backoff), or — best — subtract a fixed discount and share it out using continuation counts (Kneser–Ney). Tune the settings on a **development** set." },
        { title: "Evaluate on held-out text", text: "On a test set never used for training or tuning, average the negative log-probability per token and exponentiate: that is perplexity. Add bits per character when tokenisations differ." },
        { title: "Or replace counting with a network", text: "A neural LM turns the context into a vector (joined embeddings → MLP; an RNN state; Transformer attention) and outputs a softmax over the vocabulary, trained by gradient descent on the same objective: average negative log-likelihood." },
        { title: "Generate", text: String.raw`Sample $w_t \sim P(\cdot\mid w_{<t})$, append it, repeat. Decoding strategies (greedy, temperature, top-k, top-p) change what gets picked — see [Decoding](/learn/m20/decoding).` },
      ] },
    ],
    math: [
      { t: "h", text: "Maximum-likelihood n-gram estimates" },
      { t: "p", text: String.raw`Let $c(\cdot)$ be a count in the training text and $h = w_{t-n+1}^{t-1}$ the history. Maximise the log-likelihood $\sum_t \log P(w_t\mid h_t)$ for each history separately, with the rule $\sum_w P(w\mid h) = 1$ (one Lagrange multiplier per history). The answer is plain relative frequencies:` },
      { t: "math", tex: String.raw`P_{\text{ML}}(w\mid h) = \frac{c(h, w)}{c(h)}, \qquad \text{e.g. bigram } P(w_t\mid w_{t-1}) = \frac{c(w_{t-1}w_t)}{c(w_{t-1})}` },
      { t: "p", text: String.raw`A vocabulary of $V$ tokens has $V^n$ possible n-grams — $10^{15}$ trigrams for $V = 10^5$ — but a corpus of $N$ tokens holds at most $N$ of them. Most n-grams are unseen and get $P_{\text{ML}} = 0$. One zero in a test sentence makes its probability 0 and the perplexity infinite. This is **sparsity**, and it gets exponentially worse as $n$ grows.` },
      { t: "h", text: "Smoothing" },
      { t: "p", text: String.raw`**Add-k (Laplace when $k=1$)** pretends every n-gram was seen $k$ extra times: $P_{\text{add-}k}(w\mid h) = \frac{c(h,w)+k}{c(h)+kV}$. It is simple, but with a large $V$ it hands far too much probability to unseen events, and it treats every unseen continuation as equally likely.` },
      { t: "p", text: String.raw`**Linear interpolation** (Jelinek–Mercer) blends the orders, with weights $\lambda_m \ge 0$ that sum to 1, fitted on held-out data (by grid search or the EM algorithm):` },
      { t: "math", tex: String.raw`P_{\text{int}}(w\mid w_{t-2}w_{t-1}) = \lambda_3 P_{\text{ML}}(w\mid w_{t-2}w_{t-1}) + \lambda_2 P_{\text{ML}}(w\mid w_{t-1}) + \lambda_1 P_{\text{ML}}(w) + \lambda_0\frac{1}{V}` },
      { t: "p", text: String.raw`**Backoff** (Katz, 1987) uses the longest history whose n-gram was seen, discounted, and otherwise falls back to a scaled shorter-history estimate; the scale $\alpha(h)$ makes everything sum to one. "**Stupid backoff**" (Brants et al., 2007) skips that normalisation (score $= 0.4\times$ the shorter-history score). It is not a probability, so it has no perplexity, but it is cheap enough for web-scale counts in translation.` },
      { t: "p", text: String.raw`**Kneser–Ney** (Kneser and Ney, 1995; Chen and Goodman's 1998 "modified" version is the strongest classical method) combines two ideas. *Absolute discounting*: subtract a fixed $D\approx0.75$ from every seen count — n-grams seen $c$ times show up about $c - 0.75$ times in held-out text. *Continuation probability*: the shorter-history distribution should ask not "how common is $w$?" but "after how many different words does $w$ appear?". Classic example: "Francisco" is common but nearly always follows "San", so after a new history it deserves little probability. With $N_{1+}(\bullet\, w) = |\{v : c(v,w) > 0\}|$ (the number of different words seen before $w$):` },
      { t: "math", tex: String.raw`P_{\text{KN}}(w\mid h) = \frac{\max(c(h,w) - D,\ 0)}{c(h)} + \underbrace{\frac{D\,N_{1+}(h\,\bullet)}{c(h)}}_{\text{mass freed by discounting}}\, P_{\text{KN}}(w\mid h'), \qquad P_{\text{cont}}(w) = \frac{N_{1+}(\bullet\, w)}{N_{1+}(\bullet\,\bullet)}`, caption: String.raw`$h'$ is the history minus its oldest token; $N_{1+}(h\,\bullet)$ is the number of different tokens seen after $h$. Shorter histories use continuation counts instead of raw counts. The recursion ends at $P_{\text{cont}}$ (here mixed with a uniform $1/V$ so unseen symbols keep a little probability).` },
      { t: "h", text: "Perplexity, cross-entropy and bits per character" },
      { t: "p", text: String.raw`For a test sequence of $T$ tokens, the model's **cross-entropy** (in nats per token) and **perplexity** are (see [Information Theory](/learn/m02/information-theory)):` },
      { t: "math", tex: String.raw`H = -\frac1T\sum_{t=1}^{T}\ln p(w_t\mid w_{<t}),\qquad \mathrm{PPL} = e^{H} = P(w_1,\dots,w_T)^{-1/T} = 2^{H/\ln 2}` },
      { t: "p", text: String.raw`In plain words: perplexity is one over the geometric mean of the probabilities given to the true tokens. So it depends on what a token is. To compare models with different tokenizers, convert to a per-character (or per-byte) number. If the test text has $C$ characters cut into $T$ tokens:` },
      { t: "math", tex: String.raw`\text{bits per character} = \frac{T\,H_{\text{token}}}{C\,\ln 2}, \qquad \mathrm{PPL}_{\text{token}} = 2^{\text{bpc}\cdot C/T}` },
      { t: "p", text: String.raw`So the same model quality gives a *larger* per-token perplexity when tokens are longer. Measured below on our 752-character test text: a character model with $\mathrm{PPL} = 8.39$ has $\log_2 8.39 = 3.068$ bpc. A BPE model with 600 merges cuts the same text into 287 tokens and has per-token $\mathrm{PPL} = 201.34$ — yet $287\log_2(201.34)/752 = 2.921$ bpc, so it is the *better* model.` },
      { t: "h", text: "Worked example: a bigram model by hand" },
      { t: "p", text: String.raw`Training sentences (with start/end symbols): "<s> the cat sat </s>", "<s> the cat ran </s>", "<s> the dog sat </s>". Tokens we predict: {the, cat, dog, sat, ran, </s>}, so $V = 6$. Test sentence: "<s> the dog ran </s>" (4 predictions).` },
      { t: "steps", items: [
        { title: "Counts", text: String.raw`$c(\text{<s>}) = 3$, $c(\text{the}) = 3$, $c(\text{dog}) = 1$, $c(\text{ran}) = 1$; $c(\text{<s> the}) = 3$, $c(\text{the dog}) = 1$, $c(\text{dog ran}) = 0$, $c(\text{ran </s>}) = 1$.` },
        { title: "MLE", tex: String.raw`P = 1 \cdot \tfrac13 \cdot \tfrac{0}{1}\cdot 1 = 0 \ \Rightarrow\ \mathrm{PPL} = \infty \quad(\text{"dog ran" was never seen})` },
        { title: "Add-1", tex: String.raw`\tfrac{3+1}{3+6}=0.444444,\ \ \tfrac{1+1}{3+6}=0.222222,\ \ \tfrac{0+1}{1+6}=0.142857,\ \ \tfrac{1+1}{1+6}=0.285714;\ \ \text{product } 0.004031` },
        { title: "Perplexity (add-1)", tex: String.raw`H = -\tfrac14\ln 0.004031 = 1.378420 \text{ nats} = 1.988640 \text{ bits};\quad \mathrm{PPL} = e^{1.378420} = 3.968627` },
        { title: "Interpolation instead", text: String.raw`$\lambda_2 = 0.7$ on the bigram MLE, $\lambda_1 = 0.3$ on the unigram MLE (12 predicted tokens: the 3, </s> 3, cat 2, sat 2, dog 1, ran 1).`, tex: String.raw`P(\text{ran}\mid\text{dog}) = 0.7\cdot 0 + 0.3\cdot\tfrac{1}{12} = 0.025;\ \ \text{others } 0.775,\ 0.258333,\ 0.775;\ \ \mathrm{PPL} = 4.006997` },
        { title: "Continuation counts (Kneser–Ney's lower order)", tex: String.raw`N_{1+}(\bullet\,\text{sat}) = 2\ (\text{cat sat, dog sat}),\quad N_{1+}(\bullet\,\text{the}) = 1,\quad N_{1+}(\bullet\,\bullet) = 8 \ \Rightarrow\ P_{\text{cont}}(\text{sat}) = 0.25,\ P_{\text{cont}}(\text{the}) = 0.125` },
      ] },
      { t: "p", text: "\"the\" is predicted more often than \"sat\" (3 vs 2 times), but it only ever follows <s>. So after a new history it gets half the probability of \"sat\" — exactly the Kneser–Ney intuition." },
      { t: "h", text: "Neural language models" },
      { t: "p", text: String.raw`Bengio et al. (2003) map each of the $n-1$ context words to a learned vector $C(w)\in\mathbb{R}^m$ (one shared embedding table), join them into one vector $x$, and compute` },
      { t: "math", tex: String.raw`y = b + Wx + U\tanh(d + Hx), \qquad P(w_t = i\mid \text{context}) = \frac{e^{y_i}}{\sum_j e^{y_j}}`, caption: "$H, d$: hidden layer; $U, b$: output layer; $W$: optional direct links from the embeddings to the output. Trained by maximising log-likelihood, which is the same as minimising cross-entropy." },
      { t: "p", text: String.raw`Similar words get similar embeddings, so an unseen history still gets a sensible prediction — what count tables lack. The cost is a softmax over the whole vocabulary at every step. **RNN LMs** replace the fixed window with a state $h_t = f(h_{t-1}, x_t)$ that can in principle carry the whole history ([Vanilla RNNs](/learn/m09/rnn-basics), [LSTM](/learn/m09/lstm)). **Transformer LMs** attend directly to every earlier token under a causal mask ([Training and Generation](/learn/m11/training-inference)). All three use the same objective, $\frac1T\sum_t -\log p_\theta(w_t\mid w_{<t})$, and the same perplexity.` },
      { t: "h", text: "BLEU and ROUGE" },
      { t: "p", text: String.raw`Perplexity needs the model's probabilities. When all we have is generated text (a translation, a summary), we compare it with human references. **BLEU** (Papineni et al., 2002) focuses on precision. For n-gram order $n$, the *modified* precision caps ("clips") each candidate n-gram count at its highest count in any single reference, summed over the whole test corpus $\mathcal{C}$:` },
      { t: "math", tex: String.raw`p_n = \frac{\sum_{C\in\mathcal{C}}\sum_{g\in C}\min\big(c_C(g),\ \max_{R}c_R(g)\big)}{\sum_{C\in\mathcal{C}}\sum_{g\in C} c_C(g)},\qquad \mathrm{BP} = \begin{cases}1 & c > r\\ e^{1-r/c} & c\le r\end{cases},\qquad \mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\sum_{n=1}^{4}\tfrac14\log p_n\Big)`, caption: "$c$: total candidate length; $r$: effective reference length (summed over sentences, the reference length closest to the candidate's). The brevity penalty BP stands in for recall, which is poorly defined with several references." },
      { t: "p", text: String.raw`**ROUGE** (Lin, 2004) focuses on recall and was built for summaries. ROUGE-N counts shared n-grams relative to the reference, $R_N = \frac{\sum_g \min(c_C(g), c_R(g))}{\sum_g c_R(g)}$ (precision and F1 work the same way). ROUGE-L uses the longest common subsequence: $R_{\text{lcs}} = \frac{\mathrm{LCS}(C,R)}{|R|}$, $P_{\text{lcs}} = \frac{\mathrm{LCS}(C,R)}{|C|}$, $F = \frac{(1+\beta^2)RP}{R+\beta^2P}$; common tools report $\beta = 1$.` },
      { t: "p", text: String.raw`Worked example (checked by the code below): candidate "the cat sat on the mat", reference "the cat is on the mat". Clipped matches: unigrams 5/6, bigrams 3/5 (the cat, on the, the mat), trigrams 1/4, 4-grams 0/3. $c = r = 6$, so BP = 1. BLEU-4 $= 0$ because $p_4 = 0$; BLEU-2 $= \sqrt{5/6\cdot3/5} = \sqrt{0.5} = 0.7071$. ROUGE-1 = 0.8333, ROUGE-2 = 0.6, LCS = 5 ("the cat on the mat"), so ROUGE-L = 0.8333. The two-word candidate "the cat" has perfect precisions but $\mathrm{BP} = e^{1-6/2} = 0.1353$.` },
    ],
    visual: [
      { t: "p", text: String.raw`The first simulator runs a real character-level Transformer LM trained on the same *Alice* text. Each bar chart of next-character probabilities is one $p(w_t\mid w_{<t})$. The negative log of the bar for the character that actually comes next is that position's share of the loss; average it over held-out text and exponentiate, and you get the perplexity.` },
      { t: "sim", id: "transformer", caption: "A trained character-level Transformer LM: next-character distributions, greedy decoding and sampling with temperature." },
      { t: "list", items: [
        "Type the start of a sentence from the chapter, e.g. \"alice was beginning to get very tire\", and read the probability of \"d\". Then type a phrase not in the text and watch the distribution spread out (high entropy).",
        "Generate greedily, then sample at temperature 1 and 2. Greedy text from a model trained on 6,800 characters repeats training sentences — memorisation that *training* perplexity would hide.",
        "For reference: the exported checkpoint's held-out loss is 2.515 nats/char (3.63 bits/char) on the last 10% of the text — the same characters as our test set. Our 5-gram Kneser–Ney reaches 2.743 bits/char there: with this little data, counting wins.",
      ] },
      { t: "p", text: "The second simulator learns BPE merges. Each merge makes tokens longer and the vocabulary bigger — and, as the code section measures, changes a model's perplexity even when it predicts the *text* no better." },
      { t: "sim", id: "tokenizer", caption: "BPE merges: watch the number of tokens for the same text fall as merges are learned." },
      { t: "list", items: [
        "Raise the number of merges and note how many tokens the same text needs. Per-token perplexity averages over this many predictions, so it is not comparable across settings.",
      ] },
    ],
    scratch: [
      { t: "p", text: "**1. n-gram language models from scratch** on the opening of *Alice* (7.5k characters): 80% train, 10% development (to tune $k$ and the interpolation weights), 10% test. The script measures word sparsity, checks that Kneser–Ney sums to one, shows the unsmoothed model's zeros, compares add-k, interpolation (EM weights) and interpolated Kneser–Ney for character orders 1–5, and samples from the best model. Pure Python, about 2 seconds." },
      { t: "code", lang: "python", title: "ngram_lm.py", code: NGRAM, output: NGRAM_OUT },
      { t: "list", items: [
        "**Sparsity**: 21.8% of test word tokens never appear in training, and neither do 75.7% of word bigrams and 96.5% of word trigrams. A word trigram model would be backing off almost everywhere, which is why the comparison uses characters.",
        "**Zeros are fatal**: the unsmoothed character trigram gives probability 0 to 115 of 752 test characters.",
        "**Add-k gets worse with order**: its best test perplexity is at $n=3$ (9.45), falling to 14.98 at $n=5$, because the pseudo-counts drown the few real counts of long histories.",
        "**Interpolation** improves up to $n=4$ (8.06), then levels off. **Kneser–Ney** is best at every order ≥ 3 and still improving at $n=5$: test perplexity 6.69, or 2.743 bits per character.",
        "The sample looks locally like English (\"went it soment on like to\") but means nothing. Only 4% of its 10-character strings occur in the training text, so it is not just copying.",
      ] },
      { t: "p", text: "**2. Perplexity depends on the tokenizer.** Same test text, same kind of model (interpolated Kneser–Ney trigram); only the tokens change, from characters to BPE with 50, 200 and 600 merges learned on the training text." },
      { t: "code", lang: "python", title: "perplexity_vs_tokenizer.py", code: TOKPPL, output: TOKPPL_OUT },
      { t: "p", text: "Per-token perplexity grows 24-fold, from 8.39 to 201.34, while bits per character stay between 2.82 and 3.07. By perplexity the 600-merge model looks 24 times *worse* than the character model; per character it is *better* (2.921 vs 3.068). The bpc column shows the real trade-off: longer tokens let the trigram see further back (best at 200 merges, 2.816 bpc) until the vocabulary is so big that counts turn sparse again (600 merges)." },
      { t: "p", text: "**3. BLEU and ROUGE from scratch**, with the clipping example from the BLEU paper, a hand-checked near miss, the brevity penalty, corpus vs sentence scores, and two independent checks (exact fraction arithmetic, brute-force LCS over all subsequences). No reference library is installed here, so these hand computations validate the code." },
      { t: "code", lang: "python", title: "bleu_rouge.py", code: BLEU, output: BLEU_OUT },
      { t: "list", items: [
        "Clipping stops \"the the the the the the the\" from getting precision 7/7: \"the\" occurs at most twice in any one reference, so it gets 2/7.",
        "One wrong word in a six-word sentence removes every 4-gram match, so sentence-level BLEU-4 drops to exactly 0. Sentence-level BLEU needs smoothing (several variants exist); BLEU was designed as a *corpus-level* metric.",
        "Corpus BLEU (0.4889) pools n-gram counts before the geometric mean, so it is not the mean of sentence scores (0.2973).",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Two small neural character LMs on the same split, both reading the previous 8 characters: Bengio et al.'s NPLM (concatenated embeddings → tanh layer → softmax, with direct links) and a GRU. Both use dropout, AdamW with weight decay, and early stopping on the development set. The test set is scored once, with the best checkpoint." },
          { t: "code", lang: "python", title: "neural_lm.py", code: NEURAL, output: NEURAL_OUT },
          { t: "table", head: ["Model (test = last 752 characters)", "Test PPL / char", "Bits per char"], rows: [
            ["Unigram (any smoothing)", "19.07", "4.253"],
            ["Bigram, Kneser–Ney", "11.20", "3.485"],
            ["Trigram, Kneser–Ney", "8.35", "3.062"],
            ["NPLM, 8-char context, 37,814 parameters", "9.75", "3.285"],
            ["GRU, 8-char context, 64,950 parameters", "7.99", "2.998"],
            ["5-gram, interpolated (EM)", "8.07", "3.013"],
            ["5-gram, Kneser–Ney", "6.69", "2.743"],
          ] },
          { t: "p", text: "The neural models fit the training text (1.31 and 1.28 nats/char) far better than the test text (2.28 and 2.08): with 6,000 training characters they overfit within a few epochs despite dropout and weight decay. The GRU beats the trigram and the interpolated 5-gram, but Kneser–Ney 5-gram is still best. That matches history: well-smoothed n-grams stayed competitive until neural LMs got much more data. On millions of words, RNN and Transformer LMs push perplexity far below any n-gram model." },
          { t: "callout", kind: "note", text: "`F.cross_entropy` returns the mean negative log-likelihood in **nats**, so `exp(loss)` is the perplexity and `loss / ln 2` is bits per character. Evaluate with `model.eval()` (dropout off), and never on data used for early stopping." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The NPLM in Keras: an `Embedding` layer flattened into the joined context vector, a tanh `Dense` layer, an output layer plus direct links, trained with sparse categorical cross-entropy on logits and early stopping on the development set." },
          { t: "code", lang: "python", title: "nplm_keras.py (not executed in CI)", code: NPLM_KERAS, output: "Builds the model; with the (context, next character) arrays it prints the training and validation cross-entropy (nats per character) for each epoch, restores the best weights, and evaluate() returns the test cross-entropy, from which perplexity = exp(loss) and bits/char = loss / ln 2." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Model", "Context", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["n-gram + add-k", "$n-1$ tokens", "Very simple", "Poor estimates, worse with larger $n$", "Teaching, text classification (naive Bayes)"],
        ["n-gram + interpolation / Katz backoff", "$n-1$ tokens", "Uses all orders", "Ignores how many different contexts a word follows", "Classic speech recognition"],
        ["Interpolated (modified) Kneser–Ney", "$n-1$ tokens", "Best count-based estimates; fast lookups", "No generalisation across similar words; huge tables", "Speech/MT LMs before ~2015, KenLM"],
        ["Stupid backoff", "$n-1$ tokens", "Scales to trillions of tokens", "Not normalised (no perplexity)", "Web-scale MT (Brants et al., 2007)"],
        ["Neural n-gram (NPLM)", "Fixed window", "Embeddings generalise to unseen histories", "Fixed window; softmax cost", "First neural LMs"],
        ["RNN / LSTM LM", "Unbounded (in principle)", "Long dependencies, compact", "Sequential training; gradients through time", "2010–2017 state of the art — [LSTM](/learn/m09/lstm)"],
        ["Transformer LM (decoder-only)", "Context window of thousands of tokens", "Parallel training, scales with data and compute", "Quadratic attention cost, fixed window", "GPT-style LLMs — [Training](/learn/m11/training-inference)"],
        ["Masked LM (BERT)", "Both sides", "Bidirectional representations", "Not a generative LM; no perplexity in the usual sense", "Encoders for understanding tasks"],
      ] },
      { t: "h", text: "Evaluating generated text: the options" },
      { t: "table", head: ["Metric", "Measures", "Strengths", "Weaknesses"], rows: [
        ["Perplexity / bpc", "Probability of held-out human text", "Cheap, smooth, objective", "Needs model probabilities and the same tokenizer; says nothing about generated text"],
        ["BLEU", "n-gram precision vs references + brevity penalty", "Standard for MT, corpus-level", "Ignores synonyms and paraphrase; unstable for single sentences; sensitive to tokenization (Post, 2018, motivated sacreBLEU)"],
        ["ROUGE-N / ROUGE-L", "n-gram / LCS recall vs references", "Standard for summarisation", "Rewards copying; ignores meaning and factuality"],
        ["Embedding-based (e.g. BERTScore)", "Similarity of contextual embeddings", "Credits paraphrase", "Depends on the scoring model"],
        ["Sample statistics", "distinct-n, repetition, copy rate from training data", "Detects degeneration and memorisation", "Not a quality score on its own"],
        ["Human / model-as-judge", "Fluency, adequacy, preference", "Closest to what we care about", "Expensive (human) or biased (model judges)"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Speech recognition and OCR**: the decoder combines an acoustic (or visual) score with an LM score to choose between similar-sounding transcriptions.",
        "**Machine translation**: statistical MT combined a translation model with an n-gram LM; neural MT decoders are conditional language models, judged with BLEU.",
        "**Text entry**: next-word prediction and autocorrect on phone keyboards.",
        "**Large language models**: GPT-style models are Transformer LMs trained on next-token prediction; pretraining loss/perplexity is their main training metric — see [Pretraining LLMs](/learn/m20/llm-pretraining).",
        "**Compression**: an LM with cross-entropy $H$ bits per symbol lets an arithmetic coder use about $H$ bits per symbol; bits per character is literally a compression rate.",
        "**Data filtering and detection**: perplexity under a reference LM filters low-quality web text out of training data, and (unreliably) flags machine-written text.",
        "**Summarisation and translation evaluation**: ROUGE and BLEU are still the default automatic metrics in papers and shared tasks, usually next to human evaluation.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**n-gram LMs**: fast, exact counts, easy to update and understand. But no generalisation between similar words, a hard context limit, and tables that grow with data.",
        "**Neural LMs**: generalise through embeddings and handle long contexts. But they need far more data and compute and, as measured here, can lose to Kneser–Ney on small data.",
        "**Perplexity**: right for models of the *same* token sequence. Meaningless across tokenizers (use bits per character/byte), sensitive to the test set and unknown-word handling, and only loosely tied to generation quality or downstream accuracy.",
        "**BLEU/ROUGE**: cheap and reproducible; they track human judgement reasonably when averaged over a large test set for similar systems. Unreliable for single sentences, very different systems, and open-ended generation where many outputs are correct.",
        "**Samples**: show repetition, incoherence and memorisation that averages hide — but a handful of samples is anecdote; use statistics over many.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Comparing perplexities across tokenizers or vocabularies.** Measured above: per-token PPL 8.39 vs 201.34 for models with 3.068 vs 2.921 bits per character. Convert to bits per character/byte, or keep the tokenizer fixed.",
        "**Mapping unknown words to <unk>, then comparing perplexities.** A smaller vocabulary sends more words to the easy-to-predict <unk>, so perplexity looks better. Compare only with the same vocabulary, or use an open-vocabulary (character/byte/BPE) model.",
        String.raw`**Tuning on the test set.** $k$, $\lambda$, $D$ and the number of epochs (early stopping) must be chosen on a development set; the test set is used once.`,
        "**Forgetting the end-of-sentence symbol** in n-gram models: probabilities over sentences of all lengths then do not sum to one, and perplexities are not comparable.",
        "**Mixing nats and bits**: perplexity is $e^{H}$ for $H$ in nats and $2^{H}$ for $H$ in bits. `exp` of a loss computed with `log2` is wrong.",
        "**Reporting training perplexity.** A model that memorises (like the 6.8k-character Transformer in the simulator) can reach very low training perplexity while its held-out perplexity rises.",
        "**Computing BLEU on single sentences without smoothing**, averaging sentence BLEU instead of computing corpus BLEU, or tokenizing references differently. Scores from different scripts are not comparable: use a standard tool such as sacreBLEU and report its signature.",
        "**Unclipped n-gram precision**: rewards repeating common words (the 7/7 example).",
        "**Believing higher BLEU/ROUGE means better text**: both reward surface overlap, so a correct paraphrase can score low and a fluent but wrong summary high.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What does a language model compute?", a: String.raw`A probability for any token sequence, split by the chain rule into next-token probabilities $P(w_t\mid w_{<t})$. In other words, a next-token predictor; generating text means sampling from these one token at a time.` },
      { t: "qa", level: "beginner", q: "What is perplexity and how is it computed?", a: String.raw`The exponential of the average negative log-probability per token on held-out text: $\exp(-\frac1T\sum_t\ln p(w_t\mid w_{<t}))$. It is one over the geometric mean of the true tokens' probabilities. A perplexity of $k$ means the model is as unsure as an even choice among $k$ tokens. Lower is better.` },
      { t: "qa", level: "beginner", q: "Why do n-gram models need smoothing?", a: "Most n-grams never occur in training, so maximum-likelihood estimates give them probability 0. One zero makes a test sentence impossible and the perplexity infinite. Smoothing moves some probability to unseen events." },
      { t: "qa", level: "intermediate", q: "Explain the idea behind Kneser–Ney smoothing.", a: "Two parts. Absolute discounting subtracts a constant $D$ (≈0.75) from each seen count and gives the freed probability to a shorter-history model. That model uses continuation counts — how many different words a word follows — instead of raw frequency, because a word common only in one fixed phrase (\"Francisco\" after \"San\") is a poor guess after a new history. Interpolated modified KN is the strongest classical n-gram smoother." },
      { t: "qa", level: "intermediate", q: "Why can't you compare perplexities of models with different tokenizers, and what do you do instead?", a: String.raw`Perplexity is per token. Longer tokens mean fewer, harder predictions, so per-token perplexity rises even if the model gives the text the same total probability. Instead, normalise the total log-likelihood by a tokenizer-independent unit: bits per character or per byte, $\text{bpc} = T H_{\text{token}}/(C\ln2)$. In our measurement BPE-600 had perplexity 201 vs 8.4 for characters, but was better per character (2.92 vs 3.07 bits).` },
      { t: "qa", level: "intermediate", q: "How does BLEU work, and why does it need a brevity penalty?", a: String.raw`It computes clipped n-gram precisions $p_1..p_4$ over the whole corpus, takes their geometric mean, and multiplies by $\mathrm{BP} = e^{1-r/c}$ when the candidate is shorter than the references. Precision alone rewards outputting a few safe words ("the cat" has precision 1). Recall is awkward with several references, so BLEU penalises shortness instead.` },
      { t: "qa", level: "advanced", q: "What advantage did Bengio et al.'s neural LM have over n-gram models, and what was its bottleneck?", a: "It learns an embedding per word and predicts from a smooth function of the joined embeddings, so unseen histories still get good predictions if they resemble seen ones — it generalises across similar words, which count tables cannot. Its bottleneck was the softmax over the full vocabulary at every position (plus the fixed window), which led to hierarchical softmax, sampling-based training and, later, RNN and Transformer LMs." },
      { t: "qa", level: "advanced", q: "Your new model has lower validation perplexity than the baseline, but its generated text is worse. How can that happen?", a: "Perplexity scores the model's probabilities for human text under teacher forcing; generation judges what the model writes from its own samples. They can disagree: a badly calibrated tail (sampling picks bad unlikely tokens), exposure bias over long outputs, or repetition under greedy/beam decoding. Also check the perplexities are comparable (same tokenizer, same <unk> handling, no test leakage). Then evaluate generation directly — samples, repetition and distinct-n, task metrics, human or model-judge preference — and tune decoding (temperature, top-p)." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A model assigns probabilities 0.5, 0.25, 0.125 to the three tokens of a test text. What is its perplexity?", answer: 4, tolerance: 0.001, explain: String.raw`The geometric mean of the probabilities is $(0.5\cdot0.25\cdot0.125)^{1/3} = (1/64)^{1/3} = 1/4$, so PPL = 4.` },
        { kind: "numeric", q: "A character-level model has a test cross-entropy of 2.0 nats per character. How many bits per character is that (3 decimals)?", answer: 2.885, tolerance: 0.002, explain: String.raw`$2.0/\ln 2 = 2.885$ bits.` },
        { kind: "numeric", q: "A word-level model has per-token perplexity 100 on a text of 1,000 words and 5,500 characters. What is its bits per character (3 decimals)?", answer: 1.208, tolerance: 0.002, explain: String.raw`Total bits $= 1000\log_2 100 = 6643.9$; divided by 5,500 characters $= 1.208$ bpc.` },
        { kind: "numeric", q: String.raw`With add-1 smoothing and vocabulary size $V = 10$, a history seen 5 times was never followed by word $w$. What is $P(w\mid h)$ (4 decimals)?`, answer: 0.0667, tolerance: 0.0001, explain: "$(0+1)/(5+10) = 1/15 = 0.0667$." },
        { kind: "numeric", q: "A 5-word candidate has clipped precisions $p_1 = 1$, $p_2 = 0.5$ and the reference has 10 words. What is BLEU-2?", answer: 0.2601, tolerance: 0.0005, explain: String.raw`$\mathrm{BP} = e^{1-10/5} = e^{-1} = 0.3679$; geometric mean $\sqrt{1\cdot0.5} = 0.7071$; product 0.2601.` },
        { q: "Which lower-order distribution does Kneser–Ney use?", options: ["Raw unigram frequencies", "Continuation counts: how many distinct words precede each word", "A uniform distribution", "The development-set frequencies"], answer: 1, explain: String.raw`$P_{\text{cont}}(w)\propto N_{1+}(\bullet\,w)$ — how varied a word's contexts are, not how frequent it is.` },
        { q: "Two LMs are trained on the same corpus, one with a 32k BPE vocabulary and one with a 50k vocabulary. Model A reports PPL 12, model B PPL 15. What can you conclude?", options: ["A is better", "B is better", "Nothing until both are expressed per character/byte on the same test text", "They are equal"], answer: 2, explain: "Per-token perplexities from different tokenizers are not comparable." },
        { q: "Why is sentence-level BLEU-4 often exactly 0?", options: ["Because of the brevity penalty", "Because a short sentence often has no matching 4-gram, and the geometric mean of precisions is then 0", "Because BLEU only works for French", "Because of clipping"], answer: 1, explain: "Any $p_n = 0$ makes the product 0 — hence smoothed sentence-BLEU variants." },
      ] },
      { t: "exercise", title: "Perplexity of a bigram model by hand", prompt: String.raw`Training text (with sentence markers): "<s> a b </s>", "<s> a a b </s>", "<s> b </s>". Tokens to predict: {a, b, </s>}. (1) Give the MLE bigram probabilities $P(a\mid\text{<s>})$, $P(b\mid a)$, $P(\text{</s>}\mid b)$. (2) Compute the perplexity of the test sentence "<s> a b </s>" under the MLE model. (3) Redo it with add-1 smoothing.`, hint: "Count every history, including <s>; the test sentence has 3 predictions.", solution: [
        { t: "steps", items: [
          { title: "Counts", text: String.raw`<s> is followed by a, a, b ⇒ $c(\text{<s>}) = 3$; a is followed by b, a, b ⇒ $c(a) = 3$; b is followed by </s> three times ⇒ $c(b) = 3$.` },
          { title: "MLE", tex: String.raw`P(a\mid\text{<s>}) = \tfrac23,\quad P(b\mid a) = \tfrac23,\quad P(\text{</s>}\mid b) = 1` },
          { title: "Perplexity (MLE)", tex: String.raw`P = \tfrac23\cdot\tfrac23\cdot1 = \tfrac49,\quad \mathrm{PPL} = (4/9)^{-1/3} = 1.310371` },
          { title: "Add-1 ($V = 3$)", tex: String.raw`\tfrac{2+1}{3+3} = 0.5,\ \ \tfrac{2+1}{3+3} = 0.5,\ \ \tfrac{3+1}{3+3} = 0.666667;\ \ P = 0.166667,\ \ \mathrm{PPL} = 6^{1/3} = 1.817121` },
        ] },
        { t: "p", text: "Smoothing *raises* the perplexity of a sentence the MLE model already handles, because it moves probability to unseen events. It pays off only on test text with unseen bigrams (where the MLE perplexity would be infinite)." },
      ] },
      { t: "exercise", title: "Add stupid backoff and a word-level model", prompt: "Extend `ngram_lm.py`: (a) implement stupid backoff for characters (score $= c(h,w)/c(h)$ if $c(h,w)>0$, else $0.4\\times$ the score with the shortened history, ending at the unigram relative frequency) and show with numbers that its scores do not sum to one over the vocabulary; (b) build a word-level Kneser–Ney trigram on `words(train)`, with every word seen once in training mapped to `<unk>`, and report its test perplexity and bits per character. Why is the word model's bpc not a fair comparison with the character models?", hint: "For (a), print the sum of scores over `chars` for a few contexts. For (b), the number of characters is still `len(test)`.", solution: [
        { t: "code", lang: "text", code: String.raw`def stupid_backoff(seq_counts, ctx, w, order, alpha=0.4):
    for m in range(order, 0, -1):
        h = ctx[len(ctx) - m + 1:] if m > 1 else ""
        num, den = seq_counts[m][h + w], seq_counts[m - 1][h]
        if num > 0:
            return alpha ** (order - m) * num / den
    return 0.0

counts = [Counter({"": len(train)})] + [Counter(train[i:i + m] for i in range(len(train) - m + 1)) for m in range(1, 6)]
for ctx in ["alice was", "the rabb", "qz"]:
    print(ctx, sum(stupid_backoff(counts, ctx, ch, 5) for ch in chars))   # not 1.0 in general` },
        { t: "p", text: "Added to the script, this prints sums of 1.046 (\"alice was\"), 1.072 (\"the rabb\") and 0.086 (\"qz\", never seen). Stupid backoff returns a raw relative frequency at the longest matching history, plus scaled scores for symbols seen only with shorter histories, so the total is above 1 for seen contexts and far below 1 for unseen ones: a score, not a probability, with no perplexity." },
        { t: "p", text: "For (b): with one-off words mapped to <unk>, the word model predicts one cheap symbol for every rare word (about a fifth of test tokens here) without spelling it out. Its \"bits per character\" skip the cost of those characters and look better than they are. A fair comparison needs an open-vocabulary model (characters, bytes or BPE) or a character-level spelling model for <unk>." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        String.raw`An LM defines $P(w_1..w_T) = \prod_t P(w_t\mid w_{<t})$; n-gram models cut the history to $n-1$ tokens.`,
        String.raw`MLE: $P(w\mid h) = c(h,w)/c(h)$. Sparsity: 96.5% of test word trigrams were unseen in our text; unseen ⇒ probability 0 ⇒ infinite perplexity.`,
        String.raw`Smoothing: add-k $\frac{c+k}{c(h)+kV}$; interpolation $\sum_m\lambda_m P_m$ (λ tuned on dev data); backoff; Kneser–Ney = absolute discount $D$ + continuation counts $N_{1+}(\bullet w)$. Measured: KN 5-gram 6.69 PPL vs add-k 14.98.`,
        String.raw`$\mathrm{PPL} = e^{H}$ with $H$ the mean NLL in nats; bpc $= TH/(C\ln 2)$. Compare across tokenizers only in bits per character/byte (8.39 vs 201.34 PPL, but 3.068 vs 2.921 bpc).`,
        String.raw`Neural LMs: NPLM $y = b + Wx + U\tanh(d + Hx)$ on joined embeddings; RNN and Transformer LMs extend the context. Same loss, same perplexity. On 6k characters, KN still won (2.743 bpc vs GRU 2.998, NPLM 3.285).`,
        String.raw`BLEU $= \mathrm{BP}\exp(\frac14\sum\log p_n)$ with clipped precisions and $\mathrm{BP} = e^{1-r/c}$ for short outputs; ROUGE-N/L are recall- and LCS-based. All are corpus-level, reference-based, surface-overlap metrics.`,
        "Always look at samples too: fluency, repetition, copying from the training data.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — how Transformer LMs are trained with the causal objective and how they generate.",
        "[Decoding: Greedy, Beam, Temperature, Top-k, Top-p](/learn/m20/decoding) — the strategies that turn next-token probabilities into text, and why they matter as much as perplexity.",
        "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — language modelling at scale: data, scaling laws and evaluation beyond perplexity.",
        "[Classification, NER, QA, Summarization, Translation](/learn/m13/nlp-tasks) — the tasks where BLEU and ROUGE are used.",
      ] },
    ],
  },
};

export default lesson;
