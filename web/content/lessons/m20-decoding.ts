import type { Lesson } from "@/lib/types";

/* ------------------------------------------------------------------ code blocks (executed; outputs are real) */

const SCRATCH = String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)
tokens = ["A", "B", "C", "D", "E"]
z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])        # logits for the next token


def softmax(z, T=1.0):
    s = z / T
    e = np.exp(s - s.max())                     # subtract the max: exp never overflows
    return e / e.sum()


def top_k_filter(p, k):
    keep = np.zeros_like(p, dtype=bool)
    keep[np.argsort(-p, kind="stable")[:k]] = True
    return keep


def top_p_filter(p, top_p):
    """Nucleus: the smallest set of most-probable tokens whose total mass reaches top_p."""
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order])
    n = int(np.searchsorted(cum, top_p) + 1)    # first position where cum >= top_p, inclusive
    keep = np.zeros_like(p, dtype=bool)
    keep[order[:n]] = True
    return keep


def min_p_filter(p, min_p):
    return p >= min_p * p.max()                  # threshold relative to the top token


def renormalize(p, keep):
    q = np.where(keep, p, 0.0)
    return q / q.sum()


def repetition_penalty(z, seen, theta):
    """CTRL-style: shrink the logits of already-generated tokens (divide if > 0, multiply if < 0)."""
    z = z.copy()
    for i in set(seen):
        z[i] = z[i] / theta if z[i] > 0 else z[i] * theta
    return z


# ---- 1. temperature reshapes, truncation cuts ----
for T in (0.5, 1.0, 2.0):
    p = softmax(z, T)
    H = -(p * np.log(p)).sum()
    print(f"T={T}: p = {p}  entropy = {H:.4f} nats")
p = softmax(z)
print("top-k=2  :", renormalize(p, top_k_filter(p, 2)))
print("top-p=0.8:", renormalize(p, top_p_filter(p, 0.8)), " cumulative:", np.cumsum(p))
print("min-p=0.2:", renormalize(p, min_p_filter(p, 0.2)), " threshold:", round(0.2 * p.max(), 4))
zr = repetition_penalty(z, seen=[0, 4], theta=1.3)
print("penalized logits:", zr, " p:", softmax(zr))

# ---- 2. sampling = inverse CDF on a uniform number; frequencies match q ----
rng = np.random.default_rng(0)
q = renormalize(p, top_p_filter(p, 0.8))
u = rng.random(100_000)
draws = np.searchsorted(np.cumsum(q), u, side="right")   # index of the first cumulative value > u
print("empirical:", np.bincount(draws, minlength=5) / len(u), " target:", q)

# ---- 3. greedy vs beam search on a tiny language model with an end token ----
V = ["<eos>", "the", "a", "cat", "dog", "sat"]
P = np.zeros((7, 6))                            # row = previous token (6 = <bos>), column = next token
P[6, [1, 2, 5]] = [0.5, 0.4, 0.1]               # <bos> -> the / a / sat
P[1, [3, 4, 5]] = [0.4, 0.4, 0.2]               # the  -> cat / dog / sat
P[2, [3, 4]] = [0.9, 0.1]                       # a    -> cat / dog
P[3, [0, 5]] = [0.55, 0.45]                     # cat  -> <eos> / sat
P[4, [0, 5]] = [0.4, 0.6]                       # dog  -> <eos> / sat
P[5, 0] = 1.0                                   # sat  -> <eos>
with np.errstate(divide="ignore"):
    logP = np.log(P)                            # log 0 = -inf: impossible continuations


def show(seq):
    return " ".join(V[t] for t in seq)


def greedy(max_len=6):
    seq, prev, lp = [], 6, 0.0
    for _ in range(max_len):
        t = int(np.argmax(logP[prev]))
        seq.append(t); lp += logP[prev, t]; prev = t
        if t == 0:
            break
    return seq, lp


def beam_search(width, alpha, max_len=6):
    """Keep "width" unfinished hypotheses; finished ones (ending in <eos>) are ranked by logp / len**alpha."""
    alive, finished = [([], 6, 0.0)], []
    for _ in range(max_len):
        cands = [(seq + [t], t, lp + logP[prev, t]) for seq, prev, lp in alive
                 for t in range(len(V)) if np.isfinite(logP[prev, t])]
        cands.sort(key=lambda c: -c[2])          # rank partial hypotheses by total log-probability
        alive = []
        for seq, t, lp in cands:
            if t == 0:
                finished.append((seq, lp))
            elif len(alive) < width:
                alive.append((seq, t, lp))
        if not alive:
            break
    return max(finished, key=lambda f: f[1] / len(f[0]) ** alpha)


g, lp = greedy()
print(f"greedy          : {show(g):22s} p = {np.exp(lp):.4f}")
for width, alpha in [(1, 0.0), (2, 0.0), (3, 0.0), (3, 1.0)]:
    seq, lp = beam_search(width, alpha)
    print(f"beam w={width} a={alpha}: {show(seq):22s} p = {np.exp(lp):.4f}  logp/len^a = {lp / len(seq) ** alpha:.4f}")

# exhaustive check: every complete sentence and its probability
def all_sentences(prev=6, seq=(), lp=0.0):
    for t in range(len(V)):
        if np.isfinite(logP[prev, t]):
            if t == 0:
                yield seq + (t,), lp + logP[prev, t]
            else:
                yield from all_sentences(t, seq + (t,), lp + logP[prev, t])

ranked = sorted(all_sentences(), key=lambda s: -s[1])
print("all sentences, most probable first:")
for seq, lp in ranked[:5]:
    print(f"  {show(seq):22s} p = {np.exp(lp):.4f}  per-token logp = {lp / len(seq):.4f}")
print("total probability of all sentences:", round(float(sum(np.exp(lp) for _, lp in ranked)), 6))`;

const SCRATCH_OUT = String.raw`T=0.5: p = [0.8292 0.1122 0.0413 0.0152 0.0021]  entropy = 0.6086 nats
T=1.0: p = [0.563  0.2071 0.1256 0.0762 0.028 ]  entropy = 1.2065 nats
T=2.0: p = [0.3745 0.2272 0.1769 0.1378 0.0836]  entropy = 1.4915 nats
top-k=2  : [0.7311 0.2689 0.     0.     0.    ]
top-p=0.8: [0.6285 0.2312 0.1402 0.     0.    ]  cumulative: [0.563  0.7701 0.8958 0.972  1.    ]
min-p=0.2: [0.6285 0.2312 0.1402 0.     0.    ]  threshold: 0.1126
penalized logits: [ 1.5385  1.      0.5     0.     -1.3   ]  p: [0.4523 0.264  0.1601 0.0971 0.0265]
empirical: [0.6285 0.2329 0.1386 0.     0.    ]  target: [0.6285 0.2312 0.1402 0.     0.    ]
greedy          : the cat <eos>          p = 0.1100
beam w=1 a=0.0: the cat <eos>          p = 0.1100  logp/len^a = -2.2073
beam w=2 a=0.0: a cat <eos>            p = 0.1980  logp/len^a = -1.6195
beam w=3 a=0.0: a cat <eos>            p = 0.1980  logp/len^a = -1.6195
beam w=3 a=1.0: a cat sat <eos>        p = 0.1620  logp/len^a = -0.4550
all sentences, most probable first:
  a cat <eos>            p = 0.1980  per-token logp = -0.5398
  a cat sat <eos>        p = 0.1620  per-token logp = -0.4550
  the dog sat <eos>      p = 0.1200  per-token logp = -0.5301
  the cat <eos>          p = 0.1100  per-token logp = -0.7358
  the sat <eos>          p = 0.1000  per-token logp = -0.7675
total probability of all sentences: 1.0`;

const TORCH = String.raw`import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.generation import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.utils import logging

logging.set_verbosity_error()                   # silence generation-config notices
torch.manual_seed(0)
torch.set_num_threads(1)


def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Batched temperature -> top-k -> top-p. Removed entries become -inf. logits: (B, V)."""
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]           # k-th largest per row
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) >= top_p                 # mass BEFORE this token already >= p
        remove = remove.scatter(-1, idx, remove)                          # back to vocabulary order
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


# ---- 1. our filter vs the Hugging Face logits warpers on random logits ----
B, V = 64, 1000
logits = torch.randn(B, V) * 3
ours = filter_logits(logits, temperature=0.7, top_k=50, top_p=0.9)
hf = logits.clone()
for w in (TemperatureLogitsWarper(0.7), TopKLogitsWarper(50), TopPLogitsWarper(0.9)):
    hf = w(None, hf)
kept_ours, kept_hf = torch.isfinite(ours), torch.isfinite(hf)
print("kept-set agreement with HF warpers:", bool((kept_ours == kept_hf).all()),
      "| kept per row min/mean/max:", kept_ours.sum(1).min().item(), round(kept_ours.sum(1).float().mean().item(), 1), kept_ours.sum(1).max().item())
print("max |prob difference|:", (ours.softmax(-1) - hf.softmax(-1)).abs().max().item())

# ---- 2. a tiny randomly initialised GPT-2 (no download): manual loops vs model.generate ----
cfg = GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=4, bos_token_id=0, eos_token_id=0)
model = GPT2LMHeadModel(cfg).eval()
prompt = torch.tensor([[5, 17, 3]])


@torch.no_grad()
def manual_greedy(ids, n):
    for _ in range(n):
        nxt = model(ids).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    return ids


@torch.no_grad()
def manual_beam(ids, n, width):
    beams = [(ids, 0.0)]
    for _ in range(n):
        cands = []
        for seq, score in beams:
            lp = F.log_softmax(model(seq).logits[0, -1], -1)
            top = torch.topk(lp, width)
            cands += [(torch.cat([seq, t.view(1, 1)], 1), score + v.item()) for v, t in zip(top.values, top.indices)]
        beams = sorted(cands, key=lambda c: -c[1])[:width]
    return beams


with torch.no_grad():
    hf_greedy = model.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0)
    hf_beam = model.generate(prompt, max_new_tokens=8, num_beams=3, do_sample=False, length_penalty=0.0,
                             early_stopping=False, pad_token_id=0, output_scores=True, return_dict_in_generate=True)
mg = manual_greedy(prompt, 8)
mb = manual_beam(prompt, 8, 3)
print("greedy  manual:", mg[0, 3:].tolist(), " generate:", hf_greedy[0, 3:].tolist())
print("beam(3) manual:", mb[0][0][0, 3:].tolist(), f"logp {mb[0][1]:.4f}")
print("beam(3) generate:", hf_beam.sequences[0, 3:].tolist(), f"logp {hf_beam.sequences_scores[0].item():.4f}")


@torch.no_grad()
def seq_logp(seq):
    lp = F.log_softmax(model(seq).logits[0, :-1], -1)
    return lp[torch.arange(2, seq.shape[1] - 1), seq[0, 3:]].sum().item()


print(f"log p of greedy output {seq_logp(mg):.4f}  vs beam output {seq_logp(mb[0][0]):.4f}")

# ---- 3. seeded sampling with top-p: reproducible given the generator ----
g = torch.Generator().manual_seed(123)
probs = filter_logits(model(prompt).logits[:, -1], temperature=0.8, top_p=0.9).softmax(-1)
draws = torch.multinomial(probs.expand(5, -1), 1, generator=g).squeeze(1)
print("nucleus size:", int((probs > 0).sum()), "| 5 seeded draws:", draws.tolist())`;

const TORCH_OUT = String.raw`kept-set agreement with HF warpers: True | kept per row min/mean/max: 1 10.9 31
max |prob difference|: 0.0
greedy  manual: [3, 3, 3, 3, 3, 3, 3, 3]  generate: [3, 3, 3, 3, 3, 3, 3, 3]
beam(3) manual: [3, 3, 3, 3, 49, 45, 45, 45] logp -27.4012
beam(3) generate: [3, 3, 3, 3, 49, 45, 45, 45] logp -27.4012
log p of greedy output -28.1480  vs beam output -27.4012
nucleus size: 44 | 5 seeded draws: [1, 20, 22, 22, 49]`;

const KERAS = String.raw`import tensorflow as tf


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0, seed=None):
    """logits: (batch, vocab). Returns sampled token ids, shape (batch,)."""
    logits = logits / temperature
    if top_k > 0:
        kth = tf.math.top_k(logits, k=top_k).values[:, -1:]
        logits = tf.where(logits < kth, tf.fill(tf.shape(logits), -1e9), logits)
    if top_p < 1.0:
        sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
        probs = tf.nn.softmax(sorted_logits, axis=-1)
        before = tf.cumsum(probs, axis=-1, exclusive=True)          # mass of strictly better tokens
        cutoff = tf.reduce_min(tf.where(before < top_p, sorted_logits, 1e9), axis=-1, keepdims=True)
        logits = tf.where(logits < cutoff, tf.fill(tf.shape(logits), -1e9), logits)
    return tf.random.categorical(logits, num_samples=1, seed=seed)[:, 0]


logits = tf.random.normal((4, 50), seed=0) * 3
print(sample_next(logits, temperature=0.8, top_k=20, top_p=0.9, seed=1))`;

const EX_SOL = String.raw`import numpy as np

z = np.array([3.0, 2.5, 1.0, 0.2, -0.5, -2.0])


def nucleus_size(z, p, T=1.0):
    q = np.exp(z / T - (z / T).max())
    q /= q.sum()
    order = np.argsort(-q, kind="stable")
    return int(np.searchsorted(np.cumsum(q[order]), p) + 1)   # first index where cumulative >= p


print("p at T=1:", np.round(np.exp(z) / np.exp(z).sum(), 4))
for T in (0.5, 1.0, 2.0):
    print(f"T={T}: nucleus sizes for p = 0.5, 0.9, 0.99 ->", [nucleus_size(z, p, T) for p in (0.5, 0.9, 0.99)])`;

const EX_SOL_OUT = String.raw`p at T=1: [0.5436 0.3297 0.0736 0.0331 0.0164 0.0037]
T=0.5: nucleus sizes for p = 0.5, 0.9, 0.99 -> [1, 2, 3]
T=1.0: nucleus sizes for p = 0.5, 0.9, 0.99 -> [1, 3, 5]
T=2.0: nucleus sizes for p = 0.5, 0.9, 0.99 -> [2, 4, 6]`;

/* ------------------------------------------------------------------ lesson */

const lesson: Lesson = {
  id: "m20/decoding",
  title: "Decoding: Greedy, Beam, Temperature, Top-k, Top-p",
  summary:
    "A language model outputs a probability distribution over the next token; a decoding algorithm turns that into text. Greedy and beam search hunt for probable sequences; sampling with temperature, top-k, top-p (nucleus) and min-p trades probability for variety; repetition penalties patch the failure modes. All built from scratch and checked against Hugging Face.",
  minutes: 60,
  level: "intermediate",
  prerequisites: ["m11/training-inference", "m02/information-theory", "m02/probability", "m13/language-modeling"],
  next: ["m20/llm-inference", "m20/alignment", "m20/llm-pretraining"],
  terms: ["greedy-decoding", "beam-search", "length-normalization", "top-k-sampling", "top-p-sampling", "min-p-sampling", "repetition-penalty", "temperature", "text-degeneration"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "A language model never outputs text directly. At each step it outputs **logits** $z \\in \\mathbb{R}^{|\\mathcal V|}$ — one score per vocabulary token — and a softmax turns them into a distribution $p(\\cdot \\mid x_{<t})$ over the next token. The **decoding algorithm** picks a token, appends it and runs the model again. It is a separate choice from the model: the same weights can give terse, repetitive, creative or nonsense text depending on how you decode." },
      { t: "p", text: "Two goals pull against each other. For tasks with one right answer (translation, code completion, pulling out a date) we want the **most probable** output: greedy decoding and **beam search**. For open-ended writing (stories, dialogue, brainstorming) that output is often bland: Holtzman et al. (2020, *The Curious Case of Neural Text Degeneration*) showed that maximising probability gives text far more repetitive and less surprising than human text." },
      { t: "p", text: "So for open-ended text we **sample** — after reshaping the distribution with **temperature**, **top-k**, **top-p (nucleus)** or **min-p**. These cut off the long tail of tokens that are each unlikely but together likely to be junk. Chat assistants, code completers and translators expose these knobs (`temperature`, `top_p`, `top_k`, `num_beams`, `repetition_penalty`); we'll build each one and watch it on a real trained Transformer." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — the generation loop (prefill, append, repeat) that decoding plugs into, and why the model gives a distribution at every position.",
        "[Probability](/learn/m02/probability) — categorical distributions, sampling, and the chain rule $p(y_1,\\dots,y_T)=\\prod_t p(y_t\\mid y_{<t})$, which makes a sequence's score a sum of log-probabilities.",
        "[Information Theory](/learn/m02/information-theory) — entropy, a measure of how flat a distribution is; temperature changes it.",
        "[Language Modeling](/learn/m13/language-modeling) — perplexity and what a language model's probabilities mean.",
        "Softmax and logits from [Classification Losses](/learn/m05/classification-losses).",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "Choosing a restaurant from friends' recommendations", text: "Ten friends vote on where you should eat. **Greedy**: always the top-voted place — reliable, but the same chain every night. **Beam search**: plan the whole evening (restaurant, bar, dessert) and pick the best *combination*. **Sampling**: pick at random in proportion to votes — variety, but one night the gas-station-sushi vote wins. **Temperature** sets how much you trust the vote counts (low T: nearly always the favourite; high T: every suggestion counts almost equally). **Top-k**: only consider the k most-voted places. **Top-p**: take the most-voted places until they cover a share p of all votes — one place when everyone agrees, many when they don't." },
      { t: "p", text: "The key insight: the model's distribution is usually **right at the head and wrong in the tail**. The top few tokens are sensible; the thousands after them are each tiny but together can hold 10–30% of the mass. Pure sampling dips into that tail regularly, and one bad token derails everything after it, because the model builds on its own mistake. Truncation cuts the tail off; temperature reshapes the head." },
      { t: "p", text: "Search methods (greedy, beam) have the opposite problem: they never take risks, so they fall into loops. Once a phrase appears, copying it is easy for attention, so repeating it becomes the likeliest next step — as both models in the code below show." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "The generation loop with a pluggable decoder" },
      { t: "steps", items: [
        { title: "Get logits", text: "Run the model on the prompt plus everything generated so far; take the logits $z$ at the last position (with a KV cache this is one token of work — see [Efficient Inference](/learn/m20/llm-inference))." },
        { title: "Adjust logits", text: "Optionally apply a **repetition penalty** to tokens already present, ban tokens (e.g. disallowed words), or add a bias." },
        { title: "Reshape", text: "Divide by the **temperature** $T$: $z \\leftarrow z/T$." },
        { title: "Truncate", text: "Set the logits of tokens outside the kept set to $-\\infty$. **Top-k** keeps the $k$ largest; **top-p** keeps the smallest set whose probability reaches $p$; **min-p** keeps tokens with $p_i \\ge p_{\\min}\\cdot p_{\\max}$." },
        { title: "Choose", text: "Greedy: $\\arg\\max$. Sampling: softmax over the survivors (which renormalises them), then draw. Beam search: score many continuations of many hypotheses instead (below)." },
        { title: "Append and stop", text: "Append the token. Stop at an end-of-sequence token, a stop string or a length limit." },
      ] },
      { t: "h", text: "Beam search step by step" },
      { t: "steps", items: [
        { title: "Initialise", text: "One hypothesis: the empty continuation, score 0." },
        { title: "Expand", text: "For each of the $B$ alive hypotheses compute $\\log p(\\cdot \\mid \\text{prompt}, \\text{hypothesis})$ over the vocabulary. That gives $B\\cdot|\\mathcal V|$ candidates, each scored as its parent's score plus the new token's log-probability." },
        { title: "Prune", text: "Sort the candidates. Those ending in `<eos>` move to a *finished* list; the best $B$ others stay alive." },
        { title: "Repeat", text: "Until no hypothesis is alive or the maximum length is reached." },
        { title: "Pick", text: "Return the finished hypothesis with the best **length-normalised** score (next section). Width $B = 1$ is exactly greedy." },
      ] },
      { t: "callout", kind: "note", text: "Order matters for sampling. Libraries usually apply penalties, temperature, top-k, then top-p. Temperature *before* top-p changes which tokens survive: higher T flattens the distribution, so more tokens are needed to reach mass $p$. Top-k and top-p in either order give the intersection of both kept sets only if each is computed on the same distribution." },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Sequence scores" },
      { t: "p", text: String.raw`For a prompt $x$ and continuation $y = (y_1,\dots,y_L)$, the model gives $\log p(y\mid x) = \sum_{t=1}^{L} \log p(y_t \mid x, y_{<t})$ — the log-probabilities of each step, added up. Greedy picks $y_t = \arg\max_v p(v\mid x,y_{<t})$ one step at a time. The ideal "MAP" decode $\arg\max_y \log p(y\mid x)$ is out of reach ($|\mathcal V|^L$ sequences), so beam search approximates it with $B$ hypotheses.` },
      { t: "h", text: "Temperature" },
      { t: "math", tex: String.raw`p_T(i) = \frac{\exp(z_i/T)}{\sum_{j}\exp(z_j/T)}, \qquad \lim_{T\to 0} p_T = \text{one-hot}(\arg\max z), \qquad \lim_{T\to\infty} p_T = \text{uniform}`, caption: "$z_i$ = logit of token $i$, $T > 0$ = temperature. $T=1$ is the model's own distribution." },
      { t: "p", text: "Dividing by a positive number keeps the order, so temperature never changes the **ranking** of tokens — and greedy decoding ignores it. It changes the **entropy**: $T<1$ piles mass onto the top tokens, $T>1$ spreads it out." },
      { t: "h", text: "Worked example: logits $z = (2, 1, 0.5, 0, -1)$ for tokens A–E" },
      { t: "steps", items: [
        { title: "Softmax at T = 1", tex: String.raw`e^{z} = (7.389056,\ 2.718282,\ 1.648721,\ 1,\ 0.367879),\ \textstyle\sum = 13.123939,\ p = (0.5630,\ 0.2071,\ 0.1256,\ 0.0762,\ 0.0280)` },
        { title: "T = 0.5 (sharper)", tex: String.raw`e^{z/0.5} = (54.598,\ 7.389,\ 2.718,\ 1,\ 0.135),\ \textstyle\sum = 65.8408,\ p = (0.8292,\ 0.1122,\ 0.0413,\ 0.0152,\ 0.0021)` },
        { title: "Entropy", tex: String.raw`H(p_{0.5}) = 0.6086,\quad H(p_1) = 1.2065,\quad H(p_2) = 1.4915 \text{ nats} \;(\ln 5 = 1.609 \text{ is the maximum})` },
        { title: "Top-k, k = 2", tex: String.raw`\text{keep A, B: mass } 0.770145 \Rightarrow q = (0.5630,\ 0.2071)/0.770145 = (0.7311,\ 0.2689)` },
        { title: "Top-p, p = 0.8", tex: String.raw`\text{cumulative } (0.5630,\ 0.7701,\ 0.8958,\dots): \text{ first reaches } 0.8 \text{ at C} \Rightarrow q = (0.6285,\ 0.2312,\ 0.1402)` },
        { title: "Min-p, 0.2", tex: String.raw`\text{threshold } 0.2 \times 0.5630 = 0.1126 \Rightarrow \text{keep A, B, C (same set as top-p here)}` },
      ] },
      { t: "h", text: "Top-p (nucleus) sampling" },
      { t: "p", text: String.raw`Sort tokens by probability, $p_{(1)} \ge p_{(2)} \ge \dots$. The nucleus is $V^{(p)} = \{(1),\dots,(m)\}$ with $m = \min\{m' : \sum_{i\le m'} p_{(i)} \ge p\}$ (Holtzman et al., 2020) — the fewest top tokens whose probabilities add up to at least $p$. Sample from $q(v) = p(v)\,\mathbb 1[v\in V^{(p)}] / \sum_{u\in V^{(p)}} p(u)$. Unlike top-k, the size $m$ **adapts**: 1 or 2 when the model is confident, large when many continuations are plausible. Top-k (Fan, Lewis and Dauphin, 2018, *Hierarchical Neural Story Generation*) fixes $m=k$ — too many tokens for a peaked distribution, too few for a flat one.` },
      { t: "p", text: String.raw`**Min-p** (Nguyen et al., 2024) keeps $\{v : p(v) \ge p_{\min}\cdot \max_u p(u)\}$. The threshold scales with the model's confidence, so it stays well-behaved at high temperatures, where top-p's nucleus gets very large.` },
      { t: "h", text: "Repetition penalty" },
      { t: "math", tex: String.raw`z'_i = \begin{cases} z_i/\theta & z_i > 0 \\ z_i\cdot\theta & z_i \le 0 \end{cases} \quad \text{for every token } i \text{ already in the context}, \qquad \theta > 1`, caption: "CTRL-style penalty (Keskar et al., 2019, who used θ ≈ 1.2). Both branches lower the logit. Frequency/presence penalties instead subtract a constant (times the count) from the logit." },
      { t: "p", text: String.raw`In the example, with A and E already generated and $\theta = 1.3$: $z'_A = 2/1.3 = 1.5385$, $z'_E = -1\times1.3 = -1.3$, and $p_A$ drops from 0.5630 to 0.4523.` },
      { t: "h", text: "Length normalisation in beam search" },
      { t: "p", text: String.raw`Every extra token multiplies the probability by a number $\le 1$. So raw $\log p(y\mid x)$ prefers **short** outputs, and beam search with an end token tends to stop early. The usual fix ranks finished hypotheses by $\log p(y\mid x)/L^{\alpha}$, or GNMT's $\log p(y\mid x)/\big((5+L)/6\big)^{\alpha}$ (Wu et al., 2016) with $\alpha \in [0.6, 1]$.` },
      { t: "steps", items: [
        { title: "Two finished hypotheses from the code below", text: String.raw`"a cat <eos>" with $p = 0.198$ ($L=3$) and "a cat sat <eos>" with $p = 0.162$ ($L=4$).` },
        { title: "Raw score (α = 0)", tex: String.raw`\ln 0.198 = -1.6195 \;>\; \ln 0.162 = -1.8202 \Rightarrow \text{the shorter one wins}` },
        { title: "Per-token score (α = 1)", tex: String.raw`-1.6195/3 = -0.5398 \;<\; -1.8202/4 = -0.4550 \Rightarrow \text{the longer one wins}` },
        { title: "GNMT, α = 0.6", tex: String.raw`-1.6195/(8/6)^{0.6} = -1.3627 \;>\; -1.8202/(9/6)^{0.6} = -1.4271 \Rightarrow \text{shorter still wins}` },
      ] },
      { t: "p", text: "So $\\alpha$ is a real knob, not a bug fix: it trades the model's probability for a preference about length." },
      { t: "h", text: "Why greedy is not optimal" },
      { t: "p", text: String.raw`In the toy model below, $p(\text{the}\mid\text{bos}) = 0.5 > p(\text{a}\mid\text{bos}) = 0.4$, so greedy commits to "the". But every continuation of "the" is uncertain (cat 0.4, dog 0.4, sat 0.2), while "a" is followed by "cat" with probability 0.9. Best greedy sentence: $0.5\times0.4\times0.55 = 0.110$; best overall: $0.4\times0.9\times0.55 = 0.198$. Beam width 2 keeps "a" alive at step 1 and finds it.` },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "sim", id: "decoding", caption: "A real 2-layer character-level Transformer (trained on Alice in Wonderland, Chapter I) decoded live in your browser." },
      { t: "list", items: [
        "**Sampling, top-p = 0.9, T = 1.** Press Step a few times. Outlined bars are probabilities after temperature; filled bars are the renormalised probabilities you actually sample from; the Σ column is the cumulative mass top-p uses. For the default prompt the nucleus holds 8 of 37 characters (90.8% of the mass).",
        "**Drag the temperature** to 0.3 and 1.8 without stepping: outlines sharpen and flatten and the nucleus size changes (temperature comes before truncation), while the model-probability ticks stay put.",
        "**Set top-k = 1, or top-p = 0.05.** Sampling now always picks the arg-max, reproducing greedy. Switch to Greedy to confirm the text is identical.",
        "**Greedy for 24 steps.** Note the total log-probability in the aside. Then try **Beam search, width 3**: after 12 characters the best beam (\"golden key, \", Σ ln p = −3.875) beats greedy's \"golden key a\" (−4.248). Width 1 reproduces greedy exactly.",
        "**Repetition θ = 1.5** in greedy mode changes the arg-max. In a character model almost every letter is already in the prompt, so this blunt tool warps spelling — one reason modern systems penalise sparingly.",
        "Click **seed** repeatedly with T = 1.5 and no truncation: some samples derail into non-words after one unlucky tail character. Turn top-p back on and they stay on track.",
      ] },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "Every decoding rule from the math section in NumPy: temperature, top-k, top-p, min-p and the repetition penalty on the worked example; inverse-CDF sampling checked against 100,000 draws; and greedy vs beam search (with and without length normalisation) on a six-word language model small enough to list every sentence." },
      { t: "code", lang: "python", title: "decoding_from_scratch.py", code: SCRATCH, output: SCRATCH_OUT },
      { t: "p", text: "The filters reproduce the hand-computed $q$ vectors, and sampled frequencies match the nucleus distribution to about 0.002 (Monte Carlo error). Greedy finds a sentence with probability 0.110, while beam width 2 finds the true best (0.198, confirmed by listing them all). And $\\alpha = 1$ flips the choice to the longer sentence with the better per-token score." },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "A batched PyTorch version of temperature/top-k/top-p, checked against Hugging Face's own logits warpers, plus hand-written greedy and beam-search loops checked against `model.generate` on a tiny **randomly initialised** GPT-2 (nothing is downloaded)." },
          { t: "code", lang: "python", title: "decoding_torch.py", code: TORCH, output: TORCH_OUT },
          { t: "list", items: [
            "The kept sets match the library exactly on all 64 rows, and the nucleus size ranges from 1 to 31 tokens — top-p adapts to each distribution.",
            "Greedy decoding of the random model repeats the same token eight times: the repetition loop from the intuition section.",
            "Beam search with width 3 finds a sequence $e^{0.75} \\approx 2.1$ times more probable than greedy's, matching `generate(num_beams=3)` token for token.",
          ] },
          { t: "callout", kind: "tip", text: "In Hugging Face, `generate(do_sample=True, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1)` applies exactly these processors; `num_beams`, `length_penalty` (the exponent $\\alpha$) and `early_stopping` control beam search; recent versions also support `min_p`." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "sample_next_tf.py (not executed in CI)", code: KERAS, output: "Prints a tf.Tensor of 4 sampled token ids (int64), one per row, drawn from the temperature-scaled, top-k and top-p truncated distribution." },
          { t: "p", text: "`tf.random.categorical` samples from unnormalised logits, so truncation means setting removed logits to a large negative number. KerasNLP/KerasHub also ship ready-made `TopKSampler`, `TopPSampler` and `BeamSampler` classes." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "table", head: ["Method", "Rule", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Greedy", "argmax each step", "Deterministic, cheapest, one model call per token", "Short-sighted; repetition loops; bland", "Short factual answers, classification-as-generation, code with tests"],
        ["Beam search (B)", "Keep B best partial sequences", "Finds more probable sequences; deterministic", "B× compute; favours short, generic outputs; still repetitive", "Machine translation, speech recognition, summarisation (closed-ended)"],
        ["Pure sampling", "Draw from p", "True to the model's distribution; varied", "Tail tokens derail the text", "Estimating model statistics, RL rollouts"],
        ["Temperature", "softmax(z/T)", "One knob for confidence vs variety", "Doesn't remove the tail by itself", "Combined with every sampler"],
        ["Top-k", "Keep k most probable", "Simple, cuts the tail", "A fixed k fits neither peaked nor flat distributions", "Story generation (k ≈ 40 was common)"],
        ["Top-p (nucleus)", "Smallest set with mass ≥ p", "Adapts to model confidence", "Nucleus grows large at high T", "Default in most chat APIs (p ≈ 0.9–0.95)"],
        ["Min-p", "Keep p ≥ p_min·p_max", "Robust at high temperature", "Newer, less studied", "Creative writing with high T"],
        ["Repetition / frequency penalty", "Lower logits of seen tokens", "Breaks loops", "Also penalises legitimate repeats (names, code)", "Long open-ended generation"],
        ["Constrained decoding", "Mask tokens that break a grammar", "Guaranteed valid JSON/regex output", "Needs a grammar/state machine", "Structured outputs, tool calls"],
        ["Speculative decoding", "Draft + verify", "Same distribution, 2–3× faster", "Needs a draft model", "Serving (see Efficient Inference)"],
      ] },
      { t: "p", text: "Other names worth knowing: **diverse beam search** (penalise beams that are too alike), **contrastive search**, **typical sampling**, **Mirostat** (aims for a fixed surprise level), and **best-of-n / self-consistency** (sample n answers, then pick by a scorer or majority vote)." },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Chat assistants** expose `temperature` and `top_p` in their APIs: low temperature for facts or code, higher for brainstorming.",
        "**Machine translation and speech recognition** have long used beam search with length normalisation (GNMT used the length penalty above).",
        "**Code completion** uses low temperature or greedy for one suggestion, and higher-temperature sampling for several varied candidates that are then filtered by tests or ranked.",
        "**Reasoning with self-consistency**: sample many chains of thought at moderate temperature and take the majority answer.",
        "**Structured output**: JSON-schema or grammar-constrained decoding masks every token that would break the format.",
        "**RLHF and evaluation**: PPO rollouts are sampled at temperature 1 so the policy-gradient estimate is unbiased (see [Alignment](/learn/m20/alignment)).",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "list", items: [
        "**Quality vs variety**: lowering T or p raises average quality and coherence but cuts variety. No single setting is best for every task.",
        "**Probability is not quality**: a language model's most probable sequence is often short, generic or repetitive. Beam search with a large B can make translation *worse* than a small B.",
        "**Compute**: greedy and sampling cost one forward pass per token; beam search costs B, plus B KV caches.",
        "**Determinism**: greedy and beam are reproducible; sampling only with a fixed seed *and* deterministic kernels (on GPUs, batching and floating-point reduction order can still change results).",
        "**Truncation biases the distribution**: after top-p, kept tokens' probabilities are inflated by $1/\\text{mass}$, so any statistic estimated from truncated samples (e.g. likelihoods in RL) is no longer the model's.",
        "**Penalties are blunt**: they act on token identity, not meaning, so they discourage needed repeats (variable names, numbers) and do nothing about paraphrased loops.",
      ] },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Applying softmax, then dividing by T.** Temperature must divide the *logits*; dividing probabilities and renormalising changes nothing. Fix: `softmax(logits / T)`.",
        "**Off-by-one in top-p.** Keeping only tokens with cumulative mass strictly below $p$ can give an empty set (when the top token alone exceeds $p$) and always drops the token that crosses the line. Fix: keep the crossing token, and always keep the arg-max.",
        "**Not renormalising** after truncation in your own sampler (a CDF ending at 0.9 makes the last kept token soak up the leftover mass). Using $-\\infty$ logits plus a softmax renormalises automatically.",
        "**Multiplying probabilities instead of adding log-probabilities** in beam search: products of many small numbers underflow to 0. Always add log-probabilities.",
        "**Comparing beams of different lengths by raw log-probability**, which always favours stopping early. Use length normalisation or a fixed length.",
        "**Expecting temperature to change greedy output.** It can't — the arg-max survives positive scaling. If greedy output changes with T, sampling is happening somewhere.",
        "**Setting T = 0 literally**: division by zero. Libraries treat it as greedy; your code should too.",
        "**Measuring perplexity on top-p samples** and calling it the model's perplexity; truncation changes the distribution.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What does temperature do, and what happens as T → 0 and T → ∞?", a: "It divides the logits before the softmax. T < 1 sharpens the distribution (lower entropy); T > 1 flattens it. As T → 0 it becomes one-hot on the arg-max (greedy); as T → ∞, uniform. It never changes the ranking of tokens." },
      { t: "qa", level: "beginner", q: "Why is greedy decoding not guaranteed to find the most probable sequence?", a: "It maximises each factor of the product separately. A slightly less likely token now can lead to much more likely continuations later. Example: p(the) = 0.5 vs p(a) = 0.4, but 'a' is followed by 'cat' with 0.9 while 'the' has only uncertain continuations, so the best sentence starts with 'a'." },
      { t: "qa", level: "beginner", q: "What is the difference between top-k and top-p sampling?", a: "Top-k keeps a fixed number k of the most probable tokens. Top-p keeps the smallest set of top tokens whose total probability is at least p, so its size adapts: 1–2 tokens when the model is confident, many when it isn't. Both renormalise the kept probabilities before sampling." },
      { t: "qa", level: "intermediate", q: "Why does beam search need length normalisation, and what does the α parameter control?", a: "Each token multiplies the sequence probability by a number ≤ 1, so log p falls with length and short finished hypotheses beat long ones. Dividing by L^α (or GNMT's ((5+L)/6)^α) compensates: α = 0 means no correction, α = 1 compares average per-token log-probability. It is a length preference, tuned on a validation set." },
      { t: "qa", level: "intermediate", q: "Why does maximisation-based decoding produce repetitive text?", a: "Once a phrase is in the context, attention makes copying it very likely, so each repeat makes the next more likely — a feedback loop. Greedy and beam search always follow the most probable path, so they enter the loop and never leave. Sampling escapes by sometimes picking another token; penalties lower the logits of repeated tokens." },
      { t: "qa", level: "intermediate", q: "In which order are repetition penalty, temperature, top-k and top-p applied, and does it matter?", a: "Usually penalty → temperature → top-k → top-p. It matters: top-p after temperature keeps more tokens at higher T because the distribution is flatter. Penalties applied before temperature get scaled by it. Top-k then top-p applies top-p to the renormalised top-k distribution." },
      { t: "qa", level: "advanced", q: "Beam search improves log-probability yet often hurts open-ended generation quality. Reconcile these.", a: "The model's mode is not the best text. Human text is not the model's most probable sequence (its per-token surprise is higher and varies), and the model gives repetitive or generic sequences far more probability than their share of training-like text. Beam search finds those regions. In closed tasks (translation) the input strongly constrains the output, so the mode is usually a good answer and beam helps — though even there, very large beams can hurt (the 'beam search curse')." },
      { t: "qa", level: "advanced", q: "How would you make sampling reproducible in a production LLM server?", a: "Seed a per-request random generator, apply filters deterministically (stable sorting for ties), and use the same uniform-number stream per step. Even then, GPU logits can differ slightly with batch composition (reduction order), flipping a sample near a CDF boundary. Full reproducibility needs batch-invariant or deterministic kernels, or running the request alone." },
      { t: "qa", level: "advanced", q: "What is the computational cost of beam search with width B compared with greedy, including memory?", a: "Each step runs the model on B sequences instead of 1 (B× FLOPs, though batched on a GPU latency grows less in the memory-bound decode regime), plus a top-B selection over B·|V| candidates. Memory: B KV caches, and when beams are reordered the caches must be gathered/copied. Paged KV caches with copy-on-write sharing of common prefixes reduce this." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Logits (2, 1, 0.5, 0, −1) give p = (0.5630, 0.2071, 0.1256, 0.0762, 0.0280). How many tokens does top-p keep with p = 0.9?", answer: 4, tolerance: 0.5, explain: "Cumulative sums: 0.5630, 0.7701, 0.8958, 0.9720. The first value ≥ 0.9 is the fourth, so 4 tokens are kept." },
        { kind: "numeric", q: "With the same p and top-k = 2, what is the renormalised probability of the top token? (3 decimals)", answer: 0.731, tolerance: 0.001, explain: "0.5630 / (0.5630 + 0.2071) = 0.5630 / 0.7701 = 0.731." },
        { q: "Which statement about temperature is true?", options: ["T > 1 changes the arg-max", "T < 1 increases entropy", "Temperature never changes the order of token probabilities", "T = 1 is greedy decoding"], answer: 2, explain: "Dividing by a positive T keeps the order. T < 1 lowers entropy; greedy corresponds to T → 0." },
        { kind: "numeric", q: "A repetition penalty θ = 2 is applied to a seen token with logit −0.8. What is the new logit?", answer: -1.6, tolerance: 0.001, explain: "Negative logits are multiplied by θ: −0.8 × 2 = −1.6 (lower, so penalised)." },
        { q: "Beam search with width B = 1 is the same as…", options: ["Pure sampling", "Greedy decoding", "Top-k sampling with k = 1 at T = 2 only", "Exhaustive search"], answer: 1, explain: "Keeping one hypothesis and extending it by its best token is greedy decoding (top-k with k = 1 at any temperature is also greedy)." },
        { kind: "numeric", q: "Hypothesis A has log p = −2.4 with 3 tokens; B has log p = −3.0 with 5 tokens. What is B's score with length normalisation α = 1? (A's is −0.8.)", answer: -0.6, tolerance: 0.001, explain: "−3.0 / 5 = −0.6 > −0.8, so with α = 1 the longer hypothesis B wins, although A wins on raw log-probability." },
        { q: "Why does top-p adapt better than top-k?", options: ["It is faster to compute", "Its kept-set size depends on how peaked the distribution is", "It never removes the arg-max while top-k can", "It does not need sorting"], answer: 1, explain: "A confident distribution gives a tiny nucleus, a flat one a large nucleus; top-k always keeps k. (Neither removes the arg-max; both need sorting.)" },
      ] },
      { t: "exercise", title: "Nucleus size by hand", prompt: "For logits $z = (3, 2.5, 1, 0.2, -0.5, -2)$, compute the softmax at $T = 1$ and the number of tokens top-p keeps for $p \\in \\{0.5, 0.9, 0.99\\}$. Then predict how this changes at $T = 0.5$ and $T = 2$.", hint: "$e^{z} = (20.0855, 12.1825, 2.7183, 1.2214, 0.6065, 0.1353)$.", solution: [
        { t: "p", text: String.raw`$\sum e^{z} = 36.9496$, so $p = (0.5436,\ 0.3297,\ 0.0736,\ 0.0331,\ 0.0164,\ 0.0037)$ with cumulative sums $(0.5436,\ 0.8733,\ 0.9469,\ 0.9799,\ 0.9963,\ 1)$. The first cumulative value $\ge 0.5$ is the 1st, $\ge 0.9$ the 3rd, $\ge 0.99$ the 5th: nucleus sizes 1, 3, 5.` },
        { t: "code", lang: "python", title: "nucleus_sizes.py", code: EX_SOL, output: EX_SOL_OUT },
        { t: "p", text: "Lower temperature concentrates the mass, so fewer tokens are needed (1, 2, 3 at T = 0.5). Higher temperature spreads it, so more are needed (2, 4, 6 at T = 2 — for p = 0.99 every token survives). This is the temperature–truncation interaction you can watch in the simulator." },
      ] },
      { t: "exercise", title: "Implement beam search with a KV-cache-friendly interface", prompt: "Write `beam_search(step_fn, bos, eos, width, max_len, alpha)`, where `step_fn(prefix)` returns log-probabilities over the vocabulary. Keep finished hypotheses separately, rank them by $\\log p / L^{\\alpha}$, and test on the six-word model from the scratch section: width 1 must equal greedy, and width 2 must find \"a cat <eos>\".", hint: "Store each hypothesis as (tokens, last_token, logp); sort candidates by logp; move those ending in eos to a finished list.", solution: [
        { t: "p", text: "The `beam_search` function in the scratch code is a full solution. It expands every alive hypothesis by every token, sorts by total log-probability, moves `<eos>` endings to `finished`, keeps the best `width` others alive, and returns `max(finished, key=logp / len**alpha)`. It reproduces greedy for width 1 (\"the cat <eos>\", 0.110) and the listed optimum for width 2 (\"a cat <eos>\", 0.198)." },
        { t: "p", text: "For a KV cache in a real model, store each hypothesis's parent cache index and, after pruning, gather the caches of the surviving parents (Hugging Face calls this `_reorder_cache`)." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Decoding turns per-step distributions $p(\\cdot\\mid x, y_{<t})$ into text. It is separate from training and strongly affects output quality.",
        String.raw`Sequence score: $\log p(y\mid x) = \sum_t \log p(y_t\mid x, y_{<t})$; greedy maximises each term, beam search keeps $B$ hypotheses.`,
        String.raw`Temperature: $p_T \propto \exp(z/T)$; changes entropy, never the ranking.`,
        "Top-k keeps k tokens; top-p keeps the smallest set with mass ≥ p (adaptive); min-p keeps p ≥ p_min·p_max; kept probabilities are renormalised.",
        String.raw`Length normalisation: rank finished beams by $\log p/L^\alpha$; $\alpha$ is a length preference.`,
        "Repetition penalty: lower the logits of seen tokens (÷θ if positive, ×θ if negative).",
        "Closed tasks → greedy/beam; open-ended → sampling with T ≈ 0.7–1 and top-p ≈ 0.9.",
        "Test your code: top-k = 1 and beam width 1 must both reproduce greedy.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Efficient Inference: KV Cache, FlashAttention, MoE](/learn/m20/llm-inference) — every decoding step costs a forward pass; this is how serving systems make it cheap, including speculative decoding, which provably samples from the same distribution.",
        "[Instruction Tuning, RLHF and DPO](/learn/m20/alignment) — changing the distribution itself, not how we sample from it.",
        "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — where the distribution comes from.",
      ] },
    ],
  },
};

export default lesson;
