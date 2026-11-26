import type { Lesson } from "@/lib/types";

const IT_NUMPY = String.raw`import numpy as np

LN2 = np.log(2.0)                       # divide nats by ln 2 to get bits


def entropy(p):
    """H(p) = -sum p log p in nats, with the convention 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return -np.sum(p[nz] * np.log(p[nz]))


def cross_entropy(p, q):
    """H(p, q) = -sum p log q. Infinite if q = 0 somewhere p > 0."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nz = p > 0
    with np.errstate(divide="ignore"):
        return -np.sum(p[nz] * np.log(q[nz]))


def kl(p, q):
    """KL(p || q) = sum p log(p / q)  (equals H(p, q) - H(p))."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nz = p > 0
    with np.errstate(divide="ignore"):
        return np.sum(p[nz] * np.log(p[nz] / q[nz]))


def js(p, q):
    """Jensen-Shannon divergence: average KL to the midpoint m = (p + q) / 2."""
    m = 0.5 * (np.asarray(p, dtype=float) + np.asarray(q, dtype=float))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---- 1. the worked example from the Math section ----
p = np.array([0.5, 0.3, 0.2])          # true distribution
q = np.array([0.25, 0.5, 0.25])        # model distribution
print("H(p)     = %.6f nats = %.6f bits" % (entropy(p), entropy(p) / LN2))
print("H(p,q)   = %.6f nats = %.6f bits" % (cross_entropy(p, q), cross_entropy(p, q) / LN2))
print("KL(p||q) = %.6f nats = %.6f bits" % (kl(p, q), kl(p, q) / LN2))
print("KL(q||p) = %.6f nats   (asymmetric)" % kl(q, p))
print("JS(p,q)  = %.6f nats" % js(p, q))
# the identity holds up to floating-point round-off (~1e-16)
print("identity H(p,q) - H(p) - KL(p||q) = %.1e" % (cross_entropy(p, q) - entropy(p) - kl(p, q)))

# ---- 2. Gibbs' inequality on 100,000 random pairs of distributions ----
rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(5), size=100_000)
Q = rng.dirichlet(np.ones(5), size=100_000)
kls = np.sum(P * np.log(P / Q), axis=1)
print("\nmin KL over 100k random pairs: %.2e  (never negative)" % kls.min())
print("max entropy over 100k random p: %.6f <= log 5 = %.6f" % (
    (-np.sum(P * np.log(P), axis=1)).max(), np.log(5)))

# ---- 3. minimising cross-entropy over softmax logits drives KL to 0 ----
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.zeros(3)                         # q starts uniform
lr = 1.0
for step in range(61):
    qz = softmax(z)
    if step in (0, 5, 20, 60):
        print("step %3d  H(p,q)=%.6f  KL=%.2e  q=%s" % (step, cross_entropy(p, qz), kl(p, qz), np.round(qz, 4)))
    z -= lr * (qz - p)                  # dH(p, softmax(z))/dz = q - p

# ---- 4. forward vs reverse KL with a restricted model ----
x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]

def normal(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

p_mix = 0.5 * normal(x, -2, 0.5) + 0.5 * normal(x, 2, 0.5)   # bimodal "truth"

def kl_grid(a, b):
    """KL(a||b) for densities sampled on the grid (Riemann sum)."""
    m = a > 1e-300
    return np.sum(a[m] * np.log(a[m] / np.maximum(b[m], 1e-300))) * dx

mus = np.linspace(-3, 3, 121)
sigmas = np.linspace(0.2, 3.0, 141)
fwd = np.array([[kl_grid(p_mix, normal(x, m, s)) for s in sigmas] for m in mus])
rev = np.array([[kl_grid(normal(x, m, s), p_mix) for s in sigmas] for m in mus])
i, j = np.unravel_index(fwd.argmin(), fwd.shape)
print("\nforward KL(p||q) best Gaussian: mu=%.2f sigma=%.2f  KL=%.4f" % (mus[i], sigmas[j], fwd[i, j]))
print("  moment matching predicts      mu=0.00 sigma=%.2f" % np.sqrt(0.5 ** 2 + 2 ** 2))
# reverse KL has two equally good optima (mu = -2 and mu = +2); argmin returns the first
i, j = np.unravel_index(rev.argmin(), rev.shape)
print("reverse KL(q||p) best Gaussian: mu=%.2f sigma=%.2f  KL=%.4f  (log 2 = %.4f)" % (mus[i], sigmas[j], rev[i, j], np.log(2)))

# ---- 5. closed-form KL between Gaussians vs Monte Carlo ----
def kl_gauss(m1, s1, m2, s2):
    return np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5

samples = rng.normal(1.0, 0.5, size=1_000_000)          # x ~ q = N(1, 0.5^2)
log_ratio = np.log(normal(samples, 1.0, 0.5)) - np.log(normal(samples, 0.0, 1.0))
print("\nKL(N(1,0.5^2) || N(0,1)): closed form %.6f, Monte Carlo %.6f" % (kl_gauss(1, 0.5, 0, 1), log_ratio.mean()))`;

const IT_NUMPY_OUT = String.raw`H(p)     = 1.029653 nats = 1.485475 bits
H(p,q)   = 1.178350 nats = 1.700000 bits
KL(p||q) = 0.148697 nats = 0.214525 bits
KL(q||p) = 0.137912 nats   (asymmetric)
JS(p,q)  = 0.035263 nats
identity H(p,q) - H(p) - KL(p||q) = -1.7e-16

min KL over 100k random pairs: 1.83e-03  (never negative)
max entropy over 100k random p: 1.609112 <= log 5 = 1.609438
step   0  H(p,q)=1.098612  KL=6.90e-02  q=[0.3333 0.3333 0.3333]
step   5  H(p,q)=1.031109  KL=1.46e-03  q=[0.4815 0.2969 0.2216]
step  20  H(p,q)=1.029653  KL=3.57e-07  q=[0.4999 0.2998 0.2003]
step  60  H(p,q)=1.029653  KL=6.78e-17  q=[0.5 0.3 0.2]

forward KL(p||q) best Gaussian: mu=0.00 sigma=2.06  KL=0.7236
  moment matching predicts      mu=0.00 sigma=2.06
reverse KL(q||p) best Gaussian: mu=-2.00 sigma=0.50  KL=0.6931  (log 2 = 0.6931)

KL(N(1,0.5^2) || N(0,1)): closed form 0.818147, Monte Carlo 0.819394`;

const IT_TORCH = String.raw`import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence

torch.manual_seed(0)

# ---- 1. F.cross_entropy takes raw logits and integer class labels ----
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])
labels = torch.tensor([0, 1])
ce = F.cross_entropy(logits, labels)                       # log_softmax + NLL, averaged
manual = -F.log_softmax(logits, dim=1)[torch.arange(2), labels].mean()
print(f"F.cross_entropy        : {ce.item():.6f}")
print(f"manual -log softmax    : {manual.item():.6f}")

# label smoothing: target becomes (1 - eps) * one_hot + eps / K
ce_ls = F.cross_entropy(logits, labels, label_smoothing=0.1)
target = F.one_hot(labels, 3).float() * 0.9 + 0.1 / 3
manual_ls = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
print(f"with label_smoothing=.1: {ce_ls.item():.6f}  (manual {manual_ls.item():.6f})")

# soft (probability) targets are also accepted directly
print(f"soft-target form       : {F.cross_entropy(logits, target).item():.6f}")

# ---- 2. F.kl_div: the FIRST argument must be LOG-probabilities of the model q ----
p = torch.tensor([[0.5, 0.3, 0.2]])        # target distribution
q = torch.tensor([[0.25, 0.5, 0.25]])      # model distribution
exact = (p * (p / q).log()).sum()
right = F.kl_div(q.log(), p, reduction="batchmean")          # computes KL(p || q)
wrong = F.kl_div(q, p, reduction="batchmean")                # q passed as probabilities: silently wrong
swapped = F.kl_div(p.log(), q, reduction="batchmean")        # arguments swapped: KL(q || p)
log_t = F.kl_div(q.log(), p.log(), reduction="batchmean", log_target=True)
print(f"\nexact KL(p||q)              : {exact.item():.6f}")
print(f"kl_div(q.log(), p)          : {right.item():.6f}")
print(f"kl_div(q, p)   [WRONG]      : {wrong.item():.6f}")
print(f"kl_div(p.log(), q) [swapped]: {swapped.item():.6f}  = KL(q||p)")
print(f"log_target=True             : {log_t.item():.6f}")

# ---- 3. torch.distributions.kl_divergence: closed forms where they exist ----
P, Q = Categorical(probs=p[0]), Categorical(probs=q[0])
print(f"\nCategorical KL(p||q)        : {kl_divergence(P, Q).item():.6f}")
qz = Normal(torch.tensor(1.0), torch.tensor(0.5))      # e.g. a VAE posterior
pz = Normal(torch.tensor(0.0), torch.tensor(1.0))      # the standard normal prior
closed = math.log(1.0 / 0.5) + (0.5 ** 2 + 1.0 ** 2) / 2 - 0.5
print(f"Normal KL(N(1,.5^2)||N(0,1)): {kl_divergence(qz, pz).item():.6f}  (formula {closed:.6f})")
print(f"Normal KL reversed          : {kl_divergence(pz, qz).item():.6f}")

# ---- 4. perplexity of a language model = exp(mean token cross-entropy) ----
vocab, n_tokens = 50, 200
lm_logits = torch.randn(n_tokens, vocab)                  # an untrained "model"
tokens = torch.randint(0, vocab, (n_tokens,))
nll = F.cross_entropy(lm_logits, tokens)
print(f"\nrandom model: cross-entropy {nll.item():.4f} nats/token, perplexity {nll.exp().item():.2f}")
uniform = F.cross_entropy(torch.zeros(n_tokens, vocab), tokens)
print(f"uniform model: cross-entropy {uniform.item():.4f} = log 50, perplexity {uniform.exp().item():.2f}")`;

const IT_TORCH_OUT = String.raw`F.cross_entropy        : 0.318540
manual -log softmax    : 0.318540
with label_smoothing=.1: 0.436873  (manual 0.436873)
soft-target form       : 0.436873

exact KL(p||q)              : 0.148697
kl_div(q.log(), p)          : 0.148697
kl_div(q, p)   [WRONG]      : -1.354653
kl_div(p.log(), q) [swapped]: 0.137912  = KL(q||p)
log_target=True             : 0.148697

Categorical KL(p||q)        : 0.148697
Normal KL(N(1,.5^2)||N(0,1)): 0.818147  (formula 0.818147)
Normal KL reversed          : 2.806853

random model: cross-entropy 4.5798 nats/token, perplexity 97.49
uniform model: cross-entropy 3.9120 = log 50, perplexity 50.00`;

const IT_KERAS = String.raw`import numpy as np
import tensorflow as tf

y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype="float32")

# Categorical cross-entropy from logits, with and without label smoothing
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
cce_ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
print("CE:", float(cce(y_true, logits)), " CE with smoothing:", float(cce_ls(y_true, logits)))

# Sparse labels (integers) instead of one-hot rows
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("sparse CE:", float(scce(np.array([0, 1]), logits)))

# KL divergence: Keras takes PROBABILITIES for both arguments, KL(y_true || y_pred)
p = np.array([[0.5, 0.3, 0.2]], dtype="float32")
q = np.array([[0.25, 0.5, 0.25]], dtype="float32")
print("KL(p||q):", float(tf.keras.losses.KLDivergence()(p, q)))`;

const EX_CODE = String.raw`import numpy as np


def log_softmax(z):
    """log q = z - logsumexp(z), computed without overflow."""
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def cross_entropy_logits(z, p):
    """H(p, softmax(z)) for soft targets p (rows sum to 1)."""
    return -(p * log_softmax(z)).sum(axis=-1)


def kl_logits(z, p):
    """KL(p || softmax(z)) = H(p, q) - H(p); terms with p = 0 contribute 0."""
    nz = p > 0
    h_p = -np.sum(np.where(nz, p * np.log(np.where(nz, p, 1.0)), 0.0), axis=-1)
    return cross_entropy_logits(z, p) - h_p


z = np.array([[1000.0, 0.0, -1000.0],          # huge logits: naive softmax overflows
              [2.0, 1.0, 0.1]])
p = np.array([[1.0, 0.0, 0.0],
              [0.7, 0.2, 0.1]])
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    naive = -(p * np.log(np.exp(z) / np.exp(z).sum(axis=1, keepdims=True))).sum(axis=1)
print("naive CE :", naive)
print("stable CE:", cross_entropy_logits(z, p))
print("KL       :", kl_logits(z, p))`;

const EX_CODE_OUT = String.raw`naive CE : [       nan 0.80703002]
stable CE: [-0.          0.80703002]
KL       : [0.         0.00521146]`;

const lesson: Lesson = {
  id: "m02/information-theory",
  title: "Entropy, Cross-Entropy and KL Divergence",
  summary:
    "The maths of surprise behind deep learning losses: surprisal, entropy, joint and conditional entropy, mutual information, cross-entropy, KL divergence (and why it is one-sided), forward vs reverse KL, the Gaussian KL in VAEs, perplexity and Jensen–Shannon — every number worked out, plus a simulator.",
  minutes: 70,
  level: "intermediate",
  prerequisites: ["m02/probability", "m02/mle-map"],
  next: ["m05/classification-losses", "m14/vae", "m13/language-modeling"],
  terms: ["entropy", "surprisal", "cross-entropy", "kl-divergence", "mutual-information", "perplexity", "jensen-shannon", "label-smoothing"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Almost every classifier you will train minimises **cross-entropy**. Variational autoencoders contain a **KL divergence** term, language models are compared by **perplexity**, and the original GAN objective is secretly a **Jensen–Shannon divergence**. All of these come from **information theory** — the maths of uncertainty and communication, founded by Claude Shannon in his 1948 paper \"A Mathematical Theory of Communication\".",
      },
      {
        t: "p",
        text: "Shannon asked a practical question: on average, how many bits do you need to send messages from a source whose outcomes have known probabilities? His answer, the **entropy** of the source, became the standard measure of uncertainty. Two facts matter most for deep learning. First, if you encode data using a *wrong* distribution $q$ instead of the true one $p$, you pay extra bits; that extra cost is the **KL divergence** $\\mathrm{KL}(p\\,\\|\\,q)$. Second, the total cost, the **cross-entropy** $H(p,q)$, can be estimated from samples of $p$ alone — and a training set is exactly that.",
      },
      {
        t: "p",
        text: "So when a network minimises cross-entropy, it pulls its predicted distribution $q$ as close as possible (in the KL sense) to the distribution that made the labels. We build this up from scratch, prove the key inequalities, work every number by hand, and cover the practical traps: units, numerical stability and PyTorch's conventions.",
      },
      {
        t: "callout",
        kind: "key",
        text: String.raw`Surprise of an outcome $x$ with probability $p(x)$: $-\log p(x)$. Entropy: average surprise when outcomes come from $p$. Cross-entropy: average surprise when outcomes come from $p$ but you *believe* $q$. KL divergence: the difference, $H(p,q) - H(p) \ge 0$.`,
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Probability, Random Variables and Distributions](/learn/m02/probability) — distributions $p(x)$ (non-negative, sum to 1), expectations $\\mathbb{E}_{x\\sim p}[f(x)] = \\sum_x p(x) f(x)$ (the average of $f$ when $x$ comes from $p$), joint and conditional probabilities, and the Gaussian density.",
          "Logarithms — $\\log(ab) = \\log a + \\log b$, $\\log(1/a) = -\\log a$, change of base $\\log_2 x = \\ln x / \\ln 2$, and that $\\log$ is concave (it curves downward). No other calculus until the gradient derivations.",
          "[Maximum Likelihood and MAP Estimation](/learn/m02/mle-map) — helpful, not essential. Here we show minimising cross-entropy equals maximising likelihood; that lesson explains why maximum likelihood is sensible.",
          "Softmax — turns a vector of raw scores, the **logits** $z$, into probabilities $q_i = e^{z_i}/\\sum_j e^{z_j}$.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Weather reports and surprise",
        text: "In a desert town that is sunny 99 days out of 100, \"sunny again\" tells you almost nothing. \"Rain today\" is big news. Information is **surprise**: the rarer the event, the more you learn when it happens. Entropy is how surprised you are *on average*: desert weather has low entropy (predictable); a city where sun and rain are equally likely has high entropy. Now picture a forecaster who believes the desert rains half the time. Every sunny day mildly surprises them. Their average surprise is higher than the desert's true entropy — the excess is the KL divergence between truth and belief.",
      },
      {
        t: "p",
        text: "Three ideas carry the whole lesson. **Surprisal** $-\\log p(x)$ is large for rare outcomes and zero for certain ones. **Entropy** averages surprisal over outcomes from the true distribution: the unpredictability you can never remove. **Cross-entropy** averages the surprisal your *model* gives, still over outcomes from the truth. A model can at best match the truth, so cross-entropy is never below entropy. The gap is the KL divergence.",
      },
      {
        t: "p",
        text: "A coding view makes the numbers concrete. The best code gives an outcome of probability $p$ a codeword of about $-\\log_2 p$ bits: common outcomes get short codes (in Morse code, E is a single dot). Entropy in bits is the shortest possible average message length. Design your code for the wrong $q$ and the average becomes $H(p,q)$: you waste $\\mathrm{KL}(p\\,\\|\\,q)$ bits per message.",
      },
      {
        t: "p",
        text: "KL divergence is **not** a distance. Treating a rare event as impossible ($q \\approx 0$ where $p > 0$) is hugely expensive, because the surprisal has no upper limit. Putting some probability on events that never happen costs only a little. This one-sidedness is why the *direction* of the KL you minimise changes what a model learns.",
      },
    ],
    mechanism: [
      { t: "p", text: "How to compute the core quantities for a true distribution $p$ and a model $q$ over the same $K$ outcomes ($p_i$ is the probability of outcome $i$):" },
      {
        t: "steps",
        items: [
          { title: "Choose a log base", text: "Base 2 gives **bits**; natural log gives **nats** (1 nat = $1/\\ln 2 \\approx 1.4427$ bits). Deep learning code uses natural logs, so losses are in nats." },
          { title: "Surprisal of each outcome under each distribution", text: "Compute $-\\log p_i$ and $-\\log q_i$. If $q_i = 0$, the surprisal is $+\\infty$.", tex: String.raw`s_p(i) = -\log p_i, \qquad s_q(i) = -\log q_i` },
          { title: "Entropy: weight p's surprisal by p", text: "Terms with $p_i = 0$ count as 0 (because $p\\log p \\to 0$ as $p\\to 0$).", tex: String.raw`H(p) = \sum_i p_i\,(-\log p_i)` },
          { title: "Cross-entropy: weight q's surprisal by p", text: "Outcomes still come from $p$; only the belief (the code) is $q$.", tex: String.raw`H(p,q) = \sum_i p_i\,(-\log q_i)` },
          { title: "KL divergence: the difference, term by term", text: "Single terms are negative where $q_i > p_i$, but the sum is never negative (proved in the Math section).", tex: String.raw`\mathrm{KL}(p\,\|\,q) = \sum_i p_i \log\frac{p_i}{q_i} = H(p,q) - H(p)` },
          { title: "Optionally swap roles", text: "Reverse KL $\\mathrm{KL}(q\\,\\|\\,p)$ weights by $q$ instead — usually a different number." },
        ],
      },
      { t: "p", text: "A classifier does this for every training example, with two shortcuts:" },
      {
        t: "steps",
        items: [
          { title: "The network outputs logits", text: "A vector $z \\in \\mathbb{R}^K$ ($K$ real numbers) per input; $q = \\mathrm{softmax}(z)$." },
          { title: "The target is usually one-hot", text: "For label $c$, $p = e_c$ (1 at position $c$, 0 elsewhere), so $H(p) = 0$ and cross-entropy shrinks to one term.", tex: String.raw`H(e_c, q) = -\log q_c = -z_c + \log\sum_j e^{z_j}` },
          { title: "Average over the batch", text: "The loss is the mean of $-\\log q_{c}$ over examples — a sample estimate of $\\mathbb{E}_{(x,c)\\sim\\text{data}}[-\\log q(c\\mid x)]$." },
          { title: "Backpropagate", text: "The gradient with respect to the logits is just $q - p$ (prediction minus target), which starts the backward pass (see [Backpropagation](/learn/m04/backpropagation))." },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        text: "Never compute $\\log(\\mathrm{softmax}(z))$ in two steps in real code: $e^{z}$ overflows for large logits, and tiny probabilities give $\\log 0 = -\\infty$. Use a fused **log-softmax** (log-sum-exp with the maximum subtracted). Every framework's cross-entropy does this inside.",
      },
    ],
    math: [
      { t: "h", text: "Information content (surprisal)" },
      {
        t: "p",
        text: "We want a measure $I(x)$ of how much we learn when an outcome of probability $p(x)$ happens. Three rules: a certain event teaches nothing ($I = 0$ when $p = 1$); rarer events teach more; and for **independent** events information adds, $I(x, y) = I(x) + I(y)$ when $p(x,y) = p(x)p(y)$. Only logarithms satisfy all three:",
      },
      { t: "math", tex: String.raw`I(x) = -\log_b p(x) = \log_b \frac{1}{p(x)}`, caption: "$b = 2$: bits (shannons). $b = e$: nats. Converting: value in bits $=$ value in nats $/ \\ln 2$." },
      {
        t: "p",
        text: "Examples: probability $1/8$ carries $-\\log_2(1/8) = 3$ bits; $0.9$ carries $0.152$ bits; $0.01$ carries $6.644$ bits. A fair coin flip carries $1$ bit $= \\ln 2 = 0.693147$ nats.",
      },
      { t: "h", text: "Entropy" },
      { t: "math", tex: String.raw`H(p) = \mathbb{E}_{x\sim p}\left[-\log p(x)\right] = -\sum_{x} p(x)\log p(x), \qquad 0\log 0 \equiv 0` },
      {
        t: "p",
        text: "$x$ runs over the possible outcomes. Two edge cases. **Certain**: if one outcome has probability 1, every term is $1\\cdot\\log 1 = 0$ or $0\\log 0 = 0$, so $H = 0$. **Uniform** over $K$ outcomes: $H = -\\sum_x \\frac1K\\log\\frac1K = \\log K$. A coin with $p(\\text{heads}) = 0.9$ has $H = -(0.9\\log_2 0.9 + 0.1\\log_2 0.1) = 0.469$ bits; a fair die has $\\log_2 6 = 2.585$ bits.",
      },
      {
        t: "p",
        text: "**Claim:** $0 \\le H(p) \\le \\log K$, with the maximum only for the uniform distribution $u$. The lower bound holds because each term $-p\\log p \\ge 0$ for $p \\in [0,1]$. For the upper bound, use the KL divergence (defined below) from $p$ to $u$:",
      },
      { t: "math", tex: String.raw`\mathrm{KL}(p\,\|\,u) = \sum_x p(x)\log\frac{p(x)}{1/K} = \log K - H(p) \;\ge\; 0 \quad\Longrightarrow\quad H(p) \le \log K` },
      { t: "h", text: "Joint entropy, conditional entropy and mutual information" },
      {
        t: "p",
        text: "For two random variables $X, Y$ with joint distribution $p(x,y)$, the **joint entropy** is the entropy of the pair. The **conditional entropy** $H(Y\\mid X)$ is the uncertainty about $Y$ left once you know $X$:",
      },
      { t: "math", tex: String.raw`H(X,Y) = -\sum_{x,y} p(x,y)\log p(x,y), \qquad H(Y\mid X) = -\sum_{x,y} p(x,y)\log p(y\mid x) = H(X,Y) - H(X)` },
      {
        t: "p",
        text: "The second identity is the **chain rule of entropy**: uncertainty of the pair = uncertainty of $X$ + what is left of $Y$ given $X$. **Mutual information** is how much knowing one variable cuts uncertainty about the other:",
      },
      { t: "math", tex: String.raw`I(X;Y) = H(Y) - H(Y\mid X) = H(X) + H(Y) - H(X,Y) = \mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big) \;\ge 0` },
      {
        t: "p",
        text: "It is zero exactly when $X$ and $Y$ are independent. **Example.** $X$ = cloudy/clear, $Y$ = rain/dry, with $p(\\text{cloudy, rain}) = p(\\text{clear, dry}) = 0.4$ and $p(\\text{cloudy, dry}) = p(\\text{clear, rain}) = 0.1$. Each variable alone is $(0.5, 0.5)$, so $H(X) = H(Y) = 1$ bit. $H(X,Y) = -(2\\cdot0.4\\log_2 0.4 + 2\\cdot 0.1\\log_2 0.1) = 1.721928$ bits, so $H(Y\\mid X) = 0.721928$ bits (the entropy of $(0.8, 0.2)$, rain given either sky) and $I(X;Y) = 1 + 1 - 1.721928 = 0.278072$ bits. Mutual information shows up in representation learning (contrastive objectives are motivated as bounds on it) and feature selection.",
      },
      { t: "h", text: "Cross-entropy and KL divergence" },
      {
        t: "math",
        tex: String.raw`H(p,q) = -\sum_x p(x)\log q(x), \qquad \mathrm{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}, \qquad H(p,q) = H(p) + \mathrm{KL}(p\,\|\,q)`,
        caption: "Conventions: $0\\log(0/q) = 0$; $p\\log(p/0) = +\\infty$ for $p > 0$. So $\\mathrm{KL}(p\\|q) < \\infty$ requires $q(x) > 0$ wherever $p(x) > 0$.",
      },
      {
        t: "p",
        text: "The right-hand identity comes from writing $\\log\\frac{p}{q} = \\log p - \\log q$ inside the sum. KL is also called **relative entropy**. $\\mathrm{KL}(p\\,\\|\\,q)$ reads \"KL from $q$ to $p$\"; the first argument is the distribution that weights the average.",
      },
      { t: "h", text: "Non-negativity: Gibbs' inequality via Jensen" },
      {
        t: "p",
        text: "**Jensen's inequality** for a concave (downward-curving) function $f$ such as $\\log$: $\\mathbb{E}[f(Y)] \\le f(\\mathbb{E}[Y])$ — the average of the log is at most the log of the average — with equality only if $Y$ is constant (where it has probability). Let $S = \\{x : p(x) > 0\\}$ and use $Y = q(x)/p(x)$ for $x \\sim p$:",
      },
      { t: "math", tex: String.raw`-\mathrm{KL}(p\,\|\,q) = \sum_{x\in S} p(x)\log\frac{q(x)}{p(x)} \;\le\; \log\sum_{x\in S} p(x)\frac{q(x)}{p(x)} = \log\sum_{x\in S} q(x) \;\le\; \log 1 = 0` },
      {
        t: "p",
        text: "So $\\mathrm{KL}(p\\,\\|\\,q) \\ge 0$ (**Gibbs' inequality**), with equality only if $q(x)/p(x)$ is constant on $S$ and $q$ puts no mass outside $S$ — i.e. $q = p$. Hence $H(p,q) \\ge H(p)$: nothing beats coding with the true distribution.",
      },
      { t: "h", text: "Fully worked example" },
      {
        t: "p",
        text: "Take $p = (0.5,\\ 0.3,\\ 0.2)$ and $q = (0.25,\\ 0.5,\\ 0.25)$. The model's code lengths are whole bits: $-\\log_2 q = (2, 1, 2)$.",
      },
      {
        t: "table",
        head: ["i", "p_i", "q_i", "−log₂ p_i", "−p_i log₂ p_i", "−log₂ q_i", "−p_i log₂ q_i", "p_i log₂(p_i/q_i)"],
        rows: [
          ["1", "0.5", "0.25", "1.000000", "0.500000", "2", "1.0", "0.500000"],
          ["2", "0.3", "0.5", "1.736966", "0.521090", "1", "0.3", "−0.221090"],
          ["3", "0.2", "0.25", "2.321928", "0.464386", "2", "0.4", "−0.064386"],
          ["sum", "1", "1", "", "H(p) = 1.485475", "", "H(p,q) = 1.7", "KL = 0.214525"],
        ],
      },
      {
        t: "steps",
        items: [
          { title: "Entropy", tex: String.raw`H(p) = 0.5 + 0.521090 + 0.464386 = 1.485475\ \text{bits} = 1.029653\ \text{nats}` },
          { title: "Cross-entropy", tex: String.raw`H(p,q) = 0.5\cdot2 + 0.3\cdot1 + 0.2\cdot2 = 1.7\ \text{bits} = 1.178350\ \text{nats}` },
          { title: "KL divergence (two ways)", tex: String.raw`\mathrm{KL}(p\,\|\,q) = 1.7 - 1.485475 = 0.500000 - 0.221090 - 0.064386 = 0.214525\ \text{bits} = 0.148697\ \text{nats}` },
          { title: "Reverse direction", tex: String.raw`\mathrm{KL}(q\,\|\,p) = 0.25\ln\tfrac{0.25}{0.5} + 0.5\ln\tfrac{0.5}{0.3} + 0.25\ln\tfrac{0.25}{0.2} = -0.173287 + 0.255413 + 0.055786 = 0.137912\ \text{nats}` },
        ],
      },
      {
        t: "p",
        text: "In plain words: symbols arrive with frequencies $p$, but you use the code built for $q$, so you spend 1.7 bits per symbol instead of the best 1.485 — 0.215 bits wasted. Two KL terms are negative: $q$ over-weights outcomes 2 and 3, which *saves* bits there, but the big loss on outcome 1 (given half its true weight) more than cancels it.",
      },
      { t: "h", text: "Asymmetry and forward vs reverse KL" },
      {
        t: "p",
        text: "For $p = (0.5, 0.5)$ and $q = (0.9, 0.1)$: $\\mathrm{KL}(p\\,\\|\\,q) = 0.5\\ln\\frac{0.5}{0.9} + 0.5\\ln\\frac{0.5}{0.1} = 0.510826$ nats but $\\mathrm{KL}(q\\,\\|\\,p) = 0.9\\ln\\frac{0.9}{0.5} + 0.1\\ln\\frac{0.1}{0.5} = 0.368064$ nats. Forward is larger because it is weighted by $p$, which cares a lot about the outcome $q$ nearly rules out.",
      },
      {
        t: "p",
        text: "When the model $q_\\theta$ ($\\theta$ = its parameters) comes from a **restricted** family that cannot match $p$ exactly, the direction decides the compromise:",
      },
      {
        t: "list",
        items: [
          String.raw`**Forward KL** $\min_\theta \mathrm{KL}(p\,\|\,q_\theta)$ — averaged over $p$. Where $p(x) > 0$ and $q_\theta(x) \to 0$, the term $p\log(p/q)$ explodes, so $q$ must cover *all* of $p$'s mass: **mode-covering** (also zero-avoiding or mass-covering). It needs only samples from $p$ — this is maximum likelihood.`,
          String.raw`**Reverse KL** $\min_\theta \mathrm{KL}(q_\theta\,\|\,p)$ — averaged over $q$. Where $q(x) > 0$ but $p(x) \approx 0$, the term $q\log(q/p)$ explodes, so $q$ avoids places where $p$ is small and may ignore whole modes (peaks): **mode-seeking** (zero-forcing). It needs $p$'s density (up to a constant), not samples; variational inference uses it.`,
        ],
      },
      {
        t: "p",
        text: "**Numbers** (grid search in the From-Scratch section): $p = 0.5\\,\\mathcal N(-2, 0.5^2) + 0.5\\,\\mathcal N(2, 0.5^2)$ — two bumps — fitted with one Gaussian $q = \\mathcal N(\\mu, \\sigma^2)$. Forward KL's best is $\\mu = 0$, $\\sigma = 2.06$: exactly the **moment-matched** Gaussian (same mean and variance, $\\sigma^2 = 0.5^2 + 2^2 = 4.25$), sitting in the empty valley, with $\\mathrm{KL} = 0.7236$. Reverse KL's best is $\\mu = \\pm 2$, $\\sigma = 0.5$: one bump fitted perfectly, the other ignored, with $\\mathrm{KL} = 0.6931 = \\ln 2$ (where $q$ has mass, $p \\approx q/2$, so $\\log(q/p) \\approx \\ln 2$).",
      },
      { t: "h", text: "Why minimising cross-entropy is maximum likelihood" },
      {
        t: "p",
        text: "Let $p_{\\text{data}}$ be the dataset $\\{x_1,\\dots,x_N\\}$ seen as a distribution (probability $1/N$ on each example) and $q_\\theta$ a model. Then",
      },
      { t: "math", tex: String.raw`H(p_{\text{data}}, q_\theta) = -\frac1N\sum_{n=1}^N \log q_\theta(x_n) = -\frac1N \log \prod_{n=1}^N q_\theta(x_n)` },
      {
        t: "p",
        text: "— the negative average log-likelihood. $H(p_{\\text{data}})$ does not depend on $\\theta$, so $\\min_\\theta H(p_{\\text{data}}, q_\\theta)$, $\\min_\\theta \\mathrm{KL}(p_{\\text{data}}\\,\\|\\,q_\\theta)$ and $\\max_\\theta \\prod_n q_\\theta(x_n)$ have the **same solution**; they differ only by a constant and a sign. The same holds for classifiers $q_\\theta(y\\mid x)$. See [Maximum Likelihood and MAP](/learn/m02/mle-map) for why maximum likelihood is a good estimator.",
      },
      { t: "h", text: "One-hot labels, softmax, and the q − p gradient" },
      {
        t: "p",
        text: "With one-hot target $e_c$ and $q = \\mathrm{softmax}(z)$, the loss is $\\ell(z) = -z_c + \\log\\sum_j e^{z_j}$ and its gradient is $\\partial\\ell/\\partial z_j = q_j - [j = c]$ ($[j = c]$ is 1 if $j = c$, else 0), i.e. $\\nabla_z \\ell = q - p$. Soft targets give the same: $\\nabla_z H(p, \\mathrm{softmax}(z)) = q - p$ (using $\\sum_i p_i = 1$). Example: $z = (2, 1, 0.1)$ gives $q = (0.659001, 0.242433, 0.098566)$; with $c = 1$ the loss is $-\\ln 0.659001 = 0.417030$ nats and the gradient is $(-0.340999,\\ 0.242433,\\ 0.098566)$.",
      },
      { t: "h", text: "Label smoothing" },
      {
        t: "p",
        text: "One-hot cross-entropy reaches its minimum, 0, only as $q_c \\to 1$, which needs the logit gap $z_c - z_j \\to \\infty$: the network is always pushed toward more confidence. **Label smoothing** with a small $\\varepsilon$ swaps the target for $p^{\\text{ls}} = (1-\\varepsilon)e_c + \\varepsilon/K$. Now $H(p^{\\text{ls}}) > 0$, the loss $H(p^{\\text{ls}}, q) = H(p^{\\text{ls}}) + \\mathrm{KL}(p^{\\text{ls}}\\,\\|\\,q)$ is smallest at $q = p^{\\text{ls}}$, and the best logit gap is finite: $\\log\\frac{p^{\\text{ls}}_c}{p^{\\text{ls}}_j}$.",
      },
      {
        t: "p",
        text: "With $K = 3$, $\\varepsilon = 0.1$: $p^{\\text{ls}} = (0.933333, 0.033333, 0.033333)$, $H(p^{\\text{ls}}) = 0.291140$ nats (a floor the loss never goes below), and the best gap is $\\ln 28 = 3.332205$. For $z = (2, 1, 0.1)$ above, $H(p^{\\text{ls}}, q) = 0.513697 = 0.291140 + 0.222557$ nats. More in [Classification Losses](/learn/m05/classification-losses).",
      },
      { t: "h", text: "KL divergence between two Gaussians" },
      {
        t: "p",
        text: "For continuous distributions, sums become integrals: $\\mathrm{KL}(p\\,\\|\\,q) = \\int p(x)\\log\\frac{p(x)}{q(x)}\\,dx$. It stays non-negative and unchanged if you reparametrise $x$ — unlike *differential* entropy, which can be negative. For 1-D Gaussians $p = \\mathcal N(\\mu_1, \\sigma_1^2)$ and $q = \\mathcal N(\\mu_2, \\sigma_2^2)$, plugging in the densities and using $\\mathbb{E}_p[(x-\\mu_2)^2] = \\sigma_1^2 + (\\mu_1-\\mu_2)^2$ gives",
      },
      { t: "math", tex: String.raw`\mathrm{KL}\big(\mathcal N(\mu_1,\sigma_1^2)\,\|\,\mathcal N(\mu_2,\sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12` },
      {
        t: "p",
        text: "Sketch: $\\log\\frac{p(x)}{q(x)} = \\log\\frac{\\sigma_2}{\\sigma_1} - \\frac{(x-\\mu_1)^2}{2\\sigma_1^2} + \\frac{(x-\\mu_2)^2}{2\\sigma_2^2}$; averaging under $p$ turns the middle term into $-\\frac12$ and the last into $\\frac{\\sigma_1^2 + (\\mu_1 - \\mu_2)^2}{2\\sigma_2^2}$. In a [variational autoencoder](/learn/m14/vae) the encoder outputs $\\mu$ and $\\sigma$ per latent dimension and the prior is $\\mathcal N(0,1)$, so the KL term per dimension is",
      },
      { t: "math", tex: String.raw`\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1)\big) = \tfrac12\left(\mu^2 + \sigma^2 - 1 - \log\sigma^2\right)` },
      {
        t: "p",
        text: "Example: $\\mu = 1$, $\\sigma = 0.5$: $\\frac12(1 + 0.25 - 1 - \\ln 0.25) = \\frac12(0.25 + 1.386294) = 0.818147$ nats (Monte Carlo with $10^6$ samples: $0.819394$). The reverse, $\\mathrm{KL}(\\mathcal N(0,1)\\,\\|\\,\\mathcal N(1, 0.25))$, is $2.806853$ — a narrow $q$ pays heavily for missing the wide $p$'s mass.",
      },
      { t: "h", text: "Perplexity" },
      {
        t: "p",
        text: "A language model gives probability $q(w_t \\mid w_{<t})$ to each token $w_t$ of a held-out text of $T$ tokens, given the tokens before it. Its average cross-entropy is $\\bar H = -\\frac1T\\sum_t \\log q(w_t\\mid w_{<t})$ and its **perplexity** is",
      },
      { t: "math", tex: String.raw`\mathrm{PPL} = \exp(\bar H) \quad (\text{or } 2^{\bar H} \text{ if } \bar H \text{ is in bits — the same number})` },
      {
        t: "p",
        text: "In plain words: perplexity is the \"effective number of choices\" the model is torn between — the size of a uniform distribution with the same cross-entropy. A uniform model over $V$ words has perplexity exactly $V$. Example: token probabilities $(0.25, 0.1, 0.5, 0.05)$ give $\\bar H = 1.844440$ nats $= 2.660964$ bits per token, and $\\mathrm{PPL} = e^{1.844440} = 6.3246$ (the geometric mean of $1/q$). Being an exponential, small loss drops are big perplexity drops: a loss of 2.3 nats is perplexity 9.97. Compare perplexities only with the **same tokenizer** and test set. See [Language Modelling and Perplexity](/learn/m13/language-modeling).",
      },
      { t: "h", text: "Jensen–Shannon divergence" },
      { t: "math", tex: String.raw`\mathrm{JS}(p,q) = \tfrac12\mathrm{KL}(p\,\|\,m) + \tfrac12\mathrm{KL}(q\,\|\,m), \qquad m = \tfrac12(p+q)` },
      {
        t: "p",
        text: "JS compares both distributions to their average $m$. It is symmetric, always finite ($m > 0$ wherever $p$ or $q$ is), and bounded: $0 \\le \\mathrm{JS} \\le \\ln 2$ nats $= 1$ bit, reached when the two never overlap (disjoint supports). For the worked example, $\\mathrm{JS} = 0.035263$ nats. Its square root is a true metric. Goodfellow et al. (2014) showed the original GAN objective, with an optimal discriminator, equals $2\\,\\mathrm{JS}(p_{\\text{data}}, p_G) - \\log 4$. JS sticks at $\\ln 2$ for non-overlapping distributions, so early gradients vanish — this motivated the Wasserstein GAN ([GANs](/learn/m14/gans)).",
      },
    ],
    visual: [
      {
        t: "p",
        text: "The simulator holds two distributions over $K$ outcomes: $p$ (truth, accent bars) and $q$ (model, clay bars), each chart showing the other as dashes. Every number — the six measures, the stacked bar and the per-outcome table — is recomputed from the bars you see, in bits or nats.",
      },
      { t: "sim", id: "entropy", caption: "Drag bars (or use the fine-edit sliders) to edit p and q; Play trains q by gradient descent on its softmax logits." },
      {
        t: "list",
        items: [
          "Set $p$ to **uniform**: $H(p)$ equals the gold $\\log K$ mark. Switch $p$ to **one-hot**: $H(p) = 0$. Any other shape lands between the two.",
          "With $p$ bimodal and $q$ uniform, check that the stacked bar's accent part ($H(p)$) plus clay part ($\\mathrm{KL}$) equals $H(p,q)$, and that the table's last column sums to KL even with some negative entries.",
          "Set $q$ to **one-hot** while $p$ is not: $\\mathrm{KL}(p\\,\\|\\,q) = \\infty$, because some outcome with $p_i > 0$ gets $q_i = 0$. $\\mathrm{KL}(q\\,\\|\\,p)$ stays finite and JS stays below 1 bit.",
          "Press **Play** with the *free* model and *forward* objective: $q$ converges to $p$ and KL falls toward 0 (the sparkline). That is what training a softmax classifier does for one input.",
          "Switch to the **unimodal (μ, σ)** model — a discretised Gaussian with two parameters, which cannot match a bimodal $p$. Train with **forward** KL: the bump moves between the modes and widens to cover both (mode-covering). Reset, choose **reverse** KL and train: the bump settles narrowly on one mode (mode-seeking). Move μ to the other side and retrain to get the other mode.",
          "Compare final values in nats (default bimodal $p$, $K = 8$). After forward training the bump is very wide: $\\mathrm{KL}(p\\,\\|\\,q) \\approx 0.51$, $\\mathrm{KL}(q\\,\\|\\,p) \\approx 1.15$. After reverse training: $\\mathrm{KL}(q\\,\\|\\,p) \\approx 0.693 = \\ln 2$ (1 bit, as in the continuous example) but $\\mathrm{KL}(p\\,\\|\\,q) \\approx 16.6$, because the ignored mode gets almost no probability. Each objective is good by its own measure and bad by the other's. (Starting μ near the centre can leave reverse KL in a wide local optimum: with a multimodal $p$ it usually has several, one per mode plus maybe a wide compromise.)",
        ],
      },
      {
        t: "callout",
        kind: "note",
        text: "Honest details: training is plain gradient descent on the logits (free model) or on $(\\mu, \\log\\sigma)$ via the chain rule (unimodal model, each step clipped for stability). If $q$ has exact zeros, its logits are floored at $\\log 10^{-6}$; if $p$ has zeros, reverse-KL training floors $\\log p$ at $\\log 10^{-6}$. The narration says when this happens.",
      },
    ],
    scratch: [
      {
        t: "p",
        text: "NumPy code for the four measures plus Jensen–Shannon, then experiments: the worked example, a random check of Gibbs' inequality and the entropy bound, gradient descent on softmax logits (KL → 0), forward vs reverse KL fits of one Gaussian to two bumps, and the Gaussian KL formula against Monte Carlo.",
      },
      { t: "code", lang: "python", title: "information_theory.py", code: IT_NUMPY, output: IT_NUMPY_OUT },
      {
        t: "list",
        items: [
          "`entropy`, `cross_entropy` and `kl` skip $p = 0$ entries to get $0\\log 0 = 0$; a zero in $q$ where $p > 0$ gives `inf`, the correct answer.",
          "The identity check prints $-1.7\\times10^{-16}$, not 0: floating-point round-off.",
          "Among 100,000 random pairs the smallest KL is $1.83\\times10^{-3}$ — positive, as Gibbs' inequality says — and no random distribution beats $\\log 5$ in entropy.",
          "The softmax experiment uses the gradient $q - p$; KL drops from $6.9\\times10^{-2}$ to $3.6\\times10^{-7}$ in 20 steps.",
          "The Gaussian fit is a grid search; the two mirror-image reverse solutions ($\\mu = \\pm2$) are equally good, and the code reports the first.",
        ],
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "Three tools cover almost everything: `F.cross_entropy` (logits in; class indices or probabilities as targets; optional `label_smoothing`), `F.kl_div` (distillation and other distribution matching) and `torch.distributions.kl_divergence` (closed forms between distribution objects).",
              },
              { t: "code", lang: "python", title: "information_theory_pytorch.py", code: IT_TORCH, output: IT_TORCH_OUT },
              {
                t: "callout",
                kind: "warning",
                title: "The F.kl_div convention",
                text: "`F.kl_div(input, target)` computes $\\sum \\text{target}\\cdot(\\log\\text{target} - \\text{input})$. The **first argument must already be log-probabilities** of the model $q$; the second is the target $p$ (probabilities, or log-probabilities with `log_target=True`). So it returns $\\mathrm{KL}(\\text{target}\\,\\|\\,\\text{model})$ — the reverse order of the maths notation. Passing probabilities as `input` raises no error and gives a meaningless (here negative) number. Use `reduction=\"batchmean\"`: the default `\"mean\"` divides by the number of *elements* (batch × classes), not the batch size.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras exposes the same losses as classes. Unlike PyTorch's `kl_div`, `tf.keras.losses.KLDivergence` takes **probabilities** for both arguments in the order (y_true, y_pred) and computes $\\mathrm{KL}(y_{\\text{true}}\\,\\|\\,y_{\\text{pred}})$.",
              },
              {
                t: "code",
                lang: "python",
                title: "information_theory_keras.py (not executed in CI)",
                code: IT_KERAS,
                output: "Prints the batch-mean cross-entropy (≈ 0.3185 for these logits, the same as PyTorch), the larger smoothed value (≈ 0.4369), the sparse-label cross-entropy (equal to the first value), and KL(p||q) ≈ 0.1487 nats.",
              },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Quantity", "Definition", "Symmetric?", "Range", "Typical use"],
        rows: [
          ["Entropy H(p)", "−Σ p log p", "—", "[0, log K]", "Uncertainty of predictions; entropy bonuses in RL; active learning"],
          ["Cross-entropy H(p,q)", "−Σ p log q", "No", "[H(p), ∞]", "Classification and language-model training loss"],
          ["Forward KL(p‖q)", "Σ p log(p/q)", "No", "[0, ∞]", "Maximum likelihood; knowledge distillation (teacher p, student q)"],
          ["Reverse KL(q‖p)", "Σ q log(q/p)", "No", "[0, ∞]", "Variational inference (ELBO); policy regularisation in RL fine-tuning"],
          ["Jensen–Shannon", "½KL(p‖m)+½KL(q‖m)", "Yes", "[0, ln 2]", "Original GAN objective; comparing distributions robustly"],
          ["Mutual information I(X;Y)", "KL(p(x,y)‖p(x)p(y))", "Yes", "[0, min(H(X),H(Y))]", "Feature selection; motivation for contrastive learning"],
          ["Perplexity", "exp(cross-entropy)", "—", "[1, ∞]", "Reporting language-model quality"],
          ["Gaussian KL (closed form)", "log(σ₂/σ₁)+(σ₁²+Δμ²)/2σ₂²−½", "No", "[0, ∞]", "VAE latent regulariser"],
        ],
      },
      {
        t: "p",
        text: "**Bits vs nats** is only a unit change (nats × $1/\\ln 2 \\approx 1.4427$ = bits); it never changes which model wins or where a minimum is. Compression papers use bits; ML code uses nats because `log` is natural. Other divergences — $f$-divergences (KL and JS are members), the Wasserstein distance of WGAN, Rényi divergences — balance mode-covering vs mode-seeking and finiteness differently.",
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Classification training**: image classifiers, speech recognisers and every softmax output layer minimise cross-entropy against one-hot or smoothed labels — see [Classification Losses](/learn/m05/classification-losses).",
          "**Language models**: GPT-style models are trained to minimise next-token cross-entropy and are evaluated with perplexity ([Language Modelling](/learn/m13/language-modeling)).",
          "**Knowledge distillation** (Hinton, Vinyals and Dean, 2015): a small student matches a large teacher's softened outputs by minimising a KL divergence, learning which wrong classes are *almost* right.",
          "**Variational autoencoders**: the loss is reconstruction error plus $\\mathrm{KL}(q(z\\mid x)\\,\\|\\,\\mathcal N(0, I))$, computed with the Gaussian closed form ([VAE](/learn/m14/vae)).",
          "**Reinforcement learning from human feedback**: fine-tuned language models usually get a KL penalty to a reference model so they don't drift too far.",
          "**Compression**: entropy is the lower limit for lossless compression; a model with cross-entropy $H(p,q)$ bits per symbol can drive an arithmetic coder to about that rate — better predictors are better compressors.",
          "**Decision trees and active learning**: information gain (a mutual information) picks splits; predictive entropy picks which unlabelled examples to label next.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Cross-entropy — strengths:** the maximum-likelihood loss; needs only data samples; convex in the logits of a linear model; clean gradient $q - p$ that does not vanish when the model is confidently wrong.",
          "**Cross-entropy — weaknesses:** unbounded, so mislabelled examples with $q_c \\to 0$ give huge losses; it keeps rewarding confidence, making networks over-confident (poorly calibrated). Label smoothing and temperature scaling help.",
          "**KL — strengths:** a sound measure of the extra cost of using $q$ for $p$; zero only when they are equal; closed forms for many families (Gaussians, categoricals).",
          "**KL — weaknesses:** one-sided and not a metric (no triangle inequality); infinite when supports don't match; Monte Carlo estimates can be noisy when the distributions differ a lot.",
          "**Forward vs reverse:** forward KL gives broad fits that may put probability in implausible places (blurry samples); reverse KL gives sharp fits that may miss modes (under-estimates uncertainty). Neither is always right.",
          "**JS:** symmetric and bounded, but stuck at $\\ln 2$ when supports don't overlap, giving no useful gradient — a key problem of early GAN training.",
          "**Perplexity:** intuitive and standard, but depends on tokenizer and test set, and measures prediction quality, not usefulness or factual accuracy.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Passing probabilities to `F.kl_div`.** The first argument must be log-probabilities; probabilities silently give a wrong value (negative in our example). Fix: `F.kl_div(F.log_softmax(student_logits, -1), teacher_probs, reduction=\"batchmean\")`.",
          "**Getting the KL direction backwards.** `F.kl_div(log q, p)` is $\\mathrm{KL}(p\\,\\|\\,q)$. Swapping arguments gives reverse KL, with a different optimum. Decide which distribution weights the average first.",
          "**Applying softmax before `F.cross_entropy` / `nn.CrossEntropyLoss`.** They expect raw logits and apply log-softmax inside; a double softmax squashes outputs and weakens the gradient. Fix: feed logits.",
          "**Computing `log(softmax(z))` or `log(p)` naively.** Large logits overflow; tiny probabilities give `log(0) = -inf` and NaN gradients. Fix: `log_softmax`, `logsumexp` or a fused loss; if you must log probabilities, clamp with a small epsilon.",
          "**Mixing bits and nats.** E.g. comparing `np.log2` and `np.log` losses, or perplexity as `2 ** loss` with a loss in nats. Fix: perplexity is `exp(loss_in_nats)` = `2 ** loss_in_bits`.",
          "**Expecting cross-entropy to reach 0 with soft or smoothed labels.** Its minimum is $H(p) > 0$; only the KL part reaches 0. A smoothed loss stuck at 0.29 nats for $K=3$, $\\varepsilon=0.1$ is at its optimum.",
          "**Treating KL as a distance.** It is one-sided and breaks the triangle inequality; comparing KLs in different directions is meaningless. Use JS or a true metric for symmetry.",
          "**Wrong reduction for KL in a batch.** Sum over classes, average over the batch (`batchmean`) = per-example KL. The default `mean` also divides by $K$, silently shrinking the loss $K$ times.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is entropy, and when is it maximal and minimal?", a: String.raw`Entropy $H(p) = -\sum_x p(x)\log p(x)$ is the average surprisal of outcomes from $p$: the average uncertainty, or the fewest bits (base 2) per outcome any code can use on average. It is 0 when one outcome has probability 1, and largest, $\log K$, for the uniform distribution over $K$ outcomes.` },
      { t: "qa", level: "beginner", q: "What is the difference between entropy, cross-entropy and KL divergence?", a: String.raw`Entropy $H(p)$ averages $-\log p$ under $p$. Cross-entropy $H(p,q)$ averages $-\log q$ under $p$ — the cost of using model $q$ when data come from $p$. KL is the excess: $\mathrm{KL}(p\,\|\,q) = H(p,q) - H(p) \ge 0$, zero only when $q = p$.` },
      { t: "qa", level: "beginner", q: "Why is cross-entropy with a one-hot label just −log of the predicted probability of the true class?", a: String.raw`With $p = e_c$, every term of $-\sum_i p_i\log q_i$ vanishes except $i = c$, leaving $-\log q_c$. Also $H(e_c) = 0$, so here cross-entropy equals the KL divergence from the label to the prediction.` },
      { t: "qa", level: "beginner", q: "What is perplexity?", a: String.raw`The exponential of the average per-token cross-entropy (in nats): $\exp(\bar H)$. It is the size of an equally uncertain uniform distribution — a uniform model over $V$ tokens has perplexity $V$. Lower is better; 1 is perfect.` },
      { t: "qa", level: "intermediate", q: "Prove that KL divergence is non-negative.", a: String.raw`On the support $S$ of $p$: $-\mathrm{KL}(p\|q) = \sum_{S} p\log(q/p) \le \log\sum_S p\,(q/p) = \log\sum_S q \le \log 1 = 0$, using Jensen's inequality for the concave log. Equality requires $q/p$ constant on $S$ and no $q$-mass outside $S$, i.e. $q = p$.` },
      { t: "qa", level: "intermediate", q: "Why is minimising cross-entropy equivalent to maximum likelihood?", a: String.raw`With the data distribution $p_{\text{data}}$, $H(p_{\text{data}}, q_\theta) = -\frac1N\sum_n\log q_\theta(x_n)$, the negative mean log-likelihood. Since $H(p_{\text{data}}, q_\theta) = H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}}\|q_\theta)$ and the entropy term ignores $\theta$, minimising cross-entropy, minimising forward KL and maximising likelihood share one solution.` },
      { t: "qa", level: "intermediate", q: "What is the gradient of softmax cross-entropy with respect to the logits, and why is it convenient?", a: String.raw`$\nabla_z H(p, \mathrm{softmax}(z)) = \mathrm{softmax}(z) - p$. It is bounded (each entry in $[-1,1]$), large exactly when the prediction is wrong, and has no saturating factor like $\sigma'(z)$, so learning doesn't stall when the model is confidently wrong. Entries sum to 0, so only relative logits change.` },
      { t: "qa", level: "intermediate", q: "Explain forward vs reverse KL when fitting a unimodal model to a bimodal distribution.", a: String.raw`Forward $\mathrm{KL}(p\|q)$ averages over $p$ and is infinite if $q$ vanishes where $p$ has mass, so $q$ spreads over both modes (a Gaussian matches $p$'s mean and variance, sitting between them). Reverse $\mathrm{KL}(q\|p)$ averages over $q$ and punishes mass where $p$ is small, so $q$ locks onto one mode. E.g. for $0.5\mathcal N(-2,0.25)+0.5\mathcal N(2,0.25)$: forward gives $\mathcal N(0, 2.06^2)$, reverse gives $\mathcal N(\pm2, 0.5^2)$ with KL $= \ln 2$.` },
      { t: "qa", level: "advanced", q: "Derive the KL divergence between N(μ, σ²) and N(0, 1) used in VAEs.", a: String.raw`$\log\frac{q(z)}{p(z)} = -\log\sigma - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}$. Under $q$: $\mathbb{E}[(z-\mu)^2] = \sigma^2$ and $\mathbb{E}[z^2] = \mu^2 + \sigma^2$, so $\mathrm{KL} = -\log\sigma - \frac12 + \frac{\mu^2+\sigma^2}{2} = \frac12(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$, summed over independent latent dimensions. It is 0 only at $\mu = 0, \sigma = 1$.` },
      { t: "qa", level: "advanced", q: "How does label smoothing change the optimum of the cross-entropy loss?", a: String.raw`With target $p^{\text{ls}} = (1-\varepsilon)e_c + \varepsilon/K$, the loss $H(p^{\text{ls}}, q) = H(p^{\text{ls}}) + \mathrm{KL}(p^{\text{ls}}\|q)$ is minimised at $q = p^{\text{ls}}$, which means a finite logit gap $\log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$ (e.g. $\ln 28 \approx 3.33$ for $K=3$, $\varepsilon=0.1$) instead of an infinite one. The minimum loss is $H(p^{\text{ls}}) > 0$. This limits over-confidence and often improves calibration, at the cost of always-softened probabilities.` },
      { t: "qa", level: "advanced", q: "Why did the original GAN objective have training difficulties, in information-theoretic terms?", a: String.raw`With an optimal discriminator, the generator minimises $2\,\mathrm{JS}(p_{\text{data}}, p_G) - \log 4$. When real and generated data (nearly) never overlap — typical early on, since both live near thin low-dimensional surfaces — JS is stuck at $\ln 2$ however far apart they are, so its gradient says nothing. This motivated the non-saturating generator loss and divergences like the Wasserstein distance, which change smoothly with the distance.` },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "How many bits of information does an event of probability 1/8 carry?", answer: 3, tolerance: 0.001, explain: "$-\\log_2(1/8) = \\log_2 8 = 3$ bits." },
          { kind: "numeric", q: "What is the entropy, in bits, of a coin that lands heads with probability 0.9? (3 decimals)", answer: 0.469, tolerance: 0.002, explain: "$-(0.9\\log_2 0.9 + 0.1\\log_2 0.1) = 0.1368 + 0.3322 = 0.469$ bits — far below the fair coin's 1 bit." },
          { kind: "numeric", q: "Compute KL(p‖q) in nats for p = (0.5, 0.5) and q = (0.9, 0.1). (3 decimals)", answer: 0.511, tolerance: 0.002, explain: "$0.5\\ln(0.5/0.9) + 0.5\\ln(0.5/0.1) = -0.2939 + 0.8047 = 0.5108$ nats. The reverse direction is 0.3681." },
          { kind: "numeric", q: "A language model's average cross-entropy on a test set is 2.3 nats per token. What is its perplexity? (2 decimals)", answer: 9.97, tolerance: 0.02, explain: "$e^{2.3} = 9.974$." },
          { kind: "numeric", q: "Compute KL(N(0, 1) ‖ N(0, 2²)) in nats. (3 decimals)", answer: 0.318, tolerance: 0.002, explain: "$\\log(2/1) + (1 + 0)/(2\\cdot4) - 1/2 = 0.6931 + 0.125 - 0.5 = 0.3181$." },
          { q: "You fit a single Gaussian q to a bimodal p by minimising KL(q‖p). What do you typically get?", options: ["A wide Gaussian centred between the modes", "A narrow Gaussian on one of the modes", "Exactly p", "A uniform distribution"], answer: 1, explain: "Reverse KL is mode-seeking: it punishes q hard for mass where p is small (the valley), so q fits one mode." },
          { q: "In PyTorch, which call computes KL(p‖q) for a target distribution p and model logits z?", options: ["F.kl_div(F.softmax(z, -1), p, reduction='batchmean')", "F.kl_div(F.log_softmax(z, -1), p, reduction='batchmean')", "F.kl_div(p.log(), F.softmax(z, -1), reduction='batchmean')", "F.kl_div(p, F.log_softmax(z, -1))"], answer: 1, explain: "The input must be the model's log-probabilities; the target is the probability distribution that weights the average. batchmean gives the per-example KL." },
          { q: "With label smoothing, the training cross-entropy stops decreasing at a positive value. What does that value equal at the optimum?", options: ["Zero — it must be a bug", "The entropy of the smoothed target distribution", "log K", "The KL divergence between the smoothed and one-hot targets"], answer: 1, explain: "$H(p^{\\text{ls}}, q) = H(p^{\\text{ls}}) + \\mathrm{KL}(p^{\\text{ls}}\\|q)$; at the optimum the KL is 0 and the loss equals $H(p^{\\text{ls}})$." },
        ],
      },
      {
        t: "exercise",
        title: "Entropy, cross-entropy and KL by hand",
        prompt: "Let $p = (0.5, 0.25, 0.125, 0.125)$ and let $q$ be uniform over the 4 outcomes. In bits, compute $H(p)$, $H(p,q)$ and $\\mathrm{KL}(p\\,\\|\\,q)$, and verify that $\\mathrm{KL}(p\\,\\|\\,q) = \\log_2 4 - H(p)$.",
        hint: "All probabilities are powers of 2, so every logarithm is an integer.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Surprisals", tex: String.raw`-\log_2 p = (1, 2, 3, 3), \qquad -\log_2 q = (2, 2, 2, 2)` },
              { title: "Entropy", tex: String.raw`H(p) = 0.5\cdot1 + 0.25\cdot2 + 0.125\cdot3 + 0.125\cdot3 = 1.75\ \text{bits}` },
              { title: "Cross-entropy", tex: String.raw`H(p,q) = (0.5+0.25+0.125+0.125)\cdot 2 = 2\ \text{bits}` },
              { title: "KL", tex: String.raw`\mathrm{KL}(p\,\|\,q) = 2 - 1.75 = 0.25\ \text{bits} = \log_2 4 - H(p)\ \checkmark` },
            ],
          },
          { t: "p", text: "In plain words: the best code for $p$ uses codewords of 1, 2, 3 and 3 bits (average 1.75); a fixed 2-bit code (best for uniform $q$) wastes 0.25 bits per symbol." },
        ],
      },
      {
        t: "exercise",
        title: "Stable cross-entropy and KL from logits",
        prompt: "Write NumPy functions `cross_entropy_logits(z, p)` and `kl_logits(z, p)` that take logits `z` (rows) and target distributions `p` (rows, possibly with zeros) and never overflow. Test them on `z = [[1000, 0, -1000], [2, 1, 0.1]]` with `p = [[1, 0, 0], [0.7, 0.2, 0.1]]`, and compare with the naive formula `-(p * log(softmax(z))).sum()`.",
        hint: "log softmax(z) = z − max(z) − log Σ exp(z − max(z)). For the entropy of p, only include entries with p > 0.",
        solution: [
          { t: "code", lang: "python", title: "stable_ce.py", code: EX_CODE, output: EX_CODE_OUT },
          { t: "p", text: "The naive version computes $e^{1000}$, which overflows to `inf`, and `inf/inf = nan`. The stable version subtracts the largest logit first, so the biggest exponent is $e^0 = 1$. Row one's loss is exactly zero (printed as IEEE negative zero, `-0.`) because the model gives the true class probability $1$ (to double precision). Row two's KL, $0.005211$ nats, is small because $q = (0.659, 0.242, 0.099)$ is close to $p = (0.7, 0.2, 0.1)$." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Surprisal $-\\log p(x)$; bits use $\\log_2$, nats use $\\ln$; 1 nat $= 1.4427$ bits.",
          "Entropy $H(p) = -\\sum p\\log p$: 0 for deterministic, $\\log K$ for uniform, always in between.",
          "Joint/conditional: $H(X,Y) = H(X) + H(Y\\mid X)$; mutual information $I(X;Y) = H(X) + H(Y) - H(X,Y) = \\mathrm{KL}(p(x,y)\\|p(x)p(y)) \\ge 0$.",
          "Cross-entropy $H(p,q) = -\\sum p\\log q = H(p) + \\mathrm{KL}(p\\|q)$; KL $= \\sum p\\log(p/q) \\ge 0$ (Gibbs, via Jensen), $= 0$ iff $p = q$.",
          "Minimising cross-entropy on data = minimising forward KL to the data distribution = maximum likelihood.",
          "One-hot labels: loss $= -\\log q_c$; gradient w.r.t. logits $= q - p$. Label smoothing makes the optimum finite and the minimum loss $H(p^{\\text{ls}})$.",
          "KL is asymmetric: forward $\\mathrm{KL}(p\\|q)$ is mode-covering, reverse $\\mathrm{KL}(q\\|p)$ is mode-seeking.",
          "Gaussian KL: $\\log\\frac{\\sigma_2}{\\sigma_1} + \\frac{\\sigma_1^2 + (\\mu_1-\\mu_2)^2}{2\\sigma_2^2} - \\frac12$; VAE form $\\frac12(\\mu^2+\\sigma^2-1-\\log\\sigma^2)$.",
          "Perplexity $= \\exp(\\text{mean cross-entropy in nats})$. JS $= \\frac12\\mathrm{KL}(p\\|m) + \\frac12\\mathrm{KL}(q\\|m) \\in [0, \\ln 2]$.",
          "PyTorch: `F.cross_entropy(logits, target)`; `F.kl_div(log_q, p, reduction='batchmean')` = KL(p‖q); `torch.distributions.kl_divergence(P, Q)`.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Classification Losses](/learn/m05/classification-losses) — cross-entropy as a training loss: binary vs categorical, stability, label smoothing, focal loss.",
          "[Maximum Likelihood and MAP](/learn/m02/mle-map) — the principle behind minimising cross-entropy, and regularisation as a prior.",
          "[Variational Autoencoders](/learn/m14/vae) — the Gaussian KL and reverse-KL variational inference inside a generative model.",
          "[Language Modelling and Perplexity](/learn/m13/language-modeling) — next-token cross-entropy at scale, and reporting perplexity.",
          "[Generative Adversarial Networks](/learn/m14/gans) — the Jensen–Shannon connection and why it motivated new divergences.",
        ],
      },
    ],
  },
};

export default lesson;
