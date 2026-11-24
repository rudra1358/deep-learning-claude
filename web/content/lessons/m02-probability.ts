import type { Lesson } from "@/lib/types";

const PROB_NUMPY = String.raw`import numpy as np

rng = np.random.default_rng(0)

# ---------- 1. Joint, marginal and conditional probability from a table ----------
# rows: weather (sunny, cloudy, rainy); columns: umbrella (no, yes)
joint = np.array([[0.40, 0.02],
                  [0.18, 0.10],
                  [0.05, 0.25]])
p_weather = joint.sum(axis=1)                 # marginalise out the umbrella
p_umbrella = joint.sum(axis=0)                # marginalise out the weather
p_umb_given_rain = joint[2] / p_weather[2]    # P(U | W=rainy) = P(W=rainy, U) / P(W=rainy)
print("P(W) =", p_weather, " P(U) =", p_umbrella)
print("P(U | rainy) =", p_umb_given_rain.round(4))
independent = np.allclose(joint, np.outer(p_weather, p_umbrella))
print("weather and umbrella independent?", independent)

# ---------- 2. Bayes' theorem: a medical test, exactly and by simulation ----------
prior, sensitivity, false_pos = 0.01, 0.99, 0.05
evidence = sensitivity * prior + false_pos * (1 - prior)   # P(+), law of total probability
posterior = sensitivity * prior / evidence                  # P(D | +)
print(f"\nP(+) = {evidence:.4f}   P(disease | +) = {posterior:.4f}")

n_people = 1_000_000
sick = rng.random(n_people) < prior
positive = np.where(sick, rng.random(n_people) < sensitivity,
                          rng.random(n_people) < false_pos)
print(f"simulated P(disease | +) = {sick[positive].mean():.4f} "
      f"from {positive.sum()} positive tests")

# ---------- 3. Expectation and variance of a fair die ----------
faces = np.arange(1, 7)
mean = np.sum(faces * (1 / 6))                       # E[X] = sum x p(x)
var = np.sum((faces - mean) ** 2 * (1 / 6))          # Var[X] = E[(X - E[X])^2]
rolls = rng.integers(1, 7, size=100_000)
print(f"\ndie: E[X] = {mean:.4f} (sample {rolls.mean():.4f}), "
      f"Var[X] = {var:.4f} (sample {rolls.var():.4f})")

# ---------- 4. The standard distributions: theory vs 100,000 samples ----------
N = 100_000
draws = {
    "Bernoulli(p=0.3)":      (rng.random(N) < 0.3,          0.3,  0.3 * 0.7),
    "Binomial(n=10, p=0.3)": (rng.binomial(10, 0.3, N),     3.0,  10 * 0.3 * 0.7),
    "Gaussian(mu=1, sd=2)":  (rng.normal(1.0, 2.0, N),      1.0,  4.0),
    "Uniform(a=-1, b=3)":    (rng.uniform(-1, 3, N),        1.0,  (3 - (-1)) ** 2 / 12),
}
print()
for name, (x, true_mean, true_var) in draws.items():
    print(f"{name:22s} mean {x.mean():7.4f} (true {true_mean:6.4f})   "
          f"var {x.var():7.4f} (true {true_var:6.4f})")

probs = np.array([0.1, 0.6, 0.3])                    # a categorical over 3 classes
cats = rng.choice(3, size=N, p=probs)
print("Categorical frequencies:", np.bincount(cats) / N, " true:", probs)

# ---------- 5. Law of large numbers: the running mean settles ----------
x = rng.normal(1.0, 2.0, 100_000)
running = np.cumsum(x) / np.arange(1, x.size + 1)
print("\nrunning mean of N(1, 2^2):")
for n in [10, 100, 1_000, 10_000, 100_000]:
    print(f"  n = {n:6d}  mean = {running[n - 1]:.4f}  |error| = {abs(running[n - 1] - 1):.4f}"
          f"  (sd/sqrt(n) = {2 / np.sqrt(n):.4f})")

# ---------- 6. Central limit theorem: means of skewed exponential samples ----------
for n in [1, 5, 30]:
    means = rng.exponential(1.0, size=(50_000, n)).mean(axis=1)  # Exp(1): mean 1, var 1
    z = (means - 1.0) / (1.0 / np.sqrt(n))                       # standardise
    skew = np.mean(z ** 3)
    inside = np.mean(np.abs(z) < 1.96)
    print(f"CLT n = {n:2d}: skewness {skew:5.3f} (Gaussian: 0)   "
          f"P(|Z| < 1.96) = {inside:.4f} (Gaussian: 0.9500)")

# ---------- 7. Monte Carlo estimation: E[f(X)] ~ average of f(samples) ----------
# P(X > 2) for X ~ N(0, 1); exact value 0.022750
for n in [100, 10_000, 1_000_000]:
    est = np.mean(rng.normal(size=n) > 2.0)
    print(f"Monte Carlo P(X > 2) with n = {n:7d}: {est:.6f}")

# ---------- 8. Multivariate Gaussian: covariance and correlation ----------
mu = np.array([0.0, 1.0])
Sigma = np.array([[1.0, 0.8],
                  [0.8, 2.0]])
L = np.linalg.cholesky(Sigma)                       # Sigma = L L^T
z = rng.normal(size=(200_000, 2))                   # independent N(0, 1)
samples = mu + z @ L.T                              # x = mu + L z  has covariance Sigma
print("\nsample covariance:\n", np.cov(samples, rowvar=False).round(3))
print("sample correlation:", np.corrcoef(samples, rowvar=False)[0, 1].round(4),
      " true:", round(0.8 / np.sqrt(1.0 * 2.0), 4))`;

const PROB_NUMPY_OUT = String.raw`P(W) = [0.42 0.28 0.3 ]  P(U) = [0.63 0.37]
P(U | rainy) = [0.1667 0.8333]
weather and umbrella independent? False

P(+) = 0.0594   P(disease | +) = 0.1667
simulated P(disease | +) = 0.1670 from 59176 positive tests

die: E[X] = 3.5000 (sample 3.5114), Var[X] = 2.9167 (sample 2.9046)

Bernoulli(p=0.3)       mean  0.3029 (true 0.3000)   var  0.2112 (true 0.2100)
Binomial(n=10, p=0.3)  mean  2.9996 (true 3.0000)   var  2.1041 (true 2.1000)
Gaussian(mu=1, sd=2)   mean  0.9902 (true 1.0000)   var  3.9993 (true 4.0000)
Uniform(a=-1, b=3)     mean  0.9933 (true 1.0000)   var  1.3388 (true 1.3333)
Categorical frequencies: [0.0984  0.60279 0.29881]  true: [0.1 0.6 0.3]

running mean of N(1, 2^2):
  n =     10  mean = 2.0506  |error| = 1.0506  (sd/sqrt(n) = 0.6325)
  n =    100  mean = 1.2563  |error| = 0.2563  (sd/sqrt(n) = 0.2000)
  n =   1000  mean = 1.0677  |error| = 0.0677  (sd/sqrt(n) = 0.0632)
  n =  10000  mean = 0.9900  |error| = 0.0100  (sd/sqrt(n) = 0.0200)
  n = 100000  mean = 0.9958  |error| = 0.0042  (sd/sqrt(n) = 0.0063)
CLT n =  1: skewness 2.020 (Gaussian: 0)   P(|Z| < 1.96) = 0.9462 (Gaussian: 0.9500)
CLT n =  5: skewness 0.877 (Gaussian: 0)   P(|Z| < 1.96) = 0.9563 (Gaussian: 0.9500)
CLT n = 30: skewness 0.323 (Gaussian: 0)   P(|Z| < 1.96) = 0.9530 (Gaussian: 0.9500)
Monte Carlo P(X > 2) with n =     100: 0.020000
Monte Carlo P(X > 2) with n =   10000: 0.022100
Monte Carlo P(X > 2) with n = 1000000: 0.022859

sample covariance:
 [[0.998 0.798]
 [0.798 1.994]]
sample correlation: 0.5657  true: 0.5657`;

const PROB_TORCH = String.raw`import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Binomial, Categorical, MultivariateNormal, Normal, Uniform

torch.manual_seed(0)

# ---------- 1. Distribution objects: sample, mean, variance, log_prob ----------
dists = {
    "Bernoulli(0.3)": Bernoulli(probs=torch.tensor(0.3)),
    "Binomial(10, 0.3)": Binomial(total_count=10, probs=torch.tensor(0.3)),
    "Normal(1, 2)": Normal(loc=torch.tensor(1.0), scale=torch.tensor(2.0)),  # scale = std, not variance
    "Uniform(-1, 3)": Uniform(torch.tensor(-1.0), torch.tensor(3.0)),
}
for name, d in dists.items():
    x = d.sample((100_000,))
    print(f"{name:18s} sample mean {x.mean():7.4f} (true {d.mean:6.4f})  "
          f"sample var {x.var():7.4f} (true {d.variance:6.4f})")

print("\nNormal(1, 2).log_prob(2.0) =", round(Normal(1.0, 2.0).log_prob(torch.tensor(2.0)).item(), 6))
print("Binomial(10, 0.3).log_prob(3).exp() =", round(Binomial(10, torch.tensor(0.3)).log_prob(torch.tensor(3.0)).exp().item(), 6))

# ---------- 2. Categorical: a softmax output is a distribution over classes ----------
logits = torch.tensor([2.0, 1.0, 0.1])              # e.g. a classifier's or LM's final layer
probs = F.softmax(logits, dim=0)
cat = Categorical(logits=logits)                    # same distribution, built from logits
counts = torch.bincount(cat.sample((100_000,)), minlength=3)
print("\nsoftmax probs:", probs.numpy().round(4), " sampled freq:", (counts / 100_000).numpy().round(4))
print("-log p(class 0) =", round(-cat.log_prob(torch.tensor(0)).item(), 4),
      " = cross_entropy:", round(F.cross_entropy(logits[None], torch.tensor([0])).item(), 4))

# ---------- 3. Bernoulli in dropout: each unit kept with probability 1 - p ----------
p_drop = 0.5
h = torch.ones(8)
mask = Bernoulli(probs=torch.full_like(h, 1 - p_drop)).sample()
print("\ndropout mask:", mask.numpy(), " output:", (h * mask / (1 - p_drop)).numpy())

# ---------- 4. Gaussians in initialisation and diffusion noise ----------
W = torch.empty(256, 512)
torch.nn.init.normal_(W, mean=0.0, std=(2.0 / 512) ** 0.5)     # He-style Gaussian init
print(f"\ninit weights: mean {W.mean():.5f}, std {W.std():.5f} (target {(2 / 512) ** 0.5:.5f})")

x0 = torch.tensor([1.0, -0.5, 0.25])               # a clean "image" with 3 pixels
alpha_bar = 0.6                                     # how much signal survives at step t
eps = torch.randn_like(x0)                          # standard Gaussian noise
xt = alpha_bar ** 0.5 * x0 + (1 - alpha_bar) ** 0.5 * eps
print("diffusion: x0", x0.numpy(), "-> x_t", xt.numpy().round(4))

# ---------- 5. Multivariate Gaussian and the reparameterisation trick ----------
mvn = MultivariateNormal(loc=torch.tensor([0.0, 1.0]),
                         covariance_matrix=torch.tensor([[1.0, 0.8], [0.8, 2.0]]))
s = mvn.sample((200_000,))
print("\nMVN sample covariance:\n", torch.cov(s.T).numpy().round(3))

mu = torch.tensor(0.5, requires_grad=True)
log_sigma = torch.tensor(0.0, requires_grad=True)
q = Normal(mu, log_sigma.exp())
z = q.rsample((10_000,))                            # z = mu + sigma * eps: differentiable in mu, sigma
loss = (z ** 2).mean()                              # Monte Carlo estimate of E[z^2] = mu^2 + sigma^2
loss.backward()
print(f"MC E[z^2] = {loss.item():.4f} (exact {0.5 ** 2 + 1:.4f}); "
      f"d/dmu = {mu.grad.item():.4f} (exact {2 * 0.5:.4f}), "
      f"d/dlog_sigma = {log_sigma.grad.item():.4f} (exact {2 * 1.0:.4f})")`;

const PROB_TORCH_OUT = String.raw`Bernoulli(0.3)     sample mean  0.2985 (true 0.3000)  sample var  0.2094 (true 0.2100)
Binomial(10, 0.3)  sample mean  2.9915 (true 3.0000)  sample var  2.0940 (true 2.1000)
Normal(1, 2)       sample mean  0.9974 (true 1.0000)  sample var  4.0044 (true 4.0000)
Uniform(-1, 3)     sample mean  0.9960 (true 1.0000)  sample var  1.3361 (true 1.3333)

Normal(1, 2).log_prob(2.0) = -1.737086
Binomial(10, 0.3).log_prob(3).exp() = 0.266828

softmax probs: [0.659  0.2424 0.0986]  sampled freq: [0.6587 0.2436 0.0977]
-log p(class 0) = 0.417  = cross_entropy: 0.417

dropout mask: [1. 1. 1. 1. 0. 0. 0. 1.]  output: [2. 2. 2. 2. 0. 0. 0. 2.]

init weights: mean 0.00016, std 0.06264 (target 0.06250)
diffusion: x0 [ 1.   -0.5   0.25] -> x_t [ 1.2145 -0.1796  0.0503]

MVN sample covariance:
 [[1.003 0.8  ]
 [0.8   2.007]]
MC E[z^2] = 1.2843 (exact 1.2500); d/dmu = 1.0244 (exact 1.0000), d/dlog_sigma = 2.0563 (exact 2.0000)`;

const PROB_KERAS = String.raw`import tensorflow as tf

tf.random.set_seed(0)

# Sampling with core TensorFlow ops
bern = tf.cast(tf.random.uniform([100000]) < 0.3, tf.float32)       # Bernoulli(0.3)
gauss = tf.random.normal([100000], mean=1.0, stddev=2.0)            # N(1, 2^2)
unif = tf.random.uniform([100000], minval=-1.0, maxval=3.0)         # U(-1, 3)
for name, x in [("Bernoulli", bern), ("Gaussian", gauss), ("Uniform", unif)]:
    print(name, "mean", float(tf.reduce_mean(x)), "var", float(tf.math.reduce_variance(x)))

# Categorical sampling from logits (what a softmax output layer defines)
logits = tf.constant([[2.0, 1.0, 0.1]])
idx = tf.random.categorical(logits, num_samples=100000)
print("class frequencies", tf.math.bincount(tf.cast(idx[0], tf.int32), minlength=3).numpy() / 100000)

# Keras layers that are probability in disguise
drop = tf.keras.layers.Dropout(0.5)                  # Bernoulli mask, scaled by 1/(1-p), only when training=True
print(drop(tf.ones([1, 8]), training=True).numpy())
init = tf.keras.initializers.HeNormal(seed=0)        # truncated Gaussian, std sqrt(2 / fan_in)
print("He init std", float(tf.math.reduce_std(init(shape=(512, 256)))))`;

const PI_CODE = String.raw`import numpy as np

rng = np.random.default_rng(0)
for n in [100, 10_000, 1_000_000]:
    u = rng.random((n, 2))                       # points uniform in the unit square
    inside = (u ** 2).sum(axis=1) <= 1.0         # Bernoulli: did it land in the quarter disc?
    p_hat = inside.mean()                        # estimates pi / 4
    se = np.sqrt(p_hat * (1 - p_hat) / n)        # standard error of a Bernoulli mean
    print(f"n = {n:8d}  pi_hat = {4 * p_hat:.5f}  error = {abs(4 * p_hat - np.pi):.5f}  "
          f"4*SE = {4 * se:.5f}")`;

const PI_OUT = String.raw`n =      100  pi_hat = 2.92000  error = 0.22159  4*SE = 0.17758
n =    10000  pi_hat = 3.15160  error = 0.01001  4*SE = 0.01635
n =  1000000  pi_hat = 3.14105  error = 0.00054  4*SE = 0.00164`;

const lesson: Lesson = {
  id: "m02/probability",
  title: "Probability, Random Variables and Distributions",
  summary:
    "The probability toolkit of deep learning: axioms, joint/marginal/conditional probability, independence, the chain rule behind language models, Bayes' theorem on a medical test, random variables, expectation, variance and covariance, the Bernoulli, categorical, binomial, Gaussian and uniform distributions, the law of large numbers, the central limit theorem and Monte Carlo estimation — with NumPy and torch.distributions.",
  minutes: 80,
  level: "beginner",
  prerequisites: ["m00/numpy-arrays", "m02/vectors-matrices", "m02/derivatives-chain-rule"],
  next: ["m02/mle-map", "m02/information-theory", "m05/classification-losses"],
  terms: ["probability-distribution", "random-variable", "conditional-probability", "bayes-theorem", "expectation", "variance", "covariance", "gaussian", "bernoulli", "categorical", "law-of-large-numbers", "central-limit-theorem", "monte-carlo", "softmax"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      { t: "p", text: "**Probability** is the mathematics of uncertainty: it assigns numbers between 0 and 1 to the possible outcomes of a process whose result we cannot predict exactly, in a way that obeys a few consistent rules. Deep learning is soaked in it. A classifier does not output a label; it outputs a **probability distribution** over labels. A language model outputs a distribution over the next token. Training data is treated as a random sample from an unknown distribution, mini-batches are random draws, weights are initialised randomly, dropout switches units off at random, and generative models such as VAEs and diffusion models are defined entirely in terms of distributions they learn to sample from." },
      { t: "p", text: "Why is a formal theory needed, rather than intuition about chance? Because the questions we ask are subtle. *Given that a test is positive, how likely is the disease?* (Bayes' theorem — and the intuitive answer is usually wrong by a factor of five.) *How many samples are needed before an average can be trusted?* (The law of large numbers and the $1/\\sqrt{n}$ rule.) *Why is the sum of many small effects bell-shaped?* (The central limit theorem.) *How does a model assign a probability to a whole sentence?* (The chain rule of probability.) Each has a precise answer, and each is used daily when building and evaluating networks." },
      { t: "p", text: "This lesson builds the vocabulary from the ground up — sample spaces and axioms, then conditional probability and Bayes, then random variables and their summaries (mean, variance, covariance), then the five distributions that appear everywhere in deep learning — and finishes with sampling and Monte Carlo estimation, implemented in NumPy and in PyTorch's `torch.distributions`. The next lesson, [Maximum Likelihood and MAP](/learn/m02/mle-map), uses all of it to explain where loss functions come from." },
      { t: "callout", kind: "key", text: "A neural network classifier is a function that maps an input to the parameters of a probability distribution (a sigmoid gives a Bernoulli, a softmax gives a categorical). Understanding those distributions is understanding what the network actually predicts." },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      { t: "list", items: [
        "[NumPy arrays](/learn/m00/numpy-arrays) — the code uses array indexing, `sum(axis=...)`, boolean masks and `np.random.default_rng`.",
        String.raw`[Scalars, vectors and matrices](/learn/m02/vectors-matrices) — needed for the multivariate Gaussian: a mean vector $\boldsymbol\mu$, a covariance matrix $\Sigma$, the transpose and the matrix–vector product.`,
        String.raw`[Derivatives](/learn/m02/derivatives-chain-rule) — used only lightly: a probability density is the derivative of the cumulative distribution function, and integrals of densities give probabilities. If integrals are rusty, read $\int_a^b p(x)\,dx$ as "area under the curve between $a$ and $b$".`,
        "Summation notation $\\sum_i$ and basic algebra. No prior probability course is assumed.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      { t: "callout", kind: "analogy", title: "A bag of marbles, and a crowd of repeats", text: "Imagine a bag holding 100 marbles: 30 red, 70 blue. Reach in blindly and the chance of red is 30/100 = 0.3 — probability as a **proportion of possibilities**. Now imagine repeating the draw (with replacement) thousands of times and counting: about 30% of draws are red — probability as **long-run frequency**. Finally, imagine you only see the draws and not the bag; after many draws you can estimate what is inside — that is statistics, and it is exactly what training a model does." },
      { t: "p", text: "Three ideas carry most of the weight in this lesson:" },
      { t: "list", items: [
        "**Conditioning is zooming in.** $P(A \\mid B)$ asks: among only those situations where $B$ happened, what fraction also have $A$? You throw away everything outside $B$ and renormalise. Bayes' theorem is just this zooming done in the other direction.",
        "**A distribution is a recipe for randomness.** It tells you which values a random quantity can take and how much probability each gets. Its **mean** says where the values centre, its **variance** how widely they spread.",
        "**Averages of many random things are predictable.** One coin flip is unpredictable; the fraction of heads in 10,000 flips is almost certainly within 1% of one half. This is why stochastic gradient descent works with random mini-batches and why Monte Carlo estimates converge.",
      ] },
      { t: "p", text: "A classifier ending in a softmax is like a weather forecaster: it does not say *it will rain*; it says *70% rain, 20% cloud, 10% sun*. Everything we later do with that output — computing a loss, sampling a token, measuring calibration — is probability." },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "1. Sample spaces, events and the axioms" },
      { t: "p", text: String.raw`An **experiment** is any process with an uncertain outcome. Its **sample space** $\Omega$ is the set of all possible outcomes: for a die, $\Omega = \{1,2,3,4,5,6\}$; for a sensor reading, $\Omega = \mathbb{R}$. An **event** $A \subseteq \Omega$ is a set of outcomes we may ask about, e.g. "the die shows an even number" $= \{2,4,6\}$. A **probability measure** $P$ assigns a number to every event and must satisfy Kolmogorov's three axioms:` },
      { t: "list", ordered: true, items: [
        String.raw`**Non-negativity:** $P(A) \ge 0$ for every event $A$.`,
        String.raw`**Normalisation:** $P(\Omega) = 1$ — something certainly happens.`,
        String.raw`**Additivity:** if $A_1, A_2, \dots$ are mutually exclusive (no two can happen together), $P(A_1 \cup A_2 \cup \cdots) = \sum_i P(A_i)$.`,
      ] },
      { t: "p", text: String.raw`Everything else follows. The **complement rule** $P(A^c) = 1 - P(A)$ (since $A$ and "not $A$" are exclusive and together make $\Omega$); $P(\varnothing) = 0$; and the **inclusion–exclusion rule** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$, which subtracts the overlap that was counted twice.` },
      { t: "h", text: "2. Joint, marginal and conditional probability" },
      { t: "p", text: String.raw`With two random quantities — say the weather $W$ and whether a person carries an umbrella $U$ — the **joint probability** $P(W = w, U = u)$ gives the probability of each combination. Summing the joint over one variable gives the **marginal** of the other (the name comes from writing the sums in the margins of the table):` },
      { t: "math", tex: String.raw`P(W = w) = \sum_{u} P(W = w,\, U = u)` },
      { t: "p", text: String.raw`The **conditional probability** of $A$ given $B$ (defined when $P(B) > 0$) restricts attention to the outcomes in $B$ and renormalises:` },
      { t: "math", tex: String.raw`P(A \mid B) = \frac{P(A \cap B)}{P(B)}` },
      { t: "p", text: String.raw`Rearranging gives the **product rule** $P(A \cap B) = P(A \mid B)\,P(B)$, and summing it over a partition $B_1,\dots,B_k$ of $\Omega$ gives the **law of total probability** $P(A) = \sum_i P(A \mid B_i)\,P(B_i)$.` },
      { t: "h", text: "3. Independence" },
      { t: "p", text: String.raw`$A$ and $B$ are **independent** if learning one tells you nothing about the other: $P(A \mid B) = P(A)$, equivalently $P(A \cap B) = P(A)\,P(B)$. Random variables $X, Y$ are independent if $p(x, y) = p(x)\,p(y)$ for all values. **Conditional independence** $X \perp Y \mid Z$ means $p(x, y \mid z) = p(x \mid z)\,p(y \mid z)$ — e.g. two symptoms that are correlated in the population become independent once you know the disease. The "i.i.d." assumption — training examples are **independent and identically distributed** — is what lets us multiply per-example probabilities into a likelihood in the next lesson.` },
      { t: "h", text: "4. The chain rule of probability" },
      { t: "p", text: String.raw`Applying the product rule repeatedly factorises any joint distribution into a product of conditionals, with no assumptions at all:` },
      { t: "math", tex: String.raw`P(x_1, x_2, \dots, x_T) = P(x_1)\,P(x_2 \mid x_1)\,P(x_3 \mid x_1, x_2)\cdots = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})` },
      { t: "p", text: String.raw`This identity is the foundation of **autoregressive language modelling**: a model such as GPT never scores a whole sentence at once; it predicts the next token given all previous ones, $P(x_t \mid x_{<t})$, and the probability of the sentence is the product. Training maximises the sum of $\log P(x_t \mid x_{<t})$ over every position — see [Language Modelling](/learn/m13/language-modeling) and [LLM pretraining](/learn/m20/llm-pretraining).` },
      { t: "h", text: "5. Bayes' theorem" },
      { t: "p", text: String.raw`Writing the product rule both ways, $P(A \mid B)P(B) = P(B \mid A)P(A)$, and dividing gives **Bayes' theorem** — the rule for reversing a conditional:` },
      { t: "math", tex: String.raw`\underbrace{P(H \mid E)}_{\text{posterior}} = \frac{\overbrace{P(E \mid H)}^{\text{likelihood}}\;\overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence}}}, \qquad P(E) = \sum_{h} P(E \mid h)\,P(h)` },
      { t: "p", text: String.raw`$H$ is a hypothesis (e.g. "has the disease"), $E$ the evidence (e.g. "test positive"). The **prior** is what we believed before the evidence, the **likelihood** how probable the evidence is under the hypothesis, the **evidence** the total probability of what we observed (a normaliser), and the **posterior** our updated belief. The Math section works through a full medical example.` },
      { t: "h", text: "6. Random variables, PMF, PDF and CDF" },
      { t: "p", text: String.raw`A **random variable** $X$ is a function that assigns a number to each outcome (e.g. the number of heads in 10 flips). It is **discrete** if it takes countably many values and **continuous** if it takes values in an interval.` },
      { t: "list", items: [
        String.raw`A discrete variable has a **probability mass function (PMF)** $p(x) = P(X = x)$, with $p(x) \ge 0$ and $\sum_x p(x) = 1$.`,
        String.raw`A continuous variable has a **probability density function (PDF)** $p(x)$ with $p(x) \ge 0$ and $\int p(x)\,dx = 1$; probabilities are areas: $P(a \le X \le b) = \int_a^b p(x)\,dx$. Any single point has probability 0, and a density value can exceed 1 (it is probability per unit length).`,
        String.raw`Both have a **cumulative distribution function (CDF)** $F(x) = P(X \le x)$, which rises from 0 to 1. For continuous variables $p(x) = F'(x)$.`,
      ] },
      { t: "h", text: "7. Expectation, variance, covariance" },
      { t: "p", text: String.raw`The **expectation** (mean) is the probability-weighted average value: $\mathbb{E}[X] = \sum_x x\,p(x)$ or $\int x\,p(x)\,dx$; more generally $\mathbb{E}[f(X)] = \sum_x f(x)\,p(x)$. Expectation is **linear**: $\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c$ for any $X, Y$ — independent or not. The **variance** measures spread around the mean, and the **standard deviation** is its square root (same units as $X$):` },
      { t: "math", tex: String.raw`\operatorname{Var}[X] = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - \mathbb{E}[X]^2, \qquad \sigma_X = \sqrt{\operatorname{Var}[X]}, \qquad \operatorname{Var}[aX + c] = a^2\operatorname{Var}[X]` },
      { t: "p", text: String.raw`**Covariance** measures whether two variables move together, and **correlation** rescales it to $[-1, 1]$:` },
      { t: "math", tex: String.raw`\operatorname{Cov}[X, Y] = \mathbb{E}\big[(X - \mu_X)(Y - \mu_Y)\big], \qquad \rho_{XY} = \frac{\operatorname{Cov}[X, Y]}{\sigma_X\,\sigma_Y}, \qquad \operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y] + 2\operatorname{Cov}[X, Y]` },
      { t: "p", text: String.raw`For a random vector $\mathbf{x} \in \mathbb{R}^d$ with mean $\boldsymbol\mu$, the **covariance matrix** $\Sigma = \mathbb{E}[(\mathbf{x} - \boldsymbol\mu)(\mathbf{x} - \boldsymbol\mu)^\top]$ collects all variances (diagonal) and covariances (off-diagonal). It is symmetric and positive semi-definite. Independent variables have zero covariance, but zero covariance does **not** imply independence (e.g. $X \sim \mathcal{N}(0,1)$ and $Y = X^2$ are uncorrelated yet completely dependent).` },
      { t: "h", text: "8. Sampling, the law of large numbers and the CLT" },
      { t: "steps", items: [
        { title: "Draw i.i.d. samples", text: String.raw`$x_1, \dots, x_n \sim p$, each independent of the others. Computers generate them from a seeded pseudo-random uniform generator and transform the uniforms (e.g. $u < p$ gives a Bernoulli; Box–Muller turns two uniforms into a Gaussian).` },
        { title: "Average them", tex: String.raw`\bar{x}_n = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad \mathbb{E}[\bar{x}_n] = \mu, \qquad \operatorname{Var}[\bar{x}_n] = \frac{\sigma^2}{n}` },
        { title: "Law of large numbers", text: String.raw`$\bar{x}_n \to \mu$ as $n \to \infty$. The typical error — the **standard error** — is $\sigma/\sqrt{n}$: 100× more samples buy only 10× more accuracy.` },
        { title: "Central limit theorem", text: String.raw`Whatever the shape of $p$ (as long as its variance is finite), the standardised mean $\sqrt{n}(\bar{x}_n - \mu)/\sigma$ approaches $\mathcal{N}(0, 1)$. Sums of many small independent effects look Gaussian.` },
        { title: "Monte Carlo estimation", text: String.raw`Any expectation can be estimated by an average of samples: $\mathbb{E}[f(X)] \approx \frac{1}{n}\sum_i f(x_i)$, with error $\approx \sigma_f/\sqrt{n}$ regardless of the dimension of $X$. A mini-batch gradient is exactly a Monte Carlo estimate of the full-data gradient.` },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Worked example 1 — a joint table" },
      { t: "p", text: "Weather $W \\in$ {sunny, cloudy, rainy} and umbrella $U \\in$ {no, yes} have this joint distribution (the entries sum to 1):" },
      { t: "table", head: ["", "U = no", "U = yes", "P(W) (row sum)"], rows: [
        ["sunny", "0.40", "0.02", "0.42"],
        ["cloudy", "0.18", "0.10", "0.28"],
        ["rainy", "0.05", "0.25", "0.30"],
        ["P(U) (column sum)", "0.63", "0.37", "1.00"],
      ] },
      { t: "steps", items: [
        { title: "Marginals", tex: String.raw`P(W{=}\text{rainy}) = 0.05 + 0.25 = 0.30,\qquad P(U{=}\text{yes}) = 0.02 + 0.10 + 0.25 = 0.37` },
        { title: "Conditional: umbrella given rain", tex: String.raw`P(U{=}\text{yes} \mid \text{rainy}) = \frac{0.25}{0.30} = 0.8333` },
        { title: "Reversed conditional: rain given umbrella", tex: String.raw`P(\text{rainy} \mid U{=}\text{yes}) = \frac{0.25}{0.37} = 0.6757` },
        { title: "Independence check", text: "If $W$ and $U$ were independent, the rainy/yes cell would equal $P(\\text{rainy})P(\\text{yes})$:", tex: String.raw`0.30 \times 0.37 = 0.111 \ne 0.25 \;\Rightarrow\; \text{dependent}` },
      ] },
      { t: "h", text: "Worked example 2 — Bayes' theorem on a medical test" },
      { t: "p", text: String.raw`A disease affects 1% of a population: $P(D) = 0.01$. A test has **sensitivity** $P(+ \mid D) = 0.99$ (it catches 99% of sick people) and a **false-positive rate** $P(+ \mid \neg D) = 0.05$ (specificity 0.95). A randomly chosen person tests positive. How likely is it that they have the disease?` },
      { t: "steps", items: [
        { title: "Evidence by total probability", tex: String.raw`P(+) = P(+\mid D)P(D) + P(+\mid \neg D)P(\neg D) = 0.99(0.01) + 0.05(0.99) = 0.0099 + 0.0495 = 0.0594` },
        { title: "Posterior", tex: String.raw`P(D \mid +) = \frac{P(+\mid D)\,P(D)}{P(+)} = \frac{0.0099}{0.0594} = 0.1667` },
        { title: "Interpretation in counts", text: "Of 10,000 people, 100 are sick: 99 test positive, 1 is missed. Of the 9,900 healthy, 5% = 495 test positive. So 99 of the 99 + 495 = 594 positives are sick: 99/594 = 1/6. The rare disease makes false positives outnumber true ones five to one." },
        { title: "A second, independent positive test", text: "Use the posterior as the new prior (assuming the two test errors are conditionally independent given disease status):", tex: String.raw`P(D \mid +,+) = \frac{0.99 \times 0.1667}{0.99 \times 0.1667 + 0.05 \times 0.8333} = \frac{0.1650}{0.2067} = 0.7984` },
        { title: "And a negative test", tex: String.raw`P(D \mid -) = \frac{0.01 \times 0.01}{0.9406} = 0.000106` },
      ] },
      { t: "callout", kind: "warning", title: "Base-rate neglect", text: "Most people answer 99% to the first question — confusing $P(+ \\mid D)$ with $P(D \\mid +)$. The same error appears in ML: a fraud detector with 99% recall and a 1% false-positive rate, applied where fraud is 1 in 10,000 transactions, produces about 100 false alarms per real fraud. Always multiply by the prior." },
      { t: "h", text: "Worked example 3 — expectation and variance" },
      { t: "p", text: "A fair die, $p(x) = 1/6$ for $x = 1..6$:" },
      { t: "math", tex: String.raw`\mathbb{E}[X] = \tfrac{1+2+3+4+5+6}{6} = 3.5,\qquad \mathbb{E}[X^2] = \tfrac{1+4+9+16+25+36}{6} = \tfrac{91}{6} = 15.1667,\qquad \operatorname{Var}[X] = 15.1667 - 3.5^2 = 2.9167,\quad \sigma = 1.7078` },
      { t: "p", text: String.raw`Covariance from data: heights $h = (160, 165, 170, 175, 180)$ cm and weights $w = (55, 61, 64, 70, 75)$ kg have means 170 and 65. The centred products $(h_i - 170)(w_i - 65)$ are $(-10)(-10), (-5)(-4), 0(-1), 5(5), 10(10) = 100, 20, 0, 25, 100$; their average is $\operatorname{Cov} = 245/5 = 49$. With $\operatorname{Var}[h] = 50$ and $\operatorname{Var}[w] = 48.4$, the correlation is $49/\sqrt{50 \times 48.4} = 0.9961$ — an almost perfectly linear relationship. (Dividing by $n - 1 = 4$ instead of $n$ gives the unbiased sample covariance 61.25; the correlation is unchanged.)` },
      { t: "h", text: "The five distributions every deep learning practitioner uses" },
      { t: "table", head: ["Distribution", "PMF / PDF", "Mean", "Variance", "Where it appears in deep learning"], rows: [
        ["Bernoulli(p)", String.raw`$p^x (1-p)^{1-x},\ x \in \{0,1\}$`, "$p$", "$p(1-p)$", "Binary labels and the sigmoid output of a binary classifier; dropout masks (keep each unit with prob. $1-p_{\\text{drop}}$)"],
        ["Categorical(π)", String.raw`$\prod_k \pi_k^{[x=k]},\ \sum_k \pi_k = 1$`, "— (one-hot mean $= \\boldsymbol\\pi$)", "$\\pi_k(1-\\pi_k)$ per one-hot entry", "Softmax outputs of multi-class classifiers; next-token distributions of language models; sampling with temperature/top-k"],
        ["Binomial(n, p)", String.raw`$\binom{n}{k} p^k (1-p)^{n-k}$`, "$np$", "$np(1-p)$", "Number of correct predictions among $n$ test examples (error bars on accuracy); number of active units after dropout"],
        ["Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$", String.raw`$\frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2 / 2\sigma^2}$`, "$\\mu$", "$\\sigma^2$", "Weight initialisation (Xavier/He); noise added in diffusion models; VAE latent prior and encoder output; regression likelihood (⇒ MSE); Gaussian prior (⇒ weight decay)"],
        ["Uniform(a, b)", "$\\frac{1}{b-a}$ on $[a, b]$", "$\\frac{a+b}{2}$", "$\\frac{(b-a)^2}{12}$", "Uniform weight initialisation (Glorot uniform is the Keras default; PyTorch's `nn.Linear` default is also a uniform distribution); random search over hyperparameters; the raw source of all pseudo-random sampling"],
      ] },
      { t: "p", text: String.raw`Two numerical checks. Binomial with $n = 10, p = 0.3$: $P(X = 3) = \binom{10}{3}0.3^3\,0.7^7 = 120 \times 0.027 \times 0.0823543 = 0.2668$, and $P(X \le 3) = 0.6496$. Gaussian: the standard normal density at its peak is $1/\sqrt{2\pi} = 0.3989$, and the probabilities within 1, 2, 3 standard deviations of the mean are 0.6827, 0.9545 and 0.9973.` },
      { t: "h", text: "The multivariate Gaussian" },
      { t: "math", tex: String.raw`\mathcal{N}(\mathbf{x};\boldsymbol\mu, \Sigma) = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}} \exp\!\Big(-\tfrac{1}{2}(\mathbf{x} - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol\mu)\Big)` },
      { t: "p", text: String.raw`$\mathbf{x}, \boldsymbol\mu \in \mathbb{R}^d$; $\Sigma \in \mathbb{R}^{d\times d}$ is the covariance matrix, $|\Sigma|$ its determinant. Contours of equal density are ellipses whose axes are the eigenvectors of $\Sigma$. To sample, factor $\Sigma = LL^\top$ (Cholesky) and set $\mathbf{x} = \boldsymbol\mu + L\mathbf{z}$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$; then $\operatorname{Cov}[\mathbf{x}] = L\,\mathbb{E}[\mathbf{z}\mathbf{z}^\top]L^\top = LL^\top = \Sigma$. With $\Sigma = \begin{pmatrix}1 & 0.8\\0.8 & 2\end{pmatrix}$ the correlation is $0.8/\sqrt{1 \times 2} = 0.5657$ — reproduced from 200,000 samples in the code below. The same "mean plus scaled standard noise" construction is the **reparameterisation trick** used to train VAEs.` },
      { t: "h", text: "Why the standard error is σ/√n" },
      { t: "p", text: String.raw`For i.i.d. $x_i$ with variance $\sigma^2$, independence makes the covariances vanish, so $\operatorname{Var}\big[\sum_i x_i\big] = n\sigma^2$ and $\operatorname{Var}[\bar{x}_n] = \operatorname{Var}\big[\tfrac{1}{n}\sum_i x_i\big] = \tfrac{1}{n^2}\,n\sigma^2 = \sigma^2/n$. Taking the square root: the spread of the mean is $\sigma/\sqrt{n}$. For a mini-batch of size $B$, the noise in the gradient estimate therefore falls like $1/\sqrt{B}$ — quadrupling the batch only halves the noise.` },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "p", text: "The **Distributions & sampling** tab of the simulator draws seeded samples one at a time (or 10 or 100 per step) from the distribution you choose and builds a histogram on top of the true PMF/PDF. The lower plot tracks the running sample mean against the true mean, with a shaded band of $\\pm 2\\sigma/\\sqrt{n}$." },
      { t: "sim", id: "distributions", caption: "Tab 1 is used in this lesson; tab 2 (maximum likelihood and MAP) belongs to the next lesson." },
      { t: "list", items: [
        "Choose **Gaussian**, press **Step** ten times, and compare the ragged histogram with the smooth gold curve. Switch to 100 samples per step and Play: by a few thousand samples the bars hug the curve — the histogram is a Monte Carlo estimate of the density.",
        "Watch the running mean in the lower plot. Early on it swings widely; later it stays inside the shaded band, which narrows like $1/\\sqrt{n}$. Press **New seed** several times: every path is different, yet all of them converge (law of large numbers).",
        "Choose **Bernoulli** with $p = 0.3$. The sample mean is the fraction of ones and estimates $p$; the sample variance approaches $p(1-p) = 0.21$. Move $p$ to 0.5 and note the variance is largest there.",
        "Choose **Binomial** with $n = 30$, $p = 0.3$. The PMF already looks bell-shaped: a binomial is a sum of 30 Bernoullis, and the central limit theorem is at work.",
        "Choose **Uniform** with $a = -1, b = 2$ and check that the true variance shown is $(b-a)^2/12 = 0.75$. Narrow the interval and watch the density height $1/(b-a)$ rise above 1 — densities are not probabilities.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      { t: "p", text: "One program that exercises every idea in the lesson with NumPy: a joint table, Bayes' theorem checked by simulating a million patients, expectation and variance, the five distributions against their theoretical moments, the law of large numbers, the central limit theorem on a skewed distribution, Monte Carlo estimation of a tail probability, and a correlated multivariate Gaussian built with a Cholesky factor." },
      { t: "code", lang: "python", title: "probability_numpy.py", code: PROB_NUMPY, output: PROB_NUMPY_OUT },
      { t: "p", text: "Things to read in the output:" },
      { t: "list", items: [
        "The simulated $P(\\text{disease} \\mid +) = 0.1670$ agrees with the exact 0.1667; the simulation literally counts the sick people among the positives, which is what conditioning means.",
        "Sample moments match the theory to two or three decimals with $10^5$ draws — consistent with a standard error of $\\sigma/\\sqrt{10^5} \\approx 0.003\\sigma$.",
        "The law-of-large-numbers table: the actual error at each $n$ is of the same order as $\\sigma/\\sqrt{n}$ (sometimes larger, sometimes smaller — it is itself random).",
        "CLT: the exponential distribution has skewness 2. Means of $n$ draws have skewness $2/\\sqrt{n}$ (0.894 for $n = 5$, 0.365 for $n = 30$) — the simulated 0.877 and 0.323 agree — and the fraction within $\\pm1.96$ approaches the Gaussian 0.95.",
        "Monte Carlo: the exact $P(X > 2)$ is 0.022750. With 100 samples the estimate is 0.02 (only two hits); with $10^6$ it is 0.022859, error $1.1\\times10^{-4}$, close to the standard error $\\sqrt{0.02275 \\times 0.97725/10^6} = 1.5\\times10^{-4}$.",
        "`z @ L.T` applies $\\mathbf{x} = \\boldsymbol\\mu + L\\mathbf{z}$ to every row at once; the sample covariance reproduces $\\Sigma$.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "`torch.distributions` provides distribution objects with `.sample()`, `.rsample()` (differentiable reparameterised samples), `.log_prob()`, `.mean`, `.variance` and `.entropy()`. They are how PyTorch code expresses policies in reinforcement learning, VAE encoders and decoders, and probabilistic outputs in general. The program also shows the Bernoulli inside dropout, the categorical inside a softmax classifier, Gaussian initialisation, one step of diffusion noising and the reparameterisation trick." },
          { t: "code", lang: "python", title: "probability_torch.py", code: PROB_TORCH, output: PROB_TORCH_OUT },
          { t: "list", items: [
            "`Normal(loc, scale)` takes the **standard deviation**, not the variance: `Normal(1, 2)` has variance 4. `MultivariateNormal` takes a full covariance matrix (or `scale_tril`, its Cholesky factor).",
            "`log_prob(2.0)` of $\\mathcal{N}(1, 2^2)$ is $-\\tfrac{1}{2}\\log(2\\pi \\cdot 4) - \\tfrac{(2-1)^2}{8} = -1.737086$, i.e. density 0.1760.",
            "`-Categorical(logits).log_prob(0)` equals `F.cross_entropy` — cross-entropy is the negative log-probability of the true class (derived in the [next lesson](/learn/m02/mle-map)).",
            "Diffusion forward step: $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1 - \\bar\\alpha_t}\\,\\varepsilon$, $\\varepsilon \\sim \\mathcal{N}(0, I)$ — see [Diffusion Models](/learn/m14/diffusion).",
            "`rsample` writes $z = \\mu + \\sigma\\varepsilon$, so gradients flow to $\\mu$ and $\\sigma$. The Monte Carlo gradients (1.0244, 2.0563) are noisy estimates of the exact values (1, 2); `sample()` would give no gradient at all.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Core TensorFlow has samplers in `tf.random`; Keras layers and initialisers use them internally. (Full distribution objects live in the separate TensorFlow Probability package.)" },
          { t: "code", lang: "python", title: "probability_tf.py (not executed in CI)", code: PROB_KERAS, output: "Prints sample means and variances close to (0.3, 0.21), (1.0, 4.0) and (1.0, 1.333); class frequencies close to the softmax probabilities [0.659, 0.242, 0.099]; a dropout output containing 0s and 2s; and a He-initialiser standard deviation close to sqrt(2/512) ≈ 0.0625 (Keras's HeNormal draws from a truncated normal rescaled to have that standard deviation)." },
        ] },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "p", text: "Beyond the five core distributions, these appear regularly. Each row says what it models and where you will meet it." },
      { t: "table", head: ["Distribution", "Models", "Advantages", "Limitations", "Deep learning use"], rows: [
        ["Bernoulli / Categorical", "One binary / one-of-K outcome", "Exact likelihood; trivially sampled", "Cannot express ordinal structure or uncertainty about the probabilities themselves", "Classifier outputs, token prediction, dropout"],
        ["Multinomial", "Counts of K outcomes in n trials", "Generalises the binomial", "Ignores order of draws", "Bag-of-words models; batched categorical sampling"],
        ["Gaussian (diagonal)", "Independent continuous features", "Cheap: d parameters for variances", "No correlations", "VAE encoder outputs, diffusion noise"],
        ["Gaussian (full covariance)", "Correlated continuous features", "Captures linear dependence", "$O(d^2)$ parameters, $O(d^3)$ to factor", "Gaussian processes, whitening, Kalman filters"],
        ["Laplace", "Heavy-ish tails, peak at the mean", "Robust to outliers", "Non-differentiable at the mean", "Likelihood behind L1/MAE loss; prior behind L1 regularisation"],
        ["Beta", "A probability $p \\in [0, 1]$", "Conjugate prior for Bernoulli", "Only for one probability", "Bayesian A/B testing; mixup's mixing coefficient"],
        ["Dirichlet", "A probability vector", "Conjugate prior for categorical", "One concentration controls spread", "Topic models (LDA); label smoothing intuition"],
        ["Mixture of Gaussians", "Multi-modal continuous data", "Approximates any smooth density", "Fitting needs EM or gradient methods; local optima", "Mixture density networks; clustering"],
        ["Gumbel", "Maximum of many variables", "Gumbel-max gives exact categorical samples", "Relaxation is biased", "Gumbel-softmax for differentiable discrete sampling"],
      ] },
      { t: "p", text: "Two schools of interpretation are also worth naming: the **frequentist** view (probability = long-run frequency of repeatable events; parameters are fixed unknowns) and the **Bayesian** view (probability = degree of belief; parameters themselves get distributions). Deep learning practice mostly uses frequentist estimation (maximum likelihood) with Bayesian vocabulary for regularisation — the subject of the [next lesson](/learn/m02/mle-map)." },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      { t: "list", items: [
        "**Classification outputs.** Every binary classifier (spam filters, medical image findings) outputs a Bernoulli parameter through a sigmoid; every multi-class classifier (ImageNet models) and every language model outputs a categorical distribution through a softmax.",
        "**Text generation.** Chat assistants and code-completion models generate by repeatedly sampling from $P(x_t \\mid x_{<t})$ — the chain rule run forwards — with temperature, top-k or top-p reshaping the categorical first; see [Decoding](/learn/m20/decoding).",
        "**Generative image models.** Diffusion models (the family behind Stable Diffusion) learn to reverse a process that gradually adds Gaussian noise; VAEs model data through a Gaussian latent variable.",
        "**Initialisation and regularisation.** Xavier/He initialisation draw weights from Gaussian or uniform distributions with carefully chosen variance ([Initialization](/learn/m07/initialization)); dropout samples Bernoulli masks ([Regularization](/learn/m07/regularization)).",
        "**Stochastic optimisation.** Mini-batch SGD uses a random sample of the data to form an unbiased Monte Carlo estimate of the full gradient ([Gradient Descent](/learn/m06/gradient-descent)).",
        "**Evaluation.** Accuracy on a test set of $n$ examples is a binomial proportion, with standard error $\\sqrt{p(1-p)/n}$: on 1,000 examples, 90% accuracy is uncertain by about ±1.9% (two standard errors). Calibration asks whether predicted probabilities match observed frequencies ([Classification Metrics](/learn/m17/classification-metrics)).",
        "**Decision-making under base rates.** Medical screening, fraud and anomaly detection all need Bayes' theorem to turn a detector's error rates into the probability that an alert is real.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "list", items: [
        "**Probabilistic outputs are richer than hard labels**: they express confidence, support thresholds tuned to costs, and combine correctly with priors via Bayes. But neural networks are often **miscalibrated** — a softmax probability of 0.99 is not guaranteed to be right 99% of the time.",
        "**The i.i.d. assumption makes the mathematics tractable** (likelihoods factorise, the LLN and CLT apply) but real data violates it: time series, distribution shift between training and deployment, duplicated examples.",
        "**Gaussians are convenient** (closed-form everything, CLT justification, maximum entropy for a fixed variance) but have thin tails; real errors often have outliers, for which Laplace or Student-t models are more robust.",
        "**Monte Carlo estimation is dimension-free and simple**, but converges slowly ($1/\\sqrt{n}$) and rare events need enormous samples — the 100-sample estimate of a 2% tail probability above saw only two hits.",
        "**Full covariance matrices capture correlations** at $O(d^2)$ memory and $O(d^3)$ factorisation cost, which is why deep learning almost always uses diagonal Gaussians.",
        "**Exact probabilities over structured outputs are intractable** in general (the number of sentences is astronomical); the chain rule makes them tractable by factorising into many small categorical predictions.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      { t: "list", items: [
        "**Confusing $P(A \\mid B)$ with $P(B \\mid A)$.** A 99%-sensitive test does not mean a positive is 99% reliable. Fix: write down the prior and apply Bayes, or count in a population of 10,000.",
        "**Treating a density value as a probability.** $p(x)$ for a continuous variable can be 5 or 50; only areas are probabilities. Fix: integrate over an interval, or compare densities only relative to each other.",
        "**Assuming uncorrelated means independent.** Zero covariance rules out only linear dependence ($Y = X^2$ example). Fix: test independence on the joint distribution, not on $\\rho$.",
        "**Passing a variance where a standard deviation is expected** — `np.random.normal(0, 4)` and `torch.distributions.Normal(0, 4)` both mean $\\sigma = 4$, variance 16. Fix: read the parameter name (`scale`, `std`) and check with `.variance`.",
        "**Forgetting that $\\operatorname{Var}[X + Y]$ includes $2\\operatorname{Cov}[X, Y]$.** Variances add only for uncorrelated variables. Linearity of expectation, by contrast, needs no independence.",
        "**Multiplying many probabilities directly.** The probability of a 1,000-token sentence underflows to 0.0 in floating point. Fix: add log-probabilities (the next lesson demonstrates the underflow).",
        "**Unseeded randomness in experiments.** Results change run to run and bugs cannot be reproduced. Fix: `np.random.default_rng(seed)`, `torch.manual_seed(seed)`, and report variation across several seeds.",
        "**Using the population variance where an unbiased estimate is wanted (or vice versa).** NumPy's `x.var()` divides by $n$; `torch.var` and `np.var(ddof=1)` divide by $n - 1$. For large $n$ it hardly matters; for $n = 5$ the difference is 25%.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between a PMF and a PDF?", a: "A PMF $p(x) = P(X = x)$ belongs to a discrete variable and its values are probabilities that sum to 1. A PDF belongs to a continuous variable; its values are densities (probability per unit length) that integrate to 1, can exceed 1, and give probabilities only as areas $\\int_a^b p(x)dx$. Every single point of a continuous variable has probability 0." },
      { t: "qa", level: "beginner", q: "State Bayes' theorem and name its parts.", a: "$P(H \\mid E) = P(E \\mid H)P(H)/P(E)$: posterior = likelihood × prior / evidence, where the evidence $P(E) = \\sum_h P(E \\mid h)P(h)$ normalises the posterior. It reverses a conditional probability and updates a prior belief in light of data." },
      { t: "qa", level: "beginner", q: "Why does a softmax layer output a probability distribution?", a: "Softmax $p_k = e^{z_k}/\\sum_j e^{z_j}$ makes every entry positive (exponentials) and makes them sum to 1 (division by the total), which are exactly the requirements of a categorical PMF. The network's logits $z_k$ are unnormalised log-probabilities." },
      { t: "qa", level: "beginner", q: "What does it mean that expectation is linear, and why is it useful?", a: "$\\mathbb{E}[aX + bY + c] = a\\mathbb{E}[X] + b\\mathbb{E}[Y] + c$ for any random variables, even dependent ones. It lets you compute means of sums without knowing joint distributions — e.g. the expected number of correct predictions is the sum of per-example probabilities of being correct, and a mini-batch gradient is an unbiased estimate of the full gradient." },
      { t: "qa", level: "intermediate", q: "A test is 95% sensitive and 95% specific; prevalence is 2%. What is P(disease | positive)?", a: "$P(+) = 0.95 \\times 0.02 + 0.05 \\times 0.98 = 0.019 + 0.049 = 0.068$, so $P(D \\mid +) = 0.019/0.068 = 0.279$. Despite 95% accuracy, fewer than 3 in 10 positives are true, because healthy people vastly outnumber sick ones." },
      { t: "qa", level: "intermediate", q: "How does the chain rule of probability relate to language models?", a: "Any joint distribution factorises as $P(x_1..x_T) = \\prod_t P(x_t \\mid x_{<t})$. An autoregressive language model parameterises each factor with a network that outputs a categorical over the vocabulary given the prefix. Training maximises $\\sum_t \\log P(x_t \\mid x_{<t})$ (cross-entropy per token), and generation samples tokens one at a time from these conditionals. The factorisation is exact; the only approximation is the network." },
      { t: "qa", level: "intermediate", q: "Why does the error of a Monte Carlo estimate scale as 1/√n, and what are the consequences for SGD?", a: "For i.i.d. samples, $\\operatorname{Var}[\\bar{x}_n] = \\sigma^2/n$, so the standard deviation is $\\sigma/\\sqrt{n}$. A mini-batch gradient with batch size $B$ has noise $\\propto 1/\\sqrt{B}$: doubling $B$ doubles the compute per step but reduces noise only by a factor of $\\sqrt{2}$. This diminishing return is one reason moderate batch sizes are efficient, and why learning rates are often scaled with batch size." },
      { t: "qa", level: "intermediate", q: "Give an example of uncorrelated but dependent variables.", a: "$X \\sim \\mathcal{N}(0, 1)$, $Y = X^2$. $\\operatorname{Cov}[X, Y] = \\mathbb{E}[X^3] - \\mathbb{E}[X]\\mathbb{E}[X^2] = 0 - 0 = 0$, yet $Y$ is a deterministic function of $X$. Correlation measures only linear association." },
      { t: "qa", level: "advanced", q: "How do you sample from a multivariate Gaussian, and how is this related to the reparameterisation trick?", a: "Factor $\\Sigma = LL^\\top$ (Cholesky), draw $\\mathbf{z} \\sim \\mathcal{N}(0, I)$ and return $\\boldsymbol\\mu + L\\mathbf{z}$; its covariance is $L I L^\\top = \\Sigma$. Writing the sample as a deterministic, differentiable function of the parameters plus parameter-free noise is the reparameterisation trick: in a VAE the encoder outputs $\\mu$ and $\\sigma$, and $z = \\mu + \\sigma \\odot \\varepsilon$ lets gradients of a Monte Carlo loss estimate flow back into the encoder." },
      { t: "qa", level: "advanced", q: "The CLT says sample means become Gaussian. When does it fail or converge slowly?", a: "It needs finite variance: for heavy-tailed distributions such as the Cauchy, the mean of $n$ samples has the same distribution as one sample and never concentrates. Convergence is slow for strongly skewed distributions (skewness of the mean decays only as $1/\\sqrt{n}$) and for rare-event indicators, and it also needs independence (or weak dependence); strongly correlated samples, such as consecutive steps of a Markov chain, have a much smaller effective sample size." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Prevalence 2%, sensitivity 0.9, false-positive rate 0.1. What is $P(\\text{disease} \\mid +)$? (3 decimals)", answer: 0.155, tolerance: 0.002, explain: "$P(+) = 0.9(0.02) + 0.1(0.98) = 0.018 + 0.098 = 0.116$; $P(D \\mid +) = 0.018/0.116 = 0.1552$." },
        { kind: "numeric", q: "Four fair coin flips. What is the probability of exactly two heads?", answer: 0.375, tolerance: 0.001, explain: "Binomial: $\\binom{4}{2}0.5^4 = 6/16 = 0.375$." },
        { kind: "numeric", q: "What is the variance of a Bernoulli(0.2) variable?", answer: 0.16, tolerance: 0.001, explain: "$p(1-p) = 0.2 \\times 0.8 = 0.16$." },
        { q: "If $\\operatorname{Var}[X] = 3$, what is $\\operatorname{Var}[2X + 5]$?", options: ["6", "11", "12", "17"], answer: 2, explain: "Adding a constant does not change spread; scaling by $a$ multiplies variance by $a^2$: $4 \\times 3 = 12$." },
        { q: "Which statement about a continuous PDF $p(x)$ is true?", options: ["$p(x) \\le 1$ for all $x$", "$p(x)$ is the probability that $X = x$", "$p(x)$ can exceed 1, but its integral over all $x$ is 1", "$p(x)$ must be symmetric"], answer: 2, explain: "A density is probability per unit length. Uniform(0, 0.1) has density 10 everywhere on its support, and still integrates to 1." },
        { q: "Events $A$ and $B$ are independent exactly when…", options: ["$P(A \\cap B) = 0$", "$P(A \\cap B) = P(A)P(B)$", "$P(A \\cup B) = P(A) + P(B)$", "$P(A \\mid B) = P(B \\mid A)$"], answer: 1, explain: "Independence means the joint factorises. $P(A \\cap B) = 0$ means mutually exclusive, which (for events with positive probability) is a strong form of dependence." },
        { kind: "numeric", q: "Measurements have standard deviation 2. What is the standard error of the mean of 100 of them?", answer: 0.2, tolerance: 0.001, explain: "$\\sigma/\\sqrt{n} = 2/10 = 0.2$." },
        { kind: "numeric", q: "$\\mathbb{E}[X] = 2$ and $\\mathbb{E}[Y] = 3$, and $X$ and $Y$ are strongly correlated. What is $\\mathbb{E}[2X - Y + 1]$?", answer: 2, tolerance: 0.001, explain: "Linearity holds regardless of dependence: $2(2) - 3 + 1 = 2$." },
      ] },
      { t: "exercise", title: "Two positive tests (numerical)", prompt: "Using the medical-test numbers of the Math section (prevalence 1%, sensitivity 0.99, false-positive rate 0.05), compute $P(D \\mid +, +)$ after two independent positive tests directly from the joint — without the sequential update — and confirm it matches 0.7984. Then compute the probability that a randomly chosen person tests positive twice.", hint: "Under conditional independence, $P(+,+ \\mid D) = 0.99^2$ and $P(+,+ \\mid \\neg D) = 0.05^2$.", solution: [
        { t: "steps", items: [
          { title: "Likelihoods of two positives", tex: String.raw`P(+,+\mid D) = 0.99^2 = 0.9801,\qquad P(+,+\mid \neg D) = 0.05^2 = 0.0025` },
          { title: "Evidence", tex: String.raw`P(+,+) = 0.9801(0.01) + 0.0025(0.99) = 0.009801 + 0.002475 = 0.012276` },
          { title: "Posterior", tex: String.raw`P(D \mid +,+) = \frac{0.009801}{0.012276} = 0.7984` },
        ] },
        { t: "p", text: "The batch and sequential computations agree, as they must: Bayesian updating is order-independent for conditionally independent evidence. About 1.23% of the population tests positive twice." },
      ] },
      { t: "exercise", title: "Estimate π by Monte Carlo (coding)", prompt: "Draw $n$ points uniformly in the unit square and count the fraction that fall inside the quarter disc $x^2 + y^2 \\le 1$. Use it to estimate $\\pi$ for $n = 100, 10^4, 10^6$, and compare the error with the predicted standard error.", hint: "Each point is a Bernoulli trial with success probability $\\pi/4$; the fraction is a Bernoulli mean with standard error $\\sqrt{p(1-p)/n}$, so the estimate of $\\pi$ has standard error four times that.", solution: [
        { t: "code", lang: "python", title: "mc_pi.py", code: PI_CODE, output: PI_OUT },
        { t: "p", text: "The error falls roughly tenfold for every hundredfold increase in $n$, and stays within about one to two predicted standard errors — the $1/\\sqrt{n}$ law. The standard error at $n$ samples is $4\\sqrt{0.785(0.215)/n} = 1.64/\\sqrt{n}$." },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      { t: "list", items: [
        "Axioms: $P(A) \\ge 0$, $P(\\Omega) = 1$, additivity for exclusive events. Complement: $P(A^c) = 1 - P(A)$.",
        "Marginal $P(x) = \\sum_y P(x, y)$; conditional $P(A \\mid B) = P(A \\cap B)/P(B)$; product rule $P(A \\cap B) = P(A \\mid B)P(B)$; total probability $P(A) = \\sum_i P(A \\mid B_i)P(B_i)$.",
        "Independence: $P(A \\cap B) = P(A)P(B)$. Training data is assumed i.i.d.",
        "Chain rule: $P(x_{1:T}) = \\prod_t P(x_t \\mid x_{<t})$ — the factorisation behind autoregressive language models.",
        "Bayes: posterior = likelihood × prior / evidence. Rare conditions make false positives dominate (medical example: 99% sensitive test, 1% prevalence ⇒ 16.7%).",
        "PMF sums to 1; PDF integrates to 1 and may exceed 1; CDF $F(x) = P(X \\le x)$.",
        "$\\mathbb{E}$ is linear; $\\operatorname{Var}[X] = \\mathbb{E}[X^2] - \\mathbb{E}[X]^2$; $\\operatorname{Var}[aX + c] = a^2\\operatorname{Var}[X]$; covariance matrix $\\Sigma = \\mathbb{E}[(\\mathbf{x} - \\boldsymbol\\mu)(\\mathbf{x} - \\boldsymbol\\mu)^\\top]$.",
        "Bernoulli: mean $p$, var $p(1-p)$. Binomial: $np$, $np(1-p)$. Gaussian: $\\mu$, $\\sigma^2$. Uniform: $(a+b)/2$, $(b-a)^2/12$. Categorical = softmax output.",
        "LLN: $\\bar{x}_n \\to \\mu$ with standard error $\\sigma/\\sqrt{n}$. CLT: standardised means become $\\mathcal{N}(0, 1)$. Monte Carlo: $\\mathbb{E}[f(X)] \\approx \\frac{1}{n}\\sum_i f(x_i)$.",
        "In code: seed your generators, use `torch.distributions` for `sample`/`rsample`/`log_prob`, and remember `scale` means standard deviation.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      { t: "list", items: [
        "[Maximum Likelihood and MAP Estimation](/learn/m02/mle-map) — turn the distributions of this lesson into learning: choosing parameters that make the data most probable gives MSE, binary cross-entropy and cross-entropy, and adding a prior gives weight decay.",
        "[Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) — measuring the uncertainty of a distribution and the mismatch between two of them; the information-theoretic view of the classification loss.",
        "[Classification Losses](/learn/m05/classification-losses) — the Bernoulli and categorical distributions in practice, with numerically stable implementations.",
        "[Variational Autoencoders](/learn/m14/vae) and [Diffusion Models](/learn/m14/diffusion) — where Gaussian sampling and the reparameterisation trick become the core of a model.",
      ] },
    ],
  },
};

export default lesson;
