import type { Lesson } from "@/lib/types";

const MLE_NUMPY = String.raw`import numpy as np

rng = np.random.default_rng(0)

# ---------- 1. Why logs: the likelihood of 2,000 points underflows ----------
x = rng.normal(loc=2.0, scale=1.5, size=2000)
def gauss_pdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
def gauss_loglik(x, mu, sigma):
    return np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2))
print("product of 2000 densities:", np.prod(gauss_pdf(x, 2.0, 1.5)))
print("sum of 2000 log-densities:", round(gauss_loglik(x, 2.0, 1.5), 4))

# ---------- 2. Bernoulli MLE: closed form p = k/n vs a brute-force grid ----------
flips = rng.random(50) < 0.7                     # 50 coin flips with true p = 0.7
k, n = flips.sum(), flips.size
grid = np.linspace(0.001, 0.999, 999)
loglik = k * np.log(grid) + (n - k) * np.log(1 - grid)
print(f"\nBernoulli: k = {k} heads in n = {n}; closed-form p_hat = {k / n:.3f}, "
      f"grid argmax = {grid[np.argmax(loglik)]:.3f}")

# ---------- 3. Gaussian MLE and the bias of the variance estimate ----------
mu_hat = x.mean()
var_hat = np.mean((x - mu_hat) ** 2)             # MLE divides by n, not n - 1
print(f"\nGaussian MLE on 2000 points: mu_hat = {mu_hat:.4f} (true 2.0), "
      f"sigma_hat = {np.sqrt(var_hat):.4f} (true 1.5)")

trials = rng.normal(0.0, 1.0, size=(200_000, 5))  # many datasets of size n = 5, true variance 1
v_mle = trials.var(axis=1, ddof=0)
v_unbiased = trials.var(axis=1, ddof=1)
print(f"n = 5: average MLE variance {v_mle.mean():.4f} (theory (n-1)/n = 0.8000), "
      f"average unbiased variance {v_unbiased.mean():.4f}")

# ---------- 4. MLE by gradient ascent on (mu, log sigma) ----------
mu, log_s = 0.0, 0.0                             # optimise log sigma so sigma stays positive
lr = 0.1
for step in range(301):
    s2 = np.exp(2 * log_s)
    r = x - mu
    g_mu = np.mean(r) / s2                       # d(mean log-lik)/d mu
    g_log_s = np.mean(r ** 2) / s2 - 1.0         # d(mean log-lik)/d log sigma
    if step % 75 == 0:
        print(f"step {step:3d}  mu {mu:.4f}  sigma {np.exp(log_s):.4f}  "
              f"mean log-lik {gauss_loglik(x, mu, np.exp(log_s)) / x.size:.5f}")
    mu += lr * g_mu                              # ascent: move WITH the gradient
    log_s += lr * g_log_s

# ---------- 5. Gaussian likelihood => mean squared error ----------
X = rng.uniform(-3, 3, size=40)
y = 1.5 * X - 0.5 + rng.normal(0, 0.8, size=40)
A = np.c_[X, np.ones_like(X)]
w_ls = np.linalg.lstsq(A, y, rcond=None)[0]      # minimises MSE
ws = np.linspace(0.5, 2.5, 2001)                 # scan the slope, intercept fixed at its LS value
nll = [-gauss_loglik(y, w * X + w_ls[1], 0.8) for w in ws]
print(f"\nleast-squares slope {w_ls[0]:.3f}; slope maximising Gaussian likelihood {ws[np.argmin(nll)]:.3f}")

# ---------- 6. MAP with a Beta prior: small vs large data ----------
alpha, beta = 2.0, 2.0                           # prior pseudo-counts: 1 head and 1 tail
for heads, n_flips in [(3, 3), (7, 10), (700, 1000)]:
    mle = heads / n_flips
    map_ = (heads + alpha - 1) / (n_flips + alpha + beta - 2)
    post_mean = (heads + alpha) / (n_flips + alpha + beta)
    print(f"{heads:3d}/{n_flips:<4d}  MLE {mle:.4f}   MAP {map_:.4f}   posterior mean {post_mean:.4f}")

# ---------- 7. MLE overfits tiny data; a Gaussian prior (ridge) fixes it ----------
def features(t, degree=9):
    return np.vstack([t ** d for d in range(degree + 1)]).T
t_train = np.linspace(0, 1, 10)
y_train = np.sin(2 * np.pi * t_train) + rng.normal(0, 0.2, 10)
t_test = np.linspace(0, 1, 200)
y_test = np.sin(2 * np.pi * t_test)
Phi, Phi_test = features(t_train), features(t_test)
for lam in [0.0, 1e-3]:                          # lam = sigma^2 / tau^2 (noise var / prior var)
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(10), Phi.T @ y_train)
    train_mse = np.mean((Phi @ w - y_train) ** 2)
    test_mse = np.mean((Phi_test @ w - y_test) ** 2)
    label = "MLE (no prior)    " if lam == 0 else "MAP (lambda=1e-3) "
    print(f"{label} train MSE {train_mse:.5f}  test MSE {test_mse:.4f}  max|w| {np.abs(w).max():10.1f}")`;

const MLE_NUMPY_OUT = String.raw`product of 2000 densities: 0.0
sum of 2000 log-densities: -3649.9877

Bernoulli: k = 41 heads in n = 50; closed-form p_hat = 0.820, grid argmax = 0.820

Gaussian MLE on 2000 points: mu_hat = 1.9580 (true 2.0), sigma_hat = 1.5003 (true 1.5)
n = 5: average MLE variance 0.8015 (theory (n-1)/n = 0.8000), average unbiased variance 1.0018
step   0  mu 0.0000  sigma 1.0000  mean log-lik -3.96119
step  75  mu 1.8265  sigma 1.5102  mean log-lik -1.82843
step 150  mu 1.9536  sigma 1.5003  mean log-lik -1.82461
step 225  mu 1.9578  sigma 1.5003  mean log-lik -1.82460
step 300  mu 1.9580  sigma 1.5003  mean log-lik -1.82460

least-squares slope 1.564; slope maximising Gaussian likelihood 1.564
  3/3     MLE 1.0000   MAP 0.8000   posterior mean 0.7143
  7/10    MLE 0.7000   MAP 0.6667   posterior mean 0.6429
700/1000  MLE 0.7000   MAP 0.6996   posterior mean 0.6992
MLE (no prior)     train MSE 0.00000  test MSE 0.1046  max|w|   316655.2
MAP (lambda=1e-3)  train MSE 0.02614  test MSE 0.0159  max|w|       10.2`;

const MLE_TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

torch.manual_seed(0)

# ---------- 1. Regression as MLE: learn the mean AND the noise level ----------
X = torch.linspace(-3, 3, 200).unsqueeze(1)
y = 1.5 * X - 0.5 + 0.8 * torch.randn_like(X)          # true slope 1.5, bias -0.5, noise sd 0.8

model = nn.Linear(1, 1)                                 # predicts the mean of y given x
log_sigma = nn.Parameter(torch.zeros(1))                # learned log noise standard deviation
opt = torch.optim.Adam(list(model.parameters()) + [log_sigma], lr=0.05)
for step in range(501):
    dist = Normal(loc=model(X), scale=log_sigma.exp())  # p(y | x, theta)
    nll = -dist.log_prob(y).mean()                      # maximise likelihood = minimise NLL
    opt.zero_grad()
    nll.backward()
    opt.step()
    if step % 125 == 0:
        print(f"step {step:3d}  NLL {nll.item():.4f}  w {model.weight.item():.4f}  "
              f"b {model.bias.item():.4f}  sigma {log_sigma.exp().item():.4f}")

A = torch.cat([X, torch.ones_like(X)], dim=1)
w_ls = torch.linalg.lstsq(A, y).solution.squeeze()
resid_sd = (y - A @ w_ls.unsqueeze(1)).pow(2).mean().sqrt()
print(f"closed form (least squares): w {w_ls[0]:.4f}  b {w_ls[1]:.4f}  sigma {resid_sd:.4f}")

# ---------- 2. Bernoulli likelihood = binary cross-entropy ----------
logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
nll_bern = -Bernoulli(logits=logits).log_prob(labels).mean()
bce = F.binary_cross_entropy_with_logits(logits, labels)
print(f"\nBernoulli NLL {nll_bern:.6f}  vs  BCEWithLogits {bce:.6f}")

# ---------- 3. Categorical likelihood = cross-entropy ----------
logits3 = torch.tensor([[2.0, 1.0, 0.1], [0.2, 0.3, 2.5]])
targets = torch.tensor([0, 2])
nll_cat = -Categorical(logits=logits3).log_prob(targets).mean()
print(f"Categorical NLL {nll_cat:.6f}  vs  cross_entropy {F.cross_entropy(logits3, targets):.6f}")

# ---------- 4. MAP: a Gaussian prior on weights is weight decay ----------
torch.manual_seed(1)
Xb, yb = torch.randn(16, 4), torch.randn(16, 1)
lam = 0.1                                               # lam = 1 / (n * tau^2) for the mean loss
net_a, net_b = nn.Linear(4, 1), nn.Linear(4, 1)
net_b.load_state_dict(net_a.state_dict())               # identical starting weights

opt_a = torch.optim.SGD(net_a.parameters(), lr=0.1, weight_decay=lam)
loss_a = F.mse_loss(net_a(Xb), yb)                      # weight decay applied by the optimizer
opt_a.zero_grad(); loss_a.backward(); opt_a.step()

opt_b = torch.optim.SGD(net_b.parameters(), lr=0.1)
penalty = sum((p ** 2).sum() for p in net_b.parameters())   # -log N(0, tau^2) prior up to a constant
loss_b = F.mse_loss(net_b(Xb), yb) + 0.5 * lam * penalty
opt_b.zero_grad(); loss_b.backward(); opt_b.step()

diff = (net_a.weight - net_b.weight).abs().max().item()
print(f"\nmax |weight difference| after one step, weight_decay vs explicit prior: {diff:.2e}")`;

const MLE_TORCH_OUT = String.raw`step   0  NLL 8.8125  w -0.7322  b -0.4741  sigma 1.0513
step 125  NLL 1.2958  w 1.4341  b -0.5182  sigma 1.1622
step 250  NLL 1.1555  w 1.4572  b -0.5181  sigma 0.7686
step 375  NLL 1.1555  w 1.4572  b -0.5181  sigma 0.7684
step 500  NLL 1.1555  w 1.4572  b -0.5181  sigma 0.7684
closed form (least squares): w 1.4573  b -0.5181  sigma 0.7684

Bernoulli NLL 0.365714  vs  BCEWithLogits 0.365714
Categorical NLL 0.304264  vs  cross_entropy 0.304264

max |weight difference| after one step, weight_decay vs explicit prior: 0.00e+00`;

const MLE_KERAS = String.raw`import math
import tensorflow as tf

tf.random.set_seed(0)
X = tf.reshape(tf.linspace(-3.0, 3.0, 200), (-1, 1))
y = 1.5 * X - 0.5 + 0.8 * tf.random.normal(X.shape)

# The model outputs two numbers per input: the mean and the log standard deviation.
inp = tf.keras.Input(shape=(1,))
out = tf.keras.layers.Dense(2)(inp)
model = tf.keras.Model(inp, out)

def gaussian_nll(y_true, y_pred):
    mu, log_sigma = y_pred[:, :1], y_pred[:, 1:]
    return tf.reduce_mean(0.5 * math.log(2 * math.pi) + log_sigma
                          + 0.5 * tf.square((y_true - mu) / tf.exp(log_sigma)))

model.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss=gaussian_nll)
model.fit(X, y, epochs=300, batch_size=200, verbose=0)
print("final NLL", model.evaluate(X, y, verbose=0))

# MAP with a Gaussian prior on the weights = an L2 kernel regulariser
ridge = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(1e-3))`;

const EXP_CODE = String.raw`import numpy as np

rng = np.random.default_rng(0)
x = rng.exponential(scale=1 / 2.5, size=500)     # waiting times, true rate lambda = 2.5

def loglik(lam):
    return np.sum(np.log(lam) - lam * x)         # log p(x | lam) = log lam - lam x

lam_closed = 1 / x.mean()                        # d/dlam: n/lam - sum(x) = 0
log_lam, lr = 0.0, 0.5                           # optimise log lambda so lambda stays positive
for step in range(41):
    lam = np.exp(log_lam)
    grad = (len(x) / lam - x.sum()) * lam / len(x)   # chain rule: d/dlog_lam = lam * d/dlam, averaged
    if step % 10 == 0:
        print(f"step {step:2d}  lambda {lam:.5f}  mean log-lik {loglik(lam) / len(x):.5f}")
    log_lam += lr * grad
print(f"closed form 1/mean = {lam_closed:.5f}")`;

const EXP_OUT = String.raw`step  0  lambda 1.00000  mean log-lik -0.43851
step 10  lambda 2.27635  mean log-lik -0.17562
step 20  lambda 2.28047  mean log-lik -0.17562
step 30  lambda 2.28047  mean log-lik -0.17562
step 40  lambda 2.28047  mean log-lik -0.17562
closed form 1/mean = 2.28047`;

const lesson: Lesson = {
  id: "m02/mle-map",
  title: "Maximum Likelihood and MAP Estimation",
  summary:
    "How a model learns its numbers from data. Likelihood versus probability, the i.i.d. assumption, why we take logs, the MLE answers for a coin and a bell curve (and why the variance comes out too small), how MSE and cross-entropy come straight out of maximum likelihood, MLE by gradient ascent in PyTorch, and MAP: Gaussian priors as weight decay, Laplace priors as L1, Beta priors as pseudo-counts — compared with full Bayesian inference.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m02/probability", "m02/derivatives-chain-rule", "m03/linear-logistic-regression"],
  next: ["m02/information-theory", "m05/classification-losses", "m07/regularization"],
  terms: ["likelihood", "log-likelihood", "mle", "map", "prior", "posterior", "iid", "negative-log-likelihood", "conjugate-prior", "cross-entropy", "mse", "regularization"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      { t: "p", text: "Every model in this course has **parameters** — numbers like weights and biases that training must choose. How should we choose them? **Maximum likelihood estimation (MLE)** gives the answer used almost everywhere: pick the parameters that make the training data you actually saw as probable as possible." },
      { t: "p", text: "This one idea explains the standard losses:" },
      { t: "list", items: [
        "Minimising **mean squared error (MSE)** *is* maximum likelihood when the noise is Gaussian (bell-shaped).",
        "Minimising **binary cross-entropy** *is* maximum likelihood for a yes/no output.",
        "Minimising **softmax cross-entropy** — the loss of every classifier and every language model — *is* maximum likelihood for a choice among many classes.",
      ] },
      { t: "p", text: "MLE's weakness: with little data it trusts the data completely (three heads in three flips → chance of heads exactly 1). **Maximum a posteriori (MAP) estimation** fixes this by adding a **prior** — a belief about which parameter values are sensible before you see any data. The two most common regularisers are priors in disguise: L2 (weight decay) is a Gaussian prior, and L1 is a Laplace prior." },
      { t: "p", text: "These ideas are old (Fisher in the 1910s–1920s; priors go back to Bayes and Laplace), but they are why modern losses look the way they do." },
      { t: "callout", kind: "key", text: "Training a network = maximising $\\sum_i \\log p(y_i \\mid x_i, \\theta)$ (MLE), usually plus $\\log p(\\theta)$ (MAP). $\\theta$ (theta) stands for all the parameters; $p(y_i \\mid x_i, \\theta)$ is the probability the model gives the true answer $y_i$ for input $x_i$. The output distribution decides the loss; the prior decides the regulariser." },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      { t: "list", items: [
        String.raw`[Probability, random variables and distributions](/learn/m02/probability) — the Bernoulli (coin), categorical (dice) and Gaussian (bell curve) distributions, independence, Bayes' theorem (posterior ∝ likelihood × prior) and the i.i.d. assumption.`,
        String.raw`[Derivatives and the chain rule](/learn/m02/derivatives-chain-rule) — we find a maximum by setting the derivative to zero, and we use $\frac{d}{dx}\log x = 1/x$ and the product/chain rules.`,
        String.raw`[Linear and logistic regression](/learn/m03/linear-logistic-regression) — the simplest models; we will derive their losses (MSE, binary cross-entropy) as negative log-likelihoods.`,
        String.raw`Basic gradient ascent/descent, $\theta \leftarrow \theta \pm \eta\,\nabla_\theta$ (take a small step of size $\eta$ along the slope) — covered in [Gradient Descent](/learn/m06/gradient-descent). Here we use it to climb a log-likelihood.`,
      ] },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      { t: "callout", kind: "analogy", title: "The detective and the least surprising explanation", text: "A detective finds muddy footprints, a broken window and a missing laptop. She prefers the story that makes this evidence least surprising: maximum likelihood. A good detective also asks how common each story is to begin with — burglaries beat meteorite strikes. That starting belief is what MAP adds. With lots of evidence the starting belief barely matters; with one clue it matters a lot." },
      { t: "p", text: String.raw`Here is the key shift. A formula $p(x \mid \theta)$ (read: "probability of data $x$ given parameter $\theta$") can be read two ways:` },
      { t: "list", items: [
        String.raw`**As a probability:** fix $\theta$ and vary the data $x$. It tells you how likely different datasets are.`,
        String.raw`**As a likelihood** $\mathcal{L}(\theta)$: fix the data (we already saw it) and vary $\theta$. Now it is a score for how well each parameter value explains what we saw.`,
      ] },
      { t: "p", text: "Learning is a search over $\\theta$ for the highest score." },
      { t: "p", text: "Example: a coin lands heads 7 times in 10. If its chance of heads were $p = 0.5$, this exact sequence would have probability 0.000977; if $p = 0.7$, 0.002224 — over twice as large. No $p$ beats 0.7, so 0.7 is the maximum-likelihood estimate. If you had good reason to think the coin was fair, pulling the estimate toward 0.5 would be MAP." },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "steps", items: [
        { title: "1. Choose a model family", text: String.raw`Decide what kind of distribution made the data, with unknown parameters $\theta$: coin flips $\sim$ Bernoulli($p$); measurements $\sim \mathcal{N}(\mu, \sigma^2)$, a bell curve with centre $\mu$ and spread $\sigma$. In supervised learning a network outputs the parameters, e.g. $y \sim \text{Bernoulli}(\sigma(f_\theta(x)))$ — network output $f_\theta(x)$ through a sigmoid $\sigma$.` },
        { title: "2. Assume i.i.d. data", text: String.raw`**i.i.d.** means independent and identically distributed: examples don't affect each other and all come from the same distribution. Then the probability of the whole dataset $\mathcal{D}$ is a product over the $n$ examples.`, tex: String.raw`\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} p(x_i \mid \theta)` },
        { title: "3. Take the logarithm", text: String.raw`The log only ever goes up, so the best $\theta$ stays the same. But the product becomes a sum: easier to differentiate, easy to average over mini-batches, and no underflow to zero.`, tex: String.raw`\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)` },
        { title: "4. Maximise", text: String.raw`By hand — set the derivative $\partial \ell/\partial\theta = 0$ and solve (simple models) — or by gradient ascent on $\ell$. That equals gradient descent on the **negative log-likelihood** (NLL) $-\ell(\theta)$, which is what every deep learning loss is.` },
        { title: "5. (MAP) Add a prior", text: String.raw`Choose a prior $p(\theta)$ and maximise the log-posterior. $p(\mathcal{D})$ does not depend on $\theta$, so we ignore it.`, tex: String.raw`\hat\theta_{\text{MAP}} = \arg\max_\theta \big[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\big]` },
        { title: "6. Check the answer", text: "Make sure it is a maximum (second derivative negative, or the loss really went down). Then test on held-out data — a high training likelihood alone can mean overfitting." },
      ] },
      { t: "callout", kind: "note", title: "Likelihood is not a probability distribution over θ", text: "$\\mathcal{L}(\\theta)$ does not have to add up to 1 over $\\theta$, and its height means nothing on its own — only comparisons between parameter values do. To get a real distribution over $\\theta$ you need a prior and Bayes' theorem. That result is the posterior." },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Why logs: numerical underflow" },
      { t: "p", text: String.raw`The smallest positive double is $2^{-1074} \approx 4.9\times10^{-324}$. Multiply 1,000 probabilities of 0.1 and you get $10^{-1000}$, which is stored as exactly 0.0. Now every parameter value has "likelihood 0" and you can't compare them. In log space it is $1000\log 0.1 = -2302.585$, an ordinary number. In the NumPy program below, the product of 2,000 Gaussian densities prints 0.0 while the log-likelihood is $-3649.9877$. So frameworks always use log-probabilities, computed straight from logits (e.g. log-softmax).` },
      { t: "h", text: "MLE for a Bernoulli: the estimate is the sample mean" },
      { t: "p", text: String.raw`Take $n$ flips $x_1, \dots, x_n$, each 0 or 1, with $k = \sum_i x_i$ heads. One flip has probability $p^{x_i}(1-p)^{1-x_i}$: $p$ for a head, $1-p$ for a tail. So` },
      { t: "math", tex: String.raw`\ell(p) = \sum_{i=1}^{n}\big[x_i\log p + (1-x_i)\log(1-p)\big] = k\log p + (n-k)\log(1-p)` },
      { t: "math", tex: String.raw`\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \;\Longrightarrow\; k(1-p) = (n-k)p \;\Longrightarrow\; \hat p = \frac{k}{n} = \bar{x}` },
      { t: "p", text: String.raw`The estimate $\hat p$ ("p-hat") is the fraction of heads. The second derivative $-k/p^2 - (n-k)/(1-p)^2 < 0$, so this is a maximum. **Worked example**, 7 heads in 10 flips:` },
      { t: "table", head: ["p", "likelihood $p^7(1-p)^3$", "log-likelihood ℓ(p)"], rows: [
        ["0.5", "0.000977", "−6.9315"],
        ["0.6", "0.001792", "−6.3247"],
        ["0.7", "0.002224", "−6.1086 (maximum)"],
        ["0.8", "0.001678", "−6.3903"],
        ["0.9", "0.000478", "−7.6453"],
      ] },
      { t: "p", text: String.raw`At $p = 0.5$ the gradient is $7/0.5 - 3/0.5 = 8 > 0$ (increase $p$); at $p = 0.7$ it is $7/0.7 - 3/0.3 = 10 - 10 = 0$. The curvature there is $-7/0.49 - 3/0.09 = -47.62 = -n/(\hat p(1-\hat p))$. **Gradient ascent** on the averaged log-likelihood, $p \leftarrow p + \eta\,\frac{1}{n}\frac{d\ell}{dp}$ with step size $\eta = 0.1$, starting at 0.5, visits $0.5 \to 0.58 \to 0.6293 \to 0.6596 \to 0.6776 \to 0.6878 \to \cdots \to 0.7$ (averaged gradients $0.8, 0.4926, 0.3032, 0.1800, 0.1026$) — exactly what the simulator's stepper shows.` },
      { t: "h", text: "MLE for a Gaussian: sample mean and (biased) sample variance" },
      { t: "p", text: String.raw`Data from a bell curve with mean $\mu$ and variance $\sigma^2$ (spread squared):` },
      { t: "math", tex: String.raw`\ell(\mu, \sigma^2) = \sum_{i=1}^{n}\log\mathcal{N}(x_i;\mu,\sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2` },
      { t: "math", tex: String.raw`\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0 \Rightarrow \hat\mu = \frac{1}{n}\sum_i x_i,\qquad \frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i(x_i - \hat\mu)^2 = 0 \Rightarrow \hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \hat\mu)^2` },
      { t: "p", text: String.raw`Best centre = the average; best variance = the average squared distance from it. **Example:** $x = (2.1, 3.4, 1.9, 2.8, 3.3)$. $\hat\mu = 13.5/5 = 2.7$. Deviations $(-0.6, 0.7, -0.8, 0.1, 0.6)$, squares $(0.36, 0.49, 0.64, 0.01, 0.36)$, sum 1.86. So $\hat\sigma^2 = 1.86/5 = 0.372$ and $\hat\sigma = 0.6099$, while the unbiased estimate is $1.86/4 = 0.465$.` },
      { t: "p", text: String.raw`**Why** $\hat\sigma^2$ **is biased** (wrong on average). Using $\sum_i(x_i - \bar x)^2 = \sum_i (x_i - \mu)^2 - n(\bar x - \mu)^2$ and taking expectations ($\mathbb{E}$, the average over many datasets): $\mathbb{E}\big[\sum_i (x_i-\mu)^2\big] = n\sigma^2$ and $\mathbb{E}[n(\bar x - \mu)^2] = n\cdot\sigma^2/n = \sigma^2$, so` },
      { t: "math", tex: String.raw`\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = \frac{1}{n}(n\sigma^2 - \sigma^2) = \frac{n-1}{n}\,\sigma^2` },
      { t: "p", text: String.raw`The MLE comes out too small because $\hat\mu$ is fitted to the same points, so it sits closer to them than the true $\mu$. Dividing by $n - 1$ (Bessel's correction) removes the bias. For $n = 5$ the MLE averages $0.8\sigma^2$ (the code's simulation: 0.8015). The bias vanishes as $n \to \infty$ — typical: MLE is **consistent** (it reaches the truth with enough data) but not always unbiased.` },
      { t: "h", text: "From likelihoods to loss functions" },
      { t: "p", text: String.raw`The network $f_\theta(x)$ outputs the parameters of $p(y \mid x, \theta)$, and the loss is the average NLL. Pick the output distribution and the loss follows.` },
      { t: "p", text: String.raw`**Gaussian output ⇒ MSE.** The network predicts the centre, $\sigma$ fixed: $y \sim \mathcal{N}(f_\theta(x), \sigma^2)$:` },
      { t: "math", tex: String.raw`-\log p(y \mid x, \theta) = \frac{(y - f_\theta(x))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)` },
      { t: "p", text: String.raw`The second term ignores $\theta$ and $1/(2\sigma^2)$ is a positive constant, so maximising the likelihood is the same as minimising $\frac{1}{n}\sum_i (y_i - f_\theta(x_i))^2$ — the MSE. Example: $y = 3$, $f = 2.5$, $\sigma = 1$ gives NLL $= 0.25/2 + 0.9189 = 1.0439$. If $\sigma$ is learned too (as in the PyTorch code), the model learns its own uncertainty; its MLE is the root-mean-square residual. A **Laplace** output, $p(y) \propto e^{-|y - f|/b}$, gives the MAE loss instead — see [Regression Losses](/learn/m05/regression-losses).` },
      { t: "p", text: String.raw`**Bernoulli output ⇒ binary cross-entropy.** Say $y \sim \text{Bernoulli}(\hat p)$, where $\hat p = \sigma(f_\theta(x))$ is the predicted chance that $y = 1$:` },
      { t: "math", tex: String.raw`-\log p(y \mid x, \theta) = -\big[y\log\hat p + (1-y)\log(1-\hat p)\big]` },
      { t: "p", text: String.raw`With $\hat p = 0.8$: the loss is $-\log 0.8 = 0.2231$ if $y = 1$ and $-\log 0.2 = 1.6094$ if $y = 0$. Confident and wrong costs a lot.` },
      { t: "p", text: String.raw`**Categorical output ⇒ cross-entropy.** Say $y \sim \text{Categorical}(\hat{\mathbf p})$, where $\hat{\mathbf p} = \operatorname{softmax}(f_\theta(x))$ holds one probability per class. With a one-hot target $\mathbf y$ (1 for the true class $c$, 0 elsewhere):` },
      { t: "math", tex: String.raw`-\log p(y = c \mid x, \theta) = -\log \hat p_c = -\sum_{k} y_k \log \hat p_k` },
      { t: "p", text: String.raw`This is the cross-entropy between the one-hot target and the prediction. More generally, maximising average log-likelihood is the same as minimising the KL divergence (a measure of mismatch) from the data distribution to the model, $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$ — see [Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) and [Classification Losses](/learn/m05/classification-losses).` },
      { t: "h", text: "MAP estimation" },
      { t: "p", text: String.raw`Bayes' theorem for parameters: $p(\theta \mid \mathcal{D}) = p(\mathcal{D} \mid \theta)\,p(\theta)/p(\mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$ — **posterior ∝ likelihood × prior** (∝ means "proportional to"). MAP picks the posterior's peak. In plain words: data loss plus a penalty from the prior:` },
      { t: "math", tex: String.raw`\hat\theta_{\text{MAP}} = \arg\max_\theta\Big[\sum_{i=1}^{n}\log p(x_i \mid \theta) + \log p(\theta)\Big] = \arg\min_\theta\Big[\underbrace{-\sum_i \log p(x_i \mid \theta)}_{\text{data loss}}\;\underbrace{-\log p(\theta)}_{\text{regulariser}}\Big]` },
      { t: "p", text: String.raw`**Gaussian prior ⇒ L2 / weight decay.** Each weight $w_j \sim \mathcal{N}(0, \tau^2)$: "probably near 0, spread $\tau$". Then $-\log p(\mathbf w) = \frac{1}{2\tau^2}\|\mathbf w\|_2^2 + \text{const}$, where $\|\mathbf w\|_2^2$ is the sum of squared weights. For regression with noise $\sigma$, multiplying by $2\sigma^2$ gives ridge regression:` },
      { t: "math", tex: String.raw`\hat{\mathbf w}_{\text{MAP}} = \arg\min_{\mathbf w}\;\sum_i (y_i - \mathbf w^\top\mathbf x_i)^2 + \lambda\|\mathbf w\|_2^2,\qquad \lambda = \frac{\sigma^2}{\tau^2}` },
      { t: "p", text: String.raw`A tight prior (small $\tau$) means a large $\lambda$: strong pull toward zero. The gradient of $\frac{\lambda}{2}\|\mathbf w\|^2$ is $\lambda\mathbf w$, so each SGD step multiplies the weights by $(1 - \eta\lambda)$ — hence the name **weight decay**.` },
      { t: "p", text: String.raw`**Laplace prior ⇒ L1.** With $p(w_j) = \frac{1}{2b}e^{-|w_j|/b}$, $-\log p(\mathbf w) = \frac{1}{b}\|\mathbf w\|_1 + \text{const}$ (sum of absolute weights): L1 regularisation, the lasso. The Laplace density has a sharp peak at 0, so MAP sets many weights exactly to zero — a sparse solution. See [Regularization](/learn/m07/regularization).` },
      { t: "h", text: "Worked example: Beta–Bernoulli and pseudo-counts" },
      { t: "p", text: String.raw`For a probability $p$, the natural prior is the **Beta distribution**, $\text{Beta}(p;\alpha,\beta) \propto p^{\alpha-1}(1-p)^{\beta-1}$. Times the Bernoulli likelihood $p^k(1-p)^{n-k}$ it gives another Beta — a **conjugate prior** (posterior in the same family):` },
      { t: "math", tex: String.raw`p(p \mid \mathcal{D}) \propto p^{k+\alpha-1}(1-p)^{n-k+\beta-1} = \text{Beta}(k+\alpha,\; n-k+\beta),\qquad \hat p_{\text{MAP}} = \frac{k+\alpha-1}{n+\alpha+\beta-2},\qquad \mathbb{E}[p \mid \mathcal{D}] = \frac{k+\alpha}{n+\alpha+\beta}` },
      { t: "p", text: String.raw`The prior acts like **pseudo-counts** (imaginary data): for MAP, Beta($\alpha, \beta$) behaves as if you had already seen $\alpha - 1$ extra heads and $\beta - 1$ extra tails. With Beta(2, 2) — one imaginary head and one imaginary tail:` },
      { t: "table", head: ["Data", "MLE k/n", "MAP (k+1)/(n+2)", "Posterior mean (k+2)/(n+4)"], rows: [
        ["3 heads / 3", "1.0000", "0.8000", "0.7143"],
        ["7 heads / 10", "0.7000", "0.6667", "0.6429"],
        ["700 heads / 1000", "0.7000", "0.6996", "0.6992"],
      ] },
      { t: "p", text: "With three flips, MLE says tails is impossible; the prior keeps the estimate sensible. With 1,000 flips the two pseudo-counts hardly matter and all three estimates agree. Beta(1, 1) is the flat (uniform) prior, for which MAP equals MLE." },
      { t: "h", text: "Worked example: Gaussian mean with a Gaussian prior" },
      { t: "p", text: String.raw`Data $x_i \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma$ known, and prior $\mu \sim \mathcal{N}(\mu_0, \tau^2)$ ("I think $\mu$ is near $\mu_0$"). Set the derivative of the log-posterior to zero:` },
      { t: "math", tex: String.raw`\frac{1}{\sigma^2}\sum_i (x_i - \mu) - \frac{\mu - \mu_0}{\tau^2} = 0 \;\Longrightarrow\; \hat\mu_{\text{MAP}} = \frac{\frac{n}{\sigma^2}\bar x + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}},\qquad \operatorname{Var}[\mu \mid \mathcal{D}] = \Big(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\Big)^{-1}` },
      { t: "p", text: String.raw`A weighted average of data mean and prior mean, each weighted by its **precision** (1 / variance): the surer source counts more. With $x = (2.2, 1.6, 2.9)$, $\bar x = 2.2333$, $\sigma = \tau = 1$, $\mu_0 = 0$: $\hat\mu_{\text{MAP}} = 3(2.2333)/(3 + 1) = 1.675$, with posterior standard deviation $\sqrt{1/4} = 0.5$. Keep the same sample mean but raise $n$ to 30 and you get 2.1613; $n = 300$ gives 2.2259. The data overwhelm the prior.` },
      { t: "h", text: "MLE vs MAP vs full Bayesian inference" },
      { t: "p", text: String.raw`MLE and MAP return one best value, a **point estimate**. Full Bayesian inference keeps the whole posterior and averages predictions over it — the **posterior predictive** $p(x_{\text{new}} \mid \mathcal{D}) = \int p(x_{\text{new}} \mid \theta)\,p(\theta \mid \mathcal{D})\,d\theta$. For the Beta–Bernoulli model this equals the posterior mean: after 3 heads in 3 with a Beta(2, 2) prior, $P(\text{next is heads}) = 5/7 = 0.7143$. MLE would say probability 1 — a certainty nobody should have after three flips. With small data the three differ a lot; as $n$ grows, the posterior narrows (its width shrinks like $1/\sqrt{n}$) and all three agree.` },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "p", text: "The upper plot shows the log-likelihood (or, with MAP on, the log-posterior) across parameter values. The gradient-ascent point is gold; its tangent line (slope = gradient) is clay. The lower plot shows likelihood, prior and posterior as densities." },
      { t: "sim", id: "distributions", caption: "Switch to the second tab, 'Maximum likelihood / MAP'." },
      { t: "list", items: [
        "With the **Gaussian mean** model, press **Step** a few times. Each update is $\\mu \\leftarrow \\mu + \\eta\\,\\frac{1}{n}\\partial\\ell/\\partial\\mu$ (the narration shows the numbers). The point settles at the dashed line $\\bar x$, the closed-form MLE. Bottom ticks are the data.",
        "Choose **Coin flips**, set $n = 10$, and Resample until you get 7 heads. Start at $p_0 = 0.5$ with $\\eta = 0.1$ and step: you should see $0.5 \\to 0.58 \\to 0.6293 \\to \\dots$ from the Math section.",
        "Raise $\\eta$ past the stability limit $2/\\text{curvature}$ (the panel says when) and watch the steps overshoot. Curvature of the averaged log-likelihood is $1/\\sigma^2$ for the Gaussian mean, $1/(p(1-p))$ for the Bernoulli.",
        "Raise $n$ from 1 to 200: the likelihood narrows (width like $1/\\sqrt{n}$) — more data, less uncertainty.",
        "Tick **MAP**. For the Gaussian, set $\\mu_0 = 0$, $\\tau = 0.5$, $n = 3$: the posterior sits between prior and likelihood, and MAP is pulled toward 0. At $n = 200$ the posterior lands on the likelihood: MAP ≈ MLE.",
        "Coin flips with MAP, $n = 3$, resample until all heads: the MLE sits at the edge ($p = 1$); a Beta(2, 2) prior gives MAP = 0.8. Then try Beta(20, 20), a strong belief the coin is fair.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      { t: "p", text: "One NumPy program for the whole lesson: underflow, the Bernoulli and Gaussian MLEs, the variance bias, gradient ascent on $(\\mu, \\log\\sigma)$, Gaussian likelihood = least squares, Beta–Bernoulli MAP, and a degree-9 polynomial where MLE overfits ten points but MAP (ridge) does not." },
      { t: "code", lang: "python", title: "mle_map_numpy.py", code: MLE_NUMPY, output: MLE_NUMPY_OUT },
      { t: "list", items: [
        "The product underflows to exactly 0.0; the sum of logs is fine. A few hundred points are enough to trigger this.",
        "The 50 flips happened to contain 41 heads, so $\\hat p = 0.82$ although the true $p$ is 0.7. The MLE fits *this* sample. With $n = 50$ the standard error (typical size of the estimate's error) is $\\sqrt{0.7(0.3)/50} = 0.065$.",
        "Gradient ascent works on $\\log\\sigma$, a standard trick that keeps $\\sigma > 0$. It reaches exactly the closed-form $\\hat\\mu = 1.9580$, $\\hat\\sigma = 1.5003$. The $\\log\\sigma$ update uses $\\partial\\bar\\ell/\\partial\\log\\sigma = \\overline{r^2}/\\sigma^2 - 1$, where $r$ is the residual $x - \\mu$.",
        "The slope that maximises the Gaussian likelihood equals the least-squares slope (1.564): MSE and Gaussian MLE are the same objective.",
        "The unregularised degree-9 fit through 10 points has training error 0 but weights around $3\\times10^5$ and a test MSE of 0.1046. A Gaussian prior with $\\lambda = 10^{-3}$ keeps weights around 10 and cuts test MSE to 0.0159.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Part 1 maximises a Gaussian likelihood with `torch.distributions.Normal`, learning the mean (slope and bias) and the noise level. Parts 2–3 show PyTorch's classification losses are exactly Bernoulli and categorical NLLs. Part 4 shows `weight_decay` gives the same update as adding the Gaussian-prior penalty $\\frac{\\lambda}{2}\\|\\theta\\|^2$." },
          { t: "code", lang: "python", title: "mle_map_torch.py", code: MLE_TORCH, output: MLE_TORCH_OUT },
          { t: "list", items: [
            "Gradient-based MLE reaches the closed-form least-squares answer (w 1.4572 vs 1.4573) and learns $\\sigma = 0.7684$, the RMS residual — the MLE of the noise level.",
            "The NLL of 1.1555 per point is $\\frac{1}{2}\\log(2\\pi\\hat\\sigma^2) + \\frac{1}{2} = \\frac{1}{2}\\log(2\\pi \\cdot 0.7684^2) + 0.5$: at the optimum the squared-error term averages exactly $\\frac{1}{2}$.",
            "`BCEWithLogitsLoss` and `cross_entropy` take **logits** (raw scores before sigmoid/softmax) and compute log-probabilities stably inside (log-sigmoid, log-softmax). They are the Bernoulli and categorical NLLs, not made-up recipes.",
            "For plain SGD, `weight_decay=λ` adds $\\lambda\\theta$ to every gradient — the same as MAP with a Gaussian prior. For Adam, `AdamW` shrinks the weights directly instead, which is not the same as an L2 penalty — see [Momentum, Adam and AdamW](/learn/m06/momentum-adaptive).",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "A Keras model with two outputs (mean and log standard deviation) trained with a hand-written Gaussian NLL, plus an L2 kernel regulariser — Keras's Gaussian prior on the weights." },
          { t: "code", lang: "python", title: "mle_keras.py (not executed in CI)", code: MLE_KERAS, output: "Prints the final mean Gaussian negative log-likelihood on the training data after 300 full-batch epochs; it should end close to ½·log(2π·0.8²) + ½ ≈ 1.20 per point, the value implied by the noise standard deviation 0.8 (the exact number depends on the sampled noise; the PyTorch run above reached 1.1555)." },
        ] },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "table", head: ["Approach", "What it computes", "Pros", "Cons", "Typical use"], rows: [
        ["Maximum likelihood (MLE)", "$\\arg\\max_\\theta \\log p(\\mathcal{D} \\mid \\theta)$", "No prior to choose; consistent; asymptotically efficient", "Overfits small data; can land on edges (p̂ = 0 or 1)", "Default training objective of neural networks"],
        ["MAP, Gaussian prior", "MLE + $\\frac{1}{2\\tau^2}\\|\\theta\\|_2^2$", "Shrinks weights smoothly; closed form for linear models", "Prior variance is a hyperparameter; changes if you reparameterise", "Weight decay, ridge regression"],
        ["MAP, Laplace prior", "MLE + $\\frac{1}{b}\\|\\theta\\|_1$", "Sparse solutions (feature selection)", "Not differentiable at 0; needs subgradients or proximal steps", "Lasso, pruning-friendly training"],
        ["Full Bayesian", "Whole posterior; predictions averaged over it", "Calibrated uncertainty; sound with little data", "Usually intractable for networks; expensive", "Small-data science, Bayesian optimisation, Gaussian processes"],
        ["Approximate Bayesian (variational, MC dropout, ensembles)", "An approximate posterior", "Uncertainty at moderate cost", "Approximation quality hard to check", "Safety-critical predictions"],
        ["Unbiased / corrected estimators", "e.g. variance with $n-1$", "No systematic error", "Not MLE; may vary more", "Reporting statistics, batch-norm running variance"],
        ["Method of moments", "Match sample moments (mean, variance…) to theory", "Simple, closed form", "Less efficient than MLE", "Quick first estimates"],
        ["Expectation–maximisation (EM)", "MLE with hidden variables, alternating E and M steps", "Likelihood never goes down", "Local optima; slow", "Gaussian mixtures, hidden Markov models"],
        ["Maximising the ELBO", "A lower bound on the log-likelihood of a latent-variable model", "Trainable with gradients via reparameterisation", "Gap between bound and true likelihood", "Variational autoencoders ([VAE](/learn/m14/vae))"],
      ] },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      { t: "list", items: [
        "**Every classifier and language model** is trained by maximum likelihood of a categorical output — softmax cross-entropy over classes or over the next token. Language-model **perplexity** is exp(average NLL per token) ([Language Modelling](/learn/m13/language-modeling)).",
        "**Regression networks** trained with MSE are Gaussian MLE; predicting a variance too (as in the PyTorch example) gives per-prediction uncertainty.",
        "**Weight decay** in large models, including Transformers, is the MAP view of a Gaussian prior (exactly for SGD; decoupled in AdamW).",
        "**Generative models** maximise likelihood directly (autoregressive models, normalising flows) or a bound on it (VAEs, and diffusion models' objective).",
        "**A/B testing and click-through rates** with few observations use Beta–Bernoulli posteriors and pseudo-counts, so a handful of events never gives 0% or 100%. Add-one (Laplace) smoothing of n-gram counts is the same idea.",
        "**Calibration and probabilistic forecasting** are scored by held-out log-likelihood, which rewards honest probabilities.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "list", items: [
        "**MLE is general**: one recipe gives the right loss for any output type. With enough data it is consistent and, under standard conditions, asymptotically the most efficient estimator.",
        "**MLE overfits with little data or many parameters** (degree-9 polynomial, p̂ = 1 after three heads). Big networks lean on priors/regularisers, early stopping and data augmentation.",
        "**MAP is almost free** — one extra loss term — and gives regularisers a clear meaning. But the prior's strength is a hyperparameter tuned on validation data, and MAP gives one point, no uncertainty.",
        "**MAP depends on how you write the parameter**: a density's peak moves under a non-linear change of variables ($\\sigma$ vs $\\log\\sigma$); the MLE does not.",
        "**Full Bayesian inference** handles uncertainty and small data best, but the posterior over millions of weights is intractable (too costly to compute); approximations (ensembles, variational methods, MC dropout) trade accuracy for cost.",
        "**The likelihood is only as good as the model family**: with heavy-tailed noise, Gaussian MLE (MSE) gets dragged by outliers; with noisy labels, cross-entropy chases them.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      { t: "list", items: [
        "**Multiplying probabilities instead of adding log-probabilities.** The product underflows to 0. Fix: sum `log_prob`, computed from logits (`log_softmax`, `BCEWithLogitsLoss`), not `log` of a probability that may already be 0.",
        "**Calling the likelihood a distribution over parameters.** $\\mathcal{L}(\\theta)$ does not integrate to 1, so \"θ = 0.7 has likelihood 0.002, so it is unlikely\" means nothing. Only ratios between parameter values matter.",
        "**Climbing when you should descend (or the reverse).** MLE maximises $\\ell$; libraries minimise losses. Fix: minimise the NLL $-\\ell$ with a standard optimizer. A sign error makes the loss explode.",
        "**Optimising a limited parameter directly.** Steps on $\\sigma$ or $p$ can leave the allowed range ($\\sigma < 0$, $p > 1$). Fix: optimise $\\log\\sigma$ or a logit.",
        "**Summing the NLL in one place and averaging in another.** A summed loss grows with batch size, changing the effective learning rate and prior strength. Fix: average the data term and set the prior strength per example ($\\lambda = 1/(n\\tau^2)$ for a mean loss, as in the PyTorch code).",
        "**Using the MLE variance on small samples without thinking.** It is too low by the factor $(n-1)/n$ — 20% for $n = 5$. Fix: use `ddof=1` when you need an unbiased estimate.",
        "**Assuming weight decay equals L2 regularisation in every optimizer.** True for SGD, false for Adam: Adam's per-weight rescaling also rescales the L2 gradient. Fix: use AdamW for decoupled decay.",
        "**Reporting training likelihood as model quality.** It can mean overfitting. Fix: measure it on held-out data.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between probability and likelihood?", a: "Both use the same formula $p(x \\mid \\theta)$. As a **probability**, $\\theta$ is fixed and we ask how probable different data are; it sums/integrates to 1 over $x$. As a **likelihood**, the data are fixed (already observed) and we compare parameter values; it is a function of $\\theta$ that need not integrate to 1." },
      { t: "qa", level: "beginner", q: "Why do we maximise the log-likelihood instead of the likelihood?", a: "The log only ever increases, so the best parameters are the same. It turns a product into a sum — easier to differentiate and to average over mini-batches — and avoids underflow (a product of thousands of probabilities rounds to 0)." },
      { t: "qa", level: "beginner", q: "What is the MLE of a coin's bias after k heads in n flips?", a: "$\\hat p = k/n$. From $\\ell(p) = k\\log p + (n-k)\\log(1-p)$, setting $k/p - (n-k)/(1-p) = 0$ gives $p = k/n$. The second derivative is negative, so it is a maximum." },
      { t: "qa", level: "intermediate", q: "Show that minimising MSE is maximum likelihood.", a: "Assume $y = f_\\theta(x) + \\varepsilon$ with noise $\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)$. Then $-\\log p(y \\mid x) = (y - f_\\theta(x))^2/(2\\sigma^2) + \\frac{1}{2}\\log(2\\pi\\sigma^2)$. Summed over i.i.d. examples, the second term and $1/(2\\sigma^2)$ do not depend on $\\theta$, so maximising the likelihood means minimising $\\sum_i (y_i - f_\\theta(x_i))^2$." },
      { t: "qa", level: "intermediate", q: "Why is the MLE of the Gaussian variance biased, and by how much?", a: "It measures spread around $\\bar x$, which is fitted to the same data and sits closer to the points than the true mean. $\\mathbb{E}[\\frac{1}{n}\\sum(x_i - \\bar x)^2] = \\frac{n-1}{n}\\sigma^2$. Dividing by $n - 1$ removes the bias, which vanishes anyway as $n \\to \\infty$." },
      { t: "qa", level: "intermediate", q: "Explain L2 regularisation as MAP estimation.", a: "Put an independent Gaussian prior $\\mathcal{N}(0, \\tau^2)$ on each weight. The log-posterior is the log-likelihood plus $-\\|\\mathbf w\\|^2/(2\\tau^2)$ + const. Maximising it means minimising NLL $+ \\frac{1}{2\\tau^2}\\|\\mathbf w\\|^2$ — an L2 penalty that gets stronger as the prior variance gets smaller (for Gaussian regression, $\\lambda = \\sigma^2/\\tau^2$). A Laplace prior gives L1 instead." },
      { t: "qa", level: "intermediate", q: "What is a conjugate prior? Give the Beta–Bernoulli example.", a: "A prior is conjugate to a likelihood if the posterior is in the same family as the prior. A Beta($\\alpha, \\beta$) prior times a Bernoulli likelihood with $k$ successes in $n$ gives Beta($k + \\alpha$, $n - k + \\beta$). MAP is $(k + \\alpha - 1)/(n + \\alpha + \\beta - 2)$ and the posterior mean $(k + \\alpha)/(n + \\alpha + \\beta)$." },
      { t: "qa", level: "advanced", q: "How does maximum likelihood relate to KL divergence?", a: "Average log-likelihood $\\frac{1}{n}\\sum_i\\log p_\\theta(x_i)$ estimates $\\mathbb{E}_{p_{\\text{data}}}[\\log p_\\theta(x)]$. Since $\\mathrm{KL}(p_{\\text{data}} \\| p_\\theta) = \\mathbb{E}_{p_{\\text{data}}}[\\log p_{\\text{data}}] - \\mathbb{E}_{p_{\\text{data}}}[\\log p_\\theta]$ and the first term does not depend on $\\theta$, maximising expected log-likelihood is minimising this forward KL. Forward KL is 'mass-covering': the model pays heavily for giving near-zero probability anywhere the data has mass." },
      { t: "qa", level: "advanced", q: "When do MLE, MAP and full Bayesian inference give noticeably different predictions, and why do they agree for large n?", a: "They differ when data are scarce relative to model flexibility. MLE trusts the sample ($\\hat p = 1$ after three heads), MAP is pulled toward the prior's peak, and the Bayesian predictive averages over plausible parameters, giving hedged probabilities. The log-likelihood grows linearly in $n$ while the log-prior stays fixed, so the prior fades. The posterior concentrates around the MLE with width $O(1/\\sqrt{n})$ (Bernstein–von Mises), so all three predictions coincide." },
      { t: "qa", level: "advanced", q: "Why is MAP not invariant to reparameterisation, while MLE is?", a: "The likelihood is a function of the parameter value, so under any one-to-one transform $\\phi = g(\\theta)$ the best value maps to the best value. A prior is a density: changing variables multiplies it by the Jacobian $|d\\theta/d\\phi|$, which moves its peak. So the MAP for $\\sigma$ and the MAP for $\\log\\sigma$ (transformed back) usually differ, whereas $\\hat\\phi_{\\text{MLE}} = g(\\hat\\theta_{\\text{MLE}})$." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A coin shows 6 heads in 8 flips. What is the maximum-likelihood estimate of $p$?", answer: 0.75, tolerance: 0.001, explain: "$\\hat p = k/n = 6/8 = 0.75$." },
        { kind: "numeric", q: "Same data (6 heads in 8) with a Beta(2, 2) prior. What is the MAP estimate?", answer: 0.7, tolerance: 0.001, explain: "$(k + \\alpha - 1)/(n + \\alpha + \\beta - 2) = 7/10 = 0.7$." },
        { kind: "numeric", q: "Data $(1, 3, 5)$ from a Gaussian. What is the maximum-likelihood estimate of the variance? (4 decimals)", answer: 2.6667, tolerance: 0.001, explain: "$\\hat\\mu = 3$; squared deviations 4, 0, 4; MLE divides by $n = 3$: $8/3 = 2.6667$. (The unbiased estimate is $8/2 = 4$.)" },
        { kind: "numeric", q: "The true variance is 8 and $n = 4$. What is the expected value of the MLE variance estimate?", answer: 6, tolerance: 0.001, explain: "$\\mathbb{E}[\\hat\\sigma^2] = \\frac{n-1}{n}\\sigma^2 = \\frac{3}{4}\\times 8 = 6$." },
        { q: "Training a regression network with MSE corresponds to maximum likelihood under which noise model?", options: ["Laplace noise", "Gaussian noise with fixed variance", "Bernoulli noise", "Uniform noise"], answer: 1, explain: "The Gaussian NLL is $(y - f)^2/(2\\sigma^2)$ plus a constant. Laplace noise would give MAE." },
        { q: "Which prior on the weights corresponds to L1 regularisation?", options: ["Gaussian", "Uniform", "Laplace", "Beta"], answer: 2, explain: "$-\\log$ of a Laplace density is $|w|/b$ + const — an L1 penalty, which pushes weights to exactly zero." },
        { q: "Why do implementations sum log-probabilities rather than multiply probabilities?", options: ["Logs make the model more accurate", "Products of many probabilities underflow to 0, and sums are easier to differentiate and average", "Logs remove the need for a prior", "Probabilities can be negative"], answer: 1, explain: "The best parameters don't change (log only increases), while stability and convenience improve." },
        { q: "As the amount of data grows, the MAP estimate…", options: ["approaches the prior mean", "approaches the MLE", "oscillates", "becomes undefined"], answer: 1, explain: "The log-likelihood grows linearly with $n$ while the log-prior stays fixed, so the prior's share of the say vanishes." },
      ] },
      { t: "exercise", title: "Beta prior with lopsided data (numerical)", prompt: "You see 9 heads in 10 flips and use a Beta(3, 3) prior. Compute (a) the MLE, (b) the posterior distribution, (c) the MAP estimate, (d) the posterior mean, i.e. the Bayesian probability that the next flip is heads.", hint: "Posterior is Beta(k + α, n − k + β).", solution: [
        { t: "steps", items: [
          { title: "MLE", tex: String.raw`\hat p_{\text{MLE}} = 9/10 = 0.9` },
          { title: "Posterior", tex: String.raw`\text{Beta}(9 + 3,\; 1 + 3) = \text{Beta}(12, 4)` },
          { title: "MAP (mode)", tex: String.raw`\hat p_{\text{MAP}} = \frac{12 - 1}{12 + 4 - 2} = \frac{11}{14} = 0.7857` },
          { title: "Posterior mean / predictive", tex: String.raw`P(\text{heads next} \mid \mathcal{D}) = \frac{12}{12 + 4} = 0.75` },
        ] },
        { t: "p", text: "For MAP, the Beta(3, 3) prior is worth two imaginary heads and two imaginary tails. It pulls the estimate from 0.9 toward 0.5." },
      ] },
      { t: "exercise", title: "MLE for an exponential rate (coding)", prompt: "Waiting times $x_i \\ge 0$ follow an exponential distribution $p(x \\mid \\lambda) = \\lambda e^{-\\lambda x}$, where $\\lambda$ is the rate (events per unit time). Derive the MLE of $\\lambda$, then check it numerically by gradient ascent on $\\log\\lambda$ for 500 samples with true $\\lambda = 2.5$.", hint: "$\\ell(\\lambda) = n\\log\\lambda - \\lambda\\sum_i x_i$. For the ascent, use $\\partial\\ell/\\partial\\log\\lambda = \\lambda\\,\\partial\\ell/\\partial\\lambda$.", solution: [
        { t: "math", tex: String.raw`\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_i x_i = 0 \;\Longrightarrow\; \hat\lambda = \frac{n}{\sum_i x_i} = \frac{1}{\bar x}` },
        { t: "code", lang: "python", title: "exponential_mle.py", code: EXP_CODE, output: EXP_OUT },
        { t: "p", text: "Gradient ascent reaches $1/\\bar x = 2.28047$ in about 20 steps. It differs from the true 2.5 because of sampling noise: the standard error is roughly $\\lambda/\\sqrt{n} = 2.5/\\sqrt{500} = 0.11$, so this sample is about two standard errors low. MLE finds the best fit to the sample, not the truth itself." },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      { t: "list", items: [
        "Likelihood $\\mathcal{L}(\\theta) = p(\\mathcal{D} \\mid \\theta)$: data fixed, parameter varies. It is not a distribution over $\\theta$.",
        "i.i.d. data: $\\ell(\\theta) = \\sum_i \\log p(x_i \\mid \\theta)$. Logs keep the same best value, turn products into sums, and avoid underflow.",
        "Bernoulli: $\\hat p = k/n$. Gaussian: $\\hat\\mu = \\bar x$, $\\hat\\sigma^2 = \\frac{1}{n}\\sum(x_i - \\bar x)^2$ with $\\mathbb{E}[\\hat\\sigma^2] = \\frac{n-1}{n}\\sigma^2$.",
        "Loss = negative log-likelihood: Gaussian ⇒ MSE, Laplace ⇒ MAE, Bernoulli ⇒ binary cross-entropy, categorical ⇒ cross-entropy. MLE minimises $\\mathrm{KL}(p_{\\text{data}} \\| p_\\theta)$.",
        "Numerical MLE: gradient ascent on $\\ell$ = gradient descent on the NLL; optimise $\\log\\sigma$ and logits, not limited parameters.",
        "MAP: $\\arg\\max[\\log p(\\mathcal{D} \\mid \\theta) + \\log p(\\theta)]$; posterior ∝ likelihood × prior.",
        "Gaussian prior ⇒ L2 / weight decay with $\\lambda = \\sigma^2/\\tau^2$; Laplace prior ⇒ L1 (sparsity).",
        "Beta–Bernoulli: posterior Beta($k+\\alpha, n-k+\\beta$); MAP $(k+\\alpha-1)/(n+\\alpha+\\beta-2)$; posterior mean $(k+\\alpha)/(n+\\alpha+\\beta)$; priors act as pseudo-counts.",
        "Small data: MLE overfits, MAP shrinks, full Bayes hedges. Large data: all three agree.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      { t: "list", items: [
        "[Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) — the information-theory view of the NLL, and why cross-entropy measures mismatch.",
        "[Classification Losses](/learn/m05/classification-losses) — binary and categorical cross-entropy in practice: logits, numerical stability, label smoothing and focal loss.",
        "[Regularization](/learn/m07/regularization) — L1, L2, dropout and early stopping: this lesson's priors as practical tools against overfitting.",
        "[Optimization: Objectives, Convexity and Minima](/learn/m02/optimization-math) — what makes a log-likelihood easy or hard to maximise.",
        "[Variational Autoencoders](/learn/m14/vae) — maximum likelihood for models with hidden variables, via the ELBO.",
      ] },
    ],
  },
};

export default lesson;
