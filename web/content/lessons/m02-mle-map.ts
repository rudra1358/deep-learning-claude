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
    "How parameters are learned from data: likelihood versus probability, the i.i.d. assumption, why we take logs, closed-form MLE for the Bernoulli and Gaussian (and the bias of the variance), how MSE, binary cross-entropy and cross-entropy fall out of maximum likelihood, MLE by gradient ascent in PyTorch, and MAP estimation — Gaussian priors as weight decay, Laplace priors as L1, Beta–Bernoulli pseudo-counts — contrasted with full Bayesian inference.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m02/probability", "m02/derivatives-chain-rule", "m03/linear-logistic-regression"],
  next: ["m02/information-theory", "m05/classification-losses", "m07/regularization"],
  terms: ["likelihood", "log-likelihood", "mle", "map", "prior", "posterior", "iid", "negative-log-likelihood", "conjugate-prior", "cross-entropy", "mse", "regularization"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      { t: "p", text: "Every model in this course has **parameters** — numbers such as weights and biases that we must choose. **Maximum likelihood estimation (MLE)** is the principle behind almost every choice: pick the parameters under which the observed training data would have been most probable. It sounds modest, but it explains a surprising amount of deep learning practice. Minimising mean squared error *is* maximum likelihood under Gaussian noise. Minimising binary cross-entropy *is* maximum likelihood for a Bernoulli output. Minimising softmax cross-entropy — the loss of every classifier and every language model — *is* maximum likelihood for a categorical output." },
      { t: "p", text: "MLE has a known weakness: with little data it trusts the data completely and overfits (three heads in three flips gives an estimated probability of heads of exactly 1). **Maximum a posteriori (MAP) estimation** repairs this by combining the likelihood with a **prior** — a distribution expressing which parameter values we consider plausible before seeing data. The two most common regularisers in deep learning turn out to be priors in disguise: L2 regularisation (weight decay) is a Gaussian prior on the weights, and L1 regularisation is a Laplace prior." },
      { t: "p", text: "Both ideas were developed long before neural networks — maximum likelihood was championed by R. A. Fisher in the 1910s–1920s, and the prior/posterior view goes back to Bayes and Laplace — but they are the reason modern loss functions look the way they do. This lesson derives the estimators, shows the numerical reason we always work with logarithms, implements MLE by gradient ascent, and compares MLE, MAP and full Bayesian inference as the amount of data changes." },
      { t: "callout", kind: "key", text: "Training a network = maximising $\\sum_i \\log p(y_i \\mid x_i, \\theta)$ (MLE), usually plus $\\log p(\\theta)$ (MAP). The choice of output distribution decides the loss; the choice of prior decides the regulariser." },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      { t: "list", items: [
        String.raw`[Probability, random variables and distributions](/learn/m02/probability) — the Bernoulli, categorical and Gaussian distributions, independence, Bayes' theorem (posterior ∝ likelihood × prior) and the i.i.d. assumption.`,
        String.raw`[Derivatives and the chain rule](/learn/m02/derivatives-chain-rule) — we find maxima by setting derivatives to zero, and we need $\frac{d}{dx}\log x = 1/x$ and the product/chain rules.`,
        String.raw`[Linear and logistic regression](/learn/m03/linear-logistic-regression) — the simplest models whose losses (MSE, binary cross-entropy) we will derive as negative log-likelihoods.`,
        String.raw`Basic gradient ascent/descent, $\theta \leftarrow \theta \pm \eta\,\nabla_\theta$ — covered in [Gradient Descent](/learn/m06/gradient-descent); here it is used to maximise a log-likelihood numerically.`,
      ] },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      { t: "callout", kind: "analogy", title: "The detective and the least surprising explanation", text: "A detective finds muddy footprints, a broken window and a missing laptop. Of all the stories that could explain the evidence, she prefers the one under which the evidence is least surprising. That is maximum likelihood. A good detective also weighs how plausible each story was to begin with — a burglary is more common than a meteorite strike — and that prior plausibility is what MAP adds. With a mountain of evidence, the prior barely matters; with one clue, it matters a lot." },
      { t: "p", text: "The key shift in perspective: a distribution $p(x \\mid \\theta)$ can be read two ways. Fix the parameter $\\theta$ and vary the data $x$, and it is a **probability** — it tells you how likely different datasets are. Fix the data (we have already observed it) and vary $\\theta$, and the same formula becomes the **likelihood** $\\mathcal{L}(\\theta)$ — a score for how well each parameter value explains what we saw. Learning is a search over $\\theta$ for the highest score." },
      { t: "p", text: "Concretely: a coin lands heads 7 times in 10. If the coin's bias were $p = 0.5$, this outcome would have probability 0.000977 (for this exact sequence); if $p = 0.7$, probability 0.002224 — more than twice as large. No value of $p$ does better than 0.7, so 0.7 is the maximum-likelihood estimate. If you had good reason to believe the coin was fair, you might shade the estimate toward 0.5 — that is MAP." },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "steps", items: [
        { title: "1. Choose a model family", text: String.raw`Decide what distribution generated the data, up to unknown parameters $\theta$: coin flips $\sim$ Bernoulli($p$); measurements $\sim \mathcal{N}(\mu, \sigma^2)$; for supervised learning, a conditional model $p(y \mid x, \theta)$ whose parameters come out of a network, e.g. $y \sim \text{Bernoulli}(\sigma(f_\theta(x)))$.` },
        { title: "2. Assume i.i.d. data", text: String.raw`Examples are independent and identically distributed, so the probability of the whole dataset factorises into a product over examples.`, tex: String.raw`\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} p(x_i \mid \theta)` },
        { title: "3. Take the logarithm", text: String.raw`The log is strictly increasing, so it has the same maximiser; it turns the product into a sum (easy to differentiate and to average over mini-batches) and avoids numerical underflow.`, tex: String.raw`\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)` },
        { title: "4. Maximise", text: String.raw`Either analytically — set $\partial \ell/\partial\theta = 0$ and solve (possible for simple models) — or numerically by gradient ascent on $\ell$, equivalently gradient descent on the **negative log-likelihood** (NLL) $-\ell(\theta)$, which is what every deep learning loss is.` },
        { title: "5. (MAP) Add a prior", text: String.raw`Choose $p(\theta)$, then maximise the log-posterior. The evidence $p(\mathcal{D})$ does not depend on $\theta$ and can be ignored.`, tex: String.raw`\hat\theta_{\text{MAP}} = \arg\max_\theta \big[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\big]` },
        { title: "6. Check the answer", text: "Confirm it is a maximum (second derivative negative / the loss actually decreased), and evaluate on held-out data — a high training likelihood alone can mean overfitting." },
      ] },
      { t: "callout", kind: "note", title: "Likelihood is not a probability distribution over θ", text: "$\\mathcal{L}(\\theta)$ need not integrate to 1 over $\\theta$, and its height has no meaning on its own — only comparisons between parameter values do. Turning it into a distribution over $\\theta$ requires a prior and Bayes' theorem: that is the posterior." },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Why logs: numerical underflow" },
      { t: "p", text: String.raw`The smallest positive double-precision number is $2^{-1074} \approx 4.9\times10^{-324}$. A product of 1,000 probabilities of 0.1 is $10^{-1000}$, far below it, so it is stored as exactly 0.0 — and every parameter value then has "likelihood 0", making comparison impossible. In log space the same quantity is $1000\log 0.1 = -2302.585$, an ordinary number. In the NumPy program below, the product of 2,000 Gaussian densities evaluates to 0.0 while the log-likelihood is $-3649.9877$. Every framework therefore works with log-probabilities (and computes them from logits directly, e.g. via log-softmax).` },
      { t: "h", text: "MLE for a Bernoulli: the estimate is the sample mean" },
      { t: "p", text: String.raw`Data $x_1, \dots, x_n \in \{0, 1\}$, with $k = \sum_i x_i$ ones. Each has probability $p^{x_i}(1-p)^{1-x_i}$, so` },
      { t: "math", tex: String.raw`\ell(p) = \sum_{i=1}^{n}\big[x_i\log p + (1-x_i)\log(1-p)\big] = k\log p + (n-k)\log(1-p)` },
      { t: "math", tex: String.raw`\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \;\Longrightarrow\; k(1-p) = (n-k)p \;\Longrightarrow\; \hat p = \frac{k}{n} = \bar{x}` },
      { t: "p", text: String.raw`The second derivative $-k/p^2 - (n-k)/(1-p)^2 < 0$, so this is a maximum. **Worked example**, 7 heads in 10 flips:` },
      { t: "table", head: ["p", "likelihood $p^7(1-p)^3$", "log-likelihood ℓ(p)"], rows: [
        ["0.5", "0.000977", "−6.9315"],
        ["0.6", "0.001792", "−6.3247"],
        ["0.7", "0.002224", "−6.1086 (maximum)"],
        ["0.8", "0.001678", "−6.3903"],
        ["0.9", "0.000478", "−7.6453"],
      ] },
      { t: "p", text: String.raw`At $p = 0.5$ the gradient is $7/0.5 - 3/0.5 = 8 > 0$ (increase $p$); at $p = 0.7$ it is $7/0.7 - 3/0.3 = 10 - 10 = 0$; and the curvature there is $-7/0.49 - 3/0.09 = -47.62 = -n/(\hat p(1-\hat p))$. **Gradient ascent** on the averaged log-likelihood, $p \leftarrow p + \eta\,\frac{1}{n}\frac{d\ell}{dp}$ with $\eta = 0.1$, starting at 0.5, visits $0.5 \to 0.58 \to 0.6293 \to 0.6596 \to 0.6776 \to 0.6878 \to \cdots \to 0.7$ (averaged gradients $0.8, 0.4926, 0.3032, 0.1800, 0.1026$) — exactly what the simulator's stepper shows.` },
      { t: "h", text: "MLE for a Gaussian: sample mean and (biased) sample variance" },
      { t: "math", tex: String.raw`\ell(\mu, \sigma^2) = \sum_{i=1}^{n}\log\mathcal{N}(x_i;\mu,\sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2` },
      { t: "math", tex: String.raw`\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0 \Rightarrow \hat\mu = \frac{1}{n}\sum_i x_i,\qquad \frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i(x_i - \hat\mu)^2 = 0 \Rightarrow \hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \hat\mu)^2` },
      { t: "p", text: String.raw`**Worked example:** $x = (2.1, 3.4, 1.9, 2.8, 3.3)$. $\hat\mu = 13.5/5 = 2.7$. Deviations $(-0.6, 0.7, -0.8, 0.1, 0.6)$, squares $(0.36, 0.49, 0.64, 0.01, 0.36)$, sum 1.86. So $\hat\sigma^2 = 1.86/5 = 0.372$ and $\hat\sigma = 0.6099$, whereas the unbiased estimate is $1.86/4 = 0.465$.` },
      { t: "p", text: String.raw`**Bias of** $\hat\sigma^2$. Using $\sum_i(x_i - \bar x)^2 = \sum_i (x_i - \mu)^2 - n(\bar x - \mu)^2$ and taking expectations, $\mathbb{E}\big[\sum_i (x_i-\mu)^2\big] = n\sigma^2$ and $\mathbb{E}[n(\bar x - \mu)^2] = n\cdot\sigma^2/n = \sigma^2$, so` },
      { t: "math", tex: String.raw`\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = \frac{1}{n}(n\sigma^2 - \sigma^2) = \frac{n-1}{n}\,\sigma^2` },
      { t: "p", text: String.raw`The MLE systematically underestimates the variance because $\hat\mu$ is fitted to the same data and sits closer to the points than the true $\mu$ does. Dividing by $n - 1$ (Bessel's correction) removes the bias. For $n = 5$ the MLE averages $0.8\sigma^2$ — confirmed by simulation in the code (0.8015). The bias vanishes as $n \to \infty$, which is typical: MLE is **consistent** (converges to the truth) but not always unbiased.` },
      { t: "h", text: "From likelihoods to loss functions" },
      { t: "p", text: String.raw`In supervised learning the network $f_\theta(x)$ outputs the parameters of $p(y \mid x, \theta)$, and the loss is the average negative log-likelihood.` },
      { t: "p", text: String.raw`**Gaussian output ⇒ MSE.** Model $y \sim \mathcal{N}(f_\theta(x), \sigma^2)$ with fixed $\sigma$:` },
      { t: "math", tex: String.raw`-\log p(y \mid x, \theta) = \frac{(y - f_\theta(x))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)` },
      { t: "p", text: String.raw`The second term does not depend on $\theta$, and $1/(2\sigma^2)$ is a positive constant, so maximising the likelihood is identical to minimising $\frac{1}{n}\sum_i (y_i - f_\theta(x_i))^2$ — the MSE. Example: $y = 3$, $f = 2.5$, $\sigma = 1$ gives NLL $= 0.25/2 + 0.9189 = 1.0439$. If $\sigma$ is also learned (as in the PyTorch code), the model learns its own uncertainty; its MLE is the root-mean-square residual. A **Laplace** output distribution, $p(y) \propto e^{-|y - f|/b}$, gives the MAE loss instead — see [Regression Losses](/learn/m05/regression-losses).` },
      { t: "p", text: String.raw`**Bernoulli output ⇒ binary cross-entropy.** Model $y \sim \text{Bernoulli}(\hat p)$ with $\hat p = \sigma(f_\theta(x))$:` },
      { t: "math", tex: String.raw`-\log p(y \mid x, \theta) = -\big[y\log\hat p + (1-y)\log(1-\hat p)\big]` },
      { t: "p", text: String.raw`With $\hat p = 0.8$: the loss is $-\log 0.8 = 0.2231$ if $y = 1$ and $-\log 0.2 = 1.6094$ if $y = 0$.` },
      { t: "p", text: String.raw`**Categorical output ⇒ cross-entropy.** Model $y \sim \text{Categorical}(\hat{\mathbf p})$ with $\hat{\mathbf p} = \operatorname{softmax}(f_\theta(x))$; with one-hot target $\mathbf y$,` },
      { t: "math", tex: String.raw`-\log p(y = c \mid x, \theta) = -\log \hat p_c = -\sum_{k} y_k \log \hat p_k` },
      { t: "p", text: String.raw`This is the cross-entropy between the one-hot target and the prediction. More generally, maximising the average log-likelihood is equivalent to minimising the KL divergence from the data distribution to the model, $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$ — developed in [Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) and applied in [Classification Losses](/learn/m05/classification-losses).` },
      { t: "h", text: "MAP estimation" },
      { t: "p", text: String.raw`Bayes' theorem for parameters: $p(\theta \mid \mathcal{D}) = p(\mathcal{D} \mid \theta)\,p(\theta)/p(\mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$ — **posterior ∝ likelihood × prior**. MAP picks the mode of the posterior:` },
      { t: "math", tex: String.raw`\hat\theta_{\text{MAP}} = \arg\max_\theta\Big[\sum_{i=1}^{n}\log p(x_i \mid \theta) + \log p(\theta)\Big] = \arg\min_\theta\Big[\underbrace{-\sum_i \log p(x_i \mid \theta)}_{\text{data loss}}\;\underbrace{-\log p(\theta)}_{\text{regulariser}}\Big]` },
      { t: "p", text: String.raw`**Gaussian prior ⇒ L2 / weight decay.** With $w_j \sim \mathcal{N}(0, \tau^2)$ independently, $-\log p(\mathbf w) = \frac{1}{2\tau^2}\|\mathbf w\|_2^2 + \text{const}$. For regression with noise $\sigma$, multiplying the objective by $2\sigma^2$ gives ridge regression:` },
      { t: "math", tex: String.raw`\hat{\mathbf w}_{\text{MAP}} = \arg\min_{\mathbf w}\;\sum_i (y_i - \mathbf w^\top\mathbf x_i)^2 + \lambda\|\mathbf w\|_2^2,\qquad \lambda = \frac{\sigma^2}{\tau^2}` },
      { t: "p", text: String.raw`A tight prior (small $\tau$) means a large $\lambda$ — strong shrinkage toward zero. The gradient of $\frac{\lambda}{2}\|\mathbf w\|^2$ is $\lambda\mathbf w$, so each SGD step multiplies the weights by $(1 - \eta\lambda)$ — hence the name **weight decay**.` },
      { t: "p", text: String.raw`**Laplace prior ⇒ L1.** With $p(w_j) = \frac{1}{2b}e^{-|w_j|/b}$, $-\log p(\mathbf w) = \frac{1}{b}\|\mathbf w\|_1 + \text{const}$: L1 regularisation (the lasso). The Laplace density has a sharp peak at 0, so the MAP estimate sets many weights exactly to zero — a sparse solution. See [Regularization](/learn/m07/regularization).` },
      { t: "h", text: "Worked example: Beta–Bernoulli and pseudo-counts" },
      { t: "p", text: String.raw`For a probability $p$, the natural prior is the **Beta distribution**, $\text{Beta}(p;\alpha,\beta) \propto p^{\alpha-1}(1-p)^{\beta-1}$. Multiplying by the Bernoulli likelihood $p^k(1-p)^{n-k}$ gives a posterior of the same family — the Beta is a **conjugate prior**:` },
      { t: "math", tex: String.raw`p(p \mid \mathcal{D}) \propto p^{k+\alpha-1}(1-p)^{n-k+\beta-1} = \text{Beta}(k+\alpha,\; n-k+\beta),\qquad \hat p_{\text{MAP}} = \frac{k+\alpha-1}{n+\alpha+\beta-2},\qquad \mathbb{E}[p \mid \mathcal{D}] = \frac{k+\alpha}{n+\alpha+\beta}` },
      { t: "p", text: String.raw`The prior acts like **pseudo-counts**: for the MAP estimate, Beta($\alpha, \beta$) behaves as if we had already seen $\alpha - 1$ extra heads and $\beta - 1$ extra tails. With Beta(2, 2) — one imaginary head and one imaginary tail:` },
      { t: "table", head: ["Data", "MLE k/n", "MAP (k+1)/(n+2)", "Posterior mean (k+2)/(n+4)"], rows: [
        ["3 heads / 3", "1.0000", "0.8000", "0.7143"],
        ["7 heads / 10", "0.7000", "0.6667", "0.6429"],
        ["700 heads / 1000", "0.7000", "0.6996", "0.6992"],
      ] },
      { t: "p", text: "With three flips, MLE claims tails is impossible; the prior keeps the estimate sensible. With 1,000 flips the prior's two pseudo-counts are negligible and all three estimates agree. Beta(1, 1) is the uniform prior, for which MAP equals MLE." },
      { t: "h", text: "Worked example: Gaussian mean with a Gaussian prior" },
      { t: "p", text: String.raw`Data $x_i \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma$ known, prior $\mu \sim \mathcal{N}(\mu_0, \tau^2)$. Setting the derivative of the log-posterior to zero:` },
      { t: "math", tex: String.raw`\frac{1}{\sigma^2}\sum_i (x_i - \mu) - \frac{\mu - \mu_0}{\tau^2} = 0 \;\Longrightarrow\; \hat\mu_{\text{MAP}} = \frac{\frac{n}{\sigma^2}\bar x + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}},\qquad \operatorname{Var}[\mu \mid \mathcal{D}] = \Big(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\Big)^{-1}` },
      { t: "p", text: String.raw`A precision-weighted average of the data mean and the prior mean. With $x = (2.2, 1.6, 2.9)$, $\bar x = 2.2333$, $\sigma = \tau = 1$, $\mu_0 = 0$: $\hat\mu_{\text{MAP}} = 3(2.2333)/(3 + 1) = 1.675$, with posterior standard deviation $\sqrt{1/4} = 0.5$. Keeping the same sample mean but increasing $n$ to 30 gives 2.1613, and $n = 300$ gives 2.2259: the data overwhelm the prior.` },
      { t: "h", text: "MLE vs MAP vs full Bayesian inference" },
      { t: "p", text: String.raw`MLE and MAP both return a single **point estimate**. Full Bayesian inference keeps the whole posterior and averages predictions over it — the **posterior predictive** $p(x_{\text{new}} \mid \mathcal{D}) = \int p(x_{\text{new}} \mid \theta)\,p(\theta \mid \mathcal{D})\,d\theta$. For the Beta–Bernoulli model this integral is the posterior mean: after 3 heads in 3 with a Beta(2, 2) prior, $P(\text{next is heads}) = 5/7 = 0.7143$. MLE would predict heads with probability 1 — a certainty no one should have after three flips. With small data the three approaches differ markedly; as $n$ grows, the posterior concentrates (its width shrinks like $1/\sqrt{n}$) and all three converge.` },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "p", text: "Open the **Maximum likelihood / MAP** tab. The upper plot shows the log-likelihood (or, with MAP on, the log-posterior) as a function of the parameter, with the current gradient-ascent iterate in gold and its tangent line — whose slope is the gradient — in clay. The lower plot shows the likelihood, prior and posterior as normalised densities over the parameter." },
      { t: "sim", id: "distributions", caption: "Switch to the second tab, 'Maximum likelihood / MAP'." },
      { t: "list", items: [
        "With the **Gaussian mean** model, press **Step** repeatedly. Each update is $\\mu \\leftarrow \\mu + \\eta\\,\\frac{1}{n}\\partial\\ell/\\partial\\mu$; the narration shows the numbers. The iterate converges to the dashed line at $\\bar x$, the closed-form MLE. The data points appear as ticks along the bottom of the plot.",
        "Choose **Coin flips**, set $n = 10$, and Resample until you get 7 heads. Start at $p_0 = 0.5$ with $\\eta = 0.1$ and step: you should reproduce $0.5 \\to 0.58 \\to 0.6293 \\to \\dots$ from the Math section.",
        "Increase $\\eta$ until the panel reports it above the stability limit $2/\\text{curvature}$ and watch the iterates overshoot and oscillate. Curvature of the averaged log-likelihood is $1/\\sigma^2$ for the Gaussian mean and $1/(p(1-p))$ for the Bernoulli.",
        "Increase $n$ from 1 to 200 and note how the likelihood in the lower plot narrows: more data, less uncertainty about the parameter (the width shrinks like $1/\\sqrt{n}$).",
        "Tick **MAP**. For the Gaussian, set $\\mu_0 = 0$, $\\tau = 0.5$, and $n = 3$: the posterior sits between the prior and the likelihood, and the MAP estimate is pulled well toward 0. Raise $n$ to 200: the posterior collapses onto the likelihood and MAP ≈ MLE.",
        "For coin flips with MAP, set $n = 3$ and resample until all three are heads: the MLE is at the boundary (estimated $p = 1$), while a Beta(2, 2) prior gives MAP = 0.8. Then try Beta(20, 20) — a strong belief that the coin is fair.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      { t: "p", text: "A NumPy program covering the whole lesson: underflow of raw likelihoods, the Bernoulli and Gaussian MLEs (closed form and brute force), the bias of the MLE variance, MLE by gradient ascent on $(\\mu, \\log\\sigma)$, the equivalence of Gaussian likelihood and least squares, Beta–Bernoulli MAP at three data sizes, and a degree-9 polynomial showing MLE overfitting ten points while MAP (ridge) does not." },
      { t: "code", lang: "python", title: "mle_map_numpy.py", code: MLE_NUMPY, output: MLE_NUMPY_OUT },
      { t: "list", items: [
        "The product of densities underflows to exactly 0.0; the sum of logs is a normal number. This is not a corner case — any dataset of a few hundred points triggers it.",
        "The 50 flips happened to contain 41 heads, so $\\hat p = 0.82$ although the true $p$ is 0.7: the MLE estimates the parameter that best explains *this* sample, and with $n = 50$ the standard error is $\\sqrt{0.7(0.3)/50} = 0.065$.",
        "Gradient ascent optimises $\\log\\sigma$ rather than $\\sigma$, a standard reparameterisation that keeps $\\sigma > 0$ without constraints. It converges to exactly the closed-form $\\hat\\mu = 1.9580$, $\\hat\\sigma = 1.5003$. The update for $\\log\\sigma$ uses $\\partial\\bar\\ell/\\partial\\log\\sigma = \\overline{r^2}/\\sigma^2 - 1$.",
        "The slope that maximises the Gaussian likelihood is the least-squares slope (1.564): MSE and Gaussian MLE are the same objective.",
        "The unregularised degree-9 fit through 10 points has training error 0 but weights of size $3\\times10^5$ and a test MSE of 0.1046. The Gaussian prior with $\\lambda = 10^{-3}$ keeps weights around 10 and cuts test MSE to 0.0159.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "Part 1 trains a linear model by explicitly maximising a Gaussian likelihood with `torch.distributions.Normal` — learning both the mean (slope and bias) and the noise level. Parts 2–3 show that PyTorch's standard classification losses are exactly Bernoulli and categorical negative log-likelihoods. Part 4 shows that the optimizer's `weight_decay` produces the same update as adding the Gaussian-prior penalty $\\frac{\\lambda}{2}\\|\\theta\\|^2$ to the loss." },
          { t: "code", lang: "python", title: "mle_map_torch.py", code: MLE_TORCH, output: MLE_TORCH_OUT },
          { t: "list", items: [
            "Gradient-based MLE reaches the closed-form least-squares solution (w 1.4572 vs 1.4573) and learns $\\sigma = 0.7684$, the RMS residual — the MLE of the noise level.",
            "The NLL of 1.1555 per point is $\\frac{1}{2}\\log(2\\pi\\hat\\sigma^2) + \\frac{1}{2} = \\frac{1}{2}\\log(2\\pi \\cdot 0.7684^2) + 0.5$ — at the optimum the squared-error term averages exactly $\\frac{1}{2}$.",
            "`BCEWithLogitsLoss` and `cross_entropy` take **logits** and compute log-probabilities stably internally (log-sigmoid, log-softmax). They are not heuristics; they are the negative log-likelihoods of the Bernoulli and categorical models.",
            "For plain SGD, `weight_decay=λ` adds $\\lambda\\theta$ to every gradient — identical to MAP with a Gaussian prior. For Adam, `AdamW` applies the decay directly to the weights instead, which is not equivalent to an L2 penalty — see [Momentum, Adam and AdamW](/learn/m06/momentum-adaptive).",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "A Keras model with two outputs per example (mean and log standard deviation) trained with a hand-written Gaussian negative log-likelihood, and an L2 kernel regulariser — the Keras spelling of a Gaussian prior on the weights." },
          { t: "code", lang: "python", title: "mle_keras.py (not executed in CI)", code: MLE_KERAS, output: "Prints the final mean Gaussian negative log-likelihood on the training data after 300 full-batch epochs; it should end close to ½·log(2π·0.8²) + ½ ≈ 1.20 per point, the value implied by the noise standard deviation 0.8 (the exact number depends on the sampled noise; the PyTorch run above reached 1.1555)." },
        ] },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "table", head: ["Approach", "What it computes", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Maximum likelihood (MLE)", "$\\arg\\max_\\theta \\log p(\\mathcal{D} \\mid \\theta)$", "No prior to choose; consistent; asymptotically efficient", "Overfits small data; can sit on boundaries (p̂ = 0 or 1)", "Default training objective of neural networks"],
        ["MAP, Gaussian prior", "MLE + $\\frac{1}{2\\tau^2}\\|\\theta\\|_2^2$", "Shrinks weights smoothly; closed form for linear models", "Prior variance is a hyperparameter; not reparameterisation-invariant", "Weight decay, ridge regression"],
        ["MAP, Laplace prior", "MLE + $\\frac{1}{b}\\|\\theta\\|_1$", "Sparse solutions (feature selection)", "Non-differentiable at 0; needs subgradients or proximal steps", "Lasso, pruning-friendly training"],
        ["Full Bayesian", "Whole posterior; predictions averaged over it", "Calibrated uncertainty; principled with little data", "Posterior usually intractable for networks; expensive", "Small-data science, Bayesian optimisation, Gaussian processes"],
        ["Approximate Bayesian (variational, MC dropout, ensembles)", "A tractable approximation of the posterior", "Uncertainty estimates at moderate cost", "Approximation quality hard to verify", "Uncertainty in safety-critical predictions"],
        ["Unbiased / corrected estimators", "e.g. variance with $n-1$", "No systematic error", "Not MLE; may have higher variance", "Reporting statistics, batch-norm running variance"],
        ["Method of moments", "Match sample moments to theoretical ones", "Simple, closed form", "Less efficient than MLE", "Quick initial estimates"],
        ["Expectation–maximisation (EM)", "MLE with latent variables, alternating E and M steps", "Monotone likelihood increase", "Local optima; slow", "Gaussian mixtures, hidden Markov models"],
        ["Maximising the ELBO", "A lower bound on the log-likelihood of a latent-variable model", "Trainable with gradients via reparameterisation", "Gap between bound and true likelihood", "Variational autoencoders ([VAE](/learn/m14/vae))"],
      ] },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      { t: "list", items: [
        "**Every classifier and language model** is trained by maximum likelihood of a categorical output — softmax cross-entropy over classes or over the next token. Language-model **perplexity** is the exponential of the average negative log-likelihood per token ([Language Modelling](/learn/m13/language-modeling)).",
        "**Regression networks** (price prediction, depth estimation, forecasting) trained with MSE are Gaussian MLE; networks that also predict a variance (heteroscedastic regression, as in the PyTorch example) give per-prediction uncertainty.",
        "**Weight decay** in training large models, including Transformers, is the MAP view of a Gaussian prior (exactly so for SGD; decoupled in AdamW).",
        "**Generative models** maximise likelihood directly (autoregressive models, normalising flows) or a bound on it (VAEs; diffusion models' training objective is derived from a variational bound on the likelihood).",
        "**A/B testing and click-through-rate estimation** with few observations use Beta–Bernoulli posteriors and pseudo-counts to avoid estimating 0% or 100% from a handful of events; Laplace (add-one) smoothing of n-gram counts is the same idea.",
        "**Calibration and probabilistic forecasting** are evaluated with held-out log-likelihood, which rewards honest probabilities rather than only correct argmax decisions.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "list", items: [
        "**MLE is principled and general**: one recipe produces the right loss for any output type, and with enough data it is consistent and asymptotically the most efficient estimator under standard regularity conditions.",
        "**MLE overfits with little data or too many parameters**: it trusts every quirk of the sample (degree-9 polynomial, p̂ = 1 after three heads). Neural networks with millions of parameters rely on priors/regularisers, early stopping and data augmentation to compensate.",
        "**MAP is nearly free** — one extra term in the loss — and gives a regulariser a clear meaning. But the prior's strength is a hyperparameter tuned on validation data, and MAP still gives a single point, not uncertainty.",
        "**MAP depends on the parameterisation**: the mode of a density changes under a non-linear change of variables (e.g. optimising $\\sigma$ versus $\\log\\sigma$), whereas the MLE does not.",
        "**Full Bayesian inference** handles uncertainty and small data best, but the posterior over millions of weights is intractable; practical approximations (ensembles, variational methods, MC dropout) trade accuracy for cost.",
        "**The likelihood is only as good as the model family**: if the noise is heavy-tailed, Gaussian MLE (MSE) is dragged by outliers; if labels are noisy, cross-entropy chases them. Choosing the distribution is a modelling decision.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      { t: "list", items: [
        "**Multiplying probabilities instead of adding log-probabilities.** The product underflows to 0 for any realistic dataset. Fix: always sum `log_prob`, and compute log-probabilities from logits (`log_softmax`, `BCEWithLogitsLoss`) rather than taking `log` of a probability that may already have rounded to 0.",
        "**Calling the likelihood a probability distribution over parameters.** $\\mathcal{L}(\\theta)$ does not integrate to 1 over $\\theta$; statements like \"θ = 0.7 has likelihood 0.002, so it is unlikely\" are meaningless. Only ratios between parameter values matter.",
        "**Ascending when you should descend (or vice versa).** MLE maximises $\\ell$; libraries minimise losses. Fix: minimise the NLL $-\\ell$ with a standard optimizer; a sign error makes the loss explode.",
        "**Optimising a constrained parameter directly.** Gradient steps on $\\sigma$ or $p$ can leave the valid range ($\\sigma < 0$, $p > 1$). Fix: optimise $\\log\\sigma$ or a logit and transform.",
        "**Summing the NLL in one place and averaging in another.** A summed loss scales with batch size and changes the effective learning rate and the relative strength of the prior. Fix: average the data term and express the prior strength per example ($\\lambda = 1/(n\\tau^2)$ for a mean loss, as in the PyTorch code).",
        "**Using the MLE variance for small samples without thinking.** It is biased low by the factor $(n-1)/n$ — 20% for $n = 5$. Fix: use `ddof=1` when an unbiased estimate is needed.",
        "**Assuming weight decay equals L2 regularisation in every optimizer.** True for SGD, false for Adam: the adaptive rescaling also rescales the L2 gradient. Fix: use AdamW for decoupled decay and understand which you are using.",
        "**Reporting training likelihood as model quality.** A higher training likelihood can mean overfitting. Fix: evaluate log-likelihood (or loss) on held-out data.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between probability and likelihood?", a: "Both use the same formula $p(x \\mid \\theta)$. As a **probability**, $\\theta$ is fixed and we ask how probable different data are; it sums/integrates to 1 over $x$. As a **likelihood**, the data are fixed (observed) and we compare parameter values; it is a function of $\\theta$ that need not integrate to 1." },
      { t: "qa", level: "beginner", q: "Why do we maximise the log-likelihood instead of the likelihood?", a: "The log is monotonic, so the maximiser is the same. It turns a product over examples into a sum, which is easier to differentiate, can be averaged over mini-batches, and avoids numerical underflow (a product of thousands of probabilities rounds to 0 in floating point)." },
      { t: "qa", level: "beginner", q: "What is the MLE of a coin's bias after k heads in n flips?", a: "$\\hat p = k/n$. From $\\ell(p) = k\\log p + (n-k)\\log(1-p)$, setting $k/p - (n-k)/(1-p) = 0$ gives $p = k/n$; the second derivative is negative so it is a maximum." },
      { t: "qa", level: "intermediate", q: "Show that minimising MSE is maximum likelihood.", a: "Assume $y = f_\\theta(x) + \\varepsilon$ with $\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)$. Then $-\\log p(y \\mid x) = (y - f_\\theta(x))^2/(2\\sigma^2) + \\frac{1}{2}\\log(2\\pi\\sigma^2)$. Summing over i.i.d. examples, the second term and the factor $1/(2\\sigma^2)$ are constants with respect to $\\theta$, so the maximiser of the likelihood is the minimiser of $\\sum_i (y_i - f_\\theta(x_i))^2$." },
      { t: "qa", level: "intermediate", q: "Why is the MLE of the Gaussian variance biased, and by how much?", a: "It measures spread around $\\bar x$, which is fitted to the same data and is closer to the points than the true mean. $\\mathbb{E}[\\frac{1}{n}\\sum(x_i - \\bar x)^2] = \\frac{n-1}{n}\\sigma^2$. Dividing by $n - 1$ instead gives an unbiased estimate. The bias vanishes as $n \\to \\infty$." },
      { t: "qa", level: "intermediate", q: "Explain L2 regularisation as MAP estimation.", a: "Put an independent Gaussian prior $\\mathcal{N}(0, \\tau^2)$ on each weight. The log-posterior is the log-likelihood plus $-\\|\\mathbf w\\|^2/(2\\tau^2)$ + const, so maximising it means minimising NLL $+ \\frac{1}{2\\tau^2}\\|\\mathbf w\\|^2$ — an L2 penalty whose strength is inversely proportional to the prior variance (for Gaussian regression, $\\lambda = \\sigma^2/\\tau^2$). A Laplace prior gives L1 instead." },
      { t: "qa", level: "intermediate", q: "What is a conjugate prior? Give the Beta–Bernoulli example.", a: "A prior is conjugate to a likelihood if the posterior is in the same family. Beta($\\alpha, \\beta$) prior × Bernoulli likelihood with $k$ successes in $n$ gives Beta($k + \\alpha$, $n - k + \\beta$). The prior's parameters act as pseudo-counts; the MAP is $(k + \\alpha - 1)/(n + \\alpha + \\beta - 2)$ and the posterior mean $(k + \\alpha)/(n + \\alpha + \\beta)$." },
      { t: "qa", level: "advanced", q: "How does maximum likelihood relate to KL divergence?", a: "Average log-likelihood $\\frac{1}{n}\\sum_i\\log p_\\theta(x_i)$ estimates $\\mathbb{E}_{p_{\\text{data}}}[\\log p_\\theta(x)]$. Since $\\mathrm{KL}(p_{\\text{data}} \\| p_\\theta) = \\mathbb{E}_{p_{\\text{data}}}[\\log p_{\\text{data}}] - \\mathbb{E}_{p_{\\text{data}}}[\\log p_\\theta]$ and the first term does not depend on $\\theta$, maximising expected log-likelihood is minimising this forward KL. Forward KL is 'mass-covering': the model is heavily penalised for assigning near-zero probability anywhere the data has mass." },
      { t: "qa", level: "advanced", q: "When do MLE, MAP and full Bayesian inference give noticeably different predictions, and why do they agree for large n?", a: "They differ when data are scarce relative to model flexibility: MLE trusts the sample (e.g. $\\hat p = 1$ after three heads), MAP is shrunk toward the prior mode, and the Bayesian predictive averages over all plausible parameters, giving hedged probabilities. As $n$ grows the log-likelihood grows linearly in $n$ while the log-prior stays fixed, so the prior's influence fades; the posterior concentrates around the MLE with width $O(1/\\sqrt{n})$ (Bernstein–von Mises), so all three predictions coincide." },
      { t: "qa", level: "advanced", q: "Why is MAP not invariant to reparameterisation, while MLE is?", a: "The likelihood is a function of the parameter value, so the best value maps to the best value under any one-to-one transform $\\phi = g(\\theta)$. A prior is a density: changing variables multiplies it by the Jacobian $|d\\theta/d\\phi|$, which moves its mode. Hence the MAP for $\\sigma$ and the MAP for $\\log\\sigma$ (transformed back) generally differ, whereas $\\hat\\phi_{\\text{MLE}} = g(\\hat\\theta_{\\text{MLE}})$." },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A coin shows 6 heads in 8 flips. What is the maximum-likelihood estimate of $p$?", answer: 0.75, tolerance: 0.001, explain: "$\\hat p = k/n = 6/8 = 0.75$." },
        { kind: "numeric", q: "Same data (6 heads in 8) with a Beta(2, 2) prior. What is the MAP estimate?", answer: 0.7, tolerance: 0.001, explain: "$(k + \\alpha - 1)/(n + \\alpha + \\beta - 2) = 7/10 = 0.7$." },
        { kind: "numeric", q: "Data $(1, 3, 5)$ from a Gaussian. What is the maximum-likelihood estimate of the variance? (4 decimals)", answer: 2.6667, tolerance: 0.001, explain: "$\\hat\\mu = 3$; squared deviations 4, 0, 4; MLE divides by $n = 3$: $8/3 = 2.6667$. (The unbiased estimate is $8/2 = 4$.)" },
        { kind: "numeric", q: "The true variance is 8 and $n = 4$. What is the expected value of the MLE variance estimate?", answer: 6, tolerance: 0.001, explain: "$\\mathbb{E}[\\hat\\sigma^2] = \\frac{n-1}{n}\\sigma^2 = \\frac{3}{4}\\times 8 = 6$." },
        { q: "Training a regression network with MSE corresponds to maximum likelihood under which noise model?", options: ["Laplace noise", "Gaussian noise with fixed variance", "Bernoulli noise", "Uniform noise"], answer: 1, explain: "The Gaussian NLL is $(y - f)^2/(2\\sigma^2)$ plus a constant. Laplace noise would give MAE." },
        { q: "Which prior on the weights corresponds to L1 regularisation?", options: ["Gaussian", "Uniform", "Laplace", "Beta"], answer: 2, explain: "$-\\log$ of a Laplace density is $|w|/b$ + const — an L1 penalty, which encourages exact zeros." },
        { q: "Why do implementations sum log-probabilities rather than multiply probabilities?", options: ["Logs make the model more accurate", "Products of many probabilities underflow to 0, and sums are easier to differentiate and average", "Logs remove the need for a prior", "Probabilities can be negative"], answer: 1, explain: "The maximiser is unchanged (log is monotonic) while numerical stability and convenience improve." },
        { q: "As the amount of data grows, the MAP estimate…", options: ["approaches the prior mean", "approaches the MLE", "oscillates", "becomes undefined"], answer: 1, explain: "The log-likelihood grows linearly with $n$ while the log-prior stays fixed, so the prior's relative influence vanishes." },
      ] },
      { t: "exercise", title: "Beta prior with lopsided data (numerical)", prompt: "You observe 9 heads in 10 flips and use a Beta(3, 3) prior. Compute (a) the MLE, (b) the posterior distribution, (c) the MAP estimate, (d) the posterior mean, i.e. the Bayesian probability that the next flip is heads.", hint: "Posterior is Beta(k + α, n − k + β).", solution: [
        { t: "steps", items: [
          { title: "MLE", tex: String.raw`\hat p_{\text{MLE}} = 9/10 = 0.9` },
          { title: "Posterior", tex: String.raw`\text{Beta}(9 + 3,\; 1 + 3) = \text{Beta}(12, 4)` },
          { title: "MAP (mode)", tex: String.raw`\hat p_{\text{MAP}} = \frac{12 - 1}{12 + 4 - 2} = \frac{11}{14} = 0.7857` },
          { title: "Posterior mean / predictive", tex: String.raw`P(\text{heads next} \mid \mathcal{D}) = \frac{12}{12 + 4} = 0.75` },
        ] },
        { t: "p", text: "The prior Beta(3, 3) is worth two imaginary heads and two imaginary tails for the MAP, pulling the estimate from 0.9 toward 0.5." },
      ] },
      { t: "exercise", title: "MLE for an exponential rate (coding)", prompt: "Waiting times $x_i \\ge 0$ follow an exponential distribution $p(x \\mid \\lambda) = \\lambda e^{-\\lambda x}$. Derive the MLE of $\\lambda$, then verify it numerically by gradient ascent on $\\log\\lambda$ for 500 samples with true $\\lambda = 2.5$.", hint: "$\\ell(\\lambda) = n\\log\\lambda - \\lambda\\sum_i x_i$. For the ascent, use $\\partial\\ell/\\partial\\log\\lambda = \\lambda\\,\\partial\\ell/\\partial\\lambda$.", solution: [
        { t: "math", tex: String.raw`\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_i x_i = 0 \;\Longrightarrow\; \hat\lambda = \frac{n}{\sum_i x_i} = \frac{1}{\bar x}` },
        { t: "code", lang: "python", title: "exponential_mle.py", code: EXP_CODE, output: EXP_OUT },
        { t: "p", text: "Gradient ascent reaches $1/\\bar x = 2.28047$ in about 20 steps. It differs from the true 2.5 because of sampling noise: the estimate's standard error is roughly $\\lambda/\\sqrt{n} = 2.5/\\sqrt{500} = 0.11$, so this sample is about two standard errors low. MLE recovers the best fit to the sample, not the truth itself." },
      ] },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      { t: "list", items: [
        "Likelihood $\\mathcal{L}(\\theta) = p(\\mathcal{D} \\mid \\theta)$: the data are fixed, the parameter varies. It is not a distribution over $\\theta$.",
        "i.i.d. data: $\\ell(\\theta) = \\sum_i \\log p(x_i \\mid \\theta)$. Logs give the same maximiser, sums instead of products, and no underflow.",
        "Bernoulli: $\\hat p = k/n$. Gaussian: $\\hat\\mu = \\bar x$, $\\hat\\sigma^2 = \\frac{1}{n}\\sum(x_i - \\bar x)^2$ with $\\mathbb{E}[\\hat\\sigma^2] = \\frac{n-1}{n}\\sigma^2$.",
        "Loss = negative log-likelihood: Gaussian ⇒ MSE, Laplace ⇒ MAE, Bernoulli ⇒ binary cross-entropy, categorical ⇒ cross-entropy. MLE minimises $\\mathrm{KL}(p_{\\text{data}} \\| p_\\theta)$.",
        "Numerical MLE: gradient ascent on $\\ell$ = gradient descent on the NLL; optimise unconstrained versions of constrained parameters ($\\log\\sigma$, logits).",
        "MAP: $\\arg\\max[\\log p(\\mathcal{D} \\mid \\theta) + \\log p(\\theta)]$; posterior ∝ likelihood × prior.",
        "Gaussian prior ⇒ L2 / weight decay with $\\lambda = \\sigma^2/\\tau^2$; Laplace prior ⇒ L1 (sparsity).",
        "Beta–Bernoulli: posterior Beta($k+\\alpha, n-k+\\beta$); MAP $(k+\\alpha-1)/(n+\\alpha+\\beta-2)$; posterior mean $(k+\\alpha)/(n+\\alpha+\\beta)$; priors act as pseudo-counts.",
        "Small data: MLE overfits, MAP shrinks, full Bayes hedges. Large data: all three agree.",
      ] },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      { t: "list", items: [
        "[Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) — the information-theoretic reading of the negative log-likelihood, and why cross-entropy measures the mismatch between distributions.",
        "[Classification Losses](/learn/m05/classification-losses) — binary and categorical cross-entropy in practice: logits, numerical stability, label smoothing and focal loss.",
        "[Regularization](/learn/m07/regularization) — L1, L2, dropout and early stopping; the priors of this lesson as practical tools against overfitting.",
        "[Optimization: Objectives, Convexity and Minima](/learn/m02/optimization-math) — what makes a log-likelihood easy or hard to maximise.",
        "[Variational Autoencoders](/learn/m14/vae) — maximum likelihood for models with latent variables, via the ELBO.",
      ] },
    ],
  },
};

export default lesson;
