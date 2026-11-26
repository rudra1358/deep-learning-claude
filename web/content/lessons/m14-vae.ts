import type { Lesson } from "@/lib/types";

const ELBO_NUMPY = String.raw`import numpy as np

# A latent-variable model small enough that EVERYTHING is computable exactly:
#   prior       z ~ N(0, 1)
#   likelihood  x | z ~ N(w z + b, s^2)
# so the marginal is x ~ N(b, w^2 + s^2) and the true posterior p(z | x) is Gaussian too.
w, b, s = 2.0, 0.5, 0.5
x = 2.0                                           # one observed data point


def log_normal(v, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - (v - mean) ** 2 / (2 * var)


def kl_gauss(m1, v1, m2, v2):
    """KL( N(m1, v1) || N(m2, v2) ) for scalars (v = variance)."""
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)


log_px = log_normal(x, b, w ** 2 + s ** 2)         # exact log marginal likelihood
post_var = 1 / (1 + w ** 2 / s ** 2)               # exact posterior p(z|x) = N(post_mean, post_var)
post_mean = post_var * w * (x - b) / s ** 2
print(f"exact log p(x) = {log_px:.6f}   posterior N({post_mean:.4f}, {post_var:.4f})")


def elbo(mu, var):
    """Closed-form ELBO = E_q[log p(x|z)] - KL(q || p(z)) for q = N(mu, var)."""
    exp_loglik = -0.5 * np.log(2 * np.pi * s ** 2) - ((x - b - w * mu) ** 2 + w ** 2 * var) / (2 * s ** 2)
    return exp_loglik - kl_gauss(mu, var, 0.0, 1.0)


# ---- 1. ELBO <= log p(x), and the gap is exactly KL(q || p(z|x)) ----
print("\n   q = N(mu, var)        ELBO     log p(x) - ELBO   KL(q || posterior)")
for mu, var in [(0.0, 1.0), (1.0, 0.1), (0.6, 0.06), (post_mean, post_var)]:
    e = elbo(mu, var)
    print(f"   N({mu:.4f}, {var:.4f})   {e:9.4f}   {log_px - e:12.6f}   {kl_gauss(mu, var, post_mean, post_var):14.6f}")

# ---- 2. Monte Carlo ELBO with reparameterised samples z = mu + sigma * eps ----
rng = np.random.default_rng(0)
mu, var = 1.0, 0.1
eps = rng.standard_normal(100_000)
z = mu + np.sqrt(var) * eps
log_w = log_normal(x, w * z + b, s ** 2) + log_normal(z, 0, 1) - log_normal(z, mu, var)   # log p(x,z) - log q(z)
print(f"\nMonte Carlo ELBO (1e5 samples) {log_w.mean():.4f}  vs closed form {elbo(mu, var):.4f}")

# ---- 3. Jensen's inequality: averaging the weights INSIDE the log tightens the bound (IWAE) ----
for K in (1, 10, 100, 1000):
    lw = log_w[: (len(log_w) // K) * K].reshape(-1, K)
    bound = np.mean(np.log(np.mean(np.exp(lw), axis=1)))
    print(f"   K = {K:4d}: E[log (1/K) sum_k p(x,z_k)/q(z_k)] = {bound:.4f}")

# ---- 4. two gradient estimators for d/dmu E_q[log p(x|z)] ----
true_grad = w * (x - b - w * mu) / s ** 2
f = log_normal(x, w * z + b, s ** 2)                       # f(z) = log p(x|z)
score = f * (z - mu) / var                                 # REINFORCE / score function: f(z) d/dmu log q(z)
score_cv = (f - f.mean()) * (z - mu) / var                 # same, with a baseline (control variate)
pathwise = w * (x - b - w * z) / s ** 2                    # reparameterisation: d f(mu + sigma*eps) / d mu
print(f"\ntrue gradient {true_grad:.4f}")
for name, g in [("score function", score), ("score + baseline", score_cv), ("pathwise (reparam.)", pathwise)]:
    print(f"   {name:<20} mean {g.mean():8.4f}   per-sample std {g.std():8.4f}")

# ---- 5. maximise the ELBO by stochastic gradient ascent with the reparameterisation trick ----
m, log_v = 0.0, 0.0                                         # start at q = N(0, 1)
lr = 0.01
for step in range(2001):
    e = rng.standard_normal(16)
    sd = np.exp(0.5 * log_v)
    zz = m + sd * e
    dlogp_dz = w * (x - b - w * zz) / s ** 2 - zz           # d/dz [log p(x|z) + log p(z)]
    g_m = dlogp_dz.mean()                                   # dz/dm = 1
    g_logv = (dlogp_dz * e * 0.5 * sd).mean() + 0.5         # dz/dlog_v = 0.5*sd*eps; + d/dlog_v of entropy
    m, log_v = m + lr * g_m, log_v + lr * g_logv
    if step % 500 == 0:
        print(f"step {step:4d}  q = N({m:.4f}, {np.exp(log_v):.4f})  ELBO {elbo(m, np.exp(log_v)):.4f}")`;
const ELBO_NUMPY_OUT = String.raw`exact log p(x) = -1.907104   posterior N(0.7059, 0.0588)

   q = N(mu, var)        ELBO     log p(x) - ELBO   KL(q || posterior)
   N(0.0000, 1.0000)    -12.7258      10.818687        10.818687
   N(1.0000, 0.1000)     -2.7271       0.819980         0.819980
   N(0.6000, 0.0600)     -2.0025       0.095393         0.095393
   N(0.7059, 0.0588)     -1.9071       0.000000         0.000000

Monte Carlo ELBO (1e5 samples) -2.7257  vs closed form -2.7271
   K =    1: E[log (1/K) sum_k p(x,z_k)/q(z_k)] = -2.7257
   K =   10: E[log (1/K) sum_k p(x,z_k)/q(z_k)] = -1.9638
   K =  100: E[log (1/K) sum_k p(x,z_k)/q(z_k)] = -1.9130
   K = 1000: E[log (1/K) sum_k p(x,z_k)/q(z_k)] = -1.9078

true gradient -4.0000
   score function       mean  -3.9797   per-sample std  13.0618
   score + baseline     mean  -3.9841   per-sample std   9.9082
   pathwise (reparam.)  mean  -3.9954   per-sample std   5.0603
step    0  q = N(0.1890, 0.9082)  ELBO -10.0294
step  500  q = N(0.7104, 0.0637)  ELBO -1.9089
step 1000  q = N(0.6926, 0.0582)  ELBO -1.9086
step 1500  q = N(0.7152, 0.0573)  ELBO -1.9080
step 2000  q = N(0.6986, 0.0593)  ELBO -1.9076`;
const VAE_TORCH = String.raw`import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)
T_tr = torch.tensor(X_tr, dtype=torch.float32)
T_te = torch.tensor(X_te, dtype=torch.float32)
LATENT = 2


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, LATENT)
        self.logvar = nn.Linear(32, LATENT)            # predict log sigma^2: any real number is valid
        self.dec = nn.Sequential(nn.Linear(LATENT, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation trick
        return self.dec(z), mu, logvar                              # decoder returns LOGITS


def loss_terms(model, x):
    logits, mu, logvar = model(x)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)   # -log p(x|z)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=1)                        # KL(q(z|x) || N(0,I))
    return recon, kl


torch.manual_seed(0)
model = VAE()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
gen = torch.Generator().manual_seed(0)
for epoch in range(1, 201):
    model.train()
    for idx in torch.randperm(len(T_tr), generator=gen).split(64):
        recon, kl = loss_terms(model, T_tr[idx])
        loss = (recon + kl).mean()                      # negative ELBO, averaged over the batch
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch in (1, 10, 50, 100, 200):
        model.eval()
        with torch.no_grad():
            r_tr, k_tr = loss_terms(model, T_tr)
            r_te, k_te = loss_terms(model, T_te)
        print(f"epoch {epoch:3d}  train: recon {r_tr.mean():6.2f} + KL {k_tr.mean():5.2f} = {(r_tr + k_tr).mean():6.2f}"
              f"   test: recon {r_te.mean():6.2f} + KL {k_te.mean():5.2f} = {(r_te + k_te).mean():6.2f}  nats")


# ---- importance-weighted estimate of log p(x) on the test set: tighter than the ELBO ----
@torch.no_grad()
def log_px_estimate(x, K):
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn(K, *mu.shape)                       # (K, N, 2) samples from q(z|x)
    log_q = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_prior = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_lik = -F.binary_cross_entropy_with_logits(model.dec(z), x.expand(K, -1, -1), reduction="none").sum(-1)
    return (torch.logsumexp(log_lik + log_prior - log_q, dim=0) - math.log(K)).mean().item()


torch.manual_seed(1)
for K in (1, 10, 100, 1000):
    print(f"test log p(x) bound with K = {K:4d} importance samples: {log_px_estimate(T_te, K):.2f} nats")`;
const VAE_TORCH_OUT = String.raw`epoch   1  train: recon  32.29 + KL  0.94 =  33.23   test: recon  32.30 + KL  0.95 =  33.24  nats
epoch  10  train: recon  27.12 + KL  0.03 =  27.15   test: recon  27.23 + KL  0.03 =  27.26  nats
epoch  50  train: recon  22.67 + KL  2.05 =  24.72   test: recon  23.10 + KL  2.01 =  25.11  nats
epoch 100  train: recon  21.92 + KL  2.51 =  24.44   test: recon  22.29 + KL  2.46 =  24.75  nats
epoch 200  train: recon  21.36 + KL  2.69 =  24.05   test: recon  21.96 + KL  2.64 =  24.60  nats
test log p(x) bound with K =    1 importance samples: -24.74 nats
test log p(x) bound with K =   10 importance samples: -24.28 nats
test log p(x) bound with K =  100 importance samples: -24.24 nats
test log p(x) bound with K = 1000 importance samples: -24.21 nats`;
const VAE_GEN = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)
T_tr = torch.tensor(X_tr, dtype=torch.float32)
T_te = torch.tensor(X_te, dtype=torch.float32)
LATENT = 2


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, LATENT)
        self.logvar = nn.Linear(32, LATENT)            # predict log sigma^2: any real number is valid
        self.dec = nn.Sequential(nn.Linear(LATENT, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation trick
        return self.dec(z), mu, logvar                              # decoder returns LOGITS


def loss_terms(model, x):
    logits, mu, logvar = model(x)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)   # -log p(x|z)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=1)                        # KL(q(z|x) || N(0,I))
    return recon, kl


torch.manual_seed(0)
model = VAE()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
gen = torch.Generator().manual_seed(0)
for epoch in range(120):                                # same VAE as the previous program, 120 epochs
    for idx in torch.randperm(len(T_tr), generator=gen).split(64):
        recon, kl = loss_terms(model, T_tr[idx])
        loss = (recon + kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
model.eval()

# ---- generation: VAE prior samples vs a deterministic AE with the same architecture ----
torch.manual_seed(0)
ae_enc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, LATENT))
ae_dec = nn.Sequential(nn.Linear(LATENT, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))
opt = torch.optim.Adam([*ae_enc.parameters(), *ae_dec.parameters()], lr=2e-3)
gen = torch.Generator().manual_seed(0)
for epoch in range(120):
    for idx in torch.randperm(len(T_tr), generator=gen).split(64):
        x = T_tr[idx]
        loss = F.binary_cross_entropy_with_logits(ae_dec(ae_enc(x)), x, reduction="sum") / len(x)
        opt.zero_grad()
        loss.backward()
        opt.step()

clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)      # judge: "is this a recognisable digit?"
with torch.no_grad():
    codes_ae = ae_enc(T_tr).numpy()
    codes_vae = model.encode(T_tr)[0].numpy()
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((1000, LATENT))
    # the natural way to "sample" an AE: fit a Gaussian to its codes and decode draws from it
    L = np.linalg.cholesky(np.cov(codes_ae.T))
    z_ae = codes_ae.mean(axis=0) + eps @ L.T
    imgs_ae = torch.sigmoid(ae_dec(torch.tensor(z_ae, dtype=torch.float32))).numpy()
    imgs_vae = torch.sigmoid(model.dec(torch.tensor(eps, dtype=torch.float32))).numpy()


def gap_ratio(codes, zq):
    """distance from each query to its nearest training code, divided by the median code-to-code gap"""
    d_tr = np.sqrt(((codes[:, None] - codes[None]) ** 2).sum(-1)) + np.eye(len(codes)) * 1e9
    d_q = np.sqrt(((zq[:, None] - codes[None]) ** 2).sum(-1))
    return np.median(d_q.min(axis=1)) / np.median(d_tr.min(axis=1))


for name, codes, zq, imgs in [("AE  (Gaussian fit to codes)", codes_ae, z_ae, imgs_ae), ("VAE (prior N(0, I))", codes_vae, eps, imgs_vae)]:
    conf = clf.predict_proba(imgs).max(axis=1)
    counts = np.bincount(clf.predict(imgs), minlength=10)
    print(f"{name}: gap ratio {gap_ratio(codes, zq):.2f}, classifier confidence {conf.mean():.3f}, "
          f"share > 0.9 {np.mean(conf > 0.9):.3f}, predicted classes {counts.tolist()}")`;
const VAE_GEN_OUT = String.raw`AE  (Gaussian fit to codes): gap ratio 2.03, classifier confidence 0.802, share > 0.9 0.428, predicted classes [76, 104, 95, 73, 116, 82, 78, 113, 160, 103]
VAE (prior N(0, I)): gap ratio 1.76, classifier confidence 0.734, share > 0.9 0.291, predicted classes [94, 80, 89, 109, 123, 98, 91, 112, 102, 102]`;

const VAE_KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import load_digits

X = load_digits().data.astype("float32") / 16.0
LATENT = 2


class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.body = tf.keras.Sequential([tf.keras.layers.Dense(64, activation="relu"),
                                         tf.keras.layers.Dense(32, activation="relu")])
        self.mu = tf.keras.layers.Dense(LATENT)
        self.logvar = tf.keras.layers.Dense(LATENT)
        self.dec = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="relu"),
                                        tf.keras.layers.Dense(64, activation="relu"),
                                        tf.keras.layers.Dense(64)])            # logits

    def train_step(self, x):
        with tf.GradientTape() as tape:
            h = self.body(x)
            mu, logvar = self.mu(h), self.logvar(h)
            z = mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(mu))   # reparameterisation
            logits = self.dec(z)
            recon = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), axis=1)
            kl = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1.0 - logvar, axis=1)
            loss = tf.reduce_mean(recon + kl)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "recon": tf.reduce_mean(recon), "kl": tf.reduce_mean(kl)}


vae = VAE()
vae.compile(optimizer=tf.keras.optimizers.Adam(2e-3))
vae.fit(X, epochs=100, batch_size=64, verbose=2)
samples = tf.sigmoid(vae.dec(tf.random.normal((16, LATENT))))    # 16 new digits from the prior`;

const lesson: Lesson = {
  id: "m14/vae",
  title: "Variational Autoencoders",
  summary:
    "A probabilistic autoencoder that is a real generative model: latent-variable models, the intractable marginal likelihood, variational inference and the ELBO (derived two ways), the reparameterisation trick and why it beats the score-function gradient, the closed-form KL, β-VAE, posterior collapse, blurriness and conditional VAEs — derived, computed and trained on real digits.",
  minutes: 90,
  level: "advanced",
  prerequisites: ["m14/autoencoders", "m02/information-theory", "m02/probability", "m02/mle-map"],
  next: ["m14/diffusion", "m14/gans", "m15/self-supervised"],
  terms: ["variational-autoencoder", "latent-variable-model", "elbo", "variational-inference", "reparameterization-trick", "kl-divergence", "posterior-collapse", "beta-vae", "amortized-inference"],
  sections: {
    intro: [
      {
        t: "p",
        text: "A plain [autoencoder](/learn/m14/autoencoders) compresses and reconstructs, but it cannot *generate*: its codes follow no known distribution, and its decoder is untrained in the holes between them. The **variational autoencoder** (VAE) — Kingma and Welling, \"Auto-Encoding Variational Bayes\" (2013), and independently Rezende, Mohamed and Wierstra (2014, \"Stochastic Backpropagation and Approximate Inference in Deep Generative Models\") — fixes this by starting from a probabilistic story of how data is made and deriving the training objective from it.",
      },
      {
        t: "p",
        text: String.raw`The story: draw a hidden **latent variable** $z$ from a simple **prior** $p(z) = \mathcal N(0, I)$, then draw the data $x$ from a **decoder** distribution $p_\theta(x\mid z)$ whose parameters a network computes from $z$. Training should maximise the data's probability, $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$ — but this integral cannot be computed.`,
      },
      {
        t: "p",
        text: String.raw`So the VAE adds an **encoder** $q_\phi(z\mid x)$ that guesses which $z$ produced $x$ (approximating the posterior), and maximises a computable lower bound on $\log p_\theta(x)$, the **evidence lower bound (ELBO)**. Two ideas make it scale: **amortised variational inference** (one encoder network predicts the approximate posterior for every $x$) and the **reparameterisation trick** (sampling written as a differentiable function, so backprop trains the encoder through the sample).`,
      },
      {
        t: "p",
        text: String.raw`The result is an autoencoder with two changes: the encoder outputs a mean and variance instead of a point, and the loss gains a KL term pulling every encoding towards the prior. To generate, sample $z \sim \mathcal N(0, I)$ and decode. VAEs serve generation and representation learning, the compression stage (in KL-regularised form) of latent diffusion models like Stable Diffusion, anomaly detection with a likelihood bound, and science such as molecule design.`,
      },
      {
        t: "callout",
        kind: "key",
        text: String.raw`VAE loss per example $= \underbrace{-\mathbb E_{q_\phi(z|x)}[\log p_\theta(x\mid z)]}_{\text{reconstruction}} + \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regulariser}} = -\mathrm{ELBO} \ \ge\ -\log p_\theta(x)$.`,
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Autoencoders](/learn/m14/autoencoders) — encoder, decoder, reconstruction losses as negative log-likelihoods, and the holes in a plain AE's latent space.",
          "[Information theory](/learn/m02/information-theory) — KL divergence, why it is never negative, and the closed-form KL between Gaussians (derived there; we only use the result).",
          "[Probability](/learn/m02/probability) — joint, marginal and conditional densities, Bayes' rule, expectations, the Gaussian density.",
          "[MLE and MAP](/learn/m02/mle-map) — training as maximising the log-likelihood of the data.",
          "[Backpropagation](/learn/m04/backpropagation) — the reparameterisation trick exists so backprop can pass through a sampling step.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Addresses as neighbourhoods, not points",
        text: String.raw`A plain autoencoder gives every image an exact street address in latent space; the decoder only visits those, so the empty lots between stay unmapped. A VAE gives each image a *neighbourhood* — "around here, give or take this much" — and in training sends the decoder to a random house in it. Neighbourhoods overlap, so the decoder learns to draw something sensible everywhere. A city rule (the KL term) keeps all neighbourhoods inside one standard district, $\mathcal N(0, I)$, and forbids shrinking them to points. Afterwards any address in the district decodes to something meaningful.`,
      },
      {
        t: "p",
        text: "The two loss terms pull opposite ways. Reconstruction wants each neighbourhood tiny and far from the others (precise codes, sharp reconstructions). KL wants every neighbourhood to look like the whole district — centred at 0, spread 1 — which would make all images look alike to the decoder. Training settles on neighbourhoods as small as reconstruction needs and as close to the prior as possible. The β-VAE dial sets this trade-off; push it too far towards KL and the code becomes useless (**posterior collapse**).",
      },
    ],
    mechanism: [
      {
        t: "steps",
        items: [
          { title: "Encode to a distribution", text: String.raw`The encoder maps $x$ to two vectors of length $d$ (the latent size): a mean $\mu_\phi(x)$ and a log-variance $\log\sigma^2_\phi(x)$. Together they define $q_\phi(z\mid x) = \mathcal N(\mu, \operatorname{diag}\sigma^2)$. Predicting $\log\sigma^2$ lets the network output any real number while $\sigma = \exp(\tfrac12\log\sigma^2)$ stays positive.` },
          { title: "Sample noise", text: String.raw`Draw $\epsilon \sim \mathcal N(0, I)$, which does not depend on any parameter.` },
          { title: "Reparameterise", text: String.raw`Build the latent sample as a fixed, differentiable function of $\mu$, $\sigma$ and $\epsilon$.`, tex: String.raw`z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon` },
          { title: "Decode", text: String.raw`The decoder maps $z$ to the parameters of $p_\theta(x\mid z)$: Bernoulli probabilities (sigmoid of logits) for pixels in [0, 1], or Gaussian means for real-valued data.` },
          { title: "Reconstruction term", text: String.raw`$-\log p_\theta(x\mid z)$ for this one sample: summed binary cross-entropy (or summed squared error / $2\sigma_x^2$ for a Gaussian decoder). One sample per example is enough; over a mini-batch it is an unbiased estimate of the expectation.` },
          { title: "KL term", text: String.raw`A closed form for a diagonal Gaussian against $\mathcal N(0, I)$ — no sampling needed.`, tex: String.raw`\mathrm{KL} = \tfrac12\sum_{j=1}^{d}\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big)` },
          { title: "Backpropagate the sum", text: String.raw`Loss = reconstruction + KL (= −ELBO), averaged over the batch. Gradients reach the decoder directly, and the encoder both through the KL term and through $z$ ($\partial z/\partial\mu = 1$, $\partial z/\partial\sigma = \epsilon$).` },
          { title: "Generate", text: String.raw`After training, drop the encoder: sample $z \sim \mathcal N(0, I)$ and decode. We usually show the decoder's mean (e.g. the Bernoulli probabilities) rather than a sample from $p_\theta(x\mid z)$.` },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        text: "Keep the two terms on the same scale: sum both over dimensions (pixels and latents) per example, then average over the batch. A pixel-*mean* reconstruction with a summed KL silently multiplies the KL weight by the pixel count — a hidden β of 64 for 8×8 digits, far past where our posterior collapses.",
      },
    ],
    math: [
      { t: "h", text: "1. Latent-variable models and the intractable marginal" },
      {
        t: "p",
        text: String.raw`The generative model is $p_\theta(x, z) = p_\theta(x\mid z)\,p(z)$, with $x \in \mathbb R^D$ the data, $z \in \mathbb R^d$ the latent, $p(z) = \mathcal N(0, I)$ and $\theta$ the decoder's weights. Maximum likelihood needs the **marginal likelihood** (the *evidence*): the probability of $x$ averaged over every possible $z$:`,
      },
      { t: "math", tex: String.raw`p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz` },
      {
        t: "p",
        text: String.raw`With a neural decoder this integral has no closed form. Plain Monte Carlo ($\frac1K\sum_k p_\theta(x\mid z_k)$, $z_k \sim p(z)$) is hopeless: almost every prior sample decodes to an image unlike $x$, so rare lucky samples dominate. The **posterior** $p_\theta(z\mid x) = p_\theta(x\mid z)p(z)/p_\theta(x)$ — which $z$ could have produced $x$ — is intractable for the same reason: its normaliser is $p_\theta(x)$.`,
      },
      { t: "h", text: "2. Variational inference and the ELBO — derivation via Jensen" },
      {
        t: "p",
        text: String.raw`**Variational inference** replaces the intractable posterior with the best member of a manageable family $q_\phi(z\mid x)$. Multiply and divide by $q$ inside the integral, then apply Jensen's inequality ($\log\mathbb E[Y] \ge \mathbb E[\log Y]$, because log is concave):`,
      },
      { t: "math", tex: String.raw`\log p_\theta(x) = \log \mathbb E_{q_\phi(z|x)}\!\left[\frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right] \ \ge\ \mathbb E_{q_\phi(z|x)}\!\left[\log\frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right] \equiv \mathrm{ELBO}(\theta, \phi; x)` },
      { t: "h", text: "3. The same bound via the KL gap" },
      {
        t: "p",
        text: String.raw`Put $p_\theta(x, z) = p_\theta(z\mid x)\,p_\theta(x)$ inside the ELBO. Since $\log p_\theta(x)$ does not depend on $z$, it comes out of the expectation:`,
      },
      { t: "math", tex: String.raw`\mathrm{ELBO} = \mathbb E_q\!\left[\log\frac{p_\theta(z\mid x)\,p_\theta(x)}{q_\phi(z\mid x)}\right] = \log p_\theta(x) - \mathrm{KL}\big(q_\phi(z\mid x)\,\big\|\,p_\theta(z\mid x)\big)` },
      {
        t: "p",
        text: String.raw`So the **gap** between evidence and bound is exactly the KL from the approximate to the true posterior. Since KL ≥ 0, this proves the bound again, and shows what training does: maximising over $\phi$ makes $q_\phi$ a better posterior approximation; maximising over $\theta$ raises (a lower bound of) the data likelihood. The bound is tight exactly when $q_\phi(z\mid x) = p_\theta(z\mid x)$.`,
      },
      { t: "h", text: "4. The form used for training" },
      {
        t: "p",
        text: String.raw`Write $\log p_\theta(x, z) = \log p_\theta(x\mid z) + \log p(z)$ and split the expectation:`,
      },
      { t: "math", tex: String.raw`\mathrm{ELBO} = \underbrace{\mathbb E_{q_\phi(z|x)}\big[\log p_\theta(x\mid z)\big]}_{\text{expected reconstruction log-likelihood}} - \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\big\|\,p(z)\big)}_{\text{distance of the encoding from the prior}}` },
      {
        t: "p",
        text: String.raw`**Checked exactly** in the From-Scratch program on a fully computable model: $z \sim \mathcal N(0,1)$, $x\mid z \sim \mathcal N(2z + 0.5,\ 0.5^2)$, observed $x = 2$. Then $\log p(x) = -1.907104$ and the true posterior is $\mathcal N(0.7059, 0.0588)$. For $q = \mathcal N(1, 0.1)$ the ELBO is $-2.7271$, and the gap $0.819980$ equals $\mathrm{KL}(q\,\|\,p(z\mid x))$ to six decimals; for $q$ equal to the posterior the gap is $0$. Stochastic gradient ascent on the ELBO with reparameterised samples, started from $q = \mathcal N(0, 1)$, converges to $q \approx \mathcal N(0.70, 0.059)$ — the true posterior.`,
      },
      { t: "h", text: "5. The closed-form KL to a standard normal" },
      {
        t: "p",
        text: String.raw`For $q = \mathcal N(\mu, \operatorname{diag}\sigma^2)$ and $p = \mathcal N(0, I)$, both split into independent dimensions, so the KL is a sum of the one-dimensional Gaussian KL derived in [Information Theory](/learn/m02/information-theory):`,
      },
      { t: "math", tex: String.raw`\mathrm{KL}\big(\mathcal N(\mu, \operatorname{diag}\sigma^2)\,\|\,\mathcal N(0, I)\big) = \frac12\sum_{j=1}^{d}\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big)`, caption: String.raw`Each term is ≥ 0, and 0 only at $\mu_j = 0$, $\sigma_j = 1$. The $\mu^2$ part pulls codes towards the origin; $\sigma^2 - 1 - \log\sigma^2$ punishes posteriors that are too narrow (→ ∞ as $\sigma\to 0$) or too wide.` },
      { t: "h", text: "6. The reparameterisation trick and why the naive gradient fails" },
      {
        t: "p",
        text: String.raw`We need $\nabla_\phi\,\mathbb E_{q_\phi(z)}[f(z)]$ with $f(z) = \log p_\theta(x\mid z)$. The sampling distribution depends on $\phi$, so we cannot just sample $z$ and differentiate $f(z)$: a sample has no derivative with respect to $\phi$, and autograd sees sampling as a dead end. Two unbiased estimators exist:`,
      },
      { t: "math", tex: String.raw`\underbrace{\nabla_\phi \mathbb E_{q_\phi}[f(z)] = \mathbb E_{q_\phi}\big[f(z)\,\nabla_\phi\log q_\phi(z)\big]}_{\text{score function / REINFORCE}} \qquad \underbrace{\nabla_\phi \mathbb E_{\epsilon\sim\mathcal N(0,I)}\big[f(\mu_\phi + \sigma_\phi\odot\epsilon)\big] = \mathbb E_{\epsilon}\big[\nabla_\phi f(\mu_\phi + \sigma_\phi\odot\epsilon)\big]}_{\text{pathwise / reparameterisation}}` },
      {
        t: "p",
        text: String.raw`The **score-function** estimator (from $\nabla q = q\nabla\log q$) works for any $f$, even non-differentiable ones, but uses only the *value* $f(z)$ times a zero-mean random factor, so its variance is large and grows with the size of $f$ and with dimension. The **pathwise** estimator moves the randomness into a parameter-free $\epsilon$, so the expectation is over a fixed distribution and the gradient can go *inside*; it uses the *derivative* of $f$, which carries far more information per sample.`,
      },
      {
        t: "p",
        text: String.raw`**In numbers** (toy model, $q = \mathcal N(1, 0.1)$, true $\partial/\partial\mu = -4$): per-sample standard deviation 13.06 for the score function, 9.91 with a mean baseline subtracted, 5.06 for the pathwise estimator — 6.7 times less variance than plain REINFORCE. All are unbiased (means −3.98, −3.98, −4.00 from $10^5$ samples).`,
      },
      { t: "h", text: "7. Worked example on the trained digit VAE" },
      { t: "p", text: String.raw`Test image #0 (a 7) through the simulator's model, with the stored noise $\epsilon$ (all numbers from the PyTorch verification vector exported with the model):` },
      {
        t: "steps",
        items: [
          { title: "Encoder output", tex: String.raw`\mu = (0.189481,\ -1.688135),\quad \log\sigma^2 = (-3.341775,\ -1.774847),\quad \sigma = (0.188080,\ 0.411715)` },
          { title: "Noise and sample", tex: String.raw`\epsilon = (-0.040429,\ 1.725992),\quad z = \mu + \sigma\odot\epsilon = (0.189481 - 0.007604,\ -1.688135 + 0.710618) = (0.181877,\ -0.977517)` },
          { title: "KL per dimension", tex: String.raw`\tfrac12(0.035903 + 0.035374 - 1 + 3.341775) = 1.206526,\quad \tfrac12(2.849800 + 0.169509 - 1 + 1.774847) = 1.897078` },
          { title: "Loss", tex: String.raw`-\mathrm{ELBO} \approx \underbrace{19.5412}_{\text{BCE}} + \underbrace{3.1036}_{\text{KL}} = 22.6448\ \text{nats}` },
        ],
      },
      {
        t: "p",
        text: String.raw`Over the 360 test images the model averages reconstruction 21.23 + KL 2.94 = 24.17 nats. The KL is the price of a posterior narrower than and shifted from the prior — roughly how much information the code carries about $x$.`,
      },
      { t: "h", text: "8. Tighter bounds and evaluation" },
      {
        t: "p",
        text: String.raw`Averaging importance weights *inside* the log gives the **importance-weighted bound** (Burda, Grosse and Salakhutdinov, 2015, IWAE): $\mathcal L_K = \mathbb E\big[\log\frac1K\sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k\mid x)}\big]$, with $\mathcal L_1 = \mathrm{ELBO} \le \mathcal L_K \le \mathcal L_{K+1} \le \log p_\theta(x)$. It is the standard way to *estimate* a VAE's log-likelihood. In the toy model it climbs $-2.7257 \to -1.9638 \to -1.9130 \to -1.9078$ for $K = 1, 10, 100, 1000$, towards the exact $-1.9071$; for the trained digit VAE, from $-24.74$ to $-24.21$ nats per image.`,
      },
      { t: "h", text: "9. The full training algorithm" },
      {
        t: "steps",
        items: [
          { title: "Initialise", text: String.raw`Encoder $\phi$ (shared body + $\mu$ head + $\log\sigma^2$ head), decoder $\theta$.` },
          { title: "For each mini-batch $\\{x_n\\}_{n=1}^B$", tex: String.raw`(\mu_n, \log\sigma_n^2) = \mathrm{Enc}_\phi(x_n),\quad \epsilon_n\sim\mathcal N(0,I),\quad z_n = \mu_n + \sigma_n\odot\epsilon_n` },
          { title: "Loss", tex: String.raw`\mathcal L(\theta,\phi) = \frac1B\sum_{n=1}^{B}\Big[-\log p_\theta(x_n\mid z_n) + \beta\cdot\tfrac12\sum_j\big(\mu_{nj}^2 + \sigma_{nj}^2 - 1 - \log\sigma_{nj}^2\big)\Big],\quad \beta = 1` },
          { title: "Update", text: String.raw`Backpropagate and take an optimiser step (Adam) on $\theta$ and $\phi$ together. Repeat until the validation −ELBO stops improving.` },
        ],
      },
    ],
    visual: [
      {
        t: "p",
        text: "The simulator runs a **real VAE trained with PyTorch** on 1,437 of scikit-learn's 8×8 digits (encoder 64→64→32→(μ, log σ²), 2-D latent, decoder 2→32→64→64, 12,740 parameters). Every number is computed in your browser and checked against PyTorch.",
      },
      { t: "sim", id: "vae", caption: "Encoder → μ, σ → ε → z = μ + σ⊙ε → decoder → loss, plus prior sampling, interpolation and a β comparison." },
      {
        t: "list",
        items: [
          String.raw`Step to **μ, σ**: the gold ellipse is $q(z\mid x)$. Step to **z** and press **Resample ε** a few times: $z$ jumps around the ellipse and the reconstruction barely changes — the decoder gives similar images across the neighbourhood.`,
          "At the **loss** step, read reconstruction and KL for several digits. Unusual digits tend to have bigger reconstruction terms; codes far from the origin or with tiny σ pay more KL.",
          "Switch the latent view to **decoded map** and compare with the plain autoencoder simulator below: the VAE's map is smooth and digit-like across the whole prior disc.",
          String.raw`Press **new samples** in the generation panel: 16 new digits from $z \sim \mathcal N(0, I)$. They are plausible but a bit blurry, and some are ambiguous — a 2-D latent and a pixel-independent likelihood are strong limits.`,
          String.raw`Interpolate between two digits of different classes, and traverse $z_1$ and $z_2$: images change gradually. Read the β table: at β = 4 this model collapses completely (KL = 0, σ = 1).`,
        ],
      },
      { t: "sim", id: "autoencoder", caption: "For comparison: the deterministic autoencoder with the same layer sizes. Its codes spread over an irregular, much larger region, and its decoder extrapolates (blocky, over-saturated strokes) away from them." },
    ],
    scratch: [
      {
        t: "p",
        text: String.raw`Everything about the ELBO can be checked exactly on a 1-D linear-Gaussian latent-variable model, where $p(x)$ and $p(z\mid x)$ have closed forms. The program checks the KL-gap identity, estimates the ELBO with reparameterised samples, shows the importance-weighted bound tightening, compares the two gradient estimators' variance, and fits $q$ by stochastic gradient ascent.`,
      },
      { t: "code", lang: "python", title: "elbo_from_scratch.py", code: ELBO_NUMPY, output: ELBO_NUMPY_OUT },
      {
        t: "list",
        items: [
          String.raw`Part 1 confirms $\log p(x) - \mathrm{ELBO} = \mathrm{KL}(q\,\|\,p(z\mid x))$ to six decimals, for four different $q$.`,
          String.raw`Part 2: the Monte Carlo ELBO uses $\log w = \log p(x\mid z) + \log p(z) - \log q(z)$ — exactly what a VAE computes with one sample per example.`,
          String.raw`Part 3: averaging $w$ *before* the log (IWAE) shrinks Jensen's gap; with $K = 1000$ the bound is within 0.0007 nats of the truth.`,
          "Part 4: all three estimators are unbiased, but the pathwise gradient's per-sample standard deviation is 2.6× smaller than REINFORCE's (6.7× in variance), and the gap grows with dimension.",
          String.raw`Part 5: $\partial\,\mathrm{ELBO}/\partial\log v$ is the pathwise term $\mathbb E[\nabla_z\log p(x, z)\cdot\tfrac12\sigma\epsilon]$ plus $\tfrac12$ from the entropy of $q$, $\tfrac12\log(2\pi e v)$. The final $q$ wobbles around the true posterior because each step uses only 16 samples.`,
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
                text: String.raw`A complete VAE on the digits (same architecture as the simulator's model, Bernoulli decoder with logits, $\beta = 1$). It prints both loss terms on train and test data, then the importance-weighted estimate of the test log-likelihood.`,
              },
              { t: "code", lang: "python", title: "vae_digits.py", code: VAE_TORCH, output: VAE_TORCH_OUT },
              {
                t: "list",
                items: [
                  String.raw`**Temporary posterior collapse**: at epoch 10 the KL is only 0.03 nats — the decoder outputs the average digit and ignores $z$. Using the latent costs KL, so early on the cheapest move is to skip it. By epoch 50 the encoder has found codes worth paying for and the KL rises to ≈ 2.`,
                  "The test −ELBO (24.60) is only a little above the training value (24.05): with 12.7k parameters and a 2-D bottleneck there is little overfitting.",
                  String.raw`The importance-weighted bound tightens from −24.74 ($K = 1$, the ELBO) to −24.21 ($K = 1000$). So the ELBO underestimates this model's log-likelihood by about half a nat per image: $q$ is a good but imperfect posterior.`,
                  "`binary_cross_entropy_with_logits` fuses the sigmoid and the log for numerical stability, so the decoder returns logits and images are `sigmoid(logits)`.",
                ],
              },
              {
                t: "p",
                text: String.raw`**Generation: VAE vs plain AE.** The natural way to sample a plain AE is to fit a Gaussian to its training codes. A logistic-regression classifier trained on real digits judges both. The *gap ratio* is the median distance from a sampled $z$ to the nearest training code, over the median distance between neighbouring training codes.`,
              },
              { t: "code", lang: "python", title: "vae_vs_ae_sampling.py", code: VAE_GEN, output: VAE_GEN_OUT },
              {
                t: "p",
                text: "An honest result: in two dimensions the Gaussian-fitted AE sampler is *not* worse by these measures — the judge is slightly more confident in it (0.802 vs 0.734). The VAE's samples are blurrier and more often ambiguous, but cover the classes more evenly (the AE makes too many 8s).",
              },
              {
                t: "p",
                text: "The VAE's advantages are different: a prior fixed *before* training, not fitted afterwards; a decoder trained on noisy codes, so the latent space is smooth by design; a likelihood bound for evaluation and comparison; and a principled objective that scales to many latent dimensions, where fitting a density to AE codes becomes unreliable. Sharpness is not one of them — blurriness is the VAE's known weakness.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "Subclass `tf.keras.Model` and override `train_step` so the two loss terms are computed and logged explicitly; `fit` then runs the loop." },
              { t: "code", lang: "python", title: "vae_keras.py (not executed in CI)", code: VAE_KERAS, output: "Prints loss (= −ELBO), recon and kl for each of 100 epochs; the last line produces 16 images decoded from prior samples." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Variant", "Change", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["VAE (Kingma & Welling 2013)", "Gaussian $q$, $\\mathcal N(0,I)$ prior, β = 1", "principled; likelihood bound; smooth latent", "blurry samples; posterior collapse", "generation, representation learning"],
          ["β-VAE (Higgins et al. 2017)", "loss = recon + β·KL, β > 1", "pushes towards factorised, sometimes more interpretable latents", "worse reconstructions; disentanglement not guaranteed", "disentanglement research"],
          ["Conditional VAE (Sohn et al. 2015)", "encoder and decoder also get a condition $c$ (e.g. a label)", "controllable generation $p(x\\mid z, c)$", "needs labels/conditions", "class- or attribute-conditioned generation"],
          ["IWAE (Burda et al. 2015)", "train on $\\mathcal L_K$ with $K$ samples", "tighter bound, better likelihoods", "K× compute; the encoder's gradient signal can weaken as K grows", "density estimation"],
          ["Hierarchical VAEs (e.g. NVAE, VDVAE)", "several layers of latents", "much sharper samples", "complex, harder to train", "high-quality image generation"],
          ["VQ-VAE (van den Oord et al. 2017)", "discrete codebook latent, straight-through gradient, separately learned prior", "no posterior collapse; tokens for autoregressive models", "codebook collapse; prior trained separately", "image/audio tokenizers"],
          ["KL-regularised autoencoder of latent diffusion", "VAE-style encoder with a very small KL weight plus perceptual and adversarial losses", "sharp reconstructions, compact latent", "not meant for sampling from the prior", "Stable Diffusion's latent space — see [Diffusion](/learn/m14/diffusion)"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Latent spaces for diffusion** — latent diffusion (Rombach et al., 2022, the basis of Stable Diffusion) trains a KL-regularised autoencoder, then runs diffusion in its latent space; see [Diffusion Models](/learn/m14/diffusion).",
          "**Representation learning and visualisation** — compact, smooth latent features; 2-D VAEs for picturing data sets.",
          "**Anomaly detection with a likelihood** — a low ELBO or importance-weighted likelihood flags unusual inputs (caveat: deep generative models can give high likelihood to some out-of-distribution data).",
          "**Science and design** — VAEs over molecule representations (Gómez-Bombarelli et al., 2018) allow optimisation in a continuous latent space; also used in single-cell genomics (e.g. scVI).",
          "**Discrete tokenizers** — VQ-VAE-style models turn images and audio into token sequences that Transformers can model.",
          "**Data compression** — VAEs with quantised latents and learned entropy models underlie many neural image codecs.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Principled objective**: a lower bound on log-likelihood that can be estimated (IWAE) and compared across models, unlike GANs.",
          "**Fast sampling and encoding**: one decoder pass per sample (vs hundreds of steps for diffusion), and an encoder for free.",
          "**Stable training**: one minimisation, no adversarial game.",
          "**Smooth, structured latent space**: interpolation and traversal change images gradually; a known prior to sample from.",
          String.raw`**Blurry samples**: with a factorised Gaussian or Bernoulli decoder, the best output for an uncertain $z$ is the *average* of all plausible images, and we display the decoder's mean, so detail is averaged away. The ELBO also rewards covering all the data (mass-covering), putting probability on in-between images.`,
          String.raw`**Posterior collapse**: a powerful decoder (or a large β) can model the data without $z$; the KL term then drives $q(z\mid x)$ to the prior.`,
          String.raw`**Approximation gap**: a diagonal Gaussian $q$ cannot represent correlated or multimodal posteriors, so the bound is loose (0.53 nats per image here).`,
          String.raw`**Prior holes**: the aggregate posterior $\frac1N\sum_n q(z\mid x_n)$ need not fill the prior, so some prior samples land in low-density regions and decode to ambiguous images.`,
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          String.raw`**Predicting σ directly with a linear layer** — it can go negative or zero. Predict $\log\sigma^2$ (or use softplus) and compute $\sigma = \exp(\tfrac12\log\sigma^2)$.`,
          "**Sampling with `torch.normal(mu, sigma)` and expecting gradients to reach the encoder.** Write $z = \\mu + \\sigma\\epsilon$ explicitly (or use `rsample()` from `torch.distributions`, not `sample()`).",
          "**Mismatched reductions** — mean over pixels for reconstruction, sum over latents for KL. That is an unintended β (here 1/64, or the reverse, 64) and a common cause of posterior collapse or of ignoring the prior.",
          "**Sign errors**: the ELBO is maximised; the loss is −ELBO = recon + KL. Adding the KL with the wrong sign rewards pushing encodings away from the prior.",
          "**Sampling from the encoder at test time when you want a fixed code** — use μ for features and plots; sample only for training, stochastic reconstruction or likelihood estimation.",
          "**Treating β-VAE latents as guaranteed 'disentangled factors'.** Locatello et al. (2019) showed unsupervised disentanglement is impossible without inductive biases or supervision; results depend heavily on seeds and hyper-parameters.",
          "**Reporting the training loss as the log-likelihood.** −ELBO is only an upper bound on −log p(x); report an importance-weighted estimate with many samples on test data.",
          "**Diagnosing collapse from the total loss alone.** Watch the KL per latent dimension: dimensions with KL ≈ 0 are unused.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "How does a VAE differ from a plain autoencoder?", a: String.raw`The encoder outputs a distribution $q(z\mid x) = \mathcal N(\mu, \sigma^2)$ instead of a point; a sample $z = \mu + \sigma\epsilon$ is decoded; and the loss adds $\mathrm{KL}(q(z\mid x)\,\|\,\mathcal N(0, I))$ to the reconstruction term. This gives a known prior to sample from and a smooth latent space, making it a generative model.` },
      { t: "qa", level: "beginner", q: "What are the two terms of the VAE loss and what does each do?", a: String.raw`Reconstruction, $-\mathbb E_q[\log p(x\mid z)]$ (e.g. summed BCE), makes decoded samples look like the input. KL$(q(z\mid x)\,\|\,p(z))$ keeps every encoding close to the prior, stopping the encoder from using isolated, point-like codes. Their sum is the negative ELBO.` },
      { t: "qa", level: "beginner", q: "How do you generate a new image with a trained VAE?", a: String.raw`Sample $z \sim \mathcal N(0, I)$ and pass it through the decoder; show the decoder's mean output (e.g. Bernoulli probabilities). The encoder is not needed.` },
      { t: "qa", level: "intermediate", q: "Derive the ELBO.", a: String.raw`$\log p(x) = \log\mathbb E_q[p(x,z)/q(z\mid x)] \ge \mathbb E_q[\log p(x,z) - \log q(z\mid x)]$ by Jensen. Equivalently, $\log p(x) = \mathrm{ELBO} + \mathrm{KL}(q(z\mid x)\,\|\,p(z\mid x))$, so the gap is the KL to the true posterior. Splitting $\log p(x,z)$ gives $\mathbb E_q[\log p(x\mid z)] - \mathrm{KL}(q\,\|\,p(z))$.` },
      { t: "qa", level: "intermediate", q: "What is the reparameterisation trick and why is it needed?", a: String.raw`Writing $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$ with $\epsilon \sim \mathcal N(0, I)$ makes the sample a differentiable function of the encoder outputs, so $\nabla_\phi\mathbb E[f(z)] = \mathbb E_\epsilon[\nabla_\phi f(\mu + \sigma\epsilon)]$ can be estimated by backprop. Without it you would need the score-function estimator, which is unbiased but has much higher variance (6.7× here on a 1-D toy problem).` },
      { t: "qa", level: "intermediate", q: "Why are VAE samples blurry?", a: String.raw`The decoder treats pixels as independent given $z$ (Gaussian or Bernoulli). When several images fit one $z$, the likelihood-best output is their pixel-wise average, which is blurry; noise added to $z$ in training averages more. Showing the decoder mean, and the ELBO's mass-covering behaviour, also contribute. Hierarchical latents and autoregressive or adversarial/perceptual decoders reduce it.` },
      { t: "qa", level: "intermediate", q: "Compute the KL term for $\\mu = (1, 0)$ and $\\sigma = (1, 0.5)$.", a: String.raw`$\tfrac12[(1 + 1 - 1 - 0) + (0 + 0.25 - 1 - \ln 0.25)] = \tfrac12[1 + 0.636294] = 0.818147$ nats.` },
      { t: "qa", level: "advanced", q: "What is posterior collapse, why does it happen and how do you mitigate it?", a: String.raw`The approximate posterior equals the prior for every $x$ (KL ≈ 0), so the decoder ignores $z$. It happens when the decoder can model the data without $z$ (e.g. autoregressive decoders) or the KL weight is high: using $z$ costs KL nats at once, while the benefit comes only after the decoder learns to exploit it. Our 2-D model collapsed fully at β = 4, and temporarily around epoch 10 at β = 1. Remedies: KL annealing / warm-up (β from 0 to 1), free bits (no KL penalty below a per-dimension threshold), weaker decoders, δ-VAE-style constraints, or discrete latents (VQ-VAE).` },
      { t: "qa", level: "advanced", q: "What does β-VAE change, and what are its claims and caveats about disentanglement?", a: "It multiplies the KL by β > 1 (Higgins et al., 2017), pushing each latent dimension towards an independent standard normal. On some datasets with independent generating factors, single latents then track factors like position or scale. Caveats: worse reconstructions, collapsed dimensions if β is too large, and Locatello et al. (2019) proved unsupervised disentanglement is not identifiable without inductive biases; reported gains depend strongly on seeds and hyper-parameters." },
      { t: "qa", level: "advanced", q: "Compare the score-function and pathwise gradient estimators.", a: String.raw`Score function: $\mathbb E_q[f(z)\nabla\log q(z)]$ — needs only values of $f$, works for discrete $z$ and non-differentiable $f$, but has high variance (needs baselines/control variates). Pathwise: $\mathbb E_\epsilon[\nabla f(g_\phi(\epsilon))]$ — needs a differentiable $f$ and a reparameterisable distribution, and usually has far lower variance because it uses gradient information. VAEs use pathwise for continuous latents; discrete latents need REINFORCE, Gumbel-softmax relaxations or straight-through estimators.` },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "Which quantity does training a VAE maximise?", options: ["The exact log-likelihood $\\log p_\\theta(x)$", "The evidence lower bound $\\mathbb E_q[\\log p(x\\mid z)] - \\mathrm{KL}(q(z\\mid x)\\|p(z))$", "The KL divergence between encoder and prior", "The classification accuracy of the code"], answer: 1, explain: "The exact likelihood cannot be computed; the ELBO is a computable lower bound on it." },
          { q: "The gap $\\log p(x) - \\mathrm{ELBO}$ equals…", options: ["$\\mathrm{KL}(q(z\\mid x)\\,\\|\\,p(z))$", "$\\mathrm{KL}(q(z\\mid x)\\,\\|\\,p(z\\mid x))$", "the reconstruction loss", "0 always"], answer: 1, explain: String.raw`The gap is the KL from the approximate to the true posterior; it vanishes only when $q$ equals the posterior.` },
          { kind: "numeric", q: "One latent dimension has $\\mu = 0$, $\\log\\sigma^2 = 0$. What is its KL to $\\mathcal N(0,1)$?", answer: 0, tolerance: 0.0001, explain: String.raw`$\tfrac12(0 + 1 - 1 - 0) = 0$: the posterior equals the prior, so this dimension carries no information.` },
          { kind: "numeric", q: "A latent dimension has $\\mu = 1$, $\\sigma = 0.5$. Compute its KL to $\\mathcal N(0,1)$ in nats (3 decimals).", answer: 0.818, tolerance: 0.002, explain: String.raw`$\tfrac12(1 + 0.25 - 1 - \ln0.25) = \tfrac12(0.25 + 1.386294) = 0.818147$.` },
          { kind: "numeric", q: "With $\\mu = 0.4$, $\\sigma = 0.2$ and $\\epsilon = -1.5$, what is the reparameterised sample $z$?", answer: 0.1, tolerance: 0.0001, explain: String.raw`$z = 0.4 + 0.2\times(-1.5) = 0.1$.` },
          { q: "Why does the encoder output $\\log\\sigma^2$ instead of $\\sigma$?", options: ["It makes the KL zero", "Any real output gives a valid positive variance, and it is numerically convenient for the KL", "It is required by the reparameterisation trick", "It makes sampling deterministic"], answer: 1, explain: String.raw`$\sigma = \exp(\tfrac12\log\sigma^2) > 0$ for any real network output.` },
          { q: "During training the KL term of every example drops to ≈ 0 while reconstructions become the same average image. What has happened?", options: ["Overfitting", "Posterior collapse: the decoder ignores $z$", "Exploding gradients", "The reparameterisation trick failed"], answer: 1, explain: "q(z|x) = prior for every x, so the latent carries no information. Seen in our β = 4 model (KL 0.00) and temporarily at β = 1." },
          { kind: "numeric", q: "In the From-Scratch experiment the per-sample standard deviations of the score-function and pathwise estimators were 13.06 and 5.06. What is the ratio of their variances (1 decimal)?", answer: 6.7, tolerance: 0.1, explain: String.raw`$(13.06/5.06)^2 = 2.581^2 \approx 6.66$.` },
        ],
      },
      {
        t: "exercise",
        title: "The ELBO of the toy model by hand",
        prompt: String.raw`For $z \sim \mathcal N(0,1)$, $x\mid z \sim \mathcal N(wz + b, s^2)$ with $w = 2$, $b = 0.5$, $s = 0.5$ and $x = 2$: (a) compute $\log p(x)$; (b) find the true posterior's mean and variance; (c) compute the ELBO of $q = \mathcal N(1, 0.1)$ using $\mathbb E_q[(x - b - wz)^2] = (x - b - w\mu)^2 + w^2 v$; (d) check that the gap equals $\mathrm{KL}(q\,\|\,p(z\mid x))$.`,
        hint: String.raw`The marginal is $\mathcal N(b, w^2 + s^2) = \mathcal N(0.5, 4.25)$. Posterior precision = prior precision + $w^2/s^2$.`,
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) evidence", tex: String.raw`\log p(x) = -\tfrac12\log(2\pi\cdot4.25) - \frac{1.5^2}{2\cdot4.25} = -1.642398 - 0.264706 = -1.907104` },
              { title: "(b) posterior", tex: String.raw`v_{post} = \frac{1}{1 + 16} = 0.058824,\qquad m_{post} = v_{post}\cdot\frac{w(x-b)}{s^2} = 0.058824\cdot12 = 0.705882` },
              { title: "(c) expected log-likelihood", tex: String.raw`-\tfrac12\log(2\pi\cdot0.25) - \frac{(2 - 0.5 - 2)^2 + 4\cdot0.1}{2\cdot0.25} = -0.225791 - 1.3 = -1.525791` },
              { title: "(c) KL to prior and ELBO", tex: String.raw`\mathrm{KL}(\mathcal N(1,0.1)\|\mathcal N(0,1)) = \tfrac12(\log 10 + 0.1 + 1 - 1) = 1.201293,\qquad \mathrm{ELBO} = -2.727084` },
              { title: "(d) gap", tex: String.raw`\log p(x) - \mathrm{ELBO} = 0.819980,\qquad \mathrm{KL}(q\|p(z\mid x)) = \tfrac12\Big(\log\tfrac{0.058824}{0.1} + \tfrac{0.1 + 0.294118^2}{0.058824} - 1\Big) = 0.819980` },
            ],
          },
          { t: "p", text: String.raw`The gap $-1.907104 - (-2.727084) = 0.819980$ matches the KL to the true posterior (0.819978 by hand from rounded posterior parameters; 0.819980 in the program), confirming $\log p(x) = \mathrm{ELBO} + \mathrm{KL}(q\,\|\,p(z\mid x))$.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a conditional VAE",
        prompt: "Turn `vae_digits.py` into a **conditional VAE** (Sohn, Lee and Yan, 2015): concatenate a one-hot label $c$ (10 numbers) to the encoder input and to $z$ before the decoder. After training, generate a grid of every digit class by fixing $z$ and varying $c$. What should happen to the KL term compared with the unconditional VAE, and why?",
        hint: "`F.one_hot(torch.tensor(y_tr), 10).float()` gives the labels; the first layers become `nn.Linear(64 + 10, 64)` and `nn.Linear(LATENT + 10, 32)`.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "C_tr = F.one_hot(torch.tensor(y_tr), 10).float()\n\nclass CVAE(VAE):\n    def __init__(self):\n        super().__init__()\n        self.body[0] = nn.Linear(64 + 10, 64)\n        self.dec[0] = nn.Linear(LATENT + 10, 32)\n\n    def forward(self, x, c):\n        h = self.body(torch.cat([x, c], dim=1))\n        mu, logvar = self.mu(h), self.logvar(h)\n        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)\n        return self.dec(torch.cat([z, c], dim=1)), mu, logvar\n\n# generation: z fixed, one row per class\n# z = torch.randn(1, LATENT).repeat(10, 1); c = torch.eye(10)\n# imgs = torch.sigmoid(model.dec(torch.cat([z, c], dim=1)))",
          },
          { t: "p", text: "The label now gives the class for free, so $z$ only needs to encode style (slant, thickness, size). The decoder needs less information from $z$, so the KL term should *drop* while the reconstruction term improves or stays similar. Fixing $z$ and varying $c$ should give ten digits in the same style. Check by printing both terms, as in the original program." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          String.raw`Generative model: $z \sim \mathcal N(0, I)$, $x \sim p_\theta(x\mid z)$. The evidence $p_\theta(x) = \int p_\theta(x\mid z)p(z)dz$ and the posterior are intractable.`,
          String.raw`ELBO: $\log p(x) \ge \mathbb E_q[\log p(x\mid z)] - \mathrm{KL}(q(z\mid x)\,\|\,p(z))$, from Jensen; the gap is $\mathrm{KL}(q(z\mid x)\,\|\,p(z\mid x))$.`,
          String.raw`The encoder outputs $\mu$, $\log\sigma^2$; reparameterise $z = \mu + \sigma\odot\epsilon$ so gradients reach the encoder (pathwise estimator, far lower variance than REINFORCE).`,
          String.raw`Closed-form KL: $\tfrac12\sum_j(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2)$.`,
          "Loss = summed reconstruction NLL + β·KL, averaged over the batch (β = 1 for the true ELBO). Digits, 2-D latent: test 21.23 + 2.94 = 24.17 nats; IWAE ($K = 1000$) estimate of −log p(x) ≈ 24.2.",
          "β > 1 trades reconstruction for matching the prior (β-VAE); too much → posterior collapse (KL = 0 at β = 4 here).",
          "Strengths: principled, stable, fast sampling, smooth latent space. Weaknesses: blurry samples, loose bound, collapse, prior holes.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Diffusion Models](/learn/m14/diffusion) — also trained with a variational bound, but with a fixed multi-step noising encoder; latent diffusion runs in a VAE-style autoencoder's latent space.",
          "[Generative Adversarial Networks](/learn/m14/gans) — sharp samples without a likelihood, by training against a discriminator; compare the trade-offs.",
          "[Self-Supervised Learning](/learn/m15/self-supervised) — other ways to learn representations without labels.",
        ],
      },
    ],
  },
};

export default lesson;
