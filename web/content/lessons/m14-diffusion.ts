import type { Lesson } from "@/lib/types";

const NP_CODE = String.raw`import numpy as np
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)

# ---------------- data: two moons, standardised to zero mean / unit variance per axis ----------------
X, _ = make_moons(n_samples=4000, noise=0.05, random_state=0)
MU, SD = X.mean(0), X.std(0)
X = (X - MU) / SD

# ---------------- noise schedule (linear, Ho et al. 2020, rescaled from T=1000 to T=100) ----------------
T = 100
betas = np.linspace(1e-3, 0.2, T)          # beta_t for t = 1..T (stored at index t-1)
alphas = 1.0 - betas
abar = np.cumprod(alphas)                  # alpha-bar_t = prod_{s<=t} alpha_s
print(f"alpha_bar: t=1 {abar[0]:.4f}  t=50 {abar[49]:.4f}  t=100 {abar[-1]:.2e}")

# sanity check of the closed form: 50 sequential noising steps vs one jump to t=50
x = np.tile(X[:1], (20000, 1))             # 20000 copies of one data point x_0
for t in range(50):
    x = np.sqrt(alphas[t]) * x + np.sqrt(betas[t]) * rng.standard_normal(x.shape)
print(f"x_0 = {X[0].round(3)}")
print(f"50 single steps : mean {x.mean(0).round(3)}  std {x.std(0).round(3)}")
print(f"closed form t=50: mean {(np.sqrt(abar[49]) * X[0]).round(3)}  std {np.sqrt(1 - abar[49]):.3f}")

# ---------------- epsilon-network: [x_t, emb(t)] -> 64 -> 64 -> 2, SiLU ----------------
EMB = 16                                   # sinusoidal time-embedding size


def time_emb(t):
    freqs = np.exp(-np.log(200.0) * np.arange(EMB // 2) / (EMB // 2))
    a = t[:, None] * freqs[None]
    return np.concatenate([np.sin(a), np.cos(a)], axis=1)


def silu(z):
    return z / (1 + np.exp(-z))


def dsilu(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 + z * (1 - s))


sizes = [2 + EMB, 64, 64, 2]
W = [rng.normal(0, np.sqrt(1 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
b = [np.zeros(o) for o in sizes[1:]]


def forward(xt, t):
    h = [np.concatenate([xt, time_emb(t)], axis=1)]
    zs = []
    for l in range(3):
        z = h[-1] @ W[l] + b[l]
        zs.append(z)
        h.append(z if l == 2 else silu(z))
    return h, zs


# ---------------- training: Algorithm 1 of DDPM with Adam ----------------
m = [np.zeros_like(p) for p in W + b]
v = [np.zeros_like(p) for p in W + b]
LR, STEPS, BATCH = 3e-3, 4000, 256
for step in range(1, STEPS + 1):
    x0 = X[rng.integers(0, len(X), BATCH)]
    t = rng.integers(0, T, BATCH)          # index t-1, i.e. t uniform on 1..T
    eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps
    h, zs = forward(xt, t.astype(float))
    loss = np.mean((h[-1] - eps) ** 2)
    # backward pass (mean over batch and both coordinates)
    delta = 2 * (h[-1] - eps) / eps.size
    grads_W, grads_b = [None] * 3, [None] * 3
    for l in reversed(range(3)):
        grads_W[l] = h[l].T @ delta
        grads_b[l] = delta.sum(0)
        if l > 0:
            delta = (delta @ W[l].T) * dsilu(zs[l - 1])
    lr = LR * 0.5 * (1 + np.cos(np.pi * step / STEPS))      # cosine learning-rate decay
    for i, (p, g) in enumerate(zip(W + b, grads_W + grads_b)):
        m[i] = 0.9 * m[i] + 0.1 * g
        v[i] = 0.999 * v[i] + 0.001 * g * g
        p -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
    if step % 1000 == 0:
        print(f"step {step}  batch loss {loss:.4f}")


def eps_model(x, t_idx):
    return forward(x, np.full(len(x), float(t_idx)))[0][-1]


# ---------------- sampling: Algorithm 2 (DDPM) and DDIM ----------------
def ddpm_sample(n, rng):
    x = rng.standard_normal((n, 2))
    for t in reversed(range(T)):
        e = eps_model(x, t)
        mean = (x - betas[t] / np.sqrt(1 - abar[t]) * e) / np.sqrt(alphas[t])
        if t > 0:
            var = betas[t] * (1 - abar[t - 1]) / (1 - abar[t])   # posterior variance beta-tilde
            x = mean + np.sqrt(var) * rng.standard_normal(x.shape)
        else:
            x = mean
    return x


def ddim_sample(n, steps, rng):
    x = rng.standard_normal((n, 2))
    ts = np.round(np.linspace(T - 1, 0, steps)).astype(int)
    for i, t in enumerate(ts):
        e = eps_model(x, t)
        x0_hat = (x - np.sqrt(1 - abar[t]) * e) / np.sqrt(abar[t])
        a_prev = abar[ts[i + 1]] if i + 1 < steps else 1.0
        x = np.sqrt(a_prev) * x0_hat + np.sqrt(1 - a_prev) * e
    return x


def energy_distance(a, c):
    """Unbiased estimate of 2E|X-Y| - E|X-X'| - E|Y-Y'| (self-pairs excluded)."""
    dist = lambda p, q: np.sqrt(((p[:, None] - q[None]) ** 2).sum(-1))
    within = lambda p: dist(p, p).sum() / (len(p) * (len(p) - 1))
    return 2 * dist(a, c).mean() - within(a) - within(c)


N = 1000
real = lambda seed: (make_moons(n_samples=N, noise=0.05, random_state=seed)[0] - MU) / SD
ref = real(1)                              # held-out real points (not used for training)
print("energy distance to held-out data (lower is better, ~0 = same distribution)")
print(f"  other real data   {energy_distance(real(2), ref):.4f}")
print(f"  N(0, I) noise     {energy_distance(rng.standard_normal((N, 2)), ref):.4f}")
print(f"  DDPM, 100 steps   {energy_distance(ddpm_sample(N, rng), ref):.4f}")
for S in [5, 10, 20, 50, 100]:
    print(f"  DDIM, {S:3d} steps   {energy_distance(ddim_sample(N, S, rng), ref):.4f}")`;

const NP_OUT = String.raw`alpha_bar: t=1 0.9990  t=50 0.0742  t=100 2.04e-05
x_0 = [-0.561 -0.193]
50 single steps : mean [-0.139 -0.052]  std [0.953 0.959]
closed form t=50: mean [-0.153 -0.053]  std 0.962
step 1000  batch loss 0.2292
step 2000  batch loss 0.2419
step 3000  batch loss 0.2560
step 4000  batch loss 0.2338
energy distance to held-out data (lower is better, ~0 = same distribution)
  other real data   -0.0033
  N(0, I) noise     0.0498
  DDPM, 100 steps   0.0034
  DDIM,   5 steps   0.1277
  DDIM,  10 steps   0.0142
  DDIM,  20 steps   0.0059
  DDIM,  50 steps   0.0018
  DDIM, 100 steps   0.0030`;

const TORCH_CODE = String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(0)
torch.set_num_threads(1)

# ---------------- data: two moons with labels (which moon) for conditional generation ----------------
X, y = make_moons(n_samples=4000, noise=0.2, random_state=0)
mu, sd = X.mean(0), X.std(0)
X = torch.tensor((X - mu) / sd, dtype=torch.float32)
y = torch.tensor(y)
NULL = 2                                   # label index meaning "no condition"

# ---------------- schedule ----------------
T = 100
betas = torch.linspace(1e-3, 0.2, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)


class EpsNet(nn.Module):
    """eps_theta(x_t, t, c): sinusoidal time embedding + learned label embedding, SiLU MLP."""

    def __init__(self, hidden=64, emb=32):
        super().__init__()
        self.emb = emb
        self.label = nn.Embedding(3, emb)          # labels 0, 1 and the null label 2
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * emb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def time_emb(self, t):
        half = self.emb // 2
        f = torch.exp(-math.log(200.0) * torch.arange(half) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=1)

    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.time_emb(t), self.label(c)], dim=1))


model = EpsNet()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
STEPS = 3000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

# ---------------- training (DDPM Algorithm 1, plus label dropout for classifier-free guidance) ----------------
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(X), (256,))
    x0, c = X[idx], y[idx].clone()
    c[torch.rand(256) < 0.15] = NULL              # 15% of the time: train the unconditional model
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    a = abar[t][:, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps    # closed-form q(x_t | x_0)
    loss = F.mse_loss(model(xt, t, c), eps)       # L_simple
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 1000 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
model.eval()


@torch.no_grad()
def sample(n, label, w=0.0, sampler="ddpm", steps=T, gen=None):
    x = torch.randn(n, 2, generator=gen)
    c = torch.full((n,), label)
    null = torch.full((n,), NULL)
    ts = list(range(T - 1, -1, -1)) if sampler == "ddpm" else \
        torch.linspace(T - 1, 0, steps).round().long().tolist()
    for i, t in enumerate(ts):
        tt = torch.full((n,), t)
        e = model(x, tt, c)
        if w != 0:                                # classifier-free guidance
            e = (1 + w) * e - w * model(x, tt, null)
        if sampler == "ddpm":                     # Algorithm 2
            mean = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
            if t > 0:
                var = betas[t] * (1 - abar[t - 1]) / (1 - abar[t])
                x = mean + var.sqrt() * torch.randn(x.shape, generator=gen)
            else:
                x = mean
        else:                                     # DDIM, eta = 0
            x0 = (x - (1 - abar[t]).sqrt() * e) / abar[t].sqrt()
            a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


# a k-NN "judge" fitted on real data tells which moon a generated point lies on
judge = KNeighborsClassifier(n_neighbors=5).fit(X.numpy(), y.numpy())
gen = torch.Generator().manual_seed(1)
print("fraction of samples on the requested moon (label 1), 1000 samples:")
for w in [0.0, 1.0, 3.0]:
    s = sample(1000, 1, w=w, gen=gen)
    on = (judge.predict(s.numpy()) == 1).mean()
    spread = s.std(0).numpy().round(3)
    print(f"  DDPM  w={w:.0f}: {on:.3f}   std of samples {spread}")
for steps in [5, 10, 50]:
    s = sample(1000, 1, w=1.0, sampler="ddim", steps=steps, gen=gen)
    print(f"  DDIM {steps:2d} steps, w=1: {(judge.predict(s.numpy()) == 1).mean():.3f}")
real1 = X[y == 1]
print(f"real moon 1: std {real1.std(0).numpy().round(3)}")`;

const TORCH_OUT = String.raw`step 1000  loss 0.2389
step 2000  loss 0.2557
step 3000  loss 0.2529
fraction of samples on the requested moon (label 1), 1000 samples:
  DDPM  w=0: 0.841   std of samples [0.769 0.671]
  DDPM  w=1: 0.954   std of samples [0.622 0.57 ]
  DDPM  w=3: 0.995   std of samples [0.694 0.635]
  DDIM  5 steps, w=1: 1.000
  DDIM 10 steps, w=1: 0.995
  DDIM 50 steps, w=1: 0.993
real moon 1: std [0.822 0.688]`;

const KERAS_CODE = String.raw`import numpy as np
import tensorflow as tf

T = 100
betas = np.linspace(1e-3, 0.2, T).astype("float32")
abar = tf.constant(np.cumprod(1.0 - betas))

def time_emb(t, dim=32):
    half = dim // 2
    f = tf.exp(-np.log(200.0) * tf.range(half, dtype=tf.float32) / half)
    a = tf.cast(t, tf.float32)[:, None] * f[None]
    return tf.concat([tf.sin(a), tf.cos(a)], axis=1)

x_in = tf.keras.Input(shape=(2,))
t_in = tf.keras.Input(shape=(32,))
h = tf.keras.layers.Concatenate()([x_in, t_in])
for _ in range(3):
    h = tf.keras.layers.Dense(128, activation="swish")(h)
eps_net = tf.keras.Model([x_in, t_in], tf.keras.layers.Dense(2)(h))
opt = tf.keras.optimizers.Adam(2e-3)

@tf.function
def train_step(x0):
    n = tf.shape(x0)[0]
    t = tf.random.uniform([n], 0, T, dtype=tf.int32)
    eps = tf.random.normal(tf.shape(x0))
    a = tf.gather(abar, t)[:, None]
    xt = tf.sqrt(a) * x0 + tf.sqrt(1.0 - a) * eps          # closed-form q(x_t | x_0)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((eps_net([xt, time_emb(t)]) - eps) ** 2)
    grads = tape.gradient(loss, eps_net.trainable_variables)
    opt.apply_gradients(zip(grads, eps_net.trainable_variables))
    return loss

from sklearn.datasets import make_moons
X = make_moons(4000, noise=0.05, random_state=0)[0]
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
ds = tf.data.Dataset.from_tensor_slices(X).shuffle(4000).repeat().batch(256)
for step, batch in enumerate(ds.take(3000)):
    loss = train_step(batch)
    if step % 1000 == 0:
        print(step, float(loss))`;

const lesson: Lesson = {
  id: "m14/diffusion",
  title: "Diffusion Models",
  summary: "Generate data by learning to undo noise: the forward Gaussian process and its closed form, the reverse process and the ELBO that reduces to an ε-prediction MSE, DDPM and DDIM sampling, guidance, and latent diffusion — with real trained models, from-scratch code and measured sample quality.",
  minutes: 110,
  level: "advanced",
  prerequisites: ["m02/probability", "m02/information-theory", "m14/vae", "m04/backpropagation"],
  next: ["m14/gans", "m15/multimodal", "m12/segmentation"],
  terms: ["diffusion-model", "forward-diffusion", "reverse-diffusion", "noise-schedule", "noise-prediction", "ddpm", "ddim", "classifier-guidance", "classifier-free-guidance", "latent-diffusion", "score-function", "elbo"],
  sections: {
    intro: [
      { t: "p", text: "A **diffusion model** generates data by reversing a gradual noising process. During training, real examples are corrupted by adding Gaussian noise in many small steps until nothing but noise remains; a neural network is trained to predict, at any step, the noise that was added. To generate, you start from pure Gaussian noise and apply the network repeatedly, removing a little noise each time, until a clean sample appears." },
      { t: "p", text: "The idea was introduced by Sohl-Dickstein et al. (2015, *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*) and made practical by Ho, Jain and Abbeel (2020, *Denoising Diffusion Probabilistic Models*, **DDPM**), who showed that a simple mean-squared-error loss on the predicted noise gives image quality competitive with GANs. Song et al. (2020, *Denoising Diffusion Implicit Models*, **DDIM**) cut sampling from ~1000 network evaluations to tens; Dhariwal and Nichol (2021, *Diffusion Models Beat GANs on Image Synthesis*) introduced classifier guidance; Ho and Salimans (2022) introduced classifier-free guidance; and Rombach et al. (2021/2022, *High-Resolution Image Synthesis with Latent Diffusion Models*) moved the process into an autoencoder's latent space, which is the architecture behind **Stable Diffusion**." },
      { t: "p", text: "What problem does this solve? Earlier generative models each had a serious weakness: [VAEs](/learn/m14/vae) are stable to train but produce blurry samples; [GANs](/learn/m14/gans) produce sharp samples but their adversarial training is unstable and prone to **mode collapse** (covering only part of the data). Diffusion models are trained with a plain regression loss (stable, no adversary), cover the data distribution well, and produce high-fidelity samples. The price is slow sampling: many sequential network evaluations per sample." },
      { t: "p", text: "Today diffusion models are the dominant approach for text-to-image generation (Stable Diffusion, DALL·E 2, Imagen), and are used for video, audio and speech synthesis, image editing and inpainting, super-resolution, and scientific design problems such as protein structure generation." },
      { t: "callout", kind: "key", text: String.raw`Train a network $\epsilon_\theta(x_t, t)$ to predict the noise in $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with a mean-squared error. Generate by starting at $x_T \sim \mathcal N(0, I)$ and repeatedly subtracting the predicted noise.` },
    ],
    prerequisites: [
      { t: "list", items: [
        String.raw`[Probability and the Gaussian distribution](/learn/m02/probability) — you need $\mathcal N(\mu, \sigma^2 I)$, the fact that a sum of independent Gaussians is Gaussian with variances adding, and that $\mu + \sigma\epsilon$ with $\epsilon \sim \mathcal N(0, I)$ is a sample from $\mathcal N(\mu, \sigma^2 I)$.`,
        String.raw`[Entropy, cross-entropy and KL divergence](/learn/m02/information-theory) — the training objective is a sum of KL divergences between Gaussians; you need $D_{KL}(q\|p) \ge 0$ and its closed form for two Gaussians.`,
        String.raw`[Variational Autoencoders](/learn/m14/vae) — the **ELBO** (evidence lower bound) and the reparameterization trick. A diffusion model is a hierarchical latent-variable model whose ELBO we will derive; the VAE lesson shows the same bound for a single latent.`,
        "[Maximum likelihood](/learn/m02/mle-map) — why maximising a lower bound on $\\log p_\\theta(x)$ is a sensible training goal.",
        "[Backpropagation](/learn/m04/backpropagation) and [multilayer networks](/learn/m03/multilayer-networks) — the from-scratch code trains the noise-prediction MLP with a hand-written backward pass.",
        "For the architecture section: [U-Net / segmentation](/learn/m12/segmentation) (the denoiser of image diffusion models) and [cross-attention](/learn/m11/transformer-architecture) (how text conditions the image).",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "Ink in water, played backwards", text: "Drop ink into a glass of water and it slowly spreads until the water is uniformly grey. Each instant, every ink particle takes a tiny random step. Watching the film backwards, the grey water would reassemble into a drop — which never happens physically, because you would need to know, at every instant, which direction each particle should step. A diffusion model **learns those directions**. Because each forward step is tiny, each backward step is also simple (nearly Gaussian), and a neural network can learn it." },
      { t: "p", text: "Think of a cloud of data points (say, the two interleaving moons in the simulator). The forward process repeatedly shrinks the cloud slightly toward the origin and adds a little random jitter. After enough steps the cloud is a featureless round Gaussian blob, the same no matter what the data was. That end state is easy to sample from." },
      { t: "p", text: "To go backwards, take a noisy point and ask: *what noise was added to it?* If you can answer — even approximately — you can subtract some of that noise and move the point toward where the data lives. Doing this from a blob of pure noise, step after step, carves the blob back into moons. The network never has to produce a whole image in one shot; it only has to solve a series of easy **denoising** problems, each at a known noise level." },
      { t: "p", text: "Two remarks make this concrete. First, at high noise levels the best denoiser can only point toward the *average* of plausible clean data — coarse structure is decided early. At low noise levels it fixes fine detail. Second, because the starting noise is random, every run gives a different sample: randomness in, variety out." },
    ],
    mechanism: [
      { t: "h", text: "Training (DDPM Algorithm 1)" },
      { t: "steps", items: [
        { title: "Fix a noise schedule", text: String.raw`Choose the number of steps $T$ and variances $\beta_1 < \beta_2 < \dots < \beta_T$ in $(0,1)$. Precompute $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. Nothing here is learned.` },
        { title: "Sample a training example and a time step", text: String.raw`Draw $x_0$ from the dataset and $t$ uniformly from $\{1,\dots,T\}$.` },
        { title: "Corrupt it in one jump", text: String.raw`Draw $\epsilon \sim \mathcal N(0, I)$ (same shape as $x_0$) and form $x_t$ directly with the closed form — no loop over steps is needed.`, tex: String.raw`x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon` },
        { title: "Predict the noise", text: String.raw`Feed $x_t$ and $t$ (as a sinusoidal embedding) to the network $\epsilon_\theta$, which outputs a tensor of the same shape as $x_0$.` },
        { title: "Regress", text: "Take a gradient step on the squared error between the true and predicted noise. Repeat from step 2 until converged.", tex: String.raw`\nabla_\theta \,\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2` },
      ] },
      { t: "h", text: "Sampling (DDPM Algorithm 2)" },
      { t: "steps", items: [
        { title: "Start from noise", tex: String.raw`x_T \sim \mathcal N(0, I)` },
        { title: "For t = T, T−1, …, 1: predict the noise", tex: String.raw`\hat\epsilon = \epsilon_\theta(x_t, t)` },
        { title: "Remove the predicted noise (posterior mean)", tex: String.raw`\mu = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\hat\epsilon\Big)` },
        { title: "Add fresh noise (except at the last step)", text: String.raw`$z \sim \mathcal N(0, I)$ if $t > 1$, else $z = 0$. Ho et al. use $\sigma_t^2 = \beta_t$ or $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ (both work similarly); our code uses $\tilde\beta_t$.`, tex: String.raw`x_{t-1} = \mu + \sigma_t z` },
        { title: "Return x₀", text: "After $T$ network evaluations, $x_0$ is the generated sample." },
      ] },
      { t: "h", text: "The network" },
      { t: "p", text: String.raw`$\epsilon_\theta$ must map an input and a time step to an output of the **same shape**. For 2-D toy data an MLP on $[x_t,\ \text{emb}(t)]$ is enough (that is what the simulator runs). For images, the standard choice is a **U-Net**: a convolutional encoder–decoder with skip connections at every resolution, residual blocks into which the time embedding is added, and self-attention at low resolutions. Conditioning information (a class label or a text prompt) enters through extra embeddings or through **cross-attention** layers.` },
      { t: "p", text: String.raw`The time embedding uses the same sinusoids as Transformer [positional encodings](/learn/m10/positional-encoding): $\text{emb}(t) = [\sin(t f_0), \dots, \sin(t f_{h-1}), \cos(t f_0), \dots, \cos(t f_{h-1})]$ with geometrically spaced frequencies $f_k$. One network handles every noise level; $t$ tells it how much noise to expect.` },
      { t: "callout", kind: "warning", text: String.raw`Index conventions trip people up. Papers number steps $t = 1..T$; code usually stores $\beta$ in a 0-based array, so $\beta_t$ lives at index $t-1$. The network must be fed the **same** convention at sampling time as at training time.` },
    ],
    math: [
      { t: "h", text: "1. The forward process and its closed form" },
      { t: "p", text: String.raw`The forward (noising) process is a fixed Markov chain that adds Gaussian noise with variance $\beta_t$ at step $t$, while scaling the signal by $\sqrt{1-\beta_t}$ so the total variance stays bounded:` },
      { t: "math", tex: String.raw`q(x_t \mid x_{t-1}) = \mathcal N\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big), \qquad q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})` },
      { t: "p", text: String.raw`Here $x_0$ is a data example, $x_1, \dots, x_T$ are progressively noisier versions with the same shape, $\beta_t \in (0,1)$ is the noise variance at step $t$ and $I$ is the identity covariance. Define $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. **Claim:** $q(x_t \mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I)$.` },
      { t: "p", text: String.raw`**Derivation.** Reparameterize each step as $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}$ with independent $\epsilon_{t-1} \sim \mathcal N(0, I)$. Substitute the same expression for $x_{t-1}$:` },
      { t: "math", tex: String.raw`x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}}_{\text{sum of independent zero-mean Gaussians}}` },
      { t: "p", text: String.raw`Independent Gaussian variances add: $\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}$. So the two noise terms equal $\sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\epsilon$ for a single standard normal $\bar\epsilon$. The same pattern holds at every level: if $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\bar\epsilon$, then $x_t$ has signal coefficient $\sqrt{\alpha_t\bar\alpha_{t-1}} = \sqrt{\bar\alpha_t}$ and noise variance $\alpha_t(1-\bar\alpha_{t-1}) + 1-\alpha_t = 1-\bar\alpha_t$. By induction:` },
      { t: "math", tex: String.raw`x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal N(0, I)`, caption: "The closed form: jump from clean data to any noise level in one step. This is what makes training cheap." },
      { t: "p", text: String.raw`Two consequences. (i) If $\bar\alpha_T \approx 0$, then $q(x_T \mid x_0) \approx \mathcal N(0, I)$ whatever $x_0$ was, so generation can start from standard noise. (ii) If the data has unit variance, every $x_t$ also has unit variance ($\bar\alpha_t + 1 - \bar\alpha_t = 1$): this is why the scheme is called **variance-preserving**. The **signal-to-noise ratio** is $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$, which decreases monotonically in $t$.` },
      { t: "h", text: "2. Noise schedules" },
      { t: "p", text: String.raw`**Linear** (Ho et al. 2020): $\beta_t$ evenly spaced from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ with $T = 1000$. For a different $T$, Nichol & Dhariwal (2021) rescale both ends by $1000/T$; with our $T = 100$ that gives $\beta_1 = 10^{-3}$, $\beta_T = 0.2$, and $\bar\alpha_T \approx \exp(-\sum_t \beta_t) = e^{-10.05} \approx 4\times10^{-5}$ (exact product: $2.04\times10^{-5}$).` },
      { t: "p", text: String.raw`**Cosine** (Nichol & Dhariwal 2021, *Improved Denoising Diffusion Probabilistic Models*): define $\bar\alpha_t$ directly and derive $\beta_t = 1 - \bar\alpha_t/\bar\alpha_{t-1}$, clipped at $0.999$ to avoid a singularity at $t = T$:` },
      { t: "math", tex: String.raw`\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\!\Big(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\Big), \quad s = 0.008` },
      { t: "p", text: String.raw`With the linear schedule, $\bar\alpha_t$ collapses fast: at $T = 100$ it is $0.528$ at $t = 25$ and $0.074$ at $t = 50$, so the last half of the steps all look like near-pure noise and contribute little. The cosine schedule keeps $\bar\alpha_{25} = 0.847$, $\bar\alpha_{50} = 0.494$, $\bar\alpha_{75} = 0.144$: it destroys information more evenly, which Nichol & Dhariwal found improves log-likelihood, especially on low-resolution images. The offset $s$ keeps $\beta_1$ from being vanishingly small. Compare the two curves in the simulator's Forward tab.` },
      { t: "h", text: "3. The reverse process" },
      { t: "p", text: String.raw`Generation needs the reverse conditionals $q(x_{t-1}\mid x_t)$, which depend on the unknown data distribution. For small $\beta_t$ they are approximately Gaussian (Sohl-Dickstein et al. 2015), so we model them as` },
      { t: "math", tex: String.raw`p_\theta(x_{t-1}\mid x_t) = \mathcal N\big(x_{t-1};\ \mu_\theta(x_t, t),\ \sigma_t^2 I\big), \qquad p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t), \quad p(x_T) = \mathcal N(0, I)` },
      { t: "p", text: String.raw`Although $q(x_{t-1}\mid x_t)$ is intractable, the reverse step **conditioned on $x_0$** is a Gaussian we can compute exactly with Bayes' rule (all three factors below are Gaussians we know):` },
      { t: "math", tex: String.raw`q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)} = \mathcal N\big(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\big)` },
      { t: "math", tex: String.raw`\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t` },
      { t: "p", text: String.raw`(Complete the square in $x_{t-1}$ in the exponent: the precision is $\alpha_t/\beta_t + 1/(1-\bar\alpha_{t-1}) = (1-\bar\alpha_t)/(\beta_t(1-\bar\alpha_{t-1}))$, whose inverse is $\tilde\beta_t$.) Now substitute $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ from the closed form. Using $\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$, the coefficient of $x_t$ becomes $\frac{\beta_t + \alpha_t(1-\bar\alpha_{t-1})}{(1-\bar\alpha_t)\sqrt{\alpha_t}} = \frac{1-\bar\alpha_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}} = \frac{1}{\sqrt{\alpha_t}}$, and the coefficient of $\epsilon$ becomes $-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$:` },
      { t: "math", tex: String.raw`\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big)` },
      { t: "p", text: String.raw`This suggests the parameterization used in DDPM: let the network predict the noise and plug it in, $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\big)$. That is exactly the sampling update of Algorithm 2.` },
      { t: "h", text: "4. The ELBO and its simplification to an MSE on the noise" },
      { t: "p", text: String.raw`As in a [VAE](/learn/m14/vae), we maximise a lower bound on the log-likelihood, here with $x_{1:T}$ as latent variables and the fixed $q(x_{1:T}\mid x_0)$ as the "encoder". By Jensen's inequality:` },
      { t: "math", tex: String.raw`\log p_\theta(x_0) = \log \mathbb E_{q(x_{1:T}\mid x_0)}\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big] \ \ge\ \mathbb E_q\Big[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big] =: -L` },
      { t: "p", text: String.raw`Rewrite each forward factor with Bayes' rule conditioned on $x_0$, $q(x_t\mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0) = q(x_{t-1}\mid x_t, x_0)\,q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$ (the first equality is the Markov property). The ratio $q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$ telescopes across $t = 2..T$, leaving $q(x_T\mid x_0)/q(x_1\mid x_0)$, and the loss splits into a sum of KL divergences:` },
      { t: "math", tex: String.raw`L = \underbrace{D_{KL}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T:\ \text{no parameters}} + \sum_{t=2}^{T} \underbrace{\mathbb E_q\, D_{KL}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} \;\underbrace{-\ \mathbb E_q \log p_\theta(x_0\mid x_1)}_{L_0}` },
      { t: "p", text: String.raw`Both distributions inside $L_{t-1}$ are Gaussians with the same fixed covariance $\sigma_t^2 I$ (taking $\sigma_t^2 = \tilde\beta_t$ or $\beta_t$, not learned). For two such Gaussians, $D_{KL} = \frac{1}{2\sigma_t^2}\|\tilde\mu_t - \mu_\theta\|^2 + \text{const}$ (see [KL divergence](/learn/m02/information-theory)). Inserting both $\epsilon$-forms of the means, the $x_t$ terms cancel:` },
      { t: "math", tex: String.raw`L_{t-1} = \mathbb E_{x_0, \epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\|^2\Big] + C` },
      { t: "p", text: String.raw`Every term of the ELBO is therefore a **weighted noise-prediction MSE**. Ho et al. found empirically that dropping the weight — training all $t$ equally — gives better samples (it down-weights the very easy small-$t$ terms and lets the network concentrate on harder, noisier ones):` },
      { t: "math", tex: String.raw`L_{\text{simple}}(\theta) = \mathbb E_{t \sim \mathcal U\{1..T\},\ x_0,\ \epsilon \sim \mathcal N(0,I)} \Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\|^2\Big]` },
      { t: "callout", kind: "note", text: String.raw`$L_{\text{simple}}$ is no longer exactly the ELBO, so it is not a likelihood bound; it is a reweighted version that optimises sample quality. Sampling uniformly over $t$ and using one $(t, \epsilon)$ per example gives an unbiased stochastic estimate of it — exactly Algorithm 1.` },
      { t: "h", text: "5. Connection to score matching and denoising" },
      { t: "p", text: String.raw`The **score** of a density $p$ is $\nabla_x \log p(x)$: a vector field pointing toward higher density. For the Gaussian $q(x_t\mid x_0)$ it is $\nabla_{x_t}\log q(x_t\mid x_0) = -(x_t - \sqrt{\bar\alpha_t}x_0)/(1-\bar\alpha_t) = -\epsilon/\sqrt{1-\bar\alpha_t}$. Vincent (2011) showed that regressing a network onto this conditional score (denoising score matching) has the same minimiser as regressing onto the score of the noisy *marginal* $q_t(x_t) = \int q(x_t\mid x_0)\,p_{\text{data}}(x_0)\,dx_0$. Hence the optimal noise predictor satisfies` },
      { t: "math", tex: String.raw`\epsilon_\theta^*(x_t, t) = -\sqrt{1-\bar\alpha_t}\;\nabla_{x_t}\log q_t(x_t), \qquad \hat x_0 := \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon^*_\theta}{\sqrt{\bar\alpha_t}} = \mathbb E[x_0 \mid x_t]` },
      { t: "p", text: String.raw`The second identity (Tweedie's formula) follows because the MSE-optimal $\epsilon_\theta^*$ is the posterior mean $\mathbb E[\epsilon\mid x_t]$. So a trained diffusion model is simultaneously a **denoiser** (its $\hat x_0$ is the posterior-mean clean image) and a **score estimator** at every noise level. This is the bridge to score-based models (Song & Ermon 2019, noise-conditional score networks with Langevin sampling) and to the continuous-time view of Song et al. (2021, *Score-Based Generative Modeling through Stochastic Differential Equations*), where DDPM is a discretisation of a reverse-time SDE and DDIM of the associated *probability-flow ODE*.` },
      { t: "h", text: "6. DDIM: deterministic sampling with fewer steps" },
      { t: "p", text: String.raw`Key observation (Song, Meng & Ermon 2020): $L_{\text{simple}}$ only involves the marginals $q(x_t\mid x_0)$, never the joint chain. Any other process with the same marginals is trained by the same loss, so a trained $\epsilon_\theta$ can be used with a different — non-Markovian — sampler. Write the current state with the network's estimates of its two ingredients:` },
      { t: "math", tex: String.raw`x_t = \sqrt{\bar\alpha_t}\,\hat x_0 + \sqrt{1-\bar\alpha_t}\,\hat\epsilon, \qquad \hat\epsilon = \epsilon_\theta(x_t,t),\quad \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}` },
      { t: "p", text: String.raw`A point at an earlier noise level $s < t$ with the same marginal $q(x_s\mid x_0)$ is $\sqrt{\bar\alpha_s}\,x_0 + \sqrt{1-\bar\alpha_s}\,\epsilon'$. DDIM's general update splits the noise into a part re-used from $\hat\epsilon$ and a fresh part of variance $\sigma^2$ (choosing the coefficients so the total variance is $1-\bar\alpha_s$):` },
      { t: "math", tex: String.raw`x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\;\hat\epsilon + \sigma z, \qquad \sigma = \eta\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}}` },
      { t: "p", text: String.raw`With $\eta = 1$ and $s = t-1$, $\sigma^2 = \tilde\beta_t$ and the update is the DDPM step. With $\eta = 0$ the sampler is **deterministic**: the same $x_T$ always gives the same $x_0$, so $x_T$ acts as a latent code (enabling interpolation and inversion). Because nothing in the derivation requires $s = t-1$, we may jump along any decreasing subsequence $\tau_1 > \dots > \tau_S$ of time steps, using only $S \ll T$ network evaluations. The $\eta = 0$ update can also be read as an Euler step of an ODE: dividing by $\sqrt{\bar\alpha}$,` },
      { t: "math", tex: String.raw`\frac{x_s}{\sqrt{\bar\alpha_s}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \Big(\sqrt{\tfrac{1-\bar\alpha_s}{\bar\alpha_s}} - \sqrt{\tfrac{1-\bar\alpha_t}{\bar\alpha_t}}\Big)\,\epsilon_\theta(x_t, t)` },
      { t: "p", text: String.raw`i.e. Euler's method for $d\bar x = \epsilon_\theta\, d\gamma$ in the variables $\bar x = x/\sqrt{\bar\alpha}$, $\gamma = \sqrt{(1-\bar\alpha)/\bar\alpha}$. Smooth ODE trajectories tolerate large steps, which is why 20–50 DDIM steps typically give samples close to 1000-step DDPM.` },
      { t: "h", text: "7. Guidance" },
      { t: "p", text: String.raw`**Classifier guidance** (Dhariwal & Nichol 2021). For a class-conditional sample we want the score of $p(x_t\mid y) \propto p(x_t)\,p(y\mid x_t)$, i.e. $\nabla\log p(x_t) + \nabla\log p(y\mid x_t)$. Train a separate classifier $p_\phi(y\mid x_t, t)$ **on noisy inputs** and add its gradient, scaled by $s$, to the score. In $\epsilon$-terms (using $\epsilon = -\sqrt{1-\bar\alpha_t}\,\nabla\log p$):` },
      { t: "math", tex: String.raw`\hat\epsilon(x_t) = \epsilon_\theta(x_t, t) - s\,\sqrt{1-\bar\alpha_t}\;\nabla_{x_t}\log p_\phi(y\mid x_t)` },
      { t: "p", text: String.raw`$s > 1$ sharpens the distribution toward examples the classifier finds unambiguous ($p(y\mid x)^s$), trading diversity for fidelity. The cost is an extra noise-robust classifier and its gradient at every step.` },
      { t: "p", text: String.raw`**Classifier-free guidance** (Ho & Salimans 2022). Train *one* network that is conditional, $\epsilon_\theta(x_t, t, c)$, but replace $c$ by a null token $\varnothing$ with probability $p_{\text{uncond}}$ (they use 0.1–0.2), so the same network also learns the unconditional $\epsilon_\theta(x_t, t, \varnothing)$. At sampling time combine the two predictions. In Ho & Salimans' convention, with guidance weight $w \ge 0$:` },
      { t: "math", tex: String.raw`\tilde\epsilon = (1+w)\,\epsilon_\theta(x_t, t, c) - w\,\epsilon_\theta(x_t, t, \varnothing) \;=\; \epsilon_\theta(x_t, t, \varnothing) + s\,\big(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\varnothing)\big),\quad s = 1+w` },
      { t: "p", text: String.raw`$w = 0$ ($s = 1$) is plain conditional sampling; $w > 0$ extrapolates away from the unconditional prediction. The $s$-form is what most software calls the "guidance scale" (Stable Diffusion's commonly used default of 7.5 is $s$, i.e. $w = 6.5$). Why it works: by Bayes, $\nabla\log p(c\mid x_t) = \nabla\log p(x_t\mid c) - \nabla\log p(x_t)$, so $\tilde\epsilon$ corresponds to the score $\nabla\log p(x_t\mid c) + w\,\nabla\log p(c\mid x_t)$ — classifier guidance with an *implicit* classifier built from the model itself. Cost: two network evaluations per step (batched together in practice).` },
      { t: "h", text: "8. Worked numerical example" },
      { t: "p", text: String.raw`Linear schedule, $T = 100$, $\beta_t$ from $0.001$ to $0.2$. One-dimensional data point $x_0 = 1.5$, noise $\epsilon = -0.8$, time step $t = 10$. Every number below was computed in Python.` },
      { t: "steps", items: [
        { title: "Schedule values at t = 10", tex: String.raw`\beta_{10} = 0.001 + \tfrac{9}{99}(0.199) = 0.019091,\quad \bar\alpha_{10} = \prod_{s=1}^{10}(1-\beta_s) = 0.903813,\quad \sqrt{\bar\alpha_{10}} = 0.950691,\quad \sqrt{1-\bar\alpha_{10}} = 0.310140` },
        { title: "Forward jump (closed form)", tex: String.raw`x_{10} = 0.950691(1.5) + 0.310140(-0.8) = 1.426036 - 0.248112 = 1.177924` },
        { title: "Suppose the network predicts ε̂ = −0.7 (slightly wrong)", text: "Its implied clean estimate:", tex: String.raw`\hat x_0 = \frac{1.177924 - 0.310140(-0.7)}{0.950691} = 1.467377 \quad(\text{true } 1.5)` },
        { title: "DDPM step to t = 9: mean", tex: String.raw`\mu = \frac{1}{\sqrt{0.980909}}\Big(1.177924 - \underbrace{\tfrac{0.019091}{0.310140}}_{0.061556}(-0.7)\Big) = 1.009684 \times 1.221013 = 1.232838` },
        { title: "DDPM step: noise", text: String.raw`$\tilde\beta_{10} = \frac{1-\bar\alpha_9}{1-\bar\alpha_{10}}\beta_{10} = 0.015600$, so $\sigma = 0.124898$ and $x_9 = 1.232838 + 0.124898\,z$. (Check: the posterior-mean form $\tilde\mu(x_t, \hat x_0)$ gives the same 1.232838.)` },
        { title: "DDIM jump straight to t = 5 instead", tex: String.raw`\bar\alpha_5 = 0.975130:\quad x_5 = \sqrt{0.975130}(1.467377) + \sqrt{0.024870}(-0.7) = 1.338624` },
        { title: "Classifier-free guidance arithmetic", text: String.raw`If $\epsilon_\theta(x, c) = 0.4$ and $\epsilon_\theta(x, \varnothing) = 0.1$, then with $w = 2$: $\tilde\epsilon = 3(0.4) - 2(0.1) = 1.0$ — the guided prediction overshoots the conditional one in the direction "away from unconditional".` },
      ] },
      { t: "h", text: "9. Latent diffusion and Stable Diffusion" },
      { t: "p", text: String.raw`Running diffusion on pixels is expensive: every one of the many sampling steps processes the full-resolution image. **Latent diffusion** (Rombach et al.) first trains an autoencoder — an encoder $\mathcal E$ and decoder $\mathcal D$ with a perceptual and adversarial loss and a light KL (or vector-quantisation) regulariser, i.e. a [VAE](/learn/m14/vae)-like model — that compresses an image by a spatial factor $f$ (for Stable Diffusion v1: $512\times512\times3 \to 64\times64\times4$, $f = 8$). The diffusion model is then trained on latents $z_0 = \mathcal E(x_0)$ with exactly the same loss, and a generated latent is decoded with $\mathcal D$. The autoencoder removes imperceptible high-frequency detail; diffusion models the semantic content at ~48× fewer dimensions.` },
      { t: "p", text: String.raw`The denoiser is a U-Net (as in [segmentation](/learn/m12/segmentation)) with residual blocks that receive the time embedding, and Transformer blocks that apply self-attention followed by **cross-attention** to the conditioning: queries come from the image features, keys and values from a sequence of text-token embeddings $\tau_\theta(y)$ — exactly the encoder–decoder attention of the [Transformer](/learn/m11/transformer-architecture). Stable Diffusion v1 uses a frozen CLIP ViT-L/14 text encoder for $\tau_\theta$, and samples with classifier-free guidance.` },
    ],
    visual: [
      { t: "p", text: "The simulator runs **real trained models** in your browser: two DDPM noise-prediction networks trained by the committed script `scripts/train-diffusion.py` — an MLP on 2-D two-moons data and a class-conditional MLP on scikit-learn's 8×8 handwritten digits. The browser engine's outputs match PyTorch's to floating-point precision (shown under the simulator)." },
      { t: "sim", id: "diffusion", caption: "Forward: closed-form noising and the two schedules. Reverse: the trained network turns noise into moons, DDPM vs DDIM. Digits: guided class-conditional generation." },
      { t: "list", items: [
        String.raw`**Forward tab.** Drag $t$ from 0 to 100. Watch the moons dissolve by $t \approx 40$ with the linear schedule. Switch to cosine at the same $t$: much more structure remains. In the $\bar\alpha_t$ chart, find where each schedule crosses SNR = 1 ($\log_{10}\mathrm{SNR} = 0$). Read off point #1's numbers and verify $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ by hand.`,
        String.raw`**Reverse tab, DDPM.** Press Play. For the first ~50 steps almost nothing seems to happen (the network can only push points toward the overall mean), then the two arcs form quickly, then the points sharpen. Tick *show predicted* $\hat x_0$: early on the gold guesses collapse to the centre of the data; later they spread onto the moons — Tweedie's $\mathbb E[x_0\mid x_t]$ in action.`,
        "**Reverse tab, DDIM.** Switch to DDIM and set 10 steps, then 5, then 2. Compare the final energy distance with the PyTorch table: quality degrades gracefully down to ~10–20 steps and collapses below that. Note that DDIM with the same seed always gives the same result (it is deterministic); DDPM changes with the noise seed.",
        String.raw`**Digits tab.** Set $w = 0$ and generate 0–9: some digits are vague or the wrong class. Raise $w$ to 3: digits become bolder and more consistently correct, but within one class (pick a single class) they look more alike — the fidelity–diversity trade-off of guidance. Watch the $\hat x_0$ row: the digit's identity is fixed early, details late.`,
      ] },
    ],
    scratch: [
      { t: "p", text: "A complete DDPM in NumPy: the linear schedule, a check that the closed form matches 50 sequential noising steps, an ε-prediction MLP with a hand-written backward pass and Adam, Algorithm 1 for training, Algorithm 2 (DDPM) and DDIM for sampling, and a quantitative comparison using the **energy distance** $D_E(P,Q) = 2\mathbb E\|X-Y\| - \mathbb E\|X-X'\| - \mathbb E\|Y-Y'\|$ ($X, X' \sim P$, $Y, Y' \sim Q$), which is zero if and only if the two distributions are equal." },
      { t: "code", lang: "python", title: "ddpm_from_scratch.py", code: NP_CODE, output: NP_OUT },
      { t: "p", text: "Reading the output:" },
      { t: "list", items: [
        String.raw`**Closed form check.** Running 50 single noising steps on 20,000 copies of one point gives mean $(-0.139, -0.052)$ and standard deviation $(0.953, 0.959)$; the one-jump formula predicts mean $\sqrt{\bar\alpha_{50}}x_0 = (-0.153, -0.053)$ and standard deviation $\sqrt{1-\bar\alpha_{50}} = 0.962$. The differences are within Monte-Carlo error (the standard error of each mean is $0.96/\sqrt{20000} \approx 0.007$).`,
        String.raw`**The loss plateaus near 0.23–0.26 and never approaches 0.** At large $t$, $x_t$ is almost pure noise and the noise genuinely cannot be predicted from it; the batch loss also fluctuates because each batch draws different $t$. That is why the loss value is a poor progress indicator and sample quality must be measured.`,
        String.raw`**Distribution distance.** Two independent real samples of 1,000 points score $-0.0033$: the unbiased estimator can be slightly negative, and this number tells you the noise level of the metric at this sample size (about $\pm0.003$). Pure noise scores $0.0498$. The 100-step DDPM sampler reaches $0.0034$ — indistinguishable from real data at this resolution of the metric.`,
        String.raw`**DDIM vs steps.** With the same trained network, DDIM scores $0.1277$ with 5 steps (worse than pure noise: the points are pushed into the wrong places), $0.0142$ with 10, $0.0059$ with 20, and $0.0018$–$0.0030$ with 50–100 steps — so 50 deterministic steps match the 100-step stochastic sampler, but quality collapses below about 10 steps. The simulator's PyTorch-trained model shows the same pattern.`,
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "A **conditional** DDPM on two moons where the label says which moon to generate, trained with label dropout so the same network supports **classifier-free guidance**. Sampling supports DDPM and DDIM; a k-nearest-neighbour classifier fitted on the real data acts as a judge of whether samples land on the requested moon." },
          { t: "code", lang: "python", title: "ddpm_cfg_pytorch.py", code: TORCH_CODE, output: TORCH_OUT },
          { t: "p", text: String.raw`The moons here are deliberately noisy ($\text{noise} = 0.2$), so they overlap and the condition is genuinely ambiguous near the overlap. Without guidance ($w = 0$) 84.1% of samples requested for moon 1 land in the region the judge assigns to moon 1 (the conditional model reproduces the overlap faithfully). Guidance pushes samples away from the ambiguous region: 95.4% at $w = 1$ and 99.5% at $w = 3$. The price is diversity: the samples' spread drops from $(0.769, 0.671)$ to $(0.622, 0.570)$ at $w = 1$, versus $(0.822, 0.688)$ for the real moon. (At $w = 3$ the spread rises again to $(0.694, 0.635)$: large $w$ extrapolates well beyond both of the network's predictions, and the guided "score" is no longer the score of any particular distribution, so its effect need not be monotone.) DDIM with 5–50 steps and $w = 1$ gives 99.3–100% on-target.` },
          { t: "callout", kind: "tip", title: "Production libraries", text: "For real images use a U-Net and a maintained library such as Hugging Face `diffusers`, which provides schedulers (DDPM, DDIM, DPM-Solver, …) and pipelines. Its DDIM scheduler implements exactly the update derived above; the `guidance_scale` argument of its text-to-image pipelines is $s = 1 + w$." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same training loop (Algorithm 1) in TensorFlow: a functional Keras model on `[x_t, emb(t)]`, the closed-form corruption inside a `tf.function`, and a `GradientTape` step. Sampling is identical to the NumPy version." },
          { t: "code", lang: "python", title: "ddpm_keras.py (not executed in CI)", code: KERAS_CODE, output: "Prints the step index and the batch loss every 1000 steps; the loss falls from about 1.0 at step 0 to roughly 0.2–0.25, the same range as the NumPy and PyTorch versions (it cannot reach 0: at large t the noise is genuinely unpredictable from x_t)." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Variant", "What changes", "Advantages", "Disadvantages / use"], rows: [
        ["DDPM (Ho et al. 2020)", "Markov reverse chain, fixed variance, $L_{\\text{simple}}$", "Simple, stable, high quality", "~1000 sequential steps to sample"],
        ["Improved DDPM (Nichol & Dhariwal 2021)", "Cosine schedule, learned reverse variances, hybrid loss", "Better log-likelihood; good samples with fewer steps", "Slightly more complex objective"],
        ["DDIM (Song et al. 2020)", "Non-Markovian, deterministic ($\\eta=0$) sampler on the same trained model", "10–50 steps; deterministic latent $x_T$ enables inversion and interpolation", "Very few steps (<10) still degrade"],
        ["Score SDE / probability-flow ODE (Song et al. 2021)", "Continuous time; sample by solving reverse SDE or ODE", "Unifies DDPM and score models; exact likelihoods via the ODE; any ODE solver", "Mathematically heavier"],
        ["Higher-order solvers (e.g. DPM-Solver)", "Better ODE integrators for the same model", "Good samples in ~10–25 steps", "Sampler-only change; still iterative"],
        ["Distillation (progressive distillation, consistency models)", "Train a student to cover several steps at once", "1–4 step sampling", "Extra training; some quality loss"],
        ["Classifier guidance", "Add gradient of a noisy-input classifier", "Strong fidelity gains on class-conditional models", "Needs a separate classifier trained on noisy data"],
        ["Classifier-free guidance", "One model with label dropout; extrapolate cond. vs uncond.", "No classifier; works with text; the standard today", "2× network evaluations; high $w$ reduces diversity, oversaturates"],
        ["Latent diffusion / Stable Diffusion", "Diffuse in a pretrained autoencoder's latent space; cross-attention to text", "Much cheaper at high resolution; flexible conditioning", "Detail limited by the autoencoder; two-stage training"],
        ["v-prediction / x₀-prediction", "Network predicts $v = \\sqrt{\\bar\\alpha}\\epsilon - \\sqrt{1-\\bar\\alpha}x_0$ or $x_0$ instead of $\\epsilon$", "Better behaved at very low SNR; used for distillation", "Equivalent in principle; changes the effective loss weighting"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Text-to-image generation**: Stable Diffusion (latent diffusion, open weights), DALL·E 2 (diffusion decoder conditioned on CLIP image embeddings) and Imagen (pixel-space cascaded diffusion conditioned on a frozen T5 text encoder) — all use classifier-free guidance.",
        "**Image editing**: inpainting (generate only masked regions), image-to-image translation by partially noising an input and denoising it with a new prompt (SDEdit-style), and DDIM inversion for editing real photographs.",
        "**Super-resolution and restoration**: diffusion models conditioned on a low-resolution or degraded image (e.g. SR3, and the cascaded upsamplers inside Imagen).",
        "**Audio and speech**: neural vocoders such as WaveGrad and DiffWave generate waveforms by denoising.",
        "**Video generation**: spatio-temporal U-Nets or Transformers denoising stacks of frames or video latents.",
        "**Science**: generating 3-D molecular conformations and protein structures/backbones (e.g. RFdiffusion), where the data are coordinates rather than pixels.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Stable training**: a plain regression loss with no adversary — no mode collapse, no balancing of two networks. The loss is also a (reweighted) likelihood bound, so diffusion models cover the whole distribution.",
        "**High sample quality and diversity**: on ImageNet, Dhariwal & Nichol reported better FID than the best GANs of the time ([FID is explained in the GANs lesson](/learn/m14/gans)).",
        "**Slow sampling**: a GAN generates with one forward pass; DDPM needs $T$ (often 1000) sequential passes and DDIM still 10–50. Each pass is a large U-Net. Distillation and better solvers narrow but do not remove this gap.",
        "**Compute-hungry training**: models must learn every noise level; image models train for very many steps. Latent diffusion reduces the per-step cost substantially.",
        "**Flexible conditioning**: guidance, inpainting, editing and composition all work at sampling time with a single trained model.",
        "**No compact latent by default**: unlike a VAE, the \"latent\" $x_T$ has the same dimension as the data; DDIM makes it deterministic but it is not semantically organised.",
        "**Guidance trade-off**: larger $w$ improves prompt adherence and per-sample quality but reduces diversity and can oversaturate colours.",
        "**Memorisation**: with small datasets (or duplicated training images) the model can reproduce training examples — our 8×8 digits model partly does, as the nearest-neighbour distance in the simulator shows.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        String.raw`**Using $\beta_t$ where $\bar\alpha_t$ is needed (or vice versa).** The closed form uses the cumulative product $\bar\alpha_t$; the single-step sampler uses $\alpha_t$ and $\beta_t$. Swapping them gives noise levels that are wrong by orders of magnitude. Fix: name arrays explicitly and unit-test the closed form against iterated single steps (as the NumPy code does).`,
        "**Off-by-one in time indices** between training and sampling (feeding $t$ at training but $t-1$ at sampling, or reading $\\bar\\alpha_{t-1}$ at index $t-1$ instead of $t-2$). The model then denoises for the wrong noise level. Fix: one convention, documented, used in both loops.",
        String.raw`**Adding noise at the final step.** At $t = 1$ the sampler must return the mean; adding $\sigma_1 z$ leaves visible noise in every sample.`,
        String.raw`**Thinking the network predicts $x_{t-1}$ or $x_0$ directly.** In DDPM it predicts the noise $\epsilon$; $x_{t-1}$ is computed by the fixed formula. (Predicting $x_0$ or $v$ is a legitimate variant, but then the sampler formulas change accordingly.)`,
        "**Not scaling data to a sensible range.** The schedule assumes data of roughly unit scale (images in $[-1, 1]$, standardised features). Data with variance 100 is barely corrupted at small $t$ and never reaches $\\mathcal N(0,I)$ at $t = T$.",
        String.raw`**Mixing guidance conventions.** $\tilde\epsilon = (1+w)\epsilon_c - w\epsilon_\varnothing$ and $\tilde\epsilon = \epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)$ coincide only when $s = 1 + w$. Copying "guidance 7.5" into the $w$-form means $s = 8.5$.`,
        "**Forgetting label dropout** when training for classifier-free guidance: without it the null-token prediction $\\epsilon(x, \\varnothing)$ is untrained garbage and guidance produces noise.",
        "**Using the raw (non-EMA) weights for sampling.** Diffusion training is noisy; an exponential moving average of the weights gives noticeably cleaner samples and is standard practice.",
        "**Judging training by the loss value alone.** The loss plateaus well above 0 (at high $t$ the noise is unpredictable), and small loss changes can mean large sample-quality changes. Evaluate samples with a distribution metric (FID for images; energy distance or MMD for toy data).",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What does the neural network in a DDPM actually predict?", a: "The noise $\\epsilon$ that was added to a clean example to produce the noisy input $x_t$, given $x_t$ and the time step $t$. The less-noisy $x_{t-1}$ is then computed from this prediction with a fixed formula." },
      { t: "qa", level: "beginner", q: "Why can training sample $x_t$ at any $t$ in one step instead of simulating the whole noising chain?", a: "Because composing Gaussian noising steps gives another Gaussian: $q(x_t\\mid x_0) = \\mathcal N(\\sqrt{\\bar\\alpha_t}x_0, (1-\\bar\\alpha_t)I)$, so $x_t = \\sqrt{\\bar\\alpha_t}x_0 + \\sqrt{1-\\bar\\alpha_t}\\epsilon$ directly." },
      { t: "qa", level: "beginner", q: "Why does generation start from standard Gaussian noise?", a: "The schedule is chosen so that $\\bar\\alpha_T \\approx 0$; then $x_T$ is (almost) $\\mathcal N(0, I)$ regardless of the data, so a sample from $\\mathcal N(0, I)$ is a valid starting point for the reverse process." },
      { t: "qa", level: "beginner", q: "What is the main practical disadvantage of diffusion models compared with GANs?", a: "Sampling speed: a GAN generates in one network evaluation, a DDPM needs hundreds to a thousand sequential evaluations (DDIM and distillation reduce this to tens or a few)." },
      { t: "qa", level: "intermediate", q: "Derive why $\\mu_\\theta = \\frac{1}{\\sqrt{\\alpha_t}}(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\epsilon_\\theta)$ is a natural parameterization.", a: "The true posterior mean $\\tilde\\mu_t(x_t, x_0)$ is linear in $x_t$ and $x_0$. Substituting $x_0 = (x_t - \\sqrt{1-\\bar\\alpha_t}\\epsilon)/\\sqrt{\\bar\\alpha_t}$ gives $\\tilde\\mu_t = \\frac{1}{\\sqrt{\\alpha_t}}(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\epsilon)$. Since $x_t$ is an input, only $\\epsilon$ is unknown — so let the network predict it. The KL term then becomes a weighted MSE between $\\epsilon$ and $\\epsilon_\\theta$." },
      { t: "qa", level: "intermediate", q: "How does $L_{\\text{simple}}$ differ from the true ELBO, and why use it?", a: "The ELBO's per-step terms are $\\frac{\\beta_t^2}{2\\sigma_t^2\\alpha_t(1-\\bar\\alpha_t)}\\|\\epsilon - \\epsilon_\\theta\\|^2$; $L_{\\text{simple}}$ drops the weight. The weight is large for small $t$ (easy denoising), so dropping it shifts emphasis to noisier steps that matter for perceptual quality. Ho et al. found better FID with $L_{\\text{simple}}$, at the cost of not optimising the likelihood bound exactly." },
      { t: "qa", level: "intermediate", q: "Why can DDIM use a model trained with the DDPM objective, and why is it deterministic?", a: "The DDPM loss depends only on the marginals $q(x_t\\mid x_0)$. DDIM defines a non-Markovian process with the same marginals, so the same $\\epsilon_\\theta$ applies. With $\\eta = 0$ its update $x_s = \\sqrt{\\bar\\alpha_s}\\hat x_0 + \\sqrt{1-\\bar\\alpha_s}\\hat\\epsilon$ adds no fresh noise, so it is a deterministic map from $x_T$ to $x_0$ (an Euler discretisation of the probability-flow ODE)." },
      { t: "qa", level: "intermediate", q: "Write the classifier-free guidance formula and explain what $w$ does.", a: "$\\tilde\\epsilon = (1+w)\\epsilon_\\theta(x_t,c) - w\\epsilon_\\theta(x_t,\\varnothing)$, equivalently $\\epsilon_\\varnothing + s(\\epsilon_c - \\epsilon_\\varnothing)$ with $s = 1+w$. It corresponds to sampling with score $\\nabla\\log p(x\\mid c) + w\\nabla\\log p(c\\mid x)$: larger $w$ pushes samples toward regions where the condition is unambiguous — better fidelity and prompt adherence, less diversity. The model is trained with the condition randomly replaced by $\\varnothing$ so it can produce both predictions." },
      { t: "qa", level: "advanced", q: "Show that the optimal noise predictor is a scaled score of the noisy marginal.", a: "The MSE-optimal predictor is $\\epsilon^*(x_t) = \\mathbb E[\\epsilon\\mid x_t]$. Since $\\nabla_{x_t}\\log q(x_t\\mid x_0) = -\\epsilon/\\sqrt{1-\\bar\\alpha_t}$ and $\\nabla\\log q_t(x_t) = \\mathbb E[\\nabla\\log q(x_t\\mid x_0)\\mid x_t]$ (differentiate $q_t = \\int q(x_t\\mid x_0)p(x_0)dx_0$ and divide by $q_t$), we get $\\nabla\\log q_t(x_t) = -\\mathbb E[\\epsilon\\mid x_t]/\\sqrt{1-\\bar\\alpha_t}$, i.e. $\\epsilon^* = -\\sqrt{1-\\bar\\alpha_t}\\nabla\\log q_t$. Plugging into $\\hat x_0$ gives Tweedie's formula $\\mathbb E[x_0\\mid x_t] = (x_t + (1-\\bar\\alpha_t)\\nabla\\log q_t(x_t))/\\sqrt{\\bar\\alpha_t}$." },
      { t: "qa", level: "advanced", q: "Why does latent diffusion reduce cost so much, and what does it give up?", a: "Each sampling step runs the U-Net on the latent (e.g. $64\\times64\\times4$ = 16,384 numbers) instead of the image ($512\\times512\\times3$ = 786,432), about 48× fewer elements, and attention cost falls even more. The autoencoder is trained once and reused. It gives up exact pixel-level control: fine detail is limited by the decoder's reconstruction quality (small text and faces were known weak points), and the likelihood is no longer of pixels." },
      { t: "qa", level: "advanced", q: "Why does classifier guidance need a classifier trained on noisy images?", a: "The gradient is taken at $x_t$, which for most of sampling is heavily noised. A classifier trained only on clean images is out of distribution there and its gradients are meaningless or adversarial. Dhariwal & Nichol train $p_\\phi(y\\mid x_t, t)$ on noised inputs at all $t$ (it shares the U-Net encoder design)." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: String.raw`With $\beta_1 = 0.1$ and $\beta_2 = 0.2$, what is $\bar\alpha_2$?`, answer: 0.72, tolerance: 0.0001, explain: String.raw`$\bar\alpha_2 = (1-0.1)(1-0.2) = 0.9 \times 0.8 = 0.72$.` },
        { kind: "numeric", q: String.raw`If $\bar\alpha_t = 0.64$, $x_0 = 2$ and $\epsilon = 0.5$ (1-D), what is $x_t$?`, answer: 1.9, tolerance: 0.0001, explain: String.raw`$x_t = \sqrt{0.64}(2) + \sqrt{0.36}(0.5) = 0.8(2) + 0.6(0.5) = 1.6 + 0.3 = 1.9$.` },
        { kind: "numeric", q: String.raw`Classifier-free guidance with $w = 3$: $\epsilon_\theta(x, c) = 0.5$, $\epsilon_\theta(x, \varnothing) = 0.2$. What is $\tilde\epsilon$?`, answer: 1.4, tolerance: 0.0001, explain: String.raw`$(1+3)(0.5) - 3(0.2) = 2.0 - 0.6 = 1.4$. In the $s$-form: $0.2 + 4(0.5-0.2) = 1.4$.` },
        { kind: "numeric", q: String.raw`At some step $\bar\alpha_t = 0.25$, $x_t = 1.0$ and the network predicts $\hat\epsilon = 0.5$ (1-D). What is $\hat x_0$?`, answer: 1.133975, tolerance: 0.001, explain: String.raw`$\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t} = (1 - \sqrt{0.75}\cdot0.5)/\sqrt{0.25} = (1 - 0.433013)/0.5 = 1.133975$.` },
        { q: "What does the network in a standard DDPM output?", options: ["The less noisy image x_{t−1}", "The noise ε contained in x_t", "The probability that x_t is real", "The time step t"], answer: 1, explain: "DDPM uses the ε-parameterization: the network predicts the noise; $x_{t-1}$ follows from a fixed formula." },
        { q: "Which statement about DDIM (η = 0) is correct?", options: ["It requires retraining the network with a new loss", "It adds more noise than DDPM at each step", "It is deterministic and can skip time steps using the same trained model", "It only works with the cosine schedule"], answer: 2, explain: "DDIM reuses the DDPM-trained ε-network (same marginals), adds no fresh noise when η = 0, and can jump along a subsequence of time steps." },
        { q: "Why is a cosine schedule often preferred to a linear one?", options: ["It makes training loss zero", "It destroys information more gradually, so fewer steps are wasted on near-pure noise", "It removes the need for a time embedding", "It makes sampling deterministic"], answer: 1, explain: "With a linear schedule $\\bar\\alpha_t$ falls to ~0 well before $T$; the cosine schedule spreads the signal-to-noise decline more evenly (Nichol & Dhariwal 2021)." },
        { q: "In Stable Diffusion, how does the text prompt influence the denoiser?", options: ["It is concatenated to the noise vector x_T only", "Through cross-attention layers whose keys and values come from text-encoder token embeddings", "By selecting one of several pretrained U-Nets", "It is not used during sampling"], answer: 1, explain: "Queries come from the U-Net's spatial features, keys and values from the (frozen CLIP) text encoder's token embeddings; classifier-free guidance amplifies the effect." },
      ] },
      { t: "exercise", title: "One full reverse step by hand", prompt: String.raw`Use the $T = 100$ linear schedule of this lesson at $t = 25$: $\beta_{25} = 0.049242$, $\bar\alpha_{25} = 0.527916$, $\bar\alpha_{24} = 0.555259$. A 1-D state $x_{25} = 0.9$ and the network predicts $\hat\epsilon = 0.3$. Compute (a) $\hat x_0$; (b) the DDPM mean $\mu$; (c) $\tilde\beta_{25}$; (d) the DDIM ($\eta = 0$) update to $t = 24$.`, hint: String.raw`$\alpha_{25} = 1 - \beta_{25}$. For (d), use $x_{24} = \sqrt{\bar\alpha_{24}}\hat x_0 + \sqrt{1-\bar\alpha_{24}}\hat\epsilon$.`, solution: [
        { t: "steps", items: [
          { title: "(a) predicted clean value", tex: String.raw`\hat x_0 = \frac{0.9 - \sqrt{0.472084}\,(0.3)}{\sqrt{0.527916}} = \frac{0.9 - 0.206125}{0.726579} = 0.954990` },
          { title: "(b) DDPM mean", tex: String.raw`\mu = \frac{1}{\sqrt{0.950758}}\Big(0.9 - \frac{0.049242}{0.687083}(0.3)\Big) = 1.025570\,(0.9 - 0.021501) = 0.900962` },
          { title: "(c) posterior variance", tex: String.raw`\tilde\beta_{25} = \frac{1-0.555259}{1-0.527916}\,(0.049242) = 0.942082 \times 0.049242 = 0.046390` },
          { title: "(d) DDIM", tex: String.raw`x_{24} = \sqrt{0.555259}\,(0.954990) + \sqrt{0.444741}\,(0.3) = 0.711617 + 0.200067 = 0.911684` },
        ] },
        { t: "p", text: "Note the DDPM mean (0.901) and the DDIM point (0.912) differ: DDPM will add noise with standard deviation $\\sqrt{0.046390} = 0.215$ around its mean, whereas DDIM keeps all of the predicted noise direction deterministically." },
      ] },
      { t: "exercise", title: "Add classifier-free guidance to the from-scratch DDPM", prompt: "Extend `ddpm_from_scratch.py` to a conditional model on the moons labels (which moon). What must change in (1) the network input, (2) the training loop, (3) the sampler? Write the changed lines.", hint: "Append a 3-way one-hot (moon 0, moon 1, null) to the network input; the first-layer weight matrix grows by 3 rows.", solution: [
        { t: "list", items: [
          "**Network input**: `sizes = [2 + EMB + 3, 64, 64, 2]` and in `forward(xt, t, c)` concatenate `np.eye(3)[c]` to `[xt, time_emb(t)]`.",
          "**Training**: keep the labels from `make_moons` (`X, y = make_moons(...)`), draw `c = y[idx]`, then `c[rng.random(BATCH) < 0.1] = 2` so 10% of examples train the unconditional (null) prediction. Pass `c` to `forward`; the backward pass is unchanged except that `h[0]` now includes the one-hot columns.",
          "**Sampler**: compute `e_c = eps_model(x, t, c)` and `e_u = eps_model(x, t, np.full(n, 2))`, then `e = (1 + w) * e_c - w * e_u` and use `e` in the DDPM or DDIM update exactly as before.",
          "**Check**: fit a k-NN classifier on the real labelled moons and measure the fraction of samples landing on the requested moon for w = 0, 1, 3 — it should rise with w, while the spread of the samples shrinks (the PyTorch tab shows real numbers for this experiment).",
        ] },
      ] },
    ],
    summary: [
      { t: "list", items: [
        String.raw`Forward process: $q(x_t\mid x_{t-1}) = \mathcal N(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$; closed form $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$.`,
        String.raw`Schedules: linear $\beta$ (Ho et al.: $10^{-4}\to0.02$, $T = 1000$) or cosine $\bar\alpha_t \propto \cos^2(\frac{t/T+s}{1+s}\frac{\pi}{2})$ (Nichol & Dhariwal), which degrades the signal more evenly.`,
        String.raw`Reverse model: $p_\theta(x_{t-1}\mid x_t) = \mathcal N(\mu_\theta, \sigma_t^2 I)$ with $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t))$.`,
        String.raw`ELBO = sum of Gaussian KLs = weighted noise-prediction MSEs; DDPM trains $L_{\text{simple}} = \mathbb E\|\epsilon - \epsilon_\theta(x_t, t)\|^2$.`,
        String.raw`DDPM sampling: $x_{t-1} = \mu_\theta + \sigma_t z$, $T$ steps. DDIM ($\eta=0$): $x_s = \sqrt{\bar\alpha_s}\hat x_0 + \sqrt{1-\bar\alpha_s}\hat\epsilon$, deterministic, any subsequence of steps.`,
        String.raw`$\epsilon^*_\theta = -\sqrt{1-\bar\alpha_t}\nabla\log q_t(x_t)$: the denoiser is a score model; $\hat x_0 = \mathbb E[x_0\mid x_t]$ (Tweedie).`,
        String.raw`Guidance: classifier $\hat\epsilon = \epsilon_\theta - s\sqrt{1-\bar\alpha_t}\nabla\log p_\phi(y\mid x_t)$; classifier-free $\tilde\epsilon = (1+w)\epsilon_\theta(x,c) - w\epsilon_\theta(x,\varnothing)$, guidance scale $s = 1+w$.`,
        "Latent diffusion: diffuse in a pretrained autoencoder's latent space; U-Net denoiser with cross-attention to text embeddings (Stable Diffusion).",
        "Strengths: stable training, coverage, quality, flexible conditioning. Weakness: many sequential network evaluations per sample.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Generative Adversarial Networks](/learn/m14/gans) — the one-step alternative, and how generative models are evaluated (FID, precision/recall), which you need to compare samplers and guidance settings properly.",
        "[Multimodal models and CLIP](/learn/m15/multimodal) — where the text embeddings that condition Stable Diffusion come from, and how joint image–text spaces are trained.",
        "[Segmentation and U-Net](/learn/m12/segmentation) — the encoder–decoder with skip connections used as the denoiser in almost every image diffusion model.",
        "[Variational Autoencoders](/learn/m14/vae) — revisit the ELBO with fresh eyes: a diffusion model is a deep hierarchical VAE with a fixed encoder.",
      ] },
    ],
  },
};

export default lesson;
