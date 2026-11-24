import type { Lesson } from "@/lib/types";

const THEORY = String.raw`import numpy as np
from scipy.linalg import sqrtm

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- 1. optimal discriminator
# p_data = N(0, 1), p_g = N(1.5, 0.8^2), evaluated on a fine grid (Riemann sum)
x = np.linspace(-10, 10, 200001)
dx = x[1] - x[0]
def normal_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
p = normal_pdf(x, 0.0, 1.0)
q = normal_pdf(x, 1.5, 0.8)

def value(D):
    """V(G, D) = E_pdata[log D(x)] + E_pg[log(1 - D(x))]"""
    D = np.clip(D, 1e-300, 1 - 1e-16)
    return np.sum(p * np.log(D) + q * np.log(1 - D)) * dx

D_star = p / (p + q)
m = 0.5 * (p + q)
kl = lambda a, b: np.sum(a * np.log(a / b)) * dx
js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
print("1) optimal discriminator")
print(f"   V(G, D*)            = {value(D_star):.6f}")
print(f"   2*JS(p, q) - log 4  = {2 * js - np.log(4):.6f}")
print(f"   JS(p, q)            = {js:.6f} nats")
for name, D in [("D = 0.5 everywhere", np.full_like(x, 0.5)),
                ("D = sigmoid(-2x+1.5)", 1 / (1 + np.exp(2 * x - 1.5))),
                ("D = 0.9*D* + 0.05", 0.9 * D_star + 0.05)]:
    print(f"   V(G, {name:21s}) = {value(D):.6f}  (< V(G, D*))")
print(f"   D*(x) at x = -1, 0.75, 2.5: {np.interp([-1, 0.75, 2.5], x, D_star)}")

# ---------------------------------------------------------------- 2. saturating vs non-saturating
print("\n2) generator gradient w.r.t. D's logit l, where D(G(z)) = sigmoid(l)")
print("   D(G(z))   l        d/dl log(1-D)   d/dl -log D   (finite-difference check)")
sig = lambda v: 1 / (1 + np.exp(-v))
for d in [0.001, 0.01, 0.1, 0.5, 0.9]:
    l = np.log(d / (1 - d))
    g_sat = -sig(l)                # minimise log(1 - sigmoid(l))
    g_ns = sig(l) - 1              # minimise -log sigmoid(l)
    h = 1e-6
    fd_sat = (np.log(1 - sig(l + h)) - np.log(1 - sig(l - h))) / (2 * h)
    fd_ns = (-np.log(sig(l + h)) + np.log(sig(l - h))) / (2 * h)
    print(f"   {d:<8} {l:7.3f}   {g_sat:12.6f}   {g_ns:12.6f}   ({fd_sat:.6f}, {fd_ns:.6f})")

# ---------------------------------------------------------------- 3. the bilinear game min_x max_y x*y
print("\n3) bilinear game f(x, y) = x*y, start (1, 0), step eta = 0.1")
eta, T = 0.1, 1000
def run(update):
    z = np.array([1.0, 0.0]); out = {}
    for t in range(T + 1):
        if t in (0, 10, 100, 1000):
            out[t] = np.linalg.norm(z)
        z = update(z)
    return out
simult = lambda z: np.array([z[0] - eta * z[1], z[1] + eta * z[0]])           # both use the old values
def altern(z):
    x = z[0] - eta * z[1]                                                       # x moves first ...
    return np.array([x, z[1] + eta * x])                                        # ... y sees the new x
def extragrad(z):
    zh = np.array([z[0] - eta * z[1], z[1] + eta * z[0]])                       # look-ahead step
    return np.array([z[0] - eta * zh[1], z[1] + eta * zh[0]])                   # update with look-ahead gradient
for name, upd in [("simultaneous GDA", simult), ("alternating GDA", altern), ("extragradient", extragrad)]:
    r = run(upd)
    print(f"   {name:17s} |z| at t=0,10,100,1000: " + ", ".join(f"{r[t]:.4f}" for t in r))
print(f"   predicted simultaneous growth sqrt(1+eta^2)^1000 = {np.sqrt(1 + eta**2) ** 1000:.4f}")
# continuous-time flow dx/dt = -y, dy/dt = x: exact solution is a rotation, |z| stays 1
t = 10.0
print(f"   continuous flow at t=10: (x, y) = ({np.cos(t):.4f}, {np.sin(t):.4f}), |z| = {np.hypot(np.cos(t), np.sin(t)):.4f}")

# ---------------------------------------------------------------- 4. Frechet distance between Gaussians
def frechet(mu1, S1, mu2, S2):
    """||mu1-mu2||^2 + Tr(S1 + S2 - 2 (S1^1/2 S2 S1^1/2)^1/2), using only symmetric eigendecompositions."""
    def psd_sqrt(S):
        w, V = np.linalg.eigh(S)
        return (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    r1 = psd_sqrt(S1)
    cross = psd_sqrt(r1 @ S2 @ r1)                       # symmetric PSD, so eigh is safe
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * np.trace(cross))

print("\n4) Frechet distance between Gaussians (FID formula)")
I2 = np.eye(2)
print(f"   identical                     : {frechet(np.zeros(2), I2, np.zeros(2), I2):.6f}   (expected 0)")
print(f"   mean shift (3,4), Sigma = I   : {frechet(np.zeros(2), I2, np.array([3., 4.]), I2):.6f}  (expected 25)")
print(f"   1-D, sigma 1 vs sigma 2       : {frechet(np.zeros(1), np.eye(1), np.zeros(1), 4 * np.eye(1)):.6f}   (expected (1-2)^2 = 1)")
print(f"   diag(1,4) vs diag(4,9)        : {frechet(np.zeros(2), np.diag([1., 4.]), np.zeros(2), np.diag([4., 9.])):.6f}   (expected 1 + 1 = 2)")
rng = np.random.default_rng(0)
A = rng.normal(size=(5, 5)); B = rng.normal(size=(5, 5))
S1, S2 = A @ A.T + 0.1 * np.eye(5), B @ B.T + 0.1 * np.eye(5)
mu1, mu2 = rng.normal(size=5), rng.normal(size=5)
ref = np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * np.real(sqrtm(S1 @ S2)))
print(f"   random 5-D, ours vs scipy     : {frechet(mu1, S1, mu2, S2):.6f} vs {ref:.6f}")
# FID is estimated from samples: even two samples of the SAME Gaussian give a positive value
d = 16
for n in (100, 1000, 10000):
    X1, X2 = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    f = frechet(X1.mean(0), np.cov(X1, rowvar=False), X2.mean(0), np.cov(X2, rowvar=False))
    print(f"   two samples of N(0, I_16), n = {n:5d}: {f:.4f}")`;
const THEORY_OUT = String.raw`1) optimal discriminator
   V(G, D*)            = -0.861829
   2*JS(p, q) - log 4  = -0.861829
   JS(p, q)            = 0.262233 nats
   V(G, D = 0.5 everywhere   ) = -1.386294  (< V(G, D*))
   V(G, D = sigmoid(-2x+1.5) ) = -0.871351  (< V(G, D*))
   V(G, D = 0.9*D* + 0.05    ) = -0.877451  (< V(G, D*))
   D*(x) at x = -1, 0.75, 2.5: [0.9846 0.4838 0.0713]

2) generator gradient w.r.t. D's logit l, where D(G(z)) = sigmoid(l)
   D(G(z))   l        d/dl log(1-D)   d/dl -log D   (finite-difference check)
   0.001     -6.907      -0.001000      -0.999000   (-0.001000, -0.999000)
   0.01      -4.595      -0.010000      -0.990000   (-0.010000, -0.990000)
   0.1       -2.197      -0.100000      -0.900000   (-0.100000, -0.900000)
   0.5        0.000      -0.500000      -0.500000   (-0.500000, -0.500000)
   0.9        2.197      -0.900000      -0.100000   (-0.900000, -0.100000)

3) bilinear game f(x, y) = x*y, start (1, 0), step eta = 0.1
   simultaneous GDA  |z| at t=0,10,100,1000: 1.0000, 1.0510, 1.6446, 144.7728
   alternating GDA   |z| at t=0,10,100,1000: 1.0000, 1.0242, 1.0234, 0.9796
   extragradient     |z| at t=0,10,100,1000: 1.0000, 0.9515, 0.6081, 0.0069
   predicted simultaneous growth sqrt(1+eta^2)^1000 = 144.7728
   continuous flow at t=10: (x, y) = (-0.8391, -0.5440), |z| = 1.0000

4) Frechet distance between Gaussians (FID formula)
   identical                     : 0.000000   (expected 0)
   mean shift (3,4), Sigma = I   : 25.000000  (expected 25)
   1-D, sigma 1 vs sigma 2       : 1.000000   (expected (1-2)^2 = 1)
   diag(1,4) vs diag(4,9)        : 2.000000   (expected 1 + 1 = 2)
   random 5-D, ours vs scipy     : 18.093534 vs 18.093534
   two samples of N(0, I_16), n =   100: 1.4238
   two samples of N(0, I_16), n =  1000: 0.1775
   two samples of N(0, I_16), n = 10000: 0.0162`;
const GAN_NP = String.raw`import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # tiny matrices: extra BLAS threads only add overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

rng = np.random.default_rng(0)

# ---------------- data: a ring of 8 Gaussians (radius 2, std 0.05) -- the simulator's default target
N_MODES, RADIUS, STD = 8, 2.0, 0.05
CENTERS = RADIUS * np.stack([np.cos(2 * np.pi * np.arange(N_MODES) / N_MODES),
                             np.sin(2 * np.pi * np.arange(N_MODES) / N_MODES)], axis=1)

def sample_real(n):
    return CENTERS[rng.integers(0, N_MODES, n)] + STD * rng.normal(size=(n, 2))


class MLP:
    """Leaky-ReLU MLP with a linear output; forward caches what backward needs."""

    def __init__(self, sizes, slope=0.2):
        self.slope = slope
        self.W = [rng.uniform(-1, 1, (i, o)) / np.sqrt(i) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]
        # Adam state
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        self.cache = [x]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.cache[-1] @ W + b
            if l < len(self.W) - 1:
                self.cache.append(z)                                # pre-activation (for the slope)
                self.cache.append(np.where(z > 0, z, self.slope * z))
            else:
                self.cache.append(z)
        return self.cache[-1]

    def backward(self, dout):
        """Gradients of sum(dout * output) w.r.t. parameters and input. Returns (grads, d_input)."""
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        delta = dout
        for l in reversed(range(len(self.W))):
            a_in = self.cache[2 * l]                                # activation entering layer l
            gW[l] = a_in.T @ delta
            gb[l] = delta.sum(0)
            delta = delta @ self.W[l].T
            if l > 0:
                delta = delta * np.where(self.cache[2 * l - 1] > 0, 1.0, self.slope)
        return gW + gb, delta

    def adam(self, grads, lr, b1=0.5, b2=0.9, eps=1e-8):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.W + self.b, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            p -= lr * (self.m[i] / (1 - b1 ** self.t)) / (np.sqrt(self.v[i] / (1 - b2 ** self.t)) + eps)


sigmoid = lambda v: 1 / (1 + np.exp(-v))
softplus = lambda v: np.logaddexp(0, v)                             # log(1 + e^v), overflow-safe


def coverage(x):
    """Modes holding >= 25% of their fair share of samples within 3 std, and the fraction of such samples."""
    d = np.linalg.norm(x[:, None, :] - CENTERS[None], axis=2)
    near, k = d.min(1) <= 3 * STD, d.argmin(1)
    counts = np.bincount(k[near], minlength=N_MODES)
    return int((counts >= 0.25 * len(x) / N_MODES).sum()), near.mean(), counts


Z_DIM, H, BATCH, LR, ITERS = 8, 32, 128, 1e-3, 5000
G = MLP([Z_DIM, H, H, 2])
D = MLP([2, H, H, 1])
z_eval = rng.normal(size=(1000, Z_DIM))                              # fixed noise to track progress

for it in range(1, ITERS + 1):
    # ---- discriminator step: minimise -log D(x) - log(1 - D(G(z)))
    x_real, x_fake = sample_real(BATCH), G.forward(rng.normal(size=(BATCH, Z_DIM)))
    l_real = D.forward(x_real); g_real, _ = D.backward((sigmoid(l_real) - 1) / BATCH)
    l_fake = D.forward(x_fake); g_fake, _ = D.backward(sigmoid(l_fake) / BATCH)
    d_loss = np.mean(softplus(-l_real)) + np.mean(softplus(l_fake))
    D.adam([a + b for a, b in zip(g_real, g_fake)], LR)

    # ---- generator step (non-saturating): minimise -log D(G(z)); D is held fixed
    x_fake = G.forward(rng.normal(size=(BATCH, Z_DIM)))
    l_fake = D.forward(x_fake)
    _, dx = D.backward((sigmoid(l_fake) - 1) / BATCH)                 # dL_G/dx: only the input gradient is used
    g_grads, _ = G.backward(dx)                                      # chain rule continues into G
    G.adam(g_grads, LR)
    g_loss = np.mean(softplus(-l_fake))

    if it % 500 == 0:
        x_gen = G.forward(z_eval)
        modes, hq, counts = coverage(x_gen)
        d_fake = sigmoid(D.forward(x_gen)).mean(); d_real = sigmoid(D.forward(sample_real(1000))).mean()
        print(f"iter {it:4d}  D loss {d_loss:.3f}  G loss {g_loss:.3f}  D(real) {d_real:.2f}  "
              f"D(fake) {d_fake:.2f}  modes {modes}/8  high-quality {hq:.1%}")
print("samples per mode:", counts.tolist())`;
const GAN_NP_OUT = String.raw`iter  500  D loss 1.358  G loss 0.716  D(real) 0.51  D(fake) 0.49  modes 0/8  high-quality 2.3%
iter 1000  D loss 1.251  G loss 0.801  D(real) 0.53  D(fake) 0.45  modes 0/8  high-quality 2.8%
iter 1500  D loss 1.060  G loss 1.002  D(real) 0.60  D(fake) 0.40  modes 0/8  high-quality 3.5%
iter 2000  D loss 0.933  G loss 1.178  D(real) 0.64  D(fake) 0.36  modes 0/8  high-quality 7.3%
iter 2500  D loss 0.965  G loss 1.110  D(real) 0.63  D(fake) 0.36  modes 4/8  high-quality 30.7%
iter 3000  D loss 1.058  G loss 1.095  D(real) 0.65  D(fake) 0.40  modes 4/8  high-quality 48.2%
iter 3500  D loss 1.051  G loss 1.118  D(real) 0.62  D(fake) 0.39  modes 5/8  high-quality 60.2%
iter 4000  D loss 1.082  G loss 1.113  D(real) 0.62  D(fake) 0.39  modes 6/8  high-quality 68.2%
iter 4500  D loss 1.077  G loss 1.122  D(real) 0.60  D(fake) 0.39  modes 6/8  high-quality 70.3%
iter 5000  D loss 1.133  G loss 1.167  D(real) 0.59  D(fake) 0.39  modes 6/8  high-quality 75.5%
samples per mode: [50, 160, 6, 161, 82, 165, 0, 131]`;
const LOSSES_PT = String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)                      # single thread: bit-for-bit reproducible on CPU

# ring of 8 Gaussians, radius 2, std 0.05 (the simulator's default target)
N_MODES, RADIUS, STD, Z_DIM, H = 8, 2.0, 0.05, 8, 64
ang = torch.arange(N_MODES) * 2 * math.pi / N_MODES
CENTERS = RADIUS * torch.stack([ang.cos(), ang.sin()], 1)


def sample_real(n, gen):
    k = torch.randint(0, N_MODES, (n,), generator=gen)
    return CENTERS[k] + STD * torch.randn(n, 2, generator=gen)


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, H), nn.LeakyReLU(0.2), nn.Linear(H, H), nn.LeakyReLU(0.2), nn.Linear(H, n_out))


def coverage(x):
    """(modes with >= 25% of their fair share of samples within 3 std, fraction of samples within 3 std of a mode)"""
    d = torch.cdist(x, CENTERS)
    near, k = d.min(1).values <= 3 * STD, d.argmin(1)
    counts = torch.bincount(k[near], minlength=N_MODES)
    return int((counts >= 0.25 * len(x) / N_MODES).sum()), near.float().mean().item()


def train(loss, lr_g, lr_d, iters=3000, batch=128, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    G, D = mlp(Z_DIM, 2), mlp(2, 1)
    opt_g = torch.optim.Adam(G.parameters(), lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr_d, betas=(0.5, 0.9))
    z_eval = torch.randn(1000, Z_DIM, generator=gen)          # fixed noise for evaluation
    log = []
    for it in range(1, iters + 1):
        # ---- D step: minimise -log D(x) - log(1 - D(G(z)));  softplus(-l) = -log sigmoid(l)
        real = sample_real(batch, gen)
        fake = G(torch.randn(batch, Z_DIM, generator=gen)).detach()   # detach: no gradient into G here
        d_loss = F.softplus(-D(real)).mean() + F.softplus(D(fake)).mean()
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        # ---- G step
        l_fake = D(G(torch.randn(batch, Z_DIM, generator=gen)))
        if loss == "minimax":
            g_loss = -F.softplus(l_fake).mean()      # = mean log(1 - D(G(z))), minimised by G
        else:
            g_loss = F.softplus(-l_fake).mean()      # = mean -log D(G(z))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()   # D's stray grads are zeroed before its next step
        if it % 1000 == 0:
            with torch.no_grad():
                log.append(coverage(G(z_eval)))
    return log


for title, lr_g, lr_d in [("balanced: lr_G = lr_D = 1e-3", 1e-3, 1e-3), ("strong D: lr_G = 3e-4, lr_D = 3e-3", 3e-4, 3e-3)]:
    print(title)
    for loss in ("minimax", "non-saturating"):
        log = train(loss, lr_g, lr_d)
        print(f"  {loss:15s}" + "".join(f" | it {1000 * (i + 1)}: {m}/8, {hq:5.1%} HQ" for i, (m, hq) in enumerate(log)))`;
const LOSSES_PT_OUT = String.raw`balanced: lr_G = lr_D = 1e-3
  minimax         | it 1000: 2/8, 22.2% HQ | it 2000: 4/8, 45.9% HQ | it 3000: 6/8, 58.1% HQ
  non-saturating  | it 1000: 3/8, 21.4% HQ | it 2000: 5/8, 49.8% HQ | it 3000: 6/8, 66.4% HQ
strong D: lr_G = 3e-4, lr_D = 3e-3
  minimax         | it 1000: 0/8,  3.4% HQ | it 2000: 0/8,  2.1% HQ | it 3000: 0/8,  2.6% HQ
  non-saturating  | it 1000: 2/8, 11.3% HQ | it 2000: 3/8, 18.5% HQ | it 3000: 3/8, 20.0% HQ`;
const WGANGP_PT = String.raw`import math
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)

N_MODES, RADIUS, STD, Z_DIM, H = 8, 2.0, 0.05, 8, 64
ang = torch.arange(N_MODES) * 2 * math.pi / N_MODES
CENTERS = RADIUS * torch.stack([ang.cos(), ang.sin()], 1)
sample_real = lambda n: CENTERS[torch.randint(0, N_MODES, (n,), generator=gen)] + STD * torch.randn(n, 2, generator=gen)
mlp = lambda i, o: nn.Sequential(nn.Linear(i, H), nn.LeakyReLU(0.2), nn.Linear(H, H), nn.LeakyReLU(0.2), nn.Linear(H, o))


def gradient_penalty(critic, real, fake):
    """E[(||grad_x f(x_hat)||_2 - 1)^2] at random points on lines between real and fake samples."""
    eps = torch.rand(real.size(0), 1, generator=gen)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)   # create_graph: the penalty
    return ((grad.norm(dim=1) - 1) ** 2).mean()                                   # is itself differentiated


G, f = mlp(Z_DIM, 2), mlp(2, 1)                  # f is a critic: unbounded real output, no sigmoid
opt_g = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.9))
opt_f = torch.optim.Adam(f.parameters(), 1e-3, betas=(0.5, 0.9))
N_CRITIC, LAMBDA, BATCH = 3, 0.1, 128
z_eval = torch.randn(1000, Z_DIM, generator=gen)

for it in range(1, 3001):
    for _ in range(N_CRITIC):                    # the critic is trained more than G (it should stay near-optimal)
        real = sample_real(BATCH)
        fake = G(torch.randn(BATCH, Z_DIM, generator=gen)).detach()
        w_est = f(real).mean() - f(fake).mean()   # critic's estimate of the Wasserstein-1 distance
        gp = gradient_penalty(f, real, fake)
        loss_f = -w_est + LAMBDA * gp
        opt_f.zero_grad(); loss_f.backward(); opt_f.step()
    loss_g = -f(G(torch.randn(BATCH, Z_DIM, generator=gen))).mean()
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    if it % 500 == 0:
        with torch.no_grad():
            x = G(z_eval)
            d = torch.cdist(x, CENTERS)
            near, k = d.min(1).values <= 3 * STD, d.argmin(1)
            counts = torch.bincount(k[near], minlength=N_MODES)
            modes = int((counts >= 0.25 * 1000 / N_MODES).sum())
        print(f"iter {it:4d}  W-estimate {w_est.item():.3f}  GP {gp.item():.3f}  modes {modes}/8  HQ {near.float().mean().item():.1%}")
print("samples per mode:", counts.tolist())`;
const WGANGP_PT_OUT = String.raw`iter  500  W-estimate 2.180  GP 6.695  modes 0/8  HQ 3.8%
iter 1000  W-estimate 1.017  GP 3.114  modes 0/8  HQ 16.4%
iter 1500  W-estimate 0.830  GP 2.911  modes 5/8  HQ 23.8%
iter 2000  W-estimate 0.665  GP 2.630  modes 5/8  HQ 35.8%
iter 2500  W-estimate 0.578  GP 1.878  modes 8/8  HQ 44.0%
iter 3000  W-estimate 0.521  GP 1.458  modes 8/8  HQ 56.1%
samples per mode: [76, 78, 75, 73, 82, 64, 41, 72]`;
const CGAN_PT = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(1)
torch.manual_seed(0)

# ---------------- data: 1797 8x8 digit images (bundled with scikit-learn), pixels 0..16 -> [-1, 1]
digits = load_digits()
X = torch.tensor(digits.images / 8.0 - 1.0, dtype=torch.float32).unsqueeze(1)   # (N, 1, 8, 8)
y = torch.tensor(digits.target)
Z_DIM, N_CLS = 16, 10


class Generator(nn.Module):
    """DCGAN-style: project (z, one-hot label) to 64x2x2, then two stride-2 transposed convs -> 8x8."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(Z_DIM + N_CLS, 64 * 2 * 2)
        self.net = nn.Sequential(
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # 2x2 -> 4x4
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh())                        # 4x4 -> 8x8

    def forward(self, z, labels):
        h = self.fc(torch.cat([z, F.one_hot(labels, N_CLS).float()], 1))
        return self.net(h.view(-1, 64, 2, 2))


class Discriminator(nn.Module):
    """Strided convs + LeakyReLU, no pooling; the label enters as 10 constant one-hot channels."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1 + N_CLS, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),                  # 8x8 -> 4x4
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),      # 4x4 -> 2x2
            nn.Flatten(), nn.Linear(64 * 2 * 2, 1))

    def forward(self, x, labels):
        maps = F.one_hot(labels, N_CLS).float()[:, :, None, None].expand(-1, -1, 8, 8)
        return self.net(torch.cat([x, maps], 1))


G, D = Generator(), Discriminator()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999))
BATCH, ITERS = 64, 1000

for it in range(1, ITERS + 1):
    idx = torch.randint(0, len(X), (BATCH,))
    real, lab = X[idx], y[idx]
    fake_lab = torch.randint(0, N_CLS, (BATCH,))
    fake = G(torch.randn(BATCH, Z_DIM), fake_lab)
    # discriminator: real (image, label) pairs -> 1, generated pairs -> 0
    d_loss = F.binary_cross_entropy_with_logits(D(real, lab), torch.ones(BATCH, 1)) + \
             F.binary_cross_entropy_with_logits(D(fake.detach(), fake_lab), torch.zeros(BATCH, 1))
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    # generator: non-saturating loss, make D call the generated pairs real
    g_loss = F.binary_cross_entropy_with_logits(D(fake, fake_lab), torch.ones(BATCH, 1))
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    if it % 500 == 0:
        print(f"iter {it}  D loss {d_loss.item():.3f}  G loss {g_loss.item():.3f}")

# ---------------- evaluation
G.eval()
with torch.no_grad():
    gen_lab = torch.arange(N_CLS).repeat_interleave(100)                  # 100 samples per class
    gen = G(torch.randn(len(gen_lab), Z_DIM), gen_lab)
flat_real = X.view(len(X), -1).numpy()
flat_gen = gen.view(len(gen), -1).numpy()

# 1) does the generated image show the requested digit? judge with a classifier trained on REAL digits only
perm = np.random.default_rng(0).permutation(len(X))
tr, te = perm[:1200], perm[1200:]
clf = LogisticRegression(max_iter=2000).fit(flat_real[tr], digits.target[tr])
print(f"classifier accuracy on held-out real digits: {clf.score(flat_real[te], digits.target[te]):.3f}")
pred = clf.predict(flat_gen)
per_class = [(pred[gen_lab.numpy() == c] == c).mean() for c in range(N_CLS)]
print("generated digits classified as the requested label:", f"{np.mean(per_class):.3f}")
print("  per class:", " ".join(f"{c}:{a:.2f}" for c, a in enumerate(per_class)))

# 2) Frechet distance in a 20-D PCA feature space (an "FID" with PCA features instead of Inception features)
def frechet(a, b):
    mu1, mu2, S1, S2 = a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, V = np.linalg.eigh(S1); r1 = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    w2 = np.linalg.eigvalsh(r1 @ S2 @ r1)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * np.sqrt(np.clip(w2, 0, None)).sum())

pca = PCA(20, random_state=0).fit(flat_real[tr])
f = lambda a: pca.transform(a)
noise = np.random.default_rng(1).uniform(-1, 1, (597, 64))
print(f"FD(real train, real test) = {frechet(f(flat_real[tr]), f(flat_real[te])):.2f}   (reference floor)")
print(f"FD(real train, generated) = {frechet(f(flat_real[tr]), f(flat_gen[:597])):.2f}")
print(f"FD(real train, uniform noise) = {frechet(f(flat_real[tr]), f(noise)):.2f}")

# 3) one generated sample per class as ASCII art (dark = high ink)
shades = " .:-=+*#%@"
imgs = ((gen.view(N_CLS, 100, 8, 8)[:, 0].numpy() + 1) / 2 * 9.99).clip(0, 9).astype(int)
for r in range(8):
    print("  ".join("".join(shades[v] for v in imgs[c, r]) for c in range(N_CLS)))`;
const CGAN_PT_OUT = String.raw`iter 500  D loss 1.301  G loss 1.089
iter 1000  D loss 1.153  G loss 1.069
classifier accuracy on held-out real digits: 0.975
generated digits classified as the requested label: 0.947
  per class: 0:0.99 1:0.94 2:0.93 3:0.91 4:0.98 5:0.98 6:0.98 7:1.00 8:0.77 9:0.99
FD(real train, real test) = 0.14   (reference floor)
FD(real train, generated) = 2.58
FD(real train, uniform noise) = 10.51
  .@+       =@%       =@@.      .*%@.      **       %@@@#      =-.      +@@+       *@=      -@#
  %@..       @%      *@*%#      .=*%.      @-       %@..%     .%.       -%+%.     +=%@     .%#=+
 :@- +:      @@      #@ %%       ==%:     @% :      @%        #%           @      %@-@     .#: #-
 :#  +%     =@#      .  @*       -%*     *@ :%.    .@%       :@%%%-      +@@=     -%@@      #%=%-
 -*  +*     -@-         @.        %#     +*-@%:    .%@@      :@@@@#     -@@--     :@@+       . +=
  *  =:     -@=        +%         *%.     :@@=       +@       #. -+      @@       @@=%         +.
  %*.@.     *@.       @@@*#      :#%.      %@       -@@       *@ ##      @        @%:@.     -. -
  -@%       #%.       *@*%-.    .#@*.      =+      .%@         #@%      -@         -@%      +%+`;
const WGANGP_TF = String.raw`import tensorflow as tf

LAMBDA, N_CRITIC, Z_DIM = 0.1, 3, 8

def mlp(n_out):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64), tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(64), tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(n_out)])

G, critic = mlp(2), mlp(1)
G.build((None, Z_DIM)); critic.build((None, 2))
opt_g = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)
opt_c = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)

def sample_real(n):
    k = tf.random.uniform((n,), 0, 8, dtype=tf.int32)
    ang = tf.cast(k, tf.float32) * (2 * 3.14159265 / 8)
    return 2.0 * tf.stack([tf.cos(ang), tf.sin(ang)], 1) + 0.05 * tf.random.normal((n, 2))

@tf.function
def critic_step(real):
    fake = G(tf.random.normal((tf.shape(real)[0], Z_DIM)))
    eps = tf.random.uniform((tf.shape(real)[0], 1))
    x_hat = eps * real + (1 - eps) * fake
    with tf.GradientTape() as tape:
        with tf.GradientTape() as inner:          # inner tape: gradient of f w.r.t. its INPUT
            inner.watch(x_hat)
            f_hat = critic(x_hat)
        g = inner.gradient(f_hat, x_hat)
        gp = tf.reduce_mean((tf.norm(g, axis=1) - 1.0) ** 2)
        loss = tf.reduce_mean(critic(fake)) - tf.reduce_mean(critic(real)) + LAMBDA * gp
    opt_c.apply_gradients(zip(tape.gradient(loss, critic.trainable_variables), critic.trainable_variables))
    return loss

@tf.function
def generator_step(n):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(critic(G(tf.random.normal((n, Z_DIM)))))
    opt_g.apply_gradients(zip(tape.gradient(loss, G.trainable_variables), G.trainable_variables))
    return loss

for it in range(1, 3001):
    for _ in range(N_CRITIC):
        lc = critic_step(sample_real(128))
    lg = generator_step(128)
    if it % 500 == 0:
        print(f"iter {it}  critic loss {float(lc):.3f}  G loss {float(lg):.3f}")`;

const lesson: Lesson = {
  id: "m14/gans",
  title: "Generative Adversarial Networks",
  summary:
    "A generator and a discriminator trained against each other: the minimax objective, the optimal discriminator and the Jensen–Shannon divergence, saturating vs non-saturating losses, why training oscillates and collapses, DCGAN, conditional GANs, Pix2Pix, CycleGAN, WGAN and WGAN-GP, StyleGAN and FID — with a GAN trained live in your browser and every number from real runs.",
  minutes: 110,
  level: "advanced",
  prerequisites: ["m14/autoencoders", "m14/vae", "m02/information-theory", "m04/backpropagation", "m06/momentum-adaptive"],
  next: ["m14/diffusion", "m15/self-supervised", "m17/experimentation"],
  terms: ["gan", "generator", "discriminator", "minimax-objective", "mode-collapse", "non-saturating-loss", "wasserstein-distance", "gradient-penalty", "conditional-gan", "cycle-consistency", "fid"],
  sections: {
    intro: [
      { t: "p", text: "A **generative model** learns a probability distribution from examples so that it can produce new samples that look like they came from the same source: new faces, new handwritten digits, new textures. A [VAE](/learn/m14/vae) does this by maximising a lower bound on the likelihood. A **Generative Adversarial Network (GAN)**, introduced by Goodfellow et al. in 2014 (\"Generative Adversarial Nets\"), takes a completely different route: it never writes down a likelihood at all." },
      { t: "p", text: "A GAN has two networks. The **generator** $G$ turns random noise $z$ into a sample $G(z)$. The **discriminator** $D$ looks at a sample and outputs the probability that it is real (from the data) rather than generated. They are trained as opponents: $D$ is rewarded for telling real from fake, $G$ is rewarded for fooling $D$. The only training signal $G$ ever receives is the gradient of $D$'s judgement — $D$ is a *learned loss function* for $G$." },
      { t: "p", text: "Why was this attractive? Likelihood-based models of the time either needed tractable densities (restricting the architecture) or produced blurry samples because a per-pixel likelihood rewards averaging. A discriminator can instead penalise anything that *looks* wrong, so GANs produced much sharper images. From DCGAN (2015) to StyleGAN (2019) they defined the state of the art in image synthesis, and adversarial losses remain inside super-resolution, image-to-image translation and the decoders of modern latent diffusion models." },
      { t: "p", text: "The price is that training is a two-player game rather than a minimisation, and games do not have to converge. Most of this lesson is about understanding *why* GAN training fails — mode collapse, oscillation, vanishing gradients — and the fixes that made GANs practical: the non-saturating loss, DCGAN's architecture rules, the Wasserstein distance and the gradient penalty." },
      { t: "callout", kind: "key", text: "A GAN trains a sampler $G$ without any density: a discriminator $D$ estimates how the generated distribution differs from the data, and $G$ follows $D$'s gradient. With an optimal $D$ the original objective equals $2\\,\\mathrm{JS}(p_{\\text{data}}\\,\\|\\,p_g) - \\log 4$." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Information theory](/learn/m02/information-theory) — KL divergence and especially the **Jensen–Shannon divergence**, which is what the original GAN objective minimises.",
        "[Probability](/learn/m02/probability) — densities, expectations $\\mathbb{E}_{x\\sim p}[f(x)]$, and sampling by transforming noise (a pushforward distribution).",
        "[Classification losses](/learn/m05/classification-losses) — binary cross-entropy with a sigmoid output; the discriminator is exactly a binary classifier, and $\\log(1+e^{l})$ (softplus) appears everywhere.",
        "[Backpropagation](/learn/m04/backpropagation) — the generator is trained by back-propagating *through* the discriminator into the generator.",
        "[Momentum and Adam](/learn/m06/momentum-adaptive) — every GAN here is trained with Adam; the $\\beta_1 = 0.5$ convention matters.",
        "[Autoencoders](/learn/m14/autoencoders) and [VAEs](/learn/m14/vae) — the other family of latent-variable generators; we compare them at the end.",
        "[Convolution variants](/learn/m08/conv-variants) — **transposed convolutions**, which DCGAN generators use to upsample, and [Normalization](/learn/m07/normalization) — batch norm in DCGAN, instance norm in Pix2Pix/CycleGAN.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "The forger and the art expert", text: "A forger (generator) paints fake masterpieces; an expert (discriminator) inspects paintings and says \"genuine\" or \"forgery\". Each time the expert catches a forgery, the forger learns what gave it away — the brushstrokes, the canvas texture — and improves. Each time the forger improves, the expert must look more closely. If both keep improving, the forgeries eventually become indistinguishable from real paintings and the expert can do no better than a coin flip. The forger never sees a textbook on painting; everything it learns comes from the expert's verdicts." },
      { t: "p", text: "Two details of the analogy carry over exactly. First, the forger does not just learn *whether* it was caught but *how to change* the painting to look more genuine: in a GAN, $G$ receives the gradient $\\nabla_x D(x)$ at its samples — a direction in sample space that would make $D$ more convinced. Second, the expert's verdict is relative: it can only say \"this looks different from the paintings I know\". If the forger discovers one style the expert currently accepts, it can paint *only* that style. That is **mode collapse**: the forgeries are individually convincing but have no variety." },
      { t: "p", text: "In the simulator below you can watch this happen in two dimensions. The background colour is $D(x)$ — green where $D$ believes points are real, clay where it believes they are fake — and arrows show where $G$ is pushing its samples. Real data are eight small clusters on a ring; a good generator must put samples on all eight." },
    ],
    mechanism: [
      { t: "steps", items: [
        { title: "Define the two networks", text: "Generator $G_\\theta: \\mathbb{R}^{d_z}\\to\\mathcal{X}$ maps noise $z\\sim p_z$ (usually $\\mathcal{N}(0, I)$) to a sample. Discriminator $D_\\phi: \\mathcal{X}\\to(0,1)$ outputs $D(x) = \\sigma(l_\\phi(x))$ where $l_\\phi$ is a logit. The generated distribution $p_g$ is the distribution of $G(z)$ — it has no formula, only a sampler." },
        { title: "Sample a mini-batch of each kind", text: "$m$ real examples $x_1..x_m$ from the training set and $m$ noise vectors $z_1..z_m$, giving fakes $\\tilde x_i = G(z_i)$." },
        { title: "Discriminator step (ascend V)", text: "Treat real as label 1 and fake as label 0, compute binary cross-entropy, and take an optimizer step on $\\phi$ only. Gradients must **not** flow into $G$ here (detach the fakes).", tex: String.raw`\mathcal{L}_D = -\frac{1}{m}\sum_{i=1}^m \Big[\log D(x_i) + \log\big(1 - D(G(z_i))\big)\Big]` },
        { title: "Optionally repeat k times", text: "The original algorithm allows $k$ discriminator steps per generator step (WGAN uses $k = 5$) so $D$ stays close to its optimum for the current $G$." },
        { title: "Generator step", text: "Draw fresh noise, generate, pass through the **current, frozen** $D$, and back-propagate the generator loss through $D$ (to the sample) and on through $G$ (to $\\theta$). Only $\\theta$ is updated. In practice the non-saturating loss is used:", tex: String.raw`\mathcal{L}_G^{\text{NS}} = -\frac{1}{m}\sum_{i=1}^m \log D(G(z_i))` },
        { title: "Alternate until a budget is spent", text: "There is no loss that should go to zero. Monitor sample quality and diversity (mode counts, FID) instead; the discriminator loss hovers around $2\\log 2 \\approx 1.386$ when $D$ is maximally confused." },
        { title: "Sample", text: "After training, discard $D$. New samples are $G(z)$ for fresh $z\\sim p_z$ — one forward pass, no iterative refinement." },
      ] },
      { t: "callout", kind: "warning", text: "The generator gradient passes through $D$: $\\nabla_\\theta\\mathcal{L}_G = \\sum_i \\big(\\partial\\mathcal{L}_G/\\partial \\tilde x_i\\big)\\,\\partial G(z_i)/\\partial\\theta$, where $\\partial\\mathcal{L}_G/\\partial \\tilde x_i$ is computed by backpropagation through $D$ with $D$'s weights fixed. Forgetting to freeze $D$ (or to zero its gradients before its own next step) silently trains $D$ to *help* $G$." },
    ],
    math: [
      { t: "h", text: "The minimax objective" },
      { t: "p", text: String.raw`Goodfellow et al. (2014) defined the value function` },
      { t: "math", tex: String.raw`\min_G \max_D\; V(D, G) = \mathbb{E}_{x\sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log\big(1 - D(G(z))\big)\big]` },
      { t: "p", text: String.raw`$p_{\text{data}}$ is the data distribution, $p_z$ the noise prior, $D(x)\in(0,1)$ the probability $D$ assigns to "real". For fixed $G$, maximising $V$ over $D$ is exactly minimising $D$'s binary cross-entropy. Writing the second expectation over $x = G(z)\sim p_g$:` },
      { t: "math", tex: String.raw`V(D, G) = \int_x \Big[p_{\text{data}}(x)\log D(x) + p_g(x)\log\big(1 - D(x)\big)\Big]\,dx` },
      { t: "h", text: "The optimal discriminator" },
      { t: "p", text: String.raw`The integrand can be maximised pointwise. For fixed $a = p_{\text{data}}(x)$ and $b = p_g(x)$, the function $y\mapsto a\log y + b\log(1-y)$ on $(0,1)$ has derivative $a/y - b/(1-y)$, which is zero at $y = a/(a+b)$; the second derivative $-a/y^2 - b/(1-y)^2 < 0$ confirms a maximum. Hence` },
      { t: "math", tex: String.raw`D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}` },
      { t: "p", text: String.raw`The optimal discriminator is a **density-ratio estimator**: $D^*/(1-D^*) = p_{\text{data}}/p_g$. Where the generator puts too little mass, $D^* > 1/2$; where it puts too much, $D^* < 1/2$; when $p_g = p_{\text{data}}$, $D^* = 1/2$ everywhere.` },
      { t: "h", text: "The resulting divergence: Jensen–Shannon" },
      { t: "p", text: String.raw`Substitute $D^*$ and let $m = \tfrac12(p_{\text{data}} + p_g)$. Since $D^* = p_{\text{data}}/(2m)$ and $1 - D^* = p_g/(2m)$:` },
      { t: "math", tex: String.raw`\begin{aligned} C(G) = V(D^*_G, G) &= \mathbb{E}_{p_{\text{data}}}\Big[\log\frac{p_{\text{data}}}{2m}\Big] + \mathbb{E}_{p_g}\Big[\log\frac{p_g}{2m}\Big] \\ &= -\log 4 + \mathrm{KL}(p_{\text{data}}\,\|\,m) + \mathrm{KL}(p_g\,\|\,m) \\ &= -\log 4 + 2\,\mathrm{JS}(p_{\text{data}}\,\|\,p_g) \end{aligned}` },
      { t: "p", text: String.raw`Because $\mathrm{JS}\ge 0$ with equality only when the distributions are equal, the global minimum of $C(G)$ is $-\log 4 \approx -1.386$, reached exactly at $p_g = p_{\text{data}}$. So *if* $D$ were trained to optimality after every generator change, the generator would be minimising the [Jensen–Shannon divergence](/learn/m02/information-theory).` },
      { t: "p", text: String.raw`**Worked example (numerically verified below).** Let $p_{\text{data}} = \mathcal{N}(0, 1)$ and $p_g = \mathcal{N}(1.5, 0.8^2)$. At $x = 0.75$: $p_{\text{data}}(0.75) = 0.301137$, $p_g(0.75) = 0.321342$, so $D^*(0.75) = 0.301137/0.622479 = 0.4838$ — slightly "fake" because $p_g$ is a bit denser there. At $x = -1$, $D^* = 0.9846$ (the generator almost never goes there); at $x = 2.5$, $D^* = 0.0713$. Integrating on a grid: $\mathrm{JS} = 0.262233$ nats, so $C(G) = 2(0.262233) - 1.386294 = -0.861829$, and the direct integral of $V(D^*, G)$ gives the same $-0.861829$. Any other discriminator does worse: $D\equiv 0.5$ gives $-1.386294$, $D = \sigma(-2x + 1.5)$ gives $-0.871351$.` },
      { t: "h", text: "Saturating vs non-saturating generator loss" },
      { t: "p", text: String.raw`Write $D(G(z)) = \sigma(l)$ with logit $l$. The minimax generator minimises $\log(1 - \sigma(l))$, the non-saturating (NS) generator minimises $-\log\sigma(l)$. Their derivatives with respect to the logit are` },
      { t: "math", tex: String.raw`\frac{\partial}{\partial l}\log\big(1-\sigma(l)\big) = -\sigma(l) = -D(G(z)), \qquad \frac{\partial}{\partial l}\big(-\log\sigma(l)\big) = -\big(1-\sigma(l)\big) = -\big(1 - D(G(z))\big)` },
      { t: "p", text: String.raw`Early in training $D$ easily rejects the fakes, $D(G(z))\to 0$, and the minimax gradient vanishes exactly when $G$ most needs a signal. With $D(G(z)) = 0.01$ the minimax gradient is $-0.01$ but the NS gradient is $-0.99$ — 99 times larger. Both losses share the same fixed points ($D(G(z))$ pushed toward 1) but the NS loss is no longer a zero-sum game; with an optimal $D$ it corresponds to minimising $\mathrm{KL}(p_g\|p_{\text{data}}) - 2\,\mathrm{JS}$ (Arjovsky & Bottou, 2017), a reverse-KL-like objective that tolerates dropping modes more than adding junk — one reason NS GANs are prone to mode collapse.` },
      { t: "table", head: ["D(G(z))", "logit l", "∂/∂l of log(1 − D)", "∂/∂l of −log D"], rows: [
        ["0.001", "−6.907", "−0.001", "−0.999"],
        ["0.01", "−4.595", "−0.010", "−0.990"],
        ["0.1", "−2.197", "−0.100", "−0.900"],
        ["0.5", "0", "−0.500", "−0.500"],
        ["0.9", "2.197", "−0.900", "−0.100"],
      ] },
      { t: "h", text: "Why alternating gradient steps can circle forever" },
      { t: "p", text: String.raw`GAN training is a search for a **saddle point**, not a minimum. The smallest example is the bilinear game $\min_x\max_y f(x,y) = xy$, whose unique equilibrium is $(0,0)$. The continuous gradient dynamics are $\dot x = -\partial f/\partial x = -y$, $\dot y = +\partial f/\partial y = x$ — a pure rotation, $x(t) = \cos t$, $y(t) = \sin t$ from $(1,0)$: the radius stays exactly 1, the players orbit the equilibrium and never reach it. Discretising makes it worse. Simultaneous gradient descent–ascent with step $\eta$ is` },
      { t: "math", tex: String.raw`\begin{pmatrix}x_{t+1}\\y_{t+1}\end{pmatrix} = \begin{pmatrix}1 & -\eta\\ \eta & 1\end{pmatrix}\begin{pmatrix}x_t\\y_t\end{pmatrix}, \qquad \|z_{t+1}\| = \sqrt{1+\eta^2}\,\|z_t\|` },
      { t: "p", text: String.raw`so the iterates spiral **outwards**: with $\eta = 0.1$, after 1000 steps the radius is $1.01^{500} = 144.77$. Alternating updates ($y$ sees the new $x$) stay on a bounded orbit (radius $0.98$ after 1000 steps), and the extragradient method, which evaluates the gradient at a look-ahead point, spirals inwards to $0.0069$. Real GANs are non-linear, but near an equilibrium their dynamics have exactly this rotational component, which is why losses oscillate and why fixes such as gradient penalties, two time-scale learning rates and optimistic/extragradient methods help.` },
      { t: "h", text: "Vanishing gradients with disjoint supports and the Wasserstein distance" },
      { t: "p", text: String.raw`Natural images lie near a low-dimensional manifold, and so do generated images; two such manifolds in a high-dimensional space almost never overlap. If the supports are disjoint, $\mathrm{JS}(p_{\text{data}}\|p_g) = \log 2$ no matter how far apart they are, a perfect discriminator exists, and its gradient is zero almost everywhere on the generated samples (Arjovsky & Bottou, 2017). The textbook example: $p_{\text{data}} = \delta_0$, $p_g = \delta_\theta$ on the real line. $\mathrm{JS} = \log 2$ for every $\theta\neq 0$ — no gradient tells $\theta$ to move toward 0.` },
      { t: "p", text: String.raw`Arjovsky, Chintala & Bottou (2017, "Wasserstein GAN") replace JS by the **Earth-Mover (Wasserstein-1) distance**: the minimum total cost of transporting the mass of $p_g$ onto $p_{\text{data}}$, where moving mass $1$ by distance $d$ costs $d$. For the two Diracs, $W_1 = |\theta|$, which shrinks smoothly as $\theta\to 0$. By Kantorovich–Rubinstein duality,` },
      { t: "math", tex: String.raw`W_1(p_{\text{data}}, p_g) = \inf_{\gamma\in\Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x,y)\sim\gamma}\|x-y\| = \sup_{\|f\|_L\le 1}\; \mathbb{E}_{x\sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x\sim p_g}[f(x)]` },
      { t: "p", text: String.raw`$\Pi$ is the set of joint distributions (transport plans) with the two marginals, and the supremum is over 1-Lipschitz functions $f$ ($|f(a)-f(b)|\le\|a-b\|$). The discriminator becomes a **critic** $f$ with an unbounded real output (no sigmoid) that maximises $\mathbb{E}_{\text{data}}[f] - \mathbb{E}_g[f]$; the generator minimises $-\mathbb{E}_z[f(G(z))]$. The original WGAN enforced the Lipschitz constraint crudely by **clipping** every critic weight to $[-c, c]$ ($c = 0.01$), with RMSProp and 5 critic steps per generator step.` },
      { t: "h", text: "WGAN-GP: the gradient penalty" },
      { t: "p", text: String.raw`Weight clipping biases the critic toward overly simple functions and makes optimisation sensitive to $c$. Gulrajani et al. (2017, "Improved Training of Wasserstein GANs") observed that the optimal critic has gradient norm 1 almost everywhere on lines between coupled real and generated points, and replaced clipping with a soft penalty evaluated at random interpolates $\hat x = \epsilon x + (1-\epsilon)\tilde x$, $\epsilon\sim U[0,1]$:` },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{critic}} = \mathbb{E}_{\tilde x\sim p_g}[f(\tilde x)] - \mathbb{E}_{x\sim p_{\text{data}}}[f(x)] + \lambda\,\mathbb{E}_{\hat x}\Big[\big(\|\nabla_{\hat x} f(\hat x)\|_2 - 1\big)^2\Big]` },
      { t: "p", text: String.raw`The paper uses $\lambda = 10$, Adam with $\beta_1 = 0$, $\beta_2 = 0.9$, 5 critic steps, and no batch norm in the critic (the penalty is per-sample, batch norm couples samples). Implementing it requires a *second-order* derivative: the penalty contains $\nabla_x f$, and we need its gradient with respect to the critic's weights — "double backprop" ($\texttt{create\_graph=True}$ in PyTorch). For a leaky-ReLU critic $f(x) = w_L^\top S_{L-1}W_{L-1}\cdots S_1W_1x + \dots$ the input gradient is the product $W_1^\top S_1 W_2^\top\cdots w_L$ of weights and (locally constant) slope matrices $S_l$, which is how the simulator in this lesson computes it by hand.` },
      { t: "h", text: "Evaluation: Inception Score and FID" },
      { t: "p", text: String.raw`GANs have no likelihood to report, so they are evaluated through a pretrained classifier. The **Inception Score** (Salimans et al., 2016) is $\mathrm{IS} = \exp\big(\mathbb{E}_{x\sim p_g}\,\mathrm{KL}(p(y\mid x)\,\|\,p(y))\big)$: high when each sample is confidently classified (sharp) and the marginal over classes is broad (diverse). It ignores the real data entirely. The **Fréchet Inception Distance** (Heusel et al., 2017) fits a Gaussian to Inception-v3 pool features of real images ($\mu_r, \Sigma_r$) and of generated images ($\mu_g, \Sigma_g$) and computes the Fréchet (2-Wasserstein) distance between the two Gaussians:` },
      { t: "math", tex: String.raw`\mathrm{FID} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\Big(\Sigma_r + \Sigma_g - 2\big(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\big)^{1/2}\Big)` },
      { t: "p", text: String.raw`(The trace term is usually written $\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$; the two are equal because $\Sigma_r\Sigma_g$ is similar to the symmetric matrix $\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}$, and the symmetric form can be computed with eigendecompositions only.) Checks: identical Gaussians give 0; a pure mean shift by $(3,4)$ with equal covariances gives $3^2+4^2 = 25$; in 1-D, $\mathcal{N}(0,1)$ vs $\mathcal{N}(1, 2^2)$ gives $(0-1)^2 + 1 + 4 - 2\sqrt{1\cdot 4} = 2$ — in general $(\mu_1-\mu_2)^2 + (\sigma_1-\sigma_2)^2$. Lower is better; FID is biased upward for small sample sizes (two independent samples of the same 16-D Gaussian give 1.42 with $n = 100$ and 0.016 with $n = 10{,}000$), so only compare FIDs computed with the same $n$ and feature extractor.` },
    ],
    visual: [
      { t: "p", text: "This is a real GAN: a generator ($8\\to32\\to32\\to2$, leaky ReLU) and discriminator ($2\\to32\\to32\\to1$) trained with hand-written backpropagation and Adam ($\\beta = (0.5, 0.9)$, batch 64) in your browser. Gold dots are 600 samples $G(z)$ for a fixed set of noise vectors, dark dots are real data, the background is $D(x)$ and the arrows are $-\\partial\\mathcal{L}_G/\\partial x$ — the direction each sample is being pushed." },
      { t: "sim", id: "gan", caption: "Presets were verified by running exactly this TypeScript training code under Node with the default seeds; the numbers quoted below are from those runs." },
      { t: "list", items: [
        "**Converges (non-saturating).** Learning rates $10^{-3}$ for both. Modes covered: 4/8 at iteration 2000, 6/8 at 4000, **8/8 at 5000** (70.5% of samples within $3\\sigma$ of a mode), 8/8 with 82.5% at 8000. Watch the background flatten: at 8000 mean $D$ is 0.53 on real and 0.46 on generated data — close to the 0.5 equilibrium. With seeds 2 and 3 the same settings reached 6/8 and 7/8 at 8000: partial mode dropping is common even when training looks healthy.",
        "**Converges (WGAN-GP).** Five critic steps per generator step, $\\lambda = 0.1$: 2/8 modes at 2000, **8/8 at 4000**, 76.2% high-quality samples at 8000. In our runs WGAN-GP reached 8/8 for all three seeds tried by iteration 8000. The critic's absolute level drifts (only differences matter), so watch the gap between its mean on real and on fake data shrink.",
        "**Collapse + mode hopping (minimax).** Generator learning rate $10^{-2}$, discriminator $10^{-4}$. $D$ hardly learns, so $G$ squeezes its samples onto a thin curve that jumps around: 70 samples in mode 1 at iteration 2000, none on any mode at 3000–4000, 20 in mode 5 at 5000; never more than 1/8 modes and 12% high-quality samples.",
        "**D too strong (minimax).** $\\eta_G = 3\\times10^{-4}$, $\\eta_D = 3\\times10^{-3}$. $D$ wins (mean $D$ = 0.94 on real, 0.04 on fake at 8000) and $G$ never reaches a mode (0/8, at most 3.5% high-quality). The narration compares, at the *same* $D$, the mean push each loss would give: 1.19 per sample with $\\log(1-D)$ versus 34.7 with $-\\log D$ — 29 times larger.",
        "**D too strong (non-saturating).** Same learning rates, NS loss: $G$ now gets a strong signal — and follows it to one place. From iteration 2000 to 8000, 92–99.5% of all samples sit in a single mode (1/8 modes, recall 0.12). This is textbook mode collapse: every sample is individually excellent.",
        "Pause any run and switch the loss or learning rates mid-training (they apply immediately); try the other targets, where precision and recall are compared with what a perfect generator would score.",
      ] },
      { t: "callout", kind: "note", text: "These runs are tiny and seed-dependent; the qualitative behaviour (D-dominance stalls the saturating loss, collapses the non-saturating loss; G-dominance makes the generator hop) is what generalises, not the exact iteration counts." },
    ],
    scratch: [
      { t: "p", text: "First, the theory of the Math section, checked numerically: the optimal discriminator and the JS identity, the saturating vs non-saturating gradients (with finite-difference checks), the bilinear game under three update rules, and the Fréchet distance between Gaussians implemented from scratch and verified on cases with known answers." },
      { t: "code", lang: "python", title: "gan_theory.py", code: THEORY, output: THEORY_OUT },
      { t: "list", items: [
        "Part 1 reproduces $V(G, D^*) = 2\\,\\mathrm{JS} - \\log 4$ to six decimals; every alternative discriminator gives a smaller value, as it must since $D^*$ is the pointwise maximiser.",
        "Part 3 matches the prediction $\\sqrt{1+\\eta^2}^{\\,1000} = 144.77$ for simultaneous updates exactly.",
        "Part 4 computes the matrix square root with `np.linalg.eigh` on symmetric matrices only (stable), and agrees with `scipy.linalg.sqrtm` on a random non-commuting 5-D case.",
      ] },
      { t: "p", text: "Second, a complete GAN in NumPy: a leaky-ReLU MLP with its own backward pass (returning the gradient with respect to the *input* as well as the parameters), Adam, the discriminator step, and the non-saturating generator step that back-propagates through the frozen discriminator into the generator." },
      { t: "code", lang: "python", title: "gan_numpy.py", code: GAN_NP, output: GAN_NP_OUT },
      { t: "list", items: [
        "`D.backward(...)` returns `dx`, the gradient of the generator loss with respect to each generated point; `G.backward(dx)` continues the chain rule into the generator. $D$'s parameter gradients from this pass are simply discarded.",
        "`(sigmoid(l) - 1) / BATCH` is $\\partial\\mathcal{L}/\\partial l$ for both the real half of $\\mathcal{L}_D$ and the NS generator loss; `sigmoid(l) / BATCH` is the fake half of $\\mathcal{L}_D$.",
        "The run is honest about the outcome: after 5000 iterations the generator covers 6 of the 8 modes with 75.5% of samples near a mode, but modes 2 and 6 are almost empty (6 and 0 samples). The losses look perfectly healthy throughout — loss curves do not reveal mode dropping; coverage metrics do.",
        "`BATCH = 128` and `LR = 1e-3` with Adam $\\beta_1 = 0.5$ follow common GAN practice; `STD = 0.05` and the 3σ high-quality radius match the simulator.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**Experiment 1 — minimax vs non-saturating on the 8-Gaussian ring**, with balanced learning rates and with a discriminator that learns 10× faster. Same seed, same architecture (two hidden layers of 64), Adam $(0.5, 0.9)$, batch 128. The numbers are modes covered / fraction of samples within 3σ of a mode." },
          { t: "code", lang: "python", title: "gan_losses_2d.py", code: LOSSES_PT, output: LOSSES_PT_OUT },
          { t: "p", text: "With balanced learning rates the two generator losses behave almost identically here (6/8 modes after 3000 iterations for both, 58.1% vs 66.4% of samples near a mode): $D(G(z))$ stays around 0.4–0.5, where $-D$ and $-(1-D)$ are similar. With a discriminator that learns 10× faster the difference is stark: the minimax generator never reaches a mode (0/8, at most 3.4% high-quality samples), while the non-saturating generator keeps learning but covers only 3 of 8 modes — the same two failure modes as the simulator's *D too strong* presets." },
          { t: "p", text: "**Experiment 2 — WGAN-GP**, with the gradient penalty implemented via `torch.autograd.grad(..., create_graph=True)` so that the penalty's dependence on the critic's weights is itself differentiated." },
          { t: "code", lang: "python", title: "wgan_gp_2d.py", code: WGANGP_PT, output: WGANGP_PT_OUT },
          { t: "p", text: "WGAN-GP covers all 8 modes from iteration 2500 and ends with the most even mode counts of any run (41–82 samples per mode). The critic's estimate $\\mathbb{E}_{\\text{data}}f - \\mathbb{E}_gf$ falls steadily (2.18 → 0.52) as the samples improve — a loss that tracks quality, which the JS-based GAN losses do not provide. We use $\\lambda = 0.1$ and 3 critic steps to fit this tiny 2-D problem into a CPU budget; the paper's $\\lambda = 10$ is the right default for images (in our simulator runs on this ring it was slower to converge). With a small $\\lambda$ the penalty term stays well above 0 (1.46 at the end): the gradient-norm constraint is only loosely enforced." },
          { t: "p", text: "**Experiment 3 — a conditional DCGAN on scikit-learn's 8×8 digits.** The generator receives a one-hot class label next to the noise and upsamples with stride-2 transposed convolutions and batch norm; the discriminator receives the label as 10 constant channels and downsamples with strided convolutions and LeakyReLU — DCGAN's design rules at miniature scale. To check that the samples show the *requested* digit, a logistic-regression classifier trained only on real digits labels 100 generated images per class; a Fréchet distance in a 20-D PCA feature space plays the role of FID." },
          { t: "code", lang: "python", title: "cgan_digits.py", code: CGAN_PT, output: CGAN_PT_OUT },
          { t: "p", text: "After only 1000 iterations the class-conditioning works: a classifier that scores 97.5% on held-out real digits recognises the requested digit in 94.7% of generated images, with the weakest class 8 (77%), which the generator most often draws ambiguously. The Fréchet distance in PCA space is 2.58 for generated digits, against 0.14 between the training and held-out real digits (the floor for these sample sizes) and 10.51 for uniform noise. The ASCII row shows one sample for each class 0–9." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "The same WGAN-GP in TensorFlow. The penalty needs a gradient of a gradient, so a `GradientTape` that computes $\\nabla_{\\hat x}f$ is nested *inside* the tape that differentiates the critic loss with respect to the critic's weights." },
          { t: "code", lang: "python", title: "wgan_gp_keras.py (not executed in CI)", code: WGANGP_TF, output: "Prints the critic loss and generator loss every 500 iterations (6 lines). Numbers differ from the PyTorch run because TensorFlow uses its own random streams and initialisers." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Variant", "Key idea", "Advantages", "Disadvantages / use cases"], rows: [
        ["Original GAN (Goodfellow et al. 2014)", "Minimax BCE; optimal D ⇒ JS divergence", "Simple; sharp samples", "Saturating G loss; unstable; MLPs only in the paper"],
        ["Non-saturating GAN", "G minimises −log D(G(z))", "Strong gradients early; the practical default", "Not zero-sum; prone to mode collapse"],
        ["DCGAN (Radford et al. 2015)", "All-convolutional G/D; strided convs instead of pooling; transposed convs in G; batch norm in both (not G output / D input); ReLU in G with tanh output; LeakyReLU in D; Adam lr 2e-4, β₁ = 0.5", "First stable convolutional GAN recipe; meaningful latent arithmetic", "Still collapses at high resolution"],
        ["Conditional GAN (Mirza & Osindero 2014)", "Feed a label y to both G and D: G(z, y), D(x, y)", "Controllable class of the sample", "Needs labels; D may ignore y without care (projection D, AC-GAN help)"],
        ["Pix2Pix (Isola et al. 2017)", "Conditional GAN for paired image-to-image translation: U-Net G, PatchGAN D, loss = cGAN + λ·L1 (λ = 100)", "Sharp, aligned outputs (maps↔photos, edges→objects)", "Requires paired training data"],
        ["CycleGAN (Zhu et al. 2017)", "Two generators G: X→Y, F: Y→X, two discriminators, cycle-consistency ‖F(G(x)) − x‖₁ + ‖G(F(y)) − y‖₁ (λ = 10)", "Unpaired translation (horses↔zebras, photos↔paintings)", "Changes texture/colour more easily than shape; can hide information steganographically"],
        ["LSGAN (Mao et al. 2017)", "Least-squares loss on D's raw output", "Penalises samples far on the correct side; smoother gradients", "Still a JS-like (Pearson χ²) divergence"],
        ["WGAN (Arjovsky et al. 2017)", "Critic estimates Earth-Mover distance; weight clipping to [−c, c]", "Meaningful loss that tracks quality; no saturation", "Clipping limits capacity; slow (5 critic steps, RMSProp)"],
        ["WGAN-GP (Gulrajani et al. 2017)", "Replace clipping by λ(‖∇f(x̂)‖ − 1)² on interpolates", "Robust across architectures", "Double backprop roughly doubles critic cost; no batch norm in critic"],
        ["Spectral normalisation (Miyato et al. 2018)", "Divide each weight matrix by its largest singular value", "Cheap Lipschitz control; widely used in D", "Constrains capacity somewhat"],
        ["StyleGAN (Karras et al. 2019)", "Mapping network z → w (8-layer MLP), w modulates each synthesis layer via AdaIN-style scaling, per-layer noise inputs, constant 4×4 input, progressive resolution", "State-of-the-art faces; style mixing; disentangled w-space", "Very expensive to train; artefacts fixed in StyleGAN2 (weight demodulation)"],
      ] },
      { t: "h", text: "GANs vs VAEs vs diffusion models" },
      { t: "table", head: ["", "GAN", "VAE", "Diffusion"], rows: [
        ["Training signal", "Adversarial (learned discriminator)", "ELBO: reconstruction + KL", "Denoising (noise-prediction MSE)"],
        ["Likelihood", "None", "Lower bound", "Bound / exact via ODE"],
        ["Sample quality", "Sharp", "Often blurry", "Sharp, state of the art"],
        ["Diversity / mode coverage", "Weakest (mode collapse)", "Good", "Good"],
        ["Sampling cost", "1 forward pass", "1 forward pass", "Tens to hundreds of passes (fewer with distillation)"],
        ["Training stability", "Fragile game", "Stable", "Stable"],
        ["Encoder / latent inference", "No (needs inversion)", "Yes", "Deterministic via DDIM inversion"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Face and image synthesis**: StyleGAN and StyleGAN2 (NVIDIA) generate photorealistic faces at 1024×1024 from the FFHQ dataset; the \"This Person Does Not Exist\" website popularised StyleGAN samples.",
        "**Image-to-image translation**: Pix2Pix (maps ↔ aerial photos, edges → photos, day → night) and CycleGAN (horses ↔ zebras, Monet ↔ photos) for paired and unpaired data; NVIDIA's GauGAN (SPADE, 2019) turns segmentation sketches into landscapes.",
        "**Super-resolution and restoration**: SRGAN (Ledig et al. 2017) and ESRGAN add an adversarial loss to recover realistic texture that a pure L2 loss blurs away.",
        "**Adversarial losses inside other generators**: the autoencoder that defines the latent space of Latent Diffusion / Stable Diffusion is trained with a patch-based adversarial loss, and neural audio codecs and vocoders such as HiFi-GAN use discriminators to make waveforms sound natural.",
        "**Data augmentation and simulation**: synthetic training data for rare classes (e.g. in medical imaging research) and sim-to-real domain adaptation, where a CycleGAN-style mapping makes rendered images look real.",
        "**Interactive editing**: GAN inversion (finding $w$ such that $G(w)$ reproduces a photo) followed by moving along latent directions to change age, pose or lighting.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Sharp samples**: the discriminator penalises any detectable artefact, rather than averaging over uncertainty as a per-pixel likelihood does.",
        "**Fast sampling**: one generator forward pass, versus many denoising steps for diffusion models — still the reason GAN-style decoders and distillation losses are used for real-time generation.",
        "**No likelihood, no encoder**: you cannot evaluate $p_g(x)$ or directly infer $z$ for a given image; evaluation needs proxies such as FID.",
        "**Unstable optimisation**: a two-player game with rotational dynamics; results are sensitive to learning-rate ratios, architecture and seeds (our own runs show 6/8 to 8/8 modes for the same settings across seeds).",
        "**Mode collapse**: the generator can cover only part of the data distribution while every sample looks good — invisible in the losses.",
        "**Hyperparameter burden**: many tricks (spectral norm, gradient penalties, $R_1$ regularisation, EMA of generator weights, two-time-scale learning rates) are needed at scale; diffusion models have largely displaced GANs for text-to-image generation because they train stably and cover modes better.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Letting gradients flow into G during the D step.** If fakes are not detached, `d_loss.backward()` also computes generator gradients (wasted work) and, if an optimizer for $G$ is stepped, moves $G$ to *help* $D$. Fix: `fake.detach()` for the D step.",
        "**Not zeroing D's gradients after the G step.** Back-propagating $\\mathcal{L}_G$ through $D$ accumulates into $D$'s `.grad`. If `opt_d.zero_grad()` is not called before the next D backward, $D$ is updated with the generator's objective mixed in.",
        "**Using the saturating loss literally.** Minimising $\\log(1 - D(G(z)))$ gives vanishing gradients when $D$ is confident. Use the non-saturating loss, i.e. BCE of $D(G(z))$ against label 1.",
        "**Applying a sigmoid and then BCE-with-logits** (or computing $\\log D$ from a clipped probability): numerically unstable and double-squashed gradients. Work with logits and softplus: $-\\log\\sigma(l) = \\mathrm{softplus}(-l)$.",
        "**Reading the losses as progress.** GAN losses do not decrease monotonically and do not measure quality; in our NumPy run they look stable while two modes are dropped. Track samples, coverage metrics or FID.",
        "**Batch norm in a WGAN-GP critic.** The gradient penalty is defined per sample; batch norm makes each output depend on the whole batch. Use layer norm or no normalisation in the critic.",
        "**Comparing FIDs computed differently.** FID depends on the number of samples, the feature network and image resizing; it is biased upward at small $n$. Report the protocol and compare only like with like.",
        "**Equating a falling discriminator loss with a winning generator.** A discriminator loss near 0 means $D$ wins and (with the minimax loss) $G$'s gradient is vanishing — rebalance learning rates or regularise $D$.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What are the generator and the discriminator, and what does each try to do?", a: "The generator maps random noise to samples and tries to make them indistinguishable from real data. The discriminator is a binary classifier that outputs the probability a sample is real; it tries to tell real from generated. They are trained alternately with opposing objectives." },
      { t: "qa", level: "beginner", q: "Why does a GAN need random noise as the generator's input?", a: "A deterministic network with a fixed input produces one output. Feeding samples $z\\sim p_z$ makes $G(z)$ a random variable whose distribution $p_g$ can be shaped to match the data; different $z$ give different samples." },
      { t: "qa", level: "beginner", q: "What is mode collapse?", a: "The generator produces only a small subset of the kinds of samples present in the data (e.g. one digit class, or one cluster of a mixture), often with high individual quality. The discriminator can only reject what it sees, so the generator can satisfy it with limited variety." },
      { t: "qa", level: "intermediate", q: "Derive the optimal discriminator for a fixed generator.", a: "$V = \\int p_{\\text{data}}\\log D + p_g\\log(1-D)\\,dx$ is maximised pointwise. For $a\\log y + b\\log(1-y)$ the derivative $a/y - b/(1-y) = 0$ gives $y = a/(a+b)$, so $D^*(x) = p_{\\text{data}}(x)/(p_{\\text{data}}(x)+p_g(x))$. Substituting yields $V(D^*, G) = 2\\,\\mathrm{JS}(p_{\\text{data}}\\|p_g) - \\log 4$." },
      { t: "qa", level: "intermediate", q: "Why is the non-saturating generator loss used in practice?", a: "With $D(G(z)) = \\sigma(l)$, $\\partial\\log(1-\\sigma(l))/\\partial l = -\\sigma(l)$, which vanishes when $D$ confidently rejects fakes (early training). $\\partial(-\\log\\sigma(l))/\\partial l = -(1-\\sigma(l))$ is largest then. At $D(G(z)) = 0.01$ the gradients are $-0.01$ vs $-0.99$. Both push $D(G(z))$ upward and share fixed points." },
      { t: "qa", level: "intermediate", q: "What does the cycle-consistency loss in CycleGAN do?", a: "With unpaired data, an adversarial loss only forces $G(x)$ to look like domain $Y$, not to correspond to $x$. Adding $\\|F(G(x)) - x\\|_1 + \\|G(F(y)) - y\\|_1$ requires the translations to be (approximately) inverses, so content must be preserved through the round trip." },
      { t: "qa", level: "intermediate", q: "How is FID computed and what are its pitfalls?", a: "Extract Inception-v3 pool features for real and generated images, fit Gaussians $(\\mu_r,\\Sigma_r)$, $(\\mu_g,\\Sigma_g)$ and compute $\\|\\mu_r-\\mu_g\\|^2 + \\mathrm{Tr}(\\Sigma_r+\\Sigma_g-2(\\Sigma_r\\Sigma_g)^{1/2})$. Pitfalls: biased upward for small sample sizes, depends on the feature network and preprocessing, summarises quality and diversity in one number, and assumes Gaussian features." },
      { t: "qa", level: "advanced", q: "Why does the JS-based GAN suffer vanishing gradients when supports don't overlap, and how does WGAN fix it?", a: "If $p_{\\text{data}}$ and $p_g$ live on disjoint low-dimensional manifolds, $\\mathrm{JS} = \\log 2$ regardless of their distance and a perfect discriminator exists whose gradient is zero on the generated samples. The Wasserstein-1 distance measures transport cost, which decreases continuously as $p_g$ moves toward $p_{\\text{data}}$ (two Diracs: $W_1 = |\\theta|$ vs $\\mathrm{JS} = \\log 2$). Its dual $\\sup_{\\|f\\|_L\\le1}\\mathbb{E}_{\\text{data}}f - \\mathbb{E}_g f$ is estimated by a Lipschitz critic, enforced by weight clipping (WGAN) or a gradient penalty (WGAN-GP)." },
      { t: "qa", level: "advanced", q: "Why can simultaneous gradient descent fail to converge in a GAN even near equilibrium?", a: "The game's vector field has a rotational component. For $\\min_x\\max_y xy$, simultaneous updates multiply $(x,y)$ by a matrix with eigenvalues $1\\pm i\\eta$ of modulus $\\sqrt{1+\\eta^2} > 1$, so iterates spiral outward (radius 144.8 after 1000 steps at $\\eta=0.1$). Alternating updates stay bounded; extragradient/optimistic methods, gradient penalties (which add damping) and averaging of iterates converge." },
      { t: "qa", level: "advanced", q: "How does StyleGAN's generator differ from a DCGAN generator?", a: "Instead of feeding $z$ to the first layer, an 8-layer MLP maps $z$ to an intermediate latent $w$; a learned affine transform of $w$ produces per-channel scales and biases that modulate every synthesis layer (AdaIN in StyleGAN, weight modulation/demodulation in StyleGAN2). Synthesis starts from a learned constant, per-pixel noise adds stochastic detail, and using different $w$ at different resolutions enables style mixing. $w$-space is more disentangled than $z$-space." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "At a point $x$, $p_{\\text{data}}(x) = 0.3$ and $p_g(x) = 0.1$. What is the optimal discriminator output $D^*(x)$?", answer: 0.75, tolerance: 0.001, explain: "$D^* = 0.3/(0.3+0.1) = 0.75$." },
        { kind: "numeric", q: "What is the value of $V(D^*, G)$ at the global optimum $p_g = p_{\\text{data}}$ (natural log, 3 decimals)?", answer: -1.386, tolerance: 0.002, explain: "$D^* = 1/2$ everywhere, so $V = \\log\\tfrac12 + \\log\\tfrac12 = -\\log 4 = -1.386$." },
        { kind: "numeric", q: "If $D(G(z)) = 0.02$, what is the magnitude of the derivative of the *saturating* generator loss $\\log(1-D(G(z)))$ with respect to the logit?", answer: 0.02, tolerance: 0.0005, explain: "$\\partial\\log(1-\\sigma(l))/\\partial l = -\\sigma(l) = -0.02$; the non-saturating loss would give $-0.98$." },
        { kind: "numeric", q: "Fréchet distance between the 1-D Gaussians $\\mathcal{N}(0, 1)$ and $\\mathcal{N}(2, 1)$?", answer: 4, tolerance: 0.001, explain: "$(\\mu_1-\\mu_2)^2 + (\\sigma_1-\\sigma_2)^2 = 4 + 0 = 4$." },
        { kind: "numeric", q: "Simultaneous gradient descent–ascent on $f(x,y) = xy$ with $\\eta = 0.2$: by what factor does $\\|(x,y)\\|$ grow per step (4 decimals)?", answer: 1.0198, tolerance: 0.0002, explain: "$\\sqrt{1+\\eta^2} = \\sqrt{1.04} = 1.0198$." },
        { q: "In WGAN-GP, where is the gradient penalty evaluated?", options: ["Only at real samples", "Only at generated samples", "At random points on straight lines between real and generated samples", "At the generator's parameters"], answer: 2, explain: "$\\hat x = \\epsilon x + (1-\\epsilon)\\tilde x$ with $\\epsilon\\sim U[0,1]$; the optimal critic has unit gradient norm along such lines." },
        { q: "A discriminator's loss falls to nearly 0 early in training with the original minimax loss. What is the most likely consequence?", options: ["The generator has converged", "The generator receives vanishing gradients", "Mode collapse is impossible", "FID will be zero"], answer: 1, explain: "$D(G(z))\\approx 0$, and the minimax generator gradient is proportional to $D(G(z))$." },
        { q: "Which statement about CycleGAN is correct?", options: ["It needs paired images", "It uses one generator and one discriminator", "Cycle-consistency asks F(G(x)) ≈ x and G(F(y)) ≈ y", "It minimises the Wasserstein distance"], answer: 2, explain: "Two generators and two discriminators, unpaired data, with L1 cycle-consistency in both directions." },
      ] },
      { t: "exercise", title: "Optimal discriminator on a discrete distribution", prompt: "On three outcomes, $p_{\\text{data}} = (0.5, 0.5, 0)$ and $p_g = (0.25, 0.25, 0.5)$. Compute $D^*$ for each outcome, the value $V(D^*, G)$ directly, and check it against $2\\,\\mathrm{JS} - \\log 4$ (natural logs).", hint: "The mixture is $m = (0.375, 0.375, 0.25)$. Outcome 3 has $p_{\\text{data}} = 0$, so $D^* = 0$ there and $\\log(1 - D^*) = 0$.", solution: [
        { t: "steps", items: [
          { title: "Optimal discriminator", tex: String.raw`D^* = \Big(\tfrac{0.5}{0.75},\ \tfrac{0.5}{0.75},\ \tfrac{0}{0.5}\Big) = (0.6667,\ 0.6667,\ 0)` },
          { title: "Value directly", tex: String.raw`\mathbb{E}_{p_{\text{data}}}\log D^* = \log\tfrac23 = -0.405465,\quad \mathbb{E}_{p_g}\log(1-D^*) = 2(0.25)\log\tfrac13 + 0.5\log 1 = -0.549306,\quad V = -0.954771` },
          { title: "Jensen–Shannon", tex: String.raw`\mathrm{KL}(p_{\text{data}}\|m) = \log\tfrac43 = 0.287682,\quad \mathrm{KL}(p_g\|m) = 0.5\log\tfrac23 + 0.5\log 2 = 0.143841,\quad \mathrm{JS} = 0.215762` },
          { title: "Check", tex: String.raw`2(0.215762) - 1.386294 = -0.954771\ \checkmark` },
        ] },
      ] },
      { t: "exercise", title: "Measure saturation in the NumPy GAN", prompt: "Modify `gan_numpy.py` so the generator uses the saturating loss $\\log(1 - D(G(z)))$, and every 500 iterations also print the mean per-sample norm of $\\partial\\mathcal{L}_G/\\partial x$ for both losses evaluated at the *same* discriminator. Then raise the discriminator's learning rate to 3e-3 and the generator's down to 3e-4. What do you expect to see?", hint: "The only change to the generator step is the gradient with respect to the logit: `-sigmoid(l_fake) / BATCH` instead of `(sigmoid(l_fake) - 1) / BATCH`. For the diagnostic, back-propagate a vector of ones through D to get the per-sample input gradient of the logit.", solution: [
        { t: "code", lang: "python", title: "changes to gan_numpy.py", code: String.raw`# generator step, saturating loss: d/dl log(1 - sigmoid(l)) = -sigmoid(l)
_, dx = D.backward(-sigmoid(l_fake) / BATCH)

# diagnostic: per-sample gradient of the logit w.r.t. x, then weight it per loss
x_gen = G.forward(z_eval)
l = D.forward(x_gen)
_, grad_l = D.backward(np.ones_like(l))                  # row i = d l_i / d x_i
norm = np.linalg.norm(grad_l, axis=1)
p = sigmoid(l[:, 0])
print("saturating push", np.mean(p * norm), " non-saturating push", np.mean((1 - p) * norm))` },
        { t: "p", text: "With a strong discriminator, $D(G(z))$ falls toward 0 and the saturating push becomes a small fraction of the non-saturating push for the same $D$ — in the simulator's preset run the ratio was 1.19 vs 34.7 per sample at iteration 8000. With the saturating loss the generator stalls far from the modes; switching to the non-saturating loss restores a strong gradient (and, with these learning rates, typically produces mode collapse instead)." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "GAN = generator $G(z)$ + discriminator $D(x)$ trained as a two-player game; $G$ learns only through $\\nabla_x D$.",
        "Objective: $\\min_G\\max_D \\mathbb{E}_{\\text{data}}\\log D(x) + \\mathbb{E}_z\\log(1-D(G(z)))$.",
        "Optimal discriminator $D^* = p_{\\text{data}}/(p_{\\text{data}}+p_g)$; then $V = 2\\,\\mathrm{JS}(p_{\\text{data}}\\|p_g) - \\log 4$, minimised at $p_g = p_{\\text{data}}$ with value $-\\log 4$.",
        "Use the non-saturating loss $-\\log D(G(z))$: gradient $-(1-D)$ instead of $-D$ w.r.t. the logit.",
        "Failure modes: mode collapse, oscillation (rotational game dynamics; $\\|z\\|$ grows by $\\sqrt{1+\\eta^2}$ per simultaneous step on $xy$), vanishing gradients from a too-strong $D$ or disjoint supports.",
        "WGAN: critic $f$, $W_1 = \\sup_{\\|f\\|_L\\le1}\\mathbb{E}_{\\text{data}}f - \\mathbb{E}_gf$, Lipschitz via weight clipping; WGAN-GP: $+\\lambda\\,\\mathbb{E}(\\|\\nabla f(\\hat x)\\|-1)^2$ on interpolates.",
        "Architectures: DCGAN rules; conditional GAN $G(z,y), D(x,y)$; Pix2Pix = cGAN + L1 with paired data; CycleGAN = two GANs + cycle-consistency; StyleGAN = mapping network + style modulation.",
        "Evaluate with FID $= \\|\\mu_r-\\mu_g\\|^2 + \\mathrm{Tr}(\\Sigma_r+\\Sigma_g-2(\\Sigma_r\\Sigma_g)^{1/2})$ (and IS); never by the training losses.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Diffusion models](/learn/m14/diffusion) — the generative family that overtook GANs: stable training and full mode coverage, at the cost of iterative sampling.",
        "[Variational autoencoders](/learn/m14/vae) — revisit the likelihood-based alternative; VAE-GAN hybrids and adversarially trained autoencoders combine both.",
        "[Self-supervised learning](/learn/m15/self-supervised) — contrastive objectives also learn by discriminating between samples, without labels.",
        "[Experimentation](/learn/m17/experimentation) — seeds, variance across runs and fair comparison matter even more for unstable training procedures like GANs.",
      ] },
    ],
  },
};

export default lesson;
