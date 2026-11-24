"""
train-autoencoders.py -- trains the small autoencoders used by the "Autoencoder (trained model)"
simulator (components/sims/AutoencoderSim.tsx) and the "Variational Autoencoder (trained model)"
simulator (components/sims/VaeSim.tsx), and exports them to
    web/public/models/autoencoder.json   (deterministic AE + denoising AE, 2-D latent)
    web/public/models/vae.json           (VAE, 2-D latent, beta = 1)

DATA
  scikit-learn's bundled 8x8 handwritten digits (sklearn.datasets.load_digits: 1797 images, grey
  levels 0..16, a copy of the UCI "Optical Recognition of Handwritten Digits" test set). No download.
  Pixels are divided by 16 so they lie in [0, 1]. Split: train_test_split(test_size=0.2,
  stratify=labels, random_state=0) -> 1437 training and 360 test images. Labels are NEVER used for
  training; they only colour the latent scatter plots.

MODELS (all multilayer perceptrons, ReLU hidden layers, sigmoid output, 2-D latent for plotting)
  * AE   encoder 64 -> 64 -> 32 -> 2 (linear code z), decoder 2 -> 32 -> 64 -> 64.
  * DAE  identical architecture; each training input is corrupted with Gaussian noise of a standard
         deviation drawn uniformly from [0, 0.5] per image and clipped to [0, 1]; the target is the
         clean image (Vincent et al., 2008 use masking noise; additive Gaussian noise is the variant
         used here).
  * VAE  encoder 64 -> 64 -> 32 -> (mu: 2, log sigma^2: 2), decoder 2 -> 32 -> 64 -> 64 (Kingma &
         Welling, 2013). Reconstruction term: Bernoulli log-likelihood = binary cross-entropy summed
         over the 64 pixels (grey levels in [0, 1] used as soft targets); KL term: closed form
         KL(N(mu, diag sigma^2) || N(0, I)), summed over the 2 latent dimensions. Loss = recon + beta*KL
         per image, averaged over the mini-batch, beta = 1. One reparameterised sample per image.
  * A small beta sweep (beta in {0.1, 1, 4}) is also trained and only its test-set statistics are
    exported (no weights), for the beta discussion in the simulator.
  The AE and DAE use the same summed binary cross-entropy reconstruction loss, so their numbers are
  directly comparable with the VAE's reconstruction term.
  Optimiser: Adam, lr 2e-3 decayed by cosine to 1e-4, batch 64, 400 epochs.

EXPORT FORMAT (JSON; readers: lib/sim/autoencoder.ts and lib/sim/vae.ts)
  Every weight tensor is {"shape": [in, out], "scale": s, "q": [integers]} with value = q * s
  (symmetric per-tensor quantisation, |q| <= 9999) stored row-major in "x @ W" orientation.
  After quantising, the de-quantised weights are loaded back into the PyTorch model and EVERYTHING
  exported (losses, codes, verification vectors) is computed with exactly the weights the browser
  receives. Test images are stored as integer grey levels 0..16. Verification vectors: the hidden
  activations, codes and reconstructions for a few test images and the decoder output for a few
  fixed latent points (VAE: also mu, log sigma^2, a stored epsilon, z = mu + sigma * epsilon, and
  the loss terms), produced by model.eval() under torch.no_grad().

HOW TO RUN (from the repository root or from web/)
    python web/scripts/train-autoencoders.py
  Requires PyTorch (CPU) and scikit-learn. Deterministic: fixed seeds, one thread and
  torch.use_deterministic_algorithms(True). Runtime: about 5-6 minutes on a laptop CPU (five models
  x 400 epochs; the committed run took 333 s).

RESULT OF THE COMMITTED RUN (test set, per image, nats = summed BCE over 64 pixels)
  AE : 12,674 parameters, test BCE 20.65, test MSE 0.0339
  DAE: 12,674 parameters, test BCE 20.23, test MSE 0.0317; at input noise sigma = 0.3 its output MSE is
       0.037 vs 0.068 for the plain AE (the noisy input itself has MSE 0.051)
  VAE: 12,740 parameters, test reconstruction 21.23 + KL 2.94 = -ELBO 24.17
  beta sweep (test recon / KL): beta 0.1 -> 20.47 / 6.22, beta 1 -> 21.23 / 2.94,
       beta 4 -> 27.20 / 0.00 (complete posterior collapse: q(z|x) = prior for every x)
  JSON sizes: autoencoder.json ~243 KB, vae.json ~160 KB. The TypeScript engines reproduce the
  PyTorch verification vectors to max |diff| ~5e-6 (AE) and ~7e-7 (VAE).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "public" / "models"

SEED = 0
EPOCHS, BATCH, LR_MAX, LR_MIN = 400, 64, 2e-3, 1e-4
ENC, DEC = [64, 64, 32], [32, 64, 64]      # hidden sizes; latent dimension 2 in between
LATENT = 2
QMAX = 9999
DAE_MAX_SIGMA = 0.5

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# ----------------------------------------------------------------- data
digits = load_digits()
X_all = digits.data.astype(np.float32)            # grey levels 0..16
y_all = digits.target
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=0)
Xtr = torch.tensor(Xtr_raw / 16.0)
Xte = torch.tensor(Xte_raw / 16.0)
print(f"train {len(Xtr)}  test {len(Xte)}")


# ----------------------------------------------------------------- models
def mlp(sizes, last_act=None):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if last_act is not None:
        layers.append(last_act)
    return nn.Sequential(*layers)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = mlp(ENC + [LATENT])                 # 64 -> 64 -> 32 -> 2 (linear code)
        self.dec = mlp([LATENT] + DEC, nn.Sigmoid())   # 2 -> 32 -> 64 -> 64, sigmoid pixels

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = mlp(ENC)                           # 64 -> 64 -> 32 (ReLU after both)
        self.body.append(nn.ReLU())
        self.mu = nn.Linear(ENC[-1], LATENT)
        self.logvar = nn.Linear(ENC[-1], LATENT)
        self.dec = mlp([LATENT] + DEC, nn.Sigmoid())

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x, eps=None):
        mu, logvar = self.encode(x)
        if eps is None:
            eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps         # reparameterisation trick
        return self.dec(z), mu, logvar, z


def bce_sum(xhat, x):
    """Summed binary cross-entropy over the 64 pixels, per image (shape [B])."""
    return F.binary_cross_entropy(xhat, x, reduction="none").sum(dim=1)


def kl_std_normal(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)


def lr_at(step, total):
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / total))


def train(model, loss_fn, name, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    steps_per_epoch = math.ceil(len(Xtr) / BATCH)
    total = EPOCHS * steps_per_epoch
    step = 0
    hist = []
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=gen)
        model.train()
        for i in range(0, len(Xtr), BATCH):
            xb = Xtr[perm[i:i + BATCH]]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total)
            loss = loss_fn(model, xb, gen).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        if ep % 100 == 0 or ep == EPOCHS - 1:
            ev = evaluate(model, name)
            hist.append({"epoch": ep + 1, **ev})
            print(f"  {name} epoch {ep + 1:4d}  " + "  ".join(f"{k} {v:.3f}" for k, v in ev.items()))
    return hist


@torch.no_grad()
def evaluate(model, name):
    model.eval()
    out = {}
    if isinstance(model, VAE):
        g = torch.Generator().manual_seed(123)
        for split, X in (("train", Xtr), ("test", Xte)):
            eps = torch.randn(X.shape[0], LATENT, generator=g)
            xhat, mu, logvar, _ = model(X, eps)
            r, k = bce_sum(xhat, X).mean().item(), kl_std_normal(mu, logvar).mean().item()
            out[f"{split}_recon"] = r
            out[f"{split}_kl"] = k
            out[f"{split}_neg_elbo"] = r + k
    else:
        for split, X in (("train", Xtr), ("test", Xte)):
            xhat, _ = model(X)
            out[f"{split}_bce"] = bce_sum(xhat, X).mean().item()
            out[f"{split}_mse"] = F.mse_loss(xhat, X).item()
    model.train()
    return out


def ae_loss(model, xb, gen):
    xhat, _ = model(xb)
    return bce_sum(xhat, xb)


def dae_loss(model, xb, gen):
    sig = torch.rand(xb.shape[0], 1, generator=gen) * DAE_MAX_SIGMA
    noisy = (xb + sig * torch.randn(xb.shape, generator=gen)).clamp(0, 1)
    xhat, _ = model(noisy)
    return bce_sum(xhat, xb)


def make_vae_loss(beta):
    def f(model, xb, gen):
        eps = torch.randn(xb.shape[0], LATENT, generator=gen)
        xhat, mu, logvar, _ = model(xb, eps)
        return bce_sum(xhat, xb) + beta * kl_std_normal(mu, logvar)
    return f


# ----------------------------------------------------------------- quantisation helpers
def quantise_linear(lin: nn.Linear, prefix: str, store: dict):
    """Store W (as in x out) and b, then load the de-quantised values back into the layer."""
    with torch.no_grad():
        for key, t, tr in ((prefix + ".W", lin.weight.T, True), (prefix + ".b", lin.bias, False)):
            t = t.detach().double()
            scale = (t.abs().max().item() / QMAX) or 1.0
            scale = float(f"{scale:.9e}")
            qi = torch.round(t / scale).to(torch.int64)
            store[key] = {"shape": list(t.shape), "scale": scale, "q": qi.flatten().tolist()}
            deq = (qi.double() * scale).float()
            if tr:
                lin.weight.copy_(deq.T)
            else:
                lin.bias.copy_(deq)


def linears(seq):
    return [m for m in seq if isinstance(m, nn.Linear)]


def r(v, d=5):
    return [float(f"{x:.{d}g}") for x in (v.tolist() if hasattr(v, "tolist") else v)]


t0 = time.time()

# ----------------------------------------------------------------- AE + DAE
print("training AE")
torch.manual_seed(SEED)
ae = AE()
ae_hist = train(ae, ae_loss, "AE", SEED)
print("training DAE")
torch.manual_seed(SEED)
dae = AE()
dae_hist = train(dae, dae_loss, "DAE", SEED)

ae_models = {}
for name, m, hist in (("ae", ae, ae_hist), ("dae", dae, dae_hist)):
    store = {}
    for i, lin in enumerate(linears(m.enc)):
        quantise_linear(lin, f"enc.{i}", store)
    for i, lin in enumerate(linears(m.dec)):
        quantise_linear(lin, f"dec.{i}", store)
    m.eval()
    final = evaluate(m, name)
    m.eval()
    with torch.no_grad():
        codes_te = m.enc(Xte)
        codes_tr = m.enc(Xtr)
        # denoising evaluation at a few noise levels (fixed noise)
        g = torch.Generator().manual_seed(7)
        noise = torch.randn(Xte.shape, generator=g)
        den = {}
        for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            noisy = (Xte + s * noise).clamp(0, 1)
            den[f"{s:.1f}"] = {"noisy_mse": round(F.mse_loss(noisy, Xte).item(), 5),
                               "recon_mse": round(F.mse_loss(m(noisy)[0], Xte).item(), 5)}
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name}: params {n_params}  final {final}  denoise {den}")
    ae_models[name] = {"tensors": store, "n_params": n_params, "history": hist,
                       "final": {k: round(v, 4) for k, v in final.items()},
                       "denoise": den,
                       "codes_test": [r(c, 4) for c in codes_te],
                       "codes_train": [r(c, 4) for c in codes_tr]}

# verification vectors for the AE engine
verify_ae = []
with torch.no_grad():
    for name, m in (("ae", ae), ("dae", dae)):
        for idx in (0, 17, 123):
            x = Xte[idx:idx + 1]
            h1 = F.relu(m.enc[0](x)); h2 = F.relu(m.enc[2](h1)); z = m.enc[4](h2)
            d1 = F.relu(m.dec[0](z)); d2 = F.relu(m.dec[2](d1)); xhat = m.dec[5](m.dec[4](d2))
            assert torch.allclose(xhat, m(x)[0])
            verify_ae.append({"model": name, "test_index": idx, "enc_h1": r(h1[0], 7), "enc_h2": r(h2[0], 7),
                              "z": r(z[0], 7), "dec_h1": r(d1[0], 7), "dec_h2": r(d2[0], 7), "recon": r(xhat[0], 7)})
        for zz in ([0.0, 0.0], [3.0, -2.0], [-25.0, 40.0]):
            out = m.dec(torch.tensor([zz]))
            verify_ae.append({"model": name, "z": zz, "recon": r(out[0], 7)})

export_ae = {
    "format": "autoencoder/1",
    "description": "Deterministic MLP autoencoder (and a denoising twin) with a 2-D latent code, trained on the 8x8 scikit-learn digits.",
    "data": {"name": "scikit-learn load_digits (8x8, grey levels 0..16, scaled to [0,1])", "n_train": len(Xtr), "n_test": len(Xte),
             "split": "train_test_split(test_size=0.2, stratify=y, random_state=0)"},
    "config": {"enc": ENC + [LATENT], "dec": [LATENT] + DEC, "hidden_act": "relu", "code_act": "linear", "out_act": "sigmoid",
               "loss": "binary cross-entropy summed over 64 pixels", "dae_noise": f"x + s*N(0,1), s ~ U(0,{DAE_MAX_SIGMA}) per image, clipped to [0,1]"},
    "training": {"epochs": EPOCHS, "batch": BATCH, "optimizer": f"Adam, lr {LR_MAX} cosine to {LR_MIN}", "seed": SEED},
    "quantisation": f"value = q * scale per tensor, |q| <= {QMAX}",
    "models": ae_models,
    "test": {"labels": [int(v) for v in yte], "pixels": [[int(v) for v in row] for row in Xte_raw],
             "train_labels": [int(v) for v in ytr]},
    "verify": verify_ae,
}

# ----------------------------------------------------------------- VAE (+ beta sweep)
def train_vae(beta, seed):
    torch.manual_seed(seed)
    m = VAE()
    hist = train(m, make_vae_loss(beta), f"VAE(b={beta})", seed)
    return m, hist


beta_sweep = []
for beta in (0.1, 4.0):
    print(f"training beta-VAE beta={beta}")
    m, _ = train_vae(beta, SEED)
    ev = evaluate(m, "vae")
    with torch.no_grad():
        mu, logvar = m.encode(Xte)
        active = int(((mu.var(dim=0)) > 0.01).sum().item())
    beta_sweep.append({"beta": beta, "test_recon": round(ev["test_recon"], 3), "test_kl": round(ev["test_kl"], 3),
                       "mean_sigma": [round(v, 3) for v in torch.exp(0.5 * logvar).mean(0).tolist()], "active_dims": active})

print("training VAE beta=1")
vae, vae_hist = train_vae(1.0, SEED)
store = {}
for i, lin in enumerate(linears(vae.body)):
    quantise_linear(lin, f"enc.{i}", store)
quantise_linear(vae.mu, "mu", store)
quantise_linear(vae.logvar, "logvar", store)
for i, lin in enumerate(linears(vae.dec)):
    quantise_linear(lin, f"dec.{i}", store)
vae.eval()
final = evaluate(vae, "vae")
with torch.no_grad():
    mu_te, lv_te = vae.encode(Xte)
    mu_tr, lv_tr = vae.encode(Xtr)
    active = int(((mu_te.var(dim=0)) > 0.01).sum().item())
beta_sweep.insert(1, {"beta": 1.0, "test_recon": round(final["test_recon"], 3), "test_kl": round(final["test_kl"], 3),
                      "mean_sigma": [round(v, 3) for v in torch.exp(0.5 * lv_te).mean(0).tolist()], "active_dims": active})
n_params = sum(p.numel() for p in vae.parameters())
print(f"vae: params {n_params}  final {final}")
for b in beta_sweep:
    print("  beta sweep", b)

verify_vae = []
g = torch.Generator().manual_seed(2024)
with torch.no_grad():
    for idx in (0, 17, 123):
        x = Xte[idx:idx + 1]
        eps = torch.randn(1, LATENT, generator=g)
        h1 = F.relu(vae.body[0](x)); h2 = F.relu(vae.body[2](h1))
        mu, lv = vae.mu(h2), vae.logvar(h2)
        xhat, mu2, lv2, z = vae(x, eps)
        d1 = F.relu(vae.dec[0](z)); d2 = F.relu(vae.dec[2](d1))
        verify_vae.append({"test_index": idx, "eps": r(eps[0], 9), "enc_h1": r(h1[0], 7), "enc_h2": r(h2[0], 7),
                           "mu": r(mu[0], 7), "logvar": r(lv[0], 7), "z": r(z[0], 7), "dec_h1": r(d1[0], 7), "dec_h2": r(d2[0], 7),
                           "recon": r(xhat[0], 7), "recon_loss": float(f"{bce_sum(xhat, x).item():.7g}"),
                           "kl": float(f"{kl_std_normal(mu, lv).item():.7g}")})
    for zz in ([0.0, 0.0], [1.5, -1.0], [-2.0, 2.0]):
        verify_vae.append({"z": zz, "recon": r(vae.dec(torch.tensor([zz]))[0], 7)})

export_vae = {
    "format": "vae/1",
    "description": "Variational autoencoder (Kingma & Welling, 2013) with a 2-D Gaussian latent, trained on the 8x8 scikit-learn digits.",
    "data": export_ae["data"],
    "config": {"enc": ENC, "latent": LATENT, "dec": [LATENT] + DEC, "hidden_act": "relu", "out_act": "sigmoid",
               "likelihood": "Bernoulli (binary cross-entropy summed over 64 pixels)", "prior": "N(0, I)", "beta": 1.0,
               "samples_per_example": 1},
    "training": {"epochs": EPOCHS, "batch": BATCH, "optimizer": f"Adam, lr {LR_MAX} cosine to {LR_MIN}", "seed": SEED,
                 "history": vae_hist, "final": {k: round(v, 4) for k, v in final.items()}},
    "beta_sweep": beta_sweep,
    "n_params": n_params,
    "quantisation": f"value = q * scale per tensor, |q| <= {QMAX}",
    "tensors": store,
    "test": {"labels": export_ae["test"]["labels"], "pixels": export_ae["test"]["pixels"],
             "mu": [r(v, 4) for v in mu_te], "logvar": [r(v, 4) for v in lv_te]},
    "train_mu": [r(v, 4) for v in mu_tr],
    "train_labels": export_ae["test"]["train_labels"],
    "verify": verify_vae,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
for fname, obj in (("autoencoder.json", export_ae), ("vae.json", export_vae)):
    p = OUT_DIR / fname
    p.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {p.name}: {p.stat().st_size / 1024:.1f} KB")
print(f"done in {time.time() - t0:.0f}s")
