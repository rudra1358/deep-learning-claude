import type { Lesson } from "@/lib/types";

const AE_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- data: 8x8 digits, pixels scaled to [0, 1] ----------------
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
mean = X_tr.mean(axis=0)
Xc_tr, Xc_te = X_tr - mean, X_te - mean          # centred data (PCA needs it; so does the linear AE here)
N, D = Xc_tr.shape
K = 8                                            # bottleneck width

# ---------------- 1. PCA: the optimal rank-K linear reconstruction ----------------
_, S, Vt = np.linalg.svd(Xc_tr, full_matrices=False)
P = Vt[:K]                                       # (K, D) top-K principal directions
pca_rec = Xc_te @ P.T @ P
print(f"PCA          K={K}  train MSE {np.mean((Xc_tr @ P.T @ P - Xc_tr) ** 2):.5f}  test MSE {np.mean((pca_rec - Xc_te) ** 2):.5f}")

# ---------------- 2. linear autoencoder trained by gradient descent ----------------
# encoder z = x We  (D x K), decoder x_hat = z Wd  (K x D), loss = mean squared error
rng = np.random.default_rng(0)
We = rng.normal(0, 0.1, (D, K))
Wd = rng.normal(0, 0.1, (K, D))
lr = 0.05
for step in range(3001):
    Z = Xc_tr @ We
    R = Z @ Wd - Xc_tr                           # residual
    loss = np.mean(R ** 2)
    G = 2 * R / R.size                           # dL/dX_hat
    gWd = Z.T @ G                                # chain rule through the decoder
    gWe = Xc_tr.T @ (G @ Wd.T)                   # ... and through the encoder
    We -= lr * gWe * D                           # a D-scaled step: the mean over D pixels makes raw gradients tiny
    Wd -= lr * gWd * D
    if step % 1000 == 0:
        print(f"linear AE    step {step:4d}  train MSE {loss:.5f}")
lin_rec = Xc_te @ We @ Wd
print(f"linear AE    K={K}  test MSE {np.mean((lin_rec - Xc_te) ** 2):.5f}")

# the decoder's row space should equal the PCA subspace: principal angles via SVD
Q_ae, _ = np.linalg.qr(Wd.T)                     # orthonormal basis of the AE's K-dim subspace
cosines = np.linalg.svd(P @ Q_ae, compute_uv=False)
print("cosines of principal angles AE vs PCA:", np.round(cosines, 4))
print("decoder rows orthonormal?  max |Wd Wd^T - I| =", round(np.abs(Wd @ Wd.T - np.eye(K)).max(), 3))

# ---------------- 3. non-linear autoencoder from scratch (tanh, sigmoid output, BCE, Adam) ----------------
sizes = [D, 32, K, 32, D]
W = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
b = [np.zeros(s) for s in sizes[1:]]
params = W + b
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]


def forward(Xb):
    acts = [Xb]
    for l in range(4):
        z = acts[-1] @ W[l] + b[l]
        if l == 1:
            acts.append(z)                            # linear bottleneck (the code)
        elif l == 3:
            acts.append(1 / (1 + np.exp(-z)))         # sigmoid output pixels
        else:
            acts.append(np.tanh(z))
    return acts


def backward(acts, Xb):
    gW, gb = [None] * 4, [None] * 4
    delta = (acts[-1] - Xb) / len(Xb)                 # sigmoid + BCE (summed over pixels, mean over batch)
    for l in reversed(range(4)):
        gW[l] = acts[l].T @ delta
        gb[l] = delta.sum(axis=0)
        if l > 0:
            delta = delta @ W[l].T
            if l != 2:                                # layer-1 output (the code) is linear
                delta = delta * (1 - acts[l] ** 2)
    return gW + gb


t, lr, beta1, beta2 = 0, 3e-3, 0.9, 0.999
for epoch in range(101):
    perm = rng.permutation(N)
    for i in range(0, N, 64):
        Xb = X_tr[perm[i:i + 64]]
        grads = backward(forward(Xb), Xb)
        t += 1
        for k, (p, g) in enumerate(zip(params, grads)):
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v[k] = beta2 * v[k] + (1 - beta2) * g ** 2
            p -= lr * (m[k] / (1 - beta1 ** t)) / (np.sqrt(v[k] / (1 - beta2 ** t)) + 1e-8)
    if epoch % 50 == 0:
        rec = forward(X_te)[-1]
        print(f"non-linear AE epoch {epoch:3d}  test MSE {np.mean((rec - X_te) ** 2):.5f}")`;
const AE_NUMPY_OUT = String.raw`PCA          K=8  train MSE 0.02378  test MSE 0.02465
linear AE    step    0  train MSE 0.07768
linear AE    step 1000  train MSE 0.02385
linear AE    step 2000  train MSE 0.02379
linear AE    step 3000  train MSE 0.02379
linear AE    K=8  test MSE 0.02465
cosines of principal angles AE vs PCA: [1.     1.     1.     1.     1.     1.     1.     0.9997]
decoder rows orthonormal?  max |Wd Wd^T - I| = 0.161
non-linear AE epoch   0  test MSE 0.07840
non-linear AE epoch  50  test MSE 0.01895
non-linear AE epoch 100  test MSE 0.01644`;
const CAE_NUMPY = String.raw`import numpy as np

# ---- reconstruction losses for one 4-pixel "image" ----
x = np.array([1.0, 0.0, 0.5, 0.25])               # target pixels in [0, 1]
x_hat = np.array([0.9, 0.2, 0.4, 0.3])            # decoder output (sigmoid probabilities)
mse = np.mean((x_hat - x) ** 2)
bce = -np.sum(x * np.log(x_hat) + (1 - x) * np.log(1 - x_hat))
bce_floor = -np.sum([v * np.log(v) + (1 - v) * np.log(1 - v) for v in x if 0 < v < 1])   # entropy of soft targets
print(f"MSE = {mse:.6f}   summed BCE = {bce:.6f}   (BCE floor for these targets = {bce_floor:.6f})")

# ---- contractive penalty for a sigmoid encoder layer h = sigmoid(W x + b) ----
W = np.array([[0.5, -1.0, 2.0],
              [1.5, 0.3, -0.7]])                    # 2 hidden units, 3 inputs
b = np.array([0.1, -0.2])
xin = np.array([0.2, 0.4, 0.6])


def enc(v):
    return 1 / (1 + np.exp(-(W @ v + b)))


h = enc(xin)
J = (h * (1 - h))[:, None] * W                       # dh_j/dx_i = h_j (1 - h_j) W_ji
closed = np.sum((h * (1 - h)) ** 2 * np.sum(W ** 2, axis=1))
eps = 1e-6
J_fd = np.stack([(enc(xin + eps * e) - enc(xin - eps * e)) / (2 * eps) for e in np.eye(3)], axis=1)
print("pre-activations a =", np.round(W @ xin + b, 6))
print("h =", np.round(h, 6), "  h(1-h) =", np.round(h * (1 - h), 6))
print("row norms ||W_j||^2 =", np.sum(W ** 2, axis=1))
print("Jacobian:\n", np.round(J, 6))
print(f"||J||_F^2 closed form {closed:.6f}   from the matrix {np.sum(J ** 2):.6f}   "
      f"finite differences {np.sum(J_fd ** 2):.6f}")`;
const CAE_NUMPY_OUT = String.raw`MSE = 0.015625   summed BCE = 1.610562   (BCE floor for these targets = 1.255482)
pre-activations a = [ 1.  -0.2]
h = [0.731059 0.450166]   h(1-h) = [0.196612 0.247517]
row norms ||W_j||^2 = [5.25 2.83]
Jacobian:
 [[ 0.098306 -0.196612  0.393224]
 [ 0.371275  0.074255 -0.173262]]
||J||_F^2 closed form 0.376324   from the matrix 0.376324   finite differences 0.376324`;
const AE_TORCH = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)
X_tr = torch.tensor(X_tr, dtype=torch.float32)
X_te = torch.tensor(X_te, dtype=torch.float32)


class AE(nn.Module):
    """One sigmoid hidden layer h = s(W x + b) (the code), sigmoid output x_hat = s(V h + c)."""

    def __init__(self, hidden):
        super().__init__()
        self.enc = nn.Linear(64, hidden)
        self.dec = nn.Linear(hidden, 64)

    def forward(self, x):
        h = torch.sigmoid(self.enc(x))
        return torch.sigmoid(self.dec(h)), h


def jacobian_fro2(model, h):
    """||dh/dx||_F^2 for a sigmoid layer, in closed form: sum_j (h_j(1-h_j))^2 * ||W_j||^2."""
    return ((h * (1 - h)) ** 2 * (model.enc.weight ** 2).sum(dim=1)).sum(dim=1)


def mask(x, p, gen):
    """Masking noise (Vincent et al. 2008): set a random fraction p of the pixels to 0."""
    return x * (torch.rand(x.shape, generator=gen) >= p).float()


def train(hidden, kind="plain", lam=0.0, epochs=100):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = AE(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(epochs):
        for idx in torch.randperm(len(X_tr), generator=gen).split(64):
            x = X_tr[idx]
            x_in = mask(x, 0.25, gen) if kind == "denoising" else x
            x_hat, h = model(x_in)
            loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
            if kind == "sparse":                 # KL(rho || rho_hat_j) summed over hidden units
                rho, rho_hat = 0.05, h.mean(dim=0).clamp(1e-6, 1 - 1e-6)
                loss = loss + lam * (rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).sum()
            if kind == "contractive":
                loss = loss + lam * jacobian_fro2(model, h).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


gen = torch.Generator().manual_seed(1)
X_te_masked = mask(X_te, 0.25, gen)              # the same corrupted test set for every model
print(f"{'model':<26}{'clean MSE':>10}{'masked MSE':>11}{'mean h':>8}{'||J||_F':>9}")
for name, hidden, kind, lam in [("undercomplete (16)", 16, "plain", 0),
                                ("overcomplete (128)", 128, "plain", 0),
                                ("sparse (128, rho=.05)", 128, "sparse", 3.0),
                                ("denoising (128, 25% mask)", 128, "denoising", 0),
                                ("contractive (128)", 128, "contractive", 0.3)]:
    model = train(hidden, kind, lam)
    with torch.no_grad():
        rec, h = model(X_te)
        rec_m, _ = model(X_te_masked)
        clean = F.mse_loss(rec, X_te).item()
        masked = F.mse_loss(rec_m, X_te).item()
        jac = jacobian_fro2(model, h).sqrt().mean().item()
    print(f"{name:<26}{clean:>10.4f}{masked:>11.4f}{h.mean().item():>8.3f}{jac:>9.2f}")
print(f"{'(identity: copy the input)':<26}{0:>10.4f}{F.mse_loss(X_te_masked, X_te).item():>11.4f}")`;
const AE_TORCH_OUT = String.raw`model                      clean MSE masked MSE  mean h  ||J||_F
undercomplete (16)            0.0161     0.0317   0.583     3.57
overcomplete (128)            0.0027     0.0495   0.496    11.47
sparse (128, rho=.05)         0.0127     0.0356   0.055     2.11
denoising (128, 25% mask)     0.0113     0.0193   0.473    10.32
contractive (128)             0.0102     0.0372   0.491     1.87
(identity: copy the input)    0.0000     0.0578`;
const AE_ANOMALY = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.2, stratify=y, random_state=0)
T_tr = torch.tensor(X_tr, dtype=torch.float32)
T_te = torch.tensor(X_te, dtype=torch.float32)


def make_ae(code):
    enc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, code))
    dec = nn.Sequential(nn.Linear(code, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())
    return enc, dec


def fit(Xt, code, epochs, batch):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    enc, dec = make_ae(code)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=4e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=gen).split(batch):
            x = Xt[idx]
            loss = F.binary_cross_entropy(dec(enc(x)), x, reduction="sum") / len(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc, dec


# ---------------- anomaly detection: train on nine digits, flag the tenth by reconstruction error ----------------
print("held-out digit:  AUROC of reconstruction error (1.0 = perfect, 0.5 = chance)")
aucs = []
for c in range(10):
    enc, dec = fit(T_tr[torch.tensor(y_tr != c)], code=8, epochs=30, batch=128)
    with torch.no_grad():
        err = ((dec(enc(T_te)) - T_te) ** 2).mean(dim=1).numpy()
    aucs.append(roc_auc_score(y_te == c, err))
print("  " + "  ".join(f"{c}:{a:.2f}" for c, a in enumerate(aucs)), f"  mean {np.mean(aucs):.3f}")`;
const AE_ANOMALY_OUT = String.raw`held-out digit:  AUROC of reconstruction error (1.0 = perfect, 0.5 = chance)
  0:0.98  1:0.83  2:0.97  3:0.78  4:0.96  5:0.94  6:0.93  7:0.95  8:0.91  9:0.81   mean 0.907`;
const AE_HOLES = String.raw`import numpy as np
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


def make_ae(code):
    enc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, code))
    dec = nn.Sequential(nn.Linear(code, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())
    return enc, dec


def fit(Xt, code, epochs, batch):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    enc, dec = make_ae(code)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=4e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=gen).split(batch):
            x = Xt[idx]
            loss = F.binary_cross_entropy(dec(enc(x)), x, reduction="sum") / len(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc, dec


# ---------------- holes: decode random points of a 2-D latent space ----------------
enc, dec = fit(T_tr, code=2, epochs=150, batch=128)
clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)        # a judge of "does this look like a digit?"
print(f"classifier test accuracy on real digits: {clf.score(X_te, y_te):.3f}")
with torch.no_grad():
    Z = enc(T_tr).numpy()
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    rng = np.random.default_rng(0)
    Z_rand = lo + (hi - lo) * rng.random((1000, 2))           # uniform in the bounding box of the codes
    rec = dec(enc(T_te)).numpy()
    rnd = dec(torch.tensor(Z_rand, dtype=torch.float32)).numpy()
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    Z_far = centre + 3 * half * (2 * rng.random((1000, 2)) - 1)  # a box 3x wider: mostly far from all data
    far = dec(torch.tensor(Z_far, dtype=torch.float32)).numpy()


def nearest_code_gap(Zq):
    d = np.sqrt(((Zq[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1)


gap_train = np.sort(np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)) + np.eye(len(Z)) * 1e9, axis=1)[:, 0]
print(f"latent bounding box: z1 in [{lo[0]:.1f}, {hi[0]:.1f}], z2 in [{lo[1]:.1f}, {hi[1]:.1f}]")
print(f"median distance code -> nearest other training code: {np.median(gap_train):.3f}")
print(f"median distance random point -> nearest training code: {np.median(nearest_code_gap(Z_rand)):.3f}")
print(f"median distance far point -> nearest training code: {np.median(nearest_code_gap(Z_far)):.3f}")
print(f"{'decoded images':<26}{'confidence':>11}{'conf > 0.9':>11}")
for name, imgs in [("test reconstructions", rec), ("random z in the box", rnd), ("random z in a 3x box", far)]:
    conf = clf.predict_proba(imgs).max(axis=1)
    print(f"{name:<26}{conf.mean():>11.3f}{np.mean(conf > 0.9):>11.3f}")
conf_real = clf.predict_proba(X_te).max(axis=1)
print(f"{'real test images':<26}{conf_real.mean():>11.3f}{np.mean(conf_real > 0.9):>11.3f}")`;
const AE_HOLES_OUT = String.raw`classifier test accuracy on real digits: 0.967
latent bounding box: z1 in [-4.3, 17.5], z2 in [-14.9, 9.0]
median distance code -> nearest other training code: 0.118
median distance random point -> nearest training code: 0.819
median distance far point -> nearest training code: 16.302
decoded images             confidence conf > 0.9
test reconstructions            0.890      0.681
random z in the box             0.868      0.609
random z in a 3x box            0.828      0.524
real test images                0.903      0.725`;

const AE_KERAS = String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits

X = load_digits().data.astype("float32") / 16.0          # (1797, 64) pixels in [0, 1]

inputs = tf.keras.Input(shape=(64,))
h = tf.keras.layers.Dense(64, activation="relu")(inputs)
h = tf.keras.layers.Dense(32, activation="relu")(h)
code = tf.keras.layers.Dense(2, name="code")(h)          # linear 2-D bottleneck
d = tf.keras.layers.Dense(32, activation="relu")(code)
d = tf.keras.layers.Dense(64, activation="relu")(d)
outputs = tf.keras.layers.Dense(64, activation="sigmoid")(d)

autoencoder = tf.keras.Model(inputs, outputs)
encoder = tf.keras.Model(inputs, code)                   # shares the trained layers
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="binary_crossentropy")
autoencoder.fit(X, X, epochs=100, batch_size=64, validation_split=0.2, verbose=0)  # target = input
print("validation loss:", autoencoder.evaluate(X[-360:], X[-360:], verbose=0))
print("first codes:", encoder.predict(X[:3], verbose=0))`;

const lesson: Lesson = {
  id: "m14/autoencoders",
  title: "Autoencoders",
  summary:
    "Networks trained to reproduce their own input through a narrow bottleneck: encoder, decoder and latent space; MSE vs BCE reconstruction losses; why a linear autoencoder rediscovers PCA; sparse, denoising and contractive regularisation; anomaly detection — and why a plain autoencoder is not a generative model.",
  minutes: 70,
  level: "intermediate",
  prerequisites: ["m04/backpropagation", "m02/eigen-svd", "m05/advanced-losses", "m07/regularization"],
  next: ["m14/vae", "m15/self-supervised", "m14/diffusion"],
  terms: ["autoencoder", "encoder", "decoder", "latent-space", "bottleneck", "reconstruction-loss", "undercomplete-autoencoder", "sparse-autoencoder", "denoising-autoencoder", "contractive-autoencoder"],
  sections: {
    intro: [
      {
        t: "p",
        text: String.raw`An **autoencoder** is a neural network trained to output its own input. It has two parts. The **encoder** $f$ squeezes an input $x$ into a **code** (also called the **latent representation**) $z = f(x)$. The **decoder** $g$ turns the code back into a **reconstruction** $\hat x = g(z)$. Training minimises a **reconstruction loss** $\mathcal{L}(x, \hat x)$: how different the output is from the input. No labels are needed — the data is its own target — which makes autoencoders one of the oldest forms of **self-supervised learning**.`,
      },
      {
        t: "p",
        text: "Copying is too easy, so we make it hard on purpose. The classic way is a **bottleneck**: the code has far fewer numbers than the input (an **undercomplete** autoencoder). The network must then find a compact description of the data — which pixels move together, which shapes occur. Other ways: penalise the code (sparse and contractive autoencoders), or corrupt the input and ask for the clean original (denoising autoencoders).",
      },
      {
        t: "p",
        text: "The idea goes back to the 1980s (Rumelhart, Hinton and Williams; Baldi and Hornik's 1989 analysis of linear autoencoders). Hinton and Salakhutdinov's 2006 *Science* paper \"Reducing the dimensionality of data with neural networks\" showed deep autoencoders compressing images far better than PCA, and they became a way to **pretrain** deep networks layer by layer. Today they serve **dimensionality reduction**, **denoising**, **anomaly detection** (inputs that reconstruct badly are unusual), as **compression stages** inside bigger generative models (Stable Diffusion works in an autoencoder's latent space), and — in probabilistic form, the [variational autoencoder](/learn/m14/vae) — as generative models.",
      },
      {
        t: "callout",
        kind: "key",
        text: String.raw`Autoencoder = encoder $f$ + decoder $g$ trained so that $g(f(x)) \approx x$, plus a constraint (bottleneck, sparsity, noise or a Jacobian penalty) that blocks plain copying. Afterwards you usually keep the encoder (as a feature extractor) or the reconstruction error (as an anomaly score).`,
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Backpropagation](/learn/m04/backpropagation) — an autoencoder is an ordinary feed-forward network trained by gradient descent; nothing new is needed.",
          "[Eigendecomposition, SVD and PCA](/learn/m02/eigen-svd) — PCA as the best rank-$k$ linear reconstruction (Eckart–Young). The linear-autoencoder result here depends on it.",
          "[Advanced losses](/learn/m05/advanced-losses) and [regression losses](/learn/m05/regression-losses) — MSE as a Gaussian negative log-likelihood and binary cross-entropy as a Bernoulli one: the two reconstruction losses.",
          "[Regularization](/learn/m07/regularization) — L1 penalties create sparsity; sparse and contractive autoencoders are regularised autoencoders.",
          String.raw`[Jacobians](/learn/m02/jacobians-hessians) — the contractive autoencoder penalises the Frobenius norm of the encoder's Jacobian $\partial z/\partial x$.`,
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Describing a picture over the phone",
        text: "You must describe a handwritten digit to a friend who will redraw it, using only two numbers. You agree on a system in advance: the first number roughly means \"round versus straight\", the second \"how slanted\". Your half is the encoder, your friend's the decoder, the two numbers the code. Practice on many digits (training) improves the system until the redrawings are as good as two numbers allow. Nobody told you what the numbers should mean — the system invented its own vocabulary.",
      },
      {
        t: "p",
        text: "The code is too small to store everything, so the autoencoder keeps what many inputs **share** (the overall shape of a 7) and drops what is **unique** (a wobble in one stroke). That is why reconstructions look like smoothed, typical versions of the inputs, and why the code is useful: similar inputs get similar codes.",
      },
      {
        t: "p",
        text: "The same picture shows the main weakness. Your friend only ever heard number pairs you actually used. Ask them to draw \"(37, −40)\", a pair never used, and they will draw *something* — but nothing guarantees it is a digit. A plain autoencoder's decoder is good *at the codes of real data* and untrained elsewhere: its latent space has **holes**.",
      },
    ],
    mechanism: [
      {
        t: "steps",
        items: [
          { title: "Choose the architecture", text: "Encoder: layers narrowing from input size $D$ to code size $k$. Decoder: usually a mirror image widening from $k$ back to $D$. For images, convolutional encoders and transposed-convolution decoders; here, small MLPs on 64-pixel digits." },
          { title: "Pick the output activation and loss together", text: "Pixels in [0, 1] → sigmoid output with binary cross-entropy (a Bernoulli likelihood). Unbounded real values → linear output with MSE (a Gaussian likelihood)." },
          { title: "Forward pass", text: String.raw`Compute the code $z = f_\phi(x)$ (often with a linear last encoder layer, so the code can be any real value), then $\hat x = g_\theta(z)$.` },
          { title: "Reconstruction loss", text: String.raw`Compare $\hat x$ with the clean input $x$. (A denoising AE's encoder saw a corrupted $\tilde x$, but the target is still $x$.)`, tex: String.raw`\mathcal{L}_{\text{rec}} = \frac1N\sum_{n=1}^{N} \ell\big(x_n,\ g_\theta(f_\phi(x_n))\big)` },
          { title: "Add a regulariser if the code is not a bottleneck", text: String.raw`A sparsity penalty on the code, a Jacobian penalty, or input noise. Its weight $\lambda$ is a hyper-parameter.` },
          { title: "Backpropagate through decoder and encoder", text: "One loss trains both together: the decoder learns to invert whatever code the encoder makes, and the encoder learns to make codes the decoder can invert." },
          { title: "Use the trained parts", text: "Encoder → features for plots, clustering or a classifier. Reconstruction error → anomaly score. Encoder + decoder → a denoiser or compressor." },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        text: "The code size $k$ is the most important hyper-parameter. Too small, and reconstructions lose essential detail. Too large (and unregularised), and the network can approach the identity map and learn nothing useful. Below, an overcomplete AE (128 hidden units for 64 pixels) reaches a clean test MSE of 0.0027 but is the **worst** model on corrupted inputs — it learned to copy, noise included.",
      },
    ],
    math: [
      { t: "h", text: "Reconstruction losses are negative log-likelihoods" },
      {
        t: "p",
        text: String.raw`Treat the decoder output as the parameter of a distribution over pixels, $p_\theta(x \mid z)$, and minimise $-\log p_\theta(x\mid z)$. A Gaussian with fixed variance $\sigma^2$ per pixel gives $-\log p = \frac{1}{2\sigma^2}\sum_j (x_j - \hat x_j)^2 + \text{const}$, which is **MSE**. Independent Bernoulli pixels with probability $\hat x_j$ give:`,
      },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{BCE}}(x, \hat x) = -\sum_{j=1}^{D}\Big[x_j\log\hat x_j + (1-x_j)\log(1-\hat x_j)\Big]`, caption: String.raw`$x_j \in [0,1]$ is the target pixel, $\hat x_j \in (0,1)$ the sigmoid output, $D$ the number of pixels. Grey levels act as soft targets.` },
      {
        t: "p",
        text: String.raw`**Worked example** (from the second From-Scratch program): $x = (1, 0, 0.5, 0.25)$, $\hat x = (0.9, 0.2, 0.4, 0.3)$. MSE $= \frac14(0.01 + 0.04 + 0.01 + 0.0025) = 0.015625$. Summed BCE $= -\ln 0.9 - \ln 0.8 - [0.5\ln0.4 + 0.5\ln0.6] - [0.25\ln0.3 + 0.75\ln0.7] = 1.610562$ nats.`,
      },
      {
        t: "p",
        text: String.raw`With soft targets BCE can never reach 0. Its minimum, at $\hat x = x$, is the entropy of the targets: $1.255482$ here. So report MSE (or PSNR) for image quality and use BCE for training. BCE's gradient with respect to the pre-sigmoid logit is just $\hat x_j - x_j$, which does not vanish when the sigmoid saturates.`,
      },
      { t: "h", text: "The linear autoencoder learns the PCA subspace" },
      {
        t: "p",
        text: String.raw`Let the data be centred and stacked as rows of $X \in \mathbb{R}^{N\times D}$, with a linear encoder $z = W_e^\top x$ ($W_e \in \mathbb{R}^{D\times k}$) and decoder $\hat x = W_d^\top z$ ($W_d \in \mathbb{R}^{k\times D}$). The reconstruction $\hat X = X W_e W_d$ has rank at most $k$, so`,
      },
      { t: "math", tex: String.raw`\min_{W_e, W_d}\ \big\|X - XW_eW_d\big\|_F^2\ \ \ge\ \ \min_{\operatorname{rank}(M)\le k}\|X - M\|_F^2 = \sum_{i>k}s_i^2`, caption: String.raw`$s_i$ are the singular values of $X$. The right side is the Eckart–Young bound: the best any rank-$k$ approximation can do, reached by keeping the top $k$ terms of the SVD $X = USV^\top$.` },
      {
        t: "p",
        text: String.raw`The autoencoder can reach this bound: take $W_e = V_k$ (the top-$k$ right singular vectors, i.e. principal directions) and $W_d = V_k^\top$, which gives $XV_kV_k^\top$, the PCA reconstruction. Baldi and Hornik (1989) also showed that a linear autoencoder's MSE surface has no bad local minima: every local minimum is global and projects onto the top-$k$ principal subspace.`,
      },
      {
        t: "p",
        text: String.raw`But the answer is not unique. For any invertible $k\times k$ matrix $A$, $W_eA$ and $A^{-1}W_d$ give the same reconstruction. So a linear AE recovers the **subspace** spanned by the principal components, **not** the components themselves: its decoder rows are generally not orthogonal, not unit length, and not ordered by variance.`,
      },
      {
        t: "p",
        text: String.raw`**In numbers** (first From-Scratch program, digits, $k = 8$): PCA's test MSE is $0.02465$, and the linear AE trained by plain gradient descent reaches exactly $0.02465$. The cosines of the principal angles between the subspaces are all $1.0000$ except the last ($0.9997$, still converging), and $\max|W_dW_d^\top - I| = 0.161$: not an orthonormal basis. Non-linearities break the tie: a tanh autoencoder with the same 8-number code reaches test MSE $0.01644$ after 100 epochs, a third lower than any linear method can.`,
      },
      { t: "h", text: "Regularised autoencoders" },
      {
        t: "p",
        text: String.raw`**Sparse autoencoder.** Allow a wide code, but require most code units to be off for any given input. With an L1 penalty on the code $h = f(x)$: $\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda\sum_j|h_j|$. With sigmoid code units, a common alternative (Ng's sparse-autoencoder notes) pushes each unit's **average** activation $\hat\rho_j = \frac1N\sum_n h_j(x_n)$ towards a small target $\rho$ with a KL penalty between Bernoulli distributions:`,
      },
      { t: "math", tex: String.raw`\Omega_{\text{sparse}} = \sum_j \mathrm{KL}(\rho\,\|\,\hat\rho_j) = \sum_j\Big[\rho\log\frac{\rho}{\hat\rho_j} + (1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}\Big]` },
      {
        t: "p",
        text: String.raw`**Denoising autoencoder** (Vincent, Larochelle, Bengio and Manzagol, 2008, "Extracting and composing robust features with denoising autoencoders"). Corrupt each input with a random process $\tilde x \sim C(\tilde x\mid x)$ — they used **masking noise** (a random fraction of inputs set to 0); added Gaussian noise is also common — and train $g(f(\tilde x))$ to reproduce the **clean** $x$: $\mathcal{L} = \mathbb{E}_{x}\mathbb{E}_{\tilde x\sim C}\,\ell(x, g(f(\tilde x)))$. Copying no longer works, because the input has errors the target lacks. The network must learn what plausible images look like and pull corrupted points back towards them.`,
      },
      {
        t: "p",
        text: String.raw`A deep link: for small Gaussian noise, Alain and Bengio (2014) showed that the best denoiser's correction $g(f(\tilde x)) - \tilde x$ approximates $\sigma^2\nabla_x\log p(x)$ — the *score* of the data distribution, the quantity [diffusion models](/learn/m14/diffusion) learn.`,
      },
      {
        t: "p",
        text: String.raw`**Contractive autoencoder** (Rifai, Vincent, Muller, Glorot and Bengio, 2011, "Contractive auto-encoders: explicit invariance during feature extraction"). Penalise how strongly the code reacts to tiny input changes: the squared Frobenius norm (sum of squared entries) of the encoder Jacobian $J_f(x) = \partial h/\partial x \in \mathbb{R}^{k\times D}$:`,
      },
      { t: "math", tex: String.raw`\mathcal{L}_{\text{CAE}} = \mathcal{L}_{\text{rec}} + \lambda\,\big\|J_f(x)\big\|_F^2, \qquad \big\|J_f(x)\big\|_F^2 = \sum_{j=1}^{k}\sum_{i=1}^{D}\Big(\frac{\partial h_j}{\partial x_i}\Big)^2` },
      {
        t: "p",
        text: String.raw`For one sigmoid layer $h = \sigma(Wx + b)$, $\partial h_j/\partial x_i = h_j(1-h_j)W_{ji}$, so the penalty has a cheap closed form, $\|J_f\|_F^2 = \sum_j \big(h_j(1-h_j)\big)^2\,\|W_{j,:}\|^2$, with no explicit Jacobian. The reconstruction term wants the code to change where the data really varies; the penalty wants it flat everywhere. The compromise: an encoder sensitive only along the data manifold.`,
      },
      {
        t: "steps",
        items: [
          { title: "Contractive penalty, worked example", text: String.raw`$W = \begin{pmatrix}0.5 & -1 & 2\\ 1.5 & 0.3 & -0.7\end{pmatrix}$, $b = (0.1, -0.2)$, $x = (0.2, 0.4, 0.6)$.`, tex: String.raw`a = Wx + b = (1.0,\ -0.2),\quad h = \sigma(a) = (0.731059,\ 0.450166)` },
          { title: "Local slopes and row norms", tex: String.raw`h(1-h) = (0.196612,\ 0.247517),\qquad \|W_{1,:}\|^2 = 5.25,\ \ \|W_{2,:}\|^2 = 2.83` },
          { title: "Jacobian", tex: String.raw`J = \operatorname{diag}\big(h(1-h)\big)W = \begin{pmatrix}0.098306 & -0.196612 & 0.393224\\ 0.371275 & 0.074255 & -0.173262\end{pmatrix}` },
          { title: "Penalty", tex: String.raw`\|J\|_F^2 = 0.196612^2\cdot5.25 + 0.247517^2\cdot2.83 = 0.202946 + 0.173378 = 0.376324` },
        ],
      },
      { t: "p", text: "The program checks this against the sum of squares of the explicit matrix and against central finite differences: all three give $0.376324$." },
    ],
    visual: [
      {
        t: "p",
        text: "The simulator runs a **real autoencoder** in your browser: 64→64→32→**2**→32→64→64, trained with PyTorch on 1,437 of scikit-learn's 8×8 digits (the 360 test digits were never seen). A 2-D code is far too small for good reconstructions, but it lets us *see* the latent space.",
      },
      { t: "sim", id: "autoencoder", caption: "Step through encoder → bottleneck → decoder for any test digit; click anywhere in the latent plane to run the real decoder there." },
      {
        t: "list",
        items: [
          "Press **Step** through the seven stages. At each layer, count the ReLU units that are exactly 0 — often a large share. At the bottleneck the whole image is two numbers.",
          "Compare input and reconstruction for several digits (**Another digit**). The reconstruction is a smoothed prototype; the error map (accent = too bright, clay = too dark) gathers on stroke edges. Find a digit rebuilt as a *different* digit — it sits in the wrong cluster of the scatter.",
          "In the scatter, digits form clusters even though labels were never used. Note how unevenly they fill the plane: dense, overlapping clouds near a centre and long sparse arms.",
          "Click in empty space between clusters, then **zoom out ×3** and click far away. The panel shows your distance to the nearest training code, in units of the typical gap between neighbouring codes. Switch to **decoded map**: far from the data the decoder still outputs *something* — hard, blocky, over-bright strokes — because it was never trained there.",
          "Raise **Input noise σ** to 0.3 and compare the table: the plain AE's error jumps, while the **Denoising AE** (trained on noisy inputs) barely changes. On the whole test set (PyTorch) at σ = 0.3: noisy input MSE 0.051, plain AE output 0.068 (worse than doing nothing), denoising AE 0.037.",
        ],
      },
    ],
    scratch: [
      {
        t: "p",
        text: "Two NumPy programs. The first fits PCA, trains a **linear autoencoder** by gradient descent and compares the two subspaces, then trains a **non-linear autoencoder** with hand-written backpropagation and Adam. The second computes the reconstruction losses and contractive penalty from the worked examples.",
      },
      { t: "code", lang: "python", title: "autoencoder_from_scratch.py", code: AE_NUMPY, output: AE_NUMPY_OUT },
      {
        t: "list",
        items: [
          "The linear AE matches PCA's test MSE (0.02465), and its train MSE (0.02379) converges to PCA's optimum (0.02378), as Eckart–Young predicts — but `Wd` is not orthonormal: the AE finds the subspace, not the basis.",
          String.raw`The step size is multiplied by $D$ because the loss averages over all $N\times D$ entries; otherwise the raw gradients are tiny and plain gradient descent would need far more steps.`,
          "In the non-linear network the bottleneck layer is **linear** (no tanh), as is common, so the code can take any real value. `delta = (acts[-1] - Xb) / len(Xb)` is the sigmoid + summed-BCE shortcut from [backpropagation](/learn/m04/backpropagation).",
        ],
      },
      { t: "code", lang: "python", title: "losses_and_contractive_penalty.py", code: CAE_NUMPY, output: CAE_NUMPY_OUT },
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
                text: String.raw`**Experiment 1 — the regularisers compared.** Five one-hidden-layer autoencoders (sigmoid code, sigmoid output, summed BCE, 100 epochs) on the same split. *Clean MSE* is ordinary test reconstruction error. *Masked MSE* feeds test images with 25% of pixels zeroed and compares with the clean images. *Mean h* is the average code activation; $\|J\|_F$ the average encoder-Jacobian norm.`,
              },
              { t: "code", lang: "python", title: "autoencoder_variants.py", code: AE_TORCH, output: AE_TORCH_OUT },
              {
                t: "list",
                items: [
                  "**Overcomplete, unregularised**: best clean MSE (0.0027) but worst masked MSE (0.0495, near the 0.0578 of copying the corrupted input) — it learned something close to the identity.",
                  String.raw`**Sparse** ($\rho = 0.05$): mean activation 0.055, right at the target; each input is described by a few active units out of 128.`,
                  "**Denoising**: by far the best on corrupted inputs (0.0193), and *also* better than the undercomplete AE on clean data.",
                  "**Contractive**: the smallest Jacobian norm (1.87 vs 11.47 unregularised) at a modest cost in clean MSE — the code barely reacts to small input changes.",
                ],
              },
              {
                t: "p",
                text: "**Experiment 2 — anomaly detection.** Train on nine digit classes, then score every test image by its reconstruction error; the held-out class plays the anomaly. The AUROC (area under the ROC curve) is the chance that a random anomaly scores higher than a random normal image.",
              },
              { t: "code", lang: "python", title: "autoencoder_anomaly_detection.py", code: AE_ANOMALY, output: AE_ANOMALY_OUT },
              {
                t: "p",
                text: "Mean AUROC is 0.907, but very uneven. A new 0 or 2 is caught almost perfectly, while unseen 3s and 9s are often rebuilt well by an AE that learned 8s, 5s and 7s — their strokes are combinations of familiar parts. Reconstruction error measures *unfamiliarity to this model*, which only partly matches what a human calls an anomaly.",
              },
              {
                t: "p",
                text: "**Experiment 3 — holes in the latent space.** A 2-D AE is trained on all digits. Points are decoded at random inside the bounding box of its training codes and in a box three times wider. A logistic-regression classifier trained on real digits (96.7% test accuracy) judges how digit-like the outputs are.",
              },
              { t: "code", lang: "python", title: "autoencoder_latent_holes.py", code: AE_HOLES, output: AE_HOLES_OUT },
              {
                t: "p",
                text: "A random point in the box lies a median 0.82 units from the nearest training code — about seven times the typical code-to-code gap of 0.12 — and far points lie 16 units away. The judge's confidence falls steadily from reconstructions to random points to far points. The decoder makes *an* image everywhere, but was never trained to make it a plausible digit. And nothing tells us *where* to sample: the codes fill an arbitrary, irregular region whose position and scale change between training runs.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "The Functional API makes the shared encoder explicit: `autoencoder` and `encoder` are two `Model`s built from the same layer objects, so training one trains the other. Note `fit(X, X)` — the input is its own target." },
              { t: "code", lang: "python", title: "autoencoder_keras.py (not executed in CI)", code: AE_KERAS, output: "Prints the validation binary cross-entropy (mean over pixels) after 100 epochs and the 2-D codes of the first three digits." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Variant", "How copying is prevented", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Undercomplete AE", "code size $k \\ll D$", "simple; compact codes; a non-linear PCA", "choosing $k$; reconstructions lose detail", "dimensionality reduction, visualisation"],
          ["Linear AE", "bottleneck, no non-linearity", "provably recovers the PCA subspace", "no better than PCA (and slower)", "teaching; analysis"],
          ["Sparse AE", "L1 or KL($\\rho\\|\\hat\\rho$) penalty on activations", "wide, interpretable, parts-based codes", "extra hyper-parameters ($\\lambda$, $\\rho$)", "feature learning; interpreting activations of large models"],
          ["Denoising AE", "corrupted input, clean target", "robust features; a learned denoiser; links to score estimation", "noise type and level must suit the data", "denoising; pretraining; BERT-style masked modelling descends from it"],
          ["Contractive AE", "Frobenius norm of encoder Jacobian", "codes that ignore small input changes", "Jacobian cost grows with depth (closed form only for simple layers)", "robust representation learning"],
          ["Convolutional AE", "bottleneck in a conv encoder/decoder", "uses image structure; far fewer parameters", "transposed convolutions can cause checkerboard artifacts", "image compression, denoising, latent spaces for diffusion"],
          ["Variational AE", "probabilistic code with a KL penalty to a prior", "a real generative model with a likelihood bound", "blurrier samples; posterior collapse", "generation, representation learning — see [VAE](/learn/m14/vae)"],
          ["VQ-VAE / discrete AE", "code snapped to a learned codebook", "discrete tokens a Transformer can model", "codebook collapse; straight-through gradients", "image and audio tokenizers"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Dimensionality reduction and visualisation** — a non-linear alternative to PCA (Hinton and Salakhutdinov, 2006): 2-D or 3-D codes for plots, small codes as features for clustering.",
          "**Anomaly and fraud detection** — train on normal transactions, sensor readings or machine sounds; flag inputs with large reconstruction error. Works best when anomalies differ in structure from normal data (see the uneven per-class AUROC above).",
          "**Denoising and restoration** — denoising autoencoders and their convolutional descendants clean images, audio and scientific measurements.",
          "**Learned compression** — the encoder makes a compact code, the decoder rebuilds; neural image codecs add quantisation and an entropy model.",
          "**Latent spaces for generative models** — latent diffusion (Stable Diffusion) first trains a convolutional autoencoder, then runs diffusion in its smaller latent space; see [Diffusion Models](/learn/m14/diffusion).",
          "**Pretraining and self-supervision** — stacked (denoising) autoencoders once initialised deep networks layer by layer; today masked autoencoders (MAE) and masked language modelling follow the same corrupt-and-reconstruct recipe — see [Self-Supervised Learning](/learn/m15/self-supervised).",
          "**Interpretability** — sparse autoencoders trained on the internal activations of large language models help find more interpretable features.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**No labels needed**: any unlabelled data works, and the encoder often transfers to other tasks.",
          "**Non-linear and flexible**: strictly more expressive than PCA (8-number codes: test MSE ≈ 0.016 vs 0.025 here).",
          "**The loss decides what matters**: pixel-wise MSE/BCE treat pixels independently, so reconstructions are blurry and small shifts are punished hard; perceptual losses help.",
          "**Codes are not unique or interpretable by default**: any invertible transform of the code, undone in the decoder, gives the same loss; single latent dimensions have no guaranteed meaning.",
          "**Not a generative model**: there is no distribution over codes to sample from, the occupied region is irregular, and the decoder is untrained between and beyond the codes (holes).",
          "**Good reconstruction ≠ good features**: an overcomplete AE reconstructs best and learns least; the regulariser, not the reconstruction error, decides usefulness.",
          "**Anomaly detection is fragile**: autoencoders can generalise well enough to rebuild some anomalies too (AUROC 0.78 for held-out 3s).",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Judging an autoencoder by training reconstruction error alone.** A big enough AE drives it towards 0 by copying. Evaluate on held-out data and on the real use (features, denoising, anomaly AUROC).",
          "**Sigmoid output with MSE on [0,1] data, or a linear output with BCE.** The pairing matters: BCE needs probabilities (use `BCEWithLogitsLoss` on logits for stability); MSE goes with a linear output. Sigmoid + MSE trains, but its gradients vanish when outputs saturate.",
          "**Reporting BCE as image error for grey-level targets.** Its minimum is the targets' entropy, not 0 (1.255 nats in the worked example). Report MSE/PSNR.",
          "**Expecting a linear AE to return the principal components.** It returns a basis of the same subspace, rotated and scaled arbitrarily. Use PCA if you need the components.",
          "**Adding noise at test time in a denoising AE when you want features.** Corruption is a training device; at inference, feed clean (or the actual noisy) inputs without adding more.",
          "**Sampling random codes from a plain autoencoder and calling it generation.** The codes follow no known distribution, and random points often land in holes. Use a VAE (or fit a density to the codes).",
          "**Normalising inputs inconsistently** — training on [0, 1] pixels and testing on [0, 255] makes every test input an \"anomaly\".",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is an autoencoder and how is it trained?", a: String.raw`A network made of an encoder $z = f(x)$ and a decoder $\hat x = g(z)$, trained by gradient descent to minimise a reconstruction loss between $\hat x$ and $x$ (MSE or BCE). The input is its own target, so no labels are needed.` },
      { t: "qa", level: "beginner", q: "Why must an autoencoder be constrained?", a: "Without a constraint it can learn the identity function, which reconstructs perfectly and learns nothing. Constraints: a bottleneck smaller than the input, a sparsity or Jacobian penalty on the code, or corrupting the input (denoising)." },
      { t: "qa", level: "beginner", q: "Give three uses of autoencoders.", a: "Dimensionality reduction and visualisation (use the code), denoising (train noisy → clean), anomaly detection (large reconstruction error = unusual input). Also compression, pretraining, and latent spaces for diffusion models." },
      { t: "qa", level: "intermediate", q: "What does a linear autoencoder trained with MSE learn?", a: String.raw`The subspace spanned by the top-$k$ principal components: its best reconstruction equals PCA's rank-$k$ reconstruction (Eckart–Young), and all local minima are global (Baldi and Hornik, 1989). The weights are fixed only up to an invertible $k\times k$ transform, so it does not recover the orthonormal, variance-ordered components themselves.` },
      { t: "qa", level: "intermediate", q: "MSE or binary cross-entropy for the reconstruction loss?", a: String.raw`Choose by the data's likelihood model. MSE is the negative log-likelihood of a fixed-variance Gaussian (real-valued data, linear output); BCE is that of independent Bernoullis (values in [0,1], sigmoid output). BCE gives the logit gradient $\hat x - x$, which does not vanish under saturation. With grey-level targets BCE's minimum is the target entropy, so report quality with MSE/PSNR.` },
      { t: "qa", level: "intermediate", q: "How does a denoising autoencoder differ from a plain one, and why does it learn better features?", a: "The input is corrupted (masking or Gaussian noise) but the target is the clean input. Copying no longer minimises the loss, so the network must learn the structure of plausible inputs to undo the damage. In our experiment it had the lowest corrupted-input error (0.019 vs 0.050 for an unregularised overcomplete AE) and even beat the undercomplete AE on clean data." },
      { t: "qa", level: "intermediate", q: "How would you use an autoencoder for anomaly detection, and what can go wrong?", a: "Train on normal data only, score new data by reconstruction error, and pick a threshold on a validation set (or report AUROC). Failure modes: the AE generalises and rebuilds anomalies well (held-out 3s gave AUROC 0.78); anomalies lie in features the loss ignores; anomalies contaminate the training data; the threshold is set without representative validation data." },
      { t: "qa", level: "advanced", q: "Derive the contractive penalty for a sigmoid encoder layer and explain what it encourages.", a: String.raw`$h = \sigma(Wx+b)$ gives $\partial h_j/\partial x_i = h_j(1-h_j)W_{ji}$, so $\|J\|_F^2 = \sum_j (h_j(1-h_j))^2\|W_{j,:}\|^2$. On its own, minimising it would make the code constant; the reconstruction loss needs the code to vary where the data varies. The balance gives an encoder sensitive only along the data manifold and unmoved by perpendicular nudges.` },
      { t: "qa", level: "advanced", q: "Why is a plain autoencoder a poor generative model, and what fixes it?", a: String.raw`There is no distribution over codes. Training codes fill an arbitrary irregular region and the decoder is trained only there, so random latent points can fall in holes and decode to implausible images (in our 2-D experiment, a random point in the bounding box was ~7× the typical code gap from any data). A VAE fixes this: the encoder outputs a distribution, noise is added in training, and a KL term pulls all codes towards a fixed prior $\mathcal N(0, I)$ — a known distribution to sample, and a decoder trained across its whole region.` },
      { t: "qa", level: "advanced", q: "What is the connection between denoising autoencoders and score-based / diffusion models?", a: String.raw`For small Gaussian corruption, the best denoiser satisfies $r(\tilde x) - \tilde x \approx \sigma^2 \nabla_x \log p(x)$ (Alain and Bengio, 2014; Vincent, 2011 relates denoising score matching to DAEs). So training a denoiser estimates the score of the data distribution. Diffusion models train denoisers at many noise levels and generate by denoising step by step.` },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "An autoencoder with a 128-unit code on 64-pixel images and no regularisation achieves near-zero test reconstruction error. What is the most likely problem?", options: ["It is underfitting", "It has learned (close to) the identity map, so its code is not a useful compression", "The learning rate is too small", "BCE cannot be used with codes larger than the input"], answer: 1, explain: "An overcomplete, unregularised AE can copy its input. In the experiment it had the best clean error and the worst error on corrupted inputs." },
          { kind: "numeric", q: String.raw`Target pixels $x = (1, 0)$, reconstruction $\hat x = (0.8, 0.1)$. What is the mean squared error?`, answer: 0.025, tolerance: 0.0001, explain: String.raw`$\frac12\big((0.2)^2 + (0.1)^2\big) = \frac12(0.04 + 0.01) = 0.025$.` },
          { kind: "numeric", q: "Same example: what is the summed binary cross-entropy in nats (3 decimals)?", answer: 0.329, tolerance: 0.002, explain: String.raw`$-\ln 0.8 - \ln 0.9 = 0.223144 + 0.105361 = 0.328504$.` },
          { q: "A linear autoencoder with code size $k$ is trained to convergence with MSE on centred data. Which statement is true?", options: ["Its decoder rows equal the top-$k$ principal components", "It spans the same subspace as the top-$k$ principal components, with the same reconstruction error as PCA", "It always beats PCA", "It cannot be trained by gradient descent"], answer: 1, explain: String.raw`Same subspace and same error, but the basis is arbitrary up to an invertible $k\times k$ transform.` },
          { q: "In a denoising autoencoder, what is the training target?", options: ["The corrupted input", "The clean input", "The noise", "The code of the clean input"], answer: 1, explain: String.raw`The encoder sees $\tilde x$, but the loss compares $g(f(\tilde x))$ with the clean $x$.` },
          { kind: "numeric", q: String.raw`A sigmoid code unit has $h = 0.5$ and incoming weight vector with $\|W_j\|^2 = 4$. What is its contribution to $\|J\|_F^2$?`, answer: 0.25, tolerance: 0.0001, explain: String.raw`$(h(1-h))^2\|W_j\|^2 = 0.25^2\times4 = 0.0625\times4 = 0.25$.` },
          { q: "Why is sampling random codes from a trained plain autoencoder unreliable?", options: ["The decoder is non-differentiable", "The codes follow no known distribution and the decoder is only trained near the codes of real data", "The encoder is random", "Autoencoders cannot decode new codes"], answer: 1, explain: "The latent space has holes; a VAE adds a prior and noise to fix this." },
          { kind: "numeric", q: String.raw`Sparse AE with target $\rho = 0.05$: a unit has average activation $\hat\rho = 0.2$. Compute $\mathrm{KL}(\rho\|\hat\rho)$ in nats (3 decimals).`, answer: 0.0939, tolerance: 0.001, explain: String.raw`$0.05\ln(0.05/0.2) + 0.95\ln(0.95/0.8) = 0.05\ln 0.25 + 0.95\ln 1.1875 = -0.069315 + 0.163257 = 0.093943$ nats. The penalty grows quickly as $\hat\rho$ moves away from $\rho$.` },
        ],
      },
      {
        t: "exercise",
        title: "Contractive penalty by hand",
        prompt: String.raw`A sigmoid encoder has $W = \begin{pmatrix}1 & 0\\ 2 & -1\end{pmatrix}$, $b = (0, 0)$. For $x = (0, 0)$ compute $h$, the Jacobian $\partial h/\partial x$ and $\|J\|_F^2$. Then repeat for $x = (3, 0)$ and explain the difference.`,
        hint: "At $x = 0$ every unit has $h = 0.5$, where $h(1-h)$ is largest.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "x = (0, 0)", tex: String.raw`h = (0.5,\ 0.5),\quad h(1-h) = (0.25,\ 0.25),\quad J = \begin{pmatrix}0.25 & 0\\ 0.5 & -0.25\end{pmatrix},\quad \|J\|_F^2 = 0.0625 + 0.25 + 0.0625 = 0.375` },
              { title: "x = (3, 0)", tex: String.raw`a = (3,\ 6),\ h = (0.952574,\ 0.997527),\ h(1-h) = (0.045177,\ 0.002467),\ \|J\|_F^2 = 0.045177^2\cdot1 + 0.002467^2\cdot5 = 0.002071` },
            ],
          },
          { t: "p", text: "Saturated units barely react to input changes, so the penalty is 180 times smaller at $x = (3, 0)$. The contractive penalty therefore pushes units into saturation for most inputs, keeping them responsive only where the data needs it." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a convolutional denoising autoencoder",
        prompt: "Adapt `autoencoder_variants.py` into a convolutional denoising AE: reshape the digits to (N, 1, 8, 8); encoder `Conv2d(1,16,3,padding=1)` → ReLU → `Conv2d(16,8,3,stride=2,padding=1)` (8×4×4 code); decoder `ConvTranspose2d(8,16,4,stride=2,padding=1)` → ReLU → `Conv2d(16,1,3,padding=1)` with `BCEWithLogitsLoss`. Train with 25% masking noise and compare its masked MSE with the MLP denoising AE (0.0193).",
        hint: "Check the shapes: a stride-2 3×3 convolution with padding 1 halves 8×8 to 4×4; a 4×4 transposed convolution with stride 2 and padding 1 doubles it back.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "class ConvDAE(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),\n                                 nn.Conv2d(16, 8, 3, stride=2, padding=1))        # (N, 8, 4, 4) code\n        self.dec = nn.Sequential(nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1), nn.ReLU(),\n                                 nn.Conv2d(16, 1, 3, padding=1))                   # logits (N, 1, 8, 8)\n    def forward(self, x):\n        return self.dec(self.enc(x))\n\n# training step: x of shape (B, 1, 8, 8)\n# logits = model(mask(x, 0.25, gen)); loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum') / len(x)\n# evaluation: torch.sigmoid(model(X_te_masked)) compared with X_te by MSE",
          },
          { t: "p", text: "The code has 128 numbers (8 channels × 4 × 4), so it is not a bottleneck — the masking noise is what stops copying. Convolutions share weights across positions, so the model has far fewer parameters than the 128-unit MLP; expect a masked MSE in the same range or lower. Always confirm by running it: compare clean and masked MSE, as in the table." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          String.raw`Autoencoder: encoder $z = f_\phi(x)$, decoder $\hat x = g_\theta(z)$, trained to minimise a reconstruction loss — self-supervised, no labels.`,
          String.raw`Reconstruction losses are negative log-likelihoods: MSE ↔ Gaussian (linear output), BCE ↔ Bernoulli (sigmoid output, logit gradient $\hat x - x$).`,
          String.raw`Copying must be blocked: bottleneck (undercomplete), sparsity ($\lambda\sum|h_j|$ or $\sum_j\mathrm{KL}(\rho\|\hat\rho_j)$), denoising (corrupt input, clean target), contraction ($\lambda\|\partial h/\partial x\|_F^2$).`,
          "Linear AE + MSE = PCA subspace (Eckart–Young; Baldi–Hornik), with an arbitrary basis. Non-linear AEs do better (MSE 0.016 vs 0.025 for 8-dim codes on digits).",
          String.raw`Sigmoid layer: $\|J\|_F^2 = \sum_j (h_j(1-h_j))^2\|W_j\|^2$.`,
          "Uses: dimensionality reduction, denoising, anomaly detection via reconstruction error (AUROC 0.91 on average, very class-dependent), compression, latent spaces for diffusion, pretraining.",
          "Plain AEs are not generative: codes follow no known distribution and the decoder is untrained in the holes between them — the motivation for the VAE.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Variational Autoencoders](/learn/m14/vae) — make the code a distribution, add a prior and a KL term, and the holes close: a real generative model with a likelihood bound.",
          "[Diffusion Models](/learn/m14/diffusion) — generation by learning to denoise at many noise levels, a direct descendant of the denoising autoencoder; latent diffusion runs in an autoencoder's latent space.",
          "[Self-Supervised and Contrastive Learning](/learn/m15/self-supervised) — masked autoencoders and other ways to learn representations without labels.",
        ],
      },
    ],
  },
};

export default lesson;
