import type { Lesson } from "@/lib/types";

const PARADIGMS_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_digits

np.set_printoptions(suppress=True)               # print 0.0009, not 9e-04


def softmax(v, T=1.0):
    e = np.exp((v - v.max()) / T)
    return e / e.sum()


def kd_loss(z, t, T):
    """Distillation loss T^2 * KL(p_teacher^T || q_student^T) and its gradient w.r.t. student logits z."""
    p, q = softmax(t, T), softmax(z, T)
    return T ** 2 * np.sum(p * np.log(p / q)), T * (q - p)


# ---- 1. temperature exposes the teacher's "dark knowledge" ----
t = np.array([6.0, 2.0, 1.0, -1.0])               # teacher logits for classes (7, 1, 9, 0) of one image
for T in (1, 2, 5):
    print(f"T = {T}: teacher soft targets {np.round(softmax(t, T), 4)}")

# ---- 2. loss and gradient, checked by finite differences ----
z = np.array([4.0, 3.0, 0.0, 0.0])                # student logits
loss, g = kd_loss(z, t, T=2.0)
num = np.array([(kd_loss(z + 1e-6 * e, t, 2.0)[0] - kd_loss(z - 1e-6 * e, t, 2.0)[0]) / 2e-6 for e in np.eye(4)])
print(f"\nT = 2: KD loss {loss:.4f}, gradient {np.round(g, 4)}, max error vs finite differences {np.abs(g - num).max():.1e}")

# ---- 3. high-temperature limit: T^2-scaled gradient -> (z - t) / K for zero-mean logits (logit matching) ----
zc, tc = z - z.mean(), t - t.mean()
print(f"(z - t) / K = {np.round((zc - tc) / 4, 4)}")
for T in (1, 5, 20, 100):
    print(f"   T = {T:3d}: T (q - p) = {np.round(kd_loss(zc, tc, T)[1], 4)}")

# ---- 4. prototypical-network few-shot classification on digit classes never used for fitting ----
X, y = load_digits(return_X_y=True)
X = X / 16.0
base, novel = y < 5, y >= 5                      # "meta-train" classes 0-4, test classes 5-9
mu = X[base].mean(axis=0)
P = np.linalg.svd(X[base] - mu, full_matrices=False)[2][:20].T   # 20-D PCA embedding fitted on classes 0-4 only


def embed(A):
    return (A - mu) @ P


rng = np.random.default_rng(0)


def episode_accuracy(n_way, k_shot, f, n_query=15, episodes=600):
    """Average accuracy of nearest-prototype classification over random N-way K-shot episodes from classes 5-9."""
    accs = []
    for _ in range(episodes):
        classes = rng.choice(np.arange(5, 10), n_way, replace=False)
        protos, queries, labels = [], [], []
        for c_idx, c in enumerate(classes):
            idx = rng.permutation(np.where(y == c)[0])[: k_shot + n_query]
            protos.append(f(X[idx[:k_shot]]).mean(axis=0))          # prototype = mean support embedding
            queries.append(f(X[idx[k_shot:]]))
            labels += [c_idx] * n_query
        protos, queries = np.array(protos), np.vstack(queries)
        d2 = ((queries[:, None, :] - protos[None]) ** 2).sum(-1)   # squared Euclidean distance to each prototype
        accs.append(np.mean(d2.argmin(axis=1) == np.array(labels)))
    return np.mean(accs)


print("\n5-way episodes on unseen classes 5-9   1-shot   5-shot")
for name, f in [("raw pixels (64-D)", lambda A: A), ("PCA of classes 0-4 (20-D)", embed)]:
    print(f"{name:<38} {episode_accuracy(5, 1, f):.3f}    {episode_accuracy(5, 5, f):.3f}")`;
const PARADIGMS_NUMPY_OUT = String.raw`T = 1: teacher soft targets [0.9747 0.0179 0.0066 0.0009]
T = 2: teacher soft targets [0.8015 0.1085 0.0658 0.0242]
T = 5: teacher soft targets [0.4845 0.2177 0.1783 0.1195]

T = 2: KD loss 0.7065, gradient [-0.5376  0.4293  0.0126  0.0958], max error vs finite differences 4.2e-10
(z - t) / K = [-0.4375  0.3125 -0.1875  0.3125]
   T =   1: T (q - p) = [-0.2627  0.2441  0.0065  0.0122]
   T =   5: T (q - p) = [-0.5827  0.4179 -0.0645  0.2293]
   T =  20: T (q - p) = [-0.4835  0.3426 -0.154   0.2949]
   T = 100: T (q - p) = [-0.447   0.3186 -0.1808  0.3092]

5-way episodes on unseen classes 5-9   1-shot   5-shot
raw pixels (64-D)                      0.723    0.891
PCA of classes 0-4 (20-D)              0.702    0.873`;
const KD_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=0)
XT, YT = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr)
XE, YE = torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te)
rng = np.random.default_rng(0)
small = torch.tensor(np.concatenate([rng.choice(np.where(y_tr == c)[0], 10, replace=False) for c in range(10)]))
full = torch.arange(len(XT))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def acc(m):
    m.eval()
    with torch.no_grad():
        return (m(XE).argmax(1) == YE).float().mean().item()


def noisy(x):
    return x + 0.1 * torch.randn_like(x)                     # light input noise, used in every run


# ---- teacher: a wide MLP trained with the true labels of all 1257 training images ----
torch.manual_seed(0)
teacher = nn.Sequential(nn.Linear(64, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 512), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(512, 10))
opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
for epoch in range(40):
    teacher.train()
    for idx in torch.randperm(len(XT)).split(64):
        loss = F.cross_entropy(teacher(noisy(XT[idx])), YT[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
teacher.eval()
with torch.no_grad():
    teacher_logits = teacher(XT)


def train_student(transfer, target, T=1.0, steps=2000):
    """target: 'labels' (true labels), 'teacher-hard' (teacher argmax) or 'teacher-soft' (T^2 KL at temperature T).
    The student only ever sees the images in transfer."""
    torch.manual_seed(1)
    s = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
    opt = torch.optim.Adam(s.parameters(), lr=3e-3)
    for step in range(steps):
        s.train()
        j = transfer[torch.randint(0, len(transfer), (64,))]
        out = s(noisy(XT[j]))
        if target == "labels":
            loss = F.cross_entropy(out, YT[j])
        elif target == "teacher-hard":
            loss = F.cross_entropy(out, teacher_logits[j].argmax(1))
        else:
            p = F.softmax(teacher_logits[j] / T, dim=1)                       # softened teacher distribution
            loss = T ** 2 * F.kl_div(F.log_softmax(out / T, dim=1), p, reduction="batchmean")   # T^2 KL(p || q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return s


s = train_student(full, "labels")
print(f"teacher {n_params(teacher):,} parameters, test acc {acc(teacher):.3f};  student {n_params(s):,} parameters")
print(f"student, true labels of all 1257 images:            {acc(s):.3f}")
for name, transfer in (("100", small), ("1257", full)):
    print(f"\ntransfer set = {name} images, labels NOT used (teacher only)")
    print(f"   teacher's hard argmax labels:                    {acc(train_student(transfer, 'teacher-hard')):.3f}")
    for T in ((1.0, 4.0, 10.0) if name == "100" else (4.0,)):
        print(f"   teacher's soft targets, T = {T:<4g}                 {acc(train_student(transfer, 'teacher-soft', T)):.3f}")`;
const KD_TORCH_OUT = String.raw`teacher 301,066 parameters, test acc 0.980;  student 1,210 parameters
student, true labels of all 1257 images:            0.980

transfer set = 100 images, labels NOT used (teacher only)
   teacher's hard argmax labels:                    0.880
   teacher's soft targets, T = 1                    0.891
   teacher's soft targets, T = 4                    0.931
   teacher's soft targets, T = 10                   0.944

transfer set = 1257 images, labels NOT used (teacher only)
   teacher's hard argmax labels:                    0.980
   teacher's soft targets, T = 4                    0.967`;
const PROTO_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.set_num_threads(1)
Xd, yd = load_digits(return_X_y=True)
X = torch.tensor(Xd / 16.0, dtype=torch.float32)
y = torch.tensor(yd)
BASE, NOVEL = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]          # meta-train on digits 0-4; test on digits never seen
ROWS, COLS = torch.arange(8).view(1, 8, 1), torch.arange(8).view(1, 1, 8)


def shift(x, gen):
    """Translate each 8x8 image by a random (dy, dx) in [-1, 1] pixels."""
    B = x.shape[0]
    img = F.pad(x.view(B, 1, 8, 8), (1, 1, 1, 1))[:, 0]
    dy, dx = torch.randint(0, 3, (2, B), generator=gen)
    r, c = (ROWS + dy.view(B, 1, 1)).expand(B, 8, 8), (COLS + dx.view(B, 1, 1)).expand(B, 8, 8)
    return img[torch.arange(B).view(B, 1, 1), r, c].reshape(B, 64)


def dihedral(imgs, k):
    """k = 0..7: the 8 rotations/reflections of a square image."""
    out = torch.rot90(imgs.view(-1, 8, 8), k % 4, dims=(1, 2))
    return (out.transpose(1, 2) if k >= 4 else out).reshape(-1, 64)


def sample_episode(pool, n_way, k_shot, n_query, gen, augment=False):
    """pool: dict class -> images. Returns support (n_way, k_shot, 64) and query (n_way, n_query, 64)."""
    keys = list(pool)
    sup, qry = [], []
    for i in torch.randperm(len(keys), generator=gen)[:n_way]:
        imgs = pool[keys[i]]
        idx = torch.randperm(len(imgs), generator=gen)[: k_shot + n_query]
        s, q = imgs[idx[:k_shot]], imgs[idx[k_shot:]]
        if augment:
            s, q = shift(s, gen), shift(q, gen)
        sup.append(s)
        qry.append(q)
    return torch.stack(sup), torch.stack(qry)


def proto_logits(f, sup, qry):
    """Prototypical network: prototype = mean support embedding; logits = -squared Euclidean distance."""
    n_way, k_shot, _ = sup.shape
    protos = f(sup.reshape(-1, 64)).reshape(n_way, k_shot, -1).mean(1)       # (n_way, d)
    return -torch.cdist(f(qry.reshape(-1, 64)), protos) ** 2                 # (n_way * n_query, n_way)


def train(pool, n_way, augment, episodes=2000):
    torch.manual_seed(0)
    f = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(episodes):
        sup, qry = sample_episode(pool, n_way, 5, 10, gen, augment)
        loss = F.cross_entropy(proto_logits(f, sup, qry), torch.arange(n_way).repeat_interleave(10))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return f


def evaluate(f, pool, k_shot, episodes=600):
    gen = torch.Generator().manual_seed(123)                                 # identical test episodes for every model
    accs = []
    with torch.no_grad():
        for _ in range(episodes):
            sup, qry = sample_episode(pool, 5, k_shot, 15, gen)
            accs.append((proto_logits(f, sup, qry).argmax(1) == torch.arange(5).repeat_interleave(15)).float().mean().item())
    return np.mean(accs)


base = {c: X[y == c] for c in BASE}
base40 = {(c, k): dihedral(X[y == c], k) for c in BASE for k in range(8)}   # 40 "pseudo-classes"
novel = {c: X[y == c] for c in NOVEL}
novel_shifted = {c: shift(X[y == c], torch.Generator().manual_seed(7)) for c in NOVEL}

models = {
    "raw pixels (no training)": lambda a: a,
    "ProtoNet, 5 base classes": train(base, 5, augment=False),
    "ProtoNet, 40 pseudo-classes + shifts": train(base40, 10, augment=True),
}
print("5-way accuracy on unseen digits 5-9      clean 1-shot  5-shot   shifted 1-shot  5-shot")
for name, f in models.items():
    print(f"{name:<38} {evaluate(f, novel, 1):10.3f} {evaluate(f, novel, 5):7.3f} "
          f"{evaluate(f, novel_shifted, 1):14.3f} {evaluate(f, novel_shifted, 5):7.3f}")`;
const PROTO_TORCH_OUT = String.raw`5-way accuracy on unseen digits 5-9      clean 1-shot  5-shot   shifted 1-shot  5-shot
raw pixels (no training)                    0.721   0.888          0.311   0.414
ProtoNet, 5 base classes                    0.388   0.492          0.253   0.271
ProtoNet, 40 pseudo-classes + shifts        0.640   0.789          0.505   0.687`;
const MAML_TORCH = String.raw`import math
import torch
import torch.nn as nn
from torch.func import functional_call

torch.set_num_threads(1)
torch.manual_seed(0)


def sample_task():
    """A regression task: y = A sin(x - phase), with A ~ U[0.1, 5] and phase ~ U[0, pi] (Finn et al. 2017)."""
    A, phase = 0.1 + 4.9 * torch.rand(1), math.pi * torch.rand(1)
    return lambda x: A * torch.sin(x - phase)


def sample_points(task, n):
    x = 10 * torch.rand(n, 1) - 5                       # x ~ U[-5, 5]
    return x, task(x)


def make_net():
    return nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, 1))


def adapt(net, params, x, y, steps, lr, create_graph=False):
    """Inner loop: a few plain gradient steps on the K support points, starting from params."""
    for _ in range(steps):
        loss = ((functional_call(net, params, (x,)) - y) ** 2).mean()
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=create_graph)
        params = {k: v - lr * g for (k, v), g in zip(params.items(), grads)}
    return params


K, INNER_LR, META_BATCH = 10, 0.01, 5

# ---- MAML: learn an initialisation that adapts well in ONE inner step ----
maml = make_net()
opt = torch.optim.Adam(maml.parameters(), lr=1e-3)
for it in range(1, 2001):
    meta_loss = 0.0
    for _ in range(META_BATCH):
        task = sample_task()
        xs, ys = sample_points(task, K)                 # support set: used by the inner step
        xq, yq = sample_points(task, K)                 # query set: evaluates the adapted parameters
        fast = adapt(maml, dict(maml.named_parameters()), xs, ys, 1, INNER_LR, create_graph=True)
        meta_loss = meta_loss + ((functional_call(maml, fast, (xq,)) - yq) ** 2).mean()
    opt.zero_grad()
    (meta_loss / META_BATCH).backward()                 # second-order gradient through the inner step
    opt.step()
    if it in (1, 1000, 2000):
        print(f"meta-iteration {it:4d}  post-adaptation query MSE {meta_loss.item() / META_BATCH:.3f}")

# ---- baseline: ordinary "pretraining" on data pooled from many tasks ----
pre = make_net()
opt = torch.optim.Adam(pre.parameters(), lr=1e-3)
for it in range(2000):
    loss = 0.0
    for _ in range(META_BATCH):
        x, yv = sample_points(sample_task(), K)
        loss = loss + ((pre(x) - yv) ** 2).mean()
    opt.zero_grad()
    (loss / META_BATCH).backward()
    opt.step()

# ---- evaluation on 100 new tasks: K = 10 support points, gradient steps with lr 0.01, MSE on 100 test points ----
torch.manual_seed(1)
tasks = [sample_task() for _ in range(100)]
data = [(sample_points(t, K), sample_points(t, 100)) for t in tasks]
print("\nmean test MSE on new tasks after k gradient steps on 10 points")
print("                        k = 0     k = 1    k = 10")
for name, net in [("MAML initialisation", maml), ("pooled pretraining", pre)]:
    row = []
    for steps in (0, 1, 10):
        mse = 0.0
        for (xs, ys), (xt, yt) in data:
            p = adapt(net, {k: v.detach().requires_grad_() for k, v in net.named_parameters()}, xs, ys, steps, INNER_LR)
            with torch.no_grad():
                mse += ((functional_call(net, p, (xt,)) - yt) ** 2).mean().item()
        row.append(mse / len(data))
    print(f"{name:<22}" + "".join(f"{v:9.3f}" for v in row))`;
const MAML_TORCH_OUT = String.raw`meta-iteration    1  post-adaptation query MSE 5.238
meta-iteration 1000  post-adaptation query MSE 1.541
meta-iteration 2000  post-adaptation query MSE 2.092

mean test MSE on new tasks after k gradient steps on 10 points
                        k = 0     k = 1    k = 10
MAML initialisation       3.053    1.439    0.922
pooled pretraining        3.170    2.780    2.381`;

const KD_KERAS = String.raw`import tensorflow as tf


class Distiller(tf.keras.Model):
    """Student trained on alpha * T^2 KL(teacher_T || student_T) + (1 - alpha) * CE(labels, student)."""

    def __init__(self, student, teacher, T=4.0, alpha=0.9):
        super().__init__()
        self.student, self.teacher, self.T, self.alpha = student, teacher, T, alpha
        self.kl = tf.keras.losses.KLDivergence()
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def train_step(self, data):
        x, y = data
        t_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s_logits = self.student(x, training=True)
            soft = self.kl(tf.nn.softmax(t_logits / self.T), tf.nn.softmax(s_logits / self.T)) * self.T ** 2
            loss = self.alpha * soft + (1 - self.alpha) * self.ce(y, s_logits)
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return {"loss": loss, "distill": soft}


# teacher = a trained tf.keras.Model returning logits; student = a smaller one
# distiller = Distiller(student, teacher); distiller.compile(optimizer="adam"); distiller.fit(x_train, y_train, epochs=20)`;

const lesson: Lesson = {
  id: "m15/learning-paradigms",
  title: "Distillation, Curriculum, Meta-, Few- and Zero-Shot Learning",
  summary:
    "Ways to train beyond 'one model, one labelled dataset': knowledge distillation with temperature-softened targets (full theory, gradient and high-temperature limit), curriculum learning, meta-learning with MAML and its second-order gradient, prototypical networks for few-shot classification, Siamese and triplet networks, and zero-shot learning through attributes and shared embeddings — each computed and tested.",
  minutes: 85,
  level: "advanced",
  prerequisites: ["m05/advanced-losses", "m05/classification-losses", "m02/information-theory", "m12/transfer-learning"],
  next: ["m15/multimodal", "m18/model-optimization", "m15/self-supervised"],
  terms: ["knowledge-distillation", "temperature", "dark-knowledge", "soft-targets", "curriculum-learning", "meta-learning", "maml", "few-shot-learning", "n-way-k-shot", "prototypical-network", "siamese-network", "zero-shot-learning", "metric-learning"],
  sections: {
    intro: [
      { t: "p", text: "The usual recipe — many labelled examples per class, one network trained from scratch, then deploy — often breaks. The best model may be too big to deploy. Some examples are much harder than others. New classes may arrive with one or five examples, or none. This lesson covers methods built for these cases." },
      { t: "list", items: [
        "**Knowledge distillation** (Hinton, Vinyals and Dean, 2015) trains a small **student** to copy a large **teacher**'s full output distribution, softened by a temperature, so it learns not just *which* class is right but *how wrong* each other class is. It is the standard way to shrink models (e.g. DistilBERT, Sanh et al. 2019).",
        "**Curriculum learning** (Bengio, Louradour, Collobert and Weston, 2009) orders training examples from easy to hard, like a school syllabus.",
        "**Meta-learning** — 'learning to learn' — trains on many small tasks so a new task can be learned from very few examples. **MAML** (Finn, Abbeel and Levine, 2017) learns a starting point that adapts in a few gradient steps.",
        "**Few-shot learning** classifies new classes from $K$ examples each. **Prototypical networks** (Snell, Swersky and Zemel, 2017) classify by distance to each class's mean embedding.",
        "**Metric learning** with **Siamese** and **triplet networks** learns an embedding where distance means dissimilarity, so new identities can be compared without retraining.",
        "**Zero-shot learning** recognises classes with *no* training examples, from a description — attributes (Lampert, Nickisch and Harmeling, 2009) or text embeddings, as in CLIP.",
      ] },
      { t: "callout", kind: "key", text: String.raw`Distillation loss: $\mathcal L_{\text{KD}} = T^2\,\mathrm{KL}\big(\mathrm{softmax}(t/T)\,\|\,\mathrm{softmax}(z/T)\big)$, gradient $\partial\mathcal L_{\text{KD}}/\partial z = T\,(q^T - p^T)$. Few-shot: $p(y = c\mid x) \propto \exp(-\lVert f(x) - \mu_c\rVert^2)$ with $\mu_c$ the mean support embedding. MAML: $\min_\theta \sum_i \mathcal L_i\big(\theta - \alpha\nabla\mathcal L_i(\theta)\big)$.` },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Advanced Losses](/learn/m05/advanced-losses) — contrastive and triplet losses, hard-negative mining, and a first look at the KL distillation loss. Here we build the full distillation theory and use those metric losses without re-deriving them.",
        "[Classification Losses](/learn/m05/classification-losses) — softmax, cross-entropy and its gradient $q - y$.",
        "[Information Theory](/learn/m02/information-theory) — entropy, cross-entropy and the KL divergence.",
        "[Transfer Learning](/learn/m12/transfer-learning) — pretrained features and fine-tuning, the simplest few-shot baseline.",
        "[Backpropagation](/learn/m04/backpropagation) — MAML differentiates *through* a gradient step, so you need a clear picture of the computation graph.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "An expert's notes versus the answer key", text: "Before an exam you can study the answer key (\"the answer is 7\") or an expert's notes: \"this is a 7, though it looks a bit like a 1, slightly like a 9, and nothing like a 0\". The notes teach far more per example — they show which classes are alike and which mistakes are understandable. Distillation gives the student the expert's notes. The temperature sets how much the expert says about the less likely options." },
      { t: "p", text: "**Curriculum** is a course's order: arithmetic before calculus. **Meta-learning** is learning *how to study*: after many exams, you master a new subject from a few pages. **Few-shot learning with prototypes** is what you do after one photo each of five new colleagues: remember a 'typical' face per person and match new photos to the closest. **Zero-shot learning** is recognising a zebra on first sight because you were told it is \"a horse with black-and-white stripes\"." },
      { t: "p", text: "The common thread: each brings in knowledge from somewhere other than the target labels — a teacher, an ordering, related tasks, an embedding space, a description. That is why they work when labels or model size are scarce." },
    ],
    mechanism: [
      { t: "h", text: "Knowledge distillation" },
      { t: "steps", items: [
        { title: "Train (or obtain) a teacher", text: "A large accurate model, or an ensemble. Its logits $t$ will be the targets." },
        { title: "Choose a transfer set", text: "The inputs on which the student copies the teacher: the labelled training set, or unlabelled data — the distillation term needs no labels." },
        { title: "Soften both distributions", text: String.raw`Divide teacher and student logits by the same temperature $T > 1$ before the softmax: $p^T = \mathrm{softmax}(t/T)$, $q^T = \mathrm{softmax}(z/T)$.` },
        { title: "Distillation loss", text: String.raw`$T^2\,\mathrm{KL}(p^T\,\|\,q^T)$ — the $T^2$ keeps the gradient size independent of $T$.` },
        { title: "Optionally add the hard-label loss", text: String.raw`$\mathcal L = \alpha\,T^2\,\mathrm{KL}(p^T\|q^T) + (1-\alpha)\,\mathrm{CE}(y, \mathrm{softmax}(z))$ at $T = 1$. Hinton et al. found a much lower weight on the hard term worked best.` },
        { title: "Deploy the student at T = 1", text: "Temperature is only a training tool." },
      ] },
      { t: "h", text: "Curriculum learning" },
      { t: "steps", items: [
        { title: "Score difficulty", text: "With a rule of thumb (sentence length, noise level, object size), a pretrained model's loss, or the current model's own loss (self-paced learning, Kumar, Packer and Koller, 2010)." },
        { title: "Pacing function", text: "Start on the easiest fraction of the data and grow the fraction on a schedule until all data is used." },
        { title: "Anti-curriculum and hard-example mining", text: "The opposite — stressing hard examples, as in focal loss or online hard-example mining — is sometimes better. Test which wins; gains are usually largest with noisy labels or tiny training budgets." },
      ] },
      { t: "h", text: "Few-shot episodes and prototypical networks" },
      { t: "steps", items: [
        { title: "N-way K-shot episode", text: String.raw`Pick $N$ classes; for each, take $K$ labelled **support** examples and some **query** examples. The task: sort the queries into the $N$ classes using only the support set.` },
        { title: "Embed", text: String.raw`A network $f_\phi$ maps every image to a vector.` },
        { title: "Prototypes", text: String.raw`$\mu_c = \frac1K\sum_{x\in S_c} f_\phi(x)$ — the mean support embedding of class $c$.` },
        { title: "Classify", text: String.raw`Softmax over negative squared Euclidean distances: $p(y=c\mid x) \propto \exp(-\lVert f_\phi(x) - \mu_c\rVert^2)$.` },
        { title: "Meta-train episodically", text: "On base classes, repeat: sample an episode, compute the cross-entropy of the query predictions, update $\\phi$. Test episodes use classes never seen in training. 'Train as you test.'" },
      ] },
      { t: "h", text: "MAML" },
      { t: "steps", items: [
        { title: "Sample a batch of tasks", text: String.raw`Each task $\mathcal T_i$ has a support set and a query set.` },
        { title: "Inner loop (adapt)", text: String.raw`From the shared starting point $\theta$, take one or a few gradient steps on the support loss: $\theta_i' = \theta - \alpha\nabla_\theta\mathcal L^{\text{sup}}_i(\theta)$.` },
        { title: "Outer loop (meta-update)", text: String.raw`Score each adapted $\theta_i'$ on its query set and update $\theta \leftarrow \theta - \beta\nabla_\theta\sum_i \mathcal L^{\text{qry}}_i(\theta_i')$. The gradient flows **through** the inner step, so second derivatives appear (Hessian-vector products).` },
        { title: "At test time", text: "Take the same few inner steps on the new task's support set. Any model trained by gradient descent works (model-agnostic)." },
      ] },
      { t: "h", text: "Siamese and triplet networks" },
      { t: "p", text: "A **Siamese network** (Bromley et al., 1993, signature checking) runs one shared encoder on two inputs and compares the embeddings, trained with the contrastive loss on same/different pairs. A **triplet network** encodes an anchor, a positive and a negative and uses the triplet loss (both taught in [Advanced Losses](/learn/m05/advanced-losses)). At test time the encoder does verification (distance below a threshold?) and identification (nearest enrolled embedding) for identities never seen in training — face verification is the classic case." },
      { t: "h", text: "Zero-shot learning" },
      { t: "steps", items: [
        { title: "Describe every class", text: "With a vector of attributes (stripes, hooves, lives in water…), a word embedding of its name, or a text-encoder embedding of a sentence about it." },
        { title: "Learn a compatibility function", text: String.raw`On seen classes, learn to map inputs into the description space (or both into a shared space): $s(x, c) = \langle g(x), a_c\rangle$.` },
        { title: "Predict unseen classes", text: String.raw`A new class needs only its description $a_c$: $\hat y = \arg\max_c s(x, c)$ over the unseen (or all) classes. CLIP does this at scale with text prompts — see [Multimodal Models](/learn/m15/multimodal).` },
      ] },
    ],
    math: [
      { t: "h", text: "1. Temperature-softened softmax" },
      { t: "math", tex: String.raw`p_k^T = \frac{\exp(t_k/T)}{\sum_{j=1}^{K}\exp(t_j/T)},\qquad q_k^T = \frac{\exp(z_k/T)}{\sum_{j=1}^{K}\exp(z_j/T)}` },
      { t: "p", text: String.raw`$t, z \in \mathbb R^K$ are the teacher and student logits, $K$ the number of classes, $T > 0$ the temperature. $T = 1$ is the normal softmax; $T \to \infty$ gives a uniform distribution; $T \to 0$ a one-hot on the arg-max. Dividing logit gaps by $T$ pulls probability *ratios* $p_k/p_j = e^{(t_k - t_j)/T}$ toward 1. For teacher logits $(6, 2, 1, -1)$: $T = 1$ gives $(0.9747, 0.0179, 0.0066, 0.0009)$, almost one-hot; $T = 5$ gives $(0.4845, 0.2177, 0.1783, 0.1195)$, revealing class 2 as the likeliest alternative. Hinton et al. called this information in the small probabilities **dark knowledge**.` },
      { t: "h", text: "2. The distillation loss and its gradient" },
      { t: "math", tex: String.raw`\mathcal L_{\text{KD}} = T^2\,\mathrm{KL}(p^T\,\|\,q^T) = T^2\sum_{k=1}^{K} p_k^T\log\frac{p_k^T}{q_k^T} = T^2\Big(\underbrace{-\sum_k p_k^T\log q_k^T}_{\text{cross-entropy}} - \underbrace{H(p^T)}_{\text{const. in } z}\Big)` },
      { t: "p", text: String.raw`The teacher's entropy does not depend on $z$, so minimising the KL is the same as minimising cross-entropy with soft targets. The softmax-cross-entropy gradient with respect to its logits $z/T$ is $q^T - p^T$; the chain rule adds $\partial(z/T)/\partial z = 1/T$, and with the $T^2$ factor:` },
      { t: "math", tex: String.raw`\frac{\partial\mathcal L_{\text{KD}}}{\partial z_k} = T^2\cdot\frac1T\,(q_k^T - p_k^T) = T\,(q_k^T - p_k^T)` },
      { t: "p", text: String.raw`**Why $T^2$:** without it the gradient would be $(q^T - p^T)/T$, and $q^T - p^T$ itself shrinks like $1/T$ for large $T$ (next section), so soft-target gradients would scale as $1/T^2$. Multiplying by $T^2$ keeps the balance between soft and hard targets roughly fixed as you try different $T$.` },
      { t: "h", text: "3. High-temperature limit: distillation becomes logit matching" },
      { t: "p", text: String.raw`For large $T$, $e^{x/T} \approx 1 + x/T$, so $q_k^T \approx (1 + z_k/T)/(K + \sum_j z_j/T)$. Assume each model's logits have zero mean ($\sum_j z_j = \sum_j t_j = 0$; allowed, since adding a constant to all logits leaves the softmax unchanged). Then $q_k^T \approx \frac1K + \frac{z_k}{KT}$, likewise for $p$, and:` },
      { t: "math", tex: String.raw`\frac{\partial\mathcal L_{\text{KD}}}{\partial z_k} = T\,(q_k^T - p_k^T) \;\xrightarrow{T\to\infty}\; \frac{1}{K}\,(z_k - t_k)` },
      { t: "p", text: String.raw`That is the gradient of $\frac{1}{2K}\lVert z - t\rVert^2$. In plain words: at high temperature, distillation is squared-error **logit matching** — the older compression method, which Hinton et al. showed is a special case. At moderate temperatures the student pays less attention to very negative logits, which are noisy because the teacher was never trained to rank them. The from-scratch program shows $T(q - p)$ approaching $(z - t)/K = (-0.4375, 0.3125, -0.1875, 0.3125)$: at $T = 100$ it is $(-0.447, 0.3186, -0.1808, 0.3092)$.` },
      { t: "h", text: "4. Worked example" },
      { t: "p", text: String.raw`Teacher logits $t = (6, 2, 1, -1)$, student logits $z = (4, 3, 0, 0)$, $T = 2$, four classes.` },
      { t: "steps", items: [
        { title: "Soft targets", tex: String.raw`p^T = \mathrm{softmax}(3, 1, 0.5, -0.5) = (0.8015,\ 0.1085,\ 0.0658,\ 0.0242)` },
        { title: "Student soft predictions", tex: String.raw`q^T = \mathrm{softmax}(2, 1.5, 0, 0) = (0.5327,\ 0.3231,\ 0.0721,\ 0.0721)` },
        { title: "KL terms $p_k\\log(p_k/q_k)$", tex: String.raw`(0.3275,\ -0.1184,\ -0.0060,\ -0.0264);\quad \mathrm{KL} = 0.1766` },
        { title: "Distillation loss", tex: String.raw`\mathcal L_{\text{KD}} = 2^2\times0.1766 = 0.7065` },
        { title: "Gradient", tex: String.raw`T(q^T - p^T) = 2\,(-0.2688,\ 0.2146,\ 0.0063,\ 0.0479) = (-0.5376,\ 0.4293,\ 0.0126,\ 0.0958)` },
      ] },
      { t: "p", text: String.raw`A descent step raises the class-0 logit and lowers class 1 most, since the student over-rates class 1. Its arg-max is already right (the hard cross-entropy $-\log\mathrm{softmax}(z)_0 = 0.3397$ would push only on class 0), but soft targets also fix the *ranking* of the wrong classes: class 3 must come down further than class 2.` },
      { t: "h", text: "5. Prototypical networks" },
      { t: "math", tex: String.raw`\mu_c = \frac{1}{|S_c|}\sum_{(x_i, y_i)\in S_c} f_\phi(x_i),\qquad p_\phi(y = c\mid x) = \frac{\exp\big(-d(f_\phi(x), \mu_c)\big)}{\sum_{c'}\exp\big(-d(f_\phi(x), \mu_{c'})\big)}` },
      { t: "p", text: String.raw`$S_c$ is class $c$'s support set, $f_\phi$ the embedding network, $d$ the squared Euclidean distance. Snell et al. showed this is a **linear classifier** in embedding space: $-\lVert f - \mu_c\rVert^2 = 2\mu_c^\top f - \lVert\mu_c\rVert^2 - \lVert f\rVert^2$, and the last term is the same for every class. **Example** (2-way 2-shot, 2-D embeddings): class A support $(1, 0), (3, 2)$ → $\mu_A = (2, 1)$; class B $(-1, 2), (-3, 0)$ → $\mu_B = (-2, 1)$. Query $f(x) = (0.5, 1.5)$: $d_A = 1.5^2 + 0.5^2 = 2.5$, $d_B = 2.5^2 + 0.5^2 = 6.5$, so $p(A) = 1/(1 + e^{-4}) = 0.982$.` },
      { t: "h", text: "6. MAML and its meta-gradient" },
      { t: "math", tex: String.raw`\min_\theta\ \sum_{i}\mathcal L^{\text{qry}}_i\big(\theta_i'\big),\qquad \theta_i' = \theta - \alpha\,\nabla_\theta\mathcal L^{\text{sup}}_i(\theta)` },
      { t: "math", tex: String.raw`\nabla_\theta\,\mathcal L^{\text{qry}}_i(\theta_i') = \big(I - \alpha\,\nabla^2_\theta\mathcal L^{\text{sup}}_i(\theta)\big)\,\nabla_{\theta'}\mathcal L^{\text{qry}}_i(\theta_i')` },
      { t: "p", text: String.raw`$\alpha$ is the inner learning rate, $\nabla^2$ the Hessian (second derivatives) of the support loss. **First-order MAML** drops the Hessian term and uses $\nabla_{\theta'}\mathcal L^{\text{qry}}_i(\theta'_i)$ directly; Finn et al. found it nearly as good. Reptile (Nichol, Achiam and Schulman, 2018) is another first-order variant. **Example** with a scalar parameter, $\mathcal L^{\text{sup}} = \mathcal L^{\text{qry}} = (\theta - a)^2$, $\theta = 0$, $a = 3$, $\alpha = 0.1$: inner gradient $2(0 - 3) = -6$, so $\theta' = 0 + 0.6 = 0.6$; query loss $(0.6 - 3)^2 = 5.76$; $\nabla_{\theta'} = 2(0.6 - 3) = -4.8$; Hessian $= 2$, so the meta-gradient is $(1 - 0.1\cdot2)(-4.8) = -3.84$ (first-order MAML would use $-4.8$).` },
      { t: "h", text: "7. Zero-shot scoring" },
      { t: "p", text: String.raw`With class descriptions $a_c$ and an input mapped to $g(x)$, predict $\hat y = \arg\max_c \cos(g(x), a_c)$. **Example**: attributes (stripes, hooves, lives in water); zebra $= (1, 1, 0)$, dolphin $= (0, 0, 1)$, tiger $= (1, 0, 0)$. A network trained on *other* animals outputs $g(x) = (0.9, 0.7, 0.1)$ for a photo. Cosines: zebra 0.9885, tiger 0.7863, dolphin 0.0874 → zebra, though it never saw a zebra photo.` },
      { t: "h", text: "8. Curriculum as a weighting" },
      { t: "p", text: String.raw`A curriculum is a time-dependent weighting of examples, $\mathcal L_t = \sum_n w_t(x_n)\,\ell(x_n)$, with $w_t$ focused on easy examples early and becoming uniform later. Self-paced learning uses $w_t(x_n) = \mathbb 1[\ell(x_n) < \lambda_t]$ with a threshold $\lambda_t$ that grows during training. Bengio et al. saw curricula as a continuation method: optimise a smoothed version of the objective first, then gradually the real one.` },
    ],
    visual: [
      { t: "p", text: "The metric-learning simulator trains the kind of embedding Siamese, triplet and prototypical networks use: no classifier over fixed classes, only which points belong near each other." },
      { t: "sim", id: "metric-learning", caption: "A 2-D embedding trained with contrastive (Siamese) or triplet loss; nearest-neighbour accuracy is tracked live." },
      { t: "list", items: [
        "Pick the **contrastive** loss — the Siamese setting, with pairs labelled same/different. Press Play and watch same-class points gather.",
        "Switch to the **triplet** loss. The gold circle shows the margin around the anchor–positive distance that the negative must clear.",
        "Once clusters form, picture each cluster's centre as a **prototype**: labelling a new point by its nearest centre is exactly a prototypical network's rule (math §5).",
        "Compare **random**, **semi-hard** and **hard** negatives: hard negatives teach the most but can destabilise training — the same tension as easy-first (curriculum) versus hard-first (mining) ordering.",
      ] },
    ],
    scratch: [
      { t: "p", text: "Distillation from first principles — softened targets, the $T^2$-scaled KL and its gradient (checked by finite differences), and the high-temperature limit — then nearest-prototype few-shot classification on digit classes 5–9, comparing raw pixels with a PCA embedding fitted only on classes 0–4." },
      { t: "code", lang: "python", title: "distillation_and_prototypes.py", code: PARADIGMS_NUMPY, output: PARADIGMS_NUMPY_OUT },
      { t: "list", items: [
        "As $T$ rises, the teacher's distribution flattens; at $T = 5$ the ranking of the wrong classes (class 1 > class 2 > class 3) is clear.",
        "The gradient $T(q - p)$ matches finite differences to $4\\times10^{-10}$ and reproduces the worked example (loss 0.7065).",
        "The scaled gradient approaches $(z - t)/K$ as $T$ grows: high-temperature distillation is logit matching.",
        "**Few-shot**: on 5-way episodes of unseen digits, raw-pixel prototypes reach 0.723 (1-shot) and 0.891 (5-shot). A generic PCA embedding fitted on other classes does slightly worse (0.702 / 0.873): an embedding helps only if it captures what separates the *new* classes. Five shots instead of one help far more.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**Distillation.** A 301k-parameter teacher, a 1.2k-parameter student (250× smaller). Key experiment: a *transfer set* of only 100 images whose labels the student never sees; it learns only from the teacher's hard arg-max labels or its soft targets at various temperatures." },
          { t: "code", lang: "python", title: "distillation_digits.py", code: KD_TORCH, output: KD_TORCH_OUT },
          { t: "list", items: [
            "With only 100 transfer images, soft targets help a lot: 0.880 with the teacher's hard labels, 0.891 at $T = 1$, 0.931 at $T = 4$ and 0.944 at $T = 10$. The extra information per example — the dark knowledge — matters most when examples are few, exactly the case Hinton et al. highlighted.",
            "With all 1257 transfer images the student already matches the teacher on hard labels (0.980), and $T = 4$ is slightly worse (0.967). Distillation is not a guaranteed gain; temperature is a hyper-parameter to tune.",
            "The student matches the teacher with 250× fewer parameters because digits are easy; real compression gaps are larger. See [Model Optimization](/learn/m18/model-optimization) for distillation alongside quantization and pruning.",
            "`F.kl_div` expects **log**-probabilities for the student and probabilities for the target, and computes $\\mathrm{KL}(\\text{target}\\,\\|\\,\\text{input})$; `reduction=\"batchmean\"` gives the true per-example KL.",
          ] },
          { t: "p", text: "**Prototypical networks** trained episodically on digits 0–4 and tested on 5-way episodes of digits 5–9, on clean test images and on test images shifted by up to one pixel." },
          { t: "code", lang: "python", title: "protonet_digits.py", code: PROTO_TORCH, output: PROTO_TORCH_OUT },
          { t: "list", items: [
            "**Meta-training on only 5 base classes fails** (0.388 / 0.492, far below raw pixels): the embedding specialises in separating 0–4 and drops features needed for 5–9 — why few-shot benchmarks use dozens to thousands of base classes.",
            "Making 40 pseudo-classes (the 8 rotations/reflections of each base digit, a trick used to enlarge Omniglot) plus shift augmentation helps a lot (0.640 / 0.789 clean).",
            "On shifted test images, raw-pixel distance breaks down (0.311 / 0.414) while the learned metric holds up (0.505 / 0.687): the embedding learned the invariance it was trained with.",
            "Honest summary: on centred 8×8 digits, raw pixels are already an excellent metric. Meta-learned embeddings pay off when the raw input space is a poor metric and the base classes are diverse.",
          ] },
          { t: "p", text: "**MAML** on Finn et al.'s sinusoid-regression benchmark: each task is $y = A\\sin(x - \\phi)$ with random amplitude and phase, and the model gets 10 points of a new task." },
          { t: "code", lang: "python", title: "maml_sine.py", code: MAML_TORCH, output: MAML_TORCH_OUT },
          { t: "list", items: [
            "Before adapting ($k = 0$) both models are equally bad (MSE ≈ 3): with random phase, the best single prediction is near 0.",
            "After **one** gradient step on 10 points, the MAML start reaches 1.44 versus 2.78 for pooled pretraining; after 10 steps, 0.92 vs 2.38. MAML learned a starting point a few steps from every task's solution; the pooled model learned the average function, which is not.",
            "`create_graph=True` in the inner loop keeps the inner gradient's graph so the outer `backward()` can differentiate through it (the Hessian term of math §6). Evaluation does not need it.",
            "`torch.func.functional_call` runs the network with a dictionary of 'fast weights' without changing the module's own parameters.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "A distillation wrapper as a `tf.keras.Model` subclass with a custom `train_step`; only the student's variables are updated." },
          { t: "code", lang: "python", title: "distiller_keras.py (not executed in CI)", code: KD_KERAS, output: "Defines the Distiller model; fit() prints the total loss and the distillation term per epoch." },
        ] },
      ] },
    ],
    variations: [
      { t: "table", head: ["Paradigm / variant", "Idea", "Advantages", "Disadvantages", "Use cases"], rows: [
        ["Response-based distillation (Hinton et al. 2015)", "match softened output distributions", "simple, works with any architecture and on unlabelled data", "student size limits what it can absorb", "model compression, ensembles → one model"],
        ["Feature-based distillation (FitNets, Romero et al. 2015)", "also match intermediate feature maps (via a learned adapter)", "richer signal; helps thin, deep students", "needs matching layers", "compressing CNNs"],
        ["Self-distillation / born-again networks", "teacher and student share the same architecture", "often improves accuracy and calibration", "extra training rounds", "regularisation"],
        ["Curriculum / self-paced learning", "easy examples first", "can speed up and stabilise training, robust to noise", "gains depend on the task; needs a difficulty measure", "noisy labels, RL, language-model data ordering"],
        ["MAML / first-order MAML / Reptile", "learn a starting point for fast adaptation", "model-agnostic; adapts by fine-tuning", "second-order cost, unstable, many hyper-parameters", "few-shot regression, RL, personalisation"],
        ["Prototypical / Matching networks", "classify by distance in a learned embedding", "simple, no training at test time", "needs many diverse base classes", "few-shot classification"],
        ["Siamese / triplet networks", "shared encoder + pair or triplet loss", "open-set: new identities without retraining", "pair/triplet mining is critical", "face, signature, product verification"],
        ["Attribute-based zero-shot", "map inputs to human-defined attributes", "interpretable; no examples of new classes needed", "attributes must be designed and annotated", "fine-grained recognition"],
        ["Embedding-based zero-shot (CLIP)", "shared image–text space, classify by prompts", "open vocabulary, very general", "needs huge paired data; sensitive to prompts", "open-vocabulary classification and retrieval"],
        ["Fine-tuning a pretrained backbone", "transfer learning with few labels", "strong, simple baseline", "may overfit with 1–5 shots", "most practical few-shot problems"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Compressed language models** — DistilBERT (Sanh et al., 2019) distils BERT into a smaller, faster model; large-to-small distillation is now routine in industry.",
        "**On-device vision and speech** — students small enough for phones, trained from server-side teachers; see [Model Optimization](/learn/m18/model-optimization).",
        "**Face verification** — Siamese / triplet-style embeddings (FaceNet, ArcFace) compare a face with an enrolled template; a new user needs one photo and no retraining.",
        "**Few-shot personalisation** — adapting a model to a user, a new product category or a new robot task from a handful of examples.",
        "**Drug discovery and rare diseases** — few-shot and meta-learning where labelled examples are scarce by nature.",
        "**Open-vocabulary recognition** — zero-shot classifiers built from text descriptions; see [CLIP](/learn/m15/multimodal).",
        "**Curricula** — reinforcement learning on ever-harder environments; ordering language-model training data by quality or difficulty.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Distillation** can match or approach the teacher with far fewer parameters and use unlabelled data. But it needs a good (costly) teacher, inherits its errors, and gains depend on temperature and transfer-set size (large with 100 images, none with 1257 in our experiment).",
        "**Curriculum learning** is cheap to try but its benefits vary by task; a bad difficulty measure can hurt, and random order is a strong baseline.",
        "**MAML** adapts fast from very few examples, but second-order gradients cost much memory and compute, training is sensitive to learning rates, and plain pretraining-plus-fine-tuning often competes on vision benchmarks.",
        "**Prototypical networks** are simple and need no optimisation at test time, but they rely on the base classes covering the variety of the new classes (our 5-base-class model failed).",
        "**Siamese/triplet networks** handle open sets of identities but need careful mining and threshold calibration.",
        "**Zero-shot** methods need no new-class examples but are limited by description quality, and lean toward seen classes in generalised zero-shot (test data mixing seen and unseen classes).",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Forgetting the $T^2$ factor.** Soft-target gradients then shrink as $1/T^2$, and the hard-label term quietly takes over when you raise $T$.",
        "**Applying the temperature to only the teacher or only the student.** Both logits must be divided by the same $T$ in the KL term.",
        "**Wrong argument order or input type for `F.kl_div`.** It takes student *log*-probabilities first and teacher probabilities second; passing probabilities or swapping them gives a different, wrong objective.",
        "**Leaving the teacher in training mode** (dropout on, BatchNorm updating), making targets noisy. Call `teacher.eval()` and compute targets under `torch.no_grad()`.",
        "**Running the student at temperature T at inference.** Predict with $T = 1$ (the arg-max is unaffected, but calibrated probabilities are).",
        "**Leaking new classes into few-shot meta-training** (or the pretrained backbone) and then reporting 'few-shot' accuracy. Base and new classes must not overlap.",
        "**Reporting few-shot accuracy from a handful of episodes.** Episodes vary a lot; average over hundreds with fixed seeds and report confidence intervals.",
        "**Implementing MAML without `create_graph=True`** — that quietly gives first-order MAML. Fine if intended, but know which one you are running.",
        "**Mixing contrastive-loss conventions inside a Siamese implementation** (e.g. $Y=1$ meaning 'similar' in one place and 'dissimilar' in another). Fix one convention, as in [Advanced Losses](/learn/m05/advanced-losses).",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is knowledge distillation?", a: "Training a small student model to reproduce a larger teacher's output distribution (softened with a temperature), usually combined with the normal hard-label loss. The student gets more information per example than labels alone give, and can approach the teacher's accuracy at a fraction of the size." },
      { t: "qa", level: "beginner", q: "What does an N-way K-shot task mean?", a: "A classification episode with N classes and K labelled support examples per class; the model must sort query examples into those N classes. E.g. 5-way 1-shot: one example each of five new classes." },
      { t: "qa", level: "beginner", q: "What is zero-shot learning?", a: "Recognising classes with no training examples, using a description of each class — attributes, a word embedding or a text prompt — and a learned match between inputs and descriptions." },
      { t: "qa", level: "intermediate", q: "Why divide by a temperature in distillation, and why multiply the loss by T²?", a: "At T = 1 a confident teacher is nearly one-hot, hiding how it ranks the wrong classes; T > 1 flattens it and reveals this dark knowledge. For large T the KL gradient scales like 1/T², so the T² factor keeps the soft-target term comparable to the hard-label term as T changes." },
      { t: "qa", level: "intermediate", q: "How does a prototypical network classify, and how is it trained?", a: "It averages each class's support embeddings into a prototype and gives each query a softmax over negative squared distances to the prototypes. Training is episodic on base classes: sample N-way K-shot episodes, compute the query cross-entropy, backpropagate into the embedding network." },
      { t: "qa", level: "intermediate", q: "What is the difference between a Siamese and a triplet network?", a: "Both share one encoder. A Siamese network takes pairs with a contrastive loss on same/different labels; a triplet network takes (anchor, positive, negative) and needs the negative farther than the positive by a margin. Triplet losses constrain relative distances, which usually suits retrieval and verification better." },
      { t: "qa", level: "intermediate", q: "Compute the distillation gradient for teacher logits (6, 2, 1, −1), student logits (4, 3, 0, 0), T = 2.", a: "p = softmax(3, 1, 0.5, −0.5) = (0.8015, 0.1085, 0.0658, 0.0242); q = softmax(2, 1.5, 0, 0) = (0.5327, 0.3231, 0.0721, 0.0721); gradient T(q − p) = (−0.5376, 0.4293, 0.0126, 0.0958). Loss T²KL = 0.7065." },
      { t: "qa", level: "advanced", q: "Show that distillation at high temperature is equivalent to matching logits.", a: "For large T, softmax(v/T)_k ≈ 1/K + v_k/(KT) when the logits have zero mean (use exp(x) ≈ 1 + x). So T(q − p) ≈ T·(z − t)/(KT) = (z − t)/K, the gradient of (1/2K)‖z − t‖². Large-T distillation reduces to squared-error logit matching; moderate T down-weights very negative, noisy logits." },
      { t: "qa", level: "advanced", q: "Derive the MAML meta-gradient and explain first-order MAML.", a: "With θ' = θ − α∇L_sup(θ), the chain rule gives ∇_θ L_qry(θ') = (I − α∇²L_sup(θ)) ∇_θ' L_qry(θ'). The Hessian term is expensive; first-order MAML drops it and uses ∇_θ' L_qry(θ') directly, which Finn et al. found nearly as good. Example: L = (θ − 3)², θ = 0, α = 0.1 → θ' = 0.6, full meta-gradient −3.84 versus first-order −4.8." },
      { t: "qa", level: "advanced", q: "Your prototypical network beats the baseline on base-class validation episodes but is worse than raw-pixel nearest-centroid on novel classes. What is happening and what do you try?", a: "The embedding overfits to features separating the base classes and drops those separating the new ones — typical with few or similar base classes (our 5-class digits model: 0.388 vs 0.721). Try more, more diverse base classes (or pseudo-classes via transformations), augmentation, regularisation and early stopping on new-class-like validation episodes, a self-supervised or broadly pretrained backbone, and plain fine-tuning baselines." },
    ],
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "Teacher logits (2, 0). What is the teacher's probability for class 0 at temperature T = 2 (3 decimals)?", answer: 0.731, tolerance: 0.002, explain: "softmax(1, 0) = $e/(e + 1) = 0.7311$ (at T = 1 it would be 0.8808)." },
        { q: "What happens to the softened distribution as T → ∞?", options: ["It becomes one-hot", "It becomes uniform", "It is unchanged", "It becomes the student's distribution"], answer: 1, explain: "Every logit divided by a huge T is ≈ 0, so each class gets 1/K." },
        { kind: "numeric", q: "For the distillation loss T²·KL(p^T‖q^T), the gradient w.r.t. a student logit is T(q_k − p_k). With T = 3, q_k = 0.2, p_k = 0.5, what is it?", answer: -0.9, tolerance: 0.0001, explain: "$3\\times(0.2 - 0.5) = -0.9$, so this logit should go up." },
        { kind: "numeric", q: "2-way prototypes μ_A = (0, 0), μ_B = (2, 0), query embedding (0.5, 0). What is the probability of class A under a prototypical network (squared Euclidean distance, 3 decimals)?", answer: 0.881, tolerance: 0.002, explain: "$d_A = 0.25$, $d_B = 2.25$; $p(A) = 1/(1 + e^{-2}) = 0.8808$." },
        { kind: "numeric", q: "MAML with L(θ) = (θ − a)², θ = 1, a = 2, α = 0.25, one inner step on the same loss. What is the full meta-gradient dL(θ')/dθ?", answer: -0.5, tolerance: 0.0001, explain: "Inner gradient $2(1-2) = -2$ → θ' = 1.5. $\\nabla_{\\theta'} = 2(1.5 - 2) = -1$; factor $(1 - 0.25\\cdot2) = 0.5$; meta-gradient −0.5." },
        { q: "In our experiment, why did the prototypical network trained on digits 0–4 do worse than raw pixels on digits 5–9?", options: ["The learning rate was too low", "With only 5 base classes the embedding specialised to them and discarded features needed for new classes", "Prototypical networks cannot do 1-shot learning", "Euclidean distance is invalid"], answer: 1, explain: "Few-shot transfer needs diverse base classes; pseudo-classes and augmentation lifted it from 0.388 to 0.640." },
        { q: "Which statement about curriculum learning is most accurate?", options: ["It always speeds up training", "It orders examples from easy to hard; its benefit is task-dependent", "It requires a teacher network", "It is the same as hard-negative mining"], answer: 1, explain: "Curricula help in some settings (noisy labels, RL) and not others; hard-example mining stresses the opposite end." },
        { q: "Zero-shot classification of an unseen class requires…", options: ["at least one labelled example", "a description of the class in a space the model was trained to map inputs into", "retraining the network", "a larger batch size"], answer: 1, explain: "Attributes, word embeddings or text prompts describe the class, and the model scores how well the input matches." },
      ] },
      { t: "exercise", title: "Distillation by hand", prompt: "Teacher logits $t = (4, 1, 0)$, student logits $z = (2, 2, 0)$, $T = 2$. Compute $p^T$, $q^T$, the distillation loss $T^2\\,\\mathrm{KL}(p^T\\|q^T)$ and its gradient. Which logit does a gradient step lower most?", hint: "$p^T = \\mathrm{softmax}(2, 0.5, 0)$ and $q^T = \\mathrm{softmax}(1, 1, 0)$.", solution: [
        { t: "steps", items: [
          { title: "Soft targets", tex: String.raw`p^T = \frac{(e^{2}, e^{0.5}, e^{0})}{7.3891 + 1.6487 + 1} = (0.7361,\ 0.1643,\ 0.0996)` },
          { title: "Student", tex: String.raw`q^T = \frac{(e, e, 1)}{2e + 1} = (0.4223,\ 0.4223,\ 0.1554)` },
          { title: "KL and loss", tex: String.raw`\mathrm{KL} = 0.7361\ln\tfrac{0.7361}{0.4223} + 0.1643\ln\tfrac{0.1643}{0.4223} + 0.0996\ln\tfrac{0.0996}{0.1554} = 0.4090 - 0.1551 - 0.0443 = 0.2096;\quad \mathcal L = 4\times0.2096 = 0.8386` },
          { title: "Gradient", tex: String.raw`T(q - p) = 2\,(-0.3138,\ 0.2580,\ 0.0558) = (-0.6276,\ 0.5161,\ 0.1115)` },
        ] },
        { t: "p", text: "A gradient step raises logit 0 and lowers logit 1 most: the student rates class 1 as likely as class 0, while the teacher gives it 0.16. (Values are rounded to 4 decimals; recompute with the from-scratch `kd_loss` function to check.)" },
      ] },
      { t: "exercise", title: "Coding: a curriculum experiment", prompt: "Using the student network and data of `distillation_digits.py`, train the student on the 1257 true labels in three ways for the same number of steps: (a) random order; (b) a curriculum — score each training image by the teacher's cross-entropy loss, train on the easiest 30% for the first third of the steps, the easiest 60% for the second third, then all data; (c) an anti-curriculum (hardest first). Report test accuracy over 3 seeds each. What do you conclude?", hint: "Difficulty: `F.cross_entropy(teacher_logits, YT, reduction='none')`; sort with `argsort()`. Draw mini-batches from the currently allowed index set.", solution: [
        {
          t: "code",
          lang: "text",
          code: "difficulty = F.cross_entropy(teacher_logits, YT, reduction='none')\norder = difficulty.argsort()                      # easiest first\n\ndef allowed(step, steps, order, anti=False):\n    frac = 0.3 if step < steps // 3 else 0.6 if step < 2 * steps // 3 else 1.0\n    o = order.flip(0) if anti else order\n    return o[: int(frac * len(o))]\n\n# inside the training loop of train_student:\n#   pool = allowed(step, steps, order, anti)    (or torch.arange(len(XT)) for random order)\n#   j = pool[torch.randint(0, len(pool), (64,))]\n#   loss = F.cross_entropy(s(noisy(XT[j])), YT[j])",
        },
        { t: "p", text: "Run each variant with several seeds and compare means and spreads. On small, clean data like digits, expect differences about the size of seed-to-seed noise; curricula matter more with noisy labels, very short budgets or truly hard sub-tasks. Lesson: compare against random order over multiple seeds before claiming a curriculum effect." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Distillation: $\\mathcal L = \\alpha T^2\\mathrm{KL}(p^T\\|q^T) + (1-\\alpha)\\mathrm{CE}(y, q)$, $p^T = \\mathrm{softmax}(t/T)$; gradient $T(q^T - p^T)$; $T^2$ keeps the scale; $T\\to\\infty$ gives logit matching $(z - t)/K$.",
        "Soft targets carry dark knowledge: with 100 transfer images, 0.880 (hard teacher labels) → 0.944 (T = 10).",
        "Curriculum: easy-to-hard weighting $w_t(x)$; benefits depend on the task; compare against random order.",
        "Few-shot: N-way K-shot episodes; prototypes $\\mu_c$ = mean support embedding; $p(c\\mid x) \\propto e^{-\\lVert f(x) - \\mu_c\\rVert^2}$ (a linear classifier); needs diverse base classes.",
        "MAML: $\\theta' = \\theta - \\alpha\\nabla\\mathcal L^{sup}$, meta-gradient $(I - \\alpha\\nabla^2\\mathcal L^{sup})\\nabla\\mathcal L^{qry}(\\theta')$; first-order versions drop the Hessian; sinusoids: one step gives 1.44 vs 2.78 MSE.",
        "Siamese / triplet networks: shared encoder + pair or triplet loss → open-set verification.",
        "Zero-shot: score inputs against class descriptions, $\\arg\\max_c \\cos(g(x), a_c)$.",
      ] },
    ],
    next: [
      { t: "list", items: [
        "[Multimodal Models, CLIP and RAG](/learn/m15/multimodal) — zero-shot learning at scale: images and text in one embedding space, classification by prompts, and retrieval.",
        "[Quantization, Pruning and Distillation](/learn/m18/model-optimization) — distillation as one of the deployment tools, with latency and accuracy measured.",
        "[Self-Supervised Learning](/learn/m15/self-supervised) — learning, without labels, the embeddings that few-shot and zero-shot methods rely on.",
      ] },
    ],
  },
};

export default lesson;
