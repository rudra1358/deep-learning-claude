import type { Lesson } from "@/lib/types";

const RNN_NUMPY = String.raw`import numpy as np

np.set_printoptions(precision=6, suppress=True)


class VanillaRNN:
    """Elman RNN:  h_t = tanh(W_xh x_t + W_hh h_(t-1) + b),   y_t = W_hy h_t + b_y."""

    def __init__(self, W_xh, W_hh, b, W_hy, b_y):
        self.W_xh, self.W_hh, self.b, self.W_hy, self.b_y = W_xh, W_hh, b, W_hy, b_y

    @classmethod
    def random(cls, n_in, n_h, n_out, rho=0.9, seed=0):
        rng = np.random.default_rng(seed)
        W_hh = rng.normal(0, 1, (n_h, n_h))
        W_hh *= rho / max(abs(np.linalg.eigvals(W_hh)))        # set the spectral radius explicitly
        return cls(rng.normal(0, 1, (n_h, n_in)), W_hh, np.zeros(n_h),
                   rng.normal(0, 1 / np.sqrt(n_h), (n_out, n_h)), np.zeros(n_out))

    def n_params(self):
        return sum(w.size for w in (self.W_xh, self.W_hh, self.b, self.W_hy, self.b_y))

    def forward(self, xs, h0=None):
        """xs: (T, n_in). Returns all hidden states (T, n_h) and outputs (T, n_out).
        The SAME weights are used at every step: that is the weight sharing across time."""
        h = np.zeros(self.W_hh.shape[0]) if h0 is None else h0
        hs, ys = [], []
        for x in xs:
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b)
            hs.append(h)
            ys.append(self.W_hy @ h + self.b_y)
        return np.array(hs), np.array(ys)


# ---- 1. the worked example from the Math section ----
net = VanillaRNN(W_xh=np.array([[0.5], [-0.3]]), W_hh=np.array([[0.4, -0.2], [0.1, 0.6]]),
                 b=np.array([0.0, 0.1]), W_hy=np.array([[0.8, -0.5]]), b_y=np.array([0.1]))
hs, ys = net.forward(np.array([[1.0], [0.5], [-1.0]]))
for t in range(3):
    print(f"t={t + 1}: h = {hs[t]}  y = {ys[t]}")
print("parameters:", net.n_params(), "= n_h*n_in + n_h*n_h + n_h + n_out*n_h + n_out = 2 + 4 + 2 + 2 + 1")

# ---- 2. parameter count does not depend on the sequence length ----
big = VanillaRNN.random(n_in=64, n_h=128, n_out=10)
print("n_in=64, n_h=128, n_out=10:", big.n_params(), "parameters for ANY sequence length")

# ---- 3. memory probe: how much does h_t still depend on x_1? ----
rng = np.random.default_rng(1)
xs = rng.normal(size=(40, 1))
xs_perturbed = xs.copy()
xs_perturbed[0] += 0.1                                         # nudge only the first input
print("||h_t(perturbed) - h_t||  at t = 1, 5, 10, 20, 40")
for rho in [0.5, 0.9, 1.5]:
    r = VanillaRNN.random(1, 16, 1, rho=rho, seed=2)
    diff = np.linalg.norm(r.forward(xs_perturbed)[0] - r.forward(xs)[0], axis=1)
    print(f"  spectral radius {rho}: " + "  ".join(f"{diff[t - 1]:.1e}" for t in (1, 5, 10, 20, 40)))

# ---- 4. stateful processing: chunks + carried state == one long pass ----
r = VanillaRNN.random(1, 16, 1, seed=3)
full, _ = r.forward(xs)
h, carried, reset = None, [], []
for chunk in np.split(xs, 4):                                  # 4 chunks of 10 steps
    out, _ = r.forward(chunk, h0=h)
    carried.append(out)
    h = out[-1]                                                # carry the last state into the next chunk
    reset.append(r.forward(chunk)[0])                          # stateless: start every chunk from zeros
print("max |difference| vs one pass, carried state:", np.abs(np.concatenate(carried) - full).max())
print("max |difference| vs one pass, reset state  :", round(np.abs(np.concatenate(reset) - full).max(), 4))`;

const RNN_NUMPY_OUT = String.raw`t=1: h = [ 0.462117 -0.197375]  y = [0.568381]
t=2: h = [ 0.441685 -0.121609]  y = [0.514152]
t=3: h = [-0.290401  0.355044]  y = [-0.309843]
parameters: 11 = n_h*n_in + n_h*n_h + n_h + n_out*n_h + n_out = 2 + 4 + 2 + 2 + 1
n_in=64, n_h=128, n_out=10: 25994 parameters for ANY sequence length
||h_t(perturbed) - h_t||  at t = 1, 5, 10, 20, 40
  spectral radius 0.5: 3.1e-01  2.6e-03  2.9e-05  4.4e-09  8.9e-17
  spectral radius 0.9: 3.1e-01  2.0e-02  3.3e-03  9.5e-05  1.8e-10
  spectral radius 1.5: 3.1e-01  7.6e-02  5.2e-02  2.9e-02  4.5e-05
max |difference| vs one pass, carried state: 0.0
max |difference| vs one pass, reset state  : 0.69`;

const RNN_TORCH_CHECK = String.raw`import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, N_IN, N_H, LAYERS = 3, 5, 4, 6, 2

rnn = nn.RNN(input_size=N_IN, hidden_size=N_H, num_layers=LAYERS, nonlinearity="tanh", batch_first=True)
for name, p in rnn.named_parameters():
    print(f"{name:14s} {tuple(p.shape)}")
print("total parameters:", sum(p.numel() for p in rnn.parameters()))

x = torch.randn(B, T, N_IN)                    # batch_first=True  ->  (batch, time, features)
h0 = torch.zeros(LAYERS, B, N_H)               # h0 is ALWAYS (num_layers * num_directions, batch, hidden)
output, h_n = rnn(x, h0)
print("output:", tuple(output.shape), " <- top layer's h_t at EVERY step")
print("h_n   :", tuple(h_n.shape), " <- EVERY layer's h at the LAST step")
print("output[:, -1] == h_n[-1]:", torch.allclose(output[:, -1], h_n[-1]))


def numpy_rnn(x, params, layers):
    """Plain NumPy forward pass using PyTorch's weights (note: PyTorch has TWO bias vectors)."""
    inp = x
    last = []
    for l in range(layers):
        W_ih, W_hh = params[f"weight_ih_l{l}"], params[f"weight_hh_l{l}"]
        b = params[f"bias_ih_l{l}"] + params[f"bias_hh_l{l}"]
        h = np.zeros((x.shape[0], W_hh.shape[0]))
        outs = []
        for t in range(x.shape[1]):
            h = np.tanh(inp[:, t] @ W_ih.T + h @ W_hh.T + b)
            outs.append(h)
        inp = np.stack(outs, axis=1)           # this layer's outputs are the next layer's inputs
        last.append(h)
    return inp, np.stack(last)


params = {k: v.detach().double().numpy() for k, v in rnn.named_parameters()}
out_np, hn_np = numpy_rnn(x.double().numpy(), params, LAYERS)
print("max |torch - numpy| output:", f"{np.abs(output.detach().numpy() - out_np).max():.1e}",
      " h_n:", f"{np.abs(h_n.detach().numpy() - hn_np).max():.1e}")

# Copy the worked-example weights into a 1-layer nn.RNN and reproduce the hand calculation
cell = nn.RNN(1, 2, batch_first=True)
with torch.no_grad():
    cell.weight_ih_l0.copy_(torch.tensor([[0.5], [-0.3]]))
    cell.weight_hh_l0.copy_(torch.tensor([[0.4, -0.2], [0.1, 0.6]]))
    cell.bias_ih_l0.copy_(torch.tensor([0.0, 0.1]))
    cell.bias_hh_l0.zero_()
    out, _ = cell(torch.tensor([[[1.0], [0.5], [-1.0]]]))
print("worked example via nn.RNN:", [[round(v, 6) for v in row] for row in out[0].tolist()])`;

const RNN_TORCH_CHECK_OUT = String.raw`weight_ih_l0   (6, 4)
weight_hh_l0   (6, 6)
bias_ih_l0     (6,)
bias_hh_l0     (6,)
weight_ih_l1   (6, 6)
weight_hh_l1   (6, 6)
bias_ih_l1     (6,)
bias_hh_l1     (6,)
total parameters: 156
output: (3, 5, 6)  <- top layer's h_t at EVERY step
h_n   : (2, 3, 6)  <- EVERY layer's h at the LAST step
output[:, -1] == h_n[-1]: True
max |torch - numpy| output: 7.2e-08  h_n: 5.0e-08
worked example via nn.RNN: [[0.462117, -0.197375], [0.441685, -0.121609], [-0.290401, 0.355044]]`;

const RNN_SINE = String.raw`import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
NOISE, T = 0.1, 60


def sines(n, rng):
    """n noisy sine waves, each with its own random frequency and phase. Returns (noisy, clean)."""
    t = np.arange(T + 1)
    freq = rng.uniform(0.03, 0.12, (n, 1))               # cycles per step
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    clean = np.sin(2 * np.pi * freq * t + phase)
    return clean + NOISE * rng.normal(size=clean.shape), clean


rng = np.random.default_rng(0)
train_noisy, _ = sines(2000, rng)
test_noisy, test_clean = sines(500, rng)


# ---- baselines with a FIXED window of the last k values ----
def window_xy(data, k):
    X = np.stack([data[:, t - k:t] for t in range(k, T + 1)], axis=1).reshape(-1, k)
    y = np.stack([data[:, t] for t in range(k, T + 1)], axis=1).reshape(-1)
    return X, y


START = 20                                               # score every model on the same steps 20..60
print(f"one-step-ahead test MSE on steps {START}..{T} (noise floor = {NOISE ** 2:.4f})")
print(f"  persistence (predict x_t)      {np.mean((test_noisy[:, START - 1:T] - test_noisy[:, START:]) ** 2):.4f}")
for k in [2, 5, 20]:
    X, y = window_xy(train_noisy, k)
    A = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(A, y, rcond=None)[0]             # linear autoregression AR(k), least squares
    pred = np.stack([np.c_[test_noisy[:, t - k:t], np.ones(500)] @ w for t in range(START, T + 1)], axis=1)
    print(f"  linear window k={k:2d}             {np.mean((pred - test_noisy[:, START:]) ** 2):.4f}")


# ---- an MLP on a fixed window of k values (nonlinear, but still blind beyond k) ----
def train_mlp(k, steps=1500):
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(k, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(mlp.parameters(), lr=3e-3)
    X, y = (torch.tensor(a, dtype=torch.float32) for a in window_xy(train_noisy, k))
    for _ in range(steps):
        idx = torch.randint(0, len(X), (256,))
        loss = ((mlp(X[idx]).squeeze(-1) - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xt = torch.tensor(np.stack([test_noisy[:, t - k:t] for t in range(START, T + 1)], axis=1), dtype=torch.float32)
        return ((mlp(Xt).squeeze(-1).numpy() - test_noisy[:, START:]) ** 2).mean()


for k in [5, 20]:
    print(f"  MLP window k={k:2d} (64 units)     {train_mlp(k):.4f}")


# ---- a vanilla RNN reads the whole history, one step at a time ----
class Forecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)                 # output head y_t = W_hy h_t + b_y

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        return self.head(out), h


model = Forecaster()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
data = torch.tensor(train_noisy, dtype=torch.float32).unsqueeze(-1)   # (2000, 61, 1)
for step in range(1500):
    batch = data[torch.randint(0, 2000, (64,))]
    pred, _ = model(batch[:, :-1])                       # teacher forcing: true x_1..x_T as inputs
    loss = ((pred - batch[:, 1:]) ** 2).mean()           # targets: x_2..x_(T+1)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

test = torch.tensor(test_noisy, dtype=torch.float32).unsqueeze(-1)
with torch.no_grad():
    pred, _ = model(test[:, :-1])
    mse = ((pred[:, START - 1:] - test[:, START:]) ** 2).mean().item()
print(f"  vanilla RNN (32 units)         {mse:.4f}")

# ---- free-running generation: feed predictions back in as inputs ----
with torch.no_grad():
    _, h = model(test[:, :39])                           # warm up on the observed x_0..x_38
    x, gen = test[:, 39:40], []                          # last observed value x_39
    for _ in range(20):
        x, h = model(x, h)                               # next input = own prediction
        gen.append(x)
    gen = torch.cat(gen, dim=1).squeeze(-1).numpy()
err = np.abs(gen - test_clean[:, 40:60]).mean(axis=0)
print("free-running, mean |error| vs clean signal at 1, 5, 10, 20 steps ahead:",
      "  ".join(f"{err[i - 1]:.3f}" for i in (1, 5, 10, 20)))`;

const RNN_SINE_OUT = String.raw`one-step-ahead test MSE on steps 20..60 (noise floor = 0.0100)
  persistence (predict x_t)      0.1366
  linear window k= 2             0.0543
  linear window k= 5             0.0450
  linear window k=20             0.0341
  MLP window k= 5 (64 units)     0.0303
  MLP window k=20 (64 units)     0.0140
  vanilla RNN (32 units)         0.0143
free-running, mean |error| vs clean signal at 1, 5, 10, 20 steps ahead: 0.050  0.068  0.106  0.208`;

const RNN_CHAR = String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(0)

# Opening of "Alice's Adventures in Wonderland" (Lewis Carroll, 1865; public domain)
TEXT = ("Alice was beginning to get very tired of sitting by her sister on the bank, and of having "
        "nothing to do: once or twice she had peeped into the book her sister was reading, but it had "
        "no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without "
        "pictures or conversations?'")
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in TEXT])
V, H, SEQ = len(chars), 128, 40
print(f"{len(TEXT)} characters, vocabulary of {V}")


class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(V, H, batch_first=True)
        self.head = nn.Linear(H, V)                       # logits over the next character

    def forward(self, idx, h=None):
        out, h = self.rnn(F.one_hot(idx, V).float(), h)   # characters -> one-hot vectors
        return self.head(out), h


model = CharRNN()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(601):
    starts = torch.randint(0, len(TEXT) - SEQ - 1, (32,))
    x = torch.stack([data[s:s + SEQ] for s in starts])          # teacher forcing: true characters in
    y = torch.stack([data[s + 1:s + SEQ + 1] for s in starts])  # targets: the same text shifted by one
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 150 == 0:
        print(f"step {step:3d}  cross-entropy {loss.item():.3f} nats/char")


def sample(prompt, n, temperature, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        _, h = model(torch.tensor([[stoi[c] for c in prompt[:-1]]]))   # read the prompt
        idx, out = torch.tensor([[stoi[prompt[-1]]]]), prompt
        for _ in range(n):
            logits, h = model(idx, h)
            if temperature == 0:
                nxt = logits[0, -1].argmax()                            # greedy decoding
            else:
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=g)[0]
            out += chars[nxt]
            idx = nxt.view(1, 1)                                        # feed the sample back in
    return out


print("greedy       :", sample("Alice was ", 60, 0, 0))
print("temperature 1:", sample("Alice was ", 60, 1.0, 1))
print("temperature 2:", sample("Alice was ", 60, 2.0, 1))
print("new prompt   :", sample("the sister had ", 60, 0, 0))`;

const RNN_CHAR_OUT = String.raw`302 characters, vocabulary of 27
step   0  cross-entropy 3.275 nats/char
step 150  cross-entropy 0.211 nats/char
step 300  cross-entropy 0.118 nats/char
step 450  cross-entropy 0.103 nats/char
step 600  cross-entropy 0.106 nats/char
greedy       : Alice was beginning to get very tired of sitting by her sister on the
temperature 1: Alice was beginning to get very tired of sitting by her sister on the
temperature 2: Alice was beginning to g,c tvokghe hre perhatbod oute:donce she gld ve
new prompt   : the sister had no pictures or conversations in it, 'and what is the use of`;

const RNN_KERAS = String.raw`import numpy as np
import tensorflow as tf

B, T, N_IN, N_H = 3, 5, 4, 6
x = np.random.default_rng(0).normal(size=(B, T, N_IN)).astype("float32")

# Keras is always batch-major: (batch, time, features)
layer = tf.keras.layers.SimpleRNN(N_H, activation="tanh", return_sequences=True, return_state=True)
seq, h_T = layer(x)                      # seq: h_t at every step, h_T: the final state
print(seq.shape, h_T.shape)              # (3, 5, 6) (3, 6)

kernel, recurrent_kernel, bias = layer.get_weights()
print(kernel.shape, recurrent_kernel.shape, bias.shape)   # (4, 6) (6, 6) (6,)
# Keras stores W_xh^T and W_hh^T (inputs multiply from the left) and ONE bias vector.

h = np.zeros((B, N_H), dtype="float32")
for t in range(T):                       # h_t = tanh(x_t K + h_(t-1) R + b)
    h = np.tanh(x[:, t] @ kernel + h @ recurrent_kernel + bias)
print("max |numpy - keras|:", np.abs(h - h_T.numpy()).max())

# A stacked (deep) many-to-one model: the first layer must return the whole sequence
model = tf.keras.Sequential([
    tf.keras.Input(shape=(None, N_IN)),              # None = any sequence length
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),                   # returns only h_T
    tf.keras.layers.Dense(1, activation="sigmoid"),  # one prediction per sequence
])
model.summary()`;

const lesson: Lesson = {
  id: "m09/rnn-basics",
  title: "Sequential Data and Vanilla RNNs",
  summary:
    "Why sequences need a model with memory, the recurrence h_t = tanh(W_xh x_t + W_hh h_(t−1) + b), weight sharing across time, unrolling, output heads and task patterns — with a hand-computed three-step forward pass, a NumPy implementation matched against torch.nn.RNN, and real forecasting and character-level experiments.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m03/multilayer-networks", "m04/forward-propagation", "m03/activation-functions", "m02/eigen-svd"],
  next: ["m09/bptt", "m09/lstm", "m09/gru-variants"],
  terms: ["rnn", "hidden-state", "recurrent-connection", "unrolling", "weight-sharing-time", "teacher-forcing", "stateful-rnn", "sequence-to-sequence"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "Much of the world's data arrives as a **sequence**: an ordered list $x_1, x_2, \\dots, x_T$ in which the order carries meaning. Text is a sequence of characters or tokens; speech is a sequence of audio frames; stock prices, electricity demand and patient vital signs are time series; accelerometer and machine sensors produce continuous streams. Two facts make sequences special: the length $T$ varies from example to example (a tweet versus a novel), and the meaning of an element depends on what came before it (\"bank\" after \"river\" versus after \"savings\")." },
      { t: "p", text: "A **recurrent neural network** (RNN) processes such data one element at a time while carrying a fixed-size vector, the **hidden state** $h_t$, from each step to the next. At every step the same small network combines the new input $x_t$ with the previous state $h_{t-1}$ to produce the new state: $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$. Because the weights are the same at every step, one set of parameters handles sequences of any length, and because $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, and so on, the state can in principle carry information from the very first input to the last." },
      { t: "p", text: "This simple form is called the **vanilla** or **Elman** RNN, after Jeffrey Elman's 1990 paper *Finding Structure in Time*, which showed that such a network trained to predict the next element discovers word boundaries and grammatical categories on its own. (Jordan, 1986, had proposed a closely related network that fed back the previous *output* instead of the hidden state.) Recurrent networks went on to power statistical language models (Mikolov and colleagues' RNN language model, 2010), handwriting generation (Graves, 2013), speech recognition and machine translation — mostly in their gated forms, the [LSTM](/learn/m09/lstm) and the [GRU](/learn/m09/gru-variants)." },
      { t: "p", text: "This lesson builds the vanilla RNN completely: the recurrence, what weight sharing and unrolling mean, how outputs are read out, which input–output patterns it supports, how state is handled during training, how it generates sequences, and what it costs. The next lesson, [Backpropagation Through Time](/learn/m09/bptt), derives how it is trained and why it struggles to remember for long — the problem the LSTM was invented to fix." },
      { t: "callout", kind: "key", text: "An RNN is **one** small network applied at every time step, with its own previous output (the hidden state) fed back as an extra input. Unrolled over $T$ steps it is a $T$-layer feed-forward network whose layers all share the same weights." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Multilayer networks](/learn/m03/multilayer-networks) — a layer computes $f(Wx + b)$; you need to be comfortable with weight matrices of shape (outputs × inputs) and with stacking layers.",
        "[Forward propagation](/learn/m04/forward-propagation) — evaluating a network layer by layer and caching intermediate values. An unrolled RNN is evaluated exactly the same way, one time step per \"layer\".",
        "[Activation functions](/learn/m03/activation-functions) — the shape of $\\tanh$: output in $(-1, 1)$, $\\tanh(0) = 0$, slope $1 - \\tanh^2(z)$, which is close to 0 when $|z|$ is large (saturation).",
        "[Vectors and matrices](/learn/m02/vectors-matrices) — matrix–vector products and their shapes; a batch of inputs stored as rows.",
        "[Eigenvalues and SVD](/learn/m02/eigen-svd) — the **spectral radius** $\\rho(W)$ (largest $|\\lambda|$) controls what repeated multiplication by the same matrix does to a vector. We use it to explain how fast an RNN forgets.",
        "[Convolution](/learn/m08/convolution) (optional) — the idea of **weight sharing**: a CNN reuses one kernel at every spatial position; an RNN reuses one cell at every time position.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "Reading with a single index card", text: "Imagine reading a long story but being allowed to keep notes only on one small index card. After each sentence you rewrite the card: you look at what the card says now and at the new sentence, and write a fresh summary that fits on the card. You never re-read old sentences; everything you know about them is whatever survived on the card. The rules you use to rewrite the card are the same for every sentence. That card is the hidden state $h_t$, the fixed rewriting rule is the RNN cell with its shared weights, and the fact that old details gradually get squeezed out of a small card is exactly the RNN's fading memory." },
      { t: "p", text: "Compare this with a feed-forward network looking at a **fixed window** of the last $k$ elements. The window model sees those $k$ elements perfectly but nothing older — the 21st most recent word does not exist for a window of 20. The RNN sees everything that ever happened, but only through its compressed summary. It also does not need to know $k$ in advance: the same cell runs for 5 steps or 5000." },
      { t: "p", text: "The update has two ingredients. The input term $W_{xh}x_t$ says \"what is new\"; the recurrent term $W_{hh}h_{t-1}$ says \"what I remember, mixed by my rules\". They are added and squashed by $\\tanh$ so that the state stays in $(-1, 1)$. If the recurrent weights are small, old information is quickly overwritten by new inputs (short memory). If they are large, the state keeps echoing its past (long, but possibly chaotic, memory). The simulator below lets you see both by perturbing the first input and watching how far the change travels." },
      { t: "p", text: "Finally, the network has to produce something. An **output head** — usually a single linear layer — reads the state at whichever steps the task needs: at every step (label each word), only at the end (classify the whole review), or at every step after the input has been read (generate a translation)." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "Why not just use an MLP or a CNN?" },
      { t: "p", text: "An MLP needs a fixed input size, so a sequence must be cut into windows of $k$ elements and flattened. This has three problems. (1) **Fixed context**: nothing older than $k$ steps can influence the prediction, and choosing $k$ is a guess about the data. (2) **Parameters grow with the window**: the first layer has $k \\cdot n_{in}$ inputs, so doubling the context doubles its weights. (3) **No sharing across positions**: a pattern learned at window position 3 must be learned again at position 7. A 1-D CNN fixes (3) by sliding a shared kernel along time, and stacking or dilating convolutions widens the receptive field — but the context is still bounded by the architecture. An RNN removes the bound: the state is updated at every step, so every past input can in principle still influence the current output, with a parameter count that does not depend on the sequence length." },
      { t: "p", text: "That said, a window model with a large enough window can be very competitive when the useful context really is short — the forecasting experiment in the Framework section measures exactly this." },
      { t: "h", text: "One time step, in full" },
      { t: "steps", items: [
        { title: "Start with an initial state", text: "Usually $h_0 = \\mathbf{0}$ (a vector of zeros of size $n_h$). It can also be learned, or carried over from a previous chunk of the same stream (stateful processing, below)." },
        { title: "Project the input", text: "Multiply the input vector by the input weights: an $n_h$-vector saying how the new element should push each hidden unit.", tex: String.raw`a_t = W_{xh}\,x_t \qquad (n_h \times n_{in})(n_{in}) \to (n_h)` },
        { title: "Project the previous state", text: "Multiply the previous hidden state by the recurrent weights. This is the only place where the past enters.", tex: String.raw`r_t = W_{hh}\,h_{t-1} \qquad (n_h \times n_h)(n_h) \to (n_h)` },
        { title: "Add and apply the bias", tex: String.raw`z_t = a_t + r_t + b` },
        { title: "Squash", text: "Apply $\\tanh$ element-wise. The new state lives in $(-1,1)^{n_h}$, which keeps repeated application from blowing up the forward values.", tex: String.raw`h_t = \tanh(z_t)` },
        { title: "Read out (if this step needs an output)", text: "A linear output head turns the state into a prediction: a number for regression, or logits followed by softmax for classification over a vocabulary.", tex: String.raw`\hat y_t = W_{hy}\,h_t + b_y` },
        { title: "Hand the state to the next step", text: "$h_t$ becomes the $h_{t-1}$ of step $t+1$, and the same weights $W_{xh}, W_{hh}, b, W_{hy}, b_y$ are used again." },
      ] },
      { t: "h", text: "Weight sharing and unrolling" },
      { t: "p", text: "Drawing the loop as a chain gives the **unrolled** network: one copy of the cell per time step, each copy connected to the next by the hidden state. Unrolled over $T$ steps, an RNN is an ordinary feed-forward network with $T$ hidden layers, with two peculiarities: every layer has the **same** weights (they are tied), and every layer receives an extra input $x_t$ from the side. Training uses exactly this picture: run the unrolled graph forward, then backpropagate through it — that is [backpropagation through time](/learn/m09/bptt)." },
      { t: "math", tex: String.raw`h_3 = \tanh\!\Big(W_{xh}x_3 + W_{hh}\tanh\!\big(W_{xh}x_2 + W_{hh}\tanh(W_{xh}x_1 + W_{hh}h_0 + b) + b\big) + b\Big)`, caption: "Unrolling three steps: the same three parameters appear at every depth, and $x_1$ reaches $h_3$ through two applications of $W_{hh}$ and three of $\\tanh$." },
      { t: "p", text: "Weight sharing is what makes the model **length-agnostic** and **position-agnostic**: a rule learned from the pattern \"not good\" at position 4 applies at position 40. It also means that an RNN with 26,000 parameters processes a 10-step and a 10,000-step sequence with the same 26,000 parameters — the scratch code below prints this. What does grow with $T$ is compute (one cell evaluation per step, strictly in order) and, during training, memory (all $T$ states must be stored for the backward pass)." },
      { t: "h", text: "Input–output patterns" },
      { t: "table", head: ["Pattern", "What is read and produced", "Example tasks"], rows: [
        ["One-to-one", "One input, one output (no recurrence needed)", "Image classification — included only for comparison"],
        ["One-to-many", "A single input (often used to set $h_0$), then a sequence of outputs, each fed back as the next input", "Image captioning, music generation from a seed"],
        ["Many-to-one", "Read the whole sequence, use only $\\hat y_T$ (or a pooling of all $h_t$)", "Sentiment classification, spoken-keyword detection, activity recognition from a sensor window"],
        ["Many-to-many, aligned", "One output per input, same length", "Part-of-speech tagging, named-entity recognition, frame-level speech labels, next-step forecasting"],
        ["Many-to-many, unaligned (sequence-to-sequence)", "An **encoder** RNN reads the input; its final state initialises a **decoder** RNN that emits an output sequence of a different length", "Machine translation, summarisation, speech-to-text"],
      ] },
      { t: "p", text: "The encoder–decoder pattern (Sutskever, Vinyals and Le, 2014; Cho and colleagues, 2014) squeezes the whole input into one fixed-size vector, which becomes a bottleneck for long inputs. Letting the decoder look back at **all** encoder states instead is the idea of **attention** — see [Attention: the core idea](/learn/m10/attention-intro), which grew directly out of this limitation." },
      { t: "h", text: "Stateful vs stateless processing" },
      { t: "p", text: "**Stateless** (the default): every sequence, or every batch, starts from $h_0 = 0$. Correct when the training examples are independent (separate reviews, separate utterances)." },
      { t: "p", text: "**Stateful**: a very long stream (a year of sensor readings, a whole book) is cut into consecutive chunks, and the final state of chunk $i$ is used as $h_0$ of chunk $i+1$. The forward computation is then identical to processing the stream in one pass — the scratch code verifies this to the last bit — while each training step only needs memory for one chunk. Two rules: the batch must be arranged so that row $j$ of batch $i+1$ continues row $j$ of batch $i$; and the carried state must be **detached** from the computational graph (`h = h.detach()` in PyTorch), otherwise the backward pass tries to run through every previous chunk. Cutting the gradient while keeping the value is **truncated BPTT**, analysed in the [next lesson](/learn/m09/bptt)." },
      { t: "h", text: "Generating sequences: teacher forcing and sampling" },
      { t: "p", text: "To train a model that predicts the next element, feed it the **true** sequence $x_1 \\dots x_{T-1}$ and ask for $x_2 \\dots x_T$ at every step. Using the ground truth as input during training, rather than the model's own previous prediction, is called **teacher forcing** (Williams and Zipser, 1989). It makes every time step a supervised example and lets all steps be trained in one pass." },
      { t: "p", text: "At **generation** time there is no ground truth: the model's output at step $t$ is turned into the input of step $t+1$ — for text, by sampling a character from the softmax distribution (or taking the most likely one, *greedy decoding*); for a time series, by feeding back the predicted value. The model is now conditioned on its *own* outputs, which it never saw during training, so errors can compound (**exposure bias**). The forecasting experiment below measures this: the free-running error grows from 0.050 one step ahead to 0.208 twenty steps ahead. Remedies such as scheduled sampling (Bengio and colleagues, 2015) gradually replace true inputs with model samples during training." },
      { t: "h", text: "Going deeper: stacked RNNs (preview)" },
      { t: "p", text: "Several RNN layers can be stacked: the sequence of hidden states $h^{(1)}_1 \\dots h^{(1)}_T$ produced by layer 1 is the input sequence of layer 2, and so on. This adds depth *per time step* (more processing between input and output) on top of the depth *through time* that unrolling already creates. In PyTorch it is `num_layers=2`; the hidden state then has one slice per layer. Stacked and bidirectional RNNs are covered in [GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants)." },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Definition" },
      { t: "math", tex: String.raw`\boxed{\;h_t = \tanh\!\left(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b\right), \qquad \hat y_t = W_{hy}\,h_t + b_y, \qquad t = 1,\dots,T\;}` },
      { t: "list", items: [
        "$x_t \\in \\mathbb{R}^{n_{in}}$ — the input at time step $t$ (a number, a feature vector, or a one-hot / embedding vector for a token).",
        "$h_t \\in \\mathbb{R}^{n_h}$ — the hidden state after reading $x_1, \\dots, x_t$; $h_0$ is the initial state (usually $\\mathbf 0$).",
        "$W_{xh} \\in \\mathbb{R}^{n_h \\times n_{in}}$ — input-to-hidden weights; $W_{hh} \\in \\mathbb{R}^{n_h \\times n_h}$ — hidden-to-hidden (recurrent) weights; $b \\in \\mathbb{R}^{n_h}$ — hidden bias.",
        "$W_{hy} \\in \\mathbb{R}^{n_{out} \\times n_h}$, $b_y \\in \\mathbb{R}^{n_{out}}$ — the output head; $\\hat y_t \\in \\mathbb{R}^{n_{out}}$ is the prediction (or logits) at step $t$.",
        "$\\tanh$ acts element-wise. All weights are **independent of $t$**: that is weight sharing.",
      ] },
      { t: "p", text: "For a mini-batch of $B$ sequences stored as rows, $X_t \\in \\mathbb{R}^{B \\times n_{in}}$ and $H_t \\in \\mathbb{R}^{B\\times n_h}$, the same equation reads $H_t = \\tanh(X_t W_{xh}^\\top + H_{t-1}W_{hh}^\\top + \\mathbf 1 b^\\top)$ — this is the form used in code. The two products are sometimes merged into one by concatenation: $h_t = \\tanh(W[x_t; h_{t-1}] + b)$ with $W = [W_{xh}\\;W_{hh}]$." },
      { t: "h", text: "Parameter count" },
      { t: "math", tex: String.raw`P = \underbrace{n_h n_{in}}_{W_{xh}} + \underbrace{n_h^2}_{W_{hh}} + \underbrace{n_h}_{b} + \underbrace{n_{out} n_h}_{W_{hy}} + \underbrace{n_{out}}_{b_y}`, caption: "Independent of the sequence length T. PyTorch's nn.RNN keeps two bias vectors (bias_ih and bias_hh, simply added), so it reports n_h more per layer." },
      { t: "p", text: "Example: $n_{in} = 64$, $n_h = 128$, $n_{out} = 10$ gives $8192 + 16384 + 128 + 1280 + 10 = 25{,}994$ parameters. The recurrent matrix dominates as soon as $n_h > n_{in}$, and the cost of one step is $O(n_h^2 + n_h n_{in})$ multiply-adds, so a sequence costs $O(T(n_h^2 + n_h n_{in}))$ — linear in $T$, but strictly sequential." },
      { t: "h", text: "Output heads and losses" },
      { t: "p", text: "For **regression** (next value of a series), $\\hat y_t$ is used directly with squared error. For **classification over $K$ classes** (next character, a word's tag), the head produces logits and $p(\\cdot \\mid x_{\\le t}) = \\operatorname{softmax}(W_{hy}h_t + b_y)$, trained with cross-entropy. For language modelling the targets are the inputs shifted by one, and the model defines a distribution over whole sequences by the chain rule of probability:" },
      { t: "math", tex: String.raw`p(x_1,\dots,x_T) = \prod_{t=1}^{T} p(x_t \mid x_1,\dots,x_{t-1}) \approx \prod_{t=1}^{T} \operatorname{softmax}\!\left(W_{hy}h_{t-1} + b_y\right)_{x_t}, \qquad \mathcal L = -\sum_{t} \log p(x_t \mid x_{<t})` },
      { t: "p", text: "The total loss of a sequence is the **sum** (or mean) of per-step losses over the steps that have targets — every step for aligned tasks, only $t = T$ for many-to-one." },
      { t: "h", text: "Worked example: three time steps by hand" },
      { t: "p", text: "Take $n_{in} = 1$, $n_h = 2$, $n_{out} = 1$ with" },
      { t: "math", tex: String.raw`W_{xh} = \begin{pmatrix}0.5\\-0.3\end{pmatrix},\quad W_{hh} = \begin{pmatrix}0.4 & -0.2\\ 0.1 & 0.6\end{pmatrix},\quad b = \begin{pmatrix}0\\0.1\end{pmatrix},\quad W_{hy} = \begin{pmatrix}0.8 & -0.5\end{pmatrix},\quad b_y = 0.1` },
      { t: "p", text: "and the input sequence $x = (1.0,\\ 0.5,\\ -1.0)$, starting from $h_0 = (0, 0)$. (The same network is used for the backward pass in [BPTT](/learn/m09/bptt).)" },
      { t: "steps", items: [
        { title: "t = 1: input and recurrent terms", tex: String.raw`W_{xh}x_1 = \begin{pmatrix}0.5\\-0.3\end{pmatrix},\qquad W_{hh}h_0 = \begin{pmatrix}0\\0\end{pmatrix}` },
        { title: "t = 1: pre-activation and state", tex: String.raw`z_1 = \begin{pmatrix}0.5\\-0.3+0.1\end{pmatrix} = \begin{pmatrix}0.5\\-0.2\end{pmatrix},\qquad h_1 = \tanh(z_1) = \begin{pmatrix}0.462117\\-0.197375\end{pmatrix}` },
        { title: "t = 1: output", tex: String.raw`\hat y_1 = 0.8(0.462117) - 0.5(-0.197375) + 0.1 = 0.369694 + 0.098688 + 0.1 = 0.568381` },
        { title: "t = 2: recurrent term (the memory of step 1)", tex: String.raw`W_{hh}h_1 = \begin{pmatrix}0.4(0.462117) - 0.2(-0.197375)\\ 0.1(0.462117) + 0.6(-0.197375)\end{pmatrix} = \begin{pmatrix}0.184847 + 0.039475\\ 0.046212 - 0.118425\end{pmatrix} = \begin{pmatrix}0.224322\\-0.072213\end{pmatrix}` },
        { title: "t = 2: pre-activation and state", tex: String.raw`z_2 = \underbrace{\begin{pmatrix}0.25\\-0.15\end{pmatrix}}_{W_{xh}x_2} + \begin{pmatrix}0.224322\\-0.072213\end{pmatrix} + \begin{pmatrix}0\\0.1\end{pmatrix} = \begin{pmatrix}0.474322\\-0.122213\end{pmatrix},\qquad h_2 = \begin{pmatrix}0.441685\\-0.121609\end{pmatrix}` },
        { title: "t = 2: output", tex: String.raw`\hat y_2 = 0.353348 + 0.060805 + 0.1 = 0.514152` },
        { title: "t = 3: recurrent term", tex: String.raw`W_{hh}h_2 = \begin{pmatrix}0.176674 + 0.024322\\ 0.044169 - 0.072965\end{pmatrix} = \begin{pmatrix}0.200996\\-0.028797\end{pmatrix}` },
        { title: "t = 3: pre-activation and state", tex: String.raw`z_3 = \begin{pmatrix}-0.5\\0.3\end{pmatrix} + \begin{pmatrix}0.200996\\-0.028797\end{pmatrix} + \begin{pmatrix}0\\0.1\end{pmatrix} = \begin{pmatrix}-0.299004\\0.371203\end{pmatrix},\qquad h_3 = \begin{pmatrix}-0.290401\\0.355044\end{pmatrix}` },
        { title: "t = 3: output", tex: String.raw`\hat y_3 = -0.232321 - 0.177522 + 0.1 = -0.309843` },
      ] },
      { t: "p", text: "Notice what step 2 shows: without memory, $x_2 = 0.5$ alone would give $z_2 = (0.25, -0.05)$; the recurrent term adds $(0.224, -0.072)$, so almost half of $h_2$'s first unit is \"remembered\" from $x_1$. At step 3 the input flips sign and dominates, but $W_{hh}h_2$ still shifts the result. All nine numbers are reproduced by the NumPy code and by `torch.nn.RNN` with these weights copied in (Framework section)." },
      { t: "h", text: "The hidden state is a lossy summary" },
      { t: "p", text: "$h_t$ is a deterministic function of the whole prefix, $h_t = F_\\theta(x_1, \\dots, x_t)$, but it has only $n_h$ numbers in $(-1,1)$, regardless of $t$. A long prefix cannot be stored losslessly, so the network must learn *what to keep*. How quickly it forgets is governed by the step-to-step Jacobian: a small change $\\Delta h_{t-1}$ produces" },
      { t: "math", tex: String.raw`\Delta h_t \approx \frac{\partial h_t}{\partial h_{t-1}}\,\Delta h_{t-1} = \operatorname{diag}\!\left(1 - h_t^2\right) W_{hh}\,\Delta h_{t-1}, \qquad \Delta h_{t+k} \approx \Big(\prod_{j=t+1}^{t+k} \operatorname{diag}(1 - h_j^2)\, W_{hh}\Big)\Delta h_t` },
      { t: "p", text: "Each factor has two parts: the recurrent matrix and the tanh slopes $1 - h_j^2 \\in (0, 1]$. If the product's factors shrink vectors — e.g. when the spectral radius $\\rho(W_{hh})$ is well below 1 — the influence of an old input decays **geometrically**: the network has *fading memory*. In the worked example $\\rho(W_{hh}) = |0.5 \\pm 0.1i| = 0.510$, so the effect of $x_1$ roughly halves (or worse) at every step. The scratch code measures this directly for a 16-unit RNN by nudging $x_1$ by 0.1: with $\\rho = 0.5$ the change in $h_t$ falls from $0.31$ at $t=1$ to $2.6\\times10^{-3}$ at $t=5$ and $8.9\\times10^{-17}$ at $t=40$; with $\\rho = 1.5$ it is still $4.5\\times10^{-5}$ at $t = 40$. This same product of Jacobians is what multiplies the *gradient* during training, which is why [BPTT](/learn/m09/bptt) suffers from vanishing and exploding gradients." },
      { t: "callout", kind: "note", title: "Why tanh and not ReLU?", text: "tanh keeps the state bounded, which keeps repeated application stable in the forward direction, and it is zero-centred so the state can push units both ways. With ReLU and a recurrent matrix of spectral radius above 1, states can grow without bound over long sequences. ReLU RNNs can work with careful initialisation (e.g. the identity-initialised IRNN of Le, Jaitly and Hinton, 2015) — see [BPTT](/learn/m09/bptt)." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      { t: "p", text: "The simulator unrolls a small RNN (input size 1, or one-hot characters; hidden size 3 or 4) over the sequence you type. Each time step is computed in five sub-steps — input term, recurrent term, sum, tanh, output — with the real numbers in the side panel. Below the diagram, a **memory probe** perturbs only $x_1$ by $\\varepsilon$, re-runs the network, and plots $\\|h_t(\\text{perturbed}) - h_t\\|$ for every later step on a log scale." },
      { t: "sim", id: "rnn", caption: "The recurrent matrix W_hh is rescaled so that its spectral radius equals the slider value; everything else stays fixed." },
      { t: "list", items: [
        "Step through $t = 1$: the recurrent term is exactly zero because $h_0 = 0$. From $t = 2$ on, compare the sizes of $W_{xh}x_t$ and $W_{hh}h_{t-1}$ — which one dominates $z_t$?",
        "With the default pulse sequence `1, 0, 0, 0, …` every later input is 0, so everything after step 1 is pure memory. Set $\\rho = 0.3$: the hidden state collapses towards the fixed point set by the bias within two or three steps and the probe bars drop by orders of magnitude. Set $\\rho = 1.5$: the state keeps moving and the probe decays much more slowly.",
        "Push $\\rho$ to 2.5 and watch the tanh sub-step: many units sit near $\\pm 1$ (saturated). Large recurrent weights do not simply mean \"more memory\" — saturated units stop responding to new inputs.",
        "Make $\\varepsilon$ large (1.0) and small (0.01): for small $\\varepsilon$ the curve shape stops changing — you are measuring the Jacobian product from the Math section.",
        "Switch to **Characters** and type `abcab`. Each letter is a one-hot vector, so $W_{xh}x_t$ is just one column of $W_{xh}$: the two `a`s produce identical input terms, yet $h_1$ and $h_4$ differ because the memory differs. That is the whole point of the recurrence.",
      ] },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "A complete forward-only implementation in NumPy. It reproduces the worked example, counts parameters, measures fading memory with the perturbation probe, and checks that stateful chunked processing equals a single pass. (Training — computing gradients through time — is implemented from scratch in the [next lesson](/learn/m09/bptt).)" },
      { t: "code", lang: "python", title: "rnn_forward_numpy.py", code: RNN_NUMPY, output: RNN_NUMPY_OUT },
      { t: "list", items: [
        "`forward` is a plain loop: the state **must** be computed in order, which is why RNNs cannot be parallelised across time the way a Transformer can.",
        "`VanillaRNN.random` sets the spectral radius explicitly by dividing by the largest eigenvalue magnitude. Drawing $W_{hh}$ from $\\mathcal N(0, 1)$ and *not* rescaling would give $\\rho \\approx \\sqrt{n_h} = 4$ for 16 units — far too large.",
        "Memory probe: the three rows differ only in $\\rho(W_{hh})$. The decay is roughly geometric (straight on a log scale), and faster for smaller $\\rho$. Note that even $\\rho = 1.5$ eventually decays, because saturated tanh units have slopes $1 - h^2$ near 0.",
        "Stateful check: carrying the last state of each 10-step chunk into the next gives **exactly** the same states as one 40-step pass (difference 0.0); resetting to zeros at every chunk boundary changes the states by up to 0.69.",
      ] },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "**1. `nn.RNN` shapes, verified against NumPy.** `nn.RNN` implements exactly our equation for all time steps in one call (with a fused C++/cuDNN loop). It returns two things that are easy to confuse: `output`, the **top layer's** state at **every** step, and `h_n`, **every layer's** state at the **last** step." },
          { t: "code", lang: "python", title: "rnn_shapes_pytorch.py", code: RNN_TORCH_CHECK, output: RNN_TORCH_CHECK_OUT },
          { t: "list", items: [
            "`batch_first=True` changes the layout of `x` and `output` to (batch, time, features). It does **not** change `h0` / `h_n`, which are always (num_layers × num_directions, batch, hidden).",
            "Parameter count for the 2-layer model: layer 0 has $6\\cdot4 + 6\\cdot6 + 6 + 6 = 72$, layer 1 has $6\\cdot6 + 6\\cdot6 + 6 + 6 = 84$, total 156. The two bias vectors per layer are redundant (only their sum matters) but are kept for cuDNN compatibility — when porting weights from another library, put the bias in one of them and zero the other, as done above.",
            "The NumPy re-implementation agrees to about $10^{-7}$: PyTorch computes in float32 while the check runs in float64, so this is rounding, not a bug. The worked example is reproduced to all six printed decimals.",
          ] },
          { t: "p", text: "**2. A real experiment: next-value forecasting of noisy sine waves.** Each sequence is a sine with its own random frequency and phase plus Gaussian noise ($\\sigma = 0.1$, so the best possible mean squared error is $\\sigma^2 = 0.01$). We compare fixed-window models — linear autoregression and a one-hidden-layer MLP on the last $k$ values — with a 32-unit vanilla RNN that reads the whole history. The RNN is trained with teacher forcing; afterwards it is run **free** (its own predictions fed back as inputs) for 20 steps." },
          { t: "code", lang: "python", title: "rnn_forecast_noisy_sine.py", code: RNN_SINE, output: RNN_SINE_OUT },
          { t: "list", items: [
            "Linear windows cannot adapt to each sequence's frequency, so even $k = 20$ stays at 0.034, more than three times the noise floor.",
            "A nonlinear window model needs enough context: the MLP with $k = 5$ is at 0.030, with $k = 20$ at 0.0140. The RNN reaches 0.0143 **without choosing $k$** — its state accumulates the evidence about frequency and phase on its own. Here the needed context is short, so a large-enough window matches the RNN; the advantage of recurrence appears when the relevant context is long or of unknown length.",
            "Free-running generation: the mean absolute error against the clean signal grows from 0.050 one step ahead to 0.208 twenty steps ahead. Each prediction becomes the next input, so small errors compound — the exposure-bias effect described in the Mechanism section.",
          ] },
          { t: "p", text: "**3. A character-level language model with sampling.** A 128-unit RNN on one-hot characters learns to predict the next character of a short public-domain passage (the opening of *Alice's Adventures in Wonderland*, 1865). Training uses teacher forcing on random 40-character windows; generation feeds each sampled character back in." },
          { t: "code", lang: "python", title: "char_rnn_pytorch.py", code: RNN_CHAR, output: RNN_CHAR_OUT },
          { t: "list", items: [
            "The cross-entropy falls from 3.275 nats per character — close to $\\ln 27 = 3.30$, a uniform guess over the 27 symbols — to about 0.1: on a 302-character text the network essentially **memorises** it. With greedy decoding and at temperature 1, the continuation of \"Alice was \" is the original sentence.",
            "**Temperature** $\\tau$ divides the logits before the softmax. $\\tau = 2$ flattens the distribution, and the sample degrades into plausible-looking letter salad after a few characters: once one unlikely character is sampled, the model is in a state it never saw during training.",
            "The prompt \"the sister had \" never occurs in the text. The model latches onto the most recent context — \"had \" — and continues with \"no pictures or conversations…\", which follows \"had \" in the passage. The hidden state is dominated by the last few characters; \"the sister\" has been largely forgotten. A model trained on a large corpus would generalise instead of copying, but the same memory limitation applies.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "p", text: "Keras's `SimpleRNN` layer is the same vanilla RNN. Its inputs are always (batch, time, features). `return_sequences=True` returns every $h_t$ (needed when another recurrent layer follows, or for aligned many-to-many tasks); `return_state=True` additionally returns the final state. Keras stores the transposed matrices (`kernel` = $W_{xh}^\\top$, `recurrent_kernel` = $W_{hh}^\\top$) and a single bias, and by default initialises the recurrent kernel to a random **orthogonal** matrix." },
          { t: "code", lang: "python", title: "simple_rnn_keras.py (not executed in CI)", code: RNN_KERAS, output: "Prints the shapes (3, 5, 6) and (3, 6) for the sequence output and final state; the weight shapes (4, 6), (6, 6) and (6,); a max |numpy − keras| difference at float32 rounding level (around 1e-7); and the model summary of the stacked many-to-one network." },
          { t: "p", text: "For stateful processing of long streams, Keras offers `SimpleRNN(..., stateful=True)`, which keeps the final state of each batch as the initial state of the next; it requires a fixed batch size and an explicit state reset between independent streams." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "table", head: ["Variant", "What changes", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Elman (vanilla) RNN", "Feeds back the hidden state: $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$", "Simple, few parameters, fast per step", "Short memory; vanishing/exploding gradients", "Teaching, short sequences, baselines"],
        ["Jordan network", "Feeds back the previous *output* $\\hat y_{t-1}$ instead of $h_{t-1}$", "Memory is interpretable (it is the output)", "Memory limited to what the output encodes", "Historical; control and sequence production"],
        ["Stacked (deep) RNN", "Layer $\\ell$ reads the state sequence of layer $\\ell-1$", "More processing per step, hierarchical features", "More compute; harder to train without gating", "Speech, translation encoders — see [stacked RNNs](/learn/m09/gru-variants)"],
        ["Bidirectional RNN", "A second RNN reads the sequence backwards; states are concatenated", "Each output sees past **and** future context", "Needs the whole sequence: not for streaming or generation", "Tagging, speech recognition (offline)"],
        ["LSTM / GRU", "Gates control writing, keeping and exposing memory; additive state update", "Remembers over hundreds of steps; trains reliably", "About 4× (LSTM) or 3× (GRU) the parameters per unit", "Default recurrent choice — see [LSTM](/learn/m09/lstm)"],
        ["Echo state network (reservoir computing)", "$W_{xh}, W_{hh}$ random and fixed (spectral radius < 1); only the linear read-out is trained", "Training is a least-squares fit; very fast", "Fixed features; large reservoirs needed", "Chaotic time-series modelling, physical/analog computing"],
        ["Encoder–decoder (seq2seq)", "One RNN encodes, another decodes from its final state", "Input and output lengths can differ", "Single-vector bottleneck for long inputs", "Translation, summarisation — led to [attention](/learn/m10/attention-intro)"],
        ["Linear recurrences / state-space models", "Recurrence without a nonlinearity inside the state update (e.g. S4, Mamba), with structured or input-dependent transition matrices", "Parallel training via scans or convolutions; very long contexts", "More complex parameterisation", "Long-sequence modelling, recent language models"],
      ] },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Language modelling and text generation** — recurrent language models (Mikolov and colleagues, 2010) improved speech-recognition and translation systems before Transformers; character-level RNNs became a popular demonstration of generative modelling (Karpathy's 2015 essay *The Unreasonable Effectiveness of Recurrent Neural Networks*).",
        "**Speech recognition** — recurrent networks (mostly LSTMs) trained with CTC or encoder–decoder objectives were the backbone of large-vocabulary recognisers in the mid-2010s.",
        "**Handwriting recognition and generation** — RNNs read pen trajectories or image columns; Graves (2013) generated realistic handwriting with recurrent mixture-density networks.",
        "**Time-series forecasting** — demand, energy load, traffic and sensor forecasting, where a stream arrives continuously and a constant-memory state is natural.",
        "**Streaming and on-device processing** — keyword spotting, activity recognition from accelerometers, and anomaly detection on machine sensors: an RNN updates its state with constant memory per new sample, which suits microcontrollers and real-time pipelines.",
        "**Sequence-to-sequence machine translation** — the encoder–decoder RNN (2014) and its attention-augmented version were among the first purely neural systems to rival phrase-based statistical translation, before Transformers replaced them.",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "table", head: ["Strength", "Limitation"], rows: [
        ["Handles sequences of any length with a fixed number of parameters", "Strictly sequential computation: step $t$ waits for step $t-1$, so training cannot be parallelised over time"],
        ["Constant memory per step at inference — ideal for streaming", "Training memory grows as $O(T)$ because every state is stored for the backward pass"],
        ["Shares statistical strength across positions (weight sharing)", "The whole past must fit into one $n_h$-dimensional vector: a lossy bottleneck"],
        ["Small and fast for short contexts", "Vanilla RNNs forget within roughly 10–20 steps in practice; gradients vanish or explode over longer spans (see [BPTT](/learn/m09/bptt))"],
        ["Natural model for generation: one step in, one step out", "Exposure bias: trained with teacher forcing, run on its own outputs"],
      ] },
      { t: "p", text: "In practice: use a gated cell (LSTM/GRU) instead of a vanilla RNN when dependencies are longer than a few steps; use a Transformer when you have long sequences, lots of data and parallel hardware; keep a small recurrent model when you need streaming with constant memory per step." },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Confusing `output` and `h_n`.** `output[:, t]` is the top layer at step $t$; `h_n[l]` is layer $l$ at the last step. For a multi-layer model, `h_n[0]` is the *bottom* layer — classifying from it silently ignores the upper layers. Use `output[:, -1]` or `h_n[-1]`.",
        "**Assuming `batch_first=True` applies to the hidden state.** `h0`/`h_n` stay (layers, batch, hidden). Passing a (batch, layers, hidden) tensor raises a shape error — or, worse, silently mixes sequences when the two sizes happen to be equal.",
        "**Reading `output[:, -1]` from padded batches.** If shorter sequences are padded at the end, the last position is padding and the state there has run over meaningless zeros. Index each sequence's true last step, or use `nn.utils.rnn.pack_padded_sequence`.",
        "**Carrying state between unrelated sequences**, or carrying it without `detach()`. The first leaks information across examples; the second makes the graph grow with every batch until memory runs out or backward fails. Reset to zeros for independent sequences; detach for stateful streams.",
        "**Leaking the future in time-series evaluation.** Random train/test splits of overlapping windows put almost-identical windows in both sets. Split by time (train on the past, test on the future).",
        "**Unscaled inputs.** Raw values in the hundreds saturate every tanh unit at the first step (slope ≈ 0), so the state carries almost no information about magnitude. Standardise inputs.",
        "**Initialising $W_{hh}$ with a large variance** (e.g. $\\mathcal N(0, 1)$ for 128 units gives spectral radius ≈ 11). Scale it to spectral radius around 1 or use orthogonal initialisation.",
        "**Judging a generator only by teacher-forced loss.** A low next-step loss does not guarantee good free-running samples; always evaluate by generating.",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the hidden state of an RNN?", a: "A fixed-size vector $h_t$ computed at every time step from the current input and the previous hidden state, $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$. It is the network's memory: a learned, lossy summary of everything it has read so far, and the only channel through which the past influences the present." },
      { t: "qa", level: "beginner", q: "What does weight sharing across time mean, and why is it useful?", a: "The same matrices $W_{xh}, W_{hh}$, bias and output head are used at every time step. Consequences: the parameter count does not depend on sequence length, the model can process sequences of any length, and a pattern learned at one position applies at every position — analogous to a convolution kernel shared across image positions." },
      { t: "qa", level: "beginner", q: "Give one example each of many-to-one, one-to-many and many-to-many tasks.", a: "Many-to-one: sentiment classification of a review (read all words, output one label). One-to-many: image captioning (one image, a sequence of words). Many-to-many aligned: part-of-speech tagging (one tag per word); unaligned: machine translation, usually with an encoder–decoder." },
      { t: "qa", level: "intermediate", q: "How many parameters does a vanilla RNN layer with input size 50 and hidden size 100 have (one bias)? What does PyTorch report?", a: "$100\\cdot50 + 100\\cdot100 + 100 = 15{,}100$. PyTorch's `nn.RNN` has two bias vectors, so it reports $15{,}200$. Neither depends on the sequence length." },
      { t: "qa", level: "intermediate", q: "What is teacher forcing and what problem does it introduce?", a: "During training, the true previous element is fed as the next input instead of the model's own prediction. It makes training stable and parallel across steps. The problem is **exposure bias**: at generation time the model receives its own (imperfect) outputs, a distribution it never saw in training, so errors compound. Mitigations include scheduled sampling, sequence-level training objectives, and evaluating by free-running generation." },
      { t: "qa", level: "intermediate", q: "When would you carry the hidden state across batches, and what must you do when you do it?", a: "When batches are consecutive chunks of one long stream (stateful training), so that each chunk starts with the context of the previous one. You must arrange the batch so that row $j$ continues row $j$, reset the state at stream boundaries, and detach the carried state from the graph so that backpropagation stops at the chunk boundary (truncated BPTT)." },
      { t: "qa", level: "advanced", q: "Explain fading memory in a vanilla RNN in terms of Jacobians.", a: "A perturbation of $h_t$ propagates as $\\Delta h_{t+k} \\approx \\prod_j \\operatorname{diag}(1 - h_j^2)W_{hh}\\,\\Delta h_t$. If the factors are contractions — spectral radius of $W_{hh}$ below 1 and tanh slopes at most 1 — the product shrinks geometrically, so old inputs lose influence exponentially fast. A spectral radius well above 1 can preserve or amplify perturbations, but it also saturates tanh units, whose slopes near 0 again shrink the product, and may make the dynamics chaotic. The same product governs gradient flow in BPTT." },
      { t: "qa", level: "advanced", q: "An MLP on a window of 20 values matched the RNN on the noisy-sine task. When does an RNN have a real advantage over a window model?", a: "When the relevant context is longer than any practical window or of unknown or variable length (e.g. a character that defines the meaning of something 200 steps later), when the input is an unbounded stream that must be processed with constant memory per step, and when parameter efficiency matters — the window MLP's first layer grows linearly with $k$ while the RNN's parameters do not. When the useful context is short and fixed, a window model (or a 1-D CNN) is a strong, parallelisable baseline." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      { t: "quiz", questions: [
        { kind: "numeric", q: "A vanilla RNN has $n_{in} = 10$, $n_h = 20$ and a linear output head with $n_{out} = 5$ (one bias per layer). How many parameters does it have?", answer: 725, tolerance: 0.5, explain: "$20\\cdot10 + 20\\cdot20 + 20 + 5\\cdot20 + 5 = 200 + 400 + 20 + 100 + 5 = 725$, for any sequence length." },
        { kind: "numeric", q: "A scalar RNN $h_t = \\tanh(2x_t + 0.5h_{t-1})$ starts at $h_0 = 0$ and receives $x_1 = 0.5$. What is $h_1$ (3 decimals)?", answer: 0.762, tolerance: 0.001, explain: "$h_1 = \\tanh(2 \\cdot 0.5 + 0) = \\tanh(1) = 0.7616$." },
        { kind: "numeric", q: "In a *linear* scalar RNN $h_t = 0.5\\,h_{t-1} + x_t$, by what factor has the effect of $x_1$ on $h_t$ shrunk after 10 further steps? (4 decimals)", answer: 0.001, tolerance: 0.0001, explain: "Each step multiplies the old contribution by 0.5: $0.5^{10} = 0.000977 \\approx 0.0010$. This is fading memory with spectral radius 0.5." },
        { q: "`rnn = nn.RNN(8, 16, num_layers=3, batch_first=True)` receives `x` of shape (4, 12, 8). What are the shapes of `output` and `h_n`?", options: ["(4, 12, 16) and (3, 4, 16)", "(12, 4, 16) and (3, 4, 16)", "(4, 12, 16) and (4, 3, 16)", "(4, 16) and (3, 16)"], answer: 0, explain: "`output` follows batch_first: (batch, time, hidden) = (4, 12, 16). `h_n` is always (num_layers, batch, hidden) = (3, 4, 16)." },
        { q: "Which statement about weight sharing in RNNs is correct?", options: ["Each time step has its own $W_{hh}$, learned separately", "The same weights are used at every step, so the parameter count is independent of $T$", "Weights are shared between the input and output layers", "Weight sharing means $W_{xh} = W_{hh}$"], answer: 1, explain: "One cell, applied at every time step. Only compute and training memory grow with $T$." },
        { q: "You classify movie reviews (one label per review) with an RNN. Which pattern and which state do you use?", options: ["One-to-many; $h_0$", "Many-to-one; the state after the last real (non-padding) token", "Many-to-many aligned; the average of all outputs is forbidden", "One-to-one; $h_1$"], answer: 1, explain: "Read the whole review, classify from the final state (or a pooling of states). With padding, take the state at each review's true last token." },
        { q: "Why does the free-running forecast error grow with the horizon even for a well-trained model?", options: ["The weights change during generation", "Each prediction is fed back as input, so errors compound and the model sees inputs unlike its training data", "tanh saturates after 20 steps", "The learning rate is too high"], answer: 1, explain: "Teacher forcing trains on true inputs; generation uses the model's own outputs. Small errors propagate and accumulate (exposure bias)." },
      ] },
      { t: "exercise", title: "Two steps by hand", prompt: "A scalar RNN has $h_t = \\tanh(w_x x_t + w_h h_{t-1})$ with $w_x = 1$, $w_h = 0.5$, no bias, $h_0 = 0$, and output $\\hat y_t = 2h_t$. For the inputs $x = (1, -1)$ compute $h_1$, $z_2$, $h_2$ and $\\hat y_2$. How much of $z_2$ is memory?", hint: "$z_2 = w_x x_2 + w_h h_1$; the second term is the memory.", solution: [
        { t: "steps", items: [
          { title: "Step 1", tex: String.raw`h_1 = \tanh(1\cdot 1 + 0.5\cdot 0) = \tanh(1) = 0.761594` },
          { title: "Step 2 pre-activation", tex: String.raw`z_2 = 1\cdot(-1) + 0.5\cdot 0.761594 = -1 + 0.380797 = -0.619203` },
          { title: "Step 2 state and output", tex: String.raw`h_2 = \tanh(-0.619203) = -0.550573,\qquad \hat y_2 = 2h_2 = -1.101146` },
        ] },
        { t: "p", text: "The memory term contributes $+0.380797$ to $z_2$: without it, $h_2$ would be $\\tanh(-1) = -0.761594$. The first input still pulls the state upwards one step later." },
      ] },
      { t: "exercise", title: "Batched forward pass and a many-to-one head", prompt: "Extend `rnn_forward_numpy.py` so that `forward` accepts a batch `xs` of shape (B, T, n_in) and returns states of shape (B, T, n_h) — without a Python loop over the batch. Then add a function `classify(xs, lengths)` that returns, for each sequence, $W_{hy}$ applied to the state at its true last step `lengths[i] - 1` (sequences are right-padded with zeros). Check that a padded sequence gives the same result as running it alone, unpadded.", hint: "Replace `W_xh @ x` by `x @ W_xh.T` for a (B, n_in) slice; gather the last states with fancy indexing `hs[np.arange(B), lengths - 1]`.", solution: [
        { t: "code", lang: "python", code: String.raw`def forward_batch(self, xs, h0=None):
    B, T, _ = xs.shape
    h = np.zeros((B, self.W_hh.shape[0])) if h0 is None else h0
    hs = np.empty((B, T, self.W_hh.shape[0]))
    for t in range(T):                                   # loop over time only
        h = np.tanh(xs[:, t] @ self.W_xh.T + h @ self.W_hh.T + self.b)
        hs[:, t] = h
    return hs

def classify(self, xs, lengths):
    hs = self.forward_batch(xs)
    last = hs[np.arange(len(lengths)), np.asarray(lengths) - 1]   # (B, n_h)
    return last @ self.W_hy.T + self.b_y

# check: a sequence of length 3 padded to length 5 vs the same sequence alone
r = VanillaRNN.random(1, 8, 2, seed=0)
seq = np.array([[0.3], [-1.0], [0.7]])
padded = np.zeros((2, 5, 1)); padded[0, :3] = seq; padded[1] = 1.0
print(np.allclose(r.classify(padded, [3, 5])[0], r.classify(seq[None], [3])[0]))   # True` },
        { t: "p", text: "Padding after the last real step does not affect the state *at* that step, because the recurrence only looks backwards — which is why gathering at `lengths - 1` is correct, while taking `hs[:, -1]` would include the padded steps." },
      ] },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Sequences: ordered, variable-length data where context matters — text, audio, time series, sensor streams.",
        "Vanilla RNN: $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$, $\\hat y_t = W_{hy}h_t + b_y$, from $h_0 = 0$.",
        "Weight sharing across time: one cell for all steps; parameters $n_hn_{in} + n_h^2 + n_h + n_{out}n_h + n_{out}$, independent of $T$.",
        "Unrolled over $T$ steps, an RNN is a $T$-layer feed-forward network with tied weights and side inputs — trained by backpropagating through it.",
        "Patterns: one-to-many, many-to-one, many-to-many aligned, and encoder–decoder for unaligned sequences (which led to attention).",
        "The hidden state is a fixed-size, lossy summary; perturbations decay through $\\prod \\operatorname{diag}(1-h^2)W_{hh}$ — fading memory controlled by $\\rho(W_{hh})$ and tanh saturation.",
        "Stateful training carries $h$ across chunks of one stream and must `detach()` it; stateless training resets $h_0 = 0$.",
        "Teacher forcing trains next-step prediction on true inputs; generation feeds back samples, so errors compound (exposure bias); temperature controls sampling randomness.",
        "PyTorch `nn.RNN`: `output` = top layer, every step; `h_n` = every layer, last step; `h0` ignores `batch_first`; two bias vectors per layer.",
      ] },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[Backpropagation Through Time](/learn/m09/bptt) — how the unrolled graph is differentiated, a three-step backward pass on this lesson's worked example, and the precise reason vanilla RNNs cannot learn long dependencies.",
        "[Long Short-Term Memory (LSTM)](/learn/m09/lstm) — the gated cell whose additive memory keeps gradients alive over hundreds of steps.",
        "[GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants) — a lighter gated cell, reading in both directions, and stacking layers.",
        "[Attention: the core idea](/learn/m10/attention-intro) — removing the fixed-size bottleneck of the encoder–decoder by letting every output look at every input state.",
      ] },
    ],
  },
};

export default lesson;
