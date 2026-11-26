import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m09/lstm",
  title: "Long Short-Term Memory (LSTM)",
  summary:
    "A recurrent cell with a separate memory that is updated by addition, plus three learned gates. Every equation and shape, a worked numerical step, why the design keeps gradients alive, and a from-scratch NumPy version that matches torch.nn.LSTM to machine precision.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m09/rnn-basics", "m09/bptt", "m03/activation-functions", "m04/backpropagation"],
  next: ["m09/gru-variants", "m10/attention-intro"],
  terms: ["lstm", "vanishing-gradient", "exploding-gradient"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "A **Long Short-Term Memory** network (LSTM) is a recurrent network whose cell keeps two kinds of state: a **cell state** $c_t$, its long-term memory, and a **hidden state** $h_t$, its output at each step. Three **gates** — small sigmoid layers that output numbers between 0 and 1 — decide, unit by unit, what to erase from memory, what to write into it, and what to show from it." },
      { t: "p", text: "**Hochreiter and Schmidhuber (1997)** introduced LSTMs to fix a specific failure of vanilla RNNs: they cannot learn links that span more than a handful of steps, because the training signal shrinks (or explodes) exponentially as it flows back through time. The forget gate, now seen as essential, was added later by **Gers, Schmidhuber and Cummins (2000)**." },
      { t: "p", text: "Before Transformers, LSTMs were the default sequence model (translation, speech, handwriting, language modelling, forecasting). They are still a strong, cheap choice for streaming, small datasets and on-device models, because their memory per step is constant (no growing attention cache)." },
      { t: "callout", kind: "key", title: "The one idea to remember", text: "A vanilla RNN rewrites its whole state each step with a matrix multiply and a squashing function. An LSTM updates its memory by **element-wise scaling plus addition**: $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$. When the forget gate is near 1, information and gradients pass through almost unchanged." },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "list", items: [
        "[Sequential data and vanilla RNNs](/learn/m09/rnn-basics) — the update $h_t = \\tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$, unrolling through time, and what a hidden state is.",
        "[Backpropagation through time](/learn/m09/bptt) — why the gradient for an early hidden state is a *product* of per-step Jacobians, and why that product vanishes or explodes.",
        "[Activation functions](/learn/m03/activation-functions) — the sigmoid $\\sigma(z) = 1/(1+e^{-z})$ with range $(0,1)$, $\\tanh$ with range $(-1,1)$, and their derivatives $\\sigma' = \\sigma(1-\\sigma) \\le 0.25$ and $\\tanh' = 1 - \\tanh^2 \\le 1$.",
        "[Backpropagation](/learn/m04/backpropagation) and the [chain rule](/learn/m02/derivatives-chain-rule) — we differentiate the cell update by hand.",
        "[Vectors and matrices](/learn/m02/vectors-matrices) — matrix–vector products, stacking vectors, and the element-wise (Hadamard) product $\\odot$.",
      ] },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "A conveyor belt with three workers", text: "Picture a conveyor belt (the cell state) with one station per time step. At each station three workers look at the current input and the previous station's report (the hidden state). The **forget** worker takes some items off the belt. The **input** worker decides how much of a freshly made batch (the candidate) to put on. The **output** worker decides what to photograph from the belt and send out as this station's report. Nobody rebuilds the belt at each station: if nobody touches an item, it rides along unchanged for as long as needed." },
      { t: "h", text: "Why vanilla RNNs forget" },
      { t: "p", text: "In a vanilla RNN the only memory is $h_t = \\tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$. Every step, the old memory is multiplied by $W_{hh}$ and squashed by $\\tanh$. As [backpropagation through time](/learn/m09/bptt) showed, the gradient reaching a state $k$ steps back is a product of $k$ Jacobians $\\operatorname{diag}(1-h_t^2)\\,W_{hh}$. If they are smaller than 1 in size, the product shrinks geometrically (vanishing); if larger, it grows (exploding). In practice a vanilla RNN struggles to use information more than about 10–20 steps old." },
      { t: "h", text: "What the LSTM changes" },
      { t: "list", items: [
        "It adds a second state, the **cell state** $c_t$, which is never multiplied by a weight matrix or squashed between steps.",
        "The cell state changes only by an element-wise multiply with the forget gate $f_t \\in (0,1)$ and an element-wise **addition** of new content $i_t \\odot g_t$.",
        "If the network learns $f_t \\approx 1$ and $i_t \\approx 0$ for a unit, that unit holds its value: a learned memory cell you can switch on and off.",
        "The hidden state $h_t = o_t \\odot \\tanh(c_t)$ is a filtered view of memory. It is the output, and it feeds the next step's gates.",
      ] },
      { t: "p", text: "Gates work per unit, not globally. With hidden size 256 there are 256 separate forget values each step, so some units can remember a sentence's subject for 50 tokens while others track the last character." },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "p", text: "Notation for one step: $x_t \\in \\mathbb{R}^{d}$ is the input; $h_{t-1}, c_{t-1} \\in \\mathbb{R}^{h}$ are the previous hidden and cell states (hidden size $h$). All four inner layers read the same stacked vector $z_t = [h_{t-1};\\,x_t] \\in \\mathbb{R}^{h+d}$. Each has its own weight matrix $W_\\ast \\in \\mathbb{R}^{h \\times (h+d)}$ and bias $b_\\ast \\in \\mathbb{R}^{h}$. For a batch of $B$ examples, each vector becomes a $B \\times h$ (or $B \\times d$) matrix and the formulas apply row by row." },
      {
        t: "steps",
        items: [
          {
            title: "1. Forget gate — what to erase",
            text: "Shape $h$. Each entry in $(0,1)$ is the fraction of that memory unit that survives. $f=0$: wipe the unit. $f=1$: keep it as is.",
            tex: String.raw`f_t = \sigma\!\left(W_f\,[h_{t-1};x_t] + b_f\right)`,
          },
          {
            title: "2. Input gate — how much to write",
            text: "Shape $h$, values in $(0,1)$. Decides per unit how strongly the new proposal gets written.",
            tex: String.raw`i_t = \sigma\!\left(W_i\,[h_{t-1};x_t] + b_i\right)`,
          },
          {
            title: "3. Candidate cell state — what to write",
            text: "Shape $h$, values in $(-1,1)$. A proposal for new content, computed like a vanilla RNN state. $\\tanh$ lets it be positive or negative, so writing can raise or lower a unit. (PyTorch calls this the `g` gate; some texts write $\\tilde c_t$.)",
            tex: String.raw`g_t = \tanh\!\left(W_g\,[h_{t-1};x_t] + b_g\right)`,
          },
          {
            title: "4. Cell-state update — the conveyor belt",
            text: "Shape $h$. Keep a gated fraction of the old memory and add a gated amount of the candidate. Only element-wise operations: no matrix, no squashing — the line along the top of the classic LSTM diagram.",
            tex: String.raw`c_t = f_t \odot c_{t-1} + i_t \odot g_t`,
          },
          {
            title: "5. Output gate — what to show",
            text: "Shape $h$, values in $(0,1)$. Picks which memory units are exposed at this step.",
            tex: String.raw`o_t = \sigma\!\left(W_o\,[h_{t-1};x_t] + b_o\right)`,
          },
          {
            title: "6. Hidden state — the output",
            text: "Shape $h$, values in $(-1,1)$. $\\tanh$ squashes the unbounded cell state into a fixed range, then the output gate filters it. $h_t$ is the layer's output at time $t$ and feeds all four layers at $t+1$; $c_t$ is passed on but never output directly.",
            tex: String.raw`h_t = o_t \odot \tanh(c_t)`,
          },
        ],
      },
      { t: "h", text: "Implementation detail: one big matrix multiply" },
      { t: "p", text: "All four layers read the same $z_t$, so real code stacks the four weight matrices into one $4h \\times (h+d)$ matrix. One matrix multiply gives all pre-activations, which are then sliced into four blocks of size $h$. Frameworks also split the stacked matrix into an input part $W_{ih} \\in \\mathbb{R}^{4h \\times d}$ and a recurrent part $W_{hh} \\in \\mathbb{R}^{4h \\times h}$. That is the same maths: $W[h;x] = W_{hh}h + W_{ih}x$." },
      { t: "table", head: ["Quantity", "Shape (one example)", "Shape (batch B)", "Range"], rows: [
        ["$x_t$", "$d$", "$B \\times d$", "any"],
        ["$h_{t-1}$, $h_t$", "$h$", "$B \\times h$", "$(-1, 1)$"],
        ["$c_{t-1}$, $c_t$", "$h$", "$B \\times h$", "unbounded"],
        ["$[h_{t-1}; x_t]$", "$h + d$", "$B \\times (h+d)$", "—"],
        ["$W_f, W_i, W_g, W_o$", "$h \\times (h+d)$ each", "same (shared)", "—"],
        ["stacked pre-activations", "$4h$", "$B \\times 4h$", "any"],
        ["$f_t, i_t, o_t$", "$h$", "$B \\times h$", "$(0, 1)$"],
        ["$g_t$", "$h$", "$B \\times h$", "$(-1, 1)$"],
      ] },
      { t: "callout", kind: "warning", title: "Gate order differs between libraries", text: "PyTorch stacks the four blocks as **i, f, g, o** (input, forget, candidate, output). Keras also uses i, f, c, o. Some papers and older code use other orders. Always check the order when copying weights: silently swapping the forget and input blocks gives a network that runs but computes the wrong thing." },
      { t: "h", text: "Processing a whole sequence" },
      { t: "list", ordered: true, items: [
        "Start with $h_0 = 0$ and $c_0 = 0$ (or learned / carried-over states).",
        "For $t = 1, \\dots, T$: compute steps 1–6 with the **same** weights at every step.",
        "Keep all of $h_1, \\dots, h_T$ (for per-step outputs, e.g. tagging or next-value prediction) or only $h_T$ (for classifying the whole sequence).",
        "Train with backpropagation through time; each weight's gradient is the sum of its gradients from every step.",
      ] },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Variables" },
      { t: "list", items: [
        "$t$ — time step; $T$ — sequence length.",
        "$d$ — input size; $h$ — hidden size (number of memory units).",
        "$x_t \\in \\mathbb{R}^d$ — input; $h_t \\in \\mathbb{R}^h$ — hidden state; $c_t \\in \\mathbb{R}^h$ — cell state.",
        "$W_f, W_i, W_g, W_o \\in \\mathbb{R}^{h \\times (h+d)}$ — weights; $b_f, b_i, b_g, b_o \\in \\mathbb{R}^h$ — biases.",
        "$f_t, i_t, o_t \\in (0,1)^h$ — forget, input and output gates; $g_t \\in (-1,1)^h$ — candidate.",
        "$\\sigma$ — sigmoid; $\\odot$ — element-wise product; $[a;b]$ — stacking $a$ on top of $b$.",
      ] },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
f_t &= \sigma(W_f[h_{t-1};x_t] + b_f) \\
i_t &= \sigma(W_i[h_{t-1};x_t] + b_i) \\
g_t &= \tanh(W_g[h_{t-1};x_t] + b_g) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
o_t &= \sigma(W_o[h_{t-1};x_t] + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}`,
        caption: "The full LSTM cell (with forget gate, without peepholes).",
      },
      { t: "h", text: "Worked numerical step (h = 2, d = 1)" },
      { t: "p", text: "Take $x_t = 1$, $h_{t-1} = (0.5, -0.3)$, $c_{t-1} = (0.8, -0.6)$, so $z_t = [h_{t-1}; x_t] = (0.5, -0.3, 1)$. Each weight row below multiplies $(h_1, h_2, x)$. The forget bias is $b_f = (1, 1)$; all other biases are $0$." },
      {
        t: "math",
        tex: String.raw`W_f=\begin{pmatrix}0.4&0.2&-0.1\\0.1&-0.3&0.5\end{pmatrix},\;
W_i=\begin{pmatrix}0.3&-0.2&0.6\\-0.4&0.1&0.2\end{pmatrix},\;
W_g=\begin{pmatrix}0.5&0.3&0.8\\-0.6&0.2&-0.5\end{pmatrix},\;
W_o=\begin{pmatrix}0.2&0.4&0.3\\0.3&-0.1&0.7\end{pmatrix}`,
      },
      {
        t: "steps",
        items: [
          {
            title: "Forget gate",
            text: "Unit 1: $0.4(0.5) + 0.2(-0.3) + (-0.1)(1) + 1 = 1.04$. Unit 2: $0.1(0.5) - 0.3(-0.3) + 0.5(1) + 1 = 1.64$.",
            tex: String.raw`f_t = \sigma(1.04,\;1.64) = (0.7389,\;0.8375)`,
          },
          {
            title: "Input gate",
            text: "Unit 1: $0.15 + 0.06 + 0.6 = 0.81$. Unit 2: $-0.2 - 0.03 + 0.2 = -0.03$.",
            tex: String.raw`i_t = \sigma(0.81,\;-0.03) = (0.6921,\;0.4925)`,
          },
          {
            title: "Candidate",
            text: "Unit 1: $0.25 - 0.09 + 0.8 = 0.96$. Unit 2: $-0.3 - 0.06 - 0.5 = -0.86$.",
            tex: String.raw`g_t = \tanh(0.96,\;-0.86) = (0.7443,\;-0.6963)`,
          },
          {
            title: "Cell update",
            text: "Kept: $f_t \\odot c_{t-1} = (0.7389 \\times 0.8,\\; 0.8375 \\times -0.6) = (0.5911, -0.5025)$. Written: $i_t \\odot g_t = (0.6921 \\times 0.7443,\\; 0.4925 \\times -0.6963) = (0.5151, -0.3429)$.",
            tex: String.raw`c_t = (0.5911 + 0.5151,\;-0.5025 - 0.3429) = (1.1062,\;-0.8454)`,
          },
          {
            title: "Output gate",
            text: "Unit 1: $0.1 - 0.12 + 0.3 = 0.28$. Unit 2: $0.15 + 0.03 + 0.7 = 0.88$.",
            tex: String.raw`o_t = \sigma(0.28,\;0.88) = (0.5695,\;0.7068)`,
          },
          {
            title: "Hidden state",
            text: "$\\tanh(c_t) = (0.8027, -0.6887)$.",
            tex: String.raw`h_t = (0.5695 \times 0.8027,\;0.7068 \times -0.6887) = (0.4572,\;-0.4868)`,
          },
        ],
      },
      { t: "p", text: "Unit 1's memory grew past 1 ($c = 1.1062$): the cell state is not bounded, only $h_t$ is. All numbers were computed in float64 and match the simulator." },
      { t: "h", text: "Why additive updates keep gradients alive" },
      { t: "p", text: "Differentiate the cell update with respect to the previous cell state, holding the gates fixed (the **direct path** along the belt):" },
      {
        t: "math",
        tex: String.raw`\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = \operatorname{diag}(f_t)
\qquad\Rightarrow\qquad
\frac{\partial c_T}{\partial c_k}\bigg|_{\text{direct}} = \prod_{t=k+1}^{T}\operatorname{diag}(f_t) = \operatorname{diag}\!\Big(\prod_{t=k+1}^{T} f_t\Big)`,
      },
      { t: "p", text: "Compare the vanilla RNN, where $\\partial h_t / \\partial h_{t-1} = \\operatorname{diag}(1-h_t^2)\\,W_{hh}$. Three differences matter:" },
      { t: "list", items: [
        "**No weight matrix** on the path. A product of many $W_{hh}$ copies behaves like its largest singular value raised to a power; the LSTM path has no such power.",
        "**No squashing derivative** on the path. $\\tanh'$ and $\\sigma'$ (at most 0.25) do not appear in $\\partial c_t / \\partial c_{t-1}$.",
        "**The decay rate is learned and depends on the input.** Each unit's per-step factor is its own $f_t$, which the network can push towards 1 when it needs to remember. With $f_t = 1$ and $i_t = 0$ the gradient passes unchanged — Hochreiter and Schmidhuber called this the *constant error carousel*.",
      ] },
      { t: "p", text: "The full Jacobian has extra terms, because $c_{t-1}$ also affects $h_{t-1}$, which feeds the gates at step $t$. Those indirect terms go through weight matrices and can still vanish. But the direct path guarantees there is *a* route where the gradient decays only as fast as the forget gates say. For example, $f = 0.9$ for 20 steps keeps $0.9^{20} = 0.1216$ of the signal; $f = 0.99$ for 100 steps keeps $0.99^{100} = 0.3660$. A vanilla RNN with per-step factor 0.5 keeps $0.5^{10} \\approx 0.00098$ after only 10 steps." },
      {
        t: "code",
        lang: "python",
        title: "Measured: |∂h_T/∂x_t| for a vanilla RNN vs an LSTM (60 steps, random init)",
        code: String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
T, H = 60, 16
x = torch.randn(1, T, 1, requires_grad=True)       # one sequence of 60 scalar inputs

def grad_wrt_inputs(layer):
    out, _ = layer(x)
    x.grad = None
    out[0, -1].sum().backward()                     # scalar function of the LAST hidden state
    return x.grad[0, :, 0].abs()                    # |d h_T / d x_t| for every t

rnn = nn.RNN(1, H, batch_first=True)               # vanilla tanh RNN
lstm = nn.LSTM(1, H, batch_first=True)
with torch.no_grad():                               # forget-gate bias = +2 (rows H..2H in i,f,g,o order)
    lstm.bias_ih_l0[H:2*H].fill_(2.0); lstm.bias_hh_l0[H:2*H].fill_(0.0)

g_rnn, g_lstm = grad_wrt_inputs(rnn), grad_wrt_inputs(lstm)
print(" steps back | vanilla RNN |  LSTM (b_f=2)")
for k in [0, 5, 10, 20, 40, 59]:
    t = T - 1 - k
    print(f"   {k:3d}      |  {g_rnn[t]:.2e}   |  {g_lstm[t]:.2e}")`,
        output: ` steps back | vanilla RNN |  LSTM (b_f=2)
     0      |  2.56e-01   |  7.82e-02
     5      |  1.65e-02   |  1.73e-02
    10      |  2.28e-04   |  2.19e-02
    20      |  6.47e-07   |  1.55e-02
    40      |  2.27e-12   |  5.21e-03
    59      |  8.50e-19   |  2.80e-03`,
      },
      { t: "p", text: "Sixty steps back, the vanilla RNN's sensitivity has fallen by 17 orders of magnitude; the untrained LSTM's has fallen by about 30×. That is what \"LSTMs mitigate vanishing gradients\" means in practice. Note *mitigate*, not *eliminate*: the LSTM's gradient still decays, at a rate set by its forget gates." },
      { t: "h", text: "Parameter count" },
      { t: "p", text: "Each of the four layers has an $h \\times (h+d)$ weight matrix and an $h$-long bias:" },
      {
        t: "math",
        tex: String.raw`N_{\text{LSTM}} = 4\big(h(h+d) + h\big) \qquad N_{\text{RNN}} = h(h+d) + h \qquad N_{\text{GRU}} = 3\big(h(h+d) + h\big)`,
      },
      { t: "p", text: "Example: $d = 128$, $h = 256$ gives $4(256 \\cdot 384 + 256) = 394{,}240$ parameters. PyTorch and cuDNN keep two bias vectors ($b_{ih}$ and $b_{hh}$, which are just added), so `torch.nn.LSTM` reports $4h$ more: $4(h(h+d) + 2h)$. The cost per step is dominated by the $4h \\times (h+d)$ matrix multiply — about four times a vanilla RNN." },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "sim",
        id: "lstm",
        caption: "A 3- or 4-unit LSTM with seeded weights runs on your sequence. Each time step is split into five sub-steps; the narration shows each equation with the real numbers filled in.",
      },
      { t: "list", ordered: true, items: [
        "Press **Step** five times to watch one full time step: forget gate, input gate and candidate, the $+$ node on the cell line, output gate, then $h_t$. In the narration, $c_t$ is exactly kept + written.",
        "Play the default sequence `1, 0, 0, 0, 0, 1, 0, 0` with $b_f = 1$. In the timeline, the pulse at $t = 1$ writes a clear value into $c_t$, which then fades over the following zero inputs.",
        "Drag **Forget bias** down to $-2$. Now $f_t \\approx \\sigma(-2) \\approx 0.12$ when the other inputs are small, so memory is wiped within one or two steps — roughly how a vanilla RNN behaves.",
        "Drag it up to $+4$. Now $f_t \\approx 0.98$: the pulse stays in $c_t$ for the whole sequence, and the second pulse at $t = 6$ adds on top of what is stored. That is the additive update at work.",
        "Compare the **o_t** row with **c_t**: a unit can hold a large memory while its output gate keeps $h_t$ small. The memory stays private until the output gate opens.",
        "Type your own sequence (e.g. `1, -1, 1, -1` or `0, 0, 0, 2`); switching the hidden size between 3 and 4 gives different seeded weights.",
      ] },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      { t: "p", text: "The code below computes all four gates with one matrix multiply. It uses PyTorch's gate order (i, f, g, o) and two-bias convention so weights can be copied across, then checks every hidden state against `torch.nn.LSTM`. PyTorch is only the reference; the LSTM itself is pure NumPy." },
      {
        t: "code",
        lang: "python",
        title: "lstm_numpy.py — LSTM cell and sequence forward pass, verified against torch.nn.LSTM",
        code: String.raw`import numpy as np
import torch

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lstm_cell(x_t, h_prev, c_prev, W_ih, W_hh, b):
    """One LSTM step. Gate blocks are stacked in PyTorch order: i, f, g, o.
    x_t: (B, x)   h_prev, c_prev: (B, h)
    W_ih: (4h, x)  W_hh: (4h, h)  b: (4h,)"""
    H = h_prev.shape[1]
    z = x_t @ W_ih.T + h_prev @ W_hh.T + b      # (B, 4h) all pre-activations at once
    i = sigmoid(z[:, 0*H:1*H])                  # input gate
    f = sigmoid(z[:, 1*H:2*H])                  # forget gate
    g = np.tanh(z[:, 2*H:3*H])                  # candidate cell state
    o = sigmoid(z[:, 3*H:4*H])                  # output gate
    c = f * c_prev + i * g                      # additive cell update
    h = o * np.tanh(c)                          # exposed hidden state
    return h, c

def lstm_forward(X, W_ih, W_hh, b, h0, c0):
    """X: (T, B, x) -> all hidden states (T, B, h), final (h_T, c_T)."""
    h, c = h0, c0
    hs = []
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_ih, W_hh, b)
        hs.append(h)
    return np.stack(hs), (h, c)

T, B, X_DIM, H = 5, 2, 3, 4          # sequence length, batch, input size, hidden size
torch.manual_seed(0)
ref = torch.nn.LSTM(input_size=X_DIM, hidden_size=H)   # input layout (T, B, x)

# Copy PyTorch's weights into NumPy. PyTorch keeps two biases (b_ih, b_hh); they are simply added.
W_ih = ref.weight_ih_l0.detach().numpy().astype(np.float64)
W_hh = ref.weight_hh_l0.detach().numpy().astype(np.float64)
b    = (ref.bias_ih_l0 + ref.bias_hh_l0).detach().numpy().astype(np.float64)
print("W_ih", W_ih.shape, "W_hh", W_hh.shape, "b", b.shape)

rng = np.random.default_rng(0)
X  = rng.normal(size=(T, B, X_DIM))
h0 = np.zeros((B, H)); c0 = np.zeros((B, H))

hs_np, (hT_np, cT_np) = lstm_forward(X, W_ih, W_hh, b, h0, c0)
with torch.no_grad():
    hs_t, (hT_t, cT_t) = ref.double()(torch.from_numpy(X))

print("h_T (numpy):\n", np.round(hT_np, 6))
print("h_T (torch):\n", np.round(hT_t[0].numpy(), 6))
print("max |h diff| over all steps:", np.abs(hs_np - hs_t.numpy()).max())
print("max |c_T diff|:", np.abs(cT_np - cT_t[0].numpy()).max())
n_params = sum(p.numel() for p in ref.parameters())
print("PyTorch parameter count:", n_params, "| formula 4(h(h+x)+h) =", 4*(H*(H+X_DIM)+H), "+ extra bias 4h =", 4*H)`,
        output: `W_ih (16, 3) W_hh (16, 4) b (16,)
h_T (numpy):
 [[-0.022343 -0.084882  0.133186  0.106216]
 [ 0.047931  0.132517 -0.258631  0.248397]]
h_T (torch):
 [[-0.022343 -0.084882  0.133186  0.106216]
 [ 0.047931  0.132517 -0.258631  0.248397]]
max |h diff| over all steps: 5.551115123125783e-17
max |c_T diff|: 1.1102230246251565e-16
PyTorch parameter count: 144 | formula 4(h(h+x)+h) = 128 + extra bias 4h = 16`,
      },
      { t: "list", items: [
        "A difference of about $10^{-16}$ is float64 rounding: both versions compute the same function.",
        "`ref.double()` turns the PyTorch module into float64, so float32 precision does not limit the comparison. float32 weights convert to float64 exactly, so both sides use identical numbers.",
        "The parameter count confirms the formula: $4(4 \\cdot 7 + 4) = 128$, plus PyTorch's second bias vector ($4h = 16$), gives 144.",
      ] },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              { t: "p", text: "Task: given 30 samples of a noisy sine wave with random frequency and phase, predict the next value at every position. Working out the frequency needs memory. The persistence baseline (\"next value = current value\") gives a reference error." },
              {
                t: "code",
                lang: "python",
                title: "train_sine_lstm.py — runs in about 8 s on a laptop CPU",
                code: String.raw`import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- Data: noisy sine waves with random phase and frequency; predict the next value ----
SEQ_LEN, N_TRAIN, N_VAL = 30, 512, 128

def make_batch(n):
    t = torch.arange(SEQ_LEN + 1, dtype=torch.float32)
    freq  = 0.1 + 0.2 * torch.rand(n, 1)            # radians per step, in [0.1, 0.3]
    phase = 2 * math.pi * torch.rand(n, 1)
    wave  = torch.sin(freq * t + phase) + 0.05 * torch.randn(n, SEQ_LEN + 1)
    x = wave[:, :-1].unsqueeze(-1)                 # (n, T, 1) inputs  x_0 .. x_{T-1}
    y = wave[:, 1:].unsqueeze(-1)                  # (n, T, 1) targets x_1 .. x_T
    return x, y

x_train, y_train = make_batch(N_TRAIN)
x_val, y_val = make_batch(N_VAL)

class SineLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)           # hidden state -> next value
    def forward(self, x):
        hs, _ = self.lstm(x)                       # (B, T, hidden): h_t for every step
        return self.head(hs)                       # (B, T, 1): prediction at every step

model = SineLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# Baseline: "predict the next value equals the current value"
print(f"persistence baseline val MSE: {loss_fn(x_val, y_val).item():.4f}")

for epoch in range(1, 61):
    model.train()
    perm = torch.randperm(N_TRAIN)
    for k in range(0, N_TRAIN, 64):
        idx = perm[k:k + 64]
        opt.zero_grad()
        loss = loss_fn(model(x_train[idx]), y_train[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # standard safety for RNNs
        opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_val), y_val).item()
        print(f"epoch {epoch:2d}  train MSE {loss.item():.4f}  val MSE {val:.4f}")

# One-step-ahead prediction at the last position of one validation sequence
with torch.no_grad():
    pred = model(x_val[:1])[0, -1, 0].item()
print(f"true next value {y_val[0, -1, 0].item():+.3f}  predicted {pred:+.3f}")`,
                output: `persistence baseline val MSE: 0.0269
epoch 10  train MSE 0.0114  val MSE 0.0109
epoch 20  train MSE 0.0089  val MSE 0.0087
epoch 30  train MSE 0.0084  val MSE 0.0079
epoch 40  train MSE 0.0072  val MSE 0.0077
epoch 50  train MSE 0.0068  val MSE 0.0064
epoch 60  train MSE 0.0067  val MSE 0.0068
true next value +0.914  predicted +0.938`,
              },
              { t: "list", items: [
                "The model reaches a validation MSE of about 0.0065, four times lower than the persistence baseline (0.0269). The noise sets a floor no model can beat: the target itself has noise of variance $0.05^2 = 0.0025$, and the inputs are noisy too.",
                "`batch_first=True` makes inputs `(B, T, features)`; the default is `(T, B, features)`.",
                "`nn.LSTM` returns `(output, (h_n, c_n))`: `output` holds $h_t$ for every step of the last layer; `h_n` and `c_n` hold every layer's final states.",
                "Gradient clipping (`clip_grad_norm_`) is cheap insurance: LSTMs ease vanishing gradients but can still produce the odd large gradient.",
              ] },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "code",
                lang: "python",
                title: "train_sine_lstm_keras.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
tf.random.set_seed(0)
SEQ_LEN = 30

def make_batch(n):
    t = np.arange(SEQ_LEN + 1, dtype=np.float32)
    freq = 0.1 + 0.2 * rng.random((n, 1), dtype=np.float32)
    phase = 2 * np.pi * rng.random((n, 1), dtype=np.float32)
    wave = np.sin(freq * t + phase) + 0.05 * rng.standard_normal((n, SEQ_LEN + 1)).astype(np.float32)
    return wave[:, :-1, None], wave[:, 1:, None]   # (n, T, 1) inputs and next-value targets

x_train, y_train = make_batch(512)
x_val, y_val = make_batch(128)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(SEQ_LEN, 1)),
    # return_sequences=True -> output h_t at every step; unit_forget_bias=True (default) sets b_f = 1
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-2, clipnorm=1.0), loss="mse")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=60, batch_size=64, verbose=2)`,
                output: "Prints the model summary (the LSTM layer has 4(32·(32+1)+32) = 4352 parameters, the Dense layer 33), then one line per epoch with training and validation MSE, which should fall well below the persistence baseline of roughly 0.027, similar to the PyTorch run.",
              },
              { t: "list", items: [
                "Keras stores LSTM weights as `kernel` (input part, shape `(d, 4h)`), `recurrent_kernel` (shape `(h, 4h)`) and one `bias` (shape `(4h,)`), with gate blocks in the order i, f, c, o. Note the layout is transposed compared with PyTorch's `(4h, d)`.",
                "Keras uses one bias vector, so its parameter count matches $4(h(h+d)+h)$ exactly.",
                "`unit_forget_bias=True` sets the forget-gate bias to 1 at the start, as Jozefowicz, Zaremba and Sutskever (2015) recommend.",
              ] },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      { t: "h", text: "Peephole connections" },
      { t: "p", text: "In the standard cell the gates see only $h_{t-1}$ and $x_t$, not the memory they control. **Peephole** connections (Gers and Schmidhuber, 2000) let the gates look at the cell state through per-unit weight vectors $p_\\ast \\in \\mathbb{R}^h$:" },
      {
        t: "math",
        tex: String.raw`f_t = \sigma(W_f[h_{t-1};x_t] + p_f \odot c_{t-1} + b_f),\quad
i_t = \sigma(W_i[h_{t-1};x_t] + p_i \odot c_{t-1} + b_i),\quad
o_t = \sigma(W_o[h_{t-1};x_t] + p_o \odot c_t + b_o)`,
      },
      { t: "p", text: "They were meant for tasks that need precise timing. A large comparison by Greff et al. (2017), *LSTM: A Search Space Odyssey*, found peepholes did not clearly help on their benchmarks, while removing the forget gate or the output activation hurt a lot. `torch.nn.LSTM` and Keras `LSTM` have no peepholes." },
      { t: "h", text: "Gated Recurrent Unit (GRU)" },
      { t: "p", text: "The GRU (Cho et al., 2014) merges the cell and hidden states and uses two gates. An **update gate** $z_t$ blends the old state with a candidate; a **reset gate** $r_t$ controls how much of the old state goes into the candidate. See [GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants) for the details." },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
z_t &= \sigma(W_z[h_{t-1};x_t] + b_z) \\
r_t &= \sigma(W_r[h_{t-1};x_t] + b_r) \\
\tilde h_t &= \tanh(W_h[r_t \odot h_{t-1};\,x_t] + b_h) \\
h_t &= z_t\odot h_{t-1} + (1 - z_t) \odot \tilde h_t
\end{aligned}`,
        caption: "GRU in the convention of Cho et al. (2014), which PyTorch also uses for the blend; some texts swap z and 1 − z. PyTorch applies the reset gate after the recurrent matrix multiply: ñ = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)). The full [GRU lesson](/learn/m09/gru-variants) uses the other (Chung et al., 2014) convention, where z is the fraction updated, and shows how to convert.",
      },
      { t: "table", head: ["", "Vanilla RNN", "LSTM", "GRU"], rows: [
        ["State", "$h_t$", "$h_t$ and $c_t$", "$h_t$"],
        ["Gates", "none", "forget, input, output (+ candidate)", "update, reset (+ candidate)"],
        ["Parameters", "$h(h+d)+h$", "$4(h(h+d)+h)$", "$3(h(h+d)+h)$"],
        ["Memory path", "matrix multiply + tanh every step", "element-wise × and + (cell line)", "element-wise blend"],
        ["Long-range dependencies", "poor (≈10–20 steps)", "good", "good, similar to LSTM on many tasks"],
        ["Pros", "simplest, fastest per step", "separate memory and output; solid default", "≈25% fewer parameters than LSTM; simpler"],
        ["Cons", "vanishing/exploding gradients", "most parameters and compute", "no separate output gate; memory always exposed"],
        ["Typical use", "teaching, very short sequences", "speech, time series, language models before 2018", "same areas when compute/memory is tight"],
      ] },
      { t: "h", text: "Other common variations" },
      { t: "list", items: [
        "**Stacked (deep) LSTM**: layer $\\ell$'s hidden sequence is layer $\\ell+1$'s input (`num_layers` in PyTorch).",
        "**Bidirectional LSTM**: one LSTM reads left-to-right, another right-to-left, and their hidden states are joined; used when the whole sequence is available (tagging, encoding).",
        "**Projected LSTM (LSTMP)**: $h_t$ is multiplied by a projection matrix to make it smaller, which cuts the cost of the recurrent matrix (`proj_size` in PyTorch).",
        "**Coupled input–forget gate**: set $i_t = 1 - f_t$, saving one gate — a step towards the GRU.",
        "**ConvLSTM**: swaps the matrix multiplies for convolutions so the states are feature maps; used for space-and-time data such as rain nowcasting.",
      ] },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      { t: "list", items: [
        "**Machine translation**: Sutskever, Vinyals and Le (2014) used a deep LSTM encoder and decoder; Google's translation system (Wu et al., 2016) ran stacked LSTMs with attention in production before Transformers.",
        "**Speech recognition**: deep bidirectional LSTMs (Graves, Mohamed and Hinton, 2013) set records on TIMIT phoneme recognition, and LSTMs ran in commercial speech recognisers in the mid-2010s.",
        "**Handwriting**: LSTMs trained with CTC loss read handwriting; Graves (2013) generated realistic handwriting with one.",
        "**Time-series forecasting**: energy demand, sensors, sales, anomaly detection — anywhere data streams in and a small, constant-memory model is wanted.",
        "**On-device and streaming models**: keyboards, wake-word detection and low-latency audio, where a fixed-size state is cheaper than attention over a growing history.",
        "**Reinforcement learning**: LSTM policies give agents memory when they cannot see the whole environment at once.",
      ] },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "table", head: ["Strengths", "Limitations"], rows: [
        ["Learns links over hundreds of steps where vanilla RNNs fail.", "Still sequential: step $t$ needs step $t-1$, so training cannot run in parallel over time like a Transformer."],
        ["Constant memory and compute per step: ideal for streaming and long-running inference.", "Everything must be squeezed into a fixed-size state; exact recall over very long ranges (copying, retrieval) is hard."],
        ["Robust and well understood; works with small datasets.", "Four times the parameters and compute of a vanilla RNN of the same hidden size."],
        ["Gates act like soft switches you can inspect.", "Gradients still decay at the forget-gate rate; very long contexts stay difficult."],
      ] },
      { t: "p", text: "In practice: GRUs are often as accurate with fewer parameters; Transformers win with plenty of data and compute when the sequence fits the context window; LSTMs suit low-latency streaming, small data and tight memory." },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      { t: "list", items: [
        "**Mixing up gate order when copying weights.** PyTorch is i, f, g, o; copying in f, i, g, o order gives a model that runs but behaves differently. Fix: slice by name and test against a reference, as in the NumPy section.",
        "**Forgetting that PyTorch has two biases.** `bias_ih` and `bias_hh` are added together. Exporting only one gives wrong outputs. Fix: sum them.",
        "**Starting the forget bias at 0.** Then $f_t \\approx 0.5$ at first and memory halves every step, which brings back vanishing gradients early in training. Fix: set $b_f$ to 1 (Keras does this by default; in PyTorch do it by hand).",
        "**Wrong tensor layout.** `nn.LSTM` defaults to `(T, B, features)`; passing `(B, T, features)` without `batch_first=True` silently treats batch as time. Set `batch_first` explicitly.",
        "**Using `output[:, -1]` with padded sequences.** For short sequences in a padded batch, the last position is padding. Fix: use `pack_padded_sequence` and take `h_n`, or pick the true last index.",
        "**Thinking the cell state is bounded.** Only $h_t$ is in $(-1,1)$; $c_t$ can grow without limit (it reached 1.106 in the worked example). A large $c_t$ saturates $\\tanh(c_t)$.",
        "**Claiming LSTMs eliminate vanishing gradients.** They give a path whose decay is set by learned forget gates. With $f_t < 1$ the signal still decays, and the indirect paths through $h$ can still vanish. Those paths can also explode, so clip.",
        "**Carrying state across unrelated batches.** Reusing $(h, c)$ between independent sequences leaks information. Fix: reset to zeros, unless you are deliberately doing stateful truncated BPTT (and then `detach()` the carried state).",
      ] },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What problem does the LSTM solve?", a: "Vanilla RNNs cannot learn long-range dependencies: gradients flowing back through time are multiplied over and over by $W_{hh}$ and $\\tanh'$, so they vanish or explode exponentially. The LSTM adds a gated cell state, updated by addition, along which gradients decay only as fast as the learned forget gates allow." },
      { t: "qa", level: "beginner", q: "Name the gates and what each does.", a: "The forget gate $f_t$ picks what fraction of each memory unit to keep. The input gate $i_t$ picks how much of the candidate $g_t$ to write. The output gate $o_t$ picks which parts of $\\tanh(c_t)$ to expose as $h_t$. All three are sigmoids with outputs in $(0,1)$, one value per hidden unit." },
      { t: "qa", level: "beginner", q: "What is the difference between $h_t$ and $c_t$?", a: "$c_t$ is the internal long-term memory, updated by $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$ and unbounded. $h_t = o_t \\odot \\tanh(c_t)$ is a filtered, bounded view of it: the layer's output and the recurrent input to the next step's gates." },
      { t: "qa", level: "intermediate", q: "Derive $\\partial c_t/\\partial c_{t-1}$ and explain why it matters.", a: "Holding the gates fixed, $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$ gives $\\partial c_t/\\partial c_{t-1} = \\operatorname{diag}(f_t)$. Over $k$ steps the direct-path Jacobian is $\\operatorname{diag}(\\prod f_t)$: no weight-matrix powers and no activation derivatives. If the network sets $f_t \\approx 1$, the gradient is preserved. The full Jacobian also has indirect terms via $h_{t-1}$ and the gates, which can vanish, but the direct path makes long-range credit assignment possible." },
      { t: "qa", level: "intermediate", q: "How many parameters does an LSTM layer with input size 100 and hidden size 200 have?", a: "$4(h(h+d)+h) = 4(200 \\cdot 300 + 200) = 240{,}800$. In PyTorch, which has two bias vectors, it is $240{,}800 + 800 = 241{,}600$." },
      { t: "qa", level: "intermediate", q: "Why initialise the forget-gate bias to 1 (or higher)?", a: "With zero bias and small weights, $f_t \\approx \\sigma(0) = 0.5$: memory halves every step and cell-path gradients shrink like $0.5^k$ from the start. A bias of 1 gives $f_t \\approx 0.73$; larger values give longer default memory, so the network can learn long dependencies before learning when to forget. Jozefowicz et al. (2015) found a forget bias of 1 helped consistently." },
      { t: "qa", level: "intermediate", q: "Compare LSTM and GRU.", a: "The GRU merges cell and hidden state. An update gate blends the old state with a candidate; a reset gate controls how much history goes into the candidate. It has three weight blocks instead of four (about 25% fewer parameters) and no output gate, so its whole memory is always exposed. They perform similarly on many tasks; LSTM is often slightly better for precise counting or long memory, GRU is cheaper." },
      { t: "qa", level: "advanced", q: "Why can't LSTM training be parallelised over time like a Transformer, and what does that imply?", a: "Each step's gates depend on $h_{t-1}$, the previous step's nonlinear output, so the $T$ steps must run in order. Only the input projections $W_{ih}x_t$ can be computed for all $t$ at once. So training time grows linearly with sequence length on any hardware, while a Transformer layer handles all positions at once. This, more than accuracy, drove the shift to Transformers on large datasets. At inference, though, the LSTM's constant per-step cost is an advantage." },
      { t: "qa", level: "advanced", q: "What are peephole connections, and are they worth using?", a: "Peepholes feed the cell state into the gates ($p_f \\odot c_{t-1}$, $p_i \\odot c_{t-1}$, $p_o \\odot c_t$), so gates see the memory they control. Gers and Schmidhuber (2000) introduced them for precise timing tasks. Greff et al. (2017) found no significant benefit on speech, handwriting and music benchmarks, and mainstream libraries leave them out. They can matter on tasks that need counting time intervals." },
      { t: "qa", level: "advanced", q: "Can an LSTM's gradients still explode? Where from?", a: "Yes. The direct cell path is bounded by $\\prod f_t \\le 1$, but gradients also flow through $h_{t-1} \\to$ gates $\\to c_t$, which passes through the recurrent matrices $W_{hh}$ and can grow. Also, a weight's gradient sums pieces from every time step, so long sequences can give large totals. Gradient-norm clipping is standard practice." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "Which operation updates the LSTM cell state from $c_{t-1}$ to $c_t$?",
            options: ["A matrix multiplication followed by tanh", "Element-wise multiplication by $f_t$ plus element-wise addition of $i_t \\odot g_t$", "Softmax over the hidden units", "Concatenation with $x_t$"],
            answer: 1,
            explain: "$c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$. No weight matrix and no squashing touch the cell path.",
          },
          {
            kind: "numeric",
            q: "How many parameters does an LSTM layer with input size 10 and hidden size 20 have, using the single-bias formula $4(h(h+d)+h)$?",
            answer: 2480,
            tolerance: 0,
            explain: "$4(20 \\cdot 30 + 20) = 4 \\cdot 620 = 2480$. (PyTorch would report 2560 because of its second bias vector.)",
          },
          {
            kind: "numeric",
            q: "A unit's forget gate stays at 0.9 and its input gate is 0. What fraction of its cell value is left after 20 steps? (3 decimals)",
            answer: 0.122,
            tolerance: 0.002,
            explain: "$0.9^{20} = 0.1216$.",
          },
          {
            q: "In PyTorch, in which gate order are the rows of `weight_ih_l0` stacked?",
            options: ["f, i, g, o", "i, f, g, o", "i, g, f, o", "o, f, i, g"],
            answer: 1,
            explain: "PyTorch uses input, forget, cell (g), output.",
          },
          {
            kind: "numeric",
            q: "If the weights contribute zero and the forget bias is $b_f = 1$, what is the forget gate value? (3 decimals)",
            answer: 0.731,
            tolerance: 0.001,
            explain: "$\\sigma(1) = 1/(1+e^{-1}) = 0.7311$.",
          },
          {
            q: "Which statement about $h_t$ and $c_t$ is true?",
            options: ["Both are bounded in (−1, 1)", "$c_t$ is bounded, $h_t$ is not", "$h_t$ is bounded in (−1, 1), $c_t$ is not", "Neither is bounded"],
            answer: 2,
            explain: "$h_t = o_t \\odot \\tanh(c_t)$ lies in $(-1,1)$; $c_t$ adds up updates and can go past 1.",
          },
          {
            q: "What does the GRU's update gate $z_t$ do?",
            options: ["Scales the output like an output gate", "Interpolates between the previous state and the candidate", "Resets the input", "Normalises the hidden state"],
            answer: 1,
            explain: "$h_t = z_t \\odot h_{t-1} + (1-z_t) \\odot \\tilde h_t$: $z_t$ near 1 keeps the old state; near 0 replaces it with the candidate.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Numerical: one step of a scalar LSTM",
        prompt: "A 1-unit LSTM gets $x_t = 2$ with $h_{t-1} = 0.5$, $c_{t-1} = 1$. Each gate has weights $(w_h, w_x, b)$: forget $(0.5, 0.25, 0)$, input $(-1, 0.5, 0)$, candidate $(1, -0.25, 0)$, output $(0, 1, -1)$. Compute $f_t, i_t, g_t, c_t, o_t, h_t$ to 4 decimals.",
        hint: "Pre-activation = $w_h h_{t-1} + w_x x_t + b$.",
        solution: [
          { t: "list", items: [
            "Forget: $0.5(0.5) + 0.25(2) = 0.75$, $f_t = \\sigma(0.75) = 0.6792$.",
            "Input: $-1(0.5) + 0.5(2) = 0.5$, $i_t = \\sigma(0.5) = 0.6225$.",
            "Candidate: $1(0.5) - 0.25(2) = 0$, $g_t = \\tanh(0) = 0$. Nothing new is written, whatever $i_t$ is.",
            "Cell: $c_t = 0.6792 \\times 1 + 0.6225 \\times 0 = 0.6792$.",
            "Output: $0(0.5) + 1(2) - 1 = 1$, $o_t = \\sigma(1) = 0.7311$.",
            "Hidden: $h_t = 0.7311 \\times \\tanh(0.6792) = 0.7311 \\times 0.5910 = 0.4320$.",
          ] },
        ],
      },
      {
        t: "exercise",
        title: "Coding: implement a GRU cell in NumPy and match torch.nn.GRU",
        prompt: "Write a NumPy GRU step with PyTorch's conventions (gate blocks r, z, n; two biases; reset applied to $W_{hn}h + b_{hn}$; $h_t = (1-z)\\odot n + z\\odot h_{t-1}$). Run it over a 6-step sequence and compare with `torch.nn.GRU`.",
        hint: "Compute the input and hidden parts separately, because the reset gate multiplies only the hidden part of the candidate.",
        solution: [
          {
            t: "code",
            lang: "python",
            code: String.raw`import numpy as np
import torch

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """PyTorch GRU step. Gate blocks stacked as r, z, n."""
    H = h.shape[1]
    gi = x @ W_ih.T + b_ih                     # input contributions   (B, 3H)
    gh = h @ W_hh.T + b_hh                     # hidden contributions  (B, 3H)
    r = sigmoid(gi[:, :H] + gh[:, :H])         # reset gate
    z = sigmoid(gi[:, H:2*H] + gh[:, H:2*H])   # update gate
    n = np.tanh(gi[:, 2*H:] + r * gh[:, 2*H:]) # candidate (reset applied to W_hn h + b_hn)
    return (1 - z) * n + z * h

torch.manual_seed(0)
ref = torch.nn.GRU(input_size=3, hidden_size=4).double()
P = {k: v.detach().numpy() for k, v in ref.named_parameters()}
X = np.random.default_rng(1).normal(size=(6, 2, 3))      # (T, B, x)
h = np.zeros((2, 4))
for t in range(6):
    h = gru_cell(X[t], h, P["weight_ih_l0"], P["weight_hh_l0"], P["bias_ih_l0"], P["bias_hh_l0"])
with torch.no_grad():
    _, hT = ref(torch.from_numpy(X))
print("max |difference| vs torch.nn.GRU:", np.abs(h - hT[0].numpy()).max())
print("GRU parameters:", sum(p.numel() for p in ref.parameters()))`,
            output: `max |difference| vs torch.nn.GRU: 1.1102230246251565e-16
GRU parameters: 108`,
          },
          { t: "p", text: "The parameter count is $3(4 \\cdot 7 + 4) = 96$ plus PyTorch's extra bias of $3h = 12$, giving 108." },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      { t: "list", items: [
        "Vanilla RNNs forget because gradients pass through $\\operatorname{diag}(1-h_t^2)W_{hh}$ every step and vanish or explode geometrically.",
        "An LSTM keeps a cell state $c_t$ updated by addition: $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$.",
        "Gates (sigmoid, per unit): forget $f_t$ erases, input $i_t$ writes, output $o_t$ shows. The candidate $g_t = \\tanh(\\cdot)$ proposes content.",
        "Hidden state: $h_t = o_t \\odot \\tanh(c_t)$ — the bounded output and recurrent input; $c_t$ is unbounded and private.",
        "Direct gradient path: $\\partial c_t / \\partial c_{t-1} = \\operatorname{diag}(f_t)$ — no weight matrix, no activation derivative; the decay rate is learned.",
        "Parameters: $4(h(h+d)+h)$ (PyTorch adds $4h$ for its second bias). GRU: $3(h(h+d)+h)$.",
        "Start the forget bias near 1; clip gradients; watch gate order (PyTorch i, f, g, o) and tensor layout.",
        "History: Hochreiter & Schmidhuber (1997) introduced the LSTM; Gers, Schmidhuber & Cummins (2000) added the forget gate; Gers & Schmidhuber (2000) added peepholes.",
      ] },
      {
        t: "math",
        tex: String.raw`c_t = \sigma(\cdot)_f \odot c_{t-1} + \sigma(\cdot)_i \odot \tanh(\cdot)_g, \qquad h_t = \sigma(\cdot)_o \odot \tanh(c_t)`,
        caption: "The LSTM in one line: each (·) is W[h_{t−1}; x_t] + b with that gate's own weights.",
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[GRU, bidirectional and stacked RNNs](/learn/m09/gru-variants) — the lighter gated cell, and how to stack recurrent layers and run them in both directions.",
        "[From Seq2Seq bottlenecks to attention](/learn/m10/attention-intro) — LSTM encoder–decoders squeeze a whole sentence into one vector; attention removes that bottleneck and leads straight to Transformers.",
        "[Text preprocessing and subword tokenization](/learn/m13/text-preprocessing) — how text becomes the token sequences that LSTMs and Transformers read.",
      ] },
    ],
  },
};

export default lesson;
