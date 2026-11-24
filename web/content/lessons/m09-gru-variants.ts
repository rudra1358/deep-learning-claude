import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m09/gru-variants",
  title: "GRU, Bidirectional and Stacked RNNs",
  summary:
    "The gated recurrent unit in full — update and reset gates, the two interpolation conventions, where the reset gate really sits in PyTorch, and why the (1 − z) path keeps gradients alive — then bidirectional and stacked recurrent networks, sequence packing, and a measured comparison of RNN, GRU, LSTM and bidirectional models on a task that needs future context.",
  minutes: 80,
  level: "intermediate",
  prerequisites: ["m09/rnn-basics", "m09/bptt", "m09/lstm", "m03/activation-functions"],
  next: ["m10/attention-intro", "m13/nlp-tasks"],
  terms: ["gru", "update-gate", "reset-gate", "bidirectional-rnn", "stacked-rnn", "sequence-packing"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "A **Gated Recurrent Unit** (GRU) is a recurrent cell that, like the [LSTM](/learn/m09/lstm), uses learned gates to decide what to remember and what to overwrite — but with a single state vector $h_t$ and two gates instead of a separate cell state and three gates. The **update gate** $z_t$ decides how much of each hidden unit is rewritten at this step; the **reset gate** $r_t$ decides how much of the old state is used to compute the proposed new content.",
      },
      {
        t: "p",
        text: "The GRU was introduced by **Cho et al. (2014)**, *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*, as the hidden unit of their encoder–decoder. Later that year **Chung, Gulcehre, Cho and Bengio (2014)**, *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*, compared it with the LSTM and a plain tanh RNN on polyphonic music and raw speech modelling: both gated units clearly beat the tanh RNN, and neither gated unit was consistently better than the other.",
      },
      {
        t: "p",
        text: "This lesson also covers two ways of *combining* recurrent layers that apply to any cell type. A **bidirectional RNN** (Schuster and Paliwal, 1997, *Bidirectional Recurrent Neural Networks*) runs one RNN forwards and another backwards over the same sequence, so every position's representation summarises both its past and its future. A **stacked** (deep) RNN feeds the hidden-state sequence of one recurrent layer into the next. Together with **sequence packing** — the mechanism frameworks use to process batches of sequences with different lengths correctly — these are the building blocks of practically every pre-Transformer sequence model: taggers, speech recognisers, translation encoders and contextual embeddings.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The one idea to remember",
        text: String.raw`A GRU's new state is a per-unit weighted average of its old state and a candidate: $h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t$. When $z_t \approx 0$ the state is copied forward unchanged — an identity connection through time, opened and closed by a learned gate.`,
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Sequential data and vanilla RNNs](/learn/m09/rnn-basics) — the recurrence $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$, hidden states and unrolling. The GRU's candidate is exactly this computation with a gate in front of $h_{t-1}$.`,
          "[Backpropagation through time](/learn/m09/bptt) — the gradient reaching an early state is a product of per-step Jacobians $\\partial h_t/\\partial h_{t-1}$; we will compute this Jacobian for the GRU.",
          "[Long Short-Term Memory](/learn/m09/lstm) — gates as sigmoid layers in $(0,1)$, the additive cell update and the forget-gate bias trick. This lesson builds on it and does not repeat it.",
          "[Activation functions](/learn/m03/activation-functions) — $\\sigma(a) = 1/(1+e^{-a})$ with $\\sigma' = \\sigma(1-\\sigma)$, and $\\tanh$ with $\\tanh' = 1-\\tanh^2$.",
          "[Jacobians](/learn/m02/jacobians-hessians) and the element-wise product $\\odot$ — needed for the gradient derivation.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Editing a shared document",
        text: "Think of the hidden state as a document with one paragraph per hidden unit. At every step an editor reads the new input. First she decides how much of the *current* document to consult while drafting (the **reset gate**): to start a fresh topic she ignores what is written. Then she drafts replacement paragraphs (the **candidate**). Finally, paragraph by paragraph, she decides how much of the old text to replace with her draft (the **update gate**). A paragraph whose update value is near zero is left exactly as it was — so a fact written on page one can survive untouched until page fifty.",
      },
      {
        t: "p",
        text: "The LSTM solves the forgetting problem of vanilla RNNs with a separate memory (the cell state), a forget gate, an input gate and an output gate. The GRU asks: do we need all of that? It makes three simplifications:",
      },
      {
        t: "list",
        items: [
          String.raw`**One state.** There is no separate cell state; $h_t$ is both memory and output. Nothing like the LSTM's output gate hides parts of memory from the rest of the network.`,
          String.raw`**Coupled forget and write.** Instead of independent forget and input gates, one update gate does both: keep a fraction $1-z_t$ of the old value and write a fraction $z_t$ of the new one. Keeping more automatically means writing less.`,
          String.raw`**A reset gate for the candidate.** The candidate is computed like a vanilla RNN state, except that the old state is first multiplied by $r_t$. With $r_t \approx 0$ the candidate depends almost only on the current input, which lets a unit "start over" (for example at a phrase boundary).`,
        ],
      },
      {
        t: "p",
        text: "Bidirectionality and stacking are separate ideas. Reading a sentence like \"the **bank** of the river\", the meaning of *bank* is settled by words that come *after* it. A forward RNN at position 2 has not seen them yet; a backward RNN has. Concatenating the two gives every position access to the whole sentence. Stacking layers, as in deep feed-forward networks, lets lower layers compute local features and higher layers compose them into more abstract ones.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      {
        t: "p",
        text: String.raw`Notation for one time step: $x_t \in \mathbb{R}^d$ is the input, $h_{t-1} \in \mathbb{R}^h$ the previous hidden state. Each of the three internal layers has a weight matrix of shape $h \times (h+d)$ acting on a concatenated vector, and a bias of size $h$. We write the interpolation with $z_t$ = **fraction updated**, the convention of Chung et al. (2014); the next callout explains the other convention, which PyTorch and Keras use.`,
      },
      {
        t: "steps",
        items: [
          {
            title: "1. Reset gate — how much history the candidate may use",
            text: "Shape $h$, values in $(0,1)$, one per unit.",
            tex: String.raw`r_t = \sigma\!\left(W_r\,[h_{t-1};x_t] + b_r\right)`,
          },
          {
            title: "2. Candidate state — the proposed new content",
            text: String.raw`Shape $h$, values in $(-1,1)$. A vanilla-RNN step whose recurrent input is the *reset* state $r_t \odot h_{t-1}$. This is the original placement of Cho et al. (2014).`,
            tex: String.raw`\tilde h_t = \tanh\!\left(W_h\,[r_t \odot h_{t-1};\,x_t] + b_h\right)`,
          },
          {
            title: "3. Update gate — how much of each unit to rewrite",
            text: "Shape $h$, values in $(0,1)$. Reads the same $[h_{t-1}; x_t]$ as the reset gate, with its own weights.",
            tex: String.raw`z_t = \sigma\!\left(W_z\,[h_{t-1};x_t] + b_z\right)`,
          },
          {
            title: "4. New hidden state — interpolate",
            text: String.raw`Each unit moves a fraction $z_t$ of the way from its old value towards the candidate. Because $h_{t-1}$ and $\tilde h_t$ are both in $(-1,1)$ and the weights sum to 1, so is $h_t$.`,
            tex: String.raw`h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t`,
          },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        title: "Two conventions for z — check before you copy equations or weights",
        text: String.raw`Cho et al. (2014), PyTorch's nn.GRU and Keras's GRU write $h_t = z_t \odot h_{t-1} + (1-z_t)\odot \tilde h_t$: there $z_t$ is the fraction **kept** (the [LSTM lesson](/learn/m09/lstm) quotes this form). Chung et al. (2014) and many tutorials write $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t$, with $z_t$ the fraction **updated** — the form used in this lesson and its simulator. The two are the same model family: since $1-\sigma(a) = \sigma(-a)$, negating $W_z$ and $b_z$ converts one into the other. What changes is the sign of everything you say about $z$: "small $z$ = long memory" here is "large $z$ = long memory" in PyTorch.`,
      },
      {
        t: "h",
        text: "Where the reset gate sits: the PyTorch / cuDNN variant",
      },
      {
        t: "p",
        text: String.raw`PyTorch's nn.GRU documentation gives $n_t = \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn}))$: the reset gate multiplies the recurrent term **after** the matrix multiply (and after its bias), not the state before it. This is the formulation used by NVIDIA's cuDNN kernel, because it lets all three recurrent matrix products $W_{hr}h, W_{hz}h, W_{hn}h$ be computed as one $3h \times h$ multiply before the gates are known. Keras exposes the choice as reset_after (default True in TensorFlow 2, matching cuDNN); reset_after=False gives the original Cho placement. The two placements are **different functions** — in the worked example below they give $h_t$ values that differ in the second decimal — so weights trained with one cannot be loaded into the other. The NumPy implementation in this lesson reproduces PyTorch's version bit for bit.`,
      },
      {
        t: "table",
        head: ["Quantity", "Shape (one example)", "Shape (batch B)", "Range"],
        rows: [
          ["$x_t$", "$d$", "$B \\times d$", "any"],
          ["$h_{t-1}$, $h_t$, $\\tilde h_t$", "$h$", "$B \\times h$", "$(-1,1)$"],
          ["$W_r, W_z, W_h$", "$h \\times (h+d)$ each", "shared", "—"],
          ["$r_t$, $z_t$", "$h$", "$B \\times h$", "$(0,1)$"],
          ["PyTorch weight_ih_l0 / weight_hh_l0", "$3h \\times d$ / $3h \\times h$, blocks $r, z, n$", "shared", "—"],
          ["PyTorch bias_ih_l0, bias_hh_l0", "$3h$ each", "shared", "—"],
        ],
      },
      {
        t: "h",
        text: "Bidirectional RNNs",
      },
      {
        t: "list",
        ordered: true,
        items: [
          String.raw`A **forward** RNN reads $x_1, \dots, x_T$ and produces $\overrightarrow{h}_1, \dots, \overrightarrow{h}_T$; $\overrightarrow{h}_t$ summarises $x_1..x_t$.`,
          String.raw`A **backward** RNN, with its own weights, reads $x_T, \dots, x_1$ and produces $\overleftarrow{h}_T, \dots, \overleftarrow{h}_1$; $\overleftarrow{h}_t$ summarises $x_t..x_T$.`,
          String.raw`The output at position $t$ is the concatenation $y_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2h}$, which depends on the **entire** sequence. Both passes are independent and can run in parallel.`,
          String.raw`For a whole-sequence summary use the two **final** states: $\overrightarrow{h}_T$ (forward, at the last position) and $\overleftarrow{h}_1$ (backward, at the *first* position). In PyTorch these are h_n[0] and h_n[1] — not output[:, -1], whose backward half has seen only $x_T$.`,
        ],
      },
      {
        t: "callout",
        kind: "warning",
        title: "When you cannot use a bidirectional RNN",
        text: String.raw`The backward pass needs $x_T$ before it can produce anything, so a bidirectional model can only be used when the whole sequence is available at prediction time: tagging a finished sentence, encoding a source sentence for translation, classifying a recorded utterance. It is **not** usable for causal or online tasks — next-token language modelling (the backward RNN would see the answer), streaming speech recognition with low latency, real-time control or forecasting. Training a bidirectional model on next-step prediction leaks the target and gives a meaninglessly low loss.`,
      },
      {
        t: "h",
        text: "Stacked (deep) RNNs",
      },
      {
        t: "p",
        text: String.raw`In an $L$-layer stack, layer 1 reads $x_1..x_T$; layer $\ell > 1$ reads the hidden-state sequence $h^{(\ell-1)}_1..h^{(\ell-1)}_T$ of the layer below. Each layer has its own weights and its own recurrence, so the state at $(\ell, t)$ depends on $(\ell, t-1)$ and $(\ell-1, t)$. With bidirectional layers, layer $\ell+1$ reads the $2h$-dimensional concatenations, so its input size is $2h$. Three practical additions:`,
      },
      {
        t: "list",
        items: [
          "**Dropout between layers.** Zaremba, Sutskever and Vinyals (2014), *Recurrent Neural Network Regularization*, applied dropout only to the non-recurrent (layer-to-layer) connections, leaving the recurrent path intact. PyTorch's `dropout` argument does exactly this: it applies to the outputs of every layer except the last, and it has no effect when `num_layers=1` (PyTorch warns).",
          "**Residual connections between layers.** $h^{(\\ell)}_t = h^{(\\ell-1)}_t + \\mathrm{RNN}^{(\\ell)}(\\cdot)_t$, as in [residual networks](/learn/m07/residual-connections). Google's Neural Machine Translation system (Wu et al., 2016) used residual connections in its 8-layer LSTM encoder and decoder, which the authors reported was necessary to train stacks that deep; only its bottom encoder layer was bidirectional.",
          "**Depth is usually small.** 2–4 recurrent layers is typical; beyond that, gains are small unless residual connections are used, and each layer adds a full sequential pass.",
        ],
      },
      {
        t: "h",
        text: "Variable-length sequences and packing",
      },
      {
        t: "p",
        text: "To batch sequences of different lengths they are padded to the longest length. A plain RNN then keeps stepping through the padding: a forward RNN's final state is computed *after* the padding, and a backward RNN starts *on* the padding. `torch.nn.utils.rnn.pack_padded_sequence` turns the padded batch plus a lengths vector into a `PackedSequence` that stores only real time steps (sorted by length internally); the RNN then processes, at each time step, only the sequences that are still active. `h_n` holds each sequence's true final state, and `pad_packed_sequence` converts the output back to a padded tensor with zeros after each sequence's end.",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      {
        t: "h",
        text: "Variables",
      },
      {
        t: "list",
        items: [
          "$t$ — time index; $T$ — sequence length; $d$ — input size; $h$ — hidden size.",
          "$x_t \\in \\mathbb{R}^d$ — input; $h_t \\in \\mathbb{R}^h$ — hidden state; $\\tilde h_t \\in (-1,1)^h$ — candidate.",
          "$r_t, z_t \\in (0,1)^h$ — reset and update gates.",
          "$W_r, W_z, W_h \\in \\mathbb{R}^{h\\times(h+d)}$; we split each as $W_\\ast = [U_\\ast \\; V_\\ast]$ with recurrent part $U_\\ast \\in \\mathbb{R}^{h\\times h}$ and input part $V_\\ast \\in \\mathbb{R}^{h \\times d}$. Biases $b_r, b_z, b_h \\in \\mathbb{R}^h$.",
          "$\\sigma$ — logistic sigmoid; $\\odot$ — element-wise product; $\\operatorname{diag}(v)$ — diagonal matrix with $v$ on the diagonal.",
        ],
      },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
r_t &= \sigma(U_r h_{t-1} + V_r x_t + b_r) \\
z_t &= \sigma(U_z h_{t-1} + V_z x_t + b_z) \\
\tilde h_t &= \tanh\big(U_h (r_t \odot h_{t-1}) + V_h x_t + b_h\big) \\
h_t &= (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t
\end{aligned}`,
        caption: "GRU with z = fraction updated (Chung et al., 2014) and the original reset placement (Cho et al., 2014).",
      },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
r_t &= \sigma(W_{ir}x_t + b_{ir} + W_{hr}h_{t-1} + b_{hr}) \\
z_t &= \sigma(W_{iz}x_t + b_{iz} + W_{hz}h_{t-1} + b_{hz}) \\
n_t &= \tanh\big(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn})\big) \\
h_t &= (1-z_t)\odot n_t + z_t \odot h_{t-1}
\end{aligned}`,
        caption: "torch.nn.GRU as documented: z = fraction kept, reset applied after the recurrent matrix multiply, two bias vectors.",
      },
      {
        t: "h",
        text: "Worked numerical step (h = 2, d = 1)",
      },
      {
        t: "p",
        text: String.raw`Take $x_t = 1$ and $h_{t-1} = (0.5, -0.3)$, so $[h_{t-1}; x_t] = (0.5, -0.3, 1)$. Each weight row multiplies $(h_1, h_2, x)$. The update bias is $b_z = (-1, -1)$, which biases the cell towards keeping its state; all other biases are 0.`,
      },
      {
        t: "math",
        tex: String.raw`W_r=\begin{pmatrix}0.4&-0.2&0.3\\0.1&0.5&-0.6\end{pmatrix},\;
W_z=\begin{pmatrix}0.3&0.2&-0.5\\-0.4&0.1&0.2\end{pmatrix},\;
W_h=\begin{pmatrix}0.6&-0.4&0.9\\-0.5&0.7&0.4\end{pmatrix}`,
      },
      {
        t: "steps",
        items: [
          {
            title: "Reset gate",
            text: "Unit 1: $0.4(0.5) - 0.2(-0.3) + 0.3(1) = 0.56$. Unit 2: $0.1(0.5) + 0.5(-0.3) - 0.6(1) = -0.70$.",
            tex: String.raw`r_t = \sigma(0.56,\,-0.70) = (0.6365,\;0.3318),\qquad r_t\odot h_{t-1} = (0.3182,\;-0.0995)`,
          },
          {
            title: "Candidate",
            text: "Unit 1: $0.6(0.3182) - 0.4(-0.0995) + 0.9(1) = 1.1308$. Unit 2: $-0.5(0.3182) + 0.7(-0.0995) + 0.4(1) = 0.1712$. Unit 2 sees only a third of its old value, because $r_2 = 0.33$.",
            tex: String.raw`\tilde h_t = \tanh(1.1308,\;0.1712) = (0.8113,\;0.1696)`,
          },
          {
            title: "Update gate",
            text: "Unit 1: $0.3(0.5) + 0.2(-0.3) - 0.5(1) - 1 = -1.41$. Unit 2: $-0.4(0.5) + 0.1(-0.3) + 0.2(1) - 1 = -1.03$.",
            tex: String.raw`z_t = \sigma(-1.41,\;-1.03) = (0.1962,\;0.2631)`,
          },
          {
            title: "Interpolate",
            text: "Kept: $(1-z_t)\\odot h_{t-1} = (0.8038 \\times 0.5,\\; 0.7369 \\times -0.3) = (0.4019, -0.2211)$. Written: $z_t \\odot \\tilde h_t = (0.1962 \\times 0.8113,\\; 0.2631 \\times 0.1696) = (0.1592, 0.0446)$.",
            tex: String.raw`h_t = (0.4019 + 0.1592,\;-0.2211 + 0.0446) = (0.5611,\;-0.1765)`,
          },
        ],
      },
      {
        t: "p",
        text: String.raw`With the reset gate applied after the matrix multiply (PyTorch placement), the candidate's pre-activation becomes $V_h x + r_t\odot(U_h h_{t-1}) = (0.9 + 0.6365 \times 0.42,\; 0.4 + 0.3318 \times (-0.46)) = (1.1673, 0.2474)$, so $\tilde h_t = (0.8234, 0.2424)$ and $h_t = (0.5635, -0.1573)$ — close to, but not the same as, $(0.5611, -0.1765)$.`,
      },
      {
        t: "h",
        text: "The Jacobian: an identity-like path through time",
      },
      {
        t: "p",
        text: String.raw`Differentiate $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t$ with respect to $h_{t-1}$, remembering that $z_t$ and $\tilde h_t$ also depend on $h_{t-1}$ (product rule):`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial h_t}{\partial h_{t-1}} = \underbrace{\operatorname{diag}(1-z_t)}_{\text{direct copy}} \;+\; \underbrace{\operatorname{diag}(z_t)\,\frac{\partial \tilde h_t}{\partial h_{t-1}}}_{\text{through the candidate}} \;+\; \underbrace{\operatorname{diag}(\tilde h_t - h_{t-1})\,\frac{\partial z_t}{\partial h_{t-1}}}_{\text{through the gate}}`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial z_t}{\partial h_{t-1}} = \operatorname{diag}\big(z_t(1-z_t)\big)U_z,\qquad
\frac{\partial \tilde h_t}{\partial h_{t-1}} = \operatorname{diag}(1-\tilde h_t^2)\,U_h\Big(\operatorname{diag}(r_t) + \operatorname{diag}(h_{t-1})\operatorname{diag}\big(r_t(1-r_t)\big)U_r\Big)`,
      },
      {
        t: "p",
        text: String.raw`The first term is the key. It contains **no weight matrix and no activation derivative** — just $1-z_t$. If a unit's $z_t \approx 0$, its entry is $\approx 1$, the other two terms are scaled by $z_t$ or $z_t(1-z_t)$ and nearly vanish, and the Jacobian along that unit is close to the identity. Over $k$ steps the direct path contributes $\prod (1-z_\tau)$, a decay rate the network *learns* per unit and per step — the GRU's counterpart of the LSTM's $\prod f_\tau$. A vanilla RNN, whose Jacobian $\operatorname{diag}(1-h_t^2)W_{hh}$ always passes through the weight matrix, has no such path.`,
      },
      {
        t: "p",
        text: String.raw`In the worked example the three terms are $\begin{pmatrix}0.8038&0\\0&0.7369\end{pmatrix} + \begin{pmatrix}0.0277&-0.0089\\-0.0884&0.0564\end{pmatrix} + \begin{pmatrix}0.0147&0.0098\\-0.0364&0.0091\end{pmatrix}$, giving $\begin{pmatrix}0.8462&0.0009\\-0.1248&0.8024\end{pmatrix}$: dominated by the identity-like diagonal. The script below recomputes the step and checks the Jacobian with automatic differentiation.`,
      },
      {
        t: "code",
        lang: "python",
        title: "gru_worked_step.py — the worked example, both reset placements, and the Jacobian by autograd",
        code: String.raw`import numpy as np
import torch

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

x = np.array([1.0])                               # input x_t
h_prev = np.array([0.5, -0.3])                    # previous hidden state h_{t-1}
# weight rows multiply [h1, h2, x]
W_r = np.array([[0.4, -0.2, 0.3], [0.1, 0.5, -0.6]])
W_z = np.array([[0.3, 0.2, -0.5], [-0.4, 0.1, 0.2]]); b_z = np.array([-1.0, -1.0])
W_h = np.array([[0.6, -0.4, 0.9], [-0.5, 0.7, 0.4]])

def gru_step(h_prev, reset_after=False):
    v = np.concatenate([h_prev, x])
    r = sigmoid(W_r @ v)                                        # reset gate
    if reset_after:                                             # PyTorch / cuDNN placement
        a_h = W_h[:, 2:] @ x + r * (W_h[:, :2] @ h_prev)
    else:                                                       # Cho et al. (2014) placement
        a_h = W_h @ np.concatenate([r * h_prev, x])
    h_tilde = np.tanh(a_h)                                      # candidate
    z = sigmoid(W_z @ v + b_z)                                  # update gate (fraction rewritten)
    return r, h_tilde, z, (1 - z) * h_prev + z * h_tilde

r, h_tilde, z, h = gru_step(h_prev)
print("r       =", np.round(r, 4))
print("h_tilde =", np.round(h_tilde, 4))
print("z       =", np.round(z, 4))
print("h_t     =", np.round(h, 4))
print("h_t with reset applied after the matmul =", np.round(gru_step(h_prev, reset_after=True)[3], 4))

# Jacobian dh_t/dh_{t-1} by automatic differentiation, compared with its identity-like part diag(1 - z)
def step_torch(hp):
    T = lambda a: torch.tensor(a)
    v = torch.cat([hp, T(x)])
    r = torch.sigmoid(T(W_r) @ v)
    z = torch.sigmoid(T(W_z) @ v + T(b_z))
    h_tilde = torch.tanh(T(W_h) @ torch.cat([r * hp, T(x)]))
    return (1 - z) * hp + z * h_tilde
J = torch.autograd.functional.jacobian(step_torch, torch.tensor(h_prev)).numpy()
print("dh_t/dh_{t-1} =\n", np.round(J, 4))
print("diag(1 - z)   =", np.round(1 - z, 4))`,
        output: `r       = [0.6365 0.3318]
h_tilde = [0.8113 0.1696]
z       = [0.1962 0.2631]
h_t     = [ 0.5611 -0.1765]
h_t with reset applied after the matmul = [ 0.5635 -0.1573]
dh_t/dh_{t-1} =
 [[ 0.8462  0.0009]
 [-0.1248  0.8024]]
diag(1 - z)   = [0.8038 0.7369]`,
      },
      {
        t: "h",
        text: "Measured: how far back does the gradient reach?",
      },
      {
        t: "p",
        text: String.raw`The same experiment as in the LSTM lesson: one random 60-step sequence, 16 hidden units, untrained layers, and the size of $\partial h_T / \partial x_t$ for inputs $k$ steps back. Recall that in PyTorch $z$ is the fraction *kept*, so a bias of $+2$ on PyTorch's $z$ means $1-z \approx \sigma(-2) \approx 0.12$ is rewritten per step — the "small update" regime of our convention.`,
      },
      {
        t: "code",
        lang: "python",
        title: "Measured: |∂h_T/∂x_t| for a vanilla RNN vs a GRU (60 steps, random init)",
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
gru0 = nn.GRU(1, H, batch_first=True)               # default init: update-gate bias ~ 0
gru2 = nn.GRU(1, H, batch_first=True)
with torch.no_grad():                               # PyTorch z = fraction KEPT; rows H..2H are z
    gru2.bias_ih_l0[H:2*H].fill_(2.0); gru2.bias_hh_l0[H:2*H].fill_(0.0)

g_rnn, g_gru0, g_gru2 = grad_wrt_inputs(rnn), grad_wrt_inputs(gru0), grad_wrt_inputs(gru2)
print(" steps back | vanilla RNN | GRU (default) | GRU (keep bias +2)")
for k in [0, 5, 10, 20, 40, 59]:
    t = T - 1 - k
    print(f"   {k:3d}      |  {g_rnn[t]:.2e}   |   {g_gru0[t]:.2e}    |   {g_gru2[t]:.2e}")`,
        output: ` steps back | vanilla RNN | GRU (default) | GRU (keep bias +2)
     0      |  2.56e-01   |   1.03e-01    |   6.55e-02
     5      |  1.65e-02   |   8.61e-04    |   2.99e-02
    10      |  2.28e-04   |   1.47e-05    |   1.77e-02
    20      |  6.47e-07   |   3.65e-06    |   8.80e-03
    40      |  2.27e-12   |   7.11e-10    |   1.53e-03
    59      |  8.50e-19   |   1.53e-13    |   1.50e-04`,
      },
      {
        t: "p",
        text: "With default initialisation the GRU's update gates sit near 0.5, so the direct path decays like $0.5^k$ and the untrained GRU is not better than the vanilla RNN at 20 steps. Biasing the gate towards keeping the state changes this completely: 59 steps back the gradient is still $1.5 \\times 10^{-4}$, about 15 orders of magnitude larger than the vanilla RNN's. Training typically learns such biases for units that need long memory; initialising them is the GRU analogue of the LSTM forget-bias trick.",
      },
      {
        t: "h",
        text: "Parameter count",
      },
      {
        t: "math",
        tex: String.raw`N_{\text{GRU}} = 3\big(h(h+d) + h\big), \qquad N_{\text{GRU}}^{\text{PyTorch}} = 3\big(h(h+d) + 2h\big), \qquad N_{\text{LSTM}} = 4\big(h(h+d)+h\big)`,
      },
      {
        t: "p",
        text: String.raw`Three blocks (reset, update, candidate), each an $h\times(h+d)$ matrix plus a bias. PyTorch (and Keras with reset_after=True) keep a second bias vector per block, adding $3h$; with reset-after these two biases are *not* redundant, because $b_{hn}$ sits inside the reset product. Example: $d = 128$, $h = 256$ gives $3(256 \cdot 384 + 256) = 295{,}680$ parameters (PyTorch: $296{,}448$), against $394{,}240$ for an LSTM — 25% fewer. A bidirectional layer doubles the count; in a stack, layers above the first have input size $h$ (or $2h$ if bidirectional).`,
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "sim",
        id: "gru",
        caption: "A 3- or 4-unit GRU with seeded weights. Each time step is split into four sub-steps (reset gate, candidate, update gate, new state), with the equation and substituted numbers in the narration. Below: the timeline and a memory probe comparing the GRU with a vanilla RNN on the same input.",
      },
      {
        t: "list",
        ordered: true,
        items: [
          "Press **Step** four times to walk through $t = 1$. Check the narration at sub-step 4: $h_t$ is exactly *kept* + *written*, and each unit of $h_t$ lies between $h_{t-1}$ and $\\tilde h_t$.",
          "Switch to hidden size 4 and watch the $z$ rows of the timeline on the default sequence (a pulse at $t = 1$, then zeros). Unit 3 has $z = 0.92$ at $t = 1$, when the strong input arrives, so the pulse is written almost completely; afterwards its $z$ stays near 0.19, so the value is mostly kept and decays slowly (0.824 at $t = 1$, still 0.311 at $t = 10$).",
          "Look at the **memory probe**. It adds 1 to $x_1$, reruns both models and plots $\\lVert \\Delta h_t \\rVert$. With the default $b_z = -1.5$ and hidden size 3, the GRU still carries about 40% of the perturbation at $t = 10$ (0.109 vs 0.264 at $t = 1$), while the vanilla RNN keeps 7% (0.016 vs 0.228).",
          "Drag **Update bias** to $-3$: $z_t \\approx 0.05$, the GRU state is copied forward almost unchanged and the probe stays nearly flat (0.120 at $t = 10$ vs 0.158 at $t = 1$). The price: less of the pulse is written in the first place, because $z_1$ is also smaller.",
          "Drag it to $+2$: $z_t \\approx 0.9$, the state is rewritten every step, and the GRU forgets $x_1$ faster than the vanilla RNN. Gating gives *control* over memory, not memory for free.",
          "Watch the **reset gate** sub-step for a unit whose $r$ is small: $r \\odot h_{t-1}$ is near zero and the candidate is computed almost from $x_t$ alone.",
          "Try your own input, e.g. `1, 0, 0, -1, 0, 0` or `2, 2, 2, 0, 0, 0`, and switch hidden size 3 ↔ 4 to get a different set of seeded weights.",
        ],
      },
      {
        t: "p",
        text: "For comparison, the LSTM simulator below runs the same kind of experiment with a forget-gate bias slider. The LSTM's forget bias plays the role of the GRU's $-b_z$ in this lesson's convention: high forget bias ↔ low update bias ↔ long memory.",
      },
      {
        t: "sim",
        id: "lstm",
        caption: "The LSTM cell for comparison: separate cell state, three gates.",
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "A complete NumPy GRU in PyTorch's exact convention (gate blocks r, z, n; two biases; reset after the recurrent multiply; $z$ = fraction kept), extended to **bidirectional** and **multi-layer** operation with PyTorch's output layout. It is checked against `torch.nn.GRU(num_layers=2, bidirectional=True)`; PyTorch is used only as the reference.",
      },
      {
        t: "code",
        lang: "python",
        title: "gru_numpy.py — stacked bidirectional GRU from scratch, verified against torch.nn.GRU",
        code: String.raw`import numpy as np
import torch

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

def gru_cell(x_t, h_prev, W_ih, W_hh, b_ih, b_hh):
    """One GRU step in PyTorch's convention.
    Row blocks of W_ih (3H, D) and W_hh (3H, H) are stacked as [reset r | update z | candidate n].
    x_t: (B, D)   h_prev: (B, H)   returns h_t: (B, H)."""
    H = h_prev.shape[1]
    gi = x_t @ W_ih.T + b_ih                      # input contributions      (B, 3H)
    gh = h_prev @ W_hh.T + b_hh                   # recurrent contributions  (B, 3H)
    r = sigmoid(gi[:, :H] + gh[:, :H])            # reset gate
    z = sigmoid(gi[:, H:2*H] + gh[:, H:2*H])      # update gate (PyTorch: z = fraction KEPT)
    n = np.tanh(gi[:, 2*H:] + r * gh[:, 2*H:])    # candidate; r scales (W_hn h + b_hn)
    return (1.0 - z) * n + z * h_prev             # interpolation

def gru_direction(X, h0, p, reverse=False):
    """Run one direction over X: (B, T, D) -> outputs (B, T, H), final state (B, H)."""
    T = X.shape[1]
    order = range(T - 1, -1, -1) if reverse else range(T)
    h, outs = h0, [None] * T
    for t in order:
        h = gru_cell(X[:, t], h, *p)
        outs[t] = h                               # store at the ORIGINAL position t
    return np.stack(outs, axis=1), h

def gru_stack(X, params, num_layers, bidirectional):
    """Multi-layer (optionally bidirectional) GRU, layout identical to torch.nn.GRU(batch_first=True)."""
    B = X.shape[0]
    finals, inp = [], X
    for layer in range(num_layers):
        dirs = []
        for d, rev in enumerate([False, True][: 2 if bidirectional else 1]):
            sfx = f"_l{layer}" + ("_reverse" if rev else "")
            p = [params[k + sfx] for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            H = p[1].shape[1]
            out, hT = gru_direction(inp, np.zeros((B, H)), p, reverse=rev)
            dirs.append(out); finals.append(hT)
        inp = np.concatenate(dirs, axis=-1)       # next layer reads [forward ; backward]
    return inp, np.stack(finals)                  # (B, T, dirs*H), (layers*dirs, B, H)

# ---- reference model: 2 layers, bidirectional, input 3, hidden 4 ----
B, T, D, H, L = 2, 6, 3, 4, 2
torch.manual_seed(0)
ref = torch.nn.GRU(D, H, num_layers=L, bidirectional=True, batch_first=True).double()
params = {k: v.detach().numpy() for k, v in ref.named_parameters()}
print("parameter tensors:", list(params)[:4], "...", len(params), "in total")

X = np.random.default_rng(0).normal(size=(B, T, D))
out_np, hn_np = gru_stack(X, params, L, bidirectional=True)
with torch.no_grad():
    out_t, hn_t = ref(torch.from_numpy(X))

print("output shape", out_np.shape, "| h_n shape", hn_np.shape)
print("max |output diff|:", np.abs(out_np - out_t.numpy()).max())
print("max |h_n diff|   :", np.abs(hn_np - hn_t.numpy()).max())

# ---- parameter count: 3(H(H+D) + H) per direction and layer, plus PyTorch's second bias 3H ----
def gru_params(d_in, h):
    return 3 * (h * (h + d_in) + h) + 3 * h
expected = 2 * gru_params(D, H) + 2 * gru_params(2 * H, H)   # layer 2 input is 2H (concatenated)
print("torch count:", sum(p.numel() for p in ref.parameters()), "| formula:", expected)

# ---- the other interpolation convention: z' = 1 - z ----
x1, h1 = X[:, 0], np.zeros((B, H))
p0 = [params[k + "_l0"] for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
gi = x1 @ p0[0].T + p0[2]; gh = h1 @ p0[1].T + p0[3]
z_torch = sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
z_other = sigmoid(-(gi[:, H:2*H] + gh[:, H:2*H]))    # negate the update-gate weights and biases
print("z_torch + z_other == 1:", np.allclose(z_torch + z_other, 1.0))`,
        output: `parameter tensors: ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'] ... 16 in total
output shape (2, 6, 8) | h_n shape (4, 2, 4)
max |output diff|: 1.1102230246251565e-16
max |h_n diff|   : 1.1102230246251565e-16
torch count: 552 | formula: 552
z_torch + z_other == 1: True`,
      },
      {
        t: "list",
        items: [
          "The differences of $10^{-16}$ are float64 rounding: the NumPy code computes the same function as PyTorch, including the layout of `h_n` (layer 0 forward, layer 0 backward, layer 1 forward, layer 1 backward).",
          "The backward direction stores each output at its **original** time index (`outs[t] = h`), so position $t$ of the concatenated output pairs $\\overrightarrow{h}_t$ with $\\overleftarrow{h}_t$. Its final state is the state at $t = 0$.",
          "Parameter count: layer 0 has $2 \\times 3(4\\cdot 7 + 2\\cdot 4) = 2 \\times 108 = 216$ parameters, layer 1 reads $2h = 8$ inputs and has $2 \\times 3(4 \\cdot 12 + 2 \\cdot 4) = 336$; total 552.",
          "The last line shows the convention change: negating the update-gate pre-activation turns PyTorch's \"fraction kept\" $z$ into this lesson's \"fraction updated\" $1-z$.",
        ],
      },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "h",
                text: "Experiment: a tagging task that needs the future",
              },
              {
                t: "p",
                text: String.raw`Each sequence is 40 random digits from $\{0,..,4\}$. Position $t$ gets label 1 if $x_{t+2} > x_0$, else 0. The answer needs the **first** token (long-range past) and a token **two steps ahead**. A causal model cannot see $x_{t+2}$; its best possible strategy is to guess from $x_0$ alone, which gives an accuracy of 0.760. A bidirectional model can in principle reach 100%. All six models get the same embedding (size 8), hidden size 24, Adam (lr $10^{-2}$), gradient clipping and the same 9-epoch budget.`,
              },
              {
                t: "code",
                lang: "python",
                title: "rnn_vs_gru_vs_lstm_vs_bigru.py — about 20 s on one CPU thread",
                code: String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)                    # single thread: bit-for-bit reproducible on CPU

# ---- Task: tag every position t with 1 if x[t+2] > x[0] (tokens are digits 0..4) ----
# The answer needs the FIRST token (long-range past) and a token two steps in the FUTURE.
V, T, AHEAD = 5, 40, 2
N_TRAIN, N_TEST = 2000, 1000

def make_data(n):
    x = torch.randint(0, V, (n, T))
    y = torch.zeros(n, T, dtype=torch.long)
    y[:, 1:T - AHEAD] = (x[:, 1 + AHEAD:] > x[:, :1]).long()
    mask = torch.zeros(n, T, dtype=torch.bool)
    mask[:, 1:T - AHEAD] = True             # positions that have a label
    return x, y, mask

x_tr, y_tr, m_tr = make_data(N_TRAIN)
x_te, y_te, m_te = make_data(N_TEST)

# A causal model can only guess from x[0]: predict 1 iff P(x[t+2] > x[0]) > 1/2.
a = torch.arange(V).float()
ceiling = torch.maximum((V - 1 - a) / V, (a + 1) / V).mean().item()
print(f"best possible accuracy without the future: {ceiling:.3f}")

class Tagger(nn.Module):
    def __init__(self, cell, bidirectional, hidden=24):
        super().__init__()
        self.emb = nn.Embedding(V, 8)
        self.rnn = cell(8, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), 2)
    def forward(self, x):
        out, _ = self.rnn(self.emb(x))      # (B, T, H) or (B, T, 2H)
        return self.head(out)               # (B, T, 2): a decision at every position

def train_and_test(cell, bidirectional, epochs=9):
    torch.manual_seed(1)
    model = Tagger(cell, bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(N_TRAIN)
        for k in range(0, N_TRAIN, 100):
            i = perm[k:k + 100]
            logits = model(x_tr[i])
            loss = nn.functional.cross_entropy(logits[m_tr[i]], y_tr[i][m_tr[i]])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        correct = model(x_te).argmax(-1) == y_te
    n_par = sum(p.numel() for p in model.rnn.parameters())
    return n_par, correct[m_te].float().mean().item()

print("model    recurrent params  test accuracy")
for name, cell, bi in [("RNN", nn.RNN, False), ("LSTM", nn.LSTM, False), ("GRU", nn.GRU, False),
                       ("BiRNN", nn.RNN, True), ("BiLSTM", nn.LSTM, True), ("BiGRU", nn.GRU, True)]:
    n_par, acc = train_and_test(cell, bi)
    print(f"{name:7s}  {n_par:16d}  {acc:13.3f}")`,
                output: `best possible accuracy without the future: 0.760
model    recurrent params  test accuracy
RNN                   816          0.758
LSTM                 3264          0.722
GRU                  2448          0.758
BiRNN                1632          0.867
BiLSTM               6528          0.856
BiGRU                4896          1.000`,
              },
              {
                t: "list",
                items: [
                  "**Unidirectional models are capped by the causal ceiling.** RNN and GRU reach 0.758, the best a model without future context can do (0.760 in expectation). No amount of training or memory can beat it, because the information is not in the input yet.",
                  "**Bidirectionality is what breaks the ceiling**, and the cell type matters once it does: BiGRU solves the task (1.000), BiRNN reaches 0.867 and BiLSTM 0.856 within the same budget.",
                  "Why the RNN is fine causally but not bidirectionally: the causal optimum only needs **one bit** about $x_0$ (is it 0 or 1?), which a vanilla RNN can hold for 40 steps. The exact comparison $x_{t+2} > x_0$ needs the full value of $x_0$ at every position, i.e. precise long-range memory in the forward direction — where the gated unit wins.",
                  "The LSTM rows had not converged after 9 epochs (180 updates); in a longer 15-epoch run with the same setup the unidirectional LSTM also reached 0.758. This is one seed and one small task: it shows that the GRU can learn faster on such problems, not that it is better in general — consistent with Chung et al. (2014), who found no consistent winner between GRU and LSTM.",
                  "Training wall-clock in our single-thread run: every model trained in under 4 s; the BiGRU took about 3.3 s, roughly twice the unidirectional GRU (1.7 s). See the speed benchmark under *Advantages and Limitations* for a controlled comparison.",
                ],
              },
              {
                t: "h",
                text: "Shapes: bidirectional, stacked and residual",
              },
              {
                t: "code",
                lang: "python",
                title: "gru_shapes.py — output and h_n layouts",
                code: String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, D, H = 4, 7, 5, 8
x = torch.randn(B, T, D)

# ---- 1. Bidirectional, single layer ----
bigru = nn.GRU(D, H, batch_first=True, bidirectional=True)
out, h_n = bigru(x)
print("bidirectional  output", tuple(out.shape), " h_n", tuple(h_n.shape))
# forward direction finishes at t = T-1, backward direction finishes at t = 0
print("  out[:, -1, :H] == h_n[0] (forward final): ", torch.allclose(out[:, -1, :H], h_n[0]))
print("  out[:, 0, H:]  == h_n[1] (backward final):", torch.allclose(out[:, 0, H:], h_n[1]))
print("  out[:, -1, H:] == h_n[1]?                 ", torch.allclose(out[:, -1, H:], h_n[1]))

# ---- 2. Stacked (3 layers) + bidirectional, dropout between layers ----
deep = nn.GRU(D, H, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
out, h_n = deep(x)
print("3-layer bi     output", tuple(out.shape), " h_n", tuple(h_n.shape))
h_n = h_n.view(3, 2, B, H)                      # (num_layers, num_directions, B, H)
sentence_vec = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)   # last layer, both directions
print("  sentence vector from last layer:", tuple(sentence_vec.shape))
for name, p in deep.named_parameters():
    if name.startswith("weight_ih"):
        print(f"  {name:22s} {tuple(p.shape)}")

# ---- 3. Residual connections between stacked layers (GNMT-style) ----
class ResidualGRUStack(nn.Module):
    def __init__(self, d, h, layers, p_drop=0.2):
        super().__init__()
        self.inp = nn.Linear(d, h)                  # project input to width h so it can be added
        self.layers = nn.ModuleList([nn.GRU(h, h, batch_first=True) for _ in range(layers)])
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        y = self.inp(x)
        for gru in self.layers:
            out, _ = gru(self.drop(y))
            y = y + out                             # residual: layer learns a correction
        return y
res = ResidualGRUStack(D, H, layers=4)
print("residual stack output", tuple(res(x).shape))`,
                output: `bidirectional  output (4, 7, 16)  h_n (2, 4, 8)
  out[:, -1, :H] == h_n[0] (forward final):  True
  out[:, 0, H:]  == h_n[1] (backward final): True
  out[:, -1, H:] == h_n[1]?                  False
3-layer bi     output (4, 7, 16)  h_n (6, 4, 8)
  sentence vector from last layer: (4, 16)
  weight_ih_l0           (24, 5)
  weight_ih_l0_reverse   (24, 5)
  weight_ih_l1           (24, 16)
  weight_ih_l1_reverse   (24, 16)
  weight_ih_l2           (24, 16)
  weight_ih_l2_reverse   (24, 16)
residual stack output (4, 7, 8)`,
              },
              {
                t: "list",
                items: [
                  "`output` is `(B, T, num_directions·H)` and contains only the **last layer**; `h_n` is `(num_layers·num_directions, B, H)` and contains the final state of **every** layer and direction. Use `h_n.view(num_layers, num_directions, B, H)` to index it safely.",
                  "`out[:, -1, H:]` is *not* the backward final state: at the last position the backward RNN has seen only one token. Its final state is at position 0.",
                  "Layers above the first in a bidirectional stack have `weight_ih` of shape `(3H, 2H)` because they read the concatenated $2H$ outputs.",
                  "`dropout=0.2` is applied between layers during training only (`model.eval()` disables it).",
                ],
              },
              {
                t: "h",
                text: "Packing variable-length sequences",
              },
              {
                t: "code",
                lang: "python",
                title: "gru_packing.py — padded vs packed batch, compared with running each sequence alone",
                code: String.raw`import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

torch.manual_seed(0)
D, H = 3, 4
gru = nn.GRU(D, H, batch_first=True, bidirectional=True)

# Three sequences of different lengths (e.g. sentences of 5, 2 and 3 tokens)
seqs = [torch.randn(5, D), torch.randn(2, D), torch.randn(3, D)]
lengths = torch.tensor([len(s) for s in seqs])
padded = pad_sequence(seqs, batch_first=True)          # (3, 5, D), zeros after each sequence ends
print("padded batch", tuple(padded.shape), "lengths", lengths.tolist())

with torch.no_grad():
    # Ground truth: run every sequence on its own, no padding at all
    truth = [gru(s.unsqueeze(0))[1][:, 0] for s in seqs]        # each (2, H): [forward final, backward final]
    truth = torch.stack(truth, dim=1)                          # (2, B, H)

    # (a) Naive: feed the padded batch directly
    _, h_naive = gru(padded)
    # (b) Packed: the GRU only sees the real steps of each sequence
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    out_packed, h_packed = gru(packed)
    out, out_lens = pad_packed_sequence(out_packed, batch_first=True)

for i in range(3):
    print(f"seq {i} (len {lengths[i]}): naive fwd err {(h_naive[0, i] - truth[0, i]).abs().max():.4f}"
          f"  naive bwd err {(h_naive[1, i] - truth[1, i]).abs().max():.4f}"
          f"  | packed err {(h_packed[:, i] - truth[:, i]).abs().max():.6f}")
print("unpacked output", tuple(out.shape), "lengths", out_lens.tolist())
print("outputs beyond each length are zero:", bool((out[1, 2:] == 0).all() and (out[2, 3:] == 0).all()))
print("packed data rows:", out_packed.data.shape[0], "= total real steps", int(lengths.sum()))`,
                output: `padded batch (3, 5, 3) lengths [5, 2, 3]
seq 0 (len 5): naive fwd err 0.0000  naive bwd err 0.0000  | packed err 0.000000
seq 1 (len 2): naive fwd err 0.3611  naive bwd err 0.0989  | packed err 0.000000
seq 2 (len 3): naive fwd err 0.0609  naive bwd err 0.0778  | packed err 0.000000
unpacked output (3, 5, 8) lengths [5, 2, 3]
outputs beyond each length are zero: True
packed data rows: 10 = total real steps 10`,
              },
              {
                t: "list",
                items: [
                  "Feeding the padded batch directly gives wrong final states for every sequence shorter than the longest one, in **both** directions: the forward RNN keeps updating on padding zeros, and the backward RNN starts on padding.",
                  "With packing the errors are zero: every sequence's `h_n` equals the result of running it alone.",
                  "`enforce_sorted=False` lets you pass sequences in any order; PyTorch sorts internally and restores the order in `h_n`. The packed data has exactly 10 rows — one per real time step — so no compute is spent on padding.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "code",
                lang: "python",
                title: "bigru_tagger_keras.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf

V, T, AHEAD = 5, 40, 2
rng = np.random.default_rng(0)
tf.random.set_seed(0)

def make_data(n):
    x = rng.integers(0, V, size=(n, T))
    y = np.zeros((n, T), dtype=np.int64)
    y[:, 1:T - AHEAD] = (x[:, 1 + AHEAD:] > x[:, :1]).astype(np.int64)
    w = np.zeros((n, T), dtype=np.float32)
    w[:, 1:T - AHEAD] = 1.0                  # sample weights: ignore positions without a label
    return x, y, w

x_tr, y_tr, w_tr = make_data(2000)
x_te, y_te, w_te = make_data(1000)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(T,), dtype="int32"),
    tf.keras.layers.Embedding(V, 8),
    # forward and backward GRU, outputs concatenated -> (B, T, 48)
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(24, return_sequences=True), merge_mode="concat"),
    tf.keras.layers.Dense(2),                # logits at every position
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-2, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              weighted_metrics=["accuracy"])
model.summary()
model.fit(x_tr, y_tr, sample_weight=w_tr, epochs=9, batch_size=100, verbose=2)
print(model.evaluate(x_te, y_te, sample_weight=w_te, verbose=0))`,
                output: "Prints the model summary — the Bidirectional layer has 4,896 parameters (2 × 2,448, the same as the PyTorch BiGRU, because Keras's default reset_after=True also keeps two bias vectors) — then the loss and accuracy for each of 9 epochs and finally the test [loss, accuracy]. The test accuracy should be far above the 0.76 causal ceiling, as in the PyTorch run.",
              },
              {
                t: "list",
                items: [
                  "`tf.keras.layers.GRU(units, return_sequences=..., return_state=..., reset_after=True)`: `return_sequences=True` returns $h_t$ for every step (needed for tagging); `return_state=True` additionally returns the final state.",
                  "Keras stores `kernel` `(d, 3h)`, `recurrent_kernel` `(h, 3h)` and `bias` `(2, 3h)` when `reset_after=True`, with gate blocks in the order **z, r, h** — different from PyTorch's r, z, n. Keras, like PyTorch, uses $z$ as the fraction kept: $h_t = z\\odot h_{t-1} + (1-z)\\odot \\tilde h_t$.",
                  "`reset_after=False` gives the original Cho et al. placement and a single bias `(3h,)`; such a layer cannot use the cuDNN kernel.",
                  "`Bidirectional(layer, merge_mode=\"concat\")` also accepts `\"sum\"`, `\"mul\"`, `\"ave\"` or `None` (return both directions separately).",
                  "Stacking: put several recurrent layers in sequence, all but the last with `return_sequences=True`. For variable lengths, use `Embedding(..., mask_zero=True)` (token 0 reserved for padding) and the mask propagates to the GRU, which then skips padded steps — Keras's counterpart of packing.",
                ],
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Variant", "Gates / state", "Parameters (per layer, one direction)", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          [String.raw`Vanilla RNN`, "none / $h_t$", String.raw`$h(h+d)+h$`, "fastest, simplest", "vanishing/exploding gradients; short memory", "teaching, very short sequences"],
          [String.raw`GRU (Cho placement)`, "update, reset / $h_t$", String.raw`$3(h(h+d)+h)$`, "25% fewer parameters than LSTM; often equally accurate", "no output gate: whole state is always exposed", "original formulation; Keras reset_after=False"],
          [String.raw`GRU (reset-after, cuDNN)`, "update, reset / $h_t$", String.raw`$3(h(h+d)+2h)$`, "one fused recurrent matmul; GPU-kernel friendly", "slightly different function; weights not interchangeable", "PyTorch nn.GRU, Keras default"],
          [String.raw`LSTM`, "forget, input, output / $h_t, c_t$", String.raw`$4(h(h+d)+h)$`, "separate memory and output; robust default", "most parameters and compute", "speech, time series, pre-2018 NLP"],
          [String.raw`Minimal gated unit (MGU)`, "one gate / $h_t$", String.raw`$2(h(h+d)+h)$`, "fewest parameters of the gated cells", "less studied; less widely available", "resource-constrained models"],
          [String.raw`Bidirectional (any cell)`, "two independent directions", "2× the cell", "every output sees past and future", "needs the full sequence; 2× cost", "tagging, encoders, offline speech"],
          [String.raw`Stacked (any cell)`, "$L$ layers", "sum over layers", "hierarchical features", "more sequential passes; harder to train without residuals", "translation encoders/decoders, speech"],
        ],
      },
      {
        t: "h",
        text: "Minimal gated unit",
      },
      {
        t: "p",
        text: String.raw`Zhou, Wu, Zhang and Zhou (2016), *Minimal Gated Unit for Recurrent Neural Networks*, go one step further and use a single gate $f_t$ in both roles: it gates the old state inside the candidate (like $r_t$) and sets the interpolation (like $z_t$):`,
      },
      {
        t: "math",
        tex: String.raw`f_t = \sigma(W_f[h_{t-1};x_t] + b_f),\quad \tilde h_t = \tanh(W_h[f_t\odot h_{t-1};\,x_t] + b_h),\quad h_t = (1-f_t)\odot h_{t-1} + f_t\odot\tilde h_t`,
      },
      {
        t: "h",
        text: "GRU vs LSTM: what the evidence says",
      },
      {
        t: "list",
        items: [
          "**Chung et al. (2014)**: on polyphonic music and speech-signal modelling, GRU and LSTM both outperformed the tanh RNN; the comparison between them depended on the dataset.",
          "**Jozefowicz, Zaremba and Sutskever (2015)**, *An Empirical Exploration of Recurrent Network Architectures*: in a large architecture search, the GRU beat the plain LSTM on several tasks, but initialising the LSTM's forget-gate bias to 1 closed most of the gap.",
          "**This lesson's experiment**: same accuracy for GRU and vanilla RNN on the causal version of the task (both at the ceiling); with bidirectional models, GRU solved the task within the budget and LSTM had not yet converged.",
          "Practical rule: try both if the difference matters; GRU when parameters or memory are tight, LSTM when you want a separate memory and output (e.g. precise counting), and check the speed claim on your hardware (see below).",
        ],
      },
      {
        t: "h",
        text: "Other variations",
      },
      {
        t: "list",
        items: [
          "**Coupled-gate LSTM**: an LSTM with $i_t = 1 - f_t$ — the GRU's coupling applied to the LSTM.",
          "**Bidirectional merge modes**: concatenation (default), sum, average or product of the two directions; summing halves the output size.",
          "**Pyramidal / strided stacks**: in speech, upper layers can halve the time resolution by concatenating adjacent outputs, reducing the cost of later layers.",
          "**Learned or carried initial states**: $h_0$ can be a learned parameter, or carried from the previous chunk for stateful processing of long streams.",
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Neural machine translation**: the GRU itself was introduced in an RNN encoder–decoder for translation (Cho et al., 2014). Bahdanau, Cho and Bengio (2015) used a **bidirectional** RNN encoder with gated units so that each source-word annotation summarised the whole sentence — the representation their attention mechanism looks at ([attention](/learn/m10/attention-intro)). Google's production NMT system (Wu et al., 2016) used a deep **stacked** LSTM with residual connections and a bidirectional bottom layer.",
          "**Sequence tagging**: part-of-speech tagging and named-entity recognition were dominated for several years by bidirectional LSTMs, often with a CRF output layer (Huang, Xu and Yu, 2015; Lample et al., 2016), because each tag depends on words on both sides ([NLP tasks](/learn/m13/nlp-tasks)).",
          "**Speech recognition**: Graves and Schmidhuber (2005) showed bidirectional LSTMs outperforming unidirectional ones on framewise phoneme classification; deep bidirectional LSTMs became standard acoustic models for offline recognition. Streaming recognisers use unidirectional layers (or limited look-ahead) because of the latency constraint.",
          "**Contextual word embeddings**: ELMo (Peters et al., 2018) built word representations from a two-layer bidirectional LSTM language model — a forward and a backward LM trained separately, so neither direction sees its own target.",
          "**Time series and sensors**: GRUs are a common choice for forecasting, anomaly detection and activity recognition on devices with limited memory; for *online* forecasting they must be unidirectional, while offline classification of a recorded window can use a bidirectional GRU.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      {
        t: "table",
        head: ["Choice", "Strengths", "Limitations"],
        rows: [
          ["GRU", "25% fewer parameters than LSTM; one state; often matches LSTM accuracy; learns long dependencies via the $(1-z)$ path.", "No output gate, so memory is always exposed; fewer FLOPs do not guarantee faster training (see benchmark); still sequential over time."],
          ["Bidirectional", "Every position sees the whole sequence; large accuracy gains when labels depend on future context.", "Unusable for causal/online tasks; doubles parameters and compute; final-state handling is easy to get wrong."],
          ["Stacked", "More expressive, hierarchical features; standard in strong recurrent systems.", "Each layer is another sequential pass; deeper stacks need dropout and residual connections to train well."],
          ["Packing", "Correct results for variable-length batches; no compute wasted on padding.", "Extra bookkeeping; must pass correct lengths; a few operations are not available on packed data."],
        ],
      },
      {
        t: "h",
        text: "Speed is a measurement, not a parameter count",
      },
      {
        t: "code",
        lang: "python",
        title: "rnn_speed.py — forward + backward time per batch (timings vary between runs and machines)",
        code: String.raw`import time
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)
B, T, D, H = 32, 100, 64, 128
x = torch.randn(B, T, D)

def ms_per_batch(layer, reps=10):
    for _ in range(2):                                  # warm-up
        out, _ = layer(x); out.sum().backward()
    t0 = time.perf_counter()
    for _ in range(reps):
        out, _ = layer(x)
        out.sum().backward()                            # forward + backward, like a training step
    return 1000 * (time.perf_counter() - t0) / reps

print(f"batch {B}, length {T}, input {D}, hidden {H}, 1 CPU thread")
base = None
for name, cls, kw in [("RNN", nn.RNN, {}), ("GRU", nn.GRU, {}), ("LSTM", nn.LSTM, {}),
                      ("BiGRU", nn.GRU, {"bidirectional": True}), ("GRU x2 layers", nn.GRU, {"num_layers": 2})]:
    layer = cls(D, H, batch_first=True, **kw)
    n = sum(p.numel() for p in layer.parameters())
    ms = ms_per_batch(layer)
    base = base or ms
    print(f"{name:14s} params {n:7d}   {ms:6.1f} ms/batch   ({ms / base:.1f}x RNN)")`,
        output: `batch 32, length 100, input 64, hidden 128, 1 CPU thread
RNN            params   24832     13.0 ms/batch   (1.0x RNN)
GRU            params   74496     30.9 ms/batch   (2.4x RNN)
LSTM           params   99328     28.1 ms/batch   (2.2x RNN)
BiGRU          params  148992     65.6 ms/batch   (5.0x RNN)
GRU x2 layers  params  173568     69.7 ms/batch   (5.3x RNN)`,
      },
      {
        t: "p",
        text: "In three runs on our CPU (PyTorch 2, single thread) the GRU took 29–34 ms per batch and the LSTM 25–28 ms — the GRU was consistently **slower**, despite having 25% fewer parameters. The vanilla RNN was 2.2–3.4× faster than either. A bidirectional layer and a 2-layer stack each cost roughly twice a single GRU layer, as expected. Wall-clock speed depends on how well each cell's kernel is optimised (fused matrix multiplies, the extra element-wise work of reset-after, memory traffic), and on GPU with cuDNN the ranking can differ. If speed matters, benchmark your configuration rather than assuming the smaller cell is faster.",
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Mixing the two $z$ conventions.** Saying \"$z \\approx 1$ means remember\" is true in PyTorch/Keras and false in Chung et al.'s notation. Fix: always write out the interpolation equation you mean; to convert, negate $W_z$ and $b_z$.",
          "**Porting weights between reset placements or libraries.** PyTorch (reset after, blocks r, z, n) and Keras (blocks z, r, h; reset_after configurable) differ in both order and formula. Fix: map blocks by name and test against a reference, as in the NumPy section.",
          "**Using `output[:, -1]` as the summary of a bidirectional RNN.** Its backward half has seen only the last token. Fix: use `h_n` (or `torch.cat([h_n[-2], h_n[-1]], -1)` for the last layer), or `output[:, -1, :H]` together with `output[:, 0, H:]`.",
          "**Indexing `h_n` of a deep bidirectional RNN as `h_n[-1]` and calling it \"the last layer\".** It is the last layer's *backward* direction only. Fix: `h_n.view(num_layers, 2, B, H)[-1]` gives both directions of the last layer.",
          "**Using a bidirectional model for causal prediction.** Training a BiRNN for next-step prediction lets the backward pass read the target, giving excellent training loss and a model that cannot be deployed. Fix: unidirectional layers for anything online or autoregressive.",
          "**Running padded batches without packing (or masking).** Final states are computed after padding steps; in the packing example the error reached 0.36. Fix: `pack_padded_sequence(..., enforce_sorted=False)`, or gather the last real index for unidirectional models.",
          "**Expecting `dropout=` to do something in a 1-layer RNN.** PyTorch applies it only between stacked layers. Fix: add `nn.Dropout` on the RNN's inputs/outputs explicitly.",
          "**Assuming GRU is always faster.** It has fewer parameters, but in our CPU benchmark it was slower than the LSTM. Fix: measure.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What are the two gates of a GRU and what does each do?", a: "The **update gate** $z_t$ decides, per hidden unit, how much of the state is replaced by the candidate: $h_t = (1-z_t)\\odot h_{t-1} + z_t \\odot \\tilde h_t$ (in Chung et al.'s convention). The **reset gate** $r_t$ decides how much of the previous state is used when computing the candidate $\\tilde h_t = \\tanh(W_h[r_t\\odot h_{t-1}; x_t] + b_h)$. Both are sigmoid layers reading $[h_{t-1}; x_t]$." },
      { t: "qa", level: "beginner", q: "How does a GRU differ from an LSTM?", a: "The GRU has one state ($h_t$) instead of two ($h_t, c_t$), two gates instead of three, and no output gate, so its whole memory is always exposed. Its update gate couples forgetting and writing (keep $1-z$, write $z$), where the LSTM has independent forget and input gates. It has three weight blocks instead of four, so 25% fewer parameters for the same sizes." },
      { t: "qa", level: "beginner", q: "What is a bidirectional RNN and when can't you use it?", a: "Two RNNs with separate weights, one reading the sequence forwards and one backwards; their hidden states are concatenated at each position so every output depends on the whole sequence. It cannot be used when the future is not available at prediction time: streaming/online prediction, autoregressive generation and real-time forecasting." },
      { t: "qa", level: "intermediate", q: "Why does a GRU suffer less from vanishing gradients than a vanilla RNN?", a: "$\\partial h_t/\\partial h_{t-1} = \\operatorname{diag}(1-z_t) + \\operatorname{diag}(z_t)\\,\\partial\\tilde h_t/\\partial h_{t-1} + \\operatorname{diag}(\\tilde h_t - h_{t-1})\\,\\partial z_t/\\partial h_{t-1}$. The first term has no weight matrix and no activation derivative. When $z_t \\approx 0$ it is close to the identity and the other terms are small, so gradients pass through many steps scaled only by $\\prod(1-z_\\tau)$, a learned per-unit decay. A vanilla RNN's Jacobian always contains $W_{hh}$ and $\\tanh'$." },
      { t: "qa", level: "intermediate", q: "How many parameters does torch.nn.GRU(input_size=64, hidden_size=128) have? And with bidirectional=True, num_layers=2?", a: "One direction, one layer: $3(h(h+d)+2h) = 3(128\\cdot192 + 256) = 74{,}496$. Bidirectional doubles the first layer: $148{,}992$. The second layer reads $2h = 256$ inputs: $3(128\\cdot 384 + 256) = 148{,}224$ per direction, $296{,}448$ for both. Total: $445{,}440$." },
      { t: "qa", level: "intermediate", q: "Explain the output and h_n shapes of nn.GRU(D, H, num_layers=L, bidirectional=True, batch_first=True).", a: "`output`: `(B, T, 2H)` — last layer only, forward and backward states concatenated at each position. `h_n`: `(2L, B, H)` — final state of each layer and direction, ordered layer 0 fwd, layer 0 bwd, layer 1 fwd, …; reshape with `view(L, 2, B, H)`. The forward final equals `output[:, -1, :H]`, the backward final equals `output[:, 0, H:]` (for unpadded input)." },
      { t: "qa", level: "intermediate", q: "Why do we need pack_padded_sequence?", a: "Padding makes all sequences the same length, but an RNN would keep updating its state on padding steps, so final states (and, for a backward RNN, every state) are corrupted. Packing tells the RNN the real length of each sequence so it processes only real steps; `h_n` then contains each sequence's true final state and no compute is spent on padding. For a unidirectional model you could instead gather the output at index length−1, but for a bidirectional model packing (or masking) is required." },
      { t: "qa", level: "advanced", q: "PyTorch computes n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)). How does this differ from Cho et al.'s GRU and why was it adopted?", a: "Cho et al. apply the reset gate to the state before the matrix multiply, $U_h(r\\odot h)$; PyTorch/cuDNN apply it after, $r\\odot(U_h h + b_{hn})$. They are different functions (in this lesson's worked example $h_t$ differs in the second decimal) with different parameter counts ($b_{hn}$ is not redundant). The reset-after form lets all three recurrent products $U_r h, U_z h, U_h h$ be computed in one fused $3h\\times h$ matrix multiply before $r$ is known, which is much more efficient on GPUs; empirically both work similarly well." },
      { t: "qa", level: "advanced", q: "A colleague trains a bidirectional GRU language model and reports a perplexity far better than any published unidirectional model. What is wrong?", a: "The backward direction at position $t$ has read $x_{t+1}$ — the token being predicted — so the model is copying the answer, not modelling language. Bidirectional encoders cannot be trained with a plain next-token objective. ELMo avoids this by training a forward LM and a backward LM separately (each predicting in its own direction) and only combining their representations afterwards; BERT-style models use masking instead." },
      { t: "qa", level: "advanced", q: "When would you add residual connections between stacked RNN layers, and what constraint do they impose?", a: "When stacking more than a few layers: without them, deep recurrent stacks are hard to optimise, and GNMT (Wu et al., 2016) reported needing residual connections to train its 8-layer encoder and decoder. A residual $h^{(\\ell)} = h^{(\\ell-1)} + \\mathrm{RNN}(h^{(\\ell-1)})$ requires equal widths across layers, so the input usually needs a linear projection to width $h$ (and bidirectional layers must keep $2h$ consistent). Dropout is typically applied to the layer input before the RNN, not to the residual path." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "In the convention $h_t = (1-z_t)\\odot h_{t-1} + z_t\\odot\\tilde h_t$, what happens to a unit with $z_t \\approx 0$?",
            options: ["It is reset to zero", "It copies its previous value forward almost unchanged", "It outputs the candidate", "Its gradient vanishes"],
            answer: 1,
            explain: "With $z_t = 0$, $h_t = h_{t-1}$ for that unit, and $\\partial h_t/\\partial h_{t-1}$ is close to 1 along it.",
          },
          {
            kind: "numeric",
            q: "How many parameters does a GRU layer with input size 10 and hidden size 20 have, using the single-bias formula $3(h(h+d)+h)$?",
            answer: 1860,
            tolerance: 0,
            explain: "$3(20\\cdot 30 + 20) = 3 \\cdot 620 = 1860$. PyTorch reports 1920 because of its second bias vector ($3h = 60$).",
          },
          {
            q: "Where does torch.nn.GRU apply the reset gate?",
            options: ["To $h_{t-1}$ before the recurrent matrix multiply", "To $(W_{hn}h_{t-1} + b_{hn})$, after the multiply", "To the input $x_t$", "To the output $h_t$"],
            answer: 1,
            explain: "$n_t = \\tanh(W_{in}x_t + b_{in} + r_t\\odot(W_{hn}h_{t-1}+b_{hn}))$ — the cuDNN-compatible placement.",
          },
          {
            kind: "numeric",
            q: "A unit keeps $1 - z = 0.95$ of its value at each step and writes nothing useful. What fraction of its original value is left after 50 steps? (3 decimals)",
            answer: 0.077,
            tolerance: 0.002,
            explain: "$0.95^{50} = 0.0769$.",
          },
          {
            q: "For nn.GRU(8, 16, num_layers=3, bidirectional=True, batch_first=True) and input (4, 10, 8), what is the shape of h_n?",
            options: ["(4, 10, 32)", "(3, 4, 32)", "(6, 4, 16)", "(4, 6, 16)"],
            answer: 2,
            explain: "$h_n$ is (num_layers × num_directions, B, H) = (6, 4, 16), regardless of batch_first.",
          },
          {
            q: "Which task can a bidirectional GRU legitimately be used for?",
            options: ["Next-word prediction for an autocomplete keyboard", "Named-entity tagging of complete sentences", "Real-time forecasting of tomorrow's demand from today's data", "Streaming speech recognition with no delay"],
            answer: 1,
            explain: "Only tagging has the whole sequence available at prediction time; the other three are causal.",
          },
          {
            kind: "numeric",
            q: "In this lesson's tagging task (digits 0–4, label $x_{t+2} > x_0$), what is the best accuracy a causal model can achieve? (3 decimals)",
            answer: 0.76,
            tolerance: 0.001,
            explain: "For $x_0 = a$, $P(x > a) = (4-a)/5$; the best guess is correct with probability $\\max((4-a)/5, (a+1)/5)$ = 0.8, 0.6, 0.6, 0.8, 1.0 for $a = 0..4$, averaging 0.76.",
          },
          {
            q: "In a 2-layer bidirectional GRU with hidden size H, what is the shape of weight_ih_l1?",
            options: ["(3H, D)", "(3H, H)", "(3H, 2H)", "(6H, 2H)"],
            answer: 2,
            explain: "Layer 1 reads the concatenated forward/backward outputs of layer 0, of size 2H; the three gate blocks give 3H rows.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Numerical: one step of a scalar GRU",
        prompt: "A 1-unit GRU (convention $h_t = (1-z)h_{t-1} + z\\tilde h_t$, reset before the multiply) receives $x_t = 2$ with $h_{t-1} = 0.5$. Weights $(w_h, w_x, b)$: reset $(1, -0.5, 0)$, candidate $(2, 0.25, 0)$ acting on $(r h_{t-1}, x_t)$, update $(0.5, -1, 0.5)$. Compute $r_t, \\tilde h_t, z_t, h_t$ to 4 decimals, and the direct-path factor $1 - z_t$.",
        hint: "Pre-activation = $w_h \\cdot (\\text{state}) + w_x x_t + b$; the candidate uses $r_t h_{t-1}$ as its state.",
        solution: [
          {
            t: "list",
            items: [
              "Reset: $1(0.5) - 0.5(2) = -0.5$, $r_t = \\sigma(-0.5) = 0.3775$, so $r_t h_{t-1} = 0.1888$.",
              "Candidate: $2(0.1888) + 0.25(2) = 0.8775$, $\\tilde h_t = \\tanh(0.8775) = 0.7052$.",
              "Update: $0.5(0.5) - 1(2) + 0.5 = -1.25$, $z_t = \\sigma(-1.25) = 0.2227$.",
              "New state: $h_t = 0.7773 \\times 0.5 + 0.2227 \\times 0.7052 = 0.3886 + 0.1570 = 0.5457$.",
              "Direct-path factor $1 - z_t = 0.7773$: if this held for 20 steps, $0.7773^{20} \\approx 0.0065$ of a signal would survive along that path alone.",
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Coding: the last real output of a padded batch",
        prompt: "For a unidirectional `nn.GRU` with `batch_first=True` and a padded batch of 4 sequences of lengths 6, 3, 1, 4, write code that extracts each sequence's output at its last **real** step from the padded output tensor (without packing). Verify it against `h_n` from a packed run, and show how wrong the common `output[:, -1]` is.",
        hint: "Build an index tensor of shape (B, 1, H) holding length−1 and use `gather` along dimension 1.",
        solution: [
          {
            t: "code",
            lang: "python",
            code: String.raw`import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

torch.manual_seed(0)
gru = nn.GRU(input_size=3, hidden_size=4, batch_first=True)     # unidirectional
seqs = [torch.randn(n, 3) for n in (6, 3, 1, 4)]
lengths = torch.tensor([len(s) for s in seqs])
padded = pad_sequence(seqs, batch_first=True)                   # (4, 6, 3)

with torch.no_grad():
    out, _ = gru(padded)                                        # (4, 6, 4), padded positions included
    # index of the last REAL step of each sequence, broadcast to (B, 1, H) for gather
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
    last = out.gather(1, idx).squeeze(1)                        # (4, 4)

    _, h_n = gru(pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False))
    wrong = out[:, -1]                                          # the common bug

print("gathered last step == packed h_n:", torch.allclose(last, h_n[0], atol=1e-6))
print("out[:, -1] == packed h_n:        ", torch.allclose(wrong, h_n[0], atol=1e-6))
print("per-sequence error of out[:, -1]:", [round(v, 4) for v in (wrong - h_n[0]).abs().amax(1).tolist()])`,
            output: `gathered last step == packed h_n: True
out[:, -1] == packed h_n:         False
per-sequence error of out[:, -1]: [0.0, 0.0627, 0.2416, 0.2387]`,
          },
          {
            t: "p",
            text: "Only the longest sequence (length 6) is correct with `output[:, -1]`; the others have errors of 0.06–0.24 because the GRU kept updating on padding. Gathering at `length − 1` works for unidirectional models; for bidirectional ones the backward direction is still wrong on padded input, so use packing.",
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "GRU (Cho et al., 2014): one state $h_t$, two gates. Reset $r_t$ controls how much history the candidate sees; update $z_t$ interpolates between old state and candidate.",
          "Conventions: this lesson and Chung et al. use $h_t = (1-z_t)\\odot h_{t-1} + z_t\\odot\\tilde h_t$; Cho et al., PyTorch and Keras swap the roles ($z$ = fraction kept). Convert by negating $W_z, b_z$.",
          "PyTorch/cuDNN apply the reset gate after the recurrent multiply: $n_t = \\tanh(W_{in}x + b_{in} + r\\odot(W_{hn}h + b_{hn}))$ — a different function from Cho's $U_h(r\\odot h)$.",
          "Jacobian: $\\partial h_t/\\partial h_{t-1} = \\operatorname{diag}(1-z_t) + \\dots$ — an identity-like path with learned decay $\\prod(1-z_\\tau)$; bias the gate towards keeping for long memory.",
          "Parameters: $3(h(h+d)+h)$; PyTorch/Keras-reset-after add $3h$. LSTM: $4(\\cdot)$. Fewer parameters did not mean faster in our CPU benchmark.",
          "Bidirectional RNNs (Schuster & Paliwal, 1997): concatenate forward and backward states; output $(B, T, 2H)$, $h_n$ $(2L, B, H)$; backward final state is at position 0. Not for causal tasks.",
          "Stacked RNNs: layer $\\ell$ reads layer $\\ell-1$'s sequence; dropout between layers; residual connections for deep stacks (GNMT).",
          "Variable lengths: pack with `pack_padded_sequence(..., enforce_sorted=False)`; never take `output[:, -1]` of a padded batch.",
          "Experiment: on a task needing future context, unidirectional models stopped at the 0.760 causal ceiling; BiGRU reached 1.000, BiRNN 0.867, BiLSTM 0.856 within the same budget.",
        ],
      },
      {
        t: "math",
        tex: String.raw`h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tanh\!\big(W_h[r_t\odot h_{t-1};x_t]+b_h\big),\qquad y_t^{\text{bi}} = [\overrightarrow{h}_t;\overleftarrow{h}_t]`,
        caption: "The GRU update and the bidirectional output in one line.",
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[From Seq2Seq bottlenecks to attention](/learn/m10/attention-intro) — bidirectional GRU encoders produce one annotation per source word; attention learns which of them to read at each decoding step, removing the fixed-size bottleneck.",
          "[Classification, NER, QA, summarization, translation](/learn/m13/nlp-tasks) — the tasks where bidirectional and stacked recurrent models were the standard before Transformers, and how they are framed.",
          "[The Transformer architecture](/learn/m11/transformer-architecture) — replaces recurrence with attention so all positions are processed in parallel; its encoder is \"bidirectional\" by construction, its decoder causal by masking.",
        ],
      },
    ],
  },
};

export default lesson;
