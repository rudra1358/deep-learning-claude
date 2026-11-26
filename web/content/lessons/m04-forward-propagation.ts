import type { Lesson } from "@/lib/types";

const FP_NUMPY = String.raw`import numpy as np

# ----- activation functions -----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0.0, z)

ACTIVATIONS = {"sigmoid": sigmoid, "relu": relu, "linear": lambda z: z}


class MLP:
    """A multilayer perceptron that only does forward propagation.

    layer_sizes = [n_inputs, n_hidden_1, ..., n_outputs]
    W[l] has shape (n_out, n_in) so that z = a_prev @ W.T + b works for a batch.
    """

    def __init__(self, layer_sizes, activations, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            std = np.sqrt(2.0 / (n_in + n_out))          # Xavier/Glorot scale
            self.W.append(rng.normal(0.0, std, size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))
        self.acts = [ACTIVATIONS[name] for name in activations]

    def forward(self, X, verbose=False):
        """X: (batch, n_inputs). Returns the output and every intermediate z and a."""
        a = X
        cache = {"a0": X}
        for l, (W, b, f) in enumerate(zip(self.W, self.b, self.acts), start=1):
            z = a @ W.T + b          # (batch, n_in) @ (n_in, n_out) + (n_out,)  -> broadcasting
            a = f(z)
            cache[f"z{l}"], cache[f"a{l}"] = z, a
            if verbose:
                print(f"layer {l}: W{l} {W.shape}, z{l} {z.shape}, a{l} {a.shape}")
        return a, cache


# ---- 1) the worked example from this lesson, one sample ----
net = MLP([2, 2, 1], ["sigmoid", "sigmoid"])
net.W = [np.array([[0.2, -0.4], [0.7, 0.1]]), np.array([[0.6, -0.3]])]
net.b = [np.array([0.1, -0.1]), np.array([0.05])]

x = np.array([[1.0, 0.5]])                    # a batch containing one example
y_hat, cache = net.forward(x)
np.set_printoptions(precision=6, suppress=True)
print("z1 =", cache["z1"], " a1 =", cache["a1"])
print("z2 =", cache["z2"], " a2 =", cache["a2"])
print("loss (y=1):", ((y_hat - 1.0) ** 2).item())

# ---- 2) a batch of three examples processed in ONE matrix product per layer ----
X = np.array([[1.0, 0.5], [0.0, 1.0], [-1.0, 2.0]])
print("\nbatch predictions:", net.forward(X)[0].ravel())

# ---- 3) a deeper random network: watch the shapes ----
deep = MLP([4, 8, 8, 3], ["relu", "relu", "linear"], seed=42)
X4 = np.random.default_rng(1).normal(size=(5, 4))   # 5 samples, 4 features
out, _ = deep.forward(X4, verbose=True)
print("output shape:", out.shape)`;

const FP_NUMPY_OUT = String.raw`z1 = [[0.1  0.65]]  a1 = [[0.524979 0.65701 ]]
z2 = [[0.167884]]  a2 = [[0.541873]]
loss (y=1): 0.20988054026252273

batch predictions: [0.541873 0.538756 0.529248]
layer 1: W1 (8, 4), z1 (5, 8), a1 (5, 8)
layer 2: W2 (8, 8), z2 (5, 8), a2 (5, 8)
layer 3: W3 (3, 8), z3 (5, 3), a3 (5, 3)
output shape: (5, 3)`;

const FP_TORCH = String.raw`import torch
import torch.nn as nn

torch.set_printoptions(precision=6)

# Same 2-2-1 network as the worked example
model = nn.Sequential(
    nn.Linear(2, 2),   # stores weight (2, 2) and bias (2,) — computes x @ W.T + b
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid(),
)

with torch.no_grad():                      # copying values in is not part of training
    model[0].weight.copy_(torch.tensor([[0.2, -0.4], [0.7, 0.1]]))
    model[0].bias.copy_(torch.tensor([0.1, -0.1]))
    model[2].weight.copy_(torch.tensor([[0.6, -0.3]]))
    model[2].bias.copy_(torch.tensor([0.05]))

X = torch.tensor([[1.0, 0.5], [0.0, 1.0], [-1.0, 2.0]])

# Forward hooks let us print every intermediate activation
for name, layer in model.named_children():
    layer.register_forward_hook(lambda m, inp, out, name=name: print(f"after layer {name} ({m.__class__.__name__}):", out.detach().squeeze()))

with torch.no_grad():                      # inference only: no graph needed
    y_hat = model(X)
print("predictions:", y_hat.squeeze())`;

const FP_TORCH_OUT = String.raw`after layer 0 (Linear): tensor([[ 0.100000,  0.650000],
        [-0.300000,  0.000000],
        [-0.900000, -0.600000]])
after layer 1 (Sigmoid): tensor([[0.524979, 0.657010],
        [0.425557, 0.500000],
        [0.289050, 0.354344]])
after layer 2 (Linear): tensor([0.167884, 0.155334, 0.117127])
after layer 3 (Sigmoid): tensor([0.541873, 0.538756, 0.529248])
predictions: tensor([0.541873, 0.538756, 0.529248])`;

const FP_KERAS = String.raw`import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation="sigmoid"),   # kernel shape (2, 2) = W.T in our notation
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Keras stores the kernel as (n_in, n_out), i.e. the TRANSPOSE of our W (n_out, n_in)
model.layers[0].set_weights([np.array([[0.2, -0.4], [0.7, 0.1]]).T, np.array([0.1, -0.1])])
model.layers[1].set_weights([np.array([[0.6, -0.3]]).T, np.array([0.05])])

X = np.array([[1.0, 0.5], [0.0, 1.0], [-1.0, 2.0]])
print(model(X).numpy().ravel())   # forward propagation`;

const lesson: Lesson = {
  id: "m04/forward-propagation",
  title: "Forward Propagation",
  summary: "How a neural network turns an input into a prediction: a weighted sum and an activation, layer after layer. First by hand, then as one matrix product for a whole batch.",
  minutes: 45,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices", "m03/artificial-neuron", "m03/activation-functions", "m03/multilayer-networks"],
  next: ["m04/backpropagation"],
  terms: ["forward-propagation", "neuron", "activation", "matrix-multiplication", "loss"],
  sections: {
    intro: [
      { t: "p", text: "**Forward propagation** (the *forward pass*) is how a neural network turns an input into an output. Data goes in, each layer transforms the previous layer's output, and the last layer gives the prediction. Every use of a trained network — labelling a photo, translating a sentence — is one forward pass." },
      { t: "p", text: "It is also the first half of training. A network learns by (1) predicting, (2) measuring how wrong it is with a **loss function** (a number that is big when the prediction is bad), and (3) changing each weight to shrink that loss. Steps (1)–(2) are the forward pass. Step (3) is [backpropagation](/learn/m04/backpropagation), which reuses the in-between numbers the forward pass computed." },
      { t: "p", text: "Why chain layers? One neuron can only draw a straight boundary. Stack layers — each a linear step plus a non-linear activation — and the network can represent very complex functions. The forward pass works out that chain: $f(x) = f_L(\\dots f_2(f_1(x)))$." },
      { t: "callout", kind: "key", text: "Forward propagation = start with **a = x**, then repeat **z = Wa + b** and **a = f(z)** for each layer until you reach the output." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Vectors, matrices and matrix multiplication](/learn/m02/vectors-matrices) — every layer is a matrix product, so you need the shape rule $(m\\times n)(n\\times p) = (m\\times p)$.",
        "[The artificial neuron](/learn/m03/artificial-neuron) — weights, bias and the weighted sum for one neuron.",
        "[Activation functions](/learn/m03/activation-functions) — sigmoid, tanh, ReLU, and why we need non-linearity.",
        "[Multi-layer networks](/learn/m03/multilayer-networks) — what input, hidden and output layers are.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "An assembly line of committees", text: "Picture a factory where a product moves through a row of rooms. In each room, a team of inspectors reads **every report from the previous room**. Each inspector writes one score: a weighted opinion (some reports matter more, some count against), plus a personal bias, then squashed by a rule like \"at most 1, never below 0\". The next room sees only these scores, never the product itself. The last room's scores are the network's answer." },
      { t: "p", text: "Each inspector is a neuron: weights say how much each report matters, the bias is a default leaning, and the activation turns the raw opinion into a report. Early layers learn simple things (an edge, a common word pair); later layers combine them into judgements like \"this is a cat\"." },
      { t: "p", text: "Information only moves **forward** — room 3 never sends anything back to room 2. That is why it is called a *feedforward* network." },
    ],
    mechanism: [
      { t: "p", text: "We use one small network for the whole lesson: **2 inputs → 2 hidden neurons (sigmoid) → 1 output neuron (sigmoid)**. You can build exactly this in the visualizer below." },
      { t: "steps", items: [
        { title: "Set the input activations", text: "The input layer computes nothing. Its \"activations\" are just the input features.", tex: String.raw`a^{[0]} = x` },
        { title: "Compute the weighted sum for every neuron in layer 1", text: "Each hidden neuron multiplies every input by its own weight, adds them up, and adds its bias. This number $z$ is the **pre-activation** (also called the logit).", tex: String.raw`z^{[1]}_j = \sum_k W^{[1]}_{jk}\, a^{[0]}_k + b^{[1]}_j` },
        { title: "Apply the activation function", text: "Each neuron passes $z$ through its activation function. The result is the neuron's **activation** — its output.", tex: String.raw`a^{[1]}_j = f^{[1]}\!\left(z^{[1]}_j\right)` },
        { title: "Repeat for every following layer", text: "Layer 2 does the same thing, but its inputs are layer 1's activations, not the raw input." },
        { title: "Read off the prediction", text: "The last layer's activations are the output $\\hat y$. The output activation depends on the task: sigmoid for one probability, softmax for class probabilities, identity (no activation) for regression." },
        { title: "Compute the loss (during training)", text: "Compare $\\hat y$ with the true answer $y$ using a loss, e.g. squared error $(\\hat y - y)^2$. Keep every $z$ and $a$ in memory — backpropagation needs them." },
      ] },
      { t: "callout", kind: "note", text: "Every neuron has two values: $z$ (before the activation) and $a$ (after it). Keep them apart in your head and your code — backpropagation measures the activation's slope at $z$." },
    ],
    math: [
      { t: "h", text: "Notation" },
      { t: "table", head: ["Symbol", "Meaning", "Shape"], rows: [
        ["$L$", "number of layers with weights (the input layer is not counted)", "scalar"],
        ["$n_l$", "number of neurons in layer $l$ ($n_0$ = number of input features)", "scalar"],
        ["$W^{[l]}$", "weight matrix of layer $l$; $W^{[l]}_{jk}$ connects neuron $k$ of layer $l-1$ to neuron $j$ of layer $l$", "$n_l \\times n_{l-1}$"],
        ["$b^{[l]}$", "bias vector of layer $l$", "$n_l$"],
        ["$z^{[l]}$", "pre-activations of layer $l$", "$n_l$"],
        ["$a^{[l]}$", "activations (outputs) of layer $l$; $a^{[0]} = x$", "$n_l$"],
        ["$f^{[l]}$", "activation function of layer $l$, applied to each number separately", "—"],
      ] },
      { t: "h", text: "Vector form for one example" },
      { t: "math", tex: String.raw`z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad a^{[l]} = f^{[l]}\!\left(z^{[l]}\right), \qquad l = 1,\dots,L`, caption: "The whole forward pass. Row $j$ of $W^{[l]}$ holds neuron $j$'s incoming weights, so one matrix–vector product gives every neuron's weighted sum at once." },
      { t: "p", text: "Written as one big function, you can see the layers nested inside each other:" },
      { t: "math", tex: String.raw`\hat y = f^{[L]}\Big(W^{[L]} f^{[L-1]}\big(\cdots f^{[1]}(W^{[1]}x + b^{[1]})\cdots\big) + b^{[L]}\Big)` },
      { t: "h", text: "Worked example — every number" },
      { t: "p", text: "Input $x = (1.0,\\ 0.5)$, target $y = 1$, sigmoid $\\sigma(z) = 1/(1+e^{-z})$ in both layers, and these parameters:" },
      { t: "math", tex: String.raw`W^{[1]} = \begin{pmatrix} 0.2 & -0.4 \\ 0.7 & 0.1 \end{pmatrix},\quad b^{[1]} = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix},\quad W^{[2]} = \begin{pmatrix} 0.6 & -0.3 \end{pmatrix},\quad b^{[2]} = 0.05` },
      { t: "steps", items: [
        { title: "Hidden neuron 1", tex: String.raw`z^{[1]}_1 = 0.2(1.0) + (-0.4)(0.5) + 0.1 = 0.1, \qquad a^{[1]}_1 = \sigma(0.1) = 0.524979` },
        { title: "Hidden neuron 2", tex: String.raw`z^{[1]}_2 = 0.7(1.0) + 0.1(0.5) - 0.1 = 0.65, \qquad a^{[1]}_2 = \sigma(0.65) = 0.657010` },
        { title: "Output neuron", tex: String.raw`z^{[2]} = 0.6(0.524979) - 0.3(0.657010) + 0.05 = 0.167884, \qquad \hat y = a^{[2]} = \sigma(0.167884) = 0.541873` },
        { title: "Loss (squared error)", tex: String.raw`\mathcal{L} = (\hat y - y)^2 = (0.541873 - 1)^2 = 0.209881` },
      ] },
      { t: "p", text: "The network says 0.54 when the answer is 1 — not good yet. [Backpropagation](/learn/m04/backpropagation) picks up this exact example and works out how each of the 9 parameters should change." },
      { t: "h", text: "Batch (matrix) form" },
      { t: "p", text: "In practice we feed a **batch** of $N$ examples at once. Put each example in one row of a matrix $X \\in \\mathbb{R}^{N\\times n_0}$. Then:" },
      { t: "math", tex: String.raw`Z^{[l]} = A^{[l-1]}\,W^{[l]\top} + \mathbf{1}\,b^{[l]\top}, \qquad A^{[l]} = f^{[l]}(Z^{[l]}), \qquad A^{[0]} = X`, caption: "Shapes: $(N\\times n_{l-1})(n_{l-1}\\times n_l) = N\\times n_l$. The same bias row is added to every example." },
      { t: "p", text: "One matrix product per layer now computes every neuron for every example — exactly what GPUs are built for. For the batch $X = \\begin{pmatrix}1 & 0.5\\\\ 0 & 1\\\\ -1 & 2\\end{pmatrix}$ our network gives $Z^{[1]} = \\begin{pmatrix}0.1 & 0.65\\\\ -0.3 & 0\\\\ -0.9 & -0.6\\end{pmatrix}$ and predictions $(0.541873,\\ 0.538756,\\ 0.529248)$. The first row matches the hand calculation exactly." },
      { t: "h", text: "Cost of a forward pass" },
      { t: "p", text: "Layer $l$ does $n_l n_{l-1}$ multiplications and as many additions per example. So one forward pass costs $\\sum_l n_l n_{l-1}$ multiply–adds per example — one per weight. Handy rule of thumb: **forward FLOPs ≈ 2 × parameters × batch size**." },
    ],
    visual: [
      { t: "p", text: "The visualizer starts with random weights. To rebuild the worked example, click connections and neurons to set the weights and biases above, set $x = (1, 0.5)$ and $y = 1$, then press **Step** repeatedly." },
      { t: "sim", id: "network", caption: "Gold dots carry values forward along the connections in use; the narration shows the weighted sum with the real numbers." },
      { t: "list", items: [
        "Watch the order: every neuron in layer 1 is finished before layer 2 starts, because layer 2 needs all of layer 1's outputs.",
        "Move an input slider: **every** later value changes. Change one weight into the output neuron: only the output changes.",
        "Switch the hidden activation to ReLU and give a hidden neuron a big negative bias. Its activation becomes exactly 0, so it adds nothing to the next layer (a \"dead\" neuron for this input).",
        "Set every activation to **Linear**. The output becomes a plain weighted sum of the inputs — without non-linearity, extra layers add no power.",
      ] },
    ],
    scratch: [
      { t: "p", text: "A forward-only MLP. Weights are stored as $(n_{out}\\times n_{in})$ to match the maths; with one example per row we compute `a @ W.T + b`. Part 1 repeats the worked example, part 2 batches, part 3 prints a deeper network's shapes." },
      { t: "code", lang: "python", title: "forward_propagation.py", code: FP_NUMPY, output: FP_NUMPY_OUT },
      { t: "p", text: "The important pieces:" },
      { t: "list", items: [
        "`__init__` draws weights with standard deviation $\\sqrt{2/(n_{in}+n_{out})}$ (Xavier initialization, explained in [Weight Initialization](/learn/m07/initialization)). The 2 comes from that derivation, not from tuning. Biases start at zero.",
        "`forward` loops over the layers. `a @ W.T + b` is the batch formula $A W^\\top + b$; NumPy **broadcasting** adds the bias vector to every row for you.",
        "`cache` keeps every $z$ and $a$. Pure prediction could drop them; training needs them for the backward pass.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "`nn.Linear(in, out)` stores `weight` with shape `(out, in)` — the same as our $W$ — and computes `x @ weight.T + bias`. Forward hooks print each layer's output so you can check it against the hand calculation." },
          { t: "code", lang: "python", title: "forward_pytorch.py", code: FP_TORCH, output: FP_TORCH_OUT },
          { t: "callout", kind: "tip", text: "`torch.no_grad()` tells PyTorch not to record operations for backpropagation. Use it (or `torch.inference_mode()`) when you only make predictions: it saves memory and time." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "forward_keras.py (not executed in CI)", code: FP_KERAS, output: "Prints the three predictions, which equal the PyTorch/NumPy values: approximately [0.541873 0.538756 0.529248]." },
          { t: "callout", kind: "warning", text: "Keras `Dense` kernels have shape `(n_in, n_out)` — flipped (transposed) compared with PyTorch's `(n_out, n_in)`. Forgetting the `.T` when copying weights between frameworks is a very common bug." },
        ] },
      ] },
    ],
    variations: [
      { t: "p", text: "The same basic rule $a^{[l]} = f(W^{[l]} a^{[l-1]} + b^{[l]})$ runs every architecture. What changes is the operation used instead of the matrix product, and how layers connect." },
      { t: "table", head: ["Variant", "Forward computation", "Used for"], rows: [
        ["Fully-connected (dense) MLP", "$a = f(Wa_{prev} + b)$", "Tabular data, final layers of bigger models"],
        ["Convolutional layer", "The same weights slid over small patches (see [Convolution](/learn/m08/convolution))", "Images, audio spectrograms"],
        ["Residual block", "$a = a_{prev} + F(a_{prev})$ — the input is added back to the output", "Very deep CNNs and every Transformer"],
        ["Recurrent layer", "$h_t = f(W_x x_t + W_h h_{t-1} + b)$ — the forward pass also runs through time", "Sequences (see [RNNs](/learn/m09/rnn-basics))"],
        ["Attention layer", "Output = a weighted average of value vectors, with weights that depend on the data", "Transformers (see [Attention](/learn/m10/scaled-dot-product-attention))"],
        ["Training vs inference mode", "Dropout and batch-norm act differently: at inference dropout is off and batch-norm uses running statistics", "Every production model"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Inference / serving**: every prediction a deployed model makes (search ranking, recommendations, fraud scores) is a forward pass. Batching, quantization and ONNX export exist to make it cheap.",
        "**Training**: every training step starts with a forward pass on a mini-batch.",
        "**Feature extraction**: run a pretrained network and keep a middle layer's activations as embeddings for search, clustering and transfer learning.",
        "**Latency budgeting**: cost grows with parameter count, so engineers estimate forward-pass FLOPs to pick a model size that answers fast enough.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Strength — simple and parallel**: a fixed list of matrix products, perfect for GPUs and easy to batch.",
        "**Strength — deterministic**: with fixed weights (and no dropout), the same input always gives the same output.",
        "**Limitation — memory during training**: every activation is kept for backpropagation, often using more memory than the weights. Gradient checkpointing recomputes some to save memory.",
        "**Limitation — layers run one after another**: layer $l$ can't start until layer $l-1$ is done, which limits parallelism across depth.",
        "**Limitation — fixed effort**: a plain feedforward network spends the same compute on every input, easy or hard.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Shape errors from the wrong weight orientation.** Mixing up $(n_{out}, n_{in})$ and $(n_{in}, n_{out})$ gives errors like `shapes (3,2) and (3,2) not aligned` — or, for square layers, silently wrong results. Fix: write each array's shape next to each line, as in the code above.",
        "**Forgetting the activation on hidden layers.** Without it the network stays linear, however deep. Fix: make sure every hidden layer is followed by a non-linearity.",
        "**Applying sigmoid/softmax twice.** E.g. ending a PyTorch model with `nn.Sigmoid()` *and* using `BCEWithLogitsLoss`, which already applies sigmoid inside. Training still runs, but gradients are weak and wrong. Fix: output raw logits when the loss expects logits.",
        "**Bias added along the wrong axis.** With one example per row, the bias must have shape `(n_out,)` (or `(1, n_out)`). Shape `(n_out, 1)` would broadcast into an $n_{out}\\times n_{out}$ mess.",
        "**Leaving the model in training mode for predictions.** In PyTorch, forgetting `model.eval()` keeps dropout on and makes batch-norm use batch statistics, so predictions get noisy.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is forward propagation?", a: "Computing a network's output from its input. For each layer in order, compute the weighted sums $z = Wa_{prev} + b$, then apply the activation $a = f(z)$. The last layer's activations are the prediction. During training, the loss is then computed from them." },
      { t: "qa", level: "beginner", q: "Why do we need a bias term?", a: "Without a bias, a neuron's weighted sum is 0 whenever all inputs are 0, so its decision boundary must pass through the origin. The bias lets the boundary (or the activation's input) shift freely, which makes each neuron much more flexible." },
      { t: "qa", level: "beginner", q: "What is the difference between $z$ and $a$?", a: "$z$ is the pre-activation — weighted sum plus bias. $a = f(z)$ is the activation, the neuron's output after the non-linearity. Both are stored during training: the backward pass needs the previous layer's $a$ (for weight gradients) and $f'(z)$." },
      { t: "qa", level: "intermediate", q: "How many multiply–add operations does one forward pass of a 784–256–128–10 MLP need per example?", a: "$784\\cdot256 + 256\\cdot128 + 128\\cdot10 = 200{,}704 + 32{,}768 + 1{,}280 = 234{,}752$ multiply–adds (≈ 0.47 MFLOPs if you count multiply and add separately), plus cheap bias additions and activations." },
      { t: "qa", level: "intermediate", q: "Why is batching efficient?", a: "It turns many matrix–vector products into one matrix–matrix product, so each loaded weight is reused for many examples (higher arithmetic intensity) — what GPUs and BLAS libraries exploit. It also spreads per-call overhead." },
      { t: "qa", level: "intermediate", q: "Why must intermediate activations be cached during training?", a: "The gradient of a weight $W^{[l]}_{jk}$ is $\\delta^{[l]}_j a^{[l-1]}_k$, and $\\delta$ itself needs $f'(z^{[l]})$. Without the saved $a$ and $z$, the backward pass would have to redo the forward pass." },
      { t: "qa", level: "advanced", q: "What does gradient checkpointing trade off?", a: "It saves activations only at chosen layers and recomputes the rest during the backward pass. This cuts activation memory (to about $O(\\sqrt{L})$ for $L$ layers with the best split) at the cost of roughly one extra forward pass of compute." },
      { t: "qa", level: "advanced", q: "Why do frameworks behave differently in train and eval modes during the forward pass?", a: "Some layers are random or batch-dependent in training. Dropout randomly zeroes units (and scales up the rest). Batch-norm normalizes with the current mini-batch's statistics and updates running averages. At inference we want stable outputs that don't depend on the rest of the batch, so dropout is off and batch-norm uses running statistics." },
    ],
    practice: [
      { t: "quiz", questions: [
        { q: "A layer maps 64 inputs to 32 outputs. In the convention $z = Wa + b$, what is the shape of $W$?", options: ["64 × 32", "32 × 64", "32 × 32", "64 × 64"], answer: 1, explain: "$W$ has one row per output neuron and one column per input: $n_{out}\\times n_{in} = 32\\times64$." },
        { kind: "numeric", q: "A neuron has weights $(0.5, -1.0)$, bias $0.2$ and receives $(2, 1)$. What is its pre-activation $z$?", answer: 0.2, tolerance: 0.001, explain: "$0.5\\cdot2 + (-1.0)\\cdot1 + 0.2 = 1 - 1 + 0.2 = 0.2$." },
        { kind: "numeric", q: "Same neuron, now with ReLU activation. What is its activation $a$?", answer: 0.2, tolerance: 0.001, explain: "$\\mathrm{ReLU}(0.2) = 0.2$ because $z > 0$." },
        { q: "A batch of 16 examples with 10 features enters a layer with 8 neurons. What is the shape of $Z$?", options: ["10 × 8", "8 × 16", "16 × 8", "16 × 10"], answer: 2, explain: "$(16\\times10)(10\\times8) = 16\\times8$: one row per example, one column per neuron." },
        { q: "What happens to a 5-layer network if every activation is the identity?", options: ["It becomes more expressive", "It is equivalent to a single linear layer", "It cannot be trained", "It becomes a convolutional network"], answer: 1, explain: "Linear steps chained together are still linear: $W_5\\cdots W_1 x + b'$ — one matrix and one bias." },
        { kind: "numeric", q: "How many parameters (weights + biases) does a 3–4–2 MLP have?", answer: 26, tolerance: 0, explain: "Layer 1: $3\\cdot4+4 = 16$. Layer 2: $4\\cdot2+2 = 10$. Total 26." },
      ] },
      { t: "exercise", title: "Forward pass by hand", prompt: "A 2–2–1 network uses ReLU in the hidden layer and no activation (identity) at the output. $W^{[1]} = \\begin{pmatrix}1 & -1\\\\ 2 & 0.5\\end{pmatrix}$, $b^{[1]} = (0, -1)$, $W^{[2]} = (1, -2)$, $b^{[2]} = 0.5$. Compute the output for $x = (1, 2)$.", hint: "Compute $z^{[1]}$, apply ReLU to each entry, then do the output layer.", solution: [
        { t: "steps", items: [
          { title: "Hidden pre-activations", tex: String.raw`z^{[1]} = \begin{pmatrix}1\cdot1 - 1\cdot2 + 0\\ 2\cdot1 + 0.5\cdot2 - 1\end{pmatrix} = \begin{pmatrix}-1\\ 2\end{pmatrix}` },
          { title: "ReLU", tex: String.raw`a^{[1]} = (\max(0,-1),\ \max(0,2)) = (0,\ 2)` },
          { title: "Output", tex: String.raw`\hat y = 1\cdot0 - 2\cdot2 + 0.5 = -3.5` },
        ] },
        { t: "p", text: "Hidden neuron 1 is switched off for this input. Its weight into the output has no effect on $\\hat y$, and (as backpropagation will show) it gets zero gradient." },
      ] },
      { t: "exercise", title: "Add softmax output", prompt: "Change the NumPy `MLP` so the last layer can use **softmax** (for multi-class classification). Softmax must work on each row of the batch separately, and must be numerically stable.", hint: "Subtract each row's maximum before taking exponentials: $\\mathrm{softmax}(z) = \\mathrm{softmax}(z - \\max z)$.", solution: [
        { t: "code", lang: "python", title: "softmax_output.py", code: String.raw`import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)     # stability: largest logit becomes 0
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)  # each row sums to 1

logits = np.array([[2.0, 1.0, 0.1], [1000.0, 1000.0, 999.0]])
p = softmax(logits)
print(p.round(4))
print("row sums:", p.sum(axis=1))`, output: String.raw`[[0.659  0.2424 0.0986]
 [0.4223 0.4223 0.1554]]
row sums: [1. 1.]` },
        { t: "p", text: "Register it with `ACTIVATIONS[\"softmax\"] = softmax`. Without subtracting the max, `np.exp(1000)` overflows to `inf` and the second row becomes `nan`." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "The forward pass runs the network layer by layer: $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$, $a^{[l]} = f^{[l]}(z^{[l]})$, starting from $a^{[0]} = x$.",
        "$W^{[l]}$ has shape $n_l\\times n_{l-1}$: row $j$ holds neuron $j$'s incoming weights.",
        "For a batch with one example per row: $Z = AW^\\top + b$ — one matrix product per layer for all examples.",
        "Non-linear activations make depth useful; without them the network collapses to one linear map.",
        "During training, every $z$ and $a$ is saved for the backward pass.",
        "Cost ≈ one multiply–add per weight per example.",
      ] },
      { t: "math", tex: String.raw`\hat y = f^{[L]}\big(W^{[L]} f^{[L-1]}(\cdots f^{[1]}(W^{[1]}x + b^{[1]}) \cdots) + b^{[L]}\big)` },
    ],
    next: [
      { t: "p", text: "You can now compute a prediction and its loss. Next question: **how should each weight change to shrink the loss?** [Backpropagation](/learn/m04/backpropagation) answers it for millions of weights, continuing this worked example to compute all nine gradients." },
    ],
  },
};

export default lesson;
