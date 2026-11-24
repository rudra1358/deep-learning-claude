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
  summary: "How a neural network turns an input into a prediction: weighted sums and activations computed layer by layer — first by hand, then as matrix products over a whole batch.",
  minutes: 45,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices", "m03/artificial-neuron", "m03/activation-functions", "m03/multilayer-networks"],
  next: ["m04/backpropagation"],
  terms: ["forward-propagation", "neuron", "activation", "matrix-multiplication", "loss"],
  sections: {
    intro: [
      { t: "p", text: "**Forward propagation** (the *forward pass*) is the computation a neural network performs to turn an input into an output. Data enters the input layer, each layer transforms the previous layer's output, and the final layer produces a prediction. Every use of a trained network — classifying a photo, translating a sentence, predicting a price — is one forward pass." },
      { t: "p", text: "It is also the first half of training. To learn, a network must (1) make a prediction, (2) measure how wrong it is with a **loss function**, and (3) work out how to change each weight to reduce that loss. Steps (1) and (2) are forward propagation; step (3) is [backpropagation](/learn/m04/backpropagation), which reuses the intermediate values the forward pass computes. That is why we study the forward pass in such detail: backpropagation is impossible to understand without knowing exactly which numbers it was computed from." },
      { t: "p", text: "The problem it solves is composition. A single neuron can only draw a straight decision boundary. By chaining layers — each one a linear map followed by a non-linear activation — a network can represent extremely complex functions. Forward propagation is simply the evaluation of that chain, $f(x) = f_L(\\dots f_2(f_1(x)))$." },
      { t: "callout", kind: "key", text: "Forward propagation = repeatedly apply **z = Wa + b** then **a = f(z)**, starting from **a = x**, until you reach the output layer." },
    ],
    prerequisites: [
      { t: "list", items: [
        "[Vectors, matrices and matrix multiplication](/learn/m02/vectors-matrices) — every layer is a matrix–vector (or matrix–matrix) product; you need the shape rule $(m\\times n)(n\\times p) = (m\\times p)$.",
        "[The artificial neuron](/learn/m03/artificial-neuron) — weights, bias and weighted sum for one unit.",
        "[Activation functions](/learn/m03/activation-functions) — sigmoid, tanh, ReLU and why non-linearity is required.",
        "[Multi-layer networks](/learn/m03/multilayer-networks) — what input, hidden and output layers are.",
      ] },
    ],
    intuition: [
      { t: "callout", kind: "analogy", title: "An assembly line of committees", text: "Picture a factory where a product passes through a series of rooms. In each room a committee of inspectors looks at **everything the previous room reported**, and each inspector writes down a single score: a weighted opinion (some earlier reports matter more to them than others, some count against), plus their own personal bias, then squashed through a rule like \"report at most 1, and never below 0\". The next room only sees these scores, never the raw product. The final room's scores are the network's answer." },
      { t: "p", text: "Each \"inspector\" is a neuron. The weights say how much each earlier report matters to them, the bias is their default inclination, and the activation function is the rule for turning their raw opinion into a report. Early rooms (layers) learn to report simple things — an edge in an image, a common word pair — and later rooms combine those reports into complex judgements — \"this is a cat\", \"this review is sarcastic\"." },
      { t: "p", text: "Crucially, information only flows **forward**: room 3 never sends anything back to room 2 during the forward pass. That is what makes this kind of network a *feedforward* network, and why the computation is a simple, ordered sequence of steps." },
    ],
    mechanism: [
      { t: "p", text: "We use a concrete network throughout this lesson: **2 inputs → 2 hidden neurons (sigmoid) → 1 output neuron (sigmoid)**. You can build exactly this in the visualizer below." },
      { t: "steps", items: [
        { title: "Set the input activations", text: "The input layer does no computation. Its \"activations\" are the input features themselves.", tex: String.raw`a^{[0]} = x` },
        { title: "Compute the weighted sum for every neuron in layer 1", text: "Each hidden neuron multiplies every input by its own weight, adds them, and adds its bias. This number $z$ is called the **pre-activation** (or logit).", tex: String.raw`z^{[1]}_j = \sum_k W^{[1]}_{jk}\, a^{[0]}_k + b^{[1]}_j` },
        { title: "Apply the activation function", text: "Each neuron passes its pre-activation through its activation function. The result is the neuron's **activation** — its output.", tex: String.raw`a^{[1]}_j = f^{[1]}\!\left(z^{[1]}_j\right)` },
        { title: "Repeat for every following layer", text: "Layer 2 does exactly the same thing, but its inputs are layer 1's activations, not the raw input." },
        { title: "Read off the prediction", text: "The activations of the last layer are the network's output $\\hat y$. Which activation the output layer uses depends on the task: sigmoid for a probability, softmax for class probabilities, identity (no activation) for regression." },
        { title: "Compute the loss (during training)", text: "Compare $\\hat y$ with the true target $y$ using a loss function, e.g. squared error $(\\hat y - y)^2$. Keep every $z$ and $a$ in memory — backpropagation needs them." },
      ] },
      { t: "callout", kind: "note", text: "Two values exist for every neuron: $z$ (before the activation) and $a$ (after it). Keeping them separate in your head — and in your code — is the single most useful habit for understanding backpropagation, because the derivative of the activation is evaluated at $z$." },
    ],
    math: [
      { t: "h", text: "Notation" },
      { t: "table", head: ["Symbol", "Meaning", "Shape"], rows: [
        ["$L$", "number of layers with parameters (input layer not counted)", "scalar"],
        ["$n_l$", "number of neurons in layer $l$ ($n_0$ = number of input features)", "scalar"],
        ["$W^{[l]}$", "weight matrix of layer $l$; entry $W^{[l]}_{jk}$ connects neuron $k$ of layer $l-1$ to neuron $j$ of layer $l$", "$n_l \\times n_{l-1}$"],
        ["$b^{[l]}$", "bias vector of layer $l$", "$n_l$"],
        ["$z^{[l]}$", "pre-activations of layer $l$", "$n_l$"],
        ["$a^{[l]}$", "activations of layer $l$; $a^{[0]} = x$", "$n_l$"],
        ["$f^{[l]}$", "activation function of layer $l$, applied element-wise", "—"],
      ] },
      { t: "h", text: "Vector form for one example" },
      { t: "math", tex: String.raw`z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad a^{[l]} = f^{[l]}\!\left(z^{[l]}\right), \qquad l = 1,\dots,L`, caption: "The whole forward pass. Row $j$ of $W^{[l]}$ holds neuron $j$'s incoming weights, so the matrix–vector product computes every neuron's weighted sum at once." },
      { t: "p", text: "Writing the network as one function makes its compositional structure explicit:" },
      { t: "math", tex: String.raw`\hat y = f^{[L]}\Big(W^{[L]} f^{[L-1]}\big(\cdots f^{[1]}(W^{[1]}x + b^{[1]})\cdots\big) + b^{[L]}\Big)` },
      { t: "h", text: "Worked example — every number" },
      { t: "p", text: "Input $x = (1.0,\\ 0.5)$, target $y = 1$, sigmoid activations $\\sigma(z) = 1/(1+e^{-z})$ in both layers, and parameters:" },
      { t: "math", tex: String.raw`W^{[1]} = \begin{pmatrix} 0.2 & -0.4 \\ 0.7 & 0.1 \end{pmatrix},\quad b^{[1]} = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix},\quad W^{[2]} = \begin{pmatrix} 0.6 & -0.3 \end{pmatrix},\quad b^{[2]} = 0.05` },
      { t: "steps", items: [
        { title: "Hidden neuron 1", tex: String.raw`z^{[1]}_1 = 0.2(1.0) + (-0.4)(0.5) + 0.1 = 0.1, \qquad a^{[1]}_1 = \sigma(0.1) = 0.524979` },
        { title: "Hidden neuron 2", tex: String.raw`z^{[1]}_2 = 0.7(1.0) + 0.1(0.5) - 0.1 = 0.65, \qquad a^{[1]}_2 = \sigma(0.65) = 0.657010` },
        { title: "Output neuron", tex: String.raw`z^{[2]} = 0.6(0.524979) - 0.3(0.657010) + 0.05 = 0.167884, \qquad \hat y = a^{[2]} = \sigma(0.167884) = 0.541873` },
        { title: "Loss (squared error)", tex: String.raw`\mathcal{L} = (\hat y - y)^2 = (0.541873 - 1)^2 = 0.209881` },
      ] },
      { t: "p", text: "The network predicts 0.54 where the answer is 1 — not good yet. [Backpropagation](/learn/m04/backpropagation) continues this exact example and computes how each of the 9 parameters should change." },
      { t: "h", text: "Batch (matrix) form" },
      { t: "p", text: "In practice we process a **batch** of $N$ examples at once. Stack the examples as rows of a matrix $X \\in \\mathbb{R}^{N\\times n_0}$. Then:" },
      { t: "math", tex: String.raw`Z^{[l]} = A^{[l-1]}\,W^{[l]\top} + \mathbf{1}\,b^{[l]\top}, \qquad A^{[l]} = f^{[l]}(Z^{[l]}), \qquad A^{[0]} = X`, caption: "Shapes: $(N\\times n_{l-1})(n_{l-1}\\times n_l) = N\\times n_l$. The bias row is broadcast to every example." },
      { t: "p", text: "One matrix–matrix product per layer now computes every neuron for every example. This is why GPUs — hardware built for large matrix products — made deep learning practical. For the batch $X = \\begin{pmatrix}1 & 0.5\\\\ 0 & 1\\\\ -1 & 2\\end{pmatrix}$ our network gives $Z^{[1]} = \\begin{pmatrix}0.1 & 0.65\\\\ -0.3 & 0\\\\ -0.9 & -0.6\\end{pmatrix}$ and predictions $(0.541873,\\ 0.538756,\\ 0.529248)$ — the first row matches the single-example calculation exactly." },
      { t: "h", text: "Cost of a forward pass" },
      { t: "p", text: "Layer $l$ performs $n_l n_{l-1}$ multiplications and as many additions per example, so a forward pass costs $\\sum_l n_l n_{l-1}$ multiply–adds per example — proportional to the number of weights. A useful rule of thumb: **forward FLOPs ≈ 2 × parameters × batch size**." },
    ],
    visual: [
      { t: "p", text: "The visualizer below is initialised with random weights. To reproduce the worked example, click each connection and set its weight (and click each neuron to set its bias) to the values above, set $x = (1, 0.5)$ and $y = 1$, then press **Step** repeatedly." },
      { t: "sim", id: "network", caption: "Gold dots carry values forward along the connections being used; the narration shows the weighted sum with the actual numbers." },
      { t: "list", items: [
        "Watch the order: every neuron in layer 1 is computed before any neuron in layer 2 — layer 2 needs all of layer 1's outputs.",
        "Change an input slider and notice that **every** downstream value changes. Then change a single weight into the output neuron: only the output changes.",
        "Switch the hidden activation to ReLU and give a hidden neuron a large negative bias: its activation becomes exactly 0, and it contributes nothing to the next layer (a \"dead\" neuron for this input).",
        "Set every activation to **Linear** and observe that the output is just a weighted sum of the inputs — the depth adds no expressive power without non-linearity.",
      ] },
    ],
    scratch: [
      { t: "p", text: "A complete forward-only MLP. Weights are stored as $(n_{out}\\times n_{in})$ to match the maths; for a batch stored row-wise we therefore compute `a @ W.T + b`. Part 1 reproduces the worked example, part 2 shows batching, part 3 prints the shapes in a deeper network." },
      { t: "code", lang: "python", title: "forward_propagation.py", code: FP_NUMPY, output: FP_NUMPY_OUT },
      { t: "p", text: "Explanation of the important pieces:" },
      { t: "list", items: [
        "`__init__` draws weights with standard deviation $\\sqrt{2/(n_{in}+n_{out})}$ (Xavier initialization — explained in [Weight Initialization](/learn/m07/initialization)); the constant 2 comes from that derivation, not from tuning. Biases start at zero.",
        "`forward` loops over layers; `a @ W.T + b` is the batch formula $A W^\\top + b$, where NumPy **broadcasting** adds the bias vector to every row.",
        "The `cache` dictionary keeps every $z$ and $a$. A forward-only inference engine could discard them, but training needs them for the backward pass.",
      ] },
    ],
    framework: [
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "p", text: "`nn.Linear(in, out)` stores `weight` with shape `(out, in)` — the same convention as our $W$ — and computes `x @ weight.T + bias`. Forward hooks print each layer's output so you can compare with the hand calculation." },
          { t: "code", lang: "python", title: "forward_pytorch.py", code: FP_TORCH, output: FP_TORCH_OUT },
          { t: "callout", kind: "tip", text: "`torch.no_grad()` tells PyTorch not to record the operations for backpropagation. Use it (or `torch.inference_mode()`) for inference: it saves memory and time." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "forward_keras.py (not executed in CI)", code: FP_KERAS, output: "Prints the three predictions, which equal the PyTorch/NumPy values: approximately [0.541873 0.538756 0.529248]." },
          { t: "callout", kind: "warning", text: "Keras `Dense` kernels have shape `(n_in, n_out)` — the transpose of PyTorch's `(n_out, n_in)`. Forgetting the `.T` when porting weights between frameworks is a very common bug." },
        ] },
      ] },
    ],
    variations: [
      { t: "p", text: "The basic recurrence $a^{[l]} = f(W^{[l]} a^{[l-1]} + b^{[l]})$ generalizes to every architecture; what changes is the operation that replaces the matrix product and how layers are connected." },
      { t: "table", head: ["Variant", "Forward computation", "Used for"], rows: [
        ["Fully-connected (dense) MLP", "$a = f(Wa_{prev} + b)$", "Tabular data, heads of larger models"],
        ["Convolutional layer", "Same weights slid over local patches (see [Convolution](/learn/m08/convolution))", "Images, audio spectrograms"],
        ["Residual block", "$a = a_{prev} + F(a_{prev})$ — input added back to the output", "Very deep CNNs and every Transformer"],
        ["Recurrent layer", "$h_t = f(W_x x_t + W_h h_{t-1} + b)$ — forward pass also runs through time", "Sequences (see [RNNs](/learn/m09/rnn-basics))"],
        ["Attention layer", "Output = data-dependent weighted average of value vectors", "Transformers (see [Attention](/learn/m10/scaled-dot-product-attention))"],
        ["Training vs inference mode", "Dropout and batch-norm behave differently: dropout is off and batch-norm uses running statistics at inference", "Every production model"],
      ] },
    ],
    applications: [
      { t: "list", items: [
        "**Inference / serving**: every prediction a deployed model makes — search ranking, recommendation, speech recognition, fraud scoring — is a forward pass. Production engineering (batching, quantization, ONNX export) is largely about making forward passes cheap.",
        "**Training**: each training step starts with a forward pass on a mini-batch.",
        "**Feature extraction**: running a pretrained network forward and keeping an intermediate layer's activations gives embeddings used for search, clustering and transfer learning.",
        "**Latency budgeting**: because the cost is roughly proportional to parameter count, engineers estimate forward-pass FLOPs to choose a model size that meets a response-time target.",
      ] },
    ],
    tradeoffs: [
      { t: "list", items: [
        "**Strength — simplicity and parallelism**: the forward pass is a fixed sequence of matrix products, ideal for GPUs and easy to batch.",
        "**Strength — determinism**: with fixed weights (and no dropout) the same input always produces the same output.",
        "**Limitation — memory during training**: all intermediate activations must be stored for backpropagation; for large models and batches, activation memory often exceeds parameter memory. Techniques like gradient checkpointing recompute some activations to save memory.",
        "**Limitation — sequential across layers**: layer $l$ cannot start before layer $l-1$ finishes, which limits parallelism across depth.",
        "**Limitation — fixed computation**: a plain feedforward network spends the same compute on every input, easy or hard.",
      ] },
    ],
    mistakes: [
      { t: "list", items: [
        "**Shape mismatch from the wrong weight orientation.** Mixing the $(n_{out}, n_{in})$ and $(n_{in}, n_{out})$ conventions gives errors like `shapes (3,2) and (3,2) not aligned`, or — worse, for square layers — silently wrong results. Fix: write the shape of every array next to each line, as in the code above.",
        "**Forgetting the activation on hidden layers.** Without it the network is linear no matter how deep. Fix: check that every hidden layer is followed by a non-linearity.",
        "**Applying sigmoid/softmax twice.** E.g. ending a PyTorch model with `nn.Sigmoid()` *and* using `BCEWithLogitsLoss` (which applies sigmoid internally). Training still runs but gradients are weak and wrong. Fix: output raw logits when the loss expects logits.",
        "**Bias broadcast along the wrong axis.** With row-wise batches the bias must have shape `(n_out,)` (or `(1, n_out)`), not `(n_out, 1)`, which would broadcast into an $n_{out}\\times n_{out}$ mess.",
        "**Leaving the model in training mode at inference.** In PyTorch, forgetting `model.eval()` keeps dropout active and makes batch-norm use batch statistics, giving noisy predictions.",
      ] },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is forward propagation?", a: "The computation of a network's output from its input: for each layer in order, compute the weighted sums $z = Wa_{prev} + b$ and apply the activation $a = f(z)$. The final layer's activations are the prediction; during training the loss is then computed from them." },
      { t: "qa", level: "beginner", q: "Why do we need a bias term?", a: "Without a bias, every neuron's weighted sum is 0 when all inputs are 0, so its decision boundary must pass through the origin. The bias shifts the boundary (or the activation's input) freely, which greatly increases what each neuron can represent." },
      { t: "qa", level: "beginner", q: "What is the difference between $z$ and $a$?", a: "$z$ is the pre-activation — the weighted sum plus bias. $a = f(z)$ is the activation, the neuron's output after the non-linearity. Both are stored during training because the backward pass needs $a$ of the previous layer (for weight gradients) and $f'(z)$." },
      { t: "qa", level: "intermediate", q: "How many multiply–add operations does one forward pass of a 784–256–128–10 MLP need per example?", a: "$784\\cdot256 + 256\\cdot128 + 128\\cdot10 = 200{,}704 + 32{,}768 + 1{,}280 = 234{,}752$ multiply–adds (≈ 0.47 MFLOPs counting multiply and add separately), plus the cheap bias additions and activations." },
      { t: "qa", level: "intermediate", q: "Why is batching efficient?", a: "It turns many matrix–vector products into one matrix–matrix product. Matrix–matrix products reuse each loaded weight across many examples, giving much higher arithmetic intensity, which GPUs and optimized BLAS libraries exploit. It also amortizes per-call overhead." },
      { t: "qa", level: "intermediate", q: "Why must intermediate activations be cached during training?", a: "The gradient of a weight $W^{[l]}_{jk}$ is $\\delta^{[l]}_j a^{[l-1]}_k$, and $\\delta$ itself requires $f'(z^{[l]})$. Without the cached $a$ and $z$ the backward pass would have to recompute the forward pass." },
      { t: "qa", level: "advanced", q: "What does gradient checkpointing trade off?", a: "It stores activations only at selected layers and recomputes the others during the backward pass. This cuts activation memory (to about $O(\\sqrt{L})$ for $L$ layers with optimal segmenting) at the cost of roughly one extra forward pass of compute." },
      { t: "qa", level: "advanced", q: "Why do frameworks behave differently in train and eval modes during the forward pass?", a: "Some layers are stochastic or batch-dependent during training: dropout randomly zeroes units (and rescales survivors), and batch-norm normalizes with the current mini-batch's statistics while updating running averages. At inference we want deterministic outputs that don't depend on other examples in the batch, so dropout is disabled and batch-norm uses the running statistics." },
    ],
    practice: [
      { t: "quiz", questions: [
        { q: "A layer maps 64 inputs to 32 outputs. In the convention $z = Wa + b$, what is the shape of $W$?", options: ["64 × 32", "32 × 64", "32 × 32", "64 × 64"], answer: 1, explain: "$W$ has one row per output neuron and one column per input: $n_{out}\\times n_{in} = 32\\times64$." },
        { kind: "numeric", q: "A neuron has weights $(0.5, -1.0)$, bias $0.2$ and receives $(2, 1)$. What is its pre-activation $z$?", answer: 0.2, tolerance: 0.001, explain: "$0.5\\cdot2 + (-1.0)\\cdot1 + 0.2 = 1 - 1 + 0.2 = 0.2$." },
        { kind: "numeric", q: "Using the previous neuron with ReLU activation, what is its activation $a$?", answer: 0.2, tolerance: 0.001, explain: "$\\mathrm{ReLU}(0.2) = 0.2$ since $z > 0$." },
        { q: "A batch of 16 examples with 10 features enters a layer with 8 neurons. What is the shape of $Z$?", options: ["10 × 8", "8 × 16", "16 × 8", "16 × 10"], answer: 2, explain: "$(16\\times10)(10\\times8) = 16\\times8$: one row per example, one column per neuron." },
        { q: "What happens to a 5-layer network if every activation is the identity?", options: ["It becomes more expressive", "It is equivalent to a single linear layer", "It cannot be trained", "It becomes a convolutional network"], answer: 1, explain: "A composition of linear maps is linear: $W_5\\cdots W_1 x + b'$ — a single matrix and bias." },
        { kind: "numeric", q: "How many parameters (weights + biases) does a 3–4–2 MLP have?", answer: 26, tolerance: 0, explain: "Layer 1: $3\\cdot4+4 = 16$. Layer 2: $4\\cdot2+2 = 10$. Total 26." },
      ] },
      { t: "exercise", title: "Forward pass by hand", prompt: "A 2–2–1 network uses ReLU in the hidden layer and no activation (identity) at the output. $W^{[1]} = \\begin{pmatrix}1 & -1\\\\ 2 & 0.5\\end{pmatrix}$, $b^{[1]} = (0, -1)$, $W^{[2]} = (1, -2)$, $b^{[2]} = 0.5$. Compute the output for $x = (1, 2)$.", hint: "Compute $z^{[1]}$, apply ReLU element-wise, then the output layer.", solution: [
        { t: "steps", items: [
          { title: "Hidden pre-activations", tex: String.raw`z^{[1]} = \begin{pmatrix}1\cdot1 - 1\cdot2 + 0\\ 2\cdot1 + 0.5\cdot2 - 1\end{pmatrix} = \begin{pmatrix}-1\\ 2\end{pmatrix}` },
          { title: "ReLU", tex: String.raw`a^{[1]} = (\max(0,-1),\ \max(0,2)) = (0,\ 2)` },
          { title: "Output", tex: String.raw`\hat y = 1\cdot0 - 2\cdot2 + 0.5 = -3.5` },
        ] },
        { t: "p", text: "Note that hidden neuron 1 is inactive for this input: its weight into the output has no effect on $\\hat y$, and (as you will see in backpropagation) receives zero gradient." },
      ] },
      { t: "exercise", title: "Add softmax output", prompt: "Modify the NumPy `MLP` so the final layer can use **softmax** (for multi-class classification). Softmax must act on each row of the batch, and must be numerically stable.", hint: "Subtract each row's maximum before exponentiating: $\\mathrm{softmax}(z) = \\mathrm{softmax}(z - \\max z)$.", solution: [
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
        { t: "p", text: "Register it with `ACTIVATIONS[\"softmax\"] = softmax`. Without the max-subtraction, `np.exp(1000)` overflows to `inf` and the second row becomes `nan`." },
      ] },
    ],
    summary: [
      { t: "list", items: [
        "Forward propagation evaluates the network layer by layer: $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$, $a^{[l]} = f^{[l]}(z^{[l]})$, with $a^{[0]} = x$.",
        "$W^{[l]}$ has shape $n_l\\times n_{l-1}$: row $j$ is neuron $j$'s incoming weights.",
        "For a batch stored row-wise: $Z = AW^\\top + b$ — one matrix product per layer for all examples.",
        "Non-linear activations are what make depth useful; without them the network collapses to one linear map.",
        "During training every $z$ and $a$ is cached for the backward pass.",
        "Cost ≈ one multiply–add per weight per example.",
      ] },
      { t: "math", tex: String.raw`\hat y = f^{[L]}\big(W^{[L]} f^{[L-1]}(\cdots f^{[1]}(W^{[1]}x + b^{[1]}) \cdots) + b^{[L]}\big)` },
    ],
    next: [
      { t: "p", text: "You can now compute a prediction and its loss. The obvious next question is: **how should each weight change to make the loss smaller?** Answering it efficiently for millions of weights is the job of [Backpropagation](/learn/m04/backpropagation), which continues the exact worked example from this lesson and computes all nine gradients." },
    ],
  },
};

export default lesson;
