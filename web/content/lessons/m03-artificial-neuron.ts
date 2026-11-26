import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m03/artificial-neuron",
  title: "The Artificial Neuron and the Perceptron",
  summary:
    "Meet the tiny building block of every neural network. We go from real brain cells to the first artificial neurons (McCulloch–Pitts, Rosenblatt), learn what weights, bias and the weighted sum do, see why a neuron draws a straight line through the data, train a perceptron by hand, see when it is guaranteed to finish learning, and find out why XOR needs a hidden layer.",
  minutes: 55,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices", "m03/linear-logistic-regression"],
  next: ["m03/activation-functions", "m03/multilayer-networks"],
  terms: [
    "artificial-neuron",
    "mcculloch-pitts-neuron",
    "perceptron",
    "weights",
    "bias",
    "weighted-sum",
    "decision-boundary",
    "perceptron-learning-rule",
    "linear-separability",
    "xor-problem",
  ],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "An **artificial neuron** is the smallest working part of a neural network: a tiny calculator that makes one decision. It multiplies each input by a **weight** (how much that input matters), adds them up together with a **bias** (a baseline that moves the total up or down), and passes the total through an **activation function** that turns it into an answer, such as 'yes' or 'no'. Every network in this course — fully connected layers, convolutional filters, Transformer feed-forward blocks — is this one operation repeated many times.",
      },
      {
        t: "p",
        text: "In 1943 Warren McCulloch and Walter Pitts showed that a very simple nerve-cell model (0/1 inputs, a threshold, a 0/1 output) can compute AND, OR and NOT, and networks of them any **Boolean function** (any rule from 0/1 inputs to a 0/1 output). But it could not learn: a human picked the threshold. In 1958 Frank Rosenblatt's **perceptron** added real-valued weights and, crucially, a **learning rule** that adjusts them from labelled examples.",
      },
      {
        t: "p",
        text: String.raw`The perceptron does **binary classification**: it sorts each input $\mathbf{x}$ (a list of numbers) into class 1 or class 0, like 'spam or not spam', learning from examples. Picture the inputs as points on a page: one neuron draws one straight line and labels points by their side. With more inputs the line becomes a flat **hyperplane**.`,
      },
      {
        t: "p",
        text: "That line is also the limit. In 1969 Marvin Minsky and Seymour Papert's book “Perceptrons” proved what single-layer perceptrons cannot compute — most famously **XOR**, whose classes no straight line splits. The fix, stacking neurons into **hidden layers**, is the foundation of deep learning and where [Multi-Layer Networks](/learn/m03/multilayer-networks) begins.",
      },
      {
        t: "callout",
        kind: "key",
        title: "In one sentence",
        text: String.raw`A neuron computes $z = w_1x_1 + \dots + w_nx_n + b$ and outputs $f(z)$. The perceptron is a neuron with a step activation (output 1 or 0) plus a rule that nudges the weights whenever it gets an example wrong.`,
      },
    ],

    // --------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — the **dot product** $\mathbf{w}\cdot\mathbf{x} = \sum_i w_ix_i$ (multiply matching numbers and add; a neuron's weighted sum is exactly this), a vector's length $\lVert\mathbf{w}\rVert$, and the idea that a vector points in a direction.`,
          "[Linear and Logistic Regression as Neural Networks](/learn/m03/linear-logistic-regression) — a model whose **parameters** are fitted to labelled data, and the **decision boundary**, where the model switches answers. Logistic regression turns out to be a single neuron with a sigmoid activation.",
          String.raw`Simple inequality rules (if $a \ge 0$ and $c \ge 0$ then $a + c \ge 0$), used to prove that one neuron cannot compute XOR.`,
          "Optional: [History of Neural Networks](/learn/m01/history) for the first 'AI winter' that followed the 1969 critique.",
        ],
      },
      {
        t: "callout",
        kind: "note",
        text: "No calculus is needed here. The perceptron rule is not gradient descent on a smooth loss; it is a simple 'fix it when wrong' rule. Derivatives arrive in the next lessons, once the step function is replaced by smooth activations.",
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A loan officer with a scorecard",
        text: "A loan officer scores each application: income adds points, debt subtracts, years in the same job add a few. How much each factor counts is its **weight**. A strict bank starts everyone at −20, a relaxed one at +5: that is the **bias**. Zero or more means approved — the **threshold activation**. When an approved customer defaults, the officer lowers the weight on whatever made them look good; when a good one was rejected, they raise it. That habit is the **perceptron learning rule**. Where it stops: the neuron has no judgement, only weighted sums.",
      },
      { t: "h", text: "The biological neuron, briefly" },
      {
        t: "list",
        items: [
          "**Dendrites** — branching fibres that receive signals.",
          "**Synapses** — junctions where one cell's axon terminal releases chemicals (neurotransmitters) onto another cell's dendrite. **Excitatory** synapses push the receiving cell's voltage up ('go'), **inhibitory** ones push it down ('stop'). Each has a strength that changes with experience.",
          "**Soma** — the cell body, which combines all these pushes.",
          "**Axon, threshold, spike** — if the voltage near the start of the output cable (the **axon**) crosses a **firing threshold**, the cell fires an **action potential**: an all-or-none pulse that travels to the synapses on other neurons.",
        ],
      },
      {
        t: "table",
        head: ["Biological neuron", "Artificial neuron", "Where the comparison breaks"],
        rows: [
          ["Dendrites receive signals", "Inputs $x_1,\\dots,x_n$", "Real dendrites do non-linear processing of their own."],
          ["Synapse strength (excitatory / inhibitory)", "Weight $w_i$ (positive / negative)", "A real neuron is usually all-excitatory or all-inhibitory, not freely signed."],
          ["Soma combines inputs", "Weighted sum $z = \\sum_i w_ix_i + b$", "Real cells combine signals continuously in time, with a leaky voltage."],
          ["Firing threshold", "Bias $b$ (same as a threshold $\\theta = -b$)", "—"],
          ["All-or-none spike along the axon", "Activation $a = f(z)$", "Real neurons code information in spike timing and rate; most artificial neurons output one real number."],
          ["Learning by changing synapses", "Learning by changing weights", "Brains do not run backpropagation in the form deep learning uses."],
        ],
      },
      {
        t: "p",
        text: "The comparison explains the vocabulary, but don't take it too far. Artificial neurons are math functions chosen because they are easy to train, not because they copy biology.",
      },
      { t: "h", text: "What the neuron does, in plain language" },
      {
        t: "list",
        items: [
          "**Weigh the evidence.** A big positive weight means the input is strong evidence for class 1; a negative weight means evidence against.",
          "**Add it up, plus a baseline.** The bias shifts the total whatever the input, which sets how easy it is to say yes.",
          "**Decide.** For the perceptron: output 1 if the total is zero or more, else 0.",
          "**Learn from mistakes.** If wrong, move each weight a little toward the right answer, more for inputs that were more active.",
        ],
      },
      {
        t: "p",
        text: "As a picture: with two inputs, the points where the total is exactly zero form a straight line. Points on one side get 1, the other side 0. Learning means turning and sliding that line until it splits the classes — which works if, and only if, such a line exists.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "1. The McCulloch–Pitts neuron (1943)" },
      {
        t: "p",
        text: String.raw`Its inputs are 0 or 1 ($x_i \in \{0,1\}$), each either **excitatory** ('vote yes') or **inhibitory** ('veto'). In the original version, any active inhibitory input forces the output to 0. Otherwise the unit counts its active excitatory inputs and fires (outputs 1) if the count reaches a fixed whole-number threshold $\theta$. Nothing is learned: the designer picks $\theta$ and the wiring.`,
      },
      {
        t: "steps",
        items: [
          { title: "AND of two inputs", text: String.raw`Threshold $\theta = 2$: fires only when both inputs are 1.` },
          { title: "OR of two inputs", text: String.raw`Threshold $\theta = 1$: fires when at least one input is 1.` },
          { title: "NOT", text: String.raw`One inhibitory input and $\theta = 0$. Input off: the count $0 \ge 0$, so it fires. Input on: the veto forces 0.` },
          { title: "Any Boolean function", text: "AND, OR and NOT can build any logic rule (they are **functionally complete**), so networks of MP units can compute any Boolean function — the 1943 paper's central claim. What they lack is a way to find the thresholds from data." },
        ],
      },
      { t: "h", text: "2. The modern artificial neuron — the forward computation" },
      {
        t: "steps",
        items: [
          { title: "Receive the input vector", tex: String.raw`\mathbf{x} = (x_1, x_2, \dots, x_n)` , text: "Real numbers: pixel intensities, word counts, other neurons' outputs." },
          { title: "Multiply each input by its weight", tex: String.raw`p_i = w_i x_i, \quad i = 1,\dots,n`, text: "The weights are the **learnable parameters**; sign and size say which way and how hard each input pushes." },
          { title: "Sum the products and add the bias", tex: String.raw`z = p_1 + p_2 + \dots + p_n + b = \mathbf{w}^\top\mathbf{x} + b`, text: String.raw`$z$ — the total before the activation — is called the **pre-activation**, **net input** or **logit**. $\mathbf{w}^\top\mathbf{x}$ is the dot product. The bias $b$ is learned too.` },
          { title: "Apply the activation function", tex: String.raw`a = f(z)`, text: "For the perceptron $f$ is the step function. Later lessons use sigmoid, tanh, ReLU and others; see [Activation Functions](/learn/m03/activation-functions)." },
          { title: "Interpret the output", text: String.raw`With a step, $a \in \{0,1\}$ is the predicted class. With a sigmoid, $a \in (0,1)$ can be read as a probability. Inside a bigger network, $a$ becomes an input to the next layer.` },
        ],
      },
      { t: "h", text: "3. The Rosenblatt perceptron — the training procedure" },
      {
        t: "p",
        text: String.raw`The prediction is $\hat y = \text{step}(z)$ (the hat means 'predicted'; plain $y$ is the true label). Throughout this lesson and in the simulator, $\text{step}(z) = 1$ if $z \ge 0$ and $0$ otherwise. We train on labelled pairs $\{(\mathbf{x}^{(k)}, y^{(k)})\}$ with $y \in \{0,1\}$, **online** — one example at a time:`,
      },
      {
        t: "steps",
        items: [
          { title: "Initialise", text: String.raw`Set $\mathbf{w} = \mathbf{0}$ and $b = 0$ (or small random values). Choose a **learning rate** $\eta > 0$ ('eta'), the size of each correction.` },
          { title: "Take the next example", text: String.raw`Pick $(\mathbf{x}, y)$ — in a fixed order, or shuffled each epoch.` },
          { title: "Forward pass", tex: String.raw`z = \mathbf{w}^\top\mathbf{x} + b, \qquad \hat y = \text{step}(z)` },
          { title: "Compute the error", tex: String.raw`e = y - \hat y \in \{-1, 0, +1\}`, text: String.raw`$e = 0$: correct, nothing changes. $e = +1$: said 0, should be 1 (z too small). $e = -1$: said 1, should be 0 (z too large).` },
          { title: "Update weights and bias", tex: String.raw`\Delta\mathbf{w} = \eta\, e\, \mathbf{x}, \quad \Delta b = \eta\, e, \qquad \mathbf{w} \leftarrow \mathbf{w} + \Delta\mathbf{w}, \quad b \leftarrow b + \Delta b`, text: String.raw`$\Delta$ = 'change in', $\leftarrow$ = 'replace with'.` },
          { title: "Repeat over the dataset", text: "One full pass is an **epoch**. Count its mistakes." },
          { title: "Stop", text: "A mistake-free epoch means the weights will never change again: the perceptron has **converged**. Otherwise stop at a maximum number of epochs." },
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "Why inputs with value 0 never get their weight changed",
        text: String.raw`Since $\Delta w_i = \eta e x_i$, an input of 0 did not cause the mistake and its weight is left alone. The bias acts like a weight on a constant input of 1, so it moves on every mistake. In the hand example, on sample (0, 0) only $b$ moves.`,
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "The neuron" },
      { t: "math", tex: String.raw`z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b, \qquad a = f(z)` },
      {
        t: "p",
        text: String.raw`In words: multiply each input by its weight, add them up ($\Sigma$ means 'sum'), add the bias, apply the activation.`,
      },
      {
        t: "list",
        items: [
          String.raw`$n$ — number of inputs (**features**).`,
          String.raw`$\mathbf{x} \in \mathbb{R}^n$ — the input vector, $n$ real numbers $x_i$.`,
          String.raw`$\mathbf{w} \in \mathbb{R}^n$ — one learned weight $w_i$ per input.`,
          String.raw`$b \in \mathbb{R}$ — the learned bias. Older texts use a threshold $\theta$ with "fire if $\mathbf{w}^\top\mathbf{x} \ge \theta$"; this is the same rule with $b = -\theta$.`,
          String.raw`$z$ — the pre-activation (weighted sum plus bias).`,
          String.raw`$f$ — the activation function; $a$ — the neuron's output (activation).`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Augmented form.** Append a 1 to the input and the bias to the weights: $\tilde{\mathbf{x}} = (x_1,\dots,x_n,1)$, $\tilde{\mathbf{w}} = (w_1,\dots,w_n,b)$. Then $z = \tilde{\mathbf{w}}^\top\tilde{\mathbf{x}}$. So the bias is just a weight on an input that is always 1, and its update $\Delta b = \eta e$ is the weight rule with $x = 1$.`,
      },
      { t: "h", text: "Worked example: one forward pass" },
      {
        t: "p",
        text: String.raw`Take $\mathbf{x} = (2, -1, 0.5)$, $\mathbf{w} = (0.4, 0.3, -0.6)$ and $b = 0.1$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Products", tex: String.raw`w_1x_1 = 0.4 \cdot 2 = 0.8,\quad w_2x_2 = 0.3 \cdot (-1) = -0.3,\quad w_3x_3 = -0.6 \cdot 0.5 = -0.3` },
          { title: "Weighted sum plus bias", tex: String.raw`z = 0.8 - 0.3 - 0.3 + 0.1 = 0.3` },
          { title: "Activations of the same z", tex: String.raw`\text{step}(0.3) = 1,\quad \sigma(0.3) = \frac{1}{1+e^{-0.3}} = 0.5744,\quad \tanh(0.3) = 0.2913,\quad \text{ReLU}(0.3) = 0.3` },
        ],
      },
      {
        t: "p",
        text: "The total $z$ is the same whatever activation follows; the activation only changes how it is reported. So a step neuron, a sigmoid neuron cut at 0.5 and a tanh neuron cut at 0 all split the inputs at the same place: $z = 0$.",
      },
      { t: "h", text: "Geometry: the decision boundary is a hyperplane" },
      {
        t: "p",
        text: String.raw`The **decision boundary** is where the neuron is exactly on the fence: $\{\mathbf{x} : \mathbf{w}^\top\mathbf{x} + b = 0\}$. In $\mathbb{R}^2$ it is the line $w_1x_1 + w_2x_2 + b = 0$; in $\mathbb{R}^n$ it is a flat $(n-1)$-dimensional **hyperplane**. Three facts follow:`,
      },
      {
        t: "list",
        ordered: true,
        items: [
          String.raw`**$\mathbf{w}$ is perpendicular (normal) to the boundary.** For two points $\mathbf{x}_1$, $\mathbf{x}_2$ on it, subtracting their equations gives $\mathbf{w}^\top(\mathbf{x}_1 - \mathbf{x}_2) = 0$ — a zero dot product means a right angle to every direction along the boundary. $\mathbf{w}$ points to the side where $z > 0$ (class 1).`,
          String.raw`**$z$ measures signed distance.** A point's perpendicular distance from the boundary, with a sign for the side, is $d(\mathbf{x}) = z / \lVert\mathbf{w}\rVert = (\mathbf{w}^\top\mathbf{x} + b)/\lVert\mathbf{w}\rVert$. So $|z|$ grows with distance — a measure of confidence.`,
          String.raw`**The bias slides the boundary.** Its distance from the origin is $|b| / \lVert\mathbf{w}\rVert$. With $b = 0$ it must pass through the origin; changing $b$ slides it without turning it. Changing $\mathbf{w}$ turns it (and rescales $z$).`,
        ],
      },
      {
        t: "callout",
        kind: "key",
        title: "Why a bias is necessary",
        text: String.raw`Without a bias, input $(0,0)$ always gives $z = 0$, so with $\text{step}(0) = 1$ the neuron outputs 1 for $(0,0)$ whatever the weights. AND, OR and XOR all need 0 there, so none could be learned. In general, without $b$ every boundary passes through the origin, which removes one degree of freedom.`,
      },
      {
        t: "p",
        text: String.raw`Check with the weights the perceptron learns for AND below, $\mathbf{w} = (2, 1)$, $b = -3$: $\lVert\mathbf{w}\rVert = \sqrt{5} = 2.2361$; the boundary $2x_1 + x_2 = 3$ lies $3/2.2361 = 1.3416$ from the origin; the signed distances of $(0,0), (0,1), (1,0), (1,1)$ are $-1.3416, -0.8944, -0.4472, 0$. The last point sits exactly on the line — it gets class 1 only because of the tie-break $\text{step}(0) = 1$.`,
      },
      { t: "h", text: "The perceptron learning rule and why it makes sense" },
      { t: "math", tex: String.raw`\hat y = \text{step}(\mathbf{w}^\top\mathbf{x} + b), \qquad e = y - \hat y, \qquad \mathbf{w} \leftarrow \mathbf{w} + \eta\, e\, \mathbf{x}, \qquad b \leftarrow b + \eta\, e` },
      {
        t: "p",
        text: String.raw`$y \in \{0,1\}$ is the target, $\hat y$ the prediction, $e$ the error, $\eta > 0$ the learning rate. Does an update help? Recompute $z$ on the same example after updating:`,
      },
      { t: "math", tex: String.raw`z_{\text{new}} = (\mathbf{w} + \eta e \mathbf{x})^\top \mathbf{x} + (b + \eta e) = z + \eta\, e\,\big(\lVert\mathbf{x}\rVert^2 + 1\big)` },
      {
        t: "p",
        text: String.raw`The bracket $\lVert\mathbf{x}\rVert^2 + 1$ is always positive, so $z$ moves in the direction of the error: a false negative ($e = +1$) raises $z$, a false positive ($e = -1$) lowers it. One update may not fix the example (if $\eta$ is small), but it always moves $z$ toward the correct side.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "The learning rate does not matter when starting from zero",
        text: String.raw`From $\mathbf{w} = \mathbf{0}$, $b = 0$, every weight vector visited is $\eta$ times the one visited with $\eta = 1$. Scaling by a positive number never flips a sign, so every prediction and the number of epochs are identical — in exact arithmetic; with values like 0.1, rounding can flip exact ties $z = 0$. $\eta$ truly matters only when the initial weights are non-zero.`,
      },
      { t: "h", text: "Fully worked example: learning AND by hand" },
      {
        t: "p",
        text: String.raw`Data in order $(0,0)\to0$, $(0,1)\to0$, $(1,0)\to0$, $(1,1)\to1$; start from $\mathbf{w} = (0,0)$, $b = 0$, $\eta = 1$; predict 1 iff $z \ge 0$. Each row shows the example, the forward pass, the error, the update and the parameters **after** the update. (Every number was generated by running the rule in Python.)`,
      },
      {
        t: "table",
        head: ["#", "Epoch", "x", "y", "z", "ŷ", "e", "Δw", "Δb", "w after", "b after"],
        rows: [
          ["1", "1", "(0, 0)", "0", "0", "1", "−1", "(0, 0)", "−1", "(0, 0)", "−1"],
          ["2", "1", "(0, 1)", "0", "−1", "0", "0", "(0, 0)", "0", "(0, 0)", "−1"],
          ["3", "1", "(1, 0)", "0", "−1", "0", "0", "(0, 0)", "0", "(0, 0)", "−1"],
          ["4", "1", "(1, 1)", "1", "−1", "0", "+1", "(1, 1)", "+1", "(1, 1)", "0"],
          ["5", "2", "(0, 0)", "0", "0", "1", "−1", "(0, 0)", "−1", "(1, 1)", "−1"],
          ["6", "2", "(0, 1)", "0", "0", "1", "−1", "(0, −1)", "−1", "(1, 0)", "−2"],
          ["7", "2", "(1, 0)", "0", "−1", "0", "0", "(0, 0)", "0", "(1, 0)", "−2"],
          ["8", "2", "(1, 1)", "1", "−1", "0", "+1", "(1, 1)", "+1", "(2, 1)", "−1"],
          ["9", "3", "(0, 0)", "0", "−1", "0", "0", "(0, 0)", "0", "(2, 1)", "−1"],
          ["10", "3", "(0, 1)", "0", "0", "1", "−1", "(0, −1)", "−1", "(2, 0)", "−2"],
          ["11", "3", "(1, 0)", "0", "0", "1", "−1", "(−1, 0)", "−1", "(1, 0)", "−3"],
          ["12", "3", "(1, 1)", "1", "−2", "0", "+1", "(1, 1)", "+1", "(2, 1)", "−2"],
          ["13", "4", "(0, 0)", "0", "−2", "0", "0", "(0, 0)", "0", "(2, 1)", "−2"],
          ["14", "4", "(0, 1)", "0", "−1", "0", "0", "(0, 0)", "0", "(2, 1)", "−2"],
          ["15", "4", "(1, 0)", "0", "0", "1", "−1", "(−1, 0)", "−1", "(1, 1)", "−3"],
          ["16", "4", "(1, 1)", "1", "−1", "0", "+1", "(1, 1)", "+1", "(2, 2)", "−2"],
          ["17", "5", "(0, 0)", "0", "−2", "0", "0", "(0, 0)", "0", "(2, 2)", "−2"],
          ["18", "5", "(0, 1)", "0", "0", "1", "−1", "(0, −1)", "−1", "(2, 1)", "−3"],
          ["19", "5", "(1, 0)", "0", "−1", "0", "0", "(0, 0)", "0", "(2, 1)", "−3"],
          ["20", "5", "(1, 1)", "1", "0", "1", "0", "(0, 0)", "0", "(2, 1)", "−3"],
          ["21", "6", "(0, 0)", "0", "−3", "0", "0", "(0, 0)", "0", "(2, 1)", "−3"],
          ["22", "6", "(0, 1)", "0", "−2", "0", "0", "(0, 0)", "0", "(2, 1)", "−3"],
          ["23", "6", "(1, 0)", "0", "−1", "0", "0", "(0, 0)", "0", "(2, 1)", "−3"],
          ["24", "6", "(1, 1)", "1", "0", "1", "0", "(0, 0)", "0", "(2, 1)", "−3"],
        ],
      },
      {
        t: "p",
        text: String.raw`Mistakes per epoch: 2, 3, 3, 2, 1, 0. Epoch 6 is the first clean pass, so the perceptron **converges in epoch 6** with $\mathbf{w} = (2, 1)$, $b = -3$: the boundary $2x_1 + x_2 = 3$. Two rows in detail:`,
      },
      {
        t: "list",
        items: [
          String.raw`**Row 1:** $z = 0\cdot0 + 0\cdot0 + 0 = 0$, so $\hat y = 1$ but $y = 0$: $e = -1$. Since $\mathbf{x} = (0,0)$ only the bias changes: $b = 0 + 1\cdot(-1) = -1$.`,
          String.raw`**Row 4:** $z = 0\cdot1 + 0\cdot1 - 1 = -1 < 0$, so $\hat y = 0$ but $y = 1$: $e = +1$, $\Delta\mathbf{w} = 1\cdot1\cdot(1,1) = (1,1)$, $\Delta b = +1$, giving $\mathbf{w} = (1,1)$, $b = 0$.`,
        ],
      },
      {
        t: "callout",
        kind: "tip",
        text: "Rebuild this table in the simulator below: choose AND, leave the sliders at their defaults (w1 = 0, w2 = 0, b = 0, η = 1) and press Step 24 times.",
      },
      { t: "h", text: "Convergence on linearly separable data" },
      {
        t: "p",
        text: String.raw`Data is **linearly separable** if some hyperplane puts all class-1 points strictly on one side and all class-0 points strictly on the other. Use augmented vectors and relabel targets as $s = 2y - 1 \in \{-1,+1\}$. Suppose a unit vector $\mathbf{u}$ ($\lVert\mathbf{u}\rVert = 1$) separates the data with a safety gap, the **margin** $\gamma > 0$: $s^{(k)}\,\mathbf{u}^\top\tilde{\mathbf{x}}^{(k)} \ge \gamma$ for every example. Let $R = \max_k \lVert\tilde{\mathbf{x}}^{(k)}\rVert$ (how far the data spreads).`,
      },
      { t: "math", tex: String.raw`\text{number of updates} \;\le\; \left(\frac{R}{\gamma}\right)^2`, caption: "The perceptron convergence theorem (Block 1962; Novikoff 1962), starting from zero weights." },
      {
        t: "p",
        text: String.raw`In plain words: a wide gap means few corrections, widely spread data means more — but the number is always finite.`,
      },
      {
        t: "details",
        summary: "Proof sketch",
        blocks: [
          {
            t: "p",
            text: String.raw`Each update on a mistake is $\tilde{\mathbf{w}} \leftarrow \tilde{\mathbf{w}} + \eta s \tilde{\mathbf{x}}$ (for $y \in \{0,1\}$, $e$ equals $s$ on a mistake). After $m$ updates from $\tilde{\mathbf{w}} = \mathbf{0}$, the weights grow fast along $\mathbf{u}$ but slowly in length — which clashes if $m$ is large:`,
          },
          {
            t: "list",
            items: [
              String.raw`**Progress along $\mathbf{u}$:** each update adds $\eta s\,\mathbf{u}^\top\tilde{\mathbf{x}} \ge \eta\gamma$, so $\mathbf{u}^\top\tilde{\mathbf{w}}_m \ge m\eta\gamma$.`,
              String.raw`**Slow growth of the length:** $\lVert\tilde{\mathbf{w}} + \eta s\tilde{\mathbf{x}}\rVert^2 = \lVert\tilde{\mathbf{w}}\rVert^2 + 2\eta s\,\tilde{\mathbf{w}}^\top\tilde{\mathbf{x}} + \eta^2\lVert\tilde{\mathbf{x}}\rVert^2$, and the middle term is $\le 0$ because the example was misclassified. So $\lVert\tilde{\mathbf{w}}_m\rVert^2 \le m\eta^2R^2$.`,
              String.raw`**Combine with Cauchy–Schwarz** (a dot product with a unit vector is at most the other vector's length): $m\eta\gamma \le \mathbf{u}^\top\tilde{\mathbf{w}}_m \le \lVert\tilde{\mathbf{w}}_m\rVert \le \sqrt{m}\,\eta R$, so $\sqrt{m} \le R/\gamma$, i.e. $m \le (R/\gamma)^2$.`,
            ],
          },
          {
            t: "p",
            text: String.raw`$\eta$ cancels, matching the observation that the learning rate is irrelevant from zero initialisation. Wide margins mean fast learning; tiny margins mean many updates.`,
          },
        ],
      },
      {
        t: "p",
        text: "The theorem promises a finite number of mistakes, not a good line: the perceptron stops at the first one that works, often one that barely clears some points (in the AND example, (1, 1) lies exactly on the boundary). If the data is **not** separable, the theorem does not apply and the weights never settle.",
      },
      { t: "h", text: "The XOR problem: a proof of impossibility" },
      {
        t: "p",
        text: String.raw`XOR outputs 1 when exactly one input is 1: $(0,0)\to0$, $(0,1)\to1$, $(1,0)\to1$, $(1,1)\to0$. Suppose one step neuron computed it. Each example gives a condition on $z$:`,
      },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
(0,0)\to 0:&\quad b < 0 \\
(0,1)\to 1:&\quad w_2 + b \ge 0 \\
(1,0)\to 1:&\quad w_1 + b \ge 0 \\
(1,1)\to 0:&\quad w_1 + w_2 + b < 0
\end{aligned}`,
      },
      {
        t: "p",
        text: String.raw`Add the middle two: $w_1 + w_2 + 2b \ge 0$, so $w_1 + w_2 + b \ge -b$. The first line says $-b > 0$, hence $w_1 + w_2 + b > 0$ — contradicting the last line. No weights exist: **XOR is not linearly separable**, so the perceptron rule cannot converge on it. In the picture, the class-1 points sit on one diagonal of the unit square and the class-0 points on the other — no line separates the diagonals.`,
      },
      {
        t: "p",
        text: String.raw`In practice, from $\mathbf{w} = (0,0)$, $b = 0$, $\eta = 1$ (order above), the mistakes per epoch are 3, 3, 4, 4, 4, … and from epoch 2 on the parameters return to exactly $\mathbf{w} = (-1, 0)$, $b = 0$ at the end of every epoch: it cycles forever. The **perceptron cycling theorem** predicts this: on non-separable data with bounded inputs the weights stay bounded and (for integer data) revisit the same states.`,
      },
      { t: "h", text: "How a hidden layer solves XOR" },
      {
        t: "p",
        text: String.raw`XOR is 'at least one on' AND 'not both on': $\text{XOR}(x_1,x_2) = \text{AND}\big(\text{OR}(x_1,x_2),\ \text{NAND}(x_1,x_2)\big)$, where NAND ('not AND') outputs 0 only when both inputs are 1. OR, NAND and AND are each linearly separable, so each is one neuron. Put OR and NAND in a **hidden layer** and feed them to an AND neuron:`,
      },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
h_1 &= \text{step}(x_1 + x_2 - 0.5) &&\text{(OR)}\\
h_2 &= \text{step}(-x_1 - x_2 + 1.5) &&\text{(NAND)}\\
\hat y &= \text{step}(h_1 + h_2 - 1.5) &&\text{(AND)}
\end{aligned}`,
      },
      {
        t: "table",
        head: ["x", "h₁ = OR", "h₂ = NAND", "ŷ = AND(h₁, h₂)", "XOR target"],
        rows: [
          ["(0, 0)", "0", "1", "0", "0"],
          ["(0, 1)", "1", "1", "1", "1"],
          ["(1, 0)", "1", "1", "1", "1"],
          ["(1, 1)", "1", "0", "0", "0"],
        ],
      },
      {
        t: "p",
        text: String.raw`The hidden layer **re-represents** the input. In the $(x_1,x_2)$ plane the classes are tangled; in the $(h_1,h_2)$ plane the four inputs land on three points — $(0,1)$ for class-0 input $(0,0)$, $(1,1)$ for both class-1 inputs, $(1,0)$ for class-0 input $(1,1)$ — and $(1,1)$ alone is easily cut off by a line. That is the core idea of deep learning: layers reshape data until the final layer can split it with a line.`,
      },
      {
        t: "callout",
        kind: "warning",
        title: "These weights were set by hand",
        text: "The perceptron rule cannot train hidden units, because it has no target for them. Training hidden layers needed smooth activations and [backpropagation](/learn/m04/backpropagation).",
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "neuron", caption: "A two-input neuron: live weighted sum, decision boundary, and the perceptron rule stepping through AND, OR, NAND and XOR." },
      { t: "h", text: "What you are looking at" },
      {
        t: "list",
        items: [
          "**Sliders** set $w_1$, $w_2$ and $b$ (each from −3 to 3) and the learning rate η (0.1 to 1, default 1). Moving any slider restarts training from the slider values.",
          "**The plot** shows inputs $x_1, x_2 \\in [-0.5, 1.5]$ with the four points coloured by target class. The line is the boundary $w_1x_1 + w_2x_2 + b = 0$; the shaded side is where the neuron predicts 1; the arrow is the weight vector (at a right angle to the line, pointing into the shaded side); wrongly classified points are ringed.",
          "**The weighted-sum panel** writes out the sum for the current sample, e.g. z = 0·1 + 0·0 + 0 = 0, then the activation.",
          "**The activation selector** (step, sigmoid, tanh, ReLU, linear) changes only the displayed output $a = f(z)$. Classification always uses $z \\ge 0 \\Rightarrow$ class 1, so the line does not move when you switch.",
          "**Play / Step** apply the perceptron rule to one sample at a time, in the order (0,0), (0,1), (1,0), (1,1), showing x, y, z, ŷ, e = y − ŷ, Δw = η·e·x, Δb = η·e and the new parameters as the line moves, plus an epoch counter and the epoch's error count.",
        ],
      },
      { t: "h", text: "Things to try" },
      {
        t: "steps",
        items: [
          { title: "Reproduce the hand example", text: "Choose AND, keep the defaults and Step through, comparing with the 24-row table in *The Math, Step by Step*. It converges in epoch 6 with w = (2, 1), b = −3; (1, 1) sits exactly on the final line." },
          { title: "Watch only the bias move", text: "On step 1, x = (0, 0) is wrong but Δw = (0, 0), so the line only slides as b changes." },
          { title: "Compare the gates", text: "Train OR (epoch 4 from the defaults) and NAND (epoch 6). NAND ends at w = (−2, −1), b = +2: the same line as AND with the arrow reversed." },
          { title: "Set a boundary by hand", text: "With AND selected, set w1 = 1, w2 = 1, b = −1.5 before training. All four points are already right, so Play gives a clean first epoch. Now try b = −0.5: you have built OR." },
          { title: "Learning rate from zero", text: "With all sliders at 0, set η = 0.5 and train AND. Every Δw, Δb and z is halved, but the predictions are identical and it still converges in epoch 6, at w = (1, 0.5), b = −1.5. Now set w1 = 1 (others 0): η = 1 converges in epoch 6, η = 0.5 in epoch 5. With a non-zero start, η matters." },
          { title: "A floating-point surprise", text: "From zero, try η = 0.1 on AND. In exact arithmetic it would match η = 1, but the trace has many exact ties z = 0 and 0.1 is not exact in binary (0.1 + 0.1 + 0.1 − 0.3 is not 0). Some ties tip the other way, and in double precision this run converges in epoch 4 instead of 6. Tie-breaking at z = 0 is a small but real weak spot." },
          { title: "Break it with XOR", text: "Select XOR and press Play. Errors never reach 0; from epoch 2 the parameters return to w = (−1, 0), b = 0 every epoch, and the simulator stops at its 30-epoch cap reporting no convergence. Try placing the line by hand: you can never get all four points right." },
          { title: "Inspect activations", text: "Switch between step, sigmoid and tanh. The output changes (e.g. 1 vs 0.5 vs 0 when z = 0), but the predicted class and the line stay the same." },
        ],
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "Everything in NumPy, in four parts: one neuron's forward pass, a `Perceptron` class trained online on the four logic gates, a larger random separable problem, and the hand-wired two-layer XOR network.",
      },
      {
        t: "code",
        lang: "python",
        title: "perceptron_from_scratch.py",
        runnable: true,
        code: String.raw`import numpy as np

# ---------- 1. A single artificial neuron ----------
def neuron(x, w, b, activation):
    """Weighted sum z = w . x + b, then a = activation(z)."""
    z = np.dot(w, x) + b
    return z, activation(z)

step = lambda z: np.where(z >= 0, 1, 0)        # Heaviside step, threshold at z = 0
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))

x = np.array([2.0, -1.0, 0.5])
w = np.array([0.4, 0.3, -0.6])
b = 0.1
z, a = neuron(x, w, b, sigmoid)
print(f"z = {z:.4f}, sigmoid(z) = {a:.4f}, step(z) = {step(z)}")

# ---------- 2. The perceptron learning rule ----------
class Perceptron:
    def __init__(self, n_features, lr=1.0):
        self.w = np.zeros(n_features)   # start at zero (as in the hand example)
        self.b = 0.0
        self.lr = lr                     # learning rate eta

    def predict(self, X):
        return step(X @ self.w + self.b)

    def fit(self, X, y, max_epochs=30):
        """Online training: one sample at a time. Returns epochs used, or None."""
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for xi, yi in zip(X, y):
                e = yi - step(xi @ self.w + self.b)    # error in {-1, 0, +1}
                if e != 0:
                    self.w += self.lr * e * xi         # delta w = eta * e * x
                    self.b += self.lr * e              # delta b = eta * e
                    errors += 1
            if errors == 0:                            # a full clean pass
                return epoch
        return None

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
gates = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1],
         "NAND": [1, 1, 1, 0], "XOR": [0, 1, 1, 0]}

for name, targets in gates.items():
    y = np.array(targets)
    p = Perceptron(2, lr=1.0)
    ep = p.fit(X, y)
    status = f"converged in epoch {ep}" if ep else "did NOT converge in 30 epochs"
    print(f"{name:4s}: {status:30s} w={p.w}, b={p.b:+.0f}, "
          f"predictions={p.predict(X)}, targets={y}")

# ---------- 3. A larger linearly separable problem ----------
rng = np.random.default_rng(0)
n = 200
Xr = rng.uniform(-1, 1, size=(n, 2))
true_w, true_b = np.array([1.5, -2.0]), 0.3          # hidden "teacher" line
margin = Xr @ true_w + true_b
keep = np.abs(margin) > 0.2                          # leave a gap -> positive margin
Xr, yr = Xr[keep], (margin[keep] >= 0).astype(int)
p = Perceptron(2, lr=0.1)
ep = p.fit(Xr, yr, max_epochs=100)
acc = np.mean(p.predict(Xr) == yr)
print(f"\nrandom data: {len(yr)} points, converged in epoch {ep}, "
      f"training accuracy = {acc:.3f}")
print("learned direction:", np.round(p.w / np.linalg.norm(p.w), 3),
      " teacher direction:", np.round(true_w / np.linalg.norm(true_w), 3))

# ---------- 4. XOR with one hidden layer (hand-set weights) ----------
h_or   = step(X @ np.array([1, 1]) - 0.5)      # hidden unit 1 computes OR
h_nand = step(X @ np.array([-1, -1]) + 1.5)    # hidden unit 2 computes NAND
out    = step(h_or + h_nand - 1.5)             # output unit computes AND
print("\nXOR via hidden layer:")
for xi, a1, a2, o in zip(X, h_or, h_nand, out):
    print(f"  x={xi.astype(int)}  OR={a1}  NAND={a2}  -> AND = {o}")`,
        output: String.raw`z = 0.3000, sigmoid(z) = 0.5744, step(z) = 1
AND : converged in epoch 6           w=[2. 1.], b=-3, predictions=[0 0 0 1], targets=[0 0 0 1]
OR  : converged in epoch 4           w=[1. 1.], b=-1, predictions=[0 1 1 1], targets=[0 1 1 1]
NAND: converged in epoch 6           w=[-2. -1.], b=+2, predictions=[1 1 1 0], targets=[1 1 1 0]
XOR : did NOT converge in 30 epochs  w=[-1.  0.], b=+0, predictions=[1 1 0 0], targets=[0 1 1 0]

random data: 183 points, converged in epoch 6, training accuracy = 1.000
learned direction: [ 0.633 -0.774]  teacher direction: [ 0.6 -0.8]

XOR via hidden layer:
  x=[0 0]  OR=0  NAND=1  -> AND = 0
  x=[0 1]  OR=1  NAND=1  -> AND = 1
  x=[1 0]  OR=1  NAND=1  -> AND = 1
  x=[1 1]  OR=1  NAND=0  -> AND = 0`,
      },
      {
        t: "list",
        items: [
          "The gates match the hand trace: AND converges in epoch 6 with w = (2, 1), b = −3.",
          "On XOR the final weights (−1, 0), b = 0 get only 2 of 4 points right — they are just where the weights were when the 30-epoch cap hit.",
          "On the random data, the 0.2 gap around the teacher line guarantees a positive margin, so the theorem applies. The learned direction is close to the teacher's but not equal: the perceptron finds *a* separating line, not the true one.",
          "The XOR network is not trained; it only shows the weights exist. Finding them automatically is backpropagation's job.",
        ],
      },
    ],

    // -------------------------------------------------------------- framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "**Part A** uses `nn.Linear(2, 1)` for the weighted sum and applies the perceptron rule by hand inside `torch.no_grad()`. Autograd (PyTorch's automatic gradients) can't help: the step function is flat almost everywhere, so its gradient is zero. **Part B** shows that one linear unit trained by gradient descent cannot fit XOR, while a 2-4-1 network (4 tanh hidden units) can.",
              },
              {
                t: "code",
                lang: "python",
                title: "perceptron_torch.py",
                code: String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

# ---------- Part A: a single neuron = nn.Linear(2, 1) + step ----------
# We train it with the perceptron rule (no autograd: the step has no useful gradient).
neuron = nn.Linear(2, 1)
with torch.no_grad():
    neuron.weight.zero_()
    neuron.bias.zero_()
y_and = torch.tensor([0., 0., 0., 1.])
eta = 1.0
for epoch in range(1, 31):
    errors = 0
    for xi, yi in zip(X, y_and):
        with torch.no_grad():
            y_hat = (neuron(xi) >= 0).float()          # step(z)
            e = yi - y_hat                             # error
            neuron.weight += eta * e * xi              # delta w = eta * e * x
            neuron.bias += eta * e                     # delta b = eta * e
            errors += int(e.abs().item() > 0)
    if errors == 0:
        break
print(f"AND perceptron: converged in epoch {epoch}, "
      f"w={neuron.weight.data.tolist()}, b={neuron.bias.item():+.1f}")

# ---------- Part B: XOR needs a hidden layer ----------
y_xor = torch.tensor([[0.], [1.], [1.], [0.]])

def train(model, epochs=2000, lr=0.5):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()          # sigmoid + binary cross-entropy
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y_xor)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)) >= 0.5).int().flatten().tolist()
    return loss.item(), pred

single = nn.Linear(2, 1)                                     # no hidden layer
mlp = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))  # 4 hidden units

for name, model in [("single neuron", single), ("2-4-1 MLP", mlp)]:
    loss, pred = train(model)
    print(f"{name:13s}: final loss = {loss:.4f}, predictions = {pred}, targets = [0, 1, 1, 0]")`,
                output: String.raw`AND perceptron: converged in epoch 6, w=[[2.0, 1.0]], b=-3.0
single neuron: final loss = 0.6931, predictions = [1, 1, 1, 1], targets = [0, 1, 1, 0]
2-4-1 MLP    : final loss = 0.0027, predictions = [0, 1, 1, 0], targets = [0, 1, 1, 0]`,
              },
              {
                t: "p",
                text: String.raw`The single unit ends at loss $0.6931 = \ln 2$: on XOR, the best a linear unit can do under binary cross-entropy is output probability 0.5 for every input — a shrug, which the 0.5 threshold then labels 1. The hidden layer breaks this ceiling.`,
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras has no perceptron rule, so this compares a single `Dense(1)` unit with a small hidden-layer network on XOR, both trained by gradient descent on binary cross-entropy.",
              },
              {
                t: "code",
                lang: "python",
                title: "xor_keras.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")

def build(hidden_units):
    layers = [tf.keras.Input(shape=(2,))]
    if hidden_units:
        layers.append(tf.keras.layers.Dense(hidden_units, activation="tanh"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

for name, hidden in [("single neuron", 0), ("2-4-1 MLP", 4)]:
    model = build(hidden)
    model.fit(X, y, epochs=2000, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    pred = (model.predict(X, verbose=0) >= 0.5).astype(int).ravel()
    print(f"{name:13s}: loss={loss:.4f}, accuracy={acc:.2f}, predictions={pred.tolist()}")`,
                output: "Prints one line per model: its final loss, accuracy and predictions. The single neuron's loss stays near ln 2 ≈ 0.693 and its accuracy cannot exceed 0.75 (at most three of the four XOR points can be on the correct side of a line); the 2-4-1 network should reach accuracy 1.00 with predictions [0, 1, 1, 0] (exact numbers depend on the TensorFlow version and initialisation).",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      {
        t: "p",
        text: "The basic neuron has many relatives. They differ in input type, activation, update rule, and how they handle data no line can split.",
      },
      {
        t: "table",
        head: ["Variant", "What changes", "Good at", "Weak at", "Typical use"],
        rows: [
          ["McCulloch–Pitts unit (1943)", "0/1 inputs, fixed threshold, inhibitory veto, no learning", "Networks provably compute any Boolean function; easy to analyse", "Thresholds set by hand; cannot learn", "Theory of computation; historical foundation"],
          ["Rosenblatt perceptron (1958)", "Real weights + bias, step activation, error-correction rule", "Simple, online, guaranteed to converge on separable data", "Never settles on non-separable data; no confidence; stops at any separating line", "Teaching; linear baselines"],
          ["Pocket perceptron", "Keeps ('pockets') the best weights seen so far", "Usable result on non-separable data", "Extra evaluation cost; still linear", "Noisy linear problems"],
          ["Averaged / voted perceptron", "Predicts with the average (or weighted vote) of all weight vectors visited", "Much more stable; generalises well; still online", "Must store an average or many vectors", "Online NLP tagging and parsing (structured perceptron)"],
          ["ADALINE (Widrow & Hoff, 1960)", "Learns from the error on the raw z (least mean squares); threshold only to predict", "Smooth squared error; reaches a least-squares solution even when not separable", "Squared error is not ideal for classification", "Adaptive filters, signal processing"],
          ["Sigmoid neuron (logistic regression)", "Sigmoid activation, gradient descent on cross-entropy", "Outputs probabilities; differentiable, so it can be stacked and trained with backprop", "Saturates (flattens) for large |z|, so gradients get small", "Binary classification output layers"],
          ["ReLU / GELU neuron", "Modern activations, trained by backprop in deep networks", "Cheap; does not saturate for positive z; the workhorse of deep nets", "Unbounded output; ReLU units can 'die' (stuck at 0)", "Hidden layers of MLPs, CNNs, Transformers"],
          ["Multiclass perceptron", "One weight vector per class; predict the highest score; on a mistake add x to the true class, subtract from the predicted one", "Natural K-class extension with the same guarantees", "Still linear decision regions", "Multiclass linear classification"],
          ["Kernel perceptron", "Replaces dot products with a kernel k(x, x'), another similarity measure", "Curved boundaries without explicit hidden layers", "Cost grows with the number of stored mistakes", "Small non-linear problems; precursor of SVM ideas"],
        ],
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Every modern neural network.** A fully connected layer with $m$ outputs is $m$ neurons sharing an input; a convolutional filter is one neuron applied at every image position; each Transformer feed-forward block is two layers of neurons. One neuron is the atom of them all.",
          "**Historical hardware.** Rosenblatt's group built the Mark I Perceptron: a grid of photocells as input and motor-driven potentiometers (adjustable dials) as weights. It learned to classify simple visual patterns.",
          "**Fast linear classifiers.** Perceptron-style online linear models still help when data streams in and must be handled in one pass with tiny memory: spam scoring from word counts, click prediction with hashed features, and quick baselines before deeper models.",
          "**Natural language processing.** Before neural models, the averaged structured perceptron (Collins, 2002) was a standard, fast and simple method for part-of-speech tagging and parsing.",
          "**Output units.** A binary classifier's last layer is one sigmoid neuron; for many classes, a layer of neurons followed by softmax.",
          "**Interpretable baselines.** Each weight's sign and size say directly how a feature moves the decision, which makes linear models a common first step in regulated settings such as credit scoring.",
        ],
      },
    ],

    // -------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Very cheap:** one dot product to predict, one vector addition to learn. Memory is $O(n)$.",
          "**Online:** learns one example at a time, so it handles streaming data without holding the whole dataset.",
          "**Guaranteed convergence** on linearly separable data, at most $(R/\\gamma)^2$ mistakes — a bound that does not depend directly on the number of examples or the dimension.",
          "**Interpretable:** the weight vector is normal to the boundary; each weight's sign shows which way a feature pushes.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Straight lines only.** It cannot represent XOR, parity, or any classes a hyperplane cannot split.",
          "**No convergence on non-separable data.** Weights cycle, so the result depends on when you stop — and real data with label noise is almost never perfectly separable.",
          "**Any separator will do.** It stops at the first separating line, often with a tiny margin, which generalises worse than the maximum-margin line (the idea behind support vector machines).",
          "**No probabilities.** The step output shows no confidence; only $z$ tells a borderline case from a clear one.",
          "**Not differentiable.** The step's derivative is zero everywhere except at 0, where it is undefined. Gradient-based training — and so training hidden layers — needs smooth activations.",
          "**Order-sensitive.** Final weights and epoch count depend on the order of the examples.",
        ],
      },
      { t: "h", text: "Practical trade-offs" },
      {
        t: "table",
        head: ["Choice", "Option A", "Option B"],
        rows: [
          ["Activation", "Step: crisp decisions, perceptron rule, no gradients", "Sigmoid/tanh/ReLU: graded outputs, trainable with gradient descent and backprop"],
          ["Learning rule", "Perceptron: updates only on mistakes; fast; unstable on noisy data", "Gradient descent on a loss: updates on every example; converges on non-separable data"],
          ["Model size", "One neuron: interpretable, linear only", "Layers: curved boundaries, more data and compute, less interpretable"],
          ["Final weights", "Last weights: simple", "Averaged weights: more stable, usually more accurate"],
        ],
      },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting the bias.** The boundary must then pass through the origin, so for AND/OR on 0/1 inputs, (0, 0) can never be classified 0. **Fix:** always include a bias (frameworks do by default: `nn.Linear(..., bias=True)`), or add a constant-1 feature.",
          "**Updating the bias with x instead of 1.** `b += lr * e * x` is a shape error at best and a wrong rule at worst. **Why:** the bias is a weight on a constant input of 1, so its update is `lr * e`.",
          "**Inconsistent tie-breaking at z = 0.** Sources use step(0) = 0, step(0) = 1, or sign(0) = 0, and each changes the trace and convergence epoch. **Fix:** pick one rule and use it everywhere — this lesson uses z ≥ 0 → 1.",
          "**Mixing label conventions.** $\\Delta\\mathbf{w} = \\eta(y - \\hat y)\\mathbf{x}$ assumes $y, \\hat y \\in \\{0,1\\}$. With labels in $\\{-1,+1\\}$ the error is $\\pm 2$, or the rule is written $\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta y\\mathbf{x}$ on mistakes. Mixing them gives wrong updates. **Fix:** convert labels once at load time.",
          "**Expecting convergence on non-separable data.** Waiting for zero errors on XOR (or noisy real data) loops forever. **Fix:** cap the epochs and use the pocket or averaged perceptron, or a differentiable model.",
          "**Thinking the learning rate speeds up the perceptron.** From zero initialisation η only rescales the weights; in exact arithmetic predictions and epoch count are identical. **Fix:** tune η only for non-zero initial weights, and scale features instead.",
          "**Unscaled features.** Because $\\Delta\\mathbf{w} \\propto \\mathbf{x}$ (proportional to the input), a feature measured in thousands dominates every update, and the bound $(R/\\gamma)^2$ grows with $R$. **Fix:** standardise features (mean 0, variance 1).",
          "**Believing a neuron models a brain cell.** It is inspired by biology but is a simple algebraic function; 'the brain does X, so the network must too' is not a reliable design guide.",
          "**Treating the hand-built XOR network as 'learned'.** The perceptron rule cannot train hidden units — they have no target. **Fix:** train hidden layers with backpropagation on differentiable activations.",
        ],
      },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "What does a single artificial neuron compute?",
        a: "A weighted sum of its inputs plus a bias, $z = \\mathbf{w}^\\top\\mathbf{x} + b$, passed through an activation: $a = f(z)$. The weights and bias are learned; the activation sets the output's form (a class with a step, a probability with a sigmoid, a non-negative feature with ReLU).",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What is the role of the bias?",
        a: "It moves the decision boundary away from the origin: $\\mathbf{w}^\\top\\mathbf{x} + b = 0$ lies at distance $|b|/\\lVert\\mathbf{w}\\rVert$ from it. Without a bias every boundary passes through the origin, making even AND on 0/1 inputs impossible. It equals a firing threshold $\\theta = -b$, or a weight on a constant input of 1.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "Why can't a single perceptron learn XOR?",
        a: "Its boundary is one straight line (hyperplane), and XOR's classes sit on opposite diagonals of the unit square, so no line splits them. Algebraically, $b<0$, $w_2+b\\ge0$, $w_1+b\\ge0$, $w_1+w_2+b<0$ contradict each other: adding the middle two gives $w_1+w_2+b \\ge -b > 0$.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "State the perceptron learning rule and show that each update moves the prediction in the right direction.",
        a: "With $\\hat y = \\text{step}(z)$ and $e = y - \\hat y$: $\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta e\\mathbf{x}$, $b \\leftarrow b + \\eta e$. The new pre-activation on the same example is $z + \\eta e(\\lVert\\mathbf{x}\\rVert^2 + 1)$. The bracket is positive, so $z$ rises on a false negative ($e=+1$) and falls on a false positive ($e=-1$).",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why is the weight vector perpendicular to the decision boundary, and what does z tell you about a point?",
        a: "For two boundary points, $\\mathbf{w}^\\top\\mathbf{x}_1 + b = \\mathbf{w}^\\top\\mathbf{x}_2 + b = 0$, so $\\mathbf{w}^\\top(\\mathbf{x}_1 - \\mathbf{x}_2) = 0$: $\\mathbf{w}$ is orthogonal to every direction along the boundary. $z/\\lVert\\mathbf{w}\\rVert$ is the signed perpendicular distance from the boundary — positive on the side $\\mathbf{w}$ points to — so $|z|$ shows how far, and how confidently, a point is classified.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "How does the perceptron differ from logistic regression?",
        a: "Both compute $z = \\mathbf{w}^\\top\\mathbf{x} + b$ and draw the same kind of linear boundary. The perceptron uses a step, updates only on mistakes, gives no probabilities and does not converge on non-separable data. Logistic regression uses a sigmoid, outputs probabilities, and minimises cross-entropy by gradient descent, updating on every example in proportion to $y - \\sigma(z)$; it reaches a well-defined optimum (with regularisation) even on non-separable data. Its update, $\\eta(y - \\sigma(z))\\mathbf{x}$, is exactly the perceptron's form with the hard prediction replaced by a soft one.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "State and sketch the proof of the perceptron convergence theorem.",
        a: "If (augmented) data satisfy $s^{(k)}\\mathbf{u}^\\top\\tilde{\\mathbf{x}}^{(k)} \\ge \\gamma > 0$ for some unit $\\mathbf{u}$ and $\\lVert\\tilde{\\mathbf{x}}^{(k)}\\rVert \\le R$, the perceptron started at zero makes at most $(R/\\gamma)^2$ updates. Proof: each update raises $\\mathbf{u}^\\top\\tilde{\\mathbf{w}}$ by at least $\\eta\\gamma$, so after $m$ updates it is $\\ge m\\eta\\gamma$. Each raises $\\lVert\\tilde{\\mathbf{w}}\\rVert^2$ by at most $\\eta^2R^2$, because the cross term is non-positive on a mistake, so $\\lVert\\tilde{\\mathbf{w}}\\rVert \\le \\sqrt{m}\\eta R$. Cauchy–Schwarz gives $m\\eta\\gamma \\le \\sqrt{m}\\eta R$, i.e. $m \\le (R/\\gamma)^2$ — independent of $\\eta$ and of the number of examples.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Explain how a hidden layer makes XOR solvable, and why the perceptron rule cannot train that hidden layer.",
        a: "Hidden units compute new features. With $h_1 = \\text{OR}(x_1,x_2)$ and $h_2 = \\text{NAND}(x_1,x_2)$, the four inputs map to $(0,1), (1,1), (1,1), (1,0)$, where the class-1 point $(1,1)$ is cut off by the AND line $h_1 + h_2 = 1.5$: separable at the last layer though not at the input. The perceptron rule needs a target for each unit it updates; hidden units have none, and with step activations the output is piecewise constant in the hidden weights, so no gradient tells them how to change. Smooth activations plus backpropagation fix this, giving each hidden unit a share of the output error via the chain rule.",
      },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "Comparing brain and artificial neurons, what plays the role of synapse strength?",
            options: ["The activation function", "The weight", "The input value", "The learning rate"],
            answer: 1,
            explain: "Synapse strength sets how much an incoming signal affects the cell — exactly what a weight does to its input.",
          },
          {
            kind: "numeric",
            q: "A neuron has w = (0.5, −0.25), b = 0.2 and receives x = (1, 2). What is the weighted sum z?",
            answer: 0.2,
            tolerance: 0.001,
            explain: "z = 0.5·1 + (−0.25)·2 + 0.2 = 0.5 − 0.5 + 0.2 = 0.2.",
          },
          {
            q: "A neuron's decision boundary is where w·x + b = 0. Increasing b while keeping w fixed…",
            options: [
              "rotates the boundary toward w",
              "slides the boundary parallel to itself",
              "has no effect on the boundary",
              "makes the boundary curved",
            ],
            answer: 1,
            explain: "w, the boundary's normal, is unchanged, so the boundary keeps its angle; only its distance |b|/‖w‖ from the origin changes.",
          },
          {
            kind: "numeric",
            q: "How far is the boundary 3x₁ + 4x₂ − 10 = 0 from the origin?",
            answer: 2,
            tolerance: 0.001,
            explain: "Here w = (3, 4) and b = −10. Distance = |b| / ‖w‖ = 10 / √(3² + 4²) = 10 / 5 = 2.",
          },
          {
            q: "A perceptron with labels 0 or 1 predicts ŷ = 1, but the true label is y = 0. Which update does it apply (η > 0)?",
            options: ["w ← w + ηx, b ← b + η", "w ← w − ηx, b ← b − η", "w ← w − ηx, b unchanged", "No update"],
            answer: 1,
            explain: "e = y − ŷ = −1, so Δw = −ηx and Δb = −η: both push z down, which it needs because z was too high.",
          },
          {
            kind: "numeric",
            q: "Linearly separable data has R = 2 and margin γ = 0.5 (augmented, unit-length separator). What is the convergence theorem's upper bound on the number of updates?",
            answer: 16,
            tolerance: 0.001,
            explain: "(R/γ)² = (2/0.5)² = 4² = 16.",
          },
          {
            q: "Which two-input logic function can a single perceptron NOT represent?",
            options: ["AND", "NAND", "OR", "XOR"],
            answer: 3,
            explain: "AND, NAND and OR can each be split by one line; XOR's classes sit on opposite diagonals of the unit square.",
          },
          {
            q: "Starting from w = 0 and b = 0, what happens to the sequence of predictions if you change η from 1 to 0.1?",
            options: [
              "It converges 10 times slower",
              "It converges 10 times faster",
              "It is identical — the weights are just scaled by 0.1",
              "It may no longer converge",
            ],
            answer: 2,
            explain: "Every weight vector visited is 0.1 times the η = 1 one. Positive scaling never flips a sign, so every prediction is unchanged (in exact arithmetic — rounding can occasionally flip an exact tie at z = 0).",
          },
          {
            kind: "numeric",
            q: "A McCulloch–Pitts unit fires if at least θ excitatory inputs are active. What θ makes a two-input OR?",
            answer: 1,
            tolerance: 0,
            explain: "OR fires when at least one input is on, so θ = 1. (AND uses θ = 2.)",
          },
        ],
      },
      {
        t: "exercise",
        title: "One perceptron update by hand",
        prompt: String.raw`A perceptron has $\mathbf{w} = (1, -2)$, $b = 0.5$, $\eta = 0.5$, and predicts $\hat y = 1$ iff $z \ge 0$. It gets $\mathbf{x} = (1, 1)$ with target $y = 1$. Compute $z$, $\hat y$, $e$, $\Delta\mathbf{w}$, $\Delta b$, the new parameters, and the new $z$ on the same example. Then check $z_{\text{new}} = z + \eta e(\lVert\mathbf{x}\rVert^2 + 1)$.`,
        hint: "Compute the weighted sum first; weights change only if the prediction is wrong.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Forward pass", tex: String.raw`z = 1\cdot1 + (-2)\cdot1 + 0.5 = -0.5 \;\Rightarrow\; \hat y = 0` },
              { title: "Error", tex: String.raw`e = y - \hat y = 1 - 0 = 1` },
              { title: "Update", tex: String.raw`\Delta\mathbf{w} = 0.5\cdot1\cdot(1,1) = (0.5, 0.5),\quad \Delta b = 0.5 \;\Rightarrow\; \mathbf{w} = (1.5, -1.5),\; b = 1.0` },
              { title: "New z", tex: String.raw`z_{\text{new}} = 1.5 - 1.5 + 1.0 = 1.0 \;\Rightarrow\; \hat y = 1 \text{ (now correct)}` },
              { title: "Check the formula", tex: String.raw`z + \eta e(\lVert\mathbf{x}\rVert^2 + 1) = -0.5 + 0.5\cdot1\cdot(2 + 1) = 1.0 \;\checkmark` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Coding: majority vs parity in three dimensions",
        prompt: "Using only NumPy, train a perceptron (zero start, η = 1, predict 1 iff z ≥ 0, at most 50 epochs) on all 8 three-bit inputs for two targets: **majority** (1 if at least two inputs are 1) and **parity** (1 if an odd number are 1). For each, report the convergence epoch (or None), total mistakes, final weights and training accuracy, and explain the difference.",
        hint: "itertools.product([0, 1], repeat=3) lists the inputs. Three-bit parity is XOR extended to three inputs.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "majority_vs_parity.py",
            runnable: true,
            code: String.raw`import itertools
import numpy as np

X = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)  # all 8 inputs
majority = (X.sum(axis=1) >= 2).astype(int)   # 1 if at least two inputs are 1
parity = (X.sum(axis=1) % 2).astype(int)      # 1 if an odd number of inputs are 1

def train_perceptron(X, y, lr=1.0, max_epochs=50):
    w, b = np.zeros(X.shape[1]), 0.0
    mistakes = 0
    for epoch in range(1, max_epochs + 1):
        errors = 0
        for xi, yi in zip(X, y):
            e = yi - int(xi @ w + b >= 0)
            if e:
                w += lr * e * xi
                b += lr * e
                errors += 1
        mistakes += errors
        if errors == 0:
            return w, b, epoch, mistakes
    return w, b, None, mistakes

for name, y in [("majority", majority), ("parity", parity)]:
    w, b, ep, m = train_perceptron(X, y)
    acc = np.mean((X @ w + b >= 0).astype(int) == y)
    print(f"{name:8s}: epoch={ep}, total mistakes={m}, w={w}, b={b:+.0f}, accuracy={acc:.3f}")`,
            output: String.raw`majority: epoch=4, total mistakes=8, w=[1. 1. 1.], b=-2, accuracy=1.000
parity  : epoch=None, total mistakes=203, w=[1. 0. 0.], b=-1, accuracy=0.500`,
          },
          {
            t: "p",
            text: String.raw`**Majority** is linearly separable — the learned plane $x_1 + x_2 + x_3 = 2$ is exactly "at least two ones" — so it converges (epoch 4, 8 mistakes). **Parity** is not: fix $x_3 = 0$ and it becomes XOR. So the rule keeps erring every epoch (203 mistakes in 50 epochs) and ends no better than a coin flip, 0.5 accuracy. Parity needs hidden layers.`,
          },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "A real neuron combines excitatory and inhibitory inputs at the soma and fires an all-or-none spike past a threshold. The artificial neuron borrows the words but is a simple algebraic function.",
          "**McCulloch–Pitts (1943):** 0/1 inputs, fixed threshold, inhibitory veto; networks compute any Boolean function but cannot learn.",
          String.raw`**Artificial neuron:** $z = \mathbf{w}^\top\mathbf{x} + b$, $a = f(z)$. Weights scale inputs, the bias shifts the threshold ($b = -\theta$), the activation shapes the output.`,
          String.raw`**Geometry:** the boundary $\mathbf{w}^\top\mathbf{x} + b = 0$ is a hyperplane; $\mathbf{w}$ is its normal; $z/\lVert\mathbf{w}\rVert$ is signed distance; $|b|/\lVert\mathbf{w}\rVert$ is the offset from the origin. Without a bias it must pass through the origin.`,
          String.raw`**Perceptron rule (Rosenblatt, 1958):** $e = y - \hat y$, $\mathbf{w} \leftarrow \mathbf{w} + \eta e\mathbf{x}$, $b \leftarrow b + \eta e$. Each update moves $z$ on that example by $\eta e(\lVert\mathbf{x}\rVert^2 + 1)$, toward the right answer.`,
          String.raw`**Worked example:** AND from zero weights, $\eta = 1$, converges in epoch 6 to $\mathbf{w} = (2,1)$, $b = -3$; OR in epoch 4; NAND in epoch 6.`,
          String.raw`**Convergence theorem:** separable data with margin $\gamma$ and radius $R$ need at most $(R/\gamma)^2$ updates. From zero initialisation the learning rate does not affect predictions.`,
          "**XOR (Minsky & Papert, 1969):** not linearly separable, so one perceptron cannot represent it and the rule cycles forever. A hidden layer (OR and NAND feeding AND) solves it by re-representing the input — the key idea of deep learning.",
          "Hidden layers cannot be trained by the perceptron rule; they need differentiable activations and backpropagation.",
        ],
      },
      { t: "math", tex: String.raw`z = \mathbf{w}^\top\mathbf{x} + b,\quad \hat y = \mathbb{1}[z \ge 0],\quad \mathbf{w} \leftarrow \mathbf{w} + \eta(y - \hat y)\mathbf{x},\quad b \leftarrow b + \eta(y - \hat y),\quad m \le (R/\gamma)^2`, caption: "Key formulas of the lesson (𝟙[z ≥ 0] is the step: 1 if z ≥ 0, else 0)." },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[Activation Functions](/learn/m03/activation-functions) — the step function made training hidden layers impossible. Next: the smooth activations that replaced it (sigmoid, tanh, ReLU, GELU and others), why non-linearity is essential, and how each derivative affects learning.",
          "[Multi-Layer Networks and Model Capacity](/learn/m03/multilayer-networks) — your hand-built XOR network grows into the multi-layer perceptron; the universal approximation theorem says how expressive it can become.",
          "[Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) — the epoch is one part of the full training loop, with batches, iterations and validation.",
          "Later, [Backpropagation](/learn/m04/backpropagation) answers this lesson's open question: how to give hidden units an error signal so they can learn.",
        ],
      },
    ],
  },
};

export default lesson;
