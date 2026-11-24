import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m03/artificial-neuron",
  title: "The Artificial Neuron and the Perceptron",
  summary:
    "From the biological neuron to McCulloch–Pitts and Rosenblatt: weights, bias, the weighted sum, the decision boundary as a hyperplane, the perceptron learning rule worked by hand, its convergence guarantee, and why XOR needs a hidden layer.",
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
        text: "An **artificial neuron** is the smallest computational unit of a neural network. It takes a list of numbers as input, multiplies each by a **weight**, adds them together with a **bias**, and passes the result through an **activation function**. Every layer of every network you will meet in this course — fully connected layers, convolutional filters, the feed-forward blocks inside Transformers — is built from this one operation repeated many times.",
      },
      {
        t: "p",
        text: "The idea has a precise history. In 1943 Warren McCulloch and Walter Pitts published “A Logical Calculus of the Ideas Immanent in Nervous Activity”, showing that a highly simplified model of a nerve cell — binary inputs, a threshold, a binary output — can compute logical functions such as AND, OR and NOT, and that networks of such units can compute any Boolean function. Their neuron had no way to learn: a human had to choose the threshold. In 1958 Frank Rosenblatt introduced the **perceptron** (“The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain”), which added real-valued weights and, crucially, a **learning rule** that adjusts those weights from labelled examples.",
      },
      {
        t: "p",
        text: "The problem the perceptron solves is **binary classification**: given an input vector $\\mathbf{x}$, decide whether it belongs to class 1 or class 0, learning the decision from examples rather than hand-written rules. Geometrically, a single neuron draws a straight line (in 2-D), a plane (in 3-D) or a **hyperplane** (in general) through input space and labels points by which side they fall on.",
      },
      {
        t: "p",
        text: "That geometric fact is also the neuron's limit. In 1969 Marvin Minsky and Seymour Papert's book “Perceptrons” analysed rigorously what single-layer perceptrons cannot compute; the best-known example is **XOR**, whose two classes cannot be separated by any single line. The way out — stacking neurons into **hidden layers** — is the foundation of all modern deep learning, and the reason this lesson ends where [Multi-Layer Networks](/learn/m03/multilayer-networks) begins.",
      },
      {
        t: "callout",
        kind: "key",
        title: "In one sentence",
        text: "A neuron computes $z = w_1x_1 + \\dots + w_nx_n + b$ and outputs $f(z)$; the perceptron is a neuron with a step activation plus a rule that nudges the weights whenever it misclassifies an example.",
      },
    ],

    // --------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — you need the **dot product** $\\mathbf{w}\\cdot\\mathbf{x} = \\sum_i w_ix_i$ (the weighted sum is exactly a dot product), vector length $\\lVert\\mathbf{w}\\rVert$, and the idea that a vector has a direction.",
          "[Linear and Logistic Regression as Neural Networks](/learn/m03/linear-logistic-regression) — you need the notion of a model with parameters fitted to labelled data, and the idea of a **decision boundary**. Logistic regression turns out to be a single neuron with a sigmoid activation.",
          "Basic algebra of linear inequalities (if $a \\ge 0$ and $c \\ge 0$ then $a + c \\ge 0$) — used in the proof that XOR is impossible for one neuron.",
          "Optional: [History of Neural Networks](/learn/m01/history) for the wider story of the first AI winter that followed the 1969 critique.",
        ],
      },
      {
        t: "callout",
        kind: "note",
        text: "No calculus is needed for this lesson. The perceptron rule is not gradient descent on a smooth loss; it is a simple error-correction rule. Derivatives enter in the next lessons, once the step function is replaced by smooth activations.",
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A loan officer with a scorecard",
        text: "Imagine a loan officer who scores each application. Income adds points, existing debt subtracts points, years at the same job add a few. Each factor has an importance — a **weight**. The officer also has a baseline leniency: a strict bank starts every applicant at minus 20 points, a lenient one at plus 5 — that is the **bias**. If the total is at least zero, the loan is approved; otherwise it is rejected — that is the **threshold activation**. When the officer approves someone who later defaults, they lower the weight on whatever made that applicant look good; when they reject a good applicant, they raise it. That correction habit is the **perceptron learning rule**.",
      },
      { t: "h", text: "The biological neuron, briefly" },
      {
        t: "p",
        text: "A biological neuron receives signals through branching fibres called **dendrites**. Signals arrive at **synapses**, junctions where the axon terminal of one neuron releases chemical neurotransmitters onto another neuron's dendrite. Some synapses are **excitatory** (they push the receiving cell's membrane voltage up) and some are **inhibitory** (they push it down); the strength of each synapse varies and changes with experience. The cell body, or **soma**, integrates these incoming effects. If the membrane potential near the start of the **axon** crosses a **firing threshold**, the neuron emits an **action potential** (a spike) — an all-or-none electrical pulse that travels down the axon to the synapses on other neurons.",
      },
      {
        t: "table",
        head: ["Biological neuron", "Artificial neuron", "Caveat"],
        rows: [
          ["Dendrites receive signals", "Inputs $x_1,\\dots,x_n$", "Real dendrites perform non-linear computation of their own."],
          ["Synaptic strength (excitatory / inhibitory)", "Weight $w_i$ (positive / negative)", "Real synapses are not freely signed; a neuron is usually all-excitatory or all-inhibitory."],
          ["Soma integrates inputs", "Weighted sum $z = \\sum_i w_ix_i + b$", "Real integration is in continuous time with leaky membrane dynamics."],
          ["Firing threshold", "Bias $b$ (equivalently, a threshold $\\theta = -b$)", "—"],
          ["All-or-none spike along the axon", "Activation $a = f(z)$", "Real neurons encode information in spike timing and rate; most artificial neurons output a real number."],
          ["Learning by changing synapses", "Learning by changing weights", "Brains do not run backpropagation in the form used by deep learning."],
        ],
      },
      {
        t: "p",
        text: "The analogy explains the vocabulary, but do not over-read it. Modern artificial neurons are mathematical functions chosen because they are easy to optimise, not because they are faithful to biology.",
      },
      { t: "h", text: "What the neuron does, in plain language" },
      {
        t: "list",
        items: [
          "**Weigh the evidence.** Each input is multiplied by how much it matters. A large positive weight means the input is strong evidence for class 1; a negative weight means it is evidence against.",
          "**Add it up, plus a baseline.** The bias shifts the total up or down regardless of the input, which sets how easy it is to say yes.",
          "**Decide.** The activation turns the total into an output. For the perceptron, that is a hard yes/no: output 1 if the total is at least zero, else 0.",
          "**Learn from mistakes.** If the answer was wrong, move each weight a little in the direction that would have made the answer right, in proportion to how active that input was.",
        ],
      },
      {
        t: "p",
        text: "Seen as geometry, all inputs that make the total exactly zero form a straight line (for two inputs). Points on one side get 1, points on the other get 0. Learning means rotating and shifting that line until it separates the classes — which works if, and only if, a separating line exists.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "1. The McCulloch–Pitts neuron (1943)" },
      {
        t: "p",
        text: "The McCulloch–Pitts (MP) unit takes binary inputs $x_i \\in \\{0,1\\}$, divided into excitatory and inhibitory inputs. In its original formulation, any active inhibitory input forces the output to 0 (an absolute veto). Otherwise the unit counts its active excitatory inputs and fires (outputs 1) if the count reaches a fixed integer threshold $\\theta$. There are no learned weights and no learning rule — the designer picks $\\theta$ (and the wiring).",
      },
      {
        t: "steps",
        items: [
          { title: "AND of two inputs", text: "Threshold $\\theta = 2$: fires only when both inputs are 1." },
          { title: "OR of two inputs", text: "Threshold $\\theta = 1$: fires when at least one input is 1." },
          { title: "NOT", text: "One inhibitory input and threshold $\\theta = 0$: with the input off, the count $0 \\ge 0$ fires; with it on, the inhibitory veto forces 0." },
          { title: "Any Boolean function", text: "Because AND, OR and NOT are functionally complete, networks of MP units can compute any Boolean function — the 1943 paper's central claim. What they lack is a way to find the right thresholds from data." },
        ],
      },
      { t: "h", text: "2. The modern artificial neuron — the forward computation" },
      {
        t: "steps",
        items: [
          { title: "Receive the input vector", tex: String.raw`\mathbf{x} = (x_1, x_2, \dots, x_n)` , text: "Each $x_i$ is a real number (a pixel intensity, a word count, the output of another neuron)." },
          { title: "Multiply each input by its weight", tex: String.raw`p_i = w_i x_i, \quad i = 1,\dots,n`, text: "The weights $w_i$ are the learnable parameters; their sign and size express how each input influences the output." },
          { title: "Sum the products and add the bias", tex: String.raw`z = p_1 + p_2 + \dots + p_n + b = \mathbf{w}^\top\mathbf{x} + b`, text: "$z$ is called the **pre-activation**, **net input** or **logit**. The bias $b$ is also learned." },
          { title: "Apply the activation function", tex: String.raw`a = f(z)`, text: "For the perceptron $f$ is the step function. Later lessons use sigmoid, tanh, ReLU and others; see [Activation Functions](/learn/m03/activation-functions)." },
          { title: "Interpret the output", text: "With a step activation, $a \\in \\{0,1\\}$ is the predicted class. With a sigmoid, $a \\in (0,1)$ can be read as a probability. Inside a larger network, $a$ is simply passed on as an input to the next layer." },
        ],
      },
      { t: "h", text: "3. The Rosenblatt perceptron — the training procedure" },
      {
        t: "p",
        text: "The perceptron uses the step activation $\\hat y = \\text{step}(z)$, with the convention used throughout this lesson (and in the simulator) that $\\text{step}(z) = 1$ if $z \\ge 0$ and $0$ otherwise. Training on a labelled dataset $\\{(\\mathbf{x}^{(k)}, y^{(k)})\\}$ with $y \\in \\{0,1\\}$ proceeds **online**, one example at a time:",
      },
      {
        t: "steps",
        items: [
          { title: "Initialise", text: "Set $\\mathbf{w} = \\mathbf{0}$ and $b = 0$ (or small random values), and choose a learning rate $\\eta > 0$." },
          { title: "Take the next example", text: "Pick $(\\mathbf{x}, y)$ — in a fixed order, or shuffled each epoch." },
          { title: "Forward pass", tex: String.raw`z = \mathbf{w}^\top\mathbf{x} + b, \qquad \hat y = \text{step}(z)` },
          { title: "Compute the error", tex: String.raw`e = y - \hat y \in \{-1, 0, +1\}`, text: "$e = 0$: correct, nothing changes. $e = +1$: the neuron said 0 but should have said 1 (z too small). $e = -1$: it said 1 but should have said 0 (z too large)." },
          { title: "Update weights and bias", tex: String.raw`\Delta\mathbf{w} = \eta\, e\, \mathbf{x}, \quad \Delta b = \eta\, e, \qquad \mathbf{w} \leftarrow \mathbf{w} + \Delta\mathbf{w}, \quad b \leftarrow b + \Delta b` },
          { title: "Repeat over the dataset", text: "One full pass over all examples is an **epoch**. Count the mistakes made in the epoch." },
          { title: "Stop", text: "When an entire epoch produces zero mistakes, every example is classified correctly and the weights will never change again — the perceptron has **converged**. Otherwise stop at a maximum number of epochs." },
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "Why inputs with value 0 never get their weight changed",
        text: "Because $\\Delta w_i = \\eta e x_i$, an input that is 0 contributes nothing to the mistake and its weight is left alone. Only the bias, which acts like a weight on a constant input of 1, is always updated on a mistake. You will see this in the hand example: on sample (0, 0) only $b$ moves.",
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      { t: "h", text: "The neuron" },
      { t: "math", tex: String.raw`z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b, \qquad a = f(z)` },
      {
        t: "list",
        items: [
          String.raw`$n$ — number of inputs (features).`,
          String.raw`$\mathbf{x} \in \mathbb{R}^n$ — input vector with components $x_i$.`,
          String.raw`$\mathbf{w} \in \mathbb{R}^n$ — weight vector with components $w_i$ (learned).`,
          String.raw`$b \in \mathbb{R}$ — bias (learned). Older texts use a threshold $\theta$ with "fire if $\mathbf{w}^\top\mathbf{x} \ge \theta$"; this is the same rule with $b = -\theta$.`,
          String.raw`$z$ — pre-activation (weighted sum plus bias).`,
          String.raw`$f$ — activation function; $a$ — the neuron's output (activation).`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Augmented form.** Append a constant 1 to the input and the bias to the weights: $\tilde{\mathbf{x}} = (x_1,\dots,x_n,1)$, $\tilde{\mathbf{w}} = (w_1,\dots,w_n,b)$. Then $z = \tilde{\mathbf{w}}^\top\tilde{\mathbf{x}}$. This is why the bias is sometimes described as "a weight on an input that is always 1", and why its update rule $\Delta b = \eta e$ is the weight rule with $x = 1$.`,
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
        text: "The weighted sum is the same whatever activation follows; the activation only changes how that number is reported. This is why the decision boundary of a step neuron, a sigmoid neuron thresholded at 0.5 and a tanh neuron thresholded at 0 is the same set $z = 0$.",
      },
      { t: "h", text: "Geometry: the decision boundary is a hyperplane" },
      {
        t: "p",
        text: String.raw`The **decision boundary** is the set of inputs where the neuron is exactly undecided, $\{\mathbf{x} : \mathbf{w}^\top\mathbf{x} + b = 0\}$. In $\mathbb{R}^2$ this is the line $w_1x_1 + w_2x_2 + b = 0$; in $\mathbb{R}^n$ it is an $(n-1)$-dimensional **hyperplane**. Three facts follow from basic linear algebra:`,
      },
      {
        t: "list",
        ordered: true,
        items: [
          String.raw`**$\mathbf{w}$ is perpendicular (normal) to the boundary.** If $\mathbf{x}_1$ and $\mathbf{x}_2$ both lie on it, subtracting their equations gives $\mathbf{w}^\top(\mathbf{x}_1 - \mathbf{x}_2) = 0$: $\mathbf{w}$ is orthogonal to every direction lying in the boundary. It points toward the side where $z > 0$ (predicted class 1).`,
          String.raw`**$z$ measures signed distance.** The signed perpendicular distance of a point $\mathbf{x}$ from the boundary is $d(\mathbf{x}) = z / \lVert\mathbf{w}\rVert = (\mathbf{w}^\top\mathbf{x} + b)/\lVert\mathbf{w}\rVert$. So $|z|$ grows the further a point is from the boundary — a measure of confidence.`,
          String.raw`**The bias shifts the boundary.** The distance from the origin to the boundary is $|b| / \lVert\mathbf{w}\rVert$. With $b = 0$ the boundary is forced through the origin; changing $b$ slides it parallel to itself without rotating it. Changing $\mathbf{w}$ rotates it (and rescales $z$).`,
        ],
      },
      {
        t: "callout",
        kind: "key",
        title: "Why a bias is necessary",
        text: String.raw`Without a bias, the input $(0,0)$ always gives $z = 0$, so with the convention $\text{step}(0) = 1$ the neuron would output 1 for $(0,0)$ no matter what the weights are. AND, OR and XOR all require output 0 at $(0,0)$, so none of them could be learned. More generally, without $b$ every boundary passes through the origin, which removes one degree of freedom from the classifier.`,
      },
      {
        t: "p",
        text: String.raw`Numerical check with the weights the perceptron will learn for AND below, $\mathbf{w} = (2, 1)$, $b = -3$: $\lVert\mathbf{w}\rVert = \sqrt{5} = 2.2361$, the boundary $2x_1 + x_2 = 3$ lies $3/2.2361 = 1.3416$ from the origin, and the signed distances of $(0,0), (0,1), (1,0), (1,1)$ are $-1.3416, -0.8944, -0.4472, 0$. The last point sits exactly on the line — it is classified 1 only because of the tie-break $\text{step}(0) = 1$.`,
      },
      { t: "h", text: "The perceptron learning rule and why it makes sense" },
      { t: "math", tex: String.raw`\hat y = \text{step}(\mathbf{w}^\top\mathbf{x} + b), \qquad e = y - \hat y, \qquad \mathbf{w} \leftarrow \mathbf{w} + \eta\, e\, \mathbf{x}, \qquad b \leftarrow b + \eta\, e` },
      {
        t: "p",
        text: String.raw`Here $y \in \{0,1\}$ is the target, $\hat y$ the prediction, $e$ the error and $\eta > 0$ the learning rate. To see that the update always pushes in the right direction, compute the new pre-activation on the same example after an update:`,
      },
      { t: "math", tex: String.raw`z_{\text{new}} = (\mathbf{w} + \eta e \mathbf{x})^\top \mathbf{x} + (b + \eta e) = z + \eta\, e\,\big(\lVert\mathbf{x}\rVert^2 + 1\big)` },
      {
        t: "p",
        text: String.raw`Since $\lVert\mathbf{x}\rVert^2 + 1 > 0$, a false negative ($e = +1$) strictly increases $z$ and a false positive ($e = -1$) strictly decreases it. One update need not fix the example (if $\eta$ is small) but it always moves $z$ toward the correct side.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "The learning rate does not matter when starting from zero",
        text: String.raw`If $\mathbf{w}$ and $b$ start at 0, every weight vector the perceptron visits is $\eta$ times the one it would visit with $\eta = 1$, so all signs of $z$ — and hence every prediction and the number of epochs — are identical (in exact arithmetic; with floating-point values such as 0.1, rounding can flip exact ties $z = 0$). $\eta$ genuinely matters only when the initial weights are non-zero.`,
      },
      { t: "h", text: "Fully worked example: learning AND by hand" },
      {
        t: "p",
        text: String.raw`Data in order $(0,0)\to0$, $(0,1)\to0$, $(1,0)\to0$, $(1,1)\to1$. Start from $\mathbf{w} = (0,0)$, $b = 0$, $\eta = 1$, and predict 1 iff $z \ge 0$. Each row shows the example, the forward pass, the error, the update and the parameters **after** the update. (Every number was generated by running the rule in Python.)`,
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
        text: String.raw`Mistakes per epoch: 2, 3, 3, 2, 1, 0. Epoch 6 is the first clean pass, so the perceptron **converges in epoch 6** with $\mathbf{w} = (2, 1)$, $b = -3$, i.e. the boundary $2x_1 + x_2 = 3$. Check rows 1 and 4 in detail. Row 1: $z = 0\cdot0 + 0\cdot0 + 0 = 0$, so $\hat y = 1$ but $y = 0$, giving $e = -1$; since $\mathbf{x} = (0,0)$ only the bias changes, $b = 0 + 1\cdot(-1) = -1$. Row 4: $z = 0\cdot1 + 0\cdot1 - 1 = -1 < 0$, so $\hat y = 0$ but $y = 1$, giving $e = +1$, $\Delta\mathbf{w} = 1\cdot1\cdot(1,1) = (1,1)$ and $\Delta b = +1$, hence $\mathbf{w} = (1,1)$, $b = 0$.`,
      },
      {
        t: "callout",
        kind: "tip",
        text: "You can reproduce this table exactly in the simulator below: choose the AND preset, leave the sliders at their defaults (w1 = 0, w2 = 0, b = 0, η = 1) and press Step 24 times.",
      },
      { t: "h", text: "Convergence on linearly separable data" },
      {
        t: "p",
        text: String.raw`A dataset is **linearly separable** if some hyperplane puts all class-1 points strictly on one side and all class-0 points strictly on the other. Use the augmented vectors $\tilde{\mathbf{x}}$, $\tilde{\mathbf{w}}$ and relabel targets as $s = 2y - 1 \in \{-1,+1\}$. Suppose there is a unit vector $\mathbf{u}$ ($\lVert\mathbf{u}\rVert = 1$) and a **margin** $\gamma > 0$ with $s^{(k)}\,\mathbf{u}^\top\tilde{\mathbf{x}}^{(k)} \ge \gamma$ for every example, and let $R = \max_k \lVert\tilde{\mathbf{x}}^{(k)}\rVert$.`,
      },
      { t: "math", tex: String.raw`\text{number of updates} \;\le\; \left(\frac{R}{\gamma}\right)^2`, caption: "The perceptron convergence theorem (Block 1962; Novikoff 1962), starting from zero weights." },
      {
        t: "details",
        summary: "Proof sketch",
        blocks: [
          {
            t: "p",
            text: String.raw`Each update on a misclassified example is $\tilde{\mathbf{w}} \leftarrow \tilde{\mathbf{w}} + \eta s \tilde{\mathbf{x}}$ (for $y \in \{0,1\}$ the error $e$ equals $s$ on a mistake). After $m$ updates from $\tilde{\mathbf{w}} = \mathbf{0}$:`,
          },
          {
            t: "list",
            items: [
              String.raw`**Progress along $\mathbf{u}$:** each update adds $\eta s\,\mathbf{u}^\top\tilde{\mathbf{x}} \ge \eta\gamma$, so $\mathbf{u}^\top\tilde{\mathbf{w}}_m \ge m\eta\gamma$.`,
              String.raw`**Bounded growth of the norm:** $\lVert\tilde{\mathbf{w}} + \eta s\tilde{\mathbf{x}}\rVert^2 = \lVert\tilde{\mathbf{w}}\rVert^2 + 2\eta s\,\tilde{\mathbf{w}}^\top\tilde{\mathbf{x}} + \eta^2\lVert\tilde{\mathbf{x}}\rVert^2$, and the middle term is $\le 0$ because the example was misclassified. So $\lVert\tilde{\mathbf{w}}_m\rVert^2 \le m\eta^2R^2$.`,
              String.raw`**Combine with Cauchy–Schwarz:** $m\eta\gamma \le \mathbf{u}^\top\tilde{\mathbf{w}}_m \le \lVert\tilde{\mathbf{w}}_m\rVert \le \sqrt{m}\,\eta R$, which gives $\sqrt{m} \le R/\gamma$, i.e. $m \le (R/\gamma)^2$.`,
            ],
          },
          {
            t: "p",
            text: String.raw`Note that $\eta$ cancels — consistent with the earlier observation that the learning rate is irrelevant from zero initialisation. The bound depends only on the geometry: wide margins mean fast learning; tiny margins mean many updates.`,
          },
        ],
      },
      {
        t: "p",
        text: "The theorem guarantees a finite number of mistakes, but says nothing about which separating line is found — the perceptron stops at the first one that works, often one that barely clears some points (as in the AND example, where (1, 1) lies exactly on the boundary). If the data are **not** separable, the theorem does not apply and the weights never settle.",
      },
      { t: "h", text: "The XOR problem: a proof of impossibility" },
      {
        t: "p",
        text: String.raw`XOR outputs 1 when exactly one input is 1: $(0,0)\to0$, $(0,1)\to1$, $(1,0)\to1$, $(1,1)\to0$. Suppose a single neuron with step activation computed it. Then:`,
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
        text: String.raw`Adding the middle two inequalities gives $w_1 + w_2 + 2b \ge 0$, so $w_1 + w_2 + b \ge -b$. The first inequality says $-b > 0$, hence $w_1 + w_2 + b > 0$ — contradicting the last line. No weights and bias exist: **XOR is not linearly separable**, and the perceptron rule cannot converge on it. Geometrically, the two class-1 points sit on one diagonal of the unit square and the two class-0 points on the other; any line that puts both class-1 points on one side also captures at least one class-0 point.`,
      },
      {
        t: "p",
        text: String.raw`What happens in practice? Running the rule from $\mathbf{w} = (0,0)$, $b = 0$, $\eta = 1$ on XOR (in the order above), the mistakes per epoch are 3, 3, 4, 4, 4, … and from epoch 2 on the parameters return to exactly $\mathbf{w} = (-1, 0)$, $b = 0$ at the end of every epoch: the algorithm cycles forever. This behaviour is predicted by the **perceptron cycling theorem**: on non-separable data with bounded inputs, the weights stay bounded and (for integer data) revisit the same states.`,
      },
      { t: "h", text: "How a hidden layer solves XOR" },
      {
        t: "p",
        text: String.raw`XOR can be written as $\text{XOR}(x_1,x_2) = \text{AND}\big(\text{OR}(x_1,x_2),\ \text{NAND}(x_1,x_2)\big)$. Each of OR, NAND and AND is linearly separable, so each is one neuron. Put OR and NAND in a **hidden layer** and feed their outputs to an AND neuron:`,
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
        text: String.raw`The hidden layer **re-represents** the input. In the original $(x_1,x_2)$ plane the classes are tangled; in the hidden $(h_1,h_2)$ plane the four inputs map to only three points — $(0,1)$ for class 0 input $(0,0)$, $(1,1)$ for both class-1 inputs, $(1,0)$ for class 0 input $(1,1)$ — and $(1,1)$ alone is easily cut off by a line. This is the core idea of deep learning: layers transform data until it becomes linearly separable for the final layer. Note that this network's weights were set by hand; the perceptron rule cannot train hidden units, because it has no target for them. Training hidden layers required smooth activations and [backpropagation](/learn/m04/backpropagation).`,
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "neuron", caption: "A two-input neuron: live weighted sum, decision boundary, and the perceptron rule stepping through AND, OR, NAND and XOR." },
      { t: "h", text: "What you are looking at" },
      {
        t: "list",
        items: [
          "**Sliders** set $w_1$, $w_2$ and $b$ (each in [−3, 3]) and the learning rate η (0.1 to 1, default 1). Moving any slider restarts training from the slider values.",
          "**The plot** shows the input square $x_1, x_2 \\in [-0.5, 1.5]$ with the four dataset points coloured by their target class. The line is the decision boundary $w_1x_1 + w_2x_2 + b = 0$, the shaded half-plane is where the neuron predicts 1, the arrow is the weight vector (always perpendicular to the line, pointing into the shaded side), and misclassified points are ringed.",
          "**The weighted-sum panel** writes out the arithmetic for the current sample, e.g. z = 0·1 + 0·0 + 0 = 0, followed by the activation.",
          "**The activation selector** (step, sigmoid, tanh, ReLU, linear) changes only the displayed output $a = f(z)$ for the current sample. The classification and boundary always use $z \\ge 0 \\Rightarrow$ class 1, so the line does not move when you switch activation.",
          "**Play / Step** apply the perceptron rule to one sample at a time, in the order (0,0), (0,1), (1,0), (1,1). For each you see x, the target y, z, the prediction ŷ, the error e = y − ŷ, Δw = η·e·x, Δb = η·e and the new parameters, and the boundary moves. An epoch counter and the error count for the current epoch are shown.",
        ],
      },
      { t: "h", text: "Things to try" },
      {
        t: "steps",
        items: [
          { title: "Reproduce the hand example", text: "Choose AND, keep the defaults, and Step through. Compare each step with the 24-row table in the Mathematical Explanation. The simulator should report convergence in epoch 6 with w = (2, 1), b = −3. Notice that (1, 1) sits exactly on the final line." },
          { title: "Watch only the bias move", text: "On the first step, x = (0, 0) is misclassified, yet the line does not rotate — Δw = (0, 0) because both inputs are 0. Only b changes, sliding the line." },
          { title: "Compare the gates", text: "Reset and train OR (converges in epoch 4 from the defaults) and NAND (epoch 6). NAND ends with the mirror-image weights w = (−2, −1), b = +2: the same line as AND, with the arrow reversed." },
          { title: "Set a boundary by hand", text: "Before training, drag the sliders to w1 = 1, w2 = 1, b = −1.5 with AND selected. All four points are already correct; pressing Play gives a clean first epoch. Now try b = −0.5: you have built OR instead." },
          { title: "Learning rate from zero", text: "With all sliders at 0, set η = 0.5 and train AND again. Every Δw, Δb and z in the panel is halved, but the sequence of predictions is identical and it still converges in epoch 6, now at w = (1, 0.5), b = −1.5 — η does not matter from zero initialisation. Now set w1 = 1 (others 0) and compare: η = 1 converges in epoch 6, η = 0.5 in epoch 5. With a non-zero start, η matters." },
          { title: "A floating-point surprise", text: "From zero, try η = 0.1 on AND. In exact arithmetic it would behave like η = 1, but the AND trace contains many exact ties z = 0, and sums of 0.1 are not exact in binary floating point (0.1 + 0.1 + 0.1 − 0.3 is not 0). Some ties tip the other way, and in double precision this run converges in epoch 4 instead of 6. Tie-breaking at z = 0 is a real, if minor, source of fragility." },
          { title: "Break it with XOR", text: "Select XOR and press Play. The error count never reaches 0; from epoch 2 onward the parameters return to w = (−1, 0), b = 0 at the end of every epoch, and the simulator stops at the 30-epoch cap reporting that it never converged. Try to position the line by hand with the sliders: you can never get all four points right." },
          { title: "Inspect activations", text: "Pick any sample and switch the activation between step, sigmoid and tanh. The output value changes (e.g. 1 vs 0.5 vs 0 when z = 0), but the predicted class and the line stay the same." },
        ],
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "A complete NumPy implementation: a single neuron's forward pass, a `Perceptron` class trained online with the perceptron rule, experiments on the four logic gates, a larger randomly generated separable problem, and the hand-wired two-layer XOR network.",
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
          "The gate results match the hand trace: AND converges in epoch 6 with w = (2, 1), b = −3.",
          "On XOR the final weights (−1, 0), b = 0 classify only 2 of 4 points correctly — and those are just the parameters at the moment the 30-epoch cap hit.",
          "On the random data, the 0.2 gap around the teacher line guarantees a positive margin, so the convergence theorem applies. The learned weight direction is close to, but not equal to, the teacher's: the perceptron finds a separating line, not the true one.",
          "The hidden-layer network is not trained; it demonstrates that the representation exists. Finding such weights automatically is the job of backpropagation.",
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
                text: "Part A uses `nn.Linear(2, 1)` as the neuron's weighted sum and applies the perceptron rule manually inside `torch.no_grad()` — the step function has zero gradient almost everywhere, so autograd is of no use for it. Part B shows that a single linear unit trained with gradient descent cannot fit XOR, while a 2-4-1 network with a tanh hidden layer can.",
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
                text: String.raw`The single linear unit ends at loss $0.6931 = \ln 2$: on XOR the best a linear logit can do under binary cross-entropy is output probability 0.5 for every input (which the 0.5 threshold then labels 1). The hidden layer breaks this ceiling.`,
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras has no built-in perceptron rule, so this example contrasts a single `Dense(1)` unit with a small hidden-layer network on XOR, both trained by gradient descent on binary cross-entropy.",
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
        text: "The basic neuron has many relatives. They differ in input type, activation, the rule used to update weights, and what is done with non-separable data.",
      },
      {
        t: "table",
        head: ["Variant", "What changes", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["McCulloch–Pitts unit (1943)", "Binary inputs, fixed threshold, absolute inhibition, no learning", "Provably computes any Boolean function in networks; simple to analyse", "Weights/thresholds hand-designed; no learning", "Theory of computation, historical foundation"],
          ["Rosenblatt perceptron (1958)", "Real weights + bias, step activation, error-correction rule", "Simple, online, guaranteed to converge on separable data", "Never settles on non-separable data; no confidence output; finds an arbitrary separator", "Teaching; linear classification baselines"],
          ["Pocket perceptron", "Keeps (\"pockets\") the best weights seen so far", "Usable result on non-separable data", "Extra evaluation cost; still linear", "Noisy linear problems"],
          ["Averaged / voted perceptron", "Prediction uses the average (or weighted vote) of all weight vectors visited", "Much more stable; good generalisation; still online", "Must store an average or many vectors", "Online NLP tagging and parsing (structured perceptron)"],
          ["ADALINE (Widrow & Hoff, 1960)", "Updates using the error on the linear output z (least-mean-squares), threshold only at prediction", "Minimises a smooth squared error; converges to a least-squares solution even when not separable", "Squared error is not ideal for classification", "Adaptive filters, signal processing"],
          ["Sigmoid neuron (logistic regression)", "Sigmoid activation, trained by gradient descent on cross-entropy", "Outputs probabilities; differentiable, so it can be stacked and trained with backprop", "Saturates for large |z| (small gradients)", "Binary classification output layers"],
          ["ReLU / GELU neuron", "Modern activations, trained by backprop as part of deep networks", "Cheap, non-saturating for positive z; the workhorse of deep nets", "Unbounded output; ReLU units can 'die'", "Hidden layers of MLPs, CNNs, Transformers"],
          ["Multiclass perceptron", "One weight vector per class; predict the arg-max score; on a mistake add x to the true class, subtract from the predicted class", "Natural K-class extension with the same guarantees", "Still linear decision regions", "Multiclass linear classification"],
          ["Kernel perceptron", "Replaces dot products with a kernel function k(x, x')", "Non-linear boundaries without explicit hidden layers", "Cost grows with the number of stored mistakes", "Small non-linear problems; precursor of SVM ideas"],
        ],
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Every modern neural network.** A fully connected layer with $m$ outputs is $m$ neurons sharing an input; a convolutional filter is one neuron applied at every image position; the feed-forward block in each Transformer layer is two layers of neurons. Understanding one neuron is understanding the atom of all of them.",
          "**Historical hardware.** Rosenblatt's group built the Mark I Perceptron, a machine with a grid of photocells as input and motor-driven potentiometers as adjustable weights, used to learn to classify simple visual patterns.",
          "**Fast linear classifiers.** Perceptron-style and closely related online linear models remain useful when data streams in and must be processed in one pass with tiny memory: spam/ham scoring on bag-of-words features, click prediction with hashed features, and as quick baselines before building deeper models.",
          "**Natural language processing.** The averaged structured perceptron (Collins, 2002) was a standard method for part-of-speech tagging and parsing before neural models, valued for its speed and simplicity.",
          "**Output units.** The final layer of a binary classifier is literally one sigmoid neuron; for multiclass problems it is a layer of neurons followed by softmax.",
          "**Interpretability baselines.** For a single neuron, each weight's sign and magnitude directly say how a feature moves the decision, which makes linear models a common first step in regulated settings such as credit scoring.",
        ],
      },
    ],

    // -------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Extremely cheap:** one dot product to predict, one vector addition to learn. Memory is $O(n)$.",
          "**Online:** learns from one example at a time, so it handles streaming data and never needs the whole dataset in memory.",
          "**Guaranteed convergence** on linearly separable data, with a bound $(R/\\gamma)^2$ on the number of mistakes that does not depend on the number of examples or the dimension directly.",
          "**Interpretable:** the weight vector is the normal of the boundary; each weight's sign tells you the direction of a feature's influence.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Only linear boundaries.** It cannot represent XOR, parity, or any concept whose classes are not separable by a hyperplane.",
          "**No convergence on non-separable data.** Weights cycle; the result depends on when you stop. Real data with label noise is almost never perfectly separable.",
          "**Arbitrary separator.** It stops at the first separating hyperplane, often one with a tiny margin, which generalises worse than a maximum-margin solution (the idea behind support vector machines).",
          "**No probabilities.** The step output gives no confidence; you cannot tell a borderline case from a clear one without looking at $z$.",
          "**Not differentiable.** The step function has zero derivative everywhere except at 0, where it is undefined. Gradient-based training — and therefore training of hidden layers — needs smooth activations.",
          "**Order-sensitive.** The final weights and the number of epochs depend on the order examples are presented.",
        ],
      },
      { t: "h", text: "Practical trade-offs" },
      {
        t: "table",
        head: ["Choice", "Option A", "Option B"],
        rows: [
          ["Activation", "Step: crisp decisions, perceptron rule, no gradients", "Sigmoid/tanh/ReLU: graded outputs, trainable with gradient descent and backprop"],
          ["Learning rule", "Perceptron: updates only on mistakes, fast, unstable on noisy data", "Gradient descent on a loss: updates on every example, converges on non-separable data"],
          ["Model size", "One neuron: interpretable, linear only", "Layers of neurons: non-linear boundaries, more data and compute needed, less interpretable"],
          ["Final weights", "Last iterate: simple", "Averaged weights: more stable and usually more accurate"],
        ],
      },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting the bias.** Without $b$ the boundary must pass through the origin; on AND/OR with inputs in {0,1} the point (0, 0) can never be classified 0. **Fix:** always include a bias term (frameworks do by default: `nn.Linear(..., bias=True)`), or append a constant-1 feature.",
          "**Updating the bias with x instead of 1.** Writing `b += lr * e * x` is a shape error at best and a wrong rule at worst. **Why:** the bias is a weight on a constant input of 1, so its update is `lr * e`.",
          "**Inconsistent tie-breaking at z = 0.** Some sources use step(0) = 0, others 1, others the sign function with sign(0) = 0. The trace (and the convergence epoch) changes. **Fix:** pick one convention and use it everywhere — this lesson uses z ≥ 0 → 1.",
          "**Mixing label conventions.** The rule $\\Delta\\mathbf{w} = \\eta(y - \\hat y)\\mathbf{x}$ assumes $y, \\hat y \\in \\{0,1\\}$. With labels in $\\{-1,+1\\}$ the error is $\\pm 2$, or the rule is written as $\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta y\\mathbf{x}$ on mistakes. Mixing the two gives wrong updates. **Fix:** convert labels once at load time.",
          "**Expecting convergence on non-separable data.** Waiting for zero errors on XOR (or noisy real data) loops forever. **Fix:** cap the epochs, and use the pocket or averaged perceptron, or switch to a differentiable model.",
          "**Thinking the learning rate speeds up the perceptron.** From zero initialisation η only rescales the weights; in exact arithmetic predictions and the number of epochs are identical. **Fix:** tune η only if the initial weights are non-zero, and scale features instead.",
          "**Unscaled features.** Because $\\Delta\\mathbf{w} \\propto \\mathbf{x}$, a feature measured in thousands dominates every update and the bound $(R/\\gamma)^2$ grows with $R$. **Fix:** standardise features (zero mean, unit variance) before training.",
          "**Believing a neuron is a model of a brain cell.** The artificial neuron is inspired by biology but is a simple algebraic function. Arguments of the form 'the brain does X, so the network must too' are not reliable guides to design.",
          "**Treating the hand-built XOR network as 'learned'.** The perceptron rule cannot train hidden units because there is no target for them. **Fix:** hidden layers are trained with backpropagation on differentiable activations.",
        ],
      },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "What does a single artificial neuron compute?",
        a: "It computes a weighted sum of its inputs plus a bias, $z = \\mathbf{w}^\\top\\mathbf{x} + b$, and passes it through an activation function to produce $a = f(z)$. The weights and bias are learnable parameters; the activation decides the output's form (a class with a step, a probability with a sigmoid, a non-negative feature with ReLU).",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What is the role of the bias?",
        a: "It shifts the decision boundary away from the origin. The boundary $\\mathbf{w}^\\top\\mathbf{x} + b = 0$ lies at distance $|b|/\\lVert\\mathbf{w}\\rVert$ from the origin; without a bias every boundary passes through the origin, which makes simple functions such as AND on {0,1} inputs impossible. It is equivalent to a firing threshold $\\theta = -b$, or to a weight on a constant input of 1.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "Why can't a single perceptron learn XOR?",
        a: "A single perceptron's decision boundary is a straight line (hyperplane), and XOR's two classes lie on opposite diagonals of the unit square, so no line separates them. Algebraically, the four constraints $b<0$, $w_2+b\\ge0$, $w_1+b\\ge0$, $w_1+w_2+b<0$ are contradictory: adding the middle two gives $w_1+w_2+b \\ge -b > 0$.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "State the perceptron learning rule and show that each update moves the prediction in the right direction.",
        a: "On example $(\\mathbf{x}, y)$ with prediction $\\hat y = \\text{step}(z)$ and error $e = y - \\hat y$: $\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta e\\mathbf{x}$, $b \\leftarrow b + \\eta e$. The new pre-activation on the same example is $z + \\eta e(\\lVert\\mathbf{x}\\rVert^2 + 1)$; since the bracket is positive, $z$ increases on a false negative ($e=+1$) and decreases on a false positive ($e=-1$).",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why is the weight vector perpendicular to the decision boundary, and what does z tell you about a point?",
        a: "For any two points on the boundary, $\\mathbf{w}^\\top\\mathbf{x}_1 + b = \\mathbf{w}^\\top\\mathbf{x}_2 + b = 0$, so $\\mathbf{w}^\\top(\\mathbf{x}_1 - \\mathbf{x}_2) = 0$: $\\mathbf{w}$ is orthogonal to every in-boundary direction. The quantity $z/\\lVert\\mathbf{w}\\rVert$ is the signed perpendicular distance from the boundary — positive on the side $\\mathbf{w}$ points to — so $|z|$ measures how far, and hence how confidently, a point is classified.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "How does the perceptron differ from logistic regression?",
        a: "Both compute $z = \\mathbf{w}^\\top\\mathbf{x} + b$ and have the same kind of linear boundary. The perceptron uses a step activation and updates only on mistakes with the error-correction rule; it has no probabilistic output and does not converge on non-separable data. Logistic regression uses a sigmoid, outputs probabilities, and minimises cross-entropy by gradient descent, updating on every example in proportion to $y - \\sigma(z)$; it converges to a well-defined optimum (with regularisation) even when data are not separable. Interestingly, the gradient-descent update for logistic regression, $\\eta(y - \\sigma(z))\\mathbf{x}$, has exactly the perceptron's form with the hard prediction replaced by a soft one.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "State and sketch the proof of the perceptron convergence theorem.",
        a: "If (augmented) data satisfy $s^{(k)}\\mathbf{u}^\\top\\tilde{\\mathbf{x}}^{(k)} \\ge \\gamma > 0$ for some unit $\\mathbf{u}$ and $\\lVert\\tilde{\\mathbf{x}}^{(k)}\\rVert \\le R$, the perceptron started at zero makes at most $(R/\\gamma)^2$ updates. Proof: each update increases $\\mathbf{u}^\\top\\tilde{\\mathbf{w}}$ by at least $\\eta\\gamma$, so after $m$ updates it is $\\ge m\\eta\\gamma$; each update increases $\\lVert\\tilde{\\mathbf{w}}\\rVert^2$ by at most $\\eta^2R^2$ because the cross term is non-positive on a mistake, so $\\lVert\\tilde{\\mathbf{w}}\\rVert \\le \\sqrt{m}\\eta R$. Cauchy–Schwarz gives $m\\eta\\gamma \\le \\sqrt{m}\\eta R$, i.e. $m \\le (R/\\gamma)^2$. The bound is independent of $\\eta$ and of the number of examples.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Explain how a hidden layer makes XOR solvable, and why the perceptron rule cannot train that hidden layer.",
        a: "Hidden units compute new features of the input. With $h_1 = \\text{OR}(x_1,x_2)$ and $h_2 = \\text{NAND}(x_1,x_2)$, the four inputs map to $(0,1), (1,1), (1,1), (1,0)$ in hidden space, where the class-1 point $(1,1)$ is separated from the others by the AND line $h_1 + h_2 = 1.5$. So the composition is linear-separable at the last layer even though the input is not. The perceptron rule needs a target for each unit it updates; hidden units have none, and with step activations the output is a piecewise-constant function of the hidden weights, so no gradient signal tells them how to change. The solution is smooth activations plus backpropagation, which assigns each hidden unit a share of the output error via the chain rule.",
      },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "In the analogy between biological and artificial neurons, what corresponds to the synaptic strength?",
            options: ["The activation function", "The weight", "The input value", "The learning rate"],
            answer: 1,
            explain: "Synaptic strength scales how much an incoming signal affects the cell, which is exactly what a weight does to its input.",
          },
          {
            kind: "numeric",
            q: "A neuron has w = (0.5, −0.25), b = 0.2 and receives x = (1, 2). What is the weighted sum z?",
            answer: 0.2,
            tolerance: 0.001,
            explain: "z = 0.5·1 + (−0.25)·2 + 0.2 = 0.5 − 0.5 + 0.2 = 0.2.",
          },
          {
            q: "The decision boundary of a neuron with weights w and bias b is the set where w·x + b = 0. Increasing b while keeping w fixed…",
            options: [
              "rotates the boundary toward w",
              "slides the boundary parallel to itself",
              "has no effect on the boundary",
              "makes the boundary curved",
            ],
            answer: 1,
            explain: "The normal direction w is unchanged, so the boundary keeps its orientation; only its offset |b|/‖w‖ from the origin changes.",
          },
          {
            kind: "numeric",
            q: "The boundary 3x₁ + 4x₂ − 10 = 0: what is its distance from the origin?",
            answer: 2,
            tolerance: 0.001,
            explain: "Distance = |b| / ‖w‖ = 10 / √(3² + 4²) = 10 / 5 = 2.",
          },
          {
            q: "A perceptron with targets in {0,1} predicts ŷ = 1 for an example whose label is y = 0. Which update is applied (η > 0)?",
            options: ["w ← w + ηx, b ← b + η", "w ← w − ηx, b ← b − η", "w ← w − ηx, b unchanged", "No update"],
            answer: 1,
            explain: "e = y − ŷ = −1, so Δw = −ηx and Δb = −η: both push z down for this example.",
          },
          {
            kind: "numeric",
            q: "Linearly separable data have R = 2 and margin γ = 0.5 (augmented, unit-norm separator). What is the perceptron convergence theorem's upper bound on the number of updates?",
            answer: 16,
            tolerance: 0.001,
            explain: "(R/γ)² = (2/0.5)² = 4² = 16.",
          },
          {
            q: "Which of these Boolean functions of two inputs can a single perceptron NOT represent?",
            options: ["AND", "NAND", "OR", "XOR"],
            answer: 3,
            explain: "AND, NAND and OR are linearly separable; XOR's classes sit on opposite diagonals of the unit square.",
          },
          {
            q: "Starting from w = 0 and b = 0, what happens to the perceptron's sequence of predictions if you change η from 1 to 0.1?",
            options: [
              "It converges 10 times slower",
              "It converges 10 times faster",
              "It is identical — the weights are just scaled by 0.1",
              "It may no longer converge",
            ],
            answer: 2,
            explain: "Every weight vector visited is η times the η = 1 trajectory, so every sign of z, and hence every prediction, is unchanged (in exact arithmetic — floating-point rounding can occasionally flip an exact tie at z = 0).",
          },
          {
            kind: "numeric",
            q: "What was the McCulloch–Pitts threshold θ for a two-input OR unit (fire if the number of active excitatory inputs is at least θ)?",
            answer: 1,
            tolerance: 0,
            explain: "OR must fire whenever at least one input is active, so θ = 1. (AND uses θ = 2.)",
          },
        ],
      },
      {
        t: "exercise",
        title: "One perceptron update by hand",
        prompt: String.raw`A perceptron has $\mathbf{w} = (1, -2)$, $b = 0.5$ and learning rate $\eta = 0.5$, with the convention $\hat y = 1$ iff $z \ge 0$. It receives $\mathbf{x} = (1, 1)$ with target $y = 1$. Compute $z$, $\hat y$, $e$, $\Delta\mathbf{w}$, $\Delta b$, the new parameters, and the new $z$ on the same example. Then verify the formula $z_{\text{new}} = z + \eta e(\lVert\mathbf{x}\rVert^2 + 1)$.`,
        hint: "Compute the weighted sum first; the update only happens if the prediction is wrong.",
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
        prompt: "Using only NumPy, train a perceptron (zero initialisation, η = 1, predict 1 iff z ≥ 0, at most 50 epochs) on all 8 three-bit inputs for two targets: **majority** (1 if at least two inputs are 1) and **parity** (1 if an odd number of inputs are 1). Report, for each, the convergence epoch (or None), the total number of mistakes, the final weights and the training accuracy. Explain the difference.",
        hint: "Use itertools.product([0, 1], repeat=3) to enumerate the inputs. Three-bit parity is XOR generalised to three inputs.",
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
            text: String.raw`Majority is linearly separable — the learned plane $x_1 + x_2 + x_3 = 2$ is exactly "at least two ones" — so the perceptron converges (epoch 4, 8 mistakes in total). Parity is not separable (it contains XOR as a special case: fix $x_3 = 0$), so the rule keeps making mistakes every epoch (203 over 50 epochs) and the final weights are no better than chance, 0.5 accuracy. Computing parity needs hidden layers.`,
          },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "A biological neuron integrates excitatory and inhibitory inputs at the soma and fires an all-or-none spike when a threshold is crossed. The artificial neuron borrows this vocabulary but is a simple algebraic function.",
          "**McCulloch–Pitts (1943):** binary inputs, fixed threshold, inhibitory veto; networks of them compute any Boolean function, but they cannot learn.",
          String.raw`**Artificial neuron:** $z = \mathbf{w}^\top\mathbf{x} + b$, $a = f(z)$. Weights scale inputs, the bias shifts the threshold ($b = -\theta$), the activation shapes the output.`,
          String.raw`**Geometry:** the boundary $\mathbf{w}^\top\mathbf{x} + b = 0$ is a hyperplane; $\mathbf{w}$ is its normal; $z/\lVert\mathbf{w}\rVert$ is signed distance; $|b|/\lVert\mathbf{w}\rVert$ is the offset from the origin. Without a bias the boundary must pass through the origin.`,
          String.raw`**Perceptron rule (Rosenblatt, 1958):** $e = y - \hat y$, $\mathbf{w} \leftarrow \mathbf{w} + \eta e\mathbf{x}$, $b \leftarrow b + \eta e$. Each update changes $z$ on that example by $\eta e(\lVert\mathbf{x}\rVert^2 + 1)$.`,
          String.raw`**Worked example:** AND from zero weights, $\eta = 1$, converges in epoch 6 to $\mathbf{w} = (2,1)$, $b = -3$; OR converges in epoch 4; NAND in epoch 6.`,
          String.raw`**Convergence theorem:** on linearly separable data with margin $\gamma$ and radius $R$, at most $(R/\gamma)^2$ updates. From zero initialisation the learning rate does not affect predictions.`,
          "**XOR (Minsky & Papert, 1969):** not linearly separable, so a single perceptron cannot represent it and the rule cycles forever. A hidden layer (OR and NAND feeding AND) solves it by re-representing the input — the key idea of deep learning.",
          "Hidden layers cannot be trained by the perceptron rule; they need differentiable activations and backpropagation.",
        ],
      },
      { t: "math", tex: String.raw`z = \mathbf{w}^\top\mathbf{x} + b,\quad \hat y = \mathbb{1}[z \ge 0],\quad \mathbf{w} \leftarrow \mathbf{w} + \eta(y - \hat y)\mathbf{x},\quad b \leftarrow b + \eta(y - \hat y),\quad m \le (R/\gamma)^2`, caption: "Key formulas of the lesson." },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[Activation Functions](/learn/m03/activation-functions) — the step function made training of hidden layers impossible. Next you will see the smooth activations (sigmoid, tanh, ReLU, GELU and others) that replaced it, why some non-linearity is essential, and how each one's derivative affects learning.",
          "[Multi-Layer Networks and Model Capacity](/learn/m03/multilayer-networks) — the XOR network you built by hand generalises to multi-layer perceptrons; the universal approximation theorem says how expressive they can become.",
          "[Training: Epochs, Batches and Overfitting](/learn/m03/training-basics) — the epoch you met here is one part of the general training loop; you will learn how batches, iterations and validation fit together.",
          "Later, [Backpropagation](/learn/m04/backpropagation) answers the question this lesson leaves open: how to assign an error signal to hidden units so that they can learn.",
        ],
      },
    ],
  },
};

export default lesson;
