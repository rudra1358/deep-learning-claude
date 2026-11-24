import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m03/activation-functions",
  title: "Activation Functions",
  summary:
    "Why a neural network needs non-linearity, and a complete reference for step, linear, sigmoid, tanh, ReLU, Leaky ReLU, PReLU, ELU, SELU, GELU, Swish/SiLU, Mish and softmax: formulas, derivatives, ranges, failure modes and where each is used.",
  minutes: 60,
  level: "beginner",
  prerequisites: ["m03/artificial-neuron", "m02/derivatives-chain-rule", "m02/vectors-matrices"],
  next: ["m03/multilayer-networks", "m04/gradient-problems", "m04/backpropagation"],
  terms: ["activation-function", "non-linearity", "sigmoid", "tanh", "relu", "leaky-relu", "gelu", "swish", "softmax", "saturation", "vanishing-gradient", "dead-relu", "logits"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: String.raw`An **activation function** is the function $f$ that a neuron applies to its weighted sum. A neuron first computes the **pre-activation** $z = \mathbf{w}^\top \mathbf{x} + b$ and then outputs the **activation** $a = f(z)$. In a layer of many neurons the same $f$ is applied to every entry of the vector $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ independently (softmax is the one common exception: it acts on the whole vector at once).`,
      },
      {
        t: "p",
        text: String.raw`Activation functions exist to solve one specific problem: **without a non-linear $f$, a network of any depth computes nothing more than a single linear map**. Matrix multiplications and additions compose into another matrix multiplication and addition, so stacking 100 linear layers buys exactly the same expressive power as one. Inserting a non-linear function between layers breaks that collapse and lets the network bend, fold and threshold its input space, which is what makes deep networks universal function approximators.`,
      },
      {
        t: "p",
        text: "The choice of activation is not cosmetic. It decides (1) what range of values a layer can output, (2) how gradients flow backwards during training — whether they survive many layers or shrink to nothing — and (3) how cheap each forward and backward pass is. Historically the field moved from the **binary step** of the perceptron, to the smooth **sigmoid** and **tanh** that made gradient-based training possible, to the **ReLU** family that made deep networks trainable, and most recently to smooth ReLU-like functions such as **GELU** and **Swish/SiLU** that dominate Transformers and modern CNNs.",
      },
      {
        t: "list",
        items: [
          "**Hidden layers**: ReLU (most CNNs and MLPs), GELU (BERT, GPT-style Transformers), Swish/SiLU (EfficientNet and many recent models), tanh and sigmoid (inside LSTM and GRU gates).",
          "**Output layers**: sigmoid for a probability of one event (binary or multi-label classification), softmax for a probability distribution over mutually exclusive classes, identity (linear) for unbounded regression targets.",
          "**Attention**: softmax turns query–key similarity scores into weights that sum to 1.",
        ],
      },
    ],

    // ---------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[The Artificial Neuron and the Perceptron](/learn/m03/artificial-neuron) — the weighted sum $z = \mathbf{w}^\top\mathbf{x} + b$ and the idea that a neuron applies a function to it. The binary step activation is introduced there.`,
          "[Derivatives, Partial Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — every derivative in this lesson is a one-variable derivative, and the vanishing-gradient argument is the chain rule applied across layers: gradients are **products** of local derivatives.",
          "[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — matrix multiplication and its associativity, $(AB)C = A(BC)$, which is the whole proof that linear layers collapse.",
          String.raw`[Probability, Random Variables and Distributions](/learn/m02/probability) — only for GELU, which uses the cumulative distribution function $\Phi$ of the standard Gaussian, and for reading softmax outputs as a probability distribution.`,
          String.raw`Comfort with $e^x$ and $\ln x$: $e^x > 0$ for every $x$, $e^{a-b} = e^a / e^b$, and $e^x$ overflows a 64-bit float once $x$ exceeds about 709.`,
        ],
      },
    ],

    // -------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Folding a sheet of paper",
        text: "Think of each linear layer as stretching, rotating and sliding a flat sheet of paper. However many times you stretch and rotate it, it stays flat — a straight line drawn on it stays straight. An activation function is a **fold**: ReLU, for example, keeps one half of the sheet and flattens the other half onto a crease. After a few folds and stretches, the sheet can take on shapes no single stretch could produce, and a straight cut through the folded sheet becomes a complicated curve when you unfold it. That is exactly how a network separates classes that no straight line can separate.",
      },
      {
        t: "p",
        text: "For a first-time learner, three questions describe any activation function:",
      },
      {
        t: "list",
        ordered: true,
        items: [
          "**What does it output?** Sigmoid squashes everything into (0, 1), so it can be read as a probability. Tanh squashes into (−1, 1) and is centred on zero. ReLU passes positive numbers through unchanged and replaces negative numbers by 0.",
          "**How steep is it?** Training sends error signals backwards through each neuron, and the signal is multiplied by the slope of the activation at that point. Where the curve is flat (sigmoid far from 0, ReLU for negative inputs) almost no signal passes; this is called **saturation**.",
          "**How expensive is it?** ReLU is a single comparison. Sigmoid, tanh, ELU need an exponential. GELU needs the Gaussian error function. In large models this cost is small next to matrix multiplications, but it still matters on mobile hardware.",
        ],
      },
      {
        t: "p",
        text: "A good hidden-layer activation is non-linear (so depth helps), has a slope near 1 over a wide range of inputs (so gradients survive many layers), is cheap, and keeps outputs roughly centred on zero (so the next layer receives well-scaled inputs). No single function is best on all four counts, which is why so many exist.",
      },
    ],

    // -------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Where the activation sits in a layer" },
      {
        t: "steps",
        items: [
          { title: "Linear part", text: String.raw`Layer $\ell$ receives the previous activations $\mathbf{a}^{(\ell-1)}$ and computes the pre-activations $\mathbf{z}^{(\ell)} = W^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$.`, tex: String.raw`\mathbf{z}^{(\ell)} = W^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}` },
          { title: "Element-wise non-linearity", text: String.raw`Each entry is passed through $f$ independently: $a^{(\ell)}_i = f(z^{(\ell)}_i)$. Nothing mixes between neurons in this step.`, tex: String.raw`\mathbf{a}^{(\ell)} = f(\mathbf{z}^{(\ell)})` },
          { title: "Repeat", text: "The activations become the input of the next layer. The final layer usually uses a task-specific output activation (identity, sigmoid or softmax), or none at all when the loss function applies it internally." },
          { title: "Backward pass through the activation", text: String.raw`During backpropagation the gradient arriving from above, $\partial L / \partial \mathbf{a}^{(\ell)}$, is multiplied element-wise by the local slope $f'(\mathbf{z}^{(\ell)})$. This is why the derivative of every activation matters as much as the function itself.`, tex: String.raw`\frac{\partial L}{\partial \mathbf{z}^{(\ell)}} = \frac{\partial L}{\partial \mathbf{a}^{(\ell)}} \odot f'(\mathbf{z}^{(\ell)})` },
        ],
      },
      { t: "h", text: "Why non-linearity is required: linear layers collapse" },
      {
        t: "p",
        text: String.raw`Suppose two layers both use the identity activation $f(z) = z$. Then $\mathbf{h} = W_1\mathbf{x} + \mathbf{b}_1$ and $\mathbf{y} = W_2\mathbf{h} + \mathbf{b}_2$. Substituting:`,
      },
      {
        t: "math",
        tex: String.raw`\mathbf{y} = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \underbrace{(W_2W_1)}_{W}\,\mathbf{x} + \underbrace{(W_2\mathbf{b}_1 + \mathbf{b}_2)}_{\mathbf{b}}`,
        caption: "Two affine layers are one affine layer. By induction, any number of them are one affine layer.",
      },
      {
        t: "p",
        text: String.raw`By induction on depth, $L$ affine layers equal a single affine map $W\mathbf{x} + \mathbf{b}$ with $W = W_L \cdots W_1$. The deep network therefore has no more expressive power than logistic or linear regression: its decision boundaries are hyperplanes and it cannot represent XOR, a circle, or $\sin x$. (Depth without non-linearity can even **reduce** what is representable: if a hidden layer is narrower than the input, $W = W_2W_1$ has rank at most that width.)`,
      },
      {
        t: "p",
        text: String.raw`**Concrete check.** Take $W_1 = \begin{bmatrix}1&2\\0&-1\\3&1\end{bmatrix}$, $\mathbf{b}_1 = (1, 0, -2)$, $W_2 = \begin{bmatrix}2&-1&0.5\end{bmatrix}$, $b_2 = 0.5$ and $\mathbf{x} = (1, -1)$. Layer by layer: $\mathbf{h} = (1-2+1,\; 0+1+0,\; 3-1-2) = (0, 1, 0)$ and $y = 2\cdot0 - 1\cdot1 + 0.5\cdot0 + 0.5 = -0.5$. Collapsed: $W = W_2W_1 = (2\cdot1 - 0 + 1.5,\; 4 + 1 + 0.5) = (3.5, 5.5)$, $b = W_2\mathbf{b}_1 + b_2 = 2 - 0 - 1 + 0.5 = 1.5$, and $W\mathbf{x} + b = 3.5 - 5.5 + 1.5 = -0.5$. Identical.`,
      },
      {
        t: "p",
        text: "The PyTorch program in the Framework section shows the same thing empirically: a three-layer MLP with no activation, trained on y = sin(x) over [−3, 3], ends with mean squared error 0.168, which is exactly the error of the best straight line through that data (0.168, from ordinary least squares). With ReLU or GELU between the layers, the same network reaches an error below 1e−5.",
      },
      { t: "h", text: "Saturation and vanishing gradients" },
      {
        t: "p",
        text: String.raw`A function **saturates** where it becomes flat: its derivative approaches 0. Sigmoid and tanh saturate on both sides: $\sigma'(5) \approx 0.0066$ and $\tanh'(3) \approx 0.0099$. Backpropagation multiplies the gradient by $f'(z)$ at every layer it passes through, so after $L$ layers the gradient reaching the first layer contains the product $\prod_{\ell} f'(z^{(\ell)})$ (times weight terms).`,
      },
      {
        t: "p",
        text: String.raw`For the sigmoid, $\sigma'(z) \le 0.25$ everywhere, so with weights of order 1 the product over 10 layers is at most $0.25^{10} \approx 9.5\times10^{-7}$. Early layers receive almost no learning signal: the **vanishing gradient problem**. Tanh is better ($\tanh'(0) = 1$) but still saturates. ReLU has slope exactly 1 for every positive input, so along active paths the product does not shrink at all — the main reason ReLU made deep networks trainable (it was used in AlexNet, Krizhevsky et al., 2012).`,
      },
      { t: "h", text: "Dead ReLUs" },
      {
        t: "p",
        text: String.raw`ReLU has the opposite failure. For $z < 0$ its output and its derivative are both exactly 0. If a neuron's bias and weights drift so that $z < 0$ for **every** training example (often after one large update caused by a high learning rate), the neuron outputs 0 everywhere, receives zero gradient everywhere, and its weights never change again. It is **dead**. Leaky ReLU, PReLU, ELU, SELU, GELU, Swish and Mish all keep a non-zero slope (or at least a non-zero output) for negative inputs to avoid this.`,
      },
      { t: "h", text: "Softmax works on a whole vector" },
      {
        t: "p",
        text: String.raw`Softmax maps a vector of $K$ real scores (called **logits**) to $K$ positive numbers that sum to 1: $p_i = e^{z_i} / \sum_j e^{z_j}$. Each output depends on every input, so its derivative is a $K\times K$ Jacobian matrix rather than a single slope. Computing $e^{z_i}$ directly overflows for large logits; the fix is to subtract $\max_j z_j$ from every logit first, which leaves the result mathematically unchanged (derived in the next section).`,
      },
    ],

    // ------------------------------------------------------------------- math
    math: [
      {
        t: "p",
        text: String.raw`Notation: $x \in \mathbb{R}$ is the scalar input of the activation (the pre-activation $z$ of a neuron; we write $x$ to match the plots). $f(x)$ is the output and $f'(x) = df/dx$ its derivative. $\sigma(x) = 1/(1+e^{-x})$ is the logistic sigmoid. $\Phi(x)$ and $\varphi(x)$ are the cumulative distribution function and density of the standard normal distribution. $\alpha, \beta, \lambda$ are fixed or learnable shape parameters defined where they appear.`,
      },

      { t: "h", text: "1. Binary step (Heaviside)" },
      { t: "math", tex: String.raw`f(x) = \begin{cases}1 & x \ge 0\\ 0 & x < 0\end{cases} \qquad f'(x) = 0 \;\;(x \ne 0),\quad \text{undefined at } 0` },
      {
        t: "p",
        text: String.raw`Range $\{0, 1\}$. Used by the McCulloch–Pitts neuron and Rosenblatt's perceptron. **Pros**: trivially cheap, output is a hard yes/no decision. **Cons**: the derivative is 0 wherever it exists, so gradient descent receives no information at all — this is why it was abandoned once networks were trained with backpropagation. **Used**: only in historical models, in the perceptron learning rule, and as a conceptual baseline.`,
      },

      { t: "h", text: "2. Linear (identity)" },
      { t: "math", tex: String.raw`f(x) = x \qquad f'(x) = 1` },
      {
        t: "p",
        text: String.raw`Range $(-\infty, \infty)$. **Pros**: no saturation, constant gradient. **Cons**: adds no non-linearity, so hidden layers using it collapse (proof above). **Used**: the output layer of regression networks, where targets are unbounded real numbers; also as the "no activation" in bottleneck projections and residual branches' final layers.`,
      },

      { t: "h", text: "3. Sigmoid (logistic)" },
      { t: "math", tex: String.raw`\sigma(x) = \frac{1}{1+e^{-x}} \qquad \sigma'(x) = \sigma(x)\bigl(1-\sigma(x)\bigr)` },
      {
        t: "p",
        text: String.raw`**Derivation.** Write $\sigma = (1+e^{-x})^{-1}$. By the chain rule $\sigma' = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(x)\,(1-\sigma(x))$, since $\frac{e^{-x}}{1+e^{-x}} = 1 - \frac{1}{1+e^{-x}}$. The product $s(1-s)$ with $s\in(0,1)$ is maximal at $s = 0.5$, i.e. $x=0$, so $\sigma'(x) \le 0.25$.`,
      },
      {
        t: "p",
        text: String.raw`Range $(0, 1)$. **Pros**: smooth, output interpretable as a probability, derivative is cheap given the forward value. **Cons**: saturates on both sides; maximum slope 0.25 shrinks gradients at every layer; outputs are always positive (not zero-centred), so all weight gradients of the next layer share the same sign for a given example, which makes optimisation zig-zag; needs an exponential. **Used**: binary and multi-label output layers, LSTM/GRU gates, attention-style gating. Rarely in hidden layers of modern networks.`,
      },

      { t: "h", text: "4. Hyperbolic tangent (tanh)" },
      { t: "math", tex: String.raw`\tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = 2\sigma(2x) - 1 \qquad \tanh'(x) = 1 - \tanh^2(x)` },
      {
        t: "p",
        text: String.raw`The identity $\tanh(x) = 2\sigma(2x) - 1$ follows by multiplying numerator and denominator of $\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ by $e^{-x}$: $\frac{1-e^{-2x}}{1+e^{-2x}} = \frac{2}{1+e^{-2x}} - 1$. So tanh is a rescaled, re-centred sigmoid. For the derivative, the quotient rule gives $\frac{(e^x+e^{-x})^2 - (e^x-e^{-x})^2}{(e^x+e^{-x})^2} = 1 - \tanh^2(x)$.`,
      },
      {
        t: "p",
        text: String.raw`Range $(-1, 1)$, $\tanh'(0) = 1$. **Pros**: zero-centred, four times steeper than sigmoid at the origin. **Cons**: still saturates for $|x| \gtrsim 3$; exponential cost. **Used**: the candidate state and output squashing of LSTM cells, vanilla RNN hidden states, some normalising-flow and control networks where bounded outputs are needed.`,
      },

      { t: "h", text: "5. ReLU (rectified linear unit)" },
      { t: "math", tex: String.raw`\mathrm{ReLU}(x) = \max(0, x) \qquad \mathrm{ReLU}'(x) = \begin{cases}1 & x>0\\ 0 & x<0\end{cases}` },
      {
        t: "p",
        text: String.raw`At $x = 0$ the function has a kink and no derivative; frameworks use the subgradient value 0 there (any value in $[0,1]$ is a valid subgradient, and exact zeros are rare in floating point). Range $[0, \infty)$. Popularised for deep networks by Nair & Hinton (2010) and Glorot, Bordes & Bengio (2011). **Pros**: one comparison to compute; slope exactly 1 for positive inputs, so no vanishing through active units; produces exact zeros (sparse activations). **Cons**: dead neurons; not zero-centred; unbounded output can grow if not normalised; non-differentiable at 0. **Used**: default hidden activation for CNNs (ResNet, VGG) and MLPs.`,
      },

      { t: "h", text: "6. Leaky ReLU" },
      { t: "math", tex: String.raw`f(x) = \begin{cases}x & x>0\\ \alpha x & x\le 0\end{cases} \qquad f'(x) = \begin{cases}1 & x>0\\ \alpha & x<0\end{cases}` },
      {
        t: "p",
        text: String.raw`$\alpha$ is a small fixed slope, typically $0.01$ (introduced by Maas, Hannun & Ng, 2013); GAN discriminators commonly use $0.2$. Range $(-\infty, \infty)$. **Pros**: negative inputs still pass a gradient of $\alpha$, so units cannot die; as cheap as ReLU. **Cons**: one more hyperparameter; the benefit over ReLU is often small in practice. **Used**: GAN discriminators, YOLO-family detectors, any setting where dead units have been observed.`,
      },

      { t: "h", text: "7. PReLU (parametric ReLU)" },
      { t: "math", tex: String.raw`f(x) = \begin{cases}x & x>0\\ a x & x\le 0\end{cases} \qquad \frac{\partial f}{\partial x} = \begin{cases}1 & x>0\\ a & x<0\end{cases} \qquad \frac{\partial f}{\partial a} = \begin{cases}0 & x>0\\ x & x\le 0\end{cases}` },
      {
        t: "p",
        text: String.raw`Same shape as Leaky ReLU, but the negative slope $a$ is a **learnable parameter** (one per layer or one per channel), trained by backpropagation with the extra derivative $\partial f/\partial a = x$ for $x \le 0$. Introduced by He, Zhang, Ren & Sun (2015) in "Delving Deep into Rectifiers", the same paper that derived He initialisation. **Pros**: the network chooses its own slope; negligible extra parameters. **Cons**: can overfit on small data; weight decay should usually not be applied to $a$. **Used**: image classification and super-resolution CNNs, speaker/face-recognition networks.`,
      },

      { t: "h", text: "8. ELU (exponential linear unit)" },
      { t: "math", tex: String.raw`f(x) = \begin{cases}x & x>0\\ \alpha\,(e^{x}-1) & x\le 0\end{cases} \qquad f'(x) = \begin{cases}1 & x>0\\ \alpha e^{x} = f(x) + \alpha & x\le 0\end{cases}` },
      {
        t: "p",
        text: String.raw`Clevert, Unterthiner & Hochreiter (2015). With $\alpha = 1$ (the usual choice) the function is continuous and its derivative is continuous at 0 (both pieces have slope 1 there). Range $(-\alpha, \infty)$. **Pros**: negative outputs push the mean activation towards zero; smooth for $\alpha = 1$; saturates gently to $-\alpha$ for very negative inputs, which makes the response to strong negative inputs robust to noise. **Cons**: an exponential for every negative input; still saturates on the negative side. **Used**: some CNNs and MLPs, particularly where batch normalisation is not used.`,
      },

      { t: "h", text: "9. SELU (scaled ELU)" },
      { t: "math", tex: String.raw`f(x) = \lambda\begin{cases}x & x>0\\ \alpha\,(e^{x}-1) & x\le 0\end{cases} \qquad f'(x) = \lambda\begin{cases}1 & x>0\\ \alpha e^{x} & x\le 0\end{cases}, \quad \lambda \approx 1.0507,\ \alpha \approx 1.6733` },
      {
        t: "p",
        text: String.raw`Klambauer, Unterthiner, Mayr & Hochreiter (2017), "Self-Normalizing Neural Networks". The constants $\lambda = 1.0507009873554805$ and $\alpha = 1.6732632423543772$ are chosen so that, for a fully connected layer with weights drawn with variance $1/n_{\text{in}}$ (LeCun normal initialisation), inputs with mean 0 and variance 1 produce outputs with mean 0 and variance 1. That fixed point is attracting, so activations stay normalised through many layers without batch normalisation. Range $(-\lambda\alpha, \infty) \approx (-1.7581, \infty)$. Slope for positive inputs is $\lambda > 1$, which compensates for the damped negative side. **Pros**: self-normalising deep MLPs. **Cons**: the guarantee requires LeCun-normal initialisation, plain feed-forward layers, and "alpha dropout" instead of standard dropout; it does not transfer cleanly to CNNs with batch normalisation or to residual networks. **Used**: deep fully connected networks on tabular data.`,
      },

      { t: "h", text: "10. GELU (Gaussian error linear unit)" },
      { t: "math", tex: String.raw`\mathrm{GELU}(x) = x\,\Phi(x) = \frac{x}{2}\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt 2}\right)\right] \qquad \mathrm{GELU}'(x) = \Phi(x) + x\,\varphi(x),\ \ \varphi(x) = \frac{e^{-x^2/2}}{\sqrt{2\pi}}` },
      {
        t: "p",
        text: String.raw`Hendrycks & Gimpel (2016). $\Phi(x) = P(Z \le x)$ for $Z \sim \mathcal N(0,1)$, and $\operatorname{erf}$ is the Gaussian error function. **Interpretation**: ReLU multiplies $x$ by a hard gate $\mathbb 1[x>0]$; GELU multiplies $x$ by a soft gate $\Phi(x)$, the probability that a standard normal variable is below $x$. The derivative follows from the product rule and $\Phi' = \varphi$. The **tanh approximation**, used in the original BERT and GPT-2 code and still available as an option in frameworks, is`,
      },
      { t: "math", tex: String.raw`\mathrm{GELU}(x) \approx \tfrac12\, x\left[1 + \tanh\!\left(\sqrt{2/\pi}\,\bigl(x + 0.044715\,x^3\bigr)\right)\right]` },
      {
        t: "p",
        text: String.raw`Over $[-4, 4]$ the two forms differ by at most $4.7\times10^{-4}$ (computed in the From-Scratch code below). GELU is smooth and **non-monotonic**: it dips to a minimum of about $-0.170$ at $x \approx -0.752$, so its range is roughly $[-0.170, \infty)$ and its derivative is slightly negative for $x \lesssim -0.75$. **Pros**: smooth everywhere (better-behaved optimisation than ReLU's kink), no dead units, strong empirical results in Transformers. **Cons**: costlier than ReLU; the erf form needs a special function. **Used**: BERT, GPT-2/3 and most Transformer feed-forward blocks, Vision Transformers.`,
      },

      { t: "h", text: "11. Swish / SiLU" },
      { t: "math", tex: String.raw`\mathrm{swish}_\beta(x) = x\,\sigma(\beta x) \qquad \mathrm{swish}_\beta'(x) = \sigma(\beta x) + \beta x\,\sigma(\beta x)\bigl(1 - \sigma(\beta x)\bigr)` },
      {
        t: "p",
        text: String.raw`With $\beta = 1$ this is the **SiLU** (sigmoid-weighted linear unit), $x\sigma(x)$. Ramachandran, Zoph & Le (2017) found it with an automated search over activation functions ("Searching for Activation Functions") and named it Swish. As $\beta \to \infty$, $\sigma(\beta x)$ becomes a step and Swish becomes ReLU; at $\beta = 0$ it is the linear function $x/2$. So $\beta$ interpolates between linear and ReLU. Like GELU it is smooth and non-monotonic; for $\beta = 1$ its minimum is about $-0.278$ at $x \approx -1.278$. (In fact $x\sigma(1.702x)$ is a well-known cheap approximation to GELU.) **Pros**: smooth, no dead units, cheap given a sigmoid. **Cons**: slightly costlier than ReLU; unbounded above. **Used**: EfficientNet, MobileNetV3 (as the cheaper "hard-swish"), YOLOv5-era detectors, and as the gate in SwiGLU feed-forward blocks of recent large language models.`,
      },

      { t: "h", text: "12. Mish" },
      { t: "math", tex: String.raw`\mathrm{mish}(x) = x\,\tanh\bigl(\operatorname{softplus}(x)\bigr),\ \ \operatorname{softplus}(x) = \ln(1+e^{x}) \qquad \mathrm{mish}'(x) = \tanh(s) + x\,\sigma(x)\bigl(1-\tanh^2 s\bigr),\ \ s = \operatorname{softplus}(x)` },
      {
        t: "p",
        text: String.raw`Misra (2019). The derivative uses $\operatorname{softplus}'(x) = \sigma(x)$ and the chain rule through $\tanh$. Shape is very close to Swish: smooth, non-monotonic, minimum about $-0.309$ at $x \approx -1.192$, range roughly $[-0.309, \infty)$. **Pros**: smooth, self-regularising small negative region; reported gains in some detection backbones. **Cons**: three transcendental functions per element, so the most expensive function in this list; gains over Swish are small and task-dependent. **Used**: YOLOv4's backbone, some image-classification CNNs.`,
      },

      { t: "h", text: "13. Softmax" },
      { t: "math", tex: String.raw`\operatorname{softmax}(\mathbf z)_i = p_i = \frac{e^{z_i/T}}{\sum_{j=1}^{K} e^{z_j/T}}, \qquad i = 1,\dots,K` },
      {
        t: "p",
        text: String.raw`$\mathbf z \in \mathbb R^K$ is the vector of logits, $K$ the number of classes, $T > 0$ the **temperature** (normally $T=1$). Range: every $p_i \in (0,1)$ and $\sum_i p_i = 1$. Small $T$ sharpens the distribution towards a one-hot vector at the largest logit (as $T\to0$ it becomes argmax); large $T$ flattens it towards uniform. **Pros**: turns arbitrary scores into a valid categorical distribution; pairs with cross-entropy to give the simple gradient $\mathbf p - \mathbf y$. **Cons**: not element-wise, so its derivative is a matrix; overflows if implemented naively; probabilities from it are often over-confident. **Used**: multi-class output layers, attention weights, policy networks in reinforcement learning, sampling from language models (with temperature).`,
      },
      {
        t: "p",
        text: String.raw`**Shift invariance and numerical stability.** For any constant $c$, $\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c}e^{z_i}}{e^{-c}\sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Choosing $c = \max_j z_j$ makes the largest exponent $e^0 = 1$ and every other exponent at most 1, so nothing overflows and the denominator is at least 1, so nothing divides by zero. For logits $(1000, 999, 998)$ the naive formula evaluates $e^{1000} = \infty$ and returns $\infty/\infty = $ NaN; the shifted formula returns $(0.6652, 0.2447, 0.0900)$.`,
      },
      {
        t: "p",
        text: String.raw`**Jacobian derivation** ($T = 1$). Using the quotient rule on $p_i = e^{z_i}/S$ with $S = \sum_k e^{z_k}$, and $\partial S/\partial z_j = e^{z_j}$:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial p_i}{\partial z_j} = \frac{\delta_{ij}e^{z_i}S - e^{z_i}e^{z_j}}{S^2} = p_i\,(\delta_{ij} - p_j) \quad\Longrightarrow\quad J = \operatorname{diag}(\mathbf p) - \mathbf p\,\mathbf p^\top`,
        caption: String.raw`$\delta_{ij}$ is 1 if $i = j$ and 0 otherwise. Diagonal entries $p_i(1-p_i)$ are positive; off-diagonal entries $-p_ip_j$ are negative.`,
      },
      {
        t: "p",
        text: String.raw`Each row of $J$ sums to $p_i - p_i\sum_j p_j = 0$: adding the same constant to every logit changes nothing, so the derivative in the direction $(1,1,\dots,1)$ is zero. $J$ is symmetric and positive semi-definite with that all-ones vector in its null space. When softmax is followed by cross-entropy $L = -\sum_i y_i \ln p_i$ with a one-hot $\mathbf y$, the chain rule through $J$ simplifies to $\partial L/\partial \mathbf z = \mathbf p - \mathbf y$.`,
      },

      { t: "h", text: "Worked example: every activation at four inputs" },
      {
        t: "p",
        text: String.raw`Values computed in Python (Leaky ReLU with $\alpha = 0.01$, ELU with $\alpha = 1$; each cell is $f(x)\,/\,f'(x)$, rounded to 4 decimals).`,
      },
      {
        t: "table",
        head: ["Function", "x = −2", "x = −0.5", "x = 0.5", "x = 2"],
        rows: [
          ["Step", "0 / 0", "0 / 0", "1 / 0", "1 / 0"],
          ["Linear", "−2 / 1", "−0.5 / 1", "0.5 / 1", "2 / 1"],
          ["Sigmoid", "0.1192 / 0.1050", "0.3775 / 0.2350", "0.6225 / 0.2350", "0.8808 / 0.1050"],
          ["Tanh", "−0.9640 / 0.0707", "−0.4621 / 0.7864", "0.4621 / 0.7864", "0.9640 / 0.0707"],
          ["ReLU", "0 / 0", "0 / 0", "0.5 / 1", "2 / 1"],
          ["Leaky ReLU", "−0.0200 / 0.0100", "−0.0050 / 0.0100", "0.5 / 1", "2 / 1"],
          ["ELU", "−0.8647 / 0.1353", "−0.3935 / 0.6065", "0.5 / 1", "2 / 1"],
          ["SELU", "−1.5202 / 0.2379", "−0.6918 / 1.0663", "0.5254 / 1.0507", "2.1014 / 1.0507"],
          ["GELU (erf)", "−0.0455 / −0.0852", "−0.1543 / 0.1325", "0.3457 / 0.8675", "1.9545 / 1.0852"],
          ["GELU (tanh)", "−0.0454 / −0.0861", "−0.1543 / 0.1326", "0.3457 / 0.8674", "1.9546 / 1.0861"],
          ["Swish (β=1)", "−0.2384 / −0.0908", "−0.1888 / 0.2600", "0.3112 / 0.7400", "1.7616 / 1.0908"],
          ["Mish", "−0.2525 / −0.1084", "−0.2207 / 0.2895", "0.3752 / 0.8864", "1.9440 / 1.0693"],
        ],
      },
      {
        t: "p",
        text: String.raw`Check one by hand: $\sigma(2) = 1/(1+e^{-2}) = 1/(1+0.135335) = 0.880797$, and $\sigma'(2) = 0.880797 \times 0.119203 = 0.104994$. Notice that GELU, Swish and Mish have derivatives **greater than 1** at $x = 2$ (they approach the line $y=x$ from below) and **negative** at $x = -2$ (they are past their minimum and rising back towards 0 as $x \to -\infty$).`,
      },

      { t: "h", text: "Worked example: softmax, temperature and Jacobian" },
      {
        t: "steps",
        items: [
          { title: "Logits and shift", text: String.raw`$\mathbf z = (2.0, 1.0, 0.1, -1.0)$, $\max = 2.0$, so the shifted logits are $(0, -1.0, -1.9, -3.0)$.` },
          { title: "Exponentials", text: String.raw`$e^{0} = 1$, $e^{-1} = 0.3679$, $e^{-1.9} = 0.1496$, $e^{-3} = 0.0498$. Sum $= 1.5672$.` },
          { title: "Normalise", text: String.raw`$\mathbf p = (1, 0.3679, 0.1496, 0.0498)/1.5672 = (0.6381, 0.2347, 0.0954, 0.0318)$.` },
          { title: "Temperature", text: String.raw`$T = 0.5$ (divide shifted logits by 0.5): exponentials $(1, 0.1353, 0.0224, 0.0025)$, sum $1.1602$, $\mathbf p = (0.8619, 0.1166, 0.0193, 0.0021)$ — sharper. $T = 2$: exponentials $(1, 0.6065, 0.3867, 0.2231)$, sum $2.2164$, $\mathbf p = (0.4512, 0.2737, 0.1745, 0.1007)$ — flatter.` },
          { title: "Jacobian at T = 1", text: String.raw`$J_{11} = p_1(1-p_1) = 0.6381\times0.3619 = 0.2309$, $J_{12} = -p_1p_2 = -0.6381\times0.2347 = -0.1498$, and so on.`, tex: String.raw`J = \begin{bmatrix} 0.2309 & -0.1498 & -0.0609 & -0.0203\\ -0.1498 & 0.1796 & -0.0224 & -0.0075\\ -0.0609 & -0.0224 & 0.0863 & -0.0030\\ -0.0203 & -0.0075 & -0.0030 & 0.0308\end{bmatrix}` },
        ],
      },
      {
        t: "p",
        text: "Every row of this matrix sums to 0 (up to rounding), as the derivation predicts.",
      },
    ],

    // ----------------------------------------------------------------- visual
    visual: [
      { t: "sim", id: "activations", caption: "Activation Function Explorer: functions (solid) and derivatives (dashed) over x ∈ [−5, 5], plus a softmax panel." },
      { t: "p", text: "The explorer evaluates the same formulas as the Mathematical Explanation section. Try the following, in order:" },
      {
        t: "list",
        ordered: true,
        items: [
          "**Saturation.** With the default selection (sigmoid, tanh, ReLU) and **Show derivatives** on, press **Play** to sweep the probe x from −5 to 5. Watch the readout table: the sigmoid derivative never exceeds 0.25 (at x = 0) and is below 0.01 at both ends; tanh peaks at 1 but is just as flat beyond |x| ≈ 3; ReLU's derivative is exactly 0 on the left and exactly 1 on the right.",
          "**Dead side vs leaky side.** Add Leaky ReLU and PReLU, then drag the **α** slider from 0 to 1. At α = 0 they coincide with ReLU (dead negative side); at α = 1 they become the linear function. In a real network PReLU's α is learned rather than set by hand.",
          "**ELU and SELU.** Select ELU and SELU and hover over the negative side. ELU levels off at −α (move α to see the floor move); SELU levels off near −1.758 and has slope ≈ 1.0507 on the positive side, not 1.",
          "**Smooth ReLUs.** Select ReLU, GELU, GELU (tanh) and Swish/SiLU and Mish together. Hover near x = −0.75 and x = −1.2 to find their negative minima, and confirm their derivatives go slightly negative there and exceed 1 for x around 1.5–2. The exact and tanh-approximated GELU curves are visually indistinguishable — read the fourth decimal in the table to see the difference.",
          "**Swish β.** Move the **β** slider: near 0.1 Swish is almost the straight line x/2; at 5 it is almost ReLU.",
          "**Softmax panel.** Starting from the default logits (2.0, 1.0, 0.1, −1.0) at T = 1 you should see probabilities 0.638, 0.235, 0.095, 0.032 and the exponentials (after subtracting the maximum) 1, 0.368, 0.150, 0.050 with sum 1.567. Lower **T** towards 0.1 and the bar for the largest logit approaches 1; raise it to 5 and the bars level out. Add the same number to every logit — the probabilities do not change (shift invariance). Type a huge logit such as 1000: because the panel subtracts the maximum, nothing overflows.",
        ],
      },
      {
        t: "callout",
        kind: "tip",
        title: "What to look for",
        text: "The area under the derivative curve where it is close to 1 is a good visual proxy for how well gradients will flow through that function in a deep network. Compare how wide that region is for sigmoid, tanh and the ReLU family.",
      },
    ],

    // ---------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "A complete NumPy implementation of every activation and its analytic derivative, a finite-difference check of all derivatives, a numerical demonstration that linear layers collapse, the product of local derivatives through 20 layers, and a numerically stable softmax with temperature and its Jacobian.",
      },
      {
        t: "code",
        lang: "python",
        title: "activations_numpy.py",
        code: String.raw`import numpy as np
from math import erf

# ---------- activation functions and their derivatives ----------
SELU_LAMBDA = 1.0507009873554805   # constants from Klambauer et al. (2017)
SELU_ALPHA = 1.6732632423543772
erf_vec = np.vectorize(erf)        # exact error function, element-wise

def sigmoid(x):
    # numerically stable: never evaluates exp of a large positive number
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def softplus(x):
    return np.logaddexp(0.0, x)    # log(1 + e^x) without overflow

def Phi(x):                        # standard normal CDF
    return 0.5 * (1.0 + erf_vec(x / np.sqrt(2.0)))

def phi(x):                        # standard normal PDF
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

ACTS = {
    # name: (f, f')
    "step":      (lambda x: (x >= 0).astype(float),       lambda x: np.zeros_like(x)),
    "linear":    (lambda x: x,                             lambda x: np.ones_like(x)),
    "sigmoid":   (sigmoid,                                 lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "tanh":      (np.tanh,                                 lambda x: 1 - np.tanh(x)**2),
    "relu":      (lambda x: np.maximum(0.0, x),            lambda x: (x > 0).astype(float)),
    "leaky_relu":(lambda x: np.where(x > 0, x, 0.01 * x),  lambda x: np.where(x > 0, 1.0, 0.01)),
    "elu":       (lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))),
                  lambda x: np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))),
    "selu":      (lambda x: SELU_LAMBDA * np.where(x > 0, x, SELU_ALPHA * np.expm1(np.minimum(x, 0))),
                  lambda x: SELU_LAMBDA * np.where(x > 0, 1.0, SELU_ALPHA * np.exp(np.minimum(x, 0)))),
    "gelu":      (lambda x: x * Phi(x),                    lambda x: Phi(x) + x * phi(x)),
    "gelu_tanh": (lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))), None),
    "swish":     (lambda x: x * sigmoid(x),                lambda x: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))),
    "mish":      (lambda x: x * np.tanh(softplus(x)),
                  lambda x: np.tanh(softplus(x)) + x * sigmoid(x) * (1 - np.tanh(softplus(x))**2)),
}

def numerical_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)   # central difference

# ---------- 1. table of values + gradient check ----------
xs = np.array([-2.0, -0.5, 0.5, 2.0])
print("f(x) at x =", xs)
for name, (f, df) in ACTS.items():
    print(f"  {name:10s}", np.round(f(xs), 4))

grid = np.linspace(-4, 4, 801)
grid = grid[np.abs(grid) > 1e-3]          # skip the kink at 0 (ReLU family)
print("max |analytic - numerical| derivative error:")
for name, (f, df) in ACTS.items():
    if df is None or name == "step":
        continue
    err = np.max(np.abs(df(grid) - numerical_grad(f, grid)))
    print(f"  {name:10s} {err:.1e}")

gap = np.max(np.abs(ACTS["gelu"][0](grid) - ACTS["gelu_tanh"][0](grid)))
print("max |GELU exact - tanh approx| on [-4,4]:", f"{gap:.1e}")

# ---------- 2. stacked linear layers collapse ----------
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(5, 3)), rng.normal(size=5)
W2, b2 = rng.normal(size=(2, 5)), rng.normal(size=2)
X = rng.normal(size=(4, 3))                       # 4 samples, 3 features
two_layer = (X @ W1.T + b1) @ W2.T + b2
W, b = W2 @ W1, W2 @ b1 + b2                      # the single equivalent layer
print("linear stack == single layer:", np.allclose(two_layer, X @ W.T + b))

# ---------- 3. vanishing gradient: product of local derivatives ----------
z = rng.normal(size=20)                           # one pre-activation per layer
for name in ["sigmoid", "tanh", "relu"]:
    df = ACTS[name][1]
    print(f"product of 20 {name:7s} derivatives: {np.prod(df(z)):.3e}")

# ---------- 4. softmax: naive vs stable, temperature, Jacobian ----------
def softmax(z, T=1.0):
    s = (z - np.max(z)) / T                       # subtract max: largest exponent is 0
    e = np.exp(s)
    return e / e.sum()

big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive softmax :", np.exp(big) / np.exp(big).sum())
print("stable softmax:", np.round(softmax(big), 4))

logits = np.array([2.0, 1.0, 0.1, -1.0])
for T in [0.5, 1.0, 2.0]:
    print(f"T={T}: p =", np.round(softmax(logits, T), 4))

p = softmax(logits)
J = np.diag(p) - np.outer(p, p)                   # dp_i/dz_j
J_num = np.stack([numerical_grad(lambda t: softmax(logits + t * np.eye(4)[j]), 0.0)
                  for j in range(4)], axis=1)
print("Jacobian:\n", np.round(J, 4))
print("Jacobian matches finite differences:", np.allclose(J, J_num, atol=1e-8))
print("row sums:", np.round(J.sum(axis=1), 12))`,
        output: String.raw`f(x) at x = [-2.  -0.5  0.5  2. ]
  step       [0. 0. 1. 1.]
  linear     [-2.  -0.5  0.5  2. ]
  sigmoid    [0.1192 0.3775 0.6225 0.8808]
  tanh       [-0.964  -0.4621  0.4621  0.964 ]
  relu       [0.  0.  0.5 2. ]
  leaky_relu [-0.02  -0.005  0.5    2.   ]
  elu        [-0.8647 -0.3935  0.5     2.    ]
  selu       [-1.5202 -0.6918  0.5254  2.1014]
  gelu       [-0.0455 -0.1543  0.3457  1.9545]
  gelu_tanh  [-0.0454 -0.1543  0.3457  1.9546]
  swish      [-0.2384 -0.1888  0.3112  1.7616]
  mish       [-0.2525 -0.2207  0.3752  1.944 ]
max |analytic - numerical| derivative error:
  linear     1.6e-11
  sigmoid    1.4e-11
  tanh       3.3e-11
  relu       1.6e-11
  leaky_relu 1.6e-11
  elu        1.8e-11
  selu       3.7e-11
  gelu       4.5e-11
  swish      5.1e-11
  mish       3.2e-11
max |GELU exact - tanh approx| on [-4,4]: 4.7e-04
linear stack == single layer: True
product of 20 sigmoid derivatives: 5.446e-15
product of 20 tanh    derivatives: 3.534e-08
product of 20 relu    derivatives: 0.000e+00
naive softmax : [nan nan nan]
stable softmax: [0.6652 0.2447 0.09  ]
T=0.5: p = [0.8619 0.1166 0.0193 0.0021]
T=1.0: p = [0.6381 0.2347 0.0954 0.0318]
T=2.0: p = [0.4512 0.2737 0.1745 0.1007]
Jacobian:
 [[ 0.2309 -0.1498 -0.0609 -0.0203]
 [-0.1498  0.1796 -0.0224 -0.0075]
 [-0.0609 -0.0224  0.0863 -0.003 ]
 [-0.0203 -0.0075 -0.003   0.0308]]
Jacobian matches finite differences: True
row sums: [-0. -0. -0. -0.]`,
      },
      {
        t: "list",
        items: [
          "Every analytic derivative agrees with the central-difference estimate to about 1e−11, which is the expected accuracy of a central difference with step 1e−5 in 64-bit floats.",
          "The 20-layer product with random pre-activations is about 5e−15 for sigmoid and 4e−8 for tanh. For ReLU it is exactly 0 because at least one of the 20 pre-activations is negative: a single inactive unit on a path blocks that path completely. In a real network the gradient reaches a neuron by summing over many paths, so ReLU networks still train as long as each unit is active for some inputs; but this is the mechanism behind dead units.",
          "The sigmoid is written in a two-branch form so it never computes the exponential of a large positive number; ELU and SELU use the minimum of x and 0 inside the exponential for the same reason.",
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
                text: "Built-in activation modules, their derivatives via autograd (which match the From-Scratch table exactly), the learnable PReLU slope, softmax with temperature, and an experiment showing that a network without non-linearities can only fit a straight line.",
              },
              {
                t: "code",
                lang: "python",
                title: "activations_torch.py",
                code: String.raw`import torch
import torch.nn as nn

torch.manual_seed(0)

# 1. Built-in activation modules and their autograd derivatives
x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
acts = {
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(negative_slope=0.01),
    "PReLU": nn.PReLU(num_parameters=1, init=0.25),   # slope is a learnable parameter
    "ELU": nn.ELU(alpha=1.0),
    "SELU": nn.SELU(),
    "GELU": nn.GELU(),                                 # exact erf form
    "GELU-tanh": nn.GELU(approximate="tanh"),
    "SiLU": nn.SiLU(),                                 # Swish with beta = 1
    "Mish": nn.Mish(),
}
for name, act in acts.items():
    y = act(x)
    (grad,) = torch.autograd.grad(y.sum(), x)          # dy_i/dx_i, since y_i depends only on x_i
    print(f"{name:10s} f={[round(v, 4) for v in y.tolist()]}  f'={[round(v, 4) for v in grad.tolist()]}")

# 2. PReLU's slope receives a gradient like any weight
prelu = acts["PReLU"]
prelu(x).sum().backward()
print("PReLU slope a =", prelu.weight.item(), " dL/da =", prelu.weight.grad.item())

# 3. Softmax over the class dimension, and why losses take raw logits
logits = torch.tensor([[2.0, 1.0, 0.1, -1.0]])
print("softmax:", [round(v, 4) for v in torch.softmax(logits, dim=1)[0].tolist()])
print("softmax T=2:", [round(v, 4) for v in torch.softmax(logits / 2.0, dim=1)[0].tolist()])
target = torch.tensor([0])
ce = nn.CrossEntropyLoss()(logits, target)             # applies log-softmax internally
print("cross-entropy from logits:", round(ce.item(), 4))

# 4. Same small MLP with and without a non-linearity, on a non-linear target
X = torch.linspace(-3, 3, 200).unsqueeze(1)
Y = torch.sin(X)
def train(hidden_act):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 32), hidden_act, nn.Linear(32, 32), hidden_act, nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(1000):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
    return loss.item()
print("final MSE, no activation (Identity):", f"{train(nn.Identity()):.2e}")
print("final MSE, ReLU                    :", f"{train(nn.ReLU()):.2e}")
print("final MSE, GELU                    :", f"{train(nn.GELU()):.2e}")`,
                output: String.raw`Sigmoid    f=[0.1192, 0.3775, 0.6225, 0.8808]  f'=[0.105, 0.235, 0.235, 0.105]
Tanh       f=[-0.964, -0.4621, 0.4621, 0.964]  f'=[0.0707, 0.7864, 0.7864, 0.0707]
ReLU       f=[0.0, 0.0, 0.5, 2.0]  f'=[0.0, 0.0, 1.0, 1.0]
LeakyReLU  f=[-0.02, -0.005, 0.5, 2.0]  f'=[0.01, 0.01, 1.0, 1.0]
PReLU      f=[-0.5, -0.125, 0.5, 2.0]  f'=[0.25, 0.25, 1.0, 1.0]
ELU        f=[-0.8647, -0.3935, 0.5, 2.0]  f'=[0.1353, 0.6065, 1.0, 1.0]
SELU       f=[-1.5202, -0.6918, 0.5254, 2.1014]  f'=[0.2379, 1.0663, 1.0507, 1.0507]
GELU       f=[-0.0455, -0.1543, 0.3457, 1.9545]  f'=[-0.0852, 0.1325, 0.8675, 1.0852]
GELU-tanh  f=[-0.0454, -0.1543, 0.3457, 1.9546]  f'=[-0.0861, 0.1326, 0.8674, 1.0861]
SiLU       f=[-0.2384, -0.1888, 0.3112, 1.7616]  f'=[-0.0908, 0.26, 0.74, 1.0908]
Mish       f=[-0.2525, -0.2207, 0.3752, 1.944]  f'=[-0.1084, 0.2895, 0.8864, 1.0693]
PReLU slope a = 0.25  dL/da = -2.5
softmax: [0.6381, 0.2347, 0.0954, 0.0318]
softmax T=2: [0.4512, 0.2737, 0.1745, 0.1007]
cross-entropy from logits: 0.4493
final MSE, no activation (Identity): 1.68e-01
final MSE, ReLU                    : 8.46e-06
final MSE, GELU                    : 2.72e-06`,
              },
              {
                t: "p",
                text: "The PReLU slope gradient is the sum of the negative inputs, −2 + (−0.5) = −2.5, exactly as the formula ∂f/∂a = x (for x ≤ 0) predicts. The Identity network's error 0.168 equals the least-squares straight-line fit of sin(x) on these 200 points, confirming the collapse argument.",
              },
              {
                t: "p",
                text: "Note that `nn.CrossEntropyLoss` takes raw logits and applies log-softmax internally; never put `nn.Softmax` in front of it. Functional forms are also available: `torch.relu`, `torch.sigmoid`, `torch.tanh`, `torch.nn.functional.gelu(x, approximate='tanh')`, `torch.nn.functional.silu`, `torch.nn.functional.mish`, `torch.softmax(x, dim=...)`.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "code",
                lang: "python",
                title: "activations_keras.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf

tf.random.set_seed(0)

# 1. Built-in activations and their derivatives via GradientTape
x = tf.Variable([-2.0, -0.5, 0.5, 2.0])
acts = {
    "sigmoid": tf.keras.activations.sigmoid,
    "tanh": tf.keras.activations.tanh,
    "relu": tf.keras.activations.relu,
    "leaky_relu": lambda t: tf.nn.leaky_relu(t, alpha=0.01),
    "elu": tf.keras.activations.elu,
    "selu": tf.keras.activations.selu,
    "gelu": lambda t: tf.keras.activations.gelu(t, approximate=False),
    "gelu_tanh": lambda t: tf.keras.activations.gelu(t, approximate=True),
    "swish": tf.keras.activations.swish,
}
for name, f in acts.items():
    with tf.GradientTape() as tape:
        y = f(x)
    dy = tape.gradient(tf.reduce_sum(y), x)
    print(name, np.round(y.numpy(), 4), np.round(dy.numpy(), 4))

# 2. Softmax with temperature
logits = tf.constant([[2.0, 1.0, 0.1, -1.0]])
print("softmax T=1:", np.round(tf.nn.softmax(logits, axis=-1).numpy(), 4))
print("softmax T=2:", np.round(tf.nn.softmax(logits / 2.0, axis=-1).numpy(), 4))

# 3. A small regression network using GELU hidden layers and a linear output
X = np.linspace(-3, 3, 200, dtype="float32").reshape(-1, 1)
Y = np.sin(X)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(32, activation="gelu"),
    tf.keras.layers.Dense(32),
    tf.keras.layers.PReLU(),                 # learnable negative slope
    tf.keras.layers.Dense(1),                # linear output for regression
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
model.summary()
history = model.fit(X, Y, epochs=300, batch_size=200, verbose=0)
print("final MSE:", history.history["loss"][-1])

# For classification, keep the last layer linear and let the loss apply softmax:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("cross-entropy from logits:", float(loss_fn([0], logits)))`,
                output: "Prints, for each activation, its four output values and the four derivatives computed by GradientTape (these match the PyTorch and NumPy tables), then the softmax probabilities of the logits (2.0, 1.0, 0.1, -1.0) at T = 1 and T = 2, then a model summary and the final training loss of a small GELU network fitted to sin(x).",
              },
              {
                t: "p",
                text: "Keras accepts activations either as a string argument of a layer (`activation=\"relu\"`) or as separate layers (`tf.keras.layers.ReLU`, `LeakyReLU`, `PReLU`, `ELU`, `Softmax`). As in PyTorch, pass `from_logits=True` to the cross-entropy loss and leave the last Dense layer linear.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------- variations
    variations: [
      { t: "h", text: "Complete comparison" },
      {
        t: "table",
        head: ["Function", "Formula", "Derivative", "Range", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Binary step", String.raw`$\mathbb 1[x \ge 0]$`, "0 (undefined at 0)", String.raw`$\{0,1\}$`, "Simplest hard decision", "No gradient — untrainable by backprop", "Perceptron, history"],
          ["Linear", String.raw`$x$`, String.raw`$1$`, String.raw`$(-\infty,\infty)$`, "No saturation", "No non-linearity; layers collapse", "Regression output"],
          ["Sigmoid", String.raw`$\frac{1}{1+e^{-x}}$`, String.raw`$\sigma(1-\sigma)\le 0.25$`, String.raw`$(0,1)$`, "Probability output, smooth", "Saturates, not zero-centred, vanishing gradients", "Binary/multi-label output, gates"],
          ["Tanh", String.raw`$\frac{e^x-e^{-x}}{e^x+e^{-x}}$`, String.raw`$1-\tanh^2 x \le 1$`, String.raw`$(-1,1)$`, "Zero-centred, steeper than sigmoid", "Saturates for |x| > 3", "RNN/LSTM states"],
          ["ReLU", String.raw`$\max(0,x)$`, String.raw`$\mathbb 1[x>0]$`, String.raw`$[0,\infty)$`, "Cheap, no vanishing on active side, sparse", "Dead units, not zero-centred, kink at 0", "Default for CNNs/MLPs"],
          ["Leaky ReLU", String.raw`$\max(\alpha x, x)$`, String.raw`$1$ or $\alpha$`, String.raw`$(-\infty,\infty)$`, "No dead units, cheap", "Extra hyperparameter α", "GAN discriminators, detectors"],
          ["PReLU", String.raw`$\max(ax, x)$, $a$ learned`, String.raw`$1$ or $a$; $\partial/\partial a = x$`, String.raw`$(-\infty,\infty)$`, "Learns its own slope", "Can overfit small data", "Image CNNs"],
          ["ELU", String.raw`$x$ or $\alpha(e^x-1)$`, String.raw`$1$ or $\alpha e^x$`, String.raw`$(-\alpha,\infty)$`, "Near-zero mean, smooth (α=1)", "Exponential cost", "CNNs/MLPs without BN"],
          ["SELU", String.raw`$\lambda\cdot\mathrm{ELU}_\alpha(x)$`, String.raw`$\lambda$ or $\lambda\alpha e^x$`, String.raw`$(-1.758,\infty)$`, "Self-normalising", "Needs LeCun init, plain MLP, alpha dropout", "Deep tabular MLPs"],
          ["GELU", String.raw`$x\,\Phi(x)$`, String.raw`$\Phi(x)+x\varphi(x)$`, String.raw`$[-0.170,\infty)$`, "Smooth, no dead units, strong in Transformers", "Costlier; non-monotonic", "BERT, GPT, ViT"],
          ["Swish / SiLU", String.raw`$x\,\sigma(\beta x)$`, String.raw`$\sigma(\beta x)+\beta x\sigma(\beta x)(1-\sigma(\beta x))$`, String.raw`$[-0.278,\infty)$ (β=1)`, "Smooth, interpolates linear↔ReLU", "Costlier than ReLU", "EfficientNet, SwiGLU in LLMs"],
          ["Mish", String.raw`$x\tanh(\ln(1+e^x))$`, String.raw`$\tanh s + x\sigma(x)(1-\tanh^2 s)$`, String.raw`$[-0.309,\infty)$`, "Smooth, small accuracy gains in some CNNs", "Most expensive", "YOLOv4 backbone"],
          ["Softmax", String.raw`$e^{z_i}/\sum_j e^{z_j}$`, String.raw`$\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$`, String.raw`$(0,1)$, sums to 1`, "Valid distribution; clean gradient with CE", "Vector-valued; overflow if naive; over-confident", "Multi-class output, attention"],
        ],
      },
      { t: "h", text: "Related variants you will meet" },
      {
        t: "list",
        items: [
          String.raw`**Softplus** $\ln(1+e^x)$: a smooth ReLU with derivative $\sigma(x)$; used where strictly positive outputs are needed (e.g. predicting a standard deviation).`,
          String.raw`**ReLU6** $\min(\max(0,x),6)$: a capped ReLU used in MobileNet for robustness under low-precision arithmetic.`,
          String.raw`**Hard sigmoid / hard swish**: piecewise-linear approximations, e.g. $\text{h-swish}(x) = x\,\mathrm{ReLU6}(x+3)/6$ in MobileNetV3, avoiding exponentials on mobile hardware.`,
          String.raw`**Gated linear units (GLU, SwiGLU, GeGLU)**: not element-wise activations but gating layers, $(\mathbf x W) \odot f(\mathbf x V)$, where $f$ is sigmoid, Swish or GELU. Shazeer (2020) showed these improve Transformer feed-forward blocks; SwiGLU is used in models such as PaLM and LLaMA.`,
          String.raw`**Log-softmax** $z_i - \ln\sum_j e^{z_j}$: computed directly (with the max-shift) rather than as $\ln(\operatorname{softmax})$, because the logarithm of a probability that underflowed to 0 is $-\infty$.`,
        ],
      },
    ],

    // ----------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Image classification CNNs** — ResNet and VGG use ReLU after every convolution (with batch normalisation); EfficientNet uses Swish/SiLU; MobileNetV2 uses ReLU6 and MobileNetV3 uses hard-swish for speed on phones.",
          "**Language models and Transformers** — BERT and GPT-2 use GELU in their feed-forward blocks; many recent large language models (for example PaLM and LLaMA) use SwiGLU. The softmax inside every attention head turns query–key scores into mixing weights.",
          "**Recurrent networks** — LSTM and GRU cells use sigmoid for their gates (a value in (0,1) is a natural ‘how much to let through’) and tanh for the candidate and output states (bounded, zero-centred).",
          "**Classification outputs** — softmax over classes for single-label problems (ImageNet's 1000 classes, next-token prediction over a vocabulary of tens of thousands); independent sigmoids for multi-label problems such as tagging an image with every object it contains.",
          "**Text generation** — the temperature parameter of softmax is the ‘temperature’ setting of LLM sampling: T < 1 makes outputs more deterministic, T > 1 more diverse.",
          "**Object detection** — YOLO-family detectors used Leaky ReLU (YOLOv3) and Mish in YOLOv4's backbone; sigmoid outputs predict objectness and box offsets.",
          "**Generative adversarial networks** — DCGAN-style discriminators use Leaky ReLU (slope 0.2) so gradients reach the generator even for strongly ‘fake’ inputs, and generators often end in tanh so pixel outputs lie in (−1, 1).",
          "**Regression and probabilistic outputs** — linear outputs for unbounded targets, softplus or exp for strictly positive quantities like variances.",
        ],
      },
    ],

    // -------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "How to choose" },
      {
        t: "list",
        items: [
          "**Default for hidden layers**: ReLU for CNNs and MLPs (cheap, well understood, pairs with He initialisation); GELU or SiLU for Transformers and when following a published recipe. The difference between good choices is usually small compared with learning rate, initialisation and normalisation.",
          "**Output layer is decided by the task, not by taste**: identity for regression, sigmoid for independent probabilities, softmax for one-of-K classes — and in all three cases the framework loss usually prefers raw logits for numerical stability.",
          "**Bounded vs unbounded**: bounded functions (sigmoid, tanh) cannot blow up but saturate; unbounded ones (ReLU family) keep gradients alive but let activations grow, so they rely on careful initialisation and normalisation.",
          "**Smooth vs piecewise linear**: smooth functions (GELU, Swish, Mish, ELU with α = 1) give a continuous gradient, which tends to help optimisation in very deep or attention-based networks; ReLU's kink is harmless in most CNNs and is the cheapest option.",
          "**Compute and memory**: ReLU's backward pass needs only a 1-bit mask of which inputs were positive; GELU/Swish/Mish backward passes need the input value and transcendental functions. On edge hardware and with quantisation, piecewise-linear functions (ReLU, ReLU6, hard-swish) are strongly preferred.",
          "**Monotonic vs non-monotonic**: GELU, Swish and Mish are non-monotonic (small negative dip). This is generally harmless and may help, but it means f is not invertible, which matters for e.g. normalising flows that need invertible layers (they use Leaky ReLU or similar instead).",
          "**Self-normalisation (SELU)** removes the need for batch normalisation in plain MLPs, at the price of strict requirements on initialisation, dropout and architecture.",
        ],
      },
      {
        t: "callout",
        kind: "key",
        text: "There is no universally best activation. Choose the output activation from the task, choose the hidden activation from the architecture family (ReLU for CNNs, GELU/SiLU for Transformers), and keep initialisation consistent with it.",
      },
    ],

    // --------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Applying softmax before `nn.CrossEntropyLoss` (or Keras loss with `from_logits=True`).** The loss already applies log-softmax, so the model is trained on softmax-of-softmax: probabilities are squashed into a narrow range and gradients become tiny. Fix: output raw logits from the last layer; apply softmax only when you need probabilities for reporting or sampling.",
          "**Computing softmax as exp(z) / sum(exp(z)) without subtracting the max.** Logits above about 709 overflow 64-bit floats (about 88 for 32-bit), giving inf/inf = NaN. Fix: subtract the max, or use the framework's softmax/log_softmax, which already do.",
          "**Computing log(softmax(z)) in two steps.** A probability that underflows to 0 gives log(0) = −inf and NaN gradients. Fix: use log_softmax / logsumexp.",
          "**Softmax over the wrong axis.** For a batch of shape (N, K), softmax must run over the class axis (dim=1 / axis=−1); running over dim=0 normalises across examples and produces meaningless outputs that still sum to 1 along the wrong axis. Fix: always pass dim explicitly and check that rows sum to 1.",
          "**Using sigmoid or tanh in every hidden layer of a deep network.** Their derivatives multiply to vanishingly small numbers, so early layers barely train. Fix: ReLU-family or GELU/SiLU in hidden layers, plus proper initialisation.",
          "**Putting ReLU (or sigmoid) on the output of a regression model.** ReLU makes negative targets impossible to predict and gives zero gradient whenever the output is negative; sigmoid restricts outputs to (0, 1). Fix: linear output for unbounded targets; rescale targets if you want a bounded output.",
          "**Forgetting the activation altogether** (e.g. `nn.Sequential(nn.Linear(...), nn.Linear(...))`). The network silently collapses to a linear model. Fix: an activation between every pair of linear layers.",
          "**Large learning rates with ReLU, then wondering why many units output 0.** A big update can push biases so negative that units die permanently. Fix: lower learning rate or warmup, He initialisation, Leaky ReLU/GELU, and monitor the fraction of zero activations per layer.",
          "**Using SELU with standard dropout, batch norm or default initialisation.** The self-normalising fixed point only exists under LeCun-normal initialisation and alpha dropout. Fix: use those, or use a different activation.",
          "**Believing sigmoid outputs of separate units form a distribution.** Independent sigmoids do not sum to 1; that is correct for multi-label tasks and wrong for one-of-K classification. Fix: softmax for mutually exclusive classes.",
          "**Worrying about ReLU's derivative at exactly 0.** It is undefined mathematically, but frameworks use 0 and exact zeros essentially never occur with continuous inputs; this has no practical effect on training.",
        ],
      },
    ],

    // -------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "Why do neural networks need activation functions?", a: String.raw`Without a non-linear activation, every layer is an affine map $W\mathbf x + \mathbf b$, and a composition of affine maps is affine: $W_2(W_1\mathbf x + \mathbf b_1) + \mathbf b_2 = (W_2W_1)\mathbf x + (W_2\mathbf b_1 + \mathbf b_2)$. So any depth collapses to a single linear layer and can only draw hyperplane decision boundaries. A non-linear activation between layers prevents the collapse and, with enough hidden units, lets the network approximate any continuous function on a bounded domain (universal approximation).` },
      { t: "qa", level: "beginner", q: "What is the range and maximum derivative of the sigmoid, and why does it matter?", a: String.raw`Range $(0,1)$. Its derivative $\sigma(1-\sigma)$ peaks at $x=0$ with value $0.25$ and decays to 0 in both tails. In backpropagation, gradients are multiplied by this derivative at every layer, so in a deep sigmoid network the gradient reaching early layers is scaled by at most $0.25^L$ (with unit-scale weights) — the vanishing gradient problem.` },
      { t: "qa", level: "beginner", q: "Which activation should the output layer use for (a) house-price regression, (b) spam vs not spam, (c) one of 10 digit classes, (d) multiple tags per image?", a: "(a) Linear/identity, since prices are unbounded real numbers (or predict log-price). (b) A single sigmoid giving P(spam). (c) Softmax over 10 logits, giving a distribution that sums to 1. (d) One sigmoid per tag, since tags are not mutually exclusive. In frameworks, (b)–(d) are usually implemented by outputting logits and using a loss with the activation built in (BCEWithLogitsLoss, CrossEntropyLoss)." },
      { t: "qa", level: "intermediate", q: "What is a dead ReLU and how do you prevent it?", a: String.raw`A ReLU unit whose pre-activation is negative for every input in the training set. Its output is always 0 and its local derivative is always 0, so no gradient flows to its incoming weights and bias, and it can never recover. Causes: too high a learning rate, a large negative bias update, poor initialisation. Prevention: smaller learning rate or warmup, He initialisation, batch normalisation, or activations with a non-zero negative side — Leaky ReLU, PReLU, ELU, GELU, SiLU. Diagnose it by logging the fraction of units that are zero for an entire batch or epoch.` },
      { t: "qa", level: "intermediate", q: "Derive the derivative of tanh in terms of tanh itself and relate tanh to sigmoid.", a: String.raw`With $t = \tanh x = \frac{e^x-e^{-x}}{e^x+e^{-x}}$, the quotient rule gives $t' = \frac{(e^x+e^{-x})^2-(e^x-e^{-x})^2}{(e^x+e^{-x})^2} = 1 - t^2$. Multiplying numerator and denominator by $e^{-x}$ gives $\tanh x = \frac{1-e^{-2x}}{1+e^{-2x}} = 2\sigma(2x) - 1$. So tanh is a sigmoid stretched to $(-1,1)$ and compressed horizontally by 2; its slope at 0 is $2\cdot2\cdot0.25 = 1$.` },
      { t: "qa", level: "intermediate", q: "Why subtract the maximum logit before computing softmax? Does it change the result?", a: String.raw`It does not change the result, because softmax is invariant to adding a constant $c$ to all logits: the factor $e^{-c}$ cancels between numerator and denominator. It matters numerically: $e^{z}$ overflows to infinity for $z > 709$ in float64 (about 88 in float32), giving NaN. After subtracting the max, the largest exponent is $e^0 = 1$, all others are in $(0,1]$, and the denominator is at least 1, so neither overflow nor division by zero can occur.` },
      { t: "qa", level: "advanced", q: "Derive the Jacobian of softmax and show that softmax followed by cross-entropy has gradient p − y.", a: String.raw`With $p_i = e^{z_i}/S$, $S=\sum_k e^{z_k}$, the quotient rule gives $\partial p_i/\partial z_j = (\delta_{ij}e^{z_i}S - e^{z_i}e^{z_j})/S^2 = p_i(\delta_{ij}-p_j)$, i.e. $J = \operatorname{diag}(\mathbf p) - \mathbf p\mathbf p^\top$. For $L = -\sum_i y_i\ln p_i$ with $\sum_i y_i = 1$: $\partial L/\partial z_j = -\sum_i \frac{y_i}{p_i}p_i(\delta_{ij}-p_j) = -\sum_i y_i\delta_{ij} + p_j\sum_i y_i = p_j - y_j$. The rows of $J$ sum to zero, so $J$ is singular with the all-ones vector in its null space — the gradient never changes the logits' common offset, matching shift invariance.` },
      { t: "qa", level: "advanced", q: "Compare GELU and ReLU mathematically. Why might GELU work better in Transformers?", a: String.raw`ReLU is $x\cdot\mathbb 1[x>0]$: a deterministic hard gate. GELU is $x\cdot\Phi(x)$: the expected value of $x\cdot m$ where the gate $m \sim \text{Bernoulli}(\Phi(x))$, i.e. a soft gate whose probability of passing $x$ grows smoothly with $x$. Consequences: GELU is infinitely differentiable (ReLU has a kink), it passes small non-zero gradients for negative inputs (no permanently dead units), it is non-monotonic with a minimum of about $-0.17$ near $x \approx -0.75$, and its derivative slightly exceeds 1 for moderate positive $x$. In large attention models trained with Adam, the smooth gradient and absence of dead units have empirically given small but consistent improvements, which is why BERT and GPT adopted it; the cost difference is negligible next to matrix multiplications. For CNNs with batch normalisation the gap is usually small.` },
      { t: "qa", level: "advanced", q: "What conditions make SELU self-normalising, and what breaks them?", a: String.raw`Klambauer et al. (2017) showed that for a dense layer with $n$ inputs whose weights have mean 0 and variance $1/n$ (LeCun normal initialisation), the map from the (mean, variance) of the inputs to the (mean, variance) of the SELU outputs has an attracting fixed point at (0, 1) when $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$. Activations therefore converge to zero mean and unit variance layer after layer, keeping gradients well scaled without batch normalisation. Things that break it: other initialisations (e.g. He or Xavier-uniform with different variance), standard dropout (which changes the variance; ‘alpha dropout’ is designed to preserve it), batch norm, residual additions and convolutions with structured inputs, all of which violate the assumptions of the derivation.` },
    ],

    // --------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "A network has three hidden layers with the identity activation and a sigmoid output. What can it represent?",
            options: [
              "Any continuous function, given enough units",
              "Exactly the same functions as logistic regression on the inputs",
              "Only functions with three linear pieces",
              "Nothing — it cannot be trained",
            ],
            answer: 1,
            explain: "The three linear hidden layers collapse into one affine map W x + b, followed by a sigmoid: that is logistic regression, with a hyperplane decision boundary.",
          },
          {
            kind: "numeric",
            q: String.raw`What is the maximum value of the sigmoid's derivative $\sigma'(x)$?`,
            answer: 0.25,
            tolerance: 0.001,
            explain: String.raw`$\sigma'(x) = \sigma(x)(1-\sigma(x))$ is maximal when $\sigma = 0.5$, i.e. at $x=0$: $0.5\times0.5 = 0.25$.`,
          },
          {
            kind: "numeric",
            q: String.raw`Compute ELU($-1$) with $\alpha = 1$ (to 3 decimals).`,
            answer: -0.632,
            tolerance: 0.002,
            explain: String.raw`$\alpha(e^{-1} - 1) = 0.367879 - 1 = -0.632121$.`,
          },
          {
            kind: "numeric",
            q: String.raw`Compute the softmax probability of the third class for logits $(1, 2, 3)$ (to 3 decimals).`,
            answer: 0.665,
            tolerance: 0.002,
            explain: String.raw`Shift by 3: exponentials $(e^{-2}, e^{-1}, 1) = (0.1353, 0.3679, 1)$, sum $1.5032$, so $p_3 = 1/1.5032 = 0.6652$.`,
          },
          {
            q: "Which statement about ReLU is true?",
            options: [
              "Its derivative is at most 0.25",
              "It is zero-centred",
              "A unit whose pre-activation is negative for every input receives no gradient and stops learning",
              "It saturates for large positive inputs",
            ],
            answer: 2,
            explain: "For x < 0 both the output and the derivative are 0, so a unit negative on all inputs gets zero gradient — a dead ReLU. Its derivative is 1 for x > 0, it is not zero-centred, and it does not saturate on the positive side.",
          },
          {
            q: "You lower the softmax temperature from T = 1 to T = 0.5. What happens to the distribution?",
            options: [
              "It becomes flatter (closer to uniform)",
              "It becomes sharper (more mass on the largest logit)",
              "It does not change, because softmax is shift-invariant",
              "The probabilities no longer sum to 1",
            ],
            answer: 1,
            explain: "Dividing by a smaller T scales the logit differences up, so the largest logit dominates more. For (2, 1, 0.1, −1), p₁ goes from 0.638 to 0.862.",
          },
          {
            kind: "numeric",
            q: String.raw`Ignoring weights, an upper bound on the product of sigmoid derivatives through 5 layers is $0.25^5$. Compute it (to 6 decimals).`,
            answer: 0.000977,
            tolerance: 0.000002,
            explain: String.raw`$0.25^5 = 1/1024 = 0.0009765625$. The gradient is scaled down by at least about a thousand.`,
          },
          {
            q: "Which activation has a learnable parameter trained by backpropagation?",
            options: ["Leaky ReLU", "SELU", "PReLU", "GELU"],
            answer: 2,
            explain: "PReLU learns its negative slope a, with ∂f/∂a = x for x ≤ 0. Leaky ReLU's slope is a fixed hyperparameter; SELU's λ and α are fixed constants; GELU has no parameter.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Softmax and its Jacobian by hand",
        prompt: String.raw`For logits $\mathbf z = (3, 1, 0)$: (a) compute $\mathbf p = \operatorname{softmax}(\mathbf z)$ using the max-subtraction trick, (b) compute the Jacobian entries $\partial p_1/\partial z_1$, $\partial p_1/\partial z_2$ and $\partial p_2/\partial z_2$, (c) explain the sign of each entry.`,
        hint: String.raw`Use $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$.`,
        solution: [
          { t: "p", text: String.raw`(a) Subtract the max 3: shifted logits $(0, -2, -3)$, exponentials $(1, 0.135335, 0.049787)$, sum $1.185122$. So $\mathbf p = (0.843795, 0.114195, 0.042010)$.` },
          { t: "p", text: String.raw`(b) $\partial p_1/\partial z_1 = p_1(1-p_1) = 0.843795 \times 0.156205 = 0.131805$. $\partial p_1/\partial z_2 = -p_1p_2 = -0.843795 \times 0.114195 = -0.096357$. $\partial p_2/\partial z_2 = p_2(1-p_2) = 0.114195\times0.885805 = 0.101155$.` },
          { t: "p", text: "(c) Raising a class's own logit always increases its probability (positive diagonal). Raising another class's logit takes probability mass away, because the total must stay 1 (negative off-diagonal). Note that p₁ is already large, so its own derivative 0.1318 is well below the maximum 0.25: softmax saturates as a class becomes confident, just like the sigmoid." },
        ],
      },
      {
        t: "exercise",
        title: "Stable log-softmax and cross-entropy in NumPy",
        prompt: "Implement `log_softmax(z)` that is safe for logits like 1000, a mean cross-entropy loss over a batch, and its analytic gradient with respect to the logits. Verify the gradient with central finite differences on a batch whose second row contains (1000, 999, 998, 0).",
        hint: "log_softmax(z) = z − (m + log Σ exp(z − m)) with m = max(z). The gradient of the mean loss is (softmax − one_hot) / N.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "stable_cross_entropy.py",
            code: String.raw`import numpy as np

def log_softmax(z):
    """Stable log-softmax along the last axis: z - logsumexp(z)."""
    m = np.max(z, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))
    return z - lse

def cross_entropy(logits, labels):
    lp = log_softmax(logits)
    return -np.mean(lp[np.arange(len(labels)), labels])

def ce_grad(logits, labels):
    """Gradient of mean cross-entropy w.r.t. logits: (softmax - one_hot) / N."""
    p = np.exp(log_softmax(logits))
    p[np.arange(len(labels)), labels] -= 1.0
    return p / len(labels)

logits = np.array([[2.0, 1.0, 0.1, -1.0],
                   [1000.0, 999.0, 998.0, 0.0]])   # second row would overflow a naive softmax
labels = np.array([0, 2])
print("log-softmax row 2:", np.round(log_softmax(logits)[1], 4))
print("loss:", round(cross_entropy(logits, labels), 4))

# finite-difference check of the gradient
h, g_num = 1e-6, np.zeros_like(logits)
for i in range(logits.shape[0]):
    for j in range(logits.shape[1]):
        d = np.zeros_like(logits); d[i, j] = h
        g_num[i, j] = (cross_entropy(logits + d, labels) - cross_entropy(logits - d, labels)) / (2 * h)
print("analytic grad:\n", np.round(ce_grad(logits, labels), 4))
print("max |analytic - numerical|:", f"{np.max(np.abs(ce_grad(logits, labels) - g_num)):.1e}")`,
            output: String.raw`log-softmax row 2: [-4.0760000e-01 -1.4076000e+00 -2.4076000e+00 -1.0004076e+03]
loss: 1.4285
analytic grad:
 [[-0.181   0.1174  0.0477  0.0159]
 [ 0.3326  0.1224 -0.455   0.    ]]
max |analytic - numerical|: 1.6e-08`,
          },
          { t: "p", text: "The second row's loss uses log p₃ = −2.4076 without any overflow, even though e¹⁰⁰⁰ is not representable. The gradient for the first example is p − y for p = (0.6381, 0.2347, 0.0954, 0.0318) and y = (1, 0, 0, 0), divided by the batch size 2: (−0.1810, 0.1174, 0.0477, 0.0159). The finite-difference check agrees to about 1e−8." },
        ],
      },
    ],

    // ---------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "An activation function is applied to each neuron's weighted sum, a = f(z). Its purpose is non-linearity: without it, any stack of layers collapses to one affine map W x + b with W = W_L⋯W_1.",
          "During backpropagation the gradient is multiplied by f′(z) at every layer, so the derivative's size over the typical input range decides whether deep networks can train.",
          String.raw`Sigmoid $\sigma(x) = 1/(1+e^{-x})$, $\sigma' = \sigma(1-\sigma) \le 0.25$, range $(0,1)$: outputs probabilities; saturates and vanishes in deep stacks.`,
          String.raw`Tanh $= 2\sigma(2x)-1$, $\tanh' = 1-\tanh^2 \le 1$, range $(-1,1)$: zero-centred, still saturates.`,
          String.raw`ReLU $\max(0,x)$, derivative 0 or 1: cheap, no vanishing on the active side, but units can die. Leaky ReLU / PReLU keep slope $\alpha$ (fixed / learned) for $x<0$.`,
          String.raw`ELU $\alpha(e^x-1)$ for $x\le0$; SELU $= \lambda\,\mathrm{ELU}_\alpha$ with $\lambda\approx1.0507$, $\alpha\approx1.6733$ is self-normalising under LeCun init.`,
          String.raw`GELU $x\Phi(x)$ (tanh approximation $\tfrac12x[1+\tanh(\sqrt{2/\pi}(x+0.044715x^3))]$), Swish $x\sigma(\beta x)$, Mish $x\tanh(\operatorname{softplus} x)$: smooth, non-monotonic, no dead units; GELU is standard in Transformers.`,
          String.raw`Softmax $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$: shift-invariant, so subtract the max for stability; Jacobian $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$; with cross-entropy the gradient is $\mathbf p - \mathbf y$.`,
          "Output activation follows the task (identity / sigmoid / softmax), and frameworks want logits so they can fuse the activation into a numerically stable loss.",
        ],
      },
    ],

    // ------------------------------------------------------------------- next
    next: [
      {
        t: "list",
        items: [
          "[Multi-Layer Networks and Model Capacity](/learn/m03/multilayer-networks) — now that you know why non-linearity is needed, see what stacking non-linear layers buys: hidden representations and the universal approximation theorem.",
          "[Forward Propagation](/learn/m04/forward-propagation) — activation functions applied layer by layer, in scalar and matrix form, on a full numerical example.",
          "[Backpropagation](/learn/m04/backpropagation) — where every derivative in this lesson is actually used: the element-wise multiplication by f′(z) in each layer's backward step.",
          "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — the saturation argument made precise, together with the weight-scale side of the story and the remedies (ReLU, initialisation, normalisation, residual connections).",
          "[Classification Losses](/learn/m05/classification-losses) — why softmax and sigmoid are paired with cross-entropy, and why losses take logits.",
        ],
      },
    ],
  },
};

export default lesson;
