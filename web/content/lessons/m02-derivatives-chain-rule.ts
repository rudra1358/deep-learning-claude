import type { Lesson } from "@/lib/types";

const SCRATCH_CODE = String.raw`import numpy as np

# ---------- 1. Numerical derivatives: forward vs central differences ----------
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h            # error shrinks like O(h)

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)  # error shrinks like O(h^2)

exact = np.cos(1.0)                         # d/dx sin(x) at x = 1
print(" h        forward error   central error")
for h in [1e-1, 1e-3, 1e-5, 1e-8, 1e-11]:
    ef = abs(forward_diff(np.sin, 1.0, h) - exact)
    ec = abs(central_diff(np.sin, 1.0, h) - exact)
    print(f"{h:7.0e}   {ef:12.3e}   {ec:12.3e}")

# ---------- 2. Gradient vector = direction of steepest ascent ----------
def f2(p):                                  # f(x, y) = x^2 + 3 y^2
    return p[0] ** 2 + 3 * p[1] ** 2

def grad_f2(p):                             # analytic gradient (2x, 6y)
    return np.array([2 * p[0], 6 * p[1]])

p = np.array([1.0, 2.0])
g = grad_f2(p)
numeric_g = np.array([central_diff(lambda t: f2(np.array([t, p[1]])), p[0], 1e-5),
                      central_diff(lambda t: f2(np.array([p[0], t])), p[1], 1e-5)])
print("\ngradient at (1,2): analytic", g, " numeric", np.round(numeric_g, 6))

# directional derivative D_u f = grad . u for many unit vectors u; the max is ||grad||, along grad
angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
slopes = dirs @ g
best = dirs[np.argmax(slopes)]
print(f"steepest slope {slopes.max():.4f} vs ||grad|| {np.linalg.norm(g):.4f}")
gap = np.degrees(np.arccos(np.clip(best @ g / np.linalg.norm(g), -1, 1)))
print(f"angle between best direction and gradient: {gap:.2f} deg (grid spacing 0.1 deg)")

# ---------- 3. Chain rule on the graph L = (sigmoid(w x + b) - y)^2 ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(x, w, b, y):
    u = w * x            # multiply node
    z = u + b            # add node
    a = sigmoid(z)       # sigmoid node
    d = a - y            # subtract node
    L = d ** 2           # square node
    cache = dict(x=x, w=w, u=u, z=z, a=a, d=d)
    return L, cache

def backward(cache):
    dL_dL = 1.0                                   # seed
    dL_dd = dL_dL * 2 * cache["d"]                # square:   local 2d
    dL_da = dL_dd * 1.0                           # subtract: local 1
    dL_dz = dL_da * cache["a"] * (1 - cache["a"]) # sigmoid:  local a(1-a)
    dL_du = dL_dz * 1.0                           # add:      local 1 (to u)
    dL_db = dL_dz * 1.0                           # add:      local 1 (to b)
    dL_dw = dL_du * cache["x"]                    # multiply: local x (to w)
    dL_dx = dL_du * cache["w"]                    # multiply: local w (to x)
    return dict(dd=dL_dd, da=dL_da, dz=dL_dz, dw=dL_dw, db=dL_db, dx=dL_dx)

x, w, b, y = 1.5, 0.8, 0.1, 1.0
L, cache = forward(x, w, b, y)
grads = backward(cache)
print(f"\nforward: u={cache['u']:.4f} z={cache['z']:.4f} a={cache['a']:.6f} d={cache['d']:.6f} L={L:.6f}")
print("backward:", {k: round(float(v), 6) for k, v in grads.items()})

h = 1e-5
fd_w = (forward(x, w + h, b, y)[0] - forward(x, w - h, b, y)[0]) / (2 * h)
fd_b = (forward(x, w, b + h, y)[0] - forward(x, w, b - h, y)[0]) / (2 * h)
print(f"dL/dw chain={grads['dw']:.8f} finite-diff={fd_w:.8f}")
print(f"dL/db chain={grads['db']:.8f} finite-diff={fd_b:.8f}")

# ---------- 4. Same graph on a batch: gradients of the mean loss ----------
rng = np.random.default_rng(0)
xs = rng.normal(size=8)
ys = (xs > 0).astype(float)
Ls, c = forward(xs, w, b, ys)                     # NumPy evaluates every node element-wise
mean_L = Ls.mean()
g = backward(c)
dw_batch = g["dw"].mean()                         # d(mean L)/dw = mean of per-sample dL/dw
db_batch = g["db"].mean()
fd_wb = (forward(xs, w + h, b, ys)[0].mean() - forward(xs, w - h, b, ys)[0].mean()) / (2 * h)
print(f"\nbatch mean loss={mean_L:.6f}  dw={dw_batch:.6f} (fd {fd_wb:.6f})  db={db_batch:.6f}")
`;

const SCRATCH_OUTPUT = String.raw` h        forward error   central error
  1e-01      4.294e-02      9.001e-04
  1e-03      4.208e-04      9.005e-08
  1e-05      4.207e-06      1.114e-11
  1e-08      2.970e-09      2.581e-09
  1e-11      1.169e-06      1.169e-06

gradient at (1,2): analytic [ 2. 12.]  numeric [ 2. 12.]
steepest slope 12.1655 vs ||grad|| 12.1655
angle between best direction and gradient: 0.04 deg (grid spacing 0.1 deg)

forward: u=1.2000 z=1.3000 a=0.785835 d=-0.214165 L=0.045867
backward: {'dd': -0.42833, 'da': -0.42833, 'dz': -0.072087, 'dw': -0.108131, 'db': -0.072087, 'dx': -0.05767}
dL/dw chain=-0.10813086 finite-diff=-0.10813086
dL/db chain=-0.07208724 finite-diff=-0.07208724

batch mean loss=0.158301  dw=-0.075025 (fd -0.075025)  db=-0.070353`;

const TORCH_CODE = String.raw`import torch

# Leaf tensors with requires_grad=True are the quantities we want gradients for
x = torch.tensor(1.5)
y = torch.tensor(1.0)
w = torch.tensor(0.8, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

# Forward pass: autograd records each operation as a node in a graph
u = w * x
z = u + b
a = torch.sigmoid(z)
d = a - y
L = d ** 2
for t in (u, z, a, d, L):
    t.retain_grad()                      # keep gradients of intermediate (non-leaf) nodes
print(f"L = {L.item():.6f}   grad_fn of L: {type(L.grad_fn).__name__}")

# Backward pass: reverse-mode chain rule from L back to every leaf
L.backward()
print(f"dL/dd = {d.grad.item():.6f}")
print(f"dL/da = {a.grad.item():.6f}")
print(f"dL/dz = {z.grad.item():.6f}")
print(f"dL/dw = {w.grad.item():.6f}")
print(f"dL/db = {b.grad.item():.6f}")

# torch.autograd.grad returns gradients without accumulating into .grad
w2 = torch.tensor(0.8, requires_grad=True)
L2 = (torch.sigmoid(w2 * x + 0.1) - y) ** 2
(g,) = torch.autograd.grad(L2, w2)
print(f"autograd.grad dL/dw = {g.item():.6f}")

# Gradient check in float64 with torch.autograd.gradcheck (finite differences vs autograd)
def f(w_, b_):
    return (torch.sigmoid(w_ * 1.5 + b_) - 1.0) ** 2
inputs = (torch.tensor(0.8, dtype=torch.float64, requires_grad=True),
          torch.tensor(0.1, dtype=torch.float64, requires_grad=True))
print("gradcheck passed:", torch.autograd.gradcheck(f, inputs))

# Gradient of a multivariable function: f(p) = p0^2 + 3 p1^2 at (1, 2)
p = torch.tensor([1.0, 2.0], requires_grad=True)
(p[0] ** 2 + 3 * p[1] ** 2).backward()
print("grad f(1,2) =", p.grad)
`;

const TORCH_OUTPUT = String.raw`L = 0.045867   grad_fn of L: PowBackward0
dL/dd = -0.428330
dL/da = -0.428330
dL/dz = -0.072087
dL/dw = -0.108131
dL/db = -0.072087
autograd.grad dL/dw = -0.108131
gradcheck passed: True
grad f(1,2) = tensor([ 2., 12.])`;

const TF_CODE = String.raw`import tensorflow as tf

x = tf.constant(1.5)
y = tf.constant(1.0)
w = tf.Variable(0.8)          # trainable variables are watched by the tape automatically
b = tf.Variable(0.1)

# GradientTape records the forward computation ...
with tf.GradientTape() as tape:
    z = w * x + b
    a = tf.sigmoid(z)
    L = (a - y) ** 2

# ... and replays it backwards (reverse-mode chain rule) to get dL/dw and dL/db
dL_dw, dL_db = tape.gradient(L, [w, b])
print(f"L = {float(L):.6f}  dL/dw = {float(dL_dw):.6f}  dL/db = {float(dL_db):.6f}")
`;

const EX_CODE = String.raw`import numpy as np

def numerical_gradient(f, w, h=1e-5):
    """Central-difference estimate of grad f at w, one coordinate at a time."""
    g = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 3))              # 20 samples, 3 features
y = rng.choice([-1.0, 1.0], size=20)      # labels in {-1, +1}

def loss(w):
    m = y * (X @ w)                       # margins m_i = y_i * (w . x_i)
    return np.mean(np.log1p(np.exp(-m)))  # mean of log(1 + e^{-m_i})

def grad(w):
    # chain rule: dL_i/dm_i = -sigmoid(-m_i),  dm_i/dw = y_i x_i
    m = y * (X @ w)
    return X.T @ (-sigmoid(-m) * y) / len(y)

w = rng.normal(size=3)
ga, gn = grad(w), numerical_gradient(loss, w)
rel_err = np.linalg.norm(ga - gn) / (np.linalg.norm(ga) + np.linalg.norm(gn))
print("analytic :", np.round(ga, 6))
print("numerical:", np.round(gn, 6))
print(f"relative error: {rel_err:.2e}")
`;

const EX_OUTPUT = String.raw`analytic : [ 0.057654 -0.081311  0.133202]
numerical: [ 0.057654 -0.081311  0.133202]
relative error: 1.66e-11`;

const lesson: Lesson = {
  id: "m02/derivatives-chain-rule",
  title: "Derivatives, Partial Derivatives and the Chain Rule",
  summary:
    "Rates of change, partial derivatives, the gradient vector and the chain rule — and how applying the chain rule on a computational graph is exactly backpropagation, checked numerically with finite differences.",
  minutes: 60,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices"],
  next: ["m02/jacobians-hessians", "m04/backpropagation", "m06/gradient-descent", "m04/gradient-checking"],
  terms: ["function", "limit", "derivative", "partial-derivative", "gradient", "chain-rule", "computational-graph", "finite-difference"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "Training a neural network means repeatedly asking one question: if I nudge this weight a little, does the loss go up or down, and by how much? The **derivative** is the precise answer. It measures the instantaneous rate of change of a function's output with respect to its input — the slope of the function at a point.",
      },
      {
        t: "p",
        text: "A network has not one weight but millions, so we need **partial derivatives** (the rate of change with respect to one input while the others are held fixed) and the **gradient**, the vector collecting all of them. The gradient points in the direction in which the loss increases fastest, so stepping against it decreases the loss — that is gradient descent.",
      },
      {
        t: "p",
        text: "The loss of a network is a long composition of simple functions: multiply by weights, add biases, apply activations, compare with the target. The **chain rule** says how to differentiate a composition: multiply the derivatives of the pieces. Organising that multiplication on a **computational graph**, from the loss backwards, is the **backpropagation** algorithm. Every deep-learning framework — PyTorch's autograd, TensorFlow's GradientTape, JAX's `grad` — is an implementation of this idea.",
      },
      {
        t: "p",
        text: "Beyond training, derivatives with respect to the input rather than the weights are used to explain predictions (gradient-based saliency maps), to construct adversarial examples (the fast gradient sign method perturbs an image along the sign of the input gradient), and in physics-informed models whose loss contains derivatives of the network's output.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The one idea to keep",
        text: String.raw`For a chain $x \to u \to L$, $\frac{dL}{dx} = \frac{dL}{du}\cdot\frac{du}{dx}$: the sensitivity of the output to an early variable is the product of the local sensitivities along the path (summed over paths when there are several). Backpropagation computes these products for every parameter in one backward sweep.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`**Functions and their graphs.** Reading $y = f(x)$ as a rule that turns an input into an output, and picturing it as a curve. Familiarity with $x^n$, $e^x$, $\ln x$, $\sin x$.`,
          String.raw`**Slope of a line.** Slope $= \Delta y / \Delta x$ ("rise over run"). The derivative is the slope of the tangent line, obtained as a limit of such slopes.`,
          "**Vectors and the dot product** ([Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices)). The gradient is a vector, and the rate of change in a direction is a dot product with the gradient — its geometric meaning ($\\|\\mathbf{u}\\|\\|\\mathbf{v}\\|\\cos\\theta$) is what proves the gradient is the steepest-ascent direction.",
          "**Basic NumPy and Python functions.** The code passes functions as arguments (`f(x + h)`) and evaluates them on arrays.",
        ],
      },
      {
        t: "p",
        text: "No prior calculus course is assumed: limits and every derivative rule used are introduced here, with the emphasis on what deep learning needs.",
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Gears in a gearbox",
        text: "Three gears are meshed in a row. Each turn of gear A turns gear B 3 times; each turn of B turns gear C 2 times. How many times does C turn per turn of A? 3 × 2 = 6. Each gear only knows its local ratio with its neighbour, yet the overall ratio is simply the product of the local ratios. The chain rule is this statement for functions: the rate at which the loss changes with a weight is the product of the local rates along the chain of computations connecting them.",
      },
      {
        t: "p",
        text: "**Derivative.** Zoom in far enough on any smooth curve and it looks like a straight line. The slope of that line is the derivative. A derivative of 3 at a point means: near this point, increasing the input by a tiny amount $\\varepsilon$ increases the output by about $3\\varepsilon$. A negative derivative means the output decreases; zero means the curve is momentarily flat (as at the bottom of a valley).",
      },
      {
        t: "p",
        text: "**Partial derivatives and the gradient.** Picture standing on a hillside in fog, where the height is a function of your east-west and north-south position. The partial derivative with respect to 'east' is the slope you would feel stepping east; the one for 'north', stepping north. Together they form the gradient, an arrow on the ground pointing in the steepest uphill direction; its length is how steep that is. To go downhill as fast as possible, walk against the arrow — that is all gradient descent does, in millions of dimensions.",
      },
      {
        t: "p",
        text: "**Computational graph.** Write a calculation as a flowchart: inputs on the left, each box doing one simple operation, the loss on the right. The forward pass pushes numbers through the boxes. The backward pass pushes sensitivities back: each box receives 'how much the loss cares about my output' from the right, multiplies it by its own local slope, and passes 'how much the loss cares about my input' to the left. No box needs to understand the whole network.",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "From slopes to derivatives" },
      {
        t: "steps",
        items: [
          { title: "Pick two nearby points", text: "On the curve $y = f(x)$, take $x$ and $x + h$. The secant line through them has slope $\\frac{f(x+h) - f(x)}{h}$." },
          { title: "Shrink the gap", text: "Let $h$ get smaller and smaller. For a smooth function the secant slopes settle on one number." },
          { title: "Take the limit", text: "That limiting number is the derivative $f'(x)$, the slope of the tangent line." },
          { title: "Use rules instead of limits", text: "In practice derivatives are not computed from the limit each time; a small set of rules (power, exponential, product, chain) covers every function built from elementary pieces." },
        ],
      },
      { t: "h", text: "From one input to many" },
      {
        t: "steps",
        items: [
          { title: "Freeze all but one input", text: "For $f(x_1, \\dots, x_n)$, treat every input except $x_i$ as a constant. The result is a one-variable function." },
          { title: "Differentiate it", text: "Its derivative is the partial derivative $\\partial f / \\partial x_i$." },
          { title: "Collect them", text: "Stack the $n$ partials into the gradient vector $\\nabla f$. Its direction is steepest ascent; its length is the slope in that direction." },
        ],
      },
      { t: "h", text: "Backpropagation on a computational graph" },
      {
        t: "steps",
        items: [
          { title: "Decompose", text: "Break the expression into primitive operations, each a node: here $u = wx$, $z = u + b$, $a = \\sigma(z)$, $d = a - y$, $L = d^2$." },
          { title: "Forward pass", text: "Evaluate the nodes left to right and **store** every intermediate value; local derivatives such as $\\sigma'(z) = a(1-a)$ need them." },
          { title: "Seed", text: "Set the gradient of the output with respect to itself: $\\partial L / \\partial L = 1$." },
          { title: "Visit nodes in reverse order", text: "For each node, take the upstream gradient $\\partial L / \\partial(\\text{node output})$ and multiply it by the local derivative of the output with respect to each input. That gives $\\partial L / \\partial(\\text{input})$." },
          { title: "Accumulate over paths", text: "If a value feeds several nodes, it receives a gradient contribution from each; **add** them. This is the multivariable chain rule." },
          { title: "Read off parameter gradients", text: "When the sweep reaches the leaves, $\\partial L / \\partial w$ and $\\partial L / \\partial b$ are available — every parameter's gradient from one backward pass." },
          { title: "Verify numerically", text: "Perturb a parameter by $\\pm h$, rerun the forward pass, and compare the finite-difference slope with the backpropagated value." },
        ],
      },
      {
        t: "p",
        text: "Each primitive has a characteristic backward behaviour worth memorising. **Addition** copies the upstream gradient unchanged to each input (local derivative 1). **Multiplication** sends each input the upstream gradient times the other input. **Subtraction** is addition with a sign flip on the second input. **Element-wise functions** like sigmoid multiply by their derivative evaluated at the stored forward value. **Squaring** multiplies by twice its input.",
      },
      {
        t: "p",
        text: "Frameworks implement exactly this. During the forward pass, PyTorch's autograd records each operation and the tensors it will need (the 'tape' or graph, visible as `grad_fn`); `loss.backward()` walks that graph in reverse topological order calling each operation's backward function, and writes the results into `.grad` of every leaf tensor with `requires_grad=True`. See [Backpropagation](/learn/m04/backpropagation) for the full layer-by-layer version.",
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "Functions and limits" },
      {
        t: "p",
        text: String.raw`A **function** $f: \mathbb{R}^n \to \mathbb{R}^m$ assigns to each input exactly one output. A **limit** $\lim_{x \to a} f(x) = L$ means that $f(x)$ can be made as close to $L$ as we like by taking $x$ close enough to $a$ (but not equal to it). Limits let us talk about the slope at a single point, where the ordinary "rise over run" would divide by zero.`,
      },
      { t: "h", text: "The derivative" },
      {
        t: "math",
        tex: String.raw`f'(x) \;=\; \frac{df}{dx} \;=\; \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}`,
        caption: "$h$ is the size of the nudge to the input; the fraction is the slope of the secant line.",
      },
      {
        t: "p",
        text: String.raw`**Example from the definition.** For $f(x) = x^2$: $\frac{(x+h)^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h \to 2x$ as $h \to 0$. At $x = 3$ the secant slopes for $h = 1, 0.1, 0.01, 0.001$ are $7, 6.1, 6.01, 6.001$ — approaching $f'(3) = 6$, with error exactly $h$.`,
      },
      { t: "h", text: "Rules you will use constantly" },
      {
        t: "table",
        head: ["Rule", "Formula", "Example"],
        rows: [
          ["Constant", "$\\frac{d}{dx} c = 0$", "$\\frac{d}{dx} 5 = 0$"],
          ["Power", "$\\frac{d}{dx} x^n = n x^{n-1}$", "$\\frac{d}{dx} x^3 = 3x^2$"],
          ["Exponential / log", "$\\frac{d}{dx} e^x = e^x$, $\\frac{d}{dx} \\ln x = \\frac{1}{x}$", "$\\frac{d}{dx} e^{2x} = 2e^{2x}$ (chain rule)"],
          ["Sum and constant multiple", "$(af + bg)' = af' + bg'$", "$\\frac{d}{dx}(3x^2 + x) = 6x + 1$"],
          ["Product", "$(fg)' = f'g + fg'$", "$\\frac{d}{dx}(x e^x) = e^x + x e^x$"],
          ["Quotient", "$(f/g)' = (f'g - fg')/g^2$", "$\\frac{d}{dx}\\frac{1}{x} = -\\frac{1}{x^2}$"],
          ["Chain", "$(f \\circ g)'(x) = f'(g(x))\\,g'(x)$", "$\\frac{d}{dx}\\sin(x^2) = \\cos(x^2)\\cdot 2x$"],
          ["Sigmoid", "$\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$", "$\\sigma'(0) = 0.5 \\times 0.5 = 0.25$"],
          ["Tanh", "$\\tanh'(z) = 1 - \\tanh^2(z)$", "$\\tanh'(0) = 1$"],
          ["ReLU", "$\\mathrm{ReLU}'(z) = 1$ if $z > 0$, $0$ if $z < 0$", "undefined at 0; frameworks use 0"],
        ],
      },
      {
        t: "p",
        text: String.raw`**Deriving the sigmoid derivative.** $\sigma(z) = (1 + e^{-z})^{-1}$. By the chain rule, $\sigma'(z) = -(1 + e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1 + e^{-z}}\cdot\frac{e^{-z}}{1 + e^{-z}}$. The second factor is $1 - \sigma(z)$, so $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Its maximum is $0.25$ at $z = 0$, and it approaches 0 for large $|z|$ — the root of the vanishing-gradient problem in deep sigmoid networks.`,
      },
      { t: "h", text: "Partial derivatives" },
      {
        t: "math",
        tex: String.raw`\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_n)}{h}`,
      },
      {
        t: "p",
        text: String.raw`Only $x_i$ moves; all other inputs are held constant. Example: $f(x, y) = x^2 y$. Treating $y$ as a constant, $\frac{\partial f}{\partial x} = 2xy$; treating $x$ as a constant, $\frac{\partial f}{\partial y} = x^2$. At $(3, 2)$: $\frac{\partial f}{\partial x} = 12$ and $\frac{\partial f}{\partial y} = 9$.`,
      },
      { t: "h", text: "The gradient and steepest ascent" },
      {
        t: "math",
        tex: String.raw`\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right)^{\!\top}, \qquad D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} \quad (\|\mathbf{u}\| = 1)`,
      },
      {
        t: "p",
        text: String.raw`$D_{\mathbf{u}} f$ is the **directional derivative**: the slope when moving in unit direction $\mathbf{u}$. By the geometric dot product, $D_{\mathbf{u}} f = \|\nabla f\|\,\|\mathbf{u}\|\cos\theta = \|\nabla f\|\cos\theta$, where $\theta$ is the angle between $\mathbf{u}$ and the gradient. This is largest ($= \|\nabla f\|$) when $\theta = 0$, i.e. moving along $\nabla f$; most negative when moving along $-\nabla f$; and zero for directions perpendicular to $\nabla f$, which run along the **level set** (contour) through the point. Hence: the gradient points in the direction of steepest ascent, its length is the steepest slope, and it is perpendicular to contour lines. Gradient descent updates parameters $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla_{\boldsymbol{\theta}} L$ with learning rate $\eta > 0$.`,
      },
      {
        t: "p",
        text: String.raw`**Example.** $f(x, y) = x^2 + 3y^2$ has $\nabla f = (2x, 6y)$. At $(1, 2)$: $\nabla f = (2, 12)$, $\|\nabla f\| = \sqrt{4 + 144} = \sqrt{148} \approx 12.1655$. No direction has a slope steeper than 12.1655 — the NumPy program checks this by trying 3,600 directions.`,
      },
      { t: "h", text: "The chain rule" },
      {
        t: "math",
        tex: String.raw`y = g(x),\; L = f(y) \;\;\Longrightarrow\;\; \frac{dL}{dx} = \frac{dL}{dy}\cdot\frac{dy}{dx} = f'(g(x))\, g'(x)`,
      },
      {
        t: "p",
        text: String.raw`**Why it holds.** If $x$ changes by a small $\Delta x$, then $y$ changes by about $\Delta y \approx g'(x)\Delta x$, and that change in $y$ changes $L$ by about $\Delta L \approx f'(y)\Delta y \approx f'(y)g'(x)\Delta x$. Dividing by $\Delta x$ and letting it shrink gives the rule. For longer chains $x \to y_1 \to y_2 \to \dots \to L$ the local derivatives simply multiply: $\frac{dL}{dx} = \frac{dL}{dy_k}\frac{dy_k}{dy_{k-1}}\cdots\frac{dy_1}{dx}$.`,
      },
      { t: "h", text: "The multivariable chain rule" },
      {
        t: "math",
        tex: String.raw`L = f(y_1, \dots, y_m),\; y_j = g_j(x) \;\;\Longrightarrow\;\; \frac{dL}{dx} = \sum_{j=1}^{m} \frac{\partial L}{\partial y_j}\,\frac{dy_j}{dx}`,
        caption: "When $x$ influences $L$ through several paths, the contributions of the paths add.",
      },
      {
        t: "p",
        text: String.raw`**Example.** $f(x, y) = xy$ with $x = t^2$ and $y = \sin t$. Then $\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} = y\cdot 2t + x \cdot\cos t = 2t\sin t + t^2\cos t$. At $t = 1$: $2(0.841471) + 0.540302 = 2.223244$. A simple special case: $L = x \cdot x$ viewed as a multiplication node whose two inputs are both $x$ gives $\frac{dL}{dx} = x + x = 2x$ — forgetting to sum the two paths would give the wrong answer $x$.`,
      },
      { t: "h", text: "Worked example: the computational graph in the simulator" },
      {
        t: "p",
        text: String.raw`Single-neuron regression with squared error, $L = (\sigma(wx + b) - y)^2$, with input $x = 1.5$, weight $w = 0.8$, bias $b = 0.1$, target $y = 1$. Here $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid. All numbers are rounded to 6 decimals from a float64 computation.`,
      },
      {
        t: "steps",
        items: [
          { title: "Forward: multiply", tex: String.raw`u = w x = 0.8 \times 1.5 = 1.2` },
          { title: "Forward: add bias", tex: String.raw`z = u + b = 1.2 + 0.1 = 1.3` },
          { title: "Forward: sigmoid", tex: String.raw`a = \sigma(1.3) = \frac{1}{1 + e^{-1.3}} = \frac{1}{1 + 0.272532} = 0.785835` },
          { title: "Forward: subtract target", tex: String.raw`d = a - y = 0.785835 - 1 = -0.214165` },
          { title: "Forward: square", tex: String.raw`L = d^2 = (-0.214165)^2 = 0.045867` },
          { title: "Backward: seed and square node", tex: String.raw`\frac{\partial L}{\partial L} = 1, \qquad \frac{\partial L}{\partial d} = 1 \cdot 2d = -0.428330` },
          { title: "Backward: subtract node", tex: String.raw`\frac{\partial L}{\partial a} = -0.428330 \cdot 1 = -0.428330, \qquad \frac{\partial L}{\partial y} = -0.428330 \cdot (-1) = 0.428330` },
          { title: "Backward: sigmoid node", tex: String.raw`\sigma'(z) = a(1-a) = 0.785835 \times 0.214165 = 0.168298, \qquad \frac{\partial L}{\partial z} = -0.428330 \times 0.168298 = -0.072087` },
          { title: "Backward: add node", tex: String.raw`\frac{\partial L}{\partial u} = \frac{\partial L}{\partial b} = -0.072087 \cdot 1 = -0.072087` },
          { title: "Backward: multiply node", tex: String.raw`\frac{\partial L}{\partial w} = -0.072087 \cdot x = -0.072087 \times 1.5 = -0.108131, \qquad \frac{\partial L}{\partial x} = -0.072087 \cdot w = -0.057670` },
        ],
      },
      {
        t: "p",
        text: String.raw`Collapsing the chain gives the closed form $\frac{\partial L}{\partial w} = 2(a - y)\cdot a(1 - a)\cdot x$, one factor per node on the path from $L$ back to $w$. Both gradients are negative, so increasing $w$ or $b$ would decrease the loss — sensible, since the prediction $0.786$ is below the target $1$. A gradient-descent step with $\eta = 0.5$ would set $w \leftarrow 0.8 - 0.5 \times (-0.108131) = 0.854065$.`,
      },
      { t: "h", text: "Numerical derivatives by finite differences" },
      {
        t: "math",
        tex: String.raw`\underbrace{\frac{f(x+h) - f(x)}{h}}_{\text{forward: error } O(h)} \qquad\qquad \underbrace{\frac{f(x+h) - f(x-h)}{2h}}_{\text{central: error } O(h^2)}`,
      },
      {
        t: "p",
        text: String.raw`**Where the error orders come from.** Taylor's theorem gives $f(x \pm h) = f(x) \pm h f'(x) + \tfrac{h^2}{2} f''(x) \pm \tfrac{h^3}{6} f'''(x) + \dots$. The forward difference therefore equals $f'(x) + \tfrac{h}{2}f''(x) + \dots$ (error proportional to $h$). Subtracting the two expansions cancels the even terms, so the central difference equals $f'(x) + \tfrac{h^2}{6}f'''(x) + \dots$ (error proportional to $h^2$). Making $h$ smaller is not always better: the numerator subtracts two nearly equal floating-point numbers, losing roughly $\varepsilon/h$ in accuracy, where $\varepsilon \approx 2.2 \times 10^{-16}$ in float64. Balancing the two errors gives a best $h$ around $10^{-8}$ for forward and $10^{-5}$ for central differences — exactly the pattern in the error table printed by the NumPy program.`,
      },
      {
        t: "p",
        text: String.raw`For the worked example, central differences on $w$ give $-0.108672$ with $h = 0.1$, $-0.108136$ with $h = 0.01$, and $-0.1081308655$ with $h = 10^{-4}$, converging to the chain-rule value $-0.1081308650$.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The simulator draws the graph of $L = (\\sigma(wx + b) - y)^2$. Press **Step** to run the forward pass (values appear in the accent colour under each node), then the backward pass (gradients $\\partial L/\\partial(\\text{node})$ appear in clay above each node, and the edge being crossed is drawn as a clay arrow labelled with its local derivative). The last step compares the backpropagated $\\partial L/\\partial w$ and $\\partial L/\\partial b$ with central finite differences. The table under the graph lists every value and gradient.",
      },
      { t: "sim", id: "chain-rule", caption: "Forward values (accent) and backward gradients (clay) on a five-node computational graph." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "Run the default example to the end and compare every number with the worked example: $L = 0.0459$, $\\partial L/\\partial w = -0.1081$, $\\partial L/\\partial b = -0.0721$.",
          "On the multiply node, notice the 'swap': the gradient reaching $w$ is the upstream gradient times $x$, and the gradient reaching $x$ is the upstream gradient times $w$. Set $x = 0$ and $\\partial L/\\partial w$ becomes exactly 0 — a weight attached to a zero input cannot be learned from that example.",
          "Move the sliders until the prediction $a$ is very close to $y$ (for example $y = 0.8$ with the default $x, w, b$). Every gradient shrinks toward 0 because the square node's local derivative $2d$ is near 0: a correct prediction produces no learning signal.",
          "Set $w = 3$ and $x = 3$ so that $z$ is large. The sigmoid saturates, $\\sigma'(z)$ on the sigmoid edge becomes tiny, and the gradients for $w$ and $b$ nearly vanish even though the loss can still be large if $y$ is small. This is the vanishing-gradient effect in a single neuron.",
          "Set $y$ above versus below the prediction $a$ and watch the sign of every gradient flip. The subtract node's edge to $y$ always carries $-1$, so $\\partial L/\\partial y$ is the negative of $\\partial L/\\partial a$.",
          "At the final step, change any slider: the finite-difference estimate and the chain-rule value continue to agree to around $10^{-9}$, which is how you test a hand-written backward pass.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "The program implements forward and central differences and measures their error against the exact derivative of $\\sin$, checks that the gradient of $f(x, y) = x^2 + 3y^2$ is the steepest-ascent direction, and then implements the computational graph from the worked example as an explicit forward function (which caches intermediate values) and a backward function (which applies the chain rule node by node). Finally it checks the gradients with finite differences, for one example and for a batch.",
      },
      { t: "code", lang: "python", title: "chain_rule_from_scratch.py", code: SCRATCH_CODE, output: SCRATCH_OUTPUT },
      {
        t: "p",
        text: "Read the error table carefully. The forward-difference error falls by 100× when $h$ falls by 100× (first order); the central-difference error falls by 10,000× (second order) until about $h = 10^{-5}$. Below that, rounding error takes over and both get worse — at $h = 10^{-11}$ the two estimates are equally poor. Gradient checks therefore use central differences in float64 with $h$ around $10^{-5}$ to $10^{-6}$.",
      },
      {
        t: "p",
        text: "The batch section works because every node is element-wise: NumPy evaluates the same graph for all 8 examples at once, and the derivative of a mean is the mean of the derivatives.",
      },
    ],

    /* ------------------------------------------------------------------ */
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "Autograd builds the computational graph while the forward code runs (each result's `grad_fn` names the backward function of the operation that produced it). `L.backward()` performs the reverse sweep. By default only leaf tensors keep their `.grad`; `retain_grad()` keeps intermediate gradients so they can be compared with the hand computation. `torch.autograd.gradcheck` automates the finite-difference comparison and should be run in float64.",
              },
              { t: "code", lang: "python", title: "chain_rule_autograd.py", code: TORCH_CODE, output: TORCH_OUTPUT, runnable: false },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "TensorFlow records operations inside a `tf.GradientTape` context; `tape.gradient(L, [w, b])` runs reverse-mode differentiation over the recorded operations. `tf.Variable`s are watched automatically; plain tensors need `tape.watch(t)`.",
              },
              {
                t: "code",
                lang: "python",
                title: "chain_rule_gradient_tape.py (not executed in CI)",
                code: TF_CODE,
                output: "Prints one line with L, dL/dw and dL/db. These are the same quantities as in the PyTorch tab (0.045867, -0.108131, -0.072087), up to float32 rounding in the last digit.",
                runnable: false,
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      { t: "h", text: "Kinds of derivative" },
      {
        t: "table",
        head: ["Object", "For a function", "Shape", "Deep-learning role"],
        rows: [
          ["Derivative $f'(x)$", "$\\mathbb{R} \\to \\mathbb{R}$", "scalar", "activation derivatives $\\sigma'$, $\\tanh'$, ReLU$'$"],
          ["Partial derivative $\\partial f/\\partial x_i$", "$\\mathbb{R}^n \\to \\mathbb{R}$", "scalar", "sensitivity of the loss to one weight"],
          ["Gradient $\\nabla f$", "$\\mathbb{R}^n \\to \\mathbb{R}$", "vector $(n)$", "the update direction for all parameters"],
          ["Directional derivative $\\nabla f \\cdot \\mathbf{u}$", "$\\mathbb{R}^n \\to \\mathbb{R}$", "scalar", "slope along a search direction (line search, analysis of descent)"],
          ["Jacobian $J$", "$\\mathbb{R}^n \\to \\mathbb{R}^m$", "$m \\times n$ matrix", "local derivative of a whole layer; backprop multiplies by its transpose"],
          ["Hessian $H$", "$\\mathbb{R}^n \\to \\mathbb{R}$", "$n \\times n$ matrix", "curvature: step sizes, second-order methods, sharpness"],
        ],
      },
      {
        t: "p",
        text: "Jacobians and Hessians are developed in [Jacobians, Hessians and Computational Graphs](/learn/m02/jacobians-hessians).",
      },
      { t: "h", text: "Ways to compute derivatives" },
      {
        t: "table",
        head: ["Method", "How", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["Symbolic differentiation", "apply rules to the formula, producing a new formula", "exact; readable result", "expressions can blow up in size; awkward with loops and branches", "computer-algebra systems; deriving formulas by hand"],
          ["Finite differences", "evaluate $f$ at perturbed inputs", "trivial to implement; works on black boxes", "approximate; one or two extra evaluations per input — hopeless for millions of parameters", "gradient checking, debugging"],
          ["Forward-mode autodiff", "carry derivatives forward alongside values", "exact; low memory", "one pass per input direction", "functions with few inputs and many outputs; Jacobian-vector products"],
          ["Reverse-mode autodiff (backprop)", "record the forward pass, then propagate gradients backward", "exact; all $n$ partials of a scalar output for a small constant multiple of one forward pass", "must store intermediate values (memory)", "training neural networks — scalar loss, many parameters"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Training every neural network.** Each optimisation step computes $\\nabla_{\\boldsymbol\\theta} L$ by reverse-mode differentiation — PyTorch autograd, TensorFlow GradientTape and JAX `grad` all implement the graph-based chain rule from this lesson.",
          "**Testing new layers.** When a custom operation has a hand-written backward function, it is verified against finite differences (for example with `torch.autograd.gradcheck`) before being trusted. See [Numerical Gradient Checking](/learn/m04/gradient-checking).",
          "**Explaining predictions.** Gradient-based saliency maps display $|\\partial(\\text{class score})/\\partial(\\text{pixel})|$, the sensitivity of a classifier's output to each input pixel.",
          "**Adversarial robustness.** The fast gradient sign method perturbs an input by a small step along the sign of $\\nabla_{\\mathbf{x}} L$, often flipping the prediction; adversarial training uses such examples to harden models.",
          "**Physics-informed and scientific models.** Losses can include derivatives of the network output with respect to its inputs (for example the terms of a differential equation), computed by the same autodiff machinery.",
          "**Sensitivity analysis.** In engineering and finance, partial derivatives of a model's output with respect to its inputs quantify which inputs matter most.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      { t: "h", text: "Strengths of derivative-based learning" },
      {
        t: "list",
        items: [
          "**Cheap.** Reverse mode computes the gradient with respect to every parameter for a small constant multiple of the cost of one forward pass, independent of the number of parameters. This is what makes training billion-parameter models feasible.",
          "**Local and modular.** Each operation only needs its own local derivative, so new layers can be added to a framework without touching the rest of the system.",
          "**Exact.** Automatic differentiation is exact up to floating-point rounding, unlike finite differences.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Only local information.** A derivative describes the function in an infinitesimal neighbourhood. Following it can lead to local minima, plateaus or saddle points, and a step that is too large can overshoot.",
          "**Requires differentiability.** Operations such as `argmax`, rounding or sampling a discrete value have zero or undefined derivatives. Workarounds (straight-through estimators, the reparameterisation trick, continuous relaxations) are approximations. Kinks such as ReLU at 0 are handled by convention (a subgradient).",
          String.raw`**Products of many factors.** A deep chain multiplies many local derivatives. If they are mostly below 1 (sigmoid's never exceeds 0.25) the gradient vanishes; if above 1 it explodes. Activation choice, initialisation, normalisation and residual connections exist largely to control this product.`,
          "**Memory.** Reverse mode must keep the forward pass's intermediate values until the backward pass uses them; activation memory often dominates GPU memory in training. Gradient checkpointing trades recomputation for memory.",
          "**Finite differences do not scale.** They need about two forward passes per parameter and are approximate, so they are a test tool, not a training method.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Dropping the inner derivative.** Writing $\\frac{d}{dx}\\sin(x^2) = \\cos(x^2)$ forgets the factor $2x$. Every application of the chain rule multiplies by the derivative of the inside function. **Fix:** name the intermediate ($u = x^2$) and write both factors explicitly.",
          "**Not summing over multiple paths.** If a value is used twice (a shared weight, a residual connection, $x \\cdot x$), its gradient is the sum of the contributions from every use. Overwriting instead of accumulating gives wrong gradients. Frameworks accumulate into `.grad` for exactly this reason.",
          "**Evaluating a local derivative at the wrong point.** $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$ must use the stored output $a = \\sigma(z)$, not $z$ itself. Writing `z * (1 - z)` is a common bug that still runs.",
          "**Thinking the gradient points downhill.** It points uphill (steepest ascent). Minimisation subtracts it: $\\boldsymbol\\theta - \\eta\\nabla L$. Adding it maximises the loss.",
          "**Poor finite-difference checks.** One-sided differences, float32 arithmetic, or $h$ that is too small (e.g. $10^{-12}$) produce large discrepancies even for correct code. **Fix:** central differences, float64, $h \\approx 10^{-5}$, and compare with a relative error. Avoid checking exactly at non-differentiable points such as ReLU at 0.",
          "**Forgetting that PyTorch accumulates gradients.** Calling `backward()` twice without zeroing adds the second gradient to the first. **Fix:** `optimizer.zero_grad()` (or `param.grad = None`) before each backward pass.",
          "**Calling backward on a non-scalar.** `y.backward()` for a vector `y` raises an error because the gradient of a vector is a Jacobian, not a vector. Reduce to a scalar loss first (e.g. `.mean()`), or pass an explicit upstream gradient.",
          "**Confusing partial and total derivatives.** If $y$ depends on $x$, then $\\partial f/\\partial x$ (holding $y$ fixed) differs from $df/dx$ (letting $y$ change too). The multivariable chain rule is exactly the bookkeeping that connects them.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "What is a derivative, intuitively and formally?",
        a: "Intuitively it is the instantaneous rate of change — the slope of the function at a point, or how much the output changes per unit change of the input for very small changes. Formally $f'(x) = \\lim_{h \\to 0} (f(x+h) - f(x))/h$, the limit of secant slopes.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What is the gradient and why do we step in the negative gradient direction during training?",
        a: "The gradient is the vector of partial derivatives of the loss with respect to all parameters. It points in the direction of steepest increase of the loss, with length equal to that steepest slope. Moving a small step in the opposite direction gives the steepest local decrease, so gradient descent updates $\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\nabla L$.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "State the chain rule and apply it to d/dx of e^(3x²).",
        a: "For $L = f(g(x))$, $dL/dx = f'(g(x))\\,g'(x)$ — the outer derivative evaluated at the inner function, times the inner derivative. With $u = 3x^2$: $\\frac{d}{dx}e^{u} = e^{u}\\cdot\\frac{du}{dx} = e^{3x^2}\\cdot 6x$.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "In a computational graph, why does a variable that feeds two nodes receive the sum of two gradients?",
        a: "By the multivariable chain rule, $\\frac{dL}{dx} = \\sum_j \\frac{\\partial L}{\\partial y_j}\\frac{\\partial y_j}{\\partial x}$ over every node $y_j$ that uses $x$. Intuitively, nudging $x$ changes the loss through each path, and to first order those small changes add. This is why frameworks accumulate gradients (`+=`) and why shared weights (RNNs, tied embeddings) and residual connections get gradient contributions from every use.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Derive the derivative of the sigmoid and explain its consequence for deep networks.",
        a: "$\\sigma(z) = 1/(1 + e^{-z})$, so $\\sigma'(z) = e^{-z}/(1 + e^{-z})^2 = \\sigma(z)(1 - \\sigma(z))$. It peaks at $0.25$ for $z = 0$ and decays to 0 as $|z|$ grows. In backpropagation each sigmoid layer multiplies the gradient by at most 0.25 (times the weights), so across many layers the gradient for early layers shrinks geometrically — the vanishing-gradient problem. ReLU, whose derivative is 1 for positive inputs, together with careful initialisation and normalisation, largely avoids this.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why is the central difference more accurate than the forward difference, and how should h be chosen?",
        a: "Taylor expansion shows the forward difference has error $\\tfrac{h}{2}f''(x) + O(h^2)$, while in the central difference the even-order terms cancel, leaving $\\tfrac{h^2}{6}f'''(x) + O(h^4)$. But both subtract nearly equal numbers, so rounding error grows like $\\varepsilon/h$. The total error is minimised at an intermediate $h$: about $\\sqrt{\\varepsilon} \\approx 10^{-8}$ for forward and $\\varepsilon^{1/3} \\approx 10^{-5}$ for central differences in float64.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Why do neural networks use reverse-mode rather than forward-mode automatic differentiation?",
        a: "For $f: \\mathbb{R}^n \\to \\mathbb{R}^m$, forward mode computes one Jacobian-vector product $J\\mathbf{v}$ per pass, so obtaining the full gradient of a scalar loss ($m = 1$) needs $n$ passes — one per parameter. Reverse mode computes one vector-Jacobian product $\\mathbf{v}^\\top J$ per pass, so a single backward pass, seeded with $\\partial L/\\partial L = 1$, yields all $n$ partial derivatives. Training has $m = 1$ and $n$ in the millions or billions, so reverse mode is cheaper by a factor of order $n$. The price is memory: the forward pass's intermediates must be stored for the backward pass.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "ReLU is not differentiable at 0. Why does gradient-based training still work, and what do frameworks do there?",
        a: "The non-differentiable set is a single point, and with continuous-valued inputs and weights pre-activations land exactly on 0 with negligible probability. At that point frameworks return a fixed subgradient — PyTorch and TensorFlow use 0 — which is a valid element of the subdifferential $[0, 1]$. More formally, ReLU networks are piecewise linear and differentiable almost everywhere, and (sub)gradient methods converge under conditions similar to the smooth case. The practical caveat is for gradient checking: finite differences straddling a kink disagree with the analytic gradient, so checks should avoid inputs near 0.",
      },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          {
            kind: "numeric",
            q: "What is the derivative of $f(x) = x^3$ at $x = 2$?",
            answer: 12,
            tolerance: 0.001,
            explain: "Power rule: $f'(x) = 3x^2$, so $f'(2) = 3 \\cdot 4 = 12$.",
          },
          {
            kind: "numeric",
            q: "What is $\\sigma'(0)$ for the sigmoid $\\sigma(z) = 1/(1 + e^{-z})$?",
            answer: 0.25,
            tolerance: 0.001,
            explain: "$\\sigma(0) = 0.5$ and $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$, so $\\sigma'(0) = 0.5 \\times 0.5 = 0.25$ — the largest value the sigmoid's derivative ever takes.",
          },
          {
            kind: "numeric",
            q: "For $f(x, y) = x^2 y$, compute $\\partial f/\\partial x$ at $(3, 2)$.",
            answer: 12,
            tolerance: 0.001,
            explain: "Holding $y$ constant, $\\partial f/\\partial x = 2xy = 2 \\cdot 3 \\cdot 2 = 12$.",
          },
          {
            kind: "numeric",
            q: "For $L(w) = (3w - 2)^2$, compute $dL/dw$ at $w = 1$.",
            answer: 6,
            tolerance: 0.001,
            explain: "Chain rule: $dL/dw = 2(3w - 2) \\cdot 3$. At $w = 1$: $2 \\cdot 1 \\cdot 3 = 6$.",
          },
          {
            q: "What is $\\frac{d}{dx}\\sin(x^2)$?",
            options: ["$\\cos(x^2)$", "$2x\\cos(x^2)$", "$\\cos(2x)$", "$2x\\sin(x^2)$"],
            answer: 1,
            explain: "Outer derivative $\\cos(\\cdot)$ evaluated at $x^2$, times the inner derivative $2x$.",
          },
          {
            q: "At a point where $\\nabla f = (3, 4)$, in which unit direction does $f$ increase fastest, and how fast?",
            options: ["$(1, 0)$, rate 3", "$(0.6, 0.8)$, rate 5", "$(0.6, 0.8)$, rate 7", "$(-0.6, -0.8)$, rate 5"],
            answer: 1,
            explain: "Steepest ascent is along $\\nabla f / \\|\\nabla f\\| = (3, 4)/5 = (0.6, 0.8)$, with slope $\\|\\nabla f\\| = 5$. The direction $(-0.6, -0.8)$ is steepest descent.",
          },
          {
            q: "A variable $x$ is used by two different nodes of a computational graph. How is $\\partial L/\\partial x$ obtained?",
            options: [
              "Take the gradient from whichever node is computed last",
              "Multiply the gradients from the two nodes",
              "Add the gradient contributions from the two nodes",
              "Average the two contributions",
            ],
            answer: 2,
            explain: "Multivariable chain rule: contributions from all paths add.",
          },
          {
            q: "Halving $h$ in a central finite difference (while rounding error is negligible) reduces the approximation error by roughly what factor?",
            options: ["2", "4", "8", "It does not change"],
            answer: 1,
            explain: "The central-difference error is proportional to $h^2$, so halving $h$ divides it by about $2^2 = 4$. For the forward difference (error $\\propto h$) the factor would be 2.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Backpropagate through a ReLU neuron by hand",
        prompt: String.raw`Consider $L = (\mathrm{ReLU}(wx + b) - y)^2$ with $x = 2$, $w = 0.5$, $b = 1$, $y = 1$. Draw the graph (multiply, add, ReLU, subtract, square), run the forward pass, then compute $\partial L/\partial w$, $\partial L/\partial b$ and $\partial L/\partial x$ with the chain rule. Then say what would change if $b = -2$.`,
        hint: "ReLU's local derivative is 1 when its input is positive and 0 when negative.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Forward", tex: String.raw`u = wx = 1, \quad z = u + b = 2, \quad r = \mathrm{ReLU}(2) = 2, \quad d = r - y = 1, \quad L = d^2 = 1` },
              { title: "Square and subtract", tex: String.raw`\frac{\partial L}{\partial d} = 2d = 2, \qquad \frac{\partial L}{\partial r} = 2 \cdot 1 = 2` },
              { title: "ReLU (input 2 > 0, local derivative 1)", tex: String.raw`\frac{\partial L}{\partial z} = 2 \cdot 1 = 2` },
              { title: "Add", tex: String.raw`\frac{\partial L}{\partial u} = \frac{\partial L}{\partial b} = 2` },
              { title: "Multiply", tex: String.raw`\frac{\partial L}{\partial w} = 2 \cdot x = 4, \qquad \frac{\partial L}{\partial x} = 2 \cdot w = 1` },
            ],
          },
          {
            t: "p",
            text: String.raw`With $b = -2$: $z = 1 - 2 = -1 < 0$, so $r = 0$, $d = -1$, $L = 1$ — the same loss — but ReLU's local derivative is now 0, so $\partial L/\partial w = \partial L/\partial b = \partial L/\partial x = 0$. The neuron is "dead" for this input: it contributes a loss but receives no gradient to fix it.`,
          },
        ],
      },
      {
        t: "exercise",
        title: "Derive and check the gradient of logistic loss",
        prompt: String.raw`For data $(\mathbf{x}_i, y_i)$ with labels $y_i \in \{-1, +1\}$, the logistic loss is $L(\mathbf{w}) = \frac{1}{N}\sum_i \ln(1 + e^{-m_i})$ with margin $m_i = y_i\,\mathbf{w}\cdot\mathbf{x}_i$. Use the chain rule to derive $\nabla_{\mathbf{w}} L$, implement it in NumPy (vectorised), and verify it with a central-difference gradient checker that perturbs one coordinate at a time. Report the relative error $\|g_a - g_n\| / (\|g_a\| + \|g_n\|)$.`,
        hint: String.raw`Chain: $\mathbf{w} \to m_i \to \ln(1 + e^{-m_i})$. The outer derivative is $-e^{-m}/(1 + e^{-m}) = -\sigma(-m)$ and $\partial m_i / \partial \mathbf{w} = y_i \mathbf{x}_i$.`,
        solution: [
          {
            t: "p",
            text: String.raw`By the chain rule, $\frac{\partial}{\partial \mathbf{w}}\ln(1 + e^{-m_i}) = \frac{-e^{-m_i}}{1 + e^{-m_i}}\cdot y_i\mathbf{x}_i = -\sigma(-m_i)\,y_i\mathbf{x}_i$. Averaging, $\nabla_{\mathbf{w}}L = \frac{1}{N}X^\top\big(-\sigma(-\mathbf{m}) \odot \mathbf{y}\big)$, where $X$ stacks the $\mathbf{x}_i$ as rows and $\odot$ is element-wise multiplication.`,
          },
          { t: "code", lang: "python", title: "logistic_gradient_check.py", code: EX_CODE, output: EX_OUTPUT },
          {
            t: "p",
            text: "A relative error around $10^{-11}$ confirms the derivation; values above about $10^{-6}$ in float64 usually indicate a bug. `np.log1p(t)` computes $\\ln(1 + t)$ accurately when $t$ is small.",
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`**Derivative:** $f'(x) = \lim_{h\to 0}\frac{f(x+h) - f(x)}{h}$ — local slope; the output changes by about $f'(x)\,\varepsilon$ when the input changes by a small $\varepsilon$.`,
          String.raw`**Key rules:** power $nx^{n-1}$; $(e^x)' = e^x$; $(\ln x)' = 1/x$; product $f'g + fg'$; $\sigma' = \sigma(1-\sigma)$ (max 0.25); $\tanh' = 1 - \tanh^2$; ReLU$'$ = 1 for positive inputs, 0 for negative.`,
          String.raw`**Partial derivative:** vary one input, hold the rest fixed. **Gradient** $\nabla f$ collects them; $D_{\mathbf{u}}f = \nabla f\cdot\mathbf{u}$, so $\nabla f$ is the steepest-ascent direction with slope $\|\nabla f\|$, perpendicular to contours. Descent: $\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\nabla L$.`,
          String.raw`**Chain rule:** $\frac{dL}{dx} = \frac{dL}{du}\frac{du}{dx}$; along a chain, local derivatives multiply; over multiple paths, contributions **add**.`,
          "**Backpropagation** = chain rule on a computational graph: forward pass stores values; backward pass seeds $\\partial L/\\partial L = 1$ and multiplies upstream gradients by local derivatives in reverse order. Add copies, multiply swaps, element-wise functions scale by their derivative.",
          String.raw`**Worked numbers:** for $L = (\sigma(wx+b) - y)^2$ at $x = 1.5, w = 0.8, b = 0.1, y = 1$: $L = 0.045867$, $\partial L/\partial w = -0.108131$, $\partial L/\partial b = -0.072087$.`,
          String.raw`**Finite differences:** central $\frac{f(x+h) - f(x-h)}{2h}$ has $O(h^2)$ error; use float64 and $h \approx 10^{-5}$; a test tool, never a training method.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Jacobians, Hessians and Computational Graphs](/learn/m02/jacobians-hessians) — extends the chain rule to vector-valued layers, where local derivatives become Jacobian matrices and backprop becomes a sequence of vector-Jacobian products.",
          "[Backpropagation](/learn/m04/backpropagation) — applies this lesson's graph procedure to full multi-layer networks with matrix-valued weights, deriving every layer's gradient.",
          "[Gradient Descent](/learn/m06/gradient-descent) — what to do with the gradient once you have it: learning rates, batches, and convergence.",
          "[Numerical Gradient Checking](/learn/m04/gradient-checking) — turns the finite-difference check from this lesson into a systematic debugging tool for whole networks.",
        ],
      },
    ],
  },
};

export default lesson;
