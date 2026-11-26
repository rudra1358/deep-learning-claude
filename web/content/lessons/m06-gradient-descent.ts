import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m06/gradient-descent",
  title: "Gradient Descent and Its Descendants: SGD, Momentum, Adam, AdamW",
  summary:
    "How a network's parameters actually move: the gradient descent update derived from a Taylor expansion, why the learning rate must stay below 2/λ_max, batch vs stochastic vs mini-batch gradients, and the full update equations of Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW and Nadam.",
  minutes: 75,
  level: "intermediate",
  prerequisites: ["m02/derivatives-chain-rule", "m02/vectors-matrices", "m04/backpropagation", "m03/training-basics"],
  next: ["m06/lr-schedules", "m07/initialization", "m07/normalization", "m07/regularization"],
  terms: ["gradient-descent", "learning-rate", "adam", "batch", "epoch", "hyperparameter"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: "Training a network means choosing millions (or billions) of numbers — the weights and biases — so that a **loss function**, which measures the network's error on the training data, becomes small. Backpropagation gives the **gradient** of that loss: for every parameter, how much the loss would change if we nudged it. It does not say how to change the parameters. That is the job of an **optimizer**.",
      },
      {
        t: "p",
        text: "**Gradient descent** is the simplest optimizer and the ancestor of all the others: again and again, take a small step in the direction where the loss falls fastest — the negative gradient. The idea goes back to Cauchy (1847). Its stochastic version (Robbins and Monro, 1951) uses a random subset of the data for each step; it is what trains modern networks, because an exact gradient over millions of examples per step would be far too slow.",
      },
      {
        t: "p",
        text: "Plain gradient descent has two known weaknesses:",
      },
      {
        t: "list",
        items: [
          "On surfaces that are steep in some directions and shallow in others, a learning rate small enough to be stable in the steep direction crawls in the shallow one. **Momentum** methods (Polyak 1964; Nesterov 1983) fix this by building up a velocity.",
          "One learning rate for every parameter ignores that gradients differ hugely in size between parameters. **Adaptive** methods (AdaGrad 2011, RMSProp 2012, Adam 2015) give each parameter its own effective step size.",
        ],
      },
      {
        t: "p",
        text: "**AdamW** (Loshchilov & Hutter, 2019) fixes how Adam interacts with weight decay and is today the default optimizer for training Transformers.",
      },
      {
        t: "callout",
        kind: "key",
        title: "What this lesson covers",
        text: String.raw`The optimization problem; the update rule from a first-order Taylor expansion; an exact analysis of the learning rate on quadratics (convergence iff $0 < \eta < 2/\lambda_{\max}$); batch, stochastic and mini-batch gradients; then Momentum, Nesterov, AdaGrad, RMSProp, Adam (with the bias-correction derivation), AdamW and Nadam, each with full equations and worked numbers; a first look at learning-rate schedules and gradient clipping.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Derivatives and the chain rule](/learn/m02/derivatives-chain-rule) — the derivative as a slope, partial derivatives, and the gradient vector $\nabla L$ (all partial derivatives together). We also use first- and second-order Taylor expansions.`,
          String.raw`[Vectors and matrices](/learn/m02/vectors-matrices) — dot products, norms $\lVert g \rVert$, and (for the convergence analysis) the fact that a symmetric matrix has real eigenvalues and orthogonal eigenvectors.`,
          "[Backpropagation](/learn/m04/backpropagation) — how the loss gradient for every parameter is computed. Here the gradient is just an input that backprop hands us.",
          "[Training basics: epochs, batches and iterations](/learn/m03/training-basics) — the vocabulary of a training loop. We make *batch*, *mini-batch* and *iteration* precise here.",
          "[Regression losses](/learn/m05/regression-losses) — the mean squared error, used for the small regression problem in the code sections.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Walking down a foggy mountain",
        text: "You are on a mountainside in thick fog and want to reach the valley. You cannot see it, but you can feel the slope under your feet. So: feel which way the ground tilts most steeply downhill, step that way, repeat. The **slope under your feet** is the gradient; the **length of your stride** is the learning rate. Too short wastes the day; too long overshoots a narrow valley. Momentum is a heavy rolling ball: it gains speed along a steady downhill direction and ignores small bumps. Adaptive methods shorten the stride where the ground has been steep and lengthen it where it has been gentle.",
      },
      {
        t: "p",
        text: "The gradient only describes the ground **right where you stand** — a local, straight-line picture of the loss. Every gradient method trusts this picture for one step, moves, then looks again. The learning rate is how far you trust it: where the surface curves sharply, the step must be short.",
      },
      {
        t: "p",
        text: "In deep learning, the mountain is the loss as a function of all parameters at once — a landscape with millions of dimensions. You only measure the slope where you are, and usually only **approximately**, from a random handful of examples (a mini-batch). That noisy slope is right on average and thousands of times cheaper, which is why stochastic gradient descent dominates.",
      },
      {
        t: "list",
        items: [
          "**Gradient** — direction of steepest *increase*; we step the opposite way.",
          String.raw`**Learning rate $\eta$** — the stride-length multiplier. The single most important hyperparameter in deep learning.`,
          "**Momentum** — remembers past step directions, so steady directions speed up and back-and-forth directions cancel.",
          "**Adaptive step size** — each parameter divides its gradient by a running measure of how big its gradients have been.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "The training loop around an optimizer" },
      {
        t: "steps",
        items: [
          { title: "Sample a mini-batch", text: "Draw $B$ training examples (usually by walking through a shuffled copy of the dataset)." },
          { title: "Forward pass", text: "Compute the network's predictions and the mean loss over the batch." },
          { title: "Backward pass", text: "Backpropagation computes the gradient $g_t$ of the batch loss for every parameter." },
          { title: "Optional: clip the gradient", text: String.raw`If $\lVert g_t \rVert$ is above a threshold, rescale it (see the end of the Math section).` },
          { title: "Optimizer update", text: String.raw`Turn $g_t$ (plus the optimizer's internal state: velocities, running averages) into a parameter change $\Delta\theta_t$ and apply it.` },
          { title: "Optional: update the learning rate", text: String.raw`A schedule changes $\eta$ during training (warm-up, decay).` },
          { title: "Repeat", text: "One pass over the whole dataset is an **epoch**; one update is an **iteration** (or step)." },
        ],
      },
      { t: "h", text: "Inside one plain gradient-descent update" },
      {
        t: "steps",
        items: [
          { title: "Read the gradient", tex: String.raw`g_t = \nabla_\theta L(\theta_t)` },
          { title: "Scale it by the learning rate", tex: String.raw`\Delta\theta_t = -\eta\, g_t` },
          { title: "Apply the step", tex: String.raw`\theta_{t+1} = \theta_t + \Delta\theta_t` },
        ],
      },
      {
        t: "p",
        text: "That is all of plain gradient descent: nothing is carried between steps. Every other optimizer here replaces step 2 with something that also uses **state** — a quantity stored per parameter and updated every step.",
      },
      { t: "h", text: "What state each optimizer keeps" },
      {
        t: "table",
        head: ["Optimizer", "State per parameter", "What the state remembers", "Extra memory"],
        rows: [
          ["SGD", "none", "—", "0×"],
          ["Momentum / Nesterov", "velocity $v$", "exponentially weighted sum of past gradients", "1×"],
          ["AdaGrad", "accumulator $G$", "sum of all past squared gradients", "1×"],
          ["RMSProp", "average $s$", "exponential moving average of squared gradients", "1× (2× with momentum)"],
          ["Adam / AdamW / Nadam", "$m$ and $v$", "EMA of gradients (1st moment) and of squared gradients (2nd moment), plus the step counter $t$", "2×"],
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "Element-wise operations",
        text: String.raw`In every adaptive method, squares, square roots and divisions such as $g_t^2$ or $m_t / \sqrt{v_t}$ are **element-wise**: each parameter has its own accumulator and is scaled on its own. No matrix inverse is involved, which keeps these methods as cheap per step as SGD.`,
      },
      { t: "h", text: "Batch, stochastic and mini-batch: what changes per step" },
      {
        t: "table",
        head: ["Variant", "Examples per gradient", "Updates per epoch", "Gradient noise", "Hardware use"],
        rows: [
          ["Batch (full) GD", "all $N$", "1", "none (exact gradient)", "one huge computation; memory grows with $N$"],
          ["Stochastic GD (strict)", "1", "$N$", "highest", "poor: one example cannot fill a GPU's parallel units"],
          ["Mini-batch SGD", "$B$ (e.g. 32–4096)", "$N/B$", String.raw`std. dev. $\propto 1/\sqrt{B}$`, "good: a batch is one big matrix multiply"],
        ],
      },
      {
        t: "p",
        text: String.raw`In deep learning, "SGD" nearly always means mini-batch SGD. A step's cost grows linearly with $B$, but the gradient noise only shrinks like $1/\sqrt{B}$: four times the batch halves the noise at four times the cost. Until the GPU is saturated, bigger batches are nearly free in wall-clock time (examples run in parallel); after that, they are not.`,
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. The optimization problem" },
      {
        t: "math",
        tex: String.raw`\theta^\star \in \arg\min_{\theta \in \mathbb{R}^d} L(\theta), \qquad L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell\big(f(x_i;\theta),\, y_i\big)`,
        caption: "Empirical risk minimisation.",
      },
      {
        t: "p",
        text: String.raw`$\theta \in \mathbb{R}^d$ collects all $d$ parameters, $f(x;\theta)$ is the network's output for input $x$, $\ell$ is the per-example loss (e.g. squared error or cross-entropy), $(x_i, y_i)$ are the $N$ training pairs, and $L$ is their average — the **empirical risk**. For neural networks $L$ is non-convex: it can have many local minima, saddle points and flat regions. So we look for a good stationary point, not a guaranteed global minimum.`,
      },

      { t: "h", text: "2. Deriving the update rule from a Taylor expansion" },
      {
        t: "p",
        text: String.raw`Suppose we are at $\theta$ and move by a small vector $\Delta$. The first-order Taylor expansion of $L$ around $\theta$ is:`,
      },
      { t: "math", tex: String.raw`L(\theta + \Delta) = L(\theta) + \nabla L(\theta)^\top \Delta + O(\lVert \Delta \rVert^2)` },
      {
        t: "p",
        text: String.raw`Among all steps of a fixed small length $\lVert \Delta \rVert = \epsilon$, which lowers $L$ most? Only the linear term depends on direction. By the Cauchy–Schwarz inequality $\nabla L^\top \Delta \ge -\lVert \nabla L \rVert\,\lVert \Delta \rVert$, with equality exactly when $\Delta$ points opposite to the gradient. So the **steepest-descent direction** is $-\nabla L / \lVert \nabla L \rVert$. Folding the step length and normalisation into one constant $\eta > 0$ gives gradient descent:`,
      },
      { t: "math", tex: String.raw`\boxed{\;\theta_{t+1} = \theta_t - \eta\, \nabla L(\theta_t)\;}` },
      {
        t: "p",
        text: String.raw`Putting $\Delta = -\eta \nabla L$ back into the expansion predicts the decrease: $L(\theta_{t+1}) \approx L(\theta_t) - \eta \lVert \nabla L(\theta_t) \rVert^2$, negative whenever the gradient is non-zero. The dropped second-order term tells us when this prediction fails. Including it,`,
      },
      {
        t: "math",
        tex: String.raw`L(\theta - \eta g) \approx L(\theta) - \eta \lVert g \rVert^2 + \tfrac{1}{2}\eta^2\, g^\top H g`,
        caption: String.raw`$H = \nabla^2 L(\theta)$ is the Hessian (matrix of second derivatives), $g = \nabla L(\theta)$.`,
      },
      {
        t: "p",
        text: String.raw`The linear term rewards big steps; the curvature term $\tfrac12 \eta^2 g^\top H g$ punishes them quadratically. If the gradient of $L$ is $\beta$-Lipschitz (curvature never above $\beta$, called $\beta$-**smooth**), then $g^\top H g \le \beta \lVert g \rVert^2$, and we get the *descent lemma*: $L(\theta - \eta g) \le L(\theta) - \eta\,(1 - \tfrac{\beta\eta}{2}) \lVert g \rVert^2$. The loss is guaranteed to fall when $0 < \eta < 2/\beta$, and the guaranteed drop is biggest at $\eta = 1/\beta$.`,
      },
      { t: "h", text: "Worked example: three steps on a parabola" },
      {
        t: "p",
        text: String.raw`Take $f(w) = (w-3)^2$, so $f'(w) = 2(w-3)$ and $f''(w) = 2$. Start at $w_0 = 0$ with $\eta = 0.1$.`,
      },
      {
        t: "table",
        head: ["$t$", "$w_t$", "$f(w_t)$", "$f'(w_t)$", String.raw`$\eta f'(w_t)$`, "$w_{t+1}$", "$f(w_{t+1})$"],
        rows: [
          ["0", "0.0000", "9.0000", "−6.0000", "−0.6000", "0.6000", "5.7600"],
          ["1", "0.6000", "5.7600", "−4.8000", "−0.4800", "1.0800", "3.6864"],
          ["2", "1.0800", "3.6864", "−3.8400", "−0.3840", "1.4640", "2.3593"],
        ],
      },
      {
        t: "p",
        text: String.raw`Check the first step against Taylor: the linear model predicts $f(w_1) \approx 9 - \eta (f')^2 = 9 - 0.1 \times 36 = 5.40$. The true value is $5.76$. The gap $0.36$ is exactly the curvature term $\tfrac12 \eta^2 f'' (f')^2 = \tfrac12 (0.01)(2)(36) = 0.36$ — for a quadratic, the second-order expansion is exact. Each step multiplies the distance to the minimum by $1 - \eta f'' = 0.8$: $3 \to 2.4 \to 1.92 \to 1.536$.`,
      },

      { t: "h", text: "3. The learning rate on a quadratic: convergence iff 0 < η < 2/λ_max" },
      {
        t: "p",
        text: String.raw`Near a minimum $\theta^\star$, any smooth loss looks like a quadratic: $L(\theta) \approx L(\theta^\star) + \tfrac12 (\theta - \theta^\star)^\top H (\theta - \theta^\star)$, with $H$ symmetric positive definite. Let the error be $e_t = \theta_t - \theta^\star$. The gradient is $H e_t$, so gradient descent becomes a linear recursion:`,
      },
      { t: "math", tex: String.raw`e_{t+1} = e_t - \eta H e_t = (I - \eta H)\, e_t \quad\Longrightarrow\quad e_t = (I - \eta H)^t e_0` },
      {
        t: "p",
        text: String.raw`Diagonalise $H = Q \Lambda Q^\top$ with eigenvalues $\lambda_1 \le \dots \le \lambda_d$ (the curvatures along the main axes) and orthonormal eigenvectors $q_i$. In that basis, $c_i^{(t)} = q_i^\top e_t$, the coordinates separate completely:`,
      },
      { t: "math", tex: String.raw`c_i^{(t)} = (1 - \eta \lambda_i)^t\, c_i^{(0)}` },
      {
        t: "p",
        text: String.raw`Each direction converges if and only if $|1 - \eta\lambda_i| < 1$, i.e. $0 < \eta < 2/\lambda_i$. All directions converge if and only if`,
      },
      { t: "math", tex: String.raw`\boxed{\;0 < \eta < \frac{2}{\lambda_{\max}}\;}` },
      {
        t: "table",
        head: [String.raw`Range of $\eta\lambda_i$`, String.raw`Factor $1-\eta\lambda_i$`, "Behaviour in that direction"],
        rows: [
          [String.raw`$0 < \eta\lambda_i < 1$`, "in $(0, 1)$", "steady convergence (always approaching from one side)"],
          [String.raw`$\eta\lambda_i = 1$`, "0", "reaches the minimum in one step"],
          [String.raw`$1 < \eta\lambda_i < 2$`, "in $(-1, 0)$", "oscillating convergence: overshoots, sign flips every step, swings shrink"],
          [String.raw`$\eta\lambda_i = 2$`, "−1", "bounces between two points forever"],
          [String.raw`$\eta\lambda_i > 2$`, "below −1", "divergence: each step overshoots more than the last"],
        ],
      },
      {
        t: "p",
        text: String.raw`The slowest direction is the flattest: its factor is $1 - \eta\lambda_{\min}$. Balancing it against the steepest, $|1-\eta\lambda_{\min}| = |1-\eta\lambda_{\max}|$, gives the best fixed learning rate and its convergence rate:`,
      },
      { t: "math", tex: String.raw`\eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}}, \qquad \max_i |1 - \eta^\star \lambda_i| = \frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}}` },
      {
        t: "p",
        text: String.raw`$\kappa$ is the **condition number** of $H$. When $\kappa$ is large — a long, narrow valley — even the best learning rate converges slowly: the step is limited by the steepest direction, progress by the flattest. This one fact motivates momentum and preconditioning.`,
      },
      {
        t: "p",
        text: String.raw`**Numerical check.** $L(\theta) = \tfrac12(\theta_1^2 + 10\theta_2^2)$ has $H = \mathrm{diag}(1, 10)$, so $\lambda_{\max} = 10$ and the stability limit is $\eta < 0.2$. The best rate is $\eta^\star = 2/11 = 0.1818$, shrinking the error by $9/11 = 0.818$ per step. 50 steps from $(1, 1)$:`,
      },
      {
        t: "table",
        head: [String.raw`$\eta$`, String.raw`$\theta_{50}$`, String.raw`$L(\theta_{50})$`, "Comment"],
        rows: [
          ["0.05", "(0.076945, 0.000000)", "2.96e−3", "stable, but the flat direction crawls ($0.95^{50}$)"],
          ["0.1818", "(0.000044, 0.000044)", "1.06e−8", "optimal: both directions shrink by 9/11"],
          ["0.19", "(0.000027, 0.005154)", "1.33e−4", "steep direction oscillates ($1 - 1.9 = -0.9$)"],
          ["0.21", "(0.000008, 117.390853)", "6.89e+4", "diverged: factor $1 - 2.1 = -1.1$ in the steep direction"],
        ],
      },

      { t: "h", text: "4. Stochastic and mini-batch gradients" },
      {
        t: "p",
        text: String.raw`Let $\mathcal{B}_t$ be a mini-batch of $B$ indices drawn uniformly at random. The mini-batch gradient and its properties:`,
      },
      { t: "math", tex: String.raw`g_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla_\theta\, \ell_i(\theta_t), \qquad \mathbb{E}[g_t] = \nabla L(\theta_t), \qquad \operatorname{Cov}[g_t] = \frac{\Sigma(\theta_t)}{B}` },
      {
        t: "p",
        text: String.raw`$\ell_i$ is the loss on example $i$ and $\Sigma$ the covariance of a single-example gradient (the formula assumes sampling with replacement; without replacement multiply by $(N-B)/(N-1)$, which is zero at $B = N$). So the mini-batch gradient is **unbiased** (right on average) but **noisy**, with standard deviation falling like $1/\sqrt{B}$. With a fixed learning rate, SGD never settles: it bounces in a region around the minimum that grows with $\eta$ and the noise. Classical convergence to the minimum (Robbins & Monro, 1951) needs a decaying learning rate with $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$, e.g. $\eta_t \propto 1/t$ — one reason learning-rate schedules exist.`,
      },
      {
        t: "p",
        text: String.raw`The noise is not only a nuisance. It helps SGD escape saddle points and sharp minima, and small batches often generalise at least as well as large ones. When the batch size goes up, the learning rate usually goes up with it: the **linear scaling rule** (Goyal et al., 2017, "Accurate, Large Minibatch SGD") multiplies $\eta$ by the same factor as $B$, together with a warm-up period.`,
      },

      { t: "h", text: "5. Momentum (heavy ball, Polyak 1964)" },
      {
        t: "math",
        tex: String.raw`v_t = \beta\, v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t, \qquad v_0 = 0`,
        caption: String.raw`$v_t$ — velocity; $\beta \in [0, 1)$ — momentum coefficient (typically 0.9). This is PyTorch's convention.`,
      },
      {
        t: "p",
        text: String.raw`Unrolled, $v_t = \sum_{k=0}^{t-1} \beta^k g_{t-k}$: the velocity is an exponentially weighted sum of past gradients. If the gradient is constant, $v_t \to g/(1-\beta)$, so along a steady direction the effective learning rate is $\eta/(1-\beta)$ — **ten times** larger for $\beta = 0.9$. Where the gradient flips sign every step (the steep, oscillating direction of a valley), successive terms cancel and the velocity stays small. So momentum speeds up along the valley floor and damps the zig-zag across it. On a quadratic with well-chosen $\eta$ and $\beta$, the rate improves from about $(\kappa-1)/(\kappa+1)$ to $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$. The equivalent form $v_t = \beta v_{t-1} - \eta g_t$, $\theta_{t+1} = \theta_t + v_t$ only differs in where $\eta$ goes.`,
      },

      { t: "h", text: "6. Nesterov accelerated gradient (NAG)" },
      {
        t: "p",
        text: String.raw`Momentum will move the parameters by about $-\eta\beta v_{t-1}$ whatever the current gradient says. Nesterov's idea (1983, brought to deep learning by Sutskever et al., 2013): evaluate the gradient at that **look-ahead point** instead, so the correction is computed where we are about to be:`,
      },
      { t: "math", tex: String.raw`\tilde\theta_t = \theta_t - \eta\beta\, v_{t-1}, \qquad v_t = \beta v_{t-1} + \nabla L(\tilde\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\, v_t` },
      {
        t: "p",
        text: String.raw`If momentum is about to carry us past the minimum, the look-ahead gradient already points back, so the velocity shrinks one step earlier — less overshoot. Frameworks use an equivalent form that needs only the gradient at the stored parameters. Storing the look-ahead point $\phi_t = \tilde\theta_t$ instead of $\theta_t$ gives:`,
      },
      { t: "math", tex: String.raw`v_t = \beta v_{t-1} + g_t, \qquad \phi_{t+1} = \phi_t - \eta\,(g_t + \beta v_t), \qquad g_t = \nabla L(\phi_t)` },
      {
        t: "p",
        text: "This is exactly `torch.optim.SGD(..., momentum=β, nesterov=True)`. Both forms evaluate gradients at the same points; the stored parameters differ only by the shift η·β·v (checked numerically below).",
      },

      { t: "h", text: "7. AdaGrad (Duchi, Hazan & Singer, 2011)" },
      {
        t: "math",
        tex: String.raw`G_t = G_{t-1} + g_t \odot g_t, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t`,
        caption: String.raw`$\odot$ — element-wise product; $G_t$ — per-parameter sum of all squared gradients so far; $\epsilon$ (e.g. $10^{-8}$) avoids division by zero.`,
      },
      {
        t: "p",
        text: String.raw`Each parameter's step is divided by the root of its summed squared gradients. Parameters with big or frequent gradients get small steps; rarely-updated ones (rare-word embeddings, seldom-active features) keep big steps — excellent for sparse, convex problems. Its weakness: $G_t$ only grows, so the effective learning rate $\eta/\sqrt{G_t}$ decays like $1/\sqrt{t}$ even when it should not, and long non-convex runs stall.`,
      },

      { t: "h", text: "8. RMSProp (Tieleman & Hinton, 2012, Coursera lecture notes)" },
      {
        t: "math",
        tex: String.raw`s_t = \rho\, s_{t-1} + (1-\rho)\, g_t \odot g_t, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \odot g_t`,
        caption: String.raw`$\rho$ — decay rate (0.9 in the original lecture; PyTorch calls it alpha, default 0.99).`,
      },
      {
        t: "p",
        text: String.raw`RMSProp swaps AdaGrad's sum for an **exponential moving average** (EMA), so old gradients fade after about $1/(1-\rho)$ steps and the step size no longer decays to zero. $\sqrt{s_t}$ estimates each parameter's root-mean-square gradient, so $g_t/\sqrt{s_t}$ is a gradient measured in units of its own typical size. Because $s_0 = 0$, the first steps are inflated: at $t = 1$, $s_1 = (1-\rho) g_1^2$, so the step has size $\eta\, g_1 / (\sqrt{1-\rho}\,|g_1|) = \eta/\sqrt{1-\rho}$ — $3.16\eta$ for $\rho = 0.9$. Adam's bias correction removes exactly this artefact.`,
      },

      { t: "h", text: "9. Adam (Kingma & Ba, 2015) and its bias correction" },
      {
        t: "math",
        tex: String.raw`\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t && \text{(1st moment: EMA of gradients)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t \odot g_t && \text{(2nd moment: EMA of squared gradients)}\\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} && \text{(bias correction)}\\
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}`,
        caption: String.raw`Defaults from the paper: $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$; $m_0 = v_0 = 0$; $t$ counts from 1.`,
      },
      {
        t: "p",
        text: "Adam is RMSProp (the denominator) plus momentum (the numerator, written as an average instead of a sum). **Why the bias correction?** Unroll the first-moment recursion from $m_0 = 0$:",
      },
      { t: "math", tex: String.raw`m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{\,t-i}\, g_i` },
      {
        t: "p",
        text: String.raw`Take expectations and assume the gradient distribution is stationary, $\mathbb{E}[g_i] = \mathbb{E}[g]$ (any drift adds a small term that the EMA keeps small by down-weighting old terms). Using the geometric series $\sum_{i=1}^{t}\beta_1^{t-i} = (1-\beta_1^t)/(1-\beta_1)$,`,
      },
      { t: "math", tex: String.raw`\mathbb{E}[m_t] = \mathbb{E}[g]\,(1-\beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i} = \mathbb{E}[g]\,(1 - \beta_1^{\,t})` },
      {
        t: "p",
        text: String.raw`So $m_t$ underestimates the true mean by the factor $1 - \beta_1^t$, only because the average started at zero. Dividing by that factor makes it unbiased: $\mathbb{E}[\hat m_t] = \mathbb{E}[g]$. The same argument with $g \odot g$ gives $\mathbb{E}[v_t] = \mathbb{E}[g^2](1-\beta_2^t)$. The correction matters most for $v$: with $\beta_2 = 0.999$, $1 - \beta_2^{t}$ is $0.001$ at $t = 1$ and still only $0.095$ at $t = 100$. Without it, $\sqrt{v_t}$ would be far too small early on and the first steps far too big. At $t = 1$ the corrected update is $\eta\, g_1/|g_1| = \pm\eta$ per coordinate: Adam's first step has size $\eta$ in every coordinate, whatever the gradient's scale.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "Why Adam is (approximately) scale-invariant",
        text: String.raw`Multiplying the loss by $c > 0$ multiplies $m_t$ by $c$ and $\sqrt{v_t}$ by $c$, so the update $\hat m_t/\sqrt{\hat v_t}$ is unchanged (up to $\epsilon$). Each coordinate moves by at most about $\eta$ per step, so Adam's $\eta$ is a step size in parameter units and Adam copes far better than SGD with badly scaled gradients.`,
      },

      { t: "h", text: "10. AdamW: decoupled weight decay (Loshchilov & Hutter, 2019)" },
      {
        t: "p",
        text: String.raw`**Weight decay** shrinks weights toward zero every step: $\theta \leftarrow (1 - \eta\lambda)\theta$. For plain SGD this is the same as adding an L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss, because the penalty's gradient $\lambda\theta$ just joins $g_t$. For Adam it is **not** the same. If $\lambda\theta$ is added to the gradient ("Adam + L2"), it then gets divided by $\sqrt{\hat v_t}$: weights with big gradients get almost no regularisation, and weights with tiny gradients get very strong regularisation. Loshchilov & Hutter ("Decoupled Weight Decay Regularization", ICLR 2019) apply the decay outside the adaptive machinery:`,
      },
      { t: "math", tex: String.raw`\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_t\right)` },
      {
        t: "p",
        text: String.raw`where $m_t, v_t$ come from the loss gradient only. Now every weight decays at the same relative rate $\eta\lambda$ per step. In the paper the decay is also multiplied by the schedule multiplier; PyTorch's AdamW multiplies it by the current learning rate, as written above (default weight decay 0.01). AdamW generalises better than Adam + L2 and separates the choice of $\lambda$ from $\eta$ more cleanly.`,
      },

      { t: "h", text: "11. Nadam (Dozat, 2016)" },
      {
        t: "p",
        text: String.raw`Nadam puts Nesterov's look-ahead into Adam. In the simplified textbook form (constant $\beta_1$), the numerator's momentum becomes a mix of next step's momentum and the current gradient:`,
      },
      { t: "math", tex: String.raw`\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t} + \epsilon}\left(\beta_1 \hat m_t + \frac{(1-\beta_1)\, g_t}{1 - \beta_1^{\,t}}\right)` },
      {
        t: "p",
        text: "PyTorch's `NAdam` follows Dozat's paper more closely and also warms up the momentum coefficient during training (`momentum_decay=4e-3`), so its numbers differ slightly from this formula in early steps.",
      },

      { t: "h", text: "12. Worked example: two steps of every optimizer" },
      {
        t: "p",
        text: String.raw`Loss $L(\theta) = \tfrac12(\theta_1^2 + 10\,\theta_2^2)$, gradient $g = (\theta_1,\; 10\theta_2)$, start $\theta_0 = (1, 1)$ with $L(\theta_0) = 5.5$ and $g_1 = (1, 10)$. Hyperparameters: SGD/Momentum/Nesterov $\eta = 0.05$, $\beta = 0.9$; AdaGrad $\eta = 0.5$; RMSProp $\eta = 0.05$, $\rho = 0.9$; Adam/AdamW/Nadam $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$; AdamW $\lambda = 0.1$; $\epsilon = 10^{-8}$. All numbers were computed in Python; all but Nadam match PyTorch's built-in optimizers to 6 decimals.`,
      },
      {
        t: "table",
        head: ["Optimizer", "Step 1 intermediates", String.raw`$\theta_1$`, "Step 2 intermediates", String.raw`$\theta_2$`, String.raw`$L(\theta_2)$`],
        rows: [
          ["SGD", String.raw`$\Delta = -0.05\,(1, 10)$`, "(0.95, 0.5)", "$g_2 = (0.95, 5)$", "(0.9025, 0.25)", "0.719753"],
          ["Momentum", "$v_1 = (1, 10)$", "(0.95, 0.5)", "$v_2 = 0.9(1,10) + (0.95,5) = (1.85, 14)$", "(0.8575, −0.2)", "0.567653"],
          ["Nesterov (look-ahead form)", String.raw`look-ahead = $\theta_0$, $v_1 = (1, 10)$`, "(0.95, 0.5)", "look-ahead $(0.905, 0.05)$, its gradient $(0.905, 0.5)$, $v_2 = (1.805, 9.5)$", "(0.85975, 0.025)", "0.372710"],
          ["AdaGrad", String.raw`$G_1 = (1, 100)$, step $0.5\,(1/1, 10/10) = (0.5, 0.5)$`, "(0.5, 0.5)", "$g_2 = (0.5, 5)$, $G_2 = (1.25, 125)$, step $(0.223607, 0.223607)$", "(0.276393, 0.276393)", "0.420163"],
          ["RMSProp", "$s_1 = (0.1, 10)$, step $(0.158114, 0.158114)$", "(0.841886, 0.841886)", "$s_2 = (0.160877, 16.087722)$, step $0.104948$", "(0.736938, 0.736938)", "2.986924"],
          ["Adam", String.raw`$m_1 = (0.1, 1)$, $v_1 = (0.001, 0.1)$, $\hat m_1 = (1, 10)$, $\hat v_1 = (1, 100)$, step $(0.1, 0.1)$`, "(0.9, 0.9)", String.raw`$\hat m_2 = (0.947368, 9.473684)$, $\hat v_2 = (0.904952, 90.495248)$, step $0.099588$`, "(0.800412, 0.800412)", "3.523629"],
          ["AdamW", String.raw`Adam step $0.1$, then decay $\eta\lambda\theta_0 = 0.01$`, "(0.89, 0.89)", "Adam step $0.099528$, decay $0.0089$", "(0.781572, 0.781572)", "3.359700"],
          ["Nadam", "numerator $0.9(1,10) + 0.1(1,10)/0.1 = (1.9, 19)$, step $0.19$", "(0.81, 0.81)", "step $0.13587$", "(0.67413, 0.67413)", "2.499482"],
        ],
      },
      {
        t: "p",
        text: String.raw`Three things to read off the table. (i) Momentum's second step overshoots $\theta_2$ to $-0.2$ because the velocity $14$ carries the big first gradient forward; Nesterov's look-ahead gradient is already small ($0.5$) and stops at $0.025$. (ii) The adaptive methods move both coordinates by **exactly the same amount**, though one gradient is ten times bigger: dividing by the root of the squared gradient turns each coordinate's update into about $\pm\eta$. (iii) RMSProp's first step is $0.05/\sqrt{0.1} = 0.158$, i.e. $\eta/\sqrt{1-\rho}$ — the start-up inflation that Adam's bias correction removes (Adam's first step is exactly $\eta = 0.1$).`,
      },
      {
        t: "details",
        summary: "Nesterov: why PyTorch's iterates look different",
        blocks: [
          {
            t: "p",
            text: String.raw`PyTorch's SGD with lr = 0.05, momentum = 0.9 and nesterov enabled gives $\phi_1 = (0.905, 0.05)$ and $\phi_2 = (0.778525, -0.4025)$ on this problem. These are not wrong: $\phi_1 = \theta_1 - \eta\beta v_1 = (0.95, 0.5) - 0.045\,(1, 10) = (0.905, 0.05)$, exactly the look-ahead point used in step 2 of the table. PyTorch stores the look-ahead point instead of $\theta$.`,
          },
        ],
      },

      { t: "h", text: "13. Learning-rate schedules and gradient clipping (preview)" },
      {
        t: "p",
        text: String.raw`A **learning-rate schedule** makes $\eta$ depend on the step, $\eta_t$. Common choices: step decay (divide by 10 at fixed epochs), exponential decay $\eta_t = \eta_0\gamma^t$, cosine annealing $\eta_t = \eta_{\min} + \tfrac12(\eta_0 - \eta_{\min})(1 + \cos(\pi t/T))$, and a linear **warm-up** over the first few hundred or thousand steps, standard for Transformers trained with Adam. Big early steps explore; small late steps settle in and lower the noise floor. Details in [Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules).`,
      },
      {
        t: "p",
        text: "**Gradient clipping by norm** guards against rare huge gradients (common in recurrent networks and early Transformer training) by rescaling the whole gradient vector when its norm exceeds a threshold $c$:",
      },
      { t: "math", tex: String.raw`g \leftarrow g \cdot \min\!\left(1,\; \frac{c}{\lVert g \rVert_2}\right)` },
      {
        t: "p",
        text: "The direction stays the same; only the length is capped. Clipping by **value** (clamping each component to $[-c, c]$) is simpler but changes the direction.",
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      {
        t: "p",
        text: "The first simulator runs plain gradient descent on a 1-D function, so you see each update as a tangent line and a jump along the x-axis. The second races six optimizers on 2-D loss surfaces.",
      },
      { t: "sim", id: "gradient-descent", caption: "Gradient descent in 1-D: the gold line is the tangent (slope f′(x)); the clay arrow is the step −η f′(x)." },
      {
        t: "list",
        items: [
          String.raw`**Quadratic, η = 0.2:** each step multiplies x by $1 - 2\eta = 0.6$ — the history table shows $x_{k+1}/x_k = 0.6$ exactly.`,
          String.raw`**Quadratic, η between 0.5 and 1:** the factor $1-2\eta$ is negative — x flips sign each step but still converges. At **η = 1** it bounces between ±2.5 forever; at **η = 1.05** it diverges (the stability check turns clay).`,
          "**Two minima:** start at $x_0 = 1.9$ with η = 0.03 and you settle in the *local* minimum at 1.13. Start at $x_0 = -0.5$ and you reach the global one at −1.30. Gradient descent only knows the local slope.",
          "**Plateau + cliff:** start at −2.6 — the gradient is about 0.006, so even η = 0.3 barely moves. Now click near x = 1.5 with η = 0.3: the cliff's gradient (about 9.3) throws you 2.8 units left, far past the minimum. Huge and tiny gradients in one landscape are what adaptive optimizers are built for.",
          "**Flat bottom (x⁴):** η = 0.1 converges, but slowly at the end because $f'(x) = x^3$ vanishes faster than $x$. Raise η to about 0.55 from $x_0 = 2$: the local stability limit $2/f''(x) = 2/(3x^2) = 0.167$ at the start is broken and the run diverges.",
        ],
      },
      { t: "sim", id: "optimizers", caption: "Optimizer race. Click the map to choose a start point; toggle optimizers in the legend." },
      {
        t: "list",
        items: [
          "**Elongated valley (curvatures 0.2 and 4):** SGD with η = 0.2 flattens the steep y-direction quickly, then crawls along x. Momentum and Nesterov travel the valley much faster. Raise the SGD-family η toward 0.5: SGD starts to zig-zag in y; above $2/4 = 0.5$ it diverges.",
          "**Same surface, adaptive methods:** Adam heads almost diagonally at first — its normalised steps are about η in **each** coordinate, whatever the curvature. RMSProp ends up hovering around the minimum with steps of about η, since it has no momentum averaging to cancel the sign flips. AdaGrad's step keeps shrinking.",
          String.raw`**Rosenbrock:** everyone reaches the curved valley quickly; travelling along it is the hard part. Nesterov usually finishes first in the SGD family. Try η = 0.002 for the SGD family: momentum methods diverge, because their effective step $\eta/(1-\beta)$ is ten times bigger.`,
          "**Saddle:** starting at $y = 0.01$ (almost on the ridge), SGD lingers near the saddle point until the tiny $y$-component grows. Adam escapes almost at once — its normalisation inflates the small $y$-gradient to a step of size η.",
          "**Multi-modal:** most optimizers fall into the nearest well, but Momentum can roll into another. Try different start points.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "Every optimizer from the Math section in NumPy, each a small class with a `step(params, grads)` method, tested on the Rosenbrock function and on mini-batch linear regression with badly scaled features.",
      },
      {
        t: "code",
        lang: "python",
        title: "optimizers_from_scratch.py",
        code: String.raw`import numpy as np

# ---------------------------------------------------------------
# Optimizers from scratch. Each one owns its state (velocity,
# moment estimates...) and exposes step(params, grads) -> new params.
# ---------------------------------------------------------------
class SGD:
    def __init__(self, lr): self.lr = lr
    def step(self, p, g): return p - self.lr * g

class Momentum:                       # heavy ball (Polyak 1964), PyTorch convention
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, None
    def step(self, p, g):
        self.v = g if self.v is None else self.beta * self.v + g
        return p - self.lr * self.v

class Nesterov:                       # look-ahead gradient, in the PyTorch re-parameterisation
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, None
    def step(self, p, g):
        self.v = g if self.v is None else self.beta * self.v + g
        return p - self.lr * (g + self.beta * self.v)

class AdaGrad:
    def __init__(self, lr, eps=1e-8): self.lr, self.eps, self.G = lr, eps, 0.0
    def step(self, p, g):
        self.G = self.G + g * g        # running SUM of squared gradients
        return p - self.lr * g / (np.sqrt(self.G) + self.eps)

class RMSProp:
    def __init__(self, lr, rho=0.9, eps=1e-8): self.lr, self.rho, self.eps, self.s = lr, rho, eps, 0.0
    def step(self, p, g):
        self.s = self.rho * self.s + (1 - self.rho) * g * g   # running AVERAGE
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)

class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, nesterov=False):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.nesterov = weight_decay, nesterov      # wd > 0 -> AdamW, nesterov -> Nadam
        self.m = self.v = 0.0; self.t = 0
    def step(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)              # bias correction
        v_hat = self.v / (1 - self.b2 ** self.t)
        if self.nesterov:                                       # Nadam numerator
            m_hat = self.b1 * m_hat + (1 - self.b1) * g / (1 - self.b1 ** self.t)
        p = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return p - self.lr * self.wd * p if self.wd else p     # decoupled weight decay

# ---------------------------------------------------------------
# Test 1: Rosenbrock f(x,y) = (1-x)^2 + 100 (y-x^2)^2, minimum at (1,1)
# ---------------------------------------------------------------
def rosen(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2

def rosen_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])

def make(name):
    # SGD, Momentum, Nesterov share lr=1e-3 so the effect of momentum is visible;
    # adaptive methods take steps of size ~lr per coordinate, so lr means something different
    return {"SGD": SGD(1e-3), "Momentum": Momentum(1e-3), "Nesterov": Nesterov(1e-3),
            "AdaGrad": AdaGrad(0.5), "RMSProp": RMSProp(1e-3), "Adam": Adam(2e-2),
            "AdamW": Adam(2e-2, weight_decay=1e-3), "Nadam": Adam(2e-2, nesterov=True)}[name]

print("Rosenbrock, start (-1.5, 2.0), 5000 steps")
for name in ["SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam", "AdamW", "Nadam"]:
    opt, p = make(name), np.array([-1.5, 2.0])
    for _ in range(5000):
        p = opt.step(p, rosen_grad(p))
    print(f"  {name:9s} final=({p[0]:+.4f}, {p[1]:+.4f})  f={rosen(p):.2e}")

# ---------------------------------------------------------------
# Test 2: linear regression y = X w + b + noise, mini-batch training
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
n, d = 512, 5
X = rng.normal(size=(n, d)) * np.array([1, 1, 1, 5, 0.2])  # unequal feature scales
w_true = np.array([2.0, -1.0, 0.5, 0.3, 3.0]); b_true = 1.0
y = X @ w_true + b_true + 0.1 * rng.normal(size=n)

def mse_and_grad(theta, Xb, yb):
    w, b = theta[:-1], theta[-1]
    r = Xb @ w + b - yb                         # residuals
    loss = np.mean(r ** 2)
    g = np.concatenate([2 * Xb.T @ r / len(yb), [2 * r.mean()]])
    return loss, g

lrs = {"SGD": 0.01, "Momentum": 0.002, "Nesterov": 0.002, "AdaGrad": 0.5,
       "RMSProp": 0.01, "Adam": 0.05, "AdamW": 0.05, "Nadam": 0.05}
batch, epochs = 32, 20
print(f"\nLinear regression, n={n}, batch={batch}, {epochs} epochs  (full-batch MSE)")
for name, lr in lrs.items():
    opt = {"SGD": SGD, "Momentum": Momentum, "Nesterov": Nesterov, "AdaGrad": AdaGrad,
           "RMSProp": RMSProp, "Adam": Adam}.get(name, Adam)(lr)
    if name == "AdamW": opt.wd = 1e-3
    if name == "Nadam": opt.nesterov = True
    theta, order_rng = np.zeros(d + 1), np.random.default_rng(1)
    for ep in range(epochs):
        idx = order_rng.permutation(n)             # reshuffle every epoch
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            _, g = mse_and_grad(theta, X[bi], y[bi])
            theta = opt.step(theta, g)
    full, _ = mse_and_grad(theta, X, y)
    print(f"  {name:9s} MSE={full:.5f}  w={np.round(theta[:-1], 3)}  b={theta[-1]:.3f}")`,
        output: String.raw`Rosenbrock, start (-1.5, 2.0), 5000 steps
  SGD       final=(+0.9194, +0.8449)  f=6.51e-03
  Momentum  final=(+1.0000, +1.0000)  f=1.99e-19
  Nesterov  final=(+1.0000, +1.0000)  f=9.32e-20
  AdaGrad   final=(+0.9484, +0.8992)  f=2.67e-03
  RMSProp   final=(+0.9757, +0.9534)  f=7.85e-04
  Adam      final=(+1.0000, +1.0000)  f=3.09e-29
  AdamW     final=(+0.9994, +0.9987)  f=4.06e-07
  Nadam     final=(+0.9992, +0.9984)  f=6.01e-07

Linear regression, n=512, batch=32, 20 epochs  (full-batch MSE)
  SGD       MSE=0.23595  w=[ 1.971 -0.961  0.5    0.303  0.674]  b=1.018
  Momentum  MSE=0.14698  w=[ 1.981 -0.963  0.502  0.296  1.202]  b=1.016
  Nesterov  MSE=0.14658  w=[ 1.981 -0.964  0.502  0.298  1.202]  b=1.016
  AdaGrad   MSE=0.01156  w=[ 1.998 -0.99   0.498  0.308  3.039]  b=0.995
  RMSProp   MSE=0.01641  w=[ 1.996 -1.002  0.496  0.31   2.731]  b=0.994
  Adam      MSE=0.01028  w=[ 2.002 -0.997  0.506  0.296  3.051]  b=1.001
  AdamW     MSE=0.01029  w=[ 2.001 -0.997  0.506  0.296  3.047]  b=1.001
  Nadam     MSE=0.00977  w=[ 2.001 -0.995  0.504  0.3    3.051]  b=1.001`,
      },
      {
        t: "p",
        text: String.raw`Two lessons from the output. On Rosenbrock with the same $\eta = 10^{-3}$, plain SGD is still crawling along the valley after 5000 steps while Momentum and Nesterov converge to machine precision — the $1/(1-\beta)$ speed-up at work. In the regression, feature 5 has scale 0.2 and feature 4 scale 5, so their curvatures differ by $(5/0.2)^2 = 625$. SGD's learning rate is limited by the steep feature, so after 20 epochs it has learned only $w_5 \approx 0.67$ of the true 3.0; every adaptive method, normalising per parameter, recovers $w_5 \approx 3$.`,
      },
      { t: "h", text: "Batch size: noise versus number of updates" },
      {
        t: "code",
        lang: "python",
        title: "batch_vs_minibatch.py",
        code: String.raw`import numpy as np

rng = np.random.default_rng(0)
n, d = 1024, 3
X = rng.normal(size=(n, d))
w_true = np.array([1.5, -2.0, 0.7])
y = X @ w_true + 0.5 * rng.normal(size=n)

def grad(w, Xb, yb):                        # gradient of mean squared error
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

def full_loss(w):
    return np.mean((X @ w - y) ** 2)

# 1) How noisy is a mini-batch gradient? Compare to the full gradient at w = 0.
w0 = np.zeros(d)
g_full = grad(w0, X, y)
for B in [1, 8, 64, 512]:
    errs = []
    for _ in range(2000):
        idx = rng.choice(n, size=B, replace=False)
        errs.append(np.linalg.norm(grad(w0, X[idx], y[idx]) - g_full))
    print(f"B={B:4d}  mean ||g_B - g_full|| = {np.mean(errs):.3f}")

# 2) Same budget of 5 epochs (= 5*n example-gradients) for each batch size.
print()
for B, lr in [(n, 0.1), (64, 0.1), (8, 0.05), (1, 0.01)]:
    w, order = np.zeros(d), np.random.default_rng(1)
    updates = 0
    for ep in range(5):
        idx = order.permutation(n)
        for s in range(0, n, B):
            bi = idx[s:s + B]
            w -= lr * grad(w, X[bi], y[bi])
            updates += 1
    print(f"B={B:4d} lr={lr:<5} updates={updates:5d}  final loss={full_loss(w):.4f}")
print(f"noise floor (loss at w_true) = {full_loss(w_true):.4f}")`,
        output: String.raw`B=   1  mean ||g_B - g_full|| = 8.476
B=   8  mean ||g_B - g_full|| = 3.280
B=  64  mean ||g_B - g_full|| = 1.131
B= 512  mean ||g_B - g_full|| = 0.298

B=1024 lr=0.1   updates=    5  final loss=0.9911
B=  64 lr=0.1   updates=   80  final loss=0.2498
B=   8 lr=0.05  updates=  640  final loss=0.2513
B=   1 lr=0.01  updates= 5120  final loss=0.2529
noise floor (loss at w_true) = 0.2503`,
      },
      {
        t: "p",
        text: String.raw`The gradient error falls roughly like $1/\sqrt{B}$ (8.48 → 3.28 → 1.13: each 8× increase in $B$ divides it by about 2.6–2.9, close to $\sqrt 8 = 2.83$; at $B = 512$, sampling without replacement from 1024 examples shrinks it further). With the same budget of 5 epochs, full-batch gradient descent makes only 5 updates and is still far off, while mini-batches make 80–5120 updates and all reach the noise floor. Per unit of compute, many noisy steps beat a few exact ones.`,
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
                text: "Every optimizer lives in `torch.optim` and uses the same three calls: `zero_grad()`, `loss.backward()`, `step()`. Gradient clipping and schedulers slot in between.",
              },
              {
                t: "code",
                lang: "python",
                title: "torch_optimizers.py",
                code: String.raw`import torch
torch.manual_seed(0)

# Same kind of data as the NumPy version: 5 features with very different scales
n, d = 512, 5
X = torch.randn(n, d) * torch.tensor([1, 1, 1, 5, 0.2])
w_true = torch.tensor([2.0, -1.0, 0.5, 0.3, 3.0])
y = X @ w_true + 1.0 + 0.1 * torch.randn(n)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True,
                                     generator=torch.Generator().manual_seed(1))

def train(make_opt, epochs=20, clip=None, schedule=False):
    torch.manual_seed(0)
    model = torch.nn.Linear(d, 1)
    opt = make_opt(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader)) if schedule else None
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()                       # 1. clear old gradients
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()                       # 2. compute new gradients
            if clip is not None:                  # 3. optional: rescale if ||g|| > clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()                            # 4. apply the update rule
            if sched: sched.step()                # 5. optional: update the learning rate
    with torch.no_grad():
        return loss_fn(model(X).squeeze(1), y).item(), model.weight.squeeze().numpy().round(3)

optims = {
    "SGD":          lambda p: torch.optim.SGD(p, lr=0.01),
    "SGD+momentum": lambda p: torch.optim.SGD(p, lr=0.002, momentum=0.9),
    "Nesterov":     lambda p: torch.optim.SGD(p, lr=0.002, momentum=0.9, nesterov=True),
    "Adagrad":      lambda p: torch.optim.Adagrad(p, lr=0.5),
    "RMSprop":      lambda p: torch.optim.RMSprop(p, lr=0.01, alpha=0.9),
    "Adam":         lambda p: torch.optim.Adam(p, lr=0.05),
    "AdamW":        lambda p: torch.optim.AdamW(p, lr=0.05, weight_decay=1e-3),
    "NAdam":        lambda p: torch.optim.NAdam(p, lr=0.05),
}
for name, mk in optims.items():
    mse, w = train(mk)
    print(f"{name:13s} MSE={mse:.5f}  w={w}")

mse, w = train(optims["Adam"], clip=1.0, schedule=True)
print(f"{'Adam+clip+cos':13s} MSE={mse:.5f}  w={w}")`,
                output: String.raw`SGD           MSE=0.26263  w=[ 1.95  -1.008  0.511  0.294  0.569]
SGD+momentum  MSE=0.15905  w=[ 1.968 -1.014  0.518  0.286  1.152]
Nesterov      MSE=0.16068  w=[ 1.961 -1.008  0.509  0.291  1.129]
Adagrad       MSE=0.00956  w=[ 1.992 -1.008  0.498  0.302  3.007]
RMSprop       MSE=0.01821  w=[ 1.98  -1.018  0.519  0.301  2.579]
Adam          MSE=0.00954  w=[ 1.989 -1.014  0.499  0.298  3.007]
AdamW         MSE=0.00948  w=[ 1.99  -1.006  0.497  0.301  3.009]
NAdam         MSE=0.00944  w=[ 1.993 -1.006  0.5    0.298  3.008]
Adam+clip+cos MSE=0.00938  w=[ 1.992 -1.007  0.501  0.3    3.006]`,
              },
              {
                t: "list",
                items: [
                  "`torch.optim.SGD` covers plain SGD, momentum (`momentum=0.9`) and Nesterov (`nesterov=True`). Its `weight_decay` argument adds λθ to the gradient (L2), which for SGD equals weight decay.",
                  "`torch.optim.Adam(weight_decay=...)` implements **Adam + L2** (coupled). Use `torch.optim.AdamW` for decoupled weight decay.",
                  "`RMSprop`'s decay rate is called `alpha` (default 0.99). Adagrad, RMSprop, Adam and NAdam default to `eps=1e-10`, `1e-8`, `1e-8` and `1e-8`.",
                  "`clip_grad_norm_` computes one norm over all parameters together (global norm) and rescales in place; call it after `backward()` and before `step()`.",
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
                title: "keras_optimizers.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
X = (rng.normal(size=(512, 5)) * np.array([1, 1, 1, 5, 0.2])).astype("float32")
w_true = np.array([2.0, -1.0, 0.5, 0.3, 3.0], dtype="float32")
y = (X @ w_true + 1.0 + 0.1 * rng.normal(size=512)).astype("float32")

optimizers = {
    "SGD":      lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": lambda: tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),
    "Nesterov": lambda: tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.9, nesterov=True),
    "Adagrad":  lambda: tf.keras.optimizers.Adagrad(learning_rate=0.5),
    "RMSprop":  lambda: tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9),
    "Adam":     lambda: tf.keras.optimizers.Adam(learning_rate=0.05),
    "AdamW":    lambda: tf.keras.optimizers.AdamW(learning_rate=0.05, weight_decay=1e-3),
    "Nadam":    lambda: tf.keras.optimizers.Nadam(learning_rate=0.05),
}

for name, make_opt in optimizers.items():
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential([tf.keras.Input(shape=(5,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer=make_opt(), loss="mse")
    model.fit(X, y, batch_size=32, epochs=20, shuffle=True, verbose=0)
    print(f"{name:9s} MSE={model.evaluate(X, y, verbose=0):.5f}")

# Cosine-decayed learning rate + clipping of the GLOBAL gradient norm (like clip_grad_norm_)
steps_per_epoch = 512 // 32
schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.05,
                                                     decay_steps=20 * steps_per_epoch)
opt = tf.keras.optimizers.Adam(learning_rate=schedule, global_clipnorm=1.0)
tf.keras.utils.set_random_seed(0)
model = tf.keras.Sequential([tf.keras.Input(shape=(5,)), tf.keras.layers.Dense(1)])
model.compile(optimizer=opt, loss="mse")
model.fit(X, y, batch_size=32, epochs=20, verbose=0)
print("Adam+clip+cosine MSE:", model.evaluate(X, y, verbose=0))`,
                output: "Prints one line per optimizer with its full-dataset MSE after 20 epochs, then the MSE of Adam with a cosine schedule and global-norm clipping. Expect the same qualitative pattern as the PyTorch run: SGD and the momentum variants leave the small-scale feature under-fitted (MSE around 0.1–0.3), the adaptive optimizers reach roughly the noise level of 0.01.",
              },
              {
                t: "list",
                items: [
                  "Keras defaults differ from PyTorch: `epsilon=1e-7` for Adam/RMSprop, and Adagrad starts its accumulator at `initial_accumulator_value=0.1`, not 0.",
                  "`clipnorm` clips each variable's gradient separately; `global_clipnorm` clips the norm of all gradients together, matching PyTorch's `clip_grad_norm_`.",
                  "`tf.keras.optimizers.AdamW` (TensorFlow ≥ 2.11) applies decoupled weight decay; `Adam` has no L2 term — add regularisers to layers for that.",
                ],
              },
            ],
          },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    variations: [
      {
        t: "table",
        head: ["Method", "Update idea", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Batch GD", "exact gradient over all data", "deterministic, simple analysis", "one update per epoch; infeasible for large $N$", "small convex problems, analysis"],
          ["Mini-batch SGD", "noisy gradient from $B$ examples", "cheap steps, uses GPU parallelism, noise helps escape saddles", String.raw`needs tuned $\eta$ and schedule; slow on ill-conditioned problems`, "the default way to use any optimizer"],
          ["Momentum", "EMA-like sum of gradients (velocity)", "speeds up steady directions, damps oscillation", "one more hyperparameter; can overshoot", "CNNs (with SGD), part of every modern optimizer"],
          ["Nesterov", "gradient at the look-ahead point", "less overshoot than momentum, better theory on convex problems", "small practical gains over momentum", "CNN training with SGD"],
          ["AdaGrad", String.raw`divide by $\sqrt{\sum g^2}$`, "great for sparse features; little tuning", "step size decays to zero in long runs", "sparse linear models, embeddings, online learning"],
          ["RMSProp", String.raw`divide by $\sqrt{\text{EMA}(g^2)}$`, "no decay to zero; handles changing problems", "no bias correction; inflated first steps", "RNNs, reinforcement learning (historically)"],
          ["Adam", "momentum ÷ RMS, bias-corrected", "robust default, little tuning, scale-invariant", "2× memory; L2 regularisation behaves badly; sometimes generalises worse than SGD on vision", "most deep learning, especially NLP"],
          ["AdamW", "Adam + decoupled weight decay", "correct weight decay; better generalisation than Adam + L2", String.raw`2× memory; $\lambda$ must be tuned`, "Transformers, LLMs, ViTs — current default"],
          ["Nadam", "Adam with Nesterov look-ahead", "sometimes slightly faster early progress", "rarely a decisive gain", "drop-in Adam alternative"],
        ],
      },
      {
        t: "p",
        text: "Beyond these: layer-wise adaptive methods such as LARS and LAMB rescale each layer's update by the ratio of weight norm to update norm, which helps with very large batches; second-order-inspired methods (K-FAC, Shampoo) approximate curvature with structured matrices instead of a diagonal.",
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          "**Image classification CNNs.** The ResNet paper (He et al., 2016) trained on ImageNet with SGD, momentum 0.9, weight decay $10^{-4}$, batch size 256 and a learning rate of 0.1 divided by 10 when the error plateaued. SGD with momentum and a step or cosine schedule is still a strong choice for convolutional networks.",
          String.raw`**Transformers and large language models.** The original Transformer (Vaswani et al., 2017) used Adam with $\beta_2 = 0.98$ and warm-up followed by inverse-square-root decay. Modern large language models are almost all trained with AdamW, warm-up, cosine or linear decay, and gradient-norm clipping (usually at 1.0).`,
          "**Large-batch distributed training.** Goyal et al. (2017) trained ResNet-50 on ImageNet with batch size 8192 by scaling the learning rate linearly with the batch size plus a gradual warm-up — the linear scaling rule.",
          "**Recommender systems and NLP with sparse features.** AdaGrad-style per-parameter rates suit embedding tables where most rows rarely get gradients.",
          "**Fine-tuning pretrained models.** AdamW with a small learning rate (often $10^{-5}$ to $10^{-4}$) and warm-up is the usual recipe for adapting a pretrained Transformer to a new task.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "table",
        head: ["Aspect", "SGD + momentum", "Adam / AdamW"],
        rows: [
          ["Tuning effort", "learning rate and schedule are critical; sensitive to gradient scale", String.raw`the defaults ($\eta = 10^{-3}$, $\beta = (0.9, 0.999)$) often work out of the box`],
          ["Ill-conditioning", "slow when curvatures differ greatly", "per-parameter normalisation makes up for axis-aligned scale differences"],
          ["Memory", "1 extra buffer per parameter", "2 extra buffers (significant for billion-parameter models)"],
          ["Generalisation", "often slightly better on image classification", "excellent for Transformers; AdamW closes much of the gap elsewhere"],
          ["Robustness to noisy / sparse gradients", "weaker", "strong"],
        ],
      },
      {
        t: "list",
        items: [
          String.raw`**Strength of all first-order methods:** cost per step is linear in the number of parameters — essential at deep-learning scale, where storing the $d \times d$ Hessian is impossible ($d = 10^9$ would need $10^{18}$ entries).`,
          "**Weakness:** they only use local slope. Adaptive methods use a **diagonal** curvature stand-in, so they cannot fix ill-conditioning along directions not aligned with the parameter axes (e.g. a rotated narrow valley).",
          "**Noise is a trade-off:** small batches give noisier, more frequent updates and some regularisation; large batches use hardware better but need learning-rate scaling and warm-up, with diminishing returns past a critical batch size.",
          "**The learning rate stays the dominant hyperparameter** for every optimizer. A good schedule often matters more than choosing Adam vs SGD.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting `optimizer.zero_grad()`.** PyTorch *adds up* gradients across `backward()` calls. Without zeroing, step $t$ uses the sum of all gradients so far — like an ever-growing learning rate. Fix: call `zero_grad()` once per iteration (or accumulate on purpose and divide by the number of micro-batches).",
          String.raw`**Re-using an SGD learning rate for Adam (or the reverse).** SGD's $\eta$ multiplies the raw gradient; Adam's $\eta$ is about the per-coordinate step size. $\eta = 0.1$ is fine for SGD on a ResNet and disastrous for Adam. Fix: start Adam near $10^{-3}$ (fine-tuning: $10^{-5}$–$10^{-4}$) and tune SGD separately.`,
          "**Thinking `Adam(weight_decay=λ)` is weight decay.** In PyTorch it is L2 added to the gradient, then normalised by the root of the second-moment estimate — weights with big gradients are barely regularised. Fix: use `AdamW`.",
          "**Applying weight decay to biases and normalisation parameters.** Shrinking LayerNorm/BatchNorm gains and biases toward zero rarely helps and can hurt. Fix: two parameter groups, one with `weight_decay=0` for biases and norm parameters.",
          String.raw`**Forgetting that momentum multiplies the effective step by $1/(1-\beta)$.** Turning on momentum 0.9 without lowering $\eta$ makes steps ten times bigger and can cause divergence (as in the simulator's Rosenbrock run with $\eta = 0.002$).`,
          String.raw`**Loss turns NaN after a few steps.** Usually $\eta$ is above the $2/\lambda_{\max}$ stability limit for some layer, or a rare batch caused an exploding gradient. Fix: lower $\eta$, add warm-up, clip the gradient norm, and check the data for extreme values.`,
          "**Clipping after `optimizer.step()`.** Clipping must happen between `backward()` and `step()`; afterwards it changes nothing that matters for the current update.",
          "**Not shuffling each epoch.** With a fixed order, mini-batches repeat across epochs and the gradient is no longer an unbiased sample; sorted data (e.g. by class) can make training swing badly. Fix: `shuffle=True` / a new permutation every epoch.",
          "**Judging progress from single mini-batch losses.** They are noisy by design. Fix: track a moving average or evaluate on a fixed validation set.",
          "**Calling `scheduler.step()` per epoch when the schedule was defined per iteration (or the reverse).** The learning rate then decays 100× too slowly or too fast. Fix: match `T_max`/`step_size` to how often you call it.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is gradient descent and why do we move against the gradient?", a: String.raw`Gradient descent repeatedly updates parameters by $\theta \leftarrow \theta - \eta \nabla L(\theta)$. The gradient points where the loss rises fastest. By the first-order Taylor expansion, a small step $\Delta$ changes the loss by about $\nabla L^\top \Delta$, which (for a fixed step length) is most negative when $\Delta$ points opposite to the gradient. So moving against the gradient gives the biggest local decrease.` },
      { t: "qa", level: "beginner", q: "What happens if the learning rate is too small or too large?", a: String.raw`Too small: every step is safe but tiny, so training takes far too long and may stall on plateaus. Too large: steps overshoot; the loss oscillates or diverges (for a quadratic, divergence happens exactly when $\eta > 2/\lambda_{\max}$, the largest curvature). In between, a rate that slightly overshoots in steep directions while moving well in flat ones is usually best.` },
      { t: "qa", level: "beginner", q: "What is the difference between batch, stochastic and mini-batch gradient descent?", a: String.raw`How many examples estimate the gradient for one update. Batch GD uses the whole dataset (exact gradient, one update per epoch). Stochastic GD uses one example (very noisy, $N$ updates per epoch, poor hardware use). Mini-batch uses $B$ examples, typically 32–4096: noise falls like $1/\sqrt{B}$, a batch is one efficient GPU matrix operation, and there are $N/B$ updates per epoch. In deep learning 'SGD' nearly always means mini-batch.` },
      { t: "qa", level: "intermediate", q: "Derive the condition on the learning rate for gradient descent to converge on a quadratic.", a: String.raw`For $L = \tfrac12 e^\top H e$ with $e = \theta - \theta^\star$, the update is $e_{t+1} = (I - \eta H)e_t$. In the eigenbasis of $H$ each component evolves as $(1 - \eta\lambda_i)^t$. All shrink iff $|1 - \eta\lambda_i| < 1$ for every $i$, i.e. $0 < \eta < 2/\lambda_{\max}$. The best fixed rate is $2/(\lambda_{\min} + \lambda_{\max})$, shrinking the error by $(\kappa - 1)/(\kappa + 1)$ per step, where $\kappa$ is the condition number.` },
      { t: "qa", level: "intermediate", q: "Why does momentum help on ill-conditioned problems?", a: String.raw`The velocity is an exponentially weighted sum of past gradients. Along the valley floor, gradients keep pointing the same way, so they add up and the effective step grows to $\eta/(1-\beta)$. Across the valley the gradient flips sign as the iterate oscillates, so contributions cancel. Result: speed in the flat direction without more oscillation in the steep one. On quadratics, well-tuned momentum needs a number of iterations that scales with $\sqrt{\kappa}$ instead of $\kappa$.` },
      { t: "qa", level: "intermediate", q: "Explain Adam's bias correction. What would happen without it?", a: String.raw`Adam's moment estimates start at zero, so $m_t = (1-\beta_1)\sum_i \beta_1^{t-i} g_i$ has expectation $(1 - \beta_1^t)\mathbb{E}[g]$ — biased toward zero early on. Dividing by $1-\beta_1^t$ (and $1-\beta_2^t$ for $v_t$) removes the bias. Without it, both moments are too small early, and $v_t$ much more so (with $\beta_2 = 0.999$, $v_1 = 0.001\,g_1^2$). The uncorrected first update would be $\eta\, m_1/\sqrt{v_1} = \eta\,(0.1\,g_1)/(\sqrt{0.001}\,|g_1|) \approx \pm 3.16\,\eta$ per coordinate — bigger steps than intended exactly when the estimates are least reliable. With correction, the first step is exactly $\pm\eta$ per coordinate.` },
      { t: "qa", level: "intermediate", q: "What does AdaGrad do well and why was RMSProp introduced?", a: String.raw`AdaGrad divides each parameter's step by the root of its summed squared gradients, which gives rarely-updated (sparse) parameters bigger steps — ideal for sparse convex problems. But the sum only grows, so the effective learning rate decays like $1/\sqrt{t}$ and long non-convex runs stall. RMSProp replaces the sum with an exponential moving average, so the normaliser tracks the *recent* gradient scale and does not decay to zero.` },
      { t: "qa", level: "advanced", q: "Why is L2 regularisation not equivalent to weight decay in Adam, and how does AdamW fix it?", a: String.raw`With SGD, adding $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss adds $\lambda\theta$ to the gradient, giving $\theta \leftarrow (1 - \eta\lambda)\theta - \eta g$ — identical to weight decay. In Adam, $\lambda\theta$ is added to $g$ and then divided by $\sqrt{\hat v}$, so each weight's regularisation is scaled by $1/\sqrt{\hat v}$: weights with historically big gradients are barely decayed and weights with small gradients are decayed very strongly. AdamW (Loshchilov & Hutter, 2019) computes the Adam step from the loss gradient alone and applies $\theta \leftarrow \theta - \eta\lambda\theta$ separately, so every weight shrinks at the same relative rate. In practice this generalises better and makes the best $\lambda$ less dependent on $\eta$.` },
      { t: "qa", level: "advanced", q: "If Adam adapts the step per parameter, why does it not solve ill-conditioning in general?", a: "Adam's preconditioner is diagonal: it rescales each coordinate on its own by the RMS of its gradient. That fixes curvature differences only when the high- and low-curvature directions line up with the coordinate axes. For a narrow valley rotated 45°, both coordinates see similar gradient sizes, the scaling is nearly uniform, and Adam behaves much like momentum SGD. Fixing that needs non-diagonal curvature information (Newton, natural gradient, K-FAC, Shampoo) or reparameterisation such as normalisation layers." },
      { t: "qa", level: "advanced", q: "How should the learning rate change when the batch size is multiplied by k, and why?", a: String.raw`A common SGD heuristic is the linear scaling rule: multiply $\eta$ by $k$ (Goyal et al., 2017). Reasoning: $k$ small steps of size $\eta$ move the parameters by about $-\eta\sum_j g_j$; one big-batch step moves by $-\eta' \bar g$ with $\bar g = \tfrac1k \sum_j g_j$; matching needs $\eta' = k\eta$, valid while the gradient changes little over those $k$ steps. It breaks early in training (hence warm-up) and beyond a critical batch size where noise is no longer the bottleneck. For adaptive methods such as Adam, a square-root rule ($\eta' = \sqrt{k}\,\eta$) is often suggested instead; in practice the best value is found by experiment.` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "Gradient descent on f(w) = (w − 3)² starts at w = 0 with η = 0.1. What is w after one step?", answer: 0.6, tolerance: 0.001, explain: "f'(0) = 2(0 − 3) = −6, so w₁ = 0 − 0.1 × (−6) = 0.6." },
          { kind: "numeric", q: "A quadratic loss has Hessian eigenvalues 0.5 and 8. What is the largest learning rate for which gradient descent converges (the supremum)?", answer: 0.25, tolerance: 0.001, explain: "Convergence iff 0 < η < 2/λ_max = 2/8 = 0.25." },
          { kind: "numeric", q: "For the same Hessian (eigenvalues 0.5 and 8), what is the optimal fixed learning rate 2/(λ_min + λ_max)?", answer: 0.2353, tolerance: 0.001, explain: "2/(0.5 + 8) = 2/8.5 = 0.2353; the contraction factor is (κ−1)/(κ+1) = 15/17 ≈ 0.882." },
          { kind: "numeric", q: "Momentum (v ← βv + g, β = 0.9, v₀ = 0) sees the constant gradient g = 1 twice. What is v after the second step?", answer: 1.9, tolerance: 0.001, explain: "v₁ = 1, v₂ = 0.9 × 1 + 1 = 1.9. In the limit v → 1/(1 − 0.9) = 10." },
          { kind: "numeric", q: "Adam with β₁ = 0.9: by what factor is m₂ multiplied in the bias correction at t = 2?", answer: 5.263, tolerance: 0.01, explain: "1/(1 − β₁²) = 1/(1 − 0.81) = 1/0.19 ≈ 5.263." },
          { q: "Mini-batch size is increased from 32 to 128 (with replacement sampling). The standard deviation of the gradient noise…", options: ["stays the same", "halves", "falls by a factor of 4", "doubles"], answer: 1, explain: "Covariance scales as Σ/B, so the standard deviation scales as 1/√B: √(128/32) = 2." },
          { q: "Which statement about AdamW is correct?", options: ["It adds λθ to the gradient before computing m and v", "It shrinks each weight by ηλθ separately from the adaptive update", "It removes the second-moment estimate", "It is identical to Adam with L2 regularisation"], answer: 1, explain: "AdamW separates weight decay from the gradient-based update: the Adam step uses the loss gradient only, and θ is also multiplied by (1 − ηλ)." },
          { q: "Why does AdaGrad tend to stall in long training runs?", options: ["Its momentum grows without bound", "Its accumulated squared-gradient sum only grows, so the effective learning rate keeps shrinking", "It has no ε term", "It uses the sign of the gradient only"], answer: 1, explain: "G_t is a running sum; η/√G_t decays roughly like 1/√t even if big steps would still help." },
        ],
      },
      {
        t: "exercise",
        title: "Learning-rate regimes by hand",
        prompt: "For f(w) = 2w² (so f'(w) = 4w), (a) find the range of η for which gradient descent converges, (b) find the η that reaches the minimum in one step, (c) compute w₁, w₂, w₃ for η = 0.3 from w₀ = 1 and say which regime this is, (d) show that Adam's first step from w₀ = 1 has size exactly η (ignore ε).",
        hint: "Write the update as w ← (1 − 4η)w.",
        solution: [
          { t: "p", text: String.raw`(a) $w_{t+1} = w_t - 4\eta w_t = (1-4\eta)w_t$. Convergence iff $|1 - 4\eta| < 1 \iff 0 < \eta < 0.5$ (here $\lambda = f'' = 4$, so $2/\lambda = 0.5$).` },
          { t: "p", text: String.raw`(b) $1 - 4\eta = 0 \iff \eta = 0.25 = 1/\lambda$.` },
          { t: "p", text: String.raw`(c) With $\eta = 0.3$ the factor is $1 - 1.2 = -0.2$: $w_1 = -0.2$, $w_2 = 0.04$, $w_3 = -0.008$. The sign flips every step while the size shrinks — oscillating convergence ($1 < \eta\lambda = 1.2 < 2$).` },
          { t: "p", text: String.raw`(d) $g_1 = 4$. $m_1 = 0.1 \times 4 = 0.4$, $v_1 = 0.001 \times 16 = 0.016$. $\hat m_1 = 0.4/0.1 = 4$, $\hat v_1 = 0.016/0.001 = 16$. Step $= \eta \cdot 4/\sqrt{16} = \eta$. In general $\hat m_1/\sqrt{\hat v_1} = g_1/|g_1| = \pm 1$.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: Adam + L2 versus AdamW",
        prompt: "Write a NumPy experiment showing why decoupled weight decay matters. Use two weights starting at 1 whose loss gradient is pure zero-mean noise, with standard deviation 10 for the first weight and 0.01 for the second, so any steady shrinkage comes only from the regulariser. Run 3000 steps of (i) Adam with λθ added to the gradient and (ii) AdamW with the decay applied separately, using η = 10⁻³ and λ = 1 and identical noise. Compare both to the pure decay factor (1 − ηλ)³⁰⁰⁰.",
        hint: "Re-create the random generator with the same seed inside each run so both see identical noise.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "adam_l2_vs_adamw.py",
            code: String.raw`import numpy as np

# The data loss contributes only zero-mean noise, with very different scales per weight:
# weight 0 sees gradients ~ N(0, 10^2), weight 1 sees gradients ~ N(0, 0.01^2).
# Any systematic shrinkage therefore comes from the regulariser alone.
scale = np.array([10.0, 0.01])

def run(decoupled, lam=1.0, lr=1e-3, steps=3000, b1=0.9, b2=0.999, eps=1e-8):
    rng = np.random.default_rng(0)                   # identical noise for both variants
    th, m, v = np.ones(2), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = scale * rng.normal(size=2)
        if not decoupled:
            g = g + lam * th                          # Adam + L2: penalty gradient joins g
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        th = th - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if decoupled:
            th = th - lr * lam * th                   # AdamW: decay applied outside Adam
    return th

print("start                   : [1. 1.]")
print("pure decay (1-lr*lam)^T : %.4f" % (1 - 1e-3) ** 3000)
print("Adam + L2   final theta :", np.round(run(False), 4))
print("AdamW       final theta :", np.round(run(True), 4))`,
            output: String.raw`start                   : [1. 1.]
pure decay (1-lr*lam)^T : 0.0497
Adam + L2   final theta : [0.7559 0.    ]
AdamW       final theta : [0.0367 0.0371]`,
          },
          { t: "p", text: String.raw`With Adam + L2, the decay gradient $\lambda\theta \approx 1$ is tiny next to the noise of scale 10 on the first weight, so after dividing by $\sqrt{\hat v} \approx 10$ it barely acts: that weight is still 0.76. On the second weight the decay gradient dominates, $\sqrt{\hat v} \approx \lambda\theta$, and Adam turns it into a full-size step of about $\eta$ per iteration, driving the weight to 0 within roughly 1000 steps. AdamW shrinks both weights at the same relative rate; both end near the pure decay value 0.0497 (the gap is the random walk from noisy Adam steps).` },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          String.raw`Training solves $\min_\theta L(\theta)$, the average loss over the training set; for networks $L$ is non-convex.`,
          String.raw`Gradient descent $\theta \leftarrow \theta - \eta\nabla L$ follows from the first-order Taylor expansion: the negative gradient is the steepest local descent direction.`,
          String.raw`On a quadratic with Hessian eigenvalues $\lambda_i$: each direction scales by $1 - \eta\lambda_i$ per step; convergence iff $0 < \eta < 2/\lambda_{\max}$; best rate $(\kappa-1)/(\kappa+1)$ at $\eta = 2/(\lambda_{\min}+\lambda_{\max})$.`,
          String.raw`Mini-batch gradients are unbiased with noise standard deviation $\propto 1/\sqrt{B}$; many cheap noisy steps beat few exact ones.`,
          String.raw`Momentum: $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \eta v$; effective step $\eta/(1-\beta)$ along steady directions. Nesterov evaluates the gradient at the look-ahead point.`,
          String.raw`AdaGrad: divide by $\sqrt{\sum g^2}$ (decays to zero). RMSProp: divide by $\sqrt{\text{EMA}(g^2)}$.`,
          String.raw`Adam: $m$ and $v$ are EMAs of $g$ and $g^2$; bias correction divides by $1-\beta^t$ because $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$; update $-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, about $\eta$ per coordinate.`,
          String.raw`AdamW: apply weight decay $-\eta\lambda\theta$ outside the adaptive update; Adam + L2 regularises weights unevenly.`,
          String.raw`Nadam: Adam with a Nesterov-style numerator $\beta_1\hat m_t + (1-\beta_1)g_t/(1-\beta_1^t)$.`,
          String.raw`Clip gradients by global norm, $g \leftarrow g\min(1, c/\lVert g\rVert)$, between the backward pass and the optimizer step; schedule $\eta$ with warm-up and decay.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Learning Rate Schedules and Gradient Clipping](/learn/m06/lr-schedules) — how to vary the most important hyperparameter during training (warm-up, step, cosine) and how clipping keeps updates bounded.",
          "[Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW](/learn/m06/momentum-adaptive) — a companion topic page for the adaptive optimizers covered here.",
          "[Weight Initialization](/learn/m07/initialization) — initialisation sets the starting gradient scale the optimizer depends on.",
          "[Batch, Layer and Group Normalization](/learn/m07/normalization) — normalisation layers improve the conditioning of the loss surface, exactly what limits gradient descent.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — puts AdamW's weight decay into the wider fight against overfitting.",
        ],
      },
    ],
  },
};

export default lesson;
