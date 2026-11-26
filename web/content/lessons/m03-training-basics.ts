import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m03/training-basics",
  title: "Training: Epochs, Batches and Overfitting",
  summary:
    "What one pass of a training loop really does, how to count epochs, batches and iterations, and how to spot and fix underfitting and overfitting with the bias–variance trade-off and learning curves.",
  minutes: 55,
  level: "beginner",
  prerequisites: ["m03/artificial-neuron", "m03/multilayer-networks", "m06/gradient-descent"],
  next: ["m06/gradient-descent", "m07/regularization", "m05/regression-losses"],
  terms: [
    "training-loop",
    "epoch",
    "batch",
    "iteration",
    "parameters",
    "hyperparameters",
    "underfitting",
    "overfitting",
    "generalization",
    "bias-variance",
    "model-capacity",
    "learning-curve",
    "validation-set",
  ],
  sections: {
    // ------------------------------------------------------------------
    intro: [
      {
        t: "p",
        text: "A new neural network has random parameters and is useless. **Training** turns it into something useful: show it examples, measure how wrong it is, work out which way each parameter should move to be less wrong, move a little, and repeat — tens of thousands to millions of times. The code that does this is the **training loop**. Almost every deep learning program, from a 10-line demo to a large language model, has the same five-step loop at its heart.",
      },
      {
        t: "p",
        text: "This lesson covers two day-one skills. First, the **mechanics and words** of training: what happens in one iteration, and how **epochs**, **batches** and **iterations** relate (with exact counting, including the awkward last batch). Second, the big question of machine learning: **does the model generalize** — does it work on new data? A model that only memorizes its training data is worthless. We define **underfitting** and **overfitting**, explain them with the **bias–variance trade-off** and **model capacity**, and learn to read **learning curves**, the plots you will check every time you train.",
      },
      {
        t: "p",
        text: "Why do we need all this? Because driving training error down is easy, and it is not the goal. A flexible enough model can reach zero training error and still predict terribly (curve fitters knew this long before neural networks). Held-out validation data, learning curves and capacity control turn \"fit the data\" into the real goal: \"do well on data we have not seen\".",
      },
      {
        t: "callout",
        kind: "key",
        title: "Where this is used",
        text: "Everywhere. How many epochs, what batch size, when to stop, whether to collect more data, whether to grow the network or add regularization — you make each of these daily choices by reading train and validation curves, as shown here.",
      },
    ],

    // ------------------------------------------------------------------
    prerequisites: [
      {
        t: "list",
        items: [
          "[The Artificial Neuron](/learn/m03/artificial-neuron) — a model with **weights and a bias** that turns an input into a prediction, and the idea that learning means changing those weights.",
          "[Multi-Layer Networks and Model Capacity](/learn/m03/multilayer-networks) — bigger networks can represent more functions. A quick read is enough; capacity is re-explained here.",
          "[Gradient Descent](/learn/m06/gradient-descent) — the update rule $w \\leftarrow w - \\eta\\,\\partial L/\\partial w$. A quick read is enough: the gradient points uphill, and we step the other way. Optimizer details come later.",
          "[Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — for the one worked gradient in the Math section (derivative of a squared error).",
          "[Probability](/learn/m02/probability) — expectation and variance, for the bias–variance decomposition. You need $\\mathbb{E}[X]$, $\\mathrm{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}X)^2]$, and the fact that independent zero-mean noise has no cross-term with anything else.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A student preparing for an exam",
        text: "Training data is the practice book; the validation set is a mock exam with new questions; the test set is the real exam. One **epoch** is working through the whole practice book once. A **batch** is the handful of problems you do before checking answers and adjusting your understanding; each adjustment is an **iteration**. A student who has not studied enough fails both the practice and the mock exam: **underfitting**. A student who memorized the answer key scores 100% on the practice book but fails the mock exam: **overfitting**. We want the student who learned the method. (Where the analogy stops: a model never \"understands\"; it only fits numbers.)",
      },
      {
        t: "p",
        text: "Picture fitting a curve through noisy dots that follow a wave. A straight line misses the pattern: too simple, so it is wrong on old and new points alike. A wildly wiggling curve through every dot has also captured the noise — the random jitter of each dot — and swings to silly values between and beyond them. The sensible curve sits in between: bendy enough to follow the wave, stiff enough not to chase each dot.",
      },
      {
        t: "p",
        text: "The key point: **training error alone points the wrong way here**. The wiggly curve has the lowest training error of all. So we always keep some data aside — the **validation set** — that the model never trains on, and judge models there. Training error measures fit; validation error estimates how well the model **generalizes**.",
      },
      {
        t: "p",
        text: "Why use batches at all? Computing the exact gradient over a million examples before one step is slow and wasteful. The gradient from a random 64 examples already points roughly the right way. So we take many cheap, slightly noisy steps instead of a few costly exact ones. One pass over all the data, in batches, is an epoch; we usually need many.",
      },
    ],

    // ------------------------------------------------------------------
    mechanism: [
      { t: "h", text: "Anatomy of one training iteration" },
      {
        t: "steps",
        items: [
          { title: "Sample a mini-batch", text: "Take the next $B$ examples $(x_i, y_i)$ from a shuffled order of the training set. $B$ is the **batch size**." },
          { title: "Zero the gradients", text: "Clear the gradients left from the last iteration. PyTorch and similar frameworks **add up** gradients by default, so skipping this mixes the old gradient into the new one." },
          { title: "Forward pass", tex: String.raw`\hat{y}_i = f(x_i;\,\theta)`, text: "Run the batch through the model with the current parameters $\\theta$ to get predictions." },
          { title: "Compute the loss", tex: String.raw`L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell(\hat{y}_i, y_i)`, text: "One number saying how wrong the predictions are on this batch (e.g. mean squared error)." },
          { title: "Backward pass", tex: String.raw`g = \nabla_\theta L`, text: "Backpropagation computes the gradient of the loss for every parameter." },
          { title: "Update the parameters", tex: String.raw`\theta \leftarrow \theta - \eta\, g`, text: "The optimizer moves each parameter a small step against its gradient. $\\eta$ is the **learning rate**. Adam and friends change how the step is computed, not the loop." },
          { title: "Repeat until the epoch ends, then evaluate", text: "Once every batch has been used (one epoch), switch to evaluation mode, compute training and validation loss **without** updating anything, log them, and maybe save a checkpoint if validation improved." },
        ],
      },
      {
        t: "p",
        text: "Around this sits an outer loop over epochs. At the start of each epoch the training set is **reshuffled**, so batches differ each time. This matters: with sorted data (all cats, then all dogs), consecutive gradients push the model the same way and training swings back and forth.",
      },
      { t: "h", text: "Epochs, batches and iterations" },
      {
        t: "table",
        head: ["Term", "Definition", "Typical values"],
        rows: [
          ["Batch (mini-batch)", "$B$ training examples processed together to compute one gradient estimate.", "32–512 for vision; thousands of sequences for large language models"],
          ["Iteration (step)", "One parameter update = one forward + backward pass on one batch.", "Thousands to millions over a run"],
          ["Epoch", "One full pass over every training example.", "1 (huge datasets) to hundreds (small datasets)"],
          ["Batch gradient descent", "$B = N$: one iteration per epoch, exact gradient.", "Small datasets, convex problems"],
          ["Stochastic gradient descent (pure)", "$B = 1$: $N$ iterations per epoch, very noisy gradient.", "Mostly historical / theory"],
        ],
      },
      {
        t: "p",
        text: String.raw`With $N$ examples and batch size $B$, there are $\lceil N/B \rceil$ iterations per epoch (the ceiling: round up). The leftovers form a smaller last batch of size $N - (\lceil N/B\rceil - 1)B$. Setting **drop_last** (a PyTorch DataLoader argument) throws that partial batch away instead: $\lfloor N/B \rfloor$ iterations, skipping $N \bmod B$ examples per epoch. Shuffling changes which ones are skipped each epoch, so none is lost for good. For $E$ epochs, total iterations are $E \cdot \lceil N/B \rceil$.`,
      },
      {
        t: "callout",
        kind: "note",
        title: "Example: 1,050 examples, batch size 32, 10 epochs",
        text: String.raw`$\lceil 1050/32 \rceil = 33$ iterations per epoch: 32 full batches (1,024 examples) and a final batch of $1050 - 32\cdot 32 = 26$. Ten epochs = 330 updates. With drop_last: 32 iterations per epoch, 26 examples skipped per epoch, 320 updates in total.`,
      },
      { t: "h", text: "Parameters vs hyperparameters" },
      {
        t: "p",
        text: "**Parameters** are the numbers the loop learns: weights and biases. **Hyperparameters** are everything you choose that the gradient does not set: learning rate, batch size, epochs, layers and units, activation, optimizer and its settings, regularization strength, dropout rate, data augmentation. Tune them by training several models and comparing **validation** scores — never test scores, or the test set stops being a fair estimate.",
      },
      { t: "h", text: "Three data splits" },
      {
        t: "list",
        items: [
          "**Training set** — used to compute gradients and update parameters.",
          "**Validation set** (dev set) — never used for gradients; used to choose hyperparameters, pick the best epoch (early stopping) and spot overfitting.",
          "**Test set** — used once at the end, for a fair estimate on new data. Common splits are 80/10/10 or 70/15/15; with millions of examples, a few percent each for validation and test is enough.",
        ],
      },
      { t: "h", text: "Underfitting, overfitting and capacity" },
      {
        t: "p",
        text: "**Model capacity** is, roughly, how big and flexible the set of functions a model can represent is. For polynomials it grows with the degree. For neural networks it grows with parameters, depth, and even training time (after a few steps a network can only reach functions close to where it started). Capacity compared with the amount of data decides which of three regimes you are in:",
      },
      {
        t: "table",
        head: ["Regime", "Training error", "Validation error", "Gap (val − train)", "Cause"],
        rows: [
          ["Underfitting", "High", "High", "Small", "Capacity too low, or training stopped too early / learning rate wrong"],
          ["Good fit", "Low (near the noise level)", "Low, close to its minimum", "Small to moderate", "Capacity matched to data"],
          ["Overfitting", "Very low (below the noise level)", "Rising / high", "Large", "Capacity too high for the amount of data; model learns noise"],
        ],
      },
      { t: "h", text: "Learning curves and how to read them" },
      {
        t: "p",
        text: "Two kinds of plot are called learning curves. (1) **Loss vs epoch**, for training and validation. Training loss normally keeps falling (with some batch noise). If validation loss falls with it, keep training. If validation loss bottoms out and starts rising while training loss keeps falling, overfitting has begun; the epoch at the minimum is where **early stopping** would stop. If both curves are flat and high, the model underfits: train longer, raise the learning rate, or add capacity. If training loss itself rises or explodes, the learning rate is too large (a training problem, not a generalization problem).",
      },
      {
        t: "p",
        text: "(2) **Error vs training-set size**, for a fixed model. Training error **rises** with more data (more points are harder to fit), validation error **falls**, and both level off. If they meet at a high value, the model is biased: more data won't help, more capacity will. If a large gap is still shrinking, variance is the limit: more data (or regularization) will help.",
      },
      { t: "h", text: "Remedies" },
      {
        t: "table",
        head: ["Symptom", "Remedies (roughly in order of what to try)"],
        rows: [
          ["Underfitting (both errors high)", "Train longer; tune the learning rate; add capacity (wider/deeper network, better architecture); reduce regularization; better input features; check for bugs (shuffled labels, wrong normalization)."],
          ["Overfitting (large gap)", "More data; data augmentation; early stopping; L2 regularization / weight decay; dropout; smaller model; ensembling; transfer learning from a pre-trained model."],
          ["Validation noisy/unreliable", "Bigger validation set; k-fold cross-validation for small datasets; average over several random seeds."],
          ["Training loss diverges or oscillates", "Lower the learning rate; gradient clipping; normalize inputs; check the loss code."],
        ],
      },
    ],

    // ------------------------------------------------------------------
    math: [
      { t: "h", text: "Empirical risk and true risk" },
      {
        t: "p",
        text: String.raw`Say the data comes from an unknown distribution $p(x,y)$, $f(x;\theta)$ is the model with parameters $\theta$, and $\ell$ is the loss on one example. We want to minimize the **true risk** (average loss on new data). We can only compute the **empirical risk** on the $N$ training examples:`,
      },
      {
        t: "math",
        tex: String.raw`R(\theta) = \mathbb{E}_{(x,y)\sim p}\big[\ell(f(x;\theta), y)\big], \qquad \hat{R}_{\text{train}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i;\theta), y_i)`,
      },
      {
        t: "p",
        text: String.raw`Training minimizes $\hat R_{\text{train}}$. The **generalization gap** is $R(\theta) - \hat R_{\text{train}}(\theta)$. Since $\theta$ was picked using the training points, $\hat R_{\text{train}}$ is too optimistic. The error on a separate validation set is an unbiased (fair) estimate of $R$ for a fixed $\theta$.`,
      },
      { t: "h", text: "Mini-batch gradient and the update" },
      {
        t: "p",
        text: String.raw`For a mini-batch $\mathcal{B}$ of size $B$, the loss and gradient are $L_{\mathcal B}(\theta) = \frac1B\sum_{i\in\mathcal B}\ell_i$ and $g_{\mathcal B} = \frac1B\sum_{i\in\mathcal B}\nabla_\theta \ell_i$. If $\mathcal B$ is a uniformly random subset, $\mathbb E[g_{\mathcal B}] = \nabla_\theta \hat R_{\text{train}}$. In plain words: on average the batch gradient equals the full gradient (it is **unbiased**), and its noise (variance) shrinks like $1/B$. The update is $\theta \leftarrow \theta - \eta\, g_{\mathcal B}$.`,
      },
      { t: "h", text: "Worked example: one iteration by hand" },
      {
        t: "p",
        text: String.raw`Model $\hat y = w x + b$ with $w = 0.5$, $b = 0$, learning rate $\eta = 0.05$, MSE loss $L = \frac1B\sum_i(\hat y_i - y_i)^2$. A mini-batch of $B = 4$ examples from the true relation $y = 2x + 1$: $x = (1, 2, 3, 4)$, $y = (3, 5, 7, 9)$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Forward pass", tex: String.raw`\hat y = 0.5\,x + 0 = (0.5,\ 1.0,\ 1.5,\ 2.0)` },
          { title: "Residuals", tex: String.raw`r = \hat y - y = (-2.5,\ -4.0,\ -5.5,\ -7.0)` },
          { title: "Loss", tex: String.raw`L = \tfrac14(6.25 + 16 + 30.25 + 49) = \tfrac{101.5}{4} = 25.375` },
          { title: "Gradient w.r.t. w", tex: String.raw`\frac{\partial L}{\partial w} = \frac{2}{B}\sum_i r_i x_i = \frac{2}{4}(-2.5 - 8 - 16.5 - 28) = \frac{2}{4}(-55) = -27.5` },
          { title: "Gradient w.r.t. b", tex: String.raw`\frac{\partial L}{\partial b} = \frac{2}{B}\sum_i r_i = \frac{2}{4}(-19) = -9.5` },
          { title: "Update", tex: String.raw`w \leftarrow 0.5 - 0.05(-27.5) = 1.875, \qquad b \leftarrow 0 - 0.05(-9.5) = 0.475` },
          { title: "Check: loss after the step", tex: String.raw`\hat y = (2.35,\ 4.225,\ 6.1,\ 7.975), \quad L = 0.7209375`, text: "One iteration reduced this batch's loss from 25.375 to about 0.721. (Verified in Python.)" },
        ],
      },
      { t: "h", text: "Epoch arithmetic" },
      {
        t: "math",
        tex: String.raw`\text{iters/epoch} = \left\lceil \tfrac{N}{B} \right\rceil, \quad B_{\text{last}} = N - \left(\left\lceil \tfrac{N}{B} \right\rceil - 1\right)B, \quad \text{total iters} = E\left\lceil \tfrac{N}{B} \right\rceil`,
        caption: "With drop_last: $\\lfloor N/B \\rfloor$ iterations and $N \\bmod B$ examples skipped per epoch.",
      },
      {
        t: "p",
        text: "MNIST-sized: $N = 60{,}000$, $B = 128$: $\\lceil 468.75 \\rceil = 469$ iterations per epoch, last batch $60000 - 468\\cdot128 = 96$; 5 epochs = 2,345 updates. With drop_last: 468 iterations per epoch.",
      },
      { t: "h", text: "The bias–variance decomposition" },
      {
        t: "p",
        text: String.raw`Assume $y = f(x) + \varepsilon$ with $\mathbb E[\varepsilon] = 0$, $\mathrm{Var}(\varepsilon) = \sigma^2$, and noise independent of everything else. Training on a random training set $D$ gives $\hat f_D$. At a fixed input $x$, let $\bar f(x) = \mathbb E_D[\hat f_D(x)]$ be the average prediction over all possible training sets. The expected squared error on a **new** noisy label at $x$ is:`,
      },
      {
        t: "math",
        tex: String.raw`\mathbb{E}_{D,\varepsilon}\big[(y - \hat f_D(x))^2\big] = \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big[(\hat f_D(x) - \bar f(x))^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}`,
      },
      {
        t: "details",
        summary: "Derivation",
        blocks: [
          {
            t: "p",
            text: String.raw`Write $y - \hat f_D = \varepsilon + (f - \bar f) + (\bar f - \hat f_D)$ (all at the same $x$). Square and take expectations. The cross-terms vanish: $\mathbb E[\varepsilon \cdot (\cdot)] = 0$ because $\varepsilon$ is zero-mean and independent of $D$; and $\mathbb E_D[(f - \bar f)(\bar f - \hat f_D)] = (f - \bar f)\,\mathbb E_D[\bar f - \hat f_D] = (f-\bar f)\cdot 0$, because $f - \bar f$ is a constant and $\bar f$ is the mean of $\hat f_D$. What remains is $\mathbb E[\varepsilon^2] + (f - \bar f)^2 + \mathbb E_D[(\bar f - \hat f_D)^2] = \sigma^2 + \text{bias}^2 + \text{variance}$. Averaging over $x$ gives the same identity for the expected test MSE.`,
          },
        ],
      },
      {
        t: "list",
        items: [
          "**Bias** — the error of the *average* model: a built-in mistake because the model family is too limited (a line cannot follow a sine). High bias ↔ underfitting.",
          "**Variance** — how much the learned function changes if you train on a different sample of the same size. High variance ↔ overfitting.",
          String.raw`**Irreducible error** $\sigma^2$ — noise in the labels themselves; no model can beat it on average.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Numerical check.** Fit degree-$d$ polynomials to 25 points of $y = \sin(2\pi x) + \varepsilon$, $\sigma = 0.3$, over 2,000 independent training sets, and evaluate on 50 fixed inputs. Bias² + variance + σ² matches a direct Monte-Carlo estimate of the test MSE within sampling error, and the U-shape shows up: bias falls with degree, variance rises.`,
      },
      {
        t: "code",
        lang: "python",
        title: "Monte-Carlo verification of the bias–variance decomposition",
        code: String.raw`import numpy as np

# Monte-Carlo check of  E[(y - f_hat(x))^2] = bias^2 + variance + sigma^2
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)          # the true function
sigma = 0.3                                   # noise standard deviation
n_train = 25                                  # points per training set
n_sets = 2000                                 # number of independent training sets
x_test = np.linspace(0.05, 0.95, 50)          # fixed evaluation inputs

def features(x, d):
    return np.vander(2 * x - 1, d + 1, increasing=True)   # rescale x to [-1, 1]

print(f"{'degree':>6} {'bias^2':>8} {'variance':>9} {'noise':>7} {'sum':>7} {'MC test MSE':>12}")
for d in [0, 1, 3, 5, 7, 8]:
    preds = np.empty((n_sets, x_test.size))
    sq_err = np.empty(n_sets)
    for s in range(n_sets):
        x = rng.uniform(0, 1, n_train)
        y = f(x) + rng.normal(0, sigma, n_train)
        coef, *_ = np.linalg.lstsq(features(x, d), y, rcond=None)
        preds[s] = features(x_test, d) @ coef
        y_new = f(x_test) + rng.normal(0, sigma, x_test.size)   # fresh test noise
        sq_err[s] = np.mean((y_new - preds[s]) ** 2)
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - f(x_test)) ** 2)
    var = np.mean(preds.var(axis=0))
    print(f"{d:>6} {bias2:8.4f} {var:9.4f} {sigma**2:7.4f} {bias2 + var + sigma**2:7.4f} {sq_err.mean():12.4f}")`,
        output: String.raw`degree   bias^2  variance   noise     sum  MC test MSE
     0   0.5426    0.0235  0.0900  0.6562       0.6592
     1   0.1535    0.0246  0.0900  0.2681       0.2689
     3   0.0029    0.0164  0.0900  0.1093       0.1084
     5   0.0000    0.0528  0.0900  0.1429       0.1419
     7   0.0009    0.8695  0.0900  0.9604       0.9619
     8   0.0047    6.0523  0.0900  6.1470       6.1511`,
      },
      {
        t: "p",
        text: "Degree 3 has the lowest total. Degrees 0–1 are dominated by bias (underfitting), degrees 7–8 by variance (overfitting). With only 25 points, a degree-8 polynomial sometimes gets a training set with a gap and swings wildly, and those rare blow-ups dominate the average.",
      },
      { t: "h", text: "Why training error is optimistic" },
      {
        t: "p",
        text: String.raw`Take linear least squares with $p$ parameters, $n$ training points, Gaussian noise of variance $\sigma^2$, and a model family that contains the true function. The expected training MSE is exactly $\sigma^2(1 - p/n)$ — **below** the noise floor, because the fit spends its $p$ degrees of freedom soaking up noise. With $\sigma^2 = 0.0625$ and $n = 20$: $p = 4$ (cubic) gives $0.05$ and $p = 10$ (degree 9) gives $0.03125$. The coding exercise measures 0.0509 and 0.0319. As $p \to n$, training error goes to zero while validation error explodes.`,
      },
    ],

    // ------------------------------------------------------------------
    visual: [
      {
        t: "p",
        text: String.raw`The simulator draws noisy samples $y = \sin(2\pi x) + \varepsilon$, with $\varepsilon \sim \mathcal N(0, \sigma^2)$ and $x$ uniform in $[0, 1]$, from a seeded random generator. It splits them 2/3 training (filled dots) and 1/3 validation (hollow dots). It fits a degree-$d$ polynomial by least squares on the training points only (normal equations, a tiny ridge of $10^{-8}$ for stability, $x$ rescaled to $[-1, 1]$). Left plot: the data, the true function (dashed) and the fit. Right plot: training and validation MSE for degrees 0–15 on a log scale, marking the current degree and the best-validation degree.`,
      },
      { t: "sim", id: "fitting", caption: "Polynomial fitting: underfitting, good fit and overfitting as the degree grows." },
      {
        t: "steps",
        items: [
          { title: "Press Play", text: "The degree runs from 0 to 15. The narration moves from **underfitting** (both errors high; a flat line or gentle slope) through **good fit** (validation near its minimum — degrees 5–7 for the default sample; the range moves with each new sample) to **overfitting** (training error keeps falling, validation rises, the curve wiggles between dots)." },
          { title: "Read the right-hand plot as a capacity curve", text: "The classic U-shape. Training error only goes down with degree, since each higher degree includes all lower ones. Validation error is U-shaped. The best-validation marker is the degree you would pick." },
          { title: "Small vs large data", text: "Set degree 9 (then 12) and compare n = 10 with n = 200. With 10 points (about 6–7 for training), the degree-9 fit passes through every training dot and swings off the chart. With 200 points the same polynomial is smooth and close to the sine. More data cuts variance: overfitting depends on capacity **relative to** data." },
          { title: "No noise vs heavy noise", text: "Set σ = 0: high degrees barely hurt, because there is no noise to memorize (polynomials approximate the smooth sine well). Set σ = 0.5: every validation error rises (it cannot go much below σ² = 0.25), and high-degree fits swing far more wildly, with more noise to memorize. On average noisier data supports less capacity. But with only about 10 validation points, the best degree itself jumps around — press **New sample** a few times to see how unreliable one small validation set is." },
          { title: "Resample", text: "Press **New sample** several times at a high degree with small n. The curve changes a lot each time: that is **variance**, seen directly. At degree 1 the line barely moves but is always wrong: that is **bias**." },
        ],
      },
      {
        t: "callout",
        kind: "tip",
        title: "What to notice",
        text: "At high degrees, training MSE drops below the noise variance σ²: the model is fitting noise, which never helps on new data. On average validation MSE cannot go much below σ², so a validation error near σ² means you are close to the best possible.",
      },
    ],

    // ------------------------------------------------------------------
    scratch: [
      {
        t: "p",
        text: "A NumPy program in two parts. Part 1 is a real mini-batch training loop for a cubic model (shuffle → batch → forward → loss → gradient → update, then train/validation evaluation each epoch). Part 2 sweeps capacity (degree 0–15) with the exact least-squares fit, showing underfitting, good fit and overfitting on the same data.",
      },
      {
        t: "code",
        lang: "python",
        title: "training_basics.py",
        runnable: true,
        code: String.raw`import math
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# 1. Data: noisy samples of a known function, split train / val
# ---------------------------------------------------------------
N_SAMPLES = 30          # total number of examples
NOISE_STD = 0.25        # standard deviation of the label noise
x = rng.uniform(0, 1, N_SAMPLES)
y = np.sin(2 * np.pi * x) + rng.normal(0, NOISE_STD, N_SAMPLES)

perm = rng.permutation(N_SAMPLES)
n_train = 2 * N_SAMPLES // 3                     # 2/3 train, 1/3 validation
tr, va = perm[:n_train], perm[n_train:]

def poly_features(x, degree):
    """Columns 1, t, t^2, ..., t^degree with t = 2x - 1 in [-1, 1]."""
    return np.vander(2 * x - 1, degree + 1, increasing=True)

def mse(pred, target):
    return float(np.mean((pred - target) ** 2))

# ---------------------------------------------------------------
# 2. A complete mini-batch training loop (gradient descent on MSE)
# ---------------------------------------------------------------
def train(X_tr, y_tr, X_va, y_va, epochs, batch_size, lr, seed=0):
    r = np.random.default_rng(seed)
    w = np.zeros(X_tr.shape[1])                   # parameters (learned)
    n = len(y_tr)
    iters_per_epoch = math.ceil(n / batch_size)  # last batch may be smaller
    step = 0
    history = []
    for epoch in range(1, epochs + 1):
        order = r.permutation(n)                  # reshuffle every epoch
        for i in range(iters_per_epoch):
            idx = order[i * batch_size:(i + 1) * batch_size]
            Xb, yb = X_tr[idx], y_tr[idx]
            pred = Xb @ w                         # forward pass
            resid = pred - yb                     # loss = mean(resid^2)
            grad = 2 * Xb.T @ resid / len(yb)     # backward pass (dL/dw)
            w -= lr * grad                        # update
            step += 1
        history.append((epoch, step, mse(X_tr @ w, y_tr), mse(X_va @ w, y_va)))
    return w, history

DEGREE, EPOCHS, BATCH, LR = 3, 300, 8, 0.1
Xtr, Xva = poly_features(x[tr], DEGREE), poly_features(x[va], DEGREE)
w, hist = train(Xtr, y[tr], Xva, y[va], EPOCHS, BATCH, LR)
print(f"train examples={n_train}, batch={BATCH} -> iterations/epoch={math.ceil(n_train / BATCH)}")
print("epoch  iters  train_MSE  val_MSE")
for e, s, a, b in hist:
    if e in (1, 2, 5, 10, 50, 100, 200, 300):
        print(f"{e:5d} {s:6d} {a:10.4f} {b:8.4f}")

# ---------------------------------------------------------------
# 3. Capacity sweep: exact least-squares fit for each degree
# ---------------------------------------------------------------
RIDGE = 1e-8   # tiny ridge so X^T X is always invertible
print("\ndegree  train_MSE  val_MSE")
results = []
for d in range(0, 16):
    Xt, Xv = poly_features(x[tr], d), poly_features(x[va], d)
    A = Xt.T @ Xt + RIDGE * np.eye(d + 1)
    coef = np.linalg.solve(A, Xt.T @ y[tr])
    results.append((d, mse(Xt @ coef, y[tr]), mse(Xv @ coef, y[va])))
    print(f"{d:6d} {results[-1][1]:10.4f} {results[-1][2]:8.4f}")
best = min(results, key=lambda r: r[2])
print(f"best validation degree = {best[0]} (val MSE {best[2]:.4f}); noise variance = {NOISE_STD**2:.4f}")`,
        output: String.raw`train examples=20, batch=8 -> iterations/epoch=3
epoch  iters  train_MSE  val_MSE
    1      3     0.3436   0.6442
    2      6     0.3169   0.5928
    5     15     0.2734   0.5320
   10     30     0.2401   0.4945
   50    150     0.1389   0.3238
  100    300     0.0909   0.2216
  200    600     0.0674   0.1468
  300    900     0.0644   0.1286

degree  train_MSE  val_MSE
     0     0.4034   0.7527
     1     0.2653   0.4803
     2     0.2322   0.5254
     3     0.0630   0.1147
     4     0.0629   0.1129
     5     0.0245   0.0425
     6     0.0237   0.0419
     7     0.0201   0.0672
     8     0.0192   0.0610
     9     0.0192   0.0612
    10     0.0171   0.0818
    11     0.0156   0.1192
    12     0.0154   0.1422
    13     0.0133   0.0995
    14     0.0128   0.1324
    15     0.0100   0.1447
best validation degree = 6 (val MSE 0.0419); noise variance = 0.0625`,
      },
      {
        t: "list",
        items: [
          "**Iteration counting**: 20 training examples with batch 8 give $\\lceil 20/8 \\rceil = 3$ iterations per epoch (batches of 8, 8 and 4), so 300 epochs = 900 updates, as the iters column shows.",
          "**Part 1 learning curve**: both errors fall together. After 300 epochs the gradient-descent cubic (train 0.0644) nears the exact least-squares cubic from Part 2 (train 0.0630). It is still slowly converging, not overfitting.",
          "**Part 2 capacity sweep**: degrees 0–2 underfit (both errors high), degrees 5–6 fit well (best validation 0.0419). From degree 10 on, training error keeps shrinking (to 0.0100, far below the noise variance 0.0625) while validation error climbs back above 0.14: overfitting.",
          "The validation curve is bumpy (degree 13 dips) because 10 validation points give a small, noisy estimate — a reason to use bigger validation sets or cross-validation.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "The standard PyTorch training loop: a DataLoader, explicit zero_grad → forward → loss → backward → step, a separate evaluation pass under torch.no_grad(), and a best-validation checkpoint (how early stopping works). The network is purposely large (4,353 parameters) for 20 training points, so overfitting shows up in the validation curve.",
              },
              {
                t: "code",
                lang: "python",
                title: "train_loop_pytorch.py",
                code: String.raw`import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# Data: 30 noisy samples of sin(2*pi*x); 20 for training, 10 for validation
N, NOISE_STD = 30, 0.25
x = torch.rand(N, 1)
y = torch.sin(2 * math.pi * x) + NOISE_STD * torch.randn(N, 1)
x_tr, y_tr, x_va, y_va = x[:20], y[:20], x[20:], y[20:]

BATCH = 8
train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)
print("batches per epoch:", len(train_loader),
      "| with drop_last=True:",
      len(DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, drop_last=True)))

# A deliberately high-capacity model for 20 points (4,353 parameters)
model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(),
                      nn.Linear(64, 64), nn.Tanh(),
                      nn.Linear(64, 1))
print("trainable parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)   # hyperparameter
EPOCHS = 1500
best_val, best_epoch, best_state = float("inf"), 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:          # one iteration per mini-batch
        opt.zero_grad()                  # 1. clear old gradients
        pred = model(xb)                 # 2. forward pass
        loss = loss_fn(pred, yb)         # 3. loss
        loss.backward()                  # 4. backward pass
        opt.step()                       # 5. parameter update

    model.eval()
    with torch.no_grad():                # evaluation: no gradients, no updates
        tr = loss_fn(model(x_tr), y_tr).item()
        va = loss_fn(model(x_va), y_va).item()
    if va < best_val:                    # remember the best validation checkpoint
        best_val, best_epoch, best_state = va, epoch, copy.deepcopy(model.state_dict())
    if epoch in (1, 10, 50, 100, 250, 500, 1000, 1500):
        print(f"epoch {epoch:4d}  train MSE {tr:.4f}  val MSE {va:.4f}")

print(f"best validation MSE {best_val:.4f} at epoch {best_epoch}")
model.load_state_dict(best_state)        # roll back to the best checkpoint`,
                output: String.raw`batches per epoch: 3 | with drop_last=True: 2
trainable parameters: 4353
epoch    1  train MSE 0.4511  val MSE 0.1843
epoch   10  train MSE 0.2140  val MSE 0.2735
epoch   50  train MSE 0.0507  val MSE 0.1081
epoch  100  train MSE 0.0479  val MSE 0.1308
epoch  250  train MSE 0.0352  val MSE 0.0609
epoch  500  train MSE 0.0276  val MSE 0.0523
epoch 1000  train MSE 0.0160  val MSE 0.0625
epoch 1500  train MSE 0.0131  val MSE 0.0685
best validation MSE 0.0427 at epoch 526`,
              },
              {
                t: "p",
                text: "Reading the curve: training MSE falls steadily to 0.0131, well below the noise variance 0.0625. Validation MSE hits its minimum (0.0427) at epoch 526, then drifts up. Restoring the epoch-526 weights is exactly what early stopping does. Note the noisy early validation values (0.18 at epoch 1, 0.27 at epoch 10): with 10 validation points, single epochs are unreliable, so watch the trend.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras hides the inner loop inside model.fit: batch_size and epochs are arguments, validation_data runs a validation pass after each epoch, and the EarlyStopping callback rolls back to the best checkpoint.",
              },
              {
                t: "code",
                lang: "python",
                title: "train_loop_keras.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
tf.random.set_seed(0)

N, NOISE_STD = 30, 0.25
x = rng.uniform(0, 1, (N, 1)).astype("float32")
y = (np.sin(2 * np.pi * x) + rng.normal(0, NOISE_STD, (N, 1))).astype("float32")
x_tr, y_tr, x_va, y_va = x[:20], y[:20], x[20:], y[20:]

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss="mse")

early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=200, restore_best_weights=True)

history = model.fit(
    x_tr, y_tr,
    batch_size=8,                 # 3 iterations per epoch (8, 8, 4)
    epochs=1500,
    validation_data=(x_va, y_va),
    callbacks=[early],
    verbose=0,
)

h = history.history
best = int(np.argmin(h["val_loss"]))
print("epochs run:", len(h["loss"]))
print(f"best epoch {best + 1}: train {h['loss'][best]:.4f}  val {h['val_loss'][best]:.4f}")`,
                output: "Prints the number of epochs actually run (training stops once validation loss has not improved for 200 epochs) and the training and validation MSE at the best epoch; the model's weights are restored to that epoch. Exact numbers depend on TensorFlow's initialization and are not reproduced here.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------
    variations: [
      { t: "h", text: "Variants of the batching scheme" },
      {
        t: "table",
        head: ["Scheme", "Batch size", "Advantages", "Disadvantages", "Use cases"],
        rows: [
          ["Batch (full) gradient descent", "$B = N$", "Exact gradient, repeatable, smooth loss curve", "One update per epoch; memory grows with $N$; no gradient noise to help escape bad regions", "Small datasets, classical convex problems"],
          ["Stochastic gradient descent", "$B = 1$", "Cheapest update; most gradient noise (mild regularizing effect)", "Very noisy; wastes parallel hardware", "Online learning, streaming data"],
          ["Mini-batch gradient descent", "$1 < B < N$ (32–512 typical)", "Good GPU use, moderate noise, many updates per epoch", "Batch size becomes a hyperparameter tied to learning rate", "Default for nearly all deep learning"],
          ["Large-batch training", "thousands+", "High throughput across many accelerators", "Needs learning-rate scaling and warmup; can generalize slightly worse if not tuned", "Pre-training large models"],
          ["Gradient accumulation", "effective $B = k \\cdot B_{\\text{micro}}$", "Fakes a large batch on limited memory", "$k$ times fewer updates per epoch; BatchNorm still sees micro-batches", "Fine-tuning large models on one GPU"],
        ],
      },
      { t: "h", text: "Variants of validation" },
      {
        t: "table",
        head: ["Method", "How", "Advantages", "Disadvantages"],
        rows: [
          ["Hold-out validation", "One fixed train/val split", "Cheap, simple", "Noisy estimate when the validation set is small"],
          ["k-fold cross-validation", "Split into k folds; train k times, validating on a different fold each time; average", "All data used for both; steadier estimate", "k times the compute; rarely affordable for large networks"],
          ["Time-based split", "Validate on data strictly later than the training data", "Realistic for forecasting; stops peeking into the future", "Changes over time make the gap larger"],
          ["Grouped split", "Keep all samples of one patient/user/document in the same split", "Stops leakage from near-duplicates", "Needs group information"],
        ],
      },
      { t: "h", text: "Two kinds of learning curve" },
      {
        t: "table",
        head: ["Curve", "x-axis", "What it diagnoses"],
        rows: [
          ["Loss vs epoch (training curve)", "Epoch or iteration", "How fast training converges, learning-rate problems, when overfitting starts (early stopping point)"],
          ["Error vs training-set size", "Number of training examples", "Whether more data will help (high variance) or not (high bias)"],
          ["Error vs capacity (validation curve)", "Degree, width, depth, regularization strength", "Which capacity to pick: the bottom of the U-shaped validation curve"],
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "Beyond the U-curve: double descent",
        text: "For hugely over-parameterized models (far more parameters than training points), test error has been seen to rise near the point where the model can just fit every training point, then fall again as capacity keeps growing. This is \"double descent\" (Belkin et al., 2019), and it is why huge networks can generalize well even though they could memorize their training sets. The classic U-curve is still the right picture for most small and medium models, and validation error is the judge either way.",
      },
    ],

    // ------------------------------------------------------------------
    applications: [
      {
        t: "list",
        items: [
          "**Choosing the number of epochs** — almost every supervised model (image classifiers, speech recognizers, recommenders) is trained with a validation curve and stopped or checkpointed at the best validation epoch.",
          "**Hyperparameter search** — grid/random search and Bayesian optimization (e.g. Optuna, Ray Tune) compare runs only by the validation metrics from the loop's evaluation phase.",
          "**Data-collection decisions** — in industry (medical imaging, defect detection), error-vs-dataset-size curves show whether labeling more data is worth the cost.",
          "**Fine-tuning pre-trained models** — fine-tuning a pre-trained language or vision model on a small labeled set overfits within a few epochs, so people train only a handful of epochs and watch validation loss closely.",
          "**Large-scale pre-training** — large language models often train for about one epoch or less. With that much data, overfitting is rarely the limit, and the training loss curve is the main check.",
          "**Monitoring in production** — a gap between validation scores at training time and live scores reveals distribution shift. It looks like overfitting, but the cause is the data changing.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    tradeoffs: [
      { t: "h", text: "Batch size" },
      {
        t: "table",
        head: ["", "Small batches", "Large batches"],
        rows: [
          ["Updates per epoch", "Many", "Few"],
          ["Gradient noise", "High (variance ∝ 1/B)", "Low"],
          ["Hardware efficiency", "Poor parallelism", "Good, until memory runs out"],
          ["Learning rate", "Needs a smaller η", "Tolerates a larger η (often scaled roughly with B, with warmup)"],
          ["Generalization", "Noise often acts as a mild regularizer", "May need extra tuning to match small-batch generalization"],
        ],
      },
      { t: "h", text: "Capacity" },
      {
        t: "list",
        items: [
          "**More capacity** lowers bias and captures complex patterns, but raises variance and compute, and needs more data or regularization.",
          "**Less capacity** is cheaper, steadier and easier to interpret, but error cannot go below a floor (bias).",
          "**More epochs** lower training error but, past the validation minimum, add overfitting; **fewer epochs** act as a regularizer (early stopping) but risk underfitting.",
          "**Validation set size**: bigger gives more reliable decisions but leaves less for training. With small datasets, k-fold cross-validation buys both with extra compute.",
        ],
      },
      {
        t: "callout",
        kind: "warning",
        title: "Limitations of the bias–variance picture",
        text: "The decomposition is exact for squared error. Classification losses (0–1 loss, cross-entropy) have similar but not identical versions. It describes an average over training sets, which you never see: in practice you have one training set and estimate everything from train/validation errors.",
      },
    ],

    // ------------------------------------------------------------------
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting to zero gradients** (PyTorch). Why wrong: .backward() adds to the existing .grad buffers, so each step uses the sum of all past gradients and training diverges or acts strangely. Fix: call optimizer.zero_grad() once per iteration, before backward (or set_to_none=True, the default in recent versions).",
          "**Evaluating in training mode or with gradients on.** Why wrong: dropout stays active and BatchNorm updates its running statistics with validation data, so numbers are noisy and evaluation changes the model; gradient tracking wastes memory. Fix: model.eval() and torch.no_grad() to evaluate, model.train() before the next epoch.",
          "**Tuning hyperparameters on the test set.** Why wrong: every choice made by looking at test scores leaks information, so the final test score is too optimistic. Fix: tune on validation (or cross-validation); use the test set once.",
          "**Not shuffling the training data**, or changing the validation split each run. Why wrong: sorted data gives similar batches that push the model back and forth; changing splits makes runs impossible to compare. Fix: shuffle=True for training each epoch; a fixed, seeded validation split.",
          "**Leakage between splits** — e.g. normalization statistics computed on the full dataset, or near-duplicate images (or the same patient) in train and validation. Why wrong: validation looks too good and overfitting goes unnoticed. Fix: fit preprocessing on training data only; split by group.",
          "**Mixing up epochs and iterations** in a learning-rate schedule or log. Why wrong: a schedule made for 100 epochs but stepped every iteration decays hundreds of times too fast. Fix: be clear whether scheduler.step() runs per batch or per epoch, and compute iterations = epochs × ⌈N/B⌉.",
          "**Reading one noisy validation value** as a trend. Why wrong: small validation sets jump around (see the PyTorch output, epochs 1 and 10). Fix: look over several epochs, use a bigger validation set, and use patience in early stopping.",
          "**Thinking low training loss means a good model.** Why wrong: training loss below the noise level signals memorization. Fix: always report validation (and finally test) metrics next to training ones.",
          "**Calling every high error overfitting.** Why wrong: if training error is also high, the problem is underfitting or a training bug, and regularization makes it worse. Fix: first compare training error to a baseline (e.g. predicting the mean) and to the expected noise or human level.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between an epoch, a batch and an iteration?", a: "A batch is a group of training examples processed together. An iteration is one parameter update from one batch (forward pass, loss, backward pass, optimizer step). An epoch is one full pass over the training set: ⌈N/B⌉ iterations for N examples and batch size B. For example, 1,050 examples with batch size 32 take 33 iterations per epoch, and the last batch holds 26." },
      { t: "qa", level: "beginner", q: "What is the difference between a parameter and a hyperparameter?", a: "Parameters (weights, biases) are learned by gradient descent from training data. Hyperparameters (learning rate, batch size, epochs, model size, regularization strength, optimizer) are set by you, not by the gradient; you pick them by comparing validation scores across runs." },
      { t: "qa", level: "beginner", q: "How do you recognize overfitting and underfitting?", a: "Underfitting: training and validation error are both high and close together — the model cannot even fit the training data. Overfitting: training error is low (often below the noise level) while validation error is much higher, and usually rising as training goes on or capacity grows. A good fit has low training error and validation error near its minimum, with a small gap." },
      { t: "qa", level: "intermediate", q: "Derive or state the bias–variance decomposition and explain each term.", a: "For y = f(x) + ε with zero-mean noise of variance σ², the expected squared error of a model trained on a random dataset D is E[(y − f̂_D(x))²] = (f(x) − E_D f̂_D(x))² + E_D[(f̂_D(x) − E_D f̂_D(x))²] + σ². Bias² is the error of the average model (a built-in mistake from a too-limited model family). Variance is how much the model changes between training sets (sensitivity to the sample). σ² is label noise no model can remove. Cross-terms vanish because the noise is independent and zero-mean, and the deviation from the average model averages to zero. More capacity usually lowers bias and raises variance." },
      { t: "qa", level: "intermediate", q: "Why do we need a validation set if we already have a test set?", a: "Every decision made by looking at a dataset — when to stop, which learning rate, which architecture — fits the model to that dataset a little. Made on the test set, those decisions would make the test score too optimistic, so it would no longer estimate performance on truly new data. The validation set soaks up that selection bias; the test set is used once at the end." },
      { t: "qa", level: "intermediate", q: "Your training loss is decreasing but your validation loss started increasing after epoch 20. What do you do?", a: "The model is overfitting after epoch 20. First, use the checkpoint with the best validation loss (early stopping). Then shrink the gap: more data or augmentation, weight decay or dropout, a smaller model, or a pre-trained model. Also check the rise is a real trend, not noise from a small validation set, and look for leakage or differences between the splits." },
      { t: "qa", level: "advanced", q: "Why is the mini-batch gradient an unbiased estimator of the full-batch gradient, and how does batch size affect its variance and the choice of learning rate?", a: "If a batch is a uniformly random subset (with or without replacement), every example is equally likely to appear, so the expected batch-average gradient equals the dataset-average gradient. Its covariance scales like Σ/B (times a finite-population correction (N−B)/(N−1) when sampling without replacement), where Σ is the per-example gradient covariance. So larger batches give less noisy steps and allow larger learning rates — a common rule scales η roughly linearly with B, plus warmup. But each epoch has fewer updates, and past a problem-dependent \"critical batch size\", raising B cuts the number of steps needed less and less." },
      { t: "qa", level: "advanced", q: "Modern networks have far more parameters than training examples and can fit random labels, yet they generalize. How does this square with the bias–variance trade-off?", a: "Parameter count is a poor measure of real capacity. What gets learned depends on the optimizer, initialization, architecture and training time. Gradient descent from a small initialization leans toward simple (e.g. low-norm) solutions, which acts as hidden (implicit) regularization. In experiments, test error vs model size can show double descent: a peak near the point where the model fits every training example, then a drop for larger models. The decomposition still holds for squared error; what changes is that variance can fall again for very large models. In practice, held-out validation error, not parameter count, decides whether a model overfits." },
    ],

    // ------------------------------------------------------------------
    practice: [
      {
        t: "quiz",
        questions: [
          {
            kind: "numeric",
            q: "A dataset has 60,000 training examples. With batch size 128 (keeping the partial last batch), how many parameter updates happen in 5 epochs?",
            answer: 2345,
            tolerance: 0,
            explain: "⌈60000/128⌉ = ⌈468.75⌉ = 469 iterations per epoch; 469 × 5 = 2,345.",
          },
          {
            kind: "numeric",
            q: "N = 50,000 examples, batch size 256, drop_last=True. How many iterations per epoch?",
            answer: 195,
            tolerance: 0,
            explain: "⌊50000/256⌋ = ⌊195.3⌋ = 195; the remaining 80 examples are skipped each epoch.",
          },
          {
            kind: "numeric",
            q: "At some input the model has bias² = 0.02 and variance = 0.05, and the label noise has standard deviation σ = 0.2. What is the expected squared test error?",
            answer: 0.11,
            tolerance: 0.001,
            explain: "bias² + variance + σ² = 0.02 + 0.05 + 0.04 = 0.11.",
          },
          {
            q: "Training error 0.01, validation error 0.40, noise variance about 0.05. What is the most likely diagnosis?",
            options: ["Underfitting", "Overfitting", "Learning rate too large", "A good fit"],
            answer: 1,
            explain: "Training error is far below the noise level and the gap to validation is large: the model is memorizing noise.",
          },
          {
            q: "Training and validation error are both 0.35 and nearly flat over many epochs; a simple baseline gets 0.40. Which change is most likely to help?",
            options: ["Add dropout", "Collect more data", "Increase model capacity or train with a better-tuned learning rate", "Stop training earlier"],
            answer: 2,
            explain: "Both errors are high and close: high bias, underfitting. Regularization, more data or earlier stopping fix variance, not bias.",
          },
          {
            q: "In a PyTorch training loop, what happens if you never call optimizer.zero_grad()?",
            options: [
              "Nothing — gradients are overwritten on each backward",
              "Gradients accumulate across iterations, so each update uses the sum of all previous gradients",
              "The optimizer raises an error",
              "Only the bias gradients accumulate",
            ],
            answer: 1,
            explain: ".backward() adds into the .grad buffers; without zeroing them, the step uses a growing sum of stale gradients.",
          },
          {
            q: "On an error-vs-training-set-size curve, training and validation error converge to the same high value. What does this indicate?",
            options: ["High variance; more data will help", "High bias; more data will not help much — increase capacity", "Data leakage", "The learning rate is too small"],
            answer: 1,
            explain: "Converged curves with no gap mean variance is already small; the remaining error is bias (plus noise).",
          },
          {
            kind: "numeric",
            q: "For linear least squares with p = 6 parameters, n = 24 training points and noise variance σ² = 0.08 (true function inside the model family), what is the expected training MSE?",
            answer: 0.06,
            tolerance: 0.0005,
            explain: "σ²(1 − p/n) = 0.08 × (1 − 6/24) = 0.08 × 0.75 = 0.06 — below the noise variance, which is why training error is optimistic.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Epoch and iteration arithmetic",
        prompt: "You train on N = 10,000 examples with batch size 64 for 20 epochs. (a) How many iterations per epoch and how large is the last batch? (b) How many updates in total? (c) Repeat (a) and (b) with drop_last=True and say how many examples are skipped per epoch. (d) With drop_last=True, how many epochs are needed for at least 5,000 updates?",
        hint: "Use ⌈N/B⌉, the last-batch formula N − (⌈N/B⌉ − 1)B, and ⌊N/B⌋.",
        solution: [
          {
            t: "list",
            ordered: true,
            items: [
              "(a) ⌈10000/64⌉ = ⌈156.25⌉ = 157 iterations; the last batch has 10000 − 156·64 = 16 examples.",
              "(b) 157 × 20 = 3,140 updates.",
              "(c) ⌊10000/64⌋ = 156 iterations per epoch, 156 × 20 = 3,120 updates; 10000 mod 64 = 16 examples skipped each epoch (different ones each time, thanks to shuffling).",
              "(d) 5000/156 = 32.05, so you need 33 epochs (32 give only 4,992 updates).",
            ],
          },
          {
            t: "code",
            lang: "python",
            title: "Check",
            code: String.raw`import math
N, B, E = 10000, 64, 20
i = math.ceil(N / B)
print(i, N - (i - 1) * B, i * E, N // B, N % B, (N // B) * E, math.ceil(5000 / (N // B)), 5000 / (N // B))`,
            output: "157 16 3140 156 16 3120 33 32.05128205128205",
          },
        ],
      },
      {
        t: "exercise",
        title: "Learning curves vs training-set size (coding)",
        prompt: "Using y = sin(2πx) + ε with σ = 0.25, fit polynomials of degree 1, 3 and 9 by least squares on training sets of size n ∈ {12, 20, 40, 80, 160, 320}. For each (degree, n), average training MSE over 200 random training sets and report the median validation MSE on a fixed set of 2,000 validation points. Explain the three curves in terms of bias and variance.",
        hint: "Rescale x to [−1, 1] before building np.vander features, and add a tiny ridge (1e-8) to the normal equations. Use the median, not the mean: a few unlucky small training sets make the degree-9 fit explode.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "learning_curve_size.py",
            code: String.raw`import numpy as np

rng = np.random.default_rng(1)
f = lambda x: np.sin(2 * np.pi * x)
SIGMA = 0.25
REPEATS = 200                                # average over many random training sets

x_val = rng.uniform(0, 1, 2000)              # large fixed validation set
y_val = f(x_val) + rng.normal(0, SIGMA, x_val.size)

def fit_predict(x_tr, y_tr, x_ev, degree, ridge=1e-8):
    V = lambda x: np.vander(2 * x - 1, degree + 1, increasing=True)
    A = V(x_tr).T @ V(x_tr) + ridge * np.eye(degree + 1)
    coef = np.linalg.solve(A, V(x_tr).T @ y_tr)
    return V(x_tr) @ coef, V(x_ev) @ coef

for degree in (1, 3, 9):
    print(f"degree {degree}:   n   train_MSE    val_MSE")
    for n in (12, 20, 40, 80, 160, 320):
        tr_err, va_err = [], []
        for _ in range(REPEATS):
            x = rng.uniform(0, 1, n)
            y = f(x) + rng.normal(0, SIGMA, n)
            p_tr, p_va = fit_predict(x, y, x_val, degree)
            tr_err.append(np.mean((p_tr - y) ** 2))
            va_err.append(np.mean((p_va - y_val) ** 2))
        print(f"          {n:5d} {np.mean(tr_err):10.4f} {np.median(va_err):10.4f}")
print("noise floor sigma^2 =", SIGMA ** 2)`,
            output: String.raw`degree 1:   n   train_MSE    val_MSE
             12     0.2135     0.2912
             20     0.2306     0.2773
             40     0.2402     0.2690
             80     0.2506     0.2620
            160     0.2571     0.2596
            320     0.2565     0.2581
degree 3:   n   train_MSE    val_MSE
             12     0.0438     0.0995
             20     0.0509     0.0792
             40     0.0596     0.0717
             80     0.0636     0.0685
            160     0.0650     0.0667
            320     0.0655     0.0658
degree 9:   n   train_MSE    val_MSE
             12     0.0130    90.7466
             20     0.0319     0.6793
             40     0.0462     0.0893
             80     0.0552     0.0700
            160     0.0590     0.0652
            320     0.0602     0.0632
noise floor sigma^2 = 0.0625`,
          },
          {
            t: "list",
            items: [
              "**Degree 1 (high bias):** the curves quickly meet at about 0.26, four times the noise floor. More data barely helps; only more capacity would.",
              "**Degree 3 (balanced):** a small gap that closes to about 0.066, close to σ² = 0.0625.",
              "**Degree 9 (high variance):** with 12 points the median validation MSE is about 90 (the polynomial explodes between points) while training MSE is 0.013. The gap closes as n grows, and by n = 320 degree 9 slightly beats degree 3 (0.0632 vs 0.0658). With enough data, the bigger model wins.",
              String.raw`Training MSE rises with n toward σ², matching $\sigma^2(1-p/n)$: for n = 20, degree 3 (p = 4) predicts 0.05 (measured 0.0509) and degree 9 (p = 10) predicts 0.03125 (measured 0.0319).`,
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------
    summary: [
      {
        t: "list",
        items: [
          "**Training loop**: each epoch, shuffle; for each batch: zero gradients → forward → loss → backward → update. Then evaluate on validation data in eval mode, without gradients.",
          String.raw`**Arithmetic**: iterations per epoch $=\lceil N/B\rceil$; last batch $= N - (\lceil N/B\rceil - 1)B$; total updates $= E\lceil N/B\rceil$; with drop_last $\lfloor N/B\rfloor$ and $N \bmod B$ skipped per epoch.`,
          "**Parameters** are learned by gradients; **hyperparameters** are chosen by validation performance.",
          "**Train / validation / test**: train for gradients, validation for decisions, test once for the final report.",
          "**Underfitting** = high bias: both errors high, small gap. **Overfitting** = high variance: training error low (below noise), large and growing gap.",
          String.raw`**Bias–variance**: $\mathbb E[(y-\hat f)^2] = \text{bias}^2 + \text{variance} + \sigma^2$ — verified by Monte-Carlo; capacity trades bias for variance, giving a U-shaped validation curve.`,
          String.raw`**Training error is optimistic**: for least squares, $\mathbb E[\text{train MSE}] = \sigma^2(1 - p/n)$.`,
          "**Learning curves**: loss vs epoch shows when to stop and whether training works; error vs data size shows whether more data will help.",
          "**Remedies**: underfitting → more capacity, longer training, better learning rate, less regularization; overfitting → more data, augmentation, early stopping, weight decay, dropout, smaller model.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    next: [
      {
        t: "list",
        items: [
          "[Gradient Descent: Batch, Stochastic, Mini-Batch](/learn/m06/gradient-descent) — the loop's update step in depth: how learning rate and batch size control convergence, with a simulator for divergence and oscillation.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — the toolbox for the overfitting fixes listed here, and what each does to the weights and the loss.",
          "[Regression Losses: MSE, MAE, Huber, Log-Cosh](/learn/m05/regression-losses) — the loss computed in every iteration, and how its choice changes what the model learns.",
        ],
      },
    ],
  },
};

export default lesson;
