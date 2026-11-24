import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m03/training-basics",
  title: "Training: Epochs, Batches and Overfitting",
  summary:
    "What one pass of a training loop actually does, how epochs, batches and iterations relate arithmetically, and how to diagnose and fix underfitting and overfitting using the bias–variance trade-off and learning curves.",
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
        text: "A neural network starts with random parameters and is useless. **Training** is the procedure that turns it into a useful function: show it examples, measure how wrong it is, compute in which direction each parameter should move to be less wrong, move a little, and repeat — tens of thousands to millions of times. The code that does this is the **training loop**, and almost every deep learning program, from a 10-line regression demo to a large language model, has the same five-step loop at its core.",
      },
      {
        t: "p",
        text: "This lesson covers two things that every practitioner needs on day one. First, the **mechanics and vocabulary** of training: what happens inside one iteration, and how **epochs**, **batches** and **iterations** relate (with the exact arithmetic, including the awkward last batch). Second, the central question of machine learning: **does the model generalize?** A model that only memorizes its training data is worthless on new data. We will define **underfitting** and **overfitting**, explain them with the **bias–variance trade-off** and **model capacity**, and learn to read **learning curves** — the plots you will look at every time you train anything.",
      },
      {
        t: "p",
        text: "Why was this framework introduced? Because minimizing training error is easy and is not the goal. Early statistical learning theory (and long before it, polynomial curve fitting) showed that a sufficiently flexible model can drive training error to zero while making terrible predictions. The tools in this lesson — held-out validation data, learning curves, capacity control — are how the field turned \"fit the data\" into the well-posed goal \"perform well on data we have not seen\".",
      },
      {
        t: "callout",
        kind: "key",
        title: "Where this is used",
        text: "Everywhere. Choosing the number of epochs, the batch size, when to stop training, whether to collect more data, whether to make the network bigger or add regularization — every one of these daily decisions is made by reading train and validation curves as described here.",
      },
    ],

    // ------------------------------------------------------------------
    prerequisites: [
      {
        t: "list",
        items: [
          "[The Artificial Neuron](/learn/m03/artificial-neuron) — you need the idea of a model with **weights and a bias** that produces a prediction from an input, and the notion that learning means changing those weights.",
          "[Multi-Layer Networks and Model Capacity](/learn/m03/multilayer-networks) — helpful for understanding that bigger networks can represent more functions; a light reading is enough, this lesson re-explains capacity.",
          "[Gradient Descent](/learn/m06/gradient-descent) — the update rule $w \\leftarrow w - \\eta\\,\\partial L/\\partial w$. A light reading suffices: here we only need the idea that the gradient points uphill and we step the other way; the details of optimizers come later.",
          "[Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — to follow the one worked gradient computation in the Mathematical Explanation (derivative of a squared error).",
          "[Probability](/learn/m02/probability) — expectation and variance, used in the bias–variance decomposition. You need $\\mathbb{E}[X]$, $\\mathrm{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}X)^2]$ and the fact that independent zero-mean noise has no cross-term with anything else.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A student preparing for an exam",
        text: "Training data is the set of practice problems; the validation set is a mock exam with new problems; the test set is the real exam. One **epoch** is working through the whole practice book once. A **batch** is the handful of problems you do before checking the answers and adjusting your understanding; each such adjustment is an **iteration**. A student who has not studied enough gets both the practice problems and the mock exam wrong — **underfitting**. A student who memorized the answer key of the practice book scores 100% on it but fails the mock exam — **overfitting**. What we want is the student who learned the underlying method.",
      },
      {
        t: "p",
        text: "Think about fitting a curve through noisy dots. A straight line through dots that clearly follow a wave misses the pattern: it is too simple, so it is wrong on old and new points alike. A wildly wiggling curve that passes exactly through every dot has also captured the noise — the random jitter of each particular dot — and between and beyond the dots it swings to absurd values. The sensible curve is in between: flexible enough to follow the wave, too stiff to chase each individual dot.",
      },
      {
        t: "p",
        text: "The key insight is that **training error alone cannot tell these cases apart in the right direction**: the wiggly curve has the lowest training error of all. That is why we always keep some data aside (the **validation set**) that the model never trains on, and judge models by their error there. Training error measures how well the model fits; validation error estimates how well it **generalizes**.",
      },
      {
        t: "p",
        text: "Why split data into batches at all? Computing the exact gradient over a million examples before taking a single step is slow and wasteful: the gradient from a random sample of 64 examples already points roughly the right way. So we take many cheap, slightly noisy steps instead of a few expensive exact ones. One pass over all the data, in batches, is an epoch; we usually need many epochs.",
      },
    ],

    // ------------------------------------------------------------------
    mechanism: [
      { t: "h", text: "Anatomy of one training iteration" },
      {
        t: "steps",
        items: [
          { title: "Sample a mini-batch", text: "Take the next $B$ examples $(x_i, y_i)$ from a shuffled ordering of the training set. $B$ is the **batch size**." },
          { title: "Zero the gradients", text: "Clear gradient buffers left over from the previous iteration. Frameworks such as PyTorch **accumulate** gradients by default, so skipping this adds the old gradient to the new one." },
          { title: "Forward pass", tex: String.raw`\hat{y}_i = f(x_i;\,\theta)`, text: "Run the batch through the model with current parameters $\\theta$ to get predictions." },
          { title: "Compute the loss", tex: String.raw`L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell(\hat{y}_i, y_i)`, text: "A single number measuring how wrong the predictions are on this batch (e.g. mean squared error)." },
          { title: "Backward pass", tex: String.raw`g = \nabla_\theta L`, text: "Backpropagation computes the gradient of the loss with respect to every parameter." },
          { title: "Update the parameters", tex: String.raw`\theta \leftarrow \theta - \eta\, g`, text: "The optimizer moves each parameter a small step against its gradient. $\\eta$ is the **learning rate**. Adam and other optimizers change how the step is computed, not the structure of the loop." },
          { title: "Repeat until the epoch ends, then evaluate", text: "After all batches have been used once (one epoch), switch the model to evaluation mode, compute training and validation loss **without** updating anything, log them, and optionally save a checkpoint if validation improved." },
        ],
      },
      {
        t: "p",
        text: "Wrapped around this is an outer loop over epochs, and at the start of each epoch the training set is **reshuffled**, so batches differ from epoch to epoch. Shuffling matters: if data is sorted (e.g. all cats, then all dogs), consecutive gradients would all push the model the same way and training oscillates.",
      },
      { t: "h", text: "Epochs, batches and iterations" },
      {
        t: "table",
        head: ["Term", "Definition", "Typical values"],
        rows: [
          ["Batch (mini-batch)", "A subset of $B$ training examples processed together to compute one gradient estimate.", "32–512 for vision; thousands of sequences for large language models"],
          ["Iteration (step)", "One parameter update = one forward + backward pass on one batch.", "Thousands to millions over a run"],
          ["Epoch", "One full pass over every training example once.", "1 (huge datasets) to hundreds (small datasets)"],
          ["Batch gradient descent", "$B = N$: one iteration per epoch, exact gradient.", "Small datasets, convex problems"],
          ["Stochastic gradient descent (pure)", "$B = 1$: $N$ iterations per epoch, very noisy gradient.", "Mostly of historical/theoretical interest"],
        ],
      },
      {
        t: "p",
        text: String.raw`With $N$ training examples and batch size $B$, the number of iterations per epoch is $\lceil N/B \rceil$ (the ceiling: round up), because the leftover examples form a smaller final batch of size $N - (\lceil N/B\rceil - 1)B$. Setting **drop_last** (PyTorch's DataLoader argument of that name) discards that partial batch instead, giving $\lfloor N/B \rfloor$ iterations and skipping $N \bmod B$ examples per epoch — which ones are skipped changes every epoch because of shuffling, so no example is permanently lost. Total iterations for $E$ epochs is $E \cdot \lceil N/B \rceil$.`,
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
        text: "**Parameters** are the numbers the training loop learns: weights and biases. **Hyperparameters** are everything you choose before or around training that the gradient does not set: learning rate, batch size, number of epochs, number of layers and units, activation function, optimizer and its settings, regularization strength, dropout rate, data augmentation. Hyperparameters are tuned by training several models and comparing their **validation** performance — never their test performance, otherwise the test set is no longer an unbiased estimate.",
      },
      { t: "h", text: "Three data splits" },
      {
        t: "list",
        items: [
          "**Training set** — used to compute gradients and update parameters.",
          "**Validation set** (dev set) — never used for gradients; used to choose hyperparameters, pick the best epoch (early stopping) and detect overfitting.",
          "**Test set** — touched once at the end to report an unbiased estimate of performance on new data. Common splits are 80/10/10 or 70/15/15; with millions of examples a few percent each for validation and test is enough.",
        ],
      },
      { t: "h", text: "Underfitting, overfitting and capacity" },
      {
        t: "p",
        text: "**Model capacity** is, informally, how large and flexible the family of functions a model can represent is. For polynomials it grows with the degree; for neural networks with the number of parameters, depth, and also with how long you train (a network trained for a few steps can only reach functions close to its initialization). Capacity, relative to the amount of data, determines which of three regimes you land in:",
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
        text: "Two kinds of plots are called learning curves. (1) **Loss vs epoch** for training and validation: training loss normally falls monotonically (with batch noise). If validation loss falls with it, keep training. If validation loss bottoms out and starts rising while training loss keeps falling, the model has begun to overfit; the epoch at the minimum is where **early stopping** would stop. If both curves are flat and high, the model underfits — train longer, raise the learning rate, or add capacity. If training loss itself goes up or explodes, the learning rate is too large (an optimization problem, not a generalization problem).",
      },
      {
        t: "p",
        text: "(2) **Error vs training-set size** for a fixed model: training error **rises** with more data (it is harder to fit more points), validation error **falls**, and both converge toward a plateau. If they converge at a high value, the model is biased — more data will not help, more capacity will. If there is a large gap that is still shrinking, the model is variance-limited — more data (or regularization) will help.",
      },
      { t: "h", text: "Remedies" },
      {
        t: "table",
        head: ["Symptom", "Remedies (roughly in order of what to try)"],
        rows: [
          ["Underfitting (both errors high)", "Train longer; tune the learning rate; increase capacity (wider/deeper network, better architecture); reduce regularization; engineer better input features; check for bugs (labels shuffled, wrong normalization)."],
          ["Overfitting (large gap)", "Get more data; data augmentation; early stopping; L2 regularization / weight decay; dropout; smaller model; ensembling; transfer learning from a pre-trained model."],
          ["Validation noisy/unreliable", "Larger validation set; k-fold cross-validation for small datasets; average over several random seeds."],
          ["Training loss diverges or oscillates", "Lower the learning rate; gradient clipping; normalize inputs; check the loss implementation."],
        ],
      },
    ],

    // ------------------------------------------------------------------
    math: [
      { t: "h", text: "Empirical risk and true risk" },
      {
        t: "p",
        text: String.raw`Let the data be drawn from an unknown distribution $p(x,y)$, let $f(x;\theta)$ be the model with parameters $\theta$ and $\ell$ a per-example loss. What we want to minimize is the **true risk** (expected loss on new data); what we can compute is the **empirical risk** on the $N$ training examples:`,
      },
      {
        t: "math",
        tex: String.raw`R(\theta) = \mathbb{E}_{(x,y)\sim p}\big[\ell(f(x;\theta), y)\big], \qquad \hat{R}_{\text{train}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i;\theta), y_i)`,
      },
      {
        t: "p",
        text: String.raw`Training minimizes $\hat R_{\text{train}}$. The **generalization gap** is $R(\theta) - \hat R_{\text{train}}(\theta)$. Because $\theta$ was chosen using the training points, $\hat R_{\text{train}}$ is an optimistically biased estimate of $R$; the error on an independent validation set is an unbiased estimate of $R$ for a fixed $\theta$.`,
      },
      { t: "h", text: "Mini-batch gradient and the update" },
      {
        t: "p",
        text: String.raw`For a mini-batch $\mathcal{B}$ of size $B$ the loss and its gradient are $L_{\mathcal B}(\theta) = \frac1B\sum_{i\in\mathcal B}\ell_i$ and $g_{\mathcal B} = \frac1B\sum_{i\in\mathcal B}\nabla_\theta \ell_i$. If $\mathcal B$ is a uniformly random subset, $\mathbb E[g_{\mathcal B}] = \nabla_\theta \hat R_{\text{train}}$ — the batch gradient is an **unbiased** estimate of the full gradient, and its variance shrinks like $1/B$. The update is $\theta \leftarrow \theta - \eta\, g_{\mathcal B}$.`,
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
        text: "MNIST-sized example: $N = 60{,}000$, $B = 128$: $\\lceil 468.75 \\rceil = 469$ iterations per epoch, last batch $60000 - 468\\cdot128 = 96$; 5 epochs = 2,345 updates. With drop_last: 468 iterations per epoch.",
      },
      { t: "h", text: "The bias–variance decomposition" },
      {
        t: "p",
        text: String.raw`Assume $y = f(x) + \varepsilon$ with $\mathbb E[\varepsilon] = 0$, $\mathrm{Var}(\varepsilon) = \sigma^2$, noise independent of everything else. Train a model on a random training set $D$ to get $\hat f_D$. At a fixed input $x$, write $\bar f(x) = \mathbb E_D[\hat f_D(x)]$ for the average prediction over all possible training sets. The expected squared error on a **new** noisy label at $x$ is:`,
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
            text: String.raw`Write $y - \hat f_D = \varepsilon + (f - \bar f) + (\bar f - \hat f_D)$ (all at the same $x$). Square and take expectations. The cross-terms vanish: $\mathbb E[\varepsilon \cdot (\cdot)] = 0$ because $\varepsilon$ is zero-mean and independent of $D$; $\mathbb E_D[(f - \bar f)(\bar f - \hat f_D)] = (f - \bar f)\,\mathbb E_D[\bar f - \hat f_D] = (f-\bar f)\cdot 0$ because $f - \bar f$ is a constant and $\bar f$ is by definition the mean of $\hat f_D$. What remains is $\mathbb E[\varepsilon^2] + (f - \bar f)^2 + \mathbb E_D[(\bar f - \hat f_D)^2] = \sigma^2 + \text{bias}^2 + \text{variance}$. Averaging over $x$ gives the same identity for the expected test MSE.`,
          },
        ],
      },
      {
        t: "list",
        items: [
          "**Bias** — the error of the *average* model: systematic error from the model family being too restrictive (a line cannot follow a sine). High bias ↔ underfitting.",
          "**Variance** — how much the learned function changes when you train on a different sample of the same size. High variance ↔ overfitting.",
          String.raw`**Irreducible error** $\sigma^2$ — noise in the labels themselves; no model can go below it on average.`,
        ],
      },
      {
        t: "p",
        text: String.raw`**Numerical check.** Fit polynomials of degree $d$ to 25 points of $y = \sin(2\pi x) + \varepsilon$, $\sigma = 0.3$, repeated over 2,000 independent training sets, and evaluate on 50 fixed inputs. The sum bias² + variance + σ² matches a direct Monte-Carlo estimate of the test MSE to within sampling error, and the U-shape appears: bias falls with degree, variance rises.`,
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
        text: "Degree 3 minimizes the total: degrees 0–1 are dominated by bias (underfitting), degrees 7–8 by variance (overfitting) — with only 25 points, a degree-8 polynomial occasionally lands on a training set with a gap and swings wildly, and those rare blow-ups dominate the average.",
      },
      { t: "h", text: "Why training error is optimistic" },
      {
        t: "p",
        text: String.raw`For linear least squares with $p$ parameters, $n$ training points, Gaussian noise of variance $\sigma^2$ and a model family that contains the true function, the expected training MSE is exactly $\sigma^2(1 - p/n)$ — **below** the noise floor, because the fit uses $p$ degrees of freedom to absorb noise. With $\sigma^2 = 0.0625$ and $n = 20$: $p = 4$ (cubic) gives $0.05$ and $p = 10$ (degree 9) gives $0.03125$. The coding exercise below measures 0.0509 and 0.0319 for these cases. As $p \to n$ training error goes to zero while validation error explodes.`,
      },
    ],

    // ------------------------------------------------------------------
    visual: [
      {
        t: "p",
        text: String.raw`The simulator draws noisy samples $y = \sin(2\pi x) + \varepsilon$ with $\varepsilon \sim \mathcal N(0, \sigma^2)$ and $x$ uniform in $[0, 1]$ from a seeded random generator, and splits them 2/3 training (filled dots) and 1/3 validation (hollow dots). It fits a polynomial of degree $d$ by least squares on the training points only (normal equations with a tiny ridge of $10^{-8}$ for numerical stability, with $x$ rescaled to $[-1, 1]$). The left plot shows the data, the true function (dashed) and the fitted polynomial. The right plot shows training and validation MSE for every degree 0–15 on a logarithmic scale, with a marker at the current degree and the degree with the best validation error highlighted.`,
      },
      { t: "sim", id: "fitting", caption: "Polynomial fitting: underfitting, good fit and overfitting as the degree grows." },
      {
        t: "steps",
        items: [
          { title: "Press Play", text: "The degree animates from 0 to 15. Watch the narration change from **underfitting** (both errors high; the curve is a flat line or a gentle slope) through **good fit** (validation near its minimum — with the default sample, degrees 5–7; the exact range moves with every new sample) to **overfitting** (training error keeps falling, validation error rises, the curve wiggles between the dots)." },
          { title: "Read the right-hand plot as a capacity curve", text: "This is the classic U-shape: the training curve decreases monotonically with degree because each higher degree contains all lower ones; the validation curve is U-shaped. The best-validation marker is the degree you would choose." },
          { title: "Small vs large data", text: "Set degree 9 (then 12) and compare n = 10 with n = 200. With 10 points (about 6–7 for training) the degree-9 fit interpolates the training dots and swings off the chart; with 200 points the same degree-9 polynomial is smooth and close to the sine. More data reduces variance — overfitting is a property of capacity **relative to** data." },
          { title: "No noise vs heavy noise", text: "Set σ = 0: high degrees no longer hurt much, because there is no noise to memorize (the sine is smooth and well approximated by polynomials). Set σ = 0.5: every validation error rises (it can no longer go below about σ² = 0.25), and the high-degree fits swing far more wildly between the dots, because there is more noise to memorize. In expectation noisier data supports less capacity, but with only about 10 validation points the best-validation degree itself jumps around from sample to sample — press **New sample** a few times to see how unreliable a single small validation set is." },
          { title: "Resample", text: "Press **New sample** several times at a high degree with small n. The fitted curve changes dramatically from sample to sample — that is **variance**, seen directly. At degree 1 the line barely changes but is always wrong — that is **bias**." },
        ],
      },
      {
        t: "callout",
        kind: "tip",
        title: "What to notice",
        text: "Training MSE drops below the noise variance σ² once the degree is high: the model is fitting noise, which can never help on new data. Validation MSE can never go much below σ² on average, so a validation error near σ² means you are close to the best achievable.",
      },
    ],

    // ------------------------------------------------------------------
    scratch: [
      {
        t: "p",
        text: "A complete NumPy program with two parts. Part 1 is a real mini-batch training loop (shuffle → batch → forward → loss → gradient → update, with train/validation evaluation after each epoch) for a cubic polynomial model. Part 2 sweeps model capacity (polynomial degree 0–15) using the exact least-squares solution, showing underfitting, good fit and overfitting on the same data.",
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
          "**Iteration arithmetic**: 20 training examples with batch 8 give $\\lceil 20/8 \\rceil = 3$ iterations per epoch (batches of 8, 8 and 4), so 300 epochs = 900 updates — the iters column confirms it.",
          "**Part 1 learning curve**: both errors fall together; after 300 epochs the gradient-descent cubic (train 0.0644) approaches the exact least-squares cubic from Part 2 (train 0.0630). It is still slowly converging — not overfitting.",
          "**Part 2 capacity sweep**: degrees 0–2 underfit (both errors high), degrees 5–6 fit well (best validation 0.0419), and from degree 10 on the training error keeps shrinking (to 0.0100, far below the noise variance 0.0625) while validation error climbs back above 0.14 — overfitting.",
          "The validation curve is not perfectly smooth (degree 13 dips) because 10 validation points is a small, noisy estimate — a reason to use larger validation sets or cross-validation.",
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
                text: "The canonical PyTorch training loop with a DataLoader, explicit zero_grad → forward → loss → backward → step, a separate evaluation pass under torch.no_grad(), and a best-validation checkpoint (the mechanism behind early stopping). The network is deliberately large (4,353 parameters) for 20 training points so overfitting appears in the validation curve.",
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
                text: "Reading the curve: training MSE falls steadily to 0.0131 — well below the noise variance 0.0625 — while validation MSE reaches its minimum (0.0427) at epoch 526 and then drifts up. Restoring the epoch-526 weights is exactly what early stopping does. Note the noisy early validation values (0.18 at epoch 1, 0.27 at epoch 10): with 10 validation points, single-epoch values are unreliable, so always look at the trend.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras hides the inner loop inside model.fit: batch_size and epochs are arguments, validation_data triggers a validation pass after each epoch, and the EarlyStopping callback implements best-checkpoint rollback.",
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
          ["Batch (full) gradient descent", "$B = N$", "Exact gradient, deterministic, smooth loss curve", "One update per epoch; memory grows with $N$; no gradient noise to help escape poor regions", "Small datasets, classical convex problems"],
          ["Stochastic gradient descent", "$B = 1$", "Cheapest update; maximum gradient noise (mild regularizing effect)", "Very noisy, cannot use vectorized hardware efficiently", "Online learning, streaming data"],
          ["Mini-batch gradient descent", "$1 < B < N$ (32–512 typical)", "Good GPU utilization, moderate noise, many updates per epoch", "Batch size becomes a hyperparameter interacting with learning rate", "Default for nearly all deep learning"],
          ["Large-batch training", "thousands+", "High throughput across many accelerators", "Needs learning-rate scaling and warmup; can generalize slightly worse if not tuned", "Pre-training large models"],
          ["Gradient accumulation", "effective $B = k \\cdot B_{\\text{micro}}$", "Simulates a large batch on limited memory", "$k$ times fewer updates per epoch; BatchNorm still sees micro-batches", "Fine-tuning large models on one GPU"],
        ],
      },
      { t: "h", text: "Variants of validation" },
      {
        t: "table",
        head: ["Method", "How", "Advantages", "Disadvantages"],
        rows: [
          ["Hold-out validation", "One fixed train/val split", "Cheap, simple", "Noisy estimate when the validation set is small"],
          ["k-fold cross-validation", "Split into k folds; train k times, each time validating on a different fold; average", "Uses all data for both; lower-variance estimate", "k times the compute; rarely affordable for large networks"],
          ["Time-based split", "Validate on data strictly later than the training data", "Realistic for forecasting; prevents look-ahead leakage", "Distribution shift over time makes the gap larger"],
          ["Grouped split", "Keep all samples of a patient/user/document in the same split", "Prevents leakage from near-duplicates", "Requires group metadata"],
        ],
      },
      { t: "h", text: "Two kinds of learning curve" },
      {
        t: "table",
        head: ["Curve", "x-axis", "What it diagnoses"],
        rows: [
          ["Loss vs epoch (training curve)", "Epoch or iteration", "Convergence speed, learning-rate problems, when overfitting begins (early stopping point)"],
          ["Error vs training-set size", "Number of training examples", "Whether more data will help (high variance) or not (high bias)"],
          ["Error vs capacity (validation curve)", "Degree, width, depth, regularization strength", "Which capacity to choose: the minimum of the U-shaped validation curve"],
        ],
      },
      {
        t: "callout",
        kind: "note",
        title: "Beyond the U-curve: double descent",
        text: "For very over-parameterized models (far more parameters than training points), test error has been observed to rise near the point where the model can just interpolate the training data and then fall again as capacity grows further — called \"double descent\" (Belkin et al., 2019). This is why huge neural networks can generalize well despite being able to memorize their training sets. The classical U-curve remains the correct picture in the regime most small and medium models operate in, and validation error remains the arbiter either way.",
      },
    ],

    // ------------------------------------------------------------------
    applications: [
      {
        t: "list",
        items: [
          "**Choosing the number of epochs** — practically every supervised model (image classifiers, speech recognizers, recommender models) is trained with a validation curve and stopped or checkpointed at the best validation epoch.",
          "**Hyperparameter search** — tools such as grid/random search and Bayesian optimization (e.g. Optuna, Ray Tune) compare runs purely by validation metrics computed in the evaluation phase of the training loop described here.",
          "**Data-collection decisions** — in industrial projects (medical imaging, defect detection) error-vs-dataset-size curves are used to estimate whether labeling more data is worth the cost.",
          "**Fine-tuning pre-trained models** — fine-tuning a pre-trained language or vision model on a small labeled set overfits within a few epochs; practitioners typically train for only a handful of epochs and monitor validation loss closely.",
          "**Large-scale pre-training** — large language models are often trained for about one epoch or less over their data; with that much data overfitting is rarely the limiting factor and the training loss curve itself is the main diagnostic.",
          "**Monitoring in production** — the gap between validation performance at training time and live performance reveals distribution shift, which looks like overfitting but is caused by the data changing.",
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
          "**More capacity** lowers bias and allows the model to represent complex patterns, but raises variance and compute cost and needs more data or regularization.",
          "**Less capacity** is cheaper, more stable and more interpretable, but has a floor on how low error can go (bias).",
          "**More epochs** lower training error but past the validation minimum increase overfitting; **fewer epochs** act like a regularizer (early stopping) but risk underfitting.",
          "**Validation set size**: a bigger validation set gives more reliable decisions but leaves less data for training. With small datasets, k-fold cross-validation trades compute for both.",
        ],
      },
      {
        t: "callout",
        kind: "warning",
        title: "Limitations of the bias–variance picture",
        text: "The decomposition is exact for squared error. For classification losses (0–1 loss, cross-entropy) there are analogous but not identical decompositions. It describes the average over training sets, which you never observe directly — in practice you only see one training set and estimate everything from train/validation errors.",
      },
    ],

    // ------------------------------------------------------------------
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting to zero gradients** (PyTorch). Why wrong: .backward() adds to existing .grad buffers, so each step uses the sum of all past gradients and training diverges or behaves erratically. Fix: call optimizer.zero_grad() once per iteration, before backward (or set_to_none=True, the default in recent versions).",
          "**Evaluating in training mode or with gradients enabled.** Why wrong: dropout stays active and BatchNorm updates its running statistics with validation data, so validation numbers are noisy and the model is changed by evaluation; gradient tracking wastes memory. Fix: model.eval() and torch.no_grad() for evaluation, model.train() before the next epoch.",
          "**Tuning hyperparameters on the test set.** Why wrong: every choice made by looking at test performance leaks information, so the final test score is optimistically biased. Fix: tune on validation (or cross-validation), touch the test set once.",
          "**Not shuffling the training data**, or shuffling the validation split differently each run. Why wrong: sorted data gives correlated batches that push the model back and forth; changing validation splits makes runs incomparable. Fix: shuffle=True for training each epoch; a fixed, seeded validation split.",
          "**Data leakage between splits** — e.g. computing normalization statistics on the full dataset, or having near-duplicate images (or the same patient) in train and validation. Why wrong: validation becomes optimistic and overfitting goes undetected. Fix: fit preprocessing on training data only; split by group.",
          "**Confusing epochs and iterations** in a learning-rate schedule or log. Why wrong: a schedule designed for 100 epochs that is stepped every iteration decays hundreds of times too fast. Fix: be explicit about whether scheduler.step() is called per batch or per epoch, and compute iterations = epochs × ⌈N/B⌉.",
          "**Reading a single noisy validation value** as a trend. Why wrong: small validation sets fluctuate (see the PyTorch output, epochs 1 and 10). Fix: smooth or look over several epochs, use a larger validation set, and use patience in early stopping.",
          "**Believing low training loss means a good model.** Why wrong: training loss below the noise level is a sign of memorization. Fix: always report validation (and finally test) metrics next to training metrics.",
          "**Diagnosing every high-error situation as overfitting.** Why wrong: if training error is also high, the problem is underfitting or an optimization bug, and regularization will make it worse. Fix: compare training error to a baseline (e.g. predicting the mean) and to the expected noise/human level first.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between an epoch, a batch and an iteration?", a: "A batch is a subset of training examples processed together. An iteration is one parameter update computed from one batch (forward pass, loss, backward pass, optimizer step). An epoch is one full pass over the whole training set, which takes ⌈N/B⌉ iterations for N examples and batch size B. For example 1,050 examples with batch size 32 take 33 iterations per epoch, the last batch holding 26 examples." },
      { t: "qa", level: "beginner", q: "What is the difference between a parameter and a hyperparameter?", a: "Parameters (weights, biases) are learned by gradient descent from the training data. Hyperparameters (learning rate, batch size, number of epochs, architecture size, regularization strength, optimizer choice) are set by the practitioner and are not updated by the gradient; they are selected by comparing validation performance across training runs." },
      { t: "qa", level: "beginner", q: "How do you recognize overfitting and underfitting?", a: "Underfitting: training and validation error are both high and close to each other — the model cannot even fit the training data. Overfitting: training error is low (often below the noise level) while validation error is much higher and typically rising as training continues or capacity grows. A good fit has low training error and validation error near its minimum with a small gap." },
      { t: "qa", level: "intermediate", q: "Derive or state the bias–variance decomposition and explain each term.", a: "For y = f(x) + ε with zero-mean noise of variance σ², the expected squared error of a model trained on a random dataset D is E[(y − f̂_D(x))²] = (f(x) − E_D f̂_D(x))² + E_D[(f̂_D(x) − E_D f̂_D(x))²] + σ². Bias² is the error of the average model (systematic, from too restrictive a model family); variance is how much the model changes across training sets (sensitivity to the particular sample); σ² is irreducible label noise. Cross-terms vanish because the noise is independent and zero-mean and the deviation from the mean model has zero expectation. Increasing capacity usually lowers bias and raises variance." },
      { t: "qa", level: "intermediate", q: "Why do we need a validation set if we already have a test set?", a: "Every decision made by looking at a dataset — which epoch to stop at, which learning rate, which architecture — fits the model to that dataset a little. If those decisions were made on the test set, the test score would be optimistically biased and no longer estimate performance on truly new data. The validation set absorbs that selection bias; the test set is used once at the end." },
      { t: "qa", level: "intermediate", q: "Your training loss is decreasing but your validation loss started increasing after epoch 20. What do you do?", a: "The model is overfitting after epoch 20. Immediately: use the checkpoint with the best validation loss (early stopping). Then reduce the gap: add data or data augmentation, add weight decay or dropout, reduce model size, or use a pre-trained model. Also confirm the validation increase is a trend rather than noise from a small validation set, and check for leakage or distribution differences between the splits." },
      { t: "qa", level: "advanced", q: "Why is the mini-batch gradient an unbiased estimator of the full-batch gradient, and how does batch size affect its variance and the choice of learning rate?", a: "If a batch is a uniformly random subset (with or without replacement), each example appears with equal probability, so the expectation of the batch-average gradient equals the dataset-average gradient. Its covariance scales like Σ/B (times a finite-population correction (N−B)/(N−1) when sampling without replacement), where Σ is the per-example gradient covariance. Larger batches therefore give less noisy steps, which allows larger learning rates — the common heuristic scales η roughly linearly with B, combined with warmup — but each epoch has fewer updates, and beyond a problem-dependent \"critical batch size\" increasing B gives diminishing reductions in the number of steps needed." },
      { t: "qa", level: "advanced", q: "Modern networks have far more parameters than training examples and can fit random labels, yet they generalize. How does this square with the bias–variance trade-off?", a: "Parameter count is a poor measure of effective capacity. The function actually learned depends on the optimizer, initialization, architecture and training duration — gradient descent from small initialization is biased toward simple (e.g. low-norm) solutions, which acts as implicit regularization. Empirically, test error versus model size can show double descent: a peak near the interpolation threshold, then a decrease for larger models. The decomposition itself still holds for squared error; what changes is that variance can decrease again for very large models. Practically, held-out validation error, not parameter count, is what decides whether a model overfits." },
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
            explain: "Training error is far below the noise level and the gap to validation is large — the model is memorizing noise.",
          },
          {
            q: "Training and validation error are both 0.35 and nearly flat over many epochs; a simple baseline gets 0.40. Which change is most likely to help?",
            options: ["Add dropout", "Collect more data", "Increase model capacity or train with a better-tuned learning rate", "Stop training earlier"],
            answer: 2,
            explain: "Both errors are high and close: high bias / underfitting. Regularization, more data or earlier stopping address variance, not bias.",
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
            explain: "σ²(1 − p/n) = 0.08 × (1 − 6/24) = 0.08 × 0.75 = 0.06 — lower than the noise variance, which shows why training error is optimistic.",
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
              "(c) ⌊10000/64⌋ = 156 iterations per epoch, 156 × 20 = 3,120 updates; 10000 mod 64 = 16 examples are skipped each epoch (different ones each epoch because of shuffling).",
              "(d) 5000/156 = 32.05, so 33 epochs are needed (32 epochs give only 4,992 updates).",
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
        prompt: "Using y = sin(2πx) + ε with σ = 0.25, fit polynomials of degree 1, 3 and 9 by least squares on training sets of size n ∈ {12, 20, 40, 80, 160, 320}. For each (degree, n), average the training MSE over 200 random training sets and report the median validation MSE on a fixed set of 2,000 validation points. Interpret the three curves in terms of bias and variance.",
        hint: "Rescale x to [−1, 1] before building np.vander features and add a tiny ridge (1e-8) to the normal equations. The median is more robust than the mean because a few unlucky small training sets make the degree-9 fit explode.",
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
              "**Degree 1 (high bias):** the curves converge quickly to about 0.26 — four times the noise floor. More data barely helps; only more capacity would.",
              "**Degree 3 (balanced):** small gap that closes to about 0.066, close to σ² = 0.0625.",
              "**Degree 9 (high variance):** with 12 points the median validation MSE is about 90 (the polynomial explodes between points) while training MSE is 0.013; the gap closes as n grows, and by n = 320 degree 9 is slightly better than degree 3 (0.0632 vs 0.0658). With enough data, the higher-capacity model wins.",
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
          "**Training loop**: for each epoch, shuffle; for each batch: zero gradients → forward → loss → backward → update. Then evaluate on validation data in eval mode without gradients.",
          String.raw`**Arithmetic**: iterations per epoch $=\lceil N/B\rceil$; last batch $= N - (\lceil N/B\rceil - 1)B$; total updates $= E\lceil N/B\rceil$; with drop_last $\lfloor N/B\rfloor$ and $N \bmod B$ skipped per epoch.`,
          "**Parameters** are learned by gradients; **hyperparameters** are chosen by validation performance.",
          "**Train / validation / test**: train for gradients, validation for decisions, test once for the final report.",
          "**Underfitting** = high bias: both errors high, small gap. **Overfitting** = high variance: training error low (below noise), large and growing gap.",
          String.raw`**Bias–variance**: $\mathbb E[(y-\hat f)^2] = \text{bias}^2 + \text{variance} + \sigma^2$ — verified by Monte-Carlo; capacity trades bias for variance, giving a U-shaped validation curve.`,
          String.raw`**Training error is optimistic**: for least squares, $\mathbb E[\text{train MSE}] = \sigma^2(1 - p/n)$.`,
          "**Learning curves**: loss vs epoch tells you when to stop and whether optimization works; error vs data size tells you whether more data will help.",
          "**Remedies**: underfitting → more capacity, longer training, better learning rate, less regularization; overfitting → more data, augmentation, early stopping, weight decay, dropout, smaller model.",
        ],
      },
    ],

    // ------------------------------------------------------------------
    next: [
      {
        t: "list",
        items: [
          "[Gradient Descent: Batch, Stochastic, Mini-Batch](/learn/m06/gradient-descent) — the update step of the loop in depth: how the learning rate and batch size control convergence, with the simulator for divergence and oscillation.",
          "[Regularization: L1, L2, Dropout, Early Stopping](/learn/m07/regularization) — the toolbox for the overfitting remedies listed here, with what each one does to the weights and the loss.",
          "[Regression Losses: MSE, MAE, Huber, Log-Cosh](/learn/m05/regression-losses) — the loss function computed in step 3 of every iteration, and how its choice changes what the model learns.",
        ],
      },
    ],
  },
};

export default lesson;
