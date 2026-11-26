import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m05/classification-losses",
  title: "Classification Losses: Cross-Entropy, Hinge and Focal Loss",
  summary:
    "Why classifiers are trained with cross-entropy: binary cross-entropy from the Bernoulli likelihood, softmax cross-entropy and its p − y gradient, sparse labels, numerically stable code, label smoothing, hinge loss, focal loss, and why squared error on probabilities learns slowly.",
  minutes: 65,
  level: "intermediate",
  prerequisites: ["m02/probability", "m02/mle-map", "m02/information-theory", "m03/linear-logistic-regression", "m03/activation-functions", "m04/backpropagation"],
  next: ["m05/advanced-losses", "m06/gradient-descent", "m07/regularization"],
  terms: ["loss", "cross-entropy", "logit", "sigmoid", "softmax", "mse"],
  sections: {
    /* ------------------------------------------------------------------ */
    intro: [
      {
        t: "p",
        text: String.raw`A **loss function** turns "how wrong is the network?" into one number we can differentiate. Backpropagation computes its gradient and the optimizer follows it, so the loss decides what the network learns, how fast, and what its outputs mean.`,
      },
      {
        t: "p",
        text: "For classification, the thing we really care about — accuracy — cannot be trained on directly. It only jumps when a prediction flips, so its gradient is zero almost everywhere and undefined at the jumps. So we train on a smooth stand-in, a **surrogate loss**, that rewards high probability (or a large margin) on the correct class. The usual choice is **cross-entropy**: the negative log-likelihood of the labels under the model's probabilities. Its gradient with respect to the network's raw outputs is strikingly simple: *prediction minus target*.",
      },
      {
        t: "p",
        text: "Other losses serve special needs. **Hinge loss** (from support vector machines) cares only about a margin, not probabilities. **Focal loss** (Lin et al., 2017) turns down easy examples so rare, hard ones drive learning in very imbalanced problems like object detection. **Label smoothing** softens the cross-entropy target to fight over-confidence. We derive each, implement it stably, and show why squared error on the probability trains poorly.",
      },
      {
        t: "callout",
        kind: "key",
        title: "Where these losses are used",
        text: "Binary cross-entropy: binary and multi-label classification (spam detection, medical findings, image tagging). Softmax cross-entropy: every multi-class classifier and every language model (next-token prediction). Focal loss: dense object detectors and other heavily imbalanced tasks. Hinge loss: SVMs and some margin-based models.",
      },
    ],

    /* ------------------------------------------------------------------ */
    prerequisites: [
      {
        t: "list",
        items: [
          "[Probability and distributions](/learn/m02/probability) — the Bernoulli and categorical distributions, and independent samples.",
          String.raw`[Maximum likelihood estimation](/learn/m02/mle-map) — we derive cross-entropy as a negative log-likelihood: maximise $\prod_i P(y_i \mid x_i)$ by minimising $-\sum_i \log P(y_i \mid x_i)$.`,
          String.raw`[Entropy, cross-entropy and KL divergence](/learn/m02/information-theory) — the information-theory view $H(q, p) = H(q) + \mathrm{KL}(q \,\|\, p)$.`,
          "[Linear and logistic regression](/learn/m03/linear-logistic-regression) — logistic regression is a one-neuron classifier trained with exactly the binary cross-entropy derived here.",
          String.raw`[Activation functions](/learn/m03/activation-functions) — the sigmoid $\sigma(z) = 1/(1+e^{-z})$ with $\sigma'(z) = \sigma(z)(1-\sigma(z))$, and softmax.`,
          "[Backpropagation](/learn/m04/backpropagation) — the loss gradient with respect to the logits, computed here, is the first signal backprop sends through the network.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A weather forecaster's report card",
        text: String.raw`A forecaster says "80% chance of rain" each evening. How do we score them? If we only checked whether rain was more likely than not (accuracy), saying 51% would score the same as saying 99%. Cross-entropy instead charges $-\log(\text{probability you gave to what actually happened})$. Saying 80% and seeing rain costs $-\log 0.8 = 0.22$; saying 80% and seeing sun costs $-\log 0.2 = 1.61$; saying 99% and seeing sun costs $-\log 0.01 = 4.6$. Confident mistakes hurt a lot, honest uncertainty a little, and the best long-run strategy is to report your true belief. That is why cross-entropy produces calibrated probabilities.`,
      },
      {
        t: "p",
        text: String.raw`A classifier ends in a layer that outputs one unbounded real number per class, called a **logit**. A logit is a score: bigger means "more likely". To turn logits into probabilities we squash them — with the sigmoid for a yes/no question, or the softmax across several classes. The loss then takes the probability given to the correct answer and charges its negative log.`,
      },
      {
        t: "p",
        text: "The gradient tells each logit how to move. For cross-entropy it is (predicted probability − target). If the model says 0.9 for the right class, the push is only −0.1; if it says 0.01, the push is −0.99. The worse the prediction, the stronger the correction — it never fades. Squared error on the probability lacks this: when the model is confidently wrong, the sigmoid is flat (saturated) and the gradient almost vanishes. The model learns slowest exactly when it most needs to learn.",
      },
    ],

    /* ------------------------------------------------------------------ */
    mechanism: [
      { t: "h", text: "From network output to loss value" },
      {
        t: "steps",
        items: [
          { title: "The last linear layer produces logits", text: String.raw`One real number per class: $z \in \mathbb{R}^K$ (or one $z \in \mathbb{R}$ for binary problems). No activation yet.` },
          { title: "Logits become probabilities", text: String.raw`Binary: $p = \sigma(z)$. Multi-class: $p_k = e^{z_k} / \sum_j e^{z_j}$ (softmax). Every $p_k$ is in $(0, 1)$ and they sum to 1.` },
          { title: "Pick the probability of the true class", text: "With label $c$, read off $p_c$ (binary: $p$ if $y = 1$, $1 - p$ if $y = 0$)." },
          { title: "Take the negative log", text: String.raw`Per-example loss $-\log p_c$: zero when $p_c = 1$, growing without limit as $p_c \to 0$.` },
          { title: "Average over the mini-batch", text: "Backprop differentiates the mean over the $B$ examples." },
          { title: "Backward pass starts with p − y", text: "The gradient with respect to the logits is $(p - y)/B$; backprop carries it into all earlier layers." },
        ],
      },
      {
        t: "callout",
        kind: "warning",
        title: "In practice steps 2–4 are fused",
        text: String.raw`Frameworks never compute the probability and then its log separately. They compute $\log p_c$ straight from the logits with the log-sum-exp trick, which is exact and cannot overflow or give $\log 0$. That is why the losses take logits as input (PyTorch's BCEWithLogitsLoss and CrossEntropyLoss, Keras's from_logits=True).`,
      },
      { t: "h", text: "Choosing the loss for the label structure" },
      {
        t: "table",
        head: ["Task", "Output layer", "Loss", "PyTorch"],
        rows: [
          ["binary (one yes/no)", "1 logit", "binary cross-entropy", "`BCEWithLogitsLoss`"],
          ["multi-label (several independent yes/no)", "$K$ logits, sigmoid each", "BCE summed/averaged over labels", "`BCEWithLogitsLoss`"],
          ["multi-class (exactly one of $K$)", "$K$ logits, softmax", "categorical cross-entropy", "`CrossEntropyLoss`"],
          ["multi-class, integer labels", "$K$ logits", "sparse categorical CE (same loss, labels as indices)", "`CrossEntropyLoss` with class indices"],
          ["severe class imbalance / dense detection", "logits", "focal loss", "write it yourself (a few lines)"],
          ["margin classifier (SVM style)", "raw scores", "hinge loss", "`MultiMarginLoss` / clamp"],
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    math: [
      { t: "h", text: "1. Logits, sigmoid and softmax" },
      {
        t: "math",
        tex: String.raw`\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big), \qquad \operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}`,
      },
      {
        t: "p",
        text: String.raw`$z$ is a logit (a real number), $K$ the number of classes. The name comes from the sigmoid's inverse, **logit function** $\operatorname{logit}(p) = \log\frac{p}{1-p}$ (log-odds): a logit of 0 means probability 0.5, a logit of $+2$ means odds $e^2 \approx 7.4 : 1$. Adding the same constant to every logit does not change the softmax, $\operatorname{softmax}(z + c\mathbf{1}) = \operatorname{softmax}(z)$; this makes the stable implementation possible.`,
      },

      { t: "h", text: "2. Binary cross-entropy from the Bernoulli likelihood" },
      {
        t: "p",
        text: String.raw`Treat the label $y \in \{0, 1\}$ of input $x$ as a Bernoulli (coin-flip) variable with success probability $p = \sigma(z)$, where $z = f(x; \theta)$ is the network's logit. One expression gives the probability of the observed label:`,
      },
      { t: "math", tex: String.raw`P(y \mid x; \theta) = p^{\,y}\,(1 - p)^{1 - y}` },
      {
        t: "p",
        text: String.raw`(For $y = 1$ this is $p$; for $y = 0$ it is $1 - p$.) If the $N$ examples are independent, the likelihood is the product $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$. Maximising it is the same as minimising the average negative log, which turns the product into a sum:`,
      },
      {
        t: "math",
        tex: String.raw`\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big], \qquad p_i = \sigma(z_i)`,
        caption: "Binary cross-entropy = mean negative log-likelihood of Bernoulli labels.",
      },
      { t: "h", text: "Its gradient with respect to the logit is p − y" },
      {
        t: "p",
        text: String.raw`For one example, $\ell = -[y\log p + (1-y)\log(1-p)]$. Differentiate with respect to $p$, then multiply by $dp/dz = p(1-p)$ (chain rule):`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)}, \qquad \frac{\partial \ell}{\partial z} = \frac{p - y}{p(1-p)}\cdot p(1-p) = \boxed{\,p - y\,}`,
      },
      {
        t: "p",
        text: String.raw`The sigmoid's derivative $p(1-p)$, which gets tiny when the sigmoid saturates, cancels exactly with the $1/(p(1-p))$ from the log. The gradient stays in $[-1, 1]$, is proportional to the error, and does not vanish when the model is confidently wrong. The same loss written in the logit is $\ell = y\,\operatorname{softplus}(-z) + (1-y)\operatorname{softplus}(z)$, with $\operatorname{softplus}(u) = \log(1 + e^u)$.`,
      },
      {
        t: "p",
        text: String.raw`**Worked example.** Logit $z = 1.5$: $p = \sigma(1.5) = 1/(1 + e^{-1.5}) = 1/(1 + 0.223130) = 0.817574$.`,
      },
      {
        t: "table",
        head: ["Case", String.raw`loss $\ell$`, String.raw`gradient $\partial\ell/\partial z = p - y$`, "reading"],
        rows: [
          ["$z = 1.5$, $y = 1$", String.raw`$-\log 0.817574 = 0.201413$`, "$0.817574 - 1 = -0.182426$", "right and fairly sure: small loss, gentle push up"],
          ["$z = 1.5$, $y = 0$", String.raw`$-\log 0.182426 = 1.701413$`, "$0.817574$", "wrong: big loss, strong push down"],
          ["$z = -2$, $y = 1$", String.raw`$-\log 0.119203 = 2.126928$`, "$-0.880797$", "confidently wrong: gradient near −1"],
          ["$z = 0$, any $y$", String.raw`$\log 2 = 0.693147$`, String.raw`$\pm 0.5$`, "maximum uncertainty"],
        ],
      },

      { t: "h", text: "3. Categorical (softmax) cross-entropy" },
      {
        t: "p",
        text: String.raw`With $K$ classes, encode the label as a one-hot vector $y \in \{0,1\}^K$ with $y_c = 1$ for the true class $c$. The model's probabilities are $p = \operatorname{softmax}(z)$. The negative log-likelihood is`,
      },
      {
        t: "math",
        tex: String.raw`\ell = -\sum_{k=1}^{K} y_k \log p_k = -\log p_c = -z_c + \log\sum_{j=1}^{K} e^{z_j}`,
      },
      {
        t: "p",
        text: String.raw`The last form (substitute the softmax and use $\log(e^{z_c}/S) = z_c - \log S$) reads: **log-sum-exp minus the correct logit**. Since $\partial/\partial z_k \log\sum_j e^{z_j} = e^{z_k}/\sum_j e^{z_j} = p_k$, the derivative is one line:`,
      },
      { t: "math", tex: String.raw`\frac{\partial \ell}{\partial z_k} = -[k = c] + p_k = \boxed{\,p_k - y_k\,}` },
      {
        t: "p",
        text: String.raw`Again prediction minus target. The entries sum to zero ($\sum_k p_k - \sum_k y_k = 1 - 1$): the loss only drops by raising the correct logit relative to the others, never by shifting all logits together. In information terms, $\ell$ is the cross-entropy $H(y, p) = H(y) + \mathrm{KL}(y \,\|\, p)$. For one-hot $y$, $H(y) = 0$, so minimising cross-entropy is minimising the KL divergence from the target to the prediction.`,
      },
      {
        t: "p",
        text: "**Worked example.** Logits $z = (2.0, 1.0, 0.1)$, true class $c = 0$ (the first).",
      },
      {
        t: "steps",
        items: [
          { title: "Exponentiate", tex: String.raw`e^{z} = (7.389056,\; 2.718282,\; 1.105171), \quad S = \textstyle\sum_j e^{z_j} = 11.212509` },
          { title: "Normalise", tex: String.raw`p = e^{z}/S = (0.659001,\; 0.242433,\; 0.098566)` },
          { title: "Loss", tex: String.raw`\ell = -\log 0.659001 = 0.417030 \;=\; \log S - z_0 = 2.417030 - 2` },
          { title: "Gradient", tex: String.raw`p - y = (-0.340999,\; 0.242433,\; 0.098566)` },
        ],
      },
      {
        t: "p",
        text: String.raw`If the true class were $c = 2$ (logit 0.1), the loss would be $\log S - 0.1 = 2.317030$. A model predicting the uniform distribution over $K$ classes has loss $\log K$ — $\log 10 = 2.302585$ for ten classes. Expect that value at initialisation; it is a handy sanity check.`,
      },

      { t: "h", text: "4. Sparse categorical cross-entropy" },
      {
        t: "p",
        text: String.raw`"Sparse" cross-entropy is the same loss with the label given as an integer index $c$ instead of a one-hot vector: $\ell = -\log p_c$. The maths is identical. The gain is memory: no $B \times K$ one-hot matrix, which matters for large $K$ (a 50,000-token vocabulary would need a 50,000-wide target row per token). Keras has both names (SparseCategoricalCrossentropy vs CategoricalCrossentropy); PyTorch's CrossEntropyLoss accepts integer indices or probability targets.`,
      },

      { t: "h", text: "5. Numerical stability" },
      {
        t: "p",
        text: String.raw`**Softmax overflow.** $e^{z}$ becomes infinity in float64 for $z > 709$ (float32: $z > 88$). With logits $(1000, 999, 998)$, the naive softmax computes $\infty/\infty$ = NaN. Since shifting all logits changes nothing, subtract $m = \max_j z_j$ first: the largest exponent becomes $e^0 = 1$ and nothing overflows. The same shift gives the **log-sum-exp** identity used for the loss:`,
      },
      { t: "math", tex: String.raw`\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \qquad m = \max_j z_j` },
      {
        t: "p",
        text: "For $(1000, 999, 998)$ this gives the correct probabilities $(0.665241, 0.244728, 0.090031)$ — the same as for $(2, 1, 0)$, as it must.",
      },
      {
        t: "p",
        text: String.raw`**Why not compute log(sigmoid(z))?** Two ways it fails. First, $\sigma(-800) = 1/(1 + e^{800})$: the exponential overflows and the result underflows to exactly 0, so $\log \sigma(-800) = -\infty$, though the true value is just $-800$. Second, for positive logits the sigmoid rounds to exactly 1: in float32, $\sigma(17)$ is already stored as $1.0$, so for a negative example $\log(1 - p) = \log 0 = -\infty$ and the gradient is NaN. The stable form never computes $p$:`,
      },
      {
        t: "math",
        tex: String.raw`-\log\sigma(z) = \operatorname{softplus}(-z), \qquad \operatorname{softplus}(u) = \max(u, 0) + \log\!\big(1 + e^{-|u|}\big)`,
      },
      {
        t: "p",
        text: String.raw`The exponent $-|u|$ is never positive, so nothing overflows, and log1p computes $\log(1 + \cdot)$ accurately. Combining both label cases gives PyTorch's BCEWithLogitsLoss formula: $\ell = \max(z, 0) - z\,y + \log(1 + e^{-|z|})$. Feeding sigmoid outputs into a probability-based BCE loses precision for good (PyTorch's BCELoss clamps the log at −100, which quietly caps the loss and kills the gradient).`,
      },

      { t: "h", text: "6. Label smoothing" },
      {
        t: "p",
        text: String.raw`With one-hot targets, $-\log p_c$ keeps falling as $p_c \to 1$, so the correct logit is pushed to grow without limit. The network is rewarded for ever more extreme confidence, which can hurt generalisation and calibration. **Label smoothing** (Szegedy et al., 2016, "Rethinking the Inception Architecture for Computer Vision") mixes the one-hot target with the uniform distribution:`,
      },
      {
        t: "math",
        tex: String.raw`q_k = (1 - \varepsilon)\, y_k + \frac{\varepsilon}{K}, \qquad \ell_{\text{LS}} = -\sum_k q_k \log p_k = (1-\varepsilon)\,(-\log p_c) + \varepsilon \cdot \frac{1}{K}\sum_k (-\log p_k), \qquad \frac{\partial \ell_{\text{LS}}}{\partial z_k} = p_k - q_k`,
      },
      {
        t: "p",
        text: String.raw`$\varepsilon$ (usually 0.1) is the smoothing strength. The optimum is now $p = q$, reached with **finite** logits: the gap between the correct logit and each other one settles at $\log(q_c/q_{\text{other}})$, e.g. $\log(0.91/0.01) = 4.51$ for $K = 10$, $\varepsilon = 0.1$. **Worked example** with the logits above, $\varepsilon = 0.1$, $K = 3$: $q = (0.933333, 0.033333, 0.033333)$, loss $0.513697$ (vs. 0.417030 unsmoothed), gradient $p - q = (-0.274332, 0.209100, 0.065233)$. The push on the correct class is gentler, and the model is no longer asked for probability 1.`,
      },

      { t: "h", text: "7. Hinge loss (SVM)" },
      {
        t: "p",
        text: String.raw`Hinge loss uses labels $t \in \{-1, +1\}$ and a raw score $s$ (a logit, not squashed). The **margin** is $t\,s$: positive when the sign is right, large when it is right with room to spare.`,
      },
      {
        t: "math",
        tex: String.raw`\ell_{\text{hinge}} = \max(0,\; 1 - t\,s), \qquad \frac{\partial \ell}{\partial s} = \begin{cases} -t & t\,s < 1 \\ 0 & t\,s > 1 \end{cases}`,
      },
      {
        t: "p",
        text: String.raw`(At $t\,s = 1$ there is a kink; any value in $[-t, 0]$ is a valid subgradient.) Examples with margin above 1 give **exactly zero** loss and gradient — only the "support vectors" near or past the boundary matter. With $t = +1$: $s = 2$ gives 0, $s = 0.5$ gives 0.5, $s = -1$ gives 2. Hinge loss gives no probabilities and is not a proper scoring rule: a score of 1.0 and a score of 100 are equally good. The multi-class version (Weston & Watkins) sums $\max(0, 1 - s_c + s_j)$ over wrong classes $j \ne c$; for scores $(1.2, 0.9, -0.3)$ with $c = 0$ this is $0.7 + 0 = 0.7$ (PyTorch's MultiMarginLoss also divides by $K$, giving 0.233333).`,
      },

      { t: "h", text: "8. Focal loss (Lin et al., 2017)" },
      {
        t: "p",
        text: String.raw`In dense object detection, one image gives tens of thousands of candidate boxes, almost all easy background. Each easy box has a small cross-entropy, but together they dominate the gradient. Lin et al. ("Focal Loss for Dense Object Detection", ICCV 2017, the RetinaNet paper) multiply cross-entropy by a factor that vanishes for well-classified examples. Let $p_t$ be the probability given to the true class ($p_t = p$ if $y = 1$, else $1 - p$):`,
      },
      {
        t: "math",
        tex: String.raw`\mathrm{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t`,
        caption: String.raw`$\gamma \ge 0$ — focusing parameter ($\gamma = 0$ gives back cross-entropy); $\alpha_t$ — optional class weight ($\alpha$ for positives, $1 - \alpha$ for negatives). The paper's best setting was $\gamma = 2$, $\alpha = 0.25$.`,
      },
      {
        t: "table",
        head: ["$p_t$", String.raw`$\gamma = 0$ (CE)`, String.raw`$\gamma = 1$`, String.raw`$\gamma = 2$`, String.raw`$\gamma = 5$`],
        rows: [
          ["0.9 (easy)", "0.105361", "0.010536", "0.001054", "0.000001"],
          ["0.6", "0.510826", "0.204330", "0.081732", "0.005231"],
          ["0.1 (hard)", "2.302585", "2.072327", "1.865094", "1.359653"],
        ],
      },
      {
        t: "p",
        text: String.raw`With $\gamma = 2$, an easy example ($p_t = 0.9$) is scaled by $(0.1)^2 = 0.01$ — 100 times smaller — while a hard one ($p_t = 0.1$) keeps $0.81$ of its loss. The hard-to-easy loss ratio grows from 22 under cross-entropy to about 1770. With $z_t = z$ for $y = 1$ and $z_t = -z$ for $y = 0$ (so $p_t = \sigma(z_t)$), the gradient (with $\alpha_t = 1$) is`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\,\mathrm{FL}}{\partial z_t} = \gamma\, p_t (1 - p_t)^{\gamma} \log p_t \;-\; (1 - p_t)^{\gamma + 1}, \qquad \frac{\partial\,\mathrm{FL}}{\partial z} = \pm \frac{\partial\,\mathrm{FL}}{\partial z_t}`,
      },
      {
        t: "p",
        text: String.raw`For $\gamma = 0$ this becomes $-(1 - p_t)$, i.e. $p - y$, as it should.`,
      },

      { t: "h", text: "9. Why not MSE on the sigmoid output?" },
      {
        t: "p",
        text: String.raw`Squared error on the probability, $\ell = (p - y)^2$ with $p = \sigma(z)$, looks reasonable. But its gradient with respect to the logit keeps the sigmoid derivative:`,
      },
      { t: "math", tex: String.raw`\frac{\partial \ell}{\partial z} = 2\,(p - y)\; p\,(1 - p)` },
      {
        t: "p",
        text: String.raw`When the model is confidently wrong — $y = 1$ but $p \approx 0$ — the factor $p(1-p)$ is almost zero, and so is the gradient. At $z = -5$: $p = 0.006693$, the MSE gradient is $2(0.006693 - 1)(0.006693)(0.993307) = -0.013207$, while cross-entropy's is $p - y = -0.993307$: **75 times stronger**. The MSE gradient can never exceed $8/27 \approx 0.296$ in size (reached at $p = 1/3$ for $y = 1$). Also, MSE through a sigmoid is non-convex in the logit, so it can have flat regions that trap gradient descent; cross-entropy is convex in $z$ (and in logistic regression's weights). In the from-scratch code, starting confidently wrong, BCE recovers in about 20 steps; MSE barely moves in 200.`,
      },

      { t: "h", text: "10. A short comparison with regression losses" },
      {
        t: "p",
        text: String.raw`Regression losses act on the residual $r = \hat y - y$ and are covered in [Regression Losses: MSE, MAE, Huber, Log-Cosh](/learn/m05/regression-losses). The key difference between them is how the gradient grows with the error, which controls outlier sensitivity:`,
      },
      {
        t: "math",
        tex: String.raw`\text{MSE: } r^2,\quad \text{MAE: } |r|,\quad \text{Huber}_\delta: \begin{cases}\tfrac12 r^2 & |r| \le \delta\\ \delta(|r| - \tfrac12\delta) & |r| > \delta\end{cases},\quad \text{log-cosh: } \log\cosh r`,
      },
      {
        t: "table",
        head: ["residual $r$", "MSE / grad $2r$", "MAE / grad", String.raw`Huber ($\delta = 1$) / grad`, String.raw`log-cosh / grad $\tanh r$`],
        rows: [
          ["0.5", "0.25 / 1", "0.5 / 1", "0.125 / 0.5", "0.120115 / 0.462117"],
          ["2", "4 / 4", "2 / 1", "1.5 / 1", "1.325003 / 0.964028"],
          ["10 (outlier)", "100 / 20", "10 / 1", "9.5 / 1", "9.306853 / 1.000000"],
        ],
      },
      {
        t: "p",
        text: "MSE's gradient grows with the residual, so one outlier with $r = 10$ pulls 20 times harder than a typical point — like cross-entropy's strong pull on wrong examples, but here usually unwanted. MAE, Huber and log-cosh cap the gradient. Huber's cap is to MSE what focal re-weighting is to cross-entropy: both decide which examples may dominate.",
      },
    ],

    /* ------------------------------------------------------------------ */
    visual: [
      { t: "sim", id: "loss-functions", caption: "Loss Function Explorer: each loss and its gradient as a function of the prediction." },
      {
        t: "list",
        items: [
          "**Classification tab, y = 1, logit axis.** Drag the marker to $z = -5$: binary CE's gradient is about −0.99, MSE-on-σ(z) about −0.013. Press Play: the dashed MSE gradient hugs zero on the far left (section 9).",
          String.raw`**Focal γ slider.** At γ = 0 the focal curve sits exactly on BCE. Raise γ to 2 and 5: the right side (correct, confident predictions) flattens to almost zero while the left side (wrong predictions) stays large. At $z = 2.2$ ($p_t \approx 0.9$) the focal value is about 1% of BCE at γ = 2.`,
          "**Hinge.** Its loss hits exactly zero at $z = 1$ (margin 1), where its gradient jumps from −1 to 0. Unlike BCE, it gives no reward for more confidence past the margin.",
          String.raw`**Switch y to 0.** Every curve mirrors around $z = 0$. Switch the x-axis to *probability p*: BCE shoots to infinity as $p \to 0$ for $y = 1$, while MSE-on-probability stays below 1 — it barely tells "slightly wrong" from "certainly wrong".`,
          String.raw`**Regression tab.** Move the residual to ±3: MSE's gradient is ±6, MAE's ±1, Huber's ±δ. Shrink δ toward 0.1 and Huber becomes MAE-like; raise it to 3 and Huber becomes $\tfrac12$·MSE over the whole plot. Log-cosh is the smooth middle ground with gradient $\tanh r$.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    scratch: [
      {
        t: "p",
        text: "Stable NumPy versions of BCE-with-logits, softmax cross-entropy (with optional label smoothing), hinge and focal loss, each returning the loss and its logit gradient, checked against finite differences. Then the naive formula breaks, and BCE races MSE on a logistic regression that starts confidently wrong.",
      },
      {
        t: "code",
        lang: "python",
        title: "classification_losses_from_scratch.py",
        code: String.raw`import numpy as np

# ---------------- numerically stable building blocks ----------------
def softplus(z):                      # log(1 + e^z) without overflow
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))

def sigmoid(z):                       # never evaluates exp of a large positive number
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def log_softmax(Z):                   # Z: (N, K) logits; subtract row max = log-sum-exp trick
    Zs = Z - Z.max(axis=1, keepdims=True)
    return Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))

# ---------------- losses: each returns (mean loss, dL/dlogits) ----------------
def bce_with_logits(z, y):
    # -[y log s(z) + (1-y) log(1-s(z))] = y*softplus(-z) + (1-y)*softplus(z)
    loss = y * softplus(-z) + (1 - y) * softplus(z)
    return loss.mean(), (sigmoid(z) - y) / len(z)

def cross_entropy(Z, labels, smoothing=0.0):
    N, K = Z.shape
    Q = np.full((N, K), smoothing / K)             # smoothed target distribution
    Q[np.arange(N), labels] += 1 - smoothing
    logP = log_softmax(Z)
    return -(Q * logP).sum(axis=1).mean(), (np.exp(logP) - Q) / N

def hinge(z, y):                      # y in {0,1} -> {-1,+1}; z = raw score
    s = 2 * y - 1
    margin = s * z
    return np.maximum(0, 1 - margin).mean(), np.where(margin < 1, -s, 0.0) / len(z)

def focal(z, y, gamma=2.0, alpha=0.25):
    s = 2 * y - 1
    zt = s * z                                   # logit of the true class
    pt = sigmoid(zt)
    a = np.where(y == 1, alpha, 1 - alpha)
    log_pt = -softplus(-zt)                       # stable log(p_t)
    loss = -a * (1 - pt) ** gamma * log_pt
    dz_t = a * (gamma * pt * (1 - pt) ** gamma * log_pt - (1 - pt) ** (gamma + 1))
    return loss.mean(), s * dz_t / len(z)

# ---------------- gradient check against central differences ----------------
def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

rng = np.random.default_rng(0)
z = rng.normal(size=6) * 3
y = rng.integers(0, 2, size=6).astype(float)
Z = rng.normal(size=(4, 3)) * 2
lab = rng.integers(0, 3, size=4)
checks = {
    "BCE":        (lambda v: bce_with_logits(v, y), z),
    "hinge":      (lambda v: hinge(v, y), z + 0.123),   # shift away from the kink at margin 1
    "focal":      (lambda v: focal(v, y), z),
    "CE":         (lambda v: cross_entropy(v, lab), Z),
    "CE+smooth":  (lambda v: cross_entropy(v, lab, 0.1), Z),
}
for name, (fn, x) in checks.items():
    analytic = fn(x)[1]
    numeric = num_grad(lambda v: fn(v)[0], x)
    rel = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic + numeric)
    print(f"{name:10s} loss={fn(x)[0]:.5f}  grad rel. error={rel:.1e}")

# ---------------- stability demo ----------------
with np.errstate(divide="ignore", over="ignore"):
    big = np.array([-800.0, 800.0])
    naive = -np.log(1 / (1 + np.exp(-big)))       # -log(sigmoid(z)) for y = 1
    print("\nnaive  -log(sigmoid(z)) at z=[-800, 800]:", naive)
    print("stable softplus(-z)                   :", softplus(-big))

# ---------------- why not MSE on probabilities? ----------------
# One-feature logistic regression started confidently WRONG (w = -4, so p ~ 0 for positives).
x = np.concatenate([rng.normal(2, 1, 100), rng.normal(-2, 1, 100)])
t = np.concatenate([np.ones(100), np.zeros(100)])
for loss_name in ["BCE", "MSE"]:
    w, b = -4.0, 0.0
    for step in range(1, 201):
        zz = w * x + b
        p = sigmoid(zz)
        if loss_name == "BCE":
            dz = (p - t) / len(t)
        else:                                       # d/dz of mean (p - t)^2
            dz = 2 * (p - t) * p * (1 - p) / len(t)
        w -= 0.5 * (dz * x).sum(); b -= 0.5 * dz.sum()
        if step in (1, 20, 50, 200):
            acc = ((sigmoid(w * x + b) > 0.5) == t).mean()
            print(f"{loss_name} step {step:3d}: w={w:+.3f}  accuracy={acc:.2f}")`,
        output: String.raw`BCE        loss=0.69978  grad rel. error=1.7e-10
hinge      loss=0.80324  grad rel. error=9.1e-11
focal      loss=0.07384  grad rel. error=9.2e-11
CE         loss=2.86837  grad rel. error=2.9e-10
CE+smooth  loss=2.74421  grad rel. error=2.4e-10

naive  -log(sigmoid(z)) at z=[-800, 800]: [inf -0.]
stable softplus(-z)                   : [800.   0.]
BCE step   1: w=-2.940  accuracy=0.02
BCE step  20: w=+1.664  accuracy=0.98
BCE step  50: w=+2.251  accuracy=0.98
BCE step 200: w=+3.113  accuracy=0.98
MSE step   1: w=-3.995  accuracy=0.02
MSE step  20: w=-3.892  accuracy=0.02
MSE step  50: w=-3.710  accuracy=0.02
MSE step 200: w=-0.773  accuracy=0.04`,
      },
      {
        t: "p",
        text: String.raw`All relative gradient errors are about $10^{-10}$, normal for central differences with step $10^{-6}$ in float64. The naive $-\log\sigma(z)$ returns infinity at $z = -800$; the stable softplus returns the exact 800. In the training race both models start at $w = -4$ (accuracy 2%). BCE's first step already moves $w$ from −4 to −2.94, and it reaches 98% accuracy within 20 steps. MSE's gradient is choked by the saturated sigmoid: after 200 steps $w$ is still negative and accuracy is 4%.`,
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
                t: "code",
                lang: "python",
                title: "torch_classification_losses.py",
                code: String.raw`import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- binary: pass LOGITS to BCEWithLogitsLoss, never sigmoid outputs ----
z = torch.tensor([1.5, -2.0, 40.0], requires_grad=True)
y = torch.tensor([1.0, 1.0, 0.0])
loss = nn.BCEWithLogitsLoss(reduction="sum")(z, y)
loss.backward()
print("BCEWithLogits per-example :", F.binary_cross_entropy_with_logits(z, y, reduction="none").detach())
print("dL/dz = sigmoid(z) - y    :", z.grad)

# Same thing through sigmoid + BCELoss: PyTorch clamps log() at -100, so the extreme case is wrong
p = torch.sigmoid(z.detach())
print("sigmoid then BCELoss      :", F.binary_cross_entropy(p, y, reduction="none"))

# ---- multi-class: CrossEntropyLoss = log_softmax + negative log-likelihood ----
Z = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
target = torch.tensor([0])                       # class INDEX, not one-hot ("sparse")
ce = nn.CrossEntropyLoss()(Z, target)
ce.backward()
print("\nCE loss                   :", round(ce.item(), 6))
print("dL/dZ = softmax(Z) - onehot:", Z.grad)
print("CE with label_smoothing=0.1:", round(nn.CrossEntropyLoss(label_smoothing=0.1)(Z, target).item(), 6))
# Probability (one-hot / soft) targets are also accepted:
print("CE with one-hot target     :", round(nn.CrossEntropyLoss()(Z, torch.tensor([[1.0, 0.0, 0.0]])).item(), 6))

# ---- hinge (binary, labels in {-1,+1}) and multi-class hinge ----
scores = torch.tensor([2.0, 0.5, -1.0]); s = torch.tensor([1.0, 1.0, 1.0])
print("\nbinary hinge              :", torch.clamp(1 - s * scores, min=0))
# multi-class hinge: sum over wrong classes j of max(0, 1 - s_y + s_j), divided by K
S = torch.tensor([[1.2, 0.9, -0.3]])
print("MultiMarginLoss           :", round(nn.MultiMarginLoss()(S, target).item(), 6))

# ---- focal loss (Lin et al. 2017), written out explicitly ----
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # = -log p_t
    p_t = torch.exp(-ce)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return a_t * (1 - p_t) ** gamma * ce

logits = torch.log(torch.tensor([0.9, 0.6, 0.1]) / (1 - torch.tensor([0.9, 0.6, 0.1])))  # p_t = .9,.6,.1
ones = torch.ones(3)
print("\nfocal, gamma=0, alpha=1 (= BCE):", focal_loss(logits, ones, 0.0, 1.0))
print("focal, gamma=2, alpha=1        :", focal_loss(logits, ones, 2.0, 1.0))`,
                output: String.raw`BCEWithLogits per-example : tensor([ 0.2014,  2.1269, 40.0000])
dL/dz = sigmoid(z) - y    : tensor([-0.1824, -0.8808,  1.0000])
sigmoid then BCELoss      : tensor([  0.2014,   2.1269, 100.0000])

CE loss                   : 0.41703
dL/dZ = softmax(Z) - onehot: tensor([[-0.3410,  0.2424,  0.0986]])
CE with label_smoothing=0.1: 0.513697
CE with one-hot target     : 0.41703

binary hinge              : tensor([0.0000, 0.5000, 2.0000])
MultiMarginLoss           : 0.233333

focal, gamma=0, alpha=1 (= BCE): tensor([0.1054, 0.5108, 2.3026])
focal, gamma=2, alpha=1        : tensor([1.0536e-03, 8.1732e-02, 1.8651e+00])`,
              },
              {
                t: "list",
                items: [
                  "The BCE values 0.2014 and 2.1269 and gradients −0.1824 and −0.8808 match the worked example; the CE loss 0.41703, its gradient and the smoothed loss 0.513697 match sections 3 and 6.",
                  "For $z = 40$, $y = 0$ the exact loss is 40.0. Sigmoid first, then BCELoss, returns 100 — PyTorch's clamp on the log, not the true loss.",
                  "CrossEntropyLoss applies log-softmax itself. Never put a softmax layer before it, or softmax gets applied twice.",
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
                title: "keras_classification_losses.py (not executed in CI)",
                code: String.raw`import tensorflow as tf

# Binary cross-entropy on logits (from_logits=True uses the stable softplus form)
z = tf.constant([[1.5], [-2.0], [40.0]])
y = tf.constant([[1.0], [1.0], [0.0]])
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
print("BCE per example:", bce(y, z).numpy())

# Categorical CE with one-hot labels, and sparse CE with integer labels
Z = tf.constant([[2.0, 1.0, 0.1]])
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("CE (one-hot):", cce([[1.0, 0.0, 0.0]], Z).numpy())
print("CE (sparse) :", scce([0], Z).numpy())

# Label smoothing is an argument of the one-hot version
cce_ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
print("CE, label_smoothing=0.1:", cce_ls([[1.0, 0.0, 0.0]], Z).numpy())

# Hinge: Keras converts 0/1 labels to -1/+1 internally
print("hinge:", tf.keras.losses.Hinge()([[1.0, 1.0, 1.0]], [[2.0, 0.5, -1.0]]).numpy())

# Focal loss (built in since TF 2.9)
focal = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True, reduction="none")
print("focal per example:", focal(y, z).numpy())`,
                output: "Prints the per-example BCE (about 0.2014, 2.1269, 40.0), the categorical and sparse cross-entropies (both about 0.4170), the label-smoothed loss (about 0.5137), the mean hinge loss of the three scores ((0 + 0.5 + 2)/3 ≈ 0.8333) and the per-example focal losses. Exact printed formatting depends on the TensorFlow version.",
              },
              {
                t: "list",
                items: [
                  "Keras losses default to from_logits=False: they expect probabilities. If the last layer has no activation, pass from_logits=True — forgetting this is one of the most common Keras bugs.",
                  "Keras's argument order is (y_true, y_pred); PyTorch's is (input, target).",
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
        head: ["Loss", "Formula (per example)", "Advantages", "Disadvantages", "Use cases"],
        rows: [
          ["Binary CE", String.raw`$-[y\log p + (1-y)\log(1-p)]$`, "gradient $p - y$; calibrated probabilities; convex in the logit", "sensitive to label noise (confidently wrong labels get large loss)", "binary and multi-label classification"],
          ["Categorical / sparse CE", String.raw`$-\log p_c$, $p = \mathrm{softmax}(z)$`, "same as BCE for $K$ classes; sparse form scales to huge $K$", "encourages unbounded logits; over-confidence", "multi-class classification, language models"],
          ["Weighted CE", String.raw`$-w_c \log p_c$`, "simple fix for class imbalance", "weights must be tuned; distorts calibration", "moderately imbalanced data"],
          ["Label-smoothed CE", String.raw`$-\sum_k q_k\log p_k$`, "better calibration, mild regularisation", "probabilities always less confident; can hurt distillation", "large image classifiers, Transformers"],
          ["Focal loss", String.raw`$-\alpha_t(1-p_t)^\gamma\log p_t$`, "focuses on hard examples; handles extreme imbalance", "two hyperparameters; less calibrated", "dense object detection, segmentation, rare-event detection"],
          ["Hinge", String.raw`$\max(0, 1 - t s)$`, "sparse gradients (only margin violators); max-margin", "no probabilities; not differentiable at the kink", "SVMs, some ranking and GAN objectives"],
          ["Squared hinge", String.raw`$\max(0, 1 - t s)^2$`, "differentiable everywhere", "more sensitive to outliers", "linear SVM variants"],
          ["MSE on probability", "$(p - y)^2$", "bounded, somewhat robust to label noise", "vanishing gradient when confidently wrong; non-convex", "rarely right for training classifiers"],
        ],
      },
      {
        t: "p",
        text: "More variants: class-balanced weighting by effective number of samples, the poly-loss family (adds polynomial terms in $1 - p_t$), and losses for metric learning (contrastive, triplet) and segmentation (Dice, IoU), covered in [Advanced Losses](/learn/m05/advanced-losses).",
      },
    ],

    /* ------------------------------------------------------------------ */
    applications: [
      {
        t: "list",
        items: [
          String.raw`**Image classification.** ImageNet classifiers use softmax cross-entropy over 1000 classes; label smoothing with $\varepsilon = 0.1$ came with Inception-v3 and is standard in modern training recipes.`,
          String.raw`**Language models.** Every autoregressive language model is trained with (sparse) cross-entropy on the next token over its whole vocabulary. Perplexity, the usual metric, is $\exp$ of the mean cross-entropy in nats.`,
          "**Object detection.** RetinaNet (Lin et al., 2017) introduced focal loss to train a one-stage detector over tens of thousands of anchors per image without hard-example mining; focal-style losses are now common in detection and medical segmentation.",
          "**Multi-label tagging.** Several independent tags per item (objects in a photo, topics of an article) use one sigmoid and one binary cross-entropy per tag.",
          String.raw`**Knowledge distillation.** A small student is trained with cross-entropy against the softened probabilities of a large teacher (Hinton, Vinyals & Dean, 2015, "Distilling the Knowledge in a Neural Network") — cross-entropy with soft targets.`,
          "**Support vector machines.** Linear SVMs (e.g. liblinear / scikit-learn's LinearSVC) minimise hinge loss (or its squared form, scikit-learn's default) plus L2 regularisation.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Cross-entropy — strengths:** principled (maximum likelihood), strictly proper (its expected value is smallest when you predict the true probabilities), gradient $p - y$ never vanishes for wrong predictions, convex in the logits, cheap and stable from logits.",
          "**Cross-entropy — limits:** unbounded loss on confidently wrong examples makes it sensitive to mislabelled data; it keeps pushing correct logits up, making networks over-confident (label smoothing or temperature scaling help); with heavy imbalance the many easy examples dominate.",
          String.raw`**Focal loss:** fixes easy-example domination, but adds $\gamma$ and $\alpha$, which interact (the paper found smaller $\alpha$ works better with larger $\gamma$), and its probabilities are not calibrated.`,
          "**Hinge loss:** sparse and margin-based, but no probabilities, and examples stop teaching once they clear the margin.",
          "**MSE on probabilities:** bounded, so less hurt by label noise, but the vanishing gradient when confidently wrong and non-convexity make it a poor default.",
          "**In practice:** accuracy, calibration and label-noise robustness pull different ways. The usual recipe: cross-entropy (maybe with label smoothing), plus class weights or focal loss only for severe imbalance.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    mistakes: [
      {
        t: "list",
        items: [
          String.raw`**Softmax before CrossEntropyLoss.** PyTorch's CrossEntropyLoss expects raw logits and applies log-softmax itself. Feeding it probabilities applies softmax twice: the loss stays near $\log K$ and gradients become tiny. Fix: remove the final softmax layer (apply softmax only at inference, if you need probabilities).`,
          String.raw`**Sigmoid + BCELoss instead of BCEWithLogitsLoss.** Unstable for large logits (saturated sigmoid, $\log 0$, clamping). Fix: output logits and use BCEWithLogitsLoss (Keras: from_logits=True).`,
          String.raw`**Forgetting from_logits=True in Keras.** If the last Dense layer has no activation but the loss expects probabilities, Keras clips the "probabilities" into $[\epsilon, 1-\epsilon]$ and trains on garbage. Fix: be explicit about what your model outputs.`,
          "**One-hot labels where indices are expected (or the reverse).** Keras's SparseCategoricalCrossentropy wants integer labels; CategoricalCrossentropy wants one-hot. Mixing them gives shape errors or quietly wrong losses. PyTorch's CrossEntropyLoss with integer targets needs dtype long.",
          "**Softmax cross-entropy for multi-label problems.** Softmax makes probabilities compete and sum to 1, so two tags can never both be near 1. Fix: independent sigmoids with BCE per label.",
          "**Hand-written naive softmax.** exp(z)/sum(exp(z)) overflows for logits above about 88 in float32. Fix: subtract the max, or use a log-softmax function.",
          "**Training a classifier with MSE.** Learning stalls on confidently wrong examples because of the $p(1-p)$ factor. Fix: use cross-entropy.",
          String.raw`**Initial loss far from log K.** A fresh $K$-class model should start near $\log K$. Much larger usually means badly scaled logits (initialisation or input normalisation); much smaller hints at label leakage. Check before long runs.`,
          String.raw`**Misreading focal α.** In the focal loss paper $\alpha = 0.25$ weights the *positive* (rare) class, which looks backwards; it works because $\gamma = 2$ already suppresses the easy negatives. Copying $\alpha$ without $\gamma$ (or the reverse) gives a different loss than intended.`,
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    interview: [
      { t: "qa", level: "beginner", q: "What is a logit?", a: String.raw`A logit is the raw, unbounded output of a classifier's last linear layer for a class, before sigmoid or softmax. For the sigmoid it is exactly the log-odds: $z = \log\frac{p}{1-p}$, so $z = 0$ means $p = 0.5$. BCEWithLogitsLoss and CrossEntropyLoss take logits because they can compute log-probabilities from them exactly and stably.` },
      { t: "qa", level: "beginner", q: "Why do we train classifiers with cross-entropy instead of directly maximising accuracy?", a: "Accuracy is piecewise constant in the parameters: small changes almost never flip a prediction, so its gradient is zero almost everywhere and gradient descent cannot use it. Cross-entropy is a smooth surrogate that rewards higher probability on the correct class, has useful gradients everywhere, and is the maximum-likelihood objective, so it also gives calibrated probabilities." },
      { t: "qa", level: "beginner", q: "When do you use binary cross-entropy and when categorical cross-entropy?", a: "Binary cross-entropy with a sigmoid: one yes/no output, or multi-label problems where each label is an independent yes/no. Categorical cross-entropy with a softmax: multi-class problems where exactly one of $K$ classes is correct. Sparse categorical cross-entropy is the same as categorical, with integer labels." },
      { t: "qa", level: "intermediate", q: "Derive the gradient of softmax cross-entropy with respect to the logits.", a: String.raw`Write the loss as $\ell = -z_c + \log\sum_j e^{z_j}$. The derivative of the log-sum-exp with respect to $z_k$ is $e^{z_k}/\sum_j e^{z_j} = p_k$; the derivative of $-z_c$ is $-1$ if $k = c$, else 0. So $\partial\ell/\partial z_k = p_k - y_k$ with $y$ the one-hot target. The components sum to zero, and each lies in $[-1, 1]$.` },
      { t: "qa", level: "intermediate", q: "Why does MSE with a sigmoid output train slowly when the model is confidently wrong?", a: String.raw`With $\ell = (p - y)^2$ and $p = \sigma(z)$, the chain rule gives $\partial\ell/\partial z = 2(p - y)p(1-p)$. When confidently wrong ($p \approx 0$ for $y = 1$), $p(1-p) \approx 0$, so the gradient vanishes — at $z = -5$ it is about $-0.013$, versus $-0.993$ for cross-entropy. Cross-entropy's $\log$ brings a $1/(p(1-p))$ factor that cancels the sigmoid derivative, leaving $p - y$.` },
      { t: "qa", level: "intermediate", q: "What does label smoothing do and why can it help?", a: String.raw`It replaces the one-hot target with $q = (1-\varepsilon)y + \varepsilon/K$. The optimum then has finite logits (the correct logit beats the others by $\log(q_c/q_{\text{other}})$) instead of infinite ones, so the network is not pushed toward extreme over-confidence. It acts as a regulariser and usually improves calibration, sometimes accuracy. The gradient becomes $p - q$. Downsides: probabilities are always softened, and it can hurt distillation by erasing class-similarity information.` },
      { t: "qa", level: "intermediate", q: "What is the difference between hinge loss and cross-entropy?", a: String.raw`Hinge loss $\max(0, 1 - t s)$ is zero once the margin $t s$ exceeds 1, so well-classified examples stop contributing; it outputs scores, not probabilities. Cross-entropy is never exactly zero, keeps rewarding more confidence in the correct class, has a smooth gradient $p - y$ and gives probabilities. Hinge gives max-margin solutions (SVMs); cross-entropy gives maximum-likelihood, calibrated classifiers.` },
      { t: "qa", level: "advanced", q: "Explain focal loss and the role of γ and α.", a: String.raw`Focal loss is $-\alpha_t(1-p_t)^\gamma\log p_t$. The factor $(1-p_t)^\gamma$ is near 0 for well-classified examples ($p_t \to 1$) and near 1 for misclassified ones, so with $\gamma = 2$ an example with $p_t = 0.9$ is down-weighted 100×. In dense detection, where most anchors are easy background, this stops them dominating. $\alpha_t$ is a class-balancing weight; in the paper $\alpha = 0.25$ for the positive class worked best with $\gamma = 2$, because $\gamma$ already suppresses easy negatives so much that positives need less up-weighting.` },
      { t: "qa", level: "advanced", q: "Why should a framework compute BCE from logits rather than from probabilities? Give the stable formula.", a: String.raw`Computing $p = \sigma(z)$ first loses information: for large positive $z$ the sigmoid rounds to exactly 1 (in float32 already at $z = 17$), so $\log(1-p) = -\infty$; for very negative $z$ the exponential overflows and $\log\sigma(z) = -\infty$, though the true value is about $z$. In the logit domain, $-\log\sigma(z) = \mathrm{softplus}(-z)$ and the full loss is $\max(z, 0) - zy + \log(1 + e^{-|z|})$, which never exponentiates a positive number and uses log1p for accuracy.` },
      { t: "qa", level: "advanced", q: "Cross-entropy is called a strictly proper scoring rule. What does that mean and why does it matter?", a: String.raw`A scoring rule is strictly proper if its expected value under the true label distribution is uniquely minimised by predicting that distribution. For binary labels with true probability $\pi$, the expected BCE $-\pi\log p - (1-\pi)\log(1-p)$ has derivative $(p - \pi)/(p(1-p))$, zero only at $p = \pi$. So cross-entropy pushes a model to output true conditional probabilities (calibration), which matters for decisions, thresholds and risk estimates. Hinge loss is not proper: any score with the right sign minimises it, whatever the probabilities.` },
    ],

    /* ------------------------------------------------------------------ */
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "What is the binary cross-entropy (natural log) when the logit is z = 0, whatever the label?", answer: 0.6931, tolerance: 0.001, explain: "σ(0) = 0.5, so the loss is −log 0.5 = log 2 ≈ 0.6931." },
          { kind: "numeric", q: "For z = 1.5 and y = 1, what is the gradient of BCE with respect to the logit?", answer: -0.1824, tolerance: 0.001, explain: "p = σ(1.5) = 0.8176, gradient p − y = −0.1824." },
          { kind: "numeric", q: "A freshly initialised 10-class classifier predicts the uniform distribution. What is its cross-entropy loss?", answer: 2.3026, tolerance: 0.001, explain: "−log(1/10) = log 10 ≈ 2.3026 — the sanity-check value at initialisation." },
          { kind: "numeric", q: "With label smoothing ε = 0.1 and K = 4 classes, what is the target probability for the true class?", answer: 0.925, tolerance: 0.0005, explain: "q_c = (1 − ε) + ε/K = 0.9 + 0.025 = 0.925; each other class gets 0.025." },
          { kind: "numeric", q: "Hinge loss with label t = +1 and score s = 0.5: what is the loss?", answer: 0.5, tolerance: 0.001, explain: "max(0, 1 − 1 × 0.5) = 0.5. The example is on the right side but inside the margin." },
          { q: "Why is BCEWithLogitsLoss preferred over Sigmoid followed by BCELoss?", options: ["It is a different loss with better accuracy", "It computes the same loss from logits with a numerically stable formula that cannot produce log(0)", "It does not need labels", "It applies label smoothing automatically"], answer: 1, explain: "Same maths, far more stable numerically: it uses the softplus / log-sum-exp form instead of computing σ(z) and taking its log." },
          { q: "Increasing focal loss's γ from 0 to 2 mainly…", options: ["increases the loss of easy, well-classified examples", "down-weights easy examples so hard examples dominate the gradient", "turns the loss into hinge loss", "changes which class is predicted at inference"], answer: 1, explain: "(1 − p_t)^γ is tiny for p_t near 1, so easy examples count far less; at γ = 0 focal loss equals cross-entropy." },
          { q: "The gradient of (σ(z) − y)² with respect to z is 2(p − y)p(1 − p). What happens when y = 1 and z = −8?", options: ["The gradient is about −2", "The gradient is about −1, like cross-entropy", "The gradient is nearly zero, so learning stalls", "The gradient is positive"], answer: 2, explain: "p ≈ 0.00034, so p(1 − p) ≈ 0.00034 and the gradient ≈ −0.00067: confidently wrong, yet almost no learning signal." },
        ],
      },
      {
        t: "exercise",
        title: "Cross-entropy and focal loss by hand",
        prompt: "Logits z = (1, 2, 3). (a) Compute the softmax probabilities. (b) Compute the cross-entropy loss and its gradient with respect to z when the true class is 0. (c) The same loss when the true class is 2. (d) For a binary example with p_t = 0.8, compare cross-entropy with focal loss at γ = 2 (α = 1).",
        hint: "Use ℓ = log Σ e^{z_j} − z_c; log Σ e^{z_j} = 3 + log(e^{-2} + e^{-1} + 1).",
        solution: [
          { t: "p", text: "(a) $e^{z} = (2.718282, 7.389056, 20.085537)$, sum $30.192875$, so $p = (0.090031, 0.244728, 0.665241)$. (Same as for logits $(-2, -1, 0)$: shifting changes nothing.)" },
          { t: "p", text: String.raw`(b) $\log\sum_j e^{z_j} = \log 30.192875 = 3.407606$. Loss $= 3.407606 - 1 = 2.407606$. Gradient $p - y = (-0.909969, 0.244728, 0.665241)$; it sums to zero.` },
          { t: "p", text: "(c) Loss $= 3.407606 - 3 = 0.407606$; the gradient is $(0.090031, 0.244728, -0.334759)$." },
          { t: "p", text: String.raw`(d) Cross-entropy $-\log 0.8 = 0.223144$. Focal: $(1 - 0.8)^2 \times 0.223144 = 0.04 \times 0.223144 = 0.008926$, 25 times smaller.` },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a stable sparse softmax cross-entropy",
        prompt: "Implement sparse_softmax_ce(Z, labels) in NumPy, returning the mean cross-entropy and its gradient with respect to the logits, for integer labels, using the log-sum-exp trick. Test it on logits of size around 500 (where the naive formula overflows) and check the loss and gradient against PyTorch's F.cross_entropy.",
        hint: "−log p_c = LSE(z) − z_c, and softmax(z) = exp(z − LSE(z)).",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "stable_sparse_ce.py",
            code: String.raw`import numpy as np
import torch
import torch.nn.functional as F

def naive_ce(Z, labels):
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)    # overflows for large logits
    return -np.log(P[np.arange(len(labels)), labels]).mean()

def sparse_softmax_ce(Z, labels):
    """Mean cross-entropy for integer labels, and its gradient dL/dZ."""
    N = Z.shape[0]
    m = Z.max(axis=1, keepdims=True)                       # log-sum-exp shift
    lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))      # log sum_j e^{z_j}, exact and finite
    loss = (lse - Z[np.arange(N), labels]).mean()          # -log p_c = LSE(z) - z_c
    P = np.exp(Z - lse[:, None])                           # softmax, reusing the LSE
    P[np.arange(N), labels] -= 1.0                         # p - onehot(y)
    return loss, P / N

rng = np.random.default_rng(0)
Z = rng.normal(size=(5, 4)) * 500                          # logits of size ~500: e^z overflows float64
labels = rng.integers(0, 4, size=5)

with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    print("naive loss  :", naive_ce(Z, labels))
loss, grad = sparse_softmax_ce(Z, labels)

Zt = torch.tensor(Z, requires_grad=True)
lt = F.cross_entropy(Zt, torch.tensor(labels))
lt.backward()
print("stable loss :", round(loss, 6))
print("torch loss  :", round(lt.item(), 6))
print("max |grad - torch grad| =", np.abs(grad - Zt.grad.numpy()).max())
print("each gradient row sums to", np.round(grad.sum(axis=1), 12))`,
            output: String.raw`naive loss  : inf
stable loss : 705.026961
torch loss  : 705.026961
max |grad - torch grad| = 1.761050914342067e-133
each gradient row sums to [0. 0. 0. 0. 0.]`,
          },
          { t: "p", text: String.raw`The naive version overflows ($e^{500\cdot k}$ is beyond float64's range) and returns inf. The stable version matches PyTorch to every printed digit, with gradients within $10^{-132}$. Each gradient row sums to zero, as derived in section 3.` },
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    summary: [
      {
        t: "list",
        items: [
          "Classifiers output **logits**; sigmoid (binary) or softmax (multi-class) turns them into probabilities. Accuracy cannot be optimised directly, so we minimise a smooth surrogate.",
          String.raw`**Binary cross-entropy** $= -[y\log p + (1-y)\log(1-p)]$ is the Bernoulli negative log-likelihood; $\partial\ell/\partial z = p - y$.`,
          String.raw`**Softmax cross-entropy** $= -\log p_c = \log\sum_j e^{z_j} - z_c$; $\partial\ell/\partial z_k = p_k - y_k$ (sums to zero). Uniform prediction costs $\log K$.`,
          "**Sparse** CE is the same loss with integer labels — no one-hot matrix.",
          String.raw`**Stability:** log-sum-exp $\log\sum e^{z_j} = m + \log\sum e^{z_j - m}$; $-\log\sigma(z) = \mathrm{softplus}(-z)$. Always pass logits to the loss.`,
          String.raw`**Label smoothing:** $q = (1-\varepsilon)y + \varepsilon/K$, gradient $p - q$, finite optimal logits, better calibration.`,
          String.raw`**Hinge:** $\max(0, 1 - ts)$, zero beyond margin 1, no probabilities.`,
          String.raw`**Focal:** $-\alpha_t(1-p_t)^\gamma\log p_t$; $\gamma = 2$ down-weights an example with $p_t = 0.9$ by 100×. Use for extreme imbalance.`,
          "**MSE on σ(z)** has gradient $2(p - y)p(1-p)$ — it vanishes when the model is confidently wrong (max size 8/27). Do not train classifiers with it.",
          "**Regression losses** (MSE, MAE, Huber, log-cosh) differ in how their gradient grows with the residual: linear, constant, capped, smoothly capped.",
        ],
      },
    ],

    /* ------------------------------------------------------------------ */
    next: [
      {
        t: "list",
        items: [
          "[Regression Losses: MSE, MAE, Huber, Log-Cosh](/learn/m05/regression-losses) — the same ideas for numeric targets, with full coverage of outlier robustness.",
          "[Advanced Losses: Contrastive, Triplet, Dice, IoU](/learn/m05/advanced-losses) — losses for embeddings, segmentation and generative models, built on this lesson's cross-entropy and margin ideas.",
          "[Gradient Descent and Optimizers](/learn/m06/gradient-descent) — the optimizer consumes the gradient $p - y$ computed here; see how its size interacts with the learning rate.",
          "[Regularization](/learn/m07/regularization) — label smoothing is one of several ways to fight over-confidence and overfitting.",
          "[Entropy, Cross-Entropy and KL Divergence](/learn/m02/information-theory) — the information-theory meaning of this loss.",
        ],
      },
    ],
  },
};

export default lesson;
