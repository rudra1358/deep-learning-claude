import type { Lesson } from "@/lib/types";

const LR_NUMPY = String.raw`import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)


def standardize(X_train, X_test):
    """Scale every feature to mean 0, std 1 using TRAINING statistics only."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mu) / sd, (X_test - mu) / sd


def add_bias(X):
    """Prepend a column of ones so the bias b becomes weight theta[0]."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)


# ===================== Part 1: linear regression (diabetes) =====================
# 442 patients, 10 raw features (age, sex, BMI, blood pressure, 6 blood serum values);
# target = a measure of disease progression one year later.
X, y = load_diabetes(return_X_y=True, scaled=False)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
print(f"diabetes: train {X_tr.shape}, test {X_te.shape}")

# (a) closed form: solve the normal equations  (A^T A) theta = A^T y
A_tr, A_te = add_bias(X_tr), add_bias(X_te)
theta_ne = np.linalg.solve(A_tr.T @ A_tr, A_tr.T @ y_tr)
pred = A_te @ theta_ne
print(f"normal equations      test MSE {np.mean((pred - y_te) ** 2):8.1f}   test R^2 {r2(y_te, pred):.4f}")

# (b) why scaling matters for gradient descent: condition number of the Hessian (2/N) A^T A
Z_tr, Z_te = standardize(X_tr, X_te)
B_tr, B_te = add_bias(Z_tr), add_bias(Z_te)
print(f"condition number  raw {np.linalg.cond(A_tr.T @ A_tr):.2e}   standardized {np.linalg.cond(B_tr.T @ B_tr):.1f}")


def gd_linear(A, y, lr, iters, report=(50, 500, 5000)):
    """Full-batch gradient descent on L = mean((A theta - y)^2); prints the excess training MSE."""
    theta = np.zeros(A.shape[1])
    best = np.mean((A @ np.linalg.lstsq(A, y, rcond=None)[0] - y) ** 2)   # optimum, for reference
    for it in range(1, iters + 1):
        grad = 2 / len(y) * A.T @ (A @ theta - y)       # dL/dtheta = (2/N) A^T (A theta - y)
        theta -= lr * grad
        if it in report:
            print(f"    step {it:5d}: train MSE - optimum = {np.mean((A @ theta - y) ** 2) - best:10.4f}")
    return theta


# step size 1/lambda_max: half of the largest stable step 2/lambda_max of the Hessian
for name, A_, Ate in [("raw", A_tr, A_te), ("standardized", B_tr, B_te)]:
    lam_max = np.linalg.eigvalsh(2 / len(y_tr) * A_.T @ A_).max()
    print(f"GD on {name} features, lr = {1 / lam_max:.1e}")
    th = gd_linear(A_, y_tr, lr=1 / lam_max, iters=5000)
    p = Ate @ th
    print(f"    test MSE {np.mean((p - y_te) ** 2):8.1f}   test R^2 {r2(y_te, p):.4f}")

# ===================== Part 2: logistic regression (breast cancer) =====================
# 569 tumours, 30 features computed from a cell-nucleus image; label 1 = benign, 0 = malignant.
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Z_tr, Z_te = standardize(X_tr, X_te)
N = len(y_tr)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


lam = 1 / N            # L2 strength; equals scikit-learn's default C = 1.0 (see the framework section)
w, b, lr = np.zeros(Z_tr.shape[1]), 0.0, 0.5
for it in range(3001):
    p = sigmoid(Z_tr @ w + b)
    if it % 1000 == 0:
        print(f"iter {it:4d}  BCE {bce(p, y_tr):.4f}  train acc {np.mean((p > 0.5) == y_tr):.3f}")
    grad_w = Z_tr.T @ (p - y_tr) / N + lam * w           # (p - y) x, averaged, + L2 term
    grad_b = np.mean(p - y_tr)
    w -= lr * grad_w
    b -= lr * grad_b

p_te = sigmoid(Z_te @ w + b)
pred = (p_te > 0.5).astype(int)
tp = np.sum((pred == 1) & (y_te == 1)); fp = np.sum((pred == 1) & (y_te == 0)); fn = np.sum((pred == 0) & (y_te == 1))
print(f"test: BCE {bce(p_te, y_te):.4f}  accuracy {np.mean(pred == y_te):.3f}  precision {tp / (tp + fp):.3f}  recall {tp / (tp + fn):.3f}")
top = np.argsort(-np.abs(w))[:3]
for j in top:
    print(f"  {load_breast_cancer().feature_names[j]:22s} w = {w[j]:+.3f}  odds x{np.exp(w[j]):.3f} per +1 std")

# ===================== Part 3: softmax regression (iris, 3 classes) =====================
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Z_tr, Z_te = standardize(X_tr, X_te)
K, N = 3, len(y_tr)
Y = np.eye(K)[y_tr]                                     # one-hot targets, shape (N, K)
W, c = np.zeros((Z_tr.shape[1], K)), np.zeros(K)


def softmax(S):
    S = S - S.max(axis=1, keepdims=True)                # subtract row max: avoids overflow
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


for it in range(2000):
    P = softmax(Z_tr @ W + c)
    W -= 0.5 * (Z_tr.T @ (P - Y) / N + 1e-3 * W)        # gradient (P - Y) x, same form as the binary case
    c -= 0.5 * (P - Y).mean(axis=0)
P = softmax(Z_tr @ W + c)
print(f"softmax regression: train CE {-np.mean(np.log(P[np.arange(N), y_tr])):.4f}  "
      f"train acc {np.mean(P.argmax(1) == y_tr):.3f}  test acc {np.mean(softmax(Z_te @ W + c).argmax(1) == y_te):.3f}")`;

const LR_NUMPY_OUT = String.raw`diabetes: train (331, 10), test (111, 10)
normal equations      test MSE   3180.2   test R^2 0.3594
condition number  raw 5.08e+07   standardized 442.6
GD on raw features, lr = 6.8e-06
    step    50: train MSE - optimum =  2470.1574
    step   500: train MSE - optimum =  1110.2838
    step  5000: train MSE - optimum =   340.2354
    test MSE   3453.6   test R^2 0.3043
GD on standardized features, lr = 1.2e-01
    step    50: train MSE - optimum =     9.5510
    step   500: train MSE - optimum =     1.2180
    step  5000: train MSE - optimum =     0.0000
    test MSE   3180.2   test R^2 0.3594
iter    0  BCE 0.6931  train acc 0.373
iter 1000  BCE 0.0494  train acc 0.991
iter 2000  BCE 0.0485  train acc 0.991
iter 3000  BCE 0.0484  train acc 0.991
test: BCE 0.0859  accuracy 0.958  precision 0.967  recall 0.967
  radius error           w = -1.213  odds x0.297 per +1 std
  worst texture          w = -1.108  odds x0.330 per +1 std
  mean concave points    w = -1.102  odds x0.332 per +1 std
softmax regression: train CE 0.0774  train acc 0.971  test acc 1.000`;

const LR_TORCH = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)       # float64 so results match the NumPy version closely


def prepare(X, y, **split):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, **split)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)          # training statistics only
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    return t((X_tr - mu) / sd), t((X_te - mu) / sd), t(y_tr).unsqueeze(1), t(y_te).unsqueeze(1)


# ---------- linear regression = one neuron, identity activation, MSE loss ----------
X_tr, X_te, y_tr, y_te = prepare(*load_diabetes(return_X_y=True, scaled=False))
model = nn.Linear(10, 1)                     # z = w.x + b ; no activation
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
for step in range(5000):
    loss = loss_fn(model(X_tr), y_tr)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pred = model(X_te)
    r2 = 1 - ((pred - y_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
    print(f"linear   : train MSE {loss.item():.1f}  test MSE {loss_fn(pred, y_te).item():.1f}  test R^2 {r2.item():.4f}")

# ---------- logistic regression = one neuron, sigmoid activation, BCE loss ----------
X_tr, X_te, y_tr, y_te = prepare(*load_breast_cancer(return_X_y=True), stratify=load_breast_cancer().target)
N = len(y_tr)
model = nn.Linear(30, 1)                     # outputs the logit z; the sigmoid lives inside the loss
nn.init.zeros_(model.weight); nn.init.zeros_(model.bias)   # start at 0 like the NumPy version (loss is convex)
loss_fn = nn.BCEWithLogitsLoss()             # sigmoid + BCE fused: numerically stable
opt = torch.optim.SGD([
    {"params": [model.weight], "weight_decay": 1 / N},   # L2 on w only (adds lambda*w to the gradient)
    {"params": [model.bias], "weight_decay": 0.0},       # the bias is conventionally not regularised
], lr=0.5)
for step in range(3001):
    logits = model(X_tr)
    loss = loss_fn(logits, y_tr)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    p = torch.sigmoid(model(X_te))
    acc = ((p > 0.5).double() == y_te).double().mean().item()
    print(f"logistic : train BCE {loss.item():.4f}  test BCE {nn.functional.binary_cross_entropy(p, y_te).item():.4f}  test acc {acc:.3f}")
    j = model.weight.abs().argmax().item()
    print(f"largest weight: feature {j}, w = {model.weight[0, j].item():+.3f}")`;

const LR_TORCH_OUT = String.raw`linear   : train MSE 2780.1  test MSE 3180.2  test R^2 0.3594
logistic : train BCE 0.0484  test BCE 0.0859  test acc 0.958
largest weight: feature 10, w = -1.213`;

const LR_SKLEARN = String.raw`import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------- linear regression: closed-form least squares ----------
X, y = load_diabetes(return_X_y=True, scaled=False)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
lin = LinearRegression().fit(X_tr, y_tr)                # solves least squares (no scaling needed)
pred = lin.predict(X_te)
print(f"LinearRegression   test MSE {mean_squared_error(y_te, pred):.1f}  test R^2 {r2_score(y_te, pred):.4f}")

A = np.hstack([np.ones((len(X_tr), 1)), X_tr])           # NumPy normal equations, for comparison
theta = np.linalg.solve(A.T @ A, A.T @ y_tr)
print(f"  max |coef difference| vs NumPy normal equations: {np.max(np.abs(theta[1:] - lin.coef_)):.2e}")

# ---------- logistic regression: iterative solver (L-BFGS) with L2 penalty ----------
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, tol=1e-10, max_iter=10000))
clf.fit(X_tr, y_tr)
p_te = clf.predict_proba(X_te)[:, 1]
print(f"LogisticRegression test BCE {log_loss(y_te, p_te):.4f}  test acc {accuracy_score(y_te, p_te > 0.5):.3f}")

# NumPy gradient descent on the SAME objective: mean BCE + (lambda/2)||w||^2 with lambda = 1/(C N)
Z = clf[0].transform(X_tr)
N = len(y_tr)
w, b = np.zeros(Z.shape[1]), 0.0
for _ in range(20000):
    p = 1 / (1 + np.exp(-(Z @ w + b)))
    w -= 0.5 * (Z.T @ (p - y_tr) / N + w / N)
    b -= 0.5 * np.mean(p - y_tr)
lr_ = clf[1]
print(f"  max |coef difference| vs NumPy gradient descent: {np.max(np.abs(w - lr_.coef_[0])):.2e}"
      f"   intercepts {lr_.intercept_[0]:.4f} vs {b:.4f}")`;

const LR_SKLEARN_OUT = String.raw`LinearRegression   test MSE 3180.2  test R^2 0.3594
  max |coef difference| vs NumPy normal equations: 4.21e-11
LogisticRegression test BCE 0.0859  test acc 0.958
  max |coef difference| vs NumPy gradient descent: 5.13e-07   intercepts 0.3312 vs 0.3312`;

const LR_KERAS = String.raw`import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

# logistic regression = a single Dense unit with a sigmoid activation
model = tf.keras.Sequential([
    tf.keras.Input(shape=(30,)),
    tf.keras.layers.Dense(1, activation="sigmoid",
                          kernel_regularizer=tf.keras.regularizers.L2(0.5 / len(y_tr))),
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr, batch_size=len(y_tr), epochs=3000, verbose=0)   # full batch, like the NumPy code
loss, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"test BCE {loss:.4f}  test accuracy {acc:.3f}")`;

const POLY_CODE = String.raw`import numpy as np

rng = np.random.default_rng(0)
n = 40
x = rng.uniform(-1, 1, n)                                # one input feature
y = 1 + 2 * x - 3 * x ** 2 + rng.normal(0, 0.2, n)       # true curve is quadratic, noise std 0.2
x_test = np.linspace(-1, 1, 200)
y_test = 1 + 2 * x_test - 3 * x_test ** 2                # noiseless curve, for test error


def design(x, degree):
    """Columns 1, x, x^2, ..., x^degree: polynomial features."""
    return np.vander(x, degree + 1, increasing=True)


for degree in (1, 2, 10):
    A = design(x, degree)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)       # least squares; still LINEAR in theta
    train = np.mean((A @ theta - y) ** 2)
    test = np.mean((design(x_test, degree) @ theta - y_test) ** 2)
    print(f"degree {degree:2d}: {degree + 1:2d} weights  train MSE {train:.4f}  test MSE vs true curve {test:.4f}")
    if degree == 2:
        print("           theta =", np.round(theta, 3))`;

const POLY_OUT = String.raw`degree  1:  2 weights  train MSE 0.9969  test MSE vs true curve 0.8322
degree  2:  3 weights  train MSE 0.0456  test MSE vs true curve 0.0042
           theta = [ 1.032  2.05  -2.933]
degree 10: 11 weights  train MSE 0.0386  test MSE vs true curve 0.0113`;

const lesson: Lesson = {
  id: "m03/linear-logistic-regression",
  title: "Linear and Logistic Regression as Neural Networks",
  summary:
    "The first models that learn from data: least-squares linear regression (exact formula and gradient descent), feature scaling and polynomial features, logistic regression with the sigmoid, log-odds, binary cross-entropy and its gradient, softmax regression — and why each one is a single neuron.",
  minutes: 70,
  level: "beginner",
  prerequisites: ["m02/vectors-matrices", "m02/derivatives-chain-rule", "m02/probability", "m02/mle-map"],
  next: ["m03/artificial-neuron", "m03/multilayer-networks", "m05/classification-losses"],
  terms: [
    "linear-regression",
    "logistic-regression",
    "mse",
    "normal-equations",
    "feature-scaling",
    "polynomial-features",
    "sigmoid",
    "log-odds",
    "decision-boundary",
    "binary-cross-entropy",
    "softmax-regression",
    "convexity",
  ],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "Every model in this course follows one recipe. Pick a family of functions with adjustable numbers (**parameters**). Measure how wrong the function is on example data with a **loss**. Adjust the parameters to make the loss small. This lesson runs the recipe on the two simplest families:",
      },
      {
        t: "list",
        items: [
          "**Linear regression** predicts a real number (a price, a dose, a temperature) as a weighted sum of the input **features** (the input numbers).",
          "**Logistic regression** predicts the **probability** that an input belongs to a class (malignant or benign, spam or not) by squashing a weighted sum through an S-shaped curve.",
        ],
      },
      {
        t: "p",
        text: "Both are old. Least squares, the method behind linear regression, was published by Legendre (1805) and Gauss (1809) to fit planet and comet orbits to noisy observations. Verhulst introduced the logistic curve in the 1830s–1840s for population growth; statisticians turned it into a model for yes/no outcomes in the mid-twentieth century (David Cox's 1958 paper \"The regression analysis of binary sequences\" is a standard reference). Both are still widely used in medicine, economics and industry because they are fast, stable and easy to interpret.",
      },
      {
        t: "p",
        text: "Why they matter for deep learning: **each is a neural network with a single neuron**. Linear regression is a neuron with no activation (the identity) trained with mean squared error. Logistic regression is a neuron with a sigmoid activation trained with binary cross-entropy. Softmax regression is a layer of neurons with a softmax. Everything after this — the [perceptron](/learn/m03/artificial-neuron), [multi-layer networks](/learn/m03/multilayer-networks), [backpropagation](/learn/m04/backpropagation) — stacks these units and trains them with the gradient you will derive here.",
      },
      {
        t: "callout",
        kind: "key",
        title: "In one sentence",
        text: String.raw`Linear regression: prediction $\hat y = \mathbf{w}^\top\mathbf{x} + b$ (weights $\mathbf{w}$ times inputs $\mathbf{x}$, plus bias $b$), fitted by minimising squared error (exact formula or gradient descent). Logistic regression: $p = \sigma(\mathbf{w}^\top\mathbf{x} + b)$, where the sigmoid $\sigma$ squashes to a probability, fitted by minimising cross-entropy (gradient descent only). For both, the gradient of the loss is the average of (prediction − target) × input.`,
      },
    ],

    // --------------------------------------------------------- prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          String.raw`[Scalars, Vectors, Matrices and Tensors](/learn/m02/vectors-matrices) — dot products $\mathbf{w}^\top\mathbf{x}$ (multiply matching entries and add), matrix–vector products $X\mathbf{w}$, the transpose and the inverse. The whole dataset is stored as a matrix with one row per example.`,
          String.raw`[Derivatives, Partial Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) — you will differentiate a sum of squares and $\log\sigma(z)$. We derive $\sigma'(z) = \sigma(z)(1-\sigma(z))$ again here.`,
          "[Probability Theory for Deep Learning](/learn/m02/probability) — Bernoulli and Gaussian distributions, and what $p(y\\mid\\mathbf{x})$ means.",
          "[Maximum Likelihood and MAP Estimation](/learn/m02/mle-map) — helpful, not required: least squares and cross-entropy are both negative log-likelihoods, and L2 regularisation is a Gaussian prior.",
          "[Optimization Mathematics](/learn/m02/optimization-math) — convex functions and the Hessian, which explain why both losses have no bad local minima.",
          "The gradient descent update $\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta\\mathcal{L}$: move the parameters $\\theta$ a small step (size $\\eta$, the learning rate) against the gradient of the loss $\\mathcal{L}$. Explained as needed here; in depth in [Gradient Descent](/learn/m06/gradient-descent).",
        ],
      },
    ],

    // ------------------------------------------------------------- intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A pricing rule and a risk score",
        text: "An estate agent prices flats with a rule like *€4,000 per square metre, plus €15,000 for a balcony, minus €2,000 per km from the station, plus a base of €30,000*. Each feature gets a rate (a weight), the base is a bias, and the price is the weighted sum. Linear regression *finds* those rates from flats that already sold, by picking the rule whose predictions miss the real prices least on average. A clinic's risk score works the same way — points for age, blood pressure, smoking — but the total is then turned into a probability by a table that rises slowly, then steeply, then flattens near 100%. That table is the sigmoid; learning the points from past patients is logistic regression.",
      },
      {
        t: "p",
        text: "A **linear model** draws the straightest possible relationship between inputs and output: a line through a scatter plot (one input), a tilted plane (two inputs), a flat **hyperplane** (many). Training tilts and shifts the line until the vertical gaps to the data points are as small as possible in total. We square each gap before adding, so big misses count more and positive and negative errors can't cancel.",
      },
      {
        t: "p",
        text: "For **classification** the target is 0 or 1, and a straight line is the wrong shape: it shoots above 1 and below 0, and one far-away example can drag it off course. Logistic regression keeps the weighted sum — the **score** $z$ — but reads it as *evidence*: positive favours class 1, negative class 0, and the sigmoid turns evidence into a probability. Where the evidence balances ($z = 0$, probability 0.5) is a straight line (in 2-D) through input space: the **decision boundary**.",
      },
      {
        t: "p",
        text: "Both models learn the same way. For each training example, compare prediction with truth, and nudge every weight by *how wrong the prediction was* times *how large that input was*. Well-predicted examples barely move the weights; badly predicted ones move them a lot. That sentence is the gradient of both losses — and, applied layer by layer, it is how every deep network learns.",
      },
    ],

    // ------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Linear regression, step by step" },
      {
        t: "steps",
        items: [
          { title: "Collect data", text: String.raw`$N$ examples, each a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ ($d$ numbers) with a real target $y_i$. Stack the inputs as rows of a matrix $X \in \mathbb{R}^{N\times d}$ and the targets into $\mathbf{y} \in \mathbb{R}^N$.` },
          { title: "Absorb the bias", text: String.raw`Add a column of ones in front of $X$ (call it $\tilde X$) and put $b$ first in the parameter vector: $\boldsymbol\theta = (b, w_1, \dots, w_d)$. Now $\hat y_i = \boldsymbol\theta^\top \tilde{\mathbf{x}}_i$, and all predictions at once are $\hat{\mathbf{y}} = \tilde X\boldsymbol\theta$.` },
          { title: "Measure the error", text: "Compute the residuals (misses) $r_i = \\hat y_i - y_i$ and the mean squared error $\\frac1N\\sum_i r_i^2$." },
          { title: "Fit — option A, exact formula", text: String.raw`Solve $\tilde X^\top \tilde X\,\boldsymbol\theta = \tilde X^\top \mathbf{y}$ (the **normal equations**) in one go. Cost: $O(Nd^2)$ to form $\tilde X^\top\tilde X$, $O(d^3)$ to solve.` },
          { title: "Fit — option B, gradient descent", text: String.raw`Start at $\boldsymbol\theta = 0$ and repeat $\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\cdot\frac{2}{N}\tilde X^\top(\tilde X\boldsymbol\theta - \mathbf{y})$. Each step costs $O(Nd)$; it scales to huge $N$ and $d$, and it is the only option once the model is no longer linear.` },
          { title: "Scale features first (for option B)", text: "Subtract each feature's training mean and divide by its training standard deviation. The best predictions don't change, but gradient descent can converge thousands of times faster (shown below on real data)." },
          { title: "Evaluate on held-out data", text: "Report error on a test set never used for fitting or for the scaling statistics: MSE, and $R^2 = 1 - \\text{SSE}/\\text{SST}$, the fraction of the target's variance explained." },
        ],
      },
      { t: "h", text: "Logistic regression, step by step" },
      {
        t: "steps",
        items: [
          { title: "Score", text: String.raw`For each example compute the **logit** $z_i = \mathbf{w}^\top\mathbf{x}_i + b$ — exactly the linear-regression output.` },
          { title: "Squash", text: String.raw`Turn the score into a probability: $p_i = \sigma(z_i) = 1/(1+e^{-z_i})$, read as $P(y_i = 1 \mid \mathbf{x}_i)$, "the chance that $y_i$ is 1 given $\mathbf{x}_i$".` },
          { title: "Loss", text: String.raw`Binary cross-entropy: $-\frac1N\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$. Small when the model gives high probability to what really happened.` },
          { title: "Gradient", text: String.raw`$\nabla_{\mathbf w} = \frac1N\sum_i (p_i - y_i)\,\mathbf{x}_i$, $\;\partial/\partial b = \frac1N\sum_i (p_i - y_i)$. There is no exact formula, so we iterate.` },
          { title: "Update", text: String.raw`$\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla_{\mathbf w}$, $b \leftarrow b - \eta\,\partial_b$; repeat until the loss stops falling (optionally add $\lambda\mathbf{w}$ to the gradient for L2 regularisation).` },
          { title: "Decide", text: String.raw`Predict class 1 when $p \ge t$ for a threshold $t$ (default 0.5, the same as $z \ge 0$). The threshold is a separate choice from training and should reflect what each kind of error costs.` },
        ],
      },
      { t: "h", text: "Both as one neuron" },
      {
        t: "table",
        head: ["", "Linear regression", "Logistic regression", "Softmax regression"],
        rows: [
          ["Neurons", "1", "1", "K (one per class)"],
          ["Pre-activation", "$z = \\mathbf{w}^\\top\\mathbf{x} + b$", "$z = \\mathbf{w}^\\top\\mathbf{x} + b$", "$z_k = \\mathbf{w}_k^\\top\\mathbf{x} + b_k$"],
          ["Activation", "identity: $\\hat y = z$", "sigmoid: $p = \\sigma(z)$", "softmax: $p_k = e^{z_k}/\\sum_j e^{z_j}$"],
          ["Output means", "a real number", "$P(y=1\\mid\\mathbf{x})$", "$P(y=k\\mid\\mathbf{x})$"],
          ["Loss", "mean squared error", "binary cross-entropy", "categorical cross-entropy"],
          ["$\\partial\\mathcal{L}/\\partial z$ per example", "$2(\\hat y - y)$", "$p - y$", "$p_k - [y = k]$"],
          ["Closed form?", "yes (normal equations)", "no", "no"],
          ["PyTorch", "`nn.Linear(d, 1)` + `nn.MSELoss`", "`nn.Linear(d, 1)` + `nn.BCEWithLogitsLoss`", "`nn.Linear(d, K)` + `nn.CrossEntropyLoss`"],
        ],
      },
      {
        t: "p",
        text: "The key row is $\\partial\\mathcal{L}/\\partial z$: in every case the error signal at the neuron is just *prediction minus target*. Not a coincidence — each loss is the negative log-likelihood of the distribution whose natural parameter is $z$ (Gaussian, Bernoulli, categorical), a pairing statisticians call the **canonical link**. That is why these same pairs (identity + MSE, sigmoid + BCE, softmax + cross-entropy) are the output layers of deep networks.",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Linear regression: model and loss" },
      {
        t: "p",
        text: String.raw`Notation: $N$ examples, $d$ features. The **design matrix** $\tilde X \in \mathbb{R}^{N\times(d+1)}$ has row $i$ equal to $(1, x_{i1}, \dots, x_{id})$. $\boldsymbol\theta \in \mathbb{R}^{d+1}$ holds bias and weights, $\mathbf{y}\in\mathbb{R}^N$ the targets, $\hat{\mathbf{y}} = \tilde X\boldsymbol\theta$ the predictions. The loss is the mean squared error ($\lVert\cdot\rVert^2$ = sum of squares):`,
      },
      { t: "math", tex: String.raw`\mathcal{L}(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^{N}\big(\tilde{\mathbf{x}}_i^\top\boldsymbol\theta - y_i\big)^2 = \frac{1}{N}\,\big\lVert \tilde X\boldsymbol\theta - \mathbf{y}\big\rVert^2` },
      { t: "h", text: "Deriving the gradient" },
      {
        t: "p",
        text: String.raw`Chain rule on one term: $\frac{\partial}{\partial\theta_j}(\tilde{\mathbf{x}}_i^\top\boldsymbol\theta - y_i)^2 = 2(\tilde{\mathbf{x}}_i^\top\boldsymbol\theta - y_i)\,\tilde x_{ij}$. Average over $i$ and stack all $j$ into a vector:`,
      },
      { t: "math", tex: String.raw`\nabla_{\boldsymbol\theta}\mathcal{L} = \frac{2}{N}\sum_{i=1}^{N}(\hat y_i - y_i)\,\tilde{\mathbf{x}}_i = \frac{2}{N}\,\tilde X^\top\big(\tilde X\boldsymbol\theta - \mathbf{y}\big)`, caption: "Residual times input, averaged. The matrix form computes all of them with one product." },
      { t: "h", text: "Deriving the normal equations" },
      {
        t: "p",
        text: String.raw`The loss is a bowl in $\boldsymbol\theta$, so its minimum is where the gradient is zero. Set $\nabla\mathcal{L} = 0$ and drop the constant $2/N$:`,
      },
      { t: "math", tex: String.raw`\tilde X^\top\tilde X\,\boldsymbol\theta^\star = \tilde X^\top\mathbf{y} \quad\Longrightarrow\quad \boldsymbol\theta^\star = \big(\tilde X^\top\tilde X\big)^{-1}\tilde X^\top\mathbf{y}\quad\text{(if } \tilde X^\top\tilde X \text{ is invertible)}` },
      {
        t: "p",
        text: String.raw`The Hessian (matrix of second derivatives) is $\nabla^2\mathcal{L} = \frac{2}{N}\tilde X^\top\tilde X$. For any vector $\mathbf{v}$, $\mathbf{v}^\top\tilde X^\top\tilde X\mathbf{v} = \lVert\tilde X\mathbf{v}\rVert^2 \ge 0$, so the Hessian is positive semi-definite: the loss is **convex** (one bowl) and every flat point is a global minimum. The minimum is unique exactly when the columns of $\tilde X$ are linearly independent. Otherwise — duplicated features, or more features than examples — infinitely many $\boldsymbol\theta$ fit equally well and $\tilde X^\top\tilde X$ is singular (not invertible).`,
      },
      {
        t: "p",
        text: String.raw`**Geometric meaning.** The normal equations say $\tilde X^\top(\mathbf{y} - \tilde X\boldsymbol\theta^\star) = 0$: the residual vector is **orthogonal** (at right angles) to every column of $\tilde X$. So $\hat{\mathbf{y}}$ is the projection of $\mathbf{y}$ onto the space the features span — the closest point to $\mathbf{y}$ the model can produce. Since one column is all ones, the residuals sum to zero.`,
      },
      {
        t: "p",
        text: String.raw`**Probabilistic meaning.** If $y_i = \mathbf{w}^\top\mathbf{x}_i + b + \varepsilon_i$ with independent Gaussian noise $\varepsilon_i \sim \mathcal{N}(0, s^2)$, the log-likelihood is $-\frac{1}{2s^2}\sum_i (y_i - \hat y_i)^2 + \text{const}$. Maximising it is exactly minimising squared error ([MLE](/learn/m02/mle-map)). A Gaussian prior $\mathbf{w}\sim\mathcal{N}(0, \tau^2 I)$ gives ridge regression, $\boldsymbol\theta^\star = (\tilde X^\top\tilde X + \lambda I)^{-1}\tilde X^\top\mathbf{y}$ — invertible for any $\lambda > 0$.`,
      },
      { t: "h", text: "Worked example: four points, both ways" },
      {
        t: "p",
        text: String.raw`Data: $x = (1, 2, 3, 4)$, $y = (3, 5, 6, 9)$. Model $\hat y = b + wx$, so $\tilde X$ has rows $(1, x_i)$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Form the normal equations", tex: String.raw`\tilde X^\top\tilde X = \begin{pmatrix} N & \sum x_i \\ \sum x_i & \sum x_i^2\end{pmatrix} = \begin{pmatrix} 4 & 10\\ 10 & 30\end{pmatrix},\qquad \tilde X^\top\mathbf{y} = \begin{pmatrix}\sum y_i\\ \sum x_iy_i\end{pmatrix} = \begin{pmatrix}23\\ 67\end{pmatrix}` },
          { title: "Invert the 2×2 matrix", text: "Determinant $4\\cdot30 - 10\\cdot10 = 20$.", tex: String.raw`\big(\tilde X^\top\tilde X\big)^{-1} = \frac{1}{20}\begin{pmatrix}30 & -10\\ -10 & 4\end{pmatrix}` },
          { title: "Solve", tex: String.raw`\begin{pmatrix} b\\ w\end{pmatrix} = \frac{1}{20}\begin{pmatrix}30\cdot23 - 10\cdot67\\ -10\cdot23 + 4\cdot67\end{pmatrix} = \frac{1}{20}\begin{pmatrix}20\\ 38\end{pmatrix} = \begin{pmatrix}1\\ 1.9\end{pmatrix}` },
          { title: "Predictions and residuals", tex: String.raw`\hat{\mathbf y} = (2.9,\ 4.8,\ 6.7,\ 8.6),\quad \mathbf{r} = \hat{\mathbf y} - \mathbf y = (-0.1,\ -0.2,\ 0.7,\ -0.4),\quad \text{MSE} = \tfrac{0.01+0.04+0.49+0.16}{4} = 0.175` },
          { title: "Check orthogonality", tex: String.raw`\textstyle\sum_i r_i = 0,\qquad \sum_i r_i x_i = -0.1 - 0.4 + 2.1 - 1.6 = 0\ \checkmark` },
          { title: "Gradient descent from w = b = 0, η = 0.05 — step 1", text: "Residuals $\\hat y - y = (-3, -5, -6, -9)$, loss $= (9+25+36+81)/4 = 37.75$.", tex: String.raw`\frac{\partial\mathcal L}{\partial w} = \tfrac{2}{4}\big((-3)(1) + (-5)(2) + (-6)(3) + (-9)(4)\big) = -33.5,\quad \frac{\partial\mathcal L}{\partial b} = \tfrac24(-23) = -11.5` },
          { title: "Update", tex: String.raw`w = 0 - 0.05(-33.5) = 1.675,\quad b = 0 - 0.05(-11.5) = 0.575,\quad \mathcal L = 1.213438` },
          { title: "Step 2", tex: String.raw`\nabla = (-5.5,\ -1.975)\ \Rightarrow\ w = 1.95,\ b = 0.67375,\quad \mathcal L = 0.218627` },
          { title: "Convergence", text: "After 2000 steps gradient descent reaches $w = 1.900000$, $b = 1.000000$ — the exact answer. The bias converges far more slowly than the slope; the next part explains why." },
        ],
      },
      { t: "h", text: "Why feature scaling matters for gradient descent" },
      {
        t: "p",
        text: String.raw`On a quadratic bowl, gradient descent acts separately along each eigenvector of the Hessian $H = \frac2N\tilde X^\top\tilde X$; each eigenvalue $\lambda_k$ is the curvature in that direction. A step size $\eta$ is stable only if $\eta < 2/\lambda_{\max}$, and the error along direction $k$ shrinks by $|1 - \eta\lambda_k|$ per step. With the largest sensible step $\eta \approx 1/\lambda_{\max}$, the flattest direction shrinks by only $1 - \lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa$ per step, where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the **condition number**. Roughly $\kappa$ steps are needed per factor-$e$ drop in error. In plain words: a long, narrow valley forces tiny steps, and progress along its floor crawls.`,
      },
      {
        t: "p",
        text: String.raw`In the four-point example, $H = \begin{pmatrix}2 & 5\\5 & 15\end{pmatrix}$ has eigenvalues $0.299$ and $16.70$: $\kappa = 55.8$, and steps above $2/16.70 = 0.120$ diverge. The cause: the feature $x$ (mean 2.5) and the constant column are strongly correlated. Centring $x$ removes the correlation; dividing by the standard deviation evens out the curvature. **Standardisation** $x_j \leftarrow (x_j - \mu_j)/s_j$, with mean $\mu_j$ and standard deviation $s_j$ from the training set, makes $\kappa$ small. On the real diabetes data below, raw features give $\kappa = 5.08\times10^{7}$ and standardised ones $\kappa = 442.6$; after 5000 steps the raw run is still 340 MSE units above the optimum while the standardised run has reached it.`,
      },
      {
        t: "p",
        text: String.raw`Scaling changes the parameters, not the model family: a standardised model $\hat y = b' + \sum_j w'_j (x_j-\mu_j)/s_j$ is the same function as an unscaled one with $w_j = w'_j/s_j$ and $b = b' - \sum_j w'_j\mu_j/s_j$. So the exact solution is unaffected (up to rounding) — scikit-learn's LinearRegression needs no scaling, but gradient-based training does.`,
      },
      { t: "h", text: "Multiple and polynomial features" },
      {
        t: "p",
        text: String.raw`Nothing above assumed $d = 1$. With $d$ features the same formulas hold with a $(d+1)$-column design matrix, and each weight $w_j$ is the change in $\hat y$ per unit of $x_j$ **with the other features held fixed**. "Linear" means linear in the **parameters**, not the input. Replace $x$ by fixed transformations $\phi(x) = (1, x, x^2, \dots, x^k)$ — **polynomial features** — and you get a curved fit, still solved by the normal equations with $\Phi$ in place of $\tilde X$. The coding exercise fits a quadratic this way. The catch: you choose the features by hand, and high degrees overfit (see [Training Basics](/learn/m03/training-basics)). Learning features from data is exactly what hidden layers do.`,
      },
      { t: "h", text: "Why linear regression fails for classification" },
      {
        t: "p",
        text: String.raw`Code the classes as $y\in\{0,1\}$, fit least squares, and predict class 1 when $\hat y > 0.5$. On $x = (1,2,3,4,6,7,8,9)$ with labels $(0,0,0,0,1,1,1,1)$ this works: the line $\hat y = -0.333 + 0.1667x$ crosses 0.5 at $x = 5.0$, and all eight points are right. Now add one more clear-cut class-1 example at $x = 40$. Least squares pulls $\hat y(40)$ toward 1 and flattens the line to $\hat y = 0.358 + 0.0222x$: the threshold moves to $x = 6.39$, and the point at $x = 6$ (prediction 0.491) is now wrong — accuracy drops to 8/9 because of an *easy* point. Squared error also punishes predictions that are "too correct" (1.25 for the point at 40), and outputs outside $[0,1]$ can't be probabilities. Logistic regression on the same nine points keeps its threshold at $x = 4.99$ and gets all of them right.`,
      },
      { t: "h", text: "Logistic regression: the model" },
      {
        t: "math",
        tex: String.raw`P(y = 1\mid\mathbf{x}) = p = \sigma(z) = \frac{1}{1 + e^{-z}},\qquad z = \mathbf{w}^\top\mathbf{x} + b`,
      },
      {
        t: "p",
        text: String.raw`$\sigma$ turns any real number into one between 0 and 1, always increases, satisfies $\sigma(-z) = 1 - \sigma(z)$, and has derivative $\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\big(1-\sigma(z)\big)$. Solving $p = \sigma(z)$ for $z$ gives the **log-odds** (logit):`,
      },
      { t: "math", tex: String.raw`z = \log\frac{p}{1-p} = \mathbf{w}^\top\mathbf{x} + b`, caption: "Logistic regression is a linear model of the log-odds." },
      {
        t: "p",
        text: String.raw`Odds $p/(1-p)$ of 4 mean "4 to 1": $p = 0.8$ has log-odds $\log 4 = 1.386$; $p = 0.9$ gives $2.197$; $p = 0.99$ gives $4.595$; $p = 0.5$ gives $0$. Since log-odds are linear, raising $x_j$ by one unit adds $w_j$ to the log-odds, i.e. **multiplies the odds by $e^{w_j}$**, whatever the other features are. A weight of $0.7$ multiplies the odds by $e^{0.7} = 2.014$. In the breast-cancer model below, one standard deviation more "radius error" multiplies the odds of *benign* by $e^{-1.213} = 0.297$.`,
      },
      {
        t: "p",
        text: String.raw`**Decision boundary.** $p \ge 0.5 \iff z \ge 0 \iff \mathbf{w}^\top\mathbf{x} + b \ge 0$, so the boundary is the hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$. $\mathbf{w}$ is perpendicular to it and points toward class 1; a point's signed distance from it is $(\mathbf{w}^\top\mathbf{x}+b)/\lVert\mathbf{w}\rVert$. Lines of equal probability are parallel: $p = 0.75$ where $z = \log 3 = 1.099$, $p = 0.25$ where $z = -1.099$, each $1.099/\lVert\mathbf{w}\rVert$ from the boundary — so a larger $\lVert\mathbf{w}\rVert$ means a sharper switch from 0 to 1.`,
      },
      { t: "h", text: "The loss: binary cross-entropy from maximum likelihood" },
      {
        t: "p",
        text: String.raw`Each label is a coin flip (Bernoulli) with chance $p_i$ of being 1, so $P(y_i\mid\mathbf{x}_i) = p_i^{y_i}(1-p_i)^{1-y_i}$ — that is $p_i$ if $y_i = 1$ and $1 - p_i$ if $y_i = 0$. The averaged negative log-likelihood of the training set is the **binary cross-entropy**:`,
      },
      { t: "math", tex: String.raw`\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]` },
      { t: "h", text: "Deriving the gradient (p − y)x" },
      {
        t: "p",
        text: String.raw`Take one example and write $\ell = -[y\log p + (1-y)\log(1-p)]$ with $p = \sigma(z)$. Chain rule through $p$:`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial\ell}{\partial z} = \underbrace{\Big(-\frac{y}{p} + \frac{1-y}{1-p}\Big)}_{\partial\ell/\partial p}\;\underbrace{p(1-p)}_{\sigma'(z)} = -y(1-p) + (1-y)p = p - y`,
      },
      {
        t: "p",
        text: String.raw`Since $\partial z/\partial w_j = x_j$ and $\partial z/\partial b = 1$, averaging over the data gives`,
      },
      { t: "math", tex: String.raw`\nabla_{\mathbf w}\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)\,\mathbf{x}_i = \frac{1}{N}X^\top(\mathbf{p} - \mathbf{y}),\qquad \frac{\partial\mathcal{L}}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)` },
      {
        t: "p",
        text: String.raw`The factor $\sigma'(z)$, which goes to 0 when the sigmoid flattens out, cancels. So a confidently wrong prediction ($p \approx 0$ for $y = 1$) gives an error signal near $-1$, the largest possible, not a vanishing one. With squared error $(p - y)^2$ and a sigmoid, the gradient would be $2(p-y)\,p(1-p)$ — nearly zero exactly when the model is most wrong — and that loss is not convex in $\mathbf{w}$.`,
      },
      { t: "h", text: "No closed form, but a convex problem" },
      {
        t: "p",
        text: String.raw`Setting the gradient to zero gives $\sum_i(\sigma(\mathbf{w}^\top\mathbf{x}_i + b) - y_i)\mathbf{x}_i = 0$: $d+1$ equations with the unknowns inside sigmoids. No algebraic solution exists, so we search step by step — gradient descent, Newton's method (iteratively reweighted least squares, in statistics), or quasi-Newton methods like L-BFGS, scikit-learn's default. The search is safe because the loss is convex. Differentiating once more gives the Hessian`,
      },
      { t: "math", tex: String.raw`\nabla^2_{\mathbf w}\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} p_i(1-p_i)\,\mathbf{x}_i\mathbf{x}_i^\top = \frac1N X^\top S X,\quad S = \operatorname{diag}\big(p_i(1-p_i)\big),\qquad \mathbf{v}^\top\nabla^2\mathcal{L}\,\mathbf{v} = \frac1N\sum_i p_i(1-p_i)(\mathbf{v}^\top\mathbf{x}_i)^2 \ge 0` },
      {
        t: "p",
        text: String.raw`Positive semi-definite everywhere, so every local minimum is global. One practical catch: **if a straight line separates the classes perfectly (linearly separable), there is no minimum at all.** Scaling any separating $(\mathbf{w}, b)$ by $c > 1$ pushes every $p_i$ closer to its label and lowers the loss, so gradient descent drives $\lVert\mathbf{w}\rVert \to \infty$. On four separable points (code-verified), $\lVert\mathbf{w}\rVert$ is 1.39 after 100 steps, 3.06 after 1,000, 4.72 after 10,000 and 6.35 after 100,000, while the loss falls tenfold per tenfold more steps — slow, endless growth in a fixed direction. L2 regularisation $\frac\lambda2\lVert\mathbf{w}\rVert^2$ (gradient $+\lambda\mathbf{w}$) makes the objective strictly convex with a finite minimiser: with $\lambda = 0.01$ the same run settles at $\lVert\mathbf{w}\rVert = 2.768$.`,
      },
      { t: "h", text: "Worked example: one gradient step" },
      {
        t: "p",
        text: String.raw`Four points $\mathbf{x} = (1,2), (2,1), (3,4), (4,3)$ with labels $y = (0, 0, 1, 1)$; start from $\mathbf{w} = (0.5, -0.25)$, $b = -0.5$, learning rate $\eta = 0.5$.`,
      },
      {
        t: "steps",
        items: [
          { title: "Logits", tex: String.raw`z_i = 0.5x_{i1} - 0.25x_{i2} - 0.5:\quad z = (-0.5,\ 0.25,\ 0,\ 0.75)` },
          { title: "Probabilities", tex: String.raw`p = \sigma(z) = (0.377541,\ 0.562177,\ 0.5,\ 0.679179)` },
          { title: "Loss", tex: String.raw`\mathcal L = -\tfrac14\big[\log(1-0.377541) + \log(1-0.562177) + \log 0.5 + \log 0.679179\big] = 0.595009` },
          { title: "Error signals", tex: String.raw`p - y = (0.377541,\ 0.562177,\ -0.5,\ -0.320821)` },
          { title: "Gradient", tex: String.raw`\nabla_{\mathbf w} = \tfrac14\textstyle\sum_i(p_i-y_i)\mathbf{x}_i = (-0.320348,\ -0.411302),\qquad \partial_b = \tfrac14\sum_i(p_i - y_i) = 0.029724` },
          { title: "Update", tex: String.raw`\mathbf w \leftarrow (0.5, -0.25) - 0.5(-0.320348, -0.411302) = (0.660174,\ -0.044349),\quad b \leftarrow -0.514862` },
          { title: "Check", tex: String.raw`p_{\text{new}} = (0.514150,\ 0.681600,\ 0.783853,\ 0.880038),\qquad \mathcal L_{\text{new}} = 0.559407 < 0.595009\ \checkmark` },
        ],
      },
      {
        t: "p",
        text: "The two class-1 points pulled $\\mathbf{w}$ toward themselves (both components rose), and the second class-0 point, wrongly above 0.5, gave the biggest push. One step doesn't fix it — the class-0 probabilities rose too, since the bias moved little — but the loss fell, as it must for a small enough step on a convex function.",
      },
      { t: "h", text: "Multiclass: softmax regression" },
      {
        t: "p",
        text: String.raw`For $K$ classes, give each class its own weights and bias: $z_k = \mathbf{w}_k^\top\mathbf{x} + b_k$, i.e. $\mathbf{z} = W^\top\mathbf{x} + \mathbf{b}$ with $W\in\mathbb{R}^{d\times K}$ — a layer of $K$ neurons. The [softmax](/learn/m03/activation-functions) turns the scores into probabilities that sum to 1, and the loss is categorical cross-entropy with one-hot targets $\mathbf{y}$ (1 for the true class, 0 elsewhere):`,
      },
      { t: "math", tex: String.raw`p_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},\qquad \ell = -\sum_{k=1}^{K}y_k\log p_k = -\log p_{c}\ (c = \text{true class}),\qquad \frac{\partial\ell}{\partial z_k} = p_k - y_k` },
      {
        t: "p",
        text: String.raw`Again "prediction minus target", so $\nabla_W = \frac1N X^\top(P - Y)$. Example: logits $\mathbf{z} = (2, 1, 0.1)$ give $\mathbf{p} = (0.659001, 0.242433, 0.098566)$; if the first class is true, $\ell = -\log 0.659001 = 0.417030$ and $\partial\ell/\partial\mathbf{z} = (-0.340999, 0.242433, 0.098566)$ — raise the correct score, lower the others in proportion to their probability. Adding one constant to all $z_k$ leaves $\mathbf{p}$ unchanged, so one class's parameters are redundant; with $K = 2$, $p_1 = \sigma(z_1 - z_0)$ and softmax regression is exactly logistic regression. The model has $dK + K$ parameters (15 for iris: 4 features, 3 classes), and its decision regions are convex polytopes bounded by the hyperplanes $z_j = z_k$.`,
      },
    ],

    // ---------------------------------------------------------------- visual
    visual: [
      {
        t: "p",
        text: "The simulator trains logistic regression on 2-D data with full-batch gradient descent from $\\mathbf{w} = 0$, $b = 0$. Shading is the model's probability $p(\\mathbf{x})$ (accent = class 1, clay = class 0, darker = surer); the solid line is the boundary $p = 0.5$, dashed lines are $p = 0.25$ and $0.75$, and the gold arrow is the direction of $\\mathbf{w}$. The right panel shows $\\mathbf{w}$, $b$, the gradient and the next update.",
      },
      { t: "sim", id: "logistic-regression", caption: "Every number shown is computed by the same update rule as the NumPy code: ∇w = (1/N)Σ(p − y)x + λw." },
      {
        t: "list",
        items: [
          "**Step 0.** The loss is $\\ln 2 = 0.693$ for any dataset: with $\\mathbf{w} = 0$ every probability is 0.5. Check that the first gradient points from class 1 toward class 0 (so the update points the other way).",
          "**Separable blobs, λ = 0.** Play to the end. Accuracy hits 100% in a few steps, but the loss keeps falling and $\\lVert\\mathbf{w}\\rVert$ keeps growing; the dashed lines close in on the boundary. Set λ = 0.05 — $\\lVert\\mathbf{w}\\rVert$ settles and the band stays wide.",
          "**Overlapping blobs.** The loss levels off above zero and the gradient shrinks to (almost) zero with finite weights: the true minimum of a convex function. Some points stay wrong — no line can do better.",
          "**Imbalanced (9 : 1).** Compare accuracy with the majority baseline in the panel, and read the minority-class recall in the narration. Add λ and watch the minority class lose out further.",
          "**Circles.** Logistic regression fails: the best line is nearly useless, and accuracy can be *below* the majority baseline because it minimises cross-entropy, not accuracy. Remember this for [Multi-Layer Networks](/learn/m03/multilayer-networks).",
          "**Learning rate.** On separable data, raise η to 10: big steps, but the loss still falls every step. On circles with η = 10, the loss goes *up* on many steps (flagged in the narration) — the step overshoots. Lower η to 0.01 and count how many more steps the same progress takes.",
          "**Click to add points.** Put a class-0 point deep in the class-1 cluster: it becomes the worst-classified point, gives the largest $(p - y)\\mathbf{x}$ term, and tilts the boundary. Undo removes it.",
        ],
      },
    ],

    // --------------------------------------------------------------- scratch
    scratch: [
      {
        t: "p",
        text: "Three complete models on real data, learning with NumPy only (scikit-learn just loads and splits the data). Part 1: linear regression on **diabetes** (442 patients, 10 baseline measurements, target = disease progression after a year), by normal equations and by gradient descent on raw vs standardised features. Part 2: L2-regularised logistic regression on **breast cancer Wisconsin** (569 tumours, 30 features from cell-nucleus images). Part 3: softmax regression on **iris** (150 flowers, 4 measurements, 3 species).",
      },
      { t: "code", lang: "python", title: "linear_logistic_from_scratch.py", code: LR_NUMPY, output: LR_NUMPY_OUT },
      { t: "p", text: "What to read in the output:" },
      {
        t: "list",
        items: [
          "**Exact formula vs gradient descent.** Both reach the same test MSE (3180.2) and $R^2$ (0.3594) — but only with standardised features. On raw features the condition number is $5.08\\times10^{7}$; even with the largest safe step ($1/\\lambda_{\\max}$) GD is 340 MSE units above the optimum after 5000 steps. Standardised: condition number 442.6, optimum reached.",
          "**$R^2 = 0.36$**: the 10 measurements explain about a third of the variance in progression. That is the data (outcomes depend on much that wasn't measured), not a fitting failure: the training residuals are exactly orthogonal to the features.",
          "`standardize` computes mean and standard deviation on the **training** split only and applies them to the test split. Using all the data would leak test information into training.",
          "**Logistic regression** reaches 95.8% test accuracy, with precision and recall 0.967 for benign. Training loss settles at 0.0484, not zero: with 30 features the data is nearly separable, and the L2 term (`lam = 1/N`) keeps weights finite. The largest weights are negative: bigger, more irregular nuclei lower the odds of *benign* — each standard deviation of `radius error` multiplies them by 0.297.",
          "**Softmax regression** uses the very same update with matrices: `Z_tr.T @ (P - Y) / N`. Subtracting the row maximum inside `softmax` keeps the probabilities but stops `exp` overflowing.",
        ],
      },
    ],

    // ------------------------------------------------------------- framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "In PyTorch both models are one `nn.Linear` layer (neurons with no activation). For linear regression that is the whole model. For logistic regression the sigmoid lives inside `nn.BCEWithLogitsLoss`, which computes $\\log(1+e^{-z})$ straight from the logit, stable for large $|z|$. L2 regularisation is the optimizer's `weight_decay`; two parameter groups leave the bias unregularised, matching the NumPy objective. Results match NumPy to the printed precision.",
              },
              { t: "code", lang: "python", title: "linear_logistic_pytorch.py", code: LR_TORCH, output: LR_TORCH_OUT },
              {
                t: "callout",
                kind: "tip",
                title: "Softmax regression in PyTorch",
                text: "`nn.Linear(d, K)` then `nn.CrossEntropyLoss()` on integer labels. The loss applies log-softmax inside, so output raw logits — no softmax layer.",
              },
            ],
          },
          {
            label: "scikit-learn",
            blocks: [
              {
                t: "p",
                text: "scikit-learn is the standard tool for these models outside deep learning. `LinearRegression` solves least squares directly (SVD-based), so no scaling is needed. `LogisticRegression` minimises $\\tfrac12\\lVert\\mathbf{w}\\rVert^2 + C\\sum_i \\ell_i$ with L-BFGS; dividing by $CN$ shows it equals our objective (mean BCE $+ \\tfrac{\\lambda}{2}\\lVert\\mathbf{w}\\rVert^2$) with $\\lambda = 1/(CN)$. The program checks both against NumPy.",
              },
              { t: "code", lang: "python", title: "linear_logistic_sklearn.py", code: LR_SKLEARN, output: LR_SKLEARN_OUT },
              {
                t: "p",
                text: "Linear coefficients agree to $4\\times10^{-11}$ — two exact solvers, same equations. Logistic coefficients agree to $5\\times10^{-7}$ at tight tolerance: three implementations, one convex objective, one answer.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "In Keras, logistic regression is a `Dense(1, activation=\"sigmoid\")` layer. Keras adds the regulariser to the *mean* loss, and `L2(l)` adds $l\\lVert\\mathbf{w}\\rVert^2$, so $l = \\lambda/2 = 0.5/N$ gives the objective above.",
              },
              {
                t: "code",
                lang: "python",
                title: "logistic_keras.py (not executed in CI)",
                code: LR_KERAS,
                output: "Prints the test BCE and accuracy after 3000 full-batch epochs; they should be close to the NumPy and PyTorch results (BCE ≈ 0.086, accuracy ≈ 0.958). Keras initialises the weights randomly rather than at zero, so tiny differences are expected before full convergence.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Variant", "What changes", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Ordinary least squares", "MSE, no penalty", "Exact formula, unbiased under standard assumptions, interpretable", "Unstable with collinear (highly correlated) features; sensitive to outliers", "Baselines, econometrics, calibration curves"],
          ["Ridge regression", "+ $\\lambda\\lVert\\mathbf{w}\\rVert_2^2$", "Always solvable; shrinks and steadies correlated weights", "Biased; $\\lambda$ must be tuned", "Many correlated features, $d \\gtrsim N$"],
          ["Lasso", "+ $\\lambda\\lVert\\mathbf{w}\\rVert_1$", "Sets many weights exactly to 0 (feature selection)", "No exact formula; picks one of a group of correlated features", "Sparse models, screening many features"],
          ["Elastic net", "L1 + L2", "Sparsity plus stability with correlated groups", "Two hyperparameters", "Genomics, text with many features"],
          ["Polynomial / basis regression", "Features $\\phi(x)$, still linear in $\\theta$", "Curved fits with the same solver", "Features hand-chosen; overfits at high degree", "Low-dimensional smooth relationships"],
          ["Robust regression (Huber)", "Huber loss instead of MSE", "Insensitive to outliers", "Iterative; extra threshold parameter", "Data with gross errors — see [Regression Losses](/learn/m05/regression-losses)"],
          ["Logistic regression", "Sigmoid + BCE", "Calibrated probabilities, convex, readable odds ratios", "Linear boundary only; no finite optimum on separable data without a penalty", "Risk scores, click prediction, binary baselines"],
          ["Softmax (multinomial) regression", "Softmax + CE over K classes", "One joint model, probabilities sum to 1", "Linear boundaries; $K\\times d$ weights", "Multiclass baselines, the output layer of classifiers"],
          ["One-vs-rest logistic", "K separate binary models", "Simple, parallel; handles multi-label", "Probabilities don't sum to 1; ambiguous regions", "Multi-label problems"],
          ["Probit regression", "Gaussian CDF instead of sigmoid", "Natural with latent Gaussian models", "No simple odds interpretation", "Econometrics, psychometrics"],
          ["Generalised linear models", "Other link/distribution pairs (e.g. Poisson with log link)", "Right likelihood for counts and rates", "More modelling choices", "Event counts, insurance claims"],
        ],
      },
    ],

    // ---------------------------------------------------------- applications
    applications: [
      {
        t: "list",
        items: [
          "**Clinical risk scores.** Many medical risk calculators are logistic regressions: each risk factor adds to the log-odds, so the model is auditable and easy to turn into a points table.",
          "**Credit scoring.** Lenders have long used logistic-regression scorecards for default probability, partly because regulators require decisions to be explainable from the inputs.",
          "**Click-through-rate prediction.** Large online ad systems have used logistic regression over billions of sparse features; McMahan et al.'s 2013 paper \"Ad Click Prediction: a View from the Trenches\" describes one at Google, trained online.",
          "**Probability calibration.** Platt scaling fits a one-feature logistic regression to a classifier's raw scores to get calibrated probabilities.",
          "**Linear probes.** Researchers train a logistic or softmax regression on a frozen network's hidden activations to measure what those features contain.",
          "**The last layer of almost every network.** A classifier's output layer is softmax regression on learned features; a regression network's is linear regression on learned features.",
          "**Baselines.** Before a deep model, a well-tuned linear model shows how much of the problem is easy. On tabular data it is often hard to beat.",
        ],
      },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Convex training:** one (or flat-bottomed) global optimum; no initialisation lottery, reproducible results.",
          "**Fast:** exact formula for least squares; a few hundred cheap iterations for logistic regression. Handles millions of examples and sparse features.",
          "**Interpretable:** each weight is a per-unit effect on the prediction (linear) or log-odds (logistic), other features fixed.",
          "**Calibrated probabilities:** logistic regression with cross-entropy tends to give probabilities that match observed frequencies, when the model fits the data's true form.",
          "**Little data needed:** with $d+1$ parameters, hard to overfit unless $d$ is large relative to $N$.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Linear only:** predictions are linear in the features and the boundary is a hyperplane. XOR, circles and images are out of reach without hand-made features.",
          "**Feature engineering burden:** you must supply the non-linearity (polynomials, interactions, binning). Deep networks learn such features.",
          "**Sensitivity:** outliers pull least squares; correlated features make single weights unstable and hard to interpret even when predictions are fine.",
          "**Separable data:** unregularised logistic regression has no finite solution.",
          "**Correlation is not causation:** a weight describes an association in the training data, not what happens if you change that feature.",
        ],
      },
      { t: "h", text: "Practical trade-offs" },
      {
        t: "list",
        items: [
          "**Exact formula vs iterative:** the normal equations cost $O(Nd^2 + d^3)$ and need $\\tilde X^\\top\\tilde X$ in memory; for $d$ in the tens of thousands, or streaming data, use (stochastic) gradient descent.",
          "**Regularisation strength** trades bias for variance; pick $\\lambda$ (or $C$) on validation data, never the test set.",
          "**Threshold choice** trades precision against recall; 0.5 is right only if both errors cost the same and classes are balanced.",
        ],
      },
    ],

    // -------------------------------------------------------------- mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Gradient descent on unscaled features.** A feature in thousands next to one in fractions makes the loss a narrow valley; GD diverges or crawls. *Fix:* standardise (or min–max scale) with training statistics.",
          "**Fitting the scaler on all the data.** Using the test set's mean and standard deviation leaks information and inflates test scores. *Fix:* fit on the training split, then transform validation and test with those numbers (scikit-learn's `Pipeline` does this).",
          "**Inverting $X^\\top X$ explicitly.** `np.linalg.inv(X.T @ X) @ X.T @ y` squares the condition number and loses precision. *Fix:* `np.linalg.solve(X.T @ X, X.T @ y)`, or better `np.linalg.lstsq(X, y)`, which works on $X$ directly and handles redundant columns.",
          "**Forgetting the bias.** Without a column of ones the line must pass through the origin. *Fix:* add the column, or use the framework's built-in bias.",
          "**Using linear regression (MSE) for 0/1 labels.** Predictions leave $[0,1]$, and easy far-away points distort the boundary (the outlier example above). *Fix:* logistic regression.",
          "**Applying a sigmoid and then `BCEWithLogitsLoss`.** The loss adds its own sigmoid, so you get $\\sigma(\\sigma(z))$: probabilities squashed into $(0.5, 0.73)$ and wrong gradients. *Fix:* return raw logits.",
          "**Trusting accuracy on imbalanced data.** With 95% negatives, always saying \"negative\" is 95% accurate. *Fix:* report precision, recall, the confusion matrix or ROC/PR curves — see [Classification Metrics](/learn/m17/classification-metrics).",
          "**Expecting convergence on separable data.** Without regularisation the weights grow forever, and \"converged\" coefficients depend on when you stopped. *Fix:* L2 (scikit-learn's default) or early stopping.",
          "**The dummy-variable trap.** One-hot encoding every category *and* keeping an intercept makes the columns linearly dependent, so $X^\\top X$ is singular. *Fix:* drop one category or regularise.",
          "**Reading weights of correlated features.** Two near-identical features can get large, opposite weights while their sum is stable. *Fix:* interpret them as a group, use ridge, or drop one.",
        ],
      },
    ],

    // ------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between linear and logistic regression?", a: "Linear regression predicts a real number, $\\hat y = \\mathbf{w}^\\top\\mathbf{x} + b$, trained with mean squared error. Logistic regression predicts the probability of class 1, $p = \\sigma(\\mathbf{w}^\\top\\mathbf{x} + b)$, trained with binary cross-entropy. Both are linear in the parameters; logistic regression's boundary $\\mathbf{w}^\\top\\mathbf{x} + b = 0$ is a hyperplane. Despite the name, logistic regression is a classifier." },
      { t: "qa", level: "beginner", q: "Why do we square the errors in linear regression?", a: "Squaring makes every error positive (no cancelling), punishes large errors more, and gives a smooth convex loss whose minimum has an exact formula. It is also the maximum-likelihood loss under Gaussian noise. Downside: sensitivity to outliers; absolute or Huber loss are alternatives." },
      { t: "qa", level: "beginner", q: "What does the sigmoid do, and what is its derivative?", a: "$\\sigma(z) = 1/(1+e^{-z})$ maps any real score to $(0,1)$ so it can be read as a probability; $\\sigma(0) = 0.5$ and $\\sigma(-z) = 1-\\sigma(z)$. Its derivative is $\\sigma(z)(1-\\sigma(z))$, at most 0.25 (at $z = 0$)." },
      { t: "qa", level: "beginner", q: "Why should features be standardised before gradient descent?", a: "Features on very different scales give the loss very different curvatures in different directions (a high condition number). A step small enough for the steep direction barely moves along the flat one. Standardising evens out the scales, lowers the condition number and lets GD converge in far fewer steps. It doesn't change the best possible fit." },
      { t: "qa", level: "intermediate", q: "Derive the normal equations.", a: "With $\\mathcal{L} = \\frac1N\\lVert X\\boldsymbol\\theta - \\mathbf{y}\\rVert^2$ (bias in $X$), $\\nabla\\mathcal{L} = \\frac2N X^\\top(X\\boldsymbol\\theta - \\mathbf{y})$. The loss is convex (Hessian $\\frac2N X^\\top X \\succeq 0$), so the minimum is where the gradient is zero: $X^\\top X\\boldsymbol\\theta = X^\\top\\mathbf{y}$, so $\\boldsymbol\\theta = (X^\\top X)^{-1}X^\\top\\mathbf{y}$ when $X$ has full column rank. Geometrically, the residual is orthogonal to the column space of $X$." },
      { t: "qa", level: "intermediate", q: "Derive the gradient of binary cross-entropy for logistic regression.", a: "For one example, $\\ell = -[y\\log p + (1-y)\\log(1-p)]$, $p = \\sigma(z)$. $\\partial\\ell/\\partial p = -y/p + (1-y)/(1-p)$ and $\\partial p/\\partial z = p(1-p)$, so $\\partial\\ell/\\partial z = -y(1-p) + (1-y)p = p - y$. With $z = \\mathbf{w}^\\top\\mathbf{x} + b$: $\\nabla_{\\mathbf w}\\ell = (p-y)\\mathbf{x}$, $\\partial\\ell/\\partial b = p - y$; average over the data." },
      { t: "qa", level: "intermediate", q: "Why is logistic regression trained with cross-entropy rather than MSE?", a: "Cross-entropy is the Bernoulli negative log-likelihood, so minimising it is maximum likelihood. With a sigmoid it is convex in the weights, and its gradient $p - y$ stays large when the model is confidently wrong. MSE on sigmoid outputs is non-convex, and its gradient $2(p-y)p(1-p)$ vanishes when $p$ sticks near the wrong label, so learning stalls." },
      { t: "qa", level: "intermediate", q: "How do you interpret a logistic regression coefficient of 0.7?", a: "One more unit of that feature (others fixed) adds 0.7 to the log-odds of class 1, i.e. multiplies the odds by $e^{0.7} \\approx 2.01$. It does not add a fixed amount to the probability: the change in $p$ depends on where you are on the sigmoid, and is largest near $p = 0.5$." },
      { t: "qa", level: "advanced", q: "What happens when you train unregularised logistic regression on linearly separable data?", a: "There is no finite minimiser: scaling any separating $(\\mathbf{w}, b)$ by $c > 1$ lowers every example's loss, and the lowest value, 0, is approached only as $\\lVert\\mathbf{w}\\rVert \\to \\infty$. The weights grow roughly logarithmically in the number of steps while their direction settles (for GD on separable data, toward the maximum-margin direction). Predictions become overconfident. L2 regularisation or early stopping restores a finite solution." },
      { t: "qa", level: "advanced", q: "Show that the logistic regression loss is convex, and say when the minimiser is unique.", a: "The Hessian is $\\frac1N\\sum_i p_i(1-p_i)\\mathbf{x}_i\\mathbf{x}_i^\\top = \\frac1N X^\\top S X$ with $S$ diagonal and positive, so $\\mathbf{v}^\\top H\\mathbf{v} = \\frac1N\\sum_i p_i(1-p_i)(\\mathbf{v}^\\top\\mathbf{x}_i)^2 \\ge 0$: convex. Positive definite if $X$ has full column rank (strict convexity) — but a minimiser may not exist (separable data). With L2, $H + \\lambda I \\succ 0$: strongly convex with a unique minimiser." },
      { t: "qa", level: "advanced", q: "Why is softmax regression over-parameterised, and how does it relate to logistic regression?", a: "Adding the same vector to every class's weights (and the same constant to every bias) changes nothing, because $e^{z_k + c}/\\sum_j e^{z_j + c} = e^{z_k}/\\sum_j e^{z_j}$. So one class's parameters can be fixed at zero with no loss; regularisation picks a unique representative. With $K = 2$, $p_1 = e^{z_1}/(e^{z_0} + e^{z_1}) = \\sigma(z_1 - z_0)$: logistic regression with weights $\\mathbf{w}_1 - \\mathbf{w}_0$." },
    ],

    // -------------------------------------------------------------- practice
    practice: [
      {
        t: "quiz",
        questions: [
          { kind: "numeric", q: "For $x = (0, 1, 2)$ and $y = (1, 3, 5)$, least squares gives $\\hat y = b + wx$. What is $w$?", answer: 2, tolerance: 0.001, explain: "The points lie exactly on $y = 1 + 2x$, so the residuals can be made zero: $w = 2$, $b = 1$. (The normal equations $\\begin{pmatrix}3&3\\\\3&5\\end{pmatrix}\\theta = \\begin{pmatrix}9\\\\13\\end{pmatrix}$ give the same.)" },
          { kind: "numeric", q: "What is $\\sigma(\\ln 3)$?", answer: 0.75, tolerance: 0.001, explain: "$\\sigma(\\ln 3) = 1/(1 + e^{-\\ln 3}) = 1/(1 + 1/3) = 0.75$. Equivalently, log-odds $\\ln 3$ means odds 3 : 1." },
          { kind: "numeric", q: "A logistic regression predicts $p = 0.8$ for an example with $y = 1$ and a single feature $x = 2$. What is $\\partial\\ell/\\partial w$ for this example?", answer: -0.4, tolerance: 0.001, explain: "$(p - y)x = (0.8 - 1)\\cdot2 = -0.4$. Negative: increasing $w$ lowers the loss." },
          { kind: "numeric", q: "A logistic-regression weight is $w_j = 0.7$. By what factor are the odds multiplied when $x_j$ increases by 1? (3 decimals)", answer: 2.014, tolerance: 0.002, explain: "Odds multiply by $e^{w_j} = e^{0.7} = 2.014$." },
          { kind: "numeric", q: "How many trainable parameters does softmax regression have with 4 input features and 3 classes?", answer: 15, tolerance: 0, explain: "A $4\\times3$ weight matrix (12) plus 3 biases = 15." },
          { q: "Why does gradient descent converge slowly on unscaled features?", options: ["The loss is no longer convex", "The Hessian is badly conditioned, so a step size safe for the steepest direction is tiny along the flattest one", "The normal equations have no solution", "The gradient becomes zero"], answer: 1, explain: "Scaling changes neither convexity nor the optimum; it changes the condition number $\\lambda_{\\max}/\\lambda_{\\min}$, which sets how many steps GD needs." },
          { q: "Unregularised logistic regression is trained on linearly separable data with gradient descent. What happens?", options: ["It converges to a finite optimum with zero loss", "The loss decreases forever while $\\lVert\\mathbf{w}\\rVert$ grows without bound", "It diverges to infinite loss", "It gets stuck in a local minimum"], answer: 1, explain: "Scaling up a separating $\\mathbf{w}$ always lowers the loss, so there is no finite minimiser; the loss tends to 0 and the weights to infinity." },
          { q: "Which statement about polynomial regression with features $1, x, x^2, x^3$ is true?", options: ["It is non-linear in its parameters, so it needs gradient descent", "It is linear in its parameters and is solved by the normal equations on the expanded design matrix", "It can only fit straight lines", "It always generalises better than linear regression"], answer: 1, explain: "The model is $\\sum_k\\theta_k x^k$ — linear in $\\theta$. Only the features are non-linear in $x$." },
        ],
      },
      {
        t: "exercise",
        title: "Least squares and one gradient step by hand",
        prompt: "Data $x = (0, 1, 2)$, $y = (1, 2, 6)$, model $\\hat y = b + wx$. (a) Form and solve the normal equations. (b) Give the residuals and the MSE, and check that the residuals are orthogonal to both columns of the design matrix. (c) Starting from $b = w = 0$, compute the MSE gradient.",
        hint: "$\\tilde X^\\top\\tilde X = \\begin{pmatrix}N & \\sum x\\\\ \\sum x & \\sum x^2\\end{pmatrix}$, and the gradient is $\\frac{2}{N}\\tilde X^\\top(\\hat{\\mathbf{y}} - \\mathbf{y})$.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "(a) Normal equations", tex: String.raw`\begin{pmatrix}3 & 3\\ 3 & 5\end{pmatrix}\begin{pmatrix}b\\ w\end{pmatrix} = \begin{pmatrix}9\\ 14\end{pmatrix},\quad \det = 6,\quad \begin{pmatrix}b\\ w\end{pmatrix} = \frac16\begin{pmatrix}5\cdot9 - 3\cdot14\\ -3\cdot9 + 3\cdot14\end{pmatrix} = \begin{pmatrix}0.5\\ 2.5\end{pmatrix}` },
              { title: "(b) Residuals", tex: String.raw`\hat{\mathbf y} = (0.5,\ 3,\ 5.5),\quad \hat{\mathbf y} - \mathbf y = (-0.5,\ 1,\ -0.5),\quad \text{MSE} = \tfrac{0.25 + 1 + 0.25}{3} = 0.5` },
              { title: "Orthogonality", tex: String.raw`\textstyle\sum r_i = -0.5 + 1 - 0.5 = 0,\qquad \sum r_ix_i = 0 + 1 - 1 = 0\ \checkmark` },
              { title: "(c) Gradient at zero", tex: String.raw`\hat{\mathbf y} - \mathbf y = (-1, -2, -6):\quad \frac{\partial\mathcal L}{\partial b} = \tfrac23(-9) = -6,\qquad \frac{\partial\mathcal L}{\partial w} = \tfrac23(0 - 2 - 12) = -9.333` },
            ],
          },
        ],
      },
      {
        t: "exercise",
        title: "Polynomial features",
        prompt: "Generate 40 points with $x \\sim U(-1, 1)$ and $y = 1 + 2x - 3x^2 + \\varepsilon$, $\\varepsilon\\sim\\mathcal{N}(0, 0.2^2)$ (seed 0). Fit least squares with polynomial features of degree 1, 2 and 10, and report training MSE and the MSE against the noiseless curve on a dense grid. Which degree would you choose, and why does degree 10 have the lowest training error but not the lowest test error?",
        hint: "`np.vander(x, degree + 1, increasing=True)` builds the columns $1, x, \\dots, x^{\\text{degree}}$; `np.linalg.lstsq` solves the least-squares problem.",
        solution: [
          { t: "code", lang: "python", title: "polynomial_features.py", code: POLY_CODE, output: POLY_OUT },
          {
            t: "p",
            text: "Degree 1 underfits: a line can't follow a parabola (both errors near 0.9). Degree 2 matches the true model; its coefficients (1.032, 2.05, −2.933) are close to (1, 2, −3), and its test error 0.0042 is far below the noise variance 0.04, because averaging over 40 points cancels much of the noise. Degree 10 has more freedom, so its training error is lower (0.0386) — it fits some noise — but its error against the true curve is almost three times higher (0.0113). Choose degree 2, ideally confirmed on a validation set. It is still *linear* regression: only the features changed.",
          },
        ],
      },
    ],

    // --------------------------------------------------------------- summary
    summary: [
      {
        t: "list",
        items: [
          "Linear regression: $\\hat y = \\mathbf{w}^\\top\\mathbf{x} + b$, loss $\\frac1N\\lVert\\tilde X\\boldsymbol\\theta - \\mathbf{y}\\rVert^2$, gradient $\\frac2N\\tilde X^\\top(\\tilde X\\boldsymbol\\theta - \\mathbf{y})$.",
          "Normal equations $\\tilde X^\\top\\tilde X\\boldsymbol\\theta = \\tilde X^\\top\\mathbf{y}$: residuals orthogonal to the features; prediction = projection of $\\mathbf{y}$. Solve with `solve`/`lstsq`, not `inv`.",
          "Least squares = maximum likelihood under Gaussian noise; ridge = Gaussian prior.",
          "Gradient descent speed depends on the Hessian's condition number; standardise features (with training statistics) before GD.",
          "Polynomial / basis features give curved fits while staying linear in the parameters.",
          "Logistic regression: $p = \\sigma(\\mathbf{w}^\\top\\mathbf{x}+b)$; log-odds $\\log\\frac{p}{1-p}$ are linear; $e^{w_j}$ is an odds ratio; boundary $\\mathbf{w}^\\top\\mathbf{x}+b = 0$ is a hyperplane.",
          "BCE $= -\\frac1N\\sum[y\\log p + (1-y)\\log(1-p)]$ is the Bernoulli negative log-likelihood; its gradient is $\\frac1N X^\\top(\\mathbf{p} - \\mathbf{y})$. Convex, no closed form; on separable data no finite optimum without regularisation.",
          "Softmax regression: $K$ neurons + softmax + cross-entropy; gradient $\\frac1N X^\\top(P - Y)$; reduces to logistic regression for $K = 2$.",
          "Each model is one neuron (or one layer): identity + MSE, sigmoid + BCE, softmax + CE — with error signal *prediction − target* in every case.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "p", text: "You now have a trainable neuron. The next lessons take it apart and then stack it:" },
      {
        t: "list",
        items: [
          "[The Artificial Neuron and the Perceptron](/learn/m03/artificial-neuron) — the same weighted sum with a hard threshold, its historical learning rule, and a proof that one neuron can't compute XOR.",
          "[Activation Functions](/learn/m03/activation-functions) — identity and sigmoid are two of many choices, and the choice matters once layers are stacked.",
          "[Multi-Layer Networks and Model Capacity](/learn/m03/multilayer-networks) — hidden layers *learn* the features you hand-crafted here (polynomials, $x_1^2 + x_2^2$ for the circles), and the universal approximation theorem says how far that goes.",
          "[Classification Losses](/learn/m05/classification-losses) — cross-entropy in depth, including label smoothing and focal loss for the imbalance seen in the simulator.",
        ],
      },
    ],
  },
};

export default lesson;
