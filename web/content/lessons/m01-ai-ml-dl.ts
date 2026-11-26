import type { Lesson } from "@/lib/types";

// ---------------------------------------------------------------------------
// Code blocks (all executed; outputs pasted verbatim)
// ---------------------------------------------------------------------------

const RULES_VS_LEARNING = String.raw`import numpy as np
from sklearn.datasets import load_breast_cancer   # used ONLY to load the bundled dataset

# ---------------------------------------------------------------
# Data: 569 tumour samples, 30 measurements each (computed from a
# microscope image of cells). Label: 0 = malignant, 1 = benign.
# ---------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
names = [str(n) for n in data.feature_names]
malignant = (y == 0).astype(int)          # 1 = malignant (the class we want to flag)

# Same random train/test split for both approaches (fixed seed -> reproducible)
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
n_test = 143                              # 25% of 569 held out for testing
test, train = idx[:n_test], idx[n_test:]
print(f"train samples: {len(train)}, test samples: {len(test)}")

# ===============================================================
# 1) TRADITIONAL PROGRAMMING: a human writes the rule
#    rules + data -> answers
# ===============================================================
radius = X[:, names.index("mean radius")]
concave = X[:, names.index("mean concave points")]

def hand_written_rule(radius, concave):
    """Expert-style rule: large cells OR many concave points on the
    cell boundary are suspicious. Thresholds 15 and 0.06 were picked
    by a person eyeballing typical values - nothing was optimised."""
    return ((radius > 15.0) | (concave > 0.06)).astype(int)

rule_pred = hand_written_rule(radius[test], concave[test])
print(f"hand-written rule  test accuracy: {np.mean(rule_pred == malignant[test]):.3f}")

# ===============================================================
# 2) MACHINE LEARNING: the computer learns the rule
#    data + answers -> rules (here: 30 weights + 1 bias)
# ===============================================================
# Standardise every feature with statistics from the TRAINING set only
mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
Xs = (X - mu) / sd

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

w = np.zeros(X.shape[1])                  # 30 weights, start at zero
b = 0.0
lr = 0.1                                  # learning rate (step size)
for epoch in range(500):                  # 500 full passes of gradient descent
    p = sigmoid(Xs[train] @ w + b)        # predicted P(malignant)
    err = p - malignant[train]            # gradient of cross-entropy w.r.t. z
    w -= lr * Xs[train].T @ err / len(train)
    b -= lr * err.mean()

learned_pred = (sigmoid(Xs[test] @ w + b) > 0.5).astype(int)
print(f"learned (logistic regression) test accuracy: {np.mean(learned_pred == malignant[test]):.3f}")

# The learned "rule" is just numbers. Show the 3 largest-magnitude weights:
top = np.argsort(-np.abs(w))[:3]
for i in top:
    print(f"  weight on {names[i]!r}: {w[i]:+.3f}")

# Baseline: always predict the most common class in the training set
majority = int(malignant[train].mean() > 0.5)
print(f"always-predict-majority test accuracy: {np.mean(majority == malignant[test]):.3f}")`;

const RULES_VS_LEARNING_OUT = String.raw`train samples: 426, test samples: 143
hand-written rule  test accuracy: 0.888
learned (logistic regression) test accuracy: 0.979
  weight on 'radius error': +0.708
  weight on 'worst radius': +0.699
  weight on 'worst texture': +0.673
always-predict-majority test accuracy: 0.643`;

const DIGITS_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

# 1,797 handwritten digits, each an 8x8 grey image (pixel values 0..16)
digits = load_digits()
images, labels = digits.images, digits.target
X_tr_img, X_te_img, y_tr, y_te = train_test_split(images, labels, test_size=0.3, random_state=0, stratify=labels)
print(f"train images: {len(X_tr_img)}, test images: {len(X_te_img)}, image size: {images.shape[1:]}")

# ---------- Option A: raw pixels (64 numbers per image) ----------
def raw_pixels(imgs):
    return imgs.reshape(len(imgs), -1) / 16.0

# ---------- Option B: hand-engineered features (what a human designs) ----------
def hand_features(imgs):
    imgs = imgs / 16.0
    rows = imgs.sum(axis=2)                                   # ink in each of 8 rows
    cols = imgs.sum(axis=1)                                   # ink in each of 8 columns
    total = imgs.sum(axis=(1, 2))[:, None]                    # total ink
    lr_sym = np.abs(imgs - imgs[:, :, ::-1]).sum(axis=(1, 2))[:, None]   # left-right asymmetry
    tb_sym = np.abs(imgs - imgs[:, ::-1, :]).sum(axis=(1, 2))[:, None]   # top-bottom asymmetry
    return np.hstack([rows, cols, total, lr_sym, tb_sym])     # 8 + 8 + 1 + 1 + 1 = 19 features

def standardise(train, test):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd

def fit_and_score(model, Xtr, Xte, epochs=300):
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr, yte = torch.tensor(y_tr), torch.tensor(y_te)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):                                   # full-batch training
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()

experiments = []
Xtr, Xte = standardise(raw_pixels(X_tr_img), raw_pixels(X_te_img))
experiments.append(("logistic regression, raw pixels (64)", nn.Linear(64, 10), Xtr, Xte))
Ftr, Fte = standardise(hand_features(X_tr_img), hand_features(X_te_img))
experiments.append(("logistic regression, hand features (19)", nn.Linear(19, 10), Ftr, Fte))
mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
experiments.append(("MLP 64-64-64-10, raw pixels (learns features)", mlp, Xtr, Xte))

for name, model, a, b in experiments:
    n_params = sum(p.numel() for p in model.parameters())
    acc = fit_and_score(model, a, b)
    print(f"{name:<47} params={n_params:>5}  test accuracy={acc:.3f}")`;

const DIGITS_TORCH_OUT = String.raw`train images: 1257, test images: 540, image size: (8, 8)
logistic regression, raw pixels (64)            params=  650  test accuracy=0.956
logistic regression, hand features (19)         params=  200  test accuracy=0.896
MLP 64-64-64-10, raw pixels (learns features)   params= 8970  test accuracy=0.969`;

const DIGITS_KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, random_state=0, stratify=y)

def build(hidden_layers):
    model = tf.keras.Sequential([tf.keras.Input(shape=(64,))])
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10))                      # 10 logits, one per digit
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model

for name, hidden in [("logistic regression (0 hidden layers)", 0), ("MLP (2 hidden layers)", 2)]:
    model = build(hidden)
    model.fit(X_tr, y_tr, epochs=100, batch_size=64, verbose=0)
    print(name, "test accuracy:", round(model.evaluate(X_te, y_te, verbose=0)[1], 3))`;

const PARADIGMS = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans

X, y = load_digits(return_X_y=True)        # 1,797 images, 64 pixels each, labels 0-9
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# 1) SUPERVISED: learn image -> label from labelled examples
clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
print(f"supervised      : test accuracy = {clf.score(X_te, y_te):.3f}")

# 2) UNSUPERVISED: no labels at all - group similar images into 10 clusters.
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X_tr)
# Labels are used ONLY afterwards, to check how "pure" each discovered cluster is.
purity = sum(np.bincount(y_tr[km.labels_ == c]).max() for c in range(10)) / len(y_tr)
print(f"unsupervised    : cluster purity = {purity:.3f}  (labels never used for training)")

# 3) SELF-SUPERVISED: invent a task whose 'labels' come from the data itself.
#    Task: given the top half of an image (rows 0-3), predict the bottom half (rows 4-7).
top_tr, bot_tr = X_tr[:, :32], X_tr[:, 32:]
top_te, bot_te = X_te[:, :32], X_te[:, 32:]
reg = Ridge(alpha=1.0).fit(top_tr, bot_tr)
mse_model = np.mean((reg.predict(top_te) - bot_te) ** 2)
mse_mean = np.mean((bot_tr.mean(0) - bot_te) ** 2)          # baseline: always predict the average bottom half
print(f"self-supervised : bottom-half MSE = {mse_model:.4f}  vs  {mse_mean:.4f} for 'predict the average'")

# 4) REINFORCEMENT: learn by trial, error and reward (a 3-armed bandit).
rng = np.random.default_rng(0)
true_payout = np.array([0.2, 0.5, 0.8])      # hidden from the agent
estimates, pulls, total_reward = np.zeros(3), np.zeros(3), 0.0
for t in range(1000):
    # explore 10% of the time, otherwise exploit the best estimate so far
    arm = rng.integers(3) if rng.random() < 0.1 else int(np.argmax(estimates))
    reward = float(rng.random() < true_payout[arm])           # 1 or 0
    pulls[arm] += 1
    estimates[arm] += (reward - estimates[arm]) / pulls[arm]   # running average
    total_reward += reward
print(f"reinforcement   : estimates = {np.round(estimates, 2)}, pulls = {pulls.astype(int)}, average reward = {total_reward / 1000:.3f}")`;

const PARADIGMS_OUT = String.raw`supervised      : test accuracy = 0.970
unsupervised    : cluster purity = 0.797  (labels never used for training)
self-supervised : bottom-half MSE = 0.0528  vs  0.0783 for 'predict the average'
reinforcement   : estimates = [0.19 0.41 0.8 ], pulls = [ 42  32 926], average reward = 0.765`;

const EX_ZERO_ONE = String.raw`import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()
mask = (digits.target == 0) | (digits.target == 1)          # keep only zeros and ones
imgs, y = digits.images[mask] / 16.0, (digits.target[mask] == 1).astype(int)
img_tr, img_te, y_tr, y_te = train_test_split(imgs, y, test_size=0.3, random_state=0, stratify=y)

# Hand-written rule: a "0" is a ring, so its middle columns (3 and 4) are
# nearly empty in the central rows; a "1" is a stroke through the middle.
def rule(imgs):
    centre_ink = imgs[:, 2:6, 3:5].sum(axis=(1, 2))           # 4 rows x 2 columns = 8 pixels
    return (centre_ink > 2.0).astype(int)                     # lots of ink in the centre -> "1"

print(f"hand-written rule accuracy  : {np.mean(rule(img_te) == y_te):.3f}")

clf = LogisticRegression(max_iter=1000).fit(img_tr.reshape(len(img_tr), -1), y_tr)
print(f"logistic regression accuracy: {clf.score(img_te.reshape(len(img_te), -1), y_te):.3f}")
print(f"test images: {len(y_te)}")`;

const EX_ZERO_ONE_OUT = String.raw`hand-written rule accuracy  : 0.963
logistic regression accuracy: 1.000
test images: 108`;

// ---------------------------------------------------------------------------

const lesson: Lesson = {
  id: "m01/ai-ml-dl",
  title: "AI, Machine Learning and Deep Learning",
  summary:
    "What the three terms really mean, how they fit inside each other, how learning from data differs from writing rules by hand — measured on real datasets — and what deep learning is and isn't good at.",
  minutes: 50,
  level: "beginner",
  prerequisites: [],
  next: ["m01/history", "m01/dl-workflow", "m03/linear-logistic-regression"],
  terms: [
    "artificial-intelligence",
    "machine-learning",
    "deep-learning",
    "narrow-ai",
    "agi",
    "supervised-learning",
    "unsupervised-learning",
    "self-supervised-learning",
    "reinforcement-learning",
    "feature-engineering",
    "representation-learning",
    "parameter",
    "overfitting",
  ],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "Welcome to the course! When computers do something that looks smart, three phrases get mixed up: **artificial intelligence (AI)**, **machine learning (ML)** and **deep learning (DL)**. They are not the same thing — each fits inside the one before it. We define each one, see how they fit together, and then *measure* the difference on real data." },
      { t: "list", items: [
        "**Artificial intelligence** is the big goal: computer systems that do tasks we would call intelligent if a person did them — recognising speech, planning a route, playing chess, finding a fault, answering a question. AI says nothing about *how*. A hand-written rulebook counts; so does a search algorithm, and so does a neural network.",
        "**Machine learning** is one way to build AI. Instead of a programmer writing the decision rules, the computer **learns** them from **data** (examples). The programmer writes the *learning procedure*; the data decides the final behaviour.",
        "**Deep learning** is one family of machine-learning methods. It uses **neural networks with many layers**, where each layer transforms the previous layer's output. The network learns its own internal **representations** (useful ways of describing the input) instead of relying on a human to design them.",
      ] },
      { t: "p", text: "Why does the difference matter? Each step down trades human effort for data and computing power:" },
      { t: "list", items: [
        "A rule-based system needs an expert who can put the rules into words.",
        "A classical machine-learning system needs labelled data *and* an expert who designs good input measurements (\"features\").",
        "A deep-learning system needs much more data and computing, but learns the features itself. That is why it dominates images, speech and language, where nobody can write the rules or features down.",
      ] },
      { t: "callout", kind: "key", title: "In one sentence", text: "AI is the goal (intelligent behaviour), machine learning is a strategy for reaching it (learn the behaviour from data), and deep learning is a particular machine-learning toolkit (many-layered neural networks that learn their own features)." },
      { t: "p", text: "Everything else in this course — neurons, backpropagation, CNNs, Transformers, diffusion models — lives in the innermost circle. This lesson is your map." },
    ],

    // --------------------------------------------------------- prerequisites
    prerequisites: [
      { t: "p", text: "This is the first lesson of the course. You don't need any machine-learning background. You need:" },
      { t: "list", items: [
        "**What a function is**: something that takes an input and gives back an output, like $f(x) = 2x + 1$. Every model in this course is a function from inputs (an image, a sentence, a table row) to outputs (a label, a number, the next word).",
        "**School arithmetic**: multiplying, adding, and the idea of an average. The one new-looking formula here (the sigmoid) is explained when it appears.",
        "**Reading a little Python** (optional but recommended): the code uses Python with NumPy. If arrays and shapes are new to you, skim [NumPy Arrays, Shapes and Indexing](/learn/m00/numpy-arrays) first — but you can follow this lesson without running anything.",
      ] },
      { t: "callout", kind: "tip", text: "Don't worry if terms like *gradient descent*, *logistic regression* or *neural network* show up before you have studied them. They are explained in plain words here, and each has its own full lesson later — the links take you there." },
    ],

    // ------------------------------------------------------------- intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "The rulebook versus the apprentice", text: "Imagine sorting mushrooms into *safe* and *poisonous*. **Traditional programming** is writing a rulebook: \"if the cap is white and the gills are free and there is a ring on the stem, then…\". You must know every rule yourself and write it down. **Machine learning** is training an apprentice. You show them thousands of mushrooms that experts have already labelled, and they work out the rules on their own. You supply examples and check how often they are right on new mushrooms. **Deep learning** is an apprentice who isn't even told *what to look at*. You hand over photos instead of a checklist of cap colour and gill shape, and they discover which visual details matter." },
      { t: "p", text: "The analogy also shows the costs. The rulebook needs no examples, but only works if an expert can state the rules — and fails silently on cases the expert forgot. The apprentice needs many labelled examples and can learn the wrong lesson from misleading ones (say, if every poisonous mushroom in the photos was shot on a rainy day). The photo-only apprentice needs the most examples, but can find patterns no human thought to write down." },
      { t: "callout", kind: "analogy", title: "Nesting dolls", text: "Picture Russian nesting dolls. The biggest is **AI** — everything that makes machines act intelligently. Inside it is **machine learning** — AI that learns from data. Inside that, **neural networks** — ML models built from simple connected units. The smallest is **deep learning** — neural networks with many layers. Each doll is part of the one around it, but not the other way round: plenty of AI involves no learning at all." },
      { t: "p", text: "One more idea: a learned model is **just numbers**. \"Learned the rule\" means a training algorithm adjusted numbers called **parameters** (or **weights**) until the function they define gives good answers on the examples. A logistic-regression model for 30 measurements has 31 such numbers; a large language model has billions. Same principle." },
    ],

    // ------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "1. How the three fields nest" },
      { t: "p", text: "Each field sits fully inside the one above it. The table gives each field, what it adds, and an example that belongs to it but *not* to the next one in." },
      { t: "table", head: ["Field", "Definition", "What it adds", "Example that is in this field but NOT the next one in"], rows: [
        ["Artificial intelligence", "Systems that do tasks needing intelligence when done by people", "The goal", "A route planner using a search algorithm over a road map; a hand-written medical expert system; a chess program built on search"],
        ["Machine learning", "AI whose behaviour is learned from data, not written by hand", "Learning from examples", "A decision tree or logistic regression trained on a spreadsheet; k-means clustering"],
        ["Neural networks", "ML models built from layers of simple units (artificial neurons) that compute weighted sums followed by a non-linear function", "A specific model family", "A network with one hidden layer used on hand-made features (a *shallow* network)"],
        ["Deep learning", "Neural networks with several layers that learn a hierarchy of representations straight from raw-ish input", "Depth and learned features", "— (innermost circle)"],
      ] },
      { t: "steps", items: [
        { title: "Start with the goal (AI)", text: "Decide what smart behaviour you want: classify, predict, plan, generate, act. AI techniques include logic and rules, search, optimisation, probabilistic reasoning, and learning." },
        { title: "Choose to learn it (ML)", text: "If the rules are too complicated or unknown but **examples** exist, let an algorithm work out the rules from them. Training produces a **model**: a function whose parameters are set by data." },
        { title: "Choose a neural network as the model (NN)", text: "Among many learnable model families (linear models, decision trees, nearest neighbours, support vector machines, …), pick one built from layers of artificial neurons. Its parameters are those neurons' weights and biases." },
        { title: "Make it deep (DL)", text: "Stack several layers, so each layer re-describes the previous layer's output. Early layers learn simple features (edges, common letter pairs); later layers combine them into complex ones (object parts, phrases). Nobody programs these features — they emerge from training." },
      ] },
      { t: "callout", kind: "note", title: "How deep is \"deep\"?", text: "There is no official cut-off. In practice it means more than one hidden layer (a layer between input and output) and, more importantly, learning features end-to-end from raw or lightly processed input. Modern image models have tens to hundreds of layers; large language models stack dozens of Transformer blocks." },

      { t: "h", text: "2. Traditional programming versus machine learning" },
      { t: "p", text: "The clearest way to see the difference is to compare what goes in and what comes out:" },
      { t: "table", head: ["", "Traditional programming", "Machine learning"], rows: [
        ["Inputs", "**Rules** (the program) + **data**", "**Data** + **answers** (labels)"],
        ["Output", "**Answers**", "**Rules** (a trained model)"],
        ["Who designs the decision logic", "A programmer, by hand", "A training algorithm, by improving a score on examples"],
        ["How to improve it", "Rewrite or add rules", "Get more or better data, change the model, retrain"],
        ["How it fails", "Crashes or gives wrong answers on cases nobody wrote a rule for", "Gives confident wrong answers on inputs unlike its training data"],
        ["Typical good fit", "Rules are known and stable: tax calculation, sorting, a payroll system", "Rules are unknown or too complex: spam, speech, images, recommendations"],
      ] },
      { t: "p", text: "After training, a machine-learning model is used exactly like a normal program: data in, answers out. The only difference is **how the program was made**." },
      { t: "steps", items: [
        { title: "Collect examples", text: "Gather inputs with the correct answer for each (in supervised learning): the **dataset**." },
        { title: "Split the data", text: "Put some examples aside as a **test set** that the learning algorithm never sees. Only the **training set** is used to learn. This is the only honest way to estimate how the model will do on new data." },
        { title: "Choose a model family", text: "A function with adjustable parameters — for example, a weighted sum of the inputs passed through a squashing function (logistic regression), or a multi-layer neural network." },
        { title: "Define a loss", text: "A number that measures how wrong the model's predictions are on the training examples. Lower is better." },
        { title: "Optimise", text: "Adjust the parameters again and again to lower the loss. The standard method, **gradient descent**, nudges every parameter a little in the direction that lowers the loss fastest ([Gradient Descent](/learn/m06/gradient-descent))." },
        { title: "Evaluate on the test set", text: "Measure accuracy (or another score) on the held-out examples. If it is much worse than on the training set, the model has **overfitted** — memorised the training examples instead of learning the general pattern." },
        { title: "Deploy and monitor", text: "Use the model on real inputs and keep checking it, because the world drifts away from the training data. See [The Deep Learning Workflow](/learn/m01/dl-workflow)." },
      ] },

      { t: "h", text: "3. Classical machine learning versus deep learning" },
      { t: "p", text: "Both learn from data. The difference is **who decides what the model looks at**." },
      { t: "list", ordered: true, items: [
        "**Classical ML pipeline**: raw input → **hand-designed features** (a person writes code to compute, say, the ink in each row of a digit image, or word counts in an email) → a fairly simple learned model (linear model, decision tree, SVM). Most of the human effort goes into **feature engineering**.",
        "**Deep learning pipeline**: raw input (pixels, audio samples, characters or word pieces) → many learned layers → output. Early layers act as the feature extractor, learned together with the final classifier from the same loss. This is called **end-to-end learning** or **representation learning**.",
      ] },
      { t: "p", text: "Feature engineering isn't dead: on small tabular datasets (rows and columns, like a spreadsheet), good features with a gradient-boosted tree model often win. Deep learning wins when the input is high-dimensional and unstructured (images, sound, text) and there is enough data for the network to learn good features itself." },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "p", text: "The ideas above fit in a few lines of notation. You don't need it to *use* the ideas, but it makes \"learning\" precise." },
      { t: "h", text: "Traditional programming" },
      { t: "p", text: String.raw`A programmer writes a function $f$ directly. For an input $x$, the answer is $\hat y = f(x)$. Seeing more data never changes $f$.` },
      { t: "h", text: "Machine learning" },
      { t: "p", text: String.raw`Instead of one fixed function, we pick a **family** of functions $f_\theta$, controlled by a vector of parameters $\theta$ (theta). The training data is $N$ input–answer pairs $\{(x_i, y_i)\}_{i=1}^N$. A **loss function** $\ell(\hat y, y)$ scores how bad a prediction $\hat y$ is when the truth is $y$. Learning means choosing the parameters that make the average training loss small:` },
      { t: "math", tex: String.raw`\theta^\ast = \arg\min_{\theta}\; \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i),\, y_i\big)`, caption: "Empirical risk minimisation: pick the parameters with the lowest average training loss. $\\arg\\min$ means \"the value of $\\theta$ that gives the minimum\"." },
      { t: "table", head: ["Symbol", "Meaning"], rows: [
        ["$x_i$", "the $i$-th input (e.g. 30 tumour measurements, or 64 pixel values)"],
        ["$y_i$", "the correct answer (label) for $x_i$"],
        [String.raw`$f_\theta$`, "the model: a function whose behaviour is set by the parameters"],
        [String.raw`$\theta$`, "all the parameters (weights and biases) in one vector"],
        [String.raw`$\hat y = f_\theta(x)$`, "the model's prediction"],
        [String.raw`$\ell$`, "the loss for one example; $N$ is the number of training examples"],
      ] },
      { t: "p", text: String.raw`What we really care about is how the model does on **new** data from the same source. That is why we report accuracy on a held-out test set: it estimates the expected loss on unseen examples. The gap between training and test performance is the **generalisation gap**.` },
      { t: "h", text: "Logistic regression: the simplest learned classifier" },
      { t: "p", text: String.raw`For a yes/no question with input features $x = (x_1,\dots,x_d)$, logistic regression computes a weighted sum, then squashes it into a probability between 0 and 1 with the **sigmoid** function $\sigma$:` },
      { t: "math", tex: String.raw`z = w_1x_1 + w_2x_2 + \dots + w_dx_d + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}`, caption: "$w_j$ are weights (how much each feature counts), $b$ is the bias (a starting offset), $p$ is the predicted probability of class 1. Big positive $z$ gives $p$ near 1; big negative $z$ gives $p$ near 0. The parameters are $\\theta = (w_1, \\dots, w_d, b)$ — that is $d + 1$ numbers." },
      { t: "p", text: String.raw`Its loss is the **binary cross-entropy**. It is small when the model gives a high probability to the correct answer, and grows fast when it is confidently wrong:` },
      { t: "math", tex: String.raw`\ell(p, y) = -\big[\,y \ln p + (1-y)\ln(1-p)\,\big]` },
      { t: "p", text: "This is exactly one artificial neuron with a sigmoid activation — the bridge between classical ML and neural networks, covered fully in [Linear and Logistic Regression as Neural Networks](/learn/m03/linear-logistic-regression)." },
      { t: "h", text: "Worked example — every number" },
      { t: "p", text: String.raw`Say a model uses one feature, the tumour radius $x$, with learned parameters $w = 1.2$ and $b = -17$. We score it on four tumours (label $y = 1$ means malignant):` },
      { t: "table", head: ["Radius $x$", "Label $y$", "$z = 1.2x - 17$", String.raw`$p = \sigma(z)$`, "Loss"], rows: [
        ["11", "0", "−3.8", "0.021881", String.raw`$-\ln(1 - 0.021881) = 0.022124$`],
        ["14", "0", "−0.2", "0.450166", String.raw`$-\ln(1 - 0.450166) = 0.598139$`],
        ["16", "1", "2.2", "0.900250", String.raw`$-\ln(0.900250) = 0.105083$`],
        ["19", "1", "5.8", "0.996982", String.raw`$-\ln(0.996982) = 0.003023$`],
      ] },
      { t: "steps", items: [
        { title: "Average loss", tex: String.raw`\frac{0.022124 + 0.598139 + 0.105083 + 0.003023}{4} = 0.182092` },
        { title: "Predictions with threshold 0.5", text: "Predict malignant when $p > 0.5$: the four predictions are 0, 0, 1, 1 — all correct, so accuracy is 4/4 = 1.0." },
        { title: "Why the loss is still not zero", text: "The 14-mm tumour is classified correctly, but with $p = 0.45$ — close to the boundary. Cross-entropy penalises this doubt (loss 0.598), so training would keep pushing $w$ and $b$ to separate the classes more confidently. Accuracy only counts right/wrong; the loss measures *how* right." },
      ] },
      { t: "h", text: "Counting parameters" },
      { t: "p", text: String.raw`A layer that maps $n_{in}$ numbers to $n_{out}$ numbers has $n_{in} \times n_{out}$ weights plus $n_{out}$ biases. For the 8×8 digit images used later (64 pixels, 10 classes):` },
      { t: "steps", items: [
        { title: "Logistic regression on raw pixels", tex: String.raw`64 \times 10 + 10 = 650 \text{ parameters}` },
        { title: "Logistic regression on 19 hand-made features", tex: String.raw`19 \times 10 + 10 = 200 \text{ parameters}` },
        { title: "MLP 64–64–64–10", tex: String.raw`(64 \cdot 64 + 64) + (64 \cdot 64 + 64) + (64 \cdot 10 + 10) = 4160 + 4160 + 650 = 8970` },
      ] },
      { t: "p", text: "The deep model has about 14 times more parameters: more room to learn features, but also to memorise — so it needs more data." },
    ],

    // ---------------------------------------------------------------- visual
    visual: [
      { t: "h", text: "Watch a model learn its rule from data" },
      { t: "p", text: "Below is a real logistic-regression trainer. Each dot is a training example with two features (its x and y position); its colour is its label. The straight line is the model's current **decision boundary** — the rule it has learned so far." },
      { t: "sim", id: "logistic-regression", caption: "Logistic regression trained by gradient descent: the learned rule is the line; nobody draws it by hand." },
      { t: "list", items: [
        "Start with **Separable blobs** and press **Play**. The line starts in a random place and moves as training goes on — this *is* \"data + answers → rules\". Watch the loss fall in the chart.",
        "Click to **add points** of one class on the wrong side of the line, then keep training. The rule changes because the data changed — no code editing needed.",
        "Switch to **Circles (not linear)**. Accuracy gets stuck: one straight line can't separate a ring from its centre. This is the limit of a model that only computes a weighted sum — and the reason for hidden layers.",
        "Try **Imbalanced (9 : 1)**: a model that always predicts the bigger class already scores 90% accuracy. Always compare against such a do-nothing baseline (we do the same in the code below).",
      ] },
      { t: "h", text: "Look inside a (small) neural network" },
      { t: "p", text: "Deep learning replaces the single weighted sum with layers of them. The network visualizer shows every weighted sum and activation with real numbers." },
      { t: "sim", id: "network", caption: "A multi-layer network: each neuron computes a weighted sum plus bias, then an activation; layers feed forward into the next." },
      { t: "list", items: [
        "Press **Step** to watch values flow from the inputs through the hidden layer to the output. Each hidden neuron computes a *learned* feature of the input.",
        "Press **Train ×25** a few times and watch the loss fall. Training adjusts every weight at once, including the hidden layer's.",
        "Add a hidden layer with the **+ Layer** button. That is literally what makes a network \"deeper\". You will learn exactly how these numbers are computed in [Forward Propagation](/learn/m04/forward-propagation) and [Backpropagation](/learn/m04/backpropagation).",
      ] },
    ],

    // --------------------------------------------------------------- scratch
    scratch: [
      { t: "p", text: "Here is a real, measured comparison of **traditional programming** and **machine learning** on the same problem: labelling breast-tumour samples as malignant or benign from 30 measurements of cell nuclei. The data is the *Wisconsin Diagnostic Breast Cancer* dataset that comes with scikit-learn (used only to load it). Both approaches are scored on the same 143 held-out test samples." },
      { t: "code", lang: "python", title: "rules_vs_learning.py", code: RULES_VS_LEARNING, output: RULES_VS_LEARNING_OUT },
      { t: "p", text: "What the numbers say:" },
      { t: "list", items: [
        "**Baseline 0.643**: answering \"benign\" for everything is right 92 times out of 143. Any method must beat this to be useful.",
        "**Hand-written rule 0.888** (127/143 correct): two thresholds picked by eye on two of the 30 measurements already capture a lot — how expert systems worked, and proof they can be useful.",
        "**Learned logistic regression 0.979** (140/143 correct): using all 30 measurements, training found a weighting no person wrote down — a \"rule\" of 31 numbers (the three largest weights are printed). They are not on the two measurements a human picked, though weight size is only a rough guide to importance, because related features share credit.",
        "The loop is gradient descent. `p - malignant[train]` is the slope of the cross-entropy loss with respect to each example's $z$, and the two update lines move the weights against that slope. Learning rate 0.1 and 500 epochs finish in under a second. Standardising the features (subtract the mean, divide by the standard deviation, using training-set numbers only) lets one learning rate suit all 30.",
      ] },
      { t: "callout", kind: "warning", title: "Honest reading of this result", text: "This is one random split of a small dataset, so the accuracies are uncertain (one test sample is 0.7 percentage points). What holds up is the general point: the learned model uses all the information and beats a reasonable hand rule. A real medical system would also need to weigh false negatives against false positives, outside testing and regulatory review — accuracy alone is not enough." },
    ],

    // ------------------------------------------------------------- framework
    framework: [
      { t: "p", text: "Now **classical ML versus deep learning**: the same 8×8 handwritten-digit images (scikit-learn's `digits` dataset, 1,797 images, 10 classes), classified three ways. All three models are trained the same way (Adam optimiser, 300 full-batch steps) and tested on the same 540 held-out images." },
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "code", lang: "python", title: "features_vs_learned_features.py", code: DIGITS_TORCH, output: DIGITS_TORCH_OUT },
          { t: "list", items: [
            "**Hand-made features (0.896) lose to raw pixels (0.956)** with the same linear model. Our 19 features — row and column ink totals, symmetry scores — seem sensible, but throw away *where* the ink is within each row, which is exactly what tells many digits apart. Designing features is hard.",
            "**The MLP (0.969)** is a small deep-learning model: two hidden layers of 64 ReLU units (ReLU is the function $\\max(0, z)$). Its hidden layers compute 64 learned features from the pixels, chosen by training to make the final classification easy. It is the most accurate of the three — 523 correct versus 516 for raw-pixel logistic regression.",
            "The MLP's lead is small because 8×8 digits are easy even for a linear model. On high-resolution photos, audio or text the gap between hand and learned features is far larger — the story of the [History of Neural Networks](/learn/m01/history).",
          ] },
          { t: "callout", kind: "note", text: "`nn.Linear(64, 10)` trained with cross-entropy *is* multinomial logistic regression — a neural network with no hidden layer. Add hidden layers with non-linear activations and it becomes a (small) deep network." },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "features_vs_learned_features_keras.py (not executed in CI)", code: DIGITS_KERAS, output: "Prints one test accuracy for logistic regression (a Dense layer with no hidden layers) and one for the 2-hidden-layer MLP. Expect values similar to the PyTorch run (roughly 0.95–0.97), with the MLP usually slightly ahead; exact numbers depend on TensorFlow's random initialisation." },
          { t: "callout", kind: "tip", text: "`from_logits=True` tells Keras that the last layer outputs raw scores (logits) and that the loss should apply softmax itself — the same convention as PyTorch's `CrossEntropyLoss`." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------ variations
    variations: [
      { t: "h", text: "The learning paradigms" },
      { t: "p", text: "Machine learning is split by **what kind of feedback the learner gets**. Deep learning can be used in all of them." },
      { t: "table", head: ["Paradigm", "What the data looks like", "Goal", "Examples", "Main difficulty"], rows: [
        ["**Supervised**", "Inputs with correct answers (labels)", "Predict the label of new inputs: **classification** (a category) or **regression** (a number)", "Spam filtering, image classification, price prediction", "Labels are expensive: someone must annotate every example"],
        ["**Unsupervised**", "Inputs only, no labels", "Find structure: clusters, short summaries, anomalies, the data distribution", "Customer segmentation, anomaly detection, compression", "No single right answer, so success is hard to measure"],
        ["**Self-supervised**", "Inputs only — but a task is *made* from them by hiding part of each input and predicting it", "Learn general-purpose representations from huge unlabelled datasets", "Predicting the next word (how large language models are pretrained), filling in masked words or image patches", "Designing a prediction task that forces useful features to be learned"],
        ["**Reinforcement**", "No fixed dataset; an **agent** acts in an **environment** and receives **rewards**", "Learn a strategy (**policy**) that maximises total reward over time", "Game playing, robot control, tuning language models from human preference feedback", "Rewards come late and rarely; the agent must balance trying new things against using what works"],
      ] },
      { t: "p", text: "Self-supervised learning is technically supervised learning on automatically made labels (so some books list three paradigms), but it is how most modern foundation models are pretrained. Other variants include **semi-supervised** learning (a few labels plus many unlabelled examples) and **transfer learning** (reusing a model trained on one task for another — see [Transfer Learning](/learn/m12/transfer-learning))." },
      { t: "p", text: "Here are all four paradigms on toy problems, so you can see they really are different kinds of learning:" },
      { t: "code", lang: "python", title: "four_paradigms.py", code: PARADIGMS, output: PARADIGMS_OUT },
      { t: "list", items: [
        "**Supervised**: with labels, a linear classifier reaches 0.970 test accuracy. (A bit above the PyTorch logistic regression's 0.956, because scikit-learn adds L2 regularisation and uses a different optimiser and input scaling — small choices move results by a point.)",
        "**Unsupervised**: k-means, never shown a label, groups the images so that 79.7% of them share a cluster with mostly their own digit. It found much of the digit structure from pixel similarity alone.",
        "**Self-supervised**: predicting the bottom half of a digit from its top half cuts the error from 0.0783 to 0.0528. No human labels were used — the \"answer\" is part of the image. Large language models are trained the same way, with the next word as the hidden part.",
        "**Reinforcement**: never told which machine pays best, the agent tried them and estimated payout rates (0.19, 0.41, 0.80 versus true 0.2, 0.5, 0.8), choosing the best machine 926 times out of 1,000. Its middle estimate is rough because, sensibly, it rarely played that one.",
      ] },
      { t: "h", text: "Families of AI techniques" },
      { t: "table", head: ["Approach", "How behaviour is produced", "Advantages", "Disadvantages", "Typical use"], rows: [
        ["Rule-based / symbolic AI", "Hand-written rules, logic, search", "Transparent, no data needed, exact when rules are known", "Brittle; experts can't write rules for perception or language", "Business logic, planning, verification, configuration"],
        ["Classical ML", "Learned model on hand-designed features", "Needs less data, fast, often interpretable", "Only as good as the features a human designs", "Tabular data, credit scoring, many industrial models"],
        ["Deep learning", "Many-layered neural network learns features end-to-end", "State of the art on images, audio, language; one recipe for many data types", "Hungry for data and compute; hard to interpret; can fail unpredictably on unfamiliar inputs", "Vision, speech, translation, generative models"],
        ["Hybrid", "Learned parts inside a hand-designed system", "Combines guarantees with learned perception", "More engineering complexity", "Robotics, game-playing systems that combine search with neural networks"],
      ] },
      { t: "h", text: "Narrow AI and general AI" },
      { t: "list", items: [
        "**Narrow AI** (also called *weak AI* in everyday speech) does a specific task or a limited set of tasks: recognise faces, transcribe speech, recommend films. Every deployed AI system so far was designed and tested for particular tasks, even very flexible ones.",
        "**Artificial general intelligence (AGI)** is a *hypothetical* system that could learn and do almost any mental task a human can, at human level or above. There is **no agreed definition and no agreed test** for it.",
        "Modern large language models blur the line: one model can write code, summarise, translate and answer questions. Whether any current system counts as AGI, and whether or when AGI will arrive, are **hotly disputed** and depend heavily on the definition. Be cautious about confident claims either way.",
        "Don't confuse narrow/general with the philosophers' *weak/strong AI* distinction (John Searle, 1980). That one asks whether a program could truly *understand* or only simulate understanding — a question about minds, not about the range of tasks.",
      ] },
    ],

    // ---------------------------------------------------------- applications
    applications: [
      { t: "p", text: "Deep learning sits inside a very wide range of products. In general terms (not tied to any company):" },
      { t: "table", head: ["Area", "Task", "Typical approach", "Paradigm"], rows: [
        ["Vision", "Recognise objects, read text in photos, spot defects on a production line, analyse medical scans", "Convolutional networks or Vision Transformers ([CNNs](/learn/m08/images-as-tensors))", "Supervised, often after self-supervised pretraining"],
        ["Speech", "Turn speech into text; produce natural-sounding speech", "Deep networks over audio spectrograms; sequence models", "Supervised + self-supervised"],
        ["Language", "Translation, search ranking, summarisation, chat assistants, code completion", "Transformers ([The Transformer](/learn/m11/transformer-architecture))", "Self-supervised pretraining, then supervised and reinforcement fine-tuning"],
        ["Recommendation", "Rank products, videos or posts for each user", "Embedding models and deep ranking networks", "Supervised from clicks and other feedback"],
        ["Science", "Predict protein structure, screen molecules, model weather", "Specialised deep architectures", "Supervised and self-supervised"],
        ["Generation", "Create images, audio or video from a text description", "Diffusion models and Transformers ([Diffusion Models](/learn/m14/diffusion))", "Self-supervised / generative"],
        ["Control", "Robot manipulation, game playing, data-centre cooling", "Deep reinforcement learning ([Deep RL](/learn/m15/deep-rl))", "Reinforcement"],
      ] },
      { t: "p", text: "Deep learning is usually **not** the first choice for: spreadsheets of a few thousand rows (gradient-boosted trees and linear models are strong and cheaper), problems with exact known rules (tax, accounting), and safety-critical decisions needing guarantees or full explanations — where learned parts are used, if at all, inside carefully engineered and audited systems." },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "h", text: "What deep learning is good at" },
      { t: "list", items: [
        "**Perception and language**: images, audio, video and text, where writing rules or features by hand is impractical.",
        "**Learning features automatically**: one general recipe (layers + backpropagation + data) works across very different data types, instead of new feature engineering for every problem.",
        "**Growing with data and compute**: in many areas accuracy keeps improving as models, data and computing grow, while classical methods level off.",
        "**Transfer**: a network pretrained on a huge dataset can be adapted to a new task with fairly little labelled data.",
        "**Generation**: producing realistic text, images, audio and code.",
      ] },
      { t: "h", text: "What it is bad at (or costly for)" },
      { t: "list", items: [
        "**Data hunger**: training from scratch usually needs many labelled examples; with little data, simpler models often win.",
        "**Compute and energy cost**: large models need special hardware (GPUs) to train and run.",
        "**Opacity**: learned parameters come with no human-readable explanation; interpretability tools help only partly ([Interpretability](/learn/m19/interpretability)).",
        "**Brittle on unfamiliar inputs**: inputs unlike the training data (new lighting, new slang, a different hospital's scanner) can cause confident errors, and tiny crafted changes can flip predictions ([Robustness](/learn/m19/robustness-safety)).",
        "**Learning the wrong shortcut**: a model optimises its loss on the data it gets, so it will exploit any accidental pattern (a watermark that goes with one class, a hospital ID that goes with a diagnosis).",
        "**Bias from the data**: if the training data under-represents a group or records past unfair decisions, the model repeats it.",
        "**No built-in guarantees**: unlike a hand-written program, a network can rarely be proven correct on all inputs, and generative models can state falsehoods fluently.",
      ] },
      { t: "callout", kind: "key", text: "The practical rule: use the simplest method that solves the problem. Rules when you know them; classical ML for small structured data; deep learning when the input is unstructured, data is plentiful, and the accuracy gain is worth the cost." },
    ],

    // -------------------------------------------------------------- mistakes
    mistakes: [
      { t: "list", items: [
        "**\"AI, ML and DL are the same thing.\"** No — they are nested. A route planner using a shortest-path algorithm is AI with no learning; logistic regression is ML with no deep network. Fix: use the most specific term that is true.",
        "**\"Deep learning means any neural network.\"** A single neuron (logistic regression), or a one-hidden-layer network on hand-made features, is a neural network but not what people mean by deep learning. Depth and learned representations are what define it.",
        "**Evaluating on the training data.** A model that memorises its training set scores perfectly there and badly on new data. Fix: hold out a test set the algorithm never sees, and fit preprocessing (like our standardisation) on training data only.",
        "**Skipping the do-nothing baseline.** 90% accuracy sounds good until you notice 90% of examples belong to one class. Fix: always report the majority-class baseline, as the code above does (0.643).",
        "**\"More complex is always better.\"** Our hand-written rule reached 0.888 with two thresholds, and on small tabular data a deep network is often *worse* than a well-tuned simple model. Fix: try simple models first; add complexity only when it measurably helps.",
        "**Believing the model learned what you intended.** It learned whatever lowers the loss, shortcuts included. Fix: look at the errors, test on data from other sources, and check which inputs drive predictions.",
        "**Treating a learned model as objective.** Data reflects how it was collected, and a model trained on it inherits those biases. Fix: check performance across subgroups and document where the data came from.",
        "**Overstating what AI systems are.** Calling today's systems \"general intelligence\" or \"just autocomplete\" both skip the evidence. Fix: describe what the system was trained for and how it was evaluated.",
      ] },
    ],

    // ------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the difference between AI, machine learning and deep learning?", a: "AI is the broad field of building systems that do tasks needing intelligence. Machine learning is the part of AI where behaviour is learned from data instead of programmed by hand. Deep learning is the part of machine learning that uses neural networks with many layers, which learn their own internal features (representations) from raw input. Every DL system is ML, and every ML system is AI, but not the other way round." },
      { t: "qa", level: "beginner", q: "How does machine learning differ from traditional programming?", a: "In traditional programming a person writes the rules and the computer applies them to data to produce answers. In machine learning the person supplies data plus the desired answers, and a training algorithm produces the rules — a model fitted to the examples. Once trained, it is used like any other program." },
      { t: "qa", level: "beginner", q: "Name the main learning paradigms and give an example of each.", a: "Supervised learning learns from labelled examples (sorting emails into spam and not spam). Unsupervised learning finds structure in unlabelled data (grouping customers). Self-supervised learning makes labels from the data itself by hiding part of the input and predicting it (next-word prediction for language models). Reinforcement learning learns a policy by acting in an environment and receiving rewards (game playing, robot control)." },
      { t: "qa", level: "intermediate", q: "Why did deep learning overtake classical ML on images and speech but not necessarily on tabular data?", a: "Images and audio are high-dimensional and unstructured; their useful features (edges, shapes, phonemes) are layered and very hard to design by hand, but deep networks learn them end-to-end given lots of data and compute. Tabular features are usually already meaningful and few, datasets are often small, and tree ensembles use that structure efficiently — so learning features helps less and costs more data." },
      { t: "qa", level: "intermediate", q: "What is feature engineering, and what does representation learning replace?", a: "Feature engineering is designing by hand the numbers a model receives (e.g. ink per row, word counts, edge histograms). Representation learning lets the model learn those numbers itself: a deep network's hidden layers turn raw input into features tuned for the task, through the same loss as the final classifier. It replaces the hand-designed feature extractor — not the need for good data." },
      { t: "qa", level: "intermediate", q: "Why do we need a separate test set, and why must preprocessing be fitted only on training data?", a: "The goal is performance on unseen data. Training accuracy is too optimistic, because the parameters were tuned on those exact examples; a held-out test set gives a fair estimate. If preprocessing numbers (means, standard deviations, vocabularies) are computed using test data, information about the test set leaks into the model and the estimate becomes optimistic again." },
      { t: "qa", level: "intermediate", q: "Is logistic regression a neural network?", a: "It is mathematically the same as a single artificial neuron with a sigmoid activation trained with cross-entropy loss — a neural network with no hidden layer. It is usually called classical ML because it has no hidden layers and learns no features: it learns one weight per given input feature." },
      { t: "qa", level: "advanced", q: "What is the distinction between narrow AI and AGI, and is it a precise one?", a: "Narrow AI is built and tested for specific tasks; AGI is a hypothetical system able to learn and do almost any mental task at human level. The line is not precise: there is no agreed definition or test for AGI, and modern general-purpose models do many tasks without task-specific training. A careful answer names the definition being used and the evidence (which tasks, how evaluated) instead of a plain yes/no." },
      { t: "qa", level: "advanced", q: "Formally, what is a learning algorithm minimising, and why is that not exactly what we care about?", a: "It minimises the empirical risk — the average loss over the training examples, $\\frac{1}{N}\\sum_i \\ell(f_\\theta(x_i), y_i)$. What we care about is the expected loss over the true data distribution (the population risk). They differ because the training set is a finite sample: a flexible model can push empirical risk to zero while population risk stays high (overfitting). Regularisation, more data, choosing models by validation, and good inductive biases (architectures suited to the data) all aim to shrink that gap." },
      { t: "qa", level: "advanced", q: "Self-supervised learning uses no human labels. Why is it still not unsupervised learning in the classical sense?", a: "It is trained exactly like supervised learning — a predictor, a target and a loss — but the targets come automatically from the input (the masked word, the next token, another augmented view of the same image). Classical unsupervised methods such as clustering or PCA have no prediction target at all. Self-supervision turns unlimited unlabelled data into a supervised-style signal, which large pretrained models rely on." },
    ],

    // -------------------------------------------------------------- practice
    practice: [
      { t: "quiz", questions: [
        { q: "Which statement about the relationship between the fields is correct?", options: ["Every AI system is a deep-learning system", "Deep learning ⊂ neural networks ⊂ machine learning ⊂ AI", "Machine learning and AI are disjoint fields", "Neural networks are a superset of machine learning"], answer: 1, explain: "The fields are strictly nested: deep learning is a kind of neural network, neural networks are a kind of ML model, and ML is one approach to AI." },
        { q: "In machine learning, what are the inputs to the process that produces the program?", options: ["Rules and data", "Data and answers (labels)", "Answers only", "Rules only"], answer: 1, explain: "Traditional programming: rules + data → answers. Machine learning: data + answers → rules (a trained model)." },
        { q: "A model is trained to predict the next word of web text, with no human annotation. Which paradigm is this?", options: ["Supervised with human labels", "Reinforcement learning", "Self-supervised learning", "Rule-based AI"], answer: 2, explain: "The target (the next word) comes from the data itself — the defining feature of self-supervised learning." },
        { kind: "numeric", q: "A logistic-regression model uses 30 input features for a yes/no task. How many parameters does it have?", answer: 31, tolerance: 0, explain: "One weight per feature plus one bias: 30 + 1 = 31." },
        { kind: "numeric", q: "A spam filter is right on 180 of 200 test emails. What is its accuracy (as a decimal)?", answer: 0.9, tolerance: 0.001, explain: "180 / 200 = 0.9." },
        { kind: "numeric", q: "For $z = 0$, what probability does the sigmoid $\\sigma(z) = 1/(1+e^{-z})$ output?", answer: 0.5, tolerance: 0.001, explain: "$e^0 = 1$, so $\\sigma(0) = 1/(1+1) = 0.5$ — exactly on the decision boundary." },
        { q: "On the digits experiment, logistic regression on 19 hand-made features scored 0.896 while the same model on 64 raw pixels scored 0.956. The best explanation is:", options: ["Raw pixels always beat features", "The hand features discarded information (where the ink is within each row) that the classifier needed", "19 features cause overfitting", "The hand-feature model had more parameters"], answer: 1, explain: "Row and column totals squash each image and lose position detail. Designing features that keep exactly the right information is hard — which is why we want to learn features." },
        { q: "Which problem is the WORST fit for deep learning?", options: ["Transcribing speech", "Classifying photos of plant diseases with 100,000 labelled images", "Computing payroll tax from a fixed legal formula", "Translating text between languages"], answer: 2, explain: "The rules are known exactly and must be applied exactly. A hand-written program is cheaper, transparent and correct by design." },
      ] },
      { t: "exercise", title: "Score a learned rule by hand", prompt: "A one-feature logistic-regression model predicts whether a student passes from hours studied $x$, with $w = 0.8$ and $b = -4$. For a student who studied 6 hours and passed ($y = 1$), compute (a) $z$, (b) $p = \\sigma(z)$, (c) the predicted class with threshold 0.5, and (d) the cross-entropy loss $-\\ln p$. Then (e) find the number of hours at which the model is exactly undecided.", hint: "The model is undecided where $p = 0.5$, which happens where $z = 0$.", solution: [
        { t: "steps", items: [
          { title: "(a) Weighted sum", tex: String.raw`z = 0.8 \times 6 - 4 = 0.8` },
          { title: "(b) Probability", tex: String.raw`p = \frac{1}{1 + e^{-0.8}} = \frac{1}{1 + 0.449329} = 0.689974` },
          { title: "(c) Prediction", text: "$p = 0.69 > 0.5$, so the model predicts *pass* — correct." },
          { title: "(d) Loss", tex: String.raw`-\ln(0.689974) = 0.371101` },
          { title: "(e) Decision boundary", tex: String.raw`0.8x - 4 = 0 \;\Rightarrow\; x = 5 \text{ hours}` },
        ] },
        { t: "p", text: "So the learned \"rule\" is *predict pass if hours > 5*. A programmer could have written that rule. The point of ML is that the threshold 5 came from data — and with 30 features instead of one, no programmer could find the best weighting by hand." },
      ] },
      { t: "exercise", title: "Write a rule, then let the computer learn one", prompt: "Using scikit-learn's digits dataset, keep only the digits 0 and 1. (1) Write a rule by hand that tells them apart using pixel values (hint: a 0 is a ring with an empty centre, a 1 is a stroke through the middle). (2) Train a logistic regression on the raw 64 pixels. Compare both on the same held-out 30% test split.", hint: "`digits.images` has shape (1797, 8, 8). Sum the ink in a central block such as rows 2–5, columns 3–4, and pick a threshold.", solution: [
        { t: "code", lang: "python", title: "zero_vs_one.py", code: EX_ZERO_ONE, output: EX_ZERO_ONE_OUT },
        { t: "p", text: "The hand-written rule is right 96.3% of the time (104 of 108); the learned model gets all 108. For an easy, well-understood problem a human rule can compete — but the learned model needed no insight about rings and strokes. For all ten digits, writing rules gets far harder, while the learning code barely changes." },
      ] },
    ],

    // --------------------------------------------------------------- summary
    summary: [
      { t: "list", items: [
        "**AI** = systems doing tasks that need intelligence; it includes rules, search and learning.",
        "**Machine learning** = AI whose behaviour is learned from data: data + answers → rules (a trained model).",
        "**Neural networks** = ML models built from layers of artificial neurons (weighted sum + non-linearity).",
        "**Deep learning** = neural networks with many layers that learn their own features (representations) end-to-end.",
        "Nesting: DL ⊂ NN ⊂ ML ⊂ AI — each fully inside the next.",
        "Paradigms: supervised (labels), unsupervised (structure, no labels), self-supervised (labels made from the data), reinforcement (rewards from acting).",
        "Measured here: on tumour data a hand-written rule scored 0.888, learned logistic regression 0.979 (baseline 0.643); on digits, hand features 0.896, raw-pixel linear model 0.956, small MLP 0.969.",
        "Classical ML relies on feature engineering; deep learning learns features, at the cost of more data and compute.",
        "Narrow AI does specific tasks; AGI is hypothetical, with no agreed definition — be careful with claims about it.",
        "Always evaluate on held-out data and against a do-nothing baseline.",
      ] },
      { t: "math", tex: String.raw`\theta^\ast = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \ell\big(f_\theta(x_i), y_i\big), \qquad p = \sigma(w^\top x + b) = \frac{1}{1+e^{-(w^\top x + b)}}` },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[History of Neural Networks](/learn/m01/history) — why neural networks fell out of favour twice, and what (data, GPUs, better training tricks) made deep learning work in the 2010s, with some historical results reproduced in code.",
        "[The Deep Learning Workflow](/learn/m01/dl-workflow) — the full practical process sketched above, from framing the problem to deployment.",
        "[Linear and Logistic Regression as Neural Networks](/learn/m03/linear-logistic-regression) — the model used throughout this lesson, studied properly: how gradient descent finds $w$ and $b$, and why it is a one-neuron network.",
        "If you want the maths first: [Vectors and Matrices](/learn/m02/vectors-matrices) and [Derivatives and the Chain Rule](/learn/m02/derivatives-chain-rule) are the two tools every later lesson uses.",
      ] },
    ],
  },
};

export default lesson;
