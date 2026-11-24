import type { Lesson } from "@/lib/types";

// ---------------------------------------------------------------------------
// Code blocks (all executed; outputs pasted verbatim)
// ---------------------------------------------------------------------------

const XOR_HISTORY = String.raw`import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
AND = np.array([0, 0, 0, 1])
XOR = np.array([0, 1, 1, 0])

# ===============================================================
# 1958-style: Rosenblatt's perceptron learning rule
# ===============================================================
def train_perceptron(X, y, epochs=20, lr=1.0):
    w, b = np.zeros(2), 0.0
    for epoch in range(1, epochs + 1):
        mistakes = 0
        for xi, yi in zip(X, y):
            pred = int(w @ xi + b > 0)          # step activation
            if pred != yi:                       # update ONLY on a mistake
                w += lr * (yi - pred) * xi
                b += lr * (yi - pred)
                mistakes += 1
        if mistakes == 0:
            return w, b, epoch, True
    return w, b, epochs, False

for name, y in [("AND", AND), ("XOR", XOR)]:
    w, b, epochs, ok = train_perceptron(X, y)
    preds = (X @ w + b > 0).astype(int)
    status = f"converged after {epochs} epochs" if ok else f"still making mistakes after {epochs} epochs"
    print(f"perceptron on {name}: {status}; predictions {preds}, targets {y}")

# ===============================================================
# 1986-style: a 2-H-1 sigmoid network trained with backpropagation
# ===============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_xor_net(seed, hidden=2, lr=1.0, epochs=10000, report=()):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 1, (hidden, 2)), np.zeros(hidden)   # input -> hidden
    W2, b2 = rng.normal(0, 1, hidden), 0.0                     # hidden -> output
    y = XOR.astype(float)
    for epoch in range(1, epochs + 1):
        # forward pass
        h = sigmoid(X @ W1.T + b1)                 # (4, hidden) hidden activations
        out = sigmoid(h @ W2 + b2)                 # (4,) predictions
        loss = np.mean((out - y) ** 2)
        # backward pass (chain rule) for the mean squared error
        d_out = 2 * (out - y) / len(y) * out * (1 - out)    # dL/dz at the output
        d_h = np.outer(d_out, W2) * h * (1 - h)              # dL/dz at the hidden layer
        W2 -= lr * h.T @ d_out
        b2 -= lr * d_out.sum()
        W1 -= lr * d_h.T @ X
        b1 -= lr * d_h.sum(axis=0)
        if epoch in report:
            print(f"  epoch {epoch:>5}: loss {loss:.4f}, outputs {np.round(out, 3)}")
    return out

print("2-2-1 network, seed 0:")
out = train_xor_net(seed=0, report=(1, 500, 2000, 10000))
print("  rounded predictions:", (out > 0.5).astype(int), "targets", XOR)

# Backprop is not magic: with only 2 hidden units some starting points get stuck.
for hidden in (2, 4):
    solved = sum(np.array_equal(train_xor_net(seed=s, hidden=hidden) > 0.5, XOR == 1) for s in range(20))
    print(f"{hidden} hidden units: XOR solved from {solved} of 20 random starting points")`;

const XOR_HISTORY_OUT = String.raw`perceptron on AND: converged after 6 epochs; predictions [0 0 0 1], targets [0 0 0 1]
perceptron on XOR: still making mistakes after 20 epochs; predictions [1 1 0 0], targets [0 1 1 0]
2-2-1 network, seed 0:
  epoch     1: loss 0.2502, outputs [0.478 0.485 0.488 0.495]
  epoch   500: loss 0.2455, outputs [0.471 0.513 0.507 0.529]
  epoch  2000: loss 0.0036, outputs [0.063 0.942 0.942 0.063]
  epoch 10000: loss 0.0003, outputs [0.019 0.983 0.983 0.017]
  rounded predictions: [0 1 1 0] targets [0 1 1 0]
2 hidden units: XOR solved from 16 of 20 random starting points
4 hidden units: XOR solved from 20 of 20 random starting points`;

const SIGMOID_RELU = String.raw`import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)                # 1,797 8x8 digit images
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, random_state=0, stratify=y)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

def make_net(depth, act, init):
    """depth hidden layers of 64 units, then a 10-way output layer."""
    layers, n_in = [], 64
    for _ in range(depth):
        layers += [nn.Linear(n_in, 64), act()]
        n_in = 64
    layers.append(nn.Linear(n_in, 10))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            init(m.weight)
            nn.init.zeros_(m.bias)
    return net

xavier = nn.init.xavier_normal_                                    # Glorot & Bengio, 2010
he = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")     # He et al., 2015

def run(depth, act, init, epochs=30, lr=0.1, batch=64):
    torch.manual_seed(0)                       # same randomness for every configuration
    net = make_net(depth, act, init)
    # size of the gradient that reaches the FIRST layer before any training
    nn.functional.cross_entropy(net(X_tr), y_tr).backward()
    g0 = net[0].weight.grad.norm().item()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    losses = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch):   # mini-batch stochastic gradient descent
            idx = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx]).backward()
            opt.step()
        if epoch in (1, 5, 10, 30):
            with torch.no_grad():
                losses.append(nn.functional.cross_entropy(net(X_tr), y_tr).item())
    with torch.no_grad():
        acc = (net(X_te).argmax(1) == y_te).float().mean().item()
    return g0, losses, acc

print("configuration                     first-layer grad   train loss @ epoch 1 / 5 / 10 / 30    test acc")
for label, depth, act, init in [
    ("2 hidden, sigmoid + Xavier", 2, nn.Sigmoid, xavier),
    ("8 hidden, sigmoid + Xavier", 8, nn.Sigmoid, xavier),
    ("8 hidden, tanh    + Xavier", 8, nn.Tanh, xavier),
    ("8 hidden, ReLU    + He", 8, nn.ReLU, he),
]:
    g0, losses, acc = run(depth, act, init)
    print(f"{label:<33} {g0:>12.2e}       " + " / ".join(f"{l:.3f}" for l in losses) + f"        {acc:.3f}")`;

const SIGMOID_RELU_OUT = String.raw`configuration                     first-layer grad   train loss @ epoch 1 / 5 / 10 / 30    test acc
2 hidden, sigmoid + Xavier            5.53e-02       2.302 / 2.278 / 2.238 / 1.670        0.665
8 hidden, sigmoid + Xavier            5.86e-06       2.311 / 2.308 / 2.308 / 2.306        0.100
8 hidden, tanh    + Xavier            3.04e-01       1.117 / 0.284 / 0.142 / 0.021        0.974
8 hidden, ReLU    + He                3.48e-01       1.783 / 0.590 / 0.242 / 0.002        0.970`;

const SIGMOID_RELU_KERAS = String.raw`import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, random_state=0, stratify=y)

def make_net(depth, activation, initializer):
    model = tf.keras.Sequential([tf.keras.Input(shape=(64,))])
    for _ in range(depth):
        model.add(tf.keras.layers.Dense(64, activation=activation, kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(10))
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model

for label, act, init in [("sigmoid + Glorot", "sigmoid", "glorot_normal"), ("ReLU + He", "relu", "he_normal")]:
    model = make_net(8, act, init)
    hist = model.fit(X_tr, y_tr, epochs=30, batch_size=64, verbose=0)
    print(label, "final train loss:", round(hist.history["loss"][-1], 3),
          "test accuracy:", round(model.evaluate(X_te, y_te, verbose=0)[1], 3))`;

const EX_XOR_FEATURE = String.raw`import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR = np.array([0, 1, 1, 0])

# Hand-made third feature: the product x1 * x2 (1 only for the input (1, 1))
X3 = np.hstack([X, X[:, :1] * X[:, 1:]])
print("inputs with extra feature:\n", X3)

w, b = np.zeros(3), 0.0
for epoch in range(1, 51):
    mistakes = 0
    for xi, yi in zip(X3, XOR):
        pred = int(w @ xi + b > 0)
        if pred != yi:
            w += (yi - pred) * xi
            b += yi - pred
            mistakes += 1
    if mistakes == 0:
        break
print(f"converged after {epoch} epochs: w = {w}, b = {b}")
print("predictions:", (X3 @ w + b > 0).astype(int), "targets:", XOR)`;

const EX_XOR_FEATURE_OUT = String.raw`inputs with extra feature:
 [[0. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [1. 1. 1.]]
converged after 8 epochs: w = [ 1.  1. -3.], b = 0.0
predictions: [0 1 1 0] targets: [0 1 1 0]`;

// ---------------------------------------------------------------------------

const lesson: Lesson = {
  id: "m01/history",
  title: "History of Neural Networks",
  summary:
    "From the McCulloch–Pitts neuron (1943) to Transformers and diffusion models: the ideas, the two AI winters, the breakthroughs of the 2010s, and why deep learning worked when it did — with small historical results reproduced in code.",
  minutes: 55,
  level: "beginner",
  prerequisites: ["m01/ai-ml-dl"],
  next: ["m01/dl-workflow", "m03/artificial-neuron", "m04/gradient-problems"],
  terms: [
    "mcculloch-pitts-neuron",
    "perceptron",
    "xor-problem",
    "ai-winter",
    "backpropagation",
    "layer-wise-pretraining",
    "support-vector-machine",
    "imagenet",
    "gpu",
    "vanishing-gradient",
    "relu",
    "transformer",
  ],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      { t: "p", text: "Deep learning is often presented as something that appeared suddenly around 2012. In fact almost every core idea is decades old: the artificial neuron dates from 1943, trainable networks from 1958, backpropagation from the 1970s–80s, convolutional networks from the 1980s, and LSTMs from 1997. What changed in the 2010s was not one invention but the **combination** of much larger datasets, much faster hardware (graphics processors, GPUs), and a set of practical training improvements that made deep networks reliably trainable." },
      { t: "p", text: "This lesson tells that story carefully. It matters for three reasons. First, it explains *why* modern networks look the way they do — ReLU activations, careful initialisation, normalisation and residual connections each solved a specific historical obstacle. Second, the field has twice gone through periods of inflated expectations followed by disappointment and reduced funding (the **AI winters**); knowing why helps you judge today's claims. Third, many \"new\" ideas are rediscoveries, and knowing the history makes you a better reader of research papers." },
      { t: "callout", kind: "key", title: "The story in one sentence", text: "Neural networks were invented in the 1940s–50s, hit fundamental limits with single layers, gained a training algorithm for multiple layers (backpropagation) in the 1970s–80s, were overshadowed by other methods in the 1990s–2000s because deep networks were hard to train with the data and computers available, and took over from about 2012 when big labelled datasets, GPUs and better training techniques arrived together." },
      { t: "callout", kind: "note", title: "How dates are given here", text: "Dates refer to the publication usually cited for each idea. Many ideas had several independent inventors and credit is genuinely disputed in places; where that is the case, this lesson says so rather than picking a single hero." },
    ],

    // --------------------------------------------------------- prerequisites
    prerequisites: [
      { t: "list", items: [
        "[AI, Machine Learning and Deep Learning](/learn/m01/ai-ml-dl) — you need the difference between hand-written rules and learning from data, and the idea that a model has parameters (weights) adjusted by training.",
        "No mathematics is required to follow the story. The Mathematical Explanation section uses only multiplication and the sigmoid function $\\sigma(z) = 1/(1+e^{-z})$, which is explained there.",
        "Optional: [The Artificial Neuron and the Perceptron](/learn/m03/artificial-neuron) covers the 1943–1969 part of this story in full technical depth, including the proof that a single perceptron cannot compute XOR. This lesson links to it rather than repeating it.",
      ] },
    ],

    // ------------------------------------------------------------- intuition
    intuition: [
      { t: "callout", kind: "analogy", title: "A recipe waiting for a kitchen", text: "Imagine someone writes down a recipe for bread in 1950, but nobody has an oven hot enough or enough flour to bake more than a crumb. For decades people argue about whether the recipe works at all. Then ovens get a thousand times hotter, flour becomes cheap, and a few cooks figure out small tricks (knead this way, add salt here) — and suddenly the recipe produces excellent bread. Neural networks are the recipe; large datasets are the flour; GPUs are the oven; ReLU, good initialisation, dropout, normalisation and residual connections are the tricks." },
      { t: "p", text: "Two broad schools of thought run through the history of AI. **Symbolic AI** tries to build intelligence from explicit rules, logic and search — the \"rulebook\" approach. **Connectionism** tries to build it from many simple connected units that learn from examples — the neural-network approach. For most of the 20th century the symbolic school was dominant in AI research and funding. Neural networks have been in and out of fashion several times; since the 2010s, connectionist methods have dominated perception, language and generation." },
      { t: "p", text: "A useful mental picture is a sequence of **obstacles and fixes**:" },
      { t: "list", ordered: true, items: [
        "Obstacle: a neuron model existed (1943) but could not learn. Fix: the perceptron learning rule (1958).",
        "Obstacle: one layer of perceptrons cannot represent simple functions such as XOR (analysed in 1969). Fix: hidden layers — but they need a training method.",
        "Obstacle: how to train hidden layers. Fix: backpropagation (developed in the 1970s, popularised in 1986).",
        "Obstacle: deep networks trained badly — gradients vanished, data and compute were scarce. Fixes: big datasets (ImageNet, 2009), GPUs, ReLU, better initialisation, dropout, batch normalisation and residual connections (roughly 2009–2015).",
        "Obstacle: recurrent networks process text one word at a time, which is slow and forgets long-range context. Fix: attention (2014) and the Transformer (2017), which then scaled to today's large language models.",
      ] },
    ],

    // ------------------------------------------------------------- mechanism
    mechanism: [
      { t: "h", text: "Era 1 — The first neurons (1940s–1960s)" },
      { t: "steps", items: [
        { title: "1943 — McCulloch and Pitts", text: "Warren McCulloch (a neurophysiologist) and Walter Pitts (a logician) published \"A Logical Calculus of the Ideas Immanent in Nervous Activity\". They modelled a neuron as a unit that outputs 1 if the sum of its binary inputs reaches a threshold, and showed that networks of such units can compute logical functions. The thresholds had to be set by hand: the model could not learn. Details: [The Artificial Neuron](/learn/m03/artificial-neuron)." },
        { title: "1949 — Hebb's learning idea", text: "Psychologist Donald Hebb proposed in *The Organization of Behavior* that connections between neurons that are repeatedly active together become stronger. \"Hebbian learning\" became an influential idea about how learning could happen through changing connection strengths — the conceptual ancestor of learned weights." },
        { title: "1956 — AI becomes a field", text: "The Dartmouth summer research project on artificial intelligence, proposed in 1955 by John McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon, gave the field its name. Early AI research was largely symbolic: logic, search and problem solving." },
        { title: "1958 — Rosenblatt's perceptron", text: "Frank Rosenblatt introduced the **perceptron**: a neuron with real-valued weights and a **learning rule** that adjusts them whenever it misclassifies a labelled example. He also built hardware implementations. For linearly separable data (classes that a straight line or flat plane can separate), the rule is guaranteed to find a separating set of weights — the perceptron convergence theorem." },
        { title: "1960 — Widrow and Hoff's ADALINE", text: "Bernard Widrow and Ted Hoff's ADALINE unit was trained with the least-mean-squares (LMS) rule, which adjusts weights in proportion to a continuous error rather than only on mistakes. It is essentially stochastic gradient descent on squared error for a linear unit — a direct ancestor of how networks are trained today ([Gradient Descent](/learn/m06/gradient-descent))." },
        { title: "1969 — Minsky and Papert's \"Perceptrons\"", text: "Marvin Minsky and Seymour Papert's book analysed mathematically what single-layer perceptrons can and cannot compute. Its best-known example is that no single perceptron can compute **XOR** (output 1 when exactly one input is 1). Multi-layer networks can represent XOR, but no practical method for training their hidden layers was widely known at the time." },
      ] },
      { t: "h", text: "The first AI winter (roughly the 1970s)" },
      { t: "p", text: "An **AI winter** is a period of reduced funding and interest following unmet expectations. Historians describe the first one as having several causes, affecting AI as a whole and not only neural networks:" },
      { t: "list", items: [
        "**Over-promising**: early researchers made optimistic predictions about how soon machines would translate languages or match human intelligence; progress fell far short.",
        "**Critical official reviews**: in the US, the 1966 ALPAC report judged machine-translation research disappointing, and funding for it was cut. In the UK, the 1973 Lighthill report criticised AI's failure to scale beyond toy problems and was followed by large cuts to AI funding there. US defence funding also moved towards shorter-term, mission-oriented projects.",
        "**Combinatorial explosion**: methods that worked on small puzzles failed on realistic problems, where the number of possibilities grows explosively.",
        "**Limited computers**: memory and processing power were orders of magnitude too small for the ideas being tried.",
        "**For neural networks specifically**: the limits of single-layer perceptrons were well established, and there was no widely known way to train multi-layer ones. Rosenblatt, the field's best-known advocate, died in 1971.",
      ] },
      { t: "callout", kind: "warning", title: "A contested point", text: "It is often said that Minsky and Papert's book \"killed\" neural network research. Historians disagree about how much weight to give it: the book's results concerned single-layer perceptrons, research did continue on a smaller scale, and the broader funding climate for all of AI was changing at the same time. A fair summary is that the book contributed to, but did not single-handedly cause, the decline of neural network research in the 1970s." },
      { t: "h", text: "Era 2 — Backpropagation and the connectionist revival (1970s–1990s)" },
      { t: "steps", items: [
        { title: "1970s — Reverse-mode differentiation and backpropagation", text: "The mathematics of backpropagation — computing all derivatives of a nested function efficiently by applying the chain rule backwards — was developed independently several times. Paul Werbos described its use for training neural networks in his 1974 Harvard PhD thesis. The general reverse-mode technique was described earlier in the automatic-differentiation literature (Seppo Linnainmaa's 1970 master's thesis is often cited). Details: [Backpropagation](/learn/m04/backpropagation)." },
        { title: "1980 — Fukushima's Neocognitron", text: "Kunihiko Fukushima's Neocognitron stacked layers of local feature detectors with increasing tolerance to position, inspired by David Hubel and Torsten Wiesel's studies of the visual cortex. It anticipated the architecture of convolutional networks, but was not trained end-to-end by backpropagation." },
        { title: "1982 — Hopfield networks", text: "John Hopfield described a recurrent network that stores patterns as low points of an \"energy\" function and recalls a stored pattern from a partial or noisy version (associative memory). It attracted physicists to the field and helped renew interest in neural networks." },
        { title: "1986 — Backpropagation popularised", text: "David Rumelhart, Geoffrey Hinton and Ronald Williams published \"Learning representations by back-propagating errors\" in *Nature*, demonstrating that backpropagation trains hidden layers to discover useful internal representations. Together with the *Parallel Distributed Processing* volumes of the same year, this triggered a wave of neural network research." },
        { title: "1989 — Convolutional networks and universal approximation", text: "Yann LeCun and colleagues trained a convolutional network with backpropagation to read handwritten ZIP-code digits (\"Backpropagation Applied to Handwritten Zip Code Recognition\"). In the same year George Cybenko, and separately Kurt Hornik, Maxwell Stinchcombe and Halbert White, proved **universal approximation** theorems: a network with one hidden layer and enough units can approximate any continuous function on a bounded region arbitrarily well ([Multi-Layer Networks](/learn/m03/multilayer-networks)). The theorem says such a network *exists*, not that training will find it or that it is efficient." },
        { title: "1991–1994 — The vanishing-gradient problem identified", text: "Sepp Hochreiter's 1991 diploma thesis and a 1994 paper by Yoshua Bengio, Patrice Simard and Paolo Frasconi analysed why gradient-based training struggles with long-range dependencies: error signals shrink (or blow up) exponentially as they are propagated back through many layers or time steps ([Vanishing and Exploding Gradients](/learn/m04/gradient-problems))." },
        { title: "1997 — LSTM", text: "Hochreiter and Jürgen Schmidhuber introduced Long Short-Term Memory, a recurrent unit with a gated, additively updated memory cell that lets gradients flow over long time spans ([LSTM](/learn/m09/lstm))." },
        { title: "1998 — LeNet-5 and MNIST", text: "LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner's \"Gradient-Based Learning Applied to Document Recognition\" presented LeNet-5, a convolutional network trained end-to-end for handwriting recognition, and the paper describes a cheque-reading system built with such networks. The MNIST digit dataset associated with this work became the standard benchmark for a generation ([CNN Architectures](/learn/m08/cnn-architectures))." },
      ] },
      { t: "h", text: "The second AI winter and the years of kernel methods (late 1980s–2000s)" },
      { t: "p", text: "In the late 1980s and early 1990s, AI funding fell again, mainly because of disappointment with **expert systems** — rule-based programs that had been heavily commercialised in the 1980s but proved expensive to build and maintain and brittle outside their narrow domain; the market for specialised AI hardware (Lisp machines) also collapsed around 1987. Neural networks did not disappear: LeNet and LSTM were developed in this period. But in mainstream machine learning they fell out of fashion, for practical reasons:" },
      { t: "list", items: [
        "**Training difficulty**: deeper networks trained slowly and unreliably with sigmoid units and the initialisations then in use; results depended on many hand-tuned choices.",
        "**Small data, slow computers**: most datasets had thousands, not millions, of examples, and CPUs made large networks impractical.",
        "**Strong alternatives**: **support vector machines** (Corinna Cortes and Vladimir Vapnik, 1995) and other **kernel methods** offered a convex training problem with a unique optimum, solid statistical theory and excellent results on the datasets of the time. Together with decision-tree ensembles and probabilistic graphical models, they dominated machine learning in the 2000s. Computer vision relied on hand-engineered features fed to such classifiers.",
      ] },
      { t: "h", text: "Era 3 — The deep learning revival (2006–2015)" },
      { t: "steps", items: [
        { title: "2006 — Layer-wise pretraining", text: "Geoffrey Hinton, Simon Osindero and Yee-Whye Teh's \"A Fast Learning Algorithm for Deep Belief Nets\" showed that a deep network could be trained by first learning one layer at a time without labels (each layer as a restricted Boltzmann machine), then fine-tuning the whole stack. Related work the same year (Hinton and Salakhutdinov's deep autoencoders) and soon after (Bengio and colleagues' greedy layer-wise training) showed the same idea. This renewed interest in *deep* networks. For supervised tasks with plenty of labels, this pretraining was later found to be unnecessary once better activations and initialisations were used." },
        { title: "2009 — Data and GPUs", text: "The **ImageNet** dataset (Jia Deng, Fei-Fei Li and colleagues, 2009) provided millions of labelled images; from 2010 the annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) used a subset with about 1.2 million training images in 1,000 categories. Meanwhile researchers began training neural networks on **graphics processing units (GPUs)** — chips built for the matrix arithmetic of computer graphics, which is the same arithmetic neural networks need. NVIDIA's CUDA platform (2007) made GPUs programmable for general computation." },
        { title: "2010–2011 — Better activations and initialisation", text: "Vinod Nair and Hinton (2010) used rectified linear units, and Xavier Glorot, Antoine Bordes and Bengio (2011) showed that **ReLU** ($\\max(0, z)$) networks train deep supervised networks well without pretraining ([Activation Functions](/learn/m03/activation-functions)). Glorot and Bengio (2010) analysed why deep sigmoid networks are hard to train and proposed the normalised (\"Xavier\") initialisation ([Weight Initialization](/learn/m07/initialization))." },
        { title: "2012 — AlexNet and speech recognition", text: "Alex Krizhevsky, Ilya Sutskever and Hinton's convolutional network, trained on two GPUs with ReLU and dropout, won ILSVRC-2012 with a top-5 test error of 15.3%, against 26.2% for the second-best entry. The size of that margin convinced much of the computer vision community to switch to deep learning. The same year, a joint paper from four research groups reported that deep neural networks substantially improved speech recognition acoustic models. **Dropout**, which randomly switches off units during training to reduce overfitting, was introduced by Hinton's group around this time ([Regularization](/learn/m07/regularization))." },
        { title: "2013–2014 — Representations, generation and sequences", text: "word2vec (2013) learned word embeddings at scale ([Word Embeddings](/learn/m13/word-embeddings)); variational autoencoders (2013) and generative adversarial networks (2014) launched modern deep generative modelling ([VAEs](/learn/m14/vae), [GANs](/learn/m14/gans)); sequence-to-sequence learning with LSTMs (Sutskever, Vinyals and Le, 2014) and **attention** (Bahdanau, Cho and Bengio, 2014) transformed machine translation ([Attention](/learn/m10/attention-intro)); the Adam optimiser (2014) became a default ([Adam](/learn/m06/momentum-adaptive))." },
        { title: "2015 — Training very deep networks", text: "**Batch normalisation** (Ioffe and Szegedy) stabilised and sped up training ([Normalization](/learn/m07/normalization)); He initialisation was derived for ReLU networks; and **residual networks** (He, Zhang, Ren and Sun), with identity shortcut connections, made networks of 100+ layers trainable and won ILSVRC-2015 ([Residual Connections](/learn/m07/residual-connections))." },
        { title: "2016 — AlphaGo", text: "DeepMind's AlphaGo, which combined deep neural networks with tree search and reinforcement learning, defeated the top professional Go player Lee Sedol 4–1 in March 2016 — a milestone many had expected to be a decade or more away ([Deep RL](/learn/m15/deep-rl))." },
      ] },
      { t: "h", text: "Era 4 — Transformers, scale and generative AI (2017–today)" },
      { t: "steps", items: [
        { title: "2017 — The Transformer", text: "Ashish Vaswani and colleagues' \"Attention Is All You Need\" replaced recurrence entirely with self-attention, letting every position in a sequence look at every other position in parallel. This made training on very large text corpora far more efficient on GPUs ([The Transformer](/learn/m11/transformer-architecture))." },
        { title: "2018–2020 — Pretraining and scale", text: "OpenAI's GPT (2018) and Google's BERT (2018) showed that a Transformer pretrained on large unlabelled text with a self-supervised objective could be fine-tuned to excel at many language tasks ([BERT, GPT, T5, ViT](/learn/m11/transformer-families)). GPT-2 (2019) and GPT-3 (2020, 175 billion parameters) showed abilities improving with scale, including learning new tasks from a few examples given in the prompt. Kaplan and colleagues (2020) measured smooth power-law **scaling laws** between loss and model size, data and compute ([Scaling Laws](/learn/m19/scaling-efficiency)). Vision Transformers (2020) carried the architecture to images." },
        { title: "2020+ — Diffusion and generative AI", text: "Diffusion models, introduced by Sohl-Dickstein and colleagues in 2015 and made practical by Ho, Jain and Abbeel's DDPM (2020), generate images by learning to reverse a gradual noising process ([Diffusion Models](/learn/m14/diffusion)). Latent diffusion (2021) made high-resolution text-to-image generation affordable and underlies Stable Diffusion (released 2022). Contrastive image–text models such as CLIP (2021) linked vision and language ([Multimodal Models](/learn/m15/multimodal))." },
        { title: "2020 — AlphaFold 2", text: "DeepMind's AlphaFold 2 predicted protein 3-D structures with accuracy far beyond previous methods in the CASP14 assessment (2020), a landmark for deep learning in science." },
        { title: "2022 — Instruction-tuned assistants", text: "Fine-tuning language models to follow instructions using human feedback (InstructGPT, 2022) led to conversational assistants; OpenAI's ChatGPT, released in November 2022, brought large language models to a mass audience ([Alignment](/learn/m20/alignment))." },
        { title: "Recognition", text: "Yoshua Bengio, Geoffrey Hinton and Yann LeCun received the 2018 ACM Turing Award for their work on deep learning. In 2024 the Nobel Prize in Physics went to John Hopfield and Geoffrey Hinton for foundational work on artificial neural networks, and half of the Nobel Prize in Chemistry went to Demis Hassabis and John Jumper for AlphaFold (the other half to David Baker for protein design)." },
      ] },
      { t: "h", text: "Why deep learning succeeded when it did" },
      { t: "p", text: "No single cause explains the timing. Three ingredients arrived together around 2009–2015, and each was necessary:" },
      { t: "table", head: ["Ingredient", "What changed", "Why it mattered", "Learn it in"], rows: [
        ["**Data**", "Internet-scale collections and crowdsourced labelling (ImageNet: about 1.2 million labelled training images for the challenge); later, web-scale text", "Deep networks have millions of parameters; without enough data they memorise instead of generalising", "[Training Basics](/learn/m03/training-basics)"],
        ["**Compute**", "GPUs made the matrix multiplications at the heart of neural networks one to two orders of magnitude faster than CPUs; later, specialised accelerators", "Experiments that would take months became feasible in days", "[Forward Propagation](/learn/m04/forward-propagation)"],
        ["**ReLU activations**", "$\\max(0,z)$ replaced sigmoid/tanh in hidden layers", "Derivative is 1 for active units, so gradients do not shrink layer by layer as with sigmoid", "[Activation Functions](/learn/m03/activation-functions)"],
        ["**Better initialisation**", "Xavier (2010) and He (2015) scale initial weights to the layer width", "Keeps signals and gradients at a stable size through many layers at the start of training", "[Weight Initialization](/learn/m07/initialization)"],
        ["**Dropout**", "Randomly zero units during training (2012–2014)", "Reduced overfitting of large networks", "[Regularization](/learn/m07/regularization)"],
        ["**Batch normalisation**", "Normalise each layer's inputs using mini-batch statistics (2015)", "Faster, more stable training; tolerates higher learning rates", "[Normalization](/learn/m07/normalization)"],
        ["**Residual connections**", "Add each block's input to its output (2015)", "Gives gradients a direct path, making 100+ layer networks trainable", "[Residual Connections](/learn/m07/residual-connections)"],
        ["**Better optimisers**", "Momentum and adaptive methods such as Adam (2014)", "Less learning-rate tuning, faster convergence", "[Momentum and Adam](/learn/m06/momentum-adaptive)"],
        ["**Software**", "Open-source GPU frameworks with automatic differentiation (e.g. Theano, Caffe, TensorFlow, PyTorch)", "Researchers no longer hand-derived gradients; ideas spread quickly as shared code", "[PyTorch](/learn/m16/pytorch)"],
      ] },
      { t: "p", text: "The experiments in the code sections below reproduce two parts of this story on a laptop: the perceptron's failure on XOR and backpropagation's success, and the dramatic difference that the activation function and initialisation make in a deeper network." },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "p", text: "Three small pieces of mathematics explain the main turning points: the perceptron rule (why it learns), XOR (why one layer is not enough), and the sigmoid derivative (why deep sigmoid networks were hard to train)." },
      { t: "h", text: "1. The perceptron learning rule (1958)" },
      { t: "math", tex: String.raw`\hat y = \begin{cases} 1 & \text{if } w_1x_1 + w_2x_2 + b > 0 \\ 0 & \text{otherwise}\end{cases} \qquad w_j \leftarrow w_j + \eta\,(y - \hat y)\,x_j, \qquad b \leftarrow b + \eta\,(y - \hat y)`, caption: "$x_j$: inputs; $w_j$: weights; $b$: bias; $y$: correct label; $\\hat y$: prediction; $\\eta$: learning rate (1 here). The update is zero when the prediction is right." },
      { t: "p", text: "Worked trace on AND with $w = (0, 0)$, $b = 0$, $\\eta = 1$, visiting the inputs in the order (0,0), (0,1), (1,0), (1,1):" },
      { t: "steps", items: [
        { title: "Epoch 1, inputs (0,0), (0,1), (1,0)", text: "$z = 0$ each time, so $\\hat y = 0$, which is correct (label 0). No updates." },
        { title: "Epoch 1, input (1,1), label 1", tex: String.raw`z = 0 \Rightarrow \hat y = 0 \ne 1, \quad w \leftarrow (0,0) + 1\cdot(1-0)\cdot(1,1) = (1,1), \quad b \leftarrow 0 + 1 = 1` },
        { title: "Epoch 2, input (0,0), label 0", tex: String.raw`z = 1 > 0 \Rightarrow \hat y = 1 \ne 0, \quad w \leftarrow (1,1) - (0,0) = (1,1), \quad b \leftarrow 1 - 1 = 0` },
        { title: "… epochs 2–5 continue correcting mistakes …", text: "Every mistake moves the boundary towards classifying that point correctly." },
        { title: "Final weights (no mistakes in epoch 6)", tex: String.raw`w = (2, 1),\ b = -2: \quad z = (-2,\,-1,\,0,\,1) \Rightarrow \hat y = (0, 0, 0, 1)` },
      ] },
      { t: "p", text: "This is exactly what the scratch code prints (\"converged after 6 epochs\"). Because AND is linearly separable, the perceptron convergence theorem guarantees this loop ends." },
      { t: "h", text: "2. Why one layer cannot do XOR (1969)" },
      { t: "p", text: String.raw`XOR needs $\hat y = 1$ for (0,1) and (1,0) and $\hat y = 0$ for (0,0) and (1,1). With threshold unit $z = w_1x_1 + w_2x_2 + b$ that means $b \le 0$, $w_2 + b > 0$, $w_1 + b > 0$ and $w_1 + w_2 + b \le 0$. Adding the middle two gives $w_1 + w_2 + 2b > 0$, i.e. $w_1 + w_2 + b > -b \ge 0$, contradicting the last condition. No weights exist. (The full argument and pictures are in [The Artificial Neuron](/learn/m03/artificial-neuron).) A hidden layer fixes this: for example, hidden units computing OR and AND, combined as "OR and not AND", give XOR.` },
      { t: "h", text: "3. Why deep sigmoid networks were hard to train" },
      { t: "p", text: String.raw`Backpropagation multiplies, for each layer the error passes through, by the derivative of that layer's activation (and by the weights). The sigmoid's derivative is` },
      { t: "math", tex: String.raw`\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big) \le \tfrac14`, caption: "The maximum $1/4$ occurs at $z = 0$, where $\\sigma(0) = 0.5$ and $0.5 \\times 0.5 = 0.25$. At $z = 2$ it is already $0.880797 \\times 0.119203 = 0.104994$." },
      { t: "p", text: "So, ignoring the weights, the activation derivatives alone can shrink the gradient by a factor of at least 4 per layer:" },
      { t: "steps", items: [
        { title: "Through 2 sigmoid layers", tex: String.raw`\left(\tfrac14\right)^2 = 0.0625` },
        { title: "Through 8 sigmoid layers", tex: String.raw`\left(\tfrac14\right)^8 = 1.52587890625 \times 10^{-5}` },
        { title: "Through 8 ReLU layers (active units)", tex: String.raw`\mathrm{ReLU}'(z) = 1 \text{ for } z > 0 \;\Rightarrow\; 1^8 = 1` },
      ] },
      { t: "p", text: String.raw`This is a bound, not an exact prediction — the weights also multiply in, which is why initialisation matters. But it matches the experiment in the Framework section: at initialisation the gradient reaching the first layer of an 8-hidden-layer sigmoid network was $5.86 \times 10^{-6}$, versus $5.53 \times 10^{-2}$ for a 2-hidden-layer sigmoid network — about 9,400 times smaller — and the deep sigmoid network did not learn at all in 30 epochs. With ReLU and He initialisation, the same depth trained to 97% test accuracy.` },
    ],

    // ---------------------------------------------------------------- visual
    visual: [
      { t: "h", text: "Relive 1958–1969: the perceptron and XOR" },
      { t: "sim", id: "neuron", caption: "A single neuron trained with Rosenblatt's perceptron rule on AND, OR and XOR." },
      { t: "list", items: [
        "Select **AND** and press **Play**: the decision line moves after each mistake and stops once all four points are on the correct side — the perceptron convergence theorem in action.",
        "Select **XOR** and press **Play**: the line keeps jumping and every epoch still contains mistakes. No straight line can put (0,1) and (1,0) on one side and (0,0), (1,1) on the other. This is the limitation analysed by Minsky and Papert in 1969.",
        "Try changing the starting weights and learning rate on XOR. Nothing helps — the problem is the model, not the training settings. That is what made the missing training method for *multi-layer* networks so important.",
      ] },
      { t: "h", text: "Relive 1986–2015: hidden layers, backpropagation and vanishing gradients" },
      { t: "sim", id: "network", caption: "A multi-layer network with real forward and backward passes; every gradient shown is computed by backpropagation." },
      { t: "list", items: [
        "Step through a forward pass and then the backward pass. The backward values are the error signals the 1986 paper made famous: each hidden neuron learns how much it contributed to the error.",
        "Use the **+ Layer** button to build a deeper network, set every hidden activation to **Sigmoid**, and step through the backward pass. Compare the size of the gradients at the last layer with those at the first layer.",
        "Now switch the hidden activations to **ReLU** and repeat. In the narration of each backward step, look at the activation-derivative factor: for sigmoid it is never more than 0.25, while for an active ReLU unit it is exactly 1 (and 0 for an inactive one). With only a few layers the effect is modest; the Framework section below measures it in an 8-layer network, where it decides whether the network learns at all.",
      ] },
    ],

    // --------------------------------------------------------------- scratch
    scratch: [
      { t: "p", text: "Two historical results reproduced honestly with NumPy: (1) Rosenblatt's perceptron rule succeeds on AND and fails on XOR; (2) a network with one hidden layer, trained by backpropagation as in 1986, learns XOR. We also measure how *reliably* it does so, because the real history includes networks that get stuck." },
      { t: "code", lang: "python", title: "perceptron_vs_backprop.py", code: XOR_HISTORY, output: XOR_HISTORY_OUT },
      { t: "list", items: [
        "**Perceptron on AND** converges in 6 epochs (the hand trace in the Mathematical Explanation section). **On XOR** it is still wrong after 20 epochs — and would be forever.",
        "**The 2-2-1 network** spends hundreds of epochs on a **plateau** (loss ≈ 0.25 at epoch 500, which is what predicting 0.5 for everything gives) and then suddenly drops to 0.0036 by epoch 2000. Plateaus like this were a practical frustration of early backpropagation.",
        "**Reliability**: with only 2 hidden units, 4 of 20 random starting points got stuck at a poor solution (a local minimum or very flat region) within 10,000 epochs; with 4 hidden units all 20 succeeded. Extra capacity makes the optimisation easier — a lesson that recurs throughout deep learning.",
        "The backward pass is the chain rule: `d_out` is $\\partial L/\\partial z$ at the output, and `d_h` passes it back through `W2` and multiplies by the sigmoid derivative $h(1-h)$. [Backpropagation](/learn/m04/backpropagation) derives these lines step by step.",
        "The learning rate 1.0 and 10,000 epochs are simply values large enough for this tiny problem to converge in about a second; the standard-normal initial weights match the small random initialisations used at the time.",
      ] },
    ],

    // ------------------------------------------------------------- framework
    framework: [
      { t: "p", text: "One reason the 2010s worked, measured: the same 8-hidden-layer network on scikit-learn's digits data, trained identically (SGD, learning rate 0.1, batch 64, 30 epochs), changing only the activation function and the matching initialisation." },
      { t: "tabs", tabs: [
        { label: "PyTorch", blocks: [
          { t: "code", lang: "python", title: "sigmoid_vs_relu_depth.py", code: SIGMOID_RELU, output: SIGMOID_RELU_OUT },
          { t: "list", items: [
            "**Deep sigmoid (8 layers) does not learn at all**: the loss stays at 2.30 ≈ $\\ln 10$, the loss of guessing uniformly among 10 classes, and test accuracy is 0.100 — chance. The gradient reaching the first layer is $5.86\\times10^{-6}$.",
            "**Shallow sigmoid (2 layers) learns, slowly**: loss 1.670 after 30 epochs, 0.665 test accuracy. Sigmoid networks of modest depth *were* trainable — they were just slow, and got worse with depth.",
            "**Deep ReLU + He initialisation** trains to loss 0.002 and 0.970 test accuracy. Its first-layer gradient is about 60,000 times larger than the deep sigmoid's.",
            "**Deep tanh + Xavier also works** (0.974). This is an honest and important nuance: the fix is not \"ReLU is magic\" but using an activation and initialisation that keep signals and gradients at a healthy size. Tanh is zero-centred with maximum derivative 1; sigmoid is not zero-centred and its derivative is at most 0.25. Glorot and Bengio's 2010 paper analysed exactly this.",
            "Caveat: these are single runs on a small dataset with one learning rate. A carefully tuned sigmoid network can do better than shown — but the qualitative gap with depth is robust and is what practitioners experienced.",
          ] },
        ] },
        { label: "TensorFlow / Keras", blocks: [
          { t: "code", lang: "python", title: "sigmoid_vs_relu_depth_keras.py (not executed in CI)", code: SIGMOID_RELU_KERAS, output: "Prints the final training loss and test accuracy for an 8-hidden-layer sigmoid network with Glorot initialisation and for the same network with ReLU and He initialisation. Expect the sigmoid network to stay near loss 2.30 and chance accuracy (about 0.1) and the ReLU network to reach high accuracy (above 0.9); exact values depend on TensorFlow's random seed." },
          { t: "callout", kind: "tip", text: "In Keras, `glorot_normal` is Xavier initialisation and `he_normal` is He initialisation; `Dense` layers default to `glorot_uniform`." },
        ] },
      ] },
    ],

    // ------------------------------------------------------------ variations
    variations: [
      { t: "p", text: "The history can be read as a sequence of model families, each dominant for a while. All still exist and are still used where they fit." },
      { t: "table", head: ["Family (period of peak influence)", "Key idea", "Advantages", "Disadvantages", "Still used for"], rows: [
        ["Threshold logic / perceptron (1940s–60s)", "Single layer of threshold units; mistake-driven learning", "Simple, guaranteed to converge on separable data", "Only linear decision boundaries", "Teaching; the perceptron update survives in online linear classifiers"],
        ["Symbolic AI & expert systems (1950s–80s)", "Hand-written rules, logic, search", "Transparent, exact, no data needed", "Brittle; experts cannot articulate perception or language", "Planning, verification, business rules"],
        ["Shallow neural networks + backprop (1986–1990s)", "One or two hidden layers trained by gradient descent", "Learns non-linear functions and internal features", "Slow, sensitive to settings; deeper versions hard to train", "Small regression/classification problems"],
        ["Kernel methods / SVMs (late 1990s–2000s)", "Linear separation in an implicit high-dimensional feature space", "Convex training, strong theory, great on small data", "Scale poorly to millions of examples; rely on hand-made features for images and text", "Small and medium datasets"],
        ["Tree ensembles (2000s–today)", "Many decision trees combined (random forests, gradient boosting)", "Excellent on tabular data, little tuning", "Not suited to raw images, audio or text", "Tabular prediction in industry"],
        ["Deep CNNs (2012–)", "Convolutions + depth + GPUs", "Learned visual features; state of the art in vision for a decade", "Data- and compute-hungry", "Vision backbones, mobile models"],
        ["Recurrent networks / LSTM (1997; peak 2014–2017)", "Process sequences step by step with memory", "Handles variable-length sequences", "Sequential and slow; long-range memory still limited", "Small streaming and on-device tasks"],
        ["Transformers (2017–)", "Self-attention over all positions in parallel", "Scales to huge data; one architecture for text, images, audio", "Cost grows with the square of sequence length; very large compute budgets", "Language models, vision, multimodal systems"],
        ["Diffusion models (2020–)", "Learn to reverse gradual noising", "High-quality, controllable image/audio/video generation", "Many denoising steps at generation time", "Image and video generation"],
      ] },
      { t: "h", text: "The two AI winters compared" },
      { t: "table", head: ["", "First winter (roughly 1974–1980)", "Second winter (roughly 1987–early 1990s)"], rows: [
        ["Preceded by", "Optimistic early symbolic AI, perceptron enthusiasm", "Commercial boom in expert systems"],
        ["Main causes", "Unmet promises; ALPAC (1966) and Lighthill (1973) reports; combinatorial explosion; weak computers; limits of single-layer perceptrons", "Expert systems costly and brittle; collapse of the specialised Lisp-machine market; ambitious national programmes falling short of goals"],
        ["Effect on neural networks", "Research greatly reduced (but not stopped)", "Neural networks survived but were overtaken by SVMs and other methods in the 1990s–2000s"],
      ] },
      { t: "callout", kind: "note", text: "Exact start and end years of the AI winters are a matter of convention; different historians give different ranges, and the experience differed between countries and subfields." },
    ],

    // ---------------------------------------------------------- applications
    applications: [
      { t: "p", text: "Each historical milestone lives on in systems in use today (described generically):" },
      { t: "list", items: [
        "**Document reading**: the line from the 1989 ZIP-code network and LeNet-5 leads to today's optical character recognition for forms, receipts, cheques and handwriting.",
        "**Image understanding**: the AlexNet → ResNet line underpins photo search, medical-image analysis, industrial inspection and the perception systems of driver-assistance features ([Object Detection](/learn/m12/object-detection)).",
        "**Speech**: the 2012 shift to deep acoustic models is why voice assistants and automatic captioning became usable.",
        "**Translation and language**: sequence-to-sequence models and attention, then Transformers, power machine translation, search, writing assistants and chat assistants.",
        "**Science**: AlphaFold-style structure prediction is used in biology research; deep learning is also used for weather prediction and materials discovery.",
        "**Generation**: diffusion models and Transformers generate images, audio, video and code from text prompts.",
        "**Game playing and control**: the AlphaGo combination of learned networks and search informs systems for planning and control ([Deep RL](/learn/m15/deep-rl)).",
      ] },
    ],

    // ------------------------------------------------------------- tradeoffs
    tradeoffs: [
      { t: "p", text: "What the history teaches about the strengths and limits of the deep learning approach:" },
      { t: "list", items: [
        "**Strength — ideas compound**: modern networks combine 1940s neurons, 1980s backprop and convolutions, 1990s LSTMs and 2010s training tricks. Old ideas became powerful when conditions changed.",
        "**Strength — scale keeps paying (so far)**: in many domains, bigger models trained on more data with more compute have kept improving, which earlier approaches did not.",
        "**Limitation — dependence on data and compute**: the same fact cuts the other way. Progress has been concentrated where data and hardware are plentiful, and training frontier models is very expensive.",
        "**Limitation — theory lags practice**: much of what works (ReLU, batch norm, residual connections, scaling) was found empirically and explained later, if at all.",
        "**Risk — hype cycles**: both AI winters followed periods of over-promising. Predictions about timelines have repeatedly been wrong in both directions — too optimistic in the 1960s and 1980s, too pessimistic about Go and image recognition in the 2000s.",
        "**Risk — credit and narrative simplification**: popular histories compress decades of work by many people into a few names; this distorts our understanding of how progress actually happens.",
      ] },
    ],

    // -------------------------------------------------------------- mistakes
    mistakes: [
      { t: "list", items: [
        "**\"Deep learning was invented in 2012.\"** AlexNet combined existing ideas (CNNs from the 1980s–90s, backprop, ReLU, dropout) with GPUs and ImageNet. Fix: describe 2012 as the moment the combination clearly worked at scale.",
        "**\"Hinton invented backpropagation in 1986.\"** Reverse-mode differentiation and its use for neural networks were developed earlier (e.g. Linnainmaa 1970, Werbos 1974) and independently by several people. The 1986 paper's contribution was demonstrating that it learns useful hidden representations and bringing it to wide attention.",
        "**\"Minsky and Papert proved neural networks are useless.\"** Their analysis concerned single-layer perceptrons; multi-layer networks can compute XOR. And the first winter had many causes beyond one book.",
        "**\"Neural networks disappeared between 1970 and 2006.\"** Hopfield networks, backprop, CNNs and LSTMs all came from this period. They were less fashionable, not gone.",
        "**\"GPUs alone caused the deep learning boom.\"** Without large labelled datasets and without fixes for training deep networks (ReLU, initialisation, normalisation), faster hardware would only have produced faster failures — as the deep sigmoid run above shows, it trained quickly to nowhere.",
        "**\"Transformers invented attention.\"** Attention for neural machine translation was introduced in 2014 (Bahdanau, Cho and Bengio). The Transformer's novelty (2017) was building the whole model from attention, without recurrence.",
        "**\"The universal approximation theorem explains why deep learning works.\"** It shows a one-hidden-layer network *can* represent any continuous function, not that training will find it, nor that it will generalise, nor why depth helps in practice.",
        "**Repeating unverified anecdotes.** Many colourful stories about AI history circulate without sources. Fix: go back to the original papers (see [Reading and Reproducing Research Papers](/learn/m19/reading-papers)) and flag what is contested.",
      ] },
    ],

    // ------------------------------------------------------------- interview
    interview: [
      { t: "qa", level: "beginner", q: "What was the perceptron, and what was its main limitation?", a: "Rosenblatt's perceptron (1958) is a single artificial neuron with a step activation and a learning rule that adjusts the weights whenever it misclassifies an example. It can only learn linearly separable problems — those where a straight line (or flat hyperplane) separates the classes. XOR is the classic counterexample, analysed in Minsky and Papert's 1969 book." },
      { t: "qa", level: "beginner", q: "What is an AI winter?", a: "A period of reduced funding and interest in AI following unmet expectations. The first (roughly the 1970s) followed over-optimistic predictions, critical government reports such as the 1973 Lighthill report, and the limits of available computers and methods. The second (late 1980s to early 1990s) followed the collapse of the commercial expert-systems boom." },
      { t: "qa", level: "beginner", q: "Why is AlexNet (2012) considered a turning point?", a: "It won the ImageNet ILSVRC-2012 classification challenge with a top-5 error of 15.3% versus 26.2% for the runner-up, using a deep CNN trained on GPUs with ReLU and dropout. The large margin on a large, respected benchmark convinced the computer vision community — and then other fields — to adopt deep learning." },
      { t: "qa", level: "intermediate", q: "What three factors explain why deep learning took off around 2012 rather than in 1990?", a: "Data: large labelled datasets such as ImageNet (about 1.2 million training images for the challenge) let large networks generalise. Compute: GPUs made training large networks feasible in days. Algorithms: ReLU, better initialisation, dropout and later batch normalisation and residual connections made deep networks trainable and less prone to overfitting. Open-source autodiff frameworks also lowered the cost of experimentation." },
      { t: "qa", level: "intermediate", q: "Why were SVMs preferred to neural networks in the 2000s?", a: "SVMs have a convex training objective with a single global optimum, so results are reproducible and need less tuning; they come with generalisation theory; and with good hand-made features and kernels they matched or beat neural networks on the small and medium datasets of the time. Neural networks were non-convex, slow to train on CPUs, sensitive to initialisation and hyperparameters, and deeper ones trained poorly with sigmoid units." },
      { t: "qa", level: "intermediate", q: "Explain the vanishing-gradient problem in historical context.", a: "Backpropagation multiplies by each layer's activation derivative (and weights) as the error flows backward. With sigmoid units the derivative is at most 0.25, so gradients can shrink exponentially with depth or with time steps in RNNs; early layers barely learn. It was analysed by Hochreiter (1991) and Bengio, Simard and Frasconi (1994), and addressed by LSTM gates (1997), and for feedforward networks by ReLU, Xavier/He initialisation, batch normalisation and residual connections (2010–2015)." },
      { t: "qa", level: "advanced", q: "What role did layer-wise unsupervised pretraining (2006) play, and why is it rarely used for supervised networks today?", a: "Hinton, Osindero and Teh (2006) showed that training a deep belief net one layer at a time without labels, then fine-tuning, produced good deep models when direct training failed — renewing interest in depth. Later work showed that with ReLU activations, proper initialisation, enough labelled data and GPUs (as in AlexNet, which used no pretraining), deep supervised networks train directly, so the extra stage became unnecessary. The broader idea — pretrain on unlabelled data, then fine-tune — returned in a different form as self-supervised pretraining of Transformers." },
      { t: "qa", level: "advanced", q: "Why did the Transformer, rather than better RNNs, become the basis of large language models?", a: "RNNs process tokens sequentially, so training cannot be parallelised across the sequence, and information must pass through many steps to connect distant tokens. Self-attention connects every pair of positions in one step and computes all positions in parallel with large matrix multiplications that GPUs handle efficiently. That made it practical to train on vastly more text, and performance kept improving with scale (scaling laws), whereas the sequential bottleneck limited how far RNNs could be scaled." },
      { t: "qa", level: "advanced", q: "How would you evaluate the claim \"Perceptrons (1969) caused the first AI winter\"?", a: "Partly true but oversimplified. The book rigorously established limits of single-layer perceptrons and is widely seen as having discouraged neural network funding. But the first winter affected all of AI, including symbolic approaches untouched by the book, and was driven by broader factors: over-promising, the ALPAC and Lighthill reports, shifts in US defence funding, combinatorial explosion and weak hardware. Historians debate the book's relative weight, so a careful answer presents it as one contributing factor." },
    ],

    // -------------------------------------------------------------- practice
    practice: [
      { t: "quiz", questions: [
        { q: "Put these in chronological order: (A) Rumelhart–Hinton–Williams backprop paper, (B) Rosenblatt's perceptron, (C) Transformer, (D) AlexNet.", options: ["B, A, D, C", "A, B, D, C", "B, A, C, D", "B, D, A, C"], answer: 0, explain: "Perceptron 1958, backprop paper 1986, AlexNet 2012, Transformer 2017." },
        { q: "What did Minsky and Papert's 1969 analysis mainly show?", options: ["That backpropagation does not converge", "That single-layer perceptrons cannot compute some simple functions such as XOR", "That neural networks can approximate any function", "That GPUs are needed for neural networks"], answer: 1, explain: "The book analysed the representational limits of single-layer perceptrons; XOR is the best-known example." },
        { kind: "numeric", q: "In what year was \"Attention Is All You Need\", the Transformer paper, published?", answer: 2017, tolerance: 0, explain: "Vaswani et al., 2017." },
        { kind: "numeric", q: "What is the maximum value of the sigmoid's derivative $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$?", answer: 0.25, tolerance: 0.001, explain: "At $z = 0$, $\\sigma = 0.5$, so $0.5 \\times 0.5 = 0.25$." },
        { kind: "numeric", q: "Ignoring weights, what is the largest factor by which 3 sigmoid layers can scale a gradient? (decimal)", answer: 0.015625, tolerance: 0.0001, explain: "$0.25^3 = 0.015625$." },
        { q: "Which combination best explains the timing of the deep learning breakthrough around 2012?", options: ["A single new algorithm invented that year", "Large labelled datasets + GPU computing + training improvements such as ReLU and dropout", "The invention of the perceptron", "The end of research on SVMs"], answer: 1, explain: "Data, compute and algorithmic improvements arrived together; none alone was sufficient." },
        { q: "Which method dominated much of machine learning in the 2000s?", options: ["Transformers", "Kernel methods such as support vector machines", "Diffusion models", "Generative adversarial networks"], answer: 1, explain: "SVMs (1995) and other kernel methods, along with tree ensembles and graphical models, were the mainstream before the deep learning revival." },
        { q: "In our experiment, the 8-hidden-layer sigmoid network stayed at loss ≈ 2.30. Why that particular number?", options: ["It is the learning rate", "It is $\\ln 10$, the cross-entropy of predicting all 10 classes with equal probability", "It is a bug in PyTorch", "It is the number of layers divided by 3"], answer: 1, explain: "A network that has learned nothing outputs a near-uniform distribution; the cross-entropy is then $-\\ln(1/10) = \\ln 10 \\approx 2.303$." },
      ] },
      { t: "exercise", title: "How fast do sigmoid gradients vanish?", prompt: "Using the bound $\\sigma'(z) \\le 0.25$ and ignoring the weights: (a) by what factor can a gradient shrink through 5 sigmoid layers? (b) What is the smallest number of sigmoid layers for which this bound falls below $10^{-6}$? (c) What is the corresponding factor for 5 ReLU layers whose units are all active?", hint: "For (b) solve $0.25^n < 10^{-6}$ by taking logarithms: $n > 6 / \\log_{10} 4$.", solution: [
        { t: "steps", items: [
          { title: "(a) Five sigmoid layers", tex: String.raw`0.25^5 = 0.0009765625 \approx 9.8 \times 10^{-4}` },
          { title: "(b) Solve for n", tex: String.raw`n > \frac{6}{\log_{10} 4} = \frac{6}{0.60206} = 9.966 \;\Rightarrow\; n = 10, \quad 0.25^{10} = 9.54 \times 10^{-7}` },
          { title: "Check n = 9", tex: String.raw`0.25^{9} = 3.81 \times 10^{-6} > 10^{-6}` },
          { title: "(c) Five active ReLU layers", tex: String.raw`1^5 = 1` },
        ] },
        { t: "p", text: "Ten sigmoid layers can shrink the gradient a million-fold from the activations alone. In practice the weights can partly compensate or make it worse, which is why initialisation schemes (Xavier, He) were derived to keep the overall scale near 1." },
      ] },
      { t: "exercise", title: "Rescue the perceptron with a hand-made feature", prompt: "Before hidden layers could be trained, a common way around the XOR limitation was to add a hand-designed feature. Add the product $x_1 x_2$ as a third input and train the perceptron rule on XOR. Does it converge? Explain why, and what this says about feature engineering versus learned features.", hint: "With the extra feature the four points live in 3-D. Is there now a plane separating the classes?", solution: [
        { t: "code", lang: "python", title: "xor_with_product_feature.py", code: EX_XOR_FEATURE, output: EX_XOR_FEATURE_OUT },
        { t: "p", text: "It converges in 8 epochs with $w = (1, 1, -3)$, $b = 0$: the score $z = x_1 + x_2 - 3x_1x_2$ is 0, 1, 1, −1 for the four inputs, so only (0,1) and (1,0) have $z > 0$. In the 3-D feature space the classes are linearly separable. The catch is that a *human* had to know that the product was the right feature. A hidden layer trained by backpropagation discovers such features itself — exactly the shift from feature engineering to representation learning described in [AI, Machine Learning and Deep Learning](/learn/m01/ai-ml-dl)." },
      ] },
    ],

    // --------------------------------------------------------------- summary
    summary: [
      { t: "table", head: ["Year", "Milestone", "Why it mattered", "Lesson"], rows: [
        ["1943", "McCulloch & Pitts threshold neuron", "First mathematical model of a neuron; networks compute logic", "[Artificial Neuron](/learn/m03/artificial-neuron)"],
        ["1958", "Rosenblatt's perceptron", "First trainable neural network with a learning rule", "[Artificial Neuron](/learn/m03/artificial-neuron)"],
        ["1969", "Minsky & Papert, *Perceptrons*", "Rigorous limits of single-layer networks (XOR); contributed to the first AI winter", "[Artificial Neuron](/learn/m03/artificial-neuron)"],
        ["1974 / 1986", "Backpropagation (Werbos thesis; Rumelhart, Hinton & Williams)", "A way to train hidden layers, which learn internal representations", "[Backpropagation](/learn/m04/backpropagation)"],
        ["1989 / 1998", "LeCun et al.: CNN for ZIP codes; LeNet-5", "Convolutional networks trained end-to-end on real images", "[Convolution](/learn/m08/convolution), [CNN Architectures](/learn/m08/cnn-architectures)"],
        ["1989", "Universal approximation theorems", "One hidden layer can represent any continuous function (existence only)", "[Multi-Layer Networks](/learn/m03/multilayer-networks)"],
        ["1991 / 1994", "Vanishing gradients analysed", "Explained why deep and recurrent nets were hard to train", "[Gradient Problems](/learn/m04/gradient-problems)"],
        ["1995", "Support vector machines", "Convex, theory-backed rival that dominated the 2000s", "[Classification Losses](/learn/m05/classification-losses)"],
        ["1997", "LSTM", "Gated memory for long sequences", "[LSTM](/learn/m09/lstm)"],
        ["2006", "Deep belief nets, layer-wise pretraining", "Showed deep networks could be trained; renewed interest", "[Autoencoders](/learn/m14/autoencoders)"],
        ["2009", "ImageNet; GPU training", "The data and compute that deep networks needed", "[Transfer Learning](/learn/m12/transfer-learning)"],
        ["2010–2011", "ReLU; Xavier initialisation", "Deep supervised networks trainable without pretraining", "[Activation Functions](/learn/m03/activation-functions), [Initialization](/learn/m07/initialization)"],
        ["2012", "AlexNet; dropout; deep speech models", "Deep learning wins decisively on large benchmarks", "[CNN Architectures](/learn/m08/cnn-architectures), [Regularization](/learn/m07/regularization)"],
        ["2014", "Attention; GANs; seq2seq; Adam", "Better sequence models, generative models and optimisers", "[Attention](/learn/m10/attention-intro), [GANs](/learn/m14/gans)"],
        ["2015", "Batch normalisation; ResNet", "Fast, stable training of 100+ layer networks", "[Normalization](/learn/m07/normalization), [Residual Connections](/learn/m07/residual-connections)"],
        ["2017", "Transformer", "Parallel attention-only architecture that scales", "[The Transformer](/learn/m11/transformer-architecture)"],
        ["2018–2020", "GPT, BERT, GPT-3, scaling laws", "Self-supervised pretraining at scale", "[Transformer Families](/learn/m11/transformer-families), [Scaling Laws](/learn/m19/scaling-efficiency)"],
        ["2020–2022", "DDPM, latent diffusion, CLIP, instruction tuning", "Modern generative AI and assistants", "[Diffusion](/learn/m14/diffusion), [Alignment](/learn/m20/alignment)"],
      ] },
      { t: "list", items: [
        "Most core ideas are decades old; the 2010s breakthrough came from **data + compute + training improvements** together.",
        "Single-layer perceptrons are limited to linear boundaries; hidden layers trained by backpropagation remove that limit.",
        "The AI winters had multiple causes (over-promising, critical reviews, weak hardware, unscalable methods); single-cause stories are oversimplified.",
        "Sigmoid's derivative is at most 0.25, so deep sigmoid nets suffer vanishing gradients; ReLU/tanh with Xavier/He initialisation, normalisation and residual connections fixed this.",
        "Measured here: perceptron fails on XOR; a 2-2-1 backprop net solves it from 16 of 20 starts (20 of 20 with 4 hidden units); an 8-layer sigmoid net stays at chance while ReLU + He reaches 0.970.",
      ] },
      { t: "math", tex: String.raw`w \leftarrow w + \eta\,(y-\hat y)\,x, \qquad \sigma'(z) = \sigma(z)(1-\sigma(z)) \le \tfrac14, \qquad \left(\tfrac14\right)^{L} \to 0 \text{ as } L \text{ grows}` },
    ],

    // ------------------------------------------------------------------ next
    next: [
      { t: "list", items: [
        "[The Deep Learning Workflow](/learn/m01/dl-workflow) — how a modern deep learning project is actually run, from data to deployment.",
        "[The Artificial Neuron and the Perceptron](/learn/m03/artificial-neuron) — the 1943–1969 part of this story in full technical detail, with the perceptron convergence proof and the XOR impossibility argument.",
        "[Backpropagation](/learn/m04/backpropagation) — the 1986 algorithm derived step by step on a worked example.",
        "[Vanishing and Exploding Gradients](/learn/m04/gradient-problems) — the obstacle that held back deep networks, and the fixes, explained mathematically.",
      ] },
    ],
  },
};

export default lesson;
