import type { Lesson } from "@/lib/types";

const MM_NUMPY = String.raw`import re
import numpy as np


def normalize(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def log_softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    return S - np.log(np.exp(S).sum(axis=1, keepdims=True))


# ================= 1. CLIP's symmetric contrastive loss =================
img = normalize(np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.2, 0.9]]))   # 3 image embeddings
txt = normalize(np.array([[1.0, 0.0, 0.1], [0.3, 0.9, 0.0], [0.1, 0.4, 0.8]]))   # their 3 captions, same order
tau = 0.1
logits = img @ txt.T / tau                       # logits[i, j] = cos(image i, text j) / tau
loss_i2t = -np.mean(np.diag(log_softmax(logits)))       # each image must pick its caption among all texts
loss_t2i = -np.mean(np.diag(log_softmax(logits.T)))     # each caption must pick its image among all images
print("cosine matrix (rows = images, columns = texts):\n", np.round(img @ txt.T, 4))
print(f"image->text {loss_i2t:.4f}   text->image {loss_t2i:.4f}   CLIP loss {(loss_i2t + loss_t2i) / 2:.4f}")

# zero-shot classification: the "classifier weights" are text embeddings of prompts, one per class
classes = ["cat", "car", "tree"]
prompt_emb = txt                                 # pretend these embed "a photo of a {class}"
query = normalize(np.array([0.2, 0.7, 0.35]))    # embedding of a new image
probs = np.exp(prompt_emb @ query / tau)
probs /= probs.sum()
print("zero-shot probabilities:", {c: round(float(p), 4) for c, p in zip(classes, probs)})

# ================= 2. retrieval-augmented generation: the retrieval half =================
document = (
    "Gradient descent updates parameters in the direction opposite to the gradient. "
    "The learning rate controls the size of each step and must be tuned. "
    "Adam keeps running averages of the gradient and of its square for every parameter. "
    "Dropout randomly zeroes activations during training to reduce overfitting. "
    "Batch normalization standardizes each feature over the mini-batch. "
    "A convolution slides a small kernel over the image and shares its weights across positions. "
    "Pooling layers shrink feature maps and add a little translation invariance. "
    "Attention computes a weighted average of value vectors using query-key similarities. "
    "The transformer stacks attention and feed-forward layers with residual connections. "
    "CLIP trains an image encoder and a text encoder so that matching pairs have high cosine similarity."
)


def chunk(text, size=20, overlap=5):
    """Split into chunks of size words, each sharing overlap words with the previous chunk."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def tokens(s):
    return re.findall(r"[a-z]+", s.lower())


chunks = chunk(document)
vocab = sorted({w for c in chunks for w in tokens(c)})
index = {w: i for i, w in enumerate(vocab)}


def tf(s):
    v = np.zeros(len(vocab))
    for w in tokens(s):
        if w in index:
            v[index[w]] += 1
    return v


TF = np.array([tf(c) for c in chunks])
df = (TF > 0).sum(axis=0)                                  # number of chunks containing each word
idf = np.log((1 + len(chunks)) / (1 + df)) + 1             # smoothed inverse document frequency
E = normalize(TF * idf)                                    # one TF-IDF "embedding" per chunk (the vector index)
print(f"\n{len(chunks)} chunks, vocabulary of {len(vocab)} words")

question = "How does the Adam optimizer use the gradient?"
q = normalize(tf(question) * idf)
scores = E @ q                                             # cosine similarity, since all rows are unit length
top = np.argsort(-scores)[:2]                              # top-k retrieval with k = 2
for rank, i in enumerate(top, 1):
    print(f"top-{rank}: chunk {i} (cosine {scores[i]:.3f}): {chunks[i]}")

prompt = ("Answer the question using only the context below. If the answer is not there, say so.\n\n"
          + "\n".join(f"[{k + 1}] {chunks[i]}" for k, i in enumerate(top))
          + f"\n\nQuestion: {question}\nAnswer:")
print("\n--- prompt sent to the language model ---\n" + prompt)

# ================= 3. Neural ODEs: a residual network is Euler's method =================
rng = np.random.default_rng(0)
W = rng.standard_normal((2, 2))


def f(h):
    return np.tanh(h @ W.T)                        # the "layer": dh/dt = f(h)


def rk4(h, steps, T=1.0):
    dt = T / steps
    for _ in range(steps):
        k1 = f(h)
        k2 = f(h + dt / 2 * k1)
        k3 = f(h + dt / 2 * k2)
        k4 = f(h + dt * k3)
        h = h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return h


h0 = np.array([1.0, -0.5])
exact = rk4(h0, 10_000)                            # reference solution h(1)
print(f"\nh(1) reference {np.round(exact, 6)}")
for L in (1, 2, 4, 8, 16, 32):
    h = h0.copy()
    for _ in range(L):                             # an L-layer residual net with shared weights:
        h = h + (1.0 / L) * f(h)                   #   h_{l+1} = h_l + dt * f(h_l)  == one Euler step
    print(f"residual net with {L:2d} layers: error {np.linalg.norm(h - exact):.6f}   "
          f"(RK4 with {L:2d} steps: {np.linalg.norm(rk4(h0, L) - exact):.2e})")`;
const MM_NUMPY_OUT = String.raw`cosine matrix (rows = images, columns = texts):
 [[0.9871 0.4092 0.3475]
 [0.1504 0.919  0.7362]
 [0.3059 0.2682 0.9658]]
image->text 0.0522   text->image 0.0356   CLIP loss 0.0439
zero-shot probabilities: {'cat': 0.0016, 'car': 0.7349, 'tree': 0.2634}

8 chunks, vocabulary of 91 words
top-1: chunk 0 (cosine 0.450): Gradient descent updates parameters in the direction opposite to the gradient. The learning rate controls the size of each step
top-2: chunk 1 (cosine 0.396): the size of each step and must be tuned. Adam keeps running averages of the gradient and of its square

--- prompt sent to the language model ---
Answer the question using only the context below. If the answer is not there, say so.

[1] Gradient descent updates parameters in the direction opposite to the gradient. The learning rate controls the size of each step
[2] the size of each step and must be tuned. Adam keeps running averages of the gradient and of its square

Question: How does the Adam optimizer use the gradient?
Answer:

h(1) reference [1.163608 0.08613 ]
residual net with  1 layers: error 0.063223   (RK4 with  1 steps: 2.53e-05)
residual net with  2 layers: error 0.031053   (RK4 with  2 steps: 1.53e-06)
residual net with  4 layers: error 0.015326   (RK4 with  4 steps: 9.35e-08)
residual net with  8 layers: error 0.007606   (RK4 with  8 steps: 5.77e-09)
residual net with 16 layers: error 0.003788   (RK4 with 16 steps: 3.58e-10)
residual net with 32 layers: error 0.001890   (RK4 with 32 steps: 2.23e-11)`;
const CLIP_TORCH = String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=0)
IMG_TR, IMG_TE = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)

# ---- a toy "web" of image-caption pairs: every caption is written from a template and the digit's name ----
NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TRAIN_TEMPLATES = ["a handwritten {}", "the digit {}", "a scan of the number {}", "{} written by hand",
                   "a small image of a {}", "the number {} in black ink"]
words = sorted({w for t in TRAIN_TEMPLATES for w in t.format("").split()} | set(NAMES) | {"photo", "of"})
VOCAB = {w: i + 1 for i, w in enumerate(words)}                 # index 0 = unknown word


def encode_text(s):
    return torch.tensor([VOCAB.get(w, 0) for w in s.split()])


rng = np.random.default_rng(0)
captions = [TRAIN_TEMPLATES[rng.integers(len(TRAIN_TEMPLATES))].format(NAMES[c]) for c in y_tr]
print("example pairs:", [(int(y_tr[i]), captions[i]) for i in range(3)])


class TextEncoder(nn.Module):
    """Bag-of-words: mean of word embeddings, then an MLP."""

    def __init__(self, dim=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(len(VOCAB) + 1, 64, mode="mean")
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(64, dim))

    def forward(self, texts):
        ids = [encode_text(t) for t in texts]
        offsets = torch.tensor([0] + [len(i) for i in ids[:-1]]).cumsum(0)
        return self.mlp(self.emb(torch.cat(ids), offsets))


torch.manual_seed(0)
image_enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
text_enc = TextEncoder()
log_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))   # learnable 1/temperature, as in CLIP
opt = torch.optim.Adam([*image_enc.parameters(), *text_enc.parameters(), log_scale], lr=1e-3)

for epoch in range(1, 31):
    for idx in torch.randperm(len(IMG_TR)).split(128):
        I = F.normalize(image_enc(IMG_TR[idx]), dim=1)
        T = F.normalize(text_enc([captions[i] for i in idx]), dim=1)
        logits = log_scale.exp().clamp(max=100) * I @ T.T          # (B, B) scaled cosine similarities
        labels = torch.arange(len(idx))                             # pair i matches caption i
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2   # symmetric InfoNCE
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch in (1, 10, 30):
        print(f"epoch {epoch:2d}  CLIP loss {loss.item():.4f}  temperature {1 / log_scale.exp().item():.4f}")


@torch.no_grad()
def zero_shot_accuracy(templates):
    """Build a classifier from text alone: class weight = mean normalised embedding of its prompts."""
    W = torch.stack([F.normalize(text_enc([t.format(n) for t in templates]), dim=1).mean(0) for n in NAMES])
    pred = (F.normalize(image_enc(IMG_TE), dim=1) @ F.normalize(W, dim=1).T).argmax(1)
    return (pred.numpy() == y_te).mean()


print("\nzero-shot test accuracy (no classifier was ever trained on labels):")
print(f"  prompt 'the digit {{}}'                  {zero_shot_accuracy(['the digit {}']):.3f}")
print(f"  prompt 'a photo of {{}}' (unseen words)  {zero_shot_accuracy(['a photo of {}']):.3f}")
print(f"  prompt '{{}}' (class name only)          {zero_shot_accuracy(['{}']):.3f}")
print(f"  ensemble of all 6 training templates   {zero_shot_accuracy(TRAIN_TEMPLATES):.3f}")

with torch.no_grad():                                               # text -> image retrieval
    I = F.normalize(image_enc(IMG_TE), dim=1)
    for query in ["a handwritten seven", "the number three in black ink"]:
        q = F.normalize(text_enc([query]), dim=1)
        top = (I @ q.T).squeeze(1).topk(5).indices
        print(f"query '{query}': labels of the top-5 test images {y_te[top.numpy()].tolist()}")`;
const CLIP_TORCH_OUT = String.raw`example pairs: [(6, 'the number six in black ink'), (4, 'four written by hand'), (4, 'four written by hand')]
epoch  1  CLIP loss 4.4207  temperature 0.0705
epoch 10  CLIP loss 2.3899  temperature 0.0692
epoch 30  CLIP loss 1.9101  temperature 0.0613

zero-shot test accuracy (no classifier was ever trained on labels):
  prompt 'the digit {}'                  0.941
  prompt 'a photo of {}' (unseen words)  0.889
  prompt '{}' (class name only)          0.954
  ensemble of all 6 training templates   0.967
query 'a handwritten seven': labels of the top-5 test images [7, 7, 8, 7, 7]
query 'the number three in black ink': labels of the top-5 test images [3, 3, 3, 3, 3]`;
const RAG_SKLEARN = String.raw`import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# A small knowledge base: one chunk per sentence (sentence-level chunking).
chunks = [
    "Gradient descent updates parameters in the direction opposite to the gradient.",
    "The learning rate controls the size of each step and must be tuned.",
    "Adam keeps running averages of the gradient and of its square for every parameter.",
    "Dropout randomly zeroes activations during training to reduce overfitting.",
    "Batch normalization standardizes each feature over the mini-batch.",
    "A convolution slides a small kernel over the image and shares its weights across positions.",
    "Pooling layers shrink feature maps and add a little translation invariance.",
    "Attention computes a weighted average of value vectors using query-key similarities.",
    "The transformer stacks attention and feed-forward layers with residual connections.",
    "CLIP trains an image encoder and a text encoder so that matching pairs have high cosine similarity.",
]
# Evaluation questions with the index of the chunk that answers them.
questions = [
    ("Which optimizer tracks running averages of squared gradients?", 2),
    ("How do I reduce overfitting by zeroing activations?", 3),
    ("What does a pooling layer do to feature maps?", 6),
    ("How are image and text encoders trained in CLIP?", 9),
    ("Why must the step size be tuned?", 1),
    ("What stops a model from memorising its examples?", 3),   # paraphrase: no shared content word
]

vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
index = vectorizer.fit_transform(chunks)                    # sparse (10, vocabulary) matrix, rows L2-normalised
knn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(index)
dist, nbr = knn.kneighbors(vectorizer.transform([q for q, _ in questions]))

hits1 = hits3 = 0
for (q, gold), d, n in zip(questions, dist, nbr):
    hits1 += n[0] == gold
    hits3 += gold in n
    print(f"{q:<62} top-3 {n.tolist()}  cos {np.round(1 - d, 3).tolist()}  gold {gold}")
print(f"\nrecall@1 = {hits1 / len(questions):.3f}   recall@3 = {hits3 / len(questions):.3f}")`;
const RAG_SKLEARN_OUT = String.raw`Which optimizer tracks running averages of squared gradients?  top-3 [2, 0, 1]  cos [0.545, 0.0, 0.0]  gold 2
How do I reduce overfitting by zeroing activations?            top-3 [3, 0, 1]  cos [0.655, 0.0, 0.0]  gold 3
What does a pooling layer do to feature maps?                  top-3 [6, 4, 0]  cos [0.568, 0.171, 0.0]  gold 6
How are image and text encoders trained in CLIP?               top-3 [9, 5, 2]  cos [0.485, 0.158, 0.0]  gold 9
Why must the step size be tuned?                               top-3 [1, 0, 2]  cos [0.707, 0.0, 0.0]  gold 1
What stops a model from memorising its examples?               top-3 [0, 1, 2]  cos [0.0, 0.0, 0.0]  gold 3

recall@1 = 0.833   recall@3 = 0.833`;

const CLIP_KERAS = String.raw`import numpy as np
import tensorflow as tf

EMBED = 32
image_encoder = tf.keras.Sequential([tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dense(EMBED)])
text_encoder = tf.keras.Sequential([tf.keras.layers.Embedding(1000, 64), tf.keras.layers.GlobalAveragePooling1D(),
                                    tf.keras.layers.ReLU(), tf.keras.layers.Dense(EMBED)])
log_scale = tf.Variable(np.log(1 / 0.07), dtype=tf.float32)          # learnable inverse temperature
opt = tf.keras.optimizers.Adam(1e-3)


@tf.function
def train_step(images, token_ids):
    with tf.GradientTape() as tape:
        I = tf.math.l2_normalize(image_encoder(images), axis=1)
        T = tf.math.l2_normalize(text_encoder(token_ids), axis=1)
        logits = tf.exp(log_scale) * tf.matmul(I, T, transpose_b=True)
        labels = tf.range(tf.shape(images)[0])
        ce = tf.keras.losses.sparse_categorical_crossentropy
        loss = 0.5 * (tf.reduce_mean(ce(labels, logits, from_logits=True))
                      + tf.reduce_mean(ce(labels, tf.transpose(logits), from_logits=True)))
    variables = image_encoder.trainable_variables + text_encoder.trainable_variables + [log_scale]
    opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
    return loss

# for images, token_ids in dataset: train_step(images, token_ids)   (token_ids: padded integer captions)`;

const lesson: Lesson = {
  id: "m15/multimodal",
  title: "Multimodal Models, CLIP and RAG Foundations",
  summary:
    "Models that connect modalities: joint embedding spaces, CLIP's symmetric contrastive loss and learned temperature, zero-shot classification by text prompts and prompt ensembling, how vision-language models feed image tokens to a language model, the retrieval half of retrieval-augmented generation (chunking, embedding, cosine top-k search, prompt assembly), and Neural ODEs as continuous-depth residual networks — all computed and trained.",
  minutes: 85,
  level: "advanced",
  prerequisites: ["m15/self-supervised", "m05/advanced-losses", "m13/word-embeddings", "m11/transformer-architecture"],
  next: ["m20/llm-pretraining", "m15/graph-neural-networks", "m18/serving"],
  terms: ["multimodal-learning", "joint-embedding-space", "clip", "zero-shot-classification", "prompt-ensembling", "vision-language-model", "retrieval-augmented-generation", "chunking", "vector-search", "top-k-retrieval", "neural-ode", "cosine-similarity"],
  sections: {
    intro: [
      {
        t: "p",
        text: "Humans learn from sight, sound and language together, and many useful tasks cross modalities: search photos by typing a sentence, caption an image, answer questions about a chart, generate an image from a description. **Multimodal learning** builds models that represent several modalities and relate them. The central tool is a **joint embedding space**: separate encoders map an image and a sentence to vectors in the same space, trained so that an image and its description land close together.",
      },
      {
        t: "p",
        text: String.raw`**CLIP** (Radford et al., 2021, "Learning Transferable Visual Models From Natural Language Supervision") trained an image encoder and a text encoder on about 400 million image–text pairs collected from the internet with a symmetric contrastive loss — the same InfoNCE loss as [self-supervised learning](/learn/m15/self-supervised), but the two "views" are an image and its caption. The payoff is **zero-shot classification**: to recognise classes the model was never explicitly trained on, embed a prompt such as "a photo of a dog" for each class and pick the class whose text embedding is closest to the image's. Joint embeddings also power text-to-image retrieval and condition text-to-image diffusion models. **Vision-language models** go further and feed image features into a language model so it can describe and reason about images.`,
      },
      {
        t: "p",
        text: String.raw`The same embedding-and-similarity machinery underlies **retrieval-augmented generation (RAG)** (Lewis et al., 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"): split a document collection into chunks, embed them, retrieve the chunks most similar to a question, and give them to a language model as context. RAG lets a model answer from up-to-date or private documents and cite its sources, without retraining. Finally, this lesson covers **Neural ODEs** (Chen, Rubanova, Bettencourt and Duvenaud, 2018, "Neural Ordinary Differential Equations"), which treat a residual network as the discretisation of a continuous-time dynamical system.`,
      },
      {
        t: "callout",
        kind: "key",
        text: String.raw`CLIP: logits $L_{ij} = \cos(f(x_i), g(t_j))/\tau$; loss $= \tfrac12\big[\mathrm{CE}_{\text{rows}}(L) + \mathrm{CE}_{\text{cols}}(L)\big]$ with the diagonal as targets. Zero-shot: $\hat y = \arg\max_c \cos(f(x), g(\text{prompt}_c))$. RAG retrieval: top-$k$ chunks by $\cos(e(q), e(d))$, then prompt = instructions + chunks + question.`,
      },
    ],
    prerequisites: [
      {
        t: "list",
        items: [
          "[Self-Supervised Learning](/learn/m15/self-supervised) — the contrastive InfoNCE / NT-Xent loss, positives and negatives, and the temperature. CLIP is contrastive learning across modalities.",
          "[Advanced Losses](/learn/m05/advanced-losses) — InfoNCE as a softmax cross-entropy over similarities.",
          "[Word Embeddings](/learn/m13/word-embeddings) — vectors whose cosine similarity reflects meaning; the text side of a joint space and the basis of embedding-based retrieval.",
          "[Transformer Architecture](/learn/m11/transformer-architecture) and [Vision Transformers](/learn/m11/transformer-families) — CLIP's encoders and vision-language models are Transformers; images become sequences of patch tokens.",
          "[Norms and Spaces](/learn/m02/norms-spaces) — cosine similarity, L2 normalisation, and why the dot product of unit vectors is the cosine.",
          "[Residual Connections](/learn/m07/residual-connections) — needed for the Neural ODE section: $h_{l+1} = h_l + f(h_l)$.",
        ],
      },
    ],
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A shared map for pictures and words",
        text: "Imagine a huge map where every picture and every sentence has a location. The training rule is simple: a photo and its caption must be neighbours, and must be farther from everyone else's captions. After millions of examples, \"a dog on a beach\" sits next to photos of dogs on beaches, and \"a red car\" next to red cars. To classify a new photo you no longer need a trained classifier: you write down the class names as sentences, find where they sit on the map, and see which one the photo is closest to.",
      },
      {
        t: "p",
        text: "RAG is an open-book exam. Rather than hoping the language model memorised every fact, you give it a library card: the question is placed on the same kind of map as every paragraph of your documents, the few nearest paragraphs are photocopied, and the model answers with those pages in front of it. If the right page is not retrieved, the model cannot use it — retrieval quality bounds answer quality.",
      },
      {
        t: "p",
        text: "A Neural ODE notices that a residual block, $h \\leftarrow h + f(h)$, looks like one step of a numerical solver for a differential equation $dh/dt = f(h)$. Stacking many thin residual blocks is like taking many small steps; in the limit, the network's depth becomes continuous time, and an ODE solver decides how many steps to take.",
      },
    ],
    mechanism: [
      { t: "h", text: "CLIP training" },
      {
        t: "steps",
        items: [
          { title: "Batch of pairs", text: String.raw`$N$ (image, caption) pairs $(x_i, t_i)$ — CLIP used batches of 32,768.` },
          { title: "Encode each modality", text: String.raw`Image encoder $f$ (a ResNet or Vision Transformer) and text encoder $g$ (a Transformer) each produce a vector; a linear projection maps both into the same $d$-dimensional space.` },
          { title: "Normalise", text: String.raw`$u_i = f(x_i)/\lVert f(x_i)\rVert$, $v_j = g(t_j)/\lVert g(t_j)\rVert$.` },
          { title: "Similarity matrix", text: String.raw`$L = s\,U V^\top$ ($N\times N$), with a **learned** scale $s = e^{\log s} = 1/\tau$, initialised to $1/0.07$ and clipped at 100 in the paper.` },
          { title: "Symmetric cross-entropy", text: String.raw`Row $i$ is a classification "which caption belongs to image $i$?" (target $i$); column $j$ is "which image belongs to caption $j$?". Average the two cross-entropies.` },
          { title: "Use", text: "Zero-shot classification via prompts, image–text retrieval in both directions, and frozen image features for linear probes or for conditioning generators." },
        ],
      },
      { t: "h", text: "Zero-shot classification with prompts" },
      {
        t: "steps",
        items: [
          { title: "Write prompts", text: "One or more templates per class, e.g. \"a photo of a {class}.\" Radford et al. found that templates matching the style of training captions beat the bare class name, and that context helps (\"a photo of a {class}, a type of pet.\")." },
          { title: "Build the classifier from text", text: String.raw`Embed each prompt; the class weight vector $w_c$ is the normalised mean of its prompt embeddings (**prompt ensembling**).` },
          { title: "Classify", text: String.raw`Embed the image, compute $\cos(u, w_c)$ for all classes and take the arg-max (or a softmax with the learned scale for probabilities).` },
        ],
      },
      { t: "h", text: "Vision-language models" },
      {
        t: "p",
        text: "Generative vision-language models (VLMs) answer questions about images in free text. A common recipe: a pretrained image encoder (often a CLIP-style ViT) turns the image into a sequence of patch features; a small **connector** (a linear layer, an MLP, or cross-attention with learned queries) maps them into the language model's token-embedding space; the language model then reads these 'visual tokens' followed by the text prompt and generates the answer autoregressively. Alternatively, cross-attention layers inserted into the language model attend to the image features. Training usually first aligns the connector on image–caption pairs, then fine-tunes on instruction-style image–question–answer data.",
      },
      { t: "h", text: "RAG pipeline" },
      {
        t: "steps",
        items: [
          { title: "Ingest and chunk", text: "Split documents into chunks of a few hundred tokens, often with overlap so a fact cut at a boundary still appears whole in one chunk; better still, split at paragraph or section boundaries. Keep metadata (source, page, date) with each chunk." },
          { title: "Embed", text: String.raw`Map every chunk to a vector $e(d)$ with an embedding model — dense (a Transformer sentence encoder) or sparse (TF-IDF / BM25 term weights). Normalise to unit length.` },
          { title: "Index", text: "Store the vectors in a vector index. Exact search compares the query with every vector; approximate nearest-neighbour indexes (graph-based such as HNSW, or quantisation-based) trade a little recall for orders-of-magnitude speed on millions of chunks." },
          { title: "Retrieve", text: String.raw`Embed the question with the same model, compute cosine similarities, take the top-$k$. Optional: hybrid sparse + dense search, metadata filters, and a **re-ranker** (a cross-encoder that reads question and chunk together) on the top candidates.` },
          { title: "Assemble the prompt", text: "Instructions (answer only from the context, say when it is missing, cite sources) + the retrieved chunks with identifiers + the question. Keep the total within the model's context window." },
          { title: "Generate and cite", text: "The language model answers conditioned on the context; the chunk identifiers let the answer cite its sources. Evaluate retrieval (recall@k) and answers (faithfulness to the context, correctness) separately." },
        ],
      },
      { t: "h", text: "Neural ODEs" },
      {
        t: "steps",
        items: [
          { title: "Define the dynamics", text: String.raw`A neural network $f_\theta(h, t)$ gives the time derivative of the hidden state: $dh/dt = f_\theta(h(t), t)$.` },
          { title: "Forward pass = solve the ODE", text: String.raw`Integrate from $h(0) = $ input to $h(T)$ with an ODE solver (Euler, Runge–Kutta, adaptive-step methods). The output is $h(T)$.` },
          { title: "Backward pass", text: "Either backpropagate through the solver's operations, or use the **adjoint method**: solve a second ODE backwards in time for the gradient, with memory cost independent of the number of steps." },
          { title: "Uses", text: "Continuous-depth models, time series with irregular sampling, and continuous normalising flows." },
        ],
      },
    ],
    math: [
      { t: "h", text: "1. CLIP's symmetric contrastive loss" },
      { t: "math", tex: String.raw`L_{ij} = \frac{u_i^\top v_j}{\tau},\qquad \mathcal L = \frac12\left[\frac1N\sum_{i=1}^{N} -\log\frac{e^{L_{ii}}}{\sum_{j=1}^{N} e^{L_{ij}}} \;+\; \frac1N\sum_{j=1}^{N} -\log\frac{e^{L_{jj}}}{\sum_{i=1}^{N} e^{L_{ij}}}\right]` },
      {
        t: "p",
        text: String.raw`$u_i$: unit image embedding of pair $i$; $v_j$: unit text embedding of pair $j$; $\tau$: temperature (CLIP learns $\log(1/\tau)$, initialised at $\log(1/0.07) = 2.659$); $N$: batch size. The first term is image→text (softmax over each **row**), the second text→image (softmax over each **column**). Unlike SimCLR's NT-Xent there are no within-modality negatives: an image is contrasted only with captions. A model that cannot tell pairs apart scores $\log N$ in each direction.`,
      },
      { t: "h", text: "2. Worked example" },
      {
        t: "p",
        text: String.raw`Three pairs, 3-D embeddings (normalised in the program), $\tau = 0.1$. Cosine matrix (rows images, columns captions):`,
      },
      { t: "math", tex: String.raw`U V^\top = \begin{pmatrix} 0.9871 & 0.4092 & 0.3475\\ 0.1504 & 0.9190 & 0.7362\\ 0.3059 & 0.2682 & 0.9658\end{pmatrix}` },
      {
        t: "steps",
        items: [
          { title: "Row 1 (image 1 → texts)", tex: String.raw`L_{1\cdot} = (9.871,\ 4.092,\ 3.475),\quad \mathrm{softmax} = (0.9953,\ 0.0031,\ 0.0017),\quad -\ln 0.9953 = 0.0047` },
          { title: "Row 2", tex: String.raw`L_{2\cdot} = (1.504,\ 9.190,\ 7.362),\quad \mathrm{softmax} = (0.0004,\ 0.8612,\ 0.1384),\quad -\ln 0.8612 = 0.1495` },
          { title: "Row 3", tex: String.raw`-\ln 0.9977 = 0.0023;\qquad \text{image}\to\text{text} = \tfrac13(0.0047 + 0.1495 + 0.0023) = 0.0522` },
          { title: "Columns (texts → images)", tex: String.raw`-\ln 0.9987 = 0.0013,\ \ -\ln 0.9925 = 0.0076,\ \ -\ln 0.9068 = 0.0978;\quad \text{mean} = 0.0356` },
          { title: "CLIP loss", tex: String.raw`\mathcal L = \tfrac12(0.0522 + 0.0356) = 0.0439` },
        ],
      },
      {
        t: "p",
        text: String.raw`Image 2 and caption 3 are the hard negative pair ($\cos = 0.7362$): they dominate both directions' losses, and the gradient will push them apart. The two directions differ because rows and columns normalise over different sets.`,
      },
      { t: "h", text: "3. Zero-shot classification as a linear classifier" },
      { t: "math", tex: String.raw`p(y = c\mid x) = \frac{\exp\big(s\, u^\top w_c\big)}{\sum_{c'}\exp\big(s\, u^\top w_{c'}\big)},\qquad w_c = \frac{\sum_{m} g(\text{prompt}_{c,m})/\lVert g(\text{prompt}_{c,m})\rVert}{\big\lVert \sum_{m} g(\text{prompt}_{c,m})/\lVert g(\text{prompt}_{c,m})\rVert \big\rVert}` },
      {
        t: "p",
        text: String.raw`$u$: unit image embedding; $w_c$: normalised mean of the unit embeddings of the $M$ prompts for class $c$; $s = 1/\tau$. This is a multinomial logistic regression whose weight matrix was **generated from text** instead of learned from labelled images. In the program, an image embedding $(0.2, 0.7, 0.35)$ (normalised) against the three caption embeddings with $\tau = 0.1$ gives probabilities cat 0.0016, car 0.7349, tree 0.2634.`,
      },
      { t: "h", text: "4. Cosine similarity and top-k retrieval" },
      { t: "math", tex: String.raw`\cos(q, d) = \frac{q^\top d}{\lVert q\rVert\,\lVert d\rVert},\qquad \mathrm{TopK}(q) = \operatorname*{arg\,top\text{-}k}_{d\in\mathcal D}\ \cos\big(e(q), e(d)\big)` },
      {
        t: "p",
        text: String.raw`**Example** with 3-word count vectors: query $q = (1, 1, 0)$, chunk A $= (2, 0, 1)$, chunk B $= (1, 1, 1)$. $q^\top A = 2$, $\lVert q\rVert = 1.4142$, $\lVert A\rVert = 2.2361$, $\cos = 0.6325$; $q^\top B = 2$, $\lVert B\rVert = 1.7321$, $\cos = 0.8165$. Top-1 is B: the dot products are equal, but A's length comes from a word the query does not contain. With unit-normalised vectors the cosine is just the dot product, so the whole search is one matrix–vector product $E\,e(q)$ followed by a top-$k$ selection; for $n$ chunks of dimension $d$ that is $O(nd)$ per query, which is what approximate indexes reduce.`,
      },
      { t: "h", text: "5. TF-IDF: a sparse embedding" },
      { t: "math", tex: String.raw`\text{tfidf}(w, d) = \text{tf}(w, d)\cdot\Big(\ln\frac{1 + n}{1 + \text{df}(w)} + 1\Big)` },
      {
        t: "p",
        text: String.raw`$\text{tf}(w, d)$: count of word $w$ in chunk $d$; $n$: number of chunks; $\text{df}(w)$: number of chunks containing $w$ (the smoothed form used by scikit-learn and by our from-scratch code). Rare words get high weight, ubiquitous words low weight. TF-IDF only matches **shared words**: a paraphrase with no word in common scores 0 — the reason dense, learned embeddings (and hybrid search) are used in practice.`,
      },
      { t: "h", text: "6. The RAG model" },
      { t: "math", tex: String.raw`p(y\mid q) \approx \sum_{d\,\in\,\mathrm{TopK}(q)} p_\eta(d\mid q)\; p_\theta(y\mid q, d)` },
      {
        t: "p",
        text: String.raw`$q$: question; $y$: answer; $p_\eta(d\mid q) \propto \exp(e(d)^\top e(q))$: retriever; $p_\theta(y\mid q, d)$: generator conditioned on a retrieved passage. Lewis et al. marginalised over the top-$k$ passages and trained retriever (query encoder) and generator jointly. Most practical systems simply concatenate the top-$k$ chunks into one prompt and use a frozen language model.`,
      },
      { t: "h", text: "7. Neural ODEs: residual networks as Euler steps" },
      { t: "math", tex: String.raw`\underbrace{h_{l+1} = h_l + \Delta t\, f_\theta(h_l)}_{\text{residual block}} \quad\xrightarrow{\ \Delta t\to 0\ }\quad \frac{dh(t)}{dt} = f_\theta\big(h(t), t\big),\qquad h(T) = h(0) + \int_0^T f_\theta\big(h(t), t\big)\,dt` },
      {
        t: "p",
        text: String.raw`An $L$-layer residual network with shared weights and step $\Delta t = T/L$ is exactly Euler's method. Euler's global error is $O(\Delta t)$: in the program, doubling the number of 'layers' halves the error (0.0632, 0.0311, 0.0153, …, 0.0019 for $L = 1 \dots 32$), whereas fourth-order Runge–Kutta's error falls by about $2^4 = 16$ per doubling ($2.5\times10^{-5}$ with one step). The **adjoint method** computes gradients by integrating $a(t) = \partial\mathcal L/\partial h(t)$ backwards: $da/dt = -a^\top\,\partial f_\theta/\partial h$, with $\partial\mathcal L/\partial\theta = -\int_T^0 a^\top\,\partial f_\theta/\partial\theta\,dt$ — memory independent of the number of solver steps, at the price of re-solving the ODE backwards.`,
      },
    ],
    visual: [
      {
        t: "p",
        text: "Every idea in this lesson relies on one geometric fact: in a well-trained embedding space, **cosine similarity measures relatedness**. The word-embedding explorer lets you test this on real skip-gram vectors — the same nearest-neighbour search that zero-shot classification and RAG retrieval perform.",
      },
      { t: "sim", id: "word-embeddings", caption: "Nearest neighbours, analogies and a 2-D projection of trained word vectors." },
      {
        t: "list",
        items: [
          "Search a word and read its nearest neighbours with their cosine similarities. This is top-$k$ retrieval: replace words by chunks of text and you have the retrieval step of RAG.",
          "Increase the number of neighbours $k$: the extra neighbours become less related — the precision/recall trade-off of choosing $k$ in RAG.",
          "Try analogies: directions in the space carry meaning. In CLIP, images and sentences share one such space, which is why a text vector can act as a classifier weight for images.",
          "Switch between t-SNE and PCA projections: clusters you see in 2-D are only an approximation of the high-dimensional neighbourhoods that search uses.",
        ],
      },
      {
        t: "p",
        text: "Vision-language models connect image tokens and text tokens through attention. The attention simulator shows how one token's query scores every key and averages the values — in a VLM, text tokens attend to visual tokens in exactly this way.",
      },
      { t: "sim", id: "attention", caption: "Scaled dot-product attention: the mechanism that lets text tokens read image tokens in a vision-language model." },
    ],
    scratch: [
      {
        t: "p",
        text: "Three parts in NumPy: the CLIP loss and zero-shot prediction on the worked example; the retrieval half of RAG from scratch (overlapping word chunks, TF-IDF vectors, cosine top-$k$, prompt assembly); and a residual network compared with Euler and Runge–Kutta integration of the same ODE.",
      },
      { t: "code", lang: "python", title: "clip_rag_ode_from_scratch.py", code: MM_NUMPY, output: MM_NUMPY_OUT },
      {
        t: "list",
        items: [
          "The CLIP numbers reproduce the worked example (0.0522, 0.0356, 0.0439). Zero-shot prediction is one matrix–vector product with the prompt embeddings.",
          "Chunking with 20-word windows and 5 words of overlap: chunks 0 and 1 share \"the size of each step\", so a sentence cut at the boundary is still complete somewhere.",
          "**A retrieval failure worth noticing**: the Adam chunk is ranked second, not first. The question's word \"optimizer\" occurs in no chunk, and \"gradient\" occurs twice in the gradient-descent chunk, so lexical matching favours it. With $k = 2$ the answer still reaches the prompt — one reason to retrieve more than one chunk — but a semantic embedding would rank it first.",
          "The assembled prompt tells the model to use only the context and to admit when the answer is missing — the usual defence against the generator ignoring or inventing beyond the retrieved text.",
          "Residual net = Euler: the error halves each time the number of layers doubles (first-order convergence), while RK4 is accurate to $2.5\\times10^{-5}$ with a single step. Adaptive ODE solvers exploit this to choose the 'depth' automatically.",
        ],
      },
    ],
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              { t: "p", text: "**A miniature CLIP** on scikit-learn's digits. Each training image is paired with a caption written from one of six templates (\"a handwritten seven\", \"the number seven in black ink\", …). An MLP image encoder and a bag-of-words text encoder are trained with the symmetric contrastive loss and a learnable temperature. No classifier is trained; classification is done afterwards with text prompts." },
              { t: "code", lang: "python", title: "clip_digits.py", code: CLIP_TORCH, output: CLIP_TORCH_OUT },
              {
                t: "list",
                items: [
                  "Zero-shot test accuracy 0.94–0.97 from prompts alone. Prompt ensembling over the six templates (0.967) beats any single prompt, as Radford et al. reported for CLIP.",
                  "The prompt \"a photo of {}\" contains words never seen in training (their embeddings stayed at random initialisation) and drops to 0.889 — a small-scale version of CLIP's **prompt sensitivity**: prompts should resemble the training captions.",
                  "The loss stalls near 1.9 instead of approaching 0 because a batch of 128 contains about 13 images of each digit with interchangeable captions: these are **false negatives**, the same issue as in SimCLR. Web-scale captions are far more specific, so CLIP suffers less from it.",
                  "The learned temperature drifts from 0.07 to 0.061 — the model sharpens its softmax as the embeddings separate.",
                  "Text→image retrieval works in the same space: the top-5 images for \"a handwritten seven\" are four 7s and one 8.",
                  "Honesty about \"zero-shot\": these captions contain the class names, so the model *was* supervised by language. CLIP's zero-shot transfer works the same way — class names appear in its web captions — which is why it is called natural-language supervision.",
                ],
              },
            ],
          },
          {
            label: "scikit-learn (retrieval)",
            blocks: [
              { t: "p", text: "The retrieval step of RAG with library components: `TfidfVectorizer` as a sparse embedding and `NearestNeighbors` with cosine distance as an exact vector index, evaluated with recall@k on questions whose answer chunk is known." },
              { t: "code", lang: "python", title: "rag_retrieval_sklearn.py", code: RAG_SKLEARN, output: RAG_SKLEARN_OUT },
              {
                t: "list",
                items: [
                  "Five of six questions retrieve the right chunk at rank 1. Evaluate retrieval on its own (recall@k on labelled question–chunk pairs) before blaming the generator for wrong answers.",
                  "The paraphrased question shares no content word with its answer, so every similarity is 0 and the 'top-3' is arbitrary. Dense embeddings from a trained sentence encoder, query expansion or hybrid search fix this class of failure; lexical scores such as BM25 remain valuable for exact names, codes and rare terms.",
                  "`NearestNeighbors(metric=\"cosine\")` returns cosine *distances* $1 - \\cos$. At scale, replace it with an approximate nearest-neighbour index.",
                ],
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              { t: "p", text: "The CLIP training step with a learnable inverse temperature, written with `tf.GradientTape`." },
              { t: "code", lang: "python", title: "clip_keras.py (not executed in CI)", code: CLIP_KERAS, output: "Defines the two encoders and a compiled training step that returns the symmetric contrastive loss for each batch." },
            ],
          },
        ],
      },
    ],
    variations: [
      {
        t: "table",
        head: ["Approach", "How modalities interact", "Advantages", "Disadvantages", "Use"],
        rows: [
          ["Dual encoder (CLIP; ALIGN, Jia et al. 2021)", "separate encoders, compared only by cosine similarity", "fast retrieval (pre-compute embeddings); zero-shot classification", "no fine-grained interaction; weak at counting, spatial relations", "search, zero-shot classification, filtering data"],
          ["Sigmoid-loss dual encoder (SigLIP)", "per-pair binary loss instead of a batch softmax", "no global normalisation across the batch; works well with smaller batches", "still a dual encoder", "same as CLIP"],
          ["Fusion / cross-attention encoders", "text tokens attend to image tokens inside the model", "fine-grained reasoning, strong on VQA", "every image–text pair needs a joint forward pass: slow for retrieval", "visual question answering, re-ranking"],
          ["Generative VLMs (connector + LLM)", "visual tokens fed into a language model", "free-form answers, instruction following", "hallucination; large compute", "assistants that see images, document understanding"],
          ["Text-to-image diffusion", "text embedding conditions the denoiser via cross-attention", "high-quality image generation", "expensive sampling", "image generation — see [Diffusion](/learn/m14/diffusion)"],
          ["Sparse retrieval (TF-IDF, BM25)", "term overlap", "exact matches, no training, interpretable", "misses paraphrases", "keyword search, hybrid RAG"],
          ["Dense retrieval (bi-encoders)", "learned embeddings + cosine", "semantic matches", "needs a good embedding model; misses rare exact terms", "RAG, semantic search"],
          ["Re-ranking (cross-encoder)", "reads query and chunk together", "most accurate relevance scores", "too slow for the whole corpus; used on top candidates", "second stage of RAG"],
          ["Neural ODE", "continuous-depth residual dynamics", "adaptive computation, constant-memory adjoint, irregular time series", "slow and sometimes unstable to train; solver cost varies", "time-series, continuous normalising flows"],
        ],
      },
    ],
    applications: [
      {
        t: "list",
        items: [
          "**Image search by text** and **text search by image** in photo libraries and e-commerce catalogues, built on dual-encoder embeddings.",
          "**Zero-shot and open-vocabulary recognition** — classifying or detecting categories defined at run time by a text prompt.",
          "**Dataset filtering and curation** — CLIP-style image–text similarity scores are used to filter web-scale training data for later models.",
          "**Conditioning generators** — text embeddings guide text-to-image diffusion models; see [Diffusion Models](/learn/m14/diffusion).",
          "**Vision-language assistants** — answering questions about photos, screenshots, charts and documents.",
          "**RAG** — enterprise question answering over internal documents, customer-support assistants grounded in product manuals, and search engines that answer with citations.",
          "**Neural ODEs** — modelling irregularly sampled time series (e.g. clinical measurements) and continuous normalising flows for density estimation.",
        ],
      },
    ],
    tradeoffs: [
      {
        t: "list",
        items: [
          "**Dual encoders scale**: embeddings of millions of images can be pre-computed and searched in milliseconds; but compressing everything into one vector loses details such as counts, negation and spatial relations.",
          "**Zero-shot is flexible but not free**: accuracy depends on prompt wording, and on specialised domains (medical images, satellite imagery) a linear probe or fine-tuning with a few labels usually wins.",
          "**Web-scale data brings web-scale bias**: social biases and spurious correlations in captions are inherited.",
          "**RAG** gives up-to-date, private and citable answers without retraining, and reduces hallucination — but adds latency and infrastructure, fails silently when retrieval misses, is sensitive to chunking, and the model can still ignore or misread the context.",
          "**Long context vs retrieval**: putting entire documents in a long context window avoids retrieval errors but costs more per query and can dilute attention; retrieval keeps prompts short.",
          "**Neural ODEs** offer adaptive depth and constant-memory gradients, but solver steps make training slower and harder to predict than a fixed-depth ResNet; adjoint gradients can be inaccurate when the backward solve is unstable.",
        ],
      },
    ],
    mistakes: [
      {
        t: "list",
        items: [
          "**Forgetting to L2-normalise** image and text embeddings before the dot product. Norms then dominate similarities and the temperature loses its meaning.",
          "**Using a fixed, too-large temperature** in CLIP-style training. A learnable $\\log(1/\\tau)$ (clipped) is standard; a bad fixed value can stall training.",
          "**Computing the loss in only one direction.** The symmetric loss trains both encoders to be good queries and good keys.",
          "**Embedding queries and documents with different models** (or different preprocessing) in RAG. Both must live in the same space.",
          "**Chunks that are too large or too small.** Huge chunks dilute the embedding and waste context; tiny chunks lose the surrounding information needed to answer. Split at natural boundaries and add overlap.",
          "**Evaluating RAG only end-to-end.** Measure retrieval (recall@k) separately; if the right chunk is not retrieved, no generator can fix it.",
          "**Leaving out instructions to use only the context.** The model then mixes retrieved facts with memorised (possibly outdated) ones and cannot cite reliably.",
          "**Trusting cosine scores across different embedding models** as absolute relevance probabilities. Scores are only comparable within one model and one index; calibrate thresholds on labelled data.",
          "**Calling CLIP's zero-shot results 'no supervision'.** The supervision is natural language; class names appear in the captions.",
        ],
      },
    ],
    interview: [
      { t: "qa", level: "beginner", q: "What is a joint embedding space?", a: "A vector space shared by several modalities, produced by separate encoders trained so that related items — an image and its caption — have similar vectors (high cosine similarity) and unrelated items dissimilar ones. It enables cross-modal retrieval and zero-shot classification." },
      { t: "qa", level: "beginner", q: "How does CLIP classify an image into classes it was never explicitly trained on?", a: "Write a text prompt per class (e.g. 'a photo of a dog'), embed the prompts with the text encoder, embed the image with the image encoder, and choose the class whose prompt embedding has the highest cosine similarity with the image embedding." },
      { t: "qa", level: "beginner", q: "What is retrieval-augmented generation?", a: "A language model answers using documents retrieved for the question: the documents are chunked and embedded, the chunks most similar to the question are retrieved, and they are inserted into the prompt as context. It grounds answers in up-to-date or private data and allows citations without retraining." },
      { t: "qa", level: "intermediate", q: "Write CLIP's loss and explain why it is symmetric.", a: "With unit embeddings u_i, v_j and L_ij = u_i·v_j/τ, the loss is ½[mean_i CE(row i, target i) + mean_j CE(column j, target j)]. The row term makes each image identify its caption among all captions; the column term makes each caption identify its image. Both encoders are thus trained as queries and as keys, and retrieval works in both directions." },
      { t: "qa", level: "intermediate", q: "What is prompt ensembling and why does it help?", a: "For each class, embed several prompt templates, normalise and average them into one class vector. It averages out quirks of individual phrasings and makes the text-derived classifier more robust; CLIP reported consistent gains. In our digit model: best single prompt 0.954, ensemble 0.967." },
      { t: "qa", level: "intermediate", q: "How do you choose chunk size and k in RAG?", a: "Chunks should contain a self-contained piece of information: split at paragraph/section boundaries, typically a few hundred tokens, with some overlap. k trades recall (more chunks increase the chance the answer is included) against prompt length, cost and distraction. Tune both on a labelled set of question→relevant-chunk pairs using recall@k, and consider re-ranking a larger candidate set down to a small k." },
      { t: "qa", level: "intermediate", q: "Compare sparse (BM25/TF-IDF) and dense retrieval.", a: "Sparse retrieval scores term overlap: no training, excellent for exact names, codes and rare terms, but misses paraphrases (our paraphrased question scored 0 for every chunk). Dense retrieval uses learned embeddings and captures meaning, but can miss exact rare terms and depends on the embedding model's domain. Hybrid search combines both scores." },
      { t: "qa", level: "advanced", q: "Why are dual encoders fast for retrieval but weak at fine-grained understanding, and how do systems get both?", a: "Each item is encoded independently into one vector, so the corpus can be embedded once and searched with a dot product — but all interaction between query and item is reduced to that dot product, losing compositional details (counts, negation, relations). Systems retrieve candidates with a dual encoder and then re-rank the top few with a cross-encoder or a fusion model that attends across both inputs." },
      { t: "qa", level: "advanced", q: "Explain the connection between ResNets and Neural ODEs, and the adjoint method.", a: "A residual block h_{l+1} = h_l + Δt f(h_l) is an Euler step of dh/dt = f(h); in the limit of many thin layers the network becomes an ODE, solved by any solver (Euler has O(Δt) error, RK4 O(Δt⁴)). Gradients can be computed by the adjoint method: integrate a(t) = ∂L/∂h(t) backwards with da/dt = −aᵀ∂f/∂h and accumulate ∂L/∂θ = −∫ aᵀ ∂f/∂θ dt, giving memory cost independent of the number of steps." },
      { t: "qa", level: "advanced", q: "Your RAG system gives wrong answers. How do you debug it?", a: "Separate the stages. (1) Retrieval: on labelled questions, is the relevant chunk in the top-k? If not, inspect chunking, the embedding model, query phrasing (paraphrase failures), metadata filters; try hybrid search and re-ranking. (2) Context: is the right chunk present but truncated, buried among many irrelevant ones, or contradicted? (3) Generation: with the correct chunk provided, does the model answer faithfully? Tighten instructions, require citations, and evaluate faithfulness separately from correctness." },
    ],
    practice: [
      {
        t: "quiz",
        questions: [
          { q: "In CLIP's loss for a batch of N pairs, which entries of the N×N similarity matrix are the positives?", options: ["The first row", "The diagonal", "All entries above 0.5", "The last column"], answer: 1, explain: "Pair i's image and caption are at (i, i)." },
          { kind: "numeric", q: "A CLIP batch has N = 1024 pairs. What is the image→text loss of a model whose similarities are all equal (natural log, 3 decimals)?", answer: 6.931, tolerance: 0.002, explain: "$\\log 1024 = 6.9315$." },
          { kind: "numeric", q: "Query (1, 0, 1), chunk (1, 1, 1). Cosine similarity (3 decimals)?", answer: 0.816, tolerance: 0.002, explain: "$2/(\\sqrt2\\cdot\\sqrt3) = 0.8165$." },
          { kind: "numeric", q: "CLIP initialises its learnable log inverse temperature at log(1/0.07). What is that value (3 decimals)?", answer: 2.659, tolerance: 0.002, explain: "$\\ln(14.2857) = 2.6593$." },
          { q: "A user's question shares no words with the relevant chunk. Which retriever is most likely to find it?", options: ["TF-IDF", "BM25", "A dense embedding model", "Exact string match"], answer: 2, explain: "Sparse methods score 0 without shared terms; learned embeddings can match meaning." },
          { q: "What does a residual network with L shared-weight layers and step 1/L correspond to?", options: ["RK4 with L steps", "Euler's method with L steps for dh/dt = f(h)", "The adjoint method", "A recurrent network with attention"], answer: 1, explain: "$h_{l+1} = h_l + \\Delta t f(h_l)$ is an Euler step." },
          { kind: "numeric", q: "Euler's method has global error proportional to Δt. If 8 steps give error 0.0076, roughly what error do you expect with 32 steps (3 decimals)?", answer: 0.0019, tolerance: 0.0003, explain: "4× more steps → error ≈ 0.0076/4 = 0.0019 (the program printed 0.001890)." },
          { q: "Why does RAG's prompt usually instruct the model to answer only from the context?", options: ["To reduce token count", "So answers are grounded in and citable from retrieved sources rather than mixed with memorised, possibly outdated facts", "Because language models cannot read long prompts", "It is required by the embedding model"], answer: 1, explain: "Grounding and citations are RAG's main benefits." },
        ],
      },
      {
        t: "exercise",
        title: "CLIP loss by hand",
        prompt: "Two pairs with unit embeddings: images $u_1 = (1, 0)$, $u_2 = (0, 1)$; captions $v_1 = (0.8, 0.6)$, $v_2 = (0.6, 0.8)$; $\\tau = 0.5$. Compute the similarity logits, both directional losses and the CLIP loss.",
        hint: "$L = UV^\\top/\\tau$; each direction is a 2-way softmax cross-entropy.",
        solution: [
          {
            t: "steps",
            items: [
              { title: "Logits", tex: String.raw`UV^\top = \begin{pmatrix}0.8 & 0.6\\ 0.6 & 0.8\end{pmatrix},\qquad L = \begin{pmatrix}1.6 & 1.2\\ 1.2 & 1.6\end{pmatrix}` },
              { title: "Image → text", tex: String.raw`-\ln\frac{e^{1.6}}{e^{1.6} + e^{1.2}} = \ln(1 + e^{-0.4}) = 0.5130\ \text{for both rows}` },
              { title: "Text → image and total", tex: String.raw`L \text{ is symmetric, so the column losses are also } 0.5130;\quad \mathcal L = 0.5130` },
            ],
          },
          { t: "p", text: "The captions are hard to tell apart ($\\cos(v_1, v_2) = 0.96$), so the loss is close to the uniform-guess value $\\ln 2 = 0.6931$. Lowering $\\tau$ to 0.1 would give $\\ln(1 + e^{-2}) = 0.1269$: a sharper softmax rewards the same small margin more." },
        ],
      },
      {
        t: "exercise",
        title: "Coding: a complete mini-RAG with a dense retriever",
        prompt: "Extend `rag_retrieval_sklearn.py`: (1) build a dense retriever by training a small bi-encoder — or, without training, by representing each word with the skip-gram vectors of [Word Embeddings](/learn/m13/word-embeddings) and averaging them per chunk; (2) combine dense and TF-IDF scores as $\\lambda\\cos_{\\text{dense}} + (1-\\lambda)\\cos_{\\text{tfidf}}$; (3) report recall@1 and recall@3 for $\\lambda \\in \\{0, 0.5, 1\\}$, adding at least five more paraphrased questions; (4) assemble the final prompt for the top-3 chunks.",
        hint: "Average word vectors: `E_dense[i] = mean(vec[w] for w in tokens(chunk_i) if w in vec)`, then L2-normalise. Keep the same chunk order for both score matrices so they can be added.",
        solution: [
          {
            t: "code",
            lang: "text",
            code: "def dense(texts, vec):\n    out = []\n    for t in texts:\n        ws = [vec[w] for w in re.findall(r'[a-z]+', t.lower()) if w in vec]\n        v = np.mean(ws, axis=0) if ws else np.zeros(dim)\n        out.append(v / (np.linalg.norm(v) + 1e-9))\n    return np.array(out)\n\nS_sparse = (vectorizer.transform(qs) @ index.T).toarray()   # cosine: rows are unit length\nS_dense = dense(qs, vec) @ dense(chunks, vec).T\nfor lam in (0.0, 0.5, 1.0):\n    S = lam * S_dense + (1 - lam) * S_sparse\n    top = np.argsort(-S, axis=1)[:, :3]\n    r1 = np.mean(top[:, 0] == gold); r3 = np.mean([g in t for g, t in zip(gold, top)])\n    print(lam, r1, r3)",
          },
          { t: "p", text: "Expect the dense retriever to recover paraphrased questions that TF-IDF misses, and TF-IDF to win on questions with rare exact terms; the hybrid ($\\lambda = 0.5$) is usually the most robust. Report your measured recall values — with a tiny corpus and averaged word vectors the differences can be noisy, so add questions until the comparison is meaningful." },
        ],
      },
    ],
    summary: [
      {
        t: "list",
        items: [
          "Joint embedding space: separate encoders, matching image–text pairs close in cosine similarity.",
          "CLIP loss: $L_{ij} = u_i^\\top v_j/\\tau$; $\\tfrac12$(row cross-entropy + column cross-entropy) with diagonal targets; learnable $\\log(1/\\tau)$ initialised at 2.659; worked example 0.0439.",
          "Zero-shot: class weights $w_c$ = normalised mean of prompt embeddings; a linear classifier generated from text; prompt ensembling helps (digits: 0.967).",
          "VLMs: image encoder → connector → visual tokens in a language model (or cross-attention).",
          "RAG: chunk (with overlap) → embed → index → top-$k$ by cosine → prompt = instructions + chunks + question → generate with citations. Evaluate retrieval (recall@k) separately.",
          "Sparse (TF-IDF/BM25) matches words; dense embeddings match meaning; hybrid and re-ranking combine strengths.",
          "Neural ODE: $dh/dt = f_\\theta(h, t)$; a ResNet is Euler's method (error $O(\\Delta t)$); adjoint method gives constant-memory gradients.",
        ],
      },
    ],
    next: [
      {
        t: "list",
        items: [
          "[Pretraining Large Language Models](/learn/m20/llm-pretraining) — the generators that RAG and vision-language models are built around.",
          "[Graph Neural Networks](/learn/m15/graph-neural-networks) — another structured modality: learning on nodes and edges with message passing.",
          "[Serving](/learn/m18/serving) — putting embedding models, vector search and generators behind an API.",
        ],
      },
    ],
  },
};

export default lesson;
