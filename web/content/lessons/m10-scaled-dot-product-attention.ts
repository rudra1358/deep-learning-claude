import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m10/scaled-dot-product-attention",
  title: "Queries, Keys, Values and Scaled Dot-Product Attention",
  summary:
    "The core operation of the Transformer, computed by hand: learned query, key and value projections, the QKᵀ score matrix, the √dₖ scaling argument derived from variance, row-wise softmax, padding and causal masks, self- vs cross-attention, multi-head attention with its shapes and parameter count, and the O(n²d) cost — implemented in NumPy and verified against PyTorch.",
  minutes: 70,
  level: "intermediate",
  prerequisites: ["m10/attention-intro", "m02/vectors-matrices", "m03/activation-functions", "m02/probability"],
  next: ["m10/multi-head-attention", "m10/positional-encoding", "m11/transformer-architecture"],
  terms: ["attention", "query-key-value", "multi-head-attention"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "**Scaled dot-product attention** is the operation that lets every position of a sequence build its new representation by looking at every other position, deciding how relevant each one is, and taking a weighted average of what they contain. It is a few lines of linear algebra — two matrix multiplications and a softmax — yet it is the computational heart of the Transformer and therefore of essentially every modern large language model, speech recogniser, and many vision models.",
      },
      {
        t: "p",
        text: "It was introduced to fix a structural weakness of recurrent networks. An encoder–decoder RNN for translation reads a whole source sentence and squeezes it into a single fixed-size hidden vector, the **context vector**, which the decoder must then unpack into the target sentence. Long sentences overflow this bottleneck, and information about early words has to survive many steps of repeated multiplication to reach the end, so long-range dependencies are learned poorly. You met this problem, and the first fix, in [From Seq2Seq Bottlenecks to Attention](/learn/m10/attention-intro).",
      },
      {
        t: "p",
        text: "Bahdanau, Cho & Bengio (2014/2015, “Neural Machine Translation by Jointly Learning to Align and Translate”) let the decoder look back at **all** encoder hidden states at every output step, scoring each with a small neural network (additive attention) and averaging them with softmax weights. Vaswani et al. (2017, “Attention Is All You Need”) went further: they removed recurrence entirely, replaced the additive scorer with a scaled dot product, and used attention as the main layer of the network. Every position can now reach every other position in **one** step, and all positions are processed in parallel on a GPU.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The equation of this lesson",
        text: String.raw`$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ — compare every query with every key, scale, normalise each row into weights, and average the values with those weights.`,
      },
      {
        t: "p",
        text: "Where it is used: in every layer of encoder models (BERT), decoder language models (GPT-style models), encoder–decoder models (T5, the original translation Transformer), speech models (Whisper), and Vision Transformers that treat image patches as tokens.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      { t: "p", text: "You need four things, all used in their most basic form:" },
      {
        t: "list",
        items: [
          "[Vectors and matrices](/learn/m02/vectors-matrices): the **dot product** as a similarity score, **matrix multiplication** shapes ($(n\\times d)(d\\times m)=(n\\times m)$), and the **transpose**. Attention is almost entirely matrix products, so tracking shapes is the main skill.",
          "[Activation functions — softmax](/learn/m03/activation-functions): softmax turns a vector of arbitrary real numbers into non-negative numbers that sum to 1. We apply it to each row of a matrix.",
          "[Probability](/learn/m02/probability): **expectation** and **variance** of a sum of independent random variables. This is all you need to derive the $\\sqrt{d_k}$ scaling factor.",
          "[From Seq2Seq Bottlenecks to Attention](/learn/m10/attention-intro): the idea of an alignment score between a decoder state and encoder states, and why a single context vector is a bottleneck. Familiarity with [RNNs](/learn/m09/rnn-basics) helps for the motivation but is not required for the mechanism.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A soft dictionary lookup",
        text: "A Python dictionary stores (key, value) pairs. You give it a query; it finds the one key that matches exactly and returns that key's value. Attention is a **soft** version of this: the query is compared with **every** key, each comparison gives a similarity score, the scores become weights that sum to 1, and you get back a **blend** of all the values, dominated by the values whose keys matched best. Because the blend is differentiable, the network can learn what to ask for and what to advertise.",
      },
      {
        t: "p",
        text: "A second picture: a library. You walk in with a question (the **query**). Every book has a label on its spine (the **key**) and contents inside (the **value**). You glance along the shelf and judge how well each label matches your question. Instead of taking only the best book, you read a little of every book in proportion to how well its label matched, and your notes are the weighted mix.",
      },
      {
        t: "p",
        text: "In a sentence such as “the animal didn't cross the street because **it** was too tired”, the representation of “it” should absorb information from “animal”. With self-attention, the word “it” issues a query; “animal” can expose a key that matches that query well; the output vector for “it” then contains a large share of “animal”'s value vector. Crucially, nothing about distance matters here: “animal” could be 5 or 500 tokens away and the computation is the same single step.",
      },
      {
        t: "p",
        text: "Why three different vectors per token rather than one? Because the roles are different. What a word is **looking for** (its query) is not the same as what it should be **found by** (its key), and neither is necessarily what it should **pass on** once found (its value). Three separate learned linear maps give the model the freedom to separate these roles.",
      },
      {
        t: "callout",
        kind: "note",
        title: "Where the analogy breaks",
        text: "A real dictionary has fixed keys. In attention, keys, queries and values are all computed from the same token vectors by learned matrices, so the model learns both the lookup questions and the index at the same time — by gradient descent on the final task loss.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      {
        t: "p",
        text: "We describe **self-attention** on one sequence of $n$ tokens. The input is a matrix $X$ whose row $i$ is the vector for token $i$ (an embedding plus positional information), of width $d_{\\text{model}}$.",
      },
      {
        t: "steps",
        items: [
          {
            title: "1. Project into queries, keys and values",
            text: "Multiply the input by three learned weight matrices. Every token gets a query row, a key row and a value row. Same matrices for every position: this is weight sharing across positions.",
            tex: String.raw`Q = XW_Q \in \mathbb{R}^{n\times d_k},\quad K = XW_K \in \mathbb{R}^{n\times d_k},\quad V = XW_V \in \mathbb{R}^{n\times d_v}`,
          },
          {
            title: "2. Score every query against every key",
            text: "Entry (i, j) of the score matrix is the dot product of query i with key j: how much token i wants what token j offers. One matrix product computes all n² scores.",
            tex: String.raw`S = QK^\top \in \mathbb{R}^{n\times n},\qquad S_{ij} = q_i\cdot k_j = \sum_{c=1}^{d_k} Q_{ic}K_{jc}`,
          },
          {
            title: "3. Scale by 1/√d_k",
            text: "Divide every score by the square root of the key dimension, so the scores keep roughly unit variance regardless of d_k (derived in the next section).",
            tex: String.raw`\tilde S = \frac{S}{\sqrt{d_k}}`,
          },
          {
            title: "4. Apply the mask (optional)",
            text: "Add a matrix M whose entries are 0 where attention is allowed and −∞ where it is forbidden (padding positions, or future positions in a decoder).",
            tex: String.raw`\tilde S \leftarrow \tilde S + M,\qquad M_{ij}\in\{0,-\infty\}`,
          },
          {
            title: "5. Softmax each row",
            text: "Turn each row of scores into a probability distribution over keys. Row i now says how query i divides its attention among all keys; masked entries become exactly 0.",
            tex: String.raw`A_{ij} = \frac{\exp(\tilde S_{ij})}{\sum_{l=1}^{n}\exp(\tilde S_{il})},\qquad \sum_j A_{ij}=1`,
          },
          {
            title: "6. Weighted sum of values",
            text: "Each output row is the attention-weighted average of the value rows. This is the new, context-aware representation of token i.",
            tex: String.raw`Z = AV \in \mathbb{R}^{n\times d_v},\qquad z_i = \sum_{j=1}^n A_{ij}\, v_j`,
          },
        ],
      },
      { t: "h", text: "Masks" },
      {
        t: "p",
        text: "A **padding mask** handles batches of sequences of different length. Shorter sequences are padded with dummy tokens up to a common length; their key columns are masked with $-\\infty$ so no real query attends to them. A **causal (look-ahead) mask** is used in decoders and language models trained to predict the next token: query $i$ may only see keys $j\\le i$, so the strictly upper triangle of the score matrix is set to $-\\infty$. Because $e^{-\\infty}=0$, masked entries get weight exactly zero and future tokens cannot leak information backwards. Masks combine with a logical AND (or by adding their $0/-\\infty$ matrices).",
      },
      { t: "h", text: "Self-attention vs cross-attention" },
      {
        t: "p",
        text: "In **self-attention**, queries, keys and values all come from the same sequence $X$, so the score matrix is $n\\times n$. In **cross-attention** (encoder–decoder attention), queries come from one sequence (the decoder states, $n_q$ rows) and keys and values from another (the encoder output, $n_k$ rows): $Q = X_{\\text{dec}}W_Q$, $K = X_{\\text{enc}}W_K$, $V = X_{\\text{enc}}W_V$. The score matrix is $n_q\\times n_k$ and the output has one row per query. This is exactly Bahdanau's idea — the decoder looking at the source sentence — with dot-product scoring.",
      },
      { t: "h", text: "Multi-head attention" },
      {
        t: "p",
        text: "A single softmax row is one weighted average: it tends to focus on one kind of relationship at a time. **Multi-head attention** runs $h$ attention operations in parallel on lower-dimensional projections and combines them. With $d_k=d_v=d_{\\text{model}}/h$:",
      },
      {
        t: "steps",
        items: [
          { title: "Project", text: "Compute Q, K, V with full d_model × d_model matrices (equivalently, h separate d_model × d_k matrices side by side).", tex: String.raw`Q = XW_Q,\ K = XW_K,\ V = XW_V \quad (n\times d_{\text{model}})` },
          { title: "Split", text: "Reshape each into h heads of width d_k: column block i belongs to head i. Shape (n, d_model) → (h, n, d_k).", tex: String.raw`Q^{(i)} = Q_{[:,\,(i-1)d_k : i\,d_k]}` },
          { title: "Attend per head", text: "Each head runs scaled dot-product attention independently, producing its own n × n weight matrix and n × d_k output.", tex: String.raw`\text{head}_i = \mathrm{softmax}\!\left(\frac{Q^{(i)}K^{(i)\top}}{\sqrt{d_k}}+M\right)V^{(i)}` },
          { title: "Concatenate and mix", text: "Stack the head outputs side by side back to n × d_model and apply the output projection W_O, which lets information from different heads interact.", tex: String.raw`\mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\qquad W_O\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}` },
        ],
      },
      {
        t: "p",
        text: "Because the heads split $d_{\\text{model}}$ rather than each using all of it, multi-head attention costs about the same as one full-width head. The topic gets its own lesson in [Multi-Head Attention](/learn/m10/multi-head-attention).",
      },
      { t: "h", text: "From attention to the Transformer block" },
      {
        t: "p",
        text: "One Transformer encoder layer wraps multi-head self-attention in a residual connection and layer normalisation, then applies a position-wise feed-forward network (two linear layers with a non-linearity) with another residual and normalisation. The decoder adds a causally masked self-attention sub-layer and a cross-attention sub-layer over the encoder output. Stack 6 to 100+ such layers and you have a Transformer — see [The Transformer Architecture](/learn/m11/transformer-architecture). Attention mixes information **across** positions; the feed-forward network transforms each position **independently**.",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "Definitions and shapes" },
      {
        t: "math",
        tex: String.raw`\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V`,
        caption: "Scaled dot-product attention (Vaswani et al., 2017), with optional mask M.",
      },
      {
        t: "list",
        items: [
          String.raw`$n$ — number of query positions (tokens); $n_k$ — number of key/value positions ($n_k=n$ in self-attention).`,
          String.raw`$d_{\text{model}}$ — width of the token vectors; $X\in\mathbb{R}^{n\times d_{\text{model}}}$ stacks them as rows.`,
          String.raw`$W_Q, W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}$, $W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}$ — learned projection matrices.`,
          String.raw`$Q\in\mathbb{R}^{n\times d_k}$ (queries), $K\in\mathbb{R}^{n_k\times d_k}$ (keys), $V\in\mathbb{R}^{n_k\times d_v}$ (values). Queries and keys must share width $d_k$ to be dot-producted.`,
          String.raw`$QK^\top\in\mathbb{R}^{n\times n_k}$ — score matrix; row $i$ = query $i$ against all keys.`,
          String.raw`$M\in\{0,-\infty\}^{n\times n_k}$ — mask; $\mathrm{softmax}$ is applied to each row independently.`,
          String.raw`$A\in\mathbb{R}^{n\times n_k}$ — attention weights, $A_{ij}\ge 0$, $\sum_j A_{ij}=1$; output $Z=AV\in\mathbb{R}^{n\times d_v}$.`,
        ],
      },
      { t: "h", text: "Why divide by √dₖ: a variance derivation" },
      {
        t: "p",
        text: String.raw`Assume, as at initialisation, that the components of a query $q$ and a key $k$ are independent random variables with mean 0 and variance 1. Their dot product is $q\cdot k=\sum_{c=1}^{d_k} q_c k_c$. For one term, independence gives $\mathbb{E}[q_ck_c]=\mathbb{E}[q_c]\,\mathbb{E}[k_c]=0$ and`,
      },
      {
        t: "math",
        tex: String.raw`\mathrm{Var}(q_ck_c)=\mathbb{E}[q_c^2k_c^2]-\big(\mathbb{E}[q_ck_c]\big)^2=\mathbb{E}[q_c^2]\,\mathbb{E}[k_c^2]-0=1\cdot 1=1.`,
      },
      {
        t: "p",
        text: String.raw`The $d_k$ terms are independent, so their variances add: $\mathrm{Var}(q\cdot k)=d_k$ and the standard deviation is $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ gives $\mathrm{Var}\big(q\cdot k/\sqrt{d_k}\big)=d_k/d_k=1$, independent of the head size.`,
      },
      {
        t: "p",
        text: String.raw`Why it matters: softmax is sensitive to the **scale** of its inputs. If scores have standard deviation 8 (for $d_k=64$), the largest score in a row is typically many units above the rest, and softmax puts almost all mass on it — it **saturates** into a near one-hot vector. The softmax Jacobian is`,
      },
      {
        t: "math",
        tex: String.raw`\frac{\partial A_{i}}{\partial s_{j}} = A_{i}\,(\delta_{ij}-A_{j}),`,
        caption: "δᵢⱼ is 1 if i = j and 0 otherwise. If one weight is ≈1 and the rest ≈0, every entry is ≈0.",
      },
      {
        t: "p",
        text: "so a near one-hot output has a near-zero gradient: the scores barely learn. The experiment below (in the code sections) measures both effects: the dot-product variance grows exactly like $d_k$, and without scaling the softmax Jacobian norm collapses from 0.39 to $1.2\\times10^{-5}$ as $d_k$ grows from 4 to 512, while the scaled version is unaffected.",
      },
      {
        t: "code",
        lang: "python",
        title: "Measuring the variance and the saturation",
        code: String.raw`import numpy as np

rng = np.random.default_rng(0)
n_samples = 100_000

# 1) Variance of a dot product of independent unit-variance vectors grows like d_k
for d_k in [4, 16, 64, 256]:
    q = rng.standard_normal((n_samples, d_k))
    k = rng.standard_normal((n_samples, d_k))
    dots = (q * k).sum(axis=1)
    print(f"d_k={d_k:4d}  var(q.k)={dots.var():8.2f}  "
          f"var(q.k/sqrt(d_k))={(dots / np.sqrt(d_k)).var():.3f}")

# 2) Unscaled scores saturate the softmax and kill its gradient
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def jacobian_norm(p):
    J = np.diag(p) - np.outer(p, p)      # d softmax / d z
    return np.linalg.norm(J)

n_keys = 8
z = rng.standard_normal(n_keys)          # unit-variance (properly scaled) scores
print()
for d_k in [4, 64, 512]:
    unscaled = z * np.sqrt(d_k)          # same scores without the 1/sqrt(d_k)
    p_u, p_s = softmax(unscaled), softmax(z)
    print(f"d_k={d_k:3d}  unscaled: max weight={p_u.max():.4f}, |J|={jacobian_norm(p_u):.2e}   "
          f"scaled: max weight={p_s.max():.4f}, |J|={jacobian_norm(p_s):.3f}")`,
        output: `d_k=   4  var(q.k)=    4.03  var(q.k/sqrt(d_k))=1.007
d_k=  16  var(q.k)=   16.01  var(q.k/sqrt(d_k))=1.001
d_k=  64  var(q.k)=   63.86  var(q.k/sqrt(d_k))=0.998
d_k= 256  var(q.k)=  255.13  var(q.k/sqrt(d_k))=0.997

d_k=  4  unscaled: max weight=0.5547, |J|=3.87e-01   scaled: max weight=0.3464, |J|=0.383
d_k= 64  unscaled: max weight=0.9793, |J|=3.29e-02   scaled: max weight=0.3464, |J|=0.383
d_k=512  unscaled: max weight=1.0000, |J|=1.15e-05   scaled: max weight=0.3464, |J|=0.383`,
      },
      { t: "h", text: "Fully worked example: 3 tokens, d_model = 4, d_k = d_v = 2" },
      {
        t: "p",
        text: "Take three token vectors and three small projection matrices (chosen with integer entries so you can check every product by hand):",
      },
      {
        t: "math",
        tex: String.raw`X=\begin{bmatrix}1&0&1&0\\0&2&0&1\\1&1&1&1\end{bmatrix},\;
W_Q=\begin{bmatrix}1&0\\0&1\\1&0\\0&1\end{bmatrix},\;
W_K=\begin{bmatrix}0&1\\1&0\\1&0\\0&1\end{bmatrix},\;
W_V=\begin{bmatrix}1&0\\0&2\\0&1\\1&0\end{bmatrix}`,
      },
      { t: "p", text: "**Step 1 — projections.** For example, row 1 of $Q$ is $[1,0,1,0]W_Q=[1+1,\\;0+0]=[2,0]$." },
      {
        t: "math",
        tex: String.raw`Q=XW_Q=\begin{bmatrix}2&0\\0&3\\2&2\end{bmatrix},\quad
K=XW_K=\begin{bmatrix}1&1\\2&1\\2&2\end{bmatrix},\quad
V=XW_V=\begin{bmatrix}1&1\\1&4\\2&3\end{bmatrix}`,
      },
      { t: "p", text: "**Step 2 — scores.** $S_{ij}=q_i\\cdot k_j$; e.g. $S_{23}=[0,3]\\cdot[2,2]=6$ and $S_{33}=[2,2]\\cdot[2,2]=8$." },
      {
        t: "math",
        tex: String.raw`S=QK^\top=\begin{bmatrix}2&4&4\\3&3&6\\4&6&8\end{bmatrix}`,
      },
      { t: "p", text: "**Step 3 — scale** by $1/\\sqrt{2}=0.70711$:" },
      {
        t: "math",
        tex: String.raw`\tilde S=\frac{S}{\sqrt 2}=\begin{bmatrix}1.4142&2.8284&2.8284\\2.1213&2.1213&4.2426\\2.8284&4.2426&5.6569\end{bmatrix}`,
      },
      {
        t: "p",
        text: "**Step 4 — softmax per row.** Subtract each row's maximum (this does not change the result) and exponentiate. Row 1: $e^{1.4142-2.8284}=e^{-1.4142}=0.2431$, and $e^0=1$ twice; the sum is $2.2431$. Row 2: $0.1199, 0.1199, 1$ with sum $1.2397$. Row 3: $0.0591, 0.2431, 1$ with sum $1.3022$. Dividing by the sums:",
      },
      {
        t: "math",
        tex: String.raw`A=\begin{bmatrix}0.1084&0.4458&0.4458\\0.0967&0.0967&0.8066\\0.0454&0.1867&0.7679\end{bmatrix},\qquad \text{each row sums to }1.`,
      },
      {
        t: "p",
        text: "**Step 5 — weighted sum of values.** Row 1: $0.1084[1,1]+0.4458[1,4]+0.4458[2,3]=[0.1084+0.4458+0.8916,\\;0.1084+1.7832+1.3374]=[1.4458,\\;3.2290]$.",
      },
      {
        t: "math",
        tex: String.raw`Z=AV=\begin{bmatrix}1.4458&3.2290\\1.8066&2.9033\\1.7679&3.0959\end{bmatrix}`,
      },
      {
        t: "p",
        text: String.raw`**With a causal mask.** Setting $\tilde S_{12},\tilde S_{13},\tilde S_{23}$ to $-\infty$ gives rows $[1,0,0]$, $[0.5,0.5,0]$ (the two visible scores are equal, 2.1213), and the unchanged third row $[0.0454, 0.1867, 0.7679]$. The outputs become $z_1=v_1=[1,1]$, $z_2=\tfrac12(v_1+v_2)=[1,2.5]$, and $z_3=[1.7679,3.0959]$ as before: the last token could already see everything.`,
      },
      { t: "h", text: "Multi-head attention: shapes and parameters" },
      {
        t: "math",
        tex: String.raw`\text{head}_i=\mathrm{Attention}\big(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}\big),\quad W_Q^{(i)},W_K^{(i)},W_V^{(i)}\in\mathbb{R}^{d_{\text{model}}\times d_k},\quad d_k=\frac{d_{\text{model}}}{h}`,
      },
      {
        t: "math",
        tex: String.raw`\#\text{params}=\underbrace{3\cdot h\cdot d_{\text{model}}d_k}_{W_Q,W_K,W_V}+\underbrace{h\,d_k\cdot d_{\text{model}}}_{W_O}+\underbrace{4d_{\text{model}}}_{\text{biases}}=4d_{\text{model}}^2+4d_{\text{model}}`,
        caption: "Independent of the number of heads h. For d_model = 512: 4·512² + 4·512 = 1,050,624.",
      },
      { t: "h", text: "Computational complexity" },
      {
        t: "list",
        items: [
          String.raw`Projections $Q,K,V$ and $W_O$: four $(n\times d)(d\times d)$ products, $O(n d^2)$ time.`,
          String.raw`Scores $QK^\top$ over all heads: $h\cdot n^2 d_k = n^2 d$ multiply-adds, $O(n^2 d)$ time.`,
          String.raw`Weighted sum $AV$: also $O(n^2 d)$. Memory for the weight matrices: $O(h n^2)$.`,
          String.raw`Total per layer: $O(n^2 d + n d^2)$. For long sequences the $n^2$ term dominates; doubling the context length quadruples attention cost. An RNN is $O(n d^2)$ but needs $n$ **sequential** steps; attention needs $O(1)$ sequential steps and a maximum path length of 1 between any two positions.`,
        ],
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "p",
        text: "The explorer below runs real multi-head self-attention ($d_{\\text{model}}=8$) on any sentence you type. Each stage shows the actual matrices with their numbers. The weights are seeded random numbers, not a trained model, so do not look for grammar in the patterns — look at the arithmetic.",
      },
      { t: "sim", id: "attention", caption: "Every number shown is computed live from the formulas in this lesson." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "Press **Play** once to walk through all nine stages. At stage 4 (scores), click a row: the side panel shows that query's dot products, the division by $\\sqrt{d_k}$, the exponentials and the final weights.",
          "At stage 7 (softmax), check the **Σ row** column: every row sums to 1.000. Compare the weights with the raw scores: the ordering within a row is preserved, only the scale changes.",
          "Switch **Heads** between 1, 2 and 4. Note $d_k$ shrinks (8, 4, 2) and the divisor $\\sqrt{d_k}$ changes; at stage 7 the mini heatmaps show that each head produces a different pattern from the same input.",
          "Turn on **Causal mask** and go to stage 6: the upper triangle becomes −∞. At stage 7 the first row is exactly $[1, 0, \\dots, 0]$ — the first token can only attend to itself.",
          "Type a sentence with a repeated word, e.g. “the dog saw the dog”. Turn **Positional encoding** off: the two copies of “dog” get identical rows in Q, K and V and therefore identical outputs. Turn it back on and they differ.",
          "At stage 9, verify the shape chain $(n\\times 8)\\to h\\times(n\\times d_k)\\to(n\\times 8)\\to(n\\times 8)$ and that the parameter count does not depend on the number of heads.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "A complete NumPy implementation: a batched scaled dot-product attention function with an optional boolean mask, a causal-mask helper, and a multi-head attention layer usable for both self- and cross-attention. The script reproduces the worked example, checks that causal masking really blocks the future, and runs cross-attention between sequences of different lengths.",
      },
      {
        t: "code",
        lang: "python",
        title: "attention_numpy.py",
        runnable: true,
        code: String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(z, axis=-1):
    """Numerically stable softmax: subtracting the max does not change the result."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., n_q, d_k)   K: (..., n_k, d_k)   V: (..., n_k, d_v)
    mask: boolean, broadcastable to (..., n_q, n_k); True = may attend, False = blocked.
    Returns output (..., n_q, d_v) and weights (..., n_q, n_k).
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)     # (..., n_q, n_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)          # blocked -> -inf -> weight 0
    weights = softmax(scores, axis=-1)                     # each row sums to 1
    return weights @ V, weights


def causal_mask(n):
    """Lower-triangular boolean matrix: query i may see keys j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.h = d_model, num_heads
        self.d_k = d_model // num_heads
        scale = 1.0 / np.sqrt(d_model)                     # keeps activations O(1)
        self.W_Q, self.W_K, self.W_V, self.W_O = (rng.normal(0, scale, (d_model, d_model)) for _ in range(4))
        self.b_Q, self.b_K, self.b_V, self.b_O = (np.zeros(d_model) for _ in range(4))

    def split_heads(self, X):                              # (n, d_model) -> (h, n, d_k)
        n = X.shape[0]
        return X.reshape(n, self.h, self.d_k).transpose(1, 0, 2)

    def __call__(self, x_q, x_kv, mask=None):
        Q = self.split_heads(x_q @ self.W_Q + self.b_Q)    # (h, n_q, d_k)
        K = self.split_heads(x_kv @ self.W_K + self.b_K)   # (h, n_k, d_k)
        V = self.split_heads(x_kv @ self.W_V + self.b_V)   # (h, n_k, d_k)
        heads, weights = scaled_dot_product_attention(Q, K, V, mask)   # (h, n_q, d_k)
        concat = heads.transpose(1, 0, 2).reshape(x_q.shape[0], self.d_model)  # (n_q, d_model)
        return concat @ self.W_O + self.b_O, weights

    def num_params(self):
        return 4 * self.d_model * self.d_model + 4 * self.d_model


# 1) The worked example from the lesson (3 tokens, d_model = 4, d_k = 2)
X = np.array([[1., 0., 1., 0.], [0., 2., 0., 1.], [1., 1., 1., 1.]])
W_Q = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
W_K = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
W_V = np.array([[1., 0.], [0., 2.], [0., 1.], [1., 0.]])
Z, A = scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V)
print("worked example A =\n", A)
print("worked example Z =\n", Z)
Zc, Ac = scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V, causal_mask(3))
print("causal A =\n", Ac)

# 2) Multi-head causal self-attention on a random sequence
rng = np.random.default_rng(0)
n, d_model, h = 5, 8, 2
x = rng.normal(size=(n, d_model))
mha = MultiHeadAttention(d_model, h, rng)
y, w = mha(x, x, causal_mask(n))
print("\noutput shape", y.shape, "| weights shape", w.shape)
print("row sums of head-0 weights:", w[0].sum(axis=1))
print("head-0 weights (upper triangle must be 0):\n", w[0])

# Causality check: changing the LAST token must not change outputs at earlier positions
x2 = x.copy(); x2[-1] += 10.0
y2, _ = mha(x2, x2, causal_mask(n))
print("max change at positions 0..3:", np.abs(y2[:-1] - y[:-1]).max())
print("max change at position 4   :", round(float(np.abs(y2[-1] - y[-1]).max()), 4))

# 3) Cross-attention: 5 decoder queries attend over 7 encoder positions
memory = rng.normal(size=(7, d_model))
yc, wc = mha(x, memory)
print("\ncross-attention output", yc.shape, "weights", wc.shape)
print("parameters (4*d^2 + 4*d):", mha.num_params())`,
        output: `worked example A =
 [[0.1084 0.4458 0.4458]
 [0.0967 0.0967 0.8066]
 [0.0454 0.1867 0.7679]]
worked example Z =
 [[1.4458 3.229 ]
 [1.8066 2.9033]
 [1.7679 3.0959]]
causal A =
 [[1.     0.     0.    ]
 [0.5    0.5    0.    ]
 [0.0454 0.1867 0.7679]]

output shape (5, 8) | weights shape (2, 5, 5)
row sums of head-0 weights: [1. 1. 1. 1. 1.]
head-0 weights (upper triangle must be 0):
 [[1.     0.     0.     0.     0.    ]
 [0.4123 0.5877 0.     0.     0.    ]
 [0.4852 0.1894 0.3254 0.     0.    ]
 [0.2973 0.324  0.198  0.1807 0.    ]
 [0.2124 0.0563 0.4067 0.1621 0.1624]]
max change at positions 0..3: 0.0
max change at position 4   : 2.0164

cross-attention output (5, 8) weights (2, 5, 7)
parameters (4*d^2 + 4*d): 288`,
      },
      {
        t: "callout",
        kind: "tip",
        title: "Reading the output",
        text: "The worked-example matrices match the hand calculation. The causality check is the important one: perturbing token 4 by +10 changes nothing at positions 0–3 (difference exactly 0.0), because those rows give token 4 a weight of exactly zero.",
      },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "PyTorch provides the fused kernel `torch.nn.functional.scaled_dot_product_attention` and the full layer `torch.nn.MultiheadAttention`. This program checks both against the NumPy formulation. Note two conventions: `nn.MultiheadAttention` stores the three input projections stacked in `in_proj_weight` and applies them as `x @ W.T`, and its boolean `attn_mask` uses **True = not allowed** (the opposite of our NumPy helper).",
              },
              {
                t: "code",
                lang: "python",
                title: "attention_torch_check.py",
                code: String.raw`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# --- NumPy reference (same as the from-scratch section) ---
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def sdpa_np(Q, K, V, mask=None):
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    return softmax(s) @ V


# 1) Functional scaled dot-product attention, with and without a causal mask
B, h, n, d_k = 2, 4, 6, 16                        # batch, heads, tokens, head dim
Q, K, V = (torch.randn(B, h, n, d_k) for _ in range(3))
out_t = F.scaled_dot_product_attention(Q, K, V)
out_t_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
causal = np.tril(np.ones((n, n), dtype=bool))
print("SDPA      max |torch - numpy| =", np.abs(out_t.numpy() - sdpa_np(Q.numpy(), K.numpy(), V.numpy())).max())
print("SDPA+mask max |torch - numpy| =", np.abs(out_t_causal.numpy() - sdpa_np(Q.numpy(), K.numpy(), V.numpy(), causal)).max())

# 2) nn.MultiheadAttention: copy its weights into the NumPy formulation
d_model, n_heads = 32, 4
mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
x = torch.randn(1, n, d_model)
blocked = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)   # PyTorch: True = NOT allowed
y_t, w_t = mha(x, x, x, attn_mask=blocked, average_attn_weights=False)

W_in = mha.in_proj_weight.detach().numpy()        # (3*d_model, d_model), applied as x @ W.T
b_in = mha.in_proj_bias.detach().numpy()
W_Q, W_K, W_V = (W_in[i * d_model:(i + 1) * d_model].T for i in range(3))
b_Q, b_K, b_V = (b_in[i * d_model:(i + 1) * d_model] for i in range(3))
W_O = mha.out_proj.weight.detach().numpy().T
b_O = mha.out_proj.bias.detach().numpy()

xn = x[0].numpy()
split = lambda M: M.reshape(n, n_heads, d_model // n_heads).transpose(1, 0, 2)  # (h, n, d_k)
heads = sdpa_np(split(xn @ W_Q + b_Q), split(xn @ W_K + b_K), split(xn @ W_V + b_V), causal)
y_np = heads.transpose(1, 0, 2).reshape(n, d_model) @ W_O + b_O

print("MHA output shape:", tuple(y_t.shape), "| per-head weights:", tuple(w_t.shape))
print("MHA       max |torch - numpy| =", np.abs(y_t[0].detach().numpy() - y_np).max())
print("parameters:", sum(p.numel() for p in mha.parameters()), "= 4*d^2 + 4*d =", 4 * d_model**2 + 4 * d_model)`,
                output: `SDPA      max |torch - numpy| = 2.278752512241411e-07
SDPA+mask max |torch - numpy| = 2.0092021402184912e-07
MHA output shape: (1, 6, 32) | per-head weights: (1, 4, 6, 6)
MHA       max |torch - numpy| = 2.117106739918384e-07
parameters: 4224 = 4*d^2 + 4*d = 4224`,
              },
              {
                t: "p",
                text: "Differences of about $2\\times10^{-7}$ are float32 rounding: the from-scratch implementation and PyTorch compute the same function. In real models, prefer `F.scaled_dot_product_attention`: it dispatches to memory-efficient fused kernels (such as FlashAttention) that never materialise the full $n\\times n$ matrix in GPU memory.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras exposes the full layer as `keras.layers.MultiHeadAttention`. You pass `num_heads` and `key_dim` (the per-head size $d_k$); the output is projected back to the query's last dimension. `use_causal_mask=True` applies the look-ahead mask, and `return_attention_scores=True` returns the per-head weights of shape (batch, heads, n_q, n_k).",
              },
              {
                t: "code",
                lang: "python",
                title: "keras_mha.py (not executed in CI)",
                code: String.raw`import numpy as np
from tensorflow import keras

x = np.random.default_rng(0).normal(size=(2, 6, 32)).astype("float32")   # (batch, n, d_model)

mha = keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)            # d_k = 32 / 4
y, weights = mha(query=x, value=x, key=x,
                 use_causal_mask=True, return_attention_scores=True)

print("output:", y.shape, "| weights:", weights.shape)
print("rows sum to 1:", np.allclose(weights.numpy().sum(-1), 1.0, atol=1e-5))
print("max weight above diagonal:", np.abs(np.triu(weights.numpy()[0, 0], k=1)).max())
print("parameters:", mha.count_params())

# Cross-attention: 6 decoder queries over 9 encoder positions
memory = np.random.default_rng(1).normal(size=(2, 9, 32)).astype("float32")
yc, wc = keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)(
    query=x, value=memory, return_attention_scores=True)
print("cross:", yc.shape, wc.shape)`,
                output: "Prints the output shape (2, 6, 32) and weight shape (2, 4, 6, 6); True for the row-sum check; 0.0 for the largest weight above the diagonal (the causal mask works); the parameter count 4224 (= 4·32² + 4·32, the same as PyTorch); and for cross-attention the shapes (2, 6, 32) and (2, 4, 6, 9).",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "p",
        text: "Attention variants differ in how they **score** query–key pairs, **what** each query may attend to, and how keys and values are **shared** across heads.",
      },
      {
        t: "table",
        head: ["Variant", "Score / structure", "Advantages", "Disadvantages", "Typical use"],
        rows: [
          ["Additive (Bahdanau, 2014)", "$v^\\top\\tanh(W_1q+W_2k)$", "Works when q and k have different sizes; strong for small d", "Extra parameters; slower than one matrix product", "Early RNN translation models"],
          ["Dot-product (Luong-style)", "$q\\cdot k$", "One matmul, fast", "Scores grow with $d_k$; softmax saturates", "RNN attention, small $d_k$"],
          ["Scaled dot-product", "$q\\cdot k/\\sqrt{d_k}$", "Fast and stable at any head size", "Quadratic in $n$", "Every Transformer"],
          ["Multi-head", "h parallel scaled heads + $W_O$", "Several relations at once at the same cost", "More kernels to launch; heads can be redundant", "Standard in all Transformers"],
          ["Multi-query / grouped-query", "Heads share one (or a few) K, V projections", "Much smaller KV cache, faster decoding", "Small quality cost; needs retraining or uptraining", "Large decoder LLMs at inference"],
          ["Causal (masked)", "Upper triangle $=-\\infty$", "Enables parallel next-token training", "Each token sees only the past", "GPT-style decoders, Transformer decoder"],
          ["Cross-attention", "Q from one sequence, K, V from another", "Connects modalities or encoder to decoder", "Needs a second sequence", "Translation, captioning, text-to-image"],
          ["Sparse / local / sliding-window", "Each query sees a subset of keys", "Sub-quadratic cost, long contexts", "May miss long-range pairs; complex kernels", "Long-document and long-context models"],
          ["Linear / kernelised", "Replace softmax with a feature-map kernel", "O(n) time and memory", "Usually lower quality than softmax attention", "Research, very long sequences"],
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Machine translation** — the original Transformer (Vaswani et al., 2017) used encoder self-attention, causal decoder self-attention, and decoder-to-encoder cross-attention.",
          "**Language models** — GPT-style decoder-only models are stacks of causally masked multi-head self-attention plus feed-forward layers; the causal mask lets one forward pass train on every next-token prediction in a sequence at once.",
          "**Bidirectional encoders** — BERT uses unmasked self-attention (only a padding mask) so each token's representation uses both left and right context; used for classification, retrieval and search ranking.",
          "**Speech** — Whisper is an encoder–decoder Transformer: the decoder cross-attends to audio-encoder states while generating text.",
          "**Vision** — Vision Transformers (ViT) split an image into patches, treat each patch as a token and apply the same self-attention.",
          "**Multimodal generation** — text-conditioned image generators use cross-attention from image features (queries) to text-encoder outputs (keys and values) to inject the prompt.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths" },
      {
        t: "list",
        items: [
          "**Constant path length.** Any two positions interact in one layer, so long-range dependencies do not have to survive many recurrent steps.",
          "**Parallelism.** All positions are computed with dense matrix products, which GPUs and TPUs execute extremely efficiently. RNNs must process tokens one after another.",
          "**Content-based, dynamic weights.** The mixing weights depend on the input itself, unlike a convolution kernel whose weights are fixed after training.",
          "**Flexible structure.** Masks change what can see what (bidirectional, causal, padding, block patterns) without changing the layer.",
        ],
      },
      { t: "h", text: "Limitations" },
      {
        t: "list",
        items: [
          "**Quadratic cost.** Time is $O(n^2d)$ and naive memory is $O(hn^2)$. A 32k-token context means about $10^9$ score entries per head per layer. Fused kernels (FlashAttention) remove the memory blow-up but not the quadratic arithmetic.",
          "**No built-in notion of order.** Attention is permutation-equivariant; position must be injected separately — see [Positional Encoding](/learn/m10/positional-encoding).",
          "**Weak inductive bias.** Unlike convolutions, attention does not assume locality, so it typically needs more data to learn what a CNN gets for free.",
          "**KV cache at inference.** Autoregressive decoding must store K and V for every previous token in every layer, which dominates memory for long generations (the motivation for multi-query and grouped-query attention).",
          "**Interpretability is limited.** Attention weights show where information was read from, not why the output is what it is; treat attention maps as a hint, not an explanation.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Softmax over the wrong axis.** Normalising columns (`axis=0`) instead of rows makes the weights for each **key** sum to 1, which is meaningless. Each query's row must sum to 1: use `axis=-1` on a (…, n_q, n_k) score tensor.",
          "**Forgetting the scale, or using the wrong one.** Dividing by $d_k$ or $\\sqrt{d_{\\text{model}}}$ instead of $\\sqrt{d_k}$ (the per-head size) gives scores with the wrong spread. Without any scale, large heads train slowly because softmax saturates.",
          "**Mask convention mix-ups.** PyTorch's boolean `attn_mask` in `nn.MultiheadAttention` means True = blocked; `F.scaled_dot_product_attention` treats a boolean mask as True = allowed; Keras uses 1 = allowed. Always test that the upper triangle of the causal weight matrix is exactly 0.",
          "**Hard-coding −1e9 as the mask value.** In float16 the largest finite magnitude is about 65504, so −1e9 overflows to −inf; a row that is then fully masked computes (−inf) − (−inf) = NaN inside the softmax. Use the dtype's own minimum (`torch.finfo(dtype).min`) or −inf via `masked_fill`, and make sure every query can see at least one key.",
          "**Splitting heads with the wrong reshape.** `Q.reshape(h, n, d_k)` silently mixes positions and features. The correct split is reshape to (n, h, d_k) and then transpose to (h, n, d_k); merge in the reverse order.",
          "**Masking the queries instead of the keys for padding.** The padding mask must remove padded **key columns** so real tokens do not read from padding. Outputs at padded query rows are simply ignored by the loss.",
          "**Believing attention weights are explanations.** High weight on a token does not prove it caused the prediction; values are mixed again by $W_O$, residuals and later layers.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "What are queries, keys and values, and where do they come from?",
        a: "They are three linear projections of the input token vectors: $Q=XW_Q$, $K=XW_K$, $V=XW_V$, with learned matrices. The query of token i is compared with the key of every token j by a dot product to decide how much i attends to j; the values are what gets averaged with those weights. In self-attention all three come from the same sequence; in cross-attention Q comes from one sequence and K, V from another.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "Why is softmax applied to the scores?",
        a: "It converts each row of arbitrary real scores into non-negative weights that sum to 1, so the output is a convex combination (weighted average) of the value vectors. It is differentiable, keeps the ordering of scores, and its exponential sharpens differences, letting a query focus on a few keys while still passing gradient to all of them.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What is the difference between self-attention and cross-attention?",
        a: "In self-attention Q, K and V are all computed from the same sequence, so each token gathers context from its own sequence (score matrix n × n). In cross-attention the queries come from one sequence (e.g. decoder states) and the keys and values from another (e.g. encoder outputs), giving an $n_q\\times n_k$ score matrix. Translation decoders use both.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Derive why the scores are divided by √d_k.",
        a: "If the components of q and k are independent with mean 0 and variance 1, each product $q_ck_c$ has mean 0 and variance $\\mathbb{E}[q_c^2]\\mathbb{E}[k_c^2]=1$. Summing $d_k$ independent terms gives $\\mathrm{Var}(q\\cdot k)=d_k$, standard deviation $\\sqrt{d_k}$. Dividing by $\\sqrt{d_k}$ restores unit variance. Without it, large heads produce large score gaps, softmax becomes nearly one-hot, and its Jacobian $A_i(\\delta_{ij}-A_j)$ is close to zero, so learning stalls.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "How does a causal mask work, and why does it allow parallel training?",
        a: "Before the softmax, every score with key index j > query index i is set to −∞ (equivalently, add a matrix with −∞ in the strict upper triangle). After the softmax those weights are exactly 0, so the output at position i depends only on tokens 0..i. Because this holds for every row simultaneously, one forward pass over a whole sequence produces valid next-token predictions at all positions, with no information leaking from the future — this is teacher forcing done in parallel.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "What are the shapes and parameter count of multi-head attention with d_model = 512 and h = 8?",
        a: "$d_k=d_v=512/8=64$. $W_Q,W_K,W_V,W_O$ are each 512 × 512 (each of the Q/K/V matrices is equivalent to 8 blocks of 512 × 64). For an input of n tokens: Q, K, V are n × 512, reshaped to 8 × n × 64; each head's weights are n × n; head outputs 8 × n × 64 are concatenated to n × 512 and multiplied by $W_O$. Parameters: $4\\cdot512^2=1{,}048{,}576$ weights plus $4\\cdot512=2048$ biases, total 1,050,624 — independent of h.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "What is the time and memory complexity of self-attention, and what dominates for long sequences?",
        a: "Projections cost $O(nd^2)$; the score matrix and the weighted sum each cost $O(n^2d)$; storing the weights costs $O(hn^2)$ memory. For $n > d$ the $n^2$ terms dominate, so doubling the context roughly quadruples the attention cost. FlashAttention-style kernels reduce memory to O(n) by computing softmax in tiles, but the arithmetic stays quadratic.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Self-attention without positional information is permutation-equivariant. Prove it.",
        a: "Let P be a permutation matrix and replace X by PX. Then $Q'=PXW_Q=PQ$, $K'=PK$, $V'=PV$. Scores: $Q'K'^\\top=PQK^\\top P^\\top$, which permutes rows and columns the same way. Row-wise softmax commutes with this simultaneous permutation, so $A'=PAP^\\top$. Output: $A'V'=PAP^\\top PV=PAV$ since $P^\\top P=I$. So permuting the inputs just permutes the outputs: the layer cannot tell word order, which is why positional encodings are needed.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Why do large language models use multi-query or grouped-query attention at inference time?",
        a: "During autoregressive decoding, each new token's query attends to the keys and values of all previous tokens, so K and V are cached for every layer and head: memory $\\propto$ layers × heads × context × $d_k$. Decoding is then limited by memory bandwidth (reading this cache), not arithmetic. Multi-query attention shares one K/V head across all query heads; grouped-query attention shares one K/V head per group of query heads. This divides the cache size by the number of heads (or the group size) with a small quality cost.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "What happens if an entire row of the score matrix is masked, and how do you guard against it?",
        a: "Every entry is −∞, so the softmax computes $e^{-\\infty}/\\sum e^{-\\infty}=0/0=$ NaN, and the NaN propagates through the rest of the network and into the gradients. It can happen with padding-only sequences or combined padding and causal masks on left-padded inputs. Fixes: ensure each query can see at least one key (e.g. always allow attending to itself), replace −∞ by a large finite negative number of the correct dtype, or zero out the affected rows after the softmax.",
      },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "In self-attention on n tokens, what is the shape of the attention weight matrix of one head?",
            options: ["n × d_k", "d_k × d_k", "n × n", "d_model × n"],
            answer: 2,
            explain: "Every query (n of them) gets one weight per key (n of them), so A is n × n regardless of d_k.",
          },
          {
            kind: "numeric",
            q: "Query and key components are independent with mean 0 and variance 1, and d_k = 64. What is the standard deviation of the unscaled dot product q·k?",
            answer: 8,
            tolerance: 0.01,
            explain: "Var(q·k) = d_k = 64, so the standard deviation is √64 = 8. Dividing by √d_k = 8 makes it 1.",
          },
          {
            q: "Which statement about each row of the softmax output A is true?",
            options: [
              "Its entries are between −1 and 1",
              "Its entries are non-negative and sum to 1",
              "Its largest entry is always exactly 1",
              "Its entries sum to d_k",
            ],
            answer: 1,
            explain: "Softmax produces a probability distribution over keys for each query row.",
          },
          {
            kind: "numeric",
            q: "How many learned parameters (weights and biases) does a multi-head attention layer with d_model = 512 and h = 8 have?",
            answer: 1050624,
            tolerance: 0.5,
            explain: "4·512² + 4·512 = 1,048,576 + 2,048 = 1,050,624. The number of heads does not change it.",
          },
          {
            q: "In a causal mask for a decoder, which score entries S[i][j] are set to −∞?",
            options: ["j < i", "j > i", "j = i", "all entries in padded rows"],
            answer: 1,
            explain: "Query i must not see future keys, i.e. keys with index j > i (the strict upper triangle).",
          },
          {
            kind: "numeric",
            q: "Two scaled scores in a row are equal and every other key is masked. What weight does each of the two keys receive?",
            answer: 0.5,
            tolerance: 0.001,
            explain: "Masked keys get e^(−∞) = 0; the two equal scores share the mass equally: 1/2 each (row 2 of the worked causal example).",
          },
          {
            q: "In cross-attention between 10 decoder positions and 30 encoder positions, what is the shape of the output?",
            options: ["30 × d_v", "10 × d_v", "10 × 30", "40 × d_v"],
            answer: 1,
            explain: "There is one output row per query, and queries come from the decoder: 10 × d_v. The weight matrix is 10 × 30.",
          },
          {
            kind: "numeric",
            q: "Doubling the sequence length from n to 2n multiplies the cost of computing QKᵀ by what factor?",
            answer: 4,
            tolerance: 0.01,
            explain: "The cost is proportional to n²·d_k, so (2n)² / n² = 4.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Attention by hand for one query",
        prompt: "A query $q=[1,2]$ attends over three keys $k_1=[1,0]$, $k_2=[0,1]$, $k_3=[1,1]$ with values $v_1=[1,0]$, $v_2=[0,1]$, $v_3=[1,1]$, and $d_k=2$. Compute the raw scores, the scaled scores, the softmax weights and the output vector. Give weights to 3 decimals.",
        hint: "Subtract the largest scaled score before exponentiating.",
        solution: [
          { t: "p", text: "Raw scores: $q\\cdot k_1=1$, $q\\cdot k_2=2$, $q\\cdot k_3=3$. Scaled by $1/\\sqrt2$: $[0.7071, 1.4142, 2.1213]$." },
          { t: "p", text: "Subtract the max 2.1213 and exponentiate: $[e^{-1.4142}, e^{-0.7071}, e^0]=[0.2431, 0.4931, 1]$, sum $1.7362$." },
          { t: "p", text: "Weights: $[0.2431, 0.4931, 1]/1.7362=[0.140, 0.284, 0.576]$ (they sum to 1)." },
          { t: "p", text: "Output: $0.140[1,0]+0.284[0,1]+0.576[1,1]=[0.140+0.576,\\;0.284+0.576]=[0.716, 0.860]$." },
          {
            t: "code",
            lang: "python",
            title: "Check",
            code: String.raw`import numpy as np
np.set_printoptions(precision=4, suppress=True)
q = np.array([1., 2.]); K = np.array([[1.,0.],[0.,1.],[1.,1.]]); V = np.array([[1.,0.],[0.,1.],[1.,1.]])
s = K @ q; print("scores", s); ss = s/np.sqrt(2); print("scaled", ss)
e = np.exp(ss - ss.max()); print("exp", e, e.sum()); a = e/e.sum(); print("a", a); print("z", a @ V)`,
            output: `scores [1. 2. 3.]
scaled [0.7071 1.4142 2.1213]
exp [0.2431 0.4931 1.    ] 1.7361854258294542
a [0.14  0.284 0.576]
z [0.716 0.86 ]`,
          },
        ],
      },
      {
        t: "exercise",
        title: "Combine a padding mask with a causal mask",
        prompt: "Two sequences of real lengths 4 and 2 are padded to length 4 in a batch. Build a boolean mask of shape (batch, n, n) that blocks both future keys and padded keys, apply it in attention, and verify that changing the padded positions of the second sequence does not change its real outputs.",
        hint: "Build key_is_real with shape (B, n), add a query axis to get (B, 1, n), and AND it with a (1, n, n) lower-triangular matrix.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "padding_and_causal_mask.py",
            code: String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask):
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    s = np.where(mask, s, -np.inf)
    return softmax(s) @ V, softmax(s)


rng = np.random.default_rng(1)
lengths = np.array([4, 2])                     # real lengths of two sequences padded to 4
B, n, d = 2, 4, 3
X = rng.normal(size=(B, n, d))

key_is_real = np.arange(n)[None, :] < lengths[:, None]   # (B, n): True for real tokens
pad_mask = key_is_real[:, None, :]                         # (B, 1, n) -> broadcasts over queries
causal = np.tril(np.ones((n, n), dtype=bool))[None]        # (1, n, n)
mask = pad_mask & causal                                   # combine with logical AND

out, w = attention(X, X, X, mask)
print("weights for sequence 2 (length 2):\n", w[1])

# Changing the padded positions of sequence 2 must not change its real outputs
X2 = X.copy(); X2[1, 2:] = 99.0
out2, _ = attention(X2, X2, X2, mask)
print("max change in real outputs:", np.abs(out2[1, :2] - out[1, :2]).max())`,
            output: `weights for sequence 2 (length 2):
 [[1.     0.     0.     0.    ]
 [0.3931 0.6069 0.     0.    ]
 [0.6529 0.3471 0.     0.    ]
 [0.4877 0.5123 0.     0.    ]]
max change in real outputs: 0.0`,
          },
          {
            t: "p",
            text: "Columns 2 and 3 (the padded keys) receive zero weight in every row, and the real outputs are unchanged even when the padding is set to 99. Rows 2 and 3 are padded **queries**; they still produce outputs, but a correct loss function ignores them.",
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Attention computes each output as a weighted average of value vectors; the weights come from query–key similarity. It removes the fixed-size bottleneck of RNN encoders and connects any two positions in one step.",
          String.raw`Projections: $Q=XW_Q$, $K=XW_K$, $V=XW_V$ — learned, shared across positions.`,
          String.raw`Core formula: $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\big(QK^\top/\sqrt{d_k}+M\big)V$; softmax is row-wise and each row sums to 1.`,
          String.raw`Scaling: with unit-variance components, $\mathrm{Var}(q\cdot k)=d_k$; dividing by $\sqrt{d_k}$ keeps softmax out of saturation, where its gradient $A_i(\delta_{ij}-A_j)$ vanishes.`,
          String.raw`Masks add $-\infty$ to forbidden scores: padding masks hide padded keys, causal masks hide keys with $j>i$.`,
          "Self-attention: Q, K, V from one sequence. Cross-attention: Q from one sequence, K and V from another.",
          String.raw`Multi-head: $h$ heads of width $d_k=d_{\text{model}}/h$, concatenated and mixed by $W_O$; parameters $4d_{\text{model}}^2+4d_{\text{model}}$, independent of $h$.`,
          String.raw`Cost: $O(n^2d+nd^2)$ time, $O(hn^2)$ memory for the weights — quadratic in sequence length.`,
          "Attention is permutation-equivariant, so position must be added separately.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[Positional Encoding](/learn/m10/positional-encoding) — attention cannot see order on its own; learn the sinusoidal, learned and rotary (RoPE) fixes.",
          "[Multi-Head Attention](/learn/m10/multi-head-attention) — a deeper look at heads: what different heads learn, head pruning, and efficient variants such as grouped-query attention.",
          "[The Transformer Architecture](/learn/m11/transformer-architecture) — wrap attention with residual connections, layer normalisation and feed-forward networks, and stack encoder and decoder blocks.",
          "[Training, Autoregressive Generation and Complexity](/learn/m11/training-inference) — how the causal mask enables teacher forcing, and what the O(n²) cost means for long contexts.",
        ],
      },
    ],
  },
};

export default lesson;
