import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m10/positional-encoding",
  title: "Positional Encoding",
  summary:
    "Why self-attention cannot see word order (with a proof), the sinusoidal encoding and its ladder of frequencies, the rotation trick that makes relative offsets a simple linear map, learned position tables, relative-position methods, rotary position embedding (RoPE) built from 2-D rotations, and ALiBi — all checked with code.",
  minutes: 55,
  level: "intermediate",
  prerequisites: ["m10/scaled-dot-product-attention", "m02/vectors-matrices"],
  next: ["m11/transformer-architecture", "m10/multi-head-attention", "m11/tokens-embeddings"],
  terms: ["positional-encoding"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "A **positional encoding** is extra information added to each token's vector to tell the model **where** the token sits. We need it because self-attention alone treats its input as an unordered bag: “dog bites man” and “man bites dog” hold the same three vectors, so attention gives the same three outputs, just listed in a different order.",
      },
      {
        t: "p",
        text: "RNNs read one token at a time, so order is built in; convolutions see local order through their kernels' layout. The Transformer gave up both to run in parallel, so Vaswani et al. (2017, “Attention Is All You Need”) added a fixed **sinusoidal** vector to each input embedding. Later came learned position tables (BERT, GPT-2), relative-position schemes (Shaw et al., 2018; the T5 bias), **rotary position embedding** (RoPE; Su et al., 2021) in LLaMA-family models, and **ALiBi** (Press et al., 2021) in BLOOM.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The core formula",
        text: String.raw`$PE_{(pos,\,2i)}=\sin\!\big(pos/10000^{2i/d_{\text{model}}}\big),\quad PE_{(pos,\,2i+1)}=\cos\!\big(pos/10000^{2i/d_{\text{model}}}\big)$, and the first layer gets $x_{pos}=e_{pos}+PE_{pos}$ — the token's embedding plus its position code.`,
      },
      {
        t: "p",
        text: "The choice matters: it decides whether a model handles sequences longer than in training, how well it captures relative distance (“two words to the left”), and how cheaply it works with a KV cache.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention): you need $Q=XW_Q$, $K=XW_K$, $V=XW_V$, the row-wise softmax and the score $q_i\\cdot k_j$. Position methods act on $X$ (absolute encodings), on $q$ and $k$ (RoPE), or on the score matrix (relative biases, ALiBi).",
          "[Vectors and matrices](/learn/m02/vectors-matrices): dot products, matrix–vector products, **permutation matrices** (an identity matrix with its rows shuffled) and **rotation matrices** $\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{bmatrix}$, which keep lengths the same and satisfy $R(a)^\\top R(b)=R(b-a)$.",
          "Basic trigonometry: $\\sin(a+b)=\\sin a\\cos b+\\cos a\\sin b$, $\\cos(a+b)=\\cos a\\cos b-\\sin a\\sin b$, and $\\cos(a-b)=\\cos a\\cos b+\\sin a\\sin b$. Every property in this lesson comes from these three.",
          "Optional: complex numbers in polar form, $e^{i\\theta}=\\cos\\theta+i\\sin\\theta$, which give the shortest route to RoPE.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "A wall of clocks",
        text: "Picture a row of clocks whose hands turn at different speeds. The first goes round about every 6 positions; each next hand is slower by the same factor, until the last needs tens of thousands of positions per turn. Reading every hand's angle tells you **where** you are: fast hands tell neighbours apart, slow hands tell you roughly how far into a long document you are. Moving forward k steps turns each hand by its own fixed angle, whatever time it was before — which makes relative positions easy for the model.",
      },
      {
        t: "p",
        text: "A binary counter is similar: the lowest bit flips every step, the next every two, the next every four. Sinusoidal encoding is a smooth version: each dimension pair is a (sin, cos) “bit” at its own frequency, and the frequencies form a geometric sequence, like powers of two.",
      },
      {
        t: "p",
        text: "Why add instead of concatenate (glue on)? Adding keeps the width $d_{\\text{model}}$ the same. With many dimensions, the network can use some directions mostly for content and others mostly for position, so they interfere less than you might fear.",
      },
      {
        t: "p",
        text: "RoPE takes the clocks literally: instead of **adding** a position vector, it **rotates** each query and key by an angle proportional to its position. When a query at m meets a key at n, the rotations partly cancel and only the angle for the gap $m-n$ is left — so scores depend on relative distance by design.",
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      { t: "h", text: "Step 0: the problem — attention is permutation-equivariant" },
      {
        t: "p",
        text: "**Permutation-equivariant** means: shuffle the input rows and the outputs come out shuffled the same way, with the same values. So without position information, a token's output depends on **which** tokens are present, never **where** they are. (Proof in the next section; demo in the code section.)",
      },
      { t: "h", text: "Absolute sinusoidal encoding (Vaswani et al., 2017)" },
      {
        t: "steps",
        items: [
          { title: "Choose d_model frequencies", text: "Group the dimensions in pairs: pair i uses dimensions 2i and 2i+1 and an angular frequency that shrinks by a constant factor each pair.", tex: String.raw`\omega_i = 10000^{-2i/d_{\text{model}}},\qquad i=0,1,\dots,\tfrac{d_{\text{model}}}{2}-1` },
          { title: "Fill the PE matrix", text: "Row pos holds the sin and cos of pos·ωᵢ for every pair. The matrix is fixed: no parameters to learn.", tex: String.raw`PE_{pos,2i}=\sin(pos\,\omega_i),\qquad PE_{pos,2i+1}=\cos(pos\,\omega_i)` },
          { title: "Add to the token embeddings", text: "The original Transformer multiplies embeddings by √d_model, then adds PE; dropout is applied to the sum.", tex: String.raw`x_{pos}=\sqrt{d_{\text{model}}}\;e_{pos}+PE_{pos}` },
          { title: "Run attention as usual", text: "Nothing else changes. Because PE is part of x, it flows into every query, key and value." },
        ],
      },
      { t: "h", text: "Learned absolute embeddings (BERT, GPT-2)" },
      {
        t: "p",
        text: "Swap the fixed matrix for a trainable table $P\\in\\mathbb{R}^{L_{\\max}\\times d_{\\text{model}}}$ — an embedding lookup indexed by position — and add row $p_{pos}$ to the token embedding. BERT uses $L_{\\max}=512$; the smallest GPT-2 uses 1024 positions of width 768 (786,432 parameters). Learned tables work well within the trained length, but there is no row for position $L_{\\max}$ or beyond, so they cannot extrapolate.",
      },
      { t: "h", text: "Relative position methods" },
      {
        t: "p",
        text: "Instead of each token's absolute index, give each **pair** of tokens their offset $j-i$. Shaw, Uszkoreit & Vaswani (2018) add a learned vector $a_{j-i}$ (clipped at a maximum distance) to the key inside the score. T5 simplifies this to a learned **single-number bias** per head for each bucket of offsets, added to the scores. Relative schemes generalise better, because “two tokens to the left” means the same thing anywhere.",
      },
      { t: "h", text: "Rotary position embedding (RoPE; Su et al., 2021)" },
      {
        t: "steps",
        items: [
          { title: "Compute q and k as usual", text: "Nothing is added to the input embeddings." },
          { title: "Rotate each 2-D pair by a position-dependent angle", text: "Pair i of the query at position m is turned by m·θᵢ; the key at position n by n·θᵢ. The θᵢ use the same geometric ladder as the sinusoidal encoding.", tex: String.raw`\tilde q_m = R(m)\,q_m,\quad \tilde k_n = R(n)\,k_n,\qquad \theta_i = 10000^{-2i/d}` },
          { title: "Score the rotated vectors", text: "The two rotations combine into one that depends only on n − m, so the score depends only on content and relative offset.", tex: String.raw`\tilde q_m^\top\tilde k_n = q_m^\top R(n-m)\,k_n` },
          { title: "Values are not rotated", text: "Only queries and keys carry position; values are mixed as usual. RoPE is applied in every layer, not just at the input." },
        ],
      },
      { t: "h", text: "ALiBi (Press, Smith & Lewis, 2021)" },
      {
        t: "p",
        text: "ALiBi (Attention with Linear Biases) uses no position vectors. It subtracts a distance penalty straight from the scores: $s_{ij}=q_i\\cdot k_j/\\sqrt{d_k}-m_h\\,(i-j)$ for a causal model, where each head $h$ has its own fixed slope $m_h$ (for 8 heads: $1/2, 1/4, \\dots, 1/256$). Steep heads focus nearby; gentle heads look far. The paper's title gives the goal: “Train Short, Test Long”.",
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      { t: "h", text: "1. Proof that self-attention is permutation-equivariant" },
      {
        t: "p",
        text: String.raw`Let $X\in\mathbb{R}^{n\times d}$ be the input rows and $P$ an $n\times n$ permutation matrix ($PX$ shuffles the rows; $P^\top P=I$). Write $f(X)=\mathrm{softmax}\big(XW_Q(XW_K)^\top/\sqrt{d_k}\big)\,XW_V$. Then`,
      },
      {
        t: "math",
        tex: String.raw`(PX)W_Q\big((PX)W_K\big)^\top = P\,(QK^\top)\,P^\top .`,
      },
      {
        t: "p",
        text: String.raw`Shuffling rows and columns with the same $P$ only moves entries within rows and reorders rows, which row-wise softmax ignores: $\mathrm{softmax}(PSP^\top)=P\,\mathrm{softmax}(S)\,P^\top$. Hence`,
      },
      {
        t: "math",
        tex: String.raw`f(PX)=P\,A\,P^\top\,P\,V = P\,A\,V = P\,f(X).`,
        caption: "Shuffling the inputs only shuffles the outputs: no information about order gets through.",
      },
      {
        t: "p",
        text: String.raw`With an absolute encoding the input is $X+PE$. Shuffling the **words** gives $PX+PE\neq P(X+PE)$, because the positions stay put. The symmetry is broken — exactly what we want.`,
      },
      { t: "h", text: "2. The sinusoidal encoding, symbol by symbol" },
      {
        t: "math",
        tex: String.raw`PE_{(pos,\,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad PE_{(pos,\,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`,
      },
      {
        t: "list",
        items: [
          String.raw`$pos\in\{0,1,\dots,n-1\}$ — the token's place in the sequence.`,
          String.raw`$i\in\{0,\dots,d_{\text{model}}/2-1\}$ — which dimension **pair**; dimensions $2i$ (sine) and $2i+1$ (cosine) share one frequency.`,
          String.raw`$d_{\text{model}}$ — the embedding width (must be even).`,
          String.raw`$10000$ — the **base**, a setting you choose that fixes the longest wavelength.`,
          String.raw`$\omega_i=10000^{-2i/d_{\text{model}}}$ — angular frequency, in radians per position; wavelength (positions per full turn) $\lambda_i=2\pi/\omega_i=2\pi\cdot10000^{2i/d_{\text{model}}}$.`,
        ],
      },
      { t: "h", text: "3. Why the frequencies are geometric" },
      {
        t: "p",
        text: String.raw`Each frequency is the previous one times a fixed **ratio**, $\omega_{i+1}/\omega_i=10000^{-2/d_{\text{model}}}$. So wavelengths grow geometrically from $\lambda_0=2\pi\approx6.28$ positions to $\lambda_{d/2-1}=2\pi\cdot10000^{(d-2)/d}$ (about 60,611 positions for $d_{\text{model}}=512$). Every **scale** gets the same number of dimensions (each factor of 10 in distance gets the same share), like binary bits. Fast pairs tell neighbours apart but wrap around quickly; slow pairs never wrap within a document but barely change between neighbours. Together they give a unique, smooth code over a very long range. Evenly spaced frequencies would waste most dimensions on similar high frequencies.`,
      },
      { t: "h", text: "4. PE(pos + k) is a rotation of PE(pos)" },
      {
        t: "p",
        text: "Use the angle-addition rules on one pair with frequency $\\omega$:",
      },
      {
        t: "math",
        tex: String.raw`\begin{bmatrix}\sin(\omega(pos+k))\\ \cos(\omega(pos+k))\end{bmatrix}
=\underbrace{\begin{bmatrix}\cos\omega k & \sin\omega k\\ -\sin\omega k & \cos\omega k\end{bmatrix}}_{R_\omega(k)}
\begin{bmatrix}\sin(\omega\,pos)\\ \cos(\omega\,pos)\end{bmatrix}`,
      },
      {
        t: "p",
        text: String.raw`Stacking all pairs, $PE_{pos+k}=R(k)\,PE_{pos}$ with $R(k)=\mathrm{diag}\big(R_{\omega_0}(k),\dots,R_{\omega_{d/2-1}}(k)\big)$, a block-diagonal **orthogonal** matrix (a pure rotation) that depends on $k$ but **not** on $pos$. In plain words: "the token $k$ places later" is one fixed linear map of the current code, which a linear layer like $W_Q$ or $W_K$ can learn to use. Vaswani et al. gave this as their reason for sinusoids.`,
      },
      { t: "h", text: "5. Dot products depend only on the offset" },
      {
        t: "math",
        tex: String.raw`PE_p\cdot PE_q=\sum_{i}\big[\sin(\omega_ip)\sin(\omega_iq)+\cos(\omega_ip)\cos(\omega_iq)\big]=\sum_{i=0}^{d/2-1}\cos\big(\omega_i(p-q)\big).`,
        caption: "A position's dot product with itself is always d_model/2. The value is the same for p − q and q − p, so the dot product alone cannot tell 'before' from 'after'.",
      },
      { t: "h", text: "6. Worked example: d_model = 4, positions 0 to 3" },
      {
        t: "p",
        text: String.raw`With $d_{\text{model}}=4$ there are two pairs: $\omega_0=10000^{0}=1$ and $\omega_1=10000^{-2/4}=0.01$. So $PE_{pos}=[\sin(pos),\cos(pos),\sin(0.01\,pos),\cos(0.01\,pos)]$:`,
      },
      {
        t: "math",
        tex: String.raw`PE=\begin{bmatrix}0&1&0&1\\0.8415&0.5403&0.0100&1.0000\\0.9093&-0.4161&0.0200&0.9998\\0.1411&-0.9900&0.0300&0.9996\end{bmatrix}\begin{matrix}\leftarrow pos=0\\ \leftarrow pos=1\\ \leftarrow pos=2\\ \leftarrow pos=3\end{matrix}`,
      },
      {
        t: "p",
        text: String.raw`**Rotation check** ($k=1$): $R(1)$ has blocks $\begin{bmatrix}0.5403&0.8415\\-0.8415&0.5403\end{bmatrix}$ and $\begin{bmatrix}1.0000&0.0100\\-0.0100&1.0000\end{bmatrix}$. Multiply $R(1)\,PE_1$: first entry $0.5403\cdot0.8415+0.8415\cdot0.5403=0.9093$, second $-0.8415\cdot0.8415+0.5403\cdot0.5403=-0.4161$; the slow pair gives $[0.0200, 0.9998]$. That is exactly $PE_2$.`,
      },
      {
        t: "p",
        text: String.raw`**Dot products**: the Gram matrix $PE\,PE^\top$ (all pairwise dot products) is constant along each diagonal — a Toeplitz matrix — as predicted:`,
      },
      {
        t: "math",
        tex: String.raw`PE\,PE^\top=\begin{bmatrix}2&1.5403&0.5837&0.0096\\1.5403&2&1.5403&0.5837\\0.5837&1.5403&2&1.5403\\0.0096&0.5837&1.5403&2\end{bmatrix}`,
        caption: "E.g. offset 2: cos(2) + cos(0.02) = −0.4161 + 0.9998 = 0.5837, wherever the pair sits.",
      },
      { t: "h", text: "7. Learned and relative encodings" },
      {
        t: "math",
        tex: String.raw`\text{learned absolute: } x_{pos}=e_{pos}+p_{pos},\ p_{pos}\in\mathbb{R}^{d}\ \text{trained};\qquad
\text{relative bias (T5-style): } s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}+b_{h}\big(\mathrm{bucket}(j-i)\big)`,
      },
      {
        t: "math",
        tex: String.raw`\text{Shaw et al. (2018): } s_{ij}=\frac{q_i\cdot\big(k_j+a^K_{\mathrm{clip}(j-i,\,-r,\,r)}\big)}{\sqrt{d_k}}`,
        caption: "a^K are learned vectors, one per clipped offset; r is the largest distance stored exactly.",
      },
      { t: "h", text: "8. RoPE built from 2-D rotations" },
      {
        t: "p",
        text: String.raw`Take $d=2$. We want functions $f_q(q,m)$ and $f_k(k,n)$ that add the positions $m,n$ so that $\langle f_q(q,m), f_k(k,n)\rangle$ depends only on $q$, $k$ and $m-n$. Rotations do exactly this. With $R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}$, set $f_q(q,m)=R(m\theta)q$ and $f_k(k,n)=R(n\theta)k$. Because $R(a)^\top=R(-a)$ and $R(a)R(b)=R(a+b)$:`,
      },
      {
        t: "math",
        tex: String.raw`\big(R(m\theta)q\big)^\top R(n\theta)k = q^\top R(-m\theta)R(n\theta)\,k = q^\top R\big((n-m)\theta\big)\,k.`,
      },
      {
        t: "p",
        text: String.raw`In complex numbers: treat $q=(q_1,q_2)$ as $q_1+iq_2$. Rotating by $m\theta$ is multiplying by $e^{im\theta}$, and the real dot product is $\mathrm{Re}[\tilde q\,\overline{\tilde k}]=\mathrm{Re}[q\bar k\,e^{i(m-n)\theta}]$. For $d>2$, split the vector into $d/2$ pairs and rotate pair $i$ by $m\theta_i$ with $\theta_i=10000^{-2i/d}$:`,
      },
      {
        t: "math",
        tex: String.raw`\begin{bmatrix}\tilde q_{2i}\\ \tilde q_{2i+1}\end{bmatrix}=\begin{bmatrix}\cos m\theta_i&-\sin m\theta_i\\ \sin m\theta_i&\cos m\theta_i\end{bmatrix}\begin{bmatrix}q_{2i}\\ q_{2i+1}\end{bmatrix},\qquad \tilde q_m^\top\tilde k_n=\sum_i \big[q^{(i)}\big]^\top R\big((n-m)\theta_i\big)\,k^{(i)}`,
      },
      {
        t: "p",
        text: "Rotations keep lengths, so RoPE does not change the norms of q and k. Unlike sinusoidal PE, position **multiplies** the content instead of being added, so the score cannot split into a content part plus a position part: position changes how content is compared.",
      },
      { t: "h", text: "9. ALiBi" },
      {
        t: "math",
        tex: String.raw`s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}-m_h\,(i-j)\quad (j\le i),\qquad m_h=2^{-8h/H},\ h=1,\dots,H`,
        caption: "For H = 8 heads the slopes are 1/2, 1/4, …, 1/256. The penalty grows in a straight line with distance and has no parameters.",
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "p",
        text: "The simulator draws the full sinusoidal PE matrix (positions left to right, dimensions top to bottom), one dimension as a wave across positions, and the dot product of one position with all the others.",
      },
      { t: "sim", id: "positional-encoding", caption: "Teal = positive, clay = negative. Click the heatmap to pick a position and dimension." },
      { t: "h", text: "What to try" },
      {
        t: "list",
        ordered: true,
        items: [
          "Scan the heatmap top to bottom: top rows (low dimensions) flicker fast across positions, bottom rows are almost flat. That is the geometric ladder.",
          "Drag **dimension j** up from 0 and watch the wave in the middle plot stretch; the side panel shows its wavelength growing from 6.3 positions by a fixed factor per pair.",
          "Press **Play** to sweep the position p. In the bottom plot the similarity curve slides with p but keeps the **same shape**: PE(p)·PE(q) depends only on q − p. The dashed formula curve Σcos(ωᵢ(q−p)) sits exactly on top.",
          "Compare PE(p)·PE(p+3) and PE(p)·PE(p−3) in the side panel: they are equal. The dot product is symmetric and cannot tell left from right on its own.",
          "Lower **d_model** to 4 or 8: the curve becomes a sum of 2–4 cosines with big side peaks, so distant positions can look alike. At 128 the peak at q = p is sharp and dominant.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "This NumPy program checks every claim: attention without positions is permutation-equivariant; sinusoidal PE breaks the symmetry; PE(pos + k) is a fixed rotation of PE(pos); dot products depend only on the offset; RoPE scores depend only on m − n and keep norms.",
      },
      {
        t: "code",
        lang: "python",
        title: "positional_encoding_numpy.py",
        runnable: true,
        code: String.raw`import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)


def sinusoidal_pe(n_positions, d_model, base=10000.0):
    """PE[pos, 2i] = sin(pos * w_i), PE[pos, 2i+1] = cos(pos * w_i), w_i = base^(-2i/d_model)."""
    assert d_model % 2 == 0
    pos = np.arange(n_positions)[:, None]                  # (n, 1)
    omega = base ** (-np.arange(0, d_model, 2) / d_model)  # (d/2,) geometric frequencies
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe


def self_attention(X, W_Q, W_K, W_V):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    s = Q @ K.T / np.sqrt(Q.shape[1])
    a = np.exp(s - s.max(axis=1, keepdims=True))
    return (a / a.sum(axis=1, keepdims=True)) @ V


# 1) Attention alone is permutation-equivariant
n, d = 6, 8
X = rng.normal(size=(n, d))
W_Q, W_K, W_V = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
perm = rng.permutation(n)
out = self_attention(X, W_Q, W_K, W_V)
out_perm = self_attention(X[perm], W_Q, W_K, W_V)
print("permutation:", perm)
print("no PE   : f(PX) == P f(X)?", np.allclose(out_perm, out[perm]))

# Adding positional encodings breaks the symmetry (position is attached AFTER permuting the words)
PE = sinusoidal_pe(n, d)
out_pe = self_attention(X + PE, W_Q, W_K, W_V)
out_pe_perm = self_attention(X[perm] + PE, W_Q, W_K, W_V)
print("with PE : f(PX+PE) == P f(X+PE)?", np.allclose(out_pe_perm, out_pe[perm]))

# 2) PE(pos + k) is a fixed rotation of PE(pos)
d_model, k = 16, 5
P = sinusoidal_pe(50, d_model)
omega = 10000.0 ** (-np.arange(0, d_model, 2) / d_model)
R = np.zeros((d_model, d_model))
for i, w in enumerate(omega):
    c, s = np.cos(w * k), np.sin(w * k)
    R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, s], [-s, c]]   # acts on (sin, cos) pairs
err = max(np.abs(R @ P[pos] - P[pos + k]).max() for pos in range(45))
print("\nmax |R(k) PE(pos) - PE(pos+k)| over 45 positions:", f"{err:.1e}")
print("R(k) orthogonal:", np.allclose(R.T @ R, np.eye(d_model)))

# 3) The dot product depends only on the offset
for p, q in [(3, 8), (20, 25), (40, 45), (25, 20)]:
    print(f"PE({p:2d}).PE({q:2d}) = {P[p] @ P[q]:.4f}")
print("closed form sum_i cos(5 w_i) =", round(float(np.cos(omega * 5).sum()), 4))

# 4) RoPE: rotate q and k by their positions; the score depends only on m - n
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = pos * theta
    x1, x2 = x[0::2], x[1::2]                       # pairs (x_{2i}, x_{2i+1})
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out

q, kv = rng.normal(size=d_model), rng.normal(size=d_model)
for m, nn_ in [(7, 3), (107, 103), (1007, 1003)]:
    print(f"RoPE score q@{m:4d}, k@{nn_:4d} (offset {m - nn_}): {rope(q, m) @ rope(kv, nn_):.6f}")
print("norm preserved:", np.isclose(np.linalg.norm(rope(q, 123)), np.linalg.norm(q)))`,
        output: `permutation: [5 3 2 0 1 4]
no PE   : f(PX) == P f(X)? True
with PE : f(PX+PE) == P f(X+PE)? False

max |R(k) PE(pos) - PE(pos+k)| over 45 positions: 1.1e-15
R(k) orthogonal: True
PE( 3).PE( 8) = 6.1370
PE(20).PE(25) = 6.1370
PE(40).PE(45) = 6.1370
PE(25).PE(20) = 6.1370
closed form sum_i cos(5 w_i) = 6.137
RoPE score q@   7, k@   3 (offset 4): 5.778393
RoPE score q@ 107, k@ 103 (offset 4): 5.778393
RoPE score q@1007, k@1003 (offset 4): 5.778393
norm preserved: True`,
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
                text: "PyTorch has no built-in positional-encoding layer, so models write their own: the sinusoidal table via `register_buffer` (saved and moved to the GPU with the model, not trained), learned positions as an `nn.Embedding`, RoPE on queries and keys just before attention, and ALiBi as an added bias passed as a float `attn_mask`.",
              },
              {
                t: "code",
                lang: "python",
                title: "positional_encodings_torch.py",
                code: String.raw`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) encoding, stored as a buffer: moves with .to(device), not trained."""
    def __init__(self, d_model, max_len=2048, base=10000.0):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)                                  # (L, 1)
        omega = torch.exp(-math.log(base) * torch.arange(0, d_model, 2) / d_model)  # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * omega)
        pe[:, 1::2] = torch.cos(pos * omega)
        self.register_buffer("pe", pe)

    def forward(self, x):                      # x: (batch, n, d_model)
        return x + self.pe[: x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    """BERT / GPT-2 style: one trainable vector per absolute position."""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos(positions)


def apply_rope(x, base=10000.0):
    """x: (batch, heads, n, d_head). Rotates each pair (x_2i, x_2i+1) by angle pos * theta_i."""
    n, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)      # (d/2,)
    ang = torch.arange(n, dtype=torch.float32)[:, None] * theta            # (n, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def alibi_bias(n, num_heads):
    """ALiBi: add -slope * (i - j) to the score of query i and key j (causal setting)."""
    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / num_heads) for h in range(num_heads)])
    rel = torch.arange(n)[None, :] - torch.arange(n)[:, None]             # j - i (<= 0 for past keys)
    return slopes[:, None, None] * rel.clamp(max=0)                        # (heads, n, n); future keys get the causal mask


x = torch.randn(2, 10, 32)                                                 # (batch, n, d_model)
sin_pe = SinusoidalPositionalEncoding(32)
learned = LearnedPositionalEmbedding(32, max_len=512)
print("sinusoidal out:", tuple(sin_pe(x).shape), "| trainable params:", sum(p.numel() for p in sin_pe.parameters()))
print("learned    out:", tuple(learned(x).shape), "| trainable params:", sum(p.numel() for p in learned.parameters()))
print("GPT-2 small position table (1024 x 768):", 1024 * 768, "parameters")

# RoPE inside attention: the same content at a shifted location gives the same scores
B, H, n, dh = 1, 2, 8, 16
q_vec, k_vec = torch.randn(dh), torch.randn(dh)
Q = torch.zeros(B, H, n, dh); K = torch.zeros(B, H, n, dh)
Q[..., 5, :] = q_vec; K[..., 2, :] = k_vec          # query at 5, key at 2 (offset 3)
Q2 = torch.zeros_like(Q); K2 = torch.zeros_like(K)
Q2[..., 7, :] = q_vec; K2[..., 4, :] = k_vec        # query at 7, key at 4 (offset 3)
s1 = apply_rope(Q)[0, 0, 5] @ apply_rope(K)[0, 0, 2]
s2 = apply_rope(Q2)[0, 0, 7] @ apply_rope(K2)[0, 0, 4]
print(f"RoPE score offset 3 at (5,2): {s1.item():.5f}   at (7,4): {s2.item():.5f}")

q, k, v = (torch.randn(B, H, n, dh) for _ in range(3))
out = F.scaled_dot_product_attention(apply_rope(q), apply_rope(k), v, is_causal=True)
print("attention with RoPE:", tuple(out.shape))

bias = alibi_bias(5, num_heads=8)
print("ALiBi slopes:", [round(2.0 ** (-8.0 * (h + 1) / 8), 4) for h in range(8)])
print("ALiBi bias, head 0:\n", bias[0])`,
                output: `sinusoidal out: (2, 10, 32) | trainable params: 0
learned    out: (2, 10, 32) | trainable params: 16384
GPT-2 small position table (1024 x 768): 786432 parameters
RoPE score offset 3 at (5,2): 0.48440   at (7,4): 0.48440
attention with RoPE: (1, 2, 8, 16)
ALiBi slopes: [0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039]
ALiBi bias, head 0:
 tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.5000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.0000, -0.5000,  0.0000,  0.0000,  0.0000],
        [-1.5000, -1.0000, -0.5000,  0.0000,  0.0000],
        [-2.0000, -1.5000, -1.0000, -0.5000,  0.0000]])`,
              },
              {
                t: "p",
                text: "To use the ALiBi bias with `F.scaled_dot_product_attention`, add the causal $-\\infty$ mask to it and pass the result as a float `attn_mask` of shape (heads, n, n); float masks are **added** to the scaled scores.",
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "In Keras, a learned position embedding is an `Embedding` layer indexed by `tf.range(n)`; the sinusoidal table is a constant built once in the layer's constructor.",
              },
              {
                t: "code",
                lang: "python",
                title: "keras_positional.py (not executed in CI)",
                code: String.raw`import numpy as np
import tensorflow as tf
from tensorflow import keras


def sinusoidal_table(max_len, d_model, base=10000.0):
    pos = np.arange(max_len)[:, None]
    omega = base ** (-np.arange(0, d_model, 2) / d_model)
    pe = np.zeros((max_len, d_model), dtype="float32")
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return tf.constant(pe)


class TokenAndPosition(keras.layers.Layer):
    """Token embedding + (sinusoidal or learned) position embedding."""
    def __init__(self, vocab_size, max_len, d_model, learned=False, **kwargs):
        super().__init__(**kwargs)
        self.tok = keras.layers.Embedding(vocab_size, d_model)
        self.learned = learned
        if learned:
            self.pos = keras.layers.Embedding(max_len, d_model)
        else:
            self.table = sinusoidal_table(max_len, d_model)

    def call(self, token_ids):
        n = tf.shape(token_ids)[-1]
        positions = self.pos(tf.range(n)) if self.learned else self.table[:n]
        return self.tok(token_ids) + positions


ids = tf.constant([[5, 17, 2, 9, 9, 3]])
for learned in (False, True):
    layer = TokenAndPosition(vocab_size=100, max_len=64, d_model=16, learned=learned)
    x = layer(ids)
    print("learned" if learned else "sinusoidal", x.shape, "params:", layer.count_params())
    # the two copies of token 9 (positions 3 and 4) now have different vectors
    print("  rows 3 and 4 differ:", bool(tf.reduce_any(tf.abs(x[0, 3] - x[0, 4]) > 1e-6)))`,
                output: "Prints the output shape (1, 6, 16) for both variants; the parameter count 1600 (token table only, 100 × 16) for sinusoidal and 2624 (1600 + 64 × 16 position table) for learned; and True for both, showing that the repeated token 9 receives different vectors at positions 3 and 4.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Method", "Where position enters", "Advantages", "Disadvantages", "Used in"],
        rows: [
          ["Sinusoidal absolute (Vaswani et al., 2017)", "Added to input embeddings", "No parameters; defined for any position; offsets are linear maps", "Works poorly past the trained length in practice; position mixed into content", "Original Transformer"],
          ["Learned absolute", "Trainable table added to inputs", "Simple, flexible, strong within range", "Hard limit L_max; rarely seen late positions get little training", "BERT, GPT-2, ViT (per patch)"],
          ["Relative vectors (Shaw et al., 2018)", "Learned vector per clipped offset inside the score", "Same meaning at every position; generalises well", "Extra memory per (i, j) pair; more complex kernels", "Music Transformer, early relative models"],
          ["Relative scalar bias (T5)", "Learned bias per head and offset bucket, added to scores", "Cheap; log-spaced buckets cover long distances", "Bias must be computed for each (i, j); does not fit every fused kernel", "T5"],
          ["RoPE (Su et al., 2021)", "Rotates q and k in every layer", "Exactly relative; no parameters; works with KV cache; keeps norms", "Longer contexts need tricks (position interpolation, base scaling)", "LLaMA-family and many open LLMs"],
          ["ALiBi (Press et al., 2021)", "Distance penalty added to scores", "No parameters; handles longer inputs well", "Built-in preference for recent tokens; slopes are fixed", "BLOOM, MPT"],
          ["No positional encoding (causal)", "None; the causal mask leaks order indirectly", "Simplest", "Only for causal models; usually weaker", "Research ablations"],
        ],
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Language models**: every Transformer language model needs a position method. GPT-2 used learned absolute embeddings; LLaMA-family models use RoPE; BLOOM uses ALiBi.",
          "**Bidirectional encoders**: BERT adds learned embeddings for 512 positions — that is why its standard input limit is 512 tokens.",
          "**Vision Transformers**: each image patch gets a learned embedding for its spot in the patch grid; for fine-tuning at higher resolution, that grid is resized by 2-D interpolation.",
          "**Long-context extension**: models trained with RoPE are often stretched to longer contexts by rescaling positions (position interpolation) or the base frequency, then briefly fine-tuned.",
          "**Beyond text**: sinusoidal codes for continuous values (diffusion time steps, coordinates in neural radiance fields) use the same many-frequency idea to capture fine detail.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      { t: "h", text: "Strengths of each approach" },
      {
        t: "list",
        items: [
          "**Sinusoidal**: no parameters, always the same, defined for every whole-number position, and offsets act as fixed rotations.",
          "**Learned absolute**: the model picks whatever position code helps the task; very little code to write.",
          "**RoPE**: relative position enters the score exactly; applied in every layer, so depth does not wash it out; cached keys never need recomputing.",
          "**ALiBi**: almost free to compute and copes well with longer inputs.",
        ],
      },
      { t: "h", text: "Limitations and trade-offs" },
      {
        t: "list",
        items: [
          String.raw`**Extrapolation**: learned tables stop at $L_{\max}$. Sinusoidal and RoPE are defined beyond the training length, but models still get worse there because they never saw those angle combinations.`,
          "**Additive encodings mix with content**: after adding PE, the score $(e_i+p_i)^\\top W_QW_K^\\top(e_j+p_j)$ has content–content, content–position and position–position terms; some are just noise for a given task.",
          "**Symmetry**: the sinusoidal dot product $\\sum\\cos(\\omega_i(p-q))$ is the same for $p-q$ and $q-p$; direction (before vs after) must be recovered by the learned projections.",
          "**Recency bias**: ALiBi always penalises far tokens, which can hurt tasks that need exact long-range lookup.",
          "**Engineering**: relative biases must be built for every (i, j) pair, which clashes with some fused attention kernels; RoPE and absolute encodings avoid this.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Mixing up the sin/cos layout.** Some code puts all sines first and all cosines second; other code interleaves (sin, cos, sin, cos). Both are valid, but loading weights trained with one into code using the other silently breaks the model. Same for RoPE's “interleaved” vs “rotate-half” pairing.",
          "**Using the dimension index instead of the pair index.** The exponent is $2i/d_{\\text{model}}$ where $i$ is the pair index. Writing $j/d_{\\text{model}}$ with the raw dimension $j$ gives the sin and cos of a pair different frequencies, which ruins the rotation property.",
          "**Registering the sinusoidal table as a parameter.** `nn.Parameter` makes it trainable and hands it to the optimizer; use `register_buffer` for a fixed table.",
          "**Adding positions after the attention layer**, or only to the values. Positions must affect the queries and keys (the scores) for attention to see order: put them in the input embedding (absolute) or in q and k (RoPE).",
          "**Applying RoPE to the values.** Only queries and keys are rotated; rotating values changes the output without giving the relative-position property.",
          "**Wrong offsets during cached generation.** When generating one token at a time with a KV cache, the new token's position is the cache length, not 0. Forget this and every new token gets position 0.",
          "**Going past the learned table.** Feeding 600 tokens to a model with 512 learned positions raises an index error or, if clipped, gives garbage; truncate or use a sliding window.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      {
        t: "qa",
        level: "beginner",
        q: "Why does a Transformer need positional encoding when an RNN does not?",
        a: "An RNN reads tokens one by one, so order is part of the computation. Self-attention gives the same result for a set of vectors whatever their order — shuffling inputs only shuffles outputs. Without position information, “dog bites man” and “man bites dog” would look the same apart from output order.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "Write down the sinusoidal positional encoding and explain its variables.",
        a: "$PE_{(pos,2i)}=\\sin(pos/10000^{2i/d})$ and $PE_{(pos,2i+1)}=\\cos(pos/10000^{2i/d})$. Here pos is the token index, i numbers the dimension pairs (0 to d/2 − 1), d is the model width and 10000 is the base. Each pair is a sine/cosine at frequency $10000^{-2i/d}$; the result is added to the token embedding.",
      },
      {
        t: "qa",
        level: "beginner",
        q: "What is the difference between sinusoidal and learned absolute position embeddings?",
        a: "Sinusoidal encodings are a fixed formula with no parameters, defined for every position. Learned embeddings are a trainable table, one vector per position up to a maximum (512 in BERT, 1024 in GPT-2): they adapt to the data, but positions past the table have no vector. In the original Transformer paper both did almost equally well on translation.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Show that PE(pos + k) is a linear function of PE(pos).",
        a: "For one pair with frequency ω: $\\sin(\\omega(pos+k))=\\sin(\\omega pos)\\cos(\\omega k)+\\cos(\\omega pos)\\sin(\\omega k)$ and $\\cos(\\omega(pos+k))=\\cos(\\omega pos)\\cos(\\omega k)-\\sin(\\omega pos)\\sin(\\omega k)$. As a matrix, this is a 2 × 2 rotation by $\\omega k$ applied to $(\\sin\\omega pos,\\cos\\omega pos)$. Stacking pairs gives a block-diagonal orthogonal matrix R(k) that depends only on k, so $PE_{pos+k}=R(k)PE_{pos}$.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "Why are the frequencies geometric rather than linear?",
        a: "A geometric sequence covers many scales evenly: wavelengths run from 2π to about 2π·10000, and each factor of distance gets the same number of dimensions, like a binary counter's bits. Fast pairs tell neighbours apart but repeat quickly; slow pairs never repeat within a document but change little between neighbours. Evenly spaced frequencies would give mostly similar short wavelengths, capturing long-range position poorly.",
      },
      {
        t: "qa",
        level: "intermediate",
        q: "How does RoPE make attention scores depend only on relative position?",
        a: "It rotates each 2-D pair of the query at position m by $m\\theta_i$, and of the key at n by $n\\theta_i$. Because $R(a)^\\top R(b)=R(b-a)$, the score becomes $q^\\top R((n-m)\\theta)k$: absolute angles cancel, only the offset remains. No parameters, norms unchanged, and since each key is rotated once by its own position, cached keys stay valid during generation.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Prove that self-attention without positional information is permutation-equivariant.",
        a: "For a permutation matrix P: $Q'=PXW_Q=PQ$, $K'=PK$, $V'=PV$. Then $Q'K'^\\top=PQK^\\top P^\\top$, and since row-wise softmax is unaffected by shuffling rows and columns together, $A'=PAP^\\top$. The output is $A'V'=PAP^\\top PV=PAV$ because $P^\\top P=I$. Hence $f(PX)=Pf(X)$.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "A model trained with RoPE on 4k-token contexts fails at 16k tokens. Why, and what are common fixes?",
        a: "At 16k tokens the slow pairs reach rotation angles never seen in training, so the learned projections meet unfamiliar relative phases and attention degrades. Common fixes: position interpolation (divide positions by 4 so angles stay in the trained range, then fine-tune briefly); a bigger RoPE base so frequencies are lower (NTK-aware or base scaling); or scaling different frequency bands differently. ALiBi-style biases are an alternative built to extrapolate without such changes.",
      },
      {
        t: "qa",
        level: "advanced",
        q: "Expand the attention score with absolute additive encodings and interpret the terms.",
        a: "With $x_i=e_i+p_i$: $q_i^\\top k_j=(e_i+p_i)^\\top W_QW_K^\\top(e_j+p_j)=e_i^\\top Ue_j+e_i^\\top Up_j+p_i^\\top Ue_j+p_i^\\top Up_j$ with $U=W_QW_K^\\top$. The first term is content–content, the middle two are content–position cross terms, and the last is position–position. Relative-position methods (Transformer-XL's split, T5 biases, RoPE) reshape or replace these terms so that position enters only through $j-i$.",
      },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "If you shuffle the input tokens of a self-attention layer that has no position information, what happens to the output?",
            options: ["It is unchanged", "It is shuffled in the same way", "It becomes random", "It becomes all zeros"],
            answer: 1,
            explain: "Self-attention is permutation-equivariant: f(PX) = P f(X).",
          },
          {
            kind: "numeric",
            q: "With d_model = 512, what is PE(p)·PE(p) (a sinusoidal encoding's dot product with itself)?",
            answer: 256,
            tolerance: 0.001,
            explain: "Each pair gives sin² + cos² = 1, and there are d_model/2 = 256 pairs.",
          },
          {
            kind: "numeric",
            q: "What is the wavelength, in positions, of the fastest pair (i = 0) of the sinusoidal encoding? (2 decimals)",
            answer: 6.28,
            tolerance: 0.01,
            explain: "ω₀ = 10000⁰ = 1 radian per position, so one full turn takes 2π ≈ 6.28 positions.",
          },
          {
            q: "Which statement about RoPE is correct?",
            options: [
              "It adds a learned vector to the input embeddings",
              "It rotates queries and keys so their dot product depends on the relative offset",
              "It rotates the values as well",
              "It subtracts a linear distance penalty from the scores",
            ],
            answer: 1,
            explain: "RoPE rotates q and k by angles set by their positions; R(m)ᵀR(n) = R(n − m). The linear penalty is ALiBi.",
          },
          {
            kind: "numeric",
            q: "With d_model = 4 (ω₀ = 1, ω₁ = 0.01), compute PE(1)·PE(3) to 4 decimals.",
            answer: 0.5837,
            tolerance: 0.0005,
            explain: "PE(p)·PE(q) = cos(1·2) + cos(0.01·2) = −0.4161 + 0.9998 = 0.5837.",
          },
          {
            q: "Why can a model with learned absolute position embeddings for 512 positions not process 1000 tokens directly?",
            options: [
              "Softmax overflows",
              "There is no embedding vector for positions 512 to 999",
              "Attention cost becomes cubic",
              "The vocabulary is too small",
            ],
            answer: 1,
            explain: "The learned table has exactly 512 rows; later positions have no vector.",
          },
          {
            kind: "numeric",
            q: "With ALiBi and 8 heads, slopes are 2^(−h) for h = 1…8. What bias does head 2 add to the score of a query at position 10 and a key at position 6?",
            answer: -1,
            tolerance: 0.001,
            explain: "Head 2's slope is 1/4; distance i − j = 4; bias = −(1/4)·4 = −1.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Rotation and similarity by hand",
        prompt: "Let $d_{\\text{model}}=4$. (a) Write $PE_1$ and $PE_3$. (b) Compute $PE_1\\cdot PE_3$ directly and with the formula $\\sum_i\\cos(\\omega_i(p-q))$. (c) Which rotation $R(k)$ turns $PE_1$ into $PE_3$, and what are its angles?",
        hint: "ω₀ = 1, ω₁ = 0.01.",
        solution: [
          { t: "p", text: "(a) $PE_1=[\\sin1,\\cos1,\\sin0.01,\\cos0.01]=[0.8415, 0.5403, 0.0100, 1.0000]$ and $PE_3=[\\sin3,\\cos3,\\sin0.03,\\cos0.03]=[0.1411,-0.9900,0.0300,0.9996]$." },
          { t: "p", text: "(b) Directly: $0.8415\\cdot0.1411+0.5403\\cdot(-0.9900)+0.0100\\cdot0.0300+1.0000\\cdot0.9996=0.1187-0.5349+0.0003+0.9996=0.5837$. Formula: $\\cos(2)+\\cos(0.02)=-0.4161+0.9998=0.5837$." },
          { t: "p", text: "(c) The offset is $k=2$, so $R(2)$ has blocks turning by $\\omega_0\\cdot2=2$ radians and $\\omega_1\\cdot2=0.02$ radians: $\\begin{bmatrix}\\cos2&\\sin2\\\\-\\sin2&\\cos2\\end{bmatrix}$ and $\\begin{bmatrix}\\cos0.02&\\sin0.02\\\\-\\sin0.02&\\cos0.02\\end{bmatrix}$. The same R(2) maps every $PE_p$ to $PE_{p+2}$." },
        ],
      },
      {
        t: "exercise",
        title: "Verify that the similarity matrix is Toeplitz",
        prompt: "Build the sinusoidal PE for 64 positions with $d_{\\text{model}}=64$, compute the Gram matrix $G=PE\\,PE^\\top$, and check numerically that every diagonal (fixed offset $k=q-p$) is constant and that $G$ is symmetric in $k$. Print $PE_p\\cdot PE_{p+k}$ for k = 0, 1, 2, 4, 8, 16, 32 and describe how it changes.",
        hint: "np.diagonal(G, offset=k) returns the k-th diagonal; np.ptp gives max − min.",
        solution: [
          {
            t: "code",
            lang: "python",
            title: "toeplitz_check.py",
            code: String.raw`import numpy as np


def sinusoidal_pe(n_positions, d_model, base=10000.0):
    pos = np.arange(n_positions)[:, None]
    omega = base ** (-np.arange(0, d_model, 2) / d_model)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe


P = sinusoidal_pe(64, 64)
G = P @ P.T                                   # Gram matrix of all pairwise dot products

# Toeplitz check: every diagonal (fixed offset k = q - p) holds a single value
spread = max(np.ptp(np.diagonal(G, offset=k)) for k in range(-63, 64))
print("largest spread along any diagonal:", f"{spread:.1e}")
print("symmetric in k:", np.allclose(np.diagonal(G, 7), np.diagonal(G, -7)))
for k in [0, 1, 2, 4, 8, 16, 32]:
    print(f"offset {k:2d}: PE(p).PE(p+k) = {G[0, k]:7.3f}")`,
            output: `largest spread along any diagonal: 3.2e-14
symmetric in k: True
offset  0: PE(p).PE(p+k) =  32.000
offset  1: PE(p).PE(p+k) =  30.917
offset  2: PE(p).PE(p+k) =  28.304
offset  4: PE(p).PE(p+k) =  23.934
offset  8: PE(p).PE(p+k) =  22.414
offset 16: PE(p).PE(p+k) =  19.373
offset 32: PE(p).PE(p+k) =  19.554`,
          },
          {
            t: "p",
            text: "Each diagonal is constant to within $3\\times10^{-14}$ (rounding), and offsets $+7$ and $-7$ match. Similarity starts at $d/2=32$ and mostly falls with distance, but not steadily (offset 32 is a bit above offset 16), because the cosines wiggle. So the dot product is a rough distance signal, and being symmetric, it says nothing about direction.",
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          String.raw`Self-attention is permutation-equivariant, $f(PX)=Pf(X)$, so order must be added on purpose.`,
          String.raw`Sinusoidal PE: $PE_{pos,2i}=\sin(pos\,\omega_i)$, $PE_{pos,2i+1}=\cos(pos\,\omega_i)$, $\omega_i=10000^{-2i/d_{\text{model}}}$; added to embeddings; no parameters.`,
          String.raw`Frequencies are geometric: wavelengths from $2\pi$ to about $2\pi\cdot10000$, covering every distance scale evenly, like the bits of a counter.`,
          String.raw`Relative property: $PE_{pos+k}=R(k)\,PE_{pos}$, with a block-diagonal rotation that depends only on $k$.`,
          String.raw`$PE_p\cdot PE_q=\sum_i\cos(\omega_i(p-q))$ — depends only on the offset, is symmetric, and equals $d_{\text{model}}/2$ at offset 0.`,
          "Learned absolute embeddings (BERT, GPT-2): flexible but capped at the table length.",
          "Relative methods (Shaw et al., T5 bias) put the offset j − i straight into the scores.",
          String.raw`RoPE rotates $q$ and $k$: $(R(m\theta)q)^\top R(n\theta)k=q^\top R((n-m)\theta)k$; no parameters, keeps norms, cache-friendly.`,
          String.raw`ALiBi subtracts $m_h(i-j)$ from scores with fixed geometric slopes, trading a recency bias for handling longer inputs well.`,
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[The Transformer Architecture](/learn/m11/transformer-architecture) — see where the positional encoding enters the full encoder–decoder stack, next to residual connections, LayerNorm and feed-forward layers.",
          "[Multi-Head Attention](/learn/m10/multi-head-attention) — RoPE and ALiBi work per head; learn how heads are split and what they specialise in.",
          "[Tokenization, Embeddings and Context Windows](/learn/m11/tokens-embeddings) — how token embeddings are made, and how the position method limits the context window.",
          "[Scaled Dot-Product Attention](/learn/m10/scaled-dot-product-attention) — revisit the score matrix with positions in mind, and try the attention explorer with positional encoding on and off.",
        ],
      },
    ],
  },
};

export default lesson;
