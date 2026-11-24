/**
 * Pure-TypeScript inference engine for the small character-level Transformer trained by
 * `scripts/train-char-transformer.py` and served at `/models/char-transformer.json`.
 *
 * Architecture (decoder-only, pre-LN, GPT-style):
 *   x₀ = E[ids] + P[0..n-1]                                   (token + learned position embeddings)
 *   for each layer:  h = LN₁(x);  x = x + MHA(h)                (causal multi-head self-attention)
 *                    h = LN₂(x);  x = x + W₂·GELU(W₁h + b₁) + b₂ (position-wise feed-forward)
 *   logits = LN_f(x) · Eᵀ                                       (weight-tied output head)
 *
 * `forward` returns every intermediate so the simulator can display real numbers.
 * The file is self-contained (no imports) so it can also be run directly by Node for
 * the PyTorch-vs-browser logit check (`verifyModel`).
 */

export type Matrix = number[][];

export interface TensorJSON {
  shape: number[];
  scale: number;
  q: number[];
}

export interface ModelJSON {
  format: string;
  description: string;
  config: {
    vocab_size: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    d_ff: number;
    context: number;
    positions: string;
    norm: string;
    ln_eps: number;
    activation: string;
    tied_head: boolean;
  };
  vocab: string;
  n_params: number;
  training: {
    chars: number;
    train_chars: number;
    val_chars: number;
    steps: number;
    batch: number;
    optimizer: string;
    dropout: number;
    seed: number;
    history: { step: number; train: number; val: number }[];
    final_train_loss: number;
    final_val_loss: number;
  };
  quantisation: string;
  tensors: Record<string, TensorJSON>;
  verify: { prompt: string; positions: number[]; logits: number[][] }[];
  samples: { prompt: string; greedy: string }[];
}

export interface LayerWeights {
  ln1g: number[];
  ln1b: number[];
  Wq: Matrix;
  bq: number[];
  Wk: Matrix;
  bk: number[];
  Wv: Matrix;
  bv: number[];
  Wo: Matrix;
  bo: number[];
  ln2g: number[];
  ln2b: number[];
  W1: Matrix;
  b1: number[];
  W2: Matrix;
  b2: number[];
}

export interface CharModel {
  json: ModelJSON;
  vocab: string[];
  stoi: Map<string, number>;
  d: number;
  nHeads: number;
  dHead: number;
  dFF: number;
  context: number;
  eps: number;
  tokEmb: Matrix; // V × d
  posEmb: Matrix; // context × d
  layers: LayerWeights[];
  lnfG: number[];
  lnfB: number[];
  /** Number of learned parameters, counted from the decoded tensors (tied head counted once). */
  nParams: number;
}

export interface LayerNormResult {
  out: Matrix;
  mean: number[];
  std: number[];
}

export interface HeadTrace {
  Q: Matrix; // n × dHead
  K: Matrix;
  V: Matrix;
  /** Scaled scores QKᵀ/√d_head with −Infinity above the diagonal (causal mask). */
  scores: Matrix;
  /** Row-wise softmax of `scores`: weights[i][j] is how much position i attends to j. */
  weights: Matrix;
  out: Matrix; // n × dHead
}

export interface LayerTrace {
  xIn: Matrix; // residual stream entering the layer
  ln1: LayerNormResult;
  heads: HeadTrace[];
  concat: Matrix; // n × d
  attnOut: Matrix; // concat · W_O + b_O
  xMid: Matrix; // xIn + attnOut
  ln2: LayerNormResult;
  ffPre: Matrix; // n × d_ff (before GELU)
  ffAct: Matrix; // n × d_ff (after GELU)
  ffOut: Matrix; // n × d
  xOut: Matrix; // xMid + ffOut
}

export interface ForwardTrace {
  ids: number[];
  chars: string[];
  tokEmb: Matrix; // n × d rows of E
  posEmb: Matrix; // n × d rows of P
  x0: Matrix;
  layers: LayerTrace[];
  lnF: LayerNormResult;
  logits: Matrix; // n × V
}

/* ------------------------------------------------------------------ decoding */

function decode(t: TensorJSON): number[] {
  return t.q.map((v) => v * t.scale);
}

function decodeMatrix(t: TensorJSON): Matrix {
  const [r, c] = t.shape;
  const flat = decode(t);
  return Array.from({ length: r }, (_, i) => flat.slice(i * c, (i + 1) * c));
}

export function loadModel(json: ModelJSON): CharModel {
  const T = json.tensors;
  const need = (k: string) => {
    const t = T[k];
    if (!t) throw new Error(`model file is missing tensor "${k}"`);
    return t;
  };
  const cfg = json.config;
  const layers: LayerWeights[] = Array.from({ length: cfg.n_layers }, (_, l) => {
    const p = `layers.${l}.`;
    return {
      ln1g: decode(need(p + "ln1.g")),
      ln1b: decode(need(p + "ln1.b")),
      Wq: decodeMatrix(need(p + "Wq")),
      bq: decode(need(p + "bq")),
      Wk: decodeMatrix(need(p + "Wk")),
      bk: decode(need(p + "bk")),
      Wv: decodeMatrix(need(p + "Wv")),
      bv: decode(need(p + "bv")),
      Wo: decodeMatrix(need(p + "Wo")),
      bo: decode(need(p + "bo")),
      ln2g: decode(need(p + "ln2.g")),
      ln2b: decode(need(p + "ln2.b")),
      W1: decodeMatrix(need(p + "W1")),
      b1: decode(need(p + "b1")),
      W2: decodeMatrix(need(p + "W2")),
      b2: decode(need(p + "b2")),
    };
  });
  const vocab = Array.from(json.vocab);
  const nParams = Object.values(T).reduce((s, t) => s + t.q.length, 0);
  return {
    json,
    vocab,
    stoi: new Map(vocab.map((c, i) => [c, i])),
    d: cfg.d_model,
    nHeads: cfg.n_heads,
    dHead: cfg.d_model / cfg.n_heads,
    dFF: cfg.d_ff,
    context: cfg.context,
    eps: cfg.ln_eps,
    tokEmb: decodeMatrix(need("tok_emb")),
    posEmb: decodeMatrix(need("pos_emb")),
    layers,
    lnfG: decode(need("ln_f.g")),
    lnfB: decode(need("ln_f.b")),
    nParams,
  };
}

/* ------------------------------------------------------------------ primitives */

/** X (n×a) · W (a×b) + bias (b). */
export function linear(X: Matrix, W: Matrix, bias?: number[]): Matrix {
  const a = W.length;
  const b = W[0]?.length ?? 0;
  return X.map((x) => {
    const out = bias ? bias.slice() : new Array<number>(b).fill(0);
    for (let k = 0; k < a; k++) {
      const xk = x[k];
      if (xk === 0) continue;
      const w = W[k];
      for (let j = 0; j < b; j++) out[j] += xk * w[j];
    }
    return out;
  });
}

export function addM(A: Matrix, B: Matrix): Matrix {
  return A.map((r, i) => r.map((v, j) => v + B[i][j]));
}

export function layerNorm(X: Matrix, g: number[], b: number[], eps: number): LayerNormResult {
  const mean: number[] = [];
  const std: number[] = [];
  const out = X.map((x) => {
    const m = x.reduce((s, v) => s + v, 0) / x.length;
    const v = x.reduce((s, xi) => s + (xi - m) * (xi - m), 0) / x.length; // biased variance, as in PyTorch
    const sd = Math.sqrt(v + eps);
    mean.push(m);
    std.push(sd);
    return x.map((xi, j) => ((xi - m) / sd) * g[j] + b[j]);
  });
  return { out, mean, std };
}

/** GELU, tanh approximation (PyTorch `approximate="tanh"`). */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

/** Numerically stable softmax with temperature; −Infinity entries get probability 0. */
export function softmaxT(v: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  let m = -Infinity;
  for (const x of v) if (Number.isFinite(x) && x > m) m = x;
  if (!Number.isFinite(m)) m = 0;
  const e = v.map((x) => (Number.isFinite(x) ? Math.exp((x - m) / t) : 0));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((x) => x / s);
}

export function rowNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

/* ------------------------------------------------------------------ forward pass */

/** Characters → ids. Returns the list of characters that are not in the vocabulary. */
export function encode(model: CharModel, text: string): { ids: number[]; unknown: string[] } {
  const ids: number[] = [];
  const unknown: string[] = [];
  for (const c of Array.from(text)) {
    const id = model.stoi.get(c);
    if (id === undefined) {
      if (!unknown.includes(c)) unknown.push(c);
    } else ids.push(id);
  }
  return { ids, unknown };
}

/** Full forward pass over ids (at most `context` of them), returning every intermediate. */
export function forward(model: CharModel, idsIn: number[]): ForwardTrace {
  const ids = idsIn.slice(-model.context);
  const n = ids.length;
  const { d, nHeads, dHead } = model;
  const tokEmb = ids.map((id) => model.tokEmb[id].slice());
  const posEmb = ids.map((_, i) => model.posEmb[i].slice());
  const x0 = addM(tokEmb, posEmb);
  let x = x0;
  const layers: LayerTrace[] = [];
  const scale = 1 / Math.sqrt(dHead);
  for (const L of model.layers) {
    const xIn = x;
    const ln1 = layerNorm(xIn, L.ln1g, L.ln1b, model.eps);
    const Qf = linear(ln1.out, L.Wq, L.bq);
    const Kf = linear(ln1.out, L.Wk, L.bk);
    const Vf = linear(ln1.out, L.Wv, L.bv);
    const heads: HeadTrace[] = [];
    for (let h = 0; h < nHeads; h++) {
      const a = h * dHead;
      const Q = Qf.map((r) => r.slice(a, a + dHead));
      const K = Kf.map((r) => r.slice(a, a + dHead));
      const V = Vf.map((r) => r.slice(a, a + dHead));
      const scores = Q.map((q, i) =>
        K.map((k, j) => {
          if (j > i) return -Infinity; // causal mask: no attending to the future
          let s = 0;
          for (let t = 0; t < dHead; t++) s += q[t] * k[t];
          return s * scale;
        }),
      );
      const weights = scores.map((r) => softmaxT(r, 1));
      const out = weights.map((w) => {
        const o = new Array<number>(dHead).fill(0);
        for (let j = 0; j < n; j++) {
          if (w[j] === 0) continue;
          for (let t = 0; t < dHead; t++) o[t] += w[j] * V[j][t];
        }
        return o;
      });
      heads.push({ Q, K, V, scores, weights, out });
    }
    const concat = Array.from({ length: n }, (_, i) => heads.flatMap((hh) => hh.out[i]));
    const attnOut = linear(concat, L.Wo, L.bo);
    const xMid = addM(xIn, attnOut);
    const ln2 = layerNorm(xMid, L.ln2g, L.ln2b, model.eps);
    const ffPre = linear(ln2.out, L.W1, L.b1);
    const ffAct = ffPre.map((r) => r.map(gelu));
    const ffOut = linear(ffAct, L.W2, L.b2);
    const xOut = addM(xMid, ffOut);
    layers.push({ xIn, ln1, heads, concat, attnOut, xMid, ln2, ffPre, ffAct, ffOut, xOut });
    x = xOut;
  }
  const lnF = layerNorm(x, model.lnfG, model.lnfB, model.eps);
  // Weight tying: logits = LN_f(x) · Eᵀ  (each logit is a dot product with a token embedding)
  const logits = lnF.out.map((h) =>
    model.tokEmb.map((e) => {
      let s = 0;
      for (let t = 0; t < d; t++) s += h[t] * e[t];
      return s;
    }),
  );
  return { ids, chars: ids.map((i) => model.vocab[i]), tokEmb, posEmb, x0, layers, lnF, logits };
}

/* ------------------------------------------------------------------ generation */

/** Deterministic PRNG (mulberry32) — same algorithm as lib/sim/random.ts. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Draw an index from a probability vector using one uniform number u ∈ [0,1). */
export function sampleIndex(p: number[], u: number): number {
  let c = 0;
  for (let i = 0; i < p.length; i++) {
    c += p[i];
    if (u < c) return i;
  }
  return p.length - 1;
}

export interface NextToken {
  id: number;
  char: string;
  probs: number[]; // softmax(logits / T) at the last position
  trace: ForwardTrace;
}

/**
 * Predict the next character after `ids`. Greedy picks the arg-max; sampling draws from
 * softmax(logits / temperature) with a uniform number `u` (pass a seeded value for reproducibility).
 */
export function nextToken(model: CharModel, ids: number[], opts: { temperature: number; greedy: boolean; u?: number }): NextToken {
  const trace = forward(model, ids);
  const last = trace.logits[trace.logits.length - 1];
  const probs = softmaxT(last, opts.greedy ? 1 : opts.temperature);
  let id = 0;
  if (opts.greedy) {
    for (let i = 1; i < last.length; i++) if (last[i] > last[id]) id = i;
  } else id = sampleIndex(probs, opts.u ?? 0.5);
  return { id, char: model.vocab[id], probs, trace };
}

/** Generate `n` characters (greedy or seeded sampling). */
export function generate(model: CharModel, prompt: string, n: number, opts: { temperature: number; greedy: boolean; seed?: number }): string {
  const { ids } = encode(model, prompt);
  const rand = mulberry32(opts.seed ?? 0);
  for (let k = 0; k < n; k++) ids.push(nextToken(model, ids, { ...opts, u: rand() }).id);
  return ids.map((i) => model.vocab[i]).join("");
}

/* ------------------------------------------------------------------ checks */

/**
 * Compare this engine's logits with the logits PyTorch computed for the same (exported) weights.
 * Returns the largest absolute difference over all stored positions and vocabulary entries.
 */
export function verifyModel(model: CharModel): { maxAbsDiff: number; checked: number } {
  let maxAbsDiff = 0;
  let checked = 0;
  for (const v of model.json.verify) {
    const tr = forward(model, encode(model, v.prompt).ids);
    v.positions.forEach((p, k) => {
      tr.logits[p].forEach((x, j) => {
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(x - v.logits[k][j]));
        checked++;
      });
    });
  }
  return { maxAbsDiff, checked };
}

/** Indices of the k largest entries, descending. */
export function topK(v: number[], k: number): number[] {
  return v
    .map((x, i) => [x, i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .map(([, i]) => i);
}

export function maxAbsM(A: Matrix): number {
  let m = 0;
  for (const r of A) for (const v of r) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m || 1;
}
