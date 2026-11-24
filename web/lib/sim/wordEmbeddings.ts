/**
 * Word-embedding maths for the Word Embedding Explorer (components/sims/WordEmbeddingsSim.tsx).
 *
 * Part 1 — trained vectors. Reads `/models/word2vec.json`, produced by `scripts/train-word2vec.py`
 * (skip-gram with negative sampling, d = 50, trained on a generated "Toyland English" corpus).
 * Each vector is stored as integers q and a per-word scale s (vector = q · s); we de-quantise,
 * L2-normalise, and compute cosine neighbours and 3CosAdd analogies exactly as the Python
 * script did, so `verifyModel` can compare against the Python results stored in the file.
 *
 * Part 2 — a toy skip-gram model (d = 2) trained live in the browser, one (centre, context,
 * negatives) update at a time, returning every intermediate number for display.
 *
 * No React and no imports, so Node can run it directly for checks.
 */

// ============================================================================ trained vectors
export interface W2VJSON {
  format: string;
  description: string;
  corpus: { sentences: number; tokens: number; bytes: number; types: number; kind: string };
  config: { dim: number; window: number; negatives: number; subsample_t: number; epochs: number; min_count: number; optimizer: string; batch: number };
  words: string[];
  counts: number[];
  catNames: string[];
  cat: number[];
  scale: number[];
  q: number[];
  pca: [number, number][];
  tsne: [number, number][];
  analogyEval: { relation: string; questions: number; top1: number; baseline: number }[];
  analogyOverall: { questions: number; top1: number; baseline: number };
  purity: number;
  verify: {
    neighbours: { word: string; top: [string, number][] }[];
    analogies: { abc: [string, string, string]; top: [string, number][] }[];
  };
}

export interface W2VModel {
  json: W2VJSON;
  words: string[];
  index: Map<string, number>;
  dim: number;
  /** V × dim, row-major, every row unit length */
  vecs: Float64Array;
}

export interface Hit {
  word: string;
  idx: number;
  sim: number;
}

export function loadModel(j: W2VJSON): W2VModel {
  const V = j.words.length;
  const d = j.config.dim;
  if (j.q.length !== V * d) throw new Error(`model file is inconsistent: ${j.q.length} numbers for ${V}×${d}`);
  const vecs = new Float64Array(V * d);
  for (let i = 0; i < V; i++) {
    let n = 0;
    for (let k = 0; k < d; k++) {
      const x = j.q[i * d + k] * j.scale[i];
      vecs[i * d + k] = x;
      n += x * x;
    }
    n = Math.sqrt(n) || 1;
    for (let k = 0; k < d; k++) vecs[i * d + k] /= n;
  }
  const index = new Map<string, number>();
  j.words.forEach((w, i) => index.set(w, i));
  return { json: j, words: j.words, index, dim: d, vecs };
}

export function vectorOf(m: W2VModel, idx: number): number[] {
  return Array.from(m.vecs.subarray(idx * m.dim, (idx + 1) * m.dim));
}

/** Cosine similarity of every vocabulary word with the query vector q (q need not be unit length). */
export function scoreAll(m: W2VModel, q: number[]): Float64Array {
  const d = m.dim;
  let qn = 0;
  for (const x of q) qn += x * x;
  qn = Math.sqrt(qn) || 1;
  const out = new Float64Array(m.words.length);
  for (let i = 0; i < m.words.length; i++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += m.vecs[i * d + k] * q[k];
    out[i] = s / qn;
  }
  return out;
}

/** Indices of the k largest scores, skipping `exclude`; ties broken by lower index (numpy's stable argsort). */
export function topK(scores: Float64Array, k: number, exclude: number[] = []): number[] {
  const ex = new Set(exclude);
  const best: number[] = [];
  for (let i = 0; i < scores.length; i++) {
    if (ex.has(i)) continue;
    if (best.length < k || scores[i] > scores[best[best.length - 1]]) {
      let p = best.length < k ? best.length : best.length - 1;
      if (best.length < k) best.push(i);
      else best[p] = i;
      while (p > 0 && scores[best[p - 1]] < scores[best[p]]) {
        [best[p - 1], best[p]] = [best[p], best[p - 1]];
        p--;
      }
    }
  }
  return best;
}

export function neighbours(m: W2VModel, word: string, k = 10): Hit[] {
  const i = m.index.get(word);
  if (i === undefined) return [];
  const s = scoreAll(m, vectorOf(m, i));
  return topK(s, k, [i]).map((j) => ({ word: m.words[j], idx: j, sim: s[j] }));
}

/** 3CosAdd: argmax_w cos(w, b − a + c), excluding a, b, c. "a is to b as c is to ?" */
export function analogy(m: W2VModel, a: string, b: string, c: string, k = 5): { hits: Hit[]; missing: string[] } {
  const missing = [a, b, c].filter((w) => !m.index.has(w));
  if (missing.length) return { hits: [], missing };
  const [ia, ib, ic] = [a, b, c].map((w) => m.index.get(w) as number);
  const va = vectorOf(m, ia), vb = vectorOf(m, ib), vc = vectorOf(m, ic);
  const q = vb.map((x, t) => x - va[t] + vc[t]);
  const s = scoreAll(m, q);
  return { hits: topK(s, k, [ia, ib, ic]).map((j) => ({ word: m.words[j], idx: j, sim: s[j] })), missing: [] };
}

export function cosine(m: W2VModel, a: string, b: string): number {
  const ia = m.index.get(a), ib = m.index.get(b);
  if (ia === undefined || ib === undefined) return NaN;
  let s = 0;
  for (let k = 0; k < m.dim; k++) s += m.vecs[ia * m.dim + k] * m.vecs[ib * m.dim + k];
  return s;
}

/** Compare with the neighbours/analogies Python computed from the same de-quantised vectors. */
export function verifyModel(m: W2VModel): { maxErr: number; sameWords: boolean; checked: number } {
  let maxErr = 0, sameWords = true, checked = 0;
  for (const v of m.json.verify.neighbours) {
    const got = neighbours(m, v.word, v.top.length);
    v.top.forEach(([w, s], i) => {
      checked++;
      if (got[i]?.word !== w) sameWords = false;
      maxErr = Math.max(maxErr, Math.abs((got[i]?.sim ?? NaN) - s) || 0);
    });
  }
  for (const v of m.json.verify.analogies) {
    const got = analogy(m, v.abc[0], v.abc[1], v.abc[2], v.top.length).hits;
    v.top.forEach(([w, s], i) => {
      checked++;
      if (got[i]?.word !== w) sameWords = false;
      maxErr = Math.max(maxErr, Math.abs((got[i]?.sim ?? NaN) - s) || 0);
    });
  }
  return { maxErr, sameWords, checked };
}

// ============================================================================ toy skip-gram (d = 2)
export const TOY_CORPUS = [
  "king rules castle",
  "queen rules castle",
  "king wears crown",
  "queen wears crown",
  "man walks dog",
  "woman walks dog",
  "man eats bread",
  "woman eats bread",
  "cat drinks milk",
  "dog drinks water",
  "cat chases mouse",
  "dog chases cat",
];

export interface ToyConfig {
  window: number;
  negatives: number;
  lr: number;
  seed: number;
}

export interface ToyState {
  words: string[];
  counts: number[];
  noise: number[]; // unigram^0.75, normalised
  pairs: [number, number][];
  v: number[][]; // input (centre) vectors, V × 2
  u: number[][]; // output (context) vectors, V × 2
  order: number[]; // shuffled pair order of the current epoch
  pos: number; // position within order
  epoch: number;
  updates: number;
  rand: () => number;
  lossHist: number[]; // mean loss per completed epoch
  epochLoss: number;
  epochN: number;
  cfg: ToyConfig;
}

export interface ToyUpdate {
  c: number;
  o: number;
  negs: number[];
  vBefore: number[];
  uoBefore: number[];
  unBefore: number[][];
  sPos: number;
  sigPos: number;
  sNeg: number[];
  sigNeg: number[];
  loss: number;
  gV: number[];
  gUo: number[];
  gUn: number[][];
  vAfter: number[];
  uoAfter: number[];
  unAfter: number[][];
}

/** Seeded PRNG (mulberry32), duplicated here so the file has no imports. */
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

function gauss(r: () => number) {
  const a = Math.max(r(), 1e-12), b = r();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

function shuffle(n: number, r: () => number) {
  const p = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

export function initToy(cfg: ToyConfig, corpus = TOY_CORPUS): ToyState {
  const sents = corpus.map((s) => s.split(" "));
  const counts = new Map<string, number>();
  for (const s of sents) for (const w of s) counts.set(w, (counts.get(w) ?? 0) + 1);
  const words = [...counts.keys()].sort((a, b) => (counts.get(b)! - counts.get(a)!) || a.localeCompare(b));
  const id = new Map(words.map((w, i) => [w, i]));
  const pairs: [number, number][] = [];
  for (const s of sents)
    for (let i = 0; i < s.length; i++)
      for (let j = Math.max(0, i - cfg.window); j <= Math.min(s.length - 1, i + cfg.window); j++)
        if (j !== i) pairs.push([id.get(s[i])!, id.get(s[j])!]);
  const cs = words.map((w) => counts.get(w)!);
  const z = cs.reduce((a, c) => a + c ** 0.75, 0);
  const noise = cs.map((c) => c ** 0.75 / z);
  const rand = mulberry32(cfg.seed);
  const init = () => [gauss(rand) * 0.5, gauss(rand) * 0.5];
  const v = words.map(init);
  const u = words.map(init);
  return { words, counts: cs, noise, pairs, v, u, order: shuffle(pairs.length, rand), pos: 0, epoch: 0, updates: 0, rand, lossHist: [], epochLoss: 0, epochN: 0, cfg };
}

/** Draw a word from the noise distribution, rejecting the centre and the true context word. */
function drawNegative(st: ToyState, avoid: number[]): number {
  for (;;) {
    let x = st.rand();
    let k = 0;
    while (k < st.noise.length - 1 && x >= st.noise[k]) {
      x -= st.noise[k];
      k++;
    }
    if (!avoid.includes(k)) return k;
  }
}

/** SGNS loss of one pair with given negatives (no update) — used for the full-loss readout. */
export function pairLoss(v: number[], uo: number[], un: number[][]): number {
  return -Math.log(sigmoid(dot(uo, v))) - un.reduce((s, u) => s + Math.log(sigmoid(-dot(u, v))), 0);
}

/** One SGD update on the next (centre, context) pair; mutates `st` and returns every intermediate. */
export function toyStep(st: ToyState): ToyUpdate {
  const [c, o] = st.pairs[st.order[st.pos]];
  const negs: number[] = [];
  for (let k = 0; k < st.cfg.negatives; k++) negs.push(drawNegative(st, [c, o]));
  const v = st.v[c].slice();
  const uo = st.u[o].slice();
  const un = negs.map((n) => st.u[n].slice());
  const sPos = dot(uo, v);
  const sigPos = sigmoid(sPos);
  const sNeg = un.map((u) => dot(u, v));
  const sigNeg = sNeg.map(sigmoid);
  const loss = -Math.log(sigPos) - sigNeg.reduce((s, q) => s + Math.log(1 - q), 0);
  // gradients:  dL/du_o = (σ(s+) − 1) v,  dL/du_k = σ(s_k) v,  dL/dv = (σ(s+) − 1) u_o + Σ σ(s_k) u_k
  const gUo = v.map((x) => (sigPos - 1) * x);
  const gUn = sigNeg.map((q) => v.map((x) => q * x));
  const gV = uo.map((x, t) => (sigPos - 1) * x + sigNeg.reduce((s, q, k) => s + q * un[k][t], 0));
  const lr = st.cfg.lr;
  st.v[c] = v.map((x, t) => x - lr * gV[t]);
  st.u[o] = st.u[o].map((x, t) => x - lr * gUo[t]);
  negs.forEach((n, k) => {
    st.u[n] = st.u[n].map((x, t) => x - lr * gUn[k][t]);
  });
  st.updates++;
  st.epochLoss += loss;
  st.epochN++;
  st.pos++;
  if (st.pos >= st.order.length) {
    st.lossHist.push(st.epochLoss / st.epochN);
    st.epochLoss = 0;
    st.epochN = 0;
    st.epoch++;
    st.pos = 0;
    st.order = shuffle(st.pairs.length, st.rand);
  }
  return {
    c, o, negs, vBefore: v, uoBefore: uo, unBefore: un, sPos, sigPos, sNeg, sigNeg, loss, gV, gUo, gUn,
    vAfter: st.v[c].slice(), uoAfter: st.u[o].slice(), unAfter: negs.map((n) => st.u[n].slice()),
  };
}

/** Cosine similarity between two input vectors of the toy model. */
export function toyCos(st: ToyState, a: number, b: number) {
  const x = st.v[a], y = st.v[b];
  return dot(x, y) / (Math.hypot(...x) * Math.hypot(...y) || 1);
}
