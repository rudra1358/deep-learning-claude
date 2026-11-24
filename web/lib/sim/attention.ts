/**
 * Pure math for the Attention & Transformer Explorer.
 *
 * Everything here is deterministic: the same sentence and settings always give
 * the same numbers. Weights are random (untrained) — the point is the exact
 * computation, not linguistically meaningful attention patterns.
 */
import { gaussian, hashString, matmul, mulberry32, softmax, transpose, type Matrix } from "./random";

export const D_MODEL = 8;
export const MAX_TOKENS = 10;
export const VOCAB_SIZE = 10000;

export interface AttentionConfig {
  sentence: string;
  heads: 1 | 2 | 4;
  usePE: boolean;
  causal: boolean;
  /** Seed for the projection matrices W_Q, W_K, W_V, W_O. */
  seed: number;
}

export interface HeadResult {
  Q: Matrix; // n × d_k
  K: Matrix; // n × d_k
  V: Matrix; // n × d_k
  scores: Matrix; // n × n  (Q Kᵀ)
  scaled: Matrix; // n × n  (Q Kᵀ / √d_k)
  masked: Matrix; // n × n  (−Infinity where masked)
  weights: Matrix; // n × n  (row-wise softmax)
  out: Matrix; // n × d_k (weights · V)
}

export interface AttentionResult {
  tokens: string[];
  ids: number[];
  dK: number;
  E: Matrix; // n × d_model token embeddings
  PE: Matrix; // n × d_model positional encodings
  X: Matrix; // n × d_model  (E + PE, or E)
  WQ: Matrix; // d_model × d_model
  WK: Matrix;
  WV: Matrix;
  WO: Matrix;
  heads: HeadResult[];
  concat: Matrix; // n × d_model
  Y: Matrix; // n × d_model  (concat · W_O)
}

/** Lower-case whitespace tokenization, capped at MAX_TOKENS tokens. */
export function tokenize(sentence: string): string[] {
  return sentence
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0)
    .slice(0, MAX_TOKENS);
}

/** A toy hashed vocabulary: the same word always gets the same id. */
export function tokenId(word: string): number {
  return hashString(word) % VOCAB_SIZE;
}

/** Deterministic embedding row for a word: N(0,1) entries seeded by the word's hash. */
export function embedWord(word: string, d = D_MODEL): number[] {
  const rand = mulberry32(hashString("emb:" + word));
  return Array.from({ length: d }, () => gaussian(rand));
}

/** Sinusoidal positional encoding (Vaswani et al., 2017). PE[pos][2i] = sin(pos/10000^(2i/d)), PE[pos][2i+1] = cos(...). */
export function sinusoidalPE(n: number, d: number): Matrix {
  return Array.from({ length: n }, (_, pos) =>
    Array.from({ length: d }, (_, j) => {
      const i2 = j - (j % 2); // 2i
      const angle = pos / Math.pow(10000, i2 / d);
      return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }),
  );
}

/** Seeded d×d matrix with N(0, 1/d) entries (Xavier-like scale keeps activations O(1)). */
function seededMatrix(rows: number, cols: number, seed: number): Matrix {
  const rand = mulberry32(seed);
  const s = 1 / Math.sqrt(rows);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => gaussian(rand) * s));
}

const sliceCols = (A: Matrix, from: number, to: number): Matrix => A.map((r) => r.slice(from, to));

export function addMatrices(A: Matrix, B: Matrix): Matrix {
  return A.map((r, i) => r.map((v, j) => v + B[i][j]));
}

/** Single-head scaled dot-product attention with optional causal mask. */
export function attentionHead(Q: Matrix, K: Matrix, V: Matrix, causal: boolean): HeadResult {
  const n = Q.length;
  const dK = Q[0]?.length ?? 1;
  const scores = n ? matmul(Q, transpose(K)) : [];
  const scaled = scores.map((r) => r.map((v) => v / Math.sqrt(dK)));
  const masked = scaled.map((r, i) => r.map((v, j) => (causal && j > i ? -Infinity : v)));
  const weights = masked.map((r) => softmax(r));
  const out = n ? matmul(weights, V) : [];
  return { Q, K, V, scores, scaled, masked, weights, out };
}

export function runAttention(cfg: AttentionConfig): AttentionResult {
  const tokens = tokenize(cfg.sentence);
  const n = tokens.length;
  const d = D_MODEL;
  const dK = d / cfg.heads;
  const ids = tokens.map(tokenId);
  const E = tokens.map((w) => embedWord(w, d));
  const PE = sinusoidalPE(n, d);
  const X = cfg.usePE ? addMatrices(E, PE) : E.map((r) => r.slice());
  const WQ = seededMatrix(d, d, cfg.seed * 4 + 1);
  const WK = seededMatrix(d, d, cfg.seed * 4 + 2);
  const WV = seededMatrix(d, d, cfg.seed * 4 + 3);
  const WO = seededMatrix(d, d, cfg.seed * 4 + 4);
  const empty: Matrix = [];
  const Qf = n ? matmul(X, WQ) : empty;
  const Kf = n ? matmul(X, WK) : empty;
  const Vf = n ? matmul(X, WV) : empty;
  const heads: HeadResult[] = [];
  for (let h = 0; h < cfg.heads; h++) {
    const a = h * dK;
    const b = a + dK;
    heads.push(attentionHead(sliceCols(Qf, a, b), sliceCols(Kf, a, b), sliceCols(Vf, a, b), cfg.causal));
  }
  const concat = Array.from({ length: n }, (_, i) => heads.flatMap((hr) => hr.out[i]));
  const Y = n ? matmul(concat, WO) : empty;
  return { tokens, ids, dK, E, PE, X, WQ, WK, WV, WO, heads, concat, Y };
}

/** Number of learned parameters in a multi-head attention layer with biases (4 d² + 4 d). */
export function mhaParamCount(d: number, withBias = true) {
  return 4 * d * d + (withBias ? 4 * d : 0);
}

export function maxAbs(A: Matrix): number {
  let m = 0;
  for (const r of A) for (const v of r) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m || 1;
}
