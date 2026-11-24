/**
 * A tiny vanilla (Elman) RNN used by the RNN and BPTT simulators.
 * Pure math, no React — every number the simulators show comes from here.
 *
 *   z_t = W_xh x_t + W_hh h_{t-1} + b
 *   h_t = tanh(z_t)
 *   y_t = W_hy · h_t + b_y          (scalar output head)
 */
import { gaussian, mulberry32 } from "./random";

export type Mat = number[][];

export interface RnnWeights {
  hidden: number;
  input: number;
  /** hidden × input */
  Wxh: Mat;
  /** hidden × hidden */
  Whh: Mat;
  b: number[];
  /** output head (1 × hidden), stored as a vector */
  Why: number[];
  by: number;
}

export interface RnnStep {
  x: number[];
  hPrev: number[];
  /** W_xh x_t */
  xin: number[];
  /** W_hh h_{t-1} */
  rec: number[];
  /** z_t = W_xh x_t + W_hh h_{t-1} + b */
  z: number[];
  h: number[];
  y: number;
}

export const matVec = (A: Mat, v: number[]) => A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
export const matTVec = (A: Mat, v: number[]) => A[0].map((_, j) => A.reduce((s, row, i) => s + row[j] * v[i], 0));
export const norm2 = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
export const frob = (A: Mat) => Math.sqrt(A.reduce((s, row) => s + row.reduce((r, x) => r + x * x, 0), 0));

function matMul(A: Mat, B: Mat): Mat {
  return A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
}

/**
 * Spectral radius ρ(A) = max |λ| via Gelfand's formula ρ = lim ‖A^n‖^{1/n},
 * evaluated at n = 2^12 by repeated squaring with renormalisation (works for complex eigenvalues too).
 */
export function spectralRadius(A: Mat): number {
  let B = A.map((r) => r.slice());
  let s = frob(B);
  if (s === 0) return 0;
  B = B.map((r) => r.map((x) => x / s));
  let logC = Math.log(s);
  const K = 12;
  for (let k = 0; k < K; k++) {
    B = matMul(B, B);
    s = frob(B);
    if (s === 0 || !Number.isFinite(s)) return 0;
    B = B.map((r) => r.map((x) => x / s));
    logC = 2 * logC + Math.log(s);
  }
  return Math.exp(logC / 2 ** K);
}

/** Spectral norm ‖A‖₂ (largest singular value) by power iteration on AᵀA. */
export function spectralNorm(A: Mat, iters = 60): number {
  let v = A[0].map((_, j) => 1 / Math.sqrt(A[0].length) + 0.01 * j);
  let sigma = 0;
  for (let it = 0; it < iters; it++) {
    const u = matVec(A, v);
    const w = matTVec(A, u);
    const n = norm2(w);
    if (n === 0) return 0;
    v = w.map((x) => x / n);
    sigma = norm2(matVec(A, v));
  }
  return sigma;
}

/**
 * Seeded weights. W_hh is drawn once from N(0,1) and then rescaled so that its spectral radius equals `rho`
 * exactly — the slider therefore changes only the *scale* of the recurrent matrix, not its shape.
 */
export function makeRnnWeights(hidden: number, input: number, rho: number, seed = 7, xScale = 1.0): RnnWeights {
  const rand = mulberry32(seed + 31 * hidden + 977 * input);
  const base = Array.from({ length: hidden }, () => Array.from({ length: hidden }, () => gaussian(rand)));
  const r0 = spectralRadius(base) || 1;
  const Whh = base.map((row) => row.map((v) => (v * rho) / r0));
  const Wxh = Array.from({ length: hidden }, () => Array.from({ length: input }, () => gaussian(rand) * xScale));
  const b = Array.from({ length: hidden }, () => gaussian(rand) * 0.1);
  const Why = Array.from({ length: hidden }, () => gaussian(rand) / Math.sqrt(hidden));
  return { hidden, input, Wxh, Whh, b, Why, by: 0 };
}

export function rnnStep(w: RnnWeights, x: number[], hPrev: number[], linear = false): RnnStep {
  const xin = matVec(w.Wxh, x);
  const rec = matVec(w.Whh, hPrev);
  const z = xin.map((v, i) => v + rec[i] + w.b[i]);
  const h = linear ? z.slice() : z.map(Math.tanh);
  const y = h.reduce((s, v, i) => s + v * w.Why[i], w.by);
  return { x, hPrev, xin, rec, z, h, y };
}

/** Run from h_0 = 0 over a sequence of input vectors (linear = true drops the tanh, for the BPTT simulator). */
export function runRnn(w: RnnWeights, xs: number[][], linear = false): RnnStep[] {
  let h = new Array(w.hidden).fill(0);
  const out: RnnStep[] = [];
  for (const x of xs) {
    const s = rnnStep(w, x, h, linear);
    out.push(s);
    h = s.h;
  }
  return out;
}

/**
 * Memory probe: add `eps` to every component of x_1, re-run the whole sequence, and return
 * ‖h_t(perturbed) − h_t‖ and |y_t(perturbed) − y_t| for every t. A real recomputation, not an estimate.
 */
export function memoryProbe(w: RnnWeights, xs: number[][], eps: number) {
  const base = runRnn(w, xs);
  const pert = runRnn(
    w,
    xs.map((x, t) => (t === 0 ? x.map((v) => v + eps) : x)),
  );
  const dh = base.map((s, t) => norm2(s.h.map((v, i) => v - pert[t].h[i])));
  const dy = base.map((s, t) => Math.abs(s.y - pert[t].y));
  return { dh, dy };
}

/** "0.5, -1 2" → [[0.5], [-1], [2]] (invalid tokens skipped, values clamped, length capped). */
export function parseNumbers(text: string, maxLen = 10): number[][] {
  return text
    .split(/[\s,;]+/)
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((v) => Number.isFinite(v))
    .map((v) => [Math.max(-3, Math.min(3, v))])
    .slice(0, maxLen);
}

/**
 * Characters → one-hot vectors over the sorted set of distinct characters (vocabulary capped at `maxVocab`;
 * characters beyond the cap are dropped). Input size = vocabulary size.
 */
export function parseChars(text: string, maxLen = 10, maxVocab = 4): { vocab: string[]; xs: number[][]; chars: string[] } {
  const raw = Array.from(text.replace(/\s+/g, "")).slice(0, 64);
  const vocab: string[] = [];
  for (const c of raw) if (!vocab.includes(c) && vocab.length < maxVocab) vocab.push(c);
  vocab.sort();
  const chars = raw.filter((c) => vocab.includes(c)).slice(0, maxLen);
  const xs = chars.map((c) => vocab.map((v) => (v === c ? 1 : 0)));
  return { vocab, xs, chars };
}

/** Sub-steps the RNN simulator walks through inside each time step. */
export const RNN_SUBSTEPS = ["input", "recurrent", "sum", "tanh", "output"] as const;
export type RnnSubStep = (typeof RNN_SUBSTEPS)[number];
