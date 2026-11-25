/**
 * Pure math for the quantization simulator (no React).
 *
 * Asymmetric (affine): grid 0..2^b-1, s = (β-α)/(2^b-1), z = round(-α/s), range extended to contain 0.
 * Symmetric: grid -(2^(b-1)-1)..2^(b-1)-1, z = 0, s = max|x| / (2^(b-1)-1).
 * q = clip(round(x/s) + z, qmin, qmax),  x̂ = s (q - z).
 * `clipRatio` (0, 1] shrinks the calibrated range to trade clipping error for a finer grid.
 */
import { gaussian, mulberry32 } from "./random";

export type QMode = "symmetric" | "asymmetric";
export type Granularity = "per-tensor" | "per-channel";

export interface QParams {
  scale: number;
  zero: number;
  qmin: number;
  qmax: number;
  lo: number; // calibrated (clipped) range actually representable
  hi: number;
}

export interface QConfig {
  bits: number;
  mode: QMode;
  granularity: Granularity;
  clipRatio: number;
}

/** Round half away from zero (deterministic, matches the usual textbook definition). */
export function roundHalfAway(v: number): number {
  return v < 0 ? -Math.round(-v) : Math.round(v);
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Scale and zero-point for the values `xs`. Safe for constant / all-zero inputs (never NaN). */
export function qparams(xs: number[], bits: number, mode: QMode, clipRatio = 1): QParams {
  const r = clamp(Number.isFinite(clipRatio) ? clipRatio : 1, 1e-3, 1);
  let mn = Infinity;
  let mx = -Infinity;
  for (const x of xs) {
    if (x < mn) mn = x;
    if (x > mx) mx = x;
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) {
    mn = 0;
    mx = 0;
  }
  if (mode === "symmetric") {
    const qmax = 2 ** (bits - 1) - 1;
    const amax = Math.max(Math.abs(mn), Math.abs(mx)) * r;
    const scale = amax > 0 ? amax / qmax : 1;
    return { scale, zero: 0, qmin: -qmax, qmax, lo: -qmax * scale, hi: qmax * scale };
  }
  const qmin = 0;
  const qmax = 2 ** bits - 1;
  const lo = Math.min(mn, 0) * r;
  const hi = Math.max(mx, 0) * r;
  const scale = hi > lo ? (hi - lo) / (qmax - qmin) : 1;
  const zero = clamp(roundHalfAway(qmin - lo / scale), qmin, qmax);
  return { scale, zero, qmin, qmax, lo: (qmin - zero) * scale, hi: (qmax - zero) * scale };
}

export function quantize(x: number, p: QParams): number {
  return clamp(roundHalfAway(x / p.scale + p.zero), p.qmin, p.qmax);
}

export function dequantize(q: number, p: QParams): number {
  return p.scale * (q - p.zero);
}

/** Every intermediate of quantizing one value — used for the step-through. */
export interface ValueTrace {
  x: number;
  divided: number; // x / s
  shifted: number; // x / s + z
  rounded: number; // round(...)
  q: number; // clipped integer
  clipped: boolean;
  xhat: number;
  error: number; // x - x̂
}

export function traceValue(x: number, p: QParams): ValueTrace {
  const divided = x / p.scale;
  const shifted = divided + p.zero;
  const rounded = roundHalfAway(shifted);
  const q = clamp(rounded, p.qmin, p.qmax);
  const xhat = dequantize(q, p);
  return { x, divided, shifted, rounded, q, clipped: q !== rounded, xhat, error: x - xhat };
}

export interface ErrorStats {
  mse: number;
  maxErr: number;
  sqnrDb: number; // Infinity when the error is exactly 0
}

export function errorStats(xs: number[], xhat: number[]): ErrorStats {
  let se = 0;
  let sig = 0;
  let maxErr = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = xs[i] - xhat[i];
    se += e * e;
    sig += xs[i] * xs[i];
    maxErr = Math.max(maxErr, Math.abs(e));
  }
  const n = Math.max(1, xs.length);
  const sqnrDb = se === 0 ? Infinity : sig === 0 ? -Infinity : 10 * Math.log10(sig / se);
  return { mse: se / n, maxErr, sqnrDb };
}

export interface QuantizedMatrix {
  params: QParams[]; // one per row (repeated for per-tensor)
  q: number[][];
  deq: number[][];
  rowStats: ErrorStats[];
  stats: ErrorStats;
  clippedCount: number;
}

export function quantizeMatrix(W: number[][], cfg: QConfig): QuantizedMatrix {
  const flat = W.flat();
  const shared = cfg.granularity === "per-tensor" ? qparams(flat, cfg.bits, cfg.mode, cfg.clipRatio) : null;
  const params = W.map((row) => shared ?? qparams(row, cfg.bits, cfg.mode, cfg.clipRatio));
  let clippedCount = 0;
  const q = W.map((row, i) =>
    row.map((x) => {
      const t = traceValue(x, params[i]);
      if (t.clipped) clippedCount++;
      return t.q;
    }),
  );
  const deq = q.map((row, i) => row.map((v) => dequantize(v, params[i])));
  const rowStats = W.map((row, i) => errorStats(row, deq[i]));
  return { params, q, deq, rowStats, stats: errorStats(flat, deq.flat()), clippedCount };
}

/* ------------------------------------------------------------------ */
/* The tiny fixed model: a 3-class linear classifier on 8 features.     */
/* ------------------------------------------------------------------ */

export const N_CLASSES = 3;
export const N_FEATURES = 8;
/** Row (output-channel) magnitudes differ by ~10×, as in trained layers — this is what per-channel helps with. */
export const ROW_SCALES = [0.12, 0.45, 1.3];

export interface TinyModel {
  W: number[][]; // N_CLASSES × N_FEATURES
  b: number[];
  X: number[][]; // evaluation points
}

/** Laplacian-ish weights: Gaussian × |Gaussian| gives heavier tails than a Gaussian. */
export function makeModel(seed: number, nPoints = 6): TinyModel {
  const rand = mulberry32(seed);
  const W = ROW_SCALES.map((s) =>
    Array.from({ length: N_FEATURES }, () => {
      const g = gaussian(rand);
      return s * g * (0.6 + 0.8 * Math.abs(gaussian(rand)));
    }),
  );
  const b = Array.from({ length: N_CLASSES }, () => 0.1 * gaussian(rand));
  // Evaluation points: standard-normal feature vectors.
  const X = Array.from({ length: nPoints }, () => Array.from({ length: N_FEATURES }, () => gaussian(rand)));
  return { W, b, X };
}

export function logits(W: number[][], b: number[], x: number[]): number[] {
  return W.map((row, k) => row.reduce((acc, w, j) => acc + w * x[j], b[k]));
}

export function argmax(v: number[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

/** Grid points (dequantized levels) of `p` that fall inside [lo, hi]; capped to `max` for drawing. */
export function gridLevels(p: QParams, lo: number, hi: number, max = 300): number[] {
  const out: number[] = [];
  const qa = Math.max(p.qmin, Math.ceil(lo / p.scale + p.zero - 1e-9));
  const qb = Math.min(p.qmax, Math.floor(hi / p.scale + p.zero + 1e-9));
  if (qb - qa + 1 > max) return out;
  for (let q = qa; q <= qb; q++) out.push(dequantize(q, p));
  return out;
}
