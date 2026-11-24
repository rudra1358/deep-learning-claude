/**
 * Pure math for the BatchNorm / LayerNorm simulator (no React).
 *
 * A mini-batch X is an N×D matrix: row n = one example, column d = one feature.
 *  - Batch normalization: one mean/variance per FEATURE (column), computed over the N examples.
 *  - Layer normalization: one mean/variance per EXAMPLE (row), computed over the D features.
 * Running statistics follow PyTorch: new = (1 − m)·old + m·batch, with the UNBIASED batch
 * variance (× N/(N−1)) stored in running_var, while the BIASED variance (÷ N) normalises the batch.
 */
import { gaussian, mulberry32 } from "./random";

export const NF = 4; // number of features shown
export type NormMode = "batch" | "layer";
export type Matrix = number[][];

export interface BatchSpec {
  /** batch size N (1–32) */
  n: number;
  /** per-feature offset added to every example */
  shift: number[];
  /** per-feature spread (standard deviation of the generating distribution) */
  scale: number[];
  /** seed of the tracked example (row 0) */
  seed: number;
  /** seed of the OTHER examples (rows 1..N−1); changing it resamples only them */
  othersSeed: number;
}

const mix = (a: number, b: number) => (Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35)) >>> 0;

/** Draw the batch. Row 0 depends only on `seed`; rows 1.. depend on (seed, othersSeed). */
export function sampleBatch(spec: BatchSpec): Matrix {
  const { n, shift, scale } = spec;
  const r0 = mulberry32(mix(spec.seed, 1));
  const rOthers = mulberry32(mix(spec.seed, 1000 + spec.othersSeed));
  const X: Matrix = [];
  for (let i = 0; i < n; i++) {
    const r = i === 0 ? r0 : rOthers;
    X.push(Array.from({ length: shift.length }, (_, d) => shift[d] + scale[d] * gaussian(r)));
  }
  return X;
}

export interface NormStats {
  /** one entry per statistics group: D groups (batch mode) or N groups (layer mode) */
  mean: number[];
  /** biased variance (divide by the group size M) */
  var: number[];
  /** number of values that share one mean/variance */
  groupSize: number;
}

/** Mean and biased variance per group. Batch mode: per column; layer mode: per row. */
export function groupStats(X: Matrix, mode: NormMode): NormStats {
  const N = X.length;
  const D = X[0]?.length ?? 0;
  if (mode === "batch") {
    const mean = Array.from({ length: D }, (_, d) => X.reduce((s, r) => s + r[d], 0) / N);
    const v = Array.from({ length: D }, (_, d) => X.reduce((s, r) => s + (r[d] - mean[d]) ** 2, 0) / N);
    return { mean, var: v, groupSize: N };
  }
  const mean = X.map((r) => r.reduce((s, x) => s + x, 0) / D);
  const v = X.map((r, i) => r.reduce((s, x) => s + (x - mean[i]) ** 2, 0) / D);
  return { mean, var: v, groupSize: D };
}

/** x̂ = (x − μ)/√(σ² + ε) using the given group statistics. */
export function standardize(X: Matrix, mode: NormMode, mean: number[], variance: number[], eps: number): Matrix {
  return X.map((r, i) => r.map((x, d) => {
    const g = mode === "batch" ? d : i;
    return (x - mean[g]) / Math.sqrt(variance[g] + eps);
  }));
}

/** y = γ ⊙ x̂ + β with per-feature γ, β (true for BatchNorm and for LayerNorm). */
export function affine(Xh: Matrix, gamma: number[], beta: number[]): Matrix {
  return Xh.map((r) => r.map((v, d) => gamma[d] * v + beta[d]));
}

export interface EmaUpdate {
  newMean: number[];
  /** null when N = 1: the unbiased variance N/(N−1)·σ² is undefined (PyTorch raises an error) */
  newVar: number[] | null;
  unbiasedVar: number[] | null;
}

/** One PyTorch-style running-statistics update with momentum m (weight of the NEW batch). */
export function emaUpdate(runMean: number[], runVar: number[], batch: NormStats, n: number, m: number): EmaUpdate {
  const newMean = runMean.map((r, d) => (1 - m) * r + m * batch.mean[d]);
  if (n < 2) return { newMean, newVar: null, unbiasedVar: null };
  const unbiasedVar = batch.var.map((v) => (v * n) / (n - 1));
  const newVar = runVar.map((r, d) => (1 - m) * r + m * unbiasedVar[d]);
  return { newMean, newVar, unbiasedVar };
}

/**
 * Running statistics after `k` earlier training batches drawn from the same distribution
 * (starting from PyTorch's initial buffers: mean 0, var 1). Earlier batches use size max(N, 2).
 */
export function warmRunning(spec: BatchSpec, k: number, m: number): { mean: number[]; var: number[] } {
  let mean = new Array(spec.shift.length).fill(0);
  let v = new Array(spec.shift.length).fill(1);
  const n = Math.max(2, spec.n);
  for (let b = 0; b < k; b++) {
    const X = sampleBatch({ ...spec, n, seed: mix(spec.seed, 50_000 + b), othersSeed: b });
    const st = groupStats(X, "batch");
    const u = emaUpdate(mean, v, st, n, m);
    mean = u.newMean;
    v = u.newVar ?? v;
  }
  return { mean, var: v };
}

export function colMeanStd(Y: Matrix): { mean: number[]; std: number[] } {
  const st = groupStats(Y, "batch");
  return { mean: st.mean, std: st.var.map(Math.sqrt) };
}
