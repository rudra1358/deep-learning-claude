/**
 * Pure math for the under/over-fitting simulator (no React).
 *
 * Data:  x ~ U[0, 1],  y = sin(2πx) + σ·ε,  ε ~ N(0, 1), from a seeded PRNG.
 *        Every third point (i % 3 === 2) is held out for validation, so ~2/3 train, ~1/3 validation,
 *        and growing n only appends points (the first points never change).
 * Model: polynomial of degree d fitted by least squares on the training points.
 *        x is rescaled to t = 2x − 1 ∈ [−1, 1] and expanded in Legendre polynomials P0..Pd.
 *        They span exactly the same space as 1, t, …, t^d (so the fitted curve is the same
 *        least-squares polynomial) but give a far better-conditioned system.
 *        Solve the normal equations (ΦᵀΦ + λI)c = Φᵀy with a tiny ridge λ = 1e-8 for stability.
 */

import { gaussian, mulberry32 } from "./random";

export const RIDGE = 1e-8;
export const MAX_DEGREE = 15;

export const trueFn = (x: number) => Math.sin(2 * Math.PI * x);

export interface Dataset {
  xTrain: number[];
  yTrain: number[];
  xVal: number[];
  yVal: number[];
}

export function makeData(n: number, sigma: number, seed: number): Dataset {
  const rand = mulberry32(seed);
  const d: Dataset = { xTrain: [], yTrain: [], xVal: [], yVal: [] };
  for (let i = 0; i < n; i++) {
    const x = rand();
    const eps = gaussian(rand);
    const y = trueFn(x) + sigma * eps;
    if (i % 3 === 2) {
      d.xVal.push(x);
      d.yVal.push(y);
    } else {
      d.xTrain.push(x);
      d.yTrain.push(y);
    }
  }
  return d;
}

/** Legendre polynomials P0..Pd at t (Bonnet recursion). */
export function legendre(t: number, d: number): number[] {
  const p = [1];
  if (d >= 1) p.push(t);
  for (let k = 1; k < d; k++) p.push(((2 * k + 1) * t * p[k] - k * p[k - 1]) / (k + 1));
  return p;
}

const features = (x: number, d: number) => legendre(2 * x - 1, d);

/** Solve A c = b by Gaussian elimination with partial pivoting (A is small and square). */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const pv = M[col][col] || 1e-300;
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / pv;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const c = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * c[k];
    c[r] = s / (M[r][r] || 1e-300);
  }
  return c;
}

/** Least-squares polynomial fit of degree d; returns Legendre coefficients. */
export function fitPoly(xs: number[], ys: number[], d: number, ridge = RIDGE): number[] {
  const m = d + 1;
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const rhs = new Array(m).fill(0);
  for (let i = 0; i < xs.length; i++) {
    const phi = features(xs[i], d);
    for (let a = 0; a < m; a++) {
      rhs[a] += phi[a] * ys[i];
      for (let b = 0; b < m; b++) A[a][b] += phi[a] * phi[b];
    }
  }
  for (let a = 0; a < m; a++) A[a][a] += ridge;
  return solve(A, rhs);
}

export function evalPoly(coef: number[], x: number): number {
  const phi = features(x, coef.length - 1);
  let s = 0;
  for (let k = 0; k < coef.length; k++) s += coef[k] * phi[k];
  return s;
}

export function mse(coef: number[], xs: number[], ys: number[]): number {
  if (xs.length === 0) return NaN;
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = evalPoly(coef, xs[i]) - ys[i];
    s += r * r;
  }
  return s / xs.length;
}

export interface DegreeResult {
  degree: number;
  coef: number[];
  trainMse: number;
  valMse: number;
}

/** Fit every degree 0..maxDegree and report train / validation MSE. */
export function sweepDegrees(data: Dataset, maxDegree = MAX_DEGREE): DegreeResult[] {
  const out: DegreeResult[] = [];
  for (let d = 0; d <= maxDegree; d++) {
    const coef = fitPoly(data.xTrain, data.yTrain, d);
    out.push({ degree: d, coef, trainMse: mse(coef, data.xTrain, data.yTrain), valMse: mse(coef, data.xVal, data.yVal) });
  }
  return out;
}

export type Regime = "underfit" | "good" | "overfit";

/**
 * Label a degree relative to the sweep (a heuristic for the narration, not a statistical test).
 * "good": validation MSE within 50% (+0.01) of the best validation MSE, and — for degrees below the
 * best — training MSE also within 50% (+0.01) of the best model's training MSE (a model that cannot
 * even fit its training data is underfitting, however lucky its validation score).
 * Otherwise "underfit" below the best degree and "overfit" above it.
 */
export function regimeOf(results: DegreeResult[], degree: number): { regime: Regime; best: number } {
  let best = 0;
  for (const r of results) if (r.valMse < results[best].valMse) best = r.degree;
  const cur = results[degree];
  const ref = results[best];
  const valOk = cur.valMse <= ref.valMse * 1.5 + 0.01;
  const trainOk = degree >= best || cur.trainMse <= ref.trainMse * 1.5 + 0.01;
  if (valOk && trainOk) return { regime: "good", best };
  return { regime: degree < best ? "underfit" : "overfit", best };
}
