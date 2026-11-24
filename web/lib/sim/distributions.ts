/**
 * Probability distributions, seeded sampling, and maximum-likelihood / MAP
 * estimation for one scalar parameter. Pure functions, no React.
 */
import { gaussian, mulberry32 } from "./random";

/* ------------------------------------------------------------------ */
/* Part A — distributions                                             */
/* ------------------------------------------------------------------ */

export type DistKind = "bernoulli" | "binomial" | "gaussian" | "uniform";

export interface DistParams {
  p: number; // Bernoulli / binomial success probability
  n: number; // binomial number of trials
  mu: number; // Gaussian mean
  sigma: number; // Gaussian standard deviation
  a: number; // uniform lower bound
  b: number; // uniform upper bound
}

export const DEFAULT_PARAMS: DistParams = { p: 0.3, n: 10, mu: 0, sigma: 1, a: -1, b: 2 };

export const isDiscrete = (k: DistKind) => k === "bernoulli" || k === "binomial";

/** log of n choose k, computed as a sum of logs (exact enough for n ≤ 1000). */
export function logChoose(n: number, k: number) {
  if (k < 0 || k > n) return -Infinity;
  let s = 0;
  const m = Math.min(k, n - k);
  for (let i = 1; i <= m; i++) s += Math.log(n - m + i) - Math.log(i);
  return s;
}

export function binomialPmf(k: number, n: number, p: number) {
  if (k < 0 || k > n || !Number.isInteger(k)) return 0;
  if (p <= 0) return k === 0 ? 1 : 0;
  if (p >= 1) return k === n ? 1 : 0;
  return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
}

export function normalPdf(x: number, mu: number, sigma: number) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/** PMF (discrete) or PDF (continuous) at x. */
export function density(kind: DistKind, x: number, q: DistParams) {
  switch (kind) {
    case "bernoulli":
      return x === 1 ? q.p : x === 0 ? 1 - q.p : 0;
    case "binomial":
      return binomialPmf(x, q.n, q.p);
    case "gaussian":
      return normalPdf(x, q.mu, q.sigma);
    case "uniform":
      return x >= q.a && x <= q.b ? 1 / (q.b - q.a) : 0;
  }
}

export function trueMean(kind: DistKind, q: DistParams) {
  switch (kind) {
    case "bernoulli":
      return q.p;
    case "binomial":
      return q.n * q.p;
    case "gaussian":
      return q.mu;
    case "uniform":
      return (q.a + q.b) / 2;
  }
}

export function trueVariance(kind: DistKind, q: DistParams) {
  switch (kind) {
    case "bernoulli":
      return q.p * (1 - q.p);
    case "binomial":
      return q.n * q.p * (1 - q.p);
    case "gaussian":
      return q.sigma * q.sigma;
    case "uniform":
      return ((q.b - q.a) * (q.b - q.a)) / 12;
  }
}

/** Draw `count` samples from a seeded stream. Sample i depends only on (seed, params, i ≤ count). */
export function drawSamples(kind: DistKind, q: DistParams, count: number, seed: number): number[] {
  const rand = mulberry32(seed);
  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    switch (kind) {
      case "bernoulli":
        out[i] = rand() < q.p ? 1 : 0;
        break;
      case "binomial": {
        let k = 0;
        for (let t = 0; t < q.n; t++) if (rand() < q.p) k++;
        out[i] = k;
        break;
      }
      case "gaussian":
        out[i] = q.mu + q.sigma * gaussian(rand);
        break;
      case "uniform":
        out[i] = q.a + (q.b - q.a) * rand();
        break;
    }
  }
  return out;
}

/** Prefix sums of x and x² so the running mean/variance of the first N samples is O(1). */
export function prefixSums(xs: number[]) {
  const s1 = new Float64Array(xs.length + 1);
  const s2 = new Float64Array(xs.length + 1);
  for (let i = 0; i < xs.length; i++) {
    s1[i + 1] = s1[i] + xs[i];
    s2[i + 1] = s2[i] + xs[i] * xs[i];
  }
  return { s1, s2 };
}

/** Sample mean and (1/N) sample variance of the first N samples. */
export function runningStats(ps: { s1: Float64Array; s2: Float64Array }, N: number) {
  if (N <= 0) return { mean: NaN, variance: NaN };
  const mean = ps.s1[N] / N;
  const variance = Math.max(0, ps.s2[N] / N - mean * mean);
  return { mean, variance };
}

/** The x-range drawn for each distribution. */
export function plotRange(kind: DistKind, q: DistParams): [number, number] {
  switch (kind) {
    case "bernoulli":
      return [-0.5, 1.5];
    case "binomial":
      return [-0.5, q.n + 0.5];
    case "gaussian":
      return [q.mu - 4 * q.sigma, q.mu + 4 * q.sigma];
    case "uniform": {
      const w = q.b - q.a;
      return [q.a - 0.3 * w, q.b + 0.3 * w];
    }
  }
}

export interface Histogram {
  /** For discrete: one bin per integer value lo..hi. For continuous: equal-width bins. */
  edges: number[];
  /** Empirical probability (discrete) or empirical density (continuous) per bin. */
  heights: number[];
  counts: number[];
}

export function histogram(kind: DistKind, q: DistParams, xs: number[], N: number, bins = 32): Histogram {
  if (isDiscrete(kind)) {
    const hi = kind === "bernoulli" ? 1 : q.n;
    const counts = new Array<number>(hi + 1).fill(0);
    for (let i = 0; i < N; i++) counts[xs[i]]++;
    const edges = Array.from({ length: hi + 2 }, (_, k) => k - 0.5);
    return { edges, counts, heights: counts.map((c) => (N > 0 ? c / N : 0)) };
  }
  const [lo, hi] = plotRange(kind, q);
  const w = (hi - lo) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (let i = 0; i < N; i++) {
    const j = Math.floor((xs[i] - lo) / w);
    if (j >= 0 && j < bins) counts[j]++;
  }
  const edges = Array.from({ length: bins + 1 }, (_, k) => lo + k * w);
  return { edges, counts, heights: counts.map((c) => (N > 0 ? c / (N * w) : 0)) };
}

/* ------------------------------------------------------------------ */
/* Part B — maximum likelihood and MAP for one parameter              */
/* ------------------------------------------------------------------ */

export type ModelKind = "gaussian-mean" | "bernoulli-p";

export interface MleSetup {
  model: ModelKind;
  data: number[];
  /** Known noise standard deviation for the Gaussian-mean model. */
  sigma: number;
  useMap: boolean;
  /** Gaussian prior N(mu0, tau²) on μ. */
  mu0: number;
  tau: number;
  /** Beta(alpha, beta) prior on p. */
  alpha: number;
  beta: number;
}

export const P_MIN = 0.001;
export const P_MAX = 0.999;

export function makeDataset(model: ModelKind, n: number, truth: number, sigma: number, seed: number): number[] {
  const rand = mulberry32(seed * 7919 + 17);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(model === "gaussian-mean" ? truth + sigma * gaussian(rand) : rand() < truth ? 1 : 0);
  return out;
}

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** Total log-likelihood ℓ(θ) = Σ log p(xᵢ | θ). */
export function logLik(s: MleSetup, th: number) {
  const n = s.data.length;
  if (s.model === "gaussian-mean") {
    let ss = 0;
    for (const x of s.data) ss += (x - th) * (x - th);
    return -0.5 * n * Math.log(2 * Math.PI * s.sigma * s.sigma) - ss / (2 * s.sigma * s.sigma);
  }
  const k = sum(s.data);
  const lp = (c: number, v: number) => (c === 0 ? 0 : c * Math.log(v));
  return lp(k, th) + lp(n - k, 1 - th);
}

/** dℓ/dθ (total, not averaged). */
export function gradLogLik(s: MleSetup, th: number) {
  const n = s.data.length;
  if (s.model === "gaussian-mean") return (sum(s.data) - n * th) / (s.sigma * s.sigma);
  const k = sum(s.data);
  return k / th - (n - k) / (1 - th);
}

/** Log of the prior density at θ (normalised). */
export function logPrior(s: MleSetup, th: number) {
  if (s.model === "gaussian-mean") {
    const z = (th - s.mu0) / s.tau; // computed in log space so a tight prior never underflows to log(0)
    return -0.5 * z * z - Math.log(s.tau * Math.sqrt(2 * Math.PI));
  }
  return (s.alpha - 1) * Math.log(th) + (s.beta - 1) * Math.log(1 - th) - logBeta(s.alpha, s.beta);
}

export function gradLogPrior(s: MleSetup, th: number) {
  if (s.model === "gaussian-mean") return -(th - s.mu0) / (s.tau * s.tau);
  return (s.alpha - 1) / th - (s.beta - 1) / (1 - th);
}

/** Objective that the stepper climbs: log-likelihood, or log-posterior (up to a constant) when MAP is on. */
export function objective(s: MleSetup, th: number) {
  return logLik(s, th) + (s.useMap ? logPrior(s, th) : 0);
}

export function gradObjective(s: MleSetup, th: number) {
  return gradLogLik(s, th) + (s.useMap ? gradLogPrior(s, th) : 0);
}

/** Negative second derivative (curvature) of the averaged objective at θ — sets the stable step size. */
export function curvature(s: MleSetup, th: number) {
  const n = Math.max(1, s.data.length);
  if (s.model === "gaussian-mean") return (n / (s.sigma * s.sigma) + (s.useMap ? 1 / (s.tau * s.tau) : 0)) / n;
  const k = sum(s.data);
  let c = k / (th * th) + (n - k) / ((1 - th) * (1 - th));
  if (s.useMap) c += (s.alpha - 1) / (th * th) + (s.beta - 1) / ((1 - th) * (1 - th));
  return c / n;
}

/** Closed-form maximum-likelihood estimate. */
export function mleClosedForm(s: MleSetup) {
  const n = s.data.length;
  const m = n ? sum(s.data) / n : NaN;
  return s.model === "gaussian-mean" ? m : Math.min(P_MAX, Math.max(P_MIN, m));
}

/** Closed-form MAP estimate (mode of the posterior). */
export function mapClosedForm(s: MleSetup) {
  const n = s.data.length;
  const S = sum(s.data);
  if (s.model === "gaussian-mean") {
    const prec = n / (s.sigma * s.sigma) + 1 / (s.tau * s.tau);
    return (S / (s.sigma * s.sigma) + s.mu0 / (s.tau * s.tau)) / prec;
  }
  const den = n + s.alpha + s.beta - 2;
  const raw = den > 0 ? (S + s.alpha - 1) / den : 0.5;
  return Math.min(P_MAX, Math.max(P_MIN, raw));
}

/** The estimate the ascent should converge to. */
export const target = (s: MleSetup) => (s.useMap ? mapClosedForm(s) : mleClosedForm(s));

export function paramRange(s: MleSetup): [number, number] {
  return s.model === "gaussian-mean" ? [-4, 6] : [P_MIN, P_MAX];
}

function clampParam(s: MleSetup, th: number) {
  const [lo, hi] = s.model === "gaussian-mean" ? [-50, 50] : [P_MIN, P_MAX];
  return Math.min(hi, Math.max(lo, th));
}

export interface AscentStep {
  theta: number;
  obj: number;
  /** Gradient of the total objective. */
  grad: number;
  /** Gradient of the averaged objective (total / n), which the update uses. */
  gradAvg: number;
  thetaNext: number;
}

/**
 * Gradient ascent θ ← θ + η · (1/n) ∂/∂θ [objective]. Averaging by n keeps the step size
 * independent of the dataset size, exactly as a mean loss does in deep learning.
 */
export function runAscent(s: MleSetup, theta0: number, eta: number, steps: number): AscentStep[] {
  const n = Math.max(1, s.data.length);
  const out: AscentStep[] = [];
  let th = clampParam(s, theta0);
  for (let i = 0; i < steps; i++) {
    const g = gradObjective(s, th);
    const ga = g / n;
    const next = clampParam(s, th + eta * ga);
    out.push({ theta: th, obj: objective(s, th), grad: g, gradAvg: ga, thetaNext: next });
    th = next;
  }
  return out;
}

/** Lanczos approximation of log Γ(x), x > 0. */
export function logGamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  const tmp0 = x + 5.5;
  const tmp = tmp0 - (x + 0.5) * Math.log(tmp0);
  let ser = 1.000000000190015;
  for (const ci of c) ser += ci / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

export const logBeta = (a: number, b: number) => logGamma(a) + logGamma(b) - logGamma(a + b);

/**
 * Prior, normalised likelihood and posterior densities on a grid over the parameter range.
 * Normalisation is numerical (trapezoid rule), so the three curves are directly comparable.
 */
export function densityCurves(s: MleSetup, points = 241) {
  const [lo, hi] = paramRange(s);
  const xs = Array.from({ length: points }, (_, i) => lo + ((hi - lo) * i) / (points - 1));
  const normalise = (logv: number[]) => {
    const m = Math.max(...logv.filter(Number.isFinite));
    const v = logv.map((l) => (Number.isFinite(l) ? Math.exp(l - m) : 0));
    let area = 0;
    for (let i = 1; i < v.length; i++) area += ((v[i] + v[i - 1]) / 2) * (xs[i] - xs[i - 1]);
    return v.map((y) => (area > 0 ? y / area : 0));
  };
  const ll = xs.map((x) => logLik(s, x));
  const lp = xs.map((x) => logPrior({ ...s, useMap: true }, x));
  return {
    xs,
    likelihood: normalise(ll),
    prior: normalise(lp),
    posterior: normalise(ll.map((l, i) => l + lp[i])),
  };
}
