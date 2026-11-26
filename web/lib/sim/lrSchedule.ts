/**
 * Learning-rate schedules and a real noisy training problem they drive (pure math, no React).
 *
 * Schedules (t = 0 … T−1 is the optimizer step, η0 = base / peak learning rate):
 *   constant        η0
 *   step            η0 · γ^⌊t/s⌋
 *   exponential     η0 · γ^(t/s)              (smooth version of step decay; per-step factor γ^(1/s))
 *   cosine          ½ η0 (1 + cos(π t/T))
 *   restarts        SGDR: cosine inside cycles of length T0, 2T0, 4T0, …
 *   warmup-cosine   η0 (t+1)/W for t < W, then cosine over the remaining T − W steps
 *   one-cycle       PyTorch OneCycleLR: cosine from η0/25 up to η0 over 30% of T, then down to η0/(25·10⁴)
 *   inv-sqrt        η0 · min((t+1)/W, √(W/(t+1)))   (Vaswani et al. 2017, normalised to peak η0)
 *
 * Training problem: logistic regression on a fixed, seeded 2-feature dataset with badly scaled
 * features, 15% flipped labels (so the gradient noise never vanishes and a constant learning
 * rate settles on a noise floor) and 3% outliers with 8× larger features (so a few mini-batches
 * produce very large gradients — what gradient clipping is for). Mini-batch SGD + momentum (β = 0.9, PyTorch form), optional
 * gradient clipping by global norm. Every schedule sees the same mini-batch sequence.
 */

import { gaussian, mulberry32 } from "./random";

export type ScheduleId = "constant" | "step" | "exponential" | "cosine" | "restarts" | "warmup-cosine" | "one-cycle" | "inv-sqrt";

export const SCHEDULES: { id: ScheduleId; label: string }[] = [
  { id: "constant", label: "Constant" },
  { id: "step", label: "Step decay" },
  { id: "exponential", label: "Exponential" },
  { id: "cosine", label: "Cosine" },
  { id: "restarts", label: "Cosine + restarts" },
  { id: "warmup-cosine", label: "Warm-up + cosine" },
  { id: "one-cycle", label: "One-cycle" },
  { id: "inv-sqrt", label: "Inverse sqrt" },
];

export interface ScheduleParams {
  /** peak / base learning rate η0 */
  base: number;
  /** total number of optimizer steps T */
  total: number;
  /** warm-up length W (warm-up + cosine, inverse sqrt) */
  warmup: number;
  /** step decay interval s (also the time scale of exponential decay) */
  stepSize: number;
  /** decay factor γ per interval (step, exponential) */
  gamma: number;
  /** length of the first SGDR cycle T0 (cycle lengths double) */
  restartPeriod: number;
}

const ONE_CYCLE = { pctStart: 0.3, div: 25, finalDiv: 1e4 };

/** Cosine interpolation from a (frac = 0) to b (frac = 1). */
const cosInterp = (a: number, b: number, frac: number) => b + 0.5 * (a - b) * (1 + Math.cos(Math.PI * Math.min(1, Math.max(0, frac))));

/** Start and length of the SGDR cycle that contains step t (periods T0, 2T0, 4T0, …). */
function sgdrCycle(t: number, t0: number) {
  let start = 0, period = Math.max(1, t0), index = 0;
  while (t >= start + period) {
    start += period;
    period *= 2;
    index += 1;
  }
  return { start, period, index };
}

export function learningRate(id: ScheduleId, t: number, p: ScheduleParams): number {
  const { base, total: T } = p;
  const W = Math.max(1, Math.round(p.warmup));
  switch (id) {
    case "constant":
      return base;
    case "step":
      return base * p.gamma ** Math.floor(t / p.stepSize);
    case "exponential":
      return base * p.gamma ** (t / p.stepSize);
    case "cosine":
      return cosInterp(base, 0, t / T);
    case "restarts": {
      const c = sgdrCycle(t, p.restartPeriod);
      return cosInterp(base, 0, (t - c.start) / c.period);
    }
    case "warmup-cosine":
      if (t < W) return (base * (t + 1)) / W;
      return cosInterp(base, 0, (t - W) / Math.max(1, T - W));
    case "one-cycle": {
      const lo = base / ONE_CYCLE.div;
      const end = lo / ONE_CYCLE.finalDiv;
      const upEnd = ONE_CYCLE.pctStart * T - 1;
      if (t <= upEnd) return cosInterp(lo, base, t / upEnd);
      return cosInterp(base, end, (t - upEnd) / (T - 1 - upEnd));
    }
    case "inv-sqrt":
      return base * Math.min((t + 1) / W, Math.sqrt(W / (t + 1)));
  }
}

/** Plain-language description of where schedule `id` is at step t. */
export function schedulePhase(id: ScheduleId, t: number, p: ScheduleParams): string {
  const T = p.total;
  const W = Math.max(1, Math.round(p.warmup));
  const pct = (x: number) => `${Math.round(100 * Math.min(1, Math.max(0, x)))}%`;
  switch (id) {
    case "constant":
      return "constant: the same rate on every step";
    case "step": {
      const k = Math.floor(t / p.stepSize);
      const next = (k + 1) * p.stepSize;
      return `plateau ${k + 1} (η0·γ^${k})${next < T ? `, next drop at step ${next}` : ", final plateau"}`;
    }
    case "exponential":
      return `smooth decay, ×${p.gamma} every ${p.stepSize} steps`;
    case "cosine":
      return t < T / 2 ? `cosine, ${pct(t / T)} done, still in the high first half` : `cosine, ${pct(t / T)} done, slowly dropping to 0`;
    case "restarts": {
      const c = sgdrCycle(t, p.restartPeriod);
      return `cycle ${c.index + 1} (length ${c.period}), ${pct((t - c.start) / c.period)} through; restart at step ${c.start + c.period}`;
    }
    case "warmup-cosine":
      return t < W ? `linear warm-up, ${t + 1}/${W}` : `cosine decay, ${pct((t - W) / Math.max(1, T - W))} done`;
    case "one-cycle": {
      const upEnd = ONE_CYCLE.pctStart * T - 1;
      return t <= upEnd ? `rising from η0/25 towards the peak (${pct(t / upEnd)})` : `dropping towards η0/250000 (${pct((t - upEnd) / (T - 1 - upEnd))})`;
    }
    case "inv-sqrt":
      return t + 1 < W ? `linear warm-up, ${t + 1}/${W}` : `1/√t decay (rate = peak·√(W/t), ${(Math.sqrt(W / (t + 1)) * 100).toFixed(0)}% of peak)`;
  }
}

/* ------------------------------------------------------------------ */
/* The training problem                                                */
/* ------------------------------------------------------------------ */

export const PROBLEM = {
  n: 400,
  batch: 8,
  beta: 0.9,
  /** feature scales: the second feature is 3× larger → curvature ratio ≈ 9 */
  scales: [1, 3] as const,
  /** 3% of the examples are outliers with 8× larger features: occasional huge mini-batch gradients */
  outlierFrac: 0.03,
  outlierScale: 8,
  labelNoise: 0.15,
  seed: 7,
};

export interface Dataset {
  X: number[][]; // n × 3 (two features + constant 1 for the bias)
  y: number[]; // 0/1 labels (15% flipped)
  /** minimum full-batch loss, from Newton's method */
  lossStar: number;
  /** curvature (largest Hessian eigenvalue) at w = 0 — the initial stability scale */
  lambdaMax0: number;
}

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** log(1 + e^z), numerically stable */
const softplus = (z: number) => (z > 30 ? z : z < -30 ? Math.exp(z) : Math.log1p(Math.exp(z)));

export function fullLoss(ds: Dataset, w: number[]): number {
  let s = 0;
  for (let i = 0; i < ds.X.length; i++) {
    const z = ds.X[i][0] * w[0] + ds.X[i][1] * w[1] + ds.X[i][2] * w[2];
    s += softplus(z) - ds.y[i] * z;
  }
  return s / ds.X.length;
}

function gradient(ds: Dataset, w: number[], idx: number[]): number[] {
  const g = [0, 0, 0];
  for (const i of idx) {
    const x = ds.X[i];
    const r = sigmoid(x[0] * w[0] + x[1] * w[1] + x[2] * w[2]) - ds.y[i];
    g[0] += r * x[0];
    g[1] += r * x[1];
    g[2] += r * x[2];
  }
  return g.map((v) => v / idx.length);
}

/** Solve the 3×3 system A x = b by Gaussian elimination with partial pivoting. */
function solve3(A: number[][], b: number[]): number[] {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < 3; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = M[r][3];
    for (let k = r + 1; k < 3; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

function hessian(ds: Dataset | { X: number[][] }, w: number[]): number[][] {
  const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const n = ds.X.length;
  for (const x of ds.X) {
    const s = sigmoid(x[0] * w[0] + x[1] * w[1] + x[2] * w[2]);
    const c = (s * (1 - s)) / n;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) H[a][b] += c * x[a] * x[b];
  }
  return H;
}

/** Largest eigenvalue of a symmetric 3×3 matrix by power iteration. */
function lambdaMax(H: number[][]): number {
  let v = [1, 1, 1];
  let lam = 0;
  for (let k = 0; k < 200; k++) {
    const u = H.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
    lam = Math.hypot(u[0], u[1], u[2]);
    v = u.map((x) => x / (lam || 1));
  }
  return lam;
}

let cachedDataset: Dataset | null = null;

/** The fixed, seeded dataset (identical on every reload). */
export function makeDataset(): Dataset {
  if (cachedDataset) return cachedDataset;
  const rand = mulberry32(PROBLEM.seed);
  const wTrue = [1.5, -0.5, 0.3];
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < PROBLEM.n; i++) {
    const big = rand() < PROBLEM.outlierFrac ? PROBLEM.outlierScale : 1; // rare outliers -> rare huge gradients
    const x = [gaussian(rand) * PROBLEM.scales[0] * big, gaussian(rand) * PROBLEM.scales[1] * big, 1];
    const z = x[0] * wTrue[0] + x[1] * wTrue[1] + wTrue[2];
    let label = rand() < sigmoid(2 * z) ? 1 : 0; // fairly separable signal …
    if (rand() < PROBLEM.labelNoise) label = 1 - label; // … plus label noise that cannot be fitted
    X.push(x);
    y.push(label);
  }
  const partial = { X, y, lossStar: 0, lambdaMax0: 0 };
  // Newton's method for the exact minimum (3 parameters → trivial)
  let w = [0, 0, 0];
  for (let k = 0; k < 30; k++) {
    const g = gradient(partial, w, X.map((_, i) => i));
    const step = solve3(hessian(partial, w), g);
    w = w.map((v, i) => v - step[i]);
  }
  partial.lossStar = fullLoss(partial, w);
  partial.lambdaMax0 = lambdaMax(hessian(partial, [0, 0, 0]));
  cachedDataset = partial;
  return partial;
}

/** The mini-batch index sequence shared by every schedule (seeded, so runs are comparable). */
let cachedBatches: { total: number; batches: number[][] } | null = null;
function batchSequence(total: number): number[][] {
  if (cachedBatches && cachedBatches.total >= total) return cachedBatches.batches;
  const rand = mulberry32(PROBLEM.seed + 1000);
  const batches: number[][] = [];
  for (let t = 0; t < total; t++) {
    const b: number[] = [];
    for (let k = 0; k < PROBLEM.batch; k++) b.push(Math.floor(rand() * PROBLEM.n));
    batches.push(b);
  }
  cachedBatches = { total, batches };
  return batches;
}

export interface TrainRun {
  /** lr[t] — learning rate used at step t (t = 0 … T−1) */
  lr: number[];
  /** excess[t] — full-batch loss minus the optimum after t steps (t = 0 … T); NaN once diverged */
  excess: number[];
  /** number of steps whose gradient norm was clipped, cumulative: clippedUpTo[t] after t steps */
  clippedUpTo: number[];
  /** gradient norm before clipping at each step */
  gradNorm: number[];
  diverged: boolean;
}

/**
 * Train logistic regression from w = 0 with mini-batch SGD + momentum (β = 0.9) under schedule `id`.
 * clip: global-norm threshold, or null for no clipping.
 */
export function trainWithSchedule(id: ScheduleId, p: ScheduleParams, clip: number | null): TrainRun {
  const ds = makeDataset();
  const T = Math.max(1, Math.round(p.total));
  const batches = batchSequence(T);
  let w = [0, 0, 0];
  let v = [0, 0, 0];
  const lr: number[] = [];
  const excess: number[] = [fullLoss(ds, w) - ds.lossStar];
  const clippedUpTo: number[] = [0];
  const gradNorm: number[] = [];
  let clipped = 0;
  let diverged = false;
  for (let t = 0; t < T; t++) {
    const eta = learningRate(id, t, p);
    lr.push(eta);
    if (diverged) {
      excess.push(NaN);
      clippedUpTo.push(clipped);
      gradNorm.push(NaN);
      continue;
    }
    let g = gradient(ds, w, batches[t]);
    const norm = Math.hypot(g[0], g[1], g[2]);
    gradNorm.push(norm);
    if (clip !== null && norm > clip) {
      g = g.map((x) => (x * clip) / norm);
      clipped += 1;
    }
    v = v.map((vi, i) => PROBLEM.beta * vi + g[i]);
    w = w.map((wi, i) => wi - eta * v[i]);
    const L = fullLoss(ds, w) - ds.lossStar;
    if (!Number.isFinite(L) || Math.abs(w[0]) + Math.abs(w[1]) + Math.abs(w[2]) > 1e4) {
      diverged = true;
      excess.push(NaN);
    } else excess.push(Math.max(L, 1e-9));
    clippedUpTo.push(clipped);
  }
  return { lr, excess, clippedUpTo, gradNorm, diverged };
}
