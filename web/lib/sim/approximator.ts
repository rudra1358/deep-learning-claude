/**
 * Pure math for the universal-approximation simulator (no React).
 *
 * Network (1 input, N hidden ReLUs, 1 output):
 *   h_j(x) = max(0, a_j·x + c_j),   ŷ(x) = Σ_j v_j h_j(x) + d
 * Each term v_j·h_j(x) is a "hinge": zero on one side of its kink x = −c_j / a_j, linear on the other.
 * ŷ is therefore piecewise linear with at most N kinks.
 *
 * Loss: MSE on the training points, (1/n) Σ (ŷ(x_i) − y_i)².
 * Gradients (δ_i = 2(ŷ_i − y_i)/n):
 *   ∂L/∂d = Σ δ_i,   ∂L/∂v_j = Σ δ_i h_j(x_i),
 *   ∂L/∂a_j = Σ δ_i v_j 1[a_j x_i + c_j > 0] x_i,   ∂L/∂c_j = Σ δ_i v_j 1[a_j x_i + c_j > 0]
 * Optimisers: plain gradient descent, or Adam (β1 = 0.9, β2 = 0.999, ε = 1e−8, bias-corrected).
 *
 * Initialisation (seeded): a_j ~ N(0, 2) (He for fan-in 1), kink positions u_j ~ U(−1, 1) with c_j = −a_j u_j
 * so the kinks are spread over the domain, v_j ~ N(0, 1/N), d = 0.
 */

import { gaussian, mulberry32 } from "./random";

export type TargetId = "sine" | "step" | "abs" | "bumps";
export type SampleId = "dense" | "sparse" | "noisy";
export type OptId = "adam" | "gd";

export const X_MIN = -1;
export const X_MAX = 1;

export const TARGETS: { id: TargetId; label: string; f: (x: number) => number; formula: string }[] = [
  { id: "sine", label: "Sine", f: (x) => Math.sin(Math.PI * x), formula: "sin(πx)" },
  { id: "step", label: "Step", f: (x) => (x >= 0.1 ? 0.6 : -0.6), formula: "±0.6, jump at x = 0.1" },
  { id: "abs", label: "|x|", f: (x) => Math.abs(x) - 0.5, formula: "|x| − 0.5" },
  {
    id: "bumps",
    label: "Bumps",
    f: (x) => 0.9 * Math.exp(-(((x + 0.5) / 0.2) ** 2)) - 0.7 * Math.exp(-(((x - 0.15) / 0.15) ** 2)) + 0.5 * Math.exp(-(((x - 0.65) / 0.22) ** 2)),
    formula: "sum of three Gaussians",
  },
];

export const SAMPLES: { id: SampleId; label: string }[] = [
  { id: "dense", label: "Dense (80 points)" },
  { id: "sparse", label: "Sparse (10 points)" },
  { id: "noisy", label: "Sparse + noise (10, σ = 0.15)" },
];

export interface Data {
  x: number[];
  y: number[];
}

export function makeData(target: TargetId, sample: SampleId, seed = 11): Data {
  const f = TARGETS.find((t) => t.id === target)!.f;
  if (sample === "dense") {
    const n = 80;
    const x = Array.from({ length: n }, (_, i) => X_MIN + ((i + 0.5) / n) * (X_MAX - X_MIN));
    return { x, y: x.map(f) };
  }
  const rand = mulberry32(seed);
  const n = 10;
  const x = Array.from({ length: n }, () => X_MIN + rand() * (X_MAX - X_MIN)).sort((a, b) => a - b);
  const sigma = sample === "noisy" ? 0.15 : 0;
  return { x, y: x.map((v) => f(v) + sigma * gaussian(rand)) };
}

export interface Params {
  a: number[];
  c: number[];
  v: number[];
  d: number;
}

export function initParams(N: number, seed: number): Params {
  const rand = mulberry32(seed);
  const a: number[] = [], c: number[] = [], v: number[] = [];
  for (let j = 0; j < N; j++) {
    const aj = Math.SQRT2 * gaussian(rand);
    const u = X_MIN + rand() * (X_MAX - X_MIN);
    a.push(aj);
    c.push(-aj * u);
    v.push(gaussian(rand) / Math.sqrt(N));
  }
  return { a, c, v, d: 0 };
}

export const relu = (z: number) => (z > 0 ? z : 0);

export function predict(p: Params, x: number) {
  let s = p.d;
  for (let j = 0; j < p.a.length; j++) s += p.v[j] * relu(p.a[j] * x + p.c[j]);
  return s;
}

/** Contribution v_j·h_j(x) of hidden unit j. */
export const unitOut = (p: Params, j: number, x: number) => p.v[j] * relu(p.a[j] * x + p.c[j]);

export function mse(p: Params, xs: number[], ys: number[]) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = predict(p, xs[i]) - ys[i];
    s += e * e;
  }
  return s / Math.max(1, xs.length);
}

/** Dense grid for the "true" error against the target function (what generalisation means here). */
export const GRID_X = Array.from({ length: 201 }, (_, i) => X_MIN + (i / 200) * (X_MAX - X_MIN));

export interface Frame {
  iter: number;
  params: Params;
  trainMse: number;
  /** MSE against the noiseless target on a dense grid */
  trueMse: number;
}

const clone = (p: Params): Params => ({ a: p.a.slice(), c: p.c.slice(), v: p.v.slice(), d: p.d });

/**
 * Train for frames × itersPerFrame full-batch iterations; return frames 0..frames.
 * Frame k holds the parameters after k·itersPerFrame updates.
 */
export function train(opts: { N: number; seed: number; data: Data; target: TargetId; opt: OptId; lr: number; frames: number; itersPerFrame: number }): Frame[] {
  const { N, data, opt, lr, frames, itersPerFrame } = opts;
  const f = TARGETS.find((t) => t.id === opts.target)!.f;
  const gridY = GRID_X.map(f);
  const p = initParams(N, opts.seed);
  const n = data.x.length;
  const P = 3 * N + 1; // a, c, v, d flattened
  const m = new Float64Array(P), s2 = new Float64Array(P);
  const g = new Float64Array(P);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  let t = 0;
  const out: Frame[] = [];
  const snap = (iter: number) => out.push({ iter, params: clone(p), trainMse: mse(p, data.x, data.y), trueMse: mse(p, GRID_X, gridY) });
  snap(0);
  const h = new Float64Array(N);
  for (let k = 1; k <= frames; k++) {
    for (let it = 0; it < itersPerFrame; it++) {
      g.fill(0);
      for (let i = 0; i < n; i++) {
        const x = data.x[i];
        let yhat = p.d;
        for (let j = 0; j < N; j++) {
          const z = p.a[j] * x + p.c[j];
          h[j] = z > 0 ? z : 0;
          yhat += p.v[j] * h[j];
        }
        const delta = (2 * (yhat - data.y[i])) / n;
        g[3 * N] += delta;
        for (let j = 0; j < N; j++) {
          g[2 * N + j] += delta * h[j];
          if (h[j] > 0) {
            const back = delta * p.v[j];
            g[j] += back * x;
            g[N + j] += back;
          }
        }
      }
      t++;
      for (let q = 0; q < P; q++) {
        let step: number;
        if (opt === "adam") {
          m[q] = b1 * m[q] + (1 - b1) * g[q];
          s2[q] = b2 * s2[q] + (1 - b2) * g[q] * g[q];
          const mh = m[q] / (1 - b1 ** t), vh = s2[q] / (1 - b2 ** t);
          step = (lr * mh) / (Math.sqrt(vh) + eps);
        } else step = lr * g[q];
        if (q < N) p.a[q] -= step;
        else if (q < 2 * N) p.c[q - N] -= step;
        else if (q < 3 * N) p.v[q - 2 * N] -= step;
        else p.d -= step;
      }
    }
    snap(k * itersPerFrame);
  }
  return out;
}

/** Kinks x = −c_j / a_j of units whose hinge actually bends inside the domain (|v_j| not negligible). */
export function kinksInDomain(p: Params) {
  const out: number[] = [];
  for (let j = 0; j < p.a.length; j++) {
    if (Math.abs(p.a[j]) < 1e-9 || Math.abs(p.v[j] * p.a[j]) < 1e-4) continue;
    const k = -p.c[j] / p.a[j];
    if (k > X_MIN && k < X_MAX) out.push(k);
  }
  return out.sort((a, b) => a - b);
}

/** Units that are zero on every training point (no gradient reaches a_j, c_j, v_j: "dead" on this data). */
export function deadUnits(p: Params, xs: number[]) {
  let dead = 0;
  for (let j = 0; j < p.a.length; j++) if (xs.every((x) => p.a[j] * x + p.c[j] <= 0)) dead++;
  return dead;
}
