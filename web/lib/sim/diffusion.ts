/**
 * Pure-TypeScript engine for the diffusion-model simulator.
 *
 * Reads the DDPM noise-prediction networks trained by `scripts/train-diffusion.py` and served at
 * `/models/diffusion.json`, and implements, with no dependencies:
 *   - noise schedules: linear (Ho et al. 2020) and cosine (Nichol & Dhariwal 2021)
 *   - the closed-form forward process  x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε
 *   - the ε-network forward pass: [x_t, sinusoidal(t), one-hot(class)] → SiLU MLP → ε̂
 *   - classifier-free guidance        ε̃ = (1+w) ε(x_t, c) − w ε(x_t, ∅)
 *   - one DDPM ancestral step         x_{t−1} = (x_t − β_t/√(1−ᾱ_t) ε̂)/√α_t + σ_t z,  σ_t² = β̃_t
 *   - one DDIM (η = 0) step           x_{t'} = √ᾱ_{t'} x̂_0 + √(1−ᾱ_{t'}) ε̂
 *   - the energy distance between two point clouds (sample-quality metric)
 *
 * Index convention: arrays are 0-based, so array index i holds paper time step t = i + 1
 * (β[i] = β_{i+1}, ᾱ[i] = ᾱ_{i+1}). The network is fed the 0-based index i, exactly as in training.
 * Point sets are flat Float64Arrays of n × d numbers (row-major).
 *
 * The file has no imports so Node can run it directly (type stripping) for the PyTorch check.
 */

/* ------------------------------------------------------------------ PRNG */

/** mulberry32 — the same seeded PRNG as lib/sim/random.ts (duplicated to keep this file import-free). */
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

/** Fill a new array with n independent N(0,1) draws (Box–Muller, both outputs used). */
export function randn(n: number, rand: () => number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/* ------------------------------------------------------------------ JSON format */

export interface F16Tensor {
  shape: number[];
  /** base64 of little-endian IEEE-754 half floats, row-major */
  f16: string;
}

export interface NetJSON {
  name: string;
  data_dim: number;
  n_classes: number; // 0 = unconditional; otherwise class index n_classes is the "null" (unconditional) token
  hidden: number[];
  activation: string;
  n_params: number;
  layers: { W: F16Tensor; b: F16Tensor }[]; // W has shape [in, out] ("x @ W" orientation)
  training: Record<string, unknown> & { steps: number; batch: number; final_loss: number; history: { step: number; loss: number }[] };
  metrics: Record<string, number | string | Record<string, number>>;
}

export interface DiffusionJSON {
  format: string;
  description: string;
  schedule: { T: number; kind: string; beta_start: number; beta_end: number; betas: number[]; alpha_bar: number[] };
  time_embedding: { dim: number; max_period: number };
  nets: { moons: NetJSON; digits?: NetJSON };
  data: {
    moons_real: F16Tensor; // N × 2, standardised coordinates
    digits_real?: F16Tensor; // N × 64 in [-1, 1]
    digits_labels?: number[];
  };
  verify: {
    eps: { net: "moons" | "digits"; x: number[]; t: number[]; c: number[] | null; out: number[] }[];
    ddim: { net: "moons" | "digits"; steps: number; w: number; c: number[] | null; xT: number[]; x0: number[] }[];
  };
}

export function decodeF16(t: F16Tensor): Float64Array {
  const bin = atob(t.f16);
  const n = bin.length >> 1;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

/* ------------------------------------------------------------------ schedules */

export interface Schedule {
  T: number;
  betas: Float64Array;
  alphas: Float64Array;
  alphaBar: Float64Array;
}

function fromBetas(betas: Float64Array): Schedule {
  const T = betas.length;
  const alphas = new Float64Array(T);
  const alphaBar = new Float64Array(T);
  let p = 1;
  for (let i = 0; i < T; i++) {
    alphas[i] = 1 - betas[i];
    p *= alphas[i];
    alphaBar[i] = p;
  }
  return { T, betas, alphas, alphaBar };
}

/** Linear schedule: β_t evenly spaced from β_1 = start to β_T = end (Ho et al. 2020). */
export function linearSchedule(T: number, start: number, end: number): Schedule {
  const b = new Float64Array(T);
  for (let i = 0; i < T; i++) b[i] = start + ((end - start) * i) / (T - 1);
  return fromBetas(b);
}

/** Cosine schedule: ᾱ_t = f(t)/f(0), f(t) = cos²(((t/T)+s)/(1+s)·π/2), β_t clipped to ≤ 0.999 (Nichol & Dhariwal 2021). */
export function cosineSchedule(T: number, s = 0.008): Schedule {
  const f = (t: number) => Math.cos((((t / T) + s) / (1 + s)) * (Math.PI / 2)) ** 2;
  const b = new Float64Array(T);
  for (let i = 0; i < T; i++) b[i] = Math.min(0.999, 1 - f(i + 1) / f(i));
  return fromBetas(b);
}

/* ------------------------------------------------------------------ network */

export interface Net {
  json: NetJSON;
  d: number;
  nClasses: number;
  W: Float64Array[]; // each [in × out] row-major
  b: Float64Array[];
  dims: number[]; // [in, h1, ..., out]
  nParams: number;
}

export interface DiffusionModel {
  json: DiffusionJSON;
  sched: Schedule;
  embDim: number;
  maxPeriod: number;
  moons: Net;
  digits: Net | null;
  moonsReal: Float64Array;
  digitsReal: Float64Array | null;
}

function loadNet(j: NetJSON): Net {
  const W = j.layers.map((l) => decodeF16(l.W));
  const b = j.layers.map((l) => decodeF16(l.b));
  const dims = [j.layers[0].W.shape[0], ...j.layers.map((l) => l.W.shape[1])];
  const nParams = W.reduce((s, w) => s + w.length, 0) + b.reduce((s, v) => s + v.length, 0);
  return { json: j, d: j.data_dim, nClasses: j.n_classes, W, b, dims, nParams };
}

export function loadModel(json: DiffusionJSON): DiffusionModel {
  const sched = fromBetas(Float64Array.from(json.schedule.betas));
  return {
    json,
    sched,
    embDim: json.time_embedding.dim,
    maxPeriod: json.time_embedding.max_period,
    moons: loadNet(json.nets.moons),
    digits: json.nets.digits ? loadNet(json.nets.digits) : null,
    moonsReal: decodeF16(json.data.moons_real),
    digitsReal: json.data.digits_real ? decodeF16(json.data.digits_real) : null,
  };
}

/** Sinusoidal time embedding [sin(t·f_0..f_{h−1}), cos(t·f_0..f_{h−1})], f_k = P^(−k/h). */
export function timeEmbedding(t: number, dim: number, maxPeriod: number): Float64Array {
  const h = dim >> 1;
  const e = new Float64Array(dim);
  for (let k = 0; k < h; k++) {
    const a = t * Math.exp((-Math.log(maxPeriod) * k) / h);
    e[k] = Math.sin(a);
    e[k + h] = Math.cos(a);
  }
  return e;
}

/**
 * ε̂ = net(x_t, t, c) for n points (flat n×d) all at the same 0-based step index t.
 * `cls` gives one class index per point (use nClasses for the null token); ignored for unconditional nets.
 */
export function predictEps(m: DiffusionModel, net: Net, x: Float64Array, n: number, t: number, cls?: ArrayLike<number> | null): Float64Array {
  const d = net.d;
  const emb = timeEmbedding(t, m.embDim, m.maxPeriod);
  const nC = net.nClasses ? net.nClasses + 1 : 0;
  const inDim = net.dims[0];
  const L = net.W.length;
  const out = new Float64Array(n * d);
  let h = new Float64Array(inDim);
  for (let r = 0; r < n; r++) {
    h = new Float64Array(inDim);
    for (let j = 0; j < d; j++) h[j] = x[r * d + j];
    for (let j = 0; j < m.embDim; j++) h[d + j] = emb[j];
    if (nC && cls) h[d + m.embDim + cls[r]] = 1;
    for (let l = 0; l < L; l++) {
      const nin = net.dims[l];
      const nout = net.dims[l + 1];
      const W = net.W[l];
      const z = Float64Array.from(net.b[l]);
      for (let i = 0; i < nin; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = i * nout;
        for (let o = 0; o < nout; o++) z[o] += hi * W[row + o];
      }
      if (l < L - 1) for (let o = 0; o < nout; o++) z[o] = z[o] / (1 + Math.exp(-z[o])); // SiLU
      h = z;
    }
    for (let j = 0; j < d; j++) out[r * d + j] = h[j];
  }
  return out;
}

/** Guided noise prediction ε̃ = (1+w) ε(x,c) − w ε(x,∅) (classifier-free guidance; w = 0 → plain conditional). */
export function guidedEps(m: DiffusionModel, net: Net, x: Float64Array, n: number, t: number, cls: ArrayLike<number> | null, w: number): Float64Array {
  const ec = predictEps(m, net, x, n, t, cls);
  if (!net.nClasses || !cls || w === 0) return ec;
  const eu = predictEps(m, net, x, n, t, new Array(n).fill(net.nClasses));
  for (let i = 0; i < ec.length; i++) ec[i] = (1 + w) * ec[i] - w * eu[i];
  return ec;
}

/* ------------------------------------------------------------------ forward process */

/** Closed form q(x_t | x_0): x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε  (t is the 0-based index; t = −1 returns x_0). */
export function qSample(sched: Schedule, x0: Float64Array, t: number, eps: Float64Array): Float64Array {
  if (t < 0) return Float64Array.from(x0);
  const a = Math.sqrt(sched.alphaBar[t]);
  const s = Math.sqrt(1 - sched.alphaBar[t]);
  return x0.map((v, i) => a * v + s * eps[i]);
}

/** x̂_0 = (x_t − √(1−ᾱ_t) ε̂)/√ᾱ_t — the network's implied estimate of the clean sample. */
export function predictX0(sched: Schedule, x: Float64Array, t: number, eps: Float64Array): Float64Array {
  const a = Math.sqrt(sched.alphaBar[t]);
  const s = Math.sqrt(1 - sched.alphaBar[t]);
  return x.map((v, i) => (v - s * eps[i]) / a);
}

/* ------------------------------------------------------------------ samplers */

export interface StepResult {
  x: Float64Array; // the new, less noisy sample
  eps: Float64Array; // ε̂ used at this step
  x0: Float64Array; // x̂_0 implied by ε̂
}

/**
 * One DDPM ancestral step from 0-based index t to t−1 with σ_t² = β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t).
 * `z` is fresh N(0, I) noise (ignored at t = 0, where the posterior mean is returned).
 */
export function ddpmStep(m: DiffusionModel, net: Net, x: Float64Array, n: number, t: number, z: Float64Array, cls: ArrayLike<number> | null = null, w = 0): StepResult {
  const { betas, alphas, alphaBar } = m.sched;
  const eps = guidedEps(m, net, x, n, t, cls, w);
  const c = betas[t] / Math.sqrt(1 - alphaBar[t]);
  const sa = Math.sqrt(alphas[t]);
  const sigma = t > 0 ? Math.sqrt((betas[t] * (1 - alphaBar[t - 1])) / (1 - alphaBar[t])) : 0;
  const nx = x.map((v, i) => (v - c * eps[i]) / sa + sigma * z[i]);
  return { x: nx, eps, x0: predictX0(m.sched, x, t, eps) };
}

/** 0-based DDIM time indices: S values from T−1 down to 0, evenly spaced and rounded (as in training script). */
export function ddimTimesteps(T: number, S: number): number[] {
  if (S <= 1) return [T - 1];
  return Array.from({ length: S }, (_, i) => Math.round(T - 1 - ((T - 1) * i) / (S - 1)));
}

/** One deterministic DDIM (η = 0) step from index t to index tPrev (tPrev = −1 means "to clean data", ᾱ = 1). */
export function ddimStep(m: DiffusionModel, net: Net, x: Float64Array, n: number, t: number, tPrev: number, cls: ArrayLike<number> | null = null, w = 0): StepResult {
  const eps = guidedEps(m, net, x, n, t, cls, w);
  const x0 = predictX0(m.sched, x, t, eps);
  const ap = tPrev >= 0 ? m.sched.alphaBar[tPrev] : 1;
  const a = Math.sqrt(ap);
  const s = Math.sqrt(1 - ap);
  return { x: x0.map((v, i) => a * v + s * eps[i]), eps, x0 };
}

/** Run a full DDIM trajectory from x_T. */
export function ddimSample(m: DiffusionModel, net: Net, xT: Float64Array, n: number, S: number, cls: ArrayLike<number> | null = null, w = 0): Float64Array {
  const ts = ddimTimesteps(m.sched.T, S);
  let x = xT;
  ts.forEach((t, i) => {
    x = ddimStep(m, net, x, n, t, i + 1 < ts.length ? ts[i + 1] : -1, cls, w).x;
  });
  return x;
}

/* ------------------------------------------------------------------ metrics */

/**
 * Energy distance 2E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖ between two point sets (flat, dimension d); 0 iff same distribution.
 * Unbiased U-statistic: the within-set means exclude the zero self-distances (divide by n(n−1)),
 * matching the estimator in scripts/train-diffusion.py, so two samples of one distribution give ≈ 0 (can be slightly < 0).
 */
export function energyDistance(a: Float64Array, b: Float64Array, d: number): number {
  const na = a.length / d;
  const nb = b.length / d;
  const mean = (p: Float64Array, np: number, q: Float64Array, nq: number, same = false) => {
    let s = 0;
    for (let i = 0; i < np; i++)
      for (let j = 0; j < nq; j++) {
        let r = 0;
        for (let k = 0; k < d; k++) {
          const u = p[i * d + k] - q[j * d + k];
          r += u * u;
        }
        s += Math.sqrt(r);
      }
    return s / (same ? np * (np - 1) : np * nq);
  };
  return 2 * mean(a, na, b, nb) - mean(a, na, a, na, true) - mean(b, nb, b, nb, true);
}

/* ------------------------------------------------------------------ verification */

/** Re-compute the stored PyTorch (float64) ε outputs and DDIM trajectories; returns the worst absolute difference. */
export function verifyModel(m: DiffusionModel): { epsMaxDiff: number; ddimMaxDiff: number; checked: number } {
  let epsMaxDiff = 0;
  let ddimMaxDiff = 0;
  let checked = 0;
  const netOf = (k: "moons" | "digits") => (k === "moons" ? m.moons : m.digits);
  for (const v of m.json.verify.eps) {
    const net = netOf(v.net);
    if (!net) continue;
    const n = v.t.length;
    for (let r = 0; r < n; r++) {
      const e = predictEps(m, net, Float64Array.from(v.x.slice(r * net.d, (r + 1) * net.d)), 1, v.t[r], v.c ? [v.c[r]] : null);
      e.forEach((y, j) => {
        epsMaxDiff = Math.max(epsMaxDiff, Math.abs(y - v.out[r * net.d + j]));
        checked++;
      });
    }
  }
  for (const v of m.json.verify.ddim) {
    const net = netOf(v.net);
    if (!net) continue;
    const n = v.xT.length / net.d;
    const x0 = ddimSample(m, net, Float64Array.from(v.xT), n, v.steps, v.c, v.w);
    x0.forEach((y, j) => {
      ddimMaxDiff = Math.max(ddimMaxDiff, Math.abs(y - v.x0[j]));
      checked++;
    });
  }
  return { epsMaxDiff, ddimMaxDiff, checked };
}
