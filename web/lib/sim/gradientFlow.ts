/**
 * Gradient flow through a deep MLP — the pure computation behind the Gradient Flow simulator.
 * No React. Every number the simulator shows comes from forward() / backward() below.
 *
 * Network (all hidden layers square, width n, zero biases), layer l = 1..L:
 *   z^l   = W^l h^{l-1}                          W^l is n×n, h^0 = x ~ N(0, I)
 *   u^l   = LN(z^l)  (optional, no affine params) or u^l = z^l
 *   a^l   = f(u^l)
 *   h^l   = h^{l-1} + a^l   (residual)   or   h^l = a^l   (plain)
 * Loss:
 *   "random": L = (1/B) Σ_b g_b · h^L_b with a fixed random upstream gradient g ~ N(0, I)
 *   "mse"   : L = (1/2B) Σ_b (w_out · h^L_b − y_b)^2 with random targets y ~ N(0, 1)
 * Backward (BP2 with optional LayerNorm and identity path):
 *   δ^l = ∂L/∂z^l,   ∂L/∂W^l = Σ_b δ^l_b h^{l-1 T}_b,   ∂L/∂h^{l-1} = W^{l T} δ^l (+ ∂L/∂h^l if residual)
 *
 * Matrices are row-major Float64Arrays: a batch tensor X[b*n + j], a weight W[j*n + k] (out j, in k).
 */

import { gaussian, mulberry32 } from "./random";

export type FlowAct = "sigmoid" | "tanh" | "relu" | "leaky" | "linear";
export type FlowInit = "normal" | "xavier" | "he";
export type FlowLoss = "random" | "mse";

export const LEAKY_SLOPE = 0.01;
export const LN_EPS = 1e-5;
export const BATCH = 32;

export const FLOW_ACTS: Record<FlowAct, { label: string; f: (z: number) => number; df: (z: number) => number; dfMax: number; range?: [number, number] }> = {
  sigmoid: { label: "Sigmoid", f: (z) => 1 / (1 + Math.exp(-z)), df: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }, dfMax: 0.25, range: [0, 1] },
  tanh: { label: "Tanh", f: Math.tanh, df: (z) => { const t = Math.tanh(z); return 1 - t * t; }, dfMax: 1, range: [-1, 1] },
  relu: { label: "ReLU", f: (z) => (z > 0 ? z : 0), df: (z) => (z > 0 ? 1 : 0), dfMax: 1 },
  leaky: { label: "Leaky ReLU", f: (z) => (z > 0 ? z : LEAKY_SLOPE * z), df: (z) => (z > 0 ? 1 : LEAKY_SLOPE), dfMax: 1 },
  linear: { label: "Linear", f: (z) => z, df: () => 1, dfMax: 1 },
};

export interface FlowConfig {
  depth: number; // L, 2..50
  width: number; // n, 8..64
  act: FlowAct;
  init: FlowInit;
  sigma: number; // std for init === "normal"
  residual: boolean;
  layerNorm: boolean;
  loss: FlowLoss;
  seed: number;
}

/** Standard deviation of every weight. With square layers Xavier 2/(n_in+n_out) reduces to 1/n. */
export function initStd(cfg: Pick<FlowConfig, "init" | "sigma" | "width">) {
  const n = cfg.width;
  if (cfg.init === "xavier") return Math.sqrt(2 / (n + n));
  if (cfg.init === "he") return Math.sqrt(2 / n);
  return cfg.sigma;
}

export interface LayerStats {
  /** std of the layer output h^l over the whole batch */
  hStd: number;
  /** values of the activation a^l = f(u^l) (B·n numbers) for the histogram */
  acts: Float64Array;
  /** ReLU family: fraction of units that are zero for every example in the batch */
  deadUnits: number;
  /** fraction of (example, unit) entries with f'(u) < 10% of the activation's maximum slope */
  lowSlope: number;
  /** mean of f'(u) over the batch and units */
  meanSlope: number;
  /** ‖δ^l‖_F over the batch (δ^l = ∂L/∂z^l) */
  deltaNorm: number;
  /** ‖∂L/∂W^l‖_F */
  gradWNorm: number;
  /** ‖W^l‖_F */
  weightNorm: number;
  /** ‖∂L/∂W^l‖_F / ‖W^l‖_F — size of one gradient step relative to the weights (for η = 1) */
  relGrad: number;
}

export interface FlowResult {
  cfg: FlowConfig;
  std: number; // init std actually used
  inputStd: number;
  layers: LayerStats[]; // index 0 is layer 1
  loss: number;
  /** ‖∂L/∂W^1‖ / ‖∂L/∂W^L‖ */
  ratioW: number;
  /** ‖δ^1‖ / ‖δ^L‖ */
  ratioDelta: number;
}

/** z = H Wᵀ for a batch: out[b, j] = Σ_k H[b, k] W[j, k]. */
function forwardMatmul(H: Float64Array, W: Float64Array, B: number, n: number, out: Float64Array) {
  for (let b = 0; b < B; b++) {
    const ho = b * n;
    for (let j = 0; j < n; j++) {
      const wo = j * n;
      let s = 0;
      for (let k = 0; k < n; k++) s += H[ho + k] * W[wo + k];
      out[ho + j] = s;
    }
  }
}

function frob(X: Float64Array) {
  let s = 0;
  for (let i = 0; i < X.length; i++) s += X[i] * X[i];
  return Math.sqrt(s);
}

function std(X: Float64Array) {
  let m = 0;
  for (let i = 0; i < X.length; i++) m += X[i];
  m /= X.length;
  let v = 0;
  for (let i = 0; i < X.length; i++) v += (X[i] - m) * (X[i] - m);
  return Math.sqrt(v / X.length);
}

/** Run one forward pass and one backward pass. Deterministic for a given config (seeded). */
export function runFlow(cfg: FlowConfig): FlowResult {
  const { depth: L, width: n, act, residual, layerNorm } = cfg;
  const B = BATCH;
  const A = FLOW_ACTS[act];
  const rand = mulberry32(cfg.seed);
  const sd = initStd(cfg);

  // ---- seeded data and parameters ----
  const X = new Float64Array(B * n);
  for (let i = 0; i < X.length; i++) X[i] = gaussian(rand);
  const Ws: Float64Array[] = [];
  for (let l = 0; l < L; l++) {
    const W = new Float64Array(n * n);
    for (let i = 0; i < W.length; i++) W[i] = gaussian(rand) * sd;
    Ws.push(W);
  }
  const g = new Float64Array(B * n); // fixed random upstream gradient
  for (let i = 0; i < g.length; i++) g[i] = gaussian(rand);
  const wOut = new Float64Array(n);
  for (let i = 0; i < n; i++) wOut[i] = gaussian(rand) / Math.sqrt(n);
  const y = new Float64Array(B);
  for (let i = 0; i < B; i++) y[i] = gaussian(rand);

  // ---- forward, caching what backward needs ----
  const hs: Float64Array[] = [X]; // hs[l] = h^l
  const us: Float64Array[] = []; // us[l-1] = u^l (input of f)
  const as: Float64Array[] = []; // as[l-1] = a^l
  const invS: Float64Array[] = []; // per-example 1/sqrt(var + eps) for LayerNorm
  for (let l = 0; l < L; l++) {
    const hPrev = hs[l];
    const z = new Float64Array(B * n);
    forwardMatmul(hPrev, Ws[l], B, n, z);
    const is = new Float64Array(B);
    if (layerNorm) {
      for (let b = 0; b < B; b++) {
        const o = b * n;
        let m = 0;
        for (let j = 0; j < n; j++) m += z[o + j];
        m /= n;
        let v = 0;
        for (let j = 0; j < n; j++) v += (z[o + j] - m) ** 2;
        v /= n;
        const r = 1 / Math.sqrt(v + LN_EPS);
        is[b] = r;
        for (let j = 0; j < n; j++) z[o + j] = (z[o + j] - m) * r; // z now holds u = x̂
      }
    }
    const a = new Float64Array(B * n);
    for (let i = 0; i < a.length; i++) a[i] = A.f(z[i]);
    const h = residual ? new Float64Array(B * n) : a;
    if (residual) for (let i = 0; i < h.length; i++) h[i] = hPrev[i] + a[i];
    us.push(z);
    as.push(a);
    invS.push(is);
    hs.push(h);
  }

  // ---- loss and ∂L/∂h^L ----
  const hL = hs[L];
  let dh = new Float64Array(B * n);
  let loss = 0;
  if (cfg.loss === "random") {
    for (let i = 0; i < dh.length; i++) {
      loss += (g[i] * hL[i]) / B;
      dh[i] = g[i] / B;
    }
  } else {
    for (let b = 0; b < B; b++) {
      let p = 0;
      for (let j = 0; j < n; j++) p += wOut[j] * hL[b * n + j];
      const r = p - y[b];
      loss += (r * r) / (2 * B);
      for (let j = 0; j < n; j++) dh[b * n + j] = (r * wOut[j]) / B;
    }
  }

  // ---- backward ----
  const deltaNorm = new Array<number>(L).fill(0);
  const gradWNorm = new Array<number>(L).fill(0);
  const slopes: Float64Array[] = new Array(L);
  for (let l = L - 1; l >= 0; l--) {
    const u = us[l];
    const sl = new Float64Array(B * n);
    const delta = new Float64Array(B * n);
    for (let i = 0; i < delta.length; i++) {
      sl[i] = A.df(u[i]);
      delta[i] = dh[i] * sl[i]; // ∂L/∂u
    }
    slopes[l] = sl;
    if (layerNorm) {
      // ∂L/∂z = (1/s) (d − mean(d) − x̂ · mean(d ⊙ x̂)), row by row
      for (let b = 0; b < B; b++) {
        const o = b * n;
        let m1 = 0, m2 = 0;
        for (let j = 0; j < n; j++) {
          m1 += delta[o + j];
          m2 += delta[o + j] * u[o + j];
        }
        m1 /= n;
        m2 /= n;
        for (let j = 0; j < n; j++) delta[o + j] = invS[l][b] * (delta[o + j] - m1 - u[o + j] * m2);
      }
    }
    deltaNorm[l] = frob(delta);
    // ∂L/∂W[j,k] = Σ_b δ[b,j] h^{l-1}[b,k]
    const hPrev = hs[l];
    const W = Ws[l];
    let gw = 0;
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        let s = 0;
        for (let b = 0; b < B; b++) s += delta[b * n + j] * hPrev[b * n + k];
        gw += s * s;
      }
    }
    gradWNorm[l] = Math.sqrt(gw);
    // ∂L/∂h^{l-1}[b,k] = Σ_j δ[b,j] W[j,k]  (+ identity path)
    const dPrev = new Float64Array(B * n);
    for (let b = 0; b < B; b++) {
      const o = b * n;
      for (let j = 0; j < n; j++) {
        const d = delta[o + j];
        if (d === 0) continue;
        const wo = j * n;
        for (let k = 0; k < n; k++) dPrev[o + k] += d * W[wo + k];
      }
    }
    if (residual) for (let i = 0; i < dPrev.length; i++) dPrev[i] += dh[i];
    dh = dPrev;
  }

  // ---- per-layer statistics ----
  const isRelu = act === "relu" || act === "leaky";
  const layers: LayerStats[] = [];
  for (let l = 0; l < L; l++) {
    const a = as[l];
    const sl = slopes[l];
    let dead = 0;
    if (isRelu) {
      for (let j = 0; j < n; j++) {
        let alive = false;
        for (let b = 0; b < B; b++) if (us[l][b * n + j] > 0) { alive = true; break; }
        if (!alive) dead++;
      }
    }
    let low = 0, ms = 0;
    for (let i = 0; i < sl.length; i++) {
      if (sl[i] < 0.1 * A.dfMax) low++;
      ms += sl[i];
    }
    layers.push({
      hStd: std(hs[l + 1]),
      acts: a,
      deadUnits: dead / n,
      lowSlope: low / sl.length,
      meanSlope: ms / sl.length,
      deltaNorm: deltaNorm[l],
      gradWNorm: gradWNorm[l],
      weightNorm: frob(Ws[l]),
      relGrad: gradWNorm[l] / frob(Ws[l]),
    });
  }

  return {
    cfg,
    std: sd,
    inputStd: std(X),
    layers,
    loss,
    ratioW: gradWNorm[0] / gradWNorm[L - 1],
    ratioDelta: deltaNorm[0] / deltaNorm[L - 1],
  };
}

/** Histogram of a set of values. Bounded activations use their natural range. */
export function histogram(values: Float64Array, bins: number, range?: [number, number]) {
  let lo = Infinity, hi = -Infinity;
  if (range) [lo, hi] = range;
  else {
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
    if (hi - lo < 1e-12) { const c = lo; lo = c - 0.5 * (Math.abs(c) || 1e-12); hi = c + 0.5 * (Math.abs(c) || 1e-12); }
  }
  const counts = new Array<number>(bins).fill(0);
  const w = (hi - lo) / bins;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    let k = Math.floor((v - lo) / w);
    if (k < 0) k = 0;
    if (k >= bins) k = bins - 1;
    counts[k]++;
  }
  return { counts, lo, hi };
}

/** Geometric-mean per-layer factor implied by an end-to-end ratio over (L − 1) layer transitions. */
export function perLayerFactor(ratio: number, L: number) {
  if (L < 2 || !(ratio > 0) || !Number.isFinite(ratio)) return NaN;
  return Math.pow(ratio, 1 / (L - 1));
}
