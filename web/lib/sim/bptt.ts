/**
 * Backpropagation through time for the tiny vanilla RNN of ./rnn.ts (input size 1, scalar output).
 * Pure math, no React — every number the BPTT simulator shows comes from here.
 *
 * Loss ("last"):  L = ½ (y_T − y*)²
 * Loss ("every"): L = Σ_t ½ (y_t − y*_t)²
 *
 * Backward recursion (full BPTT), t = T … 1:
 *   ∂L/∂h_t = W_hy e_t  +  W_hhᵀ δ_{t+1}            (e_t = y_t − y*_t, 0 if no loss at t)
 *   δ_t     = ∂L/∂h_t ⊙ (1 − h_t²)                  (= ∂L/∂z_t)
 *   ∂L/∂W_hh += δ_t h_{t−1}ᵀ,  ∂L/∂W_xh += δ_t x_tᵀ,  ∂L/∂b += δ_t
 *
 * Truncation with window k: the loss term at step s sends gradient back through at most k steps
 * (time steps s−k+1 … s); contributions from further back are dropped.
 */
import { gaussian, mulberry32 } from "./random";
import { frob, makeRnnWeights, matTVec, norm2, runRnn, spectralNorm, spectralRadius, type Mat, type RnnWeights } from "./rnn";

export type LossMode = "last" | "every";

export interface BpttConfig {
  T: number;
  rho: number;
  mode: LossMode;
  /** truncation window (k ≥ T means full BPTT) */
  k: number;
  /** "tanh" (the vanilla RNN) or "linear" (no squashing: the gradient is multiplied by W_hhᵀ exactly) */
  act?: "tanh" | "linear";
  /** scale of the inputs x_t (small inputs keep tanh in its near-linear region) */
  xScale?: number;
  hidden?: number;
  seed?: number;
}

export interface Grads {
  Wxh: Mat;
  Whh: Mat;
  b: number[];
  Why: number[];
  by: number;
}

export interface BpttResult {
  w: RnnWeights;
  rhoActual: number;
  whhNorm: number;
  xs: number[];
  targets: number[];
  /** h_0 … h_T (length T+1) */
  hs: number[][];
  ys: number[];
  /** per-step loss terms (0 where there is no loss) */
  lossTerms: number[];
  loss: number;
  /** index t−1 ↔ time step t (full BPTT) */
  dh: number[][];
  dz: number[][];
  /** the part of ∂L/∂h_t coming directly from y_t, and the part arriving from h_{t+1} */
  dhFromY: number[][];
  dhFromNext: number[][];
  dhNorm: number[];
  /** ‖W_hhᵀ diag(1 − h_t²)‖₂ — the per-step factor that multiplies the gradient on its way to h_{t−1} */
  factor: number[];
  /** ‖δ_t h_{t−1}ᵀ‖_F — contribution of step t to ∂L/∂W_hh, full and with truncation */
  contribNorm: number[];
  contribKeptNorm: number[];
  /** the per-step dW_hh contribution matrices themselves (index t−1 ↔ step t) */
  contribFull: Mat[];
  contribKept: Mat[];
  gradFull: Grads;
  gradTrunc: Grads;
  normFull: number;
  normTrunc: number;
  whhGradFull: number;
  whhGradTrunc: number;
}

const zeros = (n: number) => new Array(n).fill(0);
const zerosM = (r: number, c: number) => Array.from({ length: r }, () => zeros(c));

function gradNorm(g: Grads) {
  const sq = (v: number[]) => v.reduce((s, x) => s + x * x, 0);
  return Math.sqrt(frob(g.Wxh) ** 2 + frob(g.Whh) ** 2 + sq(g.b) + sq(g.Why) + g.by * g.by);
}

/** Accumulate gradients for one window. Returns per-step dW_hh contribution matrices. */
function backwardWindowed(w: RnnWeights, xs: number[], hs: number[][], errs: number[], k: number, linear: boolean) {
  const H = w.hidden;
  const T = xs.length;
  const g: Grads = { Wxh: zerosM(H, 1), Whh: zerosM(H, H), b: zeros(H), Why: zeros(H), by: 0 };
  const contrib: Mat[] = Array.from({ length: T }, () => zerosM(H, H));
  for (let s = 1; s <= T; s++) {
    const e = errs[s - 1];
    if (e === 0) continue;
    for (let i = 0; i < H; i++) g.Why[i] += e * hs[s][i];
    g.by += e;
    let dh = w.Why.map((v) => v * e);
    for (let t = s; t >= Math.max(1, s - k + 1); t--) {
      const dz = dh.map((v, i) => (linear ? v : v * (1 - hs[t][i] ** 2)));
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          g.Whh[i][j] += dz[i] * hs[t - 1][j];
          contrib[t - 1][i][j] += dz[i] * hs[t - 1][j];
        }
        g.Wxh[i][0] += dz[i] * xs[t - 1];
        g.b[i] += dz[i];
      }
      dh = matTVec(w.Whh, dz);
    }
  }
  return { g, contrib };
}

export function runBptt(cfg: BpttConfig): BpttResult {
  const H = cfg.hidden ?? 4;
  const seed = cfg.seed ?? 3;
  const T = cfg.T;
  const w = makeRnnWeights(H, 1, cfg.rho, seed);
  const rand = mulberry32(seed * 7919 + 1);
  const allX = Array.from({ length: 64 }, () => gaussian(rand));
  const allY = Array.from({ length: 64 }, () => 0.5 * gaussian(rand));
  const xs = allX.slice(0, T).map((v) => v * (cfg.xScale ?? 1));
  const targets = cfg.mode === "last" ? xs.map((_, t) => (t === T - 1 ? 1 : 0)) : allY.slice(0, T);

  const linear = cfg.act === "linear";
  const steps = runRnn(w, xs.map((v) => [v]), linear);
  const hs = [zeros(H), ...steps.map((s) => s.h)];
  const ys = steps.map((s) => s.y);
  const hasLoss = (t: number) => cfg.mode === "every" || t === T; // t is 1-based
  const errs = ys.map((y, i) => (hasLoss(i + 1) ? y - targets[i] : 0));
  const lossTerms = ys.map((y, i) => (hasLoss(i + 1) ? 0.5 * (y - targets[i]) ** 2 : 0));
  const loss = lossTerms.reduce((a, b) => a + b, 0);

  // Full BPTT, single right-to-left sweep (for ∂L/∂h_t and the per-step factors)
  const dh: number[][] = new Array(T);
  const dz: number[][] = new Array(T);
  const dhFromY: number[][] = new Array(T);
  const dhFromNext: number[][] = new Array(T);
  const factor: number[] = new Array(T);
  let next = zeros(H);
  for (let t = T; t >= 1; t--) {
    const fromY = w.Why.map((v) => v * errs[t - 1]);
    const total = fromY.map((v, i) => v + next[i]);
    const d = hs[t].map((v) => (linear ? 1 : 1 - v * v));
    const dzt = total.map((v, i) => v * d[i]);
    dh[t - 1] = total;
    dz[t - 1] = dzt;
    dhFromY[t - 1] = fromY;
    dhFromNext[t - 1] = next;
    // M = W_hhᵀ diag(d):  M[i][j] = W_hh[j][i] · d[j]
    const M = w.Whh[0].map((_, i) => w.Whh.map((row, j) => row[i] * d[j]));
    factor[t - 1] = spectralNorm(M);
    next = matTVec(w.Whh, dzt);
  }

  const full = backwardWindowed(w, xs, hs, errs, T, linear);
  const trunc = backwardWindowed(w, xs, hs, errs, Math.max(1, Math.min(cfg.k, T)), linear);

  return {
    w,
    rhoActual: spectralRadius(w.Whh),
    whhNorm: spectralNorm(w.Whh),
    xs,
    targets,
    hs,
    ys,
    lossTerms,
    loss,
    dh,
    dz,
    dhFromY,
    dhFromNext,
    dhNorm: dh.map(norm2),
    factor,
    contribNorm: full.contrib.map(frob),
    contribKeptNorm: trunc.contrib.map(frob),
    contribFull: full.contrib,
    contribKept: trunc.contrib,
    gradFull: full.g,
    gradTrunc: trunc.g,
    normFull: gradNorm(full.g),
    normTrunc: gradNorm(trunc.g),
    whhGradFull: frob(full.g.Whh),
    whhGradTrunc: frob(trunc.g.Whh),
  };
}

/** Frobenius norm of the sum of the given matrices (used for "accumulated so far" read-outs). */
export function sumNorm(ms: Mat[]): number {
  if (!ms.length) return 0;
  const acc = ms[0].map((r) => r.map(() => 0));
  for (const m of ms) m.forEach((r, i) => r.forEach((v, j) => (acc[i][j] += v)));
  return frob(acc);
}

/** Global-norm clipping: the gradient is rescaled by min(1, c / ‖g‖). */
export function clipScale(norm: number, threshold: number) {
  return norm > threshold && norm > 0 ? threshold / norm : 1;
}
