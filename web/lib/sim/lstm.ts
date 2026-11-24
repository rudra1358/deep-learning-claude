/**
 * A tiny LSTM (input size 1, hidden size H) used by the LSTM simulator.
 * Pure math, no React — every number shown in the simulator comes from here.
 *
 * Equations (concatenation z_t = [h_{t-1}; x_t]):
 *   f_t = σ(W_f z_t + b_f)        forget gate
 *   i_t = σ(W_i z_t + b_i)        input gate
 *   g_t = tanh(W_g z_t + b_g)     candidate cell state
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
 *   o_t = σ(W_o z_t + b_o)        output gate
 *   h_t = o_t ⊙ tanh(c_t)
 */
import { gaussian, mulberry32 } from "./random";

export type GateName = "f" | "i" | "g" | "o";

export interface LstmWeights {
  hidden: number;
  input: number;
  /** Each W[gate] is H × (H + input): columns are [h_{t-1} ..., x_t ...]. */
  W: Record<GateName, number[][]>;
  b: Record<GateName, number[]>;
}

export interface LstmStep {
  x: number[];
  hPrev: number[];
  cPrev: number[];
  /** Pre-activations W z + b for each gate. */
  pre: Record<GateName, number[]>;
  f: number[];
  i: number[];
  g: number[];
  o: number[];
  /** f ⊙ c_{t-1} (what is kept) and i ⊙ g (what is written). */
  kept: number[];
  written: number[];
  c: number[];
  tanhC: number[];
  h: number[];
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/**
 * Seeded weights. The forget-gate bias is set separately (it is the simulator's slider),
 * other biases are 0 so the effect of b_f is easy to isolate.
 */
export function makeLstmWeights(hidden: number, seed = 1, input = 1, hScale = 0.6, xScale = 2.0): LstmWeights {
  const rand = mulberry32(seed);
  // Recurrent columns are kept small and input columns large, so a single input pulse is clearly visible.
  const mat = () => Array.from({ length: hidden }, () => Array.from({ length: hidden + input }, (_, j) => gaussian(rand) * (j < hidden ? hScale : xScale)));
  const W = { f: mat(), i: mat(), g: mat(), o: mat() };
  const zeros = () => new Array(hidden).fill(0);
  return { hidden, input, W, b: { f: zeros(), i: zeros(), g: zeros(), o: zeros() } };
}

export function withForgetBias(w: LstmWeights, bf: number): LstmWeights {
  return { ...w, b: { ...w.b, f: new Array(w.hidden).fill(bf) } };
}

function affine(W: number[][], b: number[], z: number[]) {
  return W.map((row, r) => row.reduce((s, wv, k) => s + wv * z[k], b[r]));
}

export function lstmStep(w: LstmWeights, x: number[], hPrev: number[], cPrev: number[]): LstmStep {
  const z = [...hPrev, ...x];
  const pre = {
    f: affine(w.W.f, w.b.f, z),
    i: affine(w.W.i, w.b.i, z),
    g: affine(w.W.g, w.b.g, z),
    o: affine(w.W.o, w.b.o, z),
  };
  const f = pre.f.map(sigmoid);
  const i = pre.i.map(sigmoid);
  const g = pre.g.map(Math.tanh);
  const o = pre.o.map(sigmoid);
  const kept = f.map((fv, k) => fv * cPrev[k]);
  const written = i.map((iv, k) => iv * g[k]);
  const c = kept.map((kv, k) => kv + written[k]);
  const tanhC = c.map(Math.tanh);
  const h = o.map((ov, k) => ov * tanhC[k]);
  return { x, hPrev, cPrev, pre, f, i, g, o, kept, written, c, tanhC, h };
}

/** Run the LSTM over a scalar sequence from zero initial state. */
export function runLstm(w: LstmWeights, seq: number[]): LstmStep[] {
  let h = new Array(w.hidden).fill(0);
  let c = new Array(w.hidden).fill(0);
  const out: LstmStep[] = [];
  for (const xv of seq) {
    const s = lstmStep(w, [xv], h, c);
    out.push(s);
    h = s.h;
    c = s.c;
  }
  return out;
}

/** Parse "1, 0, 0.5 -1" into numbers (invalid tokens are skipped, length capped). */
export function parseSequence(text: string, maxLen = 12): number[] {
  return text
    .split(/[\s,;]+/)
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((v) => Number.isFinite(v))
    .map((v) => Math.max(-3, Math.min(3, v)))
    .slice(0, maxLen);
}

/** Sub-steps the simulator walks through inside each time step. */
export const SUBSTEPS = ["forget", "input", "cell", "output", "hidden"] as const;
export type SubStep = (typeof SUBSTEPS)[number];
