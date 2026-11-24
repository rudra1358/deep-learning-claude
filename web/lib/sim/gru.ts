/**
 * A tiny GRU (input size 1, hidden size H) and a vanilla RNN of the same size, used by the GRU simulator.
 * Pure math, no React — every number shown in the simulator comes from here.
 *
 * GRU equations (concatenation [h_{t-1}; x_t], reset applied before the matrix multiply as in Cho et al. 2014;
 * interpolation written with z = "fraction updated", as in the lesson):
 *   r_t = σ(W_r [h_{t-1}; x_t] + b_r)               reset gate
 *   h̃_t = tanh(W_h [r_t ⊙ h_{t-1}; x_t] + b_h)      candidate
 *   z_t = σ(W_z [h_{t-1}; x_t] + b_z)               update gate
 *   h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
 *
 * Vanilla RNN:  h_t = tanh(W [h_{t-1}; x_t] + b)
 */
import { gaussian, mulberry32 } from "./random";

export type GruGate = "r" | "z" | "h";

export interface GruWeights {
  hidden: number;
  input: number;
  /** Each W[gate] is H × (H + input): columns are [h_{t-1} ..., x_t ...]. */
  W: Record<GruGate, number[][]>;
  b: Record<GruGate, number[]>;
}

export interface GruStep {
  x: number[];
  hPrev: number[];
  pre: Record<GruGate, number[]>;
  r: number[];
  /** r ⊙ h_{t-1}: the part of the old state the candidate is allowed to see. */
  rh: number[];
  htilde: number[];
  z: number[];
  /** (1 − z) ⊙ h_{t-1} and z ⊙ h̃. */
  keep: number[];
  write: number[];
  h: number[];
}

export interface RnnWeights {
  hidden: number;
  W: number[][];
  b: number[];
}

export const sigmoid = (a: number) => 1 / (1 + Math.exp(-a));

function seededMatrix(rand: () => number, rows: number, cols: number, hidden: number, hScale: number, xScale: number) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, (_, j) => gaussian(rand) * (j < hidden ? hScale : xScale)));
}

/** Seeded GRU weights. The update-gate bias is the simulator's slider; other biases are 0. */
export function makeGruWeights(hidden: number, seed = 7, input = 1, hScale = 0.6, xScale = 2.0): GruWeights {
  const rand = mulberry32(seed);
  const m = () => seededMatrix(rand, hidden, hidden + input, hidden, hScale, xScale);
  const W = { r: m(), h: m(), z: m() };
  const zeros = () => new Array(hidden).fill(0);
  return { hidden, input, W, b: { r: zeros(), h: zeros(), z: zeros() } };
}

export function withUpdateBias(w: GruWeights, bz: number): GruWeights {
  return { ...w, b: { ...w.b, z: new Array(w.hidden).fill(bz) } };
}

/** Seeded vanilla-RNN weights with the same input scale as the GRU's candidate layer. */
export function makeRnnWeights(hidden: number, seed = 11, hScale = 0.5, xScale = 2.0): RnnWeights {
  const rand = mulberry32(seed);
  return { hidden, W: seededMatrix(rand, hidden, hidden + 1, hidden, hScale, xScale), b: new Array(hidden).fill(0) };
}

function affine(W: number[][], b: number[], v: number[]) {
  return W.map((row, r) => row.reduce((s, wv, k) => s + wv * v[k], b[r]));
}

export function gruStep(w: GruWeights, x: number[], hPrev: number[]): GruStep {
  const zin = [...hPrev, ...x];
  const preR = affine(w.W.r, w.b.r, zin);
  const r = preR.map(sigmoid);
  const rh = r.map((rv, k) => rv * hPrev[k]);
  const preH = affine(w.W.h, w.b.h, [...rh, ...x]);
  const htilde = preH.map(Math.tanh);
  const preZ = affine(w.W.z, w.b.z, zin);
  const z = preZ.map(sigmoid);
  const keep = z.map((zv, k) => (1 - zv) * hPrev[k]);
  const write = z.map((zv, k) => zv * htilde[k]);
  const h = keep.map((kv, k) => kv + write[k]);
  return { x, hPrev, pre: { r: preR, h: preH, z: preZ }, r, rh, htilde, z, keep, write, h };
}

/** Run the GRU over a scalar sequence from a zero initial state. */
export function runGru(w: GruWeights, seq: number[]): GruStep[] {
  let h = new Array(w.hidden).fill(0);
  const out: GruStep[] = [];
  for (const xv of seq) {
    const s = gruStep(w, [xv], h);
    out.push(s);
    h = s.h;
  }
  return out;
}

/** Hidden states h_1..h_T of the vanilla RNN. */
export function runRnn(w: RnnWeights, seq: number[]): number[][] {
  let h = new Array(w.hidden).fill(0);
  const out: number[][] = [];
  for (const xv of seq) {
    h = affine(w.W, w.b, [...h, xv]).map(Math.tanh);
    out.push(h);
  }
  return out;
}

const dist = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0));

/**
 * Memory probe: add `delta` to x_1 and report ‖h_t(perturbed) − h_t(original)‖₂ for every t.
 * A model that remembers x_1 keeps this difference large at later steps.
 */
export function memoryProbe(gru: GruWeights, rnn: RnnWeights, seq: number[], delta = 1) {
  const pert = seq.map((v, i) => (i === 0 ? v + delta : v));
  const g0 = runGru(gru, seq).map((s) => s.h);
  const g1 = runGru(gru, pert).map((s) => s.h);
  const r0 = runRnn(rnn, seq);
  const r1 = runRnn(rnn, pert);
  return { gru: g0.map((h, t) => dist(h, g1[t])), rnn: r0.map((h, t) => dist(h, r1[t])) };
}

/** Parse "1, 0, 0.5 -1" into numbers (invalid tokens skipped, values clamped to [−3, 3], length capped). */
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
export const GRU_SUBSTEPS = ["reset", "candidate", "update", "hidden"] as const;
export type GruSubStep = (typeof GRU_SUBSTEPS)[number];
