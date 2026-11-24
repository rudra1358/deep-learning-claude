/**
 * Pure math for the dropout simulator (no React).
 *
 * Network: 8 inputs → 10 ReLU hidden units → 1 output, with fixed seeded weights and a fixed seeded input.
 *   h = ReLU(W1 x + b1)                       (hidden activations, deterministic)
 *   train:  h̃ = m ⊙ h · s_train,  m_i ~ Bernoulli(1 − p)
 *   eval :  h̃ = h · s_eval
 *   z = w2 · h̃ + b2,  y = z (linear output) or σ(z) (sigmoid output)
 *
 * Scaling schemes:
 *   inverted dropout: s_train = 1/(1 − p), s_eval = 1          (what every framework does)
 *   classic dropout : s_train = 1,         s_eval = (1 − p)    (Srivastava et al. 2014: scale weights at test time)
 * In both schemes E_m[z_train] = z_eval exactly, because z is linear in h̃.
 *
 * Masks: one uniform u_{k,i} ∈ [0,1) is drawn per (mask index k, unit i) from a seeded PRNG, and unit i is
 * kept in mask k iff u_{k,i} ≥ p. Changing p therefore re-thresholds the same random numbers instead of
 * drawing a new sequence, so the effect of p is visible in isolation.
 */

import { gaussian, mulberry32 } from "./random";

export const N_IN = 8;
export const N_HID = 10;
export const MAX_MASKS = 300;

export type Scheme = "inverted" | "classic";
export type OutAct = "linear" | "sigmoid";

export interface DropNet {
  x: number[];
  W1: number[][];
  b1: number[];
  w2: number[];
  b2: number;
  z1: number[]; // pre-activations
  h: number[]; // ReLU activations (no dropout)
}

export function makeNet(seed = 5): DropNet {
  const r = mulberry32(seed);
  const x = Array.from({ length: N_IN }, () => +(r() * 1.0).toFixed(2));
  const W1 = Array.from({ length: N_HID }, () => Array.from({ length: N_IN }, () => +(gaussian(r) * 0.55).toFixed(2)));
  const b1 = Array.from({ length: N_HID }, () => +(0.25 + 0.3 * r()).toFixed(2));
  const w2 = Array.from({ length: N_HID }, () => +(gaussian(r) * 0.45).toFixed(2));
  const b2 = 0.1;
  const z1 = W1.map((row, j) => row.reduce((s, w, i) => s + w * x[i], b1[j]));
  const h = z1.map((z) => Math.max(0, z));
  return { x, W1, b1, w2, b2, z1, h };
}

/** Uniform numbers u[k][i] for mask k (0-based), unit i. */
export function makeUniforms(seed: number, n = MAX_MASKS): number[][] {
  const r = mulberry32(seed);
  return Array.from({ length: n }, () => Array.from({ length: N_HID }, () => r()));
}

export const maskFrom = (u: number[], p: number): number[] => u.map((v) => (v >= p ? 1 : 0));

export const trainScale = (p: number, scheme: Scheme) => (scheme === "inverted" ? 1 / (1 - p) : 1);
export const evalScale = (p: number, scheme: Scheme) => (scheme === "inverted" ? 1 : 1 - p);

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const act = (z: number, out: OutAct) => (out === "sigmoid" ? sigmoid(z) : z);

export interface Pass {
  ht: number[]; // h̃ after mask and scaling
  contrib: number[]; // w2_i · h̃_i
  z: number;
  y: number;
}

export function forward(net: DropNet, mask: number[] | null, p: number, scheme: Scheme, out: OutAct): Pass {
  const ht = net.h.map((h, i) => (mask ? mask[i] * h * trainScale(p, scheme) : h * evalScale(p, scheme)));
  const contrib = ht.map((v, i) => net.w2[i] * v);
  const z = contrib.reduce((s, v) => s + v, net.b2);
  return { ht, contrib, z, y: act(z, out) };
}

/** Exact expectation of the train-mode output over all 2^N_HID masks (1024 terms). */
export function exactExpectation(net: DropNet, p: number, scheme: Scheme, out: OutAct): number {
  let e = 0;
  const n = N_HID;
  for (let bits = 0; bits < 1 << n; bits++) {
    let prob = 1;
    const mask: number[] = [];
    for (let i = 0; i < n; i++) {
      const keep = (bits >> i) & 1;
      mask.push(keep);
      prob *= keep ? 1 - p : p;
    }
    if (prob === 0) continue;
    e += prob * forward(net, mask, p, scheme, out).y;
  }
  return e;
}

/** Var[z_train] over masks: Σ w_i² h_i² · s² · p(1−p). For inverted dropout this is p/(1−p) Σ w_i² h_i². */
export function preActVariance(net: DropNet, p: number, scheme: Scheme): number {
  const s = trainScale(p, scheme);
  return net.h.reduce((acc, h, i) => acc + net.w2[i] ** 2 * h * h, 0) * s * s * p * (1 - p);
}

export interface Series {
  ys: number[]; // train-mode output for masks 1..k
  mean: number[]; // running mean after each mask
}

export function sampleSeries(net: DropNet, U: number[][], k: number, p: number, scheme: Scheme, out: OutAct): Series {
  const ys: number[] = [];
  const mean: number[] = [];
  let s = 0;
  for (let j = 0; j < k; j++) {
    const y = forward(net, maskFrom(U[j], p), p, scheme, out).y;
    ys.push(y);
    s += y;
    mean.push(s / (j + 1));
  }
  return { ys, mean };
}
