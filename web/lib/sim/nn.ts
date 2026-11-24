/**
 * A tiny, fully-transparent multilayer perceptron used by the Neural Network Visualizer.
 * Pure functions, no React — every number the simulator displays is computed here.
 *
 * Conventions (match the backpropagation lesson):
 *   layer l = 1..L,  a^0 = x
 *   z^l = W^l a^{l-1} + b^l          W^l has shape (n_l, n_{l-1})
 *   a^l = f_l(z^l)
 *   δ^l = ∂L/∂z^l
 *   ∂L/∂W^l = δ^l (a^{l-1})^T,   ∂L/∂b^l = δ^l
 *   δ^{l-1} = ((W^l)^T δ^l) ⊙ f'_{l-1}(z^{l-1})
 */

import { gaussian, mulberry32 } from "./random";

export type Act = "sigmoid" | "tanh" | "relu" | "leaky_relu" | "linear";
export type LossKind = "mse" | "bce";

export const ACTS: Record<Act, { label: string; f: (z: number) => number; df: (z: number) => number; sym: string }> = {
  sigmoid: { label: "Sigmoid", f: (z) => 1 / (1 + Math.exp(-z)), df: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }, sym: String.raw`\sigma` },
  tanh: { label: "Tanh", f: Math.tanh, df: (z) => 1 - Math.tanh(z) ** 2, sym: String.raw`\tanh` },
  relu: { label: "ReLU", f: (z) => Math.max(0, z), df: (z) => (z > 0 ? 1 : 0), sym: String.raw`\mathrm{ReLU}` },
  leaky_relu: { label: "Leaky ReLU", f: (z) => (z > 0 ? z : 0.01 * z), df: (z) => (z > 0 ? 1 : 0.01), sym: String.raw`\mathrm{LeakyReLU}` },
  linear: { label: "Linear", f: (z) => z, df: () => 1, sym: String.raw`\mathrm{id}` },
};

export interface Net {
  sizes: number[]; // [n0, n1, ..., nL]
  acts: Act[]; // length L — activation of each non-input layer
  W: number[][][]; // W[l-1] is W^l, shape (n_l, n_{l-1})
  b: number[][]; // b[l-1] is b^l
}

export interface Trace {
  a: number[][]; // a[0] = x, a[l] for l=1..L
  z: number[][]; // z[0] unused (empty), z[l]
  loss: number;
  dLda: number[]; // ∂L/∂a^L
  delta: number[][]; // delta[l] for l=1..L (delta[0] empty)
  dW: number[][][]; // dW[l-1] = ∂L/∂W^l
  db: number[][];
}

/** Xavier/Glorot-normal initialisation (variance 2/(fan_in+fan_out)), small zero-ish biases. */
export function initNet(sizes: number[], acts: Act[], seed = 1): Net {
  const rand = mulberry32(seed);
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 1; l < sizes.length; l++) {
    const std = Math.sqrt(2 / (sizes[l] + sizes[l - 1]));
    W.push(Array.from({ length: sizes[l] }, () => Array.from({ length: sizes[l - 1] }, () => round(gaussian(rand) * std))));
    b.push(Array.from({ length: sizes[l] }, () => round(gaussian(rand) * 0.1)));
  }
  return { sizes, acts, W, b };
}

const round = (v: number) => Math.round(v * 100) / 100;

export function lossValue(kind: LossKind, a: number[], y: number[]) {
  const n = a.length;
  if (kind === "mse") return a.reduce((s, ai, i) => s + (ai - y[i]) ** 2, 0) / n;
  const eps = 1e-12;
  return -a.reduce((s, ai, i) => s + y[i] * Math.log(Math.max(ai, eps)) + (1 - y[i]) * Math.log(Math.max(1 - ai, eps)), 0) / n;
}

/** ∂L/∂a^L for the chosen loss (mean over output units). */
export function lossGrad(kind: LossKind, a: number[], y: number[]) {
  const n = a.length;
  if (kind === "mse") return a.map((ai, i) => (2 * (ai - y[i])) / n);
  const eps = 1e-12;
  return a.map((ai, i) => ((ai - y[i]) / Math.max(ai * (1 - ai), eps)) / n);
}

export function forward(net: Net, x: number[]) {
  const a: number[][] = [x.slice()];
  const z: number[][] = [[]];
  for (let l = 1; l < net.sizes.length; l++) {
    const W = net.W[l - 1], b = net.b[l - 1], f = ACTS[net.acts[l - 1]].f;
    const zl = W.map((row, j) => row.reduce((s, w, k) => s + w * a[l - 1][k], 0) + b[j]);
    z.push(zl);
    a.push(zl.map(f));
  }
  return { a, z };
}

export function forwardBackward(net: Net, x: number[], y: number[], kind: LossKind): Trace {
  const { a, z } = forward(net, x);
  const L = net.sizes.length - 1;
  const loss = lossValue(kind, a[L], y);
  const dLda = lossGrad(kind, a[L], y);
  const delta: number[][] = Array.from({ length: L + 1 }, () => []);
  const dW: number[][][] = [];
  const db: number[][] = [];
  delta[L] = z[L].map((zj, j) => dLda[j] * ACTS[net.acts[L - 1]].df(zj));
  for (let l = L; l >= 1; l--) {
    dW[l - 1] = delta[l].map((dj) => a[l - 1].map((ak) => dj * ak));
    db[l - 1] = delta[l].slice();
    if (l > 1) {
      const W = net.W[l - 1];
      const dfp = ACTS[net.acts[l - 2]].df;
      delta[l - 1] = z[l - 1].map((zk, k) => W.reduce((s, row, j) => s + row[k] * delta[l][j], 0) * dfp(zk));
    }
  }
  return { a, z, loss, dLda, delta, dW, db };
}

export function applyUpdate(net: Net, tr: Trace, lr: number): Net {
  return {
    ...net,
    W: net.W.map((Wl, l) => Wl.map((row, j) => row.map((w, k) => w - lr * tr.dW[l][j][k]))),
    b: net.b.map((bl, l) => bl.map((bj, j) => bj - lr * tr.db[l][j])),
  };
}

/** Central-difference numerical gradient of a single weight — used to verify backprop. */
export function numericalGrad(net: Net, x: number[], y: number[], kind: LossKind, l: number, j: number, k: number, h = 1e-5) {
  const bump = (d: number): Net => ({ ...net, W: net.W.map((Wl, li) => Wl.map((row, ji) => row.map((w, ki) => (li === l && ji === j && ki === k ? w + d : w)))) });
  const lp = lossValue(kind, forward(bump(h), x).a.at(-1)!, y);
  const lm = lossValue(kind, forward(bump(-h), x).a.at(-1)!, y);
  return (lp - lm) / (2 * h);
}

/** The ordered list of animation steps for one forward+backward+update cycle. */
export type NetStep =
  | { kind: "idle" }
  | { kind: "forward"; layer: number; neuron: number }
  | { kind: "loss" }
  | { kind: "backward"; layer: number; neuron: number }
  | { kind: "update" };

export function buildSteps(sizes: number[]): NetStep[] {
  const steps: NetStep[] = [{ kind: "idle" }];
  const L = sizes.length - 1;
  for (let l = 1; l <= L; l++) for (let j = 0; j < sizes[l]; j++) steps.push({ kind: "forward", layer: l, neuron: j });
  steps.push({ kind: "loss" });
  for (let l = L; l >= 1; l--) for (let j = 0; j < sizes[l]; j++) steps.push({ kind: "backward", layer: l, neuron: j });
  steps.push({ kind: "update" });
  return steps;
}

/** Has neuron (l, j) been reached by the current step (for progressive reveal)? */
export function revealed(steps: NetStep[], idx: number, phase: "forward" | "backward", l: number, j: number) {
  for (let i = 0; i <= idx; i++) {
    const s = steps[i];
    if (s.kind === phase && s.layer === l && s.neuron === j) return true;
  }
  return false;
}
