/**
 * Pure math for ConvVariantsSim: dilated, transposed and depthwise-separable convolution.
 * No React, no DOM — safe to unit-test.
 *
 * Conventions (same as lib/sim/conv.ts): deep-learning "convolution" = cross-correlation,
 * grids are row-major grid[row][col], square kernels.
 */

import { pad2d, zeros, type Grid } from "./conv";

/* ------------------------------------------------------------------ */
/* Dilated convolution                                                 */
/* ------------------------------------------------------------------ */

/** Span covered by a k-tap kernel with dilation d: k + (k − 1)(d − 1) = d(k − 1) + 1. */
export function effectiveKernel(k: number, d: number): number {
  return d * (k - 1) + 1;
}

/** floor((n + 2p − d(k − 1) − 1)/s) + 1, or 0 if the dilated kernel does not fit. */
export function dilatedOutputSize(n: number, k: number, p: number, s: number, d: number): number {
  const span = n + 2 * p - effectiveKernel(k, d);
  if (span < 0 || s < 1) return 0;
  return Math.floor(span / s) + 1;
}

/** Receptive field of a stack of stride-1 k×k layers with the given dilations: 1 + (k − 1)·Σd. */
export function stackReceptiveField(k: number, dilations: number[]): number {
  return 1 + (k - 1) * dilations.reduce((a, b) => a + b, 0);
}

export interface DilatedStep {
  i: number;
  j: number;
  /** tap positions in the PADDED input, kernel-aligned (k×k) */
  taps: { r: number; c: number; v: number; w: number; prod: number }[][];
  sum: number;
  /** top-left of the dilated window in the padded input */
  r0: number;
  c0: number;
  span: number;
}

export function dilatedStep(xPadded: Grid, kernel: Grid, i: number, j: number, d: number, s: number): DilatedStep {
  const k = kernel.length;
  const r0 = i * s;
  const c0 = j * s;
  let sum = 0;
  const taps = kernel.map((row, a) =>
    row.map((w, b) => {
      const r = r0 + a * d;
      const c = c0 + b * d;
      const v = xPadded[r][c];
      const prod = v * w;
      sum += prod;
      return { r, c, v, w, prod };
    }),
  );
  return { i, j, taps, sum, r0, c0, span: effectiveKernel(k, d) };
}

/** Dilated cross-correlation of a single-channel grid. */
export function dilatedConv(x: Grid, kernel: Grid, d: number, p = 0, s = 1): Grid {
  const k = kernel.length;
  const xp = pad2d(x, p);
  const H = dilatedOutputSize(x.length, k, p, s, d);
  const W = dilatedOutputSize(x[0]?.length ?? 0, k, p, s, d);
  const out = zeros(H, W);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) out[i][j] = dilatedStep(xp, kernel, i, j, d, s).sum;
  return out;
}

/* ------------------------------------------------------------------ */
/* Transposed convolution (scatter-add)                                */
/* ------------------------------------------------------------------ */

/** (n − 1)s − 2p + d(k − 1) + output_padding + 1 (PyTorch's ConvTranspose2d formula). */
export function transposedOutputSize(n: number, k: number, s: number, p: number, op = 0, d = 1): number {
  return (n - 1) * s - 2 * p + d * (k - 1) + op + 1;
}

export interface TransposedState {
  /** full canvas, size (n − 1)s + k + op, after `stamps` stamps */
  canvas: Grid;
  /** how many kernel stamps have covered each canvas cell so far */
  count: Grid;
  /** canvas with p rows/cols cropped from every side (the layer's output) */
  out: Grid;
  full: number;
}

/**
 * Scatter-add transposed convolution, stopped after the first `stamps` input pixels (row-major).
 * Input pixel (i, j) adds x[i][j]·K into the canvas block whose top-left corner is (i·s, j·s).
 */
export function transposedConv(x: Grid, kernel: Grid, s: number, p: number, op: number, stamps = Infinity): TransposedState {
  const n = x.length;
  const m = x[0]?.length ?? 0;
  const k = kernel.length;
  const full = (n - 1) * s + k + op;
  const fullW = (m - 1) * s + k + op;
  const canvas = zeros(full, fullW);
  const count = zeros(full, fullW);
  let done = 0;
  outer: for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      if (done >= stamps) break outer;
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++) {
          canvas[i * s + a][j * s + b] += x[i][j] * kernel[a][b];
          count[i * s + a][j * s + b] += 1;
        }
      done++;
    }
  const out: Grid = [];
  for (let r = p; r < full - p; r++) out.push(canvas[r].slice(p, fullW - p));
  return { canvas, count, out, full };
}

/* ------------------------------------------------------------------ */
/* Depthwise-separable convolution                                     */
/* ------------------------------------------------------------------ */

/** Depthwise: channel c is correlated with its own kernel kernels[c] ("same" padding, stride 1). */
export function depthwise(x: Grid[], kernels: Grid[]): Grid[] {
  return x.map((ch, c) => dilatedConv(ch, kernels[c], 1, (kernels[c].length - 1) / 2, 1));
}

/** Pointwise (1×1): out[o] = Σ_c P[o][c] · x[c], pixel by pixel. */
export function pointwise(x: Grid[], P: number[][]): Grid[] {
  const H = x[0].length;
  const W = x[0][0].length;
  return P.map((row) => {
    const out = zeros(H, W);
    row.forEach((w, c) => {
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) out[i][j] += w * x[c][i][j];
    });
    return out;
  });
}

export interface ConvCosts {
  standardParams: number;
  depthwiseParams: number;
  pointwiseParams: number;
  separableParams: number;
  standardMacs: number;
  depthwiseMacs: number;
  pointwiseMacs: number;
  separableMacs: number;
  ratio: number;
  formulaRatio: number;
}

/** Weights (no biases) and multiply-adds of a stride-1, "same"-padded layer on an H×W map. */
export function convCosts(cin: number, cout: number, k: number, h: number, w: number): ConvCosts {
  const hw = h * w;
  const standardParams = cout * cin * k * k;
  const depthwiseParams = cin * k * k;
  const pointwiseParams = cin * cout;
  const separableParams = depthwiseParams + pointwiseParams;
  return {
    standardParams,
    depthwiseParams,
    pointwiseParams,
    separableParams,
    standardMacs: standardParams * hw,
    depthwiseMacs: depthwiseParams * hw,
    pointwiseMacs: pointwiseParams * hw,
    separableMacs: separableParams * hw,
    ratio: separableParams / standardParams,
    formulaRatio: 1 / cout + 1 / (k * k),
  };
}
