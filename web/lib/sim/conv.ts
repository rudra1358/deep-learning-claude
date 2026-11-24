/**
 * Pure convolution / pooling math used by ConvolutionSim and CnnPipelineSim.
 * No React, no DOM — safe to unit-test.
 *
 * Conventions:
 *  - "convolution" here means what every deep-learning library computes: cross-correlation
 *    (the kernel is NOT flipped).  `flipKernel` gives the textbook convolution.
 *  - Grids are row-major: grid[row][col].
 *  - Output size along one axis: floor((n + 2p - k) / s) + 1.
 */

export type Grid = number[][];

/** floor((n + 2p - k) / s) + 1, or 0 if the kernel does not fit even once. */
export function outputSize(n: number, k: number, p: number, s: number): number {
  const span = n + 2 * p - k;
  if (span < 0 || s < 1) return 0;
  return Math.floor(span / s) + 1;
}

export function zeros(h: number, w: number): Grid {
  return Array.from({ length: h }, () => new Array<number>(w).fill(0));
}

/** Zero-pad a grid by p cells on every side. */
export function pad2d(x: Grid, p: number): Grid {
  if (p <= 0) return x.map((r) => r.slice());
  const h = x.length;
  const w = x[0]?.length ?? 0;
  const out = zeros(h + 2 * p, w + 2 * p);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) out[i + p][j + p] = x[i][j];
  return out;
}

/** Rotate a kernel by 180° (turns cross-correlation into true convolution). */
export function flipKernel(k: Grid): Grid {
  return k.slice().reverse().map((r) => r.slice().reverse());
}

export interface ConvOptions {
  stride?: number;
  padding?: number;
  bias?: number;
}

/** Single-channel cross-correlation with stride, zero padding and bias. */
export function conv2d(x: Grid, kernel: Grid, opts: ConvOptions = {}): Grid {
  const s = opts.stride ?? 1;
  const p = opts.padding ?? 0;
  const b = opts.bias ?? 0;
  const k = kernel.length;
  const xp = pad2d(x, p);
  const H = outputSize(x.length, k, p, s);
  const W = outputSize(x[0]?.length ?? 0, k, p, s);
  const out = zeros(H, W);
  for (let i = 0; i < H; i++)
    for (let j = 0; j < W; j++) {
      let acc = 0;
      for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) acc += xp[i * s + a][j * s + c] * kernel[a][c];
      out[i][j] = acc + b;
    }
  return out;
}

/**
 * Multi-channel convolution.
 * x: [C_in][H][W], w: [C_out][C_in][k][k], bias: [C_out]  →  [C_out][H_out][W_out]
 */
export function conv2dMulti(x: Grid[], w: Grid[][], bias: number[] = [], opts: Omit<ConvOptions, "bias"> = {}): Grid[] {
  return w.map((filter, co) => {
    if (filter.length !== x.length) throw new Error(`kernel depth ${filter.length} ≠ input channels ${x.length}`);
    // sum of per-channel 2-D correlations = one 3-D correlation over (C_in, k, k)
    const out = conv2d(x[0], filter[0], opts);
    for (let ci = 1; ci < filter.length; ci++) {
      const part = conv2d(x[ci], filter[ci], opts);
      for (let i = 0; i < part.length; i++) for (let j = 0; j < part[i].length; j++) out[i][j] += part[i][j];
    }
    const b = bias[co] ?? 0;
    return out.map((r) => r.map((v) => v + b));
  });
}

/** Number of learnable parameters of a conv layer: C_out·C_in·k·k (+ C_out biases). */
export function convParams(cIn: number, cOut: number, k: number, bias = true): number {
  return cOut * cIn * k * k + (bias ? cOut : 0);
}

export const relu = (x: Grid): Grid => x.map((r) => r.map((v) => (v > 0 ? v : 0)));

/** Max or average pooling (no padding). */
export function pool2d(x: Grid, size = 2, stride = 2, mode: "max" | "avg" = "max"): Grid {
  const H = outputSize(x.length, size, 0, stride);
  const W = outputSize(x[0]?.length ?? 0, size, 0, stride);
  const out = zeros(H, W);
  for (let i = 0; i < H; i++)
    for (let j = 0; j < W; j++) {
      let m = -Infinity;
      let sum = 0;
      for (let a = 0; a < size; a++)
        for (let c = 0; c < size; c++) {
          const v = x[i * stride + a][j * stride + c];
          if (v > m) m = v;
          sum += v;
        }
      out[i][j] = mode === "max" ? m : sum / (size * size);
    }
  return out;
}

/** Everything needed to animate one output position of a single-channel convolution. */
export interface ConvStep {
  /** output coordinates */
  i: number;
  j: number;
  /** top-left corner of the window in the PADDED input */
  r: number;
  c: number;
  /** window values, kernel-aligned (k×k) */
  window: Grid;
  /** element-wise products window[a][c] * kernel[a][c] */
  products: Grid;
  sum: number;
  /** sum + bias (pre-activation) */
  value: number;
}

export function convStep(xPadded: Grid, kernel: Grid, i: number, j: number, stride: number, bias: number): ConvStep {
  const k = kernel.length;
  const r = i * stride;
  const c = j * stride;
  const window: Grid = [];
  const products: Grid = [];
  let sum = 0;
  for (let a = 0; a < k; a++) {
    const wr: number[] = [];
    const pr: number[] = [];
    for (let b = 0; b < k; b++) {
      const v = xPadded[r + a][c + b];
      const p = v * kernel[a][b];
      wr.push(v);
      pr.push(p);
      sum += p;
    }
    window.push(wr);
    products.push(pr);
  }
  return { i, j, r, c, window, products, sum, value: sum + bias };
}

/* ------------------------------------------------------------------ */
/* Flat Float32Array versions for image-sized feature maps (CNN pipeline) */
/* ------------------------------------------------------------------ */

export interface Plane {
  data: Float32Array;
  h: number;
  w: number;
}

/**
 * Multi-channel convolution on flat planes.
 * input: C_in planes of equal size; weights: [C_out][C_in][k*k] (row-major k×k); bias: [C_out].
 */
export function convPlanes(input: Plane[], weights: number[][][], bias: number[], k: number, stride: number, padding: number): Plane[] {
  const H = input[0].h;
  const W = input[0].w;
  const Ho = outputSize(H, k, padding, stride);
  const Wo = outputSize(W, k, padding, stride);
  return weights.map((filter, co) => {
    const out = new Float32Array(Ho * Wo);
    const b = bias[co] ?? 0;
    for (let i = 0; i < Ho; i++)
      for (let j = 0; j < Wo; j++) {
        let acc = b;
        for (let ci = 0; ci < input.length; ci++) {
          const src = input[ci].data;
          const kw = filter[ci];
          for (let a = 0; a < k; a++) {
            const y = i * stride + a - padding;
            if (y < 0 || y >= H) continue; // zero padding contributes nothing
            for (let c = 0; c < k; c++) {
              const x = j * stride + c - padding;
              if (x < 0 || x >= W) continue;
              acc += src[y * W + x] * kw[a * k + c];
            }
          }
        }
        out[i * Wo + j] = acc;
      }
    return { data: out, h: Ho, w: Wo };
  });
}

export function reluPlane(p: Plane): Plane {
  const out = new Float32Array(p.data.length);
  for (let i = 0; i < out.length; i++) out[i] = p.data[i] > 0 ? p.data[i] : 0;
  return { data: out, h: p.h, w: p.w };
}

export function maxPoolPlane(p: Plane, size = 2, stride = 2): Plane {
  const Ho = outputSize(p.h, size, 0, stride);
  const Wo = outputSize(p.w, size, 0, stride);
  const out = new Float32Array(Ho * Wo);
  for (let i = 0; i < Ho; i++)
    for (let j = 0; j < Wo; j++) {
      let m = -Infinity;
      for (let a = 0; a < size; a++)
        for (let c = 0; c < size; c++) {
          const v = p.data[(i * stride + a) * p.w + j * stride + c];
          if (v > m) m = v;
        }
      out[i * Wo + j] = m;
    }
  return { data: out, h: Ho, w: Wo };
}

export function planeStats(p: Plane) {
  let min = Infinity;
  let max = -Infinity;
  let zeros = 0;
  let sum = 0;
  for (let i = 0; i < p.data.length; i++) {
    const v = p.data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    if (v === 0) zeros++;
    sum += v;
  }
  return { min, max, mean: sum / (p.data.length || 1), zeroFrac: zeros / (p.data.length || 1) };
}
