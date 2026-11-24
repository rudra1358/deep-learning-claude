/**
 * Receptive-field arithmetic for a stack of convolution / pooling layers (pure math, no React).
 *
 * Conventions (one spatial axis; square kernels make 2-D the outer product of two 1-D results):
 *  - Layer l maps a length-n input to floor((n + 2p - k_eff) / s) + 1 outputs, k_eff = d(k - 1) + 1.
 *  - Output unit i of a layer reads input indices i*s - p + a*d for a = 0..k-1 (zero padding outside).
 *  - Receptive field recurrence:  r_l = r_{l-1} + (k_eff - 1) * j_{l-1},   j_l = j_{l-1} * s_l,
 *    with r_0 = 1, j_0 = 1. `start` is the input coordinate of the centre of unit 0.
 */

import { gaussian, mulberry32 } from "./random";

export type LayerKind = "conv" | "pool";

export interface RFLayer {
  kind: LayerKind;
  k: number;
  s: number;
  /** dilation (pooling layers always use 1) */
  d: number;
}

export type PaddingMode = "same" | "valid";

export interface LayerInfo {
  kEff: number;
  pad: number;
  /** spatial size of this layer's OUTPUT */
  size: number;
  /** receptive field width in input pixels (theoretical, ignoring borders) */
  r: number;
  /** jump: input-pixel distance between neighbouring units of this layer */
  j: number;
  /** input coordinate of the centre of unit 0 of this layer */
  start: number;
}

export const effectiveKernel = (l: RFLayer) => (l.kind === "pool" ? l.k : l.d * (l.k - 1) + 1);

/** Padding used by a layer: convolutions get "same"-style padding floor((k_eff-1)/2) in "same" mode; pooling never pads. */
export function layerPadding(l: RFLayer, mode: PaddingMode) {
  if (l.kind === "pool" || mode === "valid") return 0;
  return Math.floor((effectiveKernel(l) - 1) / 2);
}

export function outSize(n: number, kEff: number, p: number, s: number) {
  const span = n + 2 * p - kEff;
  return span < 0 ? 0 : Math.floor(span / s) + 1;
}

/** Forward pass of the shape / receptive-field bookkeeping. Row 0 describes the input itself. */
export function analyze(layers: RFLayer[], inputSize: number, mode: PaddingMode): LayerInfo[] {
  const rows: LayerInfo[] = [{ kEff: 1, pad: 0, size: inputSize, r: 1, j: 1, start: 0 }];
  for (const l of layers) {
    const prev = rows[rows.length - 1];
    const kEff = effectiveKernel(l);
    const pad = layerPadding(l, mode);
    const s = l.s;
    rows.push({
      kEff,
      pad,
      size: outSize(prev.size, kEff, pad, s),
      r: prev.r + (kEff - 1) * prev.j,
      j: prev.j * s,
      start: prev.start + ((kEff - 1) / 2 - pad) * prev.j,
    });
  }
  return rows;
}

/**
 * Exact back-projection of one output unit through the stack.
 * Returns, for every level (0 = input … L = output), a boolean mask of the units that
 * influence the chosen output unit, plus the unclipped index interval [lo, hi].
 * Dilated kernels can leave holes, so the mask — not the interval — is the exact answer.
 */
export function backProject(layers: RFLayer[], info: LayerInfo[], unit: number, mode: PaddingMode) {
  const L = layers.length;
  const masks: boolean[][] = info.map((row) => new Array<boolean>(row.size).fill(false));
  const ranges: { lo: number; hi: number }[] = info.map(() => ({ lo: 0, hi: -1 }));
  if (unit < 0 || unit >= info[L].size) return { masks, ranges };
  masks[L][unit] = true;
  ranges[L] = { lo: unit, hi: unit };
  for (let l = L; l >= 1; l--) {
    const layer = layers[l - 1];
    const p = layerPadding(layer, mode);
    const d = layer.kind === "pool" ? 1 : layer.d;
    const nIn = info[l - 1].size;
    for (let u = 0; u < info[l].size; u++) {
      if (!masks[l][u]) continue;
      for (let a = 0; a < layer.k; a++) {
        const idx = u * layer.s - p + a * d;
        if (idx >= 0 && idx < nIn) masks[l - 1][idx] = true;
      }
    }
    const { lo, hi } = ranges[l];
    ranges[l - 1] = { lo: lo * layer.s - p, hi: hi * layer.s - p + d * (layer.k - 1) };
  }
  return { masks, ranges };
}

/**
 * Effective receptive field: |d out[uRow][uCol] / d input| for a random-weight LINEAR version of the stack
 * (convolutions with Gaussian weights, pooling replaced by average pooling, no non-linearities),
 * averaged over `trials` random weight draws. Computed by back-propagating a unit gradient exactly.
 * Returns an n×n Float64Array (row-major), normalised to sum to 1.
 */
export function effectiveRF(layers: RFLayer[], info: LayerInfo[], uRow: number, uCol: number, mode: PaddingMode, trials = 24, seed = 7) {
  const n = info[0].size;
  const L = layers.length;
  const acc = new Float64Array(n * n);
  if (info[L].size <= 0) return acc;
  const rand = mulberry32(seed);
  for (let t = 0; t < trials; t++) {
    let gs = info[L].size;
    let g = new Float64Array(gs * gs);
    g[uRow * gs + uCol] = 1;
    for (let l = L; l >= 1; l--) {
      const layer = layers[l - 1];
      const k = layer.k;
      const p = layerPadding(layer, mode);
      const d = layer.kind === "pool" ? 1 : layer.d;
      const w = new Float64Array(k * k);
      for (let i = 0; i < k * k; i++) w[i] = layer.kind === "pool" ? 1 / (k * k) : gaussian(rand) / k;
      const nIn = info[l - 1].size;
      const gin = new Float64Array(nIn * nIn);
      for (let i = 0; i < gs; i++)
        for (let j = 0; j < gs; j++) {
          const v = g[i * gs + j];
          if (v === 0) continue;
          for (let a = 0; a < k; a++) {
            const y = i * layer.s - p + a * d;
            if (y < 0 || y >= nIn) continue;
            for (let b = 0; b < k; b++) {
              const x = j * layer.s - p + b * d;
              if (x < 0 || x >= nIn) continue;
              gin[y * nIn + x] += w[a * k + b] * v; // transpose of the forward multiply-add
            }
          }
        }
      g = gin;
      gs = nIn;
    }
    let tot = 0;
    for (let i = 0; i < g.length; i++) tot += Math.abs(g[i]);
    if (tot > 0) for (let i = 0; i < g.length; i++) acc[i] += Math.abs(g[i]) / tot;
  }
  let sum = 0;
  for (let i = 0; i < acc.length; i++) sum += acc[i];
  if (sum > 0) for (let i = 0; i < acc.length; i++) acc[i] /= sum;
  return acc;
}

/** Width (in pixels) of the smallest square centred on (cr, cc) holding `share` of a normalised n×n map. */
export function massWidth(map: Float64Array, n: number, cr: number, cc: number, share: number) {
  for (let h = 0; h <= n; h++) {
    let s = 0;
    for (let i = Math.max(0, cr - h); i <= Math.min(n - 1, cr + h); i++)
      for (let j = Math.max(0, cc - h); j <= Math.min(n - 1, cc + h); j++) s += map[i * n + j];
    if (s >= share - 1e-12) return 2 * h + 1;
  }
  return n;
}

export const PRESETS: { id: string; label: string; input: number; layers: RFLayer[] }[] = [
  {
    id: "3x3",
    label: "3×3 convs only",
    input: 24,
    layers: Array.from({ length: 4 }, () => ({ kind: "conv" as const, k: 3, s: 1, d: 1 })),
  },
  {
    id: "vgg",
    label: "VGG-style",
    input: 48,
    layers: [
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "pool", k: 2, s: 2, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "pool", k: 2, s: 2, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "pool", k: 2, s: 2, d: 1 },
    ],
  },
  {
    id: "stride2",
    label: "With stride-2 convs",
    input: 32,
    layers: [
      { kind: "conv", k: 3, s: 2, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
      { kind: "conv", k: 3, s: 2, d: 1 },
      { kind: "conv", k: 3, s: 1, d: 1 },
    ],
  },
  {
    id: "dilated",
    label: "Dilated (1, 2, 4, 8)",
    input: 40,
    layers: [1, 2, 4, 8].map((d) => ({ kind: "conv" as const, k: 3, s: 1, d })),
  },
  {
    id: "gridding",
    label: "Dilated (2, 2, 2) — gridding",
    input: 24,
    layers: [2, 2, 2].map((d) => ({ kind: "conv" as const, k: 3, s: 1, d })),
  },
];
