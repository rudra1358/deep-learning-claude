/**
 * Pure-TypeScript inference engine for the deterministic autoencoder (and its denoising twin)
 * trained by `scripts/train-autoencoders.py` and served at `/models/autoencoder.json`.
 *
 * Architecture (both models):
 *   encoder  x (64) → ReLU(W₁x+b₁) (64) → ReLU(W₂h+b₂) (32) → z = W₃h+b₃ (2, linear code)
 *   decoder  z (2)  → ReLU(V₁z+c₁) (32) → ReLU(V₂h+c₂) (64) → x̂ = σ(V₃h+c₃) (64 pixels in [0,1])
 *
 * The file is self-contained (no imports) so Node can run it directly to compare its outputs
 * with the PyTorch verification vectors stored in the JSON (`verifyAutoencoder`).
 */

export type Matrix = number[][];

export interface TensorJSON {
  shape: number[];
  scale: number;
  q: number[];
}

export interface AeModelJSON {
  tensors: Record<string, TensorJSON>;
  n_params: number;
  history: Record<string, number>[];
  final: { train_bce: number; train_mse: number; test_bce: number; test_mse: number };
  denoise: Record<string, { noisy_mse: number; recon_mse: number }>;
  codes_test: number[][];
  codes_train: number[][];
}

export interface AutoencoderJSON {
  format: string;
  description: string;
  data: { name: string; n_train: number; n_test: number; split: string };
  config: { enc: number[]; dec: number[]; hidden_act: string; code_act: string; out_act: string; loss: string; dae_noise: string };
  training: { epochs: number; batch: number; optimizer: string; seed: number };
  quantisation: string;
  models: { ae: AeModelJSON; dae: AeModelJSON };
  test: { labels: number[]; pixels: number[][]; train_labels: number[] };
  verify: {
    model: "ae" | "dae";
    test_index?: number;
    z: number[];
    enc_h1?: number[];
    enc_h2?: number[];
    dec_h1?: number[];
    dec_h2?: number[];
    recon: number[];
  }[];
}

export interface Dense {
  W: Matrix; // in × out ("x @ W" orientation)
  b: number[];
}

export interface Autoencoder {
  enc: Dense[]; // 3 layers
  dec: Dense[]; // 3 layers
  nParams: number;
}

export interface AeTrace {
  x: number[];
  encH1: number[];
  encH2: number[];
  z: number[];
  decH1: number[];
  decH2: number[];
  recon: number[];
}

/* ------------------------------------------------------------------ helpers */

export function dequant(t: TensorJSON): Matrix | number[] {
  const v = t.q.map((q) => q * t.scale);
  if (t.shape.length === 1) return v;
  const [r, c] = t.shape;
  return Array.from({ length: r }, (_, i) => v.slice(i * c, (i + 1) * c));
}

export function dense(tensors: Record<string, TensorJSON>, prefix: string): Dense {
  return { W: dequant(tensors[prefix + ".W"]) as Matrix, b: dequant(tensors[prefix + ".b"]) as number[] };
}

/** y = x·W + b */
export function applyDense(x: number[], L: Dense): number[] {
  const out = L.b.slice();
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = L.W[i];
    for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
  }
  return out;
}

export const relu = (v: number[]) => v.map((x) => (x > 0 ? x : 0));
export const sigmoid = (v: number[]) => v.map((x) => 1 / (1 + Math.exp(-x)));

/** Summed binary cross-entropy of reconstruction p against target x (both in [0,1]), as PyTorch computes it (log clamped at −100). */
export function bceSum(p: number[], x: number[]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const lp = Math.max(Math.log(p[i]), -100);
    const l1 = Math.max(Math.log(1 - p[i]), -100);
    s -= x[i] * lp + (1 - x[i]) * l1;
  }
  return s;
}

export function mse(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}

/** Deterministic PRNG (mulberry32) — same as lib/sim/random.ts, duplicated to keep this file self-contained. */
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

export function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Add seeded Gaussian noise of standard deviation `sigma` to an image in [0,1] and clip back to [0,1]. */
export function addNoise(x: number[], sigma: number, seed: number): number[] {
  const rand = mulberry32(seed);
  return x.map((v) => Math.min(1, Math.max(0, v + sigma * gaussian(rand))));
}

/* ------------------------------------------------------------------ model */

export function loadAutoencoder(m: AeModelJSON): Autoencoder {
  const enc = [0, 1, 2].map((i) => dense(m.tensors, `enc.${i}`));
  const dec = [0, 1, 2].map((i) => dense(m.tensors, `dec.${i}`));
  const nParams = [...enc, ...dec].reduce((s, L) => s + L.W.length * L.b.length + L.b.length, 0);
  return { enc, dec, nParams };
}

export function encode(ae: Autoencoder, x: number[]) {
  const encH1 = relu(applyDense(x, ae.enc[0]));
  const encH2 = relu(applyDense(encH1, ae.enc[1]));
  const z = applyDense(encH2, ae.enc[2]);
  return { encH1, encH2, z };
}

export function decode(ae: Autoencoder, z: number[]) {
  const decH1 = relu(applyDense(z, ae.dec[0]));
  const decH2 = relu(applyDense(decH1, ae.dec[1]));
  const recon = sigmoid(applyDense(decH2, ae.dec[2]));
  return { decH1, decH2, recon };
}

export function forwardAe(ae: Autoencoder, x: number[]): AeTrace {
  const e = encode(ae, x);
  const d = decode(ae, e.z);
  return { x, ...e, ...d };
}

/** Test image i as pixels in [0,1]. */
export function testImage(json: AutoencoderJSON, i: number): number[] {
  return json.test.pixels[i].map((v) => v / 16);
}

/* ------------------------------------------------------------------ verification */

/** Largest absolute difference between this engine and the PyTorch verification vectors. */
export function verifyAutoencoder(json: AutoencoderJSON): { maxAbsDiff: number; checked: number } {
  const models = { ae: loadAutoencoder(json.models.ae), dae: loadAutoencoder(json.models.dae) };
  let maxAbsDiff = 0;
  let checked = 0;
  const cmp = (a: number[], b: number[] | undefined) => {
    if (!b) return;
    a.forEach((v, i) => {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(v - b[i]));
      checked++;
    });
  };
  for (const v of json.verify) {
    const m = models[v.model];
    if (v.test_index !== undefined) {
      const t = forwardAe(m, testImage(json, v.test_index));
      cmp(t.encH1, v.enc_h1);
      cmp(t.encH2, v.enc_h2);
      cmp(t.z, v.z);
      cmp(t.decH1, v.dec_h1);
      cmp(t.decH2, v.dec_h2);
      cmp(t.recon, v.recon);
    } else {
      cmp(decode(m, v.z).recon, v.recon);
    }
  }
  return { maxAbsDiff, checked };
}
