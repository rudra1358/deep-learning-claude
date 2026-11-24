/**
 * Pure-TypeScript inference engine for the variational autoencoder trained by
 * `scripts/train-autoencoders.py` and served at `/models/vae.json`.
 *
 * Architecture:
 *   encoder  x (64) → ReLU(W₁x+b₁) (64) → h = ReLU(W₂·+b₂) (32) → μ = W_μh+b_μ (2), log σ² = W_vh+b_v (2)
 *   sample   z = μ + σ ⊙ ε,  σ = exp(½ log σ²),  ε ~ N(0, I)          (reparameterisation trick)
 *   decoder  z (2) → ReLU (32) → ReLU (64) → x̂ = σ(·) (64 Bernoulli means in [0,1])
 *   loss     −ELBO = Σ BCE(x̂, x) + ½ Σⱼ (μⱼ² + σⱼ² − 1 − log σⱼ²)      (β = 1)
 *
 * Self-contained (no imports) so Node can run `verifyVae` against the PyTorch vectors in the JSON.
 */

export type Matrix = number[][];

export interface TensorJSON {
  shape: number[];
  scale: number;
  q: number[];
}

export interface VaeJSON {
  format: string;
  description: string;
  data: { name: string; n_train: number; n_test: number; split: string };
  config: { enc: number[]; latent: number; dec: number[]; hidden_act: string; out_act: string; likelihood: string; prior: string; beta: number; samples_per_example: number };
  training: {
    epochs: number;
    batch: number;
    optimizer: string;
    seed: number;
    history: Record<string, number>[];
    final: { train_recon: number; train_kl: number; train_neg_elbo: number; test_recon: number; test_kl: number; test_neg_elbo: number };
  };
  beta_sweep: { beta: number; test_recon: number; test_kl: number; mean_sigma: number[]; active_dims: number }[];
  n_params: number;
  quantisation: string;
  tensors: Record<string, TensorJSON>;
  test: { labels: number[]; pixels: number[][]; mu: number[][]; logvar: number[][] };
  train_mu: number[][];
  train_labels: number[];
  verify: {
    test_index?: number;
    eps?: number[];
    enc_h1?: number[];
    enc_h2?: number[];
    mu?: number[];
    logvar?: number[];
    z: number[];
    dec_h1?: number[];
    dec_h2?: number[];
    recon: number[];
    recon_loss?: number;
    kl?: number;
  }[];
}

export interface Dense {
  W: Matrix; // in × out
  b: number[];
}

export interface Vae {
  enc: Dense[]; // 2 hidden layers
  mu: Dense;
  logvar: Dense;
  dec: Dense[]; // 3 layers
  latent: number;
  nParams: number;
}

export interface VaeTrace {
  x: number[];
  encH1: number[];
  encH2: number[];
  mu: number[];
  logvar: number[];
  sigma: number[];
  eps: number[];
  z: number[];
  decH1: number[];
  decH2: number[];
  recon: number[];
  reconLoss: number;
  kl: number;
  klPerDim: number[];
}

/* ------------------------------------------------------------------ helpers */

function dequant(t: TensorJSON): Matrix | number[] {
  const v = t.q.map((q) => q * t.scale);
  if (t.shape.length === 1) return v;
  const [r, c] = t.shape;
  return Array.from({ length: r }, (_, i) => v.slice(i * c, (i + 1) * c));
}

function dense(tensors: Record<string, TensorJSON>, prefix: string): Dense {
  return { W: dequant(tensors[prefix + ".W"]) as Matrix, b: dequant(tensors[prefix + ".b"]) as number[] };
}

function applyDense(x: number[], L: Dense): number[] {
  const out = L.b.slice();
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = L.W[i];
    for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
  }
  return out;
}

const relu = (v: number[]) => v.map((x) => (x > 0 ? x : 0));
const sigmoid = (v: number[]) => v.map((x) => 1 / (1 + Math.exp(-x)));

/** Summed binary cross-entropy (PyTorch convention: logs clamped at −100). */
export function bceSum(p: number[], x: number[]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s -= x[i] * Math.max(Math.log(p[i]), -100) + (1 - x[i]) * Math.max(Math.log(1 - p[i]), -100);
  return s;
}

/** Closed-form KL(N(μ, diag σ²) ‖ N(0, I)) per dimension. */
export function klPerDim(mu: number[], logvar: number[]): number[] {
  return mu.map((m, j) => 0.5 * (m * m + Math.exp(logvar[j]) - 1 - logvar[j]));
}

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

/** k independent standard-normal numbers from a seed (the simulator's ε). */
export function standardNormal(k: number, seed: number): number[] {
  const rand = mulberry32(seed);
  return Array.from({ length: k }, () => gaussian(rand));
}

/* ------------------------------------------------------------------ model */

export function loadVae(json: VaeJSON): Vae {
  const enc = [0, 1].map((i) => dense(json.tensors, `enc.${i}`));
  const mu = dense(json.tensors, "mu");
  const logvar = dense(json.tensors, "logvar");
  const dec = [0, 1, 2].map((i) => dense(json.tensors, `dec.${i}`));
  const nParams = [...enc, mu, logvar, ...dec].reduce((s, L) => s + L.W.length * L.b.length + L.b.length, 0);
  return { enc, mu, logvar, dec, latent: json.config.latent, nParams };
}

export function encodeVae(m: Vae, x: number[]) {
  const encH1 = relu(applyDense(x, m.enc[0]));
  const encH2 = relu(applyDense(encH1, m.enc[1]));
  const mu = applyDense(encH2, m.mu);
  const logvar = applyDense(encH2, m.logvar);
  return { encH1, encH2, mu, logvar, sigma: logvar.map((v) => Math.exp(0.5 * v)) };
}

export function decodeVae(m: Vae, z: number[]) {
  const decH1 = relu(applyDense(z, m.dec[0]));
  const decH2 = relu(applyDense(decH1, m.dec[1]));
  const recon = sigmoid(applyDense(decH2, m.dec[2]));
  return { decH1, decH2, recon };
}

/** Full forward pass with a given ε (pass zeros for the deterministic "mean" reconstruction). */
export function forwardVae(m: Vae, x: number[], eps: number[]): VaeTrace {
  const e = encodeVae(m, x);
  const z = e.mu.map((mu, j) => mu + e.sigma[j] * eps[j]);
  const d = decodeVae(m, z);
  const kd = klPerDim(e.mu, e.logvar);
  return { x, ...e, eps, z, ...d, reconLoss: bceSum(d.recon, x), kl: kd.reduce((a, b) => a + b, 0), klPerDim: kd };
}

export function vaeTestImage(json: VaeJSON, i: number): number[] {
  return json.test.pixels[i].map((v) => v / 16);
}

/* ------------------------------------------------------------------ verification */

export function verifyVae(json: VaeJSON): { maxAbsDiff: number; maxLossDiff: number; checked: number } {
  const m = loadVae(json);
  let maxAbsDiff = 0;
  let maxLossDiff = 0;
  let checked = 0;
  const cmp = (a: number[], b: number[] | undefined) => {
    if (!b) return;
    a.forEach((v, i) => {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(v - b[i]));
      checked++;
    });
  };
  for (const v of json.verify) {
    if (v.test_index !== undefined && v.eps) {
      const t = forwardVae(m, vaeTestImage(json, v.test_index), v.eps);
      cmp(t.encH1, v.enc_h1);
      cmp(t.encH2, v.enc_h2);
      cmp(t.mu, v.mu);
      cmp(t.logvar, v.logvar);
      cmp(t.z, v.z);
      cmp(t.decH1, v.dec_h1);
      cmp(t.decH2, v.dec_h2);
      cmp(t.recon, v.recon);
      if (v.recon_loss !== undefined) maxLossDiff = Math.max(maxLossDiff, Math.abs(t.reconLoss - v.recon_loss));
      if (v.kl !== undefined) maxLossDiff = Math.max(maxLossDiff, Math.abs(t.kl - v.kl));
    } else {
      cmp(decodeVae(m, v.z).recon, v.recon);
    }
  }
  return { maxAbsDiff, maxLossDiff, checked };
}
