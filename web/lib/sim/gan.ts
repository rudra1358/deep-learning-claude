/**
 * A real GAN on 2-D data, written from scratch (no autodiff library).
 *
 *   Generator     G: z ∈ R^{zDim}  → hidden → hidden → x ∈ R^2      (leaky-ReLU hidden layers, linear output)
 *   Discriminator D: x ∈ R^2      → hidden → hidden → logit ∈ R    (leaky-ReLU hidden layers, linear output)
 *
 * Forward/backward passes, Adam and the WGAN-GP gradient penalty (which needs a derivative of an input
 * gradient — "double backprop") are implemented by hand on Float64Arrays. Everything is seeded, so a run
 * with the same config is bit-for-bit reproducible (this is what the lesson's preset numbers rely on).
 *
 * Losses (l = D's logit / critic value, σ = sigmoid):
 *   minimax   D: −log σ(l_real) − log(1 − σ(l_fake))      G: + log(1 − σ(l_fake))     (Goodfellow et al. 2014)
 *   nonsat    D: same as minimax                          G: − log σ(l_fake)
 *   lsgan     D: ½(l_real − 1)² + ½ l_fake²               G: ½(l_fake − 1)²            (Mao et al. 2017)
 *   wgangp    D: l_fake − l_real + λ(‖∇ₓl(x̂)‖ − 1)²       G: − l_fake                  (Gulrajani et al. 2017)
 */
import { gaussian, mulberry32 } from "./random";

export type TargetId = "ring" | "moons" | "spiral" | "gaussian";
export type LossId = "minimax" | "nonsat" | "lsgan" | "wgangp";

export const TARGETS: { id: TargetId; label: string }[] = [
  { id: "ring", label: "Ring of 8 Gaussians" },
  { id: "moons", label: "Two moons" },
  { id: "spiral", label: "Spiral" },
  { id: "gaussian", label: "Single Gaussian" },
];

export const LOSSES: { id: LossId; label: string; short: string }[] = [
  { id: "minimax", label: "Original minimax (saturating)", short: "minimax" },
  { id: "nonsat", label: "Non-saturating", short: "non-sat." },
  { id: "lsgan", label: "Least squares (LSGAN)", short: "LSGAN" },
  { id: "wgangp", label: "WGAN-GP", short: "WGAN-GP" },
];

/** Plot window (data coordinates) used by the simulator and the D(x) field. */
export const EXTENT = 3;
export const RING_RADIUS = 2;
export const RING_STD = 0.05;
export const N_MODES = 8;
/** A generated sample is "high quality" if it lies within 3σ of its nearest ring mode. */
export const HQ_RADIUS = 3 * RING_STD;
/** Radius for the generic precision / recall metrics (all targets). */
export const PR_RADIUS = 0.15;

/* ------------------------------------------------------------------ data */

export function sampleTarget(target: TargetId, n: number, rand: () => number): Float64Array {
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0;
    if (target === "ring") {
      const k = Math.floor(rand() * N_MODES);
      const a = (2 * Math.PI * k) / N_MODES;
      x = RING_RADIUS * Math.cos(a) + RING_STD * gaussian(rand);
      y = RING_RADIUS * Math.sin(a) + RING_STD * gaussian(rand);
    } else if (target === "moons") {
      const t = Math.PI * rand();
      if (rand() < 0.5) {
        x = Math.cos(t);
        y = Math.sin(t);
      } else {
        x = 1 - Math.cos(t);
        y = 0.5 - Math.sin(t);
      }
      // centre and scale to roughly [-2.4, 2.4] × [-1.2, 1.2]
      x = 1.6 * (x - 0.5) + 0.08 * gaussian(rand);
      y = 1.6 * (y - 0.25) + 0.08 * gaussian(rand);
    } else if (target === "spiral") {
      const t = 1.5 * Math.PI * (1 + 2 * Math.sqrt(rand())); // sqrt → uniform density along the arm
      x = (t * Math.cos(t)) / 5.2 + 0.06 * gaussian(rand);
      y = (t * Math.sin(t)) / 5.2 + 0.06 * gaussian(rand);
    } else {
      // N(μ, Σ): μ = (0.8, 0.4), principal std 0.7 and 0.25, rotated by 30°
      const u = 0.7 * gaussian(rand), v = 0.25 * gaussian(rand);
      const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
      x = 0.8 + c * u - s * v;
      y = 0.4 + s * u + c * v;
    }
    out[2 * i] = x;
    out[2 * i + 1] = y;
  }
  return out;
}

/* ------------------------------------------------------------------ MLP */

const SLOPE = 0.2; // leaky-ReLU negative slope

export interface Net {
  sizes: number[];
  W: Float64Array[]; // W[l]: sizes[l+1] × sizes[l], row-major
  b: Float64Array[];
  gW: Float64Array[];
  gb: Float64Array[];
  mW: Float64Array[];
  vW: Float64Array[];
  mb: Float64Array[];
  vb: Float64Array[];
  t: number;
}

export function makeNet(sizes: number[], rand: () => number): Net {
  const W: Float64Array[] = [], b: Float64Array[] = [];
  for (let l = 0; l + 1 < sizes.length; l++) {
    const fin = sizes[l], fout = sizes[l + 1];
    const bound = 1 / Math.sqrt(fin); // PyTorch nn.Linear default: U(−1/√fan_in, 1/√fan_in)
    const w = new Float64Array(fout * fin);
    for (let i = 0; i < w.length; i++) w[i] = (2 * rand() - 1) * bound;
    const bb = new Float64Array(fout);
    for (let i = 0; i < fout; i++) bb[i] = (2 * rand() - 1) * bound;
    W.push(w);
    b.push(bb);
  }
  const z = (a: Float64Array[]) => a.map((x) => new Float64Array(x.length));
  return { sizes, W, b, gW: z(W), gb: z(b), mW: z(W), vW: z(W), mb: z(b), vb: z(b), t: 0 };
}

export const nParams = (n: Net) => n.W.reduce((s, w) => s + w.length, 0) + n.b.reduce((s, b) => s + b.length, 0);

interface Cache {
  N: number;
  acts: Float64Array[]; // acts[0] = input, acts[l] = output of layer l (post-activation; last = linear output)
  pres: Float64Array[]; // pres[l-1] = pre-activation of layer l
}

export function forward(net: Net, X: Float64Array, N: number): Cache {
  const L = net.W.length;
  const acts: Float64Array[] = [X];
  const pres: Float64Array[] = [];
  for (let l = 0; l < L; l++) {
    const fin = net.sizes[l], fout = net.sizes[l + 1];
    const W = net.W[l], b = net.b[l], A = acts[l];
    const Z = new Float64Array(N * fout);
    for (let n = 0; n < N; n++) {
      const ao = n * fin, zo = n * fout;
      for (let j = 0; j < fout; j++) {
        let s = b[j];
        const wo = j * fin;
        for (let k = 0; k < fin; k++) s += W[wo + k] * A[ao + k];
        Z[zo + j] = s;
      }
    }
    pres.push(Z);
    if (l < L - 1) {
      const H = new Float64Array(Z.length);
      for (let i = 0; i < Z.length; i++) H[i] = Z[i] > 0 ? Z[i] : SLOPE * Z[i];
      acts.push(H);
    } else acts.push(Z);
  }
  return { N, acts, pres };
}

/**
 * Backward pass. dOut: N × sizes[L] gradient of the loss w.r.t. the network output.
 * Adds parameter gradients into net.gW / net.gb when `accumulate` is true. Returns dL/dInput (N × sizes[0]).
 */
export function backward(net: Net, c: Cache, dOut: Float64Array, accumulate: boolean): Float64Array {
  const L = net.W.length, N = c.N;
  let delta = dOut; // gradient w.r.t. pre-activation of layer l (output layer is linear)
  for (let l = L - 1; l >= 0; l--) {
    const fin = net.sizes[l], fout = net.sizes[l + 1];
    const W = net.W[l], A = c.acts[l];
    if (accumulate) {
      const gW = net.gW[l], gb = net.gb[l];
      for (let n = 0; n < N; n++) {
        const ao = n * fin, d0 = n * fout;
        for (let j = 0; j < fout; j++) {
          const d = delta[d0 + j];
          if (d === 0) continue;
          gb[j] += d;
          const wo = j * fin;
          for (let k = 0; k < fin; k++) gW[wo + k] += d * A[ao + k];
        }
      }
    }
    const dA = new Float64Array(N * fin);
    for (let n = 0; n < N; n++) {
      const ao = n * fin, d0 = n * fout;
      for (let j = 0; j < fout; j++) {
        const d = delta[d0 + j];
        if (d === 0) continue;
        const wo = j * fin;
        for (let k = 0; k < fin; k++) dA[ao + k] += d * W[wo + k];
      }
    }
    if (l > 0) {
      const Z = c.pres[l - 1];
      for (let i = 0; i < dA.length; i++) if (Z[i] <= 0) dA[i] *= SLOPE;
    }
    delta = dA;
  }
  return delta;
}

export function zeroGrad(net: Net) {
  net.gW.forEach((g) => g.fill(0));
  net.gb.forEach((g) => g.fill(0));
}

export interface AdamCfg {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export function adamStep(net: Net, o: AdamCfg) {
  net.t++;
  const bc1 = 1 - Math.pow(o.beta1, net.t), bc2 = 1 - Math.pow(o.beta2, net.t);
  const upd = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
    for (let i = 0; i < p.length; i++) {
      m[i] = o.beta1 * m[i] + (1 - o.beta1) * g[i];
      v[i] = o.beta2 * v[i] + (1 - o.beta2) * g[i] * g[i];
      p[i] -= (o.lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + o.eps);
    }
  };
  for (let l = 0; l < net.W.length; l++) {
    upd(net.W[l], net.gW[l], net.mW[l], net.vW[l]);
    upd(net.b[l], net.gb[l], net.mb[l], net.vb[l]);
  }
}

/**
 * WGAN-GP penalty λ·mean_n (‖∇ₓ f(x̂_n)‖ − 1)², with its gradient w.r.t. the critic's weights added to net.gW.
 * For a leaky-ReLU MLP the input gradient is a product of weight matrices and (locally constant) slope
 * matrices: ∇ₓf = W₁ᵀ S₁ W₂ᵀ S₂ … W_Lᵀ·1, so the penalty depends on the weights only through that product
 * (biases get no gradient, slopes have zero derivative almost everywhere). Returns the mean penalty and the
 * mean gradient norm.
 */
export function gradientPenalty(net: Net, Xh: Float64Array, N: number, lambda: number): { gp: number; gnorm: number } {
  const c = forward(net, Xh, N);
  const L = net.W.length;
  let gpSum = 0, gnSum = 0;
  // r[l] = d f / d pre_l for this sample (r[L-1] = [1] for the linear output)
  const r: Float64Array[] = net.sizes.slice(1).map((s) => new Float64Array(s));
  for (let n = 0; n < N; n++) {
    r[L - 1][0] = 1;
    for (let l = L - 1; l >= 1; l--) {
      const fin = net.sizes[l], fout = net.sizes[l + 1], W = net.W[l];
      const dst = r[l - 1];
      dst.fill(0);
      for (let j = 0; j < fout; j++) {
        const d = r[l][j];
        const wo = j * fin;
        for (let k = 0; k < fin; k++) dst[k] += d * W[wo + k];
      }
      const Z = c.pres[l - 1];
      for (let k = 0; k < fin; k++) if (Z[n * fin + k] <= 0) dst[k] *= SLOPE;
    }
    const fin0 = net.sizes[0], fout0 = net.sizes[1], W0 = net.W[0];
    const g = new Float64Array(fin0);
    for (let j = 0; j < fout0; j++) {
      const d = r[0][j];
      for (let k = 0; k < fin0; k++) g[k] += d * W0[j * fin0 + k];
    }
    let gn = 0;
    for (let k = 0; k < fin0; k++) gn += g[k] * g[k];
    gn = Math.sqrt(gn) + 1e-12;
    gnSum += gn;
    gpSum += (gn - 1) * (gn - 1);
    // q = dP/d(gradient w.r.t. layer-l input), starting with dP/dg
    let q = new Float64Array(fin0);
    const coef = ((2 * lambda) / N) * (gn - 1) / gn;
    for (let k = 0; k < fin0; k++) q[k] = coef * g[k];
    for (let l = 0; l < L; l++) {
      const fin = net.sizes[l], fout = net.sizes[l + 1], W = net.W[l], gW = net.gW[l], rl = r[l];
      // g_in = W_lᵀ r_l  ⇒  dP/dW_l[j,k] = r_l[j]·q[k],  dP/dr_l = W_l q
      for (let j = 0; j < fout; j++) {
        const wo = j * fin;
        for (let k = 0; k < fin; k++) gW[wo + k] += rl[j] * q[k];
      }
      if (l < L - 1) {
        const qn = new Float64Array(fout);
        const Z = c.pres[l];
        for (let j = 0; j < fout; j++) {
          let s = 0;
          const wo = j * fin;
          for (let k = 0; k < fin; k++) s += W[wo + k] * q[k];
          qn[j] = Z[n * fout + j] > 0 ? s : SLOPE * s; // r_l = S_l (W_{l+1}ᵀ r_{l+1})
        }
        q = qn;
      }
    }
  }
  return { gp: (lambda * gpSum) / N, gnorm: gnSum / N };
}

/* ------------------------------------------------------------------ GAN trainer */

export interface GanConfig {
  target: TargetId;
  loss: LossId;
  lrG: number;
  lrD: number;
  dSteps: number;
  zDim: number;
  hidden: number;
  batch: number;
  seed: number;
  gpLambda: number;
}

export const DEFAULT_CONFIG: GanConfig = { target: "ring", loss: "nonsat", lrG: 1e-3, lrD: 1e-3, dSteps: 1, zDim: 8, hidden: 32, batch: 64, seed: 1, gpLambda: 0.1 };

export interface HistPoint {
  iter: number;
  dLoss: number;
  gLoss: number;
}

export interface Metrics {
  iter: number;
  modes: number; // ring only: modes holding ≥ MODE_FRACTION of the uniform share of high-quality samples
  hq: number; // ring only: fraction of samples within 3σ of a mode
  modeCounts: number[]; // ring only: high-quality samples per mode
  precision: number; // fraction of generated samples within PR_RADIUS of a real sample
  recall: number; // fraction of real samples within PR_RADIUS of a generated sample
  dReal: number; // mean D(x) (probability; for WGAN the raw critic value) on real eval data
  dFake: number; // same on generated eval samples
  gGradNorm: number; // mean ‖∂L_G/∂x‖ at generated samples
  /** Classifier losses only (else NaN): mean ‖∂L_G/∂x‖ the saturating and the non-saturating loss would give at the SAME D and samples. */
  satGrad: number;
  nsGrad: number;
}

/** A mode counts as covered if it holds at least this fraction of its fair share (1/8) of the samples. */
export const MODE_FRACTION = 0.25;

const sig = (v: number) => (v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v)));
const softplus = (v: number) => (v > 0 ? v + Math.log1p(Math.exp(-v)) : Math.log1p(Math.exp(v)));

export class GanTrainer {
  cfg: GanConfig;
  G: Net;
  D: Net;
  iter = 0;
  history: HistPoint[] = [];
  private rand: () => number;
  private accD = 0;
  private accG = 0;
  private accN = 0;
  readonly histEvery = 10;
  readonly zEval: Float64Array;
  readonly realEval: Float64Array;
  readonly nEval = 600;

  constructor(cfg: GanConfig) {
    this.cfg = { ...cfg };
    const init = mulberry32(cfg.seed * 7919 + 17);
    this.G = makeNet([cfg.zDim, cfg.hidden, cfg.hidden, 2], init);
    this.D = makeNet([2, cfg.hidden, cfg.hidden, 1], init);
    this.rand = mulberry32(cfg.seed * 104729 + 3);
    const ev = mulberry32(424242);
    this.zEval = this.noise(this.nEval, ev);
    this.realEval = sampleTarget(cfg.target, 1000, ev);
  }

  /** Change learning rates / D steps on the fly (keeps the weights). */
  setHyper(h: Partial<Pick<GanConfig, "lrG" | "lrD" | "dSteps">>) {
    Object.assign(this.cfg, h);
  }

  private noise(n: number, rand: () => number) {
    const z = new Float64Array(n * this.cfg.zDim);
    for (let i = 0; i < z.length; i++) z[i] = gaussian(rand);
    return z;
  }

  private adam(lr: number): AdamCfg {
    // β₁ = 0.5 is the standard GAN choice (Radford et al. 2015); β₂ = 0.9 as in WGAN-GP.
    return { lr, beta1: 0.5, beta2: 0.9, eps: 1e-8 };
  }

  generate(z: Float64Array, n: number): Float64Array {
    const c = forward(this.G, z, n);
    return c.acts[c.acts.length - 1];
  }

  /** dL/dlogit for D's loss (real and fake halves), and the loss value. */
  private dLossGrad(lr: Float64Array, lf: Float64Array, N: number) {
    const gr = new Float64Array(N), gf = new Float64Array(N);
    let loss = 0;
    const loss_ = this.cfg.loss;
    for (let n = 0; n < N; n++) {
      if (loss_ === "minimax" || loss_ === "nonsat") {
        loss += softplus(-lr[n]) + softplus(lf[n]);
        gr[n] = (sig(lr[n]) - 1) / N;
        gf[n] = sig(lf[n]) / N;
      } else if (loss_ === "lsgan") {
        loss += 0.5 * (lr[n] - 1) ** 2 + 0.5 * lf[n] ** 2;
        gr[n] = (lr[n] - 1) / N;
        gf[n] = lf[n] / N;
      } else {
        loss += lf[n] - lr[n];
        gr[n] = -1 / N;
        gf[n] = 1 / N;
      }
    }
    return { gr, gf, loss: loss / N };
  }

  /** dL_G/dlogit at fake samples, and L_G. */
  gLossGrad(lf: Float64Array, N: number) {
    const g = new Float64Array(N);
    let loss = 0;
    for (let n = 0; n < N; n++) {
      const v = lf[n];
      switch (this.cfg.loss) {
        case "minimax":
          loss += -softplus(v); // log(1 − σ(v))
          g[n] = -sig(v) / N;
          break;
        case "nonsat":
          loss += softplus(-v); // −log σ(v)
          g[n] = (sig(v) - 1) / N;
          break;
        case "lsgan":
          loss += 0.5 * (v - 1) ** 2;
          g[n] = (v - 1) / N;
          break;
        default:
          loss += -v;
          g[n] = -1 / N;
      }
    }
    return { g, loss: loss / N };
  }

  /** One training iteration: dSteps discriminator updates, then one generator update. */
  trainIter() {
    const { batch: N, dSteps } = this.cfg;
    let dLoss = 0;
    for (let s = 0; s < dSteps; s++) {
      const real = sampleTarget(this.cfg.target, N, this.rand);
      const fake = this.generate(this.noise(N, this.rand), N);
      zeroGrad(this.D);
      const cr = forward(this.D, real, N);
      const cf = forward(this.D, fake, N);
      const { gr, gf, loss } = this.dLossGrad(cr.acts[3], cf.acts[3], N);
      backward(this.D, cr, gr, true);
      backward(this.D, cf, gf, true);
      let total = loss;
      if (this.cfg.loss === "wgangp") {
        const xh = new Float64Array(2 * N);
        for (let n = 0; n < N; n++) {
          const e = this.rand();
          xh[2 * n] = e * real[2 * n] + (1 - e) * fake[2 * n];
          xh[2 * n + 1] = e * real[2 * n + 1] + (1 - e) * fake[2 * n + 1];
        }
        total += gradientPenalty(this.D, xh, N, this.cfg.gpLambda).gp;
      }
      adamStep(this.D, this.adam(this.cfg.lrD));
      dLoss += total;
    }
    dLoss /= dSteps;
    // generator step: backprop L_G through D (frozen) into G
    const z = this.noise(N, this.rand);
    const cg = forward(this.G, z, N);
    const fake = cg.acts[3];
    const cd = forward(this.D, fake, N);
    const { g, loss: gLoss } = this.gLossGrad(cd.acts[3], N);
    const dx = backward(this.D, cd, g, false);
    zeroGrad(this.G);
    backward(this.G, cg, dx, true);
    adamStep(this.G, this.adam(this.cfg.lrG));
    this.iter++;
    this.accD += dLoss;
    this.accG += gLoss;
    this.accN++;
    if (this.accN === this.histEvery) {
      this.history.push({ iter: this.iter, dLoss: this.accD / this.accN, gLoss: this.accG / this.accN });
      this.accD = this.accG = this.accN = 0;
      if (this.history.length > 800) this.history = this.history.filter((_, i) => i % 2 === 0); // bound memory
    }
  }

  train(n: number) {
    for (let i = 0; i < n; i++) this.trainIter();
  }

  /** D's output on arbitrary points: probability σ(logit) for the classifier losses, raw output for LSGAN/WGAN. */
  dOutput(X: Float64Array, N: number): Float64Array {
    const c = forward(this.D, X, N);
    const o = c.acts[3];
    if (this.cfg.loss === "minimax" || this.cfg.loss === "nonsat") return o.map(sig);
    return o;
  }

  /** −∂L_G/∂x at generated points: the direction in which the generator is being pushed to move each sample. */
  flow(X: Float64Array, N: number): Float64Array {
    const c = forward(this.D, X, N);
    const { g } = this.gLossGrad(c.acts[3], N);
    const dx = backward(this.D, c, g, false);
    for (let i = 0; i < dx.length; i++) dx[i] *= -N; // undo the 1/N batch mean: per-sample gradient
    return dx;
  }

  evalSamples(): Float64Array {
    return this.generate(this.zEval, this.nEval);
  }

  metrics(samples = this.evalSamples()): Metrics {
    const M = this.nEval, R = this.realEval.length / 2;
    const counts = new Array(N_MODES).fill(0);
    let hq = 0;
    if (this.cfg.target === "ring") {
      for (let i = 0; i < M; i++) {
        const x = samples[2 * i], y = samples[2 * i + 1];
        let best = Infinity, bk = 0;
        for (let k = 0; k < N_MODES; k++) {
          const a = (2 * Math.PI * k) / N_MODES;
          const d = Math.hypot(x - RING_RADIUS * Math.cos(a), y - RING_RADIUS * Math.sin(a));
          if (d < best) {
            best = d;
            bk = k;
          }
        }
        if (best <= HQ_RADIUS) {
          hq++;
          counts[bk]++;
        }
      }
    }
    const thr = (MODE_FRACTION * M) / N_MODES;
    const modes = counts.filter((c) => c >= thr).length;
    const r2 = PR_RADIUS * PR_RADIUS;
    let prec = 0;
    const realHit = new Uint8Array(R);
    for (let i = 0; i < M; i++) {
      const x = samples[2 * i], y = samples[2 * i + 1];
      let hit = false;
      for (let j = 0; j < R; j++) {
        const dx = x - this.realEval[2 * j], dy = y - this.realEval[2 * j + 1];
        if (dx * dx + dy * dy <= r2) {
          hit = true;
          realHit[j] = 1;
        }
      }
      if (hit) prec++;
    }
    let rec = 0;
    for (let j = 0; j < R; j++) rec += realHit[j];
    const dr = this.dOutput(this.realEval, R), df = this.dOutput(samples, M);
    const fl = this.flow(samples, M);
    let gn = 0;
    for (let i = 0; i < M; i++) gn += Math.hypot(fl[2 * i], fl[2 * i + 1]);
    const mean = (a: Float64Array) => a.reduce((s, v) => s + v, 0) / a.length;
    let satGrad = NaN, nsGrad = NaN;
    if (this.cfg.loss === "minimax" || this.cfg.loss === "nonsat") {
      // ∇ₓ l for every sample (dOut = 1 per sample), then weight by σ(l) (saturating) or 1 − σ(l) (non-saturating)
      const c = forward(this.D, samples, M);
      const gl = backward(this.D, c, new Float64Array(M).fill(1), false);
      satGrad = 0;
      nsGrad = 0;
      for (let i = 0; i < M; i++) {
        const n = Math.hypot(gl[2 * i], gl[2 * i + 1]), p = sig(c.acts[3][i]);
        satGrad += (p * n) / M;
        nsGrad += ((1 - p) * n) / M;
      }
    }
    return { iter: this.iter, modes, hq: hq / M, modeCounts: counts, precision: prec / M, recall: rec / R, dReal: mean(dr), dFake: mean(df), gGradNorm: gn / M, satGrad, nsGrad };
  }

  /** D's output on a res × res grid over [−EXTENT, EXTENT]², row 0 = top (y = +EXTENT). */
  dField(res: number): Float64Array {
    const X = new Float64Array(2 * res * res);
    for (let j = 0; j < res; j++)
      for (let i = 0; i < res; i++) {
        const k = j * res + i;
        X[2 * k] = -EXTENT + ((i + 0.5) / res) * 2 * EXTENT;
        X[2 * k + 1] = EXTENT - ((j + 0.5) / res) * 2 * EXTENT;
      }
    return this.dOutput(X, res * res);
  }
}

/* ------------------------------------------------------------------ presets */

export interface Preset {
  id: string;
  label: string;
  cfg: GanConfig;
  note: string;
}

export const PRESETS: Preset[] = [
  {
    id: "converge",
    label: "Converges (non-saturating)",
    cfg: { ...DEFAULT_CONFIG, loss: "nonsat", lrG: 1e-3, lrD: 1e-3, dSteps: 1, seed: 1 },
    note: "Non-saturating loss, equal learning rates 1e-3. All 8 modes are covered after about 5000 iterations.",
  },
  {
    id: "wgangp",
    label: "Converges (WGAN-GP)",
    cfg: { ...DEFAULT_CONFIG, loss: "wgangp", lrG: 1e-3, lrD: 1e-3, dSteps: 5, gpLambda: 0.1, seed: 2 },
    note: "Critic updated 5× per generator step with a gradient penalty. All 8 modes are covered after about 4000 iterations.",
  },
  {
    id: "hop",
    label: "Collapse + mode hopping (minimax)",
    cfg: { ...DEFAULT_CONFIG, loss: "minimax", lrG: 1e-2, lrD: 1e-4, dSteps: 1, seed: 1 },
    note: "Original loss with a generator 100× faster than the discriminator: G squeezes its samples onto a thin curve that jumps around the plane.",
  },
  {
    id: "saturate",
    label: "D too strong (minimax)",
    cfg: { ...DEFAULT_CONFIG, loss: "minimax", lrG: 3e-4, lrD: 3e-3, dSteps: 1, seed: 1 },
    note: "Original loss with a discriminator 10× faster: D rejects the fakes confidently, log(1 − D) saturates for the worst samples and G never reaches the modes.",
  },
  {
    id: "collapse",
    label: "D too strong (non-saturating)",
    cfg: { ...DEFAULT_CONFIG, loss: "nonsat", lrG: 3e-4, lrD: 3e-3, dSteps: 1, seed: 1 },
    note: "The same learning rates with the non-saturating loss: G does get a gradient, but piles almost all its samples onto one mode.",
  },
];

/** Precision / recall of a perfect generator (fresh real samples) — the ceiling of the finite-sample metrics. */
export function referencePR(target: TargetId, nGen = 600): { precision: number; recall: number } {
  const real = sampleTarget(target, 1000, mulberry32(424243));
  const gen = sampleTarget(target, nGen, mulberry32(777));
  const r2 = PR_RADIUS * PR_RADIUS;
  const hit = new Uint8Array(1000);
  let p = 0;
  for (let i = 0; i < nGen; i++) {
    let h = false;
    for (let j = 0; j < 1000; j++) {
      const dx = gen[2 * i] - real[2 * j], dy = gen[2 * i + 1] - real[2 * j + 1];
      if (dx * dx + dy * dy <= r2) {
        h = true;
        hit[j] = 1;
      }
    }
    if (h) p++;
  }
  let r = 0;
  for (let j = 0; j < 1000; j++) r += hit[j];
  return { precision: p / nGen, recall: r / 1000 };
}
