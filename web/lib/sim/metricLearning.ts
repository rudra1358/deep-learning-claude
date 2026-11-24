/**
 * Metric learning on a toy problem (pure math, no React).
 *
 * Data: K = 5 classes in D = 8 dimensions. The class information lives in the first 4 coordinates
 * (class means ~ N(0, 1.6²), within-class noise σ = 0.45); coordinates 5–8 are pure nuisance noise
 * (σ = 2.0) that carries no class information. A random linear projection to 2-D therefore mixes the
 * classes; a good projection must learn to ignore the nuisance directions.
 *
 * Model: a linear embedding e = W x with W ∈ R^{2×8} (no bias — distances are translation invariant).
 * Training: mini-batch SGD on either
 *   contrastive (Hadsell, Chopra & LeCun 2006):  similar pair ½D²,  dissimilar pair ½·max(0, m − D)²
 *   triplet (Schroff et al. 2015, with Euclidean distances):  max(0, ‖a − p‖ − ‖a − n‖ + m)
 * with negatives chosen at random, semi-hard or hard. Because e is linear in W, every gradient with
 * respect to W is an outer product  g (x_i − x_j)ᵀ  of an embedding-space gradient g and an input difference.
 */
import { gaussian, mulberry32 } from "./random";

export const K = 5;
export const D = 8;
export const SIGNAL_DIMS = 4;

export type Mode = "contrastive" | "triplet";
export type Mining = "random" | "semi-hard" | "hard";
export type Vec2 = [number, number];

export interface Dataset {
  Xtr: number[][];
  ytr: number[];
  Xte: number[][];
  yte: number[];
}

export function makeDataset(seed = 7, nTrain = 30, nTest = 20): Dataset {
  const r = mulberry32(seed);
  const mus = Array.from({ length: K }, () => Array.from({ length: D }, (_, d) => (d < SIGNAL_DIMS ? gaussian(r) * 1.6 : 0)));
  const sample = (n: number) => {
    const X: number[][] = [];
    const y: number[] = [];
    for (let c = 0; c < K; c++)
      for (let i = 0; i < n; i++) {
        X.push(Array.from({ length: D }, (_, d) => mus[c][d] + gaussian(r) * (d < SIGNAL_DIMS ? 0.45 : 2.0)));
        y.push(c);
      }
    return { X, y };
  };
  const tr = sample(nTrain);
  const te = sample(nTest);
  return { Xtr: tr.X, ytr: tr.y, Xte: te.X, yte: te.y };
}

export function embedOne(W: number[][], x: number[]): Vec2 {
  let a = 0, b = 0;
  for (let d = 0; d < x.length; d++) {
    a += W[0][d] * x[d];
    b += W[1][d] * x[d];
  }
  return [a, b];
}

export const embed = (W: number[][], X: number[][]) => X.map((x) => embedOne(W, x));
const dist = (u: Vec2, v: Vec2) => Math.hypot(u[0] - v[0], u[1] - v[1]);

/** 1-nearest-neighbour accuracy of query embeddings against reference embeddings. */
export function knnAccuracy(Eref: Vec2[], yref: number[], Eq: Vec2[], yq: number[], leaveOneOut = false) {
  let correct = 0;
  for (let i = 0; i < Eq.length; i++) {
    let best = Infinity, lab = -1;
    for (let j = 0; j < Eref.length; j++) {
      if (leaveOneOut && i === j) continue;
      const dx = Eq[i][0] - Eref[j][0], dy = Eq[i][1] - Eref[j][1];
      const d2 = dx * dx + dy * dy;
      if (d2 < best) {
        best = d2;
        lab = yref[j];
      }
    }
    if (lab === yq[i]) correct++;
  }
  return correct / Eq.length;
}

/** One sampled training example: anchor, positive, negative indices (into the training set). */
export interface Tuple {
  a: number;
  p: number;
  n: number;
  /** contrastive only: which of the two pairs was used ("positive" = (a,p), "negative" = (a,n)) */
  pair: "positive" | "negative" | null;
  dap: number;
  dan: number;
  loss: number;
  /** triplet: d(a,n) ≥ d(a,p) + m; contrastive negative pair: D ≥ m; contrastive positive pair: never (loss ½D² > 0 unless D = 0) */
  satisfied: boolean;
  /** −∂ℓ/∂e for the anchor, positive and negative embeddings (the direction each would move if it were free) */
  ga: Vec2;
  gp: Vec2;
  gn: Vec2;
}

export interface StepRecord {
  step: number;
  W: number[][];
  /** mean loss over the mini-batch drawn at this step (computed with W before the update) */
  loss: number;
  /** number of tuples in the batch with non-zero loss (the only ones that produce a gradient) */
  active: number;
  /** 1-NN accuracy of the held-out test points, using training embeddings as the reference set */
  knnTest: number;
  /** leave-one-out 1-NN accuracy on the training set */
  knnTrain: number;
  gradNorm: number;
  shown: Tuple;
}

export interface TrainConfig {
  mode: Mode;
  mining: Mining;
  margin: number;
  lr: number;
  steps: number;
  batch: number;
  seed: number;
}

export function train(data: Dataset, cfg: TrainConfig): StepRecord[] {
  const { Xtr, ytr, Xte, yte } = data;
  const N = ytr.length;
  const r = mulberry32(cfg.seed * 7919 + 1);
  let W = [0, 1].map(() => Array.from({ length: D }, () => gaussian(r) * 0.35));
  const sameIdx = ytr.map((c, a) => ytr.map((_, j) => j).filter((j) => ytr[j] === c && j !== a));
  const negIdx = ytr.map((c) => ytr.map((_, j) => j).filter((j) => ytr[j] !== c));
  const m = cfg.margin;
  const out: StepRecord[] = [];

  for (let s = 0; s <= cfg.steps; s++) {
    const E = embed(W, Xtr);
    const knnTest = knnAccuracy(E, ytr, embed(W, Xte), yte);
    const knnTrain = knnAccuracy(E, ytr, E, ytr, true);
    const G = [new Array(D).fill(0), new Array(D).fill(0)];
    const addOuter = (g: Vec2, i: number, j: number, sign: number) => {
      for (let d = 0; d < D; d++) {
        const dx = Xtr[i][d] - Xtr[j][d];
        G[0][d] += sign * g[0] * dx;
        G[1][d] += sign * g[1] * dx;
      }
    };
    let total = 0, active = 0;
    let shown: Tuple | null = null;

    for (let b = 0; b < cfg.batch; b++) {
      const a = Math.floor(r() * N);
      const same = sameIdx[a];
      const neg = negIdx[a];
      const p = same[Math.floor(r() * same.length)];
      const dn = neg.map((j) => dist(E[a], E[j]));
      const dap = dist(E[a], E[p]);
      let n: number;
      if (cfg.mining === "random") n = neg[Math.floor(r() * neg.length)];
      else if (cfg.mining === "hard") {
        let k = 0;
        for (let q = 1; q < dn.length; q++) if (dn[q] < dn[k]) k = q;
        n = neg[k];
      } else if (cfg.mode === "triplet") {
        // semi-hard: farther than the positive, but still inside the margin; take the closest such negative
        let k = -1;
        for (let q = 0; q < dn.length; q++) if (dn[q] > dap && dn[q] < dap + m && (k < 0 || dn[q] < dn[k])) k = q;
        n = k >= 0 ? neg[k] : neg[Math.floor(r() * neg.length)];
      } else {
        // contrastive "semi-hard": a random negative among those still inside the margin (the active ones)
        const cand: number[] = [];
        for (let q = 0; q < dn.length; q++) if (dn[q] < m) cand.push(q);
        n = cand.length ? neg[cand[Math.floor(r() * cand.length)]] : neg[Math.floor(r() * neg.length)];
      }
      const dan = dist(E[a], E[n]);
      const ua: Vec2 = [(E[a][0] - E[p][0]) / Math.max(dap, 1e-9), (E[a][1] - E[p][1]) / Math.max(dap, 1e-9)];
      const va: Vec2 = [(E[a][0] - E[n][0]) / Math.max(dan, 1e-9), (E[a][1] - E[n][1]) / Math.max(dan, 1e-9)];
      let tuple: Tuple;

      if (cfg.mode === "triplet") {
        const L = Math.max(0, dap - dan + m);
        const on = L > 0;
        if (on) {
          active++;
          addOuter(ua, a, p, 1); //  ∂ℓ/∂e_a ⊇ +u,  ∂ℓ/∂e_p = −u   →   u (x_a − x_p)ᵀ
          addOuter(va, a, n, -1); // ∂ℓ/∂e_a ⊇ −v,  ∂ℓ/∂e_n = +v   →  −v (x_a − x_n)ᵀ
        }
        total += L;
        const z: Vec2 = [0, 0];
        tuple = {
          a, p, n, pair: null, dap, dan, loss: L, satisfied: !on,
          ga: on ? [-(ua[0] - va[0]), -(ua[1] - va[1])] : z,
          gp: on ? [ua[0], ua[1]] : z,
          gn: on ? [-va[0], -va[1]] : z,
        };
      } else {
        const positive = r() < 0.5;
        const j = positive ? p : n;
        const Dd = positive ? dap : dan;
        const diff: Vec2 = [E[a][0] - E[j][0], E[a][1] - E[j][1]];
        let L: number, g: Vec2;
        if (positive) {
          L = 0.5 * Dd * Dd;
          g = diff; // ∂(½D²)/∂e_a = e_a − e_j
        } else {
          const h = Math.max(0, m - Dd);
          L = 0.5 * h * h;
          g = [(-h * diff[0]) / Math.max(Dd, 1e-9), (-h * diff[1]) / Math.max(Dd, 1e-9)]; // −(m − D)(e_a − e_j)/D
          if (h > 0) active++;
        }
        if (positive && L > 0) active++;
        addOuter(g, a, j, 1);
        total += L;
        const z: Vec2 = [0, 0];
        tuple = {
          a, p, n, pair: positive ? "positive" : "negative", dap, dan, loss: L,
          satisfied: !positive && Dd >= m,
          ga: [-g[0], -g[1]],
          gp: positive ? [g[0], g[1]] : z,
          gn: positive ? z : [g[0], g[1]],
        };
      }
      if (!shown) shown = tuple;
    }

    let gn2 = 0;
    for (let i = 0; i < 2; i++) for (let d = 0; d < D; d++) gn2 += (G[i][d] / cfg.batch) ** 2;
    out.push({ step: s, W, loss: total / cfg.batch, active, knnTest, knnTrain, gradNorm: Math.sqrt(gn2), shown: shown! });
    W = W.map((row, i) => row.map((w, d) => w - (cfg.lr * G[i][d]) / cfg.batch));
  }
  return out;
}
