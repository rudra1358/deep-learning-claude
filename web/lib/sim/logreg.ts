/**
 * Pure math for the logistic-regression simulator (no React).
 *
 * Model:   p(x) = σ(z),  z = w1·x1 + w2·x2 + b
 * Loss:    J(w, b) = BCE + (λ/2)·‖w‖²,  BCE = −(1/N) Σ [ y log p + (1−y) log(1−p) ]
 * Gradient (full batch):
 *          ∂J/∂w = (1/N) Σ (p_i − y_i) x_i + λ w,     ∂J/∂b = (1/N) Σ (p_i − y_i)
 *          (the bias is not regularised, matching scikit-learn and the lesson)
 * Update:  w ← w − η ∂J/∂w,  b ← b − η ∂J/∂b, starting from w = 0, b = 0.
 *
 * Datasets live in the square [−3, 3]² and come from a seeded PRNG, so every reload is identical.
 */

import { gaussian, mulberry32 } from "./random";

export type Label = 0 | 1;
export interface Pt {
  x1: number;
  x2: number;
  y: Label;
}

export const DOMAIN = 3; // points and plot live in [−DOMAIN, DOMAIN]²

export type PresetId = "separable" | "overlap" | "imbalanced" | "circles";

export const PRESETS: { id: PresetId; label: string; note: string }[] = [
  { id: "separable", label: "Separable blobs", note: "Two well-separated Gaussian clusters — a line can classify every point." },
  { id: "overlap", label: "Overlapping blobs", note: "Two Gaussian clusters that overlap — no line can be perfect." },
  { id: "imbalanced", label: "Imbalanced (9 : 1)", note: "45 points of class 0, 5 of class 1 — accuracy alone is misleading." },
  { id: "circles", label: "Circles (not linear)", note: "Class 1 inside a ring of class 0 — no line can separate them." },
];

const clampD = (v: number) => Math.max(-DOMAIN + 0.1, Math.min(DOMAIN - 0.1, v));

function blob(rand: () => number, n: number, cx: number, cy: number, sd: number, y: Label): Pt[] {
  return Array.from({ length: n }, () => ({ x1: clampD(cx + sd * gaussian(rand)), x2: clampD(cy + sd * gaussian(rand)), y }));
}

export function makePreset(id: PresetId, seed = 3): Pt[] {
  const rand = mulberry32(seed);
  switch (id) {
    case "separable":
      return [...blob(rand, 20, -1.3, -0.9, 0.5, 0), ...blob(rand, 20, 1.2, 1.0, 0.5, 1)];
    case "overlap":
      return [...blob(rand, 25, -0.6, -0.4, 0.9, 0), ...blob(rand, 25, 0.6, 0.5, 0.9, 1)];
    case "imbalanced":
      return [...blob(rand, 45, -0.5, -0.3, 0.9, 0), ...blob(rand, 5, 0.6, 0.5, 0.6, 1)];
    case "circles": {
      const pts: Pt[] = [];
      for (let i = 0; i < 24; i++) {
        const a = rand() * 2 * Math.PI, r = 0.7 * Math.sqrt(rand());
        pts.push({ x1: r * Math.cos(a), x2: r * Math.sin(a), y: 1 });
      }
      for (let i = 0; i < 30; i++) {
        const a = rand() * 2 * Math.PI, r = 1.7 + 0.35 * gaussian(rand);
        pts.push({ x1: clampD(r * Math.cos(a)), x2: clampD(r * Math.sin(a)), y: 0 });
      }
      return pts;
    }
  }
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export interface LogRegState {
  /** iteration index t (parameters after t updates) */
  t: number;
  w: [number, number];
  b: number;
  /** gradient of the full objective J at (w, b) */
  gw: [number, number];
  gb: number;
  /** mean binary cross-entropy (without the L2 term) */
  bce: number;
  /** BCE + (λ/2)‖w‖² — the objective being minimised */
  objective: number;
  accuracy: number;
  /** points whose prediction (p > 0.5) disagrees with the label */
  wrong: number;
}

/** −log σ(z) computed stably (softplus(−z)). */
const nlogSig = (z: number) => (z >= 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z)));

export function evaluate(pts: Pt[], w: [number, number], b: number, lambda: number): Omit<LogRegState, "t"> {
  const N = Math.max(1, pts.length);
  let bce = 0, g1 = 0, g2 = 0, gb = 0, wrong = 0;
  for (const p of pts) {
    const z = w[0] * p.x1 + w[1] * p.x2 + b;
    const pr = sigmoid(z);
    // y log p + (1−y) log(1−p) with log(1−σ(z)) = log σ(−z)
    bce += p.y === 1 ? nlogSig(z) : nlogSig(-z);
    const e = pr - p.y; // (p − y): the error signal of BCE + sigmoid
    g1 += e * p.x1;
    g2 += e * p.x2;
    gb += e;
    if ((pr > 0.5 ? 1 : 0) !== p.y) wrong++;
  }
  bce /= N;
  const reg = 0.5 * lambda * (w[0] * w[0] + w[1] * w[1]);
  return {
    w,
    b,
    gw: [g1 / N + lambda * w[0], g2 / N + lambda * w[1]],
    gb: gb / N,
    bce,
    objective: bce + reg,
    accuracy: pts.length ? 1 - wrong / pts.length : 1,
    wrong,
  };
}

/** Run `steps` full-batch gradient-descent updates from w = 0, b = 0; returns states 0..steps. */
export function trainLogReg(pts: Pt[], lr: number, lambda: number, steps: number): LogRegState[] {
  let w: [number, number] = [0, 0];
  let b = 0;
  const out: LogRegState[] = [];
  for (let t = 0; t <= steps; t++) {
    const s = evaluate(pts, w, b, lambda);
    out.push({ t, ...s });
    // stop updating once the numbers are no longer finite (divergence); keep the array length fixed
    const ok = Number.isFinite(s.objective) && Math.abs(s.w[0]) < 1e6 && Math.abs(s.w[1]) < 1e6;
    if (ok) {
      w = [w[0] - lr * s.gw[0], w[1] - lr * s.gw[1]];
      b = b - lr * s.gb;
    }
  }
  return out;
}

/** Fraction of the majority class — the accuracy of always predicting it. */
export function majorityBaseline(pts: Pt[]) {
  if (!pts.length) return 0;
  const ones = pts.reduce((a, p) => a + p.y, 0);
  return Math.max(ones, pts.length - ones) / pts.length;
}

/** Endpoints of the line w1 x1 + w2 x2 + b = c inside the square [−D, D]², or null if it misses the square. */
export function lineInBox(w: [number, number], b: number, c = 0, D = DOMAIN): [[number, number], [number, number]] | null {
  const [a1, a2] = w;
  const pts: [number, number][] = [];
  const eps = 1e-12;
  if (Math.abs(a2) > eps) {
    for (const x1 of [-D, D]) {
      const x2 = (c - b - a1 * x1) / a2;
      if (x2 >= -D - 1e-9 && x2 <= D + 1e-9) pts.push([x1, x2]);
    }
  }
  if (Math.abs(a1) > eps) {
    for (const x2 of [-D, D]) {
      const x1 = (c - b - a2 * x2) / a1;
      if (x1 >= -D - 1e-9 && x1 <= D + 1e-9) pts.push([x1, x2]);
    }
  }
  if (pts.length < 2) return null;
  // pick the two most distant intersection points (corners can be counted twice)
  let best: [[number, number], [number, number]] = [pts[0], pts[1]];
  let bd = -1;
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) {
      const d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2;
      if (d > bd) {
        bd = d;
        best = [pts[i], pts[j]];
      }
    }
  return bd > 1e-12 ? best : null;
}
