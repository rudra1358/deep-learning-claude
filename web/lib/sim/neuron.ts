/**
 * Pure math for the artificial-neuron / perceptron simulator (no React).
 *
 * Conventions (shared with the m03/artificial-neuron lesson):
 *   z = w1·x1 + w2·x2 + b,   prediction ŷ = 1 if z >= 0 else 0  (ties go to class 1)
 *   perceptron rule:  e = y − ŷ,  Δw = η·e·x,  Δb = η·e
 *   samples are visited in the fixed order (0,0), (0,1), (1,0), (1,1).
 */

export type GateName = "AND" | "OR" | "NAND" | "XOR";
export type ActName = "step" | "sigmoid" | "tanh" | "relu" | "linear";

export interface Sample {
  x: [number, number];
  y: 0 | 1;
}

const INPUTS: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const TRUTH: Record<GateName, (a: number, b: number) => 0 | 1> = {
  AND: (a, b) => (a && b ? 1 : 0),
  OR: (a, b) => (a || b ? 1 : 0),
  NAND: (a, b) => (a && b ? 0 : 1),
  XOR: (a, b) => (a !== b ? 1 : 0),
};

export const GATES: GateName[] = ["AND", "OR", "NAND", "XOR"];

export function gateData(g: GateName): Sample[] {
  return INPUTS.map(([a, b]) => ({ x: [a, b], y: TRUTH[g](a, b) }));
}

export function weightedSum(w: [number, number], b: number, x: [number, number]) {
  return w[0] * x[0] + w[1] * x[1] + b;
}

export function predict(z: number): 0 | 1 {
  return z >= 0 ? 1 : 0;
}

export function activate(act: ActName, z: number): number {
  switch (act) {
    case "step":
      return z >= 0 ? 1 : 0;
    case "sigmoid":
      return 1 / (1 + Math.exp(-z));
    case "tanh":
      return Math.tanh(z);
    case "relu":
      return Math.max(0, z);
    case "linear":
      return z;
  }
}

/** One perceptron update on one sample. */
export interface PerceptronStep {
  epoch: number; // 1-based
  index: number; // sample index within the epoch, 0..3
  x: [number, number];
  y: 0 | 1;
  wBefore: [number, number];
  bBefore: number;
  z: number;
  yhat: 0 | 1;
  err: number; // y − ŷ ∈ {−1, 0, 1}
  dw: [number, number];
  db: number;
  wAfter: [number, number];
  bAfter: number;
  /** errors made so far in this epoch, including this sample */
  epochErrors: number;
}

export interface PerceptronTrace {
  steps: PerceptronStep[];
  /** epoch (1-based) in which zero errors were made, or null if it never happened */
  convergedEpoch: number | null;
  /** number of mistakes per epoch */
  errorsPerEpoch: number[];
}

/**
 * Run the perceptron learning rule sample by sample until a full epoch makes no mistakes
 * (converged) or `maxEpochs` epochs have been run.
 */
export function perceptronTrace(data: Sample[], w0: [number, number], b0: number, lr: number, maxEpochs = 30): PerceptronTrace {
  const steps: PerceptronStep[] = [];
  const errorsPerEpoch: number[] = [];
  let w: [number, number] = [w0[0], w0[1]];
  let b = b0;
  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    let errs = 0;
    data.forEach((s, index) => {
      const z = weightedSum(w, b, s.x);
      const yhat = predict(z);
      const err = s.y - yhat;
      if (err !== 0) errs++;
      const dw: [number, number] = [lr * err * s.x[0], lr * err * s.x[1]];
      const db = lr * err;
      const wAfter: [number, number] = [w[0] + dw[0], w[1] + dw[1]];
      const bAfter = b + db;
      steps.push({ epoch, index, x: s.x, y: s.y, wBefore: w, bBefore: b, z, yhat, err, dw, db, wAfter, bAfter, epochErrors: errs });
      w = wAfter;
      b = bAfter;
    });
    errorsPerEpoch.push(errs);
    if (errs === 0) return { steps, convergedEpoch: epoch, errorsPerEpoch };
  }
  return { steps, convergedEpoch: null, errorsPerEpoch };
}

/**
 * Endpoints of the line w1·x1 + w2·x2 + b = 0 clipped to the square [lo, hi]².
 * Returns null when both weights are (numerically) zero — there is no line.
 */
export function boundarySegment(w: [number, number], b: number, lo: number, hi: number): [[number, number], [number, number]] | null {
  const [w1, w2] = w;
  if (Math.abs(w1) < 1e-12 && Math.abs(w2) < 1e-12) return null;
  const pts: [number, number][] = [];
  const add = (p: [number, number]) => {
    if (p[0] >= lo - 1e-9 && p[0] <= hi + 1e-9 && p[1] >= lo - 1e-9 && p[1] <= hi + 1e-9) {
      if (!pts.some((q) => Math.abs(q[0] - p[0]) < 1e-9 && Math.abs(q[1] - p[1]) < 1e-9)) pts.push(p);
    }
  };
  if (Math.abs(w2) > 1e-12) {
    add([lo, -(w1 * lo + b) / w2]);
    add([hi, -(w1 * hi + b) / w2]);
  }
  if (Math.abs(w1) > 1e-12) {
    add([-(w2 * lo + b) / w1, lo]);
    add([-(w2 * hi + b) / w1, hi]);
  }
  return pts.length >= 2 ? [pts[0], pts[1]] : null;
}

/**
 * Polygon (list of corners) of the part of [lo, hi]² where z >= 0, i.e. predicted class 1.
 * Uses Sutherland–Hodgman clipping of the square against the half-plane.
 */
export function positiveRegion(w: [number, number], b: number, lo: number, hi: number): [number, number][] {
  const square: [number, number][] = [
    [lo, lo],
    [hi, lo],
    [hi, hi],
    [lo, hi],
  ];
  const f = (p: [number, number]) => w[0] * p[0] + w[1] * p[1] + b;
  const out: [number, number][] = [];
  for (let i = 0; i < square.length; i++) {
    const p = square[i];
    const q = square[(i + 1) % square.length];
    const fp = f(p);
    const fq = f(q);
    if (fp >= 0) out.push(p);
    if ((fp >= 0) !== (fq >= 0)) {
      const t = fp / (fp - fq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  }
  return out;
}
