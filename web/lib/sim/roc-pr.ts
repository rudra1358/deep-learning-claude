/**
 * Pure math for the "Thresholds, ROC and PR Curves" simulator (no React).
 *
 * Data:    a binary classifier's scores for n examples. Positives ~ N(+d/2, 1), negatives ~ N(−d/2, 1),
 *          where d is the separation (in standard deviations). The positive fraction π sets the imbalance.
 *          Samples come from a seeded PRNG (mulberry32 + Box–Muller), so a given setting is identical on every reload.
 * Rule:    predict positive  ⇔  score ≥ t.
 * Metrics: TPR = recall = TP/(TP+FN), FPR = FP/(FP+TN), precision = TP/(TP+FP) (undefined when nothing is
 *          predicted positive), F1 = 2PR/(P+R).
 * Curves:  ROC = (FPR, TPR) and PR = (recall, precision), one point per distinct score used as the threshold.
 *          AUC = trapezoidal area under the ROC points (equals the Mann–Whitney rank statistic, ties counted ½).
 *          AP  = Σ_k (R_k − R_{k−1}) P_k  — the step-wise definition used by scikit-learn's average_precision_score.
 */

import { gaussian, mulberry32 } from "./random";

export interface ScoreSet {
  scores: number[];
  labels: (0 | 1)[];
  nPos: number;
  nNeg: number;
}

export interface Confusion {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface Metrics extends Confusion {
  tpr: number; // recall
  fpr: number;
  precision: number | null; // null when TP + FP = 0
  f1: number; // 0 when precision is undefined or P + R = 0
  accuracy: number;
  predictedPos: number;
}

export interface CurvePoint {
  x: number;
  y: number;
  t: number; // threshold that produces this point (+Infinity for the "predict nothing" corner)
}

/** Draw n scores: round(n·π) positives from N(+d/2, 1), the rest negatives from N(−d/2, 1). */
export function makeScores(separation: number, posFrac: number, n = 1000, seed = 7): ScoreSet {
  const rand = mulberry32(seed);
  const nPos = Math.max(1, Math.min(n - 1, Math.round(n * posFrac)));
  const nNeg = n - nPos;
  const scores: number[] = [];
  const labels: (0 | 1)[] = [];
  // Draw the same underlying standard normals regardless of π / d, so moving a slider morphs the data smoothly.
  const z = Array.from({ length: n }, () => gaussian(rand));
  for (let i = 0; i < nPos; i++) {
    scores.push(separation / 2 + z[i]);
    labels.push(1);
  }
  for (let i = 0; i < nNeg; i++) {
    scores.push(-separation / 2 + z[n - 1 - i]);
    labels.push(0);
  }
  return { scores, labels, nPos, nNeg };
}

export function confusionAt(set: ScoreSet, t: number): Confusion {
  let tp = 0, fp = 0;
  for (let i = 0; i < set.scores.length; i++) {
    if (set.scores[i] >= t) {
      if (set.labels[i] === 1) tp++;
      else fp++;
    }
  }
  return { tp, fp, fn: set.nPos - tp, tn: set.nNeg - fp };
}

export function metricsFrom(c: Confusion): Metrics {
  const P = c.tp + c.fn, N = c.fp + c.tn;
  const tpr = P ? c.tp / P : 0;
  const fpr = N ? c.fp / N : 0;
  const predictedPos = c.tp + c.fp;
  const precision = predictedPos ? c.tp / predictedPos : null;
  const f1 = precision !== null && precision + tpr > 0 ? (2 * precision * tpr) / (precision + tpr) : 0;
  return { ...c, tpr, fpr, precision, f1, accuracy: (c.tp + c.tn) / (P + N), predictedPos };
}

/** Indices sorted by descending score. */
function order(set: ScoreSet): number[] {
  return set.scores.map((_, i) => i).sort((a, b) => set.scores[b] - set.scores[a]);
}

/**
 * Sweep the threshold down through every distinct score. Returns the ROC points (starting at (0,0) for t = +∞
 * and ending at (1,1)) and the PR points (one per distinct threshold, recall increasing).
 */
export function curves(set: ScoreSet): { roc: CurvePoint[]; pr: CurvePoint[]; auc: number; ap: number } {
  const idx = order(set);
  const roc: CurvePoint[] = [{ x: 0, y: 0, t: Infinity }];
  const pr: CurvePoint[] = [];
  let tp = 0, fp = 0, k = 0;
  let auc = 0, ap = 0, prevR = 0;
  while (k < idx.length) {
    const t = set.scores[idx[k]];
    // consume every example tied at this score before emitting a point
    while (k < idx.length && set.scores[idx[k]] === t) {
      if (set.labels[idx[k]] === 1) tp++;
      else fp++;
      k++;
    }
    const tpr = tp / set.nPos, fpr = fp / set.nNeg;
    const last = roc[roc.length - 1];
    auc += ((fpr - last.x) * (tpr + last.y)) / 2; // trapezoid
    roc.push({ x: fpr, y: tpr, t });
    const prec = tp / (tp + fp);
    ap += (tpr - prevR) * prec;
    prevR = tpr;
    pr.push({ x: tpr, y: prec, t });
  }
  return { roc, pr, auc, ap };
}

/** Mann–Whitney form of the AUC: P(score of a random positive > score of a random negative), ties count ½. O(n log n). */
export function rankAuc(set: ScoreSet): number {
  const idx = set.scores.map((_, i) => i).sort((a, b) => set.scores[a] - set.scores[b]);
  // average ranks (1-based) with ties
  const rank = new Array<number>(idx.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && set.scores[idx[j + 1]] === set.scores[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) rank[idx[k]] = avg;
    i = j + 1;
  }
  let sumPos = 0;
  for (let k = 0; k < idx.length; k++) if (set.labels[k] === 1) sumPos += rank[k];
  return (sumPos - (set.nPos * (set.nPos + 1)) / 2) / (set.nPos * set.nNeg);
}

/** Histogram counts per class over [lo, hi) with `bins` equal bins (values outside are clamped into the end bins). */
export function histogram(set: ScoreSet, lo: number, hi: number, bins: number): { pos: number[]; neg: number[] } {
  const pos = new Array<number>(bins).fill(0);
  const neg = new Array<number>(bins).fill(0);
  const w = (hi - lo) / bins;
  for (let i = 0; i < set.scores.length; i++) {
    const b = Math.max(0, Math.min(bins - 1, Math.floor((set.scores[i] - lo) / w)));
    if (set.labels[i] === 1) pos[b]++;
    else neg[b]++;
  }
  return { pos, neg };
}

/** Threshold that maximises F1 over the swept points (ties → the higher threshold). */
export function bestF1(set: ScoreSet): { t: number; f1: number } {
  const { pr } = curves(set);
  let best = { t: Infinity, f1: 0 };
  for (const p of pr) {
    const f1 = p.x + p.y > 0 ? (2 * p.x * p.y) / (p.x + p.y) : 0;
    if (f1 > best.f1 + 1e-12) best = { t: p.t, f1 };
  }
  return best;
}
