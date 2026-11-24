/**
 * Pure geometry for the object-detection simulator: IoU, anchor grids and matching,
 * box encoding, greedy NMS (as a step-by-step trace) and precision/recall at an IoU threshold.
 * Boxes are corner format (x1, y1, x2, y2) in pixel units with x1 < x2, y1 < y2.
 */
import { mulberry32, gaussian } from "./random";

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export const area = (b: Box) => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

/** Intersection rectangle, or null if the boxes do not overlap. */
export function intersection(a: Box, b: Box): Box | null {
  const r = { x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1), x2: Math.min(a.x2, b.x2), y2: Math.min(a.y2, b.y2) };
  return r.x2 > r.x1 && r.y2 > r.y1 ? r : null;
}

export interface IouParts {
  areaA: number;
  areaB: number;
  inter: number;
  union: number;
  iou: number;
  interBox: Box | null;
}

export function iouParts(a: Box, b: Box): IouParts {
  const interBox = intersection(a, b);
  const inter = interBox ? area(interBox) : 0;
  const areaA = area(a);
  const areaB = area(b);
  const union = areaA + areaB - inter;
  return { areaA, areaB, inter, union, iou: union > 0 ? inter / union : 0, interBox };
}

export const iou = (a: Box, b: Box) => iouParts(a, b).iou;

export const toCxcywh = (b: Box) => ({ cx: (b.x1 + b.x2) / 2, cy: (b.y1 + b.y2) / 2, w: b.x2 - b.x1, h: b.y2 - b.y1 });

export function fromCxcywh(cx: number, cy: number, w: number, h: number): Box {
  return { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 };
}

/* ------------------------------------------------------------------ */
/* Anchors                                                              */
/* ------------------------------------------------------------------ */

export interface Anchor {
  box: Box;
  row: number;
  col: number;
  scale: number;
  ratio: number; // h / w
}

/** One anchor per (cell, scale, ratio): centre at the cell centre, area scale², aspect ratio h/w = ratio. */
export function makeAnchors(width: number, height: number, stride: number, scales: number[], ratios: number[]): Anchor[] {
  const out: Anchor[] = [];
  const rows = Math.floor(height / stride);
  const cols = Math.floor(width / stride);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (const s of scales)
        for (const q of ratios) {
          const w = s / Math.sqrt(q);
          const h = s * Math.sqrt(q);
          out.push({ box: fromCxcywh((c + 0.5) * stride, (r + 0.5) * stride, w, h), row: r, col: c, scale: s, ratio: q });
        }
  return out;
}

export type AnchorLabel = "pos" | "neg" | "ignore";

export interface AnchorMatch {
  bestIou: number;
  gt: number; // index of the best-matching ground-truth box (-1 if there is none)
  label: AnchorLabel;
  forced: boolean; // positive only because it is some GT's best anchor
}

/**
 * Standard max-IoU assignment: IoU >= posThr → positive, IoU < negThr → negative, in between → ignored.
 * With forceBest, each GT's highest-IoU anchor is made positive even below posThr (so no object is left unmatched).
 */
export function matchAnchors(anchors: Anchor[], gts: Box[], posThr: number, negThr: number, forceBest: boolean): AnchorMatch[] {
  const res: AnchorMatch[] = anchors.map((a) => {
    let best = 0;
    let gi = -1;
    gts.forEach((g, j) => {
      const v = iou(a.box, g);
      if (v > best) {
        best = v;
        gi = j;
      }
    });
    const label: AnchorLabel = best >= posThr ? "pos" : best < negThr ? "neg" : "ignore";
    return { bestIou: best, gt: gi, label, forced: false };
  });
  if (forceBest)
    gts.forEach((g, j) => {
      let bi = -1;
      let bv = -1;
      anchors.forEach((a, k) => {
        const v = iou(a.box, g);
        if (v > bv) {
          bv = v;
          bi = k;
        }
      });
      if (bi >= 0 && bv > 0) {
        const m = res[bi];
        if (m.label !== "pos") res[bi] = { bestIou: bv, gt: j, label: "pos", forced: true };
        else if (m.gt !== j && bv >= m.bestIou) res[bi] = { ...m, gt: j, bestIou: bv };
      }
    });
  return res;
}

export interface Targets {
  tx: number;
  ty: number;
  tw: number;
  th: number;
}

/** Faster R-CNN box parametrisation: centre offsets in units of anchor size, log size ratios. */
export function encode(gt: Box, anchor: Box): Targets {
  const g = toCxcywh(gt);
  const a = toCxcywh(anchor);
  return { tx: (g.cx - a.cx) / a.w, ty: (g.cy - a.cy) / a.h, tw: Math.log(g.w / a.w), th: Math.log(g.h / a.h) };
}

export function decode(t: Targets, anchor: Box): Box {
  const a = toCxcywh(anchor);
  return fromCxcywh(a.cx + t.tx * a.w, a.cy + t.ty * a.h, a.w * Math.exp(t.tw), a.h * Math.exp(t.th));
}

/* ------------------------------------------------------------------ */
/* Candidates and NMS                                                   */
/* ------------------------------------------------------------------ */

export interface Candidate {
  box: Box;
  score: number;
  source: number; // GT index it was generated around, -1 for a background false positive
}

/**
 * Seeded noisy detections. Offsets are stored relative to each GT's size, so the cloud of
 * candidates follows a GT box when the user drags it.
 */
export function makeCandidates(gts: Box[], seed: number, perObject: number, falsePos: number, width: number, height: number): Candidate[] {
  const rand = mulberry32(seed);
  const out: Candidate[] = [];
  gts.forEach((g, j) => {
    const c = toCxcywh(g);
    for (let k = 0; k < perObject; k++) {
      const jitter = k === 0 ? 0.04 : 0.1; // the first candidate is a tight, confident one
      const cx = c.cx + gaussian(rand) * jitter * c.w;
      const cy = c.cy + gaussian(rand) * jitter * c.h;
      const w = c.w * Math.exp(gaussian(rand) * jitter);
      const h = c.h * Math.exp(gaussian(rand) * jitter);
      const score = k === 0 ? 0.8 + 0.18 * rand() : 0.3 + 0.6 * rand();
      out.push({ box: clampBox(fromCxcywh(cx, cy, w, h), width, height), score, source: j });
    }
  });
  for (let k = 0; k < falsePos; k++) {
    const w = 25 + rand() * 60;
    const h = 25 + rand() * 60;
    const x1 = rand() * (width - w);
    const y1 = rand() * (height - h);
    out.push({ box: { x1, y1, x2: x1 + w, y2: y1 + h }, score: 0.1 + 0.45 * rand(), source: -1 });
  }
  return out;
}

export function clampBox(b: Box, width: number, height: number): Box {
  return { x1: Math.max(0, b.x1), y1: Math.max(0, b.y1), x2: Math.min(width, b.x2), y2: Math.min(height, b.y2) };
}

export type NmsEvent =
  | { kind: "pick"; idx: number; remaining: number[] }
  | { kind: "suppress"; idx: number; comps: { idx: number; iou: number; suppressed: boolean }[] };

export interface NmsTrace {
  order: number[]; // candidate indices sorted by score, descending
  events: NmsEvent[];
  keep: number[];
}

/** Greedy NMS recorded as alternating pick / suppress events. */
export function nmsTrace(cands: Candidate[], idxs: number[], thr: number): NmsTrace {
  const order = [...idxs].sort((a, b) => cands[b].score - cands[a].score || a - b);
  let remaining = [...order];
  const events: NmsEvent[] = [];
  const keep: number[] = [];
  while (remaining.length) {
    const i = remaining[0];
    keep.push(i);
    events.push({ kind: "pick", idx: i, remaining: [...remaining] });
    const rest = remaining.slice(1);
    const comps = rest.map((j) => {
      const v = iou(cands[i].box, cands[j].box);
      return { idx: j, iou: v, suppressed: v > thr };
    });
    events.push({ kind: "suppress", idx: i, comps });
    remaining = comps.filter((c) => !c.suppressed).map((c) => c.idx);
  }
  return { order, events, keep };
}

export interface PrResult {
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
}

/** Greedy score-ordered matching of detections to GT at an IoU threshold (each GT matched at most once). */
export function precisionRecall(cands: Candidate[], idxs: number[], gts: Box[], thr = 0.5): PrResult {
  const order = [...idxs].sort((a, b) => cands[b].score - cands[a].score);
  const used = gts.map(() => false);
  let tp = 0;
  for (const i of order) {
    let best = -1;
    let bv = 0;
    gts.forEach((g, j) => {
      const v = iou(cands[i].box, g);
      if (v > bv) {
        bv = v;
        best = j;
      }
    });
    if (best >= 0 && bv >= thr && !used[best]) {
      used[best] = true;
      tp++;
    }
  }
  const fp = order.length - tp;
  const fn = gts.length - tp;
  return { tp, fp, fn, precision: order.length ? tp / order.length : 0, recall: gts.length ? tp / gts.length : 0 };
}
