/**
 * Pure NumPy-style broadcasting logic for the Broadcasting simulator (no React).
 *
 * Shapes are arrays of positive integers (here 1–3 axes, each 1–5). Arrays are stored
 * flat in row-major (C) order, exactly like a contiguous NumPy array.
 */
import { mulberry32 } from "./random";

export type Shape = number[];
export type Op = "+" | "-" | "*";

export interface NDArray {
  shape: Shape;
  data: number[];
}

export const OPS: { op: Op; label: string; fn: (a: number, b: number) => number }[] = [
  { op: "+", label: "+ (add)", fn: (a, b) => a + b },
  { op: "-", label: "− (subtract)", fn: (a, b) => a - b },
  { op: "*", label: "× (multiply)", fn: (a, b) => a * b },
];

export const opSymbol = (op: Op) => (op === "*" ? "×" : op === "-" ? "−" : "+");
export const applyOp = (op: Op, a: number, b: number) => (op === "+" ? a + b : op === "-" ? a - b : a * b);

/** Python-style shape text: (3,) for one axis, (4, 3) otherwise. */
export const shapeStr = (s: Shape) => (s.length === 1 ? `(${s[0]},)` : `(${s.join(", ")})`);

export const sizeOf = (s: Shape) => s.reduce((p, d) => p * d, 1);

/** Row-major multi-index of flat position `flat` in an array of shape `s`. */
export function unravel(flat: number, s: Shape): number[] {
  const idx = new Array(s.length).fill(0);
  for (let k = s.length - 1; k >= 0; k--) {
    idx[k] = flat % s[k];
    flat = Math.floor(flat / s[k]);
  }
  return idx;
}

/** Inverse of unravel. */
export function ravel(idx: number[], s: Shape): number {
  let f = 0;
  for (let k = 0; k < s.length; k++) f = f * s[k] + idx[k];
  return f;
}

export type Rule = "equal" | "a-is-1" | "b-is-1" | "incompatible";

/** One column of the right-aligned shape table. `a`/`b` are null where that array has no axis (implicitly 1). */
export interface DimPair {
  /** column index in the aligned table, 0 = leftmost */
  col: number;
  a: number | null;
  b: number | null;
  rule: Rule;
  /** result size on this axis (null if incompatible) */
  out: number | null;
}

/** Right-align two shapes, padding the shorter on the left with null (treated as 1). */
export function align(a: Shape, b: Shape) {
  const rank = Math.max(a.length, b.length);
  const pad = (s: Shape) => [...new Array<number | null>(rank - s.length).fill(null), ...s];
  return { rank, a: pad(a), b: pad(b) };
}

/** Compare every aligned axis pair (the order is left-to-right in the table; the simulator walks it right-to-left). */
export function comparePairs(a: Shape, b: Shape): DimPair[] {
  const al = align(a, b);
  return al.a.map((av, col) => {
    const bv = al.b[col];
    const x = av ?? 1;
    const y = bv ?? 1;
    let rule: Rule;
    if (x === y) rule = "equal";
    else if (x === 1) rule = "a-is-1";
    else if (y === 1) rule = "b-is-1";
    else rule = "incompatible";
    return { col, a: av, b: bv, rule, out: rule === "incompatible" ? null : Math.max(x, y) };
  });
}

/** The broadcast result shape, or null if the shapes are incompatible. */
export function broadcastShape(a: Shape, b: Shape): Shape | null {
  const pairs = comparePairs(a, b);
  if (pairs.some((p) => p.rule === "incompatible")) return null;
  return pairs.map((p) => p.out as number);
}

/** The message NumPy itself raises for incompatible shapes. */
export const numpyError = (a: Shape, b: Shape) =>
  `ValueError: operands could not be broadcast together with shapes (${a.join(",")}${a.length === 1 ? "," : ""}) (${b.join(",")}${b.length === 1 ? "," : ""})`;

/**
 * Map an index into the result to the index into a source array of shape `src`.
 * The result has rank R >= src.length; the source's axes are the LAST src.length axes.
 * On a size-1 source axis the index is always 0 — that is the "stretch" (stride 0).
 */
export function sourceIndex(outIdx: number[], src: Shape): number[] {
  const offset = outIdx.length - src.length;
  return src.map((d, k) => (d === 1 ? 0 : outIdx[offset + k]));
}

/** Is the result position `outIdx` a virtual repeat (not the stored element itself) for source `src`? */
export function isRepeat(outIdx: number[], src: Shape): boolean {
  const offset = outIdx.length - src.length;
  // A position is a repeat if it uses index > 0 on an axis where the source has size 1,
  // or a non-zero index on a padded (missing) leading axis.
  for (let k = 0; k < outIdx.length; k++) {
    const sk = k - offset;
    const d = sk < 0 ? 1 : src[sk];
    if (d === 1 && outIdx[k] > 0) return true;
  }
  return false;
}

export function getAt(arr: NDArray, idx: number[]): number {
  return arr.data[ravel(idx, arr.shape)];
}

/** Small seeded integers so the arithmetic can be checked by eye. */
export function seededArray(shape: Shape, seed: number, lo: number, hi: number): NDArray {
  const r = mulberry32(seed);
  const n = sizeOf(shape);
  return { shape: [...shape], data: Array.from({ length: n }, () => lo + Math.floor(r() * (hi - lo + 1))) };
}

/** Element-wise op with broadcasting. Returns null for incompatible shapes. */
export function broadcastApply(op: Op, A: NDArray, B: NDArray): NDArray | null {
  const out = broadcastShape(A.shape, B.shape);
  if (!out) return null;
  const n = sizeOf(out);
  const data = new Array<number>(n);
  for (let f = 0; f < n; f++) {
    const idx = unravel(f, out);
    data[f] = applyOp(op, getAt(A, sourceIndex(idx, A.shape)), getAt(B, sourceIndex(idx, B.shape)));
  }
  return { shape: out, data };
}

/** Python-style index text: [1, 2]; a 0-d index (never used here) would be []. */
export const idxStr = (idx: number[]) => `[${idx.join(", ")}]`;

export interface Preset {
  id: string;
  label: string;
  a: Shape;
  b: Shape;
  op: Op;
  note: string;
}

export const PRESETS: Preset[] = [
  { id: "bias", label: "Bias add: (4,3) + (3,)", a: [4, 3], b: [3], op: "+", note: "A batch of 4 examples with 3 features plus one bias per feature." },
  { id: "colnorm", label: "Column normalise: (4,3) − (1,3)", a: [4, 3], b: [1, 3], op: "-", note: "Subtract each column's mean (kept as a (1,3) row via keepdims=True)." },
  { id: "outer", label: "Outer product: (4,1) × (1,3)", a: [4, 1], b: [1, 3], op: "*", note: "A column times a row gives every pairwise product: a 4×3 table." },
  { id: "trap", label: "The trap: (4,1) + (4,)", a: [4, 1], b: [4], op: "+", note: "Looks like adding two length-4 vectors, but silently produces a 4×4 matrix." },
  { id: "bad", label: "Incompatible: (3,4) + (3,)", a: [3, 4], b: [3], op: "+", note: "The trailing sizes 4 and 3 differ and neither is 1." },
  { id: "3d", label: "3-D: (2,1,3) + (4,1)", a: [2, 1, 3], b: [4, 1], op: "+", note: "Both arrays stretch: A along the middle axis, B along the last and the missing first axis." },
];

/** Value ranges: A in 0..9, B in 1..5 — results stay within two digits. */
export const A_RANGE: [number, number] = [0, 9];
export const B_RANGE: [number, number] = [1, 5];
