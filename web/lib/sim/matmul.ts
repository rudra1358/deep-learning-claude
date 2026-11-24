/**
 * Pure helpers for the matrix-multiplication simulator (no React).
 *
 * C = A B is computed as n·m dot products, one per output cell:
 *   C[i][j] = Σ_p A[i][p] · B[p][j]
 * The simulator walks this sum one term (or one cell) at a time, so we expose
 * a step decoder that says exactly which cell / term is current and what the
 * running sum is at that moment.
 */
import { mulberry32, type Matrix } from "./random";

export type { Matrix };

export interface Shape {
  rows: number;
  cols: number;
}

export function shapeOf(M: Matrix): Shape {
  return { rows: M.length, cols: M.length ? M[0].length : 0 };
}

/** A (n×k) times B (k'×m) is defined only when k === k'. */
export function canMultiply(A: Matrix, B: Matrix): boolean {
  return shapeOf(A).cols === shapeOf(B).rows;
}

/** The k element-wise products A[i][p]·B[p][j] for one output cell. */
export function cellProducts(A: Matrix, B: Matrix, i: number, j: number): number[] {
  const k = shapeOf(A).cols;
  return Array.from({ length: k }, (_, p) => A[i][p] * B[p][j]);
}

/** Running (prefix) sums of the products — partial[p] = Σ_{q ≤ p} products[q]. */
export function runningSums(products: number[]): number[] {
  const out: number[] = [];
  let s = 0;
  for (const v of products) {
    s += v;
    out.push(s);
  }
  return out;
}

/** Full product via explicit dot products (same loop the stepper visualises). */
export function multiply(A: Matrix, B: Matrix): Matrix {
  const { rows: n } = shapeOf(A);
  const { cols: m } = shapeOf(B);
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => cellProducts(A, B, i, j).reduce((a, b) => a + b, 0)),
  );
}

export type StepMode = "term" | "cell";

export type MatmulStep =
  | { kind: "intro" }
  | {
      kind: "work";
      i: number;
      j: number;
      /** index of the last product term included so far (0-based) */
      p: number;
      /** linear index of the current cell (row-major) */
      cell: number;
      products: number[];
      partial: number[];
    }
  | { kind: "done" };

/** Number of stepper positions: intro + work units + final summary. */
export function totalSteps(A: Matrix, B: Matrix, mode: StepMode): number {
  if (!canMultiply(A, B)) return 1;
  const n = shapeOf(A).rows;
  const k = shapeOf(A).cols;
  const m = shapeOf(B).cols;
  const units = mode === "term" ? n * m * k : n * m;
  return units + 2;
}

/** Decode a stepper index into what should be displayed. */
export function decodeStep(A: Matrix, B: Matrix, mode: StepMode, step: number): MatmulStep {
  if (!canMultiply(A, B) || step <= 0) return { kind: "intro" };
  const total = totalSteps(A, B, mode);
  if (step >= total - 1) return { kind: "done" };
  const k = shapeOf(A).cols;
  const m = shapeOf(B).cols;
  const u = step - 1;
  const cell = mode === "term" ? Math.floor(u / k) : u;
  const p = mode === "term" ? u % k : k - 1;
  const i = Math.floor(cell / m);
  const j = cell % m;
  const products = cellProducts(A, B, i, j);
  return { kind: "work", i, j, p, cell, products, partial: runningSums(products) };
}

/** Small-integer random matrix in [-lo..hi] so the arithmetic stays readable. */
export function randomIntMatrix(rows: number, cols: number, seed: number, lo = -3, hi = 3): Matrix {
  const rand = mulberry32(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => lo + Math.floor(rand() * (hi - lo + 1))),
  );
}

/** Resize a matrix, keeping existing entries and filling new ones from `fill`. */
export function resize(M: Matrix, rows: number, cols: number, fill: (r: number, c: number) => number): Matrix {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => (r < M.length && c < (M[r]?.length ?? 0) ? M[r][c] : fill(r, c))),
  );
}
