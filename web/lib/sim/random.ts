/** Deterministic PRNG (mulberry32) so simulations are reproducible across reloads. */
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

/** Standard normal sample via Box–Muller. */
export function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Simple string hash (for deterministic per-token embeddings). */
export function hashString(s: string) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export type Matrix = number[][];

export function matmul(A: Matrix, B: Matrix): Matrix {
  const n = A.length, k = B.length, m = B[0].length;
  const C: Matrix = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let s = 0;
    for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
    C[i][j] = s;
  }
  return C;
}

export function transpose(A: Matrix): Matrix {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

export function softmax(v: number[]): number[] {
  const finite = v.filter(Number.isFinite);
  const m = finite.length ? Math.max(...finite) : 0;
  const e = v.map((x) => (Number.isFinite(x) ? Math.exp(x - m) : 0));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((x) => x / s);
}

export function randomMatrix(rows: number, cols: number, rand: () => number, scale = 1): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => gaussian(rand) * scale));
}
