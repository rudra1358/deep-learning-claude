/**
 * Pure math for the L1 / L2 / elastic-net geometry simulator (no React).
 *
 * Data loss (a quadratic bowl in the (w1, w2) plane):
 *   L(w) = ½ (w − c)ᵀ A (w − c),   A = R(θ) diag(a1, a2) R(θ)ᵀ  (symmetric positive definite)
 * c is the unregularised optimum; θ rotates the contour ellipses; a1/a2 sets their elongation.
 *
 * Penalised problems (λ ≥ 0):
 *   L2:          min L(w) + (λ/2)‖w‖₂²          → closed form  w = (A + λI)⁻¹ A c
 *   L1:          min L(w) + λ‖w‖₁                → exact 2-D active-set solution (verified by ISTA)
 *   elastic net: min L(w) + λ(α‖w‖₁ + (1−α)/2 ‖w‖₂²)
 *                = L1 problem with A' = A + λ(1−α)I, c' = A'⁻¹ A c, λ' = λα   (completing the square)
 *
 * Each penalised solution w*(λ) is also the solution of the constrained problem
 *   min L(w) s.t. ‖w‖ ≤ t  with t = ‖w*(λ)‖ — the contour of L through w* touches the norm ball there.
 */

export type Vec2 = [number, number];
export type Mat2 = [[number, number], [number, number]];

export interface Problem {
  c: Vec2;
  theta: number; // radians
  a1: number;
  a2: number;
}

export function matA(pr: Problem): Mat2 {
  const co = Math.cos(pr.theta);
  const si = Math.sin(pr.theta);
  // R diag(a1,a2) Rᵀ with R = [[co, -si], [si, co]]
  const a11 = pr.a1 * co * co + pr.a2 * si * si;
  const a22 = pr.a1 * si * si + pr.a2 * co * co;
  const a12 = (pr.a1 - pr.a2) * co * si;
  return [
    [a11, a12],
    [a12, a22],
  ];
}

const mv = (A: Mat2, v: Vec2): Vec2 => [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]];

function solve(A: Mat2, b: Vec2): Vec2 {
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  return [(A[1][1] * b[0] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - A[1][0] * b[0]) / det];
}

const addI = (A: Mat2, mu: number): Mat2 => [
  [A[0][0] + mu, A[0][1]],
  [A[1][0], A[1][1] + mu],
];

export function dataLoss(A: Mat2, c: Vec2, w: Vec2): number {
  const d: Vec2 = [w[0] - c[0], w[1] - c[1]];
  const Ad = mv(A, d);
  return 0.5 * (d[0] * Ad[0] + d[1] * Ad[1]);
}

export const l1 = (w: Vec2) => Math.abs(w[0]) + Math.abs(w[1]);
export const l2 = (w: Vec2) => Math.hypot(w[0], w[1]);

/** Ridge: (A + λI)⁻¹ A c. */
export function solveL2(A: Mat2, c: Vec2, lam: number): Vec2 {
  return solve(addI(A, lam), mv(A, c));
}

/**
 * Exact minimiser of ½(w−c)ᵀA(w−c) + λ‖w‖₁ in 2-D.
 * Enumerate the faces of the orthant structure: both coordinates zero, exactly one zero (with a sign),
 * or both non-zero (with a sign pattern s). On each face the objective is a smooth strictly convex quadratic,
 * whose stationary point is valid only if its signs match the assumed pattern. The global minimiser is the
 * valid candidate with the smallest objective.
 */
export function solveL1(A: Mat2, c: Vec2, lam: number): Vec2 {
  const F = (w: Vec2) => dataLoss(A, c, w) + lam * l1(w);
  const cands: Vec2[] = [[0, 0]];
  const sgn = [-1, 1];
  // w2 = 0, w1 ≠ 0:  a11(w1 − c1) + a12(0 − c2) + λ s1 = 0
  for (const s1 of sgn) {
    const w1 = c[0] + (A[0][1] * c[1] - lam * s1) / A[0][0];
    if (w1 * s1 > 0) cands.push([w1, 0]);
  }
  // w1 = 0, w2 ≠ 0
  for (const s2 of sgn) {
    const w2 = c[1] + (A[1][0] * c[0] - lam * s2) / A[1][1];
    if (w2 * s2 > 0) cands.push([0, w2]);
  }
  // both non-zero: A(w − c) + λ s = 0 → w = c − λ A⁻¹ s
  for (const s1 of sgn)
    for (const s2 of sgn) {
      const d = solve(A, [lam * s1, lam * s2]);
      const w: Vec2 = [c[0] - d[0], c[1] - d[1]];
      if (w[0] * s1 > 0 && w[1] * s2 > 0) cands.push(w);
    }
  let best = cands[0];
  let bf = F(best);
  for (const w of cands) {
    const f = F(w);
    if (f < bf - 1e-15) {
      best = w;
      bf = f;
    }
  }
  return best;
}

export const softThreshold = (v: number, t: number) => Math.sign(v) * Math.max(Math.abs(v) - t, 0);

/** ISTA (proximal gradient) for the L1 problem; returns the iterate and number of iterations used. */
export function ista(A: Mat2, c: Vec2, lam: number, tol = 1e-12, maxIter = 20000): { w: Vec2; iters: number } {
  const tr = A[0][0] + A[1][1];
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const Lmax = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det)); // largest eigenvalue = Lipschitz const.
  const eta = 1 / Lmax;
  let w: Vec2 = [0, 0];
  for (let it = 1; it <= maxIter; it++) {
    const g = mv(A, [w[0] - c[0], w[1] - c[1]]);
    const nw: Vec2 = [softThreshold(w[0] - eta * g[0], eta * lam), softThreshold(w[1] - eta * g[1], eta * lam)];
    const ch = Math.abs(nw[0] - w[0]) + Math.abs(nw[1] - w[1]);
    w = nw;
    if (ch < tol) return { w, iters: it };
  }
  return { w, iters: maxIter };
}

/** Elastic net via reduction to an L1 problem. */
export function solveEN(A: Mat2, c: Vec2, lam: number, alpha: number): Vec2 {
  const mu = lam * (1 - alpha);
  const A2 = addI(A, mu);
  const c2 = solve(A2, mv(A, c));
  return solveL1(A2, c2, lam * alpha);
}

/** Smallest λ at which the L1 solution is exactly 0: KKT at w = 0 needs |∇L(0)|∞ = |A c|∞ ≤ λ. */
export function lambdaMaxL1(A: Mat2, c: Vec2): number {
  const g = mv(A, c);
  return Math.max(Math.abs(g[0]), Math.abs(g[1]));
}

/** λ grid: index 0 is λ = 0, the rest log-spaced from 1% to 120% of λ_max. */
export function lambdaGrid(lmax: number, n: number): number[] {
  const out = [0];
  const lo = Math.log(0.01 * lmax);
  const hi = Math.log(1.2 * lmax);
  for (let i = 1; i < n; i++) out.push(Math.exp(lo + ((hi - lo) * (i - 1)) / (n - 2)));
  return out;
}

/** Points of the ellipse {w : L(w) = level}. */
export function ellipse(pr: Problem, level: number, n = 96): Vec2[] {
  const r1 = Math.sqrt((2 * level) / pr.a1);
  const r2 = Math.sqrt((2 * level) / pr.a2);
  const co = Math.cos(pr.theta);
  const si = Math.sin(pr.theta);
  const pts: Vec2[] = [];
  for (let i = 0; i <= n; i++) {
    const t = (2 * Math.PI * i) / n;
    const u = r1 * Math.cos(t);
    const v = r2 * Math.sin(t);
    pts.push([pr.c[0] + co * u - si * v, pr.c[1] + si * u + co * v]);
  }
  return pts;
}

export interface Preset {
  id: string;
  label: string;
  pr: Problem;
}

export const PRESETS: Preset[] = [
  { id: "axis", label: "Axis-aligned", pr: { c: [2, 0.5], theta: 0, a1: 1, a2: 2.5 } },
  { id: "tilted", label: "Tilted", pr: { c: [1.8, 1.2], theta: (35 * Math.PI) / 180, a1: 1, a2: 5 } },
  { id: "correlated", label: "Correlated features", pr: { c: [2.0, 1.0], theta: (-45 * Math.PI) / 180, a1: 1, a2: 16 } },
  { id: "round", label: "Round bowl", pr: { c: [2.2, 1.0], theta: 0, a1: 1, a2: 1 } },
];

/** Exact-zero test used for read-outs. */
export const isZero = (v: number) => v === 0 || Math.abs(v) < 1e-12;
