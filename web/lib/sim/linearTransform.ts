/**
 * Pure 2×2 linear-algebra helpers for the linear-transformation simulator (no React).
 *
 * Matrices are stored row-major as { a, b, c, d } meaning
 *      | a  b |
 *      | c  d |
 * so that M·(x, y) = (a x + b y, c x + d y).
 *
 * Everything the simulator displays (determinant, eigen-pairs, SVD factors,
 * the interpolated matrices drawn on screen) comes from these functions.
 */

export interface Mat2 {
  a: number;
  b: number;
  c: number;
  d: number;
}

export type Vec2 = [number, number];

export const I2: Mat2 = { a: 1, b: 0, c: 0, d: 1 };

const EPS = 1e-9;

export function apply(M: Mat2, v: Vec2): Vec2 {
  return [M.a * v[0] + M.b * v[1], M.c * v[0] + M.d * v[1]];
}

export function mul(A: Mat2, B: Mat2): Mat2 {
  return {
    a: A.a * B.a + A.b * B.c,
    b: A.a * B.b + A.b * B.d,
    c: A.c * B.a + A.d * B.c,
    d: A.c * B.b + A.d * B.d,
  };
}

export function transpose(M: Mat2): Mat2 {
  return { a: M.a, b: M.c, c: M.b, d: M.d };
}

export function det(M: Mat2): number {
  return M.a * M.d - M.b * M.c;
}

export function trace(M: Mat2): number {
  return M.a + M.d;
}

export function lerp(A: Mat2, B: Mat2, t: number): Mat2 {
  return {
    a: A.a + (B.a - A.a) * t,
    b: A.b + (B.b - A.b) * t,
    c: A.c + (B.c - A.c) * t,
    d: A.d + (B.d - A.d) * t,
  };
}

/** Counter-clockwise rotation by angle θ (radians). */
export function rotation(theta: number): Mat2 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return { a: c, b: -s, c: s, d: c };
}

export function diag(x: number, y: number): Mat2 {
  return { a: x, b: 0, c: 0, d: y };
}

export function norm(v: Vec2): number {
  return Math.hypot(v[0], v[1]);
}

function normalize(v: Vec2): Vec2 {
  const n = norm(v);
  return n < EPS ? [1, 0] : [v[0] / n, v[1] / n];
}

/* ------------------------------------------------------------------ */
/* Eigen-decomposition                                                 */
/* ------------------------------------------------------------------ */

export type Eigen =
  | { kind: "real"; values: [number, number]; vectors: [Vec2, Vec2]; repeated: false }
  /** λ repeated with a single eigen-direction (defective, e.g. a shear). */
  | { kind: "defective"; value: number; vector: Vec2 }
  /** M = λI: every non-zero vector is an eigenvector. */
  | { kind: "scalar"; value: number }
  | { kind: "complex"; re: number; im: number };

/**
 * Eigenvalues are the roots of the characteristic polynomial
 *   λ² − tr(M) λ + det(M) = 0  ⇒  λ = tr/2 ± √(tr²/4 − det).
 * An eigenvector solves (M − λI) v = 0; for a 2×2 matrix one non-zero row of
 * M − λI determines it.
 */
export function eigen(M: Mat2): Eigen {
  const tr = trace(M);
  const dt = det(M);
  const disc = (tr * tr) / 4 - dt;
  const scale = Math.max(1, Math.abs(M.a), Math.abs(M.b), Math.abs(M.c), Math.abs(M.d));
  const tol = 1e-9 * scale * scale;

  if (disc < -tol) return { kind: "complex", re: tr / 2, im: Math.sqrt(-disc) };

  if (Math.abs(disc) <= tol) {
    const lam = tr / 2;
    if (Math.abs(M.b) < 1e-9 * scale && Math.abs(M.c) < 1e-9 * scale && Math.abs(M.a - M.d) < 1e-9 * scale) {
      return { kind: "scalar", value: lam };
    }
    return { kind: "defective", value: lam, vector: eigenvectorFor(M, lam) };
  }

  const r = Math.sqrt(disc);
  const l1 = tr / 2 + r;
  const l2 = tr / 2 - r;
  return { kind: "real", values: [l1, l2], vectors: [eigenvectorFor(M, l1), eigenvectorFor(M, l2)], repeated: false };
}

/** Unit vector spanning the null space of M − λI (choose the better-conditioned row). */
export function eigenvectorFor(M: Mat2, lam: number): Vec2 {
  // Row 1 of (M − λI) is (a−λ, b): a vector orthogonal to it is (b, λ−a).
  const v1: Vec2 = [M.b, lam - M.a];
  // Row 2 is (c, d−λ): orthogonal vector (λ−d, c).
  const v2: Vec2 = [lam - M.d, M.c];
  const v = norm(v1) >= norm(v2) ? v1 : v2;
  if (norm(v) < EPS) return [1, 0];
  const u = normalize(v);
  // Sign convention: first non-zero component positive (purely cosmetic).
  return u[0] < -EPS || (Math.abs(u[0]) <= EPS && u[1] < 0) ? [-u[0], -u[1]] : u;
}

/* ------------------------------------------------------------------ */
/* Singular value decomposition                                        */
/* ------------------------------------------------------------------ */

export interface SVD2 {
  /** Left singular vectors as columns. */
  U: Mat2;
  /** σ1 ≥ σ2 ≥ 0 */
  sigma: [number, number];
  /** Right singular vectors as columns (always a proper rotation here). */
  V: Mat2;
  /** Angle θ_V with V = R(θ_V); Vᵀ = R(−θ_V). */
  thetaV: number;
  /** Angle θ_U with U = R(θ_U)·F, F = diag(1, ±1). */
  thetaU: number;
  /** +1 if U is a rotation, −1 if U contains a reflection (det M < 0). */
  uSign: 1 | -1;
}

/**
 * SVD of a 2×2 matrix via the symmetric eigenproblem MᵀM = V Σ² Vᵀ, then
 * uᵢ = M vᵢ / σᵢ. Signs are fixed so that V is a rotation by an angle in
 * (−90°, 90°], which keeps the animation's first rotation as small as possible.
 */
export function svd(M: Mat2): SVD2 {
  const G = mul(transpose(M), M); // symmetric PSD: [[p, q], [q, r]]
  const p = G.a;
  const q = G.b;
  const r = G.d;
  const mean = (p + r) / 2;
  const rad = Math.sqrt(((p - r) / 2) ** 2 + q * q);
  const l1 = Math.max(0, mean + rad);
  const l2 = Math.max(0, mean - rad);
  const s1 = Math.sqrt(l1);
  const s2 = Math.sqrt(l2);

  // Angle of the dominant eigenvector of a symmetric 2×2 matrix.
  let theta = 0.5 * Math.atan2(2 * q, p - r);
  // Map into (−π/2, π/2]; atan2/2 already lies in (−π/2, π/2].
  if (theta <= -Math.PI / 2) theta += Math.PI;
  if (theta > Math.PI / 2) theta -= Math.PI;

  const v1: Vec2 = [Math.cos(theta), Math.sin(theta)];
  const v2: Vec2 = [-Math.sin(theta), Math.cos(theta)];
  const V: Mat2 = { a: v1[0], b: v2[0], c: v1[1], d: v2[1] };

  // Left singular vectors.
  let u1: Vec2;
  if (s1 > EPS) {
    const m = apply(M, v1);
    u1 = [m[0] / s1, m[1] / s1];
  } else {
    u1 = [1, 0]; // M = 0: any orthonormal U works
  }
  let u2: Vec2;
  if (s2 > 1e-9 * Math.max(1, s1)) {
    const m = apply(M, v2);
    u2 = [m[0] / s2, m[1] / s2];
  } else {
    // Rank ≤ 1: complete u1 to an orthonormal basis (choose a rotation).
    u2 = [-u1[1], u1[0]];
  }
  const U: Mat2 = { a: u1[0], b: u2[0], c: u1[1], d: u2[1] };
  const uSign: 1 | -1 = det(U) < 0 ? -1 : 1;
  const thetaU = Math.atan2(u1[1], u1[0]);
  return { U, sigma: [s1, s2], V, thetaV: theta, thetaU, uSign };
}

/**
 * The matrix shown at SVD-animation position `pos` ∈ [0, 3]:
 *   0 → 1 : rotate by Vᵀ           (R(−θ_V · t))
 *   1 → 2 : scale the axes by Σ    (diag(1 + t(σ1 − 1), 1 + t(σ2 − 1)))
 *   2 → 3 : rotate (and possibly reflect) by U
 * For a reflecting U = R(θ_U)·diag(1, −1) the reflection is animated as the
 * second axis passing smoothly through zero: R(θ_U t)·diag(1, 1 − 2t).
 */
export function svdStage(s: SVD2, pos: number): Mat2 {
  const t1 = clamp01(pos);
  const t2 = clamp01(pos - 1);
  const t3 = clamp01(pos - 2);
  const Vt = rotation(-s.thetaV * t1);
  const S = diag(1 + t2 * (s.sigma[0] - 1), 1 + t2 * (s.sigma[1] - 1));
  const Ut = mul(rotation(s.thetaU * t3), diag(1, s.uSign === 1 ? 1 : 1 - 2 * t3));
  return mul(Ut, mul(S, Vt));
}

/** The matrix shown at identity→A interpolation `t` ∈ [0, 1]: (1 − t) I + t A. */
export function blend(A: Mat2, t: number): Mat2 {
  return lerp(I2, A, clamp01(t));
}

export function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

/** Smooth ease-in-out used for the tween between discrete stages. */
export function ease(t: number) {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

/** Half-width of the plotted window: enough to show the image of the unit square and circle. */
export function viewRadius(A: Mat2): number {
  const s = svd(A).sigma[0];
  const corners: Vec2[] = [apply(A, [1, 1]), apply(A, [1, -1])];
  const reach = Math.max(s, ...corners.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
  return Math.min(8, Math.max(3, Math.ceil(reach * 1.15 + 0.3)));
}

export interface Preset {
  key: string;
  label: string;
  M: Mat2;
  note: string;
}

export const PRESETS: Preset[] = [
  { key: "general", label: "General (λ = 2.5, 1)", M: { a: 2, b: 1, c: 0.5, d: 1.5 }, note: "Two real eigen-directions (lines that only stretch), not at right angles." },
  { key: "symmetric", label: "Symmetric", M: { a: 2, b: 1, c: 1, d: 2 }, note: "Symmetric: real eigenvalues, eigenvectors at right angles, SVD = eigen-decomposition." },
  { key: "rotation", label: "Rotation 45°", M: { a: Math.SQRT1_2, b: -Math.SQRT1_2, c: Math.SQRT1_2, d: Math.SQRT1_2 }, note: "Rotation: every direction turns, so no real eigenvectors (eigenvalues e^{±iθ}); det = 1." },
  { key: "scaling", label: "Scaling", M: { a: 2, b: 0, c: 0, d: 0.5 }, note: "Stretch along the axes: the axes are the eigenvectors; det = 1." },
  { key: "shear", label: "Shear", M: { a: 1, b: 1, c: 0, d: 1 }, note: "Shear: λ = 1 twice, but only one eigen-direction (called defective). Area stays the same." },
  { key: "reflection", label: "Reflection", M: { a: 1, b: 0, c: 0, d: -1 }, note: "Mirror across the x-axis: det = −1 means the plane is flipped over." },
  { key: "projection", label: "Projection (singular)", M: { a: 1, b: 0, c: 0, d: 0 }, note: "Squash onto the x-axis: det = 0, rank 1. The whole plane collapses to a line." },
  { key: "lesson-eig", label: "Lesson: [[4,1],[2,3]]", M: { a: 4, b: 1, c: 2, d: 3 }, note: "The lesson's eigen example: λ = 5 along (1,1), λ = 2 along (1,−2)." },
  { key: "lesson-svd", label: "Lesson: [[3,0],[4,5]]", M: { a: 3, b: 0, c: 4, d: 5 }, note: "The lesson's SVD example: σ = √45 ≈ 6.708 and √5 ≈ 2.236." },
];
