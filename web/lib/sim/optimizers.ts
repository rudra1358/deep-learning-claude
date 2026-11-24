/**
 * 2-D loss surfaces and first-order optimizers (pure math, no React).
 * Update rules follow the PyTorch conventions used in the lesson:
 *   Momentum   v ← βv + g;               θ ← θ − η v
 *   Nesterov   v ← βv + g;               θ ← θ − η (g + βv)
 *   AdaGrad    G ← G + g²;               θ ← θ − η g / (√G + ε)
 *   RMSProp    s ← ρs + (1−ρ) g²;        θ ← θ − η g / (√s + ε)
 *   Adam       m ← β1 m + (1−β1) g;  v ← β2 v + (1−β2) g²;
 *              θ ← θ − η m̂ / (√v̂ + ε),  m̂ = m/(1−β1ᵗ), v̂ = v/(1−β2ᵗ)
 */

export type Vec2 = [number, number];

export interface Surface {
  id: string;
  label: string;
  formula: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => Vec2;
  xRange: [number, number];
  yRange: [number, number];
  start: Vec2;
  /** default learning rate for SGD / Momentum / Nesterov */
  lrSgd: number;
  /** default learning rate for AdaGrad / RMSProp / Adam */
  lrAdaptive: number;
  /** colour the heatmap by log(f − fmin + 1) instead of f */
  logScale: boolean;
  /** known minima, drawn as small crosses */
  minima: Vec2[];
  note: string;
}

/* ---------- sum-of-Gaussians surface ---------- */
const WELLS: { c: Vec2; a: number; s: number }[] = [
  { c: [-1.6, 1.1], a: 1.0, s: 0.65 },
  { c: [1.4, -0.6], a: 1.6, s: 0.75 },
  { c: [0.6, 1.9], a: 0.8, s: 0.5 },
  { c: [-1.2, -1.6], a: 0.9, s: 0.55 },
];
const BOWL = 0.05;

function wellsF(x: number, y: number) {
  let s = BOWL * (x * x + y * y);
  for (const w of WELLS) {
    const dx = x - w.c[0], dy = y - w.c[1];
    s -= w.a * Math.exp(-(dx * dx + dy * dy) / (2 * w.s * w.s));
  }
  return s;
}
function wellsG(x: number, y: number): Vec2 {
  let gx = 2 * BOWL * x, gy = 2 * BOWL * y;
  for (const w of WELLS) {
    const dx = x - w.c[0], dy = y - w.c[1];
    const e = w.a * Math.exp(-(dx * dx + dy * dy) / (2 * w.s * w.s));
    gx += (e * dx) / (w.s * w.s);
    gy += (e * dy) / (w.s * w.s);
  }
  return [gx, gy];
}

export const SURFACES: Surface[] = [
  {
    id: "valley",
    label: "Elongated valley",
    formula: "f = 0.1x² + 2y²",
    f: (x, y) => 0.1 * x * x + 2 * y * y,
    grad: (x, y) => [0.2 * x, 4 * y],
    xRange: [-5, 5],
    yRange: [-2.5, 2.5],
    start: [-4.5, 1.8],
    lrSgd: 0.2,
    lrAdaptive: 0.1,
    logScale: false,
    minima: [[0, 0]],
    note: "Curvatures 0.2 and 4 (condition number 20). Plain SGD must keep η < 2/4 = 0.5 for the steep y-direction, which makes the shallow x-direction crawl.",
  },
  {
    id: "rosenbrock",
    label: "Rosenbrock",
    formula: "f = (1 − x)² + 100(y − x²)²",
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    xRange: [-2, 2],
    yRange: [-1, 3],
    start: [-1.5, 2.5],
    lrSgd: 0.001,
    lrAdaptive: 0.05,
    logScale: true,
    minima: [[1, 1]],
    note: "A curved, narrow banana-shaped valley with its minimum at (1, 1). Reaching the valley floor is easy; travelling along it is slow.",
  },
  {
    id: "saddle",
    label: "Saddle",
    formula: "f = x² + y⁴/4 − y²/2",
    f: (x, y) => x * x + (y ** 4) / 4 - (y * y) / 2,
    grad: (x, y) => [2 * x, y ** 3 - y],
    xRange: [-2, 2],
    yRange: [-2, 2],
    start: [-1.8, 0.01],
    lrSgd: 0.05,
    lrAdaptive: 0.05,
    logScale: false,
    minima: [
      [0, 1],
      [0, -1],
    ],
    note: "A saddle point at (0, 0): a minimum along x, a maximum along y. Starting almost on the ridge y = 0, the escape direction has a tiny gradient.",
  },
  {
    id: "wells",
    label: "Multi-modal",
    formula: "f = 0.05(x² + y²) − Σ aᵢ exp(−‖p − cᵢ‖² / 2sᵢ²)",
    f: wellsF,
    grad: wellsG,
    xRange: [-3, 3],
    yRange: [-3, 3],
    start: [-2.6, 2.6],
    lrSgd: 0.3,
    lrAdaptive: 0.05,
    logScale: false,
    minima: WELLS.map((w) => w.c),
    note: "Four Gaussian wells of different depths on a shallow bowl. Which well an optimizer ends in depends on its start point and its momentum.",
  },
];

export type OptName = "sgd" | "momentum" | "nesterov" | "adagrad" | "rmsprop" | "adam";

export const OPTIMIZERS: { id: OptName; label: string; adaptive: boolean }[] = [
  { id: "sgd", label: "SGD", adaptive: false },
  { id: "momentum", label: "Momentum", adaptive: false },
  { id: "nesterov", label: "Nesterov", adaptive: false },
  { id: "adagrad", label: "AdaGrad", adaptive: true },
  { id: "rmsprop", label: "RMSProp", adaptive: true },
  { id: "adam", label: "Adam", adaptive: true },
];

export const HYPER = { beta: 0.9, rho: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

interface State {
  t: number;
  v: Vec2; // velocity / first moment
  s: Vec2; // squared-gradient accumulator / second moment
}

/** One optimizer update. Mutates and returns the new position. */
export function optStep(name: OptName, p: Vec2, g: Vec2, lr: number, st: State): Vec2 {
  const { beta, rho, beta1, beta2, eps } = HYPER;
  st.t += 1;
  const out: Vec2 = [p[0], p[1]];
  for (let i = 0; i < 2; i++) {
    const gi = g[i];
    switch (name) {
      case "sgd":
        out[i] = p[i] - lr * gi;
        break;
      case "momentum":
        st.v[i] = beta * st.v[i] + gi;
        out[i] = p[i] - lr * st.v[i];
        break;
      case "nesterov":
        st.v[i] = beta * st.v[i] + gi;
        out[i] = p[i] - lr * (gi + beta * st.v[i]);
        break;
      case "adagrad":
        st.s[i] += gi * gi;
        out[i] = p[i] - (lr * gi) / (Math.sqrt(st.s[i]) + eps);
        break;
      case "rmsprop":
        st.s[i] = rho * st.s[i] + (1 - rho) * gi * gi;
        out[i] = p[i] - (lr * gi) / (Math.sqrt(st.s[i]) + eps);
        break;
      case "adam": {
        st.v[i] = beta1 * st.v[i] + (1 - beta1) * gi;
        st.s[i] = beta2 * st.s[i] + (1 - beta2) * gi * gi;
        const mh = st.v[i] / (1 - beta1 ** st.t);
        const vh = st.s[i] / (1 - beta2 ** st.t);
        out[i] = p[i] - (lr * mh) / (Math.sqrt(vh) + eps);
        break;
      }
    }
  }
  return out;
}

export interface PathPoint {
  x: number;
  y: number;
  loss: number;
}

export interface OptRun {
  path: PathPoint[];
  diverged: boolean;
}

/** Run n steps; the path has n+1 points unless the run diverges early. */
export function runOptimizer(surface: Surface, name: OptName, start: Vec2, lr: number, n: number): OptRun {
  const st: State = { t: 0, v: [0, 0], s: [0, 0] };
  let p: Vec2 = [start[0], start[1]];
  const path: PathPoint[] = [{ x: p[0], y: p[1], loss: surface.f(p[0], p[1]) }];
  for (let k = 0; k < n; k++) {
    p = optStep(name, p, surface.grad(p[0], p[1]), lr, st);
    const loss = surface.f(p[0], p[1]);
    if (!Number.isFinite(loss) || Math.abs(p[0]) > 1e6 || Math.abs(p[1]) > 1e6) return { path, diverged: true };
    path.push({ x: p[0], y: p[1], loss });
  }
  return { path, diverged: false };
}

/**
 * Sample f on a w×h grid covering the surface's ranges (row 0 = top = yMax).
 * Returns values normalised to [0, 1] (optionally on a log scale).
 */
export function sampleGrid(surface: Surface, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  const [x0, x1] = surface.xRange;
  const [y0, y1] = surface.yRange;
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j < h; j++) {
    const y = y1 - ((j + 0.5) / h) * (y1 - y0);
    for (let i = 0; i < w; i++) {
      const x = x0 + ((i + 0.5) / w) * (x1 - x0);
      const v = surface.f(x, y);
      out[j * w + i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  for (let k = 0; k < out.length; k++) {
    const v = out[k] - lo;
    out[k] = surface.logScale ? Math.log1p(v) / Math.log1p(hi - lo) : v / (hi - lo || 1);
  }
  return out;
}
