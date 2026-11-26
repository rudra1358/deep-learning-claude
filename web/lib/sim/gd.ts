/**
 * 1-D gradient descent: test functions and the exact update sequence
 * x_{k+1} = x_k − η f'(x_k). Pure functions, no React.
 */

export interface Fn1D {
  id: string;
  label: string;
  /** Formula shown to the learner (plain text). */
  formula: string;
  f: (x: number) => number;
  df: (x: number) => number;
  d2f: (x: number) => number;
  /** Visible x-range of the plot. */
  xRange: [number, number];
  /** Visible y-range of the plot. */
  yRange: [number, number];
  x0: number;
  eta: number;
  etaMax: number;
  /** Location(s) of the minima, for the narration. */
  minima: number[];
  note: string;
}

const gauss = (x: number) => Math.exp(-x * x);

export const FUNCTIONS: Fn1D[] = [
  {
    id: "quadratic",
    label: "Quadratic",
    formula: "f(x) = x²",
    f: (x) => x * x,
    df: (x) => 2 * x,
    d2f: () => 2,
    xRange: [-3, 3],
    yRange: [-0.5, 9],
    x0: 2.5,
    eta: 0.2,
    etaMax: 1.2,
    minima: [0],
    note: "Curvature (bend) f''=2 everywhere, so each step is x ← (1 − 2η)x. It settles for 0 < η < 1, bounces forever at η = 1, blows up above.",
  },
  {
    id: "quartic",
    label: "Two minima",
    formula: "f(x) = x⁴ − 3x² + x",
    f: (x) => x ** 4 - 3 * x * x + x,
    df: (x) => 4 * x ** 3 - 6 * x + 1,
    d2f: (x) => 12 * x * x - 6,
    xRange: [-2.2, 2.2],
    yRange: [-4.5, 5],
    x0: 1.9,
    eta: 0.03,
    etaMax: 0.25,
    minima: [-1.3008, 1.1309],
    note: "Global (lowest) minimum near x = −1.30, local minimum near x = 1.13. Your start, and how far η lets you jump, decide which one you reach.",
  },
  {
    id: "plateau",
    label: "Plateau + cliff",
    formula: "f(x) = 1 − e^(−x²) + 3·max(0, x)²",
    f: (x) => 1 - gauss(x) + (x > 0 ? 3 * x * x : 0),
    df: (x) => 2 * x * gauss(x) + (x > 0 ? 6 * x : 0),
    d2f: (x) => (2 - 4 * x * x) * gauss(x) + (x > 0 ? 6 : 0),
    xRange: [-3.5, 2],
    yRange: [-0.3, 5],
    x0: -2.6,
    eta: 0.3,
    etaMax: 1.0,
    minima: [0],
    note: "Left: a nearly flat plateau where f'(x) ≈ 0, so steps are tiny. Right: a steep wall (curvature 8 near 0). A large η jumps too far, back onto the plateau.",
  },
  {
    id: "flatbottom",
    label: "Flat bottom (x⁴)",
    formula: "f(x) = x⁴ / 4",
    f: (x) => x ** 4 / 4,
    df: (x) => x ** 3,
    d2f: (x) => 3 * x * x,
    xRange: [-2.5, 2.5],
    yRange: [-0.4, 9],
    x0: 2.0,
    eta: 0.1,
    etaMax: 0.6,
    minima: [0],
    note: "Curvature 3x² is zero at the minimum. Far away the walls are steep (a large η blows up). Near the bottom the gradient x³ shrinks so fast that progress crawls (sub-linear).",
  },
];

export interface GDStep {
  k: number;
  x: number;
  fx: number;
  g: number;
  /** η · f'(x) — the actual displacement (with a minus sign) */
  delta: number;
  xNext: number;
  fNext: number;
}

export interface Trajectory {
  steps: GDStep[];
  diverged: boolean;
}

/**
 * |x| beyond which a run is declared diverged: 50× the plotted half-width.
 * Every preset grows at least quadratically outside its plot range, so no run
 * that gets this far can come back.
 */
export function divergeBound(fn: Fn1D) {
  return 50 * Math.max(Math.abs(fn.xRange[0]), Math.abs(fn.xRange[1]));
}

/** Run n updates of plain gradient descent from x0. Stops early on divergence. */
export function runGD(fn: Fn1D, x0: number, eta: number, n: number): Trajectory {
  const steps: GDStep[] = [];
  const bound = divergeBound(fn);
  let x = x0;
  for (let k = 0; k < n; k++) {
    const fx = fn.f(x);
    const g = fn.df(x);
    const delta = eta * g;
    const xNext = x - delta;
    const fNext = fn.f(xNext);
    steps.push({ k, x, fx, g, delta, xNext, fNext });
    if (!Number.isFinite(xNext) || Math.abs(xNext) > bound || !Number.isFinite(fNext)) {
      return { steps, diverged: true };
    }
    x = xNext;
  }
  return { steps, diverged: false };
}

/** Largest stable learning rate predicted by the local quadratic model: η < 2 / f''(x). */
export function stableEta(fn: Fn1D, x: number) {
  const c = fn.d2f(x);
  return c > 0 ? 2 / c : Infinity;
}
