/**
 * Activation functions and their derivatives (pure math, no React).
 * Reusable by any simulator: import { ACTIVATIONS, sigmoid, gelu, ... } from "@/lib/sim/activations".
 *
 * Derivative conventions at non-differentiable points follow PyTorch:
 *   ReLU'(0) = 0,  LeakyReLU'(0) = α,  step'(x) = 0 everywhere (undefined at 0).
 */

export interface ActParams {
  /** negative slope for Leaky ReLU / PReLU, and α for ELU */
  alpha: number;
  /** β for Swish: x·σ(βx) (β = 1 gives SiLU) */
  beta: number;
}

export const DEFAULT_PARAMS: ActParams = { alpha: 0.1, beta: 1 };

/** SELU constants from Klambauer et al. (2017). */
export const SELU_LAMBDA = 1.0507009873554805;
export const SELU_ALPHA = 1.6732632423543772;

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const GELU_C = 0.044715;
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

/**
 * Error function. Complementary-error-function Chebyshev fit (Numerical Recipes, `erfcc`),
 * fractional error below 1.2e-7 everywhere — more than enough for plotting and readouts.
 */
export function erf(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? 1 - r : r - 1;
}

/** Standard normal CDF Φ(x) and density φ(x). */
export const normCdf = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
export const normPdf = (x: number) => INV_SQRT_2PI * Math.exp(-0.5 * x * x);

/** Numerically stable logistic sigmoid. */
export function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/** Numerically stable softplus ln(1 + eˣ). */
export function softplus(x: number): number {
  return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
}

// ---- functions -----------------------------------------------------------
export const step = (x: number) => (x >= 0 ? 1 : 0);
export const linear = (x: number) => x;
export const tanh = (x: number) => Math.tanh(x);
export const relu = (x: number) => (x > 0 ? x : 0);
export const leakyRelu = (x: number, alpha = 0.01) => (x > 0 ? x : alpha * x);
/** PReLU has the same formula as Leaky ReLU; the difference is that α is learned. */
export const prelu = (x: number, alpha = 0.25) => (x > 0 ? x : alpha * x);
export const elu = (x: number, alpha = 1) => (x > 0 ? x : alpha * Math.expm1(x));
export const selu = (x: number) => SELU_LAMBDA * (x > 0 ? x : SELU_ALPHA * Math.expm1(x));
/** Exact GELU: x·Φ(x). */
export const gelu = (x: number) => x * normCdf(x);
/** tanh approximation of GELU (Hendrycks & Gimpel 2016). */
export const geluTanh = (x: number) => 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_C * x * x * x)));
export const swish = (x: number, beta = 1) => x * sigmoid(beta * x);
export const silu = (x: number) => swish(x, 1);
export const mish = (x: number) => x * Math.tanh(softplus(x));

// ---- derivatives ---------------------------------------------------------
export const dStep = (_x: number) => 0;
export const dLinear = (_x: number) => 1;
export const dSigmoid = (x: number) => {
  const s = sigmoid(x);
  return s * (1 - s);
};
export const dTanh = (x: number) => {
  const t = Math.tanh(x);
  return 1 - t * t;
};
export const dRelu = (x: number) => (x > 0 ? 1 : 0);
export const dLeakyRelu = (x: number, alpha = 0.01) => (x > 0 ? 1 : alpha);
export const dPrelu = (x: number, alpha = 0.25) => (x > 0 ? 1 : alpha);
export const dElu = (x: number, alpha = 1) => (x > 0 ? 1 : alpha * Math.exp(x));
export const dSelu = (x: number) => SELU_LAMBDA * (x > 0 ? 1 : SELU_ALPHA * Math.exp(x));
/** d/dx [x·Φ(x)] = Φ(x) + x·φ(x) */
export const dGelu = (x: number) => normCdf(x) + x * normPdf(x);
export const dGeluTanh = (x: number) => {
  const u = SQRT_2_OVER_PI * (x + GELU_C * x * x * x);
  const t = Math.tanh(u);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * GELU_C * x * x);
};
/** d/dx [x·σ(βx)] = σ(βx) + βx·σ(βx)(1 − σ(βx)) */
export const dSwish = (x: number, beta = 1) => {
  const s = sigmoid(beta * x);
  return s + beta * x * s * (1 - s);
};
/** d/dx [x·tanh(softplus x)] = tanh(sp) + x·sech²(sp)·σ(x) */
export const dMish = (x: number) => {
  const t = Math.tanh(softplus(x));
  return t + x * (1 - t * t) * sigmoid(x);
};

// ---- registry -------------------------------------------------------------
export type ActivationId = "step" | "linear" | "sigmoid" | "tanh" | "relu" | "leaky-relu" | "prelu" | "elu" | "selu" | "gelu" | "gelu-tanh" | "swish" | "mish";

export interface ActivationDef {
  id: ActivationId;
  name: string;
  /** plain-text formula for labels */
  formula: string;
  range: string;
  /** which slider parameter the function uses, if any */
  param?: "alpha" | "beta";
  f: (x: number, p: ActParams) => number;
  df: (x: number, p: ActParams) => number;
}

export const ACTIVATIONS: ActivationDef[] = [
  { id: "step", name: "Binary step", formula: "1 if x ≥ 0 else 0", range: "{0, 1}", f: (x) => step(x), df: (x) => dStep(x) },
  { id: "linear", name: "Linear", formula: "x", range: "(−∞, ∞)", f: (x) => linear(x), df: (x) => dLinear(x) },
  { id: "sigmoid", name: "Sigmoid", formula: "1 / (1 + e^(−x))", range: "(0, 1)", f: (x) => sigmoid(x), df: (x) => dSigmoid(x) },
  { id: "tanh", name: "Tanh", formula: "(eˣ − e^(−x)) / (eˣ + e^(−x))", range: "(−1, 1)", f: (x) => tanh(x), df: (x) => dTanh(x) },
  { id: "relu", name: "ReLU", formula: "max(0, x)", range: "[0, ∞)", f: (x) => relu(x), df: (x) => dRelu(x) },
  { id: "leaky-relu", name: "Leaky ReLU", formula: "x if x > 0 else αx", range: "(−∞, ∞)", param: "alpha", f: (x, p) => leakyRelu(x, p.alpha), df: (x, p) => dLeakyRelu(x, p.alpha) },
  { id: "prelu", name: "PReLU", formula: "x if x > 0 else αx (α learned)", range: "(−∞, ∞)", param: "alpha", f: (x, p) => prelu(x, p.alpha), df: (x, p) => dPrelu(x, p.alpha) },
  { id: "elu", name: "ELU", formula: "x if x > 0 else α(eˣ − 1)", range: "(−α, ∞)", param: "alpha", f: (x, p) => elu(x, p.alpha), df: (x, p) => dElu(x, p.alpha) },
  { id: "selu", name: "SELU", formula: "λx if x > 0 else λα(eˣ − 1)", range: "(−λα, ∞) ≈ (−1.758, ∞)", f: (x) => selu(x), df: (x) => dSelu(x) },
  { id: "gelu", name: "GELU (exact)", formula: "x·Φ(x)", range: "≈ [−0.170, ∞)", f: (x) => gelu(x), df: (x) => dGelu(x) },
  { id: "gelu-tanh", name: "GELU (tanh approx.)", formula: "0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))", range: "≈ [−0.170, ∞)", f: (x) => geluTanh(x), df: (x) => dGeluTanh(x) },
  { id: "swish", name: "Swish / SiLU", formula: "x·σ(βx)", range: "≈ [−0.278/β, ∞)", param: "beta", f: (x, p) => swish(x, p.beta), df: (x, p) => dSwish(x, p.beta) },
  { id: "mish", name: "Mish", formula: "x·tanh(ln(1 + eˣ))", range: "≈ [−0.309, ∞)", f: (x) => mish(x), df: (x) => dMish(x) },
];

export function getActivation(id: ActivationId): ActivationDef {
  const a = ACTIVATIONS.find((d) => d.id === id);
  if (!a) throw new Error(`unknown activation ${id}`);
  return a;
}

/** Sample f (or f') on an evenly spaced grid. */
export function sampleCurve(fn: (x: number) => number, lo: number, hi: number, n: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const x = lo + ((hi - lo) * i) / (n - 1);
    pts.push([x, fn(x)]);
  }
  return pts;
}

export interface SoftmaxResult {
  /** logits divided by temperature */
  scaled: number[];
  /** max of the scaled logits (subtracted for stability) */
  max: number;
  /** exp(scaled − max) */
  exps: number[];
  sum: number;
  probs: number[];
}

/** Softmax with temperature T, computed the numerically stable way (subtract the max). */
export function softmaxT(logits: number[], T = 1): SoftmaxResult {
  const temp = Math.max(T, 1e-6);
  const scaled = logits.map((z) => z / temp);
  const max = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return { scaled, max, exps, sum, probs: exps.map((e) => e / sum) };
}

/** Softmax Jacobian J[i][j] = p_i(δ_ij − p_j) (for T = 1). */
export function softmaxJacobian(p: number[]): number[][] {
  return p.map((pi, i) => p.map((pj, j) => pi * ((i === j ? 1 : 0) - pj)));
}
