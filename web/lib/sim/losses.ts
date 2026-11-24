/**
 * Loss functions and their derivatives (pure math, no React).
 *
 * Regression losses are functions of the residual r = ŷ − y.
 * Classification losses are functions of the logit z for a true label y ∈ {0, 1};
 * every derivative returned here is dL/dz — the gradient backpropagation sends into the network.
 */

export interface LossValue {
  loss: number;
  grad: number;
}

/* ---------------- regression (argument: residual r) ---------------- */

export const mse = (r: number): LossValue => ({ loss: r * r, grad: 2 * r });
export const mae = (r: number): LossValue => ({ loss: Math.abs(r), grad: Math.sign(r) });

export function huber(r: number, delta: number): LossValue {
  const a = Math.abs(r);
  return a <= delta ? { loss: 0.5 * r * r, grad: r } : { loss: delta * (a - 0.5 * delta), grad: delta * Math.sign(r) };
}

/** log(cosh r) computed stably as |r| + log1p(e^{−2|r|}) − log 2. */
export function logCosh(r: number): LossValue {
  const a = Math.abs(r);
  return { loss: a + Math.log1p(Math.exp(-2 * a)) - Math.LN2, grad: Math.tanh(r) };
}

/* ---------------- classification (argument: logit z, label y) ---------------- */

/** log(1 + e^z) without overflow. */
export function softplus(z: number) {
  return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
}

export function sigmoid(z: number) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export function logit(p: number) {
  return Math.log(p / (1 - p));
}

/** Binary cross-entropy from logits: y·softplus(−z) + (1−y)·softplus(z). Gradient p − y. */
export function bce(z: number, y: 0 | 1): LossValue {
  return { loss: y === 1 ? softplus(-z) : softplus(z), grad: sigmoid(z) - y };
}

/**
 * Focal loss (Lin et al., 2017) with α = 1: −(1 − p_t)^γ log p_t.
 * With z_t = z for y = 1 and −z for y = 0, p_t = σ(z_t) and
 * dL/dz_t = γ p_t (1 − p_t)^γ log p_t − (1 − p_t)^{γ+1};  dL/dz = ±dL/dz_t.
 */
export function focal(z: number, y: 0 | 1, gamma: number): LossValue {
  const s = y === 1 ? 1 : -1;
  const zt = s * z;
  const pt = sigmoid(zt);
  const logPt = -softplus(-zt);
  const q = 1 - pt;
  const loss = -(q ** gamma) * logPt;
  const dzt = gamma * pt * q ** gamma * logPt - q ** (gamma + 1);
  return { loss, grad: s * dzt };
}

/** Hinge loss on the margin m = (2y − 1)·z: max(0, 1 − m). Subgradient −(2y−1) when m < 1. */
export function hinge(z: number, y: 0 | 1): LossValue {
  const s = y === 1 ? 1 : -1;
  const m = s * z;
  return { loss: Math.max(0, 1 - m), grad: m < 1 ? -s : 0 };
}

/** Squared error on the probability: (σ(z) − y)². Gradient 2(p − y)·p(1 − p). */
export function mseProb(z: number, y: 0 | 1): LossValue {
  const p = sigmoid(z);
  return { loss: (p - y) ** 2, grad: 2 * (p - y) * p * (1 - p) };
}

/** Uniform grid of n+1 points on [a, b]. */
export function linspace(a: number, b: number, n: number) {
  return Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);
}
