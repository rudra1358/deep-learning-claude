/**
 * Pure math for the Entropy / Cross-Entropy / KL simulator (no React).
 *
 * All quantities are computed in nats (natural log) and converted for display.
 * Conventions:  0·log(0/q) = 0;  p·log(p/0) = +∞ for p > 0.
 *
 *   H(p)      = −Σ p_i log p_i
 *   H(p, q)   = −Σ p_i log q_i
 *   KL(p‖q)   =  Σ p_i log(p_i / q_i)       = H(p, q) − H(p)
 *   JS(p, q)  = ½ KL(p‖m) + ½ KL(q‖m),  m = ½(p + q)
 *
 * Training: q = softmax(z). Two model families:
 *   "free"  — z ∈ R^K (K logits), q can become any distribution with full support;
 *   "gauss" — z_i = −(i − μ)² / (2σ²), a discretised Gaussian bump with 2 parameters (μ, s = log σ).
 * Two objectives:
 *   "forward" — minimise KL(p‖q) (equivalently H(p, q)):   ∂/∂z_j = q_j − p_j
 *   "reverse" — minimise KL(q‖p):                           ∂/∂z_j = q_j (r_j − Σ_i q_i r_i),  r_i = log(q_i/p_i)
 */

import { softmax } from "./random";

export type Unit = "bits" | "nats";
export type PresetName = "uniform" | "peaked" | "one-hot" | "bimodal";
export type Family = "free" | "gauss";
export type Objective = "forward" | "reverse";

export const PRESETS: PresetName[] = ["uniform", "peaked", "one-hot", "bimodal"];

/** Floor applied to p inside log p for the reverse-KL gradient, so zero-probability outcomes give a finite (large) push. */
export const P_FLOOR = 1e-6;
/** Floor applied to q when converting an edited q into logits (log 0 = −∞ would freeze that logit forever). */
export const Q_FLOOR = 1e-6;

export const toUnit = (nats: number, unit: Unit) => (unit === "bits" ? nats / Math.LN2 : nats);

/** −p·log q with the 0·log 0 = 0 convention. */
function negPlogQ(p: number, q: number) {
  if (p <= 0) return 0;
  if (q <= 0) return Infinity;
  return -p * Math.log(q);
}

/** p·log(p/q) with the conventions above. */
export function klTerm(p: number, q: number) {
  if (p <= 0) return 0;
  if (q <= 0) return Infinity;
  return p * Math.log(p / q);
}

export function entropy(p: number[]) {
  return p.reduce((s, pi) => s + negPlogQ(pi, pi), 0);
}

export function crossEntropy(p: number[], q: number[]) {
  return p.reduce((s, pi, i) => s + negPlogQ(pi, q[i]), 0);
}

export function kl(p: number[], q: number[]) {
  // Clamp tiny negative round-off (KL ≥ 0 by Gibbs' inequality).
  return Math.max(0, p.reduce((s, pi, i) => s + klTerm(pi, q[i]), 0));
}

export function js(p: number[], q: number[]) {
  const m = p.map((pi, i) => 0.5 * (pi + q[i]));
  return 0.5 * kl(p, m) + 0.5 * kl(q, m);
}

export interface Measures {
  Hp: number;
  Hq: number;
  Hpq: number;
  klPQ: number;
  klQP: number;
  js: number;
}

export function measures(p: number[], q: number[]): Measures {
  return { Hp: entropy(p), Hq: entropy(q), Hpq: crossEntropy(p, q), klPQ: kl(p, q), klQP: kl(q, p), js: js(p, q) };
}

export function normalize(w: number[]) {
  const s = w.reduce((a, b) => a + Math.max(0, b), 0);
  if (s <= 0) return w.map(() => 1 / w.length);
  return w.map((x) => Math.max(0, x) / s);
}

/**
 * Set outcome i to probability v and rescale the others proportionally so the total stays 1.
 * If every other outcome is 0, the remaining mass is shared equally.
 */
export function setBar(dist: number[], i: number, v: number) {
  const K = dist.length;
  const val = Math.min(1, Math.max(0, v));
  const rest = dist.reduce((s, x, j) => (j === i ? s : s + x), 0);
  return dist.map((x, j) => {
    if (j === i) return val;
    if (rest <= 1e-12) return (1 - val) / (K - 1);
    return (x * (1 - val)) / rest;
  });
}

export function preset(name: PresetName, K: number): number[] {
  const idx = Array.from({ length: K }, (_, i) => i);
  const c = Math.floor((K - 1) / 3);
  switch (name) {
    case "uniform":
      return idx.map(() => 1 / K);
    case "peaked":
      return normalize(idx.map((i) => Math.pow(0.3, Math.abs(i - c))));
    case "one-hot":
      return idx.map((i) => (i === c ? 1 : 0));
    case "bimodal": {
      const a = Math.round((K - 1) * 0.15);
      const b = K - 1 - a;
      const g = (i: number, m: number) => Math.exp(-((i - m) ** 2) / (2 * 0.6 * 0.6));
      return normalize(idx.map((i) => (a === b ? 1 : g(i, a) + g(i, b))));
    }
  }
}

/** Logits for the discretised Gaussian family. */
export function gaussLogits(K: number, mu: number, logSigma: number) {
  const s2 = Math.exp(2 * logSigma);
  return Array.from({ length: K }, (_, i) => -((i - mu) ** 2) / (2 * s2));
}

export const gaussQ = (K: number, mu: number, logSigma: number) => softmax(gaussLogits(K, mu, logSigma));

export function logitsFromQ(q: number[]) {
  return q.map((qi) => Math.log(Math.max(qi, Q_FLOOR)));
}

/** Gradient of the chosen objective with respect to the softmax logits z. */
export function gradLogits(p: number[], q: number[], obj: Objective) {
  if (obj === "forward") return q.map((qi, j) => qi - p[j]);
  const r = q.map((qi, j) => Math.log(Math.max(qi, 1e-300)) - Math.log(Math.max(p[j], P_FLOOR)));
  const mean = q.reduce((s, qi, j) => s + qi * r[j], 0);
  return q.map((qi, j) => qi * (r[j] - mean));
}

/** The objective value being minimised (nats). Reverse KL uses the same p-floor as its gradient. */
export function objectiveValue(p: number[], q: number[], obj: Objective) {
  if (obj === "forward") return kl(p, q);
  return Math.max(0, q.reduce((s, qi, j) => (qi <= 0 ? s : s + qi * (Math.log(qi) - Math.log(Math.max(p[j], P_FLOOR)))), 0));
}

/** One plain gradient-descent step on the K free logits. */
export function stepFree(z: number[], p: number[], obj: Objective, lr: number) {
  const g = gradLogits(p, softmax(z), obj);
  const zn = z.map((zi, j) => zi - lr * g[j]);
  const m = Math.max(...zn); // logits are shift-invariant: re-centre to keep them bounded
  return zn.map((zi) => zi - m);
}

export const GAUSS_LIMITS = { maxMuStep: 0.3, maxSStep: 0.2, minSigma: 0.2 };

/**
 * One gradient-descent step on (μ, s = log σ) of the Gaussian family, by the chain rule
 *   ∂J/∂μ = Σ_i g_i (i − μ)/σ²,   ∂J/∂s = Σ_i g_i (i − μ)²/σ²,   g = ∂J/∂z.
 * Each step is clipped (|Δμ| ≤ 0.3, |Δs| ≤ 0.2) so very narrow bumps cannot jump; σ is kept in [0.2, K].
 */
export function stepGauss(mu: number, s: number, p: number[], obj: Objective, lr: number) {
  const K = p.length;
  const q = gaussQ(K, mu, s);
  const g = gradLogits(p, q, obj);
  const s2 = Math.exp(2 * s);
  let dmu = 0, ds = 0;
  for (let i = 0; i < K; i++) {
    dmu += (g[i] * (i - mu)) / s2;
    ds += (g[i] * (i - mu) ** 2) / s2;
  }
  const clip = (v: number, c: number) => Math.max(-c, Math.min(c, v));
  const muN = Math.max(-1, Math.min(K, mu - clip(lr * dmu, GAUSS_LIMITS.maxMuStep)));
  const sN = Math.max(Math.log(GAUSS_LIMITS.minSigma), Math.min(Math.log(K), s - clip(lr * ds, GAUSS_LIMITS.maxSStep)));
  return { mu: muN, s: sN, gradMu: dmu, gradS: ds };
}
