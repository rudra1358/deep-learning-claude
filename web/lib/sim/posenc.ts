/**
 * Pure math for the Positional Encoding simulator (no React).
 *
 * Sinusoidal encoding from Vaswani et al. (2017):
 *   PE[pos, 2i]   = sin(pos / base^(2i/d))
 *   PE[pos, 2i+1] = cos(pos / base^(2i/d))
 */
import type { Matrix } from "./random";

export const PE_BASE = 10000;

/** Angular frequency ω_i of the dimension pair i (dimensions 2i and 2i+1). */
export function frequency(i: number, d: number, base = PE_BASE): number {
  return 1 / Math.pow(base, (2 * i) / d);
}

/** Wavelength (in positions) of dimension j: 2π / ω. */
export function wavelength(j: number, d: number, base = PE_BASE): number {
  return (2 * Math.PI) / frequency(Math.floor(j / 2), d, base);
}

export function peValue(pos: number, j: number, d: number, base = PE_BASE): number {
  const w = frequency(Math.floor(j / 2), d, base);
  return j % 2 === 0 ? Math.sin(pos * w) : Math.cos(pos * w);
}

/** The full n × d positional-encoding matrix. */
export function peMatrix(n: number, d: number, base = PE_BASE): Matrix {
  return Array.from({ length: n }, (_, pos) => Array.from({ length: d }, (_, j) => peValue(pos, j, d, base)));
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** PE(p) · PE(q) for every q. Analytically equal to Σ_i cos(ω_i (p − q)), so it depends only on p − q. */
export function similarityRow(P: Matrix, p: number): number[] {
  return P.map((row) => dot(P[p], row));
}

/** Closed form Σ_i cos(ω_i k) — used to show the dot product is a function of the offset k alone. */
export function similarityClosedForm(k: number, d: number, base = PE_BASE): number {
  let s = 0;
  for (let i = 0; i < d / 2; i++) s += Math.cos(frequency(i, d, base) * k);
  return s;
}
