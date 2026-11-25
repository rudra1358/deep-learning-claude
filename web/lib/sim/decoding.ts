/**
 * Pure decoding math for the "Decoding Strategies" simulator (components/sims/DecodingSim.tsx).
 *
 * Everything here works on a vector of logits z ∈ ℝ^V (one score per vocabulary entry) and is
 * independent of the model that produced it, so it can be unit-tested in Node. The simulator
 * feeds it the real logits of the trained character-level Transformer (lib/sim/transformer.ts).
 *
 * Pipeline used for sampling (same order as common libraries: penalty → temperature → top-k → top-p → min-p):
 *   z' = repetitionPenalty(z)          (CTRL-style: positive logits ÷ θ, negative logits × θ, for seen tokens)
 *   p_T = softmax(z' / T)              (temperature)
 *   keep = topK ∩ topP ∩ minP          (truncation; at least the arg-max is always kept)
 *   q = p_T · 1[keep] / Σ_keep p_T     (renormalise over the kept set)
 *
 * The file has no imports so Node can run it directly (node --experimental-strip-types is default in Node ≥ 23).
 */

/** Numerically stable log-softmax with temperature (T > 0). */
export function logSoftmax(z: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  let m = -Infinity;
  for (const x of z) if (x > m) m = x;
  let s = 0;
  for (const x of z) s += Math.exp((x - m) / t);
  const lse = Math.log(s);
  return z.map((x) => (x - m) / t - lse);
}

export function softmax(z: number[], temperature = 1): number[] {
  return logSoftmax(z, temperature).map(Math.exp);
}

/** Indices sorted by value, largest first (ties broken by the smaller index, so the order is deterministic). */
export function argsortDesc(v: number[]): number[] {
  return v.map((_, i) => i).sort((a, b) => v[b] - v[a] || a - b);
}

export function argmax(v: number[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

/**
 * Repetition penalty (Keskar et al., 2019, CTRL): for every token id that already occurs in `history`,
 * divide its logit by θ if positive, multiply by θ if negative. θ = 1 disables it.
 */
export function repetitionPenalty(z: number[], history: number[], theta: number): number[] {
  if (theta === 1) return z.slice();
  const seen = new Set(history);
  return z.map((x, i) => (seen.has(i) ? (x > 0 ? x / theta : x * theta) : x));
}

export interface FilterOptions {
  temperature: number;
  /** 0 = off; otherwise keep the k most probable tokens. */
  topK: number;
  /** 1 = off; otherwise keep the smallest prefix (by probability) whose mass reaches topP. */
  topP: number;
  /** 0 = off; otherwise keep tokens with p ≥ minP · p_max. */
  minP: number;
}

export interface Filtered {
  /** softmax(z / T) before truncation. */
  probsT: number[];
  /** Token ids sorted by probsT, most probable first. */
  order: number[];
  /** keep[i] is true when token i survives every active truncation rule. */
  keep: boolean[];
  /** Final sampling distribution: probsT restricted to the kept set and renormalised. */
  probs: number[];
  /** Σ probsT over the kept set (the mass that survived truncation). */
  keptMass: number;
  keptCount: number;
}

/** Apply temperature, top-k, top-p (nucleus) and min-p to a logit vector. */
export function filterLogits(z: number[], o: FilterOptions): Filtered {
  const probsT = softmax(z, o.temperature);
  const order = argsortDesc(probsT);
  const V = z.length;
  const keep = new Array<boolean>(V).fill(true);
  // top-k: everything ranked k or lower is removed
  if (o.topK > 0) order.forEach((id, r) => r >= o.topK && (keep[id] = false));
  // top-p: walk down the sorted list until the cumulative mass reaches p; the token that crosses p is kept
  if (o.topP < 1) {
    let cum = 0;
    let reached = false;
    for (const id of order) {
      if (reached) keep[id] = false;
      cum += probsT[id];
      if (cum >= o.topP) reached = true;
    }
  }
  // min-p: relative threshold on the most probable token
  if (o.minP > 0) {
    const pmax = probsT[order[0]];
    for (let i = 0; i < V; i++) if (probsT[i] < o.minP * pmax) keep[i] = false;
  }
  keep[order[0]] = true; // never remove the arg-max
  let keptMass = 0;
  let keptCount = 0;
  for (let i = 0; i < V; i++)
    if (keep[i]) {
      keptMass += probsT[i];
      keptCount++;
    }
  const probs = probsT.map((p, i) => (keep[i] ? p / keptMass : 0));
  return { probsT, order, keep, probs, keptMass, keptCount };
}

/** Inverse-CDF sampling: the first index whose cumulative probability exceeds u ∈ [0, 1). */
export function sampleIndex(p: number[], u: number): number {
  let c = 0;
  let last = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    last = i;
    c += p[i];
    if (u < c) return i;
  }
  return last;
}

/** Deterministic PRNG (mulberry32), identical to lib/sim/random.ts. */
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

/* ------------------------------------------------------------------ beam search */

export interface Beam {
  /** Generated token ids (the prompt is not included). */
  ids: number[];
  /** Σ log p(token | prefix) over the generated tokens (natural log, temperature 1). */
  logp: number;
}

export interface BeamCandidate extends Beam {
  /** Index of the parent beam in the previous step's list. */
  parent: number;
  /** The token appended to the parent. */
  token: number;
  /** log p of that token given the parent. */
  tokenLogp: number;
}

export interface BeamStep {
  /** Surviving beams after this step, best first. */
  beams: BeamCandidate[];
  /** Number of candidates scored this step (width × V, or V at the first step). */
  scored: number;
}

/** Length-normalised score used to rank hypotheses of different lengths: logp / len^α (α = 0: raw sum). */
export function lengthNormalised(logp: number, len: number, alpha: number): number {
  return len === 0 ? logp : logp / Math.pow(len, alpha);
}

/**
 * One step of beam search. `logProbs(prefixIds)` returns log p(· | prompt + prefix) over the vocabulary.
 * Every surviving beam is extended by every token; the `width` best (by total log-probability) survive.
 * Without an end-of-sequence token all hypotheses have the same length, so length normalisation does not
 * change the ranking here (see the lesson for the EOS case).
 */
export function beamStep(beams: Beam[], width: number, logProbs: (ids: number[]) => number[]): BeamStep {
  const cands: BeamCandidate[] = [];
  beams.forEach((b, parent) => {
    const lp = logProbs(b.ids);
    lp.forEach((l, token) => cands.push({ ids: [...b.ids, token], logp: b.logp + l, parent, token, tokenLogp: l }));
  });
  // stable, deterministic ranking: higher score first, then lower parent index, then lower token id
  cands.sort((a, b) => b.logp - a.logp || a.parent - b.parent || a.token - b.token);
  return { beams: cands.slice(0, width), scored: cands.length };
}

/** Run `steps` beam-search steps from an empty continuation. */
export function beamSearch(width: number, steps: number, logProbs: (ids: number[]) => number[]): BeamStep[] {
  const out: BeamStep[] = [];
  let beams: Beam[] = [{ ids: [], logp: 0 }];
  for (let s = 0; s < steps; s++) {
    const st = beamStep(beams, width, logProbs);
    out.push(st);
    beams = st.beams;
  }
  return out;
}

/** Shannon entropy in nats of a probability vector. */
export function entropy(p: number[]): number {
  let h = 0;
  for (const x of p) if (x > 0) h -= x * Math.log(x);
  return h;
}
