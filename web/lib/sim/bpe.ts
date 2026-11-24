/**
 * Byte-Pair Encoding (Sennrich et al., 2016) trainer and encoder for the tokenizer simulator.
 * Pure functions, no React. Mirrors the Python implementation in the m13/text-preprocessing lesson:
 *   - normalize: NFKC + lowercase, punctuation replaced by spaces
 *   - each word becomes characters + an end-of-word marker "_" (the paper writes </w>)
 *   - repeatedly merge the most frequent adjacent pair (ties broken lexicographically)
 *   - stop early when the best pair occurs fewer than 2 times
 */

export const EOW = "_";

export interface PairCount {
  a: string;
  b: string;
  count: number;
}

export interface BpeState {
  /** Merges applied so far (in order). */
  merges: PairCount[];
  /** Current segmentation of every distinct word. */
  segs: Record<string, string[]>;
  /** Pair counts at this state, sorted (most frequent first). */
  pairs: PairCount[];
}

export interface BpeTrace {
  /** Distinct words in order of first appearance, with frequency. */
  words: { word: string; freq: number }[];
  /** Normalized corpus as a word sequence (for re-segmenting the running text). */
  corpusWords: string[];
  /** Base alphabet (sorted). */
  alphabet: string[];
  /** states[k] = after k merges. states.length = merges learned + 1. */
  states: BpeState[];
}

export function normalize(text: string): string {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " "); // also removes "_", so it can serve as the end-of-word marker
}

export function pretokenize(text: string): string[] {
  return normalize(text)
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

const cmpPair = (x: PairCount, y: PairCount) => y.count - x.count || (x.a < y.a ? -1 : x.a > y.a ? 1 : x.b < y.b ? -1 : x.b > y.b ? 1 : 0);

export function countPairs(words: { word: string; freq: number }[], segs: Record<string, string[]>): PairCount[] {
  const m = new Map<string, PairCount>();
  for (const { word, freq } of words) {
    const s = segs[word];
    for (let k = 0; k + 1 < s.length; k++) {
      const key = s[k] + "\u0000" + s[k + 1];
      const e = m.get(key);
      if (e) e.count += freq;
      else m.set(key, { a: s[k], b: s[k + 1], count: freq });
    }
  }
  return [...m.values()].sort(cmpPair);
}

/** Replace every non-overlapping occurrence of (a, b), scanning left to right. */
export function applyMerge(symbols: string[], a: string, b: string): string[] {
  const out: string[] = [];
  let k = 0;
  while (k < symbols.length) {
    if (k + 1 < symbols.length && symbols[k] === a && symbols[k + 1] === b) {
      out.push(a + b);
      k += 2;
    } else {
      out.push(symbols[k]);
      k += 1;
    }
  }
  return out;
}

/** Split a word into characters (code points) plus the end-of-word marker. */
export const initialSymbols = (word: string) => [...Array.from(word), EOW];

export function trainBpe(text: string, numMerges: number): BpeTrace {
  const corpusWords = pretokenize(text);
  const freq = new Map<string, number>();
  for (const w of corpusWords) freq.set(w, (freq.get(w) ?? 0) + 1);
  const words = [...freq.entries()].map(([word, f]) => ({ word, freq: f }));

  let segs: Record<string, string[]> = {};
  for (const { word } of words) segs[word] = initialSymbols(word);
  const alphabet = [...new Set(Object.values(segs).flat())].sort();

  const states: BpeState[] = [];
  const merges: PairCount[] = [];
  let pairs = countPairs(words, segs);
  states.push({ merges: [], segs, pairs });
  for (let m = 0; m < numMerges; m++) {
    const best = pairs[0];
    if (!best || best.count < 2) break;
    merges.push(best);
    const next: Record<string, string[]> = {};
    for (const { word } of words) next[word] = applyMerge(segs[word], best.a, best.b);
    segs = next;
    pairs = countPairs(words, segs);
    states.push({ merges: merges.slice(), segs, pairs });
  }
  return { words, corpusWords, alphabet, states };
}

/** Vocabulary after k merges: <unk>, base alphabet, then merged tokens in merge order. */
export function vocabulary(trace: BpeTrace, k: number): string[] {
  const v = ["<unk>", ...trace.alphabet];
  for (const m of trace.states[k].merges) v.push(m.a + m.b);
  return v;
}

export interface EncodedPiece {
  token: string;
  id: number;
  /** Index of the source word (for grouping chips). */
  word: number;
}

/** Encode new text by applying the first k learned merges in order to every word. */
export function encode(text: string, trace: BpeTrace, k: number): EncodedPiece[] {
  const vocab = vocabulary(trace, k);
  const id = new Map(vocab.map((t, i) => [t, i]));
  const merges = trace.states[k].merges;
  const out: EncodedPiece[] = [];
  pretokenize(text).forEach((w, wi) => {
    let s = initialSymbols(w);
    for (const m of merges) s = applyMerge(s, m.a, m.b);
    for (const tok of s) out.push({ token: tok, id: id.get(tok) ?? 0, word: wi });
  });
  return out;
}
