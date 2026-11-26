/**
 * Graph message passing on tiny graphs (pure math, no React).
 *
 * Every node carries ONE scalar feature h_i so that every number in the simulator can be shown.
 * The learnable weight matrix of a real layer is fixed to W = 1 and there is no non-linearity, which
 * isolates the part that makes a GNN a GNN: how information is aggregated along edges.
 *
 * GCN (Kipf & Welling 2017):   h' = Â h,  Â = D̃^{-1/2} (A + I) D̃^{-1/2},  d̃_i = deg(i) + 1
 *   so  h'_i = Σ_{j ∈ N(i) ∪ {i}}  h_j / sqrt(d̃_i d̃_j).
 * GAT (Veličković et al. 2018): e_ij = LeakyReLU(a_l·W h_i + a_r·W h_j),  α_ij = softmax_j over N(i) ∪ {i},
 *   h'_i = Σ_j α_ij W h_j.
 *
 * Over-smoothing: repeated GCN propagation converges to the dominant eigenvector of Â, which is
 * proportional to sqrt(d̃_i); the normalised Dirichlet energy hᵀ(I − Â)h measures the distance from that
 * limit and decays to 0. GAT rows are convex combinations, so repeated GAT layers drive all nodes towards
 * a common value; we measure that with the plain Dirichlet energy Σ_{(i,j)∈E} (h_i − h_j)².
 */
import { mulberry32, gaussian } from "./random";

export type GnnMode = "gcn" | "gat";

export interface Graph {
  key: string;
  name: string;
  n: number;
  edges: [number, number][];
  /** node positions in a 320 × 220 viewBox */
  pos: [number, number][];
  /** default initial features */
  x0: number[];
  /** optional ground-truth community of each node (for the narration) */
  group?: number[];
  blurb: string;
}

export const GRAPHS: Graph[] = [
  {
    key: "communities",
    name: "Two communities",
    n: 8,
    edges: [
      [0, 1], [0, 2], [1, 2], [1, 3], [2, 3],
      [3, 4],
      [4, 5], [4, 6], [5, 6], [5, 7], [6, 7],
    ],
    pos: [
      [40, 110], [90, 50], [90, 170], [145, 110],
      [195, 110], [250, 50], [250, 170], [290, 110],
    ],
    x0: [0.9, 1.2, -0.3, 0.8, -1.0, -0.7, -1.3, 0.4],
    group: [0, 0, 0, 0, 1, 1, 1, 1],
    blurb: "Two groups (communities), nodes 0–3 and 4–7, joined by one bridge edge (3–4). Features are mostly positive on the left, negative on the right; nodes 2 and 7 are noisy.",
  },
  {
    key: "path",
    name: "Path (one-hot)",
    n: 7,
    edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    pos: [[25, 110], [70, 110], [115, 110], [160, 110], [205, 110], [250, 110], [295, 110]],
    x0: [1, 0, 0, 0, 0, 0, 0],
    blurb: "A chain with a signal only on node 0. Each layer passes information one hop (one edge). After k layers, node k is the farthest node that has 'heard' about node 0.",
  },
  {
    key: "star",
    name: "Star (hub)",
    n: 7,
    edges: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
    pos: [[160, 110], [160, 30], [240, 70], [240, 150], [160, 190], [80, 150], [80, 70]],
    x0: [0, 1.5, -0.5, 0.8, -1.2, 0.3, 1.0],
    blurb: "A hub joined to six leaves. The hub has degree 6 (6 neighbours), each leaf 1. With self-loops, symmetric normalisation scales hub–leaf messages by 1/sqrt(7·2).",
  },
];

/** Neighbour lists including the self-loop (sorted, self first). */
export function neighbourhoods(g: Graph): number[][] {
  const nb: number[][] = Array.from({ length: g.n }, (_, i) => [i]);
  for (const [a, b] of g.edges) {
    nb[a].push(b);
    nb[b].push(a);
  }
  return nb;
}

/** Degrees with the self-loop: d̃_i = deg(i) + 1. */
export function tildeDegrees(g: Graph): number[] {
  return neighbourhoods(g).map((nb) => nb.length);
}

/** Â = D̃^{-1/2}(A + I)D̃^{-1/2} as a dense n × n matrix. */
export function gcnMatrix(g: Graph): number[][] {
  const d = tildeDegrees(g);
  const M = Array.from({ length: g.n }, () => new Array<number>(g.n).fill(0));
  neighbourhoods(g).forEach((nb, i) => {
    for (const j of nb) M[i][j] = 1 / Math.sqrt(d[i] * d[j]);
  });
  return M;
}

export const leakyRelu = (x: number, slope: number) => (x >= 0 ? x : slope * x);

/** GAT attention matrix α (row i = weights node i puts on its neighbours, rows sum to 1). */
export function gatMatrix(g: Graph, h: number[], aL: number, aR: number, slope = 0.2): { alpha: number[][]; e: number[][] } {
  const nb = neighbourhoods(g);
  const alpha = Array.from({ length: g.n }, () => new Array<number>(g.n).fill(0));
  const e = Array.from({ length: g.n }, () => new Array<number>(g.n).fill(Number.NaN));
  for (let i = 0; i < g.n; i++) {
    let m = -Infinity;
    for (const j of nb[i]) {
      e[i][j] = leakyRelu(aL * h[i] + aR * h[j], slope);
      m = Math.max(m, e[i][j]);
    }
    let s = 0;
    for (const j of nb[i]) s += Math.exp(e[i][j] - m);
    for (const j of nb[i]) alpha[i][j] = Math.exp(e[i][j] - m) / s;
  }
  return { alpha, e };
}

export function apply(M: number[][], h: number[]): number[] {
  return M.map((row) => row.reduce((s, w, j) => s + w * h[j], 0));
}

/** Normalised Dirichlet energy hᵀ(I − Â)h (0 exactly when h ∝ sqrt(d̃)). */
export function gcnEnergy(g: Graph, h: number[]): number {
  const Ah = apply(gcnMatrix(g), h);
  let s = 0;
  for (let i = 0; i < g.n; i++) s += h[i] * (h[i] - Ah[i]);
  return s;
}

/** Plain Dirichlet energy Σ_{(i,j)∈E} (h_i − h_j)² (0 exactly when all connected nodes are equal). */
export function plainEnergy(g: Graph, h: number[]): number {
  let s = 0;
  for (const [a, b] of g.edges) s += (h[a] - h[b]) ** 2;
  return s;
}

export interface Propagation {
  /** H[l] = features after l layers (H[0] = input) */
  H: number[][];
  /** M[l] = aggregation matrix used to go from layer l to l + 1 */
  M: number[][][];
  /** E[l] = raw attention scores at layer l (GAT only) */
  scores: number[][][];
  energy: number[];
}

export function propagate(g: Graph, x0: number[], mode: GnnMode, layers: number, aL = 1, aR = 1, slope = 0.2): Propagation {
  const H: number[][] = [x0.slice()];
  const M: number[][][] = [];
  const scores: number[][][] = [];
  const energy: number[] = [];
  const A = gcnMatrix(g);
  const en = (h: number[]) => (mode === "gcn" ? gcnEnergy(g, h) : plainEnergy(g, h));
  energy.push(en(x0));
  for (let l = 0; l < layers; l++) {
    const h = H[l];
    let Ml: number[][];
    if (mode === "gcn") {
      Ml = A;
      scores.push([]);
    } else {
      const r = gatMatrix(g, h, aL, aR, slope);
      Ml = r.alpha;
      scores.push(r.e);
    }
    M.push(Ml);
    const next = apply(Ml, h);
    H.push(next);
    energy.push(en(next));
  }
  return { H, M, scores, energy };
}

/** Random features with a community signal (communities) or plain N(0,1) (others). */
export function randomFeatures(g: Graph, seed: number): number[] {
  const r = mulberry32(seed);
  return Array.from({ length: g.n }, (_, i) => {
    const base = g.group ? (g.group[i] === 0 ? 0.8 : -0.8) : 0;
    return Math.round((base + gaussian(r) * (g.group ? 0.6 : 1)) * 100) / 100;
  });
}

/** The over-smoothing limit of GCN propagation: projection of x0 on the unit eigenvector u ∝ sqrt(d̃). */
export function gcnLimit(g: Graph, x0: number[]): number[] {
  const d = tildeDegrees(g);
  const norm = Math.sqrt(d.reduce((s, v) => s + v, 0));
  const u = d.map((v) => Math.sqrt(v) / norm);
  const c = u.reduce((s, ui, i) => s + ui * x0[i], 0);
  return u.map((ui) => c * ui);
}
