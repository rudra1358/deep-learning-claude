/**
 * Pure math for the chain-rule simulator (no React).
 *
 * Graph for   L = (σ(w·x + b) − y)²
 *
 *   w ─┐
 *      (×) u ──(+) z ──(σ) a ──(−) d ──(²) L
 *   x ─┘        │               │
 *               b               y
 *
 * Forward pass computes node values left→right; backward pass multiplies the
 * upstream gradient by each edge's local derivative, right→left.
 */

export interface ChainParams {
  x: number;
  w: number;
  b: number;
  y: number;
}

export interface Forward {
  u: number; // w·x
  z: number; // u + b
  a: number; // σ(z)
  d: number; // a − y
  L: number; // d²
}

/** ∂L/∂(node) for every node, plus the local derivative on every edge. */
export interface Backward {
  grad: Record<NodeId, number>;
  local: Record<EdgeId, number>;
}

export type NodeId = "x" | "w" | "b" | "y" | "u" | "z" | "a" | "d" | "L";
export type EdgeId = "w-u" | "x-u" | "u-z" | "b-z" | "z-a" | "a-d" | "y-d" | "d-L";

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function forward({ x, w, b, y }: ChainParams): Forward {
  const u = w * x;
  const z = u + b;
  const a = sigmoid(z);
  const d = a - y;
  const L = d * d;
  return { u, z, a, d, L };
}

export function loss(p: ChainParams): number {
  return forward(p).L;
}

export function backward(p: ChainParams, f: Forward = forward(p)): Backward {
  // Local derivatives (child w.r.t. parent) on each edge.
  const local: Record<EdgeId, number> = {
    "d-L": 2 * f.d, // ∂(d²)/∂d
    "a-d": 1, // ∂(a − y)/∂a
    "y-d": -1, // ∂(a − y)/∂y
    "z-a": f.a * (1 - f.a), // σ'(z) = σ(z)(1 − σ(z))
    "u-z": 1, // ∂(u + b)/∂u
    "b-z": 1, // ∂(u + b)/∂b
    "w-u": p.x, // ∂(w·x)/∂w
    "x-u": p.w, // ∂(w·x)/∂x
  };
  const gL = 1;
  const gd = gL * local["d-L"];
  const ga = gd * local["a-d"];
  const gy = gd * local["y-d"];
  const gz = ga * local["z-a"];
  const gu = gz * local["u-z"];
  const gb = gz * local["b-z"];
  const gw = gu * local["w-u"];
  const gx = gu * local["x-u"];
  return { grad: { L: gL, d: gd, a: ga, y: gy, z: gz, u: gu, b: gb, w: gw, x: gx }, local };
}

/** Central finite difference (f(θ+h) − f(θ−h)) / 2h for one parameter. */
export function finiteDiff(p: ChainParams, key: keyof ChainParams, h = 1e-4): number {
  const plus = { ...p, [key]: p[key] + h };
  const minus = { ...p, [key]: p[key] - h };
  return (loss(plus) - loss(minus)) / (2 * h);
}

/* ---------------- Stepper plan ---------------- */

export interface ChainStep {
  phase: "intro" | "forward" | "backward" | "check";
  /** nodes whose forward value is visible after this step */
  values: NodeId[];
  /** nodes whose gradient ∂L/∂node is visible after this step */
  grads: NodeId[];
  /** nodes being computed in this step (highlighted) */
  active: NodeId[];
  /** edges whose local derivative is being used in this step */
  edges: EdgeId[];
}

const LEAVES: NodeId[] = ["x", "w", "b", "y"];
const FWD_ORDER: NodeId[] = ["u", "z", "a", "d", "L"];
const BWD_ORDER: { nodes: NodeId[]; edges: EdgeId[] }[] = [
  { nodes: ["L"], edges: [] },
  { nodes: ["d"], edges: ["d-L"] },
  { nodes: ["a", "y"], edges: ["a-d", "y-d"] },
  { nodes: ["z"], edges: ["z-a"] },
  { nodes: ["u", "b"], edges: ["u-z", "b-z"] },
  { nodes: ["w", "x"], edges: ["w-u", "x-u"] },
];

export const CHAIN_STEPS: ChainStep[] = (() => {
  const steps: ChainStep[] = [{ phase: "intro", values: [...LEAVES], grads: [], active: [...LEAVES], edges: [] }];
  const vals: NodeId[] = [...LEAVES];
  for (const n of FWD_ORDER) {
    vals.push(n);
    steps.push({ phase: "forward", values: [...vals], grads: [], active: [n], edges: [] });
  }
  const grads: NodeId[] = [];
  for (const g of BWD_ORDER) {
    grads.push(...g.nodes);
    steps.push({ phase: "backward", values: [...vals], grads: [...grads], active: [...g.nodes], edges: [...g.edges] });
  }
  steps.push({ phase: "check", values: [...vals], grads: [...grads], active: ["w", "b"], edges: [] });
  return steps;
})();

/** Parents (inputs) of each computed node, used for drawing edges. */
export const EDGES: { id: EdgeId; from: NodeId; to: NodeId }[] = [
  { id: "w-u", from: "w", to: "u" },
  { id: "x-u", from: "x", to: "u" },
  { id: "u-z", from: "u", to: "z" },
  { id: "b-z", from: "b", to: "z" },
  { id: "z-a", from: "z", to: "a" },
  { id: "a-d", from: "a", to: "d" },
  { id: "y-d", from: "y", to: "d" },
  { id: "d-L", from: "d", to: "L" },
];
