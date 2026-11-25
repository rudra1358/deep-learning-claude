/**
 * Tabular Q-learning in a deterministic gridworld (pure math, no React).
 *
 * Grid (5 rows × 6 columns):           Rewards: entering G = +1 (terminal), entering P = −1 (terminal),
 *   .  .  .  .  .  G                    every other move 0. Moving into a wall or off the grid leaves the
 *   .  #  #  #  .  P                    agent where it is. Episodes start at S and are cut off after
 *   .  .  .  #  .  .                    MAX_STEPS moves (time-out, return 0).
 *   #  #  .  .  .  .
 *   S  .  .  P  .  .
 *
 * Q-learning (Watkins 1989):  Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]  (no bootstrap at terminals)
 * Optional exploring starts: each episode begins in a uniformly random free cell, so states far from the
 * path S → G keep being visited (with a fixed start and small ε they are almost never updated).
 * Behaviour policy: ε-greedy with ties broken uniformly at random (all-zero initial Q would otherwise
 * always pick the same action). The optimal values Q* are computed by value iteration for comparison.
 */
import { mulberry32 } from "./random";

export const ROWS = 5;
export const COLS = 6;
export const MAX_STEPS = 100;
export const LAYOUT = [
  "....." + "G",
  ".###.P",
  "...#..",
  "##....",
  "S..P..",
];
export const ACTIONS = ["up", "right", "down", "left"] as const;
export const ARROWS = ["↑", "→", "↓", "←"];
const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

export type Cell = "." | "#" | "G" | "P" | "S";
export const N_STATES = ROWS * COLS;
export const cellAt = (s: number): Cell => LAYOUT[Math.floor(s / COLS)][s % COLS] as Cell;
export const isWall = (s: number) => cellAt(s) === "#";
export const isTerminal = (s: number) => cellAt(s) === "G" || cellAt(s) === "P";
export const START = LAYOUT.findIndex((row) => row.includes("S")) * COLS + LAYOUT.find((row) => row.includes("S"))!.indexOf("S");

/** Non-wall, non-terminal cells (possible start states with exploring starts). */
export const FREE_STATES = Array.from({ length: ROWS * COLS }, (_, s) => s).filter((s) => !isWall(s) && !isTerminal(s));

/** Deterministic environment dynamics: (s, a) → (s', r, done). */
export function envStep(s: number, a: number): { s2: number; r: number; done: boolean } {
  const r0 = Math.floor(s / COLS), c0 = s % COLS;
  const r1 = r0 + DR[a], c1 = c0 + DC[a];
  let s2 = s;
  if (r1 >= 0 && r1 < ROWS && c1 >= 0 && c1 < COLS && LAYOUT[r1][c1] !== "#") s2 = r1 * COLS + c1;
  const cell = cellAt(s2);
  if (cell === "G") return { s2, r: 1, done: true };
  if (cell === "P") return { s2, r: -1, done: true };
  return { s2, r: 0, done: false };
}

/** Q* by value iteration on the Bellman optimality equation. */
export function optimalQ(gamma: number, iters = 500): number[][] {
  let Q = Array.from({ length: N_STATES }, () => [0, 0, 0, 0]);
  for (let k = 0; k < iters; k++) {
    const next = Q.map((row) => row.slice());
    for (let s = 0; s < N_STATES; s++) {
      if (isWall(s) || isTerminal(s)) continue;
      for (let a = 0; a < 4; a++) {
        const { s2, r, done } = envStep(s, a);
        next[s][a] = r + (done ? 0 : gamma * Math.max(...Q[s2]));
      }
    }
    Q = next;
  }
  return Q;
}

export interface Transition {
  s: number;
  a: number;
  r: number;
  s2: number;
  done: boolean;
  explored: boolean;
  /** max_a' Q(s', a') before the update (0 at terminals) and the γ used */
  maxNext: number;
  gamma: number;
  oldQ: number;
  target: number;
  tdError: number;
  newQ: number;
}

export interface EpisodeRecord {
  start: number;
  steps: number;
  outcome: "goal" | "pit" | "timeout";
  /** discounted return from the start state, Σ_t γ^t r_{t+1} */
  ret: number;
  /** non-terminal states whose greedy action is optimal, measured right after the episode (if qStar is set) */
  optimalStates: number;
}

export class QLearner {
  Q: number[][];
  s = START;
  t = 0;
  episodes: EpisodeRecord[] = [];
  totalSteps = 0;
  last: Transition | null = null;
  seed: number;
  alpha: number;
  gamma: number;
  epsilon: number;
  /** exploring starts: begin every episode in a uniformly random free cell instead of S */
  randomStart: boolean;
  /** optimal action values used to score the greedy policy after every episode */
  qStar: number[][] | null = null;
  /** start state of the current episode */
  start = START;
  private epRet = 0;
  private rand: () => number;

  constructor(seed = 1, alpha = 0.5, gamma = 0.9, epsilon = 0.1, randomStart = false) {
    this.seed = seed;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.randomStart = randomStart;
    this.rand = mulberry32(seed);
    this.s = this.start = this.pickStart();
    this.Q = Array.from({ length: N_STATES }, () => [0, 0, 0, 0]);
  }

  private pickStart(): number {
    if (!this.randomStart) return START;
    const free = FREE_STATES;
    return free[Math.floor(this.rand() * free.length)];
  }

  /** Greedy action with random tie-breaking. */
  greedy(s: number): number {
    const q = this.Q[s];
    const m = Math.max(...q);
    const best = [0, 1, 2, 3].filter((a) => Math.abs(q[a] - m) < 1e-12);
    return best[Math.floor(this.rand() * best.length)];
  }

  /** One environment step + one Q-learning update. */
  step(): Transition {
    const s = this.s;
    const explored = this.rand() < this.epsilon;
    const a = explored ? Math.floor(this.rand() * 4) : this.greedy(s);
    const { s2, r, done } = envStep(s, a);
    const oldQ = this.Q[s][a];
    const maxNext = done ? 0 : Math.max(...this.Q[s2]);
    const target = r + this.gamma * maxNext;
    const tdError = target - oldQ;
    this.Q[s][a] = oldQ + this.alpha * tdError;
    this.epRet += Math.pow(this.gamma, this.t) * r;
    this.t++;
    this.totalSteps++;
    const tr: Transition = { s, a, r, s2, done, explored, maxNext, gamma: this.gamma, oldQ, target, tdError, newQ: this.Q[s][a] };
    this.last = tr;
    if (done || this.t >= MAX_STEPS) {
      this.episodes.push({ start: this.start, steps: this.t, outcome: done ? (r > 0 ? "goal" : "pit") : "timeout", ret: this.epRet, optimalStates: this.qStar ? policyAgreement(this.Q, this.qStar).optimal : Number.NaN });
      this.s = this.start = this.pickStart();
      this.t = 0;
      this.epRet = 0;
    } else {
      this.s = s2;
    }
    return tr;
  }

  /** Run until the current episode ends; returns the number of steps taken. */
  runEpisode(): number {
    const n0 = this.episodes.length;
    let k = 0;
    while (this.episodes.length === n0) {
      this.step();
      k++;
    }
    return k;
  }

  V(s: number) {
    return Math.max(...this.Q[s]);
  }
}

/** Number of non-terminal states whose greedy action under Q is optimal under Q*. */
export function policyAgreement(Q: number[][], Qstar: number[][], tol = 1e-6): { optimal: number; total: number } {
  let optimal = 0, total = 0;
  for (let s = 0; s < N_STATES; s++) {
    if (isWall(s) || isTerminal(s)) continue;
    total++;
    const m = Math.max(...Q[s]);
    const greedySet = [0, 1, 2, 3].filter((a) => Math.abs(Q[s][a] - m) < 1e-12);
    const mStar = Math.max(...Qstar[s]);
    if (greedySet.every((a) => Qstar[s][a] >= mStar - tol)) optimal++;
  }
  return { optimal, total };
}

/** Greedy rollout from START under Q (no exploration); returns the visited path. */
export function greedyPath(Q: number[][], maxLen = 40): { path: number[]; outcome: "goal" | "pit" | "loop" } {
  const path = [START];
  let s = START;
  for (let k = 0; k < maxLen; k++) {
    const q = Q[s];
    const a = q.indexOf(Math.max(...q));
    const { s2, done, r } = envStep(s, a);
    path.push(s2);
    if (done) return { path, outcome: r > 0 ? "goal" : "pit" };
    s = s2;
  }
  return { path, outcome: "loop" };
}
