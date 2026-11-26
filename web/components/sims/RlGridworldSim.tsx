"use client";

import { useMemo, useState } from "react";
import { FastForward, Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper, type Stepper } from "./SimShell";
import { ARROWS, COLS, FREE_STATES, N_STATES, QLearner, ROWS, START, cellAt, greedyPath, isTerminal, isWall, optimalQ, policyAgreement } from "@/lib/sim/rl-gridworld";

const CELL = 50;
const GW = COLS * CELL;
const GH = ROWS * CELL;
const CW = 300;
const CH = 104;
const CP = { l: 30, r: 8, t: 8, b: 18 };
const DEFAULT_SEED = 7;

const rc = (s: number) => `(${Math.floor(s / COLS)},${s % COLS})`;

function movingAverage(v: number[], w: number) {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i];
    if (i >= w) sum -= v[i - w];
    out.push(sum / Math.min(i + 1, w));
  }
  return out;
}

export default function RlGridworldSim() {
  const [alpha, setAlpha] = useState(0.5);
  const [gamma, setGamma] = useState(0.9);
  const [epsilon, setEpsilon] = useState(0.3);
  const [randomStart, setRandomStart] = useState(true);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [unit, setUnit] = useState<"step" | "episode">("step");
  const [view, setView] = useState<"V" | "Q">("V");
  const [, setVersion] = useState(0);
  const bump = () => setVersion((v) => v + 1);

  const qStar = useMemo(() => optimalQ(gamma), [gamma]);
  const make = (sd: number, rs: boolean) => {
    const learner = new QLearner(sd, alpha, gamma, epsilon, rs);
    learner.qStar = qStar;
    return learner;
  };
  // the learner is a mutable object kept in state; `version` forces a re-render after it changes
  const [L, setL] = useState<QLearner>(() => make(DEFAULT_SEED, true));

  const base = useStepper(Infinity, {
    baseMs: 500,
    onTick: () => {
      if (unit === "step") L.step();
      else L.runEpisode();
      bump();
    },
  });
  const rebuild = (sd: number, rs: boolean) => setL(make(sd, rs));
  const s: Stepper = { ...base, reset: () => { base.reset(); rebuild(seed, randomStart); } };
  // hyper-parameters apply from the next update on
  const changeAlpha = (v: number) => { setAlpha(v); L.alpha = v; };
  const changeEps = (v: number) => { setEpsilon(v); L.epsilon = v; };
  const changeGamma = (v: number) => { setGamma(v); L.gamma = v; L.qStar = optimalQ(v); };

  const train = (n: number) => {
    for (let i = 0; i < n; i++) L.runEpisode();
    bump();
  };

  /* ---------- derived numbers ---------- */
  const agree = policyAgreement(L.Q, qStar);
  const gp = greedyPath(L.Q);
  const eps = L.episodes;
  const rets = eps.map((e) => e.ret);
  const ma = movingAverage(rets, 20);
  const vStar = (st: number) => Math.max(...qStar[st]);
  const optRet = randomStart ? FREE_STATES.reduce((a, st) => a + vStar(st), 0) / FREE_STATES.length : vStar(START);
  const tr = L.last;
  const nEp = eps.length;
  const goals = eps.slice(-50).filter((e) => e.outcome === "goal").length;
  const lastN = Math.min(50, nEp);

  /* ---------- chart ---------- */
  const xMax = Math.max(20, nEp);
  const cx = (i: number) => CP.l + (i / xMax) * (CW - CP.l - CP.r);
  const cy = (v: number) => CP.t + ((1 - v) / 2) * (CH - CP.t - CP.b); // v in [-1, 1]
  const lineMA = ma.map((v, i) => `${i ? "L" : "M"}${cx(i + 1).toFixed(1)},${cy(v).toFixed(1)}`).join(" ");
  const lineAg = eps.map((e, i) => `${i ? "L" : "M"}${cx(i + 1).toFixed(1)},${cy((2 * e.optimalStates) / agree.total - 1).toFixed(1)}`).join(" ");

  /* ---------- cell rendering ---------- */
  const cells = [];
  for (let st = 0; st < N_STATES; st++) {
    const r = Math.floor(st / COLS), c = st % COLS;
    const x = c * CELL, y = r * CELL;
    const cell = cellAt(st);
    if (isWall(st)) {
      cells.push(<rect key={st} x={x} y={y} width={CELL} height={CELL} fill="var(--text-muted)" opacity={0.55} stroke="var(--border)" />);
      continue;
    }
    if (isTerminal(st)) {
      const goal = cell === "G";
      cells.push(
        <g key={st}>
          <rect x={x} y={y} width={CELL} height={CELL} fill={goal ? "color-mix(in srgb, var(--accent) 70%, var(--surface))" : "color-mix(in srgb, var(--clay) 70%, var(--surface))"} stroke="var(--border)" />
          <text x={x + CELL / 2} y={y + CELL / 2 - 2} fontSize="13" fontWeight={700} textAnchor="middle" fill="var(--text)">
            {goal ? "G" : "P"}
          </text>
          <text x={x + CELL / 2} y={y + CELL / 2 + 12} fontSize="9" textAnchor="middle" fill="var(--text)">
            {goal ? "+1" : "−1"}
          </text>
        </g>,
      );
      continue;
    }
    const q = L.Q[st];
    const v = Math.max(...q);
    const best = q.indexOf(v);
    const visitedAny = q.some((z) => z !== 0);
    const isOpt = qStar[st][best] >= vStar(st) - 1e-6 && visitedAny;
    const cxm = x + CELL / 2, cym = y + CELL / 2;
    const tri = [
      `${x},${y} ${x + CELL},${y} ${cxm},${cym}`,
      `${x + CELL},${y} ${x + CELL},${y + CELL} ${cxm},${cym}`,
      `${x},${y + CELL} ${x + CELL},${y + CELL} ${cxm},${cym}`,
      `${x},${y} ${x},${y + CELL} ${cxm},${cym}`,
    ];
    const qPos: [number, number][] = [
      [cxm, y + 11],
      [x + CELL - 11, cym + 3],
      [cxm, y + CELL - 5],
      [x + 11, cym + 3],
    ];
    const updated = tr && tr.s === st;
    cells.push(
      <g key={st}>
        <rect x={x} y={y} width={CELL} height={CELL} fill="var(--surface)" stroke="var(--border)" />
        {view === "V" ? (
          <>
            <rect x={x} y={y} width={CELL} height={CELL} fill={signedColor(v, 1)} stroke="var(--border)" />
            <text x={cxm} y={y + 14} fontSize="9" textAnchor="middle" fill="var(--text)">
              {fmt(v, 2)}
            </text>
            {visitedAny && (
              <text x={cxm} y={cym + 14} fontSize="17" textAnchor="middle" fill={isOpt ? "var(--text)" : "var(--clay)"} fontWeight={700}>
                {ARROWS[best]}
              </text>
            )}
          </>
        ) : (
          tri.map((pts, a) => (
            <g key={a}>
              <polygon points={pts} fill={signedColor(q[a], 1)} stroke={updated && tr.a === a ? "var(--gold)" : "var(--border)"} strokeWidth={updated && tr.a === a ? 2.5 : 0.5} />
              <text x={qPos[a][0]} y={qPos[a][1]} fontSize="7.5" textAnchor="middle" fill="var(--text)">
                {fmt(q[a], 2)}
              </text>
            </g>
          ))
        )}
        {view === "V" && updated && <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill="none" stroke="var(--gold)" strokeWidth={2} />}
        {cell === "S" && (
          <text x={x + 4} y={y + CELL - 4} fontSize="8" fill="var(--text-muted)">
            S
          </text>
        )}
      </g>,
    );
  }
  const ar = Math.floor(L.s / COLS), ac = L.s % COLS;
  const pathPts = gp.path.map((st) => `${(st % COLS) * CELL + CELL / 2},${Math.floor(st / COLS) * CELL + CELL / 2}`).join(" ");

  /* ---------- narration ---------- */
  let title: string;
  let story: string;
  if (!tr) {
    title = "Before training: Q = 0 everywhere";
    story = `The agent knows nothing yet. Every action value (Q) is 0, so ties are broken at random. ${randomStart ? "Each episode (one trip) starts in a random free cell (exploring starts)" : "Each episode (one trip) starts at S"}. Press Step to take one action and do one Q-learning update. Or switch to "episode" to run whole trips per tick.`;
  } else {
    title = `Episode ${nEp + (L.t > 0 ? 1 : 0)} · update of Q${rc(tr.s)}, ${ARROWS[tr.a]}`;
    story =
      `In ${rc(tr.s)} the agent took ${ARROWS[tr.a]} (${tr.explored ? `random, probability ε = ${fmt(epsilon, 2)}` : "greedy, the best-looking move"}) and landed in ${rc(tr.s2)} with reward ${tr.r}. ` +
      (tr.done
        ? `That cell ends the episode, so the target is just r = ${fmt(tr.target, 3)}. `
        : `Target = r + γ·maxₐ′Q(s′,a′) = ${tr.r} + ${fmt(tr.gamma, 2)}×${fmt(tr.maxNext, 3)} = ${fmt(tr.target, 3)}. `) +
      `TD error (target minus old guess) = ${fmt(tr.target, 3)} − ${fmt(tr.oldQ, 3)} = ${fmt(tr.tdError, 3)}. Q ← ${fmt(tr.oldQ, 3)} + ${fmt(alpha, 2)}×${fmt(tr.tdError, 3)} = ${fmt(tr.newQ, 3)}.`;
    if (agree.optimal === agree.total) story += ` The greedy policy is now the best possible in all ${agree.total} states. Some Q-values are still off, but their order is right, and that is what counts.`;
    else if (nEp > 30) story += ` ${agree.total - agree.optimal} state(s) still have a wrong greedy arrow (red). ${randomStart ? "Keep training. Rarely chosen actions only get fixed when exploring tries them." : "With a fixed start, cells far from the S→G path are almost never visited. Try random starts or a bigger ε."}`;
  }

  return (
    <SimShell
      title="Q-Learning in a Gridworld"
      subtitle="The agent learns a table of action values (Q) by trial and error. It explores at random with chance ε, otherwise picks the best-looking move. G gives +1, P gives −1, every other move 0."
      controls={
        <div className="space-y-2.5">
          <Controls
            s={s}
            canPrev={false}
            extra={
              <>
                <button className="sim-btn" onClick={() => train(100)} aria-label="Train 100 more episodes">
                  <FastForward size={14} /> +100 episodes
                </button>
                <button className="sim-btn" onClick={() => { const sd = seed + 1; setSeed(sd); base.reset(); rebuild(sd, randomStart); }} aria-label="Restart with a new random seed">
                  <Shuffle size={14} /> New seed
                </button>
              </>
            }
          />
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted">per tick</span>
            {(["step", "episode"] as const).map((u) => (
              <button key={u} className={`sim-btn ${unit === u ? "primary" : ""}`} onClick={() => setUnit(u)} aria-pressed={unit === u}>
                {u}
              </button>
            ))}
            <span className="ml-2 text-xs text-muted">show</span>
            {(["V", "Q"] as const).map((v) => (
              <button key={v} className={`sim-btn ${view === v ? "primary" : ""}`} onClick={() => setView(v)} aria-pressed={view === v}>
                {v === "V" ? "V + policy" : "Q(s,a)"}
              </button>
            ))}
            <span className="ml-2 text-xs text-muted">start</span>
            {([true, false] as const).map((rs) => (
              <button key={String(rs)} className={`sim-btn ${randomStart === rs ? "primary" : ""}`} onClick={() => { setRandomStart(rs); base.reset(); rebuild(seed, rs); }} aria-pressed={randomStart === rs}>
                {rs ? "random" : "S"}
              </button>
            ))}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-3">
            <Slider label="α (step size)" value={alpha} min={0.05} max={1} step={0.05} onChange={changeAlpha} />
            <Slider label="γ (discount)" value={gamma} min={0.5} max={0.99} step={0.01} onChange={changeGamma} />
            <Slider label="ε (explore)" value={epsilon} min={0} max={1} step={0.05} onChange={changeEps} />
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div className="rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">
            <div>
              episodes {nEp} · steps {L.totalSteps}
            </div>
            <div>
              last {lastN} episodes: {goals} reached G
            </div>
            <div className={agree.optimal === agree.total ? "text-accent" : ""}>
              greedy action is best in {agree.optimal}/{agree.total} states
            </div>
            <div>
              greedy path from S: {gp.outcome === "goal" ? `reaches G in ${gp.path.length - 1} moves` : gp.outcome === "pit" ? "falls into P" : "loops"} (best: 9)
            </div>
            <div className="text-muted">
              V*(S) = γ⁸ = {fmt(vStar(START), 3)} · V(S) = {fmt(L.V(START), 3)}
            </div>
          </div>
          <Narration title={title}>{story}</Narration>
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-[1.1fr_1fr]">
        <figure className="min-w-0">
          <svg viewBox={`-2 -2 ${GW + 4} ${GH + 4}`} className="h-auto w-full" role="img" aria-label={`Gridworld with ${view === "V" ? "state values and greedy policy arrows" : "the four action values per cell"}; the agent is at row ${ar}, column ${ac}`}>
            {cells}
            {gp.outcome === "goal" && <polyline points={pathPts} fill="none" stroke="var(--gold)" strokeWidth={2} strokeDasharray="4 3" opacity={0.8} />}
            <circle cx={ac * CELL + CELL / 2} cy={ar * CELL + CELL / 2} r={9} fill="var(--gold)" stroke="var(--text)" strokeWidth={1.5} />
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
            <span className="text-gold">● agent</span>
            <span className="text-gold">- - greedy path from S</span>
            <span>□ gold box: cell just updated</span>
            {view === "V" && <span className="text-clay">red arrow: not the best move (checked by value iteration)</span>}
          </figcaption>
        </figure>
        <figure className="min-w-0">
          <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label="Episode return moving average and fraction of states with an optimal greedy action, per episode">
            {[-1, 0, 1].map((v) => (
              <g key={v}>
                <line x1={CP.l} x2={CW - CP.r} y1={cy(v)} y2={cy(v)} stroke="var(--border)" strokeWidth={0.6} />
                <text x={CP.l - 4} y={cy(v) + 3} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
                  {v}
                </text>
              </g>
            ))}
            <line x1={CP.l} x2={CW - CP.r} y1={cy(optRet)} y2={cy(optRet)} stroke="var(--accent)" strokeDasharray="3 3" strokeWidth={0.9} />
            <text x={CW - CP.r} y={CH - 5} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
              episode {nEp}
            </text>
            {nEp > 0 && <path d={lineAg} fill="none" stroke="var(--text-muted)" strokeWidth={1.2} />}
            {nEp > 0 && <path d={lineMA} fill="none" stroke="var(--accent)" strokeWidth={1.8} />}
          </svg>
          <figcaption className="text-xs text-muted">
            <span className="text-accent">discounted return</span> (20-episode average; dashed: best possible {randomStart ? "average over random starts" : "from S"} = {fmt(optRet, 3)}) and <span>grey: share of states where the greedy action is best</span> (−1 = 0 %, +1 = 100 %).
          </figcaption>
          <div className="mt-2 rounded-md border border-border p-2 text-xs text-muted">
            <div className="mb-1 font-medium text-[var(--text)]">Update done at every step</div>
            <div className="font-mono text-[11px]">Q(s,a) ← Q(s,a) + α[r + γ·maxₐ′ Q(s′,a′) − Q(s,a)]</div>
            <div className="mt-1">How it acts: a random action with chance ε, otherwise greedy. Q-learning is off-policy: it learns the values of the greedy policy while it actually acts ε-greedily.</div>
          </div>
        </figure>
      </div>
    </SimShell>
  );
}
