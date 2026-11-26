"use client";

import { useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  PRESETS,
  dataLoss,
  ellipse,
  isZero,
  ista,
  l1,
  l2,
  lambdaGrid,
  lambdaMaxL1,
  matA,
  solveEN,
  solveL1,
  solveL2,
  type Problem,
  type Vec2,
} from "@/lib/sim/regularization";

const N_LAM = 121;
const S = 400; // plane viewBox size
const LIM = 3; // plane shows [-LIM, LIM]²
const px = (x: number) => ((x + LIM) / (2 * LIM)) * S;
const py = (y: number) => ((LIM - y) / (2 * LIM)) * S;
const pts = (P: Vec2[]) => P.map((p) => `${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ");

/* path chart geometry */
const CW = 400;
const CH = 170;
const CP = { l: 34, r: 8, t: 10, b: 22 };

const L1C = "var(--clay)";
const L2C = "var(--accent)";
const ENC = "var(--gold)";

/** Boundary of the elastic-net ball {α‖w‖₁ + (1−α)/2‖w‖₂² ≤ t}. */
function enBall(t: number, alpha: number, n = 180): Vec2[] {
  const out: Vec2[] = [];
  for (let i = 0; i <= n; i++) {
    const phi = (2 * Math.PI * i) / n;
    const k = Math.abs(Math.cos(phi)) + Math.abs(Math.sin(phi));
    const r = alpha >= 1 ? t / k : (-alpha * k + Math.sqrt(alpha * alpha * k * k + 2 * (1 - alpha) * t)) / (1 - alpha);
    out.push([r * Math.cos(phi), r * Math.sin(phi)]);
  }
  return out;
}

function Coord({ v }: { v: number }) {
  return isZero(v) ? <span className="font-semibold text-clay">0 exact</span> : <span>{fmt(v, 3)}</span>;
}

export default function RegularizationSim() {
  const [presetId, setPresetId] = useState(PRESETS[0].id);
  const base = PRESETS.find((p) => p.id === presetId)!.pr;
  const [c1, setC1] = useState(base.c[0]);
  const [c2, setC2] = useState(base.c[1]);
  const [thDeg, setThDeg] = useState((base.theta * 180) / Math.PI);
  const [kappa, setKappa] = useState(base.a2 / base.a1);
  const [showEN, setShowEN] = useState(false);
  const [alpha, setAlpha] = useState(0.5);

  const s = useStepper(N_LAM, { baseMs: 160 });
  const { setStep } = s;

  function choose(id: string) {
    const pr = PRESETS.find((p) => p.id === id)!.pr;
    setPresetId(id);
    setC1(pr.c[0]);
    setC2(pr.c[1]);
    setThDeg((pr.theta * 180) / Math.PI);
    setKappa(pr.a2 / pr.a1);
  }

  const pr: Problem = useMemo(() => ({ c: [c1, c2], theta: (thDeg * Math.PI) / 180, a1: 1, a2: kappa }), [c1, c2, thDeg, kappa]);
  const A = useMemo(() => matA(pr), [pr]);
  const lmax = useMemo(() => lambdaMaxL1(A, pr.c), [A, pr]);
  const grid = useMemo(() => lambdaGrid(lmax, N_LAM), [lmax]);

  const paths = useMemo(() => {
    const L1: Vec2[] = [];
    const L2: Vec2[] = [];
    const EN: Vec2[] = [];
    for (const lam of grid) {
      L1.push(solveL1(A, pr.c, lam));
      L2.push(solveL2(A, pr.c, lam));
      EN.push(solveEN(A, pr.c, lam, alpha));
    }
    return { L1, L2, EN };
  }, [grid, A, pr, alpha]);

  // λ at which each L1 coordinate first becomes exactly zero (and stays zero)
  const zeroAt = useMemo(() => {
    const res: (number | null)[] = [null, null];
    for (const i of [0, 1]) {
      for (let j = N_LAM - 1; j >= 0; j--) {
        if (!isZero(paths.L1[j][i])) {
          res[i] = j + 1 < N_LAM ? grid[j + 1] : null;
          break;
        }
        if (j === 0) res[i] = 0;
      }
    }
    return res;
  }, [paths, grid]);

  const k = s.step;
  const lam = grid[k];
  const wL1 = paths.L1[k];
  const wL2 = paths.L2[k];
  const wEN = paths.EN[k];
  const check = useMemo(() => ista(A, pr.c, lam), [A, pr, lam]);
  const istaErr = Math.abs(check.w[0] - wL1[0]) + Math.abs(check.w[1] - wL1[1]);

  /* background contours: levels so that rings are evenly spaced in "distance" */
  const rings = useMemo(() => [0.05, 0.2, 0.45, 0.8, 1.25, 1.8, 2.45, 3.2, 4.05, 5, 7.2, 9.8].map((lv) => ellipse(pr, lv)), [pr]);
  const touch = (w: Vec2) => ellipse(pr, Math.max(1e-9, dataLoss(A, pr.c, w)));

  /* path chart */
  const allC = [...paths.L1.flat(), ...paths.L2.flat(), ...(showEN ? paths.EN.flat() : []), 0];
  let cLo = Math.min(...allC);
  let cHi = Math.max(...allC);
  const padC = 0.08 * (cHi - cLo || 1);
  cLo -= padC;
  cHi += padC;
  const cx = (j: number) => CP.l + (j / (N_LAM - 1)) * (CW - CP.l - CP.r);
  const cy = (v: number) => CP.t + ((cHi - v) / (cHi - cLo)) * (CH - CP.t - CP.b);
  const coefLine = (P: Vec2[], i: number) => P.map((w, j) => `${j ? "L" : "M"}${cx(j).toFixed(1)},${cy(w[i]).toFixed(1)}`).join(" ");

  const l1Zero = [isZero(wL1[0]), isZero(wL1[1])];
  const nZero = l1Zero.filter(Boolean).length;
  let title: string;
  let body: string;
  if (k === 0) {
    title = "λ = 0: no penalty yet";
    body = `With no penalty, all three answers sit at the plain loss minimum c = (${fmt(c1, 2)}, ${fmt(c2, 2)}). Press Play to raise λ (the penalty strength) and watch the allowed regions shrink.`;
  } else if (nZero === 0) {
    title = `λ = ${fmt(lam, 3)}: both penalties shrink`;
    body = `A loss contour touches the L1 diamond (radius ${fmt(l1(wL1), 3)}) and the L2 circle (radius ${fmt(l2(wL2), 3)}). Both touch points still have two non-zero coordinates. The first L1 coordinate hits exactly 0 at λ ≈ ${fmt(Math.min(...(zeroAt.filter((z) => z !== null) as number[])), 3)}.`;
  } else if (nZero === 1) {
    const i = l1Zero[0] ? 1 : 2;
    title = `λ = ${fmt(lam, 3)}: L1 has set w${i} to exactly 0`;
    body = `The contour now touches the diamond at a corner on the w${3 - i}-axis. So the L1 answer is sparse (has zeros): w${i} = 0 exactly, not 1e-6. The L2 answer is still (${fmt(wL2[0], 3)}, ${fmt(wL2[1], 3)}). A circle has no corners, so ridge only shrinks weights. ISTA (proximal gradient with soft-thresholding) reaches the same point in ${check.iters} iterations (|Δ| = ${istaErr.toExponential(1)}).`;
  } else {
    title = `λ = ${fmt(lam, 3)} ≥ λ_max: L1 answer is w = 0`;
    body = `Once λ ≥ ‖Ac‖∞ = ${fmt(lmax, 3)}, the subgradient condition |∂L/∂wᵢ(0)| ≤ λ holds for both coordinates. So lasso returns w = 0. Ridge never reaches 0 for any finite λ: (A + λI)⁻¹Ac only gets close to 0 as λ → ∞. Here ‖w_L2‖ = ${fmt(l2(wL2), 3)}.`;
  }

  return (
    <SimShell
      title="L1 vs L2 Regularization"
      subtitle="Oval loss contours in the (w₁, w₂) plane. Each penalised best answer is where a contour touches the L1 diamond or the L2 circle. Play raises λ from 0."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button key={p.id} className={`sim-btn ${p.id === presetId ? "primary" : ""}`} onClick={() => choose(p.id)} aria-pressed={p.id === presetId}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="λ (index)" value={k} min={0} max={N_LAM - 1} step={1} onChange={(v) => setStep(v)} fmt={() => fmt(lam, 3)} />
            <Slider label="centre c₁" value={c1} min={-2.5} max={2.5} step={0.05} onChange={setC1} />
            <Slider label="centre c₂" value={c2} min={-2.5} max={2.5} step={0.05} onChange={setC2} />
            <Slider label="rotation θ" value={thDeg} min={-90} max={90} step={1} onChange={setThDeg} fmt={(v) => `${v.toFixed(0)}°`} />
            <Slider label="elongation a₂/a₁" value={kappa} min={1} max={16} step={0.25} onChange={setKappa} fmt={(v) => v.toFixed(2)} />
            <div className="flex items-center gap-2 text-xs">
              <label className="flex items-center gap-1.5">
                <input type="checkbox" checked={showEN} onChange={(e) => setShowEN(e.target.checked)} aria-label="Show elastic net" />
                elastic net
              </label>
              {showEN && (
                <div className="min-w-0 flex-1">
                  <Slider label="mix α (L1 share)" value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} />
                </div>
              )}
            </div>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted">
                  <th className="py-1 pr-2 font-medium">solution</th>
                  <th className="py-1 pr-2 font-medium">w₁</th>
                  <th className="py-1 pr-2 font-medium">w₂</th>
                  <th className="py-1 font-medium">data loss</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                <tr className="border-t border-border">
                  <td className="py-1 pr-2 font-sans">none (c)</td>
                  <td className="py-1 pr-2">{fmt(c1, 3)}</td>
                  <td className="py-1 pr-2">{fmt(c2, 3)}</td>
                  <td className="py-1">0</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="py-1 pr-2 font-sans text-accent">L2 (ridge)</td>
                  <td className="py-1 pr-2">
                    <Coord v={wL2[0]} />
                  </td>
                  <td className="py-1 pr-2">
                    <Coord v={wL2[1]} />
                  </td>
                  <td className="py-1">{fmt(dataLoss(A, pr.c, wL2), 3)}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="py-1 pr-2 font-sans text-clay">L1 (lasso)</td>
                  <td className="py-1 pr-2">
                    <Coord v={wL1[0]} />
                  </td>
                  <td className="py-1 pr-2">
                    <Coord v={wL1[1]} />
                  </td>
                  <td className="py-1">{fmt(dataLoss(A, pr.c, wL1), 3)}</td>
                </tr>
                {showEN && (
                  <tr className="border-t border-border">
                    <td className="py-1 pr-2 font-sans text-gold">EN α={alpha.toFixed(2)}</td>
                    <td className="py-1 pr-2">
                      <Coord v={wEN[0]} />
                    </td>
                    <td className="py-1 pr-2">
                      <Coord v={wEN[1]} />
                    </td>
                    <td className="py-1">{fmt(dataLoss(A, pr.c, wEN), 3)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">‖w_L1‖₁ (diamond)</div>
              <div className="font-mono text-sm text-clay">{fmt(l1(wL1), 3)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">‖w_L2‖₂ (circle)</div>
              <div className="font-mono text-sm text-accent">{fmt(l2(wL2), 3)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">λ_max = ‖Ac‖∞</div>
              <div className="font-mono text-sm">{fmt(lmax, 3)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">L1 zeros at λ ≈</div>
              <div className="font-mono text-sm">
                {zeroAt.map((z, i) => `w${i + 1}:${z === null ? "—" : fmt(z, 2)}`).join(" ")}
              </div>
            </div>
          </div>
          <Narration title={title}>{body}</Narration>
          <p className="text-xs text-faint">
            Objectives: L2 = L(w) + (λ/2)‖w‖₂², L1 = L(w) + λ‖w‖₁, elastic net = L(w) + λ(α‖w‖₁ + (1−α)/2‖w‖₂²), with L(w) = ½(w−c)ᵀA(w−c). L2 is solved with a direct formula. L1 is solved exactly by trying every sign pattern, then checked with ISTA.
          </p>
        </div>
      }
    >
      <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto h-auto w-full max-w-[520px]" role="img" aria-label={`Loss contours with L1 and L2 constraint regions at lambda ${fmt(lam, 3)}`}>
        <rect x={0} y={0} width={S} height={S} fill="var(--surface)" />
        {[-2, -1, 1, 2].map((v) => (
          <g key={v}>
            <line x1={px(v)} x2={px(v)} y1={0} y2={S} stroke="var(--border)" strokeWidth={0.5} />
            <line x1={0} x2={S} y1={py(v)} y2={py(v)} stroke="var(--border)" strokeWidth={0.5} />
            <text x={px(v)} y={py(0) + 11} fontSize="9" textAnchor="middle" fill="var(--text-faint)">
              {v}
            </text>
            <text x={px(0) - 4} y={py(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-faint)">
              {v}
            </text>
          </g>
        ))}
        <line x1={px(-LIM)} x2={px(LIM)} y1={py(0)} y2={py(0)} stroke="var(--text-muted)" strokeWidth={1} />
        <line x1={px(0)} x2={px(0)} y1={py(-LIM)} y2={py(LIM)} stroke="var(--text-muted)" strokeWidth={1} />
        <text x={S - 6} y={py(0) - 5} fontSize="10" textAnchor="end" fill="var(--text-muted)">
          w₁
        </text>
        <text x={px(0) + 5} y={12} fontSize="10" fill="var(--text-muted)">
          w₂
        </text>
        {rings.map((r, i) => (
          <polyline key={i} points={pts(r)} fill="none" stroke="var(--text-muted)" strokeWidth={0.7} opacity={0.45} />
        ))}
        {/* constraint regions sized so that the penalised optimum lies on their boundary */}
        <circle cx={px(0)} cy={py(0)} r={(l2(wL2) / (2 * LIM)) * S} fill="color-mix(in srgb, var(--accent) 12%, transparent)" stroke={L2C} strokeWidth={1.6} />
        <polygon
          points={pts([
            [l1(wL1), 0],
            [0, l1(wL1)],
            [-l1(wL1), 0],
            [0, -l1(wL1)],
          ])}
          fill="color-mix(in srgb, var(--clay) 12%, transparent)"
          stroke={L1C}
          strokeWidth={1.6}
        />
        {showEN && <polyline points={pts(enBall(alpha * l1(wEN) + ((1 - alpha) / 2) * l2(wEN) ** 2, alpha))} fill="none" stroke={ENC} strokeWidth={1.4} strokeDasharray="5 3" />}
        {/* the contour through each optimum: it is tangent to the region there */}
        {k > 0 && <polyline points={pts(touch(wL2))} fill="none" stroke={L2C} strokeWidth={1} strokeDasharray="4 3" />}
        {k > 0 && <polyline points={pts(touch(wL1))} fill="none" stroke={L1C} strokeWidth={1} strokeDasharray="4 3" />}
        {/* regularisation paths */}
        <polyline points={pts(paths.L2)} fill="none" stroke={L2C} strokeWidth={1.2} opacity={0.6} />
        <polyline points={pts(paths.L1)} fill="none" stroke={L1C} strokeWidth={1.2} opacity={0.6} />
        {showEN && <polyline points={pts(paths.EN)} fill="none" stroke={ENC} strokeWidth={1.2} opacity={0.7} />}
        {/* unregularised optimum */}
        <g stroke="var(--text)" strokeWidth={1.6}>
          <line x1={px(c1) - 5} y1={py(c2) - 5} x2={px(c1) + 5} y2={py(c2) + 5} />
          <line x1={px(c1) - 5} y1={py(c2) + 5} x2={px(c1) + 5} y2={py(c2) - 5} />
        </g>
        <circle cx={px(wL2[0])} cy={py(wL2[1])} r={5} fill={L2C} stroke="var(--surface)" strokeWidth={1.2} />
        <circle cx={px(wL1[0])} cy={py(wL1[1])} r={5} fill={L1C} stroke="var(--surface)" strokeWidth={1.2} />
        {showEN && <circle cx={px(wEN[0])} cy={py(wEN[1])} r={4.5} fill={ENC} stroke="var(--surface)" strokeWidth={1.2} />}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
        <span>× best with no penalty, c</span>
        <span className="text-accent">● L2 best / circle</span>
        <span className="text-clay">● L1 best / diamond</span>
        {showEN && <span className="text-gold">● elastic net</span>}
        <span>thin lines: path as λ grows · dashed: contour through the best point</span>
      </div>

      <figure className="mt-4">
        <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label="Regularisation path: coefficients against lambda for L1 and L2">
          <line x1={CP.l} x2={CW - CP.r} y1={cy(0)} y2={cy(0)} stroke="var(--text-muted)" strokeWidth={0.8} />
          {[cLo + padC, cHi - padC].map((v, i) => (
            <text key={i} x={CP.l - 4} y={cy(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
              {v.toFixed(1)}
            </text>
          ))}
          <text x={CP.l - 4} y={cy(0) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
            0
          </text>
          <text x={CP.l} y={CH - 6} fontSize="9" fill="var(--text-muted)">
            λ = 0
          </text>
          <text x={CW - CP.r} y={CH - 6} fontSize="9" textAnchor="end" fill="var(--text-muted)">
            λ = {fmt(grid[N_LAM - 1], 2)} (log scale after 0)
          </text>
          <line x1={cx(k)} x2={cx(k)} y1={CP.t} y2={CH - CP.b} stroke="var(--gold)" strokeWidth={1.4} />
          {[0, 1].map((i) => (
            <g key={i}>
              <path d={coefLine(paths.L1, i)} fill="none" stroke={L1C} strokeWidth={i === 0 ? 2 : 1.6} opacity={i === 0 ? 1 : 0.7} />
              <path d={coefLine(paths.L2, i)} fill="none" stroke={L2C} strokeWidth={i === 0 ? 2 : 1.6} strokeDasharray="5 3" opacity={i === 0 ? 1 : 0.7} />
              {showEN && <path d={coefLine(paths.EN, i)} fill="none" stroke={ENC} strokeWidth={1.4} strokeDasharray="2 2" />}
            </g>
          ))}
          {[0, 1].map((i) => (
            <g key={`p${i}`}>
              <circle cx={cx(k)} cy={cy(wL1[i])} r={3} fill={L1C} />
              <circle cx={cx(k)} cy={cy(wL2[i])} r={3} fill={L2C} />
            </g>
          ))}
        </svg>
        <figcaption className="mt-1 text-xs text-muted">
          Regularisation path: <span className="text-clay">solid = L1 coefficients</span> (they hit 0 and stay there), <span className="text-accent">dashed = L2</span> (they shrink smoothly but never reach 0). Thicker line = w₁, thinner = w₂.
        </figcaption>
      </figure>
    </SimShell>
  );
}
