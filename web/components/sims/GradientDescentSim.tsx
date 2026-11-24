"use client";

import { useEffect, useId, useMemo, useState, type MouseEvent, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { FUNCTIONS, divergeBound, runGD, stableEta } from "@/lib/sim/gd";

const N_STEPS = 60;
const VW = 560;
const VH = 320;
const M = { l: 44, r: 14, t: 14, b: 30 };
const PW = VW - M.l - M.r;
const PH = VH - M.t - M.b;

export default function GradientDescentSim() {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [fnId, setFnId] = useState(FUNCTIONS[0].id);
  const fn = FUNCTIONS.find((f) => f.id === fnId)!;
  const [x0, setX0] = useState(fn.x0);
  const [eta, setEta] = useState(fn.eta);

  const traj = useMemo(() => runGD(fn, x0, eta, N_STEPS), [fn, x0, eta]);
  const s = useStepper(traj.steps.length + 1, { baseMs: 800 });
  const { setStep } = s;
  useEffect(() => setStep(0), [traj, setStep]);

  const choose = (id: string) => {
    const f = FUNCTIONS.find((q) => q.id === id)!;
    setFnId(id);
    setX0(f.x0);
    setEta(f.eta);
  };

  /* ---------- scales ---------- */
  const [xa, xb] = fn.xRange;
  const [ya, yb] = fn.yRange;
  const sx = (x: number) => M.l + ((x - xa) / (xb - xa)) * PW;
  const sy = (y: number) => M.t + ((yb - y) / (yb - ya)) * PH;
  const inX = (x: number) => x >= xa && x <= xb;
  const clampX = (x: number) => Math.max(xa, Math.min(xb, x));
  const clampY = (y: number) => Math.max(ya - (yb - ya), Math.min(yb + (yb - ya), y));

  const curve = useMemo(() => {
    const n = 320;
    const pts: string[] = [];
    for (let i = 0; i <= n; i++) {
      const x = xa + ((xb - xa) * i) / n;
      pts.push(`${sx(x).toFixed(1)},${sy(clampY(fn.f(x))).toFixed(1)}`);
    }
    return pts.join(" ");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fn]);

  /* ---------- current state ---------- */
  const k = s.step;
  const steps = traj.steps;
  const pending = k < steps.length ? steps[k] : null; // the update about to be applied
  const last = steps[steps.length - 1];
  const xCur = pending ? pending.x : last.xNext;
  const fCur = fn.f(xCur);
  const gCur = fn.df(xCur);
  const visited = [...steps.slice(0, k).map((st) => st.x), xCur];
  const finished = !pending;
  const etaStable = stableEta(fn, xCur);

  // Tangent line y = f(x) + f'(x)(t − x), drawn over a fixed horizontal span.
  const span = (xb - xa) * 0.18;
  const tan = inX(xCur) && Number.isFinite(fCur)
    ? { x1: xCur - span, y1: fCur - gCur * span, x2: xCur + span, y2: fCur + gCur * span }
    : null;

  function onPick(e: MouseEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * VW;
    const x = xa + ((px - M.l) / PW) * (xb - xa);
    if (inX(x)) setX0(+x.toFixed(2));
  }

  const history = steps.slice(Math.max(0, k - 6), k);
  const xTicks = useMemo(() => {
    const out: number[] = [];
    const stepT = (xb - xa) > 5 ? 1 : 0.5;
    for (let t = Math.ceil(xa / stepT) * stepT; t <= xb + 1e-9; t += stepT) out.push(+t.toFixed(2));
    return out;
  }, [xa, xb]);

  let story: ReactNode;
  if (pending) {
    const dir = pending.g > 0 ? "uphill to the right, so the step goes left" : pending.g < 0 ? "uphill to the left, so the step goes right" : "flat, so the step is zero";
    const better = pending.fNext < pending.fx;
    story = (
      <>
        <p>
          Step {k + 1}: at x = {fmt(pending.x)}, the slope is f&apos;(x) = {fmt(pending.g)} — {dir}.
        </p>
        <p className="mt-1 font-mono text-xs text-text">
          x ← {fmt(pending.x)} − {fmt(eta)} × ({fmt(pending.g)}) = {fmt(pending.xNext)}
        </p>
        <p className="mt-1">
          f goes from {fmt(pending.fx)} to {fmt(pending.fNext)} ({better ? "decrease" : "increase — overshoot"}).
        </p>
      </>
    );
  } else if (traj.diverged) {
    story = <p>Diverged after {steps.length} steps: |x| exceeded {divergeBound(fn)} and is still growing — each step overshot further than the last. Reduce η below the stability limit.</p>;
  } else {
    story = (
      <p>
        Finished {steps.length} steps at x = {fmt(xCur, 4)}, f(x) = {fmt(fCur, 4)}, |f&apos;(x)| = {fmt(Math.abs(gCur), 4)}.
      </p>
    );
  }

  return (
    <SimShell
      title="Gradient Descent in 1-D"
      subtitle="x ← x − η·f′(x). Click the plot to choose a starting point."
      controls={
        <div className="space-y-2.5">
          <Controls s={s} />
          <div className="flex flex-wrap gap-1.5">
            {FUNCTIONS.map((f) => (
              <button key={f.id} className={`sim-btn ${f.id === fnId ? "primary" : ""}`} onClick={() => choose(f.id)} aria-pressed={f.id === fnId}>
                {f.label}
              </button>
            ))}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="start x₀" value={x0} min={xa} max={xb} step={0.01} onChange={setX0} />
            <Slider label="learning rate η" value={eta} min={0.001} max={fn.etaMax} step={0.001} onChange={setEta} fmt={(v) => v.toFixed(3)} />
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={finished ? (traj.diverged ? "Diverged" : "Done") : `Update ${k + 1}`}>{story}</Narration>
          <div className="rounded-lg border border-border p-3 text-xs">
            <div className="mb-1 font-medium">{fn.formula}</div>
            <p className="text-muted">{fn.note}</p>
            {Number.isFinite(fCur) && (
              <p className="mt-1.5 text-muted">
                Local curvature f&apos;&apos;(x) = {fmt(fn.d2f(xCur))}.{" "}
                {Number.isFinite(etaStable) ? (
                  <>
                    A quadratic with this curvature is stable for η &lt; 2/f&apos;&apos; = {fmt(etaStable)}; your η = {fmt(eta)} is{" "}
                    <span className={eta < etaStable ? "text-accent" : "text-clay"}>{eta < etaStable ? "below" : "above"}</span> it.
                  </>
                ) : (
                  "Curvature ≤ 0 here: the quadratic model gives no stability limit."
                )}
              </p>
            )}
          </div>
        </div>
      }
    >
      <svg viewBox={`0 0 ${VW} ${VH}`} className="h-auto w-full cursor-crosshair" role="img" aria-label={`Plot of ${fn.formula} with the gradient descent trajectory`} onClick={onPick}>
        <defs>
          <clipPath id={`gd-clip-${uid}`}>
            <rect x={M.l} y={M.t} width={PW} height={PH} />
          </clipPath>
          <marker id={`gd-arrow-${uid}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--clay)" />
          </marker>
        </defs>
        {/* axes */}
        <rect x={M.l} y={M.t} width={PW} height={PH} fill="none" stroke="var(--border)" />
        {ya < 0 && yb > 0 && <line x1={M.l} x2={M.l + PW} y1={sy(0)} y2={sy(0)} stroke="var(--border)" />}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={M.t + PH} y2={M.t + PH + 4} stroke="var(--text-faint)" />
            <text x={sx(t)} y={VH - 10} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
              {t}
            </text>
          </g>
        ))}
        {[ya, (ya + yb) / 2, yb].map((t) => (
          <text key={t} x={M.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="var(--text-muted)">
            {t.toFixed(1)}
          </text>
        ))}
        <text x={M.l + PW - 4} y={M.t + PH - 6} textAnchor="end" fontSize={11} fill="var(--text-faint)">
          x
        </text>

        <g clipPath={`url(#gd-clip-${uid})`}>
          <polyline points={curve} fill="none" stroke="var(--accent)" strokeWidth={2} />
          {/* trajectory: path through visited points on the curve */}
          <polyline
            points={visited.filter(inX).map((x) => `${sx(x).toFixed(1)},${sy(clampY(fn.f(x))).toFixed(1)}`).join(" ")}
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          {visited.slice(0, -1).filter(inX).map((x, i) => (
            <circle key={i} cx={sx(x)} cy={sy(clampY(fn.f(x)))} r={2.6} fill="var(--text-muted)" />
          ))}
          {tan && <line x1={sx(tan.x1)} y1={sy(tan.y1)} x2={sx(tan.x2)} y2={sy(tan.y2)} stroke="var(--gold)" strokeWidth={1.6} />}
          {pending && inX(pending.x) && (
            <>
              <line x1={sx(pending.x)} x2={sx(pending.x)} y1={sy(clampY(pending.fx))} y2={M.t + PH} stroke="var(--gold)" strokeDasharray="2 3" />
              {inX(pending.xNext) && (
                <>
                  <line x1={sx(pending.xNext)} x2={sx(pending.xNext)} y1={sy(clampY(pending.fNext))} y2={M.t + PH} stroke="var(--clay)" strokeDasharray="2 3" />
                  <circle cx={sx(pending.xNext)} cy={sy(clampY(pending.fNext))} r={4.5} fill="none" stroke="var(--clay)" strokeWidth={1.5} />
                </>
              )}
              {Math.abs(sx(clampX(pending.xNext)) - sx(pending.x)) > 4 && (
                <line x1={sx(pending.x)} x2={sx(clampX(pending.xNext))} y1={M.t + PH - 8} y2={M.t + PH - 8} stroke="var(--clay)" strokeWidth={1.8} markerEnd={`url(#gd-arrow-${uid})`} />
              )}
            </>
          )}
          {inX(xCur) && Number.isFinite(fCur) && <circle cx={sx(xCur)} cy={sy(clampY(fCur))} r={5.5} fill="var(--gold)" stroke="var(--surface)" strokeWidth={1.5} />}
        </g>
        {pending && inX(pending.x) && (
          <text x={Math.min(sx(pending.x) + 6, M.l + PW - 110)} y={M.t + 14} fontSize={11} fill="var(--text)">
            f′(x) = {fmt(pending.g, 2)}, step = {fmt(-pending.delta, 2)}
          </text>
        )}
        {!inX(xCur) && (
          <text x={xCur > xb ? M.l + PW - 6 : M.l + 6} y={M.t + PH / 2} textAnchor={xCur > xb ? "end" : "start"} fontSize={11} fill="var(--clay)">
            {xCur > xb ? "→" : "←"} off chart: x = {fmt(xCur, 2)}
          </text>
        )}
      </svg>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-muted">
              {["k", "x_k", "f(x_k)", "f′(x_k)", "η·f′(x_k)", "x_k+1"].map((h) => (
                <th key={h} className="py-1 pr-3 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono">
            {history.length === 0 && (
              <tr>
                <td colSpan={6} className="py-1 text-faint">
                  No updates yet — press Step.
                </td>
              </tr>
            )}
            {history.map((h) => (
              <tr key={h.k} className="border-t border-border">
                <td className="py-1 pr-3">{h.k}</td>
                <td className="py-1 pr-3">{fmt(h.x, 4)}</td>
                <td className="py-1 pr-3">{fmt(h.fx, 4)}</td>
                <td className="py-1 pr-3">{fmt(h.g, 4)}</td>
                <td className="py-1 pr-3">{fmt(h.delta, 4)}</td>
                <td className="py-1 pr-3">{fmt(h.xNext, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SimShell>
  );
}
