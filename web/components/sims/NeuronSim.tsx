"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { GATES, activate, boundarySegment, gateData, perceptronTrace, positiveRegion, predict, weightedSum, type ActName, type GateName } from "@/lib/sim/neuron";

const LO = -0.5;
const HI = 1.5;
const SIZE = 320;
const PAD = 28;
const MAX_EPOCHS = 30;

const ACTS: { id: ActName; label: string }[] = [
  { id: "step", label: "Step" },
  { id: "sigmoid", label: "Sigmoid" },
  { id: "tanh", label: "Tanh" },
  { id: "relu", label: "ReLU" },
  { id: "linear", label: "Linear" },
];

const sx = (x: number) => PAD + ((x - LO) / (HI - LO)) * (SIZE - 2 * PAD);
const sy = (y: number) => SIZE - PAD - ((y - LO) / (HI - LO)) * (SIZE - 2 * PAD);
/** compact number formatting for weights (they are usually short decimals) */
const n2 = (v: number) => {
  const r = Math.round(v * 1000) / 1000;
  return Object.is(r, -0) ? "0" : String(r);
};
const paren = (v: number) => (v < 0 ? `(${n2(v)})` : n2(v));

export default function NeuronSim() {
  const [gate, setGate] = useState<GateName>("AND");
  const [w1, setW1] = useState(0);
  const [w2, setW2] = useState(0);
  const [b0, setB0] = useState(0);
  const [lr, setLr] = useState(1);
  const [act, setAct] = useState<ActName>("step");

  const data = useMemo(() => gateData(gate), [gate]);
  const trace = useMemo(() => perceptronTrace(data, [w1, w2], b0, lr, MAX_EPOCHS), [data, w1, w2, b0, lr]);

  // step 0 = initial weights; step k = after the k-th single-sample update
  const s = useStepper(trace.steps.length + 1, { baseMs: 1000 });
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [trace, reset]);

  const k = Math.min(s.step, trace.steps.length);
  const last = k > 0 ? trace.steps[k - 1] : null;
  const w: [number, number] = last ? last.wAfter : [w1, w2];
  const b = last ? last.bAfter : b0;
  const finished = k === trace.steps.length;

  // the sample whose arithmetic is displayed: the update just made, or (at step 0) the first sample
  const shown = last ?? null;
  const previewX = data[0].x;
  const zPreview = weightedSum([w1, w2], b0, previewX);
  const curX = shown ? shown.x : previewX;
  const curZ = shown ? shown.z : zPreview;
  const curA = activate(act, curZ);

  const preds = data.map((d) => predict(weightedSum(w, b, d.x)));
  const wrong = preds.filter((p, i) => p !== data[i].y).length;

  const seg = boundarySegment(w, b, LO, HI);
  const region = positiveRegion(w, b, LO, HI);

  // weight-vector arrow: from the point on the boundary closest to the square centre, along w
  let arrow: { x0: number; y0: number; x1: number; y1: number } | null = null;
  const wn = Math.hypot(w[0], w[1]);
  if (wn > 1e-9) {
    const c: [number, number] = [0.5, 0.5];
    const t = (w[0] * c[0] + w[1] * c[1] + b) / (wn * wn);
    const p: [number, number] = [c[0] - t * w[0], c[1] - t * w[1]];
    if (p[0] >= LO && p[0] <= HI && p[1] >= LO && p[1] <= HI) {
      const L = 0.32;
      arrow = { x0: p[0], y0: p[1], x1: p[0] + (L * w[0]) / wn, y1: p[1] + (L * w[1]) / wn };
    }
  }

  let narrTitle: string;
  let narr: string;
  if (!last) {
    narrTitle = "Ready";
    narr = `Starting weights w = (${n2(w1)}, ${n2(w2)}), b = ${n2(b0)}. The neuron says 1 wherever z = w1·x1 + w2·x2 + b ≥ 0 (the shaded side of the line). Press Step to show it the first sample (0, 0) and apply the perceptron rule.`;
  } else if (last.err === 0) {
    narrTitle = `Epoch ${last.epoch}, sample ${last.index + 1}/4: correct`;
    narr = `Input (${last.x.join(", ")}) gives z = ${n2(last.z)}, so ŷ = ${last.yhat}, which matches the target y = ${last.y}. The error is 0, so Δw = (0, 0) and Δb = 0. The perceptron only learns from mistakes.`;
  } else {
    narrTitle = `Epoch ${last.epoch}, sample ${last.index + 1}/4: mistake`;
    const dir = last.err > 0 ? "pushed up (the target was 1 but z was negative)" : "pushed down (the target was 0 but z was 0 or more)";
    narr = `Input (${last.x.join(", ")}) gives z = ${n2(last.z)}, so ŷ = ${last.yhat}, but y = ${last.y}. Error e = ${last.err}. The fix adds η·e·x = (${n2(last.dw[0])}, ${n2(last.dw[1])}) to w and η·e = ${n2(last.db)} to b. So z for this input gets ${dir}. The line moves.`;
  }
  if (finished) {
    if (trace.convergedEpoch !== null) {
      narrTitle = `Done: learned it in epoch ${trace.convergedEpoch}`;
      narr = `Epoch ${trace.convergedEpoch} (one pass over all 4 samples) had no mistakes, so the weights stop changing: w = (${n2(w[0])}, ${n2(w[1])}), b = ${n2(b)}. A straight line can split ${gate}, and the perceptron convergence theorem promises this always happens then. Mistakes per epoch: ${trace.errorsPerEpoch.join(", ")}.`;
    } else {
      narrTitle = `Still not learned after ${MAX_EPOCHS} epochs`;
      narr = `Every epoch still has mistakes (per epoch: ${trace.errorsPerEpoch.slice(0, 8).join(", ")}${trace.errorsPerEpoch.length > 8 ? ", …" : ""}). ${gate === "XOR" ? "No straight line puts (0,1) and (1,0) on one side and (0,0) and (1,1) on the other. So the weights go round in circles forever. You need a hidden layer." : "Try more epochs or different starting weights."}`;
    }
  }

  const accent = "var(--accent)";
  const clay = "var(--clay)";

  return (
    <SimShell
      title="Artificial Neuron & Perceptron"
      subtitle="z = w1·x1 + w2·x2 + b. Predict 1 when z ≥ 0. After every mistake, the perceptron rule nudges w and b."
      controls={
        <Controls
          s={s}
          extra={
            <div className="flex flex-wrap items-center gap-1.5">
              {GATES.map((g) => (
                <button key={g} className={`sim-btn ${g === gate ? "primary" : ""}`} onClick={() => setGate(g)} aria-pressed={g === gate} aria-label={`Dataset ${g}`}>
                  {g}
                </button>
              ))}
            </div>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <div className="space-y-1.5">
            <div className="text-xs font-medium uppercase tracking-wide text-faint">Starting parameters</div>
            <Slider label="w1" value={w1} min={-3} max={3} step={0.1} onChange={setW1} fmt={(v) => v.toFixed(1)} />
            <Slider label="w2" value={w2} min={-3} max={3} step={0.1} onChange={setW2} fmt={(v) => v.toFixed(1)} />
            <Slider label="bias b" value={b0} min={-3} max={3} step={0.1} onChange={setB0} fmt={(v) => v.toFixed(1)} />
            <Slider label="learning rate η" value={lr} min={0.1} max={1} step={0.1} onChange={setLr} fmt={(v) => v.toFixed(1)} />
            <p className="text-xs text-faint">Moving a slider restarts training from these values.</p>
          </div>
          <label className="flex items-center gap-2 text-xs">
            <span className="w-24 shrink-0 text-muted">activation f</span>
            <select className="field min-w-0 flex-1" value={act} onChange={(e) => setAct(e.target.value as ActName)} aria-label="Activation function">
              {ACTS.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </select>
          </label>
          <div className="rounded-lg border border-border p-2.5 font-mono text-xs leading-relaxed">
            <div className="mb-1 font-sans text-faint">{shown ? "Sample just shown" : "First sample"}: x = ({curX.join(", ")})</div>
            <div>
              z = {paren(shown ? shown.wBefore[0] : w1)}·{curX[0]} + {paren(shown ? shown.wBefore[1] : w2)}·{curX[1]} + {paren(shown ? shown.bBefore : b0)} = <span className="text-gold">{n2(curZ)}</span>
            </div>
            <div>
              a = {act}(z) = {fmt(curA)}
            </div>
            <div>ŷ = [z ≥ 0] = {predict(curZ)}</div>
            {shown && (
              <>
                <div>
                  y = {shown.y}, e = y − ŷ = <span className={shown.err === 0 ? "" : "text-clay"}>{shown.err}</span>
                </div>
                <div>
                  Δw = η·e·x = ({n2(shown.dw[0])}, {n2(shown.dw[1])})
                </div>
                <div>Δb = η·e = {n2(shown.db)}</div>
                <div className="mt-1 border-t border-border pt-1">
                  w ← ({n2(shown.wAfter[0])}, {n2(shown.wAfter[1])}), b ← {n2(shown.bAfter)}
                </div>
              </>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">epoch</div>
              <div className="font-mono text-sm">{last ? last.epoch : 0}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">mistakes this epoch</div>
              <div className="font-mono text-sm">{last ? last.epochErrors : 0}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">wrong now</div>
              <div className="font-mono text-sm">{wrong}/4</div>
            </div>
          </div>
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)]">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="h-auto w-full" role="img" aria-label={`Input plane for ${gate}: four points, the decision line and the predicted-1 region`}>
          <defs>
            <marker id="neuron-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--gold)" />
            </marker>
          </defs>
          <rect x={sx(LO)} y={sy(HI)} width={sx(HI) - sx(LO)} height={sy(LO) - sy(HI)} fill="var(--surface)" stroke="var(--border)" />
          {region.length >= 3 && <polygon points={region.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} fill="color-mix(in srgb, var(--accent) 14%, transparent)" />}
          {[0, 1].map((v) => (
            <g key={v}>
              <line x1={sx(v)} x2={sx(v)} y1={sy(LO)} y2={sy(HI)} stroke="var(--border)" strokeDasharray="2 3" />
              <line x1={sx(LO)} x2={sx(HI)} y1={sy(v)} y2={sy(v)} stroke="var(--border)" strokeDasharray="2 3" />
              <text x={sx(v)} y={SIZE - 10} fontSize="10" textAnchor="middle" fill="var(--text-muted)">
                {v}
              </text>
              <text x={12} y={sy(v) + 3} fontSize="10" textAnchor="middle" fill="var(--text-muted)">
                {v}
              </text>
            </g>
          ))}
          <text x={sx(HI) - 4} y={sy(LO) - 6} fontSize="10" textAnchor="end" fill="var(--text-muted)">
            x1
          </text>
          <text x={sx(LO) + 4} y={sy(HI) + 12} fontSize="10" fill="var(--text-muted)">
            x2
          </text>
          {seg && <line x1={sx(seg[0][0])} y1={sy(seg[0][1])} x2={sx(seg[1][0])} y2={sy(seg[1][1])} stroke="var(--text)" strokeWidth={2} />}
          {arrow && <line x1={sx(arrow.x0)} y1={sy(arrow.y0)} x2={sx(arrow.x1)} y2={sy(arrow.y1)} stroke="var(--gold)" strokeWidth={2} markerEnd="url(#neuron-arrow)" />}
          {data.map((d, i) => {
            const isWrong = preds[i] !== d.y;
            const isCur = shown ? shown.index === i : i === 0;
            return (
              <g key={i}>
                {isCur && <circle cx={sx(d.x[0])} cy={sy(d.x[1])} r={17} fill="none" stroke="var(--gold)" strokeWidth={2} />}
                {isWrong && <circle cx={sx(d.x[0])} cy={sy(d.x[1])} r={12.5} fill="none" stroke={clay} strokeWidth={1.5} strokeDasharray="3 2" />}
                <circle cx={sx(d.x[0])} cy={sy(d.x[1])} r={8} fill={d.y === 1 ? accent : clay} stroke="var(--surface)" strokeWidth={1.5} />
                <text x={sx(d.x[0])} y={sy(d.x[1]) + 3.5} fontSize="9" textAnchor="middle" fill="var(--surface)" fontWeight={600}>
                  {d.y}
                </text>
              </g>
            );
          })}
          {!seg && (
            <text x={SIZE / 2} y={PAD + 14} fontSize="11" textAnchor="middle" fill="var(--text-muted)">
              w = (0, 0): no line, so every point gets the same answer
            </text>
          )}
        </svg>

        <div className="space-y-3 text-sm">
          <table className="w-full border-collapse text-center font-mono text-xs">
            <caption className="mb-1 text-left font-sans text-xs text-muted">{gate} with the current weights</caption>
            <thead>
              <tr className="border-b border-border text-faint">
                <th className="py-1 font-normal">x1</th>
                <th className="font-normal">x2</th>
                <th className="font-normal">z</th>
                <th className="font-normal">f(z)</th>
                <th className="font-normal">ŷ</th>
                <th className="font-normal">y</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d, i) => {
                const z = weightedSum(w, b, d.x);
                return (
                  <tr key={i} className="border-b border-border/60">
                    <td className="py-1">{d.x[0]}</td>
                    <td>{d.x[1]}</td>
                    <td>{n2(z)}</td>
                    <td>{fmt(activate(act, z), 2)}</td>
                    <td className={preds[i] === d.y ? "text-accent" : "text-clay"}>{preds[i]}</td>
                    <td>{d.y}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="font-mono text-xs text-muted">
            w = ({n2(w[0])}, {n2(w[1])}), b = {n2(b)}
          </div>
          <div className="flex flex-wrap gap-3 text-xs text-muted">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: accent }} /> class 1
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: clay }} /> class 0
            </span>
            <span>shaded: predicted 1</span>
            <span className="text-gold">arrow: w (at right angles to the line)</span>
          </div>
          <Narration title={narrTitle}>{narr}</Narration>
        </div>
      </div>
    </SimShell>
  );
}
