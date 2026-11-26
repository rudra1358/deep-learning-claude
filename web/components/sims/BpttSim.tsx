"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { clipScale, runBptt, sumNorm, type LossMode } from "@/lib/sim/bptt";

const vec = (v: number[], d = 3) => "[" + v.map((x) => sci3(x, d)).join(", ") + "]";
/** Fixed notation for ordinary magnitudes, scientific for very small or very large ones. */
function sci3(v: number, d = 3) {
  if (!Number.isFinite(v)) return "inf";
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(2);
  return v.toFixed(d);
}
const sci = (v: number) => (!Number.isFinite(v) ? "inf" : v === 0 ? "0" : v.toExponential(2));

export default function BpttSim() {
  const [T, setT] = useState(20);
  const [rho, setRho] = useState(0.9);
  const [mode, setMode] = useState<LossMode>("last");
  const [act, setAct] = useState<"tanh" | "linear">("tanh");
  const [xScale, setXScale] = useState(1);
  const [kRaw, setK] = useState(50);
  const [clipOn, setClipOn] = useState(false);
  const [clip, setClip] = useState(1);

  const k = Math.min(kRaw, T);
  const res = useMemo(() => runBptt({ T, rho, mode, k, act, xScale }), [T, rho, mode, k, act, xScale]);

  const total = T + 1;
  const s = useStepper(total, { baseMs: 700 });
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [T, mode, act, reset]);

  const step = Math.min(s.step, total - 1);
  const cur = step === 0 ? null : T - step + 1; // 1-based time step being processed
  const processed = (t: number) => step > 0 && t >= T - step + 1;
  const finished = step === total - 1;
  const kept = (t: number) => (mode === "last" ? T - t < k : true);

  // ---------- geometry ----------
  const W = 720;
  const left = 74;
  const colW = (W - left - 8) / T;
  const cx = (t: number) => left + (t - 1) * colW + colW / 2;
  const bw = Math.max(2, Math.min(22, colW * 0.7));

  const logs = res.dhNorm.map((v) => (v > 0 ? Math.log10(v) : -30));
  const cLogs = res.contribNorm.map((v) => (v > 0 ? Math.log10(v) : -30));
  const allLogs = [...logs, ...cLogs].filter((v) => v > -30);
  const lo = Math.max(-30, Math.floor(Math.min(...allLogs, -1)));
  const hi = Math.ceil(Math.max(...allLogs, 0));
  const facMax = Math.max(2, Math.ceil(Math.max(...res.factor)));

  // rows: cells 0..60, dh 70..170, factor 185..245, contrib 260..360
  const rowY = { cells: 22, dhTop: 72, dhBot: 168, fTop: 186, fBot: 244, cTop: 262, cBot: 358 };
  const yLog = (l: number, top: number, bot: number) => bot - ((Math.max(lo, l) - lo) / (hi - lo || 1)) * (bot - top);
  const yFac = (f: number) => rowY.fBot - (Math.min(f, facMax) / facMax) * (rowY.fBot - rowY.fTop);
  const tickStep = Math.max(1, Math.ceil((hi - lo) / 4));
  const ticks: number[] = [];
  for (let l = hi; l >= lo; l -= tickStep) ticks.push(l);

  // ---------- read-outs ----------
  const doneSteps = Array.from({ length: T }, (_, i) => i + 1).filter(processed);
  const accFull = sumNorm(doneSteps.map((t) => res.contribFull[t - 1]));
  const accKept = sumNorm(doneSteps.map((t) => res.contribKept[t - 1]));
  const usedNorm = res.normTrunc; // the gradient actually used for the update (truncated if k < T)
  const scale = clipOn ? clipScale(usedNorm, clip) : 1;
  const ratio = res.dhNorm[0] / res.dhNorm[T - 1];

  // ---------- narration ----------
  let title = "";
  let explain = "";
  let lines: string[] = [];
  if (cur === null) {
    title = "Forward pass done: we have the loss";
    explain = mode === "last" ? `The loss sits only at the last step: L = ½(y_T − 1)². Press Step to send its gradient backwards through all ${T} steps.` : `Every step has its own loss term ½(y_t − y*_t)². Each one adds gradient at its own step, and all of it flows left.`;
    lines = [`L = ${sci3(res.loss, 4)}`, `ρ(W_hh) = ${fmt(res.rhoActual, 3)},  ‖W_hh‖₂ = ${fmt(res.whhNorm, 3)}`];
  } else {
    const t = cur;
    const i = t - 1;
    const h = res.hs[t];
    title = `Backward through step t = ${t}`;
    const fromY = res.dhFromY[i];
    const fromNext = res.dhFromNext[i];
    lines = [
      `∂L/∂h_${t} = W_hy·e_${t} + W_hhᵀδ_${t + 1}`,
      `= ${vec(fromY)} + ${vec(fromNext)}`,
      `= ${vec(res.dh[i])}   (‖·‖ = ${sci(res.dhNorm[i])})`,
      `1 − h_${t}² = ${vec(h.map((v) => (act === "linear" ? 1 : 1 - v * v)))}`,
      `δ_${t} = ∂L/∂h_${t} ⊙ (1 − h_${t}²) = ${vec(res.dz[i])}`,
      `contribution δ_${t} h_${t - 1}ᵀ to ∂L/∂W_hh: ‖·‖_F = ${sci(res.contribNorm[i])}${kept(t) || mode === "every" ? "" : "  — DROPPED by truncation"}`,
      `per-step factor ‖W_hhᵀ diag(1 − h_${t}²)‖₂ = ${fmt(res.factor[i], 3)}`,
    ];
    const r = res.dhNorm[i] / res.dhNorm[T - 1];
    explain =
      t === T
        ? "The gradient enters at the last hidden state. Each step to the left multiplies it by W_hhᵀ diag(1 − h²)."
        : `‖∂L/∂h_${t}‖ is ${sci(r)} × ‖∂L/∂h_${T}‖. ${r < 1e-3 ? "The signal from the loss has almost vanished. The weights learn almost nothing about how step " + t + " affected the loss." : r > 1e3 ? "The signal has exploded. Without clipping, these far-away steps would swamp the whole update." : "The signal is still a useful size."}`;
  }

  return (
    <SimShell
      title="Backpropagation Through Time"
      subtitle={`A plain RNN: input size 1, hidden size 4, T = ${T}. The forward pass runs once. Then each Step runs the backward pass, right → left.`}
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="Sequence length T" value={T} min={2} max={50} step={1} onChange={(v) => setT(Math.round(v))} fmt={(v) => String(Math.round(v))} />
            <Slider label="Spectral radius ρ" value={rho} min={0.1} max={3} step={0.05} onChange={setRho} fmt={(v) => v.toFixed(2)} />
            <Slider label="Truncation k" value={k} min={1} max={T} step={1} onChange={(v) => setK(Math.round(v) >= T ? 50 : Math.round(v))} fmt={(v) => (Math.round(v) >= T ? "full" : String(Math.round(v)))} />
            <Slider label="Input scale" value={xScale} min={0.05} max={2} step={0.05} onChange={setXScale} fmt={(v) => v.toFixed(2)} />
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Loss at</span>
              {(["last", "every"] as const).map((m) => (
                <button key={m} className={`sim-btn ${mode === m ? "primary" : ""}`} onClick={() => setMode(m)} aria-label={m === "last" ? "Loss at the last step only" : "Loss at every step"}>
                  {m === "last" ? "last step" : "every step"}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Activation</span>
              {(["tanh", "linear"] as const).map((a) => (
                <button key={a} className={`sim-btn ${act === a ? "primary" : ""}`} onClick={() => setAct(a)} aria-label={`Activation ${a}`}>
                  {a}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs sm:col-span-2">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={clipOn} onChange={(e) => setClipOn(e.target.checked)} aria-label="Enable gradient norm clipping" />
                <span className="text-muted">Clip global norm at</span>
              </label>
              <div className="min-w-0 flex-1">
                <Slider label="threshold c" value={clip} min={0.1} max={10} step={0.1} onChange={setClip} fmt={(v) => v.toFixed(1)} />
              </div>
            </div>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={title}>
            <p className="mb-2">{explain}</p>
            <div className="font-mono text-[11px] leading-relaxed text-text">
              {lines.map((l, i) => (
                <div key={i} className="break-words">
                  {l}
                </div>
              ))}
            </div>
          </Narration>
          <div className="text-xs text-muted">
            Try: T = 40, tanh, ρ = 0.9. The bars shrink by many powers of ten (vanishing). Switch to linear and drag ρ above 1: they multiply up fast (exploding). Then tick clipping. Last, set k = 5 and see which parts of ∂L/∂W_hh truncation throws away.
          </div>
        </div>
      }
    >
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W} 372`} className="h-auto w-full" style={{ minWidth: 560 }} role="img" aria-label={`Backward pass through ${T} time steps; ${step === 0 ? "not started" : `${step} steps processed`}`}>
          {/* row labels */}
          <text x={4} y={rowY.cells + 14} fontSize={10} fill="var(--text-muted)">
            steps
          </text>
          <text x={4} y={rowY.dhTop + 8} fontSize={10} fill="var(--text-muted)">
            ‖∂L/∂h_t‖
          </text>
          <text x={4} y={rowY.dhTop + 20} fontSize={9} fill="var(--text-muted)">
            (log)
          </text>
          <text x={4} y={rowY.fTop + 8} fontSize={10} fill="var(--text-muted)">
            factor
          </text>
          <text x={4} y={rowY.fTop + 20} fontSize={9} fill="var(--text-muted)">
            ‖W_hhᵀD_t‖₂
          </text>
          <text x={4} y={rowY.cTop + 8} fontSize={10} fill="var(--text-muted)">
            ∂L/∂W_hh
          </text>
          <text x={4} y={rowY.cTop + 20} fontSize={9} fill="var(--text-muted)">
            from step t
          </text>

          {/* log grid for the two log rows */}
          {ticks.map((l) => (
            <g key={l}>
              {[
                [rowY.dhTop, rowY.dhBot],
                [rowY.cTop, rowY.cBot],
              ].map(([a, b2]) => (
                <g key={a}>
                  <line x1={left} x2={W - 4} y1={yLog(l, a, b2)} y2={yLog(l, a, b2)} stroke="var(--border)" strokeDasharray="2 3" />
                  <text x={left - 3} y={yLog(l, a, b2) + 3} textAnchor="end" fontSize={8.5} fill="var(--text-muted)">
                    1e{l}
                  </text>
                </g>
              ))}
            </g>
          ))}
          {/* factor axis: line at 1 */}
          <line x1={left} x2={W - 4} y1={yFac(1)} y2={yFac(1)} stroke="var(--gold)" strokeDasharray="4 3" />
          <text x={left - 3} y={yFac(1) + 3} textAnchor="end" fontSize={8.5} fill="var(--gold)">
            1
          </text>
          <text x={left - 3} y={rowY.fTop + 3} textAnchor="end" fontSize={8.5} fill="var(--text-muted)">
            {facMax}
          </text>

          {Array.from({ length: T }, (_, i) => i + 1).map((t) => {
            const done = processed(t);
            const isCur = cur === t;
            const keep = kept(t);
            const x = cx(t);
            const cellStroke = isCur ? "var(--gold)" : done ? "var(--clay)" : "var(--border)";
            const hasLoss = mode === "every" || t === T;
            const keptFrac = res.contribNorm[t - 1] > 0 ? res.contribKeptNorm[t - 1] / res.contribNorm[t - 1] : 1;
            return (
              <g key={t}>
                {/* cell */}
                <rect x={x - colW * 0.42} y={rowY.cells} width={colW * 0.84} height={22} rx={3} fill={keep ? "var(--surface)" : "var(--surface-2)"} stroke={cellStroke} strokeWidth={isCur ? 2 : 1} strokeDasharray={keep ? undefined : "3 2"} />
                {hasLoss && (
                  <text x={x} y={rowY.cells - 4} textAnchor="middle" fontSize={Math.min(10, colW * 0.8)} fill="var(--clay)">
                    L
                  </text>
                )}
                {(T <= 25 || t % 5 === 0 || t === 1) && (
                  <text x={x} y={rowY.cells + 36} textAnchor="middle" fontSize={8.5} fill={isCur ? "var(--gold)" : "var(--text-muted)"}>
                    {t}
                  </text>
                )}
                {t < T && <line x1={x + colW * 0.42} x2={x + colW * 0.58} y1={rowY.cells + 11} y2={rowY.cells + 11} stroke={done && processed(t + 1) ? "var(--clay)" : "var(--border)"} />}

                {/* dL/dh_t */}
                {done && (
                  <rect x={x - bw / 2} y={yLog(logs[t - 1], rowY.dhTop, rowY.dhBot)} width={bw} height={Math.max(0.6, rowY.dhBot - yLog(logs[t - 1], rowY.dhTop, rowY.dhBot))} fill={isCur ? "var(--gold)" : "var(--clay)"} />
                )}
                {/* per-step factor */}
                {done && <rect x={x - bw / 2} y={yFac(res.factor[t - 1])} width={bw} height={Math.max(0.6, rowY.fBot - yFac(res.factor[t - 1]))} fill={res.factor[t - 1] > 1 ? "var(--clay)" : "var(--accent)"} opacity={0.8} />}
                {/* contribution to dL/dW_hh: outline = full, filled = kept after truncation */}
                {done && (
                  <g>
                    <rect x={x - bw / 2} y={yLog(cLogs[t - 1], rowY.cTop, rowY.cBot)} width={bw} height={Math.max(0.6, rowY.cBot - yLog(cLogs[t - 1], rowY.cTop, rowY.cBot))} fill="none" stroke="var(--text-muted)" strokeWidth={0.8} />
                    {keptFrac > 0 && (
                      <rect
                        x={x - bw / 2}
                        y={yLog(Math.log10(Math.max(1e-300, res.contribKeptNorm[t - 1])), rowY.cTop, rowY.cBot)}
                        width={bw}
                        height={Math.max(0.6, rowY.cBot - yLog(Math.log10(Math.max(1e-300, res.contribKeptNorm[t - 1])), rowY.cTop, rowY.cBot))}
                        fill={isCur ? "var(--gold)" : "var(--accent)"}
                        opacity={0.85}
                      />
                    )}
                  </g>
                )}
              </g>
            );
          })}
          <text x={W - 4} y={rowY.cBot + 12} textAnchor="end" fontSize={9} fill="var(--text-muted)">
            outline = full BPTT share · filled = part kept with truncation k = {k >= T ? "full" : k}
          </text>
        </svg>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-2 font-mono text-[11px] sm:grid-cols-2">
        <div className="rounded border border-border p-2">
          <div className="text-muted">loss L</div>
          <div>{sci3(res.loss, 4)}</div>
        </div>
        <div className="rounded border border-border p-2">
          <div className="text-muted">‖∂L/∂h_1‖ / ‖∂L/∂h_T‖</div>
          <div>{finished ? sci(ratio) : "run the backward pass first"}</div>
        </div>
        <div className="rounded border border-border p-2">
          <div className="text-muted">‖∂L/∂W_hh‖ summed so far (full | truncated)</div>
          <div>
            {sci(accFull)} | {sci(accKept)}
          </div>
        </div>
        <div className="rounded border border-border p-2">
          <div className="text-muted">ρ(W_hh) · ‖W_hh‖₂ · mean factor</div>
          <div>
            {fmt(res.rhoActual, 2)} · {fmt(res.whhNorm, 2)} · {fmt(res.factor.reduce((a, b) => a + b, 0) / T, 2)}
          </div>
        </div>
        <div className="rounded border border-border p-2 sm:col-span-2">
          <div className="text-muted">global gradient norm (all parameters{k < T ? ", truncated" : ""}) {clipOn ? `→ after clipping at c = ${clip.toFixed(1)}` : "(clipping off)"}</div>
          <div className="mt-1 space-y-1">
            {[
              { label: "raw", v: usedNorm, color: "var(--clay)" },
              ...(clipOn ? [{ label: "clipped", v: usedNorm * scale, color: "var(--accent)" }] : []),
            ].map((b) => {
              const ref = Math.max(usedNorm, clip, 1e-12);
              const frac = Math.log10(1 + b.v) / Math.log10(1 + ref);
              return (
                <div key={b.label} className="flex items-center gap-2">
                  <span className="w-14 shrink-0 text-muted">{b.label}</span>
                  <svg viewBox="0 0 200 10" className="h-2.5 min-w-0 flex-1" aria-hidden>
                    <rect x={0} y={1} width={Math.max(0.5, 200 * Math.min(1, frac))} height={8} fill={b.color} />
                  </svg>
                  <span className="w-20 shrink-0 text-right">{sci(b.v)}</span>
                </div>
              );
            })}
            {clipOn && <div className="text-muted">scale factor min(1, c/‖g‖) = {sci3(scale, 4)}. The direction never changes</div>}
          </div>
        </div>
      </div>
    </SimShell>
  );
}
