"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  PROBLEM,
  SCHEDULES,
  makeDataset,
  schedulePhase,
  trainWithSchedule,
  type ScheduleId,
  type ScheduleParams,
  type TrainRun,
} from "@/lib/sim/lrSchedule";

/** Per-schedule line style: theme colours only, plus dash patterns so overlapping lines stay distinguishable. */
const STYLE: Record<ScheduleId, { color: string; dash?: string; width: number }> = {
  constant: { color: "var(--text-muted)", width: 1.8 },
  step: { color: "var(--clay)", width: 1.8 },
  exponential: { color: "color-mix(in srgb, var(--clay) 50%, var(--text))", dash: "6 3", width: 1.8 },
  cosine: { color: "var(--accent)", width: 2 },
  restarts: { color: "color-mix(in srgb, var(--accent) 55%, var(--text))", dash: "2 3", width: 2 },
  "warmup-cosine": { color: "var(--gold)", width: 2.2 },
  "one-cycle": { color: "var(--text)", dash: "8 3 2 3", width: 1.8 },
  "inv-sqrt": { color: "color-mix(in srgb, var(--gold) 55%, var(--text))", dash: "5 4", width: 1.8 },
};

const DEFAULTS: ScheduleParams = { base: 0.05, total: 600, warmup: 60, stepSize: 150, gamma: 0.3, restartPeriod: 100 };
const DEFAULT_ON: ScheduleId[] = ["constant", "step", "cosine", "warmup-cosine"];

/* ---------- plot geometry (SVG user units; scaled by viewBox) ---------- */
const PW = 560;
const PH = 190;
const M = { l: 50, r: 10, t: 10, b: 26 };
const IW = PW - M.l - M.r;
const IH = PH - M.t - M.b;

const SMOOTH = 10;

const lrFmt = (v: number) => (v >= 0.01 ? v.toFixed(3) : v === 0 ? "0" : v.toExponential(1));

/** Indices 0..n-1 thinned to at most `max` points (always keeps the last). */
function sampleIdx(n: number, max = 500) {
  const stride = Math.max(1, Math.ceil(n / max));
  const out: number[] = [];
  for (let i = 0; i < n; i += stride) out.push(i);
  if (out[out.length - 1] !== n - 1) out.push(n - 1);
  return out;
}

export default function LrScheduleSim() {
  const [params, setParams] = useState<ScheduleParams>(DEFAULTS);
  const [logBase, setLogBase] = useState(Math.log10(DEFAULTS.base));
  const [enabled, setEnabled] = useState<Record<ScheduleId, boolean>>(
    () => Object.fromEntries(SCHEDULES.map((s) => [s.id, DEFAULT_ON.includes(s.id)])) as Record<ScheduleId, boolean>,
  );
  const [clipOn, setClipOn] = useState(false);
  const [clip, setClip] = useState(1);

  const p: ScheduleParams = useMemo(() => ({ ...params, base: 10 ** logBase }), [params, logBase]);
  const T = p.total;
  const set = (k: keyof ScheduleParams) => (v: number) => setParams((q) => ({ ...q, [k]: v }));

  // Real training runs for every schedule (cheap: 3 parameters, 400 examples).
  const ds = useMemo(() => makeDataset(), []);
  const runs = useMemo(() => {
    const out = {} as Record<ScheduleId, TrainRun>;
    for (const s of SCHEDULES) out[s.id] = trainWithSchedule(s.id, p, clipOn ? clip : null);
    return out;
  }, [p, clipOn, clip]);

  // Playback: each frame advances `stride` optimizer steps so a full run takes ~300 frames.
  const stride = Math.max(1, Math.round(T / 300));
  const frames = Math.ceil(T / stride) + 1;
  const s = useStepper(frames, { baseMs: 60 });
  const { setStep } = s;
  useEffect(() => setStep(0), [runs, setStep]);
  const t = Math.min(T, s.step * stride); // number of optimizer steps taken so far

  const active = SCHEDULES.filter((sc) => enabled[sc.id]);

  /* ---------- LR plot scales ---------- */
  const lrMax = p.base * 1.08;
  const xS = (step: number) => M.l + (step / T) * IW;
  const yLr = (lr: number) => M.t + IH - (lr / lrMax) * IH;
  const idx = useMemo(() => sampleIdx(T), [T]);

  // Trailing 10-step moving average of the per-step excess loss, so trends are readable through the SGD noise.
  const smooth = useMemo(() => {
    const out = {} as Record<ScheduleId, number[]>;
    for (const sc of SCHEDULES) {
      const ex = runs[sc.id].excess;
      out[sc.id] = ex.map((_, i) => {
        let sum = 0, n = 0;
        for (let k = Math.max(0, i - SMOOTH + 1); k <= i; k++) if (Number.isFinite(ex[k])) { sum += ex[k]; n += 1; }
        return Number.isFinite(ex[i]) && n ? sum / n : NaN;
      });
    }
    return out;
  }, [runs]);

  /* ---------- loss plot scales (log10 of the smoothed excess loss) ---------- */
  const { lo, hi } = useMemo(() => {
    let mn = Infinity, mx = -Infinity;
    for (const sc of SCHEDULES) {
      if (!enabled[sc.id]) continue;
      for (const v of smooth[sc.id]) {
        if (!Number.isFinite(v)) continue;
        mn = Math.min(mn, v);
        mx = Math.max(mx, v);
      }
    }
    if (!Number.isFinite(mn)) return { lo: -4, hi: 0 };
    return { lo: Math.max(-5, Math.floor(Math.log10(mn))), hi: Math.max(Math.floor(Math.log10(mn)) + 1, Math.ceil(Math.log10(mx))) };
  }, [smooth, enabled]);
  const yLoss = (v: number) => M.t + IH - ((Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo || 1)) * IH;

  const lossIdx = useMemo(() => sampleIdx(T + 1, 600), [T]);

  /* ---------- current numbers ---------- */
  const rows = active.map((sc) => {
    const run = runs[sc.id];
    const tLr = Math.min(t, T - 1);
    const loss = run.excess[t];
    const win = run.excess.slice(Math.max(1, t - 19), t + 1).filter(Number.isFinite);
    const recent = win.length ? win.reduce((a, b) => a + b, 0) / win.length : NaN;
    return { sc, lr: run.lr[tLr], loss, recent, clipped: run.clippedUpTo[t], diverged: !Number.isFinite(loss) };
  });
  const finiteRows = rows.filter((r) => !r.diverged && t > 0);
  const best = finiteRows.length ? finiteRows.reduce((a, b) => (b.recent < a.recent ? b : a)) : null;
  const constRow = rows.find((r) => r.sc.id === "constant");

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * T));
  const lrTicks = [0, 0.5, 1].map((f) => f * p.base);
  const lossTicks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);

  return (
    <SimShell
      title="Learning-Rate Schedules"
      subtitle="A schedule decides how the learning rate changes during training. Each one you pick gets its own real training run (logistic regression, mini-batch SGD + momentum) on the same data and batches."
      controls={
        <div className="space-y-2.5">
          <Controls s={s} extra={<span className="font-mono text-xs text-faint">optimizer step {t}/{T}</span>} />
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            {SCHEDULES.map((sc) => (
              <label key={sc.id} className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={enabled[sc.id]}
                  onChange={(e) => setEnabled({ ...enabled, [sc.id]: e.target.checked })}
                  aria-label={`Show ${sc.label}`}
                />
                <svg width="22" height="8" aria-hidden="true">
                  <line x1="1" y1="4" x2="21" y2="4" stroke={STYLE[sc.id].color} strokeWidth={2.2} strokeDasharray={STYLE[sc.id].dash} />
                </svg>
                {sc.label}
              </label>
            ))}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="Base / peak η₀" value={logBase} min={-3} max={-0.3} step={0.05} onChange={setLogBase} fmt={() => lrFmt(p.base)} />
            <Slider label="Total steps T" value={params.total} min={100} max={2000} step={50} onChange={(v) => setParams((q) => ({ ...q, total: v, warmup: Math.min(q.warmup, Math.floor(v / 2)) }))} fmt={(v) => v.toFixed(0)} />
            <Slider label="Warm-up W" value={params.warmup} min={0} max={Math.floor(params.total / 2)} step={10} onChange={set("warmup")} fmt={(v) => v.toFixed(0)} />
            <Slider label="Step size s" value={params.stepSize} min={20} max={1000} step={10} onChange={set("stepSize")} fmt={(v) => v.toFixed(0)} />
            <Slider label="Decay γ" value={params.gamma} min={0.05} max={0.9} step={0.05} onChange={set("gamma")} fmt={(v) => v.toFixed(2)} />
            <Slider label="Restart T₀" value={params.restartPeriod} min={20} max={500} step={10} onChange={set("restartPeriod")} fmt={(v) => v.toFixed(0)} />
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={clipOn} onChange={(e) => setClipOn(e.target.checked)} aria-label="Enable gradient clipping by global norm" />
              Clip gradient norm
            </label>
            <div className={`min-w-[200px] flex-1 ${clipOn ? "" : "opacity-40"}`}>
              <Slider label="threshold c" value={clip} min={0.1} max={5} step={0.1} onChange={setClip} fmt={(v) => v.toFixed(1)} />
            </div>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={t === 0 ? "Ready" : `After ${t} optimizer step${t > 1 ? "s" : ""}`}>
            {t === 0 ? (
              <p>
                Every run starts from w = 0. Press Play or Step. The gold line on the top plot marks the current step. The bottom plot shows how far the training loss is above its
                lowest possible value (log scale).
              </p>
            ) : (
              <>
                <ul className="mb-1.5 space-y-1">
                  {rows.map((r) => (
                    <li key={r.sc.id}>
                      <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: STYLE[r.sc.id].color }} />
                      <strong className="text-text">{r.sc.label}</strong>: {schedulePhase(r.sc.id, Math.min(t, T - 1), p)}.
                    </li>
                  ))}
                </ul>
                {best && (
                  <p>
                    Lowest recent loss: <strong className="text-text">{best.sc.label}</strong> ({fmt(best.recent, 4)} above the best possible, averaged over the last 20 steps).
                    {constRow && !constRow.diverged && best.sc.id !== "constant" && t > T / 2 && (
                      <> The constant rate is stuck at its noise floor ({fmt(constRow.recent, 3)}). Its steps never get smaller, so the noise from each mini-batch keeps bumping it around.</>
                    )}
                  </p>
                )}
              </>
            )}
          </Narration>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted">
                  <th className="py-1 pr-2 font-medium">Schedule</th>
                  <th className="py-1 pr-2 font-medium">η now</th>
                  <th className="py-1 pr-2 font-medium">loss − L*</th>
                  {clipOn && <th className="py-1 font-medium">clipped</th>}
                </tr>
              </thead>
              <tbody className="font-mono">
                {rows.map((r) => (
                  <tr key={r.sc.id} className="border-t border-border">
                    <td className="py-1 pr-2 font-sans">{r.sc.label}</td>
                    <td className="py-1 pr-2">{lrFmt(r.lr)}</td>
                    <td className={`py-1 pr-2 ${r.diverged ? "text-clay" : ""}`}>{r.diverged ? "diverged" : fmt(r.loss, 4)}</td>
                    {clipOn && <td className="py-1">{r.clipped}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-faint">
            {PROBLEM.n} examples, 2 features on scales 1 and 3, {Math.round(PROBLEM.labelNoise * 100)}% flipped labels, {Math.round(PROBLEM.outlierFrac * 100)}% outliers with 8×
            larger features. Batch {PROBLEM.batch}, momentum β = {PROBLEM.beta}. Lowest loss L* = {fmt(ds.lossStar, 4)} (found with Newton&apos;s method). Curvature (how sharply the loss bends) at w = 0 is{" "}
            {fmt(ds.lambdaMax0, 2)}, so momentum SGD stays stable there only for η below about 3.8/{fmt(ds.lambdaMax0, 1)} = {fmt(3.8 / ds.lambdaMax0, 2)}. Exponential decay
            uses η₀·γ^(t/s).
          </p>
        </div>
      }
    >
      <div className="space-y-3">
        {/* ---------------- learning-rate plot ---------------- */}
        <figure className="m-0">
          <figcaption className="mb-1 text-xs text-muted">Learning rate η_t</figcaption>
          <svg viewBox={`0 0 ${PW} ${PH}`} className="h-auto w-full" role="img" aria-label={`Learning rate versus step for ${active.map((a) => a.label).join(", ") || "no schedule"}; current step ${t}`}>
            {lrTicks.map((v) => (
              <g key={v}>
                <line x1={M.l} x2={PW - M.r} y1={yLr(v)} y2={yLr(v)} stroke="var(--border)" strokeWidth={1} />
                <text x={M.l - 5} y={yLr(v) + 3.5} textAnchor="end" fontSize={12} fill="var(--text-muted)">
                  {lrFmt(v)}
                </text>
              </g>
            ))}
            {xTicks.map((v) => (
              <text key={v} x={xS(v)} y={PH - 6} textAnchor="middle" fontSize={12} fill="var(--text-muted)">
                {v}
              </text>
            ))}
            {active.map((sc) => {
              const lr = runs[sc.id].lr;
              const pts = idx.map((i) => `${xS(i).toFixed(1)},${yLr(lr[i]).toFixed(1)}`).join(" ");
              return <polyline key={sc.id} points={pts} fill="none" stroke={STYLE[sc.id].color} strokeWidth={STYLE[sc.id].width} strokeDasharray={STYLE[sc.id].dash} strokeLinejoin="round" />;
            })}
            <line x1={xS(t)} x2={xS(t)} y1={M.t} y2={M.t + IH} stroke="var(--gold)" strokeWidth={1.5} />
            {active.map((sc) => (
              <circle key={sc.id} cx={xS(t)} cy={yLr(runs[sc.id].lr[Math.min(t, T - 1)])} r={3.6} fill={STYLE[sc.id].color} stroke="var(--surface)" strokeWidth={1.2} />
            ))}
          </svg>
        </figure>

        {/* ---------------- loss plot ---------------- */}
        <figure className="m-0">
          <figcaption className="mb-1 text-xs text-muted">Training loss above its lowest value, L(w_t) − L* (log scale, averaged over {SMOOTH} steps)</figcaption>
          <svg viewBox={`0 0 ${PW} ${PH}`} className="h-auto w-full" role="img" aria-label={`Excess training loss on a log scale for each selected schedule up to step ${t}`}>
            {lossTicks.map((e) => (
              <g key={e}>
                <line x1={M.l} x2={PW - M.r} y1={yLoss(10 ** e)} y2={yLoss(10 ** e)} stroke="var(--border)" strokeWidth={1} />
                <text x={M.l - 5} y={yLoss(10 ** e) + 3.5} textAnchor="end" fontSize={12} fill="var(--text-muted)">
                  1e{e}
                </text>
              </g>
            ))}
            {xTicks.map((v) => (
              <text key={v} x={xS(v)} y={PH - 6} textAnchor="middle" fontSize={12} fill="var(--text-muted)">
                {v}
              </text>
            ))}
            {active.map((sc) => {
              const ex = smooth[sc.id];
              // diverged (NaN) points are skipped
              const pts = [...lossIdx.filter((i) => i < t), t]
                .filter((i) => Number.isFinite(ex[i]))
                .map((i) => `${xS(i).toFixed(1)},${yLoss(ex[i]).toFixed(1)}`)
                .join(" ");
              return <polyline key={sc.id} points={pts} fill="none" stroke={STYLE[sc.id].color} strokeWidth={STYLE[sc.id].width * 0.8} strokeDasharray={STYLE[sc.id].dash} strokeLinejoin="round" />;
            })}
            <line x1={xS(t)} x2={xS(t)} y1={M.t} y2={M.t + IH} stroke="var(--gold)" strokeWidth={1} opacity={0.6} />
            {rows
              .filter((r) => !r.diverged)
              .map((r) => (
                <circle key={r.sc.id} cx={xS(t)} cy={yLoss(smooth[r.sc.id][t])} r={3.2} fill={STYLE[r.sc.id].color} stroke="var(--surface)" strokeWidth={1.2} />
              ))}
          </svg>
        </figure>
      </div>
    </SimShell>
  );
}
