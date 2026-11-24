"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, fmt, seqColor, useStepper } from "./SimShell";
import {
  cosineSchedule,
  ddimStep,
  ddimTimesteps,
  ddpmStep,
  energyDistance,
  linearSchedule,
  loadModel,
  mulberry32,
  qSample,
  randn,
  verifyModel,
  type DiffusionJSON,
  type DiffusionModel,
  type Net,
  type Schedule,
} from "@/lib/sim/diffusion";

const MODEL_URL = "/models/diffusion.json";
const TITLE = "Diffusion Model Explorer (trained models)";
const SUBTITLE = "Real DDPM noise-prediction networks running in your browser: forward noising, reverse denoising, DDPM vs DDIM, guided digits.";
const N_POINTS = 300;
const LIM = 3.6; // half-width of the 2-D plot (standardised units)

type Tab = "forward" | "reverse" | "digits";

/* ------------------------------------------------------------------ small pieces */

function Toggle<T extends string | number>({ value, options, onChange, label }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={label}>
      <span className="text-xs text-muted">{label}</span>
      {options.map((o) => (
        <button key={String(o.v)} className={`sim-btn ${o.v === value ? "primary" : ""}`} aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Scatter({ pts, real, ghost, label, caption }: { pts: Float64Array; real?: Float64Array; ghost?: Float64Array | null; label: string; caption?: ReactNode }) {
  const S = 100 / LIM;
  const X = (v: number) => Math.max(-LIM, Math.min(LIM, v)) * S;
  const Y = (v: number) => -Math.max(-LIM, Math.min(LIM, v)) * S;
  const dots = (a: Float64Array, r: number, fill: string, op: number) => {
    const out: ReactNode[] = [];
    for (let i = 0; i < a.length; i += 2) out.push(<circle key={i} cx={X(a[i])} cy={Y(a[i + 1])} r={r} fill={fill} opacity={op} />);
    return out;
  };
  return (
    <figure className="min-w-0">
      <svg viewBox="-104 -104 208 208" className="h-auto w-full rounded-md border border-border bg-surface" role="img" aria-label={label}>
        <line x1={-100} x2={100} y1={0} y2={0} stroke="var(--border)" strokeWidth={0.5} />
        <line y1={-100} y2={100} x1={0} x2={0} stroke="var(--border)" strokeWidth={0.5} />
        {[-3, -2, -1, 1, 2, 3].map((k) => (
          <text key={k} x={k * S} y={8} fontSize={5} textAnchor="middle" fill="var(--text-muted)">
            {k}
          </text>
        ))}
        {real && dots(real, 1.3, "var(--text-muted)", 0.35)}
        {ghost && dots(ghost, 1.2, "var(--gold)", 0.8)}
        {dots(pts, 1.6, "var(--accent)", 0.85)}
      </svg>
      {caption && <figcaption className="mt-1 text-xs text-muted">{caption}</figcaption>}
    </figure>
  );
}

function LineChart({ title, series, t, T, yMin, yMax, yTicks, label }: { title: string; series: { name: string; color: string; y: number[] }[]; t: number; T: number; yMin: number; yMax: number; yTicks: number[]; label: string }) {
  const W = 300, H = 130, L = 34, R = 8, Tp = 16, B = 22;
  const px = (i: number) => L + ((i + 1) / T) * (W - L - R);
  const py = (v: number) => Tp + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * (H - Tp - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={label}>
      <text x={L} y={10} fontSize={9} fill="var(--text)">
        {title}
      </text>
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={py(v)} y2={py(v)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={L - 3} y={py(v) + 3} fontSize={8} textAnchor="end" fill="var(--text-muted)">
            {v}
          </text>
        </g>
      ))}
      {[0, 25, 50, 75, 100].filter((k) => k <= T).map((k) => (
        <text key={k} x={k === 0 ? L : px(k - 1)} y={H - 8} fontSize={8} textAnchor="middle" fill="var(--text-muted)">
          {k}
        </text>
      ))}
      <text x={W - R} y={H - 1} fontSize={8} textAnchor="end" fill="var(--text-muted)">
        t
      </text>
      {series.map((s) => (
        <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={1.6} points={s.y.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ")} />
      ))}
      {t > 0 && <line x1={px(t - 1)} x2={px(t - 1)} y1={Tp} y2={H - B} stroke="var(--gold)" strokeWidth={1.2} />}
      {series.map((s, k) => (
        <g key={s.name} transform={`translate(${W - R - 92}, ${Tp + 4 + k * 11})`}>
          <line x1={0} x2={12} y1={0} y2={0} stroke={s.color} strokeWidth={2} />
          <text x={16} y={3} fontSize={8} fill="var(--text)">
            {s.name}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** An 8×8 image in [-1, 1] drawn as a pixel grid. */
function Digit({ px, size = 64, label }: { px: ArrayLike<number>; size?: number; label: string }) {
  const cells: ReactNode[] = [];
  for (let i = 0; i < 64; i++) {
    const v = (Math.max(-1, Math.min(1, px[i])) + 1) / 2;
    cells.push(<rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1.02} height={1.02} fill={seqColor(v)} />);
  }
  return (
    <svg viewBox="0 0 8 8" style={{ maxWidth: size }} className="block h-auto w-full rounded-sm border border-border" role="img" aria-label={label} shapeRendering="crispEdges">
      {cells}
    </svg>
  );
}

function Stat({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex justify-between gap-2 border-b border-border/60 py-0.5 text-xs">
      <span className="text-muted">{k}</span>
      <span className="font-mono tabular-nums">{v}</span>
    </div>
  );
}

function TabBar({ tab, setTab, hasDigits }: { tab: Tab; setTab: (t: Tab) => void; hasDigits: boolean }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "forward", label: "(a) Forward" },
    { id: "reverse", label: "(b) Reverse" },
    ...(hasDigits ? [{ id: "digits" as Tab, label: "(c) Digits" }] : []),
  ];
  return (
    <div className="mb-2 flex flex-wrap gap-2" role="tablist" aria-label="Simulator view">
      {tabs.map((x) => (
        <button key={x.id} role="tab" aria-selected={tab === x.id} className={`sim-btn ${tab === x.id ? "primary" : ""}`} onClick={() => setTab(x.id)}>
          {x.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ (a) forward process */

function ForwardTab({ model, tabs }: { model: DiffusionModel; tabs: ReactNode }) {
  const T = model.sched.T;
  const s = useStepper(T + 1, { baseMs: 160 });
  const t = s.step; // paper time step 0..T (0 = clean data)
  const [kind, setKind] = useState<"linear" | "cosine">("linear");
  const scheds = useMemo(() => ({ linear: model.sched, cosine: cosineSchedule(T) }), [model, T]);
  const sched: Schedule = scheds[kind];
  const x0 = useMemo(() => model.moonsReal.slice(0, 2 * N_POINTS), [model]);
  const eps = useMemo(() => randn(2 * N_POINTS, mulberry32(7)), []);
  const xt = useMemo(() => qSample(sched, x0, t - 1, eps), [sched, x0, t, eps]);
  const digit0 = useMemo(() => (model.digitsReal ? model.digitsReal.slice(0, 64) : null), [model]);
  const digitEps = useMemo(() => randn(64, mulberry32(11)), []);
  const digitT = useMemo(() => (digit0 ? qSample(sched, digit0, t - 1, digitEps) : null), [sched, digit0, t, digitEps]);
  const noiseRef = useMemo(() => randn(2 * N_POINTS, mulberry32(99)), []);
  const ed = useMemo(() => energyDistance(xt, noiseRef, 2), [xt, noiseRef]);

  const ab = t > 0 ? sched.alphaBar[t - 1] : 1;
  const beta = t > 0 ? sched.betas[t - 1] : 0;
  const snr = ab / Math.max(1 - ab, 1e-12);
  const lin = linearSchedule(T, model.json.schedule.beta_start, model.json.schedule.beta_end);
  const logSnr = (sc: Schedule) => Array.from(sc.alphaBar, (a) => Math.log10(a / (1 - a)));
  const i0 = 0; // tracked point
  const pt = xt.slice(2 * i0, 2 * i0 + 2);

  return (
    <SimShell
      title={TITLE}
      subtitle={SUBTITLE}
      controls={
        <div className="space-y-2">
          {tabs}
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <div className="min-w-0 flex-1 basis-56">
              <Slider label="time step t" value={t} min={0} max={T} step={1} onChange={(v) => s.setStep(v)} fmt={(v) => String(v)} />
            </div>
            <Toggle label="schedule" value={kind} onChange={setKind} options={[{ v: "linear", label: "linear (trained)" }, { v: "cosine", label: "cosine" }]} />
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={t === 0 ? "t = 0: the data" : `t = ${t}: one closed-form jump`}>
            {t === 0
              ? "300 real points from the two-moons data the model was trained on (standardised). Press Play: every frame jumps straight from x₀ to x_t with the closed form — no step-by-step simulation is needed."
              : `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε with ᾱ_${t} = ${fmt(ab, 4)}: each point keeps ${fmt(Math.sqrt(ab), 3)} of its position and receives ${fmt(Math.sqrt(1 - ab), 3)} units of standard Gaussian noise. ${snr > 1 ? "Signal still dominates (SNR > 1): the moons are visible." : snr > 0.05 ? "Noise now dominates (SNR < 1): the shape is fading." : "Essentially pure N(0, I) noise: the data is forgotten."}`}
          </Narration>
          <div>
            <Stat k={`β_${t}`} v={fmt(beta, 4)} />
            <Stat k={`ᾱ_${t} = Π(1−β)`} v={fmt(ab, 5)} />
            <Stat k="√ᾱ_t (signal scale)" v={fmt(Math.sqrt(ab), 4)} />
            <Stat k="√(1−ᾱ_t) (noise scale)" v={fmt(Math.sqrt(1 - ab), 4)} />
            <Stat k="SNR = ᾱ/(1−ᾱ)" v={snr > 1e4 ? snr.toExponential(2) : fmt(snr, 4)} />
            <Stat k="energy dist. to N(0, I)" v={fmt(ed, 4)} />
          </div>
          <div className="rounded-md border border-border p-2 font-mono text-[11px] leading-relaxed">
            point #1: x₀ = ({fmt(x0[0], 2)}, {fmt(x0[1], 2)}), ε = ({fmt(eps[0], 2)}, {fmt(eps[1], 2)})
            <br />x_t = {fmt(Math.sqrt(ab), 3)}·x₀ + {fmt(Math.sqrt(1 - ab), 3)}·ε = ({fmt(pt[0], 2)}, {fmt(pt[1], 2)})
          </div>
        </div>
      }
    >
      <div className="grid min-w-0 gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <Scatter pts={xt} label={`Two-moons data noised to step ${t}`} caption={`x_t for 300 points, ${kind} schedule (the same ε for every t, so you can follow each point).`} />
          {digitT && (
            <div className="mt-2 flex items-center gap-3">
              <div className="w-14 shrink-0">
                <Digit px={digit0 as Float64Array} label="Clean digit" size={56} />
              </div>
              <span className="text-xs text-muted">→</span>
              <div className="w-14 shrink-0">
                <Digit px={digitT} label={`Digit noised to step ${t}`} size={56} />
              </div>
              <span className="text-xs text-muted">the same q(x_t | x₀) applied to a 64-pixel digit</span>
            </div>
          )}
        </div>
        <div className="min-w-0 space-y-1">
          <LineChart title="ᾱ_t (fraction of signal variance left)" T={T} t={t} yMin={0} yMax={1} yTicks={[0, 0.5, 1]} label="alpha-bar for linear and cosine schedules" series={[{ name: "linear", color: "var(--accent)", y: Array.from(lin.alphaBar) }, { name: "cosine", color: "var(--clay)", y: Array.from(scheds.cosine.alphaBar) }]} />
          <LineChart title="log₁₀ SNR = log₁₀ ᾱ/(1−ᾱ)" T={T} t={t} yMin={-6} yMax={3} yTicks={[-6, -3, 0, 3]} label="log signal-to-noise ratio for both schedules" series={[{ name: "linear", color: "var(--accent)", y: logSnr(lin) }, { name: "cosine", color: "var(--clay)", y: logSnr(scheds.cosine) }]} />
          <p className="text-xs text-muted">The linear schedule destroys information quickly in the middle steps; the cosine schedule keeps ᾱ near 1 longer and declines almost linearly in the middle. Both models below were trained with the linear one.</p>
        </div>
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ (b) reverse process */

interface Traj {
  key: string;
  xs: Float64Array[];
  eps: Float64Array[];
  x0: Float64Array[];
}

function useTrajectory(model: DiffusionModel, net: Net, n: number, sampler: "ddpm" | "ddim", S: number, seed: number, cls: number[] | null, w: number) {
  const T = model.sched.T;
  const times = useMemo(() => (sampler === "ddpm" ? Array.from({ length: T }, (_, i) => T - 1 - i) : ddimTimesteps(T, S)), [sampler, S, T]);
  const key = `${net.json.name}|${n}|${sampler}|${S}|${seed}|${cls?.join(",") ?? ""}|${w}`;
  const ref = useRef<Traj | null>(null);
  if (!ref.current || ref.current.key !== key) ref.current = { key, xs: [randn(n * net.d, mulberry32(seed))], eps: [], x0: [] };
  const ensure = (k: number) => {
    const tr = ref.current as Traj;
    while (tr.xs.length <= k && tr.xs.length <= times.length) {
      const i = tr.xs.length - 1;
      const t = times[i];
      const x = tr.xs[i];
      const r =
        sampler === "ddpm"
          ? ddpmStep(model, net, x, n, t, randn(n * net.d, mulberry32(seed * 7919 + t + 1)), cls, w)
          : ddimStep(model, net, x, n, t, i + 1 < times.length ? times[i + 1] : -1, cls, w);
      tr.xs.push(r.x);
      tr.eps.push(r.eps);
      tr.x0.push(r.x0);
    }
    return tr;
  };
  return { times, ensure };
}

function ReverseTab({ model, tabs }: { model: DiffusionModel; tabs: ReactNode }) {
  const T = model.sched.T;
  const [sampler, setSampler] = useState<"ddpm" | "ddim">("ddpm");
  const [S, setS] = useState(20);
  const [seed, setSeed] = useState(1);
  const [showX0, setShowX0] = useState(false);
  const net = model.moons;
  const { times, ensure } = useTrajectory(model, net, N_POINTS, sampler, S, seed, null, 0);
  const nSteps = times.length;
  const s = useStepper(nSteps + 1, { baseMs: sampler === "ddpm" ? 120 : 500 });
  const { reset } = s;
  useEffect(() => reset(), [sampler, S, seed, reset]);
  const k = Math.min(s.step, nSteps);
  const tr = ensure(k);
  const x = tr.xs[k];
  const done = k === nSteps;
  const tNow = done ? 0 : times[k] + 1; // paper time of the current state
  const lastEps = k > 0 ? tr.eps[k - 1] : null;
  const lastX0 = k > 0 ? tr.x0[k - 1] : null;
  const real = useMemo(() => model.moonsReal.slice(0, 2 * N_POINTS), [model]);
  const realOther = useMemo(() => model.moonsReal.slice(2 * N_POINTS, 4 * N_POINTS), [model]);
  const ed = useMemo(() => energyDistance(x, real, 2), [x, real]);
  const edRef = useMemo(() => energyDistance(realOther, real, 2), [realOther, real]);
  const tLastIdx = k > 0 ? times[k - 1] : -1; // 0-based index of the step just taken
  const { betas, alphaBar } = model.sched;
  const sigmaLast = tLastIdx > 0 ? Math.sqrt((betas[tLastIdx] * (1 - alphaBar[tLastIdx - 1])) / (1 - alphaBar[tLastIdx])) : 0;
  const epsNorm = lastEps ?Math.sqrt(lastEps.reduce((a, v) => a + v * v, 0) / N_POINTS) : 0;
  const met = net.json.metrics as Record<string, number>;
  const table = [5, 10, 20, 50, 100].map((q) => [q, met[`energy_ddim_${q}`]] as const);

  return (
    <SimShell
      title={TITLE}
      subtitle={SUBTITLE}
      controls={
        <div className="space-y-2">
          {tabs}
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Toggle label="sampler" value={sampler} onChange={setSampler} options={[{ v: "ddpm", label: `DDPM (${T} steps)` }, { v: "ddim", label: "DDIM" }]} />
            {sampler === "ddim" && (
              <div className="min-w-0 flex-1 basis-48">
                <Slider label="DDIM steps" value={S} min={2} max={T} step={1} onChange={(v) => setS(Math.round(v))} fmt={(v) => String(v)} />
              </div>
            )}
            <button className="sim-btn" onClick={() => setSeed((q) => q + 1)} aria-label="Draw new starting noise">
              New noise (seed {seed})
            </button>
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input type="checkbox" checked={showX0} onChange={(e) => setShowX0(e.target.checked)} /> show predicted x̂₀
            </label>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={k === 0 ? "Start: pure Gaussian noise" : done ? "Done: generated samples" : `${sampler.toUpperCase()} step ${k} of ${nSteps}`}>
            {k === 0
              ? `x_T ~ N(0, I): ${N_POINTS} points with no information about the moons. Press Play — the trained network predicts the noise ε̂ in each point and the sampler removes part of it.`
              : done
                ? `All ${nSteps} steps taken. Energy distance to real data: ${fmt(ed, 4)} (two independent real samples of this size differ by ${fmt(edRef, 4)}; 0 = identical distributions).`
                : sampler === "ddpm"
                  ? `Now at t = ${tNow}. The step from t = ${tNow + 1} used ε̂ (RMS ${fmt(epsNorm, 3)}) to compute the posterior mean, then added fresh noise σ·z with σ = √β̃ = ${fmt(sigmaLast, 3)}.`
                  : `Now at t = ${tNow} (jumped from t = ${times[k - 1] + 1}). DDIM moves each point deterministically toward its predicted x̂₀ and re-noises it with the same ε̂ — no fresh noise.`}
          </Narration>
          <div>
            <Stat k="current t" v={tNow} />
            <Stat k="ᾱ_t" v={done ? "1" : fmt(model.sched.alphaBar[tNow - 1], 5)} />
            <Stat k="RMS of last ε̂" v={k ? fmt(epsNorm, 3) : "—"} />
            <Stat k="energy distance to data" v={fmt(ed, 4)} />
            <Stat k="real vs real (reference)" v={fmt(edRef, 4)} />
          </div>
          <div className="text-xs">
            <div className="mb-1 font-medium">Energy distance vs sampler (PyTorch, 4,000 samples each)</div>
            <table className="w-full font-mono tabular-nums">
              <tbody>
                {table.map(([q, v]) => (
                  <tr key={q}>
                    <td className="text-muted">DDIM {q}</td>
                    <td className="text-right">{fmt(v, 4)}</td>
                  </tr>
                ))}
                <tr>
                  <td className="text-muted">DDPM {T}</td>
                  <td className="text-right">{fmt(met.energy_ddpm_100, 4)}</td>
                </tr>
                <tr>
                  <td className="text-muted">real vs real</td>
                  <td className="text-right">{fmt(met.energy_real_vs_real, 4)}</td>
                </tr>
                <tr>
                  <td className="text-muted">N(0, I) noise</td>
                  <td className="text-right">{fmt(met.energy_noise, 4)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      }
    >
      <div className="grid min-w-0 gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Scatter pts={x} ghost={showX0 && lastX0 && !done ? lastX0 : null} label={`Samples at reverse step ${k}`} caption={<>Generated points (blue){showX0 ? ", the network's current guess x̂₀ (gold)" : ""}. t = {tNow}.</>} />
        <Scatter pts={done ? x : real} real={done ? real : undefined} label={done ? "Generated samples over real data" : "Real training distribution"} caption={done ? "Final comparison: generated (blue) over real data (grey)." : "Target: real two-moons data (what the model was trained on)."} />
      </div>
      <p className="mt-2 text-xs text-muted">
        <span className="font-medium text-accent">A real trained model.</span> ε-prediction MLP with {net.nParams.toLocaleString()} parameters ({net.json.hidden.join("–")} SiLU units, 32-d sinusoidal time embedding), trained in PyTorch for {net.json.training.steps.toLocaleString()} steps on 20,000 two-moons points, T = {T}, linear β schedule; final training loss {net.json.training.final_loss}.
      </p>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ (c) digits */

function DigitsTab({ model, net, tabs }: { model: DiffusionModel; net: Net; tabs: ReactNode }) {
  const T = model.sched.T;
  const [sampler, setSampler] = useState<"ddpm" | "ddim">("ddim");
  const [S, setS] = useState(25);
  const [w, setW] = useState(1);
  const [mode, setMode] = useState<"all" | number>("all");
  const [seed, setSeed] = useState(3);
  const N = 10;
  const cls = useMemo(() => (mode === "all" ? Array.from({ length: N }, (_, i) => i) : new Array(N).fill(mode)), [mode]);
  const { times, ensure } = useTrajectory(model, net, N, sampler, S, seed, cls, w);
  const nSteps = times.length;
  const s = useStepper(nSteps + 1, { baseMs: sampler === "ddpm" ? 120 : 400 });
  const { reset } = s;
  useEffect(() => reset(), [sampler, S, seed, w, mode, reset]);
  const k = Math.min(s.step, nSteps);
  const tr = ensure(k);
  const x = tr.xs[k];
  const x0 = k > 0 ? tr.x0[k - 1] : null;
  const done = k === nSteps;
  const tNow = done ? 0 : times[k] + 1;
  const met = net.json.metrics as Record<string, number>;
  const realLabels = model.json.data.digits_labels ?? [];
  const real = model.digitsReal;
  const realIdx = mode === "all" ? Array.from({ length: 10 }, (_, c) => realLabels.indexOf(c)) : realLabels.map((l, i) => (l === mode ? i : -1)).filter((i) => i >= 0);

  return (
    <SimShell
      title={TITLE}
      subtitle={SUBTITLE}
      controls={
        <div className="space-y-2">
          {tabs}
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Toggle label="sampler" value={sampler} onChange={setSampler} options={[{ v: "ddpm", label: `DDPM (${T})` }, { v: "ddim", label: "DDIM" }]} />
            {sampler === "ddim" && (
              <div className="min-w-0 flex-1 basis-48">
                <Slider label="DDIM steps" value={S} min={2} max={T} step={1} onChange={(v) => setS(Math.round(v))} fmt={(v) => String(v)} />
              </div>
            )}
            <div className="min-w-0 flex-1 basis-48">
              <Slider label="guidance w" value={w} min={0} max={5} step={0.5} onChange={setW} fmt={(v) => v.toFixed(1)} />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Class to generate">
            <span className="text-xs text-muted">class</span>
            <button className={`sim-btn ${mode === "all" ? "primary" : ""}`} aria-pressed={mode === "all"} onClick={() => setMode("all")}>
              0–9
            </button>
            {Array.from({ length: 10 }, (_, c) => (
              <button key={c} className={`sim-btn ${mode === c ? "primary" : ""}`} aria-pressed={mode === c} onClick={() => setMode(c)} aria-label={`Generate ten ${c}s`}>
                {c}
              </button>
            ))}
            <button className="sim-btn" onClick={() => setSeed((q) => q + 1)} aria-label="Draw new starting noise">
              New noise
            </button>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={k === 0 ? "Start: 10 images of pure noise" : done ? "Done" : `${sampler.toUpperCase()} step ${k} of ${nSteps} (t = ${tNow})`}>
            {k === 0
              ? `Each image is 64 independent N(0, 1) pixels. The network gets the class label c and, for guidance, is also run with the null label ∅; the two predictions are combined as ε̃ = (1+w)·ε(x, c) − w·ε(x, ∅) with w = ${w}.`
              : done
                ? `Generated in ${nSteps} network evaluations per image${w ? " (×2 for guidance)" : ""}. Compare with real digits below. Try w = 0 (no guidance): shapes get vaguer and some come out as the wrong class; large w gives bolder, more uniform digits.`
                : `Top row: the current noisy x_t. Bottom row: the network's guess of the finished image x̂₀ = (x_t − √(1−ᾱ_t)ε̃)/√ᾱ_t. The guess settles early (coarse shape) and later steps refine detail.`}
          </Narration>
          <div className="text-xs">
            <div className="mb-1 font-medium">Measured in PyTorch (300 images, 30 per class)</div>
            <p className="mb-1 text-muted">Share classified as the requested digit by a logistic-regression classifier trained on the real digits (which scores {fmt(met.classifier_train_acc_on_real * 100, 1)}% on them):</p>
            <Stat k="DDPM, w = 0" v={`${fmt(met.acc_ddpm_w0 * 100, 1)}%`} />
            <Stat k="DDPM, w = 1" v={`${fmt(met.acc_ddpm_w1 * 100, 1)}%`} />
            <Stat k="DDPM, w = 3" v={`${fmt(met.acc_ddpm_w3 * 100, 1)}%`} />
            <Stat k="DDIM 10 steps, w = 1" v={`${fmt(met.acc_ddim10_w1 * 100, 1)}%`} />
            <Stat k="DDIM 25 steps, w = 1" v={`${fmt(met.acc_ddim25_w1 * 100, 1)}%`} />
            <Stat k="DDIM 50 steps, w = 1" v={`${fmt(met.acc_ddim50_w1 * 100, 1)}%`} />
            <Stat k="nearest-train distance (w = 1)" v={fmt(met.nn_dist_ddpm_w1, 2)} />
            <Stat k="real → other real distance" v={fmt(met.nn_dist_real_to_other_real, 2)} />
          </div>
        </div>
      }
    >
      <div className="min-w-0 space-y-2">
        <div className="text-xs text-muted">x_t (current state)</div>
        <div className="grid grid-cols-5 gap-1.5 sm:grid-cols-10">
          {cls.map((c, i) => (
            <div key={i} className="min-w-0">
              <Digit px={x.subarray(i * 64, (i + 1) * 64)} size={56} label={`Sample ${i + 1}, class ${c}, step ${k}`} />
              <div className="text-center font-mono text-[10px] text-muted">{c}</div>
            </div>
          ))}
        </div>
        <div className="text-xs text-muted">predicted clean image x̂₀ at this step</div>
        <div className="grid grid-cols-5 gap-1.5 sm:grid-cols-10">
          {cls.map((c, i) => (
            <div key={i} className="min-w-0">
              {x0 ? <Digit px={x0.subarray(i * 64, (i + 1) * 64)} size={56} label={`Predicted clean image ${i + 1}`} /> : <div className="aspect-square w-full max-w-[56px] rounded-sm border border-dashed border-border" />}
            </div>
          ))}
        </div>
        {real && realIdx.length > 0 && (
          <>
            <div className="text-xs text-muted">real training digits (for comparison)</div>
            <div className="grid grid-cols-5 gap-1.5 sm:grid-cols-10">
              {realIdx.slice(0, 10).map((r, i) => (
                <Digit key={i} px={real.subarray(r * 64, (r + 1) * 64)} size={56} label={`Real digit ${realLabels[r]}`} />
              ))}
            </div>
          </>
        )}
        <p className="text-xs text-muted">
          <span className="font-medium text-accent">A real trained model.</span> Class-conditional ε-prediction MLP with {net.nParams.toLocaleString()} parameters ({net.json.hidden.join("–")} SiLU units), trained in PyTorch for {net.json.training.steps.toLocaleString()} steps on the 1,797 8×8 scikit-learn digits; the label was dropped 10% of the time so the same network also predicts unconditionally (classifier-free guidance). Final training loss {net.json.training.final_loss}. With so little data it partly memorises: compare the nearest-train distance with the real-to-real distance.
        </p>
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ main */

export default function DiffusionSim() {
  const [model, setModel] = useState<DiffusionModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<{ epsMaxDiff: number; ddimMaxDiff: number; checked: number } | null>(null);
  const [tab, setTab] = useState<Tab>("forward");

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<DiffusionJSON>;
      })
      .then((j) => {
        if (!alive) return;
        const m = loadModel(j);
        setModel(m);
        setCheck(verifyModel(m));
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <SimShell title={TITLE} subtitle={SUBTITLE}>
        <p className="text-sm text-clay">Could not load the model ({error}). Check your connection and reload the page.</p>
      </SimShell>
    );
  }
  if (!model) {
    return (
      <SimShell title={TITLE} subtitle={SUBTITLE}>
        <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
          Loading the trained models (≈350 KB of weights)…
        </div>
      </SimShell>
    );
  }

  const tabs = <TabBar tab={tab} setTab={setTab} hasDigits={!!model.digits} />;
  return (
    <div>
      {tab === "forward" && <ForwardTab model={model} tabs={tabs} />}
      {tab === "reverse" && <ReverseTab model={model} tabs={tabs} />}
      {tab === "digits" && model.digits && <DigitsTab model={model} net={model.digits} tabs={tabs} />}
      {check && (
        <p className="-mt-4 mb-6 px-1 text-[11px] text-faint">
          Browser engine vs PyTorch on stored verification vectors: max |Δε| = {check.epsMaxDiff.toExponential(1)}, max |Δx₀| over full DDIM trajectories = {check.ddimMaxDiff.toExponential(1)} ({check.checked} numbers checked).
        </p>
      )}
    </div>
  );
}
