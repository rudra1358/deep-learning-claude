"use client";

import { useMemo, useState, type ReactNode } from "react";
import { Shuffle, Users } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { NF, affine, emaUpdate, groupStats, sampleBatch, standardize, warmRunning, type Matrix, type NormMode } from "@/lib/sim/batchnorm";

const EPS = 1e-5;
const STAGES = ["Raw mini-batch", "Mean", "Variance", "Standardise", "Scale and shift (γ, β)", "Running statistics"] as const;
const FEATS = Array.from({ length: NF }, (_, d) => `x${d + 1}`);

type Preset = { label: string; shift: number[]; scale: number[] };
const PRESETS: Preset[] = [
  { label: "Badly scaled features", shift: [5, -40, 0, 20], scale: [0.5, 15, 3, 1] },
  { label: "Similar scales", shift: [0.5, -0.5, 0, 1], scale: [1, 1.5, 0.8, 1.2] },
  { label: "One huge feature", shift: [0, 0, 0, 0], scale: [1, 1, 1, 30] },
];

/** Compact number format for values spanning several orders of magnitude. */
const num = (v: number) => {
  if (!Number.isFinite(v)) return "NaN";
  const a = Math.abs(v);
  if (a < 1e-9) return "0.000";
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2);
  return v.toFixed(a >= 100 ? 1 : a >= 10 ? 2 : 3);
};

function range(values: number[], include: number[] = []): [number, number] {
  const all = [...values, ...include].filter(Number.isFinite);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  }
  const pad = 0.06 * (hi - lo);
  return [lo - pad, hi + pad];
}

function niceTicks(lo: number, hi: number, count = 6): number[] {
  const span = hi - lo;
  const raw = span / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * p).find((s) => s >= raw) ?? 10 * p;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.abs(t) < 1e-12 ? 0 : t);
  return out;
}

/* ------------------------------------------------------------------ */
/* strip plot: one strip per statistics group                          */
/* ------------------------------------------------------------------ */

const SW = 640;
const SL = 58;
const SR = 12;

interface Strip {
  label: string;
  values: { v: number; tag: string; tracked: boolean }[];
  mean?: number;
  std?: number;
  highlighted: boolean;
}

function StripPlot({ strips, lo, hi, ariaLabel, onPick }: { strips: Strip[]; lo: number; hi: number; ariaLabel: string; onPick: (i: number) => void }) {
  const rowH = Math.max(11, Math.min(44, 300 / strips.length));
  const top = 6;
  const H = top + rowH * strips.length + 26;
  const x = (v: number) => SL + ((v - lo) / (hi - lo)) * (SW - SL - SR);
  const ticks = niceTicks(lo, hi);
  const showTags = rowH >= 20;
  const r = Math.max(2.5, Math.min(5, rowH / 5));
  return (
    <svg viewBox={`0 0 ${SW} ${H}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={top + rowH * strips.length} stroke="var(--border)" strokeWidth={1} />
          <text x={x(t)} y={H - 10} textAnchor="middle" fontSize={10} fill="var(--text-muted)" fontFamily="var(--font-mono, monospace)">
            {num(t)}
          </text>
        </g>
      ))}
      {strips.map((s, i) => {
        const y0 = top + i * rowH;
        const cy = y0 + rowH / 2;
        return (
          <g key={i} onClick={() => onPick(i)} style={{ cursor: "pointer" }}>
            <rect x={0} y={y0} width={SW} height={rowH} fill={s.highlighted ? "color-mix(in srgb, var(--gold) 10%, transparent)" : "transparent"} />
            <text x={SL - 8} y={cy + 3.5} textAnchor="end" fontSize={Math.min(11, rowH - 1)} fill={s.highlighted ? "var(--text)" : "var(--text-muted)"} fontWeight={s.highlighted ? 600 : 400}>
              {s.label}
            </text>
            <line x1={SL} x2={SW - SR} y1={y0 + rowH} y2={y0 + rowH} stroke="var(--border)" strokeWidth={0.6} />
            {s.mean !== undefined && s.std !== undefined && Number.isFinite(s.std) && (
              <rect x={x(s.mean - s.std)} y={y0 + 2} width={Math.max(0, x(s.mean + s.std) - x(s.mean - s.std))} height={rowH - 4} fill="color-mix(in srgb, var(--accent) 16%, transparent)" />
            )}
            {s.mean !== undefined && Number.isFinite(s.mean) && <line x1={x(s.mean)} x2={x(s.mean)} y1={y0 + 1} y2={y0 + rowH - 1} stroke="var(--accent)" strokeWidth={2} />}
            {s.values.map((p, k) => {
              const jitter = s.values.length > 6 ? (((k * 37) % 9) - 4) * (rowH / 14) : 0;
              const px = Math.max(SL - 4, Math.min(SW - SR + 4, x(p.v)));
              return (
                <g key={k}>
                  <circle cx={px} cy={cy + jitter} r={p.tracked ? r + 1.2 : r} fill={p.tracked ? "var(--gold)" : "var(--text-muted)"} fillOpacity={p.tracked ? 1 : 0.55} stroke={p.tracked ? "var(--text)" : "none"} strokeWidth={0.8} />
                  {showTags && p.tag && (
                    <text x={px} y={cy + jitter - r - 2.5} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
                      {p.tag}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* matrix heat-map: which cells share statistics                       */
/* ------------------------------------------------------------------ */

function MatrixView({ M, mode, selFeat, selRow, onPick }: { M: Matrix; mode: NormMode; selFeat: number; selRow: number; onPick: (row: number, col: number) => void }) {
  const N = M.length;
  const cw = 66;
  const lw = 34;
  const rowH = Math.max(7, Math.min(22, 330 / N));
  const W = lw + cw * NF + 4;
  const H = 18 + rowH * N + 4;
  const maxAbs = Math.max(1e-9, ...M.flat().map((v) => Math.abs(v)).filter(Number.isFinite));
  const showNums = rowH >= 15;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full max-w-[360px]" role="img" aria-label={`Mini-batch matrix, ${N} examples by ${NF} features; highlighted cells share one mean and variance`}>
      {FEATS.map((f, d) => (
        <text key={f} x={lw + d * cw + cw / 2} y={12} textAnchor="middle" fontSize={11} fill={mode === "batch" && d === selFeat ? "var(--text)" : "var(--text-muted)"} fontWeight={mode === "batch" && d === selFeat ? 600 : 400}>
          {f}
        </text>
      ))}
      {M.map((row, i) => (
        <g key={i}>
          {rowH >= 10 && (
            <text x={lw - 5} y={18 + i * rowH + rowH / 2 + 3.5} textAnchor="end" fontSize={Math.min(10, rowH - 1)} fill={i === 0 ? "var(--gold)" : "var(--text-muted)"} fontWeight={i === 0 ? 600 : 400}>
              #{i + 1}
            </text>
          )}
          {row.map((v, d) => {
            const inGroup = mode === "batch" ? d === selFeat : i === selRow;
            return (
              <g key={d} onClick={() => onPick(i, d)} style={{ cursor: "pointer" }}>
                <rect x={lw + d * cw + 1} y={18 + i * rowH + 0.5} width={cw - 2} height={rowH - 1} fill={signedColor(v, maxAbs)} opacity={inGroup ? 1 : 0.45} />
                {showNums && (
                  <text x={lw + d * cw + cw / 2} y={18 + i * rowH + rowH / 2 + 3.5} textAnchor="middle" fontSize={10} fill="var(--text)" fontFamily="var(--font-mono, monospace)">
                    {num(v)}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      ))}
      {/* outline of the group that shares the selected statistics */}
      {mode === "batch" ? (
        <rect x={lw + selFeat * cw + 0.5} y={17.5} width={cw - 1} height={rowH * N + 1} fill="none" stroke="var(--gold)" strokeWidth={2} />
      ) : (
        <rect x={lw + 0.5} y={18 + selRow * rowH - 0.5} width={cw * NF - 1} height={rowH + 1} fill="none" stroke="var(--gold)" strokeWidth={2} />
      )}
      {/* tracked example marker */}
      <rect x={lw - 2} y={18} width={2} height={rowH} fill="var(--gold)" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* the simulator                                                       */
/* ------------------------------------------------------------------ */

export default function BatchNormSim() {
  const [mode, setMode] = useState<NormMode>("batch");
  const [phase, setPhase] = useState<"train" | "eval">("train");
  const [n, setN] = useState(8);
  const [shift, setShift] = useState<number[]>(PRESETS[0].shift);
  const [scale, setScale] = useState<number[]>(PRESETS[0].scale);
  const [gamma, setGamma] = useState<number[]>(() => new Array(NF).fill(1));
  const [beta, setBeta] = useState<number[]>(() => new Array(NF).fill(0));
  const [seed, setSeed] = useState(3);
  const [othersSeed, setOthersSeed] = useState(0);
  const [momentum, setMomentum] = useState(0.1);
  const [warm, setWarm] = useState(30);
  const [selFeat, setSelFeat] = useState(1);
  const [selRowRaw, setSelRow] = useState(0);
  const [prev, setPrev] = useState<{ key: string; y: number[] } | null>(null);

  const s = useStepper(STAGES.length, { baseMs: 1600 });
  const stage = s.step;
  const isBN = mode === "batch";
  const evalBN = isBN && phase === "eval";

  const spec = useMemo(() => ({ n, shift, scale, seed, othersSeed }), [n, shift, scale, seed, othersSeed]);
  const X = useMemo(() => sampleBatch(spec), [spec]);
  const running = useMemo(() => warmRunning({ n, shift, scale, seed, othersSeed: 0 }, warm, momentum), [n, shift, scale, seed, warm, momentum]);

  const batchStats = useMemo(() => groupStats(X, mode), [X, mode]);
  // the statistics actually used to normalise: batch (train BN), running (eval BN) or per-example (LN)
  const usedMean = evalBN ? running.mean : batchStats.mean;
  const usedVar = evalBN ? running.var : batchStats.var;
  const Xh = useMemo(() => standardize(X, mode, usedMean, usedVar, EPS), [X, mode, usedMean, usedVar]);
  const Y = useMemo(() => affine(Xh, gamma, beta), [Xh, gamma, beta]);
  const ema = useMemo(() => (isBN && !evalBN ? emaUpdate(running.mean, running.var, groupStats(X, "batch"), n, momentum) : null), [isBN, evalBN, running, X, n, momentum]);

  // batch-dependence bookkeeping: everything except the other examples' seed
  const depKey = JSON.stringify([mode, phase, n, shift, scale, gamma, beta, seed, warm, momentum]);
  const prevY = prev && prev.key === depKey ? prev.y : null;

  const selRow = Math.min(selRowRaw, n - 1); // clamp immediately when N shrinks

  const resampleOthers = () => {
    setPrev({ key: depKey, y: Y[0] });
    setOthersSeed((v) => v + 1);
  };

  const shown: Matrix = stage <= 2 ? X : stage === 3 ? Xh : Y;
  const g = isBN ? selFeat : selRow; // the statistics group being narrated
  const M = isBN ? n : NF; // values sharing one statistic

  /* ---------- strips ---------- */
  const strips: Strip[] = useMemo(() => {
    const out: Strip[] = [];
    const groups = isBN ? NF : n;
    // overlays: stage 1 → mean; stage ≥ 2 → mean ± std. Stages 1–2 show the statistics USED; later stages
    // show the actual mean/std of the transformed values (exactly 0/1 only for batch statistics).
    const after = stage >= 3 ? groupStats(shown, mode) : null;
    for (let k = 0; k < groups; k++) {
      const values = isBN
        ? shown.map((r, i) => ({ v: r[k], tag: n <= 8 ? `#${i + 1}` : "", tracked: i === 0 }))
        : shown[k].map((v, d) => ({ v, tag: FEATS[d], tracked: k === 0 }));
      let mean: number | undefined;
      let std: number | undefined;
      if (stage >= 1) {
        mean = after ? after.mean[k] : usedMean[k];
        if (stage >= 2) std = Math.sqrt(after ? after.var[k] : usedVar[k]);
      }
      out.push({ label: isBN ? FEATS[k] : `#${k + 1}`, values, mean, std, highlighted: k === g });
    }
    return out;
  }, [isBN, n, shown, stage, mode, usedMean, usedVar, g]);

  const [lo, hi] = range(shown.flat(), stage >= 3 ? [-2.5, 2.5] : []);

  /* ---------- narration ---------- */
  const groupVals = isBN ? X.map((r) => r[g]) : X[g];
  const listVals = (vals: number[]) => (vals.length <= 5 ? vals.map(num).join(" + ") : `${vals.slice(0, 3).map(num).join(" + ")} + … + ${num(vals[vals.length - 1])}`);
  const sum = groupVals.reduce((a, b) => a + b, 0);
  const tCol = isBN ? g : selFeat;
  const tx = X[isBN ? 0 : g][tCol];
  const sd = Math.sqrt(usedVar[g] + EPS);
  const gName = isBN ? `feature ${FEATS[g]}` : `example #${g + 1}`;

  let title = `${stage + 1}. ${STAGES[stage]}`;
  let body: ReactNode;
  if (stage === 0) {
    const cs = groupStats(X, "batch");
    body = (
      <>
        A mini-batch (a small group of examples) of <b className="text-text">N = {n}</b> examples × {NF} features. These are one layer&apos;s values before its activation. Spread of each feature: {FEATS.map((f, d) => `${f} σ≈${num(Math.sqrt(cs.var[d]))}`).join(", ")}. The gold dot is example #1; we will follow it. {isBN ? "BatchNorm will compute one mean and variance for each feature column (gold outline)." : "LayerNorm will compute one mean and variance for each example row (gold outline)."}
      </>
    );
  } else if (stage === 1) {
    body = evalBN ? (
      <>
        Eval mode: <b className="text-text">the batch&apos;s own statistics are not used</b>. The layer uses the running mean it collected during training: μ̂<sub>{FEATS[g]}</sub> = <b className="text-text">{num(usedMean[g])}</b> (after {warm} training batches; this batch&apos;s own mean would be {num(batchStats.mean[g])}).
      </>
    ) : (
      <>
        Mean (average) of {gName} over its {M} values: μ = ({listVals(groupVals)}) / {M} = {num(sum)} / {M} = <b className="text-text">{num(usedMean[g])}</b>.
      </>
    );
  } else if (stage === 2) {
    body = evalBN ? (
      <>
        Running variance σ̂²<sub>{FEATS[g]}</sub> = <b className="text-text">{num(usedVar[g])}</b> (this batch&apos;s own biased variance would be {num(batchStats.var[g])}). The shaded band is μ̂ ± σ̂.
      </>
    ) : (
      <>
        Variance tells how spread out the values are. Biased variance (divide by M = {M}): σ² = (1/{M}) Σ (x − {num(usedMean[g])})² = <b className="text-text">{num(usedVar[g])}</b>, so σ = {num(Math.sqrt(usedVar[g]))}. The band shows μ ± σ.
        {n === 1 && isBN ? " With one example, x − μ = 0 for every value, so σ² = 0." : ""}
      </>
    );
  } else if (stage === 3) {
    const after = groupStats(Xh, mode);
    body = (
      <>
        x̂ = (x − μ)/√(σ² + ε), ε = 10⁻⁵. For example #{isBN ? 1 : g + 1}, {FEATS[tCol]}: ({num(tx)} − {num(usedMean[g])}) / {num(sd)} = <b className="text-text">{num(Xh[isBN ? 0 : g][tCol])}</b>. The {gName} now has mean {num(after.mean[g])} and std {num(Math.sqrt(after.var[g]))}
        {evalBN ? ". Not exactly 0 and 1, because the running statistics are estimates, not this batch's own numbers." : "."}
        {isBN && !evalBN && n === 1 ? " With N = 1 every x̂ is 0, so the output says nothing about the input (PyTorch refuses to train on such a batch)." : ""}
      </>
    );
  } else if (stage === 4) {
    const d = isBN ? g : selFeat;
    const row = isBN ? 0 : g;
    body = (
      <>
        y = γ·x̂ + β, where γ (scale) and β (shift) are learned for each feature. For {FEATS[d]}: γ = {num(gamma[d])}, β = {num(beta[d])}, so example #{row + 1} gives {num(gamma[d])} × {num(Xh[row][d])} + {num(beta[d])} = <b className="text-text">{num(Y[row][d])}</b>. Setting γ = √(σ² + ε), β = μ would undo the normalisation exactly. So normalising never takes away what the layer can express.
      </>
    );
  } else {
    if (!isBN) {
      title = "6. No running statistics";
      body = <>LayerNorm computes its statistics from each example alone. So it keeps no running averages, and it behaves the same in training and inference. Example #1&apos;s output never depends on the other rows.</>;
    } else if (evalBN) {
      title = "6. Running averages are frozen";
      body = <>In eval mode the running mean and variance are only read, never updated. Each example&apos;s output is a fixed scale-and-shift of that example alone: y = γ(x − μ̂)/√(σ̂² + ε) + β. Resampling the other examples changes nothing.</>;
    } else if (ema) {
      const m = momentum;
      body = (
        <>
          PyTorch-style EMA (a running average; momentum m = {m} is how much the new batch counts):
          <br />
          running_mean ← (1 − {m})·{num(running.mean[g])} + {m}·{num(batchStats.mean[g])} = <b className="text-text">{num(ema.newMean[g])}</b>
          <br />
          {ema.newVar && ema.unbiasedVar ? (
            <>
              running_var ← (1 − {m})·{num(running.var[g])} + {m}·({n}/{n - 1})·{num(batchStats.var[g])} = (1 − {m})·{num(running.var[g])} + {m}·{num(ema.unbiasedVar[g])} = <b className="text-text">{num(ema.newVar[g])}</b>
              <br />
              Notice: the running average stores the unbiased variance N/(N−1)·σ², but the batch was normalised with the biased σ².
            </>
          ) : (
            <>running_var: undefined. The unbiased variance needs N ≥ 2 (PyTorch raises &quot;Expected more than 1 value per channel when training&quot;).</>
          )}
        </>
      );
    }
  }

  const prevDelta = prevY ? Math.max(...prevY.map((v, d) => Math.abs(v - Y[0][d]))) : null;

  const setFeat = (arr: number[], setter: (v: number[]) => void, v: number) => setter(arr.map((x, d) => (d === selFeat ? v : x)));

  return (
    <SimShell
      title="Batch Normalization, step by step"
      subtitle="Normalise a real mini-batch, per feature (BatchNorm) or per example (LayerNorm). See learnable γ/β, running averages, and train vs eval mode."
      controls={
        <div className="space-y-2.5">
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Normalisation type">
            <span className="w-16 text-xs text-muted">normalise</span>
            <button className={`sim-btn ${isBN ? "primary" : ""}`} aria-pressed={isBN} onClick={() => setMode("batch")}>
              BatchNorm
            </button>
            <button className={`sim-btn ${!isBN ? "primary" : ""}`} aria-pressed={!isBN} onClick={() => setMode("layer")}>
              LayerNorm
            </button>
            <span className="ml-2 w-10 text-xs text-muted">mode</span>
            <button className={`sim-btn ${phase === "train" ? "primary" : ""}`} aria-pressed={phase === "train"} onClick={() => setPhase("train")}>
              Train
            </button>
            <button className={`sim-btn ${phase === "eval" ? "primary" : ""}`} aria-pressed={phase === "eval"} onClick={() => setPhase("eval")}>
              Eval
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                className="sim-btn"
                onClick={() => {
                  setShift(p.shift);
                  setScale(p.scale);
                }}
              >
                {p.label}
              </button>
            ))}
            <button className="sim-btn" onClick={resampleOthers} aria-label="Resample the other examples in the batch, keeping example 1 fixed">
              <Users size={14} /> Resample other examples
            </button>
            <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Draw a completely new batch">
              <Shuffle size={14} /> New batch
            </button>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="batch size N" value={n} min={1} max={32} step={1} onChange={(v) => setN(v)} fmt={(v) => String(v)} />
            <Slider label="momentum m" value={momentum} min={0.01} max={0.5} step={0.01} onChange={setMomentum} />
            <Slider label="past batches" value={warm} min={0} max={100} step={1} onChange={(v) => setWarm(v)} fmt={(v) => String(v)} />
          </div>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Feature to edit">
            <span className="w-16 text-xs text-muted">feature</span>
            {FEATS.map((f, d) => (
              <button key={f} className={`sim-btn ${selFeat === d ? "primary" : ""}`} aria-pressed={selFeat === d} onClick={() => setSelFeat(d)}>
                {f}
              </button>
            ))}
            <button
              className="sim-btn"
              disabled={!isBN || evalBN}
              onClick={() => {
                setGamma(batchStats.var.map((v) => Math.sqrt(v + EPS)));
                setBeta(batchStats.mean.slice());
              }}
              aria-label="Set gamma to the batch standard deviation and beta to the batch mean"
            >
              γ = σ, β = μ (undo)
            </button>
            <button
              className="sim-btn"
              onClick={() => {
                setGamma(new Array(NF).fill(1));
                setBeta(new Array(NF).fill(0));
              }}
            >
              γ = 1, β = 0
            </button>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label={`shift of ${FEATS[selFeat]}`} value={shift[selFeat]} min={-50} max={50} step={0.5} onChange={(v) => setFeat(shift, setShift, v)} fmt={(v) => v.toFixed(1)} />
            <Slider label={`spread of ${FEATS[selFeat]}`} value={Math.log10(scale[selFeat])} min={-1} max={1.5} step={0.01} onChange={(v) => setFeat(scale, setScale, +(10 ** v).toPrecision(3))} fmt={() => num(scale[selFeat])} />
            <Slider label={`γ of ${FEATS[selFeat]}`} value={Math.max(-3, Math.min(30, gamma[selFeat]))} min={-3} max={30} step={0.05} onChange={(v) => setFeat(gamma, setGamma, v)} fmt={() => num(gamma[selFeat])} />
            <Slider label={`β of ${FEATS[selFeat]}`} value={Math.max(-50, Math.min(50, beta[selFeat]))} min={-50} max={50} step={0.1} onChange={(v) => setFeat(beta, setBeta, v)} fmt={() => num(beta[selFeat])} />
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={title}>{body}</Narration>
          <div>
            <div className="mb-1 text-xs font-medium">Statistics of {gName}</div>
            <table className="w-full border-collapse font-mono text-xs">
              <tbody>
                {(
                  [
                    [evalBN ? "used mean (running)" : "batch/group mean μ", num(usedMean[g])],
                    [evalBN ? "used var (running)" : "biased variance σ²", num(usedVar[g])],
                    ["values per statistic", isBN ? String(n) : String(NF)],
                    ...(isBN
                      ? [
                          ["running_mean (before)", num(running.mean[g])],
                          ["running_var (before)", num(running.var[g])],
                          ...(ema ? [["running_mean (after)", num(ema.newMean[g])], ["running_var (after)", ema.newVar ? num(ema.newVar[g]) : "undefined"]] : []),
                        ]
                      : [["running stats", "none"]]),
                  ] as [string, string][]
                ).map(([k, v]) => (
                  <tr key={k} className="border-b border-border/60">
                    <td className="py-1 pr-2 font-sans text-muted">{k}</td>
                    <td className="py-1 text-right">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="rounded-lg border border-border p-2.5 text-xs">
            <div className="mb-1 font-medium">Does the batch matter? Output y of example #1</div>
            <div className="font-mono">now: [{Y[0].map((v) => num(v)).join(", ")}]</div>
            {prevY ? (
              <>
                <div className="font-mono text-muted">before: [{prevY.map((v) => num(v)).join(", ")}]</div>
                <div className="mt-1">
                  max change: <b className={prevDelta !== null && prevDelta > 1e-9 ? "text-clay" : "text-accent"}>{num(prevDelta ?? 0)}</b>{" "}
                  {prevDelta !== null && prevDelta > 1e-9
                    ? "— example #1 did not change, but its output did. The other examples changed the batch statistics."
                    : "— the rest of the batch had no effect."}
                </div>
              </>
            ) : (
              <div className="mt-1 text-muted">Press “Resample other examples”. Example #1 stays the same; the rest of the batch is redrawn.</div>
            )}
          </div>
        </div>
      }
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
          <span className="font-medium text-text">
            {stage <= 2 ? "Raw values x" : stage === 3 ? "Standardised x̂" : "Output y = γx̂ + β"} — one strip per {isBN ? "feature (stats over the batch)" : "example (stats over its features)"}
          </span>
          <span>line = mean, band = ±1 std · gold = example #1 · click a strip to look closer</span>
        </div>
        <StripPlot
          strips={strips}
          lo={lo}
          hi={hi}
          ariaLabel={`Strip plot of ${stage <= 2 ? "raw" : stage === 3 ? "standardised" : "output"} values, ${isBN ? "one strip per feature" : "one strip per example"}`}
          onPick={(i) => (isBN ? setSelFeat(i) : setSelRow(i))}
        />
        <div className="flex flex-wrap items-start gap-4">
          <div className="min-w-0 flex-1 basis-[260px]">
            <div className="mb-1 text-xs text-muted">
              <span className="font-medium text-text">Mini-batch matrix</span> (rows = examples, columns = features). Gold outline = cells that share one mean and variance in {isBN ? "BatchNorm (a column)" : "LayerNorm (a row)"}.
            </div>
            <MatrixView
              M={shown}
              mode={mode}
              selFeat={selFeat}
              selRow={selRow}
              onPick={(row, col) => {
                setSelFeat(col);
                setSelRow(row);
              }}
            />
          </div>
          <div className="min-w-0 flex-1 basis-[220px] text-xs text-muted">
            <p className="mb-2">
              <span className="font-medium text-text">Current setting:</span> {isBN ? "BatchNorm" : "LayerNorm"}, {phase} mode, N = {n}
              {isBN && phase === "eval" ? `, running statistics from ${warm} earlier batches` : ""}.
            </p>
            {isBN && n === 1 && phase === "train" && <p className="mb-2 text-clay">N = 1 in train mode: every x̂ is 0, so y = β whatever the input is. Switch to Eval to see the running statistics fix it.</p>}
            {isBN && n === 1 && phase === "eval" && <p className="mb-2 text-accent">N = 1 works in eval mode, because the running statistics do not depend on the batch.</p>}
            <p>Try this: set N = 2 in Train mode. Each BatchNorm column becomes ±1 (up to ε), because two values always sit exactly one σ either side of their mean.</p>
          </div>
        </div>
        <p className="text-xs text-faint">ε = {EPS}. Running averages start at mean 0, variance 1 (PyTorch defaults). Then {warm} earlier batches from the same data update them before this one.</p>
      </div>
    </SimShell>
  );
}
