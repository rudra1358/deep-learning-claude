"use client";

import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { BATCH, FLOW_ACTS, histogram, perLayerFactor, runFlow, type FlowAct, type FlowConfig, type FlowInit, type FlowLoss } from "@/lib/sim/gradientFlow";

/* ------------------------------------------------------------------ */
/* formatting helpers                                                  */
/* ------------------------------------------------------------------ */

/** Compact scientific format for quantities spanning many orders of magnitude. */
const sci = (v: number) => {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "NaN" : "∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1000) return v.toPrecision(3);
  return v.toExponential(2);
};
const pct = (v: number) => `${(100 * v).toFixed(v > 0 && v < 0.01 ? 2 : 1)}%`;

const INIT_LABEL: Record<FlowInit, string> = { normal: "Normal σ", xavier: "Xavier", he: "He" };
const LOSS_LABEL: Record<FlowLoss, string> = { mse: "MSE loss", random: "Random upstream ∂L/∂h" };

type Preset = { label: string; cfg: Partial<FlowConfig> };
const BASE: Omit<FlowConfig, "seed"> = { depth: 30, width: 64, act: "sigmoid", init: "xavier", sigma: 0.1, residual: false, layerNorm: false, loss: "mse" };
const PRESETS: Preset[] = [
  { label: "Sigmoid, Xavier (vanishing)", cfg: { act: "sigmoid", init: "xavier" } },
  { label: "ReLU, He (healthy)", cfg: { act: "relu", init: "he" } },
  { label: "ReLU, σ = 0.5 (exploding)", cfg: { act: "relu", init: "normal", sigma: 0.5 } },
  { label: "Tanh, σ = 0.05 (collapse)", cfg: { act: "tanh", init: "normal", sigma: 0.05 } },
  { label: "Sigmoid + residual", cfg: { act: "sigmoid", init: "xavier", residual: true } },
  { label: "ReLU σ = 0.5 + LayerNorm", cfg: { act: "relu", init: "normal", sigma: 0.5, layerNorm: true } },
];

/* ------------------------------------------------------------------ */
/* log-scale per-layer bar chart                                       */
/* ------------------------------------------------------------------ */

const CW = 640;
const CH = 178;
const ML = 46;
const MR = 8;
const MT = 12;
const MB = 28;

function logRange(values: number[]): [number, number] {
  const pos = values.filter((v) => Number.isFinite(v) && v > 0);
  if (!pos.length) return [-1, 1];
  let lo = Math.floor(Math.log10(Math.min(...pos)));
  let hi = Math.ceil(Math.log10(Math.max(...pos)));
  if (hi - lo < 2) {
    lo -= 1;
    hi += 1;
  }
  return [lo, hi];
}

function LogBars({
  title,
  values,
  line,
  lineLabel,
  color,
  shown,
  current,
  selected,
  onSelect,
  ariaLabel,
}: {
  title: string;
  values: number[];
  line?: number[];
  lineLabel?: string;
  color: string;
  shown: (l: number) => boolean;
  current: number | null;
  selected: number;
  onSelect: (l: number) => void;
  ariaLabel: string;
}) {
  const L = values.length;
  const [lo, hi] = logRange(line ? [...values, ...line] : values);
  const plotW = CW - ML - MR;
  const plotH = CH - MT - MB;
  const bw = plotW / L;
  const y = (v: number) => MT + ((hi - Math.log10(v)) / (hi - lo)) * plotH;
  const yBottom = MT + plotH;
  const tickStep = Math.max(1, Math.ceil((hi - lo) / 6));
  const ticks: number[] = [];
  for (let k = hi; k >= lo; k -= tickStep) ticks.push(k);
  const labelEvery = Math.max(1, Math.ceil(L / 12));

  const linePts = line
    ?.map((v, i) => (shown(i + 1) && Number.isFinite(v) && v > 0 ? `${(ML + (i + 0.5) * bw).toFixed(1)},${y(v).toFixed(1)}` : null))
    .filter(Boolean)
    .join(" ");

  return (
    <figure className="m-0">
      <figcaption className="mb-1 flex flex-wrap items-center gap-x-3 text-xs text-muted">
        <span className="font-medium text-text">{title}</span>
        {line && (
          <span className="flex items-center gap-1">
            <svg width="22" height="8" aria-hidden="true">
              <line x1="1" y1="4" x2="21" y2="4" stroke="var(--text-muted)" strokeWidth={1.6} strokeDasharray="4 3" />
            </svg>
            {lineLabel}
          </span>
        )}
        <span className="text-faint">log scale · click a bar to look at that layer</span>
      </figcaption>
      <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
        {ticks.map((k) => (
          <g key={k}>
            <line x1={ML} x2={CW - MR} y1={MT + ((hi - k) / (hi - lo)) * plotH} y2={MT + ((hi - k) / (hi - lo)) * plotH} stroke="var(--border)" strokeWidth={1} />
            <text x={ML - 5} y={MT + ((hi - k) / (hi - lo)) * plotH + 3.5} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontFamily="var(--font-mono, monospace)">
              {k === 0 ? "1" : `1e${k}`}
            </text>
          </g>
        ))}
        {values.map((v, i) => {
          const l = i + 1;
          const vis = shown(l);
          const x = ML + i * bw + Math.min(1.5, bw * 0.12);
          const w = Math.max(1, bw - 2 * Math.min(1.5, bw * 0.12));
          const finite = Number.isFinite(v);
          const top = finite ? (v > 0 ? Math.max(MT, Math.min(yBottom, y(v))) : yBottom) : MT;
          const isCur = current === l;
          const isSel = selected === l;
          return (
            <g key={l} onClick={() => onSelect(l)} style={{ cursor: "pointer" }}>
              <title>{`layer ${l}: ${sci(v)}`}</title>
              <rect x={ML + i * bw} y={MT} width={bw} height={plotH} fill="transparent" />
              <rect
                x={x}
                y={top}
                width={w}
                height={Math.max(0, yBottom - top)}
                fill={finite ? color : "none"}
                stroke={isCur ? "var(--gold)" : finite ? "none" : color}
                strokeWidth={isCur ? 2 : 1}
                strokeDasharray={finite ? undefined : "3 2"}
                opacity={vis ? 0.9 : 0.16}
              />
              {isSel && <rect x={ML + i * bw} y={yBottom + 2} width={bw} height={3} fill="var(--gold)" />}
            </g>
          );
        })}
        {linePts && <polyline points={linePts} fill="none" stroke="var(--text-muted)" strokeWidth={1.6} strokeDasharray="4 3" />}
        <line x1={ML} x2={CW - MR} y1={yBottom} y2={yBottom} stroke="var(--text-muted)" strokeWidth={1} />
        {values.map((_, i) =>
          (i + 1) % labelEvery === 0 || i === 0 ? (
            <text key={i} x={ML + (i + 0.5) * bw} y={yBottom + 16} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
              {i + 1}
            </text>
          ) : null,
        )}
        <text x={CW - MR} y={CH - 1} textAnchor="end" fontSize={10} fill="var(--text-faint)">
          layer l
        </text>
      </svg>
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* histogram of activations at one layer                               */
/* ------------------------------------------------------------------ */

const HW = 640;
const HH = 140;

function ActHistogram({ values, range, title }: { values: Float64Array; range?: [number, number]; title: string }) {
  const BINS = 32;
  const { counts, lo, hi } = histogram(values, BINS, range);
  const max = Math.max(1, ...counts);
  const pl = 10;
  const pr = 10;
  const pt = 8;
  const pb = 24;
  const bw = (HW - pl - pr) / BINS;
  const ph = HH - pt - pb;
  const zeroX = lo < 0 && hi > 0 ? pl + ((0 - lo) / (hi - lo)) * (HW - pl - pr) : null;
  return (
    <figure className="m-0">
      <figcaption className="mb-1 text-xs text-muted">
        <span className="font-medium text-text">{title}</span> <span className="text-faint">({values.length} values = {BATCH} examples × width)</span>
      </figcaption>
      <svg viewBox={`0 0 ${HW} ${HH}`} className="h-auto w-full" role="img" aria-label={`${title}: histogram of activation values`}>
        {counts.map((c, i) => {
          const h = (c / max) * ph;
          return <rect key={i} x={pl + i * bw + 0.75} y={pt + ph - h} width={Math.max(0.5, bw - 1.5)} height={h} fill="var(--accent)" opacity={0.8} />;
        })}
        <line x1={pl} x2={HW - pr} y1={pt + ph} y2={pt + ph} stroke="var(--text-muted)" />
        {zeroX !== null && <line x1={zeroX} x2={zeroX} y1={pt} y2={pt + ph} stroke="var(--text-faint)" strokeDasharray="3 3" />}
        <text x={pl} y={HH - 6} fontSize={10} fill="var(--text-muted)">
          {sci(lo)}
        </text>
        {zeroX !== null && (
          <text x={zeroX} y={HH - 6} fontSize={10} textAnchor="middle" fill="var(--text-muted)">
            0
          </text>
        )}
        <text x={HW - pr} y={HH - 6} fontSize={10} textAnchor="end" fill="var(--text-muted)">
          {sci(hi)}
        </text>
      </svg>
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* the simulator                                                       */
/* ------------------------------------------------------------------ */

export default function GradientFlowSim() {
  const [cfg, setCfg] = useState<FlowConfig>({ ...BASE, seed: 1 });
  const deferred = useDeferredValue(cfg);
  const res = useMemo(() => runFlow(deferred), [deferred]);
  const L = res.cfg.depth;
  const total = 2 * L + 2;
  const s = useStepper(total, { baseMs: Math.max(140, Math.round(9000 / total)) });
  const { reset } = s;
  const step = Math.min(s.step, total - 1);
  const [selected, setSelected] = useState(1);

  // Any change of configuration restarts the animation (charts stay visible as faint outlines).
  useEffect(() => reset(), [res, reset]);
  useEffect(() => setSelected((v) => Math.min(Math.max(1, v), L)), [L]);

  // phase bookkeeping
  const phase: "idle" | "forward" | "backward" | "done" = step === 0 ? "idle" : step <= L ? "forward" : step <= 2 * L ? "backward" : "done";
  const fwdCount = phase === "idle" ? 0 : phase === "forward" ? step : L;
  const bwdCount = phase === "backward" ? step - L : phase === "done" ? L : 0;
  const curLayer = phase === "forward" ? step : phase === "backward" ? L - bwdCount + 1 : null;

  // During the forward sweep the histogram follows the layer being computed.
  useEffect(() => {
    if (phase === "forward" && curLayer !== null) setSelected(curLayer);
  }, [phase, curLayer]);

  const set = (patch: Partial<FlowConfig>) => setCfg((c) => ({ ...c, ...patch }));
  const A = FLOW_ACTS[res.cfg.act];
  const isRelu = res.cfg.act === "relu" || res.cfg.act === "leaky";
  const layers = res.layers;
  const sel = layers[Math.min(selected, L) - 1];
  const gW = layers.map((x) => x.gradWNorm);
  const dN = layers.map((x) => x.deltaNorm);
  const hStd = layers.map((x) => x.hStd);
  const factor = perLayerFactor(res.ratioW, L);
  const stale = deferred !== cfg;

  const rels = layers.map((x) => x.relGrad);
  const maxRel = Math.max(...rels);
  const minRel = Math.min(...rels);

  const verdict = (r: number) => {
    if (!Number.isFinite(r) || !Number.isFinite(maxRel)) return "the numbers overflowed: the signal grew too big for the computer to store.";
    if (r < 1e-3) return "vanishing. The first layer gets a gradient over 1000× smaller than the last, so it learns far more slowly (or not at all).";
    if (r > 1e3) return "exploding. The first layer's gradient is over 1000× the last layer's. A learning rate that suits one layer would wreck the other.";
    if (maxRel > 100)
      return `exploding in size. The layers are balanced, but ‖∂L/∂W‖ is up to ${sci(maxRel)}× ‖W‖. So even a small learning rate would wipe out the weights in one step. (Compare the δ ratio: δ grows going backward while the activations shrink, and the two cancel in δ·hᵀ.)`;
    if (maxRel < 1e-6)
      return `vanishing everywhere. The forward signal itself died out, so every layer's ‖∂L/∂W‖ is below ${sci(maxRel)}× ‖W‖. The network barely learns at any depth.`;
    return `healthy. All layers get gradients within 1000× of each other, and ‖∂L/∂W‖/‖W‖ lies between ${sci(minRel)} and ${sci(maxRel)}.`;
  };

  /* ---------------- narration ---------------- */
  let nTitle = "";
  let nBody: ReactNode = null;
  if (phase === "idle") {
    nTitle = "Setup";
    nBody = (
      <>
        A batch of {BATCH} inputs x ~ N(0, 1) goes into a {L}-layer MLP of width {res.cfg.width} ({A.label}
        {res.cfg.residual ? ", residual connections" : ""}
        {res.cfg.layerNorm ? ", LayerNorm before each activation" : ""}). Every weight is drawn from N(0, {fmt(res.std, 4)}²) ({INIT_LABEL[res.cfg.init]}). Press <strong>Play</strong> or <strong>Step</strong>. First the forward pass fills the top chart left → right. Then the backward pass fills the bottom chart right → left.
      </>
    );
  } else if (phase === "forward" && curLayer !== null) {
    const st = layers[curLayer - 1];
    nTitle = `Forward pass: layer ${curLayer} of ${L}`;
    nBody = (
      <>
        z = W<sup>[{curLayer}]</sup>h<sup>[{curLayer - 1}]</sup>
        {res.cfg.layerNorm ? ", normalised per example," : ""} then a = f(z)
        {res.cfg.residual ? ` and h = h[${curLayer - 1}] + a` : ""}. The std of h<sup>[{curLayer}]</sup> is <strong className="text-text">{sci(st.hStd)}</strong> (input: {fmt(res.inputStd, 3)}).{" "}
        {isRelu
          ? `${pct(st.deadUnits)} of the units are zero for every example in the batch (dead for this batch). The average slope f′ is ${fmt(st.meanSlope, 3)}.`
          : res.cfg.act === "linear"
            ? "The activation is linear, so f′ = 1 everywhere. Only the weights change the signal's size."
            : `${pct(st.lowSlope)} of the values sit where f′ is below 10% of its max (saturated, the curve is flat there). The average slope f′ is ${fmt(st.meanSlope, 3)} (max ${A.dfMax}).`}
      </>
    );
  } else if (phase === "backward" && curLayer !== null) {
    const st = layers[curLayer - 1];
    const rel = st.gradWNorm / gW[L - 1];
    nTitle = `Backward pass: layer ${curLayer}`;
    nBody = (
      <>
        δ<sup>[{curLayer}]</sup> = (W<sup>[{curLayer + 1}]ᵀ</sup>δ<sup>[{curLayer + 1}]</sup>) ⊙ f′(z<sup>[{curLayer}]</sup>){res.cfg.residual ? " + the identity path" : ""}. ‖δ<sup>[{curLayer}]</sup>‖ = <strong className="text-text">{sci(st.deltaNorm)}</strong>, ‖∂L/∂W<sup>[{curLayer}]</sup>‖ = <strong className="text-text">{sci(st.gradWNorm)}</strong>. Compared with the last layer, this layer&apos;s weight gradient is <strong className="text-text">{sci(rel)}×</strong> as big.
      </>
    );
    if (curLayer === L) {
      nBody = (
        <>
          The loss ({res.cfg.loss === "mse" ? `MSE = ${sci(res.loss)}` : "random upstream gradient"}) gives ∂L/∂h<sup>[{L}]</sup>; multiplying by f′ gives δ<sup>[{L}]</sup> with ‖δ<sup>[{L}]</sup>‖ = <strong className="text-text">{sci(st.deltaNorm)}</strong> and ‖∂L/∂W<sup>[{L}]</sup>‖ = <strong className="text-text">{sci(st.gradWNorm)}</strong>. Every earlier layer is compared with this one.
        </>
      );
    }
  } else {
    nTitle = "Result";
    nBody = (
      <>
        ‖∂L/∂W<sup>[1]</sup>‖ / ‖∂L/∂W<sup>[{L}]</sup>‖ = <strong className="text-text">{sci(res.ratioW)}</strong> and ‖δ<sup>[1]</sup>‖ / ‖δ<sup>[{L}]</sup>‖ = <strong className="text-text">{sci(res.ratioDelta)}</strong>
        {Number.isFinite(factor) ? <>, an average factor of {fmt(factor, 3)} per layer</> : null}. Verdict: {verdict(res.ratioW)}
      </>
    );
  }

  const shownFwd = (l: number) => l <= fwdCount;
  const shownBwd = (l: number) => l > L - bwdCount;

  return (
    <SimShell
      title="Gradient Flow Through Depth"
      subtitle="A real forward and backward pass through a deep MLP. Top: how big the signal is at each layer. Bottom: how big the gradient is at each layer."
      controls={
        <div className="space-y-2.5">
          <Controls s={s} />
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Presets">
            {PRESETS.map((p) => (
              <button key={p.label} className="sim-btn" onClick={() => setCfg((c) => ({ ...BASE, seed: c.seed, ...p.cfg }))}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Activation function">
            <span className="w-20 text-xs text-muted">activation</span>
            {(Object.keys(FLOW_ACTS) as FlowAct[]).map((a) => (
              <button key={a} className={`sim-btn ${cfg.act === a ? "primary" : ""}`} aria-pressed={cfg.act === a} onClick={() => set({ act: a })}>
                {FLOW_ACTS[a].label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Initialisation">
            <span className="w-20 text-xs text-muted">init</span>
            {(Object.keys(INIT_LABEL) as FlowInit[]).map((k) => (
              <button key={k} className={`sim-btn ${cfg.init === k ? "primary" : ""}`} aria-pressed={cfg.init === k} onClick={() => set({ init: k })}>
                {INIT_LABEL[k]}
              </button>
            ))}
            <span className="font-mono text-xs text-faint">std = {fmt(res.std, 4)}</span>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-3">
            <Slider label="depth L" value={cfg.depth} min={2} max={50} step={1} onChange={(v) => set({ depth: v })} fmt={(v) => String(v)} />
            <Slider label="width n" value={cfg.width} min={8} max={64} step={1} onChange={(v) => set({ width: v })} fmt={(v) => String(v)} />
            <div className={cfg.init === "normal" ? "" : "opacity-40"}>
              <Slider label="σ (Normal)" value={Math.log10(cfg.sigma)} min={-2} max={0.3} step={0.01} onChange={(v) => set({ sigma: +(10 ** v).toPrecision(3), init: "normal" })} fmt={() => cfg.sigma.toPrecision(2)} />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={cfg.residual} onChange={(e) => set({ residual: e.target.checked })} aria-label="Residual connections" /> residual connections
            </label>
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={cfg.layerNorm} onChange={(e) => set({ layerNorm: e.target.checked })} aria-label="Layer normalisation" /> layer normalisation
            </label>
            <label className="flex items-center gap-1.5">
              <span className="text-muted">loss</span>
              <select className="rounded border border-border bg-surface px-1.5 py-0.5" value={cfg.loss} onChange={(e) => set({ loss: e.target.value as FlowLoss })} aria-label="Loss / upstream gradient">
                {(Object.keys(LOSS_LABEL) as FlowLoss[]).map((k) => (
                  <option key={k} value={k}>
                    {LOSS_LABEL[k]}
                  </option>
                ))}
              </select>
            </label>
            <button className="sim-btn" onClick={() => set({ seed: cfg.seed + 1 })} aria-label="Draw a new random batch and weights">
              <Shuffle size={14} /> New sample
            </button>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={nTitle}>{nBody}</Narration>
          <table className="w-full border-collapse font-mono text-xs">
            <tbody>
              {[
                ["std of input x", fmt(res.inputStd, 3)],
                ["std of h[1]", sci(hStd[0])],
                [`std of h[${L}]`, sci(hStd[L - 1])],
                ["‖∂L/∂W[1]‖", sci(gW[0])],
                [`‖∂L/∂W[${L}]‖`, sci(gW[L - 1])],
                ["first / last (W)", sci(res.ratioW)],
                ["first / last (δ)", sci(res.ratioDelta)],
                ["per-layer factor", Number.isFinite(factor) ? fmt(factor, 3) : "—"],
                ["max ‖∂L/∂W‖/‖W‖", sci(maxRel)],
                ...(res.cfg.loss === "mse" ? [["MSE loss", sci(res.loss)]] : []),
              ].map(([k, v]) => (
                <tr key={k} className="border-b border-border/60">
                  <td className="py-1 pr-2 font-sans text-muted">{k}</td>
                  <td className="py-1 text-right">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-faint">
            Biases are zero; batch = {BATCH}. ‖·‖ is the Frobenius norm (overall size) over the batch. δ<sup>[l]</sup> = ∂L/∂z<sup>[l]</sup>. Leaky ReLU slope {0.01}. {stale ? "Updating…" : ""}
          </p>
        </div>
      }
    >
      <div className="space-y-4">
        <LogBars
          title="Forward: std (spread) of each layer's output h[l]"
          values={hStd}
          color="var(--accent)"
          shown={shownFwd}
          current={phase === "forward" ? curLayer : null}
          selected={selected}
          onSelect={setSelected}
          ariaLabel={`Bar chart of activation standard deviation for ${L} layers, log scale`}
        />
        <LogBars
          title="Backward: ‖∂L/∂W[l]‖ (bars)"
          values={gW}
          line={dN}
          lineLabel="‖δ[l]‖"
          color="var(--clay)"
          shown={shownBwd}
          current={phase === "backward" ? curLayer : null}
          selected={selected}
          onSelect={setSelected}
          ariaLabel={`Bar chart of weight-gradient norm for ${L} layers, log scale`}
        />
        <div>
          <ActHistogram values={sel.acts} range={A.range} title={`Activations a[${Math.min(selected, L)}] = f(z) at layer ${Math.min(selected, L)}`} />
          <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
            <div className="min-w-[220px] flex-1">
              <Slider label="inspect layer" value={Math.min(selected, L)} min={1} max={L} step={1} onChange={(v) => setSelected(v)} fmt={(v) => String(v)} />
            </div>
            <span className="font-mono text-xs text-muted">
              {isRelu ? `${res.cfg.act === "leaky" ? "units negative for all examples" : "dead units"} ${pct(sel.deadUnits)} · ` : ""}
              {res.cfg.act !== "linear" ? `f′ < 10% of max: ${pct(sel.lowSlope)} · ` : ""}
              mean f′ {fmt(sel.meanSlope, 3)}
            </span>
          </div>
        </div>
      </div>
    </SimShell>
  );
}
