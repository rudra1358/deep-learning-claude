"use client";

import { useMemo, useState } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  N_FEATURES,
  argmax,
  gridLevels,
  logits,
  makeModel,
  quantizeMatrix,
  traceValue,
  type Granularity,
  type QMode,
} from "@/lib/sim/quantization";

const STAGES = [
  "Float weights",
  "Calibrate the range",
  "Scale s",
  "Zero-point z",
  "Divide by s",
  "Add z and round",
  "Clip to the integer grid",
  "Dequantize",
  "Error and model output",
] as const;

const BITS = [8, 4, 3, 2] as const;

function Seg<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: { v: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="w-24 shrink-0 text-muted">{label}</span>
      {options.map((o) => (
        <button key={String(o.v)} className={`sim-btn ${o.v === value ? "primary" : ""}`} aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

const db = (v: number) => (v === Infinity ? "∞" : v === -Infinity ? "−∞" : v.toFixed(2));
const sgn = (v: number, d = 3) => fmt(v, d).replace("-", "−");

/* ------------------------------------------------------------------ */
/* number lines: one per weight row (output channel)                    */
/* ------------------------------------------------------------------ */

const VW = 640;
const PADL = 14;
const PADR = 14;
const ROWH = 92;

export default function QuantizationSim() {
  const [seed, setSeed] = useState(7);
  const [bits, setBits] = useState<number>(4);
  const [mode, setMode] = useState<QMode>("symmetric");
  const [gran, setGran] = useState<Granularity>("per-tensor");
  const [clip, setClip] = useState(1);
  const [selRow, setSelRow] = useState(0);
  const [selCol, setSelCol] = useState(2);
  const s = useStepper(STAGES.length, { baseMs: 1300 });
  const stage = s.step;

  const model = useMemo(() => makeModel(seed), [seed]);
  const qm = useMemo(() => quantizeMatrix(model.W, { bits, mode, granularity: gran, clipRatio: clip }), [model, bits, mode, gran, clip]);
  const p = qm.params[selRow];
  const x = model.W[selRow][selCol];
  const tr = traceValue(x, p);
  const rowVals = model.W[selRow];
  const rowMin = Math.min(...rowVals);
  const rowMax = Math.max(...rowVals);
  const rangeVals = gran === "per-tensor" ? model.W.flat() : rowVals;
  const rMin = Math.min(...rangeVals);
  const rMax = Math.max(...rangeVals);

  const outputs = model.X.map((pt) => {
    const lf = logits(model.W, model.b, pt);
    const lq = logits(qm.deq, model.b, pt);
    return { lf, lq, cf: argmax(lf), cq: argmax(lq), diff: Math.max(...lf.map((v, k) => Math.abs(v - lq[k]))) };
  });
  const kept = outputs.filter((o) => o.cf === o.cq).length;
  const levels = 2 ** bits;

  const narration = (() => {
    const where = `weight w[${selRow}][${selCol}] = ${sgn(x, 4)}`;
    switch (stage) {
      case 0:
        return `Three rows (output channels) of a real weight matrix; their magnitudes differ by about 10×. We follow ${where}. A ${bits}-bit grid has ${levels} levels.`;
      case 1:
        return mode === "symmetric"
          ? `Symmetric: the range is ±max|w| ${gran === "per-tensor" ? "over the whole matrix" : "of this row"} = ±${fmt(Math.max(Math.abs(rMin), Math.abs(rMax)), 4)}${clip < 1 ? `, clipped to ${Math.round(clip * 100)}% → ±${fmt(p.hi, 4)}` : ""}.`
          : `Asymmetric: min = ${sgn(rMin, 4)}, max = ${sgn(rMax, 4)} ${gran === "per-tensor" ? "over the whole matrix" : "of this row"}, extended to contain 0${clip < 1 ? ` and clipped to ${Math.round(clip * 100)}%` : ""}.`;
      case 2:
        return mode === "symmetric"
          ? `s = range max / (2^(b−1) − 1) = ${fmt(p.hi, 4)} / ${p.qmax} = ${fmt(p.scale, 5)}. Grid ticks are spaced s apart.`
          : `s = (β − α) / (2^b − 1) = ${fmt(p.qmax * p.scale, 4)} / ${p.qmax} = ${fmt(p.scale, 5)}. Grid ticks are spaced s apart.`;
      case 3:
        return mode === "symmetric" ? "Symmetric quantization fixes z = 0: real 0 maps to integer 0." : `z = round(−α / s) = ${p.zero}: the integer that represents real 0 exactly.`;
      case 4:
        return `${where} → w / s = ${sgn(x, 4)} / ${fmt(p.scale, 5)} = ${sgn(tr.divided, 4)} (in units of grid steps).`;
      case 5:
        return `w / s + z = ${sgn(tr.shifted, 4)} → round → ${tr.rounded}.`;
      case 6:
        return tr.clipped
          ? `${tr.rounded} lies outside [${p.qmin}, ${p.qmax}], so it is clipped to q = ${tr.q}. Clipping error can exceed s/2.`
          : `${tr.rounded} is inside [${p.qmin}, ${p.qmax}]: stored integer q = ${tr.q}. ${qm.clippedCount} of 24 weights were clipped.`;
      case 7:
        return `ŵ = s (q − z) = ${fmt(p.scale, 5)} × (${tr.q} − ${p.zero}) = ${sgn(tr.xhat, 4)}.`;
      default:
        return `Error w − ŵ = ${sgn(tr.error, 5)} (|error| ${tr.clipped ? "includes clipping" : `≤ s/2 = ${fmt(p.scale / 2, 5)}`}). Whole matrix: SQNR ${db(qm.stats.sqnrDb)} dB; the classifier keeps ${kept}/${outputs.length} decisions.`;
    }
  })();

  const innerW = VW - PADL - PADR;
  const VH = ROWH * model.W.length + 8;

  return (
    <SimShell
      title="Quantization"
      subtitle="Round real weights onto a b-bit integer grid: scale, zero-point, rounding, clipping, error — and the effect on a tiny classifier."
      controls={
        <Controls
          s={s}
          extra={
            <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Draw new weights">
              <Shuffle size={14} /> New weights
            </button>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <div>
            <div className="mb-1 font-medium">Error per row</div>
            <div className="overflow-x-auto">
              <table className="w-full font-mono text-xs">
                <thead>
                  <tr className="text-muted">
                    <th className="text-left font-normal">row</th>
                    <th className="text-right font-normal">s</th>
                    <th className="text-right font-normal">z</th>
                    <th className="text-right font-normal">max|e|</th>
                    <th className="text-right font-normal">SQNR</th>
                  </tr>
                </thead>
                <tbody>
                  {qm.rowStats.map((st, i) => (
                    <tr key={i} className={i === selRow ? "text-[color:var(--gold)]" : ""}>
                      <td>{i}</td>
                      <td className="text-right">{fmt(qm.params[i].scale, 4)}</td>
                      <td className="text-right">{qm.params[i].zero}</td>
                      <td className="text-right">{fmt(st.maxErr, 4)}</td>
                      <td className="text-right">{db(st.sqnrDb)}</td>
                    </tr>
                  ))}
                  <tr className="border-t border-border">
                    <td colSpan={3}>all</td>
                    <td className="text-right">{fmt(qm.stats.maxErr, 4)}</td>
                    <td className="text-right">{db(qm.stats.sqnrDb)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="mt-1 text-xs text-muted">
              MSE {fmt(qm.stats.mse, 6)} · SQNR in dB · {qm.clippedCount} clipped
            </div>
          </div>
          <div>
            <div className="mb-1 font-medium">
              Tiny classifier: {kept}/{outputs.length} decisions kept
            </div>
            <div className="overflow-x-auto">
              <table className="w-full font-mono text-xs">
                <thead>
                  <tr className="text-muted">
                    <th className="text-left font-normal">pt</th>
                    <th className="text-right font-normal">float logits</th>
                    <th className="text-right font-normal">quant logits</th>
                    <th className="text-right font-normal">ok</th>
                  </tr>
                </thead>
                <tbody>
                  {outputs.map((o, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td className="whitespace-nowrap text-right">
                        {o.lf.map((v, k) => (
                          <span key={k} className={k === o.cf ? "font-bold" : "text-muted"}>
                            {" "}
                            {sgn(v, 2)}
                          </span>
                        ))}
                      </td>
                      <td className="whitespace-nowrap text-right">
                        {o.lq.map((v, k) => (
                          <span key={k} className={k === o.cq ? "font-bold" : "text-muted"}>
                            {" "}
                            {sgn(v, 2)}
                          </span>
                        ))}
                      </td>
                      <td className={`text-right ${o.cf === o.cq ? "text-accent" : "text-clay"}`}>{o.cf === o.cq ? "✓" : "✗"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-1 text-xs text-muted">Logits = W x + b on 6 fixed points; bias kept in float. Bold = predicted class.</div>
          </div>
        </div>
      }
    >
      <div className="mb-3 grid gap-2">
        <Seg label="Bits" value={bits} options={BITS.map((b) => ({ v: b, label: `${b}-bit` }))} onChange={setBits} />
        <Seg label="Mode" value={mode} options={[{ v: "symmetric" as QMode, label: "Symmetric" }, { v: "asymmetric" as QMode, label: "Asymmetric" }]} onChange={setMode} />
        <Seg label="Granularity" value={gran} options={[{ v: "per-tensor" as Granularity, label: "Per-tensor" }, { v: "per-channel" as Granularity, label: "Per-channel" }]} onChange={setGran} />
        <Slider label="Clip ratio" value={clip} min={0.3} max={1} step={0.01} onChange={setClip} fmt={(v) => `${Math.round(v * 100)}%`} />
        <Seg label="Follow row" value={selRow} options={[0, 1, 2].map((r) => ({ v: r, label: `row ${r}` }))} onChange={setSelRow} />
        <Slider label="Follow weight" value={selCol} min={0} max={N_FEATURES - 1} step={1} onChange={(v) => setSelCol(Math.round(v))} fmt={(v) => `#${Math.round(v)}`} />
      </div>

      <svg viewBox={`0 0 ${VW} ${VH}`} className="h-auto w-full" role="img" aria-label="Number lines of float weights, the quantization grid and dequantized weights for each row">
        {model.W.map((row, i) => {
          const pr = qm.params[i];
          const amax = Math.max(1e-9, ...row.map(Math.abs)) * 1.08;
          const X = (v: number) => PADL + ((Math.max(-amax, Math.min(amax, v)) + amax) / (2 * amax)) * innerW;
          const y0 = i * ROWH + 40;
          const grid = stage >= 2 ? gridLevels(pr, -amax, amax) : [];
          const dense = stage >= 2 && grid.length === 0;
          return (
            <g key={i}>
              <text x={PADL} y={i * ROWH + 13} fontSize={14} fill="var(--text-muted)">
                row {i} · view ±{fmt(amax / 1.08, 3)} · s = {fmt(pr.scale, 4)}
                {stage >= 3 ? ` · z = ${pr.zero}` : ""}
              </text>
              {/* clipping range */}
              {stage >= 1 &&
                [pr.lo, pr.hi].map((v, k) =>
                  v > -amax && v < amax ? <line key={k} x1={X(v)} x2={X(v)} y1={y0 - 22} y2={y0 + 26} stroke="var(--gold)" strokeDasharray="4 3" strokeWidth={1.5} /> : null,
                )}
              <line x1={PADL} x2={VW - PADR} y1={y0} y2={y0} stroke="var(--border)" strokeWidth={1.5} />
              <line x1={X(0)} x2={X(0)} y1={y0 - 5} y2={y0 + 5} stroke="var(--text-muted)" />
              {grid.map((g, k) => (
                <line key={k} x1={X(g)} x2={X(g)} y1={y0 - 7} y2={y0 + 7} stroke="var(--text-muted)" strokeOpacity={grid.length > 60 ? 0.35 : 0.8} />
              ))}
              {dense && (
                <text x={VW - PADR} y={y0 + 22} fontSize={12} textAnchor="end" fill="var(--text-muted)">
                  grid too dense to draw
                </text>
              )}
              {row.map((w, j) => {
                const sel = i === selRow && j === selCol;
                const d = qm.deq[i][j];
                return (
                  <g key={j}>
                    {stage >= 7 && <line x1={X(w)} y1={y0 - 12} x2={X(d)} y2={y0 + 14} stroke={sel ? "var(--gold)" : "var(--text-muted)"} strokeOpacity={sel ? 1 : 0.4} />}
                    <circle cx={X(w)} cy={y0 - 14} r={sel ? 6 : 4} fill={sel ? "var(--gold)" : "var(--accent)"} stroke="var(--surface)" />
                    {stage >= 7 && <circle cx={X(d)} cy={y0 + 16} r={sel ? 6 : 4} fill={sel ? "var(--gold)" : "var(--clay)"} stroke="var(--surface)" />}
                  </g>
                );
              })}
              {stage >= 7 && i === 0 && (
                <text x={VW - PADR} y={i * ROWH + 13} fontSize={12} textAnchor="end" fill="var(--text-muted)">
                  top: float · bottom: dequantized
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full font-mono text-xs">
          <tbody>
            {[
              ["w", sgn(x, 5), 0],
              ["range [α, β]", `[${sgn(p.lo, 4)}, ${sgn(p.hi, 4)}]`, 1],
              ["s", fmt(p.scale, 6), 2],
              ["z", String(p.zero), 3],
              ["w / s", sgn(tr.divided, 4), 4],
              ["round(w/s + z)", `${tr.rounded}  (from ${sgn(tr.shifted, 4)})`, 5],
              [`q = clip to [${p.qmin}, ${p.qmax}]`, `${tr.q}${tr.clipped ? "  (clipped)" : ""}`, 6],
              ["ŵ = s(q − z)", sgn(tr.xhat, 5), 7],
              ["error w − ŵ", sgn(tr.error, 5), 8],
            ].map(([k, v, st]) => (
              <tr key={String(k)} className={Number(st) === stage ? "bg-surface-2" : Number(st) > stage ? "opacity-30" : ""}>
                <td className="py-0.5 pr-3 text-muted">{k}</td>
                <td className="py-0.5 text-right">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3">
        <Narration title={`Step ${stage + 1}: ${STAGES[stage]}`}>{narration}</Narration>
      </div>
      <p className="mt-2 text-xs text-muted">
        Range of row {selRow}: [{sgn(rowMin, 3)}, {sgn(rowMax, 3)}]. Gold dashed lines mark the representable (clipping) range; ticks are grid levels.
      </p>
    </SimShell>
  );
}
