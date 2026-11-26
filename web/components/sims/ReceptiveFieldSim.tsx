"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, seqColor, useStepper } from "./SimShell";
import {
  PRESETS,
  analyze,
  backProject,
  effectiveRF,
  layerPadding,
  massWidth,
  type LayerInfo,
  type PaddingMode,
  type RFLayer,
} from "@/lib/sim/receptiveField";

const MAX_LAYERS = 10;
const INPUT_SIZES = [16, 24, 32, 40, 48, 64];

function layerName(l: RFLayer) {
  if (l.kind === "pool") return `pool ${l.k}×${l.k}${l.s !== 1 ? ` s${l.s}` : ""}`;
  return `conv ${l.k}×${l.k}${l.s !== 1 ? ` s${l.s}` : ""}${l.d !== 1 ? ` d${l.d}` : ""}`;
}

/** 1-D layered diagram: one row per level, units placed at their centre in input coordinates. */
function StripDiagram({
  layers,
  info,
  masks,
  ranges,
  level,
  mode,
}: {
  layers: RFLayer[];
  info: LayerInfo[];
  masks: boolean[][];
  ranges: { lo: number; hi: number }[];
  level: number;
  mode: PaddingMode;
}) {
  const L = layers.length;
  const n = info[0].size;
  const labelW = 112;
  const plotW = 440;
  const cw = plotW / n;
  const rowH = 30;
  const top = 8;
  const H = top + (L + 1) * rowH + 18;
  const yOf = (lv: number) => top + (L - lv) * rowH + rowH / 2;
  const xOf = (lv: number, u: number) => labelW + (info[lv].start + u * info[lv].j + 0.5) * cw;
  const sz = Math.max(3, Math.min(cw * 0.8, 16));

  // connection lines for the most recent projection step (from level+1 down to level)
  const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
  if (level < L) {
    const layer = layers[level];
    const p = layerPadding(layer, mode);
    const d = layer.kind === "pool" ? 1 : layer.d;
    const nIn = info[level].size;
    const yTop = yOf(level + 1) + sz / 2;
    const yBot = yOf(level) - (level === 0 ? 7 : sz / 2);
    for (let u = 0; u < info[level + 1].size && lines.length < 400; u++) {
      if (!masks[level + 1][u]) continue;
      for (let a = 0; a < layer.k; a++) {
        const idx = u * layer.s - p + a * d;
        if (idx < 0 || idx >= nIn) continue;
        lines.push({ x1: xOf(level + 1, u), y1: yTop, x2: xOf(level, idx), y2: yBot });
      }
    }
  }
  return (
    <svg viewBox={`0 0 ${labelW + plotW + 8} ${H}`} className="h-auto w-full" role="img" aria-label="Layered diagram: each row is a layer, highlighted units influence the selected output unit">
      {lines.map((l, i) => (
        <line key={i} {...l} stroke="var(--gold)" strokeWidth={0.6} opacity={0.7} />
      ))}
      {info.map((row, lv) => {
        const y = yOf(lv);
        const active = lv >= level;
        const isOut = lv === L;
        return (
          <g key={lv}>
            <text x={4} y={y + 4} fontSize={12} fill={lv === level ? "var(--gold)" : "var(--text-muted)"} fontFamily="var(--font-mono, monospace)">
              {lv === 0 ? `input ${n}` : `${lv}: ${layerName(layers[lv - 1])}`}
            </text>
            {Array.from({ length: row.size }, (_, u) => {
              const on = active && masks[lv][u];
              const fill = on ? (isOut ? "var(--gold)" : lv === level ? "color-mix(in srgb, var(--gold) 55%, var(--accent))" : "var(--accent)") : "var(--surface)";
              const w = lv === 0 ? cw : sz;
              return (
                <rect
                  key={u}
                  x={xOf(lv, u) - w / 2}
                  y={y - (lv === 0 ? 7 : sz / 2)}
                  width={w}
                  height={lv === 0 ? 14 : sz}
                  rx={lv === 0 ? 0 : 2}
                  fill={fill}
                  stroke="var(--border)"
                  strokeWidth={0.8}
                />
              );
            })}
            {active && ranges[lv].hi >= ranges[lv].lo && (
              <line
                x1={xOf(lv, ranges[lv].lo) - (lv === 0 ? cw / 2 : sz / 2)}
                x2={xOf(lv, ranges[lv].hi) + (lv === 0 ? cw / 2 : sz / 2)}
                y1={y + (lv === 0 ? 11 : sz / 2 + 3)}
                y2={y + (lv === 0 ? 11 : sz / 2 + 3)}
                stroke="var(--gold)"
                strokeWidth={1.5}
                strokeDasharray="4 3"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

export default function ReceptiveFieldSim() {
  const [layers, setLayers] = useState<RFLayer[]>(PRESETS[1].layers);
  const [inputSize, setInputSize] = useState(PRESETS[1].input);
  const [preset, setPreset] = useState(PRESETS[1].id);
  const [mode, setMode] = useState<PaddingMode>("same");
  const [unitSel, setUnitSel] = useState<number | null>(null);
  const [showErf, setShowErf] = useState(false);

  const info = useMemo(() => analyze(layers, inputSize, mode), [layers, inputSize, mode]);
  const L = layers.length;
  const outN = info[L].size;
  const broken = info.some((r) => r.size <= 0);
  const centre = Math.floor((outN - 1) / 2);
  const unit = unitSel === null ? centre : Math.min(Math.max(0, unitSel), Math.max(0, outN - 1));

  const s = useStepper(L + 1, { baseMs: 1100 });
  const step = Math.min(s.step, L);
  const level = L - step; // lowest level highlighted so far
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [layers, inputSize, mode, unit, reset]);

  const colProj = useMemo(() => (broken ? null : backProject(layers, info, unit, mode)), [broken, layers, info, unit, mode]);
  const rowProj = useMemo(() => (broken ? null : backProject(layers, info, centre, mode)), [broken, layers, info, centre, mode]);
  const erf = useMemo(() => (showErf && !broken ? effectiveRF(layers, info, centre, unit, mode) : null), [showErf, broken, layers, info, centre, unit, mode]);

  const n = inputSize;
  const last = info[L];
  const cRow = Math.round(last.start + centre * last.j);
  const cCol = Math.round(last.start + unit * last.j);
  const erfStats = useMemo(() => {
    if (!erf) return null;
    let mx = 0;
    for (let i = 0; i < erf.length; i++) mx = Math.max(mx, erf[i]);
    return { max: mx || 1, w50: massWidth(erf, n, cRow, cCol, 0.5), w90: massWidth(erf, n, cRow, cCol, 0.9) };
  }, [erf, n, cRow, cCol]);

  const applyPreset = (id: string) => {
    const p = PRESETS.find((q) => q.id === id);
    if (!p) return;
    setPreset(id);
    setLayers(p.layers.map((l) => ({ ...l })));
    setInputSize(p.input);
    setUnitSel(null);
  };
  const edit = (i: number, patch: Partial<RFLayer>) => {
    setPreset("custom");
    setUnitSel(null);
    setLayers((ls) =>
      ls.map((l, j) => {
        if (j !== i) return l;
        const nl = { ...l, ...patch };
        if (nl.kind === "pool") {
          nl.d = 1;
          if (nl.k < 2) nl.k = 2;
          if (patch.kind === "pool") {
            nl.k = 2;
            nl.s = 2;
          }
        }
        return nl;
      }),
    );
  };
  const remove = (i: number) => {
    setPreset("custom");
    setUnitSel(null);
    setLayers((ls) => (ls.length > 1 ? ls.filter((_, j) => j !== i) : ls));
  };
  const add = (l: RFLayer) => {
    setPreset("custom");
    setUnitSel(null);
    setLayers((ls) => (ls.length < MAX_LAYERS ? [...ls, l] : ls));
  };

  const selCls = "field py-1 text-xs";

  /* ---------------- narration ---------------- */
  let narr: { title: string; body: string } = { title: "", body: "" };
  if (broken) {
    narr = { title: "Too many layers for this input", body: "Some layer would end up with an output of size 0. Make the input bigger, use padding “same”, or remove a layer that shrinks the image." };
  } else if (colProj) {
    const { masks, ranges } = colProj;
    if (step === 0) {
      narr = {
        title: `Start: output unit ${unit} of layer ${L}`,
        body: `Layer ${L} has ${outN} units per row. We pick unit ${unit} (gold) and ask: which units in each earlier layer can change its value? The formula in the table predicts the answer before we look: r = ${last.r} input pixels, jump j = ${last.j}. Press Step to trace it down one layer at a time.`,
      };
    } else {
      const layer = layers[level];
      const p = layerPadding(layer, mode);
      const d = layer.kind === "pool" ? 1 : layer.d;
      const up = ranges[level + 1];
      const here = ranges[level];
      const count = masks[level].filter(Boolean).length;
      const span = here.hi - here.lo + 1;
      const clippedLo = Math.max(0, -here.lo);
      const clippedHi = Math.max(0, here.hi - (info[level].size - 1));
      const holes = Math.max(0, Math.min(here.hi, info[level].size - 1) - Math.max(here.lo, 0) + 1 - count);
      const kTerm = d > 1 ? `${d}·(${layer.k}−1)` : `${layer.k}−1`;
      narr = {
        title: `Layer ${level + 1} (${layerName(layer)}, padding ${p}) → level ${level === 0 ? "input" : level}`,
        body:
          `Unit i of this layer reads positions i·${layer.s} − ${p} + a·${d} (a = 0…${layer.k - 1}) of the level below. ` +
          `So the highlighted range [${up.lo}, ${up.hi}] maps to [${up.lo}·${layer.s} − ${p}, ${up.hi}·${layer.s} − ${p} + ${kTerm}] = [${here.lo}, ${here.hi}]: ${span} position${span === 1 ? "" : "s"}. ` +
          (clippedLo + clippedHi > 0 ? `${clippedLo + clippedHi} of them land in the zero padding past the edge, so they carry no information. ` : "") +
          (holes > 0 ? `${holes} position${holes === 1 ? " is" : "s are"} skipped by the spread-out (dilated) kernel, leaving a hole (“gridding”). ` : "") +
          (level === 0
            ? `At the input the region is ${span} pixels wide. That is exactly the r = ${last.r} predicted by r_l = r_(l−1) + (k_eff − 1)·j_(l−1). In 2-D it is ${last.r}×${last.r}.`
            : `${count} unit${count === 1 ? "" : "s"} of level ${level} contribute. Neighbouring units are ${info[level].j} input pixel${info[level].j === 1 ? "" : "s"} apart (the jump). That is why deeper layers widen the region faster.`),
      };
    }
  }

  /* ---------------- 2-D input view ---------------- */
  const cs = 10;
  const rf2d = (() => {
    if (broken || !colProj || !rowProj) return null;
    const rm = rowProj.masks[0];
    const cm = colProj.masks[0];
    const rr = rowProj.ranges[0];
    const cr = colProj.ranges[0];
    const done = step === L;
    return (
      <svg viewBox={`-1 -1 ${n * cs + 2} ${n * cs + 2}`} className="h-auto w-full" style={{ maxWidth: 340 }} role="img" aria-label={`Input grid ${n} by ${n} with the receptive field of the selected unit${showErf ? " and the effective receptive field heatmap" : ""}`}>
        {Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => {
            let fill = "var(--surface)";
            if (erf && erfStats) fill = seqColor(Math.sqrt(erf[i * n + j] / erfStats.max));
            else if (done && rm[i] && cm[j]) fill = "color-mix(in srgb, var(--accent) 70%, var(--surface))";
            return <rect key={`${i}-${j}`} x={j * cs} y={i * cs} width={cs} height={cs} fill={fill} stroke="var(--border)" strokeWidth={0.4} />;
          }),
        )}
        <rect
          x={Math.max(0, cr.lo) * cs}
          y={Math.max(0, rr.lo) * cs}
          width={(Math.min(n - 1, cr.hi) - Math.max(0, cr.lo) + 1) * cs}
          height={(Math.min(n - 1, rr.hi) - Math.max(0, rr.lo) + 1) * cs}
          fill="none"
          stroke="var(--gold)"
          strokeWidth={2}
          strokeDasharray={done ? undefined : "5 4"}
        />
        <circle cx={(last.start + unit * last.j + 0.5) * cs} cy={(last.start + centre * last.j + 0.5) * cs} r={2.5} fill="var(--clay)" />
      </svg>
    );
  })();

  const aside = (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="pr-2 font-normal">layer</th>
              <th className="pr-2 font-normal">k_eff</th>
              <th className="pr-2 font-normal">out</th>
              <th className="pr-2 font-normal">r</th>
              <th className="font-normal">j</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {info.map((row, lv) => (
              <tr key={lv} style={{ background: lv === level && !broken ? "color-mix(in srgb, var(--gold) 22%, transparent)" : undefined }}>
                <td className="pr-2">{lv === 0 ? "input" : `${lv}. ${layerName(layers[lv - 1])}`}</td>
                <td className="pr-2">{lv === 0 ? "–" : row.kEff}</td>
                <td className="pr-2" style={{ color: row.size <= 0 ? "var(--clay)" : undefined }}>{row.size}</td>
                <td className="pr-2">{row.r}</td>
                <td>{row.j}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Narration title={narr.title}>{narr.body}</Narration>
      {erfStats && (
        <div className="rounded-lg border border-border p-3 text-xs text-muted">
          <div className="mb-1 font-medium text-[var(--text)]">Effective receptive field</div>
          Theoretical receptive field (RF): <span className="font-mono">{last.r}×{last.r}</span>. But half of the real influence (gradient size) sits in the central{" "}
          <span className="font-mono">{erfStats.w50}×{erfStats.w50}</span> square, 90% in{" "}
          <span className="font-mono">{erfStats.w90}×{erfStats.w90}</span>
          {last.r > n ? " (the theoretical RF is wider than the input here)" : ""}. Influence bunches up near the centre because many more paths pass through central pixels.
        </div>
      )}
    </div>
  );

  return (
    <SimShell
      title="Receptive Field Explorer"
      subtitle="Stack conv and pool layers, then trace one output unit back down to see exactly which input pixels it can see."
      controls={<Controls s={s} />}
      aside={aside}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <label className="flex min-w-0 items-center gap-2">
          <span className="text-muted">Preset</span>
          <select className={selCls} value={preset} onChange={(e) => applyPreset(e.target.value)} aria-label="Layer stack preset">
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
            {preset === "custom" && <option value="custom">Custom</option>}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted">Input</span>
          <select className={selCls} value={inputSize} onChange={(e) => { setInputSize(+e.target.value); setUnitSel(null); }} aria-label="Input size">
            {INPUT_SIZES.map((v) => (
              <option key={v} value={v}>
                {v}×{v}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted">Conv padding</span>
          <select className={selCls} value={mode} onChange={(e) => { setMode(e.target.value as PaddingMode); setUnitSel(null); }} aria-label="Convolution padding mode">
            <option value="same">same</option>
            <option value="valid">valid (0)</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showErf} onChange={(e) => setShowErf(e.target.checked)} /> Effective RF heatmap
        </label>
      </div>

      {/* layer editor */}
      <div className="mt-3 flex flex-col gap-1.5">
        {layers.map((l, i) => (
          <div key={i} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
            <span className="w-5 font-mono text-muted">{i + 1}.</span>
            <select className={selCls} value={l.kind} onChange={(e) => edit(i, { kind: e.target.value as RFLayer["kind"] })} aria-label={`Layer ${i + 1} type`}>
              <option value="conv">conv</option>
              <option value="pool">pool</option>
            </select>
            <label className="flex items-center gap-1">
              k
              <select className={selCls} value={l.k} onChange={(e) => edit(i, { k: +e.target.value })} aria-label={`Layer ${i + 1} kernel size`}>
                {(l.kind === "pool" ? [2, 3, 4] : [1, 2, 3, 5, 7]).map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1">
              s
              <select className={selCls} value={l.s} onChange={(e) => edit(i, { s: +e.target.value })} aria-label={`Layer ${i + 1} stride`}>
                {[1, 2, 3].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            {l.kind === "conv" && (
              <label className="flex items-center gap-1">
                d
                <select className={selCls} value={l.d} onChange={(e) => edit(i, { d: +e.target.value })} aria-label={`Layer ${i + 1} dilation`}>
                  {[1, 2, 3, 4, 8].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <span className="font-mono text-faint">→ {info[i + 1].size}</span>
            <button className="sim-btn ml-auto py-0.5" onClick={() => remove(i)} disabled={layers.length <= 1} aria-label={`Remove layer ${i + 1}`}>
              Remove
            </button>
          </div>
        ))}
        <div className="flex flex-wrap gap-2">
          <button className="sim-btn" onClick={() => add({ kind: "conv", k: 3, s: 1, d: 1 })} disabled={layers.length >= MAX_LAYERS} aria-label="Add a 3 by 3 convolution layer">
            + conv 3×3
          </button>
          <button className="sim-btn" onClick={() => add({ kind: "pool", k: 2, s: 2, d: 1 })} disabled={layers.length >= MAX_LAYERS} aria-label="Add a 2 by 2 pooling layer">
            + pool 2×2
          </button>
          <button className="sim-btn" onClick={() => add({ kind: "conv", k: 3, s: 2, d: 1 })} disabled={layers.length >= MAX_LAYERS} aria-label="Add a stride 2 convolution layer">
            + conv s2
          </button>
        </div>
      </div>

      {!broken && (
        <div className="mt-3">
          <Slider label="Output unit (column)" value={unit} min={0} max={Math.max(0, outN - 1)} step={1} onChange={(v) => setUnitSel(v)} fmt={(v) => `${v} / ${outN - 1}`} />
        </div>
      )}

      {!broken && colProj ? (
        <div className="mt-4 flex flex-col gap-4">
          <figure className="min-w-0">
            <figcaption className="mb-1 text-xs text-muted">
              One row per layer (output at the top). Each unit sits above the input pixel at its centre, so the spacing shows the jump j. Gold dashes mark the range traced back so far.
            </figcaption>
            <StripDiagram layers={layers} info={info} masks={colProj.masks} ranges={colProj.ranges} level={level} mode={mode} />
          </figure>
          <figure className="min-w-0">
            <figcaption className="mb-1 text-xs text-muted">
              Input {n}×{n}: receptive field of unit (row {centre}, column {unit}) of the last layer{showErf ? " — shaded by |∂output/∂input| of a random linear version of the stack (24 weight draws)" : step === L ? " — shaded pixels can affect it" : " — press Step until you reach the input row"}.
            </figcaption>
            {rf2d}
          </figure>
        </div>
      ) : (
        <p className="mt-4 text-sm text-clay">This stack shrinks the image to nothing. See the table.</p>
      )}
    </SimShell>
  );
}
