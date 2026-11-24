"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { PE_BASE, frequency, peMatrix, similarityClosedForm, similarityRow, wavelength } from "@/lib/sim/posenc";

/** Read a CSS colour variable as RGB so the canvas follows light/dark mode. */
function cssRgb(name: string, fallback: [number, number, number]): [number, number, number] {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = /^#([0-9a-f]{6})$/i.exec(v);
  if (!m) return fallback;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function useThemeVersion() {
  const [v, setV] = useState(0);
  useEffect(() => {
    const obs = new MutationObserver(() => setV((x) => x + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    return () => obs.disconnect();
  }, []);
  return v;
}

const CW = 640;
const CH = 300;

function LinePlot({
  ys,
  marker,
  yMin,
  yMax,
  label,
  xLabel,
  ariaLabel,
  ref2,
}: {
  ys: number[];
  marker: number;
  yMin: number;
  yMax: number;
  label: string;
  xLabel: string;
  ariaLabel: string;
  ref2?: number[];
}) {
  const W = 640;
  const H = 180;
  const L = 44;
  const R = 12;
  const T = 14;
  const B = 28;
  const n = ys.length;
  const x = (i: number) => L + (n <= 1 ? 0 : (i * (W - L - R)) / (n - 1));
  const y = (v: number) => T + ((yMax - v) * (H - T - B)) / (yMax - yMin || 1);
  const path = (arr: number[]) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticks = [yMin, (yMin + yMax) / 2, yMax];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      {ticks.map((t, k) => (
        <g key={k}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--border)" />
          <text x={L - 6} y={y(t) + 3} fontSize={10} textAnchor="end" fill="var(--text-faint)">
            {Number.isInteger(t) ? t : t.toFixed(1)}
          </text>
        </g>
      ))}
      {yMin < 0 && yMax > 0 && <line x1={L} x2={W - R} y1={y(0)} y2={y(0)} stroke="var(--border-strong)" />}
      {ref2 && <path d={path(ref2)} fill="none" stroke="var(--clay)" strokeWidth={1.2} strokeDasharray="4 4" />}
      <path d={path(ys)} fill="none" stroke="var(--accent)" strokeWidth={1.8} />
      <line x1={x(marker)} x2={x(marker)} y1={T} y2={H - B} stroke="var(--gold)" strokeDasharray="3 3" />
      <circle cx={x(marker)} cy={y(ys[marker] ?? 0)} r={4} fill="var(--gold)" />
      <text x={L} y={10} fontSize={10.5} fill="var(--text-muted)">
        {label}
      </text>
      <text x={W - R} y={H - 6} fontSize={10} textAnchor="end" fill="var(--text-faint)">
        {xLabel} = {n - 1}
      </text>
      <text x={L} y={H - 6} fontSize={10} fill="var(--text-faint)">
        {xLabel} = 0
      </text>
    </svg>
  );
}

export default function PositionalEncodingSim() {
  const [maxPos, setMaxPos] = useState(64);
  const [dModel, setDModel] = useState(32);
  const [dim, setDim] = useState(4);
  const s = useStepper(maxPos, { baseMs: 260 });
  const p = Math.min(s.step, maxPos - 1);
  const j = Math.min(dim, dModel - 1);
  const theme = useThemeVersion();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const P = useMemo(() => peMatrix(maxPos, dModel), [maxPos, dModel]);
  const sim = useMemo(() => similarityRow(P, p), [P, p]);
  const closed = useMemo(() => sim.map((_, q) => similarityClosedForm(q - p, dModel)), [sim, p, dModel]);
  const column = useMemo(() => P.map((r) => r[j]), [P, j]);

  // Heatmap: x = position, y = dimension (0 at top). Colour: accent for +, clay for −.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const acc = cssRgb("--accent", [61, 122, 107]);
    const clay = cssRgb("--clay", [181, 111, 85]);
    const surf = cssRgb("--surface", [255, 255, 255]);
    const gold = getComputedStyle(document.documentElement).getPropertyValue("--gold").trim() || "#a8832f";
    const cw = CW / maxPos;
    const ch = CH / dModel;
    for (let pos = 0; pos < maxPos; pos++) {
      for (let d = 0; d < dModel; d++) {
        const v = P[pos][d];
        const col = v >= 0 ? acc : clay;
        const a = Math.abs(v);
        const rgb = col.map((cc, k) => Math.round(surf[k] + (cc - surf[k]) * a));
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(Math.floor(pos * cw), Math.floor(d * ch), Math.ceil(cw), Math.ceil(ch));
      }
    }
    ctx.strokeStyle = gold;
    ctx.lineWidth = 2;
    ctx.strokeRect(p * cw + 1, 1, Math.max(cw - 2, 1), CH - 2);
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(1, j * ch + 1, CW - 2, Math.max(ch - 2, 1));
    ctx.setLineDash([]);
  }, [P, maxPos, dModel, p, j, theme]);

  const onCanvasClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    s.setPlaying(false);
    s.setStep(Math.min(maxPos - 1, Math.max(0, Math.floor(fx * maxPos))));
    setDim(Math.min(dModel - 1, Math.max(0, Math.floor(fy * dModel))));
  };

  const i = Math.floor(j / 2);
  const omega = frequency(i, dModel);
  const lambda = wavelength(j, dModel);
  const k = 3;
  const symA = p + k < maxPos ? sim[p + k] : null;
  const symB = p - k >= 0 ? sim[p - k] : null;

  const controls = (
    <div className="flex flex-col gap-2">
      <Controls s={s} />
      <div className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
        <Slider label="max positions n" value={maxPos} min={8} max={128} step={4} onChange={(v) => { setMaxPos(v); s.reset(); }} fmt={(v) => String(v)} />
        <Slider label="d_model" value={dModel} min={4} max={128} step={4} onChange={(v) => setDModel(v)} fmt={(v) => String(v)} />
        <Slider label="dimension j" value={j} min={0} max={dModel - 1} step={1} onChange={(v) => setDim(v)} fmt={(v) => String(v)} />
        <Slider label="position p" value={p} min={0} max={maxPos - 1} step={1} onChange={(v) => { s.setPlaying(false); s.setStep(v); }} fmt={(v) => String(v)} />
      </div>
    </div>
  );

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={`Position p = ${p}, dimension j = ${j}`}>
        Dimension {j} is the {j % 2 === 0 ? "sine" : "cosine"} of pair i = {i}, with angular frequency ω<sub>{i}</sub> = 1/{PE_BASE}
        <sup>{(2 * i)}/{dModel}</sup> = {fmt(omega, 4)} rad per position, so its wavelength is 2π/ω = {lambda < 1e4 ? lambda.toFixed(1) : lambda.toExponential(2)} positions.
        Low j oscillate fast (fine position), high j barely move (coarse position): a geometric ladder of clocks from 2π to about {(2 * Math.PI * PE_BASE).toFixed(0)}.
      </Narration>
      <div className="rounded-lg border border-border p-2.5 text-xs">
        <div className="mb-1 font-medium">Similarity depends only on the offset</div>
        <p className="text-muted">
          PE(p)·PE(q) = Σ<sub>i</sub> cos(ω<sub>i</sub>(p − q)). Self-similarity is always d/2 = {dModel / 2}.
        </p>
        <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-muted">
          <li>PE({p})·PE({p}) = {fmt(sim[p], 3)}</li>
          {symA !== null && <li>PE({p})·PE({p + k}) = {fmt(symA, 3)}</li>}
          {symB !== null && <li>PE({p})·PE({p - k}) = {fmt(symB, 3)}</li>}
          <li>
            Σcos(3ω<sub>i</sub>) = {fmt(similarityClosedForm(k, dModel), 3)}
          </li>
        </ul>
      </div>
      <div className="rounded-lg border border-border p-2.5 text-xs">
        <div className="mb-1 font-medium">PE({p}), first {Math.min(8, dModel)} dims</div>
        <div className="font-mono text-[11px] text-muted">[{P[p].slice(0, 8).map((v) => v.toFixed(3)).join(", ")}{dModel > 8 ? ", …" : ""}]</div>
      </div>
    </div>
  );

  return (
    <SimShell title="Positional Encoding" subtitle="The sinusoidal encoding matrix, its frequencies, and why dot products between positions depend on distance." controls={controls} aside={aside}>
      <div className="mb-1 flex flex-wrap justify-between text-xs text-muted">
        <span>PE matrix: positions 0…{maxPos - 1} → (horizontal), dimensions 0…{dModel - 1} ↓ (vertical). Click to select.</span>
        <span>
          <span className="text-accent">■</span> +1 &nbsp; <span className="text-clay">■</span> −1
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={CW}
        height={CH}
        onClick={onCanvasClick}
        className="h-auto w-full cursor-crosshair rounded border border-border"
        role="img"
        aria-label={`Heatmap of the ${maxPos} by ${dModel} sinusoidal positional encoding matrix. Selected position ${p}, dimension ${j}.`}
      />
      <div className="mt-4 grid gap-4">
        <LinePlot
          ys={column}
          marker={p}
          yMin={-1}
          yMax={1}
          label={`PE[pos, ${j}] = ${j % 2 === 0 ? "sin" : "cos"}(pos · ${fmt(omega, 4)}) across positions`}
          xLabel="position"
          ariaLabel={`Dimension ${j} of the positional encoding plotted across positions.`}
        />
        <LinePlot
          ys={sim}
          ref2={closed}
          marker={p}
          yMin={Math.min(0, Math.floor(Math.min(...sim)))}
          yMax={dModel / 2}
          label={`PE(${p}) · PE(q) for every q (solid) and closed form Σ cos(ω·(q−${p})) (dashed): identical`}
          xLabel="q"
          ariaLabel={`Dot product between the encoding of position ${p} and every other position.`}
        />
      </div>
    </SimShell>
  );
}
