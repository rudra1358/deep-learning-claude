/**
 * Hand-written charts (no chart library): horizontal bars in HTML, a multi-series line chart in SVG.
 * Colours per label come from the same hash as the canvas overlays, so "car" matches everywhere.
 */
"use client";

import { useId, useMemo, useState } from "react";
import { colorForLabel } from "@/lib/draw";
import { EmptyState } from "./ui";

export function BarList({
  data,
  format = (v) => String(v),
  colorByLabel = true,
  emptyText = "No data yet.",
  max,
}: {
  data: [string, number][];
  format?: (v: number) => string;
  colorByLabel?: boolean;
  emptyText?: string;
  /** scale maximum (defaults to the largest value) */
  max?: number;
}) {
  if (data.length === 0) return <EmptyState>{emptyText}</EmptyState>;
  const top = max ?? Math.max(...data.map(([, v]) => v), 1e-9);
  return (
    <ul className="space-y-2">
      {data.map(([label, v]) => (
        <li key={label} className="grid grid-cols-[minmax(4rem,8rem)_1fr_auto] items-center gap-2 text-sm">
          <span className="truncate text-muted" title={label}>
            {label}
          </span>
          <span className="h-2.5 overflow-hidden rounded-full bg-surface-2" aria-hidden>
            <span
              className="block h-full rounded-full"
              style={{
                width: `${Math.max((v / top) * 100, v > 0 ? 1.5 : 0)}%`,
                background: colorByLabel ? colorForLabel(label) : "var(--accent)",
              }}
            />
          </span>
          <span className="tabular-nums">{format(v)}</span>
        </li>
      ))}
    </ul>
  );
}

export interface Series {
  name: string;
  values: number[];
}

/** Multi-series line chart. `x` are seconds. Hover shows values at the nearest x. */
export function LineChart({ x, series, height = 220, yLabel }: { x: number[]; series: Series[]; height?: number; yLabel?: string }) {
  const titleId = useId();
  const [hover, setHover] = useState<number | null>(null);
  const W = 640;
  const H = height;
  const pad = { l: 36, r: 12, t: 10, b: 26 };

  const { xMin, xMax, yMax } = useMemo(() => {
    const all = series.flatMap((s) => s.values);
    return {
      xMin: x.length ? Math.min(...x) : 0,
      xMax: x.length ? Math.max(...x) : 1,
      yMax: Math.max(1, ...all),
    };
  }, [x, series]);

  if (x.length === 0 || series.length === 0) return <EmptyState>No samples recorded for this run.</EmptyState>;

  const sx = (v: number) => pad.l + ((v - xMin) / Math.max(xMax - xMin, 1e-9)) * (W - pad.l - pad.r);
  const sy = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const yTicks = niceTicks(yMax);
  const xTicks = niceTicks(xMax - xMin).map((t) => t + xMin);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    for (let i = 1; i < x.length; i++) {
      if (Math.abs(sx(x[i] ?? 0) - px) < Math.abs(sx(x[best] ?? 0) - px)) best = i;
    }
    setHover(best);
  };

  const hx = hover !== null ? x[hover] : undefined;

  return (
    <figure aria-labelledby={titleId}>
      <figcaption id={titleId} className="sr-only">
        {yLabel ?? "Value"} over time for {series.map((s) => s.name).join(", ")}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none select-none"
        role="img"
        aria-hidden={false}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" />
            <text x={pad.l - 6} y={sy(t)} textAnchor="end" dominantBaseline="middle" fontSize="11" fill="var(--muted)">
              {t}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - 8} textAnchor="middle" fontSize="11" fill="var(--muted)">
            {t}s
          </text>
        ))}
        {series.map((s) => (
          <polyline
            key={s.name}
            fill="none"
            stroke={colorForLabel(s.name)}
            strokeWidth="2"
            strokeLinejoin="round"
            points={s.values.map((v, i) => `${sx(x[i] ?? 0)},${sy(v)}`).join(" ")}
          />
        ))}
        {hx !== undefined && (
          <line x1={sx(hx)} x2={sx(hx)} y1={pad.t} y2={H - pad.b} stroke="var(--muted)" strokeDasharray="3 3" />
        )}
      </svg>
      <div className="mt-2 flex min-h-5 flex-wrap gap-x-4 gap-y-1 text-xs">
        {hover !== null && <span className="text-muted tabular-nums">t = {hx}s</span>}
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ background: colorForLabel(s.name) }} aria-hidden />
            {s.name}
            {hover !== null && <span className="tabular-nums font-medium">{s.values[hover] ?? 0}</span>}
          </span>
        ))}
      </div>
    </figure>
  );
}

/** ~5 round tick values from 0 to max. */
function niceTicks(max: number): number[] {
  if (max <= 0) return [0];
  const raw = max / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}
