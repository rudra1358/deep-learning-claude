"use client";

import { useState } from "react";

export type NormLine = [number, number, number, number];

/**
 * Transparent overlay for drawing a counting line: click point A, then point B.
 * Coordinates are normalised to 0..1 of the overlay box, which matches the media
 * (video / first frame) because the overlay covers it exactly.
 *
 * Rendered as SVG with viewBox 0..1 so it scales with the element automatically.
 */
export function LineEditor({
  value,
  onChange,
  active,
  onDone,
}: {
  value: NormLine | null;
  onChange: (l: NormLine | null) => void;
  /** when true, clicks place points; otherwise the line is only displayed */
  active: boolean;
  onDone?: () => void;
}) {
  const [first, setFirst] = useState<[number, number] | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);

  const pos = (e: React.PointerEvent<SVGSVGElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    const clamp = (v: number) => Math.min(1, Math.max(0, v));
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
  };

  const onClick = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!active) return;
    const p = pos(e);
    if (!first) {
      setFirst(p);
      return;
    }
    const line: NormLine = [first[0], first[1], p[0], p[1]].map((v) => Number(v.toFixed(4))) as NormLine;
    setFirst(null);
    setHover(null);
    onChange(line);
    onDone?.();
  };

  const draft: NormLine | null = first && hover ? [first[0], first[1], hover[0], hover[1]] : null;
  const shown = draft ?? (first ? null : value);

  const dot = (x: number, y: number, key: string) => (
    <span
      key={key}
      aria-hidden
      className="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-yellow-400 ring-2 ring-black/40"
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
    />
  );

  return (
    <>
      <svg
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        className={`absolute inset-0 h-full w-full ${active ? "cursor-crosshair" : "pointer-events-none"}`}
        onPointerUp={onClick}
        onPointerMove={(e) => active && first && setHover(pos(e))}
        role={active ? "application" : undefined}
        aria-label={
          active
            ? first
              ? "Click to place the second point of the counting line"
              : "Click to place the first point of the counting line"
            : undefined
        }
      >
        {shown && (
          <line
            x1={shown[0]}
            y1={shown[1]}
            x2={shown[2]}
            y2={shown[3]}
            stroke="#facc15"
            strokeWidth="3"
            strokeDasharray="10 6"
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
      {first && dot(first[0], first[1], "a")}
      {!first && shown && [dot(shown[0], shown[1], "a"), dot(shown[2], shown[3], "b")]}
      {active && (
        <div className="pointer-events-none absolute left-2 top-2 rounded bg-black/70 px-2 py-1 text-xs text-white">
          {first ? "Click the second point" : "Click the first point of the line"}
        </div>
      )}
    </>
  );
}
