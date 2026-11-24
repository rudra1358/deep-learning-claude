"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";

/**
 * useStepper — the shared playback engine for every simulator.
 * A simulation is modelled as a sequence of discrete steps 0..total-1.
 * `total` may be Infinity for open-ended simulations (e.g. optimizer training).
 */
export function useStepper(total: number, opts: { baseMs?: number; onTick?: () => void } = {}) {
  const baseMs = opts.baseMs ?? 900;
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const tickRef = useRef(opts.onTick);
  tickRef.current = opts.onTick;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      tickRef.current?.();
      setStep((s) => {
        if (s + 1 >= total) {
          setPlaying(false);
          return Math.max(0, total - 1);
        }
        return s + 1;
      });
    }, baseMs / speed);
    return () => clearInterval(id);
  }, [playing, speed, total, baseMs]);

  const next = useCallback(() => {
    tickRef.current?.();
    setStep((s) => Math.min(s + 1, Math.max(0, total - 1)));
  }, [total]);
  const prev = useCallback(() => setStep((s) => Math.max(0, s - 1)), []);
  const reset = useCallback(() => {
    setPlaying(false);
    setStep(0);
  }, []);

  return { step, setStep, playing, setPlaying, speed, setSpeed, next, prev, reset, total };
}

export type Stepper = ReturnType<typeof useStepper>;

export function Controls({ s, canPrev = true, extra }: { s: Stepper; canPrev?: boolean; extra?: ReactNode }) {
  const finite = Number.isFinite(s.total);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className="sim-btn primary" onClick={() => s.setPlaying(!s.playing)} aria-label={s.playing ? "Pause" : "Play"}>
        {s.playing ? <Pause size={14} /> : <Play size={14} />} {s.playing ? "Pause" : "Play"}
      </button>
      {canPrev && (
        <button className="sim-btn" onClick={s.prev} disabled={s.step === 0} aria-label="Previous step">
          <SkipBack size={14} />
        </button>
      )}
      <button className="sim-btn" onClick={s.next} disabled={finite && s.step >= s.total - 1} aria-label="Next step">
        <SkipForward size={14} /> Step
      </button>
      <button className="sim-btn" onClick={s.reset} aria-label="Reset">
        <RotateCcw size={14} /> Reset
      </button>
      <label className="ml-1 flex items-center gap-2 text-xs text-muted">
        Speed
        <input type="range" min={0.25} max={4} step={0.25} value={s.speed} onChange={(e) => s.setSpeed(+e.target.value)} className="w-20" aria-label="Animation speed" />
        <span className="w-8 font-mono">{s.speed}×</span>
      </label>
      {finite && (
        <span className="ml-auto font-mono text-xs text-faint">
          step {s.step + 1}/{s.total}
        </span>
      )}
      {extra}
    </div>
  );
}

export function SimShell({ title, subtitle, controls, children, aside }: { title: string; subtitle?: string; controls?: ReactNode; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="card my-6 overflow-hidden" aria-label={`${title} simulator`}>
      <header className="border-b border-border px-4 py-3">
        <div className="font-serif text-[1.05rem] font-semibold">{title}</div>
        {subtitle && <div className="text-sm text-muted">{subtitle}</div>}
      </header>
      {controls && <div className="border-b border-border bg-surface-2/60 px-4 py-2.5">{controls}</div>}
      <div className={aside ? "grid gap-0 lg:grid-cols-[1fr_300px]" : ""}>
        <div className="min-w-0 p-4">{children}</div>
        {aside && <aside className="border-t border-border p-4 text-sm lg:border-t-0 lg:border-l">{aside}</aside>}
      </div>
    </section>
  );
}

/** A labelled numeric slider used by every simulator. */
export function Slider({ label, value, min, max, step = 0.01, onChange, fmt }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="flex items-center gap-2 text-xs">
      <span className="w-24 shrink-0 text-muted">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} className="min-w-0 flex-1" aria-label={label} />
      <span className="w-14 text-right font-mono">{fmt ? fmt(value) : value.toFixed(2)}</span>
    </label>
  );
}

/** The annotation panel explaining the current step. */
export function Narration({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface-2/60 p-3 text-sm" aria-live="polite">
      <div className="mb-1 font-medium">{title}</div>
      <div className="text-muted">{children}</div>
    </div>
  );
}

export const fmt = (v: number, d = 3) => (Number.isFinite(v) ? (Math.abs(v) < 1e-4 && v !== 0 ? v.toExponential(1) : v.toFixed(d)) : "NaN");

/**
 * Diverging colour for a signed value (negative → clay, positive → accent),
 * using CSS variables so it adapts to light/dark mode.
 */
export function signedColor(v: number, maxAbs = 1) {
  const a = Math.min(1, Math.abs(v) / (maxAbs || 1));
  const pct = Math.round(15 + a * 85);
  return v >= 0 ? `color-mix(in srgb, var(--accent) ${pct}%, transparent)` : `color-mix(in srgb, var(--clay) ${pct}%, transparent)`;
}

/** Sequential colour for magnitudes in [0,1] (e.g. attention weights, pixels). */
export function seqColor(v: number) {
  const pct = Math.round(Math.max(0, Math.min(1, v)) * 100);
  return `color-mix(in srgb, var(--accent) ${pct}%, var(--surface))`;
}
