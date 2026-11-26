"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { HYPER, OPTIMIZERS, SURFACES, runOptimizer, sampleGrid, type OptName, type OptRun, type Vec2 } from "@/lib/sim/optimizers";

const N_STEPS = 600;
const W = 480;
const BANDS = 16;

/** Distinct but calm per-optimizer styles: colour (theme tokens only) + dash pattern. */
const STYLE: Record<OptName, { color: string; dash?: string; width: number }> = {
  sgd: { color: "var(--text-muted)", width: 1.6 },
  momentum: { color: "var(--clay)", width: 1.8 },
  nesterov: { color: "var(--gold)", dash: "7 4", width: 1.8 },
  adagrad: { color: "color-mix(in srgb, var(--accent) 60%, var(--text))", dash: "2 3", width: 2 },
  rmsprop: { color: "color-mix(in srgb, var(--clay) 50%, var(--text))", dash: "9 3 2 3", width: 1.8 },
  adam: { color: "var(--text)", width: 2 },
};

type RGB = [number, number, number];

function parseColor(s: string, fallback: RGB): RGB {
  const v = s.trim();
  let m = /^#([0-9a-f]{6})$/i.exec(v);
  if (m) {
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  m = /^#([0-9a-f]{3})$/i.exec(v);
  if (m) return [0, 1, 2].map((i) => parseInt(m![1][i] + m![1][i], 16)) as RGB;
  const r = /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(v);
  if (r) return [+r[1], +r[2], +r[3]];
  return fallback;
}

const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Increments whenever <html data-theme> changes, so canvas colours can be re-read. */
function useThemeVersion() {
  const [v, setV] = useState(0);
  useEffect(() => {
    const obs = new MutationObserver(() => setV((x) => x + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    return () => obs.disconnect();
  }, []);
  return v;
}

const lrFmt = (v: number) => (v >= 0.01 ? v.toFixed(v >= 0.1 ? 2 : 3) : v.toExponential(0));

export default function OptimizersSim() {
  const [surfaceId, setSurfaceId] = useState(SURFACES[0].id);
  const surface = SURFACES.find((s) => s.id === surfaceId)!;
  const [start, setStart] = useState<Vec2>(surface.start);
  const [logLrSgd, setLogLrSgd] = useState(Math.log10(surface.lrSgd));
  const [logLrAd, setLogLrAd] = useState(Math.log10(surface.lrAdaptive));
  const [enabled, setEnabled] = useState<Record<OptName, boolean>>({ sgd: true, momentum: true, nesterov: true, adagrad: true, rmsprop: true, adam: true });
  const lrSgd = 10 ** logLrSgd;
  const lrAd = 10 ** logLrAd;

  const H = Math.round((W * (surface.yRange[1] - surface.yRange[0])) / (surface.xRange[1] - surface.xRange[0]));
  const s = useStepper(N_STEPS + 1, { baseMs: 70 });
  const { setStep } = s;

  const runs = useMemo(() => {
    const out = {} as Record<OptName, OptRun>;
    for (const o of OPTIMIZERS) out[o.id] = runOptimizer(surface, o.id, start, o.adaptive ? lrAd : lrSgd, N_STEPS);
    return out;
  }, [surface, start, lrSgd, lrAd]);

  // Any parameter change restarts the race from step 0.
  useEffect(() => setStep(0), [runs, setStep]);

  /* ---------- heatmap + contour bands on canvas ---------- */
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const grid = useMemo(() => sampleGrid(surface, W, H), [surface, H]);
  const theme = useThemeVersion();
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const css = getComputedStyle(document.documentElement);
    const surf = parseColor(css.getPropertyValue("--surface"), [255, 255, 255]);
    const acc = parseColor(css.getPropertyValue("--accent"), [61, 122, 107]);
    const line = parseColor(css.getPropertyValue("--text-muted"), [95, 100, 107]);
    const img = ctx.createImageData(W, H);
    // Non-log surfaces use sqrt so that more contour bands sit near the minimum.
    const band = (t: number) => Math.min(BANDS - 1, Math.floor((surface.logScale ? t : Math.sqrt(t)) * BANDS));
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const k = j * W + i;
        const t = grid[k];
        const b = band(t);
        const edge = (i + 1 < W && band(grid[k + 1]) !== b) || (j + 1 < H && band(grid[k + W]) !== b);
        let c = mix(surf, acc, 0.5 * (1 - b / (BANDS - 1)));
        if (edge) c = mix(c, line, 0.35);
        img.data[4 * k] = c[0];
        img.data[4 * k + 1] = c[1];
        img.data[4 * k + 2] = c[2];
        img.data[4 * k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [grid, theme, H, surface.logScale]);

  /* ---------- coordinate mapping ---------- */
  const [x0, x1] = surface.xRange;
  const [y0, y1] = surface.yRange;
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * W;
  const sy = (y: number) => ((y1 - y) / (y1 - y0)) * H;

  function onPick(e: MouseEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const py = ((e.clientY - r.top) / r.height) * H;
    const x = x0 + (px / W) * (x1 - x0);
    const y = y1 - (py / H) * (y1 - y0);
    setStart([+x.toFixed(3), +y.toFixed(3)]);
  }

  function chooseSurface(id: string) {
    const sf = SURFACES.find((q) => q.id === id)!;
    setSurfaceId(id);
    setStart(sf.start);
    setLogLrSgd(Math.log10(sf.lrSgd));
    setLogLrAd(Math.log10(sf.lrAdaptive));
  }

  const k = s.step;
  const rows = OPTIMIZERS.map((o) => {
    const run = runs[o.id];
    const idx = Math.min(k, run.path.length - 1);
    const p = run.path[idx];
    const divergedNow = run.diverged && k >= run.path.length;
    return { o, run, idx, p, divergedNow };
  });
  const active = rows.filter((r) => enabled[r.o.id]);
  const finite = active.filter((r) => !r.divergedNow);
  const leader = finite.length ? finite.reduce((a, b) => (b.p.loss < a.p.loss ? b : a)) : null;
  const diverged = active.filter((r) => r.divergedNow).map((r) => r.o.label);

  const clampPt = (v: number, lo: number, hi: number) => Math.max(lo - 20, Math.min(hi + 20, v));

  return (
    <SimShell
      title="Optimizer Race"
      subtitle="Six optimizers race downhill on the same 2-D loss surface. Click the map to pick where they start."
      controls={
        <div className="space-y-2.5">
          <Controls s={s} />
          <div className="flex flex-wrap gap-1.5">
            {SURFACES.map((sf) => (
              <button key={sf.id} className={`sim-btn ${sf.id === surfaceId ? "primary" : ""}`} onClick={() => chooseSurface(sf.id)} aria-pressed={sf.id === surfaceId}>
                {sf.label}
              </button>
            ))}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="η (SGD family)" value={logLrSgd} min={-4} max={0} step={0.05} onChange={setLogLrSgd} fmt={() => lrFmt(lrSgd)} />
            <Slider label="η (adaptive)" value={logLrAd} min={-3} max={0} step={0.05} onChange={setLogLrAd} fmt={() => lrFmt(lrAd)} />
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            {OPTIMIZERS.map((o) => (
              <label key={o.id} className="flex items-center gap-1.5">
                <input type="checkbox" checked={enabled[o.id]} onChange={(e) => setEnabled({ ...enabled, [o.id]: e.target.checked })} aria-label={`Show ${o.label}`} />
                <svg width="22" height="8" aria-hidden="true">
                  <line x1="1" y1="4" x2="21" y2="4" stroke={STYLE[o.id].color} strokeWidth={2.2} strokeDasharray={STYLE[o.id].dash} />
                </svg>
                {o.label}
              </label>
            ))}
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={`${surface.label}: ${surface.formula}`}>
            <p className="mb-1.5">{surface.note}</p>
            {k === 0 ? (
              <p>Every optimizer starts at ({fmt(start[0], 2)}, {fmt(start[1], 2)}). Press Play or Step.</p>
            ) : (
              <p>
                After {k} step{k > 1 ? "s" : ""}
                {leader ? (
                  <>
                    , the lowest loss so far is <strong className="text-text">{leader.o.label}</strong> (f = {fmt(leader.p.loss, 4)}).
                  </>
                ) : (
                  "."
                )}
                {diverged.length > 0 && <> Blew up (diverged): {diverged.join(", ")}. Lower its η.</>}
              </p>
            )}
          </Narration>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted">
                  <th className="py-1 pr-2 font-medium">Optimizer</th>
                  <th className="py-1 pr-2 font-medium">η</th>
                  <th className="py-1 pr-2 font-medium">steps</th>
                  <th className="py-1 font-medium">loss</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {rows.map(({ o, idx, p, divergedNow }) => (
                  <tr key={o.id} className={`border-t border-border ${enabled[o.id] ? "" : "opacity-40"}`}>
                    <td className="py-1 pr-2 font-sans">
                      <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: STYLE[o.id].color }} />
                      {o.label}
                    </td>
                    <td className="py-1 pr-2">{lrFmt(o.adaptive ? lrAd : lrSgd)}</td>
                    <td className="py-1 pr-2">{idx}</td>
                    <td className={`py-1 ${divergedNow ? "text-clay" : ""}`}>{divergedNow ? "diverged" : fmt(p.loss, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-faint">
            β = {HYPER.beta} (Momentum, Nesterov), ρ = {HYPER.rho} (RMSProp), β₁ = {HYPER.beta1}, β₂ = {HYPER.beta2} (Adam), ε = 1e-8. Adaptive methods move each coordinate by about η per step, so their η uses a different scale from SGD&apos;s.
          </p>
        </div>
      }
    >
      <div className="relative mx-auto w-full max-w-[560px]">
        <canvas ref={canvasRef} width={W} height={H} className="block h-auto w-full rounded-md border border-border" aria-hidden="true" />
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="absolute inset-0 h-full w-full cursor-crosshair"
          role="img"
          aria-label={`Contour map of ${surface.label} with optimizer trajectories; click to set the start point`}
          onClick={onPick}
        >
          {surface.minima.map((m, i) => (
            <g key={i} stroke="var(--text)" strokeWidth={1.4} opacity={0.7}>
              <line x1={sx(m[0]) - 4} y1={sy(m[1]) - 4} x2={sx(m[0]) + 4} y2={sy(m[1]) + 4} />
              <line x1={sx(m[0]) - 4} y1={sy(m[1]) + 4} x2={sx(m[0]) + 4} y2={sy(m[1]) - 4} />
            </g>
          ))}
          {active.map(({ o, run, idx }) => {
            const pts = run.path
              .slice(0, idx + 1)
              .map((p) => `${clampPt(sx(p.x), 0, W).toFixed(1)},${clampPt(sy(p.y), 0, H).toFixed(1)}`)
              .join(" ");
            const cur = run.path[idx];
            return (
              <g key={o.id}>
                <polyline points={pts} fill="none" stroke={STYLE[o.id].color} strokeWidth={STYLE[o.id].width} strokeDasharray={STYLE[o.id].dash} strokeLinejoin="round" />
                <circle cx={clampPt(sx(cur.x), 0, W)} cy={clampPt(sy(cur.y), 0, H)} r={4} fill={STYLE[o.id].color} stroke="var(--surface)" strokeWidth={1.2} />
              </g>
            );
          })}
          <circle cx={sx(start[0])} cy={sy(start[1])} r={6} fill="none" stroke="var(--text)" strokeWidth={1.5} />
        </svg>
      </div>
      <div className="mt-2 flex flex-wrap justify-between gap-2 font-mono text-[11px] text-faint">
        <span>
          x ∈ [{x0}, {x1}], y ∈ [{y0}, {y1}]
        </span>
        <span>darker tint = lower loss{surface.logScale ? " (log scale)" : ""} · × = lowest point · ○ = start</span>
      </div>
    </SimShell>
  );
}
