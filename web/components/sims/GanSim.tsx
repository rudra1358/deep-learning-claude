"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper, type Stepper } from "./SimShell";
import {
  EXTENT,
  HQ_RADIUS,
  LOSSES,
  MODE_FRACTION,
  N_MODES,
  PRESETS,
  RING_RADIUS,
  TARGETS,
  GanTrainer,
  nParams,
  referencePR,
  type GanConfig,
  type LossId,
  type Metrics,
  type TargetId,
} from "@/lib/sim/gan";

const CW = 480; // canvas resolution (CSS-scaled to the container width)
const HEAT_RES = 48;
const N_REAL_DRAWN = 500;
const N_ARROWS = 48;
const STEP_SIZES = [1, 10, 100, 500];

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
const rgb = (c: RGB, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

interface Palette {
  surface: RGB;
  accent: RGB;
  clay: RGB;
  gold: RGB;
  text: RGB;
  muted: RGB;
  border: RGB;
}
function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const g = (n: string, f: RGB) => parseColor(cs.getPropertyValue(n), f);
  return {
    surface: g("--surface", [255, 255, 255]),
    accent: g("--accent", [61, 122, 107]),
    clay: g("--clay", [181, 111, 85]),
    gold: g("--gold", [168, 131, 47]),
    text: g("--text", [30, 30, 30]),
    muted: g("--text-muted", [95, 100, 107]),
    border: g("--border", [220, 220, 220]),
  };
}

interface MetricRec extends Metrics {
  dominant: number; // ring: index of the fullest mode (−1 if no mode has ≥ 10 samples)
}

const isClassifier = (l: LossId) => l === "minimax" || l === "nonsat";
const lrFmt = (v: number) => (v >= 0.01 ? v.toFixed(2) : v >= 0.001 ? v.toFixed(4) : v.toExponential(0));

/* ---------------------------------------------------------------- narration */

function narrate(cfg: GanConfig, m: MetricRec | null, hist: MetricRec[], ref: { precision: number; recall: number }): { title: string; text: string } {
  const lossName = LOSSES.find((l) => l.id === cfg.loss)!.label;
  if (!m || m.iter === 0) {
    return {
      title: "Iteration 0 — two untrained networks",
      text: `G maps ${cfg.zDim}-D Gaussian noise through two hidden layers of ${cfg.hidden} leaky-ReLU units to a point in the plane; its ${600} samples (gold) start as a small blob. D (2 → ${cfg.hidden} → ${cfg.hidden} → 1) shades the background by how "real" it thinks each location is. Press Play: each iteration does ${cfg.dSteps} D update${cfg.dSteps > 1 ? "s" : ""} on a batch of ${cfg.batch} real and ${cfg.batch} generated points, then one G update that back-propagates the ${lossName} generator loss through D into G.`,
    };
  }
  const parts: string[] = [];
  const cls = isClassifier(cfg.loss);
  const ring = cfg.target === "ring";
  let state = "G chasing D";
  if (cls) parts.push(`Mean D on real data ${fmt(m.dReal, 2)}, on generated data ${fmt(m.dFake, 2)} (the theoretical equilibrium is 0.5 on both).`);
  else if (cfg.loss === "wgangp") parts.push(`Critic gap E[f(real)] − E[f(fake)] = ${fmt(m.dReal - m.dFake, 2)}: an estimate of the Wasserstein-1 distance (up to scale), which shrinks as G improves.`);
  else parts.push(`LSGAN targets: D → 1 on real, 0 on fake. Mean D(real) ${fmt(m.dReal, 2)}, D(fake) ${fmt(m.dFake, 2)}; at equilibrium both approach 0.5.`);

  const dWins = cls && m.dReal > 0.75 && m.dFake < 0.2;
  const maxShare = Math.max(...m.modeCounts) / 600;
  const recent = hist.slice(-8).filter((h) => h.dominant >= 0 && h.modes <= 3);
  const hops = new Set(recent.map((h) => h.dominant)).size;

  if (ring && m.modes <= 2 && maxShare > 0.3) {
    state = "mode collapse";
    parts.push(`Mode collapse: ${fmt(100 * maxShare, 0)}% of all samples sit in a single mode and only ${m.modes} of 8 modes are covered. Every sample there looks real to D, so G has no incentive to spread out; D can only push it away by lowering D(x) at that mode, and G then follows the gradient to wherever D is now highest.`);
  } else if (ring && hops >= 3) {
    state = "oscillation";
    parts.push(`Oscillation: over the last few checkpoints the fullest mode changed ${hops - 1}+ times. G concentrates its samples, D learns to reject that region, G jumps to the next region D still likes — the alternating updates circle around the equilibrium instead of settling into it.`);
  } else if (ring && m.hq < 0.15 && m.recall < 0.5 && m.iter > 1500) {
    state = "collapsed off the data";
    parts.push(`G's samples are squeezed into a small region (only ${fmt(100 * m.recall, 0)}% of real points have a generated neighbour within 0.15) and few land on a mode (${fmt(100 * m.hq, 0)}% high quality). With G learning much faster than D, each G step overshoots to wherever D currently scores highest.`);
  } else if (dWins) {
    state = "D winning";
    parts.push(
      cfg.loss === "minimax"
        ? `D is winning: it rejects the fakes with high confidence. With the original loss G minimises log(1 − D(G(z))), whose gradient −σ(l)·∇l is weighted by D(G(z)) itself: samples D confidently rejects get almost no push. At this very D the saturating loss gives a mean push of ${fmt(m.satGrad, 3)} per sample, the non-saturating loss would give ${fmt(m.nsGrad, 3)} (${fmt(m.nsGrad / Math.max(m.satGrad, 1e-12), 1)}×). Try the non-saturating loss with the same learning rates.`
        : `D is winning, but the non-saturating loss −log D(G(z)) has gradient −(1 − σ(l))·∇l, which is largest for the samples D rejects: here a mean push of ${fmt(m.nsGrad, 3)} per sample, versus ${fmt(m.satGrad, 3)} with the original loss. The danger now is that G follows that strong signal to D's single favourite region.`,
    );
  } else if (ring && m.modes === N_MODES && m.hq > 0.6) {
    state = "all modes covered";
    parts.push(`All 8 modes hold at least ${MODE_FRACTION * 100}% of their fair share of samples and ${fmt(100 * m.hq, 0)}% of the samples lie within 3σ of a mode. D's background is nearly flat near the data: it can no longer tell real from generated, so G receives little further pressure.`);
  } else if (!ring && m.precision > 0.85 * ref.precision && m.recall > 0.85 * ref.recall) {
    state = "good fit";
    parts.push(`Precision ${fmt(m.precision, 2)} and recall ${fmt(m.recall, 2)} are close to the values a perfect generator would get with these sample sizes (${fmt(ref.precision, 2)} / ${fmt(ref.recall, 2)}): the generated cloud matches the data.`);
  } else {
    parts.push(
      ring
        ? `${m.modes} of 8 modes covered, ${fmt(100 * m.hq, 0)}% of samples within 3σ of a mode. G moves its samples up D's slope (the arrows show −∂L_G/∂x); D then re-draws its boundary around the new fakes.`
        : `Precision ${fmt(m.precision, 2)} (generated points near real ones) and recall ${fmt(m.recall, 2)} (real points near generated ones); a perfect generator scores ${fmt(ref.precision, 2)} / ${fmt(ref.recall, 2)}. G moves its samples up D's slope (arrows); D re-draws its boundary.`,
    );
  }
  return { title: `Iteration ${m.iter} — ${state}`, text: parts.join(" ") };
}

/* ---------------------------------------------------------------- component */

export default function GanSim() {
  const [cfg, setCfg] = useState<GanConfig>(PRESETS[0].cfg);
  const [presetId, setPresetId] = useState<string | null>(PRESETS[0].id);
  const [stepSize, setStepSize] = useState(100);
  const [showHeat, setShowHeat] = useState(true);
  const [showFlow, setShowFlow] = useState(true);
  const [, setTick] = useState(0);
  const [metrics, setMetrics] = useState<MetricRec | null>(null);
  const [itsPerSec, setItsPerSec] = useState(0);

  const s = useStepper(Infinity, { baseMs: 1e8 }); // playback is driven by requestAnimationFrame below
  const { setPlaying } = s;
  const trRef = useRef<GanTrainer | null>(null);
  const histRef = useRef<MetricRec[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement | null>(null);
  const heatVals = useRef<Float64Array | null>(null);
  const palRef = useRef<Palette | null>(null);

  const ref = useMemo(() => referencePR(cfg.target), [cfg.target]);

  /* ---- metrics, heatmap, drawing ---- */
  const computeMetrics = useCallback(() => {
    const tr = trRef.current;
    if (!tr) return;
    const m = tr.metrics();
    const max = Math.max(...m.modeCounts);
    const rec: MetricRec = { ...m, dominant: max >= 10 ? m.modeCounts.indexOf(max) : -1 };
    const h = histRef.current;
    if (!h.length || h[h.length - 1].iter !== rec.iter) h.push(rec);
    if (h.length > 400) histRef.current = h.filter((_, i) => i % 2 === 0 || i === h.length - 1);
    setMetrics(rec);
  }, []);

  const computeHeat = useCallback(() => {
    const tr = trRef.current;
    if (tr) heatVals.current = tr.dField(HEAT_RES);
  }, []);

  const draw = useCallback(() => {
    const cv = canvasRef.current, tr = trRef.current;
    if (!cv || !tr) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    if (!palRef.current) palRef.current = readPalette();
    const P = palRef.current;
    const sx = (x: number) => ((x + EXTENT) / (2 * EXTENT)) * CW;
    const sy = (y: number) => ((EXTENT - y) / (2 * EXTENT)) * CW;
    ctx.fillStyle = rgb(P.surface);
    ctx.fillRect(0, 0, CW, CW);

    // D(x) background
    if (showHeat && heatVals.current) {
      const v = heatVals.current;
      const cls = isClassifier(tr.cfg.loss);
      let center = 0.5, half = 0.5;
      if (!cls) {
        // LSGAN / WGAN: centre between the mean outputs on real and generated data, scale by the spread
        const dr = tr.dOutput(tr.realEval, 200), df = tr.dOutput(tr.evalSamples().subarray(0, 400), 200);
        const mr = dr.reduce((a, b) => a + b, 0) / 200, mf = df.reduce((a, b) => a + b, 0) / 200;
        center = (mr + mf) / 2;
        half = Math.max(Math.abs(mr - mf), 1e-3);
      }
      if (!heatRef.current) heatRef.current = document.createElement("canvas");
      const hc = heatRef.current;
      hc.width = HEAT_RES;
      hc.height = HEAT_RES;
      const hctx = hc.getContext("2d");
      if (hctx) {
        const img = hctx.createImageData(HEAT_RES, HEAT_RES);
        for (let k = 0; k < v.length; k++) {
          const t = Math.max(-1, Math.min(1, (v[k] - center) / half));
          const c = t >= 0 ? mix(P.surface, P.accent, 0.45 * t) : mix(P.surface, P.clay, -0.45 * t);
          img.data[4 * k] = c[0];
          img.data[4 * k + 1] = c[1];
          img.data[4 * k + 2] = c[2];
          img.data[4 * k + 3] = 255;
        }
        hctx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(hc, 0, 0, CW, CW);
      }
    }

    // grid + ring guides
    ctx.strokeStyle = rgb(P.border, 0.9);
    ctx.lineWidth = 1;
    for (let g = -EXTENT + 1; g < EXTENT; g++) {
      ctx.beginPath();
      ctx.moveTo(sx(g), 0);
      ctx.lineTo(sx(g), CW);
      ctx.moveTo(0, sy(g));
      ctx.lineTo(CW, sy(g));
      ctx.stroke();
    }
    if (tr.cfg.target === "ring") {
      ctx.strokeStyle = rgb(P.muted, 0.6);
      ctx.setLineDash([3, 3]);
      for (let k = 0; k < N_MODES; k++) {
        const a = (2 * Math.PI * k) / N_MODES;
        ctx.beginPath();
        ctx.arc(sx(RING_RADIUS * Math.cos(a)), sy(RING_RADIUS * Math.sin(a)), (HQ_RADIUS / (2 * EXTENT)) * CW, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // real samples
    ctx.fillStyle = rgb(P.text, 0.55);
    for (let i = 0; i < N_REAL_DRAWN; i++) {
      ctx.fillRect(sx(tr.realEval[2 * i]) - 1.2, sy(tr.realEval[2 * i + 1]) - 1.2, 2.4, 2.4);
    }
    // generated samples
    const gen = tr.evalSamples();
    ctx.fillStyle = rgb(P.gold, 0.85);
    for (let i = 0; i < tr.nEval; i++) {
      const x = sx(gen[2 * i]), y = sy(gen[2 * i + 1]);
      if (x < -5 || x > CW + 5 || y < -5 || y > CW + 5) continue;
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, 2 * Math.PI);
      ctx.fill();
    }
    // generator flow arrows: −∂L_G/∂x at a subset of generated points (normalised to the largest)
    if (showFlow) {
      const pts = gen.subarray(0, 2 * N_ARROWS);
      const fl = tr.flow(pts, N_ARROWS);
      let mx = 1e-12;
      for (let i = 0; i < N_ARROWS; i++) mx = Math.max(mx, Math.hypot(fl[2 * i], fl[2 * i + 1]));
      ctx.strokeStyle = rgb(P.text, 0.85);
      ctx.lineWidth = 1.2;
      for (let i = 0; i < N_ARROWS; i++) {
        const gx = fl[2 * i] / mx, gy = fl[2 * i + 1] / mx;
        const len = Math.hypot(gx, gy);
        if (len < 0.02) continue;
        const x0 = sx(pts[2 * i]), y0 = sy(pts[2 * i + 1]);
        const x1 = x0 + gx * 34, y1 = y0 - gy * 34;
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 5 * Math.cos(ang - 0.45), y1 - 5 * Math.sin(ang - 0.45));
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 5 * Math.cos(ang + 0.45), y1 - 5 * Math.sin(ang + 0.45));
        ctx.stroke();
      }
    }
  }, [showHeat, showFlow]);

  const refreshAll = useCallback(() => {
    computeHeat();
    computeMetrics();
    draw();
    setTick((t) => t + 1);
  }, [computeHeat, computeMetrics, draw]);

  /* ---- (re)build the trainer when a structural setting changes ---- */
  const rebuild = useCallback(
    (c: GanConfig) => {
      trRef.current = new GanTrainer(c);
      histRef.current = [];
      setPlaying(false);
      refreshAll();
    },
    [refreshAll, setPlaying],
  );
  const structKey = `${cfg.target}|${cfg.loss}|${cfg.zDim}|${cfg.seed}|${cfg.gpLambda}`;
  useEffect(() => {
    rebuild(cfg);
    // only structural changes rebuild; learning rates / D steps are applied live below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structKey]);
  useEffect(() => {
    trRef.current?.setHyper({ lrG: cfg.lrG, lrD: cfg.lrD, dSteps: cfg.dSteps });
  }, [cfg.lrG, cfg.lrD, cfg.dSteps]);

  // theme changes → re-read canvas colours
  useEffect(() => {
    const obs = new MutationObserver(() => {
      palRef.current = null;
      draw();
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    return () => obs.disconnect();
  }, [draw]);
  useEffect(() => draw(), [draw]);

  /* ---- playback: batches of iterations per animation frame ---- */
  useEffect(() => {
    if (!s.playing) return;
    let raf = 0, frame = 0, itCount = 0, tStart = performance.now();
    const loop = () => {
      const tr = trRef.current;
      if (!tr) return;
      const t0 = performance.now();
      const want = Math.max(1, Math.round(s.speed * 4));
      let n = 0;
      while (n < want && performance.now() - t0 < 14) {
        tr.trainIter();
        n++;
      }
      itCount += n;
      frame++;
      if (frame % 3 === 0) computeHeat();
      if (frame % 12 === 0) computeMetrics();
      draw();
      if (frame % 2 === 0) setTick((t) => t + 1);
      const now = performance.now();
      if (now - tStart > 1000) {
        setItsPerSec(Math.round((itCount * 1000) / (now - tStart)));
        itCount = 0;
        tStart = now;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      computeMetrics();
    };
  }, [s.playing, s.speed, computeHeat, computeMetrics, draw]);

  const stepper: Stepper = {
    ...s,
    next: () => {
      trRef.current?.train(stepSize);
      refreshAll();
    },
    reset: () => rebuild(cfg),
  };

  const tr = trRef.current;
  const hist = tr?.history ?? [];
  const { title, text } = narrate(cfg, metrics, histRef.current, ref);
  const update = (p: Partial<GanConfig>) => {
    setCfg((c) => ({ ...c, ...p }));
    setPresetId(null);
  };
  const applyPreset = (id: string) => {
    const p = PRESETS.find((q) => q.id === id)!;
    setPresetId(id);
    setCfg(p.cfg);
    if (p.cfg.target === cfg.target && p.cfg.loss === cfg.loss && p.cfg.zDim === cfg.zDim && p.cfg.seed === cfg.seed && p.cfg.gpLambda === cfg.gpLambda) {
      // same structure: the rebuild effect will not fire, so restart explicitly with the preset's rates
      rebuild(p.cfg);
    }
  };
  const preset = PRESETS.find((p) => p.id === presetId);

  /* ---- loss chart ---- */
  const LW = 360, LH = 150, PL = 34, PR = 8, PT = 8, PB = 20;
  const lossChart = useMemo(() => {
    if (hist.length < 2) return null;
    const vals = hist.flatMap((h) => [h.dLoss, h.gLoss]).filter(Number.isFinite).sort((a, b) => a - b);
    let lo = vals[Math.floor(0.01 * (vals.length - 1))], hi = vals[Math.ceil(0.99 * (vals.length - 1))];
    if (hi - lo < 1e-6) {
      hi += 0.5;
      lo -= 0.5;
    }
    const pad = 0.08 * (hi - lo);
    lo -= pad;
    hi += pad;
    const last = hist[hist.length - 1].iter;
    const x = (it: number) => PL + (it / Math.max(1, last)) * (LW - PL - PR);
    const y = (v: number) => PT + ((hi - Math.max(lo, Math.min(hi, v))) / (hi - lo)) * (LH - PT - PB);
    const path = (k: "dLoss" | "gLoss") => hist.map((h, i) => `${i ? "L" : "M"}${x(h.iter).toFixed(1)},${y(h[k]).toFixed(1)}`).join(" ");
    const ticks = [lo + pad, (lo + hi) / 2, hi - pad];
    return { d: path("dLoss"), g: path("gLoss"), ticks, y, last };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hist.length, hist[hist.length - 1]?.iter]);

  /* ---- coverage-over-time chart ---- */
  const mh = histRef.current;
  const ring = cfg.target === "ring";
  const RX0 = 150; // left edge of the coverage-over-time plot (the per-mode bars sit to its left)
  const lastIt = Math.max(1, mh.length ? mh[mh.length - 1].iter : 1);
  const covX = (it: number, x0: number) => x0 + (it / lastIt) * (LW - x0 - PR);
  const covY = (v: number) => PT + (1 - v) * (LH - PT - PB);
  const covPath = (f: (m: MetricRec) => number, x0 = PL) => mh.map((m, i) => `${i ? "L" : "M"}${covX(m.iter, x0).toFixed(1)},${covY(f(m)).toFixed(1)}`).join(" ");

  const lossLabels: Record<LossId, [string, string]> = {
    minimax: ["D: −log D(x) − log(1 − D(G(z)))", "G: log(1 − D(G(z)))"],
    nonsat: ["D: −log D(x) − log(1 − D(G(z)))", "G: −log D(G(z))"],
    lsgan: ["D: ½(D(x) − 1)² + ½D(G(z))²", "G: ½(D(G(z)) − 1)²"],
    wgangp: ["critic: f(fake) − f(real) + GP", "G: −f(G(z))"],
  };

  return (
    <SimShell
      title="GAN Training"
      subtitle="A real generator and discriminator trained live in your browser (hand-written backprop + Adam). Gold = generated samples, dark = real data, background = D(x)."
      controls={
        <Controls
          s={stepper}
          canPrev={false}
          extra={
            <div className="flex flex-wrap items-center gap-2">
              <label className="flex items-center gap-1 text-xs text-muted">
                Step =
                <select className="field text-xs" value={stepSize} onChange={(e) => setStepSize(+e.target.value)} aria-label="Iterations per Step">
                  {STEP_SIZES.map((k) => (
                    <option key={k} value={k}>
                      {k} it.
                    </option>
                  ))}
                </select>
              </label>
              <button className="sim-btn" onClick={() => update({ seed: cfg.seed + 1 })} aria-label="Re-initialise both networks with a new seed">
                <Shuffle size={14} /> New seed
              </button>
            </div>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <div>
            <div className="mb-1 text-xs text-muted">Presets (verified runs)</div>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button key={p.id} className={`sim-btn text-xs ${presetId === p.id ? "primary" : ""}`} aria-pressed={presetId === p.id} onClick={() => applyPreset(p.id)}>
                  {p.label}
                </button>
              ))}
            </div>
            {preset && <p className="mt-1 text-xs text-muted">{preset.note}</p>}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <select className="field min-w-0 text-xs" value={cfg.target} onChange={(e) => update({ target: e.target.value as TargetId })} aria-label="Target distribution">
              {TARGETS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <select className="field min-w-0 text-xs" value={cfg.loss} onChange={(e) => update({ loss: e.target.value as LossId, dSteps: e.target.value === "wgangp" ? 5 : cfg.dSteps })} aria-label="GAN loss">
              {LOSSES.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
          <Slider label="G learning rate" value={Math.log10(cfg.lrG)} min={-4} max={-1.5} step={0.05} onChange={(v) => update({ lrG: +(10 ** v).toPrecision(2) })} fmt={() => lrFmt(cfg.lrG)} />
          <Slider label="D learning rate" value={Math.log10(cfg.lrD)} min={-4} max={-1.5} step={0.05} onChange={(v) => update({ lrD: +(10 ** v).toPrecision(2) })} fmt={() => lrFmt(cfg.lrD)} />
          <Slider label="D steps per G step" value={cfg.dSteps} min={1} max={5} step={1} onChange={(v) => update({ dSteps: v })} fmt={(v) => String(v)} />
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="w-24 shrink-0 text-muted">noise dim</span>
            {[2, 4, 8].map((z) => (
              <button key={z} className={`sim-btn ${cfg.zDim === z ? "primary" : ""}`} aria-pressed={cfg.zDim === z} onClick={() => update({ zDim: z })}>
                {z}
              </button>
            ))}
          </div>
          {cfg.loss === "wgangp" && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">GP weight λ</span>
              {[0.1, 1, 10].map((l) => (
                <button key={l} className={`sim-btn ${cfg.gpLambda === l ? "primary" : ""}`} aria-pressed={cfg.gpLambda === l} onClick={() => update({ gpLambda: l })}>
                  {l}
                </button>
              ))}
            </div>
          )}
          <div className="flex flex-wrap gap-3 text-xs text-muted">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> D(x) background
            </label>
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={showFlow} onChange={(e) => setShowFlow(e.target.checked)} /> G gradient arrows
            </label>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">iteration</div>
              <div className="font-mono text-sm">{tr?.iter ?? 0}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">speed</div>
              <div className="font-mono text-sm">{s.playing ? `${itsPerSec} it/s` : "paused"}</div>
            </div>
            {ring ? (
              <>
                <div className="rounded-md bg-surface-2 p-1.5">
                  <div className="text-faint">modes covered</div>
                  <div className="font-mono text-sm text-accent">{metrics ? `${metrics.modes} / 8` : "–"}</div>
                </div>
                <div className="rounded-md bg-surface-2 p-1.5">
                  <div className="text-faint">high-quality (≤3σ)</div>
                  <div className="font-mono text-sm text-gold">{metrics ? `${fmt(100 * metrics.hq, 1)}%` : "–"}</div>
                </div>
              </>
            ) : (
              <>
                <div className="rounded-md bg-surface-2 p-1.5">
                  <div className="text-faint">precision (ideal {fmt(ref.precision, 2)})</div>
                  <div className="font-mono text-sm text-accent">{metrics ? fmt(metrics.precision, 2) : "–"}</div>
                </div>
                <div className="rounded-md bg-surface-2 p-1.5">
                  <div className="text-faint">recall (ideal {fmt(ref.recall, 2)})</div>
                  <div className="font-mono text-sm text-gold">{metrics ? fmt(metrics.recall, 2) : "–"}</div>
                </div>
              </>
            )}
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">{isClassifier(cfg.loss) ? "mean D(real)" : "mean out(real)"}</div>
              <div className="font-mono text-sm">{metrics ? fmt(metrics.dReal, 2) : "–"}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">{isClassifier(cfg.loss) ? "mean D(fake)" : "mean out(fake)"}</div>
              <div className="font-mono text-sm">{metrics ? fmt(metrics.dFake, 2) : "–"}</div>
            </div>
          </div>
          <Narration title={title}>{text}</Narration>
          <p className="text-xs text-faint">
            G: {cfg.zDim}→{cfg.hidden}→{cfg.hidden}→2 ({tr ? nParams(tr.G) : 0} params) · D: 2→{cfg.hidden}→{cfg.hidden}→1 ({tr ? nParams(tr.D) : 0} params) · batch {cfg.batch} · Adam β = (0.5, 0.9). Metrics use 600 fixed noise vectors and 1000 real points.
          </p>
        </div>
      }
    >
      <figure className="mx-auto max-w-[480px]">
        <canvas ref={canvasRef} width={CW} height={CW} className="block h-auto w-full rounded-md border border-border" role="img" aria-label={`Real samples, generated samples and the discriminator output over the plane after ${tr?.iter ?? 0} iterations`} />
        <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
          <span className="text-gold">● generated G(z)</span>
          <span>■ real data</span>
          <span className="text-accent">background: D says real</span>
          <span className="text-clay">· D says fake</span>
          {showFlow && <span>→ −∂L_G/∂x (where G is pushing its samples)</span>}
          {ring && <span>dashed: 3σ around each mode</span>}
        </figcaption>
      </figure>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <figure className="min-w-0">
          <svg viewBox={`0 0 ${LW} ${LH}`} className="h-auto w-full" role="img" aria-label="Discriminator and generator losses over training">
            <line x1={PL} x2={LW - PR} y1={LH - PB} y2={LH - PB} stroke="var(--border)" />
            {lossChart?.ticks.map((t, i) => (
              <g key={i}>
                <line x1={PL} x2={LW - PR} y1={lossChart.y(t)} y2={lossChart.y(t)} stroke="var(--border)" strokeWidth={0.6} />
                <text x={PL - 4} y={lossChart.y(t) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  {fmt(t, Math.abs(t) >= 10 ? 0 : 2)}
                </text>
              </g>
            ))}
            {lossChart && (
              <>
                <path d={lossChart.d} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
                <path d={lossChart.g} fill="none" stroke="var(--clay)" strokeWidth={1.5} />
                <text x={LW - PR} y={LH - 6} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  iteration {lossChart.last}
                </text>
              </>
            )}
            {!lossChart && (
              <text x={LW / 2} y={LH / 2} fontSize="11" textAnchor="middle" fill="var(--text-muted)">
                losses appear after 20 iterations
              </text>
            )}
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
            <span className="text-accent">— {lossLabels[cfg.loss][0]}</span>
            <span className="text-clay">— {lossLabels[cfg.loss][1]}</span>
          </figcaption>
        </figure>

        <figure className="min-w-0">
          {ring ? (
            <svg viewBox={`0 0 ${LW} ${LH}`} className="h-auto w-full" role="img" aria-label="Samples per mode and mode coverage over training">
              {/* per-mode bars (left third) */}
              {(metrics?.modeCounts ?? new Array(8).fill(0)).map((c, k) => {
                const bw = 11, x0 = 6 + k * (bw + 3), hMax = LH - PT - PB;
                const h = Math.min(1, c / 150) * hMax;
                const covered = c >= (MODE_FRACTION * 600) / N_MODES;
                return (
                  <g key={k}>
                    <rect x={x0} y={LH - PB - h} width={bw} height={h} fill={covered ? "var(--accent)" : "var(--clay)"} opacity={0.85} />
                    <text x={x0 + bw / 2} y={LH - 8} fontSize="8" textAnchor="middle" fill="var(--text-muted)">
                      {k}
                    </text>
                  </g>
                );
              })}
              <line x1={4} x2={116} y1={LH - PB - ((MODE_FRACTION * 600) / N_MODES / 150) * (LH - PT - PB)} y2={LH - PB - ((MODE_FRACTION * 600) / N_MODES / 150) * (LH - PT - PB)} stroke="var(--gold)" strokeDasharray="3 2" />
              {/* coverage over time (right) */}
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={RX0} x2={LW - PR} y1={covY(v)} y2={covY(v)} stroke="var(--border)" strokeWidth={v === 0 ? 1 : 0.6} />
                  <text x={RX0 - 4} y={covY(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                    {v}
                  </text>
                </g>
              ))}
              {mh.length > 1 && <path d={covPath((m) => m.modes / 8, RX0)} fill="none" stroke="var(--accent)" strokeWidth={1.5} />}
              {mh.length > 1 && <path d={covPath((m) => m.hq, RX0)} fill="none" stroke="var(--gold)" strokeWidth={1.5} />}
              {mh.length > 1 && (
                <text x={LW - PR} y={LH - 6} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  iteration {mh[mh.length - 1].iter}
                </text>
              )}
            </svg>
          ) : (
            <svg viewBox={`0 0 ${LW} ${LH}`} className="h-auto w-full" role="img" aria-label="Precision and recall over training">
              <line x1={PL} x2={LW - PR} y1={covY(0)} y2={covY(0)} stroke="var(--border)" />
              {[0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={PL} x2={LW - PR} y1={covY(v)} y2={covY(v)} stroke="var(--border)" strokeWidth={0.6} />
                  <text x={PL - 4} y={covY(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                    {v}
                  </text>
                </g>
              ))}
              <line x1={PL} x2={LW - PR} y1={covY(ref.precision)} y2={covY(ref.precision)} stroke="var(--accent)" strokeDasharray="3 3" strokeWidth={0.8} />
              <line x1={PL} x2={LW - PR} y1={covY(ref.recall)} y2={covY(ref.recall)} stroke="var(--gold)" strokeDasharray="3 3" strokeWidth={0.8} />
              {mh.length > 1 && <path d={covPath((m) => m.precision)} fill="none" stroke="var(--accent)" strokeWidth={1.5} />}
              {mh.length > 1 && <path d={covPath((m) => m.recall)} fill="none" stroke="var(--gold)" strokeWidth={1.5} />}
            </svg>
          )}
          <figcaption className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
            {ring ? (
              <>
                <span>bars: high-quality samples per mode (dashed = coverage threshold, {MODE_FRACTION * 100}% of fair share)</span>
                <span className="text-accent">— modes covered / 8</span>
                <span className="text-gold">— high-quality fraction</span>
              </>
            ) : (
              <>
                <span className="text-accent">— precision</span>
                <span className="text-gold">— recall</span>
                <span>dashed: perfect-generator values</span>
              </>
            )}
          </figcaption>
        </figure>
      </div>
    </SimShell>
  );
}
