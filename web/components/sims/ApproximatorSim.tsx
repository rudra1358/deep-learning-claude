"use client";

import { useId, useMemo, useState } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { GRID_X, SAMPLES, TARGETS, X_MAX, X_MIN, deadUnits, kinksInDomain, makeData, predict, train, unitOut, type OptId, type SampleId, type TargetId } from "@/lib/sim/approximator";

const FRAMES = 200;
const ITERS_PER_FRAME = 20; // one Step = 20 full-batch training iterations
const W = 340;
const H = 250;
const P = { l: 30, r: 8, t: 10, b: 22 };
const YMIN = -1.3;
const YMAX = 1.3;
const LOG_MIN = -6;
const LOG_MAX = 0;

const DEFAULT_LR: Record<OptId, number> = { adam: 0.02, gd: 0.05 };

export default function ApproximatorSim() {
  const [target, setTarget] = useState<TargetId>("bumps");
  const [sample, setSample] = useState<SampleId>("dense");
  const [N, setN] = useState(12);
  const [opt, setOpt] = useState<OptId>("adam");
  const [logLr, setLogLr] = useState(Math.log10(DEFAULT_LR.adam));
  const [seed, setSeed] = useState(1);
  const [showUnits, setShowUnits] = useState(true);
  const lr = 10 ** logLr;
  const clip = useId().replace(/:/g, "");

  const s = useStepper(FRAMES + 1, { baseMs: 140 });
  const data = useMemo(() => makeData(target, sample), [target, sample]);
  const frames = useMemo(() => train({ N, seed, data, target, opt, lr, frames: FRAMES, itersPerFrame: ITERS_PER_FRAME }), [N, seed, data, target, opt, lr]);
  const fr = frames[s.step];
  const p = fr.params;
  const tgt = TARGETS.find((t) => t.id === target)!;

  const fx = (x: number) => P.l + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - P.l - P.r);
  const fy = (y: number) => P.t + ((YMAX - y) / (YMAX - YMIN)) * (H - P.t - P.b);
  const curve = (fn: (x: number) => number) => {
    let d = "";
    GRID_X.forEach((x, i) => {
      const y = Math.max(YMIN - 1, Math.min(YMAX + 1, fn(x)));
      d += `${i ? "L" : "M"}${fx(x).toFixed(2)},${fy(y).toFixed(2)} `;
    });
    return d;
  };

  const kinks = kinksInDomain(p);
  const dead = deadUnits(p, data.x);
  const pieces = kinks.length + 1;

  // MSE history (log scale)
  const ex = (k: number) => P.l + (k / FRAMES) * (W - P.l - P.r);
  const ey = (v: number) => {
    const lv = Math.max(LOG_MIN, Math.min(LOG_MAX, Math.log10(Math.max(v, 1e-12))));
    return P.t + ((LOG_MAX - lv) / (LOG_MAX - LOG_MIN)) * (H - P.t - P.b);
  };
  const hist = (key: "trainMse" | "trueMse") => frames.slice(0, s.step + 1).map((f, k) => `${k ? "L" : "M"}${ex(k).toFixed(1)},${ey(f[key]).toFixed(1)}`).join(" ");

  /* ---------- narration ---------- */
  const iter = fr.iter;
  const sparse = sample !== "dense";
  let title: string;
  let text: string;
  if (s.step === 0) {
    title = "Iteration 0 — random initialisation";
    text = `The network is ŷ(x) = Σⱼ vⱼ·max(0, aⱼx + cⱼ) + d with N = ${N} hidden ReLUs (${3 * N + 1} parameters). Each unit contributes one hinge that bends at x = −cⱼ/aⱼ; the kinks were initialised uniformly over [−1, 1] and the output weights vⱼ are small, so ŷ starts as a gentle piecewise-linear curve. Press Play: every Step runs ${ITERS_PER_FRAME} iterations of ${opt === "adam" ? "Adam" : "gradient descent"} on the MSE over the ${data.x.length} training points.`;
  } else {
    const fitWord = fr.trainMse < 1e-3 ? "fits the training points closely" : fr.trainMse < 1e-2 ? "roughly follows the training points" : "cannot yet follow the training points";
    title = `Iteration ${iter} — ${sparse ? "fitting 10 samples" : `N = ${N} hidden units`}`;
    const parts: string[] = [`Training MSE ${fmt(fr.trainMse, 5)}; the network ${fitWord}. Its output is piecewise linear with ${pieces} piece${pieces > 1 ? "s" : ""} on [−1, 1] (never more than N + 1 = ${N + 1}).`];
    if (sparse) {
      parts.push(
        `Error against the true ${tgt.formula} on a dense grid: ${fmt(fr.trueMse, 5)} — ${fr.trueMse > 1000 * fr.trainMse ? "more than 1000× the training error" : fr.trueMse > 5 * fr.trainMse ? `${fmt(fr.trueMse / fr.trainMse, 0)}× the training error` : "close to the training error"}. Between the samples nothing constrains the curve. The universal approximation theorem says a network this size can represent a good fit; it says nothing about whether gradient descent on ${data.x.length} points will find it.`,
      );
      if (sample === "noisy") parts.push("With noisy labels, driving the training MSE below the noise variance (0.0225) means memorising the noise.");
    } else {
      parts.push(`On dense samples the training error and the error against the target (${fmt(fr.trueMse, 5)}) agree, because the points cover the whole interval.`);
      if (target === "abs") parts.push(N >= 2 ? "|x| − 0.5 = max(0, x) + max(0, −x) − 0.5 is exactly a 2-unit ReLU network, so the error can reach (numerically) zero — if training finds that arrangement." : "One hinge can bend only once; |x| needs two.");
      if (target === "step") parts.push("A ReLU network is continuous, so it can only replace the jump by a steep ramp; the error concentrates at x = 0.1. More units let other pieces stay flat while one gets steep.");
      if (target !== "abs" && N <= 4) parts.push(`With only ${N} unit${N > 1 ? "s" : ""} there are too few pieces for this curve: increase N and watch the error floor drop.`);
    }
    if (dead > 0) parts.push(`${dead} unit${dead > 1 ? "s are" : " is"} zero on every training point, so ${dead > 1 ? "they receive" : "it receives"} no gradient ("dead" on this data).`);
    text = parts.join(" ");
  }

  const unitColor = (v: number) => (v >= 0 ? "var(--accent)" : "var(--clay)");

  return (
    <SimShell
      title="Universal Approximation"
      subtitle="A one-hidden-layer ReLU network trained in your browser to fit a 1-D target. Every curve shown is computed from the current weights."
      controls={
        <Controls
          s={s}
          extra={
            <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Re-initialise the weights with a new seed">
              <Shuffle size={14} /> New init
            </button>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <select className="field text-xs" value={target} onChange={(e) => setTarget(e.target.value as TargetId)} aria-label="Target function">
              {TARGETS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <select className="field text-xs" value={sample} onChange={(e) => setSample(e.target.value as SampleId)} aria-label="Training samples">
              {SAMPLES.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
          <Slider label="hidden units N" value={N} min={1} max={50} step={1} onChange={setN} fmt={(v) => String(v)} />
          <div className="flex items-center gap-2 text-xs">
            <span className="w-24 shrink-0 text-muted">optimiser</span>
            {(["adam", "gd"] as const).map((o) => (
              <button
                key={o}
                className={`sim-btn ${opt === o ? "primary" : ""}`}
                aria-pressed={opt === o}
                onClick={() => {
                  setOpt(o);
                  setLogLr(Math.log10(DEFAULT_LR[o]));
                }}
              >
                {o === "adam" ? "Adam" : "GD"}
              </button>
            ))}
          </div>
          <Slider label="learning rate" value={logLr} min={-3} max={-0.5} step={0.05} onChange={setLogLr} fmt={() => fmt(lr, lr < 0.01 ? 4 : 3)} />
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={showUnits} onChange={(e) => setShowUnits(e.target.checked)} />
            show each unit&apos;s hinge vⱼ·hⱼ(x)
          </label>
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">train MSE</div>
              <div className="font-mono text-sm text-accent">{fmt(fr.trainMse, 5)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">MSE vs target</div>
              <div className="font-mono text-sm text-clay">{fmt(fr.trueMse, 5)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">iteration</div>
              <div className="font-mono text-sm">{iter}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">linear pieces / params</div>
              <div className="font-mono text-sm">
                {pieces} / {3 * N + 1}
              </div>
            </div>
          </div>
          <Narration title={title}>{text}</Narration>
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <figure>
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Target ${tgt.formula}, training points and network output after ${iter} iterations`}>
            <defs>
              <clipPath id={clip}>
                <rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} />
              </clipPath>
            </defs>
            {[-1, -0.5, 0, 0.5, 1].map((v) => (
              <g key={v}>
                <line x1={P.l} x2={W - P.r} y1={fy(v)} y2={fy(v)} stroke="var(--border)" strokeWidth={v === 0 ? 1.1 : 0.6} />
                <text x={P.l - 4} y={fy(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  {v}
                </text>
                <text x={fx(v)} y={H - 7} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
                  {v}
                </text>
              </g>
            ))}
            <g clipPath={`url(#${clip})`}>
              {showUnits &&
                p.v.map((v, j) => (
                  <path key={j} d={curve((x) => unitOut(p, j, x))} fill="none" stroke={unitColor(v)} strokeOpacity={0.45} strokeWidth={1} />
                ))}
              <path d={curve(tgt.f)} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="5 4" />
              <path d={curve((x) => predict(p, x))} fill="none" stroke="var(--gold)" strokeWidth={2.4} />
              {data.x.map((x, i) => (
                <circle key={i} cx={fx(x)} cy={fy(data.y[i])} r={sparse ? 3.6 : 2.2} fill="var(--accent)" />
              ))}
            </g>
            {kinks.map((k, i) => (
              <line key={i} x1={fx(k)} x2={fx(k)} y1={H - P.b} y2={H - P.b - 6} stroke="var(--gold)" strokeWidth={1.2} />
            ))}
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
            <span>– – target {tgt.formula}</span>
            <span className="text-accent">● training points</span>
            <span className="text-gold">— network ŷ(x) · ticks = kinks</span>
            {showUnits && <span>thin: vⱼ·hⱼ(x) (accent v &gt; 0, clay v &lt; 0)</span>}
          </figcaption>
        </figure>

        <figure>
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Training MSE and MSE against the target, log scale, over training">
            {[-6, -5, -4, -3, -2, -1, 0].map((lv) => (
              <g key={lv}>
                <line x1={P.l} x2={W - P.r} y1={ey(10 ** lv)} y2={ey(10 ** lv)} stroke="var(--border)" strokeWidth={0.6} />
                <text x={P.l - 4} y={ey(10 ** lv) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  1e{lv}
                </text>
              </g>
            ))}
            {[0, 50, 100, 150, 200].map((k) => (
              <text key={k} x={ex(k)} y={H - 7} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
                {k * ITERS_PER_FRAME}
              </text>
            ))}
            <path d={hist("trueMse")} fill="none" stroke="var(--clay)" strokeWidth={2} />
            <path d={hist("trainMse")} fill="none" stroke="var(--accent)" strokeWidth={2} />
            <line x1={ex(s.step)} x2={ex(s.step)} y1={P.t} y2={H - P.b} stroke="var(--gold)" strokeWidth={1} />
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
            <span className="text-accent">— train MSE</span>
            <span className="text-clay">— MSE vs target (dense grid)</span>
            <span>x: iteration · y: log scale</span>
          </figcaption>
        </figure>
      </div>
    </SimShell>
  );
}
