"use client";

import { useEffect, useId, useMemo, useState } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { MAX_DEGREE, evalPoly, makeData, regimeOf, sweepDegrees, trueFn } from "@/lib/sim/fitting";

const W = 340;
const H = 260;
const P = { l: 32, r: 10, t: 10, b: 24 };
const YMIN = -2;
const YMAX = 2;
const LOG_MIN = -4; // 1e-4
const LOG_MAX = 2; // 1e2

export default function FittingSim() {
  const [n, setN] = useState(30);
  const [sigma, setSigma] = useState(0.25);
  const [seed, setSeed] = useState(7);
  const clip = useId().replace(/:/g, "");

  const s = useStepper(MAX_DEGREE + 1, { baseMs: 1100 });
  const { setStep } = s;
  useEffect(() => {
    setStep(3);
  }, [setStep]);
  const degree = s.step;

  const data = useMemo(() => makeData(n, sigma, seed), [n, sigma, seed]);
  const results = useMemo(() => sweepDegrees(data), [data]);
  const cur = results[degree];
  const { regime, best } = regimeOf(results, degree);

  // ---- data plot ----
  const fx = (x: number) => P.l + x * (W - P.l - P.r);
  const fy = (y: number) => P.t + ((YMAX - y) / (YMAX - YMIN)) * (H - P.t - P.b);
  const curve = (fn: (x: number) => number) => {
    let d = "";
    for (let i = 0; i <= 240; i++) {
      const x = i / 240;
      const y = Math.max(YMIN - 1, Math.min(YMAX + 1, fn(x)));
      d += `${i ? "L" : "M"}${fx(x).toFixed(2)},${fy(y).toFixed(2)} `;
    }
    return d;
  };

  // ---- error chart ----
  const ex = (d: number) => P.l + (d / MAX_DEGREE) * (W - P.l - P.r);
  const ey = (v: number) => {
    const lv = Math.max(LOG_MIN, Math.min(LOG_MAX, Math.log10(Math.max(v, 1e-12))));
    return P.t + ((LOG_MAX - lv) / (LOG_MAX - LOG_MIN)) * (H - P.t - P.b);
  };
  const line = (key: "trainMse" | "valMse") => results.map((r, i) => `${i ? "L" : "M"}${ex(r.degree).toFixed(2)},${ey(r[key]).toFixed(2)}`).join(" ");

  const noiseVar = sigma * sigma;
  const nTrain = data.xTrain.length;
  const params = degree + 1;

  let title: string;
  let text: string;
  if (regime === "underfit") {
    title = `Degree ${degree}: underfitting`;
    text = `A degree-${degree} polynomial is too simple to follow sin(2πx). Training MSE (${fmt(cur.trainMse)}) and validation MSE (${fmt(cur.valMse)}) are both high and close to each other — the error is dominated by bias. Adding capacity will help; validation error is lowest at degree ${best}.`;
  } else if (regime === "good") {
    title = `Degree ${degree}: good fit`;
    text = `Validation MSE ${fmt(cur.valMse)} is near its minimum (best: degree ${best}, ${fmt(results[best].valMse)}). The noise variance is σ² = ${fmt(noiseVar)}, which is the lowest validation MSE any model could expect on average; the fit captures the signal without chasing the noise.`;
  } else {
    title = `Degree ${degree}: overfitting`;
    text = `Training MSE keeps falling (${fmt(cur.trainMse)}${cur.trainMse < noiseVar ? `, already below the noise variance ${fmt(noiseVar)} — the curve is fitting noise` : ""}) while validation MSE rises to ${fmt(cur.valMse)}. ${params >= nTrain ? `With ${params} coefficients and only ${nTrain} training points the polynomial can pass through (almost) every training point. ` : ""}Variance dominates: a new sample would give a very different curve. More data or a lower degree fixes it.`;
  }

  return (
    <SimShell
      title="Underfitting vs Overfitting"
      subtitle="Least-squares polynomial fits to noisy samples of sin(2πx); filled = training, hollow = validation."
      controls={
        <Controls
          s={s}
          extra={
            <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Draw a new random sample">
              <Shuffle size={14} /> New sample
            </button>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <Slider label="degree d" value={degree} min={0} max={MAX_DEGREE} step={1} onChange={(v) => setStep(v)} fmt={(v) => String(v)} />
          <Slider label="samples n" value={n} min={10} max={200} step={1} onChange={setN} fmt={(v) => String(v)} />
          <Slider label="noise σ" value={sigma} min={0} max={0.5} step={0.01} onChange={setSigma} />
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">train MSE</div>
              <div className="font-mono text-sm text-accent">{fmt(cur.trainMse, 4)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">validation MSE</div>
              <div className="font-mono text-sm text-clay">{fmt(cur.valMse, 4)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">train / val points</div>
              <div className="font-mono text-sm">
                {nTrain} / {data.xVal.length}
              </div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">coefficients</div>
              <div className="font-mono text-sm">{params}</div>
            </div>
          </div>
          <Narration title={title}>{text}</Narration>
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <figure>
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Data and degree-${degree} polynomial fit`}>
            <defs>
              <clipPath id={clip}>
                <rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} />
              </clipPath>
            </defs>
            {[-2, -1, 0, 1, 2].map((v) => (
              <g key={v}>
                <line x1={P.l} x2={W - P.r} y1={fy(v)} y2={fy(v)} stroke="var(--border)" strokeWidth={v === 0 ? 1.1 : 0.6} />
                <text x={P.l - 4} y={fy(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  {v}
                </text>
              </g>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <text key={v} x={fx(v)} y={H - 8} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
                {v}
              </text>
            ))}
            <g clipPath={`url(#${clip})`}>
              <path d={curve(trueFn)} fill="none" stroke="var(--text-muted)" strokeWidth={1.4} strokeDasharray="5 4" />
              <path d={curve((x) => evalPoly(cur.coef, x))} fill="none" stroke="var(--gold)" strokeWidth={2.2} />
              {data.xTrain.map((x, i) => (
                <circle key={`t${i}`} cx={fx(x)} cy={fy(data.yTrain[i])} r={3.2} fill="var(--accent)" />
              ))}
              {data.xVal.map((x, i) => (
                <circle key={`v${i}`} cx={fx(x)} cy={fy(data.yVal[i])} r={3.2} fill="var(--surface)" stroke="var(--clay)" strokeWidth={1.5} />
              ))}
            </g>
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
            <span className="text-accent">● train</span>
            <span className="text-clay">○ validation</span>
            <span>– – true sin(2πx)</span>
            <span className="text-gold">— degree-{degree} fit</span>
          </figcaption>
        </figure>

        <figure>
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Training and validation MSE against polynomial degree, log scale">
            {[-4, -3, -2, -1, 0, 1, 2].map((lv) => (
              <g key={lv}>
                <line x1={P.l} x2={W - P.r} y1={ey(10 ** lv)} y2={ey(10 ** lv)} stroke="var(--border)" strokeWidth={0.6} />
                <text x={P.l - 4} y={ey(10 ** lv) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  1e{lv}
                </text>
              </g>
            ))}
            {sigma > 0 && (
              <g>
                <line x1={P.l} x2={W - P.r} y1={ey(noiseVar)} y2={ey(noiseVar)} stroke="var(--text-muted)" strokeDasharray="2 3" />
                <text x={W - P.r - 2} y={ey(noiseVar) - 3} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
                  σ² (noise floor)
                </text>
              </g>
            )}
            {results.map((r) => (
              <text key={r.degree} x={ex(r.degree)} y={H - 8} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
                {r.degree % 3 === 0 ? r.degree : ""}
              </text>
            ))}
            <line x1={ex(degree)} x2={ex(degree)} y1={P.t} y2={H - P.b} stroke="var(--gold)" strokeWidth={1.5} />
            <path d={line("trainMse")} fill="none" stroke="var(--accent)" strokeWidth={2} />
            <path d={line("valMse")} fill="none" stroke="var(--clay)" strokeWidth={2} />
            {results.map((r) => (
              <g key={`m${r.degree}`}>
                <circle cx={ex(r.degree)} cy={ey(r.trainMse)} r={2.4} fill="var(--accent)" />
                <circle cx={ex(r.degree)} cy={ey(r.valMse)} r={r.degree === best ? 5 : 2.4} fill={r.degree === best ? "var(--surface)" : "var(--clay)"} stroke="var(--clay)" strokeWidth={r.degree === best ? 2 : 0} />
              </g>
            ))}
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
            <span className="text-accent">— train MSE</span>
            <span className="text-clay">— validation MSE (ring = best, d = {best})</span>
            <span>x-axis: degree 0–15, y: log scale</span>
          </figcaption>
        </figure>
      </div>
    </SimShell>
  );
}
