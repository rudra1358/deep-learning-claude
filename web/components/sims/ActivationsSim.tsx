"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { ACTIVATIONS, softmaxT, type ActParams, type ActivationId } from "@/lib/sim/activations";

const XMIN = -5;
const XMAX = 5;
const N_PTS = 401;
const SWEEP_STEP = 0.1;
const SWEEP_TOTAL = Math.round((XMAX - XMIN) / SWEEP_STEP) + 1; // 101 positions

const W = 560;
const H = 330;
const PL = 36;
const PR = 12;
const PT = 12;
const PB = 26;

/** Colours in selection order — CSS variables only so light/dark both work. */
const PALETTE = [
  "var(--accent)",
  "var(--clay)",
  "var(--gold)",
  "var(--text)",
  "color-mix(in srgb, var(--accent) 50%, var(--text))",
  "color-mix(in srgb, var(--clay) 50%, var(--text))",
  "color-mix(in srgb, var(--gold) 50%, var(--text))",
  "var(--text-muted)",
  "color-mix(in srgb, var(--accent) 50%, var(--gold))",
  "color-mix(in srgb, var(--clay) 50%, var(--gold))",
  "color-mix(in srgb, var(--accent) 40%, var(--clay))",
  "color-mix(in srgb, var(--text-muted) 50%, var(--gold))",
  "color-mix(in srgb, var(--text-muted) 50%, var(--accent))",
];

const DEFAULT_LOGITS = [2.0, 1.0, 0.1, -1.0];

function niceTicks(lo: number, hi: number) {
  const span = hi - lo;
  const stepRaw = span / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(stepRaw)));
  const m = stepRaw / mag;
  const st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

export default function ActivationsSim() {
  const [selected, setSelected] = useState<ActivationId[]>(["sigmoid", "tanh", "relu"]);
  const [showDeriv, setShowDeriv] = useState(true);
  const [alpha, setAlpha] = useState(0.1);
  const [beta, setBeta] = useState(1);
  const [x, setX] = useState(1);
  const [logits, setLogits] = useState<number[]>(DEFAULT_LOGITS);
  const [texts, setTexts] = useState<string[]>(DEFAULT_LOGITS.map(String));
  const [temp, setTemp] = useState(1);
  const svgRef = useRef<SVGSVGElement>(null);

  const s = useStepper(SWEEP_TOTAL, { baseMs: 120 });
  const { step, playing } = s;
  useEffect(() => {
    if (playing || step > 0) setX(Math.round((XMIN + step * SWEEP_STEP) * 10) / 10);
  }, [step, playing]);

  const p: ActParams = { alpha, beta };
  const defs = ACTIVATIONS.filter((a) => selected.includes(a.id)).sort((a, b) => selected.indexOf(a.id) - selected.indexOf(b.id));
  const colorOf = (id: ActivationId) => PALETTE[selected.indexOf(id) % PALETTE.length];

  const curves = useMemo(() => {
    const prm: ActParams = { alpha, beta };
    return ACTIVATIONS.filter((a) => selected.includes(a.id)).map((a) => {
      const xs = Array.from({ length: N_PTS }, (_, i) => XMIN + ((XMAX - XMIN) * i) / (N_PTS - 1));
      return { id: a.id, f: xs.map((v) => [v, a.f(v, prm)] as [number, number]), df: xs.map((v) => [v, a.df(v, prm)] as [number, number]) };
    });
  }, [selected, alpha, beta]);

  // y-range from everything drawn, clamped so linear/SELU do not squash the rest
  let ylo = -1.2;
  let yhi = 1.2;
  for (const c of curves) {
    for (const [, v] of c.f) {
      ylo = Math.min(ylo, v);
      yhi = Math.max(yhi, v);
    }
    if (showDeriv)
      for (const [, v] of c.df) {
        ylo = Math.min(ylo, v);
        yhi = Math.max(yhi, v);
      }
  }
  ylo = Math.max(-5.5, ylo - 0.2);
  yhi = Math.min(5.5, yhi + 0.2);

  const px = (v: number) => PL + ((v - XMIN) / (XMAX - XMIN)) * (W - PL - PR);
  const py = (v: number) => PT + ((yhi - Math.max(ylo, Math.min(yhi, v))) / (yhi - ylo)) * (H - PT - PB);

  const path = (pts: [number, number][]) => {
    let d = "";
    let prev: number | null = null;
    for (const [a, v] of pts) {
      // break the path at jumps (binary step) instead of drawing a vertical line
      const jump = prev !== null && Math.abs(v - prev) > 0.5;
      d += `${d === "" || jump ? "M" : "L"}${px(a).toFixed(2)},${py(v).toFixed(2)} `;
      prev = v;
    }
    return d;
  };

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r || s.playing) return;
    const vx = ((e.clientX - r.left) / r.width) * W;
    const xv = XMIN + ((vx - PL) / (W - PL - PR)) * (XMAX - XMIN);
    setX(Math.round(Math.max(XMIN, Math.min(XMAX, xv)) * 100) / 100);
  };

  const toggle = (id: ActivationId) => setSelected((cur) => (cur.includes(id) ? cur.filter((c) => c !== id) : [...cur, id]));

  // narration about the probe point
  const notes: string[] = [];
  for (const a of defs) {
    const d = a.df(x, p);
    if (a.id === "step") notes.push(`Binary step has slope 0 at x = ${fmt(x, 2)} (and everywhere except the jump), so no gradient can flow through it.`);
    else if (a.id === "relu" && x <= 0) notes.push(`ReLU is in its flat region: output 0 and slope 0. A unit whose input is negative for every example receives no gradient (a "dead ReLU").`);
    else if ((a.id === "sigmoid" || a.id === "tanh") && Math.abs(d) < 0.05) notes.push(`${a.name} is saturated: slope ${fmt(d)}, so any gradient passing back through it is multiplied by ${fmt(d)} and nearly vanishes.`);
  }
  if (notes.length === 0 && defs.length > 0) {
    const a = defs[0];
    notes.push(`At x = ${fmt(x, 2)}, ${a.name} outputs ${fmt(a.f(x, p))} with slope ${fmt(a.df(x, p))}: during backpropagation the upstream gradient is multiplied by ${fmt(a.df(x, p))} when it passes through this unit.`);
  }

  const sm = softmaxT(logits, temp);
  const BW = 320;
  const BH = 150;

  const setLogit = (i: number, txt: string) => {
    setTexts((t) => t.map((v, j) => (j === i ? txt : v)));
    const v = parseFloat(txt);
    if (Number.isFinite(v)) setLogits((l) => l.map((u, j) => (j === i ? v : u)));
  };

  const xTicks = niceTicks(XMIN, XMAX);
  const yTicks = niceTicks(ylo, yhi);

  return (
    <SimShell
      title="Activation Function Explorer"
      subtitle="Solid: f(x). Dashed: f′(x). Hover the plot or drag the x slider; Play sweeps x from −5 to 5."
      controls={<Controls s={s} />}
      aside={
        <div className="space-y-3">
          <Slider label="probe x" value={x} min={XMIN} max={XMAX} step={0.01} onChange={(v) => setX(v)} fmt={(v) => v.toFixed(2)} />
          <Slider label="α (Leaky, PReLU, ELU)" value={alpha} min={0} max={1} step={0.01} onChange={setAlpha} />
          <Slider label="β (Swish)" value={beta} min={0.1} max={5} step={0.1} onChange={setBeta} fmt={(v) => v.toFixed(1)} />
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={showDeriv} onChange={(e) => setShowDeriv(e.target.checked)} aria-label="Show derivatives" /> Show derivatives (dashed)
          </label>
          <table className="w-full border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-border text-faint">
                <th className="py-1 text-left font-sans font-normal">function</th>
                <th className="text-right font-normal">f(x)</th>
                <th className="text-right font-normal">f′(x)</th>
              </tr>
            </thead>
            <tbody>
              {defs.map((a) => (
                <tr key={a.id} className="border-b border-border/60">
                  <td className="py-1 pr-1 font-sans" style={{ color: colorOf(a.id) }}>
                    {a.name}
                  </td>
                  <td className="text-right">{fmt(a.f(x, p))}</td>
                  <td className="text-right">{fmt(a.df(x, p))}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Narration title={`At x = ${fmt(x, 2)}`}>{notes.join(" ")}</Narration>
        </div>
      }
    >
      <div className="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="Select activation functions">
        {ACTIVATIONS.map((a) => {
          const on = selected.includes(a.id);
          return (
            <button
              key={a.id}
              className="sim-btn"
              onClick={() => toggle(a.id)}
              aria-pressed={on}
              title={`${a.formula} — range ${a.range}`}
              style={on ? { borderColor: colorOf(a.id), boxShadow: `inset 0 -3px 0 ${colorOf(a.id)}` } : { opacity: 0.7 }}
            >
              {a.name}
            </button>
          );
        })}
      </div>

      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="h-auto w-full touch-none" role="img" aria-label="Plot of the selected activation functions and their derivatives" onMouseMove={onMove}>
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line x1={PL} x2={W - PR} y1={py(v)} y2={py(v)} stroke="var(--border)" strokeWidth={v === 0 ? 1.2 : 0.6} />
            <text x={PL - 4} y={py(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
              {v}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <g key={`x${v}`}>
            <line x1={px(v)} x2={px(v)} y1={PT} y2={H - PB} stroke="var(--border)" strokeWidth={v === 0 ? 1.2 : 0.6} />
            <text x={px(v)} y={H - PB + 13} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
              {v}
            </text>
          </g>
        ))}
        {curves.map((c) => (
          <g key={c.id}>
            <path d={path(c.f)} fill="none" stroke={colorOf(c.id)} strokeWidth={2} />
            {showDeriv && <path d={path(c.df)} fill="none" stroke={colorOf(c.id)} strokeWidth={1.4} strokeDasharray="5 4" />}
          </g>
        ))}
        <line x1={px(x)} x2={px(x)} y1={PT} y2={H - PB} stroke="var(--gold)" strokeWidth={1} strokeDasharray="2 2" />
        {defs.map((a) => (
          <g key={`p${a.id}`}>
            <circle cx={px(x)} cy={py(a.f(x, p))} r={3.5} fill={colorOf(a.id)} stroke="var(--surface)" />
            {showDeriv && <circle cx={px(x)} cy={py(a.df(x, p))} r={3} fill="var(--surface)" stroke={colorOf(a.id)} strokeWidth={1.5} />}
          </g>
        ))}
        <text x={W - PR - 2} y={H - PB - 4} fontSize="10" textAnchor="end" fill="var(--text-muted)">
          x
        </text>
      </svg>
      {defs.length === 0 && <p className="text-sm text-muted">Select at least one function above.</p>}

      <div className="mt-5 border-t border-border pt-4">
        <div className="mb-2 font-serif font-semibold">Softmax with temperature</div>
        <p className="mb-3 text-sm text-muted">
          pᵢ = exp((zᵢ − m)/T) / Σⱼ exp((zⱼ − m)/T), with m the largest zᵢ (subtracted so exp never overflows; it cancels in the ratio). Edit the logits and change T.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              {texts.map((t, i) => (
                <label key={i} className="flex items-center gap-1 text-xs text-muted">
                  z{i + 1}
                  <input className="field w-16 font-mono" inputMode="decimal" value={t} onChange={(e) => setLogit(i, e.target.value)} aria-label={`Logit ${i + 1}`} />
                </label>
              ))}
            </div>
            <Slider label="temperature T" value={temp} min={0.1} max={5} step={0.05} onChange={setTemp} />
            <button
              className="sim-btn"
              onClick={() => {
                setLogits(DEFAULT_LOGITS);
                setTexts(DEFAULT_LOGITS.map(String));
                setTemp(1);
              }}
              aria-label="Reset logits and temperature"
            >
              Reset logits
            </button>
            <table className="w-full border-collapse text-right font-mono text-xs">
              <thead>
                <tr className="border-b border-border text-faint">
                  <th className="py-1 text-left font-normal">i</th>
                  <th className="font-normal">z/T</th>
                  <th className="font-normal">exp(z/T − m)</th>
                  <th className="font-normal">p</th>
                </tr>
              </thead>
              <tbody>
                {sm.probs.map((pr, i) => (
                  <tr key={i} className="border-b border-border/60">
                    <td className="py-1 text-left">{i + 1}</td>
                    <td>{fmt(sm.scaled[i])}</td>
                    <td>{fmt(sm.exps[i], 4)}</td>
                    <td className="text-accent">{fmt(pr)}</td>
                  </tr>
                ))}
                <tr>
                  <td className="py-1 text-left text-faint" colSpan={2}>
                    sum (m = {fmt(sm.max)})
                  </td>
                  <td>{fmt(sm.sum, 4)}</td>
                  <td>{fmt(sm.probs.reduce((a, b) => a + b, 0))}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <svg viewBox={`0 0 ${BW} ${BH}`} className="h-auto w-full" role="img" aria-label="Bar chart of softmax probabilities">
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={30} x2={BW - 6} y1={BH - 20 - v * (BH - 34)} y2={BH - 20 - v * (BH - 34)} stroke="var(--border)" />
                  <text x={26} y={BH - 17 - v * (BH - 34)} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                    {v}
                  </text>
                </g>
              ))}
              {sm.probs.map((pr, i) => {
                const bw = (BW - 40) / sm.probs.length;
                const h = pr * (BH - 34);
                const xx = 34 + i * bw;
                const top = sm.probs.indexOf(Math.max(...sm.probs)) === i;
                return (
                  <g key={i}>
                    <rect x={xx + 6} y={BH - 20 - h} width={bw - 12} height={h} fill={top ? "var(--gold)" : "var(--accent)"} rx={2} />
                    <text x={xx + bw / 2} y={BH - 24 - h} fontSize="9" textAnchor="middle" fill="var(--text)">
                      {pr.toFixed(3)}
                    </text>
                    <text x={xx + bw / 2} y={BH - 6} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
                      z{i + 1}={logits[i]}
                    </text>
                  </g>
                );
              })}
            </svg>
            <p className="mt-1 text-xs text-muted">
              {temp < 0.5
                ? "Low T sharpens the distribution towards a one-hot vector on the largest logit (argmax)."
                : temp > 2
                  ? "High T flattens the distribution towards uniform (1/4 each); the ranking of the logits never changes."
                  : "At T = 1 this is the ordinary softmax. Adding the same constant to every logit leaves p unchanged."}
            </p>
          </div>
        </div>
      </div>
    </SimShell>
  );
}
