"use client";

import { useMemo, useState, type MouseEvent } from "react";
import { Undo2 } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { DOMAIN, PRESETS, lineInBox, majorityBaseline, makePreset, sigmoid, trainLogReg, type Label, type PresetId, type Pt } from "@/lib/sim/logreg";

const STEPS = 400; // gradient-descent iterations precomputed per configuration
const S = 300; // data plot size (viewBox units)
const PAD = 22;
const GRID = 24; // probability shading cells per side
const CW = 300; // chart width
const CH = 120;
const CP = { l: 28, r: 8, t: 8, b: 18 };

const toPx = (v: number) => PAD + ((v + DOMAIN) / (2 * DOMAIN)) * (S - 2 * PAD);
const toPy = (v: number) => S - PAD - ((v + DOMAIN) / (2 * DOMAIN)) * (S - 2 * PAD);
const fromPx = (px: number) => ((px - PAD) / (S - 2 * PAD)) * 2 * DOMAIN - DOMAIN;
const fromPy = (py: number) => ((S - PAD - py) / (S - 2 * PAD)) * 2 * DOMAIN - DOMAIN;

const LN3 = Math.log(3); // z = ±ln 3  ⇔  p = 0.75 / 0.25

export default function LogisticRegressionSim() {
  const [preset, setPreset] = useState<PresetId>("separable");
  const [pts, setPts] = useState<Pt[]>(() => makePreset("separable"));
  const [nPreset, setNPreset] = useState(() => makePreset("separable").length);
  const [cls, setCls] = useState<Label>(1);
  const [logLr, setLogLr] = useState(Math.log10(0.5));
  const [lambda, setLambda] = useState(0);
  const lr = 10 ** logLr;

  const s = useStepper(STEPS + 1, { baseMs: 160 });
  const { setStep } = s;

  const hist = useMemo(() => trainLogReg(pts, lr, lambda, STEPS), [pts, lr, lambda]);
  const cur = hist[s.step];
  const last = hist[STEPS];
  const nextW: [number, number] = [cur.w[0] - lr * cur.gw[0], cur.w[1] - lr * cur.gw[1]];
  const nextB = cur.b - lr * cur.gb;
  const wNorm = Math.hypot(cur.w[0], cur.w[1]);
  const baseline = majorityBaseline(pts);

  const loadPreset = (id: PresetId) => {
    const p = makePreset(id);
    setPreset(id);
    setPts(p);
    setNPreset(p.length);
    setStep(0);
  };

  const addPoint = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * S;
    const py = ((e.clientY - r.top) / r.height) * S;
    const x1 = fromPx(px), x2 = fromPy(py);
    if (Math.abs(x1) > DOMAIN || Math.abs(x2) > DOMAIN) return;
    setPts((old) => [...old, { x1, x2, y: cls }]);
  };

  /* ---------- probability shading: p(x) at each cell centre ---------- */
  const cells = useMemo(() => {
    const out: { x: number; y: number; p: number }[] = [];
    const step = (2 * DOMAIN) / GRID;
    for (let i = 0; i < GRID; i++)
      for (let j = 0; j < GRID; j++) {
        const x1 = -DOMAIN + (i + 0.5) * step, x2 = -DOMAIN + (j + 0.5) * step;
        out.push({ x: x1, y: x2, p: sigmoid(cur.w[0] * x1 + cur.w[1] * x2 + cur.b) });
      }
    return out;
  }, [cur]);
  const cell = (S - 2 * PAD) / GRID;

  const boundary = lineInBox(cur.w, cur.b, 0);
  const band75 = lineInBox(cur.w, cur.b, LN3);
  const band25 = lineInBox(cur.w, cur.b, -LN3);

  // weight-vector arrow: unit normal to the boundary, drawn from the boundary point closest to the origin
  let arrow: { x0: number; y0: number; x1: number; y1: number } | null = null;
  if (wNorm > 1e-9) {
    const u = [cur.w[0] / wNorm, cur.w[1] / wNorm];
    const foot = [(-cur.b / wNorm) * u[0], (-cur.b / wNorm) * u[1]];
    if (Math.abs(foot[0]) < DOMAIN - 0.3 && Math.abs(foot[1]) < DOMAIN - 0.3)
      arrow = { x0: foot[0], y0: foot[1], x1: foot[0] + u[0], y1: foot[1] + u[1] };
  }

  /* ---------- per-class recall for the narration ---------- */
  let tp = 0, pos = 0, tn = 0, neg = 0;
  for (const p of pts) {
    const pred = sigmoid(cur.w[0] * p.x1 + cur.w[1] * p.x2 + cur.b) > 0.5 ? 1 : 0;
    if (p.y === 1) {
      pos++;
      if (pred === 1) tp++;
    } else {
      neg++;
      if (pred === 0) tn++;
    }
  }

  /* ---------- history chart ---------- */
  const cx = (t: number) => CP.l + (t / STEPS) * (CW - CP.l - CP.r);
  const cy = (v: number) => CP.t + (1 - Math.max(0, Math.min(1, v))) * (CH - CP.t - CP.b);
  const path = (f: (i: number) => number) => {
    let d = "";
    for (let i = 0; i <= s.step; i++) d += `${i ? "L" : "M"}${cx(i).toFixed(1)},${cy(f(i)).toFixed(1)} `;
    return d;
  };

  /* ---------- narration ---------- */
  const diverged = !Number.isFinite(cur.objective) || hist.slice(1, s.step + 1).some((h, i) => h.objective > hist[i].objective + 1e-9);
  const separable = last.accuracy === 1 && pts.length > 0;
  const gNorm = Math.hypot(cur.gw[0], cur.gw[1], cur.gb);
  let title: string;
  let text: string;
  if (pts.length === 0 || pos === 0 || neg === 0) {
    title = "Need both classes";
    text = "Add at least one point of each class. With only one class, the model can always lower the loss by pushing the bias toward ±∞. So there is nothing useful to learn.";
  } else if (s.step === 0) {
    title = "Step 0 — every parameter is zero";
    text = `With w = 0 and b = 0, every point gets z = 0 and p = σ(0) = 0.5. So BCE = ln 2 = 0.693, whatever the data. The gradient (1/N)Σ(p − y)x is −½ × (sum of class-1 positions) + ½ × (sum of class-0 positions), divided by N. It points from the class-1 points toward the class-0 points. So the first update −η∇ turns w toward class 1. Press Step to apply it.`;
  } else if (diverged) {
    title = `Step ${s.step} — the learning rate is too large`;
    text = `The loss went up at least once along the way (now ${fmt(cur.objective, 4)}). A step of η = ${fmt(lr, 2)} is so big it jumps past the lowest point. This loss is convex (one bowl, no other dips), so a smaller η is sure to bring it down.`;
  } else {
    const head = `Update ${s.step} → ${s.step + 1}: w ← (${fmt(cur.w[0], 2)}, ${fmt(cur.w[1], 2)}) − ${fmt(lr, 2)}·(${fmt(cur.gw[0], 3)}, ${fmt(cur.gw[1], 3)}) = (${fmt(nextW[0], 2)}, ${fmt(nextW[1], 2)}). `;
    if (separable && lambda === 0) {
      const half = hist[Math.max(1, Math.floor(s.step / 2))];
      title = `Step ${s.step} — separable data: ‖w‖ keeps growing`;
      text =
        head +
        (cur.accuracy === 1
          ? `Every training point is already on the right side, yet the gradient is not zero. BCE still rewards pushing p closer to 0 or 1, and only a bigger ‖w‖ can do that. ‖w‖ = ${fmt(wNorm, 2)} now, versus ${fmt(Math.hypot(...half.w), 2)} at step ${half.t}. The dashed p = 0.25/0.75 lines squeeze toward the boundary. No finite best w exists. Try λ > 0.`
          : `${cur.wrong} point(s) are still on the wrong side. A straight line can split this data, so all of them will get fixed. After that, ‖w‖ will keep growing forever.`);
    } else if (separable) {
      title = `Step ${s.step} — separable data, L2 keeps w finite`;
      text = head + `The L2 penalty (λ/2)‖w‖² adds λw to the gradient. At some point the data's pull toward a bigger w is balanced by λw. So ‖w‖ settles near ${fmt(Math.hypot(...last.w), 2)} instead of growing forever, and the probability band stays wide.`;
    } else if (preset === "imbalanced" && pos <= 0.2 * pts.length) {
      title = `Step ${s.step} — imbalanced classes`;
      text = head + `Accuracy is ${fmt(cur.accuracy * 100, 1)}%, but always guessing the bigger class already scores ${fmt(baseline * 100, 1)}%. Recall on the smaller class (how many of its points we catch) is ${tp}/${pos}. The bias soaks up the class ratio, ln(${pos}/${neg}) = ${fmt(Math.log(pos / neg), 2)}, and pushes the boundary into the smaller class. Report scores per class, or give the classes different weights.`;
    } else {
      const done = gNorm < 1e-3;
      title = done ? `Step ${s.step} — settled (no line splits it)` : `Step ${s.step} — no straight line splits this`;
      text =
        head +
        `No straight line gets every point right, so BCE cannot reach 0. ${done ? "The gradient has dropped to almost zero (‖∇‖ = " + fmt(gNorm, 4) + ") at finite weights" : "The loss heads toward a minimum above zero"}. Loss ${fmt(cur.bce, 4)}, ${cur.wrong} point(s) wrong.` +
        (preset === "circles"
          ? ` For the circles, accuracy (${fmt(cur.accuracy * 100, 1)}%) can even drop below always guessing the bigger class (${fmt(baseline * 100, 1)}%). Lower BCE does not always mean higher accuracy. The fix is a curved feature like x₁² + x₂², or a hidden layer.`
          : "");
    }
  }

  const presetNote = PRESETS.find((p) => p.id === preset)!.note;
  const added = pts.length - nPreset;

  return (
    <SimShell
      title="Logistic Regression Trainer"
      subtitle="Gradient descent on all points at once, using binary cross-entropy (BCE) loss for p = σ(w₁x₁ + w₂x₂ + b). Click the plot to add points."
      controls={
        <Controls
          s={s}
          extra={
            <select className="field text-xs" value={preset} onChange={(e) => loadPreset(e.target.value as PresetId)} aria-label="Dataset preset">
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <Slider label="learning rate η" value={logLr} min={-2} max={1} step={0.05} onChange={setLogLr} fmt={() => fmt(lr, lr < 0.1 ? 3 : 2)} />
          <Slider label="L2 strength λ" value={lambda} min={0} max={0.2} step={0.005} onChange={setLambda} fmt={(v) => v.toFixed(3)} />
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted">click adds</span>
            <button className={`sim-btn ${cls === 1 ? "primary" : ""}`} onClick={() => setCls(1)} aria-pressed={cls === 1} aria-label="Add class 1 points">
              ● class 1
            </button>
            <button className={`sim-btn ${cls === 0 ? "primary" : ""}`} onClick={() => setCls(0)} aria-pressed={cls === 0} aria-label="Add class 0 points">
              ■ class 0
            </button>
            <button className="sim-btn" onClick={() => setPts((p) => p.slice(0, -1))} disabled={added <= 0} aria-label="Remove the last added point">
              <Undo2 size={14} /> Undo
            </button>
          </div>
          <div className="rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">
            <div>
              w = ({fmt(cur.w[0], 3)}, {fmt(cur.w[1], 3)}) b = {fmt(cur.b, 3)}
            </div>
            <div className="text-clay">
              ∇w = ({fmt(cur.gw[0], 4)}, {fmt(cur.gw[1], 4)}) ∂b = {fmt(cur.gb, 4)}
            </div>
            <div className="text-gold">
              next w = ({fmt(nextW[0], 3)}, {fmt(nextW[1], 3)}) b = {fmt(nextB, 3)}
            </div>
            <div>
              BCE {fmt(cur.bce, 4)}
              {lambda > 0 ? ` · J ${fmt(cur.objective, 4)}` : ""} · acc {fmt(cur.accuracy * 100, 1)}% · ‖w‖ {fmt(wNorm, 2)}
            </div>
            <div className="text-muted">
              n = {pts.length} ({pos} class 1, {neg} class 0) · majority baseline {fmt(baseline * 100, 1)}%
            </div>
          </div>
          <Narration title={title}>{text}</Narration>
        </div>
      }
    >
      <p className="mb-2 text-xs text-muted">{presetNote}</p>
      <div className="grid gap-4 md:grid-cols-[1.15fr_1fr]">
        <figure>
          <svg viewBox={`0 0 ${S} ${S}`} className="h-auto w-full cursor-crosshair touch-manipulation" role="img" aria-label={`Data, decision boundary and predicted probability at step ${s.step}`} onClick={addPoint}>
            {cells.map((c, k) => {
              const a = Math.abs(c.p - 0.5) * 2; // 0 at the boundary, 1 when certain
              const pct = Math.round(a * 38);
              return (
                <rect
                  key={k}
                  x={toPx(c.x) - cell / 2}
                  y={toPy(c.y) - cell / 2}
                  width={cell + 0.3}
                  height={cell + 0.3}
                  fill={`color-mix(in srgb, ${c.p >= 0.5 ? "var(--accent)" : "var(--clay)"} ${pct}%, transparent)`}
                />
              );
            })}
            <rect x={PAD} y={PAD} width={S - 2 * PAD} height={S - 2 * PAD} fill="none" stroke="var(--border)" />
            {[-2, 0, 2].map((v) => (
              <g key={v}>
                <text x={toPx(v)} y={S - 6} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
                  {v}
                </text>
                <text x={PAD - 5} y={toPy(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                  {v}
                </text>
              </g>
            ))}
            <text x={S - PAD} y={S - 6} fontSize="9" textAnchor="end" fill="var(--text-muted)">
              x₁
            </text>
            <text x={PAD - 5} y={PAD + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
              x₂
            </text>
            {band25 && <line x1={toPx(band25[0][0])} y1={toPy(band25[0][1])} x2={toPx(band25[1][0])} y2={toPy(band25[1][1])} stroke="var(--text-muted)" strokeDasharray="4 3" strokeWidth={1} />}
            {band75 && <line x1={toPx(band75[0][0])} y1={toPy(band75[0][1])} x2={toPx(band75[1][0])} y2={toPy(band75[1][1])} stroke="var(--text-muted)" strokeDasharray="4 3" strokeWidth={1} />}
            {boundary && <line x1={toPx(boundary[0][0])} y1={toPy(boundary[0][1])} x2={toPx(boundary[1][0])} y2={toPy(boundary[1][1])} stroke="var(--text)" strokeWidth={2} />}
            {arrow && (
              <g>
                <line x1={toPx(arrow.x0)} y1={toPy(arrow.y0)} x2={toPx(arrow.x1)} y2={toPy(arrow.y1)} stroke="var(--gold)" strokeWidth={2} />
                <circle cx={toPx(arrow.x1)} cy={toPy(arrow.y1)} r={3} fill="var(--gold)" />
                <text x={toPx(arrow.x1) + 5} y={toPy(arrow.y1) - 4} fontSize="9" fill="var(--gold)">
                  w
                </text>
              </g>
            )}
            {pts.map((p, i) => {
              const pr = sigmoid(cur.w[0] * p.x1 + cur.w[1] * p.x2 + cur.b) > 0.5 ? 1 : 0;
              const wrong = pr !== p.y;
              const X = toPx(p.x1), Y = toPy(p.x2);
              return (
                <g key={i}>
                  {wrong && <circle cx={X} cy={Y} r={7} fill="none" stroke="var(--gold)" strokeWidth={1.6} />}
                  {p.y === 1 ? (
                    <circle cx={X} cy={Y} r={3.8} fill="var(--accent)" stroke="var(--surface)" strokeWidth={0.8} />
                  ) : (
                    <rect x={X - 3.4} y={Y - 3.4} width={6.8} height={6.8} fill="var(--clay)" stroke="var(--surface)" strokeWidth={0.8} />
                  )}
                </g>
              );
            })}
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
            <span className="text-accent">● class 1</span>
            <span className="text-clay">■ class 0</span>
            <span>— boundary p = 0.5</span>
            <span>- - p = 0.25 / 0.75</span>
            <span className="text-gold">○ wrong · → w</span>
          </figcaption>
        </figure>

        <figure>
          <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label="Loss and accuracy history">
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line x1={CP.l} x2={CW - CP.r} y1={cy(v)} y2={cy(v)} stroke="var(--border)" strokeWidth={0.6} />
                <text x={CP.l - 4} y={cy(v) + 3} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
                  {v}
                </text>
              </g>
            ))}
            {[0, 100, 200, 300, 400].map((t) => (
              <text key={t} x={cx(t)} y={CH - 5} fontSize="8.5" textAnchor="middle" fill="var(--text-muted)">
                {t}
              </text>
            ))}
            <line x1={cx(0)} x2={CW - CP.r} y1={cy(Math.LN2)} y2={cy(Math.LN2)} stroke="var(--text-muted)" strokeDasharray="2 3" strokeWidth={0.8} />
            <path d={path((i) => hist[i].accuracy)} fill="none" stroke="var(--accent)" strokeWidth={1.8} />
            <path d={path((i) => hist[i].objective)} fill="none" stroke="var(--clay)" strokeWidth={1.8} />
            <line x1={cx(s.step)} x2={cx(s.step)} y1={CP.t} y2={CH - CP.b} stroke="var(--gold)" strokeWidth={1} />
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
            <span className="text-clay">— loss {lambda > 0 ? "(BCE + L2)" : "(BCE)"}</span>
            <span className="text-accent">— accuracy</span>
            <span>··· ln 2 (w = 0)</span>
            <span>x: step</span>
          </figcaption>
          <div className="mt-3 rounded-md border border-border p-2 text-xs text-muted">
            <div className="mb-1 font-medium text-[var(--text)]">The update rule we apply</div>
            <div className="font-mono text-[11px]">∇w = (1/N) Σ (pᵢ − yᵢ) xᵢ + λw</div>
            <div className="font-mono text-[11px]">∂b = (1/N) Σ (pᵢ − yᵢ)</div>
            <div className="font-mono text-[11px]">w ← w − η ∇w, b ← b − η ∂b</div>
            <div className="mt-1">
              Points whose p is already close to their label barely count. Points that are confidently wrong pull hardest. Step {s.step} of {STEPS}; ‖∇‖ = {fmt(gNorm, 4)}.
            </div>
          </div>
        </figure>
      </div>
    </SimShell>
  );
}
