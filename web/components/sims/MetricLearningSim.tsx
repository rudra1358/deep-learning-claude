"use client";

import { useId, useMemo, useState, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { K, embed, makeDataset, train, type Mining, type Mode, type Vec2 } from "@/lib/sim/metricLearning";

const STEPS = 300; // SGD steps precomputed per configuration
const BATCH = 16; // anchors per mini-batch
const S = 320; // embedding plot size (viewBox units)
const PAD = 16;
const CW = 300;
const CH = 104;
const CP = { l: 30, r: 8, t: 8, b: 18 };

/** Calm class palette derived from the theme variables (works in light and dark mode); shapes add a second cue. */
const CLASS_COLORS = [
  "var(--accent)",
  "var(--clay)",
  "var(--gold)",
  "var(--text-muted)",
  "var(--text)",
];
const CLASS_SHAPES = ["●", "■", "▲", "◆", "✚"];

function Marker({ c, x, y, r, opacity = 1 }: { c: number; x: number; y: number; r: number; opacity?: number }) {
  const fill = CLASS_COLORS[c];
  const common = { fill, opacity, stroke: "var(--surface)", strokeWidth: 0.7 };
  switch (c) {
    case 0:
      return <circle cx={x} cy={y} r={r} {...common} />;
    case 1:
      return <rect x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} {...common} />;
    case 2:
      return <polygon points={`${x},${y - r * 1.15} ${x + r * 1.05},${y + r * 0.8} ${x - r * 1.05},${y + r * 0.8}`} {...common} />;
    case 3:
      return <polygon points={`${x},${y - r * 1.2} ${x + r * 1.05},${y} ${x},${y + r * 1.2} ${x - r * 1.05},${y}`} {...common} />;
    default: {
      const w = r * 0.55;
      return <path d={`M${x - w},${y - r * 1.1}h${2 * w}v${r * 1.1 - w}h${r * 1.1 - w}v${2 * w}h${-(r * 1.1 - w)}v${r * 1.1 - w}h${-2 * w}v${-(r * 1.1 - w)}h${-(r * 1.1 - w)}v${-2 * w}h${r * 1.1 - w}z`} {...common} />;
    }
  }
}

function movingAverage(v: number[], w: number) {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i];
    if (i >= w) sum -= v[i - w];
    out.push(sum / Math.min(i + 1, w));
  }
  return out;
}

export default function MetricLearningSim() {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [mode, setMode] = useState<Mode>("triplet");
  const [mining, setMining] = useState<Mining>("semi-hard");
  const [margin, setMargin] = useState(1);
  const [logLr, setLogLr] = useState(Math.log10(0.05));
  const [seed, setSeed] = useState(7);
  const lr = 10 ** logLr;

  const s = useStepper(STEPS + 1, { baseMs: 220 });
  const data = useMemo(() => makeDataset(seed), [seed]);
  const hist = useMemo(() => train(data, { mode, mining, margin, lr, steps: STEPS, batch: BATCH, seed }), [data, mode, mining, margin, lr, seed]);
  const cur = hist[s.step];
  const E = useMemo(() => embed(cur.W, data.Xtr), [cur, data]);
  const t = cur.shown;

  /* ---------- view transform: centre on the embedding mean, scale to the farthest point ---------- */
  let mx = 0, my = 0;
  for (const e of E) {
    mx += e[0];
    my += e[1];
  }
  mx /= E.length;
  my /= E.length;
  let R = 1e-6;
  for (const e of E) R = Math.max(R, Math.hypot(e[0] - mx, e[1] - my));
  const k = (S / 2 - PAD) / R;
  const px = (e: Vec2) => S / 2 + (e[0] - mx) * k;
  const py = (e: Vec2) => S / 2 - (e[1] - my) * k;

  const A = E[t.a], P = E[t.p], Nn = E[t.n];
  const circleR = (mode === "triplet" ? t.dap + margin : margin) * k;

  /* gradient arrows: −∂ℓ/∂e at a, p, n, lengths relative to the largest of the three */
  const gmax = Math.max(1e-12, Math.hypot(...t.ga), Math.hypot(...t.gp), Math.hypot(...t.gn));
  const arrow = (from: Vec2, g: Vec2, color: string, key: string): ReactNode => {
    const len = Math.hypot(...g);
    if (len < 1e-9) return null;
    const L = (30 * len) / gmax;
    const x0 = px(from), y0 = py(from);
    return <line key={key} x1={x0} y1={y0} x2={x0 + (g[0] / len) * L} y2={y0 - (g[1] / len) * L} stroke={color} strokeWidth={1.8} markerEnd={`url(#ah-${uid})`} />;
  };

  const showP = mode === "triplet" || t.pair === "positive";
  const showN = mode === "triplet" || t.pair === "negative";

  /* ---------- history charts ---------- */
  const losses = useMemo(() => hist.map((h) => h.loss), [hist]);
  const smooth = useMemo(() => movingAverage(losses, 15), [losses]);
  const lossMax = Math.max(1e-6, ...losses.slice(0, Math.max(1, s.step + 1)));
  const cx = (i: number) => CP.l + (i / STEPS) * (CW - CP.l - CP.r);
  const cyL = (v: number) => CP.t + (1 - Math.min(1, v / lossMax)) * (CH - CP.t - CP.b);
  const cyA = (v: number) => CP.t + (1 - v) * (CH - CP.t - CP.b);
  const path = (f: (i: number) => number, y: (v: number) => number) => {
    let d = "";
    for (let i = 0; i <= s.step; i++) d += `${i ? "L" : "M"}${cx(i).toFixed(1)},${y(f(i)).toFixed(1)} `;
    return d;
  };

  /* ---------- narration ---------- */
  const ca = data.ytr[t.a], cn = data.ytr[t.n];
  const miningText =
    mining === "random"
      ? "The negative was picked at random from the other classes."
      : mining === "hard"
        ? "Hard mining: the negative is the point from another class that sits closest to the anchor right now."
        : mode === "triplet"
          ? `Semi-hard mining: the negative is the closest other-class point that is farther than the positive but still inside the margin (d(a,p) < d(a,n) < d(a,p) + m)${t.dan > t.dap && t.dan < t.dap + margin ? "" : ". This anchor had none, so we used a random negative"}.`
          : "Semi-hard mining (contrastive): the negative is picked at random from those still inside the margin. Only those give a gradient.";
  let title: string;
  let story: ReactNode;
  if (mode === "triplet") {
    title = t.satisfied ? `Step ${s.step}: margin met, nothing to learn` : `Step ${s.step}: margin broken, so this triplet learns`;
    story = (
      <>
        <p>
          Anchor #{t.a} (class {CLASS_SHAPES[ca]}), positive #{t.p} (same class), negative #{t.n} (class {CLASS_SHAPES[cn]}). d(a,p) = <strong className="text-text">{fmt(t.dap, 3)}</strong>, d(a,n) ={" "}
          <strong className="text-text">{fmt(t.dan, 3)}</strong>, margin m = {fmt(margin, 2)}.
        </p>
        <p className="mt-1">
          ℓ = max(0, {fmt(t.dap, 3)} − {fmt(t.dan, 3)} + {fmt(margin, 2)}) = <strong className="text-text">{fmt(t.loss, 3)}</strong>.{" "}
          {t.satisfied
            ? "The negative is already outside the dashed circle (radius d(a,p) + m). So this triplet adds nothing. Easy triplets like this waste compute."
            : "The negative is inside the dashed circle (radius d(a,p) + m). The gradient pulls the anchor and positive together and pushes the negative away. Arrows show −∂ℓ/∂e: the way each point would move."}
        </p>
        <p className="mt-1">{miningText}</p>
        <p className="mt-1">
          In this batch, {cur.active}/{BATCH} triplets were active (still had loss to learn from).
          {mining === "random" && cur.active < BATCH / 4 ? " With random negatives, most triplets soon get easy and learning slows down. Try semi-hard mining." : ""}
          {mining === "hard" && s.step > 0 ? " The hardest negatives are often odd outliers or near-copies. The FaceNet paper found that always using them early in training can make the model collapse." : ""}
        </p>
      </>
    );
  } else {
    const pos = t.pair === "positive";
    const Dd = pos ? t.dap : t.dan;
    title = pos ? `Step ${s.step}: similar pair, pulled together` : t.satisfied ? `Step ${s.step}: different pair, outside the margin` : `Step ${s.step}: different pair, inside the margin`;
    story = (
      <>
        <p>
          Anchor #{t.a} (class {CLASS_SHAPES[ca]}) is paired with #{pos ? t.p : t.n} ({pos ? "same class, Y = 0" : `class ${CLASS_SHAPES[cn]}, Y = 1`}). Distance D = <strong className="text-text">{fmt(Dd, 3)}</strong>, margin m = {fmt(margin, 2)}.
        </p>
        <p className="mt-1">
          {pos ? (
            <>
              ℓ = ½D² = <strong className="text-text">{fmt(t.loss, 3)}</strong>. A similar pair is always pulled together. The farther apart, the harder the pull (gradient e_a − e_p).
            </>
          ) : (
            <>
              ℓ = ½·max(0, m − D)² = <strong className="text-text">{fmt(t.loss, 3)}</strong>.{" "}
              {t.satisfied ? "The pair is already farther apart than m (outside the dashed circle). No loss, no gradient." : "The pair is closer than m, so it gets pushed apart. The push is m − D, which fades to zero at the margin."}
            </>
          )}
        </p>
        <p className="mt-1">{pos ? "Each pair is positive or negative with a 50/50 chance. This one is positive, so the mining rule did not matter." : miningText}</p>
        <p className="mt-1">
          {cur.active}/{BATCH} pairs in this batch produced a gradient.
        </p>
      </>
    );
  }
  if (s.step === 0)
    story = (
      <>
        <p>
          Step 0: W is a random 2×8 matrix. Five classes live in 8-D, but only the first 4 coordinates tell the classes apart. Coordinates 5–8 are big noise. A random squash to 2-D mixes both kinds, so the classes overlap. 1-NN accuracy (label each point like its nearest neighbour) is {fmt(cur.knnTest * 100, 0)}%.
        </p>
        {story}
      </>
    );

  return (
    <SimShell
      title="Contrastive & Triplet Embeddings"
      subtitle="Learn a map e = Wx from 8-D down to 2-D so same-class points end up close. Trained live in your browser with mini-batch SGD (16 anchors per step)."
      controls={
        <div className="space-y-2.5">
          <Controls
            s={s}
            extra={
              <button className="sim-btn" onClick={() => { setSeed((v) => v + 1); s.reset(); }} aria-label="Draw a new dataset and initialisation">
                <Shuffle size={14} /> New data
              </button>
            }
          />
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted">loss</span>
            {(["contrastive", "triplet"] as const).map((v) => (
              <button key={v} className={`sim-btn ${mode === v ? "primary" : ""}`} onClick={() => setMode(v)} aria-pressed={mode === v}>
                {v}
              </button>
            ))}
            <span className="ml-2 text-xs text-muted">negatives</span>
            {(["random", "semi-hard", "hard"] as const).map((v) => (
              <button key={v} className={`sim-btn ${mining === v ? "primary" : ""}`} onClick={() => setMining(v)} aria-pressed={mining === v}>
                {v}
              </button>
            ))}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="margin m" value={margin} min={0.2} max={2} step={0.05} onChange={setMargin} />
            <Slider label="learning rate" value={logLr} min={-2} max={-0.7} step={0.05} onChange={setLogLr} fmt={() => fmt(lr, 3)} />
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div className="rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">
            <div>
              step {s.step} · batch loss {fmt(cur.loss, 4)}
            </div>
            <div>
              active {mode === "triplet" ? "triplets" : "pairs"} {cur.active}/{BATCH} · ‖∇W‖ {fmt(cur.gradNorm, 4)}
            </div>
            <div className="text-accent">
              1-NN accuracy: test {fmt(cur.knnTest * 100, 1)}% · train (LOO) {fmt(cur.knnTrain * 100, 1)}%
            </div>
            <div className="text-muted">
              shown: d(a,p) {fmt(t.dap, 3)} · d(a,n) {fmt(t.dan, 3)} · ℓ {fmt(t.loss, 3)} · {t.satisfied ? "margin met" : mode === "contrastive" && t.pair === "positive" ? "positive pair" : "margin broken"}
            </div>
          </div>
          <Narration title={title}>{story}</Narration>
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-[1.15fr_1fr]">
        <figure className="min-w-0">
          <svg viewBox={`0 0 ${S} ${S}`} className="h-auto w-full" role="img" aria-label={`Two-dimensional embedding of 150 training points at step ${s.step}, with the sampled ${mode === "triplet" ? "triplet" : "pair"} highlighted`}>
            <defs>
              <marker id={`ah-${uid}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--text)" />
              </marker>
            </defs>
            <rect x={1} y={1} width={S - 2} height={S - 2} fill="none" stroke="var(--border)" />
            {E.map((e, i) => (
              <Marker key={i} c={data.ytr[i]} x={px(e)} y={py(e)} r={3.6} opacity={i === t.a || i === t.p || i === t.n ? 1 : 0.6} />
            ))}
            <circle cx={px(A)} cy={py(A)} r={circleR} fill="none" stroke="var(--gold)" strokeDasharray="4 3" strokeWidth={1.2} />
            {showP && <line x1={px(A)} y1={py(A)} x2={px(P)} y2={py(P)} stroke="var(--accent)" strokeWidth={2.2} />}
            {showN && <line x1={px(A)} y1={py(A)} x2={px(Nn)} y2={py(Nn)} stroke="var(--clay)" strokeWidth={2.2} strokeDasharray="6 3" />}
            {arrow(A, t.ga, "var(--text)", "ga")}
            {showP && arrow(P, t.gp, "var(--text)", "gp")}
            {showN && arrow(Nn, t.gn, "var(--text)", "gn")}
            <circle cx={px(A)} cy={py(A)} r={7.5} fill="none" stroke="var(--text)" strokeWidth={1.6} />
            <text x={px(A) + 9} y={py(A) - 8} fontSize="11" fontWeight={600} fill="var(--text)">
              a
            </text>
            {showP && (
              <>
                <circle cx={px(P)} cy={py(P)} r={7} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
                <text x={px(P) + 9} y={py(P) - 8} fontSize="11" fontWeight={600} fill="var(--accent)">
                  p
                </text>
              </>
            )}
            {showN && (
              <>
                <circle cx={px(Nn)} cy={py(Nn)} r={7} fill="none" stroke="var(--clay)" strokeWidth={1.6} />
                <text x={px(Nn) + 9} y={py(Nn) - 8} fontSize="11" fontWeight={600} fill="var(--clay)">
                  n
                </text>
              </>
            )}
          </svg>
          <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
            {Array.from({ length: K }, (_, c) => (
              <span key={c} style={{ color: CLASS_COLORS[c] }}>
                {CLASS_SHAPES[c]} class {c + 1}
              </span>
            ))}
            <span className="text-accent">— a–p pulled</span>
            <span className="text-clay">- - a–n pushed</span>
            <span className="text-gold">◌ {mode === "triplet" ? "radius d(a,p) + m" : "radius m"}</span>
            <span>→ −∂ℓ/∂e</span>
          </figcaption>
        </figure>

        <div className="min-w-0 space-y-2">
          <figure>
            <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label="Mini-batch loss history">
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={CP.l} x2={CW - CP.r} y1={cyL(v * lossMax)} y2={cyL(v * lossMax)} stroke="var(--border)" strokeWidth={0.6} />
                  <text x={CP.l - 4} y={cyL(v * lossMax) + 3} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
                    {fmt(v * lossMax, 2)}
                  </text>
                </g>
              ))}
              {[0, 100, 200, 300].map((x) => (
                <text key={x} x={cx(x)} y={CH - 5} fontSize="8.5" textAnchor="middle" fill="var(--text-muted)">
                  {x}
                </text>
              ))}
              <path d={path((i) => losses[i], cyL)} fill="none" stroke="var(--clay)" strokeOpacity={0.35} strokeWidth={1} />
              <path d={path((i) => smooth[i], cyL)} fill="none" stroke="var(--clay)" strokeWidth={1.8} />
              <line x1={cx(s.step)} x2={cx(s.step)} y1={CP.t} y2={CH - CP.b} stroke="var(--gold)" strokeWidth={1} />
            </svg>
            <figcaption className="text-xs text-muted">
              <span className="text-clay">batch loss</span> (thin) and its 15-step average; x: SGD step
            </figcaption>
          </figure>
          <figure>
            <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label="Nearest-neighbour accuracy history">
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={CP.l} x2={CW - CP.r} y1={cyA(v)} y2={cyA(v)} stroke="var(--border)" strokeWidth={0.6} />
                  <text x={CP.l - 4} y={cyA(v) + 3} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
                    {v * 100}%
                  </text>
                </g>
              ))}
              <line x1={CP.l} x2={CW - CP.r} y1={cyA(1 / K)} y2={cyA(1 / K)} stroke="var(--text-muted)" strokeDasharray="2 3" strokeWidth={0.8} />
              <path d={path((i) => hist[i].knnTest, cyA)} fill="none" stroke="var(--accent)" strokeWidth={1.8} />
              <line x1={cx(s.step)} x2={cx(s.step)} y1={CP.t} y2={CH - CP.b} stroke="var(--gold)" strokeWidth={1} />
            </svg>
            <figcaption className="text-xs text-muted">
              <span className="text-accent">1-NN test accuracy</span> on 100 unseen points (dotted: random guessing, 20%)
            </figcaption>
          </figure>
          <div className="rounded-md border border-border p-2 text-xs text-muted">
            <div className="mb-1 font-medium text-[var(--text)]">{mode === "triplet" ? "Triplet loss we are lowering" : "Contrastive loss we are lowering"}</div>
            {mode === "triplet" ? (
              <div className="font-mono text-[11px]">ℓ = max(0, ‖e_a − e_p‖ − ‖e_a − e_n‖ + m)</div>
            ) : (
              <div className="font-mono text-[11px]">ℓ = (1 − Y)·½D² + Y·½·max(0, m − D)²</div>
            )}
            <div className="font-mono text-[11px]">∇W = Σ g (x_i − x_j)ᵀ, W ← W − η ∇W / 16</div>
            <div className="mt-1">The plot is re-centred and re-scaled every step. So watch how the classes are arranged, not their size.</div>
          </div>
        </div>
      </div>
    </SimShell>
  );
}
