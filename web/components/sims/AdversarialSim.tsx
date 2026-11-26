"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import {
  fgsm,
  linfDist,
  lossGrad,
  pgdTrajectory,
  robustAccuracy,
  testImage,
  verifyAdversarial,
  type AdversarialJSON,
  type ModelName,
  type PgdStep,
} from "@/lib/sim/adversarial";

const MODEL_URL = "/models/adversarial.json";
const TITLE = "Adversarial examples: FGSM and PGD (trained models)";
const SUB = "Nudge each pixel of a real digit by at most ε, in the direction that raises the loss. Watch the prediction flip.";
const MODEL_LABEL: Record<ModelName, string> = { linear: "Softmax regression", mlp: "MLP", mlp_adv: "MLP, adversarially trained" };
const MODEL_COLOR: Record<ModelName, string> = { linear: "var(--text-muted)", mlp: "var(--clay)", mlp_adv: "var(--accent)" };
const NAMES: ModelName[] = ["linear", "mlp", "mlp_adv"];
const CURVE_EPS = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3];

type Attack = "fgsm" | "pgd";

/* ------------------------------------------------------------------ small views */

function Digit({ px, label, size = 92, mode = "ink", maxAbs = 1, highlight = false }: { px: number[]; label: string; size?: number; mode?: "ink" | "signed"; maxAbs?: number; highlight?: boolean }) {
  return (
    <figure className="flex min-w-0 flex-col items-center">
      <svg viewBox="0 0 8 8" width={size} height={size} role="img" aria-label={label} className="rounded-sm border" style={{ borderColor: highlight ? "var(--gold)" : "var(--border)", maxWidth: "100%", height: "auto" }}>
        <rect width={8} height={8} fill="var(--surface)" />
        {px.map((v, i) =>
          mode === "signed" ? (
            v === 0 ? null : <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill={signedColor(v, maxAbs)} />
          ) : (
            <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill="var(--text)" fillOpacity={Math.max(0, Math.min(1, v))} />
          ),
        )}
      </svg>
      <figcaption className="mt-1 max-w-[7.5rem] text-center text-[11px] leading-tight text-muted">{label}</figcaption>
    </figure>
  );
}

/** Ten classes, two bars each: clean prediction (muted) and current/attacked prediction. */
function ProbBars({ clean, cur, label }: { clean: number[]; cur: number[]; label: number }) {
  const W = 320;
  const rowH = 17;
  const L = 22;
  const BW = W - L - 44;
  const predC = clean.indexOf(Math.max(...clean));
  const predA = cur.indexOf(Math.max(...cur));
  return (
    <svg viewBox={`0 0 ${W} ${10 * rowH + 16}`} className="h-auto w-full" role="img" aria-label={`Class probabilities. Clean prediction ${predC}, current prediction ${predA}, true label ${label}`}>
      <text x={L} y={10} fontSize={9} fill="var(--text-faint)">thin grey = clean x · thick = x_t (accent = true class, clay = wrong)</text>
      {clean.map((pc, k) => {
        const y = 16 + k * rowH;
        const pa = cur[k];
        return (
          <g key={k}>
            <text x={L - 6} y={y + 11} fontSize={11} textAnchor="end" fill={k === label ? "var(--accent)" : "var(--text-muted)"} fontFamily="var(--font-mono)" fontWeight={k === label ? 700 : 400}>
              {k}
            </text>
            <rect x={L} y={y + 1} width={BW} height={rowH - 3} fill="var(--surface-2)" />
            <rect x={L} y={y + 1} width={Math.max(0.5, pc * BW)} height={5} fill="var(--text-faint)" />
            <rect x={L} y={y + 7} width={Math.max(0.5, pa * BW)} height={7} fill={k === label ? "var(--accent)" : "var(--clay)"} stroke={k === predA ? "var(--gold)" : "none"} strokeWidth={1.2} />
            <text x={L + BW + 4} y={y + 12} fontSize={10} fill={k === predA ? "var(--text)" : "var(--text-faint)"} fontFamily="var(--font-mono)">
              {(pa * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Loss of x_t against the PGD iteration t (current t marked). */
function LossPlot({ traj, t }: { traj: PgdStep[]; t: number }) {
  const W = 320;
  const H = 120;
  const P = { l: 30, r: 8, t: 8, b: 18 };
  const losses = traj.map((s) => s.loss);
  const maxL = Math.max(1, ...losses) * 1.05;
  const n = Math.max(1, traj.length - 1);
  const sx = (i: number) => P.l + (i / n) * (W - P.l - P.r);
  const sy = (v: number) => H - P.b - (v / maxL) * (H - P.t - P.b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Cross-entropy loss over ${traj.length - 1} attack iterations`}>
      <rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} fill="var(--surface)" stroke="var(--border)" />
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={P.l - 4} y={sy(f * maxL) + 3} fontSize={9} textAnchor="end" fill="var(--text-faint)">
          {(f * maxL).toFixed(1)}
        </text>
      ))}
      <line x1={P.l} x2={W - P.r} y1={sy(Math.log(2))} y2={sy(Math.log(2))} stroke="var(--border-strong)" strokeDasharray="3 3" />
      <polyline fill="none" stroke="var(--clay)" strokeWidth={1.8} points={losses.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")} />
      {losses.map((v, i) => (
        <circle key={i} cx={sx(i)} cy={sy(v)} r={i === t ? 4 : 1.8} fill={i === t ? "var(--gold)" : "var(--clay)"} />
      ))}
      <text x={W - P.r} y={H - 4} fontSize={9} textAnchor="end" fill="var(--text-muted)">iteration t →</text>
      <text x={P.l + 4} y={P.t + 10} fontSize={9} fill="var(--text-muted)">loss = −log p(y | x_t)</text>
    </svg>
  );
}

/** Test accuracy against ε for the three models (current model emphasised). */
function RobustPlot({ curves, model, eps, attack }: { curves: Record<ModelName, number[]>; model: ModelName; eps: number; attack: Attack }) {
  const W = 340;
  const H = 170;
  const P = { l: 32, r: 10, t: 10, b: 24 };
  const sx = (e: number) => P.l + (e / 0.3) * (W - P.l - P.r);
  const sy = (a: number) => H - P.b - a * (H - P.t - P.b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Test accuracy under ${attack.toUpperCase()} as a function of epsilon for three models`}>
      <rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} fill="var(--surface)" stroke="var(--border)" />
      {[0, 0.25, 0.5, 0.75, 1].map((a) => (
        <g key={a}>
          <line x1={P.l} x2={W - P.r} y1={sy(a)} y2={sy(a)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={P.l - 4} y={sy(a) + 3} fontSize={9} textAnchor="end" fill="var(--text-faint)">{Math.round(a * 100)}%</text>
        </g>
      ))}
      {[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3].map((e) => (
        <text key={e} x={sx(e)} y={H - P.b + 12} fontSize={9} textAnchor="middle" fill="var(--text-faint)">{e}</text>
      ))}
      <text x={W - P.r} y={H - 2} fontSize={9} textAnchor="end" fill="var(--text-muted)">ε (max change per pixel, pixels in [0, 1])</text>
      <line x1={sx(Math.min(eps, 0.3))} x2={sx(Math.min(eps, 0.3))} y1={P.t} y2={H - P.b} stroke="var(--gold)" strokeWidth={1.5} />
      {NAMES.map((n) => (
        <polyline key={n} fill="none" stroke={MODEL_COLOR[n]} strokeWidth={n === model ? 2.6 : 1.2} strokeDasharray={n === "linear" ? "4 3" : undefined} points={curves[n].map((a, i) => `${sx(CURVE_EPS[i])},${sy(a)}`).join(" ")} />
      ))}
      {NAMES.map((n, k) => (
        <g key={n} transform={`translate(${P.l + 8},${P.t + 12 + k * 12})`}>
          <line x1={0} x2={14} y1={-3} y2={-3} stroke={MODEL_COLOR[n]} strokeWidth={n === model ? 2.6 : 1.2} strokeDasharray={n === "linear" ? "4 3" : undefined} />
          <text x={18} y={0} fontSize={9} fill="var(--text-muted)">{MODEL_LABEL[n]}</text>
        </g>
      ))}
    </svg>
  );
}

/* ------------------------------------------------------------------ main */

export default function AdversarialSim() {
  const [json, setJson] = useState<AdversarialJSON | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<{ maxAbsDiff: number; checked: number } | null>(null);
  const [model, setModel] = useState<ModelName>("mlp");
  const [idx, setIdx] = useState(0);
  const [attack, setAttack] = useState<Attack>("pgd");
  const [eps, setEps] = useState(0.1);
  const [steps, setSteps] = useState(10);
  const [alpha, setAlpha] = useState(0.02);

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<AdversarialJSON>;
      })
      .then((j) => {
        if (!alive) return;
        setJson(j);
        setCheck(verifyAdversarial(j));
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const total = attack === "fgsm" ? 2 : steps + 1;
  const s = useStepper(total, { baseMs: 700 });
  const t = Math.min(s.step, total - 1);

  const layers = json ? json.models[model].layers : null;
  const x0 = useMemo(() => (json ? testImage(json, idx) : []), [json, idx]);
  const label = json ? json.test.labels[idx] : 0;

  // trajectory of iterates: FGSM is a single step of size ε, PGD is `steps` steps of size α
  const traj = useMemo<PgdStep[]>(() => {
    if (!layers) return [];
    if (attack === "pgd") return pgdTrajectory(layers, x0, label, eps, alpha, steps);
    const first = lossGrad(layers, x0, label);
    const x1 = fgsm(layers, x0, label, eps);
    const second = lossGrad(layers, x1, label);
    return [
      { x: x0, loss: first.loss, probs: first.probs, grad: first.grad },
      { x: x1, loss: second.loss, probs: second.probs, grad: second.grad },
    ];
  }, [layers, x0, label, attack, eps, alpha, steps]);

  // accuracy-vs-ε curves over the whole test set, computed in the browser
  const curves = useMemo<Record<ModelName, number[]> | null>(() => {
    if (!json) return null;
    const imgs = json.test.images.map((_, i) => testImage(json, i));
    const out = {} as Record<ModelName, number[]>;
    for (const n of NAMES) out[n] = CURVE_EPS.map((e) => robustAccuracy(json.models[n].layers, imgs, json.test.labels, e, attack));
    return out;
  }, [json, attack]);

  if (error) {
    return (
      <SimShell title={TITLE} subtitle={SUB}>
        <p className="text-sm text-clay">Could not load the models ({error}). Check your internet connection and reload the page.</p>
      </SimShell>
    );
  }
  if (!json || !layers || traj.length === 0 || !curves) {
    return (
      <SimShell title={TITLE} subtitle={SUB}>
        <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
          Loading three trained digit models (about 130 KB)…
        </div>
      </SimShell>
    );
  }

  const nTest = json.test.labels.length;
  const cur = traj[t];
  const clean = traj[0];
  const delta = cur.x.map((v, i) => v - x0[i]);
  const gradSign = cur.grad.map((g) => Math.sign(g));
  const predClean = clean.probs.indexOf(Math.max(...clean.probs));
  const predCur = cur.probs.indexOf(Math.max(...cur.probs));
  const flipAt = traj.findIndex((st) => st.probs.indexOf(Math.max(...st.probs)) !== label);
  const curveAt = (n: ModelName) => {
    const k = CURVE_EPS.findIndex((e) => Math.abs(e - eps) < 1e-9);
    return k >= 0 ? `${(curves[n][k] * 100).toFixed(1)}%` : "—";
  };
  const info = json.models[model];
  const atEdge = eps > 0 ? delta.filter((d) => Math.abs(d) >= eps - 1e-12).length : 0;
  const changed = delta.filter((d) => d !== 0).length;

  const controls = (
    <div className="flex flex-col gap-2.5">
      <Controls s={s} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Choose model">
          <span className="text-muted">Model</span>
          {NAMES.map((n) => (
            <button key={n} className={`sim-btn ${model === n ? "primary" : ""}`} onClick={() => setModel(n)} aria-pressed={model === n}>
              {n === "linear" ? "Linear" : n === "mlp" ? "MLP" : "MLP (adv. trained)"}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Choose attack">
          <span className="text-muted">Attack</span>
          {(["fgsm", "pgd"] as Attack[]).map((a) => (
            <button
              key={a}
              className={`sim-btn ${attack === a ? "primary" : ""}`}
              onClick={() => {
                setAttack(a);
                s.reset();
              }}
              aria-pressed={attack === a}
            >
              {a.toUpperCase()}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2">
          <span className="text-muted">Test digit #</span>
          <input type="number" className="field w-20 font-mono" min={0} max={nTest - 1} value={idx} onChange={(e) => setIdx(Math.max(0, Math.min(nTest - 1, Math.round(+e.target.value) || 0)))} aria-label="Test image index" />
        </label>
        <button className="sim-btn" onClick={() => setIdx((i) => (i * 37 + 101) % nTest)} aria-label="Pick another test digit">
          <Shuffle size={14} /> Another digit
        </button>
      </div>
      <div className="grid gap-1.5 sm:grid-cols-3">
        <Slider label="ε (budget)" value={eps} min={0} max={0.3} step={0.005} onChange={setEps} fmt={(v) => v.toFixed(3)} />
        {attack === "pgd" && (
          <>
            <Slider label="PGD steps" value={steps} min={1} max={40} step={1} onChange={setSteps} fmt={(v) => String(v)} />
            <Slider label="step size α" value={alpha} min={0.005} max={0.1} step={0.005} onChange={setAlpha} fmt={(v) => v.toFixed(3)} />
          </>
        )}
      </div>
    </div>
  );

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={attack === "fgsm" ? (t === 0 ? "The clean digit" : "FGSM: one step of size ε") : `PGD step ${t} of ${steps}`}>{narrate({ attack, t, steps, eps, alpha, label, predClean, predCur, cur, clean, flipAt, model })}</Narration>
      <div className="rounded-lg border border-border p-2.5 text-[11px] leading-snug text-muted">
        <span className="font-medium text-accent">These are real trained models.</span> We trained them in PyTorch on {json.data.n_train.toLocaleString()} of scikit-learn&apos;s 8×8 digits (pixels in [0, 1]): softmax regression ({json.models.linear.n_params} parameters, test accuracy {(json.models.linear.test_acc * 100).toFixed(1)}%), an MLP 64→32→10 ({json.models.mlp.n_params.toLocaleString()} parameters, {(json.models.mlp.test_acc * 100).toFixed(1)}%), and the same MLP trained on PGD attacks (ε = {json.adv_training.eps}, {json.adv_training.steps} steps; {(json.models.mlp_adv.test_acc * 100).toFixed(1)}% on clean digits). Your browser computes the gradients, attacks and curves below.
      </div>
      {check && (
        <div className="rounded-lg border border-border p-2.5 font-mono text-[11px] leading-snug text-muted">
          your browser vs NumPy answer: max |Δ| = {check.maxAbsDiff.toExponential(1)} over {check.checked} values {check.maxAbsDiff < 1e-6 ? "✓" : "✗"}
        </div>
      )}
    </div>
  );

  return (
    <SimShell title={TITLE} subtitle={SUB} controls={controls} aside={aside}>
      <div className="mb-3 grid grid-cols-2 justify-items-center gap-3 sm:grid-cols-4">
        <Digit px={x0} label={`clean x (true ${label}, predicted ${predClean})`} highlight={t === 0} />
        <Digit px={gradSign} mode="signed" label={`sign ∇ₓL at x_${t} (accent = +, clay = −)`} />
        <Digit px={delta} mode="signed" maxAbs={Math.max(eps, 1e-9)} label={`perturbation δ = x_${t} − x, ‖δ‖∞ = ${fmt(linfDist(cur.x, x0), 3)}`} />
        <Digit px={cur.x} label={`x_${t}: predicted ${predCur} (${(cur.probs[predCur] * 100).toFixed(1)}%)`} highlight={t > 0} />
      </div>

      <div className={`mb-3 rounded-lg border p-2.5 text-xs ${predCur !== label ? "border-clay text-clay" : "border-border text-muted"}`} role="status">
        {predCur !== label ? (
          <>
            <b>Fooled!</b> The model now says <b>{predCur}</b> with {(cur.probs[predCur] * 100).toFixed(1)}% confidence. Yet no pixel moved by more than {fmt(eps, 3)}.
          </>
        ) : (
          <>
            Still right: p(true class {label}) = {(cur.probs[label] * 100).toFixed(1)}% (was {(clean.probs[label] * 100).toFixed(1)}%). Loss went {fmt(clean.loss, 3)} → {fmt(cur.loss, 3)}.
          </>
        )}
        {t > 0 && (
          <span className="text-muted">
            {" "}
            · {changed} of 64 pixels changed. {atEdge} are at the full limit (|δ| = ε); the rest hit the [0, 1] pixel range{attack === "pgd" ? " or have not reached the limit yet" : ""}.
          </span>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium">Class probabilities</div>
          <ProbBars clean={clean.probs} cur={cur.probs} label={label} />
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium">Loss along the attack</div>
          <LossPlot traj={traj} t={t} />
          <p className="mt-1 text-[11px] text-muted">Dashed line: loss = ln 2, where p(true class) = 50%. Above it, the true class can no longer hold a majority.</p>
        </div>
      </div>

      <div className="mt-4 min-w-0">
        <div className="mb-1 text-xs font-medium">
          Robustness curve: accuracy on all {nTest} test digits under {attack === "fgsm" ? "FGSM" : "PGD (20 steps, α = 2.5ε/20)"} attack
        </div>
        <RobustPlot curves={curves} model={model} eps={eps} attack={attack} />
        <div className="mt-1 overflow-x-auto">
          <table className="w-full min-w-[17rem] text-left font-mono text-[11px] tabular-nums">
            <thead className="text-muted">
              <tr>
                <th className="pr-3 font-normal">model</th>
                <th className="pr-3 font-normal">clean</th>
                <th className="pr-3 font-normal">ε = 0.1</th>
                <th className="font-normal">ε = {eps.toFixed(3)}</th>
              </tr>
            </thead>
            <tbody>
              {NAMES.map((n) => (
                <tr key={n} className={n === model ? "text-accent" : ""}>
                  <td className="pr-3">{MODEL_LABEL[n]}</td>
                  <td className="pr-3">{(curves[n][0] * 100).toFixed(1)}%</td>
                  <td className="pr-3">{(curves[n][4] * 100).toFixed(1)}%</td>
                  <td>{curveAt(n)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[11px] text-muted">
            Curves are measured at ε ∈ {"{"}0, 0.025, …, 0.3{"}"}. The last column fills in when the slider is on one of those values. Reference numbers from the PyTorch/NumPy run, this model, ε = 0.1: FGSM {(info.curves.fgsm[4] * 100).toFixed(1)}%, PGD {(info.curves.pgd[4] * 100).toFixed(1)}%.
          </p>
        </div>
      </div>
    </SimShell>
  );
}

function narrate(a: {
  attack: Attack;
  t: number;
  steps: number;
  eps: number;
  alpha: number;
  label: number;
  predClean: number;
  predCur: number;
  cur: PgdStep;
  clean: PgdStep;
  flipAt: number;
  model: ModelName;
}): ReactNode {
  const { attack, t, steps, eps, alpha, label, predClean, predCur, cur, clean, flipAt, model } = a;
  if (t === 0) {
    return `A test ${label} the model never saw in training. The ${MODEL_LABEL[model].toLowerCase()} predicts ${predClean} with p = ${(clean.probs[predClean] * 100).toFixed(1)}%, loss ${fmt(clean.loss, 3)}. The second image is sign(∇ₓL), the gradient's sign. +1 (accent) means brightening that pixel raises the loss. −1 (clay) means darkening it does. ${attack === "fgsm" ? `FGSM will move every pixel by exactly ε = ${eps.toFixed(3)} that way, then clip to [0, 1].` : `PGD will take ${steps} small steps of size α = ${alpha.toFixed(3)} that way. After each one it recomputes the gradient and pulls x back inside the ε = ${eps.toFixed(3)} box.`}`;
  }
  const flip = predCur !== label ? ` The prediction is now ${predCur}. That is an adversarial example!` : ` The prediction is still ${label}.`;
  if (attack === "fgsm") {
    return `x' = clip(x + ε·sign(∇ₓL)). One gradient, one step: the loss rose from ${fmt(clean.loss, 3)} to ${fmt(cur.loss, 3)}.${flip} FGSM treats the loss as a straight line near x. For a linear model that gives the true worst case in the box (up to clipping). For a non-linear model it is only an estimate.`;
  }
  const atBoundary = alpha * t >= eps ? ` After ${Math.ceil(eps / alpha - 1e-9)} steps most pixels are at the ε limit, so later steps mostly shuffle the budget around.` : "";
  const flipNote = flipAt > 0 ? ` (The prediction first flipped at step ${flipAt}.)` : steps === t ? " PGD could not fool the model with this budget. Raise ε, the steps or α, or try the plain MLP." : "";
  return `x_${t} = Π(x_${t - 1} + α·sign(∇ₓL(x_${t - 1}))): take a step, then clip back into [x − ε, x + ε] and [0, 1]. Loss ${fmt(cur.loss, 3)}, p(true class) = ${(cur.probs[label] * 100).toFixed(1)}%.${flip}${atBoundary}${flipNote}`;
}
