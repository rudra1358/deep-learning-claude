"use client";

import { useEffect, useMemo, useState, type MouseEvent, type ReactNode } from "react";
import { Dices, Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, fmt, signedColor, useStepper } from "./SimShell";
import { decodeVae, forwardVae, loadVae, standardNormal, vaeTestImage, verifyVae, type Vae, type VaeJSON, type VaeTrace } from "@/lib/sim/vae";

const MODEL_URL = "/models/vae.json";
const TITLE = "Variational Autoencoder (trained model)";
const STAGES = ["Input x", "Encoder", "μ and σ: the posterior q(z|x)", "Sample ε ~ N(0, I)", "z = μ + σ ⊙ ε", "Decoder → x̂", "Loss = reconstruction + KL"] as const;
const SHORT = ["x", "enc", "μ, σ", "ε", "z", "dec", "loss"];

const CLASS_COLORS = [
  "var(--accent)",
  "var(--clay)",
  "var(--gold)",
  "var(--text)",
  "color-mix(in srgb, var(--accent) 50%, var(--clay))",
  "color-mix(in srgb, var(--accent) 45%, var(--gold))",
  "color-mix(in srgb, var(--clay) 50%, var(--gold))",
  "var(--text-muted)",
  "color-mix(in srgb, var(--accent) 50%, var(--text))",
  "color-mix(in srgb, var(--clay) 45%, var(--text))",
];

/* ------------------------------------------------------------------ small views */

function Digit({ px, label, size = 96, error = false, highlight = false }: { px: number[]; label?: string; size?: number; error?: boolean; highlight?: boolean }) {
  return (
    <figure className="flex min-w-0 flex-col items-center">
      <svg viewBox="0 0 8 8" width={size} height={size} role="img" aria-label={label ?? "digit image"} className="rounded-sm border" style={{ borderColor: highlight ? "var(--gold)" : "var(--border)", maxWidth: "100%", height: "auto" }}>
        <rect width={8} height={8} fill="var(--surface)" />
        {px.map((v, i) =>
          error ? (
            <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill={signedColor(v, 0.5)} />
          ) : (
            <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill="var(--text)" fillOpacity={Math.max(0, Math.min(1, v))} />
          ),
        )}
      </svg>
      {label && <figcaption className="mt-1 max-w-[8rem] text-center text-[11px] leading-tight text-muted">{label}</figcaption>}
    </figure>
  );
}

function VecCells({ v, cols = 16, label }: { v: number[]; cols?: number; label: string }) {
  const rows = Math.ceil(v.length / cols);
  const max = Math.max(1e-9, ...v.map(Math.abs));
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 text-xs">
        <span className="font-medium">{label}</span> <span className="font-mono text-faint">({v.length} values, {v.filter((x) => x === 0).length} exactly 0)</span>
      </figcaption>
      <svg viewBox={`0 0 ${cols} ${rows}`} className="h-auto w-full" style={{ maxWidth: cols * 22 }} role="img" aria-label={`${label}: ${v.length} activations`}>
        {v.map((x, i) => (
          <rect key={i} x={i % cols} y={Math.floor(i / cols)} width={0.92} height={0.92} rx={0.08} fill={x === 0 ? "var(--border)" : `color-mix(in srgb, var(--accent) ${Math.round(15 + 85 * (Math.abs(x) / max))}%, var(--surface))`} />
        ))}
      </svg>
    </figure>
  );
}

/* ------------------------------------------------------------------ latent plot */

const PW = 400;
const PH = 400;
const R = 3.4; // plotted range [-R, R]²

function LatentPlot({
  mus,
  labels,
  trace,
  showPosterior,
  showZ,
  cloud,
  path,
  clicked,
  grid,
  onClick,
}: {
  mus: number[][];
  labels: number[];
  trace: VaeTrace;
  showPosterior: boolean;
  showZ: boolean;
  cloud: number[][];
  path: number[][] | null;
  clicked: number[] | null;
  grid: number[][] | null;
  onClick: (z: number[]) => void;
}) {
  const sx = (x: number) => ((x + R) / (2 * R)) * PW;
  const sy = (y: number) => PH - ((y + R) / (2 * R)) * PH;
  const k = PW / (2 * R); // pixels per latent unit
  const handle = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * (PW + 24) - 20;
    const py = ((e.clientY - r.top) / r.height) * (PH + 20) - 4;
    onClick([-R + (px / PW) * 2 * R, -R + ((PH - py) / PH) * 2 * R]);
  };
  const n = grid ? Math.round(Math.sqrt(grid.length)) : 0;
  return (
    <svg viewBox={`-20 -4 ${PW + 24} ${PH + 20}`} className="h-auto w-full cursor-crosshair touch-manipulation" onClick={handle} role="img" aria-label="Latent space of the VAE: test digits at their posterior means, prior circles, and the current posterior ellipse. Click to decode a point.">
      <rect x={0} y={0} width={PW} height={PH} fill="var(--surface)" stroke="var(--border)" />
      {grid &&
        grid.map((px, i) => {
          const c = PW / n;
          const s = c * 0.92;
          return (
            <g key={i} transform={`translate(${(i % n) * c + (c - s) / 2},${Math.floor(i / n) * c + (c - s) / 2}) scale(${s / 8})`}>
              {px.map((v, j) => (v > 0.04 ? <rect key={j} x={j % 8} y={Math.floor(j / 8)} width={1} height={1} fill="var(--text)" fillOpacity={Math.min(1, v)} /> : null))}
            </g>
          );
        })}
      {[-3, -2, -1, 0, 1, 2, 3].map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={0} y2={PH} stroke="var(--border)" strokeWidth={t === 0 ? 1 : 0.5} />
          <line x1={0} x2={PW} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={t === 0 ? 1 : 0.5} />
          <text x={sx(t)} y={PH + 13} fontSize={10} textAnchor="middle" fill="var(--text-faint)">{t}</text>
          <text x={-4} y={sy(t) + 3} fontSize={10} textAnchor="end" fill="var(--text-faint)">{t}</text>
        </g>
      ))}
      {[1, 2].map((r) => (
        <circle key={r} cx={sx(0)} cy={sy(0)} r={r * k} fill="none" stroke="var(--text-faint)" strokeDasharray="4 4" strokeWidth={1} />
      ))}
      <text x={sx(0) + 2 * k * 0.72} y={sy(0) - 2 * k * 0.72 - 3} fontSize={9} fill="var(--text-faint)">prior N(0, I): 1σ, 2σ</text>
      {!grid &&
        mus.map((c, i) => (
          <text key={i} x={sx(c[0])} y={sy(c[1]) + 3.5} fontSize={10} textAnchor="middle" fill={CLASS_COLORS[labels[i]]} fontFamily="var(--font-mono)" fontWeight={600} opacity={showPosterior ? 0.55 : 1}>
            {labels[i]}
          </text>
        ))}
      {path && <polyline points={path.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} fill="none" stroke="var(--clay)" strokeWidth={2} />}
      {path && path.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={2.5} fill="var(--clay)" />)}
      {showPosterior && (
        <g>
          {cloud.map((z, i) => (
            <circle key={i} cx={sx(z[0])} cy={sy(z[1])} r={1.6} fill="var(--gold)" fillOpacity={0.55} />
          ))}
          <ellipse cx={sx(trace.mu[0])} cy={sy(trace.mu[1])} rx={2 * trace.sigma[0] * k} ry={2 * trace.sigma[1] * k} fill="none" stroke="var(--gold)" strokeWidth={1.2} strokeDasharray="3 3" />
          <ellipse cx={sx(trace.mu[0])} cy={sy(trace.mu[1])} rx={trace.sigma[0] * k} ry={trace.sigma[1] * k} fill="var(--gold-soft)" fillOpacity={0.6} stroke="var(--gold)" strokeWidth={2} />
          <circle cx={sx(trace.mu[0])} cy={sy(trace.mu[1])} r={3} fill="var(--gold)" />
        </g>
      )}
      {showZ && (
        <g>
          <line x1={sx(trace.mu[0])} y1={sy(trace.mu[1])} x2={sx(trace.z[0])} y2={sy(trace.z[1])} stroke="var(--accent)" strokeWidth={1.5} />
          <circle cx={sx(trace.z[0])} cy={sy(trace.z[1])} r={5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} />
        </g>
      )}
      {clicked && (
        <g>
          <line x1={sx(clicked[0]) - 7} x2={sx(clicked[0]) + 7} y1={sy(clicked[1])} y2={sy(clicked[1])} stroke="var(--clay)" strokeWidth={2.5} />
          <line x1={sx(clicked[0])} x2={sx(clicked[0])} y1={sy(clicked[1]) - 7} y2={sy(clicked[1]) + 7} stroke="var(--clay)" strokeWidth={2.5} />
        </g>
      )}
    </svg>
  );
}

/* ------------------------------------------------------------------ main */

type PathMode = "interp" | "z1" | "z2";

export default function VaeSim() {
  const [json, setJson] = useState<VaeJSON | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<{ maxAbsDiff: number; maxLossDiff: number; checked: number } | null>(null);
  const [idx, setIdx] = useState(0);
  const [idxB, setIdxB] = useState(5);
  const [epsSeed, setEpsSeed] = useState(1);
  const [priorSeed, setPriorSeed] = useState(1);
  const [clicked, setClicked] = useState<number[] | null>(null);
  const [view, setView] = useState<"scatter" | "grid">("scatter");
  const [pathMode, setPathMode] = useState<PathMode>("interp");

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<VaeJSON>;
      })
      .then((j) => {
        if (!alive) return;
        setJson(j);
        setCheck(verifyVae(j));
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const model: Vae | null = useMemo(() => (json ? loadVae(json) : null), [json]);
  const s = useStepper(STAGES.length, { baseMs: 2000 });
  const step = s.step;

  const x = useMemo(() => (json ? vaeTestImage(json, idx) : []), [json, idx]);
  const eps = useMemo(() => standardNormal(2, 1000 + epsSeed * 31 + idx), [epsSeed, idx]);
  const trace = useMemo(() => (model ? forwardVae(model, x, eps) : null), [model, x, eps]);
  const meanRecon = useMemo(() => (model && trace ? decodeVae(model, trace.mu).recon : null), [model, trace]);
  const cloud = useMemo(() => {
    if (!trace) return [];
    return Array.from({ length: 40 }, (_, i) => {
      const e = standardNormal(2, 50000 + i * 13 + idx);
      return [trace.mu[0] + trace.sigma[0] * e[0], trace.mu[1] + trace.sigma[1] * e[1]];
    });
  }, [trace, idx]);

  const priorSamples = useMemo(() => {
    if (!model) return [];
    return Array.from({ length: 16 }, (_, i) => {
      const z = standardNormal(2, priorSeed * 9973 + i * 101);
      return { z, px: decodeVae(model, z).recon };
    });
  }, [model, priorSeed]);

  const path = useMemo(() => {
    if (!json || !model) return null;
    const a = json.test.mu[idx];
    const b = json.test.mu[idxB];
    const n = 9;
    return Array.from({ length: n }, (_, i) => {
      const t = i / (n - 1);
      if (pathMode === "interp") return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
      const v = -2.5 + 5 * t;
      return pathMode === "z1" ? [v, a[1]] : [a[0], v];
    });
  }, [json, model, idx, idxB, pathMode]);
  const pathImgs = useMemo(() => (model && path ? path.map((z) => decodeVae(model, z).recon) : []), [model, path]);

  const grid = useMemo(() => {
    if (!model || view !== "grid") return null;
    const n = 11;
    const out: number[][] = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) out.push(decodeVae(model, [-R + ((c + 0.5) / n) * 2 * R, R - ((r + 0.5) / n) * 2 * R]).recon);
    return out;
  }, [model, view]);

  const clickedDecode = useMemo(() => (model && clicked ? decodeVae(model, clicked).recon : null), [model, clicked]);

  if (error) {
    return (
      <SimShell title={TITLE} subtitle="A real VAE trained on 8×8 handwritten digits.">
        <p className="text-sm text-clay">Couldn&apos;t load the model ({error}). Check your connection and reload the page.</p>
      </SimShell>
    );
  }
  if (!json || !model || !trace || !meanRecon) {
    return (
      <SimShell title={TITLE} subtitle="A real VAE trained on 8×8 handwritten digits.">
        <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
          Loading the trained model (≈160 KB)…
        </div>
      </SimShell>
    );
  }

  const nTest = json.test.labels.length;
  const label = json.test.labels[idx];
  const fin = json.training.final;
  const negElbo = trace.reconLoss + trace.kl;

  const controls = (
    <div className="flex flex-col gap-2.5">
      <Controls s={s} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <label className="flex items-center gap-2">
          <span className="text-muted">Test digit #</span>
          <input type="number" className="field w-20 font-mono" min={0} max={nTest - 1} value={idx} onChange={(e) => setIdx(Math.max(0, Math.min(nTest - 1, Math.round(+e.target.value) || 0)))} aria-label="Test image index" />
        </label>
        <button className="sim-btn" onClick={() => setIdx((i) => (i * 37 + 101) % nTest)} aria-label="Pick another test digit">
          <Shuffle size={14} /> Another digit
        </button>
        <button className="sim-btn" onClick={() => setEpsSeed((v) => v + 1)} aria-label="Draw a new epsilon">
          <Dices size={14} /> Resample ε
        </button>
        <span className="font-mono text-faint">ε seed {epsSeed}</span>
      </div>
    </div>
  );

  let stageBody: ReactNode;
  if (step === 0) {
    stageBody = <p className="text-xs text-muted">x = 64 grey levels from 0 to 1: [{x.slice(0, 8).map((v) => v.toFixed(2)).join(", ")}, …]. The model treats each pixel as a chance of being on (a Bernoulli probability).</p>;
  } else if (step === 1) {
    stageBody = (
      <div className="grid gap-3 sm:grid-cols-2">
        <VecCells v={trace.encH1} label="h₁ = ReLU(W₁x + b₁)" />
        <VecCells v={trace.encH2} label="h₂ = ReLU(W₂h₁ + b₂)" />
      </div>
    );
  } else if (step === 2) {
    stageBody = (
      <table className="font-mono text-[12px] tabular-nums">
        <tbody>
          <tr><td className="pr-3 text-muted">μ = W_μh₂ + b_μ</td><td>({fmt(trace.mu[0], 3)}, {fmt(trace.mu[1], 3)})</td></tr>
          <tr><td className="pr-3 text-muted">log σ² = W_vh₂ + b_v</td><td>({fmt(trace.logvar[0], 3)}, {fmt(trace.logvar[1], 3)})</td></tr>
          <tr><td className="pr-3 text-muted">σ = exp(½ log σ²)</td><td className="text-gold">({fmt(trace.sigma[0], 3)}, {fmt(trace.sigma[1], 3)})</td></tr>
        </tbody>
      </table>
    );
  } else if (step === 3) {
    stageBody = (
      <div className="font-mono text-[12px]">
        ε = ({fmt(trace.eps[0], 3)}, {fmt(trace.eps[1], 3)}) <span className="text-xs text-muted">— drawn from N(0, I) by a fixed-seed random generator. It does not depend on any parameter. The yellow dots are 40 other draws pushed through μ + σ⊙ε.</span>
      </div>
    );
  } else if (step === 4) {
    stageBody = (
      <div className="grid gap-1 font-mono text-[12px] tabular-nums">
        {[0, 1].map((j) => (
          <div key={j}>
            z<sub>{j + 1}</sub> = μ<sub>{j + 1}</sub> + σ<sub>{j + 1}</sub>·ε<sub>{j + 1}</sub> = {fmt(trace.mu[j], 3)} + {fmt(trace.sigma[j], 3)} × {fmt(trace.eps[j], 3)} = <span className="text-accent">{fmt(trace.z[j], 3)}</span>
          </div>
        ))}
        <div className="text-xs text-muted">∂z/∂μ = 1 and ∂z/∂σ = ε, so gradients can flow through the sample back to the encoder.</div>
      </div>
    );
  } else if (step === 5) {
    stageBody = (
      <div className="grid gap-3 sm:grid-cols-2">
        <VecCells v={trace.decH1} label="d₁ = ReLU(V₁z + c₁)" />
        <VecCells v={trace.decH2} label="d₂ = ReLU(V₂d₁ + c₂)" />
      </div>
    );
  } else {
    stageBody = (
      <div className="grid gap-1 font-mono text-[12px] tabular-nums">
        <div>reconstruction −log p(x|z) = Σ BCE(x̂, x) = <span className="text-accent">{fmt(trace.reconLoss, 3)}</span> nats</div>
        <div>
          KL(q(z|x) ‖ N(0,I)) = ½Σ(μ² + σ² − 1 − log σ²) = {fmt(trace.klPerDim[0], 3)} + {fmt(trace.klPerDim[1], 3)} = <span className="text-clay">{fmt(trace.kl, 3)}</span> nats
        </div>
        <div>
          −ELBO (one sample) = {fmt(trace.reconLoss, 3)} + {fmt(trace.kl, 3)} = <b>{fmt(negElbo, 3)}</b> nats
        </div>
        <div className="text-xs text-muted">Average over the test set (PyTorch): reconstruction {fin.test_recon}, KL {fin.test_kl}, −ELBO {fin.test_neg_elbo} nats per image.</div>
      </div>
    );
  }

  const showPosterior = step >= 2;
  const showZ = step >= 4;

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={`${step + 1}. ${STAGES[step]}`}>{narrate(step, trace, label)}</Narration>
      <div className="rounded-lg border border-border p-2.5 text-[11px] leading-snug text-muted">
        <span className="font-medium text-accent">A real trained model.</span> A VAE with a 2-D Gaussian code (latent): encoder 64→64→32→(μ, log σ²), decoder 2→32→64→64 with sigmoid (Bernoulli) outputs, {model.nParams.toLocaleString()} parameters. Trained with PyTorch for {json.training.epochs} epochs on {json.data.n_train.toLocaleString()} of scikit-learn&apos;s 8×8 digits (labels not used). Loss = BCE summed over pixels + KL, β = {json.config.beta}, one sample of ε per image.
      </div>
      {check && (
        <div className="rounded-lg border border-border p-2.5 font-mono text-[11px] leading-snug text-muted">
          browser vs PyTorch: max |Δ| = {check.maxAbsDiff.toExponential(1)} over {check.checked} activations, loss terms within {check.maxLossDiff.toExponential(1)} {check.maxAbsDiff < 1e-3 ? "✓" : "✗"}
        </div>
      )}
    </div>
  );

  return (
    <SimShell title={TITLE} subtitle="Follow one digit through the encoder, the random sample and the decoder. Then create brand-new digits from random codes." controls={controls} aside={aside}>
      <nav className="mb-3 flex flex-wrap gap-1" aria-label="Stages">
        {STAGES.map((name, k) => (
          <button
            key={k}
            onClick={() => s.setStep(k)}
            className="rounded-md border px-1.5 py-0.5 text-[11px]"
            style={{ borderColor: k === step ? "var(--accent)" : "var(--border)", background: k === step ? "var(--accent-soft)" : "transparent", color: k === step ? "var(--text)" : "var(--text-muted)" }}
            aria-current={k === step ? "step" : undefined}
            aria-label={name}
          >
            {SHORT[k]}
          </button>
        ))}
      </nav>
      <div className="mb-3 flex flex-wrap items-start justify-center gap-4">
        <Digit px={x} label={`test #${idx} (label ${label})`} highlight={step === 0} />
        <Digit px={trace.recon} label={step >= 5 ? `x̂ = decoder(z), z sampled` : "x̂ (after the decoder step)"} highlight={step === 5} />
        <Digit px={meanRecon} label="decoder(μ): the mean code" />
        <Digit px={trace.recon.map((v, i) => v - x[i])} error label="error x̂ − x" />
      </div>
      <div className="mb-4 min-h-[3rem] rounded-lg border border-border p-3">{stageBody}</div>

      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,11rem)]">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
            <span className="font-medium">Latent space: the mean code μ of each test digit</span>
            <button className={`sim-btn ${view === "scatter" ? "primary" : ""}`} onClick={() => setView("scatter")} aria-pressed={view === "scatter"}>
              test μ
            </button>
            <button className={`sim-btn ${view === "grid" ? "primary" : ""}`} onClick={() => setView("grid")} aria-pressed={view === "grid"}>
              decoded map
            </button>
          </div>
          <LatentPlot mus={json.test.mu} labels={json.test.labels} trace={trace} showPosterior={showPosterior} showZ={showZ} cloud={showPosterior && step >= 3 ? cloud : []} path={path} clicked={clicked} grid={grid} onClick={setClicked} />
          <p className="mt-1 text-[11px] text-muted">
            Gold oval = q(z|x) at 1σ (solid) and 2σ (dashed). Green dot = the sampled z. Clay line = the path used below. Dashed circles = the prior N(0, I). The KL term packs the codes into the prior&apos;s disc. So, unlike a plain autoencoder, almost every point inside it decodes to a digit-like image. Click to decode any point.
          </p>
        </div>
        <div className="flex min-w-0 flex-col items-center gap-2">
          {clicked && clickedDecode ? (
            <>
              <Digit px={clickedDecode} size={120} label={`decoder(${fmt(clicked[0], 2)}, ${fmt(clicked[1], 2)})`} highlight />
              <p className="text-center text-[11px] text-muted">
                ‖z‖ = {fmt(Math.hypot(clicked[0], clicked[1]), 2)}; prior density compared with the centre: {fmt(Math.exp(-0.5 * (clicked[0] ** 2 + clicked[1] ** 2)), 3)}
              </p>
              <button className="sim-btn" onClick={() => setClicked(null)} aria-label="Clear clicked point">
                clear
              </button>
            </>
          ) : (
            <p className="text-center text-[11px] text-muted">Click the latent plot to decode any point with the real decoder.</p>
          )}
        </div>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <section className="min-w-0" aria-label="Samples from the prior">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
            <span className="font-medium">Generate: z ~ N(0, I), x = decoder(z)</span>
            <button className="sim-btn" onClick={() => setPriorSeed((v) => v + 1)} aria-label="Draw new prior samples">
              <Dices size={14} /> new samples
            </button>
          </div>
          <div className="grid grid-cols-8 gap-1">
            {priorSamples.map((p, i) => (
              <button key={i} onClick={() => setClicked(p.z)} className="min-w-0" aria-label={`Prior sample at z = (${p.z[0].toFixed(2)}, ${p.z[1].toFixed(2)}); show in latent plot`}>
                <Digit px={p.px} size={44} />
              </button>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-muted">16 brand-new images. None of them is a training example. They look smooth and a bit blurry, because each pixel is the decoder&apos;s average (Bernoulli mean), not a sample. Click one to find its z.</p>
        </section>
        <section className="min-w-0" aria-label="Interpolation and traversal">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
            {(["interp", "z1", "z2"] as PathMode[]).map((m) => (
              <button key={m} className={`sim-btn ${pathMode === m ? "primary" : ""}`} onClick={() => setPathMode(m)} aria-pressed={pathMode === m}>
                {m === "interp" ? "interpolate A → B" : m === "z1" ? "traverse z₁" : "traverse z₂"}
              </button>
            ))}
            {pathMode === "interp" && (
              <label className="flex items-center gap-1">
                <span className="text-muted">B = test #</span>
                <input type="number" className="field w-16 font-mono" min={0} max={nTest - 1} value={idxB} onChange={(e) => setIdxB(Math.max(0, Math.min(nTest - 1, Math.round(+e.target.value) || 0)))} aria-label="Second test image index" />
                <span className="text-faint">({json.test.labels[idxB]})</span>
              </label>
            )}
          </div>
          <div className="grid grid-cols-9 gap-1">
            {pathImgs.map((px, i) => (
              <Digit key={i} px={px} size={40} />
            ))}
          </div>
          <p className="mt-1 text-[11px] text-muted">
            {pathMode === "interp" ? `Decoding 9 evenly spaced points on the straight line from μ(A = #${idx}, a ${label}) to μ(B = #${idxB}, a ${json.test.labels[idxB]}). The images change slowly, and every point on the way looks like a real digit.` : `Keeping the other coordinate at this digit's μ and sliding z${pathMode === "z1" ? "₁" : "₂"} from −2.5 to 2.5. There is no promise that a latent axis means something a human would name.`}
          </p>
        </section>
      </div>

      <section className="mt-5 min-w-0" aria-label="Effect of beta">
        <div className="mb-1 text-xs font-medium">β-VAE: three models trained separately (same data, seed and design; loss = reconstruction + β·KL)</div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[20rem] text-left font-mono text-[11px] tabular-nums">
            <thead className="text-muted">
              <tr>
                <th className="pr-3 font-normal">β</th>
                <th className="pr-3 font-normal">test recon (nats)</th>
                <th className="pr-3 font-normal">test KL (nats)</th>
                <th className="pr-3 font-normal">mean σ per dim</th>
                <th className="font-normal">active dims</th>
              </tr>
            </thead>
            <tbody>
              {json.beta_sweep.map((b) => (
                <tr key={b.beta} className={b.beta === json.config.beta ? "text-accent" : ""}>
                  <td className="pr-3">{b.beta}{b.beta === json.config.beta ? " (this model)" : ""}</td>
                  <td className="pr-3">{b.test_recon}</td>
                  <td className="pr-3">{b.test_kl}</td>
                  <td className="pr-3">{b.mean_sigma.join(", ")}</td>
                  <td>{b.active_dims} / {json.config.latent}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-1 text-[11px] text-muted">
          Small β acts like a plain autoencoder (tiny σ, sharp rebuilds, high KL). At β = 4 the KL penalty wins completely on this tiny model. Every q(z|x) equals the prior (σ = 1, KL = 0), so the code is ignored and every input decodes to the same average digit. This is called posterior collapse. A dimension counts as active when the variance of μ across test images is above 0.01.
        </p>
      </section>
    </SimShell>
  );
}

function narrate(step: number, t: VaeTrace, label: number): ReactNode {
  switch (step) {
    case 0:
      return `A ${label} from the test set (never seen in training). A VAE imagines each image was made from a hidden 2-D code z ~ N(0, I), passed through the decoder. Encoding means guessing which z could have made it.`;
    case 1:
      return `The encoder is an ordinary MLP: 64 → 64 → 32 ReLU units. Unlike a plain autoencoder, it does not output one single code. Its last layer has two heads.`;
    case 2:
      return `The heads output a mean μ = (${fmt(t.mu[0], 2)}, ${fmt(t.mu[1], 2)}) and a log-variance; σ = (${fmt(t.sigma[0], 3)}, ${fmt(t.sigma[1], 3)}). Together they define q(z|x) = N(μ, diag σ²), the posterior: "this image came from somewhere in this oval". Predicting log σ² keeps σ positive with no extra rules.`;
    case 3:
      return `The randomness is moved outside the network. ε = (${fmt(t.eps[0], 2)}, ${fmt(t.eps[1], 2)}) is drawn from a fixed N(0, I) that depends on no parameter. Press "Resample ε" to draw another.`;
    case 4:
      return `The reparameterisation trick: z = μ + σ ⊙ ε = (${fmt(t.z[0], 2)}, ${fmt(t.z[1], 2)}) has exactly the distribution N(μ, σ²). But now it is a smooth function of μ and σ, so backpropagation can train the encoder through the sample.`;
    case 5:
      return `The decoder p(x|z) turns the 2-D sample back into 64 pixel probabilities. Because z is noisy during training, the decoder must make similar images for nearby z. That is why the latent space is smooth.`;
    default:
      return `Loss for this sample: reconstruction ${fmt(t.reconLoss, 2)} + KL ${fmt(t.kl, 2)} = ${fmt(t.reconLoss + t.kl, 2)} nats = −ELBO. The KL term charges ${fmt(t.kl, 2)} nats because this image's q(z|x) is narrower than the prior and shifted away from it. Lowering the total over all data trains the encoder and decoder together.`;
  }
}
