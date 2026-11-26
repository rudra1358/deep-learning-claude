"use client";

import { useEffect, useMemo, useState, type MouseEvent, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import {
  addNoise,
  bceSum,
  decode,
  forwardAe,
  loadAutoencoder,
  mse,
  testImage,
  verifyAutoencoder,
  type AeTrace,
  type Autoencoder,
  type AutoencoderJSON,
} from "@/lib/sim/autoencoder";

const MODEL_URL = "/models/autoencoder.json";
const TITLE = "Autoencoder (trained model)";
const STAGES = ["Input x", "Encoder layer 1", "Encoder layer 2", "Bottleneck z", "Decoder layer 1", "Decoder layer 2", "Reconstruction x̂"] as const;
const SHORT = ["x", "enc 64", "enc 32", "z (2)", "dec 32", "dec 64", "x̂"];
const SIZES = [64, 64, 32, 2, 32, 64, 64];

/** Ten theme-safe class colours built only from the palette variables (markers are also the digit glyphs). */
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

/** An 8×8 grey image drawn as ink (var(--text)) on the surface colour. */
function Digit({ px, label, size = 96, error = false, highlight = false }: { px: number[]; label: string; size?: number; error?: boolean; highlight?: boolean }) {
  return (
    <figure className="flex min-w-0 flex-col items-center">
      <svg viewBox="0 0 8 8" width={size} height={size} role="img" aria-label={label} className="rounded-sm border" style={{ borderColor: highlight ? "var(--gold)" : "var(--border)" }}>
        <rect width={8} height={8} fill="var(--surface)" />
        {px.map((v, i) =>
          error ? (
            <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill={signedColor(v, 0.5)} />
          ) : (
            <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill="var(--text)" fillOpacity={Math.max(0, Math.min(1, v))} />
          ),
        )}
      </svg>
      <figcaption className="mt-1 max-w-[8rem] text-center text-[11px] leading-tight text-muted">{label}</figcaption>
    </figure>
  );
}

/** A vector of activations as a strip of cells (wrapped every `cols`). */
function VecCells({ v, cols = 16, signed = false, label }: { v: number[]; cols?: number; signed?: boolean; label: string }) {
  const rows = Math.ceil(v.length / cols);
  const max = Math.max(1e-9, ...v.map(Math.abs));
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 text-xs">
        <span className="font-medium">{label}</span> <span className="font-mono text-faint">({v.length} values, max |a| = {fmt(max, 2)}, {v.filter((x) => x === 0).length} exactly 0)</span>
      </figcaption>
      <svg viewBox={`0 0 ${cols} ${rows}`} className="h-auto w-full" style={{ maxWidth: cols * 22 }} role="img" aria-label={`${label}: ${v.length} activations`}>
        {v.map((x, i) => (
          <rect key={i} x={i % cols} y={Math.floor(i / cols)} width={0.92} height={0.92} rx={0.08} fill={x === 0 ? "var(--border)" : signed ? signedColor(x, max) : `color-mix(in srgb, var(--accent) ${Math.round(15 + 85 * (Math.abs(x) / max))}%, var(--surface))`} />
        ))}
      </svg>
    </figure>
  );
}

/** Pipeline diagram: 7 boxes whose heights reflect layer width; click to jump to a stage. */
function Pipeline({ step, onPick }: { step: number; onPick: (k: number) => void }) {
  const W = 560;
  const H = 92;
  const bw = 52;
  const gap = (W - 7 * bw) / 6;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mb-3 h-auto w-full" role="img" aria-label="Autoencoder architecture; the current stage is highlighted">
      {SIZES.map((n, k) => {
        const h = 10 + (n / 64) * 58;
        const x = k * (bw + gap);
        const y = (70 - h) / 2 + 2;
        const active = k === step;
        const isCode = k === 3;
        return (
          <g key={k} onClick={() => onPick(k)} className="cursor-pointer">
            {k > 0 && <line x1={x - gap + 2} x2={x - 2} y1={37} y2={37} stroke="var(--border-strong)" />}
            <rect x={x} y={y} width={bw} height={h} rx={4} fill={active ? "var(--gold-soft)" : isCode ? "var(--accent-soft)" : "var(--surface)"} stroke={active ? "var(--gold)" : "var(--border-strong)"} strokeWidth={active ? 2 : 1} />
            <text x={x + bw / 2} y={86} fontSize={11} textAnchor="middle" fill={active ? "var(--text)" : "var(--text-muted)"} fontFamily="var(--font-mono)">
              {SHORT[k]}
            </text>
          </g>
        );
      })}
      <text x={(bw * 3 + gap * 2.5) / 2} y={10} fontSize={10} textAnchor="middle" fill="var(--text-faint)">encoder</text>
      <text x={W - (bw * 3 + gap * 2.5) / 2} y={10} fontSize={10} textAnchor="middle" fill="var(--text-faint)">decoder</text>
    </svg>
  );
}

/* ------------------------------------------------------------------ latent plot */

interface Box {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

const PW = 420;
const PH = 360;

function LatentPlot({
  box,
  codes,
  labels,
  trainCodes,
  current,
  clicked,
  grid,
  onClick,
  ariaLabel,
}: {
  box: Box;
  codes: number[][];
  labels: number[];
  trainCodes: number[][] | null;
  current: number[];
  clicked: number[] | null;
  grid: { z: number[]; px: number[] }[] | null;
  onClick: (z: number[]) => void;
  ariaLabel: string;
}) {
  const sx = (x: number) => ((x - box.x0) / (box.x1 - box.x0)) * PW;
  const sy = (y: number) => PH - ((y - box.y0) / (box.y1 - box.y0)) * PH;
  const handle = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * (PW + 30) - 26; // undo the viewBox offset (-26, -4)
    const py = ((e.clientY - r.top) / r.height) * (PH + 22) - 4;
    onClick([box.x0 + (px / PW) * (box.x1 - box.x0), box.y0 + ((PH - py) / PH) * (box.y1 - box.y0)]);
  };
  const ticks = (a: number, b: number) => {
    const step = Math.pow(10, Math.floor(Math.log10((b - a) / 2)));
    const s = (b - a) / step > 8 ? step * 2 : step;
    const out: number[] = [];
    for (let t = Math.ceil(a / s) * s; t <= b; t += s) out.push(Math.round(t * 1000) / 1000);
    return out;
  };
  const cell = grid ? Math.sqrt(grid.length) : 0;
  return (
    <svg viewBox={`-26 -4 ${PW + 30} ${PH + 22}`} className="h-auto w-full cursor-crosshair touch-manipulation" onClick={handle} role="img" aria-label={ariaLabel}>
      <rect x={0} y={0} width={PW} height={PH} fill="var(--surface)" stroke="var(--border)" />
      {grid &&
        grid.map((g, k) => {
          const cw = PW / cell;
          const ch = PH / cell;
          const gx = (k % cell) * cw;
          const gy = Math.floor(k / cell) * ch;
          const s = Math.min(cw, ch) * 0.92;
          return (
            <g key={k} transform={`translate(${gx + (cw - s) / 2},${gy + (ch - s) / 2}) scale(${s / 8})`}>
              {g.px.map((v, i) => (v > 0.04 ? <rect key={i} x={i % 8} y={Math.floor(i / 8)} width={1} height={1} fill="var(--text)" fillOpacity={Math.min(1, v)} /> : null))}
            </g>
          );
        })}
      {ticks(box.x0, box.x1).map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={0} y2={PH} stroke="var(--border)" strokeWidth={t === 0 ? 1 : 0.5} strokeDasharray={t === 0 ? undefined : "2 3"} />
          <text x={sx(t)} y={PH + 13} fontSize={10} textAnchor="middle" fill="var(--text-faint)">{t}</text>
        </g>
      ))}
      {ticks(box.y0, box.y1).map((t) => (
        <g key={`y${t}`}>
          <line x1={0} x2={PW} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={t === 0 ? 1 : 0.5} strokeDasharray={t === 0 ? undefined : "2 3"} />
          <text x={-4} y={sy(t) + 3} fontSize={10} textAnchor="end" fill="var(--text-faint)">{t}</text>
        </g>
      ))}
      {!grid && trainCodes && trainCodes.map((c, i) => <circle key={`t${i}`} cx={sx(c[0])} cy={sy(c[1])} r={1.3} fill="var(--text-faint)" fillOpacity={0.45} />)}
      {!grid &&
        codes.map((c, i) => (
          <text key={i} x={sx(c[0])} y={sy(c[1]) + 3.5} fontSize={10} textAnchor="middle" fill={CLASS_COLORS[labels[i]]} fontFamily="var(--font-mono)" fontWeight={600}>
            {labels[i]}
          </text>
        ))}
      <circle cx={sx(current[0])} cy={sy(current[1])} r={7} fill="none" stroke="var(--gold)" strokeWidth={2.5} />
      {clicked && (
        <g>
          <line x1={sx(clicked[0]) - 7} x2={sx(clicked[0]) + 7} y1={sy(clicked[1])} y2={sy(clicked[1])} stroke="var(--clay)" strokeWidth={2.5} />
          <line x1={sx(clicked[0])} x2={sx(clicked[0])} y1={sy(clicked[1]) - 7} y2={sy(clicked[1]) + 7} stroke="var(--clay)" strokeWidth={2.5} />
        </g>
      )}
      <text x={PW - 4} y={PH + 13} fontSize={10} textAnchor="end" fill="var(--text-muted)">z₁ →</text>
      <text x={4} y={10} fontSize={10} fill="var(--text-muted)">z₂ ↑</text>
    </svg>
  );
}

/* ------------------------------------------------------------------ main */

type Which = "ae" | "dae";

export default function AutoencoderSim() {
  const [json, setJson] = useState<AutoencoderJSON | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<{ maxAbsDiff: number; checked: number } | null>(null);
  const [idx, setIdx] = useState(0);
  const [which, setWhich] = useState<Which>("ae");
  const [noise, setNoise] = useState(0);
  const [noiseSeed, setNoiseSeed] = useState(1);
  const [clicked, setClicked] = useState<number[] | null>(null);
  const [view, setView] = useState<"scatter" | "grid">("scatter");
  const [zoom, setZoom] = useState(1);
  const [showTrain, setShowTrain] = useState(false);

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<AutoencoderJSON>;
      })
      .then((j) => {
        if (!alive) return;
        setJson(j);
        setCheck(verifyAutoencoder(j));
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const models = useMemo<Record<Which, Autoencoder> | null>(() => (json ? { ae: loadAutoencoder(json.models.ae), dae: loadAutoencoder(json.models.dae) } : null), [json]);
  const s = useStepper(STAGES.length, { baseMs: 1800 });

  const clean = useMemo(() => (json ? testImage(json, idx) : []), [json, idx]);
  const input = useMemo(() => (noise > 0 ? addNoise(clean, noise, noiseSeed * 7919 + idx) : clean), [clean, noise, noiseSeed, idx]);
  const trace: AeTrace | null = useMemo(() => (models ? forwardAe(models[which], input) : null), [models, which, input]);
  const other: AeTrace | null = useMemo(() => (models ? forwardAe(models[which === "ae" ? "dae" : "ae"], input) : null), [models, which, input]);

  // data bounding box of this model's codes (train + test) with a margin; zoom widens it around the centre
  const box = useMemo<Box | null>(() => {
    if (!json) return null;
    const m = json.models[which];
    const all = [...m.codes_train, ...m.codes_test];
    const xs = all.map((c) => c[0]);
    const ys = all.map((c) => c[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const half = (Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2) * 1.08 * zoom;
    return { x0: cx - half, x1: cx + half, y0: cy - half * (PH / PW), y1: cy + half * (PH / PW) };
  }, [json, which, zoom]);

  // typical nearest-neighbour gap between training codes (median), for judging how far a clicked point is from the data
  const nnInfo = useMemo(() => {
    if (!json) return null;
    const tr = json.models[which].codes_train;
    const nn = tr.map((a, i) => {
      let best = Infinity;
      for (let j = 0; j < tr.length; j++) if (j !== i) best = Math.min(best, Math.hypot(a[0] - tr[j][0], a[1] - tr[j][1]));
      return best;
    });
    nn.sort((a, b) => a - b);
    return { median: nn[Math.floor(nn.length / 2)], codes: tr };
  }, [json, which]);

  const grid = useMemo(() => {
    if (!models || !box || view !== "grid") return null;
    const n = 11;
    const out: { z: number[]; px: number[] }[] = [];
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        const z = [box.x0 + ((c + 0.5) / n) * (box.x1 - box.x0), box.y1 - ((r + 0.5) / n) * (box.y1 - box.y0)];
        out.push({ z, px: decode(models[which], z).recon });
      }
    return out;
  }, [models, box, view, which]);

  const clickedDecode = useMemo(() => (models && clicked ? decode(models[which], clicked).recon : null), [models, clicked, which]);
  const clickedDist = useMemo(() => {
    if (!clicked || !nnInfo) return null;
    let best = Infinity;
    let at = 0;
    nnInfo.codes.forEach((c, i) => {
      const d = Math.hypot(c[0] - clicked[0], c[1] - clicked[1]);
      if (d < best) {
        best = d;
        at = i;
      }
    });
    return { d: best, label: json ? json.test.train_labels[at] : -1 };
  }, [clicked, nnInfo, json]);

  if (error) {
    return (
      <SimShell title={TITLE} subtitle="A real autoencoder trained on 8×8 handwritten digits.">
        <p className="text-sm text-clay">Could not load the model ({error}). Check your internet connection and reload the page.</p>
      </SimShell>
    );
  }
  if (!json || !models || !trace || !other || !box || !nnInfo) {
    return (
      <SimShell title={TITLE} subtitle="A real autoencoder trained on 8×8 handwritten digits.">
        <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
          Loading the trained model (about 240 KB: two autoencoders and 1,797 codes)…
        </div>
      </SimShell>
    );
  }

  const nTest = json.test.labels.length;
  const label = json.test.labels[idx];
  const err = trace.recon.map((v, i) => v - clean[i]);
  const mseClean = mse(trace.recon, clean);
  const bce = bceSum(trace.recon, clean);
  const step = s.step;
  const info = json.models[which];

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
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Choose model">
          <span className="text-muted">Model</span>
          <button className={`sim-btn ${which === "ae" ? "primary" : ""}`} onClick={() => setWhich("ae")} aria-pressed={which === "ae"}>
            Autoencoder
          </button>
          <button className={`sim-btn ${which === "dae" ? "primary" : ""}`} onClick={() => setWhich("dae")} aria-pressed={which === "dae"}>
            Denoising AE
          </button>
        </div>
      </div>
      <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
        <Slider label="Input noise σ" value={noise} min={0} max={0.6} step={0.05} onChange={setNoise} />
        <button className="sim-btn" onClick={() => setNoiseSeed((v) => v + 1)} disabled={noise === 0} aria-label="Draw new noise">
          new noise (seed {noiseSeed})
        </button>
      </div>
    </div>
  );

  /* --------------------------- stage body */
  let stageBody: ReactNode;
  if (step === 0) {
    stageBody = (
      <p className="text-xs text-muted">
        The input is the 8×8 image laid out as one row of 64 numbers in [0, 1] (grey level / 16){noise > 0 ? `, plus Gaussian noise of σ = ${noise.toFixed(2)}, clipped back to [0, 1]` : ""}. First pixels: [{input.slice(0, 8).map((v) => v.toFixed(2)).join(", ")}, …]
      </p>
    );
  } else if (step === 1) stageBody = <VecCells v={trace.encH1} label="h₁ = ReLU(W₁x + b₁)" cols={16} />;
  else if (step === 2) stageBody = <VecCells v={trace.encH2} label="h₂ = ReLU(W₂h₁ + b₂)" cols={16} />;
  else if (step === 3)
    stageBody = (
      <div className="font-mono text-sm">
        z = W₃h₂ + b₃ = (<span className="text-accent">{fmt(trace.z[0], 3)}</span>, <span className="text-accent">{fmt(trace.z[1], 3)}</span>) <span className="text-xs text-muted">: the whole image squeezed into 2 numbers (gold ring in the plot below)</span>
      </div>
    );
  else if (step === 4) stageBody = <VecCells v={trace.decH1} label="d₁ = ReLU(V₁z + c₁)" cols={16} />;
  else if (step === 5) stageBody = <VecCells v={trace.decH2} label="d₂ = ReLU(V₂d₁ + c₂)" cols={16} />;
  else
    stageBody = (
      <div className="grid gap-1 font-mono text-[11px] text-muted">
        <div>x̂ = sigmoid(V₃d₂ + c₃): 64 pixel intensities in (0, 1)</div>
        <div>
          reconstruction loss (vs the clean image): BCE = {fmt(bce, 3)} nats, summed over 64 pixels · MSE = {fmt(mseClean, 4)} · largest pixel error {fmt(Math.max(...err.map(Math.abs)), 3)}
        </div>
        <div>
          test-set average for this model: BCE {info.final.test_bce} · MSE {info.final.test_mse}
        </div>
      </div>
    );

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={`${step + 1}. ${STAGES[step]}`}>{narrate(step, trace, label, which, noise, mseClean)}</Narration>
      <div className="rounded-lg border border-border p-2.5 text-[11px] leading-snug text-muted">
        <span className="font-medium text-accent">This is a real trained model.</span> Two MLP autoencoders (64→64→32→<b>2</b>→32→64→64, ReLU, sigmoid output; {info.n_params.toLocaleString()} parameters each), trained in PyTorch for {json.training.epochs} epochs on {json.data.n_train.toLocaleString()} of scikit-learn&apos;s 8×8 handwritten digits. Loss = binary cross-entropy summed over pixels. Labels were never used. The scatter shows the {nTest} test digits it never trained on. The denoising twin was trained on inputs with Gaussian noise added (σ up to 0.5).
      </div>
      {check && (
        <div className="rounded-lg border border-border p-2.5 font-mono text-[11px] leading-snug text-muted">
          your browser vs PyTorch: max |Δ| = {check.maxAbsDiff.toExponential(1)} over {check.checked} activations {check.maxAbsDiff < 1e-3 ? "✓" : "✗"}
        </div>
      )}
    </div>
  );

  return (
    <SimShell title={TITLE} subtitle="Squeeze a digit into 2 numbers and rebuild it. Then click anywhere in the 2-D code space to see what the decoder draws." controls={controls} aside={aside}>
      <Pipeline step={step} onPick={(k) => s.setStep(k)} />
      <div className="mb-3 flex flex-wrap items-start justify-center gap-4">
        <Digit px={clean} label={`test #${idx} (label ${label})`} highlight={step === 0 && noise === 0} />
        {noise > 0 && <Digit px={input} label={`input with noise σ=${noise.toFixed(2)}`} highlight={step === 0} />}
        <Digit px={trace.recon} label={`reconstruction x̂ (${which === "ae" ? "AE" : "DAE"})`} highlight={step === 6} />
        <Digit px={err} error label="error x̂ − x (accent = too bright, clay = too dark)" />
      </div>
      <div className="mb-4 min-h-[3rem] rounded-lg border border-border p-3">{stageBody}</div>

      {noise > 0 && (
        <div className="mb-4 overflow-x-auto">
          <table className="w-full min-w-[18rem] text-left font-mono text-[11px] tabular-nums">
            <thead className="text-muted">
              <tr>
                <th className="pr-3 font-normal">MSE against the clean image</th>
                <th className="pr-3 font-normal">this digit</th>
                <th className="font-normal">test set at σ = {Math.min(0.5, Math.round(noise * 10) / 10).toFixed(1)} (PyTorch)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="pr-3">noisy input itself</td>
                <td className="pr-3">{fmt(mse(input, clean), 4)}</td>
                <td>{json.models.ae.denoise[Math.min(0.5, Math.round(noise * 10) / 10).toFixed(1)]?.noisy_mse}</td>
              </tr>
              {(["ae", "dae"] as Which[]).map((w) => (
                <tr key={w} className={w === which ? "text-accent" : ""}>
                  <td className="pr-3">{w === "ae" ? "plain AE output" : "denoising AE output"}</td>
                  <td className="pr-3">{fmt(mse((w === which ? trace : other).recon, clean), 4)}</td>
                  <td>{json.models[w].denoise[Math.min(0.5, Math.round(noise * 10) / 10).toFixed(1)]?.recon_mse}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,11rem)]">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
            <span className="font-medium">Latent (code) space of the {which === "ae" ? "autoencoder" : "denoising AE"}</span>
            <button className={`sim-btn ${view === "scatter" ? "primary" : ""}`} onClick={() => setView("scatter")} aria-pressed={view === "scatter"}>
              test codes
            </button>
            <button className={`sim-btn ${view === "grid" ? "primary" : ""}`} onClick={() => setView("grid")} aria-pressed={view === "grid"}>
              decoded map
            </button>
            <button className="sim-btn" onClick={() => setZoom((z) => (z === 1 ? 3 : 1))} aria-label="Toggle zoom">
              {zoom === 1 ? "zoom out ×3" : "back to data range"}
            </button>
            {view === "scatter" && (
              <label className="flex items-center gap-1 text-muted">
                <input type="checkbox" checked={showTrain} onChange={(e) => setShowTrain(e.target.checked)} /> training codes
              </label>
            )}
          </div>
          <LatentPlot
            box={box}
            codes={info.codes_test}
            labels={json.test.labels}
            trainCodes={showTrain ? info.codes_train : null}
            current={trace.z}
            clicked={clicked}
            grid={grid}
            onClick={setClicked}
            ariaLabel="Latent space: test digits plotted at their 2-D codes, labelled by class. Click to decode a point."
          />
          <p className="mt-1 text-[11px] text-muted">
            {view === "scatter" ? "Each test digit is drawn as its label, placed at its code z. Gold ring = the chosen digit. Click anywhere to decode that point." : "Each cell shows what the decoder draws for the z at its centre. Real codes fill only an uneven patch of the plane. Between and beyond the clusters, the decoder never trained, so it just guesses. Zoom out ×3 to see hard, blocky, overly bold strokes."}
          </p>
        </div>
        <div className="flex min-w-0 flex-col items-center gap-2 text-xs">
          {clicked && clickedDecode && clickedDist ? (
            <>
              <Digit px={clickedDecode} size={120} label={`decoder(${fmt(clicked[0], 2)}, ${fmt(clicked[1], 2)})`} highlight />
              <p className="text-center text-[11px] text-muted">
                The nearest training code is {fmt(clickedDist.d, 2)} away (a {clickedDist.label}). That is <b className={clickedDist.d > 5 * nnInfo.median ? "text-clay" : "text-accent"}>{(clickedDist.d / nnInfo.median).toFixed(1)}×</b> the usual (median) gap between neighbouring training codes ({fmt(nnInfo.median, 3)}).
                {clickedDist.d > 5 * nnInfo.median ? " This is a hole: the decoder was never asked to draw anything here." : " Close to real data, so expect a believable digit."}
              </p>
              <button className="sim-btn" onClick={() => setClicked(null)} aria-label="Clear clicked point">
                clear
              </button>
            </>
          ) : (
            <p className="text-center text-[11px] text-muted">Click the plot to decode any point z = (z₁, z₂) with the real decoder.</p>
          )}
        </div>
      </div>
    </SimShell>
  );
}

function narrate(step: number, t: AeTrace, label: number, which: Which, noise: number, mseClean: number): ReactNode {
  const active = (v: number[]) => v.filter((x) => x > 0).length;
  switch (step) {
    case 0:
      return `A ${label} from the test set (never seen in training): 64 grey levels in [0, 1]${noise > 0 ? `, with noise σ = ${noise.toFixed(2)} added` : ""}. The network must squeeze it down to just 2 numbers (the bottleneck), then rebuild the clean image.`;
    case 1:
      return `The first encoder layer turns 64 pixels into 64 features: h₁ = ReLU(W₁x + b₁). ${active(t.encH1)} of 64 units are active (positive). ReLU sets the rest to exactly 0.`;
    case 2:
      return `The second encoder layer shrinks it to 32 features; ${active(t.encH2)} of 32 are active. Each layer throws away details not needed to rebuild the digit.`;
    case 3:
      return `The bottleneck: a linear layer outputs the code z = (${fmt(t.z[0], 2)}, ${fmt(t.z[1], 2)}). All the decoder will know about this image is these two numbers, 32 times fewer than the 64 pixels. Training (with no labels) put similar-looking digits near each other.`;
    case 4:
      return `The decoder mirrors the encoder: d₁ = ReLU(V₁z + c₁) grows 2 → 32 (${active(t.decH1)} active). The decoder only gets z. It cannot peek at the input.`;
    case 5:
      return `d₂ = ReLU(V₂d₁ + c₂) grows to 64 features (${active(t.decH2)} active). One more step and we have pixels.`;
    default:
      return `The output layer and a sigmoid give 64 pixel values. Mean squared error against the clean image: ${fmt(mseClean, 4)}. Two numbers cannot store fine stroke details, so rebuilt digits look like smooth "typical" digits.${which === "dae" ? " The denoising AE was trained to turn noisy inputs into clean ones." : ""} Try the decoded map and click between clusters. A plain (non-variational) AE's decoder only trained at the codes of real images.`;
  }
}
