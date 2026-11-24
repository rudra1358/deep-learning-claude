"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  encode,
  forward,
  loadModel,
  maxAbsM,
  mulberry32,
  nextToken,
  rowNorm,
  softmaxT,
  topK,
  verifyModel,
  type CharModel,
  type ForwardTrace,
  type Matrix,
  type ModelJSON,
} from "@/lib/sim/transformer";

const MODEL_URL = "/models/char-transformer.json";
const DEFAULT_PROMPT = "alice was beginning to get very ";
const SUB = ["LayerNorm 1", "Q, K, V", "Attention weights", "Concat · W_O", "Residual add 1", "LayerNorm 2 + FFN", "Residual add 2"] as const;
const MAX_TEXT = 400;

type Stage = { kind: "tokens" } | { kind: "embed" } | { kind: "layer"; l: number; sub: number } | { kind: "output" };

function stageAt(step: number, nLayers: number): Stage {
  if (step === 0) return { kind: "tokens" };
  if (step === 1) return { kind: "embed" };
  const k = step - 2;
  if (k < nLayers * SUB.length) return { kind: "layer", l: Math.floor(k / SUB.length), sub: k % SUB.length };
  return { kind: "output" };
}

function stageName(st: Stage): string {
  if (st.kind === "tokens") return "Characters → token ids";
  if (st.kind === "embed") return "Embeddings + positions";
  if (st.kind === "output") return "Final LN → logits → softmax";
  return `Layer ${st.l + 1}: ${SUB[st.sub]}`;
}

const show = (c: string) => (c === " " ? "␣" : c);

/* ------------------------------------------------------------------ heatmap */

function Heatmap({
  title,
  shape,
  M,
  labels,
  mode = "signed",
  sel,
  onRow,
  colGroup,
  square = false,
}: {
  title: ReactNode;
  shape: string;
  M: Matrix;
  labels: string[];
  mode?: "signed" | "seq";
  sel?: number;
  onRow?: (i: number) => void;
  colGroup?: number;
  square?: boolean;
}) {
  const n = M.length;
  const c = M[0]?.length ?? 0;
  const labelW = 14;
  const cw = Math.max(2.5, Math.min(14, 560 / Math.max(1, c)));
  const rh = square ? cw : 9;
  const W = labelW + c * cw;
  const H = Math.max(rh, n * rh);
  const scale = mode === "seq" ? 1 : maxAbsM(M);
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline gap-x-2 text-xs">
        <span className="font-medium">{title}</span>
        <span className="font-mono text-faint">{shape}</span>
        <span className="font-mono text-faint">{mode === "seq" ? "0 … 1" : `colour scale ±${fmt(scale, 2)}`}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" style={{ maxWidth: Math.max(220, W * 1.15) }} role="img" aria-label={`${typeof title === "string" ? title : "matrix"} heatmap, ${shape}`}>
        <rect x={labelW} y={0} width={c * cw} height={H} fill="var(--surface)" />
        {M.map((row, i) => (
          <g key={i} onClick={onRow ? () => onRow(i) : undefined} className={onRow ? "cursor-pointer" : undefined}>
            <rect x={0} y={i * rh} width={labelW} height={rh} fill="transparent" />
            <text x={labelW - 2} y={i * rh + rh * 0.8} fontSize={Math.min(9, rh * 0.9)} textAnchor="end" fill={i === sel ? "var(--gold)" : "var(--text-muted)"} fontFamily="var(--font-mono)" fontWeight={i === sel ? 700 : 400}>
              {show(labels[i] ?? "")}
            </text>
            {row.map((v, j) =>
              Number.isFinite(v) ? (
                <rect
                  key={j}
                  x={labelW + j * cw}
                  y={i * rh}
                  width={Math.max(0.5, cw - 0.4)}
                  height={rh - 0.4}
                  fill={v >= 0 ? "var(--accent)" : "var(--clay)"}
                  fillOpacity={mode === "seq" ? Math.min(1, v) : Math.min(1, 0.08 + (0.92 * Math.abs(v)) / scale)}
                />
              ) : (
                <rect key={j} x={labelW + j * cw} y={i * rh} width={Math.max(0.5, cw - 0.4)} height={rh - 0.4} fill="var(--border)" fillOpacity={0.35} />
              ),
            )}
          </g>
        ))}
        {colGroup &&
          Array.from({ length: Math.floor(c / colGroup) - 1 }, (_, g) => (
            <line key={g} x1={labelW + (g + 1) * colGroup * cw} x2={labelW + (g + 1) * colGroup * cw} y1={0} y2={H} stroke="var(--gold)" strokeWidth={1} />
          ))}
        {sel !== undefined && sel < n && <rect x={labelW - 0.5} y={sel * rh - 0.5} width={c * cw + 0.5} height={rh + 0.5} fill="none" stroke="var(--gold)" strokeWidth={1.2} />}
      </svg>
    </figure>
  );
}

/** A single vector as a strip of numbers (first k entries). */
function VecPreview({ label, v, k = 8 }: { label: string; v: number[]; k?: number }) {
  return (
    <div className="min-w-0 font-mono text-[11px] leading-snug">
      <span className="text-muted">{label}</span> [{v.slice(0, k).map((x) => fmt(x, 2)).join(", ")}
      {v.length > k ? `, … (${v.length})` : ""}]
    </div>
  );
}

/* ------------------------------------------------------------------ norms bar chart */

function NormBars({ series, labels, sel, onRow }: { series: { name: string; color: string; values: number[] }[]; labels: string[]; sel: number; onRow: (i: number) => void }) {
  const n = labels.length;
  const W = 600;
  const H = 150;
  const pad = 24;
  const gw = (W - pad) / Math.max(1, n);
  const bw = Math.max(1, (gw * 0.8) / series.length);
  const max = Math.max(1e-9, ...series.flatMap((s) => s.values));
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap gap-x-3 text-xs">
        <span className="font-medium">Vector length ‖·‖₂ at every position</span>
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1 text-muted">
            <span className="inline-block h-2 w-3 rounded-sm" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H + 14}`} className="h-auto w-full" role="img" aria-label="Bar chart of vector norms per position">
        <line x1={pad} x2={W} y1={H} y2={H} stroke="var(--border-strong)" />
        <text x={pad - 4} y={10} fontSize={9} textAnchor="end" fill="var(--text-faint)">
          {max.toFixed(1)}
        </text>
        <text x={pad - 4} y={H} fontSize={9} textAnchor="end" fill="var(--text-faint)">
          0
        </text>
        {labels.map((c, i) => (
          <g key={i} onClick={() => onRow(i)} className="cursor-pointer">
            {i === sel && <rect x={pad + i * gw} y={0} width={gw} height={H} fill="var(--gold-soft)" />}
            {series.map((s, k) => {
              const h = (s.values[i] / max) * (H - 6);
              return <rect key={k} x={pad + i * gw + gw * 0.1 + k * bw} y={H - h} width={bw} height={h} fill={s.color} />;
            })}
            {gw > 7 && (
              <text x={pad + i * gw + gw / 2} y={H + 11} fontSize={Math.min(9, gw)} textAnchor="middle" fill={i === sel ? "var(--gold)" : "var(--text-muted)"} fontFamily="var(--font-mono)">
                {show(c)}
              </text>
            )}
          </g>
        ))}
      </svg>
    </figure>
  );
}

/* ------------------------------------------------------------------ probability bars */

function ProbBars({ probs, logits, vocab, k = 10, actual }: { probs: number[]; logits: number[]; vocab: string[]; k?: number; actual?: number }) {
  const idx = topK(probs, k);
  const W = 600;
  const rh = 20;
  const lab = 70;
  const H = idx.length * rh;
  const max = probs[idx[0]] || 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Top next-character probabilities">
      {idx.map((id, r) => {
        const w = (probs[id] / max) * (W - lab - 150);
        return (
          <g key={id}>
            <text x={lab - 8} y={r * rh + 14} fontSize={13} textAnchor="end" fill={id === actual ? "var(--gold)" : "var(--text)"} fontFamily="var(--font-mono)" fontWeight={r === 0 ? 700 : 400}>
              {`'${show(vocab[id])}'`}
            </text>
            <rect x={lab} y={r * rh + 3} width={Math.max(1, w)} height={rh - 6} rx={2} fill={id === actual ? "var(--gold)" : "var(--accent)"} fillOpacity={r === 0 ? 1 : 0.7} />
            <text x={lab + Math.max(1, w) + 6} y={r * rh + 14} fontSize={11} fill="var(--text-muted)" fontFamily="var(--font-mono)">
              {(probs[id] * 100).toFixed(1)}% · logit {logits[id].toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ main component */

export default function TransformerSim() {
  const [model, setModel] = useState<CharModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<{ maxAbsDiff: number; checked: number } | null>(null);
  const [input, setInput] = useState(DEFAULT_PROMPT);
  const [text, setText] = useState(DEFAULT_PROMPT);
  const [promptLen, setPromptLen] = useState(DEFAULT_PROMPT.length);
  const [head, setHead] = useState(0);
  const [posSel, setPosSel] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0.8);
  const [greedy, setGreedy] = useState(true);
  const [seed, setSeed] = useState(1);
  const [genPlaying, setGenPlaying] = useState(false);

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<ModelJSON>;
      })
      .then((j) => {
        if (!alive) return;
        const m = loadModel(j);
        setModel(m);
        setCheck(verifyModel(m));
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const nLayers = model?.layers.length ?? 2;
  const total = 2 + nLayers * SUB.length + 1;
  const s = useStepper(total, { baseMs: 2200 });
  const st = stageAt(s.step, nLayers);

  // Prompt validation against the vocabulary.
  const inputLower = input.toLowerCase();
  const unknown = model ? encode(model, inputLower).unknown : [];
  const inputError = !model ? null : inputLower.length === 0 ? "Type at least one character." : unknown.length ? `Not in the model's vocabulary: ${unknown.map((c) => `'${c}'`).join(" ")}` : null;
  useEffect(() => {
    if (model && !inputError) {
      setText(inputLower);
      setPromptLen(inputLower.length);
      setGenPlaying(false);
    }
  }, [model, inputLower, inputError]);

  const ids = useMemo(() => (model ? encode(model, text).ids : []), [model, text]);
  const trace: ForwardTrace | null = useMemo(() => (model && ids.length ? forward(model, ids) : null), [model, ids]);
  const n = trace?.ids.length ?? 0;
  const offset = text.length - n; // characters that fell out of the context window
  const pos = posSel === null || posSel >= n ? n - 1 : posSel;
  useEffect(() => setPosSel(null), [text]);

  const genStep = useCallback(() => {
    if (!model) return;
    setText((t) => {
      if (t.length >= MAX_TEXT) return t;
      const u = mulberry32(seed * 100003 + t.length)();
      return t + nextToken(model, encode(model, t).ids, { temperature, greedy, u }).char;
    });
  }, [model, seed, temperature, greedy]);

  useEffect(() => {
    if (!genPlaying) return;
    const id = setInterval(genStep, 450);
    return () => clearInterval(id);
  }, [genPlaying, genStep]);
  useEffect(() => {
    if (text.length >= MAX_TEXT) setGenPlaying(false);
  }, [text]);

  if (error) {
    return (
      <SimShell title="Transformer Explorer (trained model)" subtitle="A real character-level Transformer running in your browser.">
        <p className="text-sm text-clay">Could not load the model ({error}). Check your connection and reload the page.</p>
      </SimShell>
    );
  }
  if (!model || !trace) {
    return (
      <SimShell title="Transformer Explorer (trained model)" subtitle="A real character-level Transformer running in your browser.">
        <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
          Loading the trained model (≈300 KB of weights)…
        </div>
      </SimShell>
    );
  }

  const cfg = model.json.config;
  const labels = trace.chars;
  const L = st.kind === "layer" ? trace.layers[st.l] : null;
  const hd = L ? L.heads[head] : null;
  const lastLogits = trace.logits[pos];
  const probs = softmaxT(lastLogits, greedy ? 1 : temperature);
  const actualNext = pos < n - 1 ? trace.ids[pos + 1] : undefined;
  const dims = (k: number) => `${n} × ${k}`;

  const setLayer = (l: number) => {
    const sub = st.kind === "layer" ? st.sub : 0;
    s.setStep(2 + l * SUB.length + sub);
  };

  /* --------------------------- controls */
  const controls = (
    <div className="flex flex-col gap-2.5">
      <Controls s={s} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <label className="flex min-w-0 flex-1 basis-64 items-center gap-2">
          <span className="shrink-0 text-muted">Prompt</span>
          <input className="field min-w-0 flex-1 font-mono" value={input} maxLength={160} onChange={(e) => setInput(e.target.value)} aria-label="Prompt text" aria-invalid={!!inputError} />
        </label>
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Select layer">
          <span className="text-muted">Layer</span>
          {Array.from({ length: nLayers }, (_, l) => (
            <button key={l} className={`sim-btn ${st.kind === "layer" && st.l === l ? "primary" : ""}`} onClick={() => setLayer(l)} aria-pressed={st.kind === "layer" && st.l === l}>
              {l + 1}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Select attention head">
          <span className="text-muted">Head</span>
          {Array.from({ length: cfg.n_heads }, (_, h) => (
            <button key={h} className={`sim-btn ${h === head ? "primary" : ""}`} onClick={() => setHead(h)} aria-pressed={h === head}>
              {h + 1}
            </button>
          ))}
        </div>
      </div>
      {inputError && <p className="text-xs text-clay">{inputError} The vocabulary is: <span className="font-mono">{model.vocab.map(show).join("")}</span></p>}
    </div>
  );

  /* --------------------------- generation panel */
  const genPanel = (
    <div className="mb-4 rounded-lg border border-border p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="font-medium">Generate</span>
        <button className="sim-btn primary" onClick={() => setGenPlaying((p) => !p)} aria-label={genPlaying ? "Pause generation" : "Play generation"} disabled={text.length >= MAX_TEXT}>
          {genPlaying ? <Pause size={14} /> : <Play size={14} />} {genPlaying ? "Pause" : "Play"}
        </button>
        <button className="sim-btn" onClick={genStep} aria-label="Generate one character" disabled={genPlaying || text.length >= MAX_TEXT}>
          <SkipForward size={14} /> +1 char
        </button>
        <button
          className="sim-btn"
          onClick={() => {
            setGenPlaying(false);
            setText(text.slice(0, promptLen));
          }}
          aria-label="Remove generated characters"
        >
          <RotateCcw size={14} /> Back to prompt
        </button>
        <label className="flex items-center gap-1">
          <input type="radio" name="decode" checked={greedy} onChange={() => setGreedy(true)} /> greedy
        </label>
        <label className="flex items-center gap-1">
          <input type="radio" name="decode" checked={!greedy} onChange={() => setGreedy(false)} /> sample
        </label>
        {!greedy && (
          <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Change sampling seed">
            seed {seed}
          </button>
        )}
      </div>
      {!greedy && <Slider label="Temperature T" value={temperature} min={0.1} max={2} step={0.05} onChange={setTemperature} />}
      <p className="mt-2 break-words font-mono text-sm leading-relaxed">
        <span>{text.slice(0, promptLen)}</span>
        <span className="text-accent">{text.slice(promptLen)}</span>
        <span className="animate-pulse text-gold">▍</span>
      </p>
      <p className="mt-1 text-[11px] text-muted">
        {text.length - promptLen} generated · {greedy ? "greedy: always the arg-max character" : `sampling from softmax(logits / ${temperature.toFixed(2)}), seed ${seed}`} · each new character re-runs the full forward pass below
        {offset > 0 ? ` · the context window holds the last ${cfg.context} characters (${offset} dropped)` : ""}.
      </p>
    </div>
  );

  /* --------------------------- stage body */
  let body: ReactNode = null;
  if (st.kind === "tokens") {
    body = (
      <div>
        <p className="mb-2 text-xs text-muted">One token per character. Click a character to select its position (used by every later step).</p>
        <div className="flex flex-wrap gap-1">
          {trace.chars.map((c, i) => (
            <button key={i} onClick={() => setPosSel(i)} className="rounded border px-1.5 py-1 text-center" style={{ borderColor: i === pos ? "var(--gold)" : "var(--border-strong)", background: i === pos ? "var(--gold-soft)" : "var(--surface)" }} aria-pressed={i === pos} aria-label={`Position ${i}, character ${show(c)}, id ${trace.ids[i]}`}>
              <div className="font-mono text-sm leading-none">{show(c)}</div>
              <div className="font-mono text-[9px] text-muted">{trace.ids[i]}</div>
            </button>
          ))}
        </div>
        <p className="mt-3 font-mono text-[11px] text-muted">
          vocabulary ({model.vocab.length}): {model.vocab.map((c, i) => `${show(c)}=${i}`).join(" ")}
        </p>
      </div>
    );
  } else if (st.kind === "embed") {
    body = (
      <div className="grid gap-4">
        <div className="grid gap-4 md:grid-cols-2">
          <Heatmap title="Token embeddings E[id]" shape={dims(cfg.d_model)} M={trace.tokEmb} labels={labels} sel={pos} onRow={setPosSel} />
          <Heatmap title="Learned position embeddings P[i]" shape={dims(cfg.d_model)} M={trace.posEmb} labels={labels} sel={pos} onRow={setPosSel} />
        </div>
        <Heatmap title="Input x₀ = E[id] + P[i]" shape={dims(cfg.d_model)} M={trace.x0} labels={labels} sel={pos} onRow={setPosSel} />
        <div className="grid gap-1">
          <VecPreview label={`E['${show(labels[pos])}'] =`} v={trace.tokEmb[pos]} />
          <VecPreview label={`P[${pos}] =`} v={trace.posEmb[pos]} />
          <VecPreview label={`x₀[${pos}] =`} v={trace.x0[pos]} />
        </div>
      </div>
    );
  } else if (st.kind === "layer" && L && hd) {
    const l = st.l;
    if (st.sub === 0) {
      body = (
        <div className="grid gap-4">
          <Heatmap title={`Residual stream entering layer ${l + 1}`} shape={dims(cfg.d_model)} M={L.xIn} labels={labels} sel={pos} onRow={setPosSel} />
          <Heatmap title="LN₁(x): each row normalised to mean 0, variance 1, then × γ + β" shape={dims(cfg.d_model)} M={L.ln1.out} labels={labels} sel={pos} onRow={setPosSel} />
          <div className="grid gap-1">
            <VecPreview label={`x[${pos}] =`} v={L.xIn[pos]} />
            <VecPreview label={`LN₁(x)[${pos}] =`} v={L.ln1.out[pos]} />
            <div className="font-mono text-[11px] text-muted">
              row {pos}: mean μ = {fmt(L.ln1.mean[pos], 4)}, std σ = {fmt(L.ln1.std[pos], 4)}
            </div>
          </div>
        </div>
      );
    } else if (st.sub === 1) {
      body = (
        <div className="grid gap-4">
          <p className="text-xs text-muted">
            Q = LN₁(x)·W_Q + b_Q (and likewise K, V) are {dims(cfg.d_model)}; head {head + 1} uses columns {head * model.dHead}…{(head + 1) * model.dHead - 1}.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <Heatmap title={`Q (head ${head + 1})`} shape={dims(model.dHead)} M={hd.Q} labels={labels} sel={pos} onRow={setPosSel} />
            <Heatmap title={`K (head ${head + 1})`} shape={dims(model.dHead)} M={hd.K} labels={labels} sel={pos} onRow={setPosSel} />
            <Heatmap title={`V (head ${head + 1})`} shape={dims(model.dHead)} M={hd.V} labels={labels} sel={pos} onRow={setPosSel} />
          </div>
          <VecPreview label={`q[${pos}] =`} v={hd.Q[pos]} k={12} />
        </div>
      );
    } else if (st.sub === 2) {
      const row = hd.weights[pos];
      const best = topK(row, Math.min(6, pos + 1));
      body = (
        <div className="grid gap-4">
          <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Heatmap title={`softmax(QKᵀ/√${model.dHead} + causal mask), head ${head + 1}`} shape={`${n} × ${n} (rows = queries)`} M={hd.weights} labels={labels} mode="seq" sel={pos} onRow={setPosSel} square />
            <div className="min-w-0">
              <div className="mb-1 text-xs font-medium">
                Query position {pos} <span className="font-mono text-gold">'{show(labels[pos])}'</span> attends to:
              </div>
              <p className="break-words font-mono text-sm leading-relaxed" aria-label="Context coloured by attention weight">
                {labels.map((c, j) => (
                  <span key={j} className="rounded-[2px]" style={{ background: j <= pos ? `color-mix(in srgb, var(--accent) ${Math.round(row[j] * 100)}%, transparent)` : undefined, color: j > pos ? "var(--text-faint)" : undefined, outline: j === pos ? "1px solid var(--gold)" : undefined }}>
                    {c === " " ? " " : c}
                  </span>
                ))}
              </p>
              <table className="mt-2 font-mono text-[11px] tabular-nums">
                <tbody>
                  {best.map((j) => (
                    <tr key={j}>
                      <td className="pr-2 text-muted">pos {j}</td>
                      <td className="pr-2">'{show(labels[j])}'</td>
                      <td className="pr-2 text-right">score {fmt(hd.scores[pos][j], 2)}</td>
                      <td className="text-right text-accent">α = {row[j].toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1 text-[11px] text-muted">Grey characters are in the future: masked to −∞ before the softmax, so they receive weight exactly 0. Row sum = {row.reduce((a, b) => a + b, 0).toFixed(4)}.</p>
            </div>
          </div>
          <div>
            <div className="mb-1 text-xs font-medium">All heads of layer {l + 1} (click to select)</div>
            <div className="flex flex-wrap gap-2">
              {L.heads.map((h, k) => (
                <button key={k} onClick={() => setHead(k)} className="rounded-md border p-1" style={{ borderColor: k === head ? "var(--gold)" : "var(--border)" }} aria-label={`Show head ${k + 1}`} aria-pressed={k === head}>
                  <svg viewBox={`0 0 ${n} ${n}`} width={64} height={64} role="img" aria-label={`Head ${k + 1} attention pattern`}>
                    <rect width={n} height={n} fill="var(--surface)" />
                    {h.weights.map((r, i) => r.map((w, j) => (j <= i && w > 0.02 ? <rect key={`${i}-${j}`} x={j} y={i} width={1} height={1} fill="var(--accent)" fillOpacity={w} /> : null)))}
                  </svg>
                  <div className="text-center text-[10px] text-muted">head {k + 1}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      );
    } else if (st.sub === 3) {
      body = (
        <div className="grid gap-4">
          <Heatmap title="Concat(head₁ … head₄) = per-head Σⱼ αᵢⱼ vⱼ side by side" shape={`${n} × ${cfg.n_heads}·${model.dHead}`} M={L.concat} labels={labels} sel={pos} onRow={setPosSel} colGroup={model.dHead} />
          <Heatmap title="Attention output = Concat · W_O + b_O" shape={dims(cfg.d_model)} M={L.attnOut} labels={labels} sel={pos} onRow={setPosSel} />
        </div>
      );
    } else if (st.sub === 4) {
      body = (
        <div className="grid gap-4">
          <NormBars
            labels={labels}
            sel={pos}
            onRow={setPosSel}
            series={[
              { name: "x (in)", color: "var(--text-faint)", values: L.xIn.map(rowNorm) },
              { name: "attention output", color: "var(--gold)", values: L.attnOut.map(rowNorm) },
              { name: "x + attention", color: "var(--accent)", values: L.xMid.map(rowNorm) },
            ]}
          />
          <Heatmap title="Residual stream after attention: x + MHA(LN₁(x))" shape={dims(cfg.d_model)} M={L.xMid} labels={labels} sel={pos} onRow={setPosSel} />
        </div>
      );
    } else if (st.sub === 5) {
      const act = L.ffPre[pos].filter((v) => v > 0).length / model.dFF;
      const allAct = L.ffPre.flat().filter((v) => v > 0).length / (n * model.dFF);
      body = (
        <div className="grid gap-4">
          <Heatmap title="LN₂(x)" shape={dims(cfg.d_model)} M={L.ln2.out} labels={labels} sel={pos} onRow={setPosSel} />
          <Heatmap title={`FFN hidden layer GELU(LN₂(x)·W₁ + b₁)`} shape={dims(model.dFF)} M={L.ffAct} labels={labels} sel={pos} onRow={setPosSel} />
          <div className="font-mono text-[11px] text-muted">
            fraction of the {model.dFF} hidden units with positive pre-activation: position {pos} → {(act * 100).toFixed(1)}%, all positions → {(allAct * 100).toFixed(1)}%. Strongest units at position {pos}: {topK(L.ffAct[pos], 5).map((u) => `#${u} (${L.ffAct[pos][u].toFixed(2)})`).join(", ")}
          </div>
        </div>
      );
    } else {
      body = (
        <div className="grid gap-4">
          <Heatmap title="FFN output W₂·GELU(…) + b₂" shape={dims(cfg.d_model)} M={L.ffOut} labels={labels} sel={pos} onRow={setPosSel} />
          <NormBars
            labels={labels}
            sel={pos}
            onRow={setPosSel}
            series={[
              { name: "x (after attention)", color: "var(--text-faint)", values: L.xMid.map(rowNorm) },
              { name: "FFN output", color: "var(--gold)", values: L.ffOut.map(rowNorm) },
              { name: "x + FFN", color: "var(--accent)", values: L.xOut.map(rowNorm) },
            ]}
          />
        </div>
      );
    }
  } else {
    body = (
      <div className="grid gap-4">
        <Heatmap title="Final LayerNorm LN_f(x)" shape={dims(cfg.d_model)} M={trace.lnF.out} labels={labels} sel={pos} onRow={setPosSel} />
        <div>
          <div className="mb-1 text-xs font-medium">
            Next-character distribution after position {pos} (<span className="font-mono text-gold">'{show(labels[pos])}'</span>): softmax(logits{greedy ? "" : ` / ${temperature.toFixed(2)}`}), top 10 of {model.vocab.length}
          </div>
          <ProbBars probs={probs} logits={lastLogits} vocab={model.vocab} actual={actualNext} />
          {actualNext !== undefined && (
            <p className="mt-1 text-[11px] text-muted">
              The character that actually follows is <span className="font-mono text-gold">'{show(model.vocab[actualNext])}'</span> with probability {(probs[actualNext] * 100).toFixed(1)}% — cross-entropy loss −ln p = {(-Math.log(Math.max(probs[actualNext], 1e-12))).toFixed(3)} at this position.
            </p>
          )}
        </div>
      </div>
    );
  }

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={`${s.step + 1}. ${stageName(st)}`}>{narrate(st, trace, model, pos, head, probs)}</Narration>
      <div className="rounded-lg border border-border p-2.5 text-[11px] leading-snug text-muted">
        <span className="font-medium text-accent">A real trained model.</span> {model.nParams.toLocaleString()} parameters (computed from the loaded weights): {cfg.n_layers} pre-LN decoder layers, d_model = {cfg.d_model}, {cfg.n_heads} heads of {model.dHead}, d_ff = {cfg.d_ff}, learned positions, {cfg.context}-character context, GELU, weight-tied output. Trained with PyTorch for {model.json.training.steps} steps on {model.json.training.train_chars.toLocaleString()} characters of Chapter I of Lewis Carroll&apos;s <i>Alice&apos;s Adventures in Wonderland</i> (public domain); train loss {model.json.training.final_train_loss}, held-out loss {model.json.training.final_val_loss} nats/char. It has memorised much of that text and knows little else.
      </div>
      {check && (
        <div className="rounded-lg border border-border p-2.5 font-mono text-[11px] leading-snug text-muted">
          browser vs PyTorch: max |Δlogit| = {check.maxAbsDiff.toExponential(1)} over {check.checked} logits {check.maxAbsDiff < 1e-3 ? "✓" : "✗"}
        </div>
      )}
    </div>
  );

  return (
    <SimShell title="Transformer Explorer (trained model)" subtitle="Follow your prompt through a real, trained decoder-only Transformer — every number is computed live in your browser." controls={controls} aside={aside}>
      {genPanel}
      <nav className="mb-4 flex flex-wrap gap-1" aria-label="Stages">
        {Array.from({ length: total }, (_, k) => {
          const sk = stageAt(k, nLayers);
          const short = sk.kind === "tokens" ? "Tokens" : sk.kind === "embed" ? "Embed" : sk.kind === "output" ? "Logits" : `L${sk.l + 1}·${["LN1", "QKV", "Attn", "W_O", "Add", "FFN", "Add"][sk.sub]}`;
          return (
            <button
              key={k}
              onClick={() => s.setStep(k)}
              className="rounded-md border px-1.5 py-0.5 text-[11px]"
              style={{ borderColor: k === s.step ? "var(--accent)" : "var(--border)", background: k === s.step ? "var(--accent-soft)" : "transparent", color: k === s.step ? "var(--text)" : "var(--text-muted)" }}
              aria-current={k === s.step ? "step" : undefined}
              aria-label={stageName(sk)}
            >
              {short}
            </button>
          );
        })}
      </nav>
      {body}
    </SimShell>
  );
}

/* ------------------------------------------------------------------ narration */

function narrate(st: Stage, tr: ForwardTrace, model: CharModel, pos: number, head: number, probs: number[]): ReactNode {
  const n = tr.ids.length;
  const c = `'${show(tr.chars[pos])}'`;
  const d = model.d;
  if (st.kind === "tokens")
    return `The text is split into ${n} characters and each is replaced by its index in a ${model.vocab.length}-character vocabulary. The model never sees letters — only these integers. Position ${pos} is ${c}, id ${tr.ids[pos]}.`;
  if (st.kind === "embed")
    return `Each id selects a row of the ${model.vocab.length} × ${d} embedding table E, and position i adds row i of the learned ${model.context} × ${d} position table P. Without P, attention could not tell "ab" from "ba". Row ${pos}: ‖E‖ = ${rowNorm(tr.tokEmb[pos]).toFixed(2)}, ‖P‖ = ${rowNorm(tr.posEmb[pos]).toFixed(2)}, ‖x₀‖ = ${rowNorm(tr.x0[pos]).toFixed(2)}.`;
  if (st.kind === "output") {
    const best = topK(probs, 1)[0];
    return `The final LayerNorm is applied, then every row is multiplied by Eᵀ (the output head shares the embedding matrix — weight tying), giving ${model.vocab.length} logits per position. Softmax turns them into probabilities. After ${c} the model's top choice is '${show(model.vocab[best])}' with p = ${probs[best].toFixed(3)}. Training made these predictions good at every position simultaneously (teacher forcing).`;
  }
  const L = tr.layers[st.l];
  const hd = L.heads[head];
  switch (st.sub) {
    case 0:
      return `Pre-LN: before attention, each position's vector is normalised on its own (no mixing across positions): subtract its mean ${fmt(L.ln1.mean[pos], 3)}, divide by its std ${fmt(L.ln1.std[pos], 3)}, then scale by γ and shift by β (${d} learned values each). The residual stream x itself is left untouched and carried around the sublayer.`;
    case 1:
      return `Three linear maps turn LN₁(x) into queries, keys and values (${d} → ${d} each, with biases). The ${d} columns are split into ${model.nHeads} heads of ${model.dHead}. Head ${head + 1}'s query for ${c} has length ‖q‖ = ${rowNorm(hd.Q[pos]).toFixed(2)}; its dot products with the keys decide where it looks.`;
    case 2: {
      const row = hd.weights[pos];
      const j = topK(row, 1)[0];
      return `Scores qᵢ·kⱼ/√${model.dHead} are computed for j ≤ i only (causal mask), and softmax turns each row into weights summing to 1. In layer ${st.l + 1}, head ${head + 1}, position ${pos} ${c} puts its largest weight ${row[j].toFixed(3)} on position ${j} '${show(tr.chars[j])}'${j === pos ? " (itself)" : j === pos - 1 ? " (the previous character)" : ""}. Click rows of the heatmap or switch heads to compare patterns.`;
    }
    case 3:
      return `Each head returns a ${model.dHead}-dimensional weighted average of its value vectors. The ${model.nHeads} results are concatenated back to ${d} dimensions and mixed by W_O (${d} × ${d}) — the only place where information from different heads is combined.`;
    case 4:
      return `Residual connection: the attention output is ADDED to the stream, x ← x + MHA(LN₁(x)). At position ${pos}: ‖x‖ = ${rowNorm(L.xIn[pos]).toFixed(2)}, ‖attention‖ = ${rowNorm(L.attnOut[pos]).toFixed(2)}, ‖sum‖ = ${rowNorm(L.xMid[pos]).toFixed(2)}. Sublayers write small updates into a shared "residual stream" rather than replacing it.`;
    case 5: {
      const act = L.ffPre[pos].filter((v) => v > 0).length;
      return `The position-wise feed-forward network expands each vector ${d} → ${model.dFF}, applies GELU, and projects back. It acts on every position separately (no communication between positions) and holds two-thirds of each layer's parameters. At ${c}, ${act} of ${model.dFF} hidden units are positive.`;
    }
    default:
      return `Second residual add: x ← x + FFN(LN₂(x)). At position ${pos}: ‖x‖ = ${rowNorm(L.xMid[pos]).toFixed(2)}, ‖FFN‖ = ${rowNorm(L.ffOut[pos]).toFixed(2)}, ‖sum‖ = ${rowNorm(L.xOut[pos]).toFixed(2)}. ${st.l + 1 < model.layers.length ? `This is the input to layer ${st.l + 2}.` : "This goes to the final LayerNorm and the output head."}`;
  }
}
