"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, fmt, seqColor, signedColor, useStepper } from "./SimShell";
import { greedyDecode, loadModel, validateInput, type Seq2SeqJson, type Seq2SeqModel, type Seq2SeqTrace } from "@/lib/sim/seq2seq";

const MODEL_URL = "/models/seq2seq-attention.json";

const EXAMPLES = ["attention", "bahdanau", "encoder", "alignment", "abcdefghijklmnopqrst", "sequence", "xyz"];

/**
 * Exact-match accuracy (greedy decoding) by input length, measured by web/scripts/train-seq2seq-attention.py:
 * 200 fresh random strings per length; training used lengths 1–20 only. Same architecture, same seed,
 * same 6000 training steps — the only difference is whether the decoder gets a step-dependent context c_i
 * (attention) or one fixed context vector c.
 */
const MEASURED = {
  lengths: Array.from({ length: 25 }, (_, k) => k + 1),
  attention: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 0.905, 0.615, 0.275],
  noAttention: [1.0, 1.0, 1.0, 0.99, 0.995, 0.935, 0.88, 0.635, 0.44, 0.305, 0.16, 0.11, 0.035, 0.015, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  samplesPerLength: 200,
  trainMaxLen: 20,
};

const vecStr = (v: number[], d = 2, n = 4) => "[" + v.slice(0, n).map((x) => fmt(x, d)).join(", ") + (v.length > n ? ", …" : "") + "]";
const show = (sym: string) => (sym === "<eos>" ? "⟨eos⟩" : sym === "<sos>" ? "⟨sos⟩" : sym === "<pad>" ? "⟨pad⟩" : sym);

/* ------------------------------------------------------------------ vector strip */

function Strip({ label, v, y, x0, cw, maxAbs = 1, active }: { label: string; v: number[]; y: number; x0: number; cw: number; maxAbs?: number; active?: boolean }) {
  return (
    <g>
      <text x={x0 - 6} y={y + 10} textAnchor="end" fontSize={11} fill={active ? "var(--text)" : "var(--text-muted)"}>
        {label}
      </text>
      {v.map((x, k) => (
        <rect key={k} x={x0 + k * cw} y={y} width={cw - 1} height={14} fill={signedColor(x, maxAbs)} stroke={active ? "var(--gold)" : "none"} strokeWidth={active ? 0.4 : 0} />
      ))}
    </g>
  );
}

/* ------------------------------------------------------------------ accuracy chart */

function AccuracyChart() {
  const W = 560, H = 230, L = 44, R = 12, T = 14, B = 40;
  const n = MEASURED.lengths.length;
  const xs = (len: number) => L + ((len - 1) / (n - 1)) * (W - L - R);
  const ys = (a: number) => T + (1 - a) * (H - T - B);
  const series = [
    { name: "with attention", data: MEASURED.attention, color: "var(--accent)", dash: undefined as string | undefined },
    { name: "fixed context vector (no attention)", data: MEASURED.noAttention, color: "var(--clay)", dash: "5 3" },
  ];
  const xTrain = xs(MEASURED.trainMaxLen + 0.5);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Measured exact-match accuracy versus input length, with and without attention">
      <rect x={xTrain} y={T} width={W - R - xTrain} height={H - T - B} fill="var(--surface-2)" />
      <text x={W - R - 4} y={T + 12} textAnchor="end" fontSize={10} fill="var(--text-muted)">
        longer than any
      </text>
      <text x={W - R - 4} y={T + 24} textAnchor="end" fontSize={10} fill="var(--text-muted)">
        training string
      </text>
      {[0, 0.25, 0.5, 0.75, 1].map((a) => (
        <g key={a}>
          <line x1={L} x2={W - R} y1={ys(a)} y2={ys(a)} stroke="var(--border)" strokeWidth={0.6} />
          <text x={L - 6} y={ys(a) + 4} textAnchor="end" fontSize={10} fill="var(--text-muted)">
            {Math.round(a * 100)}%
          </text>
        </g>
      ))}
      {MEASURED.lengths.filter((l) => l === 1 || l % 5 === 0).map((l) => (
        <text key={l} x={xs(l)} y={H - B + 14} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
          {l}
        </text>
      ))}
      <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
        input length (letters)
      </text>
      {series.map((s) => (
        <g key={s.name}>
          <polyline points={s.data.map((a, k) => `${xs(k + 1)},${ys(a)}`).join(" ")} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash} />
          {s.data.map((a, k) => (
            <circle key={k} cx={xs(k + 1)} cy={ys(a)} r={2.6} fill={s.color}>
              <title>{`length ${k + 1}: ${(a * 100).toFixed(1)}% (${s.name})`}</title>
            </circle>
          ))}
        </g>
      ))}
    </svg>
  );
}

/* ------------------------------------------------------------------ main component */

export default function Seq2SeqAttentionSim() {
  const [model, setModel] = useState<Seq2SeqModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState("attention");
  const [exIdx, setExIdx] = useState(0);

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<Seq2SeqJson>;
      })
      .then((j) => alive && setModel(loadModel(j)))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const inputError = model ? validateInput(model, text) : null;
  const [validText, setValidText] = useState("attention");
  useEffect(() => {
    if (model && !inputError) setValidText(text);
  }, [model, text, inputError]);

  const trace: Seq2SeqTrace | null = useMemo(() => (model ? greedyDecode(model, validText) : null), [model, validText]);

  const total = trace ? trace.steps.length + 1 : 1;
  const s = useStepper(total, { baseMs: 1300 });
  const { reset, setStep } = s;
  useEffect(() => {
    reset();
  }, [validText, reset]);

  const nextExample = () => {
    const k = (exIdx + 1) % EXAMPLES.length;
    setExIdx(k);
    setText(EXAMPLES[k]);
  };
  const randomString = () => {
    const len = 4 + Math.floor(Math.random() * 15);
    setText(Array.from({ length: len }, () => String.fromCharCode(97 + Math.floor(Math.random() * 26))).join(""));
  };

  const subtitle =
    "A real trained model. It learned (in PyTorch) to reverse lowercase strings of 1–20 letters. A two-way GRU encoder reads the input; a GRU decoder writes the output, using additive (Bahdanau) attention to look back. Every number below is computed live in your browser.";

  if (!model || !trace) {
    return (
      <SimShell title="Seq2Seq with Additive Attention (trained model)" subtitle={subtitle}>
        {error ? (
          <div className="text-sm text-clay" role="alert">
            Couldn&apos;t load the trained model ({MODEL_URL}): {error}. You can still see the measured accuracy results below.
          </div>
        ) : (
          <div className="text-sm text-muted" role="status">
            Loading trained weights (~180 KB)…
          </div>
        )}
        <div className="mt-4 rounded-lg border border-border p-3">
          <div className="text-xs font-medium">Measured: share of strings reversed perfectly, by input length</div>
          <AccuracyChart />
        </div>
      </SimShell>
    );
  }

  const T = trace.input.length;
  const step = Math.min(s.step, total - 1);
  const cur = step >= 1 ? trace.steps[step - 1] : null;
  const decoded = trace.steps.slice(0, step);
  const chars = [...trace.input];

  /* ---------- geometry of the input panel (α bars, chars, e_j, encoder heatmap) ---------- */
  const cw = 26, lx = 64;
  const inW = Math.max(360, lx + T * cw + 10);
  const barTop = 18, barH = 70;
  const charY = barTop + barH + 16;
  const eY = charY + 16;
  const hmTop = eY + 14;
  const D = trace.enc.H[0].length;
  const cellH = 4;
  const inH = hmTop + D * cellH + 22;
  const maxE = cur ? Math.max(1e-9, ...cur.e.map(Math.abs)) : 1;
  const argmaxJ = cur ? cur.alpha.indexOf(Math.max(...cur.alpha)) : -1;

  /* ---------- alignment matrix geometry ---------- */
  const rowsN = trace.steps.length;
  const ac = 22, al = 40, at = 18;
  const amW = Math.max(300, al + T * ac + 8), amH = at + rowsN * ac + 6;

  /* ---------- vector strip geometry ---------- */
  const scw = 9, sx0 = 92;
  const stripW = sx0 + D * scw + 8;

  /* ---------- narration ---------- */
  let nTitle: string, nBody: ReactNode;
  if (!cur) {
    nTitle = `Step 0 — encode "${trace.input}"`;
    nBody = (
      <>
        <p className="mb-2">
          One GRU reads the {T} letters left→right, another reads right→left. Each position j gets a summary h_j = [→h_j ; ←h_j] of {D} numbers (the columns of the heatmap). W_h h_j is worked out once per input and reused at every output step.
        </p>
        <div className="font-mono text-[11px] text-text">s₀ = tanh(W_init ←h₁ + b) = {vecStr(trace.enc.s0)}</div>
      </>
    );
  } else {
    const prevSym = cur.i === 1 ? "⟨sos⟩" : show(trace.steps[cur.i - 2].sym);
    const target = cur.i <= T ? chars[T - cur.i] : "⟨eos⟩";
    nTitle = `Step ${cur.i} — output ${cur.i}: “${show(cur.sym)}”`;
    nBody = (
      <>
        <p className="mb-2">
          The decoder gives every input position a score, e_j = vᵀtanh(W_s s_{cur.i - 1} + W_h h_j). Softmax turns scores into weights α. The largest weight, α = {fmt(cur.alpha[argmaxJ], 3)}, falls on position {argmaxJ + 1} (“{chars[argmaxJ]}”)
          {cur.i <= T ? `. To reverse the string, output ${cur.i} should copy input ${T + 1 - cur.i} (“${chars[T - cur.i]}”).` : ". All letters are out, so the model should now output ⟨eos⟩ (end)."}
        </p>
        <div className="space-y-0.5 font-mono text-[11px] leading-relaxed text-text">
          <div>e = [{cur.e.slice(0, 6).map((x) => fmt(x, 2)).join(", ")}{T > 6 ? ", …" : ""}]</div>
          <div>α = softmax(e), Σα = {fmt(cur.alpha.reduce((a, b) => a + b, 0), 3)}</div>
          <div>c = Σ α_j h_j = {vecStr(cur.context)}</div>
          <div>
            s_{cur.i} = GRU([emb({prevSym}) ; c], s_{cur.i - 1})
          </div>
          <div>
            p(“{show(cur.sym)}”) = {fmt(cur.top[0].p, 4)} {cur.sym === (cur.i <= T ? target : "<eos>") || show(cur.sym) === target ? "✓" : `✗ (target “${target}”)`}
          </div>
        </div>
      </>
    );
  }

  return (
    <SimShell
      title="Seq2Seq with Additive Attention (trained model)"
      subtitle={subtitle}
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex min-w-0 flex-1 items-center gap-2 text-xs">
              <span className="shrink-0 text-muted">Input</span>
              <input
                className="min-w-0 flex-1 rounded border border-border bg-surface px-2 py-1 font-mono text-xs"
                value={text}
                maxLength={30}
                onChange={(e) => setText(e.target.value.toLowerCase())}
                aria-label="Input string of lowercase letters to reverse"
                aria-invalid={!!inputError}
              />
            </label>
            <button className="sim-btn" onClick={nextExample} aria-label="Load next example input">
              Example
            </button>
            <button className="sim-btn" onClick={randomString} aria-label="Load a random input string">
              Random
            </button>
          </div>
          {inputError && (
            <div className="text-xs text-clay" role="alert">
              {inputError} Showing “{validText}”.
            </div>
          )}
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={nTitle}>{nBody}</Narration>
          <div className="rounded-lg border border-border p-2 font-mono text-xs">
            <div className="text-muted">input → output so far</div>
            <div className="break-all text-text">{trace.input}</div>
            <div className="break-all">
              {decoded.map((d, k) => (
                <span key={k} style={k === step - 1 ? { color: "var(--gold)", fontWeight: 600 } : undefined}>
                  {d.token === model.cfg.eos ? "⟨eos⟩" : d.sym}
                </span>
              ))}
              {step === 0 && <span className="text-faint">(nothing decoded yet)</span>}
            </div>
            {step === total - 1 && (
              <div className={trace.correct ? "mt-1 text-accent" : "mt-1 text-clay"}>
                {trace.correct ? "✓ perfectly reversed" : `✗ not quite reversed (expected ${[...trace.input].reverse().join("")})`}
              </div>
            )}
          </div>
          <div className="text-xs text-muted">
            Try this: a 20-letter string (the longest it trained on), or one with repeated letters like “aaaabaaaa”. A 21+ letter string is rejected because it is longer than anything in training. The accuracy chart shows what happens there.
          </div>
        </div>
      }
    >
      {/* ---------- input positions: α bars, letters, e_j, encoder heatmap ---------- */}
      <div className="text-xs text-muted">
        {cur ? `Where output step ${cur.i} looks among the ${T} input positions (bars = weights α_j, numbers under letters = scores e_j), plus the encoder summaries h_j (heatmap, ${D} rows).` : `Encoder summaries h_j for the ${T} input positions (heatmap: rows = ${D} dimensions, accent positive / clay negative).`}
      </div>
      <svg viewBox={`0 0 ${inW} ${inH}`} className="h-auto w-full" style={{ maxWidth: inW * 1.4 }} role="img" aria-label="Attention weights over input letters and encoder hidden-state heatmap">
        <text x={lx - 6} y={barTop + 10} textAnchor="end" fontSize={10} fill="var(--text-muted)">α_j</text>
        <line x1={lx} x2={lx + T * cw} y1={barTop + barH} y2={barTop + barH} stroke="var(--border)" />
        {chars.map((c, j) => {
          const a = cur ? cur.alpha[j] : 0;
          const x = lx + j * cw;
          const isMax = j === argmaxJ;
          return (
            <g key={j}>
              {cur && <rect x={x + 3} y={barTop + barH - a * barH} width={cw - 6} height={Math.max(0.5, a * barH)} fill={isMax ? "var(--gold)" : "var(--accent)"} />}
              {cur && a >= 0.005 && (
                <text x={x + cw / 2} y={barTop + barH - a * barH - 3} textAnchor="middle" fontSize={8} fill="var(--text)">
                  {a.toFixed(2)}
                </text>
              )}
              <text x={x + cw / 2} y={charY} textAnchor="middle" fontSize={13} fontFamily="monospace" fill={isMax ? "var(--gold)" : "var(--text)"} fontWeight={isMax ? 700 : 400}>
                {c}
              </text>
              {cur && (
                <text x={x + cw / 2} y={eY} textAnchor="middle" fontSize={7.5} fill={cur.e[j] >= 0 ? "var(--accent)" : "var(--clay)"}>
                  {fmt(cur.e[j], 1)}
                </text>
              )}
              {trace.enc.H[j].map((v, d) => (
                <rect key={d} x={x + 2} y={hmTop + d * cellH} width={cw - 4} height={cellH} fill={signedColor(v, 1)} />
              ))}
              {isMax && <rect x={x + 1} y={hmTop - 1} width={cw - 2} height={D * cellH + 2} fill="none" stroke="var(--gold)" strokeWidth={1.5} />}
              <text x={x + cw / 2} y={hmTop + D * cellH + 12} textAnchor="middle" fontSize={8} fill="var(--text-muted)">
                {j + 1}
              </text>
            </g>
          );
        })}
        <text x={lx - 6} y={charY} textAnchor="end" fontSize={10} fill="var(--text-muted)">x_j</text>
        {cur && <text x={lx - 6} y={eY} textAnchor="end" fontSize={10} fill="var(--text-muted)">e_j</text>}
        <text x={lx - 6} y={hmTop + 10} textAnchor="end" fontSize={10} fill="var(--text-muted)">→h_j</text>
        <text x={lx - 6} y={hmTop + (D / 2) * cellH + 10} textAnchor="end" fontSize={10} fill="var(--text-muted)">←h_j</text>
        <line x1={lx} x2={lx + T * cw} y1={hmTop + (D / 2) * cellH} y2={hmTop + (D / 2) * cellH} stroke="var(--surface)" strokeWidth={1.2} />
      </svg>
      {cur && (
        <div className="mt-1 font-mono text-[11px] text-muted">
          max |e_j| = {fmt(maxE, 2)} · {cur.i <= T ? `α at the correct position (${T + 1 - cur.i}) = ${fmt(cur.alpha[T - cur.i], 3)}` : "⟨eos⟩ step: nothing to copy"}
        </div>
      )}

      {/* ---------- decoder vectors ---------- */}
      <div className="mt-4 text-xs text-muted">Decoder at this step (each cell is one dimension; accent positive, clay negative).</div>
      <svg viewBox={`0 0 ${stripW} 96`} className="h-auto w-full" role="img" aria-label="Decoder state, query projection, context vector and new decoder state">
        {cur ? (
          <>
            <Strip label="s_(i−1)" v={cur.sPrev} y={4} x0={sx0} cw={scw} />
            <Strip label="W_s s_(i−1)" v={cur.query} y={26} x0={sx0} cw={scw} maxAbs={Math.max(1e-9, ...cur.query.map(Math.abs))} />
            <Strip label="context c_i" v={cur.context} y={48} x0={sx0} cw={scw} active />
            <Strip label="s_i" v={cur.s} y={70} x0={sx0} cw={scw} />
          </>
        ) : (
          <>
            <Strip label="s_0" v={trace.enc.s0} y={4} x0={sx0} cw={scw} active />
            <text x={sx0} y={50} fontSize={11} fill="var(--text-muted)">Press Step to make the first output.</text>
          </>
        )}
      </svg>

      {/* ---------- output distribution ---------- */}
      {cur && (
        <div className="mt-3">
          <div className="text-xs text-muted">Output distribution p_i = softmax(W_o[s_i ; c_i] + b_o), top 5 of {model.vocab.length} symbols. The most likely one is output.</div>
          <svg viewBox="0 0 360 112" className="h-auto w-full" style={{ maxWidth: 520 }} role="img" aria-label="Top five output probabilities">
            {cur.top.map((t, k) => (
              <g key={t.id}>
                <text x={50} y={16 + k * 21} textAnchor="end" fontSize={12} fontFamily="monospace" fill={k === 0 ? "var(--gold)" : "var(--text)"}>
                  {show(t.sym)}
                </text>
                <rect x={56} y={5 + k * 21} width={Math.max(0.5, t.p * 230)} height={14} fill={k === 0 ? "var(--gold)" : "var(--accent)"} />
                <text x={56 + Math.max(0.5, t.p * 230) + 4} y={16 + k * 21} fontSize={10} fill="var(--text)">
                  {t.p < 1e-4 ? t.p.toExponential(1) : t.p.toFixed(4)}
                </text>
              </g>
            ))}
          </svg>
        </div>
      )}

      {/* ---------- alignment matrix ---------- */}
      <div className="mt-4 text-xs text-muted">Alignment matrix α (rows = output steps, columns = input letters), filled row by row. Click a row to jump to that step. A perfect reversal is the diagonal from top-right to bottom-left.</div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${amW} ${amH}`} className="h-auto w-full" style={{ maxWidth: amW * 1.3 }} role="img" aria-label="Alignment matrix of attention weights, outputs by inputs">
          {chars.map((c, j) => (
            <text key={j} x={al + j * ac + ac / 2} y={12} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="var(--text-muted)">
              {c}
            </text>
          ))}
          {trace.steps.map((st, i) => {
            const filled = i < step;
            const y = at + i * ac;
            return (
              <g key={i} onClick={() => setStep(i + 1)} style={{ cursor: "pointer" }}>
                <text x={al - 6} y={y + 15} textAnchor="end" fontSize={11} fontFamily="monospace" fill={i === step - 1 ? "var(--gold)" : "var(--text-muted)"}>
                  {filled ? (st.token === model.cfg.eos ? "eos" : st.sym) : "·"}
                </text>
                {st.alpha.map((a, j) => (
                  <rect key={j} x={al + j * ac} y={y} width={ac - 1} height={ac - 1} fill={filled ? seqColor(a) : "var(--surface-2)"} stroke={i === step - 1 ? "var(--gold)" : "var(--border)"} strokeWidth={i === step - 1 ? 1.2 : 0.4}>
                    {filled && <title>{`output ${i + 1} → input ${j + 1} (“${chars[j]}”): α = ${a.toFixed(4)}`}</title>}
                  </rect>
                ))}
              </g>
            );
          })}
        </svg>
      </div>

      {/* ---------- measured bottleneck result ---------- */}
      <div className="mt-4 rounded-lg border border-border p-3">
        <div className="text-xs font-medium">Measured: share reversed perfectly, by input length (from our training runs)</div>
        <div className="mb-1 text-xs text-muted">
          Same encoder, decoder, seed and training time. The only difference: does the decoder get a fresh context c_i at each step (attention, solid), or one fixed vector c = [→h_T ; ←h_1] for the whole output (dashed)? {MEASURED.samplesPerLength} random strings per length; trained on lengths 1–{MEASURED.trainMaxLen}.
        </div>
        <AccuracyChart />
        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
          <span style={{ color: "var(--accent)" }}>━ with attention</span>
          <span style={{ color: "var(--clay)" }}>╌ fixed context vector</span>
        </div>
      </div>
    </SimShell>
  );
}
